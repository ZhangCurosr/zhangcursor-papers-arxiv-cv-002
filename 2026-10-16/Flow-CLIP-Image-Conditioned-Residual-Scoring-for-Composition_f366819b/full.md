# Flow-CLIP: Image-Conditioned Residual Scoring for Compositional Image–Text Matching

Minghong Wei<sup>1</sup>, Enguang Zhu<sup>1</sup>, Pu Cao<sup>1</sup>, Lu Yang<sup>1,∗</sup> and Qing Song<sup>1</sup>

<sup>1</sup>School of Intelligent Engineering and Automation, Beijing University of Posts and Telecommunications, Beijing, 100082, China

A R T I C L E I N F O

Keywords:   
Vision-language alignment   
Compositional matching   
Global compatibility calibration   
Image-conditioned evidence refinement   
Cross-image margin ranking

## A BS T R AC T

Vision–language pre-training has enabled substantial progress in image–text retrieval and zero-shot recognition by learning transferable cross-modal representations. However, sentence-level alignment remains insuficient for compositional matching, where captions can describe the same objects and attributes while difering in how these elements are bound. To address this limitation, we propose Flow-CLIP, which refines global image–text alignment through image-conditioned changes in token– patch evidence. Global Compatibility Calibration (GCC) learns a sentence-level similarity metric, while Image-Conditioned Evidence Refinement (ICER) incorporates visual patch context into token queries and uses the resulting change in local matching evidence to adjust the global score. We further introduce Cross-Image Margin Ranking (CIMR), which compares the same positive–negative caption pair across matched and mismatched images, encouraging a larger ranking margin when the image supports the correct description. Experiments on compositional matching and attribute–object binding benchmarks show that Flow-CLIP improves binding accuracy over pretrained CLIP and learned global alignment through local evidence refinement, while cross-image ranking supervision mitigates reliance on language priors. Code will be released at link.

![](images/73c8c0e9d1e82800674f921b52cd4828ffbe710f8cc800c2019f027e18542797.jpg)

![](images/9f56c0a1c83f740520b4f0a89c13d74e2c58644732a9e826dd67112e3e7c55a6.jpg)  
Fig. 1. Compositional image–text matching with closely related caption candidates. Bars show compatibility scores, yellow outlines identify the top-ranked caption, and the curve shows the gold-caption margin.

## 1. Introduction

Vision–language pretraining enables zero-shot recognition and image–text retrieval by learning transferable representations from paired images and descriptions [1–5]. Compositional matching requires these representations to distinguish descriptions that name the same objects and attributes but assign or relate them diferently. Benchmarks of attribute binding, relations and word order document persistent errors in making these distinctions [6–11].

The caption rankings in Fig. 1 illustrate how CLIP-based models [1, 7, 12] can prefer a hard-negative caption to the positive caption. The alternatives share many words but difer in object identities, attribute assignments or spatial relations.

Targeted caption negatives expose these distinctions during training [7, 13], while dense descriptions and image–text– graph comparisons add supervision for scene details [14, 15]. FSC-CLIP [12] combines local hard-negative training with calibrated regularization to improve compositional matching while maintaining general-task performance.

LABCLIP [16] shows that a learned linear transformation of text embeddings can recover attribute–object binding information in cross-modal matching. We use learned global alignment as a baseline to measure the additional contribution of image-conditioned local refinement to attribute–object binding.

Local matching provides the representations needed for such refinement. FILIP [17] and DCSM [18] use token–patch similarities, ComCLIP [19] and OC-CLIP [20] introduce structured entity scores, and FLAIR [21] constructs captionconditioned visual representations. These methods aggregate local similarities, compare structured entities or condition visual features on text. We use local interactions to measure how image context changes support for a caption, providing a correction to its calibrated global score.

An incorrect attribute assignment can receive local support when its individual objects and attributes are present in the image. Strong token–patch matches alone therefore leave open whether the proposed assignment is supported. We use the image’s patch context to refine the token queries that assess this assignment, then compare the refined and original queries on the same patches. The original query supplies a reference for measuring the correction’s efect, yielding a signed adjustment that can strengthen or weaken the global compatibility score. To model this image-conditioned change, we propose Flow-CLIP, which combines Global Compatibility Calibration (GCC) with Image-Conditioned Evidence Refinement (ICER), as illustrated in Fig. 2. GCC learns sentence-level alignment, and ICER supplies the local adjustment through context-dependent query correction. Both components arejointly trained for caption ranking with frozen encoders.

![](images/de9f26c5fd49e35535afc430d0f5fcc826eb2f836e4048c62b96868b8f00d50b.jpg)  
Fig. 2. Flow-CLIP combines Global Compatibility Calibration (GCC) with Image-Conditioned Evidence Refinement (ICER). The global and token maps produce normalized representations. Centered patch support forms visual context, whose projection supplies the input to the learned query correction. The normalized corrected query and its original query yield a signed diference in pooled evidence, which is averaged over valid tokens and added to global compatibility. and denote frozen and learned components.

Full-caption hard negatives supervise caption ranking on matched images, following NegCLIP [7]. The construction of these alternatives also matters: SugarCrepe [22] identifies plausibility and fluency biases that allow models to distinguish positive descriptions from negative alternatives without the image. Language priors can consequently favor plausible descriptions even when their attribute assignments are unsupported by the visual input. Supervision on matched pairs does not explicitly constrain how this preference should change across images. A caption preference that persists after replacing the image can satisfy the matched-pair objective without reflecting the visual evidence for the attribute assignment. We introduce Cross-Image Margin Ranking (CIMR) to compare the same positive–negative caption margin on the matched image and one donor shared by both captions. By encouraging a larger margin on the matched image, this objective rewards a change in preference tied to visual input while holding the captions and their linguistic cues fixed. GCC and ICER are optimized jointly with the hard-negative and CIMR objectives.

Independently trained global-only heads account for much of the caption-order improvement, while training with the local branch yields additional attribute–object binding gains. Paired predictions identify corrected binding errors and new spatial errors. Image-replacement tests measure changes in caption preference when the visual match is removed. Comparisons across training sources and encoders show how binding gains vary alongside changes in spatial and retrieval accuracy.

The contributions of this work are:

• We propose Flow-CLIP, combining Global Compatibility Calibration (GCC) and Image-Conditioned Evidence Refinement (ICER) with frozen encoders. The local branch refines token queries using visual patch context and adjusts global compatibility by the signed change in pooled patch evidence.

• We introduce Cross-Image Margin Ranking (CIMR), which compares the same positive–negative caption margin on matched and shared donor images. This objective encourages caption preference to depend on the matched image using full-caption supervision.

• Our experiments separate the binding gains of local refinement from the sentence-level gains of global calibration and identify the contribution of cross-image supervision to mitigating reliance on language priors.

## 2. Related Work

## 2.1. Compositional Evaluation

Compositional benchmarks evaluate image–text matching through controlled semantic changes. Winoground [6] pairs two images with captions that share their words but difer in composition, requiring scores to recover both image– caption assignments. ARO [7] separates attribution, relation and caption-order judgments. VL-CheckList [8] organizes object, attribute and relation edits, while VALSE [9] evaluates linguistic phenomena using positive descriptions and edited foils. What’s Up? [11] focuses on spatial relations. These distinctions matter for local scoring: recognizing the concepts present in an image can support object discrimination, whereas attribute reassignment and relation reversal require resolving how those concepts are associated.

CLEVR [23] and PUG [24] provide rendered scenes with known attribute–object assignments. Swapping those assignments produces caption alternatives that preserve the scene’s object and attribute inventory, isolating the assignment distinction relevant to token-level refinement. CREPE [25] tests systematicity and productivity with candidate sets of increasing compositional complexity, and BiVLC [26] evaluates both retrieval directions. These evaluations extend the comparison to competition among multiple descriptions and to the consistency of image–text matching across retrieval directions.

Caption construction determines which cues can support a correct ranking. SugarCrepe [22] uses fluent semantic edits to reduce linguistic artifacts, and SugarCrepe++ [27] distinguishes semantic sensitivity from lexical variation. NegBench [28] evaluates negation, whereas Half-Truths [29] adds plausible but incorrect content to an otherwise valid description. Controlling the caption alternatives limits cues introduced by their construction. Image-replacement controls address the complementary question of whether the resulting preference depends on the visual match while keeping those alternatives fixed.

## 2.2. Image–Text Matching

Global scoring compares pooled image and text embeddings, as used in CLIP [1] and ALIGN [2]. LiT [4] studies alignment with a locked image encoder. LABCLIP [16] evaluates binding within each modality using linear probes, increasing object counts and conjunctive search, then learns a text-embedding transformation for cross-modal matching. These experiments separate the information represented by each encoder from the information accessible through the cross-modal scoring function.

Work on concept-to-object mappings [30] further examines how embeddings represent binding functions. The distinction relevant to our comparison is the granularity of the learned alignment: LABCLIP transforms the pooled text embedding, while ICER updates individual token queries using image patches. GCC retains the sentence-level transformation, allowing the experiments to assess the additional efect of training with the local branch under matched conditions.

Fine-grained alignment preserves visual and textual units within the matching score. RegionCLIP [31] learns region–text representations, and FILIP [17] aggregates tokenwise maximum similarities for cross-modal late interaction. FLAIR [21] uses text-conditioned attention pooling over image tokens for detailed retrieval, while Beta-CLIP [32] develops text-conditioned alignment at multiple granularities. The conditioning difers across these methods: late interaction selects matching responses, whereas text-conditioned pooling changes the visual representation.

DCSM [18] learns a scoring function over dense token– patch cosine similarities, including functional rows for spatial relationship words. ICER scores the change in pooled support after query correction.

Structured matching specifies the semantic units used for comparison. ComCLIP [19] computes separate scores for subject, object and action subimages, then combines these scores at inference. OC-CLIP [20] associates scenegraph nodes with visual slots and scores relational compatibility, while ABE-CLIP [33] introduces attribute-binding operations during inference. VQAScore [34] scores image– text compatibility through visual question answering. Flow-CLIP uses contextual token and patch states as its local comparison units, without requiring explicit scene-graph or subimage decomposition. Full-caption supervision trains the contribution of these interactions to the final compatibility score.

## 2.3. Training Objectives

Compositional training exposes distinctions that ordinary image–caption pairs may leave implicit. NegCLIP [7] uses caption perturbations and hard image negatives, and TSVLC [13] manipulates language structure to teach attributes, relations and states from paired data. DAC [14] supplies dense, image-aligned descriptions. Coarse-to-fine image–text–graph learning [15] decomposes and augments parsed scene graphs to align descriptions of diferent structural complexities with an image. TripletCLIP [35] and SPARCL [36] add synthetic visual and textual comparisons. Flow-CLIP trains its global and local terms through the same image–caption score using full-caption hard negatives.

The objective determines which relationships those training examples constrain. CE-CLIP [37] introduces intramodal contrast and cross-modal hard-negative ranking. FSC-CLIP [12] combines local hard-negative loss with selective calibrated regularization to balance compositional supervision and general multimodal capabilities. Its design addresses the degradation of pretrained multimodal representations caused by hard-negative fine-tuning. READ-CLIP [38] couples token reconstruction with paraphrase alignment, while concept-centric learning [39] aligns caption concepts with visual content through auxiliary contrasts. These objectives constrain feature alignment or its preservation during adaptation.

CLIC [40] constructs training comparisons from combinations of multiple images and their captions. CIMR evaluates the same caption pair on both the matched image and a shared donor image. Caption identity remains fixed when the two margins are compared, so an additive text-only term cancels and the objective constrains the score change induced by replacing the image.

## 3. Method

For an image � and caption �, Flow-CLIP adds an ICER local term to GCC global compatibility. The local term measures the signed change in pooled patch support after image-conditioned token-query correction:

$$
s ( I , T ) = s _ { \mathrm { g l o b a l } } ( I , T ) + \alpha s _ { \mathrm { t o k e n } } ( I , T ) .\tag{1}
$$

Algorithm 1 Flow-CLIP scoring with GCC and ICER   
Input: Image �, caption $T ;$ image and text encoders; learned   
mappings $M , M _ { \mathrm { t o k } } , W ;$ pooling temperature $\tau _ { p } > 0$ and local   
scale $\alpha \geq 0 .$   
Output: Compatibility score $s ( I , T ) .$   
Feature extraction and GCC   
1: Extract global image embedding ${ \pmb v } _ { I }$ and patch embeddings �   
from $I .$   
2: Extract sentence embedding $t _ { T }$ and projected token states �   
from �.   
3: $\pmb { v } _ { I }  \mathrm { n o r m } ( \pmb { v } _ { I } ) ; t _ { T }  \mathrm { n o r m } ( t _ { T } ) .$   
4: $p _ { j } \gets \mathrm { n o r m } ( { p } _ { j } ) , j = 1 , \ldots , N .$   
5: $\begin{array} { r } { \dot { h _ { \ell } } \gets \mathrm { n o r m } ( \bar { h _ { \ell } } ) , \ell = 1 , \dots , L . } \end{array}$   
6: Obtain $\ v _ { T }$ by excluding special and padding tokens.   
7: � ← norm( $M t _ { T } ) .$   
8: $s _ { \mathrm { g l o b a l } }  v _ { I } ^ { \top } ;$ � using Eq. (2).   
9: $\mathbf { i } \check { \mathbf { f } } \mathbf { \partial } \mathcal { V } _ { T } = \emptyset$ then   
10: return $s _ { \mathrm { g l o b a l } } .$   
11: end if   
ICER: image-conditioned token correction   
12: for $\ell \in { \mathcal { V } } _ { T }$ do   
13: $\begin{array} { r } { \pmb q _ { \ell } \gets \mathrm { n o r m } ( M _ { \mathrm { t o k } } \pmb h _ { \ell } ) . } \end{array}$   
14: $\begin{array} { r } { a _ { \ell j }  a _ { \ell } ^ { \top } p _ { j } , j = 1 , \ldots , N . } \end{array}$   
15: $\begin{array} { r } { \bar { a } _ { \ell } \gets \frac { 1 } { N } \sum _ { j = 1 } ^ { N } a _ { \ell j } . } \end{array}$   
16: $e _ { \ell j } \gets \operatorname* { i n a x } ( a _ { \ell j } - \bar { a } _ { \ell } , 0 ) .$   
17: $\begin{array} { r } { \pmb { c } _ { \ell } \gets \frac { 1 } { N } \sum _ { j = 1 } ^ { N } e _ { \ell j } \pmb { p } _ { j } . } \end{array}$   
18: $\pmb { r } _ { \ell } \gets \pmb { c } _ { \ell } - ( \pmb { q } _ { \ell } ^ { \top } \pmb { c } _ { \ell } ) \pmb { q } _ { \ell } .$   
19: $\tilde { \pmb q } _ { \ell }  \pmb q _ { \ell } + W \pmb r _ { \ell } .$   
20: $\begin{array} { r } { \pmb q _ { \ell } ^ { \mathrm { r e s } }  \mathrm { n o r m } ( \tilde { \pmb q } _ { \ell } ) . } \end{array}$   
21: $\rho _ { \ell } \gets \mathcal { P } _ { \tau _ { n } } ( \pmb q _ { \ell } , P )$ using Eq. (6).   
22: $\rho _ { \ell } ^ { \mathrm { r e s } } \gets \mathscr { P } _ { \tau _ { v } } ( \pmb { q } _ { \ell } ^ { \mathrm { r e s } } , P ) .$   
23: $\dot { \Delta _ { \ell } } \gets \rho _ { \ell } ^ { \mathrm { r e } \bar { s } } - \bar { \rho } _ { \ell }$ using Eq. (7).   
24: end for   
Score aggregation   
25: $s _ { \mathrm { t o k e n } }  \frac { 1 } { \vert \mathcal { V } _ { T } \vert } \sum _ { \ell \in \mathcal { V } _ { T } } \Delta _ { \ell } .$   
26: return $s _ { \mathrm { g l o b a l } } + \alpha s _ { \mathrm { t o k e n } } .$

Here $s _ { \mathrm { g l o b a l } }$ denotes global compatibility, $s _ { \mathrm { t o k e n } }$ denotes the local evidence change, and � scales the local contribution. The main configuration freezes both encoders; encoderadapted configurations use the same score. Global-� denotes the global-only GCC scorer.

## 3.1. Global Compatibility Calibration

For an image–caption pair (�, �), the image encoder produces a global embedding $\pmb { v } _ { I } \in \mathbb { R } ^ { d }$ and � patch embeddings $P = [ \pmb { p } _ { 1 } , \dots , \pmb { p } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d }$ . The text encoder produces a sentence embedding $t _ { T } \in \mathbb { R } ^ { d }$ and projected states for a sequence of � tokens, $\mathbf { \bar {  { H } } } = [ \pmb { h } _ { 1 } , \dots , \mathbf { \bar { h } } _ { L } ] ^ { \top } \in \mathbb { R } ^ { L \times d }$ Here � is the shared projection dimension. We $\ell _ { 2 } .$ -normalize ${ \pmb v } _ { I } , { \pmb t } _ { T }$ and every row of � and � before scoring, writing norm( ${ \pmb x } ) = { \pmb x } / \| { \pmb x } \| _ { 2 }$

As in $\mathrm { L A B C L I P \ [ 1 6 ] } .$ , we apply a learned linear transformation $M \in \mathbb { R } ^ { d \times d }$ to the sentence embedding before computing global cosine similarity:

$$
z = \mathrm { n o r m } ( M t _ { T } ) , \qquad s _ { \mathrm { g l o b a l } } ( I , T ) = v _ { I } ^ { \top } z .\tag{2}
$$

Global-� uses this score alone. Flow-CLIP jointly trains �, the token map $M _ { \mathrm { t o k } }$ and the residual map $W$ , defined

below. We initialize � and $M _ { \mathrm { t o k } }$ as identity matrices and $W$ as zero, so the initial score equals the pretrained global cosine similarity.

## 3.2. Image-Conditioned Evidence Refinement

ICER constructs visual context from each token query’s responses to the patches of the current image. Let $\ v _ { r } \subseteq$ $\{ 1 , \ldots , L \}$ contain the non-special, non-padding token positions, so $| \mathcal { V } _ { T } |$ is the number of valid tokens. A learned matrix $M _ { \mathrm { t o k } } \in \mathbb { R } ^ { d \times d }$ defines the unit query

$$
\begin{array} { r } { \pmb q _ { \ell } = \operatorname { n o r m } ( M _ { \mathrm { t o k } } \pmb h _ { \ell } ) . } \end{array}\tag{3}
$$

The valid-token set includes function words, punctuation and subwords. For token $\ell$ and patch $j ,$ let $a _ { \ell j } = { \pmb q } _ { \ell } ^ { \top } { \pmb p } _ { j }$ denote their cosine similarity. We subtract the mean response across patches and retain its positive part, using above-mean responses to construct visual context:

$$
\begin{array} { l } { { \displaystyle e _ { \ell j } = \left[ a _ { \ell j } - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } a _ { \ell k } \right] _ { + } ^ { } , } } \\ { { \displaystyle c _ { \ell } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } e _ { \ell j } p _ { j } . } } \end{array}\tag{4}
$$

Here $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ , and $e _ { \ell j }$ weights patch � in the context vector $c _ { \ell } \in \mathbb { R } ^ { d }$ . Uniform patch responses yield zero context, and centering makes the weights invariant to a common ofset in patch similarity. The factor $1 / N$ averages the weighted patch features without normalizing the weights to sum to one. Consequently, small deviations above the mean produce a small context vector, preserving the strength of the token’s spatially varying response. For a fixed caption, the original token queries remain unchanged across images, while the patch weights and features determine the image-dependent input to the correction.

We project the visual context onto the tangent space at the unit token query before applying the learned residual map:

$$
\begin{array} { r } { \pmb { r } _ { \ell } = \pmb { c } _ { \ell } - ( \pmb { q } _ { \ell } ^ { \top } \pmb { c } _ { \ell } ) \pmb { q } _ { \ell } , } \\ { \tilde { \pmb { q } } _ { \ell } = \pmb { q } _ { \ell } + W \pmb { r } _ { \ell } , \qquad } \\ { \pmb { q } _ { \ell } ^ { \mathrm { r e s } } = \mathrm { n o r m } ( \tilde { \pmb { q } } _ { \ell } ) . } \end{array}\tag{5}
$$

The projection removes the component of $c _ { \ell }$ parallel to the unit query, giving $\boldsymbol { q } _ { \ell } ^ { \intercal } \boldsymbol { r } _ { \ell } = 0$ . The learned map $W \in \mathbb { R } ^ { d \times d }$ transforms this projected context into a residual update, yielding the intermediate query $\tilde { \pmb q } _ { \ell }$ . Its normalization produces $\pmb q _ { \ell } ^ { \mathrm { r e s } }$ on the unit sphere, placing the original and corrected patch responses on the same cosine scale.

The local score measures the change in pooled patch evidence caused by query correction. For a unit query �, define

$$
\mathcal { P } _ { \tau _ { p } } ( \boldsymbol { q } , P ) = \tau _ { p } \log \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \exp ( \boldsymbol { q } ^ { \top } \boldsymbol { p } _ { j } / \tau _ { p } ) \right] .\tag{6}
$$

The temperature $\tau _ { p } ~ > ~ 0$ controls the smooth maximum over patch responses. We write $\rho _ { \ell } = \mathcal { P } _ { \tau _ { p } } ( \pmb { q } _ { \ell } , P )$ and $\rho _ { \ell } ^ { \mathrm { r e s } } =$

$\mathscr { P } _ { \tau _ { v } } ( q _ { \ell } ^ { \mathrm { r e s } } , P )$ for pooled evidence before and after correction. Both queries are evaluated against the same patch set:

$$
\begin{array} { r l } & { \Delta _ { \ell } ( I , T ) = \mathcal { P } _ { \tau _ { p } } ( q _ { \ell } ^ { \mathrm { r e s } } , P ) - \mathcal { P } _ { \tau _ { p } } ( q _ { \ell } , P ) , } \\ & { s _ { \mathrm { t o k e n } } ( I , T ) = \displaystyle \frac { 1 } { | \mathcal { V } _ { T } | } \sum _ { \ell \in \mathcal { V } _ { T } } \Delta _ { \ell } ( I , T ) . } \end{array}\tag{7}
$$

The sign of $\Delta _ { \ell }$ indicates whether correction strengthens or weakens pooled support relative to the original query. Subtraction retains this distinction even when both queries have high absolute patch similarity. Token increments within a caption can have opposite signs. Averaging them over $\mathcal { V } _ { T }$ yields the local term added to global compatibility in Eq. (1). An empty valid-token set contributes zero; $W = \mathbf { 0 } \mathrm { o r } \alpha = 0$ recovers the global branch.

## 3.3. Training Objective

Training jointly optimizes �, $M _ { \mathrm { t o k } }$ and � through the image–caption score. A mini-batch $\boldsymbol { B } = \{ ( I _ { i } , T _ { i } ^ { + } , T _ { i } ^ { - } , g _ { i } ) \} _ { i = 1 } ^ { B }$ contains � records, each with a matched image and caption, a full-caption hard negative, and image identity $g _ { i }$ . The indicator $y _ { i j } = \mathbf { 1 } [ g _ { i } = g _ { j } ]$ marks positive captions belonging to the same image. With contrastive temperature $\tau _ { c } ~ > ~ 0$ define

$$
u _ { i j } ^ { \pm } = s ( I _ { i } , T _ { j } ^ { \pm } ) / \tau _ { c } .\tag{8}
$$

For image $I _ { i } ,$ the image-to-text comparison uses logits $u _ { i j } ^ { + }$ and $u _ { i j } ^ { - }$ over all captions. For caption $T _ { i } ^ { + }$ , the text-to-image comparison uses $u _ { j i } ^ { + }$ over all in-batch images. This asymmetry makes hard negatives compete with positive descriptions of an image, while retaining positive captions as the queries for image retrieval. The bidirectional multi-positive losses are

$$
\begin{array} { r l } & { \mathcal { L } _ { I 2 T } = - \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \sum _ { j = 1 } ^ { B } y _ { i j } \exp ( u _ { i j } ^ { + } ) } { \sum _ { j = 1 } ^ { B } \left[ \exp ( u _ { i j } ^ { + } ) + \exp ( u _ { i j } ^ { - } ) \right] } \mathrm { , } } \\ & { \mathcal { L } _ { T 2 I } = - \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \sum _ { j = 1 } ^ { B } y _ { i j } \exp ( u _ { j i } ^ { + } ) } { \sum _ { j = 1 } ^ { B } \exp ( u _ { j i } ^ { + } ) } . } \end{array}\tag{9}
$$

We denote the bidirectional contrastive objective with fullcaption hard negatives by $\begin{array} { r } { \mathcal { L } _ { \mathrm { H N B } } = \frac { 1 } { \gamma } ( \mathcal { L } _ { I 2 T } + \mathcal { L } _ { T 2 I } ) } \end{array}$ , abbreviated as HNB. Summing over shared image identities in the numerators accommodates multiple positive captions for an image.

CIMR compares the positive–negative caption margin across visual inputs. For each eligible record �, donor image $J _ { i }$ has a diferent source identity and is used for both $T _ { i } ^ { + }$ and $T _ { i } ^ { - }$ . Known positive matches are excluded from donor comparisons. Let ${ \mathcal { E } } \subseteq \{ 1 , \ldots , B \}$ contain records with an eligible hard-negative caption and donor, and define

$$
m _ { i } ( I ) = s ( I , T _ { i } ^ { + } ) - s ( I , T _ { i } ^ { - } ) .\tag{10}
$$

The Cross-Image Margin Ranking loss is

$$
\mathcal { L } _ { \mathrm { C I M R } } = \frac { 1 } { | \mathcal { E } | } \sum _ { i \in \mathcal { E } } \mathrm { s o f t p l u s } \left( - \frac { m _ { i } ( I _ { i } ) - m _ { i } ( J _ { i } ) } { \tau _ { c } } \right) ,\tag{11}
$$

Here sof ${ \mathrm { t p l u s } } ( x ) = \log ( 1 + \exp x )$ . Increasing $m _ { i } ( I _ { i } ) { - } m _ { i } ( J _ { i } )$ lowers the loss, encouraging a larger caption margin on the matched image than on the donor. An additive text-only scoring term cancels because the same caption pair is used for both images. An additive image-only scoring term likewise cancels within each caption margin. CIMR therefore acts on score changes that depend jointly on image and caption content. For an empty eligible set, $\mathcal { L } _ { \mathrm { C I M R } } = 0$

The complete training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { H N B } } + \lambda _ { \mathrm { C I M R } } \mathcal { L } _ { \mathrm { C I M R } }\tag{12}
$$

We set $\lambda _ { \mathrm { C I M R } } = 0 . 1$ . Algorithm 1 summarizes the scoring procedure.

## 4. Experiments

## 4.1. Experimental Setup

Training data. We train on COCO [41] train2014 with the Karpathy restval split [42], using 113,287 training images, 566,435 image–caption records and 25,000 validation records. Each training record pairs a human-written COCO caption [43] with a full-caption hard negative. For the training-source comparison, we use Conceptual Captions (CC3M) [44] with the CC3M-100K protocol of FSC-CLIP [12]: 101,830 source records partitioned into 97,734 training and 4,096 validation records. CoCa [45] supplies positive captions, and the negative-generation schedule uses phrase swaps, WordNet replacements and bigram shufling. The released training and validation partitions share nine images, represented by 138 training and 16 validation records.

Evaluation datasets and metrics. We evaluate attribution, relations and caption order separately. ARO [7] contains four subsets: VG-Attribution (4,517 images and 28,748 pairs), VG-Relation (5,316 images and 23,937 pairs), Flickr30k Order (1,000 images and 5,000 five-caption sets), and COCO Order (5,000 images and 25,010 five-caption sets). SugarCrepe [22] contains 7,511 fluent positive–negative pairs with Add, Replace and Swap edits. CREPE-Productivity [25] evaluates Atomic and Swap candidate sets at complexity levels 4–12.

Attribute–object binding evaluation uses ABC-6K [46] (3,213 comparison pairs), PUG:SPAR [24] (19,840 attributeassignment pairs), and a two-object evaluation set derived from CLEVR [23]. We abbreviate ABC-6K and PUG:SPAR as ABC and PUG in the result tables. The CLEVR combination-based partition contains 470 records, including 449 valid pairs with non-equivalent captions; the larger CLEVR evaluation pool contains 4,449 unambiguous pairs. The ablation table reports the 449-pair split, and results on the 4,449-pair pool are identified separately in the text. COCO-Spatial and GQA-Spatial contain 2,687 and 1,451 records, respectively, and follow the spatial-relation protocol of What’s Up? [11].

Retrieval evaluation uses Flickr30k [47] and COCO [41] with the Karpathy 1K and 5K splits [42], respectively: 1,000/5,000 images and 5,000/25,000 captions. Zero-shot classification uses CIFAR-10/CIFAR-100 [48] (10,000 test images each) and ImageNetV2 [49] (10,000 images with 1,000 class prompts). We retain each benchmark’s task-level aggregation and specify the members of every cross-task mean in its caption. Pairwise accuracy tests the gold caption against one foil; candidate-set accuracy requires it to rank first among all alternatives. Retrieval extends this competition to the complete evaluation gallery, testing whether the learned score separates the correct match from many alternatives beyond the constructed foils. Image and comparison counts are reported separately because an image can appear in several evaluation records.

Implementation details. The main configuration freezes CLIP ViT-B/32 [1] and trains �, $M _ { \mathrm { t o k } }$ and � , totaling 786,432 parameters. For both head-only and encoder-adapted training, we use AdamW [50] with learning rate $8 \times 1 0 ^ { - 5 }$ weight decay $1 0 ^ { - 4 } , ( \beta _ { 1 } , \beta _ { 2 } ) \ = \ ( 0 . 9 , 0 . 9 9 9 )$ and gradient clipping at a maximum norm of 1. Each batch contains 32 image–caption records and 32 full-caption negatives. COCO training uses five full epochs with a hard-negative pool of 501,921 eligible records. CC3M training uses three epochs, with 96,463, 97,394 and 97,734 valid records for phrase swaps, WordNet replacements and bigram shufling, respectively. We set $\tau _ { c } = 0 . 0 1$ , � = 1 and $\lambda _ { \mathrm { C I M R } } = 0 . 1$ , with $\tau _ { p } = 0 . 0 5$ for COCO and 0.06 for CC3M. Experiments use NVIDIA RTX 3090 and A800 GPUs. In the encoder-adapted variants, both the image and text encoders are trainable together with �, $M _ { \mathrm { t o k } }$ and � under the HNB and CIMR objectives. Their COCO and CC3M results are evaluated after five and three training epochs, respectively. The scoring equations and objectives remain the same in both variants, allowing the comparison to assess the efect of updating encoder representations alongside the scoring maps.

Feature extraction. Scoring uses FP32 with autocast and TF32 disabled; normalized global and patch features are first rounded to FP16. CLIP [1] inputs follow the OpenAI image preprocessing pipeline. Patch states pass through vision postlayer normalization and visual projection. Text token states use the text projection and the valid-token mask defined in the method. SigLIP [5] retains its learned image-pooling operation and 196 patch embeddings, with a maximum text sequence length of 64 tokens in our evaluation. BLIP [3] uses 256-dimensional image–text contrastive (ITC) projections and 576 patch embeddings, with a maximum text sequence length of 35 tokens.

## 4.2. Main Results

The COCO-trained frozen Flow-CLIP scorer in Tab. 1 reaches 78.15% on ABC, exceeding NegCLIP [7], LAB-CLIP [16] and FSC-CLIP [12], with a 3.39-percentage-point gain over FSC-CLIP. Because the ABC alternatives exchange attribute assignments while preserving the scene concepts, this gain reflects better discrimination between competing bindings. Its ARO average of 85.94% combines attribution, relation and caption-order accuracy; the individual tasks show how these improvements are distributed.

![](images/f8457d7ef6835ddad3a60ebac2acc37cda2dc95e1c96be3345a20e91d661cc42.jpg)  
Fig. 3. CREPE Recall@1 for HNB-trained Flow-CLIP and LABCLIP [16] across complexity levels 4–12. Solid and dashed curves denote atomic and swap foils. Endpoint labels show gains over LABCLIP in percentage points at complexity 12.

The comparison with FSC-CLIP separates attributeassignment gains from caption-order performance. Frozen Flow-CLIP scores higher on VG-Attribution, VG-Relation and Swap-Attribute, but lower on both caption-order tasks and Swap-Object. VG-Attribution, Swap-Attribute and ABC test which object has a named attribute, making their gains directly relevant to binding. The matched ablations in Sec. 4.3 test the contribution of the local scoring architecture under a common training protocol.

With CC3M training, frozen Flow-CLIP exceeds CC3Mtrained LABCLIP [16] on all four ARO tasks and on Swap-Attribute, ABC and GQA-Spatial, but scores lower on Swap-Object and COCO-Spatial. Its 84.01% ARO average falls within the range of the encoder-tuned methods in the table, while both binding swap scores remain below FSC-CLIP [12]. These results extend the comparison to generated captions under the CC3M recipe and show that the binding advantage varies by task and comparator.

DCSM [18] has the highest spatial scores in this comparison: 72.47% on COCO-Spatial and 68.45% on GQA-Spatial. It combines box-derived spatial descriptions, opposite arrangements generated with CutMix and functional rows for spatial words in the scoring map. Flow-CLIP learns from full-caption negatives and computes local support from image patches. The spatial comparison therefore includes diferences in both supervision and scoring architecture.

Encoder adaptation yields ARO averages of 86.43% with COCO and 86.96% with CC3M, exceeding the frozen results in Tab. 1 by 0.49 and 2.95 percentage points.

Composition complexity. Fig. 3 compares Flow-CLIP and LABCLIP [16] on CREPE [25] atomic and swap foils across complexity levels 4–12. For the HNB-trained Flow-CLIP configurations shown, the gains over LABCLIP at complexity 12 are 0.2 and 7.2 Recall@1 percentage points on COCO atomic and swap foils, and 8.5 and 5.7 points on CC3M-100K. The COCO atomic curves converge at high complexity, while the swap curves retain a larger separation. On CC3M-100K, Flow-CLIP retains gains for both foil types at the highest tested complexity.

Compositional matching and fine-grained binding accuracy (%). ARO Avg. is the mean across four tasks; Swap-A/O denote SugarCrepe attribute/object swaps. ✓/× indicate frozen/jointly tuned encoders. Bold/underline indicate the best/second-best results within each training-source block.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Training data</td><td rowspan="2">Frozen?</td><td colspan="4">ARO</td><td rowspan="2"></td><td colspan="2">Attribute/Object Binding</td><td colspan="2">Spatial Relations</td></tr><tr><td>VG-A VG-R</td><td>Flickr Ord.</td><td>COCO Ord.</td><td>Avg.</td><td>Swap-A Swap-O</td><td>ABC</td><td></td><td>COCO-Sp. GQA-Sp.</td></tr><tr><td>CLIP [1]</td><td>ViT-B/32</td><td>1</td><td></td><td>63.21</td><td>63.36 58.88</td><td>47.60</td><td>58.26</td><td>63.96</td><td>61.63</td><td>64.11</td><td>46.88</td><td>46.69</td></tr><tr><td colspan="2">COCO-trained models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NegCLIP [7]</td><td>ViT-B/32</td><td>COCO14-HN</td><td>X</td><td>70.50 81.12</td><td>90.50</td><td>86.97</td><td>82.27</td><td>75.23</td><td>75.51</td><td>73.58</td><td>46.54</td><td>45.91</td></tr><tr><td>LABCLIP [16]</td><td>ViT-B/32</td><td>COCO14</td><td>√</td><td>68.32 81.45</td><td>82.90</td><td>79.95</td><td>78.15</td><td>75.08</td><td>72.24</td><td>69.97</td><td>48.24</td><td>45.18</td></tr><tr><td>FSC-CLIP [12]</td><td>ViT-B/32</td><td>COCO-100K</td><td>X</td><td>70.67</td><td>80.53 95.50</td><td>94.08</td><td>85.19</td><td>78.23</td><td>76.73</td><td>74.76</td><td>47.91</td><td>46.17</td></tr><tr><td>DCSM [18]</td><td>ViT-B/16</td><td>COCO17-curated</td><td>√</td><td>56.59</td><td>61.80 30.40</td><td>32.63</td><td>45.36</td><td>59.76</td><td>53.06</td><td>61.53</td><td>72.47</td><td>68.45</td></tr><tr><td>CE-CLIP [37]</td><td>ViT-B/32</td><td>COCO2014+2M</td><td>X</td><td>76.30</td><td>83.86 79.18</td><td>80.91</td><td>80.06</td><td>77.03</td><td>73.06</td><td>73.36</td><td>47.50</td><td>47.38</td></tr><tr><td>Flow-CLIP</td><td>ViT-B/32</td><td>COCO</td><td>√</td><td>74.10</td><td>86.76 92.12</td><td>90.78</td><td>85.94</td><td>79.28</td><td>75.92</td><td>78.15</td><td>46.12</td><td>45.01</td></tr><tr><td>Flow-CLIP</td><td>ViT-B/32</td><td>COCO</td><td>X</td><td>79.02</td><td>87.36 90.22</td><td>89.12</td><td>86.43</td><td>82.88</td><td>75.92</td><td>81.92</td><td>48.04</td><td>48.41</td></tr><tr><td colspan="2">CC3M-trained models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TSVLC [13]</td><td>ViT-B/32</td><td>CC3M</td><td>X</td><td>72.89 81.76</td><td>92.80</td><td>83.34</td><td></td><td>82.70 63.06</td><td>62.45</td><td>71.15</td><td>46.21</td><td>45.91</td></tr><tr><td>DAC-LLM [14]</td><td>ViT-B/32</td><td>CC3M</td><td>X</td><td>72.80</td><td>83.60 92.76</td><td>89.73</td><td>84.72</td><td>73.27</td><td>71.43</td><td>70.59</td><td>45.38</td><td>49.53</td></tr><tr><td>DAC-SAM [14]</td><td>ViT-B/32</td><td>CC3M</td><td>X</td><td>69.15</td><td>83.65 94.72</td><td>88.50</td><td></td><td>84.00 71.77</td><td>70.20</td><td>70.21</td><td>47.65</td><td>48.66</td></tr><tr><td>LABCLIP [16]]</td><td>ViT-B/32</td><td>CC3M</td><td>√</td><td>68.01</td><td>78.45 83.16</td><td>72.73</td><td></td><td>75.59 67.42</td><td>68.16</td><td>67.88</td><td>51.70</td><td>43.55</td></tr><tr><td>FSC-CLIP [12]</td><td>ViT-B/32</td><td>CC3M-100K</td><td>X</td><td>69.49</td><td>80.47 90.98</td><td>90.40</td><td></td><td>82.83 75.53</td><td>73.88</td><td>73.48</td><td>46.87</td><td>46.86</td></tr><tr><td>Flow-CLIP</td><td>ViT-B/32</td><td>CC3M-100K</td><td>√</td><td>72.45</td><td>78.96 93.02 96.96</td><td></td><td>91.63</td><td>84.01 70.57</td><td>67.76</td><td>72.49</td><td>47.35</td><td>47.76</td></tr><tr><td>Flow-CLIP</td><td>ViT-B/32</td><td>CC3M-100K</td><td>X</td><td>70.66 83.99</td><td></td><td></td><td>96.25</td><td>86.96 71.17</td><td>70.20</td><td>74.95</td><td>52.93</td><td>49.87</td></tr></table>

Component and objective ablation on COCO (%). Checks indicate learned mappings and the CIMR loss. All trained rows use HNB. Lower lines show changes from CLIP [1] in percentage points; Avg. is the mean across the 11 tasks. The gray configuration is complete Flow-CLIP.
<table><tr><td rowspan="2" colspan="3">M  $M _ { \mathrm { t o k } }$ </td><td rowspan="2">W CIMR</td><td colspan="4">ARO</td><td colspan="7">Fine-grained binding</td><td rowspan="2">Avg.</td></tr><tr><td></td><td>VG-A</td><td>VG-R</td><td>Flickr Ord.</td><td>COCO Ord.</td><td>Swap-A</td><td>Swap-O</td><td>ABC COCO-Sp.</td><td>GQA-Sp.</td><td>PUG</td><td>CLEVR</td></tr><tr><td></td><td></td><td></td><td></td><td>63.21</td><td>63.36</td><td>58.88</td><td>47.60</td><td>63.96</td><td>61.63</td><td>64.11</td><td>46.88</td><td>46.69</td><td>51.04</td><td>48.11</td><td>55.95</td></tr><tr><td>√</td><td></td><td></td><td></td><td>71.04 (+7.83)</td><td>84.23 (+20.88)</td><td>91.24 (+32.36)</td><td>90.20 (+42.61)</td><td>76.43 (+12.46)</td><td>72.24 (+10.61)</td><td>71.12 (+7.00)</td><td>49.91 (+3.03)</td><td>43.03 (-3.65)</td><td>50.63 (-0.41)</td><td>44.99 (-3.12)</td><td>67.73 (+11.78)</td></tr><tr><td>√</td><td></td><td></td><td>J</td><td>71.42 (+8.21)</td><td>84.78 (+21.43)</td><td>91.28 (+32.40)</td><td>90.25 (+42.65)</td><td>75.98 (+12.01)</td><td>72.65 (+11.02)</td><td>71.46 (+7.35)</td><td>49.25 (+2.37)</td><td>43.38 (-3.31)</td><td>50.85 (-0.19)</td><td>46.33 (-1.78)</td><td>67.97 (+12.01)</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>76.22 (+13.00)</td><td>85.50 (+22.14)</td><td>91.40 (+32.52)</td><td>89.90 (+42.30)</td><td>78.68 (+14.71)</td><td>75.51 (+13.88)</td><td>78.56 (+14.44)</td><td>48.98 (+2.10)</td><td>45.18 (-1.50)</td><td>54.20 (+3.16)</td><td>53.67 (+5.57)</td><td>70.71 (+14.76)</td></tr><tr><td>√</td><td></td><td>一</td><td>√</td><td>76.03 (+12.82)</td><td>86.16 (+22.80)</td><td>91.42 (+32.54)</td><td>90.04 (+42.44)</td><td>77.93 (+13.96)</td><td>77.55 (+15.92)</td><td>78.34 (+14.22)</td><td>48.53 (+1.65)</td><td>44.97 (-1.72)</td><td>55.05 (+4.01)</td><td>56.12 (+8.02)</td><td>71.10 (+15.15)</td></tr></table>

## 4.3. Ablation Studies

We independently train Global-� and the global/local scorers with the same data and optimization settings, evaluating all heads at COCO epoch 5. Means and standard deviations across three runs quantify training variability.

Component efects. Global calibration accounts for much of the improvement over pretrained CLIP [1] in Tab. 2. With matched COCO data and optimization at epoch 5, Global-� raises the 11-task mean from 55.95% to 67.73%. Its largest gains occur on caption order, with accuracy above 90% on both Flickr30k Order and COCO Order. Flow-CLIP reaches 71.10%, adding 3.37 percentage points to the mean. The binding tasks distinguish this additional benefit from the caption-order gains already obtained through global calibration.

The seven-task binding mean covers Swap-A, Swap-O, ABC, COCO-Spatial, GQA-Spatial, PUG and the CLEVR 449-pair split. It rises from 54.63% for CLIP to 58.34% for Global-� and 62.64% for Flow-CLIP: global calibration adds 3.70 percentage points, and the joint scorer adds another 4.30 points. This group tests attribute assignments, object identity and spatial relations. The 11-task mean additionally includes all four ARO tasks, where global calibration already achieves high caption-order accuracy.

For a fixed global mapping, the local contribution in Eq. (7) changes the caption-pair margin by � $[ s _ { \mathrm { t o k e n } } ( I , T ^ { + } ) -$ $s _ { \mathrm { t o k e n } } ( I , T ^ { - } ) ]$ . Ranking therefore depends on the diference between the two evidence changes. Equal increments leave the margin unchanged; unequal increments can increase or decrease it. In the trained ablations, each scorer learns its own global mapping, jointly with the local maps where present. The HNB and CIMR comparisons measure the efect of training the global/local architecture under each objective.

Controlled assignment tasks preserve the objects and attributes mentioned in the captions while changing which attribute belongs to each object. Under HNB, training the local branch with the global map raises accuracy on the CLEVR [23] 449-pair split from 44.99% to 53.67% and on PUG:SPAR [24] from 50.63% to 54.20%. With CIMR in both heads, Flow-CLIP exceeds Global-�+CIMR by 9.80 and 4.20 percentage points, respectively. Training with the local branch improves both assignment tasks under either objective. On the CLEVR 4,449-pair evaluation pool, HNBtrained Global-� achieves 51.29%, while Flow-CLIP trained with HNB and CIMR achieves 56.06%.

Flow-CLIP exceeds Global-� by 5.31 percentage points on Swap-Object and 7.22 points on ABC, while COCO-Spatial [11] decreases by 1.38 points. The local architecture’s gains on binding thus difer from its efect on spatial-relation discrimination. Paired predictions distinguish corrected selections from new errors.

Encoder adaptation. The encoder-update ablation compares the fully trainable Flow-CLIP models in Tab. 1 with separate frozen Flow-CLIP runs on the corresponding training source. The frozen controls achieve 85.91% on COCO and 83.85% on CC3M, giving adaptation gains of 0.52 and 3.11 points, respectively. Within the matched COCO comparison, Flickr30k Order, COCO Order, Swap-Object and COCO-Spatial decrease by 1.20, 0.91, 1.63 and 0.49 points. The ARO increase comes from attribution and relation gains despite lower caption-order accuracy.

Training variability. Across three independently trained COCO runs, mean ARO accuracy and its sample standard deviation are $8 4 . 3 1 \pm 0 . 1 9$ for Global-�, $8 4 . 4 7 \pm 0 . 2 1$ for Global-�+CIMR, $8 5 . 7 7 \pm 0 . 2 5$ for Flow-CLIP (HNB) and 85.90 ± 0.04 for Flow-CLIP with CIMR. Flow-CLIP exceeds Global-� in every run.

Confidence intervals. We estimate 95% percentile confidence intervals for three-run mean accuracy with 10,000 paired image-cluster bootstrap draws at fixed final checkpoints. Each draw uses the same image resampling weights across methods, tasks and training runs. The intervals for the Flow-CLIP minus Global-� accuracy diference are [1.26, 2.31] percentage points on SugarCrepe, [5.95, 8.13] on ABC and [−3.20, −0.04] on COCO-Spatial. They support a gain on SugarCrepe and ABC and a decline on COCO-Spatial under this evaluation protocol.

Standard deviations across runs measure training variability, while the paired bootstrap estimates image-sampling uncertainty in the three-run mean at fixed checkpoints. Each bootstrap draw must retain all benchmark aggregation groups. Some VG-Relation groups contain only one unique image, so we omit intervals for VG-Relation and the ARO mean under this protocol. We also omit GQA and imagereplacement intervals because image-cluster membership, including shared donor groups, cannot be fully determined from the evaluation records.

Decision changes. Paired predictions show which Global-� errors Flow-CLIP corrects and which correct predictions it loses. These transitions expose changes hidden by average accuracy when repairs and regressions ofset each other. On 7,511 SugarCrepe [22] records, Flow-CLIP repairs 366 errors and introduces 231, yielding 135 additional correct decisions. On the 3,213 ABC [46] pairs, it repairs 352 and introduces 120 errors. In both comparisons, repairs exceed regressions on identical images and candidate captions, with a larger net gain on ABC’s attribute-assignment task.

ARO accuracy across CIMR weight $\lambda _ { \mathrm { C I M R } } ,$ pooling temperature $\tau _ { p }$ and local scale � (%). Avg. is the mean across four tasks; the gray row is the reference configuration.
<table><tr><td> $\lambda _ { \mathrm { C I M R } }$ </td><td> $\tau _ { p }$ </td><td>α VG-A</td><td>VG-R Flickr Ord.</td><td></td><td>COCO Ord. Avg.</td></tr><tr><td>0.05</td><td>0.03</td><td>1.00</td><td>76.00 86.40</td><td>91.54</td><td>90.14</td><td>86.02</td></tr><tr><td>0.05</td><td>0.05</td><td>1.00</td><td>76.06</td><td>86.02 91.38</td><td>90.04</td><td>85.88</td></tr><tr><td>0.05</td><td>0.03</td><td>0.75</td><td>75.75</td><td>86.37 91.48</td><td>90.09</td><td>85.92</td></tr><tr><td>0.05</td><td>0.05</td><td>0.75</td><td>76.03</td><td>86.16 91.42</td><td>90.04</td><td>85.91</td></tr><tr><td>0.10</td><td>0.03</td><td>1.00</td><td>75.65</td><td>86.75 91.54</td><td>90.08</td><td>86.00</td></tr><tr><td>0.10</td><td>0.05</td><td>51.00</td><td>76.05</td><td>86.45 91.78</td><td>90.22</td><td>86.12</td></tr><tr><td>0.10</td><td>0.03</td><td>0.75</td><td>75.61</td><td>86.54 91.90</td><td>90.04</td><td>86.02</td></tr><tr><td>0.10</td><td>0.05</td><td>0.75</td><td>75.84 86.67</td><td>91.74</td><td>90.13</td><td>86.09</td></tr></table>

COCO-Spatial shows the reverse balance: 155 repairs and 213 regressions. Its oficial score decreases by 1.38 percentage points with equal weighting of the one- and twoobject subsets. This weighting difers from pooling correct decisions across all records, so the score diference and net repair count use diferent aggregations. Here, training the joint scorer introduces more spatial errors than it corrects.

Among records correctly ranked by Global-�, Flow-CLIP produces smaller gold–foil margins on 68.06% of COCO-Spatial and 67.66% of GQA-Spatial records across the three runs, using the benchmarks’ subset weights. A smaller margin can remain positive, leaving the pairwise ranking correct.

Hyperparameter sensitivity. Mean ARO accuracy ranges from 85.88% to 86.12% across the eight configurations in Tab. 3. We vary $\lambda _ { \mathrm { C I M R } } \in \{ 0 . 0 5 , 0 . 1 0 \} , \tau _ { p } \in \{ 0 . 0 3 , 0 . 0 5 \}$ and $\alpha \in \{ 0 . 7 5 , 1 . 0 0 \}$ using frozen CLIP [1] ViT-B/32 and five COCO epochs. The default is (0.10, 0.05, 1.00).

The efect of pooling temperature depends on the CIMR weight. Increasing $\tau _ { p }$ from 0.03 to 0.05 lowers the ARO mean at $\lambda _ { \mathrm { C I M R } } = 0 . 0 5$ but raises it at $\lambda _ { \mathrm { C I M R } } = 0 . 1 0$ for both local scales. Temperature controls emphasis on the highest patch responses, while � scales the signed local correction.

Similar ARO means mask opposing changes in the constituent tasks: increasing the CIMR weight raises VG-Relation and lowers VG-Attribution in all four matched $( \tau _ { p } , \alpha )$ pairs. Increasing � at fixed CIMR weight and pooling temperature raises VG-Attribution in each tested comparison and leaves COCO Order unchanged or improves it. Flickr30k Order changes in both directions.

## 4.4. Image Replacement

The controls in Tab. 4 change the visual input while keeping caption candidates fixed. Real uses the matched image and measures accuracy with the intended visual evidence. Shufle permutes image assignments, giving all candidates for a record the same replacement image. It disrupts image– caption correspondence while retaining variation in visual input across records. Fixed uses one anchor image for all records within a task, removing this variation as well. The two replacement conditions test whether preference for the original gold caption persists across diferent replacement images or under one constant visual input.

Image-replacement controls for compositional matching (%). FSC-CLIP and the four matched scoring controls each use one checkpoint across all three conditions. Real uses matched images; Shufle uses a shared permutation; Fixed uses one anchor. The 6-task average covers SugarCrepe Add/Replace/Swap, ABC, COCO-Spatial and GQA-Spatial.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Image control</td><td colspan="4">ARO</td><td colspan="3">SugarCrepe</td><td rowspan="2">ABC</td><td colspan="2">Spatial</td><td rowspan="2"></td><td rowspan="2">6-task Avg.</td></tr><tr><td>VG-A</td><td>VG-R</td><td>Flickr Ord.</td><td>COCO Ord.</td><td>Avg.</td><td>Add</td><td>Repl. Swap</td><td></td><td>COCO-Sp. GQA-Sp.</td></tr><tr><td colspan="2">Reference methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">CLIP [1]</td><td>Real</td><td>63.21</td><td>63.36</td><td>58.88</td><td>47.60</td><td>58.26</td><td>75.09</td><td>80.76</td><td>63.34</td><td>64.11</td><td>46.88</td><td>46.69</td><td>62.81</td></tr><tr><td>Shuffle</td><td>47.66</td><td>41.11</td><td>10.36</td><td>7.52</td><td>26.66</td><td>62.27</td><td>46.72</td><td>43.25</td><td>44.69</td><td>46.45</td><td>50.17</td><td>48.93</td></tr><tr><td>Fixed</td><td>48.63</td><td>51.87</td><td>5.54</td><td>8.33</td><td>28.59</td><td>55.16</td><td>46.57</td><td>44.02</td><td>43.67</td><td>46.01</td><td>50.82</td><td>47.71</td></tr><tr><td rowspan="3">LABCLIP [16]</td><td>Real</td><td>68.32</td><td>81.45</td><td>82.90</td><td>79.95</td><td>78.15</td><td>81.02</td><td>83.25</td><td>73.66</td><td>69.97</td><td>48.24</td><td>45.18</td><td>66.89</td></tr><tr><td>Shuffle</td><td>54.35</td><td>64.92</td><td>47.30</td><td>34.50</td><td>50.27</td><td>54.33</td><td>48.35</td><td>53.16</td><td>50.58</td><td>50.21</td><td>48.84</td><td>50.91</td></tr><tr><td>Fixed</td><td>55.05</td><td>66.15</td><td>35.42</td><td>49.32</td><td>51.49</td><td>54.38</td><td>45.08</td><td>52.43</td><td>50.17</td><td>43.61</td><td>48.28</td><td>48.99</td></tr><tr><td rowspan="3">NegCLIP [7]</td><td>Real</td><td>70.50</td><td>81.12</td><td>90.50</td><td>86.97</td><td>82.27</td><td>85.80</td><td>85.02</td><td>75.37</td><td>73.58</td><td>46.54</td><td>45.91</td><td>68.70</td></tr><tr><td>Shuffle</td><td>52.30</td><td>54.69</td><td>29.66</td><td>20.45</td><td>39.28</td><td>76.30</td><td>50.04</td><td>52.64</td><td>50.20</td><td>48.12</td><td>48.63</td><td>54.32</td></tr><tr><td>Fixed</td><td>56.26</td><td>66.38</td><td>17.70</td><td>33.51</td><td>43.46</td><td>69.79</td><td>47.09</td><td>48.95</td><td>49.02</td><td>47.50</td><td>49.40</td><td>51.96</td></tr><tr><td rowspan="3">FSC-CLIP [12]</td><td>Real</td><td>70.67</td><td>80.53</td><td>95.50</td><td>94.08</td><td>85.19</td><td>88.98</td><td>87.32</td><td>77.48</td><td>74.76</td><td>47.91</td><td>46.17</td><td>70.44</td></tr><tr><td>Shuffle</td><td>47.47</td><td>34.20</td><td>23.20</td><td>14.48</td><td>29.84</td><td>77.82</td><td>49.34</td><td>45.50</td><td>45.04</td><td>48.97</td><td>49.23</td><td>52.65</td></tr><tr><td>Fixed</td><td>50.12</td><td>45.79</td><td>7.26</td><td>17.95</td><td>30.28</td><td>71.48</td><td>45.24</td><td>41.65</td><td>44.26</td><td>46.95</td><td>49.01</td><td>49.77</td></tr><tr><td colspan="2">Matched scoring controls</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Global-M</td><td>Real</td><td></td><td>71.04 84.23</td><td>91.24</td><td>90.20</td><td>84.18</td><td>79.23</td><td>82.01</td><td>75.30</td><td>71.12</td><td>49.91</td><td>43.03</td><td>66.77</td></tr><tr><td>Shuffle</td><td>57.47</td><td>64.16</td><td>48.96</td><td>38.22</td><td>52.20</td><td>58.17</td><td>47.50</td><td>54.99</td><td>52.32</td><td>49.51</td><td>49.96</td><td>52.08</td></tr><tr><td>Fixed</td><td>58.19</td><td>71.94</td><td>43.82</td><td>49.76</td><td>55.93</td><td>50.87</td><td>44.62</td><td>50.71</td><td>50.39</td><td>53.15</td><td>48.75</td><td>49.75</td></tr><tr><td rowspan="3">Global-M+CIMR</td><td>Real</td><td>71.42</td><td>84.78</td><td>91.28</td><td>90.25</td><td>84.43</td><td>79.96</td><td>82.16</td><td>75.08</td><td>71.46</td><td>49.25</td><td>43.38</td><td>66.88</td></tr><tr><td>Shuffle</td><td>53.45</td><td>52.80</td><td>25.28</td><td>14.12</td><td>36.41</td><td>58.90</td><td>48.47</td><td>46.43</td><td>48.43</td><td>50.62</td><td>48.37</td><td>50.20</td></tr><tr><td>Fixed</td><td>53.37</td><td>63.66</td><td>19.20</td><td>21.06</td><td>39.32</td><td>50.80</td><td>44.93</td><td>44.57</td><td>46.69</td><td>50.45</td><td>48.54</td><td>47.66</td></tr><tr><td rowspan="3">Flow-CLIP (HNB)</td><td>Real</td><td>76.22</td><td>85.50</td><td>91.40</td><td>89.90</td><td>85.75</td><td>80.79</td><td>83.02</td><td>77.83</td><td>78.56</td><td>48.98</td><td>45.18</td><td>69.06</td></tr><tr><td>Shuffle</td><td>57.52</td><td>67.22</td><td>40.72</td><td>29.52</td><td>48.75</td><td>61.11</td><td>46.96</td><td>53.90</td><td>51.45</td><td>47.88</td><td>49.27</td><td>51.76</td></tr><tr><td>Fixed</td><td>56.94</td><td>70.47</td><td>36.60</td><td>36.67</td><td>50.17</td><td>55.85</td><td>43.11</td><td>51.92</td><td>50.82</td><td>46.82</td><td>49.36</td><td>49.65</td></tr><tr><td rowspan="3">Flow-CLIP</td><td>Real</td><td>76.03</td><td>86.16</td><td>91.42</td><td>90.04</td><td>85.91</td><td>81.66</td><td>83.18</td><td>77.83</td><td>78.34</td><td>48.53</td><td>44.97</td><td>69.08</td></tr><tr><td>Shuffle</td><td>52.57</td><td>58.21</td><td>21.04</td><td>10.83</td><td>35.66</td><td>59.22</td><td>46.67</td><td>47.09</td><td>49.49</td><td>50.12</td><td>49.66</td><td>50.38</td></tr><tr><td>Fixed</td><td>51.75</td><td>65.78</td><td>15.82</td><td>15.50</td><td>37.21</td><td>55.08</td><td>43.40</td><td>46.21</td><td>46.03</td><td>46.48</td><td>49.61</td><td>47.80</td></tr></table>

FSC-CLIP [12] uses its COCO-trained checkpoint across all three image conditions. Each of the four matched scoring controls likewise uses the same checkpoint across conditions. The HNB and CIMR comparisons within each architecture therefore measure how the training objective afects responses to image replacement.

Accuracy after replacement measures how often a scorer retains the gold-caption preference with a diferent image. Gold labels remain tied to the original images; interpreting this retention therefore requires the corresponding matchedimage accuracy.

HNB-trained Flow-CLIP achieves mean ARO accuracy of 85.75% on matched images, 48.75% under Shufle and 50.17% under Fixed. Adding CIMR during training yields 85.91% on matched images and lowers Shufle and Fixed accuracy to 35.66% and 37.21%, decreases of 13.08 and 12.96 percentage points. Adding CIMR to Global-� also lowers ARO accuracy under image replacement, with little change in matched-image accuracy. For both architectures, captionorder tasks show larger reductions under Shufle and Fixed than attribution or relation tasks.

The caption candidates and their linguistic cues remain unchanged across these conditions. CIMR thus mitigates reliance on language priors in compositional matching, reducing the gold-caption preference retained under image replacement while maintaining matched-image accuracy.

For Flow-CLIP with the CIMR objective in Eq. (11), image replacement produces a larger accuracy drop on ARO than on the six-task average over SugarCrepe, ABC and spatial-relation tasks, which remains 50.38% under Shufle.

The matched-image ablation in Tab. 2 shows that adding CIMR to the HNB-trained global/local scorer raises CLEVR and PUG accuracy by 2.45 and 0.85 percentage points, while Swap-Attribute, ABC, COCO-Spatial and GQA-Spatial decrease by 0.75, 0.22, 0.45 and 0.21 points.

Lower caption-preference retention under image replacement coexists with gains and losses across the matched-image binding tasks.

## 4.5. Encoder Comparisons

Flow-CLIP’s gains occur in diferent task groups for SigLIP [5] and BLIP [3] in Tab. 5. Both learned heads improve ARO over the unadapted encoders. Relative to Global-�, Flow-CLIP increases SigLIP’s ARO mean from 88.91% to 89.92%, while binding decreases from 68.30% to 68.03%. BLIP exhibits the opposite pattern: ARO decreases from 91.50% to 90.85%, while binding increases from 70.56% to 71.10%.

Tab. 5  
Tab. 6  
Frozen encoder families with COCO-trained scoring heads (%). Base denotes pretrained scoring. The 5-task binding average covers Swap-A, Swap-O, ABC, COCO-Spatial and GQA-Spatial; Sugar $\operatorname { A v g } .$ . is the mean across Add, Replace and Swap. T2I/I2T report COCO Recall@1 in the two retrieval directions. Bold/underline indicate the best/second-best scores within each backbone.
<table><tr><td rowspan="3">Backbone</td><td rowspan="3">Scorer</td><td colspan="3">ARO</td><td colspan="4">Fine-grained binding</td><td colspan="2">SugarCrepe</td><td colspan="2">COCO retrieval</td></tr><tr><td colspan="2">Flickr COCO Ord.</td><td colspan="2"></td><td colspan="2">Swap- Swap-</td><td colspan="2">COCO- GQA-  $_ { 5 - \tt t a s k }$ </td><td colspan="2">Sugar</td><td colspan="2"></td></tr><tr><td>VG-A VG-R</td><td>Ord.</td><td> ${ \mathsf { A v g . } }$ </td><td>A</td><td>0</td><td>ABC</td><td> $\mathsf { S p . }$   ${ \mathsf { S p . } }$ </td><td> ${ \mathsf { A v g . } }$ </td><td>Add Repl. Swap</td><td> ${ \mathsf { A v g . } }$ </td><td>T2l</td><td>I2T</td></tr><tr><td rowspan="3">CLIP ViT-B/32 [1]</td><td>Base</td><td>63.21 63.36 58.88</td><td>47.60</td><td>58.26</td><td>63.96</td><td>61.63</td><td>64.11</td><td>46.88 46.69</td><td>56.66 75.09</td><td>80.76 63.34</td><td>73.06</td><td>30.22</td><td>50.44</td></tr><tr><td>Global-M</td><td>71.04 84.23</td><td>91.24 90.20</td><td>84.18</td><td>76.43</td><td>72.24</td><td>71.12</td><td>49.91</td><td>43.03 62.55</td><td>79.23 82.01</td><td>75.30 78.85</td><td>38.06</td><td>49.60</td></tr><tr><td>Flow-CLIP</td><td>76.03 86.16 91.42</td><td>90.04</td><td>85.91</td><td>77.93</td><td>77.55</td><td>78.34</td><td>48.53 44.97</td><td>65.46</td><td>81.66 83.18 77.83</td><td>80.89</td><td>38.28</td><td>50.14</td></tr><tr><td rowspan="3">SigLIP ViT-B/16 [5]</td><td>Base</td><td>55.87 32.04</td><td>41.44 33.07</td><td>40.60</td><td>71.47</td><td>60.41</td><td>72.49</td><td>47.55 48.36</td><td>60.06</td><td>87.69 84.56 68.50</td><td>80.25</td><td>47.84</td><td>65.76</td></tr><tr><td>Global-M</td><td>78.77 86.52</td><td>95.50 94.87</td><td>88.91</td><td>85.74</td><td>74.69</td><td>81.57</td><td>50.76</td><td>48.75 68.30</td><td>88.05 87.62 82.77</td><td>86.15</td><td>49.50</td><td>60.50</td></tr><tr><td>Flow-CLIP</td><td>80.77 86.99 95.98</td><td>95.94</td><td>89.92</td><td>85.74</td><td>73.47</td><td>81.95</td><td>51.44 47.55</td><td>68.03</td><td>88.38 88.14 82.44</td><td>86.32</td><td>48.97</td><td>61.58</td></tr><tr><td rowspan="3">BLIP ITC-Base [3]</td><td>Base</td><td>81.83 53.17</td><td>28.22 21.80</td><td>46.26</td><td>84.38</td><td>75.10</td><td>86.49</td><td>45.72 49.14</td><td>68.17</td><td>95.53 90.54 81.89</td><td>89.32</td><td>57.08</td><td>73.98</td></tr><tr><td>Global-M</td><td>86.31 88.97</td><td>94.76 95.97</td><td>91.50</td><td>87.99</td><td>79.59</td><td>88.45</td><td>49.40</td><td>47.38 70.56</td><td>93.03 92.80 85.73</td><td>90.52</td><td>56.99</td><td>71.02</td></tr><tr><td>Flow-CLIP</td><td>83.98 88.62 94.64</td><td>96.17</td><td>90.85</td><td>89.04</td><td>80.41</td><td>88.80</td><td>50.10 47.16</td><td>71.10</td><td>92.92 92.93 86.72</td><td>90.86</td><td>56.83</td><td>69.88</td></tr></table>

Frozen CLIP [1] backbones with COCO-trained scoring heads (%). ARO Avg. is the mean across four tasks; the 5-task binding average covers Swap-A, Swap-O, ABC, COCO-Spatial and GQA-Spatial; SugarCrepe Avg. is the mean across Add, Replace and Swap. T2I/I2T denote COCO text-to-image/image-to-text Recall@1. Bold/underline indicate the best/second-best scores within each backbone.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Scorer</td><td colspan="4">ARO</td><td colspan="2">Binding (5-task)</td><td rowspan="2">SugarCrepe</td><td colspan="2">COCO retrieval</td></tr><tr><td>VG-A</td><td>VG-R</td><td>Flickr Ord.</td><td>COCO Ord.</td><td> $\mathsf { A v g } .$ </td><td> $\mathsf { A v g . }$ </td><td> $\mathsf { A v g } .$  T2I</td><td>I2T</td></tr><tr><td rowspan="4">ViT-B/32 [1]</td><td>Base</td><td>63.21</td><td>63.36</td><td>58.88</td><td>47.60</td><td>58.26</td><td>56.66</td><td>73.06</td><td>30.22</td><td>50.44</td></tr><tr><td>Global-M</td><td>71.04</td><td>84.23</td><td>91.24</td><td>90.20</td><td>84.18</td><td>62.55</td><td>78.85</td><td>38.06</td><td>49.60</td></tr><tr><td>Global-M+CIMR</td><td>71.42</td><td>84.78</td><td>91.28</td><td>90.25</td><td>84.43</td><td>62.54</td><td>79.07</td><td>38.16</td><td>48.68</td></tr><tr><td>Flow-CLIP</td><td>76.03</td><td>86.16</td><td>91.42</td><td>90.04</td><td>85.91</td><td>65.46</td><td>80.89</td><td>38.28</td><td>50.14</td></tr><tr><td rowspan="4">ViT-B/16 [1]</td><td>Base</td><td>62.56</td><td>63.91</td><td>57.06</td><td>49.78</td><td>58.33</td><td>57.70</td><td>73.39</td><td>32.97</td><td>52.46</td></tr><tr><td>Global-M</td><td>71.87</td><td>83.86</td><td>92.56</td><td>92.10</td><td>85.10</td><td>64.45</td><td>79.62</td><td>41.28</td><td>53.30</td></tr><tr><td>Global-M+CIMR</td><td>71.53</td><td>84.27</td><td>92.48</td><td>92.25</td><td>85.13</td><td>63.98</td><td>79.49</td><td>41.33</td><td>51.94</td></tr><tr><td>Flow-CLIP</td><td>73.34</td><td>85.13</td><td>92.48</td><td>92.19</td><td>85.78</td><td>65.57</td><td>81.53</td><td>41.26</td><td>53.54</td></tr><tr><td rowspan="4">ViT-L/14 [1]</td><td>Base</td><td>61.87</td><td>64.63</td><td>55.80</td><td>46.56</td><td>57.22</td><td>56.29</td><td>72.97</td><td>36.14</td><td>56.20</td></tr><tr><td>Global-M</td><td>69.38</td><td>84.67</td><td>93.70</td><td>93.21</td><td>85.24</td><td>64.79</td><td>79.84</td><td>43.51</td><td>55.44</td></tr><tr><td>Global-M+CIMR</td><td>68.74</td><td>84.12</td><td>94.02</td><td>93.27</td><td>85.04</td><td>64.23</td><td>79.55</td><td>43.46</td><td>54.28</td></tr><tr><td>Flow-CLIP</td><td>73.61</td><td>83.81</td><td>93.26</td><td>92.90</td><td>85.90</td><td>67.09</td><td>82.99</td><td>45.19</td><td>57.26</td></tr></table>

All heads are trained on COCO using their frozen encoder’s features, which difer in image pooling, patch resolution and projection dimension. Relative to unadapted BLIP, both learned heads improve ARO accuracy but reduce COCO image-to-text Recall@1.

Binding gains over Global-� persist across the three CLIP [1] backbones in Tab. 6, but their size does not increase monotonically with backbone capacity. For ViT-B/32, ViT-B/16 and ViT-L/14, the five-task mean increases by 2.91, 1.12 and 2.30 percentage points, respectively, while Flow-CLIP’s ARO averages range from 85.78% to 85.91%. On ViT-L/14, the binding and SugarCrepe means improve alongside both retrieval directions, despite decreases in several ARO tasks.

## 4.6. Classification and Retrieval

The results in Tab. 7 evaluate the learned score with classname prompts and caption queries. Relative to pretrained CLIP [1], training on either COCO or CC3M improves text-to-image retrieval on Flickr30k and COCO, with larger gains after COCO training. Top-1 classification decreases on all three datasets, while COCO-trained Flow-CLIP nearly preserves COCO image-to-text retrieval.

Tab. 7  
Zero-shot Top-1 classification and Recall@1 retrieval (%). T2I retrieves images from captions; I2T retrieves captions from images. Colored parentheses report percentage-point changes from frozen CLIP (blue: gain; red: loss).
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">CLIP</td><td rowspan="3"> ${ \mathsf { N e g C L I P } }$ </td><td colspan="2">Global-M</td><td colspan="2">Flow-CLIP</td></tr><tr><td>COCO CC3M</td><td></td><td>COCO</td><td>CC3M</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Classification</td><td> $( T o p \mathrm { - } \lambda )$ </td><td>89.0</td><td></td><td></td><td>85.2 86.5(-1.8) 86.4(−1.9)</td><td></td></tr><tr><td>CIFAR-10 CIFAR-100</td><td>88.3 64.5</td><td>63.0</td><td>86.1 58.9</td><td></td><td>58.9 58.0 (–6.5) 59.3(–5.2)</td><td></td></tr><tr><td>ImageNetV2</td><td>54.7</td><td>48.7</td><td>45.3</td><td>50.1</td><td>47.0(-7.7)</td><td>50.0 (-4.7)</td></tr><tr><td>Retrieval (R@1)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>Flickr30k T2l</td><td>59.1</td><td>71.0</td><td>63.3</td><td>62.2</td><td> $6 3 . 4 \ : ( + 4 . 3 )$ </td><td>61.5(+2.4)</td></tr><tr><td>Flickr30k I2T</td><td>78.5</td><td>85.0</td><td>75.5</td><td>73.7</td><td> $7 6 . 0 ( - 2 . 5 )$ </td><td>73.1 (-5.4)</td></tr><tr><td>COCO T2I</td><td>30.2</td><td>45.0</td><td>38.1</td><td></td><td></td><td>33.9 38.3 (+8.1) 32.7 (+2.5)</td></tr><tr><td>COCO I2T</td><td>50.4</td><td>59.0</td><td>49.6</td><td>43.6</td><td> $5 0 . 1 ( - 0 . 3 )$ </td><td>43.1 (−7.3)</td></tr></table>

Relative to Global-�, COCO-trained Flow-CLIP improves all four retrieval scores, whereas CC3M-trained Flow-CLIP reduces them. Classification changes vary across datasets. The retrieval efect thus difers between the COCO and CC3M adaptation settings, which also difer in caption generation, training budget and pooling temperature.

blue anlylosaurus and red armadillo red armadillo and blue anlylosaurus

![](images/8bc9bed11c695fce8154758ea6f0b09fe79af42ed198aa76d725c93a9f8b14c9.jpg)

![](images/a2458fdd53a3fd5926b3172d8828363ec3bdf4cf14414e1150899bc677417797.jpg)  
Positive: a small blue plane sitting on top of a field. Foil: field plane sitting on a small blue top of a.

![](images/5bf9ac3e92b98db37ec0d269faf176c8ef817d008e1ef05ebe4caebfcc455a24.jpg)  
Positive: Palm trees are in the foreground of a picture with a clock tower standing above them. Foil: A river flows in the background of a picture with a clock tower standing above palm trees.

![](images/0afc2c868cd32bbe560d3acfb922dea73fa89dbf2ad6ea4b5fe1917861435ff1.jpg)

![](images/c442e147a49a8efc1d39b8075c51bd04ccdb03482827dc8daede32904306095a.jpg)  
Positive: A horse pulling a carriage on a narrow path. Foil: A carriage pulling a horse on a narrow path.

Fig. 4. Compositional evidence across models. Word-order, object-replacement, relation-binding, and object-binding cases compare evidence maps and gold–foil score margins. Warm and cool colors indicate responses above and below the within-image reference. Margins are reported on each method’s native scoring scale.  
![](images/eb80bd8bd0c9b46a6467a6cca01cfd983c6cd67ac01d7adceab191102d598d65.jpg)

![](images/9b8985de2e7c8d11006fecd7be3ec79282d2391d18918677838ac96949862c3c.jpg)

![](images/6fa5bea08e7052b9972dbe4f2cc75bec046e2a856efe11b97d7e6d79c524fdbc.jpg)

![](images/48e4ead70af1ee67078b1e4dd89febb8b6a3dbd446d52e6f1ace6845a91bd30a.jpg)

![](images/b9e4474daf191537a8ea74d444e16e30a5753e72eb4e413fb3fdea8a250bb7bf.jpg)  
grass anlylosaurus and stone rhinoceros

![](images/29978b14029aae02d900a6248ac10d19a06535b67a7ad8f31115eb9692a72864.jpg)

![](images/7c705f65a0c06c901a3743b576991dca7efd541458cc772061bb4db082b5375f.jpg)

![](images/aee41e9a0111e5fab343049226a9eee284aabddd2d540f17619ebc4ea9a8bfe1.jpg)  
blue anlylosaurus and red squirrel

![](images/1fb5d9ba9c79cbd21da6ae0bc67bd488cbb7c98386d378d7283c0ab98ef4b9a3.jpg)

![](images/110ee4c10f988e8463e728913d99172f448fdc933969430427154742ceb11a3b.jpg)

![](images/8306137b220549045993d93dd5766adaef49e7d71ef316efbfa5da740f44dffd.jpg)

![](images/c9a8b4c2ab0aa2d6794f6182b76fed70bec89949811875d027b83fbf88764b79.jpg)

![](images/8226ba027995157af645d4ecb8afc44af91f5ed433c06461d48631a27eb14c25.jpg)

![](images/8ebff8263f03e21ef4550c664638c1bd6ba6407cb2610aa62dcad9b60c7a49d5.jpg)

![](images/a587275add3252c57cac21036149239cc6ab1916e3bda77523f12e4f399ca415.jpg)

![](images/f75758d9175065e9982f5dd3dc76feaf253c77db597d8342349bfda845d1b70e.jpg)  
Fig. 5. PUG:SPAR color–object binding repairs. Each example pairs an oficial PUG:SPAR image with a gold color–object assignment and a swapped foil. Bars show the gold–foil score margin for CLIP [1] and Flow-CLIP. Positive values indicate that the gold binding is ranked higher. In these selected cases, Flow-CLIP ranks the gold assignment above the foil, reversing the CLIP decision.

![](images/23bc4e2791808fd79279dd57a395df8c6deca137c0b264c21dea1da491d4be79.jpg)

<table><tr><td>Caption</td><td>CLIP CLIP</td><td>Flow-</td></tr><tr><td>√cracks in pavement and another pavement</td><td></td><td>#4 #2</td></tr><tr><td>× cracks across pavement and another #3 #5 pavement</td><td></td><td></td></tr><tr><td>× mouths in pavement and another #2 #6 pavement</td><td></td><td></td></tr><tr><td>× cracks in stand and another pavement</td><td></td><td>#6#3</td></tr><tr><td>× cracks under pavement and another #5 #4 pavement</td><td></td><td></td></tr><tr><td>× dips in pavement and another pavement</td><td></td><td>#1 #1</td></tr></table>

![](images/e61aa033fabfd47c4bbdb4bce9aef2a1326242c393fd28f287fc38db2e3e5e76.jpg)

<table><tr><td>Caption</td><td></td><td>Flow- CLIP CLIP</td></tr><tr><td>√ man wearing gray shorts. there is a #5 #1 racket.</td><td></td><td></td></tr><tr><td>× man wearing racket. there is a gray #1 #2</td><td></td><td></td></tr><tr><td>shorts. × shorts wearing gray man. there is a #6 #6 racket.</td><td></td><td></td></tr><tr><td>× racket wearing gray shorts. there is a #3 #4 man.</td><td></td><td></td></tr><tr><td>× gray shorts wearing man. there is a #4 #5 racket.</td><td></td><td></td></tr><tr><td>× man wearing gray racket. there is a #2 #3 shorts.</td><td></td><td></td></tr></table>

![](images/1f9d959fc6743c66d8e207f6803797140075b573b50f8ab71438a82b840d0a31.jpg)

<table><tr><td>Caption</td><td>CLIP</td><td>Flow- CLIP</td></tr><tr><td>✓boy playing with ball, ball2 in field</td><td>#5</td><td>#1</td></tr><tr><td>× boy playing with ball, ball2 in right X boy playing with game, ball2 in field</td><td>#3</td><td>#4</td></tr><tr><td>× boy playing with ball, ball2 in short</td><td>#4</td><td>#2</td></tr><tr><td>× boy playbyg with ball, ball2 in field</td><td>#6</td><td>#3</td></tr><tr><td>× player playing with ball, ball2 in field</td><td>#2 #1</td><td>#6 #5</td></tr></table>

![](images/d8de176bd4d616b825eac261b51606ef3074a565efae095b82a462e25ccba4ca.jpg)

<table><tr><td>Caption</td><td>CLIP CLIP</td><td>Flow-</td></tr><tr><td>√ sheep eating grass in a field</td><td>#3</td><td>#1</td></tr><tr><td>X sheep eating peanut in a field</td><td>#5</td><td>#5</td></tr><tr><td>X sheep eating bean in a field</td><td>#4</td><td>#6</td></tr><tr><td>X sheep eatbyg grass in a field</td><td>#2#3</td><td></td></tr><tr><td>X sheep eatnearg grass in a field</td><td>#1</td><td>#4</td></tr><tr><td>X sheep eating grass in a desert</td><td>#6</td><td>#2</td></tr></table>

Fig. 6. Caption rankings for four CREPE [25] examples. Each image has one gold caption and five foils. Columns give CLIP [1] and Flow-CLIP ranks, with rank 1 best. Green checks identify gold captions and red crosses identify foils.

![](images/8a82673f45c3e474fd0feac401082bae3088052185135bc0de5c4ee3989d7607.jpg)  
Fig. 7. Cross-model residual-evidence atlas on ARO. The eight examples cover VG-Attribution, VG-Relation, Flickr30k Order, and COCO Order. For each example, the input image is followed by patches selected by Flow-CLIP’s centered-feature sensitivity and the gold–hard-foil score margins of CLIP [1], NegCLIP [7], LABCLIP [16], and Flow-CLIP. Positive margins indicate correct pairwise rankings

## 4.7. Qualitative Analysis

Local evidence. The four examples in Fig. 4 compare token–patch responses and gold–foil margins for word order, object replacement, relation binding and object binding. Warm and cool regions indicate responses above and below the within-image mean. They locate patch support for each caption, while the adjacent margins show which caption the final score prefers.

LABCLIP and Flow-CLIP both select the gold caption in all four pairs. Their respective margins are 0.063 and 0.134 for the aircraft example, and 0.019 and 0.054 for the plate example. Positive signs indicate correct pairwise rankings; the magnitudes are specific to each scoring function. The aircraft captions difer in word order, while the plate captions reverse an object relation.

Attribute assignments. The PUG:SPAR [24] examples illustrated in Fig. 5 hold the object inventory and scene templates fixed while exchanging color assignments. CLIP [1] assigns a negative gold–foil score margin to every displayed pair, whereas Flow-CLIP assigns a positive margin.

Caption rankings. In the four CREPE [25] examples in Fig. 6, Flow-CLIP selects the gold caption in three cases where CLIP [1] selects a foil.

In the racket scene, the selected caption correctly identifies the man’s clothing, while CLIP prefers a description that treats the racket as clothing. In the football and sheep scenes, Flow-CLIP ranks the gold descriptions above alternatives that substitute the described object or location.

The pavement example shows a partial improvement: the gold caption rises from fourth to second, but an alternative describing a diferent pavement feature remains first. The higher gold-caption rank still leaves another candidate as the final selection.

Patch sensitivity. The eight cases in Fig. 7 compare patch sensitivity and caption-pair margins across attribution, relation and word-order tasks. Patch selection uses the firstorder change in the gold–foil evidence increment along the centered patch-feature direction, highlighting regions to which this local comparison is sensitive.

In the attribution cases, the selected regions cover the door, building, wall and pillow whose properties distinguish the captions. The adjacent margins report the final caption preference.

Margin changes. In Fig. 8, gold–foil margins increase for 47 of 48 illustrative pairs and decrease for one. After excluding two order cases with duplicate gold captions, correct top-1 selections increase from 17 to 45 of 46 cases, with 28 repairs and no correct-to-incorrect reversals. A COCO Order case remains incorrect: its gold rank worsens from 3 to 4 as another caption outranks it despite the improved margin over the displayed foil. These order-task candidate sets difer from the full benchmark.

![](images/5da9de1c0611e08dbc902c6fe426beadada6668868e482c52caa442fc97b51c7.jpg)  
Fig. 8. Changes in caption-pair margins. Each arrow connects the Global-� and Flow-CLIP gold–foil score margins for the same image and caption pair. The zero boundary separates incorrect and correct rankings. Enlarged markers show the task means.

## 5. Conclusion

We presented Flow-CLIP to improve attribute–object binding and mitigate reliance on language priors in compositional image–text matching. GCC calibrates global alignment, ICER adjusts this score through image-conditioned changes in token–patch evidence, and CIMR supervises caption-pair margins across matched and replacement images. Experiments show improved binding beyond learned global alignment and reduced reliance on language priors. These findings support using image-conditioned evidence refinement to resolve binding ambiguities that persist after sentence-level alignment.

## AI USE STATEMENT

Generative AI tools were used only to assist with manuscript writing.

## References

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning, Vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 8748–8763.

[2] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.- H. Sung, Z. Li, T. Duerig, Scaling up visual and vision-language representation learning with noisy text supervision, in: Proceedings of the 38th International Conference on Machine Learning, Vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 4904– 4916.

[3] J. Li, D. Li, C. Xiong, S. Hoi, Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation, in: International conference on machine learning, PMLR, 2022, pp. 12888–12900.

[4] X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, L. Beyer, LiT: Zero-shot transfer with locked-image text tuning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 18123–18133.

[5] X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer, Sigmoid loss for language image pre-training, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11975–11986.

[6] T. Thrush, R. Jiang, M. Bartolo, A. Singh, A. Williams, D. Kiela, C. Ross, Winoground: Probing vision and language models for visio-linguistic compositionality, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5238–5248.

[7] M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, J. Zou, When and why vision-language models behave like bags-of-words, and what to do about it?, in: 11th International Conference on Learning Representations, ICLR 2023, International Conference on Learning Representations, ICLR, 2023.

[8] T. Zhao, T. Zhang, M. Zhu, H. Shen, K. Lee, X. Lu, J. Yin, VL-CheckList: Evaluating pre-trained vision-language models with objects, attributes and relations, arXiv preprint arXiv:2207.00221 (2022).

[9] L. Parcalabescu, M. Cafagna, L. Muradjan, A. Frank, I. Calixto, A. Gatt, VALSE: A task-independent benchmark for vision and language models centered on linguistic phenomena, in: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Dublin, Ireland, 2022, pp. 8253–8280. doi:10.18653/v1/2022.acl-long. 567.

[10] M. Lewis, N. Nayak, P. Yu, J. Merullo, Q. Yu, S. Bach, E. Pavlick, Does CLIP bind concepts? probing compositionality in large image models, in: Findings of the Association for Computational Linguistics: EACL 2024, Association for Computational Linguistics, St. Julian’s, Malta, 2024, pp. 1487–1500. doi:10.18653/v1/2024.findings-eacl.101.

[11] A. Kamath, J. Hessel, K.-W. Chang, What’s “up” with vision-language models? investigating their struggle with spatial reasoning, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Singapore, 2023, pp. 9161–9175. doi:10.18653/v1/2023.emnlp-main. 568.

[12] Y. Oh, J. W. Cho, D.-J. Kim, I. S. Kweon, J. Kim, Preserving multi-modal capabilities of pre-trained VLMs for improving visionlinguistic compositionality, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Miami, Florida, USA, 2024, pp. 19060– 19076. doi:10.18653/v1/2024.emnlp-main.1062.

[13] S. Doveh, A. Arbelle, S. Harary, E. Schwartz, R. Herzig, R. Giryes, R. Feris, R. Panda, S. Ullman, L. Karlinsky, Teaching structured vision & language concepts to vision & language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2657–2668.

[14] S. Doveh, A. Arbelle, S. Harary, R. Herzig, D. Kim, P. Cascante-Bonilla, A. Alfassy, R. Panda, R. Giryes, R. Feris, S. Ullman, L. Karlinsky, Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models, in: Advances in Neural Information Processing Systems, Vol. 36, Curran Associates, Inc., 2023, pp. 76137– 76150. doi:10.52202/075280-3327.

[15] H. Singh, P. Zhang, Q. Wang, M. Wang, W. Xiong, J. Du, Y. Chen, Coarse-to-fine contrastive learning in image-text-graph space for improved vision-language compositionality, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Singapore, 2023, pp. 869– 893. doi:10.18653/v1/2023.emnlp-main.56.

[16] D. Koishigarina, A. Uselis, S. J. Oh, Clip behaves like a bag-of-words model cross-modally but not uni-modally, in: International Conference on Learning Representations, 2026, pp. 96106–96127.

[17] L. Yao, R. Huang, L. Hou, G. Lu, M. Niu, H. Xu, X. Liang, Z. Li, X. Jiang, C. Xu, FILIP: Fine-grained Interactive Language-Image Pre-Training, in: International Conference on Learning Representations, 2022.

[18] R. Kang, Y. Song, G. Gkioxari, P. Perona, Is clip ideal? no. can we fix it? yes!, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 22436–22446.

[19] K. Jiang, X. He, R. Xu, X. Wang, ComCLIP: Training-free compositional image and text matching, in: Proceedings of the 2024 Conference

of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Association for Computational Linguistics, Mexico City, Mexico, 2024, pp. 6639–6659. doi:10.18653/v1/2024.naacl-long.370.

[20] R. Assouel, P. Astolfi, F. Bordes, M. Drozdzal, A. Romero-Soriano, Object-centric binding in Contrastive Language-Image Pretraining, in: Advances in Neural Information Processing Systems, Vol. 38, Curran Associates, Inc., 2025, pp. 131418–131458. doi:10.52202/ 085713-4375.

[21] R. Xiao, S. Kim, M.-I. Georgescu, Z. Akata, S. Alaniz, Flair: Vlm with fine-grained language-informed image representations, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE Computer Society, 2025, pp. 24884–24894.

[22] C.-Y. Hsieh, J. Zhang, Z. Ma, A. Kembhavi, R. Krishna, SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality, in: Advances in Neural Information Processing Systems, Vol. 36, Curran Associates, Inc., 2023, pp. 31096–31116. doi:10.52202/ 075280-1355.

[23] J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick, R. Girshick, CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2901–2910.

[24] F. Bordes, S. Shekhar, M. Ibrahim, D. Bouchacourt, P. Vincent, A. Morcos, PUG: Photorealistic and semantically controllable synthetic data for representation learning, in: Advances in Neural Information Processing Systems, Vol. 36, Curran Associates, Inc., 2023, pp. 45020– 45054. doi:10.52202/075280-1952.

[25] Z. Ma, J. Hong, M. O. Gul, M. Gandhi, I. Gao, R. Krishna, CREPE: Can vision-language foundation models reason compositionally?, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 10910–10921.

[26] I. Miranda, A. Salaberria, E. Agirre, G. Azkune, BiVLC: Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval, in: Advances in Neural Information Processing Systems, Vol. 37, Curran Associates, Inc., 2024, pp. 101880–101904. doi: 10.52202/079017-3232.

[27] S. H. Dumpala, A. Jaiswal, C. Sastry, E. Milios, S. Oore, H. Sajjad, SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations, in: Advances in Neural Information Processing Systems, Vol. 37, Curran Associates, Inc., 2024, pp. 17972– 18018. doi:10.52202/079017-0571.

[28] K. Alhamoud, S. Alshammari, Y. Tian, G. Li, P. H. Torr, Y. Kim, M. Ghassemi, Vision-language models do not understand negation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 29612–29622.

[29] B. Kargi, A. Uselis, S. J. Oh, Half-truths break similarity-based retrieval, arXiv preprint arXiv:2602.23906 (2026).

[30] A. Uselis, D. Koishigarina, S. J. Oh, How can embedding models bind concepts?, in: Proceedings of the 43rd International Conference on Machine Learning, Vol. 306 of Proceedings of Machine Learning Research, PMLR, 2026, pp. 123108–123135.

[31] Y. Zhong, J. Yang, P. Zhang, C. Li, N. Codella, L. H. Li, L. Zhou, X. Dai, L. Yuan, Y. Li, J. Gao, Regionclip: Region-based languageimage pretraining, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16793–16803.

[32] F. Zohra, C. Zhao, H. Itani, B. Ghanem, �-CLIP: Text-Conditioned Contrastive Learning for Multi-Granular Vision-Language Alignment, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 680–689.

[33] Q. Zhang, Y. Chen, L. Deng, L. Shen, Abe-clip: Training-free attribute binding enhancement for compositional image-text matching, arXiv preprint arXiv:2512.17178 (2025).

[34] Z. Lin, D. Pathak, B. Li, J. Li, X. Xia, G. Neubig, P. Zhang, D. Ramanan, Evaluating Text-to-Visual Generation with Image-to-Text Generation, arXiv preprint arXiv:2404.01291 (2024).

URL https://arxiv.org/abs/2404.01291

[35] M. Patel, A. Kusumba, S. Cheng, C. Kim, T. Gokhale, C. Baral, Y. Yang, TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives, in: Advances in Neural Information Processing Systems, Vol. 37, Curran Associates, Inc., 2024, pp. 32731–32760. doi:10.52202/079017-1029.

[36] H. Li, B. Li, Enhancing vision-language compositional understanding with multimodal synthetic data, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 24849–24861.

[37] L. Zhang, R. Awal, A. Agrawal, Contrasting intra-modal and ranking cross-modal hard negatives to enhance visio-linguistic compositional understanding, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13774–13784.

[38] J. Kwon, K. Min, J.-y. Sohn, Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions, in: Advances in Neural Information Processing Systems, Vol. 38, Curran Associates, Inc., 2025, pp. 13939–13966. doi:10.52202/085713-0468.

[39] H. X. Pham, D. T. Hofmann, R. Guerrero, B. Martinez, No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 33901–33910.

[40] A. Peleg, N. D. Singh, M. Hein, Advancing Compositional Awareness in CLIP with Eficient Fine-Tuning, in: Advances in Neural Information Processing Systems, Vol. 38, Curran Associates, Inc., 2025, pp. 56959– 56998. doi:10.52202/085713-1908.

[41] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft coco: Common objects in context, in: European conference on computer vision, Springer, 2014, pp. 740– 755.

[42] A. Karpathy, L. Fei-Fei, Deep visual-semantic alignments for generating image descriptions, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3128–3137.

[43] X. Chen, H. Fang, T.-Y. Lin, R. Vedantam, S. Gupta, P. Dollár, C. L. Zitnick, Microsoft coco captions: Data collection and evaluation server, arXiv preprint arXiv:1504.00325 (2015).

[44] P. Sharma, N. Ding, S. Goodman, R. Soricut, Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning, in: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 2556–2565.

[45] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, Y. Wu, CoCa: Contrastive captioners are image-text foundation models, Transactions on Machine Learning Research (2022). URL https://arxiv.org/abs/2205.01917

[46] W. Feng, X. He, T.-J. Fu, V. Jampani, A. R. Akula, P. Narayana, S. Basu, X. E. Wang, W. Y. Wang, Training-free structured difusion guidance for compositional text-to-image synthesis, in: The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PUIqjT4rzq7

[47] P. Young, A. Lai, M. Hodosh, J. Hockenmaier, From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions, Transactions of the association for computational linguistics 2 (2014) 67–78.

[48] A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, Tech. rep., University of Toronto (2009). URL https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR. pdf

[49] B. Recht, R. Roelofs, L. Schmidt, V. Shankar, Do imagenet classifiers generalize to imagenet?, in: International conference on machine learning, PMLR, 2019, pp. 5389–5400.

[50] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1711.05101