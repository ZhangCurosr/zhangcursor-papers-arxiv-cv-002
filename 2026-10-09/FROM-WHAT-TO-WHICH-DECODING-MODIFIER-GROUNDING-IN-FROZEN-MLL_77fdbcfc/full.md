# FROM WHAT TO WHICH: DECODING MODIFIER GROUNDING IN FROZEN MLLMS

Barbara Toniella Corradini   
AI for Good (AIGO), Istituto Italiano di Tecnologia, Italy   
barbara.corradini@iit.it   
Caterina Gallegati   
University of Siena, Italy   
c.gallegati@student.unisi.it   
Ludovica Genovese   
AI for Good (AIGO), Istituto Italiano di Tecnologia, Italy   
University of Genoa, Italy   
ludovica.genovese@iit.it   
Vittorio Murino   
AI for Good (AIGO), Istituto Italiano di Tecnologia, Italy   
University of Verona, Italy   
vittorio.murino@iit.it

## ABSTRACT

As Multimodal Large Language Models (MLLMs) can describe increasingly complex visual scenes, token-level grounding becomes crucial. Yet, when an MLLM generates “the yellow banana on the left”, established grounding approaches focus on what is in the image (“banana”), overlooking tokens that help describe which instance is meant (“yellow”, “left”). In this work, we ask whether frozen MLLM representations contain decodable grounding information about the referred instance across generated tokens, extending to modifiers such as attributes, spatial expressions, and relational/action terms. To address this question, we introduce OTTER, a lightweight supervised probe over frozen MLLM representations that uses Optimal Transport (OT) to align generated tokens with visual regions and produce compact grounding maps. Our results show that (i) instancediscriminative visual information can be decoded from modifier tokens, with the clearest evidence for spatial terms, but (ii) is not confined to them, as contextualized object nouns also carry referential information; (iii) the recovered grounding remains informative under context perturbations, while selected regions remain relevant to generation; and (iv) the learned OT-based grounding extends beyond the controlled setting to free generation and cross-dataset transfer.

## 1 INTRODUCTION

Recent advances in Multimodal Large Language Models (MLLMs) have made it increasingly important to understand how their generated language is grounded in visual evidence. Existing tokenlevel grounding methods primarily act at an object-level (i.e., via noun tokens), focusing on what is in the image. For instance, early works exploit raw attention weights that provide a readily avail able grounding signal, however they may reflect contextual or generic visual dependencies rather than token-specific grounding evidence (Jain & Wallace, 2019; Serrano & Smith, 2019). Activationbased and gradient-based methods (Wang et al., 2020; Li et al., 2025; Selvaraju et al., 2017) have also been adapted to MLLMs for token-level visual attribution, but they struggle due to the contextual dependence of subsequently generated tokens, and evaluations are largely limited to object nouns. Current literature leaves a major blind spot in visual grounding, which we term the what-towhich problem—the shift from object-level to instance-level grounding, a pivotal but still uncovered challenge in multi-instance applications and cluttered scene understanding. In fact, given the prompt “the spotted dog sitting on the right” and an image of dogs, noun grounding may highlight a “dog” in the image, but it is not guaranteed it is the intended referent. This ambiguity is solved at language level by additional details, such as “spotted”, “right”, and “sitting”, but established grounding methods systematically discard these modifier tokens. Recently, contextual-embedding methods (Phukan et al., 2025) show that contextual MLLM representations can support modifier grounding, yet they perform the analysis at the level of answer representations rather than at that of individual generated tokens. Differently, grounding-oriented fine-tuning (Chen et al., 2023; Peng et al., 2024; Zhang et al., 2023; 2024; Ma et al., 2024) manages to improve localization by changing the model itself, neglecting to isolate grounding already present in its frozen representations.

Therefore, the what-to-which problem remains unsolved at the generated-token level. We address this gap by combining two distinct but complementary objectives: first, by expanding the analysis beyond nouns to include instance-discriminative modifiers, we target the specific tokens required to resolve the referential ambiguity; second, by keeping the underlying model strictly frozen, we frame our analysis as a pure interpretability effort, aiming to uncover the grounding mechanisms already embedded in its internal representations. This motivates our central question: how dofrozen MLLMs visually ground the what-to-which problem across their generated tokens? To address this question, we introduce OTTER (Optimal Transport for Token-level Expression-to-Region align ment), a lightweight supervised probe over frozen MLLM representations. We approach token-level grounding as a matching problem between generated tokens and visual regions. In particular, we train OTTER on Referring Instance Segmentation (RIS), aiming to identify more compatible token– region pairs and exclude those token or region with less meaningful visual match. OTTER models this structure through entropy-regularized Optimal Transport (OT), producing token-level grounding maps without modifying the underlying MLLM. Importantly, since accurate localization alone would not establish that modifier tokens carry instance-specific visual information, We design a controlled grounding protocol to test whether modifiers contribute grounding beyond object identity, to rule out simpler explanations for the recovered signal, and to assess if it extends beyond the controlled setting.

Our main contributions are the following:

• We study the what-to-which problem and propose a controlled grounding protocol. We investigate whether token representations encode information that identifies a particular object instance beyond category-level object identity. To this end, our protocol rigorously separates instance-specific grounding from generic contextual, category-level, and supervision-induced signals.

• To address the what-to-which problem, we reframe it under the lens of Optimal Transport (OT). We propose OTTER, a lightweight supervised probe that aligns generated tokens with visual regions through entropy-regularized OT, while keeping the underlying MLLM frozen.

• Through controlled and out-of-distribution experiments, we show that instance-relevant visual information is recoverable from frozen MLLM representations and is distributed across contextualized token states. We demonstrate that this signal is particularly strong for spatial modifiers, but it is not exclusive to modifiers or object nouns.

The remainder of the paper is organized as follows. We outline the relevant literature that inspired and influenced our work in Section 2. Then, we describe the protocol used to extract and analyse token-visual mapping in Section 3. In Section 4, we describe the formulation of the OTTER module. Finally, we provide details on the experimental set up and results in Section 5 and 6.

## 2 RELATED WORK

Token-level visual grounding for MLLMs. Recent work has extended visual grounding from image classifiers to autoregressive multimodal generation. Token Activation Maps (TAM) (Li et al., 2025) summarize contextual interference from preceding generated tokens and produce token-level visual maps without additional localization supervision in a zero-shot manner. Other training-free or black-box approaches derive explanations from attention, activation maps, gradients, or input perturbations, such as LVLM-Interpret (Ben Melech Stan et al., 2024), LLaVA-CAM (Zhang et al., 2025), and EAGLE (Chen et al., 2026). These methods primarily evaluate generic word- or objectlevel grounding and faithfulness. OTTER addresses a complementary question: whether instancediscriminative modifiers, such as spatial terms and attributes, can be exploited as a distinct token category. Unlike the above methods, OTTER learns a lightweight grounding geometry from RIS supervision while keeping the underlying MLLM frozen.

Visual grounding from frozen MLLM features. A related line of work investigates whether visual grounding information can be decoded directly from frozen MLLM representations. Logit-lens-style approaches project internal representations to localize visually grounded concepts (nostalgebraist, 2020; Jiang et al., 2025; Neo et al., 2025). Differently, Phukan et al. (2025) use contextual midlayer embeddings to show that internal representations contain grounding information beyond object nouns, at the answer level. While sharing this premise, OTTER goes further by decoding signals at a token level. In fact, it learns a structured token–patch geometry yielding compact, token-specific grounding maps which disentangle modifiers from head nouns and validate the functional relevance. Supervised visual grounding and referring segmentation. Grounding-oriented MLLMs—e.g., Shikra (Chen et al., 2023), Kosmos-2 (Peng et al., 2024), LLaVA-Grounding (Zhang et al., 2024), and Groma (Ma et al., 2024)—train the model to predict localized outputs such as boxes or masks. RIS approaches, instead, use referring-expression comprehension and segmentation to localize the object specified by a natural-language expression (Kazemzadeh et al., 2014; Mao et al., 2016; Yu et al., 2016). OTTER is neither a grounding fine-tuning method nor a RIS segmenter. In fact, it leverages a frozen MLLM and uses RIS masks only as supervision for learning a modifier grounding probe over its existing token and visual representations.

Optimal Transport for token–visual alignment. Optimal Transport addresses the problem of moving mass between two distributions at minimal cost. In particular, Kantorovich’s formulation (Kantorovich, 1942) seeks an optimal transport plan: a coupling that specifies how mass from one distribution is matched to the other, with the resulting optimal cost defining the Wasserstein distance. OTTER leverages an entropy-regularized relaxation of this problem given by the Sinkhorn distance (Cuturi, 2013), aligning the mass of tokens with that of visual patches. The regularization introduces the flexibility to assign non-groundable words or irrelevant patches to so-called dustbins.

## 3 EVALUATION PROTOCOL AND CONTROLS

To ground the what-to-which problem, we develop a comprehensive protocol structured as seven guiding questions. We first establish controlled instance-level grounding and analyze where the corresponding signal is represented (Q1–Q3). We then test the role of the learned token–visual mapping and the relevance of the recovered regions to generation (Q4–Q5), and finally assess whether the learned grounding generalizes to free-generation and across datasets (Q6–Q7).

Q1. Does the full phrase improve instance-level grounding over the object noun alone? Freegeneration grounding does not establish whether the linguistic information that specifies an instance improves grounding beyond the object category itself. We therefore use RefCOCO as a controlled setting and prompt the MLLM to generate either the full referring phrase, including attributes and modifiers, or only the object category noun. The resulting maps are compared with the ground-truth mask using pointing, Energy-in-GT, Soft-IoU, thresholded IoU, and entropy.

Q2. Which token types carry instance-discriminative grounding information? A phrase-level advantage does not reveal which linguistic components carry the instance-discriminative signal. We therefore introduce RefCOCO-MOD, a token-level partition of the RefCOCO validation set that groups generated tokens into spatial modifiers, attributes, relational/action terms, object nouns, and function words. We run TAM and OTTER on the same generated phrases and tokenization, and each token-level map is compared with the referred-instance mask using Soft-IoU.

Q3. Is referential information modifier-specific, or distributed across contextualized tokens? We test whether instance-specific grounding is confined to explicit modifiers or shared across representations. First, using RefCOCO counterfactual pairs (images with distinct same-category instances and different referring expressions), we use target-versus-distractor margins and switch consistency to test if modifier and noun maps correctly align with their respective targets. For a stricter control, we isolate a left/right subset where paired expressions share the noun and differ solely in opposing spatial terms.

Next, we test visual dependence by extracting token representations under correct, blank, or unrelated images, while grounding them against the original image patches. We also apply two text perturbations: scrambled (averaged over five random word-order permutations) and isolated (removing surrounding context). For token i under control condition c and in-context condition ctx, we define the retention ratio and its noun-normalized group version as

$$
r _ { i } ^ { ( c ) } = \frac { m _ { i } ^ { ( c ) } } { m _ { i } ^ { ( \mathrm { c t x ) } } } \quad \mathrm { a n d } \quad R _ { g } ^ { ( c ) } = \frac { \mathrm { m e d i a n } _ { i \in g } r _ { i } ^ { ( c ) } } { \mathrm { m e d i a n } _ { i \in \mathrm { n o u n } } r _ { i } ^ { ( c ) } } ,\tag{1}
$$

where m is the grounding metric. $R _ { g } ^ { ( c ) } < 1$ indicates greater context-dependence than the object noun. Finally, a free-generation control evaluates cases where the object noun is generated before its spatial modifier. Due to autoregressive causal masking, this tests whether instance-specific information is already present in the noun state before the explicit modifier is produced.

Q4. Does grounding depend on the learned token–visual mapping? Good localization performance alone does not clarify whether OTTER is using information carried by the textual representations rather than supervision alone to determine where each token is grounded. We therefore compare OTTER with two matched controls that preserve the same implementation while altering only the semantic mapping between text and visual features. Specifically, we use raw cosine with the original Qwen representations, and shuffled projection replacing the learned projection with a fixed random channel permutation.

Q5. Are the recovered regions relevant to the model’s own generation? Overlap with a groundtruth mask does not by itself show that the highlighted regions matter to the model’s generation. We therefore perform deletion and insertion perturbations, masking or revealing image regions according to the grounding map and measuring the resulting change in token log-likelihood. We compare OTTER with random, center, and attention baselines. Stronger deletion effects and faster insertion recovery indicate greater relevance to the model’s prediction.

Q6. Does OTTER generalize to token-level grounding under free generation? We test whether OTTER generalizes beyond RefCOCO-derived supervision to unconstrained free generation on COCO Caption and OpenPSG. Following the TAM protocol, we use the frozen Qwen2.5-VL backbone and evaluate both a fixed (0.5) and a per-image Otsu threshold, reporting Obj-IoU, Func-IoU, F1-IoU, Precision, and Recall.

Q7. Does the learned grounding geometry transfer across datasets? Generalization under free captioning does not establish that the learned probe transfers to a different referring-expression distribution. We therefore apply a trained OTTER checkpoint to RefCOCOg without retraining, where expressions are longer and more descriptive than in RefCOCO. We evaluate grounding against the referred-instance masks both at the phrase level and separately for nouns, attributes, spatial modifiers, and relational/action terms. This complements Q6 by testing cross-dataset transfer within the referring-expression family rather than unconstrained free-generation generalization.

## 4 METHOD

OTTER is a lightweight head module that exploits features from the MLLM to provide token-region attribution for the generated modifiers. Starting from the contextual token states and visual patch states produced by a frozen MLLM, we train OTTER to learn an optimal projection of both representations into a shared embedding space under a supervised segmentation objective. To this purpose, we build a token-patch cost matrix accounting for semantic, attention and spatial features, and solve an entropy-regularized OT problem whose optimal plan is aggregated into a patch-level grounding map. In what follows, we formalize the details of how OTTER learns from the frozen MLLM’s internal representations to produce an optimal token-patch grounding map. See Algorithm 1 and Figure 1 for algorithmic and graphical descriptions of the method.

## 4.1 PROBLEM SETUP

The frozen MLLM receives an image I and a prompt q as input, and generates a sequence of tokens

$$
y _ { 1 : T } = ( y _ { 1 } , \dots , y _ { T } ) .\tag{2}
$$

Additionally, the MLLM provides a contextual hidden state $h _ { i } \in \mathbb { R } ^ { d _ { t } }$ for each generated token $y _ { i }$ and a set of visual patch states $V = \{ v _ { j } \} _ { j = 1 } ^ { N }$ for each reference image, where $v _ { j } \in \mathbb { R } ^ { d _ { v } }$ . Our goal is to produce a token-level grounding map

$$
\boldsymbol { E } _ { i , j } \in [ 0 , 1 ] ^ { T \times N }\tag{3}
$$

where each $E _ { i , j }$ measures how much patch j supports token i. For words composed of several tokens, we aggregate the token maps by a max aggregation, as in Li et al. (2025).

## 4.2 OTTER HEAD TRAINING

OTTER is a lightweight head trained on the frozen MLLM internal representations for RIS, specifically the contextual token states $H = \{ h _ { i } \} _ { i = 1 } ^ { T }$ and the visual states $\dot { V } = \{ v _ { j } \} _ { j = 1 } ^ { N }$ . Features are extracted from an intermediate layer of the MLLM selected via validation grounding diagnostics (see Appendix A, Table 8 and 9). The trainable head—comprising ≈550k learnable parameters compared to the billions of the frozen backbone—computes a token projection ϕ<sub>t</sub>, a visual projection $\phi _ { v }$ , and a token scoring module $s ( h _ { i } )$ that produces source masses as

$$
\alpha _ { i } = \mathrm { s o f t m a x } _ { i } s ( h _ { i } ) .\tag{4}
$$

Specifically, these functions are implemented as linear layers with normalization. During training we use RIS triples $( I , R , M )$ , where R is a referring expression and $M \in \{ 0 , 1 \} ^ { N }$ is the target mask downsampled to the patch grid. The trainable probe consists of two lightweight projections

$$
z _ { i } ^ { t } = \frac { \phi _ { t } ( h _ { i } ) } { \lVert \phi _ { t } ( h _ { i } ) \rVert _ { 2 } } \quad \mathrm { a n d } \quad z _ { j } ^ { v } = \frac { \phi _ { v } ( v _ { j } ) } { \lVert \phi _ { v } ( v _ { j } ) \rVert _ { 2 } }\tag{5}
$$

which define a shared semantic space. Thus we compute the cost between token i and patch j as

$$
C _ { i j } ^ { \mathrm { s e m } } = 1 - \langle z _ { i } ^ { t } , z _ { j } ^ { v } \rangle .\tag{6}
$$

When the selected head uses attention and spatial terms, the implementation normalizes the attention cost per token and the spatial cost by its maximum value:

$$
C _ { i j } ^ { \mathrm { a t t } } = \frac { - \log \tilde { A } _ { i j } } { \frac { 1 } { N } \sum _ { k } - \log \tilde { A } _ { i k } } , \qquad \tilde { A } _ { i j } = \frac { A _ { i j } } { \sum _ { k } A _ { i k } } ,\tag{7}
$$

$$
C _ { i j } ^ { \mathrm { s p a } } = \frac { \| p _ { j } - \mu _ { i } \| _ { 2 } ^ { 2 } } { \operatorname* { m a x } _ { i , j } \| p _ { j } - \mu _ { i } \| _ { 2 } ^ { 2 } } , \qquad \mu _ { i } = \sum _ { j } \tilde { A } _ { i j } p _ { j } ,\tag{8}
$$

where $p _ { j }$ is the patch coordinate. Therefore, the full non-gated cost is

$$
C _ { i j } = w _ { s } C _ { i j } ^ { \mathrm { s e m } } + w _ { a } g _ { i } ^ { a } C _ { i j } ^ { \mathrm { a t t } } + w _ { p } g _ { i } ^ { p } C _ { i j } ^ { \mathrm { s p a } }\tag{9}
$$

where $w _ { s } , w _ { a } , w _ { p } \in [ 0 , 1 ]$ are learnable weights and $g _ { i } ^ { a } , g _ { i } ^ { p } \in [ 0 , 1 ]$ are optional token-specific gates that can be learnt to modulate the attention and spatial terms. In Appendix A, we present an architecture ablation by showing the results of different cost configurations. Specifically, we include/exclude semantic, spatial and attention costs, as well as fix/learn adaptive-gated heads, weights and dustbins.

## 4.3 TOKEN–PATCH OPTIMAL TRANSPORT

From the token–patch cost matrix C, OTTER solves an entropy-regularized optimal transport problem. To define the OT marginals a and b, we first compute scores for both text tokens and visual patches using Eq. 4. We then augment both marginals with dustbin entries, allowing transport mass to be assigned to non-groundable text tokens and irrelevant visual patches. Both marginals are rescaled to have the same total mass before solving OT. Let $a \in \Delta ^ { T + 1 }$ and $b \in \Delta ^ { N + 1 }$ denote the resulting source and target marginals, then the optimal plan is defined as

$$
\Pi ^ { \star } = \arg \operatorname* { m i n } _ { \Pi \in \mathcal { U } ( a , b ) } \langle \Pi , C \rangle + \varepsilon \sum _ { i j } \Pi _ { i j } ( \log \Pi _ { i j } - 1 ) ,\tag{10}
$$

where $\textstyle { \mathcal { U } } ( a , b )$ is the set of couplings with marginals $a , b$ and $\varepsilon { = } 0 . 0 5$ is the Sinkhorn temperature. We solve the resulting problem with 80 log-domain Sinkhorn iterations. The patch evidence is obtained by summing the transported mass from text tokens to visual patches:

$$
O _ { j } = \sum _ { i = 1 } ^ { T } \Pi _ { i j } ^ { \star } , \qquad j = 1 , \ldots , N .\tag{11}
$$

During head training, the transported patch mass is converted into the grid score map by multiplying by the number of visual patches and clamping to [0, 1]. Official threshold metrics resize this raw score map to image resolution without per-example min–max normalization. We remark that, since the backbone is frozen, the probe learns how to decode grounding evidence from existing MLLM representations rather than modifying the model’s generation behaviour.

Residual Semantic Expansion. Raw OT is precise in pointing but lacks coverage, whereas the semantic branch improves recall at the cost of being diffuse. We combine both these approaches in the OTTER-Residual configuration:

$$
E _ { i } ^ { \mathrm { r e s } } = O _ { i } + \beta S _ { t } ( 1 - O _ { i } ) ,\tag{12}
$$

where $\beta \in [ 0 , 1 ]$ and $S _ { t }$ is the normalized learned semantic evidence obtained from $1 - C _ { i j } ^ { \mathrm { s e m } }$ Grounding Masks Refinement. We use a lightweight morphological post-processing refinement on thresholded maps: binary closing with 2 iterations, hole filling, and removal of connected components smaller than 0.05% of the image area. Ablation on other refinements is in Appendix ??.

## 4.4 TRAINING OBJECTIVE

For a predicted score map O and target mask M on the visual grid, the training objective is

$$
\mathcal { L } _ { \mathrm { m a s k } } = \lambda _ { D } \mathcal { L } _ { \mathrm { D i c e } } ( O , M ) + \lambda _ { B } \mathcal { L } _ { \mathrm { F o c a l B C E } } ( O , M ) + \lambda _ { T V } \mathcal { L } _ { \mathrm { T V } } ( O ) .\tag{13}
$$

Here, $\mathcal { L } _ { \mathrm { D i c e } }$ promotes overlap between the predicted and target masks (Milletari et al., 2016), $\mathcal { L } _ { \mathrm { F o c a l B C E } }$ down-weights easy foreground/background predictions (Lin et al., 2017), and ${ \mathcal { L } } _ { \mathrm { T V } }$ regularizes local spatial variation (Rudin et al., 1992).

![](images/9363cf02edf09f1b117fd7d261edf2ff669230cca8abf3ead8c4af905d5e7624.jpg)  
Figure 1: Method Overview. The frozen MLLM produces token and visual patch states, which the trainable OTTER probe projects into a shared space to build a token–patch cost matrix. An entropyregularized OT solver aligns tokens and visual patches, yielding a token-level grounding map.

## 5 EXPERIMENTAL SETUP

Methods. Our probing module is trained on hidden states and visual patch states extracted from a frozen Qwen2.5-VL model (Bai et al., 2025). We additionally screen OTTER on three further backbones (Qwen3-VL-2B, InternVL3.5-2B, Gemma-3-4B), and the layer-selection protocol and results are reported in Appendix A. Unless otherwise stated, OTTER denotes the adaptive-gated OT head with learned global cost weights and learned text/visual dustbin costs. Each OTTER’s head is optimized with Eq.(13), with $\lambda _ { D } = 1 . 0 , \lambda _ { B } = 0 . 5$ and $\lambda _ { T V } = 0 . 0 2$ . Optimization uses AdamW with learning rate $2 \cdot 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , and gradient clipping at 1.0. Early stopping is decided based on the calibration set mIoU. We calculate the mIoU for both fixed (0.50), and dynamic thresholds (selected on the calibration split). We evaluate OTTER against the learned semantic branch, attention maps, center priors, and random maps. Ablations on the full head-family, training-scale, and transport-operator are reported in Appendix A.

Datasets and evaluation settings. OTTER is trained on 2,500 images from RefCOCO (Yu et al., 2016), providing the referring expressions and instance masks, calibration, and controlled evaluation. RefCOCO-MOD (see Section 3) contains 5,021 spatial, 2,722 attribute, and 516 relational/action tokens, together with 15,881 noun and 5, 017 function-word tokens used as controls. COCO Caption (Chen et al., 2015) (5k images) and OpenPSG (Zhou et al., 2024) (3,176 images) are used for evaluation under free-generation. RefCOCOg is used only for cross-dataset transfer within the referring-expression family, where grounding is evaluated both at the phrase level and by generated token group. The context-control analysis uses the full RefCOCO validation split, with 10,834 expressions attempted and 8,407 retained after removing single-word and modifier-free cases.

Counterfactual referent controls. Starting from the RefCOCO validation expressions used in the contextual analysis, we construct 2,971 natural counterfactual pairs spanning 1,238 images. Each pair contains distinct referred instances within the same image and object category, together with referring expressions that provide a discriminative modifier contrast. We also define a stricter subset of 217 clean left/right pairs over 187 images, requiring distinct target annotations, a shared clean object noun, opposing left/leftmost versus right/rightmost expressions, and no attribute modifiers.

Algorithm 1 OTTER generated-token grounding   
Require: Image I, prompt q, frozen MLLM $f ,$ probe trained with Eq.(13), residual weight $\beta .$   
1: Generate tokens y<sub>1:T</sub> with $f ( I , q ) ;$ ; store hidden states $H = \{ h _ { i } \} _ { i = 1 } ^ { \hat { T } } ;$ extract visual patches $V = \{ v _ { j } \} _ { j = 1 } ^ { N }$   
2: for each generated token t do   
3: Project h<sub>i</sub> and $v _ { j }$ into the learned token–region space.   
4: Compute the token–patch costs in Eq.(9): semantic cost, (gated) attention and spatial costs.   
5: Add text and visual dustbins and solve the Sinkhorn OT problem in Eq.(10).   
6: Obtain raw OT map $O _ { i }$ from transported patch mass, rescaled by $N$ and clamped to [0, 1].   
7: Optionally compute residual map $\dot { E } _ { i } ^ { \mathrm { r e s } } = \dot { O } _ { i } + \beta S _ { t } \dot { ( 1 - O _ { i } ) }$   
8: end for   
9: return Token-level maps {O<sub>i</sub>} and optional residual maps $\{ E _ { i } ^ { \mathrm { r e s } } \}$

Visual and context perturbations. For each counterfactual pair, we evaluate noun and modifier maps under three visual conditions, extracting token hidden states after replacing the original (correct-) image with either a blank (blank-) or unrelated (wrong-) image, while OTTER grounds those states against the original image patches, so that any drop reflects lost target-specific information rather than a different grounding target. We report target preference, target-versus-distractor margin, and bidirectional switch consistency for each condition. We further perturb the referring context itself, via scrambled (five random word-order permutations) and isolated (context removed) variants, reporting the retention ratio (see Eq. (1)).

Faithfulness evaluation. The perturbation analysis uses generated phrases from the RefCOCO validation set images, expanded to the per-token unit level. Deletion and Insertion use a fixed gray occluder. Method differences are summarized with paired bootstrap confidence intervals using 1,000 resamples of the shared example IDs.

Transfer evaluation. RefCOCOg is evaluated without retraining on RefCOCOg data, using the cleanest transfer operating point (the fixed full-OT) checkpoint, rather than the adaptive-gated checkpoint used as the selected primary architecture elsewhere in the paper.

## 6 RESULTS

We organize the results following the evaluation protocol in Section 3. We first establish controlled instance-level grounding and examine where the corresponding signal is represented (Q1–Q3). We then test the role of the learned token–visual mapping and the relevance of the recovered regions to generation (Q4–Q5). Finally, we evaluate generalization to unconstrained free generation and cross-dataset transfer (Q6–Q7).

## 6.1 CONTROLLED INSTANCE-LEVEL GROUNDING (Q1–Q2)

Phrase versus noun-only generation. Table 1 shows that grounding the full referring expression is more informative than grounding the object noun alone. This establishes that the linguistic context specifying an instance provides useful grounding information beyond the object category. Importantly, this phrase-level comparison does not establish that the additional information is localized specifically in modifier representations; we address that distinction at the token level in Q2–Q3.

Grounding across token categories. RefCOCO-MOD allows us to ask where instance-level visual information can be decoded within the generated expression. As shown in Table 2, the learned probe recovers grounding information across several linguistic roles, with particularly strong results for object nouns and spatial modifiers, followed by attributes. Relational/action terms provide a weaker but consistent signal. Function words also contain non-trivial target information under the supervised probe. We do not interpret this as independent lexical visual semantics; rather, it suggests that referential information may be distributed through contextualized token representations. This observation motivates Q3, where we test directly whether the referred instance is encoded specifically by modifiers or shared across tokens.

Table 1: Controlled RefCOCO grounding.
<table><tr><td>Protocol</td><td>Method</td><td>Energy ↑ Soft-IoU ↑ IoU@.4 ↑ Ent. ↓</td><td></td><td></td><td></td></tr><tr><td rowspan="3">Phrase</td><td>Semantic</td><td>0.170</td><td>0.156</td><td>0.177</td><td>0.993</td></tr><tr><td>OTTER</td><td>0.405</td><td>0.264</td><td>0.332</td><td>0.891</td></tr><tr><td>OTTER-Rβ=0.3</td><td>0.254</td><td>0.205</td><td>0.348</td><td>0.974</td></tr><tr><td rowspan="3">Noun-only</td><td>Semantic</td><td>0.162</td><td>0.147</td><td>0.168</td><td>0.993</td></tr><tr><td>OTTER</td><td>0.310</td><td>0.189</td><td>0.229</td><td>0.907</td></tr><tr><td>OTTER-R  $\beta { = } 0 . 4$ </td><td>0.205</td><td>0.166</td><td>0.264</td><td>0.983</td></tr></table>

Table 2: RefCOCO-MOD Soft-IoU comparison.
<table><tr><td>Token group</td><td colspan="3">n TAM OTTER ∆ [95% CI]</td></tr><tr><td>Phrase</td><td>10834 0.118</td><td>0.342 +0.224 </td><td>[0.221, 0.228]</td></tr><tr><td>Nouns</td><td>15881 0.106</td><td>0.310+0.204</td><td>[0.201, 0.207]</td></tr><tr><td>Spatial</td><td>5021 0.061</td><td>0.304+0.243</td><td>[0.238, 0.248]</td></tr><tr><td>Attributes</td><td>2722 0.119</td><td>0.285 +0.167</td><td>[0.161, 0.173]</td></tr><tr><td>Rel./action</td><td>516 0.077</td><td>0.172 +0.095</td><td>[0.082, 0.110]</td></tr><tr><td>Function</td><td>5017 0.060</td><td>0.244 +0.184 [0.179, 0.189]</td><td></td></tr></table>

## 6.2 DISTRIBUTED REFERENTIAL GROUNDING AND CONTEXT DEPENDENCE (Q3)

Q1–Q2 show that modifier states contain instance-level visual information, but not whether this information is specific to modifiers. We therefore test how referential information is distributed across contextualized tokens.

Counterfactual referent switching. On natural same-image counterfactual pairs, modifier maps reliably follow the intended same-category referent, but contextualized noun maps do so just as reliably and can be even more selective (Table 3, top). The same pattern holds in the stricter left/right subset. Thus, the linguistic distinction between what and which does not correspond to a strict noun–modifier separation in the internal representations: instance identity is distributed across contextualized token states. Decomposing by modifier type (Table 3, bottom), this pattern is clearest for spatial expressions; attribute contrasts are broadly consistent, while relational/action pairs are too sparse for a reliable conclusion.

Table 3: Counterfactual referent grounding.
<table><tr><td></td><td></td><td></td><td colspan="2">Target pref. ↑</td><td colspan="2">Gap ↑</td></tr><tr><td>Group</td><td>Condition</td><td>Pairs</td><td>Mod.</td><td>Noun</td><td>Mod.</td><td>Noun</td></tr><tr><td rowspan="3">All natural</td><td>Correct</td><td>2971</td><td>0.776</td><td>0.779</td><td>0.203</td><td>0.242</td></tr><tr><td>Blank</td><td>2971</td><td>0.703</td><td>0.705</td><td>0.112</td><td>0.137</td></tr><tr><td>Wrong</td><td>2971</td><td>0.714</td><td>0.717</td><td>0.121</td><td>0.146</td></tr><tr><td rowspan="3">Clean L/R</td><td>Correct</td><td>217</td><td>0.772</td><td>0.795</td><td>0.220</td><td>0.293</td></tr><tr><td>Blank</td><td>217</td><td>0.677</td><td>0.744</td><td>0.115</td><td>0.193</td></tr><tr><td>Wrong</td><td>217</td><td>0.705</td><td>0.749</td><td>0.131</td><td>0.204</td></tr><tr><td rowspan="3">By type</td><td>Spatial</td><td>2578</td><td>0.775</td><td>0.778</td><td>0.202</td><td>0.237</td></tr><tr><td>Attribute</td><td>173</td><td>0.786</td><td>0.769</td><td>0.197</td><td>0.244</td></tr><tr><td>Rel./act.</td><td></td><td>3 0.833</td><td>0.833</td><td>0.306</td><td>0.219</td></tr></table>

Table 4: Deletion AUC by token group.
<table><tr><td>Method</td><td>Phrase</td><td>Noun</td><td>Mod.</td><td>Spatial</td><td>Attr.</td></tr><tr><td>Random</td><td>0.310</td><td>0.305</td><td>0.182</td><td>0.151</td><td>0.226</td></tr><tr><td>Attention</td><td>0.386</td><td>0.383</td><td>0.202</td><td>0.147</td><td>0.281</td></tr><tr><td>Center</td><td>0.317</td><td>0.318</td><td>0.194</td><td>0.147</td><td>0.266</td></tr><tr><td>Semantic</td><td>0.541</td><td>0.536</td><td>0.320</td><td>0.227</td><td>0.450</td></tr><tr><td>OTTER</td><td>0.543</td><td>0.535</td><td>0.322</td><td>0.234</td><td>0.445</td></tr><tr><td>OTTER-R</td><td>0.546</td><td>0.545</td><td>0.324</td><td>0.231</td><td>0.456</td></tr></table>

Visual and contextual dependence. Replacing the correct image with a blank or unrelated one weakens referential selectivity for both nouns and modifiers, showing that the shared signal is not purely linguistic. Context perturbations further reveal that this distribution is not uniform: functionword grounding depends strongly on the surrounding expression, whereas spatial modifiers remain comparatively robust (Table 5, left). Referential information is therefore shared across tokens, but different token classes depend on context to different degrees.

Referential information before the modifier. In free generation we isolate expressions where the noun is produced before the disambiguating spatial modifier. Even in this causal setting, noun states already contain significant information about the intended instance, while the subsequently generated modifier is more discriminative. This suggests that the what-to-which transition can begin before the explicit modifier is generated and become sharper as the referring expression unfolds.

## 6.3 MATCHED PROBE CONTROLS (Q4)

The matched controls in Table 5, right, produce a substantial performance collapse when the learned token–visual projection is replaced by either raw feature similarity or a shuffled mapping. Thus, the observed grounding cannot be attributed simply to RIS supervision, the OT operator, or an immediately accessible cosine geometry in the frozen features. Instead, the results support a more specific claim: frozen Qwen representations contain referential information that can be recovered by a lightweight learned projection. Since that projection is itself trained with RIS supervision, we do not interpret this as evidence for a fully pre-existing grounding geometry.

Table 5: Context dependence and matched probe controls. (a) normalized Energy-in-GT retention under context perturbation; (b) matched projection controls on RefCOCO-val under the same OT/RIS training.  
(a) Context perturbation
<table><tr><td>Token group</td><td>Scrambled</td><td>Isolated</td></tr><tr><td>Nouns</td><td>1.000 [0.99, 1.01]</td><td>1.000 [0.98, 1.02]</td></tr><tr><td>Spatial</td><td>0.982 [0.97, 0.99]</td><td>1.077 [1.05, 1.10]</td></tr><tr><td>Attribute</td><td>0.942 [0.93, 0.96]</td><td>0.914 [0.87, 0.96]</td></tr><tr><td>Function</td><td>0.939 [0.93, 0.95]</td><td>0.597 [0.57, 0.63]</td></tr></table>

(b) Projection controls
<table><tr><td>Probe</td><td>Learned projection?</td><td>Calibrated mIoU ↑</td></tr><tr><td>OTTER</td><td>yes</td><td>0.458</td></tr><tr><td>Raw cosine</td><td>no</td><td>0.130</td></tr><tr><td>Shuffled projection</td><td>no</td><td>0.135</td></tr></table>

## 6.4 MODEL RELEVANCE UNDER PERTURBATION (Q5)

Deletion experiments provide complementary evidence that the recovered regions are relevant to the model’s own token generation. Across token groups, removing regions prioritized by OTTER (mainly in the Residual counterpart) has a stronger effect than removing random, center-selected, or attention-selected regions (Table 4). The learned semantic branch behaves similarly under this intervention, indicating that the main distinction introduced by OT lies in the spatial structure and concentration of the recovered maps rather than in their perturbation sensitivity. We therefore interpret deletion as evidence of model relevance, rather than as a complete causal account of token generation.

guy  
cat  
bottom  
top  
Table 6: Free-generation grounding on COCO Caption and OpenPSG. Best in bold; second-best underlined.
<table><tr><td>Data</td><td>Method</td><td>Thr. Obj. ↑ Func. F1 ↑</td><td></td><td></td><td>Prec. ↑ Rec. ↑</td><td></td></tr><tr><td rowspan="4">COCO</td><td>OTTER</td><td></td><td>Otsu 0.193 0.929 0.320</td><td></td><td>0.471</td><td>0.290</td></tr><tr><td>OTTER-R (β=0.5)</td><td>0.5 0.273</td><td></td><td>0.8640.415</td><td>0.424</td><td>0.504</td></tr><tr><td>TAM</td><td>Otsu 0.278</td><td></td><td>0.7780.409</td><td>0.404</td><td>0.668</td></tr><tr><td>TAM</td><td>0.5 0.255</td><td></td><td>0.8690.395</td><td>0.450</td><td>0.539</td></tr><tr><td rowspan="4">OpenPSG</td><td>OTTER</td><td></td><td></td><td>Otsu0.2100.9340.343</td><td>0.479</td><td>0.300</td></tr><tr><td>OTTER-R (β=0.5)</td><td>0.5</td><td>0.2840.8630.427</td><td></td><td>0.425</td><td>0.524</td></tr><tr><td>TAM</td><td></td><td>Otsu0.2580.784 0.388</td><td></td><td>0.385</td><td>0.644</td></tr><tr><td>TAM</td><td>0.5</td><td></td><td>0.2350.873 0.370</td><td>0.428</td><td>0.515</td></tr></table>

Table 7: RefCOCOg transfer without retraining. Thresholded mIoU. Best in bold.
<table><tr><td colspan="3">Evaluation group TAM OTTER</td></tr><tr><td>Phrase</td><td>0.171</td><td>0.372</td></tr><tr><td>Nouns</td><td>0.134</td><td>0.332</td></tr><tr><td>Attributes</td><td>0.152</td><td>0.328</td></tr><tr><td>Spatial modifiers</td><td>0.053</td><td>0.328</td></tr><tr><td>Rel./action</td><td>0.077</td><td>0.208</td></tr></table>

## 6.5 GENERALIZATION BEYOND THE CONTROLLED SETTING (Q6–Q7)

Free generation (Q6). Table 6 evaluates OTTER on unconstrained generated text grounding on COCO Caption and OpenPSG. Because TAM is zero-shot whereas OTTER is RIS-supervised, we do not interpret the comparison as an intrinsic ranking of attribution methods. Rather, the results show that the grounding learned from RefCOCO-derived supervision remains useful when the model generates unconstrained captions outside the controlled referring-expression setup. Figure 2 illustrates this qualitatively: for four free-generation examples, OTTER produces more concentrated, token-aligned maps than TAM, attention, the semantic branch, and the center/random baselines.

Cross-dataset transfer (Q7). Table 7 shows that the learned token–region mapping remains effective when transferred to RefCOCOg without retraining. The improvement extends beyond phraselevel grounding to nouns, attributes, spatial expressions, and relational/action terms. Together, Q6 and Q7 indicate that OTTER is not restricted to the specific RefCOCO expression distribution used for probe supervision: the learned grounding geometry transfers both to unconstrained generation and to a different referring-expression dataset.

Girl in purple  
girl  
in  
Bottom cat  
purple  
Guy standing playing Wii Fit  
Cat on top  
cat  
![](images/3dcb1e07c5cde2ea2b3b3dba4fd93c2b950519af2abce494077dab0a0224930d.jpg)  
Figure 2: Qualitative comparison on token-level grounding. We compare OTTER against TAM, attention, semantic similarity, center, and random baselines.

## 7 CONCLUSION

We studied the what-to-which problem in frozen MLLMs: whether their representations encode not only what visual entity a generated expression describes, but also which instance of that entity is intended. Using controlled referring expressions, counterfactual same-category pairs, context perturbations, and free-generation settings, we find that instance-level grounding is not confined to a single modifier token. Instead, it is distributed across contextualized token representations: object nouns already carry substantial information about the intended instance, while spatial modifiers typically provide the strongest additional discrimination. This information depends on both the visual input and the referring context, and remains detectable when the expression is generated before the disambiguating modifier is produced. To analyze this phenomenon, we introduced OTTER, a lightweight supervised probe that learns a token–region geometry over frozen MLLM representations. The matched controls and perturbation experiments show that the recovered maps capture more than raw feature similarity and identify regions relevant to the model’s generation. Moreover, the learned geometry remains useful under unconstrained captioning and transfers to RefCOCOg without retraining. These results suggest that the transition from what to which is better understood as a process of progressively refined, distributed instance grounding across an expression, rather than as information introduced only by an isolated modifier. While our study covers a limited set of backbones, datasets, and grounding settings, it provides a framework and empirical basis for investigating this phenomenon across broader architectures and domains.

## REFERENCES

Shuai Bai et al. Qwen2.5-vl technical report, 2025. arXiv:2502.13923.

Gabriela Ben Melech Stan, Estelle Aflalo, Raanan Yehezkel Rohekar, Anahita Bhiwandiwalla, Shao-Yen Tseng, Matthew Lyle Olson, Yaniv Gurwicz, Chenfei Wu, Nan Duan, and Vasudev Lal. Lvlm-intrepret: An interpretability tool for large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 8182–8187, June 2024.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic, 2023. arXiv:2306.15195.

Ruoyu Chen, Xiaoqing Guo, Kangwei Liu, Siyuan Liang, Shiming Liu, Qunli Zhang, Laiyuan Wang, Hua Zhang, and Xiaochun Cao. Where mllms attend and what they rely on: Explaining autoregressive token generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17057–17066, 2026.

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and´ C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, 2013.

Sarthak Jain and Byron C Wallace. Attention is not explanation. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3543–3556, 2019.

Nick Jiang, Anish Kachinthaya, Suzanne Petryk, and Yossi Gandelsman. Interpreting and editing vision-language representations to mitigate hallucinations. In International Conference on Learn ing Representations, volume 2025, pp. 63582–63605, 2025.

Leonid V Kantorovich. On the translocation of masses. In Dokl. akad. nauk. ussr (ns), volume 37, pp. 199–201, 1942.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014.

Yi Li, Hualiang Wang, Xinpeng Ding, Haonan Wang, and Xiaomeng Li. Token activation map to visually explain multimodal llms. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 48–58. IEEE, 2025.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object´ detection. In Proceedings ofthe IEEE International Conference on Computer Vision (ICCV), pp. 2980–2988, 2017. doi: 10.1109/ICCV.2017.324.

Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. In ECCV, 2024.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016.

Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 Fourth International Conference on 3D Vision (3DV), pp. 565–571. IEEE, 2016. doi: 10.1109/3DV.2016.79.

Clement Neo et al. Towards interpreting visual information processing in vision-language models. In ICLR, 2025.

nostalgebraist. Interpreting gpt: The logit lens, 2020. LessWrong.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding multimodal large language models to the world. In International Conference on Learning Representations, volume 2024, pp. 51575–51598, 2024.

Anirudh Phukan et al. Beyond logit lens: Contextual embeddings for robust hallucination detection and grounding in vlms. In NAACL, 2025.

Leonid I. Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1–4):259–268, 1992. doi: 10.1016/ 0167-2789(92)90242-F.

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, 2017.

Sofia Serrano and Noah A Smith. Is attention interpretable? In Proceedings of the 57th annual meeting ofthe associationfor computational linguistics, pp. 2931–2951, 2019.

Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, and Xia Hu. Score-cam: Score-weighted visual explanations for convolutional neural networks. In CVPR Workshops, 2020.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions. In European conference on computer vision, pp. 69–85. Springer, 2016.

Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Leizhang, Chunyuan Li, et al. Llava-grounding: Grounded visual chat with large multimodal models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024.

Shilong Zhang et al. Gpt4roi: Instruction tuning large language model on region-of-interest, 2023. arXiv:2307.03601.

Xiaofeng Zhang, Yihao Quan, Chen Shen, Xiaosong Yuan, Shaotian Yan, Liang Xie, Wenxiao Wang, Chaochen Gu, Hao Tang, and Jieping Ye. From redundancy to relevance: Information flow in lvlms across reasoning tasks. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 2289–2299, 2025.

Zijian Zhou, Zheng Zhu, Holger Caesar, and Miaojing Shi. Openpsg: Open-set panoptic scene graph generation via large multimodal models. In European Conference on Computer Vision, pp. 199–215. Springer, 2024.

## A ADDITIONAL ARCHITECTURE ABLATION DETAILS

## A.1 LAYER AND TRAINING-SCALE ABLATIONS

Motivation. OTTER reads localization evidence from hidden states of a frozen MLLM. Since these representations are not equally spatial at all depths, we first screen the feature layer used by the segmentation head before comparing architectural variants. This control is especially important in the cross-backbone setting: a poor layer choice can make a backbone appear weak even when its intermediate states contain usable grounding information. We therefore separate layer selection from the final configuration ablation.

Cross-backbone layer screening. For each backbone, we freeze the MLLM and train only the OTTER head with the minimal ot semantic only configuration. We use RefCOCO, the training split for both head training and calibration with disjoint image ranges, and the validation split for evaluation. The layer-screening protocol uses 500 training images, 300 calibration images, and 1000 validation images; it keeps the first expression per mask, trains for up to 5 epochs with early stopping, and evaluates calibrated mIoU on the validation subset. Candidate layers are chosen at approximately 25%, 50%, 75%, and 90% of the language depth, plus the last two layers. Feature extraction uses sharded caches of 64 examples.

Table 8: Cross-backbone layer screening. Metrics are percentages. The selection score is the areamatched diagnostic IoU; calibrated mIoU is reported as the task metric. For each backbone, the selected layer is highlighted in bold.
<table><tr><td>Backbone (layer)</td><td>Diag. IoU</td><td>mIoU</td><td>P@0.5</td></tr><tr><td>Qwen2.5-VL-3B L9</td><td>28.75</td><td>27.51</td><td>14.14</td></tr><tr><td>Qwen2.5-VL-3B L18</td><td>38.45</td><td>35.23</td><td>29.69</td></tr><tr><td>Qwen2.5-VL-3B L27</td><td>44.22</td><td>41.06</td><td>40.92</td></tr><tr><td>Qwen2.5-VL-3B L32</td><td>41.39</td><td>38.20</td><td>34.67</td></tr><tr><td>Qwen2.5-VL-3B L35</td><td>28.76</td><td>28.56</td><td>11.88</td></tr><tr><td>Qwen2.5-VL-3B L36</td><td>27.30</td><td>27.10</td><td>9.65</td></tr><tr><td>Qwen3-VL-2B L7</td><td>30.06</td><td>29.55</td><td>14.73</td></tr><tr><td>Qwen3-VL-2B L14</td><td>46.34</td><td>42.41</td><td>41.73</td></tr><tr><td>Qwen3-VL-2B L21</td><td>42.20</td><td>37.40</td><td>31.87</td></tr><tr><td>Qwen3-VL-2B L25</td><td>35.33</td><td>32.85</td><td>20.11</td></tr><tr><td>Qwen3-VL-2B L27</td><td>26.82</td><td>26.57</td><td>8.08</td></tr><tr><td>Qwen3-VL-2B L28</td><td>26.35</td><td>26.10</td><td>7.56</td></tr><tr><td>InternVL3.5-2B L7</td><td>10.07</td><td>13.93</td><td>1.11</td></tr><tr><td>InternVL3.5-2B L14</td><td>11.51</td><td>15.63</td><td>1.62</td></tr><tr><td>InternVL3.5-2B L21</td><td>10.62</td><td>14.71</td><td>1.31</td></tr><tr><td>InternVL3.5-2B L25</td><td>9.97</td><td>14.04</td><td>1.03</td></tr><tr><td>InternVL3.5-2B L27</td><td>9.50</td><td>13.60</td><td>1.03</td></tr><tr><td>InternVL3.5-2B L28</td><td>9.70</td><td>13.46</td><td>0.91</td></tr><tr><td>Gemma-3-4B L8</td><td>10.44</td><td>13.96</td><td>0.71</td></tr><tr><td>Gemma-3-4B L17</td><td>10.33</td><td>13.65</td><td>0.83</td></tr><tr><td>Gemma-3-4B L26</td><td>9.87</td><td>13.26</td><td>0.67</td></tr><tr><td>Gemma-3-4B L31</td><td>9.90</td><td>13.31</td><td>0.99</td></tr><tr><td>Gemma-3-4B L33</td><td>9.90</td><td>13.71</td><td>1.03</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-3-4B L34</td><td>10.10</td><td>13.93</td><td>1.19</td></tr></table>

Mini token-type protocol for non-Qwen2.5 backbones. To avoid a full paper-scale rerun for every backbone, we use a small token-type check on RefCOCO modifier expressions. The test uses the best available head from the layer sweep for each backbone and evaluates refcoco mod token groups only. This is intended as a fast sanity check of token-level grounding (attributes, nouns, spatial terms, and relational/action terms), not as a final cross-backbone benchmark.

Training-scale and configuration ablation. After layer screening, we rerun the main Qwen2.5- VL-3B experiment at the selected layer. This experiment uses all RefCOCO validation expressions (10,707 expression–mask pairs), 300 calibration images, and training subsets of increasing size. Unlike the layer screen, this run uses example mode=all pairs, so each referring expression is an independent training/evaluation example. The OT head is trained for up to 20 epochs with early stopping on calibration mIoU, and thresholds are calibrated on {0.1, 0.2, 0.3, 0.4, 0.5}. We compare softmax normalization against OT variants with the same semantic, attention, and spatial cost terms.

Table 9: Mini token-type grounding check on RefCOCO modifier expressions using sweep heads. Rows report per-token maps grouped by token type.
<table><tr><td>Backbone (layer)</td><td>Noun</td><td>Attribute</td><td>Spatial</td><td>Rel./act.</td><td>Token avg.</td></tr><tr><td>Qwen2.5-VL-3B L27</td><td>37.91</td><td>39.47</td><td>39.05</td><td>32.75</td><td>37.30</td></tr><tr><td>Qwen3-VL-2B L14</td><td>29.95</td><td>36.82</td><td>31.51</td><td>32.34</td><td>32.66</td></tr><tr><td>InternVL3.5-2B L14</td><td>12.19</td><td>12.29</td><td>12.82</td><td>10.23</td><td>11.88</td></tr><tr><td>Gemma-3-4B L17</td><td>12.08</td><td>10.93</td><td>10.48</td><td>9.45</td><td>10.74</td></tr></table>

Table 10: Combined architecture and training-scale ablation at Qwen2.5-VL-3B layer 27 on Ref-COCO validation. The left side reports the mIoU across increasing numbers of requested training images. The right side reports detailed metrics (mIoU, cIoU, P@0.5, Diag. IoU) for all configurations at the largest training scale (2500 images). The overall best configuration is highlighted in bold.
<table><tr><td></td><td colspan="6">mIoU at Training Scale (Images)</td><td colspan="4">Full Metrics at 2500 Images</td></tr><tr><td>Variant</td><td>50</td><td>100</td><td>300</td><td>600</td><td>900</td><td>1500</td><td>2000</td><td>mIoU</td><td>cIoU</td><td>P@0.5</td><td>Diag. IoU</td></tr><tr><td>softmax_semantic_only</td><td>21.83</td><td>23.81</td><td>31.31</td><td>37.51</td><td>40.36</td><td>42.09</td><td>42.96</td><td>43.74</td><td>42.81</td><td>45.79</td><td>48.65</td></tr><tr><td>softmax_no_ot</td><td>22.08</td><td>24.76</td><td>29.42</td><td>38.17</td><td>40.36</td><td>41.87</td><td>42.17</td><td>43.74</td><td>43.19</td><td>45.52</td><td>49.15</td></tr><tr><td>softmax_fixed_full</td><td>22.24</td><td>24.83</td><td>29.04</td><td>38.76</td><td>40.44</td><td>42.36</td><td>41.84</td><td>43.69</td><td>42.58</td><td>45.42</td><td>50.48</td></tr><tr><td>ot_semantic_only</td><td>21.02</td><td>24.53</td><td>34.47</td><td>39.69</td><td>40.50</td><td>43.26</td><td>44.74</td><td>45.66</td><td>46.08</td><td>49.74</td><td>49.57</td></tr><tr><td>ot_sem_spatial</td><td>20.84</td><td>24.11</td><td>34.12</td><td>39.63</td><td>41.70</td><td>42.92</td><td>44.63</td><td>45.12</td><td>45.41</td><td>47.97</td><td>49.67</td></tr><tr><td>ot_sem_att</td><td>21.81</td><td>25.00</td><td>33.94</td><td>39.40</td><td>41.41</td><td>43.29</td><td>44.41</td><td>45.17</td><td>45.43</td><td>49.41</td><td>51.20</td></tr><tr><td>ot_full</td><td>22.28</td><td>25.59</td><td>33.97</td><td>39.41</td><td>41.69</td><td>43.10</td><td>44.32</td><td>44.94</td><td>45.25</td><td>49.11</td><td>51.26</td></tr><tr><td>ot_full_fixed_dustbins</td><td>22.59</td><td>26.45</td><td>34.46</td><td>39.59</td><td>41.78</td><td>43.10</td><td>44.30</td><td>44.99</td><td>45.31</td><td>49.29</td><td>51.09</td></tr><tr><td>ot_full_learned_weights</td><td>22.85</td><td>26.02</td><td>33.92</td><td>39.52</td><td>41.79</td><td>43.60</td><td>44.85</td><td>45.44</td><td>45.55</td><td>49.14</td><td>51.30</td></tr><tr><td>ot_adaptive-gated</td><td>22.34</td><td>25.79</td><td>34.63</td><td>40.41</td><td>41.88</td><td>43.12</td><td>44.71</td><td>45.61</td><td>45.60</td><td>50.17</td><td>51.21</td></tr><tr><td>ot_adaptive_gated_learned_weights</td><td>22.27</td><td>24.99</td><td>33.95</td><td>39.80</td><td>41.68</td><td>43.21</td><td>44.73</td><td>45.83</td><td>46.05</td><td>50.66</td><td>51.32</td></tr></table>

Results. The layer screen shows that localization is strongly layer-dependent. For Qwen2.5-VL-3B, layer 27 (75% depth) is best among the screened layers, reaching 41.06 mIoU and 44.22 diagnostic IoU on the 1000-image screen. Qwen3-VL-2B peaks earlier, at layer 14 (50% depth), with 42.41 mIoU and 46.34 diagnostic IoU. Gemma-3-4B remains weak across screened depths: its best official calibrated mIoU is at layer 8.

The mini token-type protocol confirms that Qwen3-VL-2B transfers reasonably well to token-level modifier grounding: with the layer-14 semantic-only head it reaches a 32.66 average token-type soft-IoU, with especially strong attribute grounding (36.82). In contrast, InternVL3.5-2B and Gemma-3- 4B remain close to 12 and 11 average token-type soft-IoU, respectively, under the same conditions, suggesting that the current generic adapter/head combination does not yet expose comparably useful token-to-patch geometry for these backbones.

At the selected Qwen2.5-VL layer, OT-based heads consistently outperform matched softmax baselines at moderate and large training scales. At 600 training images, the best softmax variant reaches 38.76 mIoU, whereas adaptive-gated OT reaches 40.41 mIoU. The advantage grows with more supervision: at the largest requested scale, the best softmax heads reach 43.74 mIoU, while adaptivegated OT with learned weights reaches 45.83 mIoU. Semantic-only OT is already a strong lowcomplexity reference, reaching 45.66 mIoU at the largest scale, but the adaptive-gated learnedweight variant gives the best final calibrated mIoU. These results support two design choices used in the main system: selecting an intermediate representation layer by validation grounding diagnostics, and using OT rather than independent softmax normalization for map construction.

## A.2 WHY OPTIMAL TRANSPORT? A CONTROLLED OPERATOR ABLATION

The architecture ablation shows OT heads above softmax heads on segmentation mIoU, but mIoU alone does not explain why transport helps. We isolate the transport operator by holding the learned geometry and the cost terms fixed and switching the aggregation—Sinkhorn optimal transport with dustbins versus a softmax pooling—and we evaluate on the dense grounding metrics that matter for a token-level map: the target-vs-distractor gap, Soft-IoU, and map entropy, computed per token group on RefCOCO validation.

Two comparisons are controlled: at the semantic-only cost (ot semantic only vs. softmax semantic only) and at the full cost (ot full vs. softmax no ot). Both use the same cost terms and the same initial weights $( w _ { \mathrm { s e m } } \mathrm { = } 1 . 0 , w _ { \mathrm { a t t } } \mathrm { = } 0 . 5 , w _ { \mathrm { s p a } } \mathrm { = } 0 . 2 )$ . Two secondary differences accompany the operator and both cut in $\mathrm { O T } \mathbf { s }$ favor: the softmax head additionally learns its cost weights, whereas the OT head keeps them fixed, and dustbin abstention is available only under transport because it is structurally incompatible with a normalized softmax. Thus the softmax head is, if anything, the more flexible of the two on cost weighting.

Table 11: Controlled operator ablation with 95% paired bootstrap CIs (n per group). Concentration is reported as peak-to-mean ratio; higher is more concentrated. This non-saturating metric reveals the operator effect that normalized entropy hides. Effect of OT at full cost isolates the Sinkhorn operator; cost terms exploited shows the gain from adding attention+spatial to each operator. All intervals shown exclude zero. Gap is undefined for nouns because there are no same-category instance distractors.
<table><tr><td></td><td>Token group</td><td colspan="2">∆Soft-IoU</td><td colspan="2"> $\Delta \mathrm { g a p }$ </td><td colspan="2">∆peak/mean</td></tr><tr><td>Effect of OT</td><td>Spatial</td><td></td><td>+0.021 [.019, .023]</td><td></td><td>-0.018 [-.022, -.013]</td><td></td><td>+5.41 [5.21, 5.62]</td></tr><tr><td>at full cost</td><td>Attribute</td><td></td><td>+0.018 [.015, .021]</td><td></td><td>-0.028 [-.033, -.022]</td><td></td><td>+5.93 [5.65, 6.22]</td></tr><tr><td>(ot_ful1-sm_no_ot)</td><td>Noun</td><td></td><td>+0.026 [.025, .027]</td><td></td><td></td><td></td><td>+3.85 [3.75, 3.96]</td></tr><tr><td colspan="8">Cost terms exploited: gain from adding attention+spatial to the semantic cost</td></tr><tr><td>OT uses them</td><td>Spatial</td><td></td><td>+0.024 [.022, .026]</td><td></td><td>+0.031 [.027, .036]</td><td></td><td> $+ 0 . 7 7 \left[ . 6 4 , . 9 0 \right]$ </td></tr><tr><td> $( \cot \tt { \_ { \ / { G U I I } } } \mathrm { 1 - o t \it { \_ S e m } } )$ </td><td>Attribute</td><td></td><td>+0.022 [.019, .026]</td><td></td><td>+0.030 [.023, .036]</td><td></td><td> $- 1 . 3 2 [ - \dot { 1 } . 5 9 , - 1 . \dot { 0 } 6 ]$ </td></tr><tr><td>Šoftmax wastes them</td><td>Spatial</td><td></td><td>+0.001 [.000, .001</td><td></td><td>+0.004 [.004, .005]</td><td></td><td> $- 0 . 0 \dot { 6 } \left[ - . 0 8 , - . 0 4 \right]$ </td></tr><tr><td>(sm_no_ot-sm_sem)</td><td>Attribute</td><td></td><td>+0.002 [.002, .003]</td><td></td><td>+0.006 [.005, .007]</td><td></td><td> $- 0 . 2 2 \left[ - . 2 4 , - . 2 0 \right]$ </td></tr></table>

Optimal transport makes multi-term costs usable. The most decisive result is in the lower half of Table 11. Adding attention and spatial cost terms on top of the semantic cost improves the maps substantially under OT: gap +0.031 for spatial, +0.030 for attributes, and Soft-IoU about +0.02. Under softmax, the same additions barely move the metrics: gap +0.004 for spatial and +0.006 for attributes.

The paired bootstrap intervals do not overlap: OT exploits the extra cost terms roughly 5–7× as much as softmax on the gap, significant on every group and metric. This holds even though the softmax head is allowed to learn its cost weights while the OT head keeps them fixed. Given the freedom to reweight attention and spatial terms, softmax still does not use them, whereas fixed-weight OT does. The advantage is therefore not a weighting artifact; it is the assignment structure of transport. This is a structural reason to prefer OT: it integrates competing semantic, attention, and spatial signals through a global assignment, whereas independent softmax normalization dilutes them. It also explains why softmax no ot and softmax semantic only score almost identically despite different cost terms: the extra terms are present but unused.

At full cost, OT is more accurate and more compact. Isolating the operator at full cost, the top block of Table 11 shows that OT yields significantly higher Soft-IoU, about +0.02, and much more concentrated maps. The peak-to-mean ratio rises by +3.9 to +5.9, a 50–90% relative increase over softmax, with tight non-overlapping intervals. This concentration advantage is a property of the transport operator itself, not of the extra cost terms: it is large in the operator-only comparisons, but small or absent in the cost-term comparisons, where attention and spatial mainly improve gap and Soft-IoU.

We report peak-to-mean ratio because normalized entropy understates the same effect: it moves only about −0.02 on identical maps, saturating in this high-entropy regime. The one axis where softmax leads is the raw target-vs-distractor gap, with OT lower by 0.018 on spatial and 0.028 on attributes. Softmax places a slightly sharper single peak on the target but distributes the rest more diffusely, which is why its region overlap and concentration are worse. The complementary structural argument is the dustbin: OT can route non-groundable tokens to a sink and abstain, whereas softmax must place all mass on the image. We note that the full-cost OT head carries learned dustbins while the softmax head has none, so the accuracy and compactness advantages above bundle transport assignment with dustbin abstention rather than isolating transport alone. Since abstention is itself only available under transport, we treat the two as jointly constitutive of the operator rather than as separable confounds.