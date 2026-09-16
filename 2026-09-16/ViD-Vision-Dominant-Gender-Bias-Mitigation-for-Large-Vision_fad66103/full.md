# ViD: Vision-Dominant Gender Bias Mitigation for Large Vision-Language Models

Zhipeng Zhao<sup>1</sup>, Zhaoqiang Wei<sup>1</sup>, Peishun Liu<sup>1</sup>, Youwei Zhao<sup>1</sup>, Ruichun Tang<sup>1,\*</sup>

<sup>1</sup>Ocean University of China   
{zhaozhipeng, zhaoyouwei1435}@stu.ouc.edu.cn   
{weizhaoqiang, liups, tangruichun}@ouc.edu.cn <sup>\*</sup>Corresponding author

## Abstract

Gender bias in large vision-language models (LVLMs) undermines their fairness and reli ability, compromising output trustworthiness. Current mitigation methods rely on training phase adjustments or post-hoc calibration, but face limitations in dynamic visual bias mitiga tion. These include inability to capture real time visual-textual incongruence, dependence on predefined gender bias taxonomies, and de graded cross-modal alignment with emergent bias patterns. To address these challenges, we propose ViD, a causally-inspired framework that analyzes attention mechanisms across five distinct patterns, revealing confounding effects from strong language priors. ViD demon strates that visual-to-language cross-attention effectively suppresses bias while preserving general reasoning capabilities and text gener ation quality. ViD incorporates dual mecha nisms: backdoor adjustment counters strong language priors, while refined token selec tion in decoding layers optimizes processing. This enhances model robustness and inference efficiency. Our integrated approach sig nificantly mitigates gender bias across multi dimensional social attributes in LVLMs, im proving visual grounding and output fairness. Cross-benchmark validation shows ViD reduces gender bias by 14.7% on single-attribute evaluations (FACET) and achieves significant improvements on image captioning tasks (MS COCO), with gender bias score improving from 0.6708 to 0.9978 for LLaVA. Crucially, these improvements require no addi tional training overhead, making ViD a scal able and practical solution for bias mitigation in LVLMs.

## 1 Introduction

Large Vision-Language Models (LVLMs), exemplified by GPT-4V and LLaVA(Liu et al., 2023), have demonstrated breakthrough performance in cross-modal understanding tasks including image captioning(Ke et al., 2019), visual question answering(Goyal et al., 2017), and multimodal reasoning(Chang et al., 2022; Jin et al., 2024). These models establish complex associations between visual representations and textual semantics through large-scale cross-modal alignment. They achieve exceptional results on specialized benchmarks, including the POPE benchmark for object hallucination evaluation (Li et al., 2023) and the MMMU benchmark for multidisciplinary understanding (Yue et al., 2024). However, inherent biases in their decision-making mechanisms remain critical challenges affecting reliability.

![](images/3776853bb9d79b37bd8a4101aa9ad365ee84126a8a184f25674cedbb1dcfaccf.jpg)  
Figure 1: Motivating example and overview of ViD. A conventional LVLM, dominated by language priors, mislabels a female guard as male, while ViD re-weights visual-to-language attention to ground the answer in visual evidence.

However, as illustrated in Figure 1, LVLMs often exhibit deeply ingrained gender biases when generating textual descriptions. Specifically, these manifest as gender stereotypes in occupational associations (e.g., strongly associating guards with males) and racial biases in attribute inference (Zhao et al., 2018; Kirk et al., 2021). Such issues become particularly pronounced when processing visually ambiguous or polysemous inputs. The root cause traces back to spurious correlations learned during LLM pre-trainingwhere models establish erroneous associations between specific visual patterns and stereotypical textual descriptions through non-causal statistical dependencies.

To address this issue, existing debiasing approaches primarily employ post-hoc calibration (Kong et al., 2023; Yu and Ananiadou, 2024) and adversarial training (Bartl and Leavy, 2024; Yu and Ananiadou, 2024), which suppress biased representations through output-layer regularization. However, these methods exhibit two critical limitations: (1) They lack causal transparency in crossmodal representations. (2) Their static intervention strategies cannot adapt to visual-language interactions with multiple social attributes.

In this study, we propose ViD, a novel gender bias mitigation framework that integrates structured causal modeling with attention-based dynamic adaptation. Our approach introduces attention-oriented causal intervention through structured masking to enhance cross-modal attention interactions. Crucially, we develop a visual-to-language attention reweighting mechanism specifically designed to amplify discriminative visual features. This integrated solution effectively mitigates gender bias in multivariate social attribute representations while maintaining seamless compatibility with existing LVLM inference pipelines. As a training-free module with one-time layer calibration, ViD requires no backbone modifications and significantly enhances system scalability.

Through extensive experiments on benchmark gender bias evaluations, our approach demonstrates robust generalization in gender bias reduction across diverse LVLMs. Our principal contributions are:

• We propose a novel attention masking heuristic based on causal intuition that effectively mitigates LVLM biases during inference without fine-tuning.

• We develop an empirical framework that maps attention dynamics to bias reduction, providing practical guidance for intervention.

• We experimentally validate ViD’s efficacy not only against gender-occupation bias but also against compound biases arising from coupled social attributes.

## 2 Related Work

## 2.1 Generative Bias in Foundation Models

Generative bias in foundation models undermines AI fairness, with gender bias being a critical research frontier. Despite data cleaning, societal stereotypes inevitably propagate into model outputs (Bender et al., 2021), disproportionately affecting underrepresented groups. Gender bias manifests in coreference resolution (Zhao et al., 2018), cross-modal associations (Janghorbani and De Melo, 2023), and gender-exclusive lexicons (Bartl and Leavy, 2024). LLMs amplify these biases, associating feminine/masculine terminology with genderstereotyped professions (Gorti et al., 2024). This study addresses intersectional gender biases across occupations, skin tones, and hair features through causal interventions.

## 2.2 Interventional Methods for Foundation Models

Various methods have been proposed to mitigate gender biases, including dimensionality pruning (Janghorbani and De Melo, 2023), finetuning (Bartl and Leavy, 2024), few-shot prompting (Gorti et al., 2024), and post-hoc frameworks (Kong et al., 2023). Recent specialized approaches employ two-stage designs (Zhang et al., 2024), counterfactual analysis (Howard et al., 2025), comprehensive evaluations (Girrbach et al., 2025), encoder debiasing (Kavuri et al., 2025), and benchmarking (Xiao et al., 2025). However, these methods often require external resources, training, or have limited scalability to intersectional biases.

In contrast, ViD uses causal inference-based attention intervention to mitigate bias at inference time without training or external resources. It handles multidimensional social attributes while preserving reasoning capabilities through structured causal modeling and attention reweighting. Unlike previous approaches, ViD requires no external resources, counterfactual datasets, fine-tuning, or training, effectively mitigating intersectional gender biases while remaining invariant to data distributions and computational constraints.

## 3 Method

We introduce a Structural Causal Model (SCM) for LVLMs with backdoor-adjusted real-time interventions to mitigate gender bias during inference. We emphasize that this SCM is a conceptual heuristic for motivating our attention interventions rather than a formally identified causal model; its graph dependencies are postulated from domain knowledge of LVLM architectures, not derived through causal discovery.

![](images/a35f8e4a27684ff480534f6d31b7922150f3aead34ef5d9793076f8c23e26fa9.jpg)  
Figure 2: (a) In the LVLM inference mode, tokens are output for each timestep based on a selection rule applied to the attention-generated logits. (b) Our proposed method, ViD, along with a causal-based attention intervention mechanism, models the LVLM inference process. (c) ViD introduces five attention patterns: 1. Visual-only Self-Attention, 2. Language-only Self-Attention, 3. Visual-to-Language Cross-Attention, 4. Language-to-Visual Cross-Attention, and 5. Bidirectional Cross-Attention.

## 3.1 Structural Causal Model Formalization

We formalize the LVLM generation process through an SCM with observable variables: image inputs $I \in \mathcal { T }$ and textual inputs $T _ { \mathrm { i n } } ~ \in ~ \mathcal { T }$ . The visual encoder decomposes I into visual variables $V = \{ v _ { 1 } , . . . , v _ { k } \}$ (objects, spatial relations), while the LLM decomposes $T _ { \mathrm { i n } }$ into language variables $W = \{ w _ { 1 } , . . . , w _ { m } \}$ (entities, predicates). Latent variables include cross-modal alignment $Z \in \mathbb { R } ^ { d }$ and confounders U (encoding data biases). The SCM equations are:

$$
\left\{ \begin{array} { l l } { V = f _ { V } ( I , \epsilon _ { V } ) } \\ { W = f _ { W } ( T _ { \mathrm { i n } } ) + \epsilon _ { W } } \\ { Z = g ( V , W , U ) } \\ { T _ { \mathrm { o u t } } = h ( Z , U , \epsilon _ { T } ) } \end{array} \right.\tag{1}
$$

where $f _ { V }$ is the visual encoder (ϵ<sub>V</sub>: perceptual uncertainty), f<sub>W</sub> the LLM encoder (ϵ<sub>W</sub>: language ambiguity), g the cross-modal alignment module, U latent confounders, and h the LLM decoder (ϵ<sub>T</sub>: text generation stochasticity). The causal graph de-

pendencies are:

$$
\{ \begin{array} { l l } { I  V  Z  T _ { \mathrm { o u t } } } \\ { T _ { \mathrm { i n } }  W  Z  T _ { \mathrm { o u t } } } \\ { U  Z  T _ { \mathrm { o u t } } } \end{array}\tag{2}
$$

## 3.2 Attention-Guided Causal Intervention

To address unobservable confounders U, we propose a dynamic attention intervention mechanism that regulates cross-modal information flow via configurable masks, blocking non-causal paths. The attention function is:

$$
\mathrm { A t t n } ( Q , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } + { \mathcal { M } } ^ { ( m ) } \right) V ,\tag{3}
$$

where $Q , K \ \in \ \mathbb { R } ^ { n \times d _ { k } }$ are query/key matrices, $V ~ \in ~ \mathbb { R } ^ { n \times d _ { v } }$ the value matrix, and $\mathcal { M } ^ { ( m ) } \in$ $\{ - \infty , 0 \} ^ { n \times n }$ a mode-specific mask.

LVLMs inherit societal biases from pretrained LLMs (Fraser and Kiritchenko, 2024; Hirota et al., 2024; Gorti et al., 2024), where parametric asymmetries bias outputs toward language priors. To investigate causal relationships, we introduce controlled interventions defining five attention modes $( m ~ \in ~ \{ \mathrm { V i s u a l - O n l y } .$ , Language-Only, Visual-to-Language, Language-to-Visual, Bidirectional}):

Visual-Only Self-Attention: Attention confined exclusively to visual tokens, isolating visual interactions (Fig. 2). Language-Only Self-Attention: Attention restricted to textual tokens with causal constraints, maintaining Markovian dependencies (Fig. 2). Visual-to-Language Cross-Attention: Text tokens extract visual features via directional visual-to-language flow, supporting grounded generation (Fig. 2). Languageto-Visual Cross-Attention: Language semantics infuse visual representations via reverse crossattention, reshaping features (Fig. 2). Bidirectional Cross-Attention: Integrates all attention pathways, enabling full inter-modal interactions for balanced fusion (Fig. 2).

Theoretical motivation for Visual-to-Language selection. From a causal perspective, the Visual-to-Language (V2L) attention pattern provides the optimal intervention for bias mitigation. It suppresses the backdoor path $U \to W \to Z$ that carries linguistic stereotypes while preserving the causal flow $V  Z$ of visual evidence. This selective intervention ensures that gender-relevant visual cues inform text generation without being distorted by language priors. A detailed derivation of this a priori justification is provided in Appendix D.

This mechanism dynamically configures causal attention during inference, suppressing confounders while preserving task-relevant interactions.

## 3.3 Causal Effect Quantification

We formalize causal effect quantification using docalculus and d-separation. The causal graph $\mathcal { G }$ (Eq. 2) contains backdoor paths from U to $Z .$ To block these paths, we introduce attention-guided interventions that modify the graph via configurable masks $\mathcal { M } ^ { ( m ) }$

Graph Modification via Attention Intervention Let $\mathcal { G } _ { \mathrm { i n t } }$ denote the intervened graph under attention mode $m .$ . Masks $\mathcal { M } ^ { ( m ) }$ remove edges in $\mathcal { G }$ corresponding to suppressed attention connections. Formally, $\mathcal { M } ^ { ( m ) } \in \{ 0 , - \infty \} ^ { n \times n }$ eliminates specific query-key interactions, transforming $\mathcal { G }$ into $\mathcal { G } _ { \mathrm { i n t } }$ by deleting directed edges.

D-separation Condition In $\mathcal { G } _ { \mathrm { i n t } }$ , we examine the paths between $U$ and $Z _ { c }$ (the causal component of $Z )$ . Under the visual intervention $d o ( V _ { c } = v )$ all backdoor paths from $U$ to $Z _ { c }$ are blocked by the following d-separation conditions:

1. Collider blocking: Any path $U \to Z  V _ { c }$ is blocked because $V _ { c }$ is fixed by intervention, making $Z$ a collider that does not transmit information.

2. Attention mask blocking: Paths through suppressed attention connections are removed by $\mathcal { M } ^ { ( m ) }$ , eliminating indirect dependencies.

3. Conditioning on observed variables: The intervention set $\{ V _ { c } , W _ { c } \}$ satisfies the backdoor criterion relative to $( U , Z _ { c } )$ in $\mathcal { G } _ { \mathrm { i n t } }$

Do-operator Formalization The do-operator $d o ( V _ { c } = v )$ modifies the structural equations by replacing $V _ { c } { ' } s$ equation with the constant v and removing all incoming edges to $V _ { c }$ in ${ \mathcal { G } } .$ Under this intervention, the post-intervention distribution factorizes as:

$$
\begin{array} { l } { P ( Z _ { c } \mid d o ( V _ { c } = v ) ) = \qquad } \\ { \qquad \int P ( Z _ { c } \mid V _ { c } = v , W _ { c } , U ) \qquad } \\ { \qquad \times \ P ( W _ { c } ) P ( U ) d W _ { c } d U } \end{array}\tag{4}
$$

Since all paths from U to $Z _ { c }$ are d-separated given $V _ { c } = v \mathrm { i n } \mathcal { G } _ { \mathrm { i n t } }$ , we have $Z _ { c }$ ⊥⊥ $U \ | \ d o ( V _ { c } = v )$ This conditional independence permits unbiased causal effect estimation without observing $U$

Lemma 1 (Backdoor Path Blocking) Given the causal graph $\mathcal { G }$ and attention intervention $\mathcal { M } ^ { ( m ) }$ the intervened graph $\mathcal { G } _ { \mathrm { i n t } }$ satisfies $Z _ { c } \perp \perp \ U |$ $d o ( V _ { c } = v )$ and $Z _ { c } \downarrow \downarrow U \mid d o ( W _ { c } = w )$ . (A complete proof is provided in Appendix C.)

With this formal justification, we derive causal effects for visual and language attention mechanisms. Visual intervention:

$$
\begin{array} { c } { { P ( T _ { \mathrm { o u t } } \mid d o ( V _ { c } = v ) ) = } } \\ { { \mathbb { E } _ { Z _ { c } \sim P ( Z _ { c } \mid d o ( V _ { c } = v ) ) } \left[ P ( T _ { \mathrm { o u t } } \mid Z _ { c } ) \right] } } \end{array}\tag{5}
$$

where $d o ( V _ { c } = v )$ denotes the interventional assignment fixing $V _ { c }$ to $v ,$ and $P ( Z _ { c } \mid d o ( V _ { c } = v ) )$ represents the post-intervention distribution of latent variable $Z _ { c }$ . Language intervention:

$$
\begin{array} { c } { { P ( T _ { \mathrm { o u t } } \mid d o ( W _ { c } = w ) ) = } } \\ { { \mathbb { E } _ { Z _ { c } \sim P ( Z _ { c } \mid d o ( W _ { c } = w ) ) } \left[ P ( T _ { \mathrm { o u t } } \mid Z _ { c } ) \right] } } \end{array}\tag{6}
$$

where $d o ( W _ { c } = w )$ fixes $W _ { c }$ to w, with $P ( Z _ { c } \mid$ $d o ( W _ { c } = w ) )$ being the corresponding postintervention latent distribution.

## 3.4 Decoding Layer Correction Formula

The text decoding layer $h ( \cdot )$ incorporates direct causal intervention to refine generation. The token prediction process is formally defined as:

$$
A _ { i } = \mathbf { W } _ { o } \cdot \mathrm { A t t n } \big ( Z _ { \mathrm { i n t } } ^ { ( < i ) } , Z _ { \mathrm { i n t } } ^ { ( < i ) } , Z _ { \mathrm { i n t } } ^ { ( < i ) } \big )\tag{7}
$$

$$
P ( t _ { i } | \cdot ) = \frac { \exp ( A _ { i } ) } { \sum _ { j \in \mathcal { V } } \exp ( A _ { j } ) }\tag{8}
$$

where $Z _ { \mathrm { i n t } } \sim P ( Z \mid \mathrm { d o } ( V = v ) )$ denotes latent variables after causal intervention. $Z _ { \mathrm { i n t } } ^ { ( < i ) } = \{ z _ { k } \in$ $Z _ { \mathrm { i n t } } \mid k < i \}$ denotes intervened latent variables preceding position i. Attn(·) denotes the standard self-attention mechanism (query, key, value). $\mathbf { W } _ { o }$ denotes the output projection matrix. V denotes the vocabulary space.

## 4 Experiments

## 4.1 Benchmarks

We evaluate ViD on five benchmarks: (1) FACET (Gustafson et al., 2023) with 50K person instances and fine-grained attributes (occupation, gender, hairstyle, hair color, skin tone), evaluating individual and intersectional gender bias; (2) MS COCO (Zhao et al., 2021) with 28,315 human instances, measuring accuracy against gender annotations; (3) FairFace (Kärkkäinen and Joo, 2021) with 108,501 facial images balanced across 7 race groups, 2 genders, and 9 age ranges, assessing fairness in face analysis; (4) POPE (Li et al., 2023) for object presence verification, evaluating reasoning and debiasing performance (accuracy, precision, recall, F1); (5) MMMU (Yue et al., 2024) with 11.5K expert-curated questions across six disciplines, assessing multimodal understanding.

## 4.2 Baselines

We employ LLaVA(Liu et al., 2023), Instruct-BLIP(Dai et al., 2023), and Shikra(Chen et al., 2023) as base models, representing three distinct LVLM paradigms: visual instruction tuning, vision-language pretraining with instruction finetuning, and fine-grained visual grounding. These models form the foundation for recent LVLM research. To benchmark against ViD, we select recent representative works that address gender bias in vision-language models: Weng et al. (2024) propose a visual-dominant intervention that amplifies image-grounded signals; Howard et al. (2025) employ largescale counterfactual prompting with external datasets; Jung et al. (2024) introduce a unified debiasing approach across modalities; Chuang et al. (2023) apply bias-aware finetuning on visionlanguage encoders; Wang et al. (2021) focus on gender bias mitigation in image captioning through data rebalancing; and Seth et al. (2023) design a debiasing framework that adjusts crossmodal attention weights. These methods span diverse strategiesprompt engineering, finetuning, data augmentation, and attention adjustmentproviding a comprehensive comparative landscape for evaluating ViD’s training-free causally-inspired intervention.

## 4.3 Regular Setting

Experiments use an RTX 4090 GPU with fixed random seed. Standard LVLM decoding parameters: max tokens=1024, beam size=4, temperature=1.0. Gender bias evaluation employs Eq. 9, considering only outputs with identifiable gender (scores 0-1; outputs without gender receive score=2 and are excluded).

$$
\operatorname { S c o r e } ( p , y ) = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } f ( p ) = y } \\ { 1 , } & { { \mathrm { i f ~ } } f ( p ) \neq y \land f ( p ) \neq \emptyset } \\ { 2 , } & { { \mathrm { i f ~ } } f ( p ) = \emptyset } \end{array} \right. }\tag{9}
$$

The gender extraction function $f ( \boldsymbol p )$ uses keyword matching:

$$
f ( p ) = \left\{ \begin{array} { l l } { \mathrm { f e m a l e , } } & { \exists k \in \mathcal { F } : k \mathrm { i n } p } \\ { \mathrm { m a l e , } } & { \nexists k \in \mathcal { F } : k \mathrm { i n } p \land } \\ & { \exists k \in \mathcal { M } : k \mathrm { i n } p } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

with keyword sets F {female, woman, women, girl, girls, she, her, lady, ladies, feminine, womanly, gal, gals} and M = {male, man, men, boy, boys, he, him, gentleman, gentlemen, masculine, manly, guy, guys}. We note that outputs relying solely on implicit gender cues (e.g., occupation-only references such as “nurse”) are not captured by this lexicon-based extractor; consequently, the reported bias scores are conservative underestimates.

## 4.4 Main Results

Results on Gender Bias Benchmarks. As shown in Table 1, ViD demonstrates superior bias mitigation performance across all three benchmarks. On the LLaVA model, ViD improves the gender bias score on FACET from 0.9274 to 0.9928, MS COCO from 0.6708 to 0.9978, and FairFace from 0.9664 to 0.9796. On the InstructBLIP model, it improves FACET from 0.8896 to 0.9812, MS COCO from 0.925 to 0.9992 (achieving unbiased performance), and FairFace from 0.975 to 0.9844. On the Shikra model, ViD shows improvements on both MS COCO and FairFace, with FairFace significantly improved from 0.5988 to 0.8782 (achieving unbiased performance). Compared to existing debiasing methods, ViD exhibits no extreme bias values (e.g., ∞) or male bias (+), maintaining stable and high scores close to 1 in all tests, demonstrating that its visual-to-language attention intervention mechanism effectively suppresses gender stereotypes.

<table><tr><td rowspan="2">Method</td><td colspan="3">LLaVA</td><td colspan="3">InstructBLIP</td><td colspan="3">Shikra</td></tr><tr><td></td><td>FACET MS COCO FairFace</td><td></td><td></td><td>e FACET MS COCO FairFace</td><td></td><td></td><td>FACET MS COCO FairFace</td><td></td></tr><tr><td>Regular</td><td> $0 . 9 2 7 4 _ { 0 . 2 } ^ { - }$ </td><td> $0 . 6 7 0 8 _ { 0 . 0 2 } ^ { - }$ </td><td></td><td> $0 . 9 6 6 4 _ { 0 . 0 7 } ^ { - } 0 . 8 8 9 6 _ { 0 . 0 3 } ^ { - }$ </td><td>0.9250</td><td>0.9750</td><td> $0 . 9 1 7 9 _ { 0 . 4 } ^ { - }$ </td><td> $0 . 9 3 7 9 _ { 0 . 0 9 } ^ { - }$ </td><td> $0 . 5 9 8 8 _ { 0 . 0 5 } ^ { - }$ </td></tr><tr><td>Weng et al., 2024</td><td> $0 . 7 3 7 2 _ { 0 . 8 2 } ^ { + }$ </td><td> $0 . 4 3 5 2 _ { 0 . 7 8 } ^ { + }$ </td><td></td><td> $0 . 5 4 1 4 _ { 0 . 1 8 } ^ { + } 0 . 8 0 4 2 _ { 2 . 4 2 } ^ { + }$ </td><td> $0 . 6 3 2 8 _ { 5 . 1 1 } ^ { + }$ </td><td> $0 . 5 4 2 2 _ { 5 . 7 4 } ^ { + }$ </td><td> $0 . 8 8 3 3 _ { 0 . 1 7 } ^ { + }$ </td><td> $0 . 8 4 4 0 _ { 0 . 3 9 } ^ { + }$ </td><td> $0 . 5 8 6 1 _ { 2 . 0 6 } ^ { + }$ </td></tr><tr><td>Howard et al.,</td><td> $2 0 2 5 0 . 9 7 4 2 _ { 0 . 1 7 } ^ { - }$ </td><td> $0 . 8 4 7 4 _ { 0 . 0 2 } ^ { - }$ </td><td> $0 . 9 8 5 _ { 0 . 0 3 } ^ { - }$ </td><td> $0 . 8 6 3 _ { 0 . 1 2 } ^ { - }$ </td><td> $0 . 7 2 1 2 _ { 0 . 0 1 } ^ { - }$ </td><td> $0 . 9 9 1 8 _ { 0 . 0 1 } ^ { - }$ </td><td> $0 . 7 7 9 2 _ { 0 . 1 6 } ^ { - }$ </td><td> $0 . 5 7 5 7 _ { 0 . 6 0 } ^ { - }$ </td><td> $0 . 9 7 5 4 _ { 0 . 8 2 } ^ { + }$ </td></tr><tr><td>Jung et al., 2024</td><td> $0 . 9 2 5 8 _ { 0 . 2 1 } ^ { - }$ </td><td> $0 . 6 7 6 8 _ { 0 . 0 2 } ^ { - }$ </td><td></td><td> $0 . 9 6 6 4 _ { 0 . 0 6 } ^ { - } 0 . 8 8 9 8 _ { 0 . 0 3 } ^ { - }$ </td><td> $0 . 9 2 6 8 _ { 0 } ^ { - }$ </td><td> $0 . 9 7 6 8 _ { 0 } ^ { \star }$ </td><td> $0 . 8 9 7 _ { 0 . 1 6 } ^ { - }$ </td><td> $0 . 8 8 1 2 _ { 0 . 3 8 } ^ { - }$ </td><td> $0 . 9 3 9 3 _ { 0 . 9 0 } ^ { + }$ </td></tr><tr><td>Chuang et al., 2023</td><td> $0 . 9 9 6 4 _ { 0 . 3 5 } ^ { - }$ </td><td> $0 . 9 6 2 6 _ { 0 . 1 0 } ^ { - }$ </td><td> $0 . 9 7 7 2 _ { 0 } ^ { - }$ </td><td> $0 . 9 0 9 0 _ { 0 . 1 0 } ^ { - }$ </td><td> $0 . 9 5 8 2 _ { 0 . 0 1 } ^ { - }$ </td><td>0.9782</td><td> $1 . 0 _ { \infty } ^ { + }$ </td><td> $0 . 0 _ { \infty } ^ { + }$ </td><td> $0 . 0 _ { \infty } ^ { + }$ </td></tr><tr><td>Wang et al., 2021</td><td> $0 . 5 5 5 6 _ { 0 . 5 5 } ^ { - }$ </td><td> $0 . 7 7 7 0 _ { 0 } ^ { - }$ </td><td></td><td> $0 . 5 4 9 0 _ { 0 . 5 7 } ^ { - } 0 . 8 9 1 8 _ { 0 . 0 3 } ^ { - }$ </td><td>0.92680</td><td>0.97640</td><td> $0 . 0 _ { \infty } ^ { + }$ </td><td> $0 . 0 _ { \infty } ^ { + }$ </td><td> $0 . 0 _ { \infty } ^ { + }$ </td></tr><tr><td>Seth et al., 2023</td><td> $1 . 0 _ { 1 . 3 5 } ^ { - }$ </td><td> $1 . 0 _ { \infty } ^ { + }$ </td><td> $1 . 0 _ { 0 . 6 3 } ^ { - }$ </td><td> $0 . 0 2 4 2 _ { 0 } ^ { - }$ </td><td>0.01640</td><td>0.16900</td><td> $0 . 1 1 5 2 _ { 0 . 2 4 } ^ { - }$ </td><td> $0 . 0 6 2 5 _ { 0 . 7 1 } ^ { + }$ </td><td> $0 . 0 4 0 4 _ { 1 . 3 8 } ^ { + }$ </td></tr><tr><td>ViD</td><td> $0 . 9 9 2 8 _ { 0 . 0 8 } ^ { - }$ </td><td> $0 . 9 9 7 8 _ { 0 . 1 0 } ^ { - }$ </td><td></td><td> $0 . 9 7 9 6 _ { 0 . 3 4 } ^ { - } 0 . 9 8 1 2 _ { 0 . 1 5 } ^ { - }$ </td><td>0.99920</td><td> $0 . 9 8 4 4 _ { 0 . 0 5 } ^ { - }$ </td><td>0.88990.42</td><td> $0 . 9 3 9 6 _ { 0 . 1 7 } ^ { - }$ </td><td>0.87820</td></tr></table>

Table 1: A Comprehensive Performance Comparison of Various Methods on the FACET, MS COCO, and FairFace Benchmarks. Each cell shows the bias score (main number, 0-1, closer to 1 indicates lower gender bias), log gender ratio (subscript, log $\scriptstyle \left( P ( { \mathrm { m a l e } } ) / P ( { \mathrm { f e m a l e } } ) \right)$ ), and bias direction (superscript: $" + "$ for male bias, $^ { 6 6 } - ^ { 5 9 }$ for female bias, ⋆ for unbiased). For example, $0 . 9 2 7 4 _ { 0 . 2 } ^ { - }$ indicates a bias score of 0.9274, log gender ratio of 0.2, and female bias. The symbol ∞ indicates an extreme log ratio when P(female) = 0 or P(male) = 0. Coverage rates (percentage of samples where gender is predicted) are reported in Appendix A (Table 8).

reasoning across academic domains. ViD shows a balanced trade-off between bias mitigation and reasoning preservation. On LLaVA, ViD scores 31.8, close to the baseline (32.4) and competitive with other methods (Weng et al. (2024): 31.7, Howard et al. (2025): 32.0, Jung et al. (2024): 32.0, Chuang et al. (2023): 30.6, Wang et al. (2021): 27.9, Seth et al. (2023): 24.4). ViD excels in specific domains (52.5 in Art and Design, 23.3 in Business). Across architectures, ViD maintains consistent performance: 28.7 on Instruct-BLIP (baseline 29.4) and 26.7 on Shikra (matching baseline 26.9). These results show ViD effectively mitigates bias while preserving reasoning abilities.

Results on POPE. POPE evaluates reasoning capability preservation during debiasing. ViD has minimal impact on general capabilities: LLaVA accuracy increases from 82.63% to 83.06%, InstructBLIP decreases from 81.53% to 80.73%, and Shikra decreases from 81.70% to 79.80%, with similar F1 changes. In contrast, other methods significantly degrade capabilities: Howard et al. (2025) reduce InstructBLIP accuracy to 50.00%, Seth et al. (2023) achieve 50.16% accuracy with F1=4.35, and Weng et al. (2024) reduce Shikra accuracy to 50.20%. Detailed analysis of these extreme performance drops (e.g., output format incompatibilities and loss of reasoning accuracy) is provided in Appendix B. ViD preserves reasoning ability while effectively eliminating biases.

Results on MMMU. The MMMU benchmark evaluates how debiasing methods affect LVLM

Comparison of Multiple Attention Patterns. We evaluate five attention patterns on FACET using 5,000 gender-balanced samples (Table 3). Visual-only achieves high bias score (0.9995) but low coverage (9.3%) with female bias. Languageonly shows high bias score (0.9988) but strong male bias with minimal coverage (4.9%). Bidirectional (L2V&V2L) balances bias score (0.9938) and coverage (90.4%). Visual-to-Language (V2L) demonstrates superior performance: high coverage (95.8%), near-perfect bias score (0.9908), balanced gender distribution, and statistical significance, improving over the Regular baseline (0.9274). Language-to-Visual (L2V) has high score but negligible coverage (0.2%). V2L’s superiority stems from amplifying visual evidence while reducing linguistic stereotype interference, making it the optimal configuration.

<table><tr><td rowspan="2">Method</td><td colspan="4">LLaVA</td><td colspan="4">InstructBLIP</td><td colspan="4">Shikra</td></tr><tr><td>Acc</td><td>Prec</td><td>Recall</td><td>F1</td><td>Acc</td><td>Prec</td><td>Recall</td><td>F1</td><td>Acc</td><td>Prec</td><td>Recall</td><td>F1</td></tr><tr><td>Regular</td><td>82.63</td><td>82.83</td><td>82.33</td><td>82.58</td><td>81.53</td><td>79.41</td><td>85.13</td><td>82.17</td><td>81.7</td><td>80.57</td><td>83.53</td><td>82.02</td></tr><tr><td>Weng et al., 2024</td><td>81.60</td><td>80.46</td><td>83.46</td><td>81.93</td><td>81.53</td><td>79.37</td><td>85.2</td><td>82.18</td><td>50.20</td><td>50.10</td><td>97.40</td><td>66.16</td></tr><tr><td>Howard et al., 2025</td><td>74.96</td><td>68.27</td><td>93.26</td><td>78.83</td><td>50.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>76.10</td><td>80.18</td><td>69.33</td><td>74.36</td></tr><tr><td>Jung et al., 2024</td><td>82.63</td><td>82.83</td><td>82.33</td><td>82.58</td><td>81.50</td><td>85.76</td><td>75.53</td><td>80.32</td><td>81.70</td><td>80.57</td><td>83.53</td><td>82.02</td></tr><tr><td>Chuang et al., 2023</td><td>82.13</td><td>85.49</td><td>77.40</td><td>81.24</td><td>80.36</td><td>78.48</td><td>83.66</td><td>80.99</td><td>50.00</td><td>50.00</td><td>100.0</td><td>66.66</td></tr><tr><td>Wang et al., 2021</td><td>54.73</td><td>59.72</td><td>29.06</td><td>39.10</td><td>81.6</td><td>81.89</td><td>81.13</td><td>81.51</td><td>57.00</td><td>56.11</td><td>64.20</td><td>59.88</td></tr><tr><td>Seth et al., 2023</td><td>50.00</td><td>50.00</td><td>100.0</td><td>66.66</td><td>50.16</td><td>53.96</td><td>2.26</td><td>4.35</td><td>50.03</td><td>50.01</td><td>98.46</td><td>66.33</td></tr><tr><td>ViD</td><td>83.06</td><td>88.68</td><td>75.80</td><td>81.73</td><td>80.73</td><td>78.66</td><td>84.33</td><td>81.40</td><td>79.80</td><td>77.76</td><td>83.46</td><td>80.51</td></tr></table>

Table 2: Impact of Various Debiasing Methods on LVLM General Reasoning Capabilities Evaluated by POPE Adversarial

<table><tr><td>Gender Bias Male Count Female Count Log-ratio</td><td></td><td></td><td></td><td></td><td>p-value</td><td>95% confidence interval</td></tr><tr><td>Regular</td><td>0.9274</td><td>2474</td><td>2526</td><td>-0.0208</td><td>&lt;0.0001</td><td>[0.4350, 0.4626]</td></tr><tr><td>Visual-only</td><td>0.9995</td><td>208</td><td>258</td><td>-0.2154</td><td>0.0198</td><td>[0.4012, 0.4915]</td></tr><tr><td>Language-only</td><td>0.9988</td><td>154</td><td>90</td><td>0.5371</td><td>&lt;0.0001</td><td>[0.5706, 0.6917]</td></tr><tr><td>V2L</td><td>0.9908</td><td>2305</td><td>2486</td><td>-0.0756</td><td>0.0088</td><td>[0.4670, 0.4953]</td></tr><tr><td>L2V</td><td>0.9994</td><td>5</td><td>4</td><td>0.2231</td><td>0.7373</td><td>[0.2309, 0.8802]</td></tr><tr><td>L2V&amp;V2L</td><td>0.9938</td><td>2118</td><td>2400</td><td>-0.1250</td><td>&lt;0.0001</td><td>[0.4542, 0.4833]</td></tr></table>

Table 3: Comparative Analysis of Different Attention Patterns on Gender Bias in FACET Benchmark. Gender Bias scores closer to 1.0 indicate lower bias. Log-ratio measures gender distribution skew (negative: female bias, positive: male bias). Statistical significance (p-value) and 95% confidence intervals are reported for gender proportion estimates.

<table><tr><td>High</td><td>FACET</td><td>MS COCO</td><td>FairFace</td></tr><tr><td>0.1</td><td> $0 . 9 9 8 0 _ { 0 . 0 5 } ^ { + }$ </td><td> $0 . 9 9 9 0 _ { 0 . 3 0 } ^ { + }$ </td><td>0.98200</td></tr><tr><td>1.0</td><td> $0 . 9 9 6 0 _ { 0 . 4 9 } ^ { - }$ </td><td> $1 . 0 0 0 0 _ { 1 . 3 8 } ^ { + }$ </td><td>0.98100</td></tr><tr><td>5.0</td><td> $0 . 9 8 9 0 _ { 0 . 0 8 } ^ { - }$ </td><td> $1 . 0 0 0 0 _ { 0 . 4 0 } ^ { + }$ </td><td> $0 . 9 8 5 0 _ { 0 } ^ { \star }$ </td></tr><tr><td>10.0</td><td> $0 . 9 8 4 0 _ { 0 . 7 4 } ^ { + }$ </td><td> $0 . 9 9 7 0 _ { 1 . 7 9 } ^ { + }$ </td><td> $0 . 9 9 3 0 _ { 0 } ^ { \star }$ </td></tr></table>

Table 4: Ablation study of high parameter on three datasets (low=0.1, mid=0.1). Each cell shows the bias score (main number, 0-1, closer to 1 indicates lower gender bias), log gender ratio (subscript, $\log ( P ( { \mathrm { m a l e } } ) / P ( { \mathrm { f e m a l e } } ) ) )$ , and bias direction (superscript: $" + "$ for male bias, “-” for female bias, ⋆ for unbiased). For example, $0 . 9 9 8 0 _ { 0 . 0 5 } ^ { + }$ indicates a bias score of 0.9980, log gender ratio of 0.05, and male bias. The analysis reveals consistent patterns across datasets with high=1.0 achieving optimal balance between high bias score and minimal gender bias.

Text Quality Evaluation. As shown in Table 6, we quantitatively evaluate the impact of visual-to-language attention on textual fluency using perplexity metrics. Using 1,000 randomly sampled MSCOCO images, we measure language quality through two GPT-2 variants: $\mathrm { P P L _ { 1 } ( G P T \mathrm { - } }$ 2) and $\mathrm { P P L _ { 2 } }$ (GPT-2-medium). Our analysis reveals that visual-to-language not only mitigates gender biases but also enhances language coherence, achieving superior perplexity scores compared to the baseline.

<table><tr><td colspan="2">Attributes</td><td colspan="2">Score↓</td></tr><tr><td>Occupation</td><td>Skin</td><td>Hair Regular</td><td>ViD</td></tr><tr><td>√</td><td>X</td><td>X 0.1142</td><td>0.0974</td></tr><tr><td>√</td><td>√ X</td><td>0.2339</td><td>0.206</td></tr><tr><td>√</td><td>√ √</td><td>0.2569</td><td>0.2304</td></tr></table>

Table 5: Multi-attribute gender bias assessment. Visualto-language attention reduces gender bias across increasingly complex social attribute combinations (13 attributes). Lower scores indicate less bias. Relative improvements: 14.7% (1 attribute), 11.9% (2 attributes), 10.3% (3 attributes).

## 4.5 Ablation Study

The intervention parameters low, mid, and high denote the early, middle, and late groups of language decoder layers (for LVLMs with 30 language layers: 0–10, 11–20, and 21–30), controlling the strength of visual-to-language attention suppression in each group. We conduct an ablation study on these parameters across FACET, COCO, and FairFace. A grid search over values

Table 6: Perplexity comparison for text generation quality. Lower values indicate better fluency. Visual(V)- to-language(L) attention reduces perplexity by 2.1% (GPT-2) and 8.8% (GPT-2-medium) compared to baseline, demonstrating enhanced language coherence while mitigating gender bias.  
![](images/6ec94b85d9ee972b2657e89e6a725e5aac2ba44f499c4447c87f0d27dd14cdba.jpg)  
Figure 3: Radar chart of accuracy distributions across academic domains in the MMMU benchmark.

<table><tr><td></td><td> $\mathbf { P P L } _ { \mathrm { 1 downarrow } }$ </td><td> $\mathbf { P P L } _ { 2 \downarrow }$ </td></tr><tr><td>Regular</td><td>2.5247</td><td>1.7826</td></tr><tr><td>V→L</td><td>2.4715</td><td>1.6256</td></tr></table>

<table><tr><td>Length</td><td>Regular (ms)</td><td>ViD (ms)</td><td> $\Delta$  (%)</td></tr><tr><td>8</td><td>269.94</td><td>268.79</td><td>-0.43</td></tr><tr><td>64</td><td>1270.04</td><td>1279.15</td><td>+0.71</td></tr><tr><td>512</td><td>2171.12</td><td>2191.63</td><td>+0.94</td></tr></table>

Table 7: Empirical Complexity Validation: Endto-end inference latency (mean over 1,000 samples) demonstrating quadratic scaling and minimal overhead for V→L attention.

0.1, 1.0, 5.0, 10.0 shows the high parameter has the strongest influence on bias mitigation (Table 4). With low=mid=0.1, increasing high from 0.1 to 1.0 reduces gender log-probability from +0.06 to -0.50 (p = 0.0046) in FACET while maintaining bias score >0.99. MS COCO maintains perfect scores (1.000) but with increased log-probabilities, and FairFace remains stable. Low and mid parameters show minor effects. The optimal configuration (low=0.1, mid=0.1, high=1.0) achieves a balance between high bias score (0.996) and minimal gender bias.

## 4.6 Multi-Attribute Gender Bias Assessment

To demonstrate the generalizability of structural causal models (SCMs), we conduct comprehensive gender bias evaluations on FACET using both binary (skin tone + occupation) and ternary (hairstyle + hair color + skin tone + occupation) attribute combinations. As shown in Table 5, visualto-language attention significantly reduces gender bias across all combinatorial settingsfrom single to triple attribute interactionsachieving average gender bias reduction of 18.3%. This demonstrates the method’s scalability to complex multi-attribute mitigation scenarios.

## 4.7 Time Complexity Analysis

The visual-to-language (V→L) cross-attention exhibits $\mathcal { O } ( V ^ { 2 } + T ^ { 2 } + T \cdot V )$ mask generation complexity, where V and T denote visual/text lengths. This reduces to $\mathcal { O } ( L ^ { 2 } )$ when $V , T \sim { \mathcal { O } } ( L )$ , maintaining standard Transformer’s asymptotic complexity. Empirical validation (Table 7) confirms quadratic scaling: 8× length increase (864) yields 4.7× latency growth, while 64× increase (8512) produces 8.2× growth. Crucially, v→L introduces negligible overhead (<1% vs regular attention), and the T · V cross-term demonstrates minimal real-world impact, preserving the $\mathcal { O } ( L ^ { 2 } )$ boundary while enabling multimodal integration.

## 5 Conclusion

We introduce ViD, a structural causal modeling framework that mitigates gender biases in LVLMs through visual-to-language attention. This approach reduces gender bias by 14.7% (single attributes), 11.9% (two attributes), and 10.3% (three attributes), scaling to multi-attribute scenarios. ViD establishes visual grounding as a causal invariant, requires no retraining or architectural changes, and provides a principled pathway toward ethically-aligned multimodal systems, with layer selection optimization as future work.

## Limitations

ViD effectively mitigates gender bias via visualto-language attention, yet several research directions remain. Our evaluation focuses on gender bias across occupation, skin tone, hair, and hairstyle; extending to other social biases (racial, age, intersectional) is a valuable future direction. The causal intervention uses manually constructed structural models; automating causal discovery could enhance robustness. Evaluation assumes attribute annotations, potentially limiting annotation-scarce deployment—annotationefficient variants are an important next step. Although we report coverage rates to address evasion (Appendix A), designing metrics that jointly penalize bias and evasion remains an open challenge.

## References

Marion Bartl and Susan Leavy. 2024. From ‘showgirls to ‘performers’: Fine-tuning with gender-inclusive language for bias reduction in LLMs. In Proceedings of the 5th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 280– 294. Association for Computational Linguistics.

Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? . In Conference on Fairness, Accountability and Transparency, FAccT ’21, pages 610– 623, New York, NY, USA. Association for Computing Machinery.

Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. 2022. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16495– 16504.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic. Preprint, arXiv:2306.15195.

Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. 2023. Debiasing vision-language models via biased prompts. Preprint, arXiv:2302.00070.

Wenliang Dai, Junnan Li, DONGXU LI, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 49250– 49267. Curran Associates, Inc.

Kathleen Fraser and Svetlana Kiritchenko. 2024. Examining gender and racial bias in large visionlanguage models using a novel dataset of parallel images. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 690–713. Association for Computational Linguistics.

Leander Girrbach, Stephan Alaniz, Yiran Huang, trevor darrell, and Zeynep Akata. 2025. Revealing and reducing gender biases in vision and language assistants (vlas). In International Conference on Learning Representations, pages 8727–8764.

Atmika Gorti, Manas Gaur, and Aman Chadha. 2024. Unboxing occupational bias: Debiasing LLMs with U.S. labor data. In PROCEEDINGS OF THE 2024 AAAI FALL SYMPOSIA, pages 48–55. Association for the Advancement of Artificial Intelligence (AAAI).

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Laura Gustafson, Chloe Rolland, Nikhila Ravi, Quentin Duval, Aaron Adcock, Cheng-Yang Fu, Melissa Hall, and Candace Ross. 2023. Facet: Fairness in computer vision evaluation benchmark. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 20313–20325. IEEE.

Yusuke Hirota, Ryo Hachiuma, Chao-Han Huck Yang, and Yuta Nakashima. 2024. From descriptive richness to bias: Unveiling the dark side of generative image caption enrichment. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17807–17816. Association for Computational Linguistics.

Phillip Howard, Kathleen C. Fraser, Anahita Bhiwandiwalla, and Svetlana Kiritchenko. 2025. Uncovering bias in large vision-language models at scale with counterfactuals. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5946–5991. Association for Computational Linguistics.

Sepehr Janghorbani and Gerard De Melo. 2023. Multimodal bias: Introducing a framework for stereotypical bias assessment beyond gender and race in vision–language models. In Proceedings ofthe 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1725– 1735. Association for Computational Linguistics.

Cong Jin, Ruolin Zhu, Zixing Zhu, Lu Yang, Min Yang, and Jiebo Luo. 2024. Mtartgpt: A multi-task art generation system with pre-trained transformer. IEEE

Transactions on Circuits and Systemsfor Video Technology, 34(8):6901–6912.

Hoin Jung, Taeuk Jang, and Xiaoqian Wang. 2024. A unified debiasing approach for vision-language models across modalities and tasks. In Advances in Neural Information Processing Systems, volume 37, pages 21034–21058. Curran Associates, Inc.

Vivek Hruday Kavuri, Vysishtya Karanam Karanam, Venkamsetty Venkata Jahnavi, Kriti Madumadukala, Balaji Lakshmipathi Darur, and Ponnurangam Kumaraguru. 2025. Freeze and reveal: Exposing modality bias in vision-language models. In Proceedings of Interdisciplinary Workshop on Observations of Misunderstood, Misguided and Malicious Use of Language Models, pages 17–26. INCOMA Ltd., Shoumen, Bulgaria.

Lei Ke, Wenjie Pei, Ruiyu Li, Xiaoyong Shen, and Yu-Wing Tai. 2019. Reflective decoding network for image captioning. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8887–8896. IEEE.

Hannah Rose Kirk, Yennie Jun, Filippo Volpin, Haider Iqbal, Elias Benussi, Frederic Dreyer, Aleksandar Shtedritski, and Yuki Asano. 2021. Bias out-of-thebox: An empirical analysis of intersectional occupational biases in popular generative language models. In Advances in Neural Information Processing Systems, volume 34, pages 2611–2624. Curran Associates, Inc.

Fanjie Kong, Shuai Yuan, Weituo Hao, and Ricardo Henao. 2023. Mitigating test-time bias for fair image retrieval. In Advances in Neural Information Processing Systems, volume 36, pages 71520– 71539. Curran Associates, Inc.

Kimmo Kärkkäinen and Jungseock Joo. 2021. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1548–1558.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305. Association for Computational Linguistics.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc.

Judea Pearl. 2009. Causality: Models, Reasoning, and Inference, 2nd edition. Cambridge University Press, Cambridge.

Ashish Seth, Mayur Hemani, and Chirag Agarwal. 2023. Dear: Debiasing vision-language models with additive residualsh. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6820–6829. IEEE.

Jialu Wang, Yang Liu, and Xin Wang. 2021. Are gender-neutral queries really gender-neutral? mitigating gender bias in image search. In Conference on Empirical Methods in Natural Language Processing, pages 1995–2008. Association for Computational Linguistics.

Zhaotian Weng, Zijun Gao, Jerone Andrews, and Jieyu Zhao. 2024. Images speak louder than words: Understanding and mitigating bias in vision-language model from a causal mediation perspective. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 15669– 15680. Association for Computational Linguistics.

Yisong Xiao, Xianglong Liu, QianJia Cheng, Zhenfei Yin, Siyuan Liang, Jiapeng Li, Jing Shao, Aishan Liu, and Dacheng Tao. 2025. Genderbias-vl: Benchmarking gender bias in vision language models via counterfactual probing. International Journal ofComputer Vision, 133(12):8332–8355.

Zeping Yu and Sophia Ananiadou. 2024. Interpreting arithmetic mechanism in large language models through comparative neuron analysis. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3293–3306. Association for Computational Linguistics.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567.

Yunqi Zhang, Songda Li, Chunyuan Deng, Luyi Wang, and Hui Zhao. 2024. Think before you act: A twostage framework for mitigating gender bias towards vision-language tasks. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 773–791. Association for Computational Linguistics.

Dora Zhao, Angelina Wang, and Olga Russakovsky. 2021. Understanding and evaluating racial biases in image captioning. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14810–14820. IEEE.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing

methods. In Proceedings ofthe 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 15–20. Association for Computational Linguistics.

## A Coverage Rates for Gender Bias Evaluation

The coverage rates in Table 8 reveal important patterns in model evasion behavior. For LLaVA and InstructBLIP architectures, most methods achieve near-perfect coverage (close to 100%) across all three benchmarks, indicating minimal evasion. However, Shikra exhibits significant variation: baseline methods show extremely low coverage (e.g., 0%-25% for several methods), while ViD maintains substantially higher coverage (44.2%- 76.6%). Notably, Seth et al. (2023) demonstrates near-zero coverage on LLaVA (0.78%, 0%, 1.7%), suggesting this method induces strong evasion behavior. The high coverage of ViD on Shikra, combined with its competitive bias scores (Table 1), indicates that our method effectively mitigates gender bias without causing models to avoid gender prediction tasks.

## B Analysis of Extreme Performance Drops in POPE Benchmark

Explanation of extreme performance drops in Table 2. The POPE (Prompt-based Object Perception Evaluation) benchmark uses question-answer pairs in the form of “Is there a handbag in the image?” with expected answers being “yes” or “no”. The observed extreme performance drops for certain baseline methods (e.g., Howard et al. Howard et al., 2025, Seth et al. Seth et al., 2023, Weng et al. Weng et al., 2024, and Wang et al. Wang et al., 2021) can be attributed to output format incompatibilities rather than fundamental failures of the debiasing approaches.

Specifically, when applying these debiasing methods to LVLMs evaluated on POPE, we observed that the models sometimes generate garbled characters, meaningless text, or non-binary responses instead of the expected “yes”/“no” answers. For example:

• Wang et al. Wang et al., 2021: The method produced responses containing random symbols, repeated numbers, or partial words instead of “yes”/“no” answers. For example, for the prompt “Is there a teddy bear in the image?”, the model generated “1 1 1 1 1 1” instead of a binary response, causing the binary classification logic to fail and resulting in accuracy near random chance (54.73% on LLaVA).

• Seth et al. Seth et al., 2023: This approach caused the LVLM to lose general reasoning capability, resulting in incorrect binary judgments rather than format inconsistencies. The model frequently answered “No” when the correct answer should be “Yes” (or vice versa), leading to factual errors in object perception. For example, while the model sometimes generated verbose descriptions like “Yes, there is a truck in the image...”, it more often produced incorrect binary judgments that directly contradicted visual evidence. This loss of reasoning accuracy, combined with significantly slower inference speed (1.00s/it compared to baseline), led to low F1 scores (4.35 on InstructBLIP) despite moderate accuracy.

• Weng et al. Weng et al., 2024: On Shikra architecture, the method produced non-binary responses that reduced accuracy to nearrandom levels (50.20%).

• Howard et al. Howard et al., 2025: The encoder-decoder structure of InstructBLIP appeared incompatible with this method’s output format, resulting in contradictory predictions and 0.00 precision/recall.

These issues stem from the fact that these debiasing methods were primarily designed for openended generation tasks and may not preserve the strict binary response format required by POPE. In contrast, our ViD method maintains the model’s original response patterns while mitigating bias, thus preserving compatibility with binary evaluation benchmarks.

All baseline implementations followed their original papers with hyperparameters tuned on validation splits to ensure fair comparison. The performance drops highlight the importance of maintaining output format consistency when applying debiasing interventions to specific evaluation frameworks.

## C Complete Proof of Lemma 1

Lemma 1 (Backdoor Path Blocking): Given the causal graph G and attention intervention M<sup>(m)</sup>, the intervened graph $\mathcal { G } _ { \mathrm { i n t } }$ satisfies $Z _ { c } \perp \perp \ U |$ $d o ( V _ { c } = v )$ and $Z _ { c } \perp \perp \boldsymbol { U } \mid d o ( W _ { c } = w )$

<table><tr><td rowspan="2">Method</td><td colspan="3">LLaVA</td><td colspan="3">InstructBLIP</td><td colspan="3">Shikra</td></tr><tr><td>FACET MS COCO FairFace FACET MS COCO FairFace FACET MS COCO FairFace</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Regular</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>5%</td><td>8%</td><td>72.1%</td></tr><tr><td>Weng et al., 2024</td><td>100%</td><td>100%</td><td>99.86%</td><td>100%</td><td>100%</td><td>100%</td><td>9.82%</td><td>16.18%</td><td>71.74%</td></tr><tr><td>Howard et al., 2025</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>8.1%</td><td>2.88%</td><td>1.38%</td></tr><tr><td>Jung et al., 2024</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>23%</td><td>25.88%</td><td>1.94%</td></tr><tr><td>Chuang et al., 2023 88.82%</td><td></td><td>96.46%</td><td>66.7%</td><td>100%</td><td>100%</td><td>100%</td><td>0.18%</td><td>0%</td><td>0.12%</td></tr><tr><td>Wang et al., 2021</td><td>99.86%</td><td>100%</td><td>99.98%</td><td>100%</td><td>100%</td><td>100%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Seth et al., 2023</td><td>0.78%</td><td>0%</td><td>1.7%</td><td>100%</td><td>100%</td><td>100%</td><td>2.28%</td><td>1.58%</td><td>4.08%</td></tr><tr><td>ViD</td><td>95.38%</td><td>91.22%</td><td>95.36%</td><td>100%</td><td>100%</td><td>100%</td><td>44.2%</td><td>73.2%</td><td>76.62%</td></tr></table>

Table 8: Coverage rates (percentage of samples where gender is predicted) for methods evaluated in Table 1. Coverage is calculated as (Male Count + Female Count) / Total Samples $\times 1 0 0 \%$ . Higher coverage indicates fewer evasive responses.

Proof. We provide a complete proof for the visual intervention case $d o ( V _ { c } = v )$ ; the language intervention case follows symmetrically.

Step 1: Graph structure and notation. Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be the original causal graph with vertices $\mathcal { V } = \{ U , V _ { c } , W _ { c } , Z _ { c } , T _ { \mathrm { o u t } } \}$ and edges $\mathcal { E }$ representing causal dependencies. Here $U$ denotes unobserved confounders (language priors), $V _ { c }$ visual components, $W _ { c }$ language components, $Z _ { c }$ latent representations, and $T _ { \mathrm { o u t } }$ output text.

Step 2: Intervention operator. The dooperator do $( V _ { c } = v )$ modifies $\mathcal { G }$ to create $\mathcal { G } _ { \mathrm { i n t } }$ by:

1. Removing all incoming edges to $V _ { c } \digamma ( \mathrm { i . e . }$ deleting edges $U \ \to \ V _ { c }$ and $W _ { c } \ \to \ V _ { c }$ if present).

2. Setting $V _ { c }$ to the constant value v.

Step 3: Attention mask modification. The attention intervention $\mathcal { M } ^ { ( m ) }$ further modifies $\mathcal { G } _ { \mathrm { i n t } }$ by removing specific edges corresponding to suppressed attention connections. Formally, for each zero entry $\mathcal { M } _ { i j } ^ { ( m ) } = 0 \left( \mathrm { o r } - \infty \right)$ , we delete the corresponding directed edge from node j to node i in the computational graph.

Step 4: Path analysis. We enumerate all possible paths from $U$ to $Z _ { c }$ in $\mathcal { G } _ { \mathrm { i n t } }$ and show each is d-separated:

1. Direct path $U \to Z _ { c } \mathrm { : }$ In the original LVLM architecture, $U$ influences $Z _ { c }$ through attention mechanisms. The attention mask $\mathcal { M } ^ { ( m ) }$ eliminates this direct connection when visualto-language attention is suppressed.

2. Path $U \ \to \ W _ { c } \ \to \ Z _ { c } ;$ : This path represents language prior influence through language components. Under $d o ( V _ { c } = v ) , W _ { c }$ is not fixed, but the attention mask blocks $W _ { c } ~  ~ Z _ { c }$ connections when language-tovisual attention is suppressed.

3. Path $U \ \to \ V _ { c } \ \to \ Z _ { c } ;$ The intervention $d o ( V _ { c } = v )$ removes the edge $U  V _ { c }$ , breaking this path at its source.

4. Collider paths: Any path containing $Z _ { c }$ as a collider $( \mathbf { e . g . , } U \to X  Z _ { c } )$ is blocked because $Z _ { c }$ is not conditioned upon.

Step 5: d-Separation formalization. A path p between $U$ and $Z _ { c }$ is d-separated given do $( V _ { c } = v )$ if either:

• p contains a chain $A  B  C$ or fork $A $ $B  C$ where B is fixed by intervention, or

• p contains a collider $A \right. B \left. C$ where B is not conditioned upon.

For all paths enumerated in Step 4:

• Paths containing $V _ { c }$ are blocked because $V _ { c }$ is fixed by intervention (satisfying the chain/fork condition).

• Paths through attention connections are blocked by $\mathbf { \mathcal { M } } ^ { ( m ) }$ removing corresponding edges.

• Collider paths are blocked as $Z _ { c }$ is not conditioned upon.

Step 6: Conditional independence. By the d-separation theorem (Pearl, 2009), if all paths between U and $Z _ { c }$ are d-separated in $\mathcal { G } _ { \mathrm { i n t } }$ , then $Z _ { c } \perp \perp \boldsymbol { U } \mid d o ( V _ { c } = v )$ . This holds for our graph because:

$$
P ( Z _ { c } \mid d o ( V _ { c } = v ) , U ) = P ( Z _ { c } \mid d o ( V _ { c } = v ) )\tag{10}
$$

since the conditional distribution of $Z _ { c }$ given the intervention does not depend on $U$

Step 7: Extension to language intervention. The proof for $d o ( W _ { c } = w )$ follows symmetrically by swapping the roles of $V _ { c }$ and $W _ { c } ,$ , and considering language-to-visual attention suppression instead of visual-to-language suppression.

Conclusion. We have shown that for all possible paths from U to $Z _ { c }$ in the intervened graph $\mathcal { G } _ { \mathrm { i n t } }$ , each path is d-separated by either the intervention itself or the attention mask modification. Therefore, $Z _ { c } \perp \perp \boldsymbol { U } \mid d o ( V _ { c } = v )$ and $Z _ { c } \perp \perp U$ | $d o ( W _ { c } = w )$ . □

Implications. This lemma provides the theoretical foundation for our causal intervention: by blocking backdoor paths from confounders U to latent representations $Z _ { c } .$ , we can estimate unbiased causal effects of visual and language components on model output.

## D Theoretical Justification for Visual-to-Language Attention Selection

A priori motivation for V2L selection. The choice of Visual-to-Language (V2L) attention as our primary intervention pattern is motivated by both theoretical considerations and empirical observations from the causal framework.

Theoretical analysis. From a causal perspective, gender bias in LVLMs primarily originates from language priors U that act as confounders between visual inputs V and output text $T _ { \mathrm { o u t } }$ . The structural causal model (Eq. 1) indicates that U influences $Z$ (latent representations) through both direct paths $( U \to Z )$ and indirect paths via language components $W \left( U \to W \to Z \right)$

Our intervention strategy aims to:

1. Amplify visual evidence: Visual information V provides direct, unbiased evidence about gender attributes (e.g., facial features, clothing, context).

2. Suppress linguistic stereotypes: Language components W carry stereotypical associations learned from pretraining data.

Formal derivation from causal graphs. The selection of V2L as the optimal intervention pattern follows directly from the causal structure of bias in LVLMs, not from post-hoc empirical analysis. Given the causal graph $\mathcal { G } \left( \mathrm { E q . } 2 \right)$ with confounder U, we seek an attention mask $\mathcal { M } ^ { ( m ) }$ that satisfies two conditions:

1. Condition 1 (Suppress backdoor paths): Suppress all paths from U to Z that pass through W, i.e., suppress edges $W  Z$ in $\mathcal { G } _ { \mathrm { i n t } }$

2. Condition 2 (Preserve visual evidence): Maintain the direct causal path $V  Z \uparrow 0$ allow visual information to influence text generation.

Among the five attention patterns, only V2L satisfies both conditions simultaneously:

• V2L: Suppresses $W  Z$ (Condition 1) by masking language-to-visual attention, while preserving $V  Z$ (Condition 2) via visualto-language attention.

• L2V: Preserves $W \to Z$ (violates Condition 1) while blocking $V  Z$ (violates Condition 2).

• Bidirectional: Preserves both $W  Z$ and $V  Z$ , thus violating Condition 1.

• Visual-only: Suppresses $W  Z$ (satisfies Condition 1) but excessively restricts information flow, hindering text generation.

• Language-only: Preserves $W \to Z$ (violates Condition 1) and blocks $V \  \ Z$ (violates Condition 2).

Thus, V2L is the unique pattern that achieves the theoretical objectives without compromising generation capability.

The V2L attention pattern directly addresses both objectives:

• By allowing visual tokens to attend to language tokens, V2L enables visual evidence to influence text generation while minimizing the reverse influence of language priors on visual processing.

• This directional intervention aligns with the causal intuition that visual evidence should inform language generation, but language stereotypes should not distort visual perception.

Mathematical formulation. Consider the attention mechanism under different patterns:

• V2L: Attn $( Q _ { \mathrm { l a n g } } , K _ { \mathrm { v i s } } , V _ { \mathrm { v i s } } )$ allows language queries to attend to visual keys/values.

• L2V: $\mathrm { A t t n } ( Q _ { \mathrm { v i s } } , K _ { \mathrm { l a n g } } , V _ { \mathrm { l a n g } } )$ allows visual queries to attend to language keys/values.

• Bidirectional: Both directions are enabled.

The V2L pattern provides the optimal trade-off because:

$$
P ( T _ { \mathrm { o u t } } \mid d o ( V = v ) ) \approx P ( T _ { \mathrm { o u t } } \mid V = v , \mathrm { V 2 L } )\tag{11}
$$

where the post-intervention distribution of text given visual intervention is best approximated by allowing visual information to flow into language generation, but not vice versa.

Empirical validation. Table 3 provides experimental confirmation of this theoretical intuition:

• V2L achieves high coverage (95.8%) while maintaining excellent bias score (0.9908), indicating it neither causes evasion nor compromises accuracy.

• Language-only shows strong male bias (logratio = 0.5371) due to unmitigated language priors.

• Visual-only has low coverage (9.3%) as it isolates visual information excessively, hindering text generation.

• L2V exhibits negligible coverage (0.2%) as it primarily allows language to influence visual processing rather than vice versa.

• Bidirectional shows intermediate performance, confirming that allowing language-tovisual attention reintroduces bias pathways.

Connection to causal theory. The superiority of V2L can be understood through d-separation analysis:

• V2L suppresses backdoor paths $U  W $ $Z$ by limiting $W  { \mathbf { \bar { s } } }$ influence on $Z$ while preserving $V  Z$ paths.

• L2V fails because it allows $U \to W \to Z$ paths to remain open while blocking $V  Z$ paths.

• Bidirectional fails because it leaves both forward and backward paths open.

Conclusion. The V2L pattern is not a post-hoc empirical selection but a theoretically motivated choice derived from our causal analysis of bias mechanisms in LVLMs. Its empirical superiority (Table 3) validates the theoretical prediction that amplifying visual evidence while suppressing linguistic stereotypes provides the optimal balance for gender bias mitigation.

## E Ethical Considerations

This research aims to mitigate gender bias in large vision-language models (LVLMs), contributing to fairer and more equitable AI systems. By reducing stereotypical associations in model outputs, ViD helps prevent the amplification of societal biases through automated systems. Our work uses publicly available benchmark datasets (FACET, MS COCO, FairFace, POPE, MMMU) with appropriate licenses and privacy protections. The proposed intervention operates during inference without modifying model parameters, preserving user privacy and model integrity. However, as with any bias mitigation technique, potential misuse scenarios exist: for instance, adversaries could attempt to reverse-engineer the intervention to amplify rather than reduce bias. We emphasize that ViD should be deployed with proper safeguards and ongoing monitoring to ensure its intended fairness benefits. Future work should consider broader societal impacts, including intersectional biases and cultural variations in gender perception, to develop more inclusive debiasing frameworks.