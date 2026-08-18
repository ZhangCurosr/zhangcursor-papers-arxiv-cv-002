# Ultra: Unsupervised Cross-Task Optimization for Reliable Restoration Segmentation Collaboration under Adverse Weather

Shiqin Wang<sup>1</sup> Zhiqian Li<sup>1</sup> Haoyuan Du<sup>1</sup> Junming Chen<sup>1</sup> Jiayuan Li <sup>2,</sup> <sup>4</sup> Tianrun Xu <sup>3,</sup> <sup>4</sup> Haoyang Chen <sup>1,4</sup>

<sup>1</sup>Wuhan University <sup>2</sup>Beijing Institute of Technology <sup>3</sup>Tsinghua University <sup>4</sup>Zhongguancun Academy shiqinwang@whu.edu.cn, haoyangchen@whu.edu.cn

## Abstract

Unsupervised Domain Adaptationfor Adverse Weather Semantic Segmentation (UDA-ASS) aims to transfer semantic knowledge from labeled normal-weather images to unlabeled adverse environments. Existing approaches implicitly assume that restoration and segmentation provide mutually beneficial guidance. However, under severe degradation and without target-domain supervision, the validity of cross-task optimization directions becomes fundamentally unidentifiable, leading to hallucination-driven error propagation. In this work, we propose a novel Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra), which reframes cross-task interaction as direction selection under uncertainty and causal effect estimation, enabling reliable collaboration through candidate direction generation and intervention-based filtering. In detail, we propose CTDN and CMIL. The former exploits complementary visual structures and semantic information to generate candidate optimization directions and performs cooperative direction selection between restoration and segmentation. The latter reformulates cross-task information transferfrom correlation-based propagation into causal effect assessment, suppressing hallucination propagation. Extensive experiments on three widely used UDA-ASS benchmarks demonstrate state-ofthe-art segmentation performance. Beyond segmentation, ourframework achieves better unsupervised restoration results than existing UDA-ASS restoration methods and generalizes to unsupervised restoration and object detection collaboration tasks. Code and models will be available at https://github.com/Wang-Shiqin/Ultra.

## 1. Introduction

Adverse weather Semantic Segmentation (ASS) aims to assign each pixel in adverse weather images, including rain, fog, snow, and low illumination conditions, to a corresponding object category. It has been widely applied in autonomous driving [13, 25], visual surveillance [24, 31, 32], and robotic perception [29]. While real-world adverse weather images can be collected easily, building matching datasets with precise pixel-level annotations takes enormous time and effort. This greatly restricts the advancement of traditional supervised learning methods for ASS. Some studies attempt to transfer semantic segmentation techniques developed under clear daytime conditions [27, 37, 38, 40]. However, the substantial domain gap between clear-weather and adverse-weather scenarios leads to severe performance degradation in challenging weather environments. To reduce annotation requirements and bridge the domain gap, Unsupervised Domain Adaptation for Adverse Weather Semantic Segmentation (UDA-ASS) has attracted increasing attention by transferring knowledge from labeled daytime source domains to unlabeled adverse-weather target domains.

![](images/5ec1763ca5304c5afe651a2dff615cc379219c2764dba40e46c1d07416333800.jpg)  
Figure 1. Challenges in unsupervised restoration-segmentation collaboration: (a) Ambiguous cross-task optimization directions due to the lack of reliable supervision in heterogeneous task spaces; (b) Self-reinforcing hallucination loops caused by iterative propagation of unreliable cross-task biases.

Existing UDA-ASS methods mainly follow two technical paradigms. The first paradigm focuses on enhancing cross-domain consistency learning to facilitate domain adaptation [3–5, 13, 16, 18, 22, 30]. However, enhancing prediction consistency makes the model overly rely on daytime priors and cannot guarantee that the predictions are truly supported by the visual evidence from adverseweather images. The second paradigm reduces the visual discrepancy between source and target domains through style translation [21, 43] or image/feature restoration [12, 14, 36]. Nevertheless, such restoration processes typically rely on natural image statistics as priors to compensate for missing information, which may introduce visually plausible but semantically nonexistent textures and incorrect structures, thereby misleading semantic reasoning and causing high-confidence hallucinated predictions.

These limitations suggest that restoration in UDA-ASS should not be treated as a task-agnostic preprocessing step: whether recovered observations are useful is determined by whether they support reliable semantic decisions. Once restoration is tied to semantic prediction, the two tasks become interdependent. Segmentation can discourage appearance-oriented recovery, while restoration can strengthen the evidence available to segmentation when target-domain cues are weak.

However, unlike previous work on supervised learning [8], directly performing mutual guidance between restoration and segmentation under the unsupervised setting faces two critical challenges. (1) Ambiguous Cross-task Optimization Direction. Since UDA tasks simultaneously lack clean image supervision required by restoration and semantic annotations required by segmentation, both tasks cannot determine whether their current optimization directions are reliable, resulting in adaptation trajectories that may deviate from the desired objectives during mutual guidance, as shown in Figure 1 (a). (2) Self-reinforcing Hallucination Loop. Severe visual degradation leads restoration and segmentation models to rely on unreliable priors, resulting in hallucinated restoration structures and erroneous semantic predictions. During mutual interaction, these unreliable task-specific biases are repeatedly propagated and amplified, forming a closed-loop error accumulation process, as shown in Figure 1 (b).

To address these challenges, we propose a novel Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra) that enables reliable crosstask cooperation by jointly discovering cooperative optimization directions and regulating task interactions. In detail, we propose Cross-Task Direction Negotiation (CTDN) for identifying cooperative optimization trajectories. Specifically, we first transform visual structural priors and semantic discriminative information into mutually constrained candidate update directions via bidirectional task adaptation between restoration and segmentation. Then, restoration and segmentation are modeled as cooperative agents with distinct optimization utilities. We construct a Nash bargaining process within the candidate update direction space to negotiate optimal update directions. These directions maximize the joint utility while balancing improved restoration quality and enhanced semantic discrimination. To alleviate cross-task hallucination propagation caused by severe visual evidence degradation, we further propose Causal Mutual Intervention Learning (CMIL) module for preventing unreliable knowledge propagation between restoration and segmentation. In detail, we reformulate cross-task information propagation as task-level intervention effect estimation. By modeling the state evolution of a target task after the intervention of source-task factors, CMIL identifies cross-task intervention directions with positive optimization effects and suppresses negative information propagation caused by restoration hallucination and segmentation hallucination, thereby achieving reliable restoration-segmentation collaborative adaptation.

Extensive experiments on three widely used UDA-ASS benchmarks, together with image restoration evaluations in the UDA-ASS setting, demonstrate state-of-the-art segmentation and restoration performance. Besides, the proposed framework is not restricted to specific task structures. Instead, it provides a general unsupervised complementarytask collaboration paradigm by learning reliable guidance relationships between different tasks. Therefore, the framework can be naturally extended to other task combinations, such as restoration-object detection collaboration. When extended to UDA adverse-weather object detection, our framework consistently improves existing UDA detection methods by enabling restoration-detection collaboration, demonstrating its effectiveness and generalization capability.

The contributions of this work are summarized as follows:

• We identify and formulate the problem of unsupervised restoration-segmentation collaboration under adverse weather adaptation, where the absence of targetdomain restoration and semantic supervision causes two fundamental challenges: unreliable cross-task optimization directions and hallucination-induced error accumulation during mutual guidance.

• We propose Ultra, an Unsupervised Restoration-Segmentation Collaborative Learning Framework. It contains two core components: CTDN and CMIL. CTDN generates complementary candidate directions via cross-task interaction and uses Nash bargaining to select the optimal cross-task optimization direction. CMIL models cross-task knowledge exchange as task-level causal interventions to reduce error propagation caused by hallucinations.

• Extensive experiments on three widely used UDA-ASS benchmarks demonstrate that our Ultra achieves stateof-the-art semantic segmentation performance. Moreover, Ultra significantly improves the quality of unsupervised image restoration over existing UDA-ASS methods. When applied to existing UDA adverse-weather object detection frameworks, our method achieves consistent performance improvements, validating its effectiveness and generalization ability.

## 2. Related Works

## 2.1. Adverse Weather Semantic Segmentation

This task performs pixel-wise semantic parsing on images captured under degraded conditions (e.g., fog, night, snow, and rain). Because collecting and annotating adverse weather data is substantially more costly than for clear daytime scenes, one group of studies resorts to clear-to-adverse domain generalization, where a segmentation model trained solely on a clear source domain is expected to generalize to arbitrary unseen adverse scenes [1, 2, 27, 35, 37, 38, 41]. Nevertheless, without observing adverse-weather data during training, these models still suffer considerable accuracy loss on adverse scenes. UDA-ASS thus serves as a more practical alternative, which exploits unlabeled adverse weather images and adapts the semantic knowledge learned in the labeled clear domain to the target domain. Research on UDA-ASS has evolved along two directions. The first direction pursues cross-domain consistency, either via learning more consistent feature representations [3, 4, 13, 16, 22, 26, 30] or via enforcing prediction consistency across domains [5]; although such consistency benefits adaptation robustness, whether the resulting target predictions are grounded in reliable visual evidence remains unverified. The second direction narrows the domain shift at the appearance level, adopting weather style translation [21, 43] or image/feature restoration [12, 14, 36] to pull adverse-domain representations closer to the clear domain. Since both directions concentrate on aligning distributions or enhancing appearance, they remain prone to segmentation hallucinations once visual evidence is severely degraded or missing.

## 2.2. Task-aware Image Restoration and Segmentation

Task-oriented Image Restoration methods aim to enhance downstream task performance, DIP dynamically adapts image processing according to degradation factors [15]. VDR-IR unifies semantic representations of diverse degraded images to effectively recover intrinsic semantics [34]. Complementary to these approaches, another line of Segmentation-Guided Restoration works investigates how semantic information from segmentation models can guide image restoration [28, 39]. However, existing methods mainly focus on unidirectional information transfer between image restoration and semantic segmentation [33, 34]. Consequently, the restoration or segmentation results heavily depend on the reliability of the restoration or segmentation process, leading to suboptimal restoration or segmentation performance. To this end, Guan et al. first proposed a fully-supervised Restoration Adaptation for Semantic Segmentation (RASS) method, which enables joint optimization and mutual promotion between image restoration and semantic segmentation [8]. However, this fully-supervised paradigm requires paired restorationsegmentation supervision, which is unavailable in unsupervised adverse-domain adaptation scenarios.

## 3. Methodology

In this paper, we propose an Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra), as shown in Figure 2. During unsupervised domain adaptation, cross-domain mixed samples are jointly optimized for unsupervised image restoration and semantic segmentation. We propose CTDN, which enables mutual guidance between semantic segmentation and image restoration to generate candidate optimization directions and incorporates UNB to select the optimal cross-task direction. Finally, to address the Self-reinforcing Hallucination Loop, we design CMIL to estimate the causal effects of cross-task features, thereby preventing hallucination propagation.

## 3.1. Cross-Task Direction Negotiation (CTDN)

The segmentation branch provides category-aware spatial layouts and semantic structures that are complementary to the low-level appearance cues exploited by the diffusion restoration model. However, directly injecting segmentation features may introduce unreliable pseudo-semantic information under the unsupervised target domain. To this end, we propose a Segmentation-Guided Restoration Adapter (SGRA), which transforms segmentation representations into reliability-aware spatial-channel modulation signals for the diffusion denoising process. In detail, given the i-th stage segmentation feature $S _ { i } ,$ , we first obtain the semantic prior:

$$
P _ { i } ^ { S } = \psi _ { i } \left( [ \mathrm { D W C o n v } _ { 3 \times 3 } ( \phi _ { i } ( S _ { i } ) ) | | \mathrm { D D W C o n v } _ { 3 \times 3 } ( \phi _ { i } ( S _ { i } ) ) ] \right)\tag{1}
$$

where $\phi _ { i } ( \cdot )$ denotes a $1 \times 1$ projection layer for channel alignment, $\mathrm { D W C o n v } _ { 3 \times 3 } ( \cdot )$ and $\mathrm { D D W C o n v _ { 3 } } { \times } 3 \AA ( \cdot )$ denotes a $3 \times 3$ Depth-wise Convolution and a $3 \times 3$ Dilated Depth-wise Convolution, respectively, and $\psi _ { i } ( \cdot )$ denotes a lightweight fusion projection layer. ∥ means connection operation.

After semantic prior adaptation, a semantic prior modulation projection module is used to generate spatially varying scaling $( \gamma _ { i } )$ and shifting terms $( \beta _ { i } )$ from $P _ { i } ^ { S }$ . Then, conditional modulation is performed on the prior restoration feature as follows:

$$
R _ { i } ^ { + } = R _ { i } + \eta _ { i } ^ { S \to R } \left( R _ { i } \odot ( 1 + \lambda \operatorname { t a n h } ( \gamma _ { i } ) ) + \lambda \operatorname { t a n h } ( \beta _ { i } ) - R _ { i } \right) ,\tag{2}
$$

![](images/94babd396ff5f85208460a75bebea5d4c0d6fbab50e1ca1f69b80f4c2b1f624f.jpg)  
Figure 2. Overview of the proposed Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra). Given mixed images from cross-domain sampling, Ultra jointly optimizes image restoration and semantic segmentation through bidirectional cross-task guidance. Specifically, segmentation features $S _ { i }$ are transformed into semantic priors $P _ { i } ^ { S }$ by the SGRA, which performs conditional modulation on restoration features to provide potential restoration optimization directions. The modulated restoration features $R _ { i } ^ { + }$ are further converted into restoration priors $P _ { i } ^ { R }$ through the RGSA, providing potential segmentation optimization directions via conditional modulation. UNB selects the optimal cross-task optimization directions during training. To mitigate the Self-reinforcing Hal lucination Loop in unsupervised cross-task collaboration, CMIL estimates the causal effects of semantic and restoration interventions and only propagates beneficial interventions.

where λ controls the modulation magnitude, $\eta _ { i } ^ { S \to R }$ is a zero-initialized residual coefficient. Then, the segmentation branch provides a potential optimization direction for the restoration branch at the i-th stage, denoted as $\Delta _ { i } ^ { S  R } =$ $R _ { i } ^ { + } - R _ { i }$

The restoration branch reconstructs fine-grained structural details and illumination-invariant contents that are complementary to the category-level semantics exploited by the segmentation model. However, directly injecting restoration features may transfer low-level appearance statistics, such as color and illumination, into the segmentation classifier. To address this issue, we propose a Restoration-Guided Segmentation Adapter (RGSA), which transforms restoration representations into boundary-aware spatial-channel modulation signals for the segmentation features.

Given the i-th stage restoration feature $R _ { i } ^ { + }$ and segmentation feature $S _ { i }$ , we first obtain restoration prior $( P _ { i } ^ { R } ) { : }$

$$
\begin{array} { r } { \{ \begin{array} { l l } { \bar { R } _ { i } = \phi _ { i } ( \mathrm { R e s i z e } ( R _ { i } ^ { + } , S _ { i } ) ) , } \\ { L _ { i } = \mathrm { A v g P o o l } _ { 3 \times 3 } ( \bar { R } _ { i } ) , \qquad H _ { i } = \bar { R } _ { i } - L _ { i } , } \\ { E _ { i } ^ { R } = \psi _ { i } ( [ \mathrm { D W C o n v } _ { 3 \times 3 } ( H _ { i } ) ]  \mathrm { D D W C o n v } _ { 3 \times 3 } ( L _ { i } ) ] ) , } \\ { P _ { i } ^ { R } = \sigma ( \kappa _ { i } ( E _ { i } ^ { R } ) ) \odot E _ { i } ^ { R } + \phi _ { i } ( S _ { i } ) . } \end{array}  } \end{array}\tag{3}
$$

where the symbols follow Eq. 1. Then, Resize $( R _ { i } ^ { + } , S _ { i } )$ means bilinearly resizing $R _ { i } ^ { + }$ to the spatial resolution of $S _ { i }$ $L _ { i }$ and $H _ { i }$ are the low- and high-frequency components, respectively. $\sigma ( \kappa _ { i } ( \cdot ) )$ is a depthwise boundary gate that suppresses appearance-related responses.

After restoration prior adaptation, a structureconditioned modulation projection module is used to generate structure-conditioned scaling $( \gamma _ { i } ^ { R } )$ and shifting terms $( \beta _ { i } ^ { R } )$ from $P _ { i } ^ { R }$ . Then, conditional modulation is performed on the prior segmentation feature:

$$
S _ { i } ^ { + } = S _ { i } + \eta _ { i } ^ { R  S } [ S _ { i } \odot ( 1 + \lambda \operatorname { t a n h } ( \gamma _ { i } ^ { R } ) ) + \lambda \operatorname { t a n h } ( \beta _ { i } ^ { R } ) - S _ { i } ] ,\tag{4}
$$

where the coefficient $\eta _ { i } ^ { R \to S }$ is a learnable residual scale initialized to zero, ensuring that RGSA initially behaves as an identity mapping and gradually introduces restoration guidance during training. Accordingly, the restoration branch provides a potential optimization direction for the segmentation branch at stage i, defined as $\Delta _ { i } ^ { R  S } = S _ { i } ^ { + } - S _ { i } ^ { - }$

Unsupervised Nash Bargaining (UNB) To resolve this optimization conflict between unsupervised restoration and segmentation, we propose UNB to accumulate the detached segmentation gradients $\mathbf { g } _ { s }$ and restoration gradients $\mathbf { g } _ { r }$ at every backward pass, reweight them by the inverse of their EMA-smoothed task losses as a reliability estimate, and normalize them as follows:

$$
\left\{ \begin{array} { l l } { \rho _ { k } = \sqrt { \operatorname { c l a m p } \left( \frac { 1 / \bar { \mathcal { L } } _ { k } } { \frac { 1 } { 2 } \sum _ { j \in \{ s , r \} } 1 / \bar { \mathcal { L } } _ { j } } , 0 . 5 , 2 \right) } , } \\ { \hat { \mathbf { g } } _ { k } = \frac { \rho _ { k } \mathbf { g } _ { k } } { \left\| \rho _ { k } \mathbf { g } _ { k } \right\| _ { 2 } } , \qquad n _ { k } = \left\| \rho _ { k } \mathbf { g } _ { k } \right\| _ { 2 } , \qquad k \in \{ s , r \} . } \end{array} \right.\tag{5}
$$

where $\bar { \mathcal { L } } _ { k }$ denotes the EMA of the segmentation $( k { = } s )$ or restoration $\scriptstyle ( k = r )$ loss, $\rho _ { k }$ is the detached reliability weight that up-weights the better-optimized task, and $n _ { k }$ preserves the original gradient magnitude. The negotiated update is then the minimum-norm point on the convex hull of the two normalized task gradients, i.e., the Nash bargaining solution:

$$
\{ \begin{array} { l } { \boldsymbol { \alpha } ^ { * } = \mathrm { c l a m p } ( \displaystyle \frac { G _ { r r } - G _ { s r } } { G _ { s s } + G _ { r r } - 2 G _ { s r } } , \omega , 1 - \omega ) , } \\ { \nabla _ { \theta _ { A } }  ( \boldsymbol { \alpha } ^ { * } \hat { \mathbf { g } } _ { s } + ( 1 - \boldsymbol { \alpha } ^ { * } ) \hat { \mathbf { g } } _ { r } ) \cdot \displaystyle \frac { n _ { s } + n _ { r } } { 2 } . } \end{array}\tag{6}
$$

where $G _ { i j }$ is the Gram inner product between the normalized task gradients, $\alpha ^ { * }$ is the bargaining coefficient clipped to $[ \omega , 1 - \omega ]$ to avoid degenerate single-task solutions, and the resulting direction overwrites the raw gradients on the shared parameters $\theta _ { \mathcal { A } }$ before the optimizer step. Since $\alpha ^ { * }$ is derived from the gradient geometry rather than tuned by hand, the negotiated direction remains a valid descent direction for both objectives: when the two tasks cooperate $( G _ { s r } > 0 )$ the solution amplifies their consensus, and when they conflict $( G _ { s r } \ < \ 0 )$ it falls back to the direction that minimally violates either task.

## 3.2. Causal Mutual Intervention Learning (CMIL)

Considering that segmentation features from the unlabeled target domain inevitably contain pseudo-semantic noise under the unsupervised setting, blindly back-propagating restoration loss through such interventions may cause unreliable semantic gradients to dominate the restoration network. Therefore, we first estimate the causal effect of semantic interventions before gradient admission.

We use the restoration objective itself as the energy function and simulate one descent step for two parallel states. Let $\scriptstyle { \hat { x } } _ { 0 }$ and $\hat { x } _ { 0 } ^ { c }$ denote the predictions generated with and without the SGRA intervention, respectively. Both states are detached from the training graph and optimized by one gradient step on the restoration energy $\mathcal { E } ^ { R } ( \cdot )$ The causal effect of the semantic intervention $e ^ { { \breve { S } } \bar { \to } R }$ is then estimated by the counterfactual energy gap as follows:

$$
e ^ { S \to R } = \mathrm { s g } \big [ \mathcal { E } ^ { R } \big ( \hat { x } _ { 0 } ^ { c } - \epsilon \nabla _ { \hat { x } _ { 0 } ^ { c } } \mathcal { E } ^ { R } ( \hat { x } _ { 0 } ^ { c } ) \big ) - \mathcal { E } ^ { R } \big ( \hat { x } _ { 0 } - \epsilon \nabla _ { \hat { x } _ { 0 } } \mathcal { E } ^ { R } ( \hat { x } _ { 0 } ) \big ) \big ] ,\tag{7}
$$

where ϵ is the simulation step size, ∇ denotes the first-order gradient with respect to the detached state, sg[·] is the stopgradient operator. Intuitively, $e ^ { S \to R } > 0$ indicates that the restoration energy descends faster after the semantic intervention, i.e., the direction $\Delta _ { i } ^ { S  R }$ provided by the segmentation branch is genuinely beneficial for restoration.

The estimated effect is converted into a bounded gate, which controls both a directional regularizer and the mixture between treated and control gradients as follows:

$$
\left\{ \begin{array} { l l } { \displaystyle g ^ { S \to R } = \mathrm { s g } \biggl [ \mathrm { R e L U } \left( \operatorname { t a n h } \left( \frac { e ^ { S \to R } } { \operatorname* { m a x } ( | e ^ { S \to R } | , \tau ) } \right) \right) \biggr ] , } \\ { \displaystyle \mathcal { L } _ { s a f e } ^ { S \to R } = g ^ { S \to R } \left( 1 - \cos \bigl ( \hat { x } _ { 0 } - \hat { x } _ { 0 } ^ { c } , - \nabla _ { \hat { x } _ { 0 } ^ { c } } \mathcal { E } ^ { R } ( \hat { x } _ { 0 } ^ { c } ) \bigr ) \right) + \left( 1 - g ^ { S \to R } \right) \| \hat { x } _ { 0 } - \hat { x } _ { 0 } ^ { c } \| _ { 2 } ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { r e s t } = \left( \frac { 1 } { 2 } + \frac { 1 } { 2 } g ^ { S \to R } \right) \mathcal { E } ^ { R } ( \hat { x } _ { 0 } ) + \left( \frac { 1 } { 2 } - \frac { 1 } { 2 } g ^ { S \to R } \right) \mathrm { s g } \bigl [ \mathcal { E } ^ { R } ( \hat { x } _ { 0 } ^ { c } ) \bigr ] . } \end{array} \right.\tag{8}
$$

where τ is the effect temperature that normalizes the gate into [0, 1), cos(·, ·) denotes cosine similarity between the image-space intervention $\hat { x } _ { 0 } - \hat { x } _ { 0 } ^ { c } , i . e .$ , the downstream manifestation of $\Delta _ { i } ^ { S  R }$ , and the energy descent direction. Beneficial interventions $( g ^ { S  R }  \bar { 1 ) }$ are encouraged to align with the descent direction and their gradients are fully admitted; harmful interventions $( g ^ { S  \check { R _ { \mathrm { \scriptsize ~  ~ 0 ) } } } }$ are pulled back in magnitude.

Algorithm 1 Algorithm of Ultra.   
1: Initialize: Segmentation network $G _ { \theta _ { S } }$ , restoration net  
work $R _ { \theta _ { R } } ,$ UNB, CMIL module C, shared parameters   
$\theta _ { \mathcal { A } }$   
2: for $t \gets 1$ to T do   
3: Generate $\Delta _ { i } ^ { S  R } = R _ { i } ^ { + } - R _ { i }$ ▷ Eq. (1), (2)   
4: Generate $\Delta _ { i } ^ { \dot { R }  S } = S _ { i } ^ { \dot { + } } - S _ { i }$ ▷ Eq. (3), (4)   
5: Select optimal optimization direction ▷ Eq. (5), (6)   
6: Causal intervention for semantic and restoration in  
jection $e ^ { S \to R } , e ^ { R \to S }$ ▷ Eq. (7), (9)   
7: Perform interventions ▷ Eq. (8), (10)   
8: Update $S _ { i }$ with $\hat { S } _ { i } = S _ { i } + g ^ { R  S } \Delta _ { i } ^ { \hat { R }  S }$   
9: Update $\theta _ { S } , \theta _ { R } , \theta _ { A }$ by minimizing $\mathcal { L } = \mathcal { L } _ { S } + \lambda _ { R } \mathcal { L } _ { R } +$   
$\lambda _ { C } \mathcal { L } _ { C M I L }$   
10: end for

Symmetrically, the structural refinement $\Delta _ { i } ^ { R  S }$ produced by RGSA is examined before entering the segmentation loss, since restoration features may carry appearance statistics that are harmful to the classifier. The key difference from Eq. 7 is that no target-domain ground truth is available, so the review is conducted on an unsupervised segmentation energy $\mathcal { E } ^ { S } ( \cdot )$ , defined as the normalized prediction entropy plus the confidence-weighted cross-entropy against trusted pseudo-labels.

For each stage i, the factual feature S and the refined candidate $S _ { i } ^ { + }$ are detached into a control state $C _ { i } = \mathrm { s g } [ S _ { i } ]$ and a treated state $T _ { i } ~ = ~ \mathrm { s g } [ S _ { i } + \Delta _ { i } ^ { R  S } ]$ . Each state is advanced by one simulated descent step on $\mathcal { E } ^ { S } ( \cdot )$ , and the causal effect of adopting the refinement is estimated over all N stages as:

$$
e ^ { R  S } = \mathrm { s g } \Bigg [ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( { \mathcal { E } } ^ { S } ( C _ { i } - \epsilon \nabla _ { C _ { i } } { \mathcal { E } } ^ { S } ( \{ C _ { i } \} ) ) - { \mathcal { E } } ^ { S } ( T _ { i } - \epsilon \nabla _ { T _ { i } } { \mathcal { E } } ^ { S } ( \{ T _ { i } \} ) ) ) \Bigg ] \ : ,\tag{9}
$$

Intuitively, $e ^ { R \to S } > 0$ answers the question “what would happen at the next optimization step if the refinement were adopted”: the segmentation energy would descend faster, $i . e .$ , the restoration prior $P _ { i } ^ { R }$ provides structural information that the segmentation branch itself cannot obtain from the unlabeled target domain.

The refinement is accepted in proportion to its estimated effect, while a safe loss shapes the live intervention produced by RGSA as follows:

$$
\left\{ \begin{array} { l l } { \displaystyle g ^ { R \to S } = \mathrm { s g } \bigg [ \mathrm { R e L U } \bigg ( \operatorname { t a n h } \bigg ( \frac { e ^ { R \to S } } { \operatorname* { m a x } \{ | e ^ { R \to S } | , \tau ) \} } \bigg ) \bigg ] , } \\ { \displaystyle \mathcal { L } _ { s a f e } ^ { R \to S } = g ^ { R \to S } \sum _ { i = 1 } ^ { N } \big ( 1 - \cos \big ( \Delta _ { i } ^ { R \to S } , - \nabla _ { C _ { i } } \mathcal { E } ^ { S } ( \{ C _ { i } \} ) \big ) \big ) + \big ( 1 - g ^ { R \to S } \big ) \sum _ { i = 1 } ^ { N } \big \| \Delta _ { i } ^ { R \to S } \big \| _ { 2 } ^ { 2 } . } \end{array} \right.\tag{10}
$$

<table><tr><td>Method</td><td>Backbone</td><td>ACDC test</td><td>ACDC val</td></tr><tr><td>DeepLabV2 [6]</td><td>DeepLabV2</td><td>38.0</td><td>34.6</td></tr><tr><td>Refign [4]</td><td>DeepLabV2</td><td>48.0</td><td>55.8</td></tr><tr><td>CMA [3]</td><td>DeepLabV2</td><td>50.4</td><td>48.8</td></tr><tr><td>VBLC [14]</td><td>DeepLabV2</td><td>47.8</td><td>46.0</td></tr><tr><td>Ultra (Ours)</td><td>DeepLabV2</td><td>58.6</td><td>56.7</td></tr><tr><td>DAFormer [9]</td><td>DAFormer</td><td>55.3</td><td>55.3</td></tr><tr><td>Refign [4]</td><td>DAFormer</td><td>65.5</td><td>65.0</td></tr><tr><td>VBLC [14]</td><td>DAFormer</td><td>64.2</td><td>63.7</td></tr><tr><td>CoPT [17]</td><td>DAFormer</td><td>63.7</td><td>64.8</td></tr><tr><td>Instance-Warp [42]</td><td>DAFormer</td><td>61.7</td><td>61.8</td></tr><tr><td>Ultra (Ours)</td><td>DAFormer</td><td>65.5</td><td>65.6</td></tr><tr><td>HRDA [10]</td><td>HRDA</td><td>68.0</td><td>65.2</td></tr><tr><td>Refign [4]</td><td>HRDA</td><td>72.1</td><td>71.1</td></tr><tr><td>VBLC [14]</td><td>HRDA</td><td>67.2</td><td>66.5</td></tr><tr><td>CoDA [7]</td><td>HRDA</td><td>72.6</td><td>72.4</td></tr><tr><td>ACSegFormer [16]</td><td>HRDA</td><td>72.7</td><td>72.6</td></tr><tr><td>CISS [20]</td><td>HRDA</td><td>69.6</td><td>68.7</td></tr><tr><td>Ultra (Ours)</td><td>HRDA</td><td>73.0</td><td>73.0</td></tr></table>

Table 1. Comparison of the state-of-the-art in Cityscapes→ACDC domain adaptation on the ACDC test and ACDC val sets. The evaluation metric is mIoU, where a higher value indicates better performance. The best results are presented in bold.

where the symbols follow Eq. 8. Then, ${ \hat { S } } _ { i } ~ = ~ S _ { i } +$ $g ^ { R  S } \Delta _ { i } ^ { R  \overline { { S } } }$ is the accepted segmentation feature that replaces $S _ { i }$ in the subsequent decoder. Since $g ^ { R \to S }$ is detached, the gate itself cannot become a learnable shortcut; instead, $\mathcal { L } _ { s a f e } ^ { R  S }$ pulls the live intervention $\Delta _ { i } ^ { R  S }$ toward the segmentation descent direction when the refinement is beneficial, and suppresses its magnitude back to zero otherwise. The final modulated features $\{ \hat { S } _ { i } \}$ , together with the source-supervised loss, are then used to compute the segmentation objectives, and $\mathcal { L } _ { s a f e } ^ { R  S }$ is back-propagated with a small weight as an additional segmentation-side regularizer. The full training procedure is outlined in Algorithm 1.

## 4. Experiments

We adopt mean Intersection-over-Union (mIoU) as the segmentation evaluation metric, where higher scores indicate superior segmentation performance. We assess our method under three settings: 1) UDA-ASS with Cityscapes→ACDC adaptation; 2) UDA night-time semantic segmentation with Cityscapes→Dark Zurich adaptation; and 3) image restoration performance in the UDA-ASS scenario. Furthermore, we investigate the generalization ability of our approach to UDA object detection, specifically on the BDD100K Clear→ACDC Object Detection benchmark.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td>Dark Zurich-val</td><td>Nighttime Driving</td></tr><tr><td>mIoU</td><td>mIoU</td></tr><tr><td>DAFormer [9]</td><td>DAFormer</td><td>37.1</td><td>54.0</td></tr><tr><td>InforMS [23]</td><td>DAFormer</td><td>45.1</td><td>56.0</td></tr><tr><td>Ultra (Ours)</td><td>DAFormer</td><td>45.8</td><td>56.8</td></tr><tr><td>HRDA [10]</td><td>HRDA</td><td>42.1</td><td>54.1</td></tr><tr><td>InforMS [23]</td><td>HRDA</td><td>52.5</td><td>58.5</td></tr><tr><td>Ultra (Ours)</td><td>HRDA</td><td>52.8</td><td>59.3</td></tr></table>

Table 2. Comparison with state-of-the-art methods on the Dark Zurich-val set and Nighttime Driving test set when performing Cityscapes→Dark Zurich domain adaptation.

## 4.1. Experimental settings

Our framework is implemented in PyTorch and trained on a single NVIDIA 80 GB A100 GPU. We evaluate our approach with several segmentation backbones, including DeepLabV2 [6], DAFormer [9], and HRDA [10]. The training process consists of 60k iterations, where 1024×1024 image crops are randomly sampled from the Cityscapes and ACDC datasets. We optimize the model using AdamW with a weight decay of $1 \times 1 0 ^ { - 4 }$ . To alleviate the long-tail category imbalance in the source domain, we employ the rare class sampling strategy introduced in [9] with $\alpha = 0 . 9 9 9$

## 4.2. Comparison with State-of-the-art Methods

## 4.2.1. Comparison on ACDC

We present comparisons to several kinds of semantic segmentation methods, including 1) backbones: DeepLabv2 [6], DAFormer [9], and HRDA [10]; 2) UDA-SS methods under Adverse Weather: Refign [4], CMA [3], VBLC [14], CoDA [7], and ACSegFormer [16]; and 3) general UDA-SS methods: CISS [20], CoPT [17] and Instance-Warp [42]; The quantitative results of mIoU performance on the ACDC test set and the ACDC val set are reported in Table 1. We observe that our method consistently outperforms existing approaches on the same backbone on the ACDC test and val sets. Notably, with HRDA as the backbone, our method achieves 73.0 [%] mIoU on both the ACDC test and val sets, establishing state-of-the-art performance for Cityscapes → ACDC domain adaptation.

Figure 3 provides qualitative comparisons between our method and existing UDA-ASS approaches on the ACDC val set. Under the same backbone setting, our method produces segmentation results that are more consistent with the ground-truth labels than competing methods. In particular, our method based on HRDA achieves the best performance, highlighting the effectiveness of the proposed framework.

## 4.2.2. Comparison on Dark Zurich

To evaluate the effectiveness of our framework under nighttime scenarios, we conduct experiments on the Cityscapes→Dark Zurich domain adaptation setting and report the performance on the Dark Zurich val dataset in Table 2. Our HRDA-based model achieves 52.8 [%] mIoU, surpassing previous approaches and establishing state-ofthe-art performance. As shown in Figure 4, our method produces more accurate and visually consistent predictions compared with existing methods, confirming its effectiveness in challenging nighttime environments.

![](images/816b7c8048e3b5fe1606e4688ad6d7b9032a3586944691ce70eb587b321d4720.jpg)  
Figure 3. Qualitative comparison of our method and existing UDA-ASS approaches with DeepLabV2, DAFormer, and HRDA backbones on the ACDC val set. Under the same backbone configurations, our method generates predictions that better match the ground-truth annotations compared with previous approaches (e.g., Refign, CMA, VBLC, and ACSegFormer). The HRDA-based model achieves the best segmentation quality, further demonstrating the effectiveness of the proposed framework.

![](images/b02083650b59f94ffc7790530929e799e3765a39b16570b9594b59c606e85385.jpg)  
Figure 4. Qualitative comparison with existing UDA-ASS approaches using DAFormer and HRDA backbones on Dark Zurich val (top two rows) and Nighttime Driving test (bottom two rows).

## 4.2.3. Comparison on Nighttime Driving

To further examine the generalization ability of our framework under nighttime conditions, we evaluate the Cityscapes→Dark Zurich adaptation model on the Nighttime Driving test set, with quantitative results reported in Table 2 and qualitative examples shown in Figure 4. Using HRDA as the segmentation backbone, our method achieves the best performance on this benchmark, reaching 59.3 [%] mIoU. These results demonstrate the robustness of our framework in handling challenging nighttime scenarios.

## 4.2.4. Comparison with Image Restoration in UDA-ASS

To validate the effectiveness of our unsupervised image restoration method, we compare it with the image restoration algorithms used in existing UDA-ASS methods, such as VBLC, which follows a restore-then-segment paradigm. The quantitative and qualitative results of unsupervised image restoration on the ACDC val set are reported in Table 3 and Figure 5, respectively. Different from conventional static restoration methods, our framework leverages cross-task optimization to achieve superior image restoration quality, further validating the effectiveness of the proposed approach.

<table><tr><td rowspan="2">Method</td><td>DeepLabV2</td><td>DAFormer</td></tr><tr><td>LOE↓</td><td>LOE↓</td></tr><tr><td>VBLC [14]</td><td>37.90</td><td>37.90</td></tr><tr><td>Ultra (Ours)</td><td>30.26</td><td>29.98</td></tr></table>

Table 3. Comparison of image restoration performance between our method and existing UDA-ASS restoration methods on the ACDC val set. The no-reference LOE metric is adopted, where lower values indicate better performance.

![](images/daf2aae3c1fd2c8951ce9f5257edfb1ece78fa29ac85fedba83b5c3ce5f00e8a.jpg)  
Figure 5. Qualitative comparison of Ultra (ours) and VBLC on ACDC val. Our method produces more natural restored images that are closer to the coarse-aligned target-domain reference images provided by ACDC [19].

<table><tr><td colspan="2">Cross-Task Direction Negotiation (CTDN)</td><td rowspan="2">Causal Mutual Intervention Learning (CMIL)</td><td rowspan="2">ACDC val</td></tr><tr><td>w/o Unsupervised Nash Bargaining (UNB)</td><td>w/UNB</td></tr><tr><td rowspan="3">√</td><td></td><td></td><td>65.0 (+0.0) 65.2 (+0.2)</td></tr><tr><td></td><td></td><td>65.4 (+0.4)</td></tr><tr><td></td><td></td><td>65.5 (+0.5)</td></tr><tr><td></td><td>√</td><td>√ √</td><td>65.6 (+0.6)</td></tr></table>

Table 4. Ablation Study on several model variants of our method on the ACDC val (Backbone: DAFormer).
<table><tr><td>Method</td><td>mAP↑</td><td>mAP50↑</td><td>mAP75↑</td><td>mAPs↑</td><td>mAPm↑</td></tr><tr><td>2PCNet [11]</td><td>16.6</td><td>30.8</td><td>14.9</td><td>6.0</td><td>24.8</td></tr><tr><td>2PCNet + Ultra (Ours)</td><td>18.9 (+2.3)</td><td>34.2 (+3.4)</td><td>19.3 (+4.4)</td><td>7.7 (+1.7)</td><td>27.5 (+2.7)</td></tr><tr><td>Instance-Warp [42]</td><td>17.9</td><td>32.1</td><td>18.2</td><td>6.8</td><td>26.7</td></tr><tr><td>Instance-Warp + Ultra (Ours)</td><td>18.8 (+0.9)</td><td>33.4 (+1.3)</td><td>18.6 (+0.4)</td><td>6.9 (+0.1)</td><td>28.3 (+1.6)</td></tr></table>

Table 5. Comparison of UDA object detection performance on BDD100K Clear → ACDC Object Detection on the ACDC object detection validation set. The better result within each baseline pair is shown in bold.

Ambiguous Cross-task Optimization Directions and Self-reinforcing Hallucination Loops. To address these challenges, we propose Ultra, an unsupervised cross-task optimization framework that enables restoration and segmentation to collaboratively improve through reliable task interaction. Ultra introduces CTDN to discover cooperative optimization trajectories by exploiting complementary visual restoration cues and semantic information, and further incorporates CMIL to regulate cross-task knowledge exchange through task-level intervention effects. Extensive experiments on multiple UDA-ASS benchmarks demonstrate the effectiveness of Ultra across different architectures, while its extension to unsupervised restoration and object detection collaboration tasks verifies its generalization. Our work provides a new perspective for unsupervised complementary task learning, where heterogeneous visual tasks can collaboratively optimize their objectives while maintaining reliable information interaction.

## 4.3. Ablation Study

In this section, we validate the effectiveness of the three core components. For the Cityscapes → ACDC domain adaptation task, we train our model variants with different configurations and evaluate their performance on the ACDC val set, as shown in Table 4. We first perform target-domain image restoration and semantic segmentation in parallel, and then gradually introduce the proposed components. The results show that adding CTDN without UNB improves mIoU from 65.0 to 65.2 on ACDC val, demonstrating the effectiveness of transforming visual structural priors and semantic discriminative information into mutually constrained optimization directions through bidirectional task adaptation. While enabling UNB further raises it to 65.4, indicating that the effectiveness of joint-benefitbased direction negotiation. Then, incorporating CMIL further improves the performance to 65.5. Finally, the full model achieves the best performance of 65.6 [%] mIoU, verifying the complementary roles and necessity of each component in unsupervised restoration-segmentation collaborative adaptation.

## 4.4. Generalization Study

To verify the generalization of our proposed method, we incorporate CTDN and CMIL into the original unsupervised domain adaptation object detection framework to achieve unsupervised restoration and object detection collaborative optimization. Experimental results are summarized in Table 5. Notably, our method consistently yields significant performance gains, confirming its superior generalization capability.

## 5. Conclusion

In this work, we investigate two critical challenges in unsupervised restoration-segmentation mutual guidance:

## References

[1] Qi Bi, Yixian Shen, Jingjun Yi, and Gui-Song Xia. Adadcp: Learning an adapter with discrete cosine prior for clearto-adverse domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12997–13008, 2025. 3

[2] Qi Bi, Jingjun Yi, Huimin Huang, Hao Zheng, Haolan Zhan, Yawen Huang, Yuexiang Li, Xian Wu, and Yefeng Zheng. Nightadapter: Learning a frequency adapter for generaliz able night-time scene segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23838–23849, 2025. 3

[3] David Bruggemann, Christos Sakaridis, Tim Br¨ odermann,¨ and Luc Van Gool. Contrastive model adaptation for cross condition robustness in semantic segmentation. In Proceed ings of the IEEE/CVF International Conference on Computer Vision, pages 11378–11387, Paris, France, 2023. IEEE. 1, 3, 6

[4] David Bruggemann, Christos Sakaridis, Prune Truong, and Luc Van Gool. Refign: Align and refine for adaptation of semantic segmentation to adverse conditions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3174–3184, Waikola, HI, USA, 2023. IEEE. 3, 6

[5] Chong Chen, Hui Liu, Chunsheng Liu, Faliang Chang, Tuo Li, and Penghui Hao. Mpsd-net: A novel multiple progressive self-distillation network for semantic segmentation in adverse weather. Neurocomputing, page 131286, 2025. 1, 3

[6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolu tion, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4):834–848, 2017. 6

[7] Ziyang Gong, Fuhao Li, Yupeng Deng, Deblina Bhattacharjee, Xianzheng Ma, Xiangwei Zhu, and Zhenming Ji. Coda: Instructive chain-of-domain adaptation with severity-aware

visual prompt tuning. In European Conference on Computer Vision, pages 130–148. Springer, 2024. 6

[8] Kai Guan, Rongyuan Wu, Shuai Li, Wentao Zhu, Wenjun Zeng, and Lei Zhang. Restoration adaptation for semantic segmentation on low quality images. International Journal ofComputer Vision, 134(5):229, 2026. 2, 3

[9] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Daformer: Improving network architectures and training strategies for domain-adaptive semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9924–9935, 2022. 6

[10] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Hrda: Context-aware high-resolution domain-adaptive semantic segmentation. In European conference on computer vision, pages 372–391. Springer, 2022. 6

[11] Mikhail Kennerley, Jian-Gang Wang, Bharadwaj Veeravalli, and Robby T Tan. 2pcnet: Two-phase consistency training for day-to-night unsupervised domain adaptive object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11484–11493, 2023. 8

[12] Sohyun Lee, Namyup Kim, Sungyeon Kim, and Suha Kwak. Frest: Feature restoration for semantic segmentation under multiple adverse conditions. In European Conference on Computer Vision, pages 1–18. Springer, 2025. 2, 3

[13] Fuhao Li, Ziyang Gong, Yupeng Deng, Xianzheng Ma, Renrui Zhang, Zhenming Ji, Xiangwei Zhu, and Hong Zhang. Parsing all adverse scenes: Severity-aware semantic segmentation with mask-enhanced cross-domain consistency. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13483–13491, 2024. 1, 3

[14] Mingjia Li, Binhui Xie, Shuang Li, Chi Harold Liu, and Xinjing Cheng. Vblc: visibility boosting and logit-constraint learning for domain adaptive semantic segmentation under adverse conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 8605–8613, Washington, DC, USA, 2023. 2, 3, 6, 7

[15] Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, and Lei Zhang. Image-adaptive yolo for object detection in adverse weather conditions. In Proceedings of the AAAI conference on artificial intelligence, pages 1792–1800, 2022. 3

[16] Wenyu Liu, Song Wang, Jianke Zhu, Xuansong Xie, and Lei Zhang. Domain adaptation transformer for unsupervised driving-scene segmentation in adverse conditions. IEEE Transactions on Intelligent Transportation Systems, 2024. 1, 3, 6

[17] Cristina Mata, Kanchana Ranasinghe, and Michael S Ryoo. Copt: Unsupervised domain adaptive segmentation using domain-agnostic text embeddings. In European Conference on Computer Vision, pages 424–440. Springer, 2025. 6

[18] Yuwen Pan, Rui Sun, Wangkai Li, and Tianzhu Zhang. Exploring weather-aware aggregation and adaptation for semantic segmentation under adverse conditions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13952–13962, 2025. 1

[19] Christos Sakaridis, Haoran Wang, Ke Li, Rene Zurbr ´ ugg,¨ Arpit Jadon, Wim Abbeloos, Daniel Olmeda Reino, Luc Van

Gool, and Dengxin Dai. Acdc: The adverse conditions dataset with correspondences for robust semantic driving scene perception, 2024. 7

[20] Christos Sakaridis, David Bruggemann, Fisher Yu, and Luc Van Gool. Condition-invariant semantic segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 6

[21] Fengyi Shen, Li Zhou, Kagan Kuecuekaytekin, George Basem Fouad Eskandar, Ziyuan Liu, He Wang, and Alois Knoll. W-controluda: Weather-controllable diffusionassisted unsupervised domain adaptation for semantic segmentation. IEEE Robotics and Automation Letters, 2025. 2, 3

[22] Shuochen Tian, Jian Pang, Jin Wang, Bingfeng Zhang, and Weifeng Liu. Scm: Semantic segmentation with dual-stream semantic synergy under adverse weather conditions. Multimedia Systems, 32(1):23, 2026. 1, 3

[23] Shiqin Wang, Xin Xu, Xianzheng Ma, Kui Jiang, and Zheng Wang. Informative classes matter: Towards unsupervised domain adaptive nighttime semantic segmentation. In Pro ceedings ofthe 31st ACM International Conference on Mul timedia, pages 163–172, 2023. 6

[24] Shiqin Wang, Xin Xu, Haoyang Chen, Kui Jiang, Zheng Wang, and Ke Tang. Low-light salient object detection meets the small size. IEEE Transactions on Emerging Topics in Computational Intelligence, 9(4):2754–2766, 2024. 1

[25] Shiqin Wang, Xin Xu, Haoyang Chen, Kui Jiang, and Zheng Wang. The parables of the mustard seed and the yeast: Extremely low-budget, high-performance nighttime semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7853–7861, 2025. 1

[26] Shiqin Wang, Haoyang Chen, Huaizhou Huang, Yinkan He, Dongfang Sun, Xiaoqing Chen, Xingyu Liu, Zheng Wang, and Kaiyan Zhao. Heuristic self-paced learning for domain adaptive semantic segmentation under adverse conditions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3815–3824, 2026. 3

[27] Zhixiang Wei, Lin Chen, Yi Jin, Xiaoxiao Ma, Tianle Liu, Pengyang Ling, Ben Wang, Huaian Chen, and Jinjin Zheng. Stronger fewer & superior: Harnessing vision foundation models for domain generalized semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 28619–28630, 2024. 1, 3

[28] Yuhui Wu, Guoqing Wang, Shaochong Liu, Yang Yang, Wei Li, Xiongxin Tang, Shuhang Gu, Chongyi Li, and Heng Tao Shen. Towards a flexible semantic guided model for single image enhancement and restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9921– 9939, 2024. 3

[29] Yao Wu, Mingwei Xing, Yachao Zhang, Fangyong Wang, Xiaopei Zhang, and Yanyun Qu. Beyondsparse: Facilitating mamba to enhance cross-domain 3d semantic segmentation in adverse weather. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10826–10834, 2026. 1

[30] Qin Xu, Qihang Wu, Lu Hongtao, Xiaoxia Cheng, and Bo Jiang. Crope: Cross-modal semantic compensation adapta-

tion for all adverse scene understanding. Advances in Neural Information Processing Systems, 38:136226–136252, 2026. 1, 3

[31] Xin Xu, Shiqin Wang, Zheng Wang, Xiaolong Zhang, and Ruimin Hu. Exploring image enhancement for salient object detection in low light images. ACM transactions on multimedia computing, communications, and applications (TOMM), 17(1s):1–19, 2021. 1

[32] Xin Yang, Wending Yan, Yuan Yuan, Michael Bi Mi, and Robby T Tan. Semantic segmentation on raindrop degraded images using two-stage dual teacher-student learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9292–9300, 2025. 1

[33] Zhou Yang, Weisheng Dong, Xin Li, Jinjian Wu, Leida Li, and Guangming Shi. Self-feature distillation with uncertainty modeling for degraded image recognition. In European Conference on Computer Vision, pages 552–569. Springer, 2022. 3

[34] Zizheng Yang, Jie Huang, Jiahao Chang, Man Zhou, Hu Yu, Jinghao Zhang, and Feng Zhao. Visual recognition-driven image restoration for multiple degradation with intrinsic semantics recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14059–14070, 2023. 3

[35] Xin Ye, Xiaoqi Shi, and Yuxue Li. Rfglnet for adverse weather domain-generalized semantic segmentation with frequency low-rank enhancement. Scientific Reports, 16(1): 8253, 2026. 3

[36] Zuobin Ying, Zhengcheng Lin, Zhenyu Li, Xiaochun Huang, and Weiping Ding. Scdf: Seeing clearly through dark and fog, an adaptive semantic segmentation scheme for autonomous vehicle. IEEE Transactions on Intelligent Transportation Systems, 2025. 2, 3

[37] Seokju Yun, Seunghye Chae, Dongheon Lee, and Youngmin Ro. Soma: Singular value decomposed minor components adaptation for domain generalizable representation learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25602–25612, 2025. 1, 3

[38] Qi Zang, Dong Zhao, Nan Pu, Wenjing Li, Zhun Zhong, and Meng Wang. Geco: Geometry-consistent regularization for domain generalized semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 871–881, 2026. 1, 3

[39] Xiangrui Zeng, Lingyu Zhu, Wenhan Yang, Howard Leung, Shiqi Wang, and Sam Kwong. Low-light image enhancement via diffusion models with semantic priors of any region. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 3

[40] Yin Zhang, Yongqiang Zhang, Yaoyue Zheng, Bogdan Raducanu, and Dan Liu. Causal-tune: Mining causal factors from vision foundation models for domain generalized semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12916–12924, 2026. 1

[41] Dong Zhao, Jinlong Li, Shuang Wang, Mengyao Wu, Qi Zang, Nicu Sebe, and Zhun Zhong. Fishertune: Fisherguided robust tuning of vision foundation models for domain generalized segmentation. In Proceedings of the Computer

Vision and Pattern Recognition Conference, pages 15043– 15054, 2025. 3

[42] Shen Zheng, Anurag Ghosh, and Srinivasa G Narasimhan. Instance-warp: Saliency guided image warping for unsupervised domain adaptation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8197–8206. IEEE, 2025. 6, 8

[43] Ziqiang Zheng, Yingshu Chen, Binh-Son Hua, and Sai-Kit Yeung. Compuda: Compositional unsupervised domain adaptation for semantic segmentation under adverse condi tions. In 2023 IEEE/RSJ International Conference on Intel ligent Robots and Systems, pages 7675–7681, Detroit, MI, USA, 2023. IEEE. 2, 3