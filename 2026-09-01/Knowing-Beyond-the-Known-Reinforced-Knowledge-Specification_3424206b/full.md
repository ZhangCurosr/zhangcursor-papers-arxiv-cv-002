# Knowing Beyond the Known: Reinforced Knowledge Specification for Multi-Label Class-Incremental Learning

Aoting Zhang<sup>a,c</sup>, Dongbao Yang<sup>b,∗</sup>, Chang Liu<sup>e</sup>, Xiaopeng Hong<sup>d</sup>, Can Ma<sup>a,c</sup>, Yu Zhou<sup>b,∗</sup>

<sup>a</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, 100085, China

<sup>b</sup>Nankai University, Tianjin, 300350, China

<sup>c</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, 101408, China

<sup>d</sup>Harbin Institute of Technology, Harbin, 150001, China <sup>e</sup>Tsinghua University, Beijing, 100084, China

## Abstract

Existing class-incremental learning methods struggle in multi-label scenarios (MLCIL) due to the inherent contradiction of learning objectives arising from co-occurring and incomplete labels. We argue that the core obstacle is the model’s ambiguous boundary between known and unknown knowledge, which undermines historical knowledge retention, complicates current task learning, and limits adaptability to future concepts. To address this, we propose KBK (Knowing Beyond the Known), a reinforced knowledge specification framework that explicitly models what is known or not to unify historical, current, and prospective learning. Specifically, to clarify known knowledge, we develop a hierarchical feature purification module that disentangles fine grained class-specific features from global features, where high-level semantic abstraction is reinforced with low-level visual features. Additionally, an uncertainty-aware recall enhancement strategy suppresses unreliable predictions based on distribution priors, improving the quality of historical recall. For probing the unknown, KBK leverages semantic correlations to synthesize informative unknown features under co-occurring, preserving embedding space for future learning. Furthermore, to mitigate heterogeneous forgetting,

we design a category-balanced gradient compensation loss that dynamically reweights gradient backpropagation according to forgetting speeds. Experiments on multiple benchmarks validate the efectiveness and robustness of KBK, which surpasses prior best methods by 2.7% in Avg. Acc on MS-COCO B0-C10 setting even without any replay bufers.

Keywords: Multi-Label Class-Incremental Learning, Catastrophic Forgetting, Feature Purification, Gradient Compensation

## 1. Introduction

Class-incremental learning (CIL) is designed to incrementally incorporate novel classes into a unified model without compromising its ability to recognize previously learned ones. Many eforts have focused on mitigating catastrophic forgetting, a challenge exacerbated by the unavailability of old data during continual updates. Most existing studies, however, are designed for single-label class-incremental learning (SLCIL), which assumes each image is associated with only one label. In reality, real-world images often contain multiple semantically meaningful objects, such as cars, pedestrians, and buses in urban scenes. This has spurred growing interest in multi-label class-incremental learning (MLCIL) [1], which focuses on correctly assigning images to multiple classes that appear across successive sessions.

MLCIL presents a unique challenge named partial labeling–individual images may contain objects from previously learned (historical), currently introduced, and even yet-to-be-seen (future) classes, while supervision is provided only for current session (e.g., only “car” is labeled in Fig. 1 (a)), and other co-occurring classes such as “bus” and “person” remain unannotated. Naively applying anti-forgetting remedies from single-label CIL, such as knowledge distillation and exemplar replay, proves inefective in MLCIL, since these methods fail to reconcile the conflicting learning objectives inherent to incomplete labels. Specifically, MLCIL introduces a tripartite tension among three critical learning goals–preserving previously acquired knowledge, optimizing classification for current classes, and preparing for subsequent tasks. To preserve old knowledge, recent approaches employ pseudo-labeling of prior classes and multi-knowledge retention strategies. For example, KRT [1] proposes a knowledge restoration and transfer framework to compensate for missing labels of known classes. APPLE [2] adopts adaptive pseudo-label strategy to increase the label information in the training data.

![](images/1e98a59494539c316417db9ffb24f47a132fe3b03eaf2d87ae40ee43f3066fad.jpg)  
(a) label: bus, car, person

![](images/fd52d0f88c154407494c611e81654e9caa83a5cba1de977a083930db654991ef.jpg)  
(b) entangled features

![](images/84d990d457d9cd25984d1eae5514da7b83bf228fc815da1c8c91cf2e919fa00e.jpg)  
(c) feature aliasing

Specify what is known or not  
![](images/8cfc3df6ec586724a56c7cdcdd9b32d553d60b4f0d17d6d682aabd1b76b24979.jpg)  
(d) bus

![](images/7254d88794a1f1675571cd350cf56c40c2077661351fc86706fb42f24eda1b05.jpg)

![](images/3f4b069f70f0c80b72701fcd72bbe24df0575b785aac9227b826e94c43e84782.jpg)  
(e) class-aware features  
known (Historical&Current): bus&car unknown (Prospective): person  
Figure 1: The contradiction of learning objectives arises from the model’s inability to distinguish known and unknown knowledge. Current model fails to efectively recall prior known knowledge due to (a) the absence of historical labels, while (b) unknown classes attention is inadvertently overlapped with known classes, and known classes are also entangled, resulting in (c) feature aliasing and contradictory learning objectives. By specifying what is known or not, (d) fine-grained class-aware features are focused, leading to (e) enhanced inter-class discriminability, alleviating the contradiction.

Despite these advancements, existing approaches mainly focus on historical knowledge retention or current-label completion, while the ambiguous boundary between reliable known knowledge and uncertain unknown content remains underexplored. Under partial labeling and label co-occurrence, unlabeled future classes may appear in current-session images and be mistakenly absorbed into known-class regions. This leads to representation entanglement, where the model activates unseen-class features as if they belong to known classes. As illustrated in Fig. 1 (b), although not “knowing” the person class, the model erroneously focuses on the corresponding region and conflates person features with those of known classes such as car and bus. In addition, insuficient “knowing” ability–manifested as weak discrimination between past and current classes–exacerbates feature entanglement under label co-occurrence, blurring task boundaries and intensifying learning conflicts. As a result, not only is future learning compromised, but the recall of previous knowledge is also hindered due to the absence of explicit supervision, thus exacerbating forgetting.

To address this, we propose Knowing Beyond the Known (KBK), a reinforced knowledge specification framework designed to explicitly model what is known or not to accommodate historical, current, and prospective knowledge. For clarifying known knowledge, we adopt a hierarchical feature purification module to capture fine-grained class-aware representations while suppressing irrelevant noise. Specifically, high-level semantic features are reinforced with intermediate visual cues via residual fusion, enriching spatial details while preserving semantic abstraction. Each class is presented by a learnable embedding, which interacts with fused features via an attention mechanism to locate relevant regions. It not only enhances the discrimination of co-occurring objects but also allows the embedding space to be seamlessly extended to new classes. To improve the reliability of knowledge recall, we incorporate an uncertainty-aware recall enhancement strategy that adaptively filters out low-confidence and high-uncertainty predictions based on historical distribution priors, enhancing the quality of recalled supervision. For probing unknown knowledge, KBK leverages semantically correlated absentclass embeddings to synthesize prospective features that emulate potential future classes, enabling the model to reserve embedding space for unseen concepts while strengthening discrimination among current classes. Furthermore, we design a category-balanced gradient compensation loss to mitigate the heterogeneous forgetting of old classes by reweighting gradient backpropagation. The gradients of heavily forgotten classes are amplified and those of well-remembered ones are suppressed, efectively rectifying imbalanced gradient distributions.

To summarize, our major contributions are as follows:

• We propose KBK, a unified known/unknown knowledge-specification framework for MLCIL. Instead of treating historical forgetting, currentclass learning, and future-class interference as isolated problems, KBK explicitly coordinates historical, current, and prospective knowledge by specifying reliable known knowledge and uncertain unknown content across incremental sessions.

• Under this principle, we design a Hierarchical Feature Purification module to extract class-aware known representations from co-occurring objects, and an Uncertainty-aware Recall Enhancement strategy to recover reliable historical supervision from partially labeled currentsession data.

• To model prospective knowledge, we introduce Semantic-guided Probing Unknown, which synthesizes unknown features from absent-class responses and assigns unexplained content to an explicit unknown direction. We further develop Category-balanced Gradient Compensation to alleviate heterogeneous forgetting by balancing class-wise optimization signals.

• Evaluation on diverse experimental protocols confirms that KBK attains superior performance while diminishing catastrophic forgetting.

This manuscript substantially extends the conference work [3] by intro ducing several major technical contributions and a much broader empirical evaluation: 1) The feature purification module is advanced from a flat structure to a hierarchical paradigm that facilitates bidirectional fusion between high-level semantic abstractions and intermediate spatial cues, thereby yielding more discriminative and fine-grained class-aware representations. 2) To enhance the reliability of historical recall under partial labeling, we incorporate uncertainty-aware pseudo-labeling that adaptively filters out highuncertainty predictions. 3) Moving beyond random feature interpolation, we introduce a context-aware synthesis mechanism that exploits latent semantic relations to synthesize informative unknown features, reserving embedding space for future concepts. 4) We address the long-standing issue of heterogeneous forgetting by designing a category-balanced gradient compensation loss that reweights the gradient propagation based on each class’s forgetting tendency. 5) We provide an exhaustive evaluation across diverse and more rigorous experimental protocols, including long-term incremental sequences and non-overlapping session designs. Notably, the enhanced KBK achieves a substantial performance leap, specifically a 2.7% gain in Avg. Acc over the conference version (HCP) on the challenging Split-VOC B10-C5 setting. Expanded ablation studies and interpretability analyses further confirm the model’s enhanced ability to maintain distinct representations for successive incremental tasks.

## 2. Related Work

## 2.1. Single-Label Incremental Learning

SLCIL focuses on incrementally integrating novel classes from a data stream while retaining competence on previously learned ones. In scenarios without access to former data, existing solutions can be categorized into three main types to address forgetting. Regularization-based methods constrain the update of important parameters or responses from previous tasks. EWC [4] estimates parameter importance with Fisher information, while later works refine importance estimation and optimization [5, 6]. Knowledge-distillation methods, such as LwF [7] and PODNet [8], preserve previous model behaviors through logit- or feature-level distillation. Rehearsal-based methods store a small number of exemplars from old classes and replay them during later training. Representative methods include ER [9], iCaRL [10], DER++ [11], and BiC [12], which improve old-class retention through exemplar replay, distillation, or bias correction. Architectural-based methods expand or adapt model structures for new tasks [13, 14]. DER [15] introduces task-specific fea ture extractors, DyTox [16] uses task tokens, and recent prompt-based methods [17, 18] learn task-adaptive prompts on pre-trained ViTs. Unlike these SLCIL methods, our work focuses on MLCIL, where multiple co-occurring labels and partial annotations require explicit specification of known and unknown knowledge.

## 2.2. Multi-Label Classification

Multi-Label Classification (MLC) remains challenging compared with singlelabel classification. A straightforward solution treats each category independently and trains multiple one-vs-rest binary classifiers. However, this independence assumption ignores inter-label correlations and spatial co-occurrence among objects, both of which are crucial in MLC. To leverage label structure, prior works adopt sequence models, typically RNNs, to model interlabel relationships, yet they are sensitive to label ordering and often dificult to optimize. Graph-based methods instead construct a label graph and apply Graph Convolutional Networks (GCNs) to capture relationships, but when co-occurrence statistics are sparse, they risk encoding spurious correlations. Beyond architecture, loss design also matters–the standard binary cross-entropy encounters severe imbalance between positive and negative labels in MLC. Asymmetric Loss (ASL) addresses this by dynamically downweighting easy negatives and applying hard thresholding. Recent studies fur-

ther improve MLC by revisiting label correlation modeling and class-aware representation learning. For example, transformer-based methods leverage long-range visual-label dependencies to boost multi-label recognition [19]. More recent works model label correlations with latent context [20], enhance label-specific visual representations with semantic guidance [21], and pursue causal label correlations to suppress spurious contextual dependencies [22]. These methods improve static MLC from the perspectives of context modeling, semantic representation learning, and robust label-correlation modeling.

However, most existing MLC methods assume a fixed label space and jointly observed training data. They are not directly designed for the classevolving and partially labeled setting of MLCIL, where historical and prospective classes may appear in current-session images without annotations. Directly extending static MLC methods to this setting can sufer from catastrophic forgetting, feature aliasing, and interference from unlabeled co-occurring objects. In contrast, KBK explicitly specifies known and unknown knowledge to coordinate historical retention, current-class discrimination, and futureclass preparedness.

## 2.3. Multi-Label Class-Incremental Learning

Multi-Label Class-Incremental Learning (MLCIL) extends continual recognition to settings where images contain multiple co-occurring objects and only a subset of labels is annotated at each session. Recent eforts tackle this problem from complementary angles. On the memory side, PRS [23] devises a replay sampling strategy to alleviate the bufer class imbalance, and OCDM [24] adopts a greedy update policy for fast and efective exemplar maintenance. To model inter-label structure, AGCN [25] employs GCNs to maintain consistent label relations across sessions. KRT [1] tackles label absence with a knowledge restoration and transfer framework. CSC [26] constructs cross-task label relationships and calibrates confidence to curb over-confident predictions. Although these methods improve MLCIL from the perspectives of replay, label-relation modeling, pseudo-labeling, confidence calibration, and rebalancing, they focus on preserving historical knowledge or improving current-session learning. The ambiguity between reliable known knowledge and uncertain unknown content under label co-occurrence is still not explicitly modeled. In contrast, KBK formulates MLCIL from a known/unknown knowledge-specification perspective and jointly coordi nates historical, current, and prospective knowledge, thereby alleviating the conflict among knowledge retention, current-class discrimination, and futureclass preparedness.

## 2.4. Incremental Learning in Related Vision Tasks

Incremental object detection (IOD) represents a specialized form of ML-CIL extended to detection, where models continually localize and recognize novel classes while retaining prior ones [27, 28, 29]. Shmelkov et al. [30] introduce IOD for the two-stage object detector and distill the output of Fast R-CNN. MVCD [31] preserves multi-dimensional correlations within feature maps to stabilize updates. Replay-based strategies are also explored, either through feature replay or exemplar storage [32].

Incremental semantic segmentation (ISS) is required to maintain pixelwise predictions for previously seen classes. SDR [33] leverages prototype matching and contrastive learning to enforce consistency in the latent space. PLOP [34] employs multi-scale feature distillation. ALIFE [35] introduces feature replay combined with an adaptive regularizer to balance accuracy and eficiency.

Despite methodological parallels, methods in IOD and ISS cannot be directly applied to MLCIL, as they rely on task-specific architectures and supervision (bounding boxes or pixel-level annotations). In contrast, our work targets settings with only image-level labels, underscoring the unique challenges of MLCIL and the necessity of tailored solutions.

## 3. Proposed Method

## 3.1. Problem Formulation

MLCIL aims to develop a unified model that can correctly identify all classes encountered throughout the learning stream. We consider T incremental sessions. Let $\mathcal { D } = \{ ( x , y ) \}$ denote the full dataset, where x is an image and $y \in \{ 0 , 1 \} ^ { | \mathcal { C } | }$ is a multi-hot label vector over the global class set C. we partition C into T disjoint blocks $\{ \mathcal { C } ^ { 1 } , \ldots , \mathcal { C } ^ { T } \}$ in lexicographic order and induce a corresponding partition of the data $\{ \mathcal { D } ^ { 1 } , \ldots , \mathcal { D } ^ { T } \}$ , with ${ \mathcal { C } } ^ { m } \cap { \mathcal { C } } ^ { n } = \emptyset$ for $m \neq n$ and $\textstyle \bigcup _ { t = 1 } ^ { T } { \bar { \mathcal { C } } } ^ { t } = { \mathcal { C } }$ . At session t, the learner has access only to $\mathcal { D } ^ { t }$ with the current label space $\mathcal { C } ^ { t }$ . Diferent from SLCIL, images in $\mathcal { D } ^ { t }$ may simultaneously contain objects from past classes $\mathcal { C } ^ { 1 : t - 1 }$ and future classes $\mathcal { C } ^ { t + 1 : T }$ . Accordingly, we refer to $\mathcal { C } ^ { 1 : t - 1 } , \mathcal { C } ^ { t }$ , and $\mathcal { C } ^ { t + 1 : T }$ as historical knowledge, current knowledge, and prospective knowledge, respectively. Since only labels in $\mathcal { C } ^ { t }$ are observed at session t, the session-wise known knowledge is defined as the union of historical and current knowledge, i.e., $K _ { \mathrm { k n o w n } } ^ { t } = { \mathcal C } ^ { 1 : t - 1 } \cup { \mathcal C } ^ { t }$ 2 whereas prospective knowledge remains outside the observed label space and thus appears as currently unknown content. However, annotations outside $\mathcal { C } ^ { t }$ are unavailable, that is, labels in ${ \mathcal { C } } \setminus { \mathcal { C } } ^ { t }$ are treated as absent. After completing session t, the model is evaluated on all classes $\textstyle { \mathcal { C } } ^ { 1 : t } = \bigcup _ { i = 1 } ^ { t } { \mathcal { C } } ^ { i }$ , producing prediction $\hat { y } ^ { t } ( x ) \in [ 0 , 1 ] ^ { | \mathcal { C } ^ { 1 : t } | }$ for each image. The objective is to continually expand the label space with minimal degradation of previous classes.

![](images/bcbf24e0480bf19e67ca1b92e6955edbb336590267b23fe9dd41e4f7b1244ce4.jpg)  
Figure 2: Framework of KBK. KBK coordinates historical, current, and prospective knowledge under a unified known/unknown knowledge-specification principle. HFP and URE clarify reliable historical and current knowledge by extracting class-aware features and recalling high-confidence, low-uncertainty historical supervision. SPU probes unknown knowledge by synthesizing semantic-guided unknown features, while CGC balances historical retention and current learning through class-wise gradient compensation.

## 3.2. Overview

Figure 2 depicts the proposed KBK framework, which is organized around a unified known/unknown knowledge-specification principle. The goal is to identify reliable known knowledge from historical and current classes, while preventing uncertain unknown content from being absorbed into known-class representations. In this way, KBK jointly coordinates historical retention, current-class discrimination, and prospective preparedness under the ML-CIL setting. Concretely, to make the known more discriminative, KBK introduces a hierarchical feature purification module, in which for each class, a dedicated embedding is trained to extract fine-grained class features from a fused representation that combines reinforced high-level semantic abstractions with complementary low-level visual signals. This hierarchical fusion reduces feature aliasing across sessions and supports flexible extension to new classes by continually adding class embeddings. In addition, we employ an uncertainty-aware recall enhancement strategy that combines confidence with entropy-guided uncertainty to filter unreliable pseudo labels, thus improving the reliability of historical recall and mitigating uneven forgetting across classes. To probe the unknown knowledge, KBK leverages semantic relations to weight absent-class embeddings and synthesize informative unknown features, which push non-target representations away, leading to more compact class boundaries for known classes while reserving space for future learning. Finally, to balance heterogeneous forgetting of old classes, we introduce a category-balanced gradient compensation loss that dynamically reweights gradient propagation based on forgetting dynamics.

## 3.3. Clarifying Known Knowledge

Here we introduce the hierarchical feature purification module and outline how it adapts to multi-label incremental scenarios. Subsequently, we study heterogeneous class forgetting and develop an uncertainty-aware recall enhancement that incorporates historical distributional priors.

1) Hierarchical Feature Purification. To eliminate cross-session feature aliasing caused by noise and impurities from non-target features, we design a hierarchical feature purification module, which fuses shallow spatial and deep semantic information via residual integration, then uses class embeddings as queries to disentangle fine-grained class-specific signals from multi-class entanglement. Compared with other methods [1, 26], our approach ensures distinctive representations for each class by incorporating complementary information from multiple feature levels, enabling parallel prediction of both historical and current classes.

For a given sample from dataset $\mathcal { D } ^ { t }$ , we extract features from the intermediate and last layers of the backbone network, denoted as F<sub>mid</sub> ∈ R<sup>H1×W1×d1</sup> and $F _ { d e e p } \in \mathbb { R } ^ { H _ { 2 } \times W _ { 2 } \times d _ { 2 } }$ , respectively, where $( H _ { 1 } , W _ { 1 } )$ and $( H _ { 2 } , W _ { 2 } )$ represent the spatial dimension at diferent layers, and $d _ { 1 } , d _ { 2 }$ represent the corresponding feature dimension. The shallow features $F _ { m i d }$ capture low-level spatial patterns and color information at higher spatial resolution, while $F _ { d e e p }$ focuses on global semantic content at lower spatial resolution but with richer semantic representation. To efectively balance shallow spatial information with deep semantic knowledge while handling scale discrepancies, we use $F _ { m i d }$ as auxiliary information to enhance the overall feature representation. The fused hierarchical features $F _ { f u s e } \in \mathbb { R } ^ { H _ { 2 } \times W _ { 2 } \times d }$ are computed as follows:

$$
F _ { f u s e } = F _ { d e e p } + \mathcal { G } _ { 1 } ( A ( F _ { m i d } ) ) + \mathcal { G } _ { 2 } ( F _ { d e e p } ) ,\tag{1}
$$

where $\boldsymbol { \mathcal { A } } ( \cdot )$ is an adaptive pooling or interpolation operation that spatially aligns $F _ { m i d }$ to match the dimensions of $F _ { d e e p }$ (transforming from $H _ { 1 } \times W _ { 1 }$ to $H _ { 2 } \times W _ { 2 } )$ . The function $\mathcal { G } _ { 1 }$ encodes information from the intermediate layer feature $F _ { m i d }$ , and $\mathcal { G } _ { 2 }$ further refines the information from $F _ { d e e p }$ . Both $\mathcal { G } _ { 1 }$ and $\mathcal { G } _ { 2 }$ are Multi-Layer Perception (MLPs).

Subsequently, the hierarchical fused features $F _ { f u s e }$ are reshaped into patch tokens $F \in \mathbb { R } ^ { ( H _ { 2 } \cdot W _ { 2 } ) \times d }$ To aggregate object-level information and extract fine-grained class-specific features, we associate each class with a learnable embedding, forming a sequence $S \in \mathbb { R } ^ { M \times d }$ with $M = | \mathcal { C } ^ { 1 : t } |$ known classes at session t. The feature purification module consists of $L$ multi-head selfattention (MSA) blocks that jointly processes S and the hierarchical feature tokens $F$ , producing purified class features $O _ { S } \in \mathbb { R } ^ { M \times d }$ and enhanced patch features $\bar { O _ { F } } \in \mathbb { R } ^ { ( H _ { 2 } \cdot \bar { W _ { 2 } } ) \times d }$ (the mini-batch is omitted):

$$
( Q , K , V ) = ( W _ { q } , W _ { k } , W _ { v } ) [ F , S ] ,\tag{2}
$$

$$
O = W _ { o } \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d / h } } \right) V + b _ { o } ,\tag{3}
$$

where ${ \cal O } = [ { \cal O } _ { F } , { \cal O } _ { S } ]$ and h is the number of attention heads. Under the attention formulation, each class embedding acts as a query that selectively activates the spatial regions in $F _ { f u s e }$ relevant to that class and captures contextual relations with other classes, benefiting from both low-level spatial details and high-level semantic cues. This architecture naturally supports incremental tasks by appending new class embeddings, enabling eficient parallel prediction of both old and new classes with enhanced multi-scale discriminative power.

Class features $O _ { S }$ are then fed to the classifier to produce output logits. Following established practices [15], we employ a stability classifier to predict logits for previous classes and a plasticity classifier to generate logits for new classes, which are then concatenated to form complete logits $P ^ { 1 : t }$ for final classification. During learning new classes, we freeze the old embeddings $S ^ { 1 : t - 1 } \in \mathbb { R } ^ { | \mathcal { C } ^ { 1 : t - 1 } | \times d }$ and the stability classifier to preserve existing knowledge, while allowing new class embeddings $S ^ { t } \in \mathbb { R } ^ { | \mathcal { C } ^ { t } | \times \bar { d } }$ and the plasticity classifier to adapt to incoming data. Upon transitioning to session $t + 1$ , the plasticity and stability classifiers consolidated to form the updated stability classifier, and a fresh plasticity classifier is instantiated for the subsequent learning.

![](images/083224b78993537532dc1614efb3fb7f4c70cd30daa879c0119ffa2093923d3e.jpg)  
Figure 3: Heterogeneous forgetting among classes render a unified fixed threshold inadequate for recalling previously learned knowledge.

2) Uncertainty-aware Recall Enhancement.

Specifically, leveraging the historical model’s discriminative ability for past classes, we obtain the predicted class probability for old classes as $\mathsf { \bar { P } } _ { i } ^ { t - 1 } = [ p _ { i 1 } ^ { t - 1 } , \cdot \cdot \cdot , p _ { i | 1 : C _ { t - 1 } | } ^ { t - 1 } ]$ . At the same time, we characterize the prediction uncertainty for class $j$ using the binary entropy:

$$
h _ { i j } ^ { t - 1 } = - p _ { i j } ^ { t - 1 } \log { p _ { i j } ^ { t - 1 } } - ( 1 - p _ { i j } ^ { t - 1 } ) \log { ( 1 - p _ { i j } ^ { t - 1 } ) } .\tag{4}
$$

Diferent classes, however, exhibit varying levels of learning dificulties and susceptibility to forgetting. Some categories have clear, easily distinguishable features and are thus relatively resistant to forgetting, whereas others are more subtle and consequently more prone to being forgotten. Treating all categories uniformly ignores this heterogeneity, which introduces noise to pseudo-labeling, resulting in either false positives or missed true positives. To quantify such heterogeneity, we perform a statistical analysis of confidence forgetting across classes. Taking VOC dataset under B10-C2 setting as an example, we approximate the confidence distribution of each class with a Gaussian distribution $p _ { j } ^ { t } \sim \mathcal N ( \mu _ { j } ^ { t } , \sigma _ { j } ^ { 2 t } )$ , where:

$$
\mu _ { j } ^ { t } = \frac { \sum _ { i = 1 } ^ { \lfloor D ^ { t } \rfloor } p _ { i j } ^ { t } \cdot \mathbb { I } _ { ( y _ { i j } ^ { t } = 1 ) } } { \sum _ { i = 1 } ^ { \lfloor D ^ { t } \rfloor } \mathbb { I } _ { ( y _ { i j } ^ { t } = 1 ) } } ,\tag{5}
$$

$$
\sigma _ { j } ^ { 2 t } = \frac { \sum _ { i = 1 } ^ { \lfloor D ^ { t } \rfloor } \left( p _ { i j } ^ { t } \cdot \mathbb { I } _ { ( y _ { i j } ^ { t } = 1 ) } - \mu _ { j } ^ { t } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { \lfloor D ^ { t } \rfloor } \mathbb { I } _ { ( y _ { i j } ^ { t } = 1 ) } } ,\tag{6}
$$

where I(·) denotes the indicator function, which returns 1 if the condition is true, and 0 otherwise. Following the definition of accuracy forgetting in SLCIL, we extend this notion to define class-level confidence forgetting as the decline between the historical maximum mean confidence and the postsession mean confidence:

$$
C F _ { j } ^ { t } = \operatorname* { m a x } _ { m \leq t - 1 } ( \mu _ { j } ^ { m } - \mu _ { j } ^ { t } ) .\tag{7}
$$

Empirical evidence (see Fig. 3) indicates that the baseline experiences dramatic confidence decay (up to 0.85), and that the degree of decline difers substantially across categories. This heterogeneity renders a single global threshold inefective. To address this, we employ per-class adaptive thresholds and introduce entropy gating as a second-level filter to suppress highuncertainty recalls. Specifically, the confidence threshold $\varepsilon _ { j }$ of class j is derived from the confidence statistics of previous models (e.g., mean or a robust percentile over positives), while the entropy threshold $\tau _ { j }$ is estimated from the entropy distribution of reliable samples. With both thresholds taken as the mean (by default), we derive the class-specific pseudo labels to supervise training in session t:

$$
\begin{array} { r } { \hat { y } _ { i j } ^ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } y _ { i j } ^ { t } = 1 \mathrm { ~ a n d ~ } j \in \mathscr { C } ^ { t } , } \\ { 1 , } & { \mathrm { i f ~ } y _ { i j } ^ { t } = 0 \mathrm { ~ a n d ~ } j \in \mathscr { C } ^ { 1 : t - 1 } \mathrm { ~ a n d } } \\ { ~ } & { \quad p _ { i j } ^ { t - 1 } \geq \varepsilon _ { j } \mathrm { ~ a n d ~ } h _ { i j } ^ { t - 1 } \leq \tau _ { j } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{8}
$$

Both the confidence and entropy distribution queues are updated after each session to account for distribution drift. This uncertainty-aware classadaptive strategy provides a clean recall signal under partial labeling, consistently improving historical precision-recall while mitigating confirmation bias.

## 3.4. Probing Unknown Knowledge

Incorporating auxiliary classes during incremental sessions helps preserve embedding capacity for upcoming categories and improve forward compatibility. Introducing real classes from external datasets is a simple option but is often infeasible in practice due to access limitations and domain shift that could disrupt the current training process. In multi-label settings, where unlabeled co-occurring classes naturally exist, we instead exploit implicit knowledge to synthesize prospective features that embody relational semantics, thereby enriching the feature representation space. By encouraging knownclass embeddings to diverge from these synthesized unknown features, the model forms tighter, more discriminative clusters for known categories while implicitly reserving representational capacity for subsequent categories.

Formally, for an image $x _ { i }$ from $\mathcal { D } ^ { t }$ with pseudo target $\hat { Y } _ { i } ^ { t } = [ \hat { y } _ { i 1 } ^ { t } , \cdots , \hat { y } _ { i M } ^ { t } ] \in$ $\mathbb { R } ^ { 1 \times M }$ where $M = | \mathcal { C } ^ { 1 : t } |$ , we obtain purified class features $O _ { S } \in \mathbb { R } ^ { | \mathcal { C } ^ { 1 : t } | \times d }$ via the hierarchical feature purification module. We partition these into present-class features $O _ { S } ^ { + }$ and absent-class features $O _ { S } ^ { - }$ :

$$
\begin{array} { r } { O _ { S } ^ { + } = \left\{ o _ { j } \vert \hat { y } _ { i j } ^ { t } = 1 , j \in \mathcal { C } ^ { 1 : t } \right\} \in \mathbb { R } ^ { M ^ { + } \times d } , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { O _ { S } ^ { - } = \left\{ o _ { j } | \hat { y } _ { i j } ^ { t } = 0 , j \in \mathcal { C } ^ { 1 : t } \right\} \in \mathbb { R } ^ { M ^ { - } \times d } , } \end{array}\tag{10}
$$

with $M ^ { + } + M ^ { - } = M$ . Here, the attention of present-class embeddings is supervised to focus on fine-grained target regions, whereas the absent-class embeddings distribute more freely across unexplained foreground areas, implicitly encoding signals of latent unknown categories. To exploit this implicit information, we leverage semantic relations to synthesize unknown features. Instead of random mixing, we weight absent-class features according to their semantic similarities with present classes, ensuring that synthesized features better capture the relational structure between known and potential future categories. Specifically, for each present-class embedding $o _ { p } \in O _ { S } ^ { + }$ , we compute its cosine similarity with every absent-class embedding $o _ { a } \in O _ { S } ^ { - }$

$$
s ( o _ { p } , o _ { a } ) = \frac { o _ { p } \cdot o _ { a } } { \left\| o _ { p } \right\| \left\| o _ { a } \right\| } .\tag{11}
$$

We aggregate these similarities across all present classes to produce a reliability weight for each absent class:

$$
w _ { a } = \frac { 1 } { | O _ { S } ^ { + } | } \sum _ { o _ { p } \in O _ { S } ^ { + } } s ( o _ { p } , o _ { a } ) .\tag{12}
$$

Here, the cosine similarity is used as a contextual reliability prior rather than a predictor of the exact semantics of future classes. A larger similarity indicates that the absent-class feature is more related to the current scene, while a weakly correlated absent feature receives a smaller weight and is less likely to dominate the synthesized unknown feature. These weights modulate a random interpolation coeficient $\lambda _ { a }$ sampled from a Beta distribution, yielding the synthesized unknown feature $O _ { V }$ :

$$
\lambda _ { a } \sim \mathrm { B e t a } ( \alpha , \beta ) , \bar { \lambda } _ { a } = \frac { \lambda _ { a } \cdot w _ { a } } { \sum _ { j } \lambda _ { j } \cdot w _ { j } } ,\tag{13}
$$

$$
O _ { V } = \sum _ { o _ { a } \in O _ { S } ^ { - } } \bar { \lambda } _ { a } \cdot o _ { a } .\tag{14}
$$

The synthesized feature $O _ { v }$ is incorporated into training by extending the plasticity classifier’s output to $C ^ { t } + 1$ , where the extra dimension represents an explicit unknown class. Treating synthesized features as a dedicated unknown class enables the model to account for label uncertainty within the current session, sharpen discrimination among known classes, and improve preparedness for future classes.

It should be noted that SPU does not aim to construct exact prototypes for all future classes. Instead, the synthesized unknown feature serves as a coarse known/unknown boundary regularizer. When an absent-class feature has low correlation with the present-class context, its contribution is naturally suppressed by the similarity weighting, reducing the risk of feature-space distortion. If most absent features are weakly correlated with the current scene, the synthesized unknown feature may become less informative, and the forward-compatibility benefit of SPU may be weakened. Nevertheless, the unknown branch still helps prevent unexplained co-occurring content from being directly absorbed into known-class decision regions.

## 3.5. Category-balanced Gradient Compensation

To address the negative-positive sample imbalance inherent to multiple scenarios, we adopt the Asymmetric Loss (ASL) as optimization objective–a focal-loss variant that applies distinct $\gamma$ for positive and negative examples. For an image $x _ { i }$ , we obtain its class probabilities $P _ { i } ^ { t } = [ p _ { i 1 } ^ { t } , \cdot \cdot \cdot , p _ { i K } ^ { t } ] \in \mathbb { R } ^ { 1 \times K }$ where $K = | \mathcal { C } ^ { 1 : t } | + 1$ including an additional unknown class. We train the framework with ASL:

$$
L _ { c l s } = \frac { 1 } { B } \frac { 1 } { K } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { K } \left\{ \begin{array} { l l } { ( 1 - p _ { i j } ^ { t } ) ^ { \gamma + } \log ( p _ { i j } ^ { t } ) , } & { \mathrm { i f ~ } \hat { y } _ { i j } ^ { t } = 1 , } \\ { ( p _ { i j } ^ { t } ) ^ { \gamma - } \log ( 1 - p _ { i j } ^ { t } ) , } & { \mathrm { i f ~ } \hat { y } _ { i j } ^ { t } = 0 , } \end{array} \right.\tag{15}
$$

where $\gamma +$ and $\gamma -$ are used to modulate the contribution of positives and negatives and set $\gamma ^ { + } = 0 , \gamma ^ { - } = 4$ by default.

However, during training on incoming classes, the current dataset becomes biased toward new categories, which in turn skews the back-propagated gradients at the final layer and exacerbates catastrophic forgetting. Moreover, standard gradient optimization neglects heterogeneous forgetting spaces across old classes: easy-to-forget classes with diverse appearances versus hard-to-forget classes with distinctive attributes. To overcome heterogeneous forgetting from the gradient perspective, we design a category-balanced gradient compensation loss that adaptively rescales gradient contributions according to each class’s forgetting pace. Specifically, the gradient update value $\mathcal { G } _ { i j } ^ { t }$ with respect to the j-th neuron $\mathcal { N } _ { j } ^ { t }$ in the last classifier layer is expressed as:

$$
\mathcal { G } _ { i j } ^ { t } = \frac { \partial L _ { c l s } ( p _ { i j } ^ { t } , \hat { y } _ { i j } ^ { t } ) } { \partial \mathcal { N } _ { j } ^ { t } } = p _ { i j } ^ { t } - \hat { y } _ { i j } ^ { t } .\tag{16}
$$

To equalize disparate forgetting rates across easily- and hard-to-forget old classes while moderating the learning speed for newly introduced categories, we normalize gradients separately for groups of classes learned in diferent incremental tasks and use these normalization factors to reweight the standard classification loss $L _ { c l s }$ . Given a mini-batch $\{ x _ { i } ^ { t } , \hat { y } _ { i } ^ { t } \} _ { i = 1 } ^ { B }$ sampled from task $t ,$ we compute task-wise average gradient magnitudes that have been learned in the m-th $( 1 \leq m \leq t )$ incremental task as:

$$
\mathcal { G } ^ { m ^ { + } } = \frac { 1 } { \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { C } ^ { m } } \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 1 ) } } \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { C } ^ { m } } \left| \mathcal { G } _ { i j } ^ { t } \right| \cdot \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 1 ) } .\tag{17}
$$

Multi-label data are inherently imbalanced, where positives are scarce and negatives are abundant. Estimating the reference solely from positive samples yields high-variance, unstable normalization when positives are rare, which can spuriously inflate the weights. Incorporating negatives reduces variance, suppresses noise amplification, and is equally crucial for correcting hard negatives. Accordingly, we compute separate batch-only mean gradient magnitudes for positives and negatives per old class and choose the corresponding mean as the normalization baseline conditioned on the sample’s label. This category-aware design symmetrically calibrates false negatives and false positives, improving stability and old-class retention. Thus, we

rewrite Eq.(17) as follows:

$$
\mathcal { G } ^ { m } = \mathcal { G } ^ { m ^ { + } } \cdot \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 1 \wedge j \in \mathcal { C } ^ { m } ) } + \mathcal { G } ^ { m ^ { - } } \cdot \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 0 \wedge j \in \mathcal { C } ^ { m } ) } ,\tag{18}
$$

$$
\mathcal { G } ^ { m ^ { - } } = \frac { 1 } { \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { C } ^ { m } } \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 0 ) } } \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { C } ^ { m } } \left| \mathcal { G } _ { i j } ^ { t } \right| \cdot \mathbb { I } _ { ( \hat { y } _ { i j } ^ { t } = 0 ) } .\tag{19}
$$

Based on the task-level average gradients of old classes, we derive a category-balanced weight $\psi _ { i j } ^ { t }$ by normalizing the per-sample gradient of class $j$ in image i:

$$
\psi _ { i j } ^ { t } = \left\{ \begin{array} { l l } { \frac { \big | \mathcal { G } _ { i j } ^ { t } \big | } { \mathcal { G } ^ { m } } , } & { \mathrm { i f ~ } j \in \mathcal { C } ^ { m } \mathrm { ~ a n d ~ } m < t , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{20}
$$

The dynamically updated weight $\psi _ { i j } ^ { t }$ is applied to reweight the classification objective, yielding the category-balanced gradient compensation loss:

$$
L _ { g c } = \frac { 1 } { B } \frac { 1 } { K } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { K } \psi _ { i j } ^ { t } \cdot \left\{ \begin{array} { l l } { { ( 1 - p _ { i j } ^ { t } ) ^ { \gamma + } \log ( p _ { i j } ^ { t } ) , } } & { { \mathrm { i f ~ } \hat { y } _ { i j } ^ { t } = 1 , } } \\ { { ( p _ { i j } ^ { t } ) ^ { \gamma - } \log ( 1 - p _ { i j } ^ { t } ) , } } & { { \mathrm { i f ~ } \hat { y } _ { i j } ^ { t } = 0 . } } \end{array} \right.\tag{21}
$$

Intuitively, a forgotten class typically produces large gradients $\mathcal { G } _ { i j } ^ { t }$ , which inflates $\psi _ { i j } ^ { t }$ in Eq.(20). The larger weight increases the contribution of that class to $L _ { g c }$ , driving its output probability $p _ { i j } ^ { t }$ toward the pseudo-label and thereby helping to reduce catastrophic forgetting.

## 3.6. Mechanism Discussion

From the knowledge-specification perspective, partial labeling creates three coupled risks: historical classes lack explicit supervision, current classes are learned from entangled representations, and prospective classes may be prematurely absorbed into known-class regions. KBK addresses these risks through complementary mechanisms. HFP specifies historical and current knowledge at the representation level by extracting class-aware features from co-occurring objects. URE restores reliable historical supervision by filtering old-class predictions according to class-adaptive confidence and uncertainty. SPU assigns unexplained prospective information to an explicit unknown direction, preventing it from interfering with known-class representations. Finally, CGC coordinates historical retention and current learning at the optimization level by compensating heterogeneous forgetting while moderating the learning dynamics of newly introduced classes. These components form a unified progression from representation specification and supervision restoration to boundary construction and optimization balancing.

## 4. Experiments

## 4.1. Experimental Setup and Evaluation Metrics

Datasets. We evaluate our approach on MS-COCO 2014 [36] and PAS-CAL VOC 2007 [37]. MS-COCO contains 82,081 training examples and 40,137 test examples across 80 common object classes (≈ 2.9 labels/image). PASCAL VOC includes 5,011 images in the train-val split and 4,952 test images covering 20 classes (≈ 1.6 labels/image).

Protocols. We evaluate KBK under two protocols. Protocol A, adopted from [1], follows the standard VOC/COCO setting, where images are collected according to current-session classes and may overlap across sessions. Protocol B (following [2]) follows the stricter Split-VOC/Split-COCO setting, where disjoint image splits are used to avoid cross-session image overlap. Both protocols retain the key MLCIL dificulty of partial supervision, where only current classes are annotated while historical and prospective classes may appear without labels. Their diferent session constructions allow us to evaluate the robustness of KBK to protocol changes. Notably, since Protocol B constructs disjoint image splits before assigning class blocks, some sessions may contain few or no positive instances for certain assigned classes. This does not afect comparison fairness because all methods follow the same fixed protocol, but it makes Protocol B a stricter benchmark with irregular session composition.

Following previous work [11], we use the unified Bi-Cj convention, where i denotes the number of classes learned in the base session and j is the number of classes to be learned in each subsequent session. Under protocol A, experiments are conducted with MS-COCO (B40-C10, B0-C10) and Pascal VOC (B10-C2, B0-C4, B4-C2, B5-C3). For protocol B, we evaluate the model on the Split-COCO dataset with B40-C10 and B0-C20 settings and on the Split-VOC dataset with B10-C5 and B0-C5 settings.

Evaluation Metrics. Following KRT [1], we use mean average precision (mAP) as the primary metric, computed over all classes learned up to each session. We summarize mAP with two indicators: (1) average mAP, the arithmetic mean of mAP values across sessions, which reflects overall continual performance; and (2) last mAP, the mAP observed at the final session. To give a complete picture of multi-label performance, we also report the per-class F1 score (CF1) and the overall F1 score (OF1).

Table 1: Performance on MS-COCO. Comparison methods are grouped by their source task. A bufer size of 0 denotes no rehearsal. Best scores per setting are shown in bold and second-best are indicated by underline.
<table><tr><td rowspan=4 colspan=11>MS-COCO B0-C10Source BufferMethod                            Avg. Acc   Last AccTask    SizemAP CF1OF1mAPUpper-bound|Baseline|                      |76.4 79.4 81.8</td><td rowspan=1 colspan=2>MS-COCO B40-C10</td></tr><tr><td rowspan=1 colspan=1>Avg. Acc</td><td rowspan=1 colspan=1>Last Acc</td></tr><tr><td rowspan=1 colspan=1>mAP</td><td rowspan=1 colspan=1>CF1OF1mAP</td></tr><tr><td rowspan=1 colspan=2>|76.4 79.4 81.8</td></tr><tr><td rowspan=7 colspan=2>FTPODNet [8]oEWC [5]LWF [7]AGCN [25]KRT [1]</td><td rowspan=1 colspan=1>Baseline|</td><td></td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>38.3</td><td rowspan=1 colspan=1>6.1 13.4 16.9</td><td rowspan=1 colspan=1>35.1</td><td rowspan=1 colspan=1>6.0 13.6 17.0</td></tr><tr><td rowspan=5 colspan=1>SLCILSLCILSLCILMLCIL</td><td></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td></td><td rowspan=2 colspan=1></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>43.7</td><td rowspan=1 colspan=1>7.214.1 25.6</td><td rowspan=1 colspan=1>44.3</td><td rowspan=1 colspan=1>6.8 13.9 24.7</td></tr><tr><td></td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>46.9</td><td rowspan=1 colspan=1>6.713.4 24.3</td><td rowspan=1 colspan=1>44.8</td><td rowspan=1 colspan=1>11.1 16.5 27.3</td></tr><tr><td></td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>47.9</td><td rowspan=1 colspan=1>9.015.1 28.9</td><td rowspan=1 colspan=1>48.6</td><td rowspan=1 colspan=1>9.5 15.8 29.9</td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>53.9 56.6 61.4</td><td rowspan=1 colspan=1>73.9</td><td rowspan=1 colspan=1>58.7 59.9 69.1</td></tr><tr><td rowspan=1 colspan=1>MLCIL</td><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>74.6</td><td rowspan=1 colspan=1>55.6 56.5 65.9</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>64.4 63.474.0</td></tr><tr><td rowspan=2 colspan=2>CSC [26]HCP [3]KBK</td><td rowspan=2 colspan=1>MLCILMLCILMLCIL</td><td rowspan=2 colspan=6></td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>64.9 66.8 72.8</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>65.7 67.0 74.0</td></tr><tr><td rowspan=1 colspan=1>77.980.7</td><td rowspan=1 colspan=1>60.4 65.3 71.267.368.275.8</td><td rowspan=1 colspan=1>78.979.7</td><td rowspan=1 colspan=1>64.9 68.6 75.369.670.077.2</td></tr><tr><td rowspan=5 colspan=2>TPCIL [38]PODNet [8]DER++ [11]AGCN [25]KRT [1]</td><td rowspan=1 colspan=1>SLCILSLCIL</td><td></td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>63.865.7</td><td rowspan=1 colspan=1>20.1 21.6 50.813.6 17.3 53.4</td><td rowspan=1 colspan=1>63.165.4</td><td rowspan=1 colspan=1>|25.3 25.1 53.124.2 23.4 57.8</td></tr><tr><td rowspan=1 colspan=1>SLCIL</td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>68.1</td><td rowspan=1 colspan=1>33.3 36.7 54.6</td><td rowspan=1 colspan=1>69.6</td><td rowspan=1 colspan=1>41.9 43.7 59.0</td></tr><tr><td rowspan=1 colspan=1>MLCIL</td><td></td><td rowspan=1 colspan=3>5/cl</td><td></td><td rowspan=1 colspan=1>SS</td><td rowspan=3 colspan=1>72.975.8</td><td rowspan=1 colspan=1>56.7 58.5 63.6</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=1>59.8 61.3 69.7</td></tr><tr><td></td><td rowspan=2 colspan=4></td><td rowspan=2 colspan=2>MLCIL</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>60.0 61.0 68.3</td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>66.0 65.9 74.3</td></tr><tr><td rowspan=3 colspan=2>CSC[26]HCP [3]KBK</td><td rowspan=1 colspan=1></td><td rowspan=3 colspan=6>MLCILMLCILMLCIL</td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>79.2</td><td rowspan=1 colspan=1>67.3 68.1 73.7</td><td rowspan=1 colspan=1>78.4</td></tr><tr><td></td><td rowspan=2 colspan=1>79.481.0</td><td rowspan=2 colspan=1>70.3 72.9 74.572.875.776.3</td><td rowspan=2 colspan=1>79.479.9</td><td rowspan=1 colspan=1>71.5 74.1 76.7</td></tr><tr><td></td><td rowspan=1 colspan=1>73.976.477.7</td></tr><tr><td rowspan=2 colspan=2>iCaRL [10]BiC [12]</td><td rowspan=2 colspan=1>SLCILSLCIL</td><td></td><td rowspan=2 colspan=5></td><td rowspan=1 colspan=1>59.7</td><td rowspan=1 colspan=1>19.3 22.8 43.8</td><td rowspan=2 colspan=1>65.665.5</td><td rowspan=2 colspan=1>|22.1 25.5 55.738.1 40.7 55.9</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>31.0 38.1 51.1</td></tr><tr><td rowspan=8 colspan=2>ER [9]TPCIL [38]PODNet [8]DER++ [11]AGCN [25]KRT [1]</td><td rowspan=1 colspan=1>SLCIL</td><td></td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>40.6 43.6 47.2</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>58.6 61.1 61.6</td></tr><tr><td rowspan=3 colspan=1>SLCIL</td><td></td><td rowspan=3 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>69.4</td><td rowspan=1 colspan=1>51.7 52.8 60.6</td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>60.4 62.6 66.5</td></tr><tr><td rowspan=2 colspan=1>SLCILSLCIL</td><td rowspan=2 colspan=6>20/class</td><td rowspan=2 colspan=1>70.072.7</td><td rowspan=1 colspan=1>45.2 48.7 58.8</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>46.6 42.1 64.2</td></tr><tr><td rowspan=1 colspan=1>45.2 48.7 63.1</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>51.5 53.5 66.3</td></tr><tr><td rowspan=1 colspan=1>MLCIL</td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>73.2</td><td rowspan=1 colspan=1>59.5 60.3 66.0</td><td rowspan=1 colspan=1>75.2</td><td rowspan=1 colspan=1>64.1 65.2 71.7</td></tr><tr><td rowspan=1 colspan=1>MLCIL</td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>76.5</td><td rowspan=1 colspan=1>63.9 64.7 70.2</td><td rowspan=1 colspan=1>78.3</td><td rowspan=1 colspan=1>67.9 68.9 75.2</td></tr><tr><td rowspan=3 colspan=2>CSC [26]HCP [3]KBK</td><td rowspan=3 colspan=1>MLCILMLCILMLCIL</td><td rowspan=3 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>79.6</td><td rowspan=1 colspan=1>67.8 68.6 74.8</td></tr><tr><td rowspan=2 colspan=1>79.681.4</td><td rowspan=1 colspan=1>70.4 73.0 74.6</td><td rowspan=1 colspan=1>79.6</td><td rowspan=1 colspan=1>71.9 74.5 77.2</td></tr><tr><td rowspan=1 colspan=1>73.576.377.2</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>74.476.778.3</td></tr><tr><td rowspan=2 colspan=2>PRS [23]OCDM [24]</td><td rowspan=2 colspan=1>MLOILMLOIL</td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>48.8</td><td rowspan=1 colspan=1>8.514.7 27.9</td><td rowspan=1 colspan=1>50.8</td><td rowspan=1 colspan=1>9.3 15.1 33.2</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>49.5</td><td rowspan=1 colspan=1>8.614.9 28.5</td><td rowspan=1 colspan=1>51.3</td><td rowspan=1 colspan=1>9.5 15.5 34.0</td></tr><tr><td rowspan=2 colspan=2>AGCN [25]KRT [1]</td><td rowspan=2 colspan=1>MLCILMLCIL</td><td rowspan=2 colspan=5>1000</td><td rowspan=3 colspan=2>1000</td><td rowspan=2 colspan=1>73.075.7</td><td rowspan=2 colspan=1>59.4 65.9 59.061.6 63.6 69.3</td><td rowspan=1 colspan=1>75.0</td></tr><tr><td rowspan=1 colspan=1>78.3</td><td rowspan=1 colspan=1>67.5 68.5 75.1</td></tr><tr><td rowspan=1 colspan=2>CSC[26]</td><td rowspan=1 colspan=1>MLCIL</td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>79.3</td><td rowspan=1 colspan=1>67.5 68.5 73.9</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>67.8 69.7 76.0</td></tr><tr><td rowspan=2 colspan=2>HCP [3]KBK</td><td rowspan=2 colspan=1>MLCILMLCIL</td><td rowspan=2 colspan=6></td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>70.2 72.8 74.4</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>71.8 74.4 76.7</td></tr><tr><td rowspan=1 colspan=2>KBK</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>81.6</td><td rowspan=1 colspan=1>73.776.577.3</td><td rowspan=1 colspan=1>80.3</td><td rowspan=1 colspan=1>74.376.878.2</td></tr></table>

Table 2: Comparison of the results on PASCAL VOC dataset. Data in Bold represents the best results.
<table><tr><td rowspan=2 colspan=2>Method</td><td rowspan=2 colspan=1>BufferSize</td><td rowspan=1 colspan=1>|VOC B0-C4|</td><td rowspan=1 colspan=3>VOC B10-C2|</td><td rowspan=1 colspan=2>VOC B5-C3|</td><td rowspan=1 colspan=1>VOC B4-C2</td></tr><tr><td rowspan=1 colspan=1>Avg. Last</td><td rowspan=1 colspan=3>Avg.  Last</td><td rowspan=1 colspan=2>Avg. Last</td><td rowspan=1 colspan=1>Avg. Last</td></tr><tr><td rowspan=1 colspan=2>Upper bound|</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=3>1     93.6</td><td rowspan=1 colspan=2>93.6</td><td rowspan=1 colspan=1>93.6</td></tr><tr><td rowspan=5 colspan=2>FTKRT [1]CSC [26]HCP [3]KBK</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82.1  62.9</td><td rowspan=1 colspan=3>72.9  43.0</td><td rowspan=1 colspan=2>74.4  49.4</td><td rowspan=1 colspan=1>60.4 37.0</td></tr><tr><td rowspan=4 colspan=1>0</td><td rowspan=4 colspan=1>89.1  80.290.4 85.192.9 87.994.1 90.9</td><td rowspan=1 colspan=3>84.3  71.5</td><td rowspan=1 colspan=2>84.1  72.1</td><td rowspan=1 colspan=1>67.5 43.5</td></tr><tr><td rowspan=1 colspan=3>89.0  83.8</td><td rowspan=1 colspan=2>88.0  82.1</td><td rowspan=3 colspan=1>83.3 74.182.9  70.185.5 72.2</td></tr><tr><td rowspan=2 colspan=3>90.1  81.990.9  85.0</td><td rowspan=1 colspan=1>89.6  82.</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=2>90.8 84.8</td></tr><tr><td rowspan=8 colspan=2>iCarL [10]BiC [12]ER [9]TPČIL [38]PODNet [8]DER++ [11]KRT [1]CSC [26]</td><td rowspan=8 colspan=1>2/class</td><td rowspan=1 colspan=1>87.2  72.4</td><td rowspan=1 colspan=3>79.0  66.7</td><td rowspan=4 colspan=2>一65.3</td><td rowspan=6 colspan=1>一       一73.7 57.975.7 62.577.0 61.6</td></tr><tr><td rowspan=1 colspan=1>86.8  72.2</td><td rowspan=1 colspan=3>81.7  69.7</td></tr><tr><td rowspan=2 colspan=1>86.1 71.587.6 77.3</td><td rowspan=1 colspan=3>81.5  68.6</td><td rowspan=3 colspan=2>81.4  70.3</td></tr><tr><td rowspan=1 colspan=2>80.7</td><td rowspan=1 colspan=2>70.8</td></tr><tr><td rowspan=1 colspan=1>88.1 76.6</td><td rowspan=2 colspan=3>81.282.3  70.6</td><td rowspan=1 colspan=1>71.4</td></tr><tr><td rowspan=1 colspan=1>DER++</td><td rowspan=1 colspan=1>87.9 76.1</td><td rowspan=1 colspan=2>78.0  68.1</td></tr><tr><td rowspan=1 colspan=1>90.7 83.4</td><td rowspan=1 colspan=3>87.7  80.5</td><td rowspan=1 colspan=2>89.4  82.5</td><td rowspan=1 colspan=1>82.0 72.6</td></tr><tr><td rowspan=1 colspan=1>92.4 87.9</td><td rowspan=1 colspan=3>91.6  87.8</td><td rowspan=1 colspan=2>91.9  87.5</td><td rowspan=1 colspan=1>90.4 86.6</td></tr><tr><td rowspan=2 colspan=2>HCP [3]KBK</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>93.5 89.2</td><td rowspan=1 colspan=3>92.1  86.3</td><td rowspan=1 colspan=2>91.3  83.9</td><td rowspan=1 colspan=1>85.5 77.7</td></tr><tr><td rowspan=1 colspan=1>94.7 91.3</td><td rowspan=1 colspan=3>92.8  88.8</td><td rowspan=1 colspan=2>92.5 87.8</td><td rowspan=1 colspan=1>90.1 81.8</td></tr></table>

## 4.2. Implementation Details

For fair comparison, we follow the general training settings of prior ML-CIL works, especially KRT. Each experiment is run three times under the same protocol, and the average performance is reported. The model is trained for 20 epochs with a batch size of 64 using Adam, weight decay $1 \times 1 0 ^ { - 4 }$ , and the OneCycleLR schedule. The learning rate is $4 \times 1 0 ^ { - 5 }$ for the base session, and $1 \times 1 0 ^ { - 4 } \div 4 \times 1 0 ^ { - 5 }$ for incremental sessions on MS-COCO / PASCAL VOC, respectively. All experiments are conducted on NVIDIA RTX 3090 GPUs. In HFP, we set the number of attention heads to 4, use 3 MSA blocks for VOC and 1 for MS-COCO, and set the embedding dimension to $d = 2 0 4 8$ At session t, the model is initialized from the (t − 1)-th checkpoint; old class embeddings are fixed, while new class embeddings are randomly initialized. We merge the previous stability and plasticity classifiers into a frozen stability classifier for $\vert \mathcal { C } ^ { 1 : t - 1 } \vert$ historical classes, and append a plasticity classifier with $\vert \mathcal { C } ^ { t } \vert + 1$ outputs for current classes and the unknown class. For SPU, the interpolation coeficient is sampled from Beta(0.2, 0.2). For URE, the class-adaptive confidence and entropy thresholds are estimated from historical prediction distributions and updated after each session.

Table 3: Comparison with prompt-based methods on MS-COCO. KBK, KRT, KRT-R, and HCP use the same TResNetM backbone, while L2P and DualPrompt are ViT-based prompt methods and are included as cross-architecture references.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Param.</td><td colspan="2">COCO B0-C10</td><td colspan="2">COCO B40-C10</td></tr><tr><td> ${ \overline { { \mathrm { A v g . } } } }$ </td><td>Last</td><td>Avg.</td><td>Last</td></tr><tr><td>Upper bound L2P[17] L2P-R[17]</td><td rowspan="3">86.0M</td><td rowspan="3">73.28</td><td>83.16 67.72</td><td>73.07</td><td>83.16 70.42</td></tr><tr><td>Dual-prompt [18]</td><td>73.87 74.67</td><td>68.22 73.64</td><td>71.68</td></tr><tr><td rowspan="3"></td><td rowspan="3">69.39 74.87 70.20</td><td rowspan="3">74.45 74.54</td><td rowspan="3">71.95 72.60</td></tr><tr><td>Dual-prompt-R [18]</td></tr><tr><td>Upper bound KRT [1]]</td><td></td></tr><tr><td>KRT-R [1] HCP [3]</td><td rowspan="2">29.4M</td><td rowspan="2">74.64</td><td rowspan="2">81.80</td><td rowspan="2"></td><td rowspan="2">81.80</td></tr><tr><td rowspan="4"></td></tr><tr><td></td><td>76.53</td><td>65.94</td><td>77.83</td><td>74.02</td></tr><tr><td></td><td>79.61</td><td>71.24 74.60</td><td>78.34</td><td>75.18</td></tr><tr><td></td><td>81.36</td><td>77.19</td><td>79.64 80.35</td><td>77.18 78.33</td></tr></table>

## 4.3. Main Results

Results on MS-COCO. Tab. 1 exposes severe forgetting in simple finetuning (FT) and SLCIL methods: FT’s Last Acc collapses to 16.9% and PODNet to 25.6%, whereas KBK achieves 75.8% on B0-C10 with zero bufer. Multi-label online incremental learning methods likewise achieve weak results here, yet our method substantially outperforms PRS and OCDM. Compared with the latest MLCIL methods, our method consistently maintains a leading position, achieving up to 2.3% higher Avg. Acc than CSC (bufer size=1000) and clear metric-wise improvements relative to AGCN. Stable gains under the B40-C10 setting further highlight the robustness of KBK. Notably, even in a no-replay setting, it surpasses competing methods that rely on memory bufers in both average and final-session accuracy. For a fair interpretation, we note that KBK, KRT, KRT-R, and HCP are compared under the same TResNetM backbone, which forms the strictly backbone-matched comparison. In contrast, L2P and DualPrompt are based on pre-trained ViT backbones with larger parameter scales, and are therefore included only as cross-architecture references rather than strictly same-backbone baselines. Tab. 3 illustrates the comparison between KBK and prompt-based methods, which shows that KBK achieves higher performance across all scenarios with the same parameters as KRT.

Results on PASCAL VOC. Tab. 2 reports the performance comparison on VOC under four diferent protocols. In the bufer-free setting, KBK sets a new state of the art across all settings. Specifically, under B0-C4, it reaches an Avg. Acc of 94.1%, outperforming both previous best method CSC (90.4%) and HCP (92.9%). In the more challenging long-sequence B10-C2, KBK attains an Avg. Acc of 90.9%, demonstrating strong stability in longterm incremental learning. Particularly under VOC B4-C2, it significantly exceeds other methods with a 2.6% improvement over HCP, highlighting its scalability under high-frequency class increments. When a small exemplar bufer (2 samples/class) is available, KBK continues to maintain a leading position, achieving 92.8% and 92.5% in Avg. Acc under B10-C2 and B5-C3, respectively, with Last Acc also clearly exceeding competing methods (88.8% and 87.8%), confirming its ability to efectively leverage limited memory for enhanced knowledge retention and predictive consistency in multi-label classincremental learning.

Table 4: Performance on Split-VOC and Split-COCO datasets. The input size is 448 × 448, and all results are obtained from APPLE [2].
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=1>BufferSize</td><td rowspan=1 colspan=2>[Split-VOC B10-C5|</td><td rowspan=1 colspan=1>Split-VOC B0-C5</td><td rowspan=1 colspan=1>Buffer</td><td rowspan=1 colspan=2>Split-COCO B40-C10|Split-COCO B0-C20</td></tr><tr><td rowspan=1 colspan=2>Avg.    Last</td><td rowspan=1 colspan=1>Avg.    Last</td><td rowspan=1 colspan=1>Size</td><td rowspan=1 colspan=1>Avg.      Last</td><td rowspan=1 colspan=1>Avg.     Last</td></tr><tr><td rowspan=1 colspan=1>Upper bound|</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>-      94.2</td><td rowspan=1 colspan=1>-      94.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-       86.4</td><td rowspan=1 colspan=1>-      86.4</td></tr><tr><td rowspan=1 colspan=1>FTHCP [3]KBK</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>74.8    60.192.5    89.295.6    92.5</td><td rowspan=1 colspan=1>74.1    59.792.2    86.995.7   92.0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>35.8      11.183.2      80.484.5     82.5</td><td rowspan=1 colspan=1>|51.9     23.685.0     81.987.0     84.3</td></tr><tr><td rowspan=4 colspan=1>iCarL [10]ER [9]TPCIL [38]PODNet[8]PASS [39]DER++[11]APPLE[2]HCP [3]KBK</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>90.8    87.186.0    73.9</td><td rowspan=1 colspan=1>88.3    84.882.7    68.3</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>76.7      65.672.3      64.069.4      71.2</td><td rowspan=3 colspan=1>76.5     64.563.6     50.173.5     68.974.8     61.072.2     49.973.8     67.3</td></tr><tr><td rowspan=2 colspan=1>5/class</td><td rowspan=2 colspan=2>90.290.3    86.386.0    76.990.2    83.9</td><td rowspan=1 colspan=1>84.2</td><td rowspan=2 colspan=1>87.9    79.487.8    84.175.6    51.888.9    84.8</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>20/class</td><td rowspan=1 colspan=1>77.1      66.173.8      59.466.7      55.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>91.7    89.493.4    90.696.1    93.9</td><td rowspan=1 colspan=1>89.5    85.693.5   89.796.2   93.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82.1      74.684.0      82.085.4     83.7</td><td rowspan=1 colspan=1>83.5     76.685.2     82.387.2     84.8</td></tr></table>

Results on Split-COCO and Split-VOC. Tab. 4 summarizes the experimental results on the challenging Split-COCO and Split-VOC benchmarks. We observe that our method, KBK, delivers substantial gains over rehearsal-based baselines under these split protocols and advances the state of the art in MLCIL. Specifically, KBK improves Avg. Acc over APPLE by 4.4% on B10-C5 and 6.7% on B0-C5. Remarkably, even with no replay bufer (bufer = 0), KBK matches or exceeds many replay-based methods and sub stantially outperforms simple fine-tuning. These results indicate that KBK is robust to the stricter, non-overlapping session design used in the split benchmarks and can preserve and transfer prior knowledge without relying

Table 5: Ablation study w.r.t. Hierarchical Feature Purification (HFP), Uncertainty-aware Recall Enhancement (URE), Semantic-guided Probing Unknown (SPU), and Categorybalanced Gradient Compensation (CGC) on Pascal VOC B10-C2 and B4-C2 settings.
<table><tr><td rowspan="3">Method</td><td rowspan="3">FP</td><td rowspan="3">RE</td><td rowspan="3">PU</td><td rowspan="3"></td><td colspan="2">VOC B10-C2</td><td colspan="2">VOC B4-C2</td></tr><tr><td>Avg.</td><td>Last</td><td>Avg.</td><td>Last</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td></td><td>72.87</td><td>46.76</td><td>60.20</td><td>22.01</td></tr><tr><td></td><td>V</td><td></td><td></td><td></td><td>82.09</td><td>66.29</td><td>75.88</td><td>52.95</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td></td><td>86.00</td><td>73.45</td><td>80.73</td><td>64.44</td></tr><tr><td>HCP</td><td>√</td><td></td><td>√</td><td></td><td>86.47 90.05</td><td>71.64</td><td>80.30</td><td>62.30</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>81.85</td><td>82.94</td><td>70.08</td></tr><tr><td></td><td>HFP</td><td>URE</td><td>SPU</td><td>CGC</td><td>Avg.</td><td>Last</td><td>Avg.</td><td>Last</td></tr><tr><td></td><td>√</td><td></td><td></td><td></td><td>85.21</td><td>70.12</td><td>78.67</td><td>60.47</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td></td><td>87.01</td><td>75.36</td><td>81.30</td><td>72.63</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>√</td><td>89.28</td><td>80.37</td><td>84.51</td><td>70.71</td></tr><tr><td></td><td>√</td><td>V</td><td>V</td><td></td><td>90.63</td><td>83.37</td><td>85.03</td><td>68.78</td></tr><tr><td>KBK</td><td>√</td><td>√</td><td>√</td><td>V</td><td>90.89</td><td>84.95</td><td>85.53</td><td>72.24</td></tr></table>

Table 6: Ablation of recall strategy. Rows one and two correspond to a single global threshold, row three shows top-K filtering, RE refers to Recall Enhancement, and URE to uncertainty-aware recall enhancement.
<table><tr><td rowspan="2">Recall Strategy</td><td colspan="2">VOC B10-C2</td><td colspan="2">VOC B4-C2</td></tr><tr><td>Avg.</td><td>Last</td><td>Avg.</td><td>Last</td></tr><tr><td>ε=0.8</td><td>83.09</td><td>68.94</td><td>74.45</td><td>57.84</td></tr><tr><td>ε=0.9</td><td>85.08</td><td>70.82</td><td>75.63</td><td>55.63</td></tr><tr><td>Top-2</td><td>84.24</td><td>68.94</td><td>75.14</td><td>56.48</td></tr><tr><td>RE</td><td>90.05</td><td>81.85</td><td>83.02</td><td>65.51</td></tr><tr><td>URE</td><td>90.89</td><td>84.95</td><td>85.53</td><td>72.24</td></tr></table>

on exemplars.

The consistent improvements across VOC, COCO, Split-VOC, and Split-COCO also indicate that KBK is not tailored to a single dataset or session construction. These benchmarks difer in data scale, label density, class composition, and co-occurrence patterns, providing evidence for the robustness of KBK.

## 4.4. Ablation Study

Efectiveness of Component. As shown in Tab. 5, fine-tuning sufers from severe forgetting, whereas FP improves Avg. Acc by 9.22 and 15.68 points on B10-C2 and B4-C2, respectively. HFP further improves Avg. Acc by 3.12 and 2.79 points by strengthening the representation-level specification of historical and current knowledge. Based on HFP, URE improves Last Acc by 5.24 and 12.16 points through more reliable historical supervision. Compared with KBK without SPU, introducing SPU improves Avg./Last Acc by 1.61/4.58 points on B10-C2 and 1.02/1.53 points on B4-C2, confirming the benefit of separating prospective information from known-class regions. Finally, CGC further improves Last Acc by 1.58 and 3.46 points by balancing historical retention and current-class optimization. Together, HFP, URE, SPU, and CGC progressively specify knowledge at the representation, supervision, feature-boundary, and optimization levels, demonstrating that KBK forms a unified framework rather than a collection of independent components.

Table 7: Efect of freezing old class embeddings on VOC.
<table><tr><td rowspan="2">Frozen</td><td colspan="2">VOC B10-C2</td><td colspan="2">VOC B0-C4</td></tr><tr><td>Avg. Acc</td><td>Last Acc</td><td>Avg. Acc</td><td>Last Acc</td></tr><tr><td>No</td><td>89.8</td><td>83.4</td><td>93.2</td><td>88.8</td></tr><tr><td>Yes</td><td>90.9</td><td>85.0</td><td>94.1</td><td>90.9</td></tr></table>

Table 8: Sensitivity analysis of the Beta distribution parameters in SPU.
<table><tr><td rowspan="2">Beta(α, β)</td><td>VOC B10-C2</td><td>VOC B0-C4</td></tr><tr><td>Avg. Acc Last Acc</td><td>Avg. Acc Last Acc</td></tr><tr><td>Beta(0.1,0.1)</td><td>90.6 84.7</td><td>93.9 90.5</td></tr><tr><td>Beta(0.2,0.2)</td><td>90.9 85.0</td><td>94.1 90.9</td></tr><tr><td>Beta(0.5,0.5)</td><td>90.7 85.1</td><td>93.8 90.6</td></tr><tr><td>Beta(1.0, 1.0)</td><td>90.3 84.3</td><td>93.6 89.9</td></tr></table>

Efect of Freezing Old Class Embeddings. We further investigate the efect of freezing old class embeddings during incremental learning. As shown in Tab. 7, freezing old embeddings consistently improves the performance on both VOC B10-C2 and VOC B0-C4. Specifically, on VOC B10-C2, freezing improves Avg. Acc from 89.8 to 90.9 and Last Acc from 83.4 to 85.0. On VOC B0-C4, it improves Avg. Acc from 93.2 to 94.1 and Last Acc from 88.8 to 90.9. The improvement is more evident on Last Acc, indicating that freezing old embeddings helps preserve stable semantic anchors for historical classes and alleviates long-term forgetting. Meanwhile, the feature pathway, new class embeddings, and classifier remain trainable, so the model can still adapt to new classes without relying on the drift of old embeddings. These results support the stability-plasticity benefit of freezing old class embeddings in KBK.

Table 9: Computational eficiency comparison on VOC B10-C2.
<table><tr><td></td><td>Method | Params (M)</td><td>FLOPs (G)</td><td>Training Time</td><td>Memory (GB/GPU)</td></tr><tr><td>KRT</td><td>29.40</td><td>5.73</td><td>28min</td><td>7.46</td></tr><tr><td>HCP</td><td>29.44</td><td>8.34</td><td>19min</td><td>9.80</td></tr><tr><td>KBK</td><td>29.48</td><td>9.25</td><td>25min</td><td>9.86</td></tr></table>

![](images/ea55abe5371be6a73591cab605544ae9874fdda23ef1d78496e72c6894351db9.jpg)  
(a) Calinski-Harabasz indexes

![](images/edb30bbd9312d0c5f7373bbe5a6a1da950bce2a582437b3929f22a7d52ffbc73.jpg)  
(b) Influence of buffer size  
Figure 4: C-H Indexes of class features and the influence of bufer size.

Sensitivity to Beta Distribution Parameters. We further analyze the sensitivity of SPU to the Beta distribution parameters used for unknownfeature synthesis. As shown in Tab. 8, KBK maintains stable performance under diferent Beta settings. The default setting Beta(0.2, 0.2) achieves the best or comparable results. Compared with smoother interpolation distributions such as Beta(0.5, 0.5), Beta(0.2, 0.2) encourages more diverse interpolation coeficients and provides stronger perturbation for unknown-feature synthesis. The overall performance variation remains small, indicating that SPU does not rely on a narrowly tuned Beta configuration.

Computational Eficiency. We further analyze the computational cost of KBK. As shown in Tab. 9, KRT, HCP, and KBK share the same basic TResNetM backbone. HCP and KBK additionally introduce class embeddings, which add negligible additional parameter cost compared with the backbone scale. We also report FLOPs, training time, and peak GPU memory under the same hardware setting. The results show that KBK introduces only moderate computational overhead while maintaining lightweight inference. This is because the feature purification module is shared across classes and sessions, and SPU/CGC are mainly training-time operations without additional inference branches.

![](images/69ad059cf2940e52d8475fa8d17c252f4781fb1c469c038e967bb51eabd46eca.jpg)  
Figure 5: Visualization of attention maps for each ground-truth label in the Hierarchical Feature Purification (HFP) structure.

Analysis of Bufer Size. Fig. 4(b) illustrates the results of KBK on MS-COCO B40-C10 under varying bufer sizes. Increasing the bufer from 0 to 50 yields a modest improvement in average mAP, indicating that KBK performs efectively even under strict storage constraints while still benefiting from the availability of replay bufers.

Analysis of Hierarchical Feature Purification. To illustrate how KBK encodes class-specific information, we visualize the attention map between each ground-truth class embedding and the fused image tokens in the HFP module. Fig. 5 presents the input image in column one and per-class attention overlays in the remaining columns. We observe that each class embedding can capture corresponding specific local features and locate the specified object approximately, even when similar classes co-occur in the same image. All validate that the HFP structure can extract fine-grained classaware knowledge for learning multiple classes while avoiding class confusion.

![](images/19a42a590494ee74a57e7c05cb06723e15bda5fa3524eb22ccac4793c7b07d7a.jpg)  
(a) Real future features

![](images/6067f1d159d054643fd4065b82151bcf9a04656c8e0a1a547bfd70aa70b9ae60.jpg)  
(b) Generated future features  
Figure 6: Visualization of real future features and generated unknown features.

Ablation of Recall Enhancement. We evaluate several pseudo-labeling schemes for recalling past knowledge: a single global threshold ε, top-K selection, confidence-based recall enhancement (RE), and our uncertaintyaware recall enhancement (URE). Tab. 6 reveals that model performance depends strongly on the choice of ε. By contrast, URE, which discards predictions with low confidence or high uncertainty and simultaneously accounts for class-wise heterogeneous forgetting, yields more reliable supervisory signals and improved results.

Analysis of Generated Unknown Features. Fig. 6 compares real future features with synthetically generated unknown features (shown in gray), which both promote compact feature representations. It can be observed that the distribution of our synthesized features closely aligns with that of real future features, which confirms the efectiveness of probing unknown mechanism in providing valuable foresight for incremental learning. It should be noted that SPU does not aim to predict the exact semantics of future classes in advance. Instead, it prevents unlabeled prospective-class information from being prematurely absorbed into known-class decision regions, thereby maintaining a more flexible feature space for later class expansion. For quantitative evaluation, Fig. 4 (a) reports the Calinski-Harabasz Index of feature representations, where higher values indicate better inter-class separation and intra-class compactness. Our method achieves significantly higher index values at each session compared to the baseline. Moreover, removing the probing-unknown mechanism leads to a clear decline in later sessions, showing that unknown-feature synthesis is important for preserving a well-structured representation space. These results support the forwardcompatibility role of SPU: by assigning uncertain prospective information to an explicit unknown direction, KBK reduces its interference with known classes and prepares for future expansion.

![](images/a85e22f70a78e3fae97f3af0725152fe1228f98f38edfa3924768570cc9783c2.jpg)  
(a) Baseline: session 1

![](images/9fe495b9ce10225b06fed7614202eead46b30837c0aef458b844b2456b3b970e.jpg)  
(b) Baseline: session 3

![](images/ee58d50dee3d3baf33c668edc7276915ff4d5669c901c2813f9db24bc74c2c15.jpg)  
(c) Baseline: session 6

![](images/8bab7dc33d879fccc7dd95ce1ede5d7f6ffc8a7279bcc6a05c78bcade205dad6.jpg)  
(d) Ours: session 1

![](images/73773c5b33a37650df3cefc375c4bc0a7c805b993c278c3267c370fc499427a6.jpg)  
(e) Ours: session 3

![](images/00111d460215d83aad392a203edd7702c215db0120f0d4207a738991aed5e866.jpg)  
(f) Ours: session 6  
Figure 7: t-SNE visualization of feature distributions under PASCAL VOC B10-C2 setting. There are 10, 14 and 20 classes in session 1, 3 and 6, respectively.

Analysis of Feature Aliasing. Fig. 7 visualizes feature trajectories produced during incremental learning on VOC. In session 1, our method successfully learns pure knowledge representation without feature aliasing, enhancing the model’s forward compatibility by reserving suficient embedding space to accommodate future classes. In contrast, as training proceeds, the baseline embeddings progressively collapse and further succumb to catastrophic forgetting in session 6, manifested by blurred boundaries between previously learned classes and inability to maintain distinct representations for new and old classes. Our method consistently maintains class separation across all sessions, with each class’s features forming compact and nonoverlapping clusters. These results verify that our approach efectively preserves knowledge from prior sessions, adapts to current incremental tasks, and anticipates future class expansion.

## 5. Limitations

Although KBK achieves consistent improvements on standard MLCIL benchmarks, several limitations remain. First, our experiments are mainly conducted on PASCAL VOC and MS-COCO, which enable fair comparison with prior MLCIL methods but cannot fully represent more complex real-world continual streams, such as long-tailed distributions, open-world category evolution, or domain shifts. Second, SPU adopts a single aggregated unknown class to regularize the known/unknown boundary, which is lightweight but cannot model fine-grained taxonomy among unseen categories. When absent-class features are weakly correlated with the current scene or dominated by background responses, the synthesized unknown feature may become less informative. Third, URE reduces noisy historical recall through confidence and uncertainty constraints, but over-confident incorrect predictions may still pass the recall gates under severe ambiguity or distribution shift. Extending KBK to more complex tasks such as incremental object detection and semantic segmentation is an interesting direction.

## 6. Conclusion

This paper proposes KBK, an efective framework for multi-label classincremental learning that explicitly specifies which knowledge is known and what is unknown at each session, thereby accommodating historical, current and prospective information. For the known component, we introduce hierarchical feature purification to disentangle fine-grained class-aware features from fused semantic and visual information, which reduces feature aliasing both within and across sessions. To strengthen historical supervision under partial labeling, we study the heterogeneous forgetting among categories and further develop uncertainty-aware recall enhancement that suppresses lowconfidence, high-uncertainty predictions. To probe the unknown, we exploit semantic relations to generate informative unknown features as a prospective class to enhance inter-class discrimination and reserve representational room for future additions. Furthermore, a category-balanced gradient compensation loss adaptively rescales gradient contributions to balance difering forgetting velocities among classes. Experiments and ablation studies confirm the efectiveness and robustness of KBK framework.

In future work, we will explore extending the knowledge-specification principle to more complex visual tasks such as incremental object detection and semantic segmentation. In these tasks, known/unknown ambiguity may arise at the region or pixel level, where unlabeled objects or regions can interfere with both old and new categories. Adapting uncertainty-aware recall, prospective unknown modeling, and category-balanced optimization to localization-aware architectures remains an interesting future direction.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant NO 62406318, 62376266, 62076195, 62376070, 62406167 and U24B6012).

## References

[1] S. Dong, H. Luo, Y. He, X. Wei, J. Cheng, Y. Gong, Knowledge restore and transfer for multi-label class-incremental learning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18711–18720.

[2] X. Song, K. Shu, S. Dong, J. Cheng, X. Wei, Y. Gong, Overcoming catastrophic forgetting for multi-label class-incremental learning, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 2389–2398.

[3] A. Zhang, D. Yang, C. Liu, X. Hong, Y. Zhou, Specifying what you know or not for multi-label class-incremental learning, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 22345–22353.

[4] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al., Overcoming catastrophic forgetting in neural networks, National Academy of Sciences 114 (13) (2017) 3521–3526.

[5] J. Schwarz, W. Czarnecki, J. Luketina, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu, R. Hadsell, Progress & Compress: A scalable framework for continual learning, in: Proceedings of the International Conference on Machine Learning, 2018, pp. 4528–4537.

[6] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, T. Tuytelaars, Memory aware synapses: Learning what (not) to forget, in: Proceedings of the European Conference on Computer Vision, 2018, pp. 139–154.

[7] Z. Li, D. Hoiem, Learning without forgetting, IEEE Transactions on Pattern Analysis and Machine Intelligence 40 (12) (2017) 2935–2947.

[8] A. Douillard, M. Cord, C. Ollion, T. Robert, E. Valle, PODNet: Pooled outputs distillation for small-tasks incremental learning, in: Proceedings of the European Conference on Computer Vision, 2020, pp. 86–102.

[9] M. Riemer, I. Cases, R. Ajemian, M. Liu, I. Rish, Y. Tu, G. Tesauro, Learning to learn without forgetting by maximizing transfer and minimizing interference, arXiv preprint arXiv:1810.11910 (2018).

[10] S.-A. Rebufi, A. Kolesnikov, G. Sperl, C. H. Lampert, iCARL: Incremental classifier and representation learning, in: Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[11] P. Buzzega, M. Boschini, A. Porrello, D. Abati, S. Calderara, Dark experience for general continual learning: a strong, simple baseline, in: Proceedings of the Advances in Neural Information Processing Systems, Vol. 33, 2020, pp. 15920–15930.

[12] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, Y. Fu, Large scale incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 374–382.

[13] W. Sun, Q. Li, J. Zhang, D. Wang, W. Wang, Y.-a. Geng, Exemplar-free class incremental learning via discriminative and comparable parallel one-class classifiers, Pattern Recognition 140 (2023) 109561.

[14] W. Liu, X.-J. Wu, F. Zhu, M.-M. Yu, C. Wang, C.-L. Liu, Class incremental learning with self-supervised pre-training and prototype learning, Pattern Recognition 157 (2025) 110943.

[15] S. Yan, J. Xie, X. He, DER: Dynamically expandable representation for class incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3014–3023.

[16] A. Douillard, A. Ramé, G. Couairon, M. Cord, Dytox: Transformers for continual learning with dynamic token expansion, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 9285–9295.

[17] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, T. Pfister, Learning to prompt for continual learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 139–149.

[18] Z. Wang, Z. Zhang, S. Ebrahimi, R. Sun, H. Zhang, C.-Y. Lee, X. Ren, G. Su, V. Perot, J. Dy, et al., DualPrompt: Complementary prompting for rehearsal-free continual learning, in: Proceedings of the European Conference on Computer Vision, 2022, pp. 631–648.

[19] M. Li, D. Wang, X. Liu, Z. Zeng, R. Lu, B. Chen, M. Zhou, PatchCT: Aligning patch set and label set with conditional transport for multilabel image classification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15348–15358.

[20] Z. Chen, Q. Cui, R. Deng, J. Hu, G. Zhang, Modeling label correlations with latent context for multi-label recognition, in: European Conference on Computer Vision, 2024, pp. 218–234.

[21] X. Zhu, J. Li, J. Cao, D. Tang, J. Liu, B. Liu, Semantic-guided representation enhancement for multi-label image classification, IEEE Transactions on Circuits and Systems for Video Technology 34 (10) (2024) 10036–10049.

[22] Z.-M. Chen, X. Jin, Y. Ge, S. Chan, In pursuit of causal label correlations for multi-label image recognition, Advances in Neural Information Processing Systems 37 (2024) 51634–51654.

[23] C. D. Kim, J. Jeong, G. Kim, Imbalanced continual learning with partitioning reservoir sampling, in: Proceedings of the European Conference on Computer Vision, 2020, pp. 411–428.

[24] Y.-S. Liang, W.-J. Li, Optimizing class distribution in memory for multilabel online continual learning, arXiv preprint arXiv:2209.11469 (2022).

[25] K. Du, F. Lyu, L. Li, F. Hu, W. Feng, F. Xu, X. Xi, H. Cheng, Multilabel continual learning using augmented graph convolutional network, IEEE Transactions on Multimedia (2023).

[26] K. Du, Y. Zhou, F. Lyu, Y. Li, C. Lu, G. Liu, Confidence self-calibration for multi-label class-incremental learning, in: Proceedings of the European Conference on Computer Vision, 2024, pp. 234–252.

[27] D. Yang, Y. Zhou, W. Shi, D. Wu, W. Wang, RD-IOD: Two-level residual-distillation-based triple-network for incremental object detection, ACM Transactions on Multimedia Computing, Communications, and Applications 18 (1) (2022) 1–23.

[28] N. Dong, Y. Zhang, M. Ding, Y. Bai, Class-incremental object detection, Pattern Recognition 139 (2023) 109488.

[29] A. Zhang, D. Yang, C. Liu, X. Hong, M. Shang, Y. Zhou, DCA: Dividing and conquering amnesia in incremental object detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 9851–9859.

[30] K. Shmelkov, C. Schmid, K. Alahari, Incremental learning of object detectors without catastrophic forgetting, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2017, pp. 3400–3409.

[31] D. Yang, Y. Zhou, A. Zhang, X. Sun, D. Wu, W. Wang, Q. Ye, Multiview correlation distillation for incremental object detection, in: Pattern Recognition, Vol. 131, 2022, p. 108863.

[32] D. Yang, Y. Zhou, X. Hong, A. Zhang, W. Wang, One-shot replay: Boosting incremental object detection via retrospecting one object, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, 2023, pp. 3127–3135.

[33] U. Michieli, P. Zanuttigh, Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations,

in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1114–1124.

[34] A. Douillard, Y. Chen, A. Dapogny, M. Cord, PLOP: Learning without forgetting for continual semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4040–4050.

[35] Y. Oh, D. Baek, B. Ham, Alife: Adaptive logit regularizer and feature replay for incremental semantic segmentation, in: Proceedings of the Advances in Neural Information Processing Systems, Vol. 35, 2022, pp. 14516–14528.

[36] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft COCO: Common objects in context, in: Proceedings of the European Conference on Computer Vision, 2014, pp. 740–755.

[37] M. Everingham, The pascal visual object classes challenge,(voc2007) results, http://pascallin. ecs. soton. ac. uk/challenges/VOC/voc2007/index. html. (2007).

[38] X. Tao, X. Chang, X. Hong, X. Wei, Y. Gong, Topology-preserving class-incremental learning, in: Proceedings of the European Conference on Computer Vision, 2020, pp. 254–270.

[39] F. Zhu, X.-Y. Zhang, C. Wang, F. Yin, C.-L. Liu, Prototype augmentation and self-supervision for incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 5871–5880.