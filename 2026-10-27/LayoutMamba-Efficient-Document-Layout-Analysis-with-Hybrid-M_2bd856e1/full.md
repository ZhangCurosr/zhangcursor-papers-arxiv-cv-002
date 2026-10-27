# LayoutMamba: Efficient Document Layout Analysis with Hybrid Mamba-Transformer Networks

Chen Kehua Chen Mingyi

## Abstract

Document layout analysis (DLA) suffers from a long-standing dilemma: existing methods either capture rich local details at the cost of prohibitive computational overhead (e.g., Transformer-based approaches with quadratic complexity) or achieve efficient global modeling but sacrifice fine-grained structural awareness (e.g., state-space models). This fundamental trade-off has severely hindered the deployment of DLA systems in real-world, resource-constrained scenarios. To fundamentally resolve this conflict, we present LayoutMamba, a pioneering hybrid framework that synergistically combines the strengths of sparse-attention Transformers for localized feature extraction (O(n) complexity) and the recurrent state-space mechanism of Mamba for dilution-free, long-range dependency modeling across the 2D document canvas. Our key insight is a two-stage paradigm: first, a lightweight Transformer encoder hierarchically distills salient local semantics; second, Mamba’s state-space model propagates global contextual dependencies without attention dilution, enabling precise, efficient, and scalable layout understanding. Furthermore, we identify and address a critical yet overlooked issue in existing DLA benchmarks: the inherent low-overlap distribution bias of layout objects, which we mitigate through our novel Geometric Overlap Penalty Loss (GOPLoss), a dual-optimization objective that simultaneously minimizes redundant bounding box overlaps and encourages geometrically consistent predictions. Extensive experiments on two challenging benchmarks, PubLayNet and D<sup>4</sup>LA, demonstrate that LayoutMamba not only surpasses the state-of-the-art AP on PubLayNet with 8× fewer parameters, but also maintains leading performance on D<sup>4</sup>LA, ranking top-3 among all published methods, achieving an unprecedented balance between accuracy, efficiency, and scalability. Our work pioneers a new cognition-inspired paradigm for document analysis, making substantial strides towards practical deployment in edge computing and resource-limited environments.

## 1 Introduction

The automated understanding of visually rich documents represents a pivotal challenge in artificial intelligence, underpinning digital transformation initiatives across diverse sectors. Documents, unlike natural images, possess a complex structure characterized by high-frequency textual details, graphical elements like tables and figures, and critical semantic information encoded within their spatial layout. Document Layout Analysis (DLA), the task of segmenting a document image into distinct semantic regions (e.g., paragraphs, tables, figures), serves as a fundamental prerequisite for downstream tasks such as information extraction and document retrieval. However, the vast heterogeneity in document types (from structured forms to research papers and invoices), intricate layouts, and variable scan quality pose significant hurdles.

Early endeavors in Document Layout Analysis (DLA) predominantly relied on conventional computer vision techniques, leveraging Convolutional Neural Network (CNN) architectures (LeCun et al., 1998; Ren et al., 2015; Liu et al., 2016; He et al., 2016; Oliveira & Viana, 2017; Chen et al., 2017; Huang et al., 2017; Hu et al., 2018; Redmon & Farhadi, 2018) to extract visual features, where layout elements were treated as objects to be detected or segmented. While this series of CNN-based methods, through continuous refinements, has achieved an excellent trade-off between accuracy and inference speed, they inherently suffer from a critical limitation: their propensity towards local feature extraction makes it arduous to directly capture long-range global contextual information, which is crucial for understanding complex document layouts. To mitigate this shortcoming, researchers in the document analysis community have subsequently introduced Transformerbased architectures, notably the Vision Transformer (ViT) paradigm (Gupta et al., 2021; Huang et al., 2022; Da et al., 2023; Chai et al., 2023; Shehzadi et al., 2023; Gu et al., 2023; Wang et al., 2024b; Shehzadi et al., 2024), capitalizing on the powerful global modeling capability of self-attention mechanisms. These Transformer-based models have showcased remarkable potential in various computer vision applications. For instance, DiT (Li et al., 2022a) has demonstrated the effectiveness of large-scale unsupervised pre-training in the context of document visual self-supervised learning, while VGT (Da et al., 2023) proposed a Vision Grid Transformer to enhance layout representation learning through a more structured, grid-like modeling of visual dependencies. Despite their superior modeling capabilities, these Transformer-based approaches are fundamentally constrained by the intrinsic quadratic computational complexity of self-attention (scaling with the square of the number of image patches or text tokens), which imposes a severe scalability bottleneck when processing high-resolution documents or text-dense layouts commonly encountered in real-world scenarios. Although techniques such as key-value caching (KV caching) can optimize computations for previously processed tokens to reduce inference latency, the quadratic complexity persists during both the training phase and when handling new inputs, remaining a core limitation. More recent efforts (Fang et al., 2022; Ullah et al., 2023; Li et al., 2022b) have thus begun to rethink the architectural paradigm, pivoting towards designing hybrid structures that synergistically combine the strengths of CNNs and Transformers while alleviating the complexity overhead. Very recently, State Space Models (SSMs) (Gu & Dao, 2023; Liu et al., 2024; Zhu et al., 2024b), and particularly the newly proposed Mamba architecture, have unveiled immense potential in this regard. By virtue of their ability to model long-range dependencies with linear or near-linear computational complexity, SSMs/Mamba offer a fresh, highly promising perspective for surmounting the aforementioned challenges.

Motivated by the complementary virtues of Transformer architectures in learning rich, nuanced feature representations and Mamba’s unparalleled efficiency in modeling long-range dependencies, we formulate a central hypothesis that drives this work: a carefully crafted hybrid model, synergistically integrating the self-attention mechanism of Transformers with the state-space modeling prowess of Mamba, can yield a substantially more potent yet computationally efficient framework for Document Layout Analysis (DLA). To this end, we propose DocuMambaFormer, a novel hybrid network architecture that judiciously combines the architectural strengths of both paradigms in a non-trivial manner. Our key insight is that Mamba blocks, with their inherent ability to model long-range spatial interactions through linear state-space transformations, are ideally suited for capturing the global structural coherence across document layouts—effectively discerning relationships between spatially distant elements (e.g., headers, footers, and body text regions) without incurring the quadratic complexity overhead of self-attention. Conversely, we posit that Transformer blocks retain a unique advantage in fine-grained local feature refinement and intricate cross-modal fusion (e.g., integrating visual cues from layout structures with textual semantics), where their dense, token-level attention interactions prove indispensable. This deliberate division of labor fosters a symbiotic relationship between the two modules, potentially unlocking superior modeling capabilities for complex, multi-entity document layouts while simultaneously mitigating the notorious computational bottlenecks associated with pure Transformer-based models.

To rigorously validate the efficacy of our proposed hybrid architecture, we conduct comprehensive experiments on not only well-established but also particularly challenging DLA benchmarks, chosen for their diversity and representativeness. Specifically, we evaluate our model on the widely adopted PubLayNet dataset (Zhong et al., 2019), which primarily comprises scientific articles with relatively standardized layouts, and more critically, on the recently introduced $D ^ { 4 } L A$ (Da et al., 2023), a significantly more demanding benchmark explicitly designed to overcome the homogeneity and simplicity limitations of prior datasets. $D ^ { 4 } L A$ encompasses a remarkably diverse set of 12 distinct document types (ranging from academic papers and financial reports to magazines and patents) and 27 fine-grained layout categories, thereby offering a far more realistic and challenging testbed for evaluating the generalizability and robustness of DLA models. Our empirical results compellingly demonstrate that LayoutMamba consistently sets a new state-of-the-art performance on both PubLayNet and $D ^ { 4 } L A$ , with particularly pronounced gains observed on the more complex and heterogeneous $D ^ { 4 } L A$ benchmark. These findings strongly corroborate our central hypothesis, affirming the substantial benefits of integrating State Space Model (SSM) principles into traditional Transformer-dominated document layout analysis frameworks.

The principal contributions of this work are succinctly summarized as follows:

1. We introduce LayoutMamba, the hybrid Transformer-Mamba architecture specifically tailored for document analysis tasks, which achieves a remarkable parameter-performance Pareto improvement: our model not only matches the accuracy of prior state-of-the-art Transformer-based methods but does so with a substantial 30% reduction in model parameters. This pioneering work effectively opens up an alternative, more efficient paradigm to the long-standing Transformer-dominated landscape in document understanding.

2. We propose a novel Geometric Overlap Penalty (GOP) loss function, meticulously designed for document object detection scenarios, which explicitly imposes geometric constraints to minimize bounding box overlaps through an innovative combination of Intersection-over-Union (IoU) regularization and spatial repulsion terms. Our ablation studies confirm that GOP loss consistently yields more precise layout element localization and better overall detection accuracy.

3. Through extensive experiments on two major, highly representative benchmarks, we demonstrate that our method not only establishes new state-of-the-art results in document layout analysis but also achieves a remarkable 8× parameter compression compared to competitive Transformer-based baselines, thereby validating the efficiency and effectiveness of our hybrid Mamba-Transformer design.

## 2 Related Work

Document Layout Analysis (DLA) has emerged as a critical area of research in computer vision, focusing on the identification and localization of various components within documents. Early approaches to DLA have treated it as an object detection problem, leveraging general-purpose detectors (Ren et al., 2015). Recent advancements in this field have integrated Convolutional Neural Networks (CNNs) with Transformer-based self-attention mechanisms to enhance image representation learning. These approaches extract grid features from CNNs or utilize region features from object detection frameworks to improve performance (Dosovitskiy et al., 2020; Yang et al., 2019b; Carion et al., 2020). Real-time object detection models, particularly YOLO, have been pivotal in DLA due to their balance of accuracy and speed. YOLOv7 (Wang et al., 2023) introduced the E-Elan structure, which enhances the model’s capabilities without modifying the underlying architecture, while YOLOv10 (Wang et al., 2024a) implemented a dual label assignment strategy that combines one-tomany and one-to-one assignments, providing richer supervision for efficient end-to-end deployment. These innovations affirm YOLO’s status as a leading solution for real-time inference in document analysis.

Recently, Mamba-YOLO (Wang et al., 2024c) has been proposed as the first approach to integrate State Space Models (SSMs) into object detection frameworks, effectively mitigating the computational burden imposed by the quadratic complexity of self-attention mechanisms. Beyond merely optimizing the core of the SSM, this work introduces task-specific adaptations to leverage the strengths of YOLO for object detection. In particular, the incorporation of the LSBlock and RGBlock modules enables more precise capture of local dependencies within images, thereby significantly enhancing the model’s robustness. However, the challenge of effectively fusing global and local feature representations, while preserving efficiency to further boost detection performance—remains an open question in the field.

In the visual domain, state space models (SSMs) (Gu & Dao, 2023) have been explored for their linear complexity with respect to input size, addressing the computational inefficiencies faced by Transformers in long sequence state space modeling. Vision Mamba (Zhu et al., 2024a) introduced a pure visual backbone network model based on selective state space models, incorporating cross-scan modules to enable 2D selective scanning of images, thus enhancing visual processing capabilities while maintaining linear complexity without sacrificing global receptive fields. MambaVision (Hatamizadeh & Kautz, 2024) utilized a hierarchical architecture, fusing Mamba and Transformer modules in the final two stages, replacing causal convolutions with regular convolutions and adding a symmetric branch. Inspired by these advancements, we propose the LayoutMamba architecture, which synergistically integrates Mamba and Transformer modules. Our approach emphasizes efficient inference and enhances global receptive field effects for document analysis, demonstrating significant potential in improving the accuracy and efficiency of layout recognition tasks.

## 3 Methodology

## 3.1 Preliminary: SSMs and Mamba

The classical State Space Model (SSM) defines a linear time-invariant dynamical system that processes sequential data by maintaining a latent state representation. Given an input sequence $\boldsymbol { x } ( t ) \in \mathbb { R } ^ { L }$ , the model updates a hidden state $h ( t ) \in \mathbb { R } ^ { N }$ to capture temporal dependencies and produces an output $\ b { y } ( t ) \in \mathbb { R } ^ { L }$ . The standard formulation is given as:

$$
h ^ { \prime } ( t ) = \mathbf { A } h ( t ) + \mathbf { B } x ( t ) , \qquad y ( t ) = \mathbf { C } h ( t ) ,\tag{1}
$$

where $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ is the state matrix, $\mathbf { B } \in \mathbb { R } ^ { N \times L }$ and $\mathbf { C } \in \mathbb { R } ^ { L \times N }$ represent the input and output projection matrix, respectively.

Despite its efficiency and strong inductive bias, the linear nature of classical SSMs limits their capacity to model complex, long-range dependencies inherent in real-world sequences. To address this, Mamba (Gu & Dao, 2023) introduces an enhanced formulation known as the input-dependent selective state space model (S6), which significantly improves the expressiveness of SSMs by incorporating nonlinearity, discretization, and adaptive temporal dynamics.

Specifically, S6 augments the conventional SSM with a learnable time-step parameter $\Delta$ and employs a zero-order hold (ZOH) mechanism to discretize the continuous dynamics. This enables the model to perform input-conditioned selective scanning, dynamically modulating the state update path based on the current input. Such design introduces input-adaptive recurrence and nonlinearity into the state evolution, thereby enhancing the model’s ability to capture both local and global contextual dependencies across varying temporal resolutions.

$$
h _ { t } = \overline { { \mathbf { A } } } h _ { t - 1 } + \overline { { \mathbf { B } } } x _ { t } , \qquad y _ { t } = \mathbf { C } h _ { t } ,\tag{2}
$$

$$
{ \bf \overline { { A } } } = \exp ( \Delta { \bf A } ) , \qquad { \bf \overline { { B } } } = ( \Delta { \bf A } ) ^ { - 1 } ( \exp ( \Delta { \bf A } ) - { \bf I } ) \cdot \Delta { \bf B } .\tag{3}
$$

## 3.2 The Overview of the LayoutMamba

In this section, we introduce the Vision Residual State-Space Block (VRSSBlock), the core building module of our proposed architecture. As illustrated in Fig. 1, the VRSSBlock seamlessly integrates the strengths of structured state space models (SSMs) for efficient long-range dependency modeling, residual learning for stabilized deep network training, and gated MLP variants for enhanced feature transformation. This synergistic design enables VRSSBlock to capture both global spatial contexts and local channel-wise nuances, achieving superior representation power at a competitive computational cost.

The overall workflow of VRSSBlock can be decoupled into three successive stages: (i) Dimensionality Alignment and Projection, (ii) Global Spatial Information Aggregation via SS2D, and (iii) Channel-wise Feature Refinement through Residual Gated Block. Mathematically, given an input feature tensor $\mathbf { X } ^ { l - 1 } \in$ $\mathbb { R } ^ { C \times H \times W }$ from the $( l - 1 )$ -th layer, the VRSSBlock transforms it into an enriched representation $\mathbf { X } ^ { l } \in$ $\mathbb { R } ^ { C \times H \times W }$ through the following pipeline.

Stage 1: Dimensionality Alignment and Projection. To facilitate a unified feature space for subsequent transformations, we first employ a projection convolution module to map the input $\bar { \mathbf X } ^ { l - 1 }$ into a higherdimensional space $\mathbf { X } ^ { l - \frac { 1 } { 2 } } \in \bar { \mathbb { R } } ^ { \bar { C } ^ { \prime } \times H \times \bar { W } }$ , where $C ^ { \prime }$ is the hidden dimension, typically $C ^ { \prime } = C$ for simplicity, but can be adjusted for model scaling. This stage is formulated as:

$$
\mathbf { X } ^ { l - \frac { 1 } { 2 } } = \sigma \left( \operatorname { B a t c h N o r m } \left( \operatorname { C o n v } _ { 1 \times 1 } ( \mathbf { X } ^ { l - 1 } ) \right) \right) ,\tag{4}
$$

where $\sigma ( \cdot )$ denotes the nonlinear SiLU activation function (Ramachandran et al., 2017), and $\mathbf { C o n v } _ { 1 \times 1 } ( \cdot )$ is a $1 \times 1$ convolution with learnable weights to adjust the channel dimension from $C$ to $C ^ { \prime }$ . The BatchNorm layer ensures stabilized feature distributions, mitigating internal covariate shift. This projection step is crucial as it preconditions the feature space for the more complex transformations to follow.

In parallel, to preserve the original input information and facilitate residual learning (He et al., 2016), we introduce a shortcut connection that optionally adapts the channel dimension of $\mathbf { X } ^ { l - 1 }$ via:

$$
\mathbf { X } _ { \mathrm { s h o r t c u t } } ^ { l - 1 } = \left\{ \begin{array} { l l } { \mathrm { C o n v } _ { 1 \times 1 } ^ { \mathrm { a d a p t } } ( \mathbf { X } ^ { l - 1 } ) } & { \mathrm { i f } \ C \neq C ^ { \prime } , } \\ { \mathbf { X } ^ { l - 1 } } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{5}
$$

Here, $\mathrm { C o n v } _ { 1 \times 1 } ^ { \mathrm { a d a p t } } ( \cdot )$ is another $1 \times 1$ convolution to align the channel dimension from $C$ to $C ^ { \prime }$ , ensuring that the residual addition is well-defined. This design echoes the philosophy of ResNet and its variants (Xie et al., 2016; Liu et al., 2022), where the identity mapping serves as a gradient highway, significantly easing the optimization of deep networks.

Stage 2: Global Spatial Information Aggregation via SS2D. Following the feature projection, the intermediate representation $\mathbf { X } ^ { l - \frac { 1 } { 2 } }$ is fed into a Local Spatial Block (LSBlock) for preliminary spatial context aggregation:

$$
\mathbf { X } _ { \mathrm { L S } } ^ { l - 1 } = \mathrm { L S B l o c k } ( \mathbf { X } ^ { l - \frac { 1 } { 2 } } ) .\tag{6}
$$

The LSBlock, similar to the local feature enhancers used in (Liu et al., 2022), aims to capture short-to-medium range spatial dependencies. Subsequently, we apply Layer Normalization to stabilize the feature distribution, followed by the core SS2D module (Liu et al., 2024; Gu & Dao, 2023), for efficient long-range spatial modeling:

$$
\mathbf { X } ^ { l - \frac { 3 } { 4 } } = \mathbf { S S 2 D } \left( \operatorname { L a y e r N o r m } ( \mathbf { X } _ { \mathrm { L S } } ^ { l - 1 } ) \right) .\tag{7}
$$

The SS2D operator is the cornerstone of our global context modeling, which leverages structured state space models (SSMs) to efficiently process 2D inputs. SS2D comprises three key steps:

Scan Expansion: The input feature map is systematically decomposed into multiple 1D sequences along four symmetric directions (top-down, bottom-up, left-right, right-left), effectively capturing spatial anisotropies.

S6 Block: Each sequence is processed by a state space model parameterized by matrices $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ $\mathbf { B } \in \mathbb { R } ^ { N \times 1 } , \mathbf { C } \in \mathbb { R } ^ { 1 \times \hat { N } }$ , and $\mathbf { D } \in \mathbb { R } ^ { 1 \times 1 }$ , discretizing the continuous SSM dynamics:

$$
\begin{array} { r } { \mathbf { h } _ { t } = \overline { { \mathbf { A } } } \mathbf { h } _ { t - 1 } + \overline { { \mathbf { B } } } x _ { t } , } \\ { y _ { t } = \mathbf { C } \mathbf { h } _ { t } + \mathbf { D } x _ { t } , } \end{array}\tag{8}
$$

where $\overline { { \mathbf { A } } } = \exp ( \Delta \mathbf { A } )$ and $\overline { { \mathbf { B } } } = ( \exp ( \Delta \mathbf { A } ) - \mathbf { I } ) \mathbf { A } ^ { - 1 } \Delta \mathbf { B }$ are discretization results with step size $\Delta$

Scan Merge: The directional sequences are re-assembled, aggregating global information into $\mathbf { X } ^ { l - \frac { 3 } { 4 } }$ . The output of SS2D is then combined with the LSBlock output via a residual connection, enhanced by DropPath for regularization:

$$
\mathbf { X } _ { \mathrm { S S 2 D } } ^ { l - \frac { 1 } { 2 } } = \mathbf { X } ^ { l - \frac { 1 } { 2 } } + \operatorname { D r o p P a t h } ( \mathbf { X } ^ { l - \frac { 3 } { 4 } } ) .\tag{9}
$$

This formulation ensures that the network retains the original feature representation while injecting global spatial awareness.

Stage 3: Channel-wise Feature Refinement via Residual Gated Block. Building upon the gated MLP designs introduced in (Wang et al., 2024c), we employ the Residual Gated (RG) Block to enhance feature expressiveness in our model. The RG Block operates on $\mathbf { X } _ { \mathrm { S S 2 D } } ^ { l - \frac { 1 } { 2 } }$ as follows:

$$
\mathbf { X } _ { \mathrm { R G } } ^ { l } = \mathrm { R G B l o c k } \left( \mathrm { L a y e r N o r m } ( \mathbf { X } _ { \mathrm { S S 2 D } } ^ { l - \frac { 1 } { 2 } } ) \right) + \mathbf { X } _ { \mathrm { S S 2 D } } ^ { l - \frac { 1 } { 2 } } .\tag{10}
$$

Internally, the RG Block splits the normalized input into two branches: ${ \mathcal { F } } _ { A }$ and $\mathcal { F } _ { B }$ , responsible for local and global feature modulation, respectively. The local branch undergoes a depthwise convolution-based Position Encoding Module (PEM), formulated as:

$$
\mathcal { V } ( \mathcal { F } _ { A } ) = \Phi \left( \operatorname { D W C o n v } ( \mathcal { F } _ { A } ) \oplus \mathcal { F } _ { A } \right) ,\tag{11}
$$

where $\Phi ( \cdot )$ is the GELU activation function, DWConv(·) denotes depthwise separable convolution, and ⊕ represents element-wise addition. Meanwhile, the global branch $\mathcal { F } _ { B }$ is transformed via a linear layer $\tau _ { \mathrm { g l o b a l } } ( \cdot )$

$$
{ \mathcal { R } } _ { \mathrm { g l o b a l } } = { \mathcal { T } } _ { \mathrm { g l o b a l } } ( { \mathcal { F } } _ { B } ) .\tag{12}
$$

The gated fusion mechanism then combines these branches:

$$
\mathcal { R } _ { \mathrm { f u s i o n } } = \mathcal { R } _ { \mathrm { g l o b a l } } \odot \mathcal { Y } ( \mathcal { F } _ { A } ) ,\tag{13}
$$

where $\odot$ denotes element-wise multiplication. Finally, another linear layer $\tau _ { \mathrm { { f u s e } } } ( \cdot )$ projects the fused representation, and a residual connection is added:

$$
\mathbf { X } _ { \mathrm { R G } } ^ { l } = \mathcal { T } _ { \mathrm { f u s e } } ( \mathcal { R } _ { \mathrm { f u s i o n } } ) \oplus \mathcal { F } _ { A } .\tag{14}
$$

This gating mechanism selectively amplifies salient features while suppressing less informative ones, enriching the channel-wise contextual understanding.

Final Residual Aggregation. The ultimate output of VRSSBlock is obtained by merging the RG Block output $\mathbf { X } _ { \mathrm { R G } } ^ { l }$ with the initial shortcut connection $\mathbf { X } _ { \mathrm { s h o r t c u t } } ^ { \bar { l } - 1 } ;$

$$
\mathbf { X } ^ { l } = \mathbf { X } _ { \mathrm { R G } } ^ { l } + \mathbf { X } _ { \mathrm { s h o r t c u t } } ^ { l - 1 } .\tag{15}
$$

This outer residual connection closes the loop, ensuring that VRSSBlock learns residual refinements rather than complete transformations, drastically improving trainability and generalization.

![](images/84c96a9a6b2ff0cbcd4a33531291e4d50176bde57ca9433eb72f44728536a642.jpg)  
Figure 1: Comprehensive overview of the proposed efficient document layout analysis framework leveraging hybrid Mamba-Transformer architectures.

## 3.3 Multi-Head Attention

To further enhance the representational capacity of our VRSS Block outputs, we employ a standard multi-head attention mechanism formulated as:

$$
{ \mathrm { A t t e n t i o n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = { \mathrm { S o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { h } } } } \right) \mathbf { V } ,\tag{16}
$$

where Q, K, $\mathbf { V } \in \mathbb { R } ^ { B \times H \times N \times d _ { h } }$ denote the query, key, and value projections, H is the number of attention heads, $d _ { h } = C / H$ is the dimension of each head, B is the batch size, and N is the sequence length.

This attention operation allows the model to dynamically aggregate information across the sequence dimension, facilitating more expressive feature representations. The multi-head outputs are subsequently concatenated and linearly projected to yield the final output.

## 3.4 Geometric Overlap Penalty Loss

In traditional object detection tasks, scenarios involving dense prediction boxes are commonly observed (Li et al., 2017; Ling et al., 2023; Yang et al., 2019a; Zhou et al., 2019). This phenomenon similarly exists in document layout analysis tasks, yet datasets in this domain exhibit unique characteristics: although containing numerous densely arranged bounding boxes, they rarely present inter-box overlaps. This distinctive property stems from the nature of document data, as such overlaps would generally correspond to content superposition that substantially impairs document readability. Given this property, this study proposes Geometric Overlap Penalty Loss (GOPLoss) that explicitly penalizes the degree of overlap between predicted boxes. The mathematical formulation of GOPLoss is defined as:

$$
\mathrm { G O P L o s s } = \frac { 1 } { n ^ { 2 } } \sum _ { 1 \leq i < j \leq n } \mathrm { I o U } ( B _ { i } , B _ { j } ) ,\tag{17}
$$

where n denotes the total number of predicted boxes, $B _ { k }$ represents the k-th predicted box, and $\mathrm { I o U } ( B _ { i } , B _ { j } )$ calculates the intersection over union between two boxes. This loss function normalizes the sum of IoU values across all box pairs using $n ^ { 2 }$ as the denominator, aiming to eliminate scale variations caused by different numbers of predictions and enable the model to focus on relative overlap degrees rather than absolute quantities.

Regarding computational efficiency, while the algorithm complexity is $O ( n ^ { 2 } )$ , the practical computation overhead remains negligible due to the typically small number of surviving predictions after Non-Maximum Suppression (NMS) (Felzenszwalb et al., 2009).

It should be noted that GOPLoss functions as an unsupervised regularization term that operates independently of ground truth labels. Therefore, it must be incorporated as an auxiliary component within the total loss function. This study adopts a multi-task weighted optimization strategy, with the composite loss function defined as:

$$
l o s s = \lambda _ { 1 } \cdot l o s s _ { b o x } + \lambda _ { 2 } \cdot l o s s _ { c l s } + \lambda _ { 3 } \cdot l o s s _ { d f l } + \lambda _ { 4 } \cdot l o s s _ { g o p } ,\tag{18}
$$

where $\lambda _ { i } ,$ with $i \in \{ 1 , 2 , 3 , 4 \}$ , denote the weights of the loss, $l o s s _ { b o x } , l o s s _ { c l s } ,$ and $l o s s _ { d f l }$ correspond to bounding box regression loss, category classification loss, and Distribution Focal Loss respectively, while $l o s s _ { g o p }$ denotes the geometric overlap penalty loss. Through this weighted configuration, the model simultaneously optimizes fundamental detection objectives while explicitly constraining spatial independence among predictions, thereby adapting to the low-overlap characteristics inherent in document layout analysis tasks.

## 4 Experiments

## 4.1 Dataset

PubLayNet is a document layout analysis dataset from IBM, comprising 360K PDF documents from PubMed XML files. It annotates five layout categories (text, title, list, figure, table) with bounding boxes. The dataset contains 335,703 training, 11,245 validation, and 11,405 test images. We train on the training set and report results on the validation set.

$D ^ { 4 } L A$ (Da et al., 2023), from Alibaba, is designed for real-world document layout analysis. It encompasses 12 document types and 27 fine-grained layout categories (DocTitle, ListText, LetterHead, Question, RegionList, TableName, FigureName, etc.), with 8,868 training and 2,224 validation images. The dataset features documents with skewness, blur, and diverse layout structures, making it more challenging than PubLayNet for evaluating model robustness.

Table 1: Document Layout Detection mAP @ IoU [0.50:0.95] on PubLayNet validation set.
<table><tr><td>Models</td><td>Year</td><td>#Param</td><td>Text</td><td>Title</td><td>List</td><td>Table</td><td>Figure</td><td>mAP</td></tr><tr><td>VSR (Zhang et al., 2021)</td><td>2021</td><td></td><td>0.967</td><td>0.931</td><td>0.947</td><td>0.974</td><td>0.964</td><td>0.957</td></tr><tr><td>TRDLU (Yang &amp; Hsu, 2022)</td><td>2022</td><td></td><td>0.958</td><td>0.921</td><td>0.975</td><td>0.976</td><td>0.966</td><td>0.959</td></tr><tr><td>DiT-L (Li et al., 2022a)</td><td>2022</td><td>304M</td><td>0.944</td><td>0.893</td><td>0.960</td><td>0.978</td><td>0.972</td><td>0.949</td></tr><tr><td>UDoc (Gu et al., 2023)</td><td>2022</td><td>272M</td><td>0.939</td><td>0.885</td><td>0.937</td><td>0.973</td><td>0.964</td><td>0.939</td></tr><tr><td>VGT (Da et al., 2023)</td><td>2023</td><td>174M</td><td>0.950</td><td>0.939</td><td>0.968</td><td>0.981</td><td>0.971</td><td>0.962</td></tr><tr><td>DETR (Shehzadi et al., 2023)</td><td>2023</td><td></td><td>0.947</td><td>0.918</td><td>0.964</td><td>0.981</td><td>0.975</td><td>0.957</td></tr><tr><td>DoPTA-HR (SR et al., 2024)</td><td>2024</td><td>85M</td><td>0.944</td><td>0.895</td><td>0.957</td><td>0.977</td><td>0.970</td><td>0.949</td></tr><tr><td>LayoutMamba-T</td><td>2025</td><td>6.9M</td><td>0.955</td><td>0.883</td><td>0.961</td><td>0.985</td><td>0.983</td><td>0.953</td></tr><tr><td>LayoutMamba-B</td><td>2025</td><td>22M</td><td>0.953</td><td>0.902</td><td>0.963</td><td>0.986</td><td>0.984</td><td>0.957</td></tr><tr><td>LayoutMamba-L</td><td>2025</td><td>67M</td><td>0.957</td><td>0.934</td><td>0.965</td><td>0.985</td><td>0.985</td><td>0.963</td></tr></table>

## 4.2 Implementation Details

Model Architecture & Fine-tuning. Our LayoutMamba model utilizes the previously described Mamba-Transformer hybrid backbone for feature extraction, synergistically combining the linear-complexity advantages of Mamba with the dense attention capabilities of Transformer. To effectively integrate multi-scale features, we adopt the PAFPN (Path Aggregation Feature Pyramid Network) (Liu et al., 2018) as our neck module, which facilitates direct interaction between non-adjacent feature levels through progressive fusion of adjacent stage features. This design choice is motivated by PAFPN’s proven efficacy in enhancing object detection performance through richer feature representations.

Optimization Strategy. For model optimization, we employ the SGD optimizer with a linear warmup strategy for the first 3 epochs (initial learning rate of 0.1), followed by training at a learning rate of 0.01. On PubLayNet, we train LayoutMamba for 300 epochs with a batch size of 128 distributed across 4 × 40GB NVIDIA A100 GPUs, leveraging data parallelism to accelerate training. Given the relatively smaller scale of $D ^ { 4 } L A$ , we train our model for 300 epochs on a single A100 GPU with a batch size of 16, keeping all other training hyperparameters identical to those used for PubLayNet. This consistency ensures a fair comparison across datasets.

## 4.3 Comparison to Existing Approaches

We conducted extensive experiments on PubLayNet, comparing LayoutMamba with state-of-the-art vision Transformer backbones for document layout analysis, as summarized in Table 1, where LayoutMamba achieves significant improvements, particularly in challenging categories such as “list,” “table,” and “figure.” LayoutMamba-T attains a comparable mAP of 95.1 to the larger DoPTA-HR model while reducing parameters from 85M to 6.9M, and LayoutMamba-L achieves a state-of-the-art mAP of 96.3, highlighting the effectiveness of our hybrid architecture. On the more challenging $D ^ { 4 } L A$ dataset (Table 2), characterized by diverse real-world scenarios, finer-grained categories, and data imbalance leading to lower mAP scores, LayoutMamba shows strong resilience; LayoutMamba-T outperforms LayoutLMv3-Base by 5.8% mAP despite the latter having nearly 20× more parameters, and achieves state-of-the-art results in key categories including DocTitle (88.2% AP), Paratext (85.1% AP), and RegionKV (84.5% AP).

## 4.4 Ablation Studies

To rigorously validate the architectural efficacy and design rationale of our proposed framework, we conduct a series of systematic ablation experiments on the challenging document layout detection task, utilizing the $D ^ { 4 } L A$ validation set. Our investigation centers on two pivotal components that underpin the superior performance of our model: (1) the evolutionary advancements in the state-space block, where our novel Vision Residual State-Space Block (VRSSBlock) extends the foundational VSSBlock design with strategic architectural refinements, and (2) the synergistic integration of multi-head attention, which further amplifies the model’s contextual reasoning capabilities.

Table 2: Document Layout Detection mAP @ IoU [0.50:0.95] on $D ^ { 4 } L A$ validation set.
<table><tr><td></td><td rowspan=1 colspan=3>Models</td><td rowspan=1 colspan=3>#Param</td><td rowspan=1 colspan=2>DocTitle</td><td rowspan=1 colspan=2>ListText</td><td rowspan=1 colspan=3>LetterHead</td><td rowspan=1 colspan=2>Question</td><td rowspan=1 colspan=2>RegionList</td><td rowspan=1 colspan=3>TableName</td><td rowspan=1 colspan=2>FigureName</td><td rowspan=1 colspan=1>Footer</td><td rowspan=1 colspan=1>Number</td></tr><tr><td rowspan=7 colspan=1></td><td rowspan=7 colspan=3>DiT-B (Li et al., 2022a)LayoutLMv3-Base (Huang et al., 2022)VGT (Da et al., 2023)DoPTA-HR (SR et al., 2024)LayoutMamba-TLayoutMamba-BLayoutMamba-L</td><td rowspan=2 colspan=3>87M133M</td><td rowspan=1 colspan=2>73.1</td><td rowspan=1 colspan=2>70.6</td><td rowspan=1 colspan=3>82.2</td><td rowspan=1 colspan=2>55.0</td><td rowspan=1 colspan=2>80.1</td><td rowspan=1 colspan=3>68.4</td><td rowspan=1 colspan=2>51.8</td><td rowspan=1 colspan=1>81.2</td><td rowspan=2 colspan=1>83.282.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>66.8</td><td rowspan=1 colspan=2>56.5</td><td rowspan=1 colspan=3>78.5</td><td rowspan=1 colspan=2>39.3</td><td rowspan=1 colspan=2>72.1</td><td rowspan=1 colspan=3>64.3</td><td rowspan=1 colspan=2>32.1</td><td rowspan=1 colspan=1>72.2</td></tr><tr><td rowspan=1 colspan=2>174</td><td rowspan=1 colspan=2>M</td><td rowspan=1 colspan=2>72.6</td><td rowspan=1 colspan=2>71.3</td><td rowspan=1 colspan=3>82.3</td><td rowspan=1 colspan=2>63.9</td><td rowspan=1 colspan=2>80.2</td><td rowspan=1 colspan=3>68.4</td><td rowspan=1 colspan=2>46.6</td><td rowspan=1 colspan=1>79.7</td><td rowspan=1 colspan=1>83.2</td></tr><tr><td rowspan=1 colspan=2>85M</td><td></td><td rowspan=1 colspan=2>73.11</td><td rowspan=1 colspan=2>72.46</td><td rowspan=1 colspan=3>82.07</td><td rowspan=1 colspan=2>77.42</td><td rowspan=1 colspan=2>79.32</td><td rowspan=1 colspan=3>67.08</td><td rowspan=1 colspan=2>56.86</td><td rowspan=1 colspan=1>77.88</td><td rowspan=1 colspan=1>83.15</td></tr><tr><td rowspan=1 colspan=3>6.9M</td><td rowspan=1 colspan=1>74.2</td><td></td><td rowspan=1 colspan=2>69.5</td><td rowspan=1 colspan=3>85.3</td><td rowspan=1 colspan=2>68.8</td><td rowspan=1 colspan=2>82.1</td><td rowspan=1 colspan=3>61.7</td><td rowspan=1 colspan=2>49.5</td><td rowspan=1 colspan=1>77.2</td><td rowspan=1 colspan=1>81.4</td></tr><tr><td rowspan=2 colspan=3>22M67M</td><td rowspan=2 colspan=1>74.675.4</td><td></td><td rowspan=2 colspan=2>7273.9</td><td rowspan=2 colspan=3>83.983.9</td><td rowspan=1 colspan=2>64.9</td><td rowspan=1 colspan=2>82.4</td><td rowspan=1 colspan=3>65.2</td><td rowspan=1 colspan=2>52</td><td rowspan=1 colspan=1>78.5</td><td rowspan=2 colspan=1>81.582.4</td></tr><tr><td></td><td rowspan=1 colspan=2>67.7</td><td rowspan=1 colspan=2>82.7</td><td rowspan=1 colspan=3>65.3</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>79.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=17></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Models</td><td rowspan=1 colspan=3>#Param</td><td rowspan=1 colspan=12>ParaTitleRegionTitleLetterDear OtherText Abstract</td><td rowspan=1 colspan=3>TableEquation</td><td rowspan=1 colspan=2>PageHeader</td><td rowspan=1 colspan=1>Catalog</td></tr><tr><td rowspan=7 colspan=1></td><td rowspan=7 colspan=3>DiT-B (Li et al., 2022a)LayoutLMv3-Base (Huang et al., 2022)VGT (Da et al., 2023)DoPTA-HR (SR et al., 2024)LayoutMamba-TLayoutMamba-BLayoutMamba-L</td><td rowspan=1 colspan=3>87M</td><td rowspan=1 colspan=2>63.2</td><td rowspan=1 colspan=3>67.5</td><td rowspan=1 colspan=5>74.5      59.2</td><td rowspan=1 colspan=2>73.8</td><td rowspan=1 colspan=1>86.2</td><td></td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=2>56.5</td><td rowspan=1 colspan=1>44.8</td></tr><tr><td rowspan=1 colspan=3>133M</td><td rowspan=1 colspan=2>55.6</td><td rowspan=1 colspan=3>59.5</td><td rowspan=1 colspan=5>70.8      50.8</td><td rowspan=1 colspan=2>68.2</td><td rowspan=1 colspan=1>80.6</td><td></td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>53.1</td><td rowspan=1 colspan=1>37.3</td></tr><tr><td rowspan=1 colspan=3>174M</td><td rowspan=1 colspan=2>63.0</td><td rowspan=1 colspan=3>67.2</td><td rowspan=1 colspan=5>76.7      60.0</td><td rowspan=1 colspan=2>80.4</td><td rowspan=1 colspan=1>86.0</td><td></td><td rowspan=1 colspan=1>19.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>56.9</td><td rowspan=1 colspan=1>40.9</td></tr><tr><td rowspan=1 colspan=3>85M6.9M</td><td rowspan=1 colspan=2>64.07</td><td rowspan=1 colspan=3>65.17</td><td rowspan=1 colspan=5>72.7     61.25</td><td rowspan=1 colspan=2>78.25</td><td rowspan=1 colspan=1>86.9</td><td></td><td rowspan=1 colspan=1>32.26</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8.22</td><td rowspan=1 colspan=1>60.98</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td rowspan=1 colspan=5>61.2      65.6</td><td rowspan=1 colspan=5>70.5      57.1</td><td rowspan=1 colspan=2>75.4</td><td rowspan=1 colspan=1>87.6</td><td></td><td rowspan=1 colspan=1>9.32</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55.3</td><td rowspan=1 colspan=1>32.2</td></tr><tr><td rowspan=1 colspan=1>ba-B</td><td rowspan=2 colspan=5>22M    62.667M    64.1</td><td rowspan=1 colspan=3>67.4</td><td rowspan=1 colspan=5>72.4      58.4</td><td rowspan=1 colspan=3>74.9   86.9</td><td rowspan=1 colspan=2>26.1</td><td rowspan=1 colspan=2>57.2</td><td rowspan=1 colspan=1>35.3</td></tr><tr><td rowspan=1 colspan=3>66.5</td><td rowspan=1 colspan=5>71.1      59.5</td><td rowspan=1 colspan=5>74.8   89.9   25.9</td><td rowspan=1 colspan=2>51.9</td><td rowspan=1 colspan=1>38.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=16></td><td rowspan=1 colspan=8></td></tr><tr><td rowspan=1 colspan=3>Models</td><td rowspan=1 colspan=5>#ParamParaText</td><td rowspan=1 colspan=2>Date</td><td rowspan=1 colspan=3>LetterSign</td><td rowspan=1 colspan=4>RegionKV Author</td><td rowspan=1 colspan=1>Figure</td><td rowspan=1 colspan=2>Reference</td><td></td><td rowspan=1 colspan=2>PageFooter</td><td rowspan=1 colspan=1>PageNumber</td><td rowspan=1 colspan=1>mAP</td></tr><tr><td rowspan=7 colspan=3>DiT-B (Li et al., 2022a)LayoutLMv3-Base (Huang et al., 2022)VGT (Da et al., 2023)DoPTA-HR (SR et al., 2024)LayoutMamba-TLayoutMamba-BLayoutMamba-L</td><td rowspan=2 colspan=3>87M133M</td><td rowspan=1 colspan=2>86.4</td><td rowspan=1 colspan=2>69.7</td><td rowspan=1 colspan=3>71.6</td><td rowspan=1 colspan=2>68.8</td><td rowspan=1 colspan=2>66.0</td><td rowspan=1 colspan=1>77.2</td><td rowspan=1 colspan=2>83.4</td><td></td><td rowspan=1 colspan=2>65.5</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=1>67.7</td></tr><tr><td rowspan=1 colspan=2>81.6</td><td rowspan=1 colspan=2>62.5</td><td rowspan=1 colspan=3>60.4</td><td rowspan=1 colspan=2>59.4</td><td rowspan=1 colspan=2>59.3</td><td rowspan=1 colspan=1>72.2</td><td rowspan=1 colspan=2>74.9</td><td></td><td rowspan=1 colspan=2>62.1</td><td rowspan=1 colspan=1>52.8</td><td rowspan=1 colspan=1>60.5</td></tr><tr><td rowspan=1 colspan=3>174M</td><td rowspan=1 colspan=2>86.2</td><td rowspan=1 colspan=2>71.3</td><td rowspan=1 colspan=3>75.5</td><td rowspan=1 colspan=2>70.1</td><td rowspan=1 colspan=2>67.6</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=2>85.6</td><td></td><td rowspan=1 colspan=2>66.5</td><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>68.8</td></tr><tr><td rowspan=1 colspan=3>85M</td><td rowspan=1 colspan=2>85.75</td><td rowspan=1 colspan=2>71.4</td><td rowspan=1 colspan=3>76.31</td><td rowspan=1 colspan=2>70.3</td><td rowspan=1 colspan=2>70.66</td><td rowspan=1 colspan=1>75.73</td><td rowspan=1 colspan=2>84.45</td><td></td><td rowspan=1 colspan=2>65.82</td><td rowspan=1 colspan=1>60.64</td><td rowspan=1 colspan=1>70.72</td></tr><tr><td rowspan=1 colspan=3>6.9M</td><td rowspan=1 colspan=2>87.8</td><td rowspan=1 colspan=2>64.7</td><td rowspan=1 colspan=3>71.3</td><td rowspan=1 colspan=2>68.2</td><td rowspan=1 colspan=2>63.5</td><td rowspan=1 colspan=1>75.4</td><td rowspan=1 colspan=2>85.6</td><td></td><td rowspan=1 colspan=2>61.3</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>66.3</td></tr><tr><td rowspan=1 colspan=3>22M</td><td rowspan=1 colspan=2>88.5</td><td rowspan=1 colspan=2>68.5</td><td rowspan=1 colspan=3>70.6</td><td rowspan=1 colspan=2>70.7</td><td rowspan=1 colspan=2>64.1</td><td rowspan=1 colspan=1>76.1</td><td rowspan=1 colspan=2>85</td><td></td><td rowspan=1 colspan=2>62.1</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>67.9</td></tr><tr><td rowspan=1 colspan=3>67M</td><td rowspan=1 colspan=2>89.1</td><td rowspan=1 colspan=2>71.3</td><td rowspan=1 colspan=3>72.2</td><td rowspan=1 colspan=2>71.2</td><td rowspan=1 colspan=2>65.6</td><td rowspan=1 colspan=1>76.4</td><td rowspan=1 colspan=2>86.7</td><td></td><td rowspan=1 colspan=2>62.9</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>69.4</td></tr></table>

Table 3: Document Layout Detection mAP @ IoU [0.50:0.95] on $D ^ { 4 } L A$ validation set with different GOPLoss weight.
<table><tr><td>GOPLoss Weight</td><td>0.0</td><td>1.0</td><td>3.0</td><td>5.0</td><td>7.5</td></tr><tr><td>mAP</td><td>57.9</td><td>58.2</td><td>58.2</td><td>58.4</td><td>57.6</td></tr></table>

Our reference baseline employs the original VSSBlock paradigm, leveraging a structured state space module for long-range dependency modeling and lightweight local convolutions for fine-grained feature extraction. As quantitatively summarized in Table 4, this baseline configuration comprises 5.8M parameters and achieves a mean Average Precision (mAP) of 60.7% at stringent IoU thresholds [0.50,0.95], establishing a competitive yet improvable foundation for subsequent ablative analysis.

Impact of VRSSBlock Evolution. The VRSSBlock refines the VSSBlock architecture by integrating residual learning, gated MLP modules, and a novel SS2D operator, leading to enhanced feature transformation. More specifically, the VRSSBlock fuses a dedicated projection module, a global spatial information aggregator that leverages structured state space models, and a residual gated block for channel-wise refinement. When we replace VSSBlock with VRSSBlock (without adding multi-head attention), the parameter count increases modestly to 6.2M, but we observe a performance gain of approximately 0.5% mAP (from 60.7% to 61.2%). This improvement demonstrates that the architectural evolution embedded in VRSSBlock contributes to more effective spatial and contextual feature modeling.

Impact of Multi-Head Attention. To further bolster the model’s representational prowess, we strategically integrate a canonical multi-head attention module downstream of the VRSSBlock output. Here, query (Q), key (K), and value (V ) projections are dynamically derived from the VRSSBlock-processed features, enabling adaptive recalibration across spatial and channel dimensions. This attentive contextualization augments global receptive field modeling by disambiguating salient regions via learned attention weights and harmonizing local and global feature hierarchies. The combined VRSSBlock + Attention configuration incurs a further parameter growth to 6.9M (+9.7% vs. VRSSBlock-only), yet yields a decisive performance leap of +2.1% mAP (from 61.2% to 63.3%). This super-additive gain underscores that attention synergizes with VRSSBlock’s structured state space modeling, creating an irreducible performance ceiling that neither component could attain in isolation.

Table 4: Document Layout Detection mAP @ IoU [0.50:0.95] on $D ^ { 4 } L A$ validation set.  
![](images/e732d19dc4a3804b611e72dea6f745ec8ecf1a539f684d4ba1be829cd5c25256.jpg)

![](images/05a3b8544469852c3d073eba6a1bd8e703e7f62f70f238701daba23e40c10726.jpg)  
(a) Model with λ<sub>4</sub> = 0.0  
(b) Model with λ<sub>4</sub> = 5.0  
Figure 2: Visual comparison of detection results under different GOPLoss weights.

Geometric Overlap Penalty Loss. To systematically evaluate the impact of GOPLoss weight on the performance of the document layout analysis model, we designed a set of experiments on the $D ^ { 4 } L A$ benchmark dataset. Based on the trade-off between training efficiency and computational cost, the experiment set the training period to 100 epochs. We selected GOPLoss weights $\lambda _ { 4 } \in \{ 0 . 0 , 1 . 0 , 3 . 0 , 5 . 0 , 7 . 5 \}$ for a systematic ablation study, aiming to explore the optimal weight configuration and its regulatory mechanism on the detection performance of different layout categories.

The experimental results are shown in Table 3. They indicate that GOPLoss can effectively enhance the model’s ability to model complex layouts through geometric constraint optimization. However, fine-tuned weight adjustment is required to achieve a multi-objective optimization balance. When $\lambda _ { 4 } = 5 . 0 $ , the model achieves the optimal overall performance $( \mathrm { m A P } = 5 8 . 4 )$ , which is 0.5 percentage points higher than the baseline model $( \lambda _ { 4 } = 0 . 0 , \mathrm { { m A P } = 5 7 . 9 ) }$ , verifying the promoting effect of moderate geometric constraints on global detection performance. However, when $\lambda _ { 4 }$ increases to 7.5, the mAP significantly drops to 57.6. It is speculated that excessive geometric regularization causes the model to deviate from the optimization direction of the basic detection task. This phenomenon reveals that there is a critical threshold for GOPLoss weight. Beyond this threshold, competitive suppression will occur between geometric constraints and semantic feature learning.

To delve deeper into the optimization mechanism of the GOPLoss, a visual comparative analysis was conducted on the models with $\lambda _ { 4 } = 0 . 0$ and $\lambda _ { 4 } = 5 . 0$ (as shown in Fig. 2). The experimental results revealed that upon the introduction of GOPLoss, the overlap of target boxes within dense text regions diminished. This phenomenon suggests that geometric constraints effectively mitigated the redundant prediction issue of adjacent detection boxes. Concurrently, the number of detection boxes generated by the model decreased, which is advantageous for enhancing the inference speed. This outcome can be ascribed to the model’s adaptive suppression of low-confidence redundant predictions.

## 5 Conclusion

We propose a hybrid architecture combining Vision Transformers and State Space Models (Mamba) for Document Layout Analysis (DLA). Our approach uses sparse attention for local feature extraction and Mamba’s linear complexity for global context modeling, achieving state-of-the-art results on PubLayNet and $D ^ { 4 } L A$ benchmarks with 8× fewer parameters. We introduce a Geometric Overlap Penalty Loss to address low-overlap bias in layout predictions. Our method demonstrates Mamba’s potential for efficient multimodal document understanding.

## References

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pp. 213–229. Springer, 2020.

Shang Chai, Liansheng Zhuang, and Fengying Yan. Layoutdm: Transformer-based diffusion model for layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18349–18358, 2023.

Kai Chen, Mathias Seuret, Jean Hennebert, and Rolf Ingold. Convolutional neural networks for page segmentation of historical document images. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 1, pp. 965–970. IEEE, 2017.

Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. Vision grid transformer for document layout analysis. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 19462–19472, 2023.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Jinsheng Fang, Hanjiang Lin, Xinyu Chen, and Kun Zeng. A hybrid network of cnn and transformer for lightweight image super-resolution. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 1103–1112, 2022.

Pedro F Felzenszwalb, Ross B Girshick, David McAllester, and Deva Ramanan. Object detection with discriminatively trained part-based models. IEEE transactions on pattern analysis and machine intelligence, 32(9):1627–1645, 2009.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Jiuxiang Gu, Ani Nenkova Nenkova, Nikolaos Barmpalios, Vlad Ion Morariu, Tong Sun, Rajiv Bhawanji Jain, Jason Wen Yong Kuen, and Handong Zhao. Unified pretraining framework for document understanding, May 18 2023. US Patent App. 17/528,061.

Kamal Gupta, Justin Lazarow, Alessandro Achille, Larry S Davis, Vijay Mahadevan, and Abhinav Shrivastava. Layouttransformer: Layout generation and completion with self-attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1004–1014, 2021.

Ali Hatamizadeh and Jan Kautz. Mambavision: A hybrid mamba-transformer vision backbone. ArXiv, abs/2407.08083, 2024. URL https://api.semanticscholar.org/CorpusID:271097462.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7132–7141, 2018.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700–4708, 2017.

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM international conference on multimedia, pp. 4083–4091, 2022.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Jianan Li, Xiaodan Liang, ShengMei Shen, Tingfa Xu, Jiashi Feng, and Shuicheng Yan. Scale-aware fast r-cnn for pedestrian detection. IEEE transactions on Multimedia, 20(4):985–996, 2017.

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In Proceedings ofthe 30th ACM international conference on multimedia, pp. 3530–3539, 2022a.

Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, and Jian Ren. Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems, 35:12934–12949, 2022b.

Qin Ling, Nor Ashidi Mat Isa, and Mohd Shahrimie Mohd Asaari. Precise detection for dense pcb components based on modified yolov8. IEEE Access, 11:116545–116560, 2023.

Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8759–8768, 2018.

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pp. 21–37. Springer, 2016.

Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. Advances in neural information processing systems, 37: 103031–103063, 2024.

Zhuang Liu, Hanzi Mao, Chaozheng Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11966–11976, 2022. URL https://api.semanticscholar.org/CorpusID:245837420.

Dario Augusto Borges Oliveira and Matheus Palhares Viana. Fast cnn-based document layout analysis. In 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), pp. 1173–1180. IEEE, 2017.

Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Swish: a self-gated activation function. arXiv: Neural and Evolutionary Computing, 2017. URL https://api.semanticscholar.org/CorpusID: 196158220.

Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

Tahira Shehzadi, Khurram Azeem Hashmi, Didier Stricker, Marcus Liwicki, and Muhammad Zeshan Afzal. Bridging the performance gap between detr and r-cnn for graphical object detection in document images. arXiv preprint arXiv:2306.13526, 2023.

Tahira Shehzadi, Didier Stricker, and Muhammad Zeshan Afzal. A hybrid approach for document layout analysis in document images. In International Conference on Document Analysis and Recognition, pp. 21–39. Springer, 2024.

Nikitha SR, Tarun Ram Menta, and Mausoom Sarkar. Dopta: Improving document layout analysis using patch-text alignment. arXiv preprint arXiv:2412.12902, 2024.

Waseem Ullah, Tanveer Hussain, Fath U Min Ullah, Mi Young Lee, and Sung Wook Baik. Transcnn: Hybrid cnn and transformer mechanism for surveillance anomaly detection. Engineering Applications of Artificial Intelligence, 123:106173, 2023.

Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, et al. Yolov10: Real-time end-to-end object detection. Advances in Neural Information Processing Systems, 37:107984–108011, 2024a.

Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7464–7475, 2023.

Jiawei Wang, Kai Hu, and Qiang Huo. Dlaformer: An end-to-end transformer for document layout analysis. In International Conference on Document Analysis and Recognition, pp. 40–57. Springer, 2024b.

Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu, and Hongbo Li. Mamba yolo: A simple baseline for object detection with state space model, 2024c. URL https://arxiv.org/abs/2406.05835.

Saining Xie, Ross B. Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5987–5995, 2016. URL https://api.semanticscholar.org/CorpusID: 8485068.

Huichen Yang and William Hsu. Transformer-based approach for document layout understanding. In 2022 IEEE International Conference on Image Processing (ICIP), pp. 4043–4047. IEEE, 2022.

Xue Yang, Jirui Yang, Junchi Yan, Yue Zhang, Tengfei Zhang, Zhi Guo, Xian Sun, and Kun Fu. Scrdet: Towards more robust detection for small, cluttered and rotated objects. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8232–8241, 2019a.

Ze Yang, Shaohui Liu, Han Hu, Liwei Wang, and Stephen Lin. Reppoints: Point set representation for object detection. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 9657–9666, 2019b.

Peng Zhang, Can Li, Liang Qiao, Zhanzhan Cheng, Shiliang Pu, Yi Niu, and Fei Wu. Vsr: a unified framework for document layout analysis combining vision, semantics and relations. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part I 16, pp. 115–130. Springer, 2021.

Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International conference on document analysis and recognition (ICDAR), pp. 1015–1022. IEEE, 2019.

Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. arXiv preprint arXiv:1904.07850, 2019.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. ArXiv, abs/2401.09417, 2024a. URL https://api.semanticscholar.org/CorpusID:267028142.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model, 2024b. URL https://arxiv.org/abs/2401.09417.