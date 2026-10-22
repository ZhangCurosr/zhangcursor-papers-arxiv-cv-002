# LEAF-Mamba: Feature Adaptation and Fusion for Whole Slide Image Analysis

Lubin Gan<sup>a,b</sup>, Jing Zhang<sup>a,b,∗</sup>, Mingjun Ma<sup>c</sup>, Linhao Qu<sup>d</sup>, Dachun Kai<sup>a,1</sup>, Xiaoyan Sun<sup>a,b,∗</sup>

<sup>a</sup>National Engineering Laboratory for Brain-inspired Intelligence Technology and Application, MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology ofChina, Hefei, 230026, China

<sup>b</sup>Anhui Province Key Laboratory of Biomedical Imaging and Intelligent Processing, Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, 230088, China <sup>c</sup>School ofSoftware Technology, Zhejiang University, Ningbo, 315048, China <sup>d</sup>Department ofBiomedical Informatics, Harvard Medical School, Boston, MA, 02115, USA

## Abstract

Whole slide image (WSI) analysis enables computational diagnosis and prognosis, but gigapixel images and limited annotations make slide-level learning challenging. Multiple instance learning addresses this task by aggregating patch representations from frozen pretrained encoders. Recently, Mamba has enabled efficient modeling of long patch sequences. However, task supervision alone does not explicitly constrain information recovery during feature adaptation, while Mamba couples local processing with long-range propagation rather than retaining separate representations for fusion. Motivated by these challenges, we introduce LEAF-Mamba, a framework that jointly addresses latent evidence adaptation and fusion for frozen-feature WSI analysis. A reconstruction-regularized adapter learns compact, task-specific representations while reconstructing the input features. Parallel Mamba and convolutional streams then capture long-range dependencies and sequence-local interactions, with an input-dependent gate modulating their joint representation before slide-level pooling. Experiments across feature extractors demonstrate improved average performance over strong MIL baselines in both diagnostic classification and survival prediction. Component analyses further support the contributions of adaptation and fusion. These findings highlight the value of jointly designing feature transformation and contextual aggregation to translate pretrained representations into effective slide-level predictions.

Keywords: Computational pathology, Whole slide imaging, Multiple instance learning, Feature adaptation, State space models, Gated fusion

## 1. Introduction

Whole slide images (WSIs) capture the cellular morphology and tissue organization used for computational cancer diagnosis and prognosis [1]. Predicting a diagnostic category or patient outcome from a gigapixel slide requires aggregating evidence across heterogeneous tissue. Dense region-level annotation demands extensive pathologist effort, so weakly supervised WSI models learn from slide- or patient-level labels [2]. Multiple instance learning (MIL) represents each slide as a bag of patches and learns a bag-level predictor. Attention-based pooling weights patch contributions, while clustering-constrained learning adds a clustering objective to slide-level training [3, 4]. Dual-stream methods identify critical instances, while feature-distillation methods learn slide representations through pseudo-bag training [5, 6].

In frozen-feature MIL, a pretrained encoder extracts patch embeddings, and a trainable aggregator maps them to slide-level predictions. Pathology foundation models learn reusable patch representations from large tissue-image collections [7, 8]. Transformer-based MIL models contextual interactions among patches [9], while selective state-space models process long sequences with computational cost that scales linearly with sequence length [10]. MambaMIL applies this sequence modeling to WSIs through patch reordering [11]. SemaMIL further studies semantic ordering and retrieval-guided state-space aggregation [12], highlighting the role of sequence construction alongside the choice of sequence operator. We focus on two stages of this pipeline: adapting frozen embeddings to the target task and combining short- and long-range contextual representations.

The first challenge is to adapt frozen features from bag-level supervision. The frozen encoder receives no updates from the downstream diagnostic or survival loss, leaving feature adaptation to the slide-level model. Re-embedding methods such as RRTMIL learn transformations before aggregation [13]. The task loss supervises the transformed patches through the slide prediction, but does not explicitly constrain recovery of their input features. A reconstruction loss supplies this missing constraint by penalizing input-recovery error for each compressed patch representation. Joint optimization therefore gives the adapter two objectives: predict the slide-level outcome and reconstruct the frozen patch embeddings.

The second challenge is to retain distinct short- and long-range representations for fusion. In a standard Mamba block, local convolution feeds the selective state-space operation, placing short-range processing on the same path as longrange propagation [10]. The convolutional output is not retained as a separate representation for fusion with the Mamba output. A parallel convolutional stream preserves the sequence-local representation alongside the long-range output. An input-dependent gate modulates their joint features before pooling. Figure 1 illustrates the parallel streams and their input-dependent fusion.

We propose LEAF-Mamba, a frozen-feature MIL framework that combines reconstruction-regularized latent evidence adaptation (LEA) with Mamba-guided evidence fusion (EF). LEA maps each patch embedding to a compact code through a nonlinear encoder and a linear skip projection. The task loss and an auxiliary reconstruction loss jointly train the adapter, and the reconstruction decoder is removed at inference. EF processes the adapted tokens through parallel Mamba and convolutional streams along a fixed coordinate-derived ordering. The convolutional stream models consecutive tokens in this sequence. The two outputs are concatenated and projected, then modulated by a content-dependent sigmoid gate before slide-level pooling.

We evaluate LEAF-Mamba on diagnostic classification and survival prediction using frozen ResNet-50, PLIP, and CONCH v1.5 features. All reproduced baselines share the extracted features, data splits, and evaluation protocol. LEAF-Mamba achieves the highest mean diagnostic AUC and mean survival C-Index among the compared methods with each encoder. On the BRACS official split with CONCH v1.5 features, removing reconstruction reduces AUC from 76.8 to 75.8 under global-only aggregation. With both branches enabled, input-dependent fusion improves AUC by 0.8 percentage points over a learned static gate. Our main contributions are as follows:

![](images/b46a75e6911d56cd8a391c9f8286e47b48f551598afb75ed8e20580d6662c724.jpg)  
Figure 1: Conceptual visualization of Mamba-guided evidence fusion.

• We introduce a reconstruction-regularized latent adapter for frozen-feature MIL. Joint bag-level prediction and patchwise reconstruction train compact task-specific codes; the decoder is removed at inference.

• We develop an evidence-fusion module that retains parallel Mamba and sequence-local convolutional representations and modulates their projected joint features through an input-dependent gate before pooling.

• We show that LEAF-Mamba achieves the highest mean diagnostic AUC and survival C-Index among the compared methods with each of three frozen encoders. Controlled ablations quantify the contributions of reconstruction and context fusion.

## 2. Related Work

Related work for LEAF-Mamba spans slide-level aggregation, long-sequence modeling, and frozen-feature adaptation.

## 2.1. MILfor whole slide image analysis

In weakly supervised WSI analysis, MIL represents a slide as a bag of patch features and learns from slide-level labels without dense annotation, despite large bag sizes and patch heterogeneity. Attention-based MIL learns a slide summary from instance weights, and CLAM adds clustering constraints to make attention pooling more data-efficient in pathology [3, 4]. DSMIL uses a dual stream to identify critical instances, while DTFD-MIL distills large bags through pseudo-bag training [5, 6]. Recent extensions shift the aggregation assumption in different directions: CAMIL adds neighborhood context, and ACMIL perturbs attention concentration through multi-branch masking [14, 15]. RPMIL models bag predictions probabilistically, whereas Jigsaw learns order restoration from spatial correlations [16, 17]. DomSet-MIL addresses a different aspect of weak supervision: a dominant slide label can conceal coexisting lesions, motivating censored-label modeling and residual routing of lesion evidence [18]. RRTMIL is the closest feature re-embedding comparator because it adapts pathology features for foundation-model transfer [13].

## 2.2. Long-sequence modeling in pathology

Transformer and interventional MIL extend independent pooling through contextual dependency and confounder-aware bag learning [9, 19]. Graph-based models encode relational or spatial context through tissue graphs [20, 21]. Hierarchical models aggregate local-to-slide representations across scales [22, 23]. Attention-based models can be computationally expensive for large bags, whereas graph and hierarchical models depend on relation construction or multi-scale design choices. Structured state space and Mamba models offer a linear-time long-sequence alternative, with Mamba and VMamba establishing selective/statespace operators [10, 24]. S4MIL and MambaMIL adapt this direction to WSI sequences [25, 11]. SemaMIL combines semantic reordering with retrieval-guided state-space modeling [12]; its sequence-construction strategy complements the study of parallel contextual streams. MambaMIL is the closest sequence-side comparator to LEAF-Mamba [11].

## 2.3. Pathologyfoundation models and representation mismatch

Pathology foundation models improve frozen patch representations: CTrans-Path uses pathology-specific contrastive pretraining, and PLIP adds visionlanguage supervision [26, 27]. Broader general-purpose and clinical-grade encoders expand transfer across pathology tasks [7, 8]. Recent multimodal wholeslide encoders extend this trend with image-text supervision [28, 29]. FG-PAN refines visual patches and aligns them with fine-grained descriptions for zero-shot brain tumor subtyping [30]. MeDKCoOp uses graph-structured visual and textual knowledge for biomedical prompt adaptation [31]. Both study how reusable vision-language representations can be matched to domain-specific evidence. Frozen representations can still benefit from downstream task adaptation, as investigated by RRTMIL [13]. The feature mismatch addressed by this adaptation is distinct from generalization to a new acquisition domain. The comparison of interest is how the adaptation objective and aggregation operators contribute under the same frozen-feature protocol.

## 2.4. Adapters and hybrid sequence operators

Bottleneck and residual adaptation have established precedents. Houlsby et al. insert trainable adapters within a frozen language model [32], whereas CLIP-Adapter applies bottleneck transformations and residual blending to pretrained features [33]. These architectures motivate simple adapter controls; their supervision and insertion points differ from adaptation of offline WSI features. RRTMIL is the closer protocol-level reference because it re-embeds extracted patch features before MIL aggregation [13]. For aggregation, Conformer combines convolutional receptive fields with long-range attention in speech recognition [34]. This supports a general design precedent for combining local and global operators, while the value of a chosen serialization in WSI analysis remains an empirical question.

![](images/05fea4b19235baf0c3e941547fc148b5e7fdf794225ad55a999ba0af537e7ee8.jpg)  
Figure 2: Overview of LEAF-Mamba. Frozen features undergo reconstruction-regularized latent evidence adaptation and Mamba-guided fusion of long-range sequence context, short-range token interactions, and content-dependent gating for slide-level prediction.

Spatial and semantic constraints also arise in other medical imaging settings. SSCM combines spatial alignment and semantic consistency for multi-contrast MRI super-resolution [35]. Its cross-contrast reconstruction setting differs from recovering frozen patch features, but provides a related example of coordinating local detail with semantic representations. In vision-language modeling, geometryaware dense-to-expert reconstruction studies the conversion of pretrained models into specialized expert modules [36]. Here, adaptation instead operates on offline patch embeddings before contextual aggregation.

## 2.5. Structural information and distributed aggregation

Preserving discriminative evidence during representation transformation is also studied in graph learning. GraphRefiner calibrates generic-token dominance in graph-text alignment [37]; G<sup>2</sup>rammar encodes structural roles and semantic relationships [38]. The H4G preprint studies how hyperbolic embedding geometry can retain fine-grained graph structure [39]. These formulations provide related perspectives on distinguishing useful local information from a broad contextual representation.

Multi-agent research examines a complementary question: how should distributed information sources communicate? HyperAgent uses hyperedges to represent collaboration groups [40], while EvoHyper evolves a shared hypergraph of agents and memory [41]. D<sup>3</sup>MAS coordinates task decomposition, reasoning, and distributed memory [42], and graph-decoupling methods study knowledge sharing among agents [43]. At a conceptual level, these works connect the organization of individual information sources with their collective use. In WSI analysis, the corresponding design problem concerns patch representations and their aggregation under slide-level supervision.

## 3. Method

LEAF-Mamba transforms frozen patch embeddings, models them through long-range and sequence-local paths, and combines the resulting features before gated attention pooling. The adapter and aggregator are trained jointly with slidelevel supervision, while the patch encoder remains frozen. Figure 2 summarizes the complete pipeline.

## 3.1. Problem Setup

Following the standard MIL paradigm, the i-th WSI is represented as a bag

$$
\mathsf { B } _ { i } = \{ \boldsymbol { x } _ { i j } \} _ { j = 1 } ^ { N _ { i } } ,\tag{1}
$$

where $N _ { i }$ is the number of patches and $x _ { i j }$ denotes a raw image patch. A frozen feature extractor maps each patch to $h _ { i j } \in \mathbb { R } ^ { D }$ , producing a bag feature matrix $H _ { i } \in \mathbb { R } ^ { N _ { i } \times D }$ . We denote the slide-level target by $y _ { i } { \mathrm { : } }$ for diagnostic classification, $\mathcal { V } _ { i } = y _ { i }$ with $y _ { i } \in \{ 0 , 1 , \dotsc , C - 1 \}$ ; for survival prediction, $\mathcal { V } _ { i } = \left( t _ { i } , \delta _ { i } \right)$ contains the observed time and event indicator. The goal is to learn a mapping $F ( H _ { i } )$ that produces class logits or a survival risk score from frozen bag features.

## 3.2. Latent Evidence Adaptation

LEA formulates task-specific feature adaptation with an auxiliary reconstruction objective before aggregation. The input is the frozen feature matrix $H _ { i } ,$ , whose representation was learned before the downstream MIL task. Slide-level supervision trains the adapter jointly with the aggregator; reconstructing $H _ { i }$ adds an input-recovery objective to the task loss. LEA projects each token into a lowerdimensional code from which a decoder reconstructs the original feature. The module contains a nonlinear encoder $E ( \cdot )$ , a decoder $G ( \cdot )$ , and a linear skip projection $P _ { \mathrm { s k i p } } ( \cdot )$ . We define the latent representation as

$$
Z _ { i } = E ( H _ { i } ) + P _ { \mathrm { s k i p } } ( H _ { i } ) \in \mathbb { R } ^ { N _ { i } \times d } .\tag{2}
$$

The encoder and decoder are lightweight projection heads applied token-wise, with

$$
E ( h ) = W _ { 2 } ^ { E } \phi ( W _ { 1 } ^ { E } h ) , \quad G ( z ) = W _ { 2 } ^ { G } \phi ( W _ { 1 } ^ { G } z ) ,\tag{3}
$$

where $\phi ( \cdot )$ is GELU and the latent width satisfies d $< D$ . The nonlinear encoder learns the task-adaptive transformation, while the linear skip projection supplies a second route from the frozen input. The decoder reconstructs the original feature matrix:

$$
\begin{array} { r } { \hat { H } _ { i } = G ( Z _ { i } ) \in \mathbb { R } ^ { N _ { i } \times D } . } \end{array}\tag{4}
$$

The reconstruction loss is

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { N _ { i } } \sum _ { j = 1 } ^ { N _ { i } } \left\| h _ { i j } - \hat { h } _ { i j } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

The overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } , } \end{array}\tag{6}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ is cross-entropy for diagnostic classification and the negative Cox partial log-likelihood for survival prediction. The task loss trains predictive features, and the reconstruction term penalizes input-recovery error. Figure 3 compares the distributions of frozen CONCH features and task-trained latent features on BRACS, alongside predictive performance for the projection, reconstruction, and skip-connection configurations.

## 3.3. Global-Local Evidence Modeling

EF applies two sequence operators before bag-level pooling. One stream models long-range dependencies, while the other captures interactions within a finite convolutional window. Combining these receptive fields follows the hybridoperator design principle used in Conformer [34]. For coordinate-bearing bags, raster serialization sorts patch coordinates by y first and then x, preserving input order for exact ties. The ordered sequence is retained across blocks. Let $Z _ { i } ^ { ( 0 ) } = Z _ { i }$ denote the input to the first evidence-fusion block. At block l, we normalize the current token sequence:

![](images/09cec22b7dec8c79b31ec03494a7113c339f8ecf7dd50cf596c54e5b9fe9e90c.jpg)  
Figure 3: Effect of LEA on BRACS features.

$$
\tilde { Z } _ { i } ^ { ( l - 1 ) } = \mathrm { L N } ( Z _ { i } ^ { ( l - 1 ) } ) .\tag{7}
$$

The global stream uses a selective state space operation $\Phi _ { \mathrm { S S M } } ( \cdot )$ to model longrange dependencies:

$$
Z _ { \mathrm { g l o b a l } } ^ { ( l ) } = \Phi _ { \mathrm { S S M } } ( \tilde { Z } _ { i } ^ { ( l - 1 ) } ) .\tag{8}
$$

Because Mamba-style models are sequence-order sensitive, we use the same coordinate-derived ordering for LEAF-Mamba and MambaMIL in all comparisons. Alongside the global stream, a separate local stream models short-range token interactions using depthwise separable convolutions:

$$
Z _ { \mathrm { l o c a l } } ^ { ( l ) } = \phi \left( \mathrm { P W C o n v } \left( \mathrm { D W C o n v } _ { k } ( \tilde { Z } _ { i } ^ { ( l - 1 ) } ) \right) \right) ,\tag{9}
$$

where DWConv $r _ { k }$ is a depthwise one-dimensional convolution with kernel size $k ,$ PWConv is a pointwise convolution, and $\phi ( \cdot )$ is the GELU activation. The default kernel spans seven consecutive tokens in the raster-ordered sequence, defining a sequence-local receptivefield. Convolutional windows follow this sequence across row transitions, tissue holes, and disconnected regions. The kernel size determines the number of neighboring tokens processed at each position.

## 3.4. Content-Dependent Evidence Fusion

Gated fusion combines the two evidence streams through input-dependent modulation of their joint representation. We first concatenate the two streams into $U _ { i } ^ { ( l ) } = [ Z _ { \mathrm { l o c a l } } ^ { ( l ) } ; Z _ { \mathrm { g l o b a l } } ^ { ( l ) } ] \in \bar { \mathbb { R } } ^ { N _ { i } \times 2 d }$ and apply a gated projection:

$$
Z _ { \mathrm { f u s e } } ^ { ( l ) } = ( U _ { i } ^ { ( l ) } W _ { \mathrm { p r o j } } ) \odot \sigma ( U _ { i } ^ { ( l ) } W _ { \mathrm { g a t e } } ) ,\tag{10}
$$

where $W _ { \mathrm { p r o j } } , W _ { \mathrm { g a t e } } \in \mathbb { R } ^ { 2 d \times d }$ are learnable parameters, $\sigma ( \cdot )$ is the sigmoid function, and ⊙ denotes element-wise multiplication. The sigmoid gate modulates the projected joint representation, allowing the fused feature to respond differently across slides and tokens. A residual feed-forward update then refines the representation:

$$
Z _ { i } ^ { ( l ) } = Z _ { i } ^ { ( l - 1 ) } + M \left( \mathrm { L N } \left( Z _ { i } ^ { ( l - 1 ) } + Z _ { \mathrm { f u s e } } ^ { ( l ) } \right) \right) ,\tag{11}
$$

where $Z _ { i } ^ { ( l - 1 ) }$ and $Z _ { i } ^ { ( l ) }$ denote the block input and output, $M ( \cdot )$ is an MLP, and $\operatorname { L N } ( \cdot )$ denotes layer normalization. Gated attention pools the final normalized token sequence into a bag representation, which is passed to the slide-level task head. Figure 1 summarizes the long-range stream, sequence-local stream, and input-dependent feature modulation.

## 4. Experiments

We organize the experiments around three questions. First, does the representation-and-evidence design remain useful across feature extractors? Second, does the same principle carry from diagnostic classification to noisier survival prediction? Third, do controlled variants support the roles assigned to LEA, local modeling, and gated fusion? We first define the datasets and unified protocol, then answer these questions through benchmark comparisons and targeted ablations.

## 4.1. Datasets and Protocol

We evaluate LEAF-Mamba on both diagnostic classification and survival prediction under a unified frozen-feature protocol. For diagnostic classification, we use EBRAINS for 30-class brain tumor subtyping [44], BRACS for 7-class breast lesion classification [45], and Camelyon16 for breast cancer metastasis detection [46]. BRACS is evaluated with both 5-fold cross-validation and its official split. For survival prediction, we use five cohorts from The Cancer Genome Atlas (TCGA): BLCA, BRCA, COADREAD, STAD, and HNSC [47]. Prognostic performance is measured by Concordance Index (C-Index) and reported as C-Index ×100.

## 4.2. Implementation Details

All reproduced baselines reuse the same pre-extracted features, split protocol, early-stopping rule, and metric computation as LEAF-Mamba. For Mamba-based models, the same fixed coordinate-derived patch serialization is used throughout so that ordering does not confound the comparison. This protocol isolates the slide-level aggregator stage under frozen features.

LEAF-Mamba uses two evidence-fusion blocks unless otherwise stated. In the LEA module, the encoder maps the input dimension D to a latent width d through a hidden width of 2d, uses layer normalization, GELU, and dropout 0.25, and adds a linear skip projection from the frozen feature. The decoder mirrors this projection to reconstruct the original frozen feature and contributes the mean-squared reconstruction loss during training. Each evidence-fusion block applies a local depthwise one-dimensional convolution with kernel size 7, followed by a pointwise convolution, and a global Mamba path with state dimension 32, convolution width 4, and expansion factor 2. The local and global outputs are concatenated, projected, and modulated by a sigmoid fusion gate before a residual feed-forward update. After the final block, gated attention pools the refined token features into a slide representation, which is passed to a linear task head.

![](images/c1552f96df3add7fcc067c9e141c68181fd1b1b417eef14b5b270ef14d69a387.jpg)  
Figure 4: Diagnostic classification and TCGA survival prediction results across frozen feature extractors. The vertical axis shows the score difference from the best baseline using the same feature extractor, on a nonlinear scale.

## 4.3. Comparison with Recent WSI Baselines

To present the extensive comparisons within the available space, Figure 4 uses a compact multi-panel layout showing score differences relative to the strongest baseline for each feature extractor. Tables 1–4 report the complete numerical results corresponding to Figure 4.

Figure 4 includes recent WSI baselines. The MHIM variants combine masked hard-instance mining with ABMIL or TransMIL backbones [48]. CAMIL and RPMIL provide context-aware and probabilistic WSI comparators, respectively [14, 16]. ACMIL and the instance-jigsaw model provide attention-challenging and order-restoration comparators [15, 17].

In Figure 4, LEAF-Mamba achieves strongest average diagnostic performance for each feature extractor under shared frozen-feature protocol.

Specifically, average AUC reaches 87.7, 88.5, and 89.4 with ResNet-50, PLIP, and CONCH v1.5 features, respectively, exceeding the strongest baseline, Jigsaw, by 1.1, 1.9, and 1.0 percentage points. This consistent ranking shows benefits across all three frozen feature extractors. The BRACS official split provides a representative benchmark-level example: with PLIP features, LEAF-Mamba achieves AUC 77.8, exceeding Jigsaw’s 74.6 by 3.2 percentage points. Together, the average results and this task-specific comparison support the effectiveness of combining feature adaptation with sequence aggregation for diagnostic classification.

Beyond diagnostic classification, Figure 4 shows that LEAF-Mamba also achieves the best average C-Index with each evaluated feature extractor.

Across the five TCGA cohorts, the average C-Index ×100 is 67.0 with ResNet-50 features and 67.3 with PLIP and CONCH v1.5 features. These scores exceed Jigsaw, the strongest survival baseline, by 0.1, 0.2, and 0.3 points, respectively. The leading average performance across all three feature extractors complements the diagnostic results, showing that the combined adaptation and aggregation design is effective for both slide-level classification and survival prediction. This design therefore supports both prediction tasks evaluated here.

Table 1: Diagnostic classification results with ResNet-50 features. All scores are percentages; higher is better.  
(a) BRACS and BRACS official
<table><tr><td rowspan="2">Method</td><td colspan="3">BRACS</td><td colspan="3">BRACS official</td></tr><tr><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>69.9</td><td>41.4</td><td>29.0</td><td>65.2</td><td>37.0</td><td>22.8</td></tr><tr><td>MaxMIL</td><td>67.3</td><td>36.8</td><td>27.1</td><td>63.5</td><td>26.0</td><td>26.9</td></tr><tr><td>CLAM</td><td>79.3</td><td>44.4</td><td>37.5</td><td>73.2</td><td>37.0</td><td>32.3</td></tr><tr><td>ABMIL</td><td>75.9</td><td>43.2</td><td>34.7</td><td>71.5</td><td>33.0</td><td>26.1</td></tr><tr><td>MHIM-ABMIL</td><td>76.9</td><td>43.5</td><td>35.9</td><td>72.0</td><td>34.0</td><td>25.9</td></tr><tr><td>DSMIL</td><td>67.0</td><td>35.8</td><td>24.9</td><td>70.5</td><td>35.0</td><td>27.9</td></tr><tr><td>DTFD-MIL</td><td>76.9</td><td>41.3</td><td>34.3</td><td>70.9</td><td>30.0</td><td>24.3</td></tr><tr><td>TransMIL</td><td>73.0</td><td>39.5</td><td>25.8</td><td>66.4</td><td>32.0</td><td>18.9</td></tr><tr><td>MHIM-TransMIL</td><td>74.3</td><td>42.6</td><td>26.8</td><td>67.3</td><td>33.0</td><td>19.3</td></tr><tr><td>S4MIL</td><td>76.4</td><td>44.0</td><td>33.6</td><td>69.1</td><td>31.0</td><td>18.0</td></tr><tr><td>RRTMIL</td><td>78.2</td><td>44.1</td><td>34.0</td><td>68.4</td><td>31.0</td><td>19.6</td></tr><tr><td>MambaMIL</td><td>79.4</td><td>46.6</td><td>35.3</td><td>74.1</td><td>31.0</td><td>27.9</td></tr><tr><td>CAMIL</td><td>77.8</td><td>44.6</td><td>33.7</td><td>70.7</td><td>32.0</td><td>24.6</td></tr><tr><td>RPMIL</td><td>80.0</td><td>47.2</td><td>36.5</td><td>74.8</td><td>34.0</td><td>27.9</td></tr><tr><td>ACMIL</td><td>79.0</td><td>46.0</td><td>35.8</td><td>73.9</td><td>34.0</td><td>27.6</td></tr><tr><td>Jigsaw</td><td>80.4</td><td>47.6</td><td>36.9</td><td>75.0</td><td>35.0</td><td>28.2</td></tr><tr><td>LEAF-Mamba</td><td>81.6</td><td>49.2</td><td>37.3</td><td>76.9</td><td>35.0</td><td>28.9</td></tr><tr><td rowspan="2">Method</td><td colspan="3">Camelyon16</td><td colspan="3">EBRAINS</td></tr><tr><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>70.9</td><td>65.2</td><td>55.4</td><td>91.0</td><td>37.3</td><td>34.6</td></tr><tr><td>MaxMIL</td><td>94.2</td><td>88.5</td><td>85.1</td><td>87.9</td><td>32.9</td><td>30.4</td></tr><tr><td>CLAM</td><td>93.5</td><td>89.5</td><td>86.6</td><td>91.8</td><td>37.3</td><td>34.8</td></tr><tr><td>ABMIL</td><td>94.1</td><td>87.7</td><td>84.4</td><td>91.7</td><td>39.7</td><td>37.6</td></tr><tr><td>MHIM-ABMIL</td><td>94.3</td><td>89.7</td><td>82.9</td><td>91.8</td><td>41.3</td><td>38.4</td></tr><tr><td>DSMIL</td><td>88.9</td><td>83.2</td><td>78.9</td><td>90.3</td><td>40.6</td><td>39.9</td></tr><tr><td>DTFD-MIL</td><td>93.8</td><td>88.7</td><td>85.9</td><td>89.5</td><td>35.2</td><td>33.5</td></tr><tr><td>TransMIL</td><td>92.2</td><td>81.7</td><td>79.4</td><td>92.7</td><td>38.2</td><td>35.8</td></tr><tr><td>MHIM-TransMIL</td><td>93.7</td><td>82.5</td><td>80.3</td><td>91.4</td><td>38.6</td><td>34.0</td></tr><tr><td>S4MIL</td><td>96.5</td><td>90.2</td><td>88.6</td><td>93.7</td><td>44.4</td><td>41.7</td></tr><tr><td>RRTMIL</td><td>94.8</td><td>89.3</td><td>87.3</td><td>92.8</td><td>49.7</td><td>44.3</td></tr><tr><td>MambaMIL</td><td>95.5</td><td>89.2</td><td>86.4</td><td>95.9</td><td>47.9</td><td>46.4</td></tr><tr><td>CAMIL</td><td>94.6</td><td>88.8</td><td>85.2</td><td>92.9</td><td>40.4</td><td>30.5</td></tr><tr><td>RPMIL</td><td>96.0</td><td>90.5</td><td>87.8</td><td>94.4</td><td>43.9</td><td>40.6</td></tr><tr><td>ACMIL</td><td>95.6</td><td>89.7</td><td>86.9</td><td>93.9</td><td>41.9</td><td>39.7</td></tr><tr><td>Jigsaw</td><td>96.3</td><td>90.7</td><td>88.0</td><td>94.7</td><td>44.3</td><td>40.9</td></tr><tr><td>LEAF-Mamba</td><td>96.9</td><td>91.3</td><td>88.4</td><td>95.3</td><td>48.9</td><td>45.2</td></tr></table>

Best scores in each column are in bold.

(c) Average
<table><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>74.2</td><td>45.2</td><td>35.4</td></tr><tr><td>MaxMIL</td><td>78.2</td><td>46.1</td><td>42.4</td></tr><tr><td>CLAM</td><td>84.4</td><td>52.1</td><td>47.8</td></tr><tr><td>ABMIL</td><td>83.3</td><td>50.9</td><td>45.7</td></tr><tr><td>MHIM-ABMIL</td><td>83.7</td><td>52.1</td><td>45.8</td></tr><tr><td>DSMIL</td><td>79.2</td><td>48.7</td><td>42.9</td></tr><tr><td>DTFD-MIL</td><td>82.8</td><td>48.8</td><td>44.5</td></tr><tr><td>TransMIL</td><td>81.1</td><td>47.8</td><td>40.0</td></tr><tr><td>MHIM-TransMIL</td><td>81.7</td><td>49.2</td><td>40.1</td></tr><tr><td>S4MIL</td><td>83.9</td><td>52.4</td><td>45.5</td></tr><tr><td>RRTMIL</td><td>83.6</td><td>53.5</td><td>46.3</td></tr><tr><td>MambaMIL</td><td>86.2</td><td>53.7</td><td>49.0</td></tr><tr><td>CAMIL</td><td>84.0</td><td>51.5</td><td>43.5</td></tr><tr><td>RPMIL</td><td>86.3</td><td>53.9</td><td>48.2</td></tr><tr><td>ACMIL</td><td>85.6</td><td>52.9</td><td>47.5</td></tr><tr><td>Jigsaw</td><td>86.6</td><td>54.4</td><td>48.5</td></tr><tr><td>LEAF-Mamba</td><td>87.7</td><td>56.1</td><td>50.0</td></tr></table>

Best scores in each column are in bold. Average summarizes the four diagnostic settings.

Table 2: Diagnostic classification results with PLIP features. All scores are percentages; higher is better.  
(a) BRACS and BRACS official
<table><tr><td></td><td colspan="3">BRACS</td><td colspan="3">BRACS official</td></tr><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>73.2</td><td>43.3</td><td>33.7</td><td>61.2</td><td>32.0</td><td>23.2</td></tr><tr><td>MaxMIL</td><td>72.1</td><td>38.7</td><td>26.8</td><td>59.0</td><td>20.0</td><td>17.8</td></tr><tr><td>CLAM</td><td>75.9</td><td>40.7</td><td>31.0</td><td>65.6</td><td>31.0</td><td>22.9</td></tr><tr><td>ABMIL</td><td>76.9</td><td>42.5</td><td>34.4</td><td>69.1</td><td>30.0</td><td>26.8</td></tr><tr><td>MHIM-ABMIL</td><td>77.6</td><td>43.8</td><td>35.8</td><td>69.8</td><td>30.0</td><td>26.9</td></tr><tr><td>DSMIL</td><td>71.4</td><td>33.1</td><td>25.1</td><td>61.2</td><td>17.0</td><td>14.4</td></tr><tr><td>DTFD-MIL</td><td>77.1</td><td>42.5</td><td>34.7</td><td>70.1</td><td>26.0</td><td>23.8</td></tr><tr><td>TransMIL</td><td>71.2</td><td>35.3</td><td>24.0</td><td>67.1</td><td>29.0</td><td>25.1</td></tr><tr><td>MHIM-TransMIL</td><td>72.5</td><td>37.5</td><td>26.5</td><td>68.2</td><td>29.0</td><td>25.8</td></tr><tr><td>S4MIL</td><td>75.2</td><td>43.3</td><td>33.1</td><td>64.3</td><td>26.0</td><td>19.7</td></tr><tr><td>RRTMIL</td><td>78.5</td><td>45.5</td><td>35.5</td><td>71.5</td><td>28.0</td><td>24.2</td></tr><tr><td>MambaMIL</td><td>70.5</td><td>33.0</td><td>23.8</td><td>72.8</td><td>29.0</td><td>27.3</td></tr><tr><td>CAMIL</td><td>77.8</td><td>43.8</td><td>34.1</td><td>70.2</td><td>30.0</td><td>25.0</td></tr><tr><td>RPMIL</td><td>80.4</td><td>47.2</td><td>36.7</td><td>74.2</td><td>35.0</td><td>30.0</td></tr><tr><td>ACMIL</td><td>79.8</td><td>46.5</td><td>36.3</td><td>73.6</td><td>34.0</td><td>29.7</td></tr><tr><td>Jigsaw</td><td>80.5</td><td>47.8</td><td>37.0</td><td>74.6</td><td>35.0</td><td>30.2</td></tr><tr><td>LEAF-Mamba</td><td>82.2</td><td>49.6</td><td>37.9</td><td>77.8</td><td>36.0</td><td>29.8</td></tr></table>

Best scores in each column are in bold.

(b) Camelyon16 and EBRAINS
<table><tr><td rowspan="2">Method</td><td colspan="3">Camelyon16</td><td colspan="3">EBRAINS</td></tr><tr><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>78.7</td><td>71.9</td><td>65.4</td><td>94.9</td><td>51.5</td><td>50.3</td></tr><tr><td>MaxMIL</td><td>77.1</td><td>71.0</td><td>67.3</td><td>92.3</td><td>43.9</td><td>42.1</td></tr><tr><td>CLAM</td><td>95.6</td><td>90.5</td><td>88.3</td><td>95.2</td><td>53.9</td><td>53.2</td></tr><tr><td>ABMIL</td><td>96.3</td><td>91.7</td><td>89.6</td><td>95.4</td><td>54.1</td><td>53.4</td></tr><tr><td>MHIM-ABMIL</td><td>96.6</td><td>90.5</td><td>89.7</td><td>95.8</td><td>54.3</td><td>53.2</td></tr><tr><td>DSMIL</td><td>80.5</td><td>73.9</td><td>67.3</td><td>93.3</td><td>47.7</td><td>46.3</td></tr><tr><td>DTFD-MIL</td><td>95.3</td><td>90.0</td><td>87.4</td><td>94.0</td><td>51.1</td><td>50.1</td></tr><tr><td>TransMIL</td><td>94.5</td><td>88.5</td><td>86.4</td><td>92.0</td><td>49.9</td><td>50.6</td></tr><tr><td>MHIM-TransMIL</td><td>95.1</td><td>89.5</td><td>86.9</td><td>92.8</td><td>50.2</td><td>50.4</td></tr><tr><td>S4MIL</td><td>96.6</td><td>92.2</td><td>90.3</td><td>93.2</td><td>50.9</td><td>49.5</td></tr><tr><td>RRTMIL</td><td>96.8</td><td>92.1</td><td>90.1</td><td>94.1</td><td>50.7</td><td>50.3</td></tr><tr><td>MambaMIL</td><td>95.2</td><td>88.2</td><td>86.2</td><td>94.5</td><td>51.3</td><td>50.5</td></tr><tr><td>CAMIL</td><td>96.0</td><td>91.4</td><td>89.1</td><td>92.8</td><td>48.6</td><td>45.0</td></tr><tr><td>RPMIL</td><td>97.0</td><td>92.8</td><td>90.9</td><td>94.4</td><td>50.0</td><td>50.4</td></tr><tr><td>ACMIL</td><td>96.7</td><td>92.6</td><td>90.7</td><td>93.9</td><td>49.1</td><td>50.5</td></tr><tr><td>Jigsaw</td><td>97.1</td><td>92.9</td><td>91.0</td><td>94.2</td><td>51.2</td><td>51.0</td></tr><tr><td>LEAF-Mamba</td><td>97.3</td><td>92.8</td><td>91.5</td><td>96.8</td><td>55.0</td><td>53.4</td></tr></table>

Best scores in each column are in bold.

Table 2: Diagnostic classification results with PLIP features. All scores are percentages; higher is better (continued).  
(c) Average
<table><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>77.0</td><td>49.7</td><td>43.1</td></tr><tr><td>MaxMIL</td><td>75.1</td><td>43.4</td><td>38.5</td></tr><tr><td>CLAM</td><td>83.1</td><td>54.0</td><td>48.8</td></tr><tr><td>ABMIL</td><td>84.4</td><td>54.6</td><td>51.0</td></tr><tr><td>MHIM-ABMIL</td><td>85.0</td><td>54.7</td><td>51.4</td></tr><tr><td>DSMIL</td><td>76.6</td><td>42.9</td><td>38.3</td></tr><tr><td>DTFD-MIL</td><td>84.1</td><td>52.4</td><td>49.0</td></tr><tr><td>TransMIL</td><td>81.2</td><td>50.7</td><td>46.5</td></tr><tr><td>MHIM-TransMIL</td><td>82.2</td><td>51.6</td><td>47.4</td></tr><tr><td>S4MIL</td><td>82.3</td><td>53.1</td><td>48.1</td></tr><tr><td>RRTMIL</td><td>85.2</td><td>54.1</td><td>50.0</td></tr><tr><td>MambaMIL</td><td>83.3</td><td>50.4</td><td>47.0</td></tr><tr><td>CAMIL</td><td>84.2</td><td>53.5</td><td>48.3</td></tr><tr><td>RPMIL</td><td>86.5</td><td>56.3</td><td>52.0</td></tr><tr><td>ACMIL</td><td>86.0</td><td>55.6</td><td>51.8</td></tr><tr><td>Jigsaw</td><td>86.6</td><td>56.7</td><td>52.3</td></tr><tr><td>LEAF-Mamba</td><td>88.5</td><td>58.4</td><td>53.1</td></tr></table>

Best scores in each column are in bold. Average summarizes the four diagnostic settings.

Table 3: Diagnostic classification results with CONCH v1.5 features. All scores are percentages; higher is better.  
(a) BRACS and BRACS official
<table><tr><td></td><td colspan="3">BRACS</td><td colspan="3">BRACS official</td></tr><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>78.6</td><td>48.2</td><td>38.4</td><td>62.8</td><td>33.0</td><td>19.1</td></tr><tr><td>MaxMIL</td><td>79.5</td><td>47.8</td><td>36.4</td><td>69.2</td><td>33.0</td><td>26.3</td></tr><tr><td>CLAM</td><td>79.8</td><td>49.6</td><td>36.3</td><td>74.0</td><td>37.0</td><td>25.3</td></tr><tr><td>ABMIL</td><td>77.8</td><td>50.6</td><td>40.4</td><td>75.6</td><td>37.0</td><td>27.4</td></tr><tr><td>MHIM-ABMIL</td><td>78.4</td><td>51.2</td><td>41.5</td><td>76.1</td><td>37.0</td><td>28.7</td></tr><tr><td>DSMIL</td><td>78.7</td><td>46.7</td><td>35.6</td><td>74.9</td><td>37.0</td><td>29.9</td></tr><tr><td>DTFD-MIL</td><td>80.9</td><td>49.2</td><td>39.2</td><td>76.3</td><td>35.0</td><td>27.4</td></tr><tr><td>TransMIL</td><td>73.1</td><td>42.1</td><td>24.8</td><td>71.4</td><td>37.0</td><td>27.6</td></tr><tr><td>MHIM-TransMIL</td><td>74.5</td><td>44.5</td><td>28.5</td><td>72.5</td><td>37.0</td><td>28.5</td></tr><tr><td>S4MIL</td><td>77.4</td><td>46.5</td><td>36.2</td><td>72.5</td><td>33.0</td><td>22.5</td></tr><tr><td>RRTMIL</td><td>79.5</td><td>48.5</td><td>37.9</td><td>76.8</td><td>34.0</td><td>22.5</td></tr><tr><td>MambaMIL</td><td>78.0</td><td>47.0</td><td>32.6</td><td>76.6</td><td>35.0</td><td>27.7</td></tr><tr><td>CAMIL</td><td>79.7</td><td>49.8</td><td>37.5</td><td>75.2</td><td>37.0</td><td>27.0</td></tr><tr><td>RPMIL</td><td>81.7</td><td>51.6</td><td>42.5</td><td>77.5</td><td>38.0</td><td>31.4</td></tr><tr><td>ACMIL</td><td>81.2</td><td>51.0</td><td>41.6</td><td>77.2</td><td>38.0</td><td>30.7</td></tr><tr><td>Jigsaw</td><td>82.0</td><td>52.0</td><td>43.0</td><td>77.6</td><td>38.0</td><td>31.8</td></tr><tr><td>LEAF-Mamba</td><td>83.2</td><td>53.6</td><td>44.9</td><td>78.6</td><td>40.0</td><td>32.7</td></tr></table>

Best scores in each column are in bold.

(b) Camelyon16 and EBRAINS
<table><tr><td></td><td colspan="3">Camelyon16</td><td colspan="3">EBRAINS</td></tr><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>91.5</td><td>93.9</td><td>88.3</td><td>97.2</td><td>63.6</td><td>62.6</td></tr><tr><td>MaxMIL</td><td>87.7</td><td>90.2</td><td>89.7</td><td>97.4</td><td>67.6</td><td>67.0</td></tr><tr><td>CLAM</td><td>94.8</td><td>92.7</td><td>92.3</td><td>97.8</td><td>66.9</td><td>65.9</td></tr><tr><td>ABMIL</td><td>96.5</td><td>93.2</td><td>91.5</td><td>97.7</td><td>67.3</td><td>66.4</td></tr><tr><td>MHIM-ABMIL</td><td>96.9</td><td>93.4</td><td>91.5</td><td>97.9</td><td>67.9</td><td>67.1</td></tr><tr><td>DSMIL</td><td>92.7</td><td>88.5</td><td>84.8</td><td>97.8</td><td>66.7</td><td>65.9</td></tr><tr><td>DTFD-MIL</td><td>97.5</td><td>94.2</td><td>92.6</td><td>97.6</td><td>67.9</td><td>67.0</td></tr><tr><td>TransMIL</td><td>96.5</td><td>90.5</td><td>88.8</td><td>97.3</td><td>60.4</td><td>58.4</td></tr><tr><td>MHIM-TransMIL</td><td>96.9</td><td>91.7</td><td>89.7</td><td>97.6</td><td>61.8</td><td>59.7</td></tr><tr><td>S4MIL</td><td>95.8</td><td>94.0</td><td>92.5</td><td>97.5</td><td>63.2</td><td>62.5</td></tr><tr><td>RRTMIL</td><td>96.2</td><td>92.5</td><td>92.1</td><td>97.2</td><td>63.8</td><td>62.9</td></tr><tr><td>MambaMIL</td><td>97.0</td><td>92.5</td><td>90.7</td><td>97.2</td><td>64.3</td><td>62.6</td></tr><tr><td>CAMIL</td><td>96.2</td><td>92.9</td><td>91.8</td><td>96.1</td><td>62.7</td><td>64.5</td></tr><tr><td>RPMIL</td><td>97.8</td><td>94.0</td><td>92.4</td><td>95.8</td><td>64.2</td><td>65.7</td></tr><tr><td>ACMIL</td><td>97.5</td><td>93.8</td><td>92.2</td><td>96.1</td><td>64.4</td><td>64.3</td></tr><tr><td>Jigsaw</td><td>97.8</td><td>94.0</td><td>92.5</td><td>96.2</td><td>64.8</td><td>66.3</td></tr><tr><td>LEAF-Mamba</td><td>97.9</td><td>94.2</td><td>92.5</td><td>97.9</td><td>68.2</td><td>68.5</td></tr></table>

Best scores in each column are in bold.

Table 3: Diagnostic classification results with CONCH v1.5 features. All scores are percentages; higher is better (continued).  
(c) Average
<table><tr><td>Method</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>MeanMIL</td><td>82.5</td><td>59.7</td><td>52.1</td></tr><tr><td>MaxMIL</td><td>83.4</td><td>59.7</td><td>54.9</td></tr><tr><td>CLAM</td><td>86.6</td><td>61.6</td><td>55.0</td></tr><tr><td>ABMIL</td><td>86.9</td><td>62.0</td><td>56.4</td></tr><tr><td>MHIM-ABMIL</td><td>87.3</td><td>62.4</td><td>57.2</td></tr><tr><td>DSMIL</td><td>86.0</td><td>59.7</td><td>54.1</td></tr><tr><td>DTFD-MIL</td><td>88.1</td><td>61.6</td><td>56.6</td></tr><tr><td>TransMIL</td><td>84.5</td><td>57.5</td><td>49.9</td></tr><tr><td>MHIM-TransMIL</td><td>85.4</td><td>58.8</td><td>51.6</td></tr><tr><td>S4MIL</td><td>85.8</td><td>59.2</td><td>53.4</td></tr><tr><td>RRTMIL</td><td>87.4</td><td>59.7</td><td>53.9</td></tr><tr><td>MambaMIL</td><td>87.2</td><td>59.7</td><td>53.4</td></tr><tr><td>CAMIL</td><td>86.8</td><td>60.6</td><td>55.2</td></tr><tr><td>RPMIL</td><td>88.2</td><td>62.0</td><td>58.0</td></tr><tr><td>ACMIL</td><td>88.0</td><td>61.8</td><td>57.2</td></tr><tr><td>Jigsaw</td><td>88.4</td><td>62.2</td><td>58.4</td></tr><tr><td>LEAF-Mamba</td><td>89.4</td><td>64.0</td><td>59.6</td></tr></table>

Best scores in each column are in bold. Average summarizes the four diagnostic settings.  
Table 4: Survival prediction results averaged over five TCGA cohorts. Values are C-index ×100; higher is better.

Five-cohort average
<table><tr><td>Method</td><td>ResNet-50</td><td>PLIP</td><td>CONCH v1.5</td></tr><tr><td>CLAM</td><td>58.8</td><td>59.0</td><td>64.4</td></tr><tr><td>ABMIL</td><td>59.1</td><td>60.9</td><td>62.5</td></tr><tr><td>TransMIL</td><td>65.5</td><td>64.9</td><td>63.9</td></tr><tr><td>S4MIL</td><td>65.1</td><td>63.0</td><td>64.7</td></tr><tr><td>RRTMIL</td><td>66.0</td><td>63.7</td><td>65.2</td></tr><tr><td>MambaMIL</td><td>66.7</td><td>66.0</td><td>65.5</td></tr><tr><td>CAMIL</td><td>65.7</td><td>65.2</td><td>64.6</td></tr><tr><td>RPMIL</td><td>66.1</td><td>66.2</td><td>65.6</td></tr><tr><td>ACMIL</td><td>66.0</td><td>65.7</td><td>65.0</td></tr><tr><td>Jigsaw</td><td>66.9</td><td>67.1</td><td>67.0</td></tr><tr><td>LEAF-Mamba</td><td>67.0</td><td>67.3</td><td>67.3</td></tr></table>

Best scores in each column are in bold.

While Figure 4 establishes the overall predictive performance, Tables 5–7 examine the contributions of the adaptation module, sequence branches, and fusion mechanism under controlled configurations.

## 4.4. Ablation Study

Tables 5–7 evaluate three aspects of LEAF-Mamba: branch and fusion design, latent feature adaptation, and component behavior across feature extractors and prediction tasks. The branch/fusion and adapter comparisons use the BRACS official split with CONCH v1.5 features, while the cross-setting analysis additionally includes PLIP features, Camelyon16, and TCGA-BRCA.

Table 5: Branch and fusion comparisons on BRACS official with CONCH v1.5 features. Values are percentages; higher is better.
<table><tr><td>ID</td><td>LEA</td><td>Branches</td><td>Fusion</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>F1</td><td>Yes</td><td>Local only</td><td>一</td><td>75.7</td><td>35.3</td><td>27.4</td></tr><tr><td>F2</td><td>Yes</td><td>Global only</td><td>一</td><td>76.8</td><td>36.4</td><td>29.2</td></tr><tr><td>F3</td><td>Yes</td><td>Global + local</td><td>Concat</td><td>77.2</td><td>37.0</td><td>30.4</td></tr><tr><td>F4</td><td>Yes</td><td>Global + local</td><td>Add</td><td>77.0</td><td>36.9</td><td>31.0</td></tr><tr><td>F5</td><td>Yes</td><td>Global + local</td><td>Mean</td><td>77.1</td><td>36.8</td><td>30.8</td></tr><tr><td>F6</td><td>Yes</td><td>Global + local</td><td>Equal gate</td><td>77.5</td><td>37.6</td><td>31.4</td></tr><tr><td>F7</td><td>Yes</td><td>Global + local</td><td>Static gate</td><td>77.8</td><td>37.9</td><td>31.3</td></tr><tr><td>F8</td><td>Yes</td><td>Global + local</td><td>Gated</td><td>78.6</td><td>40.0</td><td>32.7</td></tr><tr><td>F9</td><td>一</td><td>Global + local</td><td>Concat</td><td>76.7</td><td>37.0</td><td>29.8</td></tr></table>

LEA denotes latent evidence adaptation. F8 is the complete model. Best scores in each metric are in bold.

Table 6: Adapter comparisons on BRACS official with CONCH v1.5 features under global-only aggregation. Performance values are percentages.  
Adapter performance and parameter counts
<table><tr><td></td><td colspan="2">Parameters (K)</td><td colspan="3"></td></tr><tr><td>Adapter</td><td>Train</td><td>Infer.</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>No adapter (frozen PCA bridge)</td><td>0.0</td><td>0.0</td><td>75.1</td><td>34.4</td><td>27.4</td></tr><tr><td>Linear projection</td><td>196.9</td><td>196.9</td><td>75.5</td><td>36.4</td><td>27.9</td></tr><tr><td>Two-layer MLP</td><td>721.9</td><td>721.9</td><td>76.8</td><td>36.1</td><td>28.3</td></tr><tr><td>Residual bottleneck</td><td>722.4</td><td>722.4</td><td>76.4</td><td>35.7</td><td>28.5</td></tr><tr><td>adapter LEA without</td><td>723.5</td><td>723.5</td><td>75.8</td><td>35.8</td><td>28.1</td></tr><tr><td>reconstruction Full LEA</td><td>1250.0</td><td>723.5</td><td>76.8</td><td>36.4</td><td>29.2</td></tr></table>

K denotes 1,000 parameters; counts are rounded to one decimal place. Training includes the reconstruction decoder; inference excludes it. Best performance scores are in bold.

(d) TCGA-BRCA / CONCH v1.5  
Table 7: Cross-setting component comparisons. Classification values are percentages; survival values are C-index ×100. Higher is better.  
(a) BRACS official / CONCH v1.5
<table><tr><td>ID</td><td>LEA</td><td>Aggregation</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>C1</td><td>一</td><td>Global only</td><td>76.6</td><td>35.3</td><td>27.7</td></tr><tr><td>C2</td><td>Yes</td><td>Global only</td><td>76.8</td><td>36.4</td><td>29.2</td></tr><tr><td>C3</td><td>一</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>77.2</td><td>37.4</td><td>31.6</td></tr><tr><td>C4</td><td>Yes</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>78.6</td><td>40.0</td><td>32.7</td></tr></table>

(b) BRACS official / PLIP
<table><tr><td>ID</td><td>LEA</td><td>Aggregation</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>C1</td><td>一</td><td>Global only</td><td>76.9</td><td>35.9</td><td>26.5</td></tr><tr><td>C2</td><td>Yes</td><td>Global only</td><td>75.5</td><td>34.4</td><td>27.6</td></tr><tr><td>C3</td><td>一</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>77.2</td><td>29.6</td><td>29.3</td></tr><tr><td>C4</td><td>Yes</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>77.8</td><td>36.0</td><td>29.8</td></tr></table>

<table><tr><td>ID</td><td>LEA</td><td>Aggregation</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>C1</td><td>一</td><td>Global only</td><td>97.2</td><td>92.2</td><td>89.9</td></tr><tr><td>C2</td><td>Yes</td><td>Global only</td><td>97.8</td><td>92.7</td><td>89.7</td></tr><tr><td>C3</td><td>一</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>97.5</td><td>93.8</td><td>92.7</td></tr><tr><td>C4</td><td>Yes</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>97.9</td><td>94.2</td><td>92.5</td></tr></table>

<table><tr><td>ID</td><td>LEA</td><td>Aggregation</td><td>C-index</td></tr><tr><td>C1</td><td>一</td><td>Global only</td><td>67.2</td></tr><tr><td>C2</td><td>Yes</td><td>Global only</td><td>68.1</td></tr><tr><td>C3</td><td>一</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>66.0</td></tr><tr><td>C4</td><td>Yes</td><td> $\mathrm { { G l o b a l + l o c a l ^ { * } } }$ </td><td>68.0</td></tr></table>

denotes gated global–local fusion. C4 is the complete model. Best scores in each setting are in bold.

Table 5 first isolates the aggregation design with LEA enabled. The global-only model achieves 76.8 AUC, 36.4 ACC, and 29.2 F1, exceeding the local-only model on all three metrics. Combining both branches through concatenation, addition, or averaging further improves all three scores over global-only aggregation. Inputdependent gated fusion achieves the highest scores of 78.6 AUC, 40.0 ACC, and 32.7 F1, improving on the global-only model by 1.8, 3.6, and 3.5 percentage points, respectively. Compared with the learned static gate, it gains 0.8 AUC, 2.1 ACC, and 1.4 F1 points. These comparisons support both the use of parallel long-range and sequence-local branches and input-dependent modulation of their joint representation.

Table 6 compares adapters under global-only aggregation. Full LEA achieves 76.8 AUC, 36.4 ACC, and 29.2 F1, compared with 75.1, 34.4, and 27.4 for the frozen PCA bridge. It matches the two-layer MLP in AUC while improving ACC and F1 by 0.3 and 0.9 points. Removing the reconstruction objective reduces AUC, ACC, and F1 to 75.8, 35.8, and 28.1, respectively, demonstrating the contribution of reconstruction regularization in this setting. Full LEA uses 1,250.0K adapter parameters during training and 723.5K during inference; the reconstruction decoder adds training capacity while leaving the inference parameter count equal to that of LEA without reconstruction.

Table 7 examines how adaptation and aggregation work together across four evaluation settings. Relative to global-only aggregation without LEA, the complete model raises AUC from 76.6 to 78.6 on BRACS with CONCH, from 76.9 to 77.8 on BRACS with PLIP, and from 97.2 to 97.9 on Camelyon16. It achieves the highest AUC and ACC in all three classification settings, and the highest F1 in both BRACS settings. On Camelyon16, gated global–local aggregation without LEA has a slightly higher F1 of 92.7 versus 92.5 for the complete model. The effect of adaptation also depends on the aggregator: with PLIP features, adding LEA changes global-only AUC from 76.9 to 75.5, but improves gated global–local AUC from 77.2 to 77.8. For TCGA-BRCA, LEA raises the global-only C-Index from 67.2 to 68.1, while the complete model scores 68.0. Thus, the combined design delivers the strongest classification AUC across the tested settings, while the survival comparison attributes the main improvement to feature adaptation.

## 4.5. Qualitative Case Study

Figure 5 shows attention patterns on one BRACS malignant carcinoma example. The baseline predicts benign hyperplasia, whereas LEAF-Mamba predicts malignant carcinoma; the displayed top-k patches and marked regions provide a qualitative comparison. Work on attention explanations shows that interpretation depends on the model, the definition of explanation, and appropriate counterfactual tests [49, 50]. We consequently use this case to describe the model’s selection pattern, without inferring localization accuracy, faithful attribution, or clinical utility from one heatmap.

![](images/0b49b312ed890b3b697a06e48e8340e9d3147dcf24eb26bb2513f7d2a3e2df09.jpg)  
Figure 5: Qualitative attention comparison on a BRACS malignant carcinoma case. The displayed top-k patches differ between the models, and LEAF-Mamba predicts the correct malignant carcinoma label in this example.

## 5. Discussion and Limitations

## 5.1. Interpretation of the Component Analyses

The component analyses distinguish the effects of feature adaptation from those of context fusion. In the BRACS branch comparison, parallel global–local processing improves on either branch alone, and input-dependent fusion further improves on a learned static gate. In the adapter comparison, removing reconstruction reduces performance, supporting the use of input recovery as an auxiliary training objective. These results connect the design choices to their controlled variants: adaptation changes the features available to aggregation, while fusion changes how complementary sequence representations are combined.

The cross-setting results also show that the contributions of the two components depend on the task and the aggregator. With PLIP features on BRACS, adaptation improves the gated global–local model but does not improve the global-only model’s AUC. For TCGA-BRCA, the adapted global-only variant slightly exceeds the complete model in C-Index. Thus, the strongest average benchmark performance of the complete framework does not imply that each component improves every setting. The evidence supports evaluating adaptation and aggregation jointly, with their individual contributions assessed under matched feature extractors and protocols.

## 5.2. Scope and Limitations

Our study focuses on learning from frozen patch representations, enabling controlled comparisons of adaptation and aggregation across feature extractors. Within this setting, the information available to the model depends on the pretrained encoder; exploring selective encoder fine-tuning would help establish when adaptation at the feature level is sufficient and when updating the encoder offers additional benefits. In addition, local interactions are defined along a fixed coordinate-derived sequence. Comparing alternative spatial orderings and incorporating explicit twodimensional relationships are promising directions for extending the fusion design. Finally, the current evaluation covers multiple public benchmarks, while transfer across institutions remains an important next step. Independent cohorts with different patient populations and slide preparation practices would help characterize the conditions under which the observed benefits generalize.

## 6. Conclusion

We presented LEAF-Mamba, a frozen-feature WSI framework combining reconstruction-regularized token adaptation, long-range sequence modeling, sequence-local convolution, and input-dependent fusion. Our aim was to address a central challenge in weakly supervised computational pathology: adapting pretrained patch representations to downstream tasks while bringing together complementary local and global information. Across feature extractors and prediction tasks, LEAF-Mamba achieves strong overall performance in both diagnostic classification and survival prediction. The component analyses further highlight the value of jointly designing feature adaptation and contextual aggregation. More broadly, these findings suggest that effective use of pathology foundation models depends on how their representations are adapted and integrated into slide-level decisions. This perspective offers a direction for connecting reusable pretrained features with diverse pathology tasks. Future work will extend this approach to cross-institutional evaluation and multimodal learning, exploring how tissue representations can be combined with complementary patient information for more comprehensive disease characterization.

## References

[1] C. L. Srinidhi, O. Ciga, A. L. Martel, Deep neural network models for computational histopathology: A survey, Medical Image Analysis 67 (2021) 101813.

[2] G. Campanella, M. G. Hanna, L. Geneslaw, A. Miraflor, et al., Clinical-grade computational pathology using weakly supervised deep learning on whole slide images, Nature Medicine 25 (8) (2019) 1301–1309.

[3] M. Ilse, J. Tomczak, M. Welling, Attention-based deep multiple instance learning, in: Proceedings of the International Conference on Machine Learning, 2018, pp. 2127–2136.

[4] M. Y. Lu, D. F. K. Williamson, T. Y. Chen, R. J. Chen, et al., Data-efficient and weakly supervised computational pathology on whole-slide images, Nature Biomedical Engineering 5 (6) (2021) 555–570.

[5] B. Li, Y. Li, K. W. Eliceiri, Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 14318–14328.

[6] H. Zhang, Y. Meng, Y. Zhao, Y. Qiao, et al., DTFD-MIL: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18802–18812.

[7] R. J. Chen, T. Ding, M. Y. Lu, D. F. K. Williamson, et al., Towards a generalpurpose foundation model for computational pathology, Nature Medicine 30 (3) (2024) 850–862. doi:10.1038/s41591-024-02857-3.

[8] E. Vorontsov, A. Bozkurt, A. Casson, G. Shaikovski, et al., A foundation model for clinical-grade computational pathology and rare cancers detection, Nature Medicine 30 (10) (2024) 2924–2935.

[9] Z. Shao, H. Bian, Y. Chen, Y. Wang, et al., TransMIL: Transformer based correlated multiple instance learning for whole slide image classification, in: Advances in Neural Information Processing Systems, Vol. 34, 2021, pp. 2136–2147.

[10] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, in: Proceedings of the Conference on Language Modeling, 2024. arXiv:2312.00752, doi:10.48550/arXiv.2312.00752. URL https://arxiv.org/abs/2312.00752

[11] S. Yang, Y. Wang, H. Chen, MambaMIL: Enhancing long sequence modeling with sequence reordering in computational pathology, in: Medical Image Computing and Computer Assisted Intervention, 2024, pp. 296–306. doi: 10.1007/978-3-031-72083-3\_28. URL https://doi.org/10.1007/978-3-031-72083-3\_28

[12] L. Gan, X. Wu, Z. Wang, L. Qu, S. Wu, J. Zhang, X. Sun, SemaMIL: Semantic-Aware Multiple Instance Learning with Retrieval-Guided State Space Modeling for Whole Slide Images, in: ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 9067–9071. doi:10.1109/icassp55912.2026.11464264.

[13] W. Tang, F. Zhou, S. Huang, X. Zhu, Y. Zhang, B. Liu, Feature re-embedding: Towards foundation model-level performance in computational pathology, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11343–11352. doi:10.1109/CVPR52733. 2024.01078.

[14] O. Fourkioti, M. De Vries, C. Jin, D. C. Alexander, C. Bakal, CAMIL: Context-aware multiple instance learning for cancer detection and subtyping in whole slide images, in: Proceedings of the Twelfth International Conference on Learning Representations, 2024. arXiv:2305.05314, doi:10.48550/arXiv.2305.05314. URL https://arxiv.org/abs/2305.05314

[15] Y. Zhang, H. Li, Y. Sun, S. Zheng, C. Zhu, L. Yang, Attention-challenging multiple instance learning for whole slide image classification, in: Proceedings of the European Conference on Computer Vision, 2024, accepted by ECCV 2024. arXiv:2311.07125, doi:10.48550/arXiv.2311.07125. URL https://arxiv.org/abs/2311.07125

[16] Z. Zhao, K. Chen, J. Zhao, RPMIL: Rethinking uncertainty-aware probabilistic multiple instance learning for whole slide pathology diagnosis, in: Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 2025, pp. 2467–2475. doi:10.24963/ijcai.2025/275. URL https://doi.org/10.24963/ijcai.2025/275

[17] X. Chen, P. Qiu, W. Zhu, H. Wang, H. Li, X. Dong, X. Sun, X. Yu, Y. Wang, A. Razi, A. Sotiras, Cracking instance jigsaw puzzles: An alternative to multiple instance learning for whole slide image analysis, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, accepted by ICCV 2025. arXiv:2507.08178, doi:10.48550/arXiv.2507.08178. URL https://arxiv.org/abs/2507.08178

[18] L. Gan, J. Zhang, X. Zhang, C. Xu, S. Wu, X. Sun, Beyond the Dominant Label: Recovering Coexisting Lesion Evidence with Multiple Instance Learning, arXiv preprint arXiv:2610.22091 (2026). doi:10.48550/arXiv.2610. 22091.

[19] T. Lin, Z. Yu, H. Hu, Y. Xu, C.-W. Chen, Interventional bag multiinstance learning on whole-slide pathological images, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19830–19839.

[20] Y. Zhao, F. Yang, Y. Fang, H. Liu, et al., Predicting lymph node metastasis using histopathological images based on multiple instance learning with deep graph convolution, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 4837–4846.

[21] M. Liang, Q. Chen, B. Li, L. Wang, et al., Interpretable classification of pathology whole-slide images using attention based context-aware graph convolutional neural network, Computer Methods and Programs in Biomedicine 229 (2023) 107268.

[22] R. J. Chen, M. Y. Lu, M. Shaban, C. Chen, et al., Whole slide images are 2d point clouds: Context-aware survival prediction using patch-based graph convolutional networks, in: Medical Image Computing and Computer Assisted Intervention, 2021, pp. 339–349. arXiv:2107.13048, doi:10. 1007/978-3-030-87237-3\_33. URL https://doi.org/10.1007/978-3-030-87237-3\_33

[23] R. J. Chen, C. Chen, Y. Li, T. Y. Chen, et al., Scaling vision transformers to gigapixel images via hierarchical self-supervised learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16123–16134. arXiv:2206.02647, doi:10.1109/CVPR52688. 2022.01567. URL https://doi.org/10.1109/CVPR52688.2022.01567

[24] Y. Liu, Y. Tian, Y. Zhao, H. Yu, et al., VMamba: Visual state space models, Advances in Neural Information Processing Systems 37 (2024) 103031– 103063.

[25] L. Fillioux, J. Boyd, M. Vakalopoulou, P.-H. Cournède, et al., Structured state space models for multiple instance learning in digital pathology, in: Medical Image Computing and Computer Assisted Intervention, 2023, pp. 594–604.

URL https://proceedings.mlr.press/v97/houlsby19a.html

[26] X. Wang, S. Yang, J. Zhang, M. Wang, et al., Transformer-based unsupervised contrastive learning for histopathological image classification, Medical Image Analysis 81 (2022) 102559. doi:10.1016/j.media.2022.102559. URL https://doi.org/10.1016/j.media.2022.102559

[27] Z. Huang, F. Bianchi, M. Yuksekgonul, T. J. Montine, et al., A visuallanguage foundation model for pathology image analysis using medical twitter, Nature Medicine 29 (9) (2023) 2307–2316.

[28] H. Xu, N. Usuyama, J. Bagga, S. Zhang, et al., A whole-slide foundation model for digital pathology from real-world data, Nature 630 (8015) (2024) 181–188. doi:10.1038/s41586-024-07441-w.

[29] T. Ding, S. J. Wagner, A. H. Song, R. J. Chen, et al., A multimodal whole-slide foundation model for pathology, Nature Medicine 31 (2025) 3749–3761.

[30] L. Gan, J. Zhang, L. Qu, Y. Wang, S. Wu, X. Sun, Enhancing zero-shot brain tumor subtype classification via fine-grained patch-text alignment, Expert Systems with Applications 300 (2026) 130161. doi:10.1016/j.eswa.2025. 130161.

[31] Y. Wang, S. Wu, L. Gan, Z. Zhang, J. Zhang, Z. Hu, H. Zhu, P. Wu, X. Sun, MeDKCoOp: Dual Knowledge-guided Graph Prompt Learning for Biomedical Vision-Language Models, in: Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 3635–3644. doi: 10.1145/3746027.3755135.

[32] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, S. Gelly, Parameter-efficient transfer learning for NLP, in: Proceedings of the 36th International Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 2790–2799.

[33] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, Y. Qiao, CLIP-Adapter: Better vision-language models with feature adapters, International Journal of Computer Vision 132 (2) (2024) 581–595. doi: 10.1007/s11263-023-01891-x.

[34] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, R. Pang, Conformer: Convolution-augmented transformer for speech recognition, in: Proceedings of Interspeech 2020, 2020, pp. 5036– 5040. doi:10.21437/Interspeech.2020-3015.

[35] X. Wu, L. Gan, S. Wu, J. Zhang, Y. Ou, X. Sun, SSCM: A Spatial-Semantic Consistent Model for Multi-Contrast MRI Super-Resolution, in: ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 11112–11116. doi:10.1109/icassp55912. 2026.11464902.

[36] H. Zhang, H. Hu, L. Gan, H. You, W. Yu, J. Huang, Geometry-Aware Reconstruction of Large Vision-Language Models from Dense into Mixtureof-Experts, in: ICASSP 2026 - 2026 IEEE International Conference on

Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 12717–12721. doi:10.1109/icassp55912.2026.11463650.

[37] H. Zhang, H. You, Z. Zhang, L. Gan, H. Zhang, W. Huang, J. Huang, Mitigating Generic Token Dominance in Cross-Domain Foundation Model for Text-Attributed Graphs, in: Database Systems for Advanced Applications, 2026, pp. 251–265. doi:10.1007/978-981-92-0366-6\_16.

[38] H. Zheng, H. You, Z. Liu, Z. Zhang, L. Gan, H. Zhang, W. Huang, J. Huang, G<sup>2</sup>rammar: Bilingual Grammar Modeling for Enhanced Text-Attributed Graph Learning, in: Database Systems for Advanced Applications, 2026, pp. 455–470. doi:10.1007/978-981-92-0366-6\_28.

[39] H. Zhang, T. Zhang, Z. Liu, Y. Shi, Y. Shen, H. You, H. Hu, L. Gan, J. Huang, H4G: Unlocking Faithful Inference for Zero-Shot Graph Learning in Hyperbolic Space, arXiv preprint arXiv:2510.12094, version 1 (2025). doi:10.48550/arXiv.2510.12094.

[40] H. Zhang, Y. Shi, X. Gu, Z. Zhang, H. You, L. Gan, Y. Yuan, J. Huang, HyperAgent: Leveraging Hypergraphs for Topology Optimization in Multi-Agent Communication, in: Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems, International Foundation for Autonomous Agents and Multiagent Systems, 2026, pp. 3087–3096. doi:10.65109/qtvf9552.

[41] H. Zhang, Y. Zhong, L. Gan, Z. Chen, J. Wu, Y. Shi, X. Gu, H. Zhang, H. You, J. Huang, EvoHyper: Evolving Hypergraph Topologies for Unified Collaboration in Multi-Agent Communication, in: Findings of the Association for Computational Linguistics: ACL 2026, Association for Computational Linguistics, 2026, pp. 25130–25143. doi:10.18653/v1/2026.findings-acl.1258.

[42] H. Zhang, Y. Shi, X. Gu, H. You, Z. Zhang, L. Gan, Y. Yuan, J. Huang, D<sup>3</sup>MAS: Decompose, Deduce, and Distribute for Enhanced Knowledge Sharing in Multi-Agent Systems, in: Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems, International Foundation for Autonomous Agents and Multiagent Systems, 2026, pp. 3106– 3115. doi:10.65109/wqmu8577.

[43] H. Zhang, H. Hu, G. Cheng, L. Gan, H. Wei, J. Huang, Boosting Knowledge Sharing Among Agents via Graph Decoupling, in: ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 17702–17706. doi:10.1109/icassp55912.2026. 11462380.

[44] T. Roetzer-Pejrimovsky, A.-C. Moser, B. Atli, C. C. Vogel, et al., The digital brain tumour atlas, an open histopathology resource, Scientific Data 9 (1) (2022) 55.

[45] N. Brancati, A. M. Anniciello, P. Pati, D. Riccio, et al., BRACS: A dataset for breast carcinoma subtyping in H&E histology images, Database 2022 (2022) baac093.

[46] B. Ehteshami Bejnordi, M. Veta, P. J. van Diest, B. van Ginneken, et al., Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer, JAMA 318 (22) (2017) 2199– 2210.

[47] J. N. Weinstein, E. A. Collisson, G. B. Mills, K. R. Shaw, et al., The cancer genome atlas pan-cancer analysis project, Nature Genetics 45 (10) (2013) 1113–1120.

[48] W. Tang, S. Huang, X. Zhang, F. Zhou, Y. Zhang, B. Liu, Multiple instance learning framework with masked hard instance mining for whole slide image classification (2023). arXiv:2307.15254, doi:10.48550/arXiv.2307. 15254. URL https://arxiv.org/abs/2307.15254

[49] S. Jain, B. C. Wallace, Attention is not explanation, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Association for Computational Linguistics, 2019, pp. 3543–3556. doi:10.18653/v1/N19-1357. URL https://aclanthology.org/N19-1357/

[50] S. Wiegreffe, Y. Pinter, Attention is not not explanation, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, 2019, pp. 11–20. doi:10.18653/v1/D19-1002. URL https://aclanthology.org/D19-1002/