# MCDA: Multi-View Graph Contrastive Learning with Dynamic Topology Augmentation for Functional MRI Analysis

Hao Zhang, Graduate Student Member, IEEE, Xiaoyun Liu, Shuo Huang, Yue Ma, Liang Cao<sup>B</sup>, Mingxia Liu<sup>B</sup>, Senior Member, IEEE, and Li Zhang<sup>B</sup>

Abstract—Resting-state fMRI (rs-fMRI) is a vital non-invasive tool for diagnosing brain disorders and uncovering pathological mechanisms. However, existing graph contrastive learning (GCL) methods often underexploit augmentation histories across training epochs and the relationships among diferent augmented views. In this paper, we introduce a topology-guided Multiview graph Contrastive learning framework with Dynamic graph topology Augmentation (MCDA) for fMRI analysis. Unlike traditional GCL that uses fixed noise or edge-dropping, MCDA employs a Dynamic Augmentation Update (DAU) module that iteratively refines the augmented topology. Specifically, we incorporate self-aware and cross-sample augmentation strategies to simultaneously capture individual-specific functional signatures and inter-subject heterogeneity. We then design a topology-guided multi-view graph contrastive learning scheme that explores interrelationships not only between the original network and its augmented views, but also across diferent augmented views to maximize structural information preservation. We characterize the score-level optimization directions and establish a lower bound for the contrastive objective. The framework combines classification and contrastive losses for brain disorder diagnosis and examines discriminative functional connectivity patterns as potential biomarkers. Experiments on two public rs-fMRI datasets comprising 1,236 subjects demonstrate that MCDA outperforms the state-of-the-art methods. The code has been made publicly available at: https://github.com/goodcodingagoodboy/MCDA.

Index Terms—Resting-state Functional MRI, Graph Contrastive Learning, Graph Topology Augmentation, Brain Disorder Diagnosis

## I. INTRODUCTION

R <sup>ESTING-STATE</sup> <sup>functional</sup> <sup>MRI</sup> <sup>(rs-fMRI)</sup> <sup>is</sup> <sup>a</sup> <sup>pow-</sup><sub>erful non-invasive tool for uncovering the pathological</sub> erful non-invasive tool for uncovering the pathological mechanisms of brain disorders [1], [2]. Rs-fMRI data can be modeled as non-Euclidean graphs that capture functional connectivity patterns and interactions between brain regions of interests (ROIs), facilitating automated diagnosis of brain disorders like major depressive disorder (MDD) and autism spectrum disorder (ASD) [3].

Various graph-based learning methods have been developed to extract discriminative representations from fMRI data, such as BrainGNN [4] and graph transformer-based foundation model [5]. Li et al. [4] proposed a clustering-based embedding method in the graph convolutional layer and provided users the flexibility to interpret diferent levels of biomarkers through graph node pooling and several innovative loss terms to regulate the pooling operations. Instead of treating brain networks as graphs for processing, some studies also convert rs-fMRI from graph structures into low-dimensional vectors for embedded learning. Wang et al. [5] proposed a graph transformer-based foundation model for brain functional connectivity networks (FCN) that integrates node and edge embeddings to extract versatile features.

Graph contrastive learning (GCL) has been investigated for fMRI representation learning and brain disorder analysis [6], [7]. By treating each brain network as a graph structure, GCL typically generates diferent augmented views of a brain network and then optimizes feature space via a contrastive loss, thereby boosting robustness and accuracy in downstream tasks. Beyond generating augmented views at the current training step, we investigate how their evolution across epochs can support history-guided augmentation refinement. Furthermore, diferent augmented views are expected to express distinct functional connectivities from brain networks, enabling a comprehensive representation of topological structures. Nevertheless, most existing GCL methods [6], [8], [9] fail to model the interrelationships among augmented views and tend to overlook feature expression between augmented networks. Those limitations often afect discriminative power of graph representation, ultimately afecting the accuracy and robustness of task-specific predictions.

To address these challenges, we introduce a topologyguided Multi-view graph Contrastive learning framework with Dynamic graph topology Augmentation (MCDA) for brain disorder identification and fMRI analysis. As illustrated in Fig. 1(a), the MCDA framework employs dynamic graph topology augmentation to continuously generate two complementary brain networks, integrating self-aware and crosssample augmentation to model both subject-specific traits and functional variability across individuals. Then, we propose a topology-guided multi-view graph contrastive learning scheme, which can capture not only the interrelationships between each augmented view and the original network but also the interrelationships between diferent augmented views. A similarity-score gradient analysis characterizes the optimization directions of these constraints, and a boundedness analysis establishes a lower bound for the contrastive objective. We evaluate the MCDA on two public rs-fMRI datasets (1,236 subjects), with experimental results validating that our MCDA surpasses several state-of-the-art (SOTA) approaches in automated brain disorder identification.

![](images/c6518c18740d17434e63f762f6ebd8f4b31f51afbe722ae83c6dda74020d56a8.jpg)  
(b) Specific process of Topology Augmentation  
(c) Specific process of Dynamic Augmentation Update  
Fig. 1. Overview of the proposed MCDA framework for rs-fMRI analysis. (a) The framework comprises brain network construction, dynamic graph augmentation, and topology-guided multi-view graph contrastive learning. (b) Self-aware augmentation provides subject-specific modulation, while crosssample augmentation introduces spatial and temporal perturbations from other subjects. (c) The DAU module uses a shared BiGRU to summarize cross-epoch augmentation histories and sigmoid-gated fusion to refine current augmented views using branch-specific historical references.

The main contributions of this work are summarized below:

1) We develop a dynamic augmentation update module that adaptively refines augmented brain networks through history-guided gated fusion.

2) The proposed dual-augmentation strategy combines subject-specific modulation with cross-sample spatial and temporal perturbations to generate complementary brain network views.

3) A topology-guided multi-view contrastive scheme balances consistency and diversity through constraints between original views and across augmented views.

4) We characterize the optimization directions of the contrastive constraints through similarity-score gradient analysis. Experiments on two public rs-fMRI datasets of 1,236 subjects assess classification performance and explore model-derived brain region associations.

The remainder of this paper is organized as follows. We briefly review the most relevant work in Section II. Section III interprets our proposed MCDA framework, while Section IV covers materials, experimental settings and experimental results. Section V examines key components, parameter sensitivity, brain region associations, and limitations of MCDA. Finally, this paper is concluded in Section VI.

## II. RELATED WORK

## A. Graph Augmentation of FMRI-Derived Brain Networks

Graph augmentation generates diverse brain network views to support robust representation learning from rs-fMRI data. Existing techniques can largely be categorized as static epochbased augmentation strategies. Early representative methods [6], [10] perform graph augmentation directly on the original BOLD time series using window slicing to generate diverse training samples while preserving temporal continuity. More recently, deep generative models have been employed to produce diverse views that conform to the data distribution. For instance, Wang et al. [11] integrated difusion models with graph augmentation, simultaneously modifying both the topological structure and node features to enhance brain network representations. Li et al. [12] perform graph augmentation via multi-view contrastive self-supervised learning, where a learner-instructor framework enforces consistency and enables self-updating to mine hidden topological connections from raw fMRI data.

Augmentation strategies driven solely by current representations do not explicitly exploit the evolution of augmented views across training epochs. In contrast, MCDA summarizes recent augmentation histories using a shared BiGRU and adaptively fuses current views with historical references through sigmoid gating, enabling history-guided refinement beyond current-step view generation.

## B. Graph Contrastive Learning for fMRI analysis

Graph contrastive learning (GCL) analyzes relationships between augmented views to learn brain network representations for rs-fMRI-based brain disorder diagnosis. By maximizing the agreement between diferently augmented views of the same brain network, GCL encourages the model to capture invariant and robust functional patterns [13]. Early works in this domain primarily utilized single-view settings, focusing on temporal or functional perturbations to project brain networks into a discriminative feature space [14]. More recently, research has shifted toward multi-view contrastive learning to leverage diverse perspectives of brain connectivity. For instance, Li et al. [12] developed a learner–instructor framework for unsupervised brain topology inference, using multi-view contrastive self-supervision to adaptively refine ROI interconnections. Zong et al. [7] proposed an alternative multi-view approach that computes similarity metrics at both node and graph level perspectives between the original brain networks and their graph difusion variants, enabling simultaneous capture of node and edge level features.

These studies illustrate the value of multi-view inputs and multi-level representations in GCL. Nevertheless, constructing multiple views does not by itself specify how consistency and diversity should be balanced across augmented representations. Building on our preliminary multi-view formulation [9], MCDA jointly regulates original to augmented and augmented to augmented relationships. Rather than uniformly maximizing agreement, it combines self-aware to original consistency with relative cross-sample to original and two augmented views diversity, encouraging complementary representations from the two augmentation branches.

## III. METHODOLOGY

In this work, we propose a topology-guided multi-view graph contrastive learning framework with dynamic graph topology augmentation (MCDA) for rs-fMRI-based brain disorder analysis. The framework is composed of three key components: 1) brain network construction, 2) dynamic graph topology augmentation, and 3) topology-guided multi-view graph contrastive learning. An overview of the proposed MCDA framework is presented in Fig. 1(a). More details are elaborated below.

## A. Brain Network Construction

To characterize brain dynamics and functional interactions, we use rs-fMRI to construct an undirected spatio-temporal brain network G and a functional connectivity matrix A [15]. For each subject, blood-oxygen-level-dependent (BOLD) signals from rs-fMRI represent an undirected spatio-temporal network $\mathbf { G } = ( \mathbf { V } , \mathbf { E } ) \in \mathbf { \bar { R } } ^ { T \times N }$ . The vertex set $\mathbf { V } = \{ v _ { t i } \mid t =$ $1 , \dots T ; i = 1 , \dots , N \}$ includes each ROI (region of interest) at every time point $t ,$ where T denotes the number of time points during scanning, and N is the number of brain regions.

The edge set E contains two subsets from both temporal and spatial connectivity. For each subject, the functional connectivity matrix $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ is constructed by computing Pearson correlations between the mean BOLD time series of all ROI pairs.

![](images/1e5e94953d4fa3d7eafd1216a30f57ccba1c073ef5cbec6dde951af45643e8b6.jpg)  
Fig. 2. The spatio-temporal GC Encoder consists of spatial and temporal graph convolution parts. Here, TP refers to time points, and ROI represents brain region nodes.

## B. Dynamic Graph Topology Augmentation

To fully leverage the topology structure of the brain network and optimize the propagation of discriminative information in augmented views, we propose a dynamic self-aware and crosssample topology augmentation to generate two complementary augmentations of brain network for both individual-specific characteristics and inter-subject functional heterogeneity.

1) Self-aware and Cross-sample Topology Augmentation: Considering the efect of sample sizes in topology augmentation, we extend the brain network G and A into three dimensions, represented as $\mathbf { R } ^ { B \times T \times N }$ and $\mathbf { R } ^ { B \times N \times N }$ . Here, B refers to the batch size used during each training epoch. And, the detailed procedure of self-aware and cross-sample topology augmentation is presented in Fig. 1(b).

As for self-aware augmentation, first, the input G is fed into Z-pool layer [16], producing the $\mathbf { G } _ { z }$ . The transformation equation for the Z-pool layer is given by:

$$
\begin{array} { l } { { \displaystyle { \bf G } _ { z } = { \bf Z } _ { \mathbf { \bar { \imath } } \mathbf { \mathrm { p o o l } } } ( { \bf G } ) = \mathbf { M a x } { \mathrm { P o o l } } ( { \bf G } ) \oplus \mathbf { A } \mathbf { v g } { \mathrm { P o o l } } ( { \bf G } ) } } \\ { = \left[ \underset { t , n \in \cal R } { \operatorname* { m a x } } \left( { \bf G } _ { t , n } \right) \oplus \frac { 1 } { \left| { \cal R } \right| } \sum _ { t , n \in \cal R } { \bf G } _ { t , n } \right] , } \end{array}\tag{1}
$$

where $\mathbf { G } _ { t , n }$ denotes the BOLD signal at time t and ROI n in brain network, and R represents the local neighborhood in the spatio-temporal domain, with size |R|. MaxPool extracts the maximum response within each local spatio-temporal neighborhood, while AvgPool computes the mean response over the same region. The operator ⊕ combines these complementary local statistics to form the pooled representation.

Next, the Spatio-temporal Graph Convolutional (ST-GC) encoder, as illustrated in Fig. 2, is employed to encode $\mathbf { G } _ { z } .$ It consists of two components—spatial and temporal parts— which jointly capture the structural and dynamic characteristics of brain functional connectivity. We use the $\mathbf { G } _ { z }$ as the initial input of the first spatial layer ${ \bf G } _ { s , i n } ^ { 1 }$ . The output feature of the l-th spatial GC encoder is expressed as:

$$
\mathbf { G } _ { s , o u t } ^ { l } = \Lambda ^ { - \frac { 1 } { 2 } } ( \mathbf { A } + \mathbf { I } ) \Lambda ^ { - \frac { 1 } { 2 } } \mathbf { G } _ { s , i n } ^ { l } W _ { S } ^ { l } ,\tag{2}
$$

For clarity, the subject and time indices are omitted in Eq. (2). Here, ${ \bf G } _ { s , \mathrm { i n } } ^ { l ^ { \prime } } \in \mathbb { R } ^ { \tilde { N } \times C _ { l } }$ contains the features of all ROIs for one subject at one time point, where $C _ { l }$ is the input feature dimension. I is the identity matrix, and Λ is diagonal with $\begin{array} { r } { \Lambda _ { i i } = \sum _ { j = 1 } ^ { N } ( A _ { i j } + I _ { i j } ) } \end{array}$ . The normalized adjacency acts on the ROI dimension, while $\mathbf { W } _ { S } ^ { l }$ transforms the feature channels. When it comes to the temporal GC encoder, the spatial features from the same layer $\mathbf { G } _ { s , o u t } ^ { l }$ serve as inputs to a temporal convolution operation, and the entire process can be formulated as follows:

![](images/c6d964ac7f31de352e20ef4bce8e20f34382f10a41554af42c385473592b2973.jpg)  
Fig. 3. Illustration of the DAU module. (a) Whole-graph projection and construction of a descriptor history window across training epochs. (b) Shared BiGRU encoding followed by decoding and sigmoid gate generation. (c) Gated fusion of the current augmentation with the preceding epoch’s pre-refinement graph. The self-aware and cross-sample branches share BiGRU parameters while maintaining separate histories, projections, and decoders.

$$
\mathbf { G } _ { t , o u t } ^ { l } = \psi \left( \phi \left( \mathbf { G } _ { s , o u t } ^ { l } \right) * W _ { T } ^ { l } \right) .\tag{3}
$$

where ∗ denotes the temporal convolution operation, $W _ { T }$ represents the convolution operation for each ROI with convolution kernel of $k \times 1$ . Here, $k = 1 1$ represents the temporal window of connected neighboring time steps. $\mathbf { G } _ { t , o u t } ^ { l }$ denotes the output of the l-th layer of the temporal part. ϕ denotes BN-ReLU operations. ψ represents BN-Dropout operations, which use a dropout rate of 0.5 to randomly discard features, preventing overfitting during training. Let $\Phi ^ { ( 0 ) } ( \cdot , A )$ denote the ST-GC encoder for self-aware augmentation, as described in Eq. 2 and 3. Following the multiplicative modulation in [13], the self-aware view is obtained as

$$
\mathbf { G } ^ { \mathrm { S A } } = \mathbf { G } _ { \mathrm { z } } \odot \sigma \Big ( \Phi ^ { ( 0 ) } ( G _ { z } , A ) \Big ) ,
$$

where σ denotes the sigmoid function and ⊙ denotes elementwise multiplication.

For cross-sample augmentation, two copies of original network G are permuted from $\mathbf { R } ^ { B \times T \times N }$ to $\mathbf { R } ^ { N \times T \times B }$ and $\mathbf { R } ^ { T \times B \times N }$ , respectively. The same Z-pool operation is applied over the last two dimensions of each permuted tensor, enabling batch–temporal and batch–spatial interactions. The permutation changes the tensor layout, while spatial graph convolution is applied along the ROI axis using the corresponding subject’s adjacency matrix. Temporal convolution operates along the time axis. Each pooled representation is processed by the ST-GC encoder and multiplied element-wise by sigmoid output. The corresponding inverse permutation restores the branch outputs to the original $\mathbf { R } ^ { B \times T \times N }$ ordering, and their equalweight average forms $\mathbf { G } ^ { \mathrm { C A } }$ . The resulting self-aware and crosssample views are subsequently refined by the DAU module.

2) Dynamic Augmentation Update Module: We introduce a dynamic augmentation update (DAU) module to refine augmented brain networks using historical information across training epochs, as shown in Fig. 3. Let $\mathbf { G } _ { e } ^ { r }$ denote the newly generated augmented network at epoch e, where $r \in$ {self-aware, cross-sample} identifies the augmentation branch.

A learnable projection $P _ { r } : \mathbf { R } ^ { \mathrm { T N } }  \mathbf { R } ^ { d }$ maps the complete augmented matrix to a d-dimensional descriptor:

$$
\mathbf { z } _ { e } ^ { r } = P _ { r } \left( \mathrm { v e c } \left( \mathbf { G } _ { e } ^ { r } \right) \right) ,\tag{4}
$$

The operator vec flattens the entire matrix into a vector. For each subject and augmentation branch, we organize the descriptors into a sliding window with a maximum length of $K \geq 1$ , containing the current epoch and up to $K - 1$ preceding epochs:

$$
\mathcal { Z } _ { e } ^ { r } = \left[ \mathbf { z } _ { \ell _ { e } } ^ { r } , \ldots , \mathbf { z } _ { e } ^ { r } \right] , \quad \ell _ { e } = \operatorname* { m a x } ( 1 , e - K + 1 ) ,\tag{5}
$$

The index $\ell _ { e }$ identifies the earliest epoch in the available window. A shared BiGRU processes the self-aware and crosssample descriptor sequences separately, using the same parameters across branches while maintaining separate hidden states. Its forward and backward recurrences are defined as follows:

$$
\overrightarrow { \mathbf { h } _ { j } ^ { r } } = \overrightarrow { \mathrm { G R U } } \left( \mathbf { z } _ { j } ^ { r } , \overrightarrow { \mathbf { h } _ { j - 1 } ^ { r } } \right) , \quad \overleftarrow { \mathbf { h } _ { j } ^ { r } } = \overleftarrow { \mathrm { G R U } } \left( \mathbf { z } _ { j } ^ { r } , \overleftarrow { \mathbf { h } _ { j + 1 } ^ { r } } \right) ,\tag{6}
$$

The index $j$ ranges from $\ell _ { e }$ to $e _ { \cdot }$ The arrows indicate the traversal directions, and each hidden state is an m-dimensional vector. The histories and hidden states remain branch-specific despite parameter sharing. The forward boundary state at $\ell _ { e } - 1$ and the backward boundary state at e+1 are initialized to zero. Both directions process only the available window and require no representations from future epochs.

The forward state at the current endpoint and the backward state at the earliest endpoint summarize the window. A learnable decoder $D _ { r } : \mathbb { R } ^ { \bar { 2 } m } \to \mathbb { R } ^ { \mathrm { T N } }$ maps their concatenation to an element-wise fusion gate. Let $\bar { \mathbf { M } } _ { e } ^ { r } \in \mathbb { R } ^ { \mathrm { T } \times \mathrm { N } }$ denote the historical reference, defined as the pre-refinement augmented graph from the preceding epoch for the same subject and branch. The reference and current graph share the same temporal and ROI coordinates, with $\mathbf { M } _ { 1 } ^ { r } = \mathbf { G } _ { 1 } ^ { r }$ in the first epoch. The gate and historical fusion are defined as follows:

$$
\begin{array} { r } { \mathbf { a } _ { e } ^ { r } = \mathrm { r e s h a p e } _ { \mathrm { T } \times \mathrm { N } } \bigg \{ \sigma \Big ( D _ { r } \left[ \overrightarrow { \mathbf { h } _ { e } ^ { r } } ; \overleftarrow { \mathbf { h } _ { \ell _ { e } } ^ { r } } \right] \Big ) \bigg \} , } \\ { \mathbf { G } _ { e } ^ { r } \gets \mathbf { G } _ { e } ^ { r } + \mathbf { a } _ { e } ^ { r } \odot ( \mathbf { M } _ { e } ^ { r } - \mathbf { G } _ { e } ^ { r } ) . \qquad } \end{array}\tag{7}
$$

Here, the semicolon denotes concatenation, σ is the elementwise sigmoid function, and $\mathrm { r e s h a p e } _ { \mathrm { T } \times \mathrm { N } }$ restores the matrix using the same ordering as vec. The gate $\mathbf { a } _ { e } ^ { r } \in ( 0 , 1 ) ^ { T \times N }$ controls the contribution of the historical reference, and $\odot$ denotes element-wise multiplication. The right-hand side uses the current graph before the update. Smaller gate values favor the current graph, whereas larger values favor the historical reference. Each updated entry lies between its current and historical values, without an additional global scaling hyperparameter. The first epoch produces an identity update.

3) Inference for Test Subjects: At the end of training, we retain the model checkpoints from the last $K$ consecutive epochs, including the final epoch $E .$ For each unseen subject $i ,$ the same input brain network and connectivity matrix are processed by the augmentation modules from checkpoints $j ~ = ~ E - K + 1 , \ldots , E$ . This produces a subject-specific sequence of pre-DAU augmented graphs $G _ { i , j } ^ { r }$ for each branch $r \in \{ \mathrm { S A } , \mathrm { C A } \}$ . Each graph is mapped to a descriptor using the projection from its corresponding checkpoint, $z _ { i , j } ^ { r }$ = $P _ { j } ^ { r } ( \mathrm { v e c } ( G _ { i , j } ^ { r } ) )$ . The descriptors are arranged in chronological checkpoint order to construct the history window $Z _ { i , E } ^ { r } .$ Thus, an augmentation history is constructed separately for each unseen subject using the saved model checkpoints.

The shared BiGRU and branch-specific decoders from the final checkpoint $\Theta _ { E }$ process these histories to compute the fusion gates, with the recurrent boundary states initialized to zero for each subject and branch. Following Eq. (7), the current augmented graph $G _ { i , E } ^ { r }$ is refined using the preceding checkpoint’s pre-DAU graph $G _ { i , E - 1 } ^ { r }$ as the historical reference. The classification branch from $\Theta _ { E }$ then processes the original network and the two refined augmented views, and their predicted probabilities are averaged according to Eq. (14).

All checkpoints are frozen and used in evaluation mode, without test-time parameter updates or access to test labels. The same history-construction and prediction procedure is applied during validation and testing.

## C. Topology-guided Multi-view Graph Contrastive Learning

We introduce a topology-guided multi-view graph contrastive learning scheme to jointly constrain the relationships among the original network and its two augmented views. Specifically, the objective ${ \mathcal { L } } _ { \mathrm { T M C } }$ combines a consistency constraint between the original network and the self-aware augmented view with diversity constraints involving the crosssample augmented view. This asymmetric design assigns consistency and diversity roles to the two augmentation branches.

1) Topology-guided Multi-view Graph Contrastive Loss: We perform multi-view graph contrastive learning by leveraging the semantic information of the topological structures in augmented brain networks. First, we use cosine similarity [17] to quantify the relationships among the three views. Taking the original brain network G and its self-aware augmented view $\mathbf { G } ^ { \mathrm { S \bar { A } } }$ as an example, the similarity is defined as:

$$
\mathrm { s i m } ( \mathbf { G } , \mathbf { G } ^ { \mathrm { S A } } ) = \frac { \left. \mathbf { G } , \mathbf { G } ^ { \mathrm { S A } } \right. _ { T , N } } { \operatorname* { m a x } ( \| \mathbf { G } \| _ { T , N } , \varepsilon ) \operatorname* { m a x } ( \| \mathbf { G } ^ { \mathrm { S A } } \| _ { T , N } , \varepsilon ) } ,\tag{8}
$$

Here, $\langle \cdot , \cdot \rangle _ { T , N }$ and $\Vert \cdot \Vert _ { T , N }$ denote the inner product and Euclidean norm computed over the temporal and ROI dimensions, respectively, with the batch dimension preserved. The

normalization and division are performed element-wise over the preserved batch dimension. Specifically, the inner product is defined as:

$$
\left. \mathbf { G } , \mathbf { G } ^ { \mathrm { { S A } } } \right. _ { T , N } = \sum _ { t = 1 } ^ { T } \sum _ { n = 1 } ^ { N } \mathbf { G } _ { : , t , n } \odot \mathbf { G } _ { : , t , n } ^ { \mathrm { { S A } } } ,\tag{9}
$$

The colon denotes the preserved batch dimension, and $\odot$ denotes element-wise multiplication. The constant ε ensures numerical stability. A higher similarity score indicates greater similarity between the original network and its self-aware augmented view.

The two augmentation branches are assigned diferent regularization roles. For the self-aware view $\mathbf { G } ^ { \mathrm { S A } }$ , we use the original network $\mathbf { G }$ as a consistency reference. Encouraging agreement between these matched views is intended to limit augmentation-induced deviations in representation direction and maintain a subject-level correspondence with the original network. Accordingly, $\mathcal { L } _ { \mathrm { S A } }$ favors stronger matching relative to other pairings within the mini-batch.

The cross-sample branch applies distinct transformations to spatial and temporal axes to generate an alternative view of the original network. Requiring this view to closely match the original representation may suppress the diferences introduced by these transformations. We therefore use $\mathcal { L } _ { \mathrm { C A } }$ to discourage the matched association relative to other pairings within the mini-batch, with the aim of retaining view diversity. The two constraints are formulated as follows:

$$
\mathcal { L } _ { \mathrm { S A } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { S A } } , \mathbf { G } _ { i } ) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { S A } } , \mathbf { G } _ { j } ) / \tau \right) } ,\tag{10}
$$

$$
\mathcal { L } _ { \mathrm { C A } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { C A } } , \mathbf { G } _ { i } ) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { C A } } , \mathbf { G } _ { j } ) / \tau \right) } ,\tag{11}
$$

where $\tau > 0$ is the temperature parameter, set to 0.5 in our experiments. The constraints involving the original network do not directly regulate the similarity between the two augmented views [18]. In particular, reducing the cross-sample to original matching score alone does not ensure that the cross-sample branch expresses a perspective distinct from the self-aware branch. We therefore introduce $\mathcal { L } _ { \mathrm { S A , C A } }$ to explicitly discourage strong matched associations between the augmented views relative to other pairings within the mini-batch. This additional constraint is intended to reduce overlap between the two augmentation branches and complement their respective relationships with the original network. The inter-view constraint is formulated as follows:

$$
\mathcal { L } _ { \mathrm { S A , C A } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { S A } } , \mathbf { G } _ { i } ^ { \mathrm { C A } } ) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \sin ( \mathbf { G } _ { i } ^ { \mathrm { S A } } , \mathbf { G } _ { j } ^ { \mathrm { C A } } ) / \tau \right) } ,\tag{12}
$$

Combining the three pairwise constraints yields the multiview graph contrastive loss: ${ \mathcal { L } } _ { \mathrm { T M C } } .$

$$
\mathcal { L } _ { \mathrm { T M C } } = \alpha \mathcal { L } _ { \mathrm { S A } } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { C A } } + \beta \mathcal { L } _ { \mathrm { S A , C A } } .\tag{13}
$$

where $\alpha \in [ 0 , 1 ]$ balances the consistency and diversity constraints, and $\beta \geq 0$ controls the contribution of the constraint between the two augmented views.

Together, these constraints combine an original-view consistency reference with two diversity-oriented relationships. They are used as similarity-based regularizers alongside the supervised classification objective. The optimization directions of these constraints can be characterized through their partial derivatives with respect to the similarity scores. For $B \geq 2$ and $\tau > 0 ,$ the negative log-ratio in $\mathcal { L } _ { \mathrm { S A } }$ has a negative derivative with respect to the matched similarity score, whereas the positive log-ratios in $\mathcal { L } _ { \mathrm { C A } }$ and $\mathcal { L } _ { \mathrm { S A , C A } }$ have positive derivatives. Thus, at the similarity-score level, minimizing these terms favors increasing the matched self-aware with original similarity and decreasing the matched cross-sample with original and self-aware with cross-sample similarities. The detailed derivation and the corresponding efects on non-matched pairs are provided in A.

2) Objective Function: For downstream brain disorder diagnosis, we employ a composite objective that combines the classification loss $\mathcal { L } _ { \mathrm { c l a s s } }$ and the contrastive loss ${ \mathcal { L } } _ { \mathrm { T M C } }$ . Specifically, the original brain networks G and the two augmented brain networks $\mathbf { G } ^ { \mathrm { S A } }$ and $\mathbf { G } ^ { \mathrm { C A } }$ are processed by the same classification branch. Each network is first passed through the ST-GC encoder to extract spatio-temporal features. These features are averaged over the ROI dimension and flattened into feature vectors, which are then fed into a shared classification head consisting of four successive $1 \times 1$ convolutional layers. The scalar output for each view is transformed by a sigmoid function to obtain its predicted patient probability. The three probabilities are averaged to obtain the final prediction, which is supervised using the ground-truth diagnostic label:

$$
\begin{array} { c } { { p _ { i } = \displaystyle \frac { 1 } { 3 } \left( p _ { i } ^ { \mathrm { o r i g } } + p _ { i } ^ { \mathrm { S A } } + p _ { i } ^ { \mathrm { C A } } \right) , } } \\ { { \displaystyle \mathcal { L } _ { \mathrm { c l a s s } } = - \sum _ { i = 1 } ^ { B } \left[ y _ { i } \log ( p _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - p _ { i } ) \right] . } } \end{array}\tag{14}
$$

Here, $p _ { i } ^ { \mathrm { o r i g } } , p _ { i } ^ { \mathrm { S A } }$ , and $p _ { i } ^ { \mathrm { C A } }$ denote the predicted patient probabilities for the original, self-aware, and cross-sample views of subject i, respectively. B is the mini-batch size, and $y _ { i } ~ \in ~ \{ 0 , 1 \}$ is the ground-truth label, with 1 indicating a patient and 0 indicating a healthy control. The classification loss provides diagnostic supervision through the averaged prediction, while ${ \mathcal { L } } _ { \mathrm { T M C } }$ regularizes the relationships among the three views through the consistency and diversity constraints described above. The overall training objective is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l a s s } } + \lambda \mathcal { L } _ { \mathrm { T M C } } . } \end{array}\tag{15}
$$

where $\lambda ~ \geq ~ 0$ controls the contribution of the multi-view contrastive regularization relative to the classification loss.

## D. Implementation Details

The framework is implemented using PyTorch [19] on an NVIDIA RTX 4090 GPU. The local neighborhood size in Zpool is set to $| R | { = } 1 6 .$ , corresponding to a 4×4 region. The ST-GC encoders contain three layers. The maximum DAU history window length is set to $K { = } 5$ , including the current epoch and up to four preceding epochs. The influence of the history window length is examined in Section V-E2. Each branch-specific projection $P _ { r }$ is implemented as a single fully connected layer mapping the flattened augmented graph to a descriptor of dimension d=128. The shared BiGRU contains three layers, with a hidden-state dimension of m=64 per direction and an inter-layer dropout rate of 0.3. The terminal states from the highest layer are concatenated into a 128- dimensional vector. Each branch-specific decoder $D _ { r }$ is a single fully connected layer mapping this vector to TN gate logits, followed by the sigmoid and reshape operations defined in Eq. (7). Both fully connected layers include biases.

Algorithm 1 Training Procedure of MCDA   
Input: Brain networks G, connectivity matrices A, and labels y   
Output: Trained model parameters Θ   
Parameters: Epochs E, batch size $B ,$ history length K, temperature   
τ, and loss weights $\lambda , \alpha , \beta$   
1: Initialize Θ, Adam, and subject- and branch-specific histories   
2: for $e = 1$ to E do   
3: for each training mini-batch do   
4: $\mathbf { G } _ { e } ^ { \mathrm { S A } }  \mathbf { Z } \mathbf { \bar { \tau } } \mathrm { p o o l } ( \mathbf { G } ) \odot \sigma ( \mathrm { S T } \mathbf { \bar { G } C } ( \mathbf { Z } \mathbf { \bar { \tau } } \mathbf { \mathrm { p o o l } } ( \mathbf { G } ) , \mathbf { A } ) )$   
5: $\mathbf { G } _ { 2 , z }  \overset { \cdot } { \underset { \cdot } { \mathrm { Z } } \mathrm { - p o o l } } ( \overset { \cdot } { \underset { \cdot } { \mathrm { R o t a t e } } } _ { N } ( \mathbf G ) )$   
6: $\mathbf { G } _ { 2 , s } \gets \mathrm { R o t a t e } _ { N } ^ { - 1 } [ \mathbf { G } _ { 2 , z } \odot \sigma ( \mathbf { S } \mathbf { T } \mathbf { - } \mathbf { G } \mathbf { C } ( \mathbf { G } _ { 2 , z } , \mathbf { A } ) ) ]$   
7: $\mathbf { G } _ { 3 , z } \gets \mathbf { Z } \mathbf { - p o o l } \dot { ( } \mathbf { R }$ otate<sub>T</sub>(G))   
8: $\mathbf { G } _ { 3 , t }  \mathrm { R o t a t e } _ { T } ^ { - 1 } [ \mathbf { G } _ { 3 , z } \odot \sigma ( \mathbf { S } \mathbf { T } \mathbf { - } \mathbf { G } \mathbf { C } ( \mathbf { G } _ { 3 , z } , \mathbf { A } ) ) ]$   
9: $\mathbf { G } _ { e } ^ { \mathrm { C A } }  \frac { 1 } { 2 } ( \mathbf { G } _ { 2 , s } ^ { \mathrm { - } } + \mathbf { G } _ { 3 , t } )$   
10: for each subject and $r \in \{ \mathrm { S A } , \mathrm { C A } \}$ do   
11: $\mathbf { z } _ { e } ^ { r }  \acute { P _ { r } } ( \mathrm { v e c } ( \mathbf { G } _ { e } ^ { r } ) ) ,$   
$\mathcal { Z } _ { e } ^ { r } \gets [ { \bf z } _ { \ell _ { e } } ^ { r } , \ldots , { \bf z } _ { e } ^ { r } ]$   
12: Retain detached copies of $( \pmb { z } _ { e } ^ { r } , \mathbf { G } _ { e } ^ { r } )$ for caching   
13: $\left( \overrightarrow { \mathbf { h } _ { e } ^ { r } } , \overleftarrow { \mathbf { h } _ { \ell _ { e } } ^ { r } } \right) \gets \mathrm { B i G R U } ( \mathcal { Z } _ { e } ^ { r } ; \bullet )$   
14: $\begin{array} { r } { \dot { \mathbf { a } } _ { e } ^ { r } \gets \mathrm { r e s h a p e } _ { T \times N } \left( \sigma \left( D _ { r } \left[ \overrightarrow { \mathbf { h } _ { e } ^ { r } } ; \overleftarrow { \mathbf { h } _ { \ell _ { e } } ^ { r } } \right] \right) \right) } \end{array}$   
15: $\mathbf { G } _ { e } ^ { r }  \mathbf { G } _ { e } ^ { r } + \mathbf { a } _ { e } ^ { r } \odot ( \operatorname { \mathbf { \dot { M } } } _ { e } ^ { r } - \mathbf { G } _ { e } ^ { \tilde { r } } ) , \quad \operatorname { i f } \mathbf { \Xi } _ { e } ^ { + } > 1$   
16: end for   
17: $p \gets \frac { 1 } { 3 } \left( p ^ { \mathrm { o r i g } } + p ^ { \mathrm { S A } } + p ^ { \mathrm { C A } } \right)$   
18: $\mathcal { L } _ { \mathrm { T M C } }  \alpha \mathcal { L } _ { \mathrm { S A } } + ( 1 - \alpha ) ^ { \prime } \mathcal { L } _ { \mathrm { C A } } + \beta \mathcal { L } _ { \mathrm { S A , C A } }$   
19: $\mathcal { L } \gets \mathcal { L } _ { \mathrm { c l a s s } } + \lambda \mathcal { L } _ { \mathrm { T M C } }$   
20: $\Theta  \mathrm { A d a m } ( \Theta , \nabla _ { \Theta } \mathcal { L } )$   
21: Update the corresponding histories using the retained copies   
22: end for   
23: end for   
24: return Θ

All trainable modules are initialized before constructing the Adam optimizer [20]. Training runs for 200 epochs with an initial learning rate of 0.001, which is multiplied by 0.5 every 50 epochs. The learning-rate scheduler is updated at the end of each epoch. The batch size is specified in Section V-E3, and the loss weights are set to $\alpha = 0 . 3 , \beta = 1 . 0 ,$ , and $\lambda = 0 . 5$ During training, the original brain network and its self-aware and cross-sample augmented views are processed by the same classification branch. Their predicted patient probabilities are averaged according to Eq. (14), and the averaged probability is used to compute the classification loss. At test time, we construct augmentation histories for each unseen subject using the last K epoch-end checkpoints. Each checkpoint generates pre-DAU views and descriptors using its augmentation modules and projections. The chronologically ordered descriptors are processed by the final checkpoint’s DAU, using the preceding checkpoint’s pre-DAU views as references in Eq. (7). The final checkpoint’s classifier processes the original and two refined views, and their predicted probabilities are averaged according to Eq. (14). All checkpoints are frozen and used in evaluation mode. For clarity, we show the detailed training process of our proposed MCDA framework in Algorithm 1.

TABLE I  
DEMOGRAPHIC INFORMATION OF STUDIED SUBJECTS FROM THE REST-MDD AND ABIDE DATASETS. THE AGE IS REPORTED AS MEAN  STANDARD DEVIATION. M: MALE; F: FEMALE; MDD: MAJOR DEPRESSIVE DISORDER; ASD: AUTISM SPECTRUM DISORDER; HC: HEALTHY CONTROL.
<table><tr><td>Dataset</td><td>Site</td><td>Category</td><td>Number</td><td>Gender(M/F)</td><td>Age</td></tr><tr><td rowspan="6">REST-MDD</td><td rowspan="2">20</td><td>MDD</td><td>282</td><td>99/183</td><td> $3 8 . 7 4 \pm 1 3 . 7 4$ </td></tr><tr><td>HC</td><td>251</td><td>87/164</td><td> $3 9 . 6 4 \pm 1 5 . 8 7$ </td></tr><tr><td rowspan="2">21</td><td>MDD</td><td>86</td><td>38/48</td><td> $3 4 . 7 1 \pm 1 2 . 6 3$ </td></tr><tr><td>HC</td><td>70</td><td>31/39</td><td> $3 6 . 1 3 \pm 1 2 . 6 4$ </td></tr><tr><td rowspan="2">25</td><td>MDD</td><td>89</td><td>21/68</td><td> $6 5 . 6 0 \pm 6 . 7 5$ </td></tr><tr><td>HC</td><td>63</td><td>29/34</td><td> $6 9 . 6 3 \pm 5 . 8 6$ </td></tr><tr><td rowspan="6">ABIDE</td><td rowspan="2">NYU</td><td>ASD</td><td>79</td><td>68/11</td><td> $1 4 . 5 2 \pm 6 . 9 7$ </td></tr><tr><td>HC</td><td>105</td><td>79/26</td><td> $1 5 . 8 1 \pm 6 . 2 5$ </td></tr><tr><td rowspan="2">UM</td><td>ASD</td><td>55</td><td>46/9</td><td> $1 2 . 7 2 \pm 2 . 4 0$ </td></tr><tr><td>HC</td><td>55</td><td>38/17</td><td> $1 4 . 0 7 \pm 3 . 1 8$ </td></tr><tr><td rowspan="2">USM</td><td>ASD</td><td>58</td><td>58/0</td><td> $2 2 . 6 5 \pm 7 . 7 3$ </td></tr><tr><td>HC</td><td>43</td><td>43/0</td><td> $2 1 . 3 6 \pm 7 . 6 4$ </td></tr></table>

## IV. EXPERIMENT

## A. Materials and Data Pre-processing

We evaluate the proposed MCDA framework on two rs-fMRI datasets: 1) REST-meta-MDD Consortium (REST-MDD<sup>1</sup>) [21] and 2) Autism Brain Imaging Data Exchange (ABIDE<sup>2</sup>) [22]. For each dataset, we use the top three largest sites in the experiments. The demographic information of these subjects from REST-MDD and ABIDE is provided in Table I.

1) REST-MDD. For the REST-MDD dataset, the three largest sites are selected for analysis: Site 20, Site 21, and Site 25, which respectively include 282 patients with major depressive disorder (MDD) and 251 healthy controls (HCs); 86 MDD patients and 70 HCs; and 89 MDD patients and 63 HCs, with acquisition details provided in the original dataset publication [21]. All rs-fMRI data from REST-MDD were preprocessed using the DPARSF pipeline [23]. The first 10 volumes were discarded, followed by standard preprocessing procedures including head motion correction, spatial smoothing, normalization, band-pass filtering (0.01–0.10,Hz), nuisance signal regression, and spatial standardization.

2) ABIDE. In ABIDE, we use three sites, including Site NYU with 79 ASD and 105 HC subjects, Site UM with 55 ASD and 55 HC subjects, and Site USM with 58 ASD and 43 HC subjects, with acquisition details provided in the original dataset publication [22]. The rs-fMRI data were preprocessed based on the DPARSF pipeline [23].

For both datasets, the brain was parcellated into 116 ROIs using the original Automated Anatomical Labeling (AAL) atlas [24], and the mean BOLD time series was extracted from each ROI. The repetition time (TR) was 2000 ms at all six sites. For each subject, the final model input consisted of 200 time points from 116 ROIs, yielding $\mathbf { G } \in \mathbf { R } ^ { 2 0 0 \times 1 1 6 }$

## B. Experiment Setting

In our experiments, we perform five-fold cross-validation independently at each site using random subject-level splits. In each fold, one subset is held out for testing, while the remaining data are split into training and validation sets at an 80:20 ratio for model fitting and selection, respectively. All competing methods are retrained using identical data partitions. Model selection is based solely on validation performance. Results are reported as the mean and standard deviation across the five test folds. The evaluation metrics include accuracy (ACC), area under the receiver operating characteristic curve (AUC), precision (PRE), and recall (REC).

We compare MCDA with the following six SOTA methods for fMRI analysis. Specifically, two methods (i.e., BrainGNN<sup>3</sup> and LCAF<sup>4</sup>) are advanced rs-fMRI analysis approaches, followed by augmentation-based approaches (i.e., TSDA<sup>5</sup> and GCDA) that enrich brain network representations. The remaining two methods (i.e., UCGL<sup>6</sup> and MCS-GL) adopt graph contrastive learning from diferent views for classification.

1) BrainGNN [4]: This is a graph neural network designed for analyzing fMRI and detecting neurological biomarkers. With the brain network/graph as input, BrainGNN uses two node-level graph convolutional layers to learn the node representations, capturing topological and functional patterns from fMRI data.

2) LCAF [25]: Kang et al. propose a unified FC-based framework integrating feature selection, feature extraction, and interpretation to reduce error propagation across stages. An adaptive attention network selects subject-specific, diseaserelated connections, while a functional network relational encoder models inter-network relationships to capture global topology without predefined edges.

3) TSDA [10]: In our implementation, TSDA generates two augmented sequences by extracting the first and last 90% of each ROI’s mean BOLD time series. Subsequent processing follows the same pipeline as MCDA for brain disorder classification.

4) GCDA [11]: GCDA combines graph difusion augmentation with two shared-parameter graph isomorphism networks for contrastive pretraining, followed by supervised fine-tuning. We adapt this two-stage procedure to our five-fold protocol: pretraining uses the training subjects without labels, while finetuning uses the same subjects with their diagnostic labels.

5) UCGL [6]: UCGL employs dual-stage fMRI augmentation and two parallel ST-GCNs for contrastive pretraining, followed by supervised fine-tuning. We replace the fixed 60%/20%/20% allocation with our fold-wise partitions, using the training subjects for both stages and their labels only during fine-tuning. For GCDA and UCGL methods, validation subjects are used only for model selection, and test subjects are reserved for final evaluation.

TABLE II  
RESULTS (%) OF SEVEN METHODS IN MDD VS. HC CLASSIFICATION WITH MEAN STANDARD DEVIATION. BEST RESULTS ARE SHOWN IN BOLD. HERE, \* DENOTES A STATISTICALLY SIGNIFICANT (P<0.05) DIFFERENCE BETWEEN MCDA AND A COMPETING METHOD IN TERMS OF ACCURACY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Site 20</td><td colspan="4"> $\mathrm { S i t e } \ 2 1$ </td><td colspan="4">Site 25</td></tr><tr><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>BrainGNN</td><td> $6 3 . 6 { \scriptstyle \pm 1 . 7 } ^ { * }$ </td><td> $6 5 . 5 { \scriptstyle \pm 3 . 3 }$ </td><td> $5 7 . 6 _ { \pm 5 . 9 }$ </td><td> ${ \bf 6 9 . 8 _ { \pm 1 4 . 2 } }$ </td><td> $6 5 . 2 _ { \pm 6 . 2 }$ </td><td> $6 7 . 0 { \scriptstyle \pm 2 . 9 }$ </td><td> $5 8 . 8 { \scriptstyle \pm 7 . 2 }$ </td><td> ${ 7 4 . 2 \pm 7 . 5 }$ </td><td> $6 6 . 2 _ { \pm 8 . 3 }$ </td><td> $6 4 . 8 { \scriptstyle \pm 7 . 6 }$ </td><td> $5 9 . 3 _ { \pm 7 . 2 }$ </td><td> $6 8 . 0 { \scriptstyle \pm 8 . 2 }$ </td></tr><tr><td>LCAF</td><td> $6 1 . 6 { \scriptstyle \pm 1 . 9 } ^ { * }$ </td><td> $6 2 . 8 _ { \pm 4 . 0 }$ </td><td> $5 4 . 3 _ { \pm 2 . 7 }$ </td><td> $6 8 . 9 { \scriptstyle \pm 7 . 3 }$ </td><td> ${ 6 1 . 1 } { \scriptstyle \pm 2 . 9 } ^ { * }$ </td><td> $5 5 . 5 { \scriptstyle \pm 6 . 6 }$ </td><td> $5 5 . 3 _ { \pm 1 0 . 1 }$ </td><td> $5 7 . 3 { \scriptstyle \pm 1 2 . 7 }$ </td><td> $6 9 . 0 { \scriptstyle \pm 7 . 3 } $ </td><td> $6 7 . 6 { \scriptstyle \pm 3 . 6 }$ </td><td> $6 3 . 0 { \scriptstyle \pm 6 . 8 }$ </td><td> $7 0 . 0 { \scriptstyle \pm 6 . 1 }$ </td></tr><tr><td>TSDA</td><td> $6 2 . 1 \pm 2 . 4 ^ { \ }$ </td><td> $6 0 . 9 { \scriptstyle \pm 4 . 6 }$ </td><td> $5 6 . 1 \pm 3 . 9$ </td><td> $6 2 . 9 { \scriptstyle \pm 7 . 8 }$ </td><td> $5 9 . 0 { \scriptstyle \pm 2 . 7 } ^ { * }$ </td><td> $5 5 . 0 { \scriptstyle \pm 5 . 8 }$ </td><td> $5 2 . 6 { \scriptstyle \pm 7 . 9 }$ </td><td> $6 2 . 9 { \scriptstyle \pm 1 7 . 0 }$ </td><td> $6 2 . 7 _ { \pm 5 . 7 }$ </td><td> $5 4 . 2 _ { \pm 4 . 1 }$ </td><td> $5 2 . 1 _ { \pm 1 . 1 }$ </td><td> $6 3 . 8 _ { \pm 1 3 . 9 }$ </td></tr><tr><td>GCDA</td><td> $6 3 . 5 { \scriptstyle \pm 1 . 0 } ^ { * }$ </td><td> $6 2 . 2 _ { \pm 4 . 0 }$ </td><td> $5 9 . 4 _ { \pm 4 . 2 }$ </td><td> $6 4 . 5 _ { \pm 1 4 . 3 }$ </td><td> $6 7 . 3 _ { \pm 4 . 1 }$ </td><td> $6 5 . 2 _ { \pm 6 . 5 }$ </td><td> $6 0 . 0 { \scriptstyle \pm 1 0 . 5 }$ </td><td> $6 9 . 4 { \scriptstyle \pm 7 . 8 }$ </td><td> $6 9 . 3 _ { \pm 3 . 3 }$ </td><td> $6 8 . 8 { \scriptstyle \pm 6 . 3 }$ </td><td> ${ \bf 7 0 . 7 \pm 7 . 6 }$ </td><td> $7 2 . 6 { \scriptstyle \pm 9 . 6 }$ </td></tr><tr><td>UCGL</td><td> $6 3 . 7 _ { \pm 2 . 7 _ { . } } \ast$ </td><td> $6 0 . 7 \pm 2 . 7$ </td><td> $5 7 . 9 { \scriptstyle \pm 2 . 9 }$ </td><td> $6 6 . 2 _ { \pm 1 1 . 1 }$ </td><td> $6 2 . 8 { \scriptstyle \pm 3 . 3 } ^ { * }$ </td><td> $6 1 . 2 { \scriptstyle \pm 5 . 7 }$ </td><td> $5 2 . 0 { \scriptstyle \pm 8 . 3 }$ </td><td> $6 3 . 4 _ { \pm 1 3 . 1 }$ </td><td> ${ 6 3 . 3 } { \pm 1 . 9 } ^ { * }$ </td><td> $5 8 . 0 { \scriptstyle \pm 4 . 4 }$ </td><td> $5 7 . 1 _ { \pm 9 . 8 }$ </td><td> $6 7 . 2 _ { \pm 1 2 . 3 }$ </td></tr><tr><td>MCS-GL</td><td> $6 4 . 3 { \scriptstyle \pm 3 . 7 } ^ { * }$ </td><td> $6 5 . 3 _ { \pm 4 . 0 }$ </td><td> $6 1 . 1 _ { \pm 6 . 1 }$ </td><td> $6 3 . 9 _ { \pm 1 4 . 1 }$ </td><td> $6 9 . 0 { \scriptstyle \pm 5 . 4 }$ </td><td> $6 8 . 5 { \scriptstyle \pm 6 . 3 }$ </td><td> $6 2 . 5 { \scriptstyle \pm 1 4 . 2 }$ </td><td> $6 9 . 8 { \scriptstyle \pm 9 . 8 }$ </td><td> $7 0 . 6 _ { \pm 3 . 4 } ^ { \ast }$ </td><td> $6 9 . 5 { \scriptstyle \pm 5 . 6 }$ </td><td> $6 9 . 2 _ { \pm 5 . 7 }$ </td><td> $7 2 . 9 _ { \pm 1 1 . 4 }$ </td></tr><tr><td>MCDA</td><td> ${ \bf 7 0 . 3 _ { \pm 2 . 9 } }$ </td><td> $7 4 . 5 _ { \pm 2 . 8 }$ </td><td> $7 2 . 3 _ { \pm 4 . 0 }$ </td><td> $6 5 . 6 { \scriptstyle \pm 8 . 4 }$ </td><td> ${ \bf 7 1 . 8 _ { \pm 6 . 4 } }$ </td><td> ${ 7 0 . 2 \pm 5 . 3 }$ </td><td> ${ \bf 6 3 . 9 { \scriptstyle \pm 4 . 8 } }$ </td><td> $6 9 . 5 { \scriptstyle \pm 8 . 8 }$ </td><td> $7 3 . 4 _ { \pm 3 . 4 }$ </td><td> ${ 7 3 . 8 \pm 2 . 5 }$ </td><td> $6 6 . 6 { \scriptstyle \pm 6 . 5 }$ </td><td> ${ \bf 7 9 . 4 \pm 9 . 8 }$ </td></tr></table>

TABLE III

RESULTS (%) OF SEVEN METHODS IN ASD VS. HC CLASSIFICATION WITH MEAN STANDARD DEVIATION. BEST RESULTS ARE SHOWN IN BOLD. HERE, \* DENOTES A STATISTICALLY SIGNIFICANT (P<0.05) DIFFERENCE BETWEEN MCDA AND A COMPETING METHOD IN TERMS OF ACCURACY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Site NYU</td><td colspan="4">Site UM</td><td colspan="4">Site USM</td></tr><tr><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>BrainGNN</td><td> $7 3 . 6 { \scriptstyle \pm 7 . 8 } ^ { * }$ </td><td> $7 0 . 3 { \scriptstyle \pm 5 . 9 }$ </td><td> $7 0 . 3 { \scriptstyle \pm 5 . 8 }$ </td><td> $4 9 . 9 { \scriptstyle \pm 7 . 5 }$ </td><td> $7 0 . 9 { \scriptstyle \pm 6 . 1 }$ </td><td> $7 0 . 4 \pm 5 . 6$ </td><td> $8 1 . 0 { \scriptstyle \pm 1 4 . 9 }$ </td><td> $5 1 . 3 { \scriptstyle \pm 1 4 . 2 }$ </td><td> $7 5 . 2 { \scriptstyle \pm 9 . 1 } ^ { * }$ </td><td> $7 1 . 1 { \scriptstyle \pm 1 0 . 2 }$ </td><td> $6 9 . 2 _ { \pm 1 3 . 9 }$ </td><td> $8 4 . 7 _ { \pm 1 1 . 5 }$ </td></tr><tr><td>LCAF</td><td> $7 1 . 7 _ { \pm 4 . 6 } ^ { } { } ^ { * }$ </td><td> $7 2 . 2 { \scriptstyle \pm 4 . 8 }$ </td><td> $7 7 . 6 { \scriptstyle \pm 9 . 2 }$ </td><td> $6 4 . 9 _ { \pm 1 2 . 4 }$ </td><td> $7 1 . 2 { \scriptstyle \pm 5 . 2 }$ </td><td> $7 3 . 0 { \scriptstyle \pm 5 . 7 }$ </td><td> $7 2 . 2 { \scriptstyle \pm 1 3 . 4 }$ </td><td> $6 1 . 8 { \scriptstyle \pm 8 . 9 }$ </td><td> $7 1 . 5 { \scriptstyle \pm 5 . 1 } ^ { * }$ </td><td> $7 3 . 9 { \scriptstyle \pm 1 1 . 2 }$ </td><td> $7 0 . 6 _ { \pm 1 5 . 9 }$ </td><td> $7 3 . 5 { \scriptstyle \pm 1 2 . 4 }$ </td></tr><tr><td>TSDA</td><td> $6 8 . 3 { \scriptstyle \pm 7 . 0 } ^ { \ast }$ </td><td> $6 9 . 3 { \scriptstyle \pm 7 . 3 }$ </td><td> $7 3 . 1 _ { \pm 9 . 8 }$ </td><td> $6 6 . 6 { \scriptstyle \pm 8 . 5 }$ </td><td> $6 9 . 5 _ { \pm 4 . 2 } ^ { \ }$ </td><td> $7 0 . 8 { \scriptstyle \pm 9 . 9 }$ </td><td> $7 2 . 5 { \scriptstyle \pm 1 7 . 7 }$ </td><td> $6 8 . 8 _ { \pm 1 5 . 9 }$ </td><td> $6 2 . 4 { \scriptstyle \pm 4 . 5 } ^ { }$ </td><td> $5 7 . 4 _ { \pm 5 . 4 }$ </td><td> $6 6 . 1 \pm 6 . 4$ </td><td> $6 9 . 2 _ { \pm 2 . 2 }$ </td></tr><tr><td>GCDA</td><td> $7 6 . 4 \pm 2 . 4$ </td><td> $6 5 . 7 _ { \pm 6 . 0 }$ </td><td> $7 6 . 2 _ { \pm 8 . 3 }$ </td><td> $6 6 . 7 _ { \pm 1 5 . 6 }$ </td><td> $7 1 . 2 _ { \pm 1 3 . 4 }$ </td><td> $7 4 . 5 _ { \pm 1 4 . 8 }$ </td><td> $6 5 . 3 _ { \pm 1 2 . 1 }$ </td><td> $7 7 . 9 _ { \pm 1 6 . 2 }$ </td><td> $7 7 . 8 _ { \pm 8 . 6 }$ </td><td> $^ { 7 3 . 9 } \pm 1 4 . 1$ </td><td> ${ \bf 8 1 . 7 \pm 6 . 9 }$ </td><td> $7 8 . 6 _ { \pm 1 5 . 1 }$ </td></tr><tr><td>UCGL</td><td> $^ { 7 3 . 0 } { \pm 6 . 6 } ^ { * }$ </td><td> $7 0 . 7 _ { \pm 8 . 2 }$ </td><td> $\mathbf { 7 8 . 9 _ { \pm 8 . 2 } }$ </td><td> $5 9 . 6 _ { \pm 1 7 . 5 }$ </td><td> $7 4 . 5 _ { \pm 5 . 2 }$ </td><td> $7 1 . 8 { \scriptstyle \pm 7 . 7 }$ </td><td> $8 1 . 1 _ { \pm 6 . 5 }$ </td><td> $5 8 . 2 _ { \pm 1 8 . 7 }$ </td><td> $7 7 . 2 _ { \pm 7 . 5 }$ </td><td> $7 8 . 6 _ { \pm 9 . 0 }$ </td><td> $7 8 . 0 { \scriptstyle \pm 9 . 4 }$ </td><td> $7 8 . 1 \pm 9 . 4$ </td></tr><tr><td>MCS-GL</td><td> $7 0 . 8 _ { \pm 8 . 3 }$ </td><td> $7 3 . 1 _ { \pm 6 . 2 }$ </td><td> $6 8 . 1 _ { \pm 6 . 4 }$ </td><td> $6 2 . 9 _ { \pm 1 2 . 2 }$ </td><td> $7 3 . 9 _ { \pm 3 . 2 }$ </td><td> $7 6 . 7 _ { \pm 1 0 . 6 }$ </td><td> $7 6 . 5 _ { \pm 8 . 9 }$ </td><td> $7 7 . 3 _ { \pm 1 4 . 7 }$ </td><td> $7 0 . 6 _ { \pm 6 . 5 } ^ { * }$ </td><td> $7 2 . 6 _ { \pm 6 . 6 }$ </td><td> $7 1 . 1 _ { \pm 8 . 6 }$ </td><td> $7 9 . 9 { \scriptstyle \pm 7 . 8 }$ </td></tr><tr><td>MCDA</td><td> ${ \bf 8 1 . 1 } _ { \pm 2 . 1 }$ </td><td> ${ \bf 8 2 . 9 } _ { \pm 2 . 4 }$ </td><td> $7 6 . 1 \pm 9 . 9$ </td><td> ${ \bf 6 8 . 1 _ { \pm 8 . 0 } }$ </td><td> $7 5 . 0 _ { \pm 2 . 6 }$ </td><td> $7 8 . 4 _ { \pm 7 . 2 }$ </td><td> ${ \bf 8 7 . 4 } _ { \pm 3 . 6 }$ </td><td> $6 1 . 2 _ { \pm 9 . 6 }$ </td><td> ${ \bf 8 0 . 2 \pm 7 . 1 }$ </td><td> ${ \bf 8 2 . 1 \pm 5 . 8 }$ </td><td> $7 7 . 9 _ { \pm 8 . 9 }$ </td><td> ${ \bf 8 6 . 2 \pm 8 . 4 }$ </td></tr></table>

6) MCS-GL [8]: The MCS-GL framework enhances functional brain network-based diagnosis by expanding samples via Personalized PageRank (PPR) and incorporating local-global information as multi-view inputs for contrastive learning. During training, it extracts informative fMRI representations from unlabeled data to enable efective feature learning without supervision. In the final stage, model parameters are updated using a supervised cross-entropy loss.

Second, compared with the two augmentation-based methods, TSDA and GCDA, MCDA achieves higher mean ACC and AUC at each evaluated site. On Site NYU of ABIDE, MCDA exceeds TSDA by 12.8 percentage points in ACC and 13.6 percentage points in AUC. The performance gap suggests that simple window-slicing augmentation may be insuficient to capture the complex functional patterns needed for ASD diagnosis. In contrast, MCDA’s dynamic topology augmentation maintains the brain’s complete structures, allowing its contrastive learning to capture both localized dysfunctions and global network characteristics.

## C. Classification Results

The quantitative results of MCDA and the six competing methods for MDD vs. HC classification on the REST-MDD dataset and ASD vs. HC classification on the ABIDE dataset are reported in Table II and Table III, respectively, under within-site five-fold cross-validation. We perform two-sided paired t-tests on the ACC values of MCDA and each competing method obtained from the same five outer test folds $( n = 5$ pairs; degrees of freedom = 4). “\*” denotes a nominal $p \ < \ 0 . 0 5$ for the diference in ACC between MCDA and the corresponding method. From these tables, we have the following observations.

Additionally, MCDA achieves higher mean ACC and AUC than the graph contrastive learning methods UCGL and MCS-GL at each evaluated site. Specifically, for MDD diagnosis on Site 25, MCDA exceeds UCGL by 10.1 percentage points in ACC and 15.8 percentage points in AUC. These results support the efectiveness of dynamic augmentation and topologyguided multi-view contrastive learning for within-site brain disorder classification.

## D. Ablation Study

First, compared to BrainGNN and LCAF, MCDA shows improved discriminative power. For instance, MCDA achieves ACC improvements of 6.7 and 6.6 percentage points over BrainGNN on Site 20 and Site 21 of REST-MDD, respectively. The possible reason is that topology-guided multiview contrastive learning efectively aligns multiple views of brain connectivity by preserving topology semantics, enabling invariant and discriminative representation learning across views.

To assess the contribution of key components in our MCDA framework, we compare it with three ablated variants, namely MCDA-I, MCDA-II, and MCDA-III. Specifically, MCDA-I removes the DAU module and directly uses the generated selfaware and cross-sample augmentation views without historyguided refinement. In MCDA-II, to evaluate the impact of the contrastive loss $\mathcal { L } _ { \mathrm { S A , C A } }$ , we set the weighting parameter β in Eq. (13) to 0. Similarly, MCDA-III sets $\lambda ~ = ~ 0$ in Eq. (15), while retaining both augmentation branches, the DAU module, and the classification loss computed from the averaged prediction probabilities of the three views. The same threeview probability averaging is used during inference, isolating the contribution of the contrastive objective. These variants are trained on the two datasets with the same experiment setting, with results reported in Table IV for MDD vs. HC classification and Table V for ASD vs. HC classification.

TABLE IV  
RESULTS (%) OF MCDA AND ITS THREE VARIANTS IN MDD VS. HC CLASSIFICATION. BEST RESULTS ARE SHOWN IN BOLD. HERE, \* DENOTES A STATISTICALLY SIGNIFICANT (P<0.05) DIFFERENCE BETWEEN MCDA AND A VARIANT IN TERMS OF ACCURACY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Site 20</td><td colspan="4"> $\mathrm { S i t e } \ 2 1$ </td><td colspan="4">Site 25</td></tr><tr><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>MCDA-I</td><td> $6 1 . 5 { \scriptstyle \pm 2 . 6 } ^ { * }$ </td><td> $5 0 . 9 { \scriptstyle \pm 3 . 2 }$ </td><td> $5 3 . 8 { \scriptstyle \pm 2 . 7 }$ </td><td> $6 3 . 8 { \scriptstyle \pm 8 . 4 }$ </td><td> $6 3 . 2 \pm 5 . 4 ^ { ^ { \ast } }$ </td><td> $6 5 . 4 _ { \pm 2 . 5 }$ </td><td> $5 4 . 0 { \scriptstyle \pm 8 . 5 }$ </td><td> $6 6 . 6 { \scriptstyle \pm 1 2 . 3 }$ </td><td> ${ 6 } 3 . 3 { \pm } 1 . 9 ^ { ^ { \ast } }$ </td><td> $5 8 . 8 { \scriptstyle \pm 4 . 6 }$ </td><td> $5 2 . 7 _ { \pm 5 . 8 }$ </td><td> $6 3 . 8 { \scriptstyle \pm 1 1 . 8 }$ </td></tr><tr><td>MCDA-II</td><td> $6 0 . 8 _ { \pm 3 . 6 } ^ { * }$ </td><td>63  $. 5 { \scriptstyle \pm 7 . 3 }$ </td><td> $5 8 . 9 { \scriptstyle \pm 4 . 5 }$ </td><td> $5 3 . 8 _ { \pm 1 0 . 7 }$ </td><td>68.0±4.9</td><td> $6 9 . 3 _ { \pm 6 . 5 }$ </td><td> $5 2 . 4 { \scriptstyle \pm 1 1 . 6 }$ </td><td> $6 0 . 9 { \scriptstyle \pm 1 9 . 0 }$ </td><td> $6 8 . 0 { \scriptstyle \pm 8 . 1 }$ </td><td> $6 4 . 1 _ { \pm 1 0 . 7 }$ </td><td> $5 9 . 5 _ { \pm 1 8 . 5 }$ </td><td> $5 4 . 1 { \pm } 7 . 8 $ </td></tr><tr><td>MCDA-III</td><td> $5 8 . 2 { \scriptstyle \pm 5 . 6 } ^ { * }$ </td><td> $5 5 . 6 { \scriptstyle \pm 6 . 5 }$ </td><td> $5 4 . 3 _ { \pm 4 . 9 }$ </td><td> $4 9 . 3 _ { \pm 8 . 9 }$ </td><td> $6 5 . 4 _ { \pm 4 . 9 } ^ { \ }$ </td><td> $5 3 . 7 _ { \pm 1 2 . 5 }$ </td><td> $5 7 . 5 { \scriptstyle \pm 1 0 . 4 }$ </td><td> $5 8 . 7 _ { \pm 1 3 . 1 }$ </td><td> $6 4 . 4 \pm 4 . 6 ^ { \ast }$ </td><td> $6 7 . 5 { \scriptstyle \pm 5 . 4 }$ </td><td> ${ \bf 6 8 . 5 } _ { \pm 2 . 6 }$ </td><td> $6 6 . 3 _ { \pm 3 . 2 }$ </td></tr><tr><td>MCDA</td><td> ${ \bf 7 0 . 3 _ { \pm 2 . 9 } }$ </td><td> $7 4 . 5 _ { \pm 2 . 8 }$ </td><td> $\mathbf { 7 2 . 3 _ { \pm 4 . 0 } }$ </td><td> ${ \bf 6 5 . 6 { \bf _ { \pm 8 . 4 } } }$ </td><td> ${ \bf 7 1 . 8 _ { \pm 6 . 4 } }$ </td><td> ${ 7 0 . 2 \pm 5 . 3 }$ </td><td> ${ \bf 6 3 . 9 { \scriptstyle \pm 4 . 8 } }$ </td><td> ${ \bf 6 9 . 5 \pm 8 . 8 }$ </td><td> $7 3 . 4 _ { \pm 3 . 4 }$ </td><td> ${ 7 3 . 8 \pm 2 . 5 }$ </td><td> $6 6 . 6 { \scriptstyle \pm 6 . 5 }$ </td><td> ${ \bf 7 9 . 4 \pm 9 . 8 }$ </td></tr></table>

TABLE V

RESULTS (%) OF MCDA AND ITS THREE VARIANTS IN ASD VS. HC CLASSIFICATION. BEST RESULTS ARE SHOWN IN BOLD. HERE, \* DENOTES A STATISTICALLY SIGNIFICANT (P<0.05) DIFFERENCE BETWEEN MCDA AND A VARIANT IN TERMS OF ACCURACY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Site NYU</td><td colspan="4">Site UM</td><td colspan="4">Site USM</td></tr><tr><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>MCDA-I</td><td> $7 6 . 0 { \scriptstyle \pm 7 . 1 }$ </td><td> $7 7 . 5 { \scriptstyle \pm 8 . 3 }$ </td><td> $7 0 . 2 { \scriptstyle \pm 1 1 . 9 }$ </td><td> $5 8 . 0 { \scriptstyle \pm 1 0 . 4 }$ </td><td> $7 0 . 9 _ { \pm 7 . 6 }$ </td><td> $7 3 . 0 { \scriptstyle \pm 8 . 3 }$ </td><td> $7 8 . 0 { \scriptstyle \pm 1 8 . 0 }$ </td><td> $5 9 . 4 _ { \pm 2 6 . 2 }$ </td><td> $^ { 7 4 . 5 } { \pm 2 . 6 }$ </td><td> $7 6 . 1 \pm 4 . 1$ </td><td> $7 6 . 0 { \scriptstyle \pm 6 . 9 }$ </td><td> $7 4 . 7 _ { \pm 6 . 8 }$ </td></tr><tr><td>MCDA-II</td><td> $^ { 7 3 . 6 \pm 6 . 8 ^ { \ast } }$ </td><td> $7 6 . 0 { \scriptstyle \pm 7 . 4 }$ </td><td> ${ \bf 8 1 . 7 \pm 7 . 6 }$ </td><td> $5 7 . 6 _ { \pm 1 4 . 2 }$ </td><td> $7 2 . 7 _ { \pm 4 . 5 }$ </td><td> $7 5 . 7 \pm 7 . 2$ </td><td> $8 1 . 9 _ { \pm 6 . 4 }$ </td><td> $6 1 . 3 { \scriptstyle \pm 1 6 . 3 }$ </td><td> $^ { 7 0 . 2 \pm 5 . 3 ^ { * } }$ </td><td> $7 0 . 6 { \scriptstyle \pm 7 . 9 }$ </td><td> $6 4 . 9 { \scriptstyle \pm 9 . 9 }$ </td><td> $7 8 . 4 { \scriptstyle \pm 9 . 1 }$ </td></tr><tr><td>MCDA-III</td><td> $6 9 . 0 { \scriptstyle \pm 9 . 0 } ^ { * }$ </td><td> $6 8 . 0 { \scriptstyle \pm 6 . 5 }$ </td><td> $7 7 . 0 _ { \pm 1 6 . 3 }$ </td><td> $6 1 . 5 { \scriptstyle \pm 1 4 . 9 }$ </td><td> $6 5 . 4 \pm 3 . 3 ^ { * }$ </td><td> $6 5 . 4 \pm 1 . 7$ </td><td> $6 5 . 7 { \scriptstyle \pm 7 . 3 }$ </td><td> ${ \bf 6 9 . 8 _ { \pm 2 . 1 } }$ </td><td> $6 7 . 4 _ { \pm 5 . 2 } ^ { }$ </td><td> $6 6 . 4 \pm 8 . 0$ </td><td> $6 6 . 2 _ { \pm 6 . 3 }$ </td><td> $7 5 . 3 _ { \pm 1 0 . 6 }$ </td></tr><tr><td>MCDA</td><td> ${ \bf 8 1 . 1 } _ { \pm 2 . 1 }$ </td><td> ${ \bf 8 2 . 9 } _ { \pm 2 . 4 }$ </td><td> $7 6 . 1 \pm 9 . 9$ </td><td> ${ \bf 6 8 . 1 _ { \pm 8 . 0 } }$ </td><td> $7 5 . 0 _ { \pm 2 . 6 }$ </td><td> ${ 7 8 . 4 \pm 7 . 2 }$ </td><td> ${ \bf 8 7 . 4 } _ { \pm 3 . 6 }$ </td><td> $6 1 . 2 { \scriptstyle \pm 9 . 6 }$ </td><td> ${ \bf 8 0 . 2 \pm 7 . 1 }$ </td><td> ${ \bf 8 2 . 1 \pm 5 . 8 }$ </td><td> $7 7 . 9 _ { \pm 8 . 9 }$ </td><td> ${ \bf 8 6 . 2 \pm 8 . 4 }$ </td></tr></table>

![](images/9f50a4a309f5670b9c38b19653c5ba9c8c6a5dc4c6bb8f14f69298047150fdbd.jpg)  
Fig. 4. Visualization of the top 10 ROI-pair associations identified by MCDA in two classification tasks: (a) MDD vs. HC classification (Site 20) on REST-MDD, and (b) ASD vs. HC classification (Site NYU) on ABIDE.

The complete MCDA model achieves the highest mean ACC and AUC in all six within-site evaluations. Its improvement over MCDA-I supports the contribution of the DAU module. The higher mean ACC and AUC compared with MCDA-II further support the benefit of including the $\mathcal { L } _ { \mathrm { S A , C A } }$ diversity constraint. Meanwhile, the comparison with the classificationonly baseline MCDA-III indicates the additional value of multi-view contrastive regularization. Overall, these ablations support the contributions of the evaluated components to within-site classification performance.

## V. DISCUSSION

## A. Exploratory Analysis of Brain Region-Level Associations

We analyze brain region-level associations in the selfaware and cross-sample augmentation features of held-out test subjects from Site 20 of REST-MDD and Site NYU of ABIDE separately. Features are extracted using frozen models trained and selected without the corresponding test subjects.

For each subject and view, ROI temporal-by-channel features are centered over time, and linear centered kernel alignment (CKA) [26] is used to construct a 116×116 association matrix. For each ROI pair, the two view-specific scores form a twodimensional response in a multivariate linear model adjusted for prespecified demographic and head-motion covariates. Diagnostic-group efects are evaluated using Pillai’s trace and Freedman–Lane residual permutations [27]. Covariate-only model residuals are permuted jointly across subjects for both views and added back to the fitted values before refitting, using identical permutations for all ROI pairs. Joint p-values are corrected across the $^ { 6 , 6 7 0 }$ unique ROI pairs within each site using the Benjamini–Yekutieli procedure [28]. Among ROI pairs with BY-adjusted p-values below 0.05, up to ten pairs with the smallest joint p-values are visualized using BrainNet Viewer [29], with line thickness representing Pillai’s trace. The MDD–HC and ASD–HC patterns are shown in Fig. 4(a) and (b), respectively.

As shown in Fig. 4(a), the visualized connections for the MDD–HC comparison involve the bilateral insula (INS.L and INS.R), right middle frontal gyrus (MFG.R), and right supplementary motor area (SMA.R). Previous studies have associated insula network connectivity with depressive symptoms in MDD [30] and reported altered structural–functional coupling of the right SMA [31]. For the ASD–HC comparison, the connections visualized in Fig. 4(b) involve the left amygdala (AMYG.L), right superior frontal gyrus, orbital part (ORBsup.R), and right precentral gyrus (PreCG.R). Previous rs-fMRI research has reported atypical amygdala connectivity in children and adolescents with ASD [32]. Overall, these observations provide exploratory anatomical context for the visualized brain region associations.

![](images/45db4188e7b09ce3208fb0de2239bf7e3391a66b7d24379559b36f098f9e0e07.jpg)  
Fig. 5. Visualization of dynamic augmented brain networks by DAU module at every 20 epochs in (a) self-aware and (b) cross-sample augmentation.

## B. Visualization of Dynamic Graph Augmentation

To elucidate the process of two dynamic graph topology augmentations, we visualize the generated brain network augmentation. Specifically, taking a dynamic self-aware augmented brain network of the first 30 ROIs from a subject in Site 20 of the REST-MDD as an example, along with the original graph at E=0, are visualized at a step interval of 20, with results presented in Fig. 5(a). Similarly, we show 11 graphs generated in a dynamic cross-sample augmented brain network with the same subject in Fig. 5(b).

As illustrated in Fig. 5, the two augmented brain networks maintain strong structural consistency during the dynamic augmentation process. This demonstrates that the backward propagation in DAU module efectively constrains and adjusts the consistency of semantic representations, ensuring meaningful topological transformations while preserving critical functional connectivity patterns. Meanwhile, from a topological structure perspective, the self-aware augmented networks maintain high similarity with the original networks. In contrast, the cross-sample augmented networks demonstrate more substantial modifications, particularly in the connections between ROIs located in the upper-right and lower-left regions.

## C. Influence of Dynamic Augmentation Update Structure

To investigate the efect of diferent dynamic update structures on the performance of our MCDA framework, the proposed DAU module is replaced with GRU [33] and LSTM [34], referred to as MCDA-GRU and MCDA-LSTM, respectively. Specifically, the MCDA-GRU employs a standard 3-layer GRU architecture with forward propagation only, while the MCDA-LSTM utilizes a 3-layer LSTM structure, both maintaining consistent configurations with our DAU module to ensure fair comparison. We conduct experiments on Site 20 of REST-MDD and report results in Fig. 6. As can be seen from this figure, our MCDA framework consistently outperforms MCDA-GRU and MCDA-LSTM across most evaluation metrics, highlighting the efectiveness of the bidirectional dynamic update mechanism for promoting the propagation of discriminative information. The ability to maintain semantic consistency in the topology augmentation during continuous updates plays a key role.

![](images/c9ff627f4dedd58bbd5e1809b7e776651655eabc97d28e669010de944e19b58d.jpg)  
Fig. 6. Results of MCDA-GRU, MCDA-LSTM and MCDA on Site 20 from the REST-MDD dataset.

TABLE VI  
RESULTS (%) OF DIFFERENT AUGMENTATION ENCODERS ON SITE 20 FROM THE REST-MDD DATASET.
<table><tr><td>Method</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>MCDA-None</td><td> $5 9 . 8 _ { \pm 4 . 3 }$ </td><td> $6 0 . 0 _ { \pm 6 . 8 }$ </td><td> $5 2 . 8 _ { \pm 8 . 3 }$ </td><td> $5 3 . 7 _ { \pm 7 . 3 }$ </td></tr><tr><td>MCDA-Ra-GConv</td><td> $6 4 . 2 _ { \pm 3 . 7 }$ </td><td> $6 6 . 2 _ { \pm 4 . 2 }$ </td><td> $5 7 . 1 _ { \pm 6 . 2 }$ </td><td> $\mathbf { 6 7 . 9 _ { \pm 5 . 9 } }$ </td></tr><tr><td>MCDA-2DCNN</td><td> $6 2 . 6 _ { \pm 3 . 0 }$ </td><td> $6 5 . 0 _ { \pm 4 . 6 }$ </td><td> $5 8 . 3 _ { \pm 4 . 4 }$ </td><td> $6 6 . 0 _ { \pm 3 . 9 }$ </td></tr><tr><td>MCDA</td><td> ${ \bf 7 0 . 3 _ { \pm 2 . 9 } }$ </td><td> $7 4 . 5 _ { \pm 2 . 8 }$ </td><td> ${ \bf 7 2 . 3 _ { \pm 4 . 0 } }$ </td><td> $6 5 . 6 _ { \pm 8 . 4 }$ </td></tr></table>

## D. Influence of Augmentation Encoder

To evaluate the contribution of the ST-GC encoder in capturing spatial and temporal brain connectivity, we conduct ablation studies by replacing it with alternative modules. Specifically, we substitute the ST-GC encoder with the Ra-GConv module [4] and a conventional 2D convolutional module [35], resulting in three variants: MCDA-None (ST-GC encoder removed), $\mathbf { M C D A - R a - G C o n v }$ (replaced with Ra-GConv), and MCDA-2DCNN (replaced with 2D CNN). Ra-GConv leverage the topological and functional information of fMRI to learn diferent embedding weights in graph convolutional kernels conditioned on the ROI (geometrically distributed information of the brain), while 2D CNN operations process rs-fMRI data as regular grid-like inputs with sliding a fixed-size convolution kernel to extract localized patterns on the graph. Consistent with the ST-GC encoder 3-layer structure, we implement both Ra-GConv and 2D-CNN variants with three layers each. The results of MCDA and three variants on Site 20 of REST-MDD are reported in Table VI. This table suggests that the MCDA-Ra-GConv and MCDA-2DCNN variants show decent yet limited efectiveness; however, MCDA-None yields significantly inferior cross-validation results, primarily due to its inability to efectively extract and utilize information from the augmented brain networks. In contrast, our proposed ST-GC encoder outperforms all three variants by a clear margin. These results suggest that fully leveraging both temporal and spatial attributes of brain networks enables the extraction of more informative and diagnostically relevant features, thereby enhancing the performance of downstream classification tasks.

![](images/445f1d4d8758e0b537ee384e5adbfa0ba71312833a332923367d55ca34c3e8d7.jpg)

![](images/241bf662036d688fb28b580ce1e080a87937a7a353e8e85327792c0057112b84.jpg)

Fig. 7. Accuracy and AUC of MCDA with diferent Topology-guided Multiview Contrastive Loss parameters α and β on Site 20 from the REST-MDD  
![](images/46def83fb76a88b501659e4aab46b952196e8af16043a4d5a08ba4bc840b1f83.jpg)  
Fig. 8. Results of MCDA with diferent loss parameter λ on Site 20 from the REST-MDD dataset.

## E. Parameter Analysis

Site 20 of REST-MDD was used as the development site. Hyperparameters were selected using validation AUC from the five-fold partitions of this site. The selected configuration was subsequently fixed for all site-wise experiments. The reported results reveal performance on the held-out test folds for the evaluated parameter settings.

1) Influence of Trade-of TMC Loss Parameters: To examine the efects of trade-of parameters in topology-guided multi-view contrastive loss ${ \mathcal { L } } _ { \mathrm { T M C } }$ of Eq. (13), we adjust the values of $\alpha$ and $\beta .$ The parameter $\alpha ,$ which regulates the contribution of self-aware and cross-sample augmentation, is varied from 0 to 1. Meanwhile, $\beta ,$ reflecting the impact of diferences between augmented brain networks, is tested at 0.001, 0.01, 0.1, 1, and 10. Fig. 7 presents a subset of the accuracy and AUC results, which indicates that MCDA achieves optimal performance when $\alpha = 0 . 3 , \beta = 1$ . Furthermore, we investigate how the TMC loss ${ \mathcal { L } } _ { \mathrm { T M C } }$ , scaled by the tradeof parameter $\lambda ,$ afects the classification performance. The corresponding results on Site 20 are represented in Fig. 8, and MCDA ultimately achieves its optimum at $\lambda = 0 . 5$ . Using large values for λ, our framework cannot achieve good results. This implies that excessive weighting of the TMC loss causes the framework to overly focus on the relative distances between multiple augmented views, leading to a less compact feature embedding space and consequently impairing classification performance.

2) Influence of History Windows Size: As shown in Table VII, increasing K from 1 to 5 improves ACC from

TABLE VII  
RESULTS (%) OF MCDA WITH DIFFERENT HISTORY WINDOW LENGTHS K ON SITE 20 FROM THE REST-MDD DATASET.
<table><tr><td> $K$ </td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>1</td><td> $6 0 . 2 _ { \pm 3 . 1 }$ </td><td> $5 6 . 4 _ { \pm 3 . 7 }$ </td><td> $5 6 . 0 _ { \pm 5 . 7 }$ </td><td> $6 5 . 5 _ { \pm 2 . 4 }$ </td></tr><tr><td>3</td><td> $6 4 . 4 _ { \pm 0 . 8 }$ </td><td> $5 9 . 2 _ { \pm 3 . 5 }$ </td><td> $6 1 . 0 _ { \pm 3 . 5 }$ </td><td> $6 1 . 4 _ { \pm 6 . 7 }$ </td></tr><tr><td>5</td><td> $\mathbf { 7 0 . 3 _ { \pm 2 . 9 } }$ </td><td> $7 4 . 5 _ { \pm 2 . 8 }$ </td><td> $7 2 . 3 _ { \pm 4 . 0 }$ </td><td> $6 5 . 6 _ { \pm 8 . 4 }$ </td></tr><tr><td>7</td><td> $6 7 . 0 _ { \pm 2 . 9 }$ </td><td> $6 6 . 6 _ { \pm 9 . 6 }$ </td><td> $6 7 . 7 _ { \pm 6 . 9 }$ </td><td> $6 2 . 1 _ { \pm 6 . 5 }$ </td></tr><tr><td>9</td><td> $6 9 . 0 _ { \pm 3 . 5 }$ </td><td> $6 8 . 0 _ { \pm 8 . 5 }$ </td><td> $6 6 . 9 _ { \pm 6 . 5 }$ </td><td> $\mathbf { 6 5 . 8 _ { \pm 5 . 1 } }$ </td></tr></table>

TABLE VIII

RESULTS (%) OF MCDA WITH DIFFERENT BATCH SIZES ON SITE 20 FROM THE REST-MDD DATASET.
<table><tr><td>Batch Size</td><td>ACC</td><td>AUC</td><td>PRE</td><td>REC</td></tr><tr><td>30</td><td> $5 9 . 1 _ { \pm 6 . 1 }$ </td><td> $5 6 . 2 _ { \pm 4 . 6 }$ </td><td> $5 3 . 2 _ { \pm 1 5 . 4 }$ </td><td> $4 4 . 5 _ { \pm 1 4 . 7 }$ </td></tr><tr><td>35</td><td> $6 1 . 6 _ { \pm 6 . 4 }$ </td><td> $6 0 . 4 _ { \pm 5 . 3 }$ </td><td> $5 7 . 3 _ { \pm 4 . 8 }$ </td><td> $5 4 . 3 _ { \pm 8 . 8 }$ </td></tr><tr><td>40</td><td> $5 8 . 9 _ { \pm 6 . 2 }$ </td><td> $6 1 . 4 _ { \pm 1 0 . 1 }$ </td><td> $5 5 . 3 _ { \pm 9 . 3 }$ </td><td> $5 0 . 4 _ { \pm 7 . 8 }$ </td></tr><tr><td>45</td><td> $\mathbf { 7 0 . 3 _ { \pm 2 . 9 } }$ </td><td> $7 4 . 5 _ { \pm 2 . 8 }$ </td><td> $7 2 . 3 _ { \pm 4 . 0 }$ </td><td> ${ \bf 6 5 . 6 _ { \pm 8 . 4 } }$ </td></tr><tr><td>50</td><td> $6 4 . 2 _ { \pm 4 . 0 }$ </td><td> $6 7 . 1 _ { \pm 7 . 5 }$ </td><td> $5 2 . 7 _ { \pm 7 . 5 }$ </td><td> $6 5 . 1 _ { \pm 8 . 2 }$ </td></tr></table>

60.23% to 70.28% and AUC from 56.36% to 74.53%, with K=5 also achieving the highest PRE of 72.29%. A moderate descriptor window may provide useful cross-epoch context for BiGRU-based gate estimation, whereas $K { = } 1$ supplies only the current descriptor. Extending the window beyond five epochs does not further improve these three metrics, possibly because older descriptors reflect earlier training states and may be less compatible with the current augmentation. Although $K { = } 9$ achieves a slightly higher REC than K=5, K=5 provides better ACC, AUC, and PRE.

3) Influence of Batch Size: We further study the impact of diferent batch sizes. Based on data from Site 20 of REST-MDD, we set variations of 30 to 50 with a interval of 5 to assess the impact of subject variability across diferent group sizes on classification tasks. Specifically, the results on Site 20 are presented Table VIII. The optimal performance across the four metrics is achieved when the batch size is set to 45. Using an excessively large or small batch size limits the ability of dynamic topology augmentation to capture intersubject variability, which in turn reduces its efectiveness in identifying discriminative features.

## F. Limitation and Future Work

Several limitation of current work need to be considered. On one hand, the brain network construction relies solely on rsfMRI. In the future, we will integrate multi-modal data (e.g., difusion tensor imaging [36], demographic factors [37] and clinical scores [38]) to explore complementary relationships between imaging and none-imaging patterns. On the other hand, our application scenario is limited to data collected from a single site, overlooking the potential heterogeneity caused by diferences in acquisition machine parameters across multiple sites in real-world settings. A possible solution is to employ domain adaption to reduce cross-domain distribution diference [39], which will be one of our future work.

## VI. CONCLUSION

This paper introduces a topology-guided multi-view graph contrastive learning framework (MCDA) with dynamic topology augmentation for brain disorder diagnosis and analysis. In our design, two complementary brain networks are generated through a dynamic augmentation update (DAU) module, incorporating both self-aware and cross-sample strategies to capture subject-specific traits and functional heterogeneity across individuals. In addition, we develop a topology-aware multi-view contrastive learning paradigm that simultaneously models the interrelationships between the original graph and its augmented counterparts, as well as between the augmented views themselves. We further characterize the optimization directions of the contrastive constraints through similarityscore gradient analysis and establish a lower bound for the contrastive objective. Experiments on two public rs-fMRI datasets demonstrate improved classification performance compared with several state-of-the-art methods.

## APPENDIXOPTIMIZATION-DIRECTION ANALYSIS OF THE MULTI-VIEWCONTRASTIVE OBJECTIVE

This appendix examines the optimization directions and boundedness of the proposed multi-view contrastive objective. The analysis is conducted with respect to the similarity scores and provides a geometric interpretation under unit-norm normalization.

## A. Similarity scores and loss formulation

The three constraints operate on the counterparts, respectively. Their similarity scores are defined as follows:

$$
\begin{array} { r l } & { ~ s _ { i j } ^ { \mathrm { \tiny { S A } } } = \sin \left( { \bf G } _ { i } ^ { \mathrm { \tiny { S A } } } , { \bf G } _ { j } \right) , } \\ & { ~ s _ { i j } ^ { \mathrm { \tiny { C A } } } = \sin \left( { \bf G } _ { i } ^ { \mathrm { \tiny { C A } } } , { \bf G } _ { j } \right) , } \\ & { ~ s _ { i j } ^ { \mathrm { \tiny { S A , C A } } } = \sin \left( { \bf G } _ { i } ^ { \mathrm { \tiny { S A } } } , { \bf G } _ { j } ^ { \mathrm { \tiny { C A } } } \right) , } \end{array}\tag{16}
$$

For each constraint indexed by $r \in \{ \mathrm { S A } , \mathrm { C A } , ( \mathrm { S A } , \mathrm { C A } ) \}$ , the similarities are normalized across the mini-batch. The resulting matching score is defined as follows:

$$
q _ { i j } ^ { r } = \frac { \exp ( s _ { i j } ^ { r } / \tau ) } { \sum _ { k = 1 } ^ { B } \exp ( s _ { i k } ^ { r } / \tau ) } ,\tag{17}
$$

The quantity $q _ { i j } ^ { r }$ represents the relative matching score within the mini-batch and is distinct from the classification probability $p _ { i }$ . Throughout this analysis, we assume $B \geq 2$ $\tau > 0$ , and finite similarity scores. Under these conditions, every matching score lies strictly between zero and one.

The three losses difer in the sign assigned to the matchedpair log-ratio. They can therefore be expressed in the following unified form:

$$
\mathcal { L } _ { r } = \frac { \eta _ { r } } { B } \sum _ { i = 1 } ^ { B } \log q _ { i i } ^ { r } , \quad \eta _ { \mathrm { S A } } = - 1 , \quad \eta _ { \mathrm { C A } } = \eta _ { \mathrm { S A , C A } } = 1 .\tag{18}
$$

The negative sign specifies the consistency constraint for the self-aware with original pair. The positive signs specify the diversity constraints for the cross-sample with original and self-aware with cross-sample pairs.

## B. Optimization directions

The logarithm of the matched score admits the following expansion:

$$
\log q _ { i i } ^ { r } = \frac { s _ { i i } ^ { r } } { \tau } - \log \left[ \sum _ { k = 1 } ^ { B } \exp ( { s _ { i k } ^ { r } } / \tau ) \right] ,\tag{19}
$$

Diferentiating the unified loss with respect to an individual similarity score yields the following expression:

$$
\frac { \partial \mathcal { L } _ { r } } { \partial s _ { i j } ^ { r } } = \frac { \eta _ { r } } { B \tau } \left( \delta _ { i j } - q _ { i j } ^ { r } \right) ,\tag{20}
$$

Here, $\delta _ { i j }$ is the Kronecker delta. It equals one for matched pairs and zero for non-matched pairs. For matched pairs, setting $j = i$ gives the following derivatives:

$$
\begin{array} { r l } & { \frac { \partial \mathcal { L } _ { \mathrm { S A } } } { \partial s _ { i i } ^ { \mathrm { S A } } } = - \frac { 1 - q _ { i i } ^ { \mathrm { S A } } } { B \tau } < 0 , } \\ & { \frac { \partial \mathcal { L } _ { \mathrm { C A } } } { \partial s _ { i i } ^ { \mathrm { C A } } } = \frac { 1 - q _ { i i } ^ { \mathrm { C A } } } { B \tau } > 0 , } \\ & { \frac { \partial \mathcal { L } _ { \mathrm { S A , C A } } } { \partial s _ { i i } ^ { \mathrm { S A , C A } } } = \frac { 1 - q _ { i i } ^ { \mathrm { S A , C A } } } { B \tau } > 0 , } \end{array}\tag{21}
$$

The strict signs follow from the matching scores lying between zero and one. When the similarity scores are treated as independent coordinates, the negative-gradient directions favor a higher matched self-aware with original score and lower matched cross-sample with original and self-aware with cross-sample scores.

For non-matched pairs, setting $j \neq i$ gives the following derivatives:

$$
\begin{array} { r l } & { \frac { \partial \mathcal { L } _ { \mathrm { S A } } } { \partial s _ { i j } ^ { \mathrm { S A } } } = \frac { q _ { i j } ^ { \mathrm { S A } } } { B \tau } > 0 , } \\ & { \frac { \partial \mathcal { L } _ { \mathrm { C A } } } { \partial s _ { i j } ^ { \mathrm { C A } } } = - \frac { q _ { i j } ^ { \mathrm { C A } } } { B \tau } < 0 , } \\ & { \frac { \partial \mathcal { L } _ { \mathrm { S A , C A } } } { \partial s _ { i j } ^ { \mathrm { S A , C A } } } = - \frac { q _ { i j } ^ { \mathrm { S A , C A } } } { B \tau } < 0 . } \end{array}\tag{22}
$$

The self-aware constraint favors lower non-matched scores, whereas the two diversity constraints favor higher non-matched scores. Together with the matched-pair derivatives, these results show that the losses regulate matched scores relative to other pairings within the mini-batch. In particular, the diversity constraints do not impose uniform repulsion between all views.

The combined objective weights the three constraints by $\alpha , 1 - \alpha .$ , and $\beta ,$ respectively. For $\alpha \in [ 0 , 1 ]$ and $\beta \geq 0 ,$ positive weights preserve the corresponding derivative signs in similarity-score space. A zero weight removes the associated constraint.

## C. Geometric interpretation

Cosine similarity provides a distance interpretation when the norms of the compared views are at least $\varepsilon > 0$ . Under this condition, the numerical-stability operation does not alter the normalization, and cosine similarity equals the inner product of unit-normalized representations. For the self-aware with original pair, the normalized representations are defined as follows:

$$
\widehat { \mathbf { G } } _ { i } = \frac { \mathbf { G } _ { i } } { \| \mathbf { G } _ { i } \| _ { T , N } } , \quad \widehat { \mathbf { G } } _ { i } ^ { \mathrm { S A } } = \frac { \mathbf { G } _ { i } ^ { \mathrm { S A } } } { \| \mathbf { G } _ { i } ^ { \mathrm { S A } } \| _ { T , N } } ,\tag{23}
$$

Expanding their squared distance gives the following identity:

$$
\left. \widehat { \mathbf { G } } _ { i } ^ { \mathrm { S A } } - \widehat { \mathbf { G } } _ { i } \right. _ { T , N } ^ { 2 } = 2 - 2 s _ { i i } ^ { \mathrm { S A } } .\tag{24}
$$

The same identity applies to the two remaining pairs under the corresponding normalization conditions. A higher cosine similarity therefore corresponds to a smaller squared distance between unit-normalized representations, and a lower similarity corresponds to a larger distance. This identity supplies a geometric interpretation of the score-level optimization directions. It does not establish corresponding changes in distances between unnormalized representations, graph topology, or diagnostic semantics.

## D. Boundedness of the objective

The cosine similarity scores lie between minus one and one. Factoring the matched exponential term out of the denominator rewrites the matching score as follows:

$$
q _ { i i } ^ { r } = \left[ 1 + \sum _ { \stackrel { k = 1 } { k \neq i } } ^ { B } \exp \left( \frac { s _ { i k } ^ { r } - s _ { i i } ^ { r } } { \tau } \right) \right] ^ { - 1 } ,\tag{25}
$$

Each diference between a non-matched and matched score is at most two. Applying this bound to the denominator yields the following inequality:

$$
\frac { 1 } { 1 + ( B - 1 ) \exp ( 2 / \tau ) } \le q _ { i i } ^ { r } < 1 ,\tag{26}
$$

Taking logarithms and applying the negative sign of the SA constraint gives the following consistency-loss bound:

$$
0 \leq \mathcal { L } _ { \mathrm { S A } } \leq \log \left[ 1 + \left( B - 1 \right) \exp ( 2 / \tau ) \right] ,\tag{27}
$$

The positive signs of the CA and SA–CA constraints instead give the following diversity-loss bounds:

$$
- \log { [ 1 + ( B - 1 ) \exp ( 2 / \tau ) ] } \leq \mathcal { L } _ { r } \leq 0 ,\tag{28}
$$

The consistency loss is non-negative. Combining the two diversity-loss lower bounds with their non-negative weights therefore gives a lower bound for the complete contrastive objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T M C } } \geq - ( 1 - \alpha + \beta ) \log \left[ 1 + \left( B - 1 \right) \exp ( 2 / \tau ) \right] . } \end{array}\tag{29}
$$

Thus, although the diversity terms can take negative values, the contrastive objective cannot decrease without bound for fixed $B , \ \tau , \ \alpha ,$ , and $\beta .$ This property does not, by itself, guarantee optimization stability or convergence.

## REFERENCES

[1] T. Liu, L. Wang, D. Suo, J. Zhang, K. Wang, J. Wang, D. Chen, and T. Yan, “Resting-state functional MRI of healthy adults: temporal dynamic brain coactivation patterns,” Radiology, vol. 304, no. 3, pp. 624–632, 2022.

[2] Y. Ma, C. Zhang, D. Xiong, H. Zhang, and S. Ying, “Multi-task dynamic graph learning for brain disorder identification with functional MRI,” Pattern Recognition, vol. 170, p. 111922, 2026.

[3] W. Marx, B. W. Penninx, M. Solmi, T. A. Furukawa, J. Firth, A. F. Carvalho, and M. Berk, “Major depressive disorder,” Nature Reviews Disease Primers, vol. 9, no. 1, p. 44, 2023.

[4] X. Li, Y. Zhou, N. Dvornek, M. Zhang, S. Gao, J. Zhuang, D. Scheinost, L. H. Staib, P. Ventola, and J. S. Duncan, “BrainGNN: Interpretable brain graph neural network for fMRI analysis,” Medical Image Analysis, vol. 74, p. 102233, 2021.

[5] Y. Wang, V. D. Calhoun, G. D. Pearlson, P. Kochunov, T. G. van Erp, and Y. Du, “A graph transformer-based foundation model for brain functional connectivity network,” Pattern Recognition, vol. 169, p. 111988, 2026.

[6] X. Wang, Y. Chu, Q. Wang, L. Cao, L. Qiao, L. Zhang, and M. Liu, “Unsupervised contrastive graph learning for resting-state functional fMRI analysis and brain disorder detection,” Human Brain Mapping, vol. 44, no. 17, pp. 5672–5692, 2023.

[7] Y. Zong, Q. Zuo, M. K.-P. Ng, B. Lei, and S. Wang, “A new brain network construction paradigm for brain disorder via difusion-based graph contrastive learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10 389–10 403, 2024.

[8] N. Tian, X. Xu, and W. Li, “Multi-View Comparative Self-Supervised Graph Learning for neuro-disease diagnosis using functional brain networks,” in 2024 International Conference on Cyber-Physical Social Intelligence (ICCSI). IEEE, 2024, pp. 1–6.

[9] H. Zhang, X. Liu, S. Huang, Y. Yuan, D. Zhang, and L. Zhang, “Multiview graph contrastive learning with dynamic self-aware and crosssample topology augmentation for brain disorder diagnosis,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. Springer, 2025, pp. 532–542.

[10] C. Oh, S. Han, and J. Jeong, “Time-series data augmentation based on interpolation,” Procedia Computer Science, vol. 175, pp. 64–71, 2020.

[11] X. Wang, Y. Fang, Q. Wang, P.-T. Yap, H. Zhu, and M. Liu, “Selfsupervised graph contrastive learning with difusion augmentation for functional MRI analysis and brain disorder detection,” Medical Image Analysis, vol. 101, p. 103403, 2025.

[12] Z. Li, Z. Zhu, Q. Li, and X. Wu, “Decoding the brain via multi-view brain topology contrastive learning,” Pattern Recognition, p. 112445, 2025.

[13] H. Zhang, X. Liu, S. Huang, Y. Ma, Y. Yuan, D. Zhang, and L. Zhang, “Graph-level contrastive learning with self-aware and cross-sample topology augmentation for brain disorder diagnosis using rs-fMRI,” Neural Networks, vol. 196, p. 108379, 2026.

[14] Y. Xiong, D.-Z. Jin, and Y. He, “Self-Supervised Temporal-Connectivity Contrastive Learning for fMRI analysis on autism spectrum disorder classification,” in 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2024, pp. 1710–1713.

[15] M. Wang, J. Huang, M. Liu, and D. Zhang, “Modeling dynamic characteristics of brain functional connectivity networks using restingstate functional MRI,” Medical Image Analysis, vol. 71, p. 102063, 2021.

[16] D. Misra, T. Nalamada, A. U. Arasanipalai, and Q. Hou, “Rotate to attend: Convolutional triplet attention module,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 3139–3148.

[17] H. Tang, G. Ma, L. Guo, X. Fu, H. Huang, and L. Zhan, “Contrastive brain network learning via hierarchical signed graph pooling model,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 6, pp. 7363–7375, 2022.

[18] Y. Xu, J. Wang, M. Guang, C. Yan, and C. Jiang, “Graph contrastive learning with min-max mutual information,” Information Sciences, vol. 665, p. 120378, 2024.

[19] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, “Automatic diferentiation in PyTorch,” in Proc. NeurIPS Workshop Autodif, 2017.

[20] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[21] C.-G. Yan, X. Chen, L. Li, F. X. Castellanos, T.-J. Bai, Q.-J. Bo, J. Cao, G.-M. Chen, N.-X. Chen, W. Chen et al., “Reduced default mode network functional connectivity in patients with recurrent major

depressive disorder,” Proceedings of the National Academy of Sciences of the United States ofAmerica, vol. 116, no. 18, pp. 9078–9083, 2019.

[22] A. Di Martino, C.-G. Yan, Q. Li, E. Denio, F. X. Castellanos, K. Alaerts, J. S. Anderson, M. Assaf, S. Y. Bookheimer, M. Dapretto et al., “The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism,” Molecular Psychiatry, vol. 19, no. 6, pp. 659–667, 2014.

[23] C. Yan and Y. Zang, “DPARSF: a MATLAB toolbox for ”pipeline” data analysis of resting-state fMRI,” Frontiers in Systems Neuroscience, vol. 4, p. 13, 2010.

[24] N. Tzourio-Mazoyer, B. Landeau, D. Papathanassiou, F. Crivello, O. Etard, N. Delcroix, B. Mazoyer, and M. Joliot, “Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain,” NeuroImage, vol. 15, no. 1, pp. 273–289, 2002.

[25] E. Kang, D.-W. Heo, J. Lee, and H.-I. Suk, “A learnable countercondition analysis framework for functional connectivity-based neurological disorder diagnosis,” IEEE Transactions on Medical Imaging, vol. 43, no. 4, pp. 1377–1387, 2023.

[26] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 3519–3529.

[27] A. M. Winkler, G. R. Ridgway, M. A. Webster, S. M. Smith, and T. E. Nichols, “Permutation inference for the general linear model,” NeuroImage, vol. 92, pp. 381–397, 2014.

[28] Y. Benjamini and D. Yekutieli, “The control of the false discovery rate in multiple testing under dependency,” The Annals of Statistics, vol. 29, no. 4, pp. 1165–1188, 2001.

[29] M. Xia, J. Wang, and Y. He, “BrainNet Viewer: a network visualization tool for human brain connectomics,” PLOS One, vol. 8, no. 7, p. e68910, 2013.

[30] C. He, D. Fan, X. Liu, Q. Wang, H. Zhang, H. Zhang, Z. Zhang, and C. Xie, “Insula network connectivity mediates the association between childhood maltreatment and depressive symptoms in major depressive disorder patients,” Translational Psychiatry, vol. 12, no. 1, p. 89, 2022.

[31] J. Wei, Y. Wang, Z. Zhang, X. Yang, L. Zhao, P. Ni, R. Ni, and X. Ma, “Disrupted association between structural and functional coupling of the supplementary motor area and neurocognition in major depressive disorder,” Chinese Medical Journal, vol. 136, no. 17, pp. 2131–2133, 2023.

[32] I. Fishman, A. C. Linke, J. Hau, R. A. Carper, and R.-A. Müller, “Atypical functional connectivity of amygdala related to reduced symptom severity in children with autism,” Journal of the American Academy of Child & Adolescent Psychiatry, vol. 57, no. 10, pp. 764–774.e3, 2018.

[33] Y. Yang, N. Razmjooy et al., “Early detection of brain tumors: Harnessing the power of GRU networks and hybrid dwarf mongoose optimization algorithm,” Biomedical Signal Processing and Control, vol. 91, p. 106093, 2024.

[34] R. Supakar, P. Satvaya, and P. Chakrabarti, “A deep learning based model using RNN-LSTM for the detection of schizophrenia from EEG data,” Computers in Biology and Medicine, vol. 151, p. 106225, 2022.

[35] J. Hu, Y. Kuang, B. Liao, L. Cao, S. Dong, and P. Li, “A multichannel 2D convolutional neural network model for task-evoked fMRI data classification,” Computational Intelligence and Neuroscience, vol. 2019, p. 5065214, 2019.

[36] F. Zhang, W. M. Wells, and L. J. O’Donnell, “Deep difusion MRI registration (DDMReg): a deep learning method for difusion MRI registration,” IEEE Transactions on Medical Imaging, vol. 41, no. 6, pp. 1454–1467, 2021.

[37] H. Hernandez, S. Baez, V. Medel, S. Moguilner, J. Cuadros, H. Santamaria-Garcia, E. Tagliazucchi, P. A. Valdes-Sosa, F. Lopera, J. F. OchoaGómez et al., “Brain health in diverse settings: How age, demographics and cognition shape brain function,” NeuroImage, vol. 295, p. 120636, 2024.

[38] S. Mishra, I. Beheshti, and P. Khanna, “A review of neuroimagingdriven brain age estimation for identification of brain disorders and health conditions,” IEEE Reviews in Biomedical Engineering, vol. 16, pp. 371–385, 2021.

[39] H. Guan and M. Liu, “Domain adaptation for medical image analysis: a survey,” IEEE Transactions on Biomedical Engineering, vol. 69, no. 3, pp. 1173–1185, 2021.