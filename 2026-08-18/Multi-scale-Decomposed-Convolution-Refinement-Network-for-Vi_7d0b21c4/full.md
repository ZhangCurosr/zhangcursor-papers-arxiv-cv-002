# Multi-scale Decomposed Convolution Refinement Network for Visible-Infrared Person Re-Identification

Mingsheng Zheng<sup>1</sup>, Zirui Jiang<sup>1</sup>, Bo Liu<sup>1</sup>, Yupeng Chen<sup>1</sup>, Jun Zhang<sup>1</sup>, and Kai Zhao<sup>1,2(B)</sup>

<sup>1</sup> School of Computer Science and Technology, Xinjiang University, Urumqi, Xinjiang 830046, China

Joint International Research Laboratory of Silk Road Multilingual Cognitive Computing, Xinjiang University, Urumqi, Xinjiang 830046, China zhawkk@xju.edu.cn

Abstract. Visible-infrared person re-identification (VI-ReID) sufers from cross-modal discrepancies and limited discriminative capabilities, leading to suboptimal recognition performance. Current approaches exhibit limitations in semantic mining, cross-modal fusion and feature constraints. To tackle these challenges, we propose MDCRNet, a Multi-scale Decomposed Convolution Refinement Network that enhances cross-modal feature learning and discriminative metric learning. Specifically, we introduce a Hierarchical Learning Module (HLM) containing four Hierarchical Decomposed Convolution Attention (HDCA) modules, each equipped with lightweight channel attention and multi-scale spatial perception blocks to capture multi-scale spatial dependencies. Moreover, we develop a Joint Discriminative Metric Loss (JDML) incorporating a novel Granularity Discriminative Loss (GDL) that simultaneously optimizes intraidentity compactness and inter-identity separability across modalities. Extensive experiments on SYSU-MM01 and RegDB datasets demonstrate that MDCRNet achieves state-of-the-art performance on both benchmarks. Code is available at https://github.com/Kevin-zms/MDCRNe

Keywords: Visible-infrared person re-identification · Cross-modal learning · Attention mechanism · Metric learning · Multi-scale.

## 1 Introduction

Person re-identification (ReID) aims to recognize individuals across diferent camera views, with critical applications in intelligent surveillance and public safety. Traditional ReID methods primarily focus on red-green-blue (RGB) images captured under visible light conditions, leading to severely degraded performance in nighttime, low-light, or adverse weather scenarios. To address these limitations, VI-ReID has emerged as a promising solution for continuous 24- hour surveillance by leveraging complementary information from both visible and infrared modalities. Nevertheless, VI-ReID encounters considerable dificulties stemming from pronounced cross-modal disparities between visible imagery that encompasses abundant chromatic information and textural details, and infrared imagery that records geometric boundaries and heat patterns, further complicated by within-modal variations such as perspective alterations, object obstructions, and apparel diferences.

Early VI-ReID approaches primarily relied on feature fusion strategies and semantic information learning techniques to bridge cross-modal discrepancies. With advances in deep learning, mainstream methods have evolved from traditional convolutional neural network (CNN) networks [19] to sophisticated architectures incorporating ResNet backbones, Transformer-based attention mechanisms [1], and CLIP-assisted networks for enhanced cross-modal representation learning [34, 32]. Recent approaches [8, 18, 9, 10] focus on learning modalityinvariant features through adversarial training or shared embedding spaces to minimize modal diferences between visible and infrared modalities and designing efective loss functions to optimize models. However, existing methods exhibit significant limitations in three critical aspects. First, methods like MSCMNet [11] employ eficient attention mechanisms to enhance cross-modal learning but rely on simple concatenation for fusion, overlooking inherent cross-modal data discrepancies. Second, channel-based approaches [30] partially address modality diferences through channel augmentation yet neglect deep semantic mining and multi-scale spatial relationship learning. Third, existing loss functions typically address either inter-modal alignment or inter-identity discrimination independently, without providing unified constraints for diverse feature relationships including same-identity cross-modal, diferent-identity cross-modal, and intramodal scenarios [37]. As shown in Fig. 1(a), fixed-scale perception in existing methods captures only partial information and fails to jointly model fine-grained cues (hands, watch, texture), mid-grained cues (face, clothing shape, upper-body pose), and coarse-grained cues (overall silhouette). Our progressive multi-scale design (Fig. 1(b)) captures all three granularities via kernels of increasing size, while channel attention adaptively fuses the cross-modal streams, jointly resolving the first two limitations. Furthermore, the resulting cross-modal crossidentity features are still poorly exploited by conventional losses during optimization, which motivates our unified discriminative metric to address the third limitation.

To address these limitations, we propose MDCRNet, a novel framework that efectively mitigates cross-modal discrepancies through hierarchical feature learning and discriminative optimization. Our approach begins with grayscale transformations [32] applied to both visible and infrared modalities, generating four complementary feature streams [11] that reduce modal gaps while preserving essential structural information. Built upon a ResNet backbone, we introduce the HLM equipped with four HDCA modules, each containing Channel Attention (CA) for eficient cross-modal fusion and Multi-scale Spatial Perception Block (MSPB) for capturing hierarchical spatial dependencies. HDCA decomposes channels into groups with separate attention weights to disentangle modality-specific/shared features, while HLM is the sequential framework combining HDCA and Progressive Distance Perception Attention (PDPA). The

![](images/600a00d9440fd3cfc3d647dd0b7a84cd08ff69fb117bb9c644ea096e849aa4aa.jpg)  
Fig. 1. Motivation of MDCRNet. (a) Existing methods adopt fixed-scale perception and capture only partial cues. (b) Our progressive multi-scale kernels jointly capture fine-grained (hands, watch, texture), mid-grained (face, clothing shape, upper-body pose), and coarse-grained (overall silhouette) information across the visible, infrared, and channel-augmented streams.

MSPB incorporates three PDPA components with three kernel sizes, systematically extracting short-range, medium-range, and long-range spatial correlations for comprehensive semantic understanding. Furthermore, we design a JDML that integrates our novel GDL with existing identity and Quadruple Center Triplet (QCT) loss. The GDL employs three synergistic constraints: center loss to align same-identity cross-modal features, cosine similarity loss to separate diferentidentity features, and adaptive identity loss to enhance intra-modal clustering. Our three contributions are as follows:

– We propose MDCRNet, an innovative architectural approach that efectively reduces cross-modal discrepancies through hierarchical decomposed convolution attention and progressive distance perception for VI-ReID.

– We design the HLM with HDCA modules that integrate CA for cross-modal fusion and multi-scale spatial perception blocks with PDPA, enabling extraction of discriminative semantic representations across diferent spatial ranges.

We develop JDML incorporating a novel GDL with three complementary constraints that jointly optimize cross-modal alignment, inter-identity separation, and intra-modal consistency, demonstrating superior performance on SYSU-MM01 and RegDB benchmarks.

## 2 Related Work

## 2.1 Cross-Modal Person Re-Identification

More broadly, recent advances in multimodal vision have emphasized eficient and visually grounded instruction tuning [5, 6], while extending visual understanding to synthetic long-form benchmarks and memory-eficient streaming scenarios [7, 13]. Cross-modal person re-identification faces significant challenges in feature extraction and embedding alignment due to inherent modality discrepancies between visible and infrared images [17]. Recent works have explored modality-shared networks to mitigate these discrepancies, yet often fail to capture multi-scale information such as texture, color, and structural cues [32, 8, 30]. Another line of research enhances semantic representation through attention mechanisms and modality conversion, but these approaches may inadvertently disrupt the original semantic structures [36, 22, 14]. Furthermore, existing methods typically concatenate visible and infrared features, which proves insuficient for efective multimodal fusion [11, 3]. To address these challenges, we propose a multi-scale network that eficiently fuses and learns hierarchical information, providing a more robust framework for cross-modal representation learning. Unlike MSCMNet [11], which relies on simple feature concatenation and lacks unified multi-granularity constraints, and DSAF [12], whose dual-space alignment lacks fine-grained intra-class discriminative constraints, our HDCA and PDPA modules explicitly enforce both channel-wise modality disentanglement and distanceaware spatial hierarchy, providing the unified multi-granularity constraints that prior methods are missing.

## 2.2 Modality Alignment

Vision-language models have demonstrated remarkable progress across various vision tasks. Inspired by CLIP [20], several VI-ReID frameworks have been proposed to leverage contrastive learning for cross-modal alignment. However, these approaches encounter challenges in efective feature alignment. For instance, SDCL [27] integrates shallow and deep features via collaborative neighbor learning and ranking association, yet lacks a unified discriminative metric to jointly optimize cross-modality alignment and identity separation. Similarly, DSAF [12] aligns visible-infrared modalities in dual spaces (Euclidean and Hilbert) with adaptive identity-consistent learning, yet lacks fine-grained discriminative constraints to simultaneously optimize intra-class compactness and inter-class separability. To address these limitations, we design a unified discriminative training objective that enforces cross-modal feature alignment while explicitly optimizing both intra-identity compactness and inter-identity separability.

## 3 Method

In visible-infrared person re-identification, given training samples $\{ \mathbf { x } _ { i } ^ { v } , \mathbf { x } _ { i } ^ { t } , y _ { i } \} _ { i = 1 } ^ { N }$ where $\mathbf { x } _ { i } ^ { v } \in \mathbb { R } ^ { H \times \tilde { W } \times 3 }$ and $\mathbf { x } _ { i } ^ { t } \in \mathbb { R } ^ { H \times W \times \dot { 3 } }$ denote visible and infrared images of identity $y _ { i }$ , the goal is to learn discriminative embeddings $\mathbf { f } _ { i } ^ { v } , \mathbf { f } _ { i } ^ { t } \in \mathbb { R } ^ { D }$ that minimize intra-identity cross-modal distances while maximizing inter-identity separability. In this section, we present MDCRNet to efectively mitigate crossmodal discrepancies while enhancing discriminative feature learning for VI-ReID, as illustrated in Fig. 2. Our approach adopts a quad-stream architecture that simultaneously processes original visible and infrared images alongside their channel-augmented variants, generating complementary feature representations through separate extraction pathways. The framework achieves efective crossmodal learning through two key innovations, namely the HLM equipped with four strategically positioned HDCA modules that capture multi-scale spatial dependencies through progressive distance perception mechanisms, and the JDML that jointly optimizes feature discrimination across modalities. During training, the HLM enhances cross-modal semantic alignment through eficient CA and multi-scale spatial perception, while JDML enforces fine-grained constraints on intra-identity compactness and inter-identity separability to achieve superior recognition performance.

![](images/5ac61e71cb76cdd3a8cf3a6530da76baa2bf64ee5449db0b5a1a0c1c783ff2e5.jpg)  
Fig. 2. MDCRNet employs a quad-stream architecture that processes cross-modal images along with their channel-augmented variants, then utilizes ResNet backbone for hierarchical feature extraction. The HLM integrates four HDCA modules, each containing CA and MSPB with three PDPA components to capture multi-scale spatial dependencies for efective cross-modal semantic learning. The JDML combines identity loss, QCT loss, and our proposed GDL.

## 3.1 Hierarchical Learning Module

The core challenge in VI-ReID lies in extracting discriminative features that are both modality-invariant and identity-specific. Existing methods often struggle to efectively fuse cross-modal information while capturing multi-scale spatial dependencies for comprehensive semantic understanding. To overcome this constraint, we introduce the HLM, which employs HDCA to facilitate eficient crossmodal learning and extract fine-grained discriminative features across diferent spatial ranges. The HLM contains four strategically positioned HDCA modules, where each HDCA integrates a CA mechanism and a MSPB. Each MSPB incorporates three PDPA components with kernel sizes 3, 5, and 7 to capture short-range, medium-range, and long-range spatial correlations respectively.

Eficient Channel Attention. To efectively integrate the extracted crossmodal features and enhance the cross-modal perception capabilities of the subsequent MSPB, we propose a lightweight channel gating attention mechanism that eficiently fuses feature representations from four distinct data streams:

$$
\mathbf { G } = \sigma ( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { R e L U } ( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { G A P } ( \mathbf { X } _ { c o m p } ) ) ) ) )\tag{1}
$$

where $\mathrm { G A P }$ denotes Global Average Pooling, σ is the Sigmoid activation and $\mathbf { X } _ { c o m p } \in \mathbb { R } ^ { B \times C \times H \times W }$ denotes the concatenated feature representation from the four input streams.

Progressive Distance Perception Attention. We employ three parallel PDPA modules that extract feature representations at diferent granularity levels:

Three parallel PDPA $\mathbf { ( P D P A _ { 3 , 5 , 7 } ) }$ : These modules, each with varying convolutional depths, facilitate the extraction of high-order structural information and semantic associations, particularly enabling the recovery of modalityspecific semantic information that may be attenuated in shallow network layers:

$$
\begin{array} { r } { \mathrm { P D P A } _ { 2 i + 1 } ( \mathbf X _ { i } ) = \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { D W C o n v } _ { ( 2 i + 3 ) \times ( 2 i + 3 ) } ^ { d = i + 1 } } \\ { ( \mathrm { D W C o n v } _ { ( 2 i + 1 ) \times ( 2 i + 1 ) } ( \mathbf X _ { i } ) ) ) } \end{array}\tag{2}
$$

where DWConv denotes depthwise convolution, i is the PDPA branch index $( i = 1 , 2 , 3 )$ , and $d = i + 1$ represents the dilation rate that progressively expands the receptive field. Compared with ASPP-style designs [2] that use fixed-size kernels with independent dilations (efective receptive field $\mathrm { E R F } = 2 r + 1 )$ , our PDPA employs cascaded progressive kernels with progressive dilations, achieving quadratic growth $\mathrm { E R F } = 2 i ^ { 2 } + 6 i + 3$ . The receptive field thus grows quadratically rather than linearly, enabling exponential expansion of spatial context while maintaining eficient parameter usage through depthwise separable convolutions. This accelerated expansion enables hierarchical distance-aware feature extraction from local to global contexts, directly addressing VI-ReID’s core challenge of discovering modality-invariant structural patterns across spatial scales while suppressing modality-specific appearance.

## 3.2 Joint Discriminative Metric Loss

To optimize the overall model performance, we propose the JDML which encompasses Identity loss [37], QCT loss [11] and our novel GDL. The GDL simultaneously enforces intra-identity compactness and inter-identity separability across diferent modalities through fine-grained constraints. Let ${ \bf f } _ { i } ^ { v }$ and $\mathbf { f } _ { i } ^ { t }$ denote normalized feature representations of the i-th identity from visible and thermal modalities, respectively. The GDL operates through three synergistic constraints to optimize diverse feature relationships across modalities and identities.

Center Loss for Cross-Modal Alignment. To minimize the spatial separation among identical-identity feature representations across distinct modal domains, we employ a center loss that pulls cross-modal features toward unified identity centers:

$$
\mathcal { L } _ { c e n t e r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \lVert \mathbf { f } _ { i } ^ { v } - \mathbf { c } _ { i } \rVert _ { 2 } ^ { 2 } + \lVert \mathbf { f } _ { i } ^ { t } - \mathbf { c } _ { i } \rVert _ { 2 } ^ { 2 } \right)\tag{3}
$$

where $\mathbf { f } _ { i } ^ { v } , \mathbf { f } _ { i } ^ { t } \in \mathbb { R } ^ { D }$ denote normalized D-dimensional features from visible and thermal modalities for the i-th identity, $\mathbf { c } _ { i } = \textstyle { \frac { 1 } { 2 } } ( \mathbf { f } _ { i } ^ { v } + \mathbf { f } _ { i } ^ { t } )$ represents the unified center for the i-th identity, and N is the number of identities in a batch.

Cosine Similarity Loss for Inter-Identity Separation. To enhance interidentity discriminability while maintaining intra-identity consistency across modalities, we introduce a bidirectional cosine similarity loss that operates on crossmodal identity relationships:

$$
\mathcal { L } _ { c o s i n e } = \mathcal { L } _ { v  t } + \mathcal { L } _ { t  v }\tag{4}
$$

where:

$$
\mathcal { L } _ { v  t } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } ( 0 , \operatorname* { m a x } _ { j \neq i } S _ { i j } - S _ { i i } + m )\tag{5}
$$

$$
\mathcal { L } _ { t  v } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } ( 0 , \operatorname* { m a x } _ { j \neq i } S _ { j i } - S _ { i i } + m )\tag{6}
$$

Here, $\mathbf { S } = \mathbf { F } ^ { v } ( \mathbf { F } ^ { t } ) ^ { T }$ represents the cosine similarity matrix between visible and thermal features, where $B$ is the batch size. $S _ { i i }$ represents the similarity between the i-th visible feature and its corresponding thermal feature (same identity), while $\operatorname* { m a x } _ { j \neq i } S _ { i j }$ represents the maximum similarity between the i-th visible feature and thermal features from diferent identities.

Adaptive Feature Aggregation for Identity Consistency. We propose an adaptive aggregation mechanism. For each identity group containing $K = 4$ samples, we compute adaptive weights based on inter-sample similarities. Here $K = 4$ follows the standard $P \times K$ sampling strategy used in MSCMNet [11] and AGW [31], where each batch contains $P$ identities with $K = 4$ instances per modality. We empirically find that $K = 2$ degenerates to a softmax over a single pair, while $K \geq 6$ inflates memory cost without further gain; $K = 4$ thus balances discriminability and computational cost. The adaptive weights are computed as:

$$
s _ { i } = \sum _ { j = 1 } ^ { K } \sin ( \mathbf { f } _ { i } , \mathbf { f } _ { j } ) , \quad i = 1 , \ldots , K\tag{7}
$$

$$
\mathbf { w } = \mathrm { s o f t m a x } ( \left[ s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \right] )\tag{8}
$$

where $\mathrm { s i m } ( \cdot , \cdot )$ denotes cosine similarity. The aggregated identity representation is then computed as:

$$
\mathbf { f } _ { a g g } = \sum _ { i = 1 } ^ { K } w _ { i } \mathbf { f } _ { i }\tag{9}
$$

where $w _ { i }$ represents the adaptive weight for the i-th sample within the identity group.

The adaptive identity loss enforces compactness between adaptively aggregated features from the same identity across diferent modalities:

$$
L _ { a d a p t i v e } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \| \mathbf { f } _ { a g g , i } ^ { v } - \mathbf { f } _ { a g g , i } ^ { t } \| _ { 2 }\tag{10}
$$

where $\mathbf { f } _ { a g g , i } ^ { v }$ and $\mathbf { f } _ { a g g , i } ^ { t }$ are the adaptively aggregated features for the i-th identity from visible and thermal modalities respectively, and $B$ is the number of identities in a batch.

Overall Loss Function. The complete Granularity Discriminative Loss combines all three components with appropriate weighting:

$$
\mathcal { L } _ { G D } = \lambda _ { 1 } \mathcal { L } _ { c e n t e r } + \lambda _ { 2 } \mathcal { L } _ { c o s i n e } + \lambda _ { 3 } \mathcal { L } _ { a d a p t i v e }\tag{11}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ are set to 0.1. Note that $\mathcal { L } _ { I D }$ enforces discriminative feature learning via softmax but lacks explicit cross-modal alignment; $\mathcal { L } _ { Q C T }$ pulls samples toward class centers using Euclidean distance but ignores angular information and direct cross-modal constraints. Our GDL addresses these limitations through three complementary components: $\mathcal { L } _ { c e n t e r }$ explicitly aligns cross-modal features and compensates for $\mathcal { L } _ { I D }$ ’s absence of cross-modal supervision, $\mathcal { L } _ { c o s i n e }$ introduces angular metrics beyond Euclidean distance, and $\mathcal { L } _ { a d a p t i v e }$ provides instance-level adaptive weighting beyond $\mathcal { L } _ { Q C T } \mathrm { ^ { * } s }$ fixed centers.

The total training objective combines the Identity loss, QCT loss, and GDL:

$$
\mathcal { L } _ { J D M } = ( \mathcal { L } _ { I D } + \mathcal { L } _ { Q C T } ) + \mathcal { L } _ { G D }\tag{12}
$$

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We perform comprehensive experimental evaluations using two wellestablished VI-ReID benchmark datasets: SYSU-MM01 [23] and RegDB [17].

Evaluation Metrics. In accordance with established research methodologies [21, 16], we utilize Cumulative Matching Characteristics (CMC) and mean Average Precision (mAP) as assessment criteria for our experimental evaluation. Implementation Details. MDCRNet architecture is developed utilizing Py-Torch framework and deployed on an NVIDIA A40 GPU platform, employing ResNet-50 as the foundational network structure. The HDCA module incorporates CA and three PDPA components with kernel sizes 3, 5, 7. The training process utilizes stochastic gradient descent (SGD) optimizer configured with a momentum parameter of 0.9 and regularization through weight decay set to $5 \times 1 0 ^ { - 4 }$ . The initial learning rate starts at 0.001 and increases to a maximum of 0.01, then undergoes reduction by a multiplicative factor of 0.1 at training epochs 30, 90, and 120 throughout a complete training duration of 220 epochs. Three components of GDL are weighted with 0.1.

Complexity Analysis. MDCRNet has 41.8M parameters (of which HDCA contributes 8.1M) and 20G FLOPs (HDCA contributes 1.7G). Inference speed is approximately 10ms per image on an NVIDIA A40 GPU, and end-to-end training takes about 20 hours on SYSU-MM01. These figures are comparable to recent VI-ReID baselines such as AGW and MSCMNet, confirming that the accuracy gains do not come at the cost of significant computational overhead.

## 4.2 Comparison with State-of-the-Art Methods

Our proposed MDCRNet is evaluated against contemporary state-of-the-art (SOTA) VI-ReID approaches using both SYSU-MM01 and RegDB benchmark datasets. Table 1 and Table 2 show the optimal performance obtained by the developed MDCRNet, which surpasses the second-best method on SYSU-MM01 dataset by 0.75% and 0.61% in all-search and indoor-search modes respectively, and on RegDB dataset by 1.06% and 0.44% in V-I and I-V modes, respectively. Notably, MDCRNet outperforms both Transformer-based methods (SPOT [1], PMT [15]) and CLIP-assisted approaches (CAJ [32]), validating that our hierarchical decomposed convolution design, despite being CNN-based, efectively captures cross-modal cues that more complex architectures may overlook.

## 4.3 Ablation Study

Efectiveness of Key Components. We conduct comprehensive ablation studies with diferent component combinations on SYSU-MM01 dataset. As shown in Table $^ { 3 , }$ the PDPA with multi-scale kernels efectively captures multiscale spatial dependencies, improving R-1 accuracy from 74.86% to 77.16% in all-search mode. The CA mechanism within HDCA further enhances cross-modal feature alignment, achieving 83.76% R-1 accuracy and 86.24% mAP in indoor mode. The introduction of our $\mathcal { L } _ { G D }$ significantly boosts performance by jointly optimizing $\mathcal { L } _ { c e n t e r } , \mathcal { L } _ { c o s i n e } ,$ and $\mathcal { L } _ { a d a p t i v e }$ , reaching the best mAP of 74.95% in all-search mode. To further validate the contribution of each GDL component, we conduct leave-one-out ablations on SYSU-MM01 (all-search). As shown in Table 4, removing $\mathcal { L } _ { a d a p t i v e } , \mathcal { L } _ { c e n t e r } , \mathrm { o r } \mathcal { L } _ { c o s i n e }$ individually drops the mAP from

Table 1. Comparison with SOTA methods on SYSU-MM01. R-k recognition accuracy (%) and mAP (%) are presented.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Publish</td><td colspan="3">All Search</td><td colspan="3">Indoor Search</td></tr><tr><td>R-1</td><td>R-10</td><td>mAP</td><td>R-1</td><td>R-10</td><td>mAP</td></tr><tr><td>AGW [31]</td><td>TPAMI 21</td><td>47.50</td><td>84.39</td><td>47.65</td><td>54.17</td><td>91.14</td><td>62.97</td></tr><tr><td>MPANet [25]</td><td>CVPR 21</td><td>70.58</td><td>96.21</td><td>68.24</td><td>76.74</td><td>98.21</td><td>80.95</td></tr><tr><td>SPOT [1]</td><td>TIP 22</td><td>65.34</td><td>92.73</td><td>62.25</td><td>69.42</td><td>96.22</td><td>74.63</td></tr><tr><td>CAL [24]</td><td>ICCV 23</td><td>42.40</td><td>85.00</td><td>40.70</td><td>45.90</td><td>87.60</td><td>54.30</td></tr><tr><td>PMT [15]</td><td>AAAI 23</td><td>67.53</td><td>95.36</td><td>51.86</td><td>71.66</td><td>96.73</td><td>76.52</td></tr><tr><td>DEEN [35]</td><td>CVPR 23</td><td>74.70</td><td>97.60</td><td>71.80</td><td>80.30</td><td>99.00</td><td>83.30</td></tr><tr><td>MUN [33]</td><td>ICCV 23</td><td>76.24</td><td>97.84</td><td>73.81</td><td>79.42</td><td>98.09</td><td>82.06</td></tr><tr><td>DMA [4]</td><td>TIFS 24</td><td>74.57</td><td>一</td><td>70.41</td><td>82.85</td><td>一</td><td>85.10</td></tr><tr><td>MUCĠ [36]</td><td>MM 24</td><td>68.80</td><td></td><td>65.90</td><td>77.40</td><td></td><td>81.00</td></tr><tr><td>CAJ [32]</td><td>TPAMI 23</td><td>71.48</td><td>96.23</td><td>68.15</td><td>78.36</td><td>98.36</td><td>81.98</td></tr><tr><td>MSCMNet [11]</td><td>PR 25</td><td>78.53</td><td>97.51</td><td>74.20</td><td>83.00</td><td>98.99</td><td>85.54</td></tr><tr><td>DSAF [12]</td><td>TMM 25</td><td>76.65</td><td>97.74</td><td>73.24</td><td>83.48</td><td>99.27</td><td>83.78</td></tr><tr><td>FMCNet+ [26]</td><td>TNNLS 25</td><td>75.34</td><td></td><td>71.14</td><td>77.01</td><td></td><td>84.25</td></tr><tr><td>HTCR [3]</td><td>TMM 25</td><td>74.60</td><td>-</td><td>72.11</td><td>82.99</td><td>-</td><td>85.63</td></tr><tr><td>MDCRNet</td><td>Ours</td><td>78.38</td><td>97.88</td><td>74.95</td><td>83.76</td><td>98.76</td><td>86.24</td></tr></table>

Table 2. Comparison with SOTA methods on RegDB. R-k recognition accuracy (%) and mAP (%) are presented.
<table><tr><td>Method</td><td>Publish</td><td colspan="2">V-I</td><td colspan="2">I-V</td></tr><tr><td></td><td></td><td>R-1</td><td>mAP</td><td>R-1</td><td>mAP</td></tr><tr><td>AGW [31]</td><td>TPAMI 21</td><td>70.00</td><td>66.30</td><td>70.40</td><td>65.90</td></tr><tr><td>SPOT [1]</td><td>TIP 22</td><td>80.30</td><td>72.40</td><td>79.30</td><td>72.20</td></tr><tr><td>PMT [15]</td><td>AAAI 23</td><td>84.81</td><td>76.55</td><td>84.16</td><td>75.13</td></tr><tr><td>SFANet [14]</td><td>TNNLS 21</td><td>76.31</td><td>68.00</td><td>70.20</td><td>63.80</td></tr><tr><td>LCNL [28]</td><td>IJCV 24</td><td>85.60</td><td>60.60</td><td>84.00</td><td>76.90</td></tr><tr><td>SDCL [27]</td><td>CVPR 24</td><td>86.91</td><td>78.92</td><td>85.76</td><td>77.25</td></tr><tr><td>MUCG [36]</td><td>MM 24</td><td>86.90</td><td>76.70</td><td>83.70</td><td>74.10</td></tr><tr><td>MSFCS [29]</td><td>TMM 24</td><td>85.34</td><td>76.39</td><td>83.88</td><td>75.16</td></tr><tr><td>CAJ [32]</td><td>TPAMI 23</td><td>85.69</td><td>79.70</td><td>84.88</td><td>77.80</td></tr><tr><td>CNGC [22]</td><td>ICASSP 25</td><td>86.31</td><td>79.90</td><td></td><td></td></tr><tr><td>CM2GT [8]</td><td>PR 25</td><td>86.72</td><td>77.79</td><td>86.47</td><td>77.51</td></tr><tr><td>MDCRNet</td><td>Ours</td><td>88.71</td><td>80.96</td><td>86.10</td><td>78.24</td></tr></table>

74.95% to 73.27%, 73.64%, and 73.77%, respectively, confirming that each component contributes a non-trivial and complementary gain. Importantly, while the gain over the strongest prior method appears modest (0.75%), the cumulative improvement over our own baseline is substantial: the baseline achieves 71.84% mAP on SYSU-MM01 (all-search), HDCA increases it to 73.47%, and GDL further pushes it to 74.95%, totaling 3.11% absolute improvement. On RegDB,

Table 3. Ablation study of HDCA and GDL on SYSU-MM01. R-k accuracy (%) and mAP (%) are presented.
<table><tr><td colspan="2">HDCA</td><td colspan="2"> $\mathcal { L } _ { J D M }$ </td><td colspan="3">All Search</td><td colspan="3">Indoor Search</td></tr><tr><td>CA PDPA</td><td></td><td> $( \mathcal { L } _ { I D } + \mathcal { L } _ { Q C T } )$ </td><td> $\mathcal { L } _ { G D }$ </td><td> $\mathrm { R } { - } 1$ </td><td> $\mathrm { R } { - } 1 0$ </td><td> $\mathrm { m A P }$ </td><td> $\mathrm { R } { - } 1$ </td><td>R-10</td><td> $\mathrm { m A P }$ </td></tr><tr><td></td><td></td><td>√</td><td></td><td>76.29</td><td>97.20</td><td>71.84</td><td>81.55</td><td>98.27</td><td>84.35</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>77.28</td><td>97.03</td><td>73.24</td><td>82.21</td><td>98.73</td><td>84.95</td></tr><tr><td> $\checkmark$ </td><td>√</td><td>√</td><td></td><td>76.60</td><td>97.56</td><td>73.47</td><td>83.51</td><td>98.28</td><td>85.95</td></tr><tr><td></td><td></td><td> $\checkmark$ </td><td>√</td><td>74.86</td><td>97.02</td><td>71.41</td><td>81.20</td><td>98.76</td><td>84.47</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>77.16</td><td>97.48</td><td>73.54</td><td>82.62</td><td>98.42</td><td>85.34</td></tr><tr><td>√</td><td>√</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>78.38</td><td>97.88</td><td>74.95</td><td>83.76</td><td>98.76</td><td>86.24</td></tr></table>

Table 4. Leave-one-out ablation of GDL components on SYSU-MM01 (all-search).
<table><tr><td>Configuration</td><td> $\mathrm { m A P }$  (%)</td></tr><tr><td>Full GDL</td><td>74.95</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { a d a p t i v e }$ </td><td>73.27</td></tr><tr><td>w/o Lcenter</td><td>73.64</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ \mathcal { L } _ { c o s i n e }$ </td><td>73.77</td></tr></table>

V→I mAP improves from 80.37% to 80.96% and I→V from 77.66% to 78.24%, demonstrating that the gains stem from the proposed modules rather than from hyperparameter tuning.

Hyperparameter Analysis. To assess the impact of key hyperparameters in MDCRNet, we conduct systematic experiments on the SYSU-MM01 dataset under all-search mode. We evaluate the number of PDPA modules, namely $i ,$ and the weighting coeficients for $\mathcal { L } _ { G D }$ components. As shown in Fig. 3, optimal performance is achieved with 3 PDPA modules and balanced weighting of $\lambda _ { 1 } { = } 0 . 1$ $\lambda _ { 2 } { = } 0 . 1 , \lambda _ { 3 } { = } 0 . 1$ for the $\mathcal { L } _ { G D }$ components. Analysis reveals that using 3 PDPA modules efectively captures multi-scale dependencies without introducing excessive computational overhead, while the uniform 0.1 weighting maintains optimal balance between intra-identity compactness and inter-identity separation across modalities. Sensitivity analysis further shows that $\lambda _ { 1 } < 0 . 0 8$ causes insuficient cross-modal alignment, while $\lambda _ { 1 } > 0 . 1 2$ over-compresses features and harms discriminability; $\lambda _ { 2 }$ remains stable across a wide range; $\lambda _ { 3 } > 0 . 1 2$ leads to a sharp decline due to over-regularization that suppresses feature diversity.

Notably, performance drops when only two PDPA modules are used. This validates our multi-scale design: incomplete hierarchies (short + medium only) sufer from feature imbalance and lack critical long-range semantic information, confirming that all three distance scales are necessary for comprehensive crossmodal feature extraction.

Visualization Analysis. As illustrated in Fig. 4, our MDCRNet demonstrates superior retrieval performance with significantly more correct matches compared to the baseline method.

![](images/3e37571072c827cb12beb103757999ef27d2fce226f5e6191c6b1e562d8e3948.jpg)

![](images/7dbf8047fc2ed7a2c3680ff9c5a788ddf01b3d658a4f4bb7d630e6f6304983aa.jpg)

![](images/62667dab526c324c7b65c66405b9b3424cc652a767bdd4167234efece99dcf19.jpg)

![](images/41cdca4c9c532e0bb978447748890e54d6768f77d0d802f40f55f4c1cc6394c0.jpg)  
Fig. 3. Ablation study for the number of PDPA module (i) in Eq. 2 and hyperparameter analysis of λ<sub>1</sub>, λ<sub>2</sub> and λ<sub>3</sub> in Eq. 11.

Query  
Rank 10 ranking  
![](images/d2398292a7d92d7ffa602ad612ddf9cdbebea2d231e5a49d92eacf4329199ebf.jpg)

![](images/f0957e4b97969479db7bb5c8794eaa195ed1fc90793b4d6ea11a72893754490b.jpg)  
Fig. 4. Visualization of R10 retrieval results for visible-infrared images on SYSU-MM01. Red for wrong and green for correct matches.

Stability Analysis. To verify that our improvements are not artifacts of random seed variation, we run MDCRNet 5 times on SYSU-MM01 (all-search) with diferent random seeds. The mean mAP is 74.95% with a standard deviation of ±0.10%, confirming that the reported gains are consistent and statistically meaningful.

## 5 Conclusion

This paper proposes MDCRNet that efectively mitigates cross-modal discrepancies through hierarchical architectural design and discriminative metric learning. Our approach introduces the HLM with eficient CA and PDPA to capture multi-scale cross-modal semantic information across short, medium, and longrange spatial dependencies. Moreover, we develop the JDML that synergistically combines identity loss, QCT loss, and our novel GDL to enforce fine-grained feature discrimination across diferent modalities and identities. Extensive experiments on SYSU-MM01 and RegDB datasets demonstrate the superiority of our method, achieving 0.75% and 1.06% mAP gains over the strongest competing methods on SYSU-MM01 and RegDB respectively, as well as a cumulative 3.11% mAP improvement over our own baseline on SYSU-MM01. In future work, we will explore incorporating skeletal information extraction and multi-granularity semantic guidance to further enhance cross-modal refinement capability.

Acknowledgments. This work was supported in part by the National Natural Science Foundation of China under Grants 62562058 and 62441213, the Key R&D Program of Xinjiang Uygur Autonomous Region under Grant 2022B01046, and the Sichuan Province Joint Fund Project of China under Grant 25QYCX0103.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Chen, C., Ye, M., Qi, M., Wu, J., Jiang, J., Lin, C.W.: Structure-aware positional transformer for visible-infrared person re-identification. IEEE Transactions on Image Processing 31, 2352–2364 (2022)

2. Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L.: DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence 40(4), 834–848 (2018)

3. Chen, S., Qiu, L., Wang, D.H., Zhu, W., Hua, Y., Yan, Y.: Hierarchical tokenaware cross-modality reconstruction for visible-infrared person re-identification. IEEE Transactions on Multimedia (2025)

4. Cui, Z., Zhou, J., Peng, Y.: Dma: Dual modality-aware alignment for visibleinfrared person re-identification. IEEE Transactions on Information Forensics and Security 19, 2696–2708 (2024)

5. Dong, M., Cai, H., Lei, X., Li, J., Zhang, T., Pu, M.: Once-For-All: A train-once and select-anytime framework for multimodal instruction tuning. arXiv preprint arXiv:2605.26761 (2026)

6. Dong, M., Cai, H., Li, J., Zhou, S., Ren, B., Peng, K., Fu, Y.: VisNec: Measuring and leveraging visual necessity for multimodal instruction tuning. In: European Conference on Computer Vision (2026)

7. Dong, M., Pu, M., Li, J., Guo, B., Chen, S., Ren, B., Zheng, X., Zhao, C., Qian, T., Elhoseiny, M., Fu, Y.: ObjectStream: Latent objects as memory anchors for streaming video understanding. arXiv preprint arXiv:2607.28312 (2026)

8. Feng, Y., Chen, F., Sun, G., Wu, F., Ji, Y., Liu, T., Liu, S., Jing, X.Y., Luo, J.: Learning multi-granularity representation with transformer for visible-infrared person re-identification. Pattern Recognition 164, 111510 (2025)

9. Feng, Y., Chen, F., Yu, J., Ji, Y., Wu, F., Liu, T., Liu, S., Jing, X.Y., Luo, J.: Crossmodality spatial-temporal transformer for video-based visible-infrared person reidentification. IEEE Transactions on Multimedia 26, 6582–6594 (2024)

10. Gao, Y., Liang, T., Jin, Y., Gu, X., Liu, W., Li, Y., Lang, C.: Mso: Multi-feature space joint optimization network for rgb-infrared person re-identification. In: Proceedings of the 29th ACM international conference on multimedia. pp. 5257–5265 (2021)

11. Hua, X., Cheng, K., Lu, H., Tu, J., Wang, Y., Wang, S.: Mscmnet: Multi-scale semantic correlation mining for visible-infrared person re-identification. Pattern Recognition 159, 111090 (2025)

12. Jiang, Y., Cheng, X., Yu, H., Liu, X., Chen, H., Zhao, G.: Dsaf: Dual space alignment framework for visible-infrared person re-identification. IEEE Transactions on Multimedia (2025)

13. Li, J., Cai, H., Dong, M., Pu, M., You, S., Wang, F., Huang, T.: Pistachio: Towards synthetic, balanced, and long-form video anomaly benchmarks. In: European Conference on Computer Vision (2026)

14. Liu, H., Ma, S., Xia, D., Li, S.: Sfanet: A spectrum-aware feature augmentation network for visible-infrared person reidentification. IEEE Transactions on Neural Networks and Learning Systems 34(4), 1958–1971 (2021)

15. Lu, H., Zou, X., Zhang, P.: Learning progressive modality-shared transformers for efective visible-infrared person re-identification. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 1835–1843 (2023)

16. Lu, Z., Lin, R., Hu, H.: Tri-level modality-information disentanglement for visibleinfrared person re-identification. IEEE Transactions on Multimedia 26, 2700–2714 (2023)

17. Nguyen, D.T., Hong, H.G., Kim, K.W., Park, K.R.: Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors 17(3), 605 (2017)

18. Qiu, L., Chen, S., Yan, Y., Xue, J.H., Wang, D.H., Zhu, S.: High-order structure based middle-feature learning for visible-infrared person re-identification. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 4596–4604 (2024)

19. Radenović, F., Tolias, G., Chum, O.: Fine-tuning cnn image retrieval with no human annotation. IEEE transactions on pattern analysis and machine intelligence 41(7), 1655–1668 (2018)

20. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

21. Wang, G., Zhang, T., Cheng, J., Liu, S., Yang, Y., Hou, Z.: Rgb-infrared crossmodality person re-identification via joint pixel and feature alignment. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3623–3632 (2019)

22. Wang, Z., Zhao, W., Dong, Y.: Learning with coupled noisy labels for visibleinfrared person re-identification via graph consistency. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)

23. Wu, A., Zheng, W.S., Yu, H.X., Gong, S., Lai, J.: Rgb-infrared cross-modality person re-identification. In: Proceedings of the IEEE international conference on computer vision. pp. 5380–5389 (2017)

24. Wu, J., Liu, H., Su, Y., Shi, W., Tang, H.: Learning concordant attention via targetaware alignment for visible-infrared person re-identification. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11122–11131 (2023)

25. Wu, Q., Dai, P., Chen, J., Lin, C.W., Wu, Y., Huang, F., Zhong, B., Ji, R.: Discover cross-modality nuances for visible-infrared person re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4330–4339 (2021)

26. Xi, R., Huang, N., Lai, C., Zhang, Q., Han, J.: Fmcnet +: Feature-level modality compensation for visible-infrared person re-identification. IEEE Transactions on Neural Networks and Learning Systems (2024)

27. Yang, B., Chen, J., Ye, M.: Shallow-deep collaborative learning for unsupervised visible-infrared person re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16870–16879 (2024)

28. Yang, M., Huang, Z., Peng, X.: Robust object re-identification with coupled noisy labels. International Journal of Computer Vision 132(7), 2511–2529 (2024)

29. Yang, X., Dong, W., Li, M., Wei, Z., Wang, N., Gao, X.: Cooperative separation of modality shared-specific features for visible-infrared person re-identification. IEEE Transactions on Multimedia 26, 8172–8183 (2024)

30. Ye, M., Ruan, W., Du, B., Shou, M.Z.: Channel augmented joint learning for visible-infrared recognition. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 13567–13576 (2021)

31. Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., Hoi, S.C.: Deep learning for person re-identification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence 44(6), 2872–2893 (2021)

32. Ye, M., Wu, Z., Chen, C., Du, B.: Channel augmentation for visible-infrared reidentification. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(4), 2299–2315 (2023)

33. Yu, H., Cheng, X., Peng, W., Liu, W., Zhao, G.: Modality unifying network for visible-infrared person re-identification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 11185–11195 (2023)

34. Yu, X., Dong, N., Zhu, L., Peng, H., Tao, D.: Clip-driven semantic discovery network for visible-infrared person re-identification. IEEE Transactions on Multimedia (2025)

35. Zhang, Y., Wang, H.: Diverse embedding expansion network and low-light crossmodality benchmark for visible-infrared person re-identification. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2153– 2162 (2023)

36. Zheng, X., Zhang, Y., Lu, Y., Wang, H.: Semi-supervised visible-infrared person re-identification via modality unification and confidence guidance. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 5761–5770 (2024)

37. Zheng, Z., Zheng, L., Yang, Y.: A discriminatively learned cnn embedding for person reidentification. ACM transactions on multimedia computing, communications, and applications (TOMM) 14(1), 1–20 (2017)