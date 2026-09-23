![](images/b054a5e3cc7d4dc514eee370e3d652618a39e2fab3de15721f87efcbd7bea6e3.jpg)  
(b) Architectural Comparison

# When Point Clouds Outperform Pixels: Rethinking Zero-Shot Multimodal Anomaly Detection

Chenglin Ye<sup>1∗</sup>, Lupeng Liu<sup>1∗</sup>, Dongbo Yu<sup>1</sup>, Jun Xiao<sup>1†</sup>, Yunbiao Wang<sup>1†</sup>

<sup>1</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences yechenglin24@mails.ucas.ac.cn, {liulupeng, yudongbo, xiaojun, wangyunbiao}@ucas.ac.cn

## Abstract

Zero-shot multimodal anomaly detection commonly assumes that RGB and point cloud modalities are equally reliable and can contribute uniformly to anomaly localization. We challenge this assumption. Using a set of recently proposed stringent metrics that penalize false anomaly responses in normal regions, we find that point clouds are substantially more reliable than RGB under zero-shot category shift. Motivated by this observation, we propose WOOPS (When Point Clouds Outperform Pixels), a reliability-aware zero-shot multimodal anomaly detection framework. To strengthen the more reliable geometric modality, we design a Multi-view Information Decoupling module to suppress heterogeneous information from multi-view point cloud projections and enhance point cloud feature quality. To avoid unconditional fusion, we further introduce a Modality Reliability Calibration module to adaptively calibrate modality contributions according to their reliability. Extensive experiments show that our method achieves the best or competitive performance under the new metrics in both unimodal and multimodal settings. Further analysis demonstrates that point cloud information also improves RGB-only inference, while ablations verify the efectiveness of both modules. Code will be released upon acceptance.

## Introduction

Anomaly Detection (AD) aims to identify and localize anomalous characteristics or defects in target objects, playing a pivotal role in applications such as industrial quality inspection(Shukla et al. 2025; Zhu et al. 2025; Guo et al. 2025; Fang et al. 2025) and medical diagnostics(Liu et al. 2023; Cai et al. 2025; Gu et al. 2026). Zero-shot AD further considers a more practical setting where test categories are unseen during training, requiring models to learn transferable notions of normality across diverse objects. Compared with conventional settings, this imposes higher demands on representation generalization, modality robustness, and response calibration.

RGB and point cloud data provide complementary information: RGB captures texture and appearance, while point clouds encode geometric structure. Recent zero-shot multimodal methods leverage both by projecting point clouds into multiple views and extracting features via image encoders(Zhou et al. 2024b; Ma et al. 2026; Deng, Liu, and Wang 2026), often with vision-language models such as

Conventional Metrics:

![](images/ef9e2589804d70e4a6b0a984984dc786e04f309840d25c04f19d946a9a842030.jpg)  
Stricter New Metrics:

![](images/1cba8a574d0119044dbfa51d0e66e7830d46488ad4f833d860f0f3d10cb3c87f.jpg)  
Previous Work:  
(a) Revisiting Zero-Shot AD  
Ours:

Figure 1: (a) We illustrate results using P-AUROC as a representative of conventional metrics(Bergmann et al. 2022) col(Zhang et al. 2026). Please zoom in for details. (b) Comparison of network architectures between our model and existing zero-shot multimodal anomaly detection methods.

CLIP(Radford et al. 2021). However, multi-view projections introduce heterogeneous information arising from variations in rendering viewpoints, such as shape discrepancies caused by self-occlusion or viewpoint divergence. Direct aggregation of such views can degrade point cloud representations and weaken geometric anomaly responses, especially under zero-shot category shifts where robust geometry is critical. Another issue is that the reliability of each modality in zeroshot AD remains unclear. Existing methods often assume that combining modalities consistently improves performance, but this may not hold in zero-shot settings. RGB responses are sensitive to appearance variations and can produce false positives in normal regions(Bergmann et al. 2022). Without modeling modality reliability, fusing such responses with point cloud outputs may amplify errors.

In this work, we first revisit zero-shot AD from the perspective of modality reliability under a stricter evaluation protocol (Zhang et al. 2026). Diferent from conventional metrics(Bergmann et al. 2022) that mainly focus on anomalous areas, the newly introduced evaluation criterion places stronger emphasis on suppressing anomaly responses in normal regions. As presented in Fig. 1, we discover that point cloud-only inference can significantly outperform RGB-only inference, suggesting that geometry plays a dominant role in cross-category generalization, while RGB may introduce unreliable responses. Moreover, training with point cloud data improves performance even for RGB-only inference, highlighting its contribution to transferable representations.

Considering the existing issues and the above experimental analysis, we propose WOOPS (When Point Clouds Outperform Pixels), a geometry-centric and reliability-aware zero-shot multimodal AD framework. Specifically, we propose a Multi-view Information Decoupling (MID) module to alleviate the interference caused by heterogeneous information across diferent views, thereby enhancing the quality of point cloud features. Furthermore, we design a Modality Reliability Calibration (MRC) module that dynamically calibrates modality contributions according to their reliability, suppressing unstable activations and improving crossdomain robustness.

Extensive experiments on widely adopted benchmarks demonstrate that our method achieves the best or competitive performance under the stricter anomaly evaluation metrics. Moreover, comprehensive analyses show that point cloud data benefits RGB inference, MID improves detection performance, and MRC enhances robustness. Although a trade-of exists under conventional metrics, our method significantly reduces false positives under stricter evaluation, which is crucial for practical anomaly localization.

Our contributions are summarized as follows:

• We revisit zero-shot multimodal AD under a stricter evaluation protocol and reveal that modality reliability is highly imbalanced under zero-shot category shift, where point cloud representations provide more transferable anomaly cues than RGB features.

• We propose the Multi-view Information Decoupling (MID) module, a point cloud feature enhancement module that mitigates heterogeneous information from multiview projections and improves geometric anomaly representation.

• We develop the Modality Reliability Calibration (MRC) module, a modality response balancing module that explicitly models modality reliability and improves robustness across datasets and modality settings.

• Extensive experiments under a newly proposed stringent evaluation protocol show that WOOPS achieves state-ofthe-art performance in both unimodal and multimodal settings on two widely adopted multimodal AD benchmarks.

## Related Work

## Multimodal Anomaly Detection

Multimodal anomaly detection typically follows unsupervised or self-supervised paradigms (Tu et al. 2025; Wang et al. 2025; Li et al. 2025) and can be broadly grouped into three categories. The first line is feature-embedding-based methods (Horwitz and Hoshen 2023; Gu et al. 2024; Rudolph et al. 2023), which characterize the distribution or decision boundary of normal samples and treat deviations therefrom as anomalies. For instance, LPFSTNet(Cheng et al. 2025) and AST(Rudolph et al. 2023) employ teacher–student networks to model the normal distribution, while M3DM(Wang et al. 2023) and M3DM-NR(Wang et al. 2025) delimit normality with a memory bank. The second line is reconstruction-based methods(ZHAO 2025; Zavrtanik, Kristan, and Skočaj 2024; Liu et al. 2025; Costanzino et al. 2024), which leverage the prior that a model trained exclusively on normal data fails to reconstruct anomalous regions, and thus localize anomalies via reconstruction error. Representative works such as CFM(Costanzino et al. 2024) and MODMAP(Costanzino et al. 2026) reconstruct features through cross-modal mapping and identify anomalies from the discrepancy between reconstructed and extracted features. The third line consists of discriminative methods that perform self-supervision by synthesizing pseudo-anomalous samples(Asad et al. 2025; Tu et al. 2025; Wang, Niu, and Huang 2025; Chen et al. 2023; Xiang et al. 2025). Nevertheless, these approaches struggle to generalize to unseen objects, rendering them less applicable under privacy-sensitive (e.g., industrial confidentiality) or data-scarce (e.g., brand-new products) scenarios.

## Zero-Shot Anomaly Detection

Zero-shot anomaly detection exploits the open-world knowledge encapsulated in vision–language models(Radford et al. 2021; Xue et al. 2023, 2024) to generalize to unseen objects. While zero-shot 2D anomaly detection has matured considerably(Zhou et al. 2024a; Cao et al. 2025; Jeong et al. 2023; Hou et al. 2026; Chen et al. 2026; Zhu et al. 2025), zero-shot multimodal anomaly detection remains underexplored, with two notable gaps. The first concerns point cloud feature extraction. Since CLIP cannot directly consume point clouds, a common workaround is to render the point cloud into multiview images(Zhou et al. 2024b; Ma et al. 2026; Deng, Liu, and Wang 2026) and treat the multi-view image features as a surrogate for point-cloud representation. However, varying rendering views induce inconsistent information across views, degrading the quality of the resulting point-cloud features. Although BTP(Li et al. 2026) and MuSc-V2(Li, Xue, and Zhou 2026) explore large models natively supporting point-cloud modality, their deployment overhead limits practical applicability. The second gap lies in the lack of modality reliability modeling. Existing zero-shot multimodal methods predominantly assume equal reliability across all modalities (Zhou et al. 2024b; Ma et al. 2026), aggregating multimodal responses via simple averaging. This assumption undermines robustness when anomalies are only discernible from a specific modality.

![](images/f03cfa0fe077bcbf097e98bc6fc9c3a07cb6d6efda607801372bd061a655f65a.jpg)  
Figure 2: The pipeline of WOOPS. WOOPS employs a frozen CLIP vision encoder $\Psi _ { \mathrm { v } }$ to extract RGB and point cloud features from images and multi-view projections, respectively, while utilizing a frozen CLIP text encoder $\Psi _ { \mathrm { t } }$ to derive text features from learnable normal and abnormal language vectors $( \mathbf { P } _ { n }$ and $\mathbf { P } _ { a } ) .$ Anomaly responses are computed via vision-language similarity. Furthermore, we propose a Multi-view Information Decoupling (MID) module to enhance point cloud features and a Modalit Reliability Calibration (MRC) module to efectively calibrate anomaly responses across modalities.

## Methodology

## Overview

We investigate zero-shot multimodal AD utilizing RGB images and point clouds.

$\mathrm { L e t } \mathfrak { D } _ { t r } \ = \ \{ \mathrm { O } _ { i } ^ { t r } \ = \ ( \mathrm { I } _ { i } , \mathrm { P } _ { i } , \mathrm { I } _ { i } ^ { g t } , \mathrm { P } _ { i } ^ { g t } ) \} _ { i = 1 } ^ { N _ { t r } }$ and $\mathfrak { D } _ { t e } ~ =$ $\{ \mathrm { O } _ { i } ^ { t e } = ( \mathrm { I } _ { i } , \mathrm { P } _ { i } , \mathrm { I } _ { i } ^ { g t } , \mathrm { P } _ { i } ^ { g t } ) \} _ { i = 1 } ^ { N _ { t e } }$ denote the training and test sets, respectively, where $\bar { \mathbf { I } } \in \mathbb { R } ^ { H \times W } , \mathrm { P } \in \mathbb { R } ^ { N _ { p c } \times 3 }$ $\mathrm { I } ^ { g t } \in \mathrm { \Omega }$ $\mathbb { R } ^ { H \times W }$ , and $\mathrm { P } ^ { g \bar { t } } \in \mathbb { R } ^ { N _ { p c } \times 1 }$ correspond to the RGB image, point cloud, RGB ground truth, and point cloud ground truth. Crucially, the object categories in the training and test sets are entirely disjoint.

Given a test sample, the goal is to produce an anomaly response map S that assigns high responses to abnormal regions and suppresses responses in anomaly-free regions.

Our framework, denoted as WOOPS (When Point Clouds Outperform Pixels), contains four core components: Input Feature Extraction, Heterogeneity-Aware Point Cloud Feature Enhancement, Anomaly Response and Reliability-Aware Modality Response Balancing. The overall pipeline is shown in Fig. 2. Details about each component are as follows.

## Input Feature Extraction

To extract point cloud features, we adopt the prevalent multiview projection paradigm commonly used in zero-shot multimodal methods(Zhou et al. 2024b; Ma et al. 2026; Deng, Liu, and Wang 2026). Specifically, given the point cloud input P and its corresponding ground truth $\mathrm { P } ^ { g t }$ for the training sample $( \mathrm { I } , \mathrm { P } , \mathrm { I } ^ { g t } , \dot { \mathrm { P } ^ { g t } } ) \in \dot { \mathfrak { D } } _ { t r }$ , we first render them into $\bar { K }$ projected views:

$$
\mathrm { R } _ { k } = \Pi _ { k } ( \mathrm { P } ) , \mathrm { Y } _ { k } = \Pi _ { k } ( \mathrm { P } ^ { g t } ) , k = 1 , \dots , K ,\tag{1}
$$

where $\Pi _ { k } ( \cdot )$ denotes the rendering matrix for the k-th view, while $\mathrm { R } _ { k } \in \mathbb { R } ^ { H \times W }$ and $\mathrm { Y } _ { k } \in \mathbb { R } ^ { H \times W }$ represent the resulting 2D rendered image and its corresponding pixel-level ground truth, respectively. For detailed information, please refer to PointAD(Zhou et al. 2024b).

Subsequently, features of I and $\mathrm { R } _ { k }$ are extracted using the visual encoder $\Psi _ { \mathbf { v } }$ of CLIP(Radford et al. 2021) to serve as inputs for the subsequent network. This process is formulated as follows:

$$
\mathcal { G } _ { \mathrm { I } } , \{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \} _ { m = 1 } ^ { M } = \Psi _ { \mathrm { v } } ( \mathrm { I } ) ,\tag{2}
$$

$$
\mathcal { G } _ { k } , \{ \mathcal { L } _ { k } ^ { ( m ) } \} _ { m = 1 } ^ { M } = \Psi _ { \mathrm { v } } ( \mathrm { R } _ { k } ) , \quad k = 1 , 2 , \hdots , K ,\tag{3}
$$

where I is the RGB image input; $\mathcal { G } _ { \mathrm { I } } \in \mathbb { R } ^ { 1 \times D }$ and $\{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \} _ { \mathrm { ~ C ~ } }$ $\mathbb { R } ^ { 1 \times D }$ are the global feature and local patch features of I, respectively; $\mathcal { G } _ { k } \in \mathbb { R } ^ { 1 \times D }$ and $\{ \mathcal { L } _ { k } ^ { ( m ) } \} \subset \mathbb { R } ^ { 1 \times D }$ are those of $\mathrm { R } _ { k } ; D$ represents the feature dimension; and $\Psi _ { \mathrm { v } }$ denotes the frozen CLIP visual encoder.

## Heterogeneity-Aware Point Cloud Feature Enhancement

Beyond discriminative geometric structures, each projected view inherently contains heterogeneous information arising from variations in rendering viewpoints. To address this issue, we propose Multi-view Information Decoupling (MID) module, a heterogeneity-aware point cloud feature enhancement module. The objective of MID module is to suppress unreliable view-specific information and emphasize geometrically consistent cues across projections. MID module operates between randomly sampled views and the aggregated statistics of all other views. The core idea is to decompose each view’s local features into two complementary subspaces: (i) a homogeneous subspace, which encodes shared geometric structures preserved across viewpoints, and (ii) a heterogeneous subspace, which captures view-specific irrelevant information.

Specifically, we first randomly sample X views from the K available projection views, denoted as $\begin{array} { r l } { { \mathcal { R } } } & { { } = } \end{array}$ $\{ R _ { x _ { 1 } } , R _ { x _ { 2 } } , \dots , R _ { x _ { X } } \}$ . The average global feature $\bar { \mathcal { G } }$ and average local patch features $\{ \bar { \mathcal { L } } ^ { ( m ) } \}$ of these sampled views are calculated as follows:

$$
\bar { \mathcal { G } } = \frac { 1 } { X } \sum _ { R _ { i } \in \mathcal { R } } \mathcal { G } _ { i } ,\tag{4}
$$

$$
\bar { \mathcal { L } } ^ { ( m ) } = \frac { 1 } { X } \sum _ { R _ { i } \in \mathcal { R } } \mathcal { L } _ { i } ^ { ( m ) } , \quad m = 1 , 2 , \dotsc , M .\tag{5}
$$

Similarly, the average global feature $\bar { \mathcal { G } } _ { A }$ and average local patch features $\bar { \mathcal { L } _ { A } } ^ { ( m ) }$ for the remaining views are obtained using the same procedure. To extract homogeneous and heterogeneous features from both the sampled subset and the remaining views, we employ a shared homogeneous subspace encoder $\Phi _ { o }$ along with two independent heterogeneous subspace encoders $\Phi _ { e _ { 1 } }$ and $\Phi _ { e _ { 2 } } .$ . All three encoders share an identical architecture, implemented as two-layer perceptrons activated by the ReLU function, with a hidden dimension of D. This process is formulated as follows:

$$
\begin{array} { r l } { { \mathcal { H } } _ { G } ^ { o } = \Phi _ { o } ( \bar { \mathcal { G } } ) , } & { { \mathcal { H } } _ { G } ^ { e } = \Phi _ { e _ { 1 } } ( \bar { \mathcal { G } } ) , } \\ { { \mathcal { H } } _ { A G } ^ { o } = \Phi _ { o } ( \bar { \mathcal { G } } _ { A } ) , } & { { \mathcal { H } } _ { A G } ^ { e } = \Phi _ { e _ { 2 } } ( \bar { \mathcal { G } } _ { A } ) , } \\ { { \mathcal { H } } _ { L } ^ { o ( m ) } = \Phi _ { o } ( \bar { \mathcal { L } } ^ { ( m ) } ) , } & { { \mathcal { H } } _ { L } ^ { e ( m ) } = \Phi _ { e _ { 1 } } ( \bar { \mathcal { L } } ^ { ( m ) } ) , } \\ { { \mathcal { H } } _ { A L } ^ { o ( m ) } = \Phi _ { o } ( \bar { \mathcal { L } _ { A } } ^ { ( m ) } ) , } & { { \mathcal { H } } _ { A L } ^ { e ( m ) } = \Phi _ { e _ { 2 } } ( \bar { \mathcal { L } _ { A } } ^ { ( m ) } ) . } \end{array}\tag{6}
$$

Here, $m = 1 , 2 , \ldots , M ; \mathcal { H } _ { G } ^ { o }$ and $\mathcal { H } _ { G } ^ { e }$ denote the global homogeneous feature and global heterogeneous feature of the sampled views, respectively; $\mathcal { H } _ { L } ^ { o ( m ) }$ and $\mathcal { H } _ { L } ^ { e ( m ) }$ represent the local patch homogeneous features and local patch heterogeneous features of the sampled views, respectively. Correspondingly, $\mathcal { H } _ { A G } ^ { o }$ and $\mathcal { H } _ { A G } ^ { e }$ denote the global homogeneous and heterogeneous features of the remaining views, while $\mathcal { H } _ { A L } ^ { o ( m ) }$ and $\bar { \mathcal { H } } _ { A L } ^ { e ( m ) }$ refer to the local patch homogeneous and heterogeneous features of the remaining views.

To ensure suficient disentanglement between homogeneous and heterogeneous features, we design a decoupling loss $L _ { d e c }$ . To avoid prohibitive computational costs, instead of modeling distributions or calculating mutual information, we employ simple yet efective cosine similarity to quantify their potential overlap. The decoupling process is formulated as follows:

$$
\begin{array} { r l } & { \displaystyle { L _ { d e c } = \frac { \mathcal { H } _ { G } ^ { o } \cdot \mathcal { H } _ { G } ^ { e ^ { \top } } } { \| \mathcal { H } _ { G } ^ { o } \| \cdot \| \mathcal { H } _ { G } ^ { e } \| } + \frac { \mathcal { H } _ { A G } ^ { o } \cdot \mathcal { H } _ { A G } ^ { e ^ { \top } } } { \| \mathcal { H } _ { A G } ^ { o } \| \cdot \| \mathcal { H } _ { A G } ^ { e } \| } } } \\ & { \quad \quad \quad + \displaystyle { \sum _ { m = 1 } ^ { M } \frac { \mathcal { H } _ { L } ^ { o ( m ) } \cdot \mathcal { H } _ { L } ^ { e ( m ) ^ { \top } } } { \| \mathcal { H } _ { L } ^ { o ( m ) } \| \cdot \| \mathcal { H } _ { L } ^ { e ( m ) } \| } } } \\ & { \quad \quad \quad + \displaystyle { \sum _ { m = 1 } ^ { M } \frac { \mathcal { H } _ { A L } ^ { o ( m ) } \cdot \mathcal { H } _ { A L } ^ { e ( m ) ^ { \top } } } { \| \mathcal { H } _ { A L } ^ { o ( m ) } \| \cdot \| \mathcal { H } _ { A L } ^ { e ( m ) } \| } } . } \end{array}\tag{7}
$$

Inspired by DecAlign(Qian et al. 2026), we leverage Gaussian distributions to model the feature spaces, aiming to facilitate the alignment between the global homogeneous features $\mathcal { H } _ { G } ^ { o }$ and $\mathsf { \bar { \mathcal { H } } } _ { A G } ^ { o }$ . This strategy is designed to strengthen the encoder $\Phi _ { o } \mathbf { \bar { s } }$ ability to capture geometrically consistent information. Concretely, we approximate the distributions of $\mathcal { H } _ { G } ^ { o }$ and $\mathcal { H } _ { A G } ^ { o }$ as $\mathcal { N } _ { 1 } ( \dot { \mu } _ { 1 } , \Sigma _ { 1 } )$ ) and $\mathcal { N } _ { 2 } ( \mu _ { 2 } , \Sigma _ { 2 } )$ , respectively, with $\mu$ and $\dot { \Sigma }$ representing the mean and variance. The calculation formulas for $\mu$ and $\Sigma$ are provided in the Supplement. To enforce similarity between these two distributions, we employ a distribution consistency loss $L _ { d c }$ and a Gaussian kernel-based Maximum Mean Discrepancy (GMMD) loss $L _ { g m m d }$ . These are defined as follows:

$$
L _ { d c } = \| \mu _ { 1 } - \mu _ { 2 } \| ^ { 2 } + \| \Sigma _ { 1 } - \Sigma _ { 2 } \| _ { F } ^ { 2 } ,\tag{8}
$$

$$
\begin{array} { r l } & { L _ { g m m d } = \mathbb { E } _ { { \mathbf { f } } , { \mathbf { f } ^ { \prime } } \sim \mathcal { N } _ { 1 } } [ k ( { \mathbf { f } } , { \mathbf { f } ^ { ' } } ) ] + \mathbb { E } _ { { \mathbf { g } } , { \mathbf { g } ^ { ' } } \sim \mathcal { N } _ { 2 } } [ k ( { \mathbf { g } } , { \mathbf { g } ^ { ' } } ) ] } \\ & { ~ - ~ 2 \mathbb { E } _ { { \mathbf { f } } \sim \mathcal { N } _ { 1 } , { \mathbf { g } } \sim \mathcal { N } _ { 2 } } [ k ( { \mathbf { f } } , { \mathbf { g } } ) ] . } \end{array}\tag{9}
$$

The aforementioned three loss functions collectively constitute the point cloud feature enhancement loss $L _ { f e } \colon$

$$
L _ { f e } = L _ { d e c } + L _ { d c } + L _ { g m m d } .\tag{10}
$$

For simplicity, we do not introduce individual weighting hyperparameters for each loss term.

Ultimately, the homogeneous subspace encoder is employed to decouple the features derived from the multi-view projections of the point cloud. The resulting homogeneous features $\hat { \mathcal { G } } _ { k }$ and $\{ \hat { \mathcal { L } } _ { k } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ correspond to the enhanced global point cloud feature and the enhanced local patch feature, respectively, which are formulated as follows:

$$
\hat { \mathcal { G } } _ { k } = \Phi _ { o } ( \mathcal { G } _ { k } ) , \quad \hat { \mathcal { L } } _ { k } ^ { ( m ) } = \Phi _ { o } ( \mathcal { L } _ { k } ^ { ( m ) } ) ,\tag{11}
$$

where $k = 1 , 2 , \dots , K$ and $m = 1 , 2 , \ldots , M$

## Anomaly Response Map

Inspired by prior works (Zhou et al. 2022, 2024a), we introduce two semantic-complementary learnable language vectors $\mathbf { P } _ { n }$ and ${ \bf P } _ { a }$ of length l to store generalizable normal and anomalous semantics. To prevent overfitting to the training categories, we adopt a category-agnostic initialization strategy(Zhou et al. 2024b). Using the CLIP text encoder $\Psi _ { \mathrm { t } }$ we extract the corresponding text features ${ \mathcal { F } } _ { n }$ and ${ \mathcal { F } } _ { a } .$ . This process is formulated as:

$$
\mathcal { F } _ { n } = \Psi _ { \mathrm { t } } ( \mathbf { P } _ { n } ) , \quad \mathcal { F } _ { a } = \Psi _ { \mathrm { t } } ( \mathbf { P } _ { a } )\tag{12}
$$

The anomaly response map $\mathrm { S }$ is generated by computing the similarity between local patch features and text features for both image and point cloud modalities. For the RGB input, we first calculate the cosine similarity between $\{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \}$ and $\mathcal { F } _ { n } , \mathcal { F } _ { a }$ , followed by Softmax normalization and upsampling to obtain the anomaly score map $\mathrm { A } _ { r q b } \in \mathbb { R } ^ { H \times \dot { W } }$ and the anomaly response map $\mathrm { S } _ { r g b } \in \mathbb { R } ^ { H \times W }$ , formulated as:

$$
\mathrm { A } _ { r g b } = \frac { \exp ( < \{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \} , \mathcal { F } _ { a } > / \tau ) } { \sum _ { i \in \{ n , a \} } \exp ( < \{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \} , \mathcal { F } _ { i } > / \tau ) }\tag{13}
$$

(14)

where $m = 1 , 2 , \ldots , M ; \tau$ is the temperature hyperparameter of $\mathrm { C L I P } ; < { \bf \nabla } \cdot { \bf \nabla } >$ denotes the calculation of cosine similarity; $G _ { \sigma }$ represents Gaussian filter(Zhou et al. 2024a; Ma et al. 2026); and Bilinear refers to bilinear interpolation. For the point cloud input, after deriving the multi-view anomaly score map $\mathrm { A } _ { p c } ^ { ( m ) } \ \in \ \mathbb { R } ^ { H \times W }$ and anomaly response maps $\mathrm { S } _ { p c } ^ { ( m ) } \in \mathbb { R } ^ { H \times \bar { W } }$ in a similar manner from $\{ \hat { \mathcal { L } } _ { k } ^ { ( m ) } \}$ , we adopt the post-processing method BackTo3D from (Zhou et al. 2024b) to map $\mathrm { A } _ { p c } ^ { ( m ) }$ and $\mathrm { S } _ { p c } ^ { ( m ) }$ into a format corresponding to $\mathrm { S } _ { r g b }$ . This mapping is defined as: $\mathrm { A } _ { p c } = \mathrm { B a c k T o 3 D } ( \mathrm { A } _ { p c } ^ { ( 1 ) } , \mathrm { A } _ { p c } ^ { ( 2 ) } , \dots , \mathrm { A } _ { p c } ^ { ( M ) } ) , \ \mathrm { S } _ { p c } =$ 1 $\overline { { { 3 \mathrm { a c k T o 3 D } ( \mathrm { S } _ { p c } ^ { ( 1 ) } , \mathrm { S } _ { p c } ^ { ( 2 ) } , \dots , \mathrm { S } _ { p c } ^ { ( M ) } ) } } }$ . The resulting maps are denoted as $\dot { \mathrm { A } } _ { p c } ^ { } \in \dot { \mathbb { R } } ^ { H \times W }$ and $\mathrm { S } _ { p c } ^ { ' } \in \mathbb { R } ^ { H \times W }$ . Although $\mathrm { S } _ { r g b }$ and $\mathrm { S } _ { p c }$ exhibit a strict correspondence, their reliability varies dynamically due to the inherent representational capacities of diferent modalities. Consequently, the final multimodal anomaly response map cannot be derived through simple averaging.

We utilize the hybrid loss $L _ { h y b }$ proposed by PointAD(Zhou et al. 2024b) to supervise the anomaly segmentation task. The definition of $L _ { h y b }$ can be found in PointAD or the Supplement.

## Reliability-Aware Modality Response Balancing

To explicitly model modality reliability, we propose Modality Reliability Calibration (MRC), a reliability-aware modality response balancing module. Given the point cloud response $\mathrm { S } _ { p c }$ and RGB response $\mathrm { S } _ { r g b }$ , MRC module estimates the reliability of each modality and adaptively assigns fusion weights. Specifically, we adopt a simple yet efective cross-attention mechanism to achieve bidirectional feature interaction and reliability estimation across modalities. Initially, $\{ \mathcal { L } _ { \mathrm { I } } ^ { ( m ) } \}$ } and $\{ \hat { \mathcal { L } } _ { k } ^ { ( m ) } \}$ are passed through $K _ { l }$ cross-attention layers to enable deep feature fusion. Crucially, based on the observation that point clouds ofer superior reliability compared to RGB images in zero-shot AD, we designate the point cloud features $\{ \hat { \mathcal { L } } _ { k } ^ { ( m ) } \}$ as the Key/ Value and RGB features as the Query. This strategic choice allows us to leverage geometric information to gauge the credibility of the RGB features, thereby producing the reliability map $\mathrm { W } \in \mathbb { R } ^ { H \times W }$ for the RGB modality. This procedure is mathematically defined as:

$$
\mathcal { Z } _ { i + 1 } = \mathrm { C r o s s A t t } ( \mathcal { Z } _ { \mathrm { i } } , \{ \hat { \mathcal { L } } _ { \mathrm { k } } ^ { ( \mathrm { m } ) } \} ) , i = 0 , 1 , \ldots , K _ { l } - 1 ,\tag{15}
$$

$$
\mathrm { W } = \mathrm { F C } ( \mathcal { Z } _ { K _ { l } } ) ,\tag{16}
$$

where ${ \mathcal Z } _ { 0 } ~ = ~ \{ { \mathcal L } _ { \mathrm { I } } ^ { ( m ) } \}$ and FC denotes a D-dimensional single-layer perceptron. Ultimately, the multimodal anomaly score map A and the multimodal anomaly response map S are computed as:

$$
\mathrm { A } = \mathrm { W } \circ \mathrm { A } _ { r g b } + ( 1 - \mathrm { W } ) \circ \mathrm { A } _ { p c } ,\tag{17}
$$

$$
\mathrm { S } = \mathrm { B i l i n e a r } [ G _ { \sigma } ( \mathrm { A } ) ] ,\tag{18}
$$

where ◦ denotes Hadamard product. We design a multimodal response loss $L _ { m u l }$ specifically for MRC as follows:

$$
\begin{array} { l } { { { \cal L } _ { m u l } = \mathrm { F o c a l } [ { \mathrm A } \oplus ( { \bf 1 } - { \mathrm A } ) , { \mathrm I } ^ { g t } ] } } \\ { { \quad \quad \quad + \mathrm { D i c e } ( { \mathrm A } , { \mathrm I } ^ { g t } ) + \mathrm { D i c e } ( { \bf 1 } - { \mathrm A } , { \bf 1 } - { \mathrm I } ^ { g t } ) , } } \end{array}\tag{19}
$$

where Focal and Dice refer to the Focal Loss(Lin et al. 2017) and Dice loss(Milletari, Navab, and Ahmadi 2016), respectively.

## Overall Loss&Training

The overall loss function is formulated as follows:

$$
L _ { a l l } = \lambda _ { f e } L _ { f e } + \lambda _ { h y b } L _ { h y b } + \lambda _ { m u l } L _ { m u l } ,\tag{20}
$$

where $\lambda _ { f e } , \lambda _ { h y b }$ , and $\lambda _ { m u l }$ are hyperparameters representing the weights of the respective loss terms. During training, we freeze the weights of both $\Psi _ { \mathrm { v } }$ and $\Psi _ { \mathrm { t } } \mathrm { t } 0$ fully leverage the robust generalization capabilities ofCLIP. By minimizing the loss $L _ { a l l }$ , we jointly optimize MID module, MRC module, and the language vectors $\mathbf { P } _ { n }$ and ${ \bf P } _ { a }$

## Experiments

## Experimental Setup

Datasets. We evaluate our method on two widely adopted benchmark datasets for multimodal AD: MVTec 3D-AD(Bergmann et al. 2022) and Eyecandies(Bonfiglioli et al. 2022). Detailed descriptions of these two datasets are provided in the supplement.

Evaluation metrics. To characterize the performance limits of existing zero-shot approaches under practical conditions, we employ the stricter threshold-based metrics m $F _ { 1 . 8 } ^ { \cdot 2 } ,$ m ${ \mathrm { A c c } } _ { . 8 } ^ { . 2 } ,$ and mIoU<sup>.2</sup>(Zhang et al. 2026). These metrics evaluate anomaly localization over an industry-relevant confidence interval [0.2, 0.8], jointly accounting for the precision–recall trade-of, overall classification accuracy, and spatial localization fidelity. In contrast to traditional metrics(Bergmann et al. 2022)—including image-level AUROC, pixel-level AUROC, AUPRO, and AP—that often exhibit saturation on contemporary zero-shot AD benchmarks(Zhang et al. 2026), these threshold-based metrics impose stronger penalties on abnormal responses in anomaly-free regions. Consequently, they ofer superior discriminative power and better align with practical application scenarios. Additionally, we incorporate pixel-level AUROC (P-AUROC) and image-level AP as complementary metrics, with the corresponding results reported in the supplement.

Baselines. We compare WOOPS with representative zeroshot unimodal AD and zero-shot multimodal AD methods, including AnomalyCLIP(Zhou et al. 2024a), FAPrompt(Zhu et al. 2025), GS-CLIP(Deng, Liu, and Wang 2026), PointAD(Zhou et al. 2024b), and ZUMA-FT(Ma et al. 2026). A brief description of the baselines is provided in the Supplement.

Implementation details. We utilize the pre-trained CLIP(Radford et al. 2021) (ViT-L/14@336px) as our backbone. Following the setup of prior methods(Zhou et al. 2024b; Ma et al. 2026), we set the number of multi-view projections to $M = 9$ , the feature dimension to $D = 7 6 8$ and the length of learnable language vectors to $l = 1 2$ . To ensure a fair comparison with PointAD(Zhou et al. 2024b), we set the loss weight $\lambda _ { h y b } = 1 . 0$ of the corresponding loss term $L _ { h y b }$ . The remaining loss weights are configured as $\lambda _ { f e } = 1 . 0$ and $\lambda _ { m u l } = 0 . 5$ . We set the number of selected multi-view projections in the MID module to X = 1. MRC module comprises $K _ { l } = 4$ cross-attention layers. We employ the Adam optimizer(Kingma and Ba 2017) with a learning rate of 0.001. The model is trained for 15 epochs. All experiments are conducted on a single NVIDIA RTX 4090 GPU. A sensitivity analysis of the hyperparameters is provided in the supplementary material.

## Main Results

Comparison with state-of-the-art methods. We evaluated the performance of our proposed WOOPS against current state-of-the-art zero-shot AD methods on both the MVTec 3D-AD and Eyecandies datasets, with quantitative results summarized in Table 1. Our approach achieves the best or competitive performance across RGB-only, point cloud-only, and multimodal inference settings. Specifically, under point cloud-only inference, WOOPS substantially outperforms PointAD and ZUMA-FT, improving m $F _ { 1 . 8 } ^ { \cdot 2 } ,$ $\mathrm { m } \mathrm { \bar { A } c c _ { . 8 } ^ { . 2 } }$ , and $\mathrm { m I o U _ { . 8 } ^ { . 2 } }$ by 5.7%, 11.0%, and 3.2%, respectively. This advantage stems from their lack of dedicated processing for multi-view projected images, which renders them vulnerable to interference from heterogeneous information. Although GS-CLIP incorporates depth maps and geometric cues to enhance prompt semantics, its performance still lags behind ours. In the multimodal setting, our method attains the best-reported results on the Eyecandies dataset, surpassing the previous best-performing approach by 3.1%, 16.9%, and 2.5% in terms of m $F _ { 1 . 8 } ^ { \cdot 2 } , \mathrm { m } \mathrm { \bar { A } c c . \dot { 8 } } .$ , and $\mathrm { m } \mathrm { \bar { I } o U _ { . 8 } ^ { . 2 } }$ , respectively. On the MVTec 3D-AD dataset, WOOPS also demonstrates competitive performance, achieving an m $F _ { 1 . 8 } ^ { \mathrm { ~ . 2 } }$ of 19.3% versus 20.6%, mAcc<sup>.2</sup> of 54.4% versus $5 7 . 9 \%$ , and $\mathrm { m I o U _ { . 8 } ^ { . 2 } }$ of 11.6% versus 12.3%. Results for RGB-only inference will be discussed in the following subsection. Qualitative comparisons and quantitative evaluations based on conventional metrics are provided in the Supplement.

<table><tr><td rowspan="2">Inference</td><td rowspan="2">Training</td><td rowspan="2">Method</td><td rowspan="2">Source</td><td colspan="3">MVTec 3D-AD</td><td colspan="3">Eyecandies</td></tr><tr><td> $\mathrm { m } F _ { 1 . 8 } ^ { \cdot 2 }$ </td><td> $\mathrm { m A c c } _ { . 8 } ^ { . 2 }$ </td><td> $\mathrm { m I o U _ { . 8 } ^ { . 2 } }$ </td><td> $\mathrm { m } F _ { 1 . 8 } ^ { \cdot 2 }$ </td><td> $\mathrm { \ m A c c _ { . 8 } ^ { . 2 } }$ </td><td> $\mathrm { m I o U _ { . 8 } ^ { . 2 } }$ </td></tr><tr><td rowspan="5">RGB</td><td>RGB</td><td>AnomalyCLIP</td><td>ICLR&#x27;24</td><td>2.1</td><td>7.9</td><td>1.0</td><td>1.4</td><td>7.4</td><td>0.7</td></tr><tr><td>RGB</td><td>FAPrompt</td><td>ICCV&#x27;25</td><td>2.8</td><td>9.7</td><td>1.5</td><td>2.5</td><td>8.8</td><td>1.2</td></tr><tr><td>PC</td><td>PointAD (RGB)</td><td>NIPS&#x27;24</td><td>19.8</td><td>25.1</td><td>11.6</td><td>13.1</td><td>18.0</td><td>7.4</td></tr><tr><td>PC</td><td>ZUMA-FT (RGB)</td><td>TPAMI&#x27;26</td><td>20.1</td><td>25.3</td><td>11.8</td><td>13.0</td><td>18.1</td><td>7.3</td></tr><tr><td>MUL</td><td>Ours (RGB)</td><td></td><td>21.0</td><td>44.7</td><td>12.1</td><td>17.2</td><td>42.7</td><td>10.3</td></tr><tr><td rowspan="4">PC</td><td>PC</td><td>PointAD (PC)</td><td>NIPS&#x27;24</td><td>15.9</td><td>29.6</td><td>9.5</td><td>14.7</td><td>28.1</td><td>8.4</td></tr><tr><td>PC</td><td>GS-CLIP</td><td>CVPR’26</td><td>17.6</td><td>24.8</td><td>10.6</td><td>16.1</td><td>24.6</td><td>9.3</td></tr><tr><td>PC</td><td>ZUMA-FT (PC)</td><td>TPAMI&#x27;26</td><td>16.9</td><td>28.6</td><td>10.1</td><td>14.8</td><td>28.7</td><td>8.5</td></tr><tr><td>MUL</td><td>Ours (PC)</td><td></td><td>22.6</td><td>40.6</td><td>13.8</td><td>22.4</td><td>32.9</td><td>13.8</td></tr><tr><td rowspan="3">MUL</td><td>PC</td><td>PointAD</td><td>NIPS&#x27;24</td><td>20.6</td><td>33.1</td><td>12.3</td><td>15.8</td><td>28.1</td><td>9.1</td></tr><tr><td>PC</td><td>ZUMA-FT</td><td>TPAMI&#x27;26</td><td>19.2</td><td>57.9</td><td>11.6</td><td>14.8</td><td>43.8</td><td>8.6</td></tr><tr><td>MUL</td><td>Ours</td><td></td><td>19.3</td><td>54.4</td><td>11.6</td><td>18.9</td><td>45.0</td><td>11.6</td></tr></table>

Table 1: Comparison with state-of-the-art methods on MVTec 3D-AD dataset and Eyecandies dataset. Results are ranked separately according to the inference modality configurations. The best and second-best results within each group are denoted by gray shading.

Point clouds matter in zero-shot multimodal anomaly detection. We further evaluate RGB-only inference under diferent training modalities. As shown in Table 1, methods trained with RGB alone perform significantly worse than methods trained with point cloud or RGB-point cloud data. This phenomenon indicates that point cloud information facilitates the learning of discriminative semantic representations for normal and anomalous patterns, and benefits cross-category generalization beyond the point cloud modality itself. Since some point-cloud-based methods use visionlanguage models, they can still be applied to RGB-only inference. The superior RGB-only performance of our method indicates that the proposed point cloud feature enhancement also improves modality-transferable anomaly cues. This result further supports our central observation that point cloud data play a crucial role in zero-shot multimodal AD. Further visualization results in Fig. 3 highlight the distinct behaviors of diferent modalities. While point cloud responses prove more reliable than their RGB counterparts, the fused multimodal response maintains high fidelity. Analysis of the RGB reliability map confirms that this reliability is achieved by mitigating spurious RGB responses within normal regions.

Analysis of the performance gap between point-cloudonly and multimodal settings. A notable observation is that the point-cloud-only version of our method performs better than its multimodal counterpart. This difers from the common expectation that multimodal inference should always outperform single-modality inference. The experimental analyses in the preceding two paragraphs demonstrate that, compared with point clouds, RGB inputs are more prone to spurious anomaly activations—a conclusion further corroborated by the qualitative results presented in Fig. 3. When the responses derived from point clouds exhibit low reliability, incorporating RGB responses as complementary information improves performance; however, when point cloud responses become suficiently reliable due to the MID module, RGB responses instead exert a detrimental efect on performance. To the best of our knowledge, this work is the first to reveal the potential multimodal collapse in zero-shot multimodal AD, and we intend to investigate more principled and efective multimodal fusion strategies in future research.

## Ablation Studies

Efect of the MID module. We first ablate the proposed point cloud feature enhancement module. As shown in Table 2, removing MID module leads to a clear performance drop, especially under point-cloud-only inference. This confirms that directly aggregating multi-view projected features is suboptimal. By suppressing unreliable view-specific features and enhancing consistent geometric cues, MID module produces higher-quality point cloud representations. We visualize the decoupled features from the MID module using $\mathrm { \ t { - } S N E } ,$ as shown in Fig. 4. As illustrated, driven by the decoupling loss $L _ { d e c } ,$ the homogeneous and heterogeneous information within both global and local patch features are efectively repelled. Furthermore, owing to the explicit constraints imposed by $L _ { d c }$ and $L _ { g m m d }$ , the homogeneous components of the global features achieve precise alignment. Building upon this aligned global representation, and leveraging the powerful semantic priors embedded in the frozen CLIP visual encoder, the homogeneous information in local patch features is also naturally aligned.

![](images/4d2a6030ac8a8b0b2d54242c6b27624787e51643a5d44b552875ddf4cfbb4b60.jpg)  
Figure 3: Visualization results of WOOPS. From top to bottom: RGB input, RGB anomaly response, RGB reliability map, point cloud anomaly response, multimodal anomaly response, and ground truth. In the reliability map, warmer red tones indicate higher confidence, whereas cooler green tones denote lower reliability. As evident, regions exhibiting erroneous RGB responses correspond to low-reliability areas, demonstrating the efectiveness of our reliability calibration.

Efect of the MRC module. We then evaluate the modality response balancing module. Table 2 shows that removing MRC module causes unstable performance across datasets. In particular, the model may perform well on one dataset but degrade substantially on another dataset across multiple modality settings. This indicates that fixed or uncalibrated fusion is sensitive to dataset-specific modality reliability. By adaptively calibrating modality contributions according to their reliability, MRC module improves robustness and reduces the risk of being dominated by unreliable modality responses.

Ablation Study of Loss Terms. To validate the efectiveness of the proposed loss terms, i.e., $L _ { f e }$ and $L _ { m u l } ,$ we conduct ablation studies on these components, with quantitative results summarized in Table 2. Removing $L _ { f e }$ leads to a moderate performance degradation, confirming its necessity. Notably, in the absence of $L _ { f e } ,$ , the $\Phi _ { o }$ acts as a post-hoc feature adapter(Gao et al. 2023), thereby preventing a catastrophic drop in performance. In contrast, discarding $L _ { m u l }$ results in a significant decline, demonstrating that it serves as a crucial supervisory signal within our training framework. Furthermore, we supplement the ablation study with an analysis of $L _ { h y b }$ . The results indicate that removing $L _ { h y b }$ causes a substantial performance drop, underscoring its role as a fundamental objective for multimodal zero-shot AD.

<table><tr><td rowspan="3">Setting</td><td colspan="5"> $\mathrm { m } F _ { 1 . 8 } ^ { \cdot 2 }$ </td></tr><tr><td colspan="2">MVTec 3D-AD</td><td colspan="3">Eyecandies</td></tr><tr><td>RGB</td><td>PC MUL</td><td>RGB</td><td>PC</td><td>MUL</td></tr><tr><td>Full</td><td>21.0</td><td>22.6 19.3</td><td>17.2</td><td>22.4</td><td>18.9</td></tr><tr><td>w/o MID</td><td>18.5</td><td>17.4 18.1</td><td>13.1</td><td>13.8</td><td>13.3</td></tr><tr><td>w/o MRC</td><td>20.4</td><td>22.5 19.1</td><td>9.3</td><td>11.6</td><td>11.4</td></tr><tr><td>w/o  $L _ { f e }$ </td><td>18.0</td><td>21.0 18.0</td><td>17.0</td><td>21.9</td><td>18.7</td></tr><tr><td>w/o  $L _ { m u l }$ </td><td>12.5</td><td>15.8 14.0</td><td>13.1</td><td>14.8</td><td>13.1</td></tr><tr><td>w/o  $L _ { h y b }$ </td><td>1.5</td><td>4.8 1.5</td><td></td><td>1.2 2.6</td><td>2.6</td></tr></table>

Table 2: Quantitative results of the ablation study. Top results are in bold, and the default setting is indicated by gray shading.

![](images/1f8c158a4b00683e0620069e48ebc5c733517fb8b06cabc7f003ec8491e516d8.jpg)

![](images/dfa3678e19c9cc9740932e472ead20754ba877f69c298cf2487d0c2c0e3ac9da.jpg)  
(a) Visualization of Global Feature Distributions via t-SNE  
(b) Visualization of Local Patch Feature Distributions via t-SNE  
Figure 4: t-SNE Visualization of Decoupled Features from the MID Module. "Sample" denotes the features of the projection views sampled by the MID module, while "Overall" refers to the point cloud features. Zoom in for details.

## Conclusion

We revisit zero-shot multimodal AD by questioning a common assumption: RGB and point cloud modalities are equally reliable and should contribute equally. Under recently proposed stringent metrics, we show that point clouds provide more reliable cross-category generalization, while RGB may introduce false anomaly responses in normal regions. Guided by this finding, we propose WOOPS, which enhances point cloud representations through MID module and calibrates modality contributions through MRC module. Extensive experiments demonstrate the efectiveness of both geometry and reliability-aware modality utilization under zero-shot multimodal AD.

## References

Asad, M.; Azeem, W.; Malik, A. A.; Jiang, H.; Ali, A.; Yang, J.; and Liu, W. 2025. 3D-MMFN: Multi-level multimodal fusion network for 3D industrial image anomaly detection. Advanced Engineering Informatics, 65: 103284.

Bergmann, P.; Jin, X.; Sattlegger, D.; and Steger, C. 2022. The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization. In Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 202–213. SCITEPRESS - Science and Technology Publications.

Bonfiglioli, L.; Toschi, M.; Silvestri, D.; Fioraio, N.; and De Gregorio, D. 2022. The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization. In Proceedings ofthe Asian Conference on Computer Vision (ACCV), 3586–3602.

Cai, Y.; Zhang, W.; Chen, H.; and Cheng, K.-T. 2025. MedIAnomaly: A comparative study of anomaly detection in medical images. Medical Image Analysis, 102: 103500.

Cao, Y.; Zhang, J.; Frittoli, L.; Cheng, Y.; Shen, W.; and Boracchi, G. 2025. AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., Computer Vision – ECCV 2024, 55– 72. Cham: Springer Nature Switzerland. ISBN 978-3-031- 72761-0.

Chen, Q.; Qu, Z.; Luo, W.; Yao, H.; Cao, Y.; Jiang, Y.; Duan, Y.; Luo, H.; Lv, C.; and Zhang, Z. 2026. CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 8554–8563.

Chen, R.; Xie, G.; Liu, J.; Wang, J.; Luo, Z.; Wang, J.; and Zheng, F. 2023. EasyNet: An Easy Network for 3D Industrial Anomaly Detection. In Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, 7038–7046. New York, NY, USA: Association for Computing Machinery. ISBN 9798400701085.

Cheng, Y.; Chen, J.; Wen, G.; Tan, X.; and Liu, X. 2025. LPFSTNet: A lightweight and parameter-free head attentionbased student–teacher network for fast 3D industrial anomaly detection. Neurocomputing, 623: 129408.

Costanzino, A.; Ramirez, P. Z.; Lisanti, G.; and Di Stefano, L. 2024. Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17234–17243.

Costanzino, A.; Ramirez, P. Z.; Lisanti, G.; and Di Stefano, L. 2026. Modulate-and-Map: Crossmodal Feature Mapping with Cross-View Modulation for 3D Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 8816–8825.

Deng, Z.; Liu, A.; and Wang, Y. 2026. GS-CLIP: Zeroshot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 35587–35596.

Fang, Q.; Su, Q.; Lv, W.; Xu, W.; and Yu, J. 2025. Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 39(16): 16532–16540.

Gao, P.; Geng, S.; Zhang, R.; Ma, T.; Fang, R.; Zhang, Y.; Li, H.; and Qiao, Y. 2023. CLIP-Adapter: Better Vision-Language Models with Feature Adapters. International Journal ofComputer Vision, 132(2): 581–595.

Gu, Z.; Zhang, J.; Liu, L.; Chen, X.; Peng, J.; Gan, Z.; Jiang, G.; Shu, A.; Wang, Y.; and Ma, L. 2024. Rethinking Reverse Distillation for Multi-Modal Anomaly Detection. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(8): 8445–8453.

Gu, Z.; Zhu, B.; Zhu, G.; Chen, Y.; Ge, W.; Tang, M.; and Wang, J. 2026. AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 40(6): 4348–4356.

Guo, J.; Lu, S.; Zhang, W.; Chen, F.; Li, H.; and Liao, H. 2025. Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20405–20415.

Horwitz, E.; and Hoshen, Y. 2023. Back to the Feature: Classical 3D Features Are (Almost) All You Need for 3D Anomaly Detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2968–2977.

Hou, Y.; Li, P.; Liu, Z.; Wang, Y.; Ruan, Y.; Qiu, J.; and Xu, K. 2026. VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 21346–21356.

Jeong, J.; Zou, Y.; Kim, T.; Zhang, D.; Ravichandran, A.; and Dabeer, O. 2023. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19606–19616.

Kingma, D. P.; and Ba, J. 2017. Adam: A Method for Stochastic Optimization. arXiv:1412.6980.

Li, K.; Li, G.; Zhou, M.; Li, M.; Han, D.; and Wan, J. 2026. Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14167–14177.

Li, X.; Xue, F.; and Zhou, Y. 2026. MuSc-V2: Zero-Shot Multimodal Industrial Anomaly Classification and Segmentation with Mutual Scoring of Unlabeled Samples. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1–14.

Li, Y.; Zhou, X.; Lan, S.; Wang, W.; Lai, X.; and Qiao, Y. 2025. MADFlow: Multimodal diference compensation flow for multimodal anomaly detection. Neurocomputing, 654: 131243.

Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollár, P. 2017. Focal Loss for Dense Object Detection. In 2017

IEEE International Conference on Computer Vision (ICCV), 2999–3007.

Liu, J.; Mou, S.; Gaw, N.; and Wang, Y. 2025. Uni-3DAD: Gan-inversion aided universal 3D anomaly detection on model-free products. Expert Systems with Applications, 272: 126665.

Liu, J.; Zhang, Y.; Chen, J.-N.; Xiao, J.; Lu, Y.; Landman, B. A.; Yuan, Y.; Yuille, A.; Tang, Y.; and Zhou, Z. 2023. CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 21095–21107.

Ma, Y.; Liu, M.; Jiang, S.; Zhou, J.; Bian, Y.; Wang, X.; and Wang, Y. 2026. ZUMA: Training-Free Zero-Shot Unified Multimodal Anomaly Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(6): 6601–6614.

Milletari, F.; Navab, N.; and Ahmadi, S.-A. 2016. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In 2016 Fourth International Conference on 3D Vision (3DV), 565–571.

Qian, C.; Xing, S.; Li, S.; Zhao, Y.; and Tu, Z. 2026. DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning. In International Conference on Learning Representations (ICLR).

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Meila, M.; and Zhang, T., eds., Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, 8748–8763. PMLR.

Rudolph, M.; Wehrbein, T.; Rosenhahn, B.; and Wandt, B. 2023. Asymmetric Student-Teacher Networks for Industrial Anomaly Detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2592–2602.

Shukla, V.; Shukla, A.; S. K., S. P.; and Shukla, S. 2025. A systematic survey: role of deep learning-based image anomaly detection in industrial inspection contexts. Frontiers in Robotics and AI, Volume 12 - 2025.

Tu, Y.; Zhang, B.; Liu, L.; Li, Y.; Zhang, J.; Wang, Y.; Wang, C.; and Zhao, C. 2025. Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., Computer Vision – ECCV2024, 75–91. Cham: Springer Nature Switzerland. ISBN 978-3-031-72627-9.

Wang, C.; Zhu, H.; Peng, J.; Wang, Y.; Yi, R.; Wu, Y.; Ma, L.; and Zhang, J. 2025. M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(11): 9981–9993.

Wang, J.; Niu, Y.; and Huang, B. 2025. Fusion-restoration model for industrial multimodal anomaly detection. Neurocomputing, 637: 130073.

Wang, Y.; Peng, J.; Zhang, J.; Yi, R.; Wang, Y.; and Wang, C. 2023. Multimodal Industrial Anomaly Detection via Hybrid

Fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8032– 8041.

Xiang, A.; Huang, Z.; Gao, X.; Ye, K.; and Xu, C.-z. 2025. BridgeNet: A Unified Multimodal Framework for Bridging 2D and 3D Industrial Anomaly Detection. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, 1579–1587. New York, NY, USA: Association for Computing Machinery. ISBN 9798400720352.

Xue, L.; Gao, M.; Xing, C.; Martín-Martín, R.; Wu, J.; Xiong, C.; Xu, R.; Niebles, J. C.; and Savarese, S. 2023. ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1179–1189.

Xue, L.; Yu, N.; Zhang, S.; Panagopoulou, A.; Li, J.; Martín-Martín, R.; Wu, J.; Xiong, C.; Xu, R.; Niebles, J. C.; and Savarese, S. 2024. ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 27091–27101.

Zavrtanik, V.; Kristan, M.; and Skočaj, D. 2024. Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2164–2172.

Zhang, J.; Wang, C.; Li, X.; Tian, G.; Xue, Z.; Liu, Y.; Pang, G.; and Tao, D. 2026. Learning Feature Inversion for Multi-class Anomaly Detection under General-purpose COCO-AD Benchmark. International Journal of Computer Vision, 134(5).

ZHAO, Y. 2025. AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 3152–3161.

Zhou, K.; Yang, J.; Loy, C. C.; and Liu, Z. 2022. Learning to Prompt for Vision-Language Models. International Journal ofComputer Vision, 130(9): 2337–2348.

Zhou, Q.; Pang, G.; Tian, Y.; He, S.; and Chen, J. 2024a. AnomalyCLIP: Object-agnostic Prompt Learning for Zeroshot Anomaly Detection. In Kim, B.; Yue, Y.; Chaudhuri, S.; Fragkiadaki, K.; Khan, M.; and Sun, Y., eds., International Conference on Learning Representations, volume 2024, 49705–49737.

Zhou, Q.; Yan, J.; He, S.; Meng, W.; and Chen, J. 2024b. PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection. In Globerson, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C., eds., Advances in Neural Information Processing Systems, volume 37, 84866–84896. Curran Associates, Inc.

Zhu, J.; Ong, Y.-S.; Shen, C.; and Pang, G. 2025. Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 22241– 22251.