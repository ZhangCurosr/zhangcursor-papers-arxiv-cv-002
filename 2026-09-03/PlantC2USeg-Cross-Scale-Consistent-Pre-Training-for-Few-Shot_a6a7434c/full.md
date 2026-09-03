# PlantC2USeg: Cross-Scale Consistent Pre-Training for Few-Shot Unified Plant Point Cloud Segmentation

Yu Tian<sup>a</sup>, Xintong Jiang<sup>a</sup>, Jan Franklin Adamowski<sup>a</sup>, Shiv O. Prasher<sup>a</sup> and Shangpeng Sun<sup>a,∗</sup>

<sup>a</sup>McGill University, 21111 Lakeshore Road, Sainte-Anne-de-Bellevue, H9X 3V9, Quebec, Canada

A R T I C L E I N F O

Keywords: Point cloud segmentation Deep transfer learning Few-shot segmentation Plant phenotyping

## A BS T RA C T

As the advancement of modern crop breeding demands precise organ-level analysis for trait quantification, plant point cloud segmentation (PPCS) is elevated to a problem of vital importance. Conventional deep learning approaches to PPCS rely heavily on densely annotated datasets, whose acquisition remains labor-intensive. Human perception adapts rapidly from only a few novel examples, whereas unified PPCS adaptation from distribution-shifted examples with minimal additional training efort remains an open challenge. Deep transfer learning (DTL) seeks to alleviate annotation dependency by pre-training on self-generated objectives, thereby structuring representation spaces that facilitate eficient adaptation to PPCS across diverse species, growth stages, and sensing conditions. Inspired by the multi-scale verification process in manual annotation, we propose cross-scale consistency learning within the DTL framework, PlantC2USeg, to explicitly align features across spatial scales, together with an information-restricted decoding strategy that prevents reconstruction shortcuts and promotes robust adaptation. Such pre-training enables stable few-shot generalization under species and sensor variations, while unified fine-tuning with inherited thresholds further reduces practical adaptation overhead. Under full supervision on the Soybean3D dataset, the complete pre-training configuration improves semantic IoU and instance mWCov over random initialization, attaining the highest values among the compared methods at 91.91% and 94.62%, respectively. With 20 labeled Soybean3D samples, PlantC2USeg leads both semantic IoU and instance mWCov at 89.78% and 90.27%, and with only 10 samples it retains the highest mWCov of 83.23% while achieving 83.19% IoU. Across HR3D, 10-shot transfer to tobacco, tomato, and sorghum averages 78.41% IoU and 79.42% mWCov, while 22-shot transfer to SYAU-Maize attains the highest IoU and mRec among the compared methods at 92.75% and 93.51%. Furthermore, achieving the leading category-averaged mIoU of 85.0% on the ShapeNet Part benchmark demonstrates the general capability of the proposed framework to handle diverse shape variations beyond agricultural domains. These results demonstrate that the proposed framework reduces the overall adaptation efort under distribution shifts, thereby enabling scalable plant phenotyping and transferable 3D representation learning to extend beyond agriculture.

## 1. Introduction

High-throughput phenotyping (HTP) has enabled the acceleration of high-yield and stress-tolerant cultivar development in plant breeding (Watson et al., 2018; Li et al., 2025b), which plays a vital role in addressing global food insecurity (van Dijk et al., 2021). By leveraging optical imaging technologies, HTP can select the ideal aboveground architecture by conducting rapid and non-destructive measurements of plant traits such as plant height and leaf angle (Yang et al., 2020). Among the available sensing modalities, 3D plant point clouds have become a dominant representation for accurate phenotyping, as they capture details that are often occluded in 2D images. Extracting traits from these data requires reliable segmentation that decomposes raw 3D point clouds into distinct plant organs.

Deep learning (DL) has become the dominant approach for plant point cloud segmentation (PPCS), owing to its ability to learn representations directly from raw 3D data and achieve superior performance compared with traditional handcrafted-feature pipelines (Johnson and Hebert, 1999; Rusu et al., 2009). PointNet (Qi et al., 2017a) made this shift possible for irregular point sets by directly consuming unordered points, avoiding conversion to voxel grids or rendered views. By preserving permutation invariance through symmetric aggregation, PointNet leaves local metric neighborhoods largely implicit, which makes PointNet++ (Qi et al., 2017b) and DGCNN (Wang et al., 2019b) a natural extension for PPCS because their metric space grouping and dynamic �-NN graph construction strengthen local spatial cues for separating adjacent and morphologically similar plant organs (Heiwolt et al., 2021). More recently, PPCS has adopted Transformer (Vaswani et al., 2017) architectures to introduce attention-driven contextual modeling for complex plant architectures, with PST (Du et al., 2023) targeting dense high-resolution rapeseed point clouds in which tiny siliques are spatially scattered and overlapping. Despite these advances, supervised PPCS still depends on labeled plant point clouds, and performance on unseen cases remains governed by sample and distribution generalization (Rohlfs, 2025). Such dependence on data scale becomes a fundamental bottleneck in practice, as acquiring dense annotations for plant point clouds is labor-intensive.

This limitation has motivated an increasing focus on developing models that remain stable under limited-annotation regimes, which is conceptually aligned with human intelligence that can learn from a few examples and adapt the knowledge to process new test cases (George et al., 2017). Data-centric strategies have been explored to enrich the morphological variety, either by incorporating physical deformation on real-world labeled data (Yang et al., 2024) or by generating synthetic plant point clouds through procedural modeling paradigms (Lee et al., 2023; Du et al., 2025). Although synthetic data provide scalable supervision, they are typically constructed under predefined morphological assumptions and simulation rules, the learned representations may therefore remain biased toward specific structural patterns. Sparse-label propagation has also been investigated through AR-assisted annotation, where graph-based transductive inference derives pseudo point-level labels from limited manual inputs and reduces manual labeling to 32.3% per sample (Li et al., 2025a). Practical HTP applications require models to generalize across point clouds collected from diferent species, growth stages, sensors, imaging settings, and environmental conditions. This expectation exceeds the distributional scope of the data synthesis (Yang et al., 2024; Lee et al., 2023; Du et al., 2025) and label inference (Li et al., 2025a) strategies, positioning deep transfer learning (DTL) as a more scalable alternative by leveraging pre-trained representations to facilitate adaptation under distributional shifts (Sohail et al., 2025).

DTL frameworks address distributional shifts by decoupling representation learning from task-specific adaptation. By optimizing self-generated tasks on unlabeled point clouds, unsupervised pre-training encourages the models to internalize structural regularities and morphological patterns that govern plant form beyond specific annotated datasets. During downstream PPCS adaptation, segmentation fine-tuning constrains task-specific optimization within a representation space shaped by plant architectural patterns learned during pre-training. Initializing segmentation models from these pre-trained representations mitigates distribution-specific overfitting under full supervision, while enabling robust feature reuse in limited-supervision settings and enhancing generalization across distributional shifts. The growing afordability and portability of sensing systems, together with advances in reconstruction technologies, have improved the feasibility of large-scale plant point cloud collection (Harandi et al., 2023), thereby strengthening the potential of DTL on abundant raw 3D phenotypic data.

Point cloud pre-training has explored several ways of deriving supervision from raw 3D shape, including patch reassembly (Sauder and Sievers, 2019), occlusion comple tion (Wang et al., 2021), orientation prediction (Poursaeed et al., 2020), autoregressive generation (Chen et al., 2023), cross-modal teacher distillation (Dong et al., 2023), and contrastive-generative learning (Qi et al., 2023). Masked point cloud modeling provides a partial-observation objective by predicting hidden tokens from visible point patches (He et al., 2022; Yu et al., 2022; Pang et al., 2022), with later extensions introducing multi-scale hierarchy (Zhang et al., 2022), feature enhancement (Zha et al., 2024), center prediction (Zhang et al., 2024), and decoupled cross-view reconstruction (Zhang et al., 2025). Within PPCS, Ef-3DPSeg (Luo et al., 2023) incorporated a viewpoint bottleneck (VIB) objective (Tian et al., 2022) in which randomly transformed versions of the same plant point cloud were encoded and optimized to maximize cross-view consistency and channel decorrelation. Soybean-PCMAE (Tian et al., 2025) incorporates cross-view consistency into 3D masked modeling and selectively predicts high-dimensional features of masked regions that exhibit high instability under perturbations. The resulting representations are therefore regularized through whole-plant cross-view consistency or neighborhood reconstruction, with limited explicit coupling between finescale organ cues and broader plant architecture, which may constrain robustness under morphological variation. The information constraint imposed by masked modeling can also be diluted during decoding, as current PPCS masked autoencoding frameworks (Xie et al., 2025; Tian et al., 2025) process visible and masked tokens together through unrestricted self-attention. From a representation learning perspective, masked modeling is fundamentally designed as an asymmetric generative task, in which masked regions are inferred solely from observable inputs. By blurring this conditional structure, unrestricted decoding may weaken the encoder-centric regularization efect of the objective.

The efect of pre-training is ultimately realized during downstream PPCS adaptation, which primarily encompasses stem–leaf segmentation and leaf instance segmentation. Stem–leaf semantic segmentation assigns organ-level labels to points, whereas leaf instance segmentation represents a more challenging task that further distinguishes spatially adjacent and morphologically similar leaves by assigning unique instance identifiers (Song et al., 2025). PST (Du et al., 2023) adopts a two-stage pipeline that first predicts semantic labels using a transformer-based backbone and subsequently performs instance-aware feature grouping to separate individual leaves. Among general-purpose joint methods, SGPN (Wang et al., 2018) forms grouping proposals from pairwise similarities, whereas ASIS (Wang et al., 2019a) uses semantic predictions to guide instance embeddings and aggregated instance features to refine semantics. PlantNet (Li et al., 2022b) proposes a dual-function segmentation network that simultaneously performs semantic classification and instance embedding, enabling leaf separation through MeanShift clustering (Comaniciu and Meer, 2002) in the learned feature space. The follow-up work, PSegNet (Li et al., 2022a), extends the dual-function architecture by introducing voxelized farthest point sampling and dual-scale feature extraction, promoting hierarchical encoding of plant geometric structures. The reliance on mean-shift clustering within the learned embedding space introduces sensitivity to bandwidth selection, such that robust instance segmentation performance is contingent upon dataset-specific parameter tuning. Deformation3D (Yang et al., 2024) applies physically based deformation that preserves local organ geometry before training separate PointNet++ semantic and HAIS instance models. Under the DTL paradigm, Ef-3DPSeg (Luo et al., 2023) fine-tunes the pre-trained backbone using two independent networks, as unified optimization cannot be achieved stably. One network is dedicated to stem–leaf semantic segmentation, while a separate instance segmentation network introduces an additional ofset branch to predict point-wise displacements toward leaf centroids, followed by dual-set grouping on original and shifted coordinates follow ing the PointGroup (Jiang et al., 2020) strategy. Soybean-PCMAE (Tian et al., 2025) adopts a two-stage fine-tuning strategy, first optimizing stem–leaf semantic segmentation and then extending the model with centroid-directed ofset regression and instance-level discrimination supervision for leaf instance segmentation. During inference, leaf instances are obtained via clustering based on the predicted ofsets.

Existing PPCS transfer studies therefore motivate a pretraining design that preserves plant evidence across scales and avoids weakening the representation before it is used for unified semantic and instance fine-tuning. In practice, annotators repeatedly alternate between detailed inspection and global assessment, verifying whether fine-grained interpretations remain consistent with organ arrangement and whole-plant architecture (Figure 1). This observation motivates an explicit cross-scale consistency principle in pretraining, where representations are learned to preserve coherence between local cues and higher-level plant organization. For masked modeling to support PPCS transfer, the information available during decoding should keep masked regions dependent on visible plant evidence shaped by crossscale consistency learning. From the downstream perspective, unified PPCS adaptation requires stem–leaf semantic segmentation and leaf instance segmentation to remain stable within the same learned representation. Current DTL pipelines separate or stage semantic and instance optimization, whereas fully supervised unified models (Li et al., ${ 2 0 2 2 6 , \mathrm { a ) } }$ still rely on manually tuned clustering to delineate leaf instances. These observations motivate a unified few-shot adaptation strategy that integrates semantic and instance objectives within a shared fine-tuning process while reducing dataset-specific threshold tuning during instance inference. Hence, we develop a cross-scale consistency pretraining framework with information-restricted decoding for PPCS, enabling robust representation learning and stable unified fine-tuning of semantic and instance objectives under few-shot supervision. Our main contributions are as follows.

(i) We formulate cross-scale consistency as a pre-training objective that draws inspiration from the cross-scale validation inherent in manual annotation (Figure 1), while introducing an information-restricted decoder to mitigate reconstruction shortcuts.

(ii) We enable joint optimization of stem–leaf semantic segmentation and instance regularization in feature and coordinate spaces. Instance inference is performed via progressive segmentation with thresholds directly inherited from the fine-tuning stage, eliminating dataset-specific parameter sweeping.

(iii) Our framework achieves improved fully supervised accuracy and maintains stable performance under few-shot adaptation with as few as 10 to 20 labeled samples. We systematically evaluate generalization across distributional shifts, including singleand multi-species datasets as well as cross-sensor scenarios.

(iv) We extended Soybean3D dataset with additional reconstructed samples, consisting of 314 samples, among which 145 are labeled. The point count per sample ranges from 200,000-point to 1,000,000-point level, providing a high-quality benchmark for organ-level segmentation.

## 2. Methodology

## 2.1. Overview

Figure 2 presents PlantC2USeg as combining cross-scale pre-training, unified fine-tuning, and progressive segmentation for stem–leaf semantic segmentation and leaf instance segmentation under limited annotation.

## 2.2. Multi-scale Neighborhood Construction

As illustrated in Figure 2(a), the input point cloud $\mathbf { P } \in$ $\mathbb { R } ^ { N \times 3 }$ is organized into local neighborhoods that serve as contextual units for encoding, following Point-BERT and Point-M2AE (Yu et al., 2022; Zhang et al., 2022). For scale $s \in \{ 1 , \ldots , S \}$ , the center set is denoted as $\mathbf { P } _ { s } \in \mathbb { R } ^ { N _ { s } \times 3 }$ with neighborhood indices $\pmb { T } ^ { ( s ) } \in \mathbb { R } ^ { N _ { s } \times k _ { s } }$ . The neighbors are normalized with respect to their center point, so that each neighborhood is represented in a local reference frame that emphasizes relative spatial arrangement. Finer-scale neighborhoods retain details around organ boundaries, whereas coarser-scale neighborhoods situate these details within a broader spatial extent. This hierarchy provides local and broader neighborhood units over the plant point cloud, which are used in the subsequent cross-scale consistency objective.

## 2.3. Cross-scale pre-training

As shown in Figure 2(a)–(b), cross-scale pre-training combines consistency among projected representations of corresponding visible plant regions with informationrestricted decoding that conditions reconstruction on those same representations. The reconstruction signal recovers the center-relative input-point coordinate set around each masked first-scale center from visible plant evidence.

## 2.3.1. Hierarchicalfeature extractionfrom visible neighborhoods

Following Point-M2AE (Zhang et al., 2022), the inherited hierarchical masking strategy samples one random mask at the coarsest scale and recursively propagates visibility to finer scales through the stored neighborhood indices. At each scale �, the selected centers form the visible set $\mathbf { P } _ { s } ^ { v } \in \mathbb { R } ^ { N _ { s } ^ { v } \times 3 }$ while the complement forms the masked set $\mathbf { P } _ { s } ^ { m } \in \mathbb { R } ^ { N _ { s } ^ { m } \times 3 }$ with corresponding neighborhood indices $\tau ^ { ( s ) , v }$ and $\pmb { T } ^ { ( s ) , m }$ Only at the first scale do visible neighborhoods supply direct point input to the encoder as sets of input-point coordinates expressed relative to their centers, while the corresponding sets around masked first-scale centers are withheld as reconstruction targets. The first encoder stage maps the visible first-scale neighborhoods to $\mathbf { T } _ { 1 } ^ { v } .$ , and each later stage aggregates groups of visible child tokens indexed by $\tau ^ { ( s ) , v }$ to produce $\{ \mathbf { T } _ { s } ^ { v } \} _ { s = 1 } ^ { \overline { { S } } }$ . These per-scale visible tokens serve as inputs to the scale-specific projectors used for cross-scale consistency learning.

![](images/58e8050978779ded0e4dfa02ab8d6e7c826ef750f49823a60fcaccfb02eaed7f.jpg)  
Figure 1: Annotators routinely cross-check the same plant region across scales to resolve local ambiguity and confirm semantic identity. Inspired by this process, our cross-scale consistency learning (C2L) aligns representations of matched regions across scales, yielding scale-consistent point cloud understanding.

## 2.3.2. Cross-scale Consistency Learning

When examining a plant point cloud, human observers relate fine-scale organ details to organ arrangement and whole-plant architecture to assess whether the evidence remains coherent across scales. To encode an analogous relation, cross-scale consistency learning maps visible tokens from each encoder stage into a shared embedding space, groups projected representations of corresponding plant regions through nearest-center associations, and regularizes their within-group consistency as shown in Figure 3.

Feature projection Multi-scale token representations differ not only in spatial extent but also in their feature dimensionality. To make tokens from diferent representation levels directly comparable, a two-layer MLP projection network is applied at each level to map them into a shared embedding space. The projected features are $\ell _ { 2 }$ normalized, yielding hyperspherical embeddings suitable for metric-based comparison. The projected representations are denoted as $\mathbf { Z } _ { s } ^ { v } \in$ $\mathbb { R } ^ { N _ { s } ^ { v } \times D }$

Cross-scale token grouping Projected representations at finer scales $\mathbf { \xi } ( s ~ < ~ S )$ are grouped around reference representations defined at the coarsest scale � using rank-based nearest center selection (Cover and Hart, 1967). For the �-th reference representation, the corresponding center point is denoted as $\mathbf { p } _ { i } ^ { ( S ) } \in \mathbf { P } _ { S } ^ { v }$ . At a finer scale $s \ < \ S .$ , candidate centers $\mathbf { p } _ { j } ^ { ( s ) } \in \mathbf { P } _ { s } ^ { v }$ are ranked by their Euclidean distance to $\mathbf { p } _ { i } ^ { ( S ) }$ . The distance is computed as $d _ { i j } ^ { ( s ) } = \left. \mathbf { p } _ { i } ^ { ( S ) } - \mathbf { p } _ { j } ^ { ( s ) } \right. _ { 2 }$ The top $\lceil N _ { s } ^ { v } / N _ { S } ^ { v } \rceil$ finer-scale centers are retained for each coarsest-scale reference, following nearest neighbor grouping practice in point cloud encoders (Wang et al., 2019b; Yu et al., 2022; Zhang et al., 2022). The selected indices form:

$$
\mathcal { G } _ { i } ^ { ( s ) } = \left\{ j \in \{ 1 , \dots , N _ { s } ^ { v } \} \Bigg | \mathrm { r a n k } \Big ( d _ { i j } ^ { ( s ) } \Big ) \leq \Bigg \lceil \frac { N _ { s } ^ { v } } { N _ { S } ^ { v } } \Bigg \rceil \right\}\tag{1}
$$

where $\mathrm { r a n k } ( d _ { i j } ^ { ( s ) } )$ denotes the ascending rank of $d _ { i j } ^ { ( s ) }$

Based on the index sets $\{ \mathcal { G } _ { i } ^ { ( s ) } \} _ { s < S }$ constructed at all finer scales, a cross-scale group for the �-th reference representation is defined as

$$
\left\{ \mathbf { Z } _ { S } ^ { v } ( i ) \right\} \cup \bigcup _ { s < S } \left\{ \mathbf { Z } _ { s } ^ { v } ( j ) \mid j \in \mathcal { G } _ { i } ^ { ( s ) } \right\}\tag{2}
$$

where projected representations at finer scales may be associated with multiple reference groups. Each group thus links finer-scale projections to a coarsest-scale reference token, making the cross-scale association explicit for consistency supervision.

Consistency supervision Based on the reference-centered groups, a cross-scale consistency learning loss is introduced to encourage agreement among representations within each group. Directly enforcing absolute agreement among projections within each group is prone to degenerate solutions, where representations collapse to a constant vector. Instead, consistency at the �-th scale is imposed in a relative manner through a batch-wise N-pair objective (Sohn, 2016):

![](images/cd101ef0704f1c83f6e180d847517c81c3f2f841f16e7784e20fde453f637380.jpg)  
Figure 2: PlantC2USeg provides an end-to-end pipeline for unified stem–leaf semantic segmentation and leaf instance segmentation on plant point clouds. (a) Multi-scale neighborhood construction organizes the input point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ into a hierarchy of centers and their neighborhoods, with the visible subsets shown at each scale. The purple target marks one schematic first-scale neighborhood withheld for reconstruction. (b) A hierarchical encoder produces visible tokens across scales, and scale-specific projectors map them into a shared embedding space where cross-scale consistency relates representations of corresponding plant regions. Information-restricted decoding reuses these projected visible representations as keys and values to condition masked queries that reconstruct first-scale targets, one of which is illustrated in panel (a). (c) The few-shot setting illustrates unified fine-tuning, in which semantic supervision and complementary instance regularization in feature and coordinate spaces adapt the encoder initialized by cross-scale pre-training. (d) Semantic output and predicted ofsets enter progressive segmentation, which shifts leaf points toward estimated centers before grouping them into leaf instances. Thresholds inherited from fine-tuning guide the grouping, and outer-to-inner processing is used for plants with broad leaves.

![](images/0e2d56e47c6349761acb909518a5ed8c0767eb802d8a249b73e2fdace3d7bb89.jpg)  
Figure 3: Cross-scale consistency learning for visible plant regions. (a) Visible tokens from each encoder scale are mapped by scalespecific projectors from their original feature spaces into a shared embedding space. (b) In Euclidean space, each coarsest-scale reference center $\mathbf { p } _ { i } ^ { ( S ) }$ selects the nearest finer-scale visible centers, with $\lceil N _ { s } ^ { v } / N _ { S } ^ { v } \rceil$ centers selected at scale �. The selected indices form $\mathcal { G } _ { i } ^ { ( s ) }$ , and the corresponding projected tokens, such as $\{ \mathbf { Z } _ { 1 } ^ { v } ( j ) \} _ { j \in \mathcal { G } _ { i } ^ { ( 1 ) } }$ and $\{ \mathbf { Z } _ { 2 } ^ { v } ( j ) \} _ { j \in \mathcal { G } _ { i } ^ { ( 2 ) } }$ ) , are grouped with the reference token $\mathbf { Z } _ { S } ^ { v } ( i )$ . (c) In the shared embedding space, the consistency loss pulls the reference token toward its within-group positives and pushes it away from negatives from other groups and other samples.

$$
\mathcal { L } _ { \mathrm { c o n s } } ^ { s } = - \sum _ { i \in B _ { S } } \sum _ { j \in \mathcal { G } _ { i } ^ { ( s ) } } \log \frac { \exp \left( \alpha \mathbf { Z } _ { s } ^ { v } ( j ) ^ { \top } \mathbf { Z } _ { S } ^ { v } ( i ) \right) } { \sum _ { b \in { \mathcal { B } _ { S } } } \exp \left( \alpha \mathbf { Z } _ { s } ^ { v } ( j ) ^ { \top } \mathbf { Z } _ { S } ^ { v } ( b ) \right) }\tag{3}
$$

where $B _ { S }$ indexes the reference groups available in the minibatch, $\mathcal { G } _ { i } ^ { ( s ) }$ contains the finer-scale projections associated with reference group $i ,$ and $\alpha = 6 4$ rescales the normalized similarities, following prior hyperspherical metric-learning (Deng et al., 2019) and point-cloud representation learning (Rao et al., 2023) practice. In implementation, the summed terms are averaged over the positive associations $\{ ( i , j ) \mid i \in$ $B _ { S } , j \in \mathcal { G } _ { i } ^ { ( s ) } \}$ , and the final consistency loss is obtained by averaging $\dot { \mathcal { L } } _ { \mathrm { c o n s } } ^ { s }$ over all finer scales $s < S$

## 2.3.3. Information-Restricted Decoding

Cross-scale consistency constrains finer-scale representations to align with the coarsest projections, yet provides no direct supervision on the validity of the final-stage encoded features themselves. As a result, suboptimal coarse representations may consistently bias learning throughout the hierarchy. Decoding is therefore used to anchor the encoded features through masked-neighborhood reconstruction, with information restrictions applied to reduce mutual interactions among masked tokens.

Self-Attention Decoding and its limitations in Point Clouds In self-attention decoders, reconstruction is performed through �−1 decoding stages operating from coarse to fine scales, as illustrated in Figure 4(a). Decoding is initialized by combining the coarsest visible tokens $\mathbf { T } _ { S } ^ { v }$ with learnable mask tokens $\mathbf { T } _ { S } ^ { m } \in \mathbb { R } ^ { N _ { S } ^ { m } \times D }$ , and self-attention is applied to infer masked representations. At each stage, selfattention is performed globally over the concatenated token set, such that queries, keys, and values are jointly derived from both visible and masked token types. The updated tokens are propagated to finer scales following the feature propagation strategy in PointNet++ (Qi et al., 2017b), after which they are concatenated with visible tokens from the corresponding encoder scale to serve as input for the next stage. The decoder output at the final (� − 1)-th stage is used to reconstruct the masked local neighborhoods by applying a reconstruction head to the masked tokens, with supervision provided by the Chamfer Distance. In point clouds, geometry is encoded explicitly through positional embeddings. Unconstrained self-attention decoding permits masked tokens to exchange meaningful spatial cues within the attention mechanism, thereby allowing positional correlations to influence reconstruction more strongly than semantic cues from visible neighborhoods. To mitigate these unrestricted interactions, an information-restricted decoder based on cross-attention is introduced.

Restricted information flow The information-restricted formulation preserves the coarse-to-fine decoding structure while enforcing asymmetric information flow through crossattention (Figure 4(b)). Masked tokens are initialized as learnable embeddings, with their role in decoding restricted to querying the projected visible representations from the consistency learning objective. The cross-attention block design is inspired by the vanilla cross-attention decoder introduced in CrossMAE (Fu et al., 2025). This design is extended to hierarchical point cloud decoding, and an attention mask is applied as an additional constraint to filter out invalid interactions during decoding. Batch-wise processing under multi-scale masking requires padding masked token embeddings at scales $s < S$ to the maximum masked length $\widehat { N } _ { s } ^ { m } = \operatorname* { m a x } _ { b } N _ { s , b } ^ { m }$ . In each block, attention is computed between padded masked tokens and the projected visible representations $\mathbf { Z } _ { s } ^ { v }$ via an information-restricted cross-attention, computed as:

$$
\begin{array} { r l } & { \mathrm { I R \mathrm { - } C r o s s A t t n } ( \widetilde { \mathbf T } _ { s } ^ { m } , \mathbf Z _ { s } ^ { v } ) } \\ & { = \mathbf M _ { s } \odot \left[ \mathrm { s o f t m a x } \left( \frac { ( \widetilde { \mathbf T } _ { s } ^ { m } \mathbf W _ { Q } ) ( \mathbf Z _ { s } ^ { v } \mathbf W _ { K } ) ^ { \top } } { \sqrt { d } } \right) ( \mathbf Z _ { s } ^ { v } \mathbf W _ { V } ) \right] } \end{array}\tag{4}
$$

where $\widetilde { \mathbf { T } } _ { s } ^ { m }$ denotes a padded embedding of the masked tokens $\mathbf { T } _ { s } ^ { m } , ~ \mathbf { M } _ { s } ~ \in ~ \{ 0 , 1 \} ^ { \widehat { N } _ { s } ^ { m } \times 1 }$ equals one at valid masked-token positions and zero at padded positions, and � denotes the dimensionality of each attention head. The mask is broadcast over the feature dimension to suppress padded outputs.

As shown in Figure 4 (c), dense pairwise interactions among all tokens are induced by self-attention, leading to quadratic complexity with respect to the token count and redundant computation over padding and masked positions. In Figure 4 (d), attention is confined to masked-to-visible interactions, preventing information exchange among masked tokens. This formulation decouples feature extraction from decoding, allowing reconstruction to be driven by geometrically grounded visible representations.

Reconstruction aligned with dense prediction tasks Dense prediction tasks such as segmentation rely on stable fine-scale spatial cues at early layers, motivating masked tokens to be updated by querying visible representations down to the finest scale. The final masked-query representation is therefore denoted as $\widetilde { \mathbf { T } } _ { 1 } ^ { m }$ . A single linear projection reconstructs the masked first-scale neighborhoods as $\widehat { \mathbf { P } } _ { 1 } ^ { m } =$ $\widetilde { \mathbf { T } } _ { 1 } ^ { m } \mathbf { W } _ { \mathrm { r e c } }$ , where $\mathbf { W } _ { \mathrm { r e c } }$ denotes the learnable projection of the reconstruction head. Reconstruction is supervised using the $\ell _ { 2 }$ Chamfer Distance (Fan et al., 2017)

$$
\mathcal { L } _ { \mathrm { r e c } } = \sum _ { \mathbf { p } \in \mathbf { P } _ { 1 } [ \cal { I } ^ { ( 1 ) , m } ] } \operatorname* { m i n } _ { \hat { \mathbf { q } } \in \hat { \mathbf { P } } _ { 1 } ^ { m } } \| \mathbf { p } - \widehat { \mathbf { q } } \| _ { 2 } ^ { 2 } + \sum _ { \hat { \mathbf { q } } \in \hat { \mathbf { P } } _ { 1 } ^ { m } } \operatorname* { m i n } _ { \mathbf { p } \in \mathbf { P } _ { 1 } [ \cal { I } ^ { ( 1 ) , m } ] } \| \widehat { \mathbf { q } } - \mathbf { p } \| _ { 2 } ^ { 2 }\tag{5}
$$

## 2.4. Unified few-shot fine-tuning

Unified fine-tuning adapts the encoder initialized by cross-scale pre-training in one stage for stem–leaf semantic segmentation and leaf instance segmentation under full-shot and few-shot supervision. At each encoder scale, representations learned for local neighborhoods are interpolated to the input points, concatenated with the coordinates of those points, and processed by a scale-specific MLP. The three resulting point-level outputs are averaged to form the shared point-wise embedding �.

## 2.4.1. Semantic supervision

The semantic head supports stem–leaf semantic segmentation through a point-wise classification objective that updates the shared encoder. For $N _ { \mathrm { s e m } }$ valid points, the head converts the globally augmented point representation into class probabilities $\mathbf { Y } _ { i }$ (Figure 5(a)), and the mean negative log-likelihood is

![](images/9ace0eb74580adeb03d938d469ef4195e5f91ef5cb0fb40c332dff40bd17936a.jpg)  
Figure 4: Comparison between self-attention decoding and information-restricted decoding for hierarchical point cloud reconstruction. (a) Self-attention decoding that freely mixes tokens across spatial regions. (b) Information-restricted decoding that reconstructs masked neighborhoods by querying visible representations only. (c) Self-attention incurs quadratic complexity due to all-to-all attention. (d) Structured restricted attention map.

$$
\mathcal { L } _ { \mathrm { s e m } } = - \frac { 1 } { N _ { \mathrm { s e m } } } \sum _ { i = 1 } ^ { N _ { \mathrm { s e m } } } \log Y _ { i , y _ { i } ^ { * } }\tag{6}
$$

where $y _ { i } ^ { * }$ is the ground-truth semantic label and $Y _ { i , y _ { i } ^ { * } }$ is its predicted probability.

## 2.4.2. Instance-center regularization

For leaf instance segmentation, the coupled design constrains learned embeddings around leaf centers in feature space and supervises predicted ofsets with displacements to the matching leaf centers in coordinate space. Discriminative instance embedding (De Brabandere et al., 2017) draws embeddings of leaf points toward their feature centers and separates centers that represent diferent leaves. Centerdirected ofset regression (Jiang et al., 2020; Chen et al., 2021) uses the globally augmented point representation to predict a three-dimensional displacement from each leaf point to the center of that leaf in coordinate space (Figure 5(b)). Coordinate ofsets provide positional guidance when spatially distinct leaves exhibit similar fine-scale shape cues, whereas proximity in feature space indicates instance afinity when leaf overlap makes center-directed ofsets ambiguous.

Feature-space regularization Let $\mathbf { F } \in \mathbb { R } ^ { N _ { \mathrm { f e a t } } \times F }$ contain the point-wise embeddings for the $N _ { \mathrm { f e a t } }$ points evaluated by feature regularization. For each leaf instance $\ell , \boldsymbol { A } _ { \ell }$ contains the indices of its leaf points and excludes stem points. The corresponding feature center is $\begin{array} { r } { \pmb { \mu } _ { \ell } = | \mathcal { A } _ { \ell } | ^ { - 1 } \sum _ { i \in \mathcal { A } _ { \ell } } \mathbf { f } _ { i } } \end{array}$ . For a sample with $L > 1$ , the discriminative instance embedding loss (De Brabandere et al., 2017) formalizes within-leaf compactness, separation between centers of leaf instances, and center regularization as

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { \mathrm { f e a t } } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \frac { 1 } { | \mathcal { A } _ { \ell } | } \sum _ { i \in \mathcal { A } _ { \ell } } \left[ \| \mathbf f _ { i } - \boldsymbol \mu _ { \ell } \| _ { c } - \delta _ { s } \right] _ { + } ^ { 2 } } } \\ { ~ + \frac { 1 } { L ( L - 1 ) } \sum _ { \ell _ { a } = 1 } ^ { L } \sum _ { \ell _ { b } = 1 } ^ { L } \left[ 2 \delta _ { d } - \| \boldsymbol { \mu } _ { \ell _ { a } } - \boldsymbol { \mu } _ { \ell _ { b } } \| _ { c } \right] _ { + } ^ { 2 } ~ ( \mathop { d , . . . } ) }  \\ { { ~ + \chi _ { \mathrm { c r } } \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \lVert \boldsymbol { \mu } _ { \ell } \rVert _ { c } } } \end{array}\tag{7}
$$

![](images/94865ebff77617ee85afd5cc7d62c92353d918fbff1d3d6c30e95aab58992cfa.jpg)  
Figure 5: Unified adaptation of the pre-trained hierarchical encoder. Representations of local neighborhoods from the three encoder scales are propagated to the input points and averaged into �. Feature regularization acts directly on �, while global mean and maximum summaries augment the point-level inputs to the semantic and ofset heads. (a) The semantic head predicts probabilities �, which are compared with ground-truth labels through $\mathcal { L } _ { \mathrm { s e m } } .$ . (b) Under $\mathcal { L } _ { \mathrm { f e a t } }$ , embeddings of leaf points are drawn toward their leaf-specific feature centers $\mu _ { \ell } ,$ and centers of diferent leaves are separated. The ofset head predicts center-directed displacements from $\mathbf { p } _ { i }$ to $c _ { \ell }$ under ${ \mathcal { L } } _ { \mathrm { c o o r } }$

where $[ x ] _ { + } = \operatorname* { m a x } ( 0 , x )$ is the hinge function and feature distances use $c = 1$ , following ASIS (Wang et al., 2019a). The margin $\delta _ { s }$ bounds the distance from each embedding to its feature center, $2 \delta _ { d }$ specifies the minimum distance between centers, and $\lambda _ { \mathrm { c t r } }$ weights center regularization. When $L = 1$ the inter-instance term is omitted.

Coordinate-space ofset regression For a leaf point $\mathbf { p } _ { i } \in$ $\mathbb { R } ^ { 3 }$ assigned to leaf instance �, the target ofset is

$$
\mathbf { 0 } _ { i } ^ { * } = { \boldsymbol { \mathbf { c } } } _ { \ell } - \mathbf { p } _ { i }\tag{8}
$$

where $\begin{array} { r } { \pmb { c } _ { \ell } = \vert \mathcal { A } _ { \ell } \vert ^ { - 1 } \sum _ { i \in \mathcal { A } , } \mathbf { p } _ { i } } \end{array}$ is the center of leaf instance $\ell$ in coordinate space. For $N _ { \mathrm { c o o r } }$ evaluated points, the ofset head predicts $\hat { \mathbf { 0 } } _ { i } \in \mathbb { R } ^ { 3 }$ , and the masked L1 objective is

$$
\mathcal { L } _ { \mathrm { c o o r } } = \frac { \sum _ { i = 1 } ^ { { N _ { \mathrm { c o o r } } } } \mathbf { 1 } _ { i } \left\| \hat { \mathbf { 0 } } _ { i } - \mathbf { 0 } _ { i } ^ { * } \right\| _ { 1 } } { \sum _ { i = 1 } ^ { { N _ { \mathrm { c o o r } } } } \mathbf { 1 } _ { i } + \epsilon }\tag{9}
$$

where $\mathbf { 1 } _ { i } = 1$ for leaf points with $y _ { i } ^ { * } \in \mathcal { V } _ { \mathrm { i n s t } }$ and $\mathbf { 1 } _ { i } = 0$ otherwise, and $\epsilon = 1 0 ^ { - 6 }$ . The three objectives jointly update the shared encoder through

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e m } } + \mathcal { L } _ { \mathrm { f e a t } } + \mathcal { L } _ { \mathrm { c o o r } }\tag{10}
$$

while aggregate feature-distance statistics from fine-tuning provide the three criteria for progressive segmentation.

## 2.5. Progressive segmentation

Progressive segmentation carries the complementary coordinate and feature cues learned during unified fine-tuning into the inference of individual leaf instances. Semantic predictions select leaf points, and predicted ofsets shift their coordinates toward estimated centers according to

$$
\tilde { \mathbf { p } } _ { i } = \mathbf { p } _ { i } + \widehat { \mathbf { o } } _ { i } \qquad \widehat { y } _ { i } \in \mathcal { V } _ { \mathrm { i n s t } }\tag{11}
$$

where $\widehat { y } _ { i }$ is the predicted semantic label of point � . Planar distances after coordinate shifting determine which points enter each radial step, whereas feature embeddings guide DBSCAN partitioning, reconnection across steps, assignment of residual points, and final consolidation. Values for the three distance criteria are inherited from the statistics recorded during fine-tuning, avoiding manual threshold sweeps or recalibration on the labeled test set. Distances from points to the feature center of their instance define $\epsilon _ { \mathrm { d b } } .$ The distance from each feature center to the nearest center of another instance defines $\epsilon _ { \mathrm { r e c } } .$ , while a high quantile of these within-instance distances defines $\epsilon _ { \mathrm { m e r g e } } .$

PlantC2USeg uses outer-to-inner progression for plants with broad leaves or rosette morphology. Following radiusdecremental grouping (Roggiolani et al., 2023), this progression begins with peripheral points and advances toward the central stem region, where closely spaced leaf petioles make assignment more dificult. For plants with narrow, grasslike leaves, radial progression is disabled and all candidates are processed in a single pass. One planar center �̄ and the maximum radius are computed once from the shifted candidate set before radial processing. At step $q ,$ let $\mathcal { V } ^ { ( q ) }$ denote the indices that remain unassigned at the start of the step, so the radially eligible set is

$$
L ^ { \left( q \right) } = \left\{ i \in \mathcal { V } ^ { \left( q \right) } \bigg \vert \bigg \Vert \tilde { \mathbf { p } } _ { i , x y } - \bar { \mathbf { p } } _ { x y } \bigg \Vert _ { 2 } > r ^ { \left( q \right) } \right\}\tag{12}
$$

where $r ^ { ( q ) }$ decreases linearly from the initial radius selected for the plant morphology and supervision regime to zero.

At each radial step, DBSCAN (Ester et al., 1996) partitions the eligible feature embeddings with the L1 radius $\epsilon _ { \mathrm { d b } } ,$ while noise and other unassigned points remain available to later inward steps. The first successful partitions initialize the instance labels. Each later partition is matched to the nearest existing instance by the L1 distance between their feature centroids. It receives that instance label when the distance is below $\epsilon _ { \mathrm { r e c } }$ and receives a new label otherwise, after which the afected feature centroids are recomputed. After the final radial step, residual points are assigned to the nearest existing feature centroid without a threshold and form one new group when no centroid exists. Final consolidation completes the partition into leaf instances by merging groups whose feature centroids are separated by no more than �<sub>merge</sub>.

## 3. Experiment

## 3.1. Datasets

Plant species exhibit substantial variations in morphological structures, leading to distinct geometric characteristics in their corresponding point clouds. Experiments were conducted on a 3D soybean point cloud dataset originally introduced in (Luo et al., 2023), with additional samples used in this work, as well as two publicly available plant datasets, HR3D (Conn et al., 2017) and SYAU-Maize (Yang et al., 2024). In addition, ShapeNet Part (Yi et al., 2016) was included as a complementary benchmark to assess the generality of the proposed method beyond plant-specific datasets, given the limited semantic diversity of existing plant point cloud datasets. The characteristics of each dataset are described below.

Soybean3D The soybean point cloud dataset used in this work consists of 314 point clouds reconstructed from multiview red-green-blue (RGB) images using a phenotyping platform based on multi-view stereo. Soybean plants were grown in a controlled indoor environment and imaged repeatedly on Mondays, Wednesdays, and Fridays for three weeks using an RGB camera (LUMIX DMC-G7W, Panasonic, Japan), covering the VC to V2 growth stages. The dataset was originally introduced in Ef-3DPSeg (Luo et al., 2023) and was extended in this work with additional reconstructed samples. Manual annotations for stem–leaf semantic segmentation and leaf instance segmentation are available for 145 samples, of which 120 were used for training and 25 for testing, following (Luo et al., 2023) and (Tian et al., 2025).

HR3D HR3D consists of 546 laser-scanned plant point clouds covering three crop species: tomato (312 samples), sorghum (129 samples), and tobacco (105 samples). These samples were captured using a blue-laser scanner (Edge ScanArm HD, FARO, USA). The point clouds were acquired under diverse growth conditions, including ambient light, shade, high heat, high light, and drought, over a period of 20–30 days. Each sample was down-sampled to 4,096 points and augmented using repeated FPS integration following

PlantNet (Li et al., 2022b), yielding 3,640 training and 1,820 testing point clouds.

SYAU-Maize SYAU-Maize is a maize point cloud dataset consisting of 428 samples from five varieties (Xian Yu 335, LD 145, LD 502, LD586, and LD 1281). The plants were grown in field environments and scanned indoors using a high-precision 3D scanner (FreeScan X3, Tianyuan Inc., China). The dataset contains both complete plant structures and incomplete point clouds caused by acquisition artifacts, which enables the evaluation of segmentation robustness under partial observations. Following the sample-selection strategy of Deformation3D (Yang et al., 2024), a training set containing 22 samples was constructed, and the same samples were used for all compared methods.

ShapeNet Part ShapeNet Part is a widely-used benchmark for 3D part segmentation, derived from ShapeNetCore 3D CAD models and covering multiple object categories. The dataset contains 16,881 samples from 16 categories, labeled with 50 semantic parts in total. Each sample is labeled with 2 to 5 semantic parts, such as wings, body, and tail for airplanes or seat, back, and legs for chairs.

## 3.2. Implementation details

The pre-training experiments used cross-scale representation learning without manual annotations. The resulting models were then fine-tuned on downstream plant segmentation tasks under both full and limited supervision, as well as on synthetic part segmentation. All experiments were implemented in PyTorch. Pre-training and full-shot finetuning were performed on an NVIDIA GeForce RTX 3090 GPU, while few-shot fine-tuning used an NVIDIA GeForce RTX 3060 GPU.

Pre-training Pre-training was conducted on the Soybean3D dataset using the same train-test split as (Tian et al., 2025), yielding 2,890 augmented samples for unsupervised representation learning. For synthetic part segmentation, pretraining was instead performed on the ShapeNet dataset (Chang et al., 2015), which contains 57,448 synthetic 3D shapes from 55 common categories. Across all pre-training experiments, input point clouds were randomly sampled to 2048 points. Local neighborhoods were constructed at $S = 3$ spatial scales with center counts $\{ N _ { s } \} _ { s = 1 } ^ { 3 } ~ = ~ \{ 1 0 2 4 , 5 1 2 ,$ 128} and �-NN sizes $\{ k _ { s } \} _ { s = 1 } ^ { 3 } = \{ 1 6 , 8 , 8 \}$ . Multi-scale masking was applied at a ratio of 0.8 at the coarsest scale, $s \ = \ S$ . The encoder followed Point-M2AE (Zhang et al., 2022), with encoder embedding dimensions $\{ C _ { s } \} _ { s = 1 } ^ { 3 } = \{ 9 6 $ 192, 384} and five Transformer blocks per stage. Feature projection at the �-th scale used a two-layer MLP [� , 128, �] for cross-scale consistency learning. The lightweight decoder had the same number of stages � = 3 and used one block per stage with dimension $D = 3 8 4$

Pre-training was configured for up to 400 epochs using AdamW (Loshchilov and Hutter, 2019) with an initial learning rate of $1 0 ^ { - 3 }$ and a weight decay of 0.05. A linear warmup was applied for the first 40 epochs starting from

$1 0 ^ { - 6 } .$ , followed by cosine decay with a minimum learning rate of $1 0 ^ { - 6 }$ . A batch size of 48 was used for both datasets. Following Point-M2AE (Zhang et al., 2022), early stopping terminated the adopted runs at epoch 300.

Plant Segmentation The limited-label protocol was defined at the sample level, with each few-shot sample fully annotated for both semantic and instance segmentation. This sample-level protocol avoided the annotation efort and sampling sensitivity associated with selecting sparse point labels across plant organs. For downstream plant segmentation, Soybean3D and SYAU-Maize were preprocessed to 16,384 points using 3DEPS (Li et al., 2022b), and nontest samples were expanded tenfold by repeated FPS integration. The released 4,096-point HR3D samples were used directly. PlantC2USeg used 4,096 input points, whereas each comparator retained its method-specific input point count. Before metric computation, all predictions were mapped by nearest-neighbor interpolation to the original-resolution point clouds for Soybean3D and SYAU-Maize and to the released evaluation point clouds for HR3D. The propagated feature dimension was fixed to $F = 3 8 4$ , and the semantic and ofset heads used MLP hidden dimensions [96, 48] with a dropout rate of 0.5. For multi-species fine-tuning, a learned category encoding was appended to the global feature context supplied to both prediction heads after � was formed.

Progressive grouping used five radial steps and initialradius ratios of 0.7, 0.6, and 0.5 for full-shot, 20-shot, and 10- shot settings, respectively. Radial progression was disabled for narrow-leaf species by setting the ratio to 0. In limitedsupervision settings, including the 22-sample SYAU-Maize setting, threshold statistics were accumulated during finetuning with an EMA decay of 0.99, whereas full-shot inference used the non-EMA values saved during training. Median aggregation was used for the reconnection threshold across all supervision settings. The DBSCAN and consolidation aggregation quantiles were reduced from 0.5 and 0.99 under full supervision to 0 and 0.9 under limited supervision, respectively. For HR3D, threshold values were maintained separately for each species.

PlantC2USeg was optimized using AdamW with a base learning rate of $2 \times 1 0 ^ { - 4 }$ followed by cosine decay. Fullshot schedules used a weight decay of 0.05, with warm-up periods of 50 and 25 epochs and maximum durations of 1500 and 800 epochs for Soybean3D and HR3D, respectively. Limited-supervision schedules reduced the weight decay to 0.01 and extended training to 2.5 and 5 times the full shot duration for 20-shot and 10-shot settings, respectively, with longer warm-up periods to mitigate underfitting. The encoder was unfrozen immediately for Soybean3D few-shot fine-tuning and after 20 epochs for full-shot training and cross-species transfer to HR3D and SYAU-Maize. The 22- sample SYAU-Maize schedule used a weight decay of 0.01, a 200-epoch warm-up, and a maximum duration of 6000 epochs.

Object Part Segmentation For part segmentation, the same parameters and training protocol as Point-M2AE (Zhang et al., 2022) were adopted.

## 3.3. Evaluation metrics

Semantic segmentation metrics Precision (Prec), Recall (Rec), F1-score (F1), and Intersection over Union (IoU) were used to measure the semantic segmentation performance at the point level for each semantic class:

$$
\mathrm { P r e c } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } }\tag{13}
$$

$$
{ \mathrm { R e c } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } }\tag{14}
$$

$$
\mathrm { F } 1 = { \frac { 2 \cdot \mathrm { P r e c } \cdot \mathrm { R e c } } { \mathrm { P r e c } + \mathrm { R e c } } }\tag{15}
$$

$$
\mathrm { I o U } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } }\tag{16}
$$

where TP, FP, and FN denote the numbers of true positive, false positive, and false negative points, respectively. Precision and recall characterize prediction reliability and ground-truth coverage, respectively. The F1-score is the harmonic mean of precision and recall, and serves to summarize their trade-of. IoU quantifies the overlap between the predicted and ground-truth regions for each semantic category.

Instance segmentation metrics Instance segmentation performance was evaluated using mean precision (mPrec), mean recall (mRec), mean coverage (mCov), and mean weighted coverage (mWCov).

$$
\mathrm { m P r e c } = \frac { 1 } { \left| \mathcal { V } _ { \mathrm { i n s t } } \right| } \sum _ { y \in \mathcal { V } _ { \mathrm { i n s t } } } \frac { \left| \mathrm { T P } _ { \mathrm { i n s } } ^ { ( y ) } \right| } { \left| \hat { I } ^ { ( y ) } \right| }\tag{17}
$$

$$
\mathrm { m R e c } = \frac { 1 } { \left| \mathcal { N } _ { \mathrm { i n s t } } \right| } \sum _ { y \in \mathcal { V } _ { \mathrm { i n s t } } } \frac { \left| \mathrm { T P } _ { \mathrm { i n s } } ^ { ( y ) } \right| } { \left| T ^ { ( y ) } \right| }\tag{18}
$$

where $\mathcal { V } _ { \mathrm { i n s t } } \subset \mathcal { V }$ denotes the set of instance-aware semantic categories. For each category $y ~ \in ~ \mathfrak { V } _ { \mathrm { i n s t } } , ~ \tau ^ { ( y ) }$ and $\hat { \cal T } ^ { ( y ) }$ represent the sets of ground-truth and predicted instances, respectively. $\mathrm { T P } _ { \mathrm { i n s } } ^ { ( y ) }$ is defined as the number of predicted instances in category � whose maximum IoU with any ground-truth instance in the same category exceeds 0.5. Mean coverage (mCov) was computed per instance-aware semantic category and then averaged across categories:

$$
\operatorname { m C o v } = { \frac { 1 } { | I ^ { ( y ) } | } } \sum _ { \ell \in I ^ { ( y ) } } \operatorname* { m a x } _ { j } \operatorname { I o U } \left( { \mathcal { A } } _ { \ell } , { \hat { \mathcal { A } } } _ { j } \right)\tag{19}
$$

where $\mathbf { \mathcal { A } } _ { \ell }$ denotes the set of points belonging to the $\ell -$ th ground-truth instance. For each instance $\ell ,$ the IoU is defined as the maximum overlap between the ground-truth $\mathbf { \mathcal { A } } _ { \ell }$ and the predicted instance point sets $\hat { \mathcal { A } } _ { j }$ in the same semantic category.

Mean weighted coverage (mWCov) extends mCov by introducing instance-size weighting:

$$
\operatorname* { m W C o v } = \sum _ { \ell \in { \cal I } ^ { ( y ) } } \omega _ { \ell } \operatorname* { m a x } _ { j } \operatorname { I o U } \left( \mathcal { A } _ { \ell } , \hat { \mathcal { A } } _ { j } \right)\tag{20}
$$

$$
\omega _ { \ell } = \frac { | \mathcal { A } _ { \ell } | } { \sum _ { \ell ^ { \prime } \in \cal I ^ { ( y ) } } | \mathcal { A } _ { \ell ^ { \prime } } | }\tag{21}
$$

where the weight $\omega _ { \ell }$ is proportional to $\begin{array} { r } { \mathbf { \mathcal { A } } _ { \ell } | , } \end{array}$ such that instances with more points contribute more to the overall mWCov score. mWCov emphasizes spatial coverage on dominant plant structures and is less sensitive to small fragmented instances.

Part segmentation metrics Following common practice on the ShapeNet Part dataset, two variants of mean IoU were adopted, including category-averaged mIoU and instanceaveraged mIoU. The category-averaged mIoU averages IoU scores over all part categories, while the instance-averaged mIoU averages IoU scores over all samples.

For all metrics, higher values indicate better segmentation performance.

## 4. Results

## 4.1.1. Soybean3D

Full-shot stem–leaf semantic predictions for one soybean plant imaged on May 06, May 13, and May 20 difered in class allocation at narrow stem–leaf junctions (Figure 6). In region b on May 13, PlantC2USeg and PlantNet retained more of the ground-truth small-leaf area, whereas Ef3DPSeg, Soybean-PCMAE, and PSegNet showed greater leaf-to-stem leakage. On the May 13 sample, PlantC2USeg and PlantNet reached 92.5% and 93.3% IoU, respectively, whereas Ef3DPSeg, Soybean-PCMAE, and PSegNet remained below 87%. In region c on May 20, PlantNet misclassified part of the ground-truth stem as leaf, whereas PlantC2USeg preserved a continuous visible stem path. For PlantNet, regions b and c contrast small-leaf retention with stem-to-leaf leakage.

Dataset-level evaluation on Soybean3D covered stem– leaf semantic and leaf instance segmentation under fullshot, 20-shot, and 10-shot supervision (Table 1), with the instance results analyzed later in Section 4.2.1. Under full supervision, PlantC2USeg combined cross-scale pre-training with joint fine-tuning to achieve the highest semantic F1 and IoU, reaching 95.69% and 91.91%, respectively. Among the pre-trained alternatives that adapt the two tasks separately, Soybean-PCMAE (Tian et al., 2025) uses staged fine-tuning and reached 90.97% IoU, whereas Ef-3DPSeg (Luo et al.,

2023) trains separate task networks and reached 88.41%. PSegNet (Li et al., 2022a), a fully supervised joint baseline with dual-scale features, reached 89.29% IoU. With the downstream architecture and joint fine-tuning setup held fixed, pre-training increased IoU by 1.72 percentage points. The increase was driven mainly by recall, which rose from 92.98% to 95.01%, while precision remained similar at 96.64% for the scratch model and 96.40% for PlantC2USeg.

Figure 7 extends the same three Soybean3D examples from Figure 6 to 20-shot and 10-shot stem–leaf semantic segmentation. In region b of the May 13 sample, PlantC2USeg retained more of the ground-truth small-leaf area as leaf, with less leaf-to-stem leakage than Soybean-PCMAE at both budgets and Deformation3D at 20 shots. For the complete May 13 sample, PlantC2USeg exceeded every alternative shown at the same annotation budget by at least 11.7 percentage points in sample-level IoU. In region c on May 20, PlantC2USeg preserved more of the ground-truth leaf tissue adjoining the stem, whereas Soybean-PCMAE at both budgets and 20-shot Deformation3D extended the stem label into this area.

At the dataset level, PlantC2USeg retained the highest semantic F1 and IoU under 20-shot supervision on Soybean3D (Table 1), reaching 94.47% and 89.78%, respectively. The 20-shot IoU exceeded the full-shot results of PlantNet (Li et al., 2022b), PSegNet (Li et al., 2022a), and Ef-3DPSeg (Luo et al., 2023) by 0.49–2.28 percentage points, and F1 was also higher across this group. At the same annotation count, PlantC2USeg exceeded Deformation3D (Yang et al., 2024) by 9.08 percentage points in IoU, supporting representation transfer over deformationbased adaptation in this comparison. With the downstream setup fixed, pre-training increased IoU over the Hierarchical Transformer trained from scratch by 2.63 percentage points at 20 shots and 1.24 percentage points at 10 shots, extending the full-shot benefit across supervision levels. At 10 shots, Soybean-PCMAE (Tian et al., 2025) reached 84.75% IoU compared with 83.19% for PlantC2USeg and was also higher in F1 and recall. PlantC2USeg retained the highest precision at 90.72% among the three 10-shot methods.

## 4.1.2. HR3D

Figure 8 compares full-shot and few-shot stem–leaf predictions on selected tobacco, tomato, and sorghum samples from HR3D. In region a of the tobacco example, full-shot and 20-shot PlantC2USeg recovered the complete emerging leaf. Full-shot PSegNet and PlantNet recovered only the part of the emerging leaf adjoining the left leaf, while 10- shot PlantC2USeg recovered less of the same structure than either baseline. The 20-shot Deformation3D model instead assigned the emerging leaf to the sorghum leaf class. Region b of the tomato example showed bidirectional leakage around a narrow stem. Full-shot PSegNet assigned the entire narrow stem to the tomato leaf class, while full-shot PlantNet and 20-shot PlantC2USeg showed partial stem-to-leaf leakage. Both 10-shot PlantC2USeg and 20-shot Deformation3D extended the stem class into surrounding leaves. Full-shot PlantC2USeg preserved the complete stem.

![](images/36fd54f42e51660f363d2cc4108af412366903ebf40a8a4cf91b267c56a7a344.jpg)  
Figure 6: Rows show one Soybean3D plant imaged on May 06, May 13, and May 20 from top to bottom. Columns from left to right are GT (ground-truth reference) followed by the full-shot stem–leaf semantic predictions from Ours (PlantC2USeg), Ef3DPSeg, Soybean PCMAE, PSegNet, and PlantNet, with displayed values denoting sample-level IoU (%). Regions a, b, and c mark a narrow stem–leaf junction, a small apical leaf, and stem continuation beside a leaf, respectively.

HR3D extends the dataset-level semantic evaluation from soybean to tobacco, tomato, and sorghum under fullshot, 20-shot, and 10-shot supervision (Table 2). Leaf instance results from the same evaluation are reported in the table and examined in Section 4.2.2. Under full supervision, PlantC2USeg exceeded PSegNet (Li et al., 2022a), the strongest plant-specific joint comparator in Table 2, by 1.80 percentage points in aggregate IoU and achieved higher F1 and IoU for all three species. For tomato, PlantC2USeg paired 0.75 percentage points higher recall with 0.22 percentage points lower precision than PSegNet, whereas sorghum showed the reverse direction of higher precision and lower recall. Despite these opposing precision–recall patterns, PlantC2USeg achieved higher F1 and IoU for both species.

At 20 shots, PlantC2USeg transfers a pre-trained representation, whereas Deformation3D (Yang et al., 2024) uses deformation-based adaptation from the same 20 original annotated HR3D samples. PlantC2USeg reached 81.50% aggregate IoU, 15.38 percentage points above Deformation3D. Across species, the diference ranged from 7.95 percentage points for tomato to 19.80 for tobacco and was also large for sorghum. The annotation-eficiency advantage of transferring a pre-trained representation over deformationbased adaptation thus persisted across the evaluated HR3D species.

Against the full-shot general-purpose joint methods SGPN (Wang et al., 2018) and ASIS (Wang et al., 2019a), transfer of a pre-trained representation preserved competitive semantic performance with substantially fewer labels. The 20-shot and 10-shot PlantC2USeg models exceeded SGPN by 13.55 and 10.46 percentage points in aggregate IoU, respectively. The 20-shot model was also within 1.50 percentage points of ASIS in IoU, with a similarly close F1, comparable recall, and lower precision.

## 4.1.3. SYAU-Maize

Figure 9 compares semantic predictions under the 22- sample Deformation3D protocol (Yang et al., 2024), focusing on leaf-to-stem errors at a cotyledon near abscission and in leaf-sheath tissue at the collar. The cotyledon in region a was assigned almost entirely to stem by PSegNet and only at the portion adjoining the stem by PlantNet, whereas PlantC2USeg and both Deformation3D variants retained the organ as leaf. Regions b and c show the adjoining leaf sheath closely appressed to the culm at the collar. PlantC2USeg followed the ground-truth collar edge in region b more closely than PSegNet, which showed the clearest stem expansion into this locally stem-like tissue. The PlantC2USeg boundary in region c remained closer to ground truth than those of all four alternatives, which extended stem assignments into leaf tissue at the collar.

Table 1  
Quantitative results of stem–leaf and leaf instance segmentation on the Soybean3D dataset (%). A checkmark denotes joint optimization of stem–leaf semantic and leaf instance segmentation within a single downstream model. H. Transformer denotes the Hierarchical Transformer trained from scratch. The best results are in boldface, and the 2nd best results are underlined.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Joint</td><td colspan="4">Semantic Segmentation</td><td colspan="4">Instance Segmentation</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>IoU</td><td>mCov</td><td>mWCov</td><td>mPrec</td><td>mRec</td></tr><tr><td>Full-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PlantNet</td><td>√</td><td>95.13</td><td>91.30</td><td>93.09</td><td>87.50</td><td>77.53</td><td>87.40</td><td>88.29</td><td>76.86</td></tr><tr><td>PSegNet</td><td>√</td><td>95.78</td><td>92.71</td><td>94.17</td><td>89.29</td><td>81.30</td><td>87.89</td><td>89.92</td><td>83.92</td></tr><tr><td>Eff-3DPSeg</td><td></td><td>94.80</td><td>92.58</td><td>93.65</td><td>88.41</td><td>81.51</td><td>88.90</td><td>79.78</td><td>86.67</td></tr><tr><td>Soybean-PCMAE</td><td></td><td>95.95</td><td>94.40</td><td>95.15</td><td>90.97</td><td>85.85</td><td>91.90</td><td>89.96</td><td>91.37</td></tr><tr><td>H. Transformer</td><td>√</td><td>96.64</td><td>92.98</td><td>94.70</td><td>90.19</td><td>85.54</td><td>92.11</td><td>94.56</td><td>88.63</td></tr><tr><td>PlantC2USeg</td><td>√</td><td>96.40</td><td>95.01</td><td>95.69</td><td>91.91</td><td>89.94</td><td>94.62</td><td>96.39</td><td>94.12</td></tr><tr><td>20-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Soybean-PCMAE</td><td></td><td>93.50</td><td>95.06</td><td>94.26</td><td>89.42</td><td>77.03</td><td>84.56</td><td>90.87</td><td>81.96</td></tr><tr><td>Deformation3D</td><td></td><td>88.23</td><td>89.27</td><td>88.74</td><td>80.70</td><td>79.41</td><td>87.43</td><td>90.91</td><td>78.43</td></tr><tr><td>H. Transformer</td><td>√</td><td>94.20</td><td>91.68</td><td>92.89</td><td>87.15</td><td>74.47</td><td>82.26</td><td>75.09</td><td>79.22</td></tr><tr><td>PlantC2USeg</td><td>√</td><td>94.31</td><td>94.63</td><td>94.47</td><td>89.78</td><td>83.94</td><td>90.27</td><td>92.71</td><td>89.80</td></tr><tr><td>10-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Soybean-PCMAE</td><td></td><td>89.88</td><td>93.14</td><td>91.41</td><td>84.75</td><td>72.28</td><td>75.92</td><td>79.03</td><td>79.67</td></tr><tr><td>H. Transformer</td><td>√</td><td>88.98</td><td>90.21</td><td>89.58</td><td>81.95</td><td>72.13</td><td>79.49</td><td>75.00</td><td>75.29</td></tr><tr><td>PlantC2USeg</td><td>√</td><td>90.72</td><td>90.06</td><td>90.39</td><td>83.19</td><td>75.76</td><td>83.23</td><td>83.40</td><td>78.82</td></tr></table>

SYAU-Maize provides a shared sparse-label test of stem–leaf semantic segmentation in which every method begins with the same 22 annotated source samples (Table 3). The table also reports leaf instance segmentation, which is examined in Section 4.2.3. Under this shared protocol, PlantC2USeg adapted a pre-trained representation through joint fine-tuning, whereas the fully supervised joint methods PlantNet (Li et al., 2022b) and PSegNet (Li et al., 2022a) were trained directly from the available labels. PlantC2USeg reached 92.75% semantic IoU, 8.94 and 9.45 percentage points above PlantNet and PSegNet, respectively, and also led both methods in precision, recall, and F1. Within the evaluated sparse-label protocol, the lead across all four semantic metrics indicates that adapting a pre-trained representation was more efective than training these two joint baselines directly from the available labels.

The Deformation3D (Yang et al., 2024) variants showed a diferent precision–recall balance, with x10 slightly higher than PlantC2USeg in recall and x1k slightly higher in precision. PlantC2USeg remained higher in F1 and IoU than both variants, reaching 92.75% IoU. From x10 to x1k, the generated data volume increased 100-fold as precision rose and recall fell. F1 and IoU also declined, with IoU decreasing from 91.19% to 90.59%. The additional burden of generating and processing data therefore did not translate into an IoU gain.

## 4.2. Plant leaf instance segmentation

## 4.2.1. Soybean3D

Figure 10 compares full-shot leaf instance predictions on the same three Soybean3D examples used for the semantic comparison in Figure 6. Soybean-PCMAE fragmented the broad leaf in region a, whereas PlantC2USeg preserved it as one instance. On the complete May 06 sample, PlantC2USeg reached 99.4% mWCov, compared with 55.6% for Soybean-PCMAE. At the compact apex in region b, PlantC2USeg and PSegNet separated all three closely grouped leaves, whereas PlantNet merged the two lateral leaves. Ef3DPSeg did not separate all three leaves, and Soybean-PCMAE failed to recover the corresponding leaf instances. PlantC2USeg and Soybean-PCMAE separated all three slender apical leaves in region c. Ef3DPSeg merged the two rear leaves, PSegNet over-segmented the group into four instances, and PlantNet merged all three.

At the dataset level, full-shot PlantC2USeg led all four Soybean3D instance metrics (Table 1). Against the plantspecific joint comparator PSegNet (Li et al., 2022a), PlantC2USeg reached 94.62% mWCov compared with 87.89%. The mRec margin reached 10.20 percentage points, the largest of the four metric diferences, showing that the strongest gain concerned the recovery of ground-truth leaves. The lead across the coverage and matched-instance measures also extended to Soybean-PCMAE (Tian et al., 2025). With the downstream model fixed, pre-training improved coverage and matched-instance performance over the Hierarchical Transformer trained from scratch, including a 5.49 percentage point gain in mRec.

Figure 11 extends the same Soybean3D instance comparison to 20-shot and 10-shot supervision. Reduced supervision exposed persistent broad-leaf fragmentation for Soybean-PCMAE across all three examples, most clearly in region a. PlantC2USeg retained the broad leaf as one instance at both budgets, as did 20-shot Deformation3D, whereas Soybean-PCMAE fragmented it at both budgets and introduced more internal partitions at 10 shots. At the compact apex in region b, all displayed predictions remained under-segmented, although PlantC2USeg separated most leaves at both budgets. Soybean-PCMAE recovered only a small distinct leaf partition, while 20-shot Deformation3D recovered no distinct apical leaf instance. In region c, PlantC2USeg separated all three slender leaves at both budgets. At 20 shots, Deformation3D merged two leaves, while Soybean-PCMAE assigned all three to one instance at both budgets. Despite this local merge, Deformation3D still had slightly higher mWCov for the complete sample than PlantC2USeg under 20-shot supervision.

![](images/45c27787099f506118fb87a99344737ce31534518e592dff84b3af860cc23ecd.jpg)  
Figure 7: Few-shot stem–leaf semantic segmentation of the May 06. May 13, and May 20 Soybean3D samples reused from Figure 6 Columns from left to right show ground truth (GT), 20-shot PlantC2USeg, 20-shot Deformation3D, 20-shot Soybean-PCMAE, 10-shot PlantC2USeg, and 10-shot Soybean-PCMAE. The May 06, May 13, and May 20 rows contain regions a, b, and c, respectively, marking boundary allocation at a narrow junction, retention of small leaves at the compact apex, and leaf-to-stem leakage beside a junction. Red denotes stem and green denotes leaf. Displayed values are sample-level mean IoU over stem and leaf (%).

At the dataset level, pre-training improved the coverage and matched-instance metrics over the Hierarchical Transformer trained from scratch at both 20 and 10 shots (Table 1). The mWCov gains ranged from 3.74 to 8.01 percentage points, while mPrec also increased at both supervision levels. At 20 shots, PlantC2USeg led every method trained with the same annotation count and also surpassed full-shot Plant-Net (Li et al., 2022b), PSegNet (Li et al., 2022a), and Ef-3DPSeg (Luo et al., 2023) across the coverage and matchedinstance measures. Its mRec exceeded every named fullshot alternative by at least 3.13 percentage points. Against Deformation3D (Yang et al., 2024) at the same annotation count, the mRec advantage reached 11.37 percentage points, while mPrec was similar and both coverage measures were higher.

At 10 shots, PlantC2USeg exceeded Soybean-PCMAE (Tian et al., 2025) in both coverage measures and mPrec, with a 7.31 percentage point advantage in mWCov. Soybean-PCMAE retained only a 0.85 percentage point advantage in mRec. For semantic segmentation, Soybean-PCMAE had 1.56 percentage points higher IoU and was also higher in F1 and recall, while PlantC2USeg retained the highest precision. Taken across both tasks, Soybean-PCMAE, which adapts the objectives in stages, retained stronger semantic F1, IoU, and recall together with a narrow mRec advantage, whereas the jointly fine-tuned PlantC2USeg model retained stronger instance coverage and mPrec together with the highest semantic precision.

![](images/26a29ea507aa20b941a74318f911f9fdad5aef45a149addbc1520babb7bc8708.jpg)  
Figure 8: Selected HR3D stem–leaf semantic segmentation results, with tobacco, tomato, and sorghum shown from top to bottom. Columns from left to right show ground truth (GT), the full-shot predictions of PlantC2USeg, PSegNet, and PlantNet, the 20-shot and 10-shot predictions of PlantC2USeg, and the 20-shot Deformation3D prediction. The legend identifies stem and the species-specific leaf classes. Region a marks an emerging tobacco leaf, while region b marks a narrow tomato stem. At the unmarked sorghum base, all methods showed local disagreement with ground truth without a consistent error direction. Displayed values are sample-level mean IoU over stem and the corresponding species leaf class (%).

## 4.2.2. HR3D

Figure 12 compares full-shot and few-shot leaf instance predictions on the same HR3D samples used for semantic evaluation in Figure 8. Overlapping broad leaves exposed repeated under-segmentation by full-shot PSegNet and PlantNet in tobacco region a and tomato region b. Both methods merged each highlighted leaf pair into one instance, whereas full-shot and 20-shot PlantC2USeg separated the two leaves. At the compact sorghum base in region c, fullshot PSegNet and 10-shot PlantC2USeg merged all three slender leaves into one instance, while full-shot PlantNet and 20-shot PlantC2USeg partitioned them into two instances. Full-shot PlantC2USeg separated all three leaves but introduced a spurious partition. At 20 shots, Deformation3D recovered only one of the three leaves as a distinct instance and omitted the distal tip of another leaf.

Among full-shot methods, PlantC2USeg led every aggregate and species-specific instance metric on HR3D (Table 2). Relative to PSegNet (Li et al., 2022a), the strongest plant-specific joint comparator, its margin was 1.80 percentage points in semantic IoU and reached 7.91 percentage points in mWCov and 13.18 percentage points in mRec. The multi-species advantage was therefore more pronounced in leaf coverage and ground-truth leaf recovery than in semantic labeling. In the selected tobacco and tomato examples,

PSegNet merged leaf pairs that PlantC2USeg separated, a sample-level relation consistent with the higher mRec of PlantC2USeg (Figure 12).

The 20-shot PlantC2USeg model ranked second in aggregate mCov, mWCov, and mRec among all evaluated configurations, behind only its full-shot counterpart. Against Deformation3D (Yang et al., 2024) under the same annotation budget, the mRec advantage reached 16.74 percentage points and mWCov was also higher, while the mPrec advantage was only 1.36 percentage points.

Even with 10 shots, PlantC2USeg exceeded full-shot SGPN (Wang et al., 2018) across the coverage and matchedinstance measures and exceeded full-shot ASIS (Wang et al., 2019a) in mCov, mWCov, and mRec. Across unequal supervision settings, it also exceeded 20-shot Deformation3D in these three metrics. Its mWCov reached 79.42%, compared with 77.45% for ASIS and 73.91% for Deformation3D, while mPrec remained lower than for both methods. Across both limited-label settings, PlantC2USeg retained the advantage in leaf-instance coverage and mRec over ASIS, whereas ASIS retained higher semantic IoU, with the 20-shot mRec margin reaching 20.90 percentage points.

## 4.2.3. SYAU-Maize

On the same three SYAU-Maize samples and regions used in Figure 9, leaf instance predictions expose cotyledon assignment near abscission and emerging-leaf separation within the central whorl (Figure 13). PlantNet and PSegNet assigned the region a cotyledon to stem, excluding it from their leaf-instance partitions. Both Deformation3D variants merged the cotyledon with an adjacent small leaf, whereas PlantC2USeg retained it as a distinct leaf instance. Regions b and c contain rolled emerging leaves closely packed within the central whorl, where limited physical separation and mutual occlusion obscure inter-leaf boundaries, particularly under the tighter enclosure in region c. PlantNet and PSeg-Net merged the central emerging leaf with an enclosing leaf in region b and, together with Deformation3D (x10), failed to recover the emerging leaf as a distinct, coherent instance under the tighter enclosure in region c. PlantC2USeg assigned most of the tightly enclosed emerging leaf to one coherent instance, whereas Deformation3D (x1k) left the corresponding assignments fragmented. On the complete sample containing region c, PlantC2USeg reached 97.2% mWCov, compared with 76.9% for PSegNet.

Table 2  
Quantitative results of stem–leaf and leaf instance segmentation on HR3D (%). Sup. denotes supervision. Full uses the complete training set, and numerical entries give the number of annotated training samples. A checkmark denotes joint optimization of stem–leaf semantic and leaf instance segmentation within a single downstream model.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Sup.</td><td rowspan="2">Joint</td><td colspan="4">Semantic Segmentation</td><td colspan="4">Instance Segmentation</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>IoU</td><td>mCov</td><td>mWCov</td><td>mPrec</td><td>mRec</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGPN</td><td>Full</td><td>√</td><td>82.08</td><td>76.48</td><td>76.61</td><td>67.95</td><td>41.68</td><td>48.78</td><td>40.66</td><td>30.14</td></tr><tr><td>ASIS</td><td>Full</td><td>√</td><td>91.31</td><td>88.57</td><td>89.18</td><td>83.00</td><td>70.39</td><td>77.45</td><td>78.66</td><td>62.96</td></tr><tr><td>PlantNet</td><td>Full</td><td>√</td><td>91.86</td><td>91.45</td><td>90.98</td><td>85.10</td><td>76.17</td><td>81.68</td><td>83.56</td><td>72.61</td></tr><tr><td>PSegNet</td><td>Full</td><td>√</td><td>93.37</td><td>92.34</td><td>92.27</td><td>87.27</td><td>79.39</td><td>83.67</td><td>86.58</td><td>73.70</td></tr><tr><td>PlantC2USeg</td><td>Full</td><td>√</td><td>94.13</td><td>93.19</td><td>93.37</td><td>89.07</td><td>87.78</td><td>91.58</td><td>89.00</td><td>86.88</td></tr><tr><td>Deformation3D</td><td>20</td><td></td><td>80.79</td><td>73.30</td><td>73.83</td><td>66.12</td><td>66.67</td><td>73.91</td><td>78.92</td><td>67.12</td></tr><tr><td>PlantC2USeg</td><td>20</td><td>√</td><td>88.24</td><td>88.77</td><td>87.43</td><td>81.50</td><td>81.20</td><td>86.21</td><td>80.28</td><td>83.86</td></tr><tr><td>PlantC2USeg</td><td>10</td><td>√</td><td>85.45</td><td>86.97</td><td>85.14</td><td>78.41</td><td>72.73</td><td>79.42</td><td>74.06</td><td>70.16</td></tr><tr><td>Tobacco</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGPN</td><td>Full</td><td>√</td><td>79.51</td><td>73.29</td><td>74.05</td><td>62.20</td><td>28.65</td><td>37.66</td><td>25.41</td><td>22.70</td></tr><tr><td>ASIS</td><td>Full</td><td>√</td><td>89.88</td><td>88.42</td><td>88.66</td><td>82.31</td><td>64.08</td><td>74.65</td><td>77.59</td><td>60.04</td></tr><tr><td>PlantNet</td><td>Full</td><td>√</td><td>90.76</td><td>89.01</td><td>89.08</td><td>82.35</td><td>70.84</td><td>79.85</td><td>84.56</td><td>68.37</td></tr><tr><td>PSegNet</td><td>Full</td><td>√</td><td>93.45</td><td>90.78</td><td>91.39</td><td>86.23</td><td>76.82</td><td>84.00</td><td>89.11</td><td>72.80</td></tr><tr><td>PlantC2USeg</td><td>Full</td><td>√</td><td>94.34</td><td>94.54</td><td>94.18</td><td>90.14</td><td>86.12</td><td>92.19</td><td>89.33</td><td>83.44</td></tr><tr><td>Deformation3D</td><td>20</td><td></td><td>79.24</td><td>72.90</td><td>73.60</td><td>64.73</td><td>65.28</td><td>75.77</td><td>83.70</td><td>65.92</td></tr><tr><td>PlantC2USeg</td><td>20</td><td>√</td><td>91.30</td><td>90.18</td><td>89.98</td><td>84.53</td><td>83.98</td><td>89.88</td><td>83.05</td><td>85.99</td></tr><tr><td>PlantC2USeg</td><td>10</td><td>√</td><td>89.84</td><td>90.16</td><td>89.64</td><td>82.98</td><td>76.44</td><td>85.29</td><td>75.45</td><td>74.01</td></tr><tr><td>Tomato</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGPN</td><td>Full</td><td>√</td><td>86.54</td><td>83.36</td><td>83.55</td><td>77.10</td><td>49.57</td><td>58.77</td><td>76.05</td><td>30.94</td></tr><tr><td>ASIS</td><td>Full</td><td>√</td><td>94.25</td><td>93.16</td><td>93.45</td><td>88.42</td><td>76.27</td><td>81.77</td><td>84.48</td><td>64.95</td></tr><tr><td>PlantNet</td><td>Full</td><td>√</td><td>94.91</td><td>94.04</td><td>94.32</td><td>89.87</td><td>80.88</td><td>84.97</td><td>85.75</td><td>73.68</td></tr><tr><td>PSegNet</td><td>Full</td><td>√</td><td>95.95</td><td>94.90</td><td>95.30</td><td>91.45</td><td>81.68</td><td>85.10</td><td>88.30</td><td>72.40</td></tr><tr><td>PlantC2USeg</td><td>Full</td><td>√</td><td>95.73</td><td>95.65</td><td>95.58</td><td>91.95</td><td>87.65</td><td>90.69</td><td>91.34</td><td>86.15</td></tr><tr><td>Deformation3D</td><td>20</td><td></td><td>90.63</td><td>86.24</td><td>86.63</td><td>79.42</td><td>70.86</td><td>78.17</td><td>86.87</td><td>68.03</td></tr><tr><td>PlantC2USeg</td><td>20</td><td>√</td><td>93.09</td><td>92.53</td><td>92.58</td><td>87.37</td><td>80.58</td><td>84.69</td><td>78.47</td><td>79.73</td></tr><tr><td>PlantC2USeg</td><td>10</td><td>√</td><td>92.22</td><td>91.55</td><td>91.60</td><td>85.53</td><td>72.95</td><td>78.65</td><td>69.59</td><td>61.88</td></tr><tr><td>Sorghum</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGPN</td><td>Full</td><td>√</td><td>80.17</td><td>72.79</td><td>72.23</td><td>64.55</td><td>46.83</td><td>49.93</td><td>20.51</td><td>36.78</td></tr><tr><td>ASIS</td><td>Full</td><td>√</td><td>89.80</td><td>84.13</td><td>85.44</td><td>78.27</td><td>70.84</td><td>75.95</td><td>73.90</td><td>63.91</td></tr><tr><td>PlantNet</td><td>Full</td><td>√</td><td>89.91</td><td>91.30</td><td>89.54</td><td>83.07</td><td>76.79</td><td>80.21</td><td>80.37</td><td>75.79</td></tr><tr><td>PSegNet</td><td>Full</td><td>√</td><td>90.70</td><td>91.34</td><td>90.10</td><td>84.13</td><td>79.67</td><td>81.90</td><td>82.33</td><td>75.89</td></tr><tr><td>PlantC2USeg</td><td>Full</td><td>√</td><td>92.32</td><td>89.38</td><td>90.34</td><td>85.12</td><td>89.58</td><td>91.86</td><td>86.34</td><td>91.04</td></tr><tr><td>Deformation3D</td><td>20</td><td></td><td>72.50</td><td>60.76</td><td>61.27</td><td>54.19</td><td>63.87</td><td>67.80</td><td>66.18</td><td>67.43</td></tr><tr><td>PlantC2USeg</td><td>20</td><td>√</td><td>80.32</td><td>83.59</td><td>79.74</td><td>72.60</td><td>79.05</td><td>84.07</td><td>79.32</td><td>85.84</td></tr><tr><td>PlantC2USeg</td><td>10</td><td>√</td><td>74.31</td><td>79.21</td><td>74.19</td><td>66.72</td><td>68.81</td><td>74.32</td><td>77.12</td><td>74.60</td></tr></table>

Using the same 22 annotated samples, PlantC2USeg exceeded the directly trained joint baselines PlantNet (Li et al., 2022b) and PSegNet (Li et al., 2022a) across all four instance metrics on the SYAU-Maize test set (Table 3). Its margins over both baselines were at least 19.50 percentage points in both coverage measures and mRec. The semantic and instance leads support the greater efectiveness of the complete PlantC2USeg strategy over both direct joint baselines in the evaluated sparse-label setting.

Within Deformation3D (Yang et al., 2024), expansion from x10 to x1k raised mCov and mRec, while mWCov rose by 3.38 percentage points as mPrec fell by 3.63 percentage points. PlantC2USeg was higher than x1k in mCov, mPrec,

![](images/6091456a3b2359d4425097e21b5419d09122889171788bc9c4e6e33f58ca35f4.jpg)  
Figure 9: Stem–leaf semantic segmentation on three SYAU-Maize samples under a 22-sample annotation budget, with displayed values denoting sample-level IoU (%). Columns from left to right show ground truth (GT), PlantC2USeg, Deformation3D (x10), PSegNet, PlantNet, and Deformation3D (x1k). The x10 and x1k settings expand the same 22-sample training set to 220 and 22,000 deformed point clouds, respectively. Region a marks a cotyledon near abscission, and regions b and c mark collar boundaries adjoining the leaf sheath.

## Table 3

Dataset-level stem–leaf semantic and leaf instance segmentation results on the 406-sample SYAU-Maize test set (%). All methods use the same 22 annotated source samples. For Deformation3D, x10 and x1k denote tenfold and 1000-fold deformation expansion of this training set. A checkmark denotes joint optimization of both tasks within a single downstream model.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Joint</td><td colspan="4">Semantic Segmentation</td><td colspan="4">Instance Segmentation</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>IoU</td><td>mCov</td><td>mWCov</td><td>mPrec</td><td>mRec</td></tr><tr><td>PlantNet</td><td>√</td><td>91.21</td><td>90.99</td><td>90.10</td><td>83.81</td><td>67.16</td><td>72.73</td><td>84.17</td><td>71.70</td></tr><tr><td>PSegNet</td><td>√</td><td>90.70</td><td>91.00</td><td>89.78</td><td>83.30</td><td>66.09</td><td>71.78</td><td>83.03</td><td>70.40</td></tr><tr><td>Deformation3D (x10)</td><td></td><td>94.61</td><td>96.20</td><td>95.07</td><td>91.19</td><td>84.93</td><td>91.52</td><td>91.33</td><td>85.34</td></tr><tr><td>Deformation3D (x1k)</td><td></td><td>97.27</td><td>92.90</td><td>94.72</td><td>90.59</td><td>88.23</td><td>94.90</td><td>87.70</td><td>88.25</td></tr><tr><td>PlantC2USeg</td><td>√</td><td>97.03</td><td>95.40</td><td>96.04</td><td>92.75</td><td>89.11</td><td>92.23</td><td>91.46</td><td>93.51</td></tr></table>

and mRec, including a 5.26 percentage point mRec margin, whereas x1k retained a 2.67 percentage point lead in mW-Cov. Because mCov weights instances equally and mWCov weights them by point count, the aggregate reversal suggests a relative overlap advantage for PlantC2USeg among leaves with fewer points and for x1k among leaves with more points. Greater deformation expansion therefore favored instance coverage weighted by point count but yielded no uniform cross-task gain, as semantic IoU declined from x10 to x1k.

## 4.3. Object part segmentation

Qualitative results in Figure 14 show that accurate part segmentation is maintained across a wide range of generic object categories. On ShapeNet Part, PlantC2USeg reached 85.0% category-averaged mIoU and 86.4% instanceaveraged mIoU, exceeding baselines trained from scratch (Qi et al., 2017a,b; Wang et al., 2019b; Vaswani et al., 2017) and extending the evaluation beyond plant-specific segmentation. Among single-modal pre-training methods (Sauder and Sievers, 2019; Wang et al., 2021; Pang et al., 2022; Zhang et al., 2022; Chen et al., 2023; Zha et al., 2024; Zhang et al., 2024, 2025), it ranked first in category-averaged mIoU and second in instance-averaged mIoU, matching the

![](images/8628280c0eff0288ebacda7a8606722f769e5783da00359f9dd24fc27e6a1392.jpg)  
Figure 10: Full-shot leaf instance segmentation on three Soybean3D examples. Regions a–c mark a broad leaf, three leaves at a compact apex, and three slender apical leaves, respectively. Values denote sample-level mWCov (%).

## Table 4

Quantitative results of part segmentation on ShapeNet Part (%). Cat. mIoU and Inst. mIoU denote category-averaged and instanceaveraged mIoU, respectively.

<table><tr><td>Method</td><td>Cat. mIoU</td><td>Inst. mIoU</td></tr><tr><td>Training from scratch</td><td></td><td>83.7</td></tr><tr><td>PointNet</td><td>80.4</td><td>85.1</td></tr><tr><td>PointNet++ DGCNN</td><td>81.9</td><td>85.2</td></tr><tr><td>Transformer</td><td>82.3 83.4</td><td>85.1</td></tr><tr><td>Single-modal pre-training</td><td></td><td></td></tr><tr><td>Jigsaw3D</td><td>83.0</td><td>85.3</td></tr><tr><td>OcCo</td><td>83.4</td><td>85.1</td></tr><tr><td>Point-MAE</td><td>84.2</td><td>86.1</td></tr><tr><td>Point-M2AE</td><td>84.8</td><td>86.5</td></tr><tr><td>PointGPT</td><td>84.1</td><td>86.2</td></tr><tr><td>Point-FEMAE</td><td>84.9</td><td>86.3</td></tr><tr><td>PCP-MAE</td><td>84.9</td><td>86.1</td></tr><tr><td>Point-PQAE</td><td>84.6</td><td>86.1</td></tr><tr><td></td><td></td><td>86.4</td></tr><tr><td>PlantC2USeg</td><td>85.0</td><td></td></tr><tr><td>Cross-modal pre-training</td><td></td><td></td></tr><tr><td>ACT</td><td>84.7</td><td>86.1</td></tr><tr><td>ReCon</td><td>84.8</td><td>86.4</td></tr></table>

best cross-modal results (Dong et al., 2023; Qi et al., 2023) for the latter metric.

## 5. Discussion

## 5.1. Ablation study

The full-shot Soybean3D ablation in Table 5 examines cross-scale coherence and reconstruction conditioned on visible context as complementary constraints on plant representation learning for stem–leaf semantic segmentation and leaf instance segmentation. The Hierarchical Transformer trained from scratch provides an aligned control with the same architecture and downstream objectives but no pretraining. With standard self-attention and without crossscale consistency learning, pre-training increased semantic recall and all four instance metrics over scratch initialization but slightly reduced F1 and IoU, exposing uneven transfer between leaf instance segmentation and aggregate stem–leaf discrimination.

By relating corresponding visible regions across spatial scales, cross-scale consistency learning raised IoU by 1.25 percentage points with the standard decoder and by 0.90 percentage points with the information-restricted decoder, with F1 improving in both cases. The consistent gains support coupling fine-scale organ cues with broader plant architecture during encoder representation learning. Informationrestricted decoding likewise raised F1 and IoU whether cross-scale consistency learning was present or absent. The information-restricted decoder conditions reconstruction on visible evidence, complementing the cross-scale encoder constraint imposed by cross-scale consistency learning. The combined design reached 95.69% F1 and 91.91% IoU while leading all four instance metrics. The joint lead across both tasks supports coordinating cross-scale encoder coherence with reconstruction grounded in visible evidence for the representation shared by organ classification and leaf separation.

Table 5  
![](images/068d486b63c4062e9e13d0585db972758224f1a22cacf0a49e841ecfe24bbd20.jpg)  
Figure 11: Few-shot leaf instance segmentation on the same Soybean3D examples and regions as Figure 10. After ground truth (GT), prediction columns show 20-shot PlantC2USeg, Deformation3D, and Soybean-PCMAE, followed by 10-shot PlantC2USeg and Soybean-PCMAE. Regions a–c mark the broad leaf, compact apex, and three slender apical leaves used in the full-shot comparison, respectively. Values denote sample-level mWCov (%).

Ablation analysis of each component for stem–leaf and leaf instance segmentation (%). C2L denotes cross-scale consistency learning, IR-Dec denotes information-restricted decoding.
<table><tr><td rowspan="2">Pre-train</td><td rowspan="2">C2L</td><td rowspan="2">IR-Dec</td><td colspan="4">Semantic Segmentation</td><td colspan="4">Instance Segmentation</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>IoU</td><td>mCov</td><td>mWCov</td><td>mPrec</td><td>mRec</td></tr><tr><td></td><td></td><td></td><td>96.64</td><td>92.98</td><td>94.70</td><td>90.19</td><td>85.54</td><td>92.11</td><td>94.56</td><td>88.63</td></tr><tr><td>√</td><td></td><td></td><td>94.82</td><td>94.08</td><td>94.45</td><td>89.75</td><td>86.73</td><td>92.62</td><td>94.69</td><td>90.98</td></tr><tr><td>√</td><td>√</td><td></td><td>95.71</td><td>94.66</td><td>95.17</td><td>91.00</td><td>88.06</td><td>93.81</td><td>95.93</td><td>92.55</td></tr><tr><td>√</td><td></td><td>√</td><td>94.96</td><td>95.40</td><td>95.18</td><td>91.01</td><td>86.89</td><td>93.06</td><td>95.16</td><td>92.55</td></tr><tr><td>√</td><td>√</td><td>√</td><td>96.40</td><td>95.01</td><td>95.69</td><td>91.91</td><td>89.94</td><td>94.62</td><td>96.39</td><td>94.12</td></tr></table>

## 5.2. Attention distribution across scales during encoding

Encoder attention around selected stem and leaf neigh borhoods shows how spatial support develops through hierarchical stages � = 1, 2, 3 in one representative point cloud (Figure 15). At the early stages, support remains local to the selected neighborhoods. Attention continues to emphasize the selected stem and leaf neighborhoods across all three stages while reaching a broader portion of the connected plant at � = 3. Key counts difer across stages and attention is normalized over keys, so the larger displayed weights at � = 3 do not by themselves establish stronger or more concentrated attention. This progression is consistent with the aim of cross-scale consistency learning to relate fine geometric cues to broader plant morphology.

![](images/d3d4a3c892d951877a82caddcd8466a81fa0fffe052ae60472ed4ff4b0078c5e.jpg)  
Figure 12: Leaf instance segmentation on selected HR3D samples, with tobacco, tomato, and sorghum shown from top to bottom. Columns from left to right show ground truth (GT), the full-shot predictions of PlantC2USeg, PSegNet, and PlantNet, the 20-shot and 10-shot predictions of PlantC2USeg, and the 20-shot Deformation3D prediction. Regions a and b mark overlapping broad-leaf pairs, while region c marks three slender leaves at a compact base. Values denote sample-level mWCov (%).

## 5.3. Diagnostic comparison of decoder attention allocation

Unrestricted decoder self-attention leaves masked tokens able to exchange information during reconstruction. In the standard diagnostic in Figure 16, masked queries allocate attention masses of $\mu ^ { m } = 0 . 8 7$ to masked keys and $\mu ^ { v } = 0 . 5 0$ to visible keys. This allocation is consistent with maskedtoken exchange providing an internal reconstruction route. This route competes with the intended use of evidence encoded from visible neighborhoods during pre-training but belongs to a decoder that is not transferred downstream. Restricting masked-to-masked exchange in one diagnostic layer reverses the allocation to $\mu ^ { v } = 1 0 . 7 7$ for visible keys and $\mu ^ { m } ~ = ~ 0 . 4 8$ for masked keys. The dominant visiblekey mass shows that visible encoder representations contain usable reconstruction evidence. This allocation reversal motivates the final information-restricted decoder, in which masked queries use projected visible representations as keys and values.

## 5.4. Sensitivity of inherited feature-distance criteria

Progressive segmentation inherits three feature-distance criteria from fine-tuning statistics, linking the learned embedding space directly to leaf grouping without an independent threshold search after training. Each criterion was varied separately for Soybean3D leaf instance segmentation under full-shot, 20-shot, and 10-shot supervision while the checkpoint and the other two criteria remained fixed within each regime (Figures 17–19). The partition radius $\epsilon _ { \mathrm { d b } }$ governs within-step feature-space partitioning, $\epsilon _ { \mathrm { r e c } }$ governs cross-step inheritance of an existing instance label, and $\epsilon _ { \mathrm { m e r g e } }$ governs final consolidation after residual assignment.

The partition radius $\epsilon _ { \mathrm { d b } }$ had the strongest efect on coverage and mRec, producing the largest mCov, mWCov, and mRec spans among the three criteria in every supervision regime. Both extremes reduced these outcomes under fewshot supervision (Figure 17). At 20 shots, mCov spanned 24.77 percentage points while the inherited value remained 0.41 percentage points below the observed maximum.

The inheritance and consolidation criteria shared a tradeof between mPrec and coverage or mRec, but their response patterns difered. As $\epsilon _ { \mathrm { r e c } }$ increased under 20-shot and 10- shot supervision, mPrec generally rose, whereas coverage and mRec peaked at lower thresholds and declined at the upper end (Figure 18). Coverage changed little across �<sub>merge</sub> under full-shot supervision. At 10 shots, the $\epsilon _ { \mathrm { m e r g e } }$ value that maximized mPrec raised it by 3.63 percentage points relative to the inherited value but reduced coverage and mRec (Figure 19).

Across all three criteria and supervision regimes, the inherited values kept mCov and mWCov within 1.04 and 0.51 percentage points of their respective observed maxima while avoiding isolated mPrec maxima associated with lower coverage or mRec.

![](images/acecfc2dfc06e6f5ffeb43ae7af8e64a6c78ec8e15e19bbce042f84478d821d4.jpg)  
Figure 13: Leaf instance segmentation on the same three SYAU-Maize samples and regions as Figure 9 under the common 22-sample annotation budget, with displayed values denoting sample-level mWCov (%). Columns from left to right show ground truth (GT), PlantC2USeg, Deformation3D (x10), PSegNet, PlantNet, and Deformation3D (x1k). The x10 and x1k settings denote tenfold and 1000- fold deformation expansion of the same training set. Region a marks a cotyledon near abscission, and regions b and c mark closely packed emerging leaves within the central whorl.

## 5.5. Sensitivity of the initial-radius schedule

Before feature-space partitioning, the initial-radius ratio schedules radial eligibility in shifted planar coordinates and remains separate from the inherited feature-distance criteria (Figure 20). Larger ratios begin with more peripheral candidates, whereas a ratio of zero processes all candidates in one pass.

The selected radial schedules favored broad leaf coverage and matched-instance recovery over marginal gains in precision. Under full-shot supervision, ratios from 0.5 to 0.8 kept all four instance metrics within 0.41 percentage points of their respective observed maxima and included the selected ratio of 0.7. At 20 shots, the selected ratio of 0.6 maximized mWCov and mPrec while mCov and mRec remained near their observed maxima. At 10 shots, increasing the ratio from the selected value of 0.5 to 0.8 raised mPrec by less than 0.70 percentage points and left mRec unchanged. The same change lowered coverage by at least 1.20 percentage points.

## 5.6. Annotation eficiency and practical adaptation

On Soybean3D, reusable pre-training reduces dependence on target annotations relative to the aligned scratch control across the 20-shot and 10-shot budgets. At 20 shots,

PlantC2USeg pre-training raised semantic IoU by 2.63 percentage points and mWCov, which weights leaf-instance coverage by size, by 8.01 percentage points over random initialization. At 10 shots, it also maintained higher IoU and mWCov than the same scratch control, extending the aligned benefit across both limited-label budgets. Under the 22- template SYAU-Maize protocol, PlantNet (Li et al., 2022b) and PSegNet (Li et al., 2022a) jointly learn semantic and instance segmentation from the annotated samples, whereas PlantC2USeg applies reusable pre-training before unified adaptation in one shared model. PlantC2USeg exceeded both fully supervised baselines by about 9 percentage points in IoU and about 20 percentage points in mWCov, supporting reusable representation transfer under this annotation budget.

The comparison with target-specific deformation spans Soybean3D, HR3D, and SYAU-Maize, which together cover five crop species and use MVS reconstruction, blue-laser scanning, and high-precision 3D scanning, respectively. Deformation3D (Yang et al., 2024) generates target-specific training clouds and trains separate PointNet++ semantic and HAIS instance models, whereas PlantC2USeg reuses a pre-trained representation before target fine-tuning. Under matched original annotation budgets, PlantC2USeg exceeded Deformation3D x10 by 1.56–15.38 percentage points (b) Attention weights to a leaf neighborhood

![](images/406089cae727e7c8e997676e047626cb4fe521f6f8559dda41a453153dafa49a.jpg)  
Figure 14: Qualitative part segmentation results on the ShapeNet Part dataset, with displayed values denoting category IoU (%).

![](images/0cd8ca81444cea169378253fa6af7f7c86b4da3f65d26da6c8d51db4a3c20abc.jpg)  
Figure 15: Encoder self-attention distributions across hierarchical stages (� = 1, 2, 3) for (a) a selected stem target neighborhood and (b) a selected leaf target neighborhood in one representative point cloud. Red stars mark the attention targets, and black points mark the centers of masked neighborhoods. Colors use a common displayed scale for the attention weights.

in semantic IoU and attained higher mRec on all three datasets. Across the three datasets, these consistent IoU and mRec gains support representation reuse over repeated target-specific data generation within the evaluated plant distributions. Target-specific generation also repeats data preparation for each template, with x1k preparation projected at 2.31 h for one 590,574-point Soybean3D template.

Moving Deformation3D from x10 to x1k increased the number of generated clouds 100-fold, raised mWCov by 3.38 percentage points, and lowered IoU by 0.60 percentage points. Among the five methods evaluated under this protocol, Deformation3D x1k attained the highest mWCov, while PlantC2USeg remained higher than x1k in semantic IoU and the other three instance metrics. In this comparison, larger expansion improves size-weighted instance coverage at the expense of semantic overlap.

PlantC2USeg fine-tunes one shared model to produce semantic predictions, feature embeddings, and coordinate ofsets, while deriving the three criteria used in progressive segmentation directly from fine-tuning statistics. This unified adaptation contrasts with the separate downstream networks of Ef-3DPSeg (Luo et al., 2023), the sequential adaptation of Soybean-PCMAE (Tian et al., 2025), and the separate semantic and instance models of Deformation3D (Yang et al., 2024).

![](images/dafe52fd2c187bcb21f4a0e4738699e2a5b71035a2d720dc732edf8f4ec928c6.jpg)  
(a) Self-attention decoders

![](images/eda472f4bc3f0a0feb338c7850bcdf2ca008c38c88af81d9664316f3adcef6f8.jpg)  
Figure 16: Diagnostic comparison of attention allocation for (a) a standard self-attention decoder and (b) a restricted decoder that limits masked-to-masked information exchange in one decoder layer. The reported $\mu ^ { v }$ and $\mu ^ { m }$ are attention masses from masked queries to visible and masked keys, respectively. Black points mark the centers of visible neighborhoods, and red stars mark the displayed attention targets. All four maps use a common displayed attention-weight scale.

![](images/a3587c1112da648e54fc7f8036370103db2e13c71320636c3ff51b5c68948b3b.jpg)

![](images/7d0a721dff6bbb7665833d1fa3df00e250eabe3a67bcd90bbf8e00a5c6b4c943.jpg)  
-0- fullshot -20shot 10shot --- Inherited  
Figure 17: Sensitivity of Soybean3D leaf instance segmentation to the feature-space $\ell _ { 1 }$ partition radius $\epsilon _ { \mathrm { d b } }$ under full-shot, 20-shot, and 10-shot supervision. Within each supervision regime, only $\epsilon _ { \mathrm { d b } }$ varies while $\epsilon _ { \mathrm { r e c } } , \epsilon _ { \mathrm { m e r g e } } ,$ and the checkpoint remain fixed. Within each regime, enlarged filled markers denote the partition radius inherited from fine-tuning statistics, while the remaining markers show the evaluated sweep values.

Table 6 reports network-core forward and backward computation for one Soybean3D sample after the native input processing of each method. PlantC2USeg recorded the shortest time among the five evaluated methods, 38.60 ms compared with 69.60 ms for the next-fastest HAIS. The timed PlantC2USeg model covers semantic and instance segmentation in one shared network, whereas the HAIS measurement covers only the Deformation3D instance network. The measured speed advantage therefore supports the practical eficiency of unified adaptation within the reported network-core scope.

## 6. Conclusion

Reusable pre-training in PlantC2USeg reduces dependence on target annotations while supporting stem–leaf semantic segmentation and leaf instance segmentation through a shared adaptation model. The full-shot ablation shows that cross-scale consistency learning and reconstruction conditioned on visible evidence impose complementary constraints on the representation adapted jointly for both tasks. Against the aligned Hierarchical Transformer trained from scratch on full-shot Soybean3D, complete pre-training increased IoU by 1.72 percentage points and mWCov by 2.51 percentage points. Complete pre-training retained higher IoU and mWCov than random initialization as Soybean3D supervision decreased. Across Soybean3D, HR3D, and

-0- fullshot 20shot 10shot --- Inherited  
![](images/8f3f7c44508c9e5378cc975f9f25d3ac50da8aed040b447e3bc73b1a11f35a9b.jpg)

![](images/2e265c8b97a36836ec31137fc29b045b4a413401aa61b71d36157b1c829eb413.jpg)

Figure 18: Sensitivity of Soybean3D leaf instance segmentation to the feature-centroid $\ell _ { 1 }$ threshold $\epsilon _ { \mathrm { r e c } }$ under full-shot, 20-shot, and 10-shot supervision. This threshold governs whether a later feature partition inherits the nearest existing instance label. Within each supervision regime, only $\epsilon _ { \mathrm { r e c } }$ varies while $\epsilon _ { \mathrm { d b } } , \epsilon _ { \mathrm { m e r g e } } ,$ , and the checkpoint remain fixed. Within each regime, enlarged filled markers denote the reconnection threshold inherited from fine-tuning statistics, while the remaining markers show the evaluated sweep values.  
![](images/a8d18448dfbbd1e66a3a29c874343c7118b38c01baeb493f969b5d3345961437.jpg)

![](images/ffccf0e09402c28589afe11b92ba1a002178d2c671c736f3ab56151321620014.jpg)  
Figure 19: Sensitivity of Soybean3D leaf instance segmentation to the feature-centroid $\ell _ { 1 }$ threshold $\epsilon _ { \mathrm { m e r g e } }$ used for final consolidation after residual assignment under full-shot, 20-shot, and 10-shot supervision. Within each supervision regime, only $\epsilon _ { \mathrm { m e r g e } }$ varies while $\epsilon _ { \mathrm { d b } } ,$ $\epsilon _ { \mathrm { r e c } } ,$ and the checkpoint remain fixed. Within each regime, enlarged filled markers denote the merge threshold inherited from fine-tuning statistics, while the remaining markers show the evaluated sweep values.

SYAU-Maize, higher IoU and mRec than Deformation3D x10 support reusing the pre-trained representation under diferent target distributions.

The limited-label regimes across the three plant datasets show that PlantC2USeg can produce the stem–leaf labels and individual-leaf partitions required for later organ-level trait extraction from small annotated collections. Extending PlantC2USeg beyond static point clouds of individual plants requires evaluation in complex multi-plant field scenes. Inter-plant overlap and occlusion, together with background vegetation and variations in reconstruction completeness and point density, require segmentation to distinguish neighboring plants and associate each stem and leaf with the correct plant. Temporal phenotyping then requires correspondence between the same organs across repeated 3D observations. Stable organ correspondence would enable longitudinal measurement of organ-specific growth trajectories across developmental stages.

![](images/780e99704426eda6cc6298f214b512a2523da3521a709cafb38106548722433d.jpg)

![](images/7dbc59b3e687f5950ffd390d2ec5200e61e2e214e1a4dbf2a38bbb86d3585a28.jpg)  
Figure 20: Sensitivity of Soybean3D leaf instance segmentation to the initial-radius ratio under full-shot, 20-shot, and 10-shot supervision. The ratio schedules radial eligibility in shifted planar coordinates before feature-space partitioning. Only the initial-radius ratio varies from 0.0 to 0.8 while $\epsilon _ { \mathrm { d b } } , \epsilon _ { \mathrm { r e c } } , \epsilon _ { \mathrm { m e r g e } }$ , and the checkpoint remain fixed within each supervision regime. Within each regime, enlarged filled markers denote the selected initial-radius ratio, while the remaining markers show the evaluated sweep values.

Controlled training eficiency for one Soybean3D sample. For parameters, FLOPs, both timing measures, and peak memory, lower values are preferable. All methods were profiled on an NVIDIA GeForce RTX 3090 at batch size 1 after five warmup iterations and over 20 timed iterations. All methods use the same Soybean3D sample, which contains 590,574 points before method-specific input processing. PlantNet, PSegNet, and PlantC2USeg process 4,096 sampled points. Ef-3DPSeg voxelizes the complete sample, whereas Deformation3D sample 20,480 points before voxelization. Forward and backward time excludes data loading, augmentation, voxelization, optimizer updates, logging, validation, and checkpointing. Methods are ordered by the combined forward and backward time.
<table><tr><td>Method</td><td>Input points</td><td>Active voxels</td><td>Parameters (M)</td><td>FLOPs (G)</td><td>Forward time (ms)</td><td>Forward + backward (ms)</td><td>Peak memory (MiB)</td></tr><tr><td>PlantC2USeg</td><td>4,096</td><td></td><td>13.61</td><td>9.01</td><td>18.31</td><td>38.60</td><td>629.6</td></tr><tr><td>Deformation3D (HAIS)</td><td>20,480</td><td>1,861</td><td>30.84</td><td>0.93</td><td>33.58</td><td>69.60</td><td>244.9</td></tr><tr><td>PlantNet</td><td>4,096</td><td></td><td>4.44</td><td>12.15</td><td>149.79</td><td>213.83</td><td>755.2</td></tr><tr><td>PSegNet</td><td>4,096</td><td></td><td>3.34</td><td>10.80</td><td>157.01</td><td>225.15</td><td>788.2</td></tr><tr><td>Eff-3DPSeg</td><td>590,574</td><td>231,971</td><td>37.86</td><td>490.50</td><td>107.47</td><td>503.60</td><td>2,962.4</td></tr></table>

## CRediT authorship contribution statement

Yu Tian: Data curation, Investigation, Methodology, Visualization, Formal analysis, Writing - original draft, Writing - review & editing. Xintong Jiang: Data curation, Investigation, Writing - review & editing. Jan Franklin Adamowski: Writing - review & editing. Shiv O. Prasher: Writing - review & editing. Shangpeng Sun: Conceptualization, Resources, Writing - review & editing, Supervision, Project administration, Funding acquisition.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This research is supported by funding from the FRQNT & MAPAQ Partnership Research Program-Sustainable Agriculture (Grant No. 259806), and the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grants Program (Grant no. G256643).

## Data availability

Data will be made available on request.

## References

Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., Xiao, J., Yi, L., Yu, F., 2015. ShapeNet: An Information-Rich 3D Model Repository. Technical Report arXiv:1512.03012 [cs.GR]. Stanford University — Princeton University — Toyota Technological Institute at Chicago.

Chen, G., Wang, M., Yang, Y., Yu, K., Yuan, L., Yue, Y., 2023. Pointgpt: auto-regressively generative pre-training from point clouds, in: Proceedings of the 37th International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA. pp. 29667–29679.

Chen, S., Fang, J., Zhang, Q., Liu, W., Wang, X., 2021. Hierarchical aggregation for 3d instance segmentation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15467–15476.

Comaniciu, D., Meer, P., 2002. Mean shift: a robust approach toward feature space analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence 24, 603–619. doi:10.1109/34.1000236.

Conn, A., Pedmale, U.V., Chory, J., Navlakha, S., 2017. High-resolution laser scanning reveals plant architectures that reflect universal network design principles. Cell systems 5, 53–62.

Cover, T., Hart, P., 1967. Nearest neighbor pattern classification. IEEE Transactions on Information Theory 13, 21–27. doi:10.1109/TIT.1967. 1053964.

De Brabandere, B., Neven, D., Van Gool, L., 2017. Semantic instance segmentation with a discriminative loss function. arXiv preprint arXiv:1708.02551 .

Deng, J., Guo, J., Xue, N., Zafeiriou, S., 2019. Arcface: Additive angular margin loss for deep face recognition, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4685–4694. doi:10.1109/CVPR.2019.00482.

van Dijk, M., Morley, T., Rau, M.L., Saghai, Y., 2021. A meta-analysis of projected global food demand and population at risk of hunger for the period 2010–2050. Nature Food 2, 494–501. URL: https://doi.org/10. 1038/s43016-021-00322-9, doi:10.1038/s43016-021-00322-9.

Dong, R., Qi, Z., Zhang, L., Zhang, J., Sun, J., Ge, Z., Yi, L., Ma, K., 2023. Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning?, in: The Eleventh International Conference on Learning Representations. URL: https: //openreview.net/forum?id=8Oun8ZUVe8N.

Du, R., Ma, Z., Xie, P., He, Y., Cen, H., 2023. Pst: Plant segmentation transformer for 3d point clouds of rapeseed plants at the podding stage. ISPRS Journal of Photogrammetry and Remote Sensing 195, 380–392.

Du, R., Zhai, G., Qiu, T., Jiang, Y., 2025. Towards scalable organ level 3d plant segmentation: Bridging the data algorithm computing gap. arXiv preprint arXiv:2509.06329 .

Ester, M., Kriegel, H.P., Sander, J., Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise, in: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, AAAI Press. pp. 226–231.

Fan, H., Su, H., Guibas, L.J., 2017. A point set generation network for 3d object reconstruction from a single image, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 605–613.

Fu, L., Lian, L., Wang, R., Shi, B., Wang, X., Yala, A., Darrell, T., Efros, A.A., Goldberg, K., 2025. Rethinking patch dependence for masked autoencoders. Transactions on Machine Learning Research .

George, D., Lehrach, W., Kansky, K., Lázaro-Gredilla, M., Laan, C., Marthi, B., Lou, X., Meng, Z., Liu, Y., Wang, H., et al., 2017. A generative vision model that trains with high data eficiency and breaks text-based captchas. Science 358, eaag2612.

Harandi, N., Vandenberghe, B., Vankerschaver, J., Depuydt, S., Van Messem, A., 2023. How to make sense of 3d representations for plant phenotyping: a compendium of processing and analysis techniques. Plant Methods 19, 60.

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R., 2022. Masked autoencoders are scalable vision learners, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009.

Heiwolt, K., Duckett, T., Cielniak, G., 2021. Deep semantic segmentation of 3d plant point clouds, in: Annual Conference Towards Autonomous Robotic Systems, Springer. pp. 36–45.

Jiang, L., Zhao, H., Shi, S., Liu, S., Fu, C.W., Jia, J., 2020. Pointgroup: Dual-set point grouping for 3d instance segmentation, in: Proceedings of the IEEE/CVF conference on computer vision and Pattern recognition, pp. 4867–4876.

Johnson, A., Hebert, M., 1999. Using spin images for eficient object recognition in cluttered 3d scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence 21, 433–449. doi:10.1109/34.765655.

Lee, J.J., Li, B., Benes, B., 2023. Latent l-systems: Transformer-based tree generator. ACM Transactions on Graphics 43, 1–16.

Li, D., Li, J., Xiang, S., Pan, A., 2022a. Psegnet: Simultaneous semantic and instance segmentation for point clouds of plants. Plant Phenomics .

Li, D., Li, T., Xu, S., Jin, S., 2025a. Ar-plant: Ar-based semi-automatic labeling system for 3d plant organs. ISPRS Journal of Photogrammetry and Remote Sensing 230, 843–860.

Li, D., Shi, G., Li, J., Chen, Y., Zhang, S., Xiang, S., Jin, S., 2022b. Plantnet: A dual-function point cloud segmentation network for multiple plant species. ISPRS Journal of Photogrammetry and Remote Sensing 184, 243–263.

Li, G., An, L., Yang, W., Yang, L., Wei, T., Shi, J., Wang, J., Doonan, J.H., Xie, K., Fernie, A.R., Lagudah, E.S., Wing, R.A., Gao, C., 2025b. Integrated biotechnological and AI innovations for crop improvement. Nature 643, 925–937. URL: https://www.nature.com/ articles/s41586-025-09122-8, doi:10.1038/s41586-025-09122-8.

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations. URL: https: //openreview.net/forum?id=Bkg6RiCqY7.

Luo, L., Jiang, X., Yang, Y., Samy, E.R.A., Lefsrud, M., Hoyos-Villegas, V., Sun, S., 2023. Ef-3dpseg: 3d organ-level plant shoot segmentation using annotation-eficient deep learning. Plant phenomics 5, 0080.

Pang, Y., Wang, W., Tay, F.E., Liu, W., Tian, Y., Yuan, L., 2022. Masked autoencoders for point cloud self-supervised learning, in: European Conference on Computer Vision, Springer. pp. 604–621.

Poursaeed, O., Jiang, T., Qiao, H., Xu, N., Kim, V.G., 2020. Self-supervised learning of point clouds via orientation estimation, in: 2020 International Conference on 3D Vision (3DV), IEEE. pp. 1018–1028.

Qi, C.R., Su, H., Mo, K., Guibas, L.J., 2017a. Pointnet: Deep learning on point sets for 3d classification and segmentation, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 652– 660.

Qi, C.R., Yi, L., Su, H., Guibas, L.J., 2017b. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30.

Qi, Z., Dong, R., Fan, G., Ge, Z., Zhang, X., Ma, K., Yi, L., 2023. Contrast with reconstruct: Contrastive 3D representation learning guided by generative pretraining, in: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (Eds.), Proceedings of the 40th International Conference on Machine Learning, PMLR. pp. 28223–28243. URL: https://proceedings.mlr.press/v202/qi23a.html.

Rao, Y., Lu, J., Zhou, J., 2023. Pointglr: Unsupervised structural representation learning of 3d point clouds. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 2193–2207. doi:10.1109/TPAMI. 2022.3159794.

Roggiolani, G., Magistri, F., Guadagnino, T., Behley, J., Stachniss, C., 2023. Unsupervised pre-training for 3d leaf instance segmentation. IEEE Robotics and Automation Letters 8, 7448–7455. doi:10.1109/LRA.2023. 3320018.

Rohlfs, C., 2025. Generalization in neural networks: A broad survey. Neurocomputing 611, 128701.

Rusu, R.B., Blodow, N., Beetz, M., 2009. Fast point feature histograms (fpfh) for 3d registration, in: 2009 IEEE International Conference on Robotics and Automation, pp. 3212–3217. doi:10.1109/ROBOT.2009. 5152473.

Sauder, J., Sievers, B., 2019. Self-supervised deep learning on point clouds by reconstructing space. Advances in neural information processing systems 32.

Sohail, S.S., Himeur, Y., Kheddar, H., Amira, A., Fadli, F., Atalla, S., Copiaco, A., Mansoor, W., 2025. Advancing 3d point cloud understanding through deep transfer learning: A comprehensive survey. Information Fusion 113, 102601.

Sohn, K., 2016. Improved deep metric learning with multi-class n-pair loss objective, in: Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., Garnett, R. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc. URL: https://proceedings.neurips.cc/paper\_files/ paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf.

Song, H., Wen, W., Wu, S., Guo, X., 2025. Comprehensive review on 3d point cloud segmentation in plants. Artificial Intelligence in Agriculture 15, 296–315.

Tian, B., Luo, L., Zhao, H., Zhou, G., 2022. Vibus: Data-eficient 3d scene parsing with viewpoint bottleneck and uncertainty-spectrum modeling. ISPRS Journal of Photogrammetry and Remote Sensing 194, 302–318.

Tian, Y., Jiang, X., Adamowski, J.F., Sun, S., 2025. Soybean-pcmae: Perturbation consistent masked autoencoders for few-shot segmentation of 3d soybean point clouds, in: 2025 ASABE Annual International Meeting, American Society of Agricultural and Biological Engineers. p. 1.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. Advances in neural information processing systems 30.

Wang, H., Liu, Q., Yue, X., Lasenby, J., Kusner, M.J., 2021. Unsupervised point cloud pre-training via occlusion completion, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9782–9792.

Wang, W., Yu, R., Huang, Q., Neumann, U., 2018. Sgpn: Similarity group proposal network for 3d point cloud instance segmentation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Wang, X., Liu, S., Shen, X., Shen, C., Jia, J., 2019a. Associatively segmenting instances and semantics in point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., Solomon, J.M., 2019b. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog) 38, 1–12.

Watson, A., Ghosh, S., Williams, M.J., Cuddy, W.S., Simmonds, J., Rey, M.D., Asyraf Md Hatta, M., Hinchlife, A., Steed, A., Reynolds, D., Adamski, N.M., Breakspear, A., Korolev, A., Rayner, T., Dixon, L.E., Riaz, A., Martin, W., Ryan, M., Edwards, D., Batley, J., Raman, H., Carter, J., Rogers, C., Domoney, C., Moore, G., Harwood, W., Nicholson, P., Dieters, M.J., DeLacy, I.H., Zhou, J., Uauy, C., Boden, S.A., Park, R.F., Wulf, B.B.H., Hickey, L.T., 2018. Speed breeding is a powerful tool to accelerate crop research and breeding. Nature Plants 4, 23–29.

Xie, K., Cui, C., Jiang, X., Zhu, J., Liu, J., Du, A., Yang, W., Song, P., Zhai, R., 2025. Automated 3d segmentation of plant organs via the plant-mae: A self-supervised learning framework. Plant Phenomics , 100049.

Yang, W., Feng, H., Zhang, X., Zhang, J., Doonan, J.H., Batchelor, W.D., Xiong, L., Yan, J., 2020. Crop Phenomics and High-Throughput Phenotyping: Past Decades, Current Challenges, and Future Perspectives. Molecular Plant 13, 187–214. URL: https://linkinghub.elsevier.com/ retrieve/pii/S1674205220300083, doi:10.1016/j.molp.2020.01.008.

Yang, X., Miao, T., Tian, X., Wang, D., Zhao, J., Lin, L., Zhu, C., Yang, T., Xu, T., 2024. Maize stem–leaf segmentation framework based on deformable point clouds. ISPRS journal of photogrammetry and remote sensing 211, 49–66.

Yi, L., Kim, V.G., Ceylan, D., Shen, I.C., Yan, M., Su, H., Lu, C., Huang, Q., Shefer, A., Guibas, L., 2016. A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics (ToG) 35, 1–12.

Yu, X., Tang, L., Rao, Y., Huang, T., Zhou, J., Lu, J., 2022. Point-bert: Pre-training 3d point cloud transformers with masked point modeling, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19313–19322.

Zha, Y., Ji, H., Li, J., Li, R., Dai, T., Chen, B., Wang, Z., Xia, S.T., 2024. Towards compact 3d representations via point feature enhancement masked autoencoders. Proceedings of the AAAI Conference on Artificial Intelligence 38, 6962–6970. URL: https://ojs.aaai.org/index. php/AAAI/article/view/28522, doi:10.1609/aaai.v38i7.28522.

Zhang, R., Guo, Z., Gao, P., Fang, R., Zhao, B., Wang, D., Qiao, Y., Li, H., 2022. Point-m2ae: multi-scale masked autoencoders for hierarchical point cloud pre-training. Advances in neural information processing systems 35, 27061–27074.

Zhang, X., Zhang, S., Yan, J., 2024. Pcp-mae: learning to predict centers for point masked autoencoders, in: Proceedings of the 38th International Conference on Neural Information Processing Systems, pp. 80303– 80327.

Zhang, X., Zhang, S., Yan, J., 2025. Towards more diverse and challenging pre-training for point cloud learning: Self-supervised cross reconstruction with decoupled views, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28696–28706.