# FM-OCC: FEATURE-DIVERSE UPSAMPLING AND MEMORY-AWARE DECODING FOR 3D OCCUPANCY PREDICTION

Yujia Zhai Guoming Huang Zhuanghuai Wang Bangquan Xie

Great Bay University

Corresponding author

## ABSTRACT

3D semantic occupancy prediction estimates voxel-wise occupancy and semantic labels throughout a scene. Our experiments reveal that conventional sparse transposed-convolution upsampling yields poorly differentiated voxel responses and that successive decoder layers discard previously correct semantic predictions. To address these issues, we present Feature-Diverse Upsampling and Memory-Aware Decoding for 3D Occupancy Prediction (FM-Occ), comprising two components. First, 3D Sparse Subvoxel Shuffle (3S) mitigates feature homogenization by constructing candidate child features in the channel dimension, jointly transforming and normalizing them, and then rearranging them onto a finer sparse grid with explicit feature–coordinate correspondence. Second, Entropy-Regulated Layer-wise Memory (ER-LM) retains semantic information across decoder layers by maintaining a layer-wise memory for each query through gated recurrent updates and using prediction entropy to regulate the historical information propagated to the next layer. During training, a class-weighted entropy loss $\mathcal { L } _ { \mathrm { c w e } }$ provides label-guided regularization for matched semantic queries. On SemanticKITTI, 3S increases the spatial coefficient of variation of normalized voxel-wise root-mean-square (RMS) responses from 0.41% to 4.03%, while ER-LM reduces semantic degradation and improves error correction across decoder layers. The combined FM-Occ improves mIoU from 12.16% to 12.73% with negligible memory overhead. ER-LM further improves mIoU in the BEV-based DISC framework from 16.78% to 17.17%, supporting its applicability across 3D voxel-based and BEV-based architectures.

## 1 INTRODUCTION

3D occupancy prediction (Cao & De Charette, 2022; Wei et al., 2023; Zhang et al., 2023) aims to infer voxel-wise occupancy states and semantic categories in 3D space, providing a dense and structured scene representation for autonomous driving and robotic perception. Compared with 3D object detection (Yan et al., 2018; Lang et al., 2019; Li et al., 2024), which outputs sparse bounding boxes, occupancy prediction explicitly models scene geometry and spatial continuity. Following sparse occupancy approaches such as SparseOcc and OPUS (Tang et al., 2024; Wang et al., 2024a), we study an image-based encoder–decoder architecture that constructs sparse 3D features and progressively predicts semantic occupancy through feature upsampling and iterative Transformer decoding.

We further examine these two key stages and identify two overlooked issues: weak spatial differentiation of upsampled voxel features and progressive loss of semantic information during iterative decoding.

Observation 1: Limited differentiation of upsampled voxel features. Our analysis shows that sparse transposed convolution followed by group normalization produces voxel features with limited spatial variation in their response magnitudes. This observation suggests that the organization of feature construction and normalization deserves attention alongside the expansion of the voxel grid. We introduce 3D Sparse Subvoxel Shuffle (3S), which constructs candidate child representations in the channel dimension, jointly transforms and normalizes them, and then rearranges them onto the finer sparse grid. Across 50 validation frames, 3S increases the spatial CV of voxel-wise rootmean-square (RMS) responses from 0.41% to 4.03% (Fig. 1, left). 3S also improves both mIoU and IoU.

![](images/9af59fe0d4c423a90fc6a3ed6efa40a2a8b16582b6ab09fb3e4bf20d2c34eda0.jpg)

![](images/648a4acf5f74923107232518919f5543aaa8f273b68438ae15fb41f3af72c433.jpg)

![](images/23225206116fe7d66c886c02164eaf978e6001998aad4a544c0d61db597b4c76.jpg)  
Figure 1: Two challenges in occupancy prediction: feature homogenization and fragile semantic consistency. Left: Spatial variation of voxel-wise feature responses after upsampling, measured by spatial CV (higher indicates greater variation). Right: Mean class-wise semantic degradation (mD; lower is better) without and with $\begin{array} { r } { \mathrm { E R - L M } + \mathcal { L } _ { \mathrm { c w e } } , } \end{array}$ , comparing (a) adjacent decoder layers and (b) intermediate layers with the final layer.

Observation 2: Semantic degradation across decoder layers. Our analysis reveals substantial semantic degradation during iterative decoding, with a mean class-wise degradation rate of 21.00% from the first to the final decoder layer (Fig. 1, right). To address this issue, we introduce Entropy-Regulated Layer-wise Memory (ER-LM), which maintains a gated memory for each query across decoder layers. Shared recurrent parameters accumulate historical query information, while queryspecific classification entropy regulates the memory state propagated to subsequent stages. During training, a complementary class-weighted entropy loss $\mathcal { L } _ { \mathrm { c w e } }$ encourages concentrated class predic tions on matched positive queries, with greater relative emphasis on rare categories. In the separate decoding experiments, ER-LM trained with $\mathcal { L } _ { \mathrm { c w e } }$ improves final mIoU from 12.16% to 12.88% and reduces the mean class-wise semantic degradation rate from 21.00% to 16.50%.

Our contributions are summarized as follows:

• We propose FM-Occ, a 3D semantic occupancy framework that improves feature differentiation during sparse upsampling and semantic preservation across decoder layers.

• We propose 3S to preserve spatial variation in feature magnitudes during sparse upsampling by reorganizing child-feature construction, normalization, and spatial expansion.

• We introduce ER-LM to mitigate cross-layer semantic degradation through entropyregulated memory, complemented by a class-weighted entropy loss during training. ER-LM generalizes across both 3D voxel-based and BEV-based frameworks.

## 2 RELATED WORK

## 2.1 REPRESENTATIVE APPROACHES FOR 3D OCCUPANCY PREDICTION

Vision-based semantic occupancy methods differ in how they represent and process 3D scenes. Dense voxel-based approaches construct features on regular 3D grids to predict scene geometry and semantics (Cao & De Charette, 2022; Wang et al., 2023; Wei et al., 2023; Zhang et al., 2023). BEV-based and tri-perspective representations reduce volumetric processing by projecting or decomposing 3D scene features onto compact planes (Hou et al., 2024; Xue et al., 2024; Liu et al., 2025; Huang et al., 2023; Zuo et al., 2023). Sparse approaches avoid maintaining features over the entire 3D grid by operating on a limited set of active elements. Tang et al. (2024) adopts a sparse voxel representation and enhances it through multi-scale feature interaction. Wang et al. (2024a) adopts a different sparse formulation, representing occupancy as a set of occupied locations and semantic classes predicted by learnable queries. These approaches illustrate different choices of scene representation; our work examines feature reconstruction and semantic information retention within sparse voxel and iterative decoding architectures.

![](images/e7cb6e2366d9d8f61be3677c487e5425d9ef2927010b5bb54ed5c0fe91347d27.jpg)  
Figure 2: Overall framework of FM-Occ. 3S performs sparse 3D feature upsampling, while ER-LM propagates information across successive decoder stages under entropy-guided regulation. $\mathcal { L } _ { \mathrm { c w e } }$ is applied only during training.

## 2.2 FEATURE UPSAMPLING

2D feature upsampling has been explored through learned kernels, content-aware reassembly, adaptive sampling, and channel-to-space rearrangement (Zeiler et al., 2010; Wang et al., 2019; Liu et al., 2023; Shi et al., 2016). These methods operate on dense feature maps with spatial positions implicitly defined by regular tensor grids. Unlike dense 2D feature maps, sparse 3D representations store only active voxel features together with explicit coordinates. Sparse 3D upsampling must therefore expand both features and coordinates while maintaining their correspondence. 3S addresses this requirement through channel-to-space feature rearrangement and matched coordinate expansion.

## 2.3 INFORMATION RETENTION IN OCCUPANCY DECODING

Query-based occupancy methods employ stacked decoder layers to progressively refine semantic predictions (Zhang et al., 2023; Wang et al., 2024b; Oh et al., 2025). In these decoders, successive layers repeatedly update the query representations, with information from earlier stages carried forward implicitly through the evolving queries. As a result, previously acquired semantic information may be altered or lost during subsequent refinement, motivating explicit mechanisms for cross-layer information retention.

## 3 METHOD

The overall framework of FM-Occ is illustrated in Fig. 2. Latent Feature Extraction first maps image features into a sparse 3D representation. The Sparse Latent Diffuser then performs scene completion, followed by a multi-scale sparse feature pyramid. 3S is used to recover higher-resolution sparse features during multi-scale feature fusion. Finally, the Transformer decoder predicts semantic occupancy, where ER-LM propagates information across decoder stages under entropy-guided regulation. The class-weighted entropy loss $\mathcal { L } _ { \mathrm { c w e } }$ is additionally applied during training.

## 3.1 3D SPARSE SUBVOXEL SHUFFLE

Sparse feature pyramids rely on upsampling to recover fine-grained voxel representations from coarse features. Conventional sparse transposed convolution generates child features while expanding the spatial grid, coupling feature construction with spatial placement. We introduce 3S, which first generates $\bar { r } ^ { 3 }$ candidate child representations in the channel dimension for each active parent voxel. These representations are jointly normalized and activated at the coarse resolution before being deterministically rearranged to their corresponding locations on the finer sparse grid. The overall process is illustrated in Fig. 3.

![](images/98627612f76839119545241b0b437d61b74b6f9d90c733e1439507e015a8a4b7.jpg)  
Figure 3: 3S. The upper branch expands feature channels through a submanifold convolution, applies group normalization and ReLU, and reshapes the features into $r ^ { 3 }$ child representations per parent voxel. The lower branch generates the corresponding fine-grid coordinates. The two branches combine to form the upsampled sparse tensor, illustrated here for $r = 2$

## 3.1.1 SPARSE TENSOR REPRESENTATION

Let the input sparse tensor be represented as $\mathcal { S } = ( \mathbf { F } , \mathbf { I } , \mathbf { s } )$ , where $\mathbf { F } \in \mathbb { R } ^ { N \times C _ { \mathrm { i n } } }$ contains the features of N active voxels, $\mathbf { I } \in \mathbb { Z } ^ { N \times \dot { 4 } }$ stores their batch and three-dimensional spatial indices, and $\mathrm { ~ \bf ~ s ~ } =$ $( S _ { 1 } , S _ { 2 } , S _ { 3 } )$ denotes the spatial shape. The coordinate of the n-th active voxel is written as ${ \mathbf I } _ { n } =$ $\left( b _ { n } , \mathbf { p } _ { n } \right)$ , where $b _ { n }$ is the batch index and ${ \bf p } _ { n } = ( p _ { n , 1 } , p _ { n , 2 } , p _ { n , 3 } )$ is its three-dimensional spatial coordinate.

## 3.1.2 CHILD-FEATURE CONSTRUCTION AND JOINT NORMALIZATION

Given an upsampling factor r, a submanifold sparse convolution expands each input feature from $C _ { \mathrm { i n } }$ channels to $r ^ { 3 } C _ { \mathrm { o u t } }$ channels, yielding $\widetilde { \mathbf { F } } = \mathrm { C o n v } _ { 1 \times 1 \times 1 } ( \mathbf { F } )$ , where $\widetilde { \mathbf { F } } \in \mathbb { R } ^ { N \times ( \overline { { r ^ { 3 } } } C _ { \mathrm { o u t } } ) }$

Before spatial rearrangement, we apply group normalization and a ReLU activation to the expanded representation, obtaining $\overline { { \mathbf { F } } } = \mathrm { R e L U } ( \mathrm { G N } ( \widetilde { \mathbf { F } } ) )$ ).

## 3.1.3 CHANNEL-TO-SPACE REARRANGEMENT

The normalized representation is reshaped as

$$
\mathbf { \overline { { F } } } \in \mathbb { R } ^ { N \times ( r ^ { 3 } C _ { \mathrm { { o u t } } } ) } \longrightarrow \mathbf { F } ^ { \mathrm { { s u b } } } \in \mathbb { R } ^ { N \times r ^ { 3 } \times C _ { \mathrm { { o u t } } } } .\tag{1}
$$

Let $q \in \{ 0 , \ldots , r ^ { 3 } - 1 \}$ denote a child index. The exact feature mapping implemented by 3S is

$$
\mathbf { F } _ { n , q , c } ^ { \mathrm { s u b } } = \overline { { \mathbf { F } } } _ { n , c r ^ { 3 } + q } ,\tag{2}
$$

where $c \in \{ 0 , \ldots , C _ { \mathrm { o u t } } - 1 \}$ . Flattening the parent and child dimensions produces

$$
\mathbf { F } _ { \mathrm { o u t } } \in \mathbb { R } ^ { ( N r ^ { 3 } ) \times C _ { \mathrm { o u t } } } , \qquad \mathbf { F } _ { \mathrm { o u t } } [ n r ^ { 3 } + q , c ] = \mathbf { F } _ { n , q , c } ^ { \mathrm { s u b } } .\tag{3}
$$

Thus, each parent voxel produces $r ^ { 3 }$ location-specific child-feature vectors.

## 3.1.4 COORDINATE EXPANSION AND ALIGNMENT

We enumerate the complete set of three-dimensional child offsets as $\mathcal { D } _ { r } ~ = ~ \{ ( d _ { 1 } , d _ { 2 } , d _ { 3 } ) ~ \}$ $d _ { 1 } , d _ { 2 } , d _ { 3 } \in \{ 0 , \ldots , r - 1 \} \}$ . Let $\mathbf { d } _ { q } \in \mathcal { D } _ { r }$ be the offset associated with child index q. The output coordinate paired with the feature in Eq. equation 3 is

$$
{ \bf I } _ { \mathrm { o u t } } [ n r ^ { 3 } + q ] = ( b _ { n } , r { \bf p } _ { n } + { \bf d } _ { q } ) .\tag{4}
$$

The output spatial shape is correspondingly updated to ${ \bf s } _ { \mathrm { o u t } } = ( r S _ { 1 } , r S _ { 2 } , r S _ { 3 } )$

The resulting sparse tensor is $S _ { \mathrm { o u t } } = ( \mathbf { F } _ { \mathrm { o u t } } , \mathbf { I } _ { \mathrm { o u t } } , \mathbf { s } _ { \mathrm { o u t } } )$ . Within each sparse pyramid stage, $S _ { \mathrm { o u t } }$ is subsequently merged with the existing feature level at the same resolution.

## 3.2 ENTROPY-REGULATED LAYER-WISE MEMORY

Iterative query updates can both correct semantic errors and disrupt previously correct predictions. To support the reuse of semantic evidence across decoding stages, we introduce ER-LM. As illustrated in the right panel of Fig. 1, ER-LM maintains a gated memory alongside the evolving query representations and incorporates its output into subsequent decoder updates. The recurrent module is shared across decoding stages, while classification entropy provides a query-specific signal for modulating memory propagation.

## 3.2.1 GATED MEMORY ACROSS DECODING STAGES

Let $\mathbf { Q } ^ { ( \ell ) } \in \mathbb { R } ^ { B \times N _ { q } \times D }$ denote the query representations after decoder layer ℓ, where B, $N _ { q } ,$ and D are the batch size, number of queries, and query feature dimension, respectively. The initial learnable queries are denoted by $\bar { \mathbf { Q } ^ { ( 0 ) } }$ . We use an LSTM with hidden dimension $D _ { h }$ to maintain a separate memory state for each query across the L decoding stages.

At the beginning of each forward pass, the hidden and cell states are initialized to zero, $\mathbf { h } ^ { ( 0 ) } = \mathbf { 0 }$ and $\widetilde { \mathbf { c } } ^ { ( 0 ) } = \mathbf { 0 }$ , where both states have shape $1 \times ( B N _ { q } ) \times D _ { h }$

Before decoder layer ℓ, the incoming queries are reshaped as

$$
\mathbf { X } ^ { ( \ell ) } = \mathcal { R } \Big ( \mathbf { Q } ^ { ( \ell - 1 ) } \Big ) \in \mathbb { R } ^ { 1 \times ( B N _ { q } ) \times D }
$$

and passed to the shared LSTM:

$$
\left( \mathbf { h } ^ { \left( \ell \right) } , \mathbf { c } ^ { \left( \ell \right) } \right) = \mathrm { L S T M } \left( \mathbf { X } ^ { \left( \ell \right) } ; \mathbf { h } ^ { \left( \ell - 1 \right) } , \widetilde { \mathbf { c } } ^ { \left( \ell - 1 \right) } \right) ,\tag{5}
$$

where R merges the batch and query dimensions, and $\mathcal { R } ^ { - 1 }$ restores their original layout. Each LSTM invocation processes one recurrent step, with the hidden and cell states propagated across decoding stages. The standard LSTM gates control the incorporation of current query information and the retention of previous memory. Projecting the hidden state to the query dimension and restoring the batch–query layout yields the memory feature $\mathbf { E } ^ { ( \ell ) } \in \mathbb { R } ^ { B \times N _ { q } \times \mathbf { \dot { D } } }$ , which is fused with the incoming queries before decoder layer ℓ:

$$
\mathbf { E } ^ { ( \ell ) } = \mathcal { R } ^ { - 1 } \Big ( \mathbf { h } ^ { ( \ell ) } \mathbf { W } _ { p } ^ { \top } + \mathbf { b } _ { p } \Big ) .\tag{6}
$$

$$
\widehat { \mathbf { Q } } ^ { ( \ell - 1 ) } = \mathbf { Q } ^ { ( \ell - 1 ) } + \alpha _ { \ell } \mathbf { E } ^ { ( \ell ) } . \qquad ( 7 ) \qquad \mathbf { Q } ^ { ( \ell ) } = \mathcal { D } _ { \ell } \Big ( \widehat { \mathbf { Q } } ^ { ( \ell - 1 ) } , \mathbf { F } _ { s ( \ell ) } \Big ) .\tag{8}
$$

where $\mathbf { W } _ { p } \in \mathbb { R } ^ { D \times D _ { h } }$ and $\mathbf { b } _ { p } \in \mathbb { R } ^ { D }$ are the shared projection matrix and bias, respectively, $\mathcal { D } _ { \ell }$ denotes decoder layer ℓ, and $\mathbf { F } _ { s ( \ell ) }$ is the feature level accessed at that stage. The fusion weight is $\alpha _ { \ell } = \sigma ( a _ { \ell } )$ ; the scalar $a \ell$ is learned separately for each decoder layer and initialized to 0.1.

## 3.2.2 ENTROPY-REGULATED MEMORY PROPAGATION

We use classification entropy to modulate the cell state transmitted between decoding stages. $\mathrm { L e t } ~ \mathbf { S } ^ { ( \ell - 1 ) }$ denote the classification logits obtained from the preceding stage, and let K be the number of classification outputs, including the no-object class. For each query $q ,$ the class probabilities are $p _ { q , k } ^ { ( \ell - 1 ) } \ = \ \mathrm { \bar { S o f t m a x } } ( \mathbf { S } _ { q } ^ { ( \ell - 1 ) } ) _ { k }$ . The entropy and its normalized form are

$$
H _ { q } ^ { ( \ell - 1 ) } = - \sum _ { k = 1 } ^ { K } p _ { q , k } ^ { ( \ell - 1 ) } \log \left( p _ { q , k } ^ { ( \ell - 1 ) } + \epsilon \right) .\tag{9}
$$

$$
\overline { { { H } } } _ { q } ^ { ( \ell - 1 ) } = \frac { H _ { q } ^ { ( \ell - 1 ) } } { \log K } .\tag{10}
$$

where $\epsilon = 1 0 ^ { - 8 }$ ensures numerical stability.

For stages $\ell \geq 2$ , a shared multilayer perceptron maps the normalized entropy to a query-specific modulation coefficient:

$$
r _ { q } ^ { ( \ell ) } = \sigma \left( g \left( \overline { { { H } } } _ { q } ^ { ( \ell - 1 ) } \right) \right) .\tag{11}
$$

where $g ( \cdot )$ is a lightweight MLP that maps prediction entropy to a gating coefficient. At the first stage, we use the default coefficient $r _ { q } ^ { ( 1 ) } = 0 . 5$

After the LSTM update in Eq. equation 5, the cell state is modulated as

$$
\widetilde { \mathbf { c } } ^ { ( \ell ) } = \mathbf { c } ^ { ( \ell ) } \odot \left( 1 + \beta \mathcal { R } \Big ( \mathbf { r } ^ { ( \ell ) } \Big ) \right) , \qquad \beta = 0 . 3 .\tag{12}
$$

The coefficient $\beta = 0 . 3$ bounds the modulation gain between 1 and 1.3. The adjusted cell state $\widetilde { \mathbf { c } } ^ { ( \ell ) }$ is passed to the next recurrent step, while the current hidden output $\mathbf { h } ^ { ( \ell ) }$ is used for query fusion.

## 3.2.3 CLASS-WEIGHTED ENTROPY REGULARIZATION

Because the prediction entropy used by ER-LM is not directly supervised, we introduce $\mathcal { L } _ { \mathrm { c w e } }$ to explicitly regularize it on matched positive queries. Ground-truth assignments determine the positive queries and their class weights. This entropy supervision also reduces uncertainty in the predicted class distributions, complementing the classification loss.

For prediction stage ℓ, let $\mathcal { Q } _ { + } ^ { ( \ell ) }$ denote the queries matched to ground-truth semantic masks, excluding those assigned to the no-object class; let $y _ { q } ^ { ( \ell ) }$ denote their assigned labels, and $w _ { y _ { q } ^ { ( \ell ) } }$ the corresponding class weights used by the classification loss. The stage-wise loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c w e } } ^ { ( \ell ) } = \biggl ( \sum _ { q \in \mathcal { Q } _ { + } ^ { ( \ell ) } } w _ { y _ { q } ^ { ( \ell ) } } H _ { q } ^ { ( \ell ) } \biggr ) \Big / \biggl ( \sum _ { q \in \mathcal { Q } _ { + } ^ { ( \ell ) } } w _ { y _ { q } ^ { ( \ell ) } } \biggr ) . } \end{array}\tag{13}
$$

We apply this regularizer to the initial query predictions and the outputs of all decoder layers; the overall training objective is

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { c w e } } { \mathcal { L } } _ { \mathrm { c w e } } , \qquad { \mathcal { L } } _ { \mathrm { c w e } } = \sum _ { \ell = 0 } ^ { L } { { \mathcal { L } } _ { \mathrm { c w e } } ^ { ( \ell ) } } .\tag{14}
$$

where $\ell = 0$ denotes the initial prediction stage, L is the number of decoder layers, and $\mathcal { L } _ { \mathrm { b a s e } }$ comprises the original classification, mask, Dice, and voxel-level losses. We set $\lambda _ { \mathrm { c w e } } = 1 . 0$ in our experiments.

## 4 EXPERIMENTAL RESULTS

## 4.1 EXPERIMENTAL SETUP

We evaluate FM-Occ on the SemanticKITTI (Behley et al., 2019) dataset. SemanticKITTI is a largescale dataset for 3D semantic occupancy prediction in autonomous driving scenarios, based on the raw KITTI sequences. The dataset contains 21 classes, including 19 semantic categories, one freespace class, and one unknown class. We train FM-Occ for 30 epochs on four NVIDIA RTX 4090 GPUs using AdamW with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a total batch size of 4. We adopt a step learning-rate schedule.

We adopt commonly used evaluation metrics for 3D occupancy prediction, including voxel-wise Intersection over Union (IoU) and mean Intersection over Union (mIoU). To provide a more systematic analysis of model performance across different semantic structures, we further propose categorygrouped evaluation metrics, namely ComplexIoU, TailIoU, and RegularIoU. Detailed definitions and the category grouping of these metrics are provided in Appendix A.1.

## 4.2 COMPARISON WITH STATE-OF-THE-ART

Table 1 compares FM-Occ with recent semantic scene completion methods on SemanticKITTI. FM-Occ achieves an IoU of 36.18% and an mIoU of 12.73%. To further evaluate the transferability of ER-LM, we apply it from the sparse 3D voxel-based FM-Occ to the BEV-based DISC framework, improving DISC from 16.78% to 17.17% mIoU and from 45.54% to 45.67% IoU. The resulting 17.17% mIoU is comparable to MRA (17.14%) and higher than SGFormer (16.68%), demonstrating competitive performance against recent methods and the transferability of ER-LM across different scene representations.

Table 1: Quantitative comparison on the SemanticKITTI validation set
<table><tr><td></td><td></td><td></td><td>SIk</td><td></td><td>oihe-md. palin (112%)</td><td>building (0.56%)</td><td>(110%)</td><td>1</td><td>beyce</td><td>motycle (0.0%)</td><td>oheeth (%£0%) (%00)</td><td>vetatton</td><td>1</td><td> </td><td>20222</td><td>beyclist</td><td>motcist (0.7%)</td><td>(0.0%)</td><td>20</td><td>2</td></tr><tr><td>Method</td><td>IoU</td><td>mIoU</td><td></td><td></td><td></td><td></td><td>■</td><td></td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>■</td><td></td><td>■</td><td></td><td></td><td>uaa-n (%800)</td></tr><tr><td>TPVFormer†(Huang et al., 2023)</td><td>35.61</td><td>11.36</td><td>56.50</td><td>25.87</td><td>20.60</td><td>0.85 13.88</td><td>23.81</td><td>8.08</td><td>0.36</td><td>0.05</td><td>4.35</td><td>16.92</td><td>2.26</td><td>30.38</td><td>0.51</td><td>0.89</td><td>0.00</td><td>5.94</td><td>3.14</td><td>1.52</td></tr><tr><td>MonoScene†(Cao &amp; De Charette, 2022)</td><td>36.86</td><td>11.08</td><td>56.52</td><td>26.72</td><td>14.27</td><td>0.46</td><td>14.09</td><td>23.26</td><td>6.98 0.61</td><td>0.45</td><td>1.48</td><td>17.89</td><td>2.81</td><td>29.64</td><td>1.86</td><td>1.20</td><td>0.00</td><td>5.84</td><td>4.14</td><td>2.25</td></tr><tr><td>OccFormer(Zhang et al., 2023)</td><td>36.50</td><td>13.46</td><td>58.85</td><td>26.88</td><td>19.61</td><td>0.31 14.40</td><td>25.09</td><td>25.53</td><td>0.81</td><td>1.19</td><td>8.52</td><td>19.63</td><td>3.93</td><td>32.62</td><td>2.78</td><td>2.82</td><td>0.00</td><td>5.61</td><td>4.26</td><td>2.86</td></tr><tr><td>VoxFormer*(Li et al., 2023)</td><td>43.92</td><td>13.12</td><td>54.44</td><td>27.05</td><td>16.97</td><td>0.28 19.64</td><td>26.78</td><td>7.66</td><td>1.50</td><td>0.41</td><td>5.68</td><td>26.12</td><td>6.33</td><td>32.80</td><td>1.15</td><td>0.26</td><td>0.00</td><td>7.93</td><td>8.94</td><td>5.41</td></tr><tr><td>Symphonies(Jiang et al., 2024)</td><td>41.92</td><td>14.89</td><td>56.37</td><td>27.58</td><td>15.28</td><td>0.95 21.64</td><td>28.68</td><td>20.44</td><td>2.54</td><td>2.82</td><td>13.89</td><td>25.72</td><td>6.60</td><td>30.87</td><td>3.52</td><td>2.24</td><td>0.00</td><td>8.40</td><td>9.57</td><td>5.76</td></tr><tr><td>SGFormer(Guo et al., 2025)</td><td>45.01</td><td>16.68</td><td>60.85</td><td>31.53</td><td>19.32</td><td>2.14 26.05</td><td>32.17</td><td>20.30</td><td>2.95</td><td>3.10</td><td>11.55</td><td>27.11</td><td>8.28</td><td>38.47</td><td>3.66</td><td>1.49</td><td>0.00</td><td>9.28</td><td>11.58</td><td>7.22</td></tr><tr><td>MRA(Yang &amp; Peng, 2026)</td><td>47.28</td><td>17.14</td><td>61.92</td><td>33.97</td><td>27.35</td><td>0.06 25.89</td><td>34.63</td><td>20.12</td><td>0.53</td><td>1.69</td><td>11.06</td><td>30.08</td><td>9.54 3.29</td><td>38.51</td><td>0.54</td><td>0.16</td><td>0.00</td><td>10.13</td><td>13.15</td><td>6.27 2.59</td></tr><tr><td>SparseOcc*(Tang et al., 2024)</td><td>35.21 45.54</td><td>12.15</td><td>57.43</td><td>27.59</td><td>21.18</td><td>0.00 14.97</td><td>25.32</td><td>1.62</td><td>0.35</td><td>0.92</td><td>11.51</td><td>18.25 27.41</td><td>7.66</td><td>31.17 39.01</td><td>3.82</td><td>0.40 3.11</td><td>0.00 0.00</td><td>6.41 9.73</td><td>4.06 11.08</td><td>6.88</td></tr><tr><td>DISC*(Liu et al., 2025)</td><td>36.18</td><td>16.78 12.73</td><td>63.65 59.00</td><td>34.35</td><td>24.46</td><td>1.56 23.42 15.88</td><td>32.95 25.56</td><td>15.55 7.54</td><td>3.04 0.48</td><td>3.06 0.73</td><td>8.06 10.72</td><td>18.84</td><td>3.48</td><td>31.83</td><td>3.82 2.66</td><td>1.69</td><td>0.00</td><td>6.12</td><td>4.31</td><td>2.77</td></tr><tr><td>FM-Occ</td><td>45.67</td><td>17.17</td><td>62.83</td><td>29.03</td><td>20.55 24.66</td><td>0.62 2.04 23.66</td><td>33.00</td><td>19.98</td><td>2.93</td><td>3.19</td><td>11.65</td><td>27.61</td><td>7.90</td><td>37.71</td><td>4.02</td><td>2.95</td><td>0.01</td><td>9.56</td><td>10.97</td><td>7.56</td></tr><tr><td>DISC + ER-LM</td><td>(+0.13)</td><td>(+0.39)</td><td>(-0.82)</td><td>33.96 (-0.39)</td><td>(+0.20)</td><td>(+0.48) (+0.24)</td><td>(+0.05)</td><td>(+4.43)</td><td>(-0.11)</td><td>(+0.13)</td><td>(+3.59)</td><td>(+0.20)</td><td>(+0.24)</td><td>(-1.30)</td><td>(+0.20)</td><td>(-0.16)</td><td>(+0.01)</td><td>(-0.17)</td><td>(-0.11)</td><td>(+0.68)</td></tr></table>

Note: \* denotes results reproduced by our implementation, and † denotes results taken from Li et al. (2023). For $\mathrm { D I S C } + \mathrm { E R } { \mathrm { - } } \mathrm { L M } ,$ values in parentheses indicate differences from the reproduced DISC baseline. The best results are highlighted in bold, and the second-best results are underlined.

## 4.3 ABLATION STUDIES

Table 2 compares the proposed components and their combinations. 3S improves IoU and mIoU by 0.49% and 0.26%, respectively. ER-LM primarily improves semantic prediction, increasing mIoU by 0.51% while maintaining comparable occupancy IoU. Adding $\mathcal { L } _ { \mathrm { c w e } }$ further increases mIoU to 12.88%. The full configuration improves all five accuracy metrics

Table 2: Ablation studies of 3S, ER-LM, and $\mathcal { L } _ { \mathrm { c w e } }$
<table><tr><td>3S</td><td>ER-LM</td><td> $\pmb { \mathcal { L } } _ { \mathrm { c w e } }$ </td><td>IoU</td><td>mIoU</td><td>Complex</td><td>Tail</td><td>Regular</td><td>Mem.</td></tr><tr><td>√</td><td></td><td></td><td>36.13</td><td>12.16</td><td>23.00</td><td>1.36</td><td>10.32</td><td>13.05</td></tr><tr><td></td><td></td><td></td><td>36.62</td><td>12.42</td><td>23.14</td><td>1.53</td><td>10.80</td><td>13.06</td></tr><tr><td></td><td>√</td><td></td><td>36.11</td><td>12.67</td><td>23.07</td><td>1.81</td><td>11.39</td><td>13.06</td></tr><tr><td></td><td>√</td><td>√</td><td>35.85</td><td>12.88</td><td>23.31</td><td>1.55</td><td>12.04</td><td>13.06</td></tr><tr><td>√</td><td>√</td><td>√</td><td>36.18</td><td>12.73</td><td>23.34</td><td>1.39</td><td>11.69</td><td>13.09</td></tr></table>

over the baseline, although the gains of the individual components are not strictly additive. These improvements introduce negligible memory overhead, with measured GPU memory increasing from 13.05 to 13.09 GB.

Table 3 further evaluates robustness across five random seeds. Both 3S and ER-LM retain their principal improvements across repeated runs. 3S shows stable semantic performance, while ER-LM provides a consistent mIoU gain and reduces the variation on regular categories. These results indicate that the observed improvements are not dependent on a single favorable training run.

Table 3: Robustness evaluation across five random seeds. Results are reported as mean ± standard deviation
<table><tr><td>Method</td><td>IoU</td><td>mIoU</td><td>Complex</td><td>Tail</td><td>Regular</td></tr><tr><td>-=</td><td>35.91±0.16</td><td>12.34±0.21</td><td>22.96±0.34</td><td>1.29±0.16</td><td>11.31±0.95</td></tr><tr><td>+ 3S</td><td>36.18±0.32</td><td>12.44±0.12</td><td>23.15±0.36 1.49±0.27</td><td></td><td> $1 0 . 8 9 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td></td><td>+ ER-LM35.91±0.26</td><td>12.56±0.25</td><td>22.78±0.51</td><td></td><td>1.32±0.15 11.87±0.39</td></tr></table>

Note. All configurations were re-run under the same setting, which differs slightly from Table 2.

## 4.4 ANALYSIS OF 3S

## 4.4.1 COMPARISON OF UPSAMPLING FEATURE RESPONSES

We further examine the spatial variation of voxel-wise feature responses produced by different upsampling schemes. As shown in Table 4, 3S achieves a mean spatial coefficient of variation (CV) of 4.03%, compared with 0.41% for sparse transposed convolution, corresponding to a 9.83× increase. Figure 4 further shows clearer response separation between the car and its surrounding background in region A and between vegetation and the ground in regions B and C. The definition and measurement protocol of the feature-response variation metric, together with details of the feature-response visualization and analysis, are provided in $\mathsf { A p - }$ pendix A.2.

Table 4: Comparison of spatial variation in voxel-wise RMS responses
<table><tr><td>Method</td><td>CV (%)</td><td>Ratio</td></tr><tr><td>Transposed conv.</td><td>0.41</td><td>1.00×</td></tr><tr><td>3S</td><td>4.03</td><td>9.83×</td></tr></table>

![](images/f199ce0c5f390a0d41b27f11bcde47d45b1bea6be2bd465f63463fd8012d6357.jpg)  
Figure 4: Feature-response comparison after upsampling. Full images are shown on the left, with enlarged regions on the right. Each response map is independently min–max normalized for visualization.

## 4.4.2 EFFECT OF NORMALIZATION AND ACTIVATION PLACEMENT

We examine two alternative explanations for the upsampling results: whether the transposedconvolution reference is disadvantaged by the absence of ReLU after normalization, and whether moving normalization after spatial rearrangement improves 3S. As shown in Table 5, adding ReLU to transposed convolution does not improve mIoU, suggesting that ReLU alone does not account for the gains of 3S. Applying GN after rearrangement gives similar mIoU to the original 3S, suggesting that normalization order has a limited effect on this metric in our experiments.

## 4.4.3 SUBVOXEL FEATURE ALLOCATION VARIANTS

As shown in Table 6, permute–reshape interprets the expanded channels as $r ^ { 3 } \times C _ { \mathrm { o u t } }$ , whereas 3S uses $C _ { \mathrm { o u t } } \times r ^ { 3 }$ . Its mIoU improvement from 12.16% to 12.32% supports channel expansion and rearrangement, while the 12.42% achieved by 3S validates the proposed feature allocation. Softmax improves mIoU and RegularIoU, but its sum-to-one competition reduces occupancy IoU. Sigmoid improves ComplexIoU and RegularIoU but lowers TailIoU, possibly because it favors dominant classes under imbalanced supervision. We therefore retain deterministic rearrangement for its bal anced performance. Both variants are defined in Appendix A.3.

Table 5: Effect of activation and normalization placement
<table><tr><td>Method</td><td>IoU</td><td>mIoU Complex Tail Regular</td><td></td><td></td><td></td></tr><tr><td>Transposed conv. + GN</td><td>36.1312.16</td><td></td><td>23.00</td><td>1.36</td><td>10.32</td></tr><tr><td>+ ReLU</td><td>36.17 12.10</td><td></td><td>22.95</td><td>1.34</td><td>10.18</td></tr><tr><td>Shuffle → GN → ReLU</td><td>36.2212.37</td><td></td><td>23.10</td><td>2.02</td><td>10.20</td></tr><tr><td>3S (GN → ReLU → shufle) 36.62 12.42</td><td></td><td></td><td>23.14</td><td>1.53</td><td>10.80</td></tr></table>

Table 6: Subvoxel feature allocation
<table><tr><td>Method</td><td>IoU</td><td></td><td>mIoU Complex Tail Regular</td><td></td><td></td></tr><tr><td>--</td><td></td><td>36.13 12.16</td><td>23.00</td><td>1.36</td><td>10.32</td></tr><tr><td>Permute-reshape</td><td></td><td>36.17 12.32</td><td>22.91</td><td>1.49</td><td>10.81</td></tr><tr><td>3S</td><td></td><td>36.6212.42</td><td>23.14</td><td>1.53</td><td>10.80</td></tr><tr><td>+ Softmax allocation 35.92 12.55</td><td></td><td></td><td>23.18</td><td>1.36</td><td>11.33</td></tr><tr><td>+ Sigmoid allocation 36.35 12.51</td><td></td><td></td><td>23.36</td><td>1.02</td><td>11.34</td></tr></table>

## 4.5 ANALYSIS OF ER-LM

As shown in Table 7, the results of EMA demonstrate the value of historical information, but simple exponential smoothing yields only limited semantic gains. ER-LM achieves a larger mIoU improvement through learnable memory and forgetting, with more balanced improvements across category groups. We further evaluate applying LayerNorm to the LSTM output before projection, with the corresponding ablation results provided in Appendix A.4.

Table 7: Analysis of historical information aggregation
<table><tr><td>Method</td><td>IoU</td><td>mIoU</td><td>Complex</td><td>Tail</td><td>Regular</td></tr><tr><td></td><td>35.78</td><td>12.16</td><td>22.84</td><td>1.15</td><td>11.33</td></tr><tr><td>EMA</td><td>36.19</td><td>12.37</td><td>22.95</td><td>1.63</td><td>10.77</td></tr><tr><td>ER-LM</td><td>36.09</td><td>12.53</td><td>22.96</td><td>1.32</td><td>11.57</td></tr></table>

Note. All configurations were re-run under the same setting, which differs slightly from Table 2.

We use mean class-wise degradation (mD) and correction (mR) rates to evaluate semantic transitions across decoder layers. As shown in Table 8, $\mathrm { E R - L M } + \mathcal { L } _ { \mathrm { c w e } }$ reduces degradation and improves correction in most adjacent-layer transitions and all intermediate-to-final comparisons. From the first to the final layer, mD decreases from 21.00% to 16.50%, while mR increases from 1.16% to 1.43%. Detailed definitions and evaluation protocols are provided in Appendix A.5.

Table 8: Class-balanced semantic transition rates (%). Lower mD and higher mR are better
<table><tr><td></td><td colspan="2">mD↓</td><td colspan="2">mR↑</td><td></td><td colspan="2"> $\mathrm { m D \downarrow }$ </td><td colspan="2">mR↑</td></tr><tr><td>Transition</td><td>--</td><td>Ours</td><td>--</td><td>Ours</td><td>Transition</td><td>--</td><td>Ours</td><td>--</td><td>Ours</td></tr><tr><td>1 → 2</td><td>17.86</td><td>11.60</td><td>0.99</td><td>0.88</td><td>1 → 9</td><td>21.00</td><td>16.50</td><td>1.16</td><td>1.43</td></tr><tr><td>2 → 3</td><td>8.60</td><td>8.84</td><td>0.64</td><td>0.99</td><td>2 → 9</td><td>15.85</td><td>12.12</td><td>0.84</td><td>1.50</td></tr><tr><td>3 → 4</td><td>11.06</td><td>6.34</td><td>0.57</td><td>0.72</td><td>3 → 9</td><td>13.01</td><td>9.08</td><td>0.73</td><td>1.33</td></tr><tr><td>4 → 5</td><td>10.10</td><td>6.45</td><td>0.55</td><td>0.99</td><td>4 → 9</td><td>12.56</td><td>9.92</td><td>0.77</td><td>1.39</td></tr><tr><td>5 → 6</td><td>9.77</td><td>5.11</td><td>0.51</td><td>0.73</td><td>5 → 9</td><td>11.40</td><td>5.95</td><td>0.71</td><td>1.03</td></tr><tr><td>6 → 7</td><td>5.89</td><td>4.84</td><td>0.60</td><td>0.87</td><td>6 → 9</td><td>9.29</td><td>5.50</td><td>0.65</td><td>0.95</td></tr><tr><td>7 → 8</td><td>13.46</td><td>8.50</td><td>0.45</td><td>0.92</td><td>7 → 9</td><td>14.22</td><td>8.39</td><td>0.56</td><td>0.97</td></tr><tr><td>8 → 9</td><td>4.31</td><td>4.08</td><td>0.38</td><td>0.55</td><td>8 → 9</td><td>4.31</td><td>4.08</td><td>0.38</td><td>0.55</td></tr></table>

Note. -- denotes the model without ER-LM or $\mathcal { L } _ { \mathrm { c w e } } ;$ Ours denotes ER- $\mathbf { \cdot L M } + \mathcal { L } _ { \mathrm { c w e } }$ . Bold indicates the better value within each layer pair.

## 5 CONCLUSION

We presented FM-Occ for sparse 3D semantic occupancy prediction, addressing feature homogenization during upsampling and semantic degradation across decoder layers. The proposed 3S constructs and jointly normalizes child features before spatial rearrangement, preserving greater variation in voxel feature responses. ER-LM retains historical query information through gated recurrent memory and entropy-guided state modulation, complemented by the class-weighted entropy loss $\mathcal { L } _ { \mathrm { c w e } } .$ . Experiments on SemanticKITTI demonstrate improved occupancy and semantic prediction, while the cross-layer analysis shows better preservation of correct semantic predictions and correction of existing errors on average across classes.

## REFERENCES

Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall. Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9297–9307, 2019.

Anh-Quan Cao and Raoul De Charette. Monoscene: Monocular 3d semantic scene completion. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3981– 3991. IEEE, 2022.

Xiyue Guo, Jiarui Hu, Junjie Hu, Hujun Bao, and Guofeng Zhang. Sgformer: Satellite-ground fusion for 3d semantic scene completion. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11929–11938. IEEE, 2025.

Jiawei Hou, Xiaoyan Li, Wenhao Guan, Gang Zhang, Di Feng, Yuheng Du, Xiangyang Xue, and Jian Pu. Fastocc: Accelerating 3d occupancy prediction by fusing the 2d bird’s-eye view and perspective view. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 16425–16431. IEEE, 2024.

Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9223–9232. IEEE, 2023.

Haoyi Jiang, Tianheng Cheng, Naiyu Gao, Haoyang Zhang, Tianwei Lin, Wenyu Liu, and Xinggang Wang. Symphonize 3d semantic scene completion with contextual instance queries. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20258–20267. IEEE, 2024.

Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 12689–12697. IEEE, 2019.

Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M Alvarez, Sanja Fidler, Chen Feng, and Anima Anandkumar. Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9087–9098. IEEE, 2023.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2020– 2036, 2024.

Enyu Liu, En Yu, Sijia Chen, and Wenbing Tao. Disentangling instance and scene contexts for 3d semantic scene completion. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 26999–27009. IEEE, 2025.

Wenze Liu, Hao Lu, Hongtao Fu, and Zhiguo Cao. Learning to upsample by learning to sample. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6004–6014. IEEE, 2023.

Gyeongrok Oh, Sungjune Kim, Heeju Ko, Hyung-gun Chi, Jinkyu Kim, Dongwook Lee, Daehyun Ji, Sungjoon Choi, Sujin Jang, and Sangpil Kim. 3d occupancy prediction with low-resolution queries via prototype-aware view transformation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17134–17144. IEEE, 2025.

Wenzhe Shi, Jose Caballero, Ferenc Huszar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel´ Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1874–1883, 2016.

Pin Tang, Zhongdao Wang, Guoqing Wang, Jilai Zheng, Xiangxuan Ren, Bailan Feng, and Chao Ma. Sparseocc: Rethinking sparse latent representation for vision-based semantic occupancy prediction. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15035–15044. IEEE, 2024.

Jiabao Wang, Zhaojiang Liu, Qiang Meng, Liujiang Yan, Ke Wang, Jie Yang, Wei Liu, Qibin Hou, and Ming-Ming Cheng. Opus: occupancy prediction using a sparse set. Advances in Neural Information Processing Systems, 37:119861–119885, 2024a.

Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, and Dahua Lin. Carafe: Contentaware reassembly of features. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3007–3016. IEEE, 2019.

Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen Lu, and Xingang Wang. Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17804–17813. IEEE, 2023.

Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, and Zhaoxiang Zhang. Panoocc: Unified occupancy representation for camera-based 3d panoptic segmentation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17158–17168. IEEE, 2024b.

Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multicamera 3d occupancy prediction for autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21672–21683. IEEE, 2023.

Yujie Xue, Ruihui Li, Fan Wu, Zhuo Tang, Kenli Li, and Mingxing Duan. Bi-ssc: Geometricsemantic bidirectional fusion for camera-based 3d semantic scene completion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20124–20134. IEEE, 2024.

Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.

Zhiwen Yang and Yuxin Peng. Multi-resolution alignment for voxel sparsity in camera-based 3d semantic scene completion. IEEE Transactions on Image Processing, 35:1771–1785, 2026.

Matthew D Zeiler, Dilip Krishnan, Graham W Taylor, and Rob Fergus. Deconvolutional networks. In 2010 IEEE Computer Society Conference on computer vision and pattern recognition, pp. 2528–2535. IEEE, 2010.

Yunpeng Zhang, Zheng Zhu, and Dalong Du. Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9399–9409. IEEE, 2023.

Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Pointocc: Cylindrical tri-perspective view for point-based 3d semantic occupancy prediction. arXiv preprint arXiv:2308.16896, 2023.

## A APPENDIX

## A.1 DEFINITION OF CATEGORY-GROUPED EVALUATION METRICS

These metrics are computed as the average IoU over predefined category groups, enabling a more fine-grained evaluation beyond overall performance. Formally, given a category group ${ \mathcal { G } } ,$ the groupwise IoU is defined as:

$$
\mathrm { I o U } _ { \mathcal { G } } = \frac { 1 } { | \mathcal { G } | } \sum _ { c \in \mathcal { G } } \mathrm { I o U } _ { c } ,\tag{15}
$$

where IoU<sub>c</sub> denotes the IoU of category c.

We divide the 19 non-empty semantic categories into three groups based on structural complexity, boundary ambiguity, and sample imbalance, aiming to enable a more reasonable evaluation and a more accurate analysis of the training results.

ComplexIoU measures performance on categories that typically occupy extended spatial regions, exhibit irregular structures, or have ambiguous semantic boundaries. This group includes road (15.30%), sidewalk (11.13%), parking (1.12%), other-ground (0.56%), terrain (9.17%), vegetation (39.30%), and trunk (0.51%), with the percentages indicating the proportion of each category in the dataset.

TailIoU focuses on long-tail categories with limited samples (all below 0.1%), where reliable feature learning is challenging. This group includes bicyclist (0.07%), person (0.07%), motorcyclist (0.05%), bicycle (0.03%), motorcycle (0.03%), and traffic-sign (0.08%).

RegularIoU evaluates categories with relatively regular geometric structures and well-defined shapes. This group includes car (3.92%), building (14.10%), truck (0.16%), fence (3.90%), pole (0.29%), and other-vehicle (0.20%).

## A.2 ADDITIONAL ANALYSIS OF UPSAMPLING FEATURE RESPONSES

## A.2.1 DEFINITION OF THE FEATURE-RESPONSE VARIATION METRIC

The response measure and spatial CV reported in the main paper are defined as follows. For a frame containing N active voxels, let $\mathbf { x } _ { n } \in \mathbb { R } ^ { C }$ denote the feature vector of the n-th active voxel, where C is the number of feature channels. Its root-mean-square (RMS) response is defined as

$$
R _ { n } = R ( \mathbf { x } _ { n } ) = \sqrt { \frac { 1 } { C } \sum _ { j = 1 } ^ { C } x _ { n , j } ^ { 2 } } ,\tag{16}
$$

where $j$ indexes the feature channels.

Let $\mu _ { \smash { \frac { R } { \min } } }$ and $\sigma _ { R }$ denote the mean and standard deviation, respectively, of the voxel responses $\{ R _ { n } \} _ { n = 1 } ^ { \bar { N } }$ within the frame. The spatial coefficient of variation is then defined as

$$
\mathrm { C V } = \frac { \sigma _ { R } } { \mu _ { R } } \times 1 0 0 \% .\tag{17}
$$

The measurements are taken after group normalization at the second sparse upsampling stage over 50 uniformly sampled SemanticKITTI validation frames, and the reported CV is averaged over these frames. The ratio in Table 4 is computed as the mean CV of each method divided by that of sparse transposed convolution.

## A.2.2 FEATURE-RESPONSE VISUALIZATION SETUP AND ANALYSIS

Fig. 4 compares the second-stage upsampling outputs before feature fusion. Both outputs use the same grid resolution and projected sample locations. Region A covers a car and its surrounding area, while regions B and C correspond to vegetation and ground areas, respectively. Compared with sparse transposed convolution, 3S exhibits clearer local variation in feature-response magnitudes within these regions. Each response map is independently min–max normalized; the visualization therefore compares relative spatial patterns rather than absolute response magnitudes or semantic prediction accuracy.

## A.3 DEFINITIONS OF SOFTMAX- AND SIGMOID-BASED FEATURE ALLOCATION

Let $\mathbf { H } _ { n } \in \mathbb { R } ^ { r ^ { 3 } \times C _ { \mathrm { o u t } } }$ denote the candidate child representations of the n-th parent voxel. We apply mean pooling along the child dimension and feed the resulting feature into a two-layer MLP. The MLP produces $\mathbf { a } _ { n } \in \mathbb { R } ^ { r ^ { 3 } }$ , where $a _ { n , q }$ is the allocation score predicted for the q-th candidate child voxel. We convert these scores into weights using either Softmax or Sigmoid.

The first variant applies Softmax across the $r ^ { 3 }$ child positions:

$$
g _ { n , q } ^ { \mathrm { s o f t m a x } } = \frac { \exp ( a _ { n , q } ) } { \sum _ { j = 1 } ^ { r ^ { 3 } } \exp ( a _ { n , j } ) } , \qquad q = 1 , \ldots , r ^ { 3 } .\tag{18}
$$

The resulting weights sum to one, introducing competition among the candidate child positions.

The second variant applies Sigmoid independently to each allocation score:

$$
g _ { n , q } ^ { \mathrm { s i g m o i d } } = \frac { 1 } { 1 + \exp ( - a _ { n , q } ) } , \qquad q = 1 , \ldots , r ^ { 3 } .\tag{19}
$$

Unlike Softmax, Sigmoid does not impose a sum-to-one constraint, allowing multiple child positions to retain relatively large responses simultaneously. In both variants, each weight $g _ { n , q }$ is applied to its corresponding candidate representation ${ \bf { H } } _ { n , q , : }$

## A.4 EFFECT OF ADDITIONAL LAYERNORM IN ER-LM

As shown in Table 9, without $\mathcal { L } _ { \mathrm { c w e } } ,$ , adding LayerNorm to the LSTM output before projection improves occupancy IoU but reduces semantic mIoU. This may reflect a trade-off: normalizing memory-feature magnitudes can help distinguish occupied from empty regions while weakening magnitude cues useful for distinguishing semantic classes. With $\mathcal { L } _ { \mathrm { c w e } } .$ , omitting this additional LayerNorm yields higher mIoU, TailIoU, and RegularIoU, at a small cost in ComplexIoU. We therefore omit this additional LayerNorm in our final configuration.

Table 9: Effect of additional LayerNorm in ER-LM. Bold numbers indicate the best result in each column.
<table><tr><td>Method</td><td>IoU</td><td>mIoU</td><td>ComplexIoU</td><td>TailIoU</td><td>RegularIoU</td></tr><tr><td>ER-LM</td><td>36.09</td><td>12.53</td><td>22.96</td><td>1.32</td><td>11.57</td></tr><tr><td>ER-LM + LayerNorm</td><td>36.53</td><td>12.45</td><td>23.05</td><td>1.26</td><td>11.26</td></tr><tr><td>ER-LM + LayerNorm +  $\mathcal { L } _ { \mathrm { c w e } }$ </td><td>36.37</td><td>12.47</td><td>23.28</td><td>1.51</td><td>10.82</td></tr><tr><td>ER-LM + Lcwe</td><td>36.45</td><td>12.82</td><td>23.23</td><td>1.64</td><td>11.84</td></tr></table>

Note. All configurations were re-run under the same setting, which differs slightly from Table 2.

## A.5 EVALUATION PROTOCOL AND SEMANTIC TRANSITION METRICS

Both models are evaluated on the same 815 SemanticKITTI validation frames using all nine decoder outputs. Predictions are evaluated at the ground-truth voxel resolution; invalid labels are excluded, and repeated samples introduced by distributed sampling are counted only once.

For a source layer a and a target layer $b > a$ , we compare semantic predictions at the same voxel locations. For each non-empty ground-truth class c, let $N _ { c } ^ { \mathrm { C C } } , N _ { c } ^ { \mathrm { C W } } , \dot { N } _ { c } ^ { \mathrm { W C } }$ , and $N _ { c } ^ { \mathrm { W W } }$ count correctto-correct, correct-to-wrong, wrong-to-correct, and wrong-to-wrong transitions, respectively. For brevity, we omit the layer-pair indices $a  b$ from these counts. Correctness requires an exact match to the ground-truth semantic label. Counts are pooled over the validation set before computing the class-wise degradation and correction rates:

$$
D _ { c } ^ { a \to b } = \frac { N _ { c } ^ { \mathrm { C W } } } { N _ { c } ^ { \mathrm { C C } } + N _ { c } ^ { \mathrm { C W } } } ,\tag{20}
$$

$$
R _ { c } ^ { a  b } = \frac { N _ { c } ^ { \mathrm { W C } } } { N _ { c } ^ { \mathrm { W C } } + N _ { c } ^ { \mathrm { W W } } } .\tag{21}
$$

Degradation measures the fraction of initially correct predictions that become incorrect, whereas correction measures the fraction of initially incorrect predictions that become correct. To give each category equal weight, we define

$$
\mathrm { m D } ^ { a  b } = \frac { 1 0 0 } { | { \mathcal C } _ { D } ^ { a  b } | } \sum _ { c \in { \mathcal C } _ { D } ^ { a  b } } D _ { c } ^ { a  b } ,\tag{22}
$$

$$
\mathrm { m R } ^ { a  b } = \frac { 1 0 0 } { \vert { \mathcal C } _ { R } ^ { a  b } \vert } \sum _ { c \in { \mathcal C } _ { R } ^ { a  b } } R _ { c } ^ { a  b } .\tag{23}
$$

For each metric and layer pair, the class set contains only non-empty categories with a non-zero denominator in both models. Undefined rates are excluded rather than set to zero. Thus, each row compares the models over identical eligible classes, although the mD class set can vary across rows. Lower mD and higher mR indicate more favorable semantic preservation and correction, respectively.