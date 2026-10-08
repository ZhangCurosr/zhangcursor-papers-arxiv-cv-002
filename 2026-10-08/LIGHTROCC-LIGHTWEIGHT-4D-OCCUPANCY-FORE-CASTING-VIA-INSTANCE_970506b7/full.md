# LIGHTROCC: LIGHTWEIGHT 4D OCCUPANCY FORE-CASTING VIA INSTANCE-CENTRIC 3D GAUSSIANS

Hwanhee Jung<sup>1</sup>, SeungHyeon Kim<sup>1</sup>, Inkyu Koo<sup>1</sup>, Qixing Huang<sup>2</sup>, Sang Ho Yoon<sup>3</sup>, Sangpil Kim<sup>1</sup>   
<sup>1</sup>Korea University <sup>2</sup>University of Texas at Austin   
<sup>3</sup>Korea Advanced Institute of Science & Technology

## ABSTRACT

Forecasting future 3D occupancy from surround-view cameras is essential for autonomous driving, yet existing approaches rely on dense voxel or bird’s-eye-view representations whose cost grows rapidly with spatial resolution and prediction horizon. Because these representations do not explicitly maintain object identities, they also struggle to preserve instance consistency over time. We present LighTROcc, a lightweight instance-centric framework that represents movable objects with a compact set of learned queries and predicts present and future occupancy in a single forward pass. LighTROcc localizes each query through attention-guided forward lifting, combining image-space cross-attention, queryspecific depth, and camera geometry to estimate its 3D center. Each instance is modeled as a mixture of anisotropic 3D Gaussians and propagated across future steps using predicted displacements, producing continuous, temporally consistent occupancy forecasts. Experiments on nuScenes and supplemented nuScenes-Occupancy show that LighTROcc outperforms the evaluated dense and instancewise baselines in instance-level forecasting accuracy while maintaining strong voxel-level occupancy quality. Across different model configurations, LighTROcc achieves a favorable balance between forecasting accuracy and computational efficiency, demonstrating the potential of compact instance-centric modeling for camera-based 4D occupancy forecasting.

![](images/5b3a5cd663dd87ff2245600e7fd3b841efc85316f78d13a4b6393b67cb76c461.jpg)

![](images/d3df24d47205f7473899584f265286fa341ef2957cc8005cbaba530637241fdd.jpg)  
Figure 1: Accuracy-efficiency comparison with current-only and 4D forecasting methods. LighTROcc achieves a favorable trade-off between inference speed and both IoU and AP while predicting current and future occupancy in a single forward pass.

## 1 INTRODUCTION

Safe and real-time autonomous driving requires an accurate and efficient understanding of which regions of the surrounding space are free or occupied. Extensive research has investigated fine-grained camera-based occupancy perception to provide a comprehensive representation of the current driving environment (Huang et al., 2023; Wei et al., 2023; Zhang et al., 2023; Wang et al., 2023b; Tong et al., 2023; Ma et al., 2024b). From the perspective of planning and control, however, estimating only the present state is insufficient. An autonomous vehicle must also anticipate whether currently free space will remain traversable or become occupied when the vehicle reaches it. This need has motivated growing interest in 4D occupancy forecasting, which predicts a sequence of future 3D occupancy states from past observations. Since temporal occupancy changes are primarily driven by movable objects, these forecasting methods commonly focus on general movable objects (GMOs).

Recent 4D occupancy forecasting methods typically model temporal evolution by extending spatially indexed representations, such as voxels or bird’s-eye-view (BEV) cells, along the temporal dimension (Ma et al., 2024a; Xu et al., 2025; Chen et al., 2025). Many of these methods rely on volumetric processing, such as 3D convolutions, to reconstruct and forecast occupancy. Although such representations retain an explicit volumetric description of the scene, they require substantial computation, and their cost increases rapidly with spatial resolution and forecasting horizon (Chen et al., 2025). Consequently, forecasting multiple future frames while satisfying real-time inference requirements remains challenging. A recent approach factorizes 3D occupancy into a 2D BEV representation and column-wise height information to avoid explicit full-volume processing (Xu et al., 2025). While this lower-dimensional formulation can alleviate the burden of volumetric processing, its column-based reconstruction may inadequately capture complex object geometry and fine-grained volumetric extents. More broadly, existing 4D occupancy forecasting methods are generally not designed in an instance-centric manner. When object identities are required, instances must instead be recovered through additional association, grouping, or flow-based processing. Such indirect recovery can lead to temporal inconsistencies, including the merging of distinct objects or the fragmentation of a single object across frames.

Maintaining consistent object identities over time therefore calls for an instance-centric formulation that directly recognizes and forecasts individual objects. Existing approaches have explored instance query candidates, where each query is intended to represent an object (Wang et al., 2022; Liu et al., 2022; 2024; Hu et al., 2026). In many such methods, queries are initialized from spatial anchor points, and a large candidate set is required to provide sufficient coverage of the 3D scene. Because this set is predefined to be substantially larger than the actual number of object instances, it can produce redundant predictions and incur unnecessary computational overhead. Furthermore, these queries are generally designed to represent instances in the current scene rather than to persist as object-specific states throughout future frames. These observations motivate a lightweight instance-centric framework that compactly represents the current scene and directly propagates object instances across multiple future time steps.

Building on this principle, we introduce LighTROcc, a lightweight instance-centric transformer for camera-based 4D occupancy forecasting. Rather than distributing a large set of spatially initialized candidate queries throughout the 3D scene, LighTROcc directly localizes a compact set of queries in 3D through attention-guided forward lifting. Specifically, each query derives its imagespace location from the attention map and combines it with a query-specific depth prediction and camera geometry. The localized query further predicts a mixture of 3D Gaussians that compactly represents the full volumetric extent of the corresponding instance. The resulting instance representation is then propagated to future frames using predicted displacements, maintaining object-level consistency over time. This compact formulation enables LighTROcc to forecast multiple future occupancy states in a single forward pass, yielding a favorable balance between forecasting accuracy and inference speed, as shown in Figure 1. We evaluate our method on nuScenes (Caesar et al., 2020) and supplemented annotations from nuScenes-Occupancy (Wang et al., 2023b), where it outperforms the evaluated baselines, demonstrating the effectiveness of the proposed formulation.

Our contributions are summarized as follows:

• We propose LighTROcc, a lightweight instance-centric network that forecasts future 4D occupancy states in a single forward pass.

• We introduce an attention-guided forward lifting formulation that directly localizes 3D instance centers from attention map and depth, enabling compact instance prediction.

• Extensive experiments on nuScenes and supplemented nuScenes-Occupancy demonstrate the highest accuracy with lower computational cost and faster inference.

## 2 RELATED WORK

## 2.1 3D OCCUPANCY REPRESENTATIONS AND QUERY-BASED PERCEPTION

Occupancy representations. 3D occupancy prediction has been explored using dense voxel grids, reduced-dimensional representations, and continuous 3D primitives. Early semantic scene completion and camera-based methods construct dense voxel volumes to preserve full 3D structure, but their computational cost scales with spatial resolution (Song et al., 2017; Roldao et al., 2020; Cao &˜ de Charette, 2022; Wei et al., 2023; Li et al., 2023b). To reduce this burden, FlashOcc compresses volumetric occupancy by mapping BEV features to height-aware outputs, while PointOcc factorizes 3D space into cylindrical tri-perspective representations (Yu et al., 2023; Zuo et al., 2023). More recently, Gaussian-based approaches provide continuous alternatives to voxel grids using sparse 3D Gaussians or Gaussian splatting representations for occupancy prediction (Huang et al., 2024; 2025; Gan et al., 2025; Jiang et al., 2025).

Geometry-indexed query methods. A major query-based line assigns queries to predefined geometric elements such as BEV cells, planes, or voxels. BEVFormer learns dense BEV queries, TPV-Former introduces three orthogonal query planes, and VoxFormer and PanoOcc adopt voxel-indexed representations for occupancy prediction (Li et al., 2022; Huang et al., 2023; Li et al., 2023a; Wang et al., 2024b). Subsequent methods improve efficiency through perspective supervision, polar layouts, dual-path decoding, octree queries, and compact query representations (Yang et al., 2023; Jiang et al., 2023; Zhang et al., 2023; Lu et al., 2024; Ma et al., 2024b; Oh et al., 2025). However, spatially indexed queries remain coupled to the predefined discretization, causing computational and memory costs to grow with spatial resolution.

Object-centric query methods. DETR3D associates queries with 3D reference points projected into camera views, while PETR and PETRv2 incorporate 3D positional information into image features (Wang et al., 2022; Liu et al., 2022; 2023b). Subsequent methods extend sparse queries to temporal modeling, tracking, sensor fusion, and end-to-end driving (Lin et al., 2022; 2023a;b; Liu et al., 2023a; Wang et al., 2023a; Xiong et al., 2023; Zhang et al., 2022; Chen et al., 2023; Sun et al., 2025). Nevertheless, several approaches still rely on reference points or anchor-based initialization, requiring large candidate sets for scene coverage (Wang et al., 2022; Lin et al., 2022). Sparse query formulations have also been extended to occupancy prediction through sparse points, sets, contextual queries, and Gaussian representations (Tang et al., 2024; Liu et al., 2024; Wang et al., 2024a; Shi et al., 2024; Jiang et al., 2024; Hu et al., 2026; Park et al., 2026). S2GO uses spatially anchored 3D queries with Gaussian decoding for streaming occupancy, whereas LighTROcc adopts compact object-centric queries with attention-guided localization for multi-step occupancy forecasting.

## 2.2 CAMERA-ONLY 4D OCCUPANCY FORECASTING

Camera-only 4D occupancy forecasting predicts present and future 3D occupancy states from historical multi-view images. Cam4DOcc introduced a benchmark and OCFNet, an end-to-end baseline that processes multi-frame 3D voxel features (Ma et al., 2024a). OccProphet improves efficiency with a lightweight Observer-Forecaster-Refiner framework that aggregates multi-frame voxel features (Chen et al., 2025), while EfficientOCF decouples spatial and temporal processing through BEV occupancy, height, and flow prediction (Xu et al., 2025). Despite these advances, their forecasts remain spatially indexed rather than maintaining objects as persistent instance states. LighTROcc instead represents each object with an instance-specific query and forecasts its occupancy across future frames, preserving object-level consistency throughout the forecasting horizon.

## 3 METHOD

## 3.1 PROBLEM FORMULATION

Let $\pmb { I } _ { t } ^ { v }$ denote the RGB image captured by camera $v \in \{ 1 , \ldots , V \}$ at time $t \in \{ - T _ { p } , \dots , 0 \}$ where $T _ { p }$ is the number of past frames preceding the present and $t = 0$ denotes the present frame. Following prior camera-only occupancy forecasting methods (Ma et al., 2024a; Xu et al., 2025; Chen et al., 2025), we focus on the occupancy evolution of general movable objects (GMOs). Given the multi-view image history and the corresponding camera intrinsics, extrinsics, and ego poses, the task is to predict the present and future voxel occupancy states:

$$
\{ \hat { O } _ { k } \} _ { k = 0 } ^ { T _ { f } } , \qquad \hat { O } _ { k } \in [ 0 , 1 ] ^ { X \times Y \times Z } .\tag{1}
$$

Here, $\hat { O } _ { k } ( x , y , z )$ denotes the probability that voxel $( x , y , z )$ is occupied at time $k ,$ where $k = 0$ corresponds to the present and $k > 0$ to the future. We set $T _ { p } = 2$ and forecast $T _ { f } = 4$ frames.

Unlike conventional dense approaches that construct voxel-wise features over a discretized 3D grid, LighTROcc represents movable objects with a compact set of instance queries. Each query predicts confidence, depth, Gaussian mixture parameters, and a future trajectory. The resulting continuous Gaussian volumes are rasterized into the occupancy sequence $\{ \hat { O } _ { k } \} _ { k = 0 } ^ { T _ { f } } .$

![](images/169c5ed06148e34a514aa743b8759cb6a07492e00dd763c19827f42baf63659f.jpg)  
Figure 2: Overview of LighTROcc. Compact instance queries are sequentially refined through a transformer decoder using multi-view image features. Each query is localized in 3D via attentionguided forward lifting, while heads predict 3D Gaussians, confidence, and future trajectory.

## 3.2 TRANSFORMER-BASED QUERY DECODER

As illustrated in Fig. 2, a shared image encoder extracts multi-scale context features ${ \cal F } _ { t , \ell } ^ { v } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { H _ { \ell } \times W _ { \ell } \times D }$ from each surrounding image $I _ { t } ^ { v }$ . Since the shared encoder processes each image independently, the resulting visual features do not explicitly encode their spatial locations, camera viewpoints, or acquisition times. We therefore augment each image token with a 2D sinusoidal positional encoding, a camera embedding derived from the corresponding camera parameters and ego pose, a learned camera-ID embedding, and a temporal embedding representing the observation time. For decoder layer $\ell ,$ the augmented tokens from all cameras are flattened to form the multi-view context features $F _ { t , \ell }$

We initialize the decoder with $N _ { q }$ learned instance queries $Q \in \mathbb { R } ^ { N _ { q } \times D }$ . For each observed frame, the queries sequentially interact with the multi-view features through cross-attention and are refined by self-attention and a feed-forward network. At decoder layer ℓ, the update is formulated as

$$
\begin{array} { r l } & { \bar { Q } _ { t } ^ { \ell } = \mathrm { L N } \big ( Q _ { t } ^ { \ell - 1 } + \mathrm { C A } _ { \ell } ( Q _ { t } ^ { \ell - 1 } , F _ { t , \ell } ) \big ) , } \\ & { \tilde { Q } _ { t } ^ { \ell } = \mathrm { L N } \big ( \bar { Q } _ { t } ^ { \ell } + \mathrm { S A } _ { \ell } ( \bar { Q } _ { t } ^ { \ell } ) \big ) , } \\ & { Q _ { t } ^ { \ell } = \mathrm { L N } \big ( \tilde { Q } _ { t } ^ { \ell } + \mathrm { F F N } _ { \ell } ( \tilde { Q } _ { t } ^ { \ell } ) \big ) . } \end{array}\tag{2}
$$

Here, LN denotes layer normalization, while $\mathrm { C A } _ { \ell } , \mathrm { S A } _ { \ell } ,$ and FFN denote the cross-attention, selfattention, and feed-forward modules at layer ℓ, respectively. The observation frames are processed chronologically, with the final query state of each frame initializing the next. This sequential propagation serves as a compact temporal memory without constructing dense spatiotemporal features.

For efficient coarse-to-fine decoding, successive layers attend to image features of increasing spatial resolution. Early layers capture broad context at low computational cost, while the final layer retains the spatial detail required for 3D lifting. We use three decoder layers with four attention heads.

## 3.3 ATTENTION-GUIDED FORWARD LIFTING

After query decoding, each query contains object-specific 2D visual evidence, but directly regressing its metric 3D location from the latent query vector remains non-trivial. We therefore decompose 3D localization into two simpler predictions: image-plane localization and query-specific depth estimation. The known camera parameters and ego pose then serve as a geometric bridge that lifts these predictions into 3D. Specifically, as illustrated in Figure 3, we reuse the query-to-image attention as an imageplane localization cue and combine it with the predicted depth to recover a 3D center.

![](images/a08ed8d4149f193edfce1fea9a9e0a7249206fae59b99e2ab965eeca03e40397.jpg)  
Figure 3: Illustration of attention-guided forward lifting, combining query attention and predicted depth to recover each instance center in 3D.

Let $\pmb { A } _ { t , q } ^ { v } \in [ 0 , 1 ] ^ { H \times W }$ denote the final-layer cross-attention map of query q over camera v at time t. We spatially normalize the attention map within each view and obtain a differentiable image-plane center through soft-argmax:

$$
\pmb { p } _ { t , q } ^ { v } = \sum _ { \pmb { u } \in \Omega } \bar { \pmb { A } } _ { t , q } ^ { v } ( \pmb { u } ) \pmb { u } ,\tag{3}
$$

where Ω denotes the spatial domain of the feature map, $\pmb { u } = [ u , v ] ^ { \top }$ is a feature-map coordinate, and $A _ { t , q } ^ { v }$ is the normalized attention map. To preserve gradients and stabilize image-plane localization, we use soft-argmax rather than discrete argmax. In parallel, a lightweight depth head predicts a categorical distribution $\pi _ { t , q } \in [ 0 , 1 ] ^ { N _ { d } }$ over $\bar { N _ { d } }$ predefined depth bins. Given a depth range $[ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$ the center of the b-th bin is defined as

$$
\rho _ { b } = d _ { \operatorname* { m i n } } + \left( b - { \frac { 1 } { 2 } } \right) { \frac { d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } } { N _ { d } } } , \qquad b = 1 , \dots , N _ { d } ,\tag{4}
$$

and the continuous query depth is obtained as the expectation over the fixed bin centers:

$$
d _ { t , q } = \sum _ { b = 1 } ^ { N _ { d } } \pi _ { t , q , b } \rho _ { b } .\tag{5}
$$

The attention map determines the image region associated with the query, while the depth distribution resolves its position along the corresponding camera ray. For each query, we select the valid camera with the largest total mass in the original cross-attention map and map its location back to the original image coordinates. The resulting image-plane location is then lifted into 3D using the predicted depth and the corresponding camera geometry:

$$
{ \pmb { c } } _ { t , q } = \mathrm { L i f t } \left( { \pmb { p } } _ { t , q } , d _ { t , q } ; { \pmb { K } } _ { t } , { \pmb { E } } _ { t } \right) ,\tag{6}
$$

where $\pmb { K } _ { t }$ and $\scriptstyle { E _ { t } }$ denote the intrinsic and extrinsic parameters of the selected camera, respectively. The $\operatorname { L i f t } ( \cdot )$ operation maps the image-plane location to 3D along its camera ray using the predicted depth and camera geometry. This attention-guided forward lifting provides each query with a consistent 3D center without introducing spatially initialized 3D candidates.

## 3.4 OCCUPANCY FORECASTING

To capture finer object geometry than the coarse occupancy volumes derived from inflated 3D boxes, we adopt an instance-level Gaussian representation similar to GUIDE (Hu et al., 2026). Given the lifted 3D center $c _ { q }$ and decoded feature of each query, we represent its volumetric extent using a compact mixture of $N _ { g }$ anisotropic 3D Gaussians. For each component, lightweight prediction heads estimate a center offset $\Delta \mu _ { q , g } ,$ , axis-wise standard deviations $\sigma _ { q , g } ,$ a rotation matrix $R _ { q , g } .$ and an occupancy weight $\alpha _ { q , g }$ . Its mean and covariance are defined as

$$
\pmb { \mu } _ { q , g } = \pmb { c } _ { q } + \Delta \pmb { \mu } _ { q , g } , \qquad \pmb { \Sigma } _ { q , g } = \pmb { R } _ { q , g } \mathrm { d i a g } \left( \pmb { \sigma } _ { q , g } ^ { 2 } \right) \pmb { R } _ { q , g } ^ { \top } .\tag{7}
$$

The resulting Gaussian components jointly describe the object geometry as a continuous occupancy field that can be discretized at different voxel resolutions. Unlike voxel-grid representations tied to a predefined discretization, our continuous Gaussian representation can be discretized at different resolutions during inference without retraining or modifying the network architecture.

To forecast future occupancy, a lightweight trajectory head predicts a sequence of displacements for each query. These displacements are accumulated from the present 3D center to determine the query location at each future step. The associated Gaussian mixture is then translated along the predicted trajectory while retaining its estimated offsets, covariance, and occupancy weights. This enables the model to generate multiple future occupancy states in a single forward pass.

## 3.5 OPTIMIZATION

In this section, we describe the ground-truth assignment and training objectives used to optimize the instance-wise queries. Since the learned queries are permutation-invariant, we establish a bipartite correspondence between the predicted queries and ground-truth instances using Hungarian matching (Kuhn, 1955). The matching cost combines the 3D center distance over the observed frames, the spatial overlap between the query attention map and the projected ground-truth instance region, and the predicted instance confidence. Geometric, occupancy, and trajectory supervision is applied only to matched pairs, while unmatched queries are treated as background for confidence supervision.

Table 1: Quantitative comparison of dense voxel-wise and instance-wise methods for GMO occupancy forecasting. LighTROcc<sup>†</sup> denotes the resolution-matched variant using $7 0 4 \times 2 5 6$ inputs. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Type</td><td>Input Res.</td><td> $\mathrm { I o U } _ { \mathrm { c u r } }$ </td><td> $\mathrm { I o U } _ { \mathrm { f u t } }$ </td><td> $\mathrm { I o U _ { a v g } }$ </td><td> $\mathrm { { A P _ { c u r } } }$ </td><td> $\mathrm { A P _ { f u t } }$ </td><td> $\mathsf { A P _ { a v g } }$ </td><td>FPS</td></tr><tr><td>Cam4DOcc (Ma et al., 2024a)</td><td>Dense</td><td> $1 6 0 0 \times 8 9 6$ </td><td>13.42</td><td>11.80</td><td>12.12</td><td>10.71</td><td>8.02</td><td>8.56</td><td>0.52</td></tr><tr><td>EfficientOCF (Xu et al., 2025) OccProphet (Chen et al., 2025)</td><td>Voxel wise</td><td> $1 6 0 0 \times 8 9 6$ </td><td>8.67</td><td>8.59</td><td>8.61</td><td>14.25</td><td>14.13</td><td>14.15</td><td>3.40</td></tr><tr><td></td><td></td><td> $8 0 0 \times 4 4 8$ </td><td>17.57</td><td>14.23</td><td>14.89</td><td>20.14</td><td>7.25</td><td>9.83</td><td>3.27</td></tr><tr><td>SparseOcc (Liu et al., 2024)</td><td rowspan="4">Instance wise</td><td> $7 0 4 \times 2 5 6$ </td><td>18.42</td><td>13.67</td><td>14.62</td><td>20.05</td><td>13.36</td><td>14.70</td><td>14.18</td></tr><tr><td>GUIDE (Hu et al., 2026)</td><td> $7 0 4 \times 2 5 6$ </td><td>15.21</td><td>12.40</td><td>12.96</td><td>34.73</td><td>24.51</td><td>26.55</td><td>8.16</td></tr><tr><td>LighTROcc† (Ours)</td><td> $7 0 4 \times 2 5 6$ </td><td>18.46</td><td>14.34</td><td>15.16</td><td>40.03</td><td>31.30</td><td>33.05</td><td>25.10</td></tr><tr><td>LighTROcc (Ours)</td><td> $1 6 0 0 \times 8 9 6$ </td><td>19.76</td><td>15.30</td><td>16.19</td><td>42.62</td><td>33.34</td><td>35.20</td><td>12.80</td></tr></table>

The confidence head predicts a foreground probability $s _ { q } ~ \in ~ [ 0 , 1 ]$ for each query. Based on the matching result, matched and unmatched queries are assigned foreground and background targets, respectively. The confidence loss ${ \mathcal { L } } _ { \mathrm { c o n f } }$ is then computed using cross-entropy between the predicted probabilities and the corresponding binary targets. At inference, only queries with $s _ { q } > \tau _ { \mathrm { c o n f } }$ are retained for occupancy prediction, where $\tau _ { \mathrm { c o n f } } = 0 . 9$

For each matched pair, we compute the lifted 3D center loss ${ \mathcal { L } } _ { \mathrm { c e n } }$ using the $\ell _ { 1 }$ norm and supervise the predicted depth distribution with the cross-entropy loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ . To encourage accurate attentionbased 2D localization, the attention localization loss $\mathcal { L } _ { \mathrm { a t t n } }$ is defined as the negative log-likelihood of the normalized attention mass assigned to the projected ground-truth region. To supervise object geometry, the predicted Gaussian mixture is voxelized and optimized against the ground-truth occupancy using focal and Tversky losses (Lin et al., 2017; Salehi et al., 2017), denoted by $\mathcal { L } _ { \mathrm { o c c } }$ . Finally, the trajectory head is supervised with an $\ell _ { 1 }$ displacement loss $\mathcal { L } _ { \mathrm { t r a j } }$ over the valid future frames. The overall training objective is

$$
{ \mathcal { L } } = \lambda _ { \mathrm { c o n f } } { \mathcal { L } } _ { \mathrm { c o n f } } + \lambda _ { \mathrm { c e n } } { \mathcal { L } } _ { \mathrm { c e n } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { a t t n } } { \mathcal { L } } _ { \mathrm { a t t n } } + \lambda _ { \mathrm { o c c } } { \mathcal { L } } _ { \mathrm { o c c } } + \lambda _ { \mathrm { t r a j } } { \mathcal { L } } _ { \mathrm { t r a j } } ,\tag{8}
$$

where each λ controls the contribution of the corresponding loss term.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. We train and evaluate LighTROcc and the baseline models using nuScenes (Caesar et al., 2020) and nuScenes-Occupancy (Wang et al., 2023b). We obtain multi-view RGB images, camera parameters, and ego poses from nuScenes, and use the fine-grained voxel occupancy annotations from nuScenes-Occupancy for supervision. Following the prior work (Ma et al., 2024a; Xu et al., 2025; Chen et al., 2025), we use 700 scenes for training and 150 for evaluation, yielding 23,930 and 5,119 sequences, respectively. Each sequence contains two past observations, one present observation, and four future occupancy targets. Since the nuScenes keyframes are sampled at 2 Hz, these targets correspond to prediction horizons of 0.5, 1.0, 1.5, and 2.0 seconds. The evaluation range is [−51.2, 51.2] m along both horizontal axes and [−5, 3] m vertically, discretized into a $[ 5 1 2 \times 5 1 2 \times 4 0 ]$ grid with a voxel size of 0.2 m. Since the LiDAR-derived nuScenes-Occupancy annotations are highly sparse, we supplement them with auxiliary assets. Further details are provided in the Appendix.

Evaluation protocols and metrics. Following prior works (Ma et al., 2024a; Xu et al., 2025; Chen et al., 2025), we evaluate foreground general movable object (GMO) instances in both the present and future frames. For current-only methods, we follow Cam4DOcc and EfficientOCF by using current-frame predictions for all future steps. To assess occupancy forecasting performance, we report voxel-wise IoU against the fine-grained 3D occupancy ground truth. For instance-level evaluation, we additionally report average precision (AP) based on the voxel IoU between predicted and ground-truth instance occupancy masks. Following GUIDE (Hu et al., 2026), we compute AP at IoU thresholds of 0.1, 0.2, and 0.3 and report the mean. For dense voxel-wise methods, we recover instance masks using the official flow-based grouping and compute AP using an evaluation scheme inspired by Panoptic-DeepLab (Cheng et al., 2020). For center localization, we report $\mathrm { A P _ { c t r } }$ and Recall at a 1 m threshold, measuring confidence-ranked center localization and active-query GT coverage, respectively. All baseline models are reproduced using their official code, and all methods are evaluated on a single NVIDIA RTX Pro 6000 GPU. Further details are provided in the Appendix.

![](images/b0faaca702bc66fa87226aa4d559b95553904489d3f795b6e3990a58981f68e0.jpg)  
Figure 4: Qualitative comparison of GMO occupancy predictions from multi-view RGB images. LighTROcc more faithfully captures object locations and future motion, with darker colors indicat ing later forecasting steps.

## 4.2 EXPERIMENTAL RESULTS

Forecasting General Movable Object Occupancy. We conduct quantitative comparisons to evaluate 4D occupancy forecasting for general movable objects. For a resolution-matched comparison with prior instance-wise methods, we additionally report LighTROcc<sup>†</sup>, a lighter variant using 704 × 256 inputs. As shown in Table 1, dense voxel-wise approaches achieve competitive voxelwise IoU, indicating effective reconstruction of overall occupied regions. However, their instanceagnostic forecasts require flow-based post-processing to recover object masks, resulting in relatively low AP and weaker object-level consistency over time. In contrast, instance-wise approaches generally achieve higher AP while maintaining competitive IoU. LighTROcc obtains the best results across all reported IoU and AP metrics, reaching $\bar { \mathrm { I o U } } _ { \mathrm { a v g } } = 1 6 . 1 9$ and $\mathrm { A P _ { a v g } = 3 5 . 2 0 }$ while running at 12.80 FPS. Notably, LighTROcc<sup>†</sup> retains comparable occupancy accuracy while running nearly twice as fast as the full-resolution model. These results demonstrate that LighTROcc maintains reliable occupancy forecasting and object-level consistency across different input resolutions while remaining suitable for real-time inference.

Qualitative results. To qualitatively assess LighTROcc, we compare its predictions with representative baselines. As shown in Figure 4, the results visualize instance-wise GMO occupancy and its evolution over time, while GUIDE (Hu et al., 2026) is shown only for the current frame because it does not model future states. GUIDE localizes objects well in the current frame, but cannot forecast future occupancy due to the absence of future-state modeling. OccProphet (Chen et al., 2025) supports future occupancy forecasting, but often fails to preserve consistent object geometry across time. In contrast, LighTROcc more accurately captures future object locations while maintaining consistent object representations throughout the forecasting horizon. Its accurate trajectory forecasting further demonstrates its potential for downstream planning and control in autonomous driving. Additional qualitative results and analyses are provided in the Appendix.

## 4.3 COMPUTATIONAL EFFICIENCY AND MODEL COMPACTNESS

We compare LighTROcc with prior baselines in terms of computational efficiency and model compactness. Table 2 reports parameters, GFLOPs, latency, FPS, and forecasting accuracy, with all models benchmarked using official implementations on a single NVIDIA RTX Pro 6000 GPU.

Table 2: Comparison of model compactness, computational efficiency, and forecasting accuracy. Runtime is measured using official implementations on a single NVIDIA RTX Pro 6000 GPU. LighTROcc<sup>†</sup> denotes the resolution-matched variant using 704 × 256 inputs. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Category</td><td>Method</td><td>Input Res.</td><td>Params. (M)</td><td> $\mathrm { G F L O P s }$ </td><td> $\begin{array} { c } { { \mathrm { L a t e n c y } } } \\ { { \mathrm { ( m s ) } } } \end{array}$ </td><td>FPS</td><td> $\mathrm { I o U _ { a v g } }$ </td><td> $\mathsf { A P _ { a v g } }$ </td></tr><tr><td rowspan="2">Current-only</td><td>SparseOcc (Liu et al., 2024)</td><td> $7 0 4 \times 2 5 6$ </td><td>80</td><td>568</td><td>70.53</td><td>14.18</td><td>14.62</td><td>14.70</td></tr><tr><td>GUIDE (Hu et al., 2026)</td><td> $7 0 4 \times 2 5 6$ </td><td>90</td><td>1,530</td><td>122.51</td><td>8.16</td><td>12.96</td><td>26.55</td></tr><tr><td rowspan="5">4D forecasting</td><td>Cam4DOcc (Ma et al., 2024a)</td><td> $1 6 0 0 \times 8 9 6$ </td><td>370</td><td>13,168</td><td>1,902.26</td><td>0.52</td><td>12.12</td><td>8.56</td></tr><tr><td>EfficientOCF (Xu et al., 2025)</td><td> $1 6 0 0 \times 8 9 6$ </td><td>388</td><td>5,143</td><td>294.24</td><td>3.40</td><td>8.61</td><td>14.15</td></tr><tr><td>OccProphet (Chen et al., 2025)</td><td> $8 0 0 \times 4 4 8$ </td><td>82</td><td>2,020</td><td>305.93</td><td>3.27</td><td>14.89</td><td>9.83</td></tr><tr><td>LighTROcc† (Ours)</td><td> $7 0 4 \times 2 5 6$ </td><td>42</td><td>154</td><td>39.84</td><td>25.10</td><td>15.16</td><td>33.05</td></tr><tr><td>LighTROcc (Ours)</td><td> $1 6 0 0 \times 8 9 6$ </td><td>42</td><td>1,225</td><td>78.12</td><td>12.80</td><td>16.19</td><td>35.20</td></tr></table>

Table 3: Ablation of 3D center localization strategies. Direct 3D regression is compared with hard- and softargmax variants of attention-guided forward lifting under the same model configuration.  
Table 4: Effect of instance query count on computational efficiency and forecasting accuracy. All variants are evaluated using full-resolution inputs.
<table><tr><td>Localization</td><td> $\mathrm { I o U _ { a v g } \uparrow }$ </td><td> $\mathrm { A P _ { a v g } \uparrow }$ </td><td> $\mathrm { A P _ { c t r } \uparrow }$ </td><td> $\operatorname { R e c a l l } _ { \mathrm { l o c } } \uparrow$ </td></tr><tr><td>Direct regression (MLP)</td><td>11.13</td><td>26.10</td><td>17.23</td><td>21.14</td></tr><tr><td>Forward Lifting (hard)</td><td>11.20</td><td>28.50</td><td>25.43</td><td>29.84</td></tr><tr><td>Forward Lifting (soft)</td><td>16.19</td><td>35.20</td><td>27.31</td><td>31.06</td></tr></table>

<table><tr><td># Queries</td><td>FPS↑</td><td> $\mathrm { I o U } _ { \mathrm { a v g } }$  ↑</td><td> $\mathrm { A P _ { a v g } \uparrow }$ </td></tr><tr><td>100</td><td>12.95</td><td>15.16</td><td>34.01</td></tr><tr><td>200</td><td>12.80</td><td>16.19</td><td>35.20</td></tr><tr><td>900</td><td>11.42</td><td>14.13</td><td>31.61</td></tr></table>

Among current-only methods, SparseOcc (Liu et al., 2024) is the most efficient, partly aided by an input resolution nearly 8× lower than our full-resolution setting. Among 4D forecasting methods, LighTROcc achieves both the highest inference speed and the best accuracy. For a resolutionmatched comparison, LighTROcc<sup>†</sup> runs about 1.8× faster than SparseOcc while also achieving higher accuracy. Notably, the full-resolution LighTROcc maintains real-time inference despite processing high-resolution inputs and forecasting multiple frames in a single pass, while achieving the best performance. This highlights the effectiveness of our compact instance-centric formulation.

## 4.4 ABLATION STUDIES

Ablation on 3D center localization. A key factor enabling accurate instance recognition and localization with a compact query set is the proposed attention-guided forward lifting. Because decoded queries primarily encode 2D visual evidence, directly regressing metric 3D centers without explicit spatial priors is highly underconstrained. Forward lifting eases this localization problem, and we further employ soft argmax rather than hard peak selection to obtain stable image-plane centers. To isolate the contribution of each design choice, we compare direct MLP regression, hard-argmax lifting, and our soft-argmax lifting. As shown in Table $^ { 3 , }$ direct regression yields the lowest $\mathrm { \bar { I } o U _ { a v g } }$ and $\mathrm { A P _ { a v g } , }$ together with substantially lower $\mathrm { A P _ { c t r } }$ and ${ \mathrm { R e c a l l } } _ { \mathrm { l o c } } ^ { - }$ indicating that poor 3D grounding also degrades overall occupancy prediction. Hard-argmax lifting improves both forecasting accuracy and center localization, but remains vulnerable to spurious outlier peaks in the attention map. In contrast, soft argmax aggregates the full attention distribution, providing more reliable center grounding and achieving the best performance across all four metrics. Figure 5 further shows that the learned attention maps capture the corresponding object regions, while soft argmax estimates their centers more reliably than hard argmax.

Query scaling and efficiency-accuracy trade-off. We analyze how query count affects instancecentric perception and forecasting. Many prior approaches based on fixed spatial anchors initialize 900 or more query candidates to ensure sufficient scene coverage (Lin et al., 2023a; Liu et al., 2023a; Hu et al., 2026). In contrast, our attention-guided forward lifting explicitly localizes queries, allowing LighTROcc to operate with a substantially smaller query set. Table 4 compares the resulting accuracy and computational overhead, with all variants trained using full-resolution inputs.

Empirically, 200 queries provide the best balance between accuracy and efficiency. Reducing the query count to 100 yields only a marginal speed improvement while slightly degrading AP and

![](images/41ccd53f62284728d0ab807aaa77506791c0bec33e38be3499672914463ebc80.jpg)  
Figure 5: Qualitative comparison of hard- and soft-argmax localization from query attention maps. Hard argmax is sensitive to spurious outlier peaks, while soft argmax provides more reliable imageplane centers within the corresponding instance regions.

![](images/d0885b11aff739f7575517bd9a1be5c68258975da56f6c7d8b2f2a639f189521.jpg)

![](images/45bff9da1c68056497c113b2660b7bfd6984f3595c3f438166ca4c4cd358583b.jpg)

![](images/e0032ed7a6990cdf4a8139fc65b154fa5bb868d4c1ab57986a01d2f0e28c2e1d.jpg)

![](images/32670e5b852e16e8cd245e37c831f0eb1d656608a75a58e383fd3b5727ea9e5c.jpg)  
Figure 6: Statistical analysis of query scaling over the evaluation set. Queries with voxel IoU ≥ 0.2 are treated as geometry-valid. Although 900 queries improve GT coverage, they introduce more redundant and invalid predictions, whereas 200 queries provide stronger confidence discrimination and cleaner query selection.

IoU, indicating insufficient representation capacity. Conversely, increasing the query count by 4.5× to 900, comparable to the scale commonly used in anchor-based methods, introduces substantial attention overhead and noticeably degrades prediction accuracy. This suggests that excessive query redundancy is not only computationally inefficient but can also hinder effective instance prediction. We provide further statistical analysis of query scaling below. We therefore adopt 200 queries as our default setting, which achieves the most favorable balance between accuracy and efficiency.

Statistical analysis of query scaling. We further analyze the effect of query scaling over the entire evaluation set. For this analysis, we regard a query as geometry-valid if its voxel IoU with a GT object is at least 0.2, and invalid otherwise. As shown in Figure 6 (a), increasing the number of queries from 200 to 900 consistently improves GT object coverage across different voxel-IoU thresholds. However, this increased coverage comes with substantial redundancy. Figure 6 (b) shows that the number of invalid queries grows considerably faster than the number of valid queries, while excess duplicate predictions also increase markedly. This excessive redundancy degrades the model’s discriminative ability, making it harder to separate valid queries from redundant or invalid predictions based on confidence.

This behavior is reflected in the confidence statistics. In Figure 6 (c), the 200-query model exhibits a consistently stronger ROC curve, indicating better discrimination between valid and invalid predictions. Figure 6 (d) provides a complementary view: across confidence thresholds, the 200-query model captures a larger fraction of geometry-valid queries while admitting fewer invalid ones. Together, these results indicate that although additional queries improve scene coverage, excessive query redundancy weakens confidence-based selection and degrades instance-level prediction quality. This analysis supports our choice of 200 queries as a compact yet expressive query set.

## 5 CONCLUSION

We presented LighTROcc, a lightweight instance-centric framework for camera-based 4D occupancy forecasting. LighTROcc represents movable objects with a compact set of learned queries, avoiding dense spatiotemporal feature processing and large sets of spatially initialized candidates. Each query is localized through attention-guided forward lifting, modeled with a mixture of 3D Gaussians, and propagated to future frames using predicted displacements. This design predicts present and future occupancy in a single forward pass while maintaining instance-specific representations. Experiments on nuScenes and supplemented nuScenes-Occupancy show that LighTROcc outperforms all evaluated baselines in IoU and AP while achieving the highest inference speed among 4D forecasting methods. Ablations validate the localization and compact query design, demonstrating an accuracy-efficiency trade-off for real-time 4D occupancy forecasting.

## AI USE STATEMENT

We used AI in a limited manner for grammar and style polishing only. It did not contribute to research ideation, experimental design, or analysis. The authors take full responsibility for all content.

## REPRODUCIBILITY STATEMENT

We provide the model architecture and training objectives in Section 3, and the datasets and evaluation protocols in Section 4.1. Appendix A details the dataset construction and evaluation metrics, while Appendix B provides the network architecture and training implementation details.

## REFERENCES

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 11618–11628. IEEE, 2020.

Anh-Quan Cao and Raoul de Charette. MonoScene: Monocular 3d semantic scene completion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3991–4001, 2022.

Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. ShapeNet: An information-rich 3D model repository. Technical Report 1512.03012, arXiv preprint, December 2015.

Junliang Chen, Huaiyuan Xu, Yi Wang, and Lap-Pui Chau. OccProphet: Pushing the efficiency frontier of camera-only 4d occupancy forecasting with an observer-forecaster-refiner framework. In International Conference on Learning Representations (ICLR), 2025.

Xuanyao Chen, Tianyuan Zhang, Yue Wang, Yilun Wang, and Hang Zhao. FUTR3D: A unified sensor fusion framework for 3d detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 172–181, 2023.

Bowen Cheng, Maxwell D Collins, Yukun Zhu, Ting Liu, Thomas S Huang, Hartwig Adam, and Liang-Chieh Chen. Panoptic-deeplab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12472–12482. IEEE, 2020.

Wanshui Gan, Fang Liu, Hongbin Xu, Ningkai Mo, and Naoto Yokoya. GaussianOcc: Fully selfsupervised and efficient 3d occupancy estimation with gaussian splatting. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28980–28990, 2025.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Chunyong Hu, Qi Luo, Jianyun Xu, Song Wang, Qiang Li, and Sheng Yang. GUIDE: Gaussian unified instance detection for enhanced obstacle perception in autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 4816–4824, 2026.

Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9223–9232, 2023.

Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. GaussianFormer: Scene as gaussians for vision-based 3d semantic occupancy prediction. In European Conference on Computer Vision (ECCV), pp. 376–393, 2024.

Yuanhui Huang, Amonnut Thammatadatrakoon, Wenzhao Zheng, Yunpeng Zhang, Dalong Du, and Jiwen Lu. GaussianFormer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27477–27486, 2025.

Haoyi Jiang, Tianheng Cheng, Naiyu Gao, Haoyang Zhang, Tianwei Lin, Wenyu Liu, and Xinggang Wang. Symphonize 3d semantic scene completion with contextual instance queries. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20258–20267, 2024.

Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and Xinggang Wang. GaussTR: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11960–11970, 2025.

Yanqin Jiang, Li Zhang, Zhenwei Miao, Xiatian Zhu, Jin Gao, Weiming Hu, and Yu-Gang Jiang. PolarFormer: Multi-camera 3d object detection with polar transformer. In AAAI Conference on Artificial Intelligence, volume 37, pp. 1042–1050, 2023.

Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M. Alvarez, Sanja Fidler, Chen Feng, and Anima Anandkumar. VoxFormer: Sparse voxel transformer for camera-based 3d semantic scene completion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9087–9098, 2023a.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In European Conference on Computer Vision (ECCV), pp. 1–18, 2022.

Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M. Alvarez. FB-OCC: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023b.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense´ object detection. In Proceedings of the IEEE international conference on computer vision, pp. 2980–2988, 2017.

Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4D: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581, 2022.

Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4D v2: Recurrent temporal fusion with sparse model. arXiv preprint arXiv:2305.14018, 2023a.

Xuewu Lin, Zixiang Pei, Tianwei Lin, Lichao Huang, and Zhizhong Su. Sparse4D v3: Advancing end-to-end 3d detection and tracking. arXiv preprint arXiv:2311.11722, 2023b.

Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, and Limin Wang. SparseBEV: High-performance sparse 3d object detection from multi-camera videos. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18580–18590, 2023a.

Haisong Liu, Yang Chen, Haiguang Wang, Zetong Yang, Tianyu Li, Jia Zeng, Li Chen, Hongyang Li, and Limin Wang. Fully sparse 3d occupancy prediction. In European Conference on Computer Vision (ECCV), pp. 54–71, 2024.

Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. PETR: Position embedding transformation for multi-view 3d object detection. In European Conference on Computer Vision (ECCV), pp. 531–548, 2022.

Yingfei Liu, Junjie Yan, Fan Jia, Shuailin Li, Aqi Gao, Tiancai Wang, and Xiangyu Zhang. PETRv2: A unified framework for 3d perception from multi-camera images. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3262–3272, 2023b.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, October 2015.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019. URL https://mlanthology.org/iclr/ 2019/loshchilov2019iclr-decoupled/.

Yuhang Lu, Xinge Zhu, Tai Wang, and Yuexin Ma. OctreeOcc: Efficient and multi-granularity occupancy prediction using octree queries. In Advances in Neural Information Processing Systems (NeurIPS), pp. 79618–79641, 2024.

Junyi Ma, Xieyuanli Chen, Jiawei Huang, Jingyi Xu, Zhen Luo, Jintao Xu, Weihao Gu, Rui Ai, and Hesheng Wang. Cam4DOcc: Benchmark for camera-only 4d occupancy forecasting in autonomous driving applications. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21486–21495, 2024a.

Qihang Ma, Xin Tan, Yanyun Qu, Lizhuang Ma, Zhizhong Zhang, and Yuan Xie. COTR: Compact occupancy transformer for vision-based 3d occupancy prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19936–19945, 2024b.

Gyeongrok Oh, Sungjune Kim, Heeju Ko, Hyung-gun Chi, Jinkyu Kim, Dongwook Lee, Daehyun Ji, Sungjoon Choi, Sujin Jang, and Sangpil Kim. 3d occupancy prediction with low-resolution queries via prototype-aware view transformation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17134–17144, 2025.

Jinhyung Park, Chensheng Peng, Yihan Hu, Wenzhao Zheng, Kris Kitani, and Wei Zhan. S2go: Streaming sparse gaussian occupancy. In International Conference on Learning Representations (ICLR), 2026.

Luis Roldao, Raoul de Charette, and Anne Verroust-Blondet. LMSCNet: Lightweight multiscale 3d˜ semantic completion. In International Conference on 3D Vision (3DV), pp. 111–119, 2020.

Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, and Ali Gholipour. Tversky loss function for image segmentation using 3d fully convolutional deep networks. In International workshop on machine learning in medical imaging, pp. 379–387. Springer, 2017.

Yiang Shi, Tianheng Cheng, Qian Zhang, Wenyu Liu, and Xinggang Wang. Occupancy as set of points. In European Conference on Computer Vision (ECCV), pp. 72–87, 2024.

Shuran Song, Fisher Yu, Andy Zeng, Angel X. Chang, Manolis Savva, and Thomas Funkhouser. Semantic scene completion from a single depth image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1746–1754, 2017.

Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. SparseDrive: End-to-end autonomous driving via sparse scene representation. In IEEE International Conference on Robotics and Automation (ICRA), pp. 8795–8801, 2025.

Pin Tang, Zhongdao Wang, Guoqing Wang, Jilai Zheng, Xiangxuan Ren, Bailan Feng, and Chao Ma. SparseOcc: Rethinking sparse latent representation for vision-based semantic occupancy prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15035–15044, 2024.

Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, and Hongyang Li. Scene as occupancy. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8406–8415, October 2023.

Jiabao Wang, Zhaojiang Liu, Qiang Meng, Liujiang Yan, Ke Wang, Jie Yang, Wei Liu, Qibin Hou, and Ming-Ming Cheng. OPUS: Occupancy prediction using a sparse set. In Advances in Neural Information Processing Systems (NeurIPS), pp. 119861–119885, 2024a.

Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3621–3631, 2023a.

Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen Lu, and Xingang Wang. OpenOccupancy: A large scale benchmark for surrounding semantic occupancy perception. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17804–17813, 2023b.

Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. DETR3D: 3d object detection from multi-view images via 3d-to-2d queries. In Conference on Robot Learning (CoRL), pp. 180–191, 2022.

Yuping Wang, Xiangyu Huang, Xiaokang Sun, Mingxuan Yan, Shuo Xing, Zhengzhong Tu, and Jiachen Li. Uniocc: A unified benchmark for occupancy forecasting and prediction in autonomous driving. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 25560– 25570. IEEE, 2025.

Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, and Zhaoxiang Zhang. PanoOcc: Unified occupancy representation for camera-based 3d panoptic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17158–17168, 2024b.

Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. SurroundOcc: Multicamera 3d occupancy prediction for autonomous driving. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21729–21740, 2023.

Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, and Xiang Bai. CAPE: Camera view position embedding for multi-view 3d object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21570–21579, 2023.

Jingyi Xu, Xieyuanli Chen, Junyi Ma, Jiawei Huang, Jintao Xu, Yue Wang, and Ling Pei. Spatiotemporal decoupling for efficient vision-based occupancy forecasting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22338–22347, 2025.

Chenyu Yang, Yuntao Chen, Hao Tian, Chenxin Tao, Xizhou Zhu, Zhaoxiang Zhang, Gao Huang, Hongyang Li, Yu Qiao, Lewei Lu, Jie Zhou, and Jifeng Dai. BEVFormer v2: Adapting modern image backbones to bird’s-eye-view recognition via perspective supervision. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17830–17839, 2023.

Zichen Yu, Changyong Shu, Jiajun Deng, Kangjie Lu, Zongdai Liu, Jiangyong Yu, Dawei Yang, Hui Li, and Yan Chen. FlashOcc: Fast and memory-efficient occupancy prediction via channelto-height plugin. arXiv preprint arXiv:2311.12058, 2023.

Tianyuan Zhang, Xuanyao Chen, Yue Wang, Yilun Wang, and Hang Zhao. MUTR3D: A multicamera tracking framework via 3d-to-2d queries. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 4537–4546, 2022.

Yunpeng Zhang, Zheng Zhu, and Dalong Du. OccFormer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9433–9443, 2023.

Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, and Jiwen Lu. PointOcc: Cylindrical tri-perspective view for point-based 3d semantic occupancy prediction. arXiv preprint arXiv:2308.16896, 2023.

## A EVALUATION PROTOCOL AND DATASET ANALYSIS

## A.1 EVALUATION METRICS

Occupancy and instance evaluation. In this section, we provide additional details on the evaluation protocol for the instance-level average precision (AP) and voxel-wise intersection over union (IoU) reported in the main paper. For instance-level AP, each predicted object is represented by a voxel occupancy mask and an instance confidence score. Predicted instances are matched to groundtruth instances based on their voxel IoU. A prediction is considered a true positive if its IoU with the matched ground-truth instance exceeds a threshold τ, while unmatched predictions are counted as false positives. We compute $\mathbf { A P }$ independently at $\tau \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ and report the mean across these thresholds for each temporal step.

Let $\mathrm { A P } _ { k }$ denote the threshold-averaged AP at time step k, where $k = 0$ corresponds to the present frame and $k \in \{ 1 , \ldots , T _ { f } \}$ denotes a future frame. The reported temporal metrics are computed as

$$
\mathrm { A P } _ { \mathrm { c u r } } = \mathrm { A P } _ { 0 } , \qquad \mathrm { A P } _ { \mathrm { f u t } } = \frac { 1 } { T _ { f } } \sum _ { k = 1 } ^ { T _ { f } } \mathrm { A P } _ { k } ,\tag{9}
$$

and

$$
\mathrm { A P _ { a v g } } = { \frac { 1 } { T _ { f } + 1 } } \sum _ { k = 0 } ^ { T _ { f } } \mathrm { A P } _ { k } .\tag{10}
$$

Thus, $\mathrm { A P _ { f u t } }$ measures forecasting accuracy exclusively over future frames, whereas $\mathrm { A P _ { a v g } }$ summarizes performance over both the present and future states.

For voxel-wise evaluation, all GMO instances at each temporal step are merged into a binary foreground occupancy map. Given the predicted and ground-truth foreground voxel sets $\mathcal { P } _ { k }$ and $\mathcal { G } _ { k }$ respectively, we compute

$$
\operatorname { I o U } _ { k } = { \frac { | { \mathcal { P } } _ { k } \cap { \mathcal { G } } _ { k } | } { | { \mathcal { P } } _ { k } \cup { \mathcal { G } } _ { k } | } } .\tag{11}
$$

$\mathrm { I o U } _ { \mathrm { c u r } } , \mathrm { I o U } _ { \mathrm { f u t } }$ , and $\mathrm { I o U _ { a v g } }$ are aggregated over the temporal dimension in the same manner as their AP counterparts.

For dense voxel-wise baselines, instance masks required for AP evaluation are recovered from their occupancy and flow predictions using the corresponding official grouping procedure. This conversion is used only for instance-level evaluation, while voxel-wise IoU is computed directly from the predicted foreground occupancy.

Center localization evaluation. In addition to occupancy prediction, we evaluate query-level center localization on the current frame. We report $\mathrm { A P _ { c t r } }$ and $\mathrm { R e c a l l _ { l o c } }$ at a 1 m threshold, measuring confidence-ranked center localization and active-query GT coverage, respectively. For $\mathrm { A P _ { c t r } , }$ foreground queries are matched to ground-truth instances based on the center distance. Queries are processed in descending confidence order, and a prediction is considered a true positive if its distance to an unmatched ground-truth instance is within the threshold. The resulting precision-recall curve is used to compute $\mathrm { A P _ { c t r } }$ . For $\mathrm { R e c a l l _ { l o c } }$ , we evaluate active queries after inference-time filtering and match them with ground-truth instances using one-to-one radius matching. It measures the ratio of successfully localized ground-truth instances:

$$
\mathrm { R e c a l l } _ { \mathrm { l o c } , \tau } = \frac { N _ { \mathrm { m a t c h e d } } ^ { d \leq \tau } } { N _ { \mathrm { G T } } } ,\tag{12}
$$

where $N _ { \mathrm { m a t c h e d } } ^ { d \le \tau }$ denotes the number of ground-truth instances matched within the distance threshold τ. Both metrics are computed using center distances.

## A.2 DATASET ANALYSIS AND GROUND TRUTH SUPPLEMENTATION

Following prior works (Ma et al., 2024a; Xu et al., 2025; Chen et al., 2025), we evaluate occupancy predictions using the ground-truth annotations from nuScenes-Occupancy (Wang et al., 2023b). Since its annotations are derived from LiDAR observations, however, they are inherently incomplete.

LiDAR primarily captures visible object surfaces rather than their complete volumetric extent, often resulting in hollow or incomplete occupancy structures even when the overall object shape appears visually plausible. This issue becomes more pronounced for distant or occluded objects, where only a small fraction of the object surface may be observed and the resulting occupancy can be severely fragmented. The upper row of Figure 7 (a) illustrates representative examples of such sparse annotations. Beyond their limited visual completeness, these annotations can also introduce undesirable biases in both training and evaluation.

Nearby objects generally contain denser occupancy observations and therefore provide relatively strong signal. In contrast, distant or low-visibility objects may occupy only a few voxels, providing substantially weaker signals. This imbalance can bias a model toward predicting well-observed nearby objects while suppressing predictions for difficult instances. As illustrated in Figure 7 (b), predicting the full spatial extent of a sparsely annotated object can introduce many apparent falsepositive voxels, even when the predicted geometry is physically reasonable. Conversely, missing sparsely observed regions may incur only a limited penalty under voxel-based evaluation. This creates an unfavorable evaluation bias that can encourage prediction suppression for difficult objects. Such behavior is undesirable in autonomous driving, where reliably detecting distant and partially occluded objects is particularly important.

![](images/ae4bcb42a77ebedaffc290c784932f545239ad88cf7dd3a18b2f3a9259c7340c.jpg)  
Figure 7: Illustration of ground-truth occupancy supplementation and evaluation bias from sparse annotations. (a) Original sparse occupancy annotations (top) and their supplemented counterparts using 3D assets (bottom). (b) Sparse ground truth can introduce apparent false-positive predictions even when the predicted geometry is physically reasonable.

To mitigate this issue, we supplement the original sparse occupancy annotations with 3D geometric prototypes from ShapeNet (Chang et al., 2015) and SMPL (Loper et al., 2015). For each object, we obtain its 3D bounding-box center, dimensions, orientation, and semantic category from the nuScenes annotations, and place a corresponding 3D mesh according to its estimated pose and extent. We use watertight meshes so that the enclosed volume can be treated as occupied rather than retaining only the visible surface. Importantly, the original nuScenes-Occupancy annotation is preserved rather than replaced. The asset-based occupancy is combined with the original annotation through a voxel-wise union, thereby retaining the observed object geometry while completing severely missing regions. The lower row of Figure 7 (a) visualizes the resulting supplemented annotations. This procedure provides more balanced evaluation across objects with different distances and visibility levels, while imposing a clearer penalty for missed instances and reducing the artificial false-positive penalty caused by sparse LiDAR observations.

To quantify the effectiveness of supplementation and its impact on evaluation fairness, we measure voxel density under different distance and visibility conditions. We analyze 518,702 3D object annotations from 29,049 training and evaluation sequences. The voxel density is computed as

$$
D = \frac { \sum _ { i } \left| S _ { i } \cap B _ { i } \right| } { \sum _ { i } \left| B _ { i } \right| } ,\tag{13}
$$

where $S _ { i }$ denotes either the original sparse voxel set or the supplemented voxel set of object i, and $B _ { i }$ denotes the set of voxels inside its ground-truth 3D bounding box from nuScenes. As shown in Table 5, the overall density increases from 17.62% to 71.51% (4.06×).

When grouped by distance, the original annotations become increasingly sparse as objects move farther away, with density falling below 14% beyond 30 m and reaching only 8.11% beyond 50 m.

Consequently, even predictions that correctly cover the object location and extent can incur a large number of apparent false-positive voxels under the original annotations. In contrast, the supplemented annotations maintain substantially more stable densities across distance, ranging only from 74.06% to 68.90%, thereby providing more consistent evaluation for distant objects. A similar trend is observed across visibility levels, where supplementation substantially reduces the density gap between poorly and highly visible objects. These results show that the supplemented annotations effectively mitigate the distance- and visibility-dependent sparsity of the original annotations, reducing the resulting evaluation bias and enabling a more balanced assessment across different observation conditions.

Table 5: Occupancy density before and after 3D asset supplementation. Gain denotes the ratio of supplemented to original occupancy density.
<table><tr><td>Subset</td><td>Range</td><td>Original (%)</td><td>Supplemented (%)</td><td>Gain (×)</td></tr><tr><td>Overall</td><td>All</td><td>17.62</td><td>71.51</td><td>4.06</td></tr><tr><td rowspan="6">Distance (m)</td><td>[0,10)</td><td>28.07</td><td>74.06</td><td>2.64</td></tr><tr><td>[10, 20)</td><td>23.39</td><td>72.73</td><td>3.11</td></tr><tr><td>[20, 30)</td><td>17.63</td><td>71.59</td><td>4.06</td></tr><tr><td>[30, 40)</td><td>13.95</td><td>70.79</td><td>5.08</td></tr><tr><td>[40, 50)</td><td>11.55</td><td>69.97</td><td>6.06</td></tr><tr><td>≥ 50</td><td>8.11</td><td>68.90</td><td>8.50</td></tr><tr><td rowspan="4">Visibility (%)</td><td>V1 (0 to 40)</td><td>11.63</td><td>69.19</td><td>5.95</td></tr><tr><td>V2 (40 to 60)</td><td>14.54</td><td>70.24</td><td>4.83</td></tr><tr><td>V3 (60 to 80)</td><td>16.94</td><td>71.15</td><td>4.20</td></tr><tr><td>V4 (80 to 100)</td><td>21.94</td><td>73.24</td><td>3.34</td></tr></table>

## A.3 GEOMETRIC CONSISTENCY OF SUPPLEMENTED ANNOTATIONS

In this section, we perform complementary geometric checks to validate the supplemented annotations against the original nuScenes-Occupancy (Wang et al., 2023b) annotations and LiDAR observations. We first measure the surface distance to quantify how closely the added assets fit the original LiDAR-derived occupied voxels. For each annotated object, let $S _ { i }$ denote the centers of its original occupied voxels and $\mathcal { M } _ { i }$ the surface of the fitted asset. For each $\mathbf { x } \in S _ { i } .$ , we compute the nearest-surface distance as

$$
d ( \mathbf { x } , { \mathcal { M } } _ { i } ) = \operatorname* { m i n } _ { \mathbf { y } \in { \mathcal { M } } _ { i } } \| \mathbf { x } - \mathbf { y } \| _ { 2 } .\tag{14}
$$

We report the mean and 90th-percentile distances, where smaller values indicate closer agreement with the geometry captured by the original observations.

We additionally measure the free-space violation rate to assess whether the supplemented geometry introduces excessive occupancy beyond the observed object geometry. We trace the original LiDAR rays and define the observed free space along each ray up to its measured return range as

$$
\mathcal { F } ( r ) = \left\{ \mathbf { s } + t \mathbf { d } \mid 0 \leq t < r \right\} ,\tag{15}
$$

where s is the LiDAR sensor origin, d is the unit ray direction, r is the measured return range, and t is the distance from the sensor origin along the ray. We then measure the fraction of supplemented occupancy that overlaps this observed free space, where a lower value indicates less conflict with the original observations.

As shown in Table 6, the fitted assets achieve a mean nearest-surface distance of 0.20 m and a P90 distance of 0.38 m. Given the 0.2 m voxel resolution used for evaluation, these correspond to approximately 1.0 and 1.9 voxels, respectively. The free-space violation rate is only 2.4%, indicating that the supplemented geometry rarely conflicts with directly observed free space. Together with the density analysis in Appendix A.2, these results show that the supplementation increases geometric completeness while remaining largely consistent with the original sensor evidence.

Table 6: Geometric consistency of supplemented annotations. Surface distance measures agreement with occupied voxels, while free-space violation measures overlap with LiDAR-observed free space.
<table><tr><td>Metric</td><td>Result</td></tr><tr><td>Nearest-surface distance, mean ↓ Nearest-surface distance, P90 ↓ LiDAR free-space violation ↓</td><td>0.20 m 0.38 m</td></tr></table>

## A.4 ROBUSTNESS TO GROUND-TRUTH ANNOTATION PROTOCOL

To examine whether the comparative advantage of LighTROcc depends on the supplemented annotations, we additionally evaluate all methods using the original sparse nuScenes-Occupancy annotations. As analyzed in Appendix A.2, these annotations cover only 17.62% of the object boundingbox volume on average and become increasingly sparse with distance. Consequently, even geometrically plausible predictions may be counted as false positives when they occupy regions that are simply unobserved in the sparse ground truth.

To mitigate this annotation-induced penalty, we adopt a conservative bbox-aware evaluation protocol. Predicted voxels that lie inside annotated 3D bounding boxes but are absent from the sparse ground truth are excluded from false-positive counting, without being reassigned as true positives. The resulting IoU is computed as

$$
\mathrm { I o U } = { \frac { T P } { T P + F N + F P - F P ^ { \mathrm { b b o x } } } } ,\tag{16}
$$

where $F P ^ { \mathrm { b b o x } }$ denotes false-positive voxels inside the annotated 3D bounding boxes. The same ignore mask is applied when computing voxel overlaps for instance-level AP. Thus, the protocol does not assume that the entire bounding box is occupied, but only avoids treating unlabeled in-box regions as reliable negative evidence.

As shown in Table 7, LighTROcc retains the highest aggregate performance, achieving $\mathrm { I o U _ { a v g } = }$ 9.16 and $\mathrm { A P _ { a v g } \ = \ 1 3 . { \bar { 9 } } 3 }$ . Although SparseOcc achieves a slightly higher current-frame IoU, LighTROcc obtains the best future and average IoU and the best AP across all temporal aggregations. Moreover, Table 1 shows that all evaluated methods achieve higher average IoU and AP under the supplemented annotations than under the original sparse-label protocol. This consistent improvement across methods suggests that supplementation alleviates a common sparsity-induced evaluation penalty rather than providing an advantage specific to LighTROcc.

Together with the density analysis in Appendix A.2, these results provide complementary evidence for our evaluation design. The original sparse annotations can introduce unintended evaluation bias due to their incomplete object coverage. Although the bbox-aware evaluation mitigates this issue by excluding ambiguous in-box false positives, it still provides no positive target for unobserved object regions. Consequently, missing predictions in these regions are not counted as false negatives, particularly for distant or occluded objects where the annotations are severely incomplete. To address both limitations, we adopt the supplemented annotations as a more complete evaluation target. The quantitative analyses and comparative experiments collectively support the rationale and effectiveness of this supplementation strategy.

## B IMPLEMENTATION DETAILS

## B.1 NETWORK ARCHITECTURE

We use a ResNet-18 (He et al., 2016) initialized with pretrained weights as the image backbone for feature extraction. The extracted image features are projected to the same embedding dimension as the instance queries, with $D = 1 2 8$ , before being processed by the transformer decoder. We initialize a compact set of 200 learnable instance queries. The transformer decoder consists of $L = 3$ layers with $\bar { H } = 4$ attention heads. To encode the spatial, camera-specific, and temporal information of the image tokens, we incorporate camera-parameter embeddings together with 2D sinusoidal positional encodings, learned camera-ID embeddings, and learned temporal embeddings.

Table 7: Quantitative comparison of dense voxel-wise and instance-wise methods under the original sparse nuScenes-Occupancy annotations using the bbox-aware evaluation protocol. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Type</td><td> $\mathrm { I o U } _ { \mathrm { c u r } }$ </td><td> $\mathrm { I o U } _ { \mathrm { f u t } }$ </td><td> $\mathrm { I o U _ { a v g } }$ </td><td> $\mathsf { A P } _ { \mathrm { c u r } }$ </td><td> $\mathrm { A P _ { f u t } }$ </td><td> $\mathsf { A P _ { a v g } }$ </td></tr><tr><td>Cam4DOcc (Ma et al., 2024a)</td><td>Dense</td><td>11.23</td><td>3.84</td><td>5.32</td><td>8.70</td><td>2.80</td><td>3.98</td></tr><tr><td>EfficientOCF (Xu et al., 2025)</td><td>Voxel</td><td>1.52</td><td>1.43</td><td>1.45</td><td>11.56</td><td>10.94</td><td>11.06</td></tr><tr><td>OccProphet (Chen et al., 2025)</td><td>wise</td><td>10.50</td><td>8.09</td><td>8.57</td><td>11.00</td><td>8.13</td><td>8.70</td></tr><tr><td>SparseOcc (Liu et al., 2024)</td><td>Instance</td><td>11.91</td><td>8.06</td><td>8.83</td><td>9.48</td><td>8.14</td><td>8.41</td></tr><tr><td>GUIDE (Hu et al., 2026)</td><td>wise</td><td>6.48</td><td>5.68</td><td>5.84</td><td>10.71</td><td>9.77</td><td>9.96</td></tr><tr><td>LighTROcc (Ours)</td><td></td><td>11.42</td><td>8.59</td><td>9.16</td><td>17.56</td><td>13.02</td><td>13.93</td></tr></table>

The depth head is implemented as a lightweight two-layer multilayer perceptron (MLP) that predicts a categorical depth distribution over $N _ { d } = 6 4$ bins within the range of [2, 58] m. The Gaussian head is likewise implemented using a two-layer MLP and predicts the parameters of 48 Gaussian components for each instance. These components are converted to an occupancy field as

$$
P _ { q , k } ( \mathbf { x } ) = 1 - \prod _ { g = 1 } ^ { N _ { g } } \left[ 1 - \alpha _ { q , g } \exp { \left( - \frac { 1 } { 2 } ( \mathbf { x } - \mu _ { q , g , k } ) ^ { \top } \Sigma _ { q , g } ^ { - 1 } ( \mathbf { x } - \mu _ { q , g , k } ) \right) } \right] .\tag{17}
$$

Here, $\alpha _ { q , g } \in [ 0 , 1 ]$ denotes the occupancy weight of each Gaussian. The trajectory and confidence heads are also lightweight two-layer MLPs. The trajectory head predicts four displacement vectors corresponding to the future forecasting steps, while the confidence head estimates the existence probability of each query.

## B.2 TRAINING DETAILS

We train LighTROcc for 20 epochs using the AdamW optimizer (Loshchilov & Hutter, 2019) with an initial learning rate of $3 \times \dot { 1 } 0 ^ { - 4 }$ and a weight decay of 0.01. The learning rate is scheduled using cosine annealing. Training is conducted on eight NVIDIA RTX Pro 6000 Blackwell GPUs with a batch size of one per GPU.

For Hungarian matching, we use the 3D center distance, attention overlap cost, and confidence cost with weights of 10.0, 0.3, and 0.2, respectively. For the training objective in Eq. (8), we set $( \lambda _ { \mathrm { c o n f } } , \lambda _ { \mathrm { c e n } } , \bar { \lambda } _ { \mathrm { d e p t h } } , \lambda _ { \mathrm { a t t n } } , \lambda _ { \mathrm { o c c } } , \lambda _ { \mathrm { t r a j } } ) = ( \bar { 1 . 0 } , 0 . 3 , \bar { 1 . 0 } , 1 . 0 , 1 . 0 , 1 . 0 )$ . For trajectory prediction, we augment each query feature with motion priors derived from the observed frames. Specifically, given three observed frames $( t - 2 , t - 1 , t )$ , we compute two displacement vectors corresponding to $t - 2 $ t − 1 and t − 1 → t. These observed displacements are concatenated with the query feature and fed to the trajectory head, which predicts four future displacement vectors. To improve training stability, we employ teacher forcing during early training and gradually transition to prediction-based training. Under teacher forcing, the motion priors are computed from ground-truth instance centers, whereas prediction-based training uses predicted centers. The teacher-forcing ratio is progressively reduced during warm-up and reaches zero from epoch 8 onward, after which all motion priors are derived from the predicted centers.

## C ADDITIONAL EXPERIMENTS AND ANALYSIS

## C.1 FLOW-BASED INSTANCE RECOVERY IN DENSE VOXEL-WISE MODELS

In this section, we further analyze the instance segmentation behavior of dense voxel-wise occupancy forecasting methods. As discussed in the main paper, methods such as Cam4DOcc (Ma et al., 2024a), EfficientOCF (Xu et al., 2025), and OccProphet (Chen et al., 2025) are not inherently designed in an instance-centric manner and therefore rely on an additional flow prediction module to recover object identities from dense occupancy maps. These methods typically predict backward centripetal flow, where each occupied voxel is encouraged to point toward the center of its corresponding object in the previous frame. The instance identity of a voxel is then determined according to the object center reached by its predicted flow. Voxels whose flows do not reach a valid object center cannot be reliably associated with any instance and may consequently remain unassigned. As a result, the instance segmentation quality of these dense methods is closely coupled with the accuracy of their flow predictions.

![](images/c05f740b123e274c613b23949f61448f94343e7475aef344e72d20ec767c0f16.jpg)  
Figure 8: Qualitative examples of failure modes in flow-based instance recovery for dense voxelwise models. (a) Instance merging, where voxels from multiple nearby objects are incorrectly assigned to a single identity. (b) Instance fragmentation, where voxels belonging to one object are split into multiple identities. Red boxes highlight representative errors.

As shown in Figure 8, even small flow errors can cause voxels to inherit incorrect instance identities, particularly in crowded scenes with nearby objects. This can merge multiple objects into a single instance or fragment one object into multiple identities, increasing both false-positive and false-negative predictions. Such association errors consequently degrade AP, which reflects both occupancy overlap and instance separation. These observations highlight the limitation of indirectly recovering object identities from dense occupancy and flow, and motivate the instance-centric design of LighTROcc, which maintains object-specific representations directly throughout forecasting.

## C.2 TEMPORAL SHAPE CONSISTENCY

A key advantage of LighTROcc is that its instance-centric representation preserves the predicted geometry of each object throughout the forecasting horizon. As discussed in Section C.1, dense voxel-wise methods recover instance identities indirectly from predicted flow. Flow errors can assign voxels to incorrect objects or leave them unassigned, resulting in fragmented instance masks and unstable object extents across frames. In contrast, LighTROcc propagates each predicted Gaussian mixture using the displacement vectors while retaining its component offsets, covariances, and occupancy weights, thereby preserving the shape and extent of each instance across future frames.

Following UniOcc’s object-aligned shape consistency evaluation (Wang et al., 2025), we compute Temporal Shape Consistency (TSC) to measure the preservation of predicted object geometry across consecutive frames, independent of global translation and orientation. Let $\hat { O } _ { i } ^ { k }$ denote the occupied voxel set of instance i at time step $k ,$ and $\kappa _ { i }$ its evaluated consecutive time steps. After transforming each shape into an object-local coordinate system through centroid subtraction and PCA-based alignment, denoted by A(·), the TSC is computed as

Table 8: Temporal shape consistency of 4D occupancy forecasting methods. Higher TSC indicates more stable geometry over time.
<table><tr><td>Method</td><td>TSC (%) ↑</td></tr><tr><td>Cam4DOcc EfficientOCF OccProphet</td><td>51.04 65.67 36.38</td></tr><tr><td>LighTROcc (Ours)</td><td>91.32</td></tr></table>

$$
\mathrm { T S C } = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } \frac { 1 } { | \mathcal { K } _ { i } | } \sum _ { k \in \mathcal { K } _ { i } } \mathrm { I o U } \left( \mathcal { A } ( \hat { O } _ { i } ^ { k } ) , \mathcal { A } ( \hat { O } _ { i } ^ { k + 1 } ) \right) ,\tag{18}
$$

where I denotes the set of evaluated predicted instances.

As shown in Table 8, dense voxel-wise baselines exhibit lower temporal shape consistency, reflecting greater variation in instance geometry across frames. LighTROcc achieves a TSC of 91.32, compared with 65.67 for EfficientOCF. This result reflects the temporal stability encouraged by the instance-centric representation.

Table 9: Forecasting performance across future horizons. LighTROcc (constant vel.) uses constantvelocity extrapolation instead of learned trajectory prediction. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="4">IoU</td><td colspan="4">AP</td></tr><tr><td>0.5 s</td><td>1.0 s</td><td>1.5 s</td><td>2.0 s</td><td>0.5 s</td><td>1.0 s</td><td>1.5 s</td><td>2.0 s</td></tr><tr><td>Cam4DOcc (Ma et al., 2024a)</td><td>Dense</td><td>13.02</td><td>12.11</td><td>11.39</td><td>10.68</td><td>9.39</td><td>8.17</td><td>7.49</td><td>7.03</td></tr><tr><td>EfficientOCF (Xu et al., 2025) OccProphet (Chen et al., 2025)</td><td>Voxel-wise</td><td>9.16</td><td>8.77</td><td>8.41</td><td>8.02</td><td>15.36</td><td>14.62</td><td>13.75</td><td>12.78</td></tr><tr><td></td><td></td><td>16.21</td><td>14.46</td><td>13.47</td><td>12.78</td><td>10.67</td><td>7.42</td><td>5.95</td><td>4.96</td></tr><tr><td>LighTROcc (Ours) LighTROcc (constant vel.)</td><td>Instance-wise</td><td>17.15 16.30</td><td>15.76 12.72</td><td>14.62 12.25</td><td>13.67 11.18</td><td>39.35 35.13</td><td>35.15 27.49</td><td>31.16 23.66</td><td>27.70 23.19</td></tr></table>

## C.3 PERFORMANCE AT DIFFERENT FORECASTING HORIZONS

To complement the aggregate results in Table 1 of the main paper, we report IoU and AP at each future forecasting horizon. The current-only instance-wise baselines are excluded from this comparison because they do not explicitly forecast future occupancy states. As shown in Table 9, all evaluated methods exhibit a gradual decrease in both IoU and AP as the forecasting horizon increases, reflecting the greater difficulty of longer-term occupancy prediction. Nevertheless, LighTROcc consistently achieves the best performance at every horizon, with particularly large margins in instancelevel AP. Notably, at the longest horizon of 2.0 s, LighTROcc reaches an IoU of 13.67 and an AP of 27.70, outperforming the strongest prior baselines by 0.89 IoU and 14.92 AP points, respectively. These results demonstrate that the proposed instance-centric formulation retains a consistent forecasting advantage across all evaluated horizons.

To further isolate the contribution of the trajectory head, we replace the learned future displacements with constant-velocity extrapolation while retaining the same current instance predictions and Gaussian geometry. As shown in Table 9, the constant-velocity variant consistently underperforms LighTROcc across all forecasting horizons. From 0.5 to 2.0 s, LighTROcc decreases from 17.15 to 13.67 IoU (−20.3%) and from 39.35 to 27.70 AP (−29.6%), whereas constant velocity decreases from 16.30 to 11.18 IoU (−31.4%) and from 35.13 to 23.19 AP (−34.0%). These results demonstrate the benefit of learned step-specific motion over linear extrapolation.

Ground Truth  
SparseOcc  
GUIDE  
Cam4DOcc  
EfficientOCF  
OccProphet  
LighTROcc (Ours)  
![](images/6bbf288d43c85f067ea25410dfa342f3ebe45d7ddeff8ec54e6b0d2299ffa519.jpg)  
Figure 9: Additional qualitative comparison of GMO occupancy predictions with a broader set of baselines. LighTROcc consistently captures object geometry and temporal evolution across diverse scenes. Colors become progressively darker for later forecasting steps.

## D ADDITIONAL QUALITATIVE RESULTS

Figure 9 provides additional qualitative comparisons with a broader set of baselines to complement the results in the main paper. For a comprehensive comparison, we visualize predictions from all evaluated baselines. Since SparseOcc (Liu et al., 2024) and GUIDE (Hu et al., 2026) are currentonly methods, their results are shown only for the present frame. Although these methods achieve competitive object localization at the current frame, they cannot model future occupancy evolution and therefore provide no predictions over the forecasting horizon. In contrast, the 4D forecasting methods explicitly reconstruct future occupancy states. Cam4DOcc (Ma et al., 2024a) produces temporally extended predictions, visualized from lighter to darker colors toward later steps, but tends to overestimate object extents. EfficientOCF (Xu et al., 2025) exhibits a similar overestimation tendency and, as shown in Scene 2, can merge nearby objects into a single instance. OccProphet (Chen et al., 2025) generally captures object extent more accurately, but its flow-based instance recovery is less stable, often causing object occupancy to progressively disappear at later forecasting steps. In comparison, LighTROcc consistently preserves instance geometry while accurately modeling future object evolution across diverse scenes. These qualitative results further demonstrate the effective ness of the proposed instance-centric forecasting formulation.