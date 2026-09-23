# Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized HD Mapping

Markus Käppeler, Rohit Mohan, and Abhinav Valada

Abstract— Camera-based 3D object detection and online vectorized HD mapping provide compact scene representations for autonomous driving, but both depend on accurate metric geometry and remain limited by depth ambiguity. Over longterm deployment, observations from repeated traversals can be accumulated into persistent point cloud priors that provide geometric context beyond the current observations. Existing explicit point cloud prior approaches, however, rely on LiDARbased map construction and therefore require expensive 3D ranging sensors. We propose a framework that constructs a static point cloud prior map from previous camera traversals using Pi3X and augments each point with DINOv3 features. At runtime, a local prior patch is retrieved using global localization, encoded with a sparse voxel backbone, and fused in bird’s-eye view (BEV) with lifted multi-view camera features. Task-specific sparse transformer heads then predict 3D objects and vectorized map elements from the fused representation. On Argoverse 2, the vision-based prior improves a strong baseline from 0.287 to 0.299 CDS and from 0.669 to 0.750 vectorized mapping mAP. Ablations show that semantic DINOv3 features are particularly important for vectorized mapping. These results demonstrate that vision-built geometric-semantic priors provide an effective form of long-term scene memory for camera-based perception, improving both tasks without LiDAR for prior-map construction or online inference.

## I. INTRODUCTION

Camera-based 3D object detection and online vectorized HD mapping are central to autonomous driving, as they provide the object- and map-level scene representations required for downstream prediction and planning [1]–[3]. LiDARonly and camera–LiDAR fusion methods [4]–[10] achieve strong performance by exploiting accurate 3D geometry, but require expensive ranging sensors at inference. Cameraonly approaches [1], [11]–[15] are more scalable and retain rich semantic and high-resolution visual cues, yet remain fundamentally challenged by depth ambiguity [16], [17]. This is particularly limiting for vectorized mapping, where the model must infer static map structure despite occlusions caused by traffic participants and other scene elements [9], [15], [18], [19].

During long-term deployment, autonomous vehicles repeatedly traverse the same environments including under open-set conditions [20], [21], allowing observations across multiple agents and previous traversals to be accumulated into a persistent spatial prior with broad coverage. This accumulated prior map can then be retrieved for online perception to incorporate geometric prior knowledge as a complementary source beyond the current observations. Previous work has exploited such priors for 3D occupancy prediction using neural rendering-based maps [22], [23] or global occupancy maps [24]. PreSight [22] additionally applies its reconstructed prior to vectorized mapping, but requires computationally heavy per-scene NeRF optimization. For 3D object detection, Hindsight [25] uses prior LiDAR maps for LiDAR-based perception, while AsyncDepth [26] and DualViewMapDet [27] introduce LiDAR-map priors for camera-based 3D object detection. While these detection methods demonstrate the value of previous-traversal geometry for 3D detection, existing explicit point cloud priors for camera-based methods are still constructed from LiDAR traversals.

![](images/9463c3c892dca14793993a87c0edc25c3bd852b857e843745e173b01ba6bfd83.jpg)  
Fig. 1: Camera-map perception with a vision-built prior. Given multi-view images and the current ego pose, we retrieve a local DINOv3-augmented point cloud reconstructed from earlier camera traversals. The prior map is encoded into bird’s-eye view (BEV) and fused with lifted image features. Task-specific transformer heads then predict 3D bounding boxes and vectorized HD map elements from the fused BEV features.

In this work, we investigate whether an explicit point cloud prior can instead be built from previous camera traversals without LiDAR-based mapping and benefit both 3D object detection and vectorized mapping. We construct such a prior using the feed-forward Pi3X reconstruction model [28] and augment each reconstructed point with dense DINOv3 features [29]. The resulting geometric-semantic representation combines metric scene structure with image-derived semantic features in a common 3D map. The reconstructed geometry provides context that can reduce depth ambiguity for 3D object detection, while the accumulated prior retains information about static map elements that may be occluded in the current observations.

We propose a camera-based framework that retrieves a local patch from a global vision-built prior map, encodes it with a sparse voxel backbone, and fuses it with camera

BEV features obtained through LSS lifting [30], as shown in Fig. 1. The fused representation is decoded with SparseDrivestyle transformer heads [1] for 3D object detection and online vectorized HD mapping, while perspective-view (PV) image features remain available to both heads through PV deformable aggregation. At inference, our method requires only calibrated multi-view RGB cameras, global localization, and access to the offline prior map. No LiDAR is required for map construction or online perception. Experiments on Argoverse 2 show improvements for both tasks, with particularly strong gains in vectorized mapping when the prior is enriched with DINOv3 features. These results demonstrate the potential of persistent cross-traversal priors to improve perception during subsequent visits using information accumulated over longterm deployment.

## II. TECHNICAL APPROACH

We propose a camera-based framework for joint 3D object detection and vectorized mapping that exploits an offline static point cloud prior map reconstructed from previous camera traversals (Fig. 2). At inference, the model requires calibrated multi-view RGB images, a globally localized ego pose, and retrieval of a local prior-map patch. Unlike previous prior-map methods based on LiDAR [26], [27], neither map construction nor online perception requires LiDAR sensing to enable scalable deployment.

## A. Problem Setup

For each time step t, the input consists of synchronized surround-view images $\{ I _ { t } ^ { c } \} _ { c = 1 } ^ { N }$ , their calibration parameters, and a globally localized ego pose $T _ { E _ { t } } ^ { G } \in S E ( 3 )$ . An offline global static map $\mathcal { M } ^ { G }$ is constructed from previous traversals. The model predicts oriented 3D boxes

$$
\mathcal { B } _ { t } = \{ b _ { i } ^ { t } \} _ { i = 1 } ^ { M _ { d } } , \qquad b _ { i } ^ { t } = ( x , y , z , w , h , l , \mathrm { y a w } , \mathbf { v } ) ,\tag{1}
$$

and a set of vectorized map elements represented as polylines [1]

$$
\begin{array} { r } { \mathcal { L } _ { t } = \{ l _ { i } ^ { t } \} _ { i = 1 } ^ { M _ { m } } , \qquad l _ { i } ^ { t } = \big ( ( x _ { i , k } , y _ { i , k } ) \big ) _ { k = 1 } ^ { N _ { p } } , } \end{array}\tag{2}
$$

with $N _ { p } = 2 0$ points per road boundary, lane divider, or pedestrian crossing. Following SparseDrive [1], detection is performed within a 50 m radius and mapping within a 30 m × 60 m ego-centered region.

## B. Vision-Based Prior Map Construction and Retrieval

Pi3X reconstruction.: We use Pi3X, the enhanced $\pi ^ { 3 }$ feed-forward reconstruction model [28], to build the prior map from camera data. Given a set of input images, Pi3X jointly predicts camera poses and intrinsics, local 3D point maps, and point confidences. Each local point map assigns every image pixel a 3D point in the coordinate system of the corresponding camera. We process each sequence in chunks of 12 multi-view frames and transform the local point maps from all cameras and time steps into a common global frame using the dataset-provided camera poses. The transformed point maps are then concatenated to accumulate all point clouds for a chunk.

Because the reconstruction scale can vary across chunks, we recover metric scale using the dataset trajectory. Specifically, we estimate a scale factor from the displacement between the first and last Pi3X-predicted poses and the corresponding dataset-provided global poses,

$$
\alpha = \frac { \| \mathbf { t } _ { K } ^ { G } - \mathbf { t } _ { 1 } ^ { G } \| _ { 2 } } { \| \hat { \mathbf { t } } _ { K } - \hat { \mathbf { t } } _ { 1 } \| _ { 2 } } ,\tag{3}
$$

and apply α to the predicted local depth before transforming the reconstructed points into the global coordinate frame using the dataset-provided camera poses. Hence, the Pi3Xpredicted poses are used only to recover metric scale, while the dataset poses establish the relative alignment between frames and the global reference frame.

We retain points with a Pi3X confidence score of at least 0.5, remove points closer than 0.6 m to their corresponding camera or farther than 60 m, and mask out dynamic objects using projected 3D boxes. The remaining static points are average-pooled in 0.2 m voxels and stored in a tiled global map as in DualViewMapDet [27].

DINOv3 semantic features.: Geometry alone is noisy and less informative than a semantic map prior. We therefore extract dense fine-grained DINOv3 ViT-S features [29] using the multi-scale procedure of [31]. Pretrained vision models have also been adapted to LiDAR semantic segmentation [32] and hyperspectral semantic segmentation [33]. Since the Pi3X local point map and DINOv3 feature map are image aligned, each reconstructed point is directly assigned the feature at its corresponding pixel. We fit a PCA projection on a large feature subset and compress the original 384-dimensional descriptors to 64 dimensions before storage. Each map point thus carries its 3D position, Pi3X confidence, RGB value, and compressed DINOv3 descriptor.

Online retrieval: Given the current ego translation $\mathbf { t } _ { E _ { t } } ^ { G }$ and retrieval range R, we query a local global-map patch

$$
\mathcal { P } _ { t } ^ { G } = \big \{ \mathbf { p } ^ { G } \in \mathcal { M } ^ { G } \ \big | \ \| ( p _ { x } ^ { G } , p _ { y } ^ { G } ) - ( t _ { x } ^ { G } , t _ { y } ^ { G } ) \| _ { \infty } \leq R \big \}\tag{4}
$$

To avoid leakage, points from the current traversal and temporally adjacent traversals are excluded. The retrieved points from multiple traversals are merged and transformed to the current ego frame:

$$
\mathcal { P } _ { t } ^ { E } = \left\{ ( T _ { E _ { t } } ^ { G } ) ^ { - 1 } \mathbf { p } ^ { G } ~ \middle | ~ \mathbf { p } ^ { G } \in \mathcal { P } _ { t } ^ { G } \right\} .\tag{5}
$$

## C. Camera–Map Encoding and Fusion

We extract multi-scale PV image features

$$
\mathbf { F } _ { \mathrm { i m g } } ^ { c , s } = \phi _ { \mathrm { i m g } } ^ { s } ( I _ { t } ^ { c } ) , \qquad s \in \mathcal { S } ,\tag{6}
$$

with VoVNet-99 [34]. An LSS-style lifting module [14], [30], [35] predicts per-pixel depth distributions and pools the corresponding frustum features from all cameras into F<sup>cam</sup> <sub>BEV</sub>.

For the prior map, each ego-frame point $j$ is represented by

$$
\mathbf { x } _ { j } = \left[ \mathbf { p } _ { j } ^ { E } \parallel q _ { j } \parallel \mathbf { r } _ { j } \parallel \mathbf { d } _ { j } \right] ,\tag{7}
$$

where $q _ { j }$ is Pi3X confidence, $\mathbf { r } _ { j }$ denotes RGB, and $\mathbf { d } _ { j } \in \mathbb { R } ^ { 6 4 }$ is the compressed DINOv3 descriptor. We voxelize the points

![](images/5f9c808383c551a991834e654fe0d81e78cacac6d2b11a4d0fc39b8f605e852e.jpg)  
Fig. 2: Overview of our approach. Multi-view images are encoded into perspective-view (PV) features and lifted to BEV via LSS, while a locally built vision-based point cloud prior is retrieved at the ego location and encoded by a BEV point cloud encoder. The camera and map BEV features are fused in a shared metric space. Sparse transformer heads for 3D detection and vectorized mapping then sequentially aggregate fused BEV features and multi-view PV image features.

at 0.2 m resolution, average their attributes within each voxel, and encode them using a sparse 3D backbone [6]:

$$
\mathbf { F } _ { \mathrm { B E V } } ^ { \mathrm { m a p } } = \phi _ { \mathrm { m a p } } \left( \mathrm { V o x e l i z e } ( \mathcal { P } _ { t } ^ { E } ) \right) .\tag{8}
$$

The map and camera BEV features have the same spatial resolution and are fused by channel-wise concatenation followed by a $3 \times 3$ Conv-BN-ReLU block [7], [27]:

$$
\mathbf { F } _ { \mathrm { B E V } } = { \psi _ { \mathrm { B E V } } } ( [ \mathbf { F } _ { \mathrm { B E V } } ^ { \mathrm { c a m } } \ | \mathbf { F } _ { \mathrm { B E V } } ^ { \mathrm { m a p } } ] ) ,\tag{9}
$$

followed by lightweight ResNet blocks and an FPN to refine the fused metric representation.

## D. Sparse Detection and Mapping Heads

We follow the symmetric sparse perception design of SparseDrive [1], augmented with the sequential BEV–PV deformable aggregation [11], [27]. The detection branch maintains 3D box anchors $A ^ { d } ~ \in ~ \mathbb { R } ^ { M _ { d } \times 1 1 }$ and instance features $F ^ { d } \in \mathbb { R } ^ { M _ { d } \times C }$ , while the mapping branch represents anchors as polylines $L ^ { m } \in \mathbb { R } ^ { M _ { m } \times N _ { p } \times 2 }$ with corresponding instance features $F ^ { m } \in \mathbb { R } ^ { M _ { m } \times C } \ [ 1 ]$ . In each decoder layer, both branches first aggregate the fused BEV representation and then the multi-scale PV image features:

$$
\begin{array} { r } { \mathbf { f } _ { i } ^ { \prime } = \mathbf { f } _ { i } + \mathrm { B E V - D e f o r m A g g } ( \mathbf { f } _ { i } , \mathbf { a } _ { i } , \mathbf { F } _ { \mathrm { B E V } } ) , } \\ { \mathbf { f } _ { i } ^ { \prime \prime } = \mathbf { f } _ { i } ^ { \prime } + \mathrm { P V - D e f o r m A g g } ( \mathbf { f } _ { i } ^ { \prime } , \mathbf { a } _ { i } , \{ \mathbf { F } _ { \mathrm { i m g } } ^ { c , s } \} ) . } \end{array}\tag{10}
$$

For detection, deformable keypoints are sampled from the 3D anchor box. For mapping, the polyline points serve as the geometric sampling locations. This exposes both tasks to the metric camera–map context in BEV while retaining fine-grained image information in PV.

## E. Training Objective

We train the complete network end-to-end with the standard SparseDrive detection and mapping objectives [1] and LiDAR depth supervision for the LSS module [14], [16]:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { \mathrm { m a p } } \mathcal { L } _ { \mathrm { m a p } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } .\tag{11}
$$

The prior-map encoder is optimized only through the downstream task losses. We additionally apply image–map grid masking during training as in DualViewMapDet [27] to reduce overreliance on locally available prior maps.

## III. EXPERIMENTAL EVALUATION

We evaluate our method on Argoverse 2 to quantify how vision-built prior accumulated across previous traversals affects camera-based 3D object detection and online vectorized HD mapping. We compare against strong camera-based baselines under standard evaluation protocols and ablate the contributions of prior geometry and DINOv3 semantics.

## A. Implementation and Training Details

The image backbone produces multi-scale features at strides {4, 8, 16, 32}, and the camera and map BEV branches use a shared 128 × 128 spatial grid. We lift stride-8 PV features to 480 × 704 resolution and stride-16 features for the higherresolution setting. Following our baselines, VoVNet-99 [34] is initialized from FCOS3D pretraining on nuScenes. We train for 80 epochs with AdamW, a base learning rate of $2 \times 1 0 ^ { - 4 }$ cosine decay, and a 500-iteration warm-up. The batch size is 24, and we use sequential iteration training. Image and BEV/3D augmentations follow [11]. The LSS depth estimator is supervised with LiDAR depth during training, while inference remains camera-based. We additionally employ image–map grid masking [27].

## B. Dataset and Metrics

Argoverse 2 [36] provides synchronized surround-view cameras and repeated coverage of urban areas, enabling prior-map construction from other traversals. Training maps are built exclusively from the training split. For validation, maps may contain training and validation traversals, but never the current traversal. We use the globally corrected Argoverse 2 poses from DualViewMapDet [27]. For 3D detection, we report the official Composite Detection Score (CDS), mAP, and true-positive errors mATE, mASE, and mAOE over the 26 classes. Vectorized mapping is evaluated with mAP following the protocol used by StreamMapNet [37] and MapTracker [15].

## C. 3D Object Detection and Vectorized Mapping

Tab. I compares our method with detection-only, mappingonly, and joint perception baselines. Under the directly matched 480×704 SparseDrive setting, the vision-based prior improves detection from 0.287 to 0.299 CDS and from 0.381 to 0.396 mAP, while reducing mATE from 0.709 to 0.696 and mAOE from 0.668 to 0.641. The gain is substantially larger for vectorized mapping, where mAP increases from 0.669 to 0.750. Thus, although the reconstructed map is less geometrically accurate than a LiDAR prior, its accumulated static structure and semantic features provide useful context for both object- and map-level perception.

TABLE I: 3D object detection and online vectorized HD mapping results on Argoverse 2 val set.
<table><tr><td>Method</td><td>3D Det. Vec. Map. Map Prior</td><td></td><td>Backbone</td><td>Image Size CDS↑</td><td>mAP↑</td><td>3D Object Detection mATE↓ mASE↓ mAOE↓</td><td></td><td></td><td></td><td>HD Mapping mAP↑</td></tr><tr><td>Far3D* [38]</td><td>√</td><td></td><td>VoVNet-99</td><td> $6 4 0 \times 9 6 0$ </td><td>0.281</td><td>0.367</td><td>0.730</td><td>0.300</td><td>0.531</td><td></td></tr><tr><td>Sparse4Dv3* [12]</td><td>√</td><td></td><td>VoVNet-99</td><td> $6 4 0 \times 9 6 0$ </td><td>0.294</td><td>0.381</td><td>0.706</td><td>0.314</td><td>0.666</td><td></td></tr><tr><td>Sparse4Dv3§ [12]</td><td>√</td><td></td><td>VoVNet-99</td><td> $6 4 0 \times 9 6 0$ </td><td>0.288</td><td>0.376</td><td>0.694</td><td>0.298</td><td>0.547</td><td></td></tr><tr><td>DualViewMapDet 8 [27]</td><td>√ √</td><td>LiDAR</td><td>VoVNet-99</td><td> $6 4 0 \times 9 6 0$ </td><td>0.311</td><td>0.397</td><td>0.622</td><td>0.279</td><td>0.527</td><td></td></tr><tr><td>Ours§</td><td></td><td>Vision</td><td>VoVNet-99</td><td>640 × 960</td><td>0.314</td><td>0.413</td><td>0.662</td><td>0.290</td><td>0.599</td><td>0.753</td></tr><tr><td>StreamMapNet [37]</td><td></td><td></td><td>ResNet50</td><td> $4 8 0 \times 8 0 0$ </td><td></td><td></td><td></td><td>一</td><td></td><td>0.640</td></tr><tr><td>SQD-MapNet§ [39]</td><td></td><td></td><td>ResNet50</td><td> $4 8 0 \times 8 0 0$ </td><td>一</td><td>一</td><td></td><td>一</td><td></td><td>0.633</td></tr><tr><td>MapTracker† [15]</td><td></td><td></td><td>ResNet50</td><td> $4 8 0 \times 8 0 0$ </td><td>1</td><td>1</td><td></td><td>一</td><td></td><td>0.714</td></tr><tr><td>SparseDrive§ [1] (+ BEV net. &amp; 3D/BEV data aug. [11])</td><td>√</td><td></td><td>VoVNet-99</td><td> $4 8 0 \times 7 0 4$ </td><td>0.287</td><td>0.381</td><td>0.709</td><td>0.287</td><td>0.668</td><td>0.669</td></tr><tr><td>Ours$</td><td>√</td><td>Vision</td><td>VoVNet-99</td><td> $4 8 0 \times 7 0 4$ </td><td>0.299</td><td>0.396</td><td>0.696</td><td>0.292</td><td>0.641</td><td>0.750</td></tr></table>

We evaluate object detection within a range of 50 meters, and vectorized mapping uses a 30 m × 60 m region. <sup>\*</sup>: Training uses the 10 Hz Argoverse 2 sequences by splitting each sequence into five offset subsequences, yielding ≈ 5× more (but redundant) training samples than strict 2 Hz subsampling <sup>§</sup>: Training with strict 2 Hz subsampling. <sup>†</sup>: Training with strict 2.5 Hz subsampling. <sup>‡</sup>: Baselines trained by us with the code provided by the authors.

TABLE II: Ablation study of the prior map and DINOv3 features.
<table><tr><td rowspan="2">Prior Map DINOv3 Feat.</td><td rowspan="2">All Prior| Maps</td><td>3D Object Detection</td><td>|Vectorized Mapping</td></tr><tr><td>CDS↑ mAP↑</td><td>mAP↑</td></tr><tr><td></td><td></td><td>|0.287 0.381</td><td>0.669</td></tr><tr><td>√</td><td></td><td>0.284 0.377</td><td>0.683</td></tr><tr><td>√ √</td><td></td><td>0.291 0.388</td><td>0.756</td></tr><tr><td>√</td><td></td><td>0.299 0.396</td><td>0.750</td></tr></table>

All variants use VoVNet-99 pretrained with FCOS3D at 480 × 704. The baseline is SparseDrive [1] augmented with a BEV network, deformable BEV aggregation, and 3D/BEV data augmentations [11].

TABLE III: Results on subsets with higher prior-map coverage.
<table><tr><td rowspan="2">Subset</td><td colspan="3">Ours (map prior)</td><td colspan="3">Baseline (no map)</td></tr><tr><td>CDS↑</td><td>mAP↑</td><td>mAP↑|</td><td></td><td>|CDS↑ mAP↑|</td><td>mAP↑</td></tr><tr><td>All</td><td>|0.299 (+1.2 pp)</td><td>0.396</td><td>0.750</td><td>0.287</td><td>0.381</td><td>0.669</td></tr><tr><td>Overlap k=1</td><td>0.331 (+1.9 pp)</td><td>0.436</td><td>0.505</td><td>0.312</td><td>0.409</td><td>0.440</td></tr><tr><td>Overlap k=2</td><td>0.315 (+1.8 pp)</td><td>0.414</td><td>0.378</td><td>0.296</td><td>0.391</td><td>0.328</td></tr></table>

Overlap k: a sequence is included iff every 50 m × 50 m tile it covers is also covered by at least k other sequences. The map-free baseline is evaluated on the same subsets to isolate the effect of the prior. Gains (green) are in percentage points (pp) for CDS.

## D. Ablation Studies

Prior map and DINOv3 features: Tab. II isolates the contribution of the prior representation. Geometry alone slightly improves mapping mAP from 0.669 to 0.683, but does not improve detection, indicating that noisy camerareconstructed geometry is not sufficient by itself. Augmenting the points with DINOv3 features raises detection to 0.291

CDS/0.388 mAP and mapping to 0.756 mAP. The particularly large mapping gain indicates that semantic foundation-model features provide strong cues for static structures such as lane dividers, road boundaries, and pedestrian crossings. Using prior maps from all prior traversals instead of only the two traversals with the largest point clouds per tile further improves detection to 0.299 CDS and 0.396 mAP, while mapping remains comparable at 0.750 mAP.

Map coverage: Not every validation sequence has equally strong prior coverage. We therefore evaluate subsets with increasing overlap and re-evaluate the map-free baseline on exactly the same samples in Tab. III. The prior remains beneficial on all subsets, and the CDS improvement is larger on the coverage-restricted subsets (+1.8–1.9 pp) than over the full validation set (+1.2 pp), confirming that reliable previous-traversal coverage is important for exploiting the map prior.

## IV. CONCLUSION

We presented a framework for joint camera-based 3D object detection and online vectorized HD mapping that constructs a persistent geometric-semantic point-cloud prior from previous camera traversals. Pi3X geometry is enriched with DINOv3 features, encoded in BEV, and fused with camera features before sparse task-specific decoding. Experiments on Argoverse 2 show consistent gains over a strong SparseDrivestyle baseline, with the largest improvements for vectorized mapping when semantic DINOv3 features are included. These results show that observations accumulated over long-term deployment can be retained as an explicit cross-traversal spatial prior and improve perception during subsequent visits, without requiring LiDAR for prior-map construction or online inference. In future work, we aim to improve camera– map fusion through better spatial alignment in BEV and stronger alignment of the two modalities in feature space. We will further investigate map change detection to increase robustness to outdated prior maps and to automatically identify when map updates are required.

[1] W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng, “Sparsedrive: End-to-end autonomous driving via sparse scene representation,” arXiv preprint arXiv:2405.19620, 2024.

[2] M. Cusumano-Towner, D. Hafner, A. Hertzberg, B. Huval, A. Petrenko, et al., “Robust autonomy emerges from self-play,” arXiv preprint arXiv:2502.03349, 2025.

[3] M. Büchner, J. Zürn, I.-G. Todoran, A. Valada, and W. Burgard, “Learning and aggregating lane graphs for urban automated driving,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 415–13 424.

[4] T. Yin, X. Zhou, and P. Krahenbuhl, “Center-based 3d object detection and tracking,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11 784–11 793.

[5] C. Lang, A. Braun, L. Schillingmann, and A. Valada, “A pointbased approach to efficient lidar multi-task perception,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024.

[6] Y. Yan, Y. Mao, and B. Li, “Second: Sparsely embedded convolutional detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.

[7] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. Rus, and S. Han, “Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” arXiv preprint arXiv:2205.13542, 2022.

[8] R. Mohan, D. Cattaneo, F. Drews, and A. Valada, “Progressive multimodal fusion for robust 3d object detection,” in Conference on Robot Learning, 2024.

[9] B. Liao, S. Chen, Y. Zhang, B. Jiang, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Maptrv2: An end-to-end framework for online vectorized hd map construction,” Int. Journal of Computer Vision, vol. 133, no. 3, pp. 1352–1374, 2025.

[10] R. Mohan, F. Drews, Y. Miron, D. Cattaneo, and A. Valada, “Up-fuse: Uncertainty-guided lidar-camera fusion for 3d panoptic segmentation,” Robotics: Science and Systems, 2026.

[11] M. Käppeler, Ö. Çiçek, D. Cattaneo, C. Gläser, Y. Miron, and A. Valada, “Bridging perspectives: Foundation model guided bev maps for 3d object detection and tracking,” arXiv preprint arXiv:2510.10287, 2025.

[12] X. Lin, Z. Pei, T. Lin, et al., “Sparse4d v3: Advancing end-to-end 3d detection and tracking,” arXiv preprint arXiv:2311.11722, 2023.

[13] S. Wang, Y. Liu, T. Wang, Y. Li, and X. Zhang, “Exploring objectcentric temporal modeling for efficient multi-view 3d object detection,” in International Conference on Computer Vision, 2023, pp. 3621–3631.

[14] Z. Li, S. Lan, J. M. Alvarez, and Z. Wu, “Bevnext: Reviving dense bev frameworks for 3d object detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 113–20 123.

[15] J. Chen, Y. Wu, J. Tan, H. Ma, and Y. Furukawa, “Maptracker: Tracking with strided memory fusion for consistent vector hd mapping,” in European Conference on Computer Vision, 2024, pp. 90–107.

[16] Y. Li, Z. Ge, G. Yu, J. Yang, Z. Wang, Y. Shi, J. Sun, and Z. Li, “Bevdepth: Acquisition of reliable depth for multi-view 3d object detection,” in AAAI Conference on Artificial Intelligence, 2023.

[17] R. Mohan, J. Arce, S. Mokhtar, D. Cattaneo, and A. Valada, “Synmediverse: a multimodal synthetic dataset for intelligent scene understanding of healthcare facilities,” Robotics and Automation Letters, vol. 9, no. 8, pp. 7094–7101, 2024.

[18] M. Luz, R. Mohan, A. R. Sekkat, O. Sawade, E. Matthes, T. Brox, and A. Valada, “Amodal optical flow,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 14 677–14 684.

[19] A. R. Sekkat, R. Mohan, O. Sawade, E. Matthes, and A. Valada, “Amodalsynthdrive: A synthetic amodal perception dataset for autonomous driving,” Robotics and Automation Letters, vol. 9, no. 11, pp. 9597–9604, 2024.

[20] R. Mohan, K. Kumaraswamy, J. V. Hurtado, K. Petek, and A. Valada, “Panoptic out-of-distribution segmentation,” Robotics and Automation Letters, vol. 9, no. 5, pp. 4075–4082, 2024.

[21] R. Mohan, J. Hindel, F. Drews, C. Gläser, D. Cattaneo, and A. Valada, “Open-set lidar panoptic segmentation guided by uncertainty-aware learning,” in IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2025, pp. 2224–2231.

[22] T. Yuan, Y. Mao, J. Yang, Y. Liu, Y. Wang, and H. Zhao, “Presight: Enhancing autonomous vehicle perception with city-scale nerf priors,” in European Conference on Computer Vision, 2024, pp. 323–339.

[23] M. Luz, R. Mohan, T. Nürnberg, Y. Miron, D. Cattaneo, and A. Valada, “Latent gaussian splatting for 4d panoptic occupancy tracking,” Robotics and Automation Letters, 2026.

[24] S. Yuan, J. Wei, M. Tie, X. Ren, Z. Gan, and W. Ding, “Lmpocc: 3d semantic occupancy prediction utilizing long-term memory prior from historical traversals,” arXiv preprint arXiv:2504.13596, 2025.

[25] Y. You, K. Z. Luo, X. Chen, J. Chen, W.-L. Chao, W. Sun, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Hindsight is 20/20: Leveraging past traversals to aid 3d perception,” in International Conference on Learning Representations, 2022.

[26] Y. You, C. P. Phoo, C. A. Diaz-Ruiz, K. Z. Luo, W.-L. Chao, M. Campbell, B. Hariharan, and K. Q. Weinberger, “Better monocular 3d detectors with lidar from the past,” in IEEE International Conference on Robotics and Automation, 2024, pp. 6634–6641.

[27] M. Käppeler, Ö. Çiçek, Y. Miron, and A. Valada, “Leveraging previoustraversal point cloud map priors for camera-based 3d object detection and tracking,” IEEE/RSJ International Conference on Intelligent Robots and Systems, 2026.

[28] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Permutation-equivariant visual geometry learning,” arXiv preprint arXiv:2507.13347, 2025.

[29] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[30] J. Philion and S. Fidler, “Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d,” in European Conference on Computer Vision, 2020, pp. 194–210.

[31] N. Vödisch, K. Petek, M. Käppeler, A. Valada, and W. Burgard, “A good foundation is worth many labels: Label-efficient panoptic segmentation,” Robotics and Automation Letters, vol. 10, no. 1, pp. 216–223, 2024.

[32] J. Hindel, R. Mohan, J. Bratulic, D. Cattaneo, T. Brox, and A. Valada,´ “Label-efficient lidar semantic segmentation with 2d-3d vision transformer adapters,” in IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2025, pp. 4115–4122.

[33] J. V. Hurtado, R. Mohan, and A. Valada, “Hyperspectral adapter for semantic segmentation with vision foundation models,” IEEE Robotics and Automation Letters, 2026.

[34] Y. Lee, J.-w. Hwang, S. Lee, Y. Bae, and J. Park, “An energy and gpucomputation efficient backbone network for real-time object detection,” in Proc. of the IEEE/CVF conf. on computer vision and pattern recognition workshops, 2019.

[35] R. Mohan, J. V. Hurtado, R. Mohan, and A. Valada, “Forecastocc: Vision-based semantic occupancy forecasting,” arXiv preprint arXiv:2602.08006, 2026.

[36] B. Wilson, W. Qi, T. Agarwal, J. Lambert, J. Singh, S. Khandelwal, et al., “Argoverse 2: Next generation datasets for self-driving perception and forecasting,” in Proc. of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[37] T. Yuan, Y. Liu, Y. Wang, Y. Wang, and H. Zhao, “Streammapnet: Streaming mapping network for vectorized online hd map construction,” in IEEE/CVF Winter Conf. on Applications of Computer Vision, 2024, pp. 7341–7350.

[38] “Far3d: Expanding the horizon for surround-view 3d object detection,” in AAAI Conference on Artificial Intelligence, 2024.

[39] S. Wang, F. Jia, W. Mao, Y. Liu, Y. Zhao, Z. Chen, T. Wang, C. Zhang, X. Zhang, and F. Zhao, “Stream query denoising for vectorized hd-map construction,” in European Conference on Computer Vision, 2024, pp. 203–220.