# SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection

Yuting Zhao<sup>1,2</sup>, Ziyi Zheng<sup>3</sup>, and Shuxiao Li<sup>1⋆</sup>

<sup>1</sup> Institute of Automation, Chinese Academy of Sciences, Beijing, China

School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing,

China

<sup>3</sup> Wuhan College, Wuhan, China {zhaoyuting2023,shuxiao.li}@ia.ac.cn, 542197655@qq.com

Abstract. Camera-LiDAR fusion has become a prevailing paradigm for 3D object detection in autonomous driving. However, existing fusion detectors often establish strong inter-modality dependencies by decoding object queries from tightly coupled multimodal representations. Under corrupted driving conditions, such dependencies make the detector vulnerable to unreliable modalities, where degraded observations may interfere with reliable modality-specific evidence and lead to suboptimal predictions. Moreover, modality reliability can vary across both global driving scenes and individual object queries, requiring adaptive fusion decisions at a finer granularity. To bridge this gap, we reformulate robust camera-LiDAR fusion as a scene-aware branch routing problem and propose SARFusion, a robust 3D object detector. Instead of producing detections from a single fused representation, SARFusion decouples object-query decoding into three parallel reasoning branches: a camera branch, a LiDAR branch, and a camera-LiDAR fusion branch. Guided by a Scene Reliability Prior estimated from the global driving context, SARFusion further incorporates object-level evidence to route each query to the most suitable branch. This query-wise routing strategy alleviates harmful cross-modal interference while preserving the benefits of multimodal fusion when complementary cues are trustworthy. On the nuScenes test set, SARFusion achieves strong performance with 72.5 mAP and 74.4 NDS. Extensive analyses demonstrate its robustness under challenging conditions, including sensor corruptions and environmental changes.

Keywords: 3D Detection · Camera-LiDAR Fusion · Robust Perception

## 1 Introduction

Reliable 3D object detection is a fundamental capability for autonomous driving systems. Modern autonomous vehicles perceive their surroundings with multiple complementary sensors, among which cameras and LiDAR are two of the most widely used modalities. Cameras provide dense appearance and semantic information, which is beneficial for object recognition and distant-object perception. LiDAR directly measures depth and 3D geometry, providing accurate localization cues. By combining these complementary signals, camera-LiDAR fusion detectors have achieved strong performance on standard 3D detection benchmarks. [2, 9, 50]

Despite this progress, robust multi-modal perception remains challenging in complex driving environments. Sensor observations can be degraded by adverse weather [3], illumination changes, motion blur, temporal inconsistency, spatial misalignment, camera failures, LiDAR beam dropping, incomplete echoes, and other corruptions [1,3,4,11,14,32]. Under such conditions, the reliability of each modality can change significantly. A modality that provides useful evidence in clean scenes may become noisy or misleading when degraded. If a detector always follows the same fusion pathway, unreliable modality information can be injected into the final prediction, leading to negative transfer and reduced robustness.

A common limitation of existing fusion detectors is that they treat fusion behavior as largely fixed [2, 30]. This design implicitly assumes that the same sensor combination is suitable for all driving scenes. However, modality reliability is strongly scene-dependent. Weather, illumination, and sensor status can change the overall usefulness of camera and LiDAR cues. For example, low-light or adverse weather may weaken visual observations, while LiDAR corruptions can reduce geometric reliability. This motivates the need for scene-aware reliability modeling rather than condition-agnostic fusion.

Moreover, modality reliability is not only scene-dependent but also objectdependent. In the same frame, diferent objects may have diferent observation quality due to distance, occlusion, visibility, and point density [11, 22, 46]. A distant pedestrian may have sparse LiDAR returns but still preserve recognizable image semantics, while a nearby vehicle may be better localized by LiDAR geometry. Therefore, a single scene-level fusion decision is insuficient: forcing all objects in a frame to use the same sensor combination may still propagate locally degraded evidence. Since object queries naturally represent candidate instances in transformer-based 3D detectors, query-level branch selection provides an appropriate granularity for reliable fusion [2, 27, 45, 48]..

Motivated by these observations, we propose SARFusion, a scene-aware branch routing framework for robust camera-LiDAR 3D object detection. SARFusion explicitly maintains three candidate branches: a camera branch, a LiDAR branch, and a camera-LiDAR fusion branch. These branches provide diferent sensing paths for object queries under varying sensor conditions. To guide branch selection, SARFusion first estimates a Scene Reliability Prior from global driving conditions such as weather and time. During training, the scene prior is aligned with textual scene prompts [34]. It then performs Scene-Aware Branch Routing, which combines the scene prior with local feature evidence around each object query to select a reliable branch. This design allows diferent objects in the same scene to rely on diferent sensor combinations, reducing negative transfer from degraded modalities while preserving useful cross-modal complementarity. We evaluate SARFusion on clean and corrupted autonomous-driving benchmarks.

In nuScenes [6], SARFusion achieves 71.1 mAP and 73.7 NDS on the validation split, and 72.5 mAP and 74.4 NDS on the test split. Under adverse nuScenes-C conditions [11, 16], SARFusion improves detection performance under adverse weather and illumination changes, including fog, snow, rain and strong sunlight. Ablation studies verify the efectiveness of our core components.

Our main contributions are summarized as follows:

1. We propose SARFusion, a scene-aware branch routing framework for robust camera-LiDAR 3D object detection, which dynamically selects among camera, LiDAR, and camera-LiDAR fusion branches instead of relying on a fixed fusion pathway.

2. We introduce a Scene Reliability Prior to model scene-dependent modality reliability from global driving conditions, providing condition-aware guidance under degraded environments.

3. We design Scene-Aware Branch Routing, which combines the scene reliability prior with local object-level evidence to perform query-level branch selection.

4. SARFusion achieved SOTA performance among existing camera-LiDAR fusion methods under various sensor failure conditions and extreme weather scenarios.

5. The code will be publicly available.

## 2 Related Work

## 2.1 3D Object Detection

3D object detection estimates object categories and 3D bounding boxes from onboard sensor observations. Camera-based methods infer 3D layouts from monocular, stereo, or surround-view images. Early methods rely on geometric priors, keypoints, or explicit depth estimation [5,8,29,43]. Recent surround-view detectors lift image features into 3D or bird’s-eye-view (BEV) space [15, 25, 26], or use 3D queries to aggregate multi-view image features [27, 45]. These methods benefit from dense appearance cues, but remain limited by depth ambiguity, especially for distant, occluded, or weakly textured objects.

LiDAR-based detectors exploit point clouds with accurate depth and geometry. Point-based methods operate on raw point sets [33, 37], while voxeland pillar-based methods convert point clouds into regular representations for eficient feature extraction [23, 49, 51, 53]. Hybrid point-voxel methods further balance geometric precision and computational eficiency [36]. However, LiDAR points are sparse at long range and provide limited semantic appearance. They can also be degraded by adverse weather, beam sparsity, incomplete echoes, and sensor malfunction. These complementary limitations motivate camera-LiDAR fusion for more accurate and reliable 3D detection.

## 2.2 Multi-modal Sensor Fusion

Camera-LiDAR fusion combines dense image semantics with accurate LiDAR geometry. Point-level methods project image semantics or features onto LiDAR points to enrich point-cloud representations [40, 41, 52]. They improve detection performance, but usually depend on accurate calibration and are sensitive to projection errors, spatial misalignment, and LiDAR degradation.

Recent methods fuse modalities in intermediate feature spaces. BEV-based methods transform camera and LiDAR features into a shared BEV representation [30, 42], which provides an eficient fusion space but still relies on depth estimation and geometric projection. Other methods explore unified voxel representations, deformable feature alignment, and sparse multi-sensor representations [9, 10, 24, 47]. Transformer-based methods use object queries or modality tokens for cross-modal interaction [2,48,50]. TransFusion refines LiDAR proposals with image features, DeepInteraction performs iterative modality interaction, and CMT uses coordinate embeddings for implicit alignment. These methods achieve strong clean-set performance, but most of them use fixed fusion pathways or predefined interaction patterns. They rarely adapt the fusion behavior to scene-dependent and object-dependent modality reliability.

## 2.3 Robust Multi-Modality Fusion

Robust multi-modal fusion has attracted increasing attention because 3D detectors can degrade severely under adverse weather, motion blur, temporal inconsistency, spatial misalignment, missing cameras, beam dropping, and incomplete echoes [3, 17, 18, 22, 46]. These corruptions change modality reliability at diferent granularities. Weather and illumination often afect the whole scene, while occlusion, distance, point density, and local sensor failures may afect individual objects within the same frame.

Existing methods improve robustness through data augmentation, modality dropout, decoupled representations, or adaptive fusion strategies [7, 12, 13, 16, 31,35,38,42,48]. CMT adopts masked-modal training to handle missing modalities [48]. MEFormer reduces negative fusion with modality-agnostic decoding and prediction ensemble [7]. Recent expert-based methods further explore routing or selecting modality-specific experts under sensor failures [31]. These studies suggest that reducing rigid inter-modality dependence is important for robust perception.

However, many existing robust fusion methods mainly focus on predefined corruptions, modality-level failures, or fixed robustness training strategies. Less attention has been paid to jointly modeling global scene reliability and querylevel local observation quality. In realistic driving scenes, the reliable modality may vary not only with weather or illumination, but also across diferent objects due to distance, visibility, occlusion, and point density [20,21,39,44]. Motivated by this observation, SARFusion formulates robust camera-LiDAR fusion as a scene-aware branch routing problem. It estimates a global scene reliability prior and combines it with local query evidence to select camera, LiDAR, or fusion branches for each object query [19, 28].

![](images/043aaa4581c9efa7784864b03e7b683eb91a48226447540f2311f4c45a068175.jpg)  
Fig. 1: Overall architecture of the proposed SARFusion.

## 3 Method

## 3.1 Overview

The overall framework of SARFusion is shown in Fig. 1. Given multi-view camera images $\mathcal { T } = \{ I _ { v } \} _ { v = 1 } ^ { V }$ and a LiDAR point cloud P, SARFusion first extracts camera features and LiDAR features using modality-specific encoders. The camera features are flattened into image tokens $\mathbf { X } _ { C } \in \dot { \mathbb { R } } ^ { N _ { C } \times D }$ , and the LiDAR features are encoded into BEV tokens $\mathbf { X } _ { L } \in \mathbb { R } ^ { N _ { L } \times D }$ . We further concatenate them as multimodal context tokens $\mathbf { X } _ { C L } = [ \mathbf { X } _ { C } ; \mathbf { X } _ { L } ]$

Instead of producing a single fused representation, SARFusion explicitly maintains three candidate decoding branches: a camera branch, a LiDAR branch, and a Camera-LiDAR fusion branch. These branches provide complementary representation choices under diferent sensing conditions. The camera branch decodes object queries using camera tokens, the LiDAR branch uses LiDAR tokens, and the fusion branch attends to multimodal tokens. To adaptively select reliable branches, SARFusion first learns a Scene Reliability Prior from multimodal context to encode global sensing conditions. Then, for each object query, a Scene-Aware Branch Router combines this scene prior with local observation evidence around the query reference point and predicts branch probabilities over the three decoding branches. The query is routed to the selected branch for final decoding and prediction.

## 3.2 Scene Reliability Prior

The reliability of camera and LiDAR observations is highly correlated with scene-level conditions, such as weather, illumination, sensor degradation, and temporal-spatial disturbance. Therefore, we introduce a Scene Reliability Prior to provide global reliability guidance for branch routing.

![](images/91472cb261f6b7026146689873081bf519037389233c4e8320b03d5f56e2253d.jpg)  
Fig. 2: The illustration of SRP module

Given multimodal context tokens $\mathbf { X } _ { C L }$ , we use a lightweight Transformer to aggregate scene-level information. Specifically, a learnable scene token $\mathbf { q } _ { s }$ is prepended to the multimodal tokens, and the output scene token is projected to obtain the scene reliability prior:

$$
{ \bf c } _ { p r i o r } = \mathrm { L N } \left( \mathrm { M L P } \left( \mathrm { T r a n s f o r m e r } ( \left[ { \bf q } _ { s } ; { \bf X } _ { C L } \right] ) _ { 0 } \right) \right) ,\tag{1}
$$

where $\mathbf { c } _ { p r i o r } \in \mathbb { R } ^ { D }$ summarizes the global sensing condition of the current scene. To encourage the scene prior to encode meaningful reliability-related information, we supervise it with textual scene prompts during training. For each training scene, a prompt describing the scene condition is constructed, e.g.,

$$
{ } ^ { \mathrm { { } ^ { \mathrm { { c } } A } \ \{ \mathrm { { w e a t h e r \ c o n d i t i o n } \} \ \mathrm { d r i v i n g \ s c e n e \ a t \ \{ t i m e \ o f \ d a y } \} , } } } ^  \mathrm { { } ^ { \mathrm { { * } } } A \ \{ \mathrm { { n e a t h e r \ c o n d i t i o n } \} \mathrm { d r i v i n g \ s c e n e \ a t \ \{ t i m e \ o f \ d a y } \} . } \mathrm { { , } ^ { \mathrm { { * } } } } }\tag{2}
$$

The prompt is encoded by a text encoder into a text embedding t. We then apply a symmetric contrastive loss between the scene prior and the text embedding:

$$
\mathcal { L } _ { s r p } = \mathcal { L } _ { c  t } + \mathcal { L } _ { t  c } .\tag{3}
$$

This supervision encourages ${ \bf c } _ { p r i o r }$ to preserve scene-level reliability cues. During inference, no text prompt or scene label is required; the scene prior is directly inferred from multimodal features.

## 3.3 Scene-Aware Branch Routing

While the Scene Reliability Prior captures global sensing conditions, modality reliability can also vary across object regions [11, 22, 46]. For example, LiDAR points may be sparse for distant objects, while camera observations may be locally occluded or corrupted. Therefore, SARFusion performs branch routing at the object-query level.

For the i-th object query $\mathbf { q } _ { i }$ with reference point $\mathbf { p } _ { i } ^ { r e f } = ( x _ { i } , y _ { i } , z _ { i } )$ , we project the reference point onto the camera feature plane and the LiDAR BEV plane.

![](images/3262913369515f6e32d0b60500cf3323b2c71d44c95f068ae30afc44a3e38f03.jpg)  
Fig. 3: The illustration of SABR module.

Around each projected location, we construct a local attention window to collect nearby modality features. The union of camera and LiDAR local tokens forms a query-specific local region $\varOmega _ { i }$

We use a local attention mask $\mathbf { M } _ { i }$ to restrict attention to $\varOmega _ { i }$ :

$$
\mathbf { M } _ { i } ( n ) = \left\{ \begin{array} { l l } { 0 , } & { n \in \varOmega _ { i } , } \\ { - \infty , } & { n \notin \varOmega _ { i } . } \end{array} \right.\tag{4}
$$

The local observation evidence for query $\mathbf { q } _ { i }$ is then computed by masked attention:

$$
\mathbf { e } _ { i } ^ { l o c a l } = \mathrm { S o f t m a x } \left( \frac { \mathbf { q } _ { i } \mathbf { K } ^ { \top } } { \sqrt { D } } + \mathbf { M } _ { i } \right) \mathbf { V } ,\tag{5}
$$

where K and V are projected from multimodal context tokens $\mathbf { X } _ { C L }$

The branch router combines local observation evidence with the global scene prior:

$$
\mathbf { h } _ { i } = [ \mathbf { e } _ { i } ^ { l o c a l } ; \mathbf { c } _ { p r i o r } ] ,\tag{6}
$$

and predicts branch probabilities:

$$
\pi _ { i } = \mathrm { S o f t m a x } \left( \mathrm { M L P } ( \mathbf { h } _ { i } ) \right) , \quad \pi _ { i } = [ \pi _ { i , C } , \pi _ { i , L } , \pi _ { i , F } ] ,\tag{7}
$$

where $C , L$ , and $F$ denote the camera, LiDAR, and fusion branches, respectively. The selected branch is:

$$
m _ { i } ^ { * } = \arg \operatorname* { m a x } _ { m \in \{ C , L , F \} } \pi _ { i , m } .\tag{8}
$$

Accordingly, object queries are partitioned into three groups:

$$
\mathcal { Q } _ { m } = \{ \mathbf { q } _ { i } \ | \ m _ { i } ^ { * } = m \} , \quad m \in \{ C , L , F \} .\tag{9}
$$

Each group is decoded by its corresponding branch. This design allows diferent object queries in the same scene to select diferent reliable representations.

## 3.4 Training Objective

Training SARFusion involves three objectives: branch-wise detection supervision, scene prior supervision, and routing supervision.

First, to ensure that each branch has independent detection ability, we apply detection losses to all three branches:

$$
\mathcal { L } _ { d e t } = \mathcal { L } _ { d e t } ^ { C } + \mathcal { L } _ { d e t } ^ { L } + \mathcal { L } _ { d e t } ^ { F } ,\tag{10}
$$

where each detection loss consists of classification and box regression terms.

Second, the Scene Reliability Prior is supervised by the contrastive sceneprompt loss $\mathcal { L } _ { s r p }$ defined above, which encourages the prior to encode global sensing conditions.

Third, we supervise the branch router with reliability-aware routing targets. During training, modality degradation or dropout is applied to simulate unreliable sensing conditions. The routing target encourages queries to select the branch corresponding to the reliable modality: the LiDAR branch when camera observations are degraded, the camera branch when LiDAR observations are unreliable, and the fusion branch when both modalities are informative. The routing loss is:

$$
\mathcal { L } _ { r o u t e } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { m \in \{ C , L , F \} } y _ { i , m } \log \pi _ { i , m } ,\tag{11}
$$

where $y _ { i , m }$ is the routing target for query $\mathbf { q } _ { i }$ .

The full training objective is:

$$
\mathcal { L } = \lambda _ { d e t } \mathcal { L } _ { d e t } + \lambda _ { s r p } \mathcal { L } _ { s r p } + \lambda _ { r o u t e } \mathcal { L } _ { r o u t e } .\tag{12}
$$

In practice, we adopt a staged training strategy. We first train the three decoding branches with branch-wise detection supervision, then learn the Scene Reliability Prior with prompt-based contrastive supervision, and finally optimize the router with reliability-aware routing supervision. This strategy stabilizes branch learning and prevents the router from degenerating into a fixed branch preference.

## 4 Experiments

## 4.1 Experimental Setup

Dataset and metrics. We evaluate SARFusion on the nuScenes 3D object detection benchmark, which provides synchronized multi-view camera images and LiDAR point clouds for autonomous driving scenes. Following the oficial protocol, the dataset is divided into 700 training scenes, 150 validation scenes, and 150 test scenes. We report the oficial mean Average Precision (mAP) and nuScenes Detection Score (NDS). mAP measures localization and classification accuracy, while NDS further summarizes multiple detection quality factors, including translation, scale, orientation, velocity, and attribute errors.

![](images/4d64ec94a9dc2d98e3f70052af8acc9a077451baa984a9a9a71d74c040504bd5.jpg)  
Fig. 4: Comparison of camera-LiDAR observations under degraded sensing conditions.

Robustness evaluation. Besides the standard clean validation and test sets, we further evaluate diferent methods under representative degraded sensing conditions, including snow, rain, fog, and strong sunlight. These conditions afect camera and LiDAR observations in diferent ways. Snow, rain, and fog may weaken image visibility and disturb point-cloud measurements, while strong sunlight mainly changes image appearance and reduces the reliability of visual cues. Unless otherwise specified, all models are trained on the original training set and directly evaluated under degraded validation conditions without conditionspecific fine-tuning. This protocol is used to examine whether the fusion strategy itself can adapt to scene-dependent sensing reliability.

Implementation details. SARFusion follows a multi-branch camera-LiDAR detection framework. The camera stream extracts image features from surroundingview images, and the LiDAR stream encodes point clouds into BEV features. Based on these representations, SARFusion maintains three candidate prediction branches: a camera branch, a LiDAR branch, and a camera-LiDAR fusion branch. The Scene Reliability Prior estimates global scene-level sensing conditions, while the Scene-Aware Branch Routing module combines this prior with local query evidence to select a suitable branch for each object query. All ablation experiments are conducted on the nuScenes validation split. We adopt a staged training strategy: the detection branches are first optimized to provide valid modality-specific and fused representations; the scene prior is then learned with scene-level supervision; finally, the routing module is jointly optimized with the detection objective.

## 4.2 Comparison with SOTA Methods

Results on the clean benchmark. Table 1 compares SARFusion with representative camera-only, LiDAR-only, and camera-LiDAR fusion methods on the standard nuScenes validation and test sets. Camera-only methods are limited by the lack of explicit 3D geometry, while LiDAR-only methods provide stronger localization due to accurate depth and structure. Multi-modal methods further improve detection by combining image semantics with LiDAR geometry.

Table 1: Comparison with representative 3D object detection methods on the nuScenes validation and test sets. All results are reported in percentage.
<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>mAP</td><td>NDS</td><td>mAP</td><td>NDS</td></tr><tr><td rowspan="3">Camera</td><td>FCOS3D [43]</td><td>34.3</td><td>41.5</td><td>35.8</td><td>42.8</td></tr><tr><td>PETR [27]</td><td>37.0</td><td>44.2</td><td>39.1</td><td>45.5</td></tr><tr><td>BEVDet [15]</td><td></td><td></td><td>42.2</td><td>48.2</td></tr><tr><td rowspan="3">LiDAR</td><td>SECOND [49]</td><td>52.6</td><td>63.0</td><td>52.8</td><td>63.3</td></tr><tr><td>CenterPoint [51]</td><td>59.6</td><td>66.8</td><td>60.3</td><td>67.3</td></tr><tr><td>TransFusion-L [2]</td><td>65.1</td><td>70.1</td><td>65.5</td><td>70.2</td></tr><tr><td rowspan="9">Camera + LiDAR</td><td>FUTR3D [9]</td><td>64.5</td><td>68.3</td><td></td><td></td></tr><tr><td>PointAugmenting [41]</td><td></td><td></td><td>66.8</td><td>71.0</td></tr><tr><td>UVTR [24]</td><td>65.4</td><td>70.2</td><td>67.1</td><td>71.1</td></tr><tr><td>AutoAlignV2 [10]</td><td>67.1</td><td>71.2</td><td>68.4</td><td>72.4</td></tr><tr><td>TransFusion [2]</td><td>67.5</td><td>71.3</td><td>68.9</td><td>71.6</td></tr><tr><td>MetaBEV [13]</td><td>68.0</td><td>71.5</td><td></td><td></td></tr><tr><td>BEVFusion [30]</td><td>68.5</td><td>71.4</td><td>70.2</td><td>72.9</td></tr><tr><td>DeepInteraction [50]</td><td>69.9</td><td>72.6</td><td>70.8</td><td>73.4</td></tr><tr><td>SparseFusion [47]</td><td>70.4</td><td>72.8</td><td>72.0</td><td>73.8</td></tr><tr><td>CMT [48] Camera + LiDAR SARFusion</td><td></td><td>70.3</td><td>72.9</td><td>72.0</td><td>74.1</td></tr></table>

SARFusion achieves 71.1 mAP and 73.7 NDS on the validation set, and 72.5 mAP and 74.4 NDS on the test set. Compared with representative camera-LiDAR fusion methods, SARFusion obtains strong performance on clean scenes. The improvement on clean data is moderate but meaningful, since most clean samples already benefit from conventional fused representations. This indicates that the proposed scene-aware routing strategy improves robustness without sacrificing standard detection accuracy.

Results under degraded conditions. Table 2 reports the robustness comparison under representative weather and lighting conditions. Compared with clean scenes, degraded scenes introduce stronger modality imbalance. Visual appearance may become unreliable under rain, fog, or strong sunlight, while LiDAR observations can become sparse or noisy in adverse weather. These corruptions make tightly coupled fusion vulnerable, since degraded observations may interfere with reliable modality-specific evidence.

SARFusion achieves the best performance under all evaluated clean and degraded conditions. Compared with the strongest competing results, SARFusion improves mAP by 0.8 on clean scenes, 1.3 under snow, 3.2 under rain, 7.8 under fog, and 2.0 under strong sunlight. The advantage becomes more evident under degraded conditions, supporting our motivation that robust camera-LiDAR fusion should adapt to scene-level and object-level reliability rather than relying on a single fixed fused representation.

![](images/c41681f9bbd9ad86b541885f153e85a3e66791a6f43fa2a3d4174276392ec9db.jpg)  
Fig. 5: Qualitative detection results under representative adverse weather conditions, including fog, snow, and rain.

## 4.3 Ablation Study

Efectiveness of each component. Table 3 studies the contribution of the main components in SARFusion. The single fusion branch baseline already achieves strong performance, since it benefits from both camera semantics and LiDAR geometry. However, directly introducing multiple branches without an efective routing mechanism leads to a significant performance drop. This shows that simply adding modality-specific branches is not suficient. Without query-level routing, the detector cannot determine which representation should be responsible for each object, and the branch predictions are not properly coordinated.

After introducing Scene-Aware Branch Routing, the performance recovers to the level of the single fusion branch baseline. This result indicates that querylevel branch selection is essential for making the multi-branch design efective. Finally, adding the Scene Reliability Prior further improves both mAP and NDS, showing that global scene reliability provides useful context for object-level routing decisions.

Efect of scene prior supervision. Table 4 evaluates the efect of supervising the Scene Reliability Prior. Without scene-level supervision, the prior is learned only through the final detection objective. In this case, the scene representation may encode information useful for detection, but it is not explicitly encouraged to describe global sensing reliability.

Table 2: Robustness comparison on the nuScenes validation set under representative weather and lighting conditions.
<table><tr><td>Method</td><td>Clean</td><td>Snow</td><td>Rain</td><td>Fog</td><td>Strong Sunlight</td></tr><tr><td>FCOS3D [43]</td><td>23.9</td><td>2.0</td><td>13.0</td><td>13.5</td><td>17.2</td></tr><tr><td>DETR3D [45]</td><td>34.7</td><td>5.1</td><td>20.4</td><td>27.9</td><td>34.7</td></tr><tr><td>PointPillars [23]</td><td>27.7</td><td>27.6</td><td>27.7</td><td>24.5</td><td>23.7</td></tr><tr><td>CenterPoint [51]</td><td>59.3</td><td>55.9</td><td>56.1</td><td>43.8</td><td>54.2</td></tr><tr><td>BEVFusion [30]]</td><td>68.5</td><td>62.8</td><td>66.1</td><td>54.1</td><td>64.4</td></tr><tr><td>TransFusion [2]</td><td>66.4</td><td>63.3</td><td>65.4</td><td>53.7</td><td>55.1</td></tr><tr><td>DeepInteraction [50]</td><td>69.9</td><td>62.3</td><td>66.5</td><td>54.8</td><td>64.9</td></tr><tr><td>FUTR3D [9]</td><td>64.2</td><td>52.7</td><td>58.4</td><td>53.2</td><td>57.7</td></tr><tr><td>CMT [48]</td><td>70.3</td><td>63.5</td><td>62.6</td><td>61.4</td><td>66.3</td></tr><tr><td>SARFusion</td><td>71.1</td><td>64.8</td><td>69.7</td><td>69.2</td><td>68.3</td></tr></table>

Table 3: Component ablation of SARFusion on the nuScenes validation set.
<table><tr><td>Setting</td><td>mAP</td><td>NDS</td></tr><tr><td>Single fusion branch</td><td>70.7</td><td>73.1</td></tr><tr><td>Multi-branch w/o routing</td><td>52.4</td><td>62.8</td></tr><tr><td>+ Scene-Aware Branch Routing</td><td>70.8</td><td>73.1</td></tr><tr><td>+ Scene Reliability Prior</td><td>71.1</td><td>73.7</td></tr></table>

With scene prior supervision, SARFusion achieves higher mAP and NDS. This improvement shows that explicitly aligning the scene prior with global driving conditions makes the routing decision more stable. The prior does not replace local evidence; instead, it calibrates local query evidence with scene-level reliability context.

Efect of routing evidence. Table 5 compares diferent inputs for branch routing. Using only local query evidence already provides strong performance, because it captures object-level cues such as local image visibility, point density, and feature consistency around the query reference point. This is important for autonomous driving scenes, where degradation is often spatially non-uniform.

Combining local evidence with the Scene Reliability Prior gives the best result. The two types of evidence are complementary: local evidence describes the reliability of a specific object query, while the scene prior describes the global sensing condition shared by the whole frame. Their combination enables SAR-Fusion to make object-specific routing decisions under a scene-aware context.

Table 4: Efect of scene prior supervision on the nuScenes validation set.
<table><tr><td>Setting</td><td> $\mathrm { m A P }$ </td><td>NDS</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  scene prior supervision</td><td>69.6</td><td>72.1</td></tr><tr><td> $\mathrm { w } / $  scene prior supervision</td><td>71.1</td><td>73.7</td></tr></table>

Table 5: Ablation of routing evidence on the nuScenes validation set.
<table><tr><td>Routing Evidence</td><td> $\mathrm { m A P }$ </td><td>NDS</td></tr><tr><td>Local evidence only</td><td>70.8</td><td>73.1</td></tr><tr><td>Scene prior + local evidence</td><td>71.1</td><td>73.7</td></tr></table>

Efect of training strategy. Table 6 compares direct end-to-end training with the staged training strategy. Direct training gives much lower performance, indicating that joint optimization of branch specialization, scene prior learning, and routing selection is dificult from scratch. In the early training stage, the router may make unstable choices before each branch becomes a reliable detector, which further weakens branch learning and may lead to routing collapse.

The staged strategy substantially improves performance. By first training the candidate branches, each branch learns a meaningful representation for detection. The scene prior is then learned as a reliability-aware condition signal. Finally, the routing module is optimized on top of already informative branches and a stable scene prior.

## 4.4 Qualitative Analysis

Routing behavior under diferent scenes. To further understand how SAR-Fusion makes routing decisions, we analyze the distribution of selected branches under diferent sensing conditions. As shown in Figure 6, the fusion branch dominates in clean scenes, which is consistent with the strong performance of conventional fusion methods under normal conditions. This confirms that SARFusion does not unnecessarily avoid fusion when both modalities are reliable.

When one modality becomes unreliable, the routing distribution changes accordingly. Under LiDAR failure, most queries are routed to the camera branch, while under camera failure, most queries are routed to the LiDAR branch. Under adverse weather or illumination, the model does not simply switch to a single modality. Instead, it keeps a considerable portion of queries in the fusion branch while increasing the use of modality-specific branches. This behavior indicates that SARFusion learns a reliability-aware routing policy rather than a fixed modality preference.

Detection examples. Figure 5 presents qualitative comparisons under representative adverse weather conditions, including rain, fog, and snow. These conditions degrade image visibility and may disturb point-cloud observations, making tightly coupled camera-LiDAR fusion vulnerable to unreliable modality features. Compared with BEVFusion and CMT, SARFusion produces more complete and accurately localized detection results across the degraded scenes.

Table 6: Efect of training strategy on the nuScenes validation set.
<table><tr><td>Training Strategy</td><td>mAP</td><td>NDS</td></tr><tr><td>Direct training</td><td>65.7</td><td>69.6</td></tr><tr><td>Staged training</td><td>71.1</td><td>73.7</td></tr></table>

![](images/a03b42d2b8f9f0ac783e886882a3db7ba68204dd60200fef12fa82df5da2c518.jpg)  
Fig. 6: Branch routing visualization under diferent sensing conditions.

In particular, competing fusion methods may miss objects or generate less precise boxes when corrupted observations interfere with reliable modality-specific evidence. By routing object queries to camera, LiDAR, or fusion branches according to scene-level and object-level reliability, SARFusion alleviates harmful cross-modal interference while preserving useful complementary cues. These qualitative results further support the robustness of scene-aware routing fusion under adverse weather conditions.

## 5 Conclusions

We introduced SARFusion, a scene-aware branch routing framework for robust camera-LiDAR 3D object detection. Unlike fixed fusion methods, SARFusion maintains camera, LiDAR, and fusion branches, and combines a global Scene Reliability Prior with local query evidence to select reliable sensing paths for each object. This design reduces the influence of degraded modalities while preserving complementary semantic and geometric cues. Experiments on nuScenes and corrupted driving scenarios demonstrate the efectiveness of SARFusion: it achieves 72.5 mAP and 74.4 NDS on the nuScenes test set and consistently improves robustness under snow, rain, fog, and strong sunlight. Ablations further validate the benefits of scene reliability modeling and query-level routing.

## References

1. Albreiki, F., Abughazal, S., Lahoud, J., Anwer, R., Cholakkal, H., Khan, F.: On the robustness of 3d object detectors (2022), https://arxiv.org/abs/2207.10205

2. Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., Tai, C.L.: Transfusion: Robust lidar-camera fusion for 3d object detection with transformers (2022), https://arxiv.org/abs/2203.11496

3. Beemelmanns, T., Zhang, Q., Geller, C., Eckstein, L.: Multicorrupt: A multi-modal robustness dataset and benchmark of lidar-camera fusion for 3d object detection (2024), https://arxiv.org/abs/2402.11677

4. Bijelic, M., Gruber, T., Mannan, F., Kraus, F., Ritter, W., Dietmayer, K., Heide, F.: Seeing through fog without seeing fog: Deep multimodal sensor fusion in unseen adverse weather (2020), https://arxiv.org/abs/1902.08913

5. Brazil, G., Liu, X.: M3d-rpn: Monocular 3d region proposal network for object detection (2019), https://arxiv.org/abs/1907.06038

6. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving (2020), https://arxiv.org/abs/1903.11027

7. Cha, J., Joo, M., Park, J., Lee, S., Kim, I., Kim, H.J.: Robust multimodal 3d object detection via modality-agnostic decoding and proximity-based modality ensemble (2024), https://arxiv.org/abs/2407.19156

8. Chen, X., Kundu, K., Zhang, Z., Ma, H., Fidler, S., Urtasun, R.: Monocular 3d object detection for autonomous driving. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)

9. Chen, X., Zhang, T., Wang, Y., Wang, Y., Zhao, H.: Futr3d: A unified sensor fusion framework for 3d detection (2023), https://arxiv.org/abs/2203.10642

10. Chen, Z., Li, Z., Zhang, S., Fang, L., Jiang, Q., Zhao, F.: Autoalignv2: Deformable feature aggregation for dynamic multi-modal 3d object detection (2022), https: //arxiv.org/abs/2207.10316

11. Dong, Y., Kang, C., Zhang, J., Zhu, Z., Wang, Y., Yang, X., Su, H., Wei, X., Zhu, J.: Benchmarking robustness of 3d object detection to common corruptions in autonomous driving (2023), https://arxiv.org/abs/2303.11040

12. Drews, F., Feng, D., Faion, F., Rosenbaum, L., Ulrich, M., Gläser, C.: Deepfusion: A robust and modular 3d object detector for lidars, cameras and radars (2022), https://arxiv.org/abs/2209.12729

13. Ge, C., Chen, J., Xie, E., Wang, Z., Hong, L., Lu, H., Li, Z., Luo, P.: Metabev: Solving sensor failures for bev detection and map segmentation (2023), https: //arxiv.org/abs/2304.09801

14. Hahner, M., Sakaridis, C., Dai, D., Gool, L.V.: Fog simulation on real lidar point clouds for 3d object detection in adverse weather (2021), https://arxiv.org/ abs/2108.05249

15. Huang, J., Huang, G., Zhu, Z., Ye, Y., Du, D.: Bevdet: High-performance multicamera 3d object detection in bird-eye-view (2022), https://arxiv.org/abs/ 2112.11790

16. Huang, Y., Yu, K., Guo, Q., Juefei-Xu, F., Jia, X., Li, T., Pu, G., Liu, Y.: Improving robustness of lidar-camera fusion model against weather corruption from fusion strategy perspective (2024), https://arxiv.org/abs/2402.02738

17. Ji, Y., Chi, C., Tan, H., Zhao, Y., Lyu, H., Zhou, E., Wang, P., Wang, Z., Zheng, X., Zhang, S.: From ill-posed to well-posed: A decomposition framework for reliable robot learning with foundation models. IEEE Transactions on Robotics (2026)

18. Ji, Y., Liu, Y., Zhang, Z., Zhang, Z., Zhao, Y., Hao, X., Zhou, G., Zhang, X., Zheng, X.: Enhancing adversarial robustness of vision-language models through low-rank adaptation. In: Proceedings of the 2025 International Conference on Multimedia Retrieval. pp. 550–559 (2025)

19. Ji, Y., Liu, Y., Tan, H., Huang, X., Huang, F., Xu, Y., Chi, C., Zhao, Y., Lyu, H., Co, P., et al.: Prm-as-a-judge: A dense evaluation paradigm for fine-grained robotic auditing. arXiv preprint arXiv:2603.21669 (2026)

20. Ji, Y., Tan, H., Chi, C., Xu, Y., Zhao, Y., Zhou, E., Lyu, H., Wang, P., Wang, Z., Zhang, S., et al.: Mathsticks: A benchmark for visual symbolic compositional reasoning with matchstick puzzles. arXiv preprint arXiv:2510.00483 (2025)

21. Ji, Y., Wang, Y., Liu, Y., Hao, X., Liu, Y., Zhao, Y., Lyu, H., Zheng, X.: Visualtrans: A benchmark for real-world visual transformation reasoning. arXiv preprint arXiv:2508.04043 (2025)

22. Kong, L., Liu, Y., Li, X., Chen, R., Zhang, W., Ren, J., Pan, L., Chen, K., Liu, Z.: Robo3d: Towards robust and reliable 3d perception against corruptions (2023), https://arxiv.org/abs/2303.17597

23. Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: Pointpillars: Fast encoders for object detection from point clouds (2019), https://arxiv.org/ abs/1812.05784

24. Li, Y., Chen, Y., Qi, X., Li, Z., Sun, J., Jia, J.: Unifying voxel-based representation with transformer for 3d object detection (2022), https://arxiv.org/abs/2206. 00630

25. Li, Y., Ge, Z., Yu, G., Yang, J., Wang, Z., Shi, Y., Sun, J., Li, Z.: Bevdepth: Acquisition of reliable depth for multi-view 3d object detection (2022), https: //arxiv.org/abs/2206.10092

26. Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Yu, Q., Dai, J.: Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers (2022), https://arxiv.org/abs/2203.17270

27. Liu, Y., Wang, T., Zhang, X., Sun, J.: Petr: Position embedding transformation for multi-view 3d object detection (2022), https://arxiv.org/abs/2203.05625

28. Liu, Y., Shen, Y., Chen, R., Zhao, J., Tian, Y., Zhang, Y., Long, T., Yin, Z., Wang, Y., Qin, Z., et al.: Prm-as-a-judge 1.5: A toolkit for robot process assessment. arXiv preprint arXiv:2608.14284 (2026)

29. Liu, Z., Wu, Z., Tóth, R.: Smoke: Single-stage monocular 3d object detection via keypoint estimation (2020), https://arxiv.org/abs/2002.10111

30. Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., Han, S.: Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation (2024), https://arxiv.org/abs/2205.13542

31. Park, K., Kim, Y., Kim, D., Choi, J.W.: Resilient sensor fusion under adverse sensor failures via multi-modal expert fusion (2025), https://arxiv.org/abs/ 2503.19776

32. Pitropov, M., Garcia, D.E., Rebello, J., Smart, M., Wang, C., Czarnecki, K., Waslander, S.: Canadian adverse driving conditions dataset. The International Journal of Robotics Research 40(4-5), 681–690 (Dec 2020). https://doi.org/10. 1177/0278364920979368, http://dx.doi.org/10.1177/0278364920979368

33. Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation (2017), https://arxiv.org/abs/1612.00593

34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision (2021), https://arxiv.org/abs/ 2103.00020

35. Sadeghian, R., Hooshyaripour, N., Joslin, C., Lee, W.: Reliability-driven lidarcamera fusion for robust 3d object detection (2025), https://arxiv.org/abs/ 2502.01856

36. Shi, S., Guo, C., Jiang, L., Wang, Z., Shi, J., Wang, X., Li, H.: Pv-rcnn: Point-voxel feature set abstraction for 3d object detection (2021), https://arxiv.org/abs/ 1912.13192

37. Shi, S., Wang, X., Li, H.: Pointrcnn: 3d object proposal generation and detection from point cloud (2019), https://arxiv.org/abs/1812.04244

38. Sural, S., Sahu, N., Rajkumar, R.: Contextualfusion: Context-based multi-sensor fusion for 3d object detection in adverse operating conditions (2024), https:// arxiv.org/abs/2404.14780

39. Tian, Y., Ji, Y., Zheng, X., Qin, Z., Wang, Y., Zheng, X., Liu, Y., Bai, S., Li, Z., Wang, L., et al.: Spatial intelligence from a cognitive map perspective: A survey (2026)

40. Vora, S., Lang, A.H., Helou, B., Beijbom, O.: Pointpainting: Sequential fusion for 3d object detection (2020), https://arxiv.org/abs/1911.10150

41. Wang, C., Ma, C., Zhu, M., Yang, X.: Pointaugmenting: Cross-modal augmentation for 3d object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11794–11803 (June 2021)

42. Wang, H., Tang, H., Shi, S., Li, A., Li, Z., Schiele, B., Wang, L.: Unitr: A unified and eficient multi-modal transformer for bird’s-eye-view representation (2023), https://arxiv.org/abs/2308.07732

43. Wang, T., Zhu, X., Pang, J., Lin, D.: Fcos3d: Fully convolutional one-stage monocular 3d object detection (2021), https://arxiv.org/abs/2104.10956

44. Wang, Y., Ji, Y., Cao, M., Shen, Y., Xiao, R., Lyu, H., Xie, S., Liu, E., Tian, K., Long, T., et al.: Orca: The world is in your mind. arXiv preprint arXiv:2606.30534 (2026)

45. Wang, Y., Guizilini, V., Zhang, T., Wang, Y., Zhao, H., Solomon, J.: Detr3d: 3d object detection from multi-view images via 3d-to-2d queries (2021), https: //arxiv.org/abs/2110.06922

46. Xie, S., Kong, L., Zhang, W., Ren, J., Pan, L., Chen, K., Liu, Z.: Robobev: Towards robust bird’s eye view perception under corruptions (2023), https://arxiv.org/ abs/2304.06719

47. Xie, Y., Xu, C., Rakotosaona, M.J., Rim, P., Tombari, F., Keutzer, K., Tomizuka, M., Zhan, W.: Sparsefusion: Fusing multi-modal sparse representations for multisensor 3d object detection (2023), https://arxiv.org/abs/2304.14340

48. Yan, J., Liu, Y., Sun, J., Jia, F., Li, S., Wang, T., Zhang, X.: Cross modal transformer: Towards fast and robust 3d object detection (2023), https://arxiv.org/ abs/2301.01283

49. Yan, Y., Mao, Y., Li, B.: Second: Sparsely embedded convolutional detection. Sensors 18(10) (2018). https://doi.org/10.3390/s18103337, https://www.mdpi. com/1424-8220/18/10/3337

50. Yang, Z., Chen, J., Miao, Z., Li, W., Zhu, X., Zhang, L.: Deepinteraction: 3d object detection via modality interaction (2022), https://arxiv.org/abs/2208.11112

51. Yin, T., Zhou, X., Krähenbühl, P.: Center-based 3d object detection and tracking (2021), https://arxiv.org/abs/2006.11275

52. Yin, T., Zhou, X., Krähenbühl, P.: Multimodal virtual point 3d detection (2021), https://arxiv.org/abs/2111.06881

53. Zhou, Y., Tuzel, O.: Voxelnet: End-to-end learning for point cloud based 3d object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4490–4499 (2018)