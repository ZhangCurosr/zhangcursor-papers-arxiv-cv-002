# KSG-Net: Key-Sparse and Global-Context Learning for Maritime 3D Ship Detection

Zhouyuan Huai<sup>1</sup>, Meiqi Wan<sup>1</sup>, Yan Yang<sup>3,4</sup>, Minshi Chen<sup>1</sup>, Xin Yuan<sup>1,2,∗</sup> Wei Wang<sup>1,2</sup>, and Xiao Wang<sup>1,2</sup>

<sup>1</sup> School of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan 430065, China

<sup>2</sup> Hubei Province Key Laboratory of Intelligent Information Processing and Real-Time Industrial System, Wuhan University of Science and Technology, Wuhan 430065, China

3 State Key Laboratory of Robotics and Intelligent Systems, Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang 110016, China 4 China University of Chinese Academy of Sciences, Beijing 100049, China

Abstract. Accurate 3D ship detection in maritime environments is critical for autonomous navigation, yet remains challenging due to large-scale vessel variations, sparse point clouds of small vessels, and severe seaclutter interference. Existing methods, primarily based on 2D features or dense repre sentations, struggle to balance detection accuracy and computational eficiency, while sparse 3D detectors designed for road scenes generalize poorly to maritime scenarios. This paper focuses on two key challenges in maritime LiDAR perception: weak feature representation for small and sparse vessels, and insuficient global structural modeling for large vessels due to the limited receptive field of local sparse convolutions. To address these issues, we propose KSG-Net, a Key-Sparse and Global-Context learning network for maritime 3D ship detection. The core idea is to jointly enhance local discriminative features and global structural awareness within a unified fully sparse detection framework. Specifically, a Key Sparse Multi-scale Aggregation (KSMA) module is designed to enhance the representation of small and sparse vessels by selecting informative key voxels and aggregating cross-scale neighborhood features. Furthermore, a Global Context Aggregation (GCA) module is introduced to capture long-range geometric dependencies through scene-level context modeling with gated residual interactions, thereby improving the representation of large vessels. Extensive experiments on the Thames River vessel dataset and simulated datasets demonstrate that KSG-Net consistently outperforms existing methods in multi-scale vessel detection and exhibits strong robustness in complex maritime environments.

Keywords: 3D Ship Detection · Sparse 3D Detection · Key-Sparse Aggregation · Global Context Modeling

∗ Corresponding author: Xin Yuan (yuanxincherry@gmail.com)

## 1 Introduction

Accurate 3D ship detection in maritime environments is essential for autonomous navigation, collision avoidance, and intelligent decision-making. Compared with road scenes, maritime environments exhibit more complex sensing conditions, including large vessel-scale variations, irregular background structures, sparse point observations, and severe sea-clutter interference [10,21]. These characteristics make maritime 3D ship detection substantially diferent from conventional autonomous-driving detection tasks.

Existing maritime perception methods are still largely dominated by imagebased detection, dense 3D representations, or multi-sensor fusion pipelines [20,22]. Although these methods have achieved meaningful progress, they often sufer from quantization errors and computational redundancy during the densification process, limiting their applicability in large-scale maritime environments. Meanwhile, fully sparse 3D detectors developed for road scenes, such as VoxelNeXt [1], provide an eficient paradigm by operating directly on non-empty sparse voxels. However, directly transferring these frameworks to maritime scenarios is non-trivial, as maritime targets exhibit more severe scale variation and less regular point distributions.

![](images/55fc6f393193adcb8557ef59420e554ee423f7553fc8a4520f4eab64b0566c6b.jpg)

![](images/70d235e1f3be68a72c781a4b20bf175df06e656d218fc8158b3519f99d6444e0.jpg)  
(a) Weak Representation of Small Sparse Vessels

![](images/c98e0cd6390d4e0c2387ca690b1b78b3f7836135087583d17757c8875e3184e6.jpg)

![](images/0542410b44353c097c0c64b3a4ed21f92abf3164631fe98946f5e9dae616892d.jpg)  
(b) Limited Global Modeling ofLarge Vessels  
Fig. 1. Illustration of two key challenges in maritime 3D ship detection. (a) Weak Representation of Small Sparse Vessels. Small vessels usually produce sparse and discontinuous point responses, leading to weak and unstable local representations. (b) Limited Global Modeling of Large Vessels. Large vessels often span broad spatial regions, while local sparse convolutions have limited receptive fields and struggle to capture long-range structural dependencies.

As shown in Fig. 1, two key challenges arise in maritime 3D ship detection: (1) Weak representation of small and sparse vessels. Small vessels often produce only sparse and discontinuous point responses, providing limited geometric cues that are easily confused with background clutter. (2) Insuficient global structural modeling for large vessels. Large vessels usually span broad spatial regions, while local sparse convolutions have limited receptive fields and struggle to capture long-range structural dependencies. These scale-dependent challenges motivate the need for a unified sparse detection framework that can simultaneously enhance local discriminative features for small vessels and strengthen global structural modeling for large vessels.

To address the above challenges, we propose KSG-Net, a Key-Sparse and Global-Context Detection Network tailored for maritime 3D ship detection. The core idea is to jointly enhance local discriminative representations and global structural awareness within a unified sparse detection framework. To this end, we design two complementary modules: (1) a Key Sparse Multi-scale Aggregation (KSMA) module, which enhances the representation of small and sparse vessels by selecting informative key voxels and performing cross-scale neighborhood aggregation under highly sparse point distributions; and (2) a Global Context Aggregation (GCA) module, which captures long-range geometric dependencies by modeling scene-level global context and injecting it into sparse features through gated residual interactions, thereby improving the representation of large vessels and robustness in cluttered maritime environments. In summary, the main contributions of this paper are as follows:

• We propose KSG-Net, a maritime-oriented fully sparse 3D detection framework for vessel detection in LiDAR point clouds, addressing the overlooked challenges of weak small-vessel representation and insuficient large-vessel global modeling.

• We design the Key Sparse Multi-scale Aggregation (KSMA) module, which enhances the representation of small and sparse vessels through informative key-voxel selection and cross-scale neighborhood interaction, focusing computation on informative sparse locations.

• We introduce the Global Context Aggregation (GCA) module, which addresses the limited receptive field of local sparse convolutions by capturing long-range geometric structures for large vessels through scene-level global context modeling and gated residual injection.

• Extensive experiments on the Thames River vessel dataset and simulated datasets demonstrate that KSG-Net outperforms existing methods in multiscale vessel detection and exhibits strong robustness in complex maritime environments.

## 2 Related Work

## 2.1 LiDAR-Based Dense Detectors

Dense detectors convert irregular point clouds into structured representations, such as bird’s-eye view (BEV) feature maps, enabling the use of mature convolutional neural networks. Early works, including MV3D and PointPillars [8], establish the paradigm of projecting point clouds into dense representations for eficient processing, while CenterPoint [25] further improves localization accuracy via center-based object modeling. Recent dense LiDAR detectors have further improved representation capability through architectural refinement, temporal modeling, and multi-modal fusion. L4DR [5] proposes a LiDAR and 4D radar fusion framework that improves detection robustness under adverse weather conditions. BevNext [9] modernizes dense BEV-based detection pipelines by integrating CRF-based depth estimation and long-term temporal feature aggregation, achieving strong performance on the nuScenes benchmark. FSD-BEV [7] introduces foreground self-distillation and frame-level point cloud densification strategies, improving single-frame BEV detection performance.

Despite their efectiveness, dense detectors inevitably introduce quantization errors and computational redundancy during the densification process, which becomes more pronounced in large-scale maritime environments. Moreover, these methods often struggle to preserve fine-grained geometric structures of small and sparse vessels, limiting their applicability in maritime LiDAR scenarios.

## 2.2 LiDAR-Based Sparse Detectors

Sparse 3D detectors improve the eficiency of LiDAR-based 3D object detection by performing feature extraction and prediction only on non-empty voxels or foreground points, avoiding the redundancy introduced by dense feature construction. Representative methods such as SECOND [23], PV-RCNN [15], and PV-RCNN++ [16] have advanced this direction through sparse voxel encoding and point-voxel collaborative representation learning.

Recent studies have further developed fully sparse detection frameworks. FSD [3] systematically constructs a fully sparse 3D detector, VoxelNeXt [1] unifies proposal generation and detection heads in sparse voxel space, and FSD V2 [4] improves instance-level representation with virtual voxels. In parallel, long-range dependency modeling has also received increasing attention. HED-Net [27], SAFDNet [26], DSVT [19], and LION [12] enhance sparse representations through hierarchical interaction, adaptive feature difusion, sparse Transformer modeling, and state-space-style feature interaction, respectively.

Although sparse detectors have achieved remarkable progress in eficiency and long-range perception, most of them are designed for autonomous driving scenarios, where object layouts are relatively structured and backgrounds are more controlled. Maritime scenes present more irregular target distributions, stronger background interference, and larger scale variations. Small vessels often produce extremely sparse point responses, while large vessels span broad spatial regions. Therefore, existing sparse detectors remain limited in stable small-target representation and global structural modeling for large vessels. Motivated by these limitations, our work builds upon VoxelNeXt [1] and further enhances key sparse representation and global context modeling for maritime LiDAR 3D detection.

## 2.3 LiDAR-based Maritime Perception

Maritime perception has attracted increasing attention in recent years, and Li-DAR has shown strong potential for object detection, tracking, and environment understanding in maritime environments. Lin et al. [10] provided a systematic review of deep-learning-based maritime perception and highlighted the substantial diferences between maritime and road scenes in terms of target scale, background structure, and environmental disturbances. Xie et al. [21] further advanced LiDAR-based ship detection and tracking in busy maritime environments, while recent datasets have provided important support for maritime LiDAR perception research [14,28].

Despite these advances, many maritime perception methods still rely on image-based detection, multi-sensor fusion, dense 3D representations, or nonfully-sparse point-cloud detection paradigms [8,10,15]. Recent maritime 3D detection methods have begun to address vessel scale variation and complex marine conditions through hierarchical vertical-aware modeling, adaptive multi-scale design, and uncertainty-aware point cloud detection [20,22]. However, a unified fully sparse framework tailored to maritime LiDAR detection remains underexplored, especially for jointly handling stable representation of small sparse vessels and global structural modeling of large vessels. Diferent from existing maritime detectors, our method starts from a fully sparse framework and designs KSMA and GCA to enhance key sparse representation and global context modeling, respectively.

## 3 Methodology

In this section, we present the proposed KSG-Net. Fig. 2 illustrates the overall architecture of KSG-Net and its two core modules. As shown in Fig. 2(a), KSG-Net follows a fully sparse 3D detection paradigm, consisting of input encoding, sparse backbone feature extraction, KSMA, GCA, and the detection head. Fig. 2(b) and Fig. 2(c) further show the internal structures of the KSMA and GCA modules, respectively.

Specifically, the input maritime LiDAR point cloud is first encoded by the pillar encoder and processed by the sparse backbone to obtain a shared sparse BEV representation. Based on this representation, KSMA enhances local discriminative features for small sparse vessels through key-voxel identification and multi-scale neighborhood aggregation. GCA then supplements long-range structural information for large vessels through scene-level global context modeling and gated residual injection. The two modules are sequentially applied to the same shared sparse representation, improving multi-scale vessel detection while preserving the eficiency of fully sparse inference.

## 3.1 Key Sparse Multi-scale Aggregation

The objective of KSMA is to selectively enhance local discriminative features for small sparse vessels. As shown in Fig. 2(b), KSMA first identifies informative key voxels from the shared sparse representation, aggregates their multi-scale sparse neighborhoods, and writes the refined responses back through adaptive scale fusion and residual enhancement. This selecting-before-enhancing strategy focuses computation on informative sparse locations and improves local representation under sparse maritime observations.

![](images/bd2632c3628ecea7cecb9fd93f4d7f78d33a81d63b8cda757450c97cecc1a1c4.jpg)  
Fig. 2. Overview of KSG-Net. (a) The overall fully sparse detection pipeline for maritime LiDAR point clouds. (b) The Key Sparse Multi-scale Aggregation (KSMA) module, which enhances informative sparse locations through key-voxel identification and cross-scale neighborhood aggregation. (c) The Global Context Aggregation (GCA) module, which injects scene-level global context into the sparse representation to improve large-vessel structural modeling.

Key Sparse Voxel Identification Let the shared sparse BEV representation be denoted as $\mathcal { F } = \{ ( \mathbf { u } _ { i } , \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { M } .$ where $\mathbf { u } _ { i } \in \mathbb { Z } ^ { 2 }$ denotes the BEV coordinate of the i-th non-empty voxel and $\mathbf { x } _ { i } \in \mathbb { R } ^ { C }$ denotes the corresponding feature vector. To estimate the importance of each sparse location, we introduce a lightweight sparse scoring head on top of the shared representation and produce a saliency score for every voxel:

$$
\omega _ { i } = \sigma ( \varPsi _ { \mathrm { s c o r e } } ( \mathbf { x } _ { i } ) ) ,\tag{1}
$$

where $\varPsi _ { \mathrm { s c o r e } } ( \cdot )$ denotes the sparse scoring function and $\sigma ( \cdot )$ is the sigmoid activation.

Based on these scores, we retain the top-K voxels within each sample and treat them as the key-voxel set. Let $\mathcal { T } _ { b }$ denote the voxel index set of the b-th sample. The corresponding key-voxel set is defined as

$$
\mathcal { K } _ { b } = \mathrm { T o p K } \left( \left\{ \omega _ { i } \mid i \in \mathcal { I } _ { b } \right\} , K \right) , \qquad \mathcal { K } = \bigcup _ { b = 1 } ^ { B } \mathcal { K } _ { b } .\tag{2}
$$

This saliency-aware selection avoids uniformly enhancing all non-empty voxels and instead uses high-response locations as anchors for subsequent aggregation. It therefore concentrates computation on vessel-related sparse structures and reduces interference from background clutter.

Multi-scale Sparse Aggregation and Feature Refinement After key voxel selection, KSMA aggregates neighborhood context from multi-scale sparse BEV features. Let $\mathcal { F } ^ { ( l ) } = ( \mathbf { u } _ { i } ^ { ( l ) } , \mathbf { x } ^ { ( l ) } \ast j ) \ast j = 1 ^ { M _ { l } }$ denote the sparse feature set at scale l. For each key voxel $i \in \mathcal K$ , its coordinate is aligned to scale l by $\delta _ { l } ,$ followed by k-nearest-neighbor search:

$$
\widetilde { \mathbf { u } } _ { i } ^ { ( l ) } = \frac { \mathbf { u } _ { i } } { \delta _ { l } } , \qquad \mathcal { N } _ { i } ^ { ( l ) } = \mathrm { K N N } \left( \widetilde { \mathbf { u } } _ { i } ^ { ( l ) } , \{ \mathbf { u } _ { j } ^ { ( l ) } \} _ { j = 1 } ^ { M _ { l } } , k _ { l } \right) ,\tag{3}
$$

where $\delta _ { l }$ and $k _ { l }$ denote the scale alignment factor and neighborhood size at scale l, respectively.

For each scale, the neighboring features are first mean-aggregated and then projected through a scale-specific transformation function to obtain a scale-wise aggregated representation:

$$
\bar { \mathbf { z } } _ { i } ^ { ( l ) } = \frac { 1 } { | \mathcal { N } _ { i } ^ { ( l ) } | } \sum _ { j \in \mathcal { N } _ { i } ^ { ( l ) } } \mathbf { x } _ { j } ^ { ( l ) } , \qquad \mathbf { z } _ { i } ^ { ( l ) } = \varPhi _ { l } \left( \bar { \mathbf { z } } _ { i } ^ { ( l ) } \right) ,\tag{4}
$$

where $\varPhi _ { l } ( \cdot )$ denotes the projection function associated with scale $l .$

Since the contributions of diferent scales may vary for diferent key voxels, we further predict adaptive scale weights from the current key feature and fuse the scale-wise aggregated representations accordingly:

$$
\pmb { \alpha } _ { i } = \mathrm { s o f t m a x } ( \mathbf { W } _ { \alpha } \mathbf { x } _ { i } ) , \qquad \tilde { \mathbf { z } } _ { i } = \sum _ { l = 1 } ^ { L } \alpha _ { i } ^ { ( l ) } \mathbf { z } _ { i } ^ { ( l ) } ,\tag{5}
$$

where ${ \pmb { \alpha } } _ { i } = [ \alpha _ { i } ^ { ( 1 ) } , \dots , \alpha _ { i } ^ { ( L ) } ]$ denotes the normalized scale-weight vector.

Based on the fused multi-scale context, we concatenate the original key feature with the aggregated feature and transform them into an enhanced representation. Furthermore, since diferent key voxels may have diferent confidence levels, we use the saliency score as a gating factor to modulate the residual update magnitude, yielding the final KSMA output:

$$
\hat { \mathbf { x } } _ { i } = \phi _ { \mathrm { f u s e } } \left( \left[ \mathbf { x } _ { i } \left. \tilde { \mathbf { z } } _ { i } \right. \right) , \right.\tag{6}
$$

$$
\begin{array} { r } { \mathbf { x } _ { i } ^ { \mathrm { K S M A } } = \mathbf { x } _ { i } + \gamma \omega _ { i } \hat { \mathbf { x } } _ { i } , \qquad i \in K , } \end{array}\tag{7}
$$

where $\varPhi _ { \mathrm { f u s e } } ( \cdot )$ denotes the feature fusion mapping and $\gamma$ is a learnable residual scaling coeficient.

In this way, KSMA enhances only informative sparse locations with crossscale neighborhood context, improving small-vessel representations while preserving the eficiency of fully sparse inference.

## 3.2 Global Context Aggregation

Although KSMA significantly improves the local representation quality of key sparse locations, large vessels in maritime scenes usually exhibit broader spatial extent and more complex geometry. Relying solely on local sparse convolutions and local enhancement remains insuficient to capture their long-range structural dependencies. To address this issue, we further design GCA to inject scene-level global context into the sparse representation in a lightweight manner, thereby improving the holistic understanding of large-scale structures. As illustrated in Fig. 2(c), GCA mainly consists of scene-level global context modeling and context-gated residual enhancement.

Scene-level Global Context Modeling Let the sparse feature set after KSMA be denoted as $\mathcal { F } ^ { \mathrm { K S M A } } = \{ ( \mathbf { u } _ { i } , \mathbf { x } _ { i } ^ { \mathrm { K S M A } } ) \} _ { i = 1 } ^ { M }$ . For each sample, we first perform global pooling over all non-empty sparse locations to extract a scenelevel context descriptor, and then project it into the same representation space as the local sparse features:

$$
{ \bf g } _ { b } = \rho \left( \{ { \bf x } _ { i } ^ { \mathrm { K S M A } } \ | \ i \in \mathcal { T } _ { b } \} \right) ,\tag{8}
$$

$$
\bar { \mathbf { g } } _ { b } = \varPhi _ { \mathrm { c t x } } ( \mathbf { g } _ { b } ) ,\tag{9}
$$

where $\rho ( \cdot )$ denotes a global pooling operator and $\varPhi _ { \mathrm { c t x } } ( \cdot )$ is the context projection mapping.

This process compresses scene statistics originally distributed across diferent sparse local positions into a unified global descriptor, which serves as a scene-level reference for the subsequent enhancement of large-scale structures. Unlike global self-attention that explicitly models pairwise interactions among all voxels, this pooling-based context modeling is considerably lighter and better suited to the eficiency requirements of fully sparse detection.

Context-Gated Residual Enhancement After obtaining the scene-level context, we broadcast it back to each non-empty voxel in the current sample and use a gating mechanism to adaptively regulate the context injection strength. For the i-th voxel in sample b, the gate vector is jointly predicted from the local feature and the global context:

$$
\mathbf { m } _ { i } = \sigma \left( \phi _ { \mathrm { g a t e } } \left( \left[ \mathbf { x } _ { i } ^ { \mathrm { K S M A } } \left. \bar { \mathbf { g } } _ { b } \right. \right) \right) , \right.\tag{10}
$$

where $\varPhi _ { \mathrm { g a t e } } ( \cdot )$ denotes the gating function.

The global context is then injected into the local feature in a channel-wise gated manner, and its overall influence is controlled by a learnable residual coeficient, yielding the final GCA-enhanced representation:

$$
\mathbf { x } _ { i } ^ { \mathrm { G C A } } = \mathbf { x } _ { i } ^ { \mathrm { K S M A } } + \lambda \left( \mathbf { m } _ { i } \odot \bar { \mathbf { g } } _ { b } \right) ,\tag{11}
$$

where $\odot$ denotes element-wise multiplication and $\lambda$ is a learnable residual scaling factor.

This gated residual injection allows the model to adaptively determine how much global information should be injected according to the local feature content at each sparse position. For large vessels in maritime scenes, this design efectively alleviates the fragmented representation caused by limited local receptive fields and enhances the modeling of long-range geometric dependencies.

After KSMA and GCA enhancement, the refined sparse features are fed into the detection head for final prediction. Let the prediction set be denoted as Y, which includes the classification heatmap, center ofsets, vertical ofset, box dimensions, orientation, and an IoU-related branch. During training, we follow the objective design of the fully sparse baseline detector, and the overall loss is formulated as

$$
\mathcal { V } = \left\{ \mathbf { H } , \varDelta \mathbf { c } , \varDelta z , \mathbf { d } , \mathbf { r } , q \right\} ,\tag{12}
$$

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { h m } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { i o u } } \mathcal { L } _ { \mathrm { i o u } } + \lambda _ { \mathrm { i o u } _ { \mathrm { - } } \mathrm { r e g } } \mathcal { L } _ { \mathrm { i o u } _ { \mathrm { - } } \mathrm { r e g } } ,\tag{13}
$$

where H denotes the sparse classification heatmap, ∆c denotes the 2D center ofset on the BEV plane, ∆z denotes the vertical ofset, d and r denote box dimensions and orientation, and q denotes the IoU prediction. With this formulation, KSG-Net can be trained end-to-end within a unified fully sparse detection framework.

## 4 Experiments

## 4.1 Experimental Setup

Ship LiDAR Dataset: We use the maritime LiDAR point cloud dataset publicly released by Zhang et al. [28]. The dataset was collected using a multibeam photon-counting LiDAR sensor and contains real-world port scenes from Wuhan, Shanghai, and Qingdao. The annotated objects are categorized into four vessel classes: Cargo ship, Tour boat, Engineering ship, and Speedboat. The dataset is organized in KITTI format and consists of 10,571 training frames, 1,200 validation frames, and 1,500 testing frames.

Thames River Real Dataset: We use the first real-world Thames River LiDAR dataset released by Xie et al. [21]. The dataset was collected in the London docklands and busy Thames waterways, covering diverse vessel types and scales, including monohulls, catamarans, and sailboats with lengths ranging from approximately 5 m to 40 m. It contains 9,200 frames in total, with 7,820 frames for training and 1,380 frames for testing, corresponding to an 85%/15% split.

Thames River Simulated Dataset: This dataset consists of synthetic data generated using the Robot Operating System (ROS), simulating port and fairway environments with maritime targets such as ships and buoys. The raw simulated data were further refined by Xie et al. [21] to correct annotation errors and improve data quality. The version used in this study contains 11,360 frames, covering diverse scenarios with isolated vessel point clouds as well as complex terrain backgrounds.

Experimental Details We evaluate all methods using KITTI-style 3D Average Precision (AP) and mean Average Precision (mAP) under the $R _ { 4 0 }$ protocol. A detection is considered correct when the predicted class matches the ground truth and the 3D IoU between the predicted and ground-truth boxes is greater than 0.5. We therefore report 3D AP and 3D mAP at an IoU threshold of 0.5 for all datasets. All experiments are implemented with OpenPCDet and PyTorch on a workstation equipped with an Intel Core i9-14900KF CPU and a single NVIDIA GeForce RTX 4060 GPU. The compared detectors were originally developed for terrestrial autonomous-driving scenes. To ensure a fair comparison, we retrain all methods on the same maritime data splits with the same point-cloud range, input preprocessing, and evaluation protocol. For each baseline, we follow its oficial training configuration as closely as possible, including the optimizer and learning-rate schedule, and adapt only the class definitions and detection range to the maritime setting. KSG-Net is trained with the Adam optimizer and a OneCycle learning-rate scheduler, using a peak learning rate of $1 \times 1 0 ^ { - 3 }$ , a weight decay of 0.01, a batch size of 4, and 80 epochs.

Table 1. Quantitative comparison on the Ship LiDAR Dataset. AP@0.5 (%) is reported for representative categories, where Cargo, Tour, Engineer, and Speed denote cargo ships, tour boats, engineering ships, and speedboats, respectively. FLOPs are efective FLOPs averaged over validation samples. FPS is measured with a batch size of 1. GBlobs is evaluated as a local geometric encoder on VoxelNeXt. The best AP results are highlighted in bold. “–” denotes a result that is not yet available.
<table><tr><td rowspan="3">Method 一</td><td rowspan="3"></td><td colspan="4">AP@0.5 (%)</td><td rowspan="3">FLOPs (G)</td><td rowspan="3">FPS (Hz)</td><td rowspan="3">mAP (%)</td></tr><tr><td>Cargo</td><td>Tour</td><td>Engineer</td><td>Speed |</td></tr><tr><td rowspan="3">Point</td><td rowspan="3">PointRCNN (CVPR 2019) [17] Part-A2 (CVPR 2020) [18]</td><td>64.71</td><td>76.16</td><td>78.76</td><td>30.00</td><td>127.41</td><td>10.37</td><td>62.40</td></tr><tr><td>63.89</td><td>78.33</td><td>71.74</td><td>33.78</td><td>508.35</td><td>13.19</td><td>61.94</td></tr><tr><td>PV-RCNN (CVPR 2020) [15] 65.81</td><td>79.30</td><td>74.21</td><td>31.02</td><td>89.02</td><td>8.89</td><td>63.09</td></tr><tr><td rowspan="6">Voxel</td><td>PointPillars (CVPR 2019) [8]</td><td>57.08</td><td>76.50</td><td>85.98</td><td>42.86</td><td>63.44</td><td>62.15</td><td>65.59</td></tr><tr><td>VoxelR-CNN (AAAI 2021) [2]</td><td>60.23</td><td>77.12</td><td>78.22</td><td>41.70</td><td>48.53</td><td>25.25</td><td>64.31</td></tr><tr><td>VoxelNeXt (CVPR 2023) [1]</td><td>67.65</td><td>87.81</td><td>82.49</td><td>60.07</td><td>32.03</td><td>23.55</td><td>74.50</td></tr><tr><td>DSVT (CVPR 2023) [19]</td><td>71.45</td><td>87.30</td><td>82.10</td><td>68.90</td><td>339.85</td><td>14.97</td><td>77.44</td></tr><tr><td>FSHNet (CVPR 2025) [11]</td><td>67.98</td><td>80.20</td><td>84.32</td><td>70.10</td><td>54.14</td><td>12.37</td><td>75.65</td></tr><tr><td>GBlobs (CVPR 2025) [13]</td><td>65.19</td><td>78.22</td><td>87.12</td><td>63.21</td><td>32.36</td><td>22.72</td><td>74.44</td></tr><tr><td rowspan="6">BEV</td><td>CenterPoint (CVPR 2021) [25]</td><td>72.16</td><td>88.52</td><td>79.60</td><td>70.22</td><td>33.23</td><td>56.65</td><td>77.63</td></tr><tr><td>HEDNet (NeurIPS 2023) [27]</td><td>66.31</td><td>79.22</td><td>85.48</td><td>68.12</td><td>106.21</td><td>17.28</td><td>74.78</td></tr><tr><td>SAFDNet (CVPR 2024) [26]</td><td>72.31</td><td>86.98</td><td>81.24</td><td>71.75</td><td>60.83</td><td>20.25</td><td>78.07</td></tr><tr><td>Fade3D (TITS 2025) [24]</td><td>72.90</td><td>89.50</td><td>84.15</td><td>70.45</td><td>78.31</td><td>51.53</td><td>79.25</td></tr><tr><td>AutoReg3D (CVPR 2026) [6]</td><td>53.12</td><td>62.94</td><td>58.39</td><td>14.22</td><td>110.3</td><td>5.29</td><td>47.18</td></tr><tr><td>KSG-Net (Ours)</td><td>72.69</td><td>95.15</td><td>91.97</td><td>82.79</td><td>31.42</td><td>33.11</td><td>85.65</td></tr></table>

## 4.2 Comparison on Ship LiDAR Dataset

The quantitative comparison results on the Ship LiDAR Dataset are summarized in Table 1. KSG-Net achieves the highest mAP@0.5 of 85.65%, showing a clear margin over previous methods across point-, voxel-, and BEV-based representations. It surpasses the strongest BEV baseline Fade3D [24] by 6.40%, SAFDNet [26] by 7.58%, and the strongest voxel baseline DSVT [19] by 8.21%. GBlobs [13], instantiated as a local geometric encoder on VoxelNeXt, reaches

Table 2. Experimental results on the Thames River Real and Thames River Simulated datasets. AP@0.5 (%) is reported. Venues are omitted here for compactness; they are given in Table 1, except LION (NeurIPS 2024). F denotes efective FLOPs (G) averaged over validation samples; FPS is measured with batch size 1. Best AP results are highlighted in bold. “–” indicates that the category is unavailable.
<table><tr><td rowspan="2">Method</td><td colspan="6">Thames River Real</td><td colspan="6">Thames River Simulated</td></tr><tr><td>Small</td><td>Med.</td><td>Large</td><td>mAP (%)</td><td>F (G)</td><td>FPS (Hz)</td><td>Small</td><td>Med. Large</td><td>Buoy</td><td>mAP</td><td>(%) F (G)</td><td>FPS (Hz)</td></tr><tr><td>PointPillars [8]</td><td>47.32</td><td>51.47</td><td>46.60</td><td>48.46</td><td>27.12</td><td>56.32</td><td>62.00 69.00</td><td>77.80</td><td>51.80</td><td>65.20</td><td>41.22</td><td>69.93</td></tr><tr><td>PV-RCNN [15]</td><td>38.10</td><td>82.70</td><td>82.40</td><td>67.74</td><td>510.09</td><td>7.18 76.70</td><td>78.10</td><td>78.00</td><td>75.10</td><td>76.90</td><td>662.11</td><td>12.12</td></tr><tr><td>Part-A2 [18]</td><td>44.65</td><td>86.50</td><td>77.30</td><td>69.48</td><td>422.31</td><td>13.14</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Voxel R-CNN [2]</td><td>40.14</td><td>84.63</td><td>81.62</td><td>68.80</td><td>12.17</td><td>24.87</td><td>75.70 74.50</td><td>77.10</td><td>72.10</td><td>74.85</td><td>8.52</td><td>35.54</td></tr><tr><td>VoxelNeXt [1]</td><td>54.35</td><td>86.96</td><td>73.29</td><td>71.20</td><td>32.01</td><td>23.54</td><td>87.16 82.56</td><td>86.50</td><td>63.16</td><td>79.84</td><td>22.40</td><td>22.09</td></tr><tr><td>DSVT [19]</td><td>74.12</td><td>91.43</td><td>79.52</td><td>81.69</td><td>337.22</td><td>21.85</td><td>85.50 76.10</td><td>78.70</td><td>77.60</td><td>79.47</td><td>374.12</td><td>14.28</td></tr><tr><td>LION [12]</td><td></td><td></td><td></td><td></td><td></td><td>90.60</td><td>75.4</td><td>78.30</td><td>85.1</td><td>82.4</td><td>1087.67</td><td>28.12</td></tr><tr><td>FSHNet [11]</td><td>42.21</td><td>75.94</td><td>69.93</td><td>62.69</td><td>46.30</td><td>16.27</td><td>81.43 75.46</td><td>79.38</td><td>68.51</td><td>76.19</td><td>32.41</td><td>15.37</td></tr><tr><td>Fade3D [24]</td><td>51.05</td><td>85.20</td><td>81.75</td><td>72.67</td><td>41.22</td><td>51.53</td><td>84.92 83.14</td><td>88.20</td><td>64.10</td><td>80.09</td><td>44.91</td><td>52.31</td></tr><tr><td>GBlobs [13]</td><td>67.09 2.50</td><td>86.04 71.84</td><td>96.83</td><td>83.32</td><td>31.01</td><td>22.74</td><td>91.14 83.12</td><td>86.55</td><td>71.08 36.50</td><td>82.98</td><td>22.40</td><td>21.34</td></tr><tr><td>AutoReg3D [6]</td><td></td><td></td><td>86.11</td><td>53.48</td><td>108.21</td><td>5.24</td><td>14.83 33.79</td><td>53.04</td><td></td><td>34.54</td><td>120.31</td><td>1.88</td></tr><tr><td>KSG-Net (Ours)</td><td>80.53</td><td>94.98</td><td>88.14</td><td>87.89</td><td>35.10</td><td>20.09</td><td>90.72 82.63</td><td>89.31</td><td>75.11</td><td>84.44</td><td>24.56</td><td>19.85</td></tr></table>

74.44% mAP, close to VoxelNeXt (74.50%) and well below KSG-Net. AutoReg3D [6] obtains 47.18% mAP and is the weakest entry, with a particularly low Speedboat AP of 14.22%. In terms of eficiency, KSG-Net uses 31.42 G FLOPs and runs at 33.11 FPS, remaining comparable to VoxelNeXt and GBlobs while being substantially cheaper than DSVT and AutoReg3D.

A detailed category-wise analysis further supports this conclusion. Fade3D [24] leads by only 0.21 % on the large and geometrically regular Cargo category (72.90% vs. ours 72.69%), whereas KSG-Net achieves the best AP on the more heterogeneous Tour boat (95.15%), Engineering ship (91.97%), and Speedboat (82.79%) categories. The gain on Speedboat is 11.04 % over SAFDNet [26] (71.75%). Speedboats typically produce the sparsest point responses and the largest shape variation; this improvement is therefore consistent with the design of KSMA for strengthening weak local responses, complemented by GCA for stabilizing structural context.

## 4.3 Comparison on Thames River Real and Simulated Datasets

As shown in Table 2, KSG-Net achieves the best overall mAP on both the Thames River Real and Thames River Simulated datasets. On the Thames River Real Dataset, KSG-Net obtains an mAP of 87.89%, outperforming GBlobs [13] by 4.57 % and DSVT [19] by 6.20 %. On the Thames River Simulated Dataset, KSG-Net achieves an mAP of 84.44%, exceeding GBlobs (82.98%), LION [12] (82.4%), and Fade3D [24] (80.09%). AutoReg3D [6] remains clearly behind on both splits. These results indicate that key-sparse aggregation and global context injection generalize across real and simulated maritime scenes. As shown in Fig. 3, KSG-Net produces focused responses around sparse small-vessel regions and coherent activations along extended hulls. This qualitative pattern is consistent with Table 2, where KSG-Net attains the highest Small and Medium AP on the Thames River Real Dataset together with the best overall mAP. A closer category-wise comparison further highlights the efectiveness of KSG-Net for multi-scale vessel detection. On the Thames River Real Dataset, KSG-Net attains the highest Small and Medium AP (80.53% and 94.98%) and the highest overall mAP (87.89%). Relative to DSVT, Small-vessel AP increases by 6.41 %. GBlobs records a higher Large-vessel AP (96.83% vs. 88.14%), while KSG-Net remains stronger in the sparse small-vessel regime and in overall mAP, which matches the role of KSMA.

![](images/3d42eaa08592912de9fd435f590d861b4e25a9c97a449a8e5552b6c49bd80f36.jpg)  
Fig. 3. Feature visualization on the Thames River Real. The heatmap shows the spatial activation patterns of KSG-Net in real maritime scenes. Stronger responses concentrate on vessel-related sparse regions and remain coherent along extended hulls, indicating that the full model preserves both local sparse cues and scene-level vessel structure.

As shown in Fig. 4, the qualitative results on the Thames River Real Dataset further demonstrate the advantage of KSG-Net in complex maritime scenes. Existing methods tend to sufer from missed detections or inaccurate localization, especially under sparse point responses or partial observations. In contrast, KSG-Net produces more complete and stable 3D bounding boxes, particularly for small and medium vessels that are easily missed. This visual comparison is consistent with the Real Small and Medium AP and the overall mAP in Table 2.

On the Thames River Simulated Dataset, KSG-Net maintains strong and balanced performance across vessel scales, achieving the best Large-vessel AP (89.31%) and the best overall mAP (84.44%). GBlobs obtains a slightly higher Small-vessel AP (91.14% vs. 90.72%). Although LION obtains a higher AP on the Buoy category, KSG-Net shows stronger overall detection capability on the vessel-related categories. This suggests that the proposed key-sparse aggregation and global context injection are particularly beneficial for maritime vessel detection, where targets often exhibit large-scale variations and irregular point distributions.

## 4.4 Ablation Study

To evaluate the individual contributions and complementarity of KSMA and GCA, we conduct ablation studies on both Thames River datasets (Table 3). Here, Baseline denotes our VoxelNeXt2D implementation with the IoU branch retained and both KSMA and GCA disabled; it is therefore diferent from the original VoxelNeXt configuration reported in Table 2. The full framework improves mAP from 78.74% to 87.89% (+9.15%) on Thames Real and from 77.13% to 84.44% (+7.31%) on Thames Sim.

![](images/2a3fcd9cc5be831bcd1a4f307447fb62e880ddf8192943a162551efb6b61546f.jpg)  
Fig. 4. Qualitative comparison on Thames River Real. Blue boxes denote predicted 3D bounding boxes. The dark-blue, red, and green ellipses indicate missed detections, error detections, and correct detections, respectively. Compared with existing methods, KSG-Net produces more accurate and stable predictions for vessels with diferent scales and reduces missed or erroneous detections in complex real maritime scenes.

Table 3. Ablation results on Thames Real and Thames Simulated datasets. The best results in each column are highlighted in bold.
<table><tr><td rowspan="3">Method</td><td rowspan="3">1</td><td colspan="4">Thames Real Dataset</td><td colspan="5">Thames Simulated Dataset</td></tr><tr><td colspan="2">AP@0.5 (%)</td><td></td><td>mAP|</td><td colspan="4">AP@0.5 (%)</td><td>mAP</td></tr><tr><td>Small</td><td>Med.</td><td>Large</td><td>(%)</td><td>Small</td><td>Med.</td><td>Large</td><td>Buoy</td><td>(%)</td></tr><tr><td>Baseline</td><td></td><td>57.86</td><td>83.98</td><td>94.39</td><td>78.74</td><td>86.94</td><td>81.39</td><td>87.18</td><td>53.03</td><td>77.13</td></tr><tr><td>Baseline + KSMA</td><td></td><td>75.23</td><td>91.31</td><td>84.62</td><td>83.72</td><td>87.62</td><td>82.07</td><td>88.32</td><td>69.35</td><td>81.84</td></tr><tr><td>Baseline + GCA</td><td></td><td>65.32</td><td>83.26</td><td>94.18</td><td>80.92</td><td>86.33</td><td>82.73</td><td>87.42</td><td>61.48</td><td>79.49</td></tr><tr><td>Ours</td><td>80.53</td><td></td><td>94.98</td><td>88.14</td><td>87.89</td><td>90.72</td><td>82.63</td><td>89.31</td><td>75.11</td><td>84.44</td></tr></table>

A closer look reveals complementary efects across categories. On Thames Real, KSMA improves Small-vessel AP by 17.37%; on Thames Sim, it improves Buoy AP by 16.32%. GCA alone raises overall mAP by 2.18% and 2.36% on the Real and Simulated datasets. Combining both modules yields the best mAP on both splits. On Real Large vessels, the baseline remains highest (94.39%), whereas the full model trades part of this category for substantial Small/Medium and mAP gains. This indicates that KSMA and GCA work best as a joint multiscale design: key-sparse enhancement recovers weak targets, while global context improves scene-level detection, with the largest benefit appearing in overall mAP.

![](images/aa03e691800987467310fc24ed8c4c20a17fe324cdd04cda650c34aba13e7a93.jpg)

![](images/5591f27d904f7f59c866affcfcf1b2d2f5508bc747e71eab3f6ddd1b7b08e8ac.jpg)

![](images/d8ca83cd5e532bb2da046aeee89f91a3e075f2213e3699b613cad8911eb3cae5.jpg)

![](images/72bc68e014855a7df0c0e7d1096bda821c3feb2b6e6ebea982f012bc8f255e8d.jpg)  
(a) Voxel size (m)

![](images/14a2cdf3c4ac5fee3d4f461cdc822f36a12199c85f55851affeef710c2c08d52.jpg)  
(b) Top-K ratio

![](images/8cc49bca3ba3c8aa225e63cf7e4c6dbef1b53aa06eb09f1aa6422be8dcdcc1c9.jpg)  
(c) KNN neighbor number  
Fig. 5. Hyperparameter analysis of the proposed method on the Thames River Real and Thames River Simulated datasets. We investigate three key hyperparameters: (a) voxel size, (b) Top-K ratio for key-voxel selection, and (c) KNN neighbor number. The best performance on both datasets is achieved with a voxel size of [0.2, 0.2, 0.4], a Top-K ratio of 15%, and a KNN neighbor number of 8.

## 4.5 Hyperparameter Analysis

To further analyze the influence of key hyperparameters in the proposed method, we conduct hyperparameter experiments on the Thames River Real and Thames River Simulated datasets. As shown in Fig. 5, we investigate three important hyperparameters, including the voxel size, the Top-K ratio for key-voxel selection, and the KNN neighbor number. Overall, these hyperparameters show consistent trends on both datasets, indicating that the proposed framework has relatively stable parameter sensitivity.

Voxel size. Fig. 5(a) shows the efect of voxel size. When the voxel size increases from [0.1, 0.1, 0.2] to [0.2, 0.2, 0.4], the performance improves noticeably on both datasets, while a further increase to [0.4, 0.4, 0.8] leads to performance degradation. This indicates that overly fine voxels may produce fragmented sparse representations, whereas overly coarse voxels tend to lose local geometric details. Therefore, a moderate voxel size provides a better balance between structural completeness and local detail preservation.

Top-K ratio. Fig. 5(b) presents the influence of the Top-K ratio. As the ratio increases from 5% to 15%, the detection performance consistently improves, while a slight decrease is observed at 20%. This suggests that selecting too few key voxels may miss informative sparse responses, whereas selecting too many may introduce redundant or noisy locations and weaken the selectivity of KSMA. KNN neighbor number. Fig. 5(c) further analyzes the efect of the KNN neighbor number. The best performance on both datasets is achieved when the neighbor number is set to 8. A smaller number of neighbors limits local context modeling, while an excessively large number may introduce irrelevant neighborhood information and reduce the precision of feature aggregation

## 5 Conclusion

In this paper, we explore a practical and challenging problem of maritime 3D ship detection, which requires accurately detecting vessels of varying scales under sparse point observations and severe background interference. To tackle the above issue, we propose KSG-Net, a Key-Sparse and Global-Context learning network tailored for maritime 3D ship detection. The core idea is to jointly enhance local discriminative representations for small vessels and global structural awareness for large vessels within a unified fully sparse detection framework. To this end, two complementary modules are designed: a Key Sparse Multiscale Aggregation (KSMA) module to enhance the representation of small and sparse vessels through informative key-voxel selection and cross-scale neighborhood aggregation; and a Global Context Aggregation (GCA) module to capture long-range geometric dependencies for large vessels through scene-level global context modeling with gated residual interactions. Extensive experiments on the Thames River vessel dataset and simulated datasets demonstrate that KSG-Net consistently outperforms existing methods in multi-scale vessel detection and exhibits strong robustness in complex maritime environments, validating the efectiveness of the proposed KSMA and GCA modules.

## References

1. Chen, Y., Liu, J., Zhang, X., Qi, X., Jia, J.: Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. In: CVPR. pp. 21674–21683 (2023)

2. Deng, J., Shi, S., Li, P., Zhou, W., Zhang, Y., Li, H.: Voxel r-cnn: Towards high performance voxel-based 3d object detection. In: AAAI. pp. 1201–1209 (2021)

3. Fan, L., Wang, F., Wang, N., Zhang, Z.X.: Fully sparse 3d object detection. In: NeurIPS. pp. 351–363 (2022)

4. Fan, L., Wang, F., Wang, N., Zhang, Z.: Fsd v2: Improving fully sparse 3d object detection with virtual voxels. IEEE TPAMI 47(2), 1279–1292 (2024)

5. Huang, X., Xu, Z., Wu, H., Wang, J., Xia, Q., Xia, Y., Li, J., Gao, K., Wen, C., Wang, C.: L4dr: Lidar-4dradar fusion for weather-robust 3d object detection. In: AAAI. pp. 3806–3814 (2025)

6. Huang, Z., Yoo, J., Jeon, S., Liu, Z., Campbell, M., Weinberger, K.Q., Hariharan, B., Chao, W.L., Luo, K.Z.: On the feasibility and opportunity of autoregressive 3d object detection. arXiv preprint arXiv:2603.07985 (2026)

7. Jiang, Z., Zhang, J., Zhang, Y., Liu, Q., Hu, Z., Wang, B., Wang, Y.: Fsd-bev: Foreground self-distillation for multi-view 3d object detection. In: ECCV. pp. 110– 126 (2024)

8. Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: Pointpillars: Fast encoders for object detection from point clouds. In: CVPR. pp. 12697–12705 (2019)

9. Li, Z., Lan, S., Alvarez, J.M., Wu, Z.: Bevnext: Reviving dense bev frameworks for 3d object detection. In: CVPR. pp. 20113–20123 (2024)

10. Lin, J., Diekmann, P., Framing, C.E., Zweigel, R., Abel, D.: Maritime environment perception based on deep learning. IEEE TITS 23(9), 15487–15497 (2022)

11. Liu, S., Cui, M., Li, B., Liang, Q., Hong, T., Huang, K., Shan, Y.: Fshnet: Fully sparse hybrid network for 3d object detection. In: CVPR. pp. 8900–8909 (2025)

12. Liu, Z., Hou, J., Wang, X., Ye, X., Wang, J., Zhao, H., Bai, X.: Lion: Linear group rnn for 3d object detection in point clouds. In: NeurIPS. pp. 13601–13626 (2024)

13. Mali´c, D., Fruhwirth-Reisinger, C., Schulter, S., Possegger, H.: Gblobs: Explicit local structure via gaussian blobs for improved cross-domain lidar-based 3d object detection. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 27357–27367. IEEE (2025)

14. Martelli, M., Faggioni, N., Ponzini, F.: Arnold annotated repository of navigational obstacles from lidar data. Autonomous Transportation Research 2(1), 45–60 (2026)

15. Shi, S., Guo, C., Jiang, L., Wang, Z., Shi, J., Wang, X., Li, H.: Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In: CVPR. pp. 10529–10538 (2020)

16. Shi, S., Jiang, L., Deng, J., Wang, Z., Guo, C., Shi, J., Wang, X., Li, H.: Pvrcnn++: Point-voxel feature set abstraction with local vector representation for 3d object detection. IJCV 131(2), 531–551 (2023)

17. Shi, S., Wang, X., Li, H.: Pointrcnn: 3d object proposal generation and detection from point cloud. In: CVPR. pp. 770–779 (2019)

18. Shi, S., Wang, Z., Wang, X., Li, H.: Part-aˆ 2 net: 3d part-aware and aggregation neural network for object detection from point cloud. arXiv preprint arXiv:1907.03670 (2019)

19. Wang, H., Shi, C., Shi, S., Lei, M., Wang, S., He, D., Schiele, B., Wang, L.: Dsvt: Dynamic sparse voxel transformer with rotated sets. In: CVPR. pp. 13520–13529 (2023)

20. Wang, Y., Wu, H., Wang, S., Kong, Y., Liu, Y., Luo, Z., Liu, C.: Hierarchical vertical-aware and adaptive multi-scale network for three-dimensional object detection in maritime environments. EAAI 162, 112626 (2025)

21. Xie, Y., Nanlal, C., Liu, Y.: Reliable lidar-based ship detection and tracking for autonomous surface vehicles in busy maritime environments. Ocean Engineering 312, 119288 (2024)

22. Xie, Y., Wu, P., Englot, B., Nanlal, C., Liu, Y.: Uncertainty-aware maritime point cloud detector (u-mpcd) for autonomous surface vehicles. IEEE Journal of Oceanic Engineering 51(1), 220–242 (2026)

23. Yan, Y., Mao, Y., Li, B.: Second: Sparsely embedded convolutional detection. Sensors 18(10), 3337 (2018)

24. Ye, W., Xia, Q., Wu, H., Dong, Z., Zhong, R., Wang, C., Wen, C.: Fade3d: Fast and deployable 3d object detection for autonomous driving. IEEE TITS 26(9), 12934–12946 (2025)

25. Yin, T., Zhou, X., Krahenbuhl, P.: Center-based 3d object detection and tracking. In: CVPR. pp. 11784–11793 (2021)

26. Zhang, G., Chen, J., Gao, G., Li, J., Liu, S., Hu, X.: Safdnet: A simple and efective network for fully sparse 3d object detection. In: CVPR. pp. 14477–14486 (2024)

27. Zhang, G., Junnan, C., Gao, G., Li, J., Hu, X.: Hednet: A hierarchical encoderdecoder network for 3d object detection in point clouds. In: NeurIPS. pp. 53076– 53089 (2023)

28. Zhang, Q., Wang, L., Meng, H., Zhang, W., Huang, G.: A lidar point clouds dataset of ships in a maritime environment. IEEE/CAA Journal of Automatica Sinica 11(7), 1681–1694 (2024)