# RSFusionDet: Underwater RGB-Sonar Multimodal Object Detection

Zhuoyan Liu<sup>1\*</sup> Yihan Wang<sup>1\*</sup> Bo Wang<sup>1†</sup> Bing Wang<sup>2</sup> Ye Li<sup>1</sup>

<sup>1</sup>National Key Laboratory of Autonomous Marine Vehicle Technology, Harbin Engineering University <sup>2</sup>Department of Aeronautical and Aviation Engineering, Hong Kong Polytechnic University

{liuzhuoyan,2020010127,wb}@hrbeu.edu.cn bingwang@polyu.edu.hk liye@hrbeu.edu.cn

## Abstract

Underwater unimodal object detection faces many challenges in sensor imaging, such as optical images limited by underwater noise and visible distance, and sonar images limited by less object structural information. While, optical images have rich object structural information, and sonar images are less affected by underwater noise and have a longer visible distance. Optical (RGB modality) and sonar (Sonar modality) images have complementary information underwater. In this paper, we create an RGB-Sonar multimodal object detection dataset, RGB-Sonar Fusion (RS-Fusion) and propose evaluation metricsfor the benchmark. And we propose the RGB-Sonar Fusion Detector (RSFusionDet) with a new RGB-Sonar multimodal object detection result expressionfor RGB-Sonar multimodal object detection. We analyze the features of RGB and Sonar modal information, and design a Cross-Attention Fusion (CA-Fusion) module to fuse RGB-Sonar spatial misalignment features and Object Matching Head (OMHead) with Loss (OMLoss) to match identical objects in RGB-Sonar modalities. Our RSFusionDet achieves 76.4/48.6 AP (RGB/Sonar) for object detection and 83.4 F1-Score<sub>match</sub> for object matching, on RSFusion, which outperforms other object detection models. Compared with the DINO baseline, our method improves by 0.7/1.4 AP (RGB/Sonar) while simultaneously providing reliable cross-modal object matching. The code and datasets are publicly available at https: //github.com/LEFTeyex/RSFusionDet.

## 1. Introduction

Object detection has been developed rapidly in the era of deep learning and is essential for many real-world applications [30, 31, 40, 42], e.g. underwater object detection. Underwater unimodal object detection relies primarily on visual sensors such as optical cameras and sonars (the forward-looking 2D multibeam sonar investigated in this paper). Objects in optical images have rich feature information, including color, texture, shape, edge, etc. However, in underwater environments, the absorption and scattering of light by the water medium significantly reduces the visible distance of optical cameras, typically 10-30 meters. In deep or turbid underwater environments, the visible distance may be reduced to 5-10 meters or less. At the same time, optical images also introduce problems such as color cast [25] and blur that severely limit the performance and perceived distance of the detector. Objects in sonar images have poor feature information, such as only shape and edge, making it difficult for detectors to identify the object category accurately. However, sonar sensors have a longer visible distance underwater, up to 40-120 meters or more, and are less affected by underwater environmental disturbances. Underwater unimodal object detection using RGB or Sonar modal information alone suffers from various degrees of limitations in perceived distance or detection performance. Optical (RGB modality) and sonar (Sonar modality) images complement each other in terms of feature information and perceived distance, so we investigate the RGB-Sonar multimodal object detection to improve the perceived performance and robustness of underwater object detection.

![](images/f050a40fce7894f159e0e63d97dbbad7880de57266e15cf5e6525dd6a6e4c64d.jpg)  
(a) Mainstream RGB-LiDAR multimodal object detection.

![](images/98cbd70ad1182ad5aac347783a788f6c03e9ad73679efc3ad34df2ad647dcf1e.jpg)  
(b) RGB-Sonar multimodal object detection (our RSFusionDet).  
Figure 1. Comparisons of multimodal object detection meth ods with multimodal spatial feature misalignment problem. RGB2BEV represents transforming RGB features to BEV features, the same for LiDAR2BEV. Sonar2RGB represents aligning Sonar features to RGB features, and the same for RGB2Sonar. Differently, our RSFusionDet fuses features and detects objects in the RGB and Sonar feature spaces separately. Then, we use the object matching head to match the identical object in RGB and Sonar modalities.

Recently, RGB-Sonar multimodal perception has attracted increasing attention in underwater robotics. Exist ing studies mainly focus on RGB-Sonar object tracking, where RGB and sonar images are jointly utilized to improve target localization robustness under challenging underwater environments. Typical representative works in clude the RGBS50 benchmark and its SCANet tracker [17], which employ cross-attention mechanisms to enhance feature interaction between RGB and sonar modalities for single-object tracking. Compared with RGB-Sonar track ing, RGB-Sonar object detection is a substantially different task. Object detection simultaneously estimates object cat egories and locations for multiple instances in both modal ities, while additionally requiring explicit cross-modal cor respondence between detected objects. These requirements introduce several new challenges that have not been addressed in existing RGB-Sonar tracking studies, including spatially misaligned feature fusion for dense scenes, independent object prediction in heterogeneous sensing spaces, and cross-modal object association among multiple de tected instances. In this paper, we analyze and study the RGB-Sonar multimodal object detection method. We find that RGB-Sonar multimodal object detection is similar to RGB-LiDAR 3D multimodal object detection in that both have the spatial feature misalignment problem. In the field of autonomous driving, 3D multimodal object detection methods (BEVFusion [24], GAFusion [14]) utilizing RGB images and LiDAR point clouds achieve RGB-LiDAR spa tial misalignment feature fusion by mapping RGB and Li-DAR features to the Bird’s Eye View (BEV) feature space (using camera and LiDAR calibration parameters and esti mated RGB image depth maps). Although the RGB-LiDAR spatial feature misalignment problem has been solved, there are still significant differences from our RGB-Sonar mul timodal object detection, their pipelines as shown in Fig. 1. In RGB-LiDAR 3D multimodal object detection, both RGB-LiDAR modalities contain 3D feature information, and RGB-LiDAR multimodal object detection results are uniformly expressed as 3D object detection results. There fore, RGB-LiDAR features can be uniformly fused in the BEV feature space. In RGB-Sonar multimodal object detection, the Sonar modality contains only 2D feature infor mation and lacks position information in the vertical direc tion of objects (due to the principle of 2D sonar imaging), and the spatial plane and coverage area of the RGB-Sonar information is slightly different, so RGB-Sonar multimodal object detection results cannot uniformly express as 3D object detection results. It further leads to RGB-Sonar multimodal object detection being unable to perform feature fusion uniformly in the same feature space. In a word, to achieve RGB-Sonar multimodal object detection, the following key problems need to be considered and addressed:

(1) How to achieve RGB-Sonar spatial misalignment feature fusion? RGB and Sonar images are not only pixel misaligned, but their image data depicts different spatial planes. (In our research, both the camera and sonar are installed facing forward, with RGB images depicting the front view and Sonar images depicting the top view.) Therefore, the features extracted from RGB and Sonar 2D images are spatially misaligned. Since RGB-Sonar multimodal object detection results cannot be expressed uniformly, we choose to fuse RGB-Sonar multimodal features separately in RGB and Sonar feature spaces. We use multi-scale crossattention to fuse RGB-Sonar multimodal features. There is no definite spatial mapping relationship between RGB and Sonar features, and the scale of the object in RGB and Sonar image features may also be inconsistent. Multi-scale crossattention can capture global contextual information to construct spatial correspondence between RGB and Sonar features, align the two features, and achieve RGB-Sonar spatial misalignment feature fusion. Meanwhile, multi-scale crossattention can achieve cross-scale feature fusion of RGB and Sonar to solve the problem of scale inconsistency of objects in RGB and Sonar features. We obtain fused features in the RGB feature space and the Sonar feature space by this feature fusion method, which are used for object detection of RGB and Sonar modalities.

(2) How to express RGB-Sonar multimodal object detection results? According to the above RGB-Sonar spatial misalignment feature fusion method, we use two detection heads to decode the RGB-Sonar fusion features and express the 2D object detection results in RGB and Sonar modalities, respectively. However, this RGB-Sonar multimodal object detection result lacks the corresponding relationship between identical objects in RGB and Sonar modalities, so it is necessary to perform object matching on object detection results of RGB and Sonar to construct the corresponding relationship between objects in RGB and Sonar modalities. RGB-Sonar multimodal object detection results are expressed using RGB object detection results, Sonar object detection results, and object matching results.

In this paper, we create an underwater RGB-Sonar multimodal object detection benchmark with 7073 RGB-Sonar image pairs and 7 classes to support our research, which is called the RGB-Sonar Fusion (RSFusion) dataset. And we also propose evaluation metrics for the underwater RGB-Sonar multimodal object detection. To profit from all this, we present the RGB-Sonar Fusion Detector (RSFusionDet). This dual-branch RGB-Sonar multimodal object detection architecture uses a designed Cross-Attention Fusion module (CAFusion) with deformable attention to align RGB-Sonar features and achieve RGB-Sonar feature fusion. Meanwhile, we design the Object Matching Head (OMHead) and Loss (OMLoss) to match identical objects in RGB and Sonar modalities, and use the object bounding box priori information and IoU weight for the object matching head and loss to improve the performance of object matching. Our method does not require sensor external parameter calibration and feature mapping operations under a fixed sensor configuration, and achieves dynamic fusion of RGB-Sonar features by the global perceptual capacity of attention.

With extensive experiments on the RSFusion dataset, RSFusionDet achieves 76.4/48.6 AP (RGB/Sonar) for object detection and 83.4 F1 $\mathbf { \mathcal { S } } \mathbf { c o r e } _ { m a t c h }$ for object matching. Extensive ablation studies and visualization results validate the effectiveness of our RGB-Sonar feature fusion and object matching methods.

The main contributions are summarized as follows:

• We create an underwater RGB-Sonar multimodal object detection benchmark (RSFusion) with 7073 image pairs and propose evaluation metrics for the benchmark to support the research of RGB-Sonar multimodal object detection.

• We propose a new dual-branch RGB-Sonar multimodal object detection architecture (RSFusionDet) with a new RGB-Sonar multimodal object detection result expression, achieving RGB-Sonar multimodal object detection.

• We design the CAFusion module to fuse RGB-Sonar features, and the OMHead with OMLoss to match identical objects in RGB and Sonar modalities.

• Our RSFusionDet is extensively evaluated on the RSFusion dataset, which validates the effectiveness of RGB-Sonar feature fusion and object matching methods, and outperforms other models.

## 2. Related Work

## 2.1. Unimodal Object Detection

Object detectors based on deep learning are mainly categorized into two-stage and one-stage detectors. Twostage detectors, such as R-CNN series [31, 33], SPP-Net [8], RepPoints [38], etc., usually use a region proposal network (RPN) to propose priori bounding boxes, which are then refined in the second stage. The most representative one-stage detectors, such as YOLO series [28– 30], YOLOX [7], SSD [22], RetinaNet [19], FCOS [34], ATSS [41], TOOD [5], etc., directly output offsets relative to predefined anchors, improving the speed of object detection. In recent years, new fully end-to-end detectors, DETR [2], Deformable DETR [44], DAB-DETR [21],

DINO [40], etc., have emerged, which remove the nonmaximum suppression (NMS) from the above detectors. These detectors could also be used for underwater unimodal object detection.

## 2.2. Multimodal Object Detection

Recently, multimodal (multi-sensor) fusion arouses increased interest in object detection, such as RGB-Thermal [1, 4, 13, 36, 39, 43], RGB-Event [12, 35], RGB-LiDAR [6, 14–16, 23, 24, 37], and RGB-Radar [10, 20]. GAFF [39] guides RGB-T feature fusion by attention mask. CSSA [1] and Scene-agnostic CBAM [4] utilize channel attention and spatial attention to fuse RGB-T features. The multimodal information fused by RGB-T and RGB-E methods is pixel-aligned, and their object detection results are uniformly expressed as 2D object detection results in the same space. DeepFusion [15] makes LiDAR features as queries and constructs aligned RGB features by crossattention layers for feature fusion. BEVFusion [24] utilizes depth estimation of multi-view RGB images and maps RGB features to the BEV feature space based on camera and LiDAR calibration parameters and multi-camera imaging angles. At the same time, the LiDAR features are compressed in the z-axis and finally, the RGB-LiDAR feature fusion is performed by convolution. MetaBEV [6] and CMT [37] construct fused BEV features by learnable BEV queries based on cross-attention layers. GAFusion [14] introduces sparse depth guidance and LiDAR occupancy guidance to generate 3D features with sufficient depth information to adaptively enhance the interaction of multimoda BEV features from a global perspective. RCM-Fusion [10] uses deformable cross-attention layers and radar-guided BEV queries to achieve feature-level fusion of RGB-Radar modality, and then uses radar grid point refinement to perform their instance-level fusion. RGB-LiDAR and RGB-Radar methods fuse multimodal misalignment features in the BEV feature space, and their object detection results are uniformly expressed as 3D object detection results in the same space. The above methods have achieved superior performance in multimodal object detection. However, they are significantly different from our RGB-Sonar multimoda object detection in terms of multimodal feature fusion and object detection result expression. Their multimodal images are pixel-aligned, or their fused features can be unified in the same feature space. But our RGB-Sonar multimodal images are not pixel-aligned, and our fused features cannot be unified in the same feature space. Their object detection results can be expressed uniformly, but ours cannot. RGB-Sonar multimodal object detection is a new direction to be explored and researched.

## 3. RSFusionDet

The overall architecture of RSFusionDet is illustrated in Fig. 2. RGB and Sonar images are fed into two individual and structurally identical backbones to extract features. We fuse multimodal features by cross-attention in both RGB and Sonar feature spaces. RSFusionDet is designed based on the end-to-end DINO [40] detector (using the encoder, decoder, and training method of DINO), and its fused RGB and Sonar features are fed into two individual encoder-decoder pipelines to generate RGB and Sonar detection results, respectively. Finally, we apply the cosine similarity, inspired by CLIP [27], to match RGB and Sonar objects based on the output of the decoder to associate identical objects in RGB and Sonar modalities.

## 3.1. Cross-Attention Fusion Module

RGB and Sonar modalities have the spatial feature misalignment problem, and their corresponding object scales may be inconsistent, resulting in scale differences in the multi-scale RGB-Sonar corresponding object features, which is not conducive to RGB-Sonar feature fusion. To address these problems, we propose a Cross-Attention Fusion (CAFusion) module based on the multi-scale deformable attention module [44], which has a simple architecture and less computational complexity. The CAFusion module contains two pipelines for feature fusion in the RGB and Sonar feature spaces. Each pipeline consists of Cross-Attention Layers (CALayer), where each CALayer consists of a multi-scale deformable attention module and a linear layer followed by a Layer Normalization [11].

In the RGB feature space, we construct Sonar features aligned with RGB features and perform feature fusion on them, as in Fig. 3. We use multi-scale RGB features ${ \pmb x } _ { r g b }$ as query and multi-scale Sonar features $\pmb { x } _ { s o n a r }$ as key and value, and add positional embeddings $p _ { p e }$ (spatial positional encoding [2]) and learnable scale-level embeddings ${ \pmb p } _ { s l e }$ to the query and key. Making RGB features ${ \pmb x } _ { r g b }$ as query and Sonar features $\pmb { x } _ { s o n a r }$ as key and value is to query Sonar Value that is of interest to RGB Query to construct aligned Sonar features with the same dimensional size as RGB features. The positional embeddings and scale-level embeddings enable the network to learn the spatial position and scale-level correspondence between RGB and Sonar object features. The output of the CALayer y is used as the key and value for the next CALayer, and the original query is used as the query for the next CALayer to further query sonar features and align them with RGB features. The new key does not add positional embeddings and scale-level embeddings because the spatial position and scale level of the new feature have changed during alignment. The overall process is formulated as follows:

$$
\pmb q = \pmb { x } _ { r g b } , \quad \pmb k = \pmb { x } _ { s o n a r } , \quad \pmb v = \pmb { x } _ { s o n a r }\tag{1}
$$

$$
\pmb { q } = \pmb { q } + \pmb { p } _ { p e } ^ { r g b } + \pmb { p } _ { s l e } , \quad \pmb { k } = \pmb { k } + \pmb { p } _ { p e } ^ { s o n a r } + \pmb { p } _ { s l e }\tag{2}
$$

$$
{ \pmb y } ^ { i } = \mathrm { C A L a y e r } ^ { i } \left( { \pmb q } ^ { i } , { \pmb k } ^ { i } , { \pmb v } ^ { i } \right)\tag{3}
$$

$$
\pmb q ^ { i + 1 } = \pmb q ^ { i } , \quad \pmb k ^ { i + 1 } = \pmb y ^ { i } , \quad \pmb v ^ { i + 1 } = \pmb y ^ { i }\tag{4}
$$

where i is the index of CALayer, $p _ { p e }$ and $\pmb { p } _ { s l e }$ are added to the input of the first CALayer only $( p _ { s l e }$ is the same in RGB and Sonar modalities). Then, the aligned Sonar features (the output of the last CALayer) $\pmb { x } _ { s o n a r } ^ { a l i g n }$ constructed by CALayers are concatenated with the original RGB features ${ \pmb x } _ { r g b }$ , and concatenated RGB-Sonar features are fused by a linear layer followed by a Layer Normalization [11] and a ReLU activation, the RGB-Sonar fused features $F _ { r g b }$ in the RGB feature space can be expressed as:

$$
\pmb { x } _ { s o n a r } ^ { a l i g n } = \pmb { y } ^ { l a s t }\tag{5}
$$

$$
F _ { r g b } = \mathrm { L i n e a r } \left( \mathrm { C o n c a t } ( \pmb { x } _ { r g b } , \pmb { x } _ { s o n a r } ^ { a l i g n } ) \right)\tag{6}
$$

where $\mathbf { \Delta } y ^ { l a s t }$ represents the output of the last CALayer, and normalization and activation layers are omitted here. In the Sonar feature space, the operation to get the fused features $F _ { s o n a r }$ is identical to $F _ { r g b }$ in the RGB feature space.

Cross-deformable attention can flexibly construct multimodal alignment features by its powerful global perception capability and reduce computational complexity. Meanwhile, multi-scale cross-attention can achieve multimodal cross-scale feature fusion, reducing the effect of scale differences in object features. CAFusion has the advantage of a simple structure and low computational complexity.

## 3.2. Object Matching Head and Loss

Detection results are independent in RGB and Sonar modalities. To construct the corresponding relationship between identical objects in RGB and Sonar modalities, we design the object matching head and object matching loss.

Object Matching Head. Object detection and object matching are two complementary but different tasks in RGB-Sonar multimodal perception. While the detector predicts object categories and locations independently in RGB and sonar modalities, it cannot determine whether two detected objects originate from identical objects. Therefore, we introduce the Object Matching Head (OMHead) to explicitly learn semantic correspondence between detected RGB and sonar objects. The primary objective of OMHead is to establish reliable cross-modal object associations. The Object Matching Head calculates the cosine similarity matrix between RGB and Sonar features (we call them matching features) output from the decoder in RGB and Sonar detection pipelines to match objects in RGB and Sonar modalities.

![](images/b244731bb34f2d80470507ac821764acf59f103dcc3c6922150f634b6ed4e7a1.jpg)  
Figure 2. The overall architecture of our proposed RSFusionDet. CAFusion aligns multi-scale Sonar features to RGB features using deformable attention (DeformAttn) and fuses the aligned Sonar features and RGB features as RGB fusion features, the same for Sonar fusion features. The fused features are used to calculate the corresponding modal object detection results by encoders, decoders, and heads OMHead utilizes matching features and object bounding box priori information for RGB-Sonar object matching by cosine similarity, to produce RGB-Sonar multimodal object detection results with object corresponding relationship.

![](images/449d509c1333e6dad02cf4a362036b75a1546b992255b0d3a3b80e6d77cdffa7.jpg)

Figure 3. The Sonar2RGB fusion part of CAFusion in detail. It aligns Sonar features to RGB features using CALayers and fuses the aligned Sonar features and RGB features, achieving the RGB-Sonar feature fusion in RGB feature space. The RGB2Sonar fusion part is the same as it.  
![](images/c37e939557f9d0866c95bc947b3f5dd260948fc6e534b4afe35da291801c2828.jpg)  
Figure 4. The detailed structure of MLP in OMHead.

To reduce the computational complexity of object matching, we design two matching feature filtering methods for training and inference to reduce the number of features involved in matching. During training, we use the one-toone matching results obtained from the Hungarian loss during DETR object detection training to filter features corresponding to GT from the matching features for OMHead. During inference, we set a filtering threshold of 0.3 (according to our RSFusion dataset) for the RGB-Sonar multimodal object detection results and filter the matching features corresponding to the object detection results whose score is greater than the filtering threshold for OMHead.

In addition to the matching features, we also add the object bounding box priori information for object matching to improve the learning of OMHead for the positional relationships of the RGB-Sonar object pair. We encode the object bounding box coordinate (xywh) as a sinusoidal embedding which is the same as the spatial positional encoding in DETR [2], then dynamically encode the embedding using a multi-layer perceptron (MLP), and finally make it the positional embedding and add it to the matching feature.

OMHead uses MLP (the details in Fig. 4) to encode RGB and Sonar matching features separately and calculates the cosine similarity matrix between RGB and Sonar objects. We set an object matching threshold of 0.8 and take RGB-Sonar object pairs with matching scores (cosine similarity scores) greater than the object matching threshold as object matching results. RGB and Sonar objects can only be matched one to one. Once a one-to-many matching result occurs, we take the one with the greatest matching score as the final result. OMHead could not only achieve RGB-Sonar object matching, but also further guide CAFusion to learn the similarity of object features in RGB and Sonar modalities, thereby achieving better RGB-Sonar feature fusion.

Object Matching Loss. Inspired by CLIP [27] loss, we develop the Object Matching Loss (OMLoss) $\mathcal { L } _ { O M }$ . OMLoss optimizes a symmetric cross-entropy loss to maximize the cosine similarity of identical object embeddings in each pair of RGB-Sonar images and minimize the cosine similarity of non-identical object embeddings for OMHead training.

![](images/16974910f9d8e9ca854442d830e8680c49722fb88ef32c3f5aa11e472d30d534.jpg)  
(a) Sensors.

![](images/8475b6f11f1450987e8b6ae96326f9b67350a213766021d1acb81999cd8fc8ab.jpg)  
(b) Distribution of different object quantities.  
Figure 5. A brief introduction to our RSFusion dataset.

During training, the OMLoss is computed by the predicted object information which is matched to the ground truth by the Hungarian algorithm. The Hungarian algorithm uses a pair-wise matching cost to match ground truth and predicted objects, but in the initial training stage, the IoU of the predicted object bounding box with its ground truth is small, and the corresponding object features contain less object information. Using these features to train OMHead could lead to a training bias, causing the network to learn a large amount of redundant background information for object matching. To solve this problem and improve the learning efficiency of OMHead, we add an IoU weight that is the product of RGB and Sonar object IoUs between predictions and GTs for OMLoss. In the OMLoss $\mathcal { L } _ { O M }$ , the RGB to Sonar object matching loss $\mathcal { L } _ { R 2 S }$ can be expressed as:

$$
\begin{array} { r } { \mathcal { L } _ { R 2 S } = - \sum _ { i } \left( { I o U _ { i } ^ { R } * I o U _ { i } ^ { R 2 S } * \log \left( { e _ { i } ^ { p } / \sum _ { j } { e _ { i j } ^ { p } } } \right) } \right) } \end{array}\tag{7}
$$

where $i , j$ are the index of RGB, Sonar matching objects. $I o U ^ { R }$ is the IoU of the predicted RGB object bounding box with its ground truth, and $I o U ^ { R 2 S }$ is the IoU of the Sonar object (matched to the RGB) with its ground truth. p represents the logits of RGB and Sonar matching objects. The calculation method of Sonar to RGB object matching loss $\mathcal { L } _ { S 2 R }$ is identical to $\mathcal { L } _ { R 2 S }$ . For RSFusionDet, its total loss $\mathcal { L } _ { R S F u s i o n D e t }$ can be expressed as:

$$
\mathcal { L } _ { O M } = \left( \mathcal { L } _ { R 2 S } + \mathcal { L } _ { S 2 R } \right) / 2\tag{8}
$$

$$
\mathcal { L } _ { R S F u s i o n D e t } = \mathcal { L } _ { D e t } ^ { r g b } + \mathcal { L } _ { D e t } ^ { s o n a r } + \mathcal { L } _ { O M }\tag{9}
$$

where object detection losses $\mathcal { L } _ { D e t } ^ { r g b }$ and $\mathcal { L } _ { D e t } ^ { s o n a r }$ are the same as the loss which contains the Hungarian loss and the contrastive denoising loss in DINO [40].

## 4. Datasets and Evaluation Metrics

## 4.1. RSFusion Dataset

We conduct experiments on the underwater RGB-Sonar multimodal object detection dataset, RGB-Sonar Fusion (RSFusion) we create, which includes 7 classes, mannequin, UUV, reflector, metal polyhedron, float, iron ball, and frustum. We use a LUCID TRI050S-QC camera (in Fig. 5a right) and an Oculus MD750d 2D sonar (in Fig. 5a left) with a 1.2 MHz operating frequency to simultaneously capture paired RGB-Sonar images in a tank. The RSFusion dataset includes paired images in light and dark scenes, and paired images at maximum sonar ranges of 5, 10, 15, and 20 meters. The maximum sonar range represents the maximum range at which the sonar sensor scans and images, which can be set. The scanning range is centered around the sonar and a sector area with a radius of the set parameters. The horizontal and vertical fields of view (FOV) for the camera are 30.75 and 23.31 degrees, respectively, while the horizontal FOV for the sonar is 130 degrees. The dataset annotation also includes metadata such as the number of beams n beams, the corresponding beam angle beam angles, the range resolution range resolution, and so on in each sonar image and the corresponding relationship flag of identical objects in RGB and Sonar modalities. The RSFusion dataset is split into 5658/1415 RGB-Sonar image pairs for training/validation (the following data is presented in this format), including 4330/1080 and 1328/335 image pairs in light and dark scenes. We also count the number of paired images when the maximum sonar range is 5, 10, 15, and 20 meters, which are 1559/388, 593/147, 2934/729, and 572/151 image pairs, respectively. The RSFusion dataset contains 21202/5275 objects for training/validation of both RGB and Sonar images, including 4084/1032 object pairs in RGB and Sonar modalities. The distribution of objects in the dataset is shown in Fig. 5b. RSFusion also contains some image pairs where only RGB or Sonar has no objects, and these data are used to improve the robustness of the model for feature fusion. The RGB image size is 1024×1224. The Sonar image has different sizes, 374×678-377×684 and 499×905-505×916, which are related to the maximum sonar range.

## 4.2. Evaluation Metrics

The evaluation metrics of RGB-Sonar multimodal object detection include two parts, object detection metrics and object matching metrics.

## 4.2.1. Object Detection Metrics

For the object detection metrics, we follow the standard COCO [18] evaluation protocol and report the Average Precision (AP) for RGB and Sonar object detection, respectively. Furthermore, according to different kinds of data in the RSFusion dataset, we set the metrics for underwater light and dark scenes $( \mathbf { A P } _ { l i g h t } , \mathbf { A P } _ { d a r k } )$ to evaluate the complementarity between modalities and their robustness to different scenes and the metrics for Sonar modality in different maximum sonar range settings $( \mathrm { A P } _ { s r 5 } , \mathrm { A P } _ { s r 1 0 } , \mathrm { A P } _ { s r 1 5 } , \mathrm { A P } _ { s r 2 0 } )$ to evaluate the impact of different maximum sonar ranges on the performance of

RGB-Sonar feature fusion. All evaluation metrics for both RGB and Sonar object detection are as follows:

AP $( \mathbf { A P _ { 5 0 : 9 5 } } )$ : Average AP value at the IoU threshold from 0.5 to 0.95 (step size 0.05).

$\mathrm { { A P } _ { 5 0 } \mathrm { { : } } }$ AP value at the IoU threshold is 0.5.

$\mathrm { A P } _ { 7 5 } { \vdots }$ AP value at the IoU threshold is 0.75.

$\operatorname { A P } _ { s } \colon \operatorname { A P }$ value for small objects.

$\operatorname { A P } _ { m } { \mathrm { : } }$ AP value for medium objects.

AP<sub>l</sub>: AP value for large objects.

$\mathbf { A P } _ { l i g h t } \colon \mathbf { A P } _ { 5 0 : 9 5 }$ value for objects in the light scene.

$\mathrm { A P } _ { d a r k } \colon \mathrm { A P } _ { 5 0 : 9 5 }$ value for objects in the dark scene.

$\mathrm { A P } _ { s r 5 } \colon \mathrm { A P } _ { 5 0 : 9 5 }$ value for objects at the sr is 5 meters.

$\mathrm { A P } _ { s r 1 0 } \colon \mathrm { A P } _ { 5 0 : 9 5 }$ value for objects at the sr is 10 meters.

$\mathrm { A P } _ { s r 1 5 } \colon \mathrm { A P } _ { 5 0 : 9 5 }$ value for objects at the sr is 15 meters.

$\mathrm { A P } _ { s r 2 0 } \colon \mathrm { A P } _ { 5 0 : 9 5 }$ value for objects at the sr is 20 meters.

## 4.2.2. Object Matching Metrics

For RGB-Sonar object matching, we design evaluation metrics $( \mathrm { P } _ { m a t c h } , \ \mathrm { R } _ { m a t c h }$ , and $\mathrm { F 1 - S c o r e } _ { m a t c h } )$ that is almost identical to Precision, Recall, and F1-Score in object detection. We assign the object with the maximum IoU to the bounding box ground truth by calculating the IoU between the object and the ground truth in the predicted matching results (and remove the result with IoU less than 0), and denote the number of assigned matching object pairs as the total number of samples predicted as correct matches $N _ { p r e d }$ Meanwhile, we calculate the number of correctly matched samples N in $N _ { p r e d } ,$ and denote the total number of samples of the object matching ground truth as $N _ { g t }$ . The evaluation metrics for object matching can be expressed as:

$$
\mathrm P _ { m a t c h } = N / N _ { p r e d } , \quad \mathrm R _ { m a t c h } = N / N _ { g t }\tag{10}
$$

$$
\mathrm { F l - S c o r e } _ { m a t c h } = 2 * \frac { \mathrm { P } _ { m a t c h } * \mathrm { R } _ { m a t c h } } { \mathrm { P } _ { m a t c h } + \mathrm { R } _ { m a t c h } }\tag{11}
$$

All evaluation metrics for RGB-Sonar object matching are as follows:

$\mathrm { P } _ { m a t c h } { : }$ Precision for RGB-Sonar object matching.   
$\mathrm { R } _ { m a t c h } ;$ Recall for RGB-Sonar object matching.   
$\mathbf { \mathrm { F 1 } } { \cdot } \mathbf { \mathrm { S c o r e } } _ { m a t c h } { \boldsymbol { : } } \mathbf { \mathrm { F 1 } }$ -Score for RGB-Sonar object matching.

## 5. Experiments

## 5.1. Implementation Details

All models are implemented and tested based on the MMDetection [3] framework. We adopt the pre-trained ResNet-50 [9] as the backbone for RGB and Sonar stream for RSFusionDet. All models are trained on 2 NVIDIA A6000 GPUs with batch size 4 per GPU. The RGB and Sonar input image resolutions are $5 1 2 \times 6 4 0$ and $5 1 2 \times 9 2 8$ respectively, and we use the horizontal RandomFlip data augmentation on the input image. During the training stage, we use AdamW [26] optimizer, with weight decay of $1 e ^ { - 4 }$ and initial learning rates (lr) as $1 e ^ { - 4 }$ and $1 e ^ { - 5 }$ for the encoder-decoder and backbone, respectively. We take overall 12 training epochs for RSFusionDet and drop lr at the 11th epoch multiplying by 0.1.

In the comparison experiment, both RGB-LiDAR methods and our RGB-Sonar method have the spatial feature misalignment problem. However, these RGB-LiDAR fusion methods [6, 14, 24, 37], cannot be applied to our RGB-Sonar fusion method. Most of these methods require feature mapping using predicted depth maps and sensor calibration parameters, or require priori object coordinate position information, which cannot be implemented in our RGB-Sonar multimodal object detection method.

During training and test, both the RGB and Sonar images of RSFusionDet are inputted in pairs simultaneously. In order to achieve fairness in comparison with unimodal object detection methods, we design a dual-stream RGB-Sonar separate object detection framework, ensuring consistency between the RGB and Sonar image data streams of the unimodal object detection method and our RSFusion-Det during training and test. As shown in Fig. 6, the RGB-Sonar separate object detection framework we design includes two independent unimodal object detectors for RGB and Sonar modalities, respectively. Like our RSFusionDet, it can simultaneously train and test for both RGB and Sonar modalities. During the training, the losses of these two unimodal object detectors are summed up to train them for both RGB and Sonar modalities simultaneously, making them the same as our RSFusionDet training data stream for fair comparison experiments. Difference from RSFusionDet, the RGB-Sonar separate object detection framework cannot implement RGB-Sonar feature fusion and RGB-Sonar object matching. In the experiment, we train and test unimodal object detection models on the RSFusion dataset using the RGB-Sonar separate object detection framework without RGB-Sonar feature fusion, and call its modality as RGB or Sonar, i.e. R/S. Meanwhile, we call the modality of RS-FusionDet with RGB-Sonar feature fusion as the fusion of RGB and Sonar, i.e. R+S. In the experiment, all models are trained and tested on the training set and the validation set of the RSFusion dataset.

Furthermore, at present, the only publicly available RGB-Sonar related dataset is RGBS50 [17]. This dataset is an RGB-Sonar single object tracking dataset constructed in our laboratory, which consists of continuous frame images with single object annotation information. Its object diversity is far less than our RSFusion dataset. And both RGBS50 and RSFusion datasets are compiled from the same experimental data, so it is convincing enough to conduct validation only on the RSFusion dataset. Although both RGBS50 and RSFusion datasets are compiled from the same experimental data, every frame is independently re-annotated for multimodal object detection. Compared with RGBS50, which only requires tracking one initialized object, RSFusion requires annotating all visible objects in both RGB and sonar images, assigning category labels, generating modality-specific bounding boxes, and establishing cross-modal correspondence labels for matched object pairs.

![](images/b059fe661f47e5e6fa9b0d8ce25be8df3cdf64607b196ea3343af3afadadef2b.jpg)  
Figure 6. RGB-Sonar separate object detection framework for training and test.

Table 1. Comparisons with other state-of-the-art models. The metrics are object detection APs in RGB/Sonar modality. Our RSFusionDe has object matching evaluation metrics (80.9 $\mathrm { P } _ { m a t c h }$ , 86.0 $\mathrm { R } _ { m a t c h }$ , and 83.4 F1-Score $\mathbf { \chi } _ { m a t c h } )$ in Table 14. Other models lack object matching, so their evaluation metrics are ignored here. † denotes the adoption of its fusion method.
<table><tr><td>Model</td><td>Modality</td><td>#Params (M)</td><td>GFLOPs</td><td>AP</td><td> $\mathbf { A P } _ { 5 0 }$ </td><td> $\mathbf { A P } _ { 7 5 }$ </td><td> ${ \bf A P } _ { s }$ </td><td> $\mathbf { A P } _ { m }$ </td><td> $\mathbf { A P } _ { l }$ </td></tr><tr><td>Faster R-CNN [31]</td><td>R/S</td><td>83</td><td>179</td><td>73.4 / 42.9</td><td>97.0 / 86.4</td><td>82.7 / 38.3</td><td>15.4/ 37.3</td><td>38.7 / 47.0</td><td>79.5 / 68.7</td></tr><tr><td>YOLOv3 [29]</td><td>R/S</td><td>123</td><td>152</td><td>66.2 / 38.4</td><td>93.5 / 81.5</td><td>74.6/31.1</td><td>12.9 / 32.4</td><td>33.0 / 42.1</td><td>72.9 / 65.5</td></tr><tr><td>FCOS [34]</td><td>R/S</td><td>64</td><td>154</td><td>70.8 / 39.3</td><td>96.6 / 83.2</td><td>79.7 / 31.9</td><td>12.4 / 33.8</td><td>35.5 / 43.1</td><td>77.4 / 68.1</td></tr><tr><td>ATSS [41]</td><td>R/S</td><td>64</td><td>158</td><td>73.6/43.5</td><td>96.6/ 85.7</td><td>82.4 / 38.9</td><td>11.6/ 38.3</td><td>35.2 / 46.8</td><td>80.0 / 69.1</td></tr><tr><td>Sparse R-CNN [33]</td><td>R/S</td><td>212</td><td>129</td><td>70.3 / 40.4</td><td>95.1 / 86.6</td><td>78.1 / 30.1</td><td>10.8 / 36.6</td><td>35.0 / 42.4</td><td>78.1/ 67.7</td></tr><tr><td>TOOD [5]</td><td>R/S</td><td>64</td><td>155</td><td>63.3 / 44.7</td><td>89.5 / 86.5</td><td>71.0 / 40.7</td><td>10.2 / 38.6</td><td>28.8 / 47.4</td><td>68.9 / 67.4</td></tr><tr><td>YOLOX-L [7]</td><td>R/S</td><td>108</td><td>152</td><td>68.4 / 43.0</td><td>90.9 / 89.8</td><td>76.3 / 36.4</td><td>9.3 / 37.0</td><td>32.5 / 45.7</td><td>77.0/ 67.1</td></tr><tr><td>Deformable DETR++ [44]</td><td>R/S</td><td>82</td><td>160</td><td>72.1 / 43.5</td><td>97.0/91.7</td><td>80.4 / 35.2</td><td>14.2 / 38.7</td><td>33.5 / 46.1</td><td>78.6 / 67.8</td></tr><tr><td>DAB-DETR [21]</td><td>R/S</td><td>87</td><td>85</td><td>63.5 / 30.8</td><td>94.0 / 73.4</td><td>71.4 / 18.5</td><td>22.7 / 25.7</td><td>26.7 / 35.0</td><td>71.0 / 57.8</td></tr><tr><td>DINO [40]</td><td>R/S</td><td>95</td><td>233</td><td>75.7 / 47.2</td><td>97.1/ 93.7</td><td>83.0 / 41.5</td><td>21.5 / 42.7</td><td>45.3 / 51.4</td><td>83.2 / 69.8</td></tr><tr><td>DINO+GAFF† [39]</td><td>R+S</td><td>96</td><td>240</td><td>75.9 / 47.8</td><td>97.3 / 94.0</td><td>83.2 / 42.2</td><td>22.5 / 43.1</td><td>48.0 / 52.5</td><td>83.6 / 70.0</td></tr><tr><td>DINO+SCANet† [17]</td><td>R+S</td><td>97</td><td>245</td><td>76.1 / 48.1</td><td>97.4 / 94.2</td><td>83.3 / 42.8</td><td>22.9 / 43.4</td><td>49.2 / 53.0</td><td>83.7 / 70.1</td></tr><tr><td>RSFusionDet</td><td>R+S</td><td>98</td><td>249</td><td>76.4 / 48.6</td><td>97.5 / 94.4</td><td>83.5 / 43.9</td><td>23.6 / 43.8</td><td>51.1 / 54.1</td><td>83.9 / 70.3</td></tr></table>

Table 2. Comparisons with other state-of-the-art models with CA-Fusion and OMHead. CA and OM represent CAFusion and OM-Head, respectively. And RSFusionDet is $\mathrm { D I N O } _ { + \mathbf { C A } + \mathbf { O M } } .$
<table><tr><td>Model</td><td>Modality</td><td>AP</td></tr><tr><td>Faster R-CNN+CA+OM</td><td>R+S</td><td> $7 4 . 2 ( + 0 . 8 ) / 4 4 . 3 ( + 1 . 4 )$ </td></tr><tr><td> $\Upsilon \mathrm { O L O v 3 } _ { + \mathbf { C A } + \mathbf { O M } }$ </td><td>R+S</td><td> $6 6 . 8 ( + 0 . 6 ) / 3 9 . 5 ( + 1 . 1 )$ </td></tr><tr><td> $\mathrm { F C O S } _ { + \mathbf { C A } + \mathbf { O M } }$ </td><td>R+S</td><td>71.5(+0.7) / 40.5(+1.2)</td></tr><tr><td> $\mathbf { A T S S _ { + C A + O M } }$ </td><td>R+S</td><td>74.3(+0.7) / 44.8(+1.3)</td></tr><tr><td>Sparse R  $\mathbf { \partial . C N N _ { + C A + O M } }$ </td><td>R+S</td><td>71.0(+0.7) / 41.6(+1.2)</td></tr><tr><td> $\mathrm { T O O D } _ { + \mathbf { C A } + \mathbf { O M } }$ </td><td>R+S</td><td>63.9(+0.6) / 46.2(+1.5)</td></tr><tr><td> $\mathbf { Y O L O X - L _ { + C A + O M } }$ </td><td>R+S</td><td>69.0(+0.6) / 44.3(+1.3)</td></tr><tr><td>Deformable  $\mathrm { D E T R + + _ { + C A + O M } }$ </td><td>R+S</td><td>72.8(+0.7) / 44.8(+1.3)</td></tr><tr><td>DAB  $\mathbf { \partial \cdot D E T R _ { + C A + O M } }$ </td><td>R+S</td><td>64.1(+0.6) / 31.8(+1.0)</td></tr><tr><td>RSFusionDet  $\mathbf { ( D I N O _ { + C A + O M } ) }$ </td><td>R+S</td><td>76.4(+0.7) / 48.6(+1.4)</td></tr></table>

## 5.2. Comparisons with State-of-the-arts

## 5.2.1. Comparisons with Other Models

As shown in Table 1, we report the results of RSFusion-Det on the RSFusion validation set, and compare them with other state-of-the-art unimodal models without RGB-Sonar feature fusion and object matching. The results show that RSFusionDet outperforms all other models, with 76.4(+0.7)/48.6(+1.4) AP (RGB/Sonar AP) (compared to the best model DINO [40]). RSFusionDet significantly improves both RGB and Sonar detection, as RGB provides richer feature information for Sonar, resulting in higher improvement on detection performance in the Sonar modality. And RSFusionDet has a significant performance improvement across all object scales and outperforms all other models, especially on medium objects with a 51.1/54.1 AP. It proves that RGB-Sonar feature fusion is beneficial for improving the performance of object detection. Compared to its base model DINO, RSFusionDet has added CAFusion and OMHead, but only gains 3 M #Params and 16 GFLOPs. Meanwhile, we plug CAFusion and OMHead into other unimodal models and compare them with our RSFusionDet in Table 2 (Note that RSFusionDet is equivalent to DINO with CAFusion and OMHead.). After plugging in CAFusion and OMHead, these models can also achieve RGB-Sonar feature fusion, with improvements of 0.6-0.8 AP and 1.0-1.5 AP in RGB and Sonar modalities, respectively. And our RSFusionDet still outperforms other models with CA-Fusion and OMHead. It further proves the effectiveness of CAFusion and OMHead in RGB-Sonar feature fusion. Besides, we visualize the detection results and the RGB-Sonar object attention maps of RSFusionDet, as in Figs.

![](images/83c08e0a6f7ab29fb053835b9917aa33449f3aff334aaded59f0b15ba330a6ce.jpg)

Figure 7. Comparisons of object detection and matching results (lines) on the RSFusion validation set with light and dark scenes. Note that some objects appear only in one modality because the RGB camera and forward-looking sonar have different fields of view and sensing ranges. Such modality-specific observations are common in practical underwater perception systems.  
![](images/354f7e69720bbd06199ddec1efb3b5986d8b052c969505ae1d7cc902b26b72a5.jpg)  
Figure 8. RGB-Sonar corresponding object attention maps by Grad-CAM [32] method. The attention map is visualized on the layer behind the CAFusion module. The RGB-Sonar object attention gradient is obtained only from the object bbox and iou losses in RGB modality. The attention in RGB and Sonar images represents the degree of correlation between object features in RGB and Sonar modalities, indicating the feature fusion effect of the CAFusion module.

7 and 8. DINO has many erroneous detections and omissions, while our RSFusionDet has achieved more accurate detection results in Fig. 7. Moreover, our RSFusionDet enables effective RGB-Sonar object matching. RSFusionDet achieves more accurate object localization and classification in Sonar modality, which compensates for the poor accuracy of object classification in Sonar modality. It proves that RGB-Sonar multimodal object detection is beneficial for improving the detection performance between modalities. However, RSFusionDet faces difficulties in detecting small objects with occlusion (in the first row of Fig. 7), mainly due to the influence of underwater noise (the attenuation effect of the water medium on light waves). And RS-FusionDet also has the problem of RGB-Sonar object mismatch (the green line connecting the float in RGB modality and the frustum in Sonar modality), as shown in the last row of Fig. 7. It is mainly caused by the high similarity of object features between different categories of RGB and Sonar modalities, which is known as the feature confusion problem of neural networks. The attention of identical object has a good correspondence between RGB and Sonar modalities in Fig. 8, proving that the attention relying on RGB object can be transmitted to the corresponding Sonar object, and the CAFusion module effectively implements RGB-Sonar spatial misalignment feature fusion.

Table 3. Comparisons (AP) on different classes in RSFusion dataset.
<table><tr><td>Model</td><td>Modality</td><td>mannequin</td><td>UUV</td><td>reflector</td><td>metal polyhedron</td><td>float</td><td>iron ball</td><td>frustum</td></tr><tr><td>Faster R-CNN</td><td>R/S</td><td>78.2 / 60.1</td><td>76.7 / 54.2</td><td>86.2 / 58.4</td><td>51.0 / 23.6</td><td>66.0 / 21.9</td><td>75.1 / 36.7</td><td>80.4 / 45.6</td></tr><tr><td>YOLOv3</td><td>R/S</td><td>60.6 / 54.8</td><td>66.4 / 48.8</td><td>80.5 / 51.6</td><td>50.9 / 17.8</td><td>62.1 /22.8</td><td>70.1/34.3</td><td>72.5 / 38.5</td></tr><tr><td>FCOS</td><td>R/S</td><td>72.6 / 54.9</td><td>72.8 / 53.6</td><td>82.3 / 55.2</td><td>51.5 / 18.1</td><td>64.3 / 19.7</td><td>74.5 /32.8</td><td>77.6/ 40.9</td></tr><tr><td>ATSS</td><td>R/S</td><td>79.2 / 60.6</td><td>73.5 / 54.4</td><td>86.3 / 59.7</td><td>51.5 / 24.3</td><td>65.5 / 18.8</td><td>78.8 / 38.6</td><td>80.2 / 48.3</td></tr><tr><td>Sparse R-CNN</td><td>R/S</td><td>75.1 / 56.6</td><td>75.0/51.1</td><td>82.3 / 51.6</td><td>47.4 / 22.3</td><td>63.9 / 29.2</td><td>74.3 / 30.6</td><td>74.0 / 41.3</td></tr><tr><td>TOOD</td><td>R/S</td><td>54.0 / 61.0</td><td>62.2 / 54.5</td><td>74.5 / 60.0</td><td>51.7 / 26.6</td><td>65.6 / 22.4</td><td>68.1 / 40.4</td><td>66.7 / 47.8</td></tr><tr><td>YOLOX-L</td><td>R/S</td><td>65.7/57.6</td><td>77.4/51.8</td><td>83.2 / 56.2</td><td>48.6 / 24.5</td><td>59.2/ 27.1</td><td>68.9/ 37.9</td><td>75.9 / 45.9</td></tr><tr><td>Deformable DETR++</td><td>R/S</td><td>79.4 / 59.0</td><td>77.3 / 49.8</td><td>83.7 / 53.7</td><td>51.0/ 31.5</td><td>64.4 /31.2</td><td>72.1/ 35.6</td><td>76.7 / 43.5</td></tr><tr><td>DAB-DETR</td><td>R/S</td><td>62.2 / 42.9</td><td>63.0/ 39.7</td><td>79.9 / 42.7</td><td>43.9 / 16.7</td><td>60.6/11.3</td><td>65.2 / 24.7</td><td>69.9 / 37.5</td></tr><tr><td>DINO</td><td>R/S</td><td>84.0/61.9</td><td>81.5 / 54.5</td><td>88.1 / 58.6</td><td>50.8 /35.7</td><td>66.0/ 33.6</td><td>79.0/ 39.9</td><td>80.7 / 46.3</td></tr><tr><td>DINO+GAFF†</td><td>R+S</td><td>84.1 / 62.0</td><td>81.7 / 54.6</td><td>88.3 / 59.1</td><td>51.2/ 36.0</td><td>66.0/ 34.5</td><td>79.1 / 40.2</td><td>81.3 / 47.1</td></tr><tr><td>DINO+SCANet†</td><td>R+S</td><td>84.3 / 62.0</td><td>81.8 / 54.7</td><td>88.4 / 59.6</td><td>51.6/ 36.3</td><td>66.0/35.4</td><td>79.2 / 40.5</td><td>82.0/ 47.9</td></tr><tr><td>RSFusionDet</td><td>R+S</td><td>84.4 / 62.1</td><td>82.0 / 54.8</td><td>88.6 / 60.1</td><td>52.0 / 36.7</td><td>66.0 / 36.5</td><td>79.3 / 40.9</td><td>82.7 / 48.8</td></tr></table>

Table 4. Comparisons (AP) on light and dark scenes.
<table><tr><td>Model</td><td>Modality</td><td> $\mathbf { A P } _ { l i g h t }$ </td><td> $\mathbf { A P } _ { d a r k }$ </td></tr><tr><td>Faster R-CNN</td><td>R/S</td><td>76.7 / 45.0</td><td>58.5 / 33.4</td></tr><tr><td>YOLOv3</td><td>R/S</td><td>68.2 / 40.8</td><td>52.0 / 28.5</td></tr><tr><td>FCOS</td><td>R/S</td><td>74.5 / 41.0</td><td>56.4 / 30.4</td></tr><tr><td>ATSS</td><td>R/S</td><td>76.5 / 45.0</td><td>59.0 / 34.5</td></tr><tr><td>Sparse R-CNN</td><td>R/S</td><td>73.9 / 42.3</td><td>55.5 / 33.8</td></tr><tr><td>TOOD</td><td>R/S</td><td>67.2 / 46.6</td><td>41.2 / 34.0</td></tr><tr><td>YOLOX-L</td><td>R/S</td><td>74.5 / 44.5</td><td>43.5 / 35.7</td></tr><tr><td>Deformable DETR++</td><td>R/S</td><td>75.5 / 45.0</td><td>58.8 / 36.4</td></tr><tr><td>DAB-DETR</td><td>R/S</td><td>66.6 / 32.4</td><td>49.0 / 25.4</td></tr><tr><td>DINO</td><td>R/S</td><td>79.8 / 48.8</td><td>59.6 / 40.3</td></tr><tr><td>DINO+GAFF†</td><td>R+S</td><td>80.0 / 49.2</td><td>59.9 / 40.7</td></tr><tr><td>DINO+SCANet†</td><td>R+S</td><td>80.3 / 49.5</td><td>60.2 / 41.1</td></tr><tr><td>RSFusionDet</td><td>R+S</td><td>80.6 / 49.9</td><td>60.6 / 41.5</td></tr></table>

## 5.2.2. Comparisons on Classes

We compare the performance of the models on different classes in Table 3. Our RSFusionDet achieves the best performance on all classes and significantly improves RGB and Sonar detection performance on reflector (+0.2/+0.5 AP) and frustum (+0.7/+0.9 AP) objects, compared to the other best performance. It indicates the effectiveness of RSFusionDet for objects of different shapes.

## 5.2.3. Comparisons on Light and Dark Scenes

We evaluate the performance of the models on the light and dark scenes of the RSFusion validation set, in Table

4. Our RSFusionDet improves 0.8/1.1 and 1.0/1.2 AP upon DINO [40] in light and dark scenes, respectively. Notably, the improvement in RGB detection is greater in dark scenes than in light scenes. In dark scenes, the features of the RGB modality are significantly reduced, while the Sonar modality is unaffected by dark scenes and can consistently provide complementary object features. It demonstrates that the superiority of RSFusionDet in underwater dark scenes, which significantly improves the performance of object detection in dark scenes.

## 5.2.4. Comparisons on Maximum Sonar Ranges

The performance of the models on the validation set with different maximum sonar range image pairs is shown in Table 5. The results present that the improvement of RSFusionDet at different maximum sonar ranges, and as the maximum sonar range increases, the improvement of RGB detection performance also improves, while the improvement of Sonar detection performance decreases, compared to the best model DINO [40]. In the maximum sonar range of 5 and 15 meters, the improvement of detection performance of RGB and Sonar is lower because these two data are more numerous and contain a large number of difficult objects (sr5, sr15, and sr10, sr20 can be separated to observe performance changes), such as small objects, objects in dark scenes, and so on. Excluding the influence of this factor, an increase in object distance results in less object feature information in RGB, while Sonar can provide complementary object feature information for RGB, thereby improving RGB detection performance at large maximum sonar ranges. As the object distance increases, the rich object feature information provided to Sonar by RGB is reduced, and the scale of the object in the Sonar image decreases, resulting in reduced Sonar detection performance at large maximum sonar ranges.

Table 5. Comparisons (AP) on different maximum sonar ranges.
<table><tr><td>Model</td><td>Modality</td><td> $\mathbf { A P } _ { s r 5 }$ </td><td> $\mathbf { A P } _ { s r 1 0 }$ </td><td> $\mathbf { A P } _ { s r 1 5 }$ </td><td> $\mathbf { A P } _ { s r 2 0 }$ </td></tr><tr><td>Faster R-CNN</td><td>R/S</td><td>72.0 / 55.5</td><td>84.7 / 46.5</td><td>70.9 / 37.0</td><td>74.9 / 39.2</td></tr><tr><td>YOLOv3</td><td>R/S</td><td>60.8 / 49.2</td><td>75.5 / 39.7</td><td>65.0 / 33.5</td><td>67.1 / 35.0</td></tr><tr><td>FCOS</td><td>R/S</td><td>71.4 /52.0</td><td>83.0 / 39.9</td><td>67.9 / 35.3</td><td>74.4 / 35.2</td></tr><tr><td>ATSS</td><td>R/S</td><td>72.1 / 52.1</td><td>86.2 / 42.3</td><td>70.9 / 40.3</td><td>75.9 / 40.4</td></tr><tr><td>Sparse R-CNN</td><td>R/S</td><td>73.9 /52.1</td><td>81.2 / 41.6</td><td>66.8 / 37.5</td><td>70.7 / 37.8</td></tr><tr><td>TOOD</td><td>R/S</td><td>63.8 / 53.8</td><td>76.6 / 44.4</td><td>60.4 / 40.4</td><td>64.0 / 42.2</td></tr><tr><td>YOLOX-L</td><td>R/S</td><td>70.4 /51.2</td><td>81.3 / 44.4</td><td>63.7 / 38.9</td><td>73.3 / 38.2</td></tr><tr><td>Deformable DETR++</td><td>R/S</td><td>73.7 / 54.8</td><td>82.7 / 44.8</td><td>69.2 / 39.7</td><td>74.2 / 40.1</td></tr><tr><td>DAB-DETR</td><td>R/S</td><td>66.0/ 40.3</td><td>75.2 / 29.1</td><td>60.9 / 28.3</td><td>62.3 / 24.0</td></tr><tr><td>DINO</td><td>R/S</td><td>76.6 / 55.7</td><td>86.4 / 49.4</td><td>73.0 / 43.4</td><td>76.7 / 45.6</td></tr><tr><td>DINO+GAFF†</td><td>R+S</td><td>76.7 / 56.2</td><td>87.0 / 50.1</td><td>73.1 / 43.8</td><td>77.4 /46.0</td></tr><tr><td>DINO+SCANet†</td><td>R+S</td><td>76.8 / 56.8</td><td>87.7 / 50.7</td><td>73.2 / 44.2</td><td>78.1 / 46.4</td></tr><tr><td>RSFusionDet</td><td>R+S</td><td>76.9 / 57.3</td><td>88.4 / 51.3</td><td>73.4 / 44.6</td><td>78.9 / 46.9</td></tr></table>

Table 6. Ablation study for CAFusion and OMHead. The selfattention represents using self-attention instead of cross-attention in CAFusion, removing the feature fusion of RGB and Sonar.
<table><tr><td>CAFusion</td><td>OMHead</td><td>AP</td><td> $\mathbf { F 1 - S c o r e } _ { m a t c h }$ </td></tr><tr><td rowspan="4">√self-attention √</td><td></td><td>75.7 / 47.2</td><td></td></tr><tr><td></td><td>75.7 / 47.3</td><td>一</td></tr><tr><td></td><td>76.2 / 48.3</td><td></td></tr><tr><td>√</td><td>75.8 / 47.4</td><td>70.1</td></tr><tr><td>√self-attention</td><td>√</td><td>75.8 / 47.5</td><td>70.4</td></tr><tr><td>√</td><td>√</td><td>76.4 / 48.6</td><td>83.4</td></tr></table>

## 5.3. Ablation Studies

## 5.3.1. Ablation Study on RSFusionDet

Table 6 shows the ablation of CAFusion and OMHead for RSFusionDet. The CAFusion significantly improves RGB and Sonar detection performance by 0.5/1.1 AP, and the OMHead achieves object matching between RGB and Sonar with an 83.4 $\mathrm { F 1 - S c o r e } _ { m a t c h }$ Besides, we change the cross-attention to self-attention in the CAFusion module (i.e. removing RGB-Sonar feature fusion), with the same #Params and GFLOPs, and find that the performance does not change in RGB modality and only improves 0.1 AP in Sonar modality. The results illustrate the effectiveness of CAFusion and OMHead for RGB-Sonar feature fusion and object matching. And it indicates that the gain of CAFusion purely comes from RGB-Sonar feature fusion rather than from the additional parameter counts and computational complexity introduced by CAFusion. Moreover, OMHead has slightly improved detection performance by 0.2/0.3 AP based on CAFusion, demonstrating that object matching contributes to RGB-Sonar feature fusion. OMHead also improves detection performance by 0.1/0.2 AP without CAFusion (or with self-attention CAFusion), but the object matching performance drops to 70.1 (or 70.4) $\mathrm { F 1 - S c o r e } _ { m a t c h }$ , indicating the importance of CAFusion in RGB-Sonar feature fusion. The CAFusion not only achieves RGB-Sonar feature fusion and improves the performance of RGB-Sonar multimodal object detection, but also enhances the correlation of RGB-Sonar features in object matching.

Table 7. Ablation study for fusion methods.
<table><tr><td>Fusion Method</td><td>AP</td><td> $\mathbf { F 1 - S c o r e } _ { m a t c h }$ </td></tr><tr><td>Add</td><td>76.0 / 47.9</td><td>78.9</td></tr><tr><td>Concat</td><td>76.4 / 48.6</td><td>83.4</td></tr></table>

Table 8. Ablation study for the structure of CAFusion.
<table><tr><td>Module</td><td>AP</td></tr><tr><td>CAFusion</td><td>76.2 / 48.3</td></tr><tr><td>CAFusion w/o  $p _ { p e }$ </td><td>75.9 / 47.8</td></tr><tr><td>CAFusion w/o  ${ \pmb p } _ { s l e }$ </td><td>76.1 / 47.9</td></tr><tr><td>CAFusion w/o Linear (in CALayer)</td><td>75.9 / 47.7</td></tr></table>

## 5.3.2. Ablation Study on CAFusion

To explore the performance of CAFusion, we perform ablation on its settings. As Eq. 6, RGB and aligned Sonar features are fused using the concatenate operation, Sonar and aligned RGB features are the same. In Table 7, we perform ablation on the fusion method of add and concatenate operations, and find that the add operation is not as effective as the concatenate operation in both RGB-Sonar multimodal object detection and object matching performance. The fusion method of the add operation can lead to the loss of important detailed information. Even if RGB and Sonar are spatially aligned, their features will still have spatial deviations. The add operation can cause interference between inconsistent areas. Compared to the concatenate operation, the add operation has a weaker ability in representing nonlinear relationships and cannot model more complex interactions between features. In Table 8, we ablate the structure of CAFusion including positional embeddings $p _ { p e } .$ , scale-level embeddings $\mathbf { \Pi } _ { p _ { s l e } , }$ and Linear in CALayer. The results indicate that without these settings, the performance of RSFusionDet decreases. It also proves that $p _ { p e }$ and ${ \mathbf { } } p _ { s l e }$ provide spatial position encoding information and scale-level position information for RGB and Sonar features, which contributes to RGB-Sonar spatial misalignment feature fusion, and the Linear in CALayer provides effective feature encoding in RGB-Sonar feature alignment. Although removing the scale-level embeddings $\pmb { p } _ { s l e }$ has a relatively small impact on the detection performance of the RGB modality (only reducing 0.1 AP), it has a significant effect on the Sonar modality. This is mainly because the objects in the RGB modality tend to be large-scale, and their multiscale features contain rich object information. Therefore, the RGB modality is not sensitive to $\pmb { p } _ { s l e }$ . Additionally, we evaluate the CAFusion w/o $\pmb { p } _ { s l e }$ on a partial dataset containing small objects, showing an improvement of 0.8/0.9 AP.

Table 9. Ablation study for the number of layers in CAFusion.
<table><tr><td>#layers</td><td>#Params (M)</td><td>GFLOPs</td><td>AP</td></tr><tr><td>1</td><td>97</td><td>241</td><td>75.9 / 47.5</td></tr><tr><td>2</td><td>97</td><td>246</td><td>75.9 / 47.9</td></tr><tr><td>3</td><td>98</td><td>251</td><td>76.2 / 48.3</td></tr><tr><td>4</td><td>99</td><td>256</td><td> $7 6 . 0 / 4 7 . 8$ </td></tr><tr><td>5</td><td>99</td><td>261</td><td>76.1 / 48.0</td></tr><tr><td>6</td><td>100</td><td>266</td><td>76.0 / 47.9</td></tr></table>

Table 10. Ablation study for the number of layers (independently in RGB and Sonar feature space) in CAFusion. Vertical and horizontal headers represent the number of layers (#layers) of CALayer in RGB and Sonar feature spaces, respectively. The metrics are object detection AP in RGB/Sonar modality.
<table><tr><td>RGB\Sonar</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>1</td><td>75.9/47.5</td><td>75.9/48.0</td><td>75.8/48.3</td><td>75.8/47.9</td><td>75.8/47.8</td><td>75.8/47.8</td></tr><tr><td>2</td><td>75.9/47.5</td><td>75.9/47.9</td><td>75.9/48.3</td><td>75.9/47.9</td><td>75.8/47.9</td><td>75.8/47.9</td></tr><tr><td>3</td><td>76.2/47.5</td><td>76.2/47.9</td><td>76.2/48.3</td><td>76.2/47.8</td><td>76.1/47.7</td><td>76.1/47.8</td></tr><tr><td>4</td><td>76.1/47.4</td><td>76.1/47.9</td><td>76.1/48.3</td><td>76.0/47.8</td><td>76.0/47.9</td><td>76.0/48.0</td></tr><tr><td>5</td><td>76.0/47.5</td><td>76.1/47.8</td><td>76.1/48.3</td><td>76.1/47.8</td><td>76.1/48.0</td><td>76.0/47.9</td></tr><tr><td>6</td><td>76.0/47.3</td><td>76.0/47.8</td><td>76.0/48.2</td><td>76.0/47.7</td><td>76.0/47.8</td><td>76.0/47.9</td></tr></table>

Table 11. Ablation study for the number of heads in CAFusion.
<table><tr><td>#heads</td><td>#Params (M)</td><td>GFLOPs</td><td>AP</td></tr><tr><td>1</td><td>97</td><td>247</td><td> $7 6 . 1 / 4 8 . 1$ </td></tr><tr><td>2</td><td>97</td><td>247</td><td> $7 6 . 1 / 4 8 . 2$ </td></tr><tr><td>4</td><td>98</td><td>249</td><td> $7 6 . 4 / 4 8 . 6$ </td></tr><tr><td>8</td><td>98</td><td>251</td><td> $7 6 . 2 / 4 8 . 3 $ </td></tr></table>

It indicates that $\pmb { p } _ { s l e }$ can bring greater performance gain on RGB modality for small objects, and ${ \pmb p } _ { s l e }$ proves its significance for both RGB and Sonar modalities. In contrast, $p _ { p e }$ has a greater impact on the detection performance of both RGB and Sonar modalities, as it plays a key role in helping RSFusionDet learn the spatial position correspondence between RGB and Sonar object features.

In Table 9, we ablate the cross-attention layers, each of which has only about 5 GFLOPs. The results illustrate that RSFusionDet performs best when the number of layers is 3 in both RGB and Sonar modalities, and that the number of layers has different effects on RGB and Sonar modalities. And we set different layers for RGB and Sonar modalities to further explore the CAFusion module in Table 10. It indicates that the number of layers in CAFusion has a relatively small impact on the RGB modality and a great impact on the Sonar modality. This is mainly because the object feature information in the RGB modality is much richer than that in the Sonar modality. Following RGB-Sonar feature fusion, Sonar modality provides a small amount of object feature information for RGB modality, resulting in RGB modality gaining less from RGB-Sonar feature fusion, while Sonar modality gains a large amount. Moreover, the imbalance of layers in CAFusion between RGB and Sonar modalities can slightly reduce model performance, primarily due to the learning imbalance between the RGB and Sonar modalities. The results show that RSFusionDet performs best when the number of layers in the RGB and Sonar modalities of CA-Fusion is balanced at 3. In CAFusion, RGB-Sonar features have highly unaligned features, and the number of heads in the cross-attention split feature vectors. We believe that the number of heads may affect the correlation between features, and thus affect the RGB-Sonar feature fusion. In Table 11, we reduce the number of heads for attention to increase the receptive field of the RGB-Sonar feature correlation calculation. The results show that when the number of heads is 4, RSFusionDet has the best performance and slightly reduces the computational complexity of attention.

Table 12. Ablation study for object matching strategies.
<table><tr><td>BBox priori</td><td>IoU weight</td><td> $\mathbf { P } _ { m a t c h }$ </td><td> ${ \bf R } _ { m a t c h }$ </td><td> $\mathbf { F 1 - S c o r e } _ { m a t c h }$ </td></tr><tr><td rowspan="3">√</td><td></td><td>8.6</td><td>8.1</td><td>8.4</td></tr><tr><td></td><td>17.2</td><td>16.7</td><td>16.9</td></tr><tr><td>√</td><td>77.2</td><td>83.8</td><td>80.4</td></tr><tr><td>√</td><td>√</td><td>80.9</td><td>86.0</td><td>83.4</td></tr></table>

Table 13. Ablation study for the number of layers in OMHead.
<table><tr><td>#layers</td><td>#Params (M)</td><td>GFLOPs</td><td>AP</td><td> $\mathbf { F 1 - S c o r e } _ { m a t c h }$ </td></tr><tr><td>1</td><td>98</td><td>251</td><td> $7 5 . 9 / 4 7 . 8$ </td><td>80.3</td></tr><tr><td>2</td><td>98</td><td>251</td><td> $7 6 . 1 / 4 7 . 9$ </td><td>80.7</td></tr><tr><td>3</td><td>98</td><td>251</td><td> $7 6 . 2 / 4 8 . 3$ </td><td>83.4</td></tr><tr><td>4</td><td>98</td><td>251</td><td> $7 6 . 2 / 4 8 . 2$ </td><td>79.7</td></tr><tr><td>5</td><td>98</td><td>251</td><td> $7 6 . 1 / 4 7 . 9$ </td><td>80.2</td></tr><tr><td>6</td><td>98</td><td>252</td><td> $7 5 . 8 / 4 7 . 9$ </td><td>79.9</td></tr></table>

Table 14. Ablation study for the filtering and matching threshold in OMHead. Vertical and horizontal headers represent the filtering and matching threshold, respectively. The metric is $\mathrm { F 1 - S c o r e } _ { m a t c h }$
<table><tr><td>filter\match</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td></tr><tr><td>0.1</td><td>80.0</td><td>80.3</td><td>80.5</td><td>80.7</td><td>81.0</td><td>81.0</td><td>81.2</td><td>81.5</td><td>81.5</td></tr><tr><td>0.2</td><td>81.6</td><td>82.1</td><td>82.3</td><td>82.5</td><td>82.7</td><td>82.7</td><td>82.9</td><td>82.9</td><td>82.8</td></tr><tr><td>0.3</td><td>82.2</td><td>82.7</td><td>82.6</td><td>83.0</td><td>83.2</td><td>83.2</td><td>83.3</td><td>83.4</td><td>83.2</td></tr><tr><td>0.4</td><td>82.2</td><td>82.6</td><td>82.7</td><td>82.9</td><td>83.0</td><td>83.1</td><td>83.3</td><td>83.3</td><td>83.1</td></tr><tr><td>0.5</td><td>82.2</td><td>82.5</td><td>82.6</td><td>82.8</td><td>83.0</td><td>83.1</td><td>83.3</td><td>83.2</td><td>83.0</td></tr><tr><td>0.6</td><td>81.1</td><td>81.4</td><td>81.7</td><td>81.9</td><td>81.9</td><td>82.0</td><td>82.2</td><td>82.3</td><td>82.3</td></tr><tr><td>0.7</td><td>80.1</td><td>80.4</td><td>80.8</td><td>81.0</td><td>81.0</td><td>80.9</td><td>81.1</td><td>81.3</td><td>81.3</td></tr><tr><td>0.8</td><td>78.1</td><td>78.4</td><td>78.8</td><td>79.0</td><td>79.0</td><td>79.1</td><td>79.2</td><td>79.5</td><td>79.4</td></tr><tr><td>0.9</td><td>59.7</td><td>60.0</td><td>60.3</td><td>60.4</td><td>60.4</td><td>60.5</td><td>60.5</td><td>60.6</td><td>60.6</td></tr></table>

## 5.3.3. Ablation Study on OMHead

To demonstrate the effectiveness of the RGB-Sonar object matching method, we perform ablation on object matching strategies in Table 12. The results indicate that adding bounding box priori information to the object feature slightly improves the matching performance of 8.5 $\mathrm { F 1 - S c o r e } _ { m a t c h }$ , and optimizing the object matching loss using IoU weight can significantly improve the matching performance of $7 2 ~ { \mathrm { F 1 - S c o r e } } _ { m a t c h }$ . Finally, with the opti mization of these two strategies, object matching achieves the best performance of $8 3 . 4 \mathrm { F 1 - S c o r e } _ { m a t c h }$ . The bounding box priori information provides object position information for RGB and Sonar modalities, which is beneficial for OM Head modelling RGB-Sonar object position mapping associations, thereby improving object matching performance. Furthermore, the IoU weight adaptively assigns learning weights to object matching features based on the richness of the object feature information, which reduces the impact of noise during the OMHead training process and improves the learning efficiency of the OMHead, thereby significantly improving its object matching performance. Meanwhile, we perform ablation on the encoding layers (MLP) of the OMHead object features in Table 13 and find tha the best object detection and object matching performance is achieved when the number of encoding layers is 3. The results also indicate that object matching can slightly affect the effect of RGB-Sonar feature fusion, while OMHead has a tiny number of parameters and computational complexity. Besides, we perform ablation on the filtering and matching threshold in OMHead, as shown in Table 14. The results illustrate that the object matching performance of OMHead is mainly affected by the filtering threshold. When the fil tering threshold is too low $( < 0 . 3 )$ , a large number of low confidence erroneous objects are introduced to OMHead. Conversely, when the filtering threshold is too high (>0.5), some low-confidence correct objects are filtered out, lead ing to a decrease in the object matching performance of OMHead. In contrast, the matching threshold has a relatively small impact on the object matching performance of OMHead. As the matching threshold increases from 0.1 to 0.8, the object matching performance of OMHead gradually improves. When the matching threshold reaches 0.9, due to its high requirements for matching scores, correct matching results with low matching scores are filtered out, thereby re ducing the object matching performance of OMHead. We find that when the filtering threshold is 0.3 and the matching threshold is 0.8, OMHead performs best with 80.9 P<sub>match</sub>, $8 6 . 0 \mathrm { R } _ { m a t c h } , 8 3 . 4 \mathrm { F } 1 . 5 \mathrm { c o r e } _ { m a t c h }$

## 6. Conclusion

In this paper, we put forward an underwater RGB-Sonar multimodal object detection dataset called RSFusion and its evaluation metrics, which is a new benchmark for RGB-Sonar multimodal object detection. Meanwhile, we propose a new RGB-Sonar multimodal object detection architecture called RSFusionDet with a new RGB-Sonar multimodal object detection result expression. The CAFusion and OM-Head with OMLoss we design effectively achieve RGB-Sonar spatial misalignment feature fusion (by deformable attention) and object matching. Our results have shown the effectiveness and better performance of RSFusionDet on RGB-Sonar multimodal object detection. The experimental results of our RSFusion benchmark also demonstrate that RGB-Sonar multimodal object detection has good adaptability and high performance improvement for objects in dark scenes, small and medium objects, and distant objects.

Although our RSFusionDet has achieved good performance and improvement on the RSFusion benchmark, further exploration and research are still needed for RGB-Sonar multimodal object detection. We have conducted pioneering research on RGB-Sonar multimodal object detection and provided an RSFusion benchmark. We look forward to further work in the future that can provide better ideas and methods on this topic.

Acknowledgments. This research is funded by the National Natural Science Foundation of China, grant number 52371350, and by the National Key Research and Development Program of China, grant number 2023YFC2809104.

## References

[1] Yue Cao, Junchi Bin, Jozsef Hamari, Erik Blasch, and Zheng Liu. Multimodal object detection by channel switching and spatial attention. In CVPR, pages 403–411, 2023. 3

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Endto-end object detection with transformers. In ECCV, pages 213–229, 2020. 3, 4, 5

[3] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 7

[4] Sri Aditya Deevi, Connor Lee, Lu Gan, Sushruth Nagesh, Gaurav Pandey, and Soon-Jo Chung. Rgb-x object detection via scene-specific fusion modules. In WACV, pages 7366– 7375, 2024. 3

[5] Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R Scott, and Weilin Huang. Tood: Task-aligned one-stage object de tection. In ICCV, pages 3490–3499, 2021. 3, 8

[6] Chongjian Ge, Junsong Chen, Enze Xie, Zhongdao Wang, Lanqing Hong, Huchuan Lu, Zhenguo Li, and Ping Luo. Metabev: Solving sensor failures for 3d detection and map segmentation. In ICCV, pages 8721–8731, 2023. 3, 7

[7] Z Ge. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021. 3, 8

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE TPAMI, 37(9):1904–1916, 2015. 3

[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 7

[10] Jisong Kim, Minjae Seong, Geonho Bang, Dongsuk Kum, and Jun Won Choi. Rcm-fusion: Radar-camera multi-level

fusion for 3d object detection. In ICRA, pages 18236–18242, 2024. 3

[11] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 4

[12] Dianze Li, Yonghong Tian, and Jianing Li. Sodformer: Streaming object detection with transformer using events and frames. IEEE TPAMI, 45(11):14020–14037, 2023. 3

[13] Gongyang Li, Yike Wang, Zhi Liu, Xinpeng Zhang, and Dan Zeng. Rgb-t semantic segmentation with location, activation, and sharpening. IEEE TCSVT, 33(3):1223–1235, 2022. 3

[14] Xiaotian Li, Baojie Fan, Jiandong Tian, and Huijie Fan. Gafusion: Adaptive fusing lidar and camera with multiple guidance for 3d object detection. In CVPR, pages 21209–21218, 2024. 2, 3, 7

[15] Yingwei Li, Adams Wei Yu, Tianjian Meng, Ben Caine, Jiquan Ngiam, Daiyi Peng, Junyang Shen, Yifeng Lu, Denny Zhou, Quoc V Le, et al. Deepfusion: Lidar-camera deep fusion for multi-modal 3d object detection. In CVPR, pages 17182–17191, 2022. 3

[16] Yingyan Li, Lue Fan, Yang Liu, Zehao Huang, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. Fully sparse fusion for 3d object detection. IEEE TPAMI, 2024. 3

[17] Yunfeng Li, Bo Wang, Jiuran Sun, Xueyi Wu, and Ye Li. Rgb-sonar tracking benchmark and spatial cross-attention transformer tracker. IEEE TCSVT, 2024. 2, 7, 8

[18] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755, 2014. 6, 1

[19] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ ICCV, pages 2980–2988, 2017. 3

[20] Zhiwei Lin, Zhe Liu, Zhongyu Xia, Xinhao Wang, Yongtao Wang, Shengxiang Qi, Yang Dong, Nan Dong, Le Zhang, and Ce Zhu. Rcbevdet: Radar-camera fusion in bird’s eye view for 3d object detection. In CVPR, pages 14928–14937, 2024. 3

[21] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. In ICLR, 2022. 3, 8

[22] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In ECCV, pages 21–37, 2016. 3

[23] Zhe Liu, Tengteng Huang, Bingling Li, Xiwu Chen, Xi Wang, and Xiang Bai. Epnet++: Cascade bi-directional fusion for multi-modal 3d object detection. IEEE TPAMI, 45 (7):8324–8341, 2022. 3

[24] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L Rus, and Song Han. Bevfusion: Multitask multi-sensor fusion with unified bird’s-eye view representation. In ICRA, pages 2774–2781, 2023. 2, 3, 7

[25] Zhuoyan Liu, Bo Wang, Ye Li, Jiaxian He, and Yunfeng Li. Unitmodule: A lightweight joint image enhancement module for underwater object detection. PR, 151:110435, 2024. 1

[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2018. 7

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn ing transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 4, 5

[28] Joseph Redmon and Ali Farhadi. Yolo9000: better, faster, stronger. In CVPR, pages 7263–7271, 2017. 3

[29] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018. 8

[30] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, pages 779–788, 2016. 1, 3

[31] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. NeurIPS, 28, 2015. 1, 3, 8

[32] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, pages 618–626, 2017. 9

[33] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In CVPR, pages 14454–14463, 2021. 3, 8

[34] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In ICCV, pages 9627–9636, 2019. 3, 8

[35] Abhishek Tomy, Anshul Paigwar, Khushdeep S Mann, Alessandro Renzaglia, and Christian Laugier. Fusing event based and rgb camera for robust object detection in adverse conditions. In ICRA, pages 933–939, 2022. 3

[36] Yike Wang, Gongyang Li, and Zhi Liu. Sgfnet: semantic guided fusion network for rgb-thermal semantic segmenta tion. IEEE TCSVT, 33(12):7737–7748, 2023. 3

[37] Junjie Yan, Yingfei Liu, Jianjian Sun, Fan Jia, Shuailin Li, Tiancai Wang, and Xiangyu Zhang. Cross modal transformer: Towards fast and robust 3d object detection. In ICCV, pages 18268–18278, 2023. 3, 7

[38] Ze Yang, Shaohui Liu, Han Hu, Liwei Wang, and Stephen Lin. Reppoints: Point set representation for object detection. In ICCV, pages 9657–9666, 2019. 3

[39] Heng Zhang, Elisa Fromont, Sebastien Lef ´ evre, and Bruno\` Avignon. Guided attentive feature fusion for multispectral pedestrian detection. In WACV, pages 72–80, 2021. 3, 8

[40] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In ICLR, 2022. 1, 3, 4, 6, 8, 10

[41] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In CVPR, pages 9759–9768, 2020. 3, 8

[42] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen.

Detrs beat yolos on real-time object detection. In CVPR, pages 16965–16974, 2024. 1

[43] Wujie Zhou, Hongping Wu, and Qiuping Jiang. Mdnet: Mamba-effective diffusion-distillation network for rgbthermal urban dense prediction. IEEE TCSVT, 2024. 3

[44] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. In ICLR, 2020. 3, 4, 8

# RSFusionDet: Underwater RGB-Sonar Multimodal Object Detection

# Supplementary Material

## 7. Details of RSFusion Dataset

We provide a detailed introduction and analysis of the RS-Fusion dataset, including visualization of all objects in RGB and Sonar modalities, processing of sonar raw images, analysis of object properties and distributions, and explanation of annotations.

## 7.1. Visualization of All Objects

We visualize all object classes with RGB and Sonar modalities in the RSFusion dataset, in Fig. 9. Objects in RGB modality have richer features than those in Sonar modality. It is obvious that there is a significant difference in the object features between the RGB and Sonar modalities, and the features have the spatial feature misalignment problem.

## 7.2. Processing of Sonar Raw Images

The sonar image includes raw and sector images in the RS-Fusion dataset, where the sector image is converted from the raw image. The brief conversion process is shown in Fig. 10. The sonar sensor returns a raw image and a list of angles when collecting sonar images. The angle is the horizontal angle of the beam in the forward-looking multibeam sonar (the horizontal field of view of the sonar is 130 degrees), and the echo intensity obtained by sending each beam forward at different vertical angles forms a column of pixel values in the height direction in the raw image. Expand each column of pixels according to their corresponding angles and use nearest neighbour interpolation to fill in missing angles to obtain the converted sonar sector image.

## 7.3. Analysis of Object Properties and Distributions

Object Scale Analysis. We analyze the object scale in our RSFusion dataset based on the object scale specifications of the COCO [18] dataset. As shown in Fig. 11, the object pixel area is defined as small objects if it is less than 32 \* 32, medium objects if it is greater than $3 2 ~ ^ { * } ~ 3 2$ and less than $9 6 \ast 9 6 .$ , and large objects if it is greater than $9 6 * 9 6$ In RGB modularity, objects are predominantly large, with metal polyhedron and float concentrated in medium objects, and the distribution of metal polyhedron is more uniform. In the sonar modality, objects are predominantly small and medium.

Object Aspect Ratio Analysis. In Fig. 12, the object aspect ratios in the RGB and Sonar modalities have a generally consistent distribution, with aspect ratios mainly distributed around 1. Due to the characteristics of the object shape, the aspect ratios of the mannequin are evenly distributed in the range of 0.5-3.0. In contrast, the UUVs are mainly distributed in the range of smaller aspect ratios.

![](images/8c864c00850c4320cac7337071fd9f2a901d97e697eaa8c7f2826ae704f8337c.jpg)  
(g) frustum.  
Figure 9. All object classes in the RSFusion dataset. The object on the left is in RGB, while the one on the right is in Sonar.

Number of Objects per Image. As shown in Fig. 13, most RGB images have 0, 1, or 2 objects, while Sonar images have a greater number of objects with an average of 2.5 objects per image, which is 1 more than the average number of objects in RGB images. At the same time, the maximum number of objects per image for both is 7. There are samples in RSFusion where one modality image has no object while the other modality image has objects, but there are no samples where both modalities have no object. Besides, the average number of objects in Sonar images is greater than in RGB images, indicating that some objects visible in Sonar images are invisible in RGB images. It is because the field of view (FOV) of the optical camera is smaller than that of the forward-looking sonar, and sometimes the visible distance of the sonar is greater than that of the optical camera during data collection.

Distribution of the Number of Objects. In Tables 15, 16, 17, and 18, we analyze the distribution of the object quantity under different conditions. There are 9298 and 17179 objects in the RGB and Sonar images respectively, and the number of objects in the light scene is about three times that of the dark scene. At the same time, the number of objects in sr15 is the greatest, far greater than in sr5, sr10, and sr20. The srx represents that the maximum sonar range is x meters. There are more small objects in sr5 and sr15, and the number of objects is also greater than in sr10 and sr20, including objects in dark scenes.

![](images/47823c6cc50b1ac6d325e60c1b8fd17f3018666daf98322fb24a3758257de76e.jpg)  
Figure 10. Conversion of sonar images from raw to sector.

## 7.4. Explanation of Annotations

Here we present the annotation of our RSFusion dataset. We divide RSFusion equally into training and validation sets in an 8:2 ratio and calculate the mean and std of all the RGB and Sonar images in the dataset, which are mean (132.779, 145.418, 126.863), std (21.792, 20.260, 25.153) for RGB images whose channel order is BGR, and mean 11.451, std 11.565 for Sonar grayscale images. The annotation contains instances train images.json, instances train sonars.json, instances val images.json, and instances val sonars.json. Each annotation has the same structure, which contains two parts: metainfo and data list. metainfo includes all kinds of classes, environments, and fusion flags information. data list contains all the image and object information, as well as the flag information fusion flag that indicates whether there is a corresponding identical object in RGB or Sonar modality. In addition, the sonar annotation also includes sonar info which contains several sonar metadata. Its details are in our code and dataset.

![](images/0d347de1931fa1d022971d72f16922023570c8c730f17e5d20a7f82873e0b6a8.jpg)

(a) Object scales in RGB modality.  
![](images/208c4b690ef923c4d0002f6a5d3656594e03838aa77615a982251205ab4ab3b2.jpg)  
(b) Object scales in Sonar modality.  
Figure 11. Distribution of object scales in the RSFusion dataset.

Table 15. Distribution of the number of objects in maximum sonar ranges with object scales.
<table><tr><td colspan="10">sr5</td><td colspan="10"></td><td colspan="5"></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2">sr10 1366</td><td colspan="2"></td><td colspan="2">sr15 4822</td><td colspan="2">sr20 1146</td><td colspan="2"> $_ { \mathrm { s r } 5 }$  2684</td><td colspan="2"></td><td colspan="2">sr10 1900</td><td colspan="2"></td><td colspan="2">sr15 10552</td><td colspan="2">sr20 2043</td><td colspan="2"></td></tr><tr><td>small</td><td>medium</td><td>large 771</td><td>small 0</td><td>medium 806</td><td>large</td><td>small 27</td><td>medium 2345</td><td>large</td><td>small</td><td>medium</td><td></td><td>large</td><td>small</td><td>medium</td><td>large</td><td>small 1029</td><td>medium</td><td>large</td><td>small</td><td>medium</td><td>large</td><td>small</td><td></td><td>medium</td><td>large</td></tr><tr><td>170</td><td>1023</td><td></td><td></td><td></td><td>560</td><td></td><td></td><td>2450</td><td>0</td><td>822</td><td>324</td><td>302</td><td>1455</td><td>927</td><td></td><td></td><td>870</td><td>1</td><td>7944</td><td>2608</td><td>0</td><td>1479</td><td>564</td><td></td><td>0</td></tr></table>

![](images/0d32b784231eb53bd8bcbdeb34f254ca859e2d8eed4a365e58bddfe86eb7d374.jpg)

(a) Object aspect ratios in RGB modality.  
![](images/63388f542a1293fb2480d6a95fd7648a1ee1a0df6d678bb4111c30ffce8effe3.jpg)  
(b) Object aspect ratios in Sonar modality.  
Figure 12. Distribution of object aspect ratios in the RSFusion dataset. The aspect ratio is calculated from h/w of the object.  
Table 16. Distribution of the number of objects in maximum sonar ranges with scenes.

![](images/6935bcff989f84be47a9f3f4f597ba274687a867da08da06d7acad0e76cc4190.jpg)

<table><tr><td colspan="10">RGB</td><td colspan="7">Sonar</td></tr><tr><td colspan="2">sr5 1964</td><td colspan="2">sr10 1366</td><td colspan="2">sr15 4822</td><td colspan="2">sr20 1146</td><td colspan="2">sr5 2684</td><td colspan="2">sr10 1900</td><td colspan="2">sr15 10552</td><td colspan="2">sr20 2043</td></tr><tr><td>light 1964</td><td>dark 0</td><td>light 1366</td><td>dark 0</td><td>light 3004</td><td>dark 1818</td><td>light 861</td><td>dark 285</td><td>light 2684</td><td>dark 0</td><td>light 1900</td><td>dark 0</td><td>light 6837</td><td>dark 3715</td><td>light 1465</td><td>dark 578</td></tr></table>

Figure 13. Distribution of the number of objects per image in the RSFusion dataset.  
Table 17. Distribution of the number of objects in scenes with object scales.
<table><tr><td colspan="6">RGB</td><td colspan="6">Sonar</td></tr><tr><td colspan="3">light 7195</td><td colspan="3">dark 2103</td><td colspan="3">light 12886</td><td colspan="3">dark 4293</td></tr><tr><td rowspan="2">small 197</td><td>medium</td><td>large</td><td>small</td><td>medium</td><td>large</td><td>small</td><td>medium</td><td>large</td><td>small</td><td>medium</td><td>large</td></tr><tr><td>3910</td><td>3088</td><td>0</td><td>1086</td><td>1017</td><td>7763</td><td>4195</td><td>928</td><td>2991</td><td>1302</td><td>0</td></tr></table>

Table 18. Distribution of the number of objects in scenes with maximum sonar ranges.
<table><tr><td colspan="8">RGB 一</td><td colspan="8">Sonar</td></tr><tr><td colspan="3">light 7195</td><td colspan="2"></td><td colspan="3">dark</td><td colspan="3">light</td><td colspan="4">dark 4293</td></tr><tr><td colspan="3">sr5</td><td>sr20</td><td>sr5 sr10</td><td colspan="3">2103 sr15</td><td colspan="3">12886</td><td colspan="4"></td></tr><tr><td>1964</td><td>sr10 1366</td><td>sr15 3004</td><td>861</td><td>0</td><td>0</td><td>1818</td><td>sr20 285</td><td>sr5 2684</td><td>sr10 1900</td><td>sr15 6837</td><td>sr20 1465</td><td>sr5 sr10 0 0</td><td>sr15 3715</td><td>sr20 578</td></tr></table>