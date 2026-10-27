# MMGNet: Multi-Modal 6-DoF Grasp Synthesis for Cluttered Scenes

Rui Cao<sup>a</sup>, Jia-Hui Pan<sup>b</sup>, Yichuan Li<sup>a</sup>, Yunfan Yu<sup>a</sup>, Chi-Wing Fu<sup>b</sup>, Jiangliu Wang<sup>a,c</sup>, Yun-Hui Liu<sup>a,∗</sup>

<sup>a</sup>Department ofMechanical and Automation Engineering, The Chinese University ofHong Kong, Hong Kong, China

<sup>b</sup>Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China

<sup>c</sup>Sino-German College ofIntelligent Manufacturing, Shenzhen Technology University, Shenzhen, China

## Abstract

Grasping is a foundational capability in robotic manipulation. Scene-level 6-DoF grasp synthesis relies on point clouds for metric geometry, yet real-world point observations are often noisy, incomplete, and over-smoothed. RGB images provide object-region, boundary, and appearance cues, yet two bottlenecks limit their efective use in scene representation and local grasp reasoning. We present MMGNet, a Multi-Modal Grasp Synthesis Network that addresses these bottlenecks through hierarchical image-to-3D feature injection and rotational graspness modeling. Hierarchical injection integrates multi-scale image features throughout 3D representation formation, while the rotational graspness jointly predicts the grasp approach direction and in-plane rotation before organizing local image-informed features through grasp-shaded grouping. Ablation studies show that these two modules provide complementary gains. MMGNet achieves a mean AP of 60.34 on GraspNet-1Billion with RealSense, surpassing all compared methods. In real-world experiments with adversarial grasp objects, MMGNet improves the grasp success rate over AnyGrasp by 8.46 percentage points. Code, models, and data will be released upon publication.

Keywords: Robotic perception, 6-DoF grasp synthesis, Grasp representation, Deep learning and computer vision.

<sup>∗</sup>Corresponding author Email addresses: rcao@mae.cuhk.edu.hk (Rui Cao), yhliu@mae.cuhk.edu.hk (Yun-Hui Liu)

## 1. Introduction

Robotic grasping supports downstream tasks such as bin picking and packing [1, 2], object rearrangement [3], and tool manipulation [4]. Scene-level 6-DoF grasp synthesis predicts executable gripper poses for objects with diverse shapes, orientations, and occlusions. Most existing scene-level methods rely primarily on point clouds to represent the metric geometry required for 6-DoF grasp prediction [5–8]. However, real-world point clouds often sufer from measurement noise, local incompleteness, and over-smoothing, while RGB images provide complementary cues about object regions, boundaries, and appearance. Accordingly, several grasp synthesis methods have incorporated image information to improve grasp prediction [9–11].

Nevertheless, two bottlenecks limit the efective use of image information in scene representation and local grasp reasoning. First, some existing methods incorporate image cues through late feature fusion [10] or during local-region processing [11]. Consequently, multi-scale image context plays a limited role during 3D scene representation formation. We therefore argue that such image information should be integrated throughout this process. Second, even when image cues are embedded in the scene representation, the local representation used for detailed grasp prediction remains too coarse to capture grasp-quality diferences across gripper rotations. Existing pipelines [6–8] first use grasp-quality scores to select promising grasp centers and approach directions, aggregate nearby 3D features, and only then predict the gripper rotation around the selected direction. Consequently, grasps that share the same center and approach direction but difer in gripper rotation are evaluated from the same local representation. Thus, the embedded image cues are not organized according to the candidate rotation, limiting their ability to capture the corresponding grasp-quality diferences.

We present the Multi-Modal Grasp Synthesis Network (MMGNet), a single-view scene-level framework designed to address these two bottlenecks. To address the first, MMGNet introduces hierarchical image-to-3D feature injection, which aligns multiscale image features with scene points and injects them into successive stages of a sparse 3D backbone, allowing image context to shape the 3D representation throughout the backbone. To address the second, MMGNet introduces rotational graspness modeling to jointly predict the grasp approach direction and in-plane rotation before local aggregation. The selected rotation then guides grasp-shaded grouping, which gathers local image-informed 3D features in the corresponding gripper frame. Together, these designs make grasp-quality diferences across rotations explicit before local analysis and organize the resulting local features according to the selected gripper orientation.

Given an RGB-D observation, MMGNet first extracts a multi-scale image feature pyramid and establishes pixel–point correspondence with the full-scene point cloud. The aligned image features are injected into multiple sparse-encoder stages to form an RGB-informed 3D scene representation while retaining eficient metric neighborhood aggregation. A foreground score then guides the sampling of spatially distributed grasp seeds. For each seed, rotational graspness selects a joint view–in-plane rotation, after which grasp-shaded grouping aggregates local features in the corresponding gripper frame. Finally, the operation head predicts approach depth and width of the gripper to produce the 6-DoF grasps.

In GraspNet-1Billion [5] dataset, MMGNet achieves a state-of-the-art mean Average Precision (AP) of 60.34 in RealSense scenes and a competitive mean AP of 50.72 in Kinect scenes. Ablation studies show that hierarchical image-to-3D feature injection improves the point-only Rotational model by 5.39 AP points, while the rotational graspness model outperforms the View model by 8.39 points under the same hierarchical injection. The two modules exhibit a positive interaction of 2.19 AP points, confirming their complementarity. Cross-modal representation analysis further reveals stronger image–3D correspondence across backbone stages than early and late fusion. Point- and RGB-corruption experiments show that image-informed 3D features complement noisy or incomplete point observations and characterize how diferent fusion strategies respond to unreliable RGB. In real-robot experiments, MMGNet improves the grasp success rate over AnyGrasp [6] by 8.46 percentage points on adversarial grasp objects [12]. Our main contributions are:

• We develop MMGNet, a scene-level multi-modal framework for single-view 6-DoF grasp synthesis.

• We introduce hierarchical image-to-3D feature injection, which integrates multiscale image cues throughout 3D representation formation.

• We introduce a rotational graspness model that scores joint view–in-plane rotations and uses the selected rotation to organize image-informed local features through grasp-shaded grouping.

• MMGNet achieves state-of-the-art performance on GraspNet-1Billion and demonstrates its applicability in real-world robotic grasping.

• Extensive ablations and analyses validate the proposed modules across fusion strategies, grasp models, representation properties, and point- and RGB-corrupted observations.

## 2. Related Work

## 2.1. 6-DoF Grasp Synthesis for Cluttered Scenes

Grasp synthesis predicts executable gripper poses from visual observations. Early image-based methods mainly considered 4-DoF top-down grasps, represented as planar rectangles in RGB or RGB-D images [13–15]. Although eficient, this formulation restricts the gripper approach direction. Six-DoF grasp synthesis removes this restriction and enables more flexible grasping in cluttered scenes.

Point-cloud-based methods. Most scene-level 6-DoF methods operate on depthderived point clouds. Zeng et al. [16] predict dense afordances for robotic manipulation, while Fang et al. [5] lift dense grasp prediction into 3D scene points. GSNet [8] introduces graspness to identify promising regions before detailed grasp prediction, reducing the scene-level search space. Subsequent methods improve candidate selection through adaptive instance sampling [7], graspable-region prediction [17], or interest-point selection [18]. AnyGrasp [6] further incorporates object center-of-mass information to improve grasp stability. These methods progressively improve candidate quality and computational eficiency, but remain primarily dependent on partial and potentially noisy point observations.

RGB-D and richer observations. Several methods incorporate image information or additional geometric context into 6-DoF grasp synthesis. RGB Matters [9] processes RGB and depth separately for gripper orientation prediction and analytic grasp search. Liu et al. [10] combine independently encoded image and point features through late fusion, while Qiu et al. [19] concatenate range-image and point-cloud features to recover boundary information missing from depth observations. CenterGrasp [20] combines object-aware shape reconstruction with grasp estimation, and ZeroGrasp [21] jointly reconstructs complete 3D geometry and predicts grasps from a single RGB-D observation using occlusion and inter-object reasoning. FlexLoG [11] combines flexible guidance with a local grasp model to support both scene-level and target-oriented grasping. Zhu et al. [22] improve robustness to occlusion by post-fusing an auxiliary RGB-D view and aggregating cross-view geometry around grasp candidates.

These works exploit richer visual information through late feature fusion, shape reconstruction, local-region modeling, or cross-view aggregation. MMGNet follows the scene-level point-cloud paradigm and integrates multi-scale image information into the 3D scene representation. The resulting image-informed representation is then coupled with rotational graspness and grasp-shaded grouping for local grasp prediction across diferent gripper rotations.

## 2.2. Image-to-3D Fusion for 3D Perception

Image-to-3D fusion has been explored at input, intermediate, and output levels in 3D object detection [23, 24], segmentation [25], and scene understanding [26, 27]. Multi-level 2D–3D interaction has also been studied for scene understanding and 6D object pose estimation [28, 29]. These studies demonstrate the benefit of integrating image information at multiple levels of 3D feature learning. Within this line of work, MMGNet introduces hierarchical image-to-3D injection into a sparse scenelevel grasp backbone. Multi-scale image features are aligned with scene points and injected into successive backbone stages, allowing image information to participate throughout 3D scene representation formation.

![](images/f7ad7037f0decbdea8b4b824a74c9016972082753c8708d79c1706471853c791.jpg)  
Figure 1: Illustration of MMGNet. Given a single-view RGB-D observation, the image backbone extracts dense image features, which are aligned with the scene points through pixel–point correspondence and injected into successive stages of the 3D backbone to form image-informed 3D features. Rotational graspness prediction then scores joint view–in-plane rotations at candidate grasp centers. Based on the selected rotations, grasp-shaded grouping aggregates local image-informed features within the corresponding gripper frames. Finally, the operation head predicts the remaining depth-wise grasp scores and gripper widths to produce the final 6-DoF grasps.

## 3. Method

## 3.1. Problem Formulation and Pipeline Overview

Grasp synthesis predicts a set of executable grasps for a robot arm to pick up objects in a cluttered scene. Given a single-view RGB-D observation, a grasp synthesis method predicts a set of 6-DoF grasp poses $G = \left\{ \mathbf { g } _ { r } \right\} _ { r = 1 } ^ { N _ { g } }$ . A grasp pose is defined as $\mathbf { g } = ( \mathbf { R } , \mathbf { t } , \omega )$ , where $\mathbf { R } \in \mathbb { R } ^ { 3 \times 3 } , \mathbf { t } \in \mathbb { R } ^ { 3 }$ , and $\omega \in \mathbb { R } _ { + }$ denote the gripper rotation, translation, and width, respectively.

We present MMGNet, a scene-level 6-DoF grasp synthesis network built around hierarchical image-to-3D feature injection and a rotational graspness model. MMGNet extracts dense multi-scale image features, establishes pixel–point correspondence, and injects the aligned image features at multiple encoder stages of a sparse 3D backbone. This allows visual cues to participate throughout 3D representation formation rather than serving only as input initialization or late augmentation. From the resulting image-informed 3D scene representation, a point-wise objectness head identifies foreground object regions, and Farthest Point Sampling (FPS) selects spatially distributed seed points for grasp prediction. MMGNet then applies the rotational graspness model, which resolves the joint view–in-plane rotation for each seed and aggregates local features within the corresponding grasp-shaded region to predict depthwise grasp scores and gripper widths.

![](images/c6dabf32f5d30dc1ee92bb941e7816def2f213277dc7b95f8333a2b2125aad7e.jpg)  
Figure 2: (a) Grasp representation in MMGNet, showing the grasp coordinate frame with the red point marking the grasp origin. (b) Grasp-shaded grouping, where the black cuboid denotes the grouping region and the red points denote the grouped scene points.

## 3.2. Hierarchical Image-to-3D Feature Injection

To form an image-informed 3D scene representation, MMGNet aligns multi-scale image features with scene points and injects them at successive stages of a sparse 3D backbone. This hierarchical injection allows RGB information to participate throughout 3D representation formation. An objectness head then identifies foreground regions, from which FPS selects spatially distributed seeds for grasp prediction.

Multi-Scale Image Feature Extraction. Given the full RGB image I, we use a convolutional neural network (CNN) Ψ to extract a multi-scale dense image feature pyramid:

$$
\{ \mathbf { F } _ { \mathrm { i m g } } ^ { ( 1 ) } , \mathbf { F } _ { \mathrm { i m g } } ^ { ( 2 ) } , \mathbf { F } _ { \mathrm { i m g } } ^ { ( 4 ) } , \mathbf { F } _ { \mathrm { i m g } } ^ { ( 8 ) } , \mathbf { F } _ { \mathrm { i m g } } ^ { ( 1 6 ) } \} = \Psi ( \mathbf { I } ) ,\tag{1}
$$

where the superscript denotes the feature stride relative to the input image. We use ResNet [30] and PSPNet [31] to encode dense visual evidence at multiple spatial scales. The resulting feature pyramid provides visual features for the hierarchical image-to-3D injection described next.

Hierarchical Image-to-3D Feature Injection. To transfer image evidence into the 3D scene representation, we first establish the correspondence between each scene point and its image pixel. Given a scene point $\mathbf { p } _ { i } \ = \ ( x _ { i } , y _ { i } , z _ { i } ) ^ { \top } \ \in \ \mathcal { P }$ in the camera coordinate system, its corresponding pixel $( u _ { i } , \nu _ { i } )$ is obtained using the pinhole camera model with camera intrinsic matrix $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$

$$
\left[ u _ { i } , \nu _ { i } , 1 \right] ^ { \top } = \frac { 1 } { z _ { i } } \mathbf { K } \mathbf { p } _ { i } .\tag{2}
$$

For each feature stride $s \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ , we gather the per-point image feature from the corresponding feature map:

$$
\mathbf { f } _ { i } ^ { ( s ) } = \mathbf { F } _ { \mathrm { i m g } } ^ { ( s ) } \left( \left\lfloor \frac { u _ { i } } { s } \right\rfloor , \left\lfloor \frac { \nu _ { i } } { s } \right\rfloor \right) .\tag{3}
$$

Rather than injecting visual cues only before sparse 3D encoding or after the 3D hierarchy has formed, MMGNet injects them throughout the hierarchical sparse encoder. At encoder stage s, let $\mathbf { x } _ { \ell } ^ { ( s ) }$ denote the feature of the ℓ-th active voxel, and let $\boldsymbol { Q } _ { \ell } ^ { ( s ) }$ be the set of scene points assigned to that voxel. We aggregate the aligned point-wise image features onto active sparse voxels by mean scattering:

$$
\tilde { \mathbf { f } } _ { \ell } ^ { ( s ) } = \frac { 1 } { | Q _ { \ell } ^ { ( s ) } | } \sum _ { i \in Q _ { \ell } ^ { ( s ) } } \mathbf { f } _ { i } ^ { ( s ) } .\tag{4}
$$

We concatenate the aggregated image feature with the sparse 3D feature and project the result back to the original channel dimension using a $1 \times 1$ sparse convolution:

$$
\bar { \mathbf { x } } _ { \ell } ^ { ( s ) } = \Omega ^ { ( s ) } ( [ \mathbf { x } _ { \ell } ^ { ( s ) } , \tilde { \mathbf { f } } _ { \ell } ^ { ( s ) } ] ) ,\tag{5}
$$

where $\Omega ^ { ( s ) } ( \cdot )$ denotes the stage-specific sparse projection layer. We apply this injection at encoder stages s, allowing dense visual cues to shape the 3D representation at successive spatial scales while preserving the active sparse voxel structure. This hierarchical injection defines the image-to-3D interface of MMGNet: it enriches the learned 3D features for grasp reasoning without modifying the point coordinates.

After sparse encoder–decoder processing, the image-informed voxel features are mapped back to the original scene points, yielding image-informed scene features $\mathbf { F } _ { \mathrm { s c e n e } } \in \mathbb { R } ^ { N _ { f } \times C }$ for objectness prediction and grasp synthesis. Given its strong performance across 3D tasks [32, 33], we employ MinkowskiEngine [34] to implement the sparse 3D backbone.

Objectness-guided Seed Selection. From the image-informed scene features $\mathbf { F } _ { \mathrm { s c e n e } } ,$ a three-layer MLP predicts a point-wise objectness score $o _ { i }$ for each full-scene point $\mathbf { p } _ { i } .$ Here, objectness denotes a class- and instance-agnostic foreground probability indicating whether a scene point lies on a visible object surface within the workspace. We then apply FPS to the predicted object points to select a compact set of N seed points for grasp prediction. This step suppresses background clutter and maintains spatial coverage over the predicted object regions, allowing the following grasp prediction modules to focus on grasp-relevant scene points.

## 3.3. Rotational Graspness Model

Following seed selection, the rotational graspness model predicts grasps through two sequential stages. Rotational graspness prediction first selects a joint view–inplane rotation for each seed, after which grasp-shaded grouping aggregates imageinformed local feature in the corresponding gripper frame. The operation head then predicts the remaining depth-wise grasp parameters.

Rotational Graspness Prediction. Previous graspness-based methods predict view graspness in the first stage and leave in-plane rotation and depth to subsequent grasp prediction. Let $\hat { q } _ { n , \nu , a , d }$ denote the analytic grasp quality for seed $n ,$ view v, in-plane rotation $^ { a , }$ and depth d. With A in-plane rotations and D depth anchors, view graspness summarizes grasp quality over both dimensions:

$$
\hat { s } _ { n , \nu } ^ { \mathrm { v i e w } } = \frac { 1 } { A D } \sum _ { a = 1 } ^ { A } \sum _ { d = 1 } ^ { D } \mathbb { 1 } ( \hat { q } _ { n , \nu , a , d } > c ) ,\tag{6}
$$

where $c$ is the grasp-quality threshold. Thus, diferent in-plane rotations under the same view share one graspness score, although subsequent grasp prediction must distinguish them. Moreover, the in-plane rotation changes the gripper placement and therefore the local contact and collision geometry.

To retain this distinction before local aggregation, we define rotational graspness for each joint view–in-plane rotation:

$$
\hat { s } _ { n , \nu , a } ^ { \mathrm { r o t } } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \mathbb { 1 } ( \hat { q } _ { n , \nu , a , d } > c ) .\tag{7}
$$

Unlike view graspness, rotational graspness averages only over depth and assigns a separate graspness target to each $( \nu , a )$ pair. Assigning a separate target to each joint view–in-plane rotation encourages the pre-grouping image-informed seed features to retain grasp-quality diferences among candidate rotations. We normalize the groundtruth rotational graspness targets across seed points in each scene to the range [0<sub>,</sub> 1].

For implementation, we flatten the $V { \times } A$ view–in-plane rotation pairs into $M = V A$ rotation anchors and denote the predicted scores as $\mathbf { S } _ { \mathrm { r o t } } = \{ s _ { n , m } \} \in \mathbb { R } ^ { N \times M }$ . An MLP predicts $\mathbf { S } _ { \mathrm { r o t } }$ from the image-informed seed features. For each seed, we select the rotation anchor with the highest predicted graspness score:

$$
( \nu _ { n } ^ { * } , a _ { n } ^ { * } ) = \arg \operatorname* { m a x } _ { \nu , a } s _ { n , \nu , a } ^ { \mathrm { r o t } } .\tag{8}
$$

The selected pair determines the gripper rotation ${ \bf R } _ { n } = { \bf R } ( \nu _ { n } ^ { * } , a _ { n } ^ { * } )$ , which is used by grasp-shaded grouping for subsequent local analysis. The operation head then predicts depth-wise grasp scores and gripper widths for this selected rotation.

Grasp-shaded Grouping. As the second stage of the rotational graspness model, grasp-shaded grouping constructs a gripper-aligned local support region after the joint view–in-plane rotation is selected. For the n-th seed point ${ \bf p } _ { n }$ with selected rotation $\mathbf { R } _ { n } ,$ each scene point $\mathbf { p } _ { i }$ is transformed into the corresponding gripper frame:

$$
\begin{array} { r } { \pmb { \xi } _ { n , i } = \mathbf { R } _ { n } ^ { \top } ( \mathbf { p } _ { i } - \mathbf { p } _ { n } ) . } \end{array}\tag{9}
$$

We define the gripper-aligned local support region as a cuboid in this gripper frame:

$$
N _ { n } = \left\{ i \left| \left| \xi _ { n , i } ^ { x } \right| \right. \leq \frac { W _ { \operatorname* { m a x } } } { 2 } , \left. \left| \xi _ { n , i } ^ { y } \right| \leq \frac { T _ { f } } { 2 } , 0 \leq \xi _ { n , i } ^ { z } \leq z _ { \operatorname* { m a x } } \right\} , \right.\tag{10}
$$

where $W _ { \mathrm { m a x } }$ is the maximum gripper width, $T _ { f }$ is the fingertip thickness, and $z _ { \mathrm { m a x } }$ is the maximum grouping extent along the depth direction. Fig. 2(b) illustrates this grouping region.

Unlike cylinder grouping [6–8], whose support region is mainly determined by the grasp center and approach direction, grasp-shaded grouping is conditioned on the full selected rotation, including the in-plane angle. Diferent in-plane rotations therefore induce diferent cuboid support regions, enabling the model to aggregate candidatespecific image-informed features.

For each selected seed, we sample K neighboring points from $N _ { n }$ . The grouped point features $\mathbf { F } _ { \mathrm { g r o u p } } ^ { \prime } \in \mathbb { R } ^ { N \times K \times C }$ are aggregated by max pooling over the K neighboring features, yielding $\mathbf { F } _ { \mathrm { g r o u p } } \in \mathbb { R } ^ { N \times C }$ . From each grouped region, a three-layer operation head predicts grasp scores and corresponding widths for the D depth anchors, yielding an output of size $N \times 2 \times D$

## 3.4. Optimization

We optimize both backbones and prediction heads using four objectives: pointwise objectness loss $( L _ { o } )$ , rotational graspness loss $( L _ { r } )$ , grasp score loss $( L _ { s } )$ , and grasp width loss $\left( L _ { \omega } \right)$ . The overall loss is

$$
L = \lambda _ { o } L _ { o } + \lambda _ { r } L _ { r } + \lambda _ { s } L _ { s } + \lambda _ { \omega } L _ { \omega } ,\tag{11}
$$

where $\lambda _ { o } , \lambda _ { r } , \lambda _ { s } ,$ and $\lambda _ { \omega }$ denote the corresponding loss weights. The individual losses are:

$$
\boldsymbol { L _ { o } } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \boldsymbol { L } _ { \mathrm { C E } } ( \boldsymbol { o _ { i } } , \hat { \boldsymbol { o } } _ { i } ) ,\tag{12}
$$

$$
L _ { r } = \frac { 1 } { N M } \sum _ { n = 1 } ^ { N } \sum _ { m = 1 } ^ { M } L _ { \mathrm { r e g } } ( s _ { n , m } , \hat { s } _ { n , m } ) ,\tag{13}
$$

$$
L _ { s } = \frac { 1 } { N D } \sum _ { n = 1 } ^ { N } \sum _ { d = 1 } ^ { D } L _ { \mathrm { r e g } } ( q _ { n , d } , \hat { q } _ { n , d } ) ,\tag{14}
$$

$$
L _ { \omega } = \frac { 1 } { N D } \sum _ { n = 1 } ^ { N } \sum _ { d = 1 } ^ { D } L _ { \mathrm { r e g } } ( \omega _ { n , d } , \hat { \omega } _ { n , d } ) \mathbb { 1 } ( \hat { q } _ { n , d } > 0 ) ,\tag{15}
$$

where $N _ { f } , N , M ,$ , and D denote the numbers of full-scene points, selected seed points, joint view–in-plane rotation anchors, and depth anchors, respectively. Here, $o _ { i }$ denotes the predicted objectness score, $s _ { n , m }$ the predicted rotational graspness score, and $q _ { n , d }$ and $\omega _ { n , d }$ the predicted grasp quality and gripper width for seed n at depth anchor d. The notation ˆ· denotes the corresponding ground-truth value. $L _ { \mathrm { C E } } ( \cdot )$ denotes crossentropy loss, and $L _ { \mathrm { r e g } } ( \cdot )$ denotes smooth $- L _ { 1 }$ loss. The width loss is applied only to grasp poses with positive ground-truth grasp scores.

## 4. Experiments

We evaluate MMGNet from five complementary perspectives: benchmark accuracy, the coupled efects of hierarchical image-to-3D feature injection and the rotational graspness model, cross-modal representation formation, robustness to point and RGB corruptions, and real-robot grasping. We first report comparisons with stateof-the-art methods, then conduct Ablation studies and transfer analysis, and finally examine robustness, deployment cost, and real-world performance.

## 4.1. Implementation Details

Benchmark. We evaluate our method using the GraspNet-1Billion dataset [5], which includes 100 training and 90 testing scenes, each captured from 256 views with two commercial-grade cameras, RealSense and Kinect. The testing scenes are split into three categories: seen, similar, and novel, depending on whether the objects are included in the training data.

Implementation. Our method is implemented using PyTorch. For scene-level grasp synthesis, we randomly sample $N _ { f } = 2 0 { , } 0 0 0$ points from the full scene point cloud as input and select $N = 1 { , } 0 2 4$ seed points for grasp prediction. Our image backbone is implemented as a ResNet34 encoder [30], followed by a PSPNet [31] with four pooling layers of kernel sizes $( 1 \times 1 , 2 \times 2 , 3 \times 3 , 6 \times 6 )$ . The image backbone produces a multi-scale image feature pyramid at strides {1 2 4 8 16}, and the channel dimension of the image features is $C _ { \mathrm { i m g } } = 6 4$ . MinkUNet14D [34] is used as our 3D backbone, producing image-informed 3D scene features with channel dimension $C = 2 5 6$ . We implement hierarchical image-to-3D feature injection at encoder stages with strides {1<sub>,</sub> 2<sub>,</sub> 4<sub>,</sub> 8<sub>,</sub> 16}, matching the multi-scale image pyramid to the corresponding sparse 3D hierarchy.

The rotational graspness prediction stage employs an MLP with channel sizes $( 2 5 6 , 2 5 6 , M )$ , where $M = 3 0 0 \times 1 2$ denotes the number of joint view–in-plane rotation anchors, consisting of 300 views and 12 in-plane rotations. We set the quality threshold to $c = 0 . 5 .$ The grasp-shaded grouping stage samples $K = 3 2$ neighboring points from each gripper-aligned cuboid region. The cuboid dimensions are set to

$W _ { \mathrm { m a x } } ~ = ~ 0 . 1 0 \mathrm { m } , ~ T _ { f } ~ = ~ 0 . 0 2 \mathrm { m }$ , and $z _ { \mathrm { m a x } } ~ = ~ 0 . 0 5 \mathrm { m }$ , corresponding to the maximum gripper width, fingertip thickness, and maximum grouping extent along the depth direction, respectively. For the operation head, we use an MLP with channel sizes $( 2 5 6 , 2 5 6 , 2 \times D )$ to predict grasp scores and gripper widths for depth anchors, where $D = 4$ is the number of depth anchors. In loss design, we set $\lambda _ { o } = 1 , \lambda _ { r } = 5 , \lambda _ { s } = 5$ and $\lambda _ { \omega } = 1$

Training. The network is trained on a single NVIDIA RTX3090 GPU using the AdamW optimizer [35] with a batch size of 6. A cosine annealing learning rate is used, starting at $1 . 5 \times 1 0 ^ { - 3 }$ and gradually decreasing to $1 0 ^ { - 5 }$ . We train separate MMGNet models for the two cameras for 20 epochs.

Evaluation. We evaluate grasp accuracy using Average Precision (AP) following the GraspNet-1Billion protocol [5, 8]. We compute $\mathbf { A P } _ { t h }$ with the friction threshold th varying from 0.2 to 1.2. Non-Maximum Suppression (NMS) is applied before retrieving the Top-50 highest-scoring grasps for evaluation, following [5]. For further details on the evaluation process, please refer to [5].

## 4.2. Comparison with State-of-the-Art Methods

We strictly follow the evaluation protocol of GraspNet-1Billion [5] to ensure fair comparisons. Tab. 1 summarizes the results on real-world scenes from the RealSense and Kinect cameras. MMGNet achieves the highest mean AP on both camera settings, reaching 60.34 on RealSense and 50.72 on Kinect. On RealSense, it surpasses the strongest prior method, FlexLoG [11], by 4.32 mean AP and achieves the best performance on the Seen, Similar, and Novel splits. On Kinect, MMGNet is comparable to the strongest prior result in mean AP (50.72 vs. 50.67), while attaining the highest Novel-split AP of 25.49. These results establish the overall competitiveness of MMGNet across the two camera settings, with particularly strong performance on RealSense and the Novel split. The contributions of hierarchical injection and rotational grasp modeling are isolated in the ablation studies below. Qualitative comparisons are provided in the supplementary material.

## 4.3. Ablation Studies and Representation Analysis

Table 1: Comparison with state-of-the-art methods on the GraspNet-1Billion dataset [5]. Each entry reports RealSense/Kinect results.
<table><tr><td>Model</td><td>Input</td><td>Seen</td><td>Similar</td><td>Novel</td><td>Mean AP</td></tr><tr><td>GPD [36]</td><td>P</td><td>22.87/24.38</td><td>21.33/23.18</td><td>8.24/9.58</td><td>17.48/19.05</td></tr><tr><td>PointNetGPD [37]</td><td>P</td><td>25.96/27.59</td><td>22.68/24.38</td><td>9.23/10.66</td><td>19.29/20.88</td></tr><tr><td>Fang et al. [5]</td><td>P</td><td>27.56/29.88</td><td>26.11/27.84</td><td>10.55/11.51</td><td>21.41/23.08</td></tr><tr><td>RGB Matters [9]</td><td>P+I</td><td>27.98/32.08</td><td>27.23/30.40</td><td>12.25/13.08</td><td>22.49/25.19</td></tr><tr><td>Li et al. [38]</td><td>P</td><td>36.55/-</td><td>28.36/-</td><td>14.01/-</td><td>26.31/-</td></tr><tr><td>HGGD [39]</td><td>P+I</td><td>59.36/60.26</td><td>51.20/48.59</td><td>22.17/18.43</td><td>44.24/42.43</td></tr><tr><td>GSNet [8]</td><td>P</td><td>67.12/63.50</td><td>54.81/49.18</td><td>24.31/19.78</td><td>48.75/44.15</td></tr><tr><td>Ma et al. [7]</td><td>P</td><td>63.83/-</td><td>58.46/-</td><td>24.63/-</td><td>48.97/-</td></tr><tr><td>AnyGrasp [6]</td><td>P</td><td>66.12/-</td><td>56.09/-</td><td>24.81/-</td><td>49.01/-</td></tr><tr><td>Wu et al. [40]</td><td>P</td><td>68.21/-</td><td>61.19/-</td><td>25.48/-</td><td>51.63/-</td></tr><tr><td>He et al. [18]</td><td>P</td><td>-/66.77</td><td>-/54.99</td><td>-/23.52</td><td>-/48.43</td></tr><tr><td>ZeroGrasp [21]</td><td>P+I</td><td>70.53/-</td><td>62.15/-</td><td>26.46/-</td><td>53.05/-</td></tr><tr><td>Zhu et al. [22]</td><td>P+I</td><td>74.08/64.20</td><td>62.38/53.41</td><td>27.27/21.38</td><td>54.58/46.33</td></tr><tr><td>FlexLoG [11]</td><td>P+I</td><td>72.81/69.44</td><td>65.21/59.01</td><td>30.04/23.67</td><td>56.02/50.67</td></tr><tr><td>Ours</td><td>P+I</td><td>77.52/68.63</td><td>70.75/58.05</td><td>32.74/25.49</td><td>60.34/50.72</td></tr></table>

P: point cloud; I: RGB image. Zhu et al. [22] uses two RGB-D views.

Our analysis follows the two bottlenecks identified in Sec. 1: preserving multiscale image context during hierarchical 3D representation formation and using the resulting image-informed features for fine-grained local grasp prediction. We first evaluate the Rotational model and analyze what grasp information its pre-grouping representation retains. We then compare alternative image-to-3D fusion strategies and examine how hierarchical injection shapes cross-modal representations. Finally, we apply hierarchical injection to other grasp models while retaining their original prediction heads.

Table 2: Comparison of graspness models and image-to-3D fusion.
<table><tr><td rowspan="2">Graspness Model</td><td rowspan="2">Fusion</td><td colspan="4">AP</td></tr><tr><td>Seen</td><td>Similar</td><td>Novel</td><td>Mean</td></tr><tr><td>View</td><td>None</td><td>67.12</td><td>54.81</td><td>24.31</td><td>48.75</td></tr><tr><td>View</td><td>Hierarchical</td><td>71.97</td><td>58.37</td><td>25.50</td><td>51.95</td></tr><tr><td>Rotational</td><td>None</td><td>71.61</td><td>62.96</td><td>30.30</td><td>54.95</td></tr><tr><td>Rotational</td><td>Direct RGB</td><td>71.86</td><td>61.84</td><td>28.85</td><td>54.18</td></tr><tr><td>Rotational</td><td>Early</td><td>76.50</td><td>68.44</td><td>32.22</td><td>59.05</td></tr><tr><td>Rotational</td><td>Late (Concat)</td><td>75.08</td><td>65.29</td><td>29.52</td><td>56.63</td></tr><tr><td>Rotational</td><td>Late (Add)</td><td>76.91</td><td>67.21</td><td>31.62</td><td>58.58</td></tr><tr><td>Rotational</td><td>Late (Gate)</td><td>74.73</td><td>64.84</td><td>30.40</td><td>56.66</td></tr><tr><td>Rotational</td><td>Hierarchical</td><td>77.52</td><td>70.75</td><td>32.74</td><td>60.34</td></tr></table>

Note: Direct RGB ‡ denotes concatenating RGB values with 3D coordinates at the input.

Rotational Graspness Model. We first compare the conventional View model, which combines view graspness with cylinder grouping, against the Rotational model. As shown in Tab. 2, the Rotational model improves mean AP from 48.75 to 54.95 without image features and from 51.95 to 60.34 with hierarchical injection. Notably, hierarchical injection improves the View model by 3.20 AP points but the Rotational model by 5.39 points, suggesting that the image-informed representation becomes more useful when grasp reasoning preserves the full gripper rotation.

We further decompose the Rotational model in Tab. 3. Under hierarchical injection, replacing view graspness with rotational graspness while retaining conventional cylinder grouping increases mean AP from 51.95 to 56.87 (+4.92). Conditioning the local support region on the selected full rotation through grasp-shaded grouping provides a further 3.47-point gain, reaching 60.34. This decomposition indicates that the improvement is not solely due to changing the grouping region: resolving joint view– in-plane rotations already improves grasp prediction before the rotation-conditioned support is introduced.

We next vary the number of grouped samples K to examine the sample eficiency of the two grouping strategies. Reducing K from 32 to 8 decreases mean AP of cylinder grouping by 4.74 points, whereas grasp-shaded grouping loses only 1.08 points. At K = 16, grasp-shaded grouping is already within 0.31 points of its K = 32 result, and with only eight samples it exceeds cylinder grouping by 7.13 points. These results show that grasp-shaded grouping provides more sample-eficient local aggregation under hierarchical injection.

Table 3: Sensitivity to the number of grouped local samples K under hierarchical image-to-3D injection. ∆ denotes the mean AP change relative to K = 32 for the same grouping type.
<table><tr><td>Grouping</td><td>K</td><td>Seen</td><td>Similar</td><td>Novel</td><td>Mean</td><td>∆Mean</td></tr><tr><td rowspan="4">Cylinder</td><td>8</td><td>68.32</td><td>59.86</td><td>28.22</td><td>52.13</td><td>-4.74</td></tr><tr><td>16</td><td>71.46</td><td>62.91</td><td>30.19</td><td>54.85</td><td>-2.02</td></tr><tr><td>32</td><td>73.50</td><td>65.69</td><td>31.41</td><td>56.87</td><td>0.00</td></tr><tr><td>64</td><td>74.02</td><td>66.04</td><td>31.67</td><td>57.24</td><td>+0.38</td></tr><tr><td rowspan="4">Grasp-shaded</td><td>8</td><td>76.13</td><td>69.90</td><td>31.76</td><td>59.26</td><td>-1.08</td></tr><tr><td>16</td><td>77.21</td><td>70.36</td><td>32.51</td><td>60.03</td><td>-0.31</td></tr><tr><td>32</td><td>77.52</td><td>70.75</td><td>32.74</td><td>60.34</td><td>0.00</td></tr><tr><td>64</td><td>77.57</td><td>70.83</td><td>32.83</td><td>60.41</td><td>+0.07</td></tr></table>

Pre-grouping Representation Analysis. To examine what graspness information is retained before local aggregation, we freeze the View and Rotational models and fit linear probes to their pre-grouping image-informed 3D features at identical query points. We obtain view graspness by averaging rotational graspness over in-plane rotations, and define the rotation residual as each rotation’s deviation from the view average. These targets capture coarse view graspability and finer within-view rotation diferences, respectively. As shown in Tab. 4, View Top-1 accuracy changes only from 3.80% to 4.25%, whereas rotation-residual Spearman correlation and Exact Top-1 accuracy improve from 0.311 to 0.367 and from 35.76% to 39.15%, respectively. This contrast indicates that the main representational change occurs in within-view rotation information rather than generic view discrimination. Rotation regret, which measures the graspness gap between the best and probe-selected rotations, also decreases from 0.2765 to 0.2516, showing that the Rotational representation better preserves the ranking among candidate in-plane rotations. Together, these results are consistent with the intended role of rotational graspness: retaining grasp-quality distinctions among rotations before local grouping. Further details are provided in the supplementary material.

Table 4: Representation analysis of pre-grouping image-informed 3D features.
<table><tr><td>Target</td><td>Metric</td><td>View</td><td>Rotational</td><td>Δ</td></tr><tr><td>View graspness</td><td>Top-1 (%) ↑</td><td>3.80</td><td>4.25</td><td>+0.45</td></tr><tr><td rowspan="4">Rotation residual</td><td>Spearman ↑</td><td>0.311</td><td>0.367</td><td>+0.056</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Exact Top-1 (%) ↑</td><td>35.76</td><td>39.15</td><td>+3.39</td></tr><tr><td>Regret ↓</td><td>0.2765</td><td>0.2516</td><td>-0.0250</td></tr></table>

Comparison of Image-to-3D Fusion Scheme. Under the same rotational graspness model, we compare direct RGB input, early fusion, three late-fusion variants, and our hierarchical image-to-3D injection. As reported in Tab. 2, directly concatenating RGB values with point coordinates slightly reduces the mean AP from 54.95 to 54.18, indicating that isolated color measurements provide insuficient visual context. Early fusion improves the mean AP to 59.05 by introducing contextualized image features before sparse 3D encoding, but exposes the 3D hierarchy to this evidence only once. The late-fusion variants provide smaller gains because image and geometric representations are formed largely independently; among them, Late (Add) reaches 58.58 mean AP. Hierarchical injection introduces aligned multi-scale image features at successive sparse-encoder stages and achieves the highest AP on all three test splits, improving the mean AP by 5.39 points. These results indicate that image features are more efective when introduced before or throughout hierarchical 3D representation formation than when fused after the 3D representation has largely formed. Further analyses in the supplementary material show that, among the learned RGB fusion variants, the advantage of hierarchical injection is not attributable to larger model capacity, or a greater optimization budget, while its computational cost remains comparable to the alternative fusion strategies.

Cross-modal Correspondence under Hierarchical Injection. We extract sparse 3D features at voxel strides s2/s4/s8 and the final backbone output, and pair them with RGB pyramid features $\mathtt { p } 1 / \mathtt { p } 2 / \mathtt { p } 4 / \mathtt { p } 8$ through pixel–point correspondence. Centered Kernel Alignment (CKA) [41, 42] measures similarity in representational geometry and has been used for both layer-wise and cross-modal representation analysis. Because CKA is symmetric, we complement it with bidirectional linear predictability [43, 44]. We fit linear mappings in both 3D→image and image→3D directions and report held-out $R ^ { 2 } ,$ , measuring how much variance in one representation can be linearly predicted from the other. Together, the two measures capture both geometric similarity and directional correspondence between the modalities.

As shown in Fig. 3, hierarchical injection produces the strongest and most consistent cross-modal correspondence across network stages and image scales. At the shallow s2 stage, its CKA reaches 0.82/0.84 with $\mathtt { p 1 } / \mathtt { p 2 }$ , compared with 0.58/0.52 for early fusion and approximately 0.10 for the late-fusion variants. Hierarchical injection also maintains positive bidirectional $R ^ { 2 }$ over all stage–scale pairs. Early fusion preserves strong fine-scale image information, but its image→3D predictability decreases for coarse p8 features as the 3D representation becomes deeper. The latefusion variants show little cross-modal correspondence at intermediate stages, consistent with the two modalities being processed independently before the final fusion operation. Overall, these patterns are consistent with hierarchical injection preserving multi-scale visual evidence throughout 3D representation formation for subsequent grasp prediction.

![](images/ece3725e2ddd3dec9e5df75794babe58be68bcad6523c4f57cd15ec1d23c9c2d.jpg)  
Figure 3: Cross-modal representation analysis on the test set. The top row reports CKA, while the middle and bottom rows report bidirectional linear-probe $R ^ { 2 }$ for 3D→image and image→3D prediction, respectively. Columns correspond to Early, Hierarchical, Late (Concat), Late (Add), and Late (Gate).

Table 5: Transfer of hierarchical image-to-3D injection across diferent grasp methods.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Modality</td><td colspan="4">AP</td></tr><tr><td>Seen</td><td>Similar</td><td>Novel</td><td>Mean</td></tr><tr><td>GSNet [8]</td><td>P</td><td>67.12</td><td>54.81</td><td>24.31</td><td>48.75</td></tr><tr><td>+ Injection</td><td>P+I</td><td>71.97</td><td>58.37</td><td>25.50</td><td>51.95 (+3.20)</td></tr><tr><td>Ma et al. [7]</td><td>P</td><td>64.02</td><td>59.26</td><td>26.18</td><td>49.82</td></tr><tr><td>+ Injection</td><td>P+I</td><td>71.07</td><td>61.02</td><td>27.36</td><td>53.15 (+3.33)</td></tr></table>

Note: P: point cloud; I: RGB image.

Transfer across Grasp Backbones. We equip GSNet [8] and Ma et al. [7] with the same image backbone and hierarchical image-to-3D injection while retaining their original grasp prediction heads. All variants are retrained under the same settings. As shown in Tab. 5, hierarchical injection improves the mean AP of GSNet from 48.75 to 51.95 and that of Ma et al. from 49.82 to 53.15, corresponding to gains of 3.20 and

3.33 points, respectively. The consistent gains on these two representative backbones indicate that the benefit of hierarchical injection is not specific to MMGNet’s rotational graspness head. The larger gain obtained by MMGNet (5.39) is also consistent with the coupled ablation in Tab. 2, where hierarchical injection provides a larger gain with the Rotational model than with the View model.

## 4.4. Robustness under Point and RGB Corruptions

Having established the benefit of hierarchical injection under standard observations, we examine controlled degradation of either modality. For point corruptions, observed RGB is paired with a synthetic point cloud whose geometry alone is perturbed; for RGB corruptions, the naturally noisy observed point cloud is fixed while the corresponding observed RGB is perturbed. Point-only denotes the same rotational graspness model without the image backbone or hierarchical image-to-3D injection. We apply Gaussian noise, smoothing, and Perlin region dropout to the point cloud. Perlin region dropout thresholds Perlin-noise masks [45] to remove contiguous depth regions. RGB corruptions include cutout, blur, brightness, and contrast. All corruptions are applied only during evaluation; severity settings are provided in the supplementary material.

Table 6: Absolute mean AP under the strongest point corruptions. AP retention (%) relative to the uncorrupted synthetic-geometry reference is shown in parentheses.
<table><tr><td rowspan="2">Method</td><td>Reference</td><td>Gaussian</td><td rowspan="2">Smoothing</td><td rowspan="2">Perlin Region Dropout</td></tr><tr><td>Mean AP</td><td>Noise</td></tr><tr><td>GSNet [8]</td><td>55.92</td><td>9.25 (16.54)</td><td>49.19 (87.96)</td><td>50.13 (89.65)</td></tr><tr><td>Ma et al. [7]</td><td>57.68</td><td>25.02 (43.37)</td><td>49.43 (85.71)</td><td>52.79 (91.53)</td></tr><tr><td>AnyGrasp [6]</td><td>60.42</td><td>30.45 (50.40)</td><td>50.27 (83.20)</td><td>53.04 (87.79)</td></tr><tr><td>Point-only</td><td>64.27</td><td>15.53 (24.16)</td><td>56.14 (87.35)</td><td>54.16 (84.26)</td></tr><tr><td>Ours</td><td>67.04</td><td>39.28 (58.60)</td><td>59.11 (88.17)</td><td>61.26 (91.38)</td></tr></table>

Robustness under Point Corruptions. Tab. 6 reports both absolute AP and AP retention under the strongest corruption of each type. The reference AP is measured using observed RGB paired with the uncorrupted synthetic point cloud, while retention normalizes the corrupted AP by this reference. MMGNet achieves the highest absolute AP under all three corruption types and the highest retention under Gaussian noise and smoothing; Ma et al. is marginally higher in retention under Perlin region dropout (91.53% vs. 91.38%).

The paired comparison with Point-only isolates the contribution of hierarchical image-to-3D injection. Under the strongest Gaussian noise, hierarchical injection raises absolute AP from 15.53 to 39.28, a 23.75-point gain, and increases retention from 24.16% to 58.60%. Under Perlin region dropout, it raises AP from 54.16 to 61.26 and retention from 84.26% to 91.38%, indicating that dense image evidence compensates for missing point regions. The smaller improvement under smoothing, from 56.14 to 59.11 AP, suggests that image evidence provides less compensation when local metric contact geometry is directly removed by over-smoothing. Together, these results indicate that hierarchical injection is particularly efective under coordinate noise and structured point loss. Full AP curves across corruption levels are provided in the supplementary material.

Table 7: Utility under clean and RGB-corrupted observations. Parentheses show AP diferences over Pointonly under the same observation condition.
<table><tr><td>Variant</td><td>Clean AP ↑</td><td>RGB-corrupted AP ↑</td></tr><tr><td>Point-only</td><td>54.95 (+0.00)</td><td>54.95 (+0.00)</td></tr><tr><td>Early</td><td>58.96 (+4.01)</td><td>56.47 (+1.51)</td></tr><tr><td>Late (Concat)</td><td>56.52 (+1.56)</td><td>54.34 (-0.61)</td></tr><tr><td>Late (Add)</td><td>58.52 (+3.56)</td><td>56.55 (+1.60)</td></tr><tr><td>Late (Gate)</td><td>56.59 (+1.64)</td><td>53.06 (-1.89)</td></tr><tr><td>Hierarchical (Ours)</td><td>60.33 (+5.38)</td><td>57.28 (+2.32)</td></tr></table>

Utility under RGB Corruptions. Tab. 7 reports clean image utility and absolute performance under RGB corruption, averaged over all corruption types and severity levels. Hierarchical injection achieves the highest clean AP, improving over Pointonly by 5.38 AP points, and remains best after corruption with a 2.32-point advantage.

Late (Add) is less afected by RGB corruption, but its lower clean and corrupted AP indicate more conservative use of image evidence. Late (Concat) and Late (Gate) fall below Point-only after corruption, showing that unreliable image features can become negative evidence.

The corruption-wise results show that cutout causes by far the largest degradation, whereas blur, brightness, and contrast have substantially smaller efects. Detailed results, visualizations, and analysis are provided in the supplementary material.

Inference Eficiency and Cost. We report the grasp-model inference latency and resource cost in Tab. 8. All measurements are conducted on a server equipped with an Intel(R) Xeon(R) Gold 6226R CPU @ 2.90 GHz and a single NVIDIA GeForce RTX 3090. All models process 20,000 scene points as input, while all other methodspecific hyperparameters and inference settings follow the configurations reported in their respective papers.

GSNet and AnyGrasp require 33.82 ms and 44.55 ms per sample, respectively. Despite additionally extracting and hierarchically injecting RGB features, MMGNet requires 71.74 ms, remaining within a practical inference range and substantially faster than Ma et al. (784.05 ms).

MMGNet contains 44.14M parameters and requires 304.47 profiled GFLOPs per sample. Its peak inference and training memory are 539.86 MiB and 16.18 GiB, respectively, while training for 20 epochs requires 1.78 GPU-days. Detailed controlled comparisons of parameters, computation, latency, memory, and training cost across diferent fusion variants are provided in the supplementary material.

## 4.5. Real-world Evaluation

To assess the practical efectiveness of our method, we conducted real-world experiments using a robotic system based on a 7-DOF Franka Panda robot arm equipped with a Franka Hand. An Intel RealSense D415 camera is mounted directly on the gripper for visual perception. The test set includes 22 object types from the GraspNet-1Billion dataset [5], split into two groups: Household (seen and similar objects) and Adversarial (novel objects with multiple color variants) [12]. Each group is evaluated over 20 trials, each conducted in a cluttered scene containing 5–6 randomly selected objects. A detailed illustration of our setup is shown in Fig. 4.

Table 8: Inference latency comparison and deployment cost of MMGNet.  
(a) Model latency comparison
<table><tr><td>Model</td><td>Time (ms)</td></tr><tr><td>GSNet [8]</td><td>33.82</td></tr><tr><td>AnyGrasp [6]</td><td>44.55</td></tr><tr><td>Ma et al. [7]</td><td>784.05</td></tr><tr><td>Ours</td><td>71.74</td></tr></table>

(b) Resource cost
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Parameters</td><td>44.14M</td></tr><tr><td>Profiled GFLOPs</td><td>304.47</td></tr><tr><td>Infer. peak VRAM</td><td>539.86 MiB</td></tr><tr><td>Train peak VRAM</td><td>16.18 GiB</td></tr><tr><td>Training cost</td><td>1.78 GPU-days</td></tr></table>

Table 9: Results of real-world evaluation
<table><tr><td rowspan="2">Model</td><td colspan="2">Household</td><td colspan="2">Adversarial</td></tr><tr><td>Succ./TA</td><td>SR(%)</td><td>Succ./TA</td><td>SR(%)</td></tr><tr><td>Point-only</td><td>108/140</td><td>77.14</td><td>102/143</td><td>71.33</td></tr><tr><td>AnyGrasp [6]</td><td>110/133</td><td>82.71</td><td>108/148</td><td>72.97</td></tr><tr><td>Ours</td><td>116/138</td><td>84.06</td><td>114/140</td><td>81.43</td></tr></table>

We evaluate the Point-only baseline, MMGNet, and AnyGrasp [6] using the same post-processing steps. We record Total Attempts (TA) and Success Rate (SR), where SR is the proportion of successful grasps among all attempts. Further details on the robot setup, exception handling, and post-processing are provided in the supplementary material.

As shown in Tab. 9, MMGNet achieves success rates of 84.06% on Household objects and 81.43% on Adversarial objects. It improves over the Point-only baseline by 6.92 and 10.10 percentage points, respectively, providing real-robot evidence for the benefit of hierarchical image-to-3D injection under the same rotational graspness model. Compared with AnyGrasp [6], MMGNet improves by 1.35 percentage points on Household objects and 8.46 percentage points on Adversarial objects. The larger gain on Adversarial objects is consistent with the corruption analysis in Sec. 4.4, where image evidence provides complementary information when point observations

![](images/d4ebfc5e030a9fd6860e64d2cf5623f4f075df9a17d58acea0ca089aa835600e.jpg)  
(a) Experiment Setting

![](images/09d1efa95c799ab4b3c9339367fbaba4ac623f92c62a05c3baee46e458066e12.jpg)

(b) Household Objects  
![](images/6848b9c94077b5caa797c397db2140293eb5cfa5da0f9630005bdf9eab6760f0.jpg)  
(c) Adversarial Objects  
Figure 4: Our setup of the real-world evaluation.

are degraded.

## 4.6. Failure Mode Analysis

In our real-world evaluations, we observe three representative failure cases: (i) suboptimal grasp selection after the execution-angle constraint, (ii) object slippage, and (iii) inaccurate grasp depth. The first occurs when a high-quality prediction is filtered by the angle constraint, leaving an easier-to-execute but less reliable candidate. Fig. 5(a) illustrates such a case. The bottom row of Fig. 5(a) shows another challenging case in which the model assigns high confidence to a grasp on a small object region. The second failure mode occurs on low-friction objects, such as soap (Fig. 5(b)), which may slip from the gripper during execution. The third example is associated with an inaccurate predicted grasp depth, which can prevent the gripper from achieving force closure (Fig. 5(c)). Additional execution examples are provided in the supplementary material.

![](images/928cccf9268de036540384961df1938c5984e9c5da105105166f0f072a59cdc7.jpg)

![](images/73e3837b8d37f0a81a3b1dc959f7083895ac9f7993ab225c810a6a3a6e8c4cb7.jpg)  
(b) Slipping Object

![](images/b9da81c2e77a6508255ca8995d3b3219ec89a787cc7c8275cd805bccfb99cadf.jpg)  
(a) Suboptimal Grasp Selection

![](images/5e23ff49049be74004869292779009c5931454ddfaa08d3c721ac1c863e10b8d.jpg)  
(c) Bad Grasp Quality  
Figure 5: Failure modes in our real-world experiments.

## 5. Conclusion

This work investigates how image information can be more efectively exploited for scene-level 6-DoF grasp synthesis. MMGNet combines hierarchical image-to-3D feature injection with a rotational graspness model, enabling multi-scale image context to participate throughout 3D representation formation and allowing the selected grasp rotation to guide local feature aggregation. Ablation and representation analyses show that these designs improve image–3D correspondence and preserve more graspquality information among candidate rotations. MMGNet achieves 60.34 mean AP on the RealSense setting of GraspNet-1Billion and improves real-robot grasp success over AnyGrasp by 8.46 percentage points on challenging objects. The consistent gains obtained when applying hierarchical injection to other grasp models further suggest that this image-to-3D interface can benefit broader scene-level grasp pipelines.

Several limitations remain. The current model relies on accurate image–point correspondence and lacks an explicit fallback when RGB information becomes unreliable. The additional image backbone also introduces extra computation. Future work will explore reliability-aware and more eficient image-to-3D fusion, execution-aware grasp scoring, and online feedback for avoiding repeated failures.

## CRediT authorship contribution statement

Rui Cao: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Visualization, Writing - Original Draft, Writing - Review & Editing; Jia-Hui Pan: Methodology, Validation, Investigation, Writing - Review & Editing; Yichuan Li: Software, Validation; Yunfan Yu: Validation; Chi-Wing Fu: Supervision, Writing - Review & Editing; Jiangliu Wang: Writing - Review & Editing; Yun-Hui Liu: Supervision, Project administration, Resources, Funding acquisition.

## Acknowledgments

This work is supported in part by Research Grants Council (RGC) of Hong Kong under Grant 14207324, in part by the HK RGC AoE under AoE/E-407/24-N, in part by the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government via the Hong Kong Centre for Logistics Robotics, and in part by the CUHK T Stone Robotics Institute.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The code, trained models, and data associated with this study will be made publicly available upon publication.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT (OpenAI) to assist with language editing, proofreading, and LAT<sub>E</sub>X formatting. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] X. Li, R. Cao, Y. Feng, K. Chen, B. Yang, C.-W. Fu, Y. Li, Q. Dou, Y.-H. Liu, P.-A. Heng, A sim-to-real object recognition and localization framework for industrial robotic bin picking, IEEE Robotics and Automation Letters 7 (2) (2022) 3961–3968.

[2] J.-H. Pan, X. Gao, K.-H. Hui, S. Zhu, Y.-H. Liu, P.-A. Heng, C.-W. Fu, PPN-Pack: placement proposal network for eficient robotic bin packing, IEEE Robotics and Automation Letters (2024).

[3] M. Danielczuk, A. Mousavian, C. Eppner, D. Fox, Object rearrangement using learned implicit collision functions, in: 2021 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2021, pp. 6010–6017.

[4] Y. Laili, Z. Chen, L. Ren, X. Wang, M. J. Deen, Custom grasping: A regionbased robotic grasping detection method in industrial cyber-physical systems, IEEE Transactions on Automation Science and Engineering 20 (1) (2022) 88– 100.

[5] H.-S. Fang, M. Gou, C. Wang, C. Lu, Robust grasping across diverse sensor qualities: The graspnet-1billion dataset, The International Journal of Robotics Research 42 (12) (2023) 1094–1103.

[6] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, C. Lu, Anygrasp: Robust and eficient grasp perception in spatial and temporal domains, IEEE Transactions on Robotics (2023).

[7] H. Ma, D. Huang, Towards scale balanced 6-DoF grasp detection in cluttered scenes, in: Conference on robot learning, PMLR, 2023, pp. 2004–2013.

[8] C. Wang, H.-S. Fang, M. Gou, H. Fang, J. Gao, C. Lu, Graspness discovery in clutters for fast and accurate grasp detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 15964–15973.

[9] M. Gou, H. Fang, Z. Zhu, S. Xu, C. Wang, C. Lu, Rgb matters: Learning 7-dof grasp poses on monocular rgbd images, in: IEEE International Conference on Robotics and Automation, 2021.

[10] X. Liu, Y. Zhang, H. Cao, D. Shan, J. Zhao, Joint segmentation and grasp pose detection with multi-modal feature fusion network, in: 2023 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2023, pp. 1751–1756.

[11] P. Xie, S. Chen, W. Tang, K. Yang, G. Wang, Rethinking 6-dof grasp detection: A flexible framework for high-quality grasping, Pattern Recognition 170 (2026) 112088.

[12] D. Wang, D. Tseng, P. Li, Y. Jiang, M. Guo, M. Danielczuk, J. Mahler, J. Ichnowski, K. Goldberg, Adversarial grasp objects, in: 2019 IEEE 15th International Conference on Automation Science and Engineering (CASE), IEEE, 2019, pp. 241–248.

[13] A. Depierre, E. Dellandréa, L. Chen, Jacquard: A large scale dataset for robotic grasp detection, in: 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2018, pp. 3511–3516.

[14] S. Ainetter, F. Fraundorfer, End-to-end trainable deep neural network for robotic grasp detection and semantic segmentation from rgb, in: 2021 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2021, pp. 13452– 13458.

[15] Y. Jiang, S. Moseson, A. Saxena, Eficient grasping from rgbd images: Learning using a new rectangle representation, in: 2011 IEEE International conference on robotics and automation, IEEE, 2011, pp. 3304–3311.

[16] A. Zeng, S. Song, K.-T. Yu, E. Donlon, F. R. Hogan, M. Bauza, D. Ma, O. Taylor, M. Liu, E. Romo, et al., Robotic pick-and-place of novel objects in clutter with multi-afordance grasping and cross-domain image matching, in: 2018 IEEE international conference on robotics and automation (ICRA), IEEE, 2018, pp. 1–8.

[17] Z. Chen, Z. Liu, S. Xie, W.-S. Zheng, Grasp region exploration for 7-dof robotic grasping in cluttered scenes, in: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2023, pp. 3169–3175.

[18] R. He, H. Chen, M. Fu, Interest point selection and feature extraction in six-dof grasp pose detection, IEEE/ASME Transactions on Mechatronics (2024).

[19] J. Qiu, F. Wang, Z. Dang, Multi-source fusion for voxel-based 7-dof grasping pose estimation, in: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2023, pp. 968–975.

[20] E. Chisari, N. Heppert, T. Welschehold, W. Burgard, A. Valada, Centergrasp: Object-aware implicit representation learning for simultaneous shape reconstruction and 6-dof grasp estimation, IEEE Robotics and Automation Letters (2024).

[21] S. Iwase, M. Z. Irshad, K. Liu, V. Guizilini, R. Lee, T. Ikeda, A. Amma, K. Nishiwaki, K. Kitani, R. Ambrus, et al., Zerograsp: Zero-shot shape reconstruction enabled robotic grasping, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 17405–17415.

[22] K. Zhu, H. Jiang, J. Qian, J. Xie, A cross-view fusion framework for robust 6-dof grasp pose estimation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 28052–28061.

[23] C. Wang, C. Ma, M. Zhu, X. Yang, PointAugmenting: cross-modal augmentation for 3D object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11794–11803.

[24] Y. Li, A. W. Yu, T. Meng, B. Caine, J. Ngiam, D. Peng, J. Shen, Y. Lu, D. Zhou, Q. V. Le, et al., Deepfusion: Lidar-camera deep fusion for multi-modal 3D object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17182–17191.

[25] K. El Madawi, H. Rashed, A. El Sallab, O. Nasr, H. Kamel, S. Yogamani, RGB and lidar fusion based 3D semantic segmentation for autonomous driving, in: 2019 IEEE Intelligent Transportation Systems Conference (ITSC), IEEE, 2019, pp. 7–12.

[26] D. Robert, B. Vallet, L. Landrieu, Learning multi-view aggregation in the wild for large-scale 3D semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5575–5584.

[27] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al., Openscene: 3D scene understanding with open vocabularies, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 815–824.

[28] W. Hu, H. Zhao, L. Jiang, J. Jia, T.-T. Wong, Bidirectional projection network for cross dimension scene understanding, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2021, pp. 14368–14377.

[29] Y. He, H. Huang, H. Fan, Q. Chen, J. Sun, Ffb6d: A full flow bidirectional fusion network for 6d pose estimation, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[30] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[31] H. Zhao, J. Shi, X. Qi, X. Wang, J. Jia, Pyramid scene parsing network, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2881–2890.

[32] I. Rocco, R. Arandjelovic, J. Sivic, E´ ficient neighbourhood consensus networks via submanifold sparse convolutions, in: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16, Springer, 2020, pp. 605–621.

[33] J. Schult, F. Engelmann, A. Hermans, O. Litany, S. Tang, B. Leibe, Mask3D for 3D Semantic Instance Segmentation, in: International Conference on Robotics and Automation (ICRA), 2023.

[34] C. Choy, J. Gwak, S. Savarese, 4D spatio-temporal convnets: Minkowski convolutional neural networks, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 3075–3084.

[35] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations, 2019.

[36] A. ten Pas, M. Gualtieri, K. Saenko, R. Platt, Grasp pose detection in point clouds, The International Journal of Robotics Research 36 (13-14) (2017) 1455– 1473.

[37] H. Liang, X. Ma, S. Li, M. Görner, S. Tang, B. Fang, F. Sun, J. Zhang, Point-NetGPD: detecting grasp configurations from point sets, in: International Conference on Robotics and Automation, 2019.

[38] Y. Li, T. Kong, R. Chu, Y. Li, P. Wang, L. Li, Simultaneous semantic and collision learning for 6-dof grasp pose estimation, in: IEEE/RSJ International Conference on Intelligent Robots and Systems, 2021.

[39] S. Chen, W. Tang, P. Xie, W. Yang, G. Wang, Eficient heatmap-guided 6- dof grasp detection in cluttered scenes, IEEE Robotics and Automation Letters (2023).

[40] X.-M. Wu, J.-F. Cai, J.-J. Jiang, D. Zheng, Y.-L. Wei, W.-S. Zheng, An economic framework for 6-dof grasp detection, in: European Conference on Computer Vision, Springer, 2024, pp. 357–375.

[41] S. Kornblith, M. Norouzi, H. Lee, G. Hinton, Similarity of neural network representations revisited, in: International conference on machine learning, PMlR, 2019, pp. 3519–3529.

[42] M. Raghu, T. Unterthiner, S. Kornblith, C. Zhang, A. Dosovitskiy, Do vision transformers see like convolutional neural networks?, Advances in neural information processing systems 34 (2021) 12116–12128.

[43] J. Li, Y. Kementchedjhieva, C. Fierro, A. Søgaard, Do vision and language models share concepts? a vector space alignment study, Transactions of the Association for Computational Linguistics 12 (2024) 1232–1249.

[44] C. Kapoor, S. Srivastava, M. Khosla, Bridging critical gaps in convergent learning: How representational alignment evolves across layers, training, and distribution shifts, in: D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, N. Chen (Eds.), Advances in Neural Information Processing Systems, Vol. 38, Curran Associates, Inc., 2025, pp. 140352–140366. doi:10.52202/085713-4690.

[45] Y. Cheng, J. Du, 3d-pnas: 3d industrial surface anomaly synthesis with perlin noise, arXiv preprint arXiv:2504.12856 (2025).