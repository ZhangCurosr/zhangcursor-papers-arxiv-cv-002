# MuCHMDPoser: Multiview RGB-Camera Full-Body Pose Tracking for Virtual Reality

Antoine Maoirca\*   
ISIA Lab   
UMONS   
Mons, Belgium   
George Fletcher <sup>†</sup>   
Department of Computer   
Science   
University of Bath   
Bath, UK   
Thierry Ravet<sup>‡</sup>   
ISIA Lab   
UMONS   
Mons, Belgium   
Thierry Dutoit<sup>§</sup>   
ISIA Lab   
UMONS   
Mons, Belgium

![](images/a260d6c2436e37641b2e2e4a5f136692ca2109dc4a104ac945f0b322f09305c9.jpg)  
Figure 1: Overview of the proposed framework for self-avatar full-body tracking from an HMD and external RGB cameras. First, the HMD and cameras are calibrated into a common coordinate system, so the 18-DoF HMD and controller motion data and the 3D keypoints from video-based pose estimation are expressed in the same frame. Then, a deep learning-based model takes these sparse tracking signals and 3D keypoints as input and estimates the full-body pose. The pipeline runs end-to-end at 60 FPS and supports a variable number of cameras, calibrated into a common coordinate system regardless of their placement.

## ABSTRACT

Accurate full-body tracking in Virtual Reality (VR) is heavily underconstrained when relying solely on consumer-grade sparse sensors, such as Head-Mounted Displays (HMDs) and handheld controllers. Indeed, the self-avatar full-body pose must be inferred based on the 3 available upper-body joints tracked by these devices. This leads to significant tracking error, especially in lowerbody pose. To overcome this issue, we propose to extend the track-

<sup>\*</sup>e-mail: antoine.maiorca@umons.ac.be <sup>†</sup>e-mail:gf321@bath.ac.uk <sup>‡</sup>e-mail: thierry.ravet@umons.ac.be <sup>§</sup>e-mail: thierry.dutoit@umons.ac.be ing setup with external multiview RGB cameras. We show that integrating RGB-based pose estimation this framework is able to significantly decrease lower-body tracking errors, even under common artifacts in RGB-based estimates. Additionally, we evaluate a only-once HMD-camera calibration procedure that does not require any physical markers. We finally propose an implementation of the overall framework, achieving an end-to-end animation at 60 FPS on a Nvidia RTX3080 GPU. Codes are available at https://github.com/antmaio/MuCHMDPoser.

Index Terms: Virtual Reality, Full-Body Pose Estimation, Tracking

## 1 INTRODUCTION

In Virtual Reality (VR), a key component of immersive experience is the concept of the self-avatar i.e., a digital representation that embodies the user within the virtual environment. A fundamental requirement for effective self-avatars is their ability to accurately follow the user’s real-world motions with high fidelity. Many works highlight the importance of an accurate full-body tracking to maintain user experience, especially regarding user-centered qualitative metrics such as Sense-of-Embodiment (SoE) or Sense-of-Agency (SoA) [17, 51].

To track user’s motion, immersive applications often rely on consumer grade VR hardware i.e., Head-Mounted Display (HMD) and handheld controllers. These devices integrate IMU-based trackers to accurately capture head and hands motion. In this context, animating the full body of the articulated self-avatar is not straightforward. The system must infer the complete body pose from the motion signals of the head and hands. This problematic is even more challenging regarding lower-body pose reconstruction. Indeed, this configuration does not involve any lower-body tracking as input to animate the lower-body of the self-avatar with an high fidelity.

To address these limitations, various approaches incorporate additional tracking modalities. Some methods use supplementary IMUs or wearable sensors attached to the lower body [35, 33], while others leverage RGB cameras [49, 32, 15, 28]. Camera-based systems offer a non-intrusive alternative: they require no user-worn hardware and avoid per-user calibration.

In this context, we propose a framework for fast full-body pose estimation in large-scale VR environments. Our setup relies on multiple RGB cameras strategically placed around the scene to capture additional motion cues, particularly for the lower body, helping to resolve pose ambiguities that arise when only head and hand signals are available. The resulting 3D joint positions are then integrated with the VR motion features to estimate the full-body avatar pose.

## 2 RELATED WORK

## 2.1 Full-body Pose Estimation from Sparse IMUs Tracking in VR

Estimating the full-body pose of an articulated avatar from 18- DoF motion data is a major challenge in Extended Reality (XR), primarily due to the complete absence of lower-body tracking in consumer-grade XR hardware. Early solutions relying on Inverse Kinematics [8], Machine Learning [1] or Motion Matching [2, 34] have increasingly been superseded by Deep Learning techniques.

Modern generative techniques such as diffusion [9, 12, 13] and normalizing flows [4] have been introduced to synthesize more natural motion by modeling the underlying data distribution. Building on this generative philosophy, Starke et al. recently introduced codebook matching techniques capable of generating full-body motion across diverse scenarios, such as sports or dance, while enabling character control via traditional inputs like joysticks [40].

Sequence modeling approaches are now widely used to capture temporal and spatial dependencies and predict poses end-to-end [22, 53, 48, 11, 35]. Following this trend, HMD-Poser further optimizes this by employing a lightweight temporal-spatial network for on-device inference, while incorporating online body shape estimation to improve accuracy across diverse user morphologies [10]. To further address the computational bottlenecks of Transformerbased architectures, SSD-Poser introduces a State Space Attention Encoder, leveraging State Space Duality to achieve high-precision pose reconstruction with significantly lower latency and a smaller memory footprint [52].

While efficient, these techniques often rely on reliable hand tracking, making it difficult to adapt when motion signals are lost. Egocentric video-based methods have been introduced to tackle this problem by modeling hand visibility within the HMD field-of-view [5, 21]. Similarly, ReliaAvatar employs an auto-regressive architecture to ensure robustness against both short- and long-term tracking signal loss [36], while DragPoser utilizes a variational latent space to maintain pose sampling efficiency and adherence to real motion distributions during such outages [33].

## 2.2 Multimodal Full-body Pose Estimation in VR

Despite recent advancements, a common limitation shared across methods presented in Section 2.1 is the reduced accuracy of synthesized lower-body motion compared to upper-body gestures. Furthermore, even the most recent models struggle to generalize across a wide variety of static poses due to inherent ambiguities in estimating lower-body configurations from sparse input signals [40].

To address the inherent ambiguity of lower-body pose estimation, several full-body animation approaches rely on complementary sensing modalities to recover additional motion cues. Optical motion capture systems provide highly accurate motion measurements at high frame rates and have been successfully used in VR contexts [45, 19]. However, these systems require expensive infrastructure and cumbersome body-worn markers or sensors, which limits their scalability and practical deployment outside controlled environments. Consequently, more affordable body-tracking solutions have been proposed that combine inertial measurement units (IMUs) with additional sensing technologies. In particular, integrating IMUs with Ultra-Wideband (UWB) localization has been shown to mitigate the drift and jitter inherent to inertial tracking [6, 24, 46]. Despite improving robustness, these approaches still rely on body-worn hardware and typically operate with a single instrumented user per sensor setup. Furthermore, they require peruser calibration procedures to align the different sensing modalities. In contrast, vision-based approaches using RGB cameras enable passive motion capture without body-worn hardware, and require only a one-time extrinsic calibration rather than per-user calibration. Several works fuse IMU measurements with 2D pose estimates from RGB images to alleviate occlusion and depth ambiguity [32, 49, 7, 50]. FusePose [7] combines inertial and visual observations through kinematic modeling and state-space fusion, while VR-oriented systems such as HybridTrak [49] and PoseVR [50] augment headset and controller tracking with additional RGB views to estimate lower-body motion. However, these approaches are typically designed around single-view configurations, which limits their scalability to larger environments and multi-user scenes.

Camera-based systems are, moreover, restricted to the space they cover. To overcome this constraint, recent works estimate fullbody motion from head-mounted sensors only, such as egocentric cameras and IMUs, leveraging stereo or temporal visual cues [23, 47, 41]. Yet, egocentric views suffer from severe self-occlusion and a narrow field of view that often leaves the lower body out of sight, making the kinematics of invisible joints highly ambiguous. These systems are also prone to “false positives,” where limbs of other people are attributed to the user.

Observing the user from the outside avoids these issues. Multiview setups reduce self-occlusion, since a joint hidden in one view usually remains visible in another, and per-view 2D detections can be lifted to 3D by triangulation, typically through the Direct Linear Transform (DLT) [20]. Yet, external cameras alone are not sufficient to animate a self-avatar: their 3D estimates remain noisy and are expressed in the camera frame rather than in the HMD tracking space, whereas the HMD and controllers already provide accurate, low-latency head and hand measurements. The two modalities are therefore complementary.

Building on this, we propose a framework that fuses sparse HMD and controller signals with 3D keypoints from external multiview RGB cameras. Our contributions are:

• A deep learning-based model that estimates the full-body pose from 18-DoF HMD and controller data and RGB-based 3D keypoints, significantly reducing lower-body tracking errors, even under common artifacts in RGB-based estimates.

• A one-time, marker-free HMD-camera calibration procedure that expresses both modalities in a common coordinate system and supports a variable number of cameras.

• An implementation of the complete pipeline achieving endto-end animation at 60 FPS on an Nvidia RTX3080 GPU.

## 3 FULL-BODY POSE ESTIMATION

## 3.1 Problem Formulation

This work addresses the challenge of accurately animating fullbody self-avatars using sparse, multimodal body tracking for immersive applications. The core objective is to define a mapping function $f _ { \phi }$ that transforms motion data from consumer-grade VR hardware $\mathbf { m } _ { J _ { V F } } ^ { t }$ and off-the-shelf RGB video-based multiview pose estimations $p _ { J _ { e x t } } ^ { t }$ into an articulated full-body avatar that follows accurately user’s movements. The overview of this framework is shown in Figure 1.

We assume that head and hands nodes $J _ { V R } = \{ H M D , L H , R H \}$ are tracked by consumer-grade level devices $i . e .$ , HMD and handheld controllers. For each timestep t, we collect motion features $\mathbf { m } _ { J _ { V R } } ^ { t } = ( m _ { H M D } ^ { t } , m _ { L H } ^ { t } , m _ { R H } ^ { t } )$ , respectively, for head, left and right hand node. we define these motion features as global position $p _ { J _ { V R } } ^ { t } \in \mathbb { R } ^ { | J _ { V R } | \times 3 }$ and rotation $\mathbf { R } _ { J _ { V R } } ^ { t } \mathbb { R } ^ { | J _ { V R } | \times 6 }$ , as well as their respective linear and angular velocities, $\nu _ { J _ { V R } } ^ { t } \in \mathbb { R } ^ { | J _ { V R } | \times 3 }$ and $\omega _ { J _ { v R } } ^ { t } \in$ $\mathbb { R } ^ { | J _ { V R } | \times 6 }$ . Rotations are expressed in 6D, particularly useful in deep learning due to the continous mapping of this representation in Eculidean space [54]. Following previous works [10], these motion features are also expressed in head space coordinate, leading to the set of motion features for HMD and hands defined in Equation 1. $J _ { V R } / \mathrm { h }$ denotes motion features expressed in HMD space.

$$
\mathbf { m } _ { J _ { V R } } ^ { t } = \left[ \left\{ p _ { J _ { V R } } ^ { t } , \mathbf { R } _ { J _ { V R } } ^ { t } , \nu _ { J _ { V R } } ^ { t } , \omega _ { J _ { V R } } ^ { t } \right\} , \left\{ p _ { J _ { V R } / \mathrm { h } } ^ { t } , \mathbf { R } _ { J _ { V R } / \mathrm { h } } ^ { t } , \nu _ { J _ { V R } / \mathrm { h } } ^ { t } , \omega _ { J _ { V R } / \mathrm { h } } ^ { t } \right\} \right]
$$

The second motion data stream consists of global 3D Cartesian positions $p _ { J _ { e x t } } ^ { t } \in \mathbb { R } ^ { | J _ { e x t } | \times 3 }$ of body joints $J _ { e x t }$ . These positions are estimated using off-the-shelf, multiview 2D pose estimation on videos combined with a 3D pose lifting method $\mathbf { \dot { \Psi } } _ { \Psi } ,$ as stated by Equation 2.

$$
p _ { J _ { e x t } } ^ { t } = \Psi \left( \left\{ \vphantom { \sum _ { i } } \left( u _ { J _ { \mathrm { e x t } } , c } ^ { t } , w _ { J _ { \mathrm { e x t } } , c } ^ { t } \right) \right\} _ { c \in \mathcal { C } } , \left\{ s _ { J _ { \mathrm { e x t } } , c } ^ { t } \right\} _ { c \in \mathcal { C } } \right)\tag{2}
$$

$\left\{ \left( u _ { J _ { \mathrm { e x t } } , c } ^ { t } , w _ { J _ { \mathrm { e x t } } , c } ^ { t } \right) \right\} _ { c \in \mathcal { C } } , \left\{ s _ { J _ { \mathrm { e x t } } , c } ^ { t } \right\} _ { c \in \mathcal { C } }$ are respectively 2D positions in image space and the estimation confidence score for joint $j \in J _ { e x t }$ in camera $c \in \mathcal { C }$ , where $\mathcal { C }$ is the set of involved calibrated cameras.

Hence, this configuration leads to an input vector formed by the concatenation of the sensor data and the estimated 3D positions $\left( \mathbf { m } _ { J _ { V R } } ^ { t } , p _ { J _ { e x t } } ^ { t } \right)$ . The predicted full-body motion of the self-avatar is given by

$$
\left[ \hat { T } ^ { t } , \hat { \mathbf { R } } _ { r o o t } ^ { t } , \hat { \Theta } _ { t } ^ { t } \right] = f _ { \phi } \left( \mathbf { m } _ { J _ { V R } } ^ { t } , p _ { J _ { e x t } } ^ { t } \right)\tag{3}
$$

where $\hat { T } _ { r o o t } ^ { t }$ and $\hat { \mathbf { R } } _ { r o o t } ^ { t }$ represent the predicted global body displacement and orientation, while $\hat { \Theta } ^ { t }$ denotes the predicted local orientation for each bone.

## 3.2 Input Data Acquisition

We use AMASS [27] as ground truth motion data source. It unifies multiple optical motion capture datasets via the SMPL body model [25]. From this dataset, we extract head and hands motion $\left\{ p _ { J _ { V R } } ^ { t } , { \bf R } _ { J _ { V R } } ^ { t } , \nu _ { J _ { V R } } ^ { t } , \omega _ { J _ { V R } } ^ { t } \right\}$ to simulate the tracking signals from a HMD and controllers.

The 3D Cartesian coordinates of key body joints $p _ { J _ { e x t } } ^ { t }$ are collected through off-the-shelf multiview 2D pose estimations on videos rendering of the motion dataset. These videos are rendered from virtual cameras strategically positioned in the scene. A sample of the scene rendering with the virtual cameras $c \in \mathcal { C }$ is shown as matter of example in Figure 2. The extrinsic parameters of these virtual cameras are optimized to ensure the entire range of motion within the dataset remains within the field of view. The distribution of the root positions on the ground plane within the dataset is displayed in Figure 3. Moreover, the cameras intrinsic parameters are fixed to match the specifications of a real-world ZED2 camera setup.

From these rendered videos, we extract 2D joint keypoints position $\left( u _ { J _ { \mathrm { e x t } } , c } ^ { t } , w _ { J _ { \mathrm { e x t } } , c } ^ { t } \right) _ { c \in \mathcal { C } }$ and the related confidence scores $\left( s _ { J _ { \mathrm { e x t } } , c } ^ { t } \right) _ { c \in \mathcal { C } }$ using an off-the-shelf 2D pose estimation model. A significant challenge is that such models, while fast, are prone to estimation artifacts e. $g . ,$ , occlusions or jitter, which severely degrade the quality of 3D poses recovered through standard pose lifting methods.

To tackle this effect, Ψ from Equation 2 combines the Direct Linear Triangulation (DLT) method with Kalman filter to estimate smooth 3D positions of full-body joints $p _ { J _ { e x } } ^ { t }$ from the multiple 2D poses estimated in each camera $c \in \mathcal { C }$ . For each frame t and joint $j \in J _ { e x t }$ , we apply DLT using the 2D detections from the two cameras with the highest confidence scores $s _ { j , c } ^ { t } .$ If either of these two selected cameras yields a confidence score $s _ { j , c } ^ { t } < \kappa ,$ , where $\kappa = 0 . 5$ is our confidence threshold, we consider the joint as occluded and set its 3D position $p _ { j } ^ { t } = \mathbf { 0 }$

## 3.3 Architecture

As defined in related work [28], we rely on an extended version of HMD-Poser [10] to estimate the full-body pose of the self-avatar, suitable for its lightweight, yet effective avatar animation framework. The overview of the proposed architecture can be found in Figure 4. First, the motion features from HMD and handheld controllers $\bf { m } _ { J _ { \mathrm { V R } } } ^ { t }$ are projected into an embedding space in a channel-wise manner. We extend this embedding layer to incorporate 3D positions $p _ { J _ { e x } } ^ { t }$ from the pose estimation framework Ψ. Then, the spatial-temporal encoder consists in LSTM and Transformer blocks, to capture respectively the temporal dynamic and spatial relationship in motion data. Finally, we estimate the full state of the articulated avatar i.e., global translation and orientation as well as local bone orientation, using a fully-connected-based layer as pose estimation from previoulsy encoded information.

## 4 HMD AND EXTERNAL CAMERA RIG CALIBRATION

Fusing HMD motion data with video-based 3D keypoints requires both streams to be expressed in a common coordinate system. The two tracking systems, however, are independent. The HMD estimates its own 6-DoF pose through inside-out tracking in a device-centric frame $\mathcal { H }$ , whereas the external cameras only know the physical world frame $\mathcal { W }$ defined by their calibration (Fig. 6). Neither system observes the other’s frame, so the rigid transform $\mathbf { T } \mathcal { H } \xrightarrow { } \mathcal { W }$ between them must be estimated explicitly. However, this estimation is non-trivial. Indeed, $\mathcal { H }$ is initialized arbitrarily at each application startup, so a transform estimated once is valid for one session only. Then, a session-invariant frame $\mathcal { H } _ { f }$ must therefore be provided by the runtime, allowing a one-time calibration procedure between HMD and cameras. Nevertheless, relocalization errors or tracking failures of $\mathcal { H } _ { f }$ can make it deviate unpredictably. Since every calibrated pose is expressed relative to $\bar { \mathcal { H } } _ { f }$ , such deviations propagate directly into the user’s pose in $\mathcal { W }$ , without any warning to the application.

![](images/f5710af8a06119ee7efe6f7c3bcbbd01768d2806f6feb89631eba48e2a13c73b.jpg)  
Figure 2: Virtual configuration that imitates hardware setup. We define 3 virtual cameras, whose the intrinsic and extrinsic parameters are known. Off-the-shelf 2D pose estimation is applied in every rendered view to capture lower-body motion cues.

![](images/9ef0f9737708d1dd9a5c37044692f6a014520f98cfdf7404b2ade5733271b311.jpg)  
Figure 3: Root global position distribution on the ground surface under a standard evaluation protocol using AMASS dataset. This delimitates an approximative $6 0 m ^ { 2 }$ interaction area.

## 4.1 Session-Invariant Reference H Estimation

We define $\mathcal { H } _ { f }$ using Meta Mixed Reality Utility Kit (MRUK)<sup>1</sup>. MRUK provides tools to perform a semantic scan of the room (walls, floor, ceiling, furniture) configured at the OS level before the application starts and exposes the resulting geometry in Unity. We build a deterministic reference axis system $\mathcal { H } _ { f }$ anchored to the ground and aligned with the longest detected wall. The rigid transform $\mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { H } } \mathbf { \bar { \Pi } } \in S E ( 3 )$ mapping coordinates from axis system $\mathcal { H } _ { f }$ to $\mathcal { H }$ is then known.

## 4.2 Calibration approach

Inspired by the standard multiview calibration procedure [20], we use a planar checkerboard of known geometry as the physical reference shared by both tracking devices. Let $\dot { \mathbf { X } } _ { i } \in \mathbb { R } ^ { 4 } , \dot { \iota } = 1 . . N .$ , be the homogeneous 3D positions of its N inner corners, defined in $\mathcal { W }$ with the board in the $\hat { Z } = 0$ plane. For each external camera k, the corners are detected and the extrinsics are recovered by solving the

![](images/0da2a0deed82e8b97d821a7182592f85f36c981e9e1141ef239c8553f322bbef.jpg)  
Figure 4: Full-body Pose Estimation model. We extend the architecture from HMD-Poser [10] to feed the filtered positions $p _ { J _ { e x t } } ^ { t }$ of body joints $J _ { e x t }$ acquired by the multiview pose estimation setup. Extending the channel-wise embedding, we are able to provide embeddings ${ \bf \bar { \boldsymbol { e } } } _ { j } ^ { t }$ from body joints $J _ { e x t }$ outside standard head and hands tracking.

Perspective-n-Point (PnP) problem [14]:

$$
\mathbf { T } _ { \mathcal { W }  k } = \underset { \mathbf { T } \in S E { ( 3 ) } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } \| \mathbf { x } _ { i } ^ { k } - \pi ( \mathbf { K } _ { k } \mathbf { T } \mathbf { X } _ { i } ) \| ^ { 2 } ,\tag{4}
$$

where $\pi ( \cdot )$ denotes the perspective division, $\mathbf { K } _ { k }$ the intrinsics provided by the manufacturer and $\left\{ \mathbf { x } _ { i } ^ { k } \right\} _ { i = 1 } ^ { N }$ the $N$ corners detected in the image captured by the camera k. The transform $\mathbf { T } _ { \mathcal { W }  k }$ represents the camera pose in world reference frame $\mathcal { W }$

Then, the HMD is registered with a single snapshot of the same checkerboard, taken with its left passthrough camera $\mathcal { C }$ , while we record the camera pose in ${ \mathcal { H } } _ { f } ,$ denoted $\dot { \mathbf { T } } _ { \mathcal { H } _ { f }  \mathcal { C } }$ . The same PnP solve (Equation 4) yields the HMD left camera pose in world reference $\mathbf { T } _ { \mathcal { W }  \mathcal { C } }$ . The transform between the virtually and physically anchored reference frames, respectively $\mathcal { H } _ { f }$ and $\dot { \mathcal { H } }$ , is retrieved by

![](images/7cd3bcc2c445ff6164b5d768d8fab6c0ad39efe34ff2dbcf408abf96abfc33e5.jpg)  
Figure 5: Overview of the calibration setup used to register the multi-camera system, the HMD, and the handheld controllers into a common reference axis system. Left: Schematic of the full setup, showing the subject wearing an HMD inside the capture volume observed by the external RGB cameras. Right: (a) Standard multiview camera calibration protocol [20]. (b) The Meta Mixed Reality Utility Kit (MRUK) provides a room scan from passthrough, allowing a physically-anchored virtual tracking space $\mathcal { H } _ { f }$ . (c) Snapshot from the HMD camera $k \equiv \mathcal { C }$ with corner detection $\mathbf { x } _ { i } ^ { k }$ and pose registration $\mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { C } }$ . This setup closes the loop between the HMD tracking space, the floor-anchored frame within the virtual application, and the external camera system.

transform composition:

![](images/6273548499262bae7832f46967153b48d43f922bac49cfec3d0280f10b1936f4.jpg)  
Figure 6: Reference frames involved in HMD pose estimation. The device-centric frame $\mathcal { H }$ , re-initialized arbitrarily at each VR application startup, is mapped to the session-invariant frame $\mathcal { H } _ { f }$ . Both $\mathcal { H }$ and $\mathcal { H } _ { f }$ live within the VR software domain, while $\mathcal { H } _ { f }$ additionally serves as the bridge to the physical world domain: the rigid-body transform $\mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { W } }$ , obtained from the multi-camera calibration, relates $\mathcal { H } _ { f }$ to the shared physical world frame $\mathcal { W }$

$$
\mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { W } } = \mathbf { T } _ { \mathcal { W }  \mathcal { C } } ^ { - 1 } \mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { C } }\tag{5}
$$

Finally, the full pipeline transform between initial tracking space for the 18DoF motion data from consumer-grade VR devices $\mathcal { H }$ and world reference axis system shared across external cameras $\mathcal { W }$

$$
\mathbf { T } _ { \mathcal { H }  \mathcal { W } } = \mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { W } } \mathbf { T } _ { \mathcal { H }  \mathcal { H } _ { f } }\tag{6}
$$

Because $\mathcal { H } _ { f }$ is assumed session-invariant, this registration is performed once. With that procedure, we are then able to express the 6DoF motion data from the HMD in the world reference frame shared with the external RGB cameras. Similarly, we can also express the 3D positions estimated via pose estimation and DLT applied to the camera videos in the virtual tracking spaces $\mathcal { H } _ { f }$ and ${ \mathcal { H } } .$

## 5 EXPERIMENTS

## 5.1 Full-Body Pose Estimation

From RGB video inputs, we estimate the 3D positions of $| J _ { e x t } | = 1 2$ body joints, including the feet, ankles, hips, elbows, shoulders, and wrists. The proposed approach is evaluated on the AMASS dataset [27] following two standard evaluation protocols introduced in prior work [53].

Protocol 1 This protocol follows a commonly used training and evaluation split constructed from the CMU [44], HDM05 [30], and BMLrub [42] motion capture subsets. In total, the dataset contains approximately 20 hours of human motion capture data. The training and testing partitions are generated by mixing motion sequences from each subset, following the standard split strategy adopted in recent self-avatar full-body tracking methods [13, 22].

![](images/61d800f9e3134973d3e2f1b0e6bf605ebf9cf7a676a0c2111a7a6c3ac6a42842.jpg)  
Figure 7: Protocol to assess the consistency and stability of $\mathcal { H } _ { f }$ tracking space. We take several snapshots from left camera of the HMD, register its pose relatively to $\mathcal { H } _ { f }$ . We use this transform to reproject the control points into the image plane of each snapshot and measure the discrepancy between these 2D points and the corners detected by OpenCV.

Protocol 2 In this protocol, the model is trained on a larger set of AMASS subsets, including ACCAD [16], BMLrub [42], BMLmovi [18], CMU [44], EKUT [29], Eyes Japan Dataset [26], KIT [29], HDM05 [30], MPI Limits [3], SFU [31], and TotalCapture [43]. Evaluation is performed on the Transitions Mocap [27] and HumanEva [38] subsets, which are excluded from training.

Protocol 1 comprises approximately 20 hours of high-fidelity motion capture sequences, whereas Protocol 2 spans roughly 43 hours of recorded motion data. For both protocols, training is performed using a temporal window of 40 frames. We consider both motion streams $i . e .$ , 3D positions $p _ { J _ { e x t } } ^ { t }$ and IMUs trackers $\mathbf { m } _ { J _ { V R } } ^ { t }$ synchronized at constant framerate 60Hz. The model is trained for 400 epochs on a single RTX 4090 GPU with a batch size of 128. The model is optimized under the same loss function as in related work [10]. This optimization is performed using the Adam optimizer with a one-cycle learning rate policy [39], reaching a maximum learning rate of $\lambda _ { m a x } = \bar { 1 } e - 3$ . To estimate 2D positions in each camera view, we rely on the nano model of YOLOv8 framework for a fast data acquisition.

Finally, full-body pose estimation performance is evaluated through standard objective metrics: Mean Per Joint Positional Error (MPJPE), Mean Per Joint Rotational Error (MPJRE), Mean Per Joint Velocity Error (MPJVE), and the jitter ratio computed between the jitter in predicted and ground-truth global positions of avatar joints.

## 5.2 Calibration

To assess the stability of a candidate frame $\mathcal { H } _ { f }$ without external ground-truth equipment, we use a cross-reprojection protocol, shown by Figure 7. We capture $S = 6$ snapshots $\{ \mathbf { S } _ { k } \} _ { k = 1 , \dots , S }$ of the board from different viewpoints, each pairing a passthrough image with the simultaneously recorded camera pose in $\mathcal { H } _ { f }$ . For every snapshot k, we compute $\mathbf { T } _ { \mathcal { W }  \mathcal { C } } ^ { k }$ by Equation 4, so the control points can be expressed in the candidate axis system following Equation 5. In a round-robin fashion, the points obtained from a reference snapshot a are projected into every other snapshot b $\neq$ a using the pose recorded for b. We report two metrics:

$$
\varepsilon _ { \mathrm { 2 D } } ^ { a } = \frac { 1 } { ( S - 1 ) N } \sum _ { b \neq a i = 1 } \sum _ { \vphantom { b ^ { a } } } ^ { N }  \pi \Big ( \mathbf { K } _ { \mathcal { C } } \mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { C } } ^ { b } \mathbf { X } _ { i , \mathcal { H } _ { f } } ^ { a } \Big ) - \mathbf { x } _ { i } ^ { b }  _ { 2 } ,\tag{7}
$$

$$
\varepsilon _ { \mathrm { 3 D } } ^ { a } = \frac { 1 } { ( S - 1 ) N } \sum _ { b \neq a } \sum _ { i = 1 } ^ { N } \left\| \mathbf { X } _ { i , \mathcal { H } _ { f } } ^ { a } - \mathbf { X } _ { i , \mathcal { H } _ { f } } ^ { b } \right\| _ { 2 } ,\tag{8}
$$

with $\mathbf { X } _ { i , \mathcal { H } _ { f } } ^ { k } = \mathbf { T } _ { \mathcal { W }  \mathcal { H } _ { f } } ^ { k } \mathbf { X } _ { i , \mathcal { W } }$

The two metrics capture complementary aspects of frame quality. $\varepsilon _ { 3 D }$ isolates the intrinsic stability of the candidate frame $\mathcal { H } _ { f }$ itself, independently of any camera- or projection-related error. Since the checkerboard is rigid and its corners $\mathbf { X } _ { i , \mathcal { W } }$ are fixed and known once and for all from the multiview calibration, any discrepancy between ${ \bf X } _ { i , \mathcal { H } _ { f } } ^ { a }$ and $\mathbf { X } _ { i , \mathcal { H } _ { f } } ^ { b }$ can only originate from an inconsistency between $\mathbf { T } _ { \mathcal { W }  \mathcal { H } _ { f } } ^ { a }$ and $\mathbf { T } _ { \mathcal { W }  \mathcal { H } _ { f } } ^ { b }$ . Thus, ε directly measures whether the candidate frame’s pose relative to the world remains consistent across snapshots, or drifts over time and viewpoint. It is expressed in physical units (mm) and does not depend on external camera intrinsics or on the accuracy of the HMD-to-camera transform.

In contrast, $\varepsilon _ { 2 D }$ is an end-to-end, application-facing metric. It evaluates whether a 3D point expressed in $\mathcal { H } _ { f }$ from a reference snapshot a correctly reprojects onto the corresponding checkerboard corner when using the camera pose $\mathbf { T } _ { \mathcal { H } _ { f }  \mathcal { C } } ^ { b }$ recorded in a different snapshot $b .$ Unlike $\varepsilon _ { 3 D } ,$ this metric additionally incorporates the accuracy of the HMD’s own tracking relative to ${ \mathcal { H } } _ { f } ,$ the camera intrinsics $\mathbf { K } _ { \mathcal { C } } ,$ , and the corner-detection noise in $\mathbf { x } _ { i } ^ { b }$ . Expressed in pixels, it answers a practical question for augmented-reality applications: if virtual content is anchored in $\mathcal { H } _ { f }$ during one session or viewpoint, how many pixels of misalignment will be observed when revisiting it from another viewpoint?

Reporting both metrics allows the source of error to be localized: a low $\varepsilon _ { 3 D }$ combined with a high $\varepsilon _ { 2 D }$ indicates that the candidate frame itself is stable but that HMD tracking or intrinsic calibration is the limiting factor, whereas a high $\varepsilon _ { 3 D }$ indicates that the frame itself drifts, which necessarily propagates into $\varepsilon _ { 2 D }$ as well. This comparison is particularly relevant when contrasting the MRUKbased frame (sceneSpace) with the baseline $\mathcal { H }$ (trackingSpace), as it reveals whether room-scan-based floor anchoring reduces drift $( \varepsilon _ { 3 D } )$ relative to the native tracking origin, and whether this improvement translates into a user-perceptible reduction in reprojection error $\left( \varepsilon _ { 2 D } \right)$ . Note that $\varepsilon _ { 3 D }$ and $\varepsilon _ { 2 D }$ are reported in different units (mm and pixels, respectively) and are therefore not directly comparable in magnitude, only in relative trend across the two candidate frames.

As hardware tracking devices, we use two ZED2 as external RGB cameras (1280px˙ ×720.px), capturing videos from their left lens. We employ a checkerboard with $5 \times 6$ inner corners and 25 cm squares. We use MetaQuest 3 as consumer-grade VR devices, and the HMD calibration relies on its left passthrough camera (1280.px×1280.px).

## 6 RESULTS

## 6.1 Calibration

Table 1 reports the cross-reprojection error $\varepsilon _ { \mathrm { 2 D } }$ and the 3D consistency error $\varepsilon _ { 3 \mathrm { D } }$ for each reference snapshot. The MRUK-based frame $\mathcal { H } _ { f }$ and the baseline tracking space $\mathcal { H }$ yield identical results, with $\Delta = 0 . 0 0$ for every snapshot. Across snapshots, ε<sub>2D</sub> ranges from 2.37 to 3.86 px (mean 3.19 px) and $\varepsilon _ { 3 \mathrm { D } }$ from $_ { 0 . 6 2 }$ to 1.03 cm (mean 0.81 cm). Defining $\mathcal { H } _ { f }$ from the room scan therefore provides the same stability as the raw tracking frame, and both remain subject to the drift of the headset’s visual-inertial odometry. This residual drift, below 1 cm on average, is smaller than the error typically reported for video-based 3D keypoint estimation, so we consider it compatible with our target application of fusing RGBbased keypoints with HMD data.

Table 1: Cross-reprojection errors for each reference snapshot, comparing the standard tracking space H with the MRUK-based frame $\mathcal { H } _ { f }$ . The reprojection procedure results in sub-centimer precision, in average. $\dot { \Delta } \stackrel { - } { = } \varepsilon ( \dot { \mathcal { H } _ { f } } ) - \varepsilon ( \mathcal { H } )$ indicates that both tracking space axis systems share similar stability properties. Lower is better.
<table><tr><td></td><td colspan="3"> $\varepsilon _ { \mathrm { 2 D } } ^ { a } \left( \mathrm { p x } \right)$ </td><td colspan="3"> $\varepsilon _ { 3 \mathrm { D } } ^ { a } \left( \mathrm { c m } \right)$ </td></tr><tr><td>Ref. snapshot a</td><td>H</td><td>Hf</td><td>Δ</td><td>H</td><td>Hf</td><td>Δ</td></tr><tr><td>S1</td><td>2.37</td><td>2.37</td><td>0.00</td><td>0.62</td><td>0.62</td><td>0.00</td></tr><tr><td>S2</td><td>3.55</td><td>3.55</td><td>0.00</td><td>0.82</td><td>0.82</td><td>0.00</td></tr><tr><td>S3</td><td>3.58</td><td>3.58</td><td>0.00</td><td>1.03</td><td>1.03</td><td>0.00</td></tr><tr><td>S4</td><td>2.82</td><td>2.82</td><td>0.00</td><td>0.77</td><td>0.77</td><td>0.00</td></tr><tr><td>S5</td><td>3.86</td><td>3.86</td><td>0.00</td><td>0.91</td><td>0.91</td><td>0.00</td></tr><tr><td> $\mathbf { S } _ { 6 }$ </td><td>2.93</td><td>2.93</td><td>0.00</td><td>0.71</td><td>0.71</td><td>0.00</td></tr><tr><td>Mean</td><td>3.19</td><td>3.19</td><td>0.00</td><td>0.81</td><td>0.81</td><td>0.00</td></tr></table>

Table 2: Multiview 2D-to-3D pose lifting with direct linear triangulation (DLT), evaluated on the two protocols. ↓: lower is better.
<table><tr><td>Protocol</td><td>Occlusion rate [%]</td><td>MPJPE↓ [cm]</td><td>MPJVE↓ [cm/s]</td><td>Jitter ↓ ratio</td></tr><tr><td>Protocol 1</td><td>0.95</td><td>12.94</td><td>155.78</td><td>51.12</td></tr><tr><td>Protocol 2</td><td>0.58</td><td>21.31</td><td>206.80</td><td>72.53</td></tr></table>

## 6.2 Multiview 2D to 3D positions lifting

Table 2 reports the reconstruction quality of direct linear triangulation (DLT), which we use as a fast, lightweight baseline for multiview 2D-to-3D lifting. For a fair comparison, we remove frames where one occlusion occurs for at least one joint, as we zeroed the positions of occluded joints. DLT requires no learned component and runs per frame, which makes it suitable for building the 3D pose data at scale. Its output is nevertheless imperfect: the MPJPE is 12.94 cm on Protocol 1 and 21.31 cm on Protocol 2, and the velocity error (155.78 and 206.80 cm/s) and jitter ratio (51.12 and 72.53) show that the reconstructed trajectories contain frame-toframe noise. We do not attempt to remove these inaccuracies at the reconstruction stage. Since the DLT poses are part of the training set, the avatar animation network Φ is exposed to this noise during training and learns to produce smooth, plausible motion from it.

## 6.3 Full-body Pose Estimation

Table 3 presents the quantitative evaluation of our proposed fullbody pose estimation pipeline across Protocol 1 and Protocol 2. We compare our fused approach against a baseline relying exclusively on the three standard sparse trackers (the HMD and two handheld controllers). This baseline is similar to the 3-tracker configuration in HMD-Poser [10]. We provide samples of generated poses with ground truth in Figure 8.

The 3-tracker baseline exhibits significant ambiguity for untracked joints, with the lower-body position error (L-PE) reaching 9.73 cm in the dynamic Protocol 2. Integrating external 3D pose observations markedly improves spatial accuracy: MPJPE drops by 36 % in Protocol 1, while L-PE is reduced to 5.48 cm in Protocol 2. Beyond positional accuracy, the fusion also refines rotational and temporal dynamics. MPJRE and MPJVE decrease substantially across both protocols, indicating a more anatomically and kinematically plausible pose reconstruction. The upper-body position error (U-PE) sees more marginal gains, as it is already highly constrained by the HMD and controller hardware. Overall, the improvement across all metrics highlights the effectiveness of the external observations in resolving the degrees of freedom that sparse tracking alone cannot capture.

The external 3D observations also introduce some highfrequency noise into the kinematic chain: the jitter ratio increases from 1.74 to 2.40 in Protocol 1, and from 2.10 to 3.44 in Protocol 2. This increase remains moderate compared with the gains in spatial and rotational accuracy.

Figure 9 shows sample of self-avatar poses estimated by our proposed framework implemented in game engine. Extending sparse head and hands tracking with RGB video-based tracking enhances the accuracy of the avatar full-body pose, especially at lower-body parts. The inference of our full-body pose estimation framework provides 60FPS animation using Unity Inference Engine <sup>2</sup> on a Nividia RTX3080.

## 7 LIMITATIONS AND FUTURE WORK

Our quantitative evaluation relies on AMASS sequences rendered from virtual cameras with ZED2 intrinsics (Section 3.2), which lets us study full-body accuracy against reliable ground truth. The realworld setup, in contrast, is only assessed through the calibration metrics and qualitative results (Figure 9), since we have no synchronized ground-truth lower-body motion for these recordings. Offthe-shelf 2D detectors may behave differently on real video than on rendered avatars (lighting, clothing, motion blur), which can create a domain gap. Future work should record a real multiview dataset with synchronized optical motion capture, so that real-world pose accuracy can be measured directly.

A second limitation concerns the quality of the 3D observations. We use direct linear triangulation from the two most confident cameras as a fast, lightweight lifting step, and we zero the position of occluded joints. As Table 2 shows, the resulting 3D positions are imprecise, with an MPJPE of 12.94 to 21.31 cm, and the avatar network is trained on this noisy input. This choice explains part of the jitter increase observed in Table 3 (1.74 to 2.40 in Protocol 1 and 2.10 to 3.44 in Protocol 2), even though spatial and rotational accuracy improve. More robust lifting, such as confidence-weighted triangulation over all available views or a learned multiview lifting module, could provide cleaner inputs at a modest additional cost. The influence of the number and placement of cameras on accuracy also remains to be studied.

The calibration has its own limits. The cross-reprojection protocol (Section 5.2) is run within a single session and shows a residual drift of 0.81 cm on average, which comes from the headset’s visualinertial odometry. In this setting, the MRUK-based frame H and the native tracking space H give identical results, so the benefit of room-scan anchoring is not visible here. Its expected advantage is session invariance, which we assume but have not tested across application restarts, relocalization events, or longer sessions. Our implementation also depends on the Meta Quest 3 and MRUK, and a checkerboard is still needed once to register the HMD with the cameras. Evaluating the frame across sessions and adapting the procedure to other headsets are natural next steps.

Moreover, an important open question is how the remaining jitter and spatial inaccuracies affect user-centered metrics, such as the Sense of Embodiment (SoE) [37]. While related work highlights the importance of animation fidelity and control in SoE [51, 17], it remains to be determined whether the accuracy gains of our pipeline outweigh the added jitter for the user. Future work must therefore include subjective user evaluations to validate the perceptual benefit of these hybrid tracking methods.

Finally, providing lower-body 3D positions is compatible with a wide range of 3-point tracking architectures, so recent lightweight state-space models such as SSD-Poser [52] could be extended with this approach. A comprehensive benchmark against other sparse tracking methods, including DragPoser [33] and SparsePoser [35], would further contextualize our results. In addition, our system currently tracks a single user in a large scene. Extending it to multi-user environments, still without additional wearable lowerbody trackers, is a promising direction, and its main challenges are user (re-)identification and fast multi-user pose estimation from RGB videos.

Table 3: Full-body pose estimation accuracy on both protocols, comparing the 3-tracker baseline (HMD and two controllers) with our method, which adds external 3D pose observations. Position errors are reported for the full body (MPJPE), upper body (U-PE) and lower body (L-PE), together with the joint rotation error (MPJRE), joint velocity error (MPJVE) and jitter ratio. ↓: lower is better; best results in bold.
<table><tr><td></td><td colspan="3">Position error</td><td>Rotation</td><td>Velocity</td><td>Smoothness</td></tr><tr><td>Configuration</td><td>MPJPE↓ [cm]</td><td>U-PE↓ [cm]</td><td>L-PE↓ [cm]</td><td>MPJRE↓ [°]</td><td>MPJVE↓ [cm/s]</td><td>Jitter ↓ ratio</td></tr><tr><td>Protocol 1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3 trackers</td><td>3.67</td><td>1.57</td><td>6.71</td><td>5.43</td><td>17.79</td><td>1.74</td></tr><tr><td>3 trackers + 3D pose (ours)</td><td>2.35</td><td>1.23</td><td>3.97</td><td>4.62</td><td>16.20</td><td>2.40</td></tr><tr><td>Protocol 2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3 trackers</td><td>5.28</td><td>2.19</td><td>9.73</td><td>9.45</td><td>26.60</td><td>2.10</td></tr><tr><td>3 trackers + 3D pose (ours)</td><td>3.26</td><td>1.73</td><td>5.48</td><td>8.12</td><td>23.73</td><td>3.44</td></tr></table>

![](images/f550d5805ed1d343d6c0fbd01226373bfdce5905901bed03089e54e3b31a4ec9.jpg)  
Figure 8: Samples of full-body pose estimated from (top to bottom): 3 trackers; 3 trackers + Pose estimation; ground truth. Incorporating 3D positions from external tracking improves the of the full-body pose estimation, especially at lower-body parts.

## 8 CONCLUSION

This paper presented a framework for full-body self-avatar tracking in VR that fuses the sparse signals of an HMD and handheld controllers with 3D joint positions estimated from external multiview RGB cameras. We proposed a one-time calibration procedure that expresses both modalities in a common coordinate system through a single checkerboard snapshot and supports a variable number of cameras. We also introduced a deep learning model that estimates the full-body pose from these inputs, trained on noisy triangulated positions. The calibration shows a residual error of 0.81 cm on average, which is compatible with fusing RGB-based keypoints with HMD data. On AMASS, external 3D observations reduce the MPJPE by 36 % and 38 % on Protocols 1 and 2, and the lower-body position error by 41 % and 44 %, at the cost of a moderate increase in jitter. The complete pipeline runs end-to-end at 60 FPS on a consumer GPU. These results suggest that combining consumer-grade VR devices with outside-in RGB tracking is a practical way to improve avatar full-body pose estimation in largescale scenes, and that its perceptual benefit for the user should be examined in future work.

## REFERENCES

[1] K. Ahuja, E. Ofek, M. Gonzalez-Franco, C. Holz, and A. D. Wilson. CoolMoves: User Motion Accentuation in Virtual Reality. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 5(2):1–23, June 2021. doi: 10.1145/3463499 2

[2] AIIDE 2016. Motion Matching for Realistic Animation in “For Honor” (Simon Clavet), Oct. 2016. 2

[3] I. Akhter and M. J. Black. Pose-conditioned joint angle limits for 3D human pose reconstruction. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1446–1455, June 2015. ISSN: 1063-6919. doi: 10.1109/CVPR.2015.7298751 6

[4] S. Aliakbarian, P. Cameron, F. Bogo, A. Fitzgibbon, and T. J. Cashman. FLAG: Flow-based 3D Avatar Generation from Sparse Observations. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13243–13252. IEEE, New Orleans, LA, USA, June 2022. doi: 10.1109/CVPR52688.2022.01290 2

[5] S. Aliakbarian, F. Saleh, D. Collier, P. Cameron, and D. Cosker. HMD-NeMo: Online 3D Avatar Motion Generation From Sparse Observations. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9588–9597, Oct. 2023. Conference Name: 2023 IEEE/CVF International Conference on Computer Vision (ICCV)

![](images/ae26e4a22462a3054f0e29a7a13bdebf45fae44d87de3043745a13d4183e99f4.jpg)

![](images/4d480d75f6db8b2fbe692fc753c28f0684421b30fb9a70a50c08c2a9331c1e1d.jpg)  
Figure 9: Poses estimated by our proposed framework (top) and from baseline, relying on HMD and controllers solely (bottom). The green spheres represent the positions $p _ { J _ { e x t } } ^ { t }$ of body joints estimated from RGB videos (except for the axis system attached to the green sphere, which corresponds to $\mathcal { H } _ { f } )$ . Our model is able to follow the user’s motion and improves lower-body tracking compared to full-body pose estimation from sparse tracking alone.

ISBN: 9798350307184 Place: Paris, France Publisher: IEEE. doi: 10 .1109/ICCV51070.2023.00882 2

[6] R. Armani, C. Qian, J. Jiang, and C. Holz. Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24, pp. 1–11. ACM, Denver CO USA, July 2024. doi: 10.1145/ 3641519.3657465 2

[7] Y. Bao, X. Zhao, and D. Qian. FusePose: IMU-Vision Sensor Fusion in Kinematic Space for Parametric Human Pose Estimation, Aug. 2022. arXiv:2208.11960 [cs]. doi: 10.48550/arXiv.2208.11960 2

[8] P. Caserman, P. Achenbach, and S. Gobel. Analysis of Inverse Kine-¨ matics Solutions for Full-Body Reconstruction in Virtual Reality. In 2019 IEEE 7th International Conference on Serious Games and Applicationsfor Health (SeGAH), pp. 1–8, Aug. 2019. ISSN: 2573-3060. doi: 10.1109/SeGAH.2019.8882429 2

[9] A. Castillo, M. Escobar, G. Jeanneret, A. Pumarola, P. Arbelaez,´ A. Thabet, and A. Sanakoyeu. BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis. 2023. Publisher: [object Object] Version Number: 1. doi: 10.48550/ARXIV.2304.11118 2

[10] P. Dai, Y. Zhang, T. Liu, Z. Fan, T. Du, Z. Su, X. Zheng, and Z. Li. HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations, Mar. 2024. arXiv:2403.03561 [cs]. doi: 10.48550/arXiv.2403.03561 2, 3, 4, 6, 7

[11] X. Di, X. Dai, X. Zhang, and X. Chen. Dual Attention Poser: Dual Path Body Tracking Based on Attention. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 2795–2804. IEEE, Vancouver, BC, Canada, June 2023. doi: 10.1109/CVPRW59228.2023.00280 2

[12] A. Dittadi, S. Dziadzio, D. Cosker, B. Lundell, T. Cashman, and J. Shotton. Full-Body Motion from a Single Head-Mounted Device: Generating SMPL Poses from Partial Observations. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11667–11677. IEEE, Montreal, QC, Canada, Oct. 2021. doi: 10.1109/ ICCV48922.2021.01148 2

[13] Y. Du, R. Kips, A. Pumarola, S. Starke, A. Thabet, and A. Sanakoyeu. Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model. 2023. Publisher: [object Object] Version Number: 1. doi: 10.48550/ARXIV.2304.08577 2, 5

[14] M. A. Fischler and R. C. Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24(6):381–395, 1981. doi: 10.1145/358669.358692 4

[15] G. Fletcher, D. Egan, R. McDonnell, and D. Cosker. Improving motion matching for VR avatars by fusing inside-out tracking with outside-in pose estimation: MIG ’23: The 16th ACM SIGGRAPH Conference on Motion, Interaction and Games. Nov. 2023. 2

[16] A. C. C. for the Arts and Design. ACCAD MoCap Dataset. 6

[17] R. Fribourg, F. Argelaguet, A. Lecuyer, and L. Hoyet. Avatar and´ Sense of Embodiment: Studying the Relative Preference Between Appearance, Control and Point of View. IEEE Transactions on Visualization and Computer Graphics, 26(5):2062–2072, May 2020. doi: 10 .1109/TVCG.2020.2973077 2, 7

[18] S. Ghorbani, K. Mahdaviani, A. Thaler, K. Kording, D. J. Cook, G. Blohm, and N. F. Troje. MoVi: A large multi-purpose human motion and video dataset. PLOS ONE, 16(6):e0253157, June 2021. Publisher: Public Library of Science. doi: 10.1371/journal.pone.0253157 6

[19] E. Ha, G. Byeon, and S. Yu. Full-Body Motion Capture-Based Virtual Reality Multi-Remote Collaboration System. Applied Sciences, 12(12):5862, Jan. 2022. Number: 12 Publisher: Multidisciplinary Digital Publishing Institute. doi: 10.3390/app12125862 2

[20] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, Cambridge, 2 ed., 2004. doi: 10. 1017/CBO9780511811685 2, 4, 5

[21] J. Jiang, P. Streli, M. Meier, and C. Holz. EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, eds., Computer Vision – ECCV

2024, pp. 277–294. Springer Nature Switzerland, Cham, 2025. doi: 10.1007/978-3-031-72627-9 16 2

[22] J. Jiang, P. Streli, H. Qiu, A. Fender, L. Laich, P. Snape, and C. Holz. AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing. 2022. Publisher: [object Object] Version Number: 1. doi: 10.48550/ARXIV.2207.13784 2, 5

[23] Z. Li, S. K. Dwivedi, F. Maric, C. Chacon, N. Bertsch, F. Arcadu, T. Hodan, M. Ramamonjisoa, P. Wonka, A. Zhao, R. Kips, C. Keskin, A. Tkach, and C. Yang. EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR, Mar. 2026. arXiv:2603.04090 [cs]. doi: 10.48550/arXiv.2603.04090 2

[24] H. Liu, H. Ota, X. Wei, Y. Hirao, M. Perusqu´ıa-Hernandez,´ H. Uchiyama, and K. Kiyokawa. UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7085–7094, June 2025. ISSN: 2575-7075. doi: 10 .1109/CVPR52734.2025.00664 2

[25] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black. SMPL: a skinned multi-person linear model. ACM Trans. Graph., 34(6):248:1–248:16, Oct. 2015. doi: 10.1145/2816795.2818013 3

[26] E. J. C. Ltd. Eyes Japan Dataset. 6

[27] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. Black. AMASS: Archive of Motion Capture As Surface Shapes. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5441–5450. IEEE, Seoul, Korea (South), Oct. 2019. doi: 10.1109/ ICCV.2019.00554 3, 5, 6

[28] A. Maiorca, T. Ravet, G. Fletcher, and T. Dutoit. Robustness of Self-Avatar Animation Beyond Sparse Tracking: Effects of Pose Estimator Discrepancies and Inaccuracies. In 2025 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pp. 272–281, Oct. 2025. ISSN: 2473-0726. doi: 10.1109/ISMAR67309.2025.00039 2, 3

[29] C. Mandery, O. Terlemez, M. Do, N. Vahrenkamp, and T. Asfour. The KIT Whole-Body Human Motion Database. International Conference on Advanced Robotics (ICAR), pp. 329–336, 2015. 6

[30] M. Muller, T. Roder, M. Clausen, B. Eberhardt, B. Kruger, and A. Weber. Documentation Mocap Database HDM05. 5, 6

[31] S. F. U. a. N. U. of Singapore. SFU MOCAP. 6

[32] S. Pan, Q. Ma, X. Yi, W. Hu, X. Wang, X. Zhou, J. Li, and F. Xu. Fusing Monocular Images and Sparse IMU Signals for Real-time Human Motion Capture. In SIGGRAPH Asia 2023 Conference Papers, pp. 1– 11. ACM, Sydney NSW Australia, Dec. 2023. doi: 10.1145/3610548 .3618145 2

[33] J. L. Ponton, E. Pujol, A. Aristidou, C. Andujar, and N. Pelechano. DragPoser: Motion Reconstruction from Variable Sparse Tracking Signals via Latent Space Optimization. Computer Graphics Forum, 44(2):e70026, 2025. eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.70026. doi: 10.1111/cgf.70026 2, 8

[34] J. L. Ponton, H. Yun, C. Andujar, and N. Pelechano. Combining Motion Matching and Orientation Prediction to Animate Avatars for Consumer-Grade VR Devices. Computer Graphics Forum, 41(8):107–118, 2022. eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14628. doi: 10.1111/cgf.14628 2

[35] J. L. Ponton, H. Yun, A. Aristidou, C. Andujar, and N. Pelechano. SparsePoser: Real-time Full-body Motion Reconstruction from Sparse Data. ACM Trans. Graph., 43(1):5:1–5:14, Oct. 2023. doi: 10.1145/3625264 2, 8

[36] B. Qian, Z. Wei, J. Li, and X. Wei. ReliaAvatar: A Robust Real-Time Avatar Animator with Integrated Motion Prediction. In Proceedings ofthe Thirty-ThirdInternational Joint Conference on Artificial Intelligence, pp. 3124–3132. International Joint Conferences on Artificial Intelligence Organization, Jeju, South Korea, Aug. 2024. doi: 10. 24963/ijcai.2024/346 2

[37] D. Roth and M. E. Latoschik. Construction of the Virtual Embodiment Questionnaire (VEQ). IEEE Transactions on Visualization and Computer Graphics, 26(12):3546–3556, Dec. 2020. doi: 10.1109/TVCG. 2020.3023603 7

[38] L. Sigal, A. O. Balan, and M. J. Black. HumanEva: Synchronized

Video and Motion Capture Dataset and Baseline Algorithm for Evaluation of Articulated Human Motion. International Journal of Computer Vision, 87(1):4–27, Mar. 2010. doi: 10.1007/s11263-009-0273 -6 6

[39] L. N. Smith and N. Topin. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates, May 2018. arXiv:1708.07120 [cs]. doi: 10.48550/arXiv.1708.07120 6

[40] S. Starke, P. Starke, N. He, T. Komura, and Y. Ye. Categorical Codebook Matching for Embodied Character Controllers. ACM Trans. Graph., 43(4):142:1–142:14, July 2024. doi: 10.1145/3658209 2

[41] M. Tran, H. Mao, Q. Chen, and Y. Kim. Head2Body: Body Pose Generation from Multi-sensory Head-mounted Inputs. 2

[42] N. F. Troje. Decomposing biological motion: a framework for analysis and synthesis of human gait patterns. Journal ofVision, 2(5):371–387, 2002. doi: 10.1167/2.5.2 5, 6

[43] M. Trumble, A. Gilbert, C. Malleson, A. Hilton, and J. Collomosse. Total Capture: 3D Human Pose Estimation Fusing Video and Inertial Sensors, 2017. 6

[44] C. M. University. CMU Graphics Lab Motion Capture Database. 5, 6

[45] L. Wu and K. B. Chen. Marker-Based Real Time Motion Tracking for Sense of Virtual Body Ownership in Virtual Reality. In C. Stephanidis, M. Antona, S. Ntoa, and G. Salvendy, eds., HCI International 2024 – Late Breaking Posters, pp. 54–59. Springer Nature Switzerland, Cham, 2025. doi: 10.1007/978-3-031-78531-3 7 2

[46] Y. Xue, J. Jiang, R. Armani, D. Hollidt, Y.-C. Liao, and C. Holz. Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging, Oct. 2025. arXiv:2510.21654 [cs]. doi: 10.48550/arXiv.2510.21654 2

[47] C. Yang, A. Tkach, S. Hampali, L. Zhang, E. J. Crowley, and C. Keskin. EgoPoseFormer: A Simple Baseline for Stereo Egocentric 3D Human Pose Estimation. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, eds., Computer Vision – ECCV 2024, vol. 15113, pp. 401–417. Springer Nature Switzerland, Cham, 2025. Series Title: Lecture Notes in Computer Science. doi: 10.1007/ 978-3-031-73001-6 23 2

[48] D. Yang, D. Kim, and S.-H. Lee. LoBSTr: Real-time Lower-body Pose Prediction from Sparse Upper-body Tracking Signals. Computer Graphics Forum, 40(2):265–275, May 2021. arXiv:2103.01500 [cs]. doi: 10.1111/cgf.142631 2

[49] J. J. Yang, T. Chen, F. Qin, M. S. Lam, and J. A. Landay. HybridTrak: Adding Full-Body Tracking to VR Using an Off-the-Shelf Webcam. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, CHI ’22, pp. 1–13. Association for Computing Machinery, New York, NY, USA, Apr. 2022. doi: 10.1145/3491102. 3502045 2

[50] Y. Yang, S. Zhang, L. Ye, N. Rao, and X. Luo. PoseVR: Structure-Aware Hybrid Full-Body Pose Estimation in Virtual Reality. In Pattern Recognition and Computer Vision: 7th Chinese Conference, PRCV 2024, Urumqi, China, October 18–20, 2024, Proceedings, Part XI, pp. 533–548. Springer-Verlag, Berlin, Heidelberg, Nov. 2024. doi: 10.1007/978-981-97-8795-1 36 2

[51] H. Yun, J. L. Ponton, C. Andujar, and N. Pelechano. Animation Fidelity in Self-Avatars: Impact on User Performance and Sense of Agency. In 2023 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pp. 286–296. IEEE, Shanghai, China, Mar. 2023. doi: 10. 1109/VR55154.2023.00044 2, 7

[52] S. Zhao, L. Bai, L. Shao, Y. Zhang, and X. Chen. SSD-Poser: Avatar Pose Estimation with State Space Duality from Sparse Observations. In Proceedings of the 2025 International Conference on Multimedia Retrieval, ICMR ’25, pp. 1849–1857. Association for Computing Machinery, New York, NY, USA, June 2025. doi: 10.1145/3731715. 3733437 2, 8

[53] X. Zheng, Z. Su, C. Wen, Z. Xue, and X. Jin. Realistic Full-Body Tracking from Sparse Observations via Joint-Level Modeling, Aug. 2023. arXiv:2308.08855 [cs]. doi: 10.48550/arXiv.2308.08855 2, 5

[54] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li. On the Continuity of Rotation Representations in Neural Networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5738–5746, June 2019. ISSN: 2575-7075. doi: 10.1109/CVPR.2019. 00589 3