# MFVINS: Multiple Fisheye Camera-based Visual Inertial System

Eunseong Jang<sup>1</sup>, YuJin Chung<sup>1</sup>, Sang Jun Lee<sup>1</sup>, Jihyun Yoon<sup>2</sup>, and HyungGi Jo<sup>1</sup>

<sup>1</sup>Division of Electronic Engineering, Jeonbuk National University, Jeonju, South Korea <sup>2</sup>BSTAR Robotics, Inc., Palo Alto, CA, USA

## Abstract

A simultaneous localization and mapping (SLAM) method using a monocular camera and a low-cost inertial measurement unit (IMU) sensor is an efective way to fulfill a low-cost sensor configuration. Using this sensor configuration, visual-inertial system (VINS) focuses on fusing data from a camera and an IMU sensor to estimate the six degrees-of-freedom (DOF) of the sensor pose. Typically, VINS uses only a single camera as visual input, which lead to problems such as error accumulation due to occlusion, various illumination, and textureless environments. In this paper, we propose a new multiple fisheye camera-based visual-inertial system called MFVINS.

We present an IMU-aided FAST feature tracker for multiple cameras that enables eficient extraction and robust matching of local features. Then, the proposed method filters out outliers caused by fisheye distortion on the normalized image plane. Subsequently, a new reprojection error with physical validity constraints is proposed for bundle adjustment using learning-based depth estimation. The proposed method is applied to various scenarios, and its efectiveness is demonstrated by comparing previous VINS methods. In particular, MFVINS is implemented in real-time process to leverage the advantages of using multiple cameras—robustness against occlusion and textureless regions—while reducing the computational burden.

Keywords: Simultaneous Localization And Mapping; Visual-Inertial Odometry; Visual SLAM; Omnidirectional Camera

## 1 Introduction

Simultaneous Localization and Mapping (SLAM) Campos et al. [2021], Cho et al. [2021], Liu et al. [2025] is a core technology that enables robots and autonomous systems to localize themselves while building a map of the surrounding environment. Among various SLAM methods, the visual-inertial system (VINS) Qin et al. [2018] has garnered significant attention for its ability to fuse camera and inertial measurement unit (IMU) data to achieve accurate six degrees-of-freedom (DOF) pose estimation of a camera and maintain trajectory consistency. Most existing VINS solutions adopt a monocular pinhole camera due to its simplicity and cost-efectiveness. However, these systems often sufer from accumulated errors in challenging environments, such as those with occlusions, dynamic lighting conditions, or textureless surfaces. Crucially, using only images with a limited FOV will result in rapid performance degradation when occlusion occurs due to dynamic objects blocking the view.

To overcome these limitations, previous studies have explored the use of wide field-of-view (FoV) cameras and multi-camera setups He et al. [2022]. While fisheye cameras can capture a broader view of the environment and mitigate occlusion or limited texture issues, they also introduce significant image distortion, complicated feature tracking and depth estimation. Additionally, multi-camera systems inevitably increase computational complexity and often process cameras independently, without fully exploiting the geometric relationships between them. Existing multi-camera systems such as MCVIO He et al. [2022], a non-overlapping multi-camera system, handle cameras independently. This characteristic makes it dificult to accurately estimate the scale and depth of extracted features. Even when inter-camera feature matching is achieved, it often occurs in severely distorted areas, which limits accurate depth recovery.

In this paper, we propose a novel multiple fisheye camera-based visualinertial system (MFVINS) that addresses the aforementioned challenges by tightly integrating multiple fisheye cameras with a low-cost IMU. Unlike prior systems, MFVINS applies a learning-based depth estimation model to each fisheye image, enabling more precise 3D reconstruction even in visually degraded scenes.

In the frontend of MFVINS, a grid-based FAST algorithm is employed to extract features eficiently, and an IMU-aided feature tracker is implemented to robustly match local features across distorted fisheye images. IMU data can provide the initial locations of feature points, thus tracking eficiency has been enhanced. We also introduce a novel outlier filtering mechanism that identifies and removes mismatched features arising from heavy fisheye distortion by projecting them onto a normalized image plane. Also, deep learning-based depth estimation is exploited to incorporate depth values for subsequent localization. In the backend, loop closure detection and pose graph optimization are performed, significantly reducing long-term drift.

The proposed system is evaluated on various real-world sequences that include occlusions and low-texture environments. The results demonstrate that MFVINS outperforms existing monocular and multi-camera VINS frameworks in terms of both accuracy and robustness. Moreover, the system is implemented in real-time, showing that the use of multiple fisheye cameras can be practically viable without excessive computational overhead. The main contributions of this paper are as follows:

1. We propose MFVINS, a visual-inertial SLAM system based on multiple fisheye cameras, which improves robustness to occlusion and poor texture while reducing drift through frontend algorithms and backend optimization.

2. In the frontend, we develop an IMU-aided FAST feature tracking method along with a distortion-aware outlier filtering mechanism to enhance feature robustness under fisheye distortion. Also, an accurate local matching method is proposed by exploiting learning-based depth estimation.

3. We validate the system through extensive real-world experiments and ablation studies, demonstrating the efectiveness of each proposed module in improving pose estimation accuracy.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 provides an overview of MFVINS, including notation and coordinate frames. Section 4 details the proposed method and its key components. Section 5 presents the experimental setup, performance comparisons, and ablation analysis. Section 6 concludes and discusses potential future work.

## 2 Related Works

## 2.1 Mono/Stereo/RGB-D camera system

In VINS, various sensor configurations are employed. VINS-Mono Qin et al. [2018] improves localization performance by tightly coupling a monocular camera and an IMU, along with four-degree-of-freedom pose graph optimization. Similarly, ORB-SLAM3 Campos et al. [2021] integrates a monocular camera and an IMU and uses Oriented FAST and Rotated BRIEF (ORB) Rublee et al. [2011] features to enhance real-time performance. However, because a monocular camera has intrinsic limitations, such as scale ambiguity, subsequent research has investigated VIO systems that use stereo cameras. MSCKF-VIO Sun et al. [2018] tightly couples Forster et al. [2016] a stereo camera with an IMU, handling state estimation with a filter-based approach Mourikis and Roumeliotis [2007].

Beyond relying on parallax from stereo configurations, some VIO methods Ye et al. [2026], Wang et al. [2026] employ RGB-D cameras (e.g., structuredlight or Time-of-Flight sensors) to directly obtain accurate depth data or use learned depth images. VINS-RGBD Shan et al. [2019] extends VINS-Mono by obtaining the depth information of feature points from an RGB-D camera, enabling more precise localization. Dynamic-VINS Liu et al. [2022] incorporates object detection algorithms along with RGB-D images, and applies a grid-based eficient FAST feature extraction method to improve computational eficiency. Meanwhile, CodeVIO Zuo et al. [2021] incorporates depth images obtained from a learning-based network into conventional pinhole images for VIO.

<table><tr><td></td><td>VINS-Mono Qin et al. [2018]</td><td>VINS-RGBD Shan et al. [2019]</td><td>Dynamic-VINS Liu et al. [2022]</td><td>MCVIO He et al. [2022]</td><td>MFVINS(Ours)</td></tr><tr><td>Multicam (≥ 2 cameras)</td><td>X</td><td>X</td><td>X</td><td>0</td><td>o Shi-Tomasi</td></tr><tr><td>Feature extraction</td><td>Shi-Tomasi</td><td>Shi-Tomasi</td><td>FAST</td><td>Shi-Tomasi VPI-harris</td><td>VPI-harris</td></tr><tr><td>IMU aided feature tracker</td><td>X</td><td>X</td><td>0</td><td>X</td><td>FAST 0</td></tr><tr><td>Depth Image</td><td>X</td><td>0</td><td>0</td><td>X</td><td>0</td></tr><tr><td>PGO (loop closure)</td><td>0</td><td>0</td><td>0</td><td>X</td><td>0</td></tr><tr><td>Relocalize</td><td>0</td><td>0</td><td>0</td><td>X</td><td>0</td></tr><tr><td>Fisheye</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 1: Comparison of VINS-based systems.

## 2.2 Multiple camera system

Using multiple cameras can improve both the accuracy and robustness of VINS. BAMF-SLAM Zhang et al. [2023] utilizes a combination of monocular and stereo fisheye cameras and employs Recurrent Field Transforms (RFT) and Bundle Adjustment (BA) to enhance the accuracy of a VI-SLAM system. ROVINS Seok and Lim [2020] integrates an omnidirectional fisheye camera with an IMU to increase the robustness of pose estimation. OmniNxt Liu et al. [2024] also uses an omnidirectional fisheye camera, generating depth images via a CNN that employs cylindrical warping and stereo matching. These depth images are then used for more accurate localization. However, since the experiments were limited to relatively small environments, the scalability to large-scale scenarios remains unverified. Furthermore, many such systems like MAVIS Wang et al. [2024] with a front stereo sensor and side monocular cameras require specific sensor configurations.

![](images/0e94a1cfba6482dead02f515c5d5941796a518c2a52a58f64a4b571813c90608.jpg)  
Figure 1: Framework of the proposed Multiple Fisheye Visual Inertial Systems (MFVINS). The upgraded modules are highlighted in red.

Table 1 summarizes the key functional comparisons between the proposed MFVINS and representative visual-inertial systems (He et al. [2022], Qin et al. [2018], Shan et al. [2019], Liu et al. [2022]). These existing methods employ diverse sensor configurations, including a single camera, RGB-D sensors, and multi-camera setups, and difer in aspects such as feature extraction algorithms (e.g., Shi-TomasiHarris et al. [1988], FASTViswanathan [2009]), the use of IMU-based tracking, and the incorporation of depth information. While MCVIO He et al. [2022] supports a multi-camera arrangement, it processes the features from each camera independently, thus failing to fully leverage the geomteric information between cameras and limiting feature depth estimation. Furthermore, neither the publicly available code nor the experiments presented in the paper include loop closure or pose graph optimization to correct cumulative errors, making the system prone to drift over extended tracking periods.

## 3 Overview

## 3.1 Proposed Frameworks

The proposed localization framework builds upon the foundational structures of VINS-Mono Qin et al. [2018] and MCVIO He et al. [2022]. As illustrated in the block diagram of Fig. 1, the system is composed of four primary components: 1) Measurement Preprocessing, 2) Fisheye depth estimation, 3) Initialization, 4) Multi-Cam VIO and Pose Graph Optimization (PGO) Carlone et al. [2015]. The core contributions of our method are highlighted in red within the Fig. 1.

## 3.1.1 Measurement Preprocessing

In this stage, images collected from multiple fisheye cameras are processed to extract features using the FAST Viswanathan [2009] algorithm. Simultaneously, angular velocity measurements $\omega ^ { b }$ obtained from the IMU are used to enhance tracking. Due to potential discrepancies in observation timing between the camera and the IMU, a temporal ofset $\Delta t _ { c a m - i m u }$ may arise, which is estimated and compensated through calibration for accurate sensor synchronization. By computing a predicted rotation matrix $\mathbf { R } _ { p r e d }$ from the calibrated data, the eficiency of feature tracking is significantly improved. Furthermore, feature outliers caused by distortion of fisheye lens are filtered, resulting in a refined set of features $\mathbf { p } _ { i }$ that contributes to the reliability and accuracy of downstream processes.

## 3.1.2 Fisheye Depth Estimation

In this module, fisheye images are passed through a deep learning-based depth estimation network to obtain either pixel-wise depth maps D or inverse depth values λ . The resulting depth information is then associated with each extracted feature from the preprocessing stage, providing essential geometric constraints for accurate localization.

## 3.1.3 Initialization

For system initialization, a structure-from-motion (SfM) Schonberger and Frahm [2016] technique is applied using primarily the front-facing camera. This step establishes the initial position and orientation of the robot (or camera) and estimates the necessary parameters for initializing visual-inertial odometry (VIO).

## 3.1.4 Multi-Cam VIO and Pose Graph Optimization (PGO)

During this phase, visual-inertial odometry is performed using measurements from the multi-fisheye camera setup. The state of the system is updated using a sliding-window optimization strategy. To mitigate drift errors accumulated over time, loop closure detection and pose graph optimization are applied using the front camera, which consistently observes the forward motion direction. This approach enhances both localization accuracy and long-term stability.

## 3.2 Notations and frame definitions

To maintain consistency with VINS-Mono Qin et al. [2018], the notations and coordinate frames used in this study follow the same conventions. We define the world frame $( \cdot ) ^ { w }$ , camera frame $( \cdot ) ^ { c }$ , IMU body frame $( \cdot ) ^ { b }$ , and an additional LiDAR frame $( \cdot ) ^ { l }$ . The transformation between these coordinate frames is represented by a 4×4 matrix in the SE(3) group, defined as

$$
\mathbf { T } = \left[ \begin{array} { l l } { \mathbf { R } } & { \mathbf { t } } \\ { \mathbf { 0 } ^ { T } } & { 1 } \end{array} \right] \in \mathrm { S E } ( 3 )\tag{1}
$$

where $\mathbf { t } \in \mathbb { R } ^ { 3 }$ is the translation vector, and $\mathbf { R } \in \mathrm { S O } ( 3 )$ is the rotation matrix, which can also be represented as a rotation vector or quaternion as

$$
\mathbf { r } = { \left( \left[ \begin{array} { l } { r _ { 1 } } \\ { r _ { 2 } } \\ { r _ { 3 } } \end{array} \right] \right) } _ { \times } = { \left[ \begin{array} { l l l } { \ 0 } & { \ - r _ { 3 } } & { \ r _ { 2 } } \\ { r _ { 3 } } & { \ 0 } & { \ - r _ { 1 } } \\ { - r _ { 2 } } & { r _ { 1 } } & { \ 0 } \end{array} \right] } \in { \mathfrak { s o } } ( 3 )\tag{2}
$$

Based on these coordinate settings, we define two projection functions, $\pi _ { l } ( \cdot ) : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ and $\pi _ { 0 } ( \cdot ) : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ , to map a 3D point $\mathbf { P } ^ { w } \in \mathbb { R } ^ { 3 }$ in the world frame to the 2D image plane. $\pi _ { l } \left( \cdot \right)$ corresponds to projection through a fisheye camera model Kannala and Brandt [2006], while $\pi _ { 0 } \left( \cdot \right)$ uses a unit sphere projection. Let $\mathbf { p } _ { n } = [ u ^ { \prime } , v ^ { \prime } ] ^ { T }$ be the projected 2D point on the normalized image plane corresponding to a 3D point $\mathbf { P } ^ { w }$ . The radial distance from the camera center to $( u ^ { \prime } , v ^ { \prime } )$ is denoted as $r ^ { \prime } = \sqrt { u ^ { \prime } { } ^ { 2 } + v ^ { \prime } { } ^ { 2 } }$ The incident angle in the normalized coordinate system is defined as $\theta ,$ and the distorted incident angle that accounts for radial distortion is denoted as $\theta _ { d }$ . The distortion is modeled using radial distortion coeficients $k _ { 1 } , k _ { 2 } , \cdots$ Under the fisheye camera model, the projection function $\pi _ { l } \left( \cdot \right)$ that maps a 3D point onto the image plane is given by

$$
\theta _ { d } = \theta + k _ { 1 } \theta ^ { 3 } + k _ { 2 } \theta ^ { 5 } + \cdot \cdot \cdot\tag{3}
$$

$$
\pi _ { l } \left( \mathbf { P } \right) = { \left[ \begin{array} { l } { f _ { x } \cdot \theta _ { d } \left( { \frac { u ^ { \prime } } { r ^ { \prime } } } \right) + c _ { x } } \\ { f _ { y } \cdot \theta _ { d } \left( { \frac { v ^ { \prime } } { r ^ { \prime } } } \right) + c _ { y } } \end{array} \right] }\tag{4}
$$

where $f _ { x } , f _ { y }$ are the focal lengths, and $c _ { x } , c _ { y }$ is the principal point of the camera.

![](images/db2a4746f14d95469f14240cb320f74de1e4d75856a1c60c6bf237873d1fbddd.jpg)  
(a)

![](images/2c290e3d2659ed67813f6b0b7cf3c43ceee0db3fec8bff5b3db20796d55ad845.jpg)  
(b)  
Figure 2: An example of feature detection and tracking (Blue: detected, Red: tracked, Green: predicted features) (a) MCVIO using VPI-Harris, (b) proposed method.

In the case of directly projecting a 3D point ${ \bf P } = [ P _ { x } \ P _ { y } \ P _ { z } ] ^ { T }$ located on the unit sphere, the incident angle $\begin{array} { r } { \theta = \cos ^ { - 1 } \left( \frac { P _ { z } } { \| \mathbf { P } \| } \right) } \end{array}$ and azimuth angle $\phi =$ tan<sup>−</sup> $^ { - 1 } \left( { \frac { P _ { y } } { P _ { x } } } \right)$ are first computed. Similar to the previous model, a distorted incident angle $\theta _ { d }$ is applied to account for radial distortion using the same distortion model. The projection function $\pi _ { 0 } \left( \cdot \right)$ , which maps a unit-sphere point to the image plane, is defined as follows

$$
\pi _ { 0 } \left( \mathbf { P } \right) = \left[ { \begin{array} { c } { f _ { x } \cdot \theta _ { d } \cdot \cos \phi + c _ { x } } \\ { f _ { y } \cdot \theta _ { d } \cdot \sin \phi + c _ { y } } \end{array} } \right] .\tag{5}
$$

The acceleration $\hat { \mathbf { a } } _ { t } ^ { b }$ and angular velocity $\hat { \omega } _ { t } ^ { b }$ measured by the IMU at time t are expressed as

$$
\begin{array} { r l } & { \hat { \mathbf { a } } _ { t } ^ { b } = \mathbf { a } _ { t } ^ { b } + \mathbf { b } _ { a { t } } + \mathbf { R } _ { w } ^ { t } \mathbf { g } ^ { w } + \mathbf { n } _ { a } } \\ & { \hat { \mathbf { \omega } } _ { t } ^ { b } = \omega _ { t } + \mathbf { b } _ { \omega _ { t } } + \mathbf { n } _ { \omega } } \end{array}\tag{6}
$$

where $\mathbf { b } _ { a _ { t } }$ and $\mathbf { b } _ { \omega _ { t } }$ are the accelerometer and gyroscope biases, respectively, and $\mathbf { n } _ { a } , \mathbf { n } _ { \omega }$ denote Gaussian white noise. To handle these biases and noise components during state estimation, T. Qin, et al. Qin et al. [2018] proposed an IMU preintegration method. In this paper, we adopt the notational conventions from Y. He, et al. He et al. [2022].

![](images/2c42f6f0718878b350dfdb421cd4f7f5f7ff3bd4e025c7f00813715a469a3994.jpg)  
Figure 3: Predicting feature point locations using a rotation matrix

## 4 Multiple Fisheye VINS(MFVINS)

## 4.1 IMU-aided FAST feature tracker

## 4.1.1 Visual Feature Extraction

Fisheye cameras ofer a significantly wider field of view (FoV) compared to conventional pinhole cameras, resulting in the detection of a larger number of visual features. In multi-fisheye camera systems, this number increases further, posing computational challenges. Therefore, lightweight and eficient feature detection becomes crucial for real-time performance.

Our proposed method is modified version of an IMU-aided FAST feature tracking method proposed in Liu et al. [2022], tailoring it for multi-fisheye camera configurations. In our implementation, the image is divided into an N×M grids, and FAST features are uniformly extracted within each grid cell to ensure even spatial coverage, as illustrated in Fig. 2(b). Parallel threading is also used to accelerate computation. This process is independently applied to each fisheye camera, enabling scalable and eficient multi-camera feature extraction suitable for real-time SLAM operations.

## 4.1.2 IMU-aided Tracking

Following the feature extraction step, feature tracking is typically performed using the Lucas-Kanade (LK) optical flow algorithm. However, fisheye images inherently exhibit strong radial distortion and often contain a number of features, which can degrade the accuracy and eficiency of conventional feature matching techniques.

Firstly, a key challenge in integrating IMU and camera data is the presence of temporal misalignment due to difering sensor acquisition time. To mitigate this, our method estimates and compensates for the temporal ofset $\Delta t _ { c a m - i m u }$ between the IMU and each camera. Unlike Liu et al. [2022], which only considers the ofset in the current frame, our method corrects the ofset for both the current frame $b _ { k + 1 }$ and previous frames $b _ { k }$ to improve the precision of the synchronization.

Using the compensated angular velocity readings from the IMU, the rotation between two consecutive frames is estimated. The 2D features $\mathbf { p } _ { c _ { i } } = [ u _ { j } , v _ { j } ] _ { j = 1 : N _ { p } } ^ { T }$ are back-projected onto the unit sphere $\mathbf { P } _ { c _ { i } }$ using the fisheye model, recovering their 3D directions. In the fisheye camera model, from azimuth $\begin{array} { r } { \phi = \tan ^ { - 1 } \left( \frac { v ^ { \prime } } { u ^ { \prime } } \right) } \end{array}$ and distance $r ^ { \prime } = \sqrt { u ^ { \prime } { } ^ { 2 } + v ^ { \prime } { } ^ { 2 } }$ between the camera origin and $( u ^ { \prime } , v ^ { \prime } )$ , thus $\begin{array} { r } { ( u ^ { \prime } , v ^ { \prime } ) = \Big ( \frac { \theta _ { d } } { r } u , \frac { \theta _ { d } } { r } v \Big ) } \end{array}$ . Therefore, $r ^ { \prime }$ exactly corresponds to the distorted incidence angle theta, and (3) can be re-written as follows

$$
\theta ( 1 + k _ { 1 } \theta ^ { 2 } + k _ { 2 } \theta ^ { 4 } + k _ { 3 } \theta ^ { 6 } + k _ { 4 } \theta ^ { 8 } ) - r ^ { \prime } = 0 .\tag{7}
$$

Then, the undistorted incident angle θ is computed by solving a monic polynomial (7) whose roots correspond to the true angle of incidence. To solve (7), the companion matrix is defined as

$$
C = \left[ \begin{array} { l l l l l l l } { 0 } & { 0 } & { 0 } & { \cdots } & { 0 } & { 0 } & { r ^ { \prime } } \\ { 1 } & { 0 } & { 0 } & { \cdots } & { 0 } & { 0 } & { - 1 } \\ { 0 } & { 1 } & { 0 } & { \cdots } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 1 } & { \cdots } & { 0 } & { 0 } & { - k _ { 1 } } \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } & { \vdots } \\ { 0 } & { 0 } & { 0 } & { \cdots } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { \cdots } & { 0 } & { 1 } & { - k _ { 4 } } \end{array} \right] \in \mathbb { R } ^ { 1 0 \times 1 0 } .\tag{8}
$$

The roots of (7) are obtained by computing the eigenvalues of $C .$ . We apply Schur decomposition Wigderson [2019] to the C as

$$
{ \cal C } = T U T ^ { - 1 } ,\tag{9}
$$

where $T$ is unitary matrix and U is upper triangular matrix. Then, the diagonal values of $U$ are eigenvalues, and the positive values are selected from the real eigenvalues with suficiently small imaginary parts, and the smallest value among them is determined as the final restored θ. Using the recovered incident angle θ and the computed azimuth angle $\phi ,$ the corresponding 3D point on the unit sphere can be reconstructed as

$$
\mathbf { P } _ { c _ { i } } ^ { c } = \pi _ { 0 } ^ { - 1 } \left( \mathbf { p } _ { c _ { i } } \right) = \left[ \begin{array} { c } { \sin \theta \cos \phi } \\ { \sin \theta \sin \phi } \\ { \cos \theta } \end{array} \right] .\tag{10}
$$

The unit vector $\mathbf { P } _ { c _ { i } } ^ { c }$ represents the direction of the visual feature relative to the i-th camera center, compensating for radial distortion introduced by the fisheye lens.

Next, to estimate the camera motion between two consecutive frames $b _ { k }$ and $b _ { k + 1 }$ , we integrate the angular velocity sequence $\omega _ { t } ^ { b }$ measured by the IMU over the time interval $[ t _ { k } , t _ { k + 1 } ]$ . The incremental rotation $\delta ,$ represented as an angle-axis vector, is computed as

$$
\pmb { \delta } = \int _ { t _ { k } } ^ { t _ { k + 1 } } \Big ( \omega _ { t } ^ { b } - \mathbf { b } _ { g } \Big ) d t .\tag{11}
$$

In this work, trapezoidal integration is used for improved numerical accuracy, computing the rotation as

$$
\delta \approx \sum _ { i = 1 } ^ { N - 1 } \frac { \Delta t _ { i } } { 2 } \left[ \omega _ { i } - \omega _ { i + 1 } \right] - \mathbf { b } _ { g } .\tag{12}
$$

The resulting rotation vector is then converted into a rotation matrix $\mathbf { R } _ { p r e d }$

$$
\mathbf { R } _ { p r e d } = \prod _ { l } \exp \Big ( { e x p \big ( { e x t _ { \mathbf { R } _ { c _ { i } } ^ { b } } ^ { T } [ \delta _ { l } ] } _ { \times } \Big ) } .\tag{13}
$$

where $e x t _ { \mathbf { R } _ { c _ { i } } ^ { b } } ^ { T }$ represents transformation between IMU and i-th camera $c _ { i } , i \in$ {front, rear, lef t, right} and ${ \mathbf { b } } _ { g }$ is bias of IMU gyroscope. (13) explains that the accumulated rotation vector obtained by integrating the IMU angular velocity over time is converted into a rotation matrix through the exponential map exp(·) Ma et al. [2004]. These rotated feature directions are reprojected onto the image plane using the fisheye projection function.

In summary, as illustrated in Fig. 3, each feature point is back-projected to a 3D unit sphere coordinate $\mathbf { P } _ { c _ { i } } ^ { c } = \pi _ { 0 } ^ { - 1 } ( \mathbf { p } _ { c _ { i } } )$ using (10), assuming a unit distance. After that, the estimated rotation amount $\mathbf { R } _ { p r e d }$ is reflected by applying $\hat { \mathbf { P } } _ { c _ { i } } ^ { c } = \mathbf { R } _ { p r e d } \mathbf { P } _ { c _ { i } } ^ { c }$ , and then converted to 2D image coordinates $\hat { \mathbf { p } } _ { c _ { i + 1 } }$ through the projection function $\pi _ { 0 } ( \hat { \mathbf { P } } _ { c _ { i } } ^ { c } )$ The calculated position is used as the initial position of the LK optical flow Lucas and Kanade [1981], contributing to improving both the accuracy and convergence rate of feature matching, especially under large rotations and fisheye distortion.

## 4.1.3 Outlier Filtering for Fisheye-Induced Distortion

Visual features are extracted and tracked using the Kanade-Lucas-Tomasi (KLT) sparse optical flow algorithm Zhou et al. [2025]. To remove mismatched correspondences of features, RANSAC Matas and Chum [2004b] and the fundamental matrix Matas and Chum [2004a] are used during the preprocessing stage, along with distortion compensation techniques. While these methods are efective under mild distortion, they become less reliable when applied to fisheye images with non-square aspect ratios (e.g., 16:9), as opposed to 1:1 image commonly used in datasets such as TUM VI Schubert et al. [2018].

Fig. 4 compares feature tracking results from our 16:9 fisheye camera setup and the TUM VI dataset. Fig. 4(a) and 4(b) show tracked feature points, Fig. $4 ( \mathrm { c } )$ and $5 4 ( \mathrm { d } )$ present distortion-corrected images, and Fig. $4 ( \mathrm { e } )$ and $4 ( \mathrm { f } )$ display the distribution of tracked features on the normalized image plane. In the TUM VI case (1:1 aspect ratio), features are concentrated near the center and symmetrically distributed, with all tracked points falling within the $1 0 \times 1 0$ grid boundary, as shown in Fig. 4(f). However, for our 16:9 images, fisheye distortion causes a horizontal spreading of feature points, resulting in outliers that extend beyond the normalized boundary, as clearly visible in Fig. $4 ( \mathrm { c } )$ and $4 ( \mathrm { e } )$ . These observations highlight the need for a more robust outlier rejection method that can efectively handle distortion-specific artifacts, particularly when using wide FoV fisheye cameras with non-standard aspect ratios.

To address this issue, a distortion-aware outlier filtering method is proposed that operates on the normalized image plane. Let $\mathbf { p } _ { n , j } \in \mathbb { R } ^ { 2 }$ denote the normalized 2D coordinates of the j-th tracked feature in a given frame. For each tracked feature, we compute its Euclidean norm $\Vert \mathbf { p } _ { n . j } \Vert _ { 2 }$ to assess its deviation from the center of the normalized plane.

A threshold τ is defined to eliminate feature points that exhibit excessive radial displacement due to fisheye distortion. In our implementation, we empirically set $\tau = 6$ . The filtering condition is defined as

$$
\begin{array} { r } { \mathbf { p } _ { f } = \{ \mathbf { p } _ { n . j }   \mathbf { p } _ { n . j }  _ { 2 } > \tau , j = 1 , \ldots , N _ { p } \} . } \end{array}
$$

Applying this filter efectively removes features located in the diverging peripheral regions of the distorted image, particularly in wide aspect ratio fisheye frames. As shown in Fig. 5, this method eliminates horizontally diverging outliers while preserving features within the valid central region of the normalized plane.

![](images/bad7bf9fb74f2c11c9e8cb87f3b701d7f5bbcbe79c50968c91d2df1f22ee5c0a.jpg)  
(a)

![](images/48c274dc8a675a24a49a84a3e0e07ecceea88d6b10d758bf4718aecf4fd7366f.jpg)  
(b)

![](images/30b5aa187452894e8f6452199278a395b5c7b5fc80f896ac161ce3c7cf03749e.jpg)  
(c)

![](images/531e219cae93d47ed8332c6d40e2db82838935aa09ad0b8e992696395062faf6.jpg)  
(d)

![](images/1f11e1e3f6b830338264dc66f1a280040a336bf7260f1c8b0395f0902b9ae532.jpg)  
(e)

![](images/bde7367dcbd8f7475c07e77eca0ed97247b0ae7927f739daca5c48d6b1c5f1d0.jpg)  
(f)  
Figure 4: Comparison of feature distributions on the normalized image plane between TUM VI and our 16:9 fisheye dataset. (a), (b): Tracked feature points on the raw fisheye images. (c), (d): Distortion-corrected fisheye images. (e), (f): Distribution of tracked features projected onto the normalized plane.

![](images/81c78c84cc2aa5187612ed17438db3543a93f551edfed5c5ce05a63fbbd6073f.jpg)  
Figure 5: Distortion-corrected image after applying the proposed filtering method

## 4.1.4 Learning-based Depth Estimation and Multi-Camera Visual Inertial Odometry

While RGB-D cameras provide direct depth measurements, they often suffer from limitations in large-scale environments Darwish et al. [2019]. Most RGB-D sensors are designed for short-range pinhole configurations, resulting in decreased depth accuracy at longer distances and poor compatibility with wide-FoV lenses. Similarly, traditional stereo matching methods encounter dificulties under fisheye distortion, leading to unreliable triangulation results.

To address these challenges, a learning-based depth estimation module tailored to our multi-fisheye camera setup is implemented. Specifically, LiDAR-generated point clouds are projected onto each fisheye image using the camera model $\pi _ { l }$ as shown in Fig. 6(a). Then, training data can be generated that reflects lens distortion. Using projected point clouds, the monocular depth estimation model Lee et al. [2019] is fine-tuned, initially trained on the KITTI dataset Geiger et al. [2012]. This model outputs a dense inverse depth map $\mathbf { D } _ { i } .$ , as illustrated in Fig. 6(b).

The predicted inverse depth $\hat { \lambda } _ { i , j }$ of the features $\mathbf { p } _ { f } = \left[ u _ { j } , v _ { j } \right] _ { j = 1 : N _ { p } } ^ { T }$ is selectively incorporated into the VIO pipeline. The learning-based depth map is adopted as an initialization if it satisfies physical validity constraints.

Otherwise, conventional triangulation is used as

$$
\hat { \lambda } _ { i , j } = \left\{ \begin{array} { l l } { \mathbf { D } _ { i } \left( u _ { j } , v _ { j } \right) , } & { \mathrm { i f ~ } 0 < \mathbf { D } _ { i } \left( u _ { j } , v _ { j } \right) < \delta } \\ { \lambda _ { i , j } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{14}
$$

Then, the depth of feature is refined through optimization using the Singular Value Decomposition (SVD) approach to minimize reprojection errors,

$$
\begin{array} { r l } & { \hat { \lambda } _ { i , j } \mathbf { p } _ { f , j + 1 } = \mathbf { R } _ { j } ^ { j + 1 } \left( \hat { \lambda } _ { i , j } \mathbf { p } _ { f , j } \right) + \mathbf { t } + \mathbf { e } _ { j } } \\ & { \sum \mathbf { e } _ { j } ^ { 2 } = \displaystyle \sum _ { j = 1 } ^ { m _ { i } } \left\| \hat { \lambda } _ { i , j } \mathbf { p } _ { f , j + 1 } - \mathbf { R } _ { j } ^ { j + 1 } \left( \hat { \lambda } _ { i , j } \mathbf { p } _ { f , j } \right) - \mathbf { t } \right\| ^ { 2 } . } \end{array}\tag{15}
$$

This scheme reduces the risk of optimization contamination in sections where deep learning results produce excessively large errors, while actively utilizing depth information in areas with high prediction stability to compensate for depth estimation uncertainty that tends to occur in wide-angle (fish-eye) camera environments. As a result, this system presents an integrated approach that improves accuracy and stability by incorporating deep learning.

## 4.2 Pose Graph Optimization for Drift Correction

To mitigate drift accumulated during visual-inertial odometry, we adopt a back-end pipeline similar to that of VINS-Mono Qin et al. [2018]. Specifically, loop closure detection is performed using DBoW2 G´alvez-L´opez and Tardos [2012] on images captured from the front-facing camera. When a loop is detected, relocalization is triggered to correct accumulated pose errors. Following loop detection, pose graph optimization (PGO) is applied to enforce global consistency across the trajectory. This optimization step reduces long-term drift by aligning revisited locations while preserving local odometry accuracy. In our system, only the front camera is used for loop closure detection and pose graph optimization. This method not only increases the probability of loop detection by stably observing the scene in the driving direction, but also reduces computational complexity, thereby securing real-time performance.

## 5 Experiments

In this paper, to evaluate the performance of localization in a large-scale indoor environment, we conducted experiments in the underground parking lot. Three diferent scenarios were considered: DRIVING, OCCLUSION, and ROTATIONAL. We used a two-wheeled UGV platform, shown in Figs. 7(b) and (c). Four fisheye cameras with a 183° FoV are installed to gather omnidirectional view, along with an IMU (Xsens MTi-100). A 3D LiDAR (Ouster OS0-32) is attached to acquire ground-truth for evaluation. To compare performance with a pinhole camera setup, we also installed a RGB-D sensor (Intel RealSense D455). All sensors were operated on an embedded board (NVIDIA Jetson AGX Xavier). As illustrated in Fig. 8, the test environment presents significant operational challenges, characterized by frequent occlusions in the front-facing camera and textureless surfaces (e.g., plain walls in the left view) that typically cause feature tracking failures in standard VINS. Please watch the video clip provided as supplemental material.

![](images/702208990e90804891ca7f641a406f85cba92af96310c54af649e56901352683.jpg)

(a)  
![](images/ee4dbbe11419742ea202db7ff64155a40744d1f7e7ab1f099050e947e9fc4a8a.jpg)  
(b)  
Figure 6: (a) Fisheye image with projected point cloud, (b) Deep learningbased depth image

![](images/701a9407b41ad35e10fd9f110ed75961725e91f42e53b79ec93eaf0632aa0bba.jpg)  
(a)

![](images/c3c0d17408db6f78c9a49a108f6d1e9f78849bd2777a7606e4d67fa8d46dc357.jpg)  
(c)  
Figure 7: (a) 3D point cloud map of underground parking lot where experiments conducted; (b), (c) Side and top views of the mobile robot platform equipped with multiple sensors, respectively.

![](images/e05da0261fa31eafc201534ba9368374a9e51942e9e8a6907c098a1fad33dd3b.jpg)  
Figure 8: A sample data in an underground parking lot. The top panel shows multi-camera sensor data captured from the robot platform, highlighting challenging conditions such as occlusion (front) and textureless surfaces (left). The bottom panel illustrates the generated 3D point cloud map and the estimated 6-degree-of-freedom (6-DoF) pose trajectory (red) aligned with the reference point cloud (gray).<sub>18</sub>

## 5.1 Evaluation Metrics

To analyze the performance of the proposed method, we use the Root Mean Square Error (RMSE), which is widely used in evaluations of localization algorithms. RMSE is used as an Absolute Trajectory Error (ATE). As reference, we use the 6-DoF pose of a LiDAR-based localization algorithm Bai et al. [2022] as ground-truth, aligning the trajectory of the visual-inertial system with this reference to evaluate accuracy.

Ablation studies were conducted to evaluate the performance of each component of the proposed method. In this process, both the RMSE and several feature-related metrics were assessed. The evaluated metrics include feature detection time, the number of untracked features, and the average number of frames per tracked feature. Feature detection time refers to the average time required to detect features between consecutive frames. The number of untracked features indicates the number of features that were detected but not successfully tracked during a 100-second period. Lastly, the average number of frames per tracked feature represents the number of frames during which each successfully tracked feature was maintained on average.

## 5.2 Experiments on DRIVING Sequences

To assess the performance of the proposed method in typical driving scenarios, we collected five diferent sequences and compared with other previous methods. For the monocular camera-based algorithm (VINS-MONO), we used diferent types of lenses to compare performance. For multiple camerabased algorithms (MCVIO and MFVINS), we tested setups using only front and rear cameras, as well as an omnidirectional camera configuration, to examine how the number of cameras afects performance. We also performed an experiment to investigate the efect of using a learning-based depth estimation method. When learning-based depth estimation was applied, we refer to the results as MFVINS(D). The quantitative results of these experiments are shown in Table 2. It can be seen that our proposed method achieves the most accurate localization. Using an omnidirectional camera configuration yields better results than using only front and rear cameras. Fig. 9(a) presents qualitative results. For clarity, only results with a pinhole lens are shown for the monocular camera-based method (VINS-MONO), and only the omnidirectional camera configuration is shown for the multiple camera-based methods (MCVIO, MFVINS).

Case 1  
![](images/e306556bc1fb73111bdb0e40374263989cae2e45b0730a296de99fb50df2683b.jpg)

![](images/675b4c19afb57aabf067469cc6e54f0b24d0a24f91cced944fd886596b23795a.jpg)

Case 3  
![](images/98005692f723ce930dfbad183d30ec873f9d85bd21f490589ce66ab246c98d67.jpg)

![](images/7acbc011796df46c4be8471295e0768a1a298a9bd338259c6abd777b86430232.jpg)  
(a)

Case 5  
![](images/5249a9ca1be7f97980eab8face4e655b81410a798d462bf278a392d10ee008be.jpg)

Case 1  
![](images/0908a9df085b474d8b84f0b6d4e5c9fbee809300e82d732c79acd516ee792b34.jpg)

Case 2  
![](images/f6755df7788acf08dbabb6849b2ae9647fa6b5ad6bf28ef2026e588cbe899e13.jpg)  
(b)  
Figure 9: (a) and (b) show qualitative results of trajectory estimation on the DRIVING and OCCLUSION sequences, respectively. The dashed line denotes the ground-truth trajectory. The blue represents VINS-Mono, green represents MCVIO, red represents MFVINS, and purple represents MFVINS(D).

<table><tr><td rowspan="2" colspan="2">Sequence</td><td colspan="6">DRIVING</td><td colspan="3">OCCLUSION</td></tr><tr><td>Case 1</td><td>Case 2</td><td>Case 3</td><td>Case 4</td><td>Case 5</td><td>Mean</td><td>Case 1</td><td>Case 2</td><td>Mean</td></tr><tr><td colspan="2">Total Length [m]</td><td>136.391</td><td>295.630</td><td>232.400</td><td>196.798</td><td>345.779</td><td>241.400</td><td>192.586</td><td>132.192</td><td>162.389</td></tr><tr><td>Method</td><td>Configuration</td><td colspan="9">RMSE of ATE [m]</td></tr><tr><td>VINS-MONO</td><td>Front(pinhole)</td><td>1.0525</td><td>1.4983</td><td>0.9452</td><td>2.1965</td><td>1.2426</td><td>1.3870</td><td>1.9443</td><td>1.5391</td><td>1.7417</td></tr><tr><td>VINS-MONO</td><td>Front</td><td>1.3605</td><td>1.1905</td><td>4.3665</td><td>2.5488</td><td>3.1643</td><td>2.5261</td><td>4.9033</td><td>3.7529</td><td>4.3281</td></tr><tr><td>MCVIO</td><td>Front,Rear</td><td>1.8731</td><td>2.4889</td><td>1.8822</td><td>3.6962</td><td>1.7403</td><td>2.3361</td><td>2.1654</td><td>1.6217</td><td>1.8936</td></tr><tr><td>MCVIO</td><td>Omnidirectional</td><td>1.7298</td><td>3.0179</td><td>1.7746</td><td>4.1562</td><td>2.3962</td><td>2.6149</td><td>2.9693</td><td>1.9459</td><td>2.4576</td></tr><tr><td>MFVINS</td><td>Front,Rear</td><td>0.7615</td><td>0.7316</td><td>0.8365</td><td>1.0311</td><td>1.5797</td><td>0.9881</td><td>0.7630</td><td>1.1729</td><td>0.9680</td></tr><tr><td>MFVINS</td><td>Omnidirectional</td><td>0.6538</td><td>0.5627</td><td>0.6511</td><td>1.2697</td><td>0.5823</td><td>0.7439</td><td>0.6519</td><td>0.4588</td><td>0.5554</td></tr><tr><td>MFVINS(D)</td><td>Front,Rear</td><td>0.2243</td><td>0.3419</td><td>0.3552</td><td>0.3579</td><td>0.3523</td><td>0.3263</td><td></td><td></td><td></td></tr><tr><td>MFVINS(D)</td><td>Omnidirectional</td><td>0.2208</td><td>0.1915</td><td>0.2470</td><td>0.1767</td><td>0.2208</td><td>0.2114</td><td></td><td></td><td></td></tr></table>

Table 2: Localization results for diferent methods and configurations on the DRIVING and OCCLUSION sequences. Bold/underlined values indicate the best and second-best results, respectively.

## 5.3 Experiments on OCCLUSION Sequences

To evaluate performance under occlusion scenarios, we collected two sequences in which occlusion persisted continuously during driving, as illustrated in the top left image of Fig. 10. The purpose of these experiments was to verify the robustness of the proposed system. Thus, depth estimation was not used in these experiments.

The quantitative results of these experiments are also shown in Table 2. Our proposed method demonstrates the highest accuracy for localization even under persistent occlusion. Additionally, the omnidirectional camera configuration is more robust than the front-and-rear-only setup in occlusion scenarios. The qualitative results are presented in Fig. 9(b).

![](images/afc90276c629731d0f668fcfecc6162fee17ab24f53694d28355aa6ffc91e2c8.jpg)  
Figure 10: A sample image of occlusion scenarios

## 5.4 Ablation Study

To verify the performance gains provided by each component of the proposed method, we conducted an ablation study. First, we examined how diferent feature extraction algorithms afect the processing time and the number of extracted features. Next, we evaluated whether using an IMU-aided feature tracking algorithm influences localization accuracy. Finally, we investigated the impact of applying a filtering algorithm to remove outlier features caused by fisheye distortion. Since the efects of a learning-based depth estimation method are already covered in Section 5.2, they are omitted here. All experiments were conducted using an omnidirectional camera configuration. For precise results of the contribution of each algorithm, the evaluation was performed on the trajectory before applying pose graph optimization.

## 5.4.1 Feature Detection Algorithm

The experiment on feature detection was conducted using “Case 1” of the DRIVING sequences. The evaluation results for the feature extraction time and the number of untracked features are presented in Table 3. The Shi-Tomasi algorithm used in the VINS-Mono system exhibited the longest processing time, indicating its unsuitability for multi-camera systems. In contrast, the VPI-Harris algorithm used in MCVIO ofers the advantage of the shortest processing time by leveraging GPU acceleration to reduce CPU usage. However, it sufers from the drawback of excessive overlapping feature detections, which leads to a significant increase in the number of untracked features. This negatively impacts localization performance and, as shown in Table 2, explains the performance degradation observed in MCVIO when using omnidirectional cameras compared to using front and rear cameras only.

<table><tr><td>METHOD</td><td>Feature detection algorithm</td><td>CPU/GPU</td><td>Running time per camera [ms]</td><td>Number of untracked features</td></tr><tr><td>VINS-Mono</td><td>Shi-Tomasi</td><td>CPU</td><td>9.83</td><td>5,254 35.05%</td></tr><tr><td>MCVIO</td><td>VPI-Harris</td><td>GPU</td><td>0.35</td><td>97,322 97.39%</td></tr><tr><td>MFVINS</td><td>FAST</td><td>CPU</td><td>0.37</td><td>3,520 32.58%</td></tr></table>

Table 3: Performance of feature detection algorithm. MFVINS uses only CPU, but the running time per camera is similar to that using GPU, and the number of untracked feature is lower.  
![](images/874f90a7f4f93c675bf4ab4f6d021029d297f8afe0a90a783ade9fda07ea91bd.jpg)  
Figure 11: Trajectories of ROTATIONAL sequence following triangle, square, and infinite-shaped paths.

In the proposed method, the FAST feature extraction algorithm was employed to reduce computation time, demonstrating its suitability for multicamera systems. It also resulted in the lowest number and ratio of untracked features, confirming its efectiveness in ensuring stable feature management and providing a favorable environment for omnidirectional camera-based localization.

<table><tr><td>ROTATIONAL</td><td>Triangle</td><td>Square</td><td>Infinite</td></tr><tr><td>Total Length [m]</td><td>119.51</td><td>139.77</td><td>157.56</td></tr><tr><td>IMU-aided tracker</td><td></td><td>RMSE of ATE [m]</td><td></td></tr><tr><td>X</td><td>0.5607</td><td>0.9846</td><td>0.5724</td></tr><tr><td>0</td><td>0.4862</td><td>0.9228</td><td>0.5479</td></tr><tr><td>IMU-aided tracker</td><td colspan="3">avg. number of tracked frames</td></tr><tr><td>X</td><td>16.3122</td><td>18.7986</td><td>17.3838</td></tr><tr><td>0</td><td>16.3383</td><td>18.8622</td><td>17.7092</td></tr></table>

Table 4: Comparison results of IMU-aided feature tracking algorithm for ROTATIONAL sequences.
<table><tr><td rowspan="2">Sequence</td><td colspan="5">DRIVING</td><td colspan="2">OCCLUSION</td></tr><tr><td>Case 1</td><td>Case 2</td><td>Case 3</td><td>Case 4</td><td>Case 5</td><td>Case 1</td><td>Case 2</td></tr><tr><td>Total Length [m]</td><td>136.39</td><td>295.63</td><td>232.40</td><td>196.79</td><td>345.77</td><td>192.58</td><td>132.19</td></tr><tr><td>Outlier Filtering</td><td></td><td></td><td></td><td>RMSE of ATE</td><td>[m]</td><td></td><td></td></tr><tr><td>X</td><td>1.1153</td><td>2.4120</td><td>1.7342</td><td>2.5005</td><td>1.8269</td><td>1.7840</td><td>0.9781</td></tr><tr><td>0</td><td>0.9659</td><td>2.3840</td><td>1.6946</td><td>2.1537</td><td>1.4079</td><td>1.6919</td><td>0.9642</td></tr></table>

Table 5: Comparison results of feature outlier filtering algorithm for all cases.

## 5.4.2 IMU-aided Feature Tracker

Evaluating the performance of the IMU-aided feature tracking algorithm in the DRIVING and OCCLUSION sequences is challenging, since rotation efects are not pronounced in these scenarios. Therefore, to verify the efect of the algorithm, we additionally performed experiments with driving paths of triangular, square, and infinity-shaped that feature prominent rotational motion. The reference trajectories for these sequences are illustrated in Fig. 11.

The quantitative results of the experiment are presented in Table 4. When the proposed algorithm was applied, the localization accuracy was improved compared to the case without it for all three types. In addition, by leveraging inertial data from the IMU, the average number of frames per tracked feature increased, indicating that features could be tracked more stably over a longer period.

## 5.4.3 Filtering Method for Distortion-included Feature Outlier

Experiments were carried out on the DRIVING and OCCLUSION sequences, and Table 5 presents the quantitative results. The findings confirm that outliers caused by distortion adversely afect the accuracy of localization, underscoring the importance of filtering.

## 6 Conclusion

In this paper, we proposed a new visual-inertial system called MFVINS that utilizes multiple fisheye cameras. Building on VINS-Mono and MCVIO, we employed a grid-based FAST feature extraction assisted by an inertial measurement unit (IMU), as well as a distance-based filtering method to remove outlier features resulting from fisheye distortion. In addition, a learningbased depth estimation method was introduced to enhance depth accuracy, and pose graph optimization was used to reduce accumulated errors. Experiments were conducted on real-world datasets demonstrated that the proposed system outperforms existing monocular and multi-camera algorithms in terms of both accuracy and robustness. In future work, we plan to further enhance system practicality by optimizing sensor configurations and improving real-time processing, thereby maximizing the benefits of multi-camera setups and achieving even more accurate and robust perception capabilities.

## References

Chunge Bai, Tao Xiao, Yajie Chen, Haoqian Wang, Fang Zhang, and Xiang Gao. Faster-lio: Lightweight tightly coupled lidar-inertial odometry using parallel sparse incremental voxels. IEEE Robotics and Automation Letters, 7(2):4861–4868, 2022.

Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez, Jos´e MM Montiel, and Juan D Tard´os. Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam. IEEE transactions on robotics, 37(6):1874–1890, 2021.

Luca Carlone, Roberto Tron, Kostas Daniilidis, and Frank Dellaert. Initialization techniques for 3d slam: A survey on rotation estimation and its use in pose graph optimization. In 2015 IEEE international conference on robotics and automation (ICRA), pages 4597–4604. IEEE, 2015.

Hae Min Cho, HyungGi Jo, and Euntai Kim. Sp-slam: Surfel-point simultaneous localization and mapping. IEEE/ASME Transactions on Mechatronics, 27(5):2568–2579, 2021.

Walid Darwish, Wenbin Li, Shengjun Tang, Bo Wu, and Wu Chen. A robust calibration method for consumer grade rgb-d sensors for precise indoor reconstruction. IEEE access, 7:8824–8833, 2019.

Christian Forster, Luca Carlone, Frank Dellaert, and Davide Scaramuzza. On-manifold preintegration for real-time visual–inertial odometry. IEEE Transactions on Robotics, 33(1):1–21, 2016.

Dorian G´alvez-L´opez and Juan D Tardos. Bags of binary words for fast place recognition in image sequences. IEEE Transactions on robotics, 28 (5):1188–1197, 2012.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.

Chris Harris, Mike Stephens, et al. A combined corner and edge detector. In Alvey vision conference, volume 15, pages 10–5244. Manchester, UK, 1988.

Yao He, Huai Yu, Wen Yang, and Sebastian Scherer. Toward robust visualinertial odometry with multiple nonoverlapping monocular cameras. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

Juho Kannala and Sami S Brandt. A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses. IEEE transactions on pattern analysis and machine intelligence, 28(8):1335–1340, 2006.

Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019.

Jianheng Liu, Xuanfu Li, Yueqian Liu, and Haoyao Chen. Rgb-d inertial odometry for a resource-restricted robot in dynamic environments. IEEE Robotics and Automation Letters, 7(4):9573–9580, 2022.

Peize Liu, Chen Feng, Yang Xu, Yan Ning, Hao Xu, and Shaojie Shen. Omninxt: A fully open-source and compact aerial robot with omnidirectional visual perception. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10605–10612. IEEE, 2024.

Xin Liu, Shuhuan Wen, Zhengzheng Guo, and Huaping Liu. Transfer learning-based sparse-reward meta-q-learning algorithm for active slam. Expert Systems with Applications, page 129799, 2025.

Bruce D Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In IJCAI’81: 7th international joint conference on Artificial intelligence, volume 2, pages 674–679, 1981.

Yi Ma, Stefano Soatto, Jana Koˇseck´a, and Shankar Sastry. An invitation to 3-d vision: from images to geometric models, volume 26. Springer, 2004.

J Matas and O Chum. Randomized ransac with td,d test. Image and Vision Computing, 22(10):837–842, 2004a. ISSN 0262-8856. British Machine Vision Computing 2002.

J Matas and O Chum. Randomized ransac with td,d test. Image and Vision Computing, 22(10):837–842, 2004b. ISSN 0262-8856. British Machine Vision Computing 2002.

Anastasios I Mourikis and Stergios I Roumeliotis. A multi-state constraint kalman filter for vision-aided inertial navigation. In Proceedings 2007 IEEE international conference on robotics and automation, pages 3565– 3572. IEEE, 2007.

Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE transactions on robotics, 34(4):1004–1020, 2018.

Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: An eficient alternative to sift or surf. In 2011 International conference on computer vision, pages 2564–2571. Ieee, 2011.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.

David Schubert, Thore Goll, Nikolaus Demmel, Vladyslav Usenko, J¨org St¨uckler, and Daniel Cremers. The tum vi benchmark for evaluating

visual-inertial odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1680–1687. IEEE, 2018.

Hochang Seok and Jongwoo Lim. Rovins: Robust omnidirectional visual inertial navigation system. IEEE Robotics and Automation Letters, 5(4): 6225–6232, 2020.

Zeyong Shan, Ruijian Li, and S¨oren Schwertfeger. Rgbd-inertial trajectory estimation and mapping for ground robots. Sensors, 19(10):2251, 2019.

Ke Sun, Kartik Mohta, Bernd Pfrommer, Michael Watterson, Sikang Liu, Yash Mulgaonkar, Camillo J Taylor, and Vijay Kumar. Robust stereo visual inertial odometry for fast autonomous flight. IEEE Robotics and Automation Letters, 3(2):965–972, 2018.

Deepak Geetha Viswanathan. Features from accelerated segment test (fast). In Proceedings of the 10th workshop on image analysis for multimedia interactive services, London, UK, pages 6–8, 2009.

Yifu Wang, Yonhon Ng, Inkyu Sa, Alvaro Parra, Cristian Rodriguez-Opazo, Taojun Lin, and Hongdong Li. Mavis: Multi-camera augmented visualinertial slam using se 2 (3) based exact imu pre-integration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 1694–1700. IEEE, 2024.

Zhiyu Wang, Weili Ding, Ying Zhang, and Changchun Hua. Otps-vo: Enhanced rgb-d odometry for indoor service robots leveraging structural features. Expert Systems with Applications, 298:129704, 2026.

Avi Wigderson. Mathematics and computation: A theory revolutionizing technology and science. Princeton University Press, 2019.

Shuping Ye, Benlian Xu, Yong Yang, Xu Zhou, Mingli Lu, Jian Shi, and Zhicheng Yang. Dc-slam: Dual-category dynamic feature suppression for rgb-d vslam. Expert Systems with Applications, 299:129952, 2026.

Wei Zhang, Sen Wang, Xingliang Dong, Rongwei Guo, and Norbert Haala. Bamf-slam: Bundle adjusted multi-fisheye visual-inertial slam using recurrent field transforms. In 2023 IEEE international conference on robotics and automation (ICRA), pages 6232–6238. IEEE, 2023.

Kang Zhou, Ming-Gang Duan, Ji-Yang Fu, Yun-Cheng He, Zi-Tong Li, Hai-Ou Shi, and Jing Song. Real-time uncertainty quantification for klt-based

displacement estimation. Engineering Structures, 339:120671, 2025. ISSN 0141-0296.

Xingxing Zuo, Nathaniel Merrill, Wei Li, Yong Liu, Marc Pollefeys, and Guoquan Huang. Codevio: Visual-inertial odometry with learned optimizable dense depth. In 2021 ieee international conference on robotics and automation (icra), pages 14382–14388. IEEE, 2021.