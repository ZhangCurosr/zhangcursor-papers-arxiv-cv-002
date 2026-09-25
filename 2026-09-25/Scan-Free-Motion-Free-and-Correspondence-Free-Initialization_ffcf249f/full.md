# Scan-Free, Motion-Free, and Correspondence-Free Initialization for Doppler LiDAR-Inertial Systems

Mingle Zhao, Jiahao Wang, Tianxiao Gao, Chengzhong Xu, and Hui Kong

Abstract—Robust initialization is crucial for online systems. In the letter, a high-frequency and resilient initialization framework is designed for LiDAR-inertial systems, leveraging both inertial sensors and Doppler LiDAR. The innovative FMCW Doppler LiDAR opens up a novel avenue for robotic sensing by capturing not only point range but also Doppler velocity via the intrinsic Doppler effect. By fusing point-wise Doppler velocity with inertial measurements under non-inertial kinematics, the proposed framework, Free-Init, eliminates reliance on motion undistortion of LiDAR scans, excitation motions, and map correspondences during the initialization phase. Free-Init is also plug-and-play compatible with typical LiDAR-inertial systems and is versatile to handle a wide range of initial motions when the system starts, including stationary, dynamic, and even violent motions. The embedded Doppler-inertial velocimeter ensures fast convergence and high-frequency performance, delivering outputs exceeding 10 kHz. Comprehensive experiments on diverse platforms and across myriad motion scenes validate the framework’s effectiveness. The results demonstrate the superior performance of Free-Init, highlighting the necessity of fast, resilient, and dynamic initialization for online systems.

Index Terms—SLAM, Localization, Doppler LiDAR, Velocity Estimation, Non-Inertial Kinematics.

## RESOURCES

IEEE Xplore Link : Free-Init   
arXiv Paper Link : P Free-Init   
Code & Sequence : <sup>§</sup> Free-Init and <sup>§</sup> FMCW-LIO   
Experiment Video : <sup>Å</sup> Free-Init

## I. INTRODUCTION

IGHT Detection And Ranging (LiDAR) sensors and ployed to facilitate real-time state estimation and environmental mapping. LiDAR-inertial odometry (LIO) systems are widely applied in various field scenarios, including extreme environments [1], [2]. For mobile robots, online LIO not only provides state estimates for planners and controllers but also generates dense maps for perception and navigation.

## A. Initialization in LiDAR-Inertial Odometry

Owing to the intrinsic non-linearity, online LIO systems require accurate initial states (e.g., initial poses, velocities,

![](images/8964ee4281f6ec1f2490a3e8e9d3590cf382efae65a6ac989c94e33bd0952ea3.jpg)  
Fig. 1. Mapping comparison with the default initialization and Free-Init when the platform is moving on the road. The red arrows indicate the platform positions and moving directions at the start of initialization. The rapid speed results in a highly blurred initial map generated with the default initialization. In contrast, the map generated with Free-Init is remarkably distinct, enabling sharp mapping of cars, poles, and traffic signs along the road.

IMU biases) or prior information (e.g., global poses) from the initialization module to ensure the accuracy, stability, and convergence of online estimators. However, there is a scarcity of research regarding the initialization in LIO systems. The initialization in mainstream LIO systems is straightforwardly based on the stationary motion assumption [3]–[7], where the initial rotation can be set to identity or be derived by normalization methods, the initial position and velocity are readily set to zero. Then IMU biases and the gravity vector can be computed using static IMU measurements. Some methods require additional sensors for initialization or calibration. In [8], the authors utilize a camera as an auxiliary sensor to calibrate the temporal-spatial offsets between LiDAR and IMU. Likewise, the Global Navigation Satellite System (GNSS) sensor is leveraged in [9] to constrain the pose estimation.

Besides, a minority of initialization methods rely on specific excitation motions to estimate temporal offsets, IMU biases, extrinsic parameters for subsequent LIO systems [10].

However, despite the methods in [8]–[10] can run online and serve as initialization modules before the LIO system starts, these methods necessitate additional sensors or specific excitation motions. Importantly, these methods essentially align with calibration methods rather than initialization methods. In addition, IMU biases, temporal-spatial offsets, and other states (e.g., poses and velocities) may vary upon each system startup. Moreover, it is challenging to perform excitation motions at each startup, particularly for specific configured platforms such as autonomous vehicles or large-scale drones. Essentially, following the design philosophy for rigorous and precise online systems, time synchronization and extrinsic calibration should be completed within the system design stage and prior to online deployment. Thus, the calibration of unknown parameters should not be handled within initialization modules but rather within calibration and identification processes. Notably, the true responsibility of initialization modules is to provide online estimators with reasonably accurate initial states and uncertainties, ensuring stability and convergence, irrespective of platform motions, operational modes, or environments, while preventing system divergences or failures.

## B. Crux and Ideal Initialization

When the system starts, significant errors often manifest in the initial states. In LIO systems, these large initial errors can lead to cumulative errors in LiDAR scans and maps, which supply observations for state estimation through undistorted scans and map correspondences. Simultaneously, erroneous observations can further exacerbate estimation errors. This dependence between the estimator, LiDAR scans, and the online-established map, creates a positive feedback loop of errors, which can potentially lead to system divergence [11]. Crucially, conventional LiDARs can only provide geometric observations which are directly associated with the known system poses and environmental structures, whereas initialization modules should inherently and independently provide accurate initial states to subsequent estimators, in the absence of known poses, velocities, and maps. Hence, employing conventional LiDAR data in initialization creates a similar chicken-and-egg paradox. Thereby, typical LIO systems [3], [5]–[7] rely solely on IMU data under certain motion assumptions during initialization. Similarly, dynamic initialization that relies on looselycoupled LiDAR odometry [10] fails to fundamentally address the above correspondence-dependency issue. Moreover, online initialization in degenerate scenes (e.g., tunnels, highways, flat terrains) can directly lead to system failures.

Overall, an ideal initialization module for online LIO should be independent of platform motions, accumulated LiDAR scans, LiDAR maps, and environmental structures. Significantly, it should also be capable of offering accurate initial states and maps to the subsequent online estimator, regardless of when, where, and how the system starts.

Fortunately, rapid advancements in LiDAR sensors are unlocking new possibilities for robotic sensing. One notable development is the Frequency Modulated Continuous Wave (FMCW) Doppler LiDAR, which utilizes laser wave modulation in the frequency domain to capture instant range sensing and Doppler velocity [12]. The Doppler measurements provide observations independent of poses and geometric structures [11]–[13]. As a result, adopting FMCW Doppler LiDARs sparks a novel route to designing the aforementioned ideal initialization framework and addresses the crux from the correspondence dependency inherent in conventional LIO systems.

## C. Proposed Methodology

In this work, we propose Free-Init, an initialization framework for LIO systems. The framework overview is shown in Fig. 2. Free-Init takes LiDAR points and IMU data as inputs. The designed Doppler-inertial velocimeter estimates the Li-DAR velocity, body velocity, and gyroscope bias under a pointwise scheme with extremely high-frequency outputs (over 10 kHz). An optimization problem is then established to estimate accelerometer bias and gravity. Finally, the estimated states, uncertainties, and static map are fed into the subsequent LIO as initial estimates. In [13], the authors use the FMCW Doppler LiDAR to design a scan-based velocity estimator, achieving an open-loop, correspondence-free LIO within a continuoustime framework. However, this estimator is not suitable for full odometry and cannot achieve the same accuracy level as conventional LIO systems over long periods, as discussed in [13]. Contrarily, we design a discrete-time velocimeter under a point-wise updating manner without any interpolation models, enabling high-frequency performance for robust initialization. Meanwhile, Free-Init is a complete and resilient initialization framework, leveraging all FMCW Doppler LiDAR, gyroscope, and accelerometer data.

In summary, the contributions are as follows:

1) A complete, LiDAR scan-free, excitation motion-free, and map correspondence-free initialization framework, Free-Init, is proposed for Doppler LiDAR-inertial systems. The framework design consistently aligns with the design philosophy of an ideal initialization module.

2) A novel high-frequency Doppler-inertial velocimeter is designed, exploiting the sensing nature of Doppler Li-DARs with an efficient point-wise filtering scheme.

3) A formulation of Doppler observations, along with the estimation of accelerometer bias and gravity, is derived from a unified non-inertial kinematics perspective.

4) Extensively diverse experiments are conducted, demonstrating the effectiveness and robustness of Free-Init.

5) The source code of Free-Init, the data sequences, the integration of Free-Init into the FMCW-LIO framework, and the experiment video are made publicly available (see the Resources section for details).

## II. NOTATION AND PRELIMINARY

## A. Notation

We use $^ { w } ( \cdot ) , ^ { b } ( \cdot )$ , and $^ l ( \cdot )$ to represent a 3D vector in the world, body, and LiDAR frame, respectively. The body frame b coincides with the IMU frame, and the world frame w is the first body frame $b _ { 0 }$ when the system starts. The inertial frame is the world frame and the earth is static. The gravity $^ w \mathbf { g }$ is constant in the world frame. For a 3D position $\textbf { p } \in$ $\mathbb { R } ^ { 3 }$ from point A to point B in the frame C is denoted as $^ C _ { \mathbf { p } _ { A B } . \mathrm { ~ A ~ } }$ 3D velocity of B with respect to A expressed in C is $c _ { \mathbf { v } _ { A B } }$ . We use $\mathbf { R } _ { A B } \in S O ( 3 )$ to represent a rotation matrix rotating a vector expressed in the frame B to A. $\mathbf { T } _ { A B } \in S E ( 3 )$ transforms a 3D vector in the frame B to A. $[ \mathbf { t } ] _ { \times }$ denotes the $3 \times 3$ skew-symmetric matrix of a 3D vector t. 0 is the $3 \times 1$ zero vector, $\mathbf { 0 } _ { n \times m }$ is the $n \times m$ zero matrix, I is the $3 \times 3$ identity matrix. $\widetilde { ( \cdot ) } , \widehat { ( \cdot ) }$ , and (·) respectively denote the measurement, the propagated state, and the updated state.

![](images/748ddbf0522c3714f02c7727e72e18aa008a83fbf6cb3633d2300b7d6c3dec77.jpg)  
Fig. 2. Framework overview. The initial states, uncertainties, and static map from Free-Init are fed into the subsequent LIO system.

## B. Non-Inertial Kinematics

Considering the motion of a sensor frame s with respect to the non-inertial body frame b, we use $^ b \omega _ { w b }$ and $\bar { b _ { \mathbf { a } _ { w b } } }$ to denote the angular velocity and the acceleration of body frame with respect to the world (inertial) frame w, respectively. Likewise, $s _ { \omega _ { w s } }$ and ${ } ^ { s } \mathbf { a } _ { w s }$ denote the angular velocity and the acceleration of sensor frame with respect to the world frame. They are all expressed in their respective frames. Therefore, both first-order and second-order non-inertial kinematics can be derived [14]. For the translational and rotational parts, the first-order and second-order kinematics are as follows:

$$
{ { \mathbf { } } ^ { b } } { \dot { \mathbf { p } } } _ { b s } = { { } ^ { b } } { \mathbf { v } } _ { w s } - { { } ^ { b } } { \mathbf { v } } _ { w b } - \left[ { { } ^ { b } } \omega _ { w b } \right] _ { \times } { { } ^ { b } } { \mathbf { p } } _ { b s }\tag{1}
$$

$$
{ { \dot { \bf p } } _ { b s } } = { { \bf \ddot { a } } _ { w s } } - { { \bf \ddot { a } } _ { w b } } - \left[ { { \bf \ddot { b } } _ { \omega _ { w b } } } \right] _ { \times } ^ { 2 } { \bf \ddot { p } } _ { b s } - \left[ { { \bf \ddot { b } } _ { \omega _ { w b } } } \right] _ { \times } { \bf \ddot { p } } _ { b s }
$$

$$
- 2 \big [ { } ^ { b } \omega _ { w b } \big ] _ { \times } { } ^ { b } \dot { \mathbf { p } } _ { b s }\tag{2}
$$

$$
\begin{array} { r } { \dot { \bf R } _ { b s } = { \bf R } _ { b s } \left[ { \bf \Xi } ^ { s } \omega _ { w s } - { \bf R } _ { b s } ^ { \top b } \omega _ { w b } \right] _ { \times } } \end{array}\tag{3}
$$

$$
\ddot { \mathbf { R } } _ { b s } = \mathbf { R } _ { b s } ( [ { } ^ { s } \omega _ { w s } - \mathbf { R } _ { b s } ^ { \top b } \omega _ { w b } ] _ { \times } ^ { 2 } + [ { } ^ { s } \dot { \omega } _ { w s } - \mathbf { R } _ { b s } ^ { \top b } \dot { \omega } _ { w b } ] _ { \times }
$$

$$
+ \big [ \big ( \mathbf { \rho } ^ { s } \omega _ { w s } - \mathbf { R } _ { b s } ^ { \top b } \omega _ { w b } \big ) \times \big ( \mathbf { R } _ { b s } ^ { \top b } \omega _ { w b } \big ) \big ] _ { \times } \big )\tag{4}
$$

## III. METHODOLOGY

## A. State and Problem Description

1) State in Typical LIO Systems: The state in general LIO systems usually includes the rotation and position of body frame with respect to the world frame, i.e., $\mathbf { R } _ { w b }$ and ${ } ^ { w } { \bf p } _ { w b }$ (typically, the first body frame is denoted as the world frame), the body velocity in the world frame ${ w } _ { \mathbf { v } _ { w b } }$ , gyroscope and accelerometer biases ${ } ^ { b } \mathbf { b } _ { g }$ and ${ } ^ { b } \mathbf { b } _ { a }$ [3]–[7]. In this work, the system state x in LIO systems is represented as an element which evolves on the 18-dimensional compound differentiable manifold $\mathcal { M } \triangleq S O ( 3 ) \times \mathbb { R } ^ { 1 5 }$ [15], [16]:

$$
\mathbf { x } \triangleq \left[ \mathbf { R } _ { w b } ^ { \top } \quad ^ { w } \mathbf { p } _ { w b } ^ { \top } \quad ^ { w } \mathbf { v } _ { w b } ^ { \top } \quad ^ { b } \mathbf { b } _ { g } ^ { \top } \quad ^ { b } \mathbf { b } _ { a } ^ { \top } \quad ^ { w } \mathbf { g } ^ { \top } \right] ^ { \top } \in \mathcal { M }\tag{5}
$$

Two general operators (“boxplus”: ⊞ and “boxminus”: ⊟) are defined to operate the state on the tangent space of manifolds [15]. The exponential map $\mathrm { E x p } ( \cdot )$ and the logarithmic map $\operatorname { L o g } ( \cdot )$ on $S O ( 3 )$ [17] are utilized to map elements on the manifold and its tangent space.

2) Problem Description of Initialization: Therefore, it is evident from (5) that the objective of initialization modules is to provide relatively accurate initial states and corresponding uncertainties to the subsequent LIO. These initial states include rotation, position, velocity, and gravity with respect to the world frame, as well as IMU biases in the body frame. Additionally, the noise covariance of IMU can also be estimated during the initialization step.

However, most initialization methods in LIO usually depend on certain motion assumptions. One type is the stationary motion assumption [3], [5]–[7], [18], while another is the specific excitation motion assumption [10]. This reliance stems from the fact that conventional LiDARs can only measure geometric point ranges for online estimators, which results in the immeasurability of velocity and IMU biases. Thus, the initial velocity can be set to zero under the stationary assumption [3], [5]–[7], [18], or the velocity state can be estimated using LiDAR-only odometry (LO) during excitation motions [10]. Nevertheless, owing to the paradox discussed in Section I-B, conventional LiDAR measurements are difficult to be utilized in the initialization phase. In contrast, FMCW Doppler LiDARs can directly measure the Doppler velocity of per-return point, which is independent of environmental structures. Thereby, we propose an initialization framework for Doppler LIO systems that leverages inherent Doppler measurements and essential non-inertial kinematic connections, without relying on scan undistortion, motion assumptions, or map correspondences.

## B. Doppler-Inertial Velocimeter

The proposed Doppler-Inertial Velocimeter (DIV) estimates the LiDAR linear velocity, body velocity (both linear and angular), and gyroscope bias using point-wise Doppler velocity and gyroscope data (Fig. (2)). DIV also outputs pose integration and a static map at an extremely high frequency, ensuring fast convergence and stability of the estimator. The integrated poses and estimated velocities are utilized in the following estimation of accelerometer bias and gravity.

1) State in DIV: The state $\mathbf { x } _ { V } \in \mathbb { R } ^ { 1 2 }$ in DIV includes the LiDAR linear velocity ${ l } _ { { \bf { V } } _ { w l } }$ , the body linear velocity ${ \boldsymbol { ^ { b } } } \mathbf { v } _ { w b } .$ , the body angular velocity $b _ { \omega _ { w b } }$ , and the gyroscope bias ${ } ^ { b } { \bf b } _ { g } .$

$$
\begin{array} { r } { \mathbf { x } _ { V } \triangleq \left[ ^ { l } \mathbf { v } _ { w l } ^ { \top } \quad ^ { b } \mathbf { v } _ { w b } ^ { \top } \quad ^ { b } \boldsymbol { \omega } _ { w b } ^ { \top } \quad ^ { b } \mathbf { b } _ { g } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 1 2 } } \end{array}
$$

where the LiDAR velocity is expressed in the LiDAR frame l, whereas other states are expressed in the body frame b. The error state of DIV can be derived in Euclidean space:

$$
\begin{array} { r } { \delta { \bf x } _ { V } \triangleq { \bf x } _ { V } \boxplus { \bf x } _ { V _ { e s t } } = \left[ { } ^ { l } \delta { \bf v } _ { w l } ^ { \top } \delta { \bf v } _ { w b } ^ { \top } \delta \omega _ { w b } ^ { \top } \mathbf { \Sigma } ^ { b } \delta { \bf b } _ { g } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 1 2 } } \end{array}
$$

where $\mathbf { x } _ { V }$ is the true state and $\mathbf { x } _ { V _ { e s t } }$ is the nominal estimate.

2) State Propagation: For the high-frequency point-wise design, the DIV dynamics can be formulated as [19]–[21]:

$$
l _  \dot { \mathbf { v } } _ { w l } = \mathbf { n } _ { v _ { l } } , ^ { b } \dot { \mathbf { v } } _ { w b } = \mathbf { n } _ { v _ { b } } , ^ { b } \dot { \omega } _ { w b } = \mathbf { n } _ { \omega } , ^ { b } \dot { \mathbf { b } } _ { g } = - \frac { 1 } { \tau _ { b _ { g } } } ^ { b } \mathbf { b } _ { g } + \mathbf { n } _ { b _ { g } }\tag{6}
$$

where $\mathbf { n } _ { v _ { l } } , \mathbf { n } _ { v _ { b } } , \mathbf { n } _ { \omega }$ , and $\mathbf { n } _ { b _ { g } }$ are zero-mean Gaussian noises. $\tau _ { b _ { g } }$ is the correlation time of the Gauss-Markov process [20]. To update states when each measurement arrives, we can propagate states with the discrete-time propagation model:

$$
\mathbf { x } _ { V _ { i + 1 } } = \mathbf { x } _ { V _ { i } } \boxplus \left( \Delta t \mathbf { f } \left( \mathbf { x } _ { V _ { i } } , \mathbf { w } _ { V _ { i } } \right) \right)
$$

The system noise $\begin{array} { r } { \mathbf { w } _ { V } \triangleq \left[ \mathbf { n } _ { v _ { l } } ^ { \top } \quad \mathbf { n } _ { v _ { b } } ^ { \top } \quad \mathbf { n } _ { \omega } ^ { \top } \quad \mathbf { n } _ { b _ { q } } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 1 2 } } \end{array}$ and $\Delta t$ is the time interval between two adjacent time steps. The propagation function $\mathbf { f } \left( \mathbf { x } _ { V _ { i } } , \mathbf { w } _ { V _ { i } } \right) \in \mathbb { R } ^ { \bar { 1 } 2 }$ can be directly derived from (6). Accordingly, the state estimate ${ \widehat { \mathbf { x } } } _ { V }$ can be propagated from the last updated state $\overline { { \mathbf { x } } } _ { V _ { l a s t } }$

$$
\widehat { \mathbf { x } } _ { V _ { i + 1 } } = \widehat { \mathbf { x } } _ { V _ { i } } \boxplus \left( \Delta t \mathbf { f } \left( \widehat { \mathbf { x } } _ { V _ { i } } , \mathbf { 0 } _ { 1 2 \times 1 } \right) \right) , \widehat { \mathbf { x } } _ { V _ { 0 } } = \overline { { \mathbf { x } } } _ { V _ { l a s t } }
$$

The error state and the covariance are propagated as:

$$
\begin{array} { r l } & { \delta \mathbf { x } _ { V _ { i + 1 } } = \mathbf { F } _ { V _ { i } } \delta \mathbf { x } _ { V _ { i } } + \mathbf { F } _ { \mathbf { w } _ { i } } \mathbf { w } _ { V _ { i } } , \delta \mathbf { x } _ { V _ { 0 } } = \mathbf { 0 } _ { 1 2 \times 1 } } \\ & { \widehat { \mathbf { P } } _ { V _ { i + 1 } } = \mathbf { F } _ { V _ { i } } \widehat { \mathbf { P } } _ { V _ { i } } \mathbf { F } _ { V _ { i } } ^ { \top } + \mathbf { F } _ { \mathbf { w } _ { i } } \mathbf { Q } _ { i } \mathbf { F } _ { \mathbf { w } _ { i } } ^ { \top } , \widehat { \mathbf { P } } _ { V _ { 0 } } = \overline { { \mathbf { P } } } _ { V _ { l a s t } } } \end{array}
$$

where $\mathbf { Q } _ { i }$ is the noise covariance. $\mathbf { F } _ { V _ { i } }$ and $\mathbf { F } _ { \mathbf { w } _ { i } }$ are the constant transition matrix and the noise Jacobian from (6).

3) State Update: DIV utilizes point-wise Doppler velocity and gyroscope measurements to update states. Doppler measurements can serve as correspondence-free observations [11]– [13], eliminating the need for data association from maps or scans during initialization. This approach effectively mitigates the cumulative effects of large initial errors. Moreover, pointwise state updating with individual LiDAR points enables extremely high-frequency outputs, which significantly facilitates the fast convergence of DIV.

Doppler Velocity Observation. For a LiDAR point $p _ { j }$ sampled at time step j, its position measurement with respect to the LiDAR position at time step j is denoted as $l _ { j } { \widetilde { \bf p } } _ { l _ { j } p _ { j } }$ . The FMCW Doppler LiDAR can also capture the instant Doppler velocity of per LiDAR point [12]. For the static point $p _ { j } \ { \mathrm { ( i . e . } }$ $w _ { \mathbf { V } _ { w p _ { i } } } \equiv \mathbf { 0 } )$ , its Doppler velocity $\widetilde { v } _ { j } ^ { d }$ is measured with respect to the LiDAR position and the LiDAR velocity at time step $j .$ From the perspective of non-inertial kinematics in (1), the true Doppler velocity of $p _ { j }$ is [12]:

$$
v _ { j } ^ { d } = \frac { { } ^ { l _ { j } } \mathbf { p } _ { l _ { j } p _ { j } } ^ { \top } } { \left\| { } ^ { l _ { j } } \mathbf { p } _ { l _ { j } p _ { j } } \right\| } { } ^ { l _ { j } } \dot { \mathbf { p } } _ { l _ { j } p _ { j } } = - \frac { { } ^ { l _ { j } } \mathbf { p } _ { l _ { j } p _ { j } } ^ { \top } } { \left\| { } ^ { l _ { j } } \mathbf { p } _ { l _ { j } p _ { j } } \right\| } { } ^ { l _ { j } } \mathbf { v } _ { w l _ { j } }
$$

where $\boldsymbol { l } _ { j }  _ { \mathbf { V } _ { w l _ { j } } }$ is the LiDAR velocity with respect to the world frame and projected in the LiDAR frame at time step $j .$ Hence, the observation model $h _ { d _ { l } } \left( \mathbf { x } _ { V _ { j } } , n _ { d _ { l } } \right)$ for LiDAR linear velocity $\boldsymbol { l } _ { \mathbf { V } _ { w l } }$ and the observation model $h _ { d _ { b } } \left( \mathbf { x } _ { V _ { j } } , n _ { d _ { b } } \right)$ for body velocity states can be derived from (1):

$$
\begin{array} { r l r } & { h _ { d _ { l } } \left( \mathbf { x } _ { V _ { j } } , n _ { d _ { l } } \right) \triangleq - ^ { l _ { j } } \mathbf { d } _ { l _ { j } p _ { j } } ^ { \top } l _ { j } \mathbf { v } _ { w l _ { j } } + n _ { d _ { l } } } & { ( 7 ) } \\ & { h _ { d _ { b } } \left( \mathbf { x } _ { V _ { j } } , n _ { d _ { b } } \right) \triangleq - ^ { l _ { j } } \mathbf { d } _ { l _ { j } p _ { j } } ^ { \top } \left[ \mathbf { R } _ { b l } ^ { \top } \left( ^ { b _ { j } } \mathbf { v } _ { w b _ { j } } + ^ { b _ { j } } \omega _ { w b _ { j } } \times ^ { b } \mathbf { p } _ { b l } \right) \right] + n _ { d _ { b } } } & \end{array}
$$

where $\begin{array} { r } { l _ { j } \mathbf { d } _ { l _ { j } p _ { j } } = \frac { \iota _ { j } \mathbf { p } \iota _ { j ^ { p _ { j } } } } { \left\| \mathbf { \iota } ^ { l _ { j } } \mathbf { p } \iota _ { j ^ { p _ { j } } } \right\| } \in \mathbb { R } ^ { 3 } } \end{array}$ is the normalized direction vector of point $\dot { p _ { j } } . \ \dot { n _ { d _ { l } } } , \dot { n _ { d _ { b } } } \ \in \ \mathbb { R }$ are the directly modeled measurement noises. $\mathbf { R } _ { b l } , \mathbf { \delta } ^ { b } \mathbf { p } _ { b l }$ are LiDAR-IMU extrinsic

parameters which can be calibrated offline [10], [22]. When point $p _ { j }$ arrives, the observation residuals $r _ { d _ { l _ { j } } } , r _ { d _ { b _ { j } } } \in \mathbb { R }$ can be obtained with the propagated state $\widehat { \mathbf { x } } _ { V _ { j } }$

$$
\begin{array} { r l } & { r _ { d _ { l } } = h _ { d _ { l } } \big ( \mathbf { x } _ { V _ { j } } , n _ { d _ { l } } \big ) - h _ { d _ { l } } \big ( \widehat \mathbf { x } _ { V _ { j } } , 0 \big ) \approx \mathbf { H } _ { d _ { l } } \delta \mathbf { x } _ { V _ { j } } + n _ { d _ { l } } } \\ & { r _ { d _ { b _ { j } } } = h _ { d _ { b } } \big ( \mathbf { x } _ { V _ { j } } , n _ { d _ { b } } \big ) - h _ { d _ { b } } \big ( \widehat \mathbf { x } _ { V _ { j } } , 0 \big ) \approx \mathbf { H } _ { d _ { b _ { j } } } \delta \mathbf { x } _ { V _ { j } } + n _ { d _ { b } } } \end{array}
$$

where the observation Jacobians $\mathbf { H } _ { d _ { l _ { j } } } , \mathbf { H } _ { d _ { b _ { j } } } \in \mathbb { R } ^ { 1 \times 1 2 }$ are:

$$
\begin{array} { r l } & { \mathbf { H } _ { d _ { l _ { j } } } = \left[ - ^ { l _ { j } } \widetilde { \mathbf { d } } _ { l _ { j } p _ { j } } ^ { \top } \quad \mathbf { 0 } ^ { \top } \quad \mathbf { 0 } ^ { \top } \quad \mathbf { 0 } ^ { \top } \right] } \\ & { \mathbf { H } _ { d _ { b _ { j } } } = \left[ \mathbf { 0 } ^ { \top } \quad - \big ( \mathbf { R } _ { b l } { l _ { j } } ^ { l _ { j } } \widetilde { \mathbf { d } } _ { l _ { j } p _ { j } } \big ) ^ { \top } \quad \big ( \mathbf { R } _ { b l } { l _ { j } } ^ { l _ { j } } \widetilde { \mathbf { d } } _ { l _ { j } p _ { j } } \big ) ^ { \top } [ { \boldsymbol { \^ { b } } } _ { \mathbf { p } b l } ] _ { \times } \quad \mathbf { 0 } ^ { \top } \right] } \end{array}
$$

To mitigate erroneous observations from dynamic points, points with observation residuals larger than a certain threshold can be efficiently detected and discarded [11], [12].

Angular Velocity Observation. Considering the gyroscope measurement $b _ { k } \widetilde { \omega } _ { k }$ arrives at time step $k ,$ the angular velocity observation model $\mathbf { h } _ { \omega } ( \mathbf { x } _ { V _ { k } } , \mathbf { n } _ { g } )$ is:

$$
{ { \mathbf { \sp { b } } } _ { k } } \widetilde { \omega } _ { k } = { \mathbf { { h } } } _ { \omega } ( \mathbf { x } _ { V _ { k } } , \mathbf { n } _ { g } ) \triangleq { \mathbf { \sp { b } } } _ { k } \omega _ { w b _ { k } } + { \mathbf { \sp { b } } } _ { k } \mathbf { \sp { b } } _ { g } + { \mathbf { { n } } } _ { g }\tag{8}
$$

where $\mathbf { n } _ { g }$ is the gyroscope measurement noise. The observation residual $\mathbf r _ { \omega _ { k } } \in \mathbb R ^ { 3 }$ and the Jacobian $\mathbf { H } _ { \omega _ { k } } \in \mathbb { R } ^ { 3 \times 1 2 }$ are:

$$
\begin{array} { r l } & { \mathbf { r } _ { \omega _ { k } } = \mathbf { h } _ { \omega } ( \mathbf { x } _ { V _ { k } } , \mathbf { n } _ { g } ) - \mathbf { h } _ { \omega } ( \widehat \mathbf { x } _ { V _ { k } } , \mathbf { 0 } ) \approx \mathbf { H } _ { \omega _ { k } } \delta \mathbf { x } _ { V _ { k } } + \mathbf { n } _ { g } } \\ & { \qquad \mathbf { H } _ { \omega _ { k } } = \left[ \mathbf { 0 } _ { 3 \times 3 } \quad \mathbf { 0 } _ { 3 \times 3 } \quad \mathbf { I } \quad \mathbf { I } \right] } \end{array}
$$

It can be found that the proposed DIV can be formulated as a linear Kalman filter. Therefore, assuming the LiDAR point or the gyroscope measurement arrives at time step $i + 1$ , the updated state $\overline { { \mathbf { x } } } _ { V _ { i + 1 } }$ and covariance $\overline { { \mathbf { P } } } _ { V _ { i + 1 } }$ can be obtained from Kalman filtering [23]. For static initialization, zero velocity can be directly detected from (7) and (8). A strategy similar to the Zero-Velocity Update (ZUPT) [24] can be integrated into the framework, yielding a reduced threedimensional filter specifically for the incremental estimation of gyroscope bias, akin to the mean-based static initialization method [6], [7]. Notably, the DIV filter is fully observable due to the weakly observable gyroscope bias, stemming from the dynamics model in (6). However, as discussed in [13], the gyroscope bias is unobservable in the velocity estimator only using the measurements from an FMCW Doppler LiDAR and a gyroscope. Hence, in practice, the gyroscope bias can also be removed from DIV during dynamic initialization and treated as a constant from the prior calibration or the latest update. In this work, the gyroscope bias is retained within DIV to unify the framework for both dynamic and static initialization.

Thereby, DIV is a high-frequency velocimeter, and the estimated velocities are used for pose integration, static mapping, and the estimation of accelerometer bias and gravity.

4) Pose Integration and Static Mapping: Once the velocities are updated at time step $i + 1$ , the body pose is integrated from the last update time i under the assumption of fixed-axis rotation [20], deriving a trajectory during initialization:

$$
\begin{array} { r l } & { \overline { { \mathbf { R } } } _ { w b _ { i + 1 } } = \overline { { \mathbf { R } } } _ { w b _ { i } } \mathrm { E x p } \biggl ( \displaystyle \int _ { t _ { i } } ^ { t _ { i + 1 } } b _ { \tau } \boldsymbol { \omega } _ { w b _ { \tau } } d \tau \biggr ) } \\ & { { } ^ { w } \overline { { \mathbf { p } } } _ { w b _ { i + 1 } } = { } ^ { w } \overline { { \mathbf { p } } } _ { w b _ { i } } + \displaystyle \int _ { t _ { i } } ^ { t _ { i + 1 } } { \mathbf { R } } _ { w b _ { \tau } } { } ^ { b _ { \tau } } { \mathbf { v } } _ { w b _ { \tau } } d \tau } \end{array}\tag{9}
$$

where ${ b } _ { \tau } { \omega } _ { w b _ { \tau } } , { b } _ { \tau } { } _ { \mathbf { V } _ { w b _ { \tau } } }$ , and $\mathbf { R } _ { w b _ { \tau } }$ are the body angular, linear velocity, and rotation in continuous time. The initial pose in the initialization phase $\overline { { \mathbf { R } } } _ { w b _ { 0 } } = \mathbf { I } , \mathbf { \Pi } ^ { w } \overline { { \mathbf { p } } } _ { w b _ { 0 } } = \mathbf { 0 }$ . Since velocities are updated at the arrival of each LiDAR point and gyroscope measurement, the pose integration in (9) can be computed using accurate higher-order methods [20].

After the pose integration at time step j upon the arrival of LiDAR point $p _ { j }$ , the static point $p _ { j }$ can be immediately mapped to the world frame based on the current pose:

$$
\begin{array} { r } { \mathbf { \overline { { p } } } _ { w p _ { j } } = \mathbf { \overline { { R } } } _ { w b _ { j } } \left( \mathbf { R } _ { b l } { } ^ { l _ { j } } \mathbf { \widetilde { p } } _ { l _ { j } p _ { j } } + \mathbf { \widetilde { p } } _ { b l } \right) + \mathbf { \overline { { p } } } _ { w b _ { j } } } \end{array}\tag{10}
$$

where ${ } ^ { w } \overline { { \mathbf { p } } } _ { w p _ { i } }$ is the updated position of LiDAR point $p _ { j }$ in the world frame. Accordingly, whether the sensor platform is in dynamic or stationary motions during the initialization phase, the proposed framework can achieve accurate pose estimation and static mapping without any motion assumptions. This approach overcomes the paradox in initialization, as discussed in Section I-B. In particular, the point-wise scheme eliminates reliance on accumulated LiDAR scans, thereby avoiding lowfrequency outputs and the need for additional undistortion or motion compensation methods [4], [6], [7], [11], [18].

## C. Estimation of Accelerometer Bias and Gravity

The estimation of accelerometer bias and gravity can be formulated as an optimization problem based on non-inertial kinematics, utilizing the accumulated data (i.e., LiDAR velocities, body velocities, poses, and accelerometer measurements) during the DIV runtime. The LiDAR acceleration $l _ { k } \mathbf { \overline { { a } } } _ { w l _ { k } }$ in the LiDAR frame, the body acceleration $w _ { \overline { { \mathbf { a } } } _ { w b _ { k } } }$ in the world frame, the body angular velocity $b _ { k } \overline { { \omega } } _ { w b _ { k } }$ , and the body angular acceleration $\bar { b } _ { k } \overline { { \dot { \omega } } } _ { w b _ { k } }$ , can be obtained using numerical differentiation at each IMU measurement time step k. The second-order non-inertial kinematics (2) represented by the accelerometer measurement $b _ { k } \widetilde { \mathbf { a } } _ { k }$ and true states is:

$$
\begin{array} { r l } & { \mathbf { R } _ { b l } { } ^ { l _ { k } } \mathbf { a } _ { w l _ { k } } = \left( { } ^ { b _ { k } } \widetilde { \mathbf { a } } _ { k } - { } ^ { b } \mathbf { b } _ { a } - \mathbf { n } _ { a } \right) + \mathbf { R } _ { w b _ { k } } ^ { \top } { } ^ { w } \mathbf { g } } \\ & { \qquad + \left[ { } ^ { b _ { k } } \omega _ { w b _ { k } } \right] _ { \times } ^ { 2 } \mathbf { b } _ { { p } l } + \left[ { } ^ { b _ { k } } \dot { \omega } _ { w b _ { k } } \right] _ { \times } { } ^ { b } \mathbf { p } _ { b l } } \end{array}\tag{11}
$$

$$
{ { \bf { \omega } } ^ { w } } \mathbf { a } _ { w b _ { k } } = \mathbf { R } _ { w b _ { k } } \left( { { \^ { b _ { k } } } { \widetilde { \mathbf { a } } } _ { k } } - { { ^ { b } } } \mathbf { b } _ { a } - \mathbf { n } _ { a } \right) + { ^ { w } } \mathbf { g }\tag{12}
$$

where ${ \bf n } _ { a }$ is the accelerometer noise. The accelerometer bias and gravity can be estimated by substituting the above estimates and measurements into the optimization problem:

$$
\begin{array} { r l } { ^ { b } \overline { { \mathbf { b } } } _ { a } , ^ { w } \overline { { \mathbf { g } } } = \underset { ^ { b } \mathbf { b } _ { a } , ^ { w } \mathbf { g } } { \arg \operatorname* { m i n } } ( \underset { ^ k } { \sum } \middle | \middle | ^ { w } \overline { { \mathbf { a } } } _ { w b _ { k } } - \overline { { \mathbf { R } } } _ { w b _ { k } } ( ^ { b _ { k } } \widetilde { \mathbf { a } } _ { k } - ^ { b } \mathbf { b } _ { a } ) - ^ { w } \mathbf { g } \middle | | ^ { 2 } } & { } \\ { + \underset { ^ k } { \sum } \middle | \middle | \mathbf { R } _ { b l } { } ^ { l _ { k } } \overline { { \mathbf { a } } } _ { w l _ { k } } - ( ^ { b _ { k } } \widetilde { \mathbf { a } } _ { k } - ^ { b } \mathbf { b } _ { a } ) - \overline { { \mathbf { R } } } _ { w b _ { k } } ^ { \top } \mathbf { \Phi } \mathbf { g } } & { } \\ { - [ ^ { b _ { k } } \overline { { \omega } } _ { w b _ { k } } ] _ { \times } ^ { 2 } \mathbf { b } \mathbf { p } _ { b l } - [ ^ { b _ { k } } \overline { { \omega } } _ { w b _ { k } } ] _ { \times } ^ { b } \mathbf { p } _ { b l } \middle | | ^ { 2 } ) } & { } \\ { \mathrm { s . t . } ^ { w } \mathbf { g } \in \mathbb { S } ^ { 2 } ( \boldsymbol { \Omega } . 8 1 ) , ^ { b } \mathbf { b } _ { a } \in \mathbf { B } ( ^ { b } \mathbf { b } _ { a } ) } & { ( 1 \boldsymbol { \Sigma } ) } \end{array}\tag{}
$$

where ${ \mathbb S } ^ { 2 } \left( r \right)$ is the 2-sphere manifold with range r [15] and $\mathbf { B } \left( { } ^ { b } \mathbf { b } _ { a } \right)$ is a plausible set for ${ } ^ { b } \mathbf { b } _ { a }$ . To alleviate measurement or difference noises, the above estimates and measurements can be filtered by a low-pass filter [25]. Besides, for more accurate estimates and a more consistent map, the LiDAR bundle adjustment can be applied in the end of initialization with a larger but less efficient optimization [26]. However, in this work, considering the target of fast initialization for LIO systems, the efficient approach is sufficiently accurate for initialization, as well as the subsequent LIO system.

![](images/04c35972702cb95ee9c414db66198f666db39391c90750e42ab7f44ef795e7aa.jpg)  
Fig. 3. The FMCW Doppler LiDAR-inertial sensor suite can be mounted on (a) handheld, (b) wheeled, and (c) vehicular platforms.

TABLE I DESCRIPTION OF DATA SEQUENCES
<table><tr><td>Sequence</td><td>Platform</td><td>Motion Feature</td><td>Initial Velocity</td><td>Distance (m)</td></tr><tr><td>dyna_01</td><td>Handheld</td><td>Rotational</td><td>5 rad/s</td><td>70</td></tr><tr><td>dyna_02</td><td>Handheld</td><td>Rotational</td><td>5 rad/s</td><td>75</td></tr><tr><td>dyna_03</td><td>Handheld</td><td>Translational</td><td>5 m/s</td><td>103</td></tr><tr><td>dyna_04</td><td>Handheld</td><td>Rotational</td><td>10 rad/s</td><td>311</td></tr><tr><td>dyna_05</td><td>Vehicular</td><td>Translational</td><td>75 km/h</td><td>100</td></tr><tr><td>dyna_06</td><td>Vehicular</td><td>Translational</td><td>60 km/h</td><td>100</td></tr><tr><td>stat_01</td><td>Handheld</td><td>Stationary</td><td>0 m/s, 0 rad/s</td><td>72</td></tr><tr><td>stat_02</td><td>Wheeled</td><td>Stationary</td><td>0 m/s, 0 rad/s</td><td>123</td></tr><tr><td>stat_03</td><td>Wheeled</td><td>Stationary</td><td>0 m/s, 0 rad/s</td><td>130</td></tr><tr><td>stat_04</td><td>Handheld</td><td>Stationary</td><td>0 m/s, 0 rad/s</td><td>314</td></tr></table>

## D. Initial State for Subsequent LIO Systems

After the above optimization, let k denote the last time step in initialization. The initial state $\mathbf { x } _ { \mathrm { 0 } }$ for the subsequent LIO system (5) is assigned with the estimated states as:

$$
\begin{array} { r } { \mathbf { x } _ { 0 } = \left[ \overline { { \mathbf { R } } } _ { w b _ { k } } ^ { \top } \mathbf { \Sigma } ^ { w } \mathbf { \overline { { p } } } _ { w b _ { k } } ^ { \top } \mathbf { \Sigma } \left( \overline { { \mathbf { R } } } _ { w b _ { k } } ^ { \top } \mathbf { \overline { { v } } } _ { w b _ { k } } \right) ^ { \top } \mathbf { \Sigma } ^ { b _ { k } } \overline { { \mathbf { b } } } _ { g } ^ { \top } \mathbf { \Sigma } ^ { b } \mathbf { \overline { { b } } } _ { a } ^ { \top } \mathbf { \Sigma } ^ { w } \mathbf { \overline { { g } } } ^ { \top } \right] ^ { \top } } \end{array}
$$

where ${ } ^ { b } { \overline { { \mathbf { b } } } } _ { a }$ and $\mathbf { \omega } ^ { w } \mathbf { \overline { { g } } }$ are solved from (13). The velocity and gyroscope bias states are updated from DIV. The body pose in the world frame is integrated by (9) up to the last time step k. Ultimately, as depicted in Fig. 2, the estimates and the static map from Free-Init are fed into the subsequent LIO, facilitating rapid convergence, stability, and robustness.

TABLE II  
ABSOLUTE TRANSLATIONAL ERRORS (RMSE, M) AND END-TO-END ERRORS (END TO END, M) OF LOCALIZATION RESULTS
<table><tr><td>System</td><td>Metric</td><td>Method</td><td>dyna_01</td><td>dyna_02</td><td>dyna_03</td><td>dyna_04</td><td>dyna_05</td><td>dyna_06</td><td>stat_01</td><td>stat_02</td><td>stat_03</td><td>stat_04</td></tr><tr><td rowspan="4">FAST-LIO2 [6]</td><td rowspan="2">RMSE</td><td>Default</td><td>8.21</td><td>X</td><td>1.92</td><td>×</td><td>2.55</td><td>28.67</td><td>0.20</td><td>0.36</td><td>0.10</td><td>0.31</td></tr><tr><td>Free-Init</td><td>0.25</td><td>0.10</td><td>0.09</td><td>0.15</td><td>0.92</td><td>2.02</td><td>0.18</td><td>0.11</td><td>0.10</td><td>0.04</td></tr><tr><td>End to End</td><td>Default</td><td>6.65</td><td>×</td><td>2.81</td><td>×</td><td></td><td></td><td>0.36</td><td>0.25</td><td>0.24</td><td>0.08</td></tr><tr><td></td><td>Free-Init</td><td>0.51</td><td>0.13</td><td>0.25</td><td>0.58</td><td></td><td></td><td>0.21</td><td>0.16</td><td>0.22</td><td>0.08</td></tr><tr><td rowspan="4">DLIO [7]</td><td>RMSE</td><td>Default</td><td>5.27</td><td>6.23</td><td>×</td><td>2.01</td><td>36.90</td><td>42.83</td><td>1.17</td><td>0.91</td><td>1.29</td><td>0.57</td></tr><tr><td></td><td>Free-Init</td><td>0.38</td><td>0.33</td><td>0.45</td><td>0.39</td><td>1.10</td><td>3.88</td><td>0.42</td><td>0.30</td><td>0.69</td><td>0.42</td></tr><tr><td>End to End</td><td>Default</td><td>5.34</td><td>11.98</td><td>×</td><td>5.02</td><td></td><td></td><td>1.01</td><td>1.93</td><td>0.03</td><td>1.37</td></tr><tr><td></td><td>Free-Init</td><td>0.64</td><td>0.56</td><td>1.08</td><td>1.99</td><td></td><td></td><td>0.10</td><td>0.12</td><td>0.04</td><td>1.16</td></tr></table>

× The cross denotes that the system fails or severely drifts in the corresponding sequence.  
− The trajectories in the corresponding two sequences (dyna 05 and dyna 06) are not end-to-end.

![](images/3ccc55bc69777465710831fc8bcb906225f4a4e818d11fe79bcfcbb915f57bb6.jpg)  
Fig. 4. Comparison of trajectory estimation on dyna 01.

## IV. EXPERIMENTS

## A. Experiment Design and Data Collection

To assess the effectiveness of the proposed initialization method, we perform comprehensive experiments and analyze its performance based on both quantitative and qualitative results. Considering the short duration of initialization and different initialization methods can lead to varying levels of localization accuracy and mapping consistency in the subsequent LIO, we integrate Free-Init into two state-of-the-art LIO systems: FAST-LIO2 [6] (tightly-coupled) and DLIO [7] (loosely-coupled), while maintaining identical parameters and configurations as the default version, except for the initialization module. We evaluate localization accuracy (Section IV-B), mapping consistency (Section IV-C), initialization performance in structure-degenerated environments (Section IV-D), as well as velocity estimation accuracy and time consumption (Section IV-E) across diverse platforms, scenarios, and initial motion features.

In [27], the authors release a heterogeneous LiDAR dataset (HeLiPR) for place recognition tasks, including an FMCW Doppler LiDAR. However, the velocity of LiDAR points in HeLiPR is the absolute velocity after post processing, rather than the raw Doppler velocity. This limitation renders HeLiPR unsuitable for online state estimation. Thus, given the absence of public FMCW Doppler LiDAR-inertial datasets containing raw Doppler measurements, it is necessary to collect a dataset to validate the effectiveness of the proposed framework. An FMCW Doppler LiDAR, Aeva Aeries II [28], and an Xsens MTi-G-710 IMU are used to assemble our FMCW Doppler LiDAR-inertial sensor suite. The sensor suite can be mounted on handheld, wheeled, and vehicular platforms (Fig. 3). The frequency of IMU measurements is 200 Hz, and the camera on the sensor suite is only for First-Person-View (FPV) recording. The computing device is equipped with an Intel i7-1165G7 CPU. The sensor suite is held by a person on the handheld platform. The speed of the handheld platform is slow (about 1.5 m/s), while the wheeled platform moves faster (up to 6.5 m/s). The speed of the vehicular platform is significantly higher, ranging from 60 to 75 km/h. It is also ensured that the starting and ending positions of data sequences (except for dyna 05 and dyna 06) precisely coincide to facilitate the end-to-end evaluation of localization accuracy. The RTK GNSS provides ground-truth positions for the handheld and wheeled platforms. For the vehicular platform, the dual RTK GNSS integrated with inertial navigation systems (RTK/INS) offers 6-DoF pose and velocity truths (Fig. 3(c)). To validate the effectiveness of Free-Init for dynamic initialization, the dyna sequences in Table I commence with vigorous dynamic rotational and translational motions, while the stat sequences remain stationary for approximately the initial 1 to 2 seconds. The distances and initial motion features of data sequences are presented in Table I. In Free-Init, the data accumulation time is a configurable parameter, and in the experiments, we maintain consistent parameters and settings across the compared methods. The demonstration video of experiments is available at: https://youtu.be/FbyzvJ-4bHI.

![](images/b4f81df7c7669cbea0dd3d2e92fce9d95ed6ceccec36bcdf432e1530a7f2860b.jpg)  
Fig. 5. Mapping comparison with the default initialization and Free-Init. Left column: FPV camera images. Middle column: Mapping results with the default initialization in the initial period. Right column: Mapping results with Free-Init in the initial period.

![](images/9f7f3a8eeb1d7405829228b189dea8eeddf9d6c36aa83f4413865bb7de21ed49.jpg)  
Fig. 6. Initialization in degenerate tunnels on the forward moving handheld (top), wheeled (middle), and vehicular (bottom) platforms. Left column: FPV camera images. Middle column: Accurate forward velocity estimation with Free-Init. Right column: Erroneous backward motion estimation with the default initialization.

## B. Localization Accuracy

We embed Free-Init into two LIO systems and evaluate its impact on localization accuracy. Notably, we maintain identical system parameters when evaluating different methods, only except for the initialization module. The results are shown in Table II. In dyna sequences, the system starts with severely dynamic rotational and translational motions. It can be found that in the dyna sequences, Free-Init significantly outperforms the default initialization method on localization accuracy. Particularly, in dyna 02 and dyna 04 sequences where the system starts with both fast translational and rotational motions, the default method leads to system failures of the subsequent LIO. Contrarily, the system equipped with Free-Init exhibits high accuracy in returning to starting positions, thanks to precise estimation of velocities and poses during initialization. The estimated trajectories of different methods with FAST-LIO2 [6] in dyna 01 are depicted in Fig. 4, and the default method yields considerable errors (6.65 m) due to the dynamic motion during initialization. Notably, the fast vehicle speeds in dyna 05 and dyna 05 pose significant challenges for both initialization and subsequent LIO systems. However, systems with Free-Init can still achieve accurate localization. In stat sequences, Free-Init achieves comparable yet more accurate localization accuracy than the default initialization. This improvement arises from the inherent dynamic point removal and the ZUPT-like strategy during static initialization, enabling the establishment of a static map. This static map can further provide consistent constraints for the subsequent LIO and prevent potential drifts. In contrast, the default methods discard all LiDAR points during initialization, thereby failing to provide sufficient constraints in the initial running stage of the subsequent LIO. The results reveal that a resilient initialization module can offer accurate initial states for online systems and prevent failures in challenging dynamic cases, regardless of loosely-coupled or tightly-coupled frameworks.

![](images/8c8f5b485f998f8fa0a1fdf0989b501e84ab8175987a84fe5a910b1286d442bc.jpg)  
Fig. 7. Comparison of velocity estimation on dyna 05 and dyna 06.

TABLE III  
AVERAGE RUNNING TIME (S) OF WHOLE FRAMEWORK
<table><tr><td rowspan=1 colspan=1>Sequence Type</td><td rowspan=1 colspan=1>dyna Sequences</td><td rowspan=1 colspan=1>stat Sequences</td></tr><tr><td rowspan=1 colspan=1>Average Running Time</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.116</td></tr></table>

## C. Mapping Consistency

After the initialization and a few seconds of subsequent LIO running, we can evaluate the estimation accuracy and consistency of different initialization methods from the established LiDAR maps. The dynamic initial motions render system failures with the default initialization, outputting chaotic maps. Contrarily, systems with Free-Init can handle these violent initial motions, generating distinct and consistent maps. In the top row of Fig. 5, the sensor suite is handheld by a person who runs vigorously, performing intense rotational motions in the front of the building. Simultaneously, the system initiates with fast shaking and intense rotations during the person’s running. It can be found that the system with the default initialization collapses and outputs an extremely chaotic map (Fig. 5(b)). On the contrary, the system with Free-Init can precisely estimate initial states while outputs a consistent, neat, and accurate map of the environment (Fig. 5(c)). Similarly, the middle row of Fig. 5 shows the mapping results during initialization in a corridor with rapid and substantial rotational motions. As expected, the system with the default initialization suffers from a system crash, deriving a messy and meaningless map (Fig. 5(e)). While the system with Free-Init can construct a sharp map, portraying distinct structures of the corridor. The top view of the corridor map is shown in Fig. 5(f). As shown in Fig. 1 and the bottom row of Fig. 5, the sensor suite is mounted on both the wheeled platform with a forward speed of 6.5 m/s and the vehicular platform with a forward speed of 75 km/h. The system initiates when the platforms are moving forward on the road. Obviously, the high speeds lead to vague mapping outputs even system collapses with the default initialization. However, the LiDAR maps generated by systems with Free-Init are consistent and clear, enabling them to offer accurate LiDAR correspondences for subsequent LIO systems.

## D. Initialization in Structure-Degenerated Environments

An insight from the correspondence-free observation [11]– [13] in the proposed framework is that the observations are independent of environmental structures, even in structuredegenerated scenes. As a result, Free-Init is expected to offer accurate initial states for online LIO in both structured and structure-degenerated environments, regardless of dynamic or stationary motions. On the contrary, typical LIO systems estimate system states solely based on geometric observations, which often leads to estimation failures in degenerate cases. In particular, in scenarios where the LIO system is moving within a degenerate environment, Free-Init should be capable of accurately estimating the initial velocity for subsequent LIO. As shown in Fig. 6, we test different initialization methods in degenerate tunnels on all forward moving handheld, wheeled, and vehicular platforms. It can be found that Free-Init can provide accurate forward velocity estimates to subsequent LIO systems (about 1.43 m/s, 6.23 m/s, and 16.88 m/s on the handheld, wheeled, and vehicular platforms, respectively), even in these degenerate scenes. However, due to the inability of the default initialization to estimate accurate initial velocity states and the lack of geometric features, the system exhibits erroneous backward motions immediately upon startup, ultimately leading to system failures. The results reveal that the Doppler-aided and correspondence-free observations in the proposed method are remarkably effective, especially in degenerate scenarios. The test results in degenerate scenarios also indicate that the proposed framework overcomes the geometric-correspondence-dependency paradox inherent in the initialization of conventional LIO systems (discussed in Section I-B), fundamentally resolving the issue from the standpoint of the correspondence-free first-order kinematics.

## E. Velocity Estimation and Time Consumption

We also evaluate the velocity estimation accuracy of the proposed Doppler-Inertial Velocimeter (DIV) and the method in Yoon et al. 2023 [13] on dyna 05 and dyna 06, leveraging the high-frequency ground truth from RTK/INS. The absolute velocity error results are shown in Fig. 7. Under the point-wise updating scheme, DIV achieves accurate velocity estimation with an extremely high output frequency exceeding 10 kHz. The point-wise updating design eliminates the reliance on explicit motion compensations for both range and Doppler measurements [11], as well as the need for specific interpolation models. Additionally, the average running time of the entire framework of Free-Init is presented in Table III.

## V. CONCLUSION

This letter presents Free-Init, a LiDAR scan-free, excitation motion-free, and map correspondence-free initialization framework for Doppler LIO systems. The design of Free-Init is enhanced by point-wise Doppler velocity, aligning consistently with the sensing nature of Doppler LiDARs. The primary focus of this work is an initialization framework that aims to achieve a near-ideal initialization design without relying on motion or observation assumptions. This is also the first work to comprehensively showcase the performance potential of Doppler LiDARs across varied platforms, including handheld, robotic, and vehicular platforms. Furthermore, a pivotal insight in this work is that the core solution for initialization lies in the first-order tangent space of system dynamics (e.g., the first-order kinematics of LIO). More importantly, beyond the initialization of Doppler LIO systems, this framework can also be applied to offline multi-session mapping, robot kidnapping cases, online re-initialization, and online startup or switching of state estimators in challenging scenarios. In addition, the Doppler-inertial velocimeter can provide high-frequency velocity estimates for online planners and controllers, facilitating agile and aggressive maneuvers. Overall, the Doppler-aided methodology highlights the potential of Doppler LiDARs for future research and real-world applications, unveiling the essential role of the intrinsically velocity-aware sensors in robotic sensing and estimation systems.

## REFERENCES

[1] D. Lee, M. Jung, W. Yang, and A. Kim, “Lidar odometry survey: recent advancements and remaining challenges,” Intelligent Service Robotics, vol. 17, no. 2, pp. 95–118, 2024.

[2] K. Ebadi, L. Bernreiter, H. Biggie, G. Catt, Y. Chang, A. Chatterjee, C. E. Denniston, S.-P. Deschenes, K. Harlow, S. Khattakˆ et al., “Present and future of slam in extreme environments: The darpa subt challenge,” IEEE Transactions on Robotics, 2023.

[3] C. Qin, H. Ye, C. E. Pranata, J. Han, S. Zhang, and M. Liu, “Lins: A lidar-inertial state estimator for robust and efficient navigation,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 8899–8906.

[4] T. Shan, B. Englot, D. Meyers, W. Wang, C. Ratti, and D. Rus, “Lio-sam: Tightly-coupled lidar inertial odometry via smoothing and mapping,” in 2020 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2020, pp. 5135–5142.

[5] W. Xu and F. Zhang, “Fast-lio: A fast, robust lidar-inertial odometry package by tightly-coupled iterated kalman filter,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 3317–3324, 2021.

[6] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidarinertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053–2073, 2022.

[7] K. Chen, R. Nemiroff, and B. T. Lopez, “Direct lidar-inertial odometry: Lightweight lio with continuous-time motion correction,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 3983–3989.

[8] Y. Wang and H. Ma, “Online spatial and temporal initialization for a monocular visual-inertial-lidar system,” IEEE Sensors Journal, vol. 22, no. 2, pp. 1609–1620, 2021.

[9] Z. Taylor and J. Nieto, “Motion-based calibration of multimodal sensor arrays,” in 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2015, pp. 4843–4850.

[10] F. Zhu, Y. Ren, and F. Zhang, “Robust real-time lidar-inertial initialization,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 3948–3955.

[11] M. Zhao, J. Wang, T. Gao, C. Xu, and H. Kong, “Fmcw-lio: A doppler lidar-inertial odometry,” IEEE Robotics and Automation Letters, 2024.

[12] B. Hexsel, H. Vhavle, and Y. Chen, “DICP: Doppler Iterative Closest Point Algorithm,” in Proceedings of Robotics: Science and Systems, New York City, NY, USA, June 2022.

[13] D. J. Yoon, K. Burnett, J. Laconte, Y. Chen, H. Vhavle, S. Kammel, J. Reuther, and T. D. Barfoot, “Need for speed: Fast correspondencefree lidar-inertial odometry using doppler velocity,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 5304–5310.

[14] L. Landau, E. Lifshitz, and J. Sykes, Mechanics: Volume 1, ser. Course of theoretical physics. Elsevier Science, 1976. [Online]. Available: https://books.google.com/books?id=e-xASAehg1sC

[15] C. Hertzberg, R. Wagner, U. Frese, and L. Schroder, “Integrating generic¨ sensor fusion algorithms with sound state representations through encapsulation of manifolds,” Information Fusion, vol. 14, no. 1, pp. 57–77, 2013.

[16] D. He, W. Xu, and F. Zhang, “Kalman filters on differentiable manifolds,” arXiv preprint arXiv:2102.03804, 2021.

[17] R. M. Murray, Z. Li, and S. S. Sastry, A mathematical introduction to robotic manipulation. CRC press, 2017.

[18] J. Zhang and S. Singh, “Loam: Lidar odometry and mapping in realtime.” in Robotics: Science and systems, vol. 2, no. 9. Berkeley, CA, 2014, pp. 1–9.

[19] H. Stalford, “High-alpha aerodynamic model identification of t-2c aircraft using the ebm method,” Journal of Aircraft, vol. 18, no. 10, pp. 801–809, 1981.

[20] P. G. Savage et al., Strapdown analytics. Strapdown Associates Maple Plain, MN, 2000, vol. 2.

[21] W. Xu, D. He, Y. Cai, and F. Zhang, “Robots’ state estimation and observability analysis based on statistical motion models,” IEEE Transactions on Control Systems Technology, vol. 30, no. 5, pp. 2030–2045, 2022.

[22] J. Lv, J. Xu, K. Hu, Y. Liu, and X. Zuo, “Targetless calibration of lidar-imu system based on continuous-time batch estimation,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 9968–9975.

[23] B. M. Bell and F. W. Cathey, “The iterated kalman filter update as a gauss-newton method,” IEEE Transactions on Automatic Control, vol. 38, no. 2, pp. 294–297, 1993.

[24] E. Foxlin, “Pedestrian tracking with shoe-mounted inertial sensors,” IEEE Computer graphics and applications, vol. 25, no. 6, pp. 38–46, 2005.

[25] F. Gustafsson, “Determining the initial states in forward-backward filtering,” IEEE Transactions on signal processing, vol. 44, no. 4, pp. 988–992, 1996.

[26] Z. Liu, X. Liu, and F. Zhang, “Efficient and consistent bundle adjustment on lidar point clouds,” IEEE Transactions on Robotics, 2023.

[27] M. Jung, W. Yang, D. Lee, H. Gil, G. Kim, and A. Kim, “Helipr: Heterogeneous lidar dataset for inter-lidar place recognition under spatiotemporal variations,” The International Journal of Robotics Research, p. 02783649241242136, 2023.

[28] Aeva Inc. Aeries II, Accessed Aug. 8, 2023. [Online]. Available: https://www.aeva.com/aeries-ii/