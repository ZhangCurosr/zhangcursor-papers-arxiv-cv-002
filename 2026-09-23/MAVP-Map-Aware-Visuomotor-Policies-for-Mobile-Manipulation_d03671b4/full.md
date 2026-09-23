# MAVP: Map-Aware Visuomotor Policies for Mobile Manipulation

Jinhe Tang<sup>1,†</sup>, Ruixiao Dai<sup>1,†</sup>, and Weiming Zhi<sup>1∗</sup>

<sup>1</sup>School of Computer Science, The University of Sydney, Australia

<sup>†</sup>These authors contributed equally.

<sup>∗</sup>Corresponding author: Weiming.Zhi@sydney.edu.au

Project website: https://123qwedsa123.github.io/mavp/

![](images/1d7b4b3a406f7c3b6371d2c011bd8cb88e7fd366440f16ea7da908f11e3b2199.jpg)  
Fig. 1: We propose Map-Aware Visuomotor Policies (MAVP), evaluated on six real-world mobile manipulation tasks requiring coordinated base motion and manipulation across multiple workspace locations.

Abstract— Successful mobile manipulation requires coordinated base and arm motion while maintaining accurate spatial positioning. However, demonstration-trained policies can struggle to realise the intended base motion reliably, leading to spatial misalignment and subsequent manipulation failures. We present MAVP (Map-Aware Visuomotor Policies), a framework that improves execution reliability by predicting explicit base-pose targets and tracking them using localisation feedback. MAVP reconstructs a static map from teleoperated demonstrations and expresses demonstrated base trajectories in a shared map frame, providing consistent spatial supervision across demonstrations. At execution time, the policy receives RGB observations, joint states, and the robot’s current mapframe base pose, and jointly predicts target base poses, arm actions, and gripper actions. A low-level controller tracks the predicted base targets using feedforward motion and poseerror feedback, enabling correction of execution deviations. We additionally use pose-noise augmentation during training to improve robustness to errors in the policy’s pose input. Across six real-world manipulation tasks and three policy families, MAVP achieves higher task success rates than unanchored velocity control in all tasks. Videos and additional results are available at https://123qwedsa123.github.io/mavp/.

## I. INTRODUCTION

Mobile manipulation requires robots to coordinate base and arm motion across multiple locations. For example, disassembling an object and delivering its parts requires the base to reach suitable poses for manipulation at successive locations. Base position and orientation affect both the arms reachable workspace and the camera viewpoints, so positioning errors can disrupt subsequent manipulation. Learning from demonstration enables robots to acquire motion skills through learned dynamical systems [1], [2] or probabilistic task-space trajectory models [3]. For mobile manipulation, teleoperated demonstrations support learning coordinated base and arm behaviours [4], but reliable execution remains challenging.

Prior work explores representations and coordination for mobile manipulation. Mobi-π selects suitable base poses for a pretrained manipulation policy [5], while HoMeR combines learned end-effector pose targets with whole-body control [6]. Spatial Action Maps represent actions in a spatial map [7], and other policies use features from a learned 3D map [8].

Without an explicit base-pose target, a velocity-based policy controls base motion incrementally, causing small prediction and tracking errors to accumulate into base-pose drift over time. The policy must also infer its spatial state from visual and proprioceptive observations to continuously adjust the base motion. Pose-based targets instead provide an explicit spatial objective for feedback control. However, independently initialised coordinate frames across demonstrations can assign different coordinates to the same base pose in the workspace. A shared map resolves this ambiguity by providing a consistent frame for demonstration supervision and target tracking.

![](images/0761314c6533a865414fe9b4d000732c54fb10e8b24bf45a1ac324f372048192.jpg)  
Fig. 2: Overview of MAVP: map localisation and visual odometry (VO) estimate the base pose in a shared static map reconstructed from masked demonstrations. The policy jointly predicts base-pose targets and arm/gripper actions; feedforward (FF) and pose-error feedback (FB) track the base targets (camera views are illustrative).

We present MAVP (Map-Aware Visuomotor Policies), which expresses demonstrated base trajectories, current mapframe base-pose estimates, and predicted base targets in a shared map frame. The policy receives visual observations, joint states, and the estimated base pose, and jointly predicts base-pose targets, arm actions, and gripper actions. A low-level controller converts the base targets into velocity commands using feedforward motion and pose-error feedback, allowing execution deviations to be corrected without requiring the policy to predict each corrective velocity command.

MAVP constructs its spatial reference from teleoperated demonstrations, masking robot arms and moving objects before reconstructing a static scene map. During execution, visual localisation, camera calibration, and robot kinematics provide the base-pose estimate. To improve tolerance to imperfect localisation, we perturb the policy’s pose inputs during training while retaining the demonstrated action targets.

Our technical contributions include,

• Shared map construction from demonstrations. We construct a shared static map from teleoperated demonstrations by masking robot arms and dynamic regions before reconstruction. The map provides a common spatial reference for aligning demonstrations and expressing base states and action targets across episodes.

• A map-aware policy learning and execution framework. We introduce MAVP, which conditions the policy on the current map-frame base pose and jointly predicts base-pose targets, arm actions, and gripper actions. A feedforward-plus-feedback controller tracks the base targets, while pose-noise augmentation improves robustness to errors in the policy pose input.

• Evaluation across tasks, policy families, and system components. We rigorously evaluate and analyse the properties of MAVP on real-world tasks using different policy classes.

## II. RELATED WORK

Mobile manipulation and spatial representations: Mobile ALOHA learns whole-body actions from teleoperation [4]. M3 composes reinforcement-learned manipulation and navigation skills [9], and Mobi-π selects base poses for a pretrained manipulation policy [5]. UMI-on-Legs and HoMeR connect learned end-effector targets to whole-body controllers [6], [10]. HoMMI uses a gripper-centred frame for hand-eye policy observations and actions [11]. Spatial Diagrammatic Instructions express spatial objectives and constraints from sketches for mobile-base placement [12]. Other methods use point clouds or maps for geometric context [7], [8], [13].

MAVP learns base-pose targets in a shared map frame and tracks them with localisation feedback, while jointly predicting arm and gripper actions.

Action representation and imitation robustness: Automatic Waypoint Extraction (AWE) compresses demonstrations into spatial waypoints [14], while HYDRA combines sparse waypoints, dense actions, and action relabelling [15]. Diffeomorphic transforms encode modular robot motions and compose them to adapt demonstrated behaviours to changed surroundings [1]. DAgger addresses policy-induced distribution shift through expert labels on learner-visited states [16], and DART perturbs demonstration execution to collect corrective behaviour [17]. MAVP addresses localisation uncertainty through pose-noise augmentation without collecting corrective demonstrations.

Mapping and localisation: Scene mapping and localisation methods reconstruct landmarks, match images, estimate camera poses, and mask dynamic regions [18]–[22]. Learned matchers such as SuperGlue and LightGlue use contextual feature relationships to estimate correspondences [23], [24]. Joint calibration and scene representation methods align reconstructed geometry with robot frames [25], [26]. In MAVP, established mapping and localisation techniques provide a shared scene reference for base-pose supervision and execution feedback.

## III. MAVP: MAP-AWARE VISUOMOTOR POLICIES

MAVP uses a shared map as a common spatial reference for teleoperated demonstrations, policy learning, and closedloop execution. As shown in Figure 2, the shared map is first reconstructed from the demonstrations while excluding nonstatic scene content. During deployment, live observations are then matched against this map, together with visual odometry, to estimate the current base pose in the map frame. Finally, the estimated pose provides the spatial reference for the visuomotor policy, which predicts map-referenced base targets together with manipulation actions.

System setup: MAVP assumes that the mapped scene geometry remains static. In our setup, we use a dual-arm wheeled robot with calibrated head and wrist cameras. The headmounted stereo RGB cameras provide map reconstruction and visual odometry; calibrated camera-to-base extrinsics transform camera-pose estimates into map-frame base poses.

## A. Shared Map Construction

MAVP builds a static map offline from demonstrations. Robot-arm and dynamic-object regions are masked out so that the map is reconstructed from static scene features. Figure 3 summarises the static-scene filtering and shared map reconstruction pipeline, together with the resulting reconstructed map.

Reference Sampling and Static-Scene Filtering: MAVP uniformly samples stereo image pairs from the head-mounted cameras in each demonstration to construct the shared map M. Let $I _ { n } ^ { c }$ denote the n-th sampled image from camera $c \in \{ L , R \}$ , where L and R denote the left and right stereo cameras. To exclude the robot from map reconstruction, we project the robot model into each image using the joint configuration, forward kinematics, and calibrated camera parameters, yielding a binary robot-arm mask $A _ { n } ^ { c }$ . Here, $A _ { n } ^ { c } ( u ) = 1$ marks pixel u as part of a projected robot arm.

Dynamic scene regions can violate the cross-view consistency required for static map reconstruction. We therefore apply VGGT4D [22] independently to the image sequence from each head-camera view to separate dynamic elements from the static scene. VGGT4D extracts dynamic cues from VGGT’s global attention, aggregates them over a temporal window, and applies projection-gradient refinement to sharpen the resulting region boundaries. This yields a binary dynamicregion mask $D _ { n } ^ { c }$ for each sampled image, where $D _ { n } ^ { c } ( u ) = 1$ indicates that pixel u is classified as dynamic.

We combine the robot-arm and dynamic-region masks to retain only regions used for static map reconstruction:

$$
M _ { n } ^ { c } ( u ) = \left( 1 - A _ { n } ^ { c } ( u ) \right) \left( 1 - D _ { n } ^ { c } ( u ) \right) .\tag{1}
$$

Thus, $M _ { n } ^ { c } ( u ) ~ = ~ 1$ indicates that pixel u is retained for map reconstruction, while robot-arm and dynamic regions are excluded. We denote the sampled images and their corresponding mapping masks by $\mathcal { T } _ { M } = \{ ( I _ { n } ^ { c } , M _ { n } ^ { c } ) \} _ { n , c } .$

![](images/efc247995d6c452276a38f64c5a84e92ffd46579f8471effc83a923317cb8d92.jpg)  
Fig. 3: Shared map reconstruction from stereo demonstrations after masking robot arms and dynamic regions.

Shared Map Reconstruction: Using the retained regions in $\mathcal { T } _ { M }$ , MAVP extracts and matches visual features across sampled frames, demonstrations, and stereo views. COLMAP [18] performs two-view geometric verification to reject inconsistent matches, and then reconstructs a shared sparse map by registering the reference images and triangulating matched features into 3D landmarks. Bundle adjustment jointly refines the camera poses and landmark positions while keeping the calibrated camera intrinsics and stereo geometry fixed. The resulting map M contains the reference camera poses, 3D landmarks, and their associated image features, which are subsequently used for visual localisation.

## B. Map-Based Localisation

During execution, MAVP localises the head camera in the shared map and transforms the estimated camera pose into a map-frame base pose using the calibrated camera geometry and robot kinematics. The resulting base pose is used by the policy and feedback controller.

Visual Query Localisation At execution time, MAVP localises the head-camera observations against the shared static map. Features on the robot arm move with the robot and therefore cannot serve as stable references in the map. We apply the robot-arm masking procedure described in Sec. III-A to the current stereo images and exclude features within the projected robot regions from localisation.

MAVP first retrieves candidate reference images from the shared map based on visual similarity. SuperPoint [27] detects salient local keypoints in both the query and reference images and computes a descriptor for each keypoint. LightGlue [24] then compares these keypoints and descriptors across the image pair to identify geometrically consistent feature matches. Because the matched reference features are already associated with 3D landmarks in M, each accepted image match links a 2D query keypoint to a corresponding 3D map point.

Perspective-n-Point (PnP) estimation with random sample consensus (RANSAC) recovers the head-camera pose from these 2D–3D matches. Pose estimates must meet inlier-count, reprojection-error, and stereo-consistency thresholds.

Map–Odometry Alignment and Base-Pose Estimation: MAVP uses cuVSLAM stereo visual odometry [28] to continuously track the head-camera motion in a local coordinate frame. cuVSLAM recovers metric geometry from stereo correspondences and estimates relative camera motion by tracking features across frames, producing a continuous local pose trajectory. Whenever map-based localisation provides the camera pose in the shared map, we align the odometry trajectory using paired local and map-frame poses.

Let L, M, C, and B denote the local tracking, shared map, head-camera, and mobile-base frames. The rigid transformation ${ \mathbf { } } ^ { X } { \mathbf { T } } _ { Y , t }$ maps coordinates from frame $Y$ to frame $X$ at time t. cuVSLAM provides $\mathbf { \Omega } ^ { L } \mathbf { T } _ { C , t }$ , and map localisation provides ${ ^ M \mathbf { T } } _ { C , t }$ at the corresponding timestamp.

Using these two corresponding poses, we compute the alignment $\mathbf { T } _ { \mathrm { a l i g n } , t }$ from the local tracking frame to the shared map frame as

$$
\mathbf { T } _ { \mathrm { a l i g n } , t } = { ^ { M } \mathbf { T } } _ { C , t } \left( { ^ { L } \mathbf { T } } _ { C , t } \right) ^ { - 1 } .\tag{2}
$$

Applying $\mathbf { T } _ { \mathrm { a l i g n } , t }$ transforms the continuous cuVSLAM trajectory into the shared map frame. Subsequent map localisations update this alignment, with smoothing to limit pose jumps.

The mobile-base pose is then obtained from the aligned camera pose using the camera-to-base transformation:

$$
{ ^ M } { \bf T } _ { B , t } = { \bf T } _ { \mathrm { a l i g n } , t } ^ { \mathrm { s m o o t h } L } { \bf T } _ { C , t } \left( { ^ B } { \bf T } _ { C , t } \right) ^ { - 1 } .\tag{3}
$$

Here, $\mathbf { T } _ { \mathrm { a l i g n } , t } ^ { \mathrm { s m o o t h } }$ denotes the smoothed local-to-map alignment, and $\mathbf { \Pi } ^ { B } \mathbf { T } _ { C , t } \mathbf { \Pi } ^ { \mathrm { ~ \tiny ~ \cup ~ } }$ represents the head-camera pose relative to the mobile base. The resulting ${ ^ { M } { \bf T } _ { B , t } }$ gives the base pose in the shared map frame. We extract its planar pose $( x , y , \psi )$ consisting of the map-frame position $( x , y )$ and yaw ψ, for use as policy input and in feedback control.

## C. Map-Aware Visuomotor Policies Learning

MAVP uses the shared map to provide a common spatial reference for the base-related components of the policy state and action. Let $\mathbf { s } _ { t }$ collect the visuomotor observations available at time $t ,$ and let $\mathbf { b } _ { t } = ( x _ { t } , y _ { t } , \psi _ { t } )$ denote the current base pose in the shared map frame obtained from Eq. 3. The policy $\pi _ { \theta } ,$ parameterised by $\theta ,$ predicts a sequence of $H$ future actions, where H is the action horizon:

$$
\widehat { \mathbf { A } } _ { t : t + H - 1 } = \pi _ { \theta } ( \mathbf { s } _ { t } , \mathbf { b } _ { t } ) ,\tag{4}
$$

where $\widehat { \mathbf { A } } _ { t : t + H - 1 }$ denotes the predicted action sequence from time t to $t + H - 1$ . Action components depend on the task and robot interface; base motion is represented by target poses in the shared map frame. We refer to tracking these base-pose targets as map-frame pose control.

The policy is trained by imitation learning from teleoperated demonstrations. Let $\mathcal { D } = \{ ( \mathbf { s } _ { t } , \mathbf { b } _ { t } , \mathbf { A } _ { t : t + H - 1 } ) \}$ denote the demonstration dataset. Imitation learning trains the policy to reproduce these demonstrated actions from the observed state by minimising

$$
\mathcal { L } _ { \mathrm { I L } } ( \theta ) = \mathbb { E } _ { \mathcal { D } } \left[ \ell ( \pi _ { \theta } ( \mathbf { s } _ { t } , \mathbf { b } _ { t } ) , \mathbf { A } _ { t : t + H - 1 } ) \right] ,\tag{5}
$$

where the expectation is over samples from D and ℓ denotes the policy-specific imitation-learning loss. Different policy families instantiate ℓ differently, while sharing the same objective of matching the demonstrated behaviour.

During execution, the current base pose is obtained from visual localisation and is therefore subject to estimation noise. To expose the policy to similar variations during training, we first estimate the localisation-noise distribution from offline localisation runs. We use the empirical standard deviations of planar position and yaw fluctuations to define a zero-mean Gaussian with covariance

$$
\Sigma = \mathrm { d i a g } ( \sigma _ { x } ^ { 2 } , \sigma _ { y } ^ { 2 } , \sigma _ { \psi } ^ { 2 } ) .
$$

Here, $\sigma _ { x } , \ \sigma _ { y } ,$ , and $\sigma _ { \psi }$ are the standard deviations of the noise in $x , y ,$ and yaw. For each training sample, we add an independent perturbation to the map-frame base pose:

$$
\begin{array} { r } { \epsilon _ { t } \sim { \mathcal { N } } ( \mathbf { 0 } , { \boldsymbol { \Sigma } } ) , \qquad \widetilde { \mathbf { b } } _ { t } = \mathbf { b } _ { t } + \epsilon _ { t } . } \end{array}\tag{6}
$$

Here, $\epsilon _ { t }$ denotes the sampled localisation perturbation and $\widetilde { \mathbf { b } } _ { t }$ the perturbed base pose. The demonstrated action sequence is kept unchanged, while the base-pose input $\mathbf { b } _ { t }$ is replaced by $\mathbf { b } _ { t }$ . The augmented imitation-learning objective is therefore

$$
\begin{array} { r } { \mathcal L _ { \mathrm { I L } } ^ { \mathrm { a u g } } ( \boldsymbol { \theta } ) = \mathbb E _ { \boldsymbol { \epsilon } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \Sigma ) } \left[ \ell \Big ( \pi _ { \boldsymbol { \theta } } ( \mathbf { s } _ { t } , \widetilde { \mathbf { b } } _ { t } ) , \mathbf { A } _ { t : t + H - 1 } \Big ) \right] , } \end{array}\tag{7}
$$

This augmentation encourages the policy to produce consistent actions under small localisation errors.

## IV. EMPIRICAL EVALUATION

We evaluate MAVP on six real-world mobile manipulation tasks. We test whether a shared spatial reference improves execution, which mapping, policy, and control components contribute, and whether ongoing map localisation improves repeated task execution. We organise the evaluation around the following questions:

Q1: Does map-frame pose control improve task success compared with unanchored velocity and episode-initialised odometry pose control?

Q2: With the same base-pose input, how do map-frame pose targets compare with base velocities?

Q3: What is the contribution of explicitly providing the current map-frame base pose to the policy?

Q4: Does pose-noise augmentation improve tolerance to errors in the policy’s pose input?

Q5: How does pose-error feedback affect task success and base-target tracking?

Q6: Which map-construction components support complete maps and reliable held-out localisation?

Q7: Does MAVP improve task success across different visuomotor policy families?

Q8: Does ongoing map localisation improve repeated task execution compared with initial alignment only?

## A. Common Experimental Setup

## Physical setup:

The robot has a fourwheel omnidirectional base, a lifting and rotating torso, and two 7-DoF arms with independent grippers. Base commands specify planar velocity $( v _ { x } , v _ { y } )$ and yaw rate. Arm commands specify joint angles and gripper states. The policy receives images from a head-mounted RGB camera and two wrist RGB cameras. Figure 4 shows the era placements.

![](images/c64e035e76d4c56532267165c2696bab81527da21082da1ddceb7a755d1ece27.jpg)  
Fig. 4: Configuration with one headmounted and two wrist cameras.

![](images/a6f91f8cd8dcba9eadafa1bbad3a7d98efd10a6b3ca49baf8417c1c2d3b8f8f2.jpg)

![](images/93640f23db2c0b344614f98af38fb3e1dc96222697714a98077cb2c431f0297b.jpg)

![](images/dcdeda3948300c3e18967d45f5a7200706ef5d7c7239aa6574944e8c27ff3231.jpg)

TABLE I: Six mobile manipulation tasks; success requires completing every listed step.
<table><tr><td>Task</td><td>Description</td></tr><tr><td>Drawer Packing</td><td>Pick up the eraser, open the drawer, put the eraser and two toys inside, and close the drawer.</td></tr><tr><td>Disassemble and Deliver</td><td>Remove the peg, move to the box, and place all parts inside.</td></tr><tr><td>Conveyor Picking Lidded Box Packing</td><td>Track and grasp an object on a moving conveyor. Approach and open the box, place an object inside,</td></tr><tr><td>Bag Packing</td><td>and close the lid. Take a bag from the rack, put the table objects</td></tr><tr><td>Dual-Drawer Return</td><td>inside, and hang the bag back on the rack. Return the two objects outside the drawers to their</td></tr></table>

TABLE II: Success rates (%) for the control variants. Pose denotes the map-frame base-pose input; aug. denotes training pose-noise augmentation.
<table><tr><td>Method / variant</td><td>Disassemble and Deliver</td><td>Lidded Box Packing</td></tr><tr><td>Unanchored velocity</td><td>38%</td><td>44%</td></tr><tr><td>Odometry pose</td><td>16%</td><td>16%</td></tr><tr><td>Map-frame pose (w/o pose)</td><td>74%</td><td>58%</td></tr><tr><td>Velocity (w/ pose)</td><td>60%</td><td>64%</td></tr><tr><td>Map-frame pose (w/ pose)</td><td>84%</td><td>76%</td></tr><tr><td>Map-frame pose (w/ pose + aug.)</td><td>90%</td><td>88%</td></tr></table>

robot configuration and cam-

Figure 1 shows Drawer

Packing, Disassemble and

Deliver, Lidded Box Packing, Conveyor Picking, Dual-Drawer Return, and Bag Packing in panels A to F. Insets provide close-up or additional views. Table I gives the task sequences and success criteria.

Training setup: We instantiate MAVP with three visuomotor policy families: ACT, DP, and FM. The controlled ablations in Q1–Q5 and the ongoing-localisation experiment in Q8 use ACT; Q7 compares all three families. We train all policies for 30,000 optimisation steps using automatic mixed precision and eight data-loader workers. ACT uses a batch size of 32 and a learning rate of $3 \times 1 0 ^ { - 5 }$ , with one observation step and an action chunk of 30 steps. The DiTbased implementations [29] used for Diffusion Policy (DP) and Flow Matching (FM) use a batch size of 64 and a learning rate of $2 \times 1 0 ^ { - 5 }$ , with two observation steps, a prediction horizon of 32 steps, and 24 action steps.

Evaluation metrics: A demonstration or policy rollout succeeds when it completes every step of the corresponding task. Success rate is the percentage of successful rollouts. Each task has 50 teleoperated demonstrations, and each evaluated policy receives 50 rollouts per task.

## B. Map-Frame Pose Control (Q1, Q2, and Q3)

Experimental setup: Table II compares six visuomotor policy variants on Disassemble and Deliver and Lidded Box Packing. All variants receive arm joint and gripper state. In variant labels, pose denotes the current map-frame base pose and aug. denotes pose-noise augmentation during training.

Unanchored velocity predicts base velocities without a map or base-pose input. Odometry pose predicts pose targets in a frame initialised at the start of each episode. Its controller uses live odometry to execute those targets. Map-frame pose (w/ pose) receives the current map-frame base pose and predicts targets in that frame.

(a) Disasse (a) Disassemble and Deliver  
Position labels Median gap (cm)  
![](images/1fcbb7f8556074a6484dd9c020dc7f4a9e67afd8cdb000e97fc002c3af65eab1.jpg)

Velocity labels Pairs (%)  
![](images/53d0c385e5d3923f226f316f8d6ced705e067f6b1c26b2ed303393e3fd6bb7ea.jpg)

(b) Box pa  (b) Lidded Box Packing  
![](images/59af04509138045f4aa7b8750dd0568ae020ea0aa52648828bdcdc129436de36.jpg)

![](images/4c6b874853b61765a0edd2a0e576bb6a54fd6e9e844e7952894c1c7a92dc5007.jpg)  
<sup>Median</sup> <sup>gap</sup> <sup>(cm)</sup>  Fig. 5: Training-label variation for (a) Disassemble and Deliver and (b) Lidded Box Packing. Panels compare median position gaps across frames and fractions of nearby-state pairs with differing velocity or stop/move labels.  
Fig. 6: Base trajectories for Lidded Box Packing (left) and Disassemble and Deliver (right). Velocity control (blue) stops early; map-frame pose control with training augmentation (yellow) continues toward task locations.

Map-frame pose (w/o pose) removes the base-pose input while retaining map-based localisation and base-pose targets. Velocity (w/ pose) retains this input and predicts velocity commands. Map-frame pose (w/ pose + aug.) adds the training augmentation from Eq. 6. This perturbs the pose input, preserving images, joint/gripper states, and action targets.

Figure 5 uses samples from the demonstration datasets that train the two task policies. The position-label plots report median pairwise position gaps among nearby demonstration states when expressed in three coordinate frames: raw odometry (episode-local), aligned odometry (odometry rigidly transformed into the shared map frame via $\mathbf { T } _ { \mathrm { a l i g n } } )$ , and map coordinates (absolute poses from visual localisation). The gap measures label inconsistency across demonstrations. The velocity-label plots compare target velocities at nearby robot states. Together, they characterise training-label variation.

Results: Table II shows complementary benefits from mapframe targets, explicit pose inputs, and training augmentation. The complete configuration performs best on both tasks, while the controlled comparisons below distinguish the roles of its components. For Q1, map-frame pose control improves task completion over both unanchored velocity and episode-initialised odometry pose control. Its spatial targets give the controller an objective to track in the same frame used for demonstration supervision, so execution can respond to deviations from a target instead of relying only on predicted velocity commands. Figure 6 contrasts velocitycontrolled examples that stop short with augmented mapframe pose examples that continue towards manipulation locations; Figure 7 shows task progression.

Lidded Box Packing  
![](images/7e4fe73d7ad7286ee274a30c0871533ebdd5736126b891a4cb386b14d493b616.jpg)  
Fig. 7: Numbered execution snapshots for Lidded Box Packing and Disassemble and Deliver.

For Q2, Map-frame pose (w/ pose) outperforms Velocity (w/ pose) on both tasks without training augmentation. Both policies receive the current map-frame pose, so the difference concerns the action representation and its execution interface. A pose target specifies where the base should move, leaving the controller to determine the velocity needed to reach it. Velocity supervision also reflects demonstration timing, including acceleration, pauses, and the decision to stop; the nearby-state label variation in Fig. 5 illustrates this dependence. This explains the advantage of spatial targets with feedback.

For Q3, the map-frame pose variant with pose input outperforms the variant without it, both trained without augmentation. The output representation is unchanged; the added input locates the base in the target frame, reducing the need to infer its location and heading from vision alone.

TABLE III: Success rates (%) under Gaussian pose-input noise, with and without training augmentation.
<table><tr><td>Policy variant</td><td>No added noise</td><td>Low</td><td>Medium</td><td>High</td></tr><tr><td>Disassemble and Deliver</td><td></td><td></td><td></td><td></td></tr><tr><td>Map-frame pose (w/ pose)</td><td>84%</td><td>76%</td><td>68%</td><td>64%</td></tr><tr><td>Map-frame pose (w/ pose + aug.)</td><td>90%</td><td>88%</td><td>78%</td><td>68%</td></tr><tr><td>Lidded Box Packing</td><td></td><td></td><td></td><td></td></tr><tr><td>Map-frame pose (w/ pose)</td><td>76%</td><td>72%</td><td>68%</td><td>62%</td></tr><tr><td>Map-frame pose (w/ pose + aug.)</td><td>88%</td><td>90%</td><td>86%</td><td>76%</td></tr></table>

![](images/c5099632aabaefb64e20a863323ef9005c0f9ea9241b6b3ef312546f59b48912.jpg)

![](images/6e4e8e6abeeb72a22e0405ce5fab56ce0d158e0b31ae1a2fb92db999a9d5b60a.jpg)  
Fig. 8: Offline base-target changes under pose-input noise: ∆p is planar displacement and ∆ψ is wrapped yaw change. Open markers show recording means; solid curves weight recordings equally.

## C. Robustness to Localisation Noise (Q4)

Experimental setup: Localisation errors affect the base pose supplied to the policy. Table III tests whether pose-noise augmentation improves tolerance to these errors. For each task, we compare map-frame pose policies trained with and without the augmentation in Eq. 6. At each policy inference, we add independent Gaussian noise to the pose input. The tested noise levels and their standard deviations $( \sigma _ { x } = \sigma _ { y } , \sigma _ { \psi } )$ are no added noise (0 cm, 0<sup>◦</sup>), low $( 1 \mathrm { c m } , 0 . 5 ^ { \circ } )$ , medium (2 cm, 1<sup>◦</sup>), and high (4 cm, 2<sup>◦</sup>). The controller uses the original localisation estimate. Entries report task success rates, with the no-added-noise results taken from Table II.

Results: Table III shows that pose-noise augmentation improves success at every tested perturbation level on both tasks. Increasing the perturbation generally makes execution less reliable, but augmentation retains an advantage despite variation between adjacent noise levels. The benefit is therefore not confined to the unperturbed evaluation condition.

This behaviour is consistent with the training objective: noisy pose inputs are paired with unchanged action targets, encouraging appropriate actions under pose noise.

Pose-input sensitivity: To complement Table III, we perturb the map-frame $( x , y , \psi )$ input offline at the same noise levels while holding reconstructed RGB and proprioceptive observations fixed. Both checkpoints receive identical observations and perturbation samples. We measure changes relative to each checkpoint’s own unperturbed base-target prediction at a nominal 0.267 s horizon. Figure 8 shows smaller mean position and yaw changes for w/ pose + aug. at every nonzero noise level, consistent with the task-success results.

## D. Contribution of Pose-Error Feedback (Q5)

Experimental setup: We compare FF only and FF + FB mapframe pose control on Conveyor Picking (Figure 9). Both use the same checkpoint trained with pose-noise augmentation;

![](images/adeebc05f4818115ec0ba10d76dba4df3e8eead5172c7e8cd18774f67519cec6.jpg)

![](images/0f1c9e5ecb64d910706e9915e143f2e74e6ea5f9467eb84383f5eb296afdda69.jpg)  
Fig. 9: Feedback ablation on Conveyor Picking: success and time-weighted tracking RMSE over 50 rollouts per variant using one augmented policy checkpoint. Feedback improves success and positional tracking, with little change in yaw RMSE.

FF denotes feedforward and FB denotes pose-error feedback. FF only converts predicted pose-target changes into base velocities; FF + FB corrects the error between the target pose and current localisation estimate.

Metrics: We report task success rate and time-weighted positional/yaw tracking RMSE (planar distance and wrapped yaw error between estimated base pose and active target, using localisation estimates).

Results: Figure 9 shows that adding pose-error feedback improves task success and positional tracking, while yaw tracking changes little. With the policy checkpoint fixed, this gain comes from target execution. Feedforward follows changes in those targets but does not respond to an offset between the target and the estimated base pose. Feedback corrects this offset to reach suitable picking positions.

## E. Geometric Map Construction Ablation (Q6)

Shared protocol: Table IV evaluates the map-construction pipeline across the six manipulation tasks. For each task, the data used for map construction and localisation queries are disjoint. All variants use the same held-out stereo queries, reconstruction budget, calibration, feature extraction, and localisation thresholds. VGGT4D-based dynamic-region masking is retained in all variants. We ablate the remaining geometric components by removing robot-arm masking, restricting matching to temporal neighbours, removing temporalneighbour matches, or replacing LightGlue with nearestneighbour matching.

Metrics: A complete map is a single connected reconstruction that registers all selected reference images and passes the geometric quality checks. A stereo query is considered consistent when both views localise independently against the largest reconstructed component and their estimated poses agree within 3 cm and 2<sup>◦</sup>. Deployable pairs additionally require the corresponding task map to be complete. Relative cost is the median ratio of feature-matching and reconstruction time to the full mapping pipeline. Stereo consistency measures agreement between the two views rather than absolute pose accuracy.

Results: The full mapping pipeline reconstructs complete maps for all six tasks. Temporal-only matching fails to produce a complete map, while the other ablations also reduce the number of deployable localisation queries. Some queries can still localise consistently within the largest component of an incomplete reconstruction, explaining why consistent-pair counts can remain high even when deployable-pair counts decrease.

TABLE IV: Geometric mapping results across the six manipulation tasks. Deployable pairs require a complete task map, and relative cost includes feature matching and reconstruction.
<table><tr><td>Variant</td><td>Complete maps</td><td>Consistent pairs</td><td>Deployable pairs</td><td>Cost (rel.)</td></tr><tr><td>Full mapping pipeline</td><td>6/6</td><td>911/912</td><td>911/912</td><td>1.00</td></tr><tr><td>Without arm mask</td><td>3/6</td><td>908/912</td><td>600/912</td><td>1.58</td></tr><tr><td>Temporal-only matching</td><td>0/6</td><td>892/912</td><td>0/912</td><td>0.31</td></tr><tr><td>Without temporal neighbours</td><td>5/6</td><td>891/912</td><td>780/912</td><td>0.94</td></tr><tr><td>Nearest-neighbour matching</td><td>4/6</td><td>878/912</td><td>708/912</td><td>0.55</td></tr></table>

TABLE V: Success rates for ACT, DP, and FM over 50 rollouts per task. V: unanchored velocity without pose input; P: map-frame pose targets with pose input and localisation feedback.
<table><tr><td rowspan="2">Task</td><td colspan="2">ACT</td><td colspan="2">DP</td><td colspan="2">FM</td></tr><tr><td>V</td><td>P</td><td>V</td><td></td><td></td><td>V</td></tr><tr><td>Disassemble and Deliver</td><td>38%</td><td>90%</td><td>42%</td><td>86%</td><td>32%</td><td>88%</td></tr><tr><td>Drawer Packing</td><td>20%</td><td>64%</td><td>44%</td><td>62%</td><td>38%</td><td>66%</td></tr><tr><td>Conveyor Picking</td><td>42%</td><td>66%</td><td>38%</td><td>74%</td><td>36%</td><td>76%</td></tr><tr><td>Lidded Box Packing</td><td>44%</td><td>88%</td><td>52%</td><td>56%</td><td>50%</td><td>60%</td></tr><tr><td>Bag Packing</td><td>18%</td><td>66%</td><td>28%</td><td>68%</td><td>32%</td><td>70%</td></tr><tr><td>Dual-Drawer Return</td><td>22%</td><td>68%</td><td>30%</td><td>70%</td><td>36%</td><td>74%</td></tr></table>

## F. Policy Architectures (Q7)

Experimental setup: Table V compares unanchored velocity control with map-frame pose control. Within each family, demonstrations, RGB and joint/gripper states, arm/gripper action representations, rollout counts, and success criteria are fixed. Unanchored velocity omits the map-frame basepose input. Map-frame pose control includes it and tracks pose targets with feedback. ACT’s two configurations match Unanchored velocity/Map-frame pose (w/ pose + aug.) in Table II. ACT uses action chunks [30], DP iterative denoising [31], and FM a learned vector field [32].

Results: Table V shows that map-frame pose control improves success across every evaluated task and policy family. The common trend across ACT, DP, and FM supports the use of MAVP with different action-generation mechanisms. Each policy learns shared-map targets tracked using the current pose estimate, whether generated by a chunked transformer, iterative denoising, or a learned flow. This comparison extends the evidence across policy architectures, evaluating pose inputs, map-frame targets, and localisation feedback jointly; preceding ablations examine the individual choices.

## G. Ongoing Map Localisation (Q8)

Experimental setup: Table VI compares initial alignment only with ongoing map localisation on Disassemble and Deliver and Lidded Box Packing. We reuse the same posenoise-augmented map-frame pose checkpoint and FF + FB controller from the Q1–Q5 ablations; the ongoing-localisation numbers are taken from Map-frame pose (w/ pose + aug.) in Table II. We evaluate 50 initial-only trials per task using the corresponding deployment settings and task-completion criteria.

Initial alignment only fixes the local-to-map alignment after the first accepted map localisation in each deployment session and disables subsequent map queries. Local odometry and camera-to-base kinematics continue to update the pose supplied to both the policy and controller. Restarting the policy between trials retains this alignment and odometry state. Ongoing map localisation instead updates the alignment using accepted map localisations during execution. Policy timing, action chunks, target lookahead, controller parameters, velocity limits, and task time limits follow the corresponding baseline settings, with no additional localisation noise. A local-tracking reset ends the initial-only session rather than triggering map-based recovery.

TABLE VI: Task success (%) with session-level initial alignment retained across policy restarts versus ongoing map localisation. Ongoing-localisation results are reused from Table II.
<table><tr><td>Task</td><td>Initial alignment only</td><td>Ongoing map localisation</td></tr><tr><td>Disassemble and Deliver</td><td>64%</td><td>90%</td></tr><tr><td>Lidded Box Packing</td><td>52%</td><td>88%</td></tr></table>

Results. Table VI shows that ongoing map localisation achieves higher task success on both tasks. This result indicates that continued localisation helps maintain a consistent spatial reference for the policy and controller. Accepted map updates correct the local-to-map alignment, keeping policy pose inputs and controller feedback referenced to the same map frame as the predicted targets.

## V. CONCLUSION AND FUTURE WORK

We presented MAVP, a map-aware visuomotor framework that gives mobile manipulation policies a shared spatial reference for learning and execution. MAVP reconstructs a map from teleoperated demonstrations, expresses demonstrated and predicted base poses in this common frame, and couples learned base-pose targets with localisationbased feedback control. This formulation separates where the base should move from the low-level velocities required to reach that pose, while explicit pose conditioning and posenoise augmentation improve robustness to spatial uncertainty. Across real-world tasks and multiple visuomotor policy families, the results consistently support shared-map pose targets over unanchored base control and demonstrate the value of feedback and continued localisation during execution. Future work will investigate more scalable map construction and localisation, richer uncertainty-aware policy and control mechanisms, and adaptation across diverse scenes and task configurations. Extending MAVP to larger workspaces, longerhorizon behaviours, and more general whole-body mobile manipulation offers a natural path towards reliable deployment in increasingly complex environments.

## REFERENCES

[1] W. Zhi, T. Lai, L. Ott, and F. Ramos, “Diffeomorphic Transforms for Generalised Imitation Learning,” in Proceedings of the 4th Annual Learning for Dynamics and Control Conference, ser. Proceedings of Machine Learning Research, vol. 168. PMLR, 2022, pp. 508–519.

[2] W. Zhi, T. Lai, L. Ott, E. V. Bonilla, and F. Ramos, “Learning Efficient and Robust Ordinary Differential Equations via Invertible Neural Networks,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 27 060–27 074.

[3] W. Zhi, T. Zhang, and M. Johnson-Roberson, “Instructing Robots by Sketching: Learning from Demonstration via Probabilistic Diagrammatic Teaching,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 15 047–15 053.

[4] Z. Fu, T. Z. Zhao, and C. Finn, “Mobile ALOHA: Learning Bimanual Mobile Manipulation using Low-Cost Whole-Body Teleoperation,” in Proceedings of the 8th Conference on Robot Learning, 2024.

[5] J. Yang, I. Huang, B. Vu, M. Bajracharya, R. Antonova, and J. Bohg, “Mobi-π: Mobilizing Your Robot Learning Policy,” in Proceedings of the 9th Conference on Robot Learning, 2025.

[6] P. Sundaresan, R. Malhotra, P. Miao, J. Yang, J. Wu, H. Hu, R. Antonova, F. Engelmann, D. Sadigh, and J. Bohg, “HoMeR: Learning In-the-Wild Mobile Manipulation via Hybrid Imitation and Whole-Body Control,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2026.

[7] J. Wu, X. Sun, A. Zeng, S. Song, J. Lee, S. Rusinkiewicz, and T. Funkhouser, “Spatial action maps for mobile manipulation,” in Proceedings of Robotics: Science and Systems (RSS), 2020.

[8] S. Kim, W. Chung, Z. Dai, D. Bhatt, A. Shukla, H. Su, Y. Tian, and N. Atanasov, “Seeing the Bigger Picture: 3D Latent Mapping for Mobile Manipulation Policy Learning,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2026.

[9] J. Gu, D. S. Chaplot, H. Su, and J. Malik, “Multi-skill mobile manipulation for object rearrangement,” in International Conference on Learning Representations (ICLR), 2023.

[10] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song, “UMI-on-Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers,” in Proceedings of the 8th Conference on Robot Learning, 2025.

[11] X. Xu, J. Park, H. Zhang, E. Cousineau, A. Bhat, J. Barreiros, D. Wang, J. Bohg, and S. Song, “HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations,” in Proceedings of Robotics: Science and Systems (RSS), 2026.

[12] Q. Sun, W. Zhi, T. Zhang, and M. Johnson-Roberson, “Teaching Robots Where To Go And How To Act With Human Sketches via Spatial Diagrammatic Instructions,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024, pp. 1094–1100.

[13] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3D diffusion policy: Generalizable visuomotor policy learning via simple 3D representations,” in Proceedings of Robotics: Science and Systems (RSS), 2024.

[14] L. X. Shi, A. Sharma, T. Z. Zhao, and C. Finn, “Waypoint-Based Imitation Learning for Robotic Manipulation,” in Proceedings of the 7th Conference on Robot Learning, 2023.

[15] S. Belkhale, Y. Cui, and D. Sadigh, “HYDRA: Hybrid Robot Actions for Imitation Learning,” in Proceedings of the 7th Conference on Robot Learning, 2023.

[16] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.

[17] M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, “DART: Noise Injection for Robust Imitation Learning,” in Proceedings of the 1st Conference on Robot Learning (CoRL), 2017.

[18] J. L. Schönberger and J.-M. Frahm, “Structure-from-Motion Revisited,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113.

[19] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, “From Coarse to Fine: Robust Hierarchical Localization at Large Scale,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 12 716–12 725.

[20] B. Bescos, J. M. Fácil, J. Civera, and J. Neira, “DynaSLAM: Tracking, Mapping, and Inpainting in Dynamic Scenes,” IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 4076–4083, 2018.

[21] L. Goli, S. Sabour, M. Matthews et al., “RoMo: Robust Motion Segmentation Improves Structure from Motion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[22] Y. Hu, C. Cheng, S. Yu, X. Guo, and H. Wang, “VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2026, pp. 414–424.

[23] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Super-Glue: Learning Feature Matching with Graph Neural Networks,” in

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[24] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local Feature Matching at Light Speed,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 17 627– 17 638.

[25] W. Zhi, H. Tang, T. Zhang, and M. Johnson-Roberson, “Unifying Representation and Calibration With 3D Foundation Models,” IEEE Robotics and Automation Letters, 2024.

[26] H. Tang, T. Zhang, M. Johnson-Roberson, and W. Zhi, “Bi-Manual Joint Camera Calibration and Scene Representation,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2026.

[27] D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperPoint: Self-Supervised Interest Point Detection and Description,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018, pp. 224–236.

[28] A. Korovko, D. Slepichev, A. Efitorov, A. Dzhumamuratova, V. Kuznetsov, H. Rabeti, J. Biswas, and S. Pouya, “cuVSLAM: CUDA Accelerated Visual Odometry and Mapping,” arXiv preprint arXiv:2506.04359, 2025.

[29] W. Peebles and S. Xie, “Scalable Diffusion Models with Transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[30] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[31] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[32] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow Matching for Generative Modeling,” arXiv preprint arXiv:2210.02747, 2022.