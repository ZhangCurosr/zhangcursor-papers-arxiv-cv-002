# KinoUAV: A Benchmark for Safe and Stable UAV Vision-Language Navigation with Kinodynamic Modeling

Chenyao Yang, Peijin Wang, Yingchao Feng, and Wenhui Diao

Abstract—Unmanned aerial vehicles (UAVs) have recently emerged as important remote sensing platforms, and UAV navigation has shown great potential in remote sensing applications. However, existing methods do not explicitly incorporate kinodynamic modeling or safety distances into trajectory generation, resulting in high absolute yaw rates, frequent collisions, and low navigation success rates, which limit safety and motion stability. To address this challenge, we introduce KinoUAV, a kinodynamic benchmark comprising a dataset and KinoUAV-Agent. First, we develop a hierarchical kinodynamic trajectory generation pipeline that drives UAV through acceleration actions. It uses waypoint chains from global planning and generates trajectories segment by segment under velocity, acceleration, and jerk constraints, reducing oscillations and improving trajectory stability. Furthermore, we introduce directional freespace distance (DFD) labels into the constructed dataset to describe the available clearance for safe motion decisions. The proposed KinoUAV-Agent predicts DFDs and uses them to generate safe actions, decoupling perception from action decisionmaking. This design enables autonomous obstacle avoidance and improves navigation safety. The benchmark incorporates the proposed Cruise-Seek task, in which the UAV first enters the target region and then searches for the target, reducing the difficulty of target search and improving navigation success. Experiments on KinoUAV show that KinoUAV-Agent outperforms existing methods, achieving the highest success rate while maintaining lower absolute yaw rates and collision rates. These results demonstrate the effectiveness of kinodynamic modeling for safe and stable UAV navigation. The code will be released at https://github.com/CabbageYang15596/KinoUAV

Index Terms—Vision-language navigation, remote sensing, kinodynamic trajectory generation, autonomous obstacle avoidance.

## I. INTRODUCTION

As important remote sensing platforms, UAVs are widely used in agricultural monitoring, infrastructure inspection, and disaster assessment [1], [2]. Through autonomous navigation, UAVs can freely approach and observe specific targets, thereby demonstrating immense potential in the field of remote sensing. To achieve autonomous navigation in complex environments, Vision-Language Navigation (VLN) has emerged as an important research direction in embodied intelligence. Early VLN studies mainly focused on ground-based agents [3], [4], [5], [6], [7], [8], [9], [10]. With the development of UAV simulation platforms, researchers have extended VLN to aerial environments [11], [12], [13], [14], greatly improving the scale of scenes and the amount of data.

![](images/70341ff7b5248493228826d3d1cc316e46fc867d281dab5345f408d773593682.jpg)  
Fig. 1. Motivation of KinoUAV. Kinodynamic modeling is introduced into the dataset construction pipeline. Acceleration-driven control is used to generate stable UAV trajectories. Motion states, DFDs, front and bottom-view images are collected at each frame. The blue region highlights the data uniquely provided by KinoUAV.

However, existing UAV VLN benchmarks still suffer from several limitations in dataset construction. First, trajectories are commonly generated using discrete actions, waypoint interpolation, or loss-based motion constraints [13], [12], [15], without explicitly integrating kinodynamic modeling into the generation. Such trajectories may exhibit abrupt motion changes and limited control precision, making them unsuitable for learning stable UAV motion. Second, existing datasets lack effective supervision for autonomous obstacle avoidance. Current datasets often introduce additional information, such as multi-source observations, known obstacle positions, or explicit instruction cues for collision avoidance [16], [17], [13]. The substantial overhead of data collection and processing limits autonomous perception. Third, data containing detailed path instructions weaken autonomous exploration [13], while target descriptions alone lead to inefficient search in largescale environments [14]. Both types of data fail to provide

This work was supported in part by the National Key R&D Program of China under Grant 2024YFF1401001, in part by the National Natural Science Foundation of China under Grant 62301538, in part by the Science and Disruptive Technology Program under Grant 2025-AIRCAS-SDTP-04, and in part by the Youth Deployment Project of the Chinese Academy of Sciences under Grant GFQN-07. (Corresponding author: Peijin Wang.)

Chenyao Yang and Peijin Wang are with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing 100190, China; the University of Chinese Academy of Sciences, Beijing 100190, China; the School of Electronic, Electrical and Communication Engineering, University of Chinese Academy of Sciences, Beijing 100190, China; and the Key Laboratory of Target Cognition and Application Technology, Beijing 100190, China (e-mail: wangpeijin17@mails.ucas.ac.cn).

guidance aligned with the current navigation stage, increasing the risk of missed targets and navigation failure.

To address these issues, we propose KinoUAV, a UAV VLN benchmark built with kinodynamic modeling. First, we design a trajectory generation pipeline based on hierarchical kinodynamic $\mathbf { A } ^ { * }$ . The proposed method performs path search according to different waypoint chains and uses acceleration actions to control UAV motion. It generates stable flight trajectories segment by segment under velocity, acceleration and jerk constraints. We then collect images and motion states at each sampling point. To support safe navigation, we introduce forward and downward DFDs to quantify the obstacle-free ranges ahead of and below the UAV. These DFDs serve as intermediate supervision, enabling the model to perceive environmental risks before making action decisions. This design decouples environmental perception from action prediction.

Based on this data generation pipeline, we construct Main Subset and Auxiliary Subset: Auxiliary Subset uses randomly generated waypoint chains to plan trajectories, strengthening the UAV’s waypoint tracking capability. The KinoUAV Dataset generates trajectories based on the start and target positions, enabling complete navigation capability. It is collected under the Cruise-Seek task setting to balance navigation guidance and autonomous exploration. During the Cruise stage, only target-region information is provided to guide the UAV for region-level navigation. After entering the target region, the task switches to the Seek stage, where target descriptions are provided for accurate target localization. This design avoids excessive path guidance and large-scale blind exploration.

Furthermore, we propose KinoUAV-Agent, which adopts a hierarchical navigation architecture. The high-level waypoint planner predicts the next waypoint during both the Cruise and Seek stages. The low-level motion controller uses estimated DFDs for autonomous obstacle avoidance and predicts acceleration actions for waypoint tracking.

Beyond existing VLN metrics, KinoUAV introduces absolute yaw rate to evaluate motion stability. Experimental results show that existing methods still face significant challenges in UAV VLN tasks with kinodynamic constraints. Meanwhile, KinoUAV-Agent achieves the highest success rate of 89.3% while obtaining the lower collision rate and yaw rate, demonstrating the importance of kinodynamically modeled data and comprehensive evaluation protocols for advancing safe and stable UAV VLN research. KinoUAV provides a unified dataset and evaluation standard for future research. Our main contributions are summarized as follows:

• KinoUAV is among the first UAV VLN benchmarks to introduce a kinodynamic pipeline for dataset construction, enabling acceleration-driven trajectory generation.

• DFDs provide intermediate supervision for obstacle avoidance using only RGB images, decoupling perception from action prediction.

• The Cruise-Seek task combines region-level navigation with target-level search, balancing navigation guidance and autonomous exploration.

## II. RELATED WORK

## A. Kinodynamic Motion Planning

Motion planning aims to generate collision-free paths from map information. $\mathbf { A } ^ { * }$ [18] provides a classical heuristic search framework that combines the accumulated path cost with an estimate of the remaining cost. However, geometric connectivity alone does not guarantee that a motion can be executed. Hybrid $\mathbf { A } ^ { * }$ [19] retains continuous states within discrete search nodes and uses orientation-dependent state transitions to satisfy nonholonomic kinematic constraints. Beyond kinematic feasibility, Donald et al. [20] formulated kinodynamic planning under motion constraints, including limits on velocity and acceleration. State lattice planning [21] constructs a graph of feasible motion primitives to maintain state continuity and accommodate the vehicle’s motion model. For aerial robots, Zhou et al. [22] combined kinodynamic search with Bspline optimization and time adjustment to improve trajectory smoothness, obstacle avoidance, and dynamic feasibility. iDb-$\mathbf { A } ^ { * }$ [23] alternates between motion-primitive search and trajectory optimization, allowing bounded discontinuities during search and repairing them during optimization. Following this principle, iDb-RRT [24] combines sampling-based exploration with motion primitives and trajectory optimization. These methods provide a foundation for generating executable trajectories.

## B. VLN Datasets

VLN benchmarks cover ground and aerial navigation, with tasks ranging from path following and object-oriented exploration to interactive navigation. Room-to-Room (R2R) [3] established a benchmark for indoor VLN using human instructions and navigation trajectories in the Matterport3D simulation environment. Building on R2R, Room-for-Room (R4R) [25] concatenates paths to form longer routes and introduces instruction fidelity, while Room-Across-Room (RxR) [4] expands multilingual coverage and provides dense spatiotemporal grounding between instructions and demonstrations. REVERIE [6] moves beyond detailed route instructions by requiring agents to navigate to and identify a remotely described target object. SOON [26] introduces scenario-oriented object search from arbitrary starting positions. To support interactive navigation, Cooperative Visionand-Dialog Navigation (CVDN) [8] collects human dialogues in which navigators request assistance from an oracle. Outdoor ground navigation is explored by LANI [27], which contains landmark-based instructions in synthetic environments, and Touchdown [7], which combines street-level navigation with spatial grounding in real urban imagery. VLN-CE [5] transfers instruction following to continuous environments with lowlevel action control, addressing the simplifying assumptions of navigation graphs. Related embodied benchmarks, such as ALFRED [9], further combine navigation with object interaction and household task execution. These datasets offer diverse settings for ground VLN, but aerial navigation also requires reasoning about altitude changes and more complex three-dimensional spaces. AVDN [10] extends dialogue-guided navigation to an aerial simulator built on xView and collects asynchronous dialogues between commanders and followers. AerialVLN [11] provides an AirSim- and UE4-based platform for path following in three-dimensional aerial environments, using manually collected trajectories and route instructions. CityNav [28] improves environmental realism with real-world point clouds from Cambridge and Birmingham and uses human flight trajectories to construct high-altitude UAV navigation tasks grounded in real-world data. TravelUAV [12] uses the OpenUAV platform to collect and annotate human flight trajectories, supporting navigation with target-direction information and assistance at different guidance levels. To reduce data collection costs, OpenFly [13] integrates pointcloud acquisition, semantic annotation, A\*-based trajectory generation, and automatic instruction generation into a pipeline for producing navigation data at scale. Together, these datasets have advanced VLN through varied task settings, simulated or real-world scenes, manually or automatically constructed trajectories, and diverse language annotations. Beyond these datasets, KinoUAV incorporates kinodynamic modeling into UAV navigation trajectory generation and records motion states to support learning safe and stable aerial navigation.

## C. VLN Methods

VLN methods have progressed from recurrent instructionfollowing policies to pretrained multimodal models, structured spatial representations, and large-model-based navigation. Early ground-based approaches mainly improve language grounding and generalization. Speaker-Follower [29] uses a speaker model to generate additional instructions and evaluate candidate routes, while EnvDrop [30] combines back translation with environmental dropout to improve transfer to unseen environments. Reinforced Cross-Modal Matching (RCM) [31] introduces local and global instruction-trajectory matching with self-supervised imitation learning. To track execution progress, the Self-Monitoring agent [32] estimates instruction completion, and the Regretful Agent [33] uses progress estimation to support backtracking and action revision. Subsequent studies strengthen cross-modal representations through pretraining. PREVALENT [34] learns representations from image-text-action triplets, whereas VLN-BERT [35] transfers visual grounding knowledge from web image-text pairs to instruction-path matching. Recurrent VLN-BERT [36] introduces a recurrent state into a multimodal Transformer for sequential decisions, and HAMT [37] explicitly encodes long-horizon observations for history-aware action prediction. Beyond unstructured temporal memory, mapbased methods model the spatial relationships needed for exploration. DUET [38] combines global topological planning with local cross-modal grounding through a dual-scale graph Transformer. Cross-Modal Map Learning [39] predicts language-conditioned semantic maps and waypoint paths, while BEVBert [40] integrates metric and topological maps through multimodal map pretraining. GridMM [41] organizes historical observations into a growing egocentric grid memory, and ETPNav [42] combines evolving topological maps, crossmodal planning, and obstacle-aware execution in continuous environments. With the development of large language and vision-language models, recent methods increasingly exploit explicit reasoning and general visual knowledge. NavGPT [43] performs zero-shot navigation through language-based rea soning, and NavGPT-2 [44] integrates large-model reasoning with a learned navigation policy. MapGPT [45] introduces map-guided prompting and adaptive path planning, whereas InstructNav [46] combines dynamic instruction planning with multiple value maps for diverse navigation tasks. Alongside these modular systems, NaVid [47] uses a video-based VLM to predict navigation actions from visual history and instructions, and Uni-NaVid [48] extends this formulation to multiple embodied navigation tasks. These ground-based studies provide useful mechanisms for semantic alignment, memory, exploration, and action prediction, which aerial methods adapt to larger spaces and additional motion dimensions. TravelUAV [12] employs a hierarchical decoder to predict a target pose and a refined local trajectory, connecting multimodal reasoning with local motion generation. FlightGPT [49] combines supervised fine-tuning, reinforcement learning, and chain-ofthought reasoning to improve navigation decisions. GeoNav [50] integrates a global cognitive map and a local hierarchical scene graph within landmark navigation, target search, and precise localization. AeroDuo [16] exploits complementary high- and low-altitude observations, coordinating target reasoning with local navigation and grounding. FineCog-Nav [51] organizes zero-shot navigation into cognitive modules with specialized prompts and structured interfaces. In parallel with advances in semantic reasoning, several studies address aerial motion execution. SkyVLN [17] combines spatial verbalization and path memory with nonlinear model predictive control for trajectory tracking and dynamic obstacle avoidance. Dyn-Fly [15] generates B-spline trajectories through flow matching and introduces motion-state supervision to improve trajectory continuity. AerialVLA [52] combines front and downward ob servations with coarse directional instructions, predicts spatial offsets and yaw changes, and converts them into velocity commands. These developments show growing attention to both semantic navigation and executable motion. Our work connects these aspects through kinodynamically generated trajectories and acceleration-driven low-level control.

## III. METHODS

## A. Trajectory Generation Pipeline

This pipeline consists of three stages. Searching directly over the kinodynamic state space for a long-range trajectory is computationally expensive. First, a waypoint chain is generated without kinematic constraints. Next, we use hierarchical kinematic $\mathbf { A } ^ { * }$ algorithm to produce a sequence of discrete reference trajectory points based on the waypoint chain. Finally, the AirSim flight control API is invoked to generate a continuous trajectory.

1) Waypoint Chain Generation: We acquire LiDAR point clouds for each scene and convert them into voxel occupancy maps [13]. Each voxel is classified as occupied or free, and its side length is set to that of the smallest cube enclosing the UAV.

For the Main Subset, we specify a start position ${ \bf p } _ { \mathrm { s } }$ and a goal position $\mathbf { p } _ { \mathrm { g } } .$ , and use geometric $\mathbf { A } ^ { * }$ to search for a backbone path between them. At each expanded node, candidate positions are generated using a fixed step length $s _ { \mathrm { s t e p } } = 3 \mathrm { m }$ . Let $\Phi = \{ 0 , \pm { \frac { \pi } { 6 } } , \pm { \frac { \pi } { 3 } } , \pm { \frac { \pi } { 2 } } , \pm { \frac { 2 \pi } { 3 } } , \pm { \frac { 5 \pi } { 6 } } , \pi \}$ be the azimuth set and $\begin{array} { r } { \Theta = \{ \frac { \pi } { 6 } , \frac { \pi } { 3 } , \frac { \pi } { 2 } , \frac { 2 \pi } { 3 } , \frac { 5 \pi } { 6 } \} } \end{array}$ be the polar-angle set, where the polar angle is measured from the positive vertical axis. A candidate displacement is

![](images/b2f505f9f06180de8288ac7e996d6c8d08f20d0d3f864184fb4c58abf515ac1d.jpg)  
Fig. 2. Overview of our dataset generation pipeline. Different waypoint chains are processed by hierarchical kinodynamic $\mathbf { A } ^ { * }$ to generate reference trajectories under motion constraints. After trajectory cleanup, the reference trajectories are tracked through AirSim’s flight-control API to collect observations and motion states for the Main Subset and Auxiliary Subset.

$$
\Delta \mathbf { p } ( \phi , \theta ) = s _ { \mathrm { s t e p } } \left[ \sin \theta \cos \phi \right] , \qquad \phi \in \Phi , \ \theta \in \Theta .\tag{1}
$$

This sampling produces 60 candidate directions per expanded node. Geometric $\mathbf { A } ^ { * }$ searches over valid candidates and returns an ordered waypoint chain ${ \mathcal W } = ( { \bf q } _ { 0 } , { \bf q } _ { 1 } , \dots , { \bf q } _ { M } )$ , where ${ \bf q } _ { 0 } = { \bf p } _ { \mathrm { s } }$ and the final waypoint corresponds to the goal.

For Auxiliary Subset, we first sample a starting waypoint q<sub>0</sub> in free space. Each subsequent waypoint is sampled relative to the preceding waypoint:

$$
\mathbf { q } _ { i + 1 } = \mathbf { q } _ { i } + r _ { i } \left[ \sin \theta _ { i } \cos \phi _ { i } \right] , \qquad 3 < r _ { i } < 2 0 \mathrm { m } .\tag{2}
$$

Here, $\phi _ { i }$ and $\theta _ { i }$ are randomly sampled without the angular discretization. If the sampled waypoint lies in an occupied voxel, it is rejected and sampled again. This procedure continues until the chain contains ten waypoints, ${ \mathcal { W } } = ( \mathbf { q } _ { 0 } , \ldots , \mathbf { q } _ { 9 } )$ including the starting waypoint. For both subsets, the resulting waypoint chain defines the order of local targets for hierarchical kinodynamic A<sup>∗</sup>.

The Main Subset and Auxiliary Subset use different methods to generate their waypoint chains but share the same subsequent kinodynamic planner.

2) Hierarchical Kinodynamic $A ^ { * }$ planning: To generate UAV trajectories that account for velocity-dependent drag and changes in motion, we adopt a simplified kinodynamic model. Let $m _ { \mathrm { U A V } }$ be the UAV mass, and let $\mathbf { p } ^ { k } , \ \mathbf { v } ^ { k } , \ \mathbf { a } ^ { k }$ and $\mathbf { j } ^ { k }$ denote its position, velocity, total acceleration and jerk at step k in an inertial coordinate frame. Taking $\mathbf { e } _ { z } = [ 0 , 0 , 1 ] ^ { \top }$ as the upward unit vector and neglecting external disturbances, Newton’s second law gives

$$
m _ { \mathrm { U A V } } \mathbf { a } ^ { k } = \mathbf { F } _ { p } ^ { k } + \mathbf { F } _ { d } ^ { k } - m _ { \mathrm { U A V } } g \mathbf { e } _ { z } ,\tag{3}
$$

where $\mathbf { F } _ { p } ^ { k }$ is the propulsion force and $\mathbf { F } _ { d } ^ { k }$ is the drag force. We assume that a low-level controller compensates for gravity and express the effective propulsion force as

$$
\mathbf { F } _ { p } ^ { k } = m _ { \mathrm { U A V } } g \mathbf { e } _ { z } + \mathbf { F } _ { u } ^ { k } ,\tag{4}
$$

where $\mathbf { F } _ { u } ^ { k }$ is the gravity-compensated control force. We approximate aerodynamic drag using linear and quadratic velocity-dependent terms:

$$
\mathbf { F } _ { d } ^ { k } = - m _ { \mathrm { U A V } } \left( k _ { 1 } \mathbf { v } ^ { k } + k _ { 2 } \| \mathbf { v } ^ { k } \| \mathbf { v } ^ { k } \right) ,\tag{5}
$$

where $k _ { 1 } , k _ { 2 } ~ > ~ 0$ are mass-normalized drag coefficients. Substituting the propulsion and drag forces into the force balance yields

$$
\mathbf { a } ^ { k } = \frac { \mathbf { F } _ { u } ^ { k } } { m _ { \mathrm { U A V } } } - k _ { 1 } \mathbf { v } ^ { k } - k _ { 2 } \| \mathbf { v } ^ { k } \| \mathbf { v } ^ { k } .\tag{6}
$$

We write this as $\mathbf { a } ^ { k } = \mathbf { a } _ { \mathrm { U A V } } ^ { k } + \mathbf { a } _ { d } ^ { k } .$ , where $\mathbf { a } _ { \mathrm { U A V } } ^ { k } = \mathbf { F } _ { u } ^ { k } / m _ { \mathrm { U A V } }$ is the control acceleration selected during planning, and $\mathbf { a } _ { d } ^ { k } = $ $- k _ { 1 } \mathbf { v } ^ { k } - k _ { 2 } \| \mathbf { v } ^ { k } \| \mathbf { v } ^ { k }$ is the drag acceleration determined by the current velocity.

During planning, the above model is used to propagate reference motion states at a fixed interval $\Delta t .$ . We use the subscript ref to distinguish these planned states from the realized UAV states obtained during flight. The reference velocity and position are updated as

$$
\begin{array} { r l } & { \mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 } = \mathbf { v } _ { \mathrm { r e f } } ^ { k } + \mathbf { a } _ { \mathrm { r e f } } ^ { k } \Delta t , } \\ & { \mathbf { p } _ { \mathrm { r e f } } ^ { k + 1 } = \mathbf { p } _ { \mathrm { r e f } } ^ { k } + \mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 } \Delta t , } \end{array}\tag{7}
$$

where $\mathbf { a } _ { \mathrm { r e f } } ^ { k }$ is the total inertial frame acceleration computed by the simplified model. Each reference point specifies a desired position. The jerk used during planning is computed as

$$
{ \bf j } _ { \mathrm { r e f } } ^ { k } = \frac { { \bf a } _ { \mathrm { r e f } } ^ { k } - { \bf a } _ { \mathrm { r e f } } ^ { k - 1 } } { \Delta t } .\tag{8}
$$

For each target $\begin{array} { c c c } { \mathbf q _ { i } , \ i } & { = } & { 1 , \dots , M } \end{array}$ , the local planner searches over states containing position, velocity, previous acceleration, and heading. For planning, we assume that this force can act along any of the six signed body-axis directions $\mathcal { E } = \{ \pm \mathbf { e } _ { x } , \pm \mathbf { e } _ { y } , \pm \mathbf { e } _ { z } \}$ , with the same maximum magnitude $F _ { \mathrm { U A V , m a x } } ,$ and let $\Gamma = \{ 0 , 0 . 5 , 1 . 0 \}$ be the set of dimensionless force levels, in the body frame at step k. We define the legal control-acceleration set as

$$
\mathcal { A } _ { \mathrm { U A V } } = \left\{ \gamma \frac { F _ { \mathrm { U A V , m a x } } } { m _ { \mathrm { U A V } } } \mathbf { e } \ \middle | \ \gamma \in \Gamma , \ \mathbf { e } \in \mathcal { E } \right\}\tag{9}
$$

$\mathbf { a } _ { \mathrm { U A V , r e f } } ^ { k } \ \in \ \mathcal { A } _ { \mathrm { U A V } }$ . It results in 13 distinct candidates. Candidate states are rejected if they violate the speed or jerk limit or the applicable collision check.

We evaluate each transition from $n ^ { k }$ to $n ^ { k + 1 }$ using the kinodynamic cost

$$
\mathrm { c o s t } _ { \mathrm { k i n } } ( n ^ { k } , n ^ { k + 1 } ) = \mathrm { c o s t } _ { v } ^ { k } + \mathrm { c o s t } _ { a } ^ { k } + \mathrm { c o s t } _ { j } ^ { k } ,\tag{10}
$$

where

$$
\begin{array} { r l } & { \mathrm { c o s t } _ { v } ^ { k } = \lambda _ { 1 } \| \mathbf { v } _ { \mathrm { r e f } } ^ { k } \| + \lambda _ { 2 } \angle \left( \mathbf { v } _ { \mathrm { r e f } } ^ { k } , \ \mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 } \right) , } \\ & { \mathrm { c o s t } _ { a } ^ { k } = \lambda _ { 3 } f ( \mathbf { a } _ { \mathrm { U A V , r e f } } ^ { k } ) , } \\ & { \mathrm { c o s t } _ { j } ^ { k } = \lambda _ { 4 } \| \mathbf { j } _ { \mathrm { r e f } } ^ { k } \| . } \end{array}\tag{11}
$$

Here, $\it { \Delta } \psi _ { \mathrm { r e f } } ^ { k } , \psi _ { \mathrm { r e f } } ^ { k + 1 } )$ is the angle between $\mathbf { v } _ { \mathrm { r e f } } ^ { k }$ and $\mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 }$ The angle is set to zero if either velocity vector is zero. The second term penalizes selected special actions:

$$
f ( \mathbf { u } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathbf { u } \in \mathcal { A } _ { \mathrm { s p } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \mathcal { A } _ { \mathrm { s p } } = \{ ( 0 , 1 , 0 ) ^ { \top } , ( 0 , - 1 , 0 ) ^ { \top } , \mathbf { 0 } \}\tag{12}
$$

The coefficients $\lambda _ { 1 } , \ldots , \lambda _ { 4 }$ control the relative contributions of these terms. Candidate nodes with cost $_ { \mathrm { { k i n } } } > \tau _ { c }$ are pruned, and the remaining nodes are ordered by their accumulated transition cost plus a distance-to-waypoint heuristic.

When the state enters the arrival radius $r _ { \mathrm { a r r } }$ of the currently active waypoint, the local search for that segment concludes, and the node information is recorded. The successful search segment is reconstructed by backtracking via parent pointers and appended to the completed trajectory segments. The terminal position, velocity and acceleration of this branch serve as the initial conditions for the search toward the next waypoint; this process continues until all waypoints have been reached. Finally, a trajectory cleanup step is performed to eliminate anomalous patterns such as oscillatory steering and loops.

Algorithm 1: Hierarchical Kinodynamic $\mathbf { A } ^ { * }$   
Input: Start p<sub>s</sub>, goal p<sub>g</sub>, map O, action set A, cost threshold $\tau _ { c }$   
Output: Trajectory T or failure   
1 Generate waypoints {q<sub>1</sub>, . . . , q<sub>M</sub> } from $\mathbf { p _ { s } }$ to $\mathbf { p } _ { \mathrm { g } }$ using geometric   
${ \mathrm { A } } ^ { * } ;$   
2 if no path is found then   
3 return failure;   
4 $\tau \gets \emptyset ;$   
5 Initialize start state n at $\mathbf { p } _ { \mathrm { { s } } } ;$   
6 for i ← 1 to M do   
7 Open set $Q \gets \{ n _ { 0 } \} ;$ closed set $C \gets \emptyset ;$   
8 g(n<sub>0</sub>) ← 0; clear the parent of n<sub>0</sub>;   
9 found ← false;   
10 while Q ̸= ∅ do   
11 Select n with the smallest $g ( n ) + h ( n )$ from Q, where   
$h ( n ) = \| \mathbf { p } ( n ) - \mathbf { q } _ { i } \| ;$   
12 Remove n from $Q ;$   
13 if the discretized state of n is already in C then   
14 continue;   
15 if n satisfies the arrival condition at q then   
16 Trace parent states from n back to n<sub>0</sub>;   
17 Append this segment to T in forward order, without   
repeating the connecting state;   
18 n<sub>0</sub> ← n; found ← true;   
19 break;   
20 Add the discretized state of n to C;   
21 foreach u ∈ A do   
22 Rotate u from the body frame to the world frame;   
23 Compute the next state n<sup>′</sup> using the UAV dynamics;   
24 if the motion collides with an obstacle or violates   
motion constraints then   
25 continue;   
26 if the discretized state of n<sup>′</sup> is in C then   
27 continue;   
28 $\begin{array} { r l r } {  { c  \mathrm { c o s t { } } _ { \mathrm { k i n } } ( n , n ^ { \prime } ) ; } } \end{array}$   
29 if $c > \tau _ { c }$ then   
30 continue;   
31 $g _ { \mathrm { n e w } }  g ( n ) + c ;$   
32 if n<sup>′</sup> represents a new state or g is lower than its   
recorded cost then   
33 g(n<sup>′</sup>) ← g<sub>new</sub>;   
34 Set n as the parent of $n ^ { \prime } ;$   
35 Add or update n<sup>′</sup> in $Q ;$   
36 if found = false then   
37 return failure;   
38 return T;

## B. Dataset Construction

We construct the dataset using the UE4 environment assets provided by OpenUAV [12]. We select 16 representative scenes and integrate hierarchical kinodynamic $\mathbf { A } ^ { * }$ into the automated data collection pipeline [52].

After generating the reference trajectory using hierarchical kinodynamic $\mathbf { A } ^ { * }$ , the trajectory $\operatorname { t r a j } = \{ n ^ { 0 } , \dots , n ^ { K - 1 } \}$ is then executed progressively through flight-control API. At each control step k, we compute a velocity command from the current UAV position $\mathbf { p } ^ { k }$ to the next reference position $\mathbf { p } _ { \mathrm { r e f } } ^ { k + 1 }$ using the time interval $\Delta t$ :

$$
\mathbf { v } _ { \mathrm { c m d } } ^ { k } = \frac { \mathbf { p } _ { \mathrm { r e f } } ^ { k + 1 } - \mathbf { p } ^ { k } } { \Delta t } .\tag{13}
$$

This velocity command is sent to the flight controller to guide the UAV toward the next reference point. The pose of the drone is obtained as:

$$
\pmb { \eta } ^ { k } = \left[ p _ { x } ^ { k } , p _ { y } ^ { k } , p _ { z } ^ { k } , \mathrm { p i t c h } ^ { k } , \mathrm { r o l l } ^ { k } , \mathrm { y a w } ^ { k } \right] ^ { \top } .\tag{14}
$$

At each valid control step, we record the realized UAV pose $\eta ^ { k }$ , velocity $\mathbf { v } ^ { k }$ , associated reference control acceleration $\mathbf { a } _ { \mathrm { U A V , r e f } } ^ { k } ,$ and the rendered front- and bottom-view images $I _ { f } ^ { k }$ and $I _ { b } ^ { k }$ . We also compute the corresponding front- and bottomview directional free-space distances (DFDs), $d _ { f } ^ { k }$ and $d _ { b } ^ { k }$ , from the occupancy map.

A DFD measures the distance from the current UAV position to the first occupied voxel intersected by a ray in the directions observed by the camera. Each ray has a maximum length of 100 m. Ray traversal stops at the first occupied voxel, and the traveled distance is recorded as the DFD. If no occupied voxel is encountered within 100 m, the DFD is assigned the maximum value of 100 m.

The resulting data support the two levels of the navigation model. At each high-level planning event, we form a waypointplanning sample

$$
D _ { h } ^ { i } = \left( T ^ { i } , \mathbf { w } ^ { i } , I _ { f } ^ { k _ { i } } , I _ { b } ^ { k _ { i } } \right) ,\tag{15}
$$

where $k _ { i }$ is the control step at which body-frame waypoint $\mathbf { w } ^ { i }$ is planned. $T ^ { i }$ is the stage-dependent navigation instruction. For low-level control, each valid control step provides a sample

$$
D _ { l } ^ { k } = \left( \eta ^ { k } , \mathbf { v } ^ { k } , \mathbf { a } _ { \mathrm { U A V , r e f } } ^ { k } , \mathbf { w } ^ { i } , I _ { f } ^ { k } , I _ { b } ^ { k } , d _ { f } ^ { k } , d _ { b } ^ { k } \right) .\tag{16}
$$

Thus, high-level samples pair observations and instructions with the next waypoint, whereas low-level samples pair the UAV state and observations with an acceleration action and free-space distance annotations.

1) Auxiliary Subset: Auxiliary Subset focuses on low-level waypoint-following behavior. We randomly generate waypoint chains and use hierarchical kinodynamic $\mathbf { A } ^ { * }$ to produce the corresponding trajectories. Each valid control step is paired with the active waypoint used during trajectory generation, rather than the final endpoint of the entire chain. As the UAV progresses through the chain, the waypoint label changes accordingly, providing supervision for successive local navigation objectives. This subset contains only low-level samples $D _ { l } ^ { k }$ and does not require language-instruction annotations. It complements the complete navigation trajectories in Main Subset by introducing varied waypoint transitions for training the low-level controller.

2) Main Subset: Main Subset contains complete targetoriented navigation trajectories generated using hierarchical kinodynamic $\mathbf { A } ^ { * }$ . It supports both high-level waypoint planning and low-level acceleration control. The high-level samples cover Cruise, where the UAV approaches the target region using spatial guidance, and Seek, where it visually locates the target using its description. Together with the recorded UAV states, images, acceleration actions, and DFDs, these annotations form the low-level samples $D _ { l } ^ { k }$

For Cruise-stage samples, we traverse each trajectory in temporal order. From the current state at step $k _ { i } ,$ we identify the first subsequent state whose distance from $\mathbf { p } ^ { k _ { i } }$ exceeds

10 m. Its displacement is transformed into the current UAV body frame and scaled to 10 m while preserving its direction. If the trajectory endpoint is less than 10 m away, its bodyframe displacement is used without scaling. The instruction $T ^ { i }$ provides the target’s relative direction and distance from the current UAV position.

The target description is generated based on four images: two top-down views captured from 5 m and 8 m directly above the target, and two frontal views captured from a height of 2 m while moving along horizontal circular paths centered on the target. For the frontal views, the camera is positioned at a height of 2 m on circular paths with radii of 5 m meters and 8 m meters, respectively, with the camera always facing the target. These images are fed into Qwen3-VL-235B to generate the target description. Additionally, each trajectory is assigned one of nine application scenarios—such as medical rescue or logistics transport—to enhance instruction diversity. We require the large model to incorporate the specified application context when describing the characteristics of a target.

For samples in the ”Seek” phase, ten points located more than 5 m meters above the target are randomly sampled. Points are then filtered based on the condition that the topdown view’s DFD value is not less than the point’s actual distance to the target, ensuring the target is unobstructed in the image. Waypoint labels represent the target’s coordinates in the vehicle’s local coordinate system; if the magnitude of these coordinates exceeds 10 meters, the vector is scaled down to a length of 10 m meters. Instructions for the ”Seek” phase consist solely of the generated target description, necessitating visual localization based on the input images.

To construct the dataset, we first reserve one of the 16 scenes exclusively for Test Unseen Map. In the remaining 15 scenes, we partition the target objects into disjoint seen and unseen sets. Target-object placement is required only for Main Subset. For each trajectory in this subset, we first select the start and endpoint positions and place a target object at the endpoint. Seen objects are used for the training set and Test Seen, whereas unseen objects are used for Test Unseen Object. We then plan the trajectory using hierarchical kinodynamic $\mathbf { A } ^ { * }$ and collect the corresponding observations and annotations following the procedures described above. For Auxiliary Subset, we select a start position without placing a target object, randomly generate a waypoint chain, and collect the resulting trajectory and low-level annotations. Auxiliary Subset is constructed entirely within the remaining 15 scenes and is used only for training. Finally, we apply the Main Subset collection procedure to the held-out scene to construct Test Unseen Map. The resulting dataset comprises 147K training trajectories and 1K trajectories for each of Test Seen, Test Unseen Object, and Test Unseen Map, while all 150K trajectories in Auxiliary Subset are used for training.

Fig. 3 summarizes the resulting dataset. Panel (a) shows the nine annotation scenarios, with urban scenarios accounting for 62.9% of the samples, while panel (b) illustrates the vocabulary of the generated target descriptions. Panels (c) and (d) show the distribution of trajectory lengths.

Table I compares how different VLN datasets are constructed and their main features. KinoUAV uses automatic trajectory collection and instruction annotation. OpenFly also automates these steps, but KinoUAV further includes kinodynamic modeling in trajectory generation to account for UAV motion constraints. This allows large-scale collection of trajectory data for full navigation and waypoint tracking.

TABLE I  
COMPARISON OF VLN DATASET PROPERTIES. $N _ { \mathrm { t r a j } }$ IS THE NUMBER OF TRAJECTORIES; len(traj) AND $\overline { { \mathrm { l e n } ( \mathrm { i n s t r } ) } }$ ARE THE AVERAGE TRAJECTORY AND INSTRUCTION LENGTHS; $\overline { { N } } _ { \mathrm { a c t } }$ IS THE AVERAGE NUMBER OF ACTIONS PER TRAJECTORY; AND $N _ { \mathrm { v o c a b } }$ IS THE VOCABULARY SIZE. KINOUAV STATISTICS REFER TO THE LANGUAGE-ANNOTATED FULL-NAVIGATION DATA, EXCLUDING AUXILIARY SUBSET.
<table><tr><td>Dataset</td><td>Kinodynamic modeling</td><td> $N _ { \mathrm { t r a j } }$ </td><td> $\overline { { \mathrm { l e n } ( \mathrm { t r a j } ) } }$ </td><td> $\overline { { \mathrm { l e n } ( \mathrm { i n s t r } ) } }$ </td><td>Action space</td><td> $\overline { { N } } _ { \mathrm { a c t } }$ </td><td>Environment</td><td> $N _ { \mathrm { v o c a b } }$ </td></tr><tr><td>R2R</td><td>No</td><td>7189</td><td>10</td><td>29</td><td>Graph</td><td>5</td><td>Matterport3D</td><td>3.1K</td></tr><tr><td>RxR</td><td>No</td><td>13992</td><td>14.9</td><td>129</td><td>Graph</td><td>8</td><td>Matterport3D</td><td>7.0K</td></tr><tr><td>TouchDown</td><td>No</td><td>9326</td><td>313.9</td><td>90</td><td>Graph</td><td>35</td><td>Google Street View</td><td>5.0K</td></tr><tr><td>VLN-CE</td><td>No</td><td>4475</td><td>11.1</td><td>19</td><td>2DoF</td><td>56</td><td>Matterport3D</td><td>4.3K</td></tr><tr><td>LANI</td><td>No</td><td>6000</td><td>16</td><td>一</td><td>2DoF</td><td>116</td><td>CHALET</td><td>2.3K</td></tr><tr><td>AVDN</td><td>No</td><td>6269</td><td>144.7</td><td>89</td><td>3DoF</td><td>7</td><td>xView</td><td>3.3K</td></tr><tr><td>AerialVLN [11]</td><td>No</td><td>8446</td><td>661.8</td><td>83</td><td>4DoF</td><td>204</td><td>AirSim</td><td>4.5K</td></tr><tr><td>CityNav</td><td>No</td><td>32637</td><td>545</td><td>26</td><td>4DoF</td><td></td><td>SensatUrban</td><td>6.6K</td></tr><tr><td>TRAVEL [12]</td><td>No</td><td>12149</td><td>255</td><td>104</td><td>6 DoF</td><td>264</td><td>AirSim</td><td>10.8K</td></tr><tr><td>OpenFly [13]</td><td>No</td><td>100K</td><td>99.1</td><td>59</td><td>4DoF</td><td>35</td><td>AirSim, GTA5, 3D GS, GE</td><td>15.6K</td></tr><tr><td>KinoUAV (Ours)</td><td>Yes</td><td>150K</td><td>146.4</td><td>78</td><td>6DoF</td><td>84</td><td>AirSim</td><td>16.4K</td></tr></table>

![](images/7e41192c6a0f16256eeeb36bbb066a336f25a0c12af2eeda95ea917dcd8bd1cf.jpg)  
(a)

![](images/118318a4d1a8e6d8b309bef8e799323d9d2d4b7de957f27032df63c5663eab58.jpg)  
(a)

![](images/7be37e11bac9c4d90a9b0077dd29f76e91176bd7359f80f0604aeef3dfbc5083.jpg)  
(c)

(b)  
![](images/8279797ac47ffeea77a8bd1ba39833d83cf1a6323202d22f1a60fa6d3c747587.jpg)  
Fig. 3. Dataset statistics. (a) Scenario categories used to annotate target descriptions; blue denotes urban scenarios and orange denotes non-urban scenarios. (b) Word cloud of the annotated descriptions. (c) and (d) Trajectory length distributions of Main Subset and Auxiliary Subset, respectively.  
(b)

Fig. 4(a) shows the distribution of acceleration actions in KinoUAV, with larger spheres indicating more frequent actions. Fig. 4(b) shows how action selection varies with DFD. The forward action becomes less likely as the forward DFD decreases. The same trend is observed for the downward action when the bottom DFD decreases. These results suggest that the UAV is less likely to move toward directions with limited free space.

## C. KinoUAV-Agent

As shown in Fig. 5, KinoUAV-Agent adopts a hierarchical navigation framework comprising a high-level waypoint planner and a low-level motion controller. The two modules operate at different frequencies: the high-level planner generates a new waypoint when a planning event occurs, whereas the lowlevel controller selects an acceleration action at every control step. We use i to index waypoints and k to index control steps.

![](images/758ee3cf14bedfc0927bde6267886fb7836d8bac3f50ab5511af34ffd3f34792.jpg)

![](images/4a09c60406620a1df637f4f56907edbc39d7aacf113286452ddd34cee9f323c9.jpg)

![](images/04ec0d692181937c5af5ed11325ff259e33918b4abf563c234d64fcd7d5b8e58.jpg)  
Fig. 4. Acceleration action statistics in our dataset. (a) Overall action distribution. (b) Action probabilities versus DFD. Actions are grouped by direction regardless of magnitude, with the infrequent stay action excluded.

The initial UAV position is $\mathbf { p } ^ { 0 } \ = \ p _ { S }$ . Before invoking the high-level planner, we assign the takeoff waypoint $\mathbf { w } ^ { 0 } =$ $( 0 , 0 , z _ { H } ) ^ { \top }$ , where $z _ { H }$ is the prescribed cruise altitude. The low-level controller first guides the UAV from the ground to this waypoint. Once the takeoff waypoint is reached, the highlevel planner begins generating waypoints for target-oriented navigation.

![](images/025decb89de587f21a70118e222e61911824f2d0c1f6757ea64f7c02c526be78.jpg)  
Fig. 5. Overview of KinoUAV-Agent. It hierarchically integrates waypoint planning, DFD estimation, and acceleration-based motion control.

As shown in Fig. 6, the subsequent flight consists of Cruise and Seek stages. During Cruise, coarse directional guidance helps the UAV move toward the vicinity of the target. During Seek, the UAV uses the target description and its visual observations to locate and approach the object. Within either stage, the low-level controller follows the current waypoint while responding to obstacles observed in the front and bottom views. When the distance to the current waypoint is less than $r _ { \mathrm { a r r } } .$ , the high-level planner uses the latest observations to generate the next one. A stage transition can also trigger immediate replanning, so the UAV need not finish a previously assigned Cruise waypoint after entering the Seek region. This cycle continues until the navigation task terminates or the maximum number of control steps is reached.

1) High-Level Waypoint Planner: Let $\mathbf { g } ~ = ~ ( x _ { g } , y _ { g } , 0 ) ^ { \top }$ denote the ground-level target position in the coordinate frame defined above. We define the reference point for the Cruise stage as $\mathbf { g } _ { H } = ( x _ { g } , y _ { g } , z _ { H } ) ^ { \top }$ , vertically above the target at cruise altitude. This point defines the vicinity in which the navigation stage changes; its exact coordinate is not provided as an input to the high-level planner.

At a planning step k, the high-level planner receives the front-view image $I _ { f } ^ { k }$ , bottom-view image $I _ { b } ^ { k }$ , and a stagedependent textual cue $T ^ { i }$ . It predicts the next waypoint as

$$
\begin{array} { r } { { \bf w } ^ { i } = f _ { h } \left( I _ { f } ^ { k } , I _ { b } ^ { k } , T ^ { i } \right) , \qquad i \ge 1 , } \end{array}\tag{17}
$$

The high-level module consistently outputs the next waypoint $\mathbf { w } ^ { i }$ in the body frame, with the corresponding waypoint in the world frame being $\mathbf { W } ^ { i } = \mathbf { p } ^ { k } + \mathbf { R } ^ { k } \mathbf { \bar { w } } ^ { i }$ , where $\mathbf { R } ^ { k }$ is the rotation matrix from the body frame to the world frame at the waypoint-generation step k. The same planner $f _ { h }$ is used in both navigation stages, while the content of $T ^ { i }$ changes with the stage.

During Cruise, $T ^ { i } = T _ { \mathrm { d i r } } ^ { i }$ specifies a discrete direction and distance relative to the UAV, such as front-left 50m, right 32m. The planner combines this coarse cue with the current visual observations to select successive waypoints. The UAV remains in Cruise while it is outside the target region. When

Algorithm 2: Closed-Loop Navigation of KinoUAV-  
Agent   
Input: Start p<sub>s</sub>, goal $\mathbf { p } _ { \mathrm { g } } ,$ target description $T _ { \mathrm { o b j } } .$ takeoff height $z _ { H } ,$   
thresholds $r _ { \mathrm { a r r } }$ and $r _ { \mathrm { s t a g e } } ,$ action set A<sub>UAV</sub>, maximum   
steps $K _ { \mathrm { m a x } }$   
Output: Executed trajectory $\tau$   
1 $\mathbf { p } ^ { 0 } \dot { \mathbf { \xi } } ^ { \prime } p _ { S } ; \mathbf { W } ^ { 0 } \gets ( \bar { 0 } , 0 , z _ { H } ) ^ { \top } ; \mathbf { g } _ { H } \gets \mathbf { p } _ { \mathrm { g } } + ( 0 , 0 , z _ { H } ) ^ { \top } ;$   
2 i ← 0; k ← 0; stage ← Cruise;   
3 Initialize T with the initial UAV state;   
4 while k < K and the task is not terminated do   
5 Observe I<sup>k</sup>, I<sup>k</sup>, and the current UAV state;   
6 update ← false;   
7 if stage = Cruise and $\| \mathbf { p } ^ { k } - \mathbf { g } _ { H } \| \leq r _ { \mathrm { s t a g e } }$ then   
8 stage ← Seek;   
9 update ← true;   
10 else   
11 if $\| \mathbf { p } ^ { k } - \mathbf { W } ^ { i } \| \leq r _ { \mathrm { a r r } }$ then   
12 update ← true;   
13 if update = true then   
14 i ← i + 1;   
15 if stage = Cruise then   
16 T<sup>i</sup> ← current direction and distance guidance;   
17 else   
18 $T ^ { i } \gets T _ { \mathrm { o b j } } ;$   
19 w<sup>i</sup> $ f _ { h } ( I _ { f } ^ { k } , I _ { b } ^ { k } , T ^ { i } )$   
20 $\mathbf { W } ^ { i }  \mathbf { p } ^ { k } + \mathbf { R } ^ { k } \mathbf { w } ^ { i } ;$   
21 Compute $\mathbf { s } _ { i } ^ { k }$ from the current UAV state and $\mathbf { W } ^ { i } ;$   
22 $\mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } \dot { \gets } f _ { l } \left( I _ { f } ^ { k } , I _ { b } ^ { k } , \mathbf { s } _ { i } ^ { k } \right)$   
23 Select the nearest acceleration in A <sub>A</sub> to $\mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } \mathrm { ; }$   
24 Generate the next reference point and track it;   
25 Append the updated UAV state to T;   
26 k ← k + 1;   
27 return T;

$$
\| \mathbf { p } ^ { k } - \mathbf { g } _ { H } \| \leq r _ { \mathrm { s t a g e } } ,\tag{18}
$$

the navigation process switches to Seek, where $r _ { \mathrm { s t a g e } }$ is the

![](images/59370f29bcfb5fb8395cfcfbab9d29ad45b0fe3d2b1bb8eb9dc4ccc1d4d3730c.jpg)  
Fig. 6. Illustration of the Cruise–Seek navigation process and stage-specific instructions. Front- and bottom-view observations are shown at six successiv navigation moments. During Cruise (1–4), the UAV approaches the target region using direction and distance guidance. During Seek (5–6), it visually localizes and approaches the target using its textual description. Red boxes highlight the target, and representative instructions for both stages are shown below.

target-region radius.

In Seek, the directional cue is removed and replaced by the appearance description $T _ { \mathrm { o b j } }$ of the ground-level target. The planner therefore receives $( I _ { f } ^ { k } , I _ { b } ^ { k } , \bar { T } _ { \mathrm { o b j } } )$ and predicts waypoints that progressively bring the UAV closer to the visually identified object. If the stage changes while the UAV is following a Cruise waypoint, that waypoint is discarded and a new Seek waypoint is generated from the current observations. Subsequent Seek waypoints are generated whenever the current one is reached.

2) Low-Level Motion Controller: For an active waypoint W<sup>i</sup>, the low-level controller operates at each control step k. It takes the two camera views and a waypoint-relative state vector as input, and predicts the control acceleration

$$
\begin{array} { r } { \mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } = f _ { l } \left( I _ { f } ^ { k } , I _ { b } ^ { k } , \mathbf { s } _ { i } ^ { k } \right) . } \end{array}\tag{19}
$$

This is the active control acceleration in the kinodynamic model; the total acceleration additionally includes velocitydependent drag.

The waypoint-relative distance and direction are

$$
\begin{array} { r l } & { { \boldsymbol { r } } _ { i } ^ { k } = \| \mathbf { W } ^ { i } - \mathbf { p } ^ { k } \| , } \\ & { \mathbf { D } _ { i } ^ { k } = \frac { \mathbf { W } ^ { i } - \mathbf { p } ^ { k } } { r _ { i } ^ { k } } , \qquad r _ { i } ^ { k } > 0 . } \end{array}\tag{20}
$$

We set $\mathbf { D } _ { i } ^ { k } = \mathbf { 0 }$ when $r _ { i } ^ { k } = 0$ . Let $v _ { h } ^ { k } = \sqrt { ( v _ { x } ^ { k } ) ^ { 2 } + ( v _ { y } ^ { k } ) ^ { 2 } }$ and $v _ { v } ^ { k } = v _ { z } ^ { k }$ denote the horizontal speed and vertical velocity, respectively. The state supplied to the controller is

$$
\mathbf { s } _ { i } ^ { k } = \left[ \mathbf { D } _ { i } ^ { k } , g _ { \mathrm { d i s t } } ( \boldsymbol { r } _ { i } ^ { k } ) , v _ { h } ^ { k } , v _ { v } ^ { k } \right] ^ { \top } ,\tag{21}
$$

where $g _ { \mathrm { d i s t } }$ applies logarithmic distance normalization:

$$
g _ { \mathrm { d i s t } } ( r ) = \frac { 1 } { 2 } \operatorname* { m i n } \left\{ \log _ { 1 0 } \left( \operatorname* { m a x } \{ r , 1 0 ^ { - 2 } \} \right) , 2 \right\} .\tag{22}
$$

The state encodes the waypoint direction, normalized distance, and the UAV’s horizontal and vertical velocities.

The low-level controller comprises a shared visual encoder $f _ { \mathrm { v i s i o n } } .$ a DFD decoder $f _ { \mathrm { D F D } }$ , and an acceleration predictor $f _ { \mathrm { a c t i o n } }$ . The encoder extracts features from both views:

$$
\mathbf { f } _ { \mathrm { i m g } } ^ { k } = f _ { \mathrm { v i s i o n } } \left( I _ { f } ^ { k } , I _ { b } ^ { k } \right) .\tag{23}
$$

The visual encoder uses a ResNet18 backbone with the final classification layer removed and the global average pooling layer retained. The front- and bottom-view images are processed separately using shared backbone weights, producing a 512-dimensional feature vector for each view. These vectors are concatenated to form the joint visual representation $\mathbf { f } _ { \mathrm { i m g } } ^ { k } ~ \in ~ \mathbb { R } ^ { 1 0 2 4 }$ . The DFD decoder estimates the front- and bottom-view directional free-space distance representations:

$$
\mathbf { d } ^ { k } = f _ { \mathrm { D F D } } \left( \mathbf { f } _ { \mathrm { i m g } } ^ { k } \right) , \qquad \mathbf { d } ^ { k } = [ \mathbf { d } _ { f } ^ { k } , \mathbf { d } _ { b } ^ { k } ] .\tag{24}
$$

The acceleration predictor then combines the visual features, waypoint-relative state, and estimated DFDs:

$$
{ \bf a } _ { \mathrm { U A V , r a w } } ^ { k } = f _ { \mathrm { a c t i o n } } \left( { \bf f } _ { \mathrm { i m g } } ^ { k } , { \bf s } _ { i } ^ { k } , { \bf d } ^ { k } \right) .\tag{25}
$$

The acceleration predicted in (25) is expressed in the UAV body frame. Before execution, we project it onto the discrete action set $\boldsymbol { \mathcal { A } } _ { \mathrm { U A V } }$ . The executed body-frame action is the prototype nearest to the network prediction in Euclidean distance:

$$
\mathbf { a } _ { \mathrm { U A V } } ^ { k } = \underset { \mathbf { u } \in \mathcal { A } _ { \mathrm { U A V } } } { \arg \operatorname* { m i n } } \left\| \mathbf { u } - \mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } \right\| _ { 2 } .\tag{26}
$$

We rotate this action into the inertial frame and add the velocity-dependent drag acceleration:

$$
\mathbf { a } _ { \mathrm { r e f } } ^ { k } = \mathbf { R } ^ { k } \mathbf { a } _ { \mathrm { U A V } } ^ { k } - k _ { 1 } \mathbf { v } ^ { k } - k _ { 2 } \| \mathbf { v } ^ { k } \| \mathbf { v } ^ { k } ,\tag{27}
$$

Starting from the current realized position and velocity, we compute the next reference position $\mathbf { p } _ { \mathrm { r e f } } ^ { k + 1 }$ as

$$
\begin{array} { r l } & { \mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 } = \mathbf { v } ^ { k } + \mathbf { a } _ { \mathrm { r e f } } ^ { k } \Delta t , } \\ & { \mathbf { p } _ { \mathrm { r e f } } ^ { k + 1 } = \mathbf { p } ^ { k } + \mathbf { v } _ { \mathrm { r e f } } ^ { k + 1 } \Delta t . } \end{array}\tag{28}
$$

The next reference position is tracked through the flightcontrol API. The realized UAV state is then read from the simulator to update the waypoint-relative state. After the action is executed, the UAV state and its distance to the active waypoint are updated. The same waypoint remains active while $r _ { i } ^ { k + 1 } ~ > ~ r _ { \mathrm { a r r } }$ . Once $r _ { i } ^ { k + 1 } ~ \le ~ { \overline { { r } } } _ { \mathrm { a r r } }$ , the waypoint is considered reached and the high-level planner is invoked again, unless the navigation task has already terminated. Here $r _ { \mathrm { a r r } }$ is the waypoint arrival radius, distinct from the larger targetregion radius $r _ { \mathrm { s t a g e } }$ used to switch from Cruise to Seek. The closed-loop interaction between the two modules is illustrated in Fig. 5.

For training, we supervise both the control acceleration and the DFD prediction using mean squared error (MSE). Let $\mathbf { a } _ { \mathrm { U A V , g t } } ^ { k }$ and $\mathbf { d } _ { \mathrm { { g t } } } ^ { k }$ denote their ground-truth labels. The losses are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a c t i o n } } = \mathrm { M S E } \big ( \mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } , \mathbf { a } _ { \mathrm { U A V , g t } } ^ { k } \big ) , } \\ & { \quad \mathcal { L } _ { \mathrm { d i s t } } = \mathrm { M S E } \big ( \mathbf { d } ^ { k } , g _ { \mathrm { d i s t } } ( \mathbf { d } _ { \mathrm { g t } } ^ { k } ) \big ) , } \\ & { \qquad \mathcal { L } = \lambda _ { \mathrm { a c t i o n } } \mathcal { L } _ { \mathrm { a c t i o n } } + \lambda _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { d i s t } } , } \end{array}\tag{29}
$$

where MSE averages the squared errors over all elements in a training batch, and $\lambda _ { \mathrm { a c t i o n } }$ and $\lambda _ { \mathrm { d i s t } }$ weight the two objectives.

## IV. EXPERIMENTS

## A. Implementation Details

All experiments were conducted on Ubuntu 22.04 with Python 3.10 and PyTorch 2.8.0. Both model training and inference used eight NVIDIA RTX 5090 GPUs.

1) Hyperparameter Settings: For dataset construction, we set the UAV mass to $m _ { \mathrm { U A V } } = 1 \mathrm { k g }$ and the maximum gravitycompensated control force to $F _ { \mathrm { U A V , m a x } } = 1 \mathrm { N } .$ . The massnormalized linear and quadratic drag coefficients are $k _ { 1 } ~ =$ $0 . 2 \mathrm { s ^ { - 1 } }$ and $k _ { 2 } ~ = ~ 0 . 0 1 \mathrm { m ^ { - 1 } }$ , respectively. The cost weights are $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 } ) = ( 1 , 2 , 2 , 0 . 5 )$ , and the waypoint arrival radius is $r _ { \mathrm { a r r } } = 3$ m. Simulation time interval $\Delta t = 1 s$

We fine-tune all the vision-language models with LoRA rank 8, targeting all eligible modules, using a learning rate of $1 . 0 \times 1 0 ^ { - 4 }$ for 2 epochs. For the low-level controller, the action and DFD loss weights are $\lambda _ { \mathrm { a c t i o n } } ~ = ~ 0 . 6 7$ and $\lambda _ { \mathrm { d i s t } } ~ = ~ 0 . 3 3$ , respectively. During evaluation, the cruise altitude is $z _ { H } ~ = ~ 1 2$ m and the stage-transition threshold is $r _ { \mathrm { s t a g e } } ~ = ~ 5  { \mathrm { m } }$ . The image resolution used for both model training and inference is 256×256. The low-level controller is trained for 50 epochs with a learning rate of $1 . 0 \times 1 0 ^ { - 4 }$ and a batch size of 128. For low-level supervision on Main Subset, we use the trajectory endpoint as the waypoint at each control step, whereas Auxiliary Subset uses the active waypoint from the sampled chain.

## B. Evaluation Metric

We evaluate N navigation episodes. For episode n, let $\mathbf { g } _ { n }$ be the target position, and let $\mathbf { p } _ { n } ^ { k }$ and $\mathbf { v } _ { n } ^ { k }$ be the UAV position and velocity at step k, with $K _ { n }$ denoting the final step. Success Rate (SR) requires the UAV to finish within 6 m of the target:

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \big ( \| \mathbf { p } _ { n } ^ { K _ { n } } - \mathbf { g } _ { n } \| < 6 \mathrm { m } \big ) ,\tag{30}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. The Oracle Success Rate (OSR) is calculated as follows: a sample is considered an ”Oracle success” if any point along the drone’s flight trajectory is within 20 m of the target, and the OSR represents the proportion of such samples. Navigation Error (NE) is the mean distance from the final UAV position to the target:

$$
\begin{array} { l } { { \displaystyle \mathrm { O S R } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \biggl ( \operatorname* { m i n } _ { 0 \le k \le K _ { n } } \| { \bf p } _ { n } ^ { k } - { \bf g } _ { n } \| < 2 0 \mathrm { m } \biggr ) } , } \\ { { \displaystyle \mathrm { N E } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \| { \bf p } _ { n } ^ { K _ { n } } - { \bf g } _ { n } \| . } } \end{array}\tag{31}
$$

To jointly evaluate navigation success and path efficiency, we compute success weighted by path length (SPL). For episode n, let $S _ { n }$ denote the binary success indicator defined by the SR criterion, and let $\ell _ { n }$ and $L _ { n }$ denote the ground-truth and executed trajectory lengths, respectively. SPL is averaged over all N episodes:

$$
\mathrm { S P L } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } S _ { n } \frac { \ell _ { n } } { \operatorname* { m a x } ( \ell _ { n } , L _ { n } ) }\tag{32}
$$

An unsuccessful episode contributes zero to SPL. The collision rate (CR) refers to the proportion of trajectories that result in a collision. In addition, we evaluate motion characteristics using average speed and the average absolute yaw rate $\overline { { | \omega _ { z } ^ { k } | } }$ . These statistics reflect flight speed and changes in heading. The yaw rate is the change in yaw divided by $\Delta t \colon$

$$
\omega _ { z } ^ { k + 1 } = \frac { \mathrm { y a w } ^ { k + 1 } - \mathrm { y a w } ^ { k } } { \Delta t } ,\tag{33}
$$

## C. Comparison Experiments

We compare KinoUAV-Agent with Random Action and three representative learned navigation baselines: CMA, TravelUAV, and AerialVLA. For a fair comparison, all three learned baselines are retrained on the KinoUAV training set and evaluated in the same simulation environments using the same test splits and evaluation criteria. Random Action randomly selects a valid acceleration action at each step until the UAV reaches the target or the maximum number of evaluation steps is reached.

All learned baselines are retrained on our dataset. For CMA [3], we adapt its LSTM to use concatenated front- and bottomview image features and replace discrete action classification with three-dimensional acceleration regression. We train the model for 2 epochs with a batch size of 16 and a learning rate of $1 0 ^ { - 4 } .$ For TravelUAV [12], we convert the input format while retaining the original model architecture and training configuration. For AerialVLA [52], we convert the next trajectory point in each training sample into the three quantized action components required by the model. We retain its original training settings, except that the number of training epochs is set to 2.

TABLE II  
NAVIGATION PERFORMANCE ON KINOUAV TEST SEEN. SR, OSR, NE, SPL, AND CR DENOTE SUCCESS RATE, ORACLE SUCCESS RATE, NAVIGATION ERROR, SUCCESS WEIGHTED BY PATH LENGTH, AND COLLISION RATE, RESPECTIVELY. MEAN SPEED AND MEAN ABSOLUTE YAW RATE ARE REPORTED IN m/s AND rad/s, RESPECTIVELY.
<table><tr><td>Method</td><td>SR (%)↑</td><td>OSR (%)↑</td><td>NE (m)↓</td><td>SPL (%)↑</td><td>CR (%)↓</td><td>v(m/s)↑</td><td>|ωz|(rad/s).↓</td></tr><tr><td colspan="8">Existing Method</td></tr><tr><td>Random</td><td>0.0</td><td>0.2</td><td>138.0</td><td>0.0</td><td>42.5</td><td>0.588</td><td>0.215</td></tr><tr><td>CMA [3]</td><td>11.5</td><td>13.6</td><td>86.4</td><td>9.6</td><td>34.5</td><td>0.811</td><td>0.096</td></tr><tr><td>TravelUAV [12]</td><td>20.1</td><td>53.9</td><td>55.9</td><td>16.4</td><td>16.9</td><td>0.917</td><td>0.100</td></tr><tr><td>AerialVLA [52]</td><td>52.3</td><td>55.0</td><td>32.0</td><td>36.9</td><td>28.3</td><td>0.648</td><td>0.113</td></tr><tr><td colspan="8">Zero-shot VLM Backbones on KinoUAV</td></tr><tr><td>Qwen3VL-2B</td><td>0.2</td><td>1.0</td><td>125.4</td><td>0.1</td><td>23.0</td><td>1.275</td><td>0.040</td></tr><tr><td>Qwen3VL-4B</td><td>1.8</td><td>42.3</td><td>89.0</td><td>0.3</td><td>43.3</td><td>1.318</td><td>0.078</td></tr><tr><td>Qwen3VL-8B</td><td>3.7</td><td>53.6</td><td>89.4</td><td>2.9</td><td>32.8</td><td>1.248</td><td>0.081</td></tr><tr><td colspan="8">Fine-tuned VLM Backbones on KinoUAV</td></tr><tr><td>InternVL3-2B</td><td>75.8</td><td>78.9</td><td>19.0</td><td>60.2</td><td>14.5</td><td>1.602</td><td>0.042</td></tr><tr><td>InternVL3-8B</td><td>78.5</td><td>81.1</td><td>9.8</td><td>62.4</td><td>12.8</td><td>1.606</td><td>0.051</td></tr><tr><td>LFM2.5VL-1.6B</td><td>78.3</td><td>80.7</td><td>12.6</td><td>62.1</td><td>14.4</td><td>1.686</td><td>0.046</td></tr><tr><td>LLaVA1.5-7B</td><td>83.6</td><td>84.6</td><td>15.0</td><td>67.3</td><td>9.2</td><td>1.790</td><td>0.041</td></tr><tr><td>Qwen3VL-2B</td><td>83.6</td><td>86.8</td><td>16.2</td><td>67.4</td><td>7.0</td><td>1.582</td><td>0.035</td></tr><tr><td>Qwen3VL-4B</td><td>87.3</td><td>89.8</td><td>10.0</td><td>71.3</td><td>6.5</td><td>1.675</td><td>0.044</td></tr><tr><td>Qwen3VL-8B (KinoUAV-Agent)</td><td>89.3</td><td>90.8</td><td>8.8</td><td>71.9</td><td>6.0</td><td>1.680</td><td>0.045</td></tr></table>

TABLE III  
NAVIGATION PERFORMANCE AND SELECTED KINEMATIC STATISTICS ON KINOUAV TEST UNSEEN OBJECT AND TEST UNSEEN MAP.
<table><tr><td>Method</td><td>SR (%)↑</td><td>OSR (%)↑</td><td>NE (m)↓</td><td>SPL (%)↑</td><td>CR (%)↓</td><td>v(m/s)↑</td><td>[ωk|(rad/s).↓</td></tr><tr><td colspan="8">Test Unseen Object</td></tr><tr><td>Random</td><td>0.0</td><td>0.0</td><td>152.1</td><td>0.0</td><td>32.8</td><td>0.634</td><td>0.189</td></tr><tr><td>CMA [3]</td><td>9.6</td><td>13.6</td><td>84.1</td><td>7.9</td><td>30.1</td><td>0.943</td><td>0.115</td></tr><tr><td>TravelUAV [12]</td><td>21.4</td><td>52.5</td><td>56.8</td><td>16.9</td><td>17.2</td><td>0.902</td><td>0.098</td></tr><tr><td>AerialVLA [52]</td><td>51.5</td><td>56.3</td><td>31.2</td><td>37.6</td><td>27.9</td><td>0.661</td><td>0.116</td></tr><tr><td>KinoUAV-Agent</td><td>87.1</td><td>89.7</td><td>8.6</td><td>72.4</td><td>6.2</td><td>1.652</td><td>0.044</td></tr><tr><td colspan="8">Test Unseen Map</td></tr><tr><td>Random</td><td>0.1</td><td>0.3</td><td>139.5</td><td>0</td><td>41.8</td><td>0.601</td><td>0.209</td></tr><tr><td>CMA [3]</td><td>3.4</td><td>6.1</td><td>105.8</td><td>1.9</td><td>32.4</td><td>0.928</td><td>0.121</td></tr><tr><td>TravelUAV [12]</td><td>12.7</td><td>31.2</td><td>68.2</td><td>9.1</td><td>24.5</td><td>0.931</td><td>0.103</td></tr><tr><td>AerialVLA [52]</td><td>39.4</td><td>47</td><td>40.7</td><td>26.2</td><td>33.7</td><td>0.637</td><td>0.11</td></tr><tr><td>KinoUAV-Agent</td><td>64.6</td><td>75.4</td><td>17.1</td><td>51.3</td><td>8.9</td><td>1.703</td><td>0.046</td></tr></table>

To assess the applicability of different vision-language models (VLMs) to our high-level planner, we fine-tune Qwen3-VL-2B/4B/8B-Instruct [53], InternVL3-2B/8B [54], LFM2.5-VL-1.6B [55], and LLaVA-1.5-7B [56]. All VLMs use the same dataset splits and inference pipeline. We additionally evaluate the three Qwen3-VL backbones in the zero-shot setting to examine the contribution of task-specific fine-tuning. In zeroshot testing, only the low-level modules are trained.

We evaluate navigation performance on KinoUAV Test Seen, Test Unseen Object, and Test Unseen Map. SR, OSR, NE, and SPL measure task completion, proximity to the target, and path efficiency. We additionally report CR, mean speed, and mean absolute yaw rate to characterize collision avoidance and flight behavior. These complementary metrics allow us to assess whether improved navigation success is accompanied by efficient motion and reduced heading variation.

1) Evaluation of KinoUAV Test Seen: We evaluate the navigation baselines and different vision-language model (VLM) backbones on KinoUAV Test Seen, as shown in Table II. Random Action fails to complete navigation tasks. CMA achieves only limited navigation success, while TravelUAV improves both SR and obstacle avoidance. AerialVLA achieves the highest SR among the existing methods, but its CR of 28.3% indicates that reliable obstacle avoidance remains challenging.

The zero-shot experiments show that general-purpose VLMs struggle with this task without fine-tuning. Although increasing the Qwen3-VL model size improves SR, task completion remains limited across all three zero-shot backbones. Finetuning substantially improves navigation success and safety. With the low-level controller unchanged, fine-tuning Qwen3- VL-8B increases SR from 3.7% to 89.3% and reduces CR from 32.8% to 6.0%, owing to its learned waypoint-planning capability. These results demonstrate the importance of adapt ing the high-level waypoint planner to the KinoUAV task.

All fine-tuned backbones achieve high success rates. The results show that larger models within the same series achieve higher SR and lower CR. Qwen3-VL-8B achieves the highest SR and lowest CR and is therefore used as KinoUAV-Agent. Compared with AerialVLA, it improves SR by 37.0 percentage points and reduces CR by 22.3 percentage points, while achieving a higher mean speed and a lower mean absolute yaw rate. These results show that KinoUAV-Agent improves navigation success and safety, accompanied by faster average motion and reduced heading variation.

TABLE IV  
ABLATION STUDY ON TRAINING SUBSETS AND DATA SCALE ON KINOUAV TEST SEEN. THE TRAINING FRACTION IS APPLIED TO EACH SELECTED SUBSET INDEPENDENTLY. ALL VARIANTS USE THE SAME SUPERVISED DFD BRANCH AND NAVIGATION PIPELINE.
<table><tr><td>Subsets</td><td>Fraction</td><td>SR (%)↑</td><td>OSR (%)↑</td><td>NE (m)↓</td><td>SPL (%)↑</td><td>CR (%)↓</td><td>v(m/s)↑</td><td>|ωk|(rad/s)↓</td></tr><tr><td>Only Main Subset</td><td>一</td><td>71.2</td><td>77.1</td><td>25.9</td><td>55.2</td><td>15.4</td><td>1.865</td><td>0.045</td></tr><tr><td>All</td><td>1%</td><td>0.9</td><td>10.5</td><td>170.4</td><td>0.2</td><td>68.5</td><td>1.583</td><td>0.023</td></tr><tr><td>All</td><td>10%</td><td>74.2</td><td>77.6</td><td>18.4</td><td>60.1</td><td>17.0</td><td>1.664</td><td>0.032</td></tr><tr><td>All</td><td>100%</td><td>89.3</td><td>90.8</td><td>8.8</td><td>71.9</td><td>6.0</td><td>1.680</td><td>0.045</td></tr></table>

TABLE V

ABLATION OF NAVIGATION AND CONTROL CONFIGURATIONS ON KINOUAV TEST SEEN. THE FULL CONFIGURATION USES CRUISE–SEEK GUIDANCE, INITIALIZES THE UAV HEADING TOWARD THE NEXT GROUND-TRUTH WAYPOINT, AND PROJECTS PREDICTED ACCELERATIONS ONTO A<sub>UAV</sub>. EACH VARIANT MODIFIES THE INDICATED SETTING WHILE RETAINING THE REMAINING SETTINGS OF THE FULL CONFIGURATION. MEAN SPEED AND MEAN ABSOLUTE YAW RATE ARE REPORTED IN m/s AND rad/s, RESPECTIVELY.
<table><tr><td>Configuration</td><td>SR (%)↑ OSR (%)↑</td><td>NE (m)↓</td><td>SPL (%)↑</td><td>CR (%)↓</td><td>v(m/s)↑</td><td>|ωk|(rad/s).↓</td></tr><tr><td>Single stage: Without distance</td><td>79.5</td><td>80.4</td><td>12.0</td><td>64.9 17.9</td><td>1.698</td><td>0.037</td></tr><tr><td>Cruise-Seek: Without distance</td><td>76.8</td><td>83.9</td><td>16.7 59.8</td><td>12.5</td><td>1.674</td><td>0.043</td></tr><tr><td>Random initial heading</td><td>82.1</td><td>83.0</td><td>13.6 58.1</td><td>11.6</td><td>1.528</td><td>0.059</td></tr><tr><td>Without acceleration projection</td><td>87.9</td><td>88.6</td><td>9.7 70.5</td><td>9.8</td><td>1.741</td><td>0.083</td></tr><tr><td>Full configuration (KinoUAV-Agent)</td><td>89.3</td><td>90.8</td><td>8.8</td><td>71.9 6.0</td><td>1.680</td><td>0.045</td></tr></table>

2) Evaluation of KinoUAV Test Unseen Object: We evaluated the model’s generalization capability to unseen target objects, and the results are presented in Table III. KinoUAV-Agent achieved the highest SR of 87.1%, outperforming the strongest competing method, AerialVLA, by 35.6 percentage points. It also achieved the lowest CR of 6.2%, whereas TravelUAV—the existing method with the best safety performance—exhibited a collision rate of 17.2%. These results demonstrate that KinoUAV-Agent can successfully navigate to unseen targets while effectively avoiding collisions.

Among all evaluated methods, KinoUAV-Agent achieved the highest average speed and the smallest heading change, reflecting faster movement and smoother heading adjustments. Compared to the ”Test Seen” scenario, there were no significant changes in its success rate, collision rate, or kinematic statistics. This indicates that the navigation strategy learned by the model generalizes effectively to target objects not encountered during training.

3) Evaluation of KinoUAV Test Unseen Map: We further evaluated the model’s generalization capability for navigation in unseen scenarios. With the experimental setup remaining unchanged, the results comparing various methods are presented in Table III; KinoUAV-Agent achieved the highest SR of 64.6%, outperforming AerialVLA by 25.2 percentage points. Simultaneously, it maintained the lowest CR, demonstrating superior safety compared to existing methods. These results indicate that KinoUAV-Agent retains its advantages in navigation success and safety even in unfamiliar environments. Its average speed and average absolute yaw rate demonstrate consistent motion characteristics across the three test subsets. Overall, the proposed model is capable of generalizing to unseen target objects and environments.

## D. Ablation Studies

We examine how training data, DFD supervision, and navigation strategy affect KinoUAV-Agent. Except for the setting being tested, all variants follow the same training and evaluation settings as the full model. We retrain models when changing their training data, supervision, or prediction targets. Changes to test-time inputs or action quantization do not require retraining. All experiments are conducted on the KinoUAV Test Seen set.

1) Analysis of the training dataset: We first examine the effects of training data composition and size. The Main Subset provides full-navigation trajectories for training both the highlevel planner and the low-level controller. Auxiliary Subset provides additional trajectories for training the low-level controller to follow random waypoints. To test the value of these additional trajectories, we fix the high-level planner trained on the full Main Subset. We then retrain the low-level controller using the Main Subset alone or together with Auxiliary Subset. To study the effect of data size, we sample 1% and 10% of the training trajectories from each set and compare them with the full-data setting. Both modules are retrained with the sampled data used for their respective tasks. As shown in Table IV, adding Auxiliary Subset increases SR by 18.1 percentage points. Navigation error and collision rate also decrease, while path efficiency improves. The random waypoints require the UAV to follow more varied directions. Training on these trajectories helps the controller track waypoints and avoid obstacles during full navigation. With only 1% of the training trajectories, the model rarely completes the task and has the highest collision rate. It also has the lowest mean absolute yaw rate. This suggests that the UAV makes few turns, which may limit its ability to avoid obstacles. Increasing the training data to 10% improves success rate and reduces collisions. Using the full training data further improves performance. The model trained on 1% of the data also performs worse than zero-shot Qwen3VL-8B in Table II. However, the low-level controller in the zero-shot setting is trained on the full training data, whereas both modules in the 1% setting use limited data. This difference may partly explain the lower performance of the 1% setting: even when the planner predicts a suitable waypoint, a poorly trained controller may fail to reach it safely.

![](images/dae0b825aa3945c7a518f7a347995df251ed50b9560ee7ea9f2136cebac47645.jpg)  
Fig. 7. Qualitative comparison on the same navigation episode: (a) without the DFD branch, (b) without DFD supervision, and (c) the full model. The vectors and arrows indicate the selected acceleration actions.

2) Analysis of the Navigation Strategy: Table V compares different navigation settings. The full model receives discrete target direction and distance during Cruise and a target description during Seek. Its initial heading is aligned with the next ground-truth waypoint. Before execution, each predicted acceleration is mapped to a predefined action in ${ \mathcal { A } } _ { \mathrm { U A V } }$ . We test two changes to navigation guidance. The first removes the Cruise–Seek stages and provides discrete target direction and description throughout navigation, without distance information. The second keeps both stages but removes distance information during Cruise. We retrain the high-level planner for each setting and keep the low-level controller fixed. We also test random initial heading and acceleration execution without quantization. These two changes are applied at test time without retraining. The single-stage model without distance information has the highest collision rate. Keeping the two stages reduces collisions but also lowers SR. Thus, stage separation alone does not improve task success. Adding distance information improves SR and further reduces collisions. Distance information may help the planner judge how close the UAV is to the target and choose suitable waypoints. Random initial heading reduces SR by 7.2 percentage points, while increasing CR by 5.6 percentage points. Mean absolute yaw rate increases, while mean speed decreases. The UAV may need to turn more at the start to face the intended route. Initial alignment helps navigation, although the model can still complete tasks with random initial headings. Removing acceleration quantization gives a similar SR but increases collisions. Mean speed and mean absolute yaw rate also increase, suggesting faster flight and more heading changes. These results show that quantization helps reduce collision risk without a large change in task success. Whether similar output constraints benefit TravelUAV or AerialVLA requires separate experiments.

3) Analysis of the DFD: We first remove the DFD branch and retrain the low-level controller. The acceleration predictor then uses only image features and the navigation state:

$$
\begin{array} { r } { \mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } = f _ { \mathrm { a c t i o n } } ^ { \prime } \left( \mathbf { f } _ { \mathrm { i m g } } ^ { k } , \mathbf { s } _ { i } ^ { k } \right) . } \end{array}\tag{34}
$$

We also train a model that keeps the DFD branch but removes the distance loss:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { a c t i o n } } \mathcal { L } _ { \mathrm { a c t i o n } } . } \end{array}\tag{35}
$$

In this setting, the DFD branch is trained only through the action loss. All other training settings remain unchanged. To test how DFD errors affect navigation, we add Gaussian noise to the predicted DFDs before passing them to the acceleration predictor:

$$
\begin{array} { r } { \mathbf { a } _ { \mathrm { U A V , r a w } } ^ { k } = f _ { \mathrm { a c t i o n } } \left( \mathbf { f } _ { \mathrm { i m g } } ^ { k } , \mathbf { s } _ { i } ^ { k } , \mathbf { d } ^ { k } + \sigma _ { n } \mathbf { n } ^ { k } \right) , } \end{array}\tag{36}
$$

where each component of $\mathbf { n } ^ { k }$ is independently sampled from $\mathcal { N } ( 0 , 1 )$ , and $\sigma _ { n }$ controls the noise level. We use the fully trained model and keep its weights fixed. As shown in Table VI, removing the DFD branch lowers SR to 81.1% and increases CR by 4.3 percentage points compared with the full model. Keeping the branch without distance supervision gives a higher SR and a lower CR than removing it. The model trained with distance supervision performs best. The branch therefore helps action prediction, and directly supervising its distance estimates brings further gains. As $\sigma _ { n }$ increases from 0 to 0.1, 0.3, and 0.5, SR decreases and NE increases. At $\sigma _ { n } ~ = ~ 0 . 1$ CR remains close to the result without noise, showing that the controller can tolerate small DFD errors. Larger noise leads to poorer navigation, with CR reaching 10.7% at $\sigma _ { n } = 0 . 5$ . These results suggest that the controller uses DFD estimates to select actions and that inaccurate estimates can reduce its ability to avoid obstacles. Mean speed and mean absolute yaw rate change only slightly across the noise settings. Unlike SR and NE, they do not show a clear trend as noise increases. A lower yaw rate does not always mean better performance: the UAV may turn less but collide more often. In these experiments, the main gains from DFD supervision are a higher success rate and fewer collisions. Fig. 7 compares the three models on the same test episode.

TABLE VI  
ANALYSIS OF DFD SUPERVISION AND NOISE ROBUSTNESS ON KINOUAV TEST SEEN. NOISE IS ADDED TO THE SUPERVISED DFD VARIANT.
<table><tr><td>DFD Configuration</td><td>SR (%)↑</td><td>OSR (%)↑</td><td>NE (m)↓</td><td>SPL (%)↑</td><td>CR (%)↓</td><td>v(m/s)↑</td><td>|ωz|(rad/s).↓</td></tr><tr><td>No DFD branch</td><td>81.1</td><td>83.5</td><td>11.6</td><td>65.2</td><td>10.3</td><td>1.796</td><td>0.040</td></tr><tr><td>Predicted DFD, no supervision</td><td>85.7</td><td>88.4</td><td>10.3</td><td>68.9</td><td>8.5</td><td>1.791</td><td>0.044</td></tr><tr><td>Predicted DFD, supervised (KinoUAV-Agent)</td><td>89.3</td><td>90.8</td><td>8.8</td><td>71.9</td><td>6.0</td><td>1.680</td><td>0.045</td></tr><tr><td>+ noise level 0.1</td><td>86.7</td><td>88.9</td><td>10.1</td><td>69.3</td><td>6.3</td><td>1.640</td><td>0.040</td></tr><tr><td>+ noise level 0.3</td><td>85.0</td><td>86.8</td><td>12.3</td><td>68.1</td><td>8.0</td><td>1.706</td><td>0.036</td></tr><tr><td>+ noise level 0.5</td><td>83.3</td><td>84.8</td><td>13.6</td><td>67.8</td><td>10.7</td><td>1.741</td><td>0.040</td></tr></table>

Maximum Absolute Yaw Rate  
![](images/8f56a6a2f1f8db133268b78d26fd8ee64704d497bd6eab5b375c1f6c163b13d1.jpg)  
Fig. 8. Qualitative comparison of generated trajectories, maximum absolute yaw rates, and navigation outcomes for TravelUAV, AerialVLA, and KinoUAV.

Without the DFD branch, the controller repeatedly selects downward acceleration and eventually hits the ground, as shown in Fig. 7(a). Without distance supervision, the UAV continues accelerating forward without gaining enough height to clear the roof and collides with it in Fig. 7(b). The fully supervised model moves sideways and accelerates upward to pass the obstacle, completing the task in Fig. 7(c). This example shows how DFD supervision helps the controller choose actions that keep the UAV clear of nearby obstacles.

## E. Generated Trajectory Analysis

Fig. 8 compares the trajectories of TravelUAV, AerialVLA, and KinoUAV in an urban navigation task. The middle columns show the UAV views near the moment of maximum absolute yaw rate. The rightmost column shows the final result. TravelUAV changes direction repeatedly within a small area. Both TravelUAV and AerialVLA eventually collide with nearby structures, while KinoUAV follows a smoother path along the street and reaches the target.

The maximum absolute yaw rate is 0.393 rad/s for KinoUAV, lower than 0.967 rad/s for TravelUAV and 0.857 rad/s for AerialVLA. This shows that KinoUAV makes less abrupt heading changes in this example. Its high-level planner selects waypoints, and its low-level controller adjusts acceleration to follow them. The resulting trajectory shows how this design allows the UAV to change its motion gradually and pass nearby obstacles safely.

## V. CONCLUSION

This paper presents KinoUAV, a benchmark for safe and stable UAV vision-language navigation. KinoUAV includes kinodynamic modeling in an automatic data collection pipeline. It combines global waypoint planning with local kinodynamic A<sup>∗</sup> search to generate trajectories using acceleration actions. The search accounts for velocity-dependent drag and motion constraints. The benchmark contains 300K trajectories from 16 simulated scenes, divided into the Main Subset and Auxiliary Subset. The former provides full-navigation trajectories, while the latter adds training data for low-level waypoint tracking. Each trajectory includes visual observations recorded with the corresponding motion states and directional free-space distance (DFD) labels.

We also develop KinoUAV-Agent, which uses a high-level planner to predict waypoints and a low-level controller to select acceleration actions. During Cruise, the planner receives the target direction and distance. During Seek, it uses the target description to search for the object. The low-level controller learns to predict DFDs from RGB images and uses these estimates to help select actions. Tests in seen environments, with unseen target objects, and on a held-out map show higher navigation success and fewer collisions than the evaluated baselines. The ablation results show that adding waypoint-tracking data and using more training data improve navigation performance. DFD supervision also helps reduce collisions. Within the Cruise–Seek strategy, distance information improves task success, while a lower yaw rate alone does not guarantee better navigation.

KinoUAV allows researchers to evaluate navigation success together with collision risk and UAV motion. The current benchmark uses simulated scenes and a simplified dynamics model. Future work will include more realistic motion disturbances and moving obstacles. We will also study how the trained models can be transferred to real UAVs.

## REFERENCES

[1] H. Shakhatreh, A. H. Sawalmeh, A. Al-Fuqaha, Z. Dou, E. Almaita, I. Khalil, N. S. Othman, A. Khreishah, and M. Guizani, “Unmanned aerial vehicles (uavs): A survey on civil applications and key research challenges,” IEEE access, vol. 7, pp. 48 572–48 634, 2019.

[2] Q. Sun, L. Chen, B. Zhang, X. Qu, Y. Cui, M. Shu, and X. Gu, “Evaluation of growth recovery grade in lodging maize via uav-based hyperspectral images,” Journal of Remote Sensing, vol. 4, p. 0253, 2024.

[3] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sunderhauf,¨ I. Reid, S. Gould, and A. Van Den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3674–3683.

[4] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-acrossroom: Multilingual vision-and-language navigation with dense spatiotemporal grounding,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 4392–4412.

[5] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” in European Conference on Computer Vision. Springer, 2020, pp. 104– 120.

[6] Y. Qi, Q. Wu, P. Anderson, X. Wang, W. Y. Wang, C. Shen, and A. v. d. Hengel, “Reverie: Remote embodied visual referring expression in real indoor environments,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, p. 9979–9988.

[7] H. Chen, A. Suhr, D. Misra, N. Snavely, and Y. Artzi, “Touchdown: Natural language navigation and spatial reasoning in visual street environments,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2019, p. 12530–12539.

[8] J. Thomason, M. Murray, M. Cakmak, and L. Zettlemoyer, “Vision-anddialog navigation,” in Conference on Robot Learning. PMLR, 2020, pp. 394–406.

[9] M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox, “Alfred: A benchmark for interpreting grounded instructions for everyday tasks,” in 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2020, pp. 10 737–10 746.

[10] Y. Fan, W. Chen, T. Jiang, C. Zhou, Y. Zhang, and X. Wang, “Aerial vision-and-dialog navigation,” in Findings of the Association for Computational Linguistics: ACL 2023. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 3043–3061.

[11] S. Liu, H. Zhang, Y. Qi, P. Wang, Y. Zhang, and Q. Wu, “AerialVLN: Vision-and-language navigation for uavs,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15 338––15 348.

[12] X. Wang, D. Yang, H. Kwan, J. Chen, H. Li, Y. Liao, S. Liu et al., “Towards realistic uav vision-language navigation: Platform, benchmark, and methodology,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 7292–7310.

[13] Y. Gao, C. Li, Z. You, J. Liu, Z. Li, P. Chen, Q. Chen, Z. Tang, L. Wang, P. Yang et al., “Openfly: A comprehensive platform for aerial visionlanguage navigation,” arXiv preprint arXiv:2502.18041, 2025.

[14] J. Xiao, Y. Sun, Y. Shao, B. Gan, R. Liu, Y. Wu, W. Guan, and X. Deng, “UAV-ON: A benchmark for open-world object goal navigation with aerial agents,” in Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025, p. 13023–13029.

[15] W. Jiang, H. Liang, L. Wang, K. Huang, W. Xu, W. Fan, J. Liu, S. Liu, H. Duan, B. Xu et al., “Dynfly: Dynamic-aware continuous trajectory generation for uav vision-language navigation in urban environments,” arXiv preprint arXiv:2606.31654, 2026.

[16] R. Wu, Y. Zhang, J. Chen, L. Huang, S. Zhang, X. Zhou, L. Wang, and S. Liu, “Aeroduo: Aerial duo for uav-based vision and language navigation,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 2576–2585.

[17] T. Li, T. Huai, Z. Li, Y. Gao, H. Li, and X. Zheng, “SkyVLN: Visionand-language navigation and NMPC control for uavs in urban environments,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 17 199–17 206.

[18] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Trans. Syst. Sci. Cybern., vol. 4, no. 2, pp. 100–107, 1968.

[19] D. Dolgov, S. Thrun, M. Montemerlo, and J. Diebel, “Practical search techniques in path planning for autonomous driving,” in Proceedings of the First International Symposium on Search Techniques in Artificial Intelligence and Robotics (STAIR-08). Chicago, USA: AAAI, Jun. 2008.

[20] B. Donald, P. Xavier, J. Canny, and J. Reif, “Kinodynamic motion planning,” Journal of the ACM (JACM), vol. 40, no. 5, pp. 1048–1066, 1993.

[21] M. Pivtoraiko, R. A. Knepper, and A. Kelly, “Differentially constrained mobile robot motion planning in state lattices,” Journal of Field Robotics, vol. 26, no. 3, pp. 308–333, 2009.

[22] B. Zhou, F. Gao, L. Wang, C. Liu, and S. Shen, “Robust and efficient quadrotor trajectory generation for fast autonomous flight,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3529–3536, 2019.

[23] J. Ortiz-Haro, W. Honig, V. N. Hartmann, and M. Toussaint, “iDb-¨ A\*: Iterative search and optimization for optimal kinodynamic motion planning,” IEEE Transactions on Robotics, vol. 41, pp. 2031–2049, 2025.

[24] J. Ortiz-Haro, W. Honig, V. N. Hartmann, M. Toussaint, and L. Righetti,¨ “iDb-RRT: Sampling-based kinodynamic motion planning with motion primitives and trajectory optimization,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 10 702–10 709.

[25] V. Jain, G. Magalhaes, A. Ku, A. Vaswani, E. Ie, and J. Baldridge, “Stay on the path: Instruction fidelity in vision-and-language navigation,” in Proceedings of the 57th annual meeting of the association for computational linguistics, 2019, pp. 1862–1872.

[26] F. Zhu, X. Liang, Y. Zhu, Q. Yu, X. Chang, and X. Liang, “Soon: Scenario oriented object navigation with graph-based exploration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, p. 12684–12694.

[27] D. Misra, A. Bennett, V. Blukis, E. Niklasson, M. Shatkhin, and Y. Artzi, “Mapping instructions to actions in 3d environments with visual goal prediction,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 2667–2678.

[28] J. Lee, T. Miyanishi, S. Kurita, K. Sakamoto, D. Azuma, Y. Matsuo, and N. Inoue, “Citynav: A large-scale dataset for real-world aerial navigation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 5912–5922.

[29] D. Fried, R. Hu, V. Cirik, A. Rohrbach, J. Andreas, L.-P. Morency, T. Berg-Kirkpatrick, K. Saenko, D. Klein, and T. Darrell, “Speakerfollower models for vision-and-language navigation,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[30] H. Tan, L. Yu, and M. Bansal, “Learning to navigate unseen environments: Back translation with environmental dropout,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 2610–2621.

[31] X. Wang, Q. Huang, A. Celikyilmaz, J. Gao, D. Shen, Y.-F. Wang, W. Y. Wang, and L. Zhang, “Reinforced cross-modal matching and selfsupervised imitation learning for vision-language navigation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 6622–6631.

[32] C.-Y. Ma, J. Lu, Z. Wu, G. AlRegib, Z. Kira, R. Socher, and C. Xiong, “Self-monitoring navigation agent via auxiliary progress estimation,” arXiv preprint arXiv:1901.03035, 2019.

[33] C.-Y. Ma, Z. Wu, G. AlRegib, C. Xiong, and Z. Kira, “The regretful agent: Heuristic-aided navigation through progress estimation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 6725–6733.

[34] W. Hao, C. Li, X. Li, L. Carin, and J. Gao, “Towards learning a generic agent for vision-and-language navigation via pre-training,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 13 134–13 143.

[35] A. Majumdar, A. Shrivastava, S. Lee, P. Anderson, D. Parikh, and D. Batra, “Improving vision-and-language navigation with image-text pairs from the web,” in European Conference on Computer Vision. Springer, 2020, pp. 259–274.

[36] Y. Hong, Q. Wu, Y. Qi, C. Rodriguez-Opazo, and S. Gould, “Vln bert: A recurrent vision-and-language bert for navigation,” in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2021, pp. 1643–1653.

[37] S. Chen, P.-L. Guhur, C. Schmid, and I. Laptev, “History aware multimodal transformer for vision-and-language navigation,” Advances in neural information processing systems, vol. 34, pp. 5834–5847, 2021.

[38] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Think global, act local: Dual-scale graph transformer for vision-and-language navigation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 16 516–16 526.

[39] G. Georgakis, K. Schmeckpeper, K. Wanchoo, S. Dan, E. Miltsakaki, D. Roth, and K. Daniilidis, “Cross-modal map learning for vision and language navigation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 15 439– 15 449.

[40] D. An, Y. Qi, Y. Li, Y. Huang, L. Wang, T. Tan, and J. Shao, “Bevbert: Multimodal map pre-training for language-guided navigation,” arXiv preprint arXiv:2212.04385, 2022.

[41] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “GridMM: Grid memory map for vision-and-language navigation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 15.579–15.590.

[42] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “Etpnav: Evolving topological planning for vision-language navigation in continuous environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5130–5145, 2024.

[43] G. Zhou, Y. Hong, and Q. Wu, “Navgpt: Explicit reasoning in visionand-language navigation with large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7641–7649.

[44] G. Zhou, Y. Hong, Z. Wang, X. E. Wang, and Q. Wu, “NavGPT-2: Unleashing navigational reasoning capability for large vision-language models,” in European Conference on Computer Vision. Springer, 2025, pp. 260–278.

[45] J. Chen, B. Lin, R. Xu, Z. Chai, X. Liang, and K.-Y. Wong, “Mapgpt: Map-guided prompting with adaptive path planning for vision-andlanguage navigation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 9796–9810.

[46] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “Instructnav: Zeroshot system for generic instruction navigation in unexplored environment,” arXiv preprint arXiv:2406.04882, 2024.

[47] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang, “Navid: Video-based vlm plans the next step for visionand-language navigation,” arXiv preprint arXiv:2402.15852, 2024.

[48] J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang, “Uni-NaVid: A video-based vision-language-action model for unifying embodied navigation tasks,” in Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, Jun. 2025.

[49] H. Cai, J. Dong, J. Tan, J. Deng, S. Li, Z. Gao, H. Wang, Z. Su, A. Sumalee, and R. Zhong, “Flightgpt: Towards generalizable and interpretable uav vision-and-language navigation with vision-language models,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 6670–6687.

[50] H. Xu, Y. Hu, C. Gao, Z. Zhu, Y. Zhao, and Q. Yin, “Geonav: Empowering mllms with dual-scale geospatial reasoning for languagegoal aerial navigation,” Pattern Recognition, p. 113365, 2026.

[51] D. Shao, Z. Xu, P. Wang, L. Liu, Y. Wang, J. Shi, and J. Huo, “Finecognav: Integrating fine-grained cognitive modules for zero-shot multimodal uav navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 1325–1334.

[52] P. Xu, Z. Deng, J. Deng, Z. Gu, and S. Wan, “Aerialvla: A visionlanguage-action model for uav navigation via minimalist end-to-end control,” arXiv preprint arXiv:2603.14363, 2026.

[53] S. Bai, Y. Cai, R. Chen et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[54] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[55] L. AI, “Lfm2 technical report,” arXiv preprint arXiv:2511.23404, 2025.

[56] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 286—-26 296.