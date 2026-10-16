# LICOA: Learning Instruction-Conditioned Optical Actions for Mobile Spatial Augmented Reality

Yuchen Deng Southwest University

Shunkai Zhou <sup>†</sup> Peking University

Ruixuan Zhu <sup>‡</sup> Southwest University

Haibin Ling <sup>§</sup> Westlake University

Bingyao Huang <sup>¶</sup> Southwest University

![](images/b1f617ea2d27e8d295197c6091e79cdee3a183d133299ad897d63f3feacc984f.jpg)  
Figure 1: LICOA makes projected light an embodied action. Left: The mobile optical embodiment combines a differential-drive chassis for room-scale approach and repositioning with a 2DoF steerable mirror for rapid local beam redirection and projection scale for footprint control. External and onboard RGB cameras provide complementary scene observations. Center: A matched simulation corpus supports adapting pretrained action models through a shared perception and optical action interface, which maps a language instruction, the external and onboard RGB images, and the robot control state to action chunks for the same physical components. Upper right: Selected frames from two real-world rollouts of the displayed instruction, starting from different initial configurations (top and bottom rows); the indices indicate progression within each rollout. Both executions coordinate chassis motion and mirror steering and end with the projection placed on the instructed target. Complete rollouts are shown in the supplementary video. Lower right: Structured light reconstruction and projector compensation results illustrate additional operations after spatial placement.

## ABSTRACT

Spatial augmented reality (SAR) augments physical environments with projected content, but most systems assume that the projector pose, target region, or motion path has been fixed or specified. We instead make projected light an embodied action: given a language instruction and current visual observations, a system must decide how to move and where and at what scale to place the projection. We introduce LICOA to realize this capability. Its differential-drive chassis provides room-scale approach and repositioning, while a steerable mirror with two degrees of freedom (2DoF) and projection scale provide rapid local control of the beam. A shared interface maps synchronized external and onboard RGB cameras with robot control state to physical optical actions across pretrained action models. Because instruction-conditioned optical action lacks dedicated training data and a common evaluation protocol, we construct a matched simulation embodiment and a corpus with 150 tasks and 7,500 rendered episodes, then evaluate models trained only on this corpus in previously unseen real-world environments. Across 480 complete closed-loop real-world execution trials (rollouts) on 12 tasks, the best adapted action model achieves a 57.50% success rate and 40.64% success weighted by path length, compared with 8.33% and 5.15% for the same architecture trained from scratch. A further 150 real-world ablation rollouts support the need for both camera views and coordinated control of the chassis and mirror. These results establish a physical platform, shared perception and optical action interface, corpus, and real-world evaluation protocol for treating projected light as an embodied action.

Index Terms: Spatial augmented reality, embodied intelligence, optical action, vision-language-action

## 1 INTRODUCTION

Spatial augmented reality (SAR) overlays digital visual content directly onto physical environments using projectors [2, 23, 35]. Unlike head-mounted or handheld displays, projection requires no worn or held device, supports shared viewing, and presents spatial cues for navigation, object localization, and collaborative work [24, 47, 49, 69]. Yet useful projection also requires placing light at the intended location, direction, and scale. Most existing SAR systems rely on fixed projector setups and predefined projection regions. Such configurations provide geometric calibration [14, 32, 62] and support diverse projection mapping applications [1, 12, 17, 18, 30, 31, 45, 54], but restrict the workspace to surfaces visible from a predetermined projector pose. Mobile and steerable systems expand the reachable workspace, but their targets, paths, or control rules are commonly predefined [19, 24, 42, 53, 57, 63, 64, 69]. They therefore do not address a general case in which a system enters an unfamiliar environment, interprets a language instruction, and decides both how to move and where to place light. Achieving this capability requires treating the placement of projected light as part of the system’s physical action. A language instruction such as “project onto the bear” specifies a desired physical effect rather than a target robot pose or a preselected display surface. The system must identify the referenced region, approach a feasible projection pose, redirect the beam, adjust its scale, and determine when placement is complete.

We call this task instruction-conditioned optical action. Throughout, physical describes tangible entities and actuator-level quantities, such as scene objects, the platform, and its actions, whereas realworld denotes the non-simulated deployment and evaluation domain. Realizing this capability introduces three practical challenges. First, room-scale coverage and local beam placement require different controls. Moving the projector offers broad reach but is slow and changes the sensing viewpoint, whereas beam steering is fast but locally bounded. Multi-projector installations provide coverage at the cost of fixed infrastructure and calibration [11, 36, 56], while arm-mounted systems provide local pose flexibility with a more complex mechanism and a workspace bounded by the arm [7,47,49]. A mobile SAR platform must therefore combine broad repositioning with fast local optical control in one action space. Second, pretrained action models target embodiments such as manipulators and grippers, with different camera inputs, state dimensions, action outputs, and training objectives [3, 4, 6, 21, 41, 51]. Training from scratch must learn both general vision-language grounding and optical action semantics from task-specific data. We therefore ask whether pretrained models can retain their grounding and action-generation knowledge while learning optical action from task-specific demonstrations [3, 5, 51]. Third, pretraining does not remove the need for task-specific data and evaluation. No existing corpus pairs spatial instructions and external and onboard RGB images with coordinated commands for the differential-drive chassis, a steerable mirror with two degrees of freedom (2DoF), and projection scale, plus episode termination outputs, and no common evaluation protocol assesses their physical outcomes. This absence is the underlying data problem; simulation is our means of addressing it. A useful real-world evaluation protocol must match the physical embodiment and sensing roles and test whether models trained without real-world demonstrations can act in unseen real-world settings.

To address these challenges, we introduce LICOA (Learning Instruction-Conditioned Optical Actions), a mobile SAR system that makes projected light an embodied action, as summarized in Fig. 1. First, its differential-drive chassis provides coarse room-scale approach and repositioning, whereas its 2DoF steerable mirror provides rapid local beam redirection and projection scale controls the footprint. The resulting action components jointly determine where light is placed. Second, LICOA combines timestamp-synchronized external and onboard RGB images with the robot control state in a shared perception and optical action interface. This interface maps the observations and action output of different pretrained action models [3, 52, 68] to the same physical embodiment. Third, we reproduce the embodiment in simulation [20, 59] and construct 150 spatial placement tasks with 7,500 rendered episodes. Models are fine-tuned only on this corpus and evaluated through real-world rollouts with new scenes, targets, instructions, and initial configurations. Experiments evaluate three action-generation paradigms, isolate pretraining with a scratch version of π<sub>0.5</sub> [3], and ablate the camera views and action components. Across 480 complete closedloop real-world executions (rollouts), adapted $\pi _ { 0 . 5 }$ achieves 57.50% success and 40.64% success weighted by path length, compared with 8.33% and 5.15% for the scratch control; another 150 rollouts support the ablations.

Our contributions can be summarized as follows:

• We formulate projected light as an instruction-conditioned optical action and realize it with a mobile optical platform whose differential-drive chassis, 2DoF steerable mirror, projection scale, and episode termination output form a coupled physical

action space.

• We define a shared perception and optical action interface comprising external and onboard RGB images, the robot control state, action chunks, and physical action components, and instantiate it with pretrained action models based on continuous action generation, autoregressive action tokenization, and world-action modeling.

• We establish a matched simulation corpus and real-world evaluation protocol for this new task, including 150 spatial placement tasks, 7,500 rendered episodes, fine-tuning only in simulation, a real-world evaluation task set of 12 previously unseen real-world tasks with 480 real-world evaluation rollouts, and 150 ablation rollouts that test the sensing and hardware design.

## 2 RELATED WORK

## 2.1 Spatial augmented reality and mobile projection

Spatial augmented reality (SAR) overlays digital information directly onto physical surfaces and supports shared viewing without worn or handheld displays [2, 23, 35]. To place an augmentation correctly, an autonomous SAR system must perceive candidate surfaces and semantic targets in the scene. If projected content is visible in the camera input, the system must also distinguish it from the underlying scene. Most projector-camera systems (ProCams) pipelines focus on geometric and photometric perception under a fixed, calibrated setup [14, 32, 62]. Temporal [34, 60] or spatial [32, 40] structured light patterns establish dense correspondences for calibra tion, shape recovery, and appearance control [39], while high-speed and infrared capture reduce motion-to-photon error on moving faces and hands [1, 18]. Recently, ProCap extends SAR perception from geometry and photometry to semantics by separating the physical scene from projected content and describing the two layers independently [8]. It therefore exposes the virtual-physical ambiguity relevant to visually guided SAR, but stops at scene description rather than selecting and executing a projection action. Once the scene and projector response are known, a complementary line of work determines what a configured projector should emit. Per-pixel correction and learned visual models compensate for geometric and photometric distortions [10, 22, 28, 29, 31, 33, 48, 55]. Neural rendering, path tracing, and Gaussian splatting model projectorcamera light transport for relighting, compensation, or optical simulation [12, 17, 30, 45]. Application-specific systems edit perceived softness, faces, glossy objects, or hands [1, 18, 39, 54], while neural deblurring and alternative optical paths address defocus, shadowing, and ambient illumination [25, 37, 38, 43, 67]. LAPIG further maps language prompts to surface-adapted, stylized content, but still assumes a calibrated projector and a given target surface [13]. Together, these perception and appearance methods define fixed Pro-Cams represented by the first row of Tab. 1, but they do not expose physical controls for deciding where light should be placed.

To distinguish configured projection from autonomous placement, Tab. 1 compares architecture classes along five requirements: projector-origin mobility, exposed action components, spatial instruction, learned closed-loop control, and a matched simulation interface. Fixed multi-projector systems enlarge an installed workspace by coordinating calibrated units, but their origins and coverage remain predetermined [11, 36, 50, 56, 58]. Fixed beam-steered systems add rapid local redirection through a rotating mirror or pantilt mechanism, yet cannot approach targets outside the useful range of their installed origin [42, 53, 64]. Arm-mounted projectors add local pose flexibility for gesture interaction, target tracking, or expressive movement, while remaining bounded by the arm workspace and mechanism complexity [7, 27, 47, 49]. Mounting a projector on a chassis makes its origin room-scale. With fixed optics, mobile systems detect suitable surfaces or visualize a planned navigation path, but the beam direction remains coupled to the chassis pose [19, 24]. Adding pan-tilt or gimbal actuation decouples local beam direction from chassis orientation: prior systems use this combination for marker-based tracking, scripted near-object overlays, or semanticmap and geometry-based surface search [57, 63, 69]. These systems establish the mechanical ingredients of mobile projection, but their spatial targets and motion are supplied by detected planes, markers, planned paths, scripts, maps, or scoring rules. None jointly learns closed-loop optical actions from current RGB observations and a natural-language spatial goal, and the optical simulators used by some fixed ProCams methods do not reproduce the mobile embodi ment, camera roles, robot control state, and action interface.

Table 1: Representative SAR architecture classes evaluated against the requirements of instruction-conditioned optical action. Cells summarize the range demonstrated by the cited systems in each class; an individual system may not support every listed capability. The comparison concerns physical projection placement rather than projected content quality.
<table><tr><td>System / architecture</td><td>Projector origin mobility</td><td>Action components</td><td></td><td>Spatial instruction Learned closed-loop Matched simulation control</td><td></td></tr><tr><td>Fixed ProCams [12, 13,17,30,45]</td><td>Fixed</td><td>No</td><td>Content prompt in some</td><td>No</td><td>Optics simulation in some</td></tr><tr><td>Fixed multi-projector systems [11,36, 56]</td><td>Fixed</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Fixed beam-steered [42,53,64]</td><td>Fixed</td><td>Optical steering only</td><td>Restricted speech / gesture in some</td><td>Rules; no learned action model</td><td>No</td></tr><tr><td>Arm-mounted [7,27,47,49]</td><td>Arm workspace</td><td>Arm only</td><td>Speech / gesture / tracking / script</td><td>Tracking / script; no learned action model</td><td>No</td></tr><tr><td>Mobile + fixed optics [19, 24,61]</td><td>Room-scale</td><td>Chassis; fixed optical axis</td><td>Detected surface</td><td>Geometry; no learned action model</td><td>No</td></tr><tr><td>Mobile + steerable optics [44,57,63,69]</td><td>Room-scale</td><td>Chassis + PT / gimbal</td><td>Marker / path / script / search</td><td>Servo / planning / script; no learned action model</td><td>No</td></tr><tr><td>LICOA (Ours)</td><td>Room-scale</td><td>Chassis + 2DoF mirror + projection scale</td><td>Language spatial instruction</td><td>Adapted action model</td><td>Matched embodiment + interfaces</td></tr></table>

Legend: green, fully supports the LICOA requirement; yellow, limited or specialized support; red, not demonstrated. Definitions: Projector origin mobility describes how far the physical projector can be repositioned, excluding beam steering from a fixed origin; action components identify the physical controls exposed by each system; spatial instruction specifies where and at what scale to project, not only what visual content to render; learned closed-loop control maps the current RGB images and instruction to physical projection actions and replans during execution (yellow denotes non-learned feedback or planning); matched simulation reproduces the embodiment, RGB images, robot control states, and actions under the shared interface; PT, pan-tilt.

LICOA instead connects a natural-language spatial goal and current observations to learned placement actions for a room-scale chassis, 2DoF steerable mirror, and projection scale. As summarized in Tab. 1, its distinction lies in the learned closed-loop optical action interface and matched embodiment.

## 2.2 Generalist action models for optical action

The hardware comparison in Tab. 1 identifies learned closed-loop control as a principal capability missing from prior mobile projection systems. LICOA addresses this gap not by designing another action-model backbone, but by adapting pretrained generalist action models to a new optical embodiment. This adaptation is nontrivial: unlike conventional manipulation, the model must coordinate differential-drive velocities, absolute mirror pan and tilt targets, projection scale, and termination from external and onboard RGB observations and the current robot control state. We therefore use generalist action models to study whether pretrained representations and diverse action-generation mechanisms transfer to instructionconditioned optical action.

Vision-language-action models. Vision-language-action (VLA) models learn mappings from visual observations and language instructions to executable controls. RT-1 and RT-2 established scalable language-conditioned robot policies and the transfer of seman tic knowledge from vision-language pretraining, while Open X-Embodiment, Octo, and OpenVLA extended this direction through cross-robot data and open generalist policies [5, 6, 21, 41, 51]. For LICOA, two action-generation paradigms are particularly relevant.

$\pi _ { 0 . 5 }$ uses a continuous flow-matching action expert, providing a direct mechanism for generating coupled optical action chunks [3,4]. FAST instead compresses continuous action sequences into discrete tokens, allowing $\pi _ { 0 } { \mathrm { - F A S T } }$ to generate action chunks autoregressively [52]. Instantiating both through the same optical action interface tests whether LICOA depends on one particular action representation. Comparing the adapted and scratch versions of π<sub>0.5</sub> additionally isolates whether pretrained visual-language and action representations transfer to optical placement.

World-action models. World-action models (WAMs) use visual dynamics as an additional learning signal. Earlier video-based robot models generate or encode possible visual futures and convert them into actions through inverse dynamics or downstream control [15,16,70]. More recent models jointly learn video evolution and robot actions, including GR-1, GR-2, Video Prediction Policy, Unified Video Action Model, and Unified World Models [9,26,46,65,71]. Fast-WAM retains this video-action co-training signal but predicts actions without explicitly generating future video at inference [68]. We include it as a third instantiation to test whether a video-derived dynamics prior transfers to optical action while remaining compati ble with LICOA’s closed-loop replanning interface.

Taken together, $\pi _ { 0 . 5 } , \pi _ { 0 } – \mathrm { F A S T } _ { \cdot }$ and Fast-WAM are not presented as an exhaustive survey or as directly comparable pretrained architectures. They instantiate three action-generation paradigms: continuous flow matching, autoregressive action tokenization, and video-action co-training. LICOA supplies each model with the same instruction, external and onboard RGB images, and robot control state, and maps its predictions to the same chassis, mirror, projectionscale, and termination actions. This controlled interface allows the experiments to test model transfer and interface compatibility across action-generation paradigms without changing the physical task or embodiment.

## 3 LICOA PHYSICAL PLATFORM

LICOA is designed around a real-world requirement: projected light must travel across a room and still support fast local placement. We meet these two scales of motion with complementary action components. The differential-drive chassis changes the projector origin for broad approach and repositioning, while the 2DoF steerable mir ror redirects the beam locally without rotating the entire platform. Projection scale changes the illuminated footprint, and an episode termination output marks completion. Together, these controls make the placement of light executable as an embodied action.

![](images/5cec3f8fa89d74d90c64e7d70155accd7c5a37151606702a07f5b810823b5bda.jpg)  
Figure 2: The LICOA physical platform and its complementary motion scales. Setup: The mobile projection robot and tripod-mounted external RGB camera. Rear: The Kobuki differential-drive chassis, supporting frame, and integrated electronics for room-scale approach and repositioning. Front: The onboard RGB camera and projector assembly. Top: The NVIDIA Jetson Nano and embedded control electronics. 2DoF steerable mirror and projector: The mirror mounted above the projector redirects the beam locally without rotating the differential-drive chassis. Projection scale controls the footprint size.

Table 2: Hardware roles in the physical LICOA platform. The chassis provides broad repositioning, while the mirror and projector provide local beam direction and scale control. Detailed interfaces, electronics, power distribution, and embedded control are provided in the Suppl. Material.
<table><tr><td>Subsystem</td><td>Hardware</td><td>Role</td></tr><tr><td>Differential- drive chassis</td><td>Kobuki mobile chassis</td><td>Room-scale approach and repositioning</td></tr><tr><td>Onboard sensing</td><td>Microsoft Kinect v1</td><td>Robot-centric RGB observation</td></tr><tr><td>External sensing</td><td>Logitech C270</td><td>Global RGB observation</td></tr><tr><td>Embedded computing</td><td>NVIDIA Jetson Nano</td><td>Robot-side sensing and execution bridge</td></tr><tr><td>Projection</td><td>Custom laser projector</td><td>Optical output and footprint scaling</td></tr><tr><td>Optical steering</td><td>Custom 2DoF steerable mirror</td><td>Rapid local beam redirection</td></tr><tr><td>Pan-tilt servo control</td><td>STM32F103C8T6</td><td>Low-level motor feedback control</td></tr></table>

## 3.1 Mobile optical embodiment

The hardware shown in Fig. 2 and summarized in Tab. 2 implements the separation between broad and local control. A rigidly mounted projector would require chassis motion for every change in beam direction; the steerable mirror instead provides pan and tilt while the chassis remains at a suitable pose. Conversely, mirror steering alone covers only a limited local range and cannot reposition the projector. The combined design therefore exposes chassis velocity, mirror pan and tilt, and projection scale as coupled action components. Two onboard battery packs support untethered operation.

The sensing design also serves two spatial roles. The onboard Kinect v1 camera observes the scene from the robot and changes viewpoint as the chassis moves. The external Logitech C270 RGB camera provides a global view of the robot, target region, and pro jected light. The robot-mounted NVIDIA Jetson Nano acquires the onboard RGB stream and maintains the robot control state accepted by the control stack: differential-drive chassis velocities, pan and tilt targets for the 2DoF steerable mirror, and projection scale. It sends the latest onboard RGB image, timestamp, and robot control state to the host PC as the robot stream, while the external camera stream reaches the host PC directly over USB. The host synchronizes both streams, forms an inference request, receives an action chunk, and relays its per-step actions to the physical action components.

On the physical platform, the Jetson dispatches differential-drive commands to the Kobuki, projection scale commands to the custom laser projector over HDMI, and absolute mirror pan and tilt targets to an STM32 over a universal asynchronous receiver-transmitter (UART) link via USB at 50 Hz per axis. The STM32 closes both pan-tilt motor loops locally at approximately 260 Hz using AS5600 encoder feedback. This local loop separates high-rate mirror control from action model inference. Detailed interfaces, calibration, power distribution, and motor control are provided in the Suppl. Material.

## 4 SHARED PERCEPTION AND OPTICAL ACTION INTERFACE

The platform becomes learnable through a shared interface that sep arates physical meaning from model implementation. Every action model receives the same instruction, external and onboard RGB im ages, and robot control state, then returns action chunks comprising commands for the differential-drive chassis, 2DoF steerable mirror, and projection scale, plus an episode termination output. Figure 3 summarizes this interface and its real-world execution loop.

## 4.1 Observation and action interface

At each replanning step t, the action model must combine local and global visual information. The host PC synchronizes the external camera stream with the robot stream to form an observation comprising the onboard image $\mathbf { I } _ { t } ^ { o } ,$ , external image I<sup>e</sup>, and robot control state s<sub>t</sub>, then sends it with instruction ℓ as an inference request. Each robot stream packet carries the latest onboard RGB image, its capture timestamp, and the robot control state, while the host continuously buffers timestamped images from the external camera stream. Let $\overline { { \tau } } _ { t } ^ { o }$ be the selected onboard image’s capture timestamp expressed on the host clock through periodic clock-offset estimation, and let $\tau _ { j } ^ { e }$ be the timestamp of buffered external image j. The host selects the latest external image not newer than the onboard image:

$$
j _ { t } = \operatorname* { m a x } \{ j \mid \tau _ { j } ^ { e } \leq \bar { \tau } _ { t } ^ { o } \} .\tag{1}
$$

The selected images are denoted by I<sup>o</sup> and I<sup>e</sup>. The robot control state $\mathbf { s } _ { t }$ is carried by the same robot stream packet as $\mathbf { I } _ { t } ^ { o }$ . Only the assembled observation at a replanning boundary is sent to the server; intermediate images from the raw camera streams are not individually forwarded. We write the robot control state as

$$
\mathbf { s } _ { t } = \left[ v _ { t } \quad \omega _ { t } \quad q _ { \mathrm { p } , t } \quad q _ { \mathrm { t } , t } \quad \rho _ { t } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { 5 } .\tag{2}
$$

Together with the external and onboard RGB images, it forms the timestamp-synchronized observation

$$
\begin{array} { r } { \mathbf { o } _ { t } = \left( \mathbf { I } _ { t } ^ { o } , \mathbf { I } _ { t } ^ { e } , \mathbf { s } _ { t } \right) . } \end{array}\tag{3}
$$

The shared index t therefore identifies one assembled replanning observation, rather than every image in either raw camera stream. Here, v and ω are the active body-frame linear and angular velocities of the differential-drive chassis in $\mathrm { m } / \mathrm { s }$ and rad $\mathbf { \Omega } / \mathrm { s } , q _ { \mathrm { p } , t }$ and $q _ { \mathrm { t } , t }$ are the absolute mirror pan and tilt targets of the 2DoF steerable mirror in radians, and $\bar { \rho } _ { t } \ \in \ [ 0 , 1 0 0 ]$ is the projection scale expressed as a percentage. Thus, s<sub>t</sub> is a synchronized robot control state rather than a complete physical robot state; in particular, it does not include the global chassis pose. The instruction ℓ conditions the action model separately because it is a task specification rather than a synchronized sensor or control measurement. The real-world and simulated platforms expose this same observation interface.

![](images/b00dcd5a175bd981ba11e8eb6437deb108eb2b3c77dc93e51f8ee9e254b56625.jpg)  
Figure 3: LICOA makes projected light executable through a shared perception and optical action interface. Top: The instruction ℓ, external and onboard RGB images, and five-dimensional robot control state pass through the encoders, backbone, and action module. Supervised fine-tuning adapts the action model to return H-step action chunks. Each six-dimensional action contains commands for the differential-drive chassis, 2DoF steerable mirror, and projection scale, plus an episode termination output; any additional model output dimensions are excluded. Bottom: The host PC synchronizes the external camera stream and robot stream, sends an inference request, executes a short prefix of the returned action chunk, and replans from fresh observations. Simulation training and real-world deployment use the same observation and action interfaces.

Given $( \mathbf { o } _ { t } , \ell )$ , the action model predicts an H-step action chunk. We separate the continuous physical control from the binary termination decision. For future offset i, the continuous control is

$$
\mathbf { u } _ { t + i | t } = \left[ v \quad \omega \quad q _ { \mathrm { p } } \quad q _ { \mathrm { t } } \quad \rho \right] _ { t + i | t } ^ { \mathsf { T } } \in \mathbb { R } ^ { 5 } .\tag{4}
$$

Appending the binary termination indicator gives the complete per-step action

$$
\begin{array} { r } { \mathbf { a } _ { t + i \mid t } = \left[ \mathbf { u } _ { t + i \mid t } ^ { \mathsf { T } } \quad b _ { t + i \mid t } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { 6 } . } \end{array}\tag{5}
$$

Stacking H successive actions gives the action chunk

$$
\mathbf { A } _ { t } = \left( \mathbf { a } _ { t \mid t } , \ldots , \mathbf { a } _ { t + H - 1 \mid t } \right) \in \mathbb { R } ^ { H \times 6 } .\tag{6}
$$

At the canonical observation and action interface, action model $\pi _ { \theta _ { m } }$ induces the conditional distribution

$$
\widehat { \bf A } _ { t } \sim \pi _ { \theta _ { m } } ( \cdot \mid { \bf o } _ { t } , \ell ) .\tag{7}
$$

Specifically, a sampled prediction is the ordered sequence

$$
\begin{array} { r l } & { \widehat { { \bf A } } _ { t } = \left( \widehat { { \bf a } } _ { t \mid t } , \ldots , \widehat { { \bf a } } _ { t + H - 1 \mid t } \right) \in \mathbb { R } ^ { H \times 6 } , } \\ & { \widehat { { \bf a } } _ { t + i \mid t } \in \mathbb { R } ^ { 6 } , \qquad 0 \leq i < H . } \end{array}\tag{8}
$$

Within $\mathbf { u } _ { t + i | t }$ , the first two entries are body-frame velocity commands for the differential-drive chassis, and the remaining three are absolute mirror pan and tilt targets for the 2DoF steerable mirror and the projection scale target. The binary indicator $b _ { t + i | t } \in \{ 0 , 1 \}$ requests episode termination when it equals one. During learning, this sixth target channel is stored as zero or one. At deployment, the decoded scalar $\widehat { b } _ { t + i \mid t } \in \mathbb { R }$ is converted to the termination decision by the controller threshold $\widehat { b } _ { t + i | t } > 0 . 5$ . Although $\mathbf { s } _ { t }$ and $\mathbf { u } _ { t + i | t }$ use the same five physical coordinates, the former records the current synchronized control values, whereas the latter specifies future controls. This canonical observation and action interface is independent of how action model family m internally represents or generates an action.

## 4.2 Connecting pretrained action models

Different pretrained action models expect different cameras, image layouts, state dimensions, action representations, and training targets. These implementation differences should not change the SAR task. We therefore keep the observation and action interfaces in Eqs. (3) and (5) fixed and place model-specific conversion only at their boundaries. At each replanning boundary, an input adapter maps the instruction, external and onboard RGB images, and robot control state to the camera slots and preprocessing expected by a model. A target adapter converts the demonstrated physical action into that model’s training representation, and an output adapter converts its prediction back to the LICOA action components.

Let $\xi _ { t }$ be a synchronized demonstration segment beginning at t. It contains $( \mathbf { o } _ { t } , \ell )$ , the target physical action chunk $\mathbf { A } _ { t } ^ { \mp }$ , and any aligned future visual observations required by a model’s native auxiliary objective. Let D denote the collection of these demonstration segments. For action model family m, the input adapter produces

$$
x _ { t } ^ { ( m ) } = \Phi _ { m } ^ { \mathrm { i n } } ( \mathbf { o } _ { t } , \ell ) .\tag{9}
$$

The target adapter converts the demonstration supervision into the corresponding target in the model’s native representation

$$
y _ { t } ^ { ( m ) } = \Phi _ { m } ^ { \mathrm { t g t } } ( \xi _ { t } ) .\tag{10}
$$

Let ${ \mathcal { L } } _ { m }$ denote the model’s native training objective, and let $\mathcal { U } _ { m }$ index its selected trainable modules, with $\hat { u } _ { m }$ denoting the complement. We minimize this objective over the selected modules:

$$
\theta _ { m , \mathcal { U } _ { m } } ^ { * } = \arg \operatorname* { m i n } _ { \theta _ { m , \mathcal { U } _ { m } } } \mathbb { E } _ { \xi _ { t } \sim \mathcal { D } } \left[ \mathcal { L } _ { m } \Big ( \theta _ { m } ; x _ { t } ^ { ( m ) } , y _ { t } ^ { ( m ) } \Big ) \right] .\tag{11}
$$

Here, $\Phi _ { m } ^ { \mathrm { i n } }$ maps the common observation to the model’s native input, including camera key mapping, resizing, normalization, and required state padding. The target adapter $\Phi _ { m } ^ { \mathrm { t g t } }$ expresses the same physical supervision in the model’s native form. It may return a normalized continuous chunk, an autoregressive sequence of action tokens, or a joint visual and action target. Accordingly, ${ \mathcal { L } } _ { m }$ remains the native objective of the pretrained action model family: it can incorporate noisy actions and a flow time variable, preceding action tokens, or an auxiliary loss on future visual observations without forcing all architectures to share one surrogate objective. In Eq. (11), $\theta _ { m }$ comprises the optimized parameters $\bar { \theta } _ { m , { \mathcal { U } } _ { m } }$ and a complementary set fixed at its pretrained value $\theta _ { m , \overline { { u } } _ { m } } ^ { 0 }$ . The minimizer $\theta _ { m , { \boldsymbol { u } } _ { m } } ^ { * }$ , together with this frozen complement, defines the full adapted parameters $\theta _ { m } ^ { * }$ This covers both full fine-tuning and partial post-training tailored to each model. Regardless of these internal choices, an output adapter maps every prediction back to the canonical physical action space in Eq. (5). Specific adapter and optimization settings are reported with the evaluated models in Sec. 6.1 and the Suppl. Material.

## 4.3 Closed-loop action execution

During deployment, the action model repeatedly observes the current state of the projection task rather than following a fixed trajectory. The timestamp-synchronized external and onboard RGB images, robot control state, and instruction are processed by the same input adapter used during training. Let $p _ { \theta _ { m } ^ { * } } ^ { ( m ) } ( \cdot \mid x _ { t } ^ { ( m ) } )$ denote the predic tive distribution induced by the adapted model in its native space; it covers both continuous sampling and autoregressive generation. The adapted action model first produces a prediction in that space

$$
\begin{array} { r } { \widehat { y } _ { t } ^ { ( m ) } \sim p _ { \theta _ { m } ^ { * } } ^ { ( m ) } \Big ( \cdot \mid x _ { t } ^ { ( m ) } \Big ) . } \end{array}\tag{12}
$$

The output adapter decodes it into physical actions

$$
\widehat { \mathbf { A } } _ { t } = \Phi _ { m } ^ { \mathrm { o u t } } \left( \widehat { y } _ { t } ^ { ( m ) } \right) .\tag{13}
$$

At the canonical observation and action interface, this yields the adapted action model

$$
\widehat { \mathbf { A } } _ { t } \sim \pi _ { \theta _ { m } ^ { * } } \big ( \cdot \mid \mathbf { o } _ { t } , \ell \big ) .\tag{14}
$$

Thus, Eqs. (12) and (13) give the realization specific to each model family of Eq. (14), which is the adapted instance of the generic physical action model in Eq. (7).

For $1 \leq h \leq H$ , the controller dispatches

$$
\widehat { \mathbf { A } } _ { t } ^ { ( h ) } = \left( \widehat { \mathbf { a } } _ { t \mid t } , \ldots , \widehat { \mathbf { a } } _ { t + h - 1 \mid t } \right) \in \mathbb { R } ^ { h \times 6 } .\tag{15}
$$

Many action chunk models maintain an internal queue so that successive calls can return actions from the same prediction. This is efficient during open-loop execution but can accidentally reuse a chunk generated from an old observation. We hence clear the queue at the beginning of every inference request, which becomes a new replanning boundary: the action model consumes the latest observation, its output adapter maps the prediction back to the six dimensional LICOA action interface, and the controller executes h = 8 actions before requesting a new chunk. Repeating this procedure at 15 Hz yields receding-horizon closed-loop action execution.

## 5 SIMULATION CORPUS AND EVALUATION PROTOCOL

Instruction-conditioned optical action has no established corpus that combines language, two camera roles, robot control state, and mobile optical action components. It also lacks a common task and evaluation protocol. We address both gaps with a matched simulation corpus for training and a real-world evaluation protocol.

Table 3: Composition of the LICOA matched simulation corpus. We use the first 10 InteriorAgent scenes, with five external camera viewpoints and three spatial placement tasks per viewpoint, yielding 150 tasks. Each source trajectory is rendered with the original appearance and four randomized variants in lighting, wall, and floor materials. Trajectory density and task diversity analyses are provided in the Suppl. Material.  
![](images/7cfa78f92f0d3bd305d6b71b6ef2e1acf881138bbdaa9cb4c5df62db1379dd65.jpg)

## 5.1 Matched simulation and data collection

We reproduce the physical LICOA embodiment in NVIDIA Isaac Sim [20] using InteriorAgent USD indoor scenes [59]. The simulation matches the physical kinematic chain, camera configuration, action ranges, and projection scale so that both embodiments share the observation and action interfaces defined in Sec. 4.1.

We collect expert demonstrations in simulation by teleoperating the differential-drive chassis, 2DoF steerable mirror, and projection scale according to a language instruction. Each accepted demonstration forms a source trajectory that starts from an initial chassis and mirror configuration and ends when the instructed projection is achieved or the maximum trajectory length is reached. Repeated source trajectories vary the initial configuration and target specification, producing both short local corrections and trajectories that require chassis repositioning before mirror adjustment. At each time step, we record the instruction, timestamp-synchronized external and onboard RGB images, the five-dimensional robot control state, and the corresponding six-dimensional action defined in Eqs. (2) and (5). The physical and simulated platforms serialize these observations and actions in the same trajectory format.

## 5.2 Task collection

The matched simulation corpus defines spatial placement tasks rather than content generation ones. We select the first 10 InteriorAgent indoor scenes, each a complete environment rather than an individual room or camera view. Within each scene, we define 5 fixed external camera viewpoints and construct three tasks per viewpoint, yielding 15 tasks per scene and 150 tasks in total. A task is specified by its scene and external camera viewpoint, an initial robot configuration, and a language instruction describing a target surface or objectrelative region and the intended projection scale. Examples include “project near the chair,” “move the light to the left of the table,” and “place a larger light patch on the wall above the cabinet.”

For the 150 tasks, we collect 10 source trajectories per task and render each trajectory with 5 appearance settings: the original appearance and four variants that randomize lighting intensity, lighting color temperature, wall materials, and floor materials. This procedure yields 50 rendered episodes per task and 7,500 rendered episodes in total while preserving the underlying geometry and source action sequence. Each episode includes the language instruction, timestamp-synchronized external and onboard images, robot control states, and actions; geometric target masks, projection centroids, and projection scales are not stored. Table 3 summarizes the corpus. All 10 scenes contribute training data for both main experiments and ablations, and configurations using the corpus include all 150 tasks. The source trajectory holdout is used only for offline validation, not as a benchmark for unseen scenes or instructions, as described in Suppl. Material. Instead, the real-world evaluation protocol assesses deployment in unseen environments with new instructions that do not appear in the matched simulation corpus.

## 5.3 Real-world evaluation protocol

The real-world evaluation protocol measures the final physical placement of a uniform white rectangular projection rather than only the robot pose. A correct chassis pose is insufficient if the mirror or projection scale leaves the beam outside the instructed region. Because the corpus does not store geometric target masks or calibrated target centroids and scales, we do not apply an automatic geometric success metric. Instead, a human operator judges whether the final projection satisfies the spatial instruction. A rollout is counted as successful when the operator confirms the placement and either the decoded episode termination output triggers termination or the projection remains stable on the instructed target for at least 3 s. The resulting annotation is used for real-world metrics (Sec. 6).

## 6 EXPERIMENTS

The experiments evaluate the three contributions in Sec. 1: three pretrained model families test interface generality; simulation-only training followed by real-world deployment in unseen environments, together with a π -scratch control, assesses transfer and the value of pretraining; and camera and action component ablations examine the complementary roles of the external and onboard sensing, chassis motion, and mirror steering.

## 6.1 Experimental setup

Action models and pretraining. To test the shared interface in Sec. 4.2, we instantiate the optimization in Eq. (11) with three action model families spanning continuous action generation, autoregressive action tokenization, and world-action modeling. The comparison includes $\pi _ { 0 . 5 } .$ , a continuous generative VLA model [3]; π<sub>0</sub>-FAST, which autoregressively generates tokenized actions [52]; and Fast-WAM [68], which builds on video-generative representations. We also train a $\pi _ { 0 . 5 }$ -scratch control with the same architecture and LICOA training data but without pretrained initialization. This control isolates the contribution of the representations acquired during large-scale pretraining.

Common simulation training and real-world deployment. All methods use the same demonstrations, external and onboard RGB images, five-dimensional robot control state, six-dimensional action space, and 16-step action horizon defined by the interfaces in Eqs. (3) and (5). Each model uses the family-specific adapters in Eqs. (9), (10) and (13), namely $\Phi _ { m } ^ { \mathrm { i n } } , \Phi _ { m } ^ { \mathrm { t g t } }$ , and $\hat { \Phi } _ { m } ^ { \mathrm { o u t } }$ <sup>t</sup>, while retaining its native action representation and training objective. On the platform, the external and onboard RGB cameras are configured to acquire $6 4 0 \times 4 8 0$ images at 30 Hz; the host applies the timestamp pairing rule in Eq. (1) before each inference request. Each run uses 30,000 optimizer updates, batch size 32, and one NVIDIA H200 GPU. We retain each model’s native optimizer and learning rate schedule. The pretrained $\pi _ { 0 . 5 }$ and $\pi _ { 0 } \cdot$ -FAST models are fully fine-tuned without low-rank adaptation (LoRA) or other parameter-efficient fine-tuning (PEFT) modules; Fast-WAM jointly trains its video and action experts. The scratch uses the same full $\pi _ { 0 . 5 }$ recipe without loading a pretrained checkpoint. During real-world evaluation, the server returns a 16-step action chunk at each replanning request. The con troller dispatches its first eight per-step actions at a nominal 15 Hz before replanning from fresh observations. Model-specific optimizer and scheduler settings are provided in the Suppl. Material.

Real-world evaluation task set. Following the success criterion defined in Sec. 5.3, we evaluate each action model on 12 tasks in previously unseen real-world environments. The tasks contain new spatial instructions absent from the matched simulation corpus, together with different target objects, initial robot configurations, and required optical actions. Each model is evaluated for 10 rollouts per task, yielding 120 rollouts per model and 480 rollouts in total. We report success rate (SR), success weighted by path length (SPL), and mean completion time (MT). For rollout $i ,$ let $S _ { i } \in \{ 0 , 1 \}$ be its success indicator, $L _ { i }$ the straight-line displacement between its initial and final chassis positions, $\bar { P } _ { i }$ its executed chassis path length, and T<sub>i</sub> its wall-clock execution time. We first define path efficiency $e _ { i }$ with $\varepsilon = 1 0 ^ { - 4 }$ m to cover the case with no chassis displacement:

Table 4: Real-world evaluation results for three pretrained action model families connected through the shared perception and optical action interface and a $\pi _ { 0 . 5 } \cdot$ scratch control. SR is averaged over all 12 tasks (10 rollouts per task; 120 rollouts per action model). SPL is averaged over the same 120 rollouts, while MT is averaged over successful rollouts only. The best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Success</td><td>SR (%) ↑</td><td>SPL (%) ↑</td><td>MT (s) ↓</td></tr><tr><td>π0.5-scratch</td><td>10/120</td><td>8.33</td><td>5.15</td><td>47.65</td></tr><tr><td>π0.5 [3]</td><td>69/120</td><td>57.50</td><td>40.64</td><td>43.44</td></tr><tr><td>π₀-FAST [52]</td><td>21/120</td><td>17.50</td><td>15.81</td><td>139.38</td></tr><tr><td>Fast-WAM [68]</td><td>35/120</td><td>29.17</td><td>21.76</td><td>64.72</td></tr></table>

$$
e _ { i } = \left\{ \begin{array} { l l } { 1 , } & { L _ { i } \leq \varepsilon \mathrm { ~ a n d ~ } P _ { i } \leq \varepsilon , } \\ { L _ { i } / \mathrm { m a x } \{ L _ { i } , P _ { i } \} , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{16}
$$

Then, for N rollouts, we define

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } , \mathrm { S P L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } e _ { i } , \mathrm { M T } = \frac { \sum _ { i = 1 } ^ { N } S _ { i } T _ { i } } { \sum _ { i = 1 } ^ { N } S _ { i } } .
$$

SPL penalizes both failure and inefficient motion of the differential-drive chassis, whereas MT measures execution speed only among successful trials. For real-world rollouts, $L _ { i }$ and $P _ { i }$ are computed by midpoint integration of the linear and angular velocities measured by the Kobuki chassis. Since an optical task can admit multiple valid terminal chassis poses, $L _ { i }$ is measured to the achieved successful endpoint. SR and SPL are computed over all 120 rollouts per action model, while MT is computed over the successful subset.

## 6.2 Interface generality and real-world deployment

Aggregate comparisons and per-task results are shown in Tab. 4 and Fig. 4, respectively. All three pretrained action model families achieve nonzero success after being connected to LICOA through the same physical interface and trained only in simulation. This result demonstrates that the interface can support continuous generation, autoregressive action tokenization, and world-action modeling, although their real-world performance differs substantially. Adapted π<sub>0.5</sub> achieves the highest overall SR of 57.50% (69/120), outperforming Fast-WAM, $\pi _ { 0 } { \mathrm { - F A S T } } ,$ and π<sub>0.5</sub>-scratch by 28.33, 40.00, and 49.17 percentage points, respectively. It is also the only action model to succeed at least once on every task. In comparison, Fast-WAM, $\pi _ { 0 } { \mathrm { - F A S T } } ,$ and $\pi _ { 0 . 5 ^ { - } \mathrm { s c r a t c l } }$ h obtain nonzero success on 7, 5, and 7 of the 12 tasks, respectively. These results show that the advantage of $\pi _ { 0 . 5 }$ is not restricted to a small subset of targets. The same model achieves the best SPL (40.64%) and the lowest MT over successful trials (43.44 s). Fast-WAM ranks second in both SR and SPL, but its SPL of 21.76% remains 18.88 percentage points below that of $\pi _ { 0 . 5 }$ . Although $\pi _ { 0 } { \mathrm { - F A S T } }$ produces relatively direct chassis trajectories when it succeeds, its low success frequency reduces its overall SPL to 15.81%, and its successful rollouts require 139.38 s on average. The low MT of $\pi _ { 0 . 5 }$ -scratch should be interpreted together with its 8.33% SR: because MT is conditioned on success, it reflects only the few tasks completed by this baseline and does not indicate robust execution.

![](images/314ba8e8af5204c005f95001e52a9bc48dc80235de5dd3aa7607b48527b99869.jpg)  
Figure 4: Per-task evidence for real-world deployment through the shared perception and optical action interface. Each action model is trained only on the matched simulation corpus and evaluated for 10 rollouts on each of 12 previously unseen real-world tasks. The instructions and target scenes are shown in reading order, and the bars report success rate. All tasks contribute equally to the overall SR in Tab. 4.

Evidence from pretraining. The gap between $\pi _ { 0 . 5 }$ and its scratch counterpart is particularly large: pretrained initialization improves average SR from 8.33% to 57.50% and SPL from 5.15% to 40.64%. Since the two variants share the action model architecture, training data, and action interface, this comparison indicates that pretrained vision-language and action representations are important for adapting to the new optical action embodiment. Nevertheless, the remaining 42.50% failure rate also shows that reliable coordination of target grounding, the differential-drive chassis, and the 2DoF steerable mirror remains challenging.

## 6.3 Evidence for sensing and action design

We use the best-performing π<sub>0.5</sub> in Tab. 4 to ablate the complementary camera views and coupled chassis-mirror control introduced in Sec. 3. Camera variants are retrained under the same fine-tuning recipe, with one view removed from both the dataset and model input mapping; action variants retain the full checkpoint and suppress the indicated commands only at deployment. All models train on the same 10 simulation scenes. Evaluation uses three novel real-world tasks with 10 rollouts per configuration and task, with instructions and initial configurations matched across configurations.

External and onboard views. The full action model receives timestamp-synchronized external and onboard images. We retrain two variants, each removing one view, to test the contribution of global scene context or robot-centric feedback, respectively. Retraining avoids the additional distribution shift that would arise from masking a view only at inference.

Coupled action components. We further compare the full coupled action space with two deployment-time restrictions. Mirror only replaces both predicted commands for the differential-drive chassis with zero while retaining the commands for the 2DoF steerable mirror and projection scale. Chassis only replaces the two predicted pan and tilt targets for the 2DoF steerable mirror with the canonical fixed pose while retaining the commands for the differential-drive chassis and projection scale. Thus, both variants use the same full action model checkpoint and differ only in the commands dispatched to the physical actuators. We report SR, SPL, and MT over successful trials using the same definitions as in Sec. 6.1.

Table 5: Camera and action component ablations for $\pi _ { 0 . 5 }$ on three novel real-world tasks (10 rollouts per task; 30 per configuration). Camera variants are retrained without the indicated view, whereas actuation variants disable the indicated component only at deployment. All variants retain the robot control state and projection-scale command. The images below show Task 13, “project white light onto the umbrella.”; Task 14, “project white light onto the brown bag.”; and Task 15, “project white light onto the white statue.”

<table><tr><td>Config.</td><td>Iº Ie</td><td></td><td>Chassis Mirror Success</td><td></td><td></td><td></td><td>SR↑ SPL ↑ MT↓</td></tr><tr><td>Full</td><td></td><td>√ √</td><td>√</td><td>√</td><td>18/30</td><td>60.00</td><td>37.09 12.20</td></tr><tr><td>W/o onb. cam.</td><td>×</td><td></td><td>√</td><td>√</td><td>9/30</td><td>30.00</td><td>24.61 29.07</td></tr><tr><td>W/o ext. cam.</td><td>V</td><td>×</td><td>√</td><td>√</td><td>8/30</td><td>26.67</td><td>13.47 6.57</td></tr><tr><td>Mirror only</td><td>√√</td><td></td><td>×</td><td>√</td><td>9/30</td><td></td><td>30.00 30.00 45.71</td></tr><tr><td>Chassis only</td><td>√ √</td><td></td><td>√</td><td>×</td><td>4/30</td><td>13.33</td><td>312.04 23.86</td></tr></table>

![](images/79f2b241332a0ed4c3ed7278a6c8567cd54c7f0b6a313d2f1cb08d9cef1b08cd.jpg)

As shown in Tab. 5, the full configuration achieves the highest SR and SPL, at 60.00% and 37.09%, respectively. Removing the external camera reduces SR to 26.67% and SPL to 13.47%, while removing the onboard camera reduces them to 30.00% and 24.61%. Thus, both the global scene context and robot-centric visual feedback make substantial contributions. Although the 6.57 s MT without the external camera is lower than the full configuration’s 12.20 s, it is conditioned on only eight successful rollouts and therefore does not indicate better overall performance. Removing the onboard camera instead increases MT over successful trials to 29.07 s. The action component ablation likewise supports coordinated control. Mirror only reaches 30.00% SR, whereas Chassis only reaches 13.33%; both remain well below the full configuration’s 60.00%. Their pertask SRs also differ: Mirror only obtains 0%, 40%, and 50%, while Chassis only obtains 30%, 0%, and 10% on the three tasks. This task dependence shows that neither actuator alone provides consistent reachability and placement across target configurations. The mirror-only SPL equals its SR because successful rollouts with a stationary chassis receive unit path efficiency under the zero-motion convention; its 45.71 s MT nevertheless remains substantially above the full configuration’s 12.20 s. The Suppl. Material further studies parameter adaptation and training data scale under a fixed offline validation protocol. It shows how the matched simulation corpus supports adaptation, which cannot substitute for the real-world evaluation results above.

![](images/cc8a7607e06b07ae6966647eb619dbaf8b8655a84a1870ab48c0cfc075d3aece.jpg)  
Figure 5: Proof-of-concept applications of LICOA. Left: RGB colored point clouds reconstructed from structured light sequences for severa surfaces and objects. Right, from top to bottom: desired projection content, compensated camera observations, and uncompensated camera observations on one textured surface. The compensation images are rectified to projector coordinates for visual comparison.

## 7 DISCUSSION

The central scope of LICOA is physical placement: it interprets a spatial instruction and controls where light arrives and how large its footprint becomes. The uniform white rectangular projection used in the real-world evaluation protocol isolates this capability from content. Within that scope, the experiments support the complete causal chain: the shared interface connects pretrained action models to the platform, both camera views contribute complementary context, and the chassis and mirror provide complementary motion scales.

## 7.1 Applicability to other settings

To explore learned placement beyond the projection appearance used for training, we use the same full $\pi _ { 0 . 5 }$ model from the preceding experiments. The model was fine-tuned only on the matched simulation corpus with a uniform white rectangular projection. At deployment, we replace this content with a circular light, green light, or an image, without additional fine-tuning. As shown in Fig. 6, the model places each content on the wooden surface. These qualitative rollouts provide initial evidence that the learned placement behavior can accommodate projection appearances beyond the finetuning corpus, suggesting potential for simple semantic projection and motivating the downstream optical applications explored next.

## 7.2 Additional applications

Building on this potential, we test two downstream operations with the placed projector-camera pair. These proof-of-concept trials follow a place-then-operate workflow: placement first establishes an operating pose, after which task-specific projection and processing support structured light reconstruction or projector compensation.

Structured light 3D reconstruction. After LICOA places light on a target, the chassis and mirror remain fixed while the projector displays complementary Gray-code and four-step phase-shifting patterns [32,60,66]. The onboard Kinect records each pattern sequence, and the Jetson decodes projector-camera correspondences and triangulates the colored point clouds shown in Fig. 5. These examples demonstrate local geometry acquisition after repositioning.

Projector compensation. We also demonstrate compensation on one textured surface after the platform is fixed. A complementary Gray-code scan provides the geometric warp as in [13], and the photometric component of a pretrained CompenNeSt [31] is finetuned on eight projected images. The scan, fine-tuning, compensated image generation, and physical reprojection run on the Jetson. This result is a single-pose proof of concept and remains separate from the real-world evaluation protocol.

![](images/2974175070bf5e62426f2eed0f303c62c5620f14f941f47e1590e83afa629ce5.jpg)  
Figure 6: Qualitative rollouts of the full $\pi _ { 0 . 5 }$ configuration with varied projection content. Each row shows selected frames from one rollout. From top to bottom, the instructions are “project circle light onto the wooden surface,” “project green light onto the wooden surface,” and “project image onto the wooden surface.”

## 7.3 Limitations and future work

The current formulation intentionally limits the scope of the problem. First, using a uniform white rectangular projection is sufficient for evaluating spatial placement but does not address the generation, layout, or readability of rich projected content. Second, the external camera simplifies supervision and evaluation, which makes the current setup more instrumented than a fully self-contained consumer system. Third, the projection result remains sensitive to surface material, ambient light, camera exposure, and actuator repeatability. Fourth, the proposed action space is continuous and hardware-specific, so generalization across different projectors, mirrors, mobile chassis, and room scales will require additional calibration or adaptation data. Fifth, the current procedure used to construct termination supervision does not produce fully reliable termination labels for every demonstration. Consequently, the learned termination output is not always dependable. More reliable termination annotation and explicit supervision of task completion over time remain important to be explored.

Future work should extend on both system and learning. On the system side, richer projected content, onboard-only state estimation, a higher-speed 2DoF steerable mirror, and more accurate projection scale control can make the platform closer to practical SAR applications. On the learning side, larger VLA or WAM models can be fine-tuned with mixed real and simulated data, and selfsupervised exploration can be used to learn how commands for the differential-drive chassis and 2DoF steerable mirror move the projection footprint. A longer-term goal is to develop spatial computing agents that can decide not only what information to show, but also where and how light should be physically routed in the environment.

## 8 CONCLUSION

This work introduces LICOA, a mobile SAR system that treats projected light as an instruction-conditioned optical action. By coupling room-scale chassis motion with rapid 2DoF mirror steering and projection-scale control, LICOA decides how to move and where to place light without fixed projector poses, predefined targets, or scripted trajectories. The shared perception and optical action interface connects diverse pretrained action models to the same platform. Trained only on a matched simulation corpus of 150 spatial placement tasks and 7,500 episodes, adapted $\pi _ { 0 . 5 }$ achieves 57.50% SR and 40.64% SPL across 480 real-world rollouts, versus 8.33% and 5.15% from scratch. Ablations further validate the complementary sensing and actuation design. Together, these results turn projection from a configured display into an embodied capability, opening a path toward autonomous and adaptive SAR.

## REFERENCES

[1] A. H. Bermano, M. Billeter, D. Iwai, and A. Grundhofer. Makeup¨ Lamps: Live augmentation of human faces via projection. Computer Graphics Forum, 36(2):311–323, 2017. 1, 2

[2] O. Bimber and R. Raskar. Spatial Augmented Reality: Merging Real and Virtual Worlds. A K Peters/CRC Press, 2005. 1, 2

[3] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, M. Y. Galliker, et al. π<sub>0.5</sub>: a visionlanguage-action model with open-world generalization. In CoRL, vol. 305 of Proceedings ofMachine Learning Research, pp. 17–40. PMLR, 2025. 2, 3, 7

[4] K. Black, N. Brown, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π : A vision-languageaction flow model for general robot control. In Robotics: Science and Systems, 2025. 2, 3

[5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. RT-2: Visionlanguage-action models transfer web knowledge to robotic control. In CoRL, vol. 229 of Proceedings ofMachine Learning Research. PMLR, 2023. 2, 3

[6] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. RT-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems, 2023. 2, 3

[7] M. A. Cabrera, I. Usachev, J. Heredia, J. Tirado, A. Fedoseev, and D. Tsetserukou. MaskBot: Real-time robotic projection mapping with head motion tracking. In SIGGRAPH Asia 2020 Emerging Technolo gies, SA ’20. ACM, 2020. 2, 3

[8] Z. Cao, Y. Deng, H. Ling, and B. Huang. ProCap: Projection-aware captioning for spatial augmented reality. In IEEE VR, pp. 22–32, 2026. 2

[9] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, H. Zhang, and M. Zhu. GR-2: A generative videolanguage-action model with web-scale knowledge for robot manipulation, 2024. 3

[10] X. Chen, X. Yang, S. Xiao, and M. Li. Color mixing property of a projector-camera system. In ACM/IEEE International Workshop on Projector Camera Systems, PROCAMS ’08. ACM, 2008. 2

[11] C. Cruz-Neira, D. J. Sandin, T. A. DeFanti, R. V. Kenyon, and J. C. Hart. The CAVE: audio visual experience automatic virtual environment. Communications ofthe ACM, 35(6):64–72, 1992. 2, 3

[12] Q. Deng, J. Li, H. Ling, and B. Huang. GS-ProCams: Gaussian splatting-based projector-camera systems. IEEE TVCG, 31(11):9845– 9855, 2025. 1, 2, 3

[13] Y. Deng, H. Ling, and B. Huang. LAPIG: Language guided projector image generation with surface adaptation and stylization. IEEE TVCG, 31(5):2515–2524, 2025. 2, 3, 9

[14] X. Dong, H. Ling, and B. Huang. Adaptive color structured light for calibration and shape reconstruction. In IEEE ISMAR, pp. 1240–1249, 2023. 1, 2

[15] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation. In NeurIPS, vol. 36, pp. 9156–9172. Curran Associates, Inc., 2023. 3

[16] Y. Du, S. Yang, P. Florence, F. Xia, A. Wahid, b. ichter, P. Sermanet, T. Yu, P. Abbeel, J. B. Tenenbaum, L. Kaelbling, A. Zeng, and J. Tompson. Video language planning. In ICLR, vol. 2024, pp. 31138–31155, 2024. 3

[17] Y. Erel, D. Iwai, and A. H. Bermano. Neural projection mapping using reflectance fields. IEEE TVCG, 29(11):4339–4349, 2023. 1, 2, 3

[18] Y. Erel, O. Kozlovsky-Mordenfeld, D. Iwai, K. Sato, and A. H. Bermano. Casper DPM: Cascaded perceptual dynamic projection mapping onto hands. In SIGGRAPH Asia. ACM, 2024. 1, 2

[19] E. Fernandez-Rodicio, V. Gonz´ alez-Pacheco, J. C. Castillo,´ A. Castro-<sup>´</sup> Gonzalez, M. Malfaz, and M. A. Salichs. Projection surfaces detection´ and image correction for mobile robots in HRI. Journal of Sensors, 2017:1–16, 2017. 1, 2, 3

[20] S. Gao, M. Pagnucco, T. Bednarz, and Y. Song. NVIDIA Isaac Sim: Enabling scalable, gpu-accelerated simulation for robotics, 2026. 2, 6

[21] D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems, 2024. 2, 3

[22] A. Grundhofer and D. Iwai. Robust, error-tolerant photometric projec-¨ tor compensation. IEEE TIP, 24(12):5086–5099, 2015. 2

[23] A. Grundhofer and D. Iwai. Recent advances in projection mapping ¨ algorithms, hardware and applications. Computer Graphics Forum, 37(2):653–675, 2018. 1, 2

[24] Z. Han, J. Parrillo, A. Wilkinson, H. A. Yanco, and T. Williams. Projecting robot navigation paths: Hardware and software for projected AR. In ACM/IEEE International Conference on Human-Robot Interaction, HRI ’22, pp. 623–628. IEEE Press, 2022. 1, 2, 3

[25] K. Hiratani, D. Iwai, Y. Kageyama, P. Punpongsanon, T. Hiraki, and K. Sato. Shadowless projection mapping using retrotransmissive optics. IEEE TVCG, 29(5):2280–2290, 2023. 2

[26] Y. Hu, Y. Guo, P. Wang, X. Chen, Y.-J. Wang, J. Zhang, K. Sreenath, C. Lu, and J. Chen. Video prediction policy: A generalist robot policy with predictive visual representations. In ICML, vol. 267 of Proceedings ofMachine Learning Research, pp. 24328–24346. PMLR, 2025. 3

[27] Y. Hu, P. Huang, M. Sivapurapu, and J. Zhang. ELEGNT: Expressive and functional movement design for non-anthropomorphic robot, 2025. 2, 3

[28] B. Huang and H. Ling. CompenNet++: End-to-end full projector compensation. In ICCV, 2019. 2

[29] B. Huang and H. Ling. End-to-end projector photometric compensation. In CVPR, 2019. 2

[30] B. Huang and H. Ling. DeProCams: Simultaneous relighting, compen sation and shape reconstruction for projector-camera systems. IEEE TVCG, 27(5):2725–2735, 2021. 1, 2, 3

[31] B. Huang, T. Sun, and H. Ling. End-to-end full projector compensation. IEEE TPAMI, 2022. 1, 2, 9

[32] B. Huang, Y. Tang, S. Ozdemir, and H. Ling. A fast and flexible projector-camera calibration system. IEEE Transactions on Automation Science and Engineering, 18(3):1049–1063, 2021. 1, 2, 9

[33] T.-H. Huang, T.-C. Wang, and H. H. Chen. Radiometric compensation of images projected on non-white surfaces by exploiting chromatic adaptation and perceptual anchoring. IEEE TIP, 26(1):147–159, 2017. 2

[34] S. Inokuchi. Range imaging system for 3-d object recognition. ICPR, 1984, 1984. 2

[35] D. Iwai. Projection mapping technologies: A review of current trends and future directions. Proceedings of the Japan Academy, Series B, 100(3):234–251, 2024. 1, 2

[36] B. Jones, R. Sodhi, M. Murdock, R. Mehra, H. Benko, A. Wilson, E. Ofek, B. MacIntyre, N. Raghuvanshi, and L. Shapira. RoomAlive: magical experiences enabled by scalable, adaptive projector-camera units. In ACM UIST, UIST ’14, pp. 637–644, 2014. 2, 3

[37] Y. Kageyama, M. Isogawa, D. Iwai, and K. Sato. ProDebNet: projector deblurring using a convolutional neural network. Optics Express, 28(14):20391–20403, 2020. 2

[38] Y. Kageyama, D. Iwai, and K. Sato. Efficient distortion-free neural projector deblurring in dynamic projection mapping. IEEE TVCG, 30(12):7544–7557, 2024. 2

[39] T. Kaminokado, D. Iwai, and K. Sato. Augmented environment mapping for appearance editing of glossy surfaces. In IEEE ISMAR, pp. 55–65, 2019. 2

[40] H. Kawasaki, R. Furukawa, R. Sagawa, and Y. Yagi. Dynamic scene shape reconstruction using a single structured light pattern. In CVPR, pp. 1–8, 2008. 2

[41] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, et al. OpenVLA: An open-source vision-language-action model. In CoRL, vol. 270 of Proceedings ofMachine Learning Research, pp. 2679–2713. PMLR, 2025. 2, 3

[42] R. Kjeldsen, C. Pinhanez, G. Pingali, J. Hartman, T. Levas, and M. Pod laseck. Interacting with steerable projected displays. In IEEE In ternational Conference on Automatic Face Gesture Recognition, pp. 402–407, 2002. 1, 2, 3

[43] H. Kusuyama, Y. Kageyama, D. Iwai, and K. Sato. A multi-aperture coaxial projector balancing shadow suppression and deblurring. IEEE TVCG, 30(11):7031–7041, 2024. 2

[44] J.-E. Lee, J. Park, G.-S. Kim, J.-H. Lee, and M.-H. Kim. Interactive multi-resolution display using a projector mounted mobile robot in intelligent space. International Journal ofAdvanced Robotic Systems, 9(5):196, 2012. 3

[45] J. Li, Q. Deng, H. Ling, and B. Huang. DPCS: Path tracing-based differentiable projector-camera systems. IEEE TVCG, 31(5):3666– 3676, 2025. 1, 2, 3

[46] S. Li, Y. Gao, D. Sadigh, and S. Song. Unified video action model. In Robotics: Science and Systems, 2025. 3

[47] N. Linder and P. Maes. LuminAR: portable robotic augmented reality interface design and prototype. In Adjunct Proceedings of the 23nd Annual ACM Symposium on User Interface Software and Technology, UIST ’10, pp. 395–396. ACM, 2010. 1, 2, 3

[48] K. Luo, G. Yang, W. Xian, H. Haraldsson, B. Hariharan, and S. Belongie. Stay Positive: Non-negative image synthesis for augmented reality. In CVPR, pp. 10050–10060, 2021. 2

[49] E. Nazarova, O. Sautenkov, M. Altamirano Cabrera, J. Tirado, V. Serpiva, V. Rakhmatulin, and D. Tsetserukou. CobotAR: Interaction with robots using omnidirectionally projected image and DNN-based ges ture recognition. In IEEE International Conference on Systems, Man, and Cybernetics, pp. 2590–2595, 2021. 1, 2, 3

[50] T. Nomoto, W. Li, H.-L. Peng, and Y. Watanabe. Dynamic multi projection mapping based on parallel intensity control. IEEE TVCG, 28(5):2125–2134, 2022. 2

[51] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open X-Embodiment: Robotic learning datasets and rt-x models : Open x-embodiment collaboration<sup>0</sup>. In IEEE ICRA, pp. 6892–6903, 2024. 2, 3

[52] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine. FAST: Efficient action tokenization for vision-language-action models. In Robotics: Science and Systems, 2025. 2, 3, 7

[53] C. Pinhanez. The everywhere displays projector: A device to create ubiquitous graphical interfaces. In Ubicomp 2001: Ubiquitous Com puting, pp. 315–331. Springer Berlin Heidelberg, Berlin, Heidelberg, 2001. 1, 2, 3

[54] P. Punpongsanon, D. Iwai, and K. Sato. SoftAR: Visually manipulating haptic softness perception in spatial augmented reality. IEEE TVCG, 21(11):1279–1288, 2015. 1, 2

[55] R. Raskar, J. van Baar, P. Beardsley, T. Willwacher, S. Rao, and C. Forlines. iLamps: geometrically aware and self-configuring projectors. ACM TOG, 22(3):809–818, 2003. 2

[56] R. Raskar, G. Welch, K.-L. Low, and D. Bandyopadhyay. Shader Lamps: Animating real objects with image-based illumination. In Rendering Techniques 2001, pp. 89–102. Springer Vienna, 2001. 2, 3

[57] K. Richter, S. B. Fischedick, and H.-M. Gross. Projection surface

detection and pose selection for autonomously displaying multimedia on walls using mobile robots. Frontiers in Robotics and AI, 13, 2026. 1, 3

[58] C. Siegl, M. Colaianni, L. Thies, J. Thies, M. Zollhofer, S. Izadi,¨ M. Stamminger, and F. Bauer. Real-time pixel luminance optimization for dynamic multi-projection mapping. ACM TOG, 34(6), 2015. 2

[59] SpatialVerse Research Team, Manycore Tech Inc. InteriorAgent: Interactive USD interior scenes for Isaac Sim-based simulation, 2025. 2, 6

[60] V. Srinivasan, H. C. Liu, and M. Halioua. Automated phase-measuring profilometry of 3-d diffuse objects. Applied Optics, 23(18):3105–3108, Sep 1984. 2, 9

[61] R. Stricker, S. Muller, and H.-M. Gross. Universal usage of a video pro-¨ jector on a mobile guide robot. In Intelligent Robotics and Applications, pp. 25–36, 2015. 3

[62] W. Sun, X. Yang, S. Xiao, and W. Hu. Robust checkerboard recognition for efficient nonplanar geometry registration in projector-camera systems. In ACM/IEEE International Workshop on Projector Camera Systems, PROCAMS ’08. ACM, 2008. 1, 2

[63] K. Tatsumoto, S. Iwaki, and T. Ikeda. Tracking projection method for 3d space by a mobile robot with camera and projector based on a structured-environment approach. Artificial Life and Robotics, 22(1):90–101, 2017. 1, 3

[64] A. Wilson, H. Benko, S. Izadi, and O. Hilliges. Steerable augmented reality with the beamatron. In ACM UIST, UIST ’12, pp. 413–422, 2012. 1, 2, 3

[65] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In ICLR, vol. 2024, pp. 10641–10662, 2024. 3

[66] S. Yamazaki, M. Mochimaru, and T. Kanade. Simultaneous selfcalibration of a projector and a camera using structured light. In CVPRW, pp. 60–67, 2011. 9

[67] M. Yasui, R. Iwataki, M. Ishikawa, and Y. Watanabe. Projection mapping with a brightly lit surrounding using a mixed light field approach. IEEE TVCG, 30(5):2217–2227, 2024. 2

[68] T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-WAM: Do world action models need test-time future imagination?, 2026. 2, 3, 7

[69] H. Zhou, Y. Zhang, Z. Zhang, W. Zhang, C. Wang, and P. An. ProjecTA: A semi-humanoid robotic teaching assistant with in-situ projection for guided tours. In ACM CHI, pp. 1–24, 2026. 1, 3

[70] S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan. RoboDreamer: Learning compositional world models for robot imagination. In ICML, vol. 235 of Proceedings ofMachine Learning Research, pp. 61885– 61896. PMLR, 2024. 3

[71] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In Robotics: Science and Systems, 2025. 3

# LICOA: Learning Instruction-Conditioned Optical Actions for Mobile Spatial Augmented Reality

– Supplementary Materials –

## 1 PHYSICAL PLATFORM AND EMBEDDED CONTROL

The physical implementation follows the design with two motion scales in the main paper. The differential-drive chassis carries the projector across the environment, while the steerable mirror with two degrees of freedom (2DoF) redirects the beam locally and projection scale changes its footprint. The embedded system must therefore deliver chassis, mirror, and scale commands through separate hardware paths while preserving the shared action components exposed to the action model.

## 1.1 System interfaces and power

The NVIDIA Jetson Nano is the robot-mounted embedded computer. It acquires the onboard RGB stream from the Microsoft Kinect v1 over USB, communicates with the Kobuki differential-drive chassis over a dedicated USB serial connection, and exchanges pan and tilt targets for the 2DoF steerable mirror and telemetry with the STM32 through a separate universal asynchronous receiver-transmitter (UART) interface over USB. The Jetson sends the robot stream, comprising the latest onboard RGB image, its timestamp, and the robot control state, to the host PC. The USB-connected Logitech C270 supplies the external camera stream. The host PC synchronizes the two streams, sends the resulting observation as an inference request to the server, receives a 16-step action chunk, and relays each executed six-dimensional per-step action to the Jetson. The Jetson then dispatches commands to the Kobuki, STM32, and custom laser projector. Table S1 summarizes the power distribution.

## 1.2 Custom laser projector

The custom laser projector receives HDMI output from the Jetson. A Lontium LT8619B bridge converts HDMI to 24-bit RGB888 for the Sony CXN0102-based projection electronics, which coordinate Sony image processing and laser control with the RGB laser sources and a scanner based on microelectromechanical systems (MEMS) to produce the projected image. Projection scale is changed by rendering the corresponding pattern over the same HDMI path.

## 1.3 2DoF steerable mirror wiring

The two physical USB connections between the Jetson and the STM32 assembly serve different purposes. The CH340 USB-to-UART adapter carries bidirectional commands and telemetry, whereas the direct USB connection supplies STM32 logic power. Motor power is independent of both links and is delivered from the 12 V rail to the two SimpleFOC Mini driver boards.

## 1.4 Jetson Nano command bridge

The NVIDIA Jetson Nano is the robot-mounted bridge between the remote host PC control client and the action components. Its bridge service receives each six-dimensional per-step action as a ZeroMQ (ZMQ) message containing the linear and angular velocities $( v , \omega )$ for the differential-drive chassis, pan and tilt targets $\left( q _ { \mathrm { p } } , q _ { \mathrm { t } } \right)$ ) for the 2DoF steerable mirror, projection scale $\rho ,$ and termination indicator b. It clamps the two mirror targets to the platform limits, $\dot { q _ { \mathrm { p } } } \dot { \in } [ \dot { - } \pi / 2 , \pi / 2 ]$ and $q _ { \mathrm { t } } \in [ 0 , \pi / 2 ]$ , dispatches chassis, mirror, and projection-scale commands through their respective interfaces, and zeros the chassis velocities when $b > 0 . 5$ . The pan-tilt serial controller converts the two targets from semantic joint coordinates in radians to the absolute angles expected by the firmware. Let $\phi _ { \mathrm { X } } ^ { 0 }$ and $\phi _ { \mathrm { Q } } ^ { 0 }$ be the configurable zero offsets; denote the resulting firmware angles by ϕ<sub>X</sub> and ϕ<sub>Q</sub>; and let wra $\mathrm { p } _ { 3 6 0 }$ map an angle in degrees to [0, 360). The pan-axis mapping is

$$
\phi _ { \mathrm { X } } = \mathrm { w r a p } _ { 3 6 0 } \bigg ( \phi _ { \mathrm { X } } ^ { 0 } - \frac { 1 8 0 } { \pi } q _ { \mathrm { p } } \bigg ) .\tag{S1}
$$

The tilt-axis mapping is

$$
\phi _ { \mathrm { Q } } = \mathrm { w r a p } _ { 3 6 0 } \biggl ( \phi _ { \mathrm { Q } } ^ { 0 } + \frac { 1 8 0 } { \pi } q _ { \mathrm { t } } \biggr ) .\tag{S2}
$$

The opposite sign in Eq. (S1), compared with Eq. (S2), accounts for the installed motor orientation.

The Jetson encodes pan (X) and tilt (Q) targets as newline-terminated ASCII commands, for example X45.00 and Q30.00. A USB-to-UART adapter connects to STM32 USART1 (PA9/PA10 and common ground) at 115,200 baud, 8N1; a second USB connection from the Jetson supplies STM32 logic power. Because the firmware uses a single line receive buffer, a background sender alternates the Q and X lines with a 10 ms offset; each axis is consequently refreshed at 50 Hz. Newly received per-step commands overwrite the previous targets, so command latency does not build an obsolete FIFO queue. The serial interface reconnects lazily after disconnection, and a missing pan-tilt link does not block control of the differential-drive chassis. Separately, a 0.3 s command watchdog zeros the velocity of the differential-drive chassis; it does not reset the pan-tilt targets. The deployed path does not use a controller area network (CAN): communication is UART-over-USB from Jetson to STM32, software inter-integrated circuit (I2C) communication from STM32 to the encoders, and pulse-width modulation (PWM) plus enable signals from STM32 to the motor drivers.

## 1.5 STM32 motor control

The USB link carries targets and diagnostic telemetry, not real-time servo waveforms. An STM32F103C8T6 parses the X/Q commands and closes the two pan-tilt motor loops locally at approximately 260 Hz. Each 2804 brushless direct-current (BLDC) axis uses a 12-bit AS5600 magnetic encoder connected by software I2C (PB6/PB7 for the vertical axis and PB10/PB11 for the horizontal axis). Startup calibration estimates sensor direction and electrical zero; during operation, a five-sample median filter suppresses angular outliers.

For streamed commands, the firmware interpolates each internal target at 0.30 rad/s using the measured control loop interval, so its motion profile remains stable when the loop rate varies. Both axes then use continuous position/velocity feedback to compute the q-axis voltage command. The deployed vertical-axis path does not use the legacy staged open-loop fast approach mode. The requested voltage is converted to space-vector pulse-width modulation (SVPWM) at approximately 25 kHz and sent as three PWM phases plus an enable signal to each SimpleFOC Mini driver. Both drivers use a 12 V supply, with firmware voltage limits of 6.5 V for the vertical motor and 6.0 V for the horizontal motor. Encoder angle, estimated velocity, loop rate, and error flags are returned over USB for diagnostics. Thus, network and Jetson timing determine when a new setpoint arrives, whereas the STM32 maintains the high-rate electromechanical feedback loop shown in Fig. S1.

Table S1: Power distribution of the physical LICOA platform. Capacities and output ratings are the nameplate specifications of the deployed supplies.
<table><tr><td>Source</td><td>Output</td><td>Load</td><td>Purpose</td></tr><tr><td>Battery pack 1 (6,000 mAh)</td><td>Chassis supply</td><td>Kobuki mobile chassis</td><td>Differential-drive locomotion</td></tr><tr><td>Battery pack 1 (6,000 mAh)</td><td>12V/1.5 A</td><td>Microsoft Kinect v1</td><td>Onboard RGB camera power; image data use USB</td></tr><tr><td>Battery pack 2 (30,000 mAh)</td><td>5V/4A</td><td>NVIDIA Jetson Nano</td><td>Embedded computing</td></tr><tr><td>Battery pack 2 (30,000 mAh)</td><td>12V/1 A</td><td>Two SimpleFOC Mini drivers</td><td>Actuation of the 2DoF steerable mirror</td></tr><tr><td>Battery pack 2 (30,000 mAh)</td><td>5V/2A</td><td>Custom laser projector</td><td>Projection</td></tr><tr><td>NVIDIA Jetson Nano</td><td>USB bus power</td><td>STM32F103C8T6</td><td>Controller logic power, separate from USB-to-UART data</td></tr><tr><td>Host PC</td><td>USB bus power</td><td>Logitech C270</td><td>External RGB camera data and power</td></tr></table>

![](images/fc504e660df9dc1eeec1da779534dee0951cb9534115046c8751367ff5fbba44.jpg)  
Figure S1: Electrical and control connections for the 2DoF steerable mirror and its pan-tilt servo assembly. The NVIDIA Jetson Nano mounted on the differential-drive chassis provides the USB-to-UART command and telemetry path, while a separate Jetson USB connection (not shown) supplies logic power to the STM32. The STM32 closes both motor loops locally using AS5600 encoder feedback and drives the two brushless direct-current (BLDC) driver boards with three-phase pulse-width modulation (PWM). The 12 V/1 A rail from battery pack 2 supplies the two SimpleFOC Mini drivers and pan-tilt motors.

## 2 MATCHED SIMULATION CORPUS CONSTRUCTION

The main paper identifies the absence of training data and common tasks for instruction-conditioned optical action as a central challenge. We address it by reproducing the physical LICOA embodiment, camera roles, robot control state, and action components in simulation, then collecting demonstrations under the same interface used for real-world deployment.

We construct the simulated environment using NVIDIA Isaac Sim 6.0.1. Indoor geometry and materials are loaded from the first 10 InteriorAgent universal scene description (USD) scenes at a scale of one meter per stage unit. Each InteriorAgent scene is a complete, large indoor environment. We define five fixed external camera viewpoints in each scene and construct three spatial placement tasks for each external camera viewpoint, yielding 150 tasks in total. For each task, we add the LICOA USD articulation, a PhysX scene, and the task-specific initial robot pose. Environment rigid bodies are frozen during collection, and static triangle-mesh collision proxies preserve the scene geometry used for navigation and optical interaction. The onboard RGB camera remains attached to the robot, whereas the external RGB camera remains at the corresponding fixed pose in the world frame. The principal recording settings are summarized in Tab. S2.

We collect source trajectories by teleoperating the differential-drive chassis, the steerable mirror with two degrees of freedom (2DoF), and the projection scale. Only demonstrations that the operator accepts as successfully satisfying the instruction are committed as source trajectories, while failed and timed-out attempts are discarded. Isaac Sim uses fixed time stepping at 15 Hz, so every saved observation is aligned with one complete physics step. We first save the full simulator state together with the synchronized trace of robot control states and actions and then replay it with physics paused to render the external and onboard RGB images at the recorded poses. This separates interactive demonstration collection from offline image generation while preserving the exact source action sequence. At each recorded simulator state, four rendering subframes are used: two interpolate motion from the preceding state and two render the final fixed pose. Only the image at the recorded pose is stored.

Table S2: Isaac Sim configuration used to match the LICOA physical embodiment and construct its matched simulation corpus.
<table><tr><td>Setting</td><td>Configuration</td></tr><tr><td>Simulator</td><td>NVIDIA Isaac Sim 6.0.1; PhysX fixed-step simulation</td></tr><tr><td>Scene assets</td><td>First 10 InteriorAgent USD indoor scenes; each a complete, large indoor environment; 1 stage unit = 1 m; frozen environment rigid bodies with static triangle-mesh collision proxies</td></tr><tr><td>Differential-drive chassis</td><td>Differential drive; wheel radius 0.035 m; wheelbase 0.230 m;  $| v | \leq 0 . 7 \mathrm { m / s }$  and  $| \omega | \leq 3 . 1 4 \mathrm { r a d / s }$ </td></tr><tr><td>2DoF steerable mirror and projection scale</td><td>Pan range  $[ - \pi / 2 , \pi / 2 ] ;$  tilt range  $[ 0 , \pi / 2 ] ;$  projection scale  $[ 0 , 1 0 0 ] \%$ </td></tr><tr><td>Virtual projector External and onboard RGB images</td><td>Textured square RectLight; uniform white  $2 5 6 \times 2 5 6$  source image; intensity 80; throw-ratio parameter 6.0 Onboard RGB camera and five fixed external camera viewpoints per scene; both streams rendered at 640 × 480</td></tr><tr><td>Sampling</td><td>Fixed 15 Hz application, timeline, and PhysX steps; synchronized RGB image, robot control state, and action</td></tr><tr><td>Offline rendering</td><td>records RTX Real-Time renderer; deep learning anti-aliasing  $( \mathrm { D L A A } ) ;$  four render subframes per recorded state; motion</td></tr><tr><td></td><td>blur disabled; 640 × 480 output</td></tr><tr><td>Appearance variants</td><td>One original rendering and four randomized renderings per source trajectory</td></tr></table>

Table S3: Training recipes used for the main real-world comparison. All runs use one NVIDIA H200 GPU, 30,000 optimizer updates, batch size 32, seed 0, BF16 computation, gradient checkpointing, and a 16-step action horizon. “Clip” denotes the global gradient norm threshold. lr and wd denote learning rate and weight decay, respectively.
<table><tr><td>Configuration</td><td>Initialization</td><td>Trainable scope</td><td>Native objective</td><td>AdamW (lr, β, wd)</td><td>Clip</td><td>Schedule</td></tr><tr><td> $\pi _ { 0 . 5 }$ </td><td>Pretrained</td><td>Full action model</td><td>Flow matching</td><td> $( 5 \times 1 0 ^ { - 5 } , ( 0 . 9 , 0 . 9 5 ) , 1 0 ^ { - 1 0 } )$ </td><td>1</td><td>10,000-step warmup, then constant</td></tr><tr><td> $\pi _ { 0 } { \mathrm { - F A S T } }$ </td><td>Pretrained</td><td>Full action model</td><td>FAST token causal cross-entropy (CE)</td><td> $( 2 . 5 \times 1 0 ^ { - 5 } , ( 0 . 9 , 0 . 9 5 ) , 1 0 ^ { - 1 0 } )$ </td><td>1</td><td>1,000-step warmup, cosine to  $2 . 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Fast-WAM</td><td>Pretrained</td><td>Video and action experts</td><td> $\mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { a c t i o n } }$ </td><td> $( 1 0 ^ { - 4 } , ( 0 . 9 , 0 . 9 9 9 ) , 1 0 ^ { - 2 } )$ </td><td>10</td><td>Zero-warmup cosine</td></tr><tr><td>π0.5-scratch</td><td>Random</td><td>Full action model</td><td>Flow matching</td><td> $( 5 \times 1 0 ^ { - 5 } , ( 0 . 9 , 0 . 9 5 ) , 1 0 ^ { - 1 0 } )$ </td><td>1</td><td>10,000-step warmup, then constant</td></tr></table>

For every source trajectory, we retain the original appearance and generate four deterministic appearance variants. For each variant, the intensities of environment lights other than the projector are multiplied by a factor sampled uniformly from [0.65, 1.35], and their color temperature is sampled uniformly from [3500, 7000] K. Floor materials are sampled from eight official Isaac Sim wood, stone, tile, and carpet materials, while wall materials are sampled from four official paint, ceramic, porcelain, and brick materials. Randomization is authored only in a disposable replay layer; the projection light, scene geometry, camera poses, and source action sequence remain unchanged. The random seed and selected lighting and material settings are stored with each rendered episode.

## 3 ACTION MODEL ADAPTATION AND TRAINING DETAILS

The shared perception and optical action interface keeps the physical inputs and outputs fixed while allowing each action model to retain it own representation and objective. This section reports the model-specific conversions and optimization settings used to test that interface; none of the runs uses real-world LICOA training data.

The settings in Tab. S3 are the configurations stored with the checkpoints used in the main comparison. Each listed run uses one GPU. All four configurations train for 30,000 updates (960,000 sample draws) with batch size 32 and seed 0. Online image transformations are disabled. The external and onboard RGB images are renamed to the feature keys required by each action model, and any unused third visual slot is supplied as a masked blank input.

The $\pi _ { 0 . 5 }$ run performs full fine-tuning: its PaliGemma backbone, Gemma-300M action expert, and action/time projections are all trainable. The π<sub>0</sub>-FAST run also performs full fine-tuning and uses neither low-rank adaptation (LoRA) nor another parameter-efficient fine-tuning (PEFT) module. It tokenizes the physical action chunk with the FAST tokenizer and optimizes causal cross-entropy. Fast-WAM consumes the five-dimensional robot control state and six-dimensional action directly. Its external and onboard RGB images are concatenated at 224 × 448 resolution, its video sequence contains 17 frames, and its video and action losses have equal weight. The π -scratch control uses the same optimizer and architecture as $\pi _ { 0 . 5 }$ but does not load a pretrained checkpoint.

For the camera input ablation in the main paper, both variants reuse the full $\pi _ { 0 . 5 }$ recipe above. Dataset preparation removes either the external or onboard RGB image, and the corresponding input key mapping for the action model is omitted; the other RGB image, robot control state, action interface, update budget, and optimizer remain unchanged.

## 4 OFFLINE ANALYSIS OF ADAPTATION AND DATA SCALE

These analyses characterize two supporting questions behind the real-world evaluation protocol: how much of the pretrained action model must be adapted, and how the amount and diversity of simulation data affect action prediction. We conduct two offline ablations with $\pi _ { 0 . 5 } { : }$ one varies the set of trainable parameters, and the other varies training data scale along two independent axes. These experiments do not use real-world rollouts. Their metric is the flow-matching loss on one fixed held-out set, which makes the training configurations inexpensive to compare but does not measure closed-loop recovery or task success. We therefore treat these results as optimization and data-efficiency evidence rather than a substitute for the success rate (SR), success weighted by path length (SPL), and mean completion time (MT) results in the main paper.

Table S4: Common offline training protocol.  
Setting Value   
Compute Single NVIDIA H200 GPU per run (143,771 MiB); batch size 32   
Precision BF16 mixed precision; gradient checkpointing; no compilation   
Optimizer AdamW, $\beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , weight decay $\mathrm { \bar { 1 0 } ^ { - 1 0 } }$   
Learning rate $5 \times 1 0 ^ { - 5 } ; 1 0 , 0 0 0$ -step linear warmup, then constant   
Update budget 30,000 steps (960,000 sample draws); gradient norm clipped to 1.0   
Logging Log every 100 steps; validate and save every 5,000 steps   
Validation Full 96,510 frames; batch size 32; no rollout evaluation   
Data loading 4 workers; prefetch factor 4; persistent workers   
Software LeRobot 0.6.0, PyTorch 2.11.0+cu128, Transformers 5.5.4, Accelerate 1.14.0, Datasets 4.8.5

## 4.1 Corpus and offline validation split

The matched simulation corpus uses the first 10 InteriorAgent scenes, each of which is a complete, large indoor environment. In each scene, we define five fixed external camera viewpoints and construct three tasks per viewpoint. The resulting $1 0 \times 5 \times 3 = 1 5 0$ spatial placement tasks each have 10 manually collected source trajectories. We render every trajectory with five appearance settings: the original appearance and four variants with randomized lighting intensity, lighting color temperature, wall materials, and floor materials. This gives 50 episodes per task and 7,500 episodes (919,950 frames at 15 Hz) in total. The 1,465 task strings reported in the LeRobot metadata are language instruction instances and should not be interpreted as distinct spatial placement tasks.

This holdout is used only to compare offline validation loss; it is not a benchmark for generalization across scenes or instructions. All 10 InteriorAgent scenes contribute data to the training split used by the main experiments and ablations, and the configurations using the complete corpus include all 150 spatial placement tasks. We split by source trajectory rather than by rendered episode. With split seed=0, one of the 10 source trajectories from every spatial placement task and all five of its appearance variants are assigned to validation; the other nine groups are assigned to training. Hence, the fixed validation set covers all 150 tasks and contains 750 episodes and 96,510 frames. The full training split contains 6,750 episodes and 823,440 frames. Grouping the appearance variants prevents the same source action sequence from appearing on both sides of the split. Every configuration below is evaluated on this identical validation set. Real-world generalization is evaluated separately in previously unseen real-world environments with new instructions

The action model receives timestamp-synchronized onboard and external RGB images; the stored projector input stream is excluded. In the $\pi _ { 0 . 5 }$ input convention, these are mapped to base 0 rgb and left wrist 0 rgb, respectively, while the unused right-wrist slot is supplied as a masked blank image. The five-dimensional robot control state is $( v , \omega , q _ { \mathrm { p } } , q _ { \mathrm { t } } , \rho )$ , where $q _ { \mathrm { p } }$ and $q _ { \mathrm { t } }$ are the pan and tilt targets and $\rho$ is the projection scale rather than an optical field-of-view angle. The six-dimensional action contains the corresponding commanded values and the additional termination indicator. The first and ninety-ninth percentiles computed over the complete 7,500 episode corpus are used to map the robot control state and action channels to [−1, 1]; these same statistics are deliberately held fixed in every data scale run. Appearance variation is generated offline, and no additional online image augmentation is applied.

## 4.2 Training and validation protocol

As summarized in Tab. S4, all runs use LeRobot’s implementation of $\pi _ { 0 . 5 }$ . Its vision-language backbone is the PaliGemma/Gemma-2B variant, and its action expert is the Gemma-300M variant. The logged action model contains 4,143,404,816 parameters in total. Its adapter pads both the five-dimensional robot control state and the six-dimensional action with zeros to form LeRobot’s 32-dimensional tensors. The loss is computed only over the six physical LICOA action channels, and only those channels are kept after inference. The π<sub>0</sub>-FAST adapter instead serializes the normalized robot control state through its prompt preprocessor and passes the unpadded six-dimensional continuous action chunk to the FAST tokenizer. Fast-WAM consumes the five-dimensional robot control state and six-dimensional action directly; its embodiment-dependent state encoder and action input/output layers are resized accordingly, while all shape-compatible pretrained weights are loaded. The action horizon is $H = 1 6$ , corresponding to approximately 1.067 s at 15 Hz. No additional temporal validity mask is introduced.

Using the main paper notation with $m = \pi _ { 0 . 5 }$ , let B be the validation batch size and index its samples by $n \in \{ 1 , \ldots , B \}$ . Sample n contains target action chunk $\mathbf { A } _ { t _ { n } } ^ { * }$ and conditioning input $\boldsymbol { x } _ { t _ { n } } ^ { ( m ) }$ , where $t _ { n }$ is its control time index. Here, H = 16 is the action horizon and $d _ { a } \ = \ 6$ is the physical action dimension, so $\mathbf { A } _ { t _ { n } } ^ { * } \ \in \ \mathbb { R } ^ { H \times d _ { a } }$ . For each sample, draw Gaussian noise $\epsilon _ { n }$ of the same shape, with independent standard normal entries, and a flow time variable $\lambda _ { n } \sim$ Beta(1.5, 1) (subsequently mapped to $0 . 9 9 9 \lambda _ { n } + 0 . 0 0 1 )$ ). Following the implementation’s flow time convention, the noisy action chunk is

$$
\widetilde { \mathbf { A } } _ { t _ { n } , \lambda _ { n } } = \lambda _ { n } \mathbf { \epsilon } _ { n } + ( 1 - \lambda _ { n } ) \mathbf { A } _ { t _ { n } } ^ { * } .\tag{S3}
$$

The corresponding flow target is

$$
\mathbf { r } _ { t _ { n } , \lambda _ { n } } = \epsilon _ { n } - \mathbf { A } _ { t _ { n } } ^ { * } .\tag{S4}
$$

Let $f _ { \theta _ { m } } ^ { ( m ) }$ denote the vector-field predictor native to model $m ,$ with $r \in \{ 1 , \ldots , H \}$ indexing tensor positions along the action horizon and k indexing action channels. We report its mean-squared flow-matching objective

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { v a l i d } } = \frac { 1 } { B H d _ { a } } \sum _ { n = 1 } ^ { B } \sum _ { r = 1 } ^ { H } \sum _ { k = 1 } ^ { d _ { a } } } } \\ & { } & { ( f _ { \theta _ { m } } ^ { ( m ) } \Big ( \tilde { \mathbf { A } } _ { t _ { n } , \lambda _ { n } } , x _ { t _ { n } } ^ { ( m ) } , \lambda _ { n } \Big ) _ { r k } - \mathbf { r } _ { t _ { n } , \lambda _ { n } , r k } ) ^ { 2 } . } \end{array}\tag{S5}
$$

The construction in Eqs. (S3) and (S4) defines the target used by the validation loss in Eq. (S5). The entire 96,510-frame validation set is evaluated every 5,000 updates. A fixed noise/time random number stream is reused for corresponding validation batches across configuration

Table S5: Parameter adaptation ablation for $\pi _ { 0 . 5 }$ . All runs use the full training split, the same 30,000-update budget, and seed. Validation loss is measured at the common final update. Peak memory is the maximum logged CUDA tensor allocation, and runtime is the W&B wall-clock runtime.
<table><tr><td>Configuration</td><td>Initialization</td><td>Trainable params.</td><td>Ratio (%)</td><td>Valid loss ↓</td><td>Peak memory (GiB) ↓</td><td>Runtime (h) ↓</td></tr><tr><td>Projection heads</td><td>Pretrained</td><td>2,165,792</td><td>0.0523</td><td>0.3901</td><td>19.65</td><td>15.27</td></tr><tr><td>Action expert</td><td>Pretrained</td><td>693,422,112</td><td>16.7356</td><td>0.3030</td><td>21.79</td><td>15.34</td></tr><tr><td>Full</td><td>Pretrained</td><td> $^ { 4 , 1 4 3 , 4 0 4 , 8 1 6 }$ </td><td>100.0000</td><td>0.3004</td><td>40.23</td><td>19.61</td></tr></table>

Table S6: Data scale ablation for $\pi _ { 0 . 5 }$ with full fine-tuning. “Available” includes the five held-out appearance variants per task. Every run uses the same validation set (150 tasks, 750 episodes, and 96,510 frames) and 30,000 optimizer updates. The table reports the common endpoint.
<table><tr><td>Scale axis</td><td>Train tasks</td><td>Available eps./task</td><td>Train eps./task</td><td>Train episodes</td><td>Train frames</td><td>Eff. epochs</td><td>Valid loss ↓</td></tr><tr><td>Trajectory density</td><td>150</td><td>10</td><td>5</td><td>750</td><td>95,445</td><td>10.06</td><td>0.8755</td></tr><tr><td>Trajectory density</td><td>150</td><td>20</td><td>15</td><td>2,250</td><td>274,845</td><td>3.49</td><td>0.5493</td></tr><tr><td>Trajectory density</td><td>150</td><td>30</td><td>25</td><td>3,750</td><td>456,510</td><td>2.10</td><td>0.4182</td></tr><tr><td>Trajectory density</td><td>150</td><td>40</td><td>35</td><td>5,250</td><td>641,470</td><td>1.50</td><td>0.3361</td></tr><tr><td>Trajectory density</td><td>150</td><td>50</td><td>45</td><td>6,750</td><td>823,440</td><td>1.17</td><td>0.3004</td></tr><tr><td>Task diversity</td><td>10</td><td>50</td><td>45</td><td>450</td><td>53,610</td><td>17.91</td><td>1.0454</td></tr><tr><td>Task diversity</td><td>50</td><td>50</td><td>45</td><td>2,250</td><td>274,245</td><td>3.50</td><td>0.5574</td></tr><tr><td>Task diversity</td><td>150</td><td>50</td><td>45</td><td>6,750</td><td>823,440</td><td>1.17</td><td>0.3004</td></tr></table>

to reduce Monte-Carlo variation. We use the loss at the common 30,000-update endpoint as the primary table entry and retain all six checkpoints for convergence plots. Runs share one seed; this controls the comparison but does not constitute a multi-seed uncertainty estimate.

## 4.3 Parameter adaptation scope

Table S5 compares three configurations. In Projection heads, only the action input/output projections and time multilayer perceptron (MLP) are trainable. Action expert, a parameter-efficient alternative, also updates the Gemma action expert while freezing the vision-language backbone. Full updates the complete pretrained action model and is the recipe used by $\pi _ { 0 . 5 }$ in the main real-world comparison. The logged parameterization comprises 3,449,982,704 parameters in the vision-language and image backbones, 691,256,320 action expert parameters, and 2,165,792 action/time projection parameters. In the Action expert configuration, the first group is frozen and the latter two are optimized.

Alongside endpoint validation loss, we record peak allocated GPU memory and wall-clock runtime. Peak memory is measured with torch.cuda.max memory allocated, not reserved memory from nvidia-smi, and we take the maximum value logged over all training updates. Runtime is the W&B runtime from run initialization to completion, so it includes action model and dataset construction in addition to optimization, validation, and checkpointing.

Updating the action expert reduces the endpoint loss by 22.3% relative to training only the projection heads (0.3030 versus 0.3901). Full fine-tuning improves the endpoint by a further 0.0026, while increasing the trainable fraction from 16.74% to 100% and peak allocated memory from 21.79 to 40.23 GiB. The Projection heads and Action expert runs both take about 15.3 h, whereas Full takes 19.6 h. Thus, Action expert is a parameter-efficient alternative that nearly matches Full on this offline metric with 83.3% fewer trainable parameters and 45.8% lower peak memory; all main paper $\pi _ { 0 . 5 }$ results nevertheless use Full fine-tuning.

## 4.4 Data scale analysis

We vary data scale in two complementary ways using Full fine-tuning in all completed data scale runs. The trajectory density axis keeps all 150 tasks and uses 10, 20, 30, 40, or 50 available episodes per task. Because five episodes per task belong to the fixed validation group, these levels contain 1, 3, 5, 7, or 9 training source trajectories (5, 15, 25, 35, or 45 rendered training episodes) per task. The subsets are nested.

The task diversity axis fixes the maximum 50 available episodes per task and trains on nested subsets of 10, 50, or 150 spatial placement tasks. Validation always covers all 150 tasks. Consequently, validation for the 10- and 50-task action models mixes seen and unseen spatial placement tasks and should not be described as a pure within-task reconstruction test.

At the common 30,000-update endpoint, increasing available episodes per task from 10 to 50 reduces validation loss monotonically from 0.8755 to 0.3004, a 65.7% reduction. Increasing the training task set from 10 to 150 likewise reduces the endpoint loss from 1.0454 to 0.3004, a 71.3% reduction. Thus both additional trajectories for covered tasks and broader spatial task coverage substantially improve held-out action prediction under the fixed compute budget.

The checkpoint curves in Fig. S2 also expose an important endpoint effect. The run using the complete training split reaches its lowest measured loss of 0.2589 at 15,000 updates before ending at 0.3004. Smaller training subsets usually turn upward earlier: for example, the 10-episode trajectory run moves from its minimum of 0.5379 at 10,000 updates to 0.8755, and the 10-task run moves from 0.6588 at 5,000 updates to 1.0454. This behavior is consistent with overfitting as smaller training subsets are revisited more often. We nevertheless use the prespecified 30,000-update endpoint for the table so that no configuration benefits from post-hoc checkpoint selection.

All data scale action models receive the same 30,000 optimizer updates, so the study holds compute steps rather than epochs fixed. Smaller subsets are therefore revisited more often, as quantified by the effective epoch column in Tab. S6. This design isolates whether additional trajectory density or spatial task diversity improves held-out action prediction under a common optimization budget, but it also means that data quantity and sample reuse cannot be interpreted independently. Together with the single seed and normalization statistics computed over the complete corpus, this is a stated limitation of the offline study.

![](images/d65547022188f9a3167fce2d59403be43ecd357b0c04db1246c6eef6fd58950d.jpg)

![](images/e37f1833892f4bfe8d7803b03b6db775387052d0b1094ae9b11baf0cecdb559d.jpg)

![](images/81ebaf104419e2e0094165d354ff8fbf7c9a564612d0db257b5a0ce44e2abe34.jpg)  
Figure S2: Offline validation loss at the six common checkpoints from 5,000 to 30,000 updates. The adaptation scope panel uses the complete training split. The data scale panels use full fine-tuning and vary either the available episodes per task or the number of training tasks. All curves use the same fixed validation set; lower is better. Curves combine distinct line styles and markers so that they remain identifiable without color.