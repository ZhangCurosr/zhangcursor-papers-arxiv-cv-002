# Distilling Privileged Control Barrier Functions into RGB-Only Safety Filters for Dynamic Visual Navigation

Seungyeon Yoo<sup>∗</sup>, Gawon Lee<sup>∗</sup>, Seungwoo Jung<sup>∗</sup>, Inkyu Jang, and H. Jin Kim

Abstract— RGB-only end-to-end visual navigation policies remain vulnerable to collisions in real-world dynamic environments, motivating a dedicated safety layer. Existing visual Control Barrier Function (CBF) approaches seek to provide safety from RGB observations, but often rely on real-time rendering or explicit scene reconstruction and are primarily designed for static scenes, limiting their practicality for onboard deployment. We propose a teacher-student visual distillation framework that transfers the safety behavior of a privileged CBF teacher to an RGB-only student filter for dynamic environments. The student maps a short RGB history, robot velocity, and a nominal control action directly to a safe action, while the teacher uses ground-truth robot and obstacle states in a real-to-sim dynamic Gaussian Splatting environment. To reduce the teacher-student information gap, the teacher constructs safety constraints only from obstacles observable within the student’s RGB history. It also accounts for obstacle-velocity uncertainty to improve robustness to motion variations, while action augmentation exposes the student to diverse safe and unsafe nominal actions to better capture the safety boundary. At deployment, the student requires only RGB observations and robot velocity, without explicit 3D reconstruction or online rendering. Experiments show that the proposed method outperforms visual CBF baselines and improves the safety of RGB-based navigation policies under dynamic obstacle motion. Project page: https://syeon-yoo.github.io/distill-cbf-site/.

## I. INTRODUCTION

RGB-only end-to-end visual navigation policies which directly map camera observations to control actions offer several practical advantages for real-world deployment. By relying only on an RGB camera, they reduce hardware complexity, cost, and weight. They also remove the need for explicit mapping or reconstruction in the control loop, enabling low-latency visual feedback. Despite continued advances in end-to-end navigation [1]–[3], however, such policies remain susceptible to collisions, particularly in dynamic environments. This lack of safety is a major obstacle to their deployment and motivates a dedicated safety layer.

Designing such a safety layer without sacrificing the simplicity of RGB-only navigation is challenging. Conventional safety mechanisms often rely on robot localization, obstacle-state estimation, depth sensing, or explicit 3D scene representations. Introducing these components at deployment would undermine the main benefit of an end-to-end RGBonly policy [4]. The difficulty is further amplified in dynamic environments, where safe control requires reasoning not only about obstacle geometry but also about obstacle motion and its uncertainty, all from RGB observations within a limited field of view [5], [6]. Thus, for RGB-only visual navigation, it is desirable for a safety filter to operate directly from visual observations in real time, without relying on explicit scene reconstruction and obstacle-state estimation at deployment.

![](images/0fae0fca010b6160cc106980375ec39eef271cf0ebde71b34c8ed8edb2b717ba.jpg)  
Fig. 1: Real-world navigation example of the proposed RGBonly safety filter. We distill a privileged state-based CBF teacher into an RGB-only student filter in a real-to-sim simulation with moving obstacles. In real-world dynamic environments, the proposed filter improves both collision avoidance and task success of nominal visual navigation policies, without requiring a 3D scene representation, robot localization, or obstacle state estimation at deployment.

Control Barrier Functions (CBFs) provide a systematic way to improve navigation safety by constraining the control input [7]. Given a set of safe states, CBFs admit only control inputs that keep the system within the set, and a CBF-based safety filter minimally modifies a nominal action to satisfy this constraint. Recent visual CBF approaches [8], [9] extend this idea to image-based observations. However, they assume static scenes, and their safety evaluation requires an 3D scene representation at deployment, such as real-time depth rendering from a Neural Radiance Field (NeRF) [10] or a 3D Gaussian Splatting (GS) map [11]. These limitations hinder their real-time onboard operations in dynamic environments.

This work proposes a different approach: rather than evaluating a CBF from a 3D scene representation at deployment, we use a privileged state-based CBF only during training and distill its safety behavior into an RGB-only student safety filter. As illustrated in Fig. 1, the student maps a nominal action directly to a safe action using RGB inputs, while the privileged teacher uses ground-truth robot and obstacle states available in simulation. To reduce the teacherstudent information gap, the teacher considers only obstacles appearing in the short history of RGB inputs provided to the student. It further accounts for uncertainty in obstacle velocities to improve robustness to changes in obstacle motion. Moreover, we augment the training data with multiple teacher-labeled candidate actions for each observation to encourage the student to learn the boundary between safe and unsafe actions. Training data are generated in a real-tosim dynamic GS simulation of each target environment [2], providing photorealistic observations and privileged states across diverse dynamic scenarios. At deployment, the student filters nominal actions using only RGB observations and robot velocity, without requiring a 3D scene representation, robot localization, or obstacle state estimation.

Our contributions can be summarized as follows:

• A privileged CBF-to-RGB distillation framework enabling RGB-only safety filtering in dynamic environments without a 3D scene representation, robot localization, or obstacle state estimation at deployment.

• A privileged CBF teacher design that reduces the teacher-student information gap by considering only obstacles appearing in the student’s short RGB input sequence and accounts for uncertainty in obstacle velocities for robustness to unpredictable obstacle motion.

• Experiments with multiple nominal policies showing that our filter outperforms prior visual CBF methods in simulation and reduces collisions in real-world dynamic environments.

## II. RELATED WORK

## A. RGB-Only Visual Navigation

RGB-only end-to-end visual navigation policies directly map visual observations to actions and are commonly trained in simulation using imitation or reinforcement learning [1], [12]–[14]. Despite efforts to improve performance through domain randomization [1] or cross-modal supervision [13], [14], their real-world applicability remains limited by the sim-to-real gap and the absence of safety considerations.

One alternative direction is general navigation models (GNMs) [3], [15], [16], which leverage large real-world cross-embodiment datasets to improve zero-shot transfer across environments and robots. Another direction is GSbased real-to-sim pipelines [2], [17]–[19], which build a digital twin from a video capture and generate large-scale photorealistic navigation data, including for dynamic scenes [2], [19]. These approaches improve the real-world applicability of RGB-only navigation policies by reducing the gap between training and deployment. However, safety around moving obstacles remains insufficiently addressed, motivating a dedicated safety layer for RGB-only visual navigation.

## B. Visual CBF-Based Safety Filter

When safety filters are deployed on real robots, the system state—obstacle shapes, poses, and velocities—is not available, and safety must be assessed from onboard perception alone. Prior methods recover this state at test time, either from a scene representation or from depth. NeRF-CBF [8] certifies each candidate action by rendering the predicted next observation with a NeRF [10], and SAFER-Splat [9] constructs its CBF from the Gaussian primitives of an onlinebuilt splat [11]. Both assume a static world, since relaxing this would require remapping the scene faster than obstacles move. On the depth side, V-CBF [20] learns an image-space barrier from RGB-D input but assumes stationary obstacles and requires segmentation labels, while Depth-CBF [21] fits a local quadratic barrier to the point cloud without training but keeps no history and hence no notion of obstacle velocity. Both presuppose a depth sensor. Instead, we distill a privileged CBF-based filter into one that runs on RGB alone, requiring no depth sensor, scene model, rendering/mapping at inference while accounting for obstacle motion.

## C. Learning from Privileged Teachers

Privileged learning and teacher-student approaches exploit information available only during training to improve policies that operate under partial observations at deployment [22]–[25]. Such approaches have been widely used for policy learning and perception, in which a privileged state-based teacher supervises a student acting on raw sensors [23]–[25], but rarely for distilling the corrective behavior of a model-based safety filter. Our work specifically distills a privileged state-based CBF into an RGB-only safety filter, while explicitly addressing the teacher-student information gap induced by partial visual observations.

III. PRELIMINARIES: CONTROL BARRIER FUNCTIONS

Consider a control-affine system

$$
{ \dot { \mathbf { x } } } = f ( \mathbf { x } ) + g ( \mathbf { x } ) { \mathbf { u } } , \qquad \mathbf { x } \in \mathcal { X } , \ { \mathbf { u } } \in \mathcal { U } ,\tag{1}
$$

where x and u denote the state and control input, respectively. Safety is described by a continuously differentiable function $h : \mathcal { X }  \mathbb { R }$ , with the safe set $\mathcal { C } : = \{ \mathbf { x } \mid h ( \mathbf { x } ) \geq 0 \}$ Thus, $h > 0$ indicates a safe state, while $h = 0$ defines the safety boundary.

A control barrier function (CBF) keeps the system inside C by constraining how quickly h can decrease. For an extended class-K function $\alpha ( \cdot ) .$ , a safe control input satisfies

$$
\begin{array} { r } { L _ { f } h ( \mathbf x ) + L _ { g } h ( \mathbf x ) \mathbf u \geq - \alpha ( h ( \mathbf x ) ) . } \end{array}\tag{2}
$$

Under standard regularity conditions, enforcing this inequality renders C forward invariant [7].

Following [26], the CBF condition (2) is affine in the control input, it can be incorporated into a quadratic program to minimally modify a reference control input:

$$
\begin{array} { r } { \mathbf { u } ^ { \star } = \arg \underset { \mathbf { u } \in \mathcal { U } } { \operatorname* { m i n } } \left\| \mathbf { u } - \mathbf { u } _ { \mathrm { r e f } } \right\| _ { 2 } ^ { 2 } } \\ { \mathrm { s . t . } \quad L _ { f } h + L _ { g } h \mathbf { u } \geq - \alpha ( h ) . } \end{array}\tag{3}
$$

This is a small quadratic program (CBF-QP) that returns the reference input unchanged whenever it is already safe and otherwise makes the smallest correction that restores safety. Multiple obstacles can be handled by stacking one CBF constraint per barrier $h _ { i }$ , provided that the resulting QP remains feasible. The reference controller and the safety constraint are fully decoupled, which is what allows us to wrap an arbitrary visual navigation policy with a CBF-QP.

## IV. PROBLEM FORMULATION

We consider a wheeled mobile robot equipped with a single forward-facing RGB camera, navigating a fixed target environment that contains static obstacles and a moving human. The robot is modeled as a unicycle with state ${ \bf x } ^ { \mathrm { r } } =$ $( p _ { x } , p _ { y } , \psi , v )$ , i.e., planar position, heading, and forward speed, driven by the control input $\mathbf { u } = \left( a , \omega \right)$ of forward acceleration and yaw rate:

$$
\dot { p } _ { x } = v \cos \psi , \quad \dot { p } _ { y } = v \sin \psi , \quad \dot { \psi } = \omega , \quad \dot { \upsilon } = a ,\tag{4}
$$

which is control-affine and hence of the form (1). The environment is described by a set of static obstacles $\mathcal { O }$ and the human state $\mathbf { x } ^ { \mathrm { { h } } } = \left( \mathbf { p } ^ { \mathrm { { h } } } , \mathbf { v } ^ { \mathrm { { h } } } \right)$ . We collect these into the privileged state $\mathbf { x } ~ = ~ ( \mathbf { x } ^ { \mathrm { { r } } } , \mathbf { x } ^ { \mathrm { { h } } } , \mathcal { O } )$ and define the safe set $\mathcal { C } = \{ \mathbf { x } ~ | ~ h ( \mathbf { x } ) ~ \geq ~ 0 \}$ , where h is positive whenever the robot keeps a minimum clearance from both the static obstacles and the human. At deployment the filter can rely only on onboard quantities: a short history of RGB frames $o _ { t } ~ = ~ ( I _ { t - 2 \Delta } , I _ { t - \Delta } , I _ { t } )$ spaced $\Delta$ control steps apart, the nominal command $\mathbf { u } _ { t } ^ { \mathrm { n o m } }$ , and the forward speed $v _ { t }$ from the wheel encoder.

Nominal policy. The robot is driven by an RGB-only navigation policy $\pi ^ { \mathrm { n o m } }$ that, at each control step t, maps the current image and the robot speed to a reference command $\mathbf { u } _ { t } ^ { \mathrm { n o m } } = \pi ^ { \mathrm { n o m } } ( o _ { t } , v _ { t } )$ . We make no assumption on how $\pi ^ { \mathrm { n o m } }$ was obtained and treat it as a black box that provides no safety guarantee of its own.

Privileged safety filter. If the privileged state were available, the CBF-QP (3) with $\mathbf { u } _ { \mathrm { r e f } } \ = \ \mathbf { u } _ { t } ^ { \mathrm { n o m } }$ would yield a minimally invasive safe command

$$
\mathbf { u } _ { t } ^ { \mathrm { s a f e } } = \mathcal { F } \big ( \mathbf { x } _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \big ) ,\tag{5}
$$

which coincides with $\mathbf { u } _ { t } ^ { \mathrm { n o m } }$ away from ∂C and overrides it only when needed to keep the system in C. However, evaluating $\mathcal { F }$ requires the robot pose, the position and velocity of the human, and the scene geometry.

Objective. Our goal is to learn an RGB-only safety filter $\mathcal { F } _ { \theta } : \left( o _ { t } , v _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \right) \mapsto \hat { \mathbf { u } } _ { t } ^ { \mathrm { s a f e } }$ that reproduces the behavior of the privileged filter without access to $\mathbf { x } _ { t } \colon$

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { \mathcal { D } } \Big \| \mathcal { F } _ { \theta } \big ( o _ { t } , v _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \big ) - \mathcal { F } \big ( \mathbf { x } _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \big ) \Big \| _ { 2 } ^ { 2 } ,\tag{6}
$$

where $\left( o _ { t } , v _ { t } , \mathbf { x } _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \right) \sim \mathcal { D }$ is the distribution of states and observations visited when the nominal policies operate in the target environment. We use a photorealistic digital twin of the target environment can be built from a single camera capture, in which the privileged state is available and dynamic scenarios can be generated at scale.

![](images/9ce3a6fea419b6968d4eb02c253b8ba34d651dd4449705ca7054bde13303e912.jpg)  
Fig. 2: Training pipeline. We train the RGB-only student safety filter by distilling a privileged teacher safety filter. The teacher draws its privileged state from the simulation, while the student learns to reproduce the teacher’s safe action from the observations available at deployment alone.

Two properties make the problem non-trivial. First, to preserve the low-latency benefit of the end-to-end nominal policy, $\mathcal { F } _ { \theta }$ must filter actions directly from $o _ { t }$ and $v _ { t } ,$ without estimating obstacle geometry at deployment. Second, (6) is only well-posed if the teacher’s decisions are inferable from $o _ { t } .$ In other words, $\mathcal { F }$ must depend on the privileged state only through quantities that are observable from a short RGB history, such as the relative position and motion of the human when it is in view. These two properties drive the distillation framework and the teacher design in Section V.

## V. RGB-ONLY SAFETY FILTER DISTILLATION

To distill a CBF-based safety filter which relies on groundtruth states into a RGB-only safety filter, we design two main components, as shown in Fig. 2. The first component is a teacher CBF-based safety filter which maps nominal actions produced by RGB-only navigation policies to safe actions, using privileged information provided from a simulator. The second component is the student RGB-only safety filter. The student maps a stack of RGB observations and a nominal action to a safe action by distilling the teacher inside the photorealistic simulator [2], which allows us to deploy the student in the real environments without privileged information and with multiple nominal policies.

## A. Privileged State-Based CBF Teacher

The teacher could read the exact simulator state, but we restrict it to what the student can in principle infer from $o _ { t } .$ , so that the distillation objective (6) is well posed. Static geometry enters as the nearest raycast hit per bearing across the robot’s field of view, giving N boundary points treated as discs. The human enters only if it was inside the field of view at one of the three steps $\{ t , t - \Delta , t - 2 \Delta \}$ of $o _ { t }$ With $t - \tau \ ( \tau \in \{ 0 , \Delta , 2 \Delta \} )$ the most recent such step, the teacher uses the finite-difference velocity $\hat { \mathbf { v } } ^ { \mathrm { h } }$ at that step and the extrapolated position $\hat { \mathbf { p } } ^ { \mathrm { h } } = \mathbf { p } ^ { \mathrm { h } } ( t - \tau ) + \tau \hat { \mathbf { v } } ^ { \mathrm { h } }$ . The teacher never sees the human’s current velocity, only a stale estimate of it, and this is the uncertainty the barrier must absorb.

Uncertainty-aware DP-CBF. We build on the dynamic parabolic CBF (DP-CBF) [27], which incorporates the relative velocity of a dynamic obstacle but assumes that velocity is known exactly and constant. In our setting, the human’s velocity is inferred from a short observation history and becomes stale when the human accelerates, turns, or leaves the field of view, so treating the estimate as exact can make the barrier overconfident. Feeding the teacher the exact privileged velocity would avoid this but widen the information gap to the RGB-only student. We therefore augment the DP-CBF with an uncertainty-dependent tightening margin, while keeping the teacher consistent with what the student can infer from its observations.

With $( \tilde { v } _ { \mathrm { r e l } , x } , \tilde { v } _ { \mathrm { r e l } , y } )$ the relative velocity in an obstacle’s frame where the x-axis is aligned with the line-of-sight,

$$
h = \tilde { v } _ { \mathrm { r e l } , x } + \lambda ( \mathbf { x } ) \tilde { v } _ { \mathrm { r e l } , y } ^ { 2 } + \mu ( \mathbf { x } ) ,\tag{7}
$$

where $\lambda , \mu ~ > ~ 0$ are clearance-dependent gains defined in [27]. For the human, h is evaluated at $\hat { \mathbf { v } } ^ { \mathrm { h } }$ , which is in error whenever the human changes speed or heading within τ. We model the error as $\varepsilon \sim \mathcal { N } ( \mathbf { 0 } , \Sigma )$ , with an anisotropic covariance in the human’s motion frame, $\Sigma = \sigma _ { \parallel } ^ { 2 } \hat { \mathbf { u } } \hat { \mathbf { u } } ^ { \top } + \sigma _ { \bot } ^ { 2 } ( I -$ $\hat { \mathbf { u } } \hat { \mathbf { u } } ^ { \top } ) , \hat { \mathbf { u } } = \hat { \mathbf { v } } ^ { \mathrm { h } } / \lVert \hat { \mathbf { v } } ^ { \mathrm { h } } \rVert$ . We use an anisotropic covariance with $\sigma _ { \parallel } > \sigma _ { \perp }$ , assuming that pedestrians are more likely to vary their speed than abruptly change heading. This emphasizes uncertainty along the current motion direction, encouraging the filter to yield to crossing pedestrians without uniformly enlarging the safety margin.

We propagate this velocity uncertainty to the barrier using a first-order approximation, $\sigma _ { h } ^ { 2 } = \nabla _ { \hat { \mathbf { v } } } h \Sigma \nabla _ { \hat { \mathbf { v } } } h ^ { \top } , \tilde { h } = h - \beta \sigma _ { h }$ , where $\beta > 0$ controls conservatism and $\beta \sigma _ { h }$ provides a tightening margin proportional to the propagated barrier uncertainty. Inspired by Gaussian chance-constraint tightening [28], we use this as an uncertainty-aware margin rather than a probabilistic guarantee, with fixed Σ to avoid excessive conservatism under prolonged occlusions.

Teacher safety filter. Over the N static barriers (7) and, when present, the uncertainty-aware human barrier, the teacher solves at every step

$$
\begin{array} { r l } & { \mathbf { u } _ { t } ^ { \mathrm { s a f e } } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { u } \in \mathcal { U } } \left\| \mathbf { u } - \mathbf { u } _ { t } ^ { \mathrm { n o m } } \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \mathrm { s . t . } ~ L _ { f } h _ { i } + L _ { g } h _ { i } \mathbf { u } \geq - \alpha h _ { i } } \\ & { \qquad a _ { \mathrm { m i n } } \leq a \leq a _ { \mathrm { m a x } } , ~ | \omega | \leq \omega _ { \mathrm { m a x } } } \end{array}\tag{8}
$$

This QP is strictly convex, so ${ \bf u } _ { t } ^ { \mathrm { s a f e } }$ is unique and varies continuously with $\left( \mathbf { x } _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } \right)$ , a well-posed regression target for (9), and it involves no quantity the student cannot recover from three frames.

## B. RGB-Only Student Safety Filter

Datasets. To train the student RGB-only safety filter, we collect datasets using the teacher in the dynamic GS simulation [2]. At every control step t we log the camera frames $o _ { t } .$ , the robot speed $v _ { t } .$ , the nominal policy’s action $\mathbf { u } _ { t } ^ { \mathrm { n o m } }$ , and the filtered action ${ \bf u } _ { t } ^ { \mathrm { s a f e } }$ . To cover both the states the unfiltered policy visits and the states the filtered system actually occupies, we collect two sets of rollouts. One set is nominal rollouts, where the robot executes $\mathbf { u } _ { t } ^ { \mathrm { n o m } }$ , and the other is filtered rollouts, where ${ \bf u } _ { t } ^ { \mathrm { s a f e } }$ is executed. We discard samples whose QP did not converge to an optimal solution. As nominal policies, we use imitation learning [2], reinforcement learning [29], and general navigation [3] policies.

Next, we augment the collected samples by perturbing the executed action and relabeling the sample with the teacher safety filter output. The teacher relabels the samples offline at each logged state with perturbed candidate actions, producing additional $\left( o _ { t } , v _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } , \mathbf { u } _ { t } ^ { \mathrm { s a f e } } \right)$ pairs without any further simulation. We draw perturbed action candidates from a mixture of a Gaussian centered around the logged action and uniform distribution over the action space. In addition to these candidates, when any CBF constraint intersects the action space, we draw more perturbed actions from that intersection. This augmentation broadens the candidate action space coverage at each visited state, thereby making student truly approximate the teacher.

Architecture and Loss Function. The student maps the observations available onboard to the teacher’s filtered action. Each frame is encoded independently by a ten-layer residual CNN into a 20-dimensional feature vector, and a three-layer MLP maps the concatenated frame features and scalar inputs to the predicted safe action $\begin{array} { r l } { \hat { \mathbf { u } } _ { t } ^ { \mathrm { s a f e } } } & { { } = } \end{array}$ $\mathbf { M L P } ( \mathbf { C N N } ( o _ { t } ) , v _ { t } , \mathbf { u } _ { t } ^ { \mathrm { n o m } } )$ . Because interventions are rare, an action loss alone admits a shortcut that passes the nominal action through without grounding the decision to intervene in perception. We therefore add an auxiliary head $\begin{array} { r l } { \hat { c } _ { t } } & { { } = } \end{array}$ $\mathbf { M L P } _ { \mathrm { v i s } } ( \mathbf { C N N } ( o _ { t } ) )$ that predicts from the image features alone whether the human is inside the camera’s field of view, with the label $c _ { t } \in \{ 0 , 1 \}$ computed in the simulator from the ground-truth human position and robot pose. Since the head never sees the scalar inputs, its gradients encourage the CNN to represent human presence, and it is discarded at deployment. With ${ \bf u } _ { t } ^ { \mathrm { s a f e } }$ the teacher’s QP solution, the training objective is

$$
\mathcal { L } ( \theta ) = \frac { 1 } { \vert \mathcal { D } \vert } \sum _ { \mathcal { D } } \left[ \mathcal { L } _ { \mathrm { d i s t i l l } } + \lambda _ { \mathrm { v i s } } \mathcal { L } _ { \mathrm { v i s } } \right] ,\tag{9}
$$

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \left. \hat { \mathbf { u } } _ { t } ^ { \mathrm { s a f e } } - \mathbf { u } _ { t } ^ { \mathrm { s a f e } } \right. _ { 2 } ^ { 2 } ,\tag{10}
$$

$$
\mathcal { L } _ { \mathrm { v i s } } = \mathrm { B C E } ( \hat { c } _ { t } , c _ { t } ) ,\tag{11}
$$

where $\lambda _ { \mathrm { v i s } } = 0 . 0 5$ and BCE is the binary cross-entropy on the head’s logit.

## VI. SIMULATION EXPERIMENTS

We first evaluate the distilled visual safety filter in the ReaDy-Go simulator [2], which renders the reconstructed scenes and dynamic human avatars with Gaussian splatting. We test whether the filter, operating only on onboard observations, improves both the safety and the task performance of the nominal policies.

## A. Experimental Setup

1) Task description: The task is point-goal navigation, where the robot must travel from a start pose to a goal position while avoiding static obstacles and a moving human. For each scene we sample random start-goal pairs in the free space. A trial succeeds if the robot reaches the goal within the time limit of 50 seconds, and fails on a human collision, a static collision, or a timeout. The trials, including the startgoal pairs and the human trajectory, are identical across all nominal policies and safety filters.

TABLE I: Simulation results. Each nominal–filter–scene combination is evaluated over 100 trials. Best values are in bold.
<table><tr><td rowspan="3">Nominal Safety filter</td><td rowspan="3"></td><td colspan="5">Outside</td><td colspan="5">Lobby</td><td colspan="5">Library</td></tr><tr><td>SR↑</td><td>CFR↑ (%)</td><td>m ↑ (m)</td><td>T (s)</td><td>Failure cases↓ S/D/TO (%)</td><td>SR↑ (%)</td><td>CFR↑</td><td>m↑</td><td>T</td><td>Failure cases↓</td><td>SR↑</td><td>CFR↑</td><td>m↑</td><td>T</td><td>Failure cases↓ S/D/TO (%)</td></tr><tr><td>(%)</td><td></td><td></td><td></td><td></td><td></td><td>(%)</td><td>(m)</td><td>(s)</td><td>S/D/TO (%)</td><td>(%)</td><td>(%)</td><td>(m)</td><td>(s)</td><td></td></tr><tr><td rowspan="5">IL</td><td>None</td><td>63</td><td>64</td><td>0.26</td><td>10.2</td><td>17/19/1</td><td>65</td><td>65</td><td>0.30</td><td>8.4</td><td>4/31/0</td><td>56</td><td>56</td><td>0.19</td><td>9.6</td><td>25/19/0</td></tr><tr><td>Teacher</td><td>88</td><td>98</td><td>0.37</td><td>12.6</td><td>0/2/10</td><td>85</td><td>87</td><td>0.41</td><td>10.4</td><td>0/13/2</td><td>89</td><td>93</td><td>0.34</td><td>12.2</td><td>0/7/4</td></tr><tr><td>SAFER-Splat</td><td>72</td><td>81</td><td>0.30</td><td>11.0</td><td>6/13/9</td><td>71</td><td>72</td><td>0.33</td><td>8.8</td><td>0/28/1</td><td>67</td><td>72</td><td>0.25</td><td>10.4</td><td>14/14/5</td></tr><tr><td>NeRF-CBF</td><td>47</td><td>74</td><td>0.19</td><td>16.3</td><td>16/10/27</td><td>22</td><td>76</td><td>0.11</td><td>16.8</td><td>1/23/54</td><td>65</td><td>68</td><td>0.23</td><td>12.7</td><td>20/12/3</td></tr><tr><td>Ours</td><td>87</td><td>93</td><td>0.35</td><td>12.2</td><td>2/5/6</td><td>78</td><td>82</td><td>0.37</td><td>10.4</td><td>0/18/4</td><td>88</td><td>91</td><td>0.32</td><td>12.2</td><td>1/8/3</td></tr><tr><td rowspan="5">RL</td><td>None</td><td>56</td><td>56</td><td>0.21</td><td>9.6</td><td>26/18/0</td><td>56</td><td>56</td><td>0.23</td><td>7.6</td><td>15/29/0</td><td>37</td><td>37</td><td>0.13</td><td>6.5</td><td>40/23/0</td></tr><tr><td>Teacher</td><td>93</td><td>95</td><td>0.43</td><td>12.1</td><td>1/4/2</td><td>85</td><td>88</td><td>0.48</td><td>10.2</td><td>1/11/3</td><td>72</td><td>90</td><td>0.23</td><td>10.1</td><td>0/10/18</td></tr><tr><td>SAFER-Splat</td><td>74</td><td>80</td><td>0.30</td><td>11.7</td><td>2/18/6</td><td>61</td><td>67</td><td>0.27</td><td>7.6</td><td>4/29/6</td><td>53</td><td>55</td><td>0.19</td><td>7.5</td><td>21/24/2</td></tr><tr><td>NeRF-CBF</td><td>65</td><td>80</td><td>0.24</td><td>16.1</td><td>4/16/15</td><td>26</td><td>76</td><td>0.11</td><td>12.6</td><td>2/22/50</td><td>37</td><td>38</td><td>0.12</td><td>9.4</td><td>34/28/1</td></tr><tr><td>Ours</td><td>87</td><td>90</td><td>0.38</td><td>11.5</td><td>2/8/3</td><td>82</td><td>84</td><td>0.46</td><td>10.6</td><td>0/16/2</td><td>65</td><td>82</td><td>0.21</td><td>10.1</td><td>7/11/17</td></tr><tr><td rowspan="5">ViNT</td><td>None</td><td>38</td><td>42</td><td>0.15</td><td>8.6</td><td>43/15/4</td><td>38</td><td>70</td><td>0.22</td><td>8.1</td><td>10/20/32</td><td>31</td><td>31</td><td>0.11</td><td>6.0</td><td>49/20/0</td></tr><tr><td>Teacher</td><td>70</td><td>95</td><td>0.29</td><td>10.3</td><td>2/3/25</td><td>49</td><td>96</td><td>0.31</td><td>10.9</td><td>0/4/47</td><td>60</td><td>81</td><td>0.21</td><td>10.5</td><td>7/12/21</td></tr><tr><td>SAFER-Splat</td><td>67</td><td>82</td><td>0.27</td><td>10.2</td><td>5/13/15</td><td>44</td><td>85</td><td>0.25</td><td>8.9</td><td>0/15/41</td><td>58</td><td>60</td><td>0.20</td><td>7.8</td><td>16/24/2</td></tr><tr><td>NeRF-CBF</td><td>25</td><td>60</td><td>0.12</td><td>14.1</td><td>27/13/35</td><td>14</td><td>82</td><td>0.09</td><td>15.4</td><td>0/18/68</td><td>32</td><td>44</td><td>0.12</td><td>11.5</td><td>30/26/12</td></tr><tr><td>Ours</td><td>72</td><td>90</td><td>0.28</td><td>10.8</td><td>4/6/18</td><td>47</td><td>89</td><td>0.28</td><td>13.1</td><td>3/8/42</td><td>60</td><td>74</td><td>0.21</td><td>9.9</td><td>9/17/14</td></tr></table>

2) Evaluation metrics: We report the success rate (SR), the collision-free rate (CFR), the mean safety margin (m¯ ), the average reaching time (T), and a breakdown of failure cases. SR is the fraction of trials in which the robot reaches within 1 m of the goal without a collision and within the time budget. CFR is the fraction of trials without any collision, regardless of whether the goal is reached. m¯ measures minimum surface-to-surface distance per trial between the robot and any obstacles, averaged over all trials with every failed trial counted as zero clearance. T is the mean time from start to goal over all trials, with failed trials assigned the maximum time limit. Failure cases are reported as the percentage of trials ending in a static collision (S), a collision with the human (D), or a timeout (TO).

3) Nominal visual navigation policies:

• IL: We train an imitation learning policy following the ReaDy-Go pipeline [2]. We roll out an expert planner in the simulator and train the policy with behavior cloning.

• RL: We train a per-scene policy with Soft Actor-Critic (SAC, [29], [30]). The policy runs at 5 Hz, holding each command for four steps of the 20 Hz physics. The reward combines a sparse terminal success reward, a collision penalty, a small per-step time penalty, and potential-based goal-progression reward.

• ViNT [3]: ViNT is a GNM trained on large-scale, cross-embodiment data. We use the public checkpoint without fine-tuning. Since ViNT is image-goal conditioned while our scenarios specify goal positions, we capture a goal image from the scene at the goal position, oriented along the start-to-goal direction. Among the five waypoint outputs, we map the third to (v, w), then convert the velocity v to an acceleration reference a.

4) Visual safety filter baselines:

• SAFER-Splat [9]: We adapt SAFER-Splat, which operates directly on Gaussian primitives, and provide it with the same 3D Gaussians in the simulated environment. At every step, the Gaussians of the human are additionally inserted into its constraint set, giving it ground-truth knowledge of the human. The original formulation assumes double-integrator dynamics, and we rewrite its constraints in terms of the unicycle’s acceleration and angular-velocity commands.

• NeRF-CBF [8]: We train NICE-SLAM [31] per scene on RGB-D data rendered from the simulator by densely sweeping the navigable region at the robot’s camera height. Random perturbations of the nominal command are scored by predicting the depth observation at the resulting next state. Because the moving human is absent from the static map, its depth is rendered from the ground-truth pose and composited into the barrier, so the baseline receives the same human information as the other methods.

Both baselines use the same nominal policies, scenarios, dynamics, and success and collision criteria as ours, and unlike our filter they observe the human at its ground-truth pose at test time, as Gaussians for SAFER-Splat and as rendered depth for NeRF-CBF.

## B. Implementation Details

We collect data in three scenes: Outside (Fig. 1), Lobby (Fig. 3), and Library (Fig. 4(a)). For each scene and nominal policy, we roll out 800 episodes with the unfiltered nominal policy and record the teacher’s safe action as a counterfactual label at every transition where the teacher QP is solved. To also cover the states the filtered system visits, we roll out another 800 episodes with the teacher filter. The student for a scene is trained on the union of the six datasets (three nominal policies × two rollout types), which corresponds to the Mixed setting in Table III. Each scene yields roughly 1.5M transitions, which grow to 12M after action augmentation. Each student is trained on a single A6000 GPU.

## C. Navigation Results

Table I shows that across all three scenes and nominal policies, the distilled filter raises the success and collisionfree rate, and enlarges the safety margin over the unfiltered nominal policy. Although the student never observes the privileged state used by the CBF-QP teacher, it closely tracks the teacher’s success and collision-free rate while acting only on the RGB image stream, the nominal command, and the robot speed. It also outperforms SAFER-Splat and NeRF-CBF in nearly every cell. Both baselines observe the human at its ground-truth pose, but their barriers are built for static scenes and have no notion of its velocity. They also depend on a 3D scene representation at test time and are therefore exposed to reconstruction artifacts. NeRF-CBF additionally relies on a sampled search over perturbed actions and stops the robot when no candidate passes, which can forfeit progress near a moving human. The distilled filter is supervised by a CBF-QP whose solution is unique and continuous, and at deployment it sees only RGB images, so it is free of these dependencies. It also removes the need to build and maintain a 3D scene representation at deployment, since it is amortized into training.

![](images/20ce732d72a8cda4c193cf7a86467c40ecefb85b6b8096be18fc2cdce40e1adf.jpg)

![](images/c557ab05b4b7ba0fb6aac650adc2936c7feb71108e61b9f1e68517debc693cbc.jpg)  
Fig. 3: Reproducing the teacher’s action-space filtering boundary. The top row shows the robot’s observations in simulation, and the bottom row shows the corresponding action-space boundaries. The color indicates the teacher CBF constraint margin, with the dashed gray curve denoting its zero-level boundary. The green curve denotes the student’s intervention boundary. The results show that the RGB-only student’s boundary closely matches the teacher’s boundary.

## D. Filtering Boundary Reproduction by the Student

We further examine whether the student reproduces the teacher’s filtering boundary in the action space, as shown in Fig. 3. The teacher boundary is given by the zero level set of the active CBF constraint margin, min<sub>i</sub> $\left( L _ { f } h _ { i } + L _ { g } h _ { i } { \bf u } + \alpha h _ { i } \right) = 0$ , which separates actions that satisfy the teacher’s CBF constraints from those requiring intervention. We define student’s intervention boundary from the magnitude of its action correction. An action is considered unmodified when $\left\| \mathbf { u } _ { \mathrm { s a f e } } - \mathbf { u } _ { \mathrm { n o m } } \right\| _ { 2 } < \epsilon .$ , with ϵ = 0.05, and the corresponding ϵ-level set defines the student intervention boundary. Fig. 3 compares these boundaries at two time steps from the same simulated scenario. Despite having no access to the privileged state or explicit CBF constraints, the learned student boundary closely follows the teacher’s constraint boundary as the scene changes. This indicates that the student distills not only individual safe-action labels, but also the local intervention structure induced by the privileged CBF teacher. Specifically, in the left snapshot, the nearby obstacle shapes the boundary to reject a range of right-turning actions. In the right snapshot, the obstacle and pedestrian jointly constrain both steering directions, leaving only decelerating actions within the admissible region.

TABLE II: Effect of the uncertainty-aware margin. (w/o margin): the teacher without the uncertainty-aware margin on the human’s velocity, and our filter distilled from its labels.
<table><tr><td>Safety filter</td><td>SR↑ (%)</td><td>CFR↑ (%)</td><td>Failure cases↓ S/D/TO (%)</td></tr><tr><td>None</td><td>48.9</td><td>53.0</td><td>25.4/21.6/4.1</td></tr><tr><td>Teacher</td><td>76.8</td><td>91.5</td><td>1.2/7.3/14.7</td></tr><tr><td>Teacher (w/o margin)</td><td>74.6</td><td>87.4</td><td>0.6/12.1/12.8</td></tr><tr><td>Ours</td><td>74.0</td><td>86.1</td><td>3.1/10.8/12.1</td></tr><tr><td>Ours (w/o margin)</td><td>73.6</td><td>82.8</td><td>1.9/15.3/9.2</td></tr></table>

TABLE III: Effect of dataset aggregation and augmentation. Mixed: The student is trained on the union of the datasets collected under all three nominal policies rather than its own nominal’s. Aug.: Action augmentation of the CBF-QP labels.
<table><tr><td>Mixed</td><td>Aug. SR↑ (%)</td><td>CFR↑ (%)</td><td>Failure cases↓ S/D/TO (%)</td></tr><tr><td>Teacher</td><td>76.8</td><td>91.5</td><td>1.2/7.3/14.7</td></tr><tr><td> $^ -$ </td><td>72.2</td><td>83.6</td><td>4.7/11.7/11.4</td></tr><tr><td> $^ -$   $\checkmark$ </td><td>69.7</td><td>81.6</td><td>4.7/13.8/11.9</td></tr><tr><td> $\checkmark$   $^ -$ </td><td>70.7</td><td>83.9</td><td>4.4/11.7/13.2</td></tr><tr><td> $\checkmark$   $\checkmark$ </td><td>74.0</td><td>86.1</td><td>3.1/10.8/12.1</td></tr></table>

## E. Ablation Study

1) Effect of the uncertainty-aware margin: To analyze how accounting for uncertainty in the human’s velocity affects the teacher and the distilled student, we ablate the uncertainty-aware margin from the teacher. The ablated teacher evaluates the barrier at the estimated human velocity alone, without the uncertainty-aware margin, and we distill a student from its labels. Table II reports results averaged over all nominal policies and scenes.

The margin mainly changes where failures occur. Removing it raises human collisions from 7.3% to 12.1% for the teacher and from 10.8% to 15.3% for the student, while static collisions and timeouts drop slightly. Without the margin, the barrier trusts a stale velocity estimate, so a human who changes pace or heading after leaving the field of view can close the gap before the filter reacts. The student reproduces this behavior without ever observing the human’s velocity or its covariance, which indicates that the uncertainty-aware tightening margin is recoverable from the RGB history and is worth its modest cost in conservatism.

2) Ablation on dataset aggregation and augmentation: We ablate two ingredients of the distillation dataset: aggregating the rollouts of all three nominal policies rather than training a separate student per nominal, and action augmentation of the teacher labels. We train each variant and evaluate the filters in the simulation using same conditions and trials as in Table I. Table III shows that combining both raises the mean success and collision-free rate over all scenes and nominal policies to 74.0% and 86.1%, close to the teacher’s 76.8% and 91.5%, whereas either ingredient alone does not improve over the per-nominal, non-augmented student. This indicates that the two are complementary. Augmentation broadens the coverage of the nominal action space, and aggregation supplies the diverse visited states, yielding a single filter that transfers across nominal policies.

TABLE IV: Real-world navigation results. Each nominal–filter–scene combination is evaluated over 10 trials.
<table><tr><td rowspan="2"></td><td rowspan="2">Nominal Safety filter</td><td colspan="5">Outside</td><td colspan="5">Lobby</td><td colspan="5">Library</td></tr><tr><td>SR↑ (%)</td><td>CFR↑ (%)</td><td>m↑ (m)</td><td>T (s)</td><td>Failure cases↓ S/D/TO (%)</td><td>SR↑ (%)</td><td>CFR↑ (%)</td><td>m↑ (m)</td><td>T (s)</td><td>Failure cases↓ S/D/TO (%)</td><td>SR↑ (%)</td><td>CFR↑ (%)</td><td>m↑ (m)</td><td>T (s)</td><td>Failure cases↓ S/D/TO (%)</td></tr><tr><td rowspan="2">IL</td><td>None</td><td>30</td><td>40</td><td>0.13</td><td>41.1</td><td>50/10/10</td><td>70</td><td>70</td><td>0.32</td><td>26.0</td><td>0/30/0</td><td>50</td><td>50</td><td>0.20</td><td>33.4</td><td>0/50/0</td></tr><tr><td>Ours</td><td>50</td><td>80</td><td>0.07</td><td>36.8</td><td>20/0/30</td><td>90</td><td>90</td><td>0.41</td><td>28.1</td><td>0/10/0</td><td>80</td><td>80</td><td>0.34</td><td>29.0</td><td>0/20/0</td></tr><tr><td rowspan="2">RL</td><td>None</td><td>50</td><td>50</td><td>0.18</td><td>34.9</td><td>20/30/0</td><td>20</td><td>20</td><td>0.05</td><td>43.3</td><td>50/30/0</td><td>50</td><td>50</td><td>0.13</td><td>32.8</td><td>10/40/0</td></tr><tr><td>Ours</td><td>80</td><td>80</td><td>0.30</td><td>22.8</td><td>10/10/0</td><td>60</td><td>80</td><td>0.24</td><td>33.9</td><td>0/20/20</td><td>70</td><td>70</td><td>0.33</td><td>30.0</td><td>0/30/0</td></tr><tr><td rowspan="2">ViNT</td><td>None</td><td>40</td><td>40</td><td>0.16</td><td>35.3</td><td>20/40/0</td><td>20</td><td>30</td><td>0.10</td><td>42.1</td><td>50/20/10</td><td>40</td><td>40</td><td>0.23</td><td>33.3</td><td>20/40/0</td></tr><tr><td>Ours</td><td>70</td><td>80</td><td>0.27</td><td>23.4</td><td>20/0/10</td><td>50</td><td>70</td><td>0.31</td><td>32.3</td><td>20/10/20</td><td>70</td><td>80</td><td>0.34</td><td>25.6</td><td>20/0/10</td></tr></table>

![](images/1d520c5e411910ba148393d278f1db0fb81d8cd286396201fd0a40e87d3ef122.jpg)

![](images/b554f3960123b466599553b7a937751dea04cb7d60b241fd39c64d0a154f8dec.jpg)

![](images/df124da1b137b86a318271166ec7a3996358e27a0aa178ece974963d808d806e.jpg)  
(a) Library

![](images/eda99ea5aa396c922f6dd40f2afdeb494470c5cb54a084e7fbc871acc7552486.jpg)  
(b) Lobby

![](images/a74b63a05277c8188fcc734e718cf223cd095d2fd26f37999ff71e194a2e5ba7.jpg)  
Fig. 4: Safety under aggressive forward commands. The nominal controller continuously commands forward motion with zero angular velocity, while the RGB-only safety filter modifies the commands in response to obstacles. (a) Library: the filter stops the robot to yield to a moving human (left) and steers left to avoid a static obstacle (right). (b) Lobby: the filter steers right to avoid the wall (left) and stops to wait for a moving human to pass (right).

## VII. HARDWARE EXPERIMENTS

To further validate our RGB-only safety filter in real world dynamic scenarios, we conduct hardware experiments across the target environments and diverse nominal policies. We assess the resulting safety improvement by measuring navigation performance and analyzing failure cases, and by examining the filter’s response to aggressive nominal inputs.

## A. Hardware Platform

We use a differential wheeled robot for real-world experiments. The robot is equipped with a forward-facing ZED2 camera for RGB observations and an NVIDIA Jetson Orin NX for onboard inference. The safety filter runs at 9.2 ms per inference on the onboard computer. For nominal policies that require a relative goal position, we obtain it from wheel odometry, which is not provided to the safety filter.

## B. Navigation Results

We evaluate navigation performance over 10 episodes for each combination of nominal policy, filter setting, and environment, resulting in a total of 180 episodes. Five humans who do not appear in the training data participate as dynamic obstacles. Table IV shows that our RGB-only safety filter improves success rates and collision-free rates, and increases clearance from obstacles across nominal policies, which is consistent with the simulation results. The failure case analysis reveals that the proposed filter effectively reduces collisions with both static and dynamic obstacles in most cases, while it occasionally increases timeout failures, as the filter reduces speed in densely occupied regions. Fig. 1 shows a qualitative example in which the filtered RL policy detours around a dynamic obstacle, whereas the nominal policy alone collides with it. These results indicate that a safety filter distilled entirely in simulation transfers to the real world and improves the safety of RGB-only navigation policies.

## C. Safety Under Aggressive Nominal Inputs

To examine the behavior of the safety filter under aggressive commands, we apply a persistent forward nominal input without obstacle avoidance. The target linear velocity is fixed at $v _ { \mathrm { t a r g e t } } = 0 . 7 5 ~ \mathrm { m / s } .$ , and the nominal acceleration is computed as $a _ { \mathrm { n o m } } = ( v _ { \mathrm { t a r g e t } } - v _ { \mathrm { c u r } } ) / \Delta t$ with $\Delta t = 0 . 1 \mathrm { s } ,$ while the nominal angular velocity is fixed at $\omega _ { \mathrm { n o m } } = 0 .$ Thus, the nominal command drives the robot straight ahead regardless of surrounding obstacles. Fig. 4 shows that the proposed filter overrides these commands when necessary. In the Library, the robot slows down and waits for a crossing human, while steering away from a static obstacle instead of continuing straight. In the Lobby, the filter similarly turns away from the wall and stops to yield to a moving human. These results demonstrate that our filter can generate appropriate braking and steering interventions even under persistently unsafe nominal commands.

## VIII. CONCLUSION

In this paper, we presented an RGB-only safety filter for visual navigation in dynamic environments. We used a photorealistic simulator of the target environment to provide privileged state that is unavailable at real-world deployment to a teacher CBF filter, and designed the teacher to account for static obstacles and human velocity uncertainty while keeping its decisions inferable from the student’s RGB observations. Distilling this teacher into a student yields a safety filter that requires no robot localization, obstacle state estimation, or 3D scene representation at deployment. Simulation and real-world experiments show that the distilled filter improves the success and collision-free rate of diverse nominal policies and enlarges the clearance to both static obstacles and the human. In future work we aim to scale training across scenes toward a single RGB-only safety filter that transfers to unseen environments and nominal policies.

## REFERENCES

[1] A. Loquercio, E. Kaufmann, R. Ranftl, A. Dosovitskiy, V. Koltun, and D. Scaramuzza, “Deep drone racing: From simulation to reality with domain randomization,” IEEE Transactions on Robotics, vol. 36, no. 1, pp. 1–14, 2020.

[2] S. Yoo, Y. Jang, D. Kim, Y. Han, S. Jung, and H. J. Kim, “Ready-go: Real-to-sim dynamic 3d gaussian splatting simulation for environmentspecific visual navigation with moving obstacles,” IEEE Robotics and Automation Letters, vol. 11, no. 8, pp. 10 058–10 065, 2026.

[3] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine, “ViNT: A foundation model for visual navigation,” in 7th Annual Conference on Robot Learning, 2023. [Online]. Available: https://arxiv.org/abs/2306.14846

[4] M. Harms, M. Kulkarni, N. Khedekar, M. Jacquet, and K. Alexis, “Neural control barrier functions for safe navigation,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024, pp. 10 415–10 422.

[5] L. Lindemann, M. Cleaveland, G. Shim, and G. J. Pappas, “Safe planning in dynamic environments using conformal prediction,” IEEE Robotics and Automation Letters, vol. 8, no. 8, pp. 5116–5123, 2023.

[6] K. J. Strawn, N. Ayanian, and L. Lindemann, “Conformal predictive safety filter for rl controllers in dynamic environments,” IEEE Robotics and Automation Letters, vol. 8, no. 11, pp. 7833–7840, 2023.

[7] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in 2019 18th European Control Conference (ECC), 2019, pp. 3420– 3431.

[8] M. Tong, C. Dawson, and C. Fan, “Enforcing safety for vision-based controllers via control barrier functions and neural radiance fields,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

[9] T. Chen, A. Swann, J. Yu, O. Shorinwa, R. Murai, M. K. III, and M. Schwager, “Safer-splat: Safety with control barrier functions in online gaussian splatting maps,” 2024.

[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in ECCV, 2020.

[11] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[12] V. Tolani, S. Bansal, A. Faust, and C. Tomlin, “Visual navigation among humans with optimal control as a supervisor,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2288–2295, 2021.

[13] R. Bonatti, R. Madaan, V. Vineet, S. Scherer, and A. Kapoor, “Learning visuomotor policies for aerial navigation using cross-modal representations,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020, pp. 1637–1644.

[14] S. Yoo, S. Jung, Y. Lee, D. Shim, and H. J. Kim, “Mono-camera-only target chasing for a drone in a dense environment by cross-modal learning,” IEEE Robotics and Automation Letters, vol. 9, no. 8, pp. 7254–7261, 2024.

[15] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, “GNM: A General Navigation Model to Drive Any Robot,” in International Conference on Robotics and Automation (ICRA), 2023. [Online]. Available: https://arxiv.org/abs/2210.03370

[16] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “Nomad: Goal masked diffusion policies for navigation and exploration,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 63–70.

[17] S. Zhu, L. Mou, D. Li, B. Ye, R. Huang, and H. Zhao, “Vrrobo: A real-to-sim-to-real framework for visual robot navigation and locomotion,” IEEE Robotics and Automation Letters, vol. 10, no. 8, pp. 7875–7882, 2025.

[18] J. Low, M. Adang, J. Yu, K. Nagami, and M. Schwager, “Sous vide: Cooking visual drone navigation policies in a gaussian splatting vacuum,” IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 5122–5129, 2025.

[19] Z. Xie, Z. Liu, Z. Peng, W. Wu, and B. Zhou, “Vid2sim: Realistic and interactive simulation from video for urban navigation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 1581–1591.

[20] H. Abdi, G. Raja, and R. Ghabcheloo, “Safe control using visionbased control barrier function (v-cbf),” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 782–788.

[21] M. De Sa, P. Kotaru, and K. Sreenath, “Point cloud-based control barrier function regression for safe and efficient vision-based control,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 366–372.

[22] V. Vapnik and A. Vashist, “A new learning paradigm: Learning using privileged information,” Neural networks, vol. 22, no. 5-6, pp. 544– 557, 2009.

[23] D. Chen, B. Zhou, V. Koltun, and P. Krahenb ¨ uhl, “Learning by¨ cheating,” in Conference on robot learning. PMLR, 2020, pp. 66–75.

[24] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,” in Robotics: Science and Systems, 2021.

[25] G. Monaci, M. Aractingi, and T. Silander, “Dipcan: Distilling privileged information for crowd-aware navigation.” in Robotics: Science and Systems, 2022.

[26] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, “Control barrier function based quadratic programs for safety critical systems,” IEEE transactions on automatic control, vol. 62, no. 8, pp. 3861–3876, 2016.

[27] H. K. Park, T. Kim, and D. Panagou, “Beyond collision cones: Dynamic obstacle avoidance for nonholonomic robots via dynamic parabolic control barrier functions,” in IEEE International Conference on Robotics and Automation (ICRA), 2026.

[28] M. Li, Z. Sun, Z. Liao, and S. Weiland, “Moving obstacle collision avoidance via chance-constrained model predictive control with control barrier function,” International Journal of Robust and Nonlinear Control, vol. 36, no. 14, pp. 6762–6774, 2026. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.70624

[29] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” Aug. 2018, arXiv:1801.01290 [cs.LG].

[30] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021.

[31] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022.