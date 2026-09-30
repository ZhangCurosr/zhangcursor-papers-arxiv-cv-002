# DYNAMICHOI: COUPLED DYNAMICS FOR PHYSICS-AWARE HOI RECONSTRUCTION

A PREPRINT

Wenliang Guo Zhanbo Huang Yu Kong Michigan State University

## ABSTRACT

We study hand-object interaction (HOI) reconstruction from monocular RGB videos, where partial observations can produce visually plausible yet mechanically inconsistent trajectories. Existing methods mainly enforce visual and geometric agreement, leaving the underlying interaction dynamics insufficiently constrained. We propose DYNAMICHOI, a physics-aware HOI reconstruction framework combining geometry-grounded diffusion refinement with coupled hand-object dynamics. Geometry spatially grounds visual evidence for trajectory refinement, while articulated inverse dynamics and Newton-Euler dynamics derive hand generalized forces and object wrenches for dynamics-level supervision. We further couple hand and object dynamics through contact-force transfer and re cover active hand actuation as an interaction-level physical quantity. We formulate its empirical magnitude distribution into a probabilistic prior that penalizes unlikely actuation and suppresses mechanically implausible reconstructed motion. Experiments on three HOI datasets show consistent improvements in both hand and object reconstruction. The reconstructed trajectories further benefit downstream applications including hand world-model generation and robotic manipulation learning, demonstrating the value of physics-aware HOI modeling beyond reconstruction. Project page: https://wenliangguo.github.io/HOI-Reconstruction-Page/.

## 1 Introduction

Hands are the primary interface through which humans perform everyday hand-object interactions (HOIs). Reconstructing these interactions from monocular videos into temporal 3D hand-object mesh trajectories is essential for immersive technologies (Zhou et al., 2023; Feng et al., 2020), embodied learning (Qin et al., 2022; Chen et al., 2025), and world modeling (Li et al., 2026). Monocular videos provide only partial observations under occlusion and viewpoint ambiguity, allowing multiple 3D trajectories to explain similar visual evidence and making reliable reconstruction challenging.

Recent work has substantially advanced HOI reconstruction. Optimization-based methods jointly fit hand, object, and scene representations (Huang et al., 2022; Ye et al., 2023; Fan et al., 2024), while feed-forward approaches exploit contact, occlusion, and learned interaction priors (Chen et al., 2022; Wang et al., 2025; Aboukhadra et al., 2026; Wang et al., 2026). Recent methods have also extended HOI reconstruction to egocentric videos (Zhang et al., 2025a; Fu et al., 2026; Ye et al., 2026). Despite this progress, two challenges remain. First, geometry is not fully exploited to ground visual representations, limiting ambiguity resolution under occlusion and restricted viewpoints (Ye et al., 2026). More importantly, reconstruction is typically supervised at the trajectory level, which encourages geometric accuracy but does not explicitly constrain the dynamics implied by the estimated motion. For example, a reconstructed object may suddenly translate or rotate during manipulation without a corresponding force being exerted by the hand. Dynamics provides complementary supervision by constraining the forces and torques required to produce reconstructed motion.

Introducing such dynamical constraints remains difficult because physical quantities such as forces are not directly observable from videos. Zhang et al. (2024) and Ismayilzada et al. (2026) pioneered articulated hand dynamics by deriving generalized forces from reconstructed trajectories through inverse dynamics, providing physical supervision beyond geometric states. However, these formulations characterize only the hand and do not account for the dynamics of the manipulated object. In HOI, object motion is governed by its own rigid-body dynamics, while the hand and object are mechanically coupled through contact: active hand actuation produces hand motion and generates interaction forces that drive the object. Therefore, modeling hand dynamics alone or treating the two bodies independently leaves this mechanical dependency unconstrained. Physics-aware HOI reconstruction thus requires both hand and object dynamics as well as their physical coupling through contact to form a unified dynamically-constrained system.

To address these challenges, we propose DYNAMICHOI, a physics-aware HOI reconstruction framework that integrates geometry-grounded visual refinement with coupled hand-object dynamics. To better resolve visual ambiguity, we encode the hand-object geometry and use it to spatially ground visual evidence within a diffusion-based iterative refinement, providing geometry-aligned guidance for trajectory estimation. To further constrain the motion itself, we derive hand generalized forces through articulated inverse dynamics (Zhang et al., 2025b; Ismayilzada et al., 2026) and rigid-object wrenches through Newton-Euler dynamics (Luh et al., 1980; Khalil, 2011; Hu et al., 2022; Lutter, 2023). Aligning these motion-implied physical quantities between reconstructed and ground-truth trajectories introduce dynamics-level supervision beyond trajectory alignment.

We further couple the hand and object through contact-force transfer (Hu et al., 2022; Nakajima et al., 2022). Active hand actuation jointly accounts for the generalized force required by hand motion and the contact reaction induced by driving the object, providing a shared physical explanation for both trajectories. Our key insight is that active actuation during everyday manipulation follows a characteristic statistical distribution (Tanghe et al., 2019). We estimate this empirical distribution and impose its negative log-likelihood as a probabilistic regularizer, penalizing reconstructions that induce low-probability actuation. Together, the hand dynamics, object dynamics, and actuation prior form a coupled dynamical system that provides physically grounded supervision for HOI reconstruction.

Experiments across three HOI reconstruction datasets (Banerjee et al., 2025; Chao et al., 2021; Hampali et al., 2020) demonstrate that DYNAMICHOI consistently improves both hand and object reconstruction. We further use the reconstructed trajectories to condition hand world-model video generation (Li et al., 2026) and train dexterous robotic manipulation (Chen et al., 2025). Improvements in both applications show that physics-aware reconstruction provides more useful motion signals beyond reconstruction accuracy. In summary, our contributions are threefold:

• We present a geometry-grounded HOI reconstruction framework that uses 3D geometry to spatially ground visual evidence during diffusion refinement for more accurate trajectory estimation.

• We introduce dynamics-level supervision by aligning hand generalized forces and object wrenches between reconstructed and ground-truth trajectories to encourage physically consistent motion.

• We couple hand-object dynamics through contact-force transfer and regularize active hand actuation with a probabilistic prior to suppress reconstructions that induce implausible actuation.

## 2 Related Work

Hand and object reconstruction. Image-based hand models such as HaMeR (Pavlakos et al., 2024) and WiLoR (Potamias et al., 2025) learn strong visual priors for mesh regression, while video-based methods such as HaWoR (Zhang et al., 2025a) and Dyn-HaMR (Yu et al., 2025) further exploit temporal context and camera motion to recover global hand trajectories. For objects, FoundationPose (Wen et al., 2024) conditions 6D pose estimation on object geometry, whereas ForeHOI (Chen et al., 2026) combines visual evidence with 3D shape priors to reconstruct objects under hand occlusion. However, these methods do not explicitly leverage joint 3D hand-object geometry to organize visual evidence. In contrast, our framework grounds visual cues in the 3D HOI geometry within diffusion refinement, improving reconstruction under occlusion and monocular ambiguity.

Hand-object interaction reconstruction. Joint HOI reconstruction reduces monocular ambiguity by exploiting dependencies between the hand and object. Early methods jointly estimate or refine their geometry and poses using contact and non-penetration constraints (Hasson et al., 2019; Grady et al., 2021; Chen et al., 2022). Video-based approaches further incorporate differentiable rendering, implicit representations, and learned interaction priors (Ye et al., 2023; Huang et al., 2022; Fan et al., 2024), while recent methods improve generalization to limited viewpoints, unseen categories, articulated objects, and egocentric settings (Wang et al., 2025; Aboukhadra et al., 2026; Wang et al., 2026; Fu et al., 2026; Ye et al., 2026). These methods primarily enforce visual and geometric consistency of the motion, while our proposed framework focuses on modeling articulated-hand and rigid-object dynamics to constrain the physical requirements underlying their joint trajectories.

Physics-aware HOI modeling. Physics-aware methods improve mechanical plausibility through explicit dynamics or learned physical constraints. Physical Interaction estimates contact forces that satisfy rigid-object dynamics from RGB-D observations (Hu et al., 2022), while Physics-Aware HOI Denoising refines hand motion given an accurate object trajectory (Luo et al., 2024). Other hand-centric approaches introduce phase-dependent physical constraints (Zhang et al., 2025b) or probabilistic articulated dynamics (Ismayilzada et al., 2026). Nevertheless, existing methods do not jointly model hand dynamics, object dynamics, and their coupling through contact. We address this gap by coupling the two dynamical systems through contact forces and regularizing the resulting active hand actuation with a probabilistic prior.

![](images/dae51a47523bd0f11da489f2b5a755c8d70987790b064f80b6798d3ce22e1b50.jpg)  
Figure 1: Framework overview. Framewise reconstruction models first estimate the initial trajectory in parametric states, which is then refined progressively by a diffusion-based refinement network, and finally decoded to HOI mesh trajectory.

## 3 Methodology

## 3.1 Problem Formulation

Given an RGB video $I _ { 1 : T } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ and the canonical mesh $\mathcal { M } ^ { o }$ of the interacting object<sup>1</sup>, HOI reconstruction aims to recover the 3D hand and object mesh trajectories over $T$ frames. Following prior work (Ismayilzada et al., 2026; Ye et al., 2026), we represent the reconstruction using compact parametric states and then deterministically decode them into meshes. Specifically, a reconstruction model $\mathcal { G } _ { \Theta }$ with learnable parameters Θ predicts the joint hand-object trajectory during inference

$$
\begin{array} { r } { \mathbf { x } _ { 1 : T } = \left\{ \left( \mathbf { x } _ { t } ^ { h } , \mathbf { x } _ { t } ^ { o } \right) \right\} _ { t = 1 } ^ { T } = \mathcal { G } _ { \Theta } \left( I _ { 1 : T } , \mathcal { M } ^ { o } \right) , } \end{array}\tag{1}
$$

where $\mathbf { x } _ { t } ^ { h }$ and $\mathbf { x } _ { t } ^ { o }$ denote the hand and object parametric states at frame $t ,$ respectively. The hand state $\mathbf { x } _ { t } ^ { h } = ( \rho _ { t } ^ { h } , \mathbf { t } _ { t } ^ { h } , \beta ^ { h } )$ is represented by MANO parameters (Romero et al., 2017), where $\rho _ { t } ^ { h } \in \mathbb { R } ^ { \bar { 1 } 6 \times 6 }$ contains the 6D rotations of the global hand orientation and the 15 articulated MANO joints, $\mathbf { t } _ { t } ^ { h } \in \mathbb { R } ^ { 3 }$ denotes the global translation, and $\beta ^ { h } \in \mathbb { R } ^ { 1 0 }$ represents the hand shape and remains fixed throughout the video. The estimated states $\mathbf { x } _ { 1 : T } ^ { h }$ are decoded through the MANO layer (Romero et al., 2017) to obtain the hand mesh trajectory $\mathbf { \mathcal { M } } _ { 1 : T } ^ { h }$ . The object state $\mathbf { x } _ { t } ^ { o } = ( \mathbf { R } _ { t } ^ { o } , \mathbf { t } _ { t } ^ { o } )$ is represented by its rigid pose, where $\mathbf { R } _ { t } ^ { o } \in S O ( 3 )$ and $\mathbf { t } _ { t } ^ { o } \in \mathbb { R } ^ { 3 }$ denote its rotation and translation, respectively. The object mesh at frame t is obtained through the rigid transformation $\mathcal { M } _ { t } ^ { o } = \mathbf { R } _ { t } ^ { o } \mathcal { M } ^ { o } + \mathbf { t } _ { t } ^ { o }$

Trajectory-level supervision alone does not guarantee physically plausible motion. We therefore further constrain the hand and object dynamics individually, as well as their physical coupling during interaction. Let $\hat { \mathbf { x } } _ { 1 : T }$ denote the ground-truth trajectory. We formulate the reconstruction objective as

$$
\mathbf { x } _ { 1 : T } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { x } _ { 1 : T } \in \mathcal { X } } \left[ \mathcal { C } _ { t } \left( \mathbf { x } _ { 1 : T } , \hat { \mathbf { x } } _ { 1 : T } \right) + \lambda _ { d } \mathcal { C } _ { d } \left( \mathbf { x } _ { 1 : T } , \hat { \mathbf { x } } _ { 1 : T } \right) + \lambda _ { a } \mathcal { C } _ { a } \left( \mathbf { x } _ { 1 : T } ^ { h } , \mathbf { x } _ { 1 : T } ^ { o } \right) \right] ,\tag{2}
$$

where $\mathcal { X }$ denotes the admissible trajectory space. The three constraints capture complementary aspects of reconstruction: $\mathcal { C } _ { t }$ enforces trajectory alignment, $\mathcal { C } _ { d }$ independently constrains the articulated-hand and rigid-object dynamics, and $\scriptstyle { { \mathcal { C } } _ { a } }$ constrains their coupled dynamics during interaction. The weights $\lambda _ { d }$ and $\lambda _ { a }$ balance the corresponding physical constraints. Eq. 2 only provides a conceptual decomposition of constraints; in practice, they are instantiated as training objectives for the model.

## 3.2 Physics-aware HOI Reconstruction

Figure 1 shows the overall framework of DYNAMICHOI, consisting of frozen reconstruction estimators and a learnable trajectory refinement network we design. Specifically, a frozen hand estimator ${ \mathcal { E } } ^ { h }$ (Potamias et al., 2025) takes the frame $I _ { t }$ as input and predicts hand parameters $\mathbf { y } _ { t } ^ { h }$ , while a frozen object estimator ${ \mathcal { E } } ^ { o }$ (Wen et al., 2024) takes $I _ { t }$ and the object template $\mathcal { M } ^ { o }$ as input and predicts the object parameters $\mathbf { y } _ { t } ^ { o } .$ , resulting an initial estimate of the hand and object trajectory $\mathbf { y } _ { 1 : T } = ( \mathbf { y } _ { 1 : T } ^ { h } , \mathbf { y } _ { 1 : T } ^ { o } )$ . Because these estimations rely on framewise visual evidence, their predictions can suffer from temporal and physical inconsistency, particularly under monocular ambiguity and occlusion. We refine the initial estimates (Zhang et al., 2025b,a; Ismayilzada et al., 2026) by modeling visual-geometric evidence and physical characteristics through an iterative diffusion process.

## 3.2.1 Refinement Network

We formulate the refinement within a diffusion process (Ho et al., 2020; Peebles and Xie, 2023) that enables progressive correction of temporal incoherence through iterative denoising. Specifically, given the initial reconstruction $\mathbf { y } _ { 1 : T }$ randomly-sampled diffusion steps n and other inputs, the refinement network $f _ { \Theta }$ estimates the temporally consistent trajectory by

$$
\mathbf { x } _ { 1 : T } ^ { ( n ) } = f _ { \Theta } \left( \mathbf { y } _ { 1 : T } ; n , I _ { 1 : T } , \mathcal { M } ^ { o } \right) .\tag{3}
$$

$f _ { \Theta }$ contains a forward noise-injection process and a learnable denoising process, which progressively refines the initial trajectory by visual-evidence grounding with 3D geometry and hand-object dependency modeling. During training, we first sample a diffusion step n and construct the corresponding noisy trajectory $\mathbf { z } _ { n , 1 : T }$ through the forward noise-injection process:

$$
\mathbf { z } _ { n , 1 : T } = \hat { \mathbf { x } } _ { 1 : T } + \eta _ { n } ^ { 2 } \left( \mathbf { y } _ { 1 : T } - \hat { \mathbf { x } } _ { 1 : T } \right) + \eta _ { n } \kappa \epsilon ,\tag{4}
$$

where $\eta _ { n }$ controls the diffusion level, κ determines the stochastic noise magnitude, and $\mathbf { \epsilon } \in \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ denotes Gaussian noise. Similar to $\mathbf { y } _ { 1 : T }$ , we represent the noisy trajectory as $\mathbf { z } _ { n , 1 : T } = \{ ( \mathbf { z } _ { n , t } ^ { h } , \mathbf { z } _ { n , t } ^ { o } ) \} _ { t = 1 } ^ { T }$ , comprising the noisy hand trajectory $\mathbf { z } _ { n , 1 : T } ^ { h }$ and the noisy object trajectory $\mathbf { z } _ { n , 1 : T } ^ { o }$ . The denoising process is performed through the following modules:

Geometric encoding. We perform geometric encoding in mesh space to expose the explicit 3D structure hidden in the compact state parameters, enabling the denoising process to reason about hand-object spatial relationships and local surface geometry. The intermediate noisy states $\mathbf { z } _ { n , 1 : T }$ are first converted into meshes by applying MANO layers to hand parameters and the rigid transformation to object parameters, which are subsequently encoded into latent features through separate geometry encoders. We further employ temporal self-attention layers to aggregate temporal information across the frame sequence, which produces the geometric feature sequences $\mathbf { h } _ { n , 1 : T } ^ { \mathrm { g e o } }$ and $\mathbf { \widetilde { o } } _ { n , 1 : T } ^ { \mathrm { g e o } }$

Visual grounding. The geometric features capture explicit 3D structure but lack appearance information from the observed video. We therefore ground them with spatially corresponding visual evidence. Specifically, a frozen visual encoder (Oquab et al., 2024) extracts a patch-level feature map $\mathbf { F } _ { t }$ <sub>t</sub> from each frame $I _ { t } .$ . To establish spatial correspondence, we derive 3D geometric anchors from the initial hand and object estimates $\mathbf { y } _ { t } ^ { h }$ and $\mathbf { y } _ { t } ^ { o }$ , using MANO joints for the hand and pose-transformed canonical keypoints for the object. These anchors are projected onto the image to retrieve and aggregate their corresponding visual features $\zeta _ { t } ^ { h }$ and $\zeta _ { t } ^ { o }$

$$
\zeta _ { t } ^ { m } = \frac { 1 } { | \mathscr { Q } _ { t } ^ { m } | } \sum _ { \mathbf { p } \in \mathscr { Q } _ { t } ^ { m } } \mathbf { F } _ { t } ( \Pi _ { t } ( \mathbf { p } ) ) , \qquad m \in \{ h , o \} ,\tag{5}
$$

where $\boldsymbol { \mathcal { Q } } _ { t } ^ { h }$ and $\mathcal { Q } _ { t } ^ { o }$ denote the sets of visible hand and object anchors, respectively, and $\Pi _ { t }$ denotes the camera projection function. The resulting geometry-aligned visual features are fused with their corresponding geometric features through residual projections: $\bar { \mathbf { h } } _ { n , t } ^ { \mathrm { v i s } } = \mathbf { h } _ { n , t } ^ { \mathrm { g e o } } + \breve { \phi } ^ { h } ( \zeta _ { t } ^ { h } )$ and ${ \bf o } _ { n , t } ^ { \mathrm { v i s } } = { \bf o } _ { n , t } ^ { \mathrm { g e o } } + \phi ^ { o } ( \zeta _ { t } ^ { o } )$ , where $\phi ^ { h }$ and $\phi ^ { o }$ are learnable projection layers. This produces the visually grounded feature sequences $\mathbf { h } _ { n , 1 : T } ^ { \mathrm { v i s } }$ and ${ \bf o } _ { n , 1 : T } ^ { \mathrm { v i s } }$

Interaction modeling. Although the hand and object features are visually grounded, they are still encoded separately, which does not explicitly capture their mutual dependencies. We therefore concatenate $\mathbf { \dot { h } } _ { n , 1 : T } ^ { \mathrm { v i s } }$ and ${ \bf o } _ { n , 1 : T } ^ { \mathrm { v i s } }$ and jointly process them with spatial-temporal self-attention layers (Dosovitskiy et al., 2021) to model HOI across time. The output tokens are then separated into the interaction-aware hand and object features according to the original token positions, with the first $T$ tokens forming $\tilde { \mathbf { h } } _ { n , 1 : T }$ and the remaining T tokens forming $\tilde { \mathbf { o } } _ { n , 1 : T }$ . Separate prediction heads regress residual updates to the initial hand and object states as $\mathbf { x } _ { t } ^ { h , ( n ) } = \mathbf { y } _ { t } ^ { h } + g ^ { h } ( \tilde { \mathbf { h } } _ { n , t } )$ and $\mathbf { x } _ { t } ^ { o , ( n ) } = \mathbf { y } _ { t } ^ { o } + g ^ { o } ( \tilde { \mathbf { o } } _ { n , t } )$ , where $g ^ { h }$ and $g ^ { o }$ are corresponding MLP projectors. We concatenate them as $\mathbf { x } _ { t } ^ { ( n ) } = [ \mathbf { x } _ { t } ^ { h , ( n ) } , \mathbf { x } _ { t } ^ { o , ( n ) } ]$ and collect over frames to obtain $\mathbf { x } _ { 1 : T } ^ { ( n ) } = \{ \mathbf { x } _ { t } ^ { ( n ) } \} _ { t = 1 } ^ { T }$ as the output of $f _ { \Theta }$

Diffusion objective. We instantiate the trajectory-level constraint $\mathcal { C } _ { t }$ in Eq. 2 with the diffusion objective ${ \mathcal { L } } _ { \mathrm { d i f f } }$ Specifically, we train the diffusion-based refinement network $f _ { \Theta }$ to recover the ground-truth trajectory from noisy trajectories sampled at different diffusion levels. We align the estimated trajectory $\mathbf { x } _ { 1 : T } ^ { ( n ) }$ with the ground-truth $\hat { \mathbf { x } } _ { 1 : T }$ by

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { n , \epsilon } \left[ d _ { x } \left( \hat { \mathbf { x } } _ { 1 : T } , \mathbf { x } _ { 1 : T } ^ { ( n ) } \right) \right] , } \end{array}\tag{6}
$$

where $d _ { x }$ measures the trajectory discrepancy in rotation and translation spaces. This objective enables $f _ { \Theta }$ to recover accurate HOI trajectories through joint visual-geometric representation learning.

## 3.2.2 Hand and Object Dynamics

Although the diffusion objective enables trajectory reconstruction, it does not explicitly account for the underlying dynamics that produce the hand and object motion, which may lead to physically implausible behaviors such as abnormal accelerations or abrupt rotations. To promote physical consistency beyond visual plausibility, we derive the hand and object dynamics from their trajectories and align the resulting physical quantities between the reconstructed and ground-truth motions.

Hand dynamics. Following previous work (Zhang et al., 2024; Ismayilzada et al., 2026), we regard the human hand as an articulated body and derive the generalized force required to produce its motion. The 16 hand rotations $\rho _ { t } ^ { h }$ are converted into Euler-ZXY angles $\phi _ { t } ^ { h } \in \mathbb { R } ^ { 4 8 }$ , corresponding to the global orientation and 15 articulated MANO joints. Together with the global translation $\mathbf { t } _ { t } ^ { h }$ , we define the generalized coordinates as $\mathbf { q } _ { t } ^ { h } = [ ( \mathbf { t } _ { t } ^ { h } ) ^ { \top } , ( \phi _ { t } ^ { h } ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { 5 \tilde { 1 } }$ . We use temporal finite differences between adjacent frames to estimate the generalized velocity $\dot { \mathbf { q } } _ { t } ^ { h }$ and acceleration $\ddot { \mathbf q } _ { t } ^ { h }$ Given these kinematic quantities, inverse dynamics is used to recover the resultant generalized force:

$$
\pmb { \tau } _ { t } ^ { h } = M ^ { h } ( \mathbf { q } _ { t } ^ { h } ) \ddot { \mathbf { q } } _ { t } ^ { h } + C ^ { h } ( \mathbf { q } _ { t } ^ { h } , \dot { \mathbf { q } } _ { t } ^ { h } ) \dot { \mathbf { q } } _ { t } ^ { h } + \mathbf { g } _ { \mathrm { g r a v } } ^ { h } ( \mathbf { q } _ { t } ^ { h } ) ,\tag{7}
$$

where $M ^ { h } , C ^ { h }$ , and $\mathbf { g } _ { \mathrm { g r a v } } ^ { h }$ denote the hand mass matrix, Coriolis and centrifugal terms, and gravitational term, respectively. Applying $\mathrm { E q . 7 }$ over the frame sequence yields the generalized-force trajectory $\{ \tau _ { t } ^ { h } \} _ { t = 1 } ^ { T }$ , which reveals the dynamic requirements for driving hand movement.

Object dynamics. We introduce the Newton–Euler formalism (Luh et al., 1980; Khalil, 2011; Hu et al., 2022; Lutter, 2023), a fundamental framework for inverse dynamics in robotics and rigid-body control, into HOI reconstruction to relate object motion to its underlying physical requirements. Specifically, as discussed in Sect. 3.1, we model the object as a rigid body with pose $\mathbf { x } _ { t } ^ { o } = \mathbf { \bar { ( R _ { t } ^ { o } , \mathbf { t } _ { t } ^ { o } ) } }$ . We compute its center-of-mass linear acceleration $\mathbf { a } _ { t } ^ { o }$ from the translation trajectory $\{ \mathbf { t } _ { t } ^ { o } \} _ { t = 1 } ^ { T }$ , and derive its angular velocity $\omega _ { t } ^ { o }$ and angular acceleration $\dot { \omega } _ { t } ^ { o }$ from the rotation trajectory $\{ \mathbf { R } _ { t } ^ { o } \} _ { t = 1 } ^ { T }$ using temporal finite differences between adjacent frames. The physical effort required to produce this motion is thus represented by its object wrench:

$$
\mathbf { w } _ { t } ^ { o } = \left[ \mathbf { f } _ { t } ^ { o } \right] = \left[ \mathbf { \Gamma } _ { \mathbf { I } _ { t } ^ { o } } \mathbf { \dot { \omega } } _ { t } ^ { m ^ { o } } ( \mathbf { a } _ { t } ^ { o } - \mathbf { g } ) \mathbf { \Gamma } _ { \mathbf { \Omega } } \right] ,\tag{8}
$$

where $\mathbf { f } _ { t } ^ { o }$ and $\tau _ { t } ^ { o }$ denote the non-gravitational external force and torque required to drive the translational and rotational motion of the object, respectively. g is the gravitational acceleration, $\mathbf { I } _ { t } ^ { o }$ is the inertia tensor in the world frame, and $m ^ { o }$ denotes the object mass. We use the object mass provided by the dataset when available; otherwise, we employ a large language model (OpenAI, 2026) to estimate it. This formulation maps the object-pose trajectory to its wrench profile $\{ \mathbf { w } _ { t } ^ { o } \} _ { t = 1 } ^ { T }$ , providing an explicit dynamics-based characterization of object motion. Additional theoretical and implementation details are provided in the supplementary material.

Dynamics objective. The hand generalized forces and object wrenches derived above provide a dynamics-level description of the motion. We instantiate the dynamics constraint $\mathcal { C } _ { d }$ in Eq. 2 by aligning the mechanical quantities implied by the reconstructed trajectory with those derived from the ground-truth trajectory. For frame $t ,$ we compute the hand generalized force and object wrench $( \tau _ { t } ^ { h , ( n ) } , \mathbf { w } _ { t } ^ { o , ( n ) } )$ ) at diffusion step n from the predicted trajectory, and $( \hat { \tau } _ { t } ^ { h } , \hat { \mathbf { w } } _ { t } ^ { o } )$ from the ground-truth trajectory. The resulting dynamics objective is

![](images/f69293b51a29fcd10909621d70ade663c20418e794e22512f40518c33626a9e2.jpg)

![](images/866a004e4fbe346aaa23fb4006b20da916bc209124b4aaf7b465f8bab19ebb82.jpg)

![](images/ea96088e909eb5975624187d6d2ddc9e53b21a5045806c5a31cea28d14b60ecb.jpg)

![](images/97e5981e5ffa8ecc1efe5de655bd683760bd0d3cce3edacf7317fe2c47acf060.jpg)  
Figure 2: Coupled HOI dynamics. The bottom-right plot shows the empirical distribution of standardized active actuation $z = ( a _ { t , d } - \mu _ { d } ) / \sigma _ { d }$ for an example DoF. The logarithmic density axis highlights the heavy-tailed behavior, which is empirically better captured by the Student-t distribution than the Gaussian distribution.

$$
\mathcal { L } _ { \mathrm { d y n } } = \sum _ { t = 1 } ^ { T } \left. \pmb { \tau } _ { t } ^ { h , ( n ) } - \hat { \pmb { \tau } } _ { t } ^ { h } \right. _ { 2 } ^ { 2 } + \sum _ { t = 1 } ^ { T } \left. \mathbf { w } _ { t } ^ { o , ( n ) } - \hat { \mathbf { w } } _ { t } ^ { o } \right. _ { 2 } ^ { 2 } .\tag{9}
$$

This objective complements the diffusion supervision by enforcing consistency in the mechanical quantities underlying the motion, thereby discouraging dynamically inconsistent trajectories.

## 3.2.3 Coupled HOI Dynamics

Even if individually plausible, hand and object dynamics may still be mutually inconsistent, since physically valid HOI requires the same interaction to explain both motions (Hu et al., 2022; Nakajima et al., 2022). Active hand actuation provides such a joint physical description by accounting for both the generalized force required to produce hand motion and the interaction force exerted to drive object motion. Therefore, we use it to couple dynamics and regularize it with a probabilistic prior (Tanghe et al., 2019) to encourage physically plausible joint motion.

To recover active hand actuation, we first infer the contact forces exerted by the hand to drive the object motion. Since we focus on in-air manipulation, where the object is primarily driven by hand contact, we do not explicitly model external supports such as table contact. We approximate distributed contact using $K = 6$ anchors on the hand mesh, including the five fingertips and palm center. Given the contact geometry at frame t, we infer $\mathbf { f } _ { t } \in \mathbb { R } ^ { 3 K }$ whose combined effect explains the object wrench $\mathbf { w } _ { t } ^ { o }$ . These contact forces are mapped to the hand generalized coordinates through $J _ { c , t } ^ { \top } \mathbf { f } _ { t }$ and combined with the generalized force $\tau _ { t } ^ { h }$ required by the hand motion, yielding the active hand actuation $\mathbf { a } _ { t } \in \mathbb { R } ^ { 5 1 }$

$$
\mathbf { a } _ { t } = \pmb { \tau } _ { t } ^ { h } + J _ { c , t } ^ { \top } \mathbf { f } _ { t } ,\tag{10}
$$

which jointly accounts for the physical effort underlying the hand and object motion, thereby coupling their dynamics.   
More details of contact-force inference are provided in the supplementary material.

We assume that hand forces during everyday object manipulation typically remain within a characteristic range, and thus active actuation should follow a structured distribution rather than vary arbitrarily. We estimate this distribution from framewise actuation computed on ground-truth training trajectories. As shown in Figure 2, most samples concentrate within a narrow range, while larger values occasionally occur during dynamic interactions such as lifting or releasing objects, yielding a heavy-tailed distribution well captured by a Student-t distribution empirically. Accordingly, we instantiate the coupled-dynamics constraint $\scriptstyle { { \mathcal { C } } _ { a } }$ in Eq. 2 by fitting a separate Student-t distribution to each DoF and using its negative log-likelihood to penalize unlikely actuation. For coordinate d $\in \{ 1 , \ldots , 5 1 \}$ } with fitted parameters $( \mu _ { d } , \sigma _ { d } , \nu _ { d } )$ , the active-actuation objective is

$$
\mathcal { L } _ { \mathrm { a c t } } = \sum _ { t , d } \alpha _ { t } \hat { c } _ { t } \left[ \log \sigma _ { d } + \frac { \nu _ { d } + 1 } { 2 } \log \left( 1 + \frac { ( a _ { t , d } - \mu _ { d } ) ^ { 2 } } { \nu _ { d } \sigma _ { d } ^ { 2 } } \right) \right] ,\tag{11}
$$

where $\boldsymbol { a } _ { t , d }$ is the d-th DoF of $\mathbf { a } _ { t } , \hat { c } _ { t } \in \{ 0 , 1 \}$ indicates ground-truth hand-object contact, and $\alpha _ { t }$ emphasizes dynamically active frames. Since $\mathbf { a } _ { t }$ jointly depends on hand motion, object motion, and contact geometry, this objective provides a learned prior over their coupled dynamics while remaining tolerant to legitimate high-actuation events.

## 3.3 Training Objective and Inference

We jointly optimize three objectives to train the model in an end-to-end fashion, which is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d y n } } + \lambda _ { \mathrm { a } } \mathcal { L } _ { \mathrm { a c t } } , } \end{array}\tag{12}
$$

where ${ \mathcal { L } } _ { \mathrm { d i f f } }$ supervises trajectory reconstruction, $\mathcal { L } _ { \mathrm { d y n } }$ constrains the individual hand and object dynamics, and $\mathcal { L } _ { \mathrm { a c t } }$ further enforces their coupled mechanics through contact. Together, these objectives realize our formulated problem in Eq. 2, which encourages reconstructed trajectories that are both geometrically accurate and physically consistent. The physical quantities are only calculated during training, thus no additional computational cost is introduced at inference.

## 4 Experiments

## 4.1 Experimental Setting

Datasets and protocols. We conduct experiments on an egocentric video dataset HOT3D (Banerjee et al., 2025), and two allocentric video datasets HO3D-v2 (Hampali et al., 2020) and DexYCB (Chao et al., 2021). For HOT3D, we follow the Aria video split (Ye et al., 2026), using 1,311 training and 147 evaluation clips, each with 150 frames. For HO3D-v2, we use the official split with 55 training sequences and 13 evaluation sequences (11,524 frames). For DexYCB, we follow the official protocol and treat each camera view as a clip, yielding 6,400/320/1,280 clips for training/validation/testing.

Evaluation metrics. For hand reconstruction, HOT3D reports W-MPJPE (W; cm), WA-MPJPE (WA; cm), PA-MPJPE (PA; mm), RTE (%), and ACCEL (mm/frame<sup>2</sup>), where W and WA align trajectories using the first two frames and the full sequence, respectively. HO3D-v2 reports PA-MPJPE for both joints and vertices (PA/PA-V; mm), their corresponding AUC/AUC-V (in [0, 1]), F@5/F@15, and ACCEL (mm/frame<sup>2</sup>). DexYCB reports wrist-relative MPJPE, PA-MPJPE (PA; mm), AUC (in [0, 1]), and ACCEL (mm/frame<sup>2</sup>). Object pose is evaluated using ADD(-S) AUC (%) at thresholds of 0.1/0.3 m, 0.1d recall (%), ACCEL (mm/frame<sup>2</sup>) and 5<sup>◦</sup>/5 cm success (%). More evaluation and implementation details are provided in the supplementary material.

## 4.2 Experimental Results Analysis

Model Comparison. We compare DYNAMICHOI with existing reconstruction methods, including WiLoR (Potamias et al., 2025), FoundationPose (Wen et al., 2024), HaWoR (Zhang et al., 2025a), WHOLE (Ye et al., 2026), AMVUR (Jiang et al., 2023), HaMeR (Pavlakos et al., 2024), Hamba (Dong et al., 2024), Deformer (Fu et al., 2023), and PAD-Hand (Ismayilzada et al., 2026). Table 1 shows that DYNAMICHOI achieves the best performance on most metrics while consistently improving the input hand and object estimates. The gains are particularly large on the egocentric HOT3D dataset, indicating that our model remains effective under moving viewpoints. Importantly, modeling object dynamics and coupling them with hand dynamics improves not only object reconstruction but also hand motion. On the DexYCB dataset, DYNAMICHOI outperforms PAD-Hand, which explicitly models hand dynamics, suggesting that object dynamics provide complementary constraints on hand reconstruction through physical interaction. This benefit is further reflected in the reduction in acceleration error (ACCEL) across all datasets for both hand and object, indicating that our proposed physical constraints improve not only trajectory accuracy but also the dynamica consistency of hand-object motion.

Module Ablation. As shown in Table 2, for hand reconstruction, removing visual grounding causes the largest degradation on egocentric HOT3D, indicating that geometry-aligned visual evidence is particularly important for correcting trajectories in a moving camera. In contrast, the interaction module brings a minor effect, because dense hand-object co-motion can already be captured by temporal refinement. On fixed-view HO3D, removing interaction causes the largest degradation, as persistent occlusion and partial visibility make explicit hand-object motion coupling more important, while geometric encoding is less critical under a stable camera. For object reconstruction on HOT3D, interaction is the most influential module, showing that hand motion provides strong constraints for recovering object pose. Geometric encoding has a smaller effect, because the rigid object structure is already well preserved, leaving the other two modules as main cues for resolving its pose changes.

Loss Ablation. As shown in Table 3, adding $\mathcal { L } _ { \mathrm { d y n } }$ (2nd vs. 4th rows) substantially improves both hand and object reconstruction on HOT3D, confirming the benefit of explicitly constraining their dynamics. Comparing the last two rows further shows that $\mathcal { L } _ { \mathrm { a c t } }$ provides a smaller but consistent additional gain. We attribute this limited improvement partly to the large variability of video content and to errors in the forces estimated by the Euler-Lagrange and Newton-Euler equations, which may propagate into the inferred actuation distribution. More accurate force estimation and actuation modeling remain promising directions for future work. On fixed-view HO3D, both physical losses yield modest gains, suggesting that dynamics supervision is particularly beneficial for egocentric reconstruction, where camera motion and trajectory ambiguity make purely visual constraints insufficient.

Table 1: Comparison on HOT3D, HO3D-v2, and DexYCB datasets. Bold / underline denote the best / second-best reconstruction methods. † denotes our reproduction, and unmarked values are their paper-reported. “–” denotes metrics that are either inapplicable to the method or unavailable due to missing reports and unreleased code.  
(a) Hand-object reconstruction on HOT3D dataset.
<table><tr><td></td><td colspan="5">Hand</td><td colspan="6">Object</td><td></td></tr><tr><td>Method</td><td>W↓</td><td>WA↓</td><td>PA↓</td><td>ACCEL↓</td><td>RTE↓</td><td>ADD@.1↑</td><td>ADD@.3↑</td><td>ADD-S@.1↑</td><td>ADD-S@.3↑</td><td>0.1d↑</td><td>ACCEL↓</td><td>5°5cm↑</td></tr><tr><td>Input estimates (y)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WiLoR†</td><td>11.74</td><td>3.70</td><td>6.52</td><td>16.22</td><td>75.16</td><td></td><td></td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>FoundationPose†</td><td></td><td></td><td></td><td>一</td><td>一</td><td>12.8</td><td>38.9</td><td>25.3</td><td>52.8</td><td>6.5</td><td>70.15</td><td>6.7</td></tr><tr><td>Reconstruction methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HaWoR†</td><td>11.73</td><td>3.92</td><td>8.38</td><td>12.90</td><td>73.69</td><td></td><td></td><td></td><td></td><td>一</td><td></td><td>一</td></tr><tr><td>WHOLE</td><td>10.41</td><td>3.26</td><td>6.67</td><td></td><td></td><td></td><td>51.1</td><td></td><td>69.9</td><td>一</td><td></td><td></td></tr><tr><td>DYNAMICHOI</td><td>5.26</td><td>1.83</td><td>4.15</td><td>1.68</td><td>73.49</td><td>25.8</td><td>65.4</td><td>48.2</td><td>78.4</td><td>7.1</td><td>5.57</td><td>9.8</td></tr></table>

(b) Hand reconstruction on HO3D-v2 dataset.
<table><tr><td>Method</td><td></td><td></td><td></td><td>PA↓ AUC↑ PA-V↓ AUC-V↑ F@5↑ F@15↑ ACCEL↓</td><td></td><td></td><td></td></tr><tr><td colspan="8">Input estimates (y)</td></tr><tr><td>WiLoR†</td><td>7.67</td><td>0.847</td><td>7.66</td><td>0.846</td><td>0.647</td><td>0.984</td><td>4.12</td></tr><tr><td colspan="8">Reconstruction methods</td></tr><tr><td>AMVUR</td><td>8.30</td><td>0.835</td><td>8.20</td><td>0.836</td><td>0.608</td><td>0.965</td><td></td></tr><tr><td>HaMeR</td><td>7.70</td><td>0.846</td><td>7.90</td><td>0.841</td><td>0.635</td><td>0.980</td><td>一</td></tr><tr><td>Hamba</td><td>7.50</td><td>0.850</td><td>7.70</td><td>0.846</td><td>0.648</td><td>0.982</td><td></td></tr><tr><td>DYNAMICHOI 7.45</td><td></td><td>0.851</td><td>7.49</td><td>0.850</td><td>0.656</td><td>0.985</td><td>2.77</td></tr></table>

<table><tr><td colspan="4">(c) Hand reconstruction on Dex YCB dataset.</td></tr><tr><td>Method MPJPE↓</td><td colspan="3">PA↓ AUC↑ ACCEL↓</td></tr><tr><td colspan="4">Input estimates (y)</td></tr><tr><td>WiLoR† 11.52</td><td>5.33</td><td>0.893</td><td>6.88</td></tr><tr><td colspan="4">Reconstruction methods</td></tr><tr><td>Deformer</td><td>13.64 5.22</td><td></td><td>6.77</td></tr><tr><td>HaWoR†</td><td>11.09 5.32</td><td>0.894</td><td>3.88</td></tr><tr><td>PAD-Hand</td><td>10.56 4.63</td><td></td><td>3.34</td></tr><tr><td>DYNAMICHOI</td><td>9.51 4.68</td><td>0.906</td><td>3.27</td></tr></table>

Table 2: Module ablation. Bold / underline denote the best / second-best configurations.  
(a) Ablations for hand-object reconstruction on the HOT3D dataset.
<table><tr><td></td><td colspan="4">Hand</td><td colspan="4">Object</td></tr><tr><td>Model</td><td>W↓</td><td>WA↓</td><td>PA↓</td><td>ACCEL↓</td><td>ADD@0.1↑ ADD-S@0.1↑</td><td></td><td>0.1d↑</td><td>5°5cm↑</td></tr><tr><td>w/o Geometric</td><td>5.83</td><td>1.92</td><td>4.33</td><td>1.71</td><td>24.4</td><td>46.0</td><td>7.1</td><td>9.5</td></tr><tr><td>w/o Grounding</td><td>6.65</td><td>2.40</td><td>4.45</td><td>1.74</td><td>21.5</td><td>43.0</td><td>5.3</td><td>8.1</td></tr><tr><td>w/o Interaction</td><td>5.72</td><td>1.90</td><td>4.24</td><td>1.73</td><td>21.2</td><td>40.0</td><td>5.5</td><td>8.0</td></tr><tr><td>Full model</td><td>5.26</td><td>1.83</td><td>4.15</td><td>1.68</td><td>25.8</td><td>48.2</td><td>7.1</td><td>9.8</td></tr></table>

(b) Ablations for hand reconstruction on the HO3D dataset.
<table><tr><td>Model</td><td>PA↓</td><td>PA-V↓</td><td>F@5↑</td></tr><tr><td>w/o Geometric</td><td>7.52</td><td>7.51</td><td>0.655</td></tr><tr><td>w/o Grounding</td><td>7.55</td><td>7.51</td><td>0.654</td></tr><tr><td>w/o Interaction</td><td>7.59</td><td>7.52</td><td>0.654</td></tr><tr><td>Full model</td><td>7.45</td><td>7.49</td><td>0.656</td></tr></table>

Actuation Distribution. As shown in Table 4, we compare actuation priors using the average NLL per valid training coordinate and PA/ADD@0.1 reconstruction metrics. We can observe that an inappropriate prior can degrade reconstruction: the Gaussian prior underperforms removing ${ \mathcal { L } } _ { \mathrm { a c t } } .$ , supporting the heavy-tailed nature of the inferred actuation. Fitting $\nu _ { d }$ separately for each DoF consistently outperforms fixed-ν Student-t priors and achieves the best overall performance. Notably, better likelihood fit does not necessarily imply better reconstruction: although ν = 2 yields lower NLL than $\nu = 1 0 ,$ , it performs worse on object reconstruction, suggesting that likelihood fit alone does not determine the effectiveness of the prior as a reconstruction constraint.

## 4.3 Downstream Applications

Beyond reconstruction, we evaluate whether physics-aware hand trajectories from DYNAMICHOI benefit downstream embodied tasks. For dexterous manipulation, we follow ViViDex (Chen et al., 2025) and retarget trajectories reconstructed on the DexYCB dataset for robot policy learning. For egocentric world modeling, we follow EgoHOI (Li et al.,

Table 3: Loss ablation. Bold / underline denote the best / second-best configurations.  
(a) Ablations for hand-object reconstruction on the HOT3D dataset.
<table><tr><td colspan="3">Training objective</td><td colspan="4">Hand</td><td colspan="4">Object</td></tr><tr><td>Ldiff</td><td> $\mathcal { L } _ { \mathrm { d y n } }$ </td><td> $\mathcal { L } _ { \mathrm { a c t } }$ </td><td>W↓</td><td></td><td>WA↓ PA↓</td><td></td><td>ACCEL↓ ADD@0.1↑ ADD-S@0.1↑ 0.1d↑ 5°5cm↑</td><td></td><td></td><td></td></tr><tr><td>√</td><td></td><td></td><td>8.27</td><td>2.82</td><td>5.16</td><td>6.10</td><td>17.8</td><td>33.6</td><td>4.7</td><td>7.4</td></tr><tr><td>√</td><td></td><td></td><td>7.77</td><td>2.31</td><td>5.06</td><td>5.01</td><td>20.0</td><td>37.9</td><td>5.3</td><td>8.5</td></tr><tr><td>√</td><td>√</td><td></td><td>6.12</td><td>1.91</td><td>4.22</td><td>2.78</td><td>24.9</td><td>47.4</td><td>6.7</td><td>9.8</td></tr><tr><td>√</td><td>√</td><td>」</td><td>5.26</td><td>1.83</td><td>4.15</td><td>1.68</td><td>25.8</td><td>48.2</td><td>7.1</td><td>9.8</td></tr></table>

(b) Ablations for hand on HO3D.
<table><tr><td colspan="3">Training objective</td><td colspan="2">Hand</td></tr><tr><td>Ldiff</td><td> $\mathcal { L } _ { \mathrm { d y n } }$ </td><td>Lact</td><td></td><td>PA↓ PA-V↓ F@5↑</td></tr><tr><td>√</td><td></td><td></td><td>7.55</td><td>7.59 0.650</td></tr><tr><td>√</td><td></td><td>√</td><td>7.48</td><td>7.50 0.651</td></tr><tr><td>√</td><td>√</td><td></td><td>7.46</td><td>7.50 0.651</td></tr><tr><td>√</td><td>√</td><td>√</td><td>7.45</td><td>7.49 0.656</td></tr></table>

![](images/1e1ec8209c9c9b9a7b6585d70a4323ce11d3d7db15b9423ffa64eb886af2c457.jpg)  
Figure 3: Visualization for reconstruction.

2026) and use trajectories reconstructed on the HOT3D dataset as hand-action conditions. In both tasks, only the input hand trajectories vary across reconstruction methods, while all other components remain fixed. We use reconstructed trajectories from models trained on the corresponding dataset without task-specific retraining.

Dexterous manipulation. We evaluate eight held-out DexYCB relocate tasks, training one PPO policy (Schulman et al., 2017) for each object and trajectory source and evaluating 100 episodes per task. As shown in Figure 4a, downstream success provides a task-level measure of reconstruction quality. Methods with comparable reconstruction accuracy, such as WiLoR and HaWoR (see Table 1c), also achieve similar success rates. In contrast, DYNAMICHOI reduces MPJPE by 17.4% over WiLoR, and this advantage is amplified in embodied learning, increasing the success rate from 25.0% to 74.9%. This suggests that improvements in physically coherent hand reconstruction can translate into substantially larger gains when the trajectories are used for robotic learning.

Egocentric world modeling. We evaluate both action fidelity and generated hand appearance. mLPIPS measures perceptual similarity within the GT hand region, while MPJPE measures the kinematic accuracy of the generated hand. Figure 4b shows that DYNAMICHOI performs best among reconstructed action sources on all metrics. Since EgoHOI conditions generation on framewise MANO renderings aligned with image pixels, more accurate trajectories provide better spatial guidance for where and how the hand should appear. Thus, improved reconstruction not only yields more faithful generated motion but also translates into better visual quality in the hand region.

Table 4: Ablation of different distribution choices for modeling active hand actuation on HOT3D dataset.
<table><tr><td>Distribution</td><td>NLL↓</td><td>PA↓</td><td>ADD@.1 ↑</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { a c t } }$ </td><td>一</td><td>4.22</td><td>24.9</td></tr><tr><td>Gaussian</td><td>-0.39</td><td>4.54</td><td>24.2</td></tr><tr><td>Laplace</td><td>-0.11</td><td>4.17</td><td>24.9</td></tr><tr><td>Cauchy  $( \nu { = } 1 )$ </td><td>-1.45</td><td>4.20</td><td>24.7</td></tr><tr><td colspan="4">Student-t distributions</td></tr><tr><td>Fixed  $\nu { = } 1 0$ </td><td> $- 0 . 7 0$ </td><td>4.17</td><td>25.6</td></tr><tr><td>Fixed  $\nu { = } 5$ </td><td>-0.96</td><td>4.16</td><td>25.2</td></tr><tr><td>Fixed  $\nu { = } 2$ </td><td>-1.15</td><td>4.16</td><td>25.3</td></tr><tr><td>Fitted  $\nu _ { d }$ </td><td>-1.59</td><td>4.15</td><td>25.8</td></tr></table>

![](images/ae46caf583390709925b77cc483166ecbc4e73274a4c5c0210fe7b4ff5ccd3f9.jpg)  
(a) Dexterous manipulation.

![](images/55f7e75e2c51a64ccde9553218997f7319f3003a79da6aad3d900900ae8e5826.jpg)  
(b) Egocentric video generation.  
Figure 4: Downstream applications with different hand-trajectory sources.

## 5 Conclusion

We presented DYNAMICHOI, a physics-aware HOI reconstruction framework that combines geometry-grounded trajectory refinement with coupled hand-object dynamics. We model articulated-hand and rigid-object dynamics and couple them through contact, with a learned probabilistic prior over active hand actuation. Experiments on egocentric and allocentric datasets demonstrate improved reconstruction, while downstream gains in dexterous manipulation and egocentric world modeling show the broader value of physically coherent HOI trajectories. Future work will explore more accurate force estimation, richer actuation priors, and unified object representations that generalize beyond known template meshes to unseen objects.

## References

Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, and Didier Stricker. GHOST: Fast category-agnostic hand-object interaction reconstruction from RGB videos using gaussian splatting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026.

Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Yuantao Chen, Jiahao Chang, Chongjie Ye, Chaoran Zhang, Zhaojie Fang, Chenghong Li, and Xiaoguang Han. ForeHOI: Feed-forward 3D object reconstruction from daily hand-object interaction videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8868–8879, 2026.

Zerui Chen, Yana Hasson, Cordelia Schmid, and Ivan Laptev. AlignSDF: Pose-aligned signed distance fields for hand-object reconstruction. In European Conference on Computer Vision (ECCV), 2022.

Zerui Chen, Shizhe Chen, Etienne Arlaud, Ivan Laptev, and Cordelia Schmid. ViViDex: Learning vision-based dexterous manipulation from human videos. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

Haoye Dong, Aviral Chharia, Wenbo Gou, Francisco Vicente Carrasco, and Fernando D De la Torre. Hamba: Singleview 3d hand reconstruction with graph-guided bi-scanning mamba. Advances in Neural Information Processing Systems, 37:2127–2160, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Muhammed Kocabas, Xu Chen, Michael J. Black, and Otmar Hilliges. HOLD: Category-agnostic 3D reconstruction of interacting hands and objects from video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Qi Feng, Hubert PH Shum, and Shigeo Morishima. Resolving hand-object occlusion for mixed reality with joint deep learning and model optimization. Computer Animation and Virtual Worlds, 31(4-5):e1956, 2020.

Hongming Fu, Wenjia Wang, Xiaozhen Qiao, Rolandos Alexandros Potamias, Taku Komura, Shuo Yang, Zheng Liu, and Bo Zhao. EgoGrasp: World-space hand-object interaction estimation from egocentric videos, 2026.

Qichen Fu, Xingyu Liu, Ran Xu, Juan Carlos Niebles, and Kris M Kitani. Deformer: Dynamic fusion transformer for robust hand pose estimation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 23543–23554. IEEE, 2023.

Patrick Grady, Chengcheng Tang, Christopher D. Twigg, Minh Vo, Samarth Brahmbhatt, and Charles C. Kemp. ContactOpt: Optimizing contact to improve grasps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3193–3203. IEEE, 2020.

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11807–11816, 2019.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Haoyu Hu, Xinyu Yi, Hao Zhang, Jun-Hai Yong, and Feng Xu. Physical interaction: Reconstructing hand-object interactions with physics. In SIGGRAPH Asia 2022 Conference Papers. Association for Computing Machinery, 2022. doi: 10.1145/3550469.3555421.

Di Huang, Xiaopeng Ji, Xingyi He, Jiaming Sun, Tong He, Qing Shuai, Wanli Ouyang, and Xiaowei Zhou. Reconstructing hand-held objects from monocular video. In ACM SIGGRAPH Asia Conference Papers, 2022.

Elkhan Ismayilzada, Yufei Zhang, and Zijun Cui. PAD-Hand: Physics-aware diffusion for hand motion recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Zheheng Jiang, Hossein Rahmani, Sue Black, and Bryan M Williams. A probabilistic attention model with occlusionaware texture regression for 3d hand reconstruction from a single rgb image. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 758–768. IEEE, 2023.

Wisama Khalil. Dynamic modeling of robots using newton-euler formulation. In Informatics in Control, Automation and Robotics: Revised and Selected Papersfrom the International Conference on Informatics in Control, Automation and Robotics 2010, pages 3–20. Springer, 2011.

Dayou Li, Lulin Liu, Bangya Liu, Shijie Zhou, Jiu Feng, Ziqi Lu, Minghui Zheng, Chenyu You, and Zhiwen Fan. Egocentric world model for photorealistic hand-object interaction synthesis, 2026.

John YS Luh, Michael W Walker, and Richard PC Paul. On-line computational scheme for mechanical manipulators. Journal ofDynamic Systems, Measurement, and Control, 102(2):69–76, 06 1980. ISSN 0022-0434. doi: 10.1115/1. 3149599. URL https://doi.org/10.1115/1.3149599.

Haowen Luo, Yunze Liu, and Li Yi. Physics-Aware Hand-Object Interaction Denoising . In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2341–2350, Los Alamitos, CA, USA, June 2024. IEEE Computer Society. doi: 10.1109/CVPR52733.2024.00227. URL https://doi.ieeecomputersociety.org/10. 1109/CVPR52733.2024.00227.

Michael Lutter. A differentiable newton–euler algorithm for real-world robotics. In Inductive Biases in Machine Learning for Robotics and Control, pages 9–34. Springer, 2023.

Takayuki Nakajima, Yuki Asami, Yui Endo, Mitsunori Tada, and Naomichi Ogihara. Prediction of anatomically and biomechanically feasible precision grip posture of the human hand based on minimization of muscle effort. Scientific Reports, 12(1):13247, 2022.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/index/gpt-5-6/, July 2026. Model family released July 9, 2026. Accessed: 2026-08-17.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3D with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182. IEEE, 2023.

Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. DexMV: Imitation learning for dexterous manipulation from human videos. In European Conference on Computer Vision (ECCV), 2022.

Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6):245:1–245:17, 2017.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Kevin Tanghe, Maarten Afschrift, Ilse Jonkers, Friedl De Groote, Joris De Schutter, and Erwin Aertbeliën. A probabilistic method to estimate gait kinetics in the absence of ground reaction force measurements. Journal of biomechanics, 96:109327, 2019.

Shibo Wang, Haonan He, Maria Parelli, Christoph Gebhardt, Zicong Fan, and Jie Song. MagicHOI: Leveraging 3D priors for accurate hand-object reconstruction from short monocular video clips. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, and Wangmeng Zuo. ArtHOI: Taming foundation models for monocular 4D reconstruction of hand-articulated-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6D pose estimation and tracking of novel objects. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Yufei Ye, Poorvi Hebbar, Abhinav Gupta, and Shubham Tulsiani. Diffusion-guided reconstruction of everyday handobject interaction clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Yufei Ye, Jiaman Li, Ryan Rong, and C. Karen Liu. WHOLE: World-grounded hand-object lifted from egocentric videos. CVPR Findings, 2026.

Zhengdi Yu, Stefanos Zafeiriou, and Tolga Birdal. Dyn-HaMR: Recovering 4D interacting hand motion from a dynamic camera. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Jinglei Zhang, Jiankang Deng, Chao Ma, and Rolandos Alexandros Potamias. HaWoR: World-space hand motion reconstruction from egocentric videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025a.

Yufei Zhang, Jeffrey O Kephart, Zijun Cui, and Qiang Ji. Physpt: Physics-aware pretrained transformer for estimating human dynamics from monocular videos. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2305–2317. IEEE, 2024.

Yufei Zhang, Zijun Cui, Jeffrey O Kephart, and Qiang Ji. Diffusion-based 3d hand motion recovery with intuitive physics. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7306–7317. IEEE, 2025b.

Kanglei Zhou, Chen Chen, Yue Ma, Zhiying Leng, Hubert PH Shum, Frederick WB Li, and Xiaohui Liang. A mixed reality training system for hand-object interaction in simulated microgravity environments. In 2023 IEEE international symposium on mixed and augmented reality (ISMAR), pages 167–176. IEEE, 2023.