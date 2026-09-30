# FracGen: Learning How Objects Stretch and Tear with Physics-Informed Video Generation

Trong-Tung Nguyen Jiahan Zhang Anand Bhattad

Johns Hopkins University

Project Website: https://fracgen.github.io

Intact Object

![](images/0befe67b3554b410e4a1630259e75be8ab3672849c887d8bb81cf5c7087bdc48.jpg)  
(b) Delay fracture  
Figure 1: Controllable stretching and tearing with FracGen. Given a single object image and specified force, material, and fracture conditions, FracGen generates a video of the object deforming and tearing. Under the same applied force, varying the fracture properties produces earlier tearing (top) or delayed tearing after greater deformation (bottom).

## Abstract

We introduce FracGen, a fracture-aware video generation model that produces plausible, controllable fracture dynamics from a single image of an intact object, conditioned on physics signals. To train FracGen, we build FracSim, a fracture-aware simulation framework that augments material point method (MPM) simulation with a continuum damage model, producing paired fracture videos and dense, pixel-aligned physical fields at no additional cost beyond standard rendering. FracGen leverages these maps in two ways: it is trained to jointly predict them alongside RGB video, encouraging the model to capture physical state rather than surface appearance; and it is supervised with physics-informed losses that encourage consistency among the predicted maps. As a result, FracGen captures distinct material-specific fracture behavior without expensive test-time simulation or per-scene tuning, while offering fine-grained control over where an object tears, how fast the crack propagates, and how much deformation precedes failure. We further introduce a benchmark for evaluating the physical plausibility of generated fracture video, and show through extensive experiments that FracGen outperforms existing video generation baselines in both physical and visual fidelity. Results are best viewed in our project website: https://fracgen.github.io/.

## 1 Introduction

Things break all the time. We drop, tear or snap objects and know what a fracture or damage looks like. Bread breaking like glass or a mug tearing apart like paper looks immediately wrong. Any convincing fracture-generation algorithm must match these physical intuitions — not just the final broken shape, but also how an object deforms, when it starts to fail and how the fracture develops.

A growing body of work shows that modern generative models internalize substantial knowledge of physical dynamics directly from data (Bhattad et al., 2023; Du et al., 2023; Liu et al., 2024b; Wang et al., 2025; Xing et al., 2025; Gillman et al., 2025). In this paper, we ask whether this extends to fracture: can a video generator learn how objects break from limited data and lightweight fine-tuning?

Recent work has begun to condition video generation on physical signals. ForcePrompting (Gillman et al., 2025) shows that generators can respond to push/pull forces, but does not explicitly model damage accumulation or fracture. Physics-based approaches address different parts of this problem. PhysGaussian (Xie et al., 2024) produces physically grounded deformation but lacks a damage model for fracture, while Fracture-GS (Wang et al., 2026) introduces a Collision Material Point Method (Collision-MPM) to simulate fracture under extreme mechanical collisions. These simulation pipelines require a reconstructed scene and configured physical conditions. Our goal is to instead start from a single photograph and specified material and force conditions, and answer by generating “what happens if I pull here until it breaks?”

To this end, we introduce FracGen, a video generation model that produces fracture dynamics for objects under tensile stretch. We focus on stretch-to-tear fracture, where substantial deformation precedes failure, making the evolution before fracture an important part of the generation task. FracGen captures distinct material-specific behaviors, from objects that fail after little stretching to those that undergo large deformation before tearing apart.

Teaching a generator these behaviors requires supervision beyond visual appearance: RGB frames alone do not explicitly describe the mechanical state that leads to fracture. We observe that simulators like PhysGaussian already track rich physical information that is not exposed in their rendered videos. Each particle carries a deformation gradient, a 3 × 3 matrix encoding local deformation, from which strain can be computed and stress obtained through the material’s constitutive law. Particle velocities further describe the local motion. We introduce a damage field that tracks the accumulated failure state of each particle and governs fracture. Rendering these quantities as pixel-aligned maps alongside RGB frames yields dense, physically grounded supervision without additional simulation runs. We call this augmented simulation framework FracSim, and its rendered physical fields FracPhys Maps.

We use these maps in two complementary ways, forming our main technical contributions. First, we extend the video generator to jointly predict FracPhys Maps alongside the fracture RGB video, forcing the model to encode the underlying mechanical state rather than surface appearance alone. These maps also enable us to inspect the physical state underlying the generated video. Second, we introduce physics-informed losses, derived from constitutive modeling, that encourage consistency among the predicted physical channels: the predicted strain, stress, and damage representations are coupled through a latent consistency loss inspired by constitutive modeling and stress degradation. We further introduce a benchmark targeting the evaluation of physical alignment in generated fracture videos, along with extensive experiments comparing FracGen against baseline methods to quantify its effectiveness. In summary, our contributions include:

1. FracSim, a fracture-aware simulation framework that incorporates continuum damage mechanics into MPM, producing paired fracture videos and FracPhys Maps for training.

2. FracGen, a video generation model that jointly predicts fracture RGB video and physical fields, trained with physics-informed losses encouraging consistency between physics maps.

3. A benchmark with new evaluation metrics for evaluating fracture video generation across diverse materials, applied forces, and fracture behaviors.

4. Extensive experiments showing FracGen generates physically plausible, controllable fracture that generalizes across unseen objects, outperforming existing baselines both visually and physically.

## 2 Related Work

Physics-informed 3D Simulation. A recent line of work grounds 3D dynamics in continuum mechanics. PhysGaussian (Xie et al., 2024) couples the Material Point Method (MPM) (Hu et al., 2018) with 3D Gaussians, treating each Gaussian as both a rendering primitive and physical particle, while Gaussian Splashing (Feng et al., 2025) adopts Position-Based Dynamics (Macklin et al., 2016) for cohesive solid-fluid simulation. However, these solvers require manual scene specification and tedious parameter tuning. To tackle this, PhysDreamer (Zhang et al., 2024) optimizes MPM material parameters against generative reference videos using differentiable simulation, whereas DreamPhysics (Huang et al., 2025) optimizes directly via Score Distillation Sampling (SDS) (Poole et al., 2023). Physics3D (Liu et al., 2024a) and OmniPhysGS (Lin et al., 2025) further expand SDS-driven optimization to richer constitutive models, incorporating viscoelastic damping or multi-material mixtures (elastic, plastic, fluid) within a single scene.

Physics-informed Video Generation. With advances in video generation models, recent works seek to inject faithful physical realism through two main strategies. The first conditions generation on explicit control signals: Force Prompting (Gillman et al., 2025) fine-tunes a video diffusion model on synthetic force-annotated data, PhysCtrl (Wang et al., 2025) controls video diffusion using predicted 3D point-trajectories conditioned on forces and material traits, and PhyCo (Narayanan et al., 2026) enables pixel-aligned ControlNet guidance over physical fields (friction, deformation, force) refined by VLM reward optimization. The second strategy enforces physics via high-level reasoning and reward alignment: DiffPhy (Zhang et al., 2025) uses LLM context-builder alongside a MLLM physical critic, while PhyGDPO (Cai et al., 2025) post-trains a video models using physics-aware groupwise preference optimization. However, these methods focus on rigid-body motion or simple deformation, lacking mechanisms for material failure. To the best of our knowledge, FracGen is the first model to generate complex fracture dynamics across the full deformation-to-fracture progression.

## 3 Background

Gaussian–MPM Simulation. 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) represents a scene using anisotropic Gaussian kernels that carry geometry and appearance. PhysGaussian (Xie et al., 2024) couples this representation with the Material Point Method (MPM) (Hu et al., 2018), associating particles with evolving positions $x _ { p } ^ { t } ,$ velocities $v _ { p } ^ { t } ,$ , and deformation gradients $F _ { p } ^ { t }$ . MPM updates these states through transfers between particles and a background grid, while the Gaussians deform with the material to render its evolving appearance. The deformation map ϕ transports a material point from its reference position $x _ { p }$ to $x _ { p } ^ { t } = \phi ( x _ { p } , t )$ . Its gradient, $\boldsymbol { F } _ { p } ^ { t } = \nabla _ { \boldsymbol { x } _ { p } } \phi ( \boldsymbol { x } _ { p } , t )$ , encodes local stretching, compression, rotation, and shearing. The polar decomposition $\pmb { F } _ { p } \doteq \pmb { R } _ { p } \pmb { S } _ { p }$ separates rigid rotation $R _ { p }$ from stretch $S _ { p } ,$ , allowing us to distinguish shape change from rigid motion.

Constitutive Models. Strain measures local deformation relative to the reference configuration, while stress describes the internal forces per unit area transmitted through the material. A constitutive model relates deformation to stress. For hyperelastic materials, a strain energy density $\psi ( F )$ determines the stress response, which can be nonlinear while remaining reversible. In isotropic linear elasticity, Hooke’s law gives ${ \pmb { \sigma } } = \lambda _ { \mathrm { L } } \operatorname { t r } ( \varepsilon ) { \bf { I } } + 2 \mu { \varepsilon }$ , where $\lambda _ { \mathrm { L } }$ and $\mu$ are the Lamé parameters. FracSim computes stress from the material’s strain energy and introduces damage-dependent degradation, while our generator uses the linear-elastic relationship to motivate an approximate consistency loss in latent space.

Particle State Update with MPM. For particle $p$ at simulation time t, let $x _ { p } ^ { t } , \ v _ { p } ^ { t } ,$ , and $F _ { p } ^ { t }$ denote its position, velocity, and deformation gradient, respectively. At each step, MPM transfers particle quantities to a background Eulerian grid, solves the momentum equation, and transfers the results back to update these states. Since each particle carries both appearance and physical state, a single representation supports simulation and rendering. Under a first-order approximation of $\phi ,$ each Gaussian’s center moves from its reference position $\boldsymbol { x } _ { p } ~ \mathrm { t o } ~ \boldsymbol { x } _ { p } ^ { t } .$ , and its reference covariance $A _ { p }$ becomes $A _ { p } ^ { t } = F _ { p } ^ { t } A _ { p } ( F _ { p } ^ { t } ) ^ { \top }$ . Its spherical harmonic (SH) appearance rotates with the particle rotation $R _ { p } ,$ while its opacity $o _ { p }$ remains fixed.

(a) Rendering FracPhys Map with FracSim  
![](images/2f6d4bf2fe9954fa2378e3fc0f82c3e63ef11bd545a3147ff659a60f3f1806ba.jpg)

![](images/fa57e35154877e3cadda483d25bce31430607a3683ee17a918777c36af8cf880.jpg)  
Figure 2: FracSim generates physically plausible fracture dynamics spanning deformation to structural failure. (a) Alongside RGB video, FracSim outputs FracPhys map. (b) Our damage model enables explicit control over fracture timing under fixed forces and material properties.

Conditional Video Generation We build on a latent video generator (Wan et al., 2025) that encodes an RGB video V as $z _ { 1 } = \mathcal { E } ( \mathbf { V } )$ using a 3D VAE. Under flow matching (Lipman et al., 2023), a transformer v<sub>θ</sub> learns to transport Gaussian noise $z _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ to video latents along the linear path $z _ { t } = t z _ { 1 } + ( 1 - t ) z _ { 0 }$ . The training objective is

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , z _ { 0 } , z _ { 1 } , c } \left[ \left| v _ { \theta } ( z _ { t } , t , c ) - ( z _ { 1 } - z _ { 0 } ) \right| ^ { 2 } \right] ,\tag{1}
$$

where $t \sim \mathcal { U } ( 0 , 1 )$ and c denotes conditioning inputs. We extend this formulation to jointly generate fracture videos and physical maps under specified material and force conditions.

## 4 FracSim and FracGen

We discuss in §4.1 how we design FracSim, which outputs dynamic fracture physics maps (FracPhys) alongside fracture videos, forming corpus for training FracGen, as detailed in §4.2.

## 4.1 FracSim: Simulating Fracture Video and Rendering FracPhys Maps

We integrate a continuum damage model into MPM to generate dynamic physics maps alongside RGB frames.   
We refer to these maps as FracPhys maps and discuss how to extract them below.

Flow. Tensile load is prescribed via particle velocities (referred as flow), making ${ \pmb v } _ { p } \in \mathbb { R } ^ { 3 }$ the most direct kinematic signal in the simulation. Rather than estimating this field from rendered frame via optical flow, we directly extract it from the MPM state to preserve exact direction and magnitude.

Strain. We require a strain measure that vanishes under rigid rotation yet remains meaningful at large deformation. The Green-Lagrange strain tensor satisfies both requirements:

$$
\begin{array} { r } { \pmb { \varepsilon } _ { p } = \frac { 1 } { 2 } \left( \pmb { F } _ { p } ^ { \top } \pmb { F } _ { p } - \pmb { I } \right) = \frac { 1 } { 2 } \left( \pmb { S } _ { p } ^ { \top } \pmb { S } _ { p } - \pmb { I } \right) , } \end{array}\tag{2}
$$

where the polar decomposition $F _ { p } = R _ { p } S _ { p }$ isolates stretch $S _ { p }$ from rotation $R _ { p } ,$ , which cancels exactly. The result is rotation-invariant, zero at rest, and nonzero only when genuine deformation occurs. We record its Frobenius norm $\| \varepsilon _ { p } \| _ { F }$ as a scalar per particle.

Stress. We compute the Kirchhoff stress tensor from the strain energy density $\psi ( \pmb { F } _ { p } )$ :

$$
\pmb { \tau } _ { p } = \frac { \partial \psi } { \partial \pmb { F } _ { p } } \pmb { F } _ { p } ^ { \top } .\tag{3}
$$

Damage degrades this tensor before it is used in the MPM momentum update. We then convert the degraded tensor to Cauchy stress and extract its von Mises magnitude for rendering, as described below.

Damage. We augment PhysGaussian with a scalar damage variable $d _ { p } \in [ 0 , 1 ]$ , where 0 denotes intact material and 1 denotes complete failure. To represent this degradation at the particle level, we use a kinematic proxy based on the maximum principal stretch $\lambda _ { \operatorname* { m a x } } ( F _ { p } )$ , the largest singular value of $F _ { p }$ , following (Patnaik & Semperlotti, 2021):

$$
d _ { p , \mathrm { r a w } } ^ { t } = \mathrm { c l a m p } \bigg ( \frac { \lambda _ { \mathrm { m a x } } \big ( F _ { p } ^ { t } \big ) - \lambda _ { \mathrm { o n s e t } } } { \lambda _ { \mathrm { c r i t } } - \lambda _ { \mathrm { o n s e t } } } , 0 , 1 \bigg ) , \qquad d _ { p } ^ { t } = \operatorname* { m a x } \big ( d _ { p } ^ { t - 1 } , d _ { p , \mathrm { r a w } } ^ { t } \big ) .\tag{4}
$$

We initialize $d _ { p } ^ { 0 } = 0$ and require $\lambda _ { \mathrm { c r i t } } > \lambda _ { \mathrm { o n s e t } } \geq 1$ . Under increasing stretch, damage begins at $\lambda _ { \mathrm { o n s e t } }$ and grows linearly to complete failure at $\lambda _ { \mathrm { c r i t } }$ . The running maximum preserves accumulated damage during unloading, enforcing irreversibility. Adjusting these thresholds controls fracture timing under fixed loading and material parameters (Fig. 2b).

Damage feeds back into the simulation by degrading the stress tensor:

$$
\begin{array} { r } { \tilde { \tau } _ { p } = ( 1 - d _ { p } ) \tau _ { p } . } \end{array}\tag{5}
$$

This degraded tensor is used in the MPM momentum update. As $d _ { p }$ approaches one, the particle progressively loses its ability to sustain load, allowing separation. For the stress map, we convert the degraded Kirchhoff stress to Cauchy stress, $\tilde { \sigma } _ { p } = \tilde { \tau } _ { p } / J _ { p } ,$ , where $J _ { p } = \operatorname* { d e t } ( \boldsymbol { F } _ { p } )$ , and compute its equivalently degraded von Mises scalar $\tilde { \sigma } _ { \mathrm { v M } , p } .$ . Damage is the sole softening mechanism; we do not model plastic flow (more discussion in $\ S \mathrm { A } . 4 )$

Rendering FracPhys Maps. As discussed in §3, 3DGS composites pixel color using appearance term $\operatorname { S H } ( l _ { p } ; \mathcal { C } _ { p } )$ . We replace this with any per-particle quantity $q _ { p } \mathrm { : }$

$$
M _ { q } = \sum _ { p \in \mathcal { P } } \alpha _ { p } q _ { p } \prod _ { j = 1 } ^ { p - 1 } ( 1 - \alpha _ { j } ) .\tag{6}
$$

where $q _ { p } = \tilde { \sigma } _ { \mathrm { v M } , p }$ for stress, $\| \varepsilon _ { p } \| _ { F }$ for strain, $d _ { p }$ for damage, or ${ \pmb v } _ { p } \in \mathbb { R } ^ { 3 }$ for velocity. For vector-valued $q _ { p } ,$ compositing applies component-wise. For every RGB frame, this yields four spatially and temporally aligned FracPhys Maps visualizing the full evolution of fracture dynamics.

## 4.2 FracGen: Teaching Video Generator Fracture Dynamics

Given an object image along with physics controls (tensile load, material, and fracture properties), we describe how we train FracGen to jointly predict fracture RGB video alongside its corresponding dynamic physics maps $\hat { \bf V } = \{ \hat { \bf V } ^ { \mathrm { r g b } } , \hat { \bf V } ^ { m } \}$ . This joint prediction enhances both visual quality and physics accuracy, enabling inspection without requiring any external physics simulators.

Constructing Joint Latents. We adopt the latent flow-matching video generation model §3. To support joint prediction, we construct an extended clean latent $z ^ { \mathrm { e x t e n d } }$ , which is formed by height concatenating encoded version of video latents and its dynamic physics map $z ^ { m }$ with $K = 4 \colon$

$$
z ^ { \mathrm { e x t e n d } } = \mathrm { C o n c a t } _ { H } \left( z ^ { \mathrm { r g b } } , z ^ { \mathrm { f o w } } , z ^ { \mathrm { s t r a i n } } , z ^ { \mathrm { s t r e s } } , z ^ { \mathrm { d a m a g e } } \right) \in \mathbb { R } ^ { ( 1 + T ^ { \prime } ) \times ( K + 1 ) H ^ { \prime } \times W ^ { \prime } \times C } .\tag{7}
$$

![](images/bf276e6b5e1b07dfe77ff8f36f172426fa18ba0714372325f12be54f826ddf4f.jpg)  
Figure 3: FracGen overview. RGB fracture videos and FracPhys Maps are encoded by a frozen VAE and concatenated along the spatial height dimension for joint generation. Applied loading, material properties and fracture properties provide channel-wise conditioning. The initial object image supplies reference tokens. We adapt the transformer with LoRA using flow-matching supervision and a physics-informed latent consistency regularizer coupling strain, stress, and damage representations.

Following (Chen et al., 2025), this design choice serves as an effective alternative to channel-wise concatenation without adding parameters. Preprocessed physics maps are projected into latent space via a frozen pretrained 3D VAE encoder $z ^ { m } = \mathcal { E } ( \mathbf { V } ^ { m } )$ , avoiding costly retraining while preserving reconstruction fidelity (see §A for more details). Finally, noise is injected into $z ^ { \mathrm { { e x t e n d } } }$ to form $z _ { t } ^ { \mathrm { e x t e n d } }$

Conditioning Physics Signals. FracGen is conditioned on three physics-guided signals: applied forces, material properties, and fracture properties. We detail their latent space encoding below:

1. Applied Loading. We use opposing velocity signals mimicking a universal testing machine (UTM) and render it as moving Gaussian blobs across frames. Blob centers denote grip locations, fixed isotropic covariances define spatial extent, and displacements reflect velocity vectors. We use a frozen 3D VAE to encode this dynamic signal into latent $\boldsymbol { y _ { \mathrm { f o r c e } } } ^ { \mathrm { ~ 2 ~ } } \in \mathbb { R } ^ { ( 1 + T ^ { \prime } ) \times H ^ { \prime } \times W ^ { \prime } \times 1 6 }$ , spatially and temporally aligned with the RGB video representation.

2. Material Properties. Material deformation and fracture depend heavily on Young’s modulus E and Poisson’s ratio $\nu .$ Instead of using an explicit encoder, we construct $y _ { \mathrm { m a t } } ~ \in ~ \mathbb { R } ^ { \breve { ( 1 + T ^ { \prime } ) } \times H ^ { \prime } \times W ^ { \prime } \times 2 }$ by broadcasting normalized parameter maps across space and time. We normalize $E \in [ 1 0 ^ { 3 } , 1 0 ^ { 9 } ]$ Pa as $\begin{array} { r } { \hat { E } = \frac { \log ( \bar { E _ { \mathrm { } } } ) - 6 } { 3 } \in [ - 1 , 1 ] } \end{array}$ and $\nu \in [ 0 . 1 , 0 . 5 ]$ as $\begin{array} { r } { \hat { \nu } = \frac { \nu - 0 . 3 } { 0 . 2 } \in [ - 1 , 1 ] } \end{array}$

3. Fracture Properties. To control fracture behavior, we first define fracture categories using a bipolar one hot category code $\mathbf { c } \in \{ - 1 , + 1 \} ^ { 2 }$ (low-stretch fracture vs. high-stretch fracture). For fracture conditions, we define the fracture onset and critical stretch, at which fracture initiates and fully fails, respectively, as discussed in §4.1 via $\lambda ^ { \prime } = [ \lambda _ { \mathrm { o n s e t } } , \lambda _ { \mathrm { c r i t } } ]$ . We normalize $\lambda \in [ 1 , 1 0 ] ^ { 2 } { \mathrm { ~ t o ~ } } [ - 1 , 1 ] ^ { 2 }$ via $\frac { \lambda ^ { \prime } { - } 5 . 5 } { 4 . 5 }$ and broadcast these across space and time via $y _ { \mathrm { f r a c } } \in \mathbb { R } ^ { ( 1 + T ^ { \prime } ) \times H ^ { \prime } \times W ^ { \prime } \times 4 }$

Following Wan 2.1-1.3B-Control (Wan et al., 2025), we allocate a 32-channel conditioning layout: 16 channels for $y _ { \mathrm { f o r c e } } ;$ 6 channels for $y _ { \mathrm { f r a c } }$ and $y _ { \mathrm { { m a t } } } .$ , and 10 zero-padded channels. We concatenate $\dot { z } _ { t } ^ { \mathrm { e x t e n d } }$ with these physics-condition channels along the channel axis before patchify as shown in Fig. 3.

![](images/4139db3bc6eb658b5d673c0a02798376ed546a12833a8e9988ea7b79219408e9.jpg)  
(a) Cake tearing with early break.

![](images/b1fe577f26b35963606187513bf25887340b2650fe42562e40ea9e84172b7e39.jpg)  
(b) Pikachu extreme stretching.  
Figure 4: Qualitative comparison on unseen objects. We compare generated responses to pulling conditions for (a) a cake that tears after limited stretching and (b) a toy that undergoes sustained elongation. FracGen captures these distinct behaviors while better preserving object appearance; competing methods show limited deformation, collapse or changes in object identity. The top row of each panel shows the force conditioning, and frames progress from left to right.

Conditioning Intact Object Image. We anchor the generated video’s first frame to the intact object, allowing the model to infer shape, texture, and geometry priors before force application. Following Wan-DiT’s (Wan et al., 2025), we encode this single RGB image using the same frozen 3D VAE. To align with the height of the multi-modal extended latent z<sup>extend</sup>, we zero-pad the reference latent along the spatial height axis to indicate the absence of initial physics-map references. The padded reference latent is then processed by a 2D convolution matching the DiT patch size and flattened into reference tokens, which are prepended to the physics-aware input tokens as shown in Fig.3.

Training Losses. Alongside the latent flow-matching loss (§3), we introduce a latent constitutive regularizer inspired by linear elasticity and damage-induced stress degradation. Our physical maps encode scalar strain magnitude, von Mises stress and damage, which are further compressed by a pretrained VAE. We therefore learn an approximate relationship among their latent representations.

During training, we estimate the clean joint latent from the noisy input and the model’s predicted flow using the training scheduler. We extract the corresponding strain, stress, and damage latents, denoted by $\hat { z } ^ { \mathrm { s t r a i n } }$ zˆ<sup>stress</sup>, and zˆ<sup>damage</sup>. From the damage latent, we construct a channel-wise gate $\hat { g } _ { d } = \mathrm { s i g m o i d } ( \hat { z } ^ { \mathrm { d a m a g e } } )$ , with the same shape as the stress latent. Motivated by the isotropic linear-elastic relationship ${ \pmb { \sigma } } = \lambda _ { \mathrm { L } } \operatorname { t r } ( { \boldsymbol { \varepsilon } } ) { \bf  I } + 2 \mu { \boldsymbol { \varepsilon } } ,$ we define an approximate latent stress target:

$$
\hat { z } ^ { \mathrm { s t r e s s , c o n s } } = \lambda _ { \mathrm { L } } f _ { \eta } ( \hat { z } ^ { \mathrm { s t r a i n } } ) + 2 \mu g _ { \xi } ( \hat { z } ^ { \mathrm { s t r a i n } } ) ,\tag{8}
$$

where $f _ { \eta }$ and $g _ { \xi }$ are zero-initialized $1 \times 1 \times 1$ convolutions that learn mappings across latent channels. The Lamé parameters are computed from each sample’s Young’s modulus E and Poisson’s ratio ν:

$$
\lambda _ { \mathrm { L } } = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } , \qquad \mu = \frac { E } { 2 ( 1 + \nu ) } .\tag{9}
$$

The regularizer couples the predicted stress latent to this target through the damage gate:

$$
\mathcal { L } _ { \mathrm { c o n s t } } = w ( t ) \left| \hat { z } ^ { \mathrm { s t r e s s } } - ( 1 - \hat { g } _ { d } ) \odot \hat { z } ^ { \mathrm { s t r e s s } , \mathrm { c o n s } } \right| _ { 2 } ^ { 2 } ,\tag{10}
$$

where ⊙ denotes element-wise multiplication and $w ( t )$ assigns greater weight at lower noise levels. When the gate approaches zero, the stress latent is encouraged to match the constitutive approximation; when it approaches one, the target approaches zero latent, a soft degradation prior rather than exact zero stress. This provides a latent-space analogue of stress degradation, without assuming that the VAE coordinates directly represent physical stress or damage. The final training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { c o n s t } } \cdot \mathcal { L } _ { \mathrm { c o n s t } } ,\tag{11}
$$

where $\lambda _ { \mathrm { c o n s t } }$ controls the regularization weight. We train FracGen in three stages. First, we train on RGB fracture videos with physics conditioning. Second, we initialize from this checkpoint and introduce joint prediction of RGB videos and FracPhys Maps. Third, we fine-tune the model with the complete objective in Eq. 11.

## 5 Experiments

Simulation and Training Details. For simulation, we sample 10 objects from the Objaverse dataset (Deitke et al., 2022) and SketchFab platform, covering two common material classes in stretch-to-tear fracture: 5 high-stretch objects (rubber toys) and 5 low-stretch objects (croissant, cake, muffin, bread, and bread roll). We then use FracSim to yield a paired dataset of RGB fracture videos and dynamic physics maps for generative training by varying different force condition, material properties, and fracture properties (more details can be seen in §A). For training, we leverage this simulated data to train our FracGen model. Among 10 objects, simulated results for 6 objects (3 low-stretch and 3 high-stretch objects) with varying conditions are used for training, yielding 4,212 training samples in total. More details on simulation and training setting could be found in §A).

Benchmark. We hold out 4 of the 10 objects for testing against groundtruth, which is mainly for quantitative evaluation against groundtruth. Note that these evaluation set includes 4 objects with varying physics conditions such as forces, material properties, and fracture properties, yielding 2,808 testing samples in total (single view images for these objects are shown in Fig.7). In addition, we collect another set of single object images (see Fig.8) with no 3DGS reconstruction, and hence no simulated video groundtruth. This is to qualitatively show real-world scenario where a user want to fracture an object from a single photo. These are used to demonstrate generalization to new objects, shown in Fig.4a, Fig.4b, Fig.5. We provide more results on our project website.

![](images/23ee21dca04c3cef0282ee7654a8bad1339666febf61927f8f8059e5fb1a93de.jpg)  
Figure 5: Generalization to in-the-wild objects. FracGen only needs one single object image and jointly generates the fracture video (left panel) and its FracPhys maps (right panel, 2×2 grid in top-left to bottomright order: flow, strain, stress, damage), under varying force directions.

Table 1: Quantitative results on held-out test set. All baselines are fine-tuned on the same dataset.
<table><tr><td>Method</td><td>Physics Map Pred</td><td>FVD↓</td><td>LPIPS↓ PSNR ↑ SAC ↑</td><td></td></tr><tr><td>ForcePrompting-FT (Gillman et al., 2025)</td><td>X</td><td>933.21</td><td>0.29 15.93</td><td>0.092</td></tr><tr><td>PhyCo-FT (Narayanan et al., 2026)</td><td>X</td><td>832.12</td><td>0.33</td><td>14.18 0.434</td></tr><tr><td>CogVideoX-5B-I2V-FT (Yang et al., 2025)</td><td>X</td><td>1533.21</td><td>0.32 14.12</td><td>0.432</td></tr><tr><td>FracGen</td><td></td><td>266.17</td><td>0.15</td><td>21.19 0.946</td></tr></table>

Evaluation Metrics. We evaluate visual quality using standard metrics: FVD (Unterthiner et al., 2019), LPIPS (Zhang et al., 2018), and PSNR. Since existing physics metrics such as VideoPhy (Bansal et al., 2024), VideoPhy2 (Bansal et al., 2026), and PhysicsIQ (Motamed et al., 2026) focus on general commonsense rather than fracture, we introduce several fracture-aware metrics:

1. Stretching Axis Coherence (SAC). SAC measures whether the object’s dominant deformation direction aligns with the ground-truth stretching axis. We estimate the dominant stretching direction from point tracks using (Lai et al., 2026) and compute its alignment with the ground-truth axis; higher scores indicate better alignment (see §A for more details).

2. Damage Progression Alignment (DPA). Because FracGen conditions on fracture timing and predicts damage maps, we evaluate damage accumulation schedule. We primarily used this for ablation studies since baselines lack this capability, further details are in the §A.

3. Constitutive Consistency $( \mathbf { R } _ { c o n s } ^ { 2 } )$ . Unlike other metrics that evaluates physics map individually, this one evaluate the consistency among physics map. Specifically, we use the damage map to mask out already-failed regions and measure linear elasticity in the material that remains intact, precisely the regime where Hooke’s law applies. The metric requires no ground truth, as it tests whether the model’s own stress and strain predictions are mutually consistent. Like DPA, we use it for ablation studies rather than baseline comparisons. Further details are in the §A.

Baselines. We compare FracGen against recent physics-aware video generation models (e.g., ForcePrompting, PhyCo) and video generation baselines (e.g., CogVideoX). We do not compare against PhysCtrl (Wang et al., 2025), as it encodes force as a single force vector, which cannot represent our dual opposing forces. In contrast, ForcePrompting and PhyCo encode force as rendered, pixel-aligned control videos, so we extend them to our two-grip setup by rendering both grips in the control signal, and fine-tune them on the same data (more details in §A). As shown in Tab. 1 and Fig. 4. FracGen achieves superior performance across all metrics, excelling in both video quality and physical alignment (SAC). Qualitatively, baselines fail to capture realistic stretching, and fracture progressions, while FracGen faithfully captures this behavior. More results are on our project website.

(i) Controllable fracture properties (a) Low-stretch objects  
![](images/8f146a257b8f9dc3874230265bc72a3a3daf7eb534096611fd11f97ca27278ef.jpg)

![](images/8cb9cea3d3c8aa052ac87bebfded9d7a5e5fe06d2478b1128014e0440806a8a9.jpg)

![](images/c13b756a73c3cae17a5cdd6250300054389843aa9c8e8e4ac760b0d1c8a87a4a.jpg)

![](images/54382807bfda3dd65ed165395deeac6651d6577208683336c9d65264f724d0a5.jpg)

![](images/47e7b4513814dfe77b2d6bc4002c07fed6cb12dd60384b8bc93c10d4f29d1fb4.jpg)  
(b) High-stretch objects

![](images/ba25b1b02db62370c1d3f2eb56b547cfd9f2d403ca2956fc1996aef7ae637e81.jpg)

![](images/0338f8295da94efadb4c2f8f757b52928b3dbd1f09960921e432342b38a53ab7.jpg)

![](images/1d99f10d692c8ac6458f058ba4010c4111365d5c64573afbeddff19a8d31ac19.jpg)

![](images/18ac2f82657ee82ab52abd774d55079a9677d1fbce1f6ad6c87bc2eecf47c8df.jpg)

![](images/28790e8919cf8b485cc621e28b118eec74d82a954d37d321b8cbad53edb2266c.jpg)

![](images/c986c5734feedf2a7daa3b0cb04a909f4f57aa116bb3caf7be2e84390701813a.jpg)

![](images/1bc05848963749d77ccd688283a7eba5a011ec03809e9dc1036624625189b9f7.jpg)

![](images/fd25ca1d7c31632fa0b000174374c027409c7e7b2c8e967e799230c044f214b9.jpg)

![](images/d72cf3be372752d0b6e052e723ccfba632d9ab9d44e2b6495428d2dee65a9aef.jpg)

![](images/c9e8ebf6bc69cf527d69ac07214e3472f95358b9bb4fa4c404f57ba4cff3e9a6.jpg)

(ii) Controllable material properties  
![](images/b82d0aaaaf8d7426badd7a36c93515fbba4c1784edabec933141ce0d5c002b08.jpg)

![](images/601ae4d97c6eb91c5ec1a3eea256ec70558473d1dcea2460088374d1ab476cec.jpg)  
(a) Soft materials

![](images/65e1c525d19056f9d65124c75ae684c1d43ad30e2a01607d8dc2a67c728d5722.jpg)

![](images/17d937533275ce371abc7af459d8bd4d1de34fcf0f4673c30930ff5f1a03e8d7.jpg)

![](images/751c188ba7ab7b26c58427de330b99708d190c0bb437aafd54c3b89f322fa621.jpg)

![](images/882986bafb5aa85600fa121fa0cfc9385cce057e9222565739b61ae4d8714d30.jpg)

![](images/ddaa1d00942dea5c3a0ee424fdf7d5a83c63a0b4a4fd74a9e0cc5d5517c65d7e.jpg)  
(b) Stiff materials

![](images/12999dbc5daa19fd5b5591edaca359a4a8c0170183aa388c6cb866fa9e6188a6.jpg)

![](images/00e8762c1a73b3b9bb4bda29ea1d77af9a440030bcc651066ea7d3e4d2b82ef1.jpg)  
Figure 6: Controlled deformation and tearing. (i) Generated sequences for objects with low- and highstretch fracture settings. (ii) Changing material stiffness for the same object produces different deformation and tearing behavior.

Controllable FracPhys Map and Fracture Prediction. As shown in Fig. 5, FracGen generates physics maps that align well with fracture videos under varying force conditions (vertical and diagonal tearing directions). Furthermore, Fig. 6 demonstrates enhanced controllability by conditioning on specific material and fracture properties across different settings: low-stretch objects (featuring low stiffness and low critical damage thresholds) versus high-stretch objects (featuring high stiffness and high thresholds). Additional video results are available on the supplement website.

Ablation: Effects of Joint Prediction. Stage 1 (physics conditioning, RGB only) serves as our base model. Adding joint FracPhys prediction (Stage 2) improves video quality (FVD 298 → 290) and additionally yields pixel-aligned physical fields that allow the generated dynamics to be inspected directly (Tab. 2).

Ablation: Effects of Constitutive Loss. With constitutive loss, Stage 3 yields the largest gains across all levels. Video quality improves (FVD 290 → 266), and damage progression aligns more closely with ground truth (DPA 0.438 → 0.536). Predicted maps become more accurate: overall MSE drops from 0.0253 to 0.0175, while strain error increases slightly, which we attribute to the loss prioritizing stress– strain coupling over per-map fidelity. We also observe a rise in consistency $( \bar { R } _ { \mathrm { c o n s } } ^ { 2 } 0 . 3 \bar { 7 } 8 \to 0 . 9 0 2 )$ . As reference, the $\mathbf { R } _ { C o n s } ^ { 2 }$ score for groundtruth physics maps is 0.948.

Table 2: Ablation: video quality, physics alignment, and evaluation of FracPhys map prediction (MSE). Stage 1 has no physics map; hence DPA and Frac-PhysMap evaluation are not applicable.  
Metric Stage 1 Stage 2 Stage 3 (× FracPhys) (✓, w/o Eq.10) (✓, w. Eq.10)
<table><tr><td colspan="3">Video Quality &amp; Physics Alignment</td></tr><tr><td>FVD↓</td><td>298.04</td><td>289.65 266.17</td></tr><tr><td>LPIPS</td><td>0.1555 0.1550</td><td>0.1502</td></tr><tr><td>PSNR ↑</td><td>20.73 20.82</td><td>21.19</td></tr><tr><td>SAC ↑</td><td>0.9460 0.9393</td><td>0.9462</td></tr><tr><td>DPA↑</td><td>0.438</td><td>0.536</td></tr></table>

<table><tr><td colspan="3">FracPhys Map Evaluation</td></tr><tr><td>Flow↓</td><td>0.03735</td><td>0.01147</td></tr><tr><td>Stress↓</td><td>0.01464 一</td><td>0.00429</td></tr><tr><td>Strain↓</td><td>0.02017 一</td><td>0.02681</td></tr><tr><td>Damage ↓</td><td>0.02904 一</td><td>0.02740</td></tr><tr><td>Overall↓</td><td>0.02530 一</td><td>0.01749</td></tr><tr><td> $\mathbf { R } _ { C o n s } ^ { 2 } \uparrow$ </td><td>0.378 一</td><td>0.902</td></tr></table>

## 6 Discussion

We show that pretrained video generators can be adapted to learn material-dependent stretching and tearing from simulation-derived supervision. To our knowledge, FracGen is the first video generation framework to jointly predict material-dependent tearing and the associated physical fields from a single image and specified physical conditions. FracSim provides paired videos and physical maps for joint prediction, while physics-consistency losses encourage agreement among the predicted fields. These allow us to inspect how estimated strain, stress and damage evolve. As video models improve and the materials community releases more datasets, we envision a feedback loop: experimental data improves generative models, and fast predictions help prioritize new experiments. With experimental validation, this could accelerate material-response prediction and support materials discovery.

Our work focuses on deformation and tearing under tensile loading. Brittle fracture and fragmentation under impact are outside our training and evaluation scope. Extending FracGen to these material types would require appropriate simulation data and failure models. The model also inherits the assumptions of its training simulator, so any errors in FracSim would also reflect in FracGen. Finally, multi-step diffusion sampling limits inference speed. Few-step distillation could reduce this cost, provided it preserves fracture behavior and consistency among the predicted physical fields.

## AI use statement

We used generative AI tools to assist with language editing and translation, drafting parts of the paper text, summarizing existing literature, proposing title and keyword candidates, setting up environments for baseline methods, cleaning and reformatting datasets, and building the project website. We did not use generative AI to generate synthetic datasets, propose or refine hypotheses, design research methodology or experiments, or implement our method; theoretical and proof-related uses are not applicable to this work. All AI-assisted outputs were reviewed by the authors: drafted and translated text was edited, baseline and data scripts were verified by running them and inspecting outputs, and literature summaries were checked against the original papers. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## References

Simon Baker, Stefan Roth, Daniel Scharstein, Michael J. Black, J.P. Lewis, and Richard Szeliski. A database and evaluation methodology for optical flow. In 2007 IEEE 11th International Conference on Computer Vision, pp. 1–8, 2007. doi: 10.1109/ICCV.2007.4408903.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation, 2024. URL https://arxiv.org/abs/2406.03520.

Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= HA8KSQW7SO.

Anand Bhattad, Daniel McKee, Derek Hoiem, and David Forsyth. Stylegan knows normal, depth, albedo, and more. Advances in Neural Information Processing Systems, 2023.

Yuanhao Cai, Kunpeng Li, Menglin Jia, Jialiang Wang, Junzhe Sun, Feng Liang, Weifeng Chen, Felix Juefei-Xu, Chu Wang, Ali Thabet, et al. Phygdpo: Physics-aware groupwise direct preference optimization for physically consistent text-to-video generation. arXiv preprint arXiv:2512.24551, 2025.

Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4dnex: Feed-forward 4d generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051, 2022.

Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Generative models: What do they know? do they know things? let’s find out! arXiv preprint arXiv:2311.17137, 2023.

Yutao Feng, Xiang Feng, Yintong Shang, Ying Jiang, Chang Yu, Zeshun Zong, Tianjia Shao, Hongzhi Wu, Kun Zhou, Chenfanfu Jiang, and Yin Yang. Gaussian splashing: Unified particles for versatile motion synthesis and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 518–529, June 2025.

Nate Gillman, Charles Herrmann, Michael Freeman, Daksh Aggarwal, Evan Luo, Deqing Sun, and Chen Sun. Force prompting: Video generation models can learn and generalize physics-based control signals. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=eX5aXfJQZc.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics (TOG), 37(4):1–14, 2018.

Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson W. H. Lau. Dreamphysics: learning physics-based 3d dynamics with video diffusion priors. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications ofArtificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’25/IAAI’25/EAAI’25. AAAI Press, 2025. ISBN 978-1-57735-897-8. doi: 10.1609/aaai.v39i4.32389. URL https://doi.org/10.1609/aaai.v39i4.32389.

Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-incell method. ACM Transactions on Graphics (TOG), 34(4):1–10, 2015.

Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey Stomakhin, and Andrew Selle. The material point method for simulating continuum materials. In Acm siggraph 2016 courses, pp. 1–52. 2016.

Lazar M. Kachanov. Rupture time under creep conditions. International Journal ofFracture, 97(1):11–18, 1999. ISSN 1573-2673. doi: 10.1023/A:1018671022008. URL https://doi.org/10.1023/A:1018671022008.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for realtime radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023. URL https://repo-sam. inria.fr/fungraph/3d-gaussian-splatting/.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

Zihang Lai, Eldar Insafutdinov, Edgar Sucar, and Andrea Vedaldi. Cowtracker: Tracking by warping instead of correlation. arXiv preprint arXiv:2602.04877, 2026.

Jean Lemaître. A continuous damage mechanics model for ductile fracture. Journal of Engineering Materials and Technology-transactions of The Asme, 107:83–89, 1985. URL https://api.semanticscholar.org/ CorpusID:136363677.

Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong MU. OmniphysGS: 3d constitutive gaussians for general physics-based dynamics generation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=9HZtP6I5lv.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024a.

Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded image-to-video generation. In European Conference on Computer Vision (ECCV), 2024b.

Miles Macklin, Matthias Müller, and Nuttapong Chentanez. Xpbd: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games, pp. 49–54, 2016.

Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 948–958. IEEE, 2026.

Sriram Narayanan, Ziyu Jiang, Srinivasa G. Narasimhan, and Manmohan Chandraker. Phyco: Learning controllable physical priors for generative motion, 2026.

Sansit Patnaik and Fabio Semperlotti. Variable-order fracture mechanics and its application to dynamic fracture. npj Computational Materials, 7(1):27, feb 2021. ISSN 2057-3960. doi: 10.1038/s41524-021-00492-x. URL https://doi.org/10.1038/s41524-021-00492-x.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=FjNys5c7VyY.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation, 2019. URL https://openreview.net/forum?id= rylgEULtdN.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. ArXiv, 2503.20314, 2025.

Chen Wang, Chuhao Chen, Yiming Huang, Zhiyang Dou, Yuan Liu, Jiatao Gu, and Lingjie Liu. Physctrl: Generative physics for controllable and physics-grounded video generation. In NeurIPS, 2025.

Xiaogang Wang, Hongyu Wu, Wenfeng Song, and Kai Xu. Fracture-GS: Dynamic fracture simulation with physics-integrated gaussian splatting. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=zcAwK50ft0.

Joshuah Wolper, Yu Fang, Minchen Li, Jiecong Lu, Ming Gao, and Chenfanfu Jiang. Cd-mpm: Continuum damage material point methods for dynamic fracture animation. ACM Transactions on Graphics (TOG), 38 (4):119, 2019.

Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4389–4398, June 2024.

Xiaoyan Xing, Konrad Groh, Sezer Karagolu, Theo Gevers, and Anand Bhattad. Luminet: Latent intrinsics meets diffusion models for indoor scene relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025.

Ke Zhang, Cihan Xiao, Yiqun Mei, Jiacong Xu, and Vishal M. Patel. Think before you diffuse: Llms-guided physics-aware video generation, 2025. URL https://arxiv.org/abs/2505.21653.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. PhysDreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision. Springer, 2024.

Zeshun Zong, Xuan Li, Minchen Li, Maurizio M Chiaramonte, Wojciech Matusik, Eitan Grinspun, Kevin Carlberg, Chenfanfu Jiang, and Peter Yichen Chen. Neural stress fields for reduced-order elastoplasticity and fracture. arXiv preprint arXiv:2310.17790, 2023.

## A Appendix

## A.1 Project Website

We provide link to our project website at https://fracgen.github.io/.

## A.2 Pre-processing dynamic physics map

FracSim 4.1 produces a set of per-frame 2D physics maps: three scalar fields (stress, strain, damage) and one vector field (flow). Before VAE encoding, each raw map is normalized per video and linearly rescale (clipping) into [0, 1]. For the scalar fields (stress, strain, damage), the normalized scalar is replicated identically across the three RGB channels, yielding a grayscale RGB video in which pixel intensity encodes physical magnitude; background pixels normalize to exactly zero and render as black. For the flow field, which is a 2D vector rather than a scalar, we instead apply a Middlebury-style optical-flow color-wheel encoding (Baker et al., 2007). Zero flow therefore also renders as black, consistent with the convention used for the scalar fields (Note that this is what the model sees during training, for visualization as shown in Fig.5, we map it back to white background for the ease of visualization). Since the 3D VAE expects RGB input, each physics map is now a standard RGB video after this pre-processing, requiring no architectural change to accommodate a new modality. We feed each pre-processed physics-map video into the frozen 3D VAE to obtain its latent, as described in §4.2.

## A.3 Additional Background in Material Point Methods

Material Point Method (MPM) (Jiang et al., 2016) is a hybrid Eulerian–Lagrangian discretization for continuum mechanics. MPM represent a body with a set of Lagrangian material points (particles), carrying position $\mathbf { x } _ { p } ,$ velocity $\mathbf { v } _ { p } ,$ mass $m _ { p } ,$ and deformation gradient $\mathbf { F } _ { p } .$ . The deformation gradient $\mathbf { F } _ { p }$ tracks the particle’s accumulated deformation, encoding rotation and strain which mainly drives the constitutive strain-stress response. At each MPM simulation step, mass and momentum are transferred to grid nodes (P2G); grid velocities are updated under internal stress and external constraints; the updated velocities are interpolated back to the particles for advection (G2P); and each particle’s deformation gradient is evolved accordingly

Particle-to-grid (P2G) transfer. Mass and momentum are accumulated onto the grid with the affine particlein-cell (APIC) scheme (Jiang et al., 2015), augmenting each particle’s velocity with a local affine velocity term $\mathbf { C } _ { p } ^ { t }$

$$
m _ { i } ^ { t } = \sum _ { p } w _ { i p } ^ { t } m _ { p } ,\tag{12}
$$

$$
( m \mathbf { v } ) _ { i } ^ { t } = \sum _ { p } w _ { i p } ^ { t } m _ { p } \left[ \mathbf { v } _ { p } ^ { t } + \mathbf { C } _ { p } ^ { t } \left( \mathbf { x } _ { i } ^ { t } - \mathbf { x } _ { p } ^ { t } \right) \right] ,\tag{13}
$$

where $\boldsymbol { w _ { i p } ^ { t } }$ is the B-spline weight coupling particle p to grid node i.

Grid update. Each grid node velocity is integrated forward under the net nodal force $\mathbf { f } _ { i } ,$ combining internal and external forces:

$$
\mathbf { v } _ { i } ^ { t + 1 } = \mathbf { v } _ { i } ^ { t } + \frac { \Delta t } { m _ { i } ^ { t } } \mathbf { f } _ { i } \left( \mathbf { x } _ { i } ^ { t } ; \pmb { \theta } _ { p } \right) .\tag{14}
$$

The force follows from a hyperelastic energy $\Psi ( \mathbf { F } )$ , and $\theta _ { p }$ gathers the governing material properties, Young’s modulus $E ,$ and Poisson’s ratio $\nu .$

Grid-to-particle (G2P) transfer. The updated nodal velocities are interpolated back to the particles, whose positions are then advanced, the local affine velocity term is also updated correspondingly:

$$
\mathbf { v } _ { p } ^ { t + 1 } = \sum _ { i } w _ { i p } ^ { t } \mathbf { v } _ { i } ^ { t + 1 } , \qquad \mathbf { x } _ { p } ^ { t + 1 } = \mathbf { x } _ { p } ^ { t } + \Delta t \mathbf { v } _ { p } ^ { t + 1 } , \qquad \mathbf { C } _ { p } ^ { t + 1 } = \frac { 4 } { ( \Delta x ) ^ { 2 } } \sum _ { i } w _ { i p } ^ { t } \mathbf { v } _ { i } ^ { t + 1 } ( \mathbf { x } _ { i } - \mathbf { x } _ { p } ^ { t } ) ^ { T }\tag{15}
$$

Deformation gradient update. Each particle’s deformation gradient is updated from the velocity gradient sampled off the grid:

$$
\mathbf { F } _ { p } ^ { t + 1 } = \left[ \mathbf { I } + \Delta t \sum _ { i } \mathbf { v } _ { i } ^ { t + 1 } \left( \nabla w _ { i p } ^ { t } \right) ^ { \top } \right] \mathbf { F } _ { p } ^ { t } .\tag{16}
$$

## A.4 Additional Simulation and Training Details

From Cauchy stress tensor to von Mises scalar. We first compute the deviatoric Cauchy stress tensor $\begin{array} { r } { { \pmb s } _ { p } = \tilde { \pmb \sigma } _ { p } - \frac { 1 } { 3 } \operatorname { t r } ( \tilde { \pmb \sigma } _ { p } ) { \bf I } } \end{array}$ and collapse it into a scalar von Mises equivalent stress $\begin{array} { r } { \tilde { \pmb { \sigma } } _ { \mathrm { v M } , p } = \sqrt { \frac { 3 } { 2 } } \pmb { s } _ { p } : \pmb { s } _ { p } . } \end{array}$

Relation to elastoplasticity. Our two material classes differ in their hyperelastic energy density (fixed corotated vs. neo-Hookean), elastic constants, and damage thresholds: FracSim couples hyperelasticity to continuum damage, with no yield surface and no return mapping $( F ^ { P } = I )$ . This is deliberate, as damage thresholds expose fracture timing as a directly controllable quantity, which is what FracGen conditions on. The cost is that we reproduce the visual distinction between materials that tear at low stretch and those that sustain large stretch, but not the permanent deformation of true ductile (plastic) fracture.

Simulation Details. Following PhysGaussian (Xie et al., 2024), we optimize 3DGS per object using an anisotropy regularizer and internal particle filling. Fracture is produced by a continuum damage model coupled to MPM (§4.1), and diversity is obtained by sampling loading, constitutive model, material properties constants, and fracture properties. Every simulation is fully specified by a single configuration, which makes each sample reproducible. All runs use 100 frames with 100 MPM substeps per frame $( \Delta t = 1 0 ^ { - 4 } s )$ , a $1 5 0 ^ { 3 }$ background grid over the normalized domain $[ 0 , 2 ] ^ { 3 }$ , and no gravity, so that the observed deformation is attributable solely to the prescribed load. Across combinations of object, loading, and material/fracture parameters, FracSim yields a paired dataset of RGB fracture videos and dynamic physics maps for generative training.

i) Uniaxial Opposing Forces: Tensile loading is applied kinematically. We select two thin slabs of MPM particles on opposite sides of the object center (thickness 0.05 in simulation units, i.e. roughly one grid cell, and wide enough to span the object’s cross-section) and prescribe their velocities to ±v uˆ for the entire clip, where uˆ is the tearing axis. Because the velocity is enforced on particles rather than on grid nodes, the grips move rigidly with the material and the remaining particles respond through the constitutive law. We vary the speed $v \in [ 0 . 2 , 1 . 5 ]$ , the tearing axis over 12 directions in the image plane $( 0 ^ { \circ }$ to 165<sup>◦</sup> in $1 5 ^ { \circ }$ steps), the grip offset from the center (0.10–0.25), and the uniaxial tensile stretch mode.

ii) Constitutive Model: We match the constitutive model to the deformation regime each material reaches before it fails. For low-stretch objects like bread, we use the fixed-corotated model, which is accurate and stable at small-to-moderate strain. For high-stretched object like rubber toys, it sustain stretches of $3 \times \mathrm { { ~ o r ~ } }$ more before failing, which requires a genuinely finite-strain energy density; for these we use the compressible neo-Hookean model, a standard model of rubber elasticity.

![](images/064ada7ce5830682d9821d5e6a00a9fcd4b092ca15818bf40dfef663015ff2e9.jpg)  
Figure 7: Ten objects used for simulation. First 3 of each category is then used for training while last two are used for testing and benchmarking.

iii) Material Properties: Young’s modulus E and Poisson’s ratio ν are sampled per class (Tab. 3). Larger E gives sharper, more localized cracks with faster snap-back; smaller E gives broader necking before the tear; ν controls how much the cross-section thins under tension. Rubber objects uses much weaker velocity damping than the bread-like objects so its stored elastic energy is released visibly on fracture.

iv) Fracture Properties: We augment each particle with a scalar damage $d _ { p } \in [ 0 , 1 ]$ (0 intact, 1 broken) (Lemaître, 1985). After each substep we take the principal stretches of $\mathbf { F } _ { p }$ (its singular values) and update damage from the largest one as discussed in Eq.4 and degrade the particle’s stress contribution to the grid. The onset stretch $\lambda _ { \mathrm { o n s e t } }$ sets how far material can stretch before weakening; while the crit stretch indicates the critical stretch. Varying these thresholds independently of E and ν decouples when an object breaks from how it deforms beforehand, and is the main source of qualitative diversity in fracture patterns.

Table 3: Per-class simulation parameters.
<table><tr><td></td><td>Low-stretch objects</td><td>High-stretch objects</td></tr><tr><td>Constitutive model Young modulus E Poisson ratio ν Onset, critical stretch  $( \lambda _ { \mathrm { o n } } , \lambda _ { \mathrm { c r } } )$ </td><td>Fixed corotated  $\{ 2 \times 1 0 ^ { 3 } , 6 \times 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$   $\{ 0 . 2 , 0 . 2 5 , 0 . 3 \}$  {(1.5, 2.3), (2.0, 3.0)}</td><td>Neo-Hookean  $\{ 2 \times 1 0 ^ { 4 } , 6 \times 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$   $\{ 0 . 4 5 , 0 . 4 7 , 0 . 4 9 \}$  {(2.6, 3.0), (3.0, 3.3)}</td></tr><tr><td>RPIC damping / grid velocity decay Grip speed v Grip offset from center</td><td>0.5 / 0.9995 {0.5, 0.6, 0.7, 0.8}  $\{ 0 ^ { \circ } , 1 5 ^ { \circ } , \ldots , 1 6 5 ^ { \circ } \}$  {0.10, 0.15, 0.20, 0.25}</td><td>0.005 / 0.9998 {0.6, 0.8, 1.2, 1.4}</td></tr><tr><td>Tearing axes</td><td colspan="2"></td></tr></table>

Training Details. We leverage simulated data from FracSim to train our FracGen model. Among 10 objects, simulated results from 6 objects are used for training (3 low-stretch and 3 high-stretch) and 4 objects are used for testing, and benchmarking (2 low-stretch and 2 high-stretch). Our model is finetuned from a pre-trained Wan 2.1-Fun-V1.1-1.3B-Control (Wan et al., 2025) using LoRA (Hu et al., 2022) with rank 64, applied to the

Frame 15

![](images/143f6e5fc601415188861ba3d45c76fc1cbbf78ee597080ba3dead2b4081ff31.jpg)

![](images/056f7daf0672d81d4cfecb2664fd2bdeb565fc0f818c2b428aaf98159081b7be.jpg)

![](images/871827c82e516beed3d3bb727a6f9ef984f966e68f65c539f903cc8f841d32b3.jpg)  
Low-stretch object (bread-like)

![](images/13b30d5fd1683454613f815bb691cb5c0ba0bda6a4ab99b0ea10c82709f2dd6c.jpg)

![](images/38a182530d1366bfd4f92ffb27d37c8c76d4aa285cb45af5349b1b6e59e72fc6.jpg)

![](images/406b53ea0d788c75335106ba3535ccebb4843511a905cd4537cddcd977c27e17.jpg)

![](images/9061c86a854b8adfffc274b17db29223a7a2f457d220f21bc42f09b65c657b4b.jpg)

![](images/97a09975a13b27448b4cc7d818aa74df4cdb4828b78a87e1a8ab5e2fc1aa9c9f.jpg)

![](images/1c386532349b9a4ad2e6a195317b197a6334931a180d8551c30e43532e8211f7.jpg)  
High-stretch object (rubber toy)  
Frame 10  
Frame 20

Figure 8: More ten random objects used for testing FracGen generalization. Results can be found on our website.

![](images/1b0bd0444fed94ffc21300cc69cce9973522382e2466f0607bf1b4a3b7ae4424.jpg)  
Figure 9: We show illustration of using point tracks to extract principal stretching axis.

query, key, value, output, and feed-forward projections of the DiT backbone. We use the AdamW optimizer with a learning rate of 1e-4, weight decay of 0.01, and a cosine learning-rate schedule with a 40-step warmup. Each video clip consists of 81 frames, and the model is trained with a per-GPU batch size of 1 and gradient accumulation of 4 steps across 4 GPUs, yielding an effective batch size of 16. The total training samples are 4,212 training samples, and total testing/benchmarking samples are 2,808. Additionally, we also collect 10 random object images to further test generalization of FracGen across different new objects. We show images of these objects in Fig.8.

## A.5 Additional Details on Evaluation Metric Design

Stretching Axis Coherence (SAC). Given a fracture video, we first segment out main object using SAM (Kirillov et al., 2023) and run point track algorithm (Lai et al., 2026) to extract all track information.

Point track results are visualized in Fig.9. Given 2D point tracks $\{ \mathbf { p } _ { i } ( t ) \} _ { i = 1 } ^ { N }$ , we estimate the dominant deformation axis over an early window $[ t _ { 0 } , t _ { 1 } ]$ and compare it against the same quantity measured on a reference (ground-truth) clip.

Displacement and drift removal. For each point i visible at both $t _ { 0 } , t _ { 1 }$ , we extract its displacement ${ \bf u } _ { i } = { \bf p } _ { i } ( t _ { 1 } ) - { \bf p } _ { i } ( t _ { 0 } )$ . The mean $\begin{array} { r } { \bar { \textbf { u } } = ~ \frac { 1 } { N } \sum _ { i } \mathbf { u } _ { i } } \end{array}$ is dominated by the object’s rigid drift, since opposing deformation components cancel in the average. We subtract this quantity from each point’s displacement, giving deformation-relative motion $\hat { \mathbf { u } } _ { i } = \mathbf { u } _ { i } - \bar { \mathbf { u } }$

Noise gating. Point tracks might be unstable due to noisy prediction from tracking algorithm or other noisy motion. We use point’s displacement magnitude $w _ { i } = \lVert \hat { \mathbf { u } } _ { i } \rVert$ as a gating function to drop these noisy points with $w _ { i } < \tau$ . If fewer than k points survive, we determine the axis to be undefined (the object has not yet shown measurable deformation).

Dominant axis extraction. Since opposite sides of a stretching object move in opposite directions, averaging directions directly would cancel to zero. We instead form the orientation tensor over unit directions ${ \bf e } _ { i } =$ $\hat { \mathbf { u } } _ { i } / w _ { i } \mathbf { : }$

$$
Q = \frac { \sum _ { i } w _ { i } \mathbf { e } _ { i } \mathbf { e } _ { i } ^ { \top } } { \sum _ { i } w _ { i } } \in \mathbb { R } ^ { 2 \times 2 } ,\tag{17}
$$

Intuitively, the outer product $\mathbf { e } _ { i } \mathbf { e } _ { i } ^ { \top }$ does not care about direction and hence $\mathbf { e } \mathbf { e } ^ { \top } = ( - \mathbf { e } ) ( - \mathbf { e } ) ^ { \top }$ . Weighted average of these terms give the overall voting principal axis. We then perform eigen-decompose on the symmetric matrix $Q$ to extract orthogonal eigenvectors with eigenvalues $\lambda _ { \operatorname* { m a x } } \geq \lambda _ { \operatorname* { m i n } } \geq 0 ;$ the eigenvector of $\lambda _ { \mathrm { m a x } }$ is the dominant axis a. We show in $\mathrm { F i g } . 9$ the principal stretching axis (overlaying red line) computed using this approach from two samples of our fracture video. We also determine a confidence score,

$$
c _ { \mathrm { p r e d } } = 1 - \frac { \lambda _ { \operatorname* { m i n } } } { \lambda _ { \operatorname* { m a x } } } \in [ 0 , 1 ]\tag{18}
$$

Intuitively, $c _ { \mathrm { p r e d } }  1$ when motion is tightly aligned along one line, $c _ { \mathrm { p r e d } }  0$ when motion is isotropically scattered.

Computing SAC. Given the principal stretching axis of prediction and groundtruth fracture video $\mathbf { a } _ { \mathrm { p r e d } } , \mathbf { a } _ { \mathrm { g t } }$ 2 we measure their angular deviation via

$$
\theta = \operatorname { a r c c o s } ( | \mathbf { a } _ { \mathrm { p r e d } } \cdot \mathbf { a } _ { \mathrm { g t } } | { \mathbf \xi } ) \in [ 0 ^ { \circ } , 9 0 ^ { \circ } ] ,\tag{19}
$$

where the absolute value dropping the axis sign ambiguity. This is squashed into a bounded similarity score via a Gaussian kernel,

$$
s _ { \mathrm { r a w } } = \exp \Bigl ( - \left( \theta / \sigma \right) ^ { 2 } \Bigr ) \in ( 0 , 1 ] ,\tag{20}
$$

In this way, a perfect match $\left( \theta = 0 \right)$ scores 1 and the score decays smoothly with angular error (we set $\sigma = 1 5 ^ { \circ } )$ . The final SAC score additionally is computed $\mathbf { a s } ,$

$$
\mathrm { S A C } = s _ { \mathrm { r a w } } \cdot c _ { \mathrm { p r e d } } ,\tag{21}
$$

with $c _ { \mathrm { p r e d } } : = 0$ whenever the predicted axis itself is undefined (scored as a hard failure, $\mathrm { S A C } = 0 _ { ; }$ , rather than excluded).

Damage Progression Alignment (DPA). Let $d ( \mathbf { x } , t ) \in [ 0 , 1 ]$ denote the damage field of a clip with $T$ frames and $H \times W$ pixels, read from its damage-map video. For each frame we record its total damage, normalized by frame size,

$$
a ( t ) = \frac { 1 } { H W } \sum _ { \mathbf { x } } d ( \mathbf { x } , t ) ,\tag{22}
$$

together with the peak damage the clip ever attains, $a ^ { \star } = \operatorname* { m a x } _ { t } a ( t )$ . The idea of DPA is to assess whether the prediction and the ground truth reach the same damage checkpoints at the same moments. A checkpoint is a fraction $\beta$ of a clip’s own peak damage, and the moment at which the clip reaches it is

$$
f _ { \beta } = \operatorname* { m i n } \big \{ t : \ a ( t ) \ \geq \ \beta a ^ { \star } \big \} , \qquad \tau _ { \beta } = \frac { f _ { \beta } } { T } .\tag{23}
$$

![](images/939cd1620b92b943122940436f567f7f34e5b865bb48061929486acfb29dac19.jpg)  
Figure 10: Screenshot of annotation tool to provide physics condition for FracGen

With $\Delta \tau _ { \beta } = \tau _ { \beta } ^ { \mathrm { p r e d } } - \tau _ { \beta } ^ { \mathrm { g t } }$ , DPA is the normalized agreement averaged over a set of checkpoints B:

$$
\mathrm { D P A } = \frac { 1 } { | { \cal B } | } \sum _ { \beta \in { \cal B } } \exp \Big ( - \big ( \Delta \tau _ { \beta } / \sigma \big ) ^ { 2 } \Big ) \in ( 0 , 1 ] , \qquad \sigma = 0 . 1 .\tag{24}
$$

We sweep through a set of $B = \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$

Constitutive Consistency $( R _ { \mathbf { c o n s } } ^ { 2 } )$ . Let $\sigma ( \mathbf { x } , t ) , \ \varepsilon ( \mathbf { x } , t )$ and $d ( \mathbf { x } , t )$ denote a predicted stress, strain and damage fields, read from its physics-map videos. In intact material linear elasticity requires stress to be proportional to strain; once damage accumulates the material softens, so stress falls while strain continues to grow. Therefore, we restrict attention to material that has not yet failed,

$$
S = \big \{ ( \mathbf { x } , t ) : \ : d ( \mathbf { x } , t ) \leq d _ { \operatorname* { m a x } } \big \} , \qquad d _ { \operatorname* { m a x } } = 0 . 0 5 ,\tag{25}
$$

On $S$ we fit a single slope through the origin — zero strain must produce zero stress — and record the coefficient of determination,

$$
k = \frac { \sum _ { S } \varepsilon \sigma } { \sum _ { S } \varepsilon ^ { 2 } } , \qquad R _ { \mathrm { c o n s } } ^ { 2 } = 1 - \frac { \sum _ { S } \left( \sigma - k \varepsilon \right) ^ { 2 } } { \sum _ { S } \sigma ^ { 2 } } .\tag{26}
$$

$R _ { \mathrm { c o n s } } ^ { 2 }$ therefore measures how tightly a single linear stress–strain law explains the maps. To validate the robustness of this metric, we evaluate the identical quantity on the ground-truth maps, obtaining $\mathbf { R } _ { \mathrm { c o n s } } ^ { 2 } = 0 . 9 4 8$ which serves as the upper bound. Against it, our FracGen (with constitutive loss) attains 0.902 (95% of the ceiling) while the variant without constitutive loss reaches only 0.378, indicating that the latter’s predicted stress and strain fields are not governed by any single elastic law.

## A.6 Additional Details on Building Annotation Tool

For comprehensive annotating force applied on the single image object, we build an annotation tool that takes an input image and allow user to annotate physics condition. Specifically, users are given a UI to specify uniaxial applied load, material properties, and fracture properties. We show a screenshot of this annotation tool in Fig.10.

## A.7 Additional Details on Baseline Implementation

We provide more details on our attempt on fine-tuning baseline for fair comparison as below.

Matched training protocol. All fine-tuned baselines are trained on the identical FracSim corpus and train/test split used for FracGen (§5). Rather than using FracGen’s hyperparameters on backbones of differing scale, each model is trained under its default setting described in its original work. When a baseline exposes a conditioning channel for applied force we use it, adapted to our two-grip setup as described below; when it does not, force is described in the text prompt only, which cannot express pull direction or magnitude.

1. ForcePrompting (Gillman et al., 2025) fine-tunes a ControlNet on CogVideoX-5B-I2V to follow a rendered Gaussian-blob control video encoding a single applied force (a point poke or a global wind direction). Since tearing requires two opposing pulls rather than one, we extend its point-force conditioning with a custom double-force variant that renders and sums two independent Gaussian blobs at the two grip locations, matching the force representation FracGen consumes. We then finetune this ControlNet on our FracSim force-blob/RGB pairs under the matched protocol. The backbone remains frozen, following the original method. There are no conditions on other channels such as Young’s modulus, Poisson’s ratio, or fracture thresholds, so this baseline sees the same grips and the same pull velocities as FracGen but cannot be told what the object is made of.

2. PhyCo (Narayanan et al., 2026) conditions a frozen Cosmos-Predict2 Video2World backbone with a ControlNet over friction, restitution, deformability, and a single static applied force point. We adapt its data pipeline to accept two independent force points rather than one, to represent our two-grip pulling setup. We then finetune their ControlNet on FracSim data under the same matched training control.

3. CogVideoX-5B-I2V (Yang et al., 2025) has no mechanism for force or physics conditioning at all. Hence, we use intact-object as first frame condition and use a caption to describe tearing behavior (e.g. “Two opposing forces pull on the object, tearing it apart at the center”). We fine-tune it on our the same data provided by FracSim with LoRA of rank 64. Pull direction and magnitude cannot be expressed beyond this, so identical captions are used for opposing pull directions of the same object.

After fine-tuning, every baseline was trained on the same data distribution as similar to FracGen, making it a fair comparison. We also report results for these baselines before and after fine-tuning on the data set produced by our FracSim in Tab.4. This shows that even with training on the same dataset, our FracGen can still outperform these fine-tuned baselines.

## A.8 Additional Results

We present more comparison and controllable results on our website. Additionally, we show qualitative example for ablation studies on the effect of physics map prediction as well as physics consistency loss in Fig.11. As shown, physics map prediction helps to improve physics fidelity of tearing bread (softening first before breaking). On the other hand, physics consistency loss helps to ensure consistency and accurate physics map prediction for stress, strain, and damage.

## A.9 More Related works

3D Gaussians Splatting. 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) represents a static 3D scene as a set of anisotropic Gaussian kernels $\{ G _ { p } \} _ { p = 1 } ^ { \mathcal { P } }$ . Each Gaussian $G _ { p } = \{ x _ { p } , \sigma _ { p } , A _ { p } , \mathcal { C } _ { p } \}$ carries a center $x _ { p } ,$ opacity $\sigma _ { p } ,$ covariance $A _ { p } ,$ and spherical harmonic coefficients $\mathcal { C } _ { p } .$ . For rendering, the Gaussians are projected onto the image plane and composited front-to-back:

$$
G _ { p } = \{ x _ { p } , \sigma _ { p } , A _ { p } , C _ { p } \} , \qquad C = \sum _ { p \in \mathcal { P } } \alpha _ { p } \operatorname { S H } ( l _ { p } ; \mathcal { C } _ { p } ) \prod _ { j = 1 } ^ { p - 1 } ( 1 - \alpha _ { j } ) .\tag{27}
$$

![](images/c35050d81b2831a70f62d646522b825e105caaa7539cd0826af7ffd782ac5013.jpg)  
Figure 11: Ablation Studies. Comparison results for ablating the effect of physics map prediction and physics consistency loss.

Table 4: Comparison of baselines with their zero-shot and fine-tuned (-FT) version.
<table><tr><td>Method</td><td>FVD↓</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SAC ↑</td></tr><tr><td>ForcePrompting</td><td>1226.32</td><td>0.32</td><td>15.07</td><td>0.075</td></tr><tr><td>ForcePrompting-FT</td><td>933.21</td><td>0.29</td><td>15.93</td><td>0.092</td></tr><tr><td>PhyCo</td><td>1595.01</td><td>0.38</td><td>12.51</td><td>0.415</td></tr><tr><td>PhyCo-FT</td><td>832.12</td><td>0.33</td><td>14.18</td><td>0.434</td></tr><tr><td>CogVideoX-5B-I2V</td><td>1844.92</td><td>0.34</td><td>13.78</td><td>0.470</td></tr><tr><td>CogVideoX-5B-I2V-FT</td><td>1533.21</td><td>0.32</td><td>14.12</td><td>0.432</td></tr></table>

where $\alpha _ { p }$ is the effective opacity at pixel location, $\operatorname { S H } ( l _ { p } ; \mathcal { C } _ { p } )$ is the view-dependent color given view $l _ { p } ,$ and the product term is the accumulated transmittance. The Gaussians are optimized via photometric loss across multiple views. We adopt the original 3DGS framework to reconstruct static objects, which are then passed to FracSim (§4.1) to simulate fracture dynamics.

Ductile Fracture Mechanics. Dynamic fracture (Kachanov, 1999; Lemaître, 1985; Wolper et al., 2019; Patnaik & Semperlotti, 2021) is among the most challenging phenomena to model in continuum mechanics. A central concept is material ductility—the capacity to deform plastically before failure—which separates the two canonical fracture modes. Ductile fracture occurs in materials that sustain large deformation under load, typically necking before the cross-section separates, whereas brittle fracture occurs at very low strains, often below 5%, with little observable deformation prior to failure. In this work, we target the kinematic regime of ductile fracture (large deformation before failure), modeled via hyperelasticity with stretch-driven damage rather than plasticity. Specifically, we study fracture under uniaxial tensile loading, where two opposing forces act along a common axis, causing the object to stretch, deform, and eventually break. Under load, the body extends along the loading (longitudinal) direction while contracting in the transverse directions. Prior work on fracture simulation achieves high physical fidelity but requires expert knowledge to configure the governing parameters. CD-MPM (Wolper et al., 2019) models dynamic ductile fracture through continuum damage mechanics, coupling a phase-field damage evolution with the MPM to capture large-deformation crack propagation. To reduce the heavy runtime and memory cost of full-order solvers, Neural Stress Fields (NSF) (Zong et al., 2023) learns a low-dimensional manifold of the Kirchhoff stress field through an implicit neural representation, enabling reduced-order simulation of elastoplasticity and fracture. These methods, however, target physical simulation or geometric fragmentation, and expose control only through low-level physical parameters. To the best of our knowledge, our work is the first to bring stretch-to-tear fracture to a generative model—teaching a video generator the full deformation-to-fracture progression under comprehensive input conditioning, so that users can control which object to break, when it should break, how it breaks. FracSim captures the kinematic signature of this regime — large stretch, necking, progressive separation — through damage accumulation rather than plastic flow.