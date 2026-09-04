# Principia: Relational Physics Tests for Video Models

Varun Varma Thozhiyoor<sup>1,∗</sup> Shivam Tripathi<sup>1,∗</sup> Venkatesh Babu Radhakrishnan<sup>1</sup> Anand Bhattad<sup>2</sup> <sup>1</sup>Indian Institute of Science <sup>2</sup>Johns Hopkins University Project: https://PrincipiaBench.github.io

![](images/77cb665670c987277d1fc758d7173fd492b19b7e2f20ce10524ac81ee825ee51.jpg)  
(a) Restitution

![](images/87d5f0e627e40bff14fcf952dda6fbf3a5c74906558676db7c02cec1f8dcbab9.jpg)  
(b) Friction

![](images/571cb1989f1042c05d09613893e733c6de8165e888b6c39191fe5a3e3c9ef2aa.jpg)  
(c) Projectile

![](images/f7a8bf9ba6d249c02f93af8668665fc0c0fee7d454561823edd7c2eb6c5eac7c.jpg)  
(d) Mass-Spring

![](images/915e264ce8b05501f9f58988b98d2f1f06a3049708ef30167e6f8af78a51d206.jpg)  
(e) Rotational Inertia

![](images/94440fdcc23926f939d02c8ec692ef8e13c9ecff937c7c24ee792da7590d537a.jpg)  
(f) Pendulum

![](images/0db6b1679f47ddc0cdafa089f54096905fc32154a25812a2c9a0d3054a3dcd14.jpg)  
(g) Momentum  
Figure 1: Principia tests physical reasoning in video models through paired-object relational tests. Each panel shows a real-world experimental setup encoding a relational constraint between two objects obeying the same physical law (a-g); these constraints hold independent of camera, scale, or frame rate. We test whether modern video generators and vision-language models preserve them.

## Abstract

Evaluating physical reasoning in video models is difficult because absolute motion measurements depend on frame rate, object scale, and camera calibration, all of which are often ambiguous or unavailable in generated video. We propose a different approach. When two objects in the same scene obey the same physical law, their motions must satisfy predictable relationships, and these relationships hold independent of calibration. We introduce Principia, a benchmark that evaluates Newtonian physics through relational consistency between paired objects. Principia spans eight phenomena – gravity, restitution, friction, rotational inertia, projectile motion, momentum, pendulum, and mass-spring oscillation—across translational, rotational, collisional, and oscillatory dynamics, using real-world scenes recorded under controlled protocols. We also introduce a calibration-independent consistency score that quantifies physical violation directly in image space. Across thousands of generations from six state-of-the-art video generators, no model exceeds 0.42 on Principia despite all scoring around 0.8 on VBench. Vision-language models are evaluated on their ability to detect relational physics violations, with the best model achieving only 67% accuracy and most performing near chance level.

![](images/189fe87132f265f94327631003a924c32448c5f1fab223124dbb87b2abe68f93.jpg)  
(a) Single object generation: motion looks physically plausible.

![](images/cf24205d09658bf4350d2f47a5f232331c1e73834f06debdf60bff9df5459bc0.jpg)  
(b) Paired object generation: arrival times do not match. Constraint: t₁ = t₂ is violated.  
Figure 2: (a) A generated video of a single block sliding down an incline, shown at three timesteps and as a stroboscopic composite (top right): motion appears plausible. (b) A generated video of two blocks with different masses released simultaneously on identical inclines, where the blocks arrive at different times, violating the mass-independence invariant of gravitational acceleration. This <sup>High</sup> <sup>Visual</sup> violation only becomes detectable through relative comparison.

## 1 Introduction

PhysicsEvaluation and benchmarks have been a strong basis for analyzing and improving modern machine learning models. Today’s video generators have reached a point where they can produce highly realistic clips that are hard to distinguish from real videos. VBench, one of the most widely used <sub>(c)</sub> <sub>Visual</sub> <sub>quality</sub> <sub>vs</sub> <sub>physical</sub> <sub>fidelity (d)</sub> <sub>Video</sub> <sub>generators</sub> <sub>on</sub> <sub>Principia (e)</sub> <sub>VLMs</sub> <sub>on</sub> <sub>Principia</sub>benchmarks for video generation, evaluates visual quality along several axes [26]. These have helped drive rapid progress in the field. However, in this paper, we show that visual realism does not imply physical consistency.

To this end, several benchmarks have recently been proposed to evaluate physical consistency in generated videos. Some of these evaluate whether motion appears physically plausible [2, 3]. Others compare generated trajectories against recorded videos [34, 29]. A few also evaluate motion against explicit physical laws [51]. However, plausibility is subjective and can be met by visually convincing but physically incorrect motion. Moreover, trajectory matching penalizes deviation from a single reference even when multiple futures are physically valid, and law-based evaluation often requires recovering or assuming metric quantities. Single-phenomenon studies like free-fall under gravity [42] further reveal targeted failure modes but do not establish whether models preserve physical structure across different physics phenomena.

To make physical evaluation central to modern video generators, especially given their potential use as world models for predicting the consequences of actions [13, 24], we propose Principia, a physics evaluation benchmark that takes into account the relative motion of two objects and measures the extent to which their relative motions satisfy the expected physical relation for eight phenomena (Fig. 1). We evaluate the model’s understanding of gravity in free fall, restitution measured from motion during and after collisions, sliding on inclined planes under friction, rotational inertia and momentum, projectile motion, pendulum swinging under gravity, and spring-mass systems under gravity.

The motivation for using relative motion is easily explained with the following example. Let us take a block moving down an inclined plane. In the case of a single block, as the visual input provides no direct information about material properties or mass, evaluation of the dynamics of sliding can only be carried out qualitatively, such as whether the block is accelerating down the inclined plane.

Consider two blocks of different masses placed under identical conditions (Fig. 2), i.e., having the same starting point on inclined planes with the same angle of inclination and coefficient of friction. Assuming that the two blocks follow the same physical laws, they will reach the bottom of the ramp simultaneously because the acceleration of a block sliding down an inclined plane with friction does not depend on its mass. This equality is an invariant of the underlying physical law; it should not depend on the absolute values of the physical variables. In general, physical laws impose invariants on the motion of multiple objects that can include equalities, ratios, and relative ordering in time and space. Such invariants can be computed from image-space measurements without estimating absolute quantities such as mass, scale, velocity, or acceleration.

For each physical phenomenon, we specify the expected relation between the motions of paired objects and measure how closely the generated motion satisfies that relation. We aggregate these deviations into a continuous invariance score. The test is straightforward: if two objects are released under the same conditions, they must reach the ground at the same time. A systematic deviation from this relationship indicates a violation of the tested physical law, irrespective of whether either individual motion appears plausible.

However, creating such a benchmark is difficult because small geometric, temporal, surface, or release asymmetries may produce signals indistinguishable from genuine physics violations, requiring strict experimental control. We captured hundreds of real-world videos, ensured synchronized release and controlled experiments, and conducted automated and manual validation at all steps to filter out bad videos. The obtained benchmark includes about 500 scenes of eight Newtonian phenomena covering the translational, rotational, collisional, and oscillatory dynamics categories.

The same relational tests can also be applied to VLMs. We additionally build a synthetic testbed in Isaac Sim[35] to generate controlled and counterfactual scenarios at scale. This provides a scalable way to evaluate both video generators and VLMs, and can further be used to develop methods for improving their physical consistency.

Contributions.

• Principia, a benchmark consisting of 500+ real-world scenes capturing 8 physics principles, with calibrated objects, aligned geometry, and coordinated release to facilitate comparison against analytic expectations.

• An invariant score ofrelational consistency which evaluates consistency relations purely in image space without any dependence on metric scale, velocity, or camera intrinsics.

• A scalable synthetic pipeline implemented in Isaac Sim that generates physical scenarios, both controlled and counterfactual, allowing for evaluation of VLMs and video generators.

• A benchmark evaluating six state-of-the-art video generators and four vision-language models reveals that no model achieves a score higher than 0.42 on Principia despite achieving scores around 0.8 on standard visual benchmarks, and that increasing the size of a network does not improve its physical consistency.

## 2 Related Work

Video generative models, world simulators, and physics-guided generation. Recent video diffusion and autoregressive models [25, 33, 9, 46] achieve high visual fidelity and are increasingly described as world models [13]. Probing studies suggest that such models somewhat encode 3D structure and motion [5, 7, 16, 47, 17, 50, 6, 27, 40], but perceptual realism does not imply adherence to physical laws. Several approaches attempt to improve physical realism through reward-based fine-tuning [29, 28], trajectory correction [52], or integration of physics engines [31, 48, 45, 11], all of which depend on how physical correctness is evaluated.

Benchmarking physical reasoning. Existing benchmarks fall into four families (Table 1): simulationbased reasoning in calibrated environments [39, 8, 4, 43, 1, 30]; VQA-based VLM evaluation [12, 18, 15]; generator commonsense scoring via human or VLM judges [2, 3, 23, 32, 22]; and real-video physics evaluation [34, 51, 49, 19, 29]. All require absolute measurements—calibration, scale, or recorded ground-truth parameters—that may be ambiguous in generated video. The closest precursor is the two-object gravity protocol of Thozhiyoor et al. [42], which sidesteps calibration by testing ratios of fall times to isolate Galileo’s principle: unit-free, relational, and quantitative. Principia generalizes this single-phenomenon protocol to eight Newtonian phenomena spanning translational, rotational, collisional, and oscillatory dynamics, and evaluates both video generators and vision-language models.

## 3 The Principia Dataset

Principia consists of 500+ real-world paired-object scenes spanning eight Newtonian phenomena (Table 2). Each scene contains two objects whose motions must satisfy a relational invariant inde-

Table 1: Comparison of physics benchmarks. Principia uniquely combines calibration-independent relational evaluation, quantitative physics-derived scoring, multi-phenomenon coverage, real video data, and evaluation of both video generators and vision-language models. Unit-Free: evaluation independent of metric scale or camera calibration. Relational: explicit intra-scene invariants between paired objects under the same physical law. Quant. Physics: continuous score derived from the physical law being tested. Multi-Phys.: spans multiple distinct physical phenomena. Real: includes real-world video data. Gen: evaluates video generators. VLM: evaluates vision-language models.
<table><tr><td>Work</td><td>Unit-Free</td><td>Relational</td><td>Quant. Physics Multi-Phys.</td><td></td><td>Real</td><td>Gen</td><td>VLM</td></tr><tr><td>Simulation-based reasoning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IntPhys [39, 8]</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>PHYRE/IPHYRE [1, 30], Physion [4, 43]</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>VQA-based VLM evaluation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PhysBench [12], CausalVQA [18], MVPBench [15]</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Generator commonsense &amp; plausibility VideoPhy [2, 3], T2VPhysBench [23],</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>L</td><td>x</td></tr><tr><td>PhyGenBench [32], PhyWorldBench [22] NewtonRewards [28]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Real-video physics evaluation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physics-IQ [34]</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Morpheus [51]</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>PhysVid [49]</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Delfys75 [19]</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>PISA [29]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Unit-free relational</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Galileo&#x27;s equivalence [42]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Principia (ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

pendent of camera, scale, or frame rate. All parameters are held fixed except the one varied (mass, height, length, or moment of inertia), isolating the law being tested.

## 3.1 Construction Protocol

Relational invariants are sensitive to small perturbations: a slight asymmetry in incline angle, a timing skew between paired releases, or a small lateral push on a pendulum can produce a relational signal indistinguishable from a real physics violation. Construction error must be small enough that it does not obscure the violations we aim to detect, which makes data collection the bulk of the work. We recorded approximately 750+ videos in total and applied heavy filtering to retain those meeting the conditions each invariant assumes. The resulting dataset is further augmented using editing models[20] to increase diversity, yielding a final dataset comprising 529 scenes.

We enforce four constraints during data collection. Matched geometry: paired objects share identical contact surfaces, ramps, supports, or springs, machined or assembled to a common specification (for example, the rotational-inertia experiment uses a solid Delrin cylinder and a hollow aluminum cylinder manufactured in-house to match in mass, height, and outer radius within 5%). Synchronized release: mechanical alignment guides hold both objects in matched starting poses and release them simultaneously; hand-released objects are avoided except for pendulum swings. Controlled surface material: contact surfaces are validated for each pairing, with no-slip conditions inspected for rolling experiments and contact faces inspected for friction experiments. Minimized externalforces: lateral velocity at release is held below visible drift, air resistance is negligible on the relevant timescales, and phenomenon-specific assumptions are verified individually (no visible slip on rolling ramps, collision axis aligned with the motion direction for momentum).

We segment and track all objects with SAM3 [10] using hand-annotated initial points. All subsequent measurements happen in pixel space; the relational consistency score depends only on ratios and equalities of pixel-space quantities, so evaluation requires no knowledge of camera intrinsics, frame rate, or metric scale.

Each session yields multiple takes per scene; we include only those passing manual inspection of the recorded video and the resulting SAM3 trajectory (for example, confirming monotonic descent for ramp scenarios, clean ball-on-ground transitions for restitution, and periodic motion for pendulum and mass-spring scenes). Common excluded failure modes include lateral drift, asymmetric release timing, pendulum motion outside the small-angle regime, ball spin biasing projectile trajectories, and surface anomalies producing slip in rolling experiments.

## 3.2 Principia-Synth

In addition to the recorded real-world videos, we construct a synthetic dataset for each phenomenon using Isaac Sim [35, 36]. The physics simulator allows us to render both physically correct videos and corresponding anti-physics videos, which are generated by explicitly violating the relational invariant associated with each phenomenon. We use this Principia-Synth dataset to evaluate whether VLMs can identify relational physics failures. Since such anti-physics videos cannot be recorded in the real world, and AI-generated videos that violate physical constraints often contain visual artifacts that could confound the evaluation, we instead rely on simulation to generate controlled anti-physics scenarios. Representative examples of the real-world physics and anti-physics videos for each phenomenon are provided in section 3.3 and Appendix B.1 respectively.

## 3.3 Phenomena

Principia covers four types of Newtonian dynamics: translational, rotational, collisional, and oscillatory dynamics. Free-fall and projectile dynamics examine translational dynamics under gravity, while inclined-plane dynamics additionally involve friction. The study of rotational inertia involves rotational dynamics and the interaction between translational and rotational motion under no-slip rolling conditions. Restitution and momentum dynamics investigate collisional dynamics and the relationships between the motions of colliding bodies. Pendulum and mass-spring dynamics investigate oscillatory dynamics under gravity and elastic restoring forces. Representative samples from Principia-Synth are shown alongside the description of each phenomenon.

Restitution and Gravity. Two identical balls are dropped from different heights onto the same surface. The rebound height satisfies $r = e ^ { 2 } h ,$ so the relational invariant is $r _ { 1 } / h _ { 1 } = r _ { 2 } / h _ { 2 }$ . The same recordings yield Galileo’s gravity invariant $h _ { 1 } / h _ { 2 } = ( t _ { 1 } / t _ { 2 } ) ^ { 2 }$

![](images/12fd38b16b0e4aaae274bb9e88d059551c329793dbdbc76d58c859e58fe519de.jpg)

Friction. Two blocks of different mass slide down opposite faces of a tentshaped ramp with identical surface material and angle. Because acceleration on an incline with friction satisfies $a = g ( \sin \theta - \mu \cos \theta ) $ , the result is mass-independent. The relational invariant is therefore equality of arrival times, $t _ { 1 } = t _ { 2 } .$ , despite the mass difference between the two blocks.

![](images/df1e08c92b1ec2730ab4fdf66c4812a4a2d636111cf832b857aaeb3400709f2b.jpg)

Rotational Inertia. A solid and a hollow cylinder of matched mass and radius roll down opposite faces of a no-slip ramp. Pure rolling acceleration is $a = g \sin \theta / ( 1 + I / m R ^ { 2 } )$ , so the cylinder with greater moment of inertia arrives later. The relational invariant is the time ratio $t _ { 1 } / t _ { 2 } = \sqrt { ( 1 + I _ { 1 } / m R ^ { 2 } ) / ( 1 + I _ { 2 } / m R ^ { 2 } ) }$

![](images/fa1f29a9d4b16f20fc0eaa2ed72964590a60375277487803239baab150caf7f0.jpg)

Momentum. Two identical balls are released from different heights on a single ramp and collide with identical blocks placed at matched distances. The ball released higher imparts greater momentum and pushes its block farther. The relational invariant is $( x \propto h )$ . In a second setup, identical balls collide with blocks of different masses. The heavier block travels a shorter distance, giving the relational invariant $( x \propto 1 / m )$ .

![](images/01e53431c2cdff4c9272eb1187975e093404475a31ee78f34dbe2776dd4318f8.jpg)

Projectile Motion. Two spheres are launched in opposite directions from a central platform at distinct heights with fixed launch angle. Greater height produces greater launch velocity $( v \propto \sqrt { h } )$ and therefore greater range $( R \propto v )$ . The relational invariant is range ordering: the higher-launched sphere lands farther.

![](images/5858af648563d17566f1dd302c340d06c599eead5f8e1c7eb06dd22d16aa9ee4.jpg)

Pendulum. Two pendulums of different length and identical bob mass are released from the same small angle. The period $T = 2 \pi \sqrt { l / g }$ is independent of bob mass, so the relational invariant is $T _ { 1 } / T _ { 2 } = \sqrt { l _ { 1 } / l _ { 2 } }$ The longer pendulum therefore oscillates more slowly, with a proportionally larger period.

![](images/decf9be291ee6d03729b55cb68e13abeb9ad952d6ebe471ec30e3d6296c48647.jpg)

Table 2: Relational invariants across the eight Principia phenomena. For each scenario, the listed constants are held fixed across paired objects and the listed variables differ. The relational invariant in the rightmost column of the form $\bar { \mathcal { F } _ { \phi } ( o _ { 1 } ) } = \mathcal { F } _ { \phi } ( o _ { 2 } )$ must hold whenever both objects obey the same physical law, regardless of camera, scale, or frame rate.
<table><tr><td>Scenario</td><td>Constants</td><td>Variables</td><td>Relational Invariant</td></tr><tr><td>Restitution</td><td>Material, surface</td><td>Drop heights  $h _ { 1 } , h _ { 2 }$ </td><td> $\sqrt { r _ { 1 } / h _ { 1 } } = \sqrt { r _ { 2 } / h _ { 2 } }$ </td></tr><tr><td>Gravity</td><td>Drop conditions</td><td>Drop heights  $h _ { 1 } , h _ { 2 }$ </td><td> $h _ { 1 } / h _ { 2 } = ( t _ { 1 } / t _ { 2 } ) ^ { 2 }$ </td></tr><tr><td>Friction</td><td>Incline angle θ, friction coeff.  $\mu$ </td><td>Block masses  $m _ { 1 } , m _ { 2 }$ </td><td> $t _ { 1 } = t _ { 2 }$ </td></tr><tr><td>Rotational Inertia</td><td>Mass m, radius R, angle θ</td><td>Moment of inertia  $I _ { 1 } , I _ { 2 }$ </td><td> $\frac { t _ { 1 } } { t _ { 2 } } = \sqrt { \frac { 1 + I _ { 1 } / m R ^ { 2 } } { 1 + I _ { 2 } / m R ^ { 2 } } }$ </td></tr><tr><td>Projectile</td><td>Launch angle θ, gravity g</td><td>Launch heights  $h _ { 1 } , h _ { 2 }$ </td><td> $R \propto v \propto { \sqrt { h } }$ </td></tr><tr><td>Momentum (height)</td><td>Block mass  $m ,$  ball mass</td><td>Release heights  $h _ { 1 } , h _ { 2 }$ </td><td> $x \propto v \propto h$ </td></tr><tr><td>Momentum (mass)</td><td>Release height h, ball mass</td><td>Block masses  $m _ { 1 } , m _ { 2 }$ </td><td> $\begin{array} { r } { x \propto \frac { 1 } { m } } \end{array}$ </td></tr><tr><td>Pendulum</td><td>Release angle θ, gravity g</td><td>Lengths  $l _ { 1 } , l _ { 2 }$ </td><td> $T _ { 1 } / T _ { 2 } = \sqrt { \left( l _ { 1 } / l _ { 2 } \right) }$ </td></tr><tr><td>Mass-Spring</td><td>Spring constant k, gravity g</td><td>Masses  $m _ { 1 } , m _ { 2 }$ </td><td> $x _ { 1 } / x _ { 2 } = \left( m _ { 1 } / m _ { 2 } \right)$ </td></tr></table>

Mass-Spring. Two springs with identical spring constants are loaded with masses $m _ { 1 }$ and $m _ { 2 }$ and allowed to settle at equilibrium. By Hooke’s law, equilibrium extension is $x = m g / k ,$ , so extension scales linearly with suspended mass. The relational invariant is $x _ { 1 } / x _ { 2 } = m _ { 1 } / m _ { 2 }$ , with proportional extensions.

![](images/3099dfa519f44517bc003be4cb1f30e28d018f97e08808fecf03720f354d979a.jpg)

## 4 Experiments

## 4.1 Setup

We evaluate six video generators (Omni [21], Veo-3.1 [38], Wan2.2-5B/14B [44], Cosmos-2.5- 2B/14B [41]) and four vision-language models (Gemini-3.1-Pro, Gemini-3-Flash [14], Qwen-32B, Qwen-4B [37]). Each generator is conditioned on a text prompt and the first frame of a real recording, with the experimenter, suspension strings, and visible release mechanisms inpainted out using Nano Banana 2 [20] so the model conditions on physics rather than apparatus. We sample each scenario using multiple random seeds and average the resulting scores across seeds to obtain a sample score. The Principia benchmark assumes that models can generate basic qualitatively correct motion (e.g., a dropped object moves downward or an object on an inclined plane moves down the slope). We therefore filter out non-conforming videos before computing the Principia scores(More details in Appendix A.6). Total inference compute exceeds 2,600 A100-hours across the four open-weights generators (Appendix A.5).

Principia Consistency Score. For each phenomenon $\phi ,$ the invariant defines two scalar quantities $\mathcal { F } _ { \phi } ( o _ { 1 } )$ and $\mathcal { F } _ { \phi } ( o _ { 2 } )$ that should be equal under correct physics. Depending on the phenomenon, these quantities may correspond either to directly measured values (friction) or to ratios derived from the measured values (gravity, spring, restitution, rotational inertia, pendulum). The specific quantities used for each phenomenon are listed in Table 2. We measure how closely a generated video satisfies the invariant using a normalized consistency score:

$$
S _ { \phi } ~ = ~ 1 - \frac { \vert { \mathcal { F } } _ { \phi } ( o _ { 1 } ) - { \mathcal { F } } _ { \phi } ( o _ { 2 } ) \vert } { \vert { \mathcal { F } } _ { \phi } ( o _ { 1 } ) \vert + \vert { \mathcal { F } } _ { \phi } ( o _ { 2 } ) \vert } .
$$

$S _ { \phi }$ equals 1 when the invariant holds exactly and decreases toward 0 as the violation grows; concretely, $S _ { \phi } = 0 . 9 5$ corresponds to a roughly $1 0 \%$ relational asymmetry. The projectile and momentum invariants is qualitative – distance ordering rather than equality – and is scored separately as the fraction of scenes where the ordering is satisfied. The normalization makes $S _ { \phi }$ unit-free and bounded in [0, 1]; per-phenomenon scores reported throughout this paper are calculated by taking mean and standard deviation across sample scores.

![](images/f8c4aca4c6824eaf5bd644131d7130b864f0a4e6b23c3bc6795083665d283814.jpg)  
Figure 3: Visual quality decouples from physical fidelity, and different model classes fail on different phenomena. (a) All six video generators score around 0.8 on VBench but below 0.5 on Principia, clustering in the high-visual-quality, low-physics-fidelity region. (b) Per-phenomenon continuous consistency scores $S _ { \phi }$ for video generators. (c) Per-phenomenon agreement scores for vision-language models. VLM polygons are smoother and more uniform than generator polygons (compare b vs c).

Table 3: Performance of video generators on Principia. We report the continuous relational consistency score $S _ { \phi }$ for each physical phenomenon as mean ± standard deviation across samples, where each samples score is averaged across multiple seeds. Higher values indicate better adherence to the underlying physical invariant. Bold indicates the best-performing model for each phenomenon. The Overall score reports the mean ± standard deviation across scenario scores and summarizes the model’s general performance across physical phenomena.
<table><tr><td>Model</td><td>Restitution</td><td>Gravity</td><td>Friction</td><td>Inertia</td><td>Projectile</td><td>Momentum</td><td>Spring</td><td>Pendulum</td><td>Overall</td></tr><tr><td>Veo-3.1 [38]</td><td>0.479 ± 0.354</td><td>0.524 ± 0.378</td><td>0.387 ± 0.400</td><td>0.669 ± 0.380</td><td>0.250 ± 0.370</td><td>0.000 ± 0.000</td><td>0.381 ± 0.160</td><td>0.343 ± 0.309</td><td>0.379 ± 0.186</td></tr><tr><td>Cosmos-2.5-2B [41]</td><td>0.272 ± 0.301</td><td>0.378 ± 0.357</td><td>0.078 ± 0.176</td><td>0.044 ± 0.154</td><td>0.000 ± 0.000</td><td>0.074 ± 0.177</td><td>0.130 ± 0.167</td><td>0.205 ± 0.191</td><td>0.148 ± 0.120</td></tr><tr><td>Cosmos-2.5-14B [41]</td><td>0.179 ± 0.233</td><td>0.365 ± 0.337</td><td>0.319 ± 0.339</td><td>0.171 ± 0.303</td><td>0.019 ± 0.094</td><td>0.015 ± 0.084</td><td>0.147 ± 0.161</td><td>0.380 ± 0.268</td><td>0.199 ± 0.135</td></tr><tr><td>Wan2.2-5B [44]</td><td>0.158 ± 0.243</td><td>0.220 ± 0.262</td><td>0.307 ± 0.357</td><td>0.470 ± 0.371</td><td>0.009 ± 0.067</td><td>0.067 ± 0.249</td><td>0.065 ± 0.125</td><td>0.346 ± 0.244</td><td>0.205 ± 0.150</td></tr><tr><td>Wan2.2-14B [44]</td><td>0.504 ± 0.338</td><td>0.540 ± 0.344</td><td>0.709 ± 0.291</td><td>0.729 ± 0.260</td><td>0.231 ± 0.300</td><td>0.017 ± 0.090</td><td>0.099 ± 0.150</td><td>0.519 ± 0.235</td><td>0.419 ± 0.253</td></tr><tr><td>Omni [21]</td><td>0.277 ± 0.327</td><td>0.295 ± 0.334</td><td>0.852 ± 0.264</td><td>0.784 ± 0.261</td><td>0.148 ± 0.266</td><td>0.054 ± 0.155</td><td>0.308 ± 0.205</td><td>0.554 ± 0.241</td><td>0.409 ± 0.272</td></tr></table>

## 4.2 Visual Quality and Physical Fidelity Are Decoupled

Figure 3(a) plots VBench[26] against Principia. All six generators cluster around 0.8 on VBench but between 0.14 and 0.42 on Principia – visual quality and physical fidelity are nearly orthogonal. Models at the visual-quality frontier are no more likely to satisfy physical invariants than substantially lower-quality alternatives.

## 4.3 Per-Phenomenon Results: Generators

Table 3 reports per-phenomenon results. No generator exceeds an average consistency score of0.5, highlighting the difficulty of maintaining relational physical consistency across diverse scenarios. Wan2.2-14B achieves the best overall performance with an average score of 0.419, narrowly outperforming Omni and Veo-3.1 at 0.409 and 0.38 respectively despite their closed-frontier status. Within-family scaling trends are highly uneven. Scaling Cosmos-2.5 from 2B to 14B produces modest gains in friction (+0.24), inertia (+0.13), and pendulum consistency (+0.18), while reducing performance on restitution (−0.09), gravity (−0.01), and momentum (−0.06). Similarly, scaling Wan2.2 from 5B to 14B produces substantial gains across restitution (+0.35), gravity (+0.32), friction (+0.40), inertia (+0.26), and projectile (+0.22), with smaller gains on spring (+0.03) and pendulum (+0.17), while momentum consistency slightly decreases (−0.05). These results suggest that larger model scale does not uniformly improve physical reasoning, and that different invariants stress distinct failure modes in current video generators. Compute also fails to predict fidelity – Cosmos-2.5-14B uses more compute per video than Wan2.2-14B (74 vs 66 min) yet scores 0.22 lower (Appendix A.3).

Friction  
![](images/19469f83ce2a2ecd824836116967d71bf3dd5a7b13180c18757d99c15d1806ab.jpg)  
Figure 4: Qualitative results on Principia. Stroboscopic composites of generated videos across all seven phenomena (columns) and the five evaluated video generators alongside ground truth (rows). For each phenomenon, we show two sets with different physical configurations and visual contexts. Green markings indicate motion that satisfies the relational invariant; red markings indicate physics violations. Across both sets, we include samples where the real recorded videos are edited to vary backgrounds, lighting, and scene appearance (e.g., indoor vs. outdoor environments, different surface textures, and illumination conditions), allowing the dataset to be scaled and diversified without requiring new physical recordings. For these samples, we additionally provide the original groundtruth first frame prior to augmentation.

Figure 3(b) visualizes per-model failure profiles. Each polygon exhibits a distinct performance profile rather than uniform weakness. Omni performs particularly well on inertia and friction but struggles with restitution, gravity, and momentum. Wan2.2-14B achieves the strongest overall performance, reaching the performance frontier among evaluated models across all scenarios except spring and momentum. The Cosmos models exhibit smaller polygon areas, indicating weaker physical consistency across phenomena. All models perform poorly on the momentum task, potentially reflecting the greater difficulty of modeling complex multi-object interactions. Notably, no model dominates across all physical phenomena, with none fully covering the radar chart. Qualitative comparisons appear in Figures 4,5.

## 4.4 Vision-Language Models

VLMs are evaluated on whether they can detect relational physics violations rather than generate physically correct motion. We evaluate the models on a dataset comprising both real-world samples and synthetic samples from the Principia-Synth dataset. Principia-Synth consists of rendered videos depicting both real-world physics and physics violations for each phenomenon. The anti-physics videos are constructed by explicitly violating the relational invariant associated with each phenomenon. For every scene, the VLM receives the full video along with a phenomenon-specific PASS/FAIL prompt (Appendix B.2); its response is then compared against the known ground truth. Table 4 reports the agreement scores for each phenomenon. No VLM achieves an average agreement exceeding

![](images/a8df7dfdeb773840142029f0017672f2af19c0aad4d6cca5c65d92eb1c67016e.jpg)  
Figure 5: Additional Qualitative results on Principia. Stroboscopic composites of generated videos across all seven phenomena (columns) and the five evaluated video generators alongside ground truth (rows). For each phenomenon, we show two sets with different physical configurations and visual contexts. Green markings indicate motion that satisfies the relational invariant; red markings indicate physics violations. Across both sets, we include samples where the real recorded videos are edited to vary backgrounds, lighting, and scene appearance (e.g., indoor vs. outdoor environments, different surface textures, and illumination conditions), allowing the dataset to be scaled and diversified without requiring new physical recordings. For these samples, we additionally provide the original ground-truth first frame prior to augmentation.

Table 4: Performance of vision-language models Each cell reports the agreement score $A _ { \phi } \colon$ the fraction of scenes where the VLM’s PASS/FAIL judgment matches the binary ground truth. Higher is better. Bold denotes the best per column. The Overall column reports the mean across scenarios for each model.
<table><tr><td>Model</td><td>Restitution</td><td>Gravity</td><td>Friction</td><td>Inertia</td><td>Projectile</td><td>Momentum</td><td>Spring</td><td>Pendulum</td><td>Overall</td></tr><tr><td>Gemini-3.1-Pro [14]</td><td>0.439</td><td>0.489</td><td>0.498</td><td>0.517</td><td>0.517</td><td>0.536</td><td>0.843</td><td>0.519</td><td>0.545</td></tr><tr><td>Gemini-3-Flash [14]</td><td>0.594</td><td>0.644</td><td>0.840</td><td>0.465</td><td>0.665</td><td>0.556</td><td>0.895</td><td>0.626</td><td>0.661</td></tr><tr><td>Qwen-32B [37]</td><td>0.460</td><td>0.553</td><td>0.467</td><td>0.427</td><td>0.517</td><td>0.411</td><td>0.587</td><td>0.579</td><td>0.500</td></tr><tr><td>Qwen-4B [37]</td><td>0.588</td><td>0.580</td><td>0.642</td><td>0.427</td><td>0.547</td><td>0.450</td><td>0.535</td><td>0.557</td><td>0.541</td></tr></table>

0.7, indicating near-chance performance and suggesting architectural, rather than scale-related, limitations.

## 5 Discussion and Conclusion

Video generators are increasingly marketed as world models. The results on Principia suggest otherwise: across six state-of-the-art generators, no model exceeds 0.45 on the continuous metric despite all scoring around 0.8 on visual quality benchmarks; scaling within an architecture regresses on at least one phenomenon in both the Cosmos and Wan families; and open-weights exceeds closed-frontier (Wan2.2-14B exceeds Veo-3.1 and Omni). Current generators produce highly realistic textures and visually appealing renderings but fail to enforce the relational invariants that physical laws require. They have learned to render the world without learning what holds it together.

Vision-language models are no better at detecting physical consistency than video generators are at producing it. Most models perform near chance level, with even the best-performing model achieving only 67% accuracy in distinguishing physically consistent videos from those containing physics violations.

Scope and limitations. Principia tests macroscopic Newtonian mechanics; it does not address fluid dynamics, soft-body deformation, or thermodynamic phenomena. Generators sometimes produce videos with hallucinated objects, missing objects, or severe deformations – failures unrelated to physical reasoning. We do not automatically separate these from genuine physics violations; our scope is physical-law adherence in videos that are otherwise interpretable. As generators mature, more of their output will be evaluable under Principia without changes to the benchmark itself.

Resources and outlook. Final-scale evaluation generated over 4,000 videos at approximately 2,600 A100-hours, with comparable additional compute consumed during prompt iteration, threshold calibration, and apparatus development. Because the consistency score is computed within each generated video, the benchmark is robust to test-set contamination. The first frames are precisely staged physical setups that current image generators cannot reproduce, but they can be edited cheaply to vary backgrounds and lighting using modern image editing models [20], enabling extension to new visual setups. We release first-frame inputs and scenario prompts; full real-world videos will be held back to preserve benchmark integrity. If video generators are to serve as world simulators, they must preserve the structural constraints that physical laws impose—not merely the appearance of motion. Closing this gap will likely require new training signals or architectural biases for relational invariants.

## References

[1] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. Advances in Neural Information Processing Systems, 32, 2019.

[2] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation, 2024.

[3] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

[4] Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2021.

[5] Anand Bhattad, Daniel McKee, Derek Hoiem, and David Forsyth. Stylegan knows normal, depth, albedo, and more. Advances in Neural Information Processing Systems, 36:73082–73103, 2023.

[6] Anand Bhattad, Konpat Preechakul, and Alexei A Efros. Visual jenga: Discovering object dependencies via counterfactual inpainting. arXiv preprint arXiv:2503.21770, 2025.

[7] Anand Bhattad, James Soole, and David A Forsyth. Stylitgan: Image-based relighting via latent control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4231–4240, 2024.

[8] Florian Bordes, Quentin Garrido, Justine T Kao, Adina Williams, Michael Rabbat, and Emmanuel Dupoux. Intphys 2: Benchmarking intuitive physics understanding in complex synthetic environments. arXiv preprint arXiv:2506.09849, 2025.

[9] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

[10] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[11] Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, and Shenlong Wang. Physgen3d: Crafting a miniature interactive world from a single image. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 6178–6189, 2025.

[12] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Campagnolo Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. In The Thirteenth International Conference on Learning Representations, 2025.

[13] Kenneth James Williams Craik. The nature ofexplanation, volume 445. CUP Archive, 1967.

[14] Google DeepMind. Gemini 3 pro: Multimodal foundation model. Online model release, 2026.

[15] Zhuobai Dong, Junchao Yi, Ziyuan Zheng, Haochen Han, Xiangxi Zheng, Alex Jinpeng Wang, Fangming Liu, and Linjie Li. Seeing is not reasoning: Mvpbench for graph-based evaluation of multi-path visual physical cot, 2025.

[16] Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Generative models: What do they know? do they know things? let’s find out! arXiv preprint arXiv:2311.17137, 2023.

[17] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3D Awareness of Visual Foundation Models. In CVPR, 2024.

[18] Aaron Foss, Chloe Evans, Sasha Mitts, Koustuv Sinha, Ammar Rizvi, and Justine T Kao. Causalvqa: A physically grounded causal reasoning benchmark for video models. arXiv preprint arXiv:2506.09943, 2025.

[19] Alejandro Castañeda Garcia, Jan Warchocki, Jan van Gemert, Daan Brinks, and Nergis Tomen. Learning physics from video: Unsupervised physical parameter estimation for continuous dynamical systems. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 27924–27933, 2025.

[20] Google DeepMind. Gemini 3.1 flash image. https://deepmind.google/models/gemini-image/ flash/, 2026.

[21] Google DeepMind. Gemini omni flash: Model card. https://deepmind.google/models/ model-cards/gemini-omni-flash/, 2026.

[22] Jing Gu, Xian Liu, Yu Zeng, Ashwin Nagarajan, Fangrui Zhu, Daniel Hong, Yue Fan, Qianqi Yan, Kaiwen Zhou, Ming-Yu Liu, and Xin Eric Wang. "phyworldbench": A comprehensive evaluation of physical realism in text-to-video models, 2025.

[23] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation, 2025.

[24] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

[25] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

[26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[27] Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel, Shahab Bakhtiari, Blake Richards, and Mike Rabbat. Interpreting physics in video world models. arXiv preprint arXiv:2602.07050, 2026.

[28] Minh-Quan Le, Yuanzhi Zhu, Vicky Kalogeiton, and Dimitris Samaras. What about gravity in video generation? post-training newton’s laws with verifiable rewards. 2025.

[29] Chenyu Li, Oscar Michel, Xichen Pan, Sainan Liu, Mike Roberts, and Saining Xie. Pisa experiments: Exploring physics post-training for video diffusion models by watching stuff drop. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

[30] Shiqian Li, Kewen Wu, Chi Zhang, and Yixin Zhu. I-PHYRE: Interactive physical reasoning. In The Twelfth International Conference on Learning Representations, 2024.

[31] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In European Conference on Computer Vision (ECCV), 2024.

[32] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Quanfeng Lu, Wenqi Shao, Kaipeng Zhang, Yu Cheng, Dianqi Li, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. In Forty-second International Conference on Machine Learning, 2025.

[33] Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-efficient world models. arXiv preprint arXiv:2209.00588, 2023.

[34] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? arXiv preprint arXiv:2501.09038, 2025.

[35] NVIDIA. Isaac Sim. https://github.com/isaac-sim/IsaacSim.

[36] Mihir Prabhudesai, Aryan Satpathy, Yangmin Li, Zheyang Qin, Nikash Bhardwaj, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, and Deepak Pathak. Solving physics olympiad via reinforcement learning on physics simulators, 2026.

[37] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[38] Google Research and DeepMind. Veo 3.1: Text-to-video model. Online model release, 2026.

[39] Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616, 2018.

[40] Ayush Sarkar, Hanlin Mai, Amitabh Mahapatra, Svetlana Lazebnik, David A Forsyth, and Anand Bhattad. Shadows don’t lie and lines can’t bend! generative models don’t know projective geometry... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 28140–28149, 2024.

[41] NVIDIA Cosmos Team. Cosmos-predict2: World foundation models for physical ai. https://github. com/nvidia-cosmos/cosmos-predict2, 2025. GitHub repository, Apache-2.0 license.

[42] Varun Varma Thozhiyoor, Shivam Tripathi, Venkatesh Babu Radhakrishnan, and Anand Bhattad. Objects in generated videos are slower than they appear: Models suffer sub-earth gravity and don’t know galileo’s principle... for now. arXiv preprint arXiv:2512.02016, 2025.

[43] Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith Fan, and Kevin Smith. Physion++: Evaluating physical scene understanding that requires online inference of different physical properties. Advances in Neural Information Processing Systems, 36:67048– 67068, 2023.

[44] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[45] Chen Wang, Chuhao Chen, Yiming Huang, Zhiyang Dou, Yuan Liu, Jiatao Gu, and Lingjie Liu. Physctrl: Generative physics for controllable and physics-grounded video generation. arXiv preprint arXiv:2509.20358, 2025.

[46] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

[47] Xiaoyan Xing, Konrad Groh, Sezer Karaoglu, Theo Gevers, and Anand Bhattad. Luminet: Latent intrinsics meets diffusion models for indoor scene relighting. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 442–452, 2025.

[48] Yu Yuan, Xijun Wang, Tharindu Wickremasinghe, Zeeshan Nadir, Bole Ma, and Stanley Chan. Newtongen: Physics-consistent and controllable text-to-video generation via neural newtonian dynamics. arXiv preprint arXiv: 2509.21309, 2025.

[49] Guanqi Zhan, Xianzheng Ma, Weidi Xie, and Andrew Zisserman. Inferring dynamic physical properties from video foundation models. arXiv preprint arXiv:2510.02311, 2025.

[50] Guanqi Zhan, Chuanxia Zheng, Weidi Xie, and Andrew Zisserman. A general protocol to probe large vision models for 3d physical understanding. Advances in Neural Information Processing Systems, 37:43468–43498, 2024.

[51] Chenyu Zhang, Daniil Cherniavskii, Antonios Tragoudaras, Antonios Vozikis, Thijmen Nijdam, Derck W. E. Prinzhorn, Mark Bodracska, Nicu Sebe, Andrii Zadaianchuk, and Efstratios Gavves. Morpheus: Benchmarking physical reasoning of video generative models with real physical experiments, 2025.

[52] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025.

## Appendix

## A Video Generator Evaluation

Video generators are evaluated on their ability to generate videos satisfying the relational constraint of the phenomenon. In this section we provide additional details on the Principia benchmark and evaluation protocol.

## A.1 Dataset Statistics:

The dataset consists of 401 real-world scenes captured in-house. For each scene, the first frame is extracted and used along with a text prompt as the conditioning for video generation. The conditioning frames are first edited using image-editing models [20] to remove the human experimenter and experimental apparatus. These edited frames are then further augmented to increase visual diversity, resulting in a total of 529 scenes. The per-scenario composition of the dataset is shown in Fig. 6.

![](images/086b6cb749a1516a12fc332204c8ed75dcb551b17cc125548e57b841dd170fbe.jpg)  
Figure 6: Dataset statistics across physical phenomena. The figure summarizes the number of samples available for each physical phenomenon, separated into real-world and augmented scenarios.

## A.2 Evaluation protocol

All quantitative measurements are extracted automatically from the generated videos. Object trajectories are obtained using SAM3[10], which tracks the position(centroid/bottom) of each object throughout the sequence. Phenomenon-specific events (e.g., impact, arrival, turning points) are then detected from the tracked trajectories to compute the quantities required for evaluating the corresponding relational invariant.

Restitution and Gravity. The centroid trajectory of each ball is tracked using SAM3. Ground impact is detected when the centroid falls below a predefined ground threshold and the vertical velocity changes sign from positive (downward) to negative (upward). The rebound apex is identified as the subsequent frame where the vertical velocity changes from negative to positive. Initial drop height, rebound height, and time of flight are computed from these detected events and used to evaluate the restitution and gravity invariants.

Friction and Rotational Inertia. For both scenarios, the arrival time of each object is measured as the first frame in which its bottom most pixel(to handle size differences in blocks) crosses a predefined annotated threshold at the end of the incline. These arrival times are used to evaluate the friction and rotational invariants.

Projectile Motion. The projectile centroid trajectory is tracked using SAM3. Ground impact is detected when the vertical velocity becomes positive after descent and the centroid crosses the ground threshold. The horizontal range is computed as the horizontal distance between the impact location and the annotated launch point at the end of the incline.

Pendulum. The oscillation period of each pendulum is estimated from the centroid trajectory of the bob. A half-period is detected when the horizontal velocity reverses sign after the bob has crossed the pendulum’s vertical midpoint (anchor line). The full period is obtained by doubling the measured half-period.

Spring. The maximum extension of each spring is measured from the first oscillation. The oscilla tion turning point is detected when the vertical velocity changes from positive to negative, and the corresponding vertical displacement is recorded. Object masses are estimated from the cube side lengths assuming constant density, i.e., $m \propto l ^ { 3 }$ , and the measured displacement ratio is compared against the expected mass ratio.

Momentum. The vertical displacement of each target block is measured by tracking the bottom edge of the block. The maximum post-collision displacement is extracted from the trajectory and compared across the two interactions to evaluate the expected ordering of momentum transfer.

Table 5: Model details. We use default configurations for high-quality generation. Inference time is measured on an A100 80GB; Veo-3.1 and Omni are accessed through their public APIs.
<table><tr><td>Model</td><td>Resolution</td><td>Frames</td><td>FPS</td><td>Time</td></tr><tr><td>Wan2.2-5B [44]</td><td>1280×704</td><td>121</td><td>24</td><td>7 min</td></tr><tr><td>Wan2.2-14B [44]</td><td>1280×720</td><td>81</td><td>16</td><td>66 min</td></tr><tr><td>Cosmos-2.5-2B [41]</td><td>1280×704</td><td>93</td><td>16</td><td>15 min</td></tr><tr><td>Cosmos-2.5-14B [41]</td><td>1280×704</td><td>93</td><td>16</td><td>74 min</td></tr><tr><td>Veo-3.1 [38]</td><td>1280×720</td><td>96</td><td>24</td><td>API</td></tr><tr><td>Omni [21]</td><td>1280×720</td><td>240</td><td>24</td><td>API</td></tr></table>

## A.3 Model Configurations.

We use the default configuration of each evaluated model to obtain high-quality generations. Resolution, frame count, and frame rate are reported in Table 5. Inference times are measured on an NVIDIA A100 80GB and refer to wall-clock time per generated video. Veo-3.1 and Omni are accessed through public APIs.

## A.4 Prompts Structure

We list the prompts used to generate evaluation videos with Principia. Each video generation model is conditioned on the first frame of the video along with the corresponding prompt.

• Restitution/Gravity: A video showing two identical balls being dropped from two different heights onto the ground. The camera is static and positioned to clearly capture the vertical motion ofboth balls. Both ballsfall naturally under gravity, acceleratingfreely with no air resistance and hit the ground. The balls bounce afew times on the ground before coming to rest.

• Friction: A video showing two identical inclined planes with the same angle of inclination. Two blocks of the same material are placed on the inclines at the same heightfrom the bottom. The blocks start from rest and slide down the planes. The camera is fixed and motion is strictly along the planes.

• Rotational Inertia: A video showing two identical inclined planes with the same angle ofinclination. A solid cylinder is placed on one plane and a hollow cylinder ofthe same mass and outer radius is placed on the other. Both cylinders startfrom rest at the same height and roll down their planes without slipping. The camera isfixed and the motion is strictly along the planes.

• Projectile: A video showing two identical inclined planes with the same angle of inclination positioned on an elevated surface. Two identical balls are placed on the inclines. The balls start from rest, roll down the planes, and launch offthe edge to the ground. The camera is fixed and captures the complete motion ofthe balls.

• Momentum: A videofrom afixed camera showing a single wide inclined planefacing directly towards the camera. Two identical tennis balls are placed on the incline at different/same starting heights. Both balls start from rest and roll down the plane towards the camera without slipping. Two solid wooden square blocks ofsame/different masses are placed at the same heightfrom the bottom, directly in the path ofeach ball. The balls collide head-on with their respective wooden blocks, causing them to moveforward. The motion is strictly along the plane and the horizontal direction after the collision.

• Mass-Spring: A video from a fixed camera showing two vertical spring-mass systems suspended side-by-side from a fixed horizontal support. Both springs are entirely identical, having the same unstretched length and the same spring constant. Two wooden cube blocks having different masses are attached to the springs. Initially, both blocks are held elevated right up near the horizontal support, fully compressing the springs. At time zero, both blocks are released simultaneously from rest. The blocks fall under gravity, stretching the springs downwards, and then oscillate vertically about their respective equilibrium positions. The camera isfixed and clearly captures the continuous vertical motion ofthe system.

• Pendulum: A videofrom afixed camera showing two simple pendulums suspendedfrom the same horizontal support. The pendulum bobs are identical in size, shape, and material, but are attached to strings ofdifferent lengths. Both pendulums are initially displaced to the same angular position and released simultaneouslyfrom rest without any initial push. The pendulums swingfreely under gravity in a plane parallel to the camera. The camera is fixed and captures the complete oscillatory motion ofboth pendulums.

## A.5 Evaluation Compute.

Generating the corpus required substantial compute. Wan2.2-14B alone required approximately 1,100 A100-hours (46 days), while Cosmos-2.5-14B required approximately 1300 A100-hours (54 days) to generate the 529 scenes across two seeds. Across all four open-weight models, inference required approximately 2,800 A100-hours, equivalent to 120 days of continuous compute on a single A100.

## A.6 Directional Consistency Score.

We define a Directional Consistency Score(DCS) that evaluates the generated object motion trajectory against an expected object motion trajectory to check for basic qualitative motion.

$$
\mathrm { D C S } = \frac { 1 } { \left| S \right| } \sum _ { i \in S } \frac { \Delta \mathbf { p } _ { i } ^ { \top } \hat { \mathbf { d } } } { \left\| \Delta \mathbf { p } _ { i } \right\| } ,\tag{1}
$$

where $\Delta \mathbf { p } _ { i } = \mathbf { p } _ { i + 1 } - \mathbf { p } _ { i }$ is the object’s displacement between consecutive frames, d<sup>ˆ</sup> is the unit vector representing the expected direction of motion, and $S = \left\{ i : \| \Delta \mathbf { p } _ { i } \| \geq \epsilon \right\}$ is the set of frames with non-negligible displacement. Since both $\Delta \mathbf { p } _ { i } / \lVert \Delta \mathbf { p } _ { i } \rVert$ and d<sup>ˆ</sup> are unit vectors, $\mathrm { D C S } \in [ - 1 , 1 ]$ where a value of 1 indicates perfect agreement with the expected direction and −1 indicates motion consistently opposite to the expected direction.

In all scenarios except the pendulum, $p _ { i }$ denotes the vertical (y) coordinate of the tracked object, with the expected direction $\hat { \mathbf { d } } = + 1$ corresponding to downward motion in image coordinates. For the pendulum scenario, $p _ { i }$ denotes the horizontal (x) coordinate, and d<sup>ˆ</sup> is defined toward the equilibrium position (the center). We use a Directional Consistency Score threshold of 0.8 to filter non-conformational videos. Videos with a DCS below this threshold are assigned a Principia Score of 0, thereby penalizing models that fail to generate the expected qualitative object motion.

## A.7 Scaling Within Architecture

Figure 7 compares performance within each model family as parameters increase: Cosmos-2B to Cosmos-14B (7× scale) and Wan2.2-5B to Wan2.2-14B (2.8× scale). The annotations show the per-phenomenon delta, with negative values shown in red.

The picture is more nuanced than "scaling helps." Cosmos shows substantial improvement on friction (+0.29) and pendulum (+0.24), but regresses on restitution (−0.09). Wan shows large improvements on friction (+0.40), inertia (+0.26), and pendulum (+0.17), but regresses on momentum (−0.05). Critically, in both families at least one phenomenon degrades with scale, suggesting that scaling within a fixed architecture does not uniformly improve physical fidelity—and may actively hurt specific physical regimes that the smaller model handled adequately.

This finding has implications for the path forward: physical fidelity is unlikely to be solved by scale alone within current architectural choices.

![](images/cd2b6c747d8d956adacd666f80558943b14851720ae4ff7568dd1ecfe75d9f6b.jpg)

![](images/c84a2a461e0023797716243e75c4f014ae33a2d7de87894c4603eb91dfe9efff.jpg)  
Figure 7: Within-architecture scaling effects. Per-phenomenon score changes when scaling Cosmos from 2B to 14B parameters (left) and Wan2.2 from 5B to 14B (right). Annotations indicate perphenomenon delta; red values denote regression. Both families show substantial gains on some phenomena (friction, pendulum) but regression on others (Cosmos restitution, Wan momentum), indicating that scaling does not uniformly improve physical fidelity.

## A.8 Camera Sensitivity Analysis

Accurate measurement of the relational invariants using SAM masks requires videos with a static camera, as camera motion can introduce apparent object motion and interfere with mask-based measurements. In an uncontrolled scene, deviations from the expected trajectory can arise from several confounding factors, making it difficult to attribute the measured error to physics alone. We therefore explicitly instruct the video generators to use a fixed camera. However, qualitative inspection of the generated videos shows that models do not always follow this instruction. We consequently evaluate the robustness of our evaluation protocol to camera motion. Specifically, we first compute Principia scores on our filtered real-world samples with a static camera. We then apply synthetic translational camera motion to these videos to simulate camera movement and recompute the scores. The results are presented in Tab. 6. Six out of the seven phenomena remain unaffected. The one exception - Projectile is sensitive to horizontal pan, for a reason specific to how it is scored. It is our only invariant evaluated as a horizontal range ordering rather than a continuous ratio, so a horizontal pan compresses one sphere’s apparent range relative to the other and flips the ordering outright, with no partial credit. The other scenarios remain unaffected indicating that the low principia scores for the models are attributable to bad physics and not to camera motion.

Table 6: Robustness to camera motion. Principia scores on real-world videos under simulated camera motion. Pan and zoom magnitudes denote the percentage change applied to the original video.
<table><tr><td>Phenomenon</td><td>Original</td><td>Pan Down 2% / 5% / 10%</td><td>Pan Left 2% / 5% / 10%</td><td>Zoom 2% / 5% / 10%</td></tr><tr><td>Gravity</td><td>0.934</td><td>0.934 / 0.936 / 0.937</td><td>0.934 / 0.934 / 0.934</td><td>0.934 / 0.934 / 0.935</td></tr><tr><td>Restitution</td><td>0.977</td><td>0.960 / 0.960 / 0.961</td><td>0.976 / 0.959 / 0.977</td><td>0.960 / 0.961 / 0.961</td></tr><tr><td>Pendulum</td><td>0.901</td><td>0.899 / 0.903 / 0.901</td><td>0.900 / 0.912 / 0.920</td><td>0.903 / 0.904 / 0.902</td></tr><tr><td>Inertia</td><td>0.936</td><td>0.935 / 0.938 / 0.901</td><td>0.940 / 0.900 / 0.898</td><td>0.936 / 0.933 / 0.900</td></tr><tr><td>Friction</td><td>0.963</td><td>0.964 / 0.964 / 0.964</td><td>0.964 / 0.964 / 0.964</td><td>0.964 / 0.964 / 0.964</td></tr><tr><td>Momentum</td><td>1.000</td><td>1.000 / 1.000 / 1.000</td><td>1.000 / 1.000 / 1.000</td><td>0.950 / 1.000 / 0.950</td></tr><tr><td>Spring</td><td>0.881</td><td>0.855 / 0.867 / 0.837</td><td>0.844 / 0.851 / 0.850</td><td>0.849 / 0.850 / 0.867</td></tr><tr><td>Projectile</td><td>1.000</td><td>1.000 / 1.000 / 1.000</td><td>0.861 / 0.722 / 0.417</td><td>1.000 / 1.000 / 0.972</td></tr></table>

## B Vision-Language Model Evaluation

We evaluated four vision-language models—Qwen-4B, Qwen-32B, Gemini-3-Flash, and Gemini-3.1-Pro—on Principia-synth. Unlike video generators, which we score on whether their generated motion preserves relational invariants, VLMs are scored on whether they correctly identify physical violations in videos shown to them.

## B.1 Principia-Synth Anti Physics Scenarios

We use Omniverse[35] to create synthetic videos having relational physics violations creating an anti-physics set. Representative examples for each scenario are in Fig.8.

![](images/33a483ee628e7e4f72a87673d05d4e11f7c06ddea270ae9698e00d21bf76c880.jpg)  
(a) Restitution

![](images/9a6976b1b297920df123525d0b6dcd3a547f1da8b5575f81302ada2f5b61e2b6.jpg)  
(b) Friction

![](images/2785b4b024898b0404837c067b2935b9f4b7d231e6ef17ccff1f1346c6469bcf.jpg)  
(c) Projectile

![](images/adf82d4ebb0d8d8c488bedaa8d2c0a57c1557dee2746109ce9fe5ddb61a53ef5.jpg)  
(d) Gravity

![](images/b381754ea4cd03a2388a51f30238a162b9ea7659ef11bdd38e05738587058cdf.jpg)  
(e) Rotational Inertia

![](images/7543bcc7d7b7022cef8fe64c8d089edddd913ad56988cc5f63226ccef52da3e4.jpg)  
(f) Momentum

![](images/baac0d663842122846dae800cdb5603b1b77265047643648b40dc880c5054ad7.jpg)  
(g) Pendulum

![](images/cdd5d4e625ce2a64efc27f81366577ab9e134323b8c287210c2308b0b6a630e9.jpg)  
(h) Spring  
Figure 8: Representative anti-physics examples from Principia-Synth. Each scene intentionally violates the relational invariant of the corresponding physical phenomenon. From left to right, top to bottom: restitution (inconsistent restitution), friction (mass-dependent sliding despite identical surface properties), projectile motion (incorrect range ordering), gravity (incorrect fall-time relationship), rotational inertia (hollow cylinder reaches the bottom before the solid cylinder), momentum (a ball with greater initial momentum transfers less momentum to the corresponding block), pendulum (the longer pendulum oscillates with a shorter period), and spring (the lighter mass exhibits a larger equilibrium displacement than the heavier mass).

![](images/e008522658de9f02e59733999c50df1c42019a04df855077eddf4af0f07a71a5.jpg)

![](images/5cf2f27a02b35344d9fd3304daa40741de03a153a5cbd5cc60e39f3f763defbd.jpg)  
Figure 9: VLM scaling effects. Per-phenomenon agreement changes from Qwen-4B to Qwen-32B (left, 8× scale) and Gemini-3-Flash to Gemini-3.1-Pro (right). Scaling does not lead to an increase in performance.

## B.2 Evaluation Protocol

For each scene, we present the VLM with the full real world physics or anti-physics video and a phenomenon-specific prompt asking the model to classify the physics as PASS or FAIL with a brief explanation. The exact prompts used are:

• Restitution: Check bouncing physics of the two objects. Both objects are identical. Limit the analysis to thefirst bounce only. Return ONLY valid JSON (no markdown, no codefences) in the followingformat: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Gravity: Checkfreefall physics ofthe two objects, specifically the time taken to hit the ground. Both objects are identical. Limit the analysis to the initial fall only, before any bouncing occurs. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Friction: Check sliding physics of the two blocks. Both are made of the same material. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Rotational Inertia: Check rolling dynamics ofthe hollow and solid cylinder. Both have the same mass and radius. Return ONLY valid JSON (no markdown, no codefences) in thefollowingformat: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Projectile: Check projectile motion trajectory of the two balls, specifically the horizontal range. Both balls are identical. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Momentum: Check collision physics ofthe balls with the blocks, specifically the distance travelled by the blocks after the collision. Both balls have the same mass. The blocks are solid and are made of the same material. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Mass-Spring: Check extension behaviour of the springs when mass is placed on them. Both springs have the same spring constant. Both blocks are solid and are made of the same material. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

• Pendulum: Check pendulum motion physics, specifically the time period of the swings. Return ONLY valid JSON (no markdown, no code fences) in the following format: {"verdict":"PASS"|"FAIL","reason":"brief explanation"}.

We score VLM responses by agreement with the ground-truth Principia-synth binary score: a VLM scene receives credit if its PASS/FAIL judgment matches whether the video actually satisfies the relational invariant. The agreement score $A _ { \phi } \in [ 0 , 1 ]$ is the fraction of scenes in phenomenon ϕ where the VLM and the Principia-synth ground truth agree.

## B.3 VLM Scaling

Figure 9 compares within-family scaling for Qwen and Gemini. Scaling does not improve performance, with Gemini and Qwen showing overall performance reductions of 0.12 and 0.04, respectively. For both model families, Friction exhibits the largest regression, with decreases of 0.34 for Gemini and 0.17 for Qwen.

## C Additional Qualitative Results of Video Generators on Principia

We provide model-wise results on our tasks in Figures10 - 14. We also recommend viewing the videos on the project webpage: https://principiabench.github.io/.

![](images/5a94a8cc3a4eef68a7769ea125b343f3e5fe243fd5dc1f08eef4a6c1ecd89266.jpg)  
Figure 10: Qualitative results of Wan2.2-5B. (1) Restitution: Two identical balls are dropped from different heights. The generated video fails to produce plausible falling motion and hallucinates a third ball in the scene. (2) Friction: The model produces implausible sliding motion of the blocks, which additionally hallucinate and deform during motion. (3) Projectile: The generated video fails to produce plausible projectile motion – both balls move upward, and the left ball disappears during the sequence. (4) Inertia: The generated video produces incorrect rolling motion of the cylinders, which additionally deform into different objects. (5) Momentum: The generated video produces implausible motion in the blocks following the collisions. (6) Pendulum: The generated video fails to produce correct oscillatory motion in the bobs – both bobs swing toward the center and then hover. (7) Springs: The generated video produces implausible motion – both blocks travel similar distances despite having different masses, and both springs hallucinate into different colors and shapes.

![](images/ab2d54f9a79c4866f3c9525c5535bc1a2498994e18dac7707e1696cb6af24bcd.jpg)  
Figure 11: Qualitative results of Wan2.2-14B. (1) Restitution: Two identical balls are dropped from different heights. The higher ball fails to rebound to a proportionally higher height than the lower ball, violating the expected restitution relationship. (2) Friction: The model produces implausible sliding motion of the blocks, which additionally hallucinate and deform during motion. (3) Projectile: The ball launched from the higher height fails to travel a larger horizontal distance than the other ball, violating the expected projectile-motion relationship. (4) Inertia: The generated video produces incorrect rolling motion of the cylinders – the solid cylinder travels approximately the same distance as the hollow cylinder. (5) Momentum: The generated video produces implausible motion in the blocks following the collisions. (6) Pendulum: The generated video fails to produce oscillatory motion in the bobs – both bobs swing toward the center and then hover. (7) Springs: The generated video produces implausible motion – both blocks travel similar distances despite having different masses.

![](images/253272c14868bbedb6425292681b639acba460ef247497b09d74d2f18fcd5eab.jpg)  
Figure 12: Qualitative results of Cosmos2.5-2B. (1) Restitution: Two identical balls are dropped from different heights. Both balls hover in the air and fail to fall to the ground. (2) Friction: The model fails to produce sliding motion in the blocks, and both blocks remain stationary. (3) Projectile: The balls launch toward the ground but hallucinate into different objects during motion. (4) Inertia: The generated video fails to produce any rolling motion in the cylinders, and both remain stationary. (5) Momentum: The balls slide down and collide with the blocks before jumping onto them. Although the blocks begin to slide, the traveled distances are not proportional to the transferred momentum – the ball released from the higher height fails to move its corresponding block farther. (6) Pendulum: The generated video produces implausible motion in the bobs – the left bob hovers momentarily while the right bob swings to the opposite side. (7) Springs: The generated video produces implausible motion – both springs and both blocks hallucinate into multiple different objects.

![](images/520409b2fd6c6e86d0a7899a4a273fd3f180f7af14dcbc88f383d023124f0913.jpg)  
Figure 13: Qualitative results of Cosmos2.5-14B. (1) Restitution: Two identical balls are dropped from different heights. The higher ball rebounds to a proportionally higher height than the lower ball, producing physically plausible behavior. (2) Friction: The model produces plausible sliding motion of both the blocks, and preserves the mass-independent timing. (3) Projectile: The right ball fails to follow a plausible projectile trajectory, instead landing on the cardboard box rather than the ground. (4) Inertia: The generated video produces correct rolling motion of both the cylinders – the solid cylinder reaches before the hollow cylinder. (5) Momentum: The generated video produces implausible motion – both balls jump onto the blocks and fail to induce any meaningful motion in the blocks. (6) Pendulum: The generated video produces implausible motion in the bobs – both bobs have similar time periods despite having different lengths. (7) Springs: The generated video produces plausible motion – the heavier block stretches the spring further downward, although the spring visuals hallucinate into a different object.

![](images/badec0b733e737d00ed714ed234a62546f781fbc9011fb15c8bbf1770b35b4a4.jpg)  
Figure 14: Qualitative results of Veo-3.1. (1) Restitution: Two identical balls are dropped from different heights. The higher ball fails to rebound to a proportionally similar height as the lower ball, violating the expected restitution relationship. (2) Friction: The model produces plausible sliding motion of both the blocks, and preserves the mass-independent timing. (3) Projectile: The ball launched from a higher height fails to travel a larger horizontal distance than the other ball, violating the expected projectile-motion relationship. (4) Inertia: The generated video produces correct rolling motion of both the cylinders – the solid cylinder reaches before the hollow cylinder (5) Momentum: The generated video produces implausible motion in the blocks – both blocks travel identical distances. (6) Pendulum: The generated video produces implausible motion in the bobs– both bobs have similar time periods despite having different lengths. (7) Springs: The generated video produces implausible motion in the blocks – both blocks travel similar distances despite having different mass.

![](images/5191457bc01739f4b6e16055ac1913e133deb4ab9459872c693058985cd20b00.jpg)  
Figure 15: Qualitative results of Omni. (1) Restitution: Two identical balls are dropped from different heights. The higher ball fails to rebound to a proportionally similar height as the lower ball, violating the expected restitution relationship. (2) Friction: The model produces plausible sliding motion of both the blocks, and preserves the mass-independent timing. (3) Projectile: The ball launched from a higher height fails to travel a larger horizontal distance than the other ball, violating the expected projectile-motion relationship. (4) Inertia: The generated video fails to produce correct rolling motion of both the cylinders – the hollow cylinder reaches before the solid cylinder and duplicate cylinders appear at the top of the inclines. (5) Momentum: The generated video produces implausible motion in the blocks. (6) Pendulum: The generated video produces implausible motion in the bobs – both bobs have similar time periods despite having different lengths. (7) Springs: The generated video produces implausible motion in the blocks – the right block hallucinates to a larger size and drops further than the left block.