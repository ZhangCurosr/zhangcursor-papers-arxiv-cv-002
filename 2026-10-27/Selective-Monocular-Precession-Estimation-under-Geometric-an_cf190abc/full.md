# Selective Monocular Precession Estimation under Geometric and Evaluation Ambiguity

An Auditable Synthetic Study with ORBIT-SCOUT

Carlos Eduardo Sanoja

Quanta Labs, LLC Universidad Monteávila ORCID: 0009-0000-0339-7072

25 October 2026

## Abstract

Monocular observations of a tumbling object can support a precise fitted trajectory without establishing that the recovered physical axis is correct. We study this distinction for elongated, approximately cylindrical targets observed by a calibrated stationary camera. ORBIT-SCOUT couples image-derived line and rim geometry to constant-cone and torquefree triaxial trajectory families, then returns a linked momentum-axis and mean-precessionrate estimate only when numerical, geometric, temporal and competing-hypothesis checks pass. We describe the measurement objective, selection policy and limitations, and situate them within spacecraft pose estimation, unknown-target reconstruction, rigid-body identification and selective prediction. An internally preregistered population of 96 new synthetic scenes was processed in two cold passes, retaining all 192 attempts. In the primary pass, 38 scenes produced jointly correct estimates, none produced an incorrect estimate, 57 were withheld and one execution was incomplete. Correct coverage on the 64 supported scenes was 25/64. Ninety-five repeated pairs were scientifically equivalent under a frozen comparison policy; the remaining pair was incomplete. The registered coverage, failure-free execution and all-case repeatability targets were not met. Historical audits also exposed an overly permissive axis metric and motion-dependent acquisition cadence. The contribution is a documented measurement-to-decision implementation and a failure-preserving evaluation case study, not a new rigid-body theory or evidence of state-of-the-art accuracy. Zero observed errors does not establish calibrated risk, and conditional temporal checks do not remove shared image-feature bias. Real-image transfer and external reproduction remain open.

Keywords: monocular vision; torque-free rotation; precession; spacecraft motion; selective prediction; geometric ambiguity; reproducible evaluation.

## 1 Introduction

The practical question in observing a tumbling object is not merely whether a model can be fitted. It is whether the available images support the physical quantity a downstream system intends to use. A projected line can belong to a cylinder generator, an end rim or surface texture. A regular sequence can admit competing frequencies. A repeatable implementation can preserve the same incorrect geometric interpretation. These issues become consequential when an estimator reports a narrow uncertainty measure or an apparently perfect success count.

This paper examines a restricted problem: estimating an unoriented angular-momentum direction in camera coordinates and a positive mean precession rate from one calibrated video. The target is assumed to be elongated and approximately cylindrical, its optical longitudinal axis is assumed to align with the relevant principal inertia axis, and the camera is stationary. We do not estimate complete six-degree-of-freedom pose, arbitrary translational motion, absolute inertia, a capture trajectory or all components of angular velocity. These distinctions are essential when comparing the problem with spacecraft navigation and pose estimation.

The literature already contains model-based monocular pose initialization, learned spacecraft pose networks, methods for unknown-target shape and motion, and inference of inertial properties from rotational observations [1–5]. Circular features and dynamical filtering have also been combined for non-cooperative targets [6]. Consequently, using one camera, avoiding an individual CAD input, fitting torque-free dynamics, retaining hypotheses or allowing rejection is not, by itself, a defensible novelty claim.

Our motivation instead comes from the interaction between physical interpretation and evaluation. During development, allowing a global depth reflection in an axis metric made two wrong solutions appear correct. A separate synthetic population tied image cadence to target rate, exposing a nonvisual proxy for a quantity intended to be inferred from images. Imagedomain probes further showed that temporally coherent wrong edges can fit a motion model well. These observations prompted stricter accounting, an explicitly selective reporting contract and a fresh synthetic study. The resulting experiment is useful partly because it failed the planned acceptance targets.

The study asks three questions. First, what can a geometry-based trajectory fit legitimately certify under its stated measurement assumptions? Second, what joint accuracy and coverage does a frozen selective pipeline attain on new synthetic scenes when withholding and execution failure remain distinct? Third, which conclusions survive correction of the metric, acquisition design and repetition accounting? These are case-study questions. They do not assume that our pipeline is the strongest available estimator or that a small synthetic evaluation is suficient for mission qualification.

The paper makes three bounded contributions:

1. A documented implementation connecting calibrated endpoint geometry, two physical motion families, a fixed training-only selection rule and an explicit decision contract separating estimates, withholding and incomplete execution. An appendix derives the endpoint correction used in the objective; its mathematical ingredients are established geometry, not claimed new theory.

2. A prospective synthetic evaluation with a fixed 96-scene denominator, two cold executions per scene, linked axis/rate scoring and preservation of incomplete attempts. The reported outcome is a failed registered validation with a separately authenticated descriptive score.

3. An analysis of how axis equivalence, acquisition metadata, shared preprocessing and postdevelopment reuse change the meaning of an apparent success. Historical corrections and negative numerical controls are retained rather than folded into the final accuracy count.

The strongest empirical statement is narrow: the frozen method emitted 38 correct joint estimates in this particular 96-scene primary pass, with no observed incorrect emission and substantial abstention. The manuscript does not convert that observation into a probability of safety, a universal identifiability claim or a ranking against methods evaluated on diferent observables.

## 2 Related work and the scope of comparison

## 2.1 Review procedure

We conducted a structured narrative review with a literature cutof of 25 October 2026. Searches combined terms for monocular spacecraft pose, unknown non-cooperative targets, rotationalstate and inertia estimation, circular features, temporal tracking, uncertainty, selective prediction and evaluation leakage. Primary papers, author-hosted manuscripts, publisher records and oficial dataset descriptions were preferred. The accompanying search ledger records thematic queries, source locations and depth of inspection. Recent preprints are identified as such; a future conference date in a preprint is not treated as evidence of completed publication.

This is an extended problem-oriented review, not an exhaustive systematic review. There is no claim of a complete database census, duplicate-screening protocol or meta-analytic performance estimate. Some publisher full texts were unavailable; claims about those papers are limited to their accessible author abstracts and previews. In particular, we do not infer undocumented implementation choices from a title. The comparison axes are the observed quantity, target prior, sensor information, temporal assumptions and evidence domain. Numerical scores are not pooled across incompatible tasks.

Broader surveys already organize the field by sensing and prior target knowledge. Opromolla et al. [7] review electro-optical methods for cooperative and uncooperative targets, including known and unknown objects. Pauly et al. [8] distinguish hybrid and direct learned pipelines and discuss datasets, computational deployment and the synthetic-to-real gap. Our review selects work relevant to the axis/rate estimand and evaluation integrity; these broader classifications and deployment concerns are established context.

## 2.2 Known-target monocular pose estimation

Classical model-based methods establish an important baseline for what image geometry can achieve when the object model is known. Sharma, Ventura and D’Amico [1] use a wireframe model, extracted edges and geometric constraints to initialize pose from a single image, followed by refinement; their evaluation includes PRISMA flight imagery. Their feature-to-model correspondence problem difers from ours, but the work rules out treating edge-based monocular estimation or ambiguity-aware refinement as new contributions here.

The Spacecraft Pose Network [2] replaces much of the hand-designed visual representation with a learned detector and attitude prediction, using geometric constraints to recover position. The target remains known. This distinction between non-cooperative and unknown is often obscured: an object may carry no fiducials or communication link while still having a detailed model available to the observer. Our method removes the individual CAD input but substitutes a restrictive cylindrical and dynamical prior. Neither setup is prior-free.

The Satellite Pose Estimation Challenge formalized shared data and metrics for comparing such systems [9]. That common-task structure is precisely what our study lacks for an external performance ranking. A pose error on a known Tango spacecraft is not the same endpoint as an unoriented momentum axis and a time-averaged precession rate on a cylindrical object. A method returning a full orientation may contain information from which our estimand could be derived, but doing so requires a defined temporal estimator, coordinate convention and reference trajectory. We have not performed that adaptation.

Photorealistic simulation supports training and controlled evaluation, but realism is not a binary property. Proença and Gao [10] introduced URSO and a pose approach that represents orientation ambiguity with a mixture formulation. That work is relevant both to appearance variability and to the need to retain alternative attitudes. It does not imply that a low-dimensional geometry model is more reliable than a learned representation, or that either transfers without testing.

## 2.3 Temporal pose tracking and dynamics

Temporal estimation can reduce frame-level ambiguity while creating another failure mode: persistence of a wrong hypothesis. Park and D’Amico [11] combine learned pose observations with an adaptive unscented Kalman filter and evaluate the navigation pipeline on synthetic and robotic-testbed trajectories in SHIRT. Their method estimates relative orbital and attitude states of a known target. In contrast, our ofline fit uses the full source video and conditional leave-block-out checks; it is not a causal navigation filter.

Recent work makes the handling of temporal alternatives explicit. HAT [12], a September 2026 preprint, preserves competing CAD-based pose histories and uses inter-frame motion to anchor and fuse a monocular SLAM trajectory. It requires calibrated RGB images, metric CAD and target regions. Its retained histories are stronger precedent for temporal hypothesis management than a generic claim that our system keeps alternatives. Our bank is finite, ofline and tied to a narrower motion family; its remaining rivals are a veto mechanism, not a global posterior.

TALON [13], a May 2026 preprint, introduces lightweight temporal adapters to a frozen vision transformer and aligns patch tokens with keypoint structure. It illustrates that temporal information can be incorporated at the representation level rather than only after pose estimation. We have not tested learned temporal features, fine-tuned a foundation model or measured the cost of substituting such features in our pipeline. A comparison would need to preserve the same source videos and downstream estimand.

Taken together, these approaches show why using physics and time is too broad a contribu tion claim. The scientific choice is where temporal consistency enters: in image features, pose correspondence, state propagation, trajectory fitting or decision gates. Our checks operate after a shared image measurement stage. That location constrains what they can detect.

## 2.4 Unknown-target reconstruction and navigation

Unknown-target estimation predates recent foundation models. Capuano et al. [3] describe simultaneous pose and shape estimation using optical flow and star-tracker information, together with a structure-from-motion route to a later model-based phase. This is a useful reminder that monocular imagery can be one component of a broader sensing and knowledge system. The auxiliary attitude information and navigation setting must be included in a fair comparison.

Park and D’Amico [14] infer a compact assembly of superquadrics and pose from one image and introduce a multi-object spacecraft reconstruction dataset. Their target prior is learned across shapes rather than supplied as one exact CAD model. Legrand et al. [15] instead construct a target-specific radiance field from a sparse image collection, then render training images for an of-the-shelf pose estimator. Their pipeline selects and annotates target views and evaluates on SPEED+ hardware-in-the-loop imagery. Thus, absence of an original CAD model can still require target-specific reconstruction and pose annotations.

FoundationPose [16] provides a broader example of unified model-based and model-free object pose estimation: a CAD model or reference imagery supplies the object representation, and its described pose pipeline consumes RGB-D observations. It is not a directly executed monocular spacecraft-precession baseline here. DreamSat-Pose [17], a July 2026 preprint revised in September, combines single-view shape reconstruction with learned image-to-geometry correspondence and PnP. This further weakens any claim that estimating something about an unseen target from a single camera is itself novel.

Francesch Huc and D’Amico [18] propose transformer-aided monocular odometry within a multi-state constraint Kalman filter for previously unseen spacecraft. Their October 2026 preprint uses servicer attitude excitation to recover metric information. Its stated simulation assumptions include exact servicer knowledge and calibration, target pointing and referencederived bounding boxes. Attitude-history evaluation permits one constant rotation because the target frame is arbitrary. These are material conditions, not reasons to dismiss the work. They demonstrate a wider navigation objective under diferent support assumptions; our fixed-camera, ofline axis/rate experiment neither reproduces nor supersedes it.

Table 1: Selected nearby approaches and information assumptions. This is a task comparison, not a performance ranking. Simulation does not imply identical rendering, noise or dynamics.
<table><tr><td>Work</td><td>Principal target quantity</td><td>Information or target prior</td><td>Evidence relevant here</td></tr><tr><td>Sharma et al. [1]</td><td>Full relative pose</td><td>Known wireframe; monocular edges</td><td>Includes flight imagery</td></tr><tr><td>Capuano et al. [3]</td><td>Shape and relative motion</td><td>Optical flow; star-tracker information</td><td>Numerical image simulations</td></tr><tr><td>Chi et al. [4]</td><td>Rotational and dynamic parameters</td><td>Monocular odometry; equal transverse inertia</td><td>Simulation</td></tr><tr><td>Zhang et al. [6]</td><td>Pose and motion</td><td>Circular docking ring; nozzle; constrained</td><td>Simulation</td></tr><tr><td>Legrand et al. [15]</td><td>Relative pose</td><td>dynamics Selected, pose-annotated target images; NeRF</td><td>SPEED+ HIL imagery</td></tr><tr><td>Kobayashi and Putkaradze [5]</td><td>Normalized inertia</td><td>Attitude observations; torque-free arcs</td><td>Noise and image-derived attitude simulations</td></tr><tr><td>Francesch Huc and D&#x27;Amico [18]</td><td>Relative navigation</td><td>Learned odometry; known servicer state and</td><td>Synthetic unseen targets</td></tr><tr><td>This study</td><td>Momentum-axis line and mean precession rate</td><td>maneuvers Calibrated fixed camera; cylindrical geometry; principal-axis alignment</td><td>96 synthetic scenes, two cold passes</td></tr></table>

The central lesson is to specify the information budget. One RGB camera does not determine how much is known about geometry, calibration, segmentation, servicer state, illumination or dynamics. A reduction in one kind of prior can be ofset by another. Table 1 makes these distinctions explicit.

## 2.5 Rotational state, inertia and circular geometry

The closest physical precedents do not all use spacecraft pose benchmarks. Chi et al. [4] filter monocular SLAM odometry and fit rotation vectors to a double-cone model for targets with equal transverse inertia. This directly precedes a claim about monocular recovery of symmetrictarget rotational parameters. We have not implemented their estimator, and their parameter errors cannot be compared with our joint axis/rate threshold without matching the observations and definitions.

Hu et al. [19] examine rotational-parameter ambiguity in a stereovision estimator: 24 righthanded principal-axis frames can represent the same measurement history. They identify mass distribution using angular-momentum conservation and constrain the frame used by an EKF. This is a target-fixed coordinate ambiguity. Renaming a target basis preserves a physical momentum vector expressed in camera coordinates; reflecting that camera-space vector generally changes our requested quantity. Their observability argument does not establish identifiability of our endpoint fit.

Zhang et al. [6] identify a docking ring as a spatial circle, use a nozzle feature to remove its pose duality and refine reprojection error before dynamical estimation. Their measurement has five degrees of freedom because rotation about the circle normal is unobserved, and an error-state filter includes inertia-ratio constraints. This is particularly close to our geometric motivation. Our current numerical loss emphasizes line endpoints; rim observations still enter geometry admission. We therefore claim neither the first use of circular spacecraft features nor the first combination of those features with tumbling dynamics.

Panicucci et al. [20] infer a small body’s rotation axis from image feature tracks associated with circular landmark motion, selecting among geometric possibilities. Their asteroid problem difers in shape and visible texture, but it shares the need to distinguish a rotational quantity from a full pose. Circular-image inversion itself is ambiguous: Mariyanayagam et al. [21] discuss the twofold solution and calibration assumptions for a single circle. Temporal data may reduce a branch ambiguity, but only if the data and model contain discriminating information.

Inertial identification is another, stronger task. Pesce et al. [22] use stereo measurements and filtering to estimate motion and inertia ratios. Candan and Servadio [23] combine monocular visual estimates with LiDAR in an augmented UKF. The extra depth channel changes the observability problem. Kobayashi and Putkaradze [5] use attitude-only torque-free arcs, an initialization from constrained equations and nonlinear shooting with exact elliptic dynamics to estimate normalized inertia. Their image-derived attitudes are upstream measurements; our direct endpoint fit does not establish the same information content.

The torque-free equations and their analytic solutions are classical. Van Zon and Schofield [24] provide an exact rigid-body formulation and numerical implementation. Our use of Jacobi elliptic functions and real third-kind elliptic integrals draws on this established mathematics and the identities collected in DLMF [25]. What requires validation in our system is the connection from extracted image structure to the physical trajectory, not the existence of Euler–Poinsot solutions.

## 2.6 Datasets and the simulation-to-reality gap

SPEED+ explicitly tests transfer from synthetic training imagery to hardware-in-the-loop observations [26]. SHIRT adds temporally structured rendezvous observations [11]. These resources provide stronger appearance-domain evidence than our all-synthetic study. However, neither can be inserted unchanged into our benchmark without defining an appropriate cylindrical target subset, motion reference and camera-motion handling.

Dataset names also conceal changing tasks. The original SPARK work [27] concerns multimodal spacecraft recognition, classification and detection. The later SPARK-2024 release pro vides pose-oriented sequences [28]. They should not be cited as interchangeable evaluations. SPE3R’s multi-object shape emphasis [14] and the 2026 SpaceSense-Bench preprint [29], which supplies RGB, depth, LiDAR and semantic information for a larger object collection, address diversity beyond our four optical assemblies. We use these works to identify missing validation dimensions, not to imply that all their modalities or labels would be permissible inference inputs.

Our renderer varies geometry, viewing direction, lighting and material parameters, but includes no added sensor noise or motion blur. There are no flight images and no robotic-testbed measurements. A physically coherent renderer and conservation checks improve internal validity; they do not measure appearance transfer, sensor calibration error in operational conditions or the statistics of real debris.

## 2.7 Selective prediction, uncertainty and evaluation integrity

Selective prediction formalizes the trade-of between output coverage and error among accepted predictions [30]. SelectiveNet [31] learns prediction and rejection jointly. Our system instead uses a fixed conjunction of engineering gates. It is a selective estimator in the operational sense that it can withhold an answer, but it does not inherit the statistical guarantees of either framework.

Uncertainty estimation in spacecraft vision is an active topic. Dixon et al. [32] use deep evidential regression to relate predicted uncertainty to pose error. Chen et al. [33] model pose uncertainty with unit dual-quaternion distributions and a multimodal formulation. Wang et al. [34] construct conformal keypoint sets and propagate sampled possibilities through PnP to pose sets. These approaches concern diferent uncertainty objects and assumptions. A retained bank of trajectories in our implementation is not automatically a calibrated prediction set.

Calibration asks whether reported probabilities correspond to empirical correctness, a distinction emphasized for learned models by Guo et al. [35]. Conformal risk control [36] provides a route to controlling expected loss under its calibration and exchangeability conditions. We have neither an untouched calibration population nor a validated loss family satisfying such a construction. Accordingly, our reports leave calibrated confidence and sigma fields null. A small local sensitivity or fold-to-fold radius is described as a diagnostic.

Finally, evaluation itself is an estimation procedure. Cawley and Talbot [37] show how finitesample model selection can bias subsequent performance assessment. Kapoor and Narayanan [38] document leakage in scientific uses of machine learning. Our pipeline is largely geometric, yet the same principle applies: selecting policies on opened cases and then treating those cases as confirmation is invalid. The study therefore separates development replays, a fresh population and repetitions of the same scenes. This separation is part of the result, not merely administrative bookkeeping.

## 3 Problem, estimand and physical assumptions

Let $I _ { i }$ denote a source image at acquisition time $t _ { i } .$ , with known pinhole intrinsics K and $t _ { 0 } <$ $\cdots < t _ { F - 1 }$ The inputs include images, their acquisition calendar and calibration. Target attitude, inertia, momentum, stratum labels and reference motion do not enter inference. We write $R ( t ) \in \mathrm { S O } ( 3 )$ for body-to-camera orientation under a stationary camera and $\boldsymbol a ( t ) = R ( t ) \boldsymbol e _ { 3 }$ for the longitudinal optical axis. The supported model assumes this optical axis coincides with a principal inertia axis.

For a torque-free rigid body, the body-frame angular velocity ω obeys

$$
\begin{array} { r } { I \dot { \boldsymbol \omega } + \boldsymbol \omega \times ( I \boldsymbol \omega ) = 0 , \qquad \dot { R } = R [ \boldsymbol \omega ] \times \cdot } \end{array}\tag{1}
$$

The kinetic energy $E = { \textstyle \frac { 1 } { 2 } } \omega ^ { \top } I \omega$ and camera-frame angular momentum $H = R I { \omega }$ are constant. Define $h = H / \lVert H \rVert$ . The requested axis is the line $[ h ] = \{ h , - h \} \in \mathbb { R P } ^ { 2 }$ , not a signed spin vector. Precession describes the azimuthal motion of $a ( t )$ about h. With an orthonormal basis u, v for $h ^ { \perp }$ , a continuous azimuth is

$$
\phi ( t ) = \mathrm { u n w r a p } \mathrm { a t a n 2 } \big ( v ^ { \top } a ( t ) , u ^ { \top } a ( t ) \big ) , \qquad \bar { \Omega } = \left| \frac { \sum _ { i } ( t _ { i } - \bar { t } ) ( \phi _ { i } - \bar { \phi } ) } { \sum _ { i } ( t _ { i } - \bar { t } ) ^ { 2 } } \right| .\tag{2}
$$

The implementation reports degrees per second. Here mean precession means the least-squares slope of unwrapped phase over the full source acquisition calendar, not an endpoint secant, instantaneous angular speed or slope only over frames admitted by feature extraction. For a nutating trajectory these definitions need not agree. When the transverse projection vanishes or the reference deems the rate unobservable, the positive-rate correctness criterion is unavailable. $\mathrm { A }$ nonzero visual frequency is not automatically this physical mean.

The inference contract is

$$
\begin{array} { r } { f ( \{ I _ { i } , t _ { i } \} , K ) \in \{ ( \widehat { [ h ] } , \widehat { \bar { \Omega } } ) , \mathrm { ~ w i t h h e l d , ~ i n c o m p l e t e } \} . } \end{array}\tag{3}
$$

![](images/216a74efb4f1945276fb172a1732ac9611c61ee742fc77f9068a4dadfbb110b4.jpg)  
Figure 1: The evaluated decision flow. Fresh temporal fits reset numerical search; they do not remeasure isolated pixel folds. The diagram describes computation and evidence, not a guarantee that admitted features have the correct physical identity.

A withheld outcome is a completed decision that the declared checks do not support a point estimate. An incomplete outcome means execution or evidence could not establish that decision. Neither is counted as a correct estimate. Withholding also does not establish that the scene is physically unobservable: the extractor, search or gate policy may simply be inadequate.

We assume a rigid, approximately cylindrical target with visible geometric support; a fixed camera and target range; accurate acquisition times; supplied, undistorted calibration; and negligible external torque over the observation window. We do not infer metric target size or absolute inertia from the video. The triaxial parameterization permits unequal transverse inertia while preserving principal-axis alignment. Noncylindrical shape, optical/inertial misalignment and biased calibration are designated stress conditions, not silently incorporated into the supported domain.

## 4 Measurement-to-decision method

## 4.1 Image measurements and their conditional nature

The native C++20/OpenCV perception stage exports measured image geometry. Downstream code organizes raw line segments into alternative groups and measures candidate circular-rim geometry. Segments in one group are treated as simultaneous support, while groups within a frame represent competing interpretations. Collinear fragments are deduplicated using a fixed pixel-scale rule; their number is not treated as a count of independent temporal observations.

The strongest unresolved assumption is feature identity. A mathematically correct line projection formula cannot determine whether an extracted line is a physical generator. Similarly, a fitted ellipse need not be the projection of a circular rim. Geometry admission recomputes selected training decisions, but the preceding feature extraction consumes the whole video, including stateful glare preprocessing. Thus, leave-block-out checks later in the pipeline are conditional on shared measured features. They are not pixel-isolated end-to-end cross-validation.

## 4.2 Endpoint projection and positive-ray support

A 3D direction $a ( t )$ has homogeneous vanishing point $q ( t ) = K a ( t )$ . For an observed segment with inhomogeneous endpoints $p _ { 1 } , p _ { 2 } \in \mathbb { R } ^ { 2 }$ , let $\ell = \widetilde { p } _ { 1 } \times \widetilde { p } _ { 2 }$ and $d _ { j } = q _ { x y } - q _ { z } p _ { j }$ . Define $S =$ $d _ { 1 } d _ { 1 } ^ { \top } + d _ { 2 } d _ { 2 } ^ { \top }$ . The free-line endpoint correction is

$$
d _ { \mathrm { l i n e } } ^ { 2 } ( q , p _ { 1 } , p _ { 2 } ) = \frac { ( \ell ^ { \top } q ) ^ { 2 } } { \lambda _ { \mathrm { m a x } } ( S ) } .\tag{4}
$$

This is a pixel-space correction, not an angular uncertainty. It has a finite parallel-line limit when $q _ { z } = 0$ . For $q _ { z } \neq 0$ , physical generator endpoints should lie on a common ray from the vanishing point under the declared geometry. The implemented profile is

$$
\begin{array} { r } { d _ { \mathrm { r a y } } ^ { 2 } = \left\{ \begin{array} { l l } { d _ { \mathrm { l i n e } } ^ { 2 } , } & { d _ { \mathrm { l } } ^ { \top } d _ { 2 } > 0 , } \\ { \operatorname* { m i n } ( \| d _ { 1 } \| ^ { 2 } , \| d _ { 2 } \| ^ { 2 } ) / q _ { z } ^ { 2 } , } & { d _ { 1 } ^ { \top } d _ { 2 } \leq 0 . } \end{array} \right. } \end{array}\tag{5}
$$

At $q _ { z } = 0$ , the first branch applies. In the second branch an endpoint reaches the vanishing point; this is generally an infimum for distinct endpoints at finite positive depth. Separate support gates therefore matter. Appendix A derives these formulas and states their limits. The profile is sign-invariant in $q ,$ consistent with an unoriented longitudinal direction.

For frame $i ,$ candidate group g contains $m _ { i g }$ deduplicated segments. With $\sigma _ { \mathrm { p x } } = 1$ pixel and Huber transition $\delta = 2 . 5$ , the working objective is

$$
\rho _ { \delta } ( r ) = { \binom { r ^ { 2 } , \quad } { 2 \delta | r | - \delta ^ { 2 } , \quad | r | > \delta , } }\tag{6}
$$

$$
L ( \theta ) = \frac { 1 } { N } \sum _ { i \in \mathcal { T } _ { \mathrm { l i n e } } } \operatorname* { m i n } _ { g \in \mathcal { G } _ { i } } \frac { 1 } { m _ { i g } } \sum _ { j = 1 } ^ { m _ { i g } } \rho _ { \delta } \left( \frac { d _ { \mathrm { r a y } } ( K a _ { \theta } ( t _ { i } ) , p _ { i g j , 1 } , p _ { i g j , 2 } ) } { \sigma _ { \mathrm { p x } } } \right) .\tag{7}
$$

Here N is the number of distinct admitted training line frames. Averaging within a group prevents raw fragment count from directly multiplying frame weight; the minimum across groups implements an OR over interpretations. It also permits a wrong but coherent group to dominate. The pixel scale is fixed, not estimated or calibrated from sensor measurements. The robust minimum-group objective is not a demonstrated likelihood.

In the evaluated candidate, rim-normal residuals are excluded from both training and heldout motion losses. Rims nevertheless participate in training geometry admission. Calling this a rim-free estimator would therefore be incorrect. Both motion families use the same canonical line objective after the implementation’s recorded cost normalization.

## 4.3 Constant-cone and torque-free triaxial families

The constant-cone family is

$$
a ( t ) = h \cos \alpha + \sin \alpha \big ( u \cos ( \Omega t + \phi _ { 0 } ) + v \sin ( \Omega t + \phi _ { 0 } ) \big ) .\tag{8}
$$

The two axis degrees of freedom, cone aperture, frequency and phase give five nominal parameters. Unlike an arbitrary afine image-direction ellipse, this construction enforces a physical circular cone in 3D. The complete bank contains 480 initialized fits. The search is finite and locally optimized; complete bank means all prescribed starts were retained, not that all physical solutions were found.

The triaxial family normalizes $I _ { 3 } = 1$ and sets $I _ { 1 } = J + d / 2 , I _ { 2 } = J - d / 2$ , with $I _ { 1 } \geq I _ { 2 } > 1$ and $0 < d < 1$ . The diference bound enforces the relevant strict triangle inequality. Let

$$
A = 1 / I _ { 1 } , \quad B = 1 / I _ { 2 } , \quad \varepsilon = 2 E / \| H \| ^ { 2 } , \quad b ^ { 2 } = \frac { \varepsilon - A } { 1 - A } , \quad \gamma = \frac { \varepsilon - B } { 1 - B } .\tag{9}
$$

An interior energy fraction selects $\varepsilon \in ( A , B )$ for the crossing (CN) regime or $ { \varepsilon } \in \left( B , 1 \right)$ for the noncrossing (DN) regime. In a momentum-aligned frame, the longitudinal coordinate takes the form

$$
z ( t ) = \left\{ { b \atop s b } { \mathrm { c n } } ( \nu t + \eta _ { 0 } \mid m ) , \quad m = b ^ { 2 } / ( b ^ { 2 } - \gamma ) , \quad \mathrm { C N } , \quad \right. \nonumber\tag{10}
$$

The signed auxiliary quantity $\gamma$ is negative in CN and positive in DN. The scale is $\nu =$ $\| H \| \sqrt { ( 1 - B ) ( 1 - A ) ( b ^ { 2 } - \gamma ) }$ for CN and $\nu = \| H \| \sqrt { ( 1 - B ) ( 1 - A ) b ^ { 2 } }$ for DN. The azimuth satisfies

$$
\dot { \phi } = \| H \| \frac { \varepsilon - z ^ { 2 } } { 1 - z ^ { 2 } } , \qquad a _ { \mathrm { l o c a l } } ( t ) = \left( \sqrt { 1 - z ^ { 2 } } \cos \phi , \sqrt { 1 - z ^ { 2 } } \sin \phi , z \right) ^ { \top } .\tag{11}
$$

A fixed three-parameter rotation maps this path to camera coordinates. Together with energy fraction, positive scale parameter, $^ { J , }$ d and phase fraction, there are eight nominal continuous parameters. The search scale is not itself the reported mean precession rate; the latter is computed from the linked path over the full source span.

The real third-kind integral implements the azimuth using Carlson reduction and explicit period counting [25]. Independent quadrature refinement is checked for the selected triaxial path. The 128 prescribed starts include the two regimes and their specified signs. All outcomes, including optimizer nonconvergence, are retained. These equations encode classical torque-free motion [24]; they do not prove that image observations identify every parameter or exclude trajectories outside the bank.

The search also has fixed parameter bounds and numerical budgets. In particular, its angular-scale ceiling is 30 degrees/s; the triaxial scale parameter is not the reported mean rate. Appendix B specifies the bounds, initializations, solver stages and termination rules. These restrictions are part of the evaluated estimator, rather than a claim to search all torque-free trajectories.

## 4.4 Fixed family selection and separate acceptance

For each family k, the minimum raw-cost fit is selected from its complete bank, including nonconverged fits. The family score is

$$
S _ { k } = N \log ( \operatorname* { m a x } ( L _ { k } , 1 0 ^ { - 1 2 } ) ) + p _ { k } \log N , \qquad p _ { \mathrm { c o n e } } = 5 , \quad p _ { \mathrm { t r i a x i a l } } = 8 .\tag{12}
$$

The canonical factors recorded in the implementation are one for cone and two for the raw triaxial cost. Ordinal ties within a bank retain the first fit; an exact family-score tie selects cone. The form is motivated by dimension-penalized selection [39], but the statistical derivation of a Bayesian information criterion does not apply automatically to correlated frames, a robust OR loss or nominal dimensions near degeneracy. We use it as a frozen engineering rule.

Acceptance cannot alter the selection. If the chosen minimum fails convergence or a required support check, a more convenient runner-up is not substituted. This prevents acceptance-aware fallback from silently changing the estimator being evaluated. It can also lower coverage, especially if numerical convergence diagnostics are conservative or the local search behaves poorly. The study does not isolate that efect causally.

## 4.5 Temporal checks and competing trajectories

The full-data fit first undergoes phase, sampling, geometry, convergence and numerical screening. A failed full-fit gate is suficient for a completed withholding decision; eight temporal blocks may then remain explicitly unexecuted. If the full checks pass, the source calendar is split into eight contiguous blocks. For each block, a fresh complete search fits the complementary training frames and evaluates the held-out block. Selected parameters, training quality, held-out support and retained rivals are all recorded.

Two points are considered materially distinct when their sign-only axis distance is at least $1 0 ^ { \circ }$ or their symmetric rate diference, $2 0 0 | \Omega _ { 1 } - \Omega _ { 2 } | / ( \Omega _ { 1 } + \Omega _ { 2 } )$ , is at least 15%. This engineering comparison of candidates difers from reference-relative rate error used in evaluation. Trainingcompatible distinct rivals must be predictively worse under the frozen rules; unresolved rival optimization can veto an estimate.

All 27 declared gates must be supported before a point is emitted. Their inventory appears in Appendix C. They include the six phase/sampling checks, full and nested convergence, pixel and angular support, positive-ray support, numerical conditioning screens, eight completed blocks, nonempty held-out observations, conditional parameter stability and rival checks. The resulting radii summarize sensitivity within these calculations. They are neither credible intervals nor confidence regions. Appendix C.1 gives their exact dispersion construction and the rival numerical rules. Missing or malformed required evidence produces an incomplete decision rather than an invented gate value.

## 5 Evaluation design and provenance

## 5.1 New synthetic population

The final study fixed seed 2026102501 before generation and registered 96 scenes: 32 axisymmetric core, 16 triaxial CN, 16 triaxial DN and 32 stress scenes. The supported subset is the first 64 scenes. Stress consists of four scenes for each of eight prespecified interventions. Four optical assemblies recur across strata; the scenes are new draws within this small geometry bank, not 96 independently designed target shapes.

Each scene contains 180 images at 768 × 768 pixels. Acquisition frequency is drawn independently of motion from a uniform distribution on [0.9, 1.1] frames/s, using a domain-separated pseudorandom stream. Consequently, source spans are approximately 163–199 seconds. The nominal rotational scale is sampled from [5, 24] degrees/s; it is a generation parameter, not necessarily the evaluated phase-regression mean. Momentum-axis view angles are drawn from $2 5 ^ { \circ } - 7 0 ^ { \circ }$ . The fixed range is sampled from 42–90 m and focal length from 42–62 mm.

The generator uses Blender/Cycles with 32 rendering samples, varying material parameters, solar geometry and exposure. Supported optical assemblies are built from positive-mass primitives with internal sphere pairs used to realize inertia diferences. The four body diameter/length pairs are 3.2/8.1, 4.4/10.8, 3.8/9.8 and 3.6/10.6 m; appendage and rim dimensions vary as well. This is an idealized mass construction, not a measured spacecraft inertia model. There is no added sensor noise or motion blur.

Reference dynamics use a separate split-Hamiltonian rigid-body integrator rather than importing the inference Jacobi path. The generator specifies 128 integration substeps per acquisition interval, doubled for the near-separatrix strata, and eightfold temporal oversampling for diagnostics. Its conservation admission thresholds are $1 0 ^ { - 5 }$ relative energy drift and $1 0 ^ { - 8 }$ relative momentum drift. Source identity and render/runtime bindings are archived. This separation reduces one form of implementation circularity, while leaving common physical assumptions and renderer limitations intact.

Boundary CN/DN cases approach the separatrix; the normalized separation draw changes from [0.15, 0.85] to [0.01, 0.05]. Low-light cases alter illumination and exposure; sparse-rim cases reduce rim geometry. Outside-domain cases introduce biased supplied intrinsics, coherent distractors, noncylindrical deformation or optical/principal-axis misalignment. For the intrinsics intervention, supplied $f _ { x }$ is multiplied by 1.08, $f _ { y }$ by 0.94, and the principal point is shifted by (12, −9) pixels. These are specific interventions, not a representative sample of all possible

deployment shifts.

## 5.2 Isolation, compute and repetitions

Generation, inference and scoring are separate stages. Inference receives opaque scene identifiers, pixels and allowed acquisition metadata; reference states and the scorer are not mounted. Each attempt uses a fresh pinned container with four CPU units, 12 GiB memory, a 7200 s limit, no network, a read-only root and source/input mounts, and only its own output writable. At most four outer attempts run concurrently. Up to four fresh inner fold processes are permitted, with numerical-library threads fixed to one.

The first pass is primary. All first-pass attempts finish before the second begins. Both passes repeat the complete image-to-report computation, rather than only rerunning a score on cached estimates. The 192 attempts are therefore 96 scene pairs, not 192 independent samples of physical accuracy. Missing or failed execution, preparation and evidence remain in the fixed denominator. No case is replaced, retried for a better score or reassigned to a more favorable stratum.

The protocol was internally frozen and committed before rendering, including sources, resource limits, the scientific comparator and the attempted-case inventory. This is an internal prospective registration, not an external registered report or independent institutional audit. The same project developed the generator and estimator. Source separation prevents direct reference input to inference; it does not establish independence of all scientific design choices.

## 5.3 Linked joint metric and outcome accounting

For unit reference and estimated axes, the metric is

$$
e _ { h } = { \frac { 1 8 0 } { \pi } } \operatorname { a r c c o s } \left( \operatorname { c l i p } ( | h ^ { \top } \hat { h } | , 0 , 1 ) \right) , \qquad e _ { \Omega } = 1 0 0 { \frac { | \widehat { \bar { \Omega } } - \bar { \Omega } | } { \bar { \Omega } } } .\tag{13}
$$

The rate metric is used only for an observable positive reference rate. Joint correctness requires $e _ { h } ~ < ~ 1 0 ^ { \circ }$ and $e _ { \Omega } \ < \ 1 5 \%$ on the same linked emitted estimate. An accurate rate from one hypothesis and an accurate axis from another do not form a success. A global depth reflection is not an allowed equivalence.

For n planned scenes, let $C , W , A , F$ count correct estimates, wrong estimates, valid withholding decisions and incomplete executions. The accounting identity is

$$
C + W + A + F = n , \qquad { \mathrm { c o v e r a g e } } = ( C + W ) / n , \qquad { \mathrm { c o r r e c t ~ c o v e r a g e } } = C / n .\tag{14}
$$

The empirical error conditional on emission is $W / ( C + W )$ when the denominator is nonzero. A method can achieve zero observed conditional error by refusing nearly everything; coverage and failure counts are therefore essential descriptive context.

The registered targets were at least $2 0 / 3 2$ correct core estimates, at least $1 6 / 3 2$ combined triaxial estimates, at least $6 / 1 6$ in each triaxial regime, zero wrong supported emissions, zero execution/evidence failures over all 192 attempts and established scientific repetition for every scene. Stress cases remain reported but do not substitute for missing supported successes.

## 5.4 Original failure and descriptive supplement

The original observer chain did not close as a successful validation. A later native, authenticated descriptive scoring supplement accounts for all attempts and produces the counts reported here. It preserves the original completion evidence and failures; it does not retroactively satisfy the registered execution or repeatability targets. The supplement’s score digest is recorded in Appendix D.

The protocol also planned a genuine-v4 comparison on the same 96 scenes. Preparation and watcher-launch evidence exist, but a completed, authenticated same-population baseline score is not present in the closed evidence used by this manuscript. That planned comparison is therefore not reported as a completed experiment. Historical v4 scores belong to a diferent population. This omission limits causal and comparative conclusions; the paper is a candidate case study rather than a completed two-arm benchmark.

## 6 Results

## 6.1 Primary-pass accuracy, withholding and incomplete execution

Table 2 and Figure 2 report every scene in the primary pass. There are 38 correct estimates, zero wrong estimates, 57 valid withholding decisions and one incomplete execution. Correct coverage is 39.6% overall and 39.1% on the supported subset. Since no incorrect point was emitted, emission coverage and correct coverage coincide in this sample. This equality is an empirical result, not a property guaranteed by the gates.

Table 2: Primary pass on the new 25 October population. Incomplete execution is a separate outcome, included in every applicable denominator. Coverage is correct joint estimates divided by planned scenes.
<table><tr><td>Population</td><td>n</td><td>Correct</td><td>Wrong</td><td>Withheld</td><td>Incomplete</td><td>Cov. (%)</td></tr><tr><td>Core axisymmetric</td><td>32</td><td>13</td><td>0</td><td>19</td><td>0</td><td>40.6</td></tr><tr><td>Triaxial CN</td><td>16</td><td>7</td><td>0</td><td>9</td><td>0</td><td>43.8</td></tr><tr><td>Triaxial DN</td><td>16</td><td>5</td><td>0</td><td>10</td><td>1</td><td>31.2</td></tr><tr><td>Supported subtotal</td><td>64</td><td>25</td><td>0</td><td>38</td><td>1</td><td>39.1</td></tr><tr><td>Stress subtotal</td><td>32</td><td>13</td><td>0</td><td>19</td><td>0</td><td>40.6</td></tr><tr><td>All scenes</td><td>96</td><td>38</td><td>0</td><td>57</td><td>1</td><td>39.6</td></tr></table>

![](images/14c13fb75f94fb8e538efd4b5e1f3df553fe5528b4ced5d1d1b6d792342d7a46.jpg)  
Figure 2: Outcome counts by the three supported strata and the stress subtotal. The accepted subset is small despite zero observed wrong emissions.

Only the CN coverage check and the zero-observed-wrong-supported check pass. Core coverage is seven successes below its target; combined triaxial coverage is four below; DN coverage is one below. The two failed attempts violate the failure-free target. One incomplete scene pair prevents the all-scene scientific-repetition target from passing. The registered conjunction therefore fails.

## 6.2 Errors among emitted estimates

Post-closure descriptive analysis of the closed score gives a median sign-only axis error of $0 . 2 4 0 ^ { \circ }$ and a maximum of 1.997<sup>◦</sup> among the 38 emissions. Median rate error is 0.00169%, with maximum 0.03347%. The percentage convention was checked against the frozen metric adapter: the stored rate field includes multiplication by 100. These are selected-subset statistics under the synthetic generator, not errors over all scenes.

Table 3: Errors conditional on emission, primary pass only. Rates are percentage errors, not fractions. These descriptive summaries were computed after study closure; they are not additional registered endpoints.
<table><tr><td>Population</td><td>Emitted</td><td>median (°)  $e _ { h }$ </td><td>max (°)  $e _ { h }$ </td><td>median (%)  $e \Omega$ </td><td>eΩ max (%)</td></tr><tr><td>Core axisymmetric</td><td>13</td><td>0.302</td><td>1.997</td><td>0.00112</td><td>0.00865</td></tr><tr><td>Triaxial CN</td><td>7</td><td>0.254</td><td>1.287</td><td>0.00207</td><td>0.03347</td></tr><tr><td>Triaxial DN</td><td>5</td><td>0.547</td><td>1.246</td><td>0.00255</td><td>0.00620</td></tr><tr><td>Stress</td><td>13</td><td>0.171</td><td>0.599</td><td>0.00279</td><td>0.02018</td></tr><tr><td>All</td><td>38</td><td>0.240</td><td>1.997</td><td>0.00169</td><td>0.03347</td></tr></table>

![](images/a9cbbe022c8d62cdd8d0052c40a26a103791e6e52545673d7a322e5804ff8dd7.jpg)  
Figure 3: The 38 emitted estimates. Axes are deliberately zoomed to show the observed errors; correctness limits are $1 0 ^ { \circ }$ and 15%, well outside this view. The 57 withheld scenes and the incomplete scene have no plotted point estimate.

The small errors deserve neither dismissal nor overinterpretation. Long synthetic sequences without added sensor noise can support precise fits when the selected features agree with the model. At the same time, acceptance selects favorable cases. We cannot assign the accepted subset error distribution to the withheld population, infer sensor-level calibration from it or compare it directly with per-frame attitude errors in the literature.

## 6.3 Stress outcomes

Table 4 keeps all eight interventions visible. All four low-light, all four biased-intrinsics and all four optical-axis-misalignment scenes are withheld. Some separatrix-boundary, sparse-rim, distractor and noncylindrical cases yield correct estimates. Four scenes per intervention are too few to establish a general rejection detector or a robust out-of-domain accuracy rate.

Table 4: Prespecified stress scenes. Correct emissions outside the supported model do not expand that model’s validated domain.
<table><tr><td>Intervention</td><td>n</td><td>Correct</td><td>Wrong</td><td>Withheld</td><td>Incomplete</td></tr><tr><td>Near-boundary CN</td><td>4</td><td>3</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Near-boundary DN</td><td>4</td><td>4</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Low light</td><td>4</td><td>0</td><td>0</td><td>4</td><td>0</td></tr><tr><td>Sparse rims</td><td>4</td><td>3</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Biased intrinsics</td><td>4</td><td>0</td><td>0</td><td>4</td><td>0</td></tr><tr><td>Coherent distractor</td><td>4</td><td>1</td><td>0</td><td>3</td><td>0</td></tr><tr><td>Noncylindrical</td><td>4</td><td>2</td><td>0</td><td>2</td><td>0</td></tr><tr><td>Optical-axis misalignment</td><td>4</td><td>0</td><td>0</td><td>4</td><td>0</td></tr></table>

The stress correct-coverage rate is slightly above the supported rate. This does not imply that the stress population is intrinsically easier or that model violations improve estimation. These are small, fixed strata with diferent nuisance realizations. Neither a matched intervention contrast nor a significance claim is supported by the aggregate table.

## 6.4 Repetition and the incomplete scene

The second pass has the same aggregate outcome counts. The frozen scientific comparison estab lishes equivalence for 95 pairs, establishes no difering pair and leaves one pair non-comparable because both attempts were incomplete. This is not 96/96 reproducibility, and the absence of an established diference does not convert the non-comparable pair into a match.

The incomplete scene, pv2-54f18b60cc016565, belongs to the supported DN stratum. Seven temporal blocks completed. Preparation of zero-indexed block 7 failed with InsufficientPixel-DataError. Both attempts exited with code 2, without timeout, and cleanup was authenticated. The runtime did not establish a valid terminal withholding decision. We therefore retain the attempts as execution failures. A later admission-contract repair would be a new method version, not grounds to rewrite this result.

Scientific equality is defined by the frozen comparator, not by a retrospective numerical tolerance. The policy permits narrow, typed operational diferences such as process identifiers and times while retaining physical parameters, costs, optimizer flags, calendars, decisions, rivals and unknown structured fields. It also excludes three authenticated perception-stage standarderror logs, corresponding to mask, spatial and ellipse extraction, from the scientific view. The raw logs remain retained and binding for raw equality; the exclusion does not apply generally to numerical child outputs. Raw artifact equality is therefore a diferent and stricter notion. The exact comparator and policy accompany the source package. Repetition on one pinned environment also falls short of reproduction by an external research group.

## 6.5 What zero observed errors can support

For intuition only, under an idealized iid Bernoulli model with a fixed accepted-population error probability $p ,$ zero errors in m emissions yields the exact one-sided 95% upper bound

$$
p _ { \mathrm { u p p e r } } = 1 - 0 . 0 5 ^ { 1 / m } .\tag{15}
$$

This is approximately 7.58% for $m = 3 8$ and 11.29% for the 25 supported emissions. Those values are not calibrated uncertainty attached to an individual estimate. More importantly, the actual study has fixed strata, repeated geometries, shared rendering assumptions and selection by a complex gate. We present the calculation as an illustration of finite-sample limitations, not as a certified pooled population or flight-risk bound. Counting the repeated pass as 76 independent emissions would be unjustified.

## 7 Historical corrections and negative controls

## 7.1 An axis metric can erase a physical error

An earlier development score permitted both global sign and a depth reflection. Under sign-only scoring, the reported $1 3 / 1 3$ development result becomes $1 1 / 1 3$ . The diference is not cosmetic. Let $M = \mathrm { d i a g } ( 1 , 1 , - 1 )$ and suppose the estimator returns $\hat { h } = M h$ . A reflection-quotiented metric can assign zero error, whereas the sign-only metric is

$$
d _ { \mathbb { R } \mathbb { P } ^ { 2 } } ( h , M h ) = \operatorname { a r c c o s } | h ^ { \top } M h | = \operatorname { a r c c o s } | 1 - 2 h _ { z } ^ { 2 } | .\tag{16}
$$

It reaches $9 0 ^ { \circ }$ when $h _ { z } ^ { 2 } = 1 / 2$ . Figure 4 illustrates this elementary counterexample. The correct equivalence must follow the intended physical quantity; it cannot be enlarged merely because an estimator confuses branches.

![](images/25a8c289a62c29858396e8e1bc0891efbcf93e3cded357bca81eaac76e0eca8d.jpg)  
Figure 4: Analytical diagnostic, not a new empirical experiment. A depth-reflected axis can have zero error under an overly permissive quotient while being orthogonal as an unoriented camera-frame axis.

## 7.2 Acquisition cadence and opened development data

The earlier 48-video generator selected acquisition cadence using true rotational motion. Its metadata therefore exposed a rate proxy. Correcting the axis convention changed a historical 38/48 score to 25/48, but metric correction alone cannot remove the acquisition confound. Those results are not independent evidence that images identify the rate. The final generator separates cadence from motion and fixes the frame count before inference.

After opening the 24 October population, a successor policy produced 51 correct candidate readouts in an archived-bank development replay. Because those cases had already informed the policy, $5 1 / 9 6$ is development evidence. It is not the result of the new study, not an estimate of generalization and not a substitute for the final 38/96. This distinction applies even if every numerical fit and reference is correctly linked.

Table 5: Evidence populations must not be merged. Historical rows describe their original studies or corrected development scores; they are not additional independent samples for the final candidate.
<table><tr><td>Population</td><td>Recorded result</td><td>Interpretation</td></tr><tr><td>13 development cases</td><td>13/13 becomes 11/13 under sign-only axis</td><td>Old reflection equivalence inflated success; reused development data</td></tr><tr><td>48 earlier videos</td><td>scoring 38/48 becomes 25/48 under strict scoring</td><td>Motion-dependent cadence remains a confound</td></tr><tr><td>24 Oct, 96 scenes, v4</td><td>38 correct, 36 wrong, 22 withheld</td><td>Predecessor on a different population</td></tr><tr><td>24 Oct, physical endpoint</td><td>15 correct, 0 wrong, 63 withheld, 18 timeouts</td><td>Same historical study; registered targets failed</td></tr><tr><td>Opened 24 Oct bank replay</td><td>withheld</td><td>51 correct, 0 wrong, 45 Policy development after outcomes opened</td></tr><tr><td>25 Oct, 96 new scenes</td><td>withheld, 1 incomplete</td><td>38 correct, 0 wrong, 57 Present frozen-candidate descriptive result</td></tr></table>

## 7.3 Feature identity, closure and aliasing

Historical image-direction probes found coherent bias that left a high fit score despite a wrong axis. Under an in-plane rotation applied consistently to every measured direction, an equivariant trajectory fit can rotate its recovered axis while preserving its residual structure. Temporal prediction from those same directions then reproduces the bias. This is a measurement-identification problem, not simply an optimizer robustness problem.

Specific extractor controls made the problem concrete: internal edge populations could imitate a longitudinal direction, and a long-segment rule did not eliminate coherent long texture. An earlier afine harmonic fit also admitted a direction ellipse whose ofset was incompatible with a physical cone. Circular harmonic arms must be orthogonal and equal in 3D, and their ofset must align with their normal. Satisfying only the arm constraints is insuficient. The current physical-family construction addresses that particular model inconsistency, while leaving feature identity unresolved.

Regular sampling introduces another ambiguity. A historical projected-direction example used a true rate of $2 5 . 9 0 5 8 6 ^ { \circ } / \mathrm { s }$ with 20 s sample spacing and produced 7.90586<sup>◦</sup>/s with a high diagnostic score. The later regular-lattice guard checks retained timestamps and conservatively excludes an unsigned search range extending beyond 90/∆t degrees/s. A tolerant lattice-gap calculation handles missing ticks. This engineering guard addresses the demonstrated alias; it is not a theorem of uniqueness for irregular samples or all physical trajectories.

These controls were conducted on historical development inputs or analytical observations. They explain failure mechanisms, but they do not add cases to the final prospective denominator or establish how often those mechanisms occur in flight imagery.

## 7.4 Numerical improvement is not physical improvement

A separate paired control restarted 36 already converged checkpoints: four full fits and 32 training fits from four development sequences. Scalar and vector residual formulations produced 72 completed endpoints. Vector residuals reduced canonical SSE in every paired comparison, but the median relative decrease from the original checkpoint was only $7 . 8 5 \times 1 0 ^ { - 9 }$ and the maximum was $2 . 8 8 \times 1 0 ^ { - 7 }$ . Objective callbacks increased by 30.7%. Maximum axis displacement was 0.001102<sup>◦</sup>.

Those are local numerical efects on correlated checkpoints. They rescue no failed original winner, establish no reduction in error against truth and supply no new coverage result. The evaluated candidate was not changed on that basis. Preserving this distinction avoids using an optimizer’s success flag or a tiny objective decrease as a surrogate for scientific progress.

## 8 Discussion

## 8.1 What is learned

The final experiment demonstrates that the implemented measurement and decision chain can produce precise, repeatable joint estimates for a subset of new synthetic scenes while withholding many others. It also shows that the chosen policy fails its own minimum-coverage and execution requirements. Both observations are needed: reporting only emitted precision would overstate usefulness, while reporting only the failed target would conceal the functioning subset.

The contribution with the clearest support is methodological and empirical. The same project records a progression from permissive evaluation and metadata confounding to a stricter, failurepreserving experiment. The endpoint objective, linked estimate identity, fixed family selector and explicit incomplete state make that experiment inspectable. We do not establish that this combination is the first of its kind, that every gate is necessary or that a simpler baseline would perform worse.

The gap between low accepted error and low coverage suggests several possible bottlenecks: missing valid image structure, overly restrictive admission, optimizer nonconvergence, unresolved competing fits and conservative temporal checks. The current aggregate score does not separate their causal contributions. Relaxing gates after seeing the new cases might improve a replay count but would reopen development; another untouched population would be needed for confirmation.

## 8.2 Four distinct notions of confidence

Physical correctness, numerical convergence, temporal stability and execution reproducibility answer diferent questions. Convergence concerns a numerical stopping rule. Temporal stability concerns sensitivity to a particular partition and measurement representation. Reproducibility concerns whether the same declared computation yields equivalent scientific outputs. Physical correctness concerns the target quantity relative to a valid reference. None of the first three alone implies the fourth.

The same separation applies to geometry. A positive-ray fit is more constrained than an arbitrary image line, but still presumes that the line belongs to the object structure the model describes. A finite list of surviving alternatives is useful evidence about the search that was performed. It does not certify that no unsearched alternative, diferent shape or illuminationdependent edge could explain the images.

## 8.3 Threats to validity

Internal validity. The reference dynamics are implemented separately from inference, and labels are absent from inference mounts. Nevertheless, both components share the torque-free modeling premise and project-developed geometry conventions. Software tests and source hashes can bind an implementation; they cannot independently validate the physical meaning of every feature or reference.

Selection and preprocessing. The final population was generated after the candidate freeze, but the method was developed extensively on earlier cases. The 27-gate conjunction embodies those choices. Shared whole-video preprocessing also means temporal checks cannot be interpreted as independent held-out pixel tests. Scene-level prospective evaluation is the relevant protection against directly reusing the final outcomes; it does not eliminate shared model misspecification.

Statistical precision. There are only 96 scene draws from four assemblies and only four cases per stress intervention. The repeated pass is not another population. No calibrated probability model for accepted error has been fitted or validated. The small conditional-error summaries should not be extrapolated to a deployment distribution.

External validity. There are no real sensor images, motion blur, injected sensor noise, flexible appendages, uncontrolled camera motion, sustained external torques or general noncylindrical targets in the supported experiment. Unknown real inertia and material properties are replaced by an idealized construction. No capture, collision-avoidance or onboard real-time claim follows.

Comparative validity. The planned same-population predecessor comparison has no completed score in the closed evidence used here, and no external estimator was run. Diferent historical populations cannot identify the efect of the new policy. Literature methods often estimate full pose under diferent sensing or target assumptions; their published errors are not interchangeable with our metric.

Reproduction and availability. The scientific repetition used a pinned environment within the project. It was not performed by an independent laboratory. The source package includes the original closed score in losslessly compressed form, the protocol, the attempted-case registration and frozen scientific source excerpts. At this manuscript version, there is no public archival DOI for the complete image-to-report evidence. The supplied score and figure-generation source support checking the descriptive arithmetic. Archived reports and reference trajectories also permit recomputation of the physical errors. Repeating image-to-report inference still requires the omitted images, runtime and remaining dependencies.

## 8.4 Experiments that would materially strengthen the claim

A credible next evaluation would first execute a matched baseline on a new or still-unopened population with an explicit common estimand. Useful arms include a constant-cone-only policy, the fixed triaxial family, the complete selector and an adapted published rotational-state method. Each arm should preserve the same pixels, calibration, resource budget and incomplete-attempt accounting. Post-hoc oracle family selection should be reported only as an oracle diagnostic.

Feature identity requires a separate intervention. Annotated generator, rim and texture correspondences could distinguish absence of a correct candidate from a selector choosing the wrong candidate. Matched renders with controlled distractors, followed by laboratory imagery of a rotating cylindrical mockup, would test whether temporal consistency survives realistic specular changes. Pixel-fold-local preprocessing would be needed before claiming end-to-end temporal holdout.

Uncertainty work should use a separate calibration split and an untouched test split, state the population and exchangeability assumptions, and report risk versus coverage under a frozen selection rule. A heuristic gate score must not be renamed a probability. For a sequential navigation application, causal processing, camera-motion compensation, latency and failure recovery would also need evaluation.

These are proposed experiments, not results or commitments fulfilled by the present study. Their value is that each would resolve a specific inferential gap instead of merely increasing a development score.

## 9 Conclusion

ORBIT-SCOUT provides a selective, inspectable implementation for a restricted monocular precession problem. A new 96-scene synthetic population produced 38 correct joint estimates, no wrong emissions, 57 withholding decisions and one incomplete execution in the primary pass. Scientific repetition was established for 95 of 96 scene pairs. The registered coverage and complete-execution requirements were not met.

The study supports an auditable case report with reproducible descriptive arithmetic and internal scientific repetition. It does not support state-of-the-art superiority, calibrated confidence, general identifiability or flight readiness. Its most transferable result is the need to preserve the distinction between a fit, an accepted estimate, a withheld answer and a computation that never established a valid answer.

## Data, code and authoring statement

The manuscript source package includes the per-attempt outcome/error table, the hash-pinned original score compressed with gzip, derivation and verification scripts, generated figures, bibliography, a literature ledger and an evidence map. It also contains the original protocol, attemptedcase registration and frozen source excerpts for the measurement, fitting and decision computations. The exact scoring adapter, metric helpers, scientific comparator, comparison policy and completion seal are included for inspection. These materials reproduce the descriptive arithmetic and permit inspection of the stated implementation. A separate archive supplies all 96 original reference files, 192 reports and 192 execution receipts. Its verifier reproduces physical scoring from these originals; it does not rerun image inference. They do not contain the complete inference runtime or all rendered images, and no public data DOI is claimed. The arXiv source archive carries this material in its ancillary reproducibility package.

AI tools assisted software development, literature discovery, evidence inspection and drafting. Automated adversarial reviews were software-agent reviews, not human peer review. Carlos Eduardo Sanoja is the sole named author and retains scientific responsibility and the decision to submit.

## A Endpoint correction and its limitations

This derivation documents the objective used by the implementation. It is a reduction of orthogonal line fitting and constrained ray fitting; no priority claim is made for the underlying mathematics.

$q _ { z } \neq 0$ , define the finite vanishing point $v = q _ { x y } / q _ { z }$ and centered endpoints $x _ { j } = p _ { j } - v$ Let n be a unit normal to a candidate line through v. The minimum squared correction of the two endpoints to that line is

$$
\operatorname* { m i n } _ { \| \boldsymbol { n } \| = 1 } \sum _ { j = 1 } ^ { 2 } ( \boldsymbol { n } ^ { \top } \boldsymbol { x } _ { j } ) ^ { 2 } = \lambda _ { \operatorname* { m i n } } \big ( x _ { 1 } x _ { 1 } ^ { \top } + x _ { 2 } x _ { 2 } ^ { \top } \big ) .\tag{17}
$$

Since $d _ { j } = - q _ { z } x _ { j }$ , the matrix in the eigenvalue expression is $S / q _ { z } ^ { 2 }$ . In two dimensions, $\lambda _ { \operatorname* { m i n } } ( S ) =$ det $( S ) / \lambda _ { \mathrm { m a x } } ( S )$ , and det $\begin{array} { r } { ( S ) = \operatorname* { d e t } [ d _ { 1 } , d _ { 2 } ] ^ { 2 } = q _ { z } ^ { 2 } ( \ell ^ { \top } q ) ^ { 2 } } \end{array}$ . Substitution $\mathrm { g i }$ ves Equation (4) without subtracting nearly equal eigenvalues. Distinct endpoints and nonzero $q$ exclude the case $\lambda _ { \operatorname* { m a x } } ( S ) = 0$ . At $q _ { z } = 0 , d _ { 1 } = d _ { 2 } = q _ { x y }$ , so

$$
d _ { \mathrm { l i n e } } ^ { 2 } = \frac { ( \ell ^ { \top } q ) ^ { 2 } } { 2 \| q _ { x y } \| ^ { 2 } } .\tag{18}
$$

This equals the sum of endpoint corrections to their best line of fixed direction $q _ { x y }$ , with the line ofset fitted through their midpoint.

For the common-ray constraint, use a unit direction u and nonnegative endpoint distances $\alpha _ { 1 } , \alpha _ { 2 }$ . The problem is

$$
\operatorname* { i n f } _ { \| u \| = 1 , \ \alpha _ { j } \geq 0 } \sum _ { j } \big \| x _ { j } - \alpha _ { j } u \big \| ^ { 2 } .\tag{19}
$$

For fixed $u , \alpha _ { j } = \operatorname* { m a x } ( 0 , u ^ { \top } x _ { j } )$ . If $x _ { 1 } ^ { \top } x _ { 2 } > 0$ , the principal direction can project both vectors with the same sign and the unconstrained line optimum is feasible. If $x _ { 1 } ^ { \top } x _ { 2 } \leq 0$ , the optimum has one endpoint at the origin of the ray; orienting toward the longer vector leaves squared cost min $( \| x _ { 1 } \| ^ { 2 } , \| x _ { 2 } \| ^ { 2 } )$ . Rescaling yields Equation (5). Equality of the dot product lies on the branch boundary; the formula remains continuous.

The nonnegative-distance problem includes the ray origin. For an actual segment with distinct finite-depth endpoints, attaining that origin can be physically inadmissible; the result is then only an infimum. This is why the objective alone cannot replace positive-ray support checks. It also cannot establish that the endpoints arose from the same physical generator, or that one pixel is an appropriate noise scale. Perspective incidence is exact under the model; its use as a statistical loss is an additional assumption.

## B Frozen numerical search envelope

These controls describe the evaluated source, not recommended general-purpose defaults. Pixel preparation subtracts the first source timestamp, and the physical runtime requires a zero-based source calendar. Let $T = t _ { F - 1 }$ be the resulting full-source span and $\Delta T _ { \mathrm { T R A I N } }$ the span of source timestamps in the declared TRAIN index set. In an end-block fold these spans can difer.

## B.1 Cone bank

The 480 starts use 240 uniformly spaced initial frequencies and two depth lifts. Frequency is bounded by

$$
\operatorname* { m a x } ( 0 . 5 , 3 6 0 / \Delta T _ { \mathrm { T R A I N } } ) \le \Omega \le 3 0 \quad \mathrm { d e g r e e s / s } .\tag{20}
$$

The aperture lies in $[ 1 0 ^ { - 6 } , \pi - 1 0 ^ { - 6 } ]$ radians, while the three rotation-vector coordinates are unbounded. Frequency is scaled by $3 6 0 / \Delta T _ { \mathrm { T R A I N } }$ . Each fit sets the L-BFGS-B iteration and objective/gradient-bundle limits to 350, with a line-search limit of 30, gradient tolerance $1 0 ^ { - 6 }$ and function tolerance $1 0 ^ { - 1 2 }$ . The finite-diference step is $1 0 ^ { - 5 }$ in the scaled coordinates, with one-sided diferences at bounds. A bundle includes derivative work; these are configured solver limits, and 350 is not a hard bound on all scalar loss callbacks. Line-search and finite-diference work are recorded separately.

The cone numerical-termination gate accepts either the solver success flag or a nonnegative canonical loss at most $1 0 ^ { - 2 0 }$ . This floor rule is not a stationarity or accuracy certificate. Selected triaxial fits instead retain their solver success flag. These implementation-specific meanings qualify the word “converged” in gate names.

## B.2 Triaxial bank

Up to eight diverse cone parents supply initializations. The seed construction explores $J \in$ $\{ 2 , 4 , 8 \} , d \in \{ 0 . 2 , 0 . 8 \}$ , both regimes, their specified signs and four phases per regime period; the frozen ranking/diversity procedure chooses 128 starts. The continuous energy fraction is bounded by $[ 1 0 ^ { - 5 } , 1 - 1 0 ^ { - 5 } ]$ , J by [1.51, 12], d by [0.001, 0.999] and phase fraction by $[ - 3 , 3 ]$ Rotation-vector coordinates are unbounded. The scale r satisfies

$$
\operatorname* { m a x } ( 0 . 5 , 3 6 0 / T ) \leq r \leq 3 0 \quad \mathrm { d e g r e e s } / \mathrm { s } , \qquad \| H \| = ( \pi / 1 8 0 ) r J .\tag{21}
$$

Thus, the bound on $r$ is not a bound of 30 degrees/s on the reported mean precession of every triaxial trajectory. The eight parameter scales are $( 1 , 1 , 1 , 0 . 3 , 3 6 0 / T , 3 , 0 . 5 , 0 . 5 )$

SciPy’s trust-region reflective least-squares solver first uses a 350-evaluation budget and function, step and gradient tolerances of $1 0 ^ { - 9 }$ . After rotation-vector canonicalization, a second 350-evaluation stage uses 10<sup>−10</sup> tolerances. If the second stage reports nonconvergence, a predefined local refinement allows 1000 evaluations at the latter tolerances and is adopted only if its raw cost is no greater than the previous cost plus 10<sup>−12</sup> max(1, previous cost). These are solvercounter budgets, not total scalar residual-call or image-processing budgets; numerical Jacobian evaluations can require additional residual calls. They do not guarantee convergence. The fixed internal refinement belongs to each original attempt; it is distinct from replacing or rerunning a scene after its outcome is known. Frozen source excerpts expose the exact seed ordering, cost definitions and callback accounting.

## C Gate inventory and decision semantics

Table 6 lists all 27 gates in the evaluated decision policy. Names are shortened for readability; their exact machine-readable names and the family-selection policy accompany the source package. A full-fit rejection can terminate before temporal searches, but an emitted point requires every applicable check. A missing diagnostic is not a negative measurement and cannot be converted into a valid withholding decision without the runtime establishing that contract.

Table 6: Complete gate inventory, grouped by execution stage.
<table><tr><td>No.</td><td>Gate</td><td>Role</td></tr><tr><td>1</td><td>Resolved rate</td><td>Rate resolution under the frozen engineering rule</td></tr><tr><td>2</td><td>Source sampling</td><td>Acquisition sampling support</td></tr><tr><td>3</td><td>Resolved cycle</td><td>Sufficient resolved temporal cycle</td></tr><tr><td>4</td><td>Camera alignment</td><td>Supported camera geometry</td></tr><tr><td>5</td><td>Cone aperture</td><td>Nondegenerate axis excursion</td></tr><tr><td>6</td><td>Phase coverage</td><td>Admitted phase support</td></tr><tr><td>7</td><td>Full winner converged</td><td>Selected minimum&#x27;s numerical-termination rule</td></tr><tr><td>8</td><td>Full pixel support</td><td>Selected fit&#x27;s training endpoint support</td></tr><tr><td>9</td><td>Full angular support</td><td>Selected fit&#x27;s training direction support</td></tr><tr><td>10</td><td>Full positive-ray support</td><td>Strong selected generator-ray support</td></tr><tr><td>11</td><td>Full quadrature converged</td><td>Triaxial path refinement below  $1 0 ^ { - 3 } ;$  explicitly not applicable to analytic cone</td></tr><tr><td>12</td><td>Full numerical screen</td><td>Local numerical parameter screening</td></tr><tr><td>13</td><td>Eight complete folds</td><td>All eight actual temporal contexts available</td></tr><tr><td>14</td><td>Nested training converged</td><td>Every selected fold satisfies its numerical- termination rule</td></tr><tr><td>15</td><td>Nested line quality</td><td>Pixel and angular training checks in all folds</td></tr><tr><td>16</td><td>Nested parameter stability</td><td>Fold/full linked points not materially distinct</td></tr><tr><td>17</td><td>Nested quadrature converged</td><td>Refinement rule in each applicable fold</td></tr><tr><td>18</td><td>Nested numerical screen</td><td>Parameter screen in each fold</td></tr><tr><td>19</td><td>Nested training rays</td><td>Strong positive-ray support in each training con- text</td></tr><tr><td>20</td><td>Nonempty held-out blocks</td><td>Held-out line evidence in every block</td></tr><tr><td>21</td><td>Pooled held-out pixels</td><td>Endpoint quality across held-out predictions</td></tr><tr><td>22</td><td>Pooled held-out angles</td><td>Direction quality across held-out predictions</td></tr><tr><td>23</td><td>Held-out rays</td><td>Positive-ray support in every held-out block</td></tr><tr><td>24</td><td>Conditional axis radius</td><td>Computed stability radius below  $1 0 ^ { \circ }$ </td></tr><tr><td>25</td><td>Conditional rate radius</td><td>Computed stability radius below 15%</td></tr><tr><td>26</td><td>Distinct rivals worse</td><td>Training-compatible distinct alternatives worse</td></tr><tr><td>27</td><td>Rival optimization</td><td>on held-out data Nested rivals satisfy declared optimizer rules</td></tr></table>

## C.1 Conditional dispersion and rival contrasts

After all $m _ { f } = 8$ folds return valid linked targets, the nominal target is $( \hat { h } _ { 0 } , \hat { \Omega } _ { 0 } )$ and the fold targets are $( \hat { h } _ { j } , \hat { \Omega } _ { j } ) , j = 1 , \dots , m _ { f }$ . Every rate is evaluated over the same full source calendar. Each unit fold axis is sign-aligned with the nominal axis:

$$
\tilde { h } _ { j } = \operatorname * { a r g m a x } _ { x \in \{ \hat { h } _ { j } , - \hat { h } _ { j } \} } { x ^ { \top } \hat { h } _ { 0 } } , \qquad \bar { h } _ { b } = \frac { 1 } { m _ { f } } \sum _ { j } \tilde { h } _ { j } , \qquad \bar { \Omega } _ { b } = \frac { 1 } { m _ { f } } \sum _ { j } \hat { \Omega } _ { j } .\tag{22}
$$

The vector mean $\bar { h } _ { b }$ is not renormalized. The implementation forms the block-dispersion quantities

$$
C _ { h } = \frac { m _ { f } - 1 } { m _ { f } } \sum _ { j } ( \tilde { h } _ { j } - \bar { h } _ { b } ) ( \tilde { h } _ { j } - \bar { h } _ { b } ) ^ { \top } , \qquad s _ { h } = \sqrt { \operatorname* { m a x } ( 0 , \lambda _ { \operatorname* { m a x } } ( C _ { h } ) ) } ,\tag{23}
$$

$$
s _ { \Omega } ^ { 2 } = \frac { m _ { f } - 1 } { m _ { f } } \sum _ { j } ( \hat { \Omega } _ { j } - \bar { \Omega } _ { b } ) ^ { 2 } .\tag{24}
$$

Let $t _ { \nu , p }$ denote the p quantile of Student’s t distribution with ν degrees of freedom. With the fixed nominal multiplier parameter $\alpha _ { \mathrm { n o m } } = 0 . 0 1$ , the reported diagnostics are

$$
R _ { h } = { \frac { 1 8 0 } { \pi } } \sqrt { 2 } t _ { m _ { f } - 1 , 1 - \alpha _ { \mathrm { n o m } } / 4 } s _ { h } , \qquad R _ { \Omega } = 1 0 0 \frac { t _ { m _ { f } - 1 , 1 - \alpha _ { \mathrm { n o m } } / 2 } s _ { \Omega } } { \hat { \Omega } _ { 0 } } .\tag{25}
$$

The gates require $R _ { h } < 1 0$ degrees and $R _ { \Omega } < 1 5 \%$ . The axis calculation converts ambient-vector dispersion into degrees; it is not an exact spherical confidence construction. The overlapping training sets and shared upstream features make the fold estimates correlated. In particular, $\alpha _ { \mathrm { n o m } } = 0 . 0 1$ is a fixed multiplier choice, not evidence of calibrated 99% coverage. These quantities are descriptive conditional stability diagnostics.

In each fold, let $w$ denote its TRAIN-selected linked point and $^ r \mathrm { ~ \textmd ~ { ~ a ~ } ~ }$ retained, materially distinct rival marked TRAIN-compatible by the frozen quality rules. Canonical training and held-out losses use the cost normalization in Equation (7). Every such rival must satisfy

$$
L _ { r } ^ { \mathrm { h e l d } } - L _ { w } ^ { \mathrm { h e l d } } > 0 , \qquad c _ { r } \vee \left[ L _ { r } ^ { \mathrm { T R A I N } } \geq 9 \operatorname* { m a x } ( L _ { w } ^ { \mathrm { T R A I N } } , 1 0 ^ { - 1 2 } ) \right] ,\tag{26}
$$

where $c _ { r }$ is the rival’s declared numerical-termination flag under Appendix B. Thus a nonconverged rival is permitted by this rule only when its training loss meets or exceeds the stated ninefold threshold. The held-out comparison has no ratio margin, efect-size threshold or population p-value. Both conditions must hold for every required rival in every fold. If there is no such retained rival, the universal check passes vacuously; this means only that none was found in the finite search. Missing or malformed required held-out evidence produces an incomplete comparison rather than a passing contrast.

The dimension penalty does not make frames independent. If two families have losses $L _ { T } , L _ { C }$ 2 their selection-score diference is

$$
\Delta S = N \log ( L _ { T } / L _ { C } ) + 3 \log N\tag{27}
$$

away from the loss floor. Duplicating every frame r times leaves the mean losses unchanged but changes this diference to $r N \log ( L _ { T } / L _ { C } ) + 3 \log ( r N )$ . The selected family can change although no new independent information has been acquired. This simple consequence is one reason the penalty is described as a heuristic and N is not called an efective sample size.

## D Evidence identity and reproduction boundaries

The final score and original protocol are distinct artifacts. The following identities bind the closed result used in this manuscript:

Descriptive score, SHA-256

8de4a733955f5c721f5f81a1321c6ccd1fb92e4f55565e27331fd40d6063a98f

Descriptive supplement completion, SHA-256

810f4009df3d0436272f3a984ea0be8407ab2f455a4149a5e8b7aaa397868dd4

Original reproduction completion, SHA-256

9454e344c8728f61bcef7d50445d29b9d79024296446cbeafe9a41d381f16aa7

## Inference OCI image identity, SHA-256

82121b44e7d1cc2051f30642a8ff8fbf1ae9b4c5c8d64fed5f0eac4cb38a7e39

## Registered generator content identity, SHA-256

c0bb0a6560d56b36832220af0b74469264da2a4a20eb8d96eb750e31737e4adb

Hashes support identity checks only when the corresponding artifacts are available. They are not proof of scientific correctness or substitutes for an accessible archive. The original descriptive score is included as a gzip file; decompression recovers the exact bytes bound by the first digest above. The compact CSV has one row for every scene and repetition; missing point errors for withheld or incomplete outcomes remain empty, not zero. Its source digest is checked by the derivation script before extracting the rows from the original score.

The source package separates four reproduction levels:

1. Manuscript build. Compile the supplied LAT<sub>E</sub>X and bibliography using the included figures and tables. This verifies the document, not the experiment.

2. Descriptive arithmetic. Run the supplied Python script on the compact CSV or the included compressed original score. This rebuilds counts, conditional-error summaries and plots without inference. A separate verifier checks all 192 CSV rows against the original score, source identities, citation coverage and elementary geometry controls.

3. Physical metric reproduction. The companion original-input archive contains 96 reference files, 192 original reports and 192 execution receipts. A hash-bound verifier extracts the archived reference targets and recomputes sign-only axis and percentage rate errors using the frozen metric functions. All 76 emitted-attempt error pairs reproduced the archived floating-point values exactly in the checked environment; all 192 outcome labels agree. The separate 192-row point export preserves unavailable values and links back to the originals. This post-closure check is a reproduction of scoring, not a new accuracy population or image-inference run.

4. Image-to-report reproduction. Obtain the pinned runtime, frozen candidate, 96-scene inputs and evaluation artifacts. The compact package alone cannot perform this level. A new execution must preserve the original attempted-case manifest and comparison policy, and must report failures rather than selecting successful runs.

The project evidence map supplies repository-relative locations for the protocol, generator, frozen measurement code, failure record, historical corrections and numerical control. A separate source map binds the exact excerpts distributed with the package. Some original project locations remain local; the maps do not imply that every historical archive is included. No access key or authenticated link token is embedded in the manuscript or source package.

## E Claim boundaries for reading and reuse

Table 7: What the evidence supports and what remains unestablished.
<table><tr><td>Supported within this study</td><td>Not established</td></tr><tr><td>38 correct linked estimates on the primary</td><td>Error probability of zero, or a calibrated</td></tr><tr><td>synthetic pass 25/64 correct supported coverage</td><td>per-scene confidence Registered minimum coverage or general target</td></tr><tr><td>95 scientifically equivalent scene pairs</td><td>recovery 96/96 repetition, cross-platform equality or</td></tr><tr><td>No reference data mounted in inference</td><td>independent-laboratory reproduction</td></tr><tr><td>Physical cone and torque-free triaxial trajectories</td><td>Independence of all generator and estimator assumptions</td></tr><tr><td>Complete prescribed search banks retained</td><td>Correct physical identity of every extracted edge</td></tr><tr><td>Cold temporal numerical searches</td><td>Exhaustive search or global identifiability Pixel-isolated end-to-end holdout</td></tr><tr><td>Correct estimates on some stress cases</td><td>A validated expansion beyond the stated</td></tr><tr><td></td><td>supported domain</td></tr><tr><td>Tiny SSE decreases in a separate numerical control</td><td>Improved physical accuracy or coverage</td></tr></table>

## References

[1] Sumant Sharma, Jacopo Ventura, and Simone D’Amico. Robust model-based monocular pose initialization for noncooperative spacecraft rendezvous. Journal of Spacecraft and Rockets, 55(6):1414–1429, 2018. doi: 10.2514/1.A34124. URL https://sumantsharma.net/assets/pdf/jsr2017\_sharmaventuradamico.pdf.

[2] Sumant Sharma and Simone D’Amico. Neural network-based pose estimation for noncooperative spacecraft rendezvous. IEEE Transactions on Aerospace and Electronic Systems, 56(6):4638–4658, 2020. doi: 10.1109/TAES.2020.2999148. URL https://arxiv.org/abs/1906.09868.

[3] Vincenzo Capuano, Kyunam Kim, Alexei Harvard, and Soon-Jo Chung. Monocular-based pose determination of uncooperative space objects. Acta Astronautica, 166:493–506, 2020. doi: 10.1016/j.actaastro.2019.09.027. URL https://authors.library.caltech.edu/records/bxxzy-w9k62.

[4] Hao Chi, Yu Liu, Ken Chen, Weichun Feng, and Jiwen Zhang. Monocular method for parameter estimation of symmetric-inertia uncooperative targets. Journal of Tsinghua University (Science and Technology), 62(9):1492–1499, 2022. doi: 10.16511/j.cnki.qhdxxb.2022.21.020. URL https://www.sciopen.com/article/10.16511/j.cnki.qhdxxb.2022.21.020. Article in Chinese; English abstract.

[5] Daigo Kobayashi and Vakhtang Putkaradze. Estimation of spacecraft inertia tensor using attitude-only data from torque-free motion. arXiv:2608.08886v1, preprint, 2026. URL https://arxiv.org/abs/2608.08886.

[6] Dezhi Zhang, Guocai Yang, Junhong Ji, Shaowei Fan, Minghe Jin, and Qiang Liu. Pose measurement and motion estimation of non-cooperative satellite based on spatial circle feature. Advances in Space Research, 71(3):1721–1734, 2023. doi:

10.1016/j.asr.2022.09.027. URL https://www.sciencedirect.com/science/article/pii/S0273117722008687.

[7] Roberto Opromolla, Giancarmine Fasano, Giancarlo Rufino, and Michele Grassi. A review of cooperative and uncooperative spacecraft pose determination techniques for close-proximity operations. Progress in Aerospace Sciences, 93:53–72, 2017. doi: 10.1016/j.paerosci.2017.07.001. URL https://www.sciencedirect.com/science/article/abs/pii/S0376042117300428.

[8] Leo Pauly, Wassim Rharbaoui, Carl Shneider, Arunkumar Rathinam, Vincent Gaudillière, and Djamila Aouada. A survey on deep learning-based monocular spacecraft pose estimation: Current state, limitations and prospects. Acta Astronautica, 212:339–360, 2023. doi: 10.1016/j.actaastro.2023.08.001. URL https://arxiv.org/abs/2305.07348v3.

[9] Mate Kisantal, Sumant Sharma, Tae Ha Park, Dario Izzo, Marcus Märtens, and Simone D’Amico. Satellite pose estimation challenge: Dataset, competition design, and results. IEEE Transactions on Aerospace and Electronic Systems, 56(5):4083–4098, 2020. doi: 10.1109/TAES.2020.2989063. URL https://arxiv.org/abs/1911.02050.

[10] Pedro F. Proença and Yang Gao. Deep learning for spacecraft pose estimation from photorealistic rendering. In IEEE International Conference on Robotics and Automation, pages 6007–6013, 2020. URL https://arxiv.org/abs/1907.04298.

[11] Tae Ha Park and Simone D’Amico. Adaptive neural-network-based unscented kalman filter for robust pose tracking of noncooperative spacecraft. Journal of Guidance, Control, and Dynamics, 46(9):1671–1688, 2023. doi: 10.2514/1.G007387. URL https://arxiv.org/abs/2206.03796.

[12] André Lopo, Atabak Dehban, and Rodrigo Ventura. HAT: Hypothesis-anchored tracking for video monocular spacecraft pose estimation. arXiv:2609.21597v1, preprint, 2026. URL https://arxiv.org/abs/2609.21597.

[13] Abid Ali, Arunkumar Rathinam, and Djamila Aouada. TALON: Token-aligned lightweight adapters for 6-DoF spacecraft pose estimation. arXiv:2605.31217v1, preprint, 2026. URL https://arxiv.org/abs/2605.31217.

[14] Tae Ha Park and Simone D’Amico. Rapid abstraction of spacecraft 3D structure from single 2D image. In AIAA SciTech Forum, 2024. doi: 10.2514/6.2024-2768. URL https://slab.stanford.edu/sites/g/files/sbiybj25201/files/media/file/ scitech\_park\_damico\_v5.pdf.

[15] Antoine Legrand, Renaud Detry, and Christophe De Vleeschouwer. Leveraging neural radiance fields for pose estimation of an unknown space object during proximity operations. arXiv:2405.12728v3; ISpaRo workshop contribution, 2024. URL https://arxiv.org/abs/2405.12728.

[16] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6D pose estimation and tracking of novel objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868–17879, 2024. doi: 10.1109/CVPR52733.2024.01692. URL https://arxiv.org/abs/2312.08344.

[17] Josiane Uwumukiza, Jocelyn Zhao, Giovanni Lavezzi, Giacomo Battaglia, Paolo Panicucci, Minduli C. Wijayatunga, Victor Rodriguez-Fernandez, and Richard Linares. DreamSat-Pose: Spacecraft pose estimation from single-view 3D reconstructions and

learned 2D-3D feature matching. arXiv:2607.13449v2, preprint, 2026. URL https://arxiv.org/abs/2607.13449.

[18] Pol Francesch Huc and Simone D’Amico. Monocular navigation relative to unknown spacecraft using a transformer-aided kalman filter. arXiv:2610.07231v1, preprint, 2026. URL https://arxiv.org/abs/2610.07231.

[19] Qiyang Hu, Shunan Wu, Fanchen Meng, and Zhigang Wu. Relative motion estimation algorithm for noncooperative targets considering multiple solutions of rotational parameters. Sensors, 24(6):1811, 2024. doi: 10.3390/s24061811. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC10975905/.

[20] Paolo Panicucci, Jérémy Lebreton, Roland Brochard, Emmanuel Zenou, and Michel Delpech. Vision-based estimation of small body rotational state. Acta Astronautica, 213: 177–196, 2023. doi: 10.1016/j.actaastro.2023.08.046. URL https://arxiv.org/abs/2302.11364.

[21] Damien Mariyanayagam, Pierre Gurdjos, Sylvie Chambon, Florent Brunet, and Vincent Charvillat. Pose estimation of a single circle using default intrinsic calibration. arXiv:1804.04922, 2018. URL https://arxiv.org/abs/1804.04922.

[22] Vincenzo Pesce, Michèle Lavagna, and Riccardo Bevilacqua. Stereovision-based pose and inertia estimation of unknown and uncooperative space objects. Advances in Space Research, 59(1):236–251, 2017. doi: 10.1016/j.asr.2016.10.002. URL https://re.public. polimi.it/retrieve/handle/11311/1007707/430992/PESCV\_OA\_01-17.pdf.

[23] Batu Candan and Simone Servadio. Online inertia tensor identification for non-cooperative spacecraft via augmented UKF. In AIAA Region V Student Conference, 2026. doi: 10.2514/6.2026-108993. URL https://arxiv.org/abs/2603.27361.

[24] Ramses van Zon and Jeremy Schofield. Numerical implementation of the exact dynamics of free rigid bodies. Journal of Computational Physics, 225:145–164, 2007. doi: 10.1016/j.jcp.2006.11.019. URL https://arxiv.org/abs/cond-mat/0607529.

[25] National Institute of Standards and Technology. Digital library of mathematical functions, section 19.25: Relations to other functions, 2026. URL https://dlmf.nist.gov/19.25. Equation 19.25.14; accessed 25 October 2026.

[26] Tae Ha Park, Marcus Märtens, Gurvan Lecuyer, Dario Izzo, and Simone D’Amico. SPEED+: Next-generation dataset for spacecraft pose estimation across domain gap. In IEEE Aerospace Conference, pages 1–15, 2022. doi: 10.1109/AERO53065.2022.9843439. URL https://arxiv.org/abs/2110.03101.

[27] Mohamed Adel Musallam, Kassem Al Ismaeil, Oyebade Oyedotun, Marcos Damian Perez, Michel Poucet, and Djamila Aouada. SPARK: SPAcecraft recognition leveraging knowledge of space environment. arXiv:2104.05978v2, 2021. URL https://arxiv.org/abs/2104.05978.

[28] CVI2, University of Luxembourg. SPARK-2024 dataset. Dataset documentation, 2024. URL https://huggingface.co/datasets/CVI2-UniLU/SPARK-2024.

[29] Aodi Wu, Jianhong Zuo, Zeyuan Zhao, Xubo Luo, Ruisuo Wang, and Xue Wan. SpaceSense-Bench: A large-scale multi-modal benchmark for spacecraft perception and pose estimation. arXiv:2603.09320v1, preprint, 2026. URL https://arxiv.org/abs/2603.09320.

[30] Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11:1605–1641, 2010. URL https://jmlr.org/papers/v11/el-yaniv10a.html.

[31] Yonatan Geifman and Ran El-Yaniv. SelectiveNet: A deep neural network with an integrated reject option. In International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2151–2159, 2019. URL https://proceedings.mlr.press/v97/geifman19a.html.

[32] Thomas O. Dixon, Stephen A. Giles, and Alex A. Gorodetsky. Predicting uncertainty in vision-based satellite pose estimation using deep evidential regression. Aerospace Science and Technology, 160:110055, 2025. doi: 10.1016/j.ast.2025.110055. URL https://www.sciencedirect.com/science/article/pii/S1270963825001269.

[33] Zilong Chen, Shijie Wang, Haichao Gui, and Rui Zhong. Deep-learning-based pose uncertainty estimation for noncooperative spacecraft with unit dual quaternion distributions. Acta Astronautica, 238:258–270, 2026. doi: 10.1016/j.actaastro.2025.10.016. URL https://www.sciencedirect.com/science/article/pii/S0094576525006915.

[34] Jinghao Wang, Zhang Li, Cong Sun, Yulan Guo, Zi Wang, and Qifeng Yu. Satellite pose set estimation by uncertainty-guided conformal keypoint detection. IEEE Transactions on Neural Networks and Learning Systems, 36(12):20120–20132, 2025. doi: 10.1109/TNNLS.2025.3598481. URL https://ieeexplore.ieee.org/document/11134511/.

[35] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330, 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

[36] Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=33XGfHLtZg.

[37] Gavin C. Cawley and Nicola L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11:2079–2107, 2010. URL https://jmlr.org/papers/v11/cawley10a.html.

[38] Sayash Kapoor and Arvind Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9):100804, 2023. doi: 10.1016/j.patter.2023.100804. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/.

[39] Gideon Schwarz. Estimating the dimension of a model. The Annals of Statistics, 6(2): 461–464, 1978. doi: 10.1214/aos/1176344136. URL https: //sites.stat.washington.edu/courses/stat527/s14/readings/ann\_stat1978.pdf.