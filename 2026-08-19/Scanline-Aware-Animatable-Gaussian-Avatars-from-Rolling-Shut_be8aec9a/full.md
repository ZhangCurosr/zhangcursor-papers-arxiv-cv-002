# Scanline-Aware Animatable Gaussian Avatars from Rolling-Shutter Videos

Youxiang Wang University of Macau

## Abstract

Animatable human avatars are routinely reconstructedfrom multi-view video under a silent assumption: that every pixel of a frame observes the same instant of the body’s motion. Rolling-shutter (RS) sensors expose image rows sequentially, so within one frame the head and the feet of a moving person are separated by tens of milliseconds of articulated motion, and every scanline sees a different pose. Feeding such video to a state-of-the-art avatar bakes the distortion into the canonical representation, where it survives as shear and wobble under novel views and novel poses. Worse, every camera in a rig follows its own readout schedule, so the multi-view consistency that drives the reconstruction is violated even when the geometry is correct. We present RS-Avatar, which reconstructs a sharp, undistorted, animatable 3D Gaussian avatar directly from RS video. The formulation is minimal: a motion-aware avatar already renders the body at several sub-frame instants, and where a blur model averages those renderings, a rolling-shutter model composites them scanline by scanline. Changing that operator is the only modification required. On RS-ZJU, a benchmark we build from ZJU-MoCap, this improves novel-view synthesis over training as if the frames were instantaneous, on every subject. A motion-aware blur model built on the same sub-frame machinery does not transfer, and in fact falls below the shutter-oblivious baseline: the machinery is reusable, the operator is not.

## 1. Introduction

Reconstructing an animatable 3D human avatar from multiview video is a cornerstone problem in vision and graphics, with applications spanning telepresence, film production, sports analysis and embodied AI. Driven by neural radiance fields [25] and, more recently, by 3D Gaussian splatting [16], a mature pipeline has emerged: a canonical set of primitives is bound to a parametric body model such as SMPL [21] and warped into each observed frame by linear blend skinning, so that a handful of minutes of video yields a drivable, photorealistic avatar [11, 12, 14, 18, 35–37, 45].

Every method in this lineage rests on an assumption that is rarely stated and almost never satisfied: that an image is an instantaneous snapshot, so all of its pixels can be explained by a single body pose. Real sensors disagree. The overwhelming majority of CMOS cameras—from smartphones and action cameras to the machine-vision modules used in capture studios—employ a rolling shutter (RS), reading out image rows sequentially rather than simultaneously. The readout of a full frame typically occupies a large fraction of the frame period; for a 30 fps sensor with a 30 ms readout, the last row of a frame is captured almost 30 ms after the first. A wrist swinging at 5 m/s therefore travels 15 cm between the top and the bottom of a single image. For a static scene this manifests as the familiar skew of a passing car; for an articulated human body, it is far worse, because different body parts occupy different rows and are thus sampled at different times, and because the distortion is non-rigid and changes with the pose itself.

Confronted with RS input, one can ignore the problem or reach for one of three families of existing tools. None of the four works, and the reasons differ.

Ignoring the shutter. Training a state-of-the-art avatar directly on RS frames does not merely blur the result—it corrupts the canonical model. The optimizer must explain a scanline-dependent geometry with a single pose per frame, and the only degrees of freedom it has are the canonical primitives themselves, so the shear is absorbed into canonical space. Once there, it reappears whenever the avatar is rendered from a novel view or driven by a novel pose. The corruption is also view-dependent: two cameras looking at the same body at the same nominal timestamp read out different rows at different instants, so their observations are mutually inconsistent even when the geometry is perfect. This violates the multi-view photometric consistency that drives the reconstruction, and the optimizer resolves the conflict by over-smoothing and thickening limbs.

Correcting the images first. A two-stage pipeline that runs an RS correction network [2, 4–6, 19, 41, 54] before avatar training inherits the weaknesses of monocular 2D correction. These methods estimate a per-pixel distortion flow from a single view (or a dual-reversed pair), which is severely ill-posed for a non-rigidly articulating body; they hallucinate content in disoccluded regions; and, critically, they correct each camera independently, so the corrected views are not guaranteed to agree in 3D. Errors introduced in stage one are indistinguishable from appearance in stage two.

Borrowing RS-aware neural rendering. A recent line of work makes radiance fields and Gaussian splatting robust to rolling shutter [17, 30, 40, 47]. All of it addresses the dual of our problem: a static scene observed by a moving camera, where the RS effect is absorbed into a six-degree-offreedom camera trajectory that is optimized alongside the scene. Our setting inverts this. The cameras are rigidly mounted and calibrated; it is the subject that moves, deforms non-rigidly and self-occludes. No camera trajectory, however finely parameterized, can express a person waving one arm while standing still—the required deformation lives in the articulated pose space of the body, not in SE(3).

Reusing the blur formulation. Closest in spirit is recent work on reconstructing avatars from blurry video [34], which also couples a body-motion model to the image formation process. But blur and rolling shutter are structurally different corruptions, and the difference lands precisely where the image formation model is written down. Blur is a temporal integration: it averages the appearance over an exposure, discarding high-frequency radiometric content and leaving a large null space. Rolling shutter, in the short-exposure regime, discards nothing—it is a measurepreserving re-indexing that assigns each row a different point on the motion trajectory. Consequently their ambiguities are opposites. The blur operator is invariant to time reversal within an exposure, so the direction of motion is unidentifiable, which is exactly what the blur-avatar formulation must regularize away. The RS operator fixes the arrow of time by construction and has no such ambiguity. So the blur formulation does not transfer—but, encouragingly, its scaffolding does.

RS-Avatar is built on this observation. A motion-aware avatar already does the expensive part of the job: it recovers a sub-frame body-motion model and renders the avatar at several instants inside a single frame. What distinguishes the two corruptions is only how those renderings are combined into the synthesized observation. Blur averages them uniformly; a rolling shutter selects among them per scanline, taking row band i of the synthesized frame from the rendering at the time that band was read out. Replacing the uniform average ${ \frac { 1 } { T } } \sum _ { i }$ with the masked sum $\textstyle \sum _ { i } \mathbf { M } _ { i } { \odot }$ is, to a first approximation, the entire algorithmic change.

The two settings are not equivalent, though. The interframe motion prior is left with a narrower role: in the blur setting it must also fix the direction of sub-frame motion, which the scan order here supplies for free, so what remains for it is continuity alone. The temporal discretization also changes character, since band composition leaves structured seams where averaging leaves unstructured softening. That turns out not to require more sub-frame renderings: sweeping the number of row bands over a factor of four does not improve the reconstruction, because the intra-frame trajectory is a low-order polynomial that a coarse discretization already oversamples.

We validate RS-Avatar on RS-ZJU, a new benchmark built by scanline-compositing temporally interpolated ZJU-MoCap [36] sequences, with the original sharp frames as ground truth. Modeling the shutter improves the reconstruction over ignoring it, on every subject. The more informative outcome concerns the blur model: with the same spline, the same T renderings and the same optimizer, changing only the operator drops it below the baseline that models no sub-frame structure at all.

Our contributions are:

1. We introduce the problem of animatable avatar reconstruction from rolling-shutter video, and give a 3Daware RS image formation model for animatable Gaussian avatars that differs from the blur model only in its combination operator.

2. We identify what does not carry over: the inter-frame motion prior loses one of its two roles, because the direction of sub-frame motion, which it must disambiguate under blur, is fixed for free by the scan order. We also show what one might expect to carry over but does not— the structured band-composition residual does not demand a finer temporal discretization, because the subframe motion model, not the band count, is what bounds fidelity.

3. We contribute RS-ZJU, a benchmark for the task, and evaluate against shutter-oblivious, two-stage-correction and blur-model baselines.

## 2. Related Work

## 2.1. Animatable Human Avatars

Modern avatar reconstruction couples a photorealistic appearance representation to a parametric body model. Following NeRF [25], Neural Body [36] anchored latent codes on SMPL [21] vertices, Animatable NeRF [35] learned neural blend-weight fields, and HumanNeRF [45] demonstrated free-viewpoint rendering from monocular video; InstantAvatar [14] brought training time to minutes. The advent of 3D Gaussian splatting [16] shifted the field to explicit primitives: GauHuman [12] binds Gaussians to SMPL and refines skinning weights, 3DGS-Avatar [37] adds a non-rigid deformation field, and Animatable Gaussians [18] and GaussianAvatar [11] learn pose-dependent Gaussian maps for high-fidelity clothing dynamics. We adopt this canonical-plus-skinning design, but replace its instantaneous-capture assumption with an explicit shutter model. Crucially, none of these methods can be repaired by simply supplying better per-frame poses: with a rolling shutter, no single pose per frame is correct.

## 2.2. Reconstruction from Degraded Observations

In practice, many degradations exists when we want to perform 3D reconstruction, such as low-light [9, 15, 27– 29, 33], blurriness [23, 34, 43, 50], and rolling shutter [17, 30, 47]. A growing body of work relaxes the assumption that inputs are clean. Deblur-NeRF [23] learns per-view blur kernels; BAD-NeRF [43] and BAD-Gaussians [50] model the camera trajectory during exposure and bundleadjust it jointly with the radiance field. For humans, MAD-Avatar [34] reconstructs sharp animatable Gaussian avatars from blurry multi-view video by modeling motion-induced blur as an average over sub-frame renderings driven by a per-frame pose spline, and resolves the resulting directionof-motion ambiguity with an inter-frame continuity prior. We deliberately inherit that scaffolding. What we change is the operator that combines the sub-frame renderings (Sec. 3.2), and what we must then re-derive is everything that operator touches: the accuracy of the temporal discretization and the purpose of the continuity prior.

## 2.3. Rolling Shutter Geometry and Correction

Rolling shutter has a long history in geometric vision. Ait-Aider et al. [1] recovered object pose and velocity from a single RS view, an early statement of how tightly the shutter couples motion to geometry. RS-aware bundle adjustment [10], stereo [39], differential SfM [55] and calibrationfree video stabilization [8] established that the shutter must enter the geometric model, while continuous-time trajectory representations [22, 42] became the standard vehicle for doing so. We use the same continuous-time idea, but place the spline in articulated body-pose space rather than in SE(3) camera space.

A parallel line learns to undistort RS images directly: DSUN [19], SUNet [5], RSSR [4] and CVR [6] predict pixel-wise displacement or intermediate flow from consecutive RS frames; JCD [53] jointly handles RS and blur; dual-reversed-distortion methods [41, 54] exploit two sensors scanning in opposite directions; and recent work improves robustness to nonlinear motion and occlusion [2, 38]. These operate in 2D and per view, and therefore cannot enforce—or even represent—the 3D consistency that avatar reconstruction requires.

## 2.4. Rolling-Shutter-Aware Neural Rendering

Most closely related in machinery, RS-NeRF [30], USB-NeRF [17] and URS-NeRF [47] embed a row-dependent camera pose into volumetric rendering and bundle-adjust it with the radiance field, and Gaussian Splatting on the Move [40] does the same for 3DGS using velocities from visual-inertial odometry. Every one of these methods assumes a rigid, static scene: the entire time dependence is carried by a camera trajectory, and the scene representation itself is time-invariant. Our subject is the opposite—a static, calibrated rig observing a non-rigidly deforming articulated body—so the time dependence must live in the deformation, which changes the rendering problem (each primitive, not each ray, acquires its own timestamp), the parameterization (pose space, not SE(3)), and the degeneracy analysis. To the best of our knowledge, RS-Avatar is the first method to reconstruct animatable human avatars from rolling-shutter video.

## 3. Method

## 3.1. Preliminaries

Animatable Gaussian avatar. We represent the subject by M Gaussian primitives in a canonical (T-pose) space, $\mathcal { G } =$ $\{ \mathcal { G } _ { k } \} _ { k = 1 } ^ { M }$ with $\mathcal { G } _ { k } = \left( \mathbf { x } _ { k } , \mathbf { q } _ { k } , \mathbf { s } _ { k } , \alpha _ { k } , \mathbf { c } _ { k } \right)$ denoting position, rotation, scale, opacity and view-dependent color [16]. A body state $\boldsymbol { S } = ( \beta , \Theta , B )$ collects the SMPL [21] shape coefficients $\beta \in \mathbb { R } ^ { 1 0 }$ , the pose Θ (local rotations of $J = 2 4$ joints plus global orientation and translation), and the skinning weights B. A canonical primitive is warped to observation space by linear blend skinning,

$$
\begin{array} { r } { \mathcal { W } ( \mathbf { x } _ { k } ; \mathcal { S } ) = \mathbf { R } \Big ( \sum _ { j = 1 } ^ { J } w _ { k j } \mathbf { T } _ { j } \big ( \Theta , \beta \big ) \mathbf { x } _ { k } \Big ) + \mathbf { t } , } \end{array}\tag{1}
$$

with the covariance transported by the rotational part of the same transform. We write $\mathcal { R } ( \mathcal { W } ( \mathcal { G } , S ) ; \Pi _ { c } )$ for the differentiable rasterization of the deformed primitives into camera c.

Capture setup. C calibrated cameras of height H pixels observe the subject for N frames at period $\Delta .$ . Camera c has readout time $\rho _ { c }$ (the interval between the start of the first and of the last row) and trigger offset $\delta _ { c }$ , both obtained by standard sensor calibration. We write $\gamma = \rho / \Delta$ for the readout ratio, the fraction of the frame period spent scanning.

## 3.2. Rolling-Shutter Image Formation

From blur to rolling shutter. Under a global shutter with a finite exposure, a moving subject produces a blurry image, classically modeled as the temporal integral of the latent sharp images over the exposure window [34],

$$
I ^ { B } ( u , v ) ~ = ~ \frac { 1 } { \tau } \int _ { 0 } ^ { \tau } I _ { t } ^ { S } ( u , v ) \mathrm { d } t ~ \approx ~ \frac { 1 } { T } \sum _ { i = 1 } ^ { T } I _ { t _ { i } } ^ { S } ( u , v ) .\tag{2}
$$

Under a rolling shutter with a short exposure, nothing is integrated. Row v simply begins—and effectively completes—its exposure at

$$
\begin{array} { r } { t _ { c , n } ( v ) ~ = ~ n \Delta ~ + ~ \delta _ { c } ~ + ~ \frac { v } { H } \rho _ { c } , } \end{array}\tag{3}
$$

![](images/78e175d1efe9e0cf4b51bb5c6deb39f621cef78c71f3520fe6dbc809ae8a0bb7.jpg)  
Figure 1. Overview of RS-Avatar. Per-frame spline control points give $T$ sub-frame body states; the canonical Gaussians are warped to each and rasterized into the stack $R _ { 1 } \ldots R _ { T }$ (stages 1–3). Stage 4 is the method: band i of the synthesized frame is taken from the rendering at the instant that band was read out, so the observation selects among the renderings where the blur model beneath it averages them. The raw RS frames are the only supervision.

<table><tr><td>Regime</td><td>Condition</td><td>Operator</td><td>Studied in</td></tr><tr><td>Global shutter</td><td> $\rho { = } 0 , \tau {  } 0$ </td><td>identity</td><td>[12,37]</td></tr><tr><td>Motion blur</td><td> $\rho { = } 0 , \tau { > } 0$ </td><td>averaging</td><td>[34]</td></tr><tr><td>Rolling shutter</td><td> $\rho > 0 , \tau \to 0$ </td><td>selection</td><td>ours</td></tr></table>

Table 1. Shutter regimes. Blur and rolling shutter differ in how sub-frame renderings are combined, not in what has to be rendered—which is why the avatar framework carries over and only the combination operator changes.

so the observed frame samples a different latent sharp image at every row:

$$
I _ { c , n } ^ { \mathrm { R S } } ( u , v ) \ = \ I _ { t _ { c , n } ( v ) } ^ { S } ( u , v ) .\tag{4}
$$

Both are special cases of one shutter model whose limits are summarized in Tab. 1: blur averages over time with a rowindependent window, rolling shutter selects a single time that depends on the row.

Discrete 3D-aware model. Writing the latent sharp image as the rendering of the avatar under the body state at that instant, and discretizing the readout interval into T contiguous row bands $\begin{array} { r } { \mathcal { V } _ { i } = \left\{ v : \lceil \frac { ( i - 1 ) H } { T } \rceil \leq v < \lceil \frac { i H } { T } \rceil \right\} } \end{array}$ captured at

$$
\begin{array} { r } { t _ { c , n , i } = n \Delta + \delta _ { c } + \frac { i - 1 / 2 } { T } \rho _ { c } , } \end{array}\tag{5}
$$

we obtain the model we optimize:

$$
\hat { I } _ { c , n } ^ { \mathrm { R S } } = \sum _ { i = 1 } ^ { T } { \bf M } _ { i } \odot \mathcal { R } \Big ( \mathcal { W } ( \mathcal { G } , S _ { t _ { c , n , i } } \Big ) ; \Pi _ { c } \Big ) ,\tag{6}
$$

where $\mathbf { M } _ { i }$ is the binary mask of band $\nu _ { i }$ and $\odot$ is the Hadamard product. Comparing Eq. (6) with the blur model $\begin{array} { r } { \hat { I } ^ { B } = \frac { 1 } { T } \sum _ { i } \mathcal { R } ( \mathcal { W } ( \mathcal { G } , \mathcal { S } _ { t _ { i } } ) ; \Pi _ { c } ) } \end{array}$ makes the change explicit:

the same T virtual sharp renderings are produced, and the uniform average $\textstyle { \frac { 1 } { T } } \sum _ { i }$ is replaced by the masked sum $\textstyle \sum _ { i } \mathbf { M } _ { i } { \odot }$ . Everything else in the pipeline is untouched: the canonical representation, the warping, the rasterizer and the optimizer are those of the blur model.

How many bands. The residual of Eq. (6) differs in character from the blur case. Averaging is a smoothing operation, so a coarse T merely softens the synthesized image and the residual is spatially unstructured. Band composition instead quantizes the capture time into a piecewise-constant function of v, leaving horizontal seams at band boundaries displaced by up to $| \dot { v } | \rho _ { c } / ( 2 T )$ pixels—a structured error, which one would expect to force a finer discretization than blur requires. Empirically it does not. Sweeping $T \in \{ K , 2 K , 4 K \}$ at fixed K (Tab. 3) changes the result by at most 0.11 dB, and does so monotonically downwards. Two effects cap the useful T far below H. The intra-frame trajectory is a degree- $( P - 1 )$ polynomial by construction (Eq. (8)), so once T oversamples that polynomial the extra bands merely re-sample the same curve; and since band i is supervised on only $1 / T$ of the rows, doubling T halves the gradient each rendering carries while the cost grows linearly. We therefore set $T = K$ throughout, and T remains a cost knob rather than an accuracy knob.

## 3.3. 3D Human Motion Model

Eq. (6) needs T body states inside every frame, which a per-frame SMPL estimate cannot supply. We adopt the subframe motion model of the blur setting [34] unchanged.

Sub-frame pose interpolation. For each frame we keep P learnable control values per joint, $\tilde { \Theta } ^ { j } = [ \tilde { \Theta } _ { 1 } ^ { j } , \dots , \tilde { \Theta } _ { P } ^ { j } ]$ and interpolate them with a B-spline of order P. With the

normalized time basis

$$
\begin{array} { r } { \mathbf { B } ( t ) = \Big [ 1 , ~ \frac { t } { T } , ~ \big ( \frac { t } { T } \big ) ^ { 2 } , ~ . ~ . ~ . ~ , ~ \big ( \frac { t } { T } \big ) ^ { P - 1 } \Big ] , } \end{array}\tag{7}
$$

the pose of joint j at sub-frame index t is

$$
\hat { \Theta } _ { t } ^ { j } = \mathbf { B } ( t ) \mathbf { M } ^ { P } \tilde { \Theta } ^ { j } ,\tag{8}
$$

where ${ \mathbf { M } } ^ { P } \in \mathbb { R } ^ { P \times P }$ is the B-spline basis matrix

$$
\mathbf { M } _ { i , j } ^ { P } = { \frac { \binom { P - 1 } { P - 1 - i } } { ( P - 1 ) ! } } \sum _ { s = j } ^ { P - 1 } ( - 1 ) ^ { s - j } { \binom { P } { s - j } } ( P - s - 1 ) ^ { P - 1 - i } .\tag{9}
$$

The same interpolation is applied to the global orientation and translation. The $T$ sub-frame indices are placed symmetrically about the frame centre and span a γ fraction of the frame interval, per Eq. (5).

Pose deformation. Skinning from an interpolated pose alone cannot account for clothing and soft-tissue dynamics, so a small network predicts a residual on top of the interpolated pose,

$$
\Delta _ { t } ^ { j } = \mathcal { G } _ { \mathrm { d i s p } } \big ( \hat { \Theta } _ { t } ^ { j } ; \theta _ { \mathrm { d i s p } } \big ) , \qquad \Theta _ { t } ^ { j } = \hat { \Theta } _ { t } ^ { j } + \Delta _ { t } ^ { j } .\tag{10}
$$

Inter-frame motion regularization. Control points are defined per frame, so nothing ties the end of one frame’s trajectory to the start of the next. We enforce continuity with a geodesic penalty on SO(3),

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { J ( N - 1 ) } \sum _ { n = 1 } ^ { N - 1 } \sum _ { j = 1 } ^ { J } \big \| \hat { \Theta } _ { n , T } ^ { j } - \hat { \Theta } _ { n + 1 , 1 } ^ { j } \big \| _ { G } .\tag{11}
$$

Its role differs from the blur setting. There, ${ \mathcal { L } } _ { \mathrm { r e g } }$ carries a second and heavier burden: the blur operator is invariant to reversing time within the exposure, so the direction of the sub-frame motion is unidentifiable without it. A rolling shutter has no such ambiguity—the scan order is an observable arrow of time—so here $\mathcal { L } _ { \mathrm { r e g } }$ only has to enforce continuity, and it is that lighter role we ablate in Sec. 4.5.

Shape and skinning refinement. The shape $\hat { \beta }$ is a single vector optimized from its initial estimate and shared by the whole sequence, and the skinning weights are refined as $\hat { B } = \tilde { B } + \delta$ with δ predicted from the canonical Gaussian coordinates.

## 3.4. Joint Optimization

All quantities—the canonical Gaussians, the per-frame control points, the deformation network, the shape and the skinning weights—are optimized jointly against the raw RS observations by minimizing

$$
\mathcal { L } = \left. \hat { I } ^ { \mathrm { R S } } - I ^ { \mathrm { R S } } \right. _ { 1 } + \lambda \mathcal { L } _ { \mathrm { r e g } } ,\tag{12}
$$

with $\hat { I } ^ { \mathrm { R S } }$ given by Eq. (6). Control points are initialized from an off-the-shelf per-frame SMPL fit [24] with small random perturbations, and adaptive density control follows GauHuman [12]. Note that a per-frame fit to RS frames is biased towards the pose at the middle band’s capture time, which makes it a reasonable initialization for the spline but not a usable final answer.

## 4. Experiments

## 4.1. Benchmark

No dataset exists for rolling-shutter human avatar reconstruction, so we build RS-ZJU. We start from the refined ZJU-MoCap [36] sequences and use the six subjects (377, 386, 387, 392, 393, 394) at their native 1024 × 1024 resolution, with 556–859 frames each. Following the protocol used to build the blur benchmark [34], a group of K consecutive source frames becomes one output frame; K is thus a frame-rate divider, and we use $K = 1 1$ . We set $\gamma = 1$ throughout, so the readout occupies the entire frame period and the last row of a frame is captured just as the first row of the next one begins—the continuously reading sensor of Tab. 1. The readout window of output frame n is therefore $[ \mathrm { m i d } - K / 2$ , $\mathrm { n i d } + K / 2 ]$ in source-frame units, with mid $= n K + \lfloor K / 2 \rfloor$ , and K alone sets how much body motion a single frame contains.

Synthesizing this needs the scene at sub-frame resolution, so every adjacent source pair the window touches is temporally upsampled $2 ^ { e }$ times with RIFE [13]; the RS frame is then assembled by taking each row from the subframe instant whose timestamp matches Eq. (3). We choose e from the seam bound of Sec. 3.2: measuring D, the imagespace displacement of the fastest body point over a whole readout window, by optical flow over all subjects and views gives $D = 4 4 { - } 6 0$ px, and we take the smallest e keeping the band-boundary discontinuity $D / ( K 2 ^ { e } )$ below half a pixel. This yields $e = 4$ , i.e. 177 sub-frame instants per frame and a seam of 0.34 px. That is far finer than a blur benchmark would need—averaging hides a coarse discretization and band composition does not—and it is the concrete price of the structured error discussed in Sec. 3.2.

Two details make the benchmark cleaner. We centre the readout window on the middle source frame of a group rather than starting it at the group boundary, so the midrow capture time coincides with a real captured frame and the evaluation ground truth is an original sharp image rather than an interpolation. Because a centred window with $\gamma = 1$ reaches half an interval beyond the group on each side, the first group of every sequence is dropped, leaving 354 frames per view summed over the six subjects. Four cameras (04, 10, 16, 22) are used for training and, following the ZJU-MoCap protocol, camera 03 is excluded, leaving 18 held-out cameras for novel-view evaluation; ground truth is the original global-shutter frame at the reference time $t _ { c , n } ( H / 2 )$

<table><tr><td>Family</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Shutter-oblivious</td><td>GauHuman [12]</td><td>23.69</td><td>0.803</td><td>0.153</td></tr><tr><td>Two-stage</td><td>CVR [6] + GH JAMNet [7] + GH</td><td>23.99 23.96</td><td>0.809 0.808</td><td>0.153 0.155</td></tr><tr><td>Motion-aware</td><td>MAD-Avatar [34]</td><td>22.99</td><td>0.789</td><td>0.170</td></tr><tr><td></td><td>RS-Avatar (ours)</td><td>24.68</td><td>0.819</td><td>0.138</td></tr></table>

Table 2. Novel-view synthesis on RS-ZJU at K = 11, averaged over the 18 held-out cameras of each subject and then over the six subjects (frame-count weighting moves every entry by less than 0.03). Note that the blur model scores below the shutter-oblivious baseline.

Metrics. We report PSNR, SSIM [44] and LPIPS [49] on held-out views, computed inside the dilated body mask.

## 4.2. Baselines

(a) Shutter-oblivious. GauHuman [12] trained directly on RS frames. This is what a practitioner does today.

(b) Two-stage. A 2D RS correction network applied per view, followed by GauHuman. We take CVR [6] and the more recent JAMNet [7]. Both synthesize a globalshutter frame at a selectable reference time; we set it to the middle scanline, matching our evaluation target, and select each method’s checkpoint by measured accuracy on our data rather than by its default.

(c) Blur-model transfer. MAD-Avatar [34] applied to RS input, testing whether a motion-aware blur model suffices.

## 4.3. Implementation Details

RS-Avatar is built on the GauHuman [12] codebase. The rasterizer is used unmodified; the only change to the training loop is that the T sub-frame renderings are combined by Eq. (6) instead of being averaged. We use P = 4 control points, T = K row bands (Tab. 3), λ = 1, and train for 20k iterations with Adam. Training takes roughly 55 minutes per subject on a single RTX 4090. At equal T this is indistinguishable from the blur model: both render the same T sub-frames, which dominates the cost, and replacing the average by the masked sum adds no rendering. Full hyperparameters and per-subject numbers are in the supplement.

## 4.4. Results

Tab. 2 reports novel-view synthesis, and Fig. 2 the corresponding qualitative behavior. The comparison separates three questions: whether the shutter must be modeled at all (row 1 vs. ours), whether modeling it in 2D is sufficient (rows 2–3), and whether an existing motion-aware model of a different corruption transfers (row 4).

Modeling the shutter is worth 0.99 dB over training on the RS frames as if they were instantaneous, and RS-Avatar is ahead on all six subjects. Fig. 2 shows where the difference sits. Every method recovers the torso and they separate on the arm: the shutter-oblivious reconstruction blurs it into the background and rounds the hand into a blob, the two 2D-corrected variants sharpen the torso but leave the forearm thickened with a soft contour, and the blur model leaves a dark streak along its length, which is what averaging renderings the sensor never averaged produces. Ours keeps the forearm’s taper and the gap between hand and hip. None of them recovers the cardigan buttons; those are absent for a different reason, being a limit of reconstructing a canonical avatar from four views rather than a shutter effect, and they bound every row of Tab. 2 equally.

The blur model is worth a closer look, because MAD-Avatar shares the sub-frame apparatus this method depends on: the same pose spline, the same T renderings per frame, the same optimizer. It differs only in combining those renderings by a uniform average rather than by scanline composition, and that costs 1.69 dB against ours and 0.70 dB against the shutter-oblivious baseline, which models no subframe structure at all. Falling behind that baseline is the part worth explaining. Averaging T renderings taken at different instants produces a synthetic image in which every row is a mixture, and it is then matched against an observation in which every row is a single sample; the residual is nonzero everywhere and the optimizer has no way to attribute it, so it distributes the error over the canonical Gaussians. The shutter-oblivious model makes a cruder assumption but a self-consistent one, and is penalised only where the body actually moved.

The two correctors both improve the images they produce. Scored against the sharp reference inside the body region, before any avatar is trained, CVR gains 0.84 dB and JAMNet 0.97 dB. Only about a third of that reaches the avatar, 0.30 and 0.27 dB, leaving both roughly 0.7 dB behind modeling the shutter directly. The shortfall belongs to the pipeline rather than to either network: each view is corrected in isolation, nothing constrains the corrected views to agree with one another in 3D, and the reconstruction downstream cannot separate a residual correction error from appearance. CVR carries a further cost that JAMNet does not, which makes the mechanism visible. Trained on driving scenes where the distortion comes from camera motion, it estimates a single flow field and resamples the whole image; on a static rig observing a moving subject that is the wrong prior for most pixels, and its full-frame PSNR drops by 4.0 dB even as the body region improves, whereas JAMNet leaves the frame essentially intact (+0.2 dB). The two nonetheless deliver almost the same avatar, which suggests that the quality of the per-view restoration is not what bounds this pipeline. Modeling the readout inside the image formation process avoids the trade, since the shutter is then explained once, in the geometry the views already share.

![](images/39800846d556b44553bc75b07399cd3a60b152f2f0968ecf7a64f875aca62ffc.jpg)  
GauHuman

![](images/0f5829696df970a05a7128587986d4279159d43e0070e21fc887665505c1404f.jpg)  
CVR + GH

![](images/a060a409a35881924124c3787aaf6711265d0784e05028429dfdc4cbee2a40be.jpg)  
JAMNet + GH

![](images/2d988d85f771776116fe3904fee20af06fcb62c70a4e8f3ef0fce7b9e0939c40.jpg)  
MAD-Avatar

![](images/52c8212f8d6cd8a47f1ef083af72223f847bee10eb6e2920218a59d811683090.jpg)  
RS-Avatar (ours)

![](images/61ab2e9ea587aa5017a15808dd904954a34fbebb46c99c4aab0343fc831ee807.jpg)  
GT  
Figure 2. Novel-view synthesis on RS-ZJU at K = 11, held-out camera. The outstretched forearm sweeps the largest image-space distance within one readout, and is where the methods differ.

## 4.5. Ablation

Tab. 3 removes one component at a time. Replacing the intra-frame trajectory with a single pose costs 3.25 dB, twice what removing the rolling-shutter model alone costs. The two are not independent, since a constant intra-frame pose leaves the operator nothing to select between; this row therefore bounds their combined contribution rather than isolating the trajectory’s. Quadrupling the number of bands moves the result by 0.11 dB, and downwards. The intraframe trajectory is a cubic (Eq. (8)) that T = K already oversamples, so the extra bands re-sample the same curve, and each rendering is supervised on only 1/T of the rows, so a larger T also thins the gradient it carries. Of the remaining components, the inter-frame prior is worth 1.39 dB and the pose deformation network 0.65 dB, while the LBS refinement contributes 0.08 dB and could be dropped.

## 4.6. Limitations

Four limitations are worth stating. First, the intra-frame trajectory is a degree-(P − 1) polynomial, so motion a cubic cannot follow within a single frame lies outside the model, and—as Tab. 3 shows—no amount of extra readout discretization recovers it. Second, we operate in sRGB and assume a well-exposed capture, so the method has no account of low-light degradation, where the sensor noise and the tone curve interact with the readout [9, 26–29]. Third, sub-frame motion is carried entirely by SMPL, so anything without a corresponding joint—a handheld object, a loose garment—has no way to move within a frame, and its scanline-dependent displacement cannot be recovered; adaptive or learned joints [48] together with non-rigid modelling are a natural route to lifting this. Finally, our evaluation is synthetic: the readout schedule is exact by construction, whereas a real sensor’s is only approximately known. Beyond these, generative priors [3, 20, 31, 32, 46, 51, 52] could plausibly refine regions that four views leave underconstrained—the cardigan buttons of Fig. 2 being one example—though doing so would trade measurement for hallucination and is out of scope here.

<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Full model (T = K)</td><td>24.65</td><td>0.858</td><td>0.096</td></tr><tr><td>w/o RS model (ρ≡ 0)</td><td>23.08</td><td>0.835</td><td>0.114</td></tr><tr><td>w/o B-spline (one pose per frame)</td><td>21.41</td><td>0.794</td><td>0.176</td></tr><tr><td>w/o pose deformation</td><td>24.00</td><td>0.847</td><td>0.109</td></tr><tr><td>w/o LBS refinement</td><td>24.57</td><td>0.855</td><td>0.098</td></tr><tr><td>w/o Lreg</td><td>23.26</td><td>0.837</td><td>0.109</td></tr><tr><td colspan="3">Row bands (cost 2× and 4× the full model)</td><td></td></tr><tr><td>T=2K</td><td>24.59</td><td>0.857</td><td>0.096</td></tr><tr><td>T=4K</td><td>24.54</td><td>0.856</td><td>0.096</td></tr></table>

Table 3. Ablations on RS-ZJU at K = 11, subject 377, which is why the full-model row differs from the six-subject average in Tab. 2. Evaluation renders one body state at the mid-row capture time and is independent of T, so the last block is directly compa rable to the full model.

## 5. Conclusion

We have argued that the rolling shutter belongs in the image formation model of animatable human avatars rather than in a preprocessing stage. Placing it there changes what a frame is taken to represent: a schedule rather than a snapshot, with the body posed once per scanline instead of once per image. Little else has to change, because a motion-aware avatar already renders the body at several sub-frame instants; going from blur to rolling shutter only replaces the uniform average of those renderings with a scanline-wise composition.

One conclusion generalizes beyond our system: corruption models are not interchangeable even when their machinery is. Blur and rolling shutter differ only in how subframe renderings are combined, yet blur discards the direction of sub-frame motion while a rolling shutter records it, so a prior written to repair one is doing a different job in the other. We measure this as a blur model that, given the identical spline, renderings and optimizer, scores below a baseline that models no sub-frame structure at all.

One question we do not answer is worth stating. Because scanlines across a rig tile the time axis densely, an RS capture samples the body’s motion at more distinct instants than a global-shutter capture at the same frame rate. Whether that latent sub-frame information can be turned into usable temporal super-resolution is something our formulation makes it possible to ask, and we leave it open.

Natural extensions include event- or IMU-assisted initialization for very fast motion, multi-person capture where scanline schedules interact with inter-person occlusion, and monocular RS avatar reconstruction, where the reduced multi-view redundancy will call for additional priors.

## References

[1] Omar Ait-Aider, Nicolas Andreff, Jean Marc Lavest, and Philippe Martinet. Simultaneous object pose and velocity computation using a single view from a rolling shutter camera. In ECCV, 2006. 3

[2] Mingdeng Cao, Sidi Yang, Yujiu Yang, and Yinqiang Zheng. Rolling shutter correction with intermediate distortion flow estimation. In CVPR, 2024. 1, 3

[3] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. Advances in Neural Information Processing Systems, 37:107064–107086, 2024. 7

[4] Bin Fan and Yuchao Dai. Inverting a rolling shutter camera: Bring rolling shutter images to high framerate global shutter video. In ICCV, 2021. 1, 3

[5] Bin Fan, Yuchao Dai, and Mingyi He. SUNet: Symmetric undistortion network for rolling shutter correction. In ICCV, 2021. 3

[6] Bin Fan, Yuchao Dai, Zhiyuan Zhang, Qi Liu, and Mingyi He. Context-aware video reconstruction for rolling shutter cameras. In CVPR, 2022. 1, 3, 6

[7] Bin Fan, Yuxin Mao, Yuchao Dai, Zhexiong Wan, and Qi Liu. Joint appearance and motion learning for efficient rolling shutter correction. In CVPR, pages 5671–5681, 2023. 6

[8] Matthias Grundmann, Vivek Kwatra, Daniel Castro, and Irfan Essa. Calibration-free rolling shutter removal. In ICCP, 2012. 3

[9] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In

2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1777–1786. Ieee, 2020. 3, 7

[10] Johan Hedborg, Per-Erik Forssen, Erik Ringaby, and´ Michael Felsberg. Rolling shutter bundle adjustment. In CVPR, 2012. 3

[11] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. GaussianAvatar: Towards realistic human avatar modeling from a single video via animatable 3D gaussians. In CVPR, 2024. 1, 2

[12] Shoukang Hu, Tao Hu, and Ziwei Liu. GauHuman: Articulated gaussian splatting from monocular human videos. In CVPR, 2024. 1, 2, 4, 5, 6

[13] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In ECCV, 2022. 5

[14] Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. InstantAvatar: Learning avatars from monocular video in 60 seconds. In CVPR, 2023. 1, 2

[15] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE transactions on image processing, 30:2340–2349, 2021. 3

[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4), 2023. 1, 2, 3

[17] Moyang Li, Peng Wang, Lingzhe Zhao, Bangyan Liao, and Peidong Liu. USB-NeRF: Unrolling shutter bundle adjusted neural radiance fields. In ICLR, 2024. 2, 3

[18] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In CVPR, 2024. 1, 2

[19] Peidong Liu, Zhaopeng Cui, Viktor Larsson, and Marc Polle feys. Deep shutter unrolling network. In CVPR, 2020. 1, 3

[20] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with view consistent 2d diffusion priors. Advances in Neural Informa tion Processing Systems, 37:133305–133327, 2024. 7

[21] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. ACM TOG, 34(6), 2015. 1, 2, 3

[22] Steven Lovegrove, Alonso Patron-Perez, and Gabe Sibley. Spline fusion: A continuous-time representation for visualinertial fusion with application to rolling shutter cameras. In BMVC, 2013. 3

[23] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, and Pedro V. Sander. Deblur-NeRF: Neural radiance fields from blurry images. In CVPR, 2022. 3

[24] EasyMoCap Contributors. EasyMoCap – make human motion capture easier. https://github.com/zju3dv/ EasyMoCap, 2021. 5

[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view syn thesis. In ECCV, 2020. 1, 2

[26] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 16169–16178. IEEE, 2022. 7

[27] Muyao Niu, Zhuoxiao Li, Yifan Zhan, Huy H Nguyen, Isao Echizen, and Yinqiang Zheng. Physics-based adversarial attack on near-infrared human detector for nighttime surveillance camera systems. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8799–8807, 2023. 3

[28] Muyao Niu, Zhuoxiao Li, Zhihang Zhong, and Yinqiang Zheng. Visibility constrained wide-band illumination spectrum design for seeing-in-the-dark. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13976–13985. IEEE, 2023.

[29] Muyao Niu, Zhihang Zhong, and Yinqiang Zheng. Nirassisted video enhancement via unpaired 24-hour data. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10744–10754. IEEE, 2023. 3, 7

[30] Muyao Niu, Tong Chen, Yifan Zhan, Zhuoxiao Li, Xiang Ji, and Yinqiang Zheng. RS-NeRF: Neural radiance fields from rolling shutter images. In ECCV, 2024. 2, 3

[31] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In European conference on computer vision, pages 111–128. Springer, 2024. 7

[32] Muyao Niu, Mingdeng Cao, Yifan Zhan, Qingtian Zhu, Mingze Ma, Jiancheng Zhao, Yanhong Zeng, Zhihang Zhong, Xiao Sun, and Yinqiang Zheng. Anicrafter: Customizing realistic human-centric animation via avatarbackground conditioning in video diffusion models. arXiv preprint arXiv:2505.20255, 2025. 7

[33] Muyao Niu, Mingze Ma, Yifan Zhan, Qingtian Zhu, Zhihang Zhong, Wei Guo, Chang Wen Chen, and Yinqiang Zheng. Toward robust and 3d-aware rgb-nir imaging in the dark. arXiv preprint arXiv:2607.29684, 2026. 3

[34] Muyao Niu, Yifan Zhan, Qingtian Zhu, Zhuoxiao Li, Wei Wang, Zhihang Zhong, Xiao Sun, and Yinqiang Zheng. Motion-aware animatable gaussian avatars deblurring. In CVPR, 2026. 2, 3, 4, 5, 6

[35] Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Xiaowei Zhou, and Hujun Bao. Animatable neural radiance fields for modeling dynamic human bodies. In ICCV, 2021. 1, 2

[36] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In CVPR, 2021. 2, 5

[37] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3DGS-Avatar: Animatable avatars via deformable 3D gaussian splatting. In CVPR, 2024. 1, 2, 4

[38] Delin Qu, Yizhen Liao, Huiqing Wang, Bin Zhao, Zhigang Cui, and Xuelong Li. Towards nonlinear-motion-aware and occlusion-robust rolling shutter correction. In ICCV, 2023. 3

[39] Olivier Saurer, Kevin Koser, Jean-Yves Bouguet, and Marc¨ Pollefeys. Rolling shutter stereo. In ICCV, 2013. 3

[40] Otto Seiskari, Jerry Ylilammi, Valtteri Kaatrasalo, Pekka Rantalankila, Matias Turkulainen, Juho Kannala, Esa Rahtu, and Arno Solin. Gaussian splatting on the move: Blur and rolling shutter compensation for natural camera motion. In ECCV, 2024. 2, 3

[41] Wei Shang, Dongwei Ren, Chaoyu Feng, Xiaotao Wang, Lei Lei, and Wangmeng Zuo. Self-supervised learning to bring dual reversed rolling shutter images alive. In ICCV, 2023. 1, 3

[42] Christiane Sommer, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers. Efficient derivative computation for cumulative B-Splines on Lie groups. In CVPR, 2020. 3

[43] Peng Wang, Lingzhe Zhao, Ruijie Ma, and Peidong Liu. BAD-NeRF: Bundle adjusted deblur neural radiance fields. In CVPR, 2023. 3

[44] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 13(4), 2004. 6

[45] Chung-Yi Weng, Brian Curless, Pratul P. Srinivasan, Jonathan T. Barron, and Ira Kemelmacher-Shlizerman. Hu manNeRF: Free-viewpoint rendering of moving people from monocular video. In CVPR, 2022. 1, 2

[46] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26024–26035. IEEE, 2025. 7

[47] Bo Xu, Ziao Liu, Mengqi Guo, Jiancheng Li, and Gim Hee Lee. URS-NeRF: Unordered rolling shutter bundle adjust ment for neural radiance fields. In ECCV, 2024. 2, 3

[48] Yifan Zhan, Qingtian Zhu, Muyao Niu, Mingze Ma, Jiancheng Zhao, Zhihang Zhong, Xiao Sun, Yu Qiao, and Yinqiang Zheng. Towards explicit exoskeleton for the reconstruction of complicated 3d human avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14259–14269, 2025. 7

[49] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

[50] Lingzhe Zhao, Peng Wang, and Peidong Liu. BAD-Gaussians: Bundle adjusted deblur gaussian splatting. In ECCV, 2024. 3

[51] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos. arXiv preprint arXiv:2409.07447, 2024. 7

[52] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cv-

vae: A compatible video vae for latent generative video models. Advances in Neural Information Processing Systems, 37: 12847–12871, 2024. 7

[53] Zhihang Zhong, Yinqiang Zheng, and Imari Sato. Towards rolling shutter correction and deblurring in dynamic scenes. In CVPR, 2021. 3

[54] Zhihang Zhong, Mingdeng Cao, Xiao Sun, Zhirong Wu, Zhongyi Zhou, Yinqiang Zheng, Stephen Lin, and Imari Sato. Bringing rolling shutter images alive with dual reversed distortion. In ECCV, 2022. 1, 3

[55] Bingbing Zhuang, Loong-Fah Cheong, and Gim Hee Lee. Rolling-shutter-aware differential SfM and image rectification. In ICCV, 2017. 3