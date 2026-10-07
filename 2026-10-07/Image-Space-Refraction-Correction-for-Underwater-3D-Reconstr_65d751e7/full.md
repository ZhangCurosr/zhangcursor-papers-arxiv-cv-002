# Image-Space Refraction Correction for Underwater 3D Reconstruction: Warping Flat-Port Views into Pinhole Perspective

Chelim Lim, Tobias Fischer, Emilio Olivastri, Beverley Gorry, Michael Milford and Alejandro Fontan

Abstract— Consumer-grade cameras in flat-port housings are widely used for underwater exploration and mapping of coral reefs and seafloor habitats due to their low cost and accessibility. However, refraction at the flat-port interfaces causes bowlshaped deformation of the reconstructed scene and camera trajectory, compromising the metric accuracy that mapping and navigation depend on. To remove the dominant refractive distortion before reconstruction, we introduce a physicsbased refraction correction in image-space. Our method is downstream-agnostic, and the refraction-corrected images can be directly used as input to any reconstruction or SLAM algorithms. We characterize the refractive distortion through ray-tracing simulations and validate the correction on two real underwater datasets with difering scene structure. Compared with conventional and refractive Structure-from-Motion (SfM), our approach removes the reconstruction deformation while registering more frames and maintaining low reprojection error. The correction further generalizes across diverse reconstruction and VSLAM backends, demonstrating its broad applicability to downstream vision pipelines. Code and interactive demo are available at github.com/cllim118/Refrax and huggingface.co/spaces/cllim118/refrax-demo.

## I. Introduction

Recovering accurate 3D geometry underwater is afected by refraction, as light bends at the camera–housing interface between media of diferent refractive indices, such as water, glass, and air, invalidating the single-viewpoint (pinhole) assumption [1], [2].

Conventional cameras protected by flat-port housings provide a simple and efective solution for underwater imaging, and are widely used instead of dedicated underwater imaging systems, which are expensive and impractical for many applications [3]–[5]. However, the water–housing interface introduces refractive distortions that violate the perspective (pinhole) assumption, resulting in geometric inaccuracies that accumulate, in 3D applications, into bent, bowl-shaped scene structure and camera trajectory [6], [7]. The resulting inconsistent reconstructions are unsuitable for downstream tasks such as robot localization and navigation, as well as ecological monitoring, including coral reef and benthic habitat surveys [8], [9].

3D reconstruction methods are mature outside of water [10]–[13], but underwater reconstruction still poses challenges, such as how refraction is handled. Two main strategies have addressed this problem. The first absorbs refraction, to a certain extent, into conventional radial-tangential distortion parameters, since the induced distortion has a strong radial component. However, this approximation is not exact, since refraction is not perfectly radial and varies with scene depth. As well as this, estimating these parameters jointly with camera pose and structure during reconstruction hinders optimization convergence and introduces systematic errors.

![](images/dd71c9ad7ca56bb90b4ac87665c16381a0d6435ec8238779446f525a30651f09.jpg)  
(a) Conventional SfM

![](images/7db4a9c84a42122ffd42226fb3eeb49503ae35101a702dfdf1995883dd5556ac.jpg)  
(b) Ours + SfM  
Fig. 1: Underwater 3D reconstruction from top and side views, comparing (a) Conventional SfM with refined intrinsics (red) and (b) Ours + SfM (green) on a survey of three coral bommies. Our correction recovers a flat, planar trajectory and scene structure (side view, green), consistent with the vehicle’s known survey pattern, whereas Conventional SfM produces a visibly bowed trajectory (side view, red).

The second strategy incorporates refraction tightly into the reconstruction through dedicated refractive camera models and modified bundle adjustment [6], [14]. Such physicsbased modeling improves reconstruction accuracy over perspective approximations, but at the cost of requiring refractive parameters to be handled jointly with camera pose and scene structure, which increases the number of coupled variables and hinders optimization convergence. In contrast, we propose to correct refractive distortion in image space, decoupling the correction from the reconstruction, which yields a simpler optimization and makes it independent of the downstream reconstruction system (Fig. 1).

The main contribution of this paper is a physics-based refraction correction method that removes the refractive distortion in image space prior to downstream vision tasks. By eliminating the need for refractive parameters from the downstream method, our approach simplifies the optimization and improves geometric fidelity and reconstruction robustness. The corrected images can be used with existing reconstruction, localization, and mapping methods without modifying their camera models or optimization procedures.

In particular, we make the following contributions:

1) Physics-based preprocessing: A ray-traced imagespace correction that characterizes the refractive displacement by tracing light through the flat-port housing’s air–glass and glass–water interfaces, and inverts it into a backward warp that maps each image to their equivalent perspective appearance, decoupling refraction correction from the reconstruction backend.

2) Failure-mode analysis: We build a simulator of the flat-port refractive model to analyze the influence of each parameter involved in the warping, and propose several ablations to discuss their relative importance and correlations, and the trade-ofs through which overparameterization hinders optimization convergence and degrades accuracy.

3) Empirical evaluation: On real underwater datasets, our correction improves geometric fidelity by up to 10× over conventional and refractive SfM and transfers across diverse vision pipelines.

## II. Related Work

Underwater 3D reconstruction faces a wide array of challenges, comprising optical efects such as attenuation and backscattering that degrade image appearance [15], [16], or sensor fusion with acoustic and inertial sensors for robust localization in turbid scenes [17], [18]. We target the geometric distortion induced by refraction at the camera housing and its impact on 3D reconstruction.

## A. Underwater Camera Modeling and Refraction

Underwater cameras enclosed in flat-port housings violate the conventional perspective imaging assumption because light rays are refracted at the interfaces between water, glass, and air. As a result, the observed image rays deviate from those predicted by a pinhole camera model [1]. This deviation is governed by Snell’s law,

$$
\eta _ { w } \sin \theta _ { w } = \eta _ { g } \sin \theta _ { g } = \eta _ { a } \sin \theta _ { a } ,\tag{1}
$$

where $\eta _ { a } , \eta _ { g } , \eta _ { w }$ are the refractive indices of air, glass, and water, and $\theta _ { a } , \theta _ { g } , \theta _ { w }$ are the ray angles with respect to the interface normal.

Treibitz et al. [1] showed that a flat-port underwater camera does not have a single viewpoint and cannot be exactly described by a pinhole model, as the required compensation varies with scene depth [19]. In practice, however, the radial component can be partially approximated with conventional camera intrinsics and distortion parameters [20].

When these parameters are refined through selfcalibration, the added degrees of freedom can trade of against one another [21]. This is especially problematic in scenes with weak geometric structure, where flat, featurelimited surfaces provide insuficient constraints for reliable self-calibration, which can lead to systematic “doming” [22], [23]. Such deformation is most pronounced on near-planar scenes and reduced on topographically varied ones [22], and is further amplified under near-parallel viewing [24], [25].

## B. Refractive Structure-from-Motion

Conventional 3D reconstruction pipelines, including COLMAP [10], and recent learning-based approaches such as VGGT [11], MASt3R [12], and Depth Anything 3 [13] rely on perspective camera models and therefore do not explicitly account for underwater refraction. To address this limitation, refractive SfM methods tightly incorporate light refraction into the reconstruction process to improve geometric accuracy. Chadebecq et al. [2] derived an explicit refractive fundamental matrix, extending the theoretical twoview refractive geometry of Chari et al. [26], avoiding the costly direct computation of the refractive reprojection error, which requires solving a 12th-degree polynomial. Jordt et al. [14] proposed a complete refractive SfM framework by introducing a refractive reprojection error and customized pose estimation and bundle adjustment procedures. More recently, She et al. [6] integrated refractive geometry into COLMAP, enabling large-scale incremental refractive SfM while jointly optimizing camera and refractive parameters.

All the above-mentioned approaches tightly built refraction into the reconstruction geometry itself – whether as a refractive two-view formulation [2], a dedicated SfM framework [14], or an extension of an existing pipeline [6] – so that the estimation must be reformulated around the refractive model. In contrast, we correct refraction as an image-level pre-processing step, separating it from reconstruction, which enables state-of-the-art reconstruction methods to be used without modification.

## C. Camera Calibration Underwater

A complementary line of work addresses underwater refraction without explicitly incorporating refractive geometry into the reconstruction pipeline. Instead, these approaches represent the refractive efect within conventional perspective camera models by adjusting intrinsic parameters or introducing additional correction terms [19], [20], [27].

The Pinax model [28] exploits the geometric constraints of practical camera configurations with flat-port housings and represents the refractive projection through a virtual pinhole approximation with precomputed ray mappings. Other approaches estimate physical housing parameters directly from calibration targets [29] that can be used by subsequent refractive reconstruction methods [6]. Singh et al. [4], [5] jointly estimate the refractive index within a visual-inertial framework by decomposing the refractive efect into virtual pinhole and radial distortion components.

Our work complements these approaches by correcting refraction directly in the image domain, as a physics-based preprocessing step before reconstruction. Rather than proposing a new distortion model, we investigate whether correcting refraction improves reconstruction robustness and geometric fidelity while remaining compatible with standard pipelines.

## III. Methodology

## A. Refractive Camera Geometry

We first review the geometry of a flat-port underwater camera [1], [30], which we use to derive the ray-traced displacement field in Sec. III-B. We consider a pinhole camera enclosed in a flat-port housing observing an underwater scene. Under the conventional perspective model, a 3D point $P = ( X , Y , Z ) \in \mathbb { R } ^ { 3 }$ projects onto the image plane as

$$
u = f _ { x } \frac { X } { Z } + c _ { x } , \qquad \nu = f _ { y } \frac { Y } { Z } + c _ { y } ,\tag{2}
$$

where $( u , \nu )$ is the resulting image coordinate, $( c _ { x } , c _ { y } )$ the principal point, and $( f _ { x } , f _ { y } )$ the focal length. Normalizing by the intrinsics gives the corresponding normalized image coordinates

$$
x = \frac { u - c _ { x } } { f _ { x } } , \qquad y = \frac { \nu - c _ { y } } { f _ { y } } ,\tag{3}
$$

with normalized image radius

$$
r = { \sqrt { x ^ { 2 } + y ^ { 2 } } } .\tag{4}
$$

For a flat port, the air–glass and glass–water interfaces are assumed parallel to each other and perpendicular to the optical axis [1], [30]. Light refracts at each interface according to Snell’s law. The housing is parameterized by the distance $d _ { 0 }$ between the camera center and the air–glass interface, and the glass thickness $t _ { g } .$ . With $Z$ denoting the camera-to-object distance along the optical axis, the refractive projection (derived in Sec. III-B) depends on both $Z$ and the housing geometry.

## B. Forward Refractive Displacement Model

We compute the refractive displacement by tracing rays through the flat-port housing (see Fig. 2a). Given a pixel $( u , \nu )$ , and its normalized image coordinates $( x , y )$ and radius $r ,$ as defined in Sec. III-A, its ray in air is then back-projected as

$$
\mathbf { r } _ { a } = \frac { ( x , y , 1 ) ^ { \top } } { \sqrt { r ^ { 2 } + 1 } } \in \mathbb { S } ^ { 2 } .\tag{5}
$$

This ray intersects the air–glass interface at $P _ { 1 } ,$ located at distance $d _ { 0 }$ from the camera center along the port normal $\mathbf { n } \in \mathbb { S } ^ { 2 }$ . Applying Snell’s law gives the ray direction in glass,

$$
\mathbf { r } _ { g } = \mathrm { r e f r a c t } ( \mathbf { r } _ { a } , \mathbf { n } , \eta _ { a } , \eta _ { g } ) ,\tag{6}
$$

where $\eta _ { a }$ and $\eta _ { g }$ are the refractive indices of air and glass. The function refract(·) applies Snell’s law in vector form:

$$
\mathrm { r e f r a c t } ( \mathbf { r } , \mathbf { n } , \eta _ { 1 } , \eta _ { 2 } ) = \mu \mathbf { r } + \left( \sqrt { 1 - \mu ^ { 2 } ( 1 - c ^ { 2 } ) } - \mu c \right) \mathbf { n } ,\tag{7}
$$

where $\mu = \eta _ { 1 } / \eta _ { 2 }$ and $c = \mathbf { n } ^ { \top } \mathbf { r } .$ . The refracted ray propagates through the glass of thickness $t _ { g }$ to the glass–water interface at $P _ { 2 }$ , where a second refraction gives the water-side direction,

$$
\mathbf { r } _ { w } = \mathrm { r e f r a c t } ( \mathbf { r } _ { g } , \mathbf { n } , \eta _ { g } , \eta _ { w } ) ,\tag{8}
$$

with $\eta _ { w }$ denoting the refractive index of water.

![](images/3d5e758827757641288abf1961841c37bcd6e8034f15a36d3e739a98bf794e9d.jpg)  
(a) Ray tracing through the flat-port housing. A ray from the camera center travels as $\mathbf { r } _ { a } ,$ refracts toward the normal at the air–glass interface $( P _ { 1 } )$ into $\mathbf { r } _ { g } ,$ and away from the normal at the glass–water interface $( P _ { 2 } )$ into $\mathbf { r } _ { w } ,$ meeting the scene plane at depth $Z$ in $P _ { w } .$ . The dashed line is the direct camera-to- $P _ { w }$ reference. Angles are exaggerated for clarity.

![](images/38312cc36f912a909dd7fe1d8d0a35493e6e48f3c7176907a3264aca1817a66e.jpg)  
(b) Backward refraction warping. For each pixel in the corrected image, we solve for the corresponding pixel in the raw underwater image whose ray-traced water-side intersection matches the ideal perspective 3D point at the assumed depth, and sample its color via bilinear interpolation. The radial displacement visualizes the pixel-wise refractive ofset between the corrected and raw underwater images. The geometric mapping (solid arrows) proceeds from the corrected image to the 3D point and to the raw image, while color values are fetched in the reverse direction (dashed arrow).

![](images/be783f1203a81cbbbde9bc310131cd242483349f8b2e869d78a6397d93765c3c.jpg)

![](images/b70eeaec5a61e4b33c48e223653b12fd630660e81bdf732607b684983038084c.jpg)  
(c) Selecting the corrected image’s scale factor �. Left: Displacementfield RMSE as a function of the scale factor $s ;$ the minimum defines $s ^ { * } .$ Right: Corrected image with the optimal scale $s ^ { * } ,$ closest to the original field of view.  
Fig. 2: Refraction-aware underwater image correction.

![](images/f5e3300614402bae3832ec3eb3051e14e723392f0a008f0648f00e9eed336434.jpg)  
Z (m)  
(a)

![](images/4a928d772c00ed53e646b47545afa2fcdf1a3b9470d79e030a84607830d0c617.jpg)  
(b)

![](images/12f938a1fce47acf37732221d8b92154927106b9bcf30fd135b0e17b110cc539.jpg)  
(c)

![](images/f15eaeccc9198a006019533d61da4b37495768076800e68712fa19f5eca51a5b.jpg)  
(d)

![](images/617f4340869853b8043d77566d4966c1cec8eb8ad4cf370616fc5662e001cb47.jpg)  
(e)

![](images/7a6f2c6eed7ccc012389d3c3d9976b145a993d5c9801a10721b5ef9cef7a0a06.jpg)  
(f)  
Fig. 3: Sensitivity analysis of refraction-correction parameters. Each panel varies one parameter of the flat-port housing model while holding the others fixed at their nominal values: (a) depth �, (b) housing-to-camera distance $d _ { 0 } ,$ (c) the ratio $\sum / d _ { 0 }$ of depth to housing-to-camera distance, (d) water refractive index $\eta _ { w } , \mathrm { ( e ) }$ glass thickness $t _ { g }$ , and (f) focal length $f _ { x } , f _ { y }$

The interface points $P _ { 1 } , \ P _ { 2 }$ , and the ray directions are determined by the housing geometry and are independent of scene depth, and are given by

$$
P _ { 1 } = { \frac { d _ { 0 } } { \mathbf { n } ^ { \top } \mathbf { r } _ { a } } } \mathbf { r } _ { a } , \qquad P _ { 2 } = P _ { 1 } + { \frac { t _ { g } } { \mathbf { n } ^ { \top } \mathbf { r } _ { g } } } \mathbf { r } _ { g } .\tag{9}
$$

Here and throughout, subscripts $x , y , z$ denote a vector’s components along the camera’s optical axis and image-plane directions. The water-side ray intersects the scene plane at

$$
P _ { w } = P _ { 2 } + \frac { Z - P _ { 2 , z } } { r _ { w , z } } \mathbf { r } _ { w } ,\tag{10}
$$

so that $P _ { w , z } = Z$ by construction. The corresponding refracted image location is then

$$
\pi _ { r } ( u , \nu ; Z ) = \left( f _ { x } \frac { P _ { w , x } } { Z } + c _ { x } , \ f _ { y } \frac { P _ { w , y } } { Z } + c _ { y } \right) .\tag{11}
$$

The refractive displacement is defined as

$$
\Delta { \bf d } ( u , \nu ; Z ) = \pi _ { r } ( u , \nu ; Z ) - ( u , \nu ) .\tag{12}
$$

## C. Backward Refraction Warping

To remove refraction, we invert the displacement field into a backward resampling map that relocates each refracted pixel toward its perspective position, and use it to warp the input image (see Fig. 2b).

The housing magnifies the scene: since water is optically denser than air, rays bend toward the optical axis as they cross the interfaces, so the housing captures a narrower field of view than an equivalent perspective camera of the same focal length [1]. Consequently, mapping raw pixels back to their perspective positions concentrates the captured content into a smaller radius around the image center, strongly shrinking the corrected image. To compensate, we introduce a scale factor � that rescales the focal length used in the corrected image’s projection, $f _ { x } ^ { \prime } \ = \ s f _ { x }$ and $f _ { \mathrm { v } } ^ { \prime } ~ { = } ~ s f _ { \mathrm { y } }$ , and resize the output image to new dimensions $( \vec { w ^ { \prime } } , h ^ { \prime } )$ accordingly. We estimate � once per housing configuration via the ray-tracing simulation of Sec. III-B: we compute the refractive displacement field $\Delta \mathbf { d } ( u , \nu ; Z )$ over a range of candidate scales and select the value of � that minimizes its RMSE, keeping the corrected image as close as possible in scale to the original, see Fig. 2c.

Given this scale factor, we compute the backward correspondence for each output pixel as follows. For each pixel $( u ^ { * } , \nu ^ { * } )$ in the corrected image, at known depth $Z ^ { * }$ , we identify the corresponding pixel $( u , \nu )$ in the raw underwater image and sample its intensity there via bilinear interpolation. We first back-project $( u ^ { * } , \nu ^ { * } , Z ^ { * } )$ to its 3D point under the ideal perspective model, using the corrected intrinsics $f _ { x } ^ { \prime } , f _ { y } ^ { \prime }$

$$
P ^ { * } = \left( { \frac { ( u ^ { * } - c _ { x } ) Z ^ { * } } { f _ { x } ^ { \prime } } } , \ { \frac { ( \nu ^ { * } - c _ { y } ) Z ^ { * } } { f _ { y } ^ { \prime } } } , \ Z ^ { * } \right) ,\tag{13}
$$

and then solve for the raw pixel $( u , \nu )$ whose ray-traced water-side intersection $P _ { \kappa } ( u , \nu ; Z ^ { * } )$ , as defined in Eq. (10), coincides with $P ^ { * }$ , i.e.,

$$
\mathrm { \Phi } ( u , \nu ) = \underset { ( u , \nu ) } { \arg \operatorname* { m i n } } \left\| P _ { w } ( u , \nu ; Z ^ { * } ) - P ^ { * } \right\| ^ { 2 } ,\tag{14}
$$

solved as nonlinear least-squares. The intensity at $( u ^ { * } , \nu ^ { * } )$ in the corrected image is then set to the bilinearly interpolated intensity of the raw image at $( u , \nu )$

## D. Depth Assumption in Practice

While the formulation above supports a per-pixel depth �, our experiments assume a single constant depth $Z _ { 0 }$ for all images in a sequence, so that the backward mapping is computed once and applied uniformly rather than solved independently for every pixel. We set $Z _ { 0 }$ to average scene depth over the sequence. This is a reasonable approximation for our datasets, which, as in most coral reef surveys, are largely planar or captured at a roughly constant altitude.

Depth variability nonetheless afects correction quality: as our results on the Lizard Island dataset show, scenes with substantial depth variation are only approximately corrected under a single $Z _ { 0 }$ . Future work should address variable depth, per image or even per pixel, to achieve maximally accurate and reliable reconstructions.

## E. Simulation and Sensitivity Analysis

We simulate the refractive displacement (Eq. 12) while varying each parameter of the flat-port housing model independently, and report the resulting RMSE relative to the nominal configuration (Fig. 3). The displacement is most sensitive to the ratio between the scene depth � and the housing-tocamera distance �<sub>0</sub>: varying � or �<sub>0</sub> independently produces substantial RMSE changes, and expressing this dependence through the ratio $Z / d _ { 0 }$ captures the dominant trend. In contrast, glass thickness $t _ { g }$ has a negligible efect on the displacement, indicating that accurately estimating $Z / d _ { 0 }$ is far more critical for the correction than precisely measuring the housing’s glass thickness.

![](images/6e762b78500da2e94e6f53ac935d9e6c6b952dd43fb5ff51d288f2b123ab720b.jpg)  
Fig. 4: Top row: two RGB images from the Malaysia dataset; bottom row: two RGB images from the Lizard Island dataset. In each pair, left is the raw underwater capture and right is the refraction-corrected image.

## IV. Experimental Setup

## A. Baselines

We compare our approach against SfM COLMAP [10] and its refractive variant, Refractive COLMAP [6]. COLMAP is the standard perspective baseline that can approximate refraction with a radial-tangential distortion model, while Refractive COLMAP jointly optimizes refractive parameters with camera poses and structure throughout the SfM pipeline. Ours instead applies our physics-based correction to the images before reconstructing with conventional COLMAP. Comparing these three approaches highlights their caveats and benefits in terms of final geometric accuracy, optimization convergence, and computational performance.

Because our correction is applied directly to the image, prior to any reconstruction step, it is agnostic to the specific downstream pipeline. We test this portability by evaluating two learning-based multi-view models, MapAnything [31] and Depth Anything 3 (DA3) [13], on the raw and corrected images. Furthermore, we evaluate five VSLAM systems spanning a range of paradigms [32]: the traditional handcrafted feature-based ORB-SLAM3 [33]; the learned featurebased AnyFeature-VSLAM [34]; the RNN-based DROID-SLAM [35] and DPVO [36]; and the transformer-based MASt3R-SLAM [37].

## B. Datasets

We evaluate on two datasets of real underwater data captured with consumer GoPro cameras in standard flat-port housings. These were collected by two diferent teams across four missions at two geographic locations, Malaysia and Lizard Island, spanning four diferent times of year. Frames from the two datasets are shown in Fig. 4. Table I summarizes the complementary characteristics of both datasets, contrasting diver-handheld versus robot-mounted capture and near-planar versus topographically varied scene geometry.

Malaysia Dataset. [38] The survey was conducted at Pom Pom Island, Borneo, Malaysia, as part of an initiative evaluating biodegradable biomaterials for coral-rubble stabilization. A diver-handheld, 3-camera GoPro HERO13 stereo rig surveyed a coral-rubble seabed, with camera-to-seafloor distance varying across passes.

Lizard Island Dataset. [39] An ASV-mounted GoPro HERO11 over a coral reef. The data was collected as part of a multi-year autonomous reef restoration program on the Great Barrier Reef, using a reconfigurable surface vehicle fleet. The scene is topographically varied, while the ASV maintains a fixed height above the water surface, producing an approximately planar camera trajectory.

TABLE I: Per-sequence characteristics of the Malaysia and Lizard Island datasets. � denotes the range of camera-to-scene distance observed in each sequence.
<table><tr><td>Dataset</td><td>Sequence</td><td>Frames</td><td>Z (m)</td><td>Characteristics</td></tr><tr><td rowspan="4">Malaysia</td><td>p1 s01</td><td>17568</td><td rowspan="4">0.5 1.0</td><td rowspan="4">Diver handheld GoPro HERO13 Coral rubble</td></tr><tr><td>p1 s02</td><td>7557</td></tr><tr><td>p1 s03</td><td>11043</td></tr><tr><td>p3 s02</td><td>9885</td></tr><tr><td rowspan="6">Lizard Island</td><td>p3 s03 Feb 2024 01</td><td>5127 10473</td><td rowspan="6">1-8</td><td rowspan="6">ASV GoPro HERO11 Coral reef</td></tr><tr><td></td><td></td></tr><tr><td>Mar 2024 01</td><td>11828</td></tr><tr><td>Mar 2024 02</td><td>4678</td></tr><tr><td>Sep 2024 01</td><td>9156</td></tr><tr><td>Sep 2024 02</td><td>14382</td></tr></table>

## C. Evaluation Metrics

Evaluating underwater 3D reconstruction against absolute ground truth remains impractical at the scale of our surveys. GPS does not propagate underwater, so ASV positioning is not accurate enough to serve as a reliable pose reference, and more sophisticated positioning or scanning systems capable of providing reliable ground truth are expensive and impractical to deploy in the field. We instead evaluate each reconstruction using a thorough combination of structural consistency, whether the reconstruction preserves the underlying geometry known a priori, and standard SfM statistics [10], [40], [41].

Structural deformation. A hallmark of refractive distortion is systematic bowl-shaped deformation in reconstructed scene geometry and camera trajectories. We quantify this deformation using a sequence-specific planar reference. For the near-planar Malaysia seafloor, we fit a plane to the reconstructed 3D points, while for Lizard Island we fit a plane to the reconstructed camera centers, reflecting its constant-altitude ASV trajectory.

Plane fit and scale normalization. Given reconstructed points $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } ,$ we estimate the best-fit plane by SVD and compute the signed point-to-plane residual

$$
r _ { i } = \mathbf { n } ^ { \top } \mathbf { x } _ { i } + d ,\tag{15}
$$

where n is the unit plane normal and � is the plane ofset. We orient n toward the mean camera center. Let $( u _ { i } , \nu _ { i } )$ denote the coordinates of the in-plane projection of $\mathbf { x } _ { i }$ in an orthonormal basis of the fitted plane. To account for arbitrary monocular scale, we normalize by the projected scene extent

$$
D = \sqrt { ( \operatorname* { m a x } _ { i } u _ { i } - \operatorname* { m i n } _ { i } u _ { i } ) ^ { 2 } + ( \operatorname* { m a x } _ { i } \nu _ { i } - \operatorname* { m i n } _ { i } \nu _ { i } ) ^ { 2 } } .\tag{16}
$$

Radial curvature. We model the systematic bowl/dome component of the residual field with a radially symmetric paraboloid,

$$
r _ { i } \approx c _ { 0 } + c _ { 1 } u _ { i } + c _ { 2 } \nu _ { i } + c _ { 3 } \big ( u _ { i } ^ { 2 } + \nu _ { i } ^ { 2 } \big ) ,\tag{17}
$$

TABLE II: Reconstruction results. Distortion parameters are fixed (fix) or refined (ref); planarity is measured against the scene (Malaysia) or trajectory (Lizard Island). � = 0 is flat; negative is a bowl, positive a dome. Parentheses: insuficient sequence coverage; —<sup>∗</sup>: failed reconstruction.
<table><tr><td>Sequence</td><td>Method</td><td>Dist.</td><td>Reg.↑</td><td> $\mathrm { P t s } ^ { \times 1 0 ^ { 5 } } \uparrow$ </td><td>Track↑</td><td>Reproj.↓</td><td>K≈0</td></tr><tr><td rowspan="3">feb_gp1</td><td>COLMAP</td><td>ref</td><td>0.88</td><td>4.22</td><td>6.78</td><td>0.35</td><td>-0.366</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>0.88</td><td>4.22</td><td>6.77</td><td>0.36</td><td>-0.453</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>0.96</td><td>5.85</td><td>7.75</td><td>0.34</td><td>+0.017</td></tr><tr><td rowspan="3">mar_gp1</td><td>COLMAP</td><td>ref</td><td>0.99</td><td>6.72</td><td>8.35</td><td>0.48</td><td>-0.304</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>0.99</td><td>6.68</td><td>8.34</td><td>0.48</td><td>+0.055</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>8.15</td><td>9.41</td><td>0.45</td><td>+0.014</td></tr><tr><td rowspan="3">mar_gp2</td><td>COLMAP</td><td>ref</td><td>0.90</td><td>5.99</td><td>7.90</td><td>0.48</td><td>+0.306</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>0.90</td><td>5.98</td><td>7.88</td><td>0.48</td><td>-0.057</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>6.99</td><td>9.25</td><td>0.47</td><td>+0.017</td></tr><tr><td rowspan="3">sep_gp1</td><td>COLMAP</td><td>ref</td><td>0.79</td><td>5.10</td><td>9.19</td><td>0.57</td><td>+0.082</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>0.79</td><td>5.10</td><td>9.16</td><td>0.57</td><td>-0.045</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>0.97</td><td>6.93</td><td>9.94</td><td>0.52</td><td>-0.024</td></tr><tr><td rowspan="3">sep_gp2</td><td>COLMAP</td><td>ref</td><td>0.39</td><td>(3.41)</td><td>(8.54)</td><td>(0.56)</td><td>(−0.029)</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>0.39</td><td>(3.41)</td><td>(8.52)</td><td>(0.56)</td><td>(+0.068)</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>0.83</td><td>6.79</td><td>8.84</td><td>0.49</td><td>+0.053</td></tr><tr><td colspan="8">Malaysia</td></tr><tr><td>Sequence</td><td>Method</td><td>Dist.</td><td>Reg.↑</td><td> $\mathrm { P t s } ^ { \times 1 0 ^ { 5 } }$  ←</td><td>Track↑</td><td>Reproj.↓</td><td>K≈0</td></tr><tr><td rowspan="3"> $\mathsf { p } 1 \_ { \mathrm { s 0 1 } }$ </td><td>COLMAP</td><td>ref</td><td>0.95</td><td>23.32</td><td>5.70</td><td>0.37</td><td>+0.188</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>0.99</td><td>27.37</td><td>6.01</td><td>0.37</td><td>-0.068</td></tr><tr><td rowspan="3"> $\mathsf { p } 1 \_ { \mathrm { s } 0 2 }$ </td><td>COLMAP</td><td>ref</td><td>1.00</td><td>12.70</td><td>7.38</td><td>0.29</td><td>+0.089</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>1.00</td><td>12.82</td><td>7.19</td><td>0.57</td><td>+0.694</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>13.14</td><td>8.57</td><td>0.32</td><td>-0.000</td></tr><tr><td rowspan="3">p1_s03</td><td>COLMAP</td><td>ref</td><td>1.00</td><td>11.10</td><td>9.43</td><td>0.36</td><td>+0.073</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>1.00</td><td>11.41</td><td>8.94</td><td>0.68</td><td>+0.105</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>10.61</td><td>12.01</td><td>0.40</td><td>+0.031</td></tr><tr><td rowspan="3">p3_s02</td><td>COLMAP</td><td>ref</td><td>1.00</td><td>12.97</td><td>5.03</td><td>0.41</td><td>-0.064</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>15.56</td><td>5.40</td><td>0.42</td><td>-0.004</td></tr><tr><td rowspan="3">p3_s03</td><td>COLMAP</td><td>ref</td><td>1.00</td><td>7.30</td><td>7.45</td><td>0.35</td><td>+0.097</td></tr><tr><td>Refractive COLMAP</td><td>fix</td><td>1.00</td><td>7.32</td><td>7.34</td><td>0.62</td><td>+0.253</td></tr><tr><td>Ours + COLMAP</td><td>fix</td><td>1.00</td><td>7.76</td><td>8.73</td><td>0.38</td><td>+0.078</td></tr></table>

where the afine terms absorb residual ofset and tilt, while $c _ { 3 }$ captures the systematic radial curvature. We report the dimensionless curvature

$$
\kappa = c _ { 3 } D ,\tag{18}
$$

where $\kappa = 0$ indicates a flat reconstruction and |�| measures the strength of the radial bowl/dome deformation. The sign indicates the deformation direction, with negative values corresponding to a bowl and positive values to a dome.

Structure-from-Motion statistics. Even though absolute accuracy cannot be obtained, a systematic analysis of SfM statistics allows us to draw useful conclusions about the performance of the diferent baselines.

Registered image ratio (Reg.) and number of reconstructed 3D points (Pts): ceteris paribus, a higher number of recovered camera poses and/or number of 3D points indicates a better fit of the model to the underlying 3D geometry.

Mean track length (Track): the average number of images in which a reconstructed point is observed reflects how consistently features are matched across views.

Reprojection error (Reproj.): smaller reprojection errors after bundle adjustment are indicative of a better fit of the model, but, as shown in Sec. V, this does not necessarily translate to a more accurate reconstructed geometry.

## V. Results

We run all experiments using calibrated in-air intrinsics and standard flat-port parameters of $t _ { g } = 2$ mm glass thickness and $d _ { 0 } ~ = ~ 2 { \mathrm { c m } }$ housing-to-camera distance. Results,

![](images/0f83f6e4276fbda29269536e47a7deba128ac8db7b3fba09d0188d0a97ebc22d.jpg)  
Fig. 5: Comparison of 3D point clouds and camera trajectories reconstructed using COLMAP (ref) and OURS + COLMAP across Lizard-Island and Malaysia datasets. Note how our approach yields more complete trajectories (fewer camera gaps) and more geometrically consistent reconstructions (no bending).

reported in Table II and Figure 5, use the operational depth $Z _ { 0 }$ obtained as detailed in the ablation study (Sec. VI).

## A. Geometric Fidelity via Structural Deformation

Conventional COLMAP (fix) with fixed intrinsics always leads to catastrophic failure on both the Malaysia, and Lizard Island datasets. Refining the camera intrinsics and distortion parameters, COLMAP (ref), substantially reduces the curvature in Malaysia scenes, with $\kappa \in [ - 0 . 0 6 , + 0 . 1 9 ]$ These results are consistent with the fact that the scene structure in this dataset is generally planar, so refraction has an overall radial efect that can be modeled with radial distortion parameters. Consistently, on Lizard Island, where the scene is topographically varied and refraction produces a strong non-radial component, COLMAP (ref) remains comparatively larger, with $\kappa \in [ - 0 . 3 7 , + 0 . 3 1 ]$

![](images/a42981be0229e136ed0cce203237bdc6c52fc210b3749c1f091bd039379e23e7.jpg)

![](images/d8ac8dec4c9747b7435cbf73b6f8c84d5817e1a3c3a194508814a11ca81d7080.jpg)  
Fig. 6: Planarity deviation maps for the Malaysia dataset p1\_s02 (left: Refractive COLMAP; right: $\mathrm { O u r s } + \mathrm { C O L M A P } ) .$ Axes denote normalized in-plane coordinates; color shows normalized signed deviation from the fitted plane. The left panel shows a strong dome artifact, a hallmark of optimization convergence overfitting. Our method (right) eliminates this systematic pattern, leaving only deviations from the actual 3D structure.

![](images/b38452d5a5d56dafb19474a92d85ed78edc0abd4cfd9a5060bd156dd39aa109c.jpg)

![](images/2e8b94241493edfa4727bd1ef836cff51b450af0bfe675c40ff4e2d2106c7ccd.jpg)  
(a) Malaysia (p1\_s02)  
(b) Lizard Island (feb\_gp1)  
Fig. 7: Depth ablation to select the operational depth $Z _ { 0 }$ used for refractive undistortion. We undistort images using a range of candidate values of $Z _ { 0 }$ and reconstruct the scene for each. Each plot shows the resulting distribution of normalized planarity residuals across the sequence.

Refractive COLMAP, with � ∈ [+0.10, +0.69] in Malaysia and $\kappa \in \left[ - 0 . 4 5 , + 0 . 0 7 \right]$ in Lizard Island, does not consistently outperform COLMAP (ref) with refined intrinsics, suggesting that explicitly modeling refraction hinders optimization convergence, likely causing it to settle in local minima.

In contrast, our image-space correction consistently produces the smallest structural curvature across the evaluated sequences, with $\kappa \in \mathsf { \Gamma } [ - 0 . 0 7 , + 0 . 0 8 ]$ in Malaysia and $\kappa \in$ $[ - 0 . 0 2 , + 0 . 0 5 ]$ in Lizard Island sequences.

## B. Reconstruction Completeness and Model Fit

As shown in Table II, our approach achieves the highest registered image ratio across all sequences with the highest geometric fidelity. As discussed in the previous section, COLMAP (ref) with refined distortion parameters consistently yields the lowest reprojection error on the Malaysia sequences, because the scene structure in this dataset is generally planar, so refraction has an overall radial efect that can be modeled with radial distortion parameters; however, this does not translate to the highest geometric fidelity.

Finally, another notable trend is that our approach produces the highest number of 3D points across 9/10 sequences and the highest mean track length across all sequences. This indicates that our correction does not merely reconstruct more points at the cost of weaker constraints; rather, it recovers more 3D points while each is observed more consistently across views, better constraining the reconstruction.

TABLE III: Generalization across 3D Reconstruction and VSLAM. Trajectory planarity (�) and number of registered images (Reg.) with our corrected inputs within each backend.  
![](images/e3ee91fcadc2c24cc5b51432a796e610e572a90fbd995cf4fbe5a9ed67d7a581.jpg)

TABLE IV: Runtime evaluation (ms / image). Our method achieves the fastest runtime across all stages and reports the fastest total runtime, despite introducing the additional refractive undistortion step.
<table><tr><td>Method</td><td>Undist.↓</td><td>Feat. Extraction↓</td><td>Feat. Matching↓</td><td>Mapping↓</td><td>Total↓</td></tr><tr><td>COLMAP (refined)</td><td>一</td><td>7.69</td><td>1158.80</td><td>3499.74</td><td>4666.23</td></tr><tr><td>Refractive COLMAP</td><td>一</td><td>8.42</td><td>1129.88</td><td>2934.84</td><td>4073.14</td></tr><tr><td>Ours + COLMAP</td><td>7.75</td><td>7.03</td><td>1015.05</td><td>2242.73</td><td>3272.56</td></tr></table>

## C. Generalization Across 3D Reconstruction and VSLAM

A key benefit of our approach is that, by correcting refraction in image space, it allows seamless combination with any 3D reconstruction or VSLAM pipeline that assumes a perspective (pinhole) camera model, many of which are trained on in-air data. Table III shows how our approach converts catastrophic failure into standard operating performance for each system (see accompanying video).

## VI. Ablation Studies

Reference depth. As shown in Sec. III-E, $Z _ { 0 }$ is strongly coupled with the flat-port parameter $d _ { 0 } ;$ we therefore vary $Z _ { 0 }$ directly and evaluate its efect on reconstructed geometry, using planarity as a proxy for geometric fidelity.

Fig. 7 shows that geometric fidelity is highest (curvature is minimized) at specific operational values $Z _ { 0 }$ . A limitation of this analysis is that we fit a single $Z _ { 0 }$ per scene, whereas the true optimal depth varies per pixel with local scene depth. The value we identify is therefore an average that minimizes the aggregate curvature over the whole scene, rather than the locally optimal depth at every point (see Fig. 6).

Computational Performance. Table IV shows that our approach speeds up feature matching (which includes geometric verification) and mapping, both by using a betterfitting model compared to COLMAP and a simpler model compared to Refractive COLMAP. We run all experiments on a desktop workstation with an Intel Core i7-14700, 32GB RAM, and an NVIDIA RTX 4090.

## VII. Conclusion and Future Work

We presented an image-space refraction correction method for underwater 3D reconstruction with flat-port cameras. Our method corrects the refractive displacement by warping images to their equivalent perspective appearance prior to reconstruction. A key benefit is that this enables any standard 3D reconstruction or VSLAM pipeline to run unmodified.

Through this work we have identified current limitations. Specifically, we determine a single operational scene distance $Z _ { 0 }$ by tuning it against prior knowledge of scene structure. However, this parameter should instead be obtained per pixel, either through stereo triangulation or recent monocular depth prediction models. More broadly, the field lacks an underwater dataset with highly accurate ground-truth geometry for 3D reconstruction.

## References

[1] T. Treibitz, Y. Schechner, C. Kunz, and H. Singh, “Flat refractive geometry,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 1, pp. 51–65, 2012.

[2] F. Chadebecq, F. Vasconcelos, R. Lacher, E. Maneas, A. Desjardins, S. Ourselin, T. Vercauteren, and D. Stoyanov, “Refractive two-view reconstruction for underwater 3d vision,” International Journal of Computer Vision, vol. 128, no. 5, pp. 1101–1117, 2020.

[3] V. Raoult, P. A. David, S. F. Dupont, C. P. Mathewson, S. J. O’Neill, N. N. Powell, and J. E. Williamson, “Gopros™ as an underwater photogrammetry tool for citizen science,” PeerJ, vol. 4, p. e1960, 2016.

[4] M. Singh and K. Alexis, “Online refractive camera model calibration in visual inertial odometry,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024, pp. 12 609–12 616.

[5] M. Singh, M. Dharmadhikari, and K. Alexis, “An online selfcalibrating refractive camera model with application to underwater odometry,” in IEEE International Conference on Robotics and Automation, 2024, pp. 10 005–10 011.

[6] M. She, F. Seegräber, D. Nakath, and K. Köser, “Refractive colmap: Refractive structure-from-motion revisited,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024, pp. 12 816– 12 823.

[7] J. Sauder, G. Banc-Prandi, A. Meibom, and D. Tuia, “Scalable semantic 3d mapping of coral reefs with deep learning,” Methods in Ecology and Evolution, vol. 15, no. 5, pp. 916–934, 2024.

[8] B. Gorry, T. Fischer, M. Milford, and A. Fontan, “Image-based relocalization and alignment for long-term monitoring of dynamic underwater environments,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025, pp. 10 749–10 756.

[9] A. Friedman, J. Monk, O. Pizarro, D. Lindsay, E. Oh, B. Thornton, A. Carroll, R. Przeslawski, and S. Williams, “Squidle+: a collaborative platform to manage, discover and annotate marine imagery,” Frontiers in Marine Science, vol. 12, p. 1677103, 2026.

[10] J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 4104–4113.

[11] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: visual geometry grounded transformer,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5294–5306.

[12] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in European Conference on Computer Vision, 2024, pp. 71–91.

[13] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: recovering the visual space from any views,” in International Conference on Learning Representations, 2026.

[14] A. Jordt-Sedlazeck and R. Koch, “Refractive structure-from-motion on underwater images,” in IEEE International Conference on Computer Vision, 2013, pp. 57–64.

[15] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, “WaterSplatting: Fast underwater 3D scene reconstruction using Gaussian Splatting,” in International Conference on 3D Vision, 2025, pp. 969–978.

[16] M. Kweon and J. Sattar, “Swimm3r: Splatting with medium-aware SfM for underwater 3D reconstruction,” arXiv 2608.00950, 2026.

[17] O. Bagoren, S. Isaacson, S. Sundar, Y.-C. Sun, A. Sheppard, H. Ma, A. Sharif, R. Vasudevan, and K. A. Skinner, “SurfSLAM: Sim-toreal underwater stereo reconstruction for real-time SLAM,” arXiv 2601.10814, 2026.

[18] S. Xu, K. Zhang, and S. Wang, “AQUA-SLAM: Tightly-coupled underwater acoustic-visual-inertial SLAM with sensor calibration,” IEEE Transactions on Robotics, vol. 41, pp. 2785–2803, 2025.

[19] A. Sedlazeck and R. Koch, “Perspective and non-perspective camera models in underwater imaging – overview and error analysis,” in Outdoor and Large-Scale Real-World Scene Analysis, 2012, pp. 212– 242.

[20] D. Senshina, D. Polevoy, E. Ershov, and I. Kunina, “Experimental study of radial distortion compensation for camera submerged underwater using open saltwaterdistortion data set,” Journal of Imaging, vol. 8, no. 10, 2022.

[21] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon, “Bundle adjustment — a modern synthesis,” in Vision Algorithms: Theory and Practice, 2000, pp. 298–372.

[22] D. Grifiths and H. Burningham, “Comparison of pre- and selfcalibrated camera calibration models for UAS-derived nadir imagery for a SfM application,” Progress in Physical Geography: Earth and Environment, vol. 43, no. 2, pp. 215–235, 2019.

[23] L. Magri and R. Toldo, “Bending the doming efect in structure from motion reconstructions through bundle adjustment,” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. XLII-2/W6, pp. 235–241, 2017.

[24] M. R. James and S. Robson, “Mitigating systematic error in topographic models derived from uav and ground-based image networks,” Earth Surface Processes and Landforms, vol. 39, no. 10, pp. 1413– 1420, 2014.

[25] C. Wu, “Critical configurations for radial distortion self-calibration,” in IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 25–32.

[26] V. Chari and P. Sturm, “Multiple-view geometry of the refractive plane,” in British Machine Vision Conference, 2009.

[27] J. M. Lavest, G. Rives, and J. T. Lapresté, “Underwater camera calibration,” in European Conference on Computer Vision, 2000, pp. 654–668.

[28] T. Łuczyński, M. Pfingsthorn, and A. Birk, “The pinax-model for accurate and eficient refraction correction of underwater cameras in flat-pane housings,” Ocean Engineering, vol. 133, pp. 9–22, 2017.

[29] F. Seegräber, M. She, F. Woelk, and K. Köser, “A calibration tool for refractive underwater vision,” in IEEE/CVF International Conference on Computer Vision Workshops, 2025, pp. 2087–2095.

[30] A. Agrawal, S. Ramalingam, Y. Taguchi, and V. Chari, “A theory of multi-layer flat refractive geometry,” in IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 3346–3353.

[31] N. Keetha et al., “MapAnything: Universal feed-forward metric 3D reconstruction,” in International Conference on 3D Vision, 2026.

[32] A. Fontan, T. Fischer, J. Civera, and M. Milford, “VSLAM-LAB: A Comprehensive Framework for Visual SLAM Methods and Datasets,” in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2025, pp. 9013–9020.

[33] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. M. Montiel, and J. D. Tardós, “ORB-SLAM3: An Accurate Open-source Library for Visual, Visual–inertial, and Multimap SLAM,” IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, 2021.

[34] A. Fontan, J. Civera, and M. Milford, “AnyFeature-VSLAM: Automating the Usage of Any Chosen Feature into Visual SLAM,” in Robotics: Science and Systems, vol. 2, 2024.

[35] Z. Teed and J. Deng, “DROID-SLAM: Deep Visual Slam for Monocular, Stereo, and RGB-D Cameras,” Advances in Neural Information Processing Systems, vol. 34, pp. 16 558–16 569, 2021.

[36] Z. Teed, L. Lipson, and J. Deng, “Deep Patch Visual SLAM,” in European Conference on Computer Vision, 2024, pp. 424–440.

[37] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 16 695–16 705.

[38] S. K. Leung, T. M. Kenyon, L. J. Raymundo, H. E. Fox, N. Cook, K. Cook, A. J. Edwards, E. E. Fisher, A. J. Brival, F. E. Nicholson et al., “A decision support tool for rubble stabilization on coral reefs,” Journal of Environmental Management, vol. 396, p. 128154, 2025.

[39] S. Mou, D. Tsai, and M. Dunbabin, “Reconfigurable robots for scaling reef restoration,” arXiv 2205.04612, 2022.

[40] L. Pan, D. Baráth, M. Pollefeys, and J. L. Schönberger, “Global structure-from-motion revisited,” in European Conference on Computer Vision, 2024, pp. 58–77.

[41] Y. Wang, L. Pan, M. Pollefeys, and V. Larsson, “Structure-from-motion with a non-parametric camera model,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 1040–1049.