# LARK: A Low-Cost, Accurate, Occlusion-Resilient, Kalman Filter-Assisted Tracking System for Image-Guided Surgery

George Siderisa,∗,1, Justin Creea,1, Andrew Stirlinga, Mamadou Lya, Étienne Légerb and D. Louis Collins<sup>b</sup>

<sup>a</sup>Department of Mechanical Engineering, McGill University, Montreal, Canada

<sup>b</sup>Montreal Neurological Institute, McGill University, Montreal, Canada

## A R T I C L E I N F O

Keywords:   
Image-Guided Surgery   
Optical Tracking   
Kalman Filtering   
Multi-Camera Calibration   
Multi-View Triangulation   
Pose Estimation   
Fiducial Markers

## A BS T RA C T

Image-guided surgery (IGS) depends on accurate tracking of surgical instruments to provide realtime navigation relative to anatomical structures. Commercial stereo infrared trackers are accurate but prone to occlusion and cost-prohibitive for many settings. This work presents LARK, a multi-camera optical tracking system using commodity RGB hardware and multi-view redundancy and fusion. We develop and evaluate two complete tracking methods: multi-view monocular pose fusion and multi-view triangulation. Both methods are assessed under varying occlusion levels using a precisionmachined grid and an anatomical head phantom, and compared against a gold-standard stereo infrared system. With five cameras and adaptive Kalman filtering, LARK achieves median target registration errors of 0.64 mm for point localization with triangulation and 0.73 mm for trajectory tracking with pose fusion on the machined grid. Camera-subset experiments show graceful degradation in adaptive pose-fusion accuracy as fewer views remain available. With tracking hardware costing under \$1,000 USD, LARK provides a low-cost platform for image-guided surgery research. Hardware designs and software are publicly available at https://nist.mni.mcgill.ca/software/, and datasets at https://nist.mni.mcgill.ca/data/.

## 1. Introduction

Image-guided surgery (IGS) has increasingly facilitated minimally invasive procedures, including renal, orthognathic, spinal, and brain surgeries, by providing realtime navigation tools that enhance accuracy and safety (Gueziri et al., 2021; Jackson et al., 2020; Van den Bempt et al., 2018; Wallace et al., 2019). IGS systems track the position of surgical instruments and align them with a three-dimensional model of the patient’s anatomy, created from pre- and/or intra-operative imaging. This information is visualized on a screen or head-mounted device (Asadi et al., 2024), allowing surgeons to accurately position their instruments relative to critical anatomical structures that are otherwise hidden from view. In the context of image-guided neurosurgery (IGNS), this feedback is crucial for planning and executing minimally invasive procedures, such as tumor resections and biopsies, where visibility is often limited (Taleb et al., 2023).

Tracking in IGNS is commonly achieved using either electromagnetic or optical systems, each with distinct advantages and limitations. Electromagnetic tracking systems typically use an external field generator and miniature sensors attached to or embedded in the surgical instrument. The sensors measure the generated field, allowing the system to estimate their position and orientation. While these systems do not require a direct line of sight, they are susceptible to interference from metal objects in the operating room (OR), which can significantly compromise their reliability and accuracy (Jin and Jung, 2022; Poulin and Amiot, 2002; Sorriento et al., 2020).

Optical tracking systems have become the dominant technology in IGNS due to their high accuracy and straightforward integration in the OR. These systems typically use stereo infrared cameras and computer-vision algorithms to detect retro-reflective markers afixed to surgical instruments and to a static reference afixed to the patient. However, they are highly susceptible to occlusion: when surgeons or staf obstruct the camera’s view, tracking is disrupted entirely, often freezing the navigation display and halting guidance until the view is restored (Han et al., 2023; Mehbodniya et al., 2019). In addition, the systems are often bulky, adding to the already crowded surgical workspace (Lin et al., 2023; Sorriento et al., 2020). Their cost, typically \$8,000 - \$25,000 USD for the tracking module alone, further limits adoption in resource-constrained settings (Léger et al., 2021).

In this paper, we address these limitations through a low-cost multi-camera optical tracking system mounted on a surgical light. The system addresses two distinct occlusion challenges: camera-level occlusion, where entire camera viewpoints are blocked by surgical staf or equipment, and marker-level occlusion, where individual markers are obscured from specific viewpoints. The hardware design, a ring-shaped fixture with distributed cameras, minimises camera-level occlusion by ensuring multiple lines of sight to the surgical field. To handle marker-level occlusion, we implement and compare two pose estimation methods with adaptive Kalman filtering: multi-view monocular pose fusion and multi-view triangulation. We evaluate both approaches using a precision-machined grid and an anatomical head phantom.

The contributions of this work are threefold. First, we present LARK, a low-cost multi-camera optical tracking system using commodity RGB hardware. The surgical-lightmounted ring fixture provides multiple views without external tracking towers. With tracking hardware costing under \$1,000 USD, LARK achieves sub-millimeter target registration error on the machined grid with five cameras and adaptive filtering (0.64 mm point localization with triangulation, 0.73 mm trajectory tracking with pose fusion).

Second, we develop a complete tracking methodology comprising a unified calibration pipeline and two pose estimation methods. The calibration pipeline includes multicamera extrinsic calibration with global bundle adjustment and instrument calibration through bundle-adjusted marker corner refinement. Building on this shared foundation, we implement and systematically compare two pose estimation methods: multi-view monocular pose fusion, which computes independent transformations from each camera view and combines them through adaptive Kalman filtering, and multi-view triangulation, which reconstructs 3D marker positions across cameras before estimating pose. Our comparison reveals complementary performance characteristics: with five cameras and adaptive filtering, monocular pose fusion achieves lower trajectory error on the grid, while triangulation achieves lower point and trajectory errors on the head phantom and is less afected by the illustrated poor single-marker view.

Third, we provide a comprehensive open-source benchmark for surgical tracking system evaluation. Using a precision-machined grid and an anatomical head phantom, we compare both methods across camera subsets representing diferent levels and patterns of camera unavailability. All hardware designs, bill of materials, software, and 5,000 multi-view tracking videos are publicly released to establish an accessible reference platform for IGS research.

The remainder of this paper is organized as follows. Section 2 reviews state-of-the-art optical tracking systems for neuronavigation and optical tracking using planar fiducial markers. Section 3 describes the system hardware, calibration, and pose estimation pipeline. Section 4 presents the experimental protocol. Section 5 presents the quantitative results. Section 6 discusses the system’s performance, limitations and clinical implications, and Section 7 concludes.

## 2. Related Work

## 2.1. State-of-the-Art Optical Tracking Systems

Contemporary state-of-the-art neuronavigation systems, such as Atracsys’ fusionTrack 500 (Atracsys, 2024), NDI’s Polaris Vega (Northern Digital Inc., 2024), Stryker’s NAV3i (Stryker Corporation, 2024), and Claronav’s Navient (ClaroNav, 2024), rely on a single stereo infrared camera for surgical instrument tracking. These systems achieve tracking through retro-reflective markers mounted on the surgical instrument and a reference fixed to the patient’s head. Fattori et al. (2021) demonstrated that NDI’s Polaris can achieve sub-millimeter accuracy with latency below 20 milliseconds from as far as 2.4 meters away. However, these systems are expensive, large, and prone to occlusion, which disrupts tracking. To expand the workspace of these systems, Pfeifer et al. (2016) employed multiple Polaris cameras with data fusion, while Marinetto et al. (2018) utilized multiple OptiTrack cameras, a more afordable alternative typically used in motion capture, but not approved for surgical use. Nevertheless, both approaches remain costly.

## 2.2. Optical Tracking Using Planar Fiducial Markers

In response to these limitations, several eforts have focused on using low-cost RGB cameras for instrument tracking through planar fiducial marker systems, with ArUco (Garrido-Jurado et al., 2014) and AprilTag (Olson, 2011) being among the most popular due to their accessibility and ease of use. However, studies by Asselin et al. (2018) and Léger et al. (2021) indicate that tracking a single planar marker with an of-the-shelf stereo RGB-Depth camera does not provide the accuracy required for surgical navigation. More recent eforts have explored using multiple planar markers attached to the instrument and/or multiple RGB cameras, either for multi-view monocular tracking with intelligent pose fusion or for stereo tracking.

## 2.2.1. Multi-Marker Instrument Tracking

Research has been done on tracking configurations of multiple fiducial markers to improve object tracking from diferent perspectives. Wu et al. (2017) proposed DodecaPen, which tracks a dodecahedron of markers using a single camera positioned near the object. However, their work does not consider the larger distances typically observed in OR settings and does not address occlusion at the camera or marker level. García-Ruiz et al. (2023) introduced DodUcoPose, an improvement on the DodecaPen system, designed to track objects using a configuration of multiple pentagonal markers arranged in a dodecahedron. However, DodUcoPose’s tracking accuracy evaluations remain confined to relatively short object-to-camera distances, and the method is based on a singular camera viewpoint, making it susceptible to camera-level occlusion. More recently, Moviglia and Stallkamp (2026) paired a single near-infrared camera with compact retroreflective dodecahedral fiducials, achieving 0.44 mm translational accuracy at 30–70 cm working distance, though from static position estimates, at short range, and again from a single viewpoint.

## 2.2.2. Multi-View RGB Tracking

Pose estimation using multiple RGB cameras requires precise calibration. For monocular tracking, intrinsic calibration is necessary for each individual camera, while multiview triangulation demands both intrinsic and extrinsic calibration for all cameras involved. Following calibration, data fusion is typically employed to enhance localization accuracy, particularly in scenarios involving partial occlusion or degraded views from one or more cameras. Wang et al. (2015) proposed a fast non-iterative multi-camera calibration algorithm for determining the extrinsic parameters of a set of monocular cameras, although their approach does not account for intrinsic camera parameters. Vaccarella et al. (2013) introduced an unscented Kalman filter-based sensor fusion method to mitigate marker occlusions and electromagnetic distortions in surgical navigation using an electromagnetic tracking system. Popescu et al. (2020) proposed an adaptive Kalman filter method for fusing planar marker poses from multiple cameras, but their approach assumes static dynamics and their results were only tested in simulation. Tsui et al. (2024) achieved a root-mean-square error of 2 mm when tracking a planar marker using a custom stereo camera pair with a wider baseline than typical ofthe-shelf systems. However, their evaluation setup did not mimic OR conditions, as the marker was only tracked within a single plane of movement and at distances typically less than a meter away from the camera pair.

## 2.2.3. Hybrid Multi-Camera, Multi-Marker Systems

Chen et al. (2023) mounted multiple RGB camera modules on a surgical light and tracked surgical instruments using planar fiducial markers. Their system uses multi-camera calibration and least-squares fusion and reported mean point and trajectory errors of 2.01 mm and 2.37 mm, respectively. However, the evaluation was limited to millimeter-level accuracy rather than sub-millimeter accuracy at neurosurgical working distances, and the system design and software were not released publicly.

In this work, we pursue a similar overhead, lightmounted form factor but target sub-millimeter localization accuracy using only commodity hardware. We also consider explicit loss of individual camera views and we present two complementary estimation modes, multi-view monocular pose fusion and multi-view triangulation. The full mechanical fixture, calibration, tracking and registration software, and bill of materials are made available for reproduction.

## 2.3. Marker-Less Instrument Tracking

Marker-less multi-view pose estimation has recently emerged as an alternative that leaves instruments unmodified. Hein et al. (2025) evaluate learning-based methods on a multi-camera dataset of ex-vivo spine surgeries, reporting 1.01 mm instrument position error with five static cameras under optimal in-domain conditions, degrading for a second instrument (2.79 mm) and in an unseen operating room. Accuracy is reported as pose error against a marker-based tracker whose own calibration accuracy the errors approach, rather than as registration-based target error in a patient reference frame, and each instrument requires its own model and training data. Marker-based tracking therefore remains the reference standard where sub-millimeter, instrumentagnostic accuracy is required.

This work compares pose fusion and triangulation on shared hardware and recordings under controlled camerasubset removal, and releases the hardware designs, software, and data as a benchmark for further evaluation.

## 3. System Overview and Methods

## 3.1. System Overview

The proposed system consists of a ring-shaped fixture equipped with multiple RGB cameras, secured atop a surgical light to maintain a clear line of sight to the surgical field (see Fig. 1). A surgical pointer instrument and a static reference are each equipped with planar fiducial markers, enabling tracking of the pointer with respect to the reference. Because the tracked entity is the marker assembly rather than the instrument itself, any rigid instrument with a welldefined working point can be tracked after a one-time pivot calibration, without per-instrument models or training. The system is capable of two tracking methods that are evaluated and compared here, both employing adaptive Kalman filtering: one based on fusion of multi-view monocular pose estimates, and one based on multi-view triangulation. Real-time visualization and navigation are achieved through integration with the IBIS neurosurgical navigation platform (Drouin et al., 2017), which displays the tracked pointer position on patient imaging data. Figure 2 demonstrates the system’s real-time tracking capability, showing the pointer position overlaid on multi-planar MRI views. The supplementary video demonstrates the assembled system, both experimental setups, and real-time tracking with the navigation display.

## 3.2. Hardware Design

## 3.2.1. Multi-Camera Fixture

The primary design objective of our system is to maintain line of sight with the surgical instruments and patient without disrupting surgical workflow and while preserving the OR workspace. To achieve this, a ring-shaped fixture is designed and built to accommodate the attachment of multiple cameras (in this work, we use 5) and be placed on surgical lights. The fixture is composed of eight 3D-printed interlocking segments joined with fasteners to form a ring. The cameras are placed in a pentagonal formation around the ring, with gaps for the lateral handles of the surgical light (see Fig. 1 top right). Each camera is mounted via an adjustable locking hinge, positioning its field of view at an approximately 30<sup>◦</sup> inclination from the horizontal so that the lines of sight converge roughly one meter below the light. This arrangement maximizes the likelihood of overlapping camera views, reducing the chance that a single occlusion obscures all viewpoints. All additional electronics are fastened to the fixture’s upper surface. Three toggle clamps secure it to the surgical lamp, allowing the entire assembly to be attached or detached via quick-release levers. The supplementary document provides the bill of materials (Supplementary Figure S1) and the assembly and component drawings (Supplementary Figures S2–S25).

![](images/d9cd54bfd7999d4a4313ff0dd3a8965ac386943d01a927a7598105c972685af0.jpg)  
Figure 1: Annotated schematic of the proposed system. The system is composed of a multi-camera fixture (top left), a surgical pointer and a static reference (bottom left). The fixture is attached to a surgical light to track the pointer relative to the reference (right). Typical surgical light geometries consist of an articulated arm whose base is mounted to the OR ceiling. The position of the light can be freely manipulated to illuminate the surgical area, and by doing so, ensure that the cameras maintain line of sight with the anatomy of interest.

![](images/127dea9222ea3cf2c2a53d983238722b5f492de77d0dfcf3189341942d5f584f.jpg)  
Figure 2: Real-time surgical navigation interface using the IBIS platform (Drouin et al., 2017). The tracked pointer position (red dot) is displayed on axial, coronal, and sagittal MRI views, with a 3D rendering showing the pointer, reference dodecahedron, and patient anatomy.

## 3.2.2. Surgical Pointer and Static Reference

To estimate the pose of the surgical instrument relative to the reference fixed to the patient, planar fiducial markers of the ArUco type (Garrido-Jurado et al., 2014) are employed, which are detected using OpenCV (Bradski, 2000). To increase resilience against marker-level occlusion and rotational ambiguity, multiple markers are attached to a regular dodecahedron, as demonstrated by successful approaches in the literature (García-Ruiz et al., 2023; Wu et al., 2017). An ArUco marker is centered on each face of the dodecahedron, except for the base, which attaches to the instrument handle or the reference shaft (see Fig. 1 bottom left). The geometry of the dodecahedron ensures that at least three faces remain visible from most viewing angles, enabling more robust pose estimation compared to simpler Platonic solids.

A 3D-printed part is fabricated to integrate the dodecahedron with a handle featuring a threaded insert for attachment of custom surgical instruments. A metal tip similar to pointer instruments employed in surgical navigation is afixed at the handle’s distal end, enabling precise sampling of measurement points and trajectories. The dodecahedron has a 77 mm circumscribed diameter to accommodate the placement of 24×24 mm ArUco markers on each face, balancing reliable detection with a user-friendly form factor. The surgical pointer features a handle diameter of 5.4 mm and a total length of 220 mm from the center of the dodecahedron’s top face to the end of the metal tip.

A similar dodecahedron of 112 mm circumscribed diameter is used for the static reference, afixed with 33×33 mm ArUco markers. In our experimental setup, the reference dodecahedron is rigidly connected to the landmarks being measured. In an actual clinical workflow, the reference dodecahedron is fixed to a head clamp or an equivalent patient immobilizing structure, enabling tracking of the pointer relative to the patient irrespective of camera movement.

## 3.2.3. Electronics and Optics

In selecting the electronics and optics, the objective is to balance image quality, field of view, and communication speed with modularity and afordability. Raspberry Pi single-board computers (1 GB) are selected for their low cost and ability to enable low-latency transmission over Ethernet, and are paired with MIPI camera modules. The MIPI interface supports direct camera integration with the selected Raspberry Pi boards. The Raspberry Pi Global Shutter Camera (5.02 mm sensor width, $1 4 5 6 \times 1 0 8 8$ at 60 fps, 3.45 �m pixel size) is chosen for its oficial support and C/CS lens mount, allowing for focal length customization (Raspberry Pi, 2026). A 6 mm lens is selected to provide a wide field of view covering the head and the 220 mm pointer at a 1 m working distance. Focal length $f$ is estimated by $f =$ $\frac { H { \times } \mathrm { W D } } { \mathrm { W i d t h } }$ , where � is the sensor width, WD is the working distance, and Width is the subject size. For the stated sensor width, focal length, and working distance, this approximation gives a horizontal field width of approximately 837 mm.

## 3.3. Calibration Pipeline

## 3.3.1. Multi-Camera Calibration

Every camera $c _ { i }$ defines a right-handed Cartesian frame $\mathcal { F } _ { c _ { i } }$ whose origin is its optical centre, �-axis points forward along the viewing direction, and �- and �-axes coincide with the image axes (+� to the right, +� downwards) (see Fig. 3). Throughout this work, the frame of camera 1, $\boldsymbol { { \mathcal { F } } } _ { c _ { 1 } } .$ , serves as the global coordinate frame: all extrinsic transforms $\mathbf { E } _ { i }$ map the global frame $\mathcal { F } _ { c _ { 1 } }$ into each camera frame $\boldsymbol { \mathcal { F } } _ { c _ { i } } ,$ with ${ \bf E } _ { 1 } =$ $\mathbf { I } _ { 4 \times 4 }$ , and every 3D point written without an explicit camera coordinate frame subscript is understood to be expressed in $\mathcal { F } _ { c _ { 1 } }$

To calibrate the system we acquire images of a ChArUco board (see Fig. 3) at many poses by positioning the board throughout the system’s field of view. The board itself carries a fixed coordinate frame $\mathcal { F } _ { b }$ whose origin is at its top left corner and whose �-axis is normal to the printed surface. The 2D pixel position of the �-th ChArUco corner observed in camera � at video frame (time step) � is denoted $\mathbf { x } _ { k . t . i } ^ { \mathbf { b } } \in \mathbb { R } ^ { 2 }$ and the known 3D location of the same corner in the board coordinate frame is $\mathbf { X } _ { k } ^ { \mathbf { b } } \in \mathbb { R } ^ { 3 }$

For each frame �, $\mathbf { C } _ { t } ^ { b c _ { i } }$ and $\mathbf { r } _ { t } ^ { b c _ { i } }$ describe the calibration board’s rotation and translation into camera $i ^ { \circ } \mathrm { s }$ coordinate frame. Intrinsic calibration jointly estimates the intrinsics $\mathbf { M } _ { i }$ of each camera and these per-frame board poses by minimising the reprojection error between every detected $\mathbf { x } _ { k , t , i } ^ { \mathbf { b } }$ and its pinhole projection $\pi \big ( \mathbf { C } _ { t } ^ { b c _ { i } } \mathbf { X } _ { k } ^ { \mathbf { b } } + \mathbf { r } _ { t } ^ { b c _ { i } } ; \mathbf { M } _ { i } \big )$ . Thus, the error is always measured between a 2D corner observed in the image and the same corner re-projected with the current camera intrinsics. In our notation, the symbol

$$
\mathbf { M } _ { i } : = \left( \tilde { \mathbf { M } } _ { i } , \mathbf { \Gamma } \mathbf { } d _ { i } \right)
$$

always subsumes both the $3 \times 3$ pinhole camera matrix

$$
\begin{array} { r } { \tilde { \mathbf { M } } _ { i } = \left[ \begin{array} { c c c } { f _ { x } } & { s } & { c _ { x } } \\ { 0 } & { f _ { y } } & { c _ { y } } \\ { 0 } & { 0 } & { 1 } \end{array} \right] , } \end{array}
$$

and the radial–tangential distortion vector

$$
\mathbf { { \pmb d } } _ { i } = [ k _ { 1 } , k _ { 2 } , k _ { 3 } , p _ { 1 } , p _ { 2 } ] ^ { \top } .
$$

Once $\tilde { \mathbf { M } } _ { i }$ and $\pmb { d } _ { i }$ have been identified, they enable accurate tracking through geometric projection and distortion correction.

Extrinsic calibration follows in two stages. First, for each camera $i \neq 1$ , stereo calibration uses board corners simultaneously visible in cameras 1 and �. With both cameras intrinsics $\mathbf { M } _ { 1 }$ and $\mathbf { M } _ { i }$ fixed, the camera-to-camera transform $\mathbf { E } _ { 1 i }$ and the board pose in each shared frame are jointly estimated by minimising reprojection error in both cameras. $\mathbf { A }$ subsequent global bundle-adjustment jointly refines the intrinsics $\mathbf { M } _ { i }$ and extrinsics $\mathbf { E } _ { i } ,$ , with ${ \bf E } _ { 1 } = { \bf I } _ { 4 \times 4 }$ fixed. At each objective evaluation, board corners are triangulated in the global frame (see Alg. 3) and board poses are estimated using OpenCV’s Perspective-n-Point (PnP) implementation (Bradski, 2000), both using the current calibration parameters. Each triangulated corner is transformed into camera $i \gamma _ { \mathrm { s } }$ frame using $\mathbf { E } _ { i }$ before comparison with the corresponding corner predicted by that camera’s PnP board pose. The sum of squared 3D discrepancies across observed corners, frames, and cameras is minimised, as summarised in Algorithm 1.

## 3.3.2. Instrument Calibration

Accurate tracking requires a good estimate of the relative 3D positions of every ArUco marker corner on the pointer and static reference: inaccurate estimates translate directly into poor pose estimation. We therefore take video of the tags and use bundle adjustment to arrive at a least-squares estimate that does not depend on precise manufacturing. This approach refines the initial corner coordinates on both instruments.

The instrument coordinate frames are defined as follows: the origin of the pointer coordinate frame ${ \mathcal { F } } _ { p }$ is at the physical pointer tip; the origin of the reference coordinate frame $\textstyle { \mathcal { F } } _ { r }$ is at the centre of its uppermost marker. Let $\mathbf { X } _ { k } ^ { \mathbf { d } , \mathrm { i n i t } } \in \mathbb { R } ^ { 3 }$ be the initial coordinate of dodecahedron marker corner � in its instrument frame $( \boldsymbol { \mathcal { F } } _ { p } \operatorname { o r } \boldsymbol { \mathcal { F } } _ { r } )$

During pointer calibration the tip is placed on the origin of the fixed board frame $\mathcal { F } _ { b }$ (see Fig. 3) and the shaft is pivoted and rotated about its axis while $T$ synchronised video frames are recorded from each camera. The initial tip position $\mathbf { p } ^ { \mathrm { i n i t } }$ is estimated by averaging the board origins detected across all � frames.

For every frame � and camera � the pixel coordinates $\mathbf { x } _ { k , t , i } ^ { \mathbf { d } }$ of each visible corner are detected and, using the previously calibrated intrinsics $\mathbf { M } _ { i }$ and extrinsics $\mathbf { E } _ { i }$ , are undistorted and triangulated (see Alg. 3) to the global points $\tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } .$ The initial corner set and the measured cloud are aligned independently in every video frame. For the pointer corner set $_ { p }$ we solve for a rotation $\mathbf { U } _ { t } \in S O ( 3 )$ about the fixed tip while for the reference corner set  we solve for a full rigid transformation $( \mathbf { U } _ { t } , \mathbf { v } _ { t } ) \in S E ( 3 )$ . Here $S O ( 3 ) = \{ \mathbf { U } \in$ $\begin{array} { r } { \mathbb { R } ^ { \mathbf { \breve { 3 } } \times 3 } \mid \mathbf { U } ^ { \mathsf { T } } \mathbf { U } = \mathbf { I } , } \end{array}$ det ${ \bf U } = 1 \}$ is the set of all 3D rotation matrices, and $S E ( 3 ) = \{ ( \mathbf { U } , \mathbf { v } ) \mid \mathbf { U } \in S O ( 3 ) , \ \mathbf { v } \in \mathbb { R } ^ { 3 } \}$ is the set of all rigid-body motions composed of a rotation � and a translation �.

Algorithm 1 Multi-Camera Calibration   
Input: Synchronized frames $t \in \{ 1 , \ldots , T \}$ from cameras   
$i \in \{ 1 , \ldots , N \}$ ; known 3D ChArUco board corners $\mathbf { X } _ { k } ^ { \mathrm { b } }$   
Output: Camera intrinsics $\mathbf { M } _ { i }$ and extrinsics $\mathbf { E } _ { i }$ mapping   
global coordinates into each camera’s frame.   
1. Pre-processing   
Detect 2D corners $\mathbf { x } _ { k , t , i } ^ { \mathrm { b } }$ in every image.   
Keep frames where the board is observed by at least two   
cameras.   
2. Intrinsic calibration   
for $i = 1$ to � do   
Jointly estimate $\mathbf { M } _ { i }$ and the per-frame board poses   
$\{ \mathbf { C } _ { t } ^ { b c _ { i } } , \mathbf { r } _ { t } ^ { b c _ { i } } \} _ { t }$ by minimising   
$\sum _ { k . t } \left\| \mathbf { x } _ { k , t , i } ^ { \flat } - \pi \big ( \mathbf { C } _ { t } ^ { b c _ { i } } \mathbf { X } _ { k } ^ { \flat } + \mathbf { r } _ { t } ^ { b c _ { i } } ; \mathbf { M } _ { i } \big ) \right\| ^ { 2 }$   
end for   
3. Pair-wise extrinsic initialization   
$\mathbf { E } _ { 1 }  \mathbf { I } _ { 4 \times 4 }$ ⊳ reference camera pose   
for $i = 2$ to $N$ do   
With $\mathbf { M } _ { 1 }$ and $\mathbf { M } _ { i }$ fixed, jointly estimate $\mathbf { E } _ { 1 i }$ and the   
board pose in each shared frame by minimising reprojec  
tion error in both cameras.   
$\mathbf { E } _ { i }  \mathbf { E } _ { 1 i }$   
end for   
4. Global bundle adjustment   
With ${ \bf E } _ { 1 } = { \bf I } _ { 4 \times 4 }$ fixed, jointly refine $\{ \mathbf { M } _ { i } , \mathbf { E } _ { i } \}$ by minimis  
ing   
$\sum _ { k \neq i } \Big \| \mathbf { C } ^ { c _ { 1 } c _ { i } } \tilde { \mathbf { X } } _ { k , t } ^ { \mathrm { b } } + \mathbf { r } ^ { c _ { 1 } c _ { i } } - \big ( \mathbf { C } _ { t } ^ { b c _ { i } } \mathbf { X } _ { k } ^ { \mathrm { b } } + \mathbf { r } _ { t } ^ { b c _ { i } } \big ) \Big \| ^ { 2 }$   
K,l,   
where $( \mathbf { C } ^ { c _ { 1 } c _ { i } } , \mathbf { r } ^ { c _ { 1 } c _ { i } } )$ are the rotation and translation of   
$\mathbf { E } _ { i } , \tilde { \mathbf { X } } _ { k , t } ^ { \mathrm { b } }$ is the triangulated corner in the global frame,   
and $( \mathbf { C } _ { t } ^ { b c _ { i } } , \mathbf { r } _ { t } ^ { b c _ { i } } )$ is the board pose estimated by PnP for   
camera �. Triangulation and PnP are recomputed using the   
current calibration parameters at each objective evalua   
tion; the sum includes only observed corners.   
return $\{ \mathbf { M } _ { i } , \mathbf { E } _ { i } \}$

The relative corner coordinates are refined across all video frames by minimising

$$
\begin{array} { r l } & { \arg \underset { \mathbf { X } , \mathbf { p } } { \operatorname* { m i n } } \sum _ { k , t \in \mathcal { P } } { \left\| \mathbf { U } _ { t } \mathbf { X } _ { k } ^ { \mathrm { d } } + \mathbf { p } - \tilde { \mathbf { X } } _ { k , t } ^ { \mathrm { d } } \right\| ^ { 2 } } , \quad \mathrm { ~ f o r ~ t h e ~ p o i n t e r } , } \\ & { \arg \underset { \mathbf { X } } { \operatorname* { m i n } } \sum _ { k , t \in \mathcal { R } } { \left\| \mathbf { U } _ { t } \mathbf { X } _ { k } ^ { \mathrm { d } } + \mathbf { v } _ { t } - \tilde { \mathbf { X } } _ { k , t } ^ { \mathrm { d } } \right\| ^ { 2 } } , \quad \mathrm { f o r ~ t h e ~ r e f e r e n c e } . } \end{array}
$$

![](images/4e7943f0e2ec31dbdbdabc30e29ac62f6e59e2f5e7eedb185871bfbd1ce308db.jpg)  
Figure 3: Instrument calibration schematic. The pointer is pivoted and rotated about its axis on the ChArUco board origin (bottom left) while video frames are recorded. After initialisation the dodecahedron marker corner coordinates are refined.

Instrument calibration outputs the refined corner set $\{ \mathbf { X } _ { k } ^ { \mathbf { d } } \}$ in its respective instrument frame $( \mathcal { F } _ { p } \ : \mathrm { o r } \ : \mathcal { F } _ { r } )$

## 3.4. Pose Estimation Methods

## 3.4.1. Multi-View Monocular Pose Fusion

Multi-view monocular pose fusion estimates the pointer pose relative to the static reference by computing independent transformations from each camera view and combining them with an adaptive Kalman filter. Let $\mathbf { r } ^ { p r } \in \mathbb { R } ^ { 3 }$ be the position of ${ \mathcal { F } } _ { p }$ with respect to $\textstyle { \mathcal { F } } _ { r }$ and $\mathbf { C } ^ { p r } \in S O ( 3 )$ the corresponding orientation. Together they form the homogeneous pose

$$
{ \bf T } ^ { p r } = \left[ \begin{array} { c c } { { \bf C } ^ { p r } } & { { \bf r } ^ { p r } } \\ { { \bf 0 } ^ { \top } } & { 1 } \end{array} \right] \in S E ( 3 ) .\tag{1}
$$

Here, $\mathbf { T } ^ { a b }$ maps homogeneous coordinates from frame ${  { \boldsymbol { F } } } _ { a }$ to frame $\mathcal { F } _ { b }$

For every camera $c _ { i }$ with an unobstructed view, the ArUco detector (Garrido-Jurado et al., 2014) yields 2D marker corners; a PnP algorithm returns the monocular pointer and reference poses $\mathbf { T } ^ { p c _ { i } }$ and $\mathbf { T } ^ { r c _ { i } }$ in $\mathcal { F } _ { c _ { i } } .$ Their composition gives an independent estimate of the pointer–to–reference transform

$$
\mathbf { T } _ { i } ^ { p r } = \left( \mathbf { T } ^ { r c _ { i } } \right) ^ { - 1 } \mathbf { T } ^ { p c _ { i } } .\tag{2}
$$

The set of computed transforms $\{ \mathbf { T } _ { i } ^ { p r } \}$ is fused frameby-frame with an adaptive Kalman filter, yielding the final smoothed estimate $\mathbf { T } _ { f i n a l } ^ { p \hat { r } }$ delivered to the navigation system for each frame.

The adaptive Kalman filter is used to improve pose estimation accuracy under measurement noise. While Popescu et al. (2020) use a zero velocity model, our approach updates both position and velocity in the state vector. Unlike their method, which updates noise covariances over all past data, we dynamically adjust the pose block of the process noise covariance matrix $\mathbf { Q } _ { t }$ and the measurement noise covariance matrix $\mathbf { R } _ { t }$ at each frame � using valid pose estimates from a rolling window of up to 30 frames. This allows the filter to respond more efectively to recent changes in noise caused from occlusions, pointer motion, and variations in lighting conditions.

Algorithm 2 Instrument Calibration   
Input: Synchronized video frames $t \in \{ 1 , \dots , T \}$ from   
cameras $i \in \{ 1 , \ldots , N \}$ ; ChArUco board corners $\mathbf { X } _ { k } ^ { \mathbf { b } }$ (for   
pointer calibration only); camera intrinsics $\mathbf { M } _ { i }$ and ex  
trinsics $\mathbf { E } _ { i } \left( \mathrm { A l g } . \ 1 \right)$ ; initial dodecahedron marker corners   
$\mathbf { X } _ { k } ^ { \mathbf { d } , \mathrm { i n i t } }$   
Output: Refined corners $\mathbf { X } _ { k } ^ { \mathbf { d } } .$   
1. Pre-processing   
if pointer then   
Initialize $\mathbf { p } ^ { \mathrm { i n i t } }$ as the mean ChArUco-board origin   
across   
all frames   
end if   
2. Marker detection and triangulation   
for $t = 1$ to � do   
Detect $\mathbf { x } _ { k , t , i } ^ { \mathbf { d } }$ in each camera $i ;$ keep corners seen by $\geq 2$   
cameras   
for all visible corners � do   
$\tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } \gets \mathrm { t r i a n g u l a t e } \big ( \mathbf { M } _ { i } , \mathbf { E } _ { i } , \mathbf { x } _ { k , t , i } ^ { \mathbf { d } } \big )$   
end for   
end for   
3. Frame-wise alignment   
for $t = 1$ to $T$ do   
if pointer then   
Estimate $\mathbf { U } _ { t }$ aligning $\{ \mathbf { X } _ { k } ^ { \mathbf { d } } \}$ to $\{ \tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } - \mathbf { p } ^ { \mathrm { i n i t } } \}$   
$\Delta \mathbf { X } _ { k , t } ^ { \mathbf { d } } \gets \mathbf { U } _ { t } ^ { \top } \big ( \mathbf { U } _ { t } \mathbf { X } _ { k } ^ { \mathbf { d } } + \mathbf { p } ^ { \mathrm { i n i t } } - \tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } \big )$   
else   
Estimate $( \mathbf { U } _ { t } , \mathbf { v } _ { t } )$ aligning the two sets   
$\Delta \mathbf { X } _ { k , t } ^ { \mathbf { d } } \gets \mathbf { U } _ { t } ^ { \dagger } \left( \mathbf { \dot { U } } _ { t } \mathbf { X } _ { k } ^ { \mathbf { d } } + \mathbf { v } _ { t } ^ { - } - \tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } \right)$   
end if   
end for   
$\begin{array} { r l } & { \mathbf { X } _ { k } ^ { \mathbf { d } } \gets \mathbf { X } _ { k } ^ { \mathbf { d } } - \frac { \sum _ { t \in \mathcal { T } _ { k } } w _ { k , t } \Delta \mathbf { X } _ { k , t } ^ { \mathbf { d } } } { \sum _ { t \in \mathcal { T } _ { k } } w _ { k , t } } } \end{array}$   
where $\tau _ { k }$ contains the frames in which corner $k$ is tri  
angulated, and $\begin{array} { c c } { \displaystyle { w _ { k , t } ~ = ~ \binom { n _ { k , t } } { 2 } } } \end{array}$ , with $n _ { k , t }$ the number of   
cameras observing its marker in frame �. The errors $\Delta \mathbf { X } _ { k , t } ^ { \mathbf { d } }$   
are expressed in the instrument frame.   
Iterate to convergence.   
Reject a marker observation in a frame if any component   
of its corner errors exceeds 2� in magnitude, where �   
is the standard deviation across all valid corner-error   
components.   
4. Bundle adjustment   
if pointer then   
$( \{ \mathbf { X } _ { k } ^ { \mathbf { d } } \} , \mathbf { p } )  \arg \operatorname* { m i n } _ { \{ \mathbf { X } _ { k } ^ { \mathbf { d } } \} , \mathbf { p } }$   
$\sum _ { k , t } { \left\| \mathbf { U } _ { t } \mathbf { X } _ { k } ^ { \bf d } + \mathbf { p } - \tilde { \mathbf { X } } _ { k , t } ^ { \bf d } \right\| ^ { 2 } }$   
else   
$\begin{array} { r } { \mathbf { X } _ { k } ^ { \mathbf { d } }  \arg \operatorname* { m i n } _ { \{ \mathbf { X } _ { k } ^ { \mathbf { d } } \} } \sum _ { k , t } \biggl \| \mathbf { U } _ { t } \mathbf { X } _ { k } ^ { \mathbf { d } } + \mathbf { v } _ { t } - \tilde { \mathbf { X } } _ { k , t } ^ { \mathbf { d } } \biggr \| ^ { 2 } } \end{array}$   
end if   
Recompute frame alignments at each objective evaluation.   
return $\big \{ \mathbf { X } _ { k } ^ { \mathbf { d } } \big \}$

![](images/fd175b31b2972f34d279cdc5a0f3ffdcab164ab00b7255e25acedf6fb32c9001.jpg)  
Figure 4: Multi-view monocular pose fusion schematic. For each camera that views the surgical pointer and static reference, an independent monocular pose measurement is computed through the PnP algorithm. At each frame, an adaptive Kalman filter is used to fuse all available measurements.

A process model is defined for the surgical pointer. Due to the steady motion of the pointer, especially around points of interest, a constant velocity model is assumed. Moreover, as no proprioceptive measurement of the pointer itself is available, any estimates of the acceleration are inherently noisy. Thus, the state is defined to only consist of the translational and rotational position and velocity of the pointer. Specifically, the state vector $\mathbf { x } _ { t }$ at frame � is represented as,

$$
\mathbf { x } _ { t } = \left[ ( \mathbf { r } ^ { p r } ) ^ { \top } \quad ( \pmb { \theta } ^ { p r } ) ^ { \top } \quad ( \dot { \mathbf { r } } ^ { p r } ) ^ { \top } \quad ( \dot { \pmb { \theta } } ^ { p r } ) ^ { \top } \right] ^ { \top } ,\tag{3}
$$

where $\mathbf { r } ^ { p r }$ represents the position of the pointer relative to the reference as defined in Eq. 1, $\theta ^ { p r } ~ = ~ ( \chi ^ { p r } , \phi ^ { p r } , \psi ^ { p r } )$ is the orientation of the pointer relative to the reference’s coordinate frame expressed as Euler angles, and the dot notation denotes the corresponding velocities. For simplicity, all coordinate frame subscripts of the position and orientation variables are omitted with the understanding that they are expressed relative to the static reference’s coordinate frame, unless otherwise stated.

The state transition model is written in discrete time as,

$$
\begin{array} { r } { \mathbf { x } _ { t } = \mathbf { A } \mathbf { x } _ { t - 1 } + \mathbf { w } _ { t - 1 } , \qquad \mathbf { w } _ { t - 1 } \sim \mathcal { N } ( 0 , \mathbf { Q } _ { t - 1 } ) , } \end{array}\tag{4}
$$

where � is the state transition matrix, and $\mathbf { w } _ { t - 1 }$ is the process noise, assumed to be zero-mean, white Gaussian noise with covariance matrix $\mathbf { Q } _ { t - 1 }$ . The state transition matrix for a constant velocity model is

$$
\mathbf { A } = \left[ \begin{array} { c c } { \mathbf { I } _ { 6 } } & { \mathbf { I } _ { 6 } \Delta t } \\ { \mathbf { 0 } _ { 6 } } & { \mathbf { I } _ { 6 } } \end{array} \right] ,\tag{5}
$$

where $\mathbf { I } _ { 6 }$ is the $6 \times 6$ identity matrix, ${ \bf 0 } _ { 6 }$ is the $6 \times 6$ zero matrix, and $\Delta t$ is the time interval between measurements.

At each frame �, measurements are obtained from multiple synchronised cameras through an electrical signal. Each camera that detects both the pointer and the static reference is considered active and provides an independent pose estimate of the pointer relative to the reference using the PnP algorithm. The measurement vector $\mathbf { y } _ { t }$ consists of the observed positions and orientations from all active cameras. The measurement vector from the �-th active camera is written as

$$
\mathbf { y } _ { t , i } = \left[ \mathbf { r } _ { t , i } ^ { p r } \right] , \qquad i = 1 , \ldots , n .\tag{6}
$$

The overall measurement vector $\mathbf { y } _ { t }$ is constructed by stacking the measurements from all � active cameras such that

$$
\mathbf { y } _ { t } = { \left[ \begin{array} { l } { \mathbf { y } _ { t , 1 } } \\ { \vdots } \\ { \mathbf { y } _ { t , n } } \end{array} \right] } .\tag{7}
$$

The measurement model relates the state vector to the measurements through the measurement matrix $\mathbf { H } _ { t }$ such that

$$
\mathbf { y } _ { t } = \mathbf { H } _ { t } \mathbf { x } _ { t } + \mathbf { n } _ { t } , \qquad \mathbf { n } _ { t } { \sim } \mathcal { N } ( \mathbf { 0 } , \mathbf { R } _ { t } ) ,\tag{8}
$$

where ${ \bf n } _ { t }$ is the measurement noise, assumed to be zero-mean Gaussian with covariance $\mathbf { R } _ { t } .$

Because the position and orientation are measured but their corresponding velocities are not, the measurement matrix $\mathbf { H } _ { t }$ at video frame � is constructed by vertically stacking submatrices corresponding to each camera � as

$$
\begin{array} { r } { \mathbf { H } _ { t } = \left[ \begin{array} { c c } { \mathbf { I } _ { 6 } } & { \mathbf { 0 } _ { 6 } } \\ { \vdots } & { \vdots } \\ { \mathbf { I } _ { 6 } } & { \mathbf { 0 } _ { 6 } } \end{array} \right] . } \end{array}\tag{9}
$$

The number of rows in $\mathbf { H } _ { t }$ is 6�, where � is the number of active cameras at frame �.

To adapt the filter to observed pose variability, the pose block of the process noise covariance matrix $\mathbf { Q } _ { t }$ and the measurement noise covariance matrix $\mathbf { R } _ { t }$ are updated at each video frame. For each camera, pose variability is estimated from valid measurements within a rolling window ofup to 30 frames. Missing poses are omitted, and the available history is used during startup. Let $\mathcal { W } _ { t , i }$ contain the frame indices of the valid poses in this window for active camera �, and let $m _ { t , i } = | \mathcal { W } _ { t , i } |$ . The measurement noise covariance $\mathbf { R } _ { t , i }$ is diagonal, with a separate variance for each of the six pose components:

$$
\begin{array} { r l } & { \mathbf { R } _ { t , i } = \mathrm { d i a g } \bigl ( s _ { t , i , 1 } ^ { 2 } , \ldots , s _ { t , i , 6 } ^ { 2 } \bigr ) , } \\ & { s _ { t , i , j } ^ { 2 } = \frac { 1 } { m _ { t , i } } \displaystyle \sum _ { \tau \in \mathcal { W } _ { t , i } } \big ( y _ { \tau , i , j } - \bar { y } _ { t , i , j } \big ) ^ { 2 } . } \end{array}\tag{10}
$$

Here $y _ { \tau , i , j }$ is component � of $\mathbf { y } _ { \tau , i }$ and $\bar { y } _ { t , i , j }$ is its mean over $\mathcal { W } _ { t , i }$ . These variances reflect both measurement fluctuations and actual tool motion and are used as empirical covariance estimates.

The overall measurement noise covariance matrix $\mathbf { R } _ { t }$ is then constructed by placing the individual $\mathbf { R } _ { t , i }$ matrices along the diagonal such that

$$
\mathbf { R } _ { t } = \mathrm { d i a g } \big ( \mathbf { R } _ { t , 1 } , \ \mathbf { R } _ { t , 2 } , \ \ldots , \ \mathbf { R } _ { t , n } \big ) ,\tag{11}
$$

where � is the number of active cameras providing measurements at frame �.

The filter uses camera-derived poses without auxiliary inertial measurements. For this camera-only implementation, we adopt an empirical covariance rule: the adaptive pose-noise block is set to the mean of the active cameras pose-variability matrices, while the velocity-noise block remains fixed. Averaging sets prediction uncertainty on the scale of observed pose variability without increasing that scale solely with camera count:

$$
\bar { \mathbf { R } } _ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { R } _ { t , i } , \qquad \mathbf { Q } _ { t } = \left[ \bar { \mathbf { R } } _ { t } \qquad \mathbf { 0 } _ { 6 } \qquad \right] .\tag{12}
$$

The velocity estimates and state covariance are initialized to zero. A fixed nonzero velocity process-noise term is included to allow pose measurements to update the velocity estimates. Each of the six velocity process-noise variances is set to 0.001 per prediction step in both the adaptive and nonadaptive filters. The translational and angular velocity variances are expressed in $( \mathrm { m m } / \mathrm { s } ) ^ { 2 }$ and $( \mathrm { d e g } / \mathrm { s } ) ^ { 2 }$ , respectively.

The standard Kalman filter equations (Kalman, 1960) are then applied to estimate the state $\mathbf { x } _ { t }$ at each frame. The prediction step is defined as

$$
\begin{array} { r l } & { \check { \mathbf { x } } _ { t } = \mathbf { A } \hat { \mathbf { x } } _ { t - 1 } , } \\ & { \check { \mathbf { P } } _ { t } = \mathbf { A } \hat { \mathbf { P } } _ { t - 1 } \mathbf { A } ^ { \top } + \mathbf { Q } _ { t - 1 } , } \end{array}\tag{13}
$$

(14)

where $\check { \mathbf { X } } _ { t }$ is the predicted state estimate, and $\check { \mathbf { P } } _ { t }$ is the predicted estimate covariance.

The update step is

$$
\mathbf { K } _ { t } = \check { \mathbf { P } } _ { t } \mathbf { H } _ { t } ^ { \top } \big ( \mathbf { H } _ { t } \check { \mathbf { P } } _ { t } \mathbf { H } _ { t } ^ { \top } + \mathbf { R } _ { t } \big ) ^ { - 1 } ,
$$

$$
\hat { \mathbf { x } } _ { t } = \check { \mathbf { x } } _ { t } + \mathbf { K } _ { t } \big ( \mathbf { y } _ { t } - \mathbf { H } _ { t } \check { \mathbf { x } } _ { t } \big ) ,\tag{15}
$$

(16)

$$
\begin{array} { r } { \hat { { \bf P } } _ { t } = \big ( { \bf I } - { \bf K } _ { t } { \bf H } _ { t } \big ) \check { { \bf P } } _ { t } , } \end{array}\tag{17}
$$

where $\mathbf { K } _ { t }$ is the Kalman gain, $\hat { \mathbf { x } } _ { t }$ is the updated state estimate, and $\hat { { \bf P } } _ { t }$ is the updated estimate covariance.

By updating the pose block of $\mathbf { Q } _ { t }$ and $\mathbf { R } _ { t }$ at each frame, the Kalman filter adapts its covariance weighting to the observed pose variability. We compare this adaptive approach with a constant-velocity Kalman filter using static covariance matrices. Noise profiling of stationary recordings provides fixed measurement noise covariance matrices $\bar { \bf R } _ { i }$ for each camera and the pose block of � for the non-adaptive filter, retaining the same fixed velocity-noise block. Details of this procedure are provided in Appendix A.

## 3.4.2. Multi-View Triangulation

Multi-view triangulation leverages the rigid positioning between cameras in the system to estimate the pose of the surgical pointer relative to the static reference. Marker corners on the surgical pointer and the static reference are first detected using OpenCV’s ArUco tools (Bradski, 2000) and undistorted using the intrinsic parameters described in Section 3.3.1. For each corner � that is observed by at least two cameras (index-set ${ \cal T } _ { k } \subseteq \{ 1 , \dots , N \} )$ , the 3D point $\tilde { \mathbf { X } } _ { k } ^ { \mathbf { d } }$ in the global (camera 1) coordinate frame is estimated by multi-view linear triangulation from the undistorted image coordinates $\mathbf { x } _ { k , i } ^ { \mathbf { d } } , i \in \mathcal { I } _ { k }$ , and the camera projection matrices, using singular value decomposition. Each corner is assigned a registration weight $\begin{array} { r } { w _ { k } \ = \ \frac 1 2 \left| { \cal I } _ { k } \right| \left( \left| { \cal I } _ { k } \right| - 1 \right) } \end{array}$ , equal to the number of camera pairs that observe it.

The triangulated corners are reprojected into the undistorted image planes of the cameras observing each marker. If the Euclidean reprojection error of any corner exceeds 8 pixels in any of these cameras, all four corners of that marker are discarded for that frame. Weighted point cloud registration of the remaining corners is then performed to minimise

$$
\arg \operatorname* { m i n } _ { \mathbf { C } \in S O ( 3 ) , \mathbf { r } \in \mathbb { R } ^ { 3 } } \sum _ { k } w _ { k } \left\| \mathbf { C } \mathbf { X } _ { k } ^ { \mathbf { d } } + \mathbf { r } - \tilde { \mathbf { X } } _ { k } ^ { \mathbf { d } } \right\| ^ { 2 } .
$$

to register the refined instrument model (Section 3.3.2) to the triangulated cloud, yielding the pointer pose $( \mathbf { C } _ { t } ^ { p c _ { 1 } } , \mathbf { r } _ { t } ^ { p c _ { 1 } } )$ and reference pose $( \mathbf { C } _ { t } ^ { r c _ { 1 } } , \mathbf { r } _ { t } ^ { r c _ { 1 } } )$ with respect to camera 1. The desired relative transform $\mathbf { \dot { T } } _ { t } ^ { p r }$ is obtained by composition, exactly as in Eq. (2) of Section 3.4.1. The complete procedure is described in Algorithm 3.

In the multi-view triangulation tracking method, the adaptive Kalman filter is applied to smooth the estimated pose trajectory. Unlike multi-view monocular pose fusion, where multiple pose estimates are merged and each camera’s measurement can be assigned an individual covariance, the filter here operates on a single triangulated pose estimate per frame. Consequently, the measurement noise covariance $\mathbf { R } _ { t }$ is constructed from the recent history of this singular pose estimate and equals the $6 \times 6$ pose block of the process noise covariance $\mathbf { Q } _ { t }$ . The velocity-noise block remains fixed at $0 . 0 0 1 \mathbf { I } _ { 6 } .$ Using the constant velocity dynamics described in Section 3.4.1, the prediction step (13) propagates the state forward in time, while the correction step (15) refines the predicted pose with the measurement. This approach suppresses high-frequency noise and ensures that the tracking profile remains consistent with the expected motion dynamics, but it cannot selectively suppress cameras with poor views.

Algorithm 3 Multi-View Triangulation (one frame)   
Input: Synchronized images from $i \in \{ 1 , \ldots , N \}$ cam  
eras; camera intrinsics $\mathbf { M } _ { i }$ and extrinsics $\mathbf { E } _ { i }$ (Alg. 1);   
dodecahedron marker corners $\mathbf { X } _ { k } ^ { \mathbf { d } }$ (for pointer & reference;   
Alg. 2).   
Output: Relative pose $\mathbf { T } ^ { p r }$ of the pointer w.r.t. the refer  
ence.   
1. Corner detection   
Detect 2D image points $\mathbf { x } _ { k , i } ^ { \mathbf { d } }$ in every camera $i ;$ keep points   
seen by at least two cameras and undistort them   
2. Multi-view triangulation   
for all corner � do   
$\boldsymbol { \mathbf { \mathit { 1 } } } _ { k } \gets \{ \ : i \mid \boldsymbol { \mathbf { \mathit { x } } } _ { k , i } ^ { \mathbf { d } }$ exists }   
$\tilde { \mathbf { X } } _ { k } ^ { \mathbf { d } } \gets \mathrm { t r i a n g u l a t e } \big ( \{ \mathbf { M } _ { i } , \mathbf { E } _ { i } , \mathbf { x } _ { k , i } ^ { \mathbf { d } } \} _ { i \in \mathcal { I } _ { k } } \big )$   
end for   
3. Re-projection outlier rejection   
for all marker with four triangulated corners $\kappa$ do   
for all $k \in \mathcal { K } , i \in \mathcal { I } _ { k }$ do   
$\Delta \mathbf { x } _ { k , i } ^ { \mathbf { d } } \gets \mathbf { x } _ { k , i } ^ { \mathbf { d } } - \pi \big ( \tilde { \mathbf { X } } _ { k } ^ { \mathbf { d } } ; \tilde { \mathbf { M } } _ { i } , \mathbf { E } _ { i } \big )$   
end for   
if any $\| \Delta \mathbf { x } _ { k , i } ^ { \mathbf { d } } \| _ { 2 } > 8$ pixels then   
Discard all four corners of this marker   
end if   
end for   
4. Pose estimation   
for all corner � not discarded do   
$\begin{array} { r } { w _ { k }  \frac { 1 } { 2 } \vert I _ { k } \vert \big ( \vert I _ { k } \vert - 1 \big ) } \end{array}$   
end for   
$\begin{array} { r } { ( \mathbf { C } ^ { p c _ { 1 } } , \mathbf { r } ^ { p c _ { 1 } } ) \gets \arg \operatorname* { m i n } _ { \{ \mathbf { C } , \mathbf { r } \} } \sum _ { k \in \mathcal { P } } w _ { k } \| \mathbf { C } \mathbf { X } _ { k } ^ { \mathbf { d } } + \mathbf { r } - \tilde { \mathbf { X } } _ { k } ^ { \mathbf { d } } \| ^ { 2 } } \end{array}$   
$\begin{array} { r } { ( \mathbf { C } _ { } ^ { r c _ { 1 } } , \mathbf { r } _ { } ^ { r c _ { 1 } } ) \gets \arg \operatorname* { m i n } _ { \{ \mathbf { C } , \mathbf { r } \} } \sum _ { k \in \mathcal { R } } w _ { k } \| \mathbf { C } \mathbf { X } _ { k } ^ { \dot { \mathbf { d } } } + \mathbf { r } - \tilde { \mathbf { X } } _ { k } ^ { \dot { \mathbf { d } } } \| ^ { 2 } } \end{array}$   
5. Relative pose composition   
$\mathbf { T } ^ { p r }  ( \mathbf { T } ^ { r c _ { 1 } } ) ^ { - 1 } \mathbf { T } ^ { p c _ { 1 } }$   
where $\mathbf { \bar { T } } ^ { p c _ { 1 } } = \left[ \begin{array} { l l } { \mathbf { C } ^ { p c _ { 1 } } } & { \mathbf { r } ^ { p c _ { 1 } } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right]$ and $\mathbf { T } ^ { r c _ { 1 } } = \left[ \begin{array} { l l } { \mathbf { C } ^ { r c _ { 1 } } } & { \mathbf { r } ^ { r c _ { 1 } } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right]$   
return $\{ { \bf T } ^ { p r } \}$

## 4. Experimental Protocol

## 4.1. Experimental Setups

The performance of the proposed system is evaluated through two experimental setups: a grid experiment designed to establish baseline accuracy and a head phantom experiment to better reflect the anatomical and operational conditions observed during IGNS procedures. A stand-in for a typical surgical light was manufactured to support the fixture and a light source. The stand-in measures 72 cm in diameter, derived from the specification sheet of our reference surgical light model (Infinium Medical, 2026). In both experiments, the ring fixture is mounted on the surgical light stand-in positioned 1 meter above the centre of the tracking volume. For benchmarking, a gold standard Atracsys fusionTrack 500 optical tracking system is placed adjacent to the phantom with a sensor-to-object distance of

![](images/d5007179d3d6e9341f11ac37b454c4ae78cb1de3a496cc1dc3dc94ebb2e85cfe.jpg)  
Figure 5: Experimental validation setups. Left column shows the experiment with the precision-machined grid. Right column shows the experiment with the SLA-printed head phantom. Both setups include the multi-camera fixture mounted on the surgical light stand-in, the surgical pointer, and the static reference. The fusionTrack 500 optical tracker (visible on the side) provides simultaneous measurements.

1.4 meters per manufacturer specifications. Retroreflective sphere patterns are attached to the surgical pointer and next to the tracking volume, enabling simultaneous tracking by both systems. Figure 5 shows photos of both experimental setups with this consistent configuration.

## 4.1.1. Grid Experiment

The baseline accuracy of the proposed system is evaluated on a machined 400 mm × 400 mm grid (Gerard and Collins, 2015) using landmark registration (see Appendix B). The grid features 25 uniformly spaced points, 9 of which are designated as fiducials. Each trial involves sampling the 9 fiducials to compute the fiducial registration error (FRE), and using the resulting transformation to cal culate the TRE of the remaining 16 points. When collecting data from either a fiducial or target point, the tracked pointer is placed in a pointer holder that is inserted into a 2 mm diameter hole drilled at 100 mm center-to-center distance grid points using a numerically controlled milling machine, with random z-axis orientation at each point. This minimises noise due to movement and should yield the best accuracy estimate. To evaluate the TRE for dynamic trajectories, a block bearing an S-shaped curve with dimensions 100 mm x 50 mm x 15 mm is machined and placed at four positions along the surface of the grid as shown in Figure 6a. Trajectory data was collected by holding the tracked pointer and sliding it along the S-shaped groove. This was designed to more realistically estimate tracking accuracy. Both multiview monocular pose fusion and multi-view triangulation tracking modes are evaluated across 20 trials. The complete dataset comprises 5,000 synchronized multi-view videos across both experimental setups, all publicly available at https://nist.mni.mcgill.ca/data/ for benchmarking alternative tracking algorithms.

## 4.1.2. Head Phantom Experiment

To validate the system in conditions more representative of neurosurgical workflows, a complementary evaluation is performed using an SLA-printed head phantom (see Fig. 6b). The phantom, derived from a 3T MRI scan, features 21 landmarks: 9 fiducial points corresponding to anatomical landmarks commonly used for fiducial registration in IGNS (Taleb et al., 2023), 10 target points distributed across the cranium, and two square grooves placed along the frontal and parietal bones for trajectory tracking. Each trial consists of sampling the 9 fiducials to calculate the FRE. The resulting transformation is used to compute the point TRE for the 10 target points and the trajectory TRE along the grooves. Results are aggregated over 20 trials, and detailed performance metrics for multi-view monocular pose fusion and multi-view triangulation are provided in Table 1. The model for the head phantom can be found at https://nist. mni.mcgill.ca/software/.

## 4.2. Ablation Studies

## 4.2.1. Camera Count

To evaluate robustness to camera-level occlusion, we perform an ablation study over camera subsets. For the grid experiment, we evaluate all possible camera combinations for comprehensive performance analysis. This includes all $\begin{array} { r l r } { \sum _ { r = 1 } ^ { N _ { \mathrm { c a m s } } } { \stackrel { \cdot } { \left( \right)} ^ { N _ { \mathrm { c a m s } } } _ { r }  } } & { { } = } & { 2 ^ { N _ { \mathrm { c a m s } } } - 1 = 3 1 } \end{array}$ combinations for $N _ { \mathrm { c a m s } } \ = \ 5 ,$ , enabling an exhaustive assessment of how camera count and configuration afect tracking accuracy. For the head phantom experiment, we simulate realistic clinical occlusion patterns by defining three specific scenarios (top, left, and right) relative to the operating table, where clinicians or instruments would typically obstruct the field of view. We categorize occlusion severity into five levels, ranging from severe, with only one active camera, to no occlusion, with all cameras active. Figure 7 illustrates these occlusion patterns. Results are aggregated across all trials for the grid experiment across all 31 combinations, and for the head phantom across each occlusion scenario and severity level. Detailed per-scenario data for the head phantom are provided in Table C.1. Table 1 presents overall performance for the grid and head phantom experiments respectively.

## 4.2.2. Filtering Method

To assess the benefits of adaptive noise modeling, an ablation study is performed comparing the system’s performance under three configurations: 1) the main implementation using adaptive Kalman filtering for both multiview monocular pose fusion and multi-view triangulation tracking modes, 2) an implementation with a non-adaptive Kalman filter (see Appendix A), and 3) an implementation without any filter (where poses in monocular pose fusion are averaged across cameras).

![](images/52afd958d5470ddd2deb7a9bace8f380338593a8e7bb27eabd7db75af0e5d5ab.jpg)

(a) Machined grid used for landmark registration. It carries a set of 9 fiducial points (red), 16 target points (green), and 4 trajectories (orange). Each trial samples the fiducials to compute FRE and uses the transformation to calculate TRE on the remaining points. Trajectory data is collected along S-shaped grooves as illustrated.  
![](images/8001c4afd144982071d9f70110edb2936ee3edf680328378272e7bc1d38120e7.jpg)  
(b) SLA-printed head phantom used for landmark registration. It carries 9 fiducial points (left), 10 target points (middle), and 2 trajectories (right). Each trial uses sampled fiducials to estimate a registration transform and compute FRE. The transform is then applied to the target points and trajectories to calculate their TRE.  
Figure 6: Experimental setups for system evaluation. (a) Machined grid for baseline accuracy measurements. (b) SLA-printed head phantom for neurosurgery-representative validation.

![](images/02a6b019b56d35e7bc6ecea85fb49c00cce2e80af3097541cf0eab309740cfeb.jpg)  
Figure 7: Head phantom experiment occlusion scenarios and severity scale.

## 5. Results

Results for all experiments are summarized in Table 1. For the grid experiment, each RMSE value represents the aggregation across all the possible camera combinations for that number of cameras. For the head phantom experiment, each RMSE value represents the aggregation across the three occlusion scenarios. One trial with a specific camera configuration results in a single RMSE value. The tables report the median and standard deviation over all RMSE experiments.

Table 1 summarizes trial- and configuration-level RMSE values. In Figures 8–10, the point TRE distributions contain individual target-point errors, whereas the trajectory TRE distributions contain one RMSE value per measured trajectory for each trial and camera configuration. The plotted medians therefore summarize diferent units of observation for point and trajectory tracking. For the respective FRE plots, see Figure C.1.

Statistical comparisons use the 20 acquisition trials per phantom as independent units. Within each trial and tracking setting, squared point errors or squared per-trajectory RMSEs are averaged within each camera configuration; configuration mean squares are then averaged equally and square-rooted. The five-camera head measurements are included once across occlusion scenarios. For pooled camera-count comparisons, mean squares are additionally averaged equally across available tracking methods and filters within each trial (pose fusion alone with one camera; both methods with two to five cameras). We use two-sided paired Wilcoxon signed-rank tests with exact sign-assignment probabilities and Holm correction (� = 0.05): three filter comparisons per experiment, camera count, tracking method, and metric, or ten camera-count comparisons per experiment and metric. Paired sign tests assess sensitivity to the signed-rank symmetry assumption;

Tables C.2–C.4 mark comparisons whose significance decisions difer between the tests.

Pooled trial-summary errors generally decrease with more cameras, although improvements are not monotonic. Compared with the basic method, adaptive Kalman filtering significantly reduces pose-fusion trajectory error at all camera counts on the grid and at three to five cameras on the head phantom. It also significantly reduces triangulation trajectory error at all grid camera counts. Both paired tests support these filtering comparisons.

To illustrate the mechanisms underlying these performance diferences, Figures 12 and 13 examine the worstcase trial from the head phantom experiment, where camera 3 observes only a single marker at an oblique angle and tracking errors accumulate towards the trajectory’s end. Section 6 analyses these contrasting behaviors in detail.

## 6. Discussion

The grid and head phantom experiments reveal distinct performance diferences between multi-view monocular pose fusion and triangulation that depend critically on viewing conditions and scene geometry.

On the machined grid, where all cameras maintain reasonable multi-marker views, both tracking methods achieve sub-millimeter accuracy with five cameras using adaptive Kalman filtering. Multi-view monocular pose fusion achieves 0.74 mm median point TRE while triangulation attains 0.64 mm. Figure 9 directly compares both methods against the gold-standard fusionTrack 500 at the five-camera configuration with adaptive Kalman filtering, showing comparable point localization accuracy.

The methods difer markedly in their response to camera count. Monocular pose fusion generally improves as cameras are added, with particularly strong performance in trajectory tracking (0.73 mm TRE with five cameras and adaptive Kalman filtering; Figure 8b). In this method, percamera covariance weighting adjusts each pose estimate’s contribution to the fused result. Triangulation depends on inter-camera extrinsic calibration, whereas monocular pose fusion estimates the pointer-to-reference pose separately in each camera. Both methods combine observations across cameras and can be afected by inter-camera timing errors during motion. On the grid, monocular pose fusion achieves lower median trajectory TRE than triangulation with five cameras and adaptive filtering.

Triangulation demonstrates unexpected non-monotonic behavior, with accuracy degrading at intermediate camera counts (1.11 mm trajectory TRE at three cameras) before recovering with five cameras (Figure 8d). This sensitivity likely arises from geometric conditioning issues when cameras are added at certain angles relative to the planar grid, afecting ray intersection quality. However, triangulation maintains superior static point localization (0.64 mm versus 0.74 mm for monocular fusion; Figure 8a,c), likely due to its explicit use of multi-view geometric constraints. The direct comparison in Figure 9 shows median trajectory TRE values of 0.73 mm for monocular pose fusion, 0.85 mm for triangulation, and 0.69 mm for the fusionTrack 500.

Table 1  
Registration RMSE in mm (median ± standard deviation) across 20 trials for Grid and Head Phantom experiments. Total trajectory RMSE pools valid samples across the measured curves for both systems.
<table><tr><td rowspan="2"></td><td rowspan="2"># of Cameras</td><td colspan="3">Monocular Pose Fusion</td><td colspan="3">Triangulation</td><td rowspan="2">fusionTrack 500</td></tr><tr><td>Basic</td><td>Kalman</td><td>Kalman Adaptive</td><td>Basic</td><td>Kalman</td><td>Kalman Adaptive</td></tr><tr><td>Grid Experiment</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fiducial Point</td><td></td><td>1.2224 (±0.2577) 0.9697 (±0.1757)</td><td>1.1693 (±0.2532) 0.9704 (±0.1734)</td><td>1.1732 (±0.2531) 1.0193 (±0.1728)</td><td>N/A 0.9942 (±0.3259)</td><td>N/A 0.9871 (±0.3241)</td><td>N/A 0.9648 (±0.3158)</td><td></td></tr><tr><td>Median Registration RMSE</td><td>1234</td><td>0.8419 (±0.1454)</td><td>0.8576 (±0.1445)</td><td>0.9174 (±0.1448)</td><td>1.0433 (±0.4894)</td><td>1.0427 (±0.4755)</td><td>1.0250 (±0.4076)</td><td>0.5410 (±0.0420)</td></tr><tr><td></td><td></td><td>0.7683 (±0.1275)</td><td>0.7820 (±0.1272)</td><td>0.8517 (±0.1307)</td><td>0.9461 (±0.2039)</td><td>0.9487 (±0.2021)</td><td>0.9249 (±0.1890)</td><td></td></tr><tr><td></td><td>5</td><td>0.7189 (±0.1148)</td><td>0.7228 (±0.1145)</td><td>0.7995 (±0.1270)</td><td>0.8307 (±0.1153)</td><td>0.8271 (±0.1150)</td><td>0.8341 (±0.1102)</td><td></td></tr><tr><td></td><td></td><td>1.4864 (±0.3216)</td><td>1.4287 (±0.3273)</td><td>1.4243 (±0.3317)</td><td>N/A</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Target Point</td><td></td><td>1.1121 (±0.1790)</td><td>1.1110 (±0.1753)</td><td>1.1504 (±0.1640)</td><td>1.0171 (±0.3414)</td><td>1.0146 (±0.3379)</td><td>0.9584 (±0.3147)</td><td></td></tr><tr><td>Median Registration RMSE</td><td></td><td>0.9568 (±0.1323)</td><td>0.9697 (±0.1314)</td><td>1.0104 (±0.1195)</td><td>1.0429 (±0.6050)</td><td>1.0396 (±0.5870)</td><td>1.0000 (±0.5187)</td><td>0.6298 (±0.0551)</td></tr><tr><td></td><td>1234</td><td>0.8690 (±0.1102)</td><td>0.8843 (±0.1121)</td><td>0.8936 (±0.0977)</td><td>0.9459 (±0.1786)</td><td>0.9367 (±0.1751)</td><td>0.9030 (±0.1597)</td><td></td></tr><tr><td></td><td>5</td><td>0.7855 (±0.0979)</td><td>0.8287 (±0.1010)</td><td>0.8290 (±0.0863)</td><td>0.7622 (±0.0540)</td><td>0.7588 (±0.0525)</td><td>0.7477 (±0.0513)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Target Trajectory</td><td></td><td>1.3576 (±0.3371)</td><td>1.1840 (±0.3143)</td><td>1.1630 (±0.3080)</td><td>N/A</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Median Registration RMSE</td><td></td><td>1.0589 (±0.2288)</td><td>0.9819 (±0.2070)</td><td>0.9595 (±0.1665)</td><td>1.1602 (±0.3665)</td><td>1.1403 (±0.3597)</td><td>1.0013 (±0.3292)</td><td>0.6688 (±0.0539)</td></tr><tr><td></td><td></td><td>0.9237 (±0.1903)</td><td>0.8788 (±0.1708)</td><td>0.8590 (±0.1334)</td><td>1.6562 (±1.0454)</td><td>1.6334 (±1.0325)</td><td>1.4215 (±0.9682)</td><td></td></tr><tr><td></td><td>12345</td><td>0.8659 (±0.1707)</td><td>0.8109 (±0.1521)</td><td>0.7895 (±0.1161)</td><td>1.3336 (±0.5978)</td><td>1.3176 (±0.5955)</td><td>1.1953 (±0.5717)</td><td></td></tr><tr><td>Head Phantom Experiment</td><td></td><td>0.8100 (±0.1569)</td><td>0.7705 (±0.1400)</td><td>0.7675 (±0.1035)</td><td>1.1109 (±0.2975)</td><td>1.1038 (±0.2953)</td><td>1.0314 (±0.2777)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>1.5134 (±0.3488)</td><td>1.4446 (±0.3284)</td><td>1.4349 (±0.3264)</td><td>N/A</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Fiducial Point</td><td>1234</td><td>1.2624 (±0.2279)</td><td>1.2429 (±0.2198)</td><td>1.2272 (±0.1796)</td><td>1.2363 (±0.2821)</td><td>1.2355 (±0.2857)</td><td>1.1997 (±0.2953)</td><td></td></tr><tr><td>Median Registration RMSE</td><td></td><td>1.0766 (±0.1450)</td><td>1.0802 (±0.1532)</td><td>0.9985 (±0.1337)</td><td>1.2541 (±0.2772)</td><td>1.2551 (±0.2742)</td><td>1.2458 (±0.2671)</td><td>0.8462 (±0.0532)</td></tr><tr><td></td><td></td><td>1.0276 (±0.1218)</td><td>1.0375 (±0.1320)</td><td>0.9348 (±0.1080)</td><td>1.1028 (±0.1022)</td><td>1.0978 (±0.0997)</td><td>1.0787 (±0.0933)</td><td></td></tr><tr><td></td><td>5</td><td>0.9952 (±0.1042)</td><td>1.0332 (±0.1047)</td><td>0.8973 (±0.0977)</td><td>1.0695 (±0.0735)</td><td>1.0686 (±0.0715)</td><td>1.0687 (±0.0649)</td><td></td></tr><tr><td></td><td></td><td>2.8598 (±1.2893)</td><td>2.7622 (±1.2042)</td><td>2.7304 (±1.2023)</td><td>N/A</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Target Point</td><td></td><td>2.0171 (±0.8254)</td><td>2.0813 (±0.8253)</td><td>2.0716 (±0.5813)</td><td>1.7860 (±0.5421)</td><td>1.7761 (±0.5373)</td><td>1.7163 (±0.5355)</td><td></td></tr><tr><td>Median Registration RMSE</td><td></td><td>1.8345 (±0.6181)</td><td>1.9141 (±0.6325)</td><td>1.6791 (±0.4441)</td><td>1.6027 (±0.5950)</td><td>1.6059 (±0.5882)</td><td>1.6082 (±0.5634)</td><td>1.0879 (±0.1140)</td></tr><tr><td></td><td>1234</td><td>1.7375 (±0.5295)</td><td>1.8053 (±0.5511)</td><td>1.5535 (±0.3524)</td><td>1.4257 (±0.2897)</td><td>1.4312 (±0.2868)</td><td>1.3957 (±0.2834)</td><td></td></tr><tr><td></td><td>5</td><td>1.5759 (±0.4657)</td><td>1.6535 (±0.5066)</td><td>1.4813 (±0.3523)</td><td>1.2997 (±0.1885)</td><td>1.2934 (±0.1897)</td><td>1.2774 (±0.1938)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>N/A</td><td></td><td>N/A</td><td></td></tr><tr><td>Target Trajectory</td><td>1</td><td>2.4384 (±0.8503)</td><td>2.4414 (±0.8697)</td><td>2.4446 (±0.8690)</td><td></td><td>N/A</td><td></td><td></td></tr><tr><td></td><td></td><td>1.5667 (±0.6589)</td><td>1.6503 (±0.6855)</td><td>1.5826 (±0.6055)</td><td>1.4446 (±0.4989)</td><td>1.4290 (±0.5039)</td><td>1.3458 (±0.5034)</td><td></td></tr><tr><td>Median Registration RMSE</td><td></td><td>2.0989 (±2.3832)</td><td>2.2846 (±2.0119)</td><td>1.5627 (±0.4856)</td><td>1.5059 (±0.6478)</td><td>1.5000 (±0.6472)</td><td>1.4609 (±0.6320)</td><td>0.7475 (±0.0948)</td></tr><tr><td></td><td></td><td>1.9652 (±1.5974)</td><td>2.1550 (±1.4467)</td><td>1.5075 (±0.3575)</td><td>1.2563 (±0.4529)</td><td>1.2563 (±0.4484)</td><td>1.2193 (±0.4420)</td><td></td></tr><tr><td></td><td>2345</td><td>2.8220 (±1.2112)</td><td>2.6056 (±1.1142)</td><td>1.4894 (±0.3249)</td><td>1.1245 (±0.1549)</td><td>1.1151 (±0.1551)</td><td>1.1197 (±0.1641)</td><td></td></tr></table>

![](images/184771dc74ea0863475e238e6c5f06ed7caf2db9dd9bb24bf7ad7b2296e298cf.jpg)  
(a) Point TRE – Monocular Pose Fusion

![](images/ae96ca1d197750504d215008b6c2bb82156fd56d591f4b7ebb7630edae335f07.jpg)  
(b) Trajectory TRE – Monocular Pose Fusion

![](images/0619143dd53fb007758b1826ce4bc35254bb6b457a94570143d416e86aa4d319.jpg)  
(c) Point TRE – Triangulation

![](images/d3a0430e12d953f2802fbd02f3f006a4b4ed03e5395aa4964ae345aedecb1cb0.jpg)  
(d) Trajectory TRE – Triangulation  
Figure 8: TRE analysis on the machined grid showing filtering method comparison across camera counts. All rows show individua point TRE values (left column) and per-trajectory RMSE values (right column). Top row: Monocular pose fusion. Bottom row: Triangulation. Boxes = IQR, red dots = median, violins = kernel density. Statistical comparisons use 20 paired trial summaries.

![](images/0944634392113ebc0eeace991e6905ad47f0700eda57ba0db7f7da7998a53f57.jpg)  
Figure 9: Five-camera performance comparison with adaptive Kalman filtering on the machined grid. Both LARK pose estimation methods (monocular pose fusion and triangulation) are compared against the gold-standard fusionTrack 500 for point localization (left) and trajectory tracking (right). Boxes = IQR, red dots = median, violins = kernel density.

The clearest benefit of adaptive Kalman filtering is observed in monocular trajectory tracking, where it reduces the spread of errors (Figure 8a–d). Its efect on point localization and triangulation varies with camera count and filtering method.

The head phantom reveals fundamentally diferent performance characteristics. The phantom’s complex threedimensional geometry creates more challenging conditions than the planar grid: registration requires steeper viewing angles that increase marker-level occlusions, the larger spatial extent along the reference frame’s z-axis challenges depth estimation, and experimental limitations include potential warping of the SLA-printed material and wear from repeated use. With five cameras and adaptive Kalman filtering, neither method achieves sub-millimeter median point or trajectory TRE on the head phantom. The fixed camera positions, while ensuring fair comparison, create a more demanding scenario than typical clinical use where cameras can be repositioned to optimize viewing angles for specific operative regions.

The critical diference emerges in how each tracking method handles severely degraded views. The occlusion scenario analysis (see Table C.1) reveals that camera 3 consistently produced the largest outliers when it could observe only a single marker. Figure 12 illustrates this problematic condition where the pointer partially occludes the reference, leaving camera 3 with a compromised view worse than complete occlusion. The oblique viewing angle in this case compounds the tracking error.

In monocular pose fusion, camera 3’s erroneous pose estimate remains a significant component of the fused result. The adaptive Kalman filter adjusts weights according to measurement covariance, but a single-marker pose does not always yield suficiently high variance to trigger strong down-weighting if the detection itself appears stable. As a result, some outliers persist (Figure 10a,b). Even in such cases, however, the filter substantially mitigates errors compared to the basic approach. For example, in the threecamera left occlusion scenario, the adaptive Kalman filter reduces the median trajectory RMSE from 4.45±2.36 mm to 1.47±0.42 mm (see Table C.1).

Triangulation proves remarkably robust to the same poor view (Figure 10c,d). As shown in Figure 13, triangulation maintains stable tracking without large outliers across all filtering implementations. This behavior is consistent with the diferent ways the methods use sparse observations: a single marker supplies four coplanar corners, which permit PnP pose estimation but can yield ambiguous or noisesensitive estimates. Triangulation instead combines corner observations across cameras before estimating the instrument pose. Additionally, camera 3’s limited observations afect only a few corner measurements among potentially dozens from all cameras combined, diluting its impact on the final pose computation. With five cameras and adaptive Kalman filtering, triangulation outperforms monocular pose fusion on the anatomical phantom for both point localization (1.13 mm versus 1.35 mm) and trajectory tracking (1.06 mm versus 1.53 mm; Figure 11).

Adaptive Kalman filtering reduces the spread of trajectory errors in monocular pose fusion, although stable erroneous single-marker estimates can retain appreciable weight. The efect of filtering on triangulation depends on camera count and phantom.

These phantom results support further development of low-cost multi-camera tracking for image-guided navigation. The method comparison depends on viewing conditions: monocular pose fusion gives lower trajectory error on the grid, whereas triangulation gives lower point and trajectory errors on the anatomical head phantom with five cameras and adaptive filtering.

The results demonstrate the potential of commodity RGB cameras for low-cost tracking with multiple viewpoints. Three observations emerge from the comparison. First, monocular pose fusion is more afected by the illustrated poor single-marker view than triangulation. Second, adaptive Kalman filtering reduces the spread of trajectory errors in monocular pose fusion. Third, camerasubset performance depends on both the number and quality of the retained views.

Future work should explore hybrid strategies that incorporate view-quality metrics to trigger dynamic switching between tracking methods or selective camera exclusion, and combine poses derived from triangulation of stereo pairs with adaptive Kalman fusion, preventing inaccurate estimates arising from poor single-marker views while preserving the accurate trajectory tracking observed under favorable conditions.

## 7. Conclusion

This work demonstrates low-cost multi-camera optical tracking using commodity RGB hardware, evaluated on a machined grid and an anatomical head phantom. Our system achieves 0.64 mm point TRE with triangulation and 0.73 mm trajectory TRE with pose fusion on a machined grid, and

![](images/11bb682cd0df9081d4a36f5fedb230fcb7a0d9e2fa77321492526653aea7abb1.jpg)  
(a) Point TRE – Monocular Pose Fusion

![](images/c2705678c8710723313f1cfc145b45b1292779263a67999e26f71a812d2f7511.jpg)  
(c) Point TRE – Triangulation

![](images/1579bdd68ca5d35f463949bd9c2da0dbc3b144439572d067a2aa99ddbe9eac8c.jpg)  
(b) Trajectory TRE – Monocular Pose Fusion

![](images/e68d43dbf1a8461f17408fdd7e9503d382cdce543cf186edcee187c5767b9ecf.jpg)  
(d) Trajectory TRE – Triangulation

Figure 10: TRE analysis on the SLA-printed head phantom showing filtering method comparison across camera counts. All rows show individual point TRE values (left column) and per-trajectory RMSE values (right column). Top row: Monocular pose fusion. Bottom row: Triangulation. Boxes = IQR, red dots = median, violins = kernel density. Statistical comparisons use 20 paired trial summaries.  
![](images/65baa7369dc37d206f5872a5a78989ec2722469ad6a3ac075930ff5f9ca4099d.jpg)  
Figure 11: Five-camera performance comparison with adaptive Kalman filtering on the anatomical head phantom. Under challenging viewing conditions with partial marker occlusion, triangulation outperforms monocular pose fusion for both point localization (1.13 mm versus 1.35 mm) and trajectory tracking (1.06 mm versus 1.53 mm). Boxes = IQR, red dots = median, violins = kernel density.

1.13 mm point and 1.06 mm trajectory TRE with triangulation on an anatomical head phantom.

The two methods have complementary strengths: with five cameras and adaptive filtering, monocular pose fusion achieves lower trajectory TRE on the grid, whereas triangulation achieves lower point and trajectory TRE on the head phantom. The illustrated poor-view condition also afects triangulation less than monocular pose fusion.

The camera-subset experiments characterize how tracking accuracy changes as views become unavailable, while the poor-view example shows that the quality of retained views also matters. With tracking hardware costing under \$1,000 USD and full open-source release of designs, technical drawings, bill of materials, software, and datasets (designs and software: https://nist.mni.mcgill.ca/software/; datasets: https://nist.mni.mcgill.ca/data/), this work lowers barriers to adoption in resource-constrained settings and provides a reproducible benchmark for evaluating tracking

![](images/5bbb9aad80c1866f645d2a483f300d4cf1c4ed3be1d34b0e1f17ac4b60b7dece.jpg)  
Figure 12: Camera views during trajectory tracking at the moment of maximum error in the head phantom experiment. All five camera views are shown with detected markers highlighted in green boxes. Camera 3 (top right) demonstrates a poor view condition where the pointer partially occludes the reference, leaving only a single marker visible at an oblique angle to the image plane. The bottom right panel shows a magnified view of camera 3’s compromised perspective.

Ground truth (GT)

Measured trajectory

Basic

Kalman

Adaptive Kalman

Monocular pose fusion | 3 cameras

![](images/3a6a4f4944570711666fdb87e9b8acad87bb58f06f911c9cf88b6852723a2a54.jpg)

![](images/bdaee26353486b059153093baea9508842d332599593b46730f32e9aff401b82.jpg)

![](images/ab35f049a9417f4e2397418572644f9d9a821f4a0750fab78a04872e87e10b31.jpg)

Monocular pose fusion

![](images/9548a4bd8e99053688b1067840ab1cb45c4cef39b627d3aabe7ac7c827f7124d.jpg)  
5 cameras

![](images/f58892cfca1d16fd70731fe1e5508d8204c076661f4127285e09c0fcd2af18db.jpg)

![](images/557e20fe5814fb0c421ecba43ff8c0ab318b4cc8e025a23d69fef89103b54005.jpg)

Triangulation | 3 cameras

![](images/13a1b745696826523ef42b36d7f396125ed8d097612ce64fcad7a49ddfab2ef3.jpg)

![](images/0232c598cfcc8289118b3a5f9c3d0276b10e11d60f9034cd8e2e4231e36b947f.jpg)

![](images/9609316318c5454495054fb25bb6d0503cf22ce48c01c1c877e85c4d3cba7756.jpg)  
(j)

Triangulation | 5 cameras approaches in IGS research. Clinical validation remains the critical next step to confirm translation of these laboratory results to improved surgical outcomes.

![](images/7a99dbe10ba875741ae6d3362faae1a2de337831afae390c8e6f233cbd6d5a75.jpg)

![](images/6417e29d479519f55b48fb75ecd4e69f5ecf4589e55861c045009329eba67181.jpg)

![](images/bfed0f41010544310aceb8ad5a70faaf6e72f388ac0d457e5c04e8b0257cd019.jpg)  
Figure 13: Comparison of tracked trajectories versus ground truth for the left occlusion scenario in the head phantom experiment trial which produced the largest overall outlier. Top two rows show monocular pose fusion for 3-camera and 5-camera configurations respectively. Bottom two rows show triangulation for 3-camera and 5-camera configurations respectively. Columns from left to right show Basic, Kalman, and Adaptive Kalman filtering. Red lines represent ground truth, blue connected dots represent tracked trajectories. The cropped, shaded head provides qualitative spatial context; the shared XYZ triad indicates orientation only. Panel scales difer, so apparent trajectory sizes should not be compared across panels.

## Acknowledgements

We acknowledge funding from the Canadian Institutes of Health Research FRN-152980, FRN-158277 and PJT-109163.

## A. Noise Profiling

To compare the performance of our adaptive Kalman filter with a constant velocity Kalman filter using static covariance matrices, we first conduct a noise profiling procedure. The procedure allows us to estimate the static measurement noise covariance matrices $\mathbf { R } _ { i }$ for each camera and the pose block of the process noise covariance matrix � for the nonadaptive filter.

The noise profiling involves collecting a series of pose measurements while the surgical instrument is held stationary at � distinct workspace locations indexed by $\ell =$ $1 , \ldots , L$ . At every location we record $T = 3 0$ consecutive pose estimates from each active camera $i \in \{ 1 , \ldots , N \}$ . The sample variance of those poses gives the location–specific measurement covariance

$$
\mathbf { R } _ { \ell , i } = \mathrm { V a r } \Big ( \mathbf { y } _ { \tau , i } \Big | \tau = 1 , \ldots , T \Big ) ,\tag{A.1}
$$

where $\mathbf { y } _ { \tau , i }$ is the pose returned by camera � on frame � while the instrument is fixed at location �.

We then compute the average measurement noise covariance for each camera across all sampled positions:

$$
\bar { \mathbf { R } } _ { i } = \mathbf { \frac { 1 } { L } } \sum _ { \ell = 1 } ^ { L } \mathbf { R } _ { \ell , i } ,\tag{A.2}
$$

which serves as the static measurement noise covariance matrix for camera � in the non-adaptive Kalman filter.

To formulate the static process noise covariance matrix $\mathbf { Q } ,$ we average the $\bar { \bf R } _ { i }$ matrices across all cameras for its pose block and retain the same fixed velocity-noise block:

$$
\begin{array} { r } { \mathbf { Q } \ = \ [ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bar { \mathbf { R } } _ { i } \qquad \ \mathbf { 0 } _ { 6 } ] . } \\ { \ \qquad \ \mathbf { 0 } _ { 6 } \qquad \ 0 . 0 0 1 \mathbf { I } _ { 6 } \ d \mathbf { \Lambda } _ { 6 } ] . } \end{array}\tag{A.3}
$$

Here � is the total number of cameras. Stationary noise profiling determines the pose-noise block; the velocity variances are fixed tuning parameters, as in the adaptive filter described in Section 3.4.1.

Beyond tuning the static covariance matrices for the non-adaptive Kalman filter, this procedure provides valuable insights into the spatial distribution of measurement noise within the workspace. By analyzing the norms of the covariance matrices $\mathbf { R } _ { \ell , i }$ , we generate heatmaps illustrating the noise profile for each camera at diferent positions. The Frobenius norm of each $\mathbf { R } _ { \ell , i }$ matrix is computed as

![](images/7dd60275f06fc3867eef72ff546580014da113e889e6b5494d2c0937cf103fde.jpg)  
Figure A.1: Spatial distribution of the norm of the measurement noise covariance matrices $\mathbf { R } _ { \ell , i }$ for each camera across all measured positions.

$$
\left\| \mathbf { R } _ { \ell , i } \right\| _ { F } = \sqrt { \mathrm { T r } \big ( \mathbf { R } _ { \ell , i } \mathbf { R } _ { \ell , i } ^ { \top } \big ) } .\tag{A.4}
$$

This norm provides a single scalar value to represent the overall noise level for each camera at each position. Figure A.1 shows the spatial distribution of noise across all measured positions for one of our experimental trials.

This noise profiling procedure configures the static Kalman filter using measured stationary pose variability, providing a baseline for comparison with the adaptive Kalman filter. Moreover, the insights gained can guide adjustments to lighting, camera placement, and calibration to enhance measurement precision. In the OR, this procedure could help technicians assess tracking precision during camera setup.

## B. Landmark Registration

Landmark registration aligns a set of measured 3-D points with their corresponding ground-truth (GT) points to determine the rigid transformation between two coordinate systems. This method is central to IGS, where precise alignment ensures that surgical instruments and patient anatomy share a consistent spatial frame of reference.

We employ a point–based registration strategy using the Orthogonal Procrustes method (Fitzpatrick et al., 1998). The aim is to find a rotation matrix $\mathbf { S } \in S O ( 3 )$ and a translation vector � $\in \mathbb { R } ^ { 3 }$ that minimise the diference between corresponding points in the measured and GT sets. Suppose $N _ { f }$ fiducial points are available. The Fiducial Registration Error (FRE) is the root-mean-square distance between corresponding fiducials after alignment:

$$
\mathrm { F R E } = \sqrt { \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \Bigl \| \mathbf { x } _ { i } ^ { \mathrm { G T } } - \bigl ( \mathbf { S } \mathbf { x } _ { i } ^ { \mathrm { m e a s } } + \mathbf { d } \bigr ) \Bigr \| ^ { 2 } }\tag{B.1}
$$

To determine � and �, the measured and GT point sets are first translated so that their centroids lie at the origin. Denote those centroids by $\bar { \mathbf { x } } ^ { \mathrm { m e a s } }$ and $\bar { \mathbf { x } } ^ { \mathrm { G T } }$ . After centring, the covariance matrix is

$$
\textbf { G } = \ \sum _ { i = 1 } ^ { N _ { f } } ( \textbf x _ { i } ^ { \mathrm { m e a s } } - \bar { \textbf x } ^ { \mathrm { m e a s } } ) ( \textbf x _ { i } ^ { \mathrm { G T } } - \bar { \textbf x } ^ { \mathrm { G T } } ) ^ { \top } .\tag{B.2}
$$

Applying Singular Value Decomposition (SVD) to $\mathbf { G }$

$$
\mathbf { G } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { V } ^ { \intercal } ,
$$

the optimal rotation is

$$
\mathbf { S } = \mathbf { V } \mathbf { U } ^ { \top } .
$$

If det $( \mathbf { S } ) < 0$ the final column of � is negated to prevent a reflection. The translation that relates the two centroids is

$$
\mathbf { d \Sigma } = \bar { \mathbf { x } } ^ { \mathrm { G T } } - \mathbf { S } \bar { \mathbf { x } } ^ { \mathrm { m e a s } } .
$$

With � and � fixed by the fiducials, we validate the transformation on a separate set of $N _ { t }$ target points. The TRE is

$$
\mathrm { T R E } = \sqrt { \frac { 1 } { N _ { t } } \sum _ { j = 1 } ^ { N _ { t } } \Bigl \| \mathbf { y } _ { j } ^ { \mathrm { G T } } - \bigl ( \mathbf { S } \mathbf { y } _ { j } ^ { \mathrm { m e a s } } + \mathbf { d } \bigr ) \Bigr \| ^ { 2 } } .\tag{B.3}
$$

The same transformation can be applied to a measured trajectory $\{ \mathbf { p } _ { 1 } ^ { \mathrm { m e a s } } , \ldots , \mathbf { p } _ { N _ { p } } ^ { \mathrm { m e a s } } \}$ to obtain

$$
\mathbf { p } _ { i } ^ { \mathrm { t r a n s } } = \mathbf { S } \mathbf { p } _ { i } ^ { \mathrm { m e a s } } + \mathbf { d } , \qquad i = 1 , \ldots , N _ { p } .
$$

G. Sideris et al.

For each transformed point ${ \bf p } _ { i } ^ { \mathrm { t r a n s } }$ we locate its nearest neighbour $\mathbf { q } _ { i } ^ { \mathrm { G T } }$ in the GT trajectory $\mathbf { T } ^ { \mathrm { G T } }$ :

$$
\begin{array} { r } { \mathbf q _ { i } ^ { \mathrm { G T } } = \underset { \mathbf q \in \mathbf T ^ { \mathrm { G T } } } { \arg \operatorname* { m i n } } \Bigl \| \mathbf p _ { i } ^ { \mathrm { t r a n s } } - \mathbf q \Bigr \| . } \end{array}
$$

The trajectory TRE is then

$$
\mathrm { T R E } _ { \mathrm { t r a j } } = \sqrt { \frac { 1 } { N _ { p } } \sum _ { i = 1 } ^ { N _ { p } } \Bigl \lVert \mathbf { p } _ { i } ^ { \mathrm { t r a n s } } - \mathbf { q } _ { i } ^ { \mathrm { G T } } \Bigr \rVert ^ { 2 } } .
$$

This nearest-neighbour metric evaluates spatial agreement with the reference trajectory without requiring temporal alignment; it does not quantify tracking latency or alongpath lag.

## C. Supplementary Data

This section provides supplementary data supporting the analyses in the main text. Table C.1 presents detailed perscenario results for the head phantom experiment, breaking down performance for the top, right, and left occlusion configurations. Figure C.1 shows FRE distributions for both experiments. Tables C.2–C.4 provide complete pairwise statistical comparisons across camera counts and filtering methods.

Registration RMSE in mm (median ± standard deviation) across 20 Head Phantom Experiment trials for Top, Right, and Left Occlusion Scenarios.
<table><tr><td rowspan="2"></td><td rowspan="2"># of Cameras</td><td colspan="3">Monocular Pose Fusion</td><td colspan="3">Triangulation</td></tr><tr><td>Basic</td><td>Kalman</td><td>Kalman Adaptive</td><td>Basic</td><td>Kalman</td><td>Kalman Adaptive</td></tr><tr><td colspan="2">Top Occlusion Scenario</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Fiducial Point Median Registration RMSE</td><td></td><td>1.7044 (±0.3059) 1.3203 (±0.2003)</td><td>1.6380 (±0.2716)</td><td>1.6414 (±0.2710) 1.2201 (±0.1589)</td><td>N/A 1.1210 (±0.1340)</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>1.1006 (±0.1276)</td><td>1.2829 (±0.1911)</td><td>1.1227 (±0.1088)</td><td>1.0796 (±0.1323)</td><td>1.1065 (±0.1322)</td><td>1.0839 (±0.1245)</td></tr><tr><td>1２３4</td><td>1.0754 (±0.1302)</td><td>1.1261 (±0.1387)</td><td>0.9978 (±0.0973)</td><td></td><td>1.0707 (±0.1310)</td><td>1.0701 (±0.1260)</td></tr><tr><td></td><td>0.9952 (±0.1042)</td><td>1.0981 (±0.1330) 1.0332 (±0.1047)</td><td>0.8973 (±0.0977)</td><td>1.0196 (±0.1273) 1.0695 (±0.0735)</td><td>1.0106 (±0.1186)</td><td>1.0093 (±0.0947) 1.0687 (±0.0649)</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td>1.0686 (±0.0715)</td><td></td></tr><tr><td rowspan="5">Target Point Median Registration RMSE</td><td></td><td>3.1144 (±0.8990) 1.9501 (±0.6432)</td><td>3.0852 (±0.8704)</td><td>3.0815 (±0.8676)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>1.5099 (±0.5814)</td><td>1.8828 (±0.6241)</td><td>2.0217 (±0.4530)</td><td>1.3348 (±0.4934)</td><td>1.3154 (±0.4601)</td><td>1.2786 (±0.3940)</td></tr><tr><td>1234</td><td>1.6747 (±0.5834)</td><td>1.5901 (±0.5923)</td><td>1.6486 (±0.4850)</td><td>1.2410 (±0.3526)</td><td>1.2443 (±0.3482)</td><td>1.2410 (±0.3305)</td></tr><tr><td></td><td>1.5759 (±0.4657)</td><td>1.7824 (±0.5952)</td><td>1.4955 (±0.3396)</td><td>1.5105 (±0.3747)</td><td>1.4975 (±0.3682)</td><td>1.4960 (±0.3562)</td></tr><tr><td>5</td><td></td><td>1.6535 (±0.5066)</td><td>1.4813 (±0.3523)</td><td>1.2997 (±0.1885)</td><td>1.2934 (±0.1897)</td><td>1.2774 (±0.1938)</td></tr><tr><td rowspan="5">Target Trajectory Median Registration RMSE</td><td>１２３</td><td>2.8075 (±0.6904) 1.4671 (±0.5825)</td><td>2.6879 (±0.6730)</td><td>2.6818 (±0.6707)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td></td><td>1.3492 (±0.5700)</td><td>1.4175 (±0.4565)</td><td>1.0095 (±0.2685)</td><td>1.0168 (±0.2660)</td><td>0.9843 (±0.2669)</td></tr><tr><td>4</td><td>0.9477 (±0.4894) 1.2358 (±0.4434)</td><td>1.0509 (±0.5248)</td><td>1.1422 (±0.4320)</td><td>0.9462 (±0.2571)</td><td>0.9396 (±0.2535)</td><td>0.8757 (±0.2505)</td></tr><tr><td>5</td><td></td><td>1.2417 (±0.4728)</td><td>1.2880 (±0.3169)</td><td>1.6845 (±0.5110)</td><td>1.6895 (±0.5013)</td><td>1.6744 (±0.4829)</td></tr><tr><td></td><td>2.8220 (±1.2112)</td><td>2.6056 (±1.1142)</td><td>1.4894 (±0.3249)</td><td>1.1245 (±0.1549)</td><td>1.1151 (±0.1551)</td><td>1.1197 (±0.1641)</td></tr><tr><td colspan="8">Right Occlusion Scenario</td></tr><tr><td rowspan="5">Fiducial Point Median Registration RMSE</td><td>123</td><td>1.4793 (±0.3102)</td><td>1.3591 (±0.2953)</td><td>1.3593 (±0.2951)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>1.3039 (±0.2155)</td><td>1.3258 (±0.1876)</td><td>1.2550 (±0.1395)</td><td>1.1168 (±0.1560)</td><td>1.0922 (±0.1625)</td><td>1.0581 (±0.1781)</td></tr><tr><td></td><td>1.1142 (±0.1380)</td><td>1.1145 (±0.1318)</td><td>0.9859 (±0.1067)</td><td>1.4885 (±0.2799)</td><td>1.4823 (±0.2776)</td><td>1.4742 (±0.2769)</td></tr><tr><td>4</td><td>1.0281 (±0.1154)</td><td>1.0401 (±0.1168)</td><td>0.9272 (±0.0980)</td><td>1.1402 (±0.0809)</td><td>1.1341 (±0.0803)</td><td>1.1175 (±0.0797)</td></tr><tr><td>5</td><td>0.9952 (±0.1042)</td><td>1.0332 (±0.1047)</td><td>0.8973 (±0.0977)</td><td>1.0695 (±0.0735)</td><td>1.0686 (±0.0715)</td><td>1.0687 (±0.0649)</td></tr><tr><td rowspan="5">Target Point Median Registration RMSE</td><td>123</td><td>3.0655 (±1.7127)</td><td>3.1219 (±1.5448)</td><td>3.1242 (±1.5422)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>2.7233 (±0.8803)</td><td>2.7652 (±0.8440)</td><td>2.6429 (±0.4463)</td><td>1.9293 (±0.4214)</td><td>1.9299 (±0.4211)</td><td>1.9409 (±0.4400)</td></tr><tr><td>4</td><td>2.1931 1 (±0.5432)</td><td>2.2858 (±0.5393)</td><td>1.9690 (±0.3273)</td><td>2.4350 (±0.3738)</td><td>2.4673 (±0.3676)</td><td>2.4078 (±0.3623)</td></tr><tr><td></td><td>1.7375 (±0.4568)</td><td>1.7870 (±0.4484)</td><td>1.7592 (±0.3087)</td><td>1.4504 (±0.2510)</td><td>1.4490 (±0.2551)</td><td>1.4051 (±0.2637)</td></tr><tr><td>5</td><td>1.5759 (±0.4657)</td><td>1.6535 (±0.5066)</td><td>1.4813 (±0.3523)</td><td>1.2997 (±0.1885)</td><td>1.2934 (±0.1897)</td><td>1.2774 (±0.1938)</td></tr><tr><td rowspan="5">Target Trajectory Median Registration RMSE</td><td>123</td><td>2.9653 (±0.4777)</td><td>2.9073 (±0.5573)</td><td>2.9148 (±0.5613)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>2.4405 (±0.4291)</td><td>2.4779 (±0.4122)</td><td>2.3539 (±0.3399)</td><td>1.9210 (±0.3796)</td><td>1.8870 (±0.3896)</td><td>1.7704 (±0.4092)</td></tr><tr><td></td><td>4.5664 (±2.2443)</td><td>3.8499 (±1.8934)</td><td>2.0279 (±0.2788)</td><td>2.3899 (±0.4939)</td><td>2.4070 (±0.4932)</td><td>2.2468 (±0.5079)</td></tr><tr><td>4 5</td><td>3.4373 (±1.6122)</td><td>3.1312 (±1.4902)</td><td>1.7619 (±0.2901)</td><td>1.0785 (±0.1689)</td><td>1.0800 (±0.1709)</td><td>1.0534 (±0.1831)</td></tr><tr><td></td><td>2.8220 (±1.2112)</td><td>2.6056 (±1.1142)</td><td>1.4894 (±0.3249)</td><td>1.1245 (±0.1549)</td><td>1.1151 (±0.1551)</td><td>1.1197 (±0.1641)</td></tr><tr><td colspan="8">Left Occlusion Scenario</td></tr><tr><td rowspan="5">Fiducial Point Median Registration RMSE</td><td></td><td>1.3240 (±0.3622)</td><td>1.2817 (±0.3480)</td><td>1.2830 (±0.3431)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>123</td><td>1.0823 (±0.1837)</td><td>1.1267 (±0.2278)</td><td>1.1479 (±0.2157)</td><td>1.4390 (±0.2801)</td><td>1.4379 (±0.2826)</td><td>1.4375 (±0.2870)</td></tr><tr><td></td><td>0.9535 (±0.1095)</td><td>0.9849 (±0.1322)</td><td>0.9397 (±0.1345)</td><td>1.2088 (±0.1177)</td><td>1.2100 (±0.1155)</td><td>1.2244 (±0.1085)</td></tr><tr><td>4</td><td>0.9884 (±0.0959)</td><td>0.9986 (±0.1153)</td><td>0.8771 (±0.1132)</td><td>1.1107 (±0.0623)</td><td>1.1086 (±0.0627)</td><td>1.1035 (±0.0648)</td></tr><tr><td>5</td><td>0.9952 (±0.1042)</td><td>1.0332 (±0.1047)</td><td>0.8973 (±0.0977)</td><td>1.0695 (±0.0735)</td><td>1.0686 (±0.0715)</td><td>1.0687 (±0.0649)</td></tr><tr><td rowspan="5">Target Point Median Registration RMSE</td><td>1</td><td>2.2955 (±0.8513)</td><td>2.1928 (±0.8360)</td><td>2.1547 (±0.8289)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>23</td><td>1.7188 (±0.6754)</td><td>1.8659 (±0.7442)</td><td>1.6537 (±0.6140)</td><td>2.0332 (±0.4706)</td><td>2.0251 (±0.4777)</td><td>1.9662 (±0.4912)</td></tr><tr><td></td><td>1.6418 (±0.5158)</td><td>1.8273 (±0.6123)</td><td>1.5231 (±0.4225)</td><td>1.4532 (±0.2639)</td><td>1.4461 (±0.2580)</td><td>1.4408 (±0.2474)</td></tr><tr><td>4</td><td>1.8609 (±0.5333)</td><td>1.8935 (±0.5903)</td><td>1.4831 (±0.3792)</td><td>1.3483 (±0.1689)</td><td>1.3472 (±0.1685)</td><td>1.3647 (±0.1721)</td></tr><tr><td>5</td><td>1.5759 (±0.4657)</td><td>1.6535 (±0.5066)</td><td>1.4813 (±0.3523)</td><td>1.2997 (±0.1885)</td><td>1.2934 (±0.1897)</td><td>1.2774 (±0.1938)</td></tr><tr><td rowspan="5">Target Trajectory Median Registration RMSE</td><td></td><td>1.5482 (±0.7737)</td><td>1.4661 (±0.7600)</td><td>1.4502 (±0.7450)</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>1.3190 (±0.4893)</td><td>1.3271 (±0.6025)</td><td>1.2349 (±0.4687)</td><td>1.3880 (±0.3130)</td><td>1.3727 (±0.3154)</td><td>1.2811 (±0.3248)</td></tr><tr><td>1234</td><td>4.4467 (±2.3635)</td><td>3.7694 (±2.0250)</td><td>1.4716 (±0.4236)</td><td>1.4905 (±0.2860)</td><td>1.4785 (±0.2856)</td><td>1.4609 (±0.2918)</td></tr><tr><td></td><td>3.4916 (±1.5786)</td><td>3.1765 (±1.4025)</td><td>1.5682 (±0.3566)</td><td>1.2017 (±0.1898)</td><td>1.1969 (±0.1919)</td><td>1.1478 (±0.2060)</td></tr><tr><td>5</td><td>2.8220 (±1.2112)</td><td>2.6056 (±1.1142)</td><td>1.4894 (±0.3249)</td><td>1.1245 (±0.1549)</td><td>1.1151 (±0.1551)</td><td>1.1197 (±0.1641)</td></tr></table>

![](images/4cd121656f043d4176b15a1ce7cfc158af719a0be4a905a9ec65f15650d5bed0.jpg)  
(a) FRE – Monocular Pose Fusion (Grid)

![](images/9f74084dc743629f4463d1d25e720a9e5621ba1cd2cc9fb3ea0c08410fe65f53.jpg)  
(b) FRE – Triangulation (Grid)

![](images/34dc82fab0a3275fa5f76aeebac38775a0548cd9e9bd272f22628838537b067e.jpg)  
(c) FRE – Monocular Pose Fusion (Head Phantom)

![](images/f5247153a43209d19c55d13612886b14d9f95cb8abe8c65b74f2969a3855ca25.jpg)  
(d) FRE – Triangulation (Head Phantom)  
Figure C.1: Error-distribution violins for FRE across the two tracking pipelines. Top row: Grid experiment; Bottom row: Head Phantom experiment. Left column = Monocular Pose Fusion, Right column = Triangulation. Boxes = IQR, red dots / curved boxes = median, violins = kernel density. Median accuracy value reported in each box below the violin plot. Statistical comparisons use 20 paired trial summaries per phantom.

Table C.2  
Two-sided paired Wilcoxon signed-rank �-values for pooled camera-count comparisons, with Holm correction over the ten count pairs within each experiment and metric. Trial mean squares are averaged equally across available tracking methods and filters before taking the square root (pose fusion only at one camera; both tracking methods at two to five cameras). All tests use 20 paired trial summaries. † marks a comparison whose significance decision at � = 0.05 difers under the Holm-adjusted paired sign-test sensitivity analysis.
<table><tr><td rowspan="2">Camera counts</td><td colspan="3">Grid</td><td colspan="3">Head phantom</td></tr><tr><td>FRE</td><td>Point TRE</td><td>Trajectory TRE</td><td>FRE</td><td>Point TRE</td><td>Trajectory TRE</td></tr><tr><td>1 versus 2</td><td>1.91e-05</td><td>1.91e-05</td><td>3.15e-04</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td></tr><tr><td>1 versus 3</td><td>2.67e-05</td><td>1.91e-05</td><td>0.0408†</td><td>2.86e-05</td><td>1.91e-05</td><td>0.8182</td></tr><tr><td>1 versus 4</td><td>1.91e-05</td><td>1.91e-05</td><td>0.1392</td><td>1.91e-05</td><td>1.91e-05</td><td>0.0099†</td></tr><tr><td>1 versus 5</td><td>1.91e-05</td><td>1.91e-05</td><td>1.14e-04</td><td>1.91e-05</td><td>1.91e-05</td><td>0.0029</td></tr><tr><td>2 versus 3</td><td>0.0136</td><td>0.0073</td><td>1.91e-05</td><td>0.0032</td><td>1.34e-04</td><td>0.0099†</td></tr><tr><td>2 versus 4</td><td>1.91e-05</td><td>1.91e-05</td><td>0.4091</td><td>1.91e-05</td><td>1.91e-05</td><td>0.0983</td></tr><tr><td>2 versus 5</td><td>1.91e-05</td><td>1.91e-05</td><td>5.34e-04</td><td>1.91e-05</td><td>1.91e-05</td><td>0.8182</td></tr><tr><td>3 versus 4</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>0.0012</td></tr><tr><td>3 versus 5</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td></tr><tr><td>4 versus 5</td><td>1.91e-05</td><td>1.91e-05</td><td>1.91e-05</td><td>0.0020</td><td>1.91e-05</td><td>1.91e-05</td></tr></table>

Table C.3  
Two-sided paired Wilcoxon signed-rank �-values for filtering methods (Grid), with Holm correction over the three filter pairs within each camera count, tracking method, and metric. All tests use 20 paired trial summaries. † marks a comparison whose significance decision at � = 0.05 difers under the Holm-adjusted paired sign-test sensitivity analysis.
<table><tr><td>Camera count</td><td>Tracking method</td><td>Metric</td><td>Basic versus Kalman</td><td>Basic versus Kalman Adaptive</td><td>Kalman versus Kalman Adaptive</td></tr><tr><td rowspan="3">1</td><td rowspan="3">Monocular Pose Fusion</td><td>FRE</td><td>1.89e-04</td><td>1.89e-04</td><td>0.8124</td></tr><tr><td>Point TRE</td><td>5.04e-04</td><td>9.65e-04</td><td>0.7562</td></tr><tr><td>Trajectory TRE</td><td>5.72e-06</td><td>5.72e-06</td><td>1.05e-04</td></tr><tr><td rowspan="3">2</td><td>Monocular</td><td>FRE</td><td>0.5459</td><td>0.0097†</td><td>0.0060^†</td></tr><tr><td>Pose Fusion</td><td>Point TRE</td><td>0.2858</td><td>0.2858</td><td>0.1595</td></tr><tr><td></td><td>Trajectory TRE</td><td>5.72e-06</td><td>1.14e-05</td><td>0.0192</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Triangulation</td><td>FRE</td><td>5.72e-06</td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td>Point TRE</td><td>5.72e-06</td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td>Trajectory TRE</td><td>5.72e-06</td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td rowspan="3">3</td><td>Monocular</td><td>FRE</td><td>0.5459</td><td>0.0026</td><td>0.0029</td></tr><tr><td>Pose Fusion</td><td>Point TRE Trajectory TRE</td><td>0.9563 1.72e-05</td><td>0.3162</td><td>0.3162</td></tr><tr><td></td><td></td><td></td><td>1.64e-04</td><td>0.0532</td></tr><tr><td rowspan="3">3</td><td>Triangulation</td><td>FRE Point TRE</td><td>2.86e-05</td><td>2.86e-05</td><td>2.86e-05</td></tr><tr><td></td><td>Trajectory TRE</td><td>5.72e-06 5.72e-06</td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td></td><td></td><td></td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td rowspan="3">4</td><td>Monocular Pose Fusion</td><td>FRE Point TRE</td><td>0.4524</td><td>0.0081</td><td>0.016†</td></tr><tr><td></td><td>Trajectory TRE</td><td>0.7768 3.15e-04</td><td>0.6485</td><td>0.7768</td></tr><tr><td></td><td></td><td></td><td>3.36e-04</td><td>0.1054</td></tr><tr><td rowspan="3">4</td><td>Triangulation</td><td>FRE Point TRE</td><td>1.34e-04</td><td>1.09e-04</td><td>1.09e-04</td></tr><tr><td></td><td>Trajectory TRE</td><td>5.72e-06 5.72e-06</td><td>1.91e-05</td><td>1.91e-05</td></tr><tr><td></td><td></td><td></td><td>5.72e-06</td><td>5.72e-06</td></tr><tr><td rowspan="3">5</td><td>Monocular</td><td>FRE</td><td>0.3884</td><td>0.0515</td><td>0.0515</td></tr><tr><td>Pose Fusion</td><td>Point TRE</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td></td><td>Trajectory TRE</td><td>5.04e-04</td><td>7.90e-04</td><td>0.1327</td></tr><tr><td rowspan="3">5</td><td>Triangulation</td><td>FRE</td><td>0.0110</td><td>0.0112†</td><td>0.0112†</td></tr><tr><td></td><td>Point TRE</td><td>1.72e-05</td><td>1.26e-04</td><td>1.34e-04</td></tr><tr><td></td><td>Trajectory TRE</td><td>5.72e-06</td><td>5.72e-06</td><td>5.72e-06</td></tr></table>

Table C.4  
Two-sided paired Wilcoxon signed-rank �-values for filtering methods (Head phantom), with Holm correction over the three filter pairs within each camera count, tracking method, and metric. All tests use 20 paired trial summaries. † marks a comparison whose significance decision at � = 0.05 difers under the Holm-adjusted paired sign-test sensitivity analysis.
<table><tr><td>Camera count</td><td>Tracking method</td><td>Metric</td><td>Basic versus Kalman</td><td>Basic versus Kalman Adaptive</td><td>Kalman versus Kalman Adaptive</td></tr><tr><td rowspan="3">1</td><td rowspan="3">Monocular Pose Fusion</td><td>FRE</td><td>7.63e-06</td><td>5.72e-06</td><td>8.51e-04</td></tr><tr><td>Point TRE</td><td>0.4049</td><td>0.4049</td><td>0.0983</td></tr><tr><td>Trajectory TRE</td><td>0.0592</td><td>0.0577</td><td>0.0592</td></tr><tr><td rowspan="3">2</td><td>Monocular</td><td>FRE</td><td>0.6976</td><td>0.6976</td><td>0.4949</td></tr><tr><td>Pose Fusion</td><td>Point TRE</td><td>0.1794</td><td>0.2611</td><td>0.0719</td></tr><tr><td></td><td>Trajectory TRE</td><td>0.8124</td><td>0.0645</td><td>0.1165</td></tr><tr><td rowspan="3">2</td><td rowspan="3">Triangulation</td><td>FRE</td><td>0.0017</td><td>0.0012</td><td>0.0012</td></tr><tr><td>Point TRE</td><td>0.0242†</td><td>0.0242†</td><td>0.0219</td></tr><tr><td>Trajectory TRE</td><td>0.0023</td><td>1.14e-05</td><td>5.72e-06</td></tr><tr><td rowspan="3">3</td><td>Monocular</td><td>FRE</td><td>0.0592†</td><td>0.0592</td><td>0.0219†</td></tr><tr><td>Pose Fusion</td><td>Point TRE Trajectory TRE</td><td>0.0242† 0.0037</td><td>0.0242†</td><td>0.0070†</td></tr><tr><td></td><td></td><td></td><td>1.64e-04</td><td>1.09e-04</td></tr><tr><td rowspan="3">3</td><td rowspan="3">Triangulation</td><td>FRE</td><td>0.0020</td><td>4.01e-04</td><td>4.01e-04</td></tr><tr><td>Point TRE</td><td>0.0095</td><td>0.0095</td><td>0.0095</td></tr><tr><td>Trajectory TRE</td><td>8.51e-04</td><td>7.63e-06</td><td>5.72e-06</td></tr><tr><td rowspan="3">4</td><td>Monocular Pose Fusion</td><td>FRE</td><td>0.0083</td><td>0.0034</td><td>4.01e-04</td></tr><tr><td></td><td>Point TRE Trajectory TRE</td><td>0.0121 0.0032†</td><td>0.0047</td><td>0.0012</td></tr><tr><td></td><td></td><td></td><td>3.36e-04</td><td>1.43e-04</td></tr><tr><td rowspan="3">4</td><td rowspan="3">Triangulation</td><td>FRE</td><td>0.0018</td><td>0.0018</td><td>0.0018</td></tr><tr><td>Point TRE Trajectory TRE</td><td>0.0321†</td><td>0.0321</td><td>0.0321†</td></tr><tr><td></td><td>0.3884</td><td>0.0242†</td><td>0.0051†</td></tr><tr><td rowspan="3">5</td><td>Monocular</td><td>FRE</td><td>0.0136†</td><td>0.0084†</td><td>5.04e-04</td></tr><tr><td>Pose Fusion</td><td>Point TRE</td><td>0.1429</td><td>0.0242†</td><td>0.0219</td></tr><tr><td></td><td>Trajectory TRE</td><td>0.0014</td><td>4.01e-04</td><td>4.01e-04</td></tr><tr><td rowspan="3">5</td><td rowspan="3">Triangulation</td><td>FRE</td><td>0.0219†</td><td>0.0385†</td><td>0.0385</td></tr><tr><td>Point TRE</td><td>0.0515</td><td>0.0515</td><td>0.0515</td></tr><tr><td>Trajectory TRE</td><td>0.3683</td><td>0.1063</td><td>0.0577</td></tr></table>

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) in order to assist with language editing, LAT<sub>E</sub>X formatting, and code review and revision. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Data availability

The hardware designs, technical drawings, bill of materials, and the calibration, tracking, and registration software are available at https://nist.mni.mcgill.ca/software/. The 5,000 synchronized multi-view tracking videos, the simultaneously recorded fusionTrack 500 measurements, and the per-trial registration-error results are available at https:// nist.mni.mcgill.ca/data/. All materials are released under the Creative Commons Attribution 4.0 (CC BY 4.0) licence.

## CRediT authorship contribution statement

George Sideris: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing, Project administration. Justin Cree: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – review & editing. Andrew Stirling: Methodology, Software, Writing – review & editing. Mamadou Ly: Methodology, Writing – review & editing. Étienne Léger: Conceptualization, Investigation, Supervision, Writing – review & editing. D. Louis Collins: Conceptualization, Funding acquisition, Resources, Supervision, Writing – review & editing.

## References

Asadi, Z., Asadi, M., Kazemipour, N., Léger, É., Kersten-Oertel, M., 2024. A decade of progress: bringing mixed reality image-guided surgery systems in the operating room. Computer Assisted Surgery 29, 2355897. URL: https://doi.org/10.1080/24699322.2024.2355897, doi:10.1080/24699322.2024.2355897. pMID: 38794834.

Asselin, M., Lasso, A., Ungi, T., Fichtinger, G., 2018. Towards webcambased tracking for interventional navigation, in: Fei, B., III, R.J.W. (Eds.), Medical Imaging 2018: Image-Guided Procedures, Robotic Interventions, and Modeling, International Society for Optics and Photonics. SPIE. p. 1057627. URL: https://doi.org/10.1117/12.2293904, doi:10. 1117/12.2293904.

Atracsys, 2024. Fusiontrack 500. https://atracsys.com/fusiontrack-500/. Accessed: 2024-01-03.

Bradski, G., 2000. The OpenCV Library. Dr. Dobb’s Journal of Software Tools .

Chen, L., Ma, L., Zhang, F., Yang, X., Sun, L., 2023. An intelligent tracking system for surgical instruments in complex surgical environment. Expert Systems with Applications 230, 120743. URL: https:// www.sciencedirect.com/science/article/pii/S0957417423012459, doi:10. 1016/j.eswa.2023.120743.

ClaroNav, 2024. Cranial navigation. https://medical.claronav.com/ cranial-navigation. Accessed: 2026-02-15.

Drouin, S., Kochanowska, A., Kersten-Oertel, M., Gerard, I.J., Zelmann, R., Nigris, D.D., Bériault, S., Arbel, T., Sirhan, D., Sadikot, A.F., Hall, J.A., Sinclair, D.S., Petrecca, K., DelMaestro, R.F., Collins, D.L., 2017. IBIS: an OR ready open-source platform for image-guided neurosurgery. International Journal of Computer Assisted Radiology and Surgery 12, 363–378. URL: https://doi.org/10.1007/s11548-016-1478-0, doi:10. 1007/s11548-016-1478-0.

Fattori, G., Lomax, A.J., Weber, D.C., Safai, S., 2021. Technical assessment of the ndi polaris vega optical tracking system. Radiation Oncology 16, 87. URL: https://doi.org/10.1186/s13014-021-01804-7, doi:10.1186/ s13014-021-01804-7.

Fitzpatrick, J., West, J., Maurer, C., 1998. Predicting error in rigid-body point-based registration. IEEE Transactions on Medical Imaging 17, 694–702. doi:10.1109/42.736021.

García-Ruiz, P., Romero-Ramirez, F.J., Muñoz-Salinas, R., Marín-Jiménez, M.J., Medina-Carnicer, R., 2023. Fiducial objects: Custom design and evaluation. Sensors 23. URL: https://www.mdpi.com/1424-8220/23/24/ 9649, doi:10.3390/s23249649.

Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F., Marín-Jiménez, M., 2014. Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition 47, 2280–2292. URL: https://www.sciencedirect.com/science/article/ pii/S0031320314000235, doi:10.1016/j.patcog.2014.01.005.

Gerard, I.J., Collins, D.L., 2015. An analysis of tracking error in imageguided neurosurgery. International Journal of Computer Assisted Radiology and Surgery 10, 1579–1588. URL: https://doi.org/10.1007/ s11548-014-1145-2, doi:10.1007/s11548-014-1145-2.

Gueziri, H.E., Rabau, O., Santaguida, C., Collins, L., 2021. Evaluation of an ultrasound-based navigation system for spine neurosurgery: A porcine cadaver study. Frontiers in Oncology 11, 619204. doi:10.3389/fonc. 2021.619204.

Han, J., Luo, M., Qin, Y., 2023. Online viewpoint optimization method for high-accuracy intraoperative optical measurement in orthopedic surgeries: An active navigation approach. Measurement 223, 113648. URL: https://www.sciencedirect.com/science/article/pii/ S0263224123012125, doi:10.1016/j.measurement.2023.113648.

Hein, J., Cavalcanti, N., Suter, D., Zingg, L., Carrillo, F., Calvet, L., Farshad, M., Navab, N., Pollefeys, M., Fürnstahl, P., 2025. Nextgeneration surgical navigation: Marker-less multi-view 6DoF pose estimation of surgical instruments. Medical Image Analysis 103, 103613. URL: https://doi.org/10.1016/j.media.2025.103613, doi:10. 1016/j.media.2025.103613.

Infinium Medical, 2026. Infinium Medical - S400 Series. https: //infiniummedical.com/product/s400-series-operating-room-lights/ ?attachment\_id=17293&download\_file=jlnhp2o0pzfpb. Accessed: 2026- 02-15.

Jackson, P., Simon, R., Linte, C., 2020. Surgical tracking, registration, and navigation characterization for image-guided renal interventions, in: 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pp. 5081–5084. doi:10.1109/ EMBC44109.2020.9175270.

Jin, R., Jung, B., 2022. Magnetic tracking system for heart surgery. IEEE Transactions on Biomedical Circuits and Systems 16, 275–286. doi:10.1109/TBCAS.2022.3163172.

Kalman, R.E., 1960. A new approach to linear filtering and prediction problems. Transactions of the ASME–Journal of Basic Engineering 82, 35–45.

Léger, É., Gueziri, H.E., Collins, D.L., Popa, T., Kersten-Oertel, M., 2021. Evaluation of low-cost hardware alternatives for 3d freehand ultrasound reconstruction in image-guided neurosurgery, in: Noble, J.A., Aylward, S., Grimwood, A., Min, Z., Lee, S.L., Hu, Y. (Eds.), Simplifying Medical Ultrasound, Springer International Publishing, Cham. pp. 106–115.

Lin, Z., Lei, C., Yang, L., 2023. Modern image-guided surgery: A narrative review of medical image processing and visualization. Sensors 23. URL: https://www.mdpi.com/1424-8220/23/24/9872, doi:10.3390/s23249872.

Marinetto, E., García-Mato, D., GarcíA, A., Martínez, S., Desco, M., Pascau, J., 2018. Multicamera optical tracker assessment for computer aided surgery applications. IEEE Access 6, 64359–64370. doi:10.1109/

ACCESS.2018.2878323.

Mehbodniya, A.H., Moghavvemi, M., Narayanan, V., Waran, V., 2019. Frequency and causes of line of sight issues during neurosurgical procedures using optical image-guided systems. World Neurosurgery 122, e449– e454. doi:10.1016/j.wneu.2018.10.069.

Moviglia, J.H., Stallkamp, J., 2026. Monocular near-infrared optical tracking with retroreflective fiducial markers for high-accuracy image-guided surgery. Sensors 26, 357. URL: https://doi.org/10.3390/s26020357, doi:10.3390/s26020357.

Northern Digital Inc., 2024. Polaris vega optical navigation technology. https://www.ndigital.com/optical-navigation-technology/ polaris-vega/. Accessed: 2026-02-15.

Olson, E., 2011. Apriltag: A robust and flexible visual fiducial system, in: 2011 IEEE International Conference on Robotics and Automation, pp. 3400–3407. doi:10.1109/ICRA.2011.5979561.

Pfeifer, J.H., Borbáth, Á., Dietz, C., Lueth, T.C., 2016. A new module combining two tracking cameras to expand the workspace of surgical navigation systems, in: 2016 IEEE/SICE International Symposium on System Integration (SII), pp. 477–482. doi:10.1109/SII.2016.7844044.

Popescu, D.C., Dumitrache, I., Caramihai, S.I., Cernaianu, M.O., 2020. High precision positioning with multi-camera setups: Adaptive kalman fusion algorithm for fiducial markers. Sensors 20. URL: https://www. mdpi.com/1424-8220/20/9/2746, doi:10.3390/s20092746.

Poulin, F., Amiot, L.P., 2002. Interference during the use of an electromagnetic tracking system under or conditions. Journal of Biomechanics 35, 733–737. URL: https://www.sciencedirect.com/science/article/pii/ S0021929002000362, doi:10.1016/S0021-9290(02)00036-2.

Raspberry Pi, 2026. Raspberry pi global shutter camera. https://www. raspberrypi.com/products/raspberry-pi-global-shutter-camera/. Accessed: 2026-02-15.

Sorriento, A., Porfido, M.B., Mazzoleni, S., Calvosa, G., Tenucci, M., Ciuti, G., Dario, P., 2020. Optical and electromagnetic tracking systems for biomedical applications: A critical review on potentialities and limitations. IEEE Reviews in Biomedical Engineering 13, 212–232. doi:10.1109/RBME.2019.2939091.

Stryker Corporation, 2024. Stryker nav3i. https://www.stryker.com/sg/en/ navigation/products/nav3i/index.html. Accessed: 2026-02-15.

Taleb, A., Guigou, C., Leclerc, S., Lalande, A., Bozorg Grayeli, A., 2023. Image-to-patient registration in computer-assisted surgery of head and neck: State-of-the-art, perspectives, and challenges. Journal of Clinical Medicine 12. URL: https://www.mdpi.com/2077-0383/12/16/5398, doi:10.3390/jcm12165398.

Tsui, D., Ramos, K., Melentyev, C., Rajan, A., Tam, M., Jo, M., Ahadian, F., Talke, F.E., 2024. A low-cost, open-source-based optical surgical navigation system using stereoscopic vision. Microsystem Technologies URL: https://doi.org/10.1007/s00542-024-05668-1, doi:10.1007/ s00542-024-05668-1.

Vaccarella, A., De Momi, E., Enquobahrie, A., Ferrigno, G., 2013. Unscented kalman filter based sensor fusion for robust optical and electromagnetic tracking in surgical navigation. IEEE Transactions on Instrumentation and Measurement 62, 2067–2081. doi:10.1109/TIM. 2013.2248304.

Van den Bempt, M., Liebregts, J., Maal, T., Bergé, S., Xi, T., 2018. Toward a higher accuracy in orthognathic surgery by using intraoperative computer navigation, 3d surgical guides, and/or customized osteosynthesis plates: A systematic review. Journal of Cranio-Maxillofacial Surgery 46, 2108–2119. URL: https://www.sciencedirect.com/science/article/ pii/S1010518218305559, doi:10.1016/j.jcms.2018.10.012.

Wallace, N., Schafer, N.E., Freedman, B.A., Nassr, A., Currier, B.L., Patel, R., Aleem, I.S., 2019. Computer-assisted navigation in complex cervical spine surgery: tips and tricks. Journal of Spine Surgery 6. URL: https://jss.amegroups.org/article/view/4814.

Wang, J., Meng, M.Q.H., Ren, H., 2015. Towards occlusion-free surgical instrument tracking: A modular monocular approach and an agile calibration method. IEEE Transactions on Automation Science and Engineering 12, 588–595. doi:10.1109/TASE.2015.2388537.

Wu, P.C., Wang, R., Kin, K., Twigg, C., Han, S., Yang, M.H., Chien, S.Y., 2017. DodecaPen: Accurate 6DoF tracking of a passive stylus, in: ACM

Symposium on User Interface Software and Technology (UIST), pp.   
365–374. doi:10.1145/3126594.3126664.