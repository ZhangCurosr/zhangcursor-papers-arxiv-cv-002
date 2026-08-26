# Syn2RealTrack: Bridging the Gap Between Synthetic and Real-World Datasets for Online Multi-View Multi-Target Tracking

Duong Nguyen-Ngoc Tran<sup>∗</sup> , Ngoc Doan-Minh Huynh<sup>∗</sup> , Cu Quoc Le<sup>∗</sup> , Khang Nguyen Hoang<sup>∗</sup> , Long Hoang Pham , Huy-Hung Nguyen , Quoc Pham-Nam Ho , Trinh Le Ba Khanh , Chi Dai Tran , Duong Khac Vu , Son Hong Phan , Hyung-Min Jeon, and Jae Wook Jeon<sup>△</sup>

Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon, South Korea {duongtran, ngochdm, lequoccu2003, khangnguyen, jwjeon}@skku.edu

Abstract. Multi-camera 3D perception systems for warehouse scenes are trained largely on synthetic data and evaluated on physically captured environments. The resulting synthetic-to-real gap, which corrupts ground-plane localization and cross-camera identity association, is usually treated as one deficiency for a single domain-adaptation module to absorb; we argue instead that it enters the pipeline at three separable points: the camera calibration, the object shape prior, and the assumption that the object census is known, each admitting a diferent local remedy. Our online pipeline, Syn2RealTrack, follows this decomposition: lens distortion is recovered from images alone under a calibration that provides none, detections are fused across views by a visibility-weighted part-based descriptor that abstains on occluded parts rather than guessing, person height is measured in closed form from calibration instead of copied from a synthetic prior, and a closed-world cardinality prior is paired with a causal filter that removes the phantom boxes the prior manufactures. The system therefore adapts by reallocating trust between geometry and appearance without retraining a feature extractor. On the AI City Challenge 2026 Track 1 evaluation server it reaches a 3D Higher Order Tracking Accuracy (HOTA) of 52.0118%. The code will be released at https://github.com/SKKUAutoLab/aic26\_mc3dp.

Keywords: Multi-target multi-camera tracking · 3D object perception · Synthetic-to-real transfer

## 1 Introduction

Warehouse automation depends on knowing where every person and every piece of equipment is, in metric floor coordinates, at every instant. A single camera cannot provide this information: it recovers a bearing but not a range, and, among tall racks, a target may disappear behind an obstruction within a few frames. Fixed multi-camera rigs with overlapping fields of view are the standard remedy, and the task they define is multi-target multi-camera (MTMC) 3D perception: recovering each object’s world-frame position, footprint, and orientation, together with an identity that persists through occlusions and handovers between cameras [31]. The obstacle is supervision. Dense 3D boxes and crosscamera identities are expensive to annotate at scale in a working facility [44], so the practical solution has been to train in simulation, where exact ground truth is freely available. What simulation does not capture is the collection of small physical irregularities that accumulate in a real installation: a lens that does not conform to the calibration on file, an object whose true dimensions difer from those in the catalogue, and a floor whose occupancy has not been counted. We present Syn2RealTrack, our AI City Challenge 2026 Track 1 system: an online multi-camera 3D perception pipeline trained mainly on synthetic data and evaluated partly on real scenes [32]; Fig. 1 illustrates the two domains and our point-cloud-guided box refinement on both. Rather than treating the syntheticto-real gap as one domain-adaptation problem, we localize it to three interfaces (camera calibration, object-shape priors, and known object counts) and address each where it arises. Fig. 2 summarizes the framework; our contributions are as follows.

![](images/2d91ff67dabba46fecc07b2cdae143823301838f843b4c52b12a70ab34babc6d.jpg)  
(b) Real-world scene  
Fig. 1: Representative point-cloud-guided 3D box refinement examples on (a) a synthetic and (b) a real-world scene of the AI City Challenge 2026 Track 1 dataset [32]. Red boxes denote the original multi-view tracking prior, green boxes denote the refined 3D boxes, and the colored points denote the local cloud reconstructed by Depth Anything 3 (DA3) [15].

– Localizing the synthetic-to-real gap. With feature extractors trained once and then frozen across domains, we expose calibration, camera overlap, and object counts as a configuration surface that shifts trust from simulationspecific geometry to appearance when these cues are unreliable in real scenes (Tab. 1).

– Distortion-aware camera grouping. As the calibration omits lens distortion, we estimate it with AnyCalib [35], split reference and fisheye views by the Unified Camera Model parameter ξ, undistort footpoints while keeping the provided geometry, and mask fisheye-only zones to avoid invalid matches.

Cross-view fusion that can abstain. We ground detections, fuse views by visibility-weighted part similarity, and propagate single-camera identities before geometric assignment to reject inconsistent world placements; this fusion adds 0.06 HOTA on top of single-view association.

– Measured rather than synthetic geometry. For people, a closed-form estimate from calibration and ankle ground points replaces the synthetic height prior without depth or extra supervision; an RGB-only Depth Anything 3 (DA3) point cloud [15] then refines footprint and yaw while preserving identity and class-specific size.

– A contained closed-world prior. We apply exact per-class counts only in suitable closed-world scenes and suppress unsupported boxes with a causal filter requiring visible-ankle confirmation after reprojection into covering cameras; this filter yields a detection-driven gain at negligible association cost.

## 2 Related Work

Synthetic-to-Real Generalization and Calibration. Multi-camera domain shift is both photometric and geometric. Prior work improves transfer through view-consistent augmentation, depth- and pose-robust BEV features, or perspective invariant rendering [6,10,17,42]; calibration methods instead recover camera geometry from person correspondences, learned perspective cues, or predicted pixel rays [35,37,39,45]. Our pipeline keeps its feature extractors and the provided intrinsics and extrinsics fixed; it adapts only at explicit downstream interfaces and estimates each camera’s omitted lens distortion before ground-plane projection. Geometry, Visibility, and Multi-View Association. Prior work localizes people and recovers metric shape through calibrated pose, projective metrology, RGB geometry, or point-supported box refinement [4, 14, 15, 19, 26, 41]. Association methods use visible-part features [28, 29], probabilistic occupancy, target-count, and camera models [21, 22, 40], or learned BEV and trajectory relationships [34, 43, 47]. Our causal pipeline combines these ideas: class-specific anchors and ankle keypoints recover ground contact and person height; RGB point clouds refine footprint and yaw; shared visible parts, local-identity memory, motion, and a bounded gallery guide sequential association. An external class census conditions assignment when available; otherwise, adaptive tracking and a camera-coverage check suppress unsupported person boxes. The key difference from these works is that our association can abstain: parts occluded in either view are excluded from similarity instead of imputed, and single-camera identities take precedence over geometric matching.

Multi-View Fusion and Tracking. Multi-view detectors aggregate projected features in BEV using convolution, transformers, or multi-height homographies, while CaMuViD exchanges features directly across uncalibrated views [5,8,9,30, 36]. AI City 2025 systems similarly combine global point-cloud detection, clusterto-tracklet mapping, or late multi-view tracking and refinement [13, 23, 31, 38], but rely on benchmark-provided depth and calibration. The 2026 protocol adds real scenes and withholds depth at inference [32]. Our pipeline therefore preserves modular per-view perception and fuses sparse object observations late rather than learning dense BEV aggregation.

![](images/aa2d3d74b9424b2fe7d01e491b2c8d65f4447ce5b768d81211eb93998b3d33d7.jpg)  
Fig. 2: Overview of the proposed framework, read left to right. (A) Data preprocessing. Any Calib [35] estimates per-camera distortion from 31 sampled frames, and the Unified Camera Model (UCM) parameter ξ separates reference (ξ ≤ 0.3, blue) and fisheye-candidate (red) views. (B) Perview perception. RF-DETR [25], ViTPose++ [46], and Keypoint Promptable Re-Identification (KPR) [28] provide detections, keypoints, and visibility-aware part descriptors, which an appearance– IoU Kalman tracker links into local identities. (C) Multi-view fusion. Class-specific ground anchors and visibility-weighted part similarity merge observations within per-class bird’s-eye-view (BEV) gates; local-identity carry-forward and gated Hungarian assignment then form global tracks under cap $\dot { N } _ { k }$ . (D) 3D box construction and refinement. Trajectory-derived yaw, monocular person height, footprint and yaw refinement guided by Depth Anything 3 (DA3) [15], and BEV visibility filtering produce the final boxes. Numbered badges identify the three synthetic-to-real gaps (calibration, shape priors, and closed-world cardinality), and dashed links show cross-stage dependencies between measured height, ground anchoring, and BEV filtering.

## 3 Methodology

We address online 3D multi-view multi-target tracking in warehouse scenes [32, 38], each observed by C static, calibrated RGB cameras over frames $t \in \{ 0 , \ldots , T -$ 1} across seven classes $\mathcal { H } = \{ 0 , \ldots , 6 \}$ (Person, Forklift, NovaCarter, Transporter, FourierGR1T2, AgilityDigit, PalletTruck). Per frame we emit oriented, class-labeled 3D boxes with time- and view-consistent identities. The syntheticto-real gap surfaces at calibration, the shape prior, and the closed-world assumption.

Notation. Camera $c \in \{ 1 , \ldots , C \}$ has intrinsics $\mathbf { K } _ { c } \in \mathbb { R } ^ { 3 \times 3 }$ and world-to-camera extrinsics ${ \bf E } _ { c } = [ { \bf R } _ { c } \ | \ { \bf t } _ { c } ] \in \mathbb { R } ^ { 3 \times 4 } ,$ . An image pixel $\mathbf { u } = ( u , v ) ^ { \top }$ has homogeneous form $\tilde { \mathbf { u } } = ( u , v , 1 ) ^ { \top } ; \mathbf { x } = ( X , Y ) ^ { \top }$ is a metric world point on the ground plane $Z = 0 . \mathrm { ~ A ~ }$ detection carries a normalized center-size box $\mathbf { b } = ( b _ { x } , b _ { y } , b _ { w } , b _ { h } )$

![](images/2e8811c5f5213285daba2037984cb87ad5fbab0c813f8d92269a3fb6892fc215.jpg)  
Fig. 3: Single-view distortion estimation and camera grouping. (1) Frames are evenly sampled from every camera of a scene. (2) AnyCalib [35] predicts a dense per-pixel ray and field-of-view map and fits a pinhole baseline, a radial Brown–Conrady model, and the Unified Camera Model (UCM), whose parameter ξ is used as a fisheye score; the estimates are used for distortion analysis and grouping only, and the dataset calibration is kept fixed for bird’s-eye-view (BEV) projection. (3) Cameras with ξ below the threshold become reference views (blue), and the rest become fisheye candidates (red), whose reprojection displacements grow toward the image borders.

confidence $s \in [ 0 , 1 ]$ , class $k \in \mathcal { H }$ , 17 COCO [16] keypoints $\{ ( \mathbf { u } _ { j } , s _ { j } ) \} _ { j = 0 } ^ { 1 6 }$ for humanoid classes, a part-based appearance descriptor $\mathbf { F } \in \mathbb { R } ^ { 6 \times 5 1 2 }$ whose rows $\mathbf { f } _ { p }$ are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ per body part, and per-part visibility $\nu \in [ 0 , 1 ] ^ { 6 }$

Ground-plane projection. A ground point has $Z = 0$ , so the third column of the world-to-image matrix $\mathbf { P } _ { c } = \mathbf { K } _ { c } \mathbf { E } _ { c } \in \mathbb { R } ^ { 3 \times 4 }$ drops out of the projection. For $\bar { \mathbf { P } } _ { c } \in \mathbb { R } ^ { 3 \times 3 }$ formed by columns 1, 2, and 4 of $\mathbf { P } _ { c } ,$ the ground-plane projection is $\lambda \tilde { \mathbf { u } } = \bar { \mathbf { P } } _ { c } ( X , Y , 1 ) ^ { \top }$ at projective depth $\lambda \neq 0$ , inverted by the homography $\mathbf { H } _ { c } \triangleq \bar { \mathbf { P } } _ { c } ^ { - 1 } \colon$ (1)

$$
\begin{array} { r } { \mathbf { q } \ = \ \mathbf { H } _ { c } \tilde { \mathbf { u } } , \ \mathbf { \Pi } _ { \mathcal { ( \pi ) } } \mathbf { \pi } _ { c } ( \mathbf { u } ) \ = \ \left( \frac { q _ { 1 } } { q _ { 3 } } , \ \frac { q _ { 2 } } { q _ { 3 } } \right) ^ { \top } , } \end{array}\tag{1}
$$

with homogeneous $\mathbf { q } = ( q _ { 1 } , q _ { 2 } , q _ { 3 } ) ^ { \top }$ and ground position $\pi _ { c }$ . We reject backprojections with $| q _ { 3 } | < \varepsilon _ { \mathrm { h } }$ , where $\varepsilon _ { \mathrm { h } } = 1 0 ^ { - 9 }$ (the pixel lies on or beyond the horizon, where its viewing ray never meets the ground), or with $\| \pi _ { c } ( \mathbf { u } ) \| _ { 2 } >$ $1 0 ^ { 3 }$ m, which bounds the world extent and discards numerically degenerate rays.

## 3.1 Distortion-Aware Camera Grouping

The dataset calibration [32] is pinhole-only: intrinsics, extrinsics, and a groundplane homography, but no distortion coeficients. With noticeable lens distortion, projected foot points land at the wrong BEV positions, and cross-view association sufers. This step identifies the ofending cameras and pulls the projected locations back (Fig. 3). We use AnyCalib [35] only for distortion analysis, grouping, and initialization; the dataset pinhole calibration remains the reference geometry for BEV projection.

Single-View Distortion Estimation (steps 1–2 in Fig. 3): We run pretrained AnyCalib on 31 evenly sampled frames per camera and take the per-parameter median. It predicts a dense pixel-to-ray field and fits a pinhole baseline, a radial Brown–Conrady model [2], and the Unified Camera Model (UCM) [7,20]; we use the UCM parameter ξ as a fisheye score.

Camera Grouping (step 3): Cameras with $\xi \le 0 . 3$ are reference cameras; the rest are fisheye candidates. This distortion-based split (not a physical lens label) agrees with the estimated radial coeficients and is confirmed by reprojection quiver plots: displacements stay small for reference cameras and grow toward the borders for fisheye candidates.

## 3.2 2D Object Detection

Because noisy 2D annotations can slow convergence, we classify each category as static, fixed-shape, or dynamic-shape from its per category dimension statistics, providing fallback shapes when matching fails, and remove redundant, tiny, unseen, or invalid boxes. Training scenes are matched to each test scenario, including the otherwise absent ‘AgilityDigit’ and ‘FourierGR1T2’ categories, with real scenes drawn from AI City [31] and MTMMC [44]. We train the RF-DETR 2x-large detector [25] for 200 epochs at 1920 pixels (batch size 4). Inference uses the same resolution with NMS at IoU 0.5 and a 0.1 confidence threshold. Further details are in the supplementary material (Sec. A).

## 3.3 Single-View Re-Identification and Tracking

For each detected person, we estimate body keypoints and use them to extract pose-aware descriptors for identity association. Within each camera, motion and these appearance cues then link detections over time into local tracklets, before any cross-view reasoning.

Pose Estimation. We apply pretrained ViTPose++ [46] to each detection crop. Its transformer encoder and lightweight decoder provide keypoints under changes in pose, scale, and occlusion.

Re-Identification. Keypoint Promptable Re-Identification (KPR) [28] uses these keypoints (positive only) to split the body into six regions and extract region-level descriptors, reducing background interference and occlusion sensitivity. As identities are scene-specific in AI City 2026 Track 1, we merge and relabel the train/val splits into 155 identities and train KPR for 110 epochs (≈3 h 7 min). These descriptors feed both the single-view tracker below and the multi-view associator of Sec. 3.4; further details are given in Sec. B.

Single-View Tracking. Following tracking-by-detection [1], we pair an appearanceaugmented IoU tracker with a constant-velocity Kalman filter [11]: box overlap and predicted motion supply the association afinity, and the part-based reidentification features disambiguate crossings and brief occlusions. Each surviving tracklet gets a local identity ℓ, and per camera and frame the stage emits instances $( \ell , k , \mathbf { b } , s , \{ ( \mathbf { u } _ { j } , s _ { j } ) \} , \mathbf { F } , \nu )$ . The global associator of Sec. 3.4 consumes these local identities as a carry-forward cue, since single-camera identity proved more reliable than cross-frame geometric matching alone.

## 3.4 Multi-View Tracking

Given the per-camera tracklets, multi-view tracking fuses observations of the same object across cameras into a globally consistent identity, in three steps: each detection is anchored to a ground-plane position by a class-adaptive rule, the resulting observations are merged across views per frame, and the merged world observations are bound to persistent global tracks.

3.4.1 Class-Adaptive Ground-Contact Anchoring.Back-projection through $\mathbf { H } _ { c }$ is exact only for a pixel that genuinely lies on the floor, so the accuracy of the entire bird’s-eye-view (BEV) representation reduces to choosing that pixel. A single rule cannot serve all seven classes: a person’s feet are visible and semantically well defined, a low, flat robot has no meaningful feet, and a tall vehicle seen from a steep overhead angle has its base occluded by its own body. We therefore select an anchoring strategy per class through a configurable map, with three strategies.

Skeleton. For humanoid classes with a reliable pose, the anchor is the midpoint of the two ankle keypoints (COCO indices 15 and 16) whenever at least one clears the ankle-confidence threshold $\theta _ { \mathrm { a n k } }$ . When neither does, we extrapolate down the leg’s kinematic chain as $\mathbf { u } _ { \mathrm { a n k l e } } = \mathbf { u } _ { \mathrm { k n e e } } + \rho \big ( \mathbf { u } _ { \mathrm { k n e e } } - \mathbf { u } _ { \mathrm { h i p } } \big )$ with $\rho = 1$ where $\mathbf { u } _ { \mathrm { h i p } }$ and $\mathbf { u } _ { \mathrm { k n e e } }$ are the hip and knee keypoints of one leg and $\rho$ is the shank-to-thigh length ratio, taken as unity for a straight leg; the anchor is the mean over whichever legs admit the construction. This extrapolation fires only when more than $n _ { \mathrm { k p } }$ keypoints clear the pose-confidence threshold $\theta _ { \mathrm { p o s e } }$ , so that a fragmentary skeleton never fabricates a ground contact.

Center point. For low, flat robots whose bounding box is efectively their footprint, the anchor is the box center $\left( b _ { x } W _ { \mathrm { i m g } } , \ b _ { y } H _ { \mathrm { i m g } } \right)$ , with $W _ { \mathrm { i m g } }$ and $H _ { \mathrm { i m g } }$ the image width and height in pixels.

Top–bottom. For tall or self-occluding objects the anchor depends on where the object sits in the frame. A box in the upper image half is viewed nearhorizontally, and its bottom edge is a trustworthy ground contact, so the anchor is the bottom-center $\left( b _ { x } W _ { \mathrm { i m g } } , \ ( b _ { y } + b _ { h } / 2 ) H _ { \mathrm { i m g } } \right)$ . A box in the lower image half is close to the camera under a steep view, where the base is routinely occluded or clipped while the top of the object stays crisp; there we anchor on the reliable top edge and descend by the object’s projected height. Let $\tilde { \mathbf { X } } _ { h } = ( X , Y , h , 1 ) ^ { \top }$ be the world point at height h above the ground point $( X , Y )$ , and let $\mathbf { p } _ { c } ^ { ( 2 ) }$ and $\mathbf { p } _ { c } ^ { ( 3 ) }$ denote the second and third rows of $\mathbf { P } _ { c }$ . The image row of that point, and the apparent pixel height of a vertical segment of length $h ,$ , are

$$
v ( h ) \ = \ \frac { { \bf p } _ { c } ^ { ( 2 ) } \tilde { \bf X } _ { h } } { { \bf p } _ { c } ^ { ( 3 ) } \tilde { \bf X } _ { h } } , \qquad \varDelta v _ { c } ( X , Y , h ) \ = \ \big | \ v ( 0 ) - v ( h ) \ \big | ,\tag{2}
$$

where $v ( h )$ is the image row of the top of that segment standing at $( X , Y )$ and $\varDelta v _ { c }$ its apparent pixel height. Seeding the footprint at the bottom-center backprojection and taking $h = H _ { k }$ , the constant world height prior of class k (defined with the other class extents in Sec. 3.5), the anchor row becomes

$$
v _ { \mathrm { a n c h o r } } ~ = ~ \operatorname* { m a x } \Bigl ( ~ ( b _ { y } - b _ { h } / 2 ) H _ { \mathrm { i m g } } ~ + ~ \varDelta v _ { c } \bigl ( \pi _ { c } ( \mathbf { u _ { b o t } } ) , H _ { k } \bigr ) , ~ ( b _ { y } + b _ { h } / 2 ) H _ { \mathrm { i m g } } \Bigr ) ,
$$

where $\mathbf { u } _ { \mathrm { b o t } }$ is the box bottom-center pixel and $H _ { k }$ the class height prior of $\operatorname { E q }$ <sup>(3)</sup><sub>.</sub> <sub>(6)</sub> in Sec. 3.5. The max enforces that the anchor never rises above the visible box bottom, while permitting it to fall below that edge, exactly the intended correction when the true base is clipped out of the box. Every anchor pixel is back-projected with $\mathbf { H } _ { c }$ through Eq. (1) and discarded if it is degenerate or falls outside the camera’s ground-coverage zone. This third strategy is powerful but circular: it derives the ground contact from the class height prior. The estimator in Sec. 3.5 breaks that circularity for persons by estimating height from a ground point constructed independently of it.

3.4.2 Cross-View Association on the Ground PlaneAt each frame, projected observations $o _ { i } \ : = \ : ( c _ { i } , k _ { i } , \ell _ { i } , \mathbf { x } _ { i } , s _ { i } , \mathbf { F } _ { i } , \pmb { \nu } _ { i } )$ , encoding camera, class, local identity, ground position, confidence, descriptor, and part visibility, are associated within each class before temporal tracking. Appearance is compared by a visibility-weighted cosine distance that ignores parts occluded in either view:

$$
d _ { \mathrm { a p p } } ( o _ { a } , o _ { b } ) = \sum _ { p = 1 } ^ { 6 } \nu _ { a , p } \nu _ { b , p } \left( 1 - \langle { \bf f } _ { a , p } , { \bf f } _ { b , p } \rangle \right) \Bigg / \sum _ { p = 1 } ^ { 6 } \nu _ { a , p } \nu _ { b , p } ,\tag{4}
$$

where p indexes six body parts; since the descriptor rows are unit-norm, $d _ { \mathrm { a p p } } \in$ $[ 0 , 2 ]$ , while no co-visible part gives +∞. Observations from diferent cameras are admissible when $\| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } < \theta _ { k } ^ { \mathrm { { b e v } } }$ , a per-class BEV merge radius, and, for classes in $\mathcal { H } _ { \mathrm { r e i d } }$ (the classes with part-based appearance descriptors), $d _ { \mathrm { a p p } } ( o _ { i } , o _ { j } ) < \theta _ { \mathrm { g r p } } ,$ an appearance grouping gate. Single-linkage clustering [27] over a union–find structure [33] processes pairs by increasing ground distance and rejects merges between clusters already containing the same camera. Each cluster $\mathcal { G } _ { \omega }$ becomes $\boldsymbol { \omega } = ( k _ { \omega } , \mathbf { x } _ { \omega } , \mathbf { F } _ { \omega } , \pmb { \nu } _ { \omega } , \mathcal { G } _ { \omega } )$ , with confidence-weighted position $\begin{array} { r } { \mathbf { x } _ { \omega } = \frac { \sum _ { m \in \mathcal { G } _ { \omega } } s _ { m } \mathbf { x } _ { m } } { \sum _ { m \in \mathcal { G } _ { \omega } } s _ { m } } } \end{array}$ (or an unweighted mean if the denominator is zero). Part descriptors are fused as $\begin{array} { r } { \mathbf { f } _ { \omega , p } = \frac { \sum _ { m \in \mathcal { G } _ { \omega } } \nu _ { m , p } \mathbf { f } _ { m , p } } { \| \sum _ { m \in \mathcal { G } _ { \omega } } \nu _ { m , p } \mathbf { f } _ { m , p } \| _ { 2 } } } \end{array}$ with $\begin{array} { r } { \nu _ { \omega , p } = \operatorname* { m a x } _ { m \in \mathcal { G } _ { \omega } } \nu _ { m , p } ; } \end{array}$ this combines complementary views without imputing occluded parts, and when no member sees part $p$ the fused weight is $\nu _ { \omega , p } = 0$ , so the part is ignored by Eq. (4).

3.4.3 Global Multi-View TrackingA global track τ holds a class $k _ { \tau } , \mathrm { a }$ ground position $\mathbf { x } _ { \tau }$ , a velocity ${ \bf v } _ { \tau } ~ ( \mathrm { m } \mathrm { f r a m e ^ { - 1 } } )$ , a member map $\mathcal { M } _ { \tau } : c \mapsto \ell$ of the local tracklets feeding it, an appearance gallery $\mathcal { A } _ { \tau }$ of up to $N _ { \mathrm { g a l } }$ recent descriptor– visibility pairs, and a missing age $a _ { \tau }$ of consecutive unmatched frames. Each frame’s world observations bind to tracks in two passes.

Pass A: carry-forward by local identity. Per observation $\omega$ and same-class track τ we count $\omega \mathrm { s }$ members $( c , \ell )$ already in $\mathcal { M } _ { \tau }$ and bind pairs greedily in descending count, each observation and track consumed once. This pass is the primary defense against identity switches: a track survives while any one of its cameras holds it.

Pass B: gated assignment for the remainder. Whatever Pass A leaves is resolved per class. The prediction $\hat { \mathbf { x } } _ { \tau } = \mathbf { x } _ { \tau } + \mathbf { v } _ { \tau }$ gives $d _ { \mathrm { b e v } } ( \omega , \tau ) = \lVert \mathbf { x } _ { \omega } - \hat { \mathbf { x } } _ { \tau } \rVert _ { 2 } ,$ gated by $\begin{array} { r } { g ( \tau ) = \theta _ { k } ^ { \mathrm { b e v } } + \frac { V _ { \mathrm { m a x } } } { f _ { \mathrm { r } } } a _ { \tau } , } \end{array}$ , exactly the distance coverable while unobserved at maximum plausible speed $V _ { \mathrm { m a x } } ~ ( \mathrm { m } \mathrm { s } ^ { - 1 } )$ and frame rate $f _ { \mathrm { r } } ,$ so a track survives long occlusions without an indiscriminate gate. Appearance uses the nearest gallery distance $\begin{array} { r } { d _ { \mathrm { g a l } } ( \omega , \tau ) = \operatorname* { m i n } _ { ( \mathbf { F } ^ { \prime } , \nu ^ { \prime } ) \in \mathcal { A } _ { \tau } } d _ { \mathrm { a p p } } \big ( ( \mathbf { F } _ { \omega } , \nu _ { \omega } ) , ( \mathbf { F } ^ { \prime } , \nu ^ { \prime } ) \big ) } \end{array}$ , where $d _ { \mathrm { a p p } }$ of Eq. (4) depends on its arguments only through the descriptor–visibility pair and is +∞ absent a gallery or co-visible part; the minimum, not the mean, lets one confident past view re-identify across pose or illumination change. Pairs are gated out by $d _ { \mathrm { b e v } } > g ( \tau )$ , or by $k \in \mathcal { H } _ { \mathrm { r e i d } } \land \theta _ { \mathrm { r e i d } } < d _ { \mathrm { g a l } } < \infty$ for appearance gate $\theta _ { \mathrm { r e i d } }$ . Survivors cost $\begin{array} { r } { \mathbf { C } [ \omega , \tau ] = \beta \frac { d _ { \mathrm { b e v } } ( \omega , \tau ) } { q ( \tau ) } + ( 1 - \beta ) \bar { d } _ { \mathrm { g a l } } ( \omega , \tau ) } \end{array}$ for $k \in \mathcal { K } _ { \mathrm { r e i d } }$ and $\mathbf { C } [ \omega , \tau ] = d _ { \mathrm { b e v } } ( \omega , \tau )$ otherwise, with $\bar { d } _ { \mathrm { g a l } } = d _ { \mathrm { g a l } }$ when finite and a neutral 1 otherwise. Normalizing by $g ( \tau )$ puts geometry on a bounded scale comparable to the appearance term’s [0, 2] range, so a single weight $\beta \in [ 0 , 1 ]$ balances the two across classes with very diferent gate radii. The Hungarian algorithm [12] minimizes $\textstyle \sum \mathbf { C }$ exactly; ∞-sentinel pairs are discarded after the assignment is solved.

Table 1: The synthetic-to-real configuration surface: parameter settings used in the synthetic and real regimes.
<table><tr><td>Parameter</td><td>Synthetic regime</td><td>Real regime</td></tr><tr><td>Cameras per scene</td><td>10-20</td><td>4-7</td></tr><tr><td>Intrinsics</td><td>ideal pinhole, cloned per-camera,</td><td> $f _ { x } \neq f _ { y }$ </td></tr><tr><td>Person ground anchor</td><td>bbox + height pole ankle skeleton</td><td></td></tr><tr><td>Ankle confidence  $\theta _ { \mathrm { a n k } }$ </td><td>0.5</td><td>0.8</td></tr><tr><td>Skeleton gate  $n _ { \mathrm { k p } }$ </td><td>3</td><td>8</td></tr><tr><td>Position smoothing α</td><td>1.0 (none)</td><td>0.6</td></tr><tr><td>Appearance gates  $\theta _ { \mathrm { g r p } } , \theta _ { \mathrm { r e i d } } > 2$  Object cardinality  $N _ { k }$ </td><td>(inactive) known and enforced unknown; adaptive</td><td>0.8, 1.6 (active)</td></tr></table>

Trajectory Lifecycle Dynamics. An unmatched observation spawns a track only when min $\tau { : } k _ { \tau } { = } k _ { \omega } \ \| \mathbf { x } _ { \omega } - \mathbf { x } _ { \tau } \| _ { 2 } \geq \delta _ { \mathrm { n e w } }$ , the minimum birth separation (m); this blocks the rebirth of a momentarily unmatched track beside itself under a fresh identity. An unmatched track instead dead-reckons, $\mathbf { x } _ { \tau }  \mathbf { x } _ { \tau } + \mathbf { v } _ { \tau } , a _ { \tau } $ $a _ { \tau } + 1$ , and retires once $a _ { \tau } > A _ { \mathrm { m a x } } .$ A matched track takes $\begin{array} { r } { \mathbf { x } _ { \tau } ^ { + } = \alpha \mathbf { x } _ { \omega } + ( 1 - \alpha ) \mathbf { x } _ { \tau } } \end{array}$ for smoothing factor $\alpha \in ( 0 , 1 ]$ and $\begin{array} { r } { \mathbf { v } _ { \tau } = \frac { \mathbf { x } _ { \tau } ^ { + } - \mathbf { x } _ { \tau } } { \operatorname* { m a x } ( 1 , { t - t _ { \tau } ^ { \mathrm { l a s t } } } ) } } \end{array}$ from its previous sighting frame $t _ { \tau } ^ { \mathrm { l a s t } }$ , then $\mathbf { x } _ { \tau } \gets \mathbf { x } _ { \tau } ^ { + } ;$ ; its members refresh $\mathcal { M } _ { \tau }$ , its fused descriptor enters $\mathcal { A } _ { \tau }$ , and $a _ { \tau }$ resets to zero.

A closed-world cardinality prior. In a controlled warehouse the per-class count $N _ { k }$ is often known a priori: trivially in simulation, by inspection for a staged capture. A fixed-cardinality variant then constrains each class’s live track set $\mathcal { T } _ { k }$

$$
| { \mathcal { T } } _ { k } | \leq N _ { k } \forall k , \qquad N _ { k } = \infty { \mathrm { i f ~ } } k { \mathrm { ~ i s ~ u n c a p p e d , } }\tag{5}
$$

under which capped tracks coast rather than retire, so the count converges to exactly $N _ { k } ;$ contested slots go to the most confident observations, and a stale slot is reclaimed past a missing-age threshold. Association thus becomes assigning a fixed identity set to the frame’s evidence rather than managing tracks from detections. This regime is sharpest with the Pass-B gates dropped: the solve force-assigns every track to its best observation, and a track coasts only when the frame yields fewer observations than its class has tracks. The prior is exact in simulation, where the census is a fact of the scene description, and precisely what a real deployment cannot assume; the real scene therefore runs the adaptive tracker with all gates active.

Finalization. In an ofline post-processing pass after the last frame (the only acausal step, applied once per sequence before submission), each track’s sightings become a per-frame position map; gaps under $G _ { \mathrm { m a x } }$ frames are linearly interpolated between bracketing sightings, and longer ones are left unfilled instead of being bridged through space the object may not have occupied.

3.4.4 The Synthetic-to-Real Configuration SurfaceThe domain gap shifts the relative reliability of geometry and appearance (Tab. 1): synthetic scenes (exact calibration, broad overlap, known counts) let geometry and the closed-world prior sufice, while real scenes (estimated calibration, sparse overlap, unknown counts) need smoothing and active appearance gating (toggled by the cosine threshold of Eq. (4)). The pipeline thus adapts by shifting trust between geometry and appearance, not by retraining the feature extractor.

## 3.5 Height and Yaw Angle Estimation

Each track position (X, Y) at frame t is lifted to an axis-parameterized 3D box $( X , Y , Z , w , l , h , \psi )$ . For yaw ψ, we fit a second-degree polynomial to the previous 45 track positions, diferentiate at the current point, and take the arctangent. The baseline lift assigns constant class extents,

$$
( w , l , h ) \ = \ ( W _ { k } , L _ { k } , H _ { k } ) , \qquad Z \ = \ { \frac { h } { 2 } } ,\tag{6}
$$

where $\left( W _ { k } , L _ { k } , H _ { k } \right)$ are class k’s width, length, and height priors, the per-class medians of the previous year’s warehouse ground-truth 3D box scales (the one class lacking a counterpart there is given a manual estimate), and $Z = h / 2$ places a floor-standing object’s centroid at half its height. Imported wholesale, this synthetic prior is the second gap point: the shape prior.

Monocular metric person height. For persons this prior becomes a perframe calibration estimate. To break the circularity of the H -dependent anchor in Eq. (3), we form an independent ground point $\tilde { \mathbf { X } } _ { 0 } = ( X , Y , 0 , 1 ) ^ { \top }$ from the ankle midpoint; with $a _ { v } = \mathbf { p } _ { c } ^ { ( 2 ) } \tilde { \mathbf { X } } _ { 0 } , b _ { v } = ( \mathbf { P } _ { c } ) _ { 2 3 } , \gamma _ { v } = \mathbf { p } _ { c } ^ { ( 3 ) } \tilde { \mathbf { X } } _ { 0 }$ , and $d _ { v } = ( \mathbf { P } _ { c } ) _ { 3 3 }$ $v ( h ) = ( a _ { v } + b _ { v } h ) / ( \gamma _ { v } + d _ { v } h )$ , and matching the detection’s top row $v _ { \mathrm { t o p } }$ yields

$$
\hat { h } \ = \ \frac { v _ { \mathrm { t o p } } \gamma _ { v } \ - \ a _ { v } } { b _ { v } \ - \ v _ { \mathrm { t o p } } d _ { v } } ,\tag{7}
$$

which inverts Eq. (2) without learned depth or 3D ground truth. A sample is accepted only if the detection is untruncated, its confidence exceeds the trackconfidence threshold $\theta _ { \mathrm { t r k } }$ , both ankles exceed $\theta _ { \mathrm { a n k } } ,$ a head keypoint exceeds $\theta _ { \mathrm { p o s e } }$ the back-projection is valid and in-zone, and $\hat { h } \in [ h _ { \operatorname* { m i n } } , h _ { \operatorname* { m a x } } ] ;$ failing samples are rejected, not clamped, which leaves a small positive bias from ankle elevation. Track $\tau$ takes the cross-camera median $\hat { h } _ { \tau , t } \ = \ \operatorname { m e d i a n } \{ \hat { h } _ { m } \ | \ m \ \in$ $\mathcal { G } _ { \omega } , \ \hat { h } _ { m }$ accepted} over its members $\mathcal { G } _ { \omega }$ , else $H _ { k }$ with no cross-frame propagation. The box then scales as $\begin{array} { r } { ( w , l , h , Z ) \ = \ \big ( \frac { W _ { k } } { H _ { k } } \hat { h } _ { \tau , t } , \frac { L _ { k } } { H _ { k } } \hat { h } _ { \tau , t } , \hat { h } _ { \tau , t } , \frac { \hat { h } _ { \tau , t } } { 2 } \big ) } \end{array}$ , which preserves the class aspect ratio.

## 3.6 Point-Cloud-Guided 3D Box Refinement

After multi-view tracking, each object has a stable identity and size but a coarse center, obtained by projecting a 2D image point to the floor via intrinsics $\mathbf { K } _ { c }$ and extrinsics $\mathbf { E } _ { c }$ . This projection fails when the point is not a true ground contact (elevated Transporters, occluded feet, distortion, calibration residuals), producing planar position and yaw errors that lower 3D IoU despite correct association. Since Track 1 withholds depth at inference [32], we refine footprint and yaw from an RGB-estimated point cloud, preserving identity and size; Fig. 1 shows typical pre-refinement misalignments.

DA3-Based Metric Point Cloud Construction. We run Depth Anything 3 (DA3) [15] in a pose-conditioned multi-view setting with the Nested Giant-Large 1.1 checkpoint, which couples any-view geometry with metric-depth scaling. Per synchronized frame, all camera images are processed with the provided $\mathbf { K } _ { c }$ and $\mathbf { E } _ { c } ,$ , and predicted depths are back-projected into a common world frame and fused into a colored point cloud (Fig. 4; details in Sec. D).

![](images/b83b54ed09f6b35106b0d69b3392403a3cc62922a7d5efb7a0b400dceba3ffc0.jpg)

(a) Synthetic scene  
![](images/b6abc08f3ea49a9e6a97120863bc180614697e9d68e40351a3a5e61928429047.jpg)  
(b) Real-world scene  
Fig. 4: Multi-view RGB inputs and DA3-estimated point clouds for (a) the synthetic scene Warehouse 025 and (b) the real-world scene Warehouse 027 at frame 0.

Synthetic Scene Refinement. Reliable calibration and controlled categories keep the class-size priors of Eq. (6) valid, so DA3 refines only footprint and yaw. For fixed-shape objects, we crop points around the tracked box, project them to a BEV density grid, drop floor, background, and weak components, and fit the prior-sized footprint near the tracked pose, reverting to the tracker output when support is weak (Fig. 5). For dynamic-shape objects (persons, humanoid robots), the tracker center is steadier than the sparse articulated cloud, so DA3 corrects only standing yaw or one-frame jumps and keeps the prior (Fig. 1(a)).

Real-World Scene Refinement. Moving targets are mainly persons, whose centers are sensitive to distorted rays, calibration residuals, and occluded feet; parked vehicles keep fixed hand-verified poses. After removing static background via temporal depth statistics, each person track’s BEV window is resolved to its densest mode by seeded coarse-to-fine mean-shift [3]: the tracker seeds the first frame, later frames reuse the refined state, and a causal temporal state bridges short losses and suppresses jitter. DA3 corrects the center while keeping the class-size prior (Fig. 1(b)).

## 3.7 Filtering Predictions Based on the BEV Map

A per-class cap (Eq. (5)) or weak detections can emit a box no camera genuinely supports: a phantom that inflates false positives. We test every emitted box against the cameras whose ground-plane coverage includes it, via per-camera visibility maps on the BEV floor map. The filter runs per frame on the emitted box set only and never alters a box’s identity, size, or pose; it proceeds in three steps:

![](images/44cc038f40ff24233f9b572c6e4cb30866493ee1b83434d2e818a915e85600c1.jpg)

Fig. 5: Local fixed-shape object refinement pipeline for the synthetic scene. Starting from a colored DA3 .ply crop around a tracked object, we project the crop to a BEV density grid, keep densityexcess cells after floor and ghost filtering, and fit a fixed-size box to the remaining support. The red dashed box is the multi-view tracking prior, and the green box is the refined result.  
![](images/3e1f9df90bab502f511a64928ef43bcd7e99296c8475a1de3a88e8449e6fc756.jpg)  
(a) BEV floor map: cameras and fisheye zones (b) BEV consistency: one matched object  
Fig. 6: BEV use of camera grouping. (a) Floor map of reference (blue) and fisheye (red) cameras; dashed violet marks fisheye-only zones, where objects retain their fisheye BEV positions. (b) For a matched object, the reference view anchors the fused target, and each fisheye projection’s error is measured in ground-plane meters.

Definition of Camera Visibility Maps: For each camera c we record the ground region it observes: its projected field of view intersected with the handannotated coverage polygons of the BEV floor map (Fig. 6), which also mark the fisheye-only zones. Rasterizing over all cameras gives, per BEV cell, the set of cameras that see it: a visibility map $\mathcal { V } ( \mathbf { x } )$ indexed by ground position x.

Object Verification and Filtering: Each emitted box at $\mathbf { x } _ { \tau }$ gets covering cameras $\mathcal { C } ( \tau , t ) = \mathcal { V } ( \mathbf { x } _ { \tau } )$ ; when this set is empty, all cameras the box geometrically projects onto are used instead. A person box is re-projected into these views and kept only if some re-extracted crop shows a visible ankle,

$\ker ( \tau , t ) \iff \mathcal { C } ( \tau , t ) \neq \emptyset \land \operatorname* { m a x } _ { c \in \mathcal { C } ( \tau , t ) } \operatorname* { m a x } _ { j \in \{ 1 5 , 1 6 \} } s _ { c , j } > \theta _ { \mathrm { v i s } }$ (kτ a person class), where $s _ { c , j }$ is the score of ankle keypoint j in camera c and $\theta _ { \mathrm { v i s } }$ (8)<sup>the</sup> <sup>visibility</sup> threshold. Unconfirmed person boxes are removed, and boxes of all other classes pass unchanged; this causal per-frame check suppresses forced-assignment phantoms while leaving every surviving box untouched.

Fisheye BEV Consistency (Fig. 6): Some ground regions are seen only by fisheye cameras and have no reference anchor; these blind spots are manually annotated as BEV polygons (dashed violet in Fig. 6(a)), and objects inside keep their fisheye-detection BEV position instead of being refined toward an unsupported target. Where both groups overlap (Fig. 6(b)), matched observations are projected to BEV and the fisheye-versus-reference disagreement is measured in meters, the error BEV tracking consumes; a foot point is then corrected by undistorting it before floor projection, refining only the distortion while intrinsics and extrinsics stay fixed.

Table 2: Leaderboard of Multi-Camera 3D Perception, AI City Challenge 2026 Track 1, ranked by 3D HOTA (%). Our entry (shaded, bold) ranks second.
<table><tr><td>Rank ID</td><td></td><td>Name 3D HOTA</td><td>(%) DetA</td><td>(%)</td><td>AssA</td><td>(%) LocA (%)</td></tr><tr><td>1</td><td>289</td><td>EVA</td><td>56.5447</td><td>55.6444</td><td>49.3929</td><td>79.5558</td></tr><tr><td>2</td><td></td><td>34 SKKU-AL-T1</td><td>52.0118</td><td>45.3056</td><td>56.5047</td><td>76.2410</td></tr><tr><td>3</td><td>130</td><td>Playbox</td><td>38.0105</td><td>40.2592</td><td>31.0978</td><td>75.1778</td></tr><tr><td>4</td><td>4</td><td>QDTers</td><td>34.1845</td><td>29.4663</td><td>33.1122</td><td>17.6841</td></tr><tr><td>5</td><td>149</td><td>Calix</td><td>33.7654</td><td>30.8748</td><td>30.9816</td><td>50.4720</td></tr></table>

Table 3: Impact of the detection backbone and training resolution on the Track 1 evaluation server. The shaded row is the setting adopted in Sec. 3.2.  
Table 4: Impact of the association cues used for single-view tracking on the Track 1 evaluation server. The shaded row is the setting adopted in Sec. 3.3.

<table><tr><td>Model</td><td colspan="3">Train Size Infer Size HOTA (%)</td></tr><tr><td rowspan="2">YOLOv26</td><td>1600</td><td>1920</td><td>50.5285</td></tr><tr><td>1920</td><td>1920</td><td>50.7207</td></tr><tr><td rowspan="2">RF-DETR</td><td>1600</td><td>1920</td><td>50.5913</td></tr><tr><td>1920</td><td>1920</td><td>50.8078</td></tr></table>

<table><tr><td colspan="4">Re-ID Pose gIoU dIoU HOTA (%)</td></tr><tr><td>X</td><td>X X</td><td>X</td><td>50.8078</td></tr><tr><td></td><td></td><td>X</td><td>51.1145</td></tr><tr><td>X</td><td>X X</td><td></td><td>50.9577</td></tr><tr><td>X</td><td>X X</td><td></td><td>50.8204</td></tr><tr><td></td><td></td><td></td><td>51.2906</td></tr><tr><td>X</td><td></td><td></td><td>51.3205</td></tr><tr><td></td><td></td><td></td><td>51.8568</td></tr></table>

## 4 Experiments & Discussion

## 4.1 Dataset and Evaluation Metrics

MTMC Tracking 2026 [32] provides synchronized 1080p/30-fps RGB videos from calibrated cameras, top-down maps, and 2D/3D annotations. Under its RGBonly protocol, we evaluate Syn2RealTrack on expanded synthetic and hidden real scenes using HOTA [18], DetA, AssA, and LocA (higher is better).

## 4.2 Quantitative Results

Tab. 2 reports the oficial Track 1 leaderboard, ranked by 3D HOTA. Our team, SKKU-AL-T1, ranks second at 52.0118%, trailing first-place EVA by 4.53 points but leading third place by 14.00. The field thus splits into two tiers: our gap to the rest is roughly three times our gap to the top.

## 4.3 Ablation Study

Since hidden-test annotations are unavailable, each ablation is a single evaluationserver submission, comparable with Tab. 2. Sequential configurations with minor scene-specific tuning give stage-wise, not independent, efects; we keep fourdecimal precision to expose small diferences.

2D Detection Backbone and Input Resolution. This experiment selects the detection backbone and training resolution. At an inference resolution of 1920, Tab. 3 shows RF-DETR beating YOLOv26 at both training resolutions; matching the training to the inference resolution adds 0.19 and 0.22 HOTA, motivating the 1920/1920 setting (Sec. 3.2). The backbones difer by only 0.09 HOTA, so the detector choice is not decisive.

Single-View Association Cues. Tab. 4 isolates the single-view afinity terms of Sec. 3.3. Over the 50.8078 baseline, generalized IoU (gIoU) [24] and distance IoU (dIoU) [48] add +0.31 and +0.15 HOTA alone, but combined without appearance reach only 50.8204. Re-identification or pose guidance raises HOTA to

Table 5: Impact of monocular metric person height estimation (Eq. (7)); the shaded row is the setting adopted in Sec. 3.5.  
Table 6: Impact of BEV-map-based prediction filtering (Sec. 3.7).
<table><tr><td>Person height source</td><td>HOTA (%)</td></tr><tr><td>Constant class prior (Eq. (6))</td><td>51.8568</td></tr><tr><td>+ monocular estimate, Warehouse 023 &amp; 024</td><td>51.9546</td></tr><tr><td>+ monocular estimate, Warehouse 025</td><td>51.9898</td></tr></table>

<table><tr><td>Filtering stage</td><td>HOTA (%)</td></tr><tr><td>No filtering</td><td>51.8813</td></tr><tr><td>+ visibility-zone &amp; static-object filtering</td><td>52.0100</td></tr><tr><td>+ 3D-box coverage verification</td><td>52.0118</td></tr></table>

Table 7: Stage-wise ablation of the full pipeline on the Track 1 evaluation server. Each row adds one component to the row above it; the shaded row 6 is the submitted system of Tab. 2.
<table><tr><td colspan="6"># Configuration HOTA (%) DetA (%) AssA (%) LocA (%) ∆ HOTA</td></tr><tr><td>1 RF-DETR detector only</td><td>50.8078</td><td>42.4111</td><td>56.2599</td><td>74.1915</td><td></td></tr><tr><td>2 + gIoU association</td><td>51.1145</td><td>43.6031</td><td>55.8617</td><td>75.0459</td><td>+0.31</td></tr><tr><td>3 + all association cues</td><td>51.8568</td><td>44.7921</td><td>56.4608</td><td>76.2410</td><td>+0.74</td></tr><tr><td>4 + cross-view ground-plane clustering</td><td>51.9205</td><td>45.1611</td><td>56.4601</td><td>76.2053</td><td>+0.06</td></tr><tr><td>5 + monocular person height estimation</td><td>51.9898</td><td>45.0712</td><td>56.5190</td><td>76.2397</td><td>+0.07</td></tr><tr><td>6 + BEV-map prediction filtering</td><td>52.0118</td><td>45.3056</td><td>56.5047</td><td>76.2410</td><td>+0.02</td></tr></table>

51.2906 and 51.3205; both together give the best 51.8568 (+1.05 over baseline), the final configuration.

Monocular Person Height Estimation. Tab. 5 replaces the constant prior in Eq. (6) with the per-frame estimate of Eq. (7), enabled scene by scene. HOTA improves monotonically, by +0.13 in total over the constant-prior baseline: calibration alone beats the synthetic prior without depth or supervision. The estimator adjusts box height and centroid, not identity, so association is essentially unafected. A residual scene-dependent ground-plane error remains, which the point-cloud refinement of Sec. 3.6 targets.

BEV-Map Prediction Filtering. Tab. 6 evaluates the false-positive filter of Sec. 3.7, which removes boxes induced by the closed-world prior of Eq. (5) at camera-unsupported locations. Disabling it costs 0.13 HOTA; in the cumulative decomposition of Tab. 7, the same stage adds +0.23 DetA while lowering AssA by only 0.01. This asymmetry supports the causal, per-frame test of Eq. (8), which suppresses phantoms without terminating identities held elsewhere. The final coverage check adds +0.0018 HOTA at negligible overhead.

Contribution of Each Pipeline Stage. The cumulative ablation in Tab. 7 raises HOTA from 50.8078 to 52.0118 (+1.20), of which single-view association supplies +1.05 (87.1%). The remaining stages add smaller, detection-driven gains: cross-view ground-plane clustering +0.06, monocular person height +0.07, and BEV-map filtering +0.02, the last through a +0.23 DetA gain. Across all stages DetA rises from 42.41 to 45.31 (+2.90), while AssA stays within 56.46– 56.52 once all association cues are enabled.

## 5 Conclusion

Syn2RealTrack addresses synthetic-to-real gaps in calibration, shape priors, and object census, achieving 52.0118% 3D HOTA and second place in AI City Challenge 2026 Track 1. It remains sensitive to calibration and RGB-derived depth, severe occlusion, frequent entry/exit, and closed-world census assumptions; future work will target uncertainty-aware calibration and depth, stronger temporal prediction, and domain adaptation.

## Acknowledgements

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2021-0-01364, An intelligent system for 24/7 real-time trafic surveillance on edge devices)

## References

1. Aharon, N., Orfaig, R., Bobrovsky, B.Z.: Bot-sort: Robust associations multipedestrian tracking. arXiv preprint arXiv:2206.14651 (2022)

2. Brown, D.C.: Decentering distortion of lenses. Photogrammetric Engineering 32(3), 444–462 (1966)

3. Comaniciu, D., Meer, P.: Mean shift: A robust approach toward feature space analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence 24(5), 603–619 (2002). https://doi.org/10.1109/34.1000236

4. Criminisi, A., Reid, I., Zisserman, A.: Single view metrology. International Journal of Computer Vision 40(2), 123–148 (2000). https://doi.org/10.1023/A: 1026598000963

5. Daryani, A.E., Bhutta, M.U.M., Hernandez, B., Medeiros, H.: CaMuViD: Calibration-free multi-view detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 1220–1229 (Jun 2025). https: //doi.org/10.1109/CVPR52734.2025.00122

6. Engilberge, M., Shi, H., Wang, Z., Fua, P.: Two-level data augmentation for calibrated multi-view detection. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 128–136 (Jan 2023). https://doi.org/10.1109/WACV56688.2023.00021

7. Geyer, C., Daniilidis, K.: A unifying theory for central panoramic systems and practical implications. In: Computer Vision – ECCV 2000. pp. 445–461 (2000)

8. Hou, Y., Zheng, L.: Multiview detection with shadow transformer (and viewcoherent data augmentation). In: Proceedings of the 29th ACM International Conference on Multimedia. pp. 1673–1682 (Oct 2021). https://doi.org/10.1145/ 3474085.3475310

9. Hou, Y., Zheng, L., Gould, S.: Multiview detection with feature perspective transformation. In: Computer Vision – ECCV 2020. pp. 1–18 (2020). https: //doi.org/10.1007/978-3-030-58571-6\_1

10. Huynh, N.D.M., Tran, D.N.N., Pham, L.H., Tran, T.H.P., Jeon, H.J., Nguyen, H.H., Khac Vu, D., Jeon, H.M., Phan, S.H., Pham-Nam Ho, Q., Tran, C.D., Khanh, T.L.B., Jeon, J.W.: TSBOW – trafic surveillance benchmark for occluded vehicles under various weather conditions. Proceedings of the AAAI Conference on Artificial Intelligence 40(7), 5239–5247 (2026). https://doi.org/10.1609/aaai.v40i7. 37439

11. Kalman, R.E.: A new approach to linear filtering and prediction problems. Journal of Basic Engineering 82(1), 35–45 (1960). https://doi.org/10.1115/1.3662552

12. Kuhn, H.W.: The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2(1–2), 83–97 (1955). https://doi.org/10.1002/nav. 3800020109

13. Lee, J., Kim, H., Lee, D., Lee, K.: Multi-camera 3D object tracking via 3D point clouds and re-identification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5417–5424 (Oct 2025). https://doi.org/10.1109/ICCVW69036.2025.00566

14. Lima, J.P., Roberto, R., Figueiredo, L., Simões, F., Teichrieb, V.: Generalizable multi-camera 3D pedestrian detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 1232–1240 (2021). https://doi.org/10.1109/CVPRW53098.2021.00135

15. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Zhao, Y., Peng, S., Guo, H., Zhou, X., Shi, G., Feng, J., Kang, B.: Depth Anything 3: Recovering the visual space from any views. In: International Conference on Learning Representations (ICLR) (2026), https://openreview.net/forum?id=yirunib8l8

16. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: Computer Vision – ECCV 2014. pp. 740–755 (2014). https://doi.org/10.1007/978-3-319-10602- 1\_48

17. Lu, H., Zhang, Y., Wang, G., Lian, Q., Du, D., Chen, Y.C.: Towards generalizable multi-camera 3D object detection via perspective rendering. Proceedings of the AAAI Conference on Artificial Intelligence 39(6), 5811–5819 (2025). https:// doi.org/10.1609/aaai.v39i6.32620

18. Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., Leibe, B.: HOTA: A higher order metric for evaluating multi-object tracking. International Journal of Computer Vision 129(2), 548–578 (2021). https://doi.org/10.1007/ s11263-020-01375-2

19. Ma, J., Wang, T., Liu, M., Ahmedt-Aristizabal, D., Nguyen, C.: DCHM: Depthconsistent human modeling for multiview detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7731–7740 (2025). https://doi.org/10.1109/ICCV51701.2025.00725

20. Mei, C., Rives, P.: Single view point omnidirectional camera calibration from planar grids. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 3945–3950 (Apr 2007). https://doi.org/10.1109/ROBOT.2007.364084

21. Ong, J., Vo, B.T., Vo, B.N., Kim, D.Y., Nordholm, S.: A Bayesian filter for multi-view 3D multi-object tracking with occlusion handling. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(5), 2246–2263 (2022). https: //doi.org/10.1109/TPAMI.2020.3034435

22. Pham, L.H., Ho, Q.P.N., Vu, D.K., Nguyen, H.H., Tran, C.D., Tran, D.N.N., Tran, T.H.P., Huynh, N.D.M., Jeon, H.J., Jeon, H.M., Phan, S.H., Le Ba Khanh, T., Jeon, J.W.: Data augmentation is all you need for robust fisheye object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5393–5401 (2025). https://doi.org/10.1109/ ICCVW69036.2025.00563

23. Phan, T.H., Dinh, D.D., Huynh, T.P., Le, Q.T., Dang, H.H., Tran, V.H., Luu, V.T., Huang, C.C.: VGCRTrack: Multi-camera 3D tracking with view-aware geometric center refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5434–5440 (Oct 2025). https://doi. org/10.1109/ICCVW69036.2025.00568

24. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized intersection over union: A metric and a loss for bounding box regression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 658–666 (2019). https://doi.org/10.1109/CVPR.2019.00075

25. Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., Peri, N.: RF-DETR: Neural architecture search for real-time detection transformers. In: International Conference on Learning Representations (ICLR) (2026)

26. Shi, S., Wang, X., Li, H.: PointRCNN: 3D object proposal generation and detection from point cloud. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–779 (2019). https://doi.org/ 10.1109/CVPR.2019.00086

27. Sibson, R.: SLINK: An optimally eficient algorithm for the single-link cluster method. The Computer Journal 16(1), 30–34 (1973). https://doi.org/10.1093/ comjnl/16.1.30

28. Somers, V., Alahi, A., De Vleeschouwer, C.: Keypoint promptable re-identification. In: Computer Vision – ECCV 2024. pp. 216–233 (2024). https://doi.org/10. 1007/978-3-031-72986-7\_13

29. Somers, V., De Vleeschouwer, C., Alahi, A.: Body part-based representation learning for occluded person re-identification. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1613–1623 (2023). https://doi.org/10.1109/WACV56688.2023.00166

30. Song, L., Wu, J., Yang, M., Zhang, Q., Li, Y., Yuan, J.: Stacked homography transformations for multi-view pedestrian detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 6049–6057 (Oct 2021). https://doi.org/10.1109/ICCV48922.2021.00599

31. Tang, Z., Wang, S., Anastasiu, D.C., Chang, M.C., Sharma, A., Kong, Q., Kobori, N., Gochoo, M., Batnasan, G., Otgonbold, M.E., Alnajjar, F., Hsieh, J.W., Kornuta, T., Li, X., Zhao, Y., Zhang, H., Radhakrishnan, S., Jain, A., Kumar, R., Murali, V.N., Wang, Y., Pusegaonkar, S.S., Wang, Y., Biswas, S., Wu, X., Zheng, Z., Chakraborty, P., Chellappa, R.: The 9th AI City Challenge. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5526–5535 (2025). https://doi.org/10.1109/ICCVW69036.2025.00579

32. Tang, Z., Wang, S., Anastasiu, D.C., Chang, M.C., et al.: The 10th AI City Challenge. In: ECCV Workshops. Malm"o, Sweden (2026)

33. Tarjan, R.E.: Eficiency of a good but not linear set union algorithm. Journal of the ACM 22(2), 215–225 (1975). https://doi.org/10.1145/321879.321884

34. Teepe, T., Wolters, P., Gilg, J., Herzog, F., Rigoll, G.: EarlyBird: Early-fusion for multi-view tracking in the Bird’s Eye View. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops. pp. 102–111 (Jan 2024). https://doi.org/10.1109/WACVW60836.2024.00018

35. Tirado-Garín, J., Civera, J.: AnyCalib: On-manifold learning for model-agnostic single-view camera calibration. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 8044–8055 (2025). https://doi. org/10.1109/ICCV51701.2025.00754

36. Tran, D.N.N., Pham, L.H., Jeon, H.J., Nguyen, H.H., Jeon, H.M., Tran, T.H.P., Wook Jeon, J.: A robust trafic-aware city-scale multi-camera vehicle tracking of vehicles. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 3149–3158 (2022). https://doi. org/10.1109/CVPRW56347.2022.00355

37. Tran, D.N.N., Pham, L.H., Nguyen, H.H., Tran, T.H.P., Jeon, H.J., Jeon, J.W.: A region-and-trajectory movement matching for multiple turn-counts at road intersection on edge device. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 4082–4089 (2021). https://doi.org/10.1109/CVPRW53098.2021.00461

38. Tran, T.H.P., Tran, D.N.N., Huynh, N.D.M., Tran, C.D., Pham, L.H., Ho, Q.P.N., Nguyen, H.H., Vu, D.K., Jeon, H.M., Jeon, H.J., Phan, S.H., Le Ba Khanh, T., Jeon, J.W.: DepthTrack: Cluster meets BEV for multi-camera multi-target 3D

tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5348–5357 (2025). https://doi.org/10.1109/ ICCVW69036.2025.00558

39. Veicht, A., Sarlin, P.E., Lindenberger, P., Pollefeys, M.: GeoCalib: Learning singleimage calibration with geometric optimization. In: Computer Vision – ECCV 2024. Lecture Notes in Computer Science, vol. 15098, pp. 1–20. Springer (2024). https: //doi.org/10.1007/978-3-031-73661-2\_1

40. Vo, B.T., Vo, B.N., Cantoni, A.: Analytic implementations of the cardinalized probability hypothesis density filter. IEEE Transactions on Signal Processing 55(7), 3553–3567 (2007). https://doi.org/10.1109/TSP.2007.894241

41. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: VGGT: Visual geometry grounded transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5294–5306 (2025). https://doi.org/10.1109/CVPR52734.2025.00499

42. Wang, S., Zhao, X., Xu, H.M., Chen, Z., Yu, D., Chang, J., Yang, Z., Zhao, F.: Towards domain generalization for multi-view 3D object detection in bird-eye-view. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13333–13342 (Jun 2023). https://doi.org/10.1109/ CVPR52729.2023.01281

43. Wang, Y., Meinhardt, T., Cetintas, O., Yang, C.Y., Pusegaonkar, S., Missaoui, B., Biswas, S., Tang, Z., Leal-Taixé, L.: MCBLT: Multi-camera multi-object 3D tracking in long videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 5304–5313 (Oct 2025). https://doi. org/10.1109/ICCVW69036.2025.00553

44. Woo, S., Park, K., Shin, I., Kim, M., Kweon, I.S.: MTMMC: A large-scale realworld multi-modal camera tracking benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22335– 22346 (2024). https://doi.org/10.1109/CVPR52733.2024.02108

45. Xu, Y., Li, Y.J., Weng, X., Kitani, K.: Wide-baseline multi-camera calibration using person re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13129–13138 (2021). https://doi.org/10.1109/CVPR46437.2021.01293

46. Xu, Y., Zhang, J., Zhang, Q., Tao, D.: ViTPose++: Vision transformer for generic body pose estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(2), 1212–1230 (2024). https://doi.org/10.1109/TPAMI.2023.3330016

47. Yamane, T., Masumura, R., Suzuki, S., Orihashi, S.: MVTrajecter: Multi-view pedestrian tracking with trajectory motion cost and trajectory appearance cost. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 13270–13280 (2025). https://doi.org/10.1109/ICCV51701.2025. 01233

48. Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., Ren, D.: Distance-IoU loss: Faster and better learning for bounding box regression. Proceedings of the AAAI Conference on Artificial Intelligence 34(7), 12993–13000 (2020). https://doi.org/10.1609/ aaai.v34i07.6999

# Syn2RealTrack: Bridging the Gap Between Synthetic and Real-World Datasets for Online Multi-View Multi-Target Tracking

Supplementary Material

The supplementary material is organized as follows:

▷ Sec. A: 2D Object Detection.

▷ Sec. B: Re-Identification and Pose Estimation.

▷ Sec. C: 2D Single-View Tracking.

▷ Sec. D: Point Cloud Construction.

## A 2D Object Detection

## A.1 Dataset Preprocessing

The detection ground truth provides both 3D bounding box annotations and their corresponding 2D bounding box annotations. Since the number of 2D bounding boxes is relatively large, directly training on all available annotations may require a longer convergence time, and using the entire training and validation sets without scenario-specific selection may reduce the model’s generalization to the test scenarios. The raw annotations also include redundant, small, or invalid 2D bounding boxes, which may increase false-positive detections. To address these issues, we apply three preprocessing steps before training: lookuptable mapping that summarizes per-object-type shape statistics for fallback object typing, scenario-specific selection that matches the training data to each test scenario, and filtering of redundant, small, or invalid 2D bounding boxes. Each step is described as follows:

Lookup Table Mapping From the provided ground-truth annotations, we derive per-object-type shape statistics by measuring the minimum, maximum, and mean size of every object category. On this basis, objects are grouped into three types:

– Static Objects: objects that stay in place for the entire observation period.

– Fixed-Shape Objects: objects whose shape stays constant whether they are moving or at rest.

– Dynamic-Shape Objects: objects whose shape changes as they move.

These statistics allow the framework to seed each object with a default shape, so that the evaluation still yields meaningful results even when later matching steps fail.

Scenario-Specific Training-Set Selection: The framework aims to reduce the number of training bounding boxes while preserving detection accuracy as much as possible. Accordingly, we select the training data according to the specific conditions of each test scenario, which shortens training without sacrificing reliability. Specific scenarios are also selected to include object categories that are

![](images/c9600574c367e1ca17aa7675601571d2534e7c100ad2c24866b7e71fd7f15f98.jpg)

Scene without filtering  
![](images/cbb9628030ce8d8a02e7f63ecf40743c13cf431636853f447c1caa6eff632d7f.jpg)  
Scene with filtering

Fig. 1: Visualization of the 2D bounding box filtering process. The bounding boxes removed from the training annotations are indicated by red ovals.

absent from other scenes, such as ‘AgilityDigit’ and ‘FourierGR1T2’. For realworld scenes, beyond the AI City training dataset [31], the MTMMC dataset [44] is used to increase the amount and diversity of training data.

Filtering Small or Invalid 2D Bounding Boxes: Redundant bounding boxes, as illustrated in Fig. 1, may increase the likelihood of false-positive detections and consequently degrade the reliability of the detection results. To mitigate this issue, we filter out irrelevant annotations: bounding boxes with excessively small spatial extent, as well as those covering unseen or invalid objects, are removed from the training annotations.

2D Object Detection Training Step: We adopt RF-DETR [25], a real-time detection transformer, as the object detector. Specifically, we train the 2x-large variant for 200 epochs on input images with a resolution of 1920 pixels and a batch size of 4; this setting balances memory usage and training stability.

2D Object Detection Inference Step: During inference, the trained RF-DETR model processes images at the same 1920-pixel resolution with a batch size of 16 to improve throughput. We perform non-maximum suppression at an Intersection over Union (IoU) threshold of 0.5 to reduce redundant overlapping bounding boxes and discard detections with confidence scores below 0.1. Qualitative detection results for the five test scenes are shown in Fig. 2.

## B Re-Identification and Pose Estimation

Pose Estimation We use the pretrained ViTPose++ model [46] for pose estimation to extract keypoints from detected person regions (as shown in Fig. 3). ViTPose++ adopts a plain, non-hierarchical vision transformer encoder with a lightweight keypoint decoder, which localizes body keypoints efectively with a simple, scalable architecture. In addition, the model introduces knowledge factorization through task-agnostic and task-specific feed-forward networks, which allows it to handle heterogeneous body keypoint categories across diferent pose estimation tasks. The pretrained model supplies the pose information used by the subsequent feature extraction, association, and refinement steps.

![](images/951f23ffc1a99c25b689d828c73d548de980b5e84db83cd17ff9baee5b942f9d.jpg)  
Fig. 2: Qualitative 2D detection results on scenes Warehouse 023–027 of the AI City Challenge 2026 Track 1 test set.

Re-Identification For person re-identification (Re-ID), which relies on deep visual features, we adopt Keypoint Promptable Re-Identification (KPR) [28] for feature extraction. KPR uses pose keypoints to divide the human body into six regions and extracts region-level features from each part. This design emphasizes human-centric cues, reduces background interference, and improves robustness under occlusion. Although KPR supports negative keypoints for modeling occluded or unreliable body regions, only positive keypoints are used during training and inference in our pipeline. Since identity labels difer across scenes in the AI City 2026 Track 1 dataset, the training and validation sets are merged and relabeled into a unified dataset containing 155 identities. The model is trained for 110 epochs, with a total training time of approximately 3 hours and 7 minutes.

![](images/ca0bd3a1b7305de4a726b37257a0c1d4fc6779f6a8fcad59be45f85a63f78751.jpg)  
Fig. 3: Qualitative pose-estimation results (ViTPose++ [46] keypoint overlays) on scenes Warehouse 023–027.

## C 2D Single-View Tracking

The result of 2D single-view tracking is visualized in Fig. 4. In the shown frames, the method tracks multiple objects in a single camera view through occlusions and close object interactions: each object keeps its identity across frames, and the bounding boxes follow the objects throughout the sequence.

## D Point Cloud Construction

Since depth maps are unavailable at test time, we generate point-cloud guidance using Depth Anything 3 (DA3) [15] in its pose-conditioned multi-view setting. Specifically, we adopt the updated DA3 Nested Giant-Large 1.1 checkpoint, which integrates any-view geometry with metric-depth scaling. At each synchronized time step, images from all cameras are processed together with the provided intrinsic and extrinsic parameters, K and E. The resulting depth estimates are back-projected into a shared world coordinate system and merged to form a colored point cloud. Fig. 4 presents the multi-view RGB inputs alongside their DA3-estimated point clouds, while Fig. 5 illustrates the corresponding BEV projections for two real-world scenes.

![](images/8e642d8f951d90ceb0144a225e44630e3411701726475cd7b1feda9f45a71577.jpg)

![](images/1d91096a4a34863bba68e5608ec6beb913e44f66f0c0b334ca640d94e00d4fc6.jpg)

![](images/b27361a14d2670b8d8ed9036435936b6e8753d5a8b5c4b5878bd65d1341a076e.jpg)

![](images/23593a08b2e977a5d30e1080ecc5fb977d4d3a386bfe88ea28fb5483c0b920be.jpg)

![](images/7f1b9677a86bf2359be52c3c6e39b72fb4b4aecb734769c112d838b3bffbf4db.jpg)  
Fig. 4: Qualitative 2D single-view tracking results on scenes Warehouse 023–027; each tracked object carries a persistent identity.

![](images/f6f3a26f8c9a4aa174458bda9a542c050f456c81230637cf00ac54b5fd157220.jpg)  
(a) Warehouse 026

![](images/d781c561773a765288f90c7f37c38b7f6aedeb5848ee017b18f33847260deaeb.jpg)  
(b) Warehouse 027  
Fig. 5: Scene-level BEV visualizations constructed from DA3-estimated point clouds for (a) Warehouse 026 and (b) Warehouse 027.