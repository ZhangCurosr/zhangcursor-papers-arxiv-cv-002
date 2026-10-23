# Evaluation of Front-End Feature Matching for Lunar Visual SLAM

Guillem Casadesus Vila and Grace Gao

Department of Aeronautics and Astronautics

Stanford University

Stanford, CA, United States

{guillemc,gracegao}@stanford.edu

Abstract—Lunar rovers will primarily rely on cameras for localization until navigation satellite services are deployed, and will continue to need them afterwards for relative navigation, hazard detection, and surface operations. Visual odometry (VO) and Simultaneous Localization and Mapping (SLAM) mostly estimate motion by matching features between frames. However, feature matchers are developed on urban, indoor and driving scenes, planetary studies compare them on orbital registration or within complete SLAM systems, and real lunar imagery with reference poses is scarce, so matching between rover frames on low-texture, harshly lit terrain, and its effect on VO, has not been compared. We benchmark eleven classical and learned detector–matcher combinations on nine synthetic, lunar-testbed, and terrestrial-analogue datasets, and a feature tracker, scoring pairwise pose accuracy against baseline and the trajectory error of the resulting VO. Under a common rotationonly metric, the SuperPoint pairings outperform SIFT with ratiotested FLANN matching on both lunar testbeds and one analogue site, with smaller or unresolved differences on synthetic data. When rotation and translation direction are scored jointly, this SIFT pairing exceeds both SuperPoint pairings on four synthetic datasets, so synthetic results alone do not establish relative performance on real rover imagery. In stereo VO, the SuperPoint pairings are not consistently more accurate than SIFT, but only they keep tracking on all eight sequential datasets. Pairings with classical detectors depend on lighting while SuperPoint changes little: a low spotlight makes them more accurate than diffuse light on the same testbed views, consistent with more contrast and detected keypoints. SuperPoint with LightGlue is the fastest learned combination (16 ms per frame), and DINOv2 retrieval proposals still need geometric verification.

## TABLE OF CONTENTS

1. INTRODUCTION. .   
2. RELATED WORK . . 2   
3. PROBLEM FORMULATION AND DATASETS . . 2   
4. METHODS . . 4   
5. RESULTS AND DISCUSSION. . 6   
6. CONCLUSION . . . 13   
ACKNOWLEDGEMENTS. 13   
REFERENCES . . . 13   
BIOGRAPHY . . 16

## 1. INTRODUCTION

Future lunar rovers will drive much farther and faster than any planetary rover so far. Artemis rovers must navigate tens of kilometers from base camps, robotic science rovers at up to 30 cm/s, and the Endurance mission concept requires a rover to traverse over 2700 km of the South Pole–Aitken basin at a nominal 1 km/h [1], whereas Perseverance has averaged about 2.6 cm/s when driving autonomously [2]. Planetary rovers have so far tracked their motion from the start of each drive and received absolute position updates from operators on Earth, which is expected to limit single autonomous drives to a few hundred meters [3]. Perseverance’s recent onboard global localization also relies on cameras [4,5]. Cameras will thus be the main localization sensor until LunaNet navigation satellite services [6] are deployed, and will remain essential afterwards for relative navigation, mapping, and hazard detection.

Rovers will rely on visual odometry (VO), which estimates camera motion from consecutive images, and on Simultaneous Localization and Mapping (SLAM), which jointly estimates the trajectory and a map of the environment and bounds accumulated error when it recognizes revisited places [7, 8]. Most VO and SLAM systems, unlike direct methods that use image intensities, detect and match keypoints and estimate the relative pose with random sample consensus (RANSAC) or its variants [9–11]. Correspondence quality therefore affects the trajectory and any map built from it.

Lunar images are difficult to match: regolith is low in texture, and camera optics add distortions and noise. The viewpoint change between matched images also grows when the rover drives faster or skips frames to save compute. At 30 cm/s and 10 Hz a rover advances 3 cm per frame, so VO on every frame matches nearly identical views, while VO on fewer frames matches views up to meters apart. Loop closure instead matches views of the same place taken from a different position and heading, which is expected to be harder on terrain with few distinctive landmarks.

Most matching studies and benchmarks report a single score per dataset [12–14], which hides how matching degrades as the viewpoint change grows, and synthetic data alone cannot show whether performance transfers to real imagery. Matching and visual place recognition are also benchmarked separately, although SLAM needs both: retrieval proposes a revisit, and matching verifies it.

We evaluate the correspondences used for VO and for loop closure with the same front ends. Our contributions are:

A benchmark of eleven detector–matcher combinations on nine synthetic, lunar-testbed, and terrestrial-analogue datasets, and a Kanade–Lucas–Tomasi (KLT) tracker, with pose error characterized by the viewpoint change between the images, and the computational cost of the main front ends.

A comparison of classical and learned matchers under camera effects and three lightings, and of whether their relative performance on synthetic imagery carries over to real imagery, scored by rotation alone and by rotation and translation direction jointly.

The operating range of each combination along the baseline, its correlation with the absolute trajectory error (ATE) of VO as a percentage of path length (ATE/L), and a test of whether pairwise matching suffices for loop closure.

The code, configurations, pair lists and exclusion masks will be released with a future version of the LuPNT simulator [15].

## 2. RELATED WORK

This section reviews VO and SLAM works, feature detection and matching, their use in space applications, existing matching benchmarks, and place recognition for loop closure.

## Visual Odometry and SLAM

Feature-based VO estimates the relative motion from keypoints matched between frames, after rejecting outliers [7, 16]. This stage is the front end of SLAM [8], and visual SLAM methods are classified as sparse, semi-dense, or dense [17]. We evaluate the sparse front end alone.

VO benchmarks score whole trajectories rather than image pairs [18, 19]. In VO systems, viewpoint change typically enters through the keyframe rule. ORB-SLAM inserts a keyframe when, among other conditions, the current frame tracks fewer than 90 % of the reference keyframe’s points [20], SVO when the camera has moved more than 12 % of the average scene depth [21], and DSO when a weighted sum of optical flow, rotation-compensated flow, and exposure change exceeds a threshold [22]. We likewise describe pair difficulty by viewpoint change, through overlap and triangulation angle.

Direct and semi-direct methods such as LSD-SLAM [23], DSO, and SVO minimize photometric error instead of matching descriptors (SVO aligns sparse patches directly), so they have no detector–matcher stage to exchange. LDSO [24] adds corners with ORB descriptors to DSO for loop closure. Our KLT baseline is the closest member of this family that yields correspondences. End-to-end systems such as DROID-SLAM [25], DPVO [26], and TartanVO [27] have no keypoint detector or descriptor matcher to exchange.

## Feature Detection and Matching

Classical methods hand-craft detection and description: SIFT [28] uses gradient histograms, while ORB [29], AKAZE [30], and BRISK [31] use cheaper binary descriptors. ORB is the front end of ORB-SLAM3 [32]. Learned methods replace one or both steps: Key.Net [33] learns only the detector, while SuperPoint [34], D2-Net [35], R2D2 [36], DISK [37], ALIKED [38], and XFeat [39] learn both. We evaluate the four classical detectors and SuperPoint, the only learned one with released weights for both SuperGlue and LightGlue.

Descriptors are matched by nearest-neighbor search, exhaustive or approximate as in FLANN [40], and filtered with Lowe’s ratio test [28] or a mutual cross-check. Learned matchers reason over both keypoint sets jointly: Super-Glue [12] uses graph attention and Sinkhorn iterations [41], and LightGlue [42] reduces its cost with early exit and point pruning. Both are trained on MegaDepth [43], LightGlue after pre-training on synthetic homographies. We use them without retraining, so our results include this domain gap.

Detector-free matchers [44] such as LoFTR [13], ASpan-

Former [45], MatchFormer [46], DKM [47], and RoMa [48], and 3D foundation models such as DUSt3R [49] and MASt3R [50], do not separate into detector and matcher stages, so we leave them out.

## Feature Matchingfor Space Applications

Stereo VO ran for two years on the Mars Exploration Rovers [51]. Localization against an orbital map, as in the Mars 2020 Lander Vision System [52] and crater matching for lunar rovers [3], gives a global fix that complements VO. In a simulated trade study for the Endurance concept, craterbased localization degraded at low sun and with computed rather than ground-truth stereo depth, while dense intensity registration against orbital imagery stayed within 3 m in every case [1].

The closest precedents evaluate features inside complete SLAM systems. One study replaces ORB with SuperPoint in ORB-SLAM3 [53], and another evaluates visual SLAM in planetary-like and agricultural environments [54]. Neither varies the matcher, characterizes pairs by viewpoint change, or scores loop closure separately. Matchers have also been compared on Chandrayaan-2 orbital imagery [55], a crosssensor registration task rather than matching between rover frames. DINO-VO [56] builds a VO front end on DINOv2 features and has been evaluated only on terrestrial datasets. Two lunar systems from our group use a SuperPoint and LightGlue front end [57, 58], and this benchmark tests that choice.

## Benchmarking Feature Matching

The Image Matching Challenge [14] made recovered pose the primary measure of matching quality. SuperGlue [12] and LoFTR [13] report it as Pose AUC, the area under the curve (AUC) of the cumulative pose error, which we adopt. HPatches [59] scores descriptors on planar scenes, which have no parallax.

RUBIK [60] bins nuScenes pairs by overlap, scale ratio, and the triangulation angle α¯ we also report, but covers one terrestrial driving dataset and scores image pairs only. We apply the same description to lunar and analogue data, sample pairs on a grid of metric translations as a rover would space its frames, and add downstream VO and loop closure.

## Visual Place Recognition and Loop-Closure Evaluation

Visual place recognition (VPR) benchmarks score retrieval [61, 62] but not the pose recovered once a place is found. Revisits are defined by a fixed radius [63], a radius with a viewing-angle limit [64], or the visual overlap of the two views [65, 66]. We use a radius combined with a minimum arc-length separation.

The closest planetary-analogue loop-closure work is MPRF [67], which combines DINOv2 retrieval with LiDAR-based pose estimation on S3LI sequences including the two we use. To our knowledge, no prior work evaluates the same cameraonly front ends for both VO and loop-closure verification across synthetic, lunar-testbed, and terrestrial-analogue data, and relates the pairwise results to VO accuracy.

## 3. PROBLEM FORMULATION AND DATASETS

This section states the problem, defines how we measure viewpoint change, and describes the nine datasets and how we sample image pairs.

## Problem Formulation

VO chains relative poses between consecutive frames, and SLAM verifies each revisit with the same pairwise step: given two frames $I _ { i } , I _ { j } ,$ detect keypoints, match them, and estimate the relative pose $( \mathbf { R } _ { i j } , \mathbf { t } _ { i j } )$ How well a combination does this depends on how different the two views are, so we score it against viewpoint change. We ask which combination recovers the relative pose most accurately on lunar and analogue imagery and over what baseline, whether its relative performance on synthetic data carries over to real data, and whether pairwise results carry over to VO and revisit verification.

## Describing Viewpoint Change

We describe viewpoint change by the overlap ratio ω and the triangulation angle $\bar { \alpha } ,$ computed from reference pose and depth as in the RUBIK benchmark [60]. Let $T _ { k } ^ { * } = [ \mathbf { \dot { R } } _ { k } ^ { * } \ | \ \mathbf { t } _ { k } ^ { * } ]$ be the ground-truth camera-from-world transform of frame $I _ { k } , \mathbf { c } _ { k } ^ { * } = - \mathbf { R } _ { k } ^ { * } { } ^ { \top } \mathbf { t } _ { k } ^ { * }$ its camera center, and $T _ { i j } ^ { * } = T _ { j } ^ { * } ( T _ { i } ^ { * } ) ^ { - 1 } =$ $[ \mathbf { R } _ { i j } ^ { * } \mid \mathbf { t } _ { i j } ^ { * } ]$ the relative pose. For a pair, V<sub>i</sub> and V<sub>j</sub> are the nonsky pixels with valid depth. A pixel $\mathbf { u } \in \mathcal { V } _ { i }$ with depth $D _ { i } ( \mathbf { u } )$ is back-projected with intrinsics K to $\mathbf { p } ~ = ~ D _ { i } ( \mathbf { u } ) \mathbf { K } ^ { - 1 } \tilde { \mathbf { u } } .$ moved to camera $j$ as $\mathbf { p } ^ { \prime } = \mathbf { R } _ { i j } ^ { * } \mathbf { p } + \mathbf { t } _ { i j } ^ { * }$ , and projected to a pixel $\mathbf { u } ^ { \prime } .$ . It is co-visible if p′ lies in front of camera j, u′ lands on a valid pixel of $I _ { j } { \mathrm { : } }$ , and the depth of $\mathbf { p } ^ { \prime }$ matches $D _ { j } ( \mathbf { u } ^ { \prime } )$ within $5 \%$ which rejects occlusions. With $\mathcal { C } _ { i \to j } \subseteq \mathcal { V } _ { i }$ and ${ \mathcal { C } } _ { j \to i } \subseteq { \mathcal { V } } _ { j }$ the co-visible pixels in each direction, the overlap ratio is

$$
\omega = \frac { | \mathcal { C } _ { i \to j } | + | \mathcal { C } _ { j \to i } | } { | \mathcal { V } _ { i } | + | \mathcal { V } _ { j } | } .\tag{1}
$$

Overlap selects training pairs in D2-Net [35] and test pairs in SuperGlue [12], and we use it as the primary axis.

Overlap does not measure parallax, so we also compute the median triangulation angle α¯ [9]. For a co-visible point X,

$$
\alpha ( \mathbf { X } ) = \operatorname { a r c c o s } \left( { \frac { ( \mathbf { X } - \mathbf { c } _ { i } ^ { * } ) \cdot ( \mathbf { X } - \mathbf { c } _ { j } ^ { * } ) } { \| \mathbf { X } - \mathbf { c } _ { i } ^ { * } \| \left\| \mathbf { X } - \mathbf { c } _ { j } ^ { * } \right\| } } \right) ,\tag{2}
$$

and α¯ is its median over the co-visible points in both directions. A pure rotation gives $\bar { \alpha } = 0 .$ For the axis comparison we also use the scale ratio δ, the median of max $( d _ { i } , d _ { j } ) /$ min $( d _ { i } , d _ { j } )$ , where $d _ { i }$ and $d _ { j }$ are the distances from X to the two camera centers.

The reference depth differs by tier (Table 1). On S3LI it comes from RAFT-Stereo [68] and is cut at $Z _ { \operatorname* { m a x } } = 0 . 1 f _ { x } B$ where a 1-pixel disparity error causes a 10 % depth error $( f _ { x }$ is the focal length in pixels and B the baseline). The cutoff drops distant terrain, which lowers ω and raises ${ \bar { \alpha } } ,$ and on LuSNAR (5) cutting rendered depth at $Z _ { \mathrm { m a x } }$ closely reproduces the stereo values. Absolute ω and α¯ are therefore not comparable across tiers, so we compare overlap-binned accuracy only within a tier.

The camera translation $\Delta t$ and rotation $\Delta \theta$ need only the ground-truth poses,

$$
\Delta t = \| \mathbf { c } _ { j } ^ { * } - \mathbf { c } _ { i } ^ { * } \| _ { 2 } ,\tag{3}
$$

$$
\begin{array} { r } { \Delta \theta = \operatorname { a r c c o s } \biggr ( \frac { \operatorname { t r } ( \mathbf { R } _ { i } ^ { * } ^ { \top } \mathbf { R } _ { j } ^ { * } ) - 1 } { 2 } \biggr ) . } \end{array}\tag{4}
$$

We use them to sample frames and to report per-step accuracy against ∆t, but not as difficulty axes, because the difficulty of a translation depends on scene depth and a rotation changes overlap without adding parallax.

## Datasets and Ground Truth

We use nine datasets in three tiers: synthetic data from LuPNT and LuSNAR [69], lunar testbed data from the indoor analogue facilities of PolarTraverse [70] and LunaPolaris [71], and terrestrial analogue data from S3LI, recorded on volcanic terrain [72, 73]. Table 1 summarizes them, and Fig. 1 shows examples.

LuPNT Simulator (Synthetic)—We use LuPNT [15], an opensource simulator for lunar communications, positioning, navigation, and timing, with its Unreal Engine 5.7 surface rendering. A rover drives over south-polar terrain built from high-resolution elevation models with procedurally generated rocks and craters. We render a stereo camera along one 378 m path that loops around a lander, in three conditions: base (sun elevation $1 5 ^ { \circ }$ , our reference condition, where the surrounding relief shadows much of the terrain), camera effects, and high sun (sun elevation 45◦). The camera-effects run adds depthof-field blur, chromatic aberration, bloom, and lens flare.

LuSNAR (Synthetic)—LuSNAR [69] provides stereo imagery of nine lunar scenes built in Unreal Engine 4. We use scenes 9 and 5, whose terrain and rendering differ from LuPNT, as two datasets, downsampled to 512 by 512 pixels, close to the real image sizes.

PolarTraverse (Lunar Testbed)— This dataset [70] was recorded over a regolith-simulant test bed shaped after southpolar terrain and lit from about $2 { - } 3 ^ { \circ }$ elevation, with a stereo bar stopping at 11 stations 1 m apart. Poses are refined with COLMAP [74] and registered to a laser scan of the test bed, which also provides the reference depth. Because these poses come from the images themselves, the PolarTraverse pose scores depend on their accuracy. We use one image set (one camera view, traverse and exposure time) and score its 38 station pairs.

LunaPolaris (Lunar Testbed)—LunaPolaris [71] was recorded by a small rover in the LunaLab facility of the University of Luxembourg, in twelve sequences that repeat one trajectory (27.5 m evaluated) under three boulder layouts and four lighting setups. We use sequence 01 (ceiling lights), and for the lighting comparison also 07 (a low spotlight) and 10 (the spotlight and a rover headlight). Images come from a ZED2 stereo camera, and the reference depth is a dense lidar map of the arena rendered into the camera. A hand-eye calibration per sequence locates the onboard lidar, and through the rig calibration the camera, relative to the motion-capture marker. Left and right images are paired by capture time, and the motion-capture pose is interpolated at that time rather than at the time of receipt. We skip frames with motion-capture dropouts. In the small test bed some pairs move under 10 cm, leaving their reference translation direction poorly defined.

Table 1: Datasets used for evaluation. GNSS: global navigation satellite system.
<table><tr><td>Dataset</td><td>Type</td><td>Conditions</td><td>Frames</td><td>Reference depth</td><td>Ground-truth pose</td></tr><tr><td>LuPNT (base)</td><td>Synthetic</td><td>Sun elevation  $\overline { { 1 5 ^ { \circ } } }$ </td><td>4,281</td><td>Rendered</td><td>Simulator</td></tr><tr><td>LuPNT (cam.)</td><td>Synthetic</td><td>Camera effects</td><td>4,276</td><td>Rendered</td><td>Simulator</td></tr><tr><td>LuPNT (high sun)</td><td>Synthetic</td><td>High sun, elevation  $4 5 ^ { \circ }$ </td><td>4,281</td><td>Rendered</td><td>Simulator</td></tr><tr><td>LuSNAR (5)</td><td>Synthetic</td><td>Scene 5</td><td>1,599</td><td>Rendered</td><td>Simulator</td></tr><tr><td>LuSNAR (9)</td><td>Synthetic</td><td>Scene 9</td><td>2,087</td><td>Rendered</td><td>Simulator</td></tr><tr><td>PolarTraverse</td><td>Lunar testbed</td><td>Regolith testbed</td><td>11 stations</td><td>Laser scan</td><td>COLMAP registered to laser scan</td></tr><tr><td>LunaPolaris</td><td>Lunar testbed</td><td>Ceiling lights (seq. 01)</td><td>2,244</td><td>Lidar</td><td>Motion capture</td></tr><tr><td>S3LI (Etna)</td><td>Terrestrial analogue</td><td>Volcanic terrain</td><td>20,833</td><td>Stereo</td><td>Differential GNSS, gyroscope</td></tr><tr><td>S3LI (Vulcano)</td><td>Terrestrial analogue</td><td>Volcanic terrain</td><td>12,191</td><td>Stereo</td><td>Differential GNSS, gyroscope</td></tr></table>

![](images/9187e16235533dd96d6ecaf87a60232f8d0bdc529f34f89a5cd0f3ce7365c910.jpg)

S3LI (Terrestrial Analogue)— S3LI contributes two sequences, Etna [72] (loops) and Vulcano [73] (moon lake), neither with dense ground-truth depth. Positions come from differential Global Navigation Satellite System (GNSS) receivers. Fixed-ambiguity solutions, accurate to 1 to 10 cm [72, 73], cover nearly all our Etna pair frames but almost none of Vulcano. We drop frames at GNSS position jumps where positions are used. Orientation is integrated from the gyroscope and aligned to the GNSS east-north-up frame using gravity and GNSS velocity. Its median error per pair, estimated against an independent visual rotation track, is at most 0.31◦, and we check the S3LI results under rotation perturbations of up to three times this error. As heading is uncertain by several degrees, epipolar precision (EP), ω and α¯ on S3LI depend on the translation direction and are approximate.

## Pair Sampling

Unless noted, pairwise results use a mixed pool of up to 1000 independent pairs per dataset, drawn at frame steps chosen per dataset so that their median translations run from about 0.1 to 8 m. Because the rover speed varies, one frame step covers a range of translations, so we label each step by the median translation ∆t of its pairs. On LunaPolaris sequence 07, evaluated at 3.0 rather than 5.0 Hz, a frame step spans a longer baseline.

LuPNT (base) fills every overlap bin, LuSNAR (5) and PolarTraverse lack low-overlap pairs, and LunaPolaris and the S3LI sites have many pairs with almost no overlap (Fig. 2). Along a traverse, translation, rotation and parallax grow jointly, so these axes are strongly coupled in synthetic data.

![](images/3d821dff9a0ee2f738c2d8b64e96bf59c6d95ec3fe5c57f4273a8b8f25b6119b.jpg)  
Figure 2: Fraction of mixed-pool pairs per overlap bin (RUBIK [60] bin edges plus a 0–5 % bin) on six datasets, with median overlap ω˜ and triangulation angle α¯.

## 4. METHODS

We evaluate the front end alone and use no pose-graph or bundle-adjustment back end, so every trajectory is open-loop.

## Detectors and Matchers

The detectors are SIFT [28], ORB [29], AKAZE [30], BRISK [31], and SuperPoint [34], a convolutional network trained by homographic adaptation. SIFT and SuperPoint have real-valued descriptors, the others binary ones.

LuPNT (base)  
![](images/9bfea0367c085c3475d1c4e0b339d8cb65c98ac5a68c581b4a8c284789574d92.jpg)

LuSNAR (9)  
![](images/d3c151530f14441698aa5a10cea2b4b45cef41cd10cf6b12681ba6cd4c5a8874.jpg)

PolarTraverse  
![](images/83b852a583962fd5f8f3c816edb52a379440d2c2473b7e372fbd8889190fa9bc.jpg)

LunaPolaris (seq. 07)  
![](images/c795b69270beb64fc53f3df9a77b528f432018b4bda25ad60200002241b6a9b4.jpg)

S3LI (Etna)  
![](images/b67abd9e269afe34eb06af2ba75f82efa08bab41e42a0eddb865ad70968ef276.jpg)

S3LI (Vulcano)  
![](images/979602f64642f70d79d53fd578ce9bd936fc63103bfbd294478791874d3731dd.jpg)

![](images/d7e291e6306fa1238b4c01c6f9e61d5eea5b2f8456fc1fb5bc29a90710c46617.jpg)  
X [m]

![](images/8aedaf675a077870606c515071385904fc9f971cd658f5c54500c76772f27ac7.jpg)  
X [m]

![](images/55e9a1b529aa09767d425aba0762fd167ff1ea0757b646a00d3a56e2587892d9.jpg)

![](images/bb690aa6b48022215934177f6c2076280b2eeaa5b4e2bfa667d9453daad1110d.jpg)  
X [m]

![](images/d52e2e95de449b74abc4cabb7d7bfdc0b21a9bd7cdb5a592b5883c4c9bf0d517.jpg)  
X [m]  
Figure 1: Sample image, reference depth (sky and invalid pixels gray) and top-down trajectory with the sample frame marked, for six of the nine datasets. LunaPolaris shows sequence 07. TLS: terrestrial laser scan.

FLANN [40,75] searches approximate nearest neighbors with randomized k-d trees for real-valued and OpenCV’s localitysensitive hashing index for binary descriptors, and Brute-Force searches exhaustively. As is standard, FLANN matches pass Lowe’s ratio test at 0.75 and BruteForce matches a mutual cross-check. Because FLANN is randomized, its results report the median over five seeds. SuperGlue [12] and LightGlue [42] keep only confident matches, and we use their MegaDepth [43] weights and feed LightGlue RootSIFT descriptors for SIFT.

As a tracking baseline, we pair Shi–Tomasi corners [76] with Lucas–Kanade tracking [77] over an image pyramid and a forward-backward check. Features are tracked either directly from $I _ { i }$ to $I _ { j } ,$ as the matchers see them, or through every intermediate frame (none on PolarTraverse).

SuperPoint (SP) is paired only with the learned matchers, which gives 11 combinations: SIFT, ORB, AKAZE and BRISK each with FLANN and BruteForce, plus SIFT+LightGlue, SP+SuperGlue and SP+LightGlue. To separate detector from matcher, four ablation combinations pair SuperPoint and RootSIFT with FLANN and BruteForce under the same filtering.

## Visual Odometry Front-End

Our frame-to-frame VO back-projects the keypoints of $I _ { k - 1 }$ matched to $I _ { k }$ with their depth, estimates the pose of $I _ { k }$ with OpenCV’s [78] RANSAC [79] Perspective-n-Point (PnP) solver, and chains it into the trajectory. If too few inliers have metric depth, the pose is rejected and the next frame is matched against the last accepted one. Since a rover has no ground-truth depth, stereo VO, run on every sequential dataset, takes depth from the disparity of left-right matches found with the same matcher. Stereo matches must be oneto-one with a row offset within 2 px of the frame’s median offset, and matches with zero disparity act as rotation constraints. On LuPNT and LuSNAR we also run VO with rendered ground-truth depth, which isolates matching and pose estimation from depth error. The real datasets also differ from the synthetic ones in terrain, so their VO results are not directly comparable. PolarTraverse has no image sequence and is evaluated pairwise only.

## Evaluation Metrics and Protocol

Pairwise metrics—EP, reprojection precision and Pose AUC follow Glue Factory [80] with known intrinsics. A star (∗) marks ground truth and a hat (ˆ·) estimates.

EP is the fraction of candidate matches that lie within $\tau _ { \mathrm { e p i } } =$ $1 0 ^ { - 3 }$ in normalized camera coordinates (about 0.5 px on LuPNT) of their epipolar line under the ground-truth essential matrix $\mathbf { E } ^ { * } ~ = ~ [ \mathbf { t } _ { i j } ^ { * } ] _ { \times } \mathbf { R } _ { i j } ^ { * }$ Because it scores all candidate matches, a matcher is penalized for wrong matches even when RANSAC would discard them. Reprojection precision is the fraction of candidate matches within a pixel threshold of the correspondence given by reference depth.

For Pose AUC, we estimate the essential matrix from the candidate matches with RANSAC and the five-point solver [81] and decompose it into a rotation R<sup>ˆ</sup> and a translation direction <sup>ˆ</sup>t. The rotation error $e _ { R }$ is the angle between R<sup>ˆ</sup> and $\mathbf { R } _ { i j } ^ { * } ,$ and the translation error $e _ { t }$ the angle between <sup>ˆ</sup>t and the reference direction, up to sign. As is standard [12, 13, 42], the joint pose error is $\mathbf { \bar { \rho } } e = \operatorname * { m a x } ( e _ { R } , e _ { t } )$ on the synthetic tier and PolarTraverse. On S3LI and LunaPolaris the reference direction is not accurate enough, so we score rotation alone, $e = e _ { R } ,$ , and mark these results rotation-only. $\mathrm { A U C @ 5 ^ { \circ } }$ is the area under the curve of the fraction of pairs with error below a threshold, for thresholds from 0 to $5 ^ { \circ }$ , as a percentage, and a pair whose pose cannot be estimated counts as a miss. On LunaPolaris, the 26 pairs that include a motion-capture glitch frame are not scored. For a common metric across datasets, Table 3 scores rotation alone on all nine.

The operating range of a combination on a dataset is the set of frame steps ∆t at which Pose AUC@5◦ (as scored on that dataset) is at least 50 %,

$$
\mathcal { R } = \big \{ \Delta t : \mathrm { A U C @ 5 ^ { \circ } ( \Delta t ) } \geq 5 0 \% \big \} ,\tag{5}
$$

and its upper end $\Delta t ^ { * } = \operatorname* { s u p } \mathcal { R }$ , interpolated log-linearly between steps, is the largest baseline the combination handles. A combination that never reaches 50 % has no range. PolarTraverse has too few pairs per step for a range.

To tell whether two combinations differ, we compute a bootstrap 95 % confidence interval of their difference in Pose $\mathrm { A U } \dot { \mathrm { C } } @ 5 ^ { \circ }$ , scoring both on the same resampled pairs. Because nearby pairs see the same terrain, we resample blocks of neighboring pairs. A difference is unresolved when this interval contains both positive and negative values, so the data cannot tell which is better. For the lighting comparison, we pair frames across sequences by pose, within 0.05 m and $3 ^ { \circ }$ (loosely 0.10 m and $\dot { 5 } ^ { \circ } )$ , and Holm-adjust the p-values over all tested changes. A qualitative check uses a smaller, screened set of pose-matched pairs.

To choose a reference difficulty axis, we bin the mixed-pool pairs of each synthetic dataset into deciles of ω, α¯, δ, ∆t and ∆θ and compute $\eta ^ { 2 }$ , the fraction of per-pair EP variance explained by the decile. We average $\eta ^ { 2 }$ over SIFT+FLANN, AKAZE+FLANN and the SuperPoint pairings.

Trajectory metrics—We align the estimated VO poses, taken at the rover body origin, to the ground truth with an Umeyama SE(3) fit (no scale). The Absolute Trajectory Error (ATE) is the root-mean-square (RMS) error of the aligned positions, and ATE/L expresses it as a percentage of the ground-truth path length L of the evaluation window (on S3LI, the first 10 % of each sequence). As a whole-trajectory error after alignment, ATE/L is not directly comparable to the startanchored end-point error used in rover operations. A run that diverges to non-finite poses is scored up to that point, with the earlier frames counted as tracked. Path coverage is the fraction of the path along which VO keeps tracking, and a gap of more than 10 frames counts as lost. We report ATE and ATE/L only for runs with at least 90 % coverage and mark the others as stopped.

A single VO run changes when its input images change slightly or when it is repeated, so we treat ATE differences below this run-to-run variation as unresolved, a practical rule, not a test of statistical equivalence. Slightly perturbed images move the ATE of SP+LightGlue on LuPNT (base) by up to about 0.26 m, our default tolerance. GPU runs and randomized FLANN matching vary more, by up to 0.64 m and 1.46 m, so their comparisons use the larger spread.

Revisit and retrieval evaluation—A ground-truth revisit is a pair of frames within a radius R of each other and at least $S =$ 20 m apart along the path. Consecutive revisit frames within 10 m along the path form one pass. Passes linked by revisits form a revisit component, which is one revisited place. A dataset’s revisits are low-angle if at least half the pairs rotate less than 30◦, and wide-angle otherwise.

Since loop closure relies on the front end to verify a candidate revisit, we score pose recovery on true revisits, not loop detection, using the ground-truth revisit pairs of LuPNT (base) within 2 m and of LuSNAR (5) within 1 m. The scores are Pose AUC, the number of candidate matches and the RANSAC inlier fraction.

We evaluate whole-image retrieval on datasets with at least three revisit components (Table 7), using the DINOv2 [82] class (CLS) token with cosine similarity on evenly spaced frames and radii R ∈ {1, 2, 5} m, smaller than the 25 m of urban benchmarks [63, 64] to match a rover traverse. Each frame queries frames more than 20 m away along the path, either only earlier ones (causal, as in online loop closure) or both sides (offline). Its top match is correct if the two cameras lie within R, and any match for a query without a revisit is a false positive. We report recall@1, the area under the precision-recall curve (AUPRC) [61], and R@100P, the recall before the first false positive [65], once the key loopclosure operating point [83].

## Comparing the Operating Range with VO ATE/L

For each combination and synthetic dataset, we compare ∆t∗ (Eq. (5)) with the ATE/L of VO with ground-truth and with stereo depth by a pooled Spearman correlation, excluding combinations without a range. The correlation is in-sample, and its points are not independent because combinations share detectors and matchers and the LuPNT conditions share one path, so we also bootstrap over datasets or combinations.

## Implementation

Experiments ran on two workstations, one with an AMD Ryzen 9 9950X CPU and an NVIDIA RTX 5090 GPU. Learned methods run on the GPU and classical ones on the CPU. All use published default settings with no per-dataset tuning, so ORB keeps at most 500 keypoints and the other detectors are uncapped. We also test relaxed ORB and AKAZE settings.

We time the FLANN and learned combinations on consecutive LuPNT (base) frames on the RTX 5090 workstation, counting one extraction and one match per frame, without stereo matching or pose estimation.

## 5. RESULTS AND DISCUSSION

We first compare pose accuracy across datasets and along the baseline, then camera effects, lighting and cross-dataset transfer, VO and loop closure, and finally the cost of each front end.

In all tables, bold and underline mark the best and secondbest value, which does not by itself mean the difference is resolved. Unless noted, pairwise metrics average over the mixed pool and VO runs use consecutive evaluated frames. A difference is “not resolved” when the data cannot tell which combination is better (Section 4).

## Accuracy Across Datasets

Fig. 3 compares the matches of a classical and a learned combination on one pair from each tier. Both recover the pose on the synthetic pair, but on the two real pairs only the learned one still matches the terrain densely, while the classical one keeps scattered correspondences and fails.

Under a common rotation-only metric (Table 3), both Super-Point pairings outperform SIFT+FLANN on PolarTraverse, LunaPolaris and S3LI (Etna), and by smaller margins on LuPNT (base) and LuPNT (cam.). Elsewhere the difference is not resolved, and SIFT+FLANN is not resolved ahead on any dataset. Under the joint metric of Table 2, SIFT+FLANN instead exceeds both SuperPoint pairings on four of the five synthetic datasets, a lead that comes from the translation direction, which short baselines constrain poorly. Which family comes out ahead therefore depends on the metric, and since rover VO and relocalization need both rotation and translation direction, neither score alone predicts how a front end will localize. The datasets with the largest margins also differ from the synthetic ones in more than appearance, so we do not attribute the margins to simulation, although synthetic results alone do not settle the relative performance expected on real rover imagery. In Table 2, a SuperPoint pairing is best on every real dataset except S3LI (Vulcano), and the order of the top three on LuPNT (base) and of the top two on LuPNT (high sun) is not resolved.

The ablation attributes most of the testbed lead to the SuperPoint detector and descriptor rather than the matcher. On PolarTraverse and LunaPolaris, SuperPoint with FLANN already exceeds every SIFT combination, although with bruteforce matching it falls below SIFT+LightGlue on PolarTraverse. On synthetic data, LightGlue clearly improves SuperPoint over FLANN but barely changes SIFT. Converting SIFT to RootSIFT, the descriptor of LightGlue’s SIFT model, changes accuracy little anywhere. Nor does the lead come from more keypoints, as SuperPoint returns fewer than SIFT on every dataset except PolarTraverse, so it lies in the quality of the keypoints and descriptors. This suggests that on other real terrain replacing the classical detector and descriptor matters more than replacing the matcher.

On S3LI (Etna) the SuperPoint lead is widest on pairs far apart (Fig. 4). On PolarTraverse, whose pairs are widely spaced, the SuperPoint pairings are more than twice as accurate as any classical combination. On S3LI (Vulcano) the lead of SIFT+LightGlue over the better SuperPoint pairing is borderline across bootstrap seeds, and perturbing the S3LI reference rotations by their modeled error barely changes the order. Evaluating only closely spaced frames would therefore understate the benefit of SuperPoint on real terrain.

EP and Pose AUC can disagree. EP is highest for SIFT+FLANN on all five synthetic datasets and for SIFT+LightGlue on PolarTraverse, whereas a SuperPoint pairing leads Pose AUC on LuPNT (base) and PolarTraverse. Only on LunaPolaris is EP highest for a SuperPoint pairing. EP counts a match as correct only within about half a pixel of its epipolar line, so it rewards precise keypoint localization. On LuPNT (base), SuperPoint matches lie within a few pixels of their reference correspondence as often as SIFT matches but much less often within one pixel, consistent with the subpixel refinement of SIFT keypoints. SuperPoint also yields about twice as many matches, leaving RANSAC more inliers. EP thus measures how precisely matches are localized rather than how well they constrain the pose, so it should not guide the choice of a front end on its own.

SIFT+FLANN is the best classical combination at default settings on every dataset, counting SIFT+LightGlue as learned throughout, though not resolved from BRISK+FLANN on LunaPolaris. It serves as the classical reference, but only with its ratio test or cross-check. With OpenCV’s default nearestneighbor matching it loses about half its accuracy on LuPNT (base), whereas the SuperPoint pairings filter internally and are unaffected. Match filtering is therefore part of a classical front end, since RANSAC alone does not recover the lost accuracy.

Table 2: Pose $\mathbf { A U C } @ 5 ^ { \circ }$ [%] of eleven combinations and, in italics, four ablation pairings. The pose error is the larger of the rotation and translation-direction errors, except on LunaPolaris and S3LI (rotation alone). Bold/underline = best/second among the eleven. FLANN cells: median of five seeds, spread ≤ 4.9. Unmarked KLT rows track between the two frames (direct) or through every intermediate frame (chained, “—” without a sequence). AUC@5<sup>◦</sup>: area under the cumulative pose-error curve up to $\mathbf { 5 } ^ { \circ }$ , FLANN: fast library for approximate nearest neighbors, KLT: Kanade– Lucas–Tomasi, SP: SuperPoint.
<table><tr><td rowspan="3">Detector Matcher</td><td rowspan="3"></td><td colspan="4">Synthetic</td><td colspan="2">Lunar testbed</td><td colspan="2">Terrestrial analogue</td></tr><tr><td colspan="3">LuPNT</td><td colspan="2">LuSNAR</td><td rowspan="2">LunaPolaris</td><td colspan="2">S3LI</td></tr><tr><td>(base)</td><td>(cam.)</td><td>(high sun)</td><td>(5) (9)</td><td></td><td>(Etna)</td><td>(Vulcano)</td></tr><tr><td>SIFT</td><td>FLANN</td><td>58.4</td><td>68.5</td><td>74.9</td><td>67.1 69.6</td><td>37.8</td><td>31.6</td><td>65.0</td><td>76.5</td></tr><tr><td>SIFT</td><td>BruteForce</td><td>43.2</td><td>53.1</td><td>55.9</td><td>51.0 53.5</td><td>28.8</td><td>15.0</td><td>47.0</td><td>69.3</td></tr><tr><td>SIFT</td><td>LightGlue</td><td>60.4</td><td>70.5</td><td>75.2</td><td>63.2 66.1</td><td>44.9</td><td>29.2</td><td>69.9</td><td>78.7</td></tr><tr><td>ORB</td><td>FLANN</td><td>14.4</td><td>25.7</td><td>26.9</td><td>32.8 33.6</td><td>3.7</td><td>9.9</td><td>34.4</td><td>49.8</td></tr><tr><td>ORB</td><td>BruteForce</td><td>13.9</td><td>27.7</td><td>26.9</td><td>35.9 37.4</td><td>3.4</td><td>6.7</td><td>34.4</td><td>50.1</td></tr><tr><td>AKAZE</td><td>FLANN</td><td>42.2</td><td>51.8</td><td>58.1</td><td>55.0 58.1</td><td>17.4</td><td>12.5</td><td>52.0</td><td>57.6</td></tr><tr><td>AKAZE</td><td>BruteForce</td><td>37.5</td><td>48.3</td><td>53.4</td><td>51.7 55.0</td><td>15.4</td><td>11.0</td><td>46.9</td><td>54.9</td></tr><tr><td>BRISK</td><td>FLANN</td><td>49.8</td><td>49.5</td><td>60.8</td><td>54.5 57.0 42.7</td><td>16.6</td><td>30.4</td><td>50.8</td><td>68.6</td></tr><tr><td>BRISK</td><td>BruteForce</td><td>36.6</td><td>41.4</td><td>46.9</td><td>38.7</td><td>5.5</td><td>9.0</td><td>40.0</td><td>59.8</td></tr><tr><td>SP</td><td>SuperGlue</td><td>62.1</td><td>61.1</td><td>63.4</td><td>56.1 59.0</td><td>93.3 93.3</td><td>56.9</td><td>77.0</td><td>77.1</td></tr><tr><td>SP</td><td>LightGlue</td><td>62.2</td><td>61.0</td><td>64.0</td><td>56.4</td><td>57.3</td><td>55.2 32.6</td><td>74.8</td><td>75.5</td></tr><tr><td>RootSIFT</td><td>FLANN</td><td>59.6</td><td>69.4</td><td>74.9</td><td>67.8</td><td>69.5 53.7</td><td></td><td>65.5</td><td>76.1</td></tr><tr><td>RootSIFT</td><td>BruteForce</td><td>43.1</td><td>53.2</td><td>54.8</td><td>51.3</td><td>39.5 30.9</td><td>17.2</td><td>47.1</td><td>70.5</td></tr><tr><td>SP</td><td>FLANN</td><td>54.1</td><td>51.5</td><td>56.6</td><td>48.2 50.0</td><td>83.8</td><td>46.4</td><td>70.7</td><td>71.5</td></tr><tr><td>SP</td><td>BruteForce</td><td>51.2</td><td>52.1</td><td>54.8</td><td>51.2 53.5</td><td>37.5 6.3</td><td>50.9 29.4</td><td>72.1</td><td>68.4</td></tr><tr><td>KLT</td><td>direct</td><td>34.5</td><td>39.8</td><td>41.3</td><td>53.2</td><td>55.7 84.0</td><td></td><td>35.5</td><td>55.7</td></tr><tr><td>KLT</td><td>chained</td><td>75.4</td><td>76.6</td><td>78.0</td><td>84.4</td><td></td><td>61.4</td><td>69.4</td><td>74.9</td></tr></table>

Table 3: Rotation-only Pose AUC@5<sup>◦</sup> [%] on all nine datasets. Bold/underline = best/second. SIFT+FLANN: median of five seeds. KLT rows, tracked as in Table 2, are not marked $( ^ { 6 6 } - ^ { 9 9 } \div$ no sequence to chain through). AUC@5<sup>◦</sup>: area under the cumulative rotation-error curve up to 5<sup>◦</sup>, FLANN: fast library for approximate nearest neighbors, KLT: Kanade–Lucas–Tomasi, SP: SuperPoint.
<table><tr><td rowspan="3">Detector</td><td rowspan="3">Matcher</td><td colspan="4">Synthetic</td><td colspan="2">Lunar testbed</td><td colspan="2">Terrestrial analogue</td></tr><tr><td colspan="3">LuPNT</td><td colspan="2">LuSNAR (9)</td><td rowspan="2">PolarTraverse LunaPolaris</td><td colspan="2">S3LI</td></tr><tr><td>(base)</td><td>(cam.)</td><td>(high sun)</td><td>(5)</td><td></td><td>(Etna)</td><td>(Vulcano)</td></tr><tr><td>SIFT</td><td>FLANN</td><td>79.5</td><td>85.2</td><td>89.7</td><td>92.2 90.2</td><td>47.3</td><td>31.6</td><td>65.0</td><td>76.5</td></tr><tr><td>SIFT</td><td>LightGlue</td><td>81.9</td><td>86.7</td><td>90.3</td><td>90.9 88.4</td><td>53.2</td><td>29.2</td><td>69.9</td><td>78.7</td></tr><tr><td>SP</td><td>SuperGlue</td><td>89.0</td><td>87.2</td><td>89.3</td><td>90.8 90.3</td><td>97.2</td><td>56.9</td><td>77.0</td><td>77.1</td></tr><tr><td>SP</td><td>LightGlue</td><td>88.8</td><td>87.6</td><td>89.7</td><td>91.1 88.4</td><td>97.5</td><td>55.2</td><td>74.8</td><td>75.5</td></tr><tr><td>KLT</td><td>direct</td><td>51.2</td><td>55.6</td><td>57.6</td><td>82.5 82.1 95.6</td><td>24.8</td><td>29.4</td><td>35.5</td><td>55.7</td></tr><tr><td>KLT</td><td>chained</td><td>88.5</td><td>88.2</td><td>90.1</td><td>95.7</td><td></td><td>61.4</td><td>69.4</td><td>74.9</td></tr></table>

![](images/c79b18003eed9e2e50fe0f1214fd3fb920c098e01631a27131f3e34942b729fc.jpg)

![](images/f718e62eae077ab104602ca44b06e722abba502d6fdfa0f526c42e60b2040793.jpg)

![](images/0940a292f937d68057212a98c702c8befa4ea92ea89ae533a064d83f7f41a93c.jpg)

![](images/11dc2486dcfee1db5be6627d7b78d0ba505647a81ea7f758261354dc06e60497.jpg)

![](images/ec2c58a8d959d05376dbe70950a9f25a940ce95c797992f978fe4957304b1eeb.jpg)

![](images/dd7e5c72f14efd5bffd61241b16b33aa213809733c44d6adeb17050c4314f984.jpg)  
Figure 3: Matches of SIFT+FLANN (left) and SuperPoint+LightGlue (right) on one pair per dataset, from top: LunaPolaris (0.34 m, 4.0<sup>◦</sup>), S3LI (Etna) (2.78 m, 29.2<sup>◦</sup>) and LuPNT (base) (1.05 m, 0.9<sup>◦</sup>). Green = RANSAC inliers, red = outliers. FLANN: fast library for approximate nearest neighbors, RANSAC: random sample consensus.

![](images/4ef59242cad0a5f9d3af5b42377ada93dc311306423869d32c3bd0f92b6860b1.jpg)

![](images/bb3600f3dbc55d896102017aef3d7dd3e5c23e1e351b58dbfdd980bea8391af3.jpg)

![](images/cd1d61e88b122c7a2b4559db4321c13db2d02178748b235bed479ebdbcbe6188.jpg)

![](images/ce442f96313bd5cba9ed6058bc9018933a589ffb0015254f25dc988b565bc082.jpg)  
SIFT+FLANN AKAZE+FLANN ORB+FLANN SP+LightGlue KLT direct KLT chained SIFT+BruteForce BRISK+FLANN SP+SuperGlue SIFT+LightGlue

Figure 4: Pose $\mathbf { A U C } @ 5 ^ { \circ }$ [%] against the median translation of each frame step [m, log axis], on LuPNT (base), LuSNAR (5), S3LI (Etna) and LunaPolaris. The two real datasets are scored rotation-only. On the synthetic panels the error includes the translation direction, poorly constrained by the small parallax of the shortest steps, so the AUC first rises with the step. At the widest steps, low overlap also fails the rotation. AUC: area under the curve, FLANN: fast library for approximate nearest neighbors, KLT: Kanade–Lucas–Tomasi, SP: SuperPoint.  
Table 4: Operating range ∆t<sup>∗</sup> [m], the largest median translation with Pose $\mathbf { A U C @ 5 ^ { \circ } > 5 0 ~ \% }$ . Pose error as in Table 2, rotation alone on LunaPolaris and S3LI. “—”: never. Bold/underline = best/second. FLANN cells: median of five seeds, spread ≤ 0.46 m. AUC: area under the curve, FLANN: fast library for approximate nearest neighbors, SP: SuperPoint.
<table><tr><td rowspan="3">Detector Matcher</td><td rowspan="3"></td><td colspan="4">Synthetic</td><td>Testbed</td><td colspan="2">Analogue</td></tr><tr><td colspan="3">LuPNT</td><td colspan="2">LuSNAR (9)</td><td colspan="2">LunaPolaris S3LI</td></tr><tr><td>(base)</td><td>(cam.)</td><td>(high sun)</td><td>(5)</td><td></td><td>(Etna)</td><td>(Vulcano)</td></tr><tr><td>SIFT</td><td>FLANN</td><td>2.3</td><td>4.1</td><td>5.3</td><td>3.7</td><td>0.3</td><td>2.8</td><td>6.2</td></tr><tr><td>SIFT</td><td>BruteForce</td><td>1.3</td><td>1.8</td><td>1.9</td><td>1.8</td><td></td><td>0.9</td><td>3.7</td></tr><tr><td>SIFT</td><td>LightGlue</td><td>3.0</td><td>4.6</td><td>5.3</td><td>3.2</td><td>0.3</td><td>3.8</td><td>6.6</td></tr><tr><td>ORB</td><td>FLANN</td><td></td><td>0.7</td><td>1.1</td><td>0.9</td><td></td><td>0.6</td><td>1.6</td></tr><tr><td>ORB</td><td>BruteForce</td><td></td><td>1.0</td><td>1.1</td><td>1.3</td><td></td><td>0.5</td><td>1.6</td></tr><tr><td>AKAZE</td><td>FLANN</td><td>1.3</td><td>1.9</td><td>2.6</td><td>2.5</td><td></td><td>1.4</td><td>2.5</td></tr><tr><td>AKAZE</td><td>BruteForce</td><td>1.1</td><td>1.6</td><td>1.9</td><td>2.3</td><td></td><td>0.9</td><td>1.9</td></tr><tr><td>BRISK</td><td>FLANN</td><td>2.0</td><td>2.2</td><td>4.1</td><td>2.7</td><td>0.3</td><td>1.5</td><td>4.0</td></tr><tr><td>BRISK</td><td>BruteForce</td><td>1.2</td><td>1.5</td><td>1.6</td><td>1.4</td><td>0.9</td><td>0.7</td><td>2.4</td></tr><tr><td>SP</td><td>SuperGlue</td><td>5.0</td><td>4.8</td><td>5.2</td><td>3.0</td><td></td><td>5.2</td><td>6.3</td></tr><tr><td>SP</td><td>LightGlue</td><td>5.0</td><td>4.8</td><td>5.2</td><td>3.0</td><td></td><td>4.7</td><td>5.3</td></tr></table>

Default detector settings understate the classical detectors on low-texture imagery. On subsamples of LuPNT (base) and LunaPolaris, raising the ORB keypoint cap and lowering the AKAZE detection threshold improves every ORB and AKAZE combination. AKAZE, which detects very few keypoints on LunaPolaris at its default threshold, gains the most. With FLANN matching it overtakes SIFT+FLANN on LunaPolaris but stays below both SuperPoint pairings, and on LuPNT (base) it roughly matches SIFT+FLANN. A classical detector should therefore be tuned to the target imagery before it is judged unsuitable for a rover.

Because the best front end differs between datasets, a front end should be chosen from the results on the dataset closest to the target terrain (Table 2) and at the rover’s frame spacing (Fig. 4), not extrapolated from the synthetic results.

## Accuracy Versus Baseline

Fig. 4 and Table 4 show how pose accuracy depends on the baseline. Accuracy is limited by the translation direction at short baselines, which barely constrain it, and by overlap at long ones, which leave few shared features. Both limits are geometric, so we expect them on other terrain as well. Binned by overlap, the curves mirror Fig. 4, since overlap falls as the baseline grows. Where the SuperPoint pairings lead, their advantage lies at low to moderate overlap and generally narrows at high overlap, disappearing on LunaPolaris and reversing on LuPNT (base). On S3LI (Etna) it holds in every bin below 60 % overlap.

Overlap and rotation as difficulty axes—In the decile test, overlap explains the most variance on average and on three of the five datasets. ∆θ comes close but varies most across datasets, and α¯ is weakest, so ω is our reference axis.

Per-step accuracy—On LuPNT (base), joint Pose AUC@5◦ peaks at intermediate baselines (Fig. 4). At the shortest step, SIFT+FLANN and SP+LightGlue recover the rotation almost perfectly but often miss the translation direction. At the widest step, low overlap makes the rotation fail as well. Scored separately, rotation accuracy is nearly equal up to about 1 m on LuPNT (base) and LuSNAR (5), so the joint Pose AUC lead of SIFT+FLANN comes from the translation direction. This lead reverses beyond 2 m on LuPNT (base), where SIFT also loses rotation accuracy, but holds at every tested baseline on LuSNAR (5).

When KLT tracks through every intermediate frame, it is the most accurate front end on all five synthetic datasets under the joint metric (Table 2), because each hop between consecutive frames is short. On LunaPolaris it is slightly above the SuperPoint pairings, by a margin we do not test for resolution. On real outdoor terrain it falls below both SuperPoint pairings on S3LI (Etna) and stays close to the other combinations on S3LI (Vulcano) (Table 3). Tracked directly between the two frames of a pair, it is less accurate than SIFT+FLANN and both SuperPoint pairings on every dataset, and on LuPNT (base) it collapses at wider steps. Chaining needs every intermediate frame, so it cannot skip frames to save computation, and it cannot verify a revisit, which has no intermediate frames.

On real imagery, the ratio of the best learned to the best classical accuracy grows with the baseline, to about two and a half times on S3LI (Etna) and nearly eight times on LunaPolaris at the widest (Fig. 4). On S3LI (Etna) the two are nearly equal up to about 1 m of translation, while on LunaPolaris it opens from the second step. On low-texture imagery such as LunaPolaris, learned front ends thus help even between closely spaced frames.

Operating range—The operating range ∆t∗ (Eq. (5)) shows the SuperPoint advantage more clearly than the pool averages. On LuPNT (base) and S3LI (Etna), the SuperPoint pairings keep 50 % Pose AUC@5◦ roughly twice as far as any classical combination, and on LunaPolaris about three times as far as any other combination. SIFT-based combinations have the longest range on the LuSNAR scenes, LuPNT (high sun) and S3LI (Vulcano). Because the baseline between matched frames is the driving speed divided by the matching rate, a longer operating range would let a rover drive faster or match fewer frames, although we do not test whether skipping frames preserves VO performance.

## Camera Effects, Lighting and Cross-Dataset Transfer

Four comparisons change the appearance of the imagery: LuPNT with camera effects or a higher sun, LuPNT (base) against S3LI (Etna), and three lightings along one LunaPolaris trajectory (Table 5 and Fig. 5). LuPNT (base) against S3LI (Etna) also changes the scene and pair geometry, so it does not isolate appearance. Lighting affects front ends with classical detectors more than the SuperPoint pairings, so a front end should be evaluated under the illumination expected during operations.

Camera effects lower EP but not classical pose accuracy. On pairs with the same frame indices in both collections, LuPNT (cam.) has lower mean EP than LuPNT (base) for every combination except the two ORB pairings. It has higher joint Pose AUC@5◦ for all but BRISK+FLANN, which is unchanged, and the SuperPoint pairings, which lose slightly. The camera-effects run retraces the base path within a few centimeters, so the differences come from the camera effects rather than pair geometry, although it is unclear why Pose AUC rises while EP falls. Modeled camera effects of this kind therefore do not by themselves reproduce the accuracy loss seen on the real datasets.

Every combination is less accurate on S3LI (Etna) than on

LuPNT (base) in rotation-only Pose AUC@5◦. The FLANN combinations already trail at the shortest step, so the deficit is not only a wide-baseline effect. The two pools also differ in pair geometry, scene, resolution and ground truth, so we do not attribute it to appearance alone. Absolute accuracy measured in simulation is therefore likely optimistic for a rover on real terrain, whatever the cause.

LunaPolaris sequences 01 (ceiling lights), 07 (a single low spotlight) and 10 (the spotlight and a rover headlight) follow one trajectory. The spotlight raises the RMS image contrast by almost half and saturates a small fraction of the pixels. It shadows cover about an eighth of the image on 07 but almost none on 10, which adds the headlight.

The full pools (Table 5) hide the spotlight’s gain for the combinations with classical detectors by mixing lighting with geometry. A frame step spans a longer baseline on 07, where several combinations, including the SuperPoint pairings, are less accurate. On pairs matched in pose across the three sequences, every combination with a classical detector gains significantly under both spotlight lightings, also at a looser pose tolerance, while the SuperPoint pairings show no resolved change, so the gap to SuperPoint narrows. The significant SuperPoint fall on the full 07 pool is consistent with the longer baselines there.

Under the spotlight every detector returns more keypoints and SIFT+FLANN keeps more matches, although few keypoints fall in the shadows. On the screened pose-matched pairs (one shown in Fig. 5), SIFT+FLANN recovers the rotation within 5◦ on just over half the pairs under the ceiling lights but on nearly all under the spotlight, while SP+LightGlue does so on nearly every pair under each lighting. This is consistent with low-elevation light shading the relief into texture the classical detectors can use, whereas diffuse light gives flat images and deep shadow shows almost none. On LuPNT, a 45◦ sun also raises the accuracy of the combinations with classical detectors while the SuperPoint pairings barely change, although camera effects at 15◦ already give most of this gain for all but BRISK (Table 2). On the Moon, where sunlight is directional, this suggests classical matching is weakest in deep shadow and under a near-overhead sun that flattens the shading, neither of which we isolate.

## Visual Odometry

Pairwise accuracy differences carry over to VO (Table 6) only where the front end is stressed. Elsewhere, with metric depth from ground truth or stereo, nearly every front end tracks accurately. With ground-truth depth, about half the ATE differences between front ends are below the variation between repeated runs. Three differences between VO and the pairwise evaluation are consistent with this. First, VO matches only consecutive frames, where both families recover rotation almost equally. Second, metric depth makes PnP well posed at zero baseline, removing the translation-direction ambiguity that dominates pairwise error at short steps. Third, VO loses a frame only when too few matches or inliers with depth remain, so match support between consecutive frames matters more than wide-baseline accuracy.

Visual odometry with ground-truth depth—PnP on groundtruth depth gives metric VO on the synthetic datasets, and all eleven combinations stay at or below about 0.7 % ATE/L.

With ground-truth depth, only ORB+FLANN stands out among the tabulated combinations. It has the highest ATE in every ground-truth-depth column, by more than its seed spread everywhere except LuPNT (high sun) and LuSNAR (5). The larger LuSNAR leads that involve a SuperPoint run are within the variation between GPU reruns and not resolved. The weakness of ORB+FLANN is consistent with the default ORB keypoint cap, since raising it improves every ORB combination in the pairwise test. Beyond this, groundtruth-depth VO barely separates the front ends.

Table 5: Rotation-only Pose AUC@5<sup>◦</sup> [%] and stereo VO ATE/L [%] on LunaPolaris sequences 01 (ceiling lights), 07 (low spotlight) and 10 (low spotlight and headlight). Marks as in Table 2. Full pools, not pose-matched. FLANN cells: median of five seeds. <sup>‡</sup>Stopped tracking: path coverage. Sequence 01 repeats the LunaPolaris columns of Tables 2, 3 and 6. ATE: absolute trajectory error, ATE/L: ATE as a percentage of path length, AUC: area under the curve, FLANN: fast library for approximate nearest neighbors, SP: SuperPoint, VO: visual odometry.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Matcher</td><td colspan="3">Pose AUC@5° [%]</td><td colspan="3">Stereo VO ATE/L [%]</td></tr><tr><td>Seq.01</td><td>Seq.07</td><td>Seq. 10</td><td>Seq.01</td><td>Seq.07</td><td>Seq.10</td></tr><tr><td>SIFT</td><td>FLANN</td><td>31.6</td><td>30.3</td><td>42.1</td><td>2%</td><td>0.6</td><td>0.8</td></tr><tr><td>SIFT</td><td>BruteForce</td><td>15.0</td><td>20.3</td><td>25.9</td><td>&lt;1%‡</td><td>11%‡</td><td>71%</td></tr><tr><td>SIFT</td><td>LightGlue</td><td>29.2</td><td>37.4</td><td>49.3</td><td>2%</td><td>0.4</td><td>1.1</td></tr><tr><td>ORB</td><td>FLANN</td><td>9.9</td><td>8.7</td><td>14.3</td><td>13 %‡</td><td>9%</td><td>2.1</td></tr><tr><td>ORB AKAZE</td><td>BruteForce FLANN</td><td>6.7 12.5</td><td>8.6 21.6</td><td>14.5 28.2</td><td>13% &lt;1%$</td><td>9% 0.6</td><td>69%</td></tr><tr><td>AKAZE</td><td>BruteForce</td><td>11.0</td><td>19.9</td><td>26.4</td><td>&lt;1%‡</td><td>43 %</td><td>0.5 0.5</td></tr><tr><td>BRISK</td><td>FLANN</td><td>30.4</td><td>26.7</td><td>36.8</td><td>38%</td><td>0.4</td><td>0.5</td></tr><tr><td>BRISK</td><td>BruteForce</td><td>9.0</td><td>17.7</td><td>19.8</td><td>1%</td><td>56%</td><td>69%</td></tr><tr><td>SP</td><td>SuperGlue</td><td>56.9</td><td>41.7</td><td>54.1</td><td>1.3</td><td>0.6</td><td>0.6</td></tr><tr><td>SP</td><td>LightGlue</td><td>55.2</td><td>41.4</td><td>53.6</td><td>1.6</td><td>0.5</td><td>0.5</td></tr></table>

![](images/177c7102a677bf82115f313d3bf2432e1871a8a03fb58e1f6f7d82e311b2616e.jpg)  
Figure 5: Matches of SIFT+FLANN (left) and SuperPoint+LightGlue (right), with inlier counts and rotation errors, on a 0.34 m pair from sequence 01 (top) and its pose-matched counterparts in 07 (middle) and 10 (bottom). Green = RANSAC inliers, red = outliers. FLANN: fast library for approximate nearest neighbors, RANSAC: random sample consensus, SP: SuperPoint.

Front ends that stay accurate over longer baselines also tend to have lower ATE/L. With ground-truth depth, the Spearman correlation between operating range and ATE/L over the synthetic combinations and datasets is ρ = −0.56. With stereo depth it weakens to ρ = −0.41, because the VO error of the learned front ends no longer follows their pairwise accuracy. A short operating range thus flags a front end likely to drift, but the pairwise benchmark is a screen, not a substitute for running VO. The correlation is in-sample and sensitive to the 50 % threshold, so it indicates a trend rather than a calibrated prediction.

Stereo visual odometry—On LunaPolaris, the lidar reference depth only describes pair geometry, not stereo VO.

On the S3LI sites stereo VO is mostly a tie, whereas on LunaPolaris only the SuperPoint pairings stay on track. On Etna every tabulated combination tracks every frame, all but ORB+FLANN tie, and every trajectory stays within 0.8 m RMS of the ground truth (Fig. 6). On Vulcano, ORB+FLANN and AKAZE+FLANN stop within the first 1 % of the path, where the camera pitches toward the sea and sky, and the others tie. On LunaPolaris sequence 01 every other combination stops within 38 % of the path. Under the spotlight (Table 5), SIFT+FLANN, SIFT+LightGlue, BRISK+FLANN and AKAZE+FLANN complete both paths, ORB+FLANN and AKAZE+BruteForce only that of 10, and the other brute-force combinations stop on both. This suggests that on low-texture terrain VO with classical detectors is limited by how much texture the lighting reveals, whereas the SuperPoint pairings track under all three lightings.

LuPNT (base) BRISK+FLANN (0.33 m) ORB+FLANN (4.68 m)  
LunaPolaris (01)  
LuSNAR (5) SIFT+FLANN (0.34 m) ORB+FLANN (3.34 m)  
S3LI (Vulcano) SIFT+FLANN (0.8 m) SP+LightGlue (0.9 m)  
![](images/79590028331dab3aac5ee94b41c65fef239f9a0af2af4cd48d71f8b6181bb716.jpg)  
X [m]

![](images/eee431a8ba77a6a20723608f376fb43d151b4d94731020527ea4839180c973aa.jpg)

![](images/03d97568c49cbe4afaa6f34cee45570b6c6360a4b164ea07025aca55a9b25451.jpg)  
X [m]

![](images/86b2d9de28d4a15aed3390bdef35bc1aa2965e6a7a62b64179b4e2745b5be64a.jpg)  
X [m]

![](images/b0fd2af2bd8dec5c7401c1bd9403357ac9f0396be579bd7e1907feb6755890f3.jpg)  
X [m]  
GT lowest ATE highest ATE Start  
Figure 6: Top-down stereo VO trajectories, SE(3)-aligned to GT, of the lowest (solid) and highest (dashed) ATE combination that tracks the whole path, on five datasets covering the three tiers. The S3LI sites show their scored first 10 %. ATE: absolute trajectory error, FLANN: fast library for approximate nearest neighbors, GT: ground truth, SP: SuperPoint, VO: visual odometry.

Table 6: VO ATE [m] with stereo depth on every dataset and with GT depth on the synthetic datasets. FLANN cells: median of five seeds (median spread 17 %). <sup>‡</sup>Stopped tracking: path coverage. Tracked: datasets with ≥ 90 % path coverage in stereo VO. Bold/underline = best/second per column and block. ATE: absolute trajectory error, FLANN: fast library for approximate nearest neighbors, GT: ground truth, SP: SuperPoint, VO: visual odometry.
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="5">Synthetic</td><td rowspan="2">Testbed</td><td colspan="2">Analogue</td><td rowspan="3"></td></tr><tr><td colspan="3">LuPNT</td><td colspan="2">LuSNAR</td><td colspan="2">LunaPolaris S3LI</td></tr><tr><td>(base)</td><td>(cam.)</td><td>(high sun)</td><td>(5)</td><td>(9)</td><td>(Etna)</td><td>(Vulcano)</td></tr><tr><td colspan="10">Stereo VO</td></tr><tr><td>SIFT</td><td>FLANN</td><td>0.56</td><td>1.21</td><td>0.05</td><td>0.34</td><td>0.30</td><td>2%</td><td>0.5</td><td>0.8</td><td>7/8</td></tr><tr><td>SIFT</td><td>LightGlue</td><td>0.66</td><td>1.20</td><td>0.04</td><td>0.42</td><td>0.28</td><td>2%</td><td>0.4</td><td>0.8</td><td>7/8</td></tr><tr><td>ORB</td><td>FLANN</td><td>4.68</td><td>2.94</td><td>1.28</td><td>3.34</td><td>1.92</td><td>13%</td><td>0.7</td><td>1%</td><td>6/8</td></tr><tr><td>AKAZE</td><td>FLANN</td><td>0.94</td><td>1.31</td><td>0.05</td><td>0.35</td><td>0.47</td><td>&lt;1%</td><td>0.5</td><td>1%</td><td>6/8</td></tr><tr><td>BRISK</td><td>FLANN</td><td>0.33</td><td>1.45</td><td>0.15</td><td>0.55</td><td>1.32</td><td>38%</td><td>0.6</td><td>0.9</td><td>7/8</td></tr><tr><td>SP</td><td>SuperGlue</td><td>0.68</td><td>1.66</td><td>0.65</td><td>0.47</td><td>0.92</td><td>0.4</td><td>0.4</td><td>0.9</td><td>8/8</td></tr><tr><td>SP</td><td>LightGlue</td><td>0.93</td><td>1.54</td><td>0.61</td><td>1.24</td><td>0.77</td><td>0.4</td><td>0.4</td><td>0.9</td><td>8/8</td></tr><tr><td colspan="10">GT-depth VO</td></tr><tr><td>SIFT</td><td>FLANN</td><td>0.50</td><td>1.00</td><td>0.14</td><td>0.31</td><td>1.09</td><td></td><td></td><td></td><td></td></tr><tr><td>SIFT</td><td>LightGlue</td><td>0.55</td><td>1.00</td><td>0.15</td><td>0.37</td><td>1.07</td><td></td><td></td><td></td><td></td></tr><tr><td>ORB</td><td>FLANN</td><td>2.47</td><td>1.36</td><td>0.92</td><td>2.19</td><td>2.62</td><td></td><td></td><td></td><td></td></tr><tr><td>AKAZE</td><td>FLANN</td><td>1.14</td><td>1.07</td><td>0.29</td><td>0.32</td><td>0.86</td><td></td><td></td><td></td><td></td></tr><tr><td>BRISK</td><td>FLANN</td><td>0.53</td><td>1.05</td><td>0.17</td><td>0.58</td><td>1.40</td><td></td><td></td><td></td><td></td></tr><tr><td>SP SP</td><td>SuperGlue</td><td>0.39 0.28</td><td>0.88</td><td>0.31 0.26</td><td>0.87</td><td>0.47</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>LightGlue</td><td></td><td>0.91</td><td></td><td>0.61</td><td>1.10</td><td></td><td></td><td></td><td></td></tr></table>

Across the eight datasets, only the SuperPoint pairings keep tracking on all of them in stereo VO (Table 6, Tracked). Front-end selection therefore depends on the task. Frameto-frame VO needs enough reliable matches to keep tracking, while wider-baseline localization also needs an accurate relative pose, so pairwise accuracy should be read alongside tracking coverage.

With stereo depth, the SuperPoint pairings lose the unresolved lead they hold with ground-truth depth on three of the synthetic datasets (Table 6), and show no resolved VO accuracy advantage over the SIFT pairings. Every combination tracks the whole path to within a few meters (Fig. 6). On LuPNT the stereo ATE of the SuperPoint pairings is 1.7 to 3.3 times that with ground-truth depth, consistent with stereo depth error, rather than matching, limiting their accuracy. The VO advantage of SuperPoint therefore lies in staying on track, as on LunaPolaris, rather than in trajectory accuracy.

## Loop Closure

No evaluated front end handles every kind of revisit. The two tested combinations rarely recover the pose on revisits seen from widely different directions, even with partial overlap.

Revisit structure—Revisits are counted at R = 1 m (R = 2 m for LuPNT (base), which otherwise has no revisits). Four datasets have fewer than three revisit components. LuS-NAR (5) has a single crossing, LuSNAR (9) has no revisit within 10 m, S3LI (Vulcano) has two, and the revisits of LunaPolaris, where the rover laps an arena of about 3 by 6 m (Fig. 6), overlap along the whole path and join into a single component of places that cannot be separated. Retrieval is therefore evaluated on LuPNT (base) and S3LI (Etna), with S3LI (Vulcano) exploratory.

Revisit rotation varies widely between datasets (Table 7). On both S3LI sites most revisit pairs exceed 30◦, so a viewingangle limit as in MSLS [64] would discard many of them.

![](images/64ef4181a1fa1d8e7981c531e5c4d6f97522d0619d605626c66b2fe15aee52b6.jpg)  
Figure 7: Loop-closure examples. (a) Matches of SIFT+FLANN (left) and SuperPoint+LightGlue (right), with RANSAC inliers (green), outliers (red) and rotation and translation-direction errors $e _ { R } , e _ { t } .$ The pairs are a same-direction revisit on LuPNT (base) and, on LuSNAR (5), a pair seen from directions $\mathbf { 1 2 3 } ^ { \circ }$ apart that shares 40 % of the view. (b) S3LI (Etna) DINOv2 retrieval: one query whose top-1 is a true revisit (left) and one whose top-1 is a false proposal (right). Top-down maps at far right. FLANN: fast library for approximate nearest neighbors, RANSAC: random sample consensus, SP: SuperPoint.

Table 7: Revisit structure (at R = 1 m, 2 m on LuPNT (base)) and DINOv2 retrieval within radius R. Rows with fewer than three components are exploratory. S3LI frames with GNSS jumps are excluded. $\stackrel { 6 6 } { \longrightarrow } \stackrel { 9 9 } { = } { \bf n 0 }$ recall at 100 % precision. AUPRC: area under the precision-recall curve, Comp.: revisit components (revisited places), R@1: recall of the top match, R@100P: recall before the first false positive.
<table><tr><td rowspan="2">Dataset</td><td colspan="5">Revisits</td><td colspan="7">Retrieval</td></tr><tr><td>Pairs</td><td>Comp.</td><td>Median Fraction  $\pmb { \Delta \theta [ ^ { \circ } ] }$ </td><td> $< 3 0 ^ { \circ }$ </td><td>Type</td><td>R Queries Comp.</td><td></td><td>R@1 (causal)</td><td>R@1 (offline)</td><td>AUPRC (causal)</td><td>AUPRC (ofline)</td><td>R@100P (causal)</td></tr><tr><td rowspan="2">LuPNT (base)</td><td rowspan="2">596</td><td rowspan="2">3</td><td rowspan="2">5.7</td><td rowspan="2">1.00</td><td rowspan="2">low-angle</td><td>32</td><td>3</td><td>0.250</td><td>0.212</td><td>0.023</td><td>0.017</td><td></td></tr><tr><td>2m 5m</td><td></td><td>0.324</td><td>0.306</td><td>0.082</td><td>0.080</td><td></td></tr><tr><td rowspan="3">S3LI (Etna)</td><td rowspan="3">220,069</td><td rowspan="3"></td><td rowspan="3">41.3</td><td rowspan="3">0.41</td><td rowspan="3"></td><td>450 246</td><td>1 5</td><td>0.142</td><td>0.124</td><td>0.012</td><td>0.010</td><td></td></tr><tr><td>1 m</td><td>423</td><td>0.232</td><td>0.232</td><td>0.064</td><td>0.066</td><td></td></tr><tr><td>2m 5m</td><td>630</td><td>0.357</td><td>0.359</td><td>0.195</td><td>0.193</td><td>0.014</td></tr><tr><td rowspan="3">S3LI (Vulcano)</td><td rowspan="3">83,027</td><td rowspan="3">2</td><td rowspan="3">62.0</td><td rowspan="3">0.36</td><td rowspan="3">wide-angle</td><td>1 m 632</td><td>3 2</td><td>0.123</td><td>0.142</td><td>0.037</td><td>0.045</td><td>0.002</td></tr><tr><td></td><td>968</td><td>0.230</td><td>0.302</td><td>0.123</td><td>0.187</td><td>0.014</td></tr><tr><td>2m 5m</td><td>1348</td><td>0.322</td><td>0.436</td><td>0.196</td><td>0.308</td><td>0.061</td></tr></table>

Pose recovery at revisits—On the same-direction retraces of LuPNT (base), both SuperPoint pairings reach about five times the Pose AUC@5◦ of SIFT+FLANN, the best classical combination (Fig. 7, top row). These three recover the rotation alone far more often, so SIFT+FLANN, whose candidate matches RANSAC mostly rejects, loses mainly the translation direction. The other classical combinations also lose the rotation. For a rover retracing its path, this suggests a SuperPoint front end would supply many more accurate loopclosure constraints than a classical one.

Being close in position does not make two frames a usable revisit. The revisit pairs of LuSNAR (5) all come from one crossing where the headings differ by 123◦, so the two views share no overlap to match. Near that crossing, frames several meters apart share part of the view but from directions too different to match, and both SIFT+FLANN and SP+LightGlue almost never recover the pose (Fig. 7, second row). A radiusonly definition still counts both kinds of pair as revisits, so loop-closure evaluation should define revisits by overlap and viewing direction, not position alone.

Retrieval—R@100P is below 0.07 in every row (Table 7), so proposals from this descriptor need verification before they enter a pose graph. Fig. 7(b) shows why: a false proposal 70 m from its query, on visually similar terrain, scores almost as high as a true revisit. Lunar terrain is likely at least as self-similar as these volcanic slopes, so we expect the same near-duplicate proposals there. A DINOv3 [84] CLS-token descriptor has lower AUPRC in all eight rows, so we report DINOv2. AUPRC measures how well true revisits rank above false proposals, which corrupt a pose graph. The CLS token, rather than the terrain, may be the limit, since AnyLoc’s patch-token VLAD aggregation on the same backbone improves recall@1 over it in unstructured environments [85]. These results do not show that appearance-based retrieval fails on lunar terrain.

![](images/e6b73c8e3dce933112177fd801c2e2a68290fc0ede0beadf6f3fa4cd028ee19a.jpg)  
Figure 8: Incremental cost per frame [ms] on LuPNT (base). Hatched bars run on the GPU. CPU: central processing unit, FLANN: fast library for approximate nearest neighbors, GPU: graphics processing unit.

## Computational Cost

SP+LightGlue is the fastest learned combination (Fig. 8) and, at 16 ms, faster than every timed classical combination except ORB+FLANN, because CPU extraction dominates the classical cost. It is about three times as fast as the SuperGlue pairing, as LightGlue exits early and prunes points [42]. On this hardware SP+LightGlue costs no more per frame than most classical ones, although rover-class processors, which we do not test, could differ. Retrieval adds one DINOv2 descriptor per keyframe, about the cost of one SP+LightGlue frame, plus a negligible similarity search.

## 6. CONCLUSION

Lunar rovers will localize with cameras, yet feature matchers are developed on terrestrial scenes and planetary work compares them on orbital registration or within complete SLAM systems, not between rover frames. We therefore benchmark eleven detector–matcher combinations on nine synthetic, lunar-testbed and terrestrial-analogue datasets, and a KLT tracker, scoring the pose accuracy of image pairs against their baseline and the trajectory error of the resulting VO.

Scored by rotation alone, the SuperPoint pairings outperform SIFT+FLANN on both lunar testbeds and one analogue site and stay accurate out to longer baselines, while the differences on the synthetic datasets are smaller or unresolved. Scored on rotation and translation direction jointly, SIFT+FLANN exceeds both SuperPoint pairings on four synthetic datasets, because its translation direction is more accurate at short baselines. Synthetic results alone therefore do not establish relative performance on real rover imagery. Modeled camera effects lower EP but not classical Pose AUC, so they do not reproduce the loss seen on real imagery. The ablation attributes most of the testbed lead to the SuperPoint detector and descriptor rather than the learned matcher. A KLT tracker chained through every frame is the most accurate front end on the synthetic datasets under the joint metric and slightly above the SuperPoint pairings on LunaPolaris, but not at the Etna analogue site, and it cannot skip frames or verify revisits.

In VO, the front ends differ more in whether they stay on track than in accuracy. With ground-truth depth, about half their ATE differences are within run-to-run variation, and with stereo depth no consistent SuperPoint advantage is resolved, but only the SuperPoint pairings keep tracking on all eight sequential datasets, including under the diffuse ceiling lighting of LunaPolaris. Pairings with classical detectors depend on lighting while SuperPoint changes little: a low spotlight makes them more accurate than diffuse light on the same testbed views, consistent with more contrast and detected keypoints. SuperPoint thus likely matters most on uniformly lit regolith, such as under a near-overhead sun, and possibly in deep polar shadow. At revisits, SuperPoint reaches about five times the Pose AUC of SIFT+FLANN on same-direction retraces, but SIFT+FLANN and SP+LightGlue almost never recover the pose between views from widely different directions, even with partial overlap, and DINOv2 retrieval proposals need geometric verification.

For a rover with a GPU, we recommend SuperPoint with LightGlue, the faster SuperPoint pairing, about as accurate as SuperGlue. Without a GPU, SIFT+FLANN has the highest classical Pose AUC at default settings, although AKAZE with a lower detection threshold surpasses it on a LunaPolaris subsample, and where it keeps tracking, its VO is not consistently less accurate than that of SuperPoint.

Future work will tune the classical detectors per dataset, sweep the sun elevation in simulation, select pairs by measured geometry, adapt the learned matchers to analogue imagery, evaluate complete SLAM on revisits defined by overlap, and build a camera-independent reference for S3LI. We will also extend the benchmark to real lunar surface imagery from Chang’e-3 and Apollo.

## ACKNOWLEDGEMENTS

We gratefully acknowledge Blue Origin for funding this project and for valuable discussions that contributed to its development. This manuscript benefited from the use of AI-based assistants, Claude Opus and Claude Sonnet, which were employed for coding assistance and revising. All final content was reviewed and edited by the authors.

## REFERENCES

[1] S. Bhamidipati, J. Nash, D. Atha, S. Daftry, and I. A. Nesnas, “A trade study of optical global localization approaches for lunar rovers,” in Proceedings of the 39th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2026), Orlando, FL, 2026.

[2] O. Toupet, M. Ono, T. Del Sesto, M. Maimone, and M. McHenry, “Enhanced autonomous navigation on the Perseverance Mars rover,” IEEE Transactions on Field Robotics, vol. 3, pp. 83–124, 2026.

[3] S. Daftry et al., “LunarNav: Crater-based localization for long-range autonomous lunar rover navigation,” in 2023 IEEE Aerospace Conference, 2023, pp. 1–15.

[4] J. Nash et al., “Censible: A robust and practical global localization framework for planetary surface missions,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), May 2024, pp. 8642–8648.

[5] NASA Jet Propulsion Laboratory, “NASA’s Perseverance now autonomously pinpoints its location

on Mars,” NASA news release, 18 February 2026, Feb. 2026, accessed 2026-10-01. [Online]. Available: https://www.nasa.gov/missions/mars-2020-persevera nce/perseverance-rover/nasas-perseverance-now-auton omously-pinpoints-its-location-on-mars/

[6] D. J. Israel et al., “LunaNet: a flexible and extensible lunar exploration communications and navigation infrastructure,” in 2020 IEEE Aerospace Conference, 2020, pp. 1–14.

[7] D. Scaramuzza and F. Fraundorfer, “Visual odometry [tutorial]. part I: The first 30 years and fundamentals,” IEEE Robotics & Automation Magazine, vol. 18, no. 4, pp. 80–92, 2011.

[8] C. Cadena et al., “Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age,” IEEE Transactions on Robotics, vol. 32, no. 6, pp. 1309–1332, 2016.

[9] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, 2004.

[10] D. Barath, J. Matas, and J. Noskova, “MAGSAC: Marginalizing sample consensus,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 10 189–10 197.

[11] D. Barath, J. Noskova, M. Ivashechkin, and J. Matas, “MAGSAC++, a fast, reliable and accurate robust estimator,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1301–1309.

[12] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperGlue: Learning feature matching with graph neural networks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 4937–4946.

[13] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detector-free local feature matching with transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8918–8927.

[14] Y. Jin et al., “Image matching across wide baselines: From paper to practice,” International Journal of Computer Vision, vol. 129, no. 2, pp. 517–547, 2021.

[15] G. Casadesus Vila, K. Iiyama, and G. Gao, “LuPNT: An Open-Source Simulator for Lunar Communications, Positioning, Navigation, and Timing,” in 2025 IEEE Aerospace Conference, Mar. 2025, pp. 1–18.

[16] F. Fraundorfer and D. Scaramuzza, “Visual odometry: Part II: Matching, robustness, optimization, and applications,” IEEE Robotics & Automation Magazine, vol. 19, no. 2, pp. 78–90, 2012.

[17] A. Macario Barros, M. Michel, Y. Moline, G. Corre, and F. Carrel, “A comprehensive survey of visual SLAM algorithms,” Robotics, vol. 11, no. 1, p. 24, 2022.

[18] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012, pp. 573–580.

[19] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the KITTI vision benchmark suite,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3354–3361.

[20] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos,´ “ORB-SLAM: A versatile and accurate monocular SLAM system,” IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147–1163, 2015.

[21] C. Forster, M. Pizzoli, and D. Scaramuzza, “SVO: Fast semi-direct monocular visual odometry,” in 2014 IEEE International Conference on Robotics and Automation (ICRA), 2014, pp. 15–22.

[22] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3, pp. 611–625, 2018.

[23] J. Engel, T. Schops, and D. Cremers, “LSD-SLAM:¨ Large-scale direct monocular SLAM,” in Computer Vision – ECCV 2014, ser. Lecture Notes in Computer Science, vol. 8690. Springer, 2014, pp. 834–849.

[24] X. Gao, R. Wang, N. Demmel, and D. Cremers, “LDSO: Direct sparse odometry with loop closure,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 2198–2204.

[25] Z. Teed and J. Deng, “DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 16 558–16 569.

[26] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 39 033– 39 051.

[27] W. Wang, Y. Hu, and S. Scherer, “TartanVO: A generalizable learning-based VO,” in Proceedings of the 2020 Conference on Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 155, 2021, pp. 1761–1772. [Online]. Available: https://proceedings.mlr.press/v155/wang21h.html

[28] D. G. Lowe, “Distinctive image features from scaleinvariant keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91–110, 2004.

[29] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” in 2011 International Conference on Computer Vision. IEEE, 2011, pp. 2564–2571.

[30] P. F. Alcantarilla, J. Nuevo, and A. Bartoli, “Fast explicit diffusion for accelerated features in nonlinear scale spaces,” in Proceedings of the British Machine Vision Conference (BMVC), 2013.

[31] S. Leutenegger, M. Chli, and R. Y. Siegwart, “BRISK: Binary robust invariant scalable keypoints,” in 2011 International Conference on Computer Vision. IEEE, 2011, pp. 2548–2555.

[32] C. Campos, R. Elvira, J. J. Gomez Rodr´ ´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: An accurate´ open-source library for visual, visual-inertial and multimap SLAM,” IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, 2021.

[33] A. Barroso-Laguna, E. Riba, D. Ponsa, and K. Mikolajczyk, “Key.Net: Keypoint detection by handcrafted and learned CNN filters,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[34] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Super-Point: Self-supervised interest point detection and description,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2018, pp. 337–349.

[35] M. Dusmanu et al., “D2-Net: A trainable CNN for joint description and detection of local features,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 8084–8093.

[36] J. Revaud, C. De Souza, M. Humenberger, and P. Weinzaepfel, “R2D2: Reliable and repeatable detector and descriptor,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[37] M. J. Tyszkiewicz, P. Fua, and E. Trulls, “DISK: Learning local features with policy gradient,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 14 254–14 265.

[38] X. Zhao, X. Wu, W. Chen, P. C. Y. Chen, Q. Xu, and Z. Li, “ALIKED: A lighter keypoint and descriptor extraction network via deformable transformation,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–16, 2023.

[39] G. Potje, F. Cadar, A. Araujo, R. Martins, and E. R. Nascimento, “XFeat: Accelerated features for lightweight image matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 2682–2691.

[40] M. Muja and D. G. Lowe, “Fast approximate nearest neighbors with automatic algorithm configuration,” in Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), 2009, pp. 331–340.

[41] R. Sinkhorn and P. Knopp, “Concerning nonnegative matrices and doubly stochastic matrices,” Pacific Journal ofMathematics, vol. 21, no. 2, pp. 343–348, 1967.

[42] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “Light-Glue: Local feature matching at light speed,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 17 581–17 592.

[43] Z. Li and N. Snavely, “MegaDepth: Learning singleview depth prediction from internet photos,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 2041–2050.

[44] S. Xu, S. Chen, R. Xu, C. Wang, P. Lu, and L. Guo, “Local feature matching using deep learning: A survey,” Information Fusion, vol. 107, p. 102344, 2024.

[45] H. Chen et al., “ASpanFormer: Detector-free image matching with adaptive span transformer,” in European Conference on Computer Vision (ECCV), 2022, pp. 20– 36.

[46] Q. Wang, J. Zhang, K. Yang, K. Peng, and R. Stiefelhagen, “MatchFormer: Interleaving attention in transformers for feature matching,” in Asian Conference on Computer Vision (ACCV), 2022, pp. 256–273.

[47] J. Edstedt, I. Athanasiadis, M. Wadenback, and M. Fels-¨ berg, “DKM: Dense kernelized feature matching for geometry estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 17 765–17 775.

[48] J. Edstedt, Q. Sun, G. Bokman, M. Wadenb¨ ack, and¨ M. Felsberg, “RoMa: Robust dense feature matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 19 790–19 800.

[49] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D vision made easy,” in Proceedings of the IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition (CVPR), 2024, pp. 20 697–20 709.

[50] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in Computer Vision – ECCV 2024, 2024, pp. 71–91.

[51] M. Maimone, Y. Cheng, and L. Matthies, “Two years of visual odometry on the Mars Exploration Rovers,” Journal of Field Robotics, vol. 24, no. 3, pp. 169–186, 2007.

[52] A. E. Johnson et al., “Mars 2020 lander vision system flight performance,” in AIAA SciTech 2022 Forum, 2022.

[53] G. Mollica, M. Legittimo, A. Dionigi, G. Costante, and P. Valigi, “Integrating sparse learning-based feature detectors into simultaneous localization and mapping a benchmark study,” Sensors, vol. 23, no. 4, p. 2286, 2023.

[54] V. Romero-Bautista, L. Altamirano-Robles, R. D´ıaz-Hernandez, S. Zapotecas-Mart´ ´ınez, and N. Sanchez-Medel, “Evaluation of visual SLAM algorithms in unstructured planetary-like and agricultural environments,” Pattern Recognition Letters, vol. 186, pp. 106– 112, 2024.

[55] R. Makharia, J. G. Singla, Amitabh, N. Dube, and H. Sharma, “Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data,” 2025.

[56] M. B. Azhari and D. H. Shim, “DINO-VO: A featurebased visual odometry leveraging a visual foundation model,” IEEE Robotics and Automation Letters, vol. 10, no. 9, pp. 9152–9159, 2025.

[57] A. Dai, G. Casadesus Vila, and G. Gao, “Visual SLAM with DEM anchoring for long-range lunar surface navigation,” in 2026 IEEE Aerospace Conference, 2026, pp. 1–9.

[58] A. Dai et al., “Full stack navigation, mapping, and planning for the lunar autonomy challenge,” in Proceedings of the 38th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS+ 2025). Baltimore, MD: Institute of Navigation, 2025, pp. 1986–2002.

[59] V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk, “HPatches: A benchmark and evaluation of handcrafted and learned local descriptors,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[60] T. Loiseau and G. Bourmaud, “RUBIK: A structured benchmark for image matching across geometric challenges,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 27 070–27 080.

[61] M. Zaffar et al., “VPR-Bench: An open-source visual place recognition evaluation framework with quantifiable viewpoint and appearance change,” International Journal of Computer Vision, vol. 129, no. 7, pp. 2136– 2174, 2021.

[62] G. Berton et al., “Deep visual geo-localization benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5386–5397.

[63] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and´ J. Sivic, “NetVLAD: CNN architecture for weakly supervised place recognition,” in Proceedings of the IEEE

Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 5297–5307.

[64] F. Warburg, S. Hauberg, M. Lopez-Antequera, P. Gargallo, Y. Kuang, and J. Civera, “Mapillary street-level sequences: A dataset for lifelong place recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2623–2632.

[65] S. Schubert, P. Neubert, S. Garg, M. Milford, and T. Fischer, “Visual place recognition: A tutorial,” IEEE Robotics & Automation Magazine, vol. 31, no. 3, pp. 139–153, 2024.

[66] M. Leyva-Vallina, N. Strisciuglio, and N. Petkov, “Data-efficient large scale place recognition with graded similarity supervision,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 23 487–23 496.

[67] L. A. Encinar Gonzalez, J. Folkesson, R. Triebel, and R. Giubilato, “Multi-modal loop closure detection with foundation models in severely unstructured environments,” in 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026, pp. 17 565– 17 572.

[68] L. Lipson, Z. Teed, and J. Deng, “RAFT-stereo: Multilevel recurrent field transforms for stereo matching,” in 2021 International Conference on 3D Vision (3DV), 2021, pp. 218–227.

[69] J. Liu et al., “LuSNAR: A lunar segmentation, navigation, and reconstruction dataset based on multisensor for autonomous exploration,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, p. 4706723, 2026.

[70] M. Hansen, U. Wong, and T. Fong, “The POLAR traverse dataset: A dataset of stereo camera images simulating traverses across lunar polar terrain under extreme lighting conditions,” arXiv:2403.12194 [cs.CV], 2024.

[71] D. van der Meer, U. Wong, and M. A. Olivares-Mendez, “LunaPolaris: A stereo camera, point cloud and IMU dataset for future lunar exploration in polar regions,” in 2025 International Conference on Space Robotics (iSpaRo), 2025, pp. 168–174, underlying dataset: https: //doi.org/10.5281/zenodo.15737938, licensed under CC BY 4.0.

[72] R. Giubilato, W. Sturzl, A. Wedler, and R. Triebel,¨ “Challenges of SLAM in extremely unstructured environments: The DLR planetary stereo, solid-state Li-DAR, inertial dataset,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 8721–8728, 2022.

[73] R. Giubilato, M. G. Muller, M. Sewtz, L. A. Enci-¨ nar Gonzalez, J. Folkesson, and R. Triebel, “The S3LI Vulcano dataset: A dataset for multi-modal SLAM in unstructured planetary environments,” in 2026 IEEE Aerospace Conference, 2026.

[74] J. L. Schonberger and J.-M. Frahm, “Structure-from-¨ motion revisited,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113.

[75] M. Muja and D. G. Lowe, “Scalable nearest neighbor algorithms for high dimensional data,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 11, pp. 2227–2240, 2014.

[76] J. Shi and C. Tomasi, “Good features to track,” in Pro-

ceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1994, pp. 593–600.

[77] B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” in Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI), 1981, pp. 674–679.

[78] G. Bradski, “The OpenCV library,” Dr. Dobb’s Journal of Software Tools, vol. 25, no. 11, pp. 120–126, 2000.

[79] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381– 395, 1981.

[80] CVG, ETH Zurich, “Glue factory: Training library for¨ local feature detection and matching,” https://github.c om/cvg/glue-factory, 2025.

[81] D. Nister, “An efficient solution to the five-point relative´ pose problem,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 6, pp. 756–770, 2004.

[82] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[83] S. Lowry et al., “Visual place recognition: A survey,” IEEE Transactions on Robotics, vol. 32, no. 1, pp. 1– 19, 2016.

[84] O. Simeoni ´ et al., “DINOv3,” 2025.

[85] N. Keetha et al., “AnyLoc: Towards universal visual place recognition,” IEEE Robotics and Automation Letters, vol. 9, no. 2, pp. 1286–1293, 2024.

## BIOGRAPHY

![](images/488447c799190b016c7b34d1e099db2c2e340a0632e415da2493672d5542cfae.jpg)

Guillem Casadesus Vila is a Ph.D. candidate in the Department of Aeronautics and Astronautics at Stanford University. He received his B.Sc. degree in Aerospace and Telecommunications Engineering in 2022 from the Universitat Politecnica de Catalunya (UPC) under\` the CFIS program. His research interests include robotic space exploration and space navigation and communication.

![](images/526c544cd479823a89bf10f9cd238fe3f40ecb94548113b1cc9878b77805a2ab.jpg)

Grace Gao is an associate professor in the Department of Aeronautics and Astronautics at Stanford University. Before joining Stanford University, she was an assistant professor at University of Illinois at Urbana-Champaign. She obtained her Ph.D. degree at Stanford University. Her research is on robust and secure positioning, navigation, and timing with applications to manned and unmanned aerial vehicles, autonomous driving cars, as well as space robotics.