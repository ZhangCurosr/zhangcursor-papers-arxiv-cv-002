# MV-dVRK: A Multi-Viewpoint Benchmark for Spatial Surgical Perception

Guido Caccianiga<sup>1,2</sup>, Sergey Prokudin<sup>2</sup>, Yutong Chen<sup>2</sup>, Bernard Javot<sup>1</sup>, Rachael L’Orsa<sup>1</sup>, Omer Burak Aladag˘<sup>1</sup>, Yarden Sharon<sup>1</sup>, Jens Rolinger<sup>3</sup>, Ivan Capobianco<sup>4</sup>, Anton Deguet<sup>5</sup>, Siyu Tang<sup>2</sup>, and Katherine J. Kuchenbecker<sup>1</sup>

<sup>1</sup>Max Planck Institute for Intelligent Systems. <sup>2</sup>ETH Zurich.¨ <sup>3</sup>Erbe Group. <sup>4</sup>Tubingen University Hospital. ¨ <sup>5</sup>Johns Hopkins University. {caccianiga@is.mpg.de kjk@is.mpg.de}

## Abstract

Large-scale training and refined optimization techniques have greatly improved sparse multi-view 3D reconstruction. Despite their relevance to surgery, such methods have never before been rigorously evaluated on real endoscopic images. Current clinical telerobots deploy a single stereo camera inside the patient, making multi-viewpoint data extremely rare. This paper presents MV-dVRK, the first ex-vivo surgical dataset to combine multiple exposuresynchronized stereo viewpoints with accurate surface geometry and camera poses. The static subset of the benchmark provides dense SfM reference geometry, validated against an industrial 3D scanner, together with ground-truth camera poses and sparse-view test sets. We use MV-dVRK to systematically compare zero-shot monocular, stereo, multistereo, and multi-view 3D reconstruction methods as the number of viewpoints increases. With two endoscopes, multi-stereo reconstruction achieves the highest coverage. With a third viewpoint, optimization-based multi-view methods perform best, covering 67% of ground-truth surface points within a 1 mm tolerance and recovering highly accurate relative camera poses. By contrast,feed-forwardfoundation models cover only 43% ofthe ground-truth surface in the same setting. MV-dVRK also includes ten dynamic sequences spanning multiple surgical tasks, with increasing kinematic complexity and tissue deformation, providing a basisforfuture research in multi-viewpoint surgical perception. The project is available at: https://mv-dvrk.is.mpg.de.

## 1. Introduction

Accurate geometric reconstruction can improve robotassisted minimally invasive surgery (RAMIS) by enabling advanced visualization for surgeons, such as free-viewpoint rendering [41], as well as geometry-aware robotic assistance [27] and autonomy [23]. Today, however, surgical 3D perception is fundamentally constrained by the imaging setup: clinical telerobots provide only a single stereo viewpoint [3]. Even sophisticated methods for handling tissue deformation and occlusions [33, 45] cannot fully overcome the ambiguity of reconstructing unseen anatomy from a single viewpoint [46]. Similarly, visuomotor policy learning benefits from additional viewpoints [28].

![](images/6730cd4b32c05f27dd56194ba47a16a6613b5676f5bac78751424cddd4f66a0e.jpg)  
Figure 1. Overview of the MV-dVRK benchmark workflow. Our custom robotic platform provides three exposuresynchronized stereo viewpoints, two of which are independently robot-controlled. For static scenes, dense multi-view SfM provides reference geometry and camera poses. Sparse zero-shot reconstructions from two to six synchronous input views are then evaluated for surface coverage, 3D error, and relative camera pose. The left inset shows a typical reconstruction, including the predicted 3D geometry, ground-truth surface coverage within a 1 mm tolerance, and pointwise 3D error to the reference geometry.

Multi-viewpoint imaging therefore offers a natural way to improve both geometric coverage and spatial awareness. At the same time, the physical constraints of minimally invasive surgery limit the number of cameras that can be deployed, making the relevant regime inherently sparse: only a few viewpoints, often with large baselines and limited overlap. This setting challenges modern 3D reconstruction methods and suffers from a scarcity of suitable surgical benchmarks. Existing datasets lack multiple exposuresynchronized endoscopic viewpoints and/or accurate measurements of scene geometry and camera poses (§2.1).

We address this gap with MV-dVRK, a dataset collected using a multi-viewpoint surgical telerobot that we developed by extending the da Vinci Research Kit [21] with a second independently controlled stereo endoscope, custom high-resolution cameras, and exposure-synchronized multi-camera acquisition (Fig. 1, §3.1). For this benchmark, we add a fixed stereo endoscope, yielding three synchronized stereo viewpoints. The resulting data enable controlled evaluation of sparse multi-view 3D reconstruction on real ex-vivo surgical imagery and provide synchronized dynamic sequences for future research in multi-viewpoint surgical perception. Beyond multi-view deployment, the geometric reference also enables rigorous evaluation of methods designed to operate from a single endoscope.

This paper contributes:

• A multi-viewpoint surgical dataset (§3) containing both large-baseline static scenes with dense reference geometry and synchronized dynamic sequences spanning multiple surgical tasks. The static reference reconstructions provide tissue-surface geometry and camera poses and are independently validated against an industrial 3D scanner.

• A sparse-view 3D reconstruction benchmark (§4) for systematically comparing monocular, stereo, multistereo, and multi-view methods as the number of synchronized input viewpoints increases.

• An analysis of current multi-view 3D perception methods (§5), showing how reconstruction coverage, geometric accuracy, and camera-pose estimation change with viewpoint count and view overlap. The results reveal complementary strengths of stereo and jointly optimized multi-view reconstruction and identify promising directions for practical multi-view surgical perception.

## 2. Related work

## 2.1. Multi-view surgical datasets

Most existing datasets for robotic surgery provide only a single viewpoint and face a fundamental trade-off between realistic data and accurate geometric ground truth. Measuring deforming tissue with sub-millimeter accuracy remains challenging both ex vivo and intraoperatively. Surgical simulators provide exact reference geometry and controlled variation but do not fully reproduce the visual appearance and tissue mechanics of real surgery. When available for real data, shape ground truth is typically obtained using a scanner, structured light, or computed tomography [1, 11, 29]; the cost, time, and disruption of these approaches limit the scale and diversity of the resulting datasets. Larger in-vivo collections generally provide no dense reference geometry of the tissue surface [17, 19, 47].

Only recently have surgical datasets begun to include multiple viewpoints. ImitateCholec [15] supplements a stationary stereo endoscope with two wrist-mounted cameras, but the streams are software-synchronized, limiting their use for precise multi-view geometric evaluation under deformation. Concurrent work, iMED [4], introduces a large dual-stereo-endoscope dataset spanning ex-vivo specimens, human cadavers, and an in-vivo porcine model. One endoscope is robot-mounted on a da Vinci 5 arm, while the second provides a fixed nearby perspective (inter-viewpoint baseline $\leq 4 \mathrm { c m } , \leq 3 0 \mathrm { d e g } )$ . iMED supports camera-pose evaluation via ArUco [12] markers but does not provide dense reference geometry of the tissue surface. No existing dataset combines multiple robot-controlled viewpoints, large baselines, sensor exposure synchronization, accurate reference geometry, and camera poses (Tab. 1). MV-dVRK is designed to fill this gap.

## 2.2. Geometric reconstruction in computer and medical vision

Stereo reconstruction. Stereo correspondence [20, 25, 40, 44] provides a direct route to metric depth from a calibrated camera pair and is particularly relevant to robot-assisted surgery, where current systems already deploy a stereo endoscope. Surgical stereo reconstruction has therefore been widely studied for recovering tissue geometry from a single endoscopic viewpoint [34]. Recent learned approaches have further improved robustness and generalization [7]. In our benchmark, this family of approaches is represented by FoundationStereo [44], a recent method designed for zero-shot stereo matching. Stereo can provide accurate local geometry where the tissue is texture-rich and wellilluminated, but a single endoscopic viewpoint remains limited by occlusions and incomplete scene coverage.

Single-view surgical reconstruction. Because surgeons typically perform laparoscopic surgery with only a single camera view, much of the surgical reconstruction literature has focused on recovering geometry from monocular video [26]. Recent methods have also adapted learned 3D priors to endoscopic imagery [13, 32]. We select Endo3R [13] as our monocular baseline because it is designed for online endoscopic reconstruction and can accumulate geometry over time. This choice allows us to test whether temporal observations from a single moving camera can achieve the coverage provided by additional simultaneous viewpoints. Nevertheless, reconstructing deforming anatomy (seen and particularly unseen) from a single camera remains fundamentally ill-posed [30, 35].

Table 1. Comparison with existing surgical datasets. Among the datasets considered, MV-dVRK uniquely combines multiple independently moving viewpoints, exposure-synchronized acquisition, and accurate reference geometry and camera poses. The static subset (I) provides geometric and pose ground truth, while the dynamic subset (II) captures synchronized tissue deformation and camera motion.
<table><tr><td>Dataset</td><td>Yr.</td><td>Type</td><td>Origin</td><td>VP</td><td>Defor.</td><td>Mov. Cam.</td><td>Sync</td><td>Geom. GT</td><td>Pose GT</td><td>Size</td><td>Hz</td><td>Resolution</td></tr><tr><td>Hamlyn [14]</td><td>&#x27;10</td><td>In-vivo</td><td>Pig/Human</td><td>1</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>&gt; 105</td><td>30</td><td> $7 2 0 \times 2 8 8$ </td></tr><tr><td>SCARED [1]</td><td>&#x27;21</td><td>Ex-vivo</td><td>Pig</td><td>1</td><td>X</td><td>√</td><td>X</td><td>√ Struc. light</td><td>≈ Robot FK (1 VP)</td><td>40</td><td>0</td><td> $1 2 8 0 \times 7 2 0$ </td></tr><tr><td>SERV-CT [11]</td><td>&#x27;22</td><td>Ex-vivo</td><td>Pig</td><td>1</td><td>X</td><td>X</td><td>X</td><td>√ CT</td><td>√Manual</td><td>16</td><td>0</td><td> $7 2 0 \times 5 7 6$ </td></tr><tr><td>EndoNeRF [41]</td><td>&#x27;22</td><td>In-vivo</td><td>Pig</td><td>1</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>807</td><td>15</td><td> $6 4 0 \times 5 1 2$ </td></tr><tr><td>StereoMIS [17]</td><td>&#x27;23</td><td>In-vivo</td><td>Pig</td><td>1</td><td>√</td><td>√</td><td>X</td><td>X</td><td>≈ Robot FK (1 VP)</td><td> $> 1 0 ^ { 4 }$ </td><td>30</td><td> $1 2 8 0 \times 1 0 2 4$ </td></tr><tr><td>SurgVU [47]</td><td>&#x27;25</td><td>In-vivo</td><td>Pig</td><td>1</td><td>√</td><td>√</td><td>×</td><td>X</td><td>X</td><td> $> 1 0 ^ { 6 }$ </td><td>60</td><td> $1 2 8 0 \times 7 2 0$ </td></tr><tr><td>SurgiSR4K [19]</td><td>&#x27;25</td><td>In-vivo</td><td>Pig</td><td>1</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td> $> 1 0 ^ { 3 }$ </td><td>60</td><td> $3 8 4 0 \times 2 1 6 0$ </td></tr><tr><td>ImitateCholec [15]</td><td>&#x27;26</td><td>Ex-vivo</td><td>Pig</td><td>3</td><td>√</td><td>X</td><td>≈ Software</td><td>X</td><td>≈ Robot FK (1 VP)</td><td> $> 1 0 ^ { 4 }$ </td><td>30</td><td> $9 6 0 \times 5 4 0$ </td></tr><tr><td>iMED [4]</td><td>&#x27;26</td><td>Ex-/In-vivo</td><td>Pig/Human</td><td>2</td><td>√</td><td>√</td><td>≈ Capture</td><td>X</td><td>√ ArUco (2 VP)</td><td> $> 1 0 ^ { 5 }$ </td><td>60</td><td> $1 2 8 0 \times 1 0 2 4$ </td></tr><tr><td>MV-dVRK I (ours)</td><td>&#x27;26</td><td>Ex-vivo</td><td>Pig</td><td>3</td><td>×</td><td>√</td><td>√ Exposure</td><td>√ Dense SfM</td><td>√ Dense SfM (3 VP)</td><td>840</td><td>0</td><td> $2 5 6 0 \times 2 0 4 8$ </td></tr><tr><td>MV-dVRK II (ours)</td><td>&#x27;26</td><td>Ex-vivo</td><td>Pig</td><td>3</td><td>√</td><td>√</td><td>√ Exposure</td><td>X</td><td>√ ArUco + Robot FK (3 VP)</td><td>&gt; 104</td><td>30</td><td> $2 5 6 0 \times 2 0 4 8$ </td></tr></table>

VP = viewpoints; Defor. = deforming tissue; Mov. Cam. = moving endoscopic camera; Sync = synchronization type: Software = timestamped by operating system; Capture = timestamped by capture card; Exposure = timestamped by image sensors. Geom. GT = ground-truth surface geometry; CT = computed tomography; Pose GT = ground-truth camera poses; FK = forward kinematics; Size = number of frames. ✓, ≈, and ×indicate availability. Pose GT cells also report the number of viewpoints tracked.

Feed-forward multi-view reconstruction. Recent feed-forward (FF) foundation models significantly improve sparse multi-view 3D reconstruction. DUSt3R [39] can directly regress camera parameters and dense 3D point maps from uncalibrated images, inspiring a rapidly growing family of models for sparse multi-view geometric prediction [22, 24, 38, 43]. The ability to reconstruct from very sparse unposed input views makes these methods particularly attractive for surgical reconstruction. However, how these foundation models perform on endoscopic imagery, which is under-represented in their training data, requires further investigation. MV-dVRK provides an extensive benchmark for this purpose; we evaluate VGGT [38], Pi3 [42], Pi3X [43], and Depth Anything 3 [24].

Geometry-refined multi-view reconstruction. Structure from motion (SfM) provides the classical framework for jointly recovering camera poses and 3D structure from overlapping images [16], with COLMAP [31] serving as a standard implementation. Classical feature-based SfM is most reliable with many overlapping observations of a static scene, whereas minimally invasive surgery provides only a few viewpoints with potentially large viewpoint change and limited overlap. Recent methods tackle this sparse regime by combining learned correspondence with explicit geometric optimization [6, 9, 37]. We evaluate MASt3R-SfM [9] and GGPT [6] as representative methods from this family. The former jointly optimizes camera poses and geometry from learned correspondences, while the latter consists of an SfM stage and a point-map refinement stage. Evaluating GGPT with several feed-forward backbones further allows us to directly compare feed-forward inference with geometry-refined multi-view reconstruction under the same sparse surgical inputs.

## 3. The MV-dVRK dataset

## 3.1. Data acquisition platform

The acquisition platform extends the da Vinci Research Kit (dVRK) [21] with a second stereo endoscope moved by an independent robotic arm. We refer to this multi-viewpoint extension as the MV-dVRK platform. For the benchmark, we add a stereo endoscope on a fixed mount, yielding three synchronized stereo viewpoints (Fig. 1).

The endoscopes are custom-built from surgical-grade optics and industrial global-shutter cameras. A shared hardware trigger exposes all six imaging sensors simultaneously, and their clocks are aligned using the Precision Time Protocol (PTP [18]), resulting in microsecond-level timestamp agreement across cameras. This exposure synchronization ensures that all cameras observe the same spatiotemporal state for a given frame, which is essential for geometric evaluation of deforming scenes. Intrinsic and stereo calibration (board: ChArUco DICT 5×5, 15×24, 150 mm×100 mm, ceramic lithography) achieve sub-pixel reprojection errors, and robot kinematics are logged at 100 Hz on the same PTP network.

## 3.2. Dataset collection and composition

We recorded MV-dVRK during mock surgery on the freshly excised abdominal organs of two adult pigs that were sacrificed for other reasons by a local meat producer. Two expert visceral surgeons teleoperated various left and right instruments to perform representative surgical tasks, including cholecystectomy, tubular dissection, and small-bowel anastomosis. At selected stages of each procedure, we locked the instruments and swept each robotic endoscope over the stationary tissue to acquire dense geometric reference data. The recordings form two complementary subsets, summarized relative to existing surgical datasets in Tab. 1.

MV-dVRK I (static). The static subset comprises eight geometric reference scenes spanning three surgical tasks and different tissue appearances, robot configurations, and camera geometries, as illustrated in Fig. 2. Each scene is acquired while the anatomy is stationary, allowing dense multi-view SfM to recover reference tissue-surface geometry and camera poses. From each scene, we select 10 synchronized keyframes to maximize diversity in relative camera poses, yielding 80 test instances with three stereo viewpoints each. Because all keyframes within a scene share the same underlying anatomy and reference reconstruction, our analysis treats the scene as the statistical unit, rather than the individual frame (§4.5). MV-dVRK I forms the quantitative core of this paper and provides the reference geometry, camera poses, and synchronized sparse inputs used throughout the benchmark.

![](images/94ac0bde871fd899c894a9b3a0de47eeea7e7ddc58a8968f0cdc92935c5a9936.jpg)  
Figure 2. MV-dVRK I test set. The three endoscopic viewpoints (E1, E2, E3) span different surgical tasks, tissue appearances, relative camera poses, and robot configurations; “L” denotes the left camera of each stereo endoscope. The scenes include large linear and angular baselines, challenging illumination, and regions that are occluded from one viewpoint but visible from another. Rows index the 10 selected synchronized keyframes per scene; intermediate keyframes are omitted for visualization.

MV-dVRK II (dynamic). The dynamic subset contains ten long multi-view sequences recorded during active surgical manipulation, together with synchronized dVRK kinematics and per-scene calibration offsets used to link the robot measurements to the three endoscopes’ optical frames, effectively delivering a continuous reference for both single- and multi-camera pose estimation. Unlike conventional single-endoscope recordings, these sequences capture substantial tool–tissue interaction and tissue deformation from both static and moving viewpoints simultaneously. This subset thus provides a complementary dynamic counterpart to the static subset of the benchmark and extends MV-dVRK to settings involving camera motion and non-rigid scene dynamics. Representative sequences appear in the supplementary video. These synchronized multi-view recordings were mainly designed to support future work on dynamic surgical 3D perception; they also provide out-ofthe-box multi-view input for tasks such as dynamic novelview synthesis and camera-pose estimation.

## 4. Benchmark design

MV-dVRK I provides a common protocol for evaluating sparse 3D reconstruction on real surgical imagery. Each test instance contains three synchronized stereo viewpoints, corresponding to six images in total. We evaluate subsets of these inputs while varying the number of viewpoints from one to three as the primary experimental variable. Depending on the reconstruction family, a viewpoint may contribute either one monocular image or a calibrated stereo pair. For every prediction, we use the same dense reference reconstruction (§4.1), sparse test inputs (§4.2), registration protocol (§4.3), and evaluation metrics (§4.4).

## 4.1. Dense reference reconstruction

Rationale. The benchmark requires both dense tissuesurface geometry and camera poses in a common coordinate frame. External scanners can provide accurate surface geometry but do not simultaneously recover the endoscopic camera poses and are difficult to deploy during surgical acquisition [5, 11]. We therefore construct the benchmark reference directly from dense multi-view SfM over the static acquisition sequences and use an industrial 3D scanner for independent geometric validation and metric scaling.

The reference and benchmark inputs originate from the same acquisitions of static scenes but operate in very different regimes. Each reference reconstruction uses a dense set of views, whereas evaluated methods receive only sparse synchronized images. This design provides a reference pose for every test image and, because test images are included in the reference reconstruction, enables pixel-aligned comparison between predicted and reference 3D points.

Reference image selection. We subsample the three left-camera streams (E1L, E2L, E3L) from each static acquisition at 0.4 Hz and manually remove blurry and redundant frames. 105 images per scene were retained for reference reconstruction. The third endoscope is sampled less densely because repeated frames from this stationary camera provide little additional viewpoint diversity for SfM.

![](images/6c406dab309c0ff0f407d623abb8abb273cf32012e652e610653a70521b71ac8.jpg)  
Figure 3. External validation of the dense SfM reference geometry. For three representative scenes, we compare the SfM reconstruction (top) with an independently acquired 3D scanner surface (middle) and visualize the resulting point-to-surface error distribution (bottom). Mean errors are about 0.6 mm across these scenes, supporting sub-millimeter accuracy of the reference geometry.

Dense reconstruction. We construct the reference using the SfM stage of GGPT [6]. The pipeline uses cameras and points predicted by VGGT [38] as initialization and adopts RoMaV2 [10] to extract dense correspondences. It runs a global bundle adjustment using a compact set of high-confidence correspondences to optimize camera poses, upon which 3D points are triangulated from a dense set of correspondences. Uncertain and inconsistent points are filtered based on their triangulation angles and reprojection errors. Compared with previous SfM pipelines operating on sparse keypoint correspondences [31, 37], GGPT-SfM is more robust to the low texture and specularities common in endoscopic imagery and provides substantially denser geometric support on our acquisitions. While classical SfM reconstructs only a few thousand points in this setting, the resulting reference models contain more than 14 million points per scene. Note that although we include GGPT in our benchmark, our reference construction uses only its SfM stage; the second stage of feed-forward refinement is not applied here, as it is not strictly geometry-grounded.

External validation. An industrial 3D scanner (Artec Eva [2]) independently records the tissue surface once per surgical session. The scanner does not provide endoscopic camera poses and is therefore used for validation and metric scale rather than as the benchmark reference itself. Scaled ICP alignment [48] is done in CloudCompare v2.12. The mean point-to-surface distance (Fig. 3) between the dense SfM reconstructions and scanner meshes is 0.6 ± 0.4 mm.

## 4.2. Synchronized sparse test set

For each of the eight reference scenes, we select 10 synchronized test instants to maximize diversity in relative camera poses, yielding 80 test samples. Each sample contains three stereo pairs, i.e., six synchronized images. The selected configurations have widely varying relative endoscope poses, amounts of view overlap, illumination conditions, robot configurations, and tool poses (Fig. 2).

The dense surface reconstruction provides projected reference geometry for every test image. Coverage differs across the three endoscopes because their fields of view observe different portions of the tissue. E1L has the highest 19reference completeness and is therefore used as the common reference view for pixel-aligned 3D error. The benchmark evaluates different subsets of the synchronized inputs while varying the number of available viewpoints.

## 4.3. Prediction-to-reference registration

Methods may express their reconstructions in different coordinate systems and, in some cases, only up to an unknown global scale. We therefore align every prediction to the reference before evaluating geometry. The benchmark measures reconstructed shape and relative camera geometry after alignment; recovery of absolute metric scale is not itself an evaluation target.

Because each test image is included in the dense reference reconstruction, predicted and reference points corresponding to the same E1L pixels provide direct 3D correspondences. We estimate a robust Sim(3) transformation from these correspondences using PyCOLMAP’s estimate sim3d robust function [31], which combines LO-RANSAC [8] with Umeyama fitting [36]. We call this full point-based similarity alignment fitted registration.

For methods that jointly predict geometry and camera poses, we additionally consider a camera-anchored Sim(3) registration. Here, the predicted E1L camera determines the rotation and origin of the alignment, while only the global scale s is estimated from the robust 3D inliers. Unlike a fully fitted Sim(3), this registration cannot independently rotate or translate the reconstructed geometry to compensate for inconsistencies with the method’s own camera estimate. The resulting transformation is applied consistently to all predicted geometry and camera poses. Methods without a directly comparable reference-camera pose are evaluated using fitted registration, as specified in §5.1.

## 4.4. Metrics

We evaluate the following three complementary properties for each reconstruction; formal metric definitions are provided in the supplementary material.

Ground-truth coverage (Cov) measures the fraction of reference surface points that have a predicted neighbor within a fixed distance threshold. Coverage within 1 mm is our primary measure, and within 5 mm is secondary. Point clouds are subsampled to $1 0 ^ { 5 }$ points before evaluation.

![](images/d98966ec25d171d076a85c044969b2597893bd11866dc5346e836bf7b2fc0bb1.jpg)  
Figure 4. Qualitative comparison of reconstruction paradigms as viewpoint count increases. (a) Dense SfM reference geometry, shown as the E1L pointmap used for pixel-aligned 3D error and as the full point cloud used for coverage evaluation. For each prediction, the rows show the reconstructed geometry, a close-up, pointwise 3D error, and ground-truth surface coverage within 1 mm. With a single viewpoint, occluded anatomy is either incorrectly completed by Endo3R [13] (b) or remains unobserved with FoundationStereo [44] (c). Adding stereo viewpoints recovers previously occluded surface regions (d), but independent reconstruction and registration can introduce duplicated structures and large local errors (e). Joint multi-view reconstruction with DA3+GGPT [6] produces more consistent geometr (f,g), with the third viewpoint substantially reducing error in the common reference region (g).

Mean 3D error (ME) measures the Euclidean distance between registered predicted points and their pixel-aligned reference counterparts in the E1L field of view. E1L is shared across all benchmark configurations and therefore provides a common region for judging geometric accuracy.

Relative pose error (Pose) measures the accuracy of predicted relative transforms between the left cameras of different endoscopes. Translation error is reported in millimeters and geodesic rotation error in degrees.

## 4.5. Statistical analysis

The eight scenes are the units of statistical analysis. The ten test samples within each scene share the same anatomy and reference reconstruction and are therefore not treated as independent observations. For each pairwise comparison between methods, we first average the metric within each scene and then compute the eight paired scene-level differences, ∆. We report their mean, the corresponding 95% confidence interval (t ), and the number of scenes in which the difference has the same sign (k/8). Result tables report the mean over all 80 test samples for compact descriptive comparison, while all inferential statements in the text are based on the paired scene-level analysis. We do not claim that the observed effect magnitudes generalize to new specimens, procedure types, or in-vivo conditions. Correlations across scenes are reported using Pearson’s r and interpreted descriptively given the small number of scenes.

## 5. Benchmark analysis

## 5.1. Implementation details for selected methods

We use publicly released implementations and default inference settings unless stated otherwise. Endo3R [13] is evaluated with two-frame and 50-frame temporal windows and uses fitted Sim(3) registration. FoundationStereo [44] uses calibrated stereo rectification and is evaluated with both fit-

Table 2. Comparison of 3D reconstruction paradigms across viewpoint count. Registered multi-stereo is most accurate in the sparse two-viewpoint setting, while a third viewpoint favors jointly optimized multi-view reconstruction: Depth Anything 3 [24] + GGPT [6] achieves the highest coverage (Cov) and lowest mean error (ME) in E1L, whereas FoundationStereo [44] + Sim(3) obtains the lowest error in E2L. Darker shading indicates better performance; outlined cells denote the best value in each column.
<table><tr><td>Family</td><td>Method</td><td></td><td>View- Sync. Cov ↑</td><td></td><td>ME↓ points views %@1mm mm @E1L mm @E2L</td><td>ME↓</td><td>Pose t↓ Pose r ↓ mm</td><td>deg</td></tr><tr><td rowspan="2">Monocular</td><td>Endo3R [13] (2 fr.)</td><td>1</td><td>1</td><td>12</td><td>17.4</td><td></td><td></td><td></td></tr><tr><td>Endo3R [13] (50 fr.)</td><td>1</td><td>1</td><td>24</td><td>18.2</td><td></td><td>_</td><td>一</td></tr><tr><td>Stereo</td><td>FS [44] (ftted)</td><td>1</td><td>2</td><td>39</td><td>1.6</td><td></td><td></td><td></td></tr><tr><td>Stereo</td><td>FS [44] (anchored)</td><td>1</td><td>2</td><td>36</td><td>1.6</td><td></td><td></td><td>1</td></tr><tr><td rowspan="2">Multi-stereo</td><td>FS [44] + SE(3) reg.</td><td>2</td><td>4</td><td>44</td><td>1.6</td><td>6.3</td><td>12.1</td><td>2.7</td></tr><tr><td>FS [44] + Sim(3) reg.</td><td>2</td><td>4</td><td>50</td><td>1.6</td><td>2.8</td><td>4.6</td><td>1.7</td></tr><tr><td>Multi-view</td><td>DA3 [24] + GGPT [6]</td><td>2</td><td>4</td><td>48</td><td>3.4</td><td>5.3</td><td>0.9</td><td>0.3</td></tr><tr><td rowspan="2">Multi-stereo</td><td>FS [44] + SE(3) reg.</td><td>3</td><td>6</td><td>52</td><td>1.6</td><td>6.3</td><td>10.6</td><td>2.1</td></tr><tr><td>FS [44] + Sim(3) reg.</td><td>3</td><td>6</td><td>58</td><td>1.6</td><td>2.8</td><td>3.7</td><td>1.4</td></tr><tr><td>Multi-view</td><td>DA3 [24] + GGPT [6]</td><td>3</td><td>6</td><td>67</td><td>1.2</td><td>3.8</td><td>0.6</td><td>0.2</td></tr></table>

Table 3. Comparison of multi-view methods across viewpoint count. Mean error (ME) in the reference view E1L measures geometric accuracy in a common image region, while E2L additionally reflects cross-camera pose and view overlap. Geometry-refined methods benefit most from the third viewpoint and achieve the highest coverage (Cov) and lowest E1L error at six synchronized views. Darke shading indicates better performance; outlined cells denote the best value in each column.  
![](images/012b4cf57d978cbb7b50b3472fddcd8910c95c40c993f4587ee3315227356997.jpg)  
Figure 5. Representative visual results for multi-view methods on the MV-dVRK benchmark. (a) Mean error (ME) in the reference view E1L across evaluated multi-view methods at six synchronized views; geometry-refined methods achieve the lowest and most spatially uniform errors. (b) For Depth Anything 3 [24] + GGPT [6], increasing input viewpoint count progressively reduces reconstruction error. (c) Co-visibility analysis at the sparsest setting: partial view overlap leaves large regions without cross-view geometric support.

ted and camera-anchored alignment. For multi-stereo, independently reconstructed FoundationStereo point clouds are registered using RoMa v2 [10] correspondences with either rigid SE(3) or similarity Sim(3) transforms. Multi-view methods use their standard pipelines, with calibrated intrinsics supplied where supported. The supplementary material provides further preprocessing and registration details.

## 5.2. Quantitative results

Tab. 2 and Fig. 4 summarize the main comparisons across reconstruction paradigms. Tab. 3 provides the full comparison of multi-view methods, while Fig. 5 visualizes their geometric errors and the effects of input viewpoint count.

Single-view reconstruction. Dense stereo provides substantially greater scene coverage than monocular reconstruction from the same endoscopic viewpoint. Compared with the two-frame Endo3R baseline, FoundationStereo increases coverage by 27 percentage points (95% CI [20, 34], 8/8 scenes). Temporal accumulation narrows this gap but does not close it: even after 50 monocular frames, stereo retains a 15-point coverage advantage (95% CI [5.8, 25], 6/8). This finding indicates that temporal observations from one moving camera do not recover all of the geometry available from simultaneous stereo observations.

Multi-stereo reconstruction. Additional endoscopes increase coverage by exposing anatomy that is occluded from the reference viewpoint (Fig. 4d). Registering a second stereo reconstruction increases coverage by 14 percentage points (95% CI [7.2, 21], 8/8), and a third viewpoint provides a further 7.2-point gain (95% CI [5.2, 9.3], 8/8). The local geometry of each stereo reconstruction remains accurate, but cross-endoscope registration can introduce duplicated structures and other inconsistencies (Fig. 4e). Registration quality depends strongly on correspondence selection and solver robustness; the corresponding ablations are provided in the supplementary material.

Multi-view reconstruction. Unlike multi-stereo, multiview methods jointly reconstruct all input images in a common geometric frame and can therefore use additional viewpoints to improve the reconstruction itself. Adding the third viewpoint consistently reduces error in the common reference view E1L. For geometry-refined methods, the mean reduction is 1.9 mm (95% CI [1.2, 2.6], 8/8 scenes), corresponding to a 54% relative improvement. The effect on E2L is more variable: the mean reduction is 2.1 mm (95% CI [−0.05, 4.2]), reflecting differences in inter-endoscope geometry and co-visibility across scenes.

These results highlight a clear crossover between reconstruction paradigms. With two endoscopes, registered stereo is more accurate in E1L than every evaluated multiview method, with the advantage holding across all eight scenes. With three endoscopes, geometry-refined multiview reconstruction overtakes multi-stereo in the reference view and achieves the highest overall coverage. At six synchronized views, the refined methods reach 66–67% coverage within 1 mm, compared with 28–44% for the feedforward models (Tab. 3). Thus, the benefit of explicit geometric optimization becomes most pronounced once sufficient cross-view support is available.

Role of view overlap. The remaining failures are strongly linked to sparse geometric support. At comparable working distance, reduced co-visibility between viewpoints is associated with increased reconstruction error for most multi-view methods; analysis is provided in the supplementary material. Fig. 5c illustrates this behavior for Depth Anything 3 [24] + GGPT [6]: under partial overlap, large regions receive no cross-view geometric constraints, forcing the point transformer to rely more on its learned prior and producing inaccurate 3D predictions.

## 5.3. Implications for multi-view surgical reconstruction

Our results suggest complementary roles for calibrated stereo and global multi-view optimization. Stereo remains particularly effective in the sparsest regime, while geometry-refined multi-view methods benefit strongly from additional cross-view support. A promising direction is therefore rig-aware optimization that preserves the known geometry within each stereo endoscope while jointly refining scene structure and relative poses across endoscopes. Practical deployment will additionally require lower-latency dense matching and optimization. MV-dVRK II extends these questions to dynamic scenes involving camera motion, tissue deformation, and robot kinematics.

## 5.4. MV-dVRK limitations

MV-dVRK I provides dense reference geometry only for static tissue; obtaining comparable ground truth under deformation remains impractical, so MV-dVRK II does not provide dense surface geometry. The camera poses estimated with dense SfM also cannot be externally validated without introducing manual and optical calibration biases. The static subset of the benchmark contains eight scenes from two porcine specimens and three surgical tasks. This dataset supports rigorous paired comparisons across methods within the benchmark, but the measured effect sizes should not be assumed to transfer unchanged to new anatomies, procedures, or in-vivo conditions. Finally, the reference surface is reconstructed rather than directly measured, though it is independently validated using an industrial 3D scanner, and the open ex-vivo acquisition with additional illumination differs from a closed abdominal cavity.

## 6. Conclusion

We introduced MV-dVRK, a surgical dataset combining three exposure-synchronized stereo viewpoints with validated reference geometry and camera poses, and we used it to establish a controlled benchmark for sparse multi-view 3D reconstruction on real robotic surgery data. Our results reveal a clear transition with viewpoint count: registered stereo is highly competitive in the sparsest two-endoscope setting, while geometry-refined multi-view methods benefit strongly from a third viewpoint and achieve the best coverage and reference-view accuracy. Beyond the static benchmark, synchronized dynamic sequences extend MV-dVRK to camera motion and tissue deformation, providing a basis for future multi-viewpoint surgical 3D perception.

## 7. Acknowledgments

The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) and the Max Planck ETH Center for Learning Systems (CLS) for supporting Guido Caccianiga.

## References

[1] Max Allan, Jonathan Mcleod, Congcong Wang, Jean Claude Rosenthal, Zhenglei Hu, Niklas Gard, Peter Eisert, Ke Xue Fu, Trevor Zeffiro, Wenyao Xia, Z. Zhu, H. Luo, X. Zhang, X. Li, F. Jia, L. Sharan, T. Kurmann, S. Schmid, R. Sznitman, D. Psychogyios, M. Azizian, D. Stoyanov, L. Maier-Hein, and S. Speidel. Stereo correspondence and reconstruction of endoscopic data challenge, 2021. arXiv:2101.01133. 2, 3

[2] Artec 3D. Artec Eva 3D scanner. https://www. artec3d.com/portable-3d-scanners/arteceva, 2026. Accessed: 28 August 2026. 5

[3] Matthew Boal, C. Giovene Di Girasole, Freweini Tesfai, T. E. M. Morrison, Simon Higgs, Jawad Ahmad, Alberto Arezzo, and Nader Francis. Evaluation status of current and emerging minimally invasive robotic surgical platforms. Surgical Endoscopy, 38(2):554–585, 2024. 1

[4] Sierra Bonilla, Fengyi Jiang, Chinedu Nwoye, Jingpei Lu, Kailey Reardon, Humphrey W. Chow, Francisco Vasconcelos, Sophia Bano, Adam Schmidt, and Omid Mohareri. iMED: a multi-endoscope dataset for surgical 3D perception. In European Conference on Computer Vision (ECCV), 2026. In press. 2, 3

[5] Guido Caccianiga, Julian Nubert, Marco Hutter, and Katherine J. Kuchenbecker. Dense 3D reconstruction through lidar: A comparative study on ex-vivo porcine tissue, 2024. arXiv:2401.10709. 4

[6] Yutong Chen, Yiming Wang, Xucong Zhang, Sergey Prokudin, and Siyu Tang. GGPT: Geometry-grounded point transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28959–28968, 2026. 3, 5, 6, 7, 8

[7] Xuelian Cheng, Yiran Zhong, Mehrtash Harandi, Tom Drummond, Zhiyong Wang, and Zongyuan Ge. Deep laparoscopic stereo matching with transformers. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 464–474. Springer, 2022. 2

[8] Ondrej Chum, Jiri Matas, and Josef Kittler. Locally optimized RANSAC. In Joint Pattern Recognition Symposium, pages 236–243. Springer, 2003. 5

[9] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. MASt3R-SfM: a fully-integrated solution for unconstrained

structure-from-motion. In International Conference on 3D Vision (3DV), pages 1–10. IEEE, 2025. 3, 7

[10] Johan Edstedt, David Nordstrom, Yushan Zhang, Georg¨ Bokman, Jonathan Astermark, Viktor Larsson, Anders Hey-¨ den, Fredrik Kahl, Marten Wadenb˚ ack, and Michael Fels-¨ berg. RoMa v2: harder better faster denser feature matching, 2026. arXiv: 2511.15706, Accepted to ECCV 2026. 5, 7

[11] P. J. Eddie Edwards, Dimitris Psychogyios, Stefanie Speidel, Lena Maier-Hein, and Danail Stoyanov. SERV-CT: a dispar ity dataset from cone-beam CT for validation of endoscopic 3D reconstruction. Medical Image Analysis, 76, 2022. Art. no. 102302. 2, 3, 4

[12] S. Garrido-Jurado, R. Munoz-Salinas, F.J. Madrid-Cuevas,˜ and M.J. Mar´ın-Jimenez. Automatic generation and detec-´ tion of highly reliable fiducial markers under occlusion. Pat tern Recognition, 47(6):2280–2292, 2014. 2

[13] Jiaxin Guo, Wenzhen Dong, Tianyu Huang, Hao Ding, Ziyi Wang, Haomin Kuang, Qi Dou, and Yun-Hui Liu. Endo3R: unified online reconstruction from dynamic monocular endoscopic video. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 170–180. Springer, 2025. 2, 6

[14] Hamlyn Centre. Hamlyn Centre surgical dataset. http:// hamlyn.doc.ic.ac.uk/vision/, 2010. Accessed: 4 June 2023. 3

[15] Pascal Hansen, Ji Woong Brian Kim, Antony Goldenberg, Juo Tung Chen, Yuanzhe Amos Li, Anton Deguet, Brandon White, De Ru Tsai, Richard Cha, Jeffrey Jopling, Paul Maria Scheikl, and Axel Krieger. ImitateCholec: a multimodal dataset for long-horizon imitation learning in robotic cholecystectomy. Scientific Data, 13, 2026. Art. no. 215. 2, 3

[16] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, 2nd edition, 2003. 3

[17] Michel Hayoz, Christopher Hahne, Mathias Gallardo, Daniel Candinas, Thomas Kurmann, Maximilian Allan, and Raphael Sznitman. Learning how to robustly estimate camera pose in endoscopic videos. International Journal of Com puter Assisted Radiology and Surgery, 18(7):1185–1192, 2023. 2, 3

[18] IEEE. IEEE standard for a precision clock synchronization protocol for networked measurement and control systems. IEEE Std 1588-2019, 2020. 3

[19] Fengyi Jiang, Xiaorui Zhang, Lingbo Jin, Ruixing Liang, Yuxin Chen, Adi Chola Venkatesh, Jason Culman, Tiantian Wu, Lirong Shao, Wenqing Sun, Cong Gao, Hallie McNamara, Jingpei Lu, and Omid Mohareri. SurgiSR4K: a high resolution endoscopic video dataset for robotic-assisted min imally invasive procedures. Machine Learning for Biomedi cal Imaging, 3:875–885, 2025. 2, 3

[20] Junpeng Jing, Jiankun Li, Pengfei Xiong, Jiangyu Liu, Shuaicheng Liu, Yichen Guo, Xin Deng, Mai Xu, Lai Jiang, and Leonid Sigal. Uncertainty guided adaptive warping for robust and efficient stereo matching. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 3295– 3304, 2023. 2

[21] Peter Kazanzides, Zihan Chen, Anton Deguet, Gregory S. Fischer, Russell H. Taylor, and Simon P. DiMaio. An open

source research kit for the da Vinci® Surgical System. In IEEE International Conference on Robotics and Automation (ICRA), pages 6434–6439, 2014. 2, 3

[22] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3D with MASt3R. In European Conference on Computer Vision (ECCV), pages 71–91, 2024. 3

[23] Yang Li, Florian Richter, Jingpei Lu, Emily K. Funk, Ryan K. Orosco, Jianke Zhu, and Michael C. Yip. SuPer: a surgical perception framework for endoscopic tissue manipulation with surgical robotics. IEEE Robotics and Automation Letters, 5(2):2294–2301, 2020. 1

[24] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: recovering the visual space from any views, 2025. arXiv:2511.10647. 3, 6, 7, 8

[25] Lahav Lipson, Zachary Teed, and Jia Deng. RAFT-Stereo: multilevel recurrent field transforms for stereo matching. In International Conference on 3D Vision (3DV), pages 218– 227. IEEE, 2021. 2

[26] Nader Mahmoud, Toby Collins, Alexandre Hostettler, Luc Soler, Christophe Doignon, and Jose Maria Martinez Montiel. Live tracking and dense reconstruction for handheld monocular endoscopy. IEEE Transactions on Medical Imaging, 38(1):79–89, 2019. 2

[27] Rocco Moccia, Mario Selvaggio, Luigi Villani, Bruno Siciliano, and Fanny Ficuciello. Vision-based virtual fixtures generation for robotic-assisted polyp dissection procedures. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7934–7939, 2019. 1

[28] Masoud Moghani, Nigel Nelson, Mohamed Ghanem, Andres Diaz-Pinto, Kush Hari, Mahdi Azizian, Ken Goldberg, Sean Huver, and Animesh Garg. SuFIA-BC: Generating high quality demonstration data for visuomotor policy learning in surgical subtasks, 2025. arXiv:2504.14857. 1

[29] Veronica Penza, Andrea S. Ciullo, Sara Moccia, Leonardo S. Mattos, and Elena De Momi. EndoAbS dataset: endoscopic abdominal stereo image dataset for benchmarking 3D stereo reconstruction algorithms. The International Journal of Medical Robotics and Computer Assisted Surgery, 14(5), 2018. Art. no. e1926. 2

[30] Adam Schmidt, Omid Mohareri, Simon DiMaio, Michael C. Yip, and Septimiu E. Salcudean. Tracking and mapping in medical computer vision: A review. Medical Image Analysis, 94, 2024. Art. No. 103131. 3

[31] J. L. Schonberger and J.-M. Frahm. Structure-from-motion¨ revisited. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4104–4113, 2016. 3, 5

[32] Mona Sheikh Zeinoddin, Mobarak I. Hoque, Zafer Tandogdu, Greg L. Shaw, Matthew J. Clarkson, Evangelos B. Mazomenos, and Danail Stoyanov. Endo-FASt3r: endoscopic foundation model adaptation for structure from motion. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 117–126. Springer, 2025. 2

[33] Tianyi Song, Danail Stoyanov, Evangelos Mazomenos, and Francisco Vasconcelos. Diff2DGS: reliable reconstruction of occluded surgical scenes via 2D Gaussian splat-

ting. IEEE Robotics and Automation Letters, 11(10):11251– 11258, 2026. 1

[34] Danail Stoyanov, Marco Visentini-Scarzanella, Philip Pratt, and Guang-Zhong Yang. Real-time stereo reconstruction in robotically assisted minimally invasive surgery. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 275–282. Springer, 2010. 2

[35] Edith Tretschk, Navami Kairanda, Mallikarjun B R, Rishabh Dabral, Adam Kortylewski, Bernhard Egger, Marc Habermann, Pascal Fua, Christian Theobalt, and Vladislav Golyanik. State of the art in dense monocular non-rigid 3D reconstruction. In Computer Graphics Forum, pages 485– 520. Wiley Online Library, 2023. 3

[36] S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991. 5

[37] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. VGGSfM: visual geometry grounded deep structure from motion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21686– 21697, 2024. 3, 5

[38] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5306, 2025. 3, 5, 7

[39] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In IEEE/CVF Conference on Computer Vi sion and Pattern Recognition (CVPR), pages 20697–20709, 2024. 3

[40] Xianqi Wang, Gangwei Xu, Hao Jia, and Xin Yang. Selective-Stereo: adaptive frequency information selection for stereo matching. In IEEE/CVF Conference on Compututer Vision and Pattern Recognition (CVPR), pages 19701–19710, 2024. 2

[41] Yuehao Wang, Yonghao Long, Siu Hin Fan, and Qi Dou. Neural rendering for stereo 3D reconstruction of deformable tissues in robotic surgery. In International Conference on Medical Image Computing and Computer-Assisted Interven tion (MICCAI), pages 431–441. Springer, 2022. 1, 3

[42] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun hua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations (ICLR), 2025. 3, 7

[43] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. Pi3X model update for “Permutationequivariant visual geometry learning”. GitHub Repository, 2025. Accessed: 1 July 2026. 3, 7

[44] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. FoundationStereo: zeroshot stereo matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5249– 5260, 2025. 2, 6

[45] Xiankang Yu, Yuichiro Hayashi, Masahiro Oda, Takayuki Kitasaka, and Kensaku Mori. Endo-PairGS: pair priors for dynamic endoscopic scene reconstruction. International Journal of Computer Assisted Radiology and Surgery, 21: 1341–1350, 2026. 1

[46] Lingting Zhu, Zhao Wang, Zhenchao Jin, Guying Lin, and Lequan Yu. EndoGS: deformable endoscopic tissues reconstruction with Gaussian splatting. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 135–145. Springer, 2024. 1

[47] Aneeq Zia, Max Berniker, Rogerio Nespolo, Conor Perreault, Ziheng Wang, Benjamin Mueller, Ryan Schmidt, Kiran Bhattacharyya, Xi Liu, and Anthony Jarc. Surgical visual understanding (SurgVU) dataset, 2026. arXiv:2501.09209. 2, 3

[48] Timo Zinßer, Jochen Schmidt, and Heinrich Niemann. Point set registration with integrated scale estimation. In International Conference on Pattern Recognition and Information Processing (PRIP), 2005. 5