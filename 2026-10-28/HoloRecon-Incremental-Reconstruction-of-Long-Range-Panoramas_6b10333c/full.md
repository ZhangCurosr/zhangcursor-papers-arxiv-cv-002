# HoloRecon: Incremental Reconstruction of Long-Range Panoramas with Global Consistency

Zhuoxiao Li<sup>1,∗</sup>, Xinyi Liu<sup>1,∗</sup>, Hongye Hou<sup>1,∗</sup>, Jing OU<sup>1,∗</sup>, Kang Du<sup>1</sup>, Shuai Zhang<sup>1</sup>, Dongli Wu<sup>1</sup>, Taoyu Wu<sup>2</sup>, Siyuan Feng<sup>3</sup>, Xiaohui Zhu<sup>4</sup>, Wufan Zhao<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (GuangZhou), <sup>2</sup>University of Liverpool, <sup>3</sup>Hong Kong Polytechnic University, <sup>4</sup>Xi’an Jiaotong-Liverpool University <sup>∗</sup>Equal contribution., <sup>†</sup>Corresponding author.

![](images/9b8cd266d1b44171027679b646cdc25ae79dda944785a4b0a2298b74f88d761d.jpg)

Figure 1 Globally consistent reconstruction from a long panoramic sequence. Our method reconstructs long-range panoramic sequence along a 1.2 km route by coupling sparse mapping with feed-forward geometry.

## Abstract

Long-sequence panoramic reconstruction requires detailed scene geometry and globally consistent camera poses across distant revisits. Panoramic SLAM and incremental SfM rely on feature correspondences, but repeated, similar structures and small partial overlap make reliable long-range connections dificult to establish, leaving accumulated drift unresolved. Feed-forward models ofer detailed local geometry, but their high computational costs limit inference window size, necessitating additional constraints for global consistency. This motivates combining continuous feature-based tracking and sparse mapping with rapid, high-quality local reconstruction from feed-forward models to achieve accurate localization and globally consistent dense mapping over long sequences. We therefore propose a panoramic reconstruction framework that couples an ORB-based sparse panoramic mapping frontend with a fine-tuned feed-forward model. Our key insight is that predicted local geometry enables 3D verification of feature matches across distant revisits, yielding long-range camera constraints even between regions. We calibrate predicted depths with sparse landmarks and use verified connections to optimize the global pose graph, updating dense geometry with the corrected camera poses. We also introduce PanoRecon, a seven-scene benchmark with repetitive architecture, large and small loops, and reference camera poses and LiDAR geometry for evaluating global consistency. Our method achieves the lowest mean absolute trajectory error and high F-1 Score among the evaluated methods. Project Page: HoloRecon.

## 1 Introduction

Panoramic reconstruction recovers camera trajectories and dense geometry from equirectangular projection (ERP) images covering the full surroundings. Along extended routes through buildings and streets, reconstruction requires detailed surfaces and consistent placement of observations, particularly across distant revisits.

Feature-based panoramic SLAM and incremental structure-from-motion (SfM) register cameras and maintain sparse landmarks through image correspondences. SLAM systems use loop closure to correct accumulated drift, with acceptance thresholds controlling which revisits enter global optimization [2, 28]. Relaxed thresholds admit false associations, whereas strict thresholds reject valid loops with limited matching support. Repeated facade textures and symmetric layouts make distinct places visually similar, while distant revisits often share only partial overlap. False loops distort the map, and missed loops leave drift uncorrected, producing misaligned structural layers. Reliable global correction of panoramic reconstruction therefore requires stronger evidence for distinguishing true revisits from repeated scene patterns.

Feed-forward reconstruction rapidly estimates camera poses and detailed local geometry from panoramic views, as demonstrated by PanoVGGT [8] and panoramic adaptations of Pi3 [34]. Memory and computational costs limit the number of jointly processed views, while separate inference windows still require global constraints to remain consistent. This motivates combining continuous feature-based tracking and mapping with rapid local feed-forward reconstruction for accurate localization and globally consistent reconstruction over long sequences.

Based on the above mentioned discussion, we therefore introduce an incremental panoramic reconstruction framework that couples ORB-based sparse mapping with a fine-tuned feed-forward panoramic model (Fig. 1). Our key insight is that the predicted local geometry can enable 3D verification of feature matches across distant revisits. We first calibrate predicted geometry with sparse landmarks and verify a common relative transform across views to establish long-range camera constraints. Accepted connections further drive posegraph correction, which establishes shared observations for panoramic bundle adjustment with a sparse diferentiation GPU backend. After refinement, we recalibrate cached depths and update each dense point using its source camera’s corrected pose, without repeating inference.

To evaluate global consistency under these conditions, we further introduce the PanoRecon benchmark, comprising seven real-world scenes with repeated architectural textures and geometric structures. Every scene contains multiple revisits separated by long travel distances. We also collect LiDAR data, providing reference camera poses and dense geometry for evaluating trajectory accuracy and reconstruction consistency across distant revisits.

Our contributions are as follows:

• We propose a long-range panoramic reconstruction framework that couples feature-based sparse mapping with feed-forward geometry to recover globally consistent camera trajectories and detailed dense maps.

• We introduce a long-range global connection mechanism that integrates feed-forward panoramic geometry with feature matching to enable 3D verification and establish camera constraints without requiring shared sparse landmarks between revisited regions.

• We introduce the PanoRecon benchmark with repetitive architecture and long-range revisits to evaluate trajectory accuracy and dense reconstruction consistency.

## 2 Related Work

## 2.1 Structure from Motion

SfM recovers cameras and structure through incremental registration [9, 26] or global motion estimation [20, 24, 37]. Resent study extends these operations to spherical imagery [3], including spherical pose estimation and bundle adjustment [13], dense matching [14], and structural constraints from lines and architectural layouts [10, 12]. These advances improve registration and geometry estimation. Distant regions with partial overlap still require reliable associations for global refinement, motivating our use of predicted depths to support feature correspondences across views.

## 2.2 Visual Odometry and Panoramic SLAM

Visual odometry estimates sequential motion, with direct methods optimizing photometric error [5], while feature-based, direct, and learned SLAM maintain maps with multi-view constraints [6, 21, 29]. Open-VSLAM [27] supports purely visual equirectangular mapping. Visual–inertial estimation [25] extends to panoramic and spherical multi-camera observations [35, 36, 38, 41], while HDPV-SLAM [1] incorporates LiDAR depth. Their global accuracy depends on distinguishing true revisits from repeated appearance.

## 2.3 Feed-Forward Reconstruction

Feed-forward geometry and matching models [16, 31, 33, 34] supply dense reconstruction priors. CUT3R maintains persistent state [32], SLAM3R learns local-to-world registration [17], and StreamVGGT and InfiniteVGGT use causal memory [42, 45]. MASt3R-SLAM, VGGT-SLAM, VGGT-Long, and VGGT-SLAM 2.0 couple predictions with graph optimization and loop closure [4, 18, 19, 22]. VGGT-Long processes overlapping chunks, while VGGT-SLAM 2.0 uses attention to verify revisits. For panoramas, PanoVGGT [8] predicts poses and geometry, while HALO-SLAM [40] exploits attention, and dense geometry for tracking and loop verification.

## 3 Method

Given an ERP image sequence, we reconstruct camera poses and dense geometry by combining a ORB-based sparse map with local feed-forward predictions (Fig. 2). We first calibrates and places the dense predictions in a common coordinate (Sec. 3.1). We then use these geometries to verify feature matches across views and construct long-range camera constraints (Sec. 3.2). Optimizing the sparse map with these constraints updates the cameras and landmarks, which we use to recalibrate and reposition the cached dense points (Sec. 3.3).

## 3.1 Anchoring Feed-Forward Geometry to the Sparse Map

Local predictions and the sparse map. An ORB-based ERP frontend [28] incrementally estimates camerato-world poses $T _ { i } = \left( R _ { i } , \mathbf { c } _ { i } \right)$ and sparse landmarks $\mathbf { L } _ { j }$ , where i indexes keyframes, j indexes landmarks, and $T _ { i } \mathbf { x } = R _ { i } \mathbf { x } + \mathbf { c } _ { i }$ . We group successive keyframes into overlapping windows and render their panoramas into calibrated perspective views. Pi3 [34], fine-tuned on Holo360D [23], jointly processes these views to predict local geometry. Each window defines a submap, whose predicted depths we anchor to the frontend’s cameras and landmarks.

We additionally replace the per-view sky segmentation network with a lightweight sky head supervised by Holo360D geometry ground truth. It predicts a sky mask directly on each ERP image and projects to the perspective views, reducing inference time per submap from 40 s to 13 s.

Camera-anchored dense representation. Global optimization of ORB frontend can update the cameras within a submap diferently. We therefore retain each prediction in its source camera coordinates and determine its world position from the current sparse-map pose. For pixel p in virtual view v of keyframe i and window w, let $\hat { z } _ { w i v } ( \mathbf { p } )$ be the optical-axis depth of the Pi3 local point. Its corresponding world point is

$$
{ \bf X } _ { w i v } ( { \bf p } ) = T _ { i } \left( s _ { w i } Q _ { v } \hat { z } _ { w i v } ( { \bf p } ) K _ { v } ^ { - 1 } \bar { \bf p } \right) , \qquad s _ { w i } = \exp ( a _ { w } + b _ { w i } ) ,\tag{1}
$$

where $\bar { \bf p }$ is the homogeneous pixel, $K _ { v }$ is the rendering intrinsic matrix, and $Q _ { v }$ rotates the virtual view into the panoramic camera. The virtual views share the panorama’s optical center and fixed relative rotations. The shared window log scale $a _ { w }$ aligns predicted depths to the sparse map, while the frame log-scale correction $b _ { w i }$ captures variation within the window. The calibrated rays and sparse-map poses determine where points are placed, while depth calibration estimates their scales.

Submap alignment from landmarks. We estimate these scales from reliable landmark observations pooled within each window. For landmark j, its observation maps to pixel $\mathbf { p } _ { i j v }$ in virtual view v, and the current pose gives its optical-axis depth $z _ { i j v } ^ { \mathrm { g e o m } }$ . Comparing this depth with the prediction gives the log ratio $y _ { w i j v } = \log [ z _ { i j v } ^ { \mathrm { g e o m } } / \hat { z } _ { w i v } ( \mathbf { p } _ { i j v } ) ]$ . We fit the shared scale and the vector of frame scale corrections $\mathbf { b } _ { w } = ( b _ { w i } ) _ { i }$ through

![](images/543504fc92eee324f70d84dcb2de65a1f8084be3a5474d202ac905439d2cd1aa.jpg)  
Figure 2 Overview of the proposed method. Sparse landmarks calibrate Pi3 depths, and registered panoramic cameras place them in the map. These predictions support cross-view ORB correspondences for verifying long-range camera constraints. Global correction and observation fusion refine the sparse map, which recalibrates and repositions the cached dense geometry.

$$
\operatorname* { m i n } _ { a _ { w } , \mathbf { b } _ { w } } \sum _ { ( i , j , v ) \in \mathcal { A } _ { w } } \rho ( y _ { w i j v } - a _ { w } - b _ { w i } ) + \frac { \lambda _ { p } } { 2 } \sum _ { i } b _ { w i } ^ { 2 } + \frac { \lambda _ { s } } { 2 } \sum _ { ( i , k ) \in \mathcal { N } _ { w } } ( b _ { w i } - b _ { w k } ) ^ { 2 } .\tag{2}
$$

Here $A _ { w }$ contains reliable observations with positive geometric and predicted depths, $\rho$ is the Huber penalty, and $\mathcal { N } _ { w }$ links adjacent frames with enough valid landmark observations. The weights $\lambda _ { p }$ and $\lambda _ { s }$ respectively constrain frame scale corrections toward the shared scale and encourage temporal smoothness. Log ratios measure proportional depth errors across scene distances. All frames share the window scale, with framespecific corrections estimated from suficient landmark support. After robust fitting, we align depths at common source pixels in overlapping windows, as detailed in Appendix B.1. Substituting the calibrated scales into Eq. (1) expresses each submap in the ORB landmarks’ coordinates for the connection estimation below.

## 3.2 Verifying Long-Range Connections with Predicted Geometry

We use the aligned submaps to identify and verify overlap between distant observations. For a new submap B, we propose historical submaps A using voxel overlap and line-of-sight visibility at their current map positions (Figure 3). Expanded spatial bounds and neighboring historical windows broaden this candidate set. Within each candidate pair, we compute mutual ORB matches across all pairs of keyframes with predicted depths. Combining matches across image pairs lets us use overlap distributed across views.

Depth-supported similarity estimation. To check their geometric consistency, we back-project matched keypoints using the calibrated depths from Sec. 3.1 (Fig. 3). A sample’s distance from its source camera is $s _ { w i } \hat { z } _ { w i v } ( \mathbf { p } ) \lVert K _ { v } ^ { - 1 } \bar { \mathbf { p } } \rVert _ { 2 }$ . Within each submap, we project samples into their source camera’s ERP image and take the median distance near each feature, denoted $d _ { i } ( \mathbf { u } )$ for ERP pixel u. With unit bearing b(u), the matched feature becomes ${ \bf P } _ { i } ( { \bf u } ) = T _ { i } [ d _ { i } ( { \bf u } ) { \bf b } ( { \bf u } ) ]$ in world coordinates. The resulting point correspondences from keyframes $a \in A$ and $b \in B$ are combined to estimate $S _ { A  B } \in \mathrm { S i m } ( 3 )$ by RANSAC and least-squares refitting [30], and define submaps’ rotation, translation, and positive scale. It aligns corresponding points through $\mathbf { P } _ { a } \approx S _ { A  B } \mathbf { P } _ { b }$ , correcting the displacement caused by drift despite their shared map coordinates.

Cross-view geometric verification. Repeated structures can produce plausible local matches, so we test whether a common similarity explains correspondences across multiple views. We check bidirectional distance and bearing agreement, nondegenerate geometry, and distinct feature support from multiple keyframes on both sides. For large corrections, we divide the current submap’s keyframes into groups with balanced numbers of matches before fitting. We estimate a transform from a group’s matches and verify it using matches in the other, then repeat with the groups exchanged. Both transforms must agree with the estimate from all matches. Verified transforms from neighboring historical submaps must also agree before acceptance. Appendix B.2 specifies these checks and their thresholds.

Camera constraint construction. We convert each verified submap transform into relative camera constraints for the sparse map. We select image pairs with enough inliers and distinct camera endpoints, then form the relative constraint mapping camera a to camera $b ,$

$$
\widehat { G } _ { b  a } = T _ { b } ^ { - 1 } S _ { A  B } ^ { - 1 } T _ { a } ,\tag{3}
$$

using the camera poses at the time the constraint is formed. The measurement remains fixed during subsequent optimization. The inverse similarity maps points reconstructed from submap A onto those from submap B before conversion into camera b. We cap the total weight for each verified submap transform based on its inlier count and divide this weight among the selected camera constraints. These weighted constraints enter the pose graph below.

![](images/2b8ee52b7d817105d98af545fc253cf2b194377ee17611feba17fc56b8b6257f.jpg)  
Figure 3 Cross-view evidence for a global connection.

## 3.3 Propagating Global Corrections to Sparse and Dense Geometry

Graph correction and observation fusion. We insert the relative camera constraints from Eq. (3) into the similarity pose graph alongside existing sparse-map constraints. We optimize the graph, remove incompatible new edges, and re-solve to update camera poses and landmark positions. In the corrected geometry, standard projection-based landmark fusion [21] runs bidirectionally between the cameras connected by accepted constraints and a limited number of their covisible neighbors. Merging duplicate landmarks establishes observations shared across revisited regions. We denote the updated camera–landmark observation set by ${ \mathcal { O } } ,$ which supplies cross-region reprojection factors for global BA.

GPU-accelerated panoramic bundle adjustment. The shared observations allow BA to jointly refine the graph-corrected poses $T _ { i }$ and landmarks $\mathbf { L } _ { j }$ . We use the original ERP feature pixels $( u _ { i j } , v _ { i j } )$ , retaining one pose per panorama so that BA directly refines the sparse map used to align dense predictions. As long-range connections enlarge this observation graph, repeated global refinement becomes costly, motivating GPU acceleration. With predicted pixels $( \hat { u } _ { i j } , \hat { v } _ { i j } ) = \pi _ { \mathrm { E R P } } ( T _ { i } ^ { - 1 } \mathbf { L } _ { j } )$ , where π<sub>ERP</sub> is the panoramic projection, we minimize

$$
\operatorname* { m i n } _ { \{ T _ { i } , \mathbf { L } _ { j } \} } E _ { \mathrm { B A } } = \sum _ { ( i , j ) \in \mathcal { O } } \psi _ { \delta _ { b } } \big ( \sigma _ { i j } ^ { - 2 } \| \mathbf { r } _ { i j } \| _ { 2 } ^ { 2 } \big ) , \qquad \mathbf { r } _ { i j } = \big [ \mathrm { w r a p } _ { W } \big ( u _ { i j } - \hat { u } _ { i j } \big ) , ~ v _ { i j } - \hat { v } _ { i j } \big ] ^ { \top } .\tag{4}
$$

Here $\psi _ { \delta _ { b } }$ is the Huber loss on the squared weighted residual norm, with transition at $\delta _ { b } ^ { 2 }$ , and $\sigma _ { i j } ^ { - 2 }$ is the feature-pyramid inverse variance. For ERP width W, $\mathrm { w r a p } _ { W } ( q ) = q - W \lfloor q / W + 1 / 2 \rfloor$ measures horizontal error across the image seam. Starting from the corrected geometry with the root pose fixed, sparse automatic diferentiation [43] linearizes the residuals. After landmark elimination, GPU preconditioned conjugate gradients solve the reduced camera system without storing the explicit Schur complement. Appendix B.3.2 details the solver, and Table 3 reports its eficiency.

Updating cached dense geometry. The refined cameras and landmarks also change the depth references used in Sec. 3.1. We recompute the landmark depths, refit the scales in Eq. (2), and place the cached predictions using the updated $T _ { i }$ and $s _ { w i }$ in Eq. (1). Each dense point is updated using its source camera’s optimized pose and recalibrated depth, without repeating Pi3 inference. At each submap boundary, any triggered global refinement and dense updates finish before frame input resumes, so subsequent tracking and connection estimation use the corrected map.

<table><tr><td>Dataset property</td><td>Value</td></tr><tr><td>Number of scenes</td><td>7</td></tr><tr><td>Mean video duration</td><td>11.49 min</td></tr><tr><td>Frame sampling interval</td><td>0.1 s</td></tr><tr><td>Mean trajectory length (km) 0.87 (0.26–1.69)</td><td></td></tr><tr><td>Reprojection RMSE (pix)</td><td>0.1917</td></tr><tr><td>Loop structure</td><td>Large and small loops</td></tr></table>

(a) Dataset statistics

![](images/2d3af491c33fe9c5c727f7b61b79cb34f65bca26cdbfaac34c6695ca7a37dbf6.jpg)  
(b) Dataset example  
Figure 4 Dataset statistics and GT examples of the PanoRecon benchmark.

## 4 The PanoRecon Benchmark

We independently capture panoramic videos and RTK-assisted LiDAR data along the same planned routes in seven scenes with low-texture surfaces and repeated architecture. Each route includes a large loop and distant revisits with limited overlap, testing reconstruction methods’ ability to correct accumulated drift.

Reference geometry and camera poses. We optimize the LiDAR trajectory with RTK corrections and export the point cloud and calibrated poses of its three onboard cameras. In RealityScan, these define LiDAR reference images with fixed poses. ERP frames sampled every 0.1 s are split into calibrated perspective rigs and globally matched to these images. We optimize only the ERP rig poses, then recover one pose per panorama using the rig extrinsics. The reference poses share the LiDAR’s metric coordinate system. The mean reprojection RMSE is 0.1917 pixels (Fig. 4). Appendix A.1 details acquisition and processing.

Reconstruction evaluation. For a reconstruction in frame E, trajectory alignment to the reference poses gives the similarity $S _ { L  E }$ into the LiDAR frame L. We apply it to the reconstructed geometry and refine registration with ICP. Following Tanks and Temples [15], bidirectional nearest-neighbor distances yield precision, recall, and F-score. The complete transformation is

$$
\mathbf { x } _ { L } ^ { \mathrm { e v a l } } = H _ { \mathrm { I C P } } S _ { L  E } \mathbf { x } _ { E } ,\tag{5}
$$

where $H _ { \mathrm { I C P } }$ refines cloud alignment for geometric evaluation.

## 5 Experiments

Implementation. We process submaps of 16 panoramic keyframes with four shared keyframes between consecutive submaps. Depth scale calibration uses a log-ratio Huber threshold of 0.05. Connection verification uses bidirectional distance and bearing thresholds of 0.3 map units and 1<sup>◦</sup>, with support from at least three keyframes on each side. GPU bundle adjustment uses double precision and at most 250 PCG iterations with a tolerance of $1 0 ^ { - 5 }$ . Further settings are provided in Appendix B. All experiments are conducted on a single NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU.

Datasets. We use the mapping sequences from all four scenes of 360LOC [11]: Atrium, Concourse, Hall, and Piatrium. We also evaluate all four sequences (A–D) of YUTO MMS [44], a panoramic dataset covering 20.1 km and the Sequence C alone spans 9.137 km with more than 9,000 ERP images. Finally, we evaluate the seven PanoRecon scenes introduced in Sec. 4. Both YUTO MMS and the PanoRecon benchmark contain long trajectories and complex loop structures, which challenge sustained tracking and global drift correction.

Table 1 Quantitative comparison of localization. Mean scene-wise RMSE on 360LOC (4 scenes), YUTO MMS (4 sequences), and PanoRecon (7 scenes). OOM denotes GPU out-of-memory.
<table><tr><td rowspan="2">Method</td><td colspan="3">360LOC</td><td colspan="3">YUTO MMS</td><td colspan="3">PanoRecon</td></tr><tr><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td colspan="10">Feed-Forward</td></tr><tr><td>PanoVGGT (ERP)</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>Scal3R (Pinhole)</td><td>5.366</td><td>0.124</td><td>1.354</td><td>187.407</td><td>2.911</td><td>1.025</td><td>9.794</td><td>0.282</td><td>0.340</td></tr><tr><td colspan="10">SLAM</td></tr><tr><td>360DVO (ERP)</td><td>0.954</td><td>0.026</td><td>0.039</td><td>68.313</td><td>0.183</td><td>0.011</td><td>2.799</td><td>0.104</td><td>0.039</td></tr><tr><td>Open-vSLAM (ERP)</td><td>0.441</td><td>0.711</td><td>22.184</td><td>281.793</td><td>41.690</td><td>37.574</td><td>7.399</td><td>6.580</td><td>25.242</td></tr><tr><td>Stella-vSLAM (ERP)</td><td>0.706</td><td>0.047</td><td>0.147</td><td>82.173</td><td>0.656</td><td>0.131</td><td>7.424</td><td>1.329</td><td>0.470</td></tr><tr><td colspan="10">Structure-from-Motion</td></tr><tr><td>COLMAP (ERP)</td><td>0.355</td><td>0.020</td><td>0.037</td><td>124.490</td><td>0.814</td><td>0.066</td><td>3.510 (6/7)</td><td>2.575 (6/7)</td><td>1.452 (6/7)</td></tr><tr><td>COLMAP (Pinhole)</td><td>0.399</td><td>0.021</td><td>0.037</td><td>113.317</td><td>25.949</td><td>1.046</td><td>1.736</td><td>0.099</td><td>0.036</td></tr><tr><td>Spherical-SfM (ERP)</td><td>0.491</td><td>0.024</td><td>0.038</td><td>21.186</td><td>3.312</td><td>3.989</td><td>11.779</td><td>6.915</td><td>1.418</td></tr><tr><td>Ours (ERP)</td><td>0.270</td><td>0.030</td><td>0.044</td><td>3.527</td><td>0.052</td><td>0.022</td><td>1.697</td><td>0.111</td><td>0.086</td></tr></table>

![](images/1c1b333dce4f66c610200b7c5a089bc3b4027a3b165a0fe35ca584f4cc2ab1d7.jpg)  
Figure 5 Qualitative comparison of localization.

Baselines. We compare against PanoVGGT [8] and Scal3R [39] for feed-forward reconstruction, 360DVO [7], Open-vSLAM [27], and Stella-vSLAM-Dense [28] for panoramic visual odometry and SLAM, and COLMAP [26] and Spherical-SfM for SfM (Table 1). Scal3R receives distortion-free perspective views rendered from ERP images using COLMAP’s oficial panorama renderer, with a 90<sup>◦</sup> field of view (front view). We evaluate COLMAP in two configurations: directly on ERP images with an equirectangular camera model, and on 12 such perspective overlapping views per panorama with known intrinsics and fixed per-ERP rig extrinsics.

Metrics. We evaluate camera poses after one global Umeyama Sim(3) alignment to the reference. ATE, translational RPE (RPE-t), and rotational RPE (RPE-r) are reported as RMSE. For dense reconstruction, we apply the trajectory similarity to the point cloud. We report mean scene-wise precision, recall, and F-score at 20 cm on 360LOC, YUTO MMS, and PanoRecon (Table 2). Appendices D and E provide per-scene results and evaluation details.

## 5.1 Trajectory Comparison

Our method achieves the lowest mean ATE on all three datasets, with the largest advantage on YUTO MMS (Table 1). False loop closures distort ORB-based Stella-vSLAM trajectories on Residence2 and Sequence C, where buildings share similar appearance and geometry (Fig. 5). Scal3R exhibits scale drift on large scenes with one perspective view per panorama, as does COLMAP during sequential reconstruction in both ERP and pinhole configurations. Our trajectories follow the reference street layouts and loop structures. On our single GPU, PanoVGGT [8] ran out of memory (OOM) on every evaluated scene under the input settings in Appendix A.2. HALO-SLAM [40] had no public implementation at evaluation time.

Table 2 Quantitative comparison of dense reconstruction. Mean scene-wise precision (P), recall (R), and F-score (F) in percent, at 20 cm on 360LOC (4 scenes), YUTO MMS (4 sequences), and PanoRecon (7 scenes). Higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="3">360LOC</td><td colspan="3">YUTO MMS</td><td colspan="3">PanoRecon</td></tr><tr><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td></tr><tr><td>Stella-vSLAM-Dense (ERP)</td><td>62.42</td><td>58.23</td><td>58.68</td><td>23.78</td><td>14.52</td><td>17.24</td><td>36.61</td><td>31.10</td><td>31.97</td></tr><tr><td>COLMAP (Pinhole)</td><td>81.27</td><td>44.79</td><td>57.25</td><td>29.15</td><td>13.23</td><td>17.28</td><td>71.32</td><td>51.83</td><td>59.75</td></tr><tr><td>Ours (ERP)</td><td>79.22</td><td>52.87</td><td>62.04</td><td>60.56</td><td>40.77</td><td>48.71</td><td>64.22</td><td>62.36</td><td>63.16</td></tr></table>

![](images/b5dc56316c33faeac78031067c1c0e7077ee01d94f1cd00c5eda4af7743058b0.jpg)

![](images/5f12404aec2301d80796300d6fcb11bb54000b5c2ef4b21459224b33b6d57760.jpg)  
GT LiDAR

![](images/1b1c458c15ea3dde6d72f89ba364c70fc96bff60da21f85d75a885a43bdc6011.jpg)  
Ours

![](images/7e8e9541e132b5b3be3bb766b7e1df8c330ef646fa804d9ab3e32897795daa0e.jpg)  
Stella-vSLAM Dense

![](images/e2946dfcdcdcb72715ac5e1c8582039b89592763d095c7a5722e2e1ac0815a91.jpg)  
COLMAP (Pinhole)  
Figure 6 Point-cloud comparison on the Residence2 and Road scenes of the PanoRecon benchmark.

## 5.2 Dense Reconstruction Comparison

Accurate camera poses keep local predictions aligned across distant revisits, reducing drift-induced misalignment in the global point cloud. Our largest F-score gains occur on the long-range YUTO MMS sequences (Table 2), consistent with the trajectory results. Feed-forward geometry also improves local completeness on weakly textured walls and reflective surfaces, where photometric matching is unreliable. In these regions, the PatchMatch-based COLMAP and Stella-vSLAM-Dense baselines exhibit holes and noisy geometry, while our method recovers more continuous surfaces (Fig. 6 and supplementary Figs. 10–12). Appendix E reports per-scene results.

## 5.3 Runtime.

Appendix C.1 provides the runtime analysis and stagewise comparison on YUTO MMS Sequence C. The GPU backend reduces the cost of global refinement while preserving comparable final reprojection error (Table 3). Matrix-free Schur optimization makes the growing observation graph more eficient to solve, supporting repeated correction as long-range connections accumulate.

Table 3 CPU and GPU bundle adjustment.
<table><tr><td>Backend</td><td>(s)↓</td><td>Solve Total BA (s)↓</td><td>Speedup↑ RMSE↓</td><td>Reproj.</td></tr><tr><td>CPU (OpenMP,</td><td></td><td></td><td></td><td></td></tr><tr><td>32 threads)</td><td>19.430</td><td>32.649</td><td>1.00×</td><td>0.8466</td></tr><tr><td>GPU (CUDA Schur)</td><td>2.244</td><td>7.581</td><td></td><td>4.31×0.8420</td></tr></table>

## 5.4 Ablation Studies

Efect of global constraints and bundle adjustment. In the Residence2 submap experiment, BA alone changes little, while a single camera constraint with BA provides most of the ATE reduction (Table 5). Additional constraints improve ATE further. Graph correction improves global alignment, and BA restores local motion consistency, showing how a few constraints can benefit reconstruction through joint refinement.

Table 4 Effect of global constraints. (Residence2 full trajectory)
<table><tr><td>Configuration</td><td>ATE RPE-t RPE-r Reproj. Long-range (m)↓ (m)↓ (°)↓</td><td>(px)↓</td><td>observations</td></tr><tr><td>No connection</td><td>1.358 0.061 0.045</td><td>1.623</td><td>160,186</td></tr><tr><td>PG</td><td>0.977 0.060 0.046</td><td>1.659</td><td>160,186</td></tr><tr><td>PG + BA</td><td>0.922 0.0570.045</td><td>1.620</td><td>160,186</td></tr><tr><td>PG + fusion</td><td>0.877 0.060 0.046</td><td>1.760</td><td>293,827</td></tr><tr><td>PG + fusion + BA 0.708 0.048 0.045</td><td></td><td>1.662</td><td>293,827</td></tr></table>

![](images/a140f68449ae6488418b720083297477dcdc8a177bdeba272b2e7bff7c3e734a.jpg)  
Figure 7 Efect of global constraints.

Table 5 Global connections and joint Table 6 Submap alignment. Table 7 Cross-view support and verifirefinement.( Submap 6→60; 16→49) (Submap 6→60; 16→49) cation. ( Submap 6→60; 16→49)
<table><tr><td>Edges Weights BA</td><td></td><td></td><td>ATE (m)↓</td><td>RPE-t (m)↓</td><td>RPE-r (°)↓</td><td>F-1 (%)↑</td><td>Alignment</td><td>P↑</td><td></td><td>R↑ F − 1 ↑</td><td>Submap Matching / pair validation</td><td></td><td>Image pairs Inliers</td><td>Support A/B</td><td>Test</td></tr><tr><td>0</td><td></td><td>Off</td><td>0.370</td><td>0.0128</td><td>0.0177</td><td>68.36</td><td>Pose-derived scale</td><td>85.14 61.22</td><td></td><td>71.236→ 60</td><td>Sampled</td><td>16</td><td>58</td><td>1/1</td><td>Fail</td></tr><tr><td>0</td><td></td><td>On</td><td>0.371</td><td>0.0129</td><td>0.0175</td><td>68.36</td><td></td><td>85.08 61.93</td><td></td><td>71.68 6→60</td><td>All frames</td><td>256</td><td>1,843</td><td>10/14</td><td>Pass</td></tr><tr><td>5</td><td>Inlier</td><td>Off</td><td>0.296</td><td>0.0633</td><td>0.0939</td><td>71.76</td><td>Whole-submap Sim(3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>Inlier</td><td>On</td><td>0.185</td><td>0.0095</td><td>0.0176</td><td>71.49</td><td></td><td></td><td></td><td>16→ 49</td><td>Sampled</td><td>32</td><td>259</td><td>3/4</td><td>Fail</td></tr><tr><td>2</td><td>Inlier</td><td>On</td><td></td><td>0.180 0.0093</td><td>0.0176</td><td>71.59</td><td>Sparse window scale</td><td>86.16</td><td>60.72</td><td>71.2416→49</td><td>All frames alternating</td><td>256</td><td>3,990</td><td>6/14</td><td>Fail</td></tr><tr><td>5</td><td>Uniform On</td><td></td><td>0.170</td><td>0.0094</td><td>0.0176</td><td>71.81</td><td>Sparse per-frame</td><td>86.48</td><td>61.62</td><td>71.97</td><td>groups</td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>Inlier</td><td>On 0.165</td><td></td><td>0.0094</td><td>0.017671.97</td><td></td><td>scale (ours)</td><td></td><td></td><td>16→ 49</td><td>All frames / balanced groups</td><td></td><td>256 3,990</td><td>6/14</td><td>Pass</td></tr></table>

Observation fusion and global refinement. Low reprojection error after local BA can coexist with global drift. PG reduces ATE while slightly increasing reprojection error, consistent with optimizing camera constraints before pixel-level refinement (Table 4, Fig. 7). Fusion merges landmark tracks across revisits, increasing long-range support and changing the reprojection constraints. Subsequent BA reduces residuals and further improves trajectory accuracy.

Efect of depth scale calibration. Frame-specific calibration improves the precision–recall balance after registration, showing the value of accommodating scale variation within a prediction window (Table 6). Whole-submap alignment retains higher recall. The advantage of per-frame calibration lies in the fidelity of the registered geometry and F-score.

Multi-view support and geometric verification. On the selected Residence2 pairs (small overlap long-range connections), all-frame matching increases support, while balanced groups allow verification to pass at unchanged thresholds (Table 7). Predicted depth expands 3D support, and pooling views improves correct recall.

## 6 Conclusion

In this paper, we study how local feed-forward geometry can support global consistency in long-sequence panoramic reconstruction. The local geometry of feed-forward model can provide 3D evidence for verifying distant feature matches and establishing camera constraints. The key insight is this mutual dependence: sparse mapping anchors local geometry, and local geometry strengthens the constraints used to correct the sparse map. Our seven-scene PanoRecon benchmark evaluates trajectory accuracy and reconstruction consistency under repetitive architecture and long-range revisits using reference poses and LiDAR geometry. Experiments across 360LOC, YUTO MMS, and PanoRecon show the improved mean ATE and high fidelity dense reconstruction results.

Limitations and Future Work. Severe motion blur or prolonged low-texture segments can reduce the reliable feature observations needed for tracking and depth calibration. Our system also does not explicitly handle dynamic objects, such as moving vehicles and pedestrians, and does not yet achieve real-time reconstruction. Future work will focus on more robust correspondences and eficient feed-forward models for streaming reconstruction from 360<sup>◦</sup> video in dynamic scenes.

## References

[1] Mostafa Ahmadi, Amin Alizadeh Naeini, Mohammad Moein Sheikholeslami, Zahra Arjmandi, Yujia Zhang, and Gunho Sohn. Hdpv-slam: Hybrid depth-augmented panoramic visual slam for mobile mapping system with tilted lidar and panoramic visual camera. In 2023 IEEE 19th International Conference on Automation Science and Engineering (CASE), pages 1–8. IEEE, August 2023. doi: 10.1109/case56687.2023.10260361. URL http://dx.doi.org/10.1109/CASE56687.2023.10260361.

[2] Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M. M. Montiel, and Juan D. Tardos. Orbslam3: An accurate open-source library for visual, visual–inertial, and multimap slam. IEEE Transactions on Robotics, 37(6):1874–1890, December 2021. ISSN 1941-0468. doi: 10.1109/tro.2021.3075644. URL http: //dx.doi.org/10.1109/TRO.2021.3075644.

[3] Thiago L. T. da Silveira, Paulo G. L. Pinto, Jefri Murrugarra-Llerena, and Cláudio R. Jung. 3d scene geometry estimation from 360<sup>◦</sup> imagery: A survey. ACM Computing Surveys, 55(4):1–39, November 2022. ISSN 1557-7341. doi: 10.1145/3519021. URL http://dx.doi.org/10.1145/3519021.

[4] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggt-long: Chunk it, loop it, align it – pushing vggt’s limits on kilometer-scale long rgb sequences, 2026. URL https://arxiv.org/abs/2507.16443.

[5] Jakob Engel, Vladlen Koltun, and Daniel Cremers. Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(3):611–625, March 2018. ISSN 2160-9292. doi: 10.1109/tpami.2017.2658577. URL http://dx.doi.org/10.1109/TPAMI.2017.2658577.

[6] Xiang Gao, Rui Wang, Nikolaus Demmel, and Daniel Cremers. Ldso: Direct sparse odometry with loop closure. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2198–2204. IEEE, October 2018. doi: 10.1109/iros.2018.8593376. URL http://dx.doi.org/10.1109/IROS.2018.8593376.

[7] Xiaopeng Guo, Yinzhe Xu, Huajian Huang, and Sai-Kit Yeung. 360dvo: Deep visual odometry for monocular 360- degree camera. IEEE Robotics and Automation Letters, 11(3):3079–3086, 2026. doi: 10.1109/LRA.2026.3655280.

[8] Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, and Yujiao Shi. Panovggt: Feed-forward 3d reconstruction from panoramic imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 36444–36453, 2026.

[9] Jared Heinly, Johannes L. Schonberger, Enrique Dunn, and Jan-Michael Frahm. Reconstructing the world\* in six days \*(as captured by the yahoo 100 million image dataset). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2015.

[10] Howard Huang, Bharath Surianarayanan, Keifer Lee, Chenyu Wang, and Chen Feng. Savmap: Structureaided visual mapping of large-scale 2.5d manhattan wireframes from panoramic video, 2026. URL https: //arxiv.org/abs/2606.01939.

[11] Huajian Huang, Changkun Liu, Yipeng Zhu, Hui Cheng, Tristan Braud, and Sai-Kit Yeung. 360loc: A dataset and benchmark for omnidirectional visual localization with cross-device queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22314–22324, June 2024.

[12] Satoshi Ikehata, Ivaylo Boyadzhiev, Qi Shan, and Yasutaka Furukawa. Panoramic structure from motion via geometric relationship detection, 2016. URL https://arxiv.org/abs/1612.01256.

[13] San Jiang, Kan You, Wu Chen, Duojie Weng, and Yaxin Li. 3d reconstruction of spherical images based on incremental structure from motion. International Journal of Remote Sensing, 45(8):2596–2621, April 2024. ISSN 1366-5901. doi: 10.1080/01431161.2024.2334810. URL http://dx.doi.org/10.1080/01431161.2024.2334810.

[14] Dongki Jung, Jaehoon Choi, Yonghan Lee, and Dinesh Manocha. Im360: Large-scale indoor mapping with 360 cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 29040–29050, October 2025.

[15] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017.

[16] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 71–91, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-73220-1.

[17] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Real-time dense scene reconstruction from monocular rgb videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16651–16662, June 2025.

[18] Dominic Maggio, Hyungtae Lim, and Luca Carlone. Vggt-slam: Dense rgb slam optimized on the sl(4) manifold. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 129839–129867. Curran Associates, Inc., 2025. doi: 10.52202/085713-4324. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ bc65ab11abfbad890171109686233f4e-Paper-Conference.pdf.

[19] Dominic Rosario Maggio and Luca Carlone. VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction. In Proceedings of Robotics: Science and Systems, Sydney, Australia, July 2026. doi: 10.15607/RSS.2026.XXII.051.

[20] Pierre Moulon, Pascal Monasse, and Renaud Marlet. Global fusion of relative motions for robust, accurate and scalable structure from motion. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2013.

[21] Raul Mur-Artal and Juan D. Tardos. Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. IEEE Transactions on Robotics, 33(5):1255–1262, October 2017. ISSN 1941-0468. doi: 10.1109/tro. 2017.2705103. URL http://dx.doi.org/10.1109/TRO.2017.2705103.

[22] Riku Murai, Eric Dexheimer, and Andrew J. Davison. Mast3r-slam: Real-time dense slam with 3d reconstruction priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16695–16705, June 2025.

[23] Jing Ou, Zidong Cao, Yinrui Ren, Zhuoxiao Li, Jinjing Zhu, Tongyan Hua, Shuai Zhang, Hui Xiong, and Wufan Zhao. Holo360d: A large-scale real-world dataset with continuous trajectories for advancing panoramic 3d reconstruction and beyond. arXiv preprint arXiv:2604.22482, 2026.

[24] Linfei Pan, Dániel Baráth, Marc Pollefeys, and Johannes L. Schönberger. Global structure-from-motion revisited. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 58–77, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-73661-2.

[25] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4):1004–1020, August 2018. ISSN 1941-0468. doi: 10.1109/tro. 2018.2853729. URL http://dx.doi.org/10.1109/TRO.2018.2853729.

[26] Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[27] Shinya Sumikura, Mikiya Shibuya, and Ken Sakurada. Openvslam: A versatile visual slam framework. In Proceedings of the 27th ACM International Conference on Multimedia, MM ’19, pages 2292–2295. ACM, October 2019. doi: 10.1145/3343031.3350539. URL http://dx.doi.org/10.1145/3343031.3350539.

[28] Hartmut Surmann, Marc Thurow, and Dominik Slomma. Patchmatch-stereo-panorama, a fast dense reconstruction from 360<sup>◦</sup> video images. In 2022 IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), pages 366–372. IEEE, November 2022. doi: 10.1109/ssrr56537.2022.10018698. URL http://dx.doi.org/10. 1109/SSRR56537.2022.10018698.

[29] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 16558–16569. Curran Associates, Inc., 2021. URL https: //proceedings.neurips.cc/paper\_files/paper/2021/file/89fcd07f20b6785b92134bd6c1d0fa42-Paper.pdf.

[30] S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, April 1991. ISSN 0162-8828. doi: 10.1109/34.88573. URL http://dx.doi.org/10.1109/34.88573.

[31] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5306, June 2025.

[32] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10510–10522, June 2025.

[33] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697–20709, June 2024.

[34] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 10481–10497, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file 11a09e0aaa74867c6b0719c639fc09f8-Paper-Conference.pdf.

[35] Ze Wang, Kailun Yang, Hao Shi, Peng Li, Fei Gao, and Kaiwei Wang. Lf-vio: A visual-inertial-odometry framework for large field-of-view cameras with negative plane. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4423–4430. IEEE, October 2022. doi: 10.1109/iros47612.2022.9981217. URL http://dx.doi.org/10.1109/IROS47612.2022.9981217.

[36] Ze Wang, Kailun Yang, Hao Shi, Peng Li, Fei Gao, Jian Bai, and Kaiwei Wang. Lf-vislam: A slam framework for large field-of-view cameras with negative imaging plane on mobile agents. IEEE Transactions on Automation Science and Engineering, 21(4):6321–6335, October 2024. ISSN 1558-3783. doi: 10.1109/tase.2023.3324495. URL http://dx.doi.org/10.1109/TASE.2023.3324495.

[37] Kyle Wilson and Noah Snavely. Robust global translations with 1dsfm. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014, pages 61–75, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10578-9.

[38] Yiyang Wu, Xiaohu Zhang, Yanjin Du, Tongsu Zhang, Chujun Li, Siyang Chen, Guoyi Zhang, and Xiangpeng Xu. Panoair: A panoramic visual-inertial slam with cross-time real-world uav dataset, 2026. URL https: //arxiv.org/abs/2604.00852.

[39] Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, and Xiaowei Zhou. Scal3r: Scalable test-time training for large-scale 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21760– 21771, June 2026.

[40] Zhuang Xiong, Guohao Zhang, Chen Zhang, Zheyu Jiang, Yuchao Mei, Qingshan Xu, and Wenbing Tao. Look up and look back: Hidden attention and latent orientation in a frozen foundation model for panoramic slam, 2026. URL https://arxiv.org/abs/2608.00925.

[41] Yueteng Yang, Yusen Xie, Hao Wei, Qianhao Wang, Boyu Zhou, Fei Gao, Jun Ma, and Jinni Zhou. Sphere-vio: Fast and robust visual-inertial odometry via unified spherical representation for heterogeneous multi-camera systems, 2026. URL https://arxiv.org/abs/2606.29910.

[42] Shuai Yuan, Yantai Yang, Xiaotian Yang, Xupeng Zhang, Zhonghao Zhao, Lingming Zhang, and Zhipeng Zhang. Infinitevggt: Visual geometry grounded transformer for endless streams, 2026. URL https://arxiv.org/abs 2601.02281.

[43] Zitong Zhan, Huan Xu, Zihang Fang, Xinpeng Wei, Yaoyu Hu, and Chen Wang. Bundle adjustment in the eager mode. IEEE Transactions on Robotics, 42:2727–2744, 2026. ISSN 1941-0468. doi: 10.1109/tro.2026.3706557. URL http://dx.doi.org/10.1109/TRO.2026.3706557.

[44] Yiujia Zhang, SeyedMostafa Ahmadi, Jungwon Kang, Zahra Arjmandi, and Gunho Sohn. Yuto mms: A comprehensive slam dataset for urban mobile mapping with tilted lidar and panoramic camera integration. The International Journal of Robotics Research, 44(1):3–21, June 2024. ISSN 1741-3176. doi: 10.1177/02783649241261079. URL http://dx.doi.org/10.1177/02783649241261079.

[45] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming visual geometry transformer. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on

Learning Representations, volume 2026, pages 88055–88072, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/file/8da04a60948be713dc766f0c7e3a5b1f-Paper-Conference.pdf.

## A Additional Experiment Details

## A.1 Datasets.

360LOC. We use the four daytime mapping sequences of 360LOC [11]: Atrium, Concourse, Hall, and Piatrium. The released reference split contains 2,244 panoramas (491–632 per scene) at 6144 × 3072 resolution, with reference poses and LiDAR maps. Their indoor and outdoor settings, symmetric layouts, repeated structures, and moving pedestrians let us assess reconstruction consistency in visually ambiguous scenes.

For dense evaluation, we back-project the oficial per-ERP depth maps along spherical viewing rays and transform the resulting points into the world frame using the oficial camera poses. Valid points across mapping frames are merged by voxel averaging to form the reference point cloud.

YUTO MMS. We evaluate all four sequences (A–D) of YUTO MMS [44], captured by a vehicle-mounted panoramic camera. The released sequences cover 20.1 km and contain 24,360 panoramas. Sequence C alone spans 9.137 km over 25 minutes with 10,778 images. The routes traverse campus and residential areas with large and small loops, frequent turns, and dynamic trafic. This scale and revisit structure let us evaluate accumulated drift and global consistency over extended trajectories, using reference poses obtained through ofline GPS and LiDAR adjustment.

For dense evaluation, we construct the reference point cloud from the released LiDAR scans, ground-truth camera poses, and sensor calibration. We apply the calibrated LiDAR-to-camera transform, and accumulate the transformed scans in a common world frame.

![](images/02277c5be9f78eccbb540dd9ca71bc481ba3b0b697cc971c411b5341110b7c64.jpg)  
Figure 8 Visualization of ours PanoRecon benchmark ground-truth LiDAR geometry.

Our PanoRecon. The PanoRecon benchmark evaluates global consistency under repeated architecture and limited overlap between distant revisits. We select seven scenes with low-texture surfaces and buildings that share similar geometry and appearance (Fig. 8). Each planned route contains a large loop, smaller loops, and repeated visits to the same buildings after long travel distances and time intervals. At revisits, the buildings are viewed from farther away or at oblique angles and occupy only a small region of the ERP image. The resulting limited overlap tests whether reconstruction methods can establish long-range connections and correct accumulated drift.

Data acquisition. We independently acquire panoramic video and LiDAR data along the same planned route in each scene. The panoramic recordings use an Insta360 X4 at 5K resolution and 30 fps. We mount the camera on a 3 m extension pole to reduce the operator’s presence in the ERP images. We export the recordings as ERP videos and sample 2K frames every 0.1 s. The separate LiDAR survey uses RTK positioning and three onboard cameras for point-cloud colorization.

Data processing. The two acquisition sessions are registered through reference images tied to the LiDAR coordinate system. We first post-process the LiDAR SLAM trajectory with RTK diferential corrections and export the resulting point cloud, the three onboard cameras’ poses, and their intrinsic and extrinsic calibration. We import these data into RealityScan to render LiDAR reference images, whose camera poses remain fixed throughout registration.

Each ERP frame is split into perspective views with known intrinsics, a shared optical center, and fixed relative rotations. We import these views as a calibrated rig and globally match them to the fixed LiDAR reference images. Pose optimization updates only the panoramic rigs, with the reference cameras and rig calibration held fixed. We then convert the estimated perspective-view poses back to one ERP pose per frame using the rig extrinsics. This yields reference panoramic poses and LiDAR geometry in the same metric coordinate system.

## A.2 Baselines Settings.

All methods use 2K input images except PanoVGGT and Scal3R, which use their oficial default resizing settings.

Input sequences and evaluation frames. On 360LOC, SLAM methods process all mapping images and retain frame-wise poses for evaluation. On PanoRecon, SLAM methods use all images for localization and retain evaluation keyframes at 1 s intervals. On YUTO MMS, SLAM methods process the full image sequences, and evaluation poses for both SLAM and feed-forward methods are sampled every three images. For these vehicle sequences with varying speeds, sampling follows frame indices. SfM methods use all 360LOC mapping images, sample PanoRecon inputs every 1 s, and use every 3 images, following the corresponding evaluation sampling intervals.

Feed-forward inference. Feed-forward methods receive the full YUTO MMS sequences, except that Scal3R uses every third image on Sequences C and D, where full-sequence inputs cause GPU out-of-memory (OOM) errors. PanoVGGT encounters OOM on every evaluated scene with its oficial default resizing and our single-GPU setup.

## B Additional Method Implementation Details

We provide implementation details for depth scale calibration, long-range connection verification, and global refinement, following the order and notation of Sec. 3. Figure 9 shows intermediate reconstructions and how global correction resolves duplicated building surfaces.

## B.1 Local Reconstruction Details

## B.1.1 Sparse mapping and prediction windows

The panoramic frontend tracks ERP images through projection matching and camera pose optimization (Fig. 9(a)). New keyframes add observations and triangulated landmarks, which local BA refines before depth scale calibration.

We use the overlapping windows specified in Implementation. The first and last two inferred panoramas supply context for Pi3 and cached depths for connection matching. Scale fitting and point-cloud output use the interior frames.

We use the camera-to-world poses $T _ { i }$ defined in Sec. 3.1, with the camera’s y axis pointing downward. For an ERP image of width $W$ and height H, pixel $\mathbf { u } = ( u , v )$ has longitude $\theta = 2 \pi ( u / W - 1 / 2 )$ and upward-positive

![](images/261d81977d5efa4b949f3c6c4058e530e0b295f0b3e78df4a3b4e052b6ab1902.jpg)  
Figure 9 From online submaps to globally consistent geometry. (a) Online pose estimation and local reconstruction. (b) Submap alignment, with distinct colors indicating diferent submaps. (c) Candidate selection from geometric overlap. (d) Graph correction using verified long-range connections. (e) BA and realignment remove the duplicated building surfaces left by pose drift after online alignment.

latitude $\phi = \pi ( 1 / 2 - v / H )$ . Its unit bearing and the inverse projection are

$$
\begin{array} { r l } & { \mathbf { b } ( u , v ) = ( \cos \phi \sin \theta , - \sin \phi , \cos \phi \cos \theta ) ^ { \top } , } \\ & { \pi _ { \mathrm { E R P } } ( \mathbf { x } ) = \left[ W \left( \frac { 1 } { 2 } + \mathrm { a t a n 2 } ( x , z ) / ( 2 \pi ) \right) \right] , \quad \mathbf { x } = ( x , y , z ) ^ { \top } . } \end{array}\tag{6}
$$

The horizontal coordinate is periodic. Reconstruction distances use the sparse map’s scale, with metric scale recovered through alignment for evaluation.

Each virtual perspective view has known intrinsics $K _ { v } ,$ , rotation $Q _ { v }$ into the panoramic camera, and the same optical center as that camera. We take the optical-axis component zˆ of each Pi3 local point and back-project it along the calibrated ray $K _ { v } ^ { - 1 } \bar { \mathbf p }$ . After scale calibration, its distance from the camera is $d = s _ { w i } \hat { z } \lVert K _ { v } ^ { - 1 } \bar { \mathbf p } \rVert _ { 2 }$ We store the source keyframe, virtual view, pixel, and unscaled depth so that Eq. (1) can update its world position after optimization.

![](images/39f0f54f22f064b1c11fa15a1cbbe4b3e744f404706d85997b4c6045b6b1d117.jpg)

![](images/306abd0eae8bfbe297be1cc155d87de6bfc81ff49292aa7a84d4c54077fcca87.jpg)

![](images/545e867225cfc62d230bc9980f24519dc1d309610fc1a8c7f20c341190d81a60.jpg)

![](images/964e16052e6450fd301d71cffb17a428b295e1540ba244108d55d495840e53c9.jpg)

![](images/66fe01bb8e96070f3bb826b7178ef403d838d8de7ac7462210c318fe6520f645.jpg)

![](images/30405666b823a68a3f467a4552a18189a09913b9761173fab3ea385d5647f963.jpg)

![](images/4b2744f6ae47c95b8a650b76193d6a8fd6b785a7f1671ee60b069c28b13cde97.jpg)

![](images/db21c6ea1cddbe638e466ac36d6ab4e7845a0c2b1791a2058e500fce8d0886ee.jpg)  
Figure 10 Point-cloud comparison of building facades and local structures in the PanoRecon benchmark’s Building1 scene.

## B.1.2 Landmark observations and depth scale calibration

For landmark $\mathbf { L } _ { j }$ observed by keyframe i, the geometric depth in virtual view v is

$$
z _ { i j v } ^ { \mathrm { g e o m } } = \left[ Q _ { v } ^ { \top } R _ { i } ^ { \top } ( { \bf L } _ { j } - { \bf c } _ { i } ) \right] _ { z } ,\tag{7}
$$

where $[ \cdot ] _ { z }$ selects the optical-axis component. Comparing this depth with the prediction at the corresponding pixel gives the log ratio in Eq. (2). The fit estimates multiplicative depth scales, which are applied to the original predicted depths in Eq. (1), producing the aligned submaps shown in Fig. 9(b).

We filter landmark observations by positive depth, observation support, parallax, and reprojection error, and exclude masked pixels, view boundaries, and depth discontinuities. We repeat these checks against the updated sparse map before recalibration.

Before fitting, we filter log ratios by median absolute deviation. Frames with suficient landmark support define $A _ { w }$ , and windows with enough pooled observations are calibrated using Eq. (2). Temporal regularization links eligible frames that are adjacent in the sequence. The shared window scale initializes all frames, with $b _ { w i } = 0$ and frame corrections are fitted for eligible frames.

We solve Eq. (2) by iteratively reweighted least squares with $( \lambda _ { p } , \lambda _ { s } ) = ( 3 2 , 6 4 )$ and the Huber threshold specified in Implementation. The shared log scale $a _ { w }$ starts from the median log ratio and is fitted without a bound. Frame corrections start at zero and are clipped after fitting to ± log(1.05) relative to the shared scale.

Overlapping windows provide an additional scale adjustment from depths at common source keyframes, views, and pixels. Each shared keyframe contributes a median log-depth ratio, and their sample-count-weighted median gives the correction toward the preceding window.

![](images/c59b681db202a0fe5748ddb3b376c7242a92972248a74461781c4164cc185e8c.jpg)  
Figure 11 Point-cloud comparison in the Residence scene.

## B.2 Global Connection Details

## B.2.1 Candidate submap selection

We use calibrated submap points to restrict feature matching to spatial candidates (Fig. 9(c)). For occupied voxel sets $V _ { A } , V _ { B }$ and one-voxel dilation D, the overlap score is

$$
\eta _ { A B } = \operatorname* { m a x } \Biggl \{ \frac { | V _ { A } \cap \mathcal { D } ( V _ { B } ) | } { | V _ { A } | } , \frac { | V _ { B } \cap \mathcal { D } ( V _ { A } ) | } { | V _ { B } | } \Biggr \} .\tag{8}
$$

Taking the larger fraction accommodates partial overlap between submaps of diferent sizes. We use voxel size 0.25m in map units and require $\eta _ { A B } \ge 0 . 1 5$ . A line-of-sight check tests for occlusion along rays from camera centers toward the overlapping region, requiring a mean visibility fraction of at least 0.6.

To include nearby candidates displaced by drift, we expand robust axis-aligned submap bounds by 20% on each side. Intersecting bounds supply up to three additional candidates. We also consider historical windows within two indices of selected candidates, adding at most 12 pairs per new window while retaining all original visibility candidates. Within this spatial neighborhood, candidate submaps separated by at least four window indices undergo the matching and verification below.

## B.2.2 Feature matching and geometric verification

We match all inferred keyframes with available poses and depths, including frames excluded from point-cloud output. The median predicted camera distance in a local ERP neighborhood lifts each matched feature into 3D. Matches with valid depths on both sides form the correspondences in Sec. 3.2.

![](images/c2a3d96aaf59074edc831ad53169a3ac3eedcab490707790196c7a6377362eb5.jpg)  
Figure 12 Point-cloud comparison in the HPC scene.

Each match retains its source keyframes and feature indices to distinguish repeated use of a feature from distinct feature support. We pool the 3D correspondences to estimate $S = S _ { A  B }$ , with $S \mathbf { x } = s R \mathbf { x } + \mathbf { t }$ , using RANSAC followed by least-squares refitting [30].

For corresponding points $\mathbf { P } _ { a } , \mathbf { P } _ { b }$ and source camera centers $\mathbf { c } _ { a } , \mathbf { c } _ { b } ,$ geometric verification requires

$$
\begin{array} { r } { \| S \mathbf { P } _ { b } - \mathbf { P } _ { a } \| _ { 2 } \le \tau _ { d } , } \\ { \| S ^ { - 1 } \mathbf { P } _ { a } - \mathbf { P } _ { b } \| _ { 2 } \le \tau _ { d } , } \\ { \angle ( S \mathbf { P } _ { b } - \mathbf { c } _ { a } , \mathbf { P } _ { a } - \mathbf { c } _ { a } ) \le \tau _ { \theta } , } \\ { \angle ( S ^ { - 1 } \mathbf { P } _ { a } - \mathbf { c } _ { b } , \mathbf { P } _ { b } - \mathbf { c } _ { b } ) \le \tau _ { \theta } . } \end{array}\tag{9}
$$

Here $\angle ( \mathbf { x } , \mathbf { y } )$ is the angle between nonzero vectors, and $\tau _ { d }$ and $\tau _ { \theta }$ are the distance and bearing thresholds specified in Implementation. The bidirectional distance test checks each submap at its own scale. We refit the similarity from the surviving correspondences and recompute inliers before the following checks.

![](images/9993fd9adef0e786ec84a9eca7c37121356a96b41e328f1886741c785be88f41.jpg)  
Figure 13 Point-cloud comparison in the 360LOC Concourse scene.

## B.2.3 Validation across views and camera constraints

We require inliers from multiple views because repeated matches can refer to the same scene point. Table 8 specifies the required support, including distinct features and keyframes on each side. We reject nearly collinear inlier points on either side while allowing planar configurations.

Large corrections require additional validation using separate frame groups. Let h be the median spacing between consecutive available keyframe centers, and let $\mathcal { P } _ { B }$ contain the current submap’s frames with predicted depths. A correction is large if its rotation exceeds $3 ^ { \circ }$ , its mean displacement of these camera centers exceeds $h ,$ or its scale lies outside [0.95, 1.05].

Before fitting, we divide the current frames into two similarly sized groups with balanced match counts. We estimate a transform from each group and verify it on the other group’s matches. Each estimate ${ \cal S } _ { f } = ( s _ { f } , R _ { f } , \mathbf { t } _ { f } )$ must also agree with the estimate from all matches, $S = ( s , R , \mathbf { t } )$ :

$$
\begin{array} { c } { { \displaystyle | \log ( s _ { f } / s ) | \le 0 . 0 5 , } } \\ { { \displaystyle \mathrm { a n g l e } ( R _ { f } R ^ { \top } ) \le 3 ^ { \circ } , } } \\ { { \displaystyle \operatorname* { m a x } _ { i \in \mathcal { P } _ { B } } \| S _ { f } \mathbf { c } _ { i } - S \mathbf { c } _ { i } \| _ { 2 } \le \operatorname* { m a x } ( 2 h , 2 \tau _ { d } ) . } } \end{array}\tag{10}
$$

Here angle denotes the rotation angle. The groups use disjoint current frames and can share historical frames and scene content. The same transform-agreement criteria apply to verified historical submaps within two window indices that connect to the same current submap.

![](images/8f060b1898805aea68138918f1703909815171109976cefbd52ddd2ac90e3720.jpg)  
Figure 14 Point-cloud comparison in the YUTO MMS Sequence B scene.

From each accepted submap pair, we select image pairs with at least ten inliers in descending count order. Their camera endpoints must be distinct on each side, and duplicate camera pairs are excluded across selected submap pairs. At most three camera constraints are formed using Eq. (3). For $n _ { A B }$ inliers and $m _ { A B } > 0$ selected constraints, each constraint receives weight

$$
w _ { e } = \frac { \mathrm { c l i p } ( n _ { A B } / 3 , 1 0 , 1 0 0 ) } { m _ { A B } } .\tag{11}
$$

This normalization keeps each submap pair’s total weight fixed across diferent numbers of selected camera edges.

## B.2.4 Descriptor matching and cache reuse

The CPU and CUDA implementations use the same mutual ORB matching rules and cache descriptors and feature matches. After global correction, we recompute 3D coordinates and geometric verification from the updated poses and depths.

## B.3 Global Refinement Details

![](images/fe13d3bf4765b59d37802648bb272bcc0254a985c9d70635fea6647574e94819.jpg)

![](images/6b1ecf8bdba1a977721a3d33edee2ad8939b215565416da78f2a1236cafe7849.jpg)

![](images/5b277eff97ce2304974e0343d628f2a1c9df684384e40a39582e3a03954ebc4c.jpg)

![](images/3db18836b86a80bbf4a6d828252511485daf279942fd4c2a24e1939410690cf4.jpg)

![](images/0cb4d5631aeed4f55334b1183b5ce06ee5a5fb0376ae439c93c761c8810463ae.jpg)

![](images/44f8640c09c0b3ecc945fad5bd091bd0dd5dd07180f219a0a5eedb0b4ad55024.jpg)

![](images/6daf8f499b1387be666a21d1c0dd8fb2fc7b0b2bc3fab2123c8b633ba8eca100.jpg)

![](images/317d97efae46410b7379068464cf5c4da26fcd9a0b18ff4070d95b36caa8b049.jpg)

![](images/1cd7aae7271e0e72c6f81c67117e074020e0aa6e60e604a66aa8e27be73150c2.jpg)

![](images/02e78e545cf7b6387f503b7cbe5cee8d8a1f017a99300d5845ed009719796e9c.jpg)

![](images/3a1a44de4ec5001fe04badd051e0651b5a428e78a066f5da74b488f535e4f98c.jpg)  
Figure 15 Point-cloud comparison in the YUTO MMS Sequence C scene.

## B.3.1 Pose graph optimization and landmark fusion

The pose graph combines spanning-tree edges, strong covisibility edges, and the new camera constraints from Sec. 3.2 to correct the trajectory (Fig. 9(d)). Let $G _ { i } \in \mathrm { S i m } ( 3 )$ be a world-to-camera vertex, initialized from $T _ { i } ^ { - 1 }$ with unit scale. For edge set $\mathcal { E } ,$ relative measurements ${ \widehat { G } } _ { b  a } .$ , and weights $w _ { a b }$ , we fix the root vertex and minimize

$$
\operatorname* { m i n } _ { \{ G _ { i } \} } \sum _ { ( a , b ) \in \mathcal { E } } w _ { a b } \| \mathrm { L o g } ( \widehat { G } _ { b  a } G _ { a } G _ { b } ^ { - 1 } ) \| _ { 2 } ^ { 2 } .\tag{12}
$$

Here Log maps a similarity transform to its seven-dimensional tangent vector. Existing edges have unit weight, and new edges use Eq. (11). After optimization, we check each new edge’s relative error ${ \tilde { H _ { a b } } } = { \widehat { G } } _ { b  a } ^ { - 1 } G _ { b } G _ { a } ^ { - { \breve { 1 } } } =$ $\left( s _ { H } , R _ { H } , \mathbf { t } _ { H } \right)$ against

$$
\mathrm { a n g l e } ( R _ { H } ) \le 5 ^ { \circ } , \qquad \| \mathbf { t } _ { H } / s _ { H } \| _ { 2 } \le 2 , \qquad | \log s _ { H } | \le 0 . 1 .\tag{13}
$$

The translation threshold uses map units. Pruning inconsistent new edges is followed by graph re-optimization and residual checks on the retained new constraints.

For each optimized world-to-camera similarity, dividing its translation by its scale gives the corresponding rigid transform. Inverting this transform yields the updated camera-to-world pose $T _ { i } ^ { + }$ . A landmark associated

Table 8 Connection-verification thresholds. Distances use map units, and h is the median keyframe spacing.
<table><tr><td>Quantity</td><td>Requirement</td></tr><tr><td>Total inliers, small / large correction</td><td> $\ge 6 0 ~ / \ge 1 0 0$ </td></tr><tr><td>Inlier fraction, small / large correction</td><td> $\ge 0 . 3 5 ~ / \ge 0 . 5 0$ </td></tr><tr><td>Unique features per side</td><td>≥ 40</td></tr><tr><td>Supporting keyframes per side</td><td>≥ 3, each with  $\geq 1 0$  inliers</td></tr><tr><td>Scale / rotation / mean camera displacement</td><td> $\left[ 2 / 3 , 1 . 5 \right] / \leq 1 5 ^ { \circ } / \leq 5 0 h$ </td></tr><tr><td>Bidirectional distance / viewing angle</td><td> $\leq 0 . 3 ~ / \leq 1 ^ { \circ }$ </td></tr><tr><td>Estimation-group inliers</td><td> $\geq 3 0$ </td></tr><tr><td>Validation-group inliers / fraction</td><td> $\geq 3 0 \mathrm { ~ / ~ } \geq 0 . 4 0$ </td></tr><tr><td>Group and neighboring-transform agreement</td><td>Eq. (10)</td></tr></table>

with reference keyframe $r ( j )$ is updated by

$$
\mathbf { L } _ { j } ^ { + } = ( G _ { r ( j ) } ^ { + } ) ^ { - 1 } G _ { r ( j ) } ^ { 0 } \mathbf { L } _ { j } ^ { 0 } ,\tag{14}
$$

where superscripts 0 and + denote the states before and after graph optimization.

We then apply the frontend’s bidirectional projection matching between accepted camera endpoints and a limited set of their strongest covisible neighbors. Duplicate-landmark fusion and covisibility updates produce the shared observation set O for global BA.

## B.3.2 Panoramic bundle adjustment

We refine one rigid pose per panorama and the shared landmarks using the ERP reprojection objective in Eq. (4). The root pose is fixed, while monocular scale remains unconstrained. The horizontal residual wraps to $[ - W / 2 , W / 2 )$ , and the Huber threshold is $\delta _ { b } = \sqrt { 5 . 9 9 1 4 6 }$

Residuals for sparse automatic diferentiation. We implement this objective in the $\mathrm { B A E / P y P o s e }$ framework [43]. To express the robust loss as a sum of squared residuals, define $\mathbf { e } _ { i j } = \sigma _ { i j } ^ { - 1 } \mathbf { r } _ { i j }$ and $q _ { i j } = \lVert \mathbf { e } _ { i j } \rVert ^ { 2 }$ , and supply

$$
\mathbf { f } _ { i j } = \left\{ \begin{array} { l l } { \sqrt { \psi _ { \delta _ { b } } ( q _ { i j } ) / q _ { i j } } \mathbf { e } _ { i j } , } & { q _ { i j } > 0 , } \\ { \mathbf { 0 } , } & { q _ { i j } = 0 . } \end{array} \right. \qquad \sum _ { ( i , j ) \in \mathcal { O } } \| \mathbf { f } _ { i j } \| ^ { 2 } = E _ { \mathrm { B A } } .\tag{15}
$$

The CPU and GPU backends optimize the same scalar objective. The GPU implementation diferentiates the residual scaling, while g2o uses a robust Gauss–Newton approximation.

Matrix-free Schur optimization. Let $J _ { c }$ and $J _ { p }$ be the Jacobians of the stacked residual vector f with respect to free camera poses and landmarks. Diagonal damping gives camera and landmark blocks $U _ { \lambda }$ and $V _ { \lambda } .$ , with coupling $C = J _ { c } ^ { \top } J _ { p }$ . For right-hand sides $\bar { \mathbf { b } } _ { c } = - J _ { c } ^ { \top } \bar { \mathbf { f } }$ and $\mathbf { b } _ { p } = - J _ { p } ^ { \top } \mathbf { f }$ , eliminating landmark increments gives

$$
\begin{array} { r l } & { S _ { \lambda } \Delta \mathbf { c } = \mathbf { b } _ { c } - C V _ { \lambda } ^ { - 1 } \mathbf { b } _ { p } , S _ { \lambda } = U _ { \lambda } - C V _ { \lambda } ^ { - 1 } C ^ { \top } , } \\ & { V _ { \lambda } \Delta \mathbf { p } = \mathbf { b } _ { p } - C ^ { \top } \Delta \mathbf { c } . } \end{array}\tag{16}
$$

Here λ denotes damping, and $\Delta \mathbf { c }$ and $\Delta \mathbf { p }$ stack camera pose and landmark increments. The reduced camera operator is evaluated through sparse products:

$$
\begin{array} { r } { S _ { \lambda } \mathbf { x } = U _ { \lambda } \mathbf { x } - J _ { c } ^ { \top } \left[ J _ { p } \big ( V _ { \lambda } ^ { - 1 } \left[ J _ { p } ^ { \top } ( J _ { c } \mathbf { x } ) \right] \big ) \right] . } \end{array}\tag{17}
$$

This avoids storing $C$ or $S _ { \lambda }$ . The block-diagonal $V _ { \lambda }$ has $3 \times 3$ blocks that can be inverted separately. GPU preconditioned conjugate gradients solve the camera system using a preconditioner derived from $U _ { \lambda }$ . The implementation also uses preconditioned conjugate gradients to recover landmark increments from the second equation in Eq. (16).

Solver settings and map updates. The GPU solver uses the precision and PCG settings in Implementation, with trust-region damping and at most 100 outer iterations. After at least two iterations, relative objective improvement below $1 0 ^ { - 3 }$ stops refinement. Accepted steps have non-increasing objective values, and the final solution updates the camera poses and landmarks. The CPU backend uses analytic ERP Jacobians, g2o robust linearization, and a CSparse direct solver with OpenMP support. Initialization and local BA use the CPU backend, while the selected backend handles global BA.

## B.3.3 Dense reconstruction after global refinement

Updated poses and landmarks change both dense point positions and their depth references. For each afected window, we recheck its landmark observations, recompute Eq. (7), refit Eq. (2), and adjust overlapping predictions through their common samples. Equation (1) then places the cached predictions using the updated poses and scales, yielding the realigned geometry in Fig. 9(e).

At each submap boundary, accepted connections trigger graph correction, landmark fusion, and global BA before dense updates. Scheduled global BA also updates dense geometry. Frame input resumes once these operations finish.

## C Additional Experiment Results

## C.1 Runtime Analysis

Table 9 summarizes the instrumented reconstruction stages on YUTO MMS Sequence C. Our 257 Pi3 inference windows take 38.96 minutes, while 91 GPU BA solves take 0.87 minutes; connection matching and verification account for 56.60 minutes. In COLMAP, incremental reconstruction and BA take 618.52 minutes, and photometric and geometric MVS together take 609.85 minutes. We exclude separate image preparation, format conversion, and export. These historical runs use 10,778 frontend frames and 3,594 keyframes for our method, versus 3,593 sampled ERP images expanded into 43,116 perspective views for COLMAP.

Table 9 Key-stage runtime on YUTO MMS Sequence C. Times are in minutes, excluding separate preprocessing and export.
<table><tr><td colspan="2">Ours</td><td colspan="2">COLMAP (Pinhole)</td></tr><tr><td>Stage</td><td>Time</td><td>Stage</td><td>Time</td></tr><tr><td>Frontend processing</td><td>42.71</td><td>Feature extraction</td><td>7.21</td></tr><tr><td>Pi3 inference</td><td>38.96</td><td>Matching and verification</td><td>26.17</td></tr><tr><td>Connection matching</td><td>44.66</td><td>Rig verification</td><td>7.39</td></tr><tr><td>Connection verification</td><td>11.94</td><td>Incremental reconstruction + BA</td><td>618.52</td></tr><tr><td>Pose-graph optimization</td><td>10.08</td><td>Photometric MVS</td><td>227.33</td></tr><tr><td>GPU BA solve</td><td>0.87</td><td>Geometric MVS</td><td>382.52</td></tr><tr><td>Post-BA realignment</td><td>32.16</td><td>Fusion</td><td>261.72</td></tr><tr><td>Total</td><td>181.38</td><td>Total</td><td>1530.86</td></tr></table>

## C.2 Applying Submap Alignment to PanoVGGT

PanoVGGT’s joint inference exceeds GPU memory on the complete sequences in our evaluation, motivating a windowed reconstruction pipeline for long panoramic inputs. We therefore combine its panoramic predictions with the submap construction and retrieval workflow of VGGT-SLAM [8, 18]. Since this integration is our own implementation, we present it separately in the supplement as a Residence2 case study, rather than as an oficial baseline. The experiment examines whether ours insight of using predicted geometry to align submaps and propagate global corrections can also benefit this reconstruction pipeline.

Re-implementation. We process Residence2 images at 518 × 1036 resolution in windows advancing by 16 images as submaps, with one image shared between consecutive windows. We then retrieval supplies at most one historical image for joint inference with each current window. PanoVGGT predicts camera poses and dense geometry directly from the ERP images. We express these predictions in the first camera’s coordinate system and pass them to the VGGT-SLAM backend, retaining its sequential scale estimation and graph optimization. Valid points have finite coordinates and positive radial depth, since the panoramic checkpoint does not provide learned depth confidence. This produces the adapted-backend reference in Table 10.

![](images/bae9abcba34541d4c6ec5f7879f4a853e581c4b940feb51fe762c6529aebb95b.jpg)  
Figure 16 Applying Submap Alignment to PanoVGGT

Table 10 Adapt PanoVGGT with VGGT-SLAM on Residence2.
<table><tr><td>Configuration</td><td>(m)</td><td>(m)</td><td>(°)</td><td>ATE ↓ RPE-t ↓ RPE-r ↓ Precision ↑ Recall ↑ F-score ↑ (%)</td><td>(%)</td><td>(%)</td></tr><tr><td>Default SL(4)</td><td colspan="6">Failed at submap 57/59</td></tr><tr><td>Adapted SE(3)</td><td>26.831</td><td>0.430</td><td>3.616</td><td>2.31</td><td>45.83</td><td>4.40</td></tr><tr><td>Without loop factors SE(3)</td><td>29.970</td><td>0.294</td><td>0.112</td><td>4.55</td><td>51.09</td><td>8.35</td></tr><tr><td>Ours submap alignment Sim(3)</td><td>4.203</td><td>0.107</td><td>0.149</td><td>10.60</td><td>78.14</td><td>18.67</td></tr></table>

Geometry-based submap alignment. We transfer our submap alignment to this adapter using images shared across adjacent or retrieved windows. Corresponding pixels provide radial-depth ratios for robust relative scale estimation, while the predicted poses of the shared camera determine relative rotation and translation. These measurements form Sim(3) constraints for jointly optimizing submap poses and scales with a Huber loss, fixing the first submap. The optimized transformations update cached camera poses and dense points without rerunning PanoVGGT, propagating global corrections throughout the reconstruction.

Results. On Residence2, geometry-based submap alignment reduces ATE from 26.831 to 4.203 m and raises F-score from 4.40% to 18.67%. Removing loop factors alone yields the lowest rotational RPE but substantially worse trajectory and reconstruction accuracy.

## D Per-Scene Localization Results

Tables 11, 12, and 13 provide per-scene trajectory results for 360LOC, YUTO MMS, and PanoRecon, respectively.

## E Dense reconstruction evaluation

Tables 14, 15, and 16 report per-scene dense reconstruction results on 360LOC, YUTO MMS, and PanoRecon, respectively. Additional point-cloud comparisons cover PanoRecon (Figs. 10, 11, and 12), 360LOC Concourse (Fig. 13), and YUTO MMS Sequences B and C (Figs. 14 and 15).

Table 11 360LOC localization by scene. ATE and RPE-t are in meters; RPE-r is in degrees. n is the number of associated poses (scene panels) or available/expected scenes (Mean). All errors are RMSE; lower is better.
<table><tr><td colspan="10">Concourse</td></tr><tr><td>Atrium Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>581</td><td>3.322</td><td>0.095</td><td>0.841</td><td>Scal3R</td><td>491</td><td>5.717</td><td>0.130</td><td>0.611</td></tr><tr><td>360DVO</td><td>581</td><td>0.654</td><td>0.044</td><td>0.034</td><td>360DVO</td><td>491</td><td>0.779</td><td>0.011</td><td>0.044</td></tr><tr><td>Open-vSLAM</td><td>580</td><td>0.255</td><td>0.691</td><td>22.408</td><td>Open-vSLAM</td><td>471</td><td>0.081</td><td>0.670</td><td>20.111</td></tr><tr><td>Stella-vSLAM</td><td>285</td><td>1.209</td><td>0.083</td><td>0.059</td><td>Stella-vSLAM</td><td>262</td><td>0.172</td><td>0.037</td><td>0.298</td></tr><tr><td>COLMAP (ERP)</td><td>581</td><td>0.201</td><td>0.043</td><td>0.033</td><td>COLMAP (ERP)</td><td>491</td><td>0.131</td><td>0.007</td><td>0.038</td></tr><tr><td>COLMAP (Pinhole)</td><td>581</td><td>0.232</td><td>0.043</td><td>0.033</td><td>COLMAP (Pinhole)</td><td>491</td><td>0.159</td><td>0.007</td><td>0.037</td></tr><tr><td>Spherical-SfM (ERP)</td><td>581</td><td>0.354</td><td>0.043</td><td>0.033</td><td>Spherical-SfM (ERP)</td><td>491</td><td>0.211</td><td>0.010</td><td>0.042</td></tr><tr><td>Ours</td><td>581</td><td>0.208</td><td>0.062</td><td>0.040</td><td>Óurs</td><td>491</td><td>0.169</td><td>0.016</td><td>0.040</td></tr><tr><td>Hall Piatrium</td><td colspan="9"></td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>540</td><td>11.008</td><td>0.172</td><td>3.758</td><td>Scal3R</td><td>632</td><td>1.417</td><td>0.099</td><td>0.205</td></tr><tr><td>360DVO</td><td>540</td><td>0.921</td><td>0.028</td><td>0.039</td><td>360DVO</td><td>632</td><td>1.460</td><td>0.021</td><td>0.038</td></tr><tr><td>Open-vSLAM</td><td>540</td><td>0.714</td><td>0.721</td><td>23.225</td><td>Open-vSLAM</td><td>612</td><td>0.713</td><td>0.762</td><td>22.993</td></tr><tr><td>Stella-vSLAM</td><td>363</td><td>0.793</td><td>0.035</td><td>0.064</td><td>Stella-vSLAM</td><td>350</td><td>0.652</td><td>0.033</td><td>0.167</td></tr><tr><td>COLMAP (ERP)</td><td>540</td><td>0.919</td><td>0.018</td><td>0.041</td><td>COLMAP (ERP)</td><td>632</td><td>0.170</td><td>0.013</td><td>0.035</td></tr><tr><td>COLMAP (Pinhole)</td><td>540</td><td>1.027</td><td>0.021</td><td>0.041</td><td>COLMAP (Pinhole)</td><td>632</td><td>0.177</td><td>0.013</td><td>0.035</td></tr><tr><td>Spherical-SfM (ERP)</td><td></td><td>1.093</td><td>0.027</td><td>0.042</td><td>Spherical-SfM (ERP)</td><td>540</td><td>0.308</td><td>0.015</td><td>0.036</td></tr><tr><td>Ours</td><td>540</td><td>0.218</td><td>0.015</td><td>0.054</td><td>Ours</td><td>632</td><td>0.484</td><td>0.027</td><td>0.041</td></tr><tr><td colspan="10">Mean</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scal3R</td><td>4/4</td><td>5.366</td><td>0.124</td><td>1.354</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>360DVO</td><td>4/4</td><td>0.954</td><td>0.026</td><td>0.039</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-vSLAM</td><td>4/4</td><td>0.441</td><td>0.711</td><td>22.184</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stella-vSLAM</td><td>4/4</td><td>0.706</td><td>0.047</td><td>0.147</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COLMAP (ERP)</td><td>4/4</td><td>0.355</td><td>0.020</td><td>0.037</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COLMAP (Pinhole)</td><td>4/4</td><td>0.399</td><td>0.021</td><td>0.037</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spherical-SfM (ERP)</td><td>4/4</td><td>0.491</td><td>0.024</td><td>0.038</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>4/4</td><td>0.270</td><td>0.030</td><td>0.044</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 12 YUTO MMS localization by scene. ATE and RPE-t are in meters; RPE-r is in degrees. n is the number of KF poses or available/expected scenes (Mean).
<table><tr><td>Sequence A</td><td colspan="9">Sequence B</td></tr><tr><td>Method</td><td>n</td><td></td><td>ATE↓ RPE-t↓ RPE-r↓</td><td></td><td>Method</td><td>n</td><td></td><td>ATE↓ RPE-t↓ RPE-r↓</td><td></td></tr><tr><td>Scal3R</td><td>234</td><td>24.591</td><td>0.301</td><td>0.409</td><td>Scal3R</td><td>2794</td><td>307.710</td><td>0.948</td><td>0.381</td></tr><tr><td>360DVO</td><td>234</td><td>0.244</td><td>0.006</td><td>0.010</td><td>360DVO</td><td>2794</td><td>135.769</td><td>0.277</td><td>0.013</td></tr><tr><td>Open-vSLAM</td><td>235</td><td>35.215</td><td>15.562</td><td>56.725</td><td>Open-vSLAM</td><td>2794</td><td>479.963</td><td>37.245</td><td>1.025</td></tr><tr><td>Stella-vSLAM</td><td>235</td><td>0.370</td><td>0.026</td><td>0.060</td><td>Stella-vSLAM</td><td>2794</td><td>252.571</td><td>1.074</td><td>0.223</td></tr><tr><td>COLMAP (ERP)</td><td>234</td><td>0.161</td><td>0.008</td><td>0.014</td><td>COLMAP (ERP)</td><td>2794</td><td>412.900</td><td>2.060</td><td>0.197</td></tr><tr><td>COLMAP (Pinhole)</td><td>234</td><td>0.474</td><td>0.017</td><td>0.017</td><td>COLMAP (Pinhole)</td><td>2794</td><td>75.064</td><td>0.954</td><td>0.036</td></tr><tr><td>Spherical-SfM (ERP)</td><td>234</td><td>0.409</td><td>2.036</td><td>4.139</td><td>Spherical-SfM (ERP)</td><td>2794</td><td>26.071</td><td>4.008</td><td>3.960</td></tr><tr><td>Óurs</td><td>235</td><td>0.082</td><td>0.009</td><td>0.018</td><td>Óurs</td><td>2795</td><td>3.367</td><td>0.041</td><td>0.024</td></tr><tr><td>Sequence C</td><td colspan="9">Sequence D</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>3593</td><td>236.298</td><td>9.695</td><td>3.007</td><td>Scal3R</td><td>1500</td><td>181.028</td><td>0.700</td><td>0.302</td></tr><tr><td>360DVO</td><td>3593</td><td>115.432</td><td>0.366</td><td>0.011</td><td>360DVO</td><td>1500</td><td>21.804</td><td>0.083</td><td>0.011</td></tr><tr><td>Open-vSLAM</td><td>3593</td><td>280.763</td><td>12.037</td><td>6.752</td><td>Open-vSLAM</td><td>1500</td><td>331.233</td><td>101.917</td><td>85.795</td></tr><tr><td>Stella-vSLAM</td><td>3593</td><td>12.848</td><td>0.714</td><td>0.166</td><td>Stella-vSLAM</td><td>1500</td><td>62.904</td><td>0.812</td><td>0.076</td></tr><tr><td>COLMAP (ERP)</td><td>3593</td><td>73.706</td><td>1.017</td><td>0.028</td><td>COLMAP (ERP)</td><td>1500</td><td>11.193</td><td>0.170</td><td>0.024</td></tr><tr><td>COLMAP (Pinhole)</td><td>3593</td><td>124.179</td><td>1.097</td><td>0.039</td><td>COLMAP (Pinhole)</td><td>1500</td><td>253.552</td><td>101.730</td><td>4.090</td></tr><tr><td>Spherical-SfM (ERP)</td><td>3593</td><td>54.137</td><td>3.858</td><td>4.076</td><td>Spherical-SfM (ERP)</td><td>1500</td><td>4.128</td><td>3.346</td><td>3.781</td></tr><tr><td>Ours</td><td>3594</td><td>2.474</td><td>0.068</td><td>0.022</td><td>Ours</td><td>1501</td><td>8.184</td><td>0.090</td><td>0.022</td></tr><tr><td colspan="10">Mean</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scal3R</td><td>4/4</td><td>187.407</td><td>2.911</td><td>1.025</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>360DVO</td><td>4/4</td><td>68.313</td><td>0.183</td><td>0.011</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-vSLAM</td><td>4/4</td><td>281.793</td><td>41.690</td><td>37.574</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stella-vSLAM</td><td>4/4</td><td>82.173</td><td>0.656</td><td>0.131</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COLMAP (ERP)</td><td>4/4</td><td>124.490</td><td>0.814</td><td>0.066</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COLMAP (Pinhole)</td><td>4/4</td><td>113.317</td><td>25.949</td><td>1.046</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spherical-SfM (ERP)</td><td>4/4</td><td>21.186</td><td>3.312</td><td>3.989</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>4/4</td><td>3.527</td><td>0.052</td><td>0.022</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 13 PanoRecon localization by scene. ATE and RPE-t are in meters; RPE-r is in degrees. n is the number of associated poses (scene panels) or available/expected scenes (Mean). All errors are RMSE; lower is better.
<table><tr><td colspan="5">Building 1</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>765</td><td>14.643</td><td>0.281</td><td>0.182</td></tr><tr><td>360DVO</td><td>765</td><td>2.726</td><td>0.240</td><td>0.023</td></tr><tr><td>Open-vSLAM</td><td>765</td><td>3.387</td><td>2.836</td><td>34.833</td></tr><tr><td>Stella-vSLAM</td><td>765</td><td>2.506</td><td>0.411</td><td>0.646</td></tr><tr><td>COLMAP (ERP)</td><td>765</td><td>10.831</td><td>15.002</td><td>8.499</td></tr><tr><td>COLMAP (Pinhole)</td><td>765</td><td>1.618</td><td>0.240</td><td>0.026</td></tr><tr><td>Spherical-SfM (ERP)</td><td>765</td><td>11.907</td><td>15.817</td><td>8.852</td></tr><tr><td>Ours</td><td>764</td><td>1.620</td><td>0.240</td><td>0.024</td></tr></table>

<table><tr><td colspan="5"></td></tr><tr><td>Building 2 Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>419</td><td>3.537</td><td>0.223</td><td>0.128</td></tr><tr><td>360DVO</td><td>419</td><td>4.548</td><td>0.175</td><td>0.047</td></tr><tr><td>Open-vSLAM</td><td>418</td><td>5.764</td><td>3.636</td><td>0.135</td></tr><tr><td>Stella-vSLAM</td><td>418</td><td>5.018</td><td>1.024</td><td>0.121</td></tr><tr><td>COLMAP (ERP)</td><td>419</td><td>5.369</td><td>0.202</td><td>0.048</td></tr><tr><td>COLMAP (Pinhole)</td><td>419</td><td>5.431</td><td>0.205</td><td>0.045</td></tr><tr><td>Spherical-SfM (ERP)</td><td>419</td><td>30.653</td><td>25.932</td><td>0.059</td></tr><tr><td>Ours</td><td>418</td><td>5.302</td><td>0.201</td><td>0.052</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="5">Gym</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>480</td><td>8.943</td><td>0.533</td><td>0.501</td></tr><tr><td>360DVO</td><td>480</td><td>1.876</td><td>0.037</td><td>0.023</td></tr><tr><td>Open-vSLAM</td><td>480</td><td>17.622</td><td>13.582</td><td>56.856</td></tr><tr><td>Stella-vSLAM</td><td>480</td><td>26.683</td><td>5.666</td><td>0.481</td></tr><tr><td>COLMAP (ERP)</td><td>480</td><td>0.629</td><td>0.013</td><td>0.020</td></tr><tr><td>COLMAP (Pinhole)</td><td>480</td><td>0.459</td><td>0.009</td><td>0.018</td></tr><tr><td>Spherical-SfM (ERP)</td><td>480</td><td>34.575</td><td>6.330</td><td>0.786</td></tr><tr><td>Ours</td><td>479</td><td>0.571</td><td>0.047</td><td>0.178</td></tr></table>

HPC
<table><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>507</td><td>1.381</td><td>0.239</td><td>0.533</td></tr><tr><td>360DVO</td><td>507</td><td>1.288</td><td>0.123</td><td>0.080</td></tr><tr><td>Open-vSLAM</td><td>507</td><td>0.673</td><td>10.790</td><td>0.224</td></tr><tr><td>Stella-vSLAM</td><td>507</td><td>2.996</td><td>0.345</td><td>0.226</td></tr><tr><td>COLMAP (ERP)</td><td>507</td><td>0.833</td><td>0.122</td><td>0.079</td></tr><tr><td>COLMAP (Pinhole)</td><td>507</td><td>0.697</td><td>0.122</td><td>0.079</td></tr><tr><td>Spherical-SfM (ERP)</td><td>507</td><td>1.224</td><td>0.148</td><td>0.102</td></tr><tr><td>Ours</td><td>506</td><td>0.976</td><td>0.106</td><td>0.064</td></tr></table>

Road

<table><tr><td colspan="5">Residence</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>844</td><td>4.364</td><td>0.197</td><td>0.414</td></tr><tr><td>360DVO</td><td>844</td><td>1.583</td><td>0.022</td><td>0.030</td></tr><tr><td>Open-vSLAM</td><td>719</td><td>1.384</td><td>4.831</td><td>37.360</td></tr><tr><td>Stella-vSLAM</td><td>495</td><td>1.113</td><td>0.170</td><td>0.194</td></tr><tr><td>COLMAP (ERP)</td><td></td><td></td><td></td><td></td></tr><tr><td>COLMAP (Pinhole)</td><td>844</td><td>0.193</td><td>0.009</td><td>0.023</td></tr><tr><td>Spherical-SfM (ERP)</td><td>844</td><td>0.395</td><td>0.044</td><td>0.043</td></tr><tr><td>Ours</td><td>843</td><td>0.115</td><td>0.046</td><td>0.039</td></tr></table>

<table><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>866</td><td>15.602</td><td>0.167</td><td>0.285</td></tr><tr><td>360DVO</td><td>866</td><td>6.252</td><td>0.099</td><td>0.024</td></tr><tr><td>Open-vSLAM</td><td>314</td><td>21.874</td><td>8.076</td><td>47.028</td></tr><tr><td>Stella-vSLAM</td><td>426</td><td>11.772</td><td>1.320</td><td>1.546</td></tr><tr><td>COLMAP (ERP)</td><td>866</td><td>2.116</td><td>0.081</td><td>0.022</td></tr><tr><td>COLMAP (Pinhole)</td><td>866</td><td>2.224</td><td>0.081</td><td>0.020</td></tr><tr><td>Spherical-SfM (ERP)</td><td>866</td><td>2.006</td><td>0.105</td><td>0.039</td></tr><tr><td>Ours</td><td>865</td><td>2.585</td><td>0.088</td><td>0.200</td></tr></table>

<table><tr><td colspan="5">Residence2</td></tr><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>944</td><td>20.089</td><td>0.336</td><td>0.338</td></tr><tr><td>360DVO</td><td>944</td><td>1.323</td><td>0.033</td><td>0.043</td></tr><tr><td>Open-vSLAM</td><td>944</td><td>1.086</td><td>2.308</td><td>0.260</td></tr><tr><td>Stella-vSLAM</td><td>283</td><td>1.883</td><td>0.364</td><td>0.074</td></tr><tr><td>COLMAP (ERP)</td><td>944</td><td>1.284</td><td>0.031</td><td>0.042</td></tr><tr><td>COLMAP (Pinhole)</td><td>944</td><td>1.528</td><td>0.030</td><td>0.042</td></tr><tr><td>Spherical-SfM (ERP)</td><td>944</td><td>1.691</td><td>0.031</td><td>0.042</td></tr><tr><td>Ours</td><td>943</td><td>0.708</td><td>0.048</td><td>0.045</td></tr></table>

Mean
<table><tr><td>Method</td><td>n</td><td>ATE↓</td><td>RPE-t↓</td><td>RPE-r↓</td></tr><tr><td>Scal3R</td><td>7/7</td><td>9.794</td><td>0.282</td><td>0.340</td></tr><tr><td>360DVO</td><td>7/7</td><td>2.799</td><td>0.104</td><td>0.039</td></tr><tr><td>Open-vSLAM</td><td>7/7</td><td>7.399</td><td>6.580</td><td>25.242</td></tr><tr><td>Stella-vSLAM</td><td>7/7</td><td>7.424</td><td>1.329</td><td>0.470</td></tr><tr><td>COLMAP (ERP)</td><td>6/7</td><td>3.510 (6/7)</td><td>2.575 (6/7)</td><td>1.452 (6/7)</td></tr><tr><td>COLMAP (Pinhole)</td><td>7/7</td><td>1.736</td><td>0.099</td><td>0.036</td></tr><tr><td>Spherical-SfM (ERP)</td><td>7/7</td><td>11.779</td><td>6.915</td><td>1.418</td></tr><tr><td>Ours</td><td>7/7</td><td>1.697</td><td>0.111</td><td>0.086</td></tr></table>

![](images/86d7824e273a567eccc5d840b806d4b6ccd1144fab1440eba09a6f3f30ec7891.jpg)  
Figure 17 Reconstruction Result on 360loc Dataset Hall Scene.

Table 14 360LOC dense reconstruction by scene. $\mathrm { P / R / F }$ in percent at 20 cm. Colors rank methods within each scene and metric.
<table><tr><td rowspan="2">Scene</td><td colspan="3">Stella-vSLAM-Dense</td><td colspan="3">COLMAP (Pinhole)</td><td colspan="3">Ours</td></tr><tr><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td></tr><tr><td>Atrium</td><td>47.83</td><td>76.16</td><td>58.76</td><td>67.84</td><td>46.63</td><td>55.27</td><td>65.51</td><td>66.92</td><td>66.20</td></tr><tr><td>Concourse</td><td>77.73</td><td>59.92</td><td>67.67</td><td>85.15</td><td>54.85</td><td>66.72</td><td>83.88</td><td>52.27</td><td>64.40</td></tr><tr><td>Hall</td><td>71.55</td><td>49.30</td><td>58.38</td><td>77.85</td><td>34.27</td><td>47.59</td><td>85.95</td><td>36.60</td><td>51.34</td></tr><tr><td>Piatrium</td><td>52.58</td><td>47.53</td><td>49.93</td><td>94.22</td><td>43.40</td><td>59.43</td><td>81.57</td><td>55.71</td><td>66.20</td></tr></table>

Table 15 YUTO MMS dense reconstruction by scene. $\mathrm { P / R / F }$ in percent at 20 cm. Colors rank methods within each scene and metric.
<table><tr><td rowspan="2">Scene</td><td colspan="3">Stella-vSLAM-Dense</td><td colspan="3">COLMAP (Pinhole)</td><td colspan="3">Ours</td></tr><tr><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td></tr><tr><td>Sequence A</td><td>46.13</td><td>39.25</td><td>42.41</td><td>88.09</td><td>49.94</td><td>63.74</td><td>88.77</td><td>62.81</td><td>73.56</td></tr><tr><td>Sequence B</td><td>18.74</td><td>4.19</td><td>6.85</td><td>10.13</td><td>0.81</td><td>1.50</td><td>48.63</td><td>32.51</td><td>38.97</td></tr><tr><td>Sequence C</td><td>13.42</td><td>7.10</td><td>9.29</td><td>9.85</td><td>1.36</td><td>2.39</td><td>66.52</td><td>43.20</td><td>52.38</td></tr><tr><td>Sequence D</td><td>16.84</td><td>7.51</td><td>10.39</td><td>8.51</td><td>0.81</td><td>1.48</td><td>38.31</td><td>24.56</td><td>29.93</td></tr></table>

![](images/141f323949dcae399f226ff081c150e24caa9a73df30ba733fd34cf62a105fc4.jpg)  
Figure 18 Reconstruction Result on Ours PanoRecon Benchmark Residence1 Scene.

Table 16 PanoRecon dense reconstruction by scene. P/R/F in percent at 20 cm. Colors rank methods within each scene and metric.
<table><tr><td rowspan="2">Scene</td><td colspan="3">Stella-vSLAM-Dense</td><td colspan="3">COLMAP (Pinhole)</td><td colspan="3">Ours</td></tr><tr><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td><td>P↑</td><td>R↑</td><td>F↑</td></tr><tr><td>Building 1</td><td>22.72</td><td>37.68</td><td>28.34</td><td>86.74</td><td>71.51</td><td>78.39</td><td>70.69</td><td>68.87</td><td>69.77</td></tr><tr><td>Building 2</td><td>49.98</td><td>55.64</td><td>52.66</td><td>67.20</td><td>62.37</td><td>64.69</td><td>64.06</td><td>58.56</td><td>61.19</td></tr><tr><td>Gym</td><td>34.40</td><td>10.87</td><td>16.52</td><td>58.60</td><td>41.62</td><td>48.67</td><td>66.03</td><td>53.86</td><td>59.33</td></tr><tr><td>HPC</td><td>51.50</td><td>30.84</td><td>38.58</td><td>84.42</td><td>55.84</td><td>67.21</td><td>65.74</td><td>66.36</td><td>66.05</td></tr><tr><td>Residence</td><td>50.17</td><td>52.52</td><td>51.32</td><td>89.47</td><td>56.71</td><td>69.42</td><td>76.00</td><td>74.44</td><td>75.22</td></tr><tr><td>Road</td><td>10.89</td><td>10.20</td><td>10.54</td><td>48.96</td><td>34.74</td><td>40.65</td><td>36.14</td><td>40.14</td><td>38.04</td></tr><tr><td>Residence2</td><td>36.58</td><td>19.95</td><td>25.82</td><td>63.86</td><td>40.04</td><td>49.22</td><td>70.89</td><td>74.30</td><td>72.56</td></tr></table>

![](images/a4c947667d61596d722b256c0c09a5b18439a328a5890976e1aed3aff6f19a37.jpg)  
Figure 19 Reconstruction Result on Ours PanoRecon Benchmark Road Scene.

![](images/74286495383f550448086708237c1eafde4a1cb9fc74c55f5b49e1d43ecccd81.jpg)  
Figure 20 Reconstruction Result on Ours PanoRecon Benchmark Residence2 Scene.

![](images/a79492626889c3021675b3293f13952188820fa5043e2b12f1399f735650f13a.jpg)  
Figure 21 Reconstruction Result on YUTO MMS Benchmark Sequence B Scene.

![](images/8d059de9d4965d67ba1e8f02bfd6f9f9486c9e1206ae9fc3b8bd771e6c87a338.jpg)  
Figure 22 Reconstruction Result on YUTO MMS Benchmark Sequence C Scene.