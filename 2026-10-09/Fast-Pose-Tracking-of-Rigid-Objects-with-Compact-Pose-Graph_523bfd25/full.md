# Fast Pose Tracking of Rigid Objects with Compact Pose Graph Optimization

Xiaojie Zhang<sup>1</sup> Tom Fischer<sup>1</sup> Viktor Larsson<sup>2</sup> Eddy Ilg<sup>1†</sup>

<sup>1</sup>University of Technology Nuremberg <sup>2</sup>Lund University

## Abstract

Tracking a novel object’s 6D pose over long horizons currently requires either expensive onboarding or a reconstruction maintained throughout the sequence. This makes current trackers impracticalfor robotic manipulation and augmented reality, which need trackers that are ready to use and run in real time. We show that a lightweight tracking module can be applied on top of a wide range of correspondence estimation methods to keep drifts bounded while maintaining fast runtime. Our key idea is to avoid pointbased optimization in the pose graph and operate only on relative pose constraints, which we weight by a derived uncertainty from the geometric alignment. This makes optimization independent of the number of correspondences while avoiding the direct inclusion of noisy point measurements, leading tofast and robust long-term tracking. Across four real-world benchmarks, our approach achieves tracking accuracy comparable to reconstruction-based trackers with a fraction of the optimization cost. Overall, these results suggest that a compact and reliable pose graph optimization can provide long-horizon consistency at substantially lower computational cost.

## 1. Introduction

Tracking an object’s 6D pose over time is important for understanding dynamic environments, with applications in robotic manipulation [31, 44], augmented reality [1], and human–object interaction [14]. Two requirements make this problem challenging in practice: the tracking must run online at interactive rates to support downstream planning, and it must remain accurate and temporally consistent over long horizons despite occlusions and fast motions.

Existing approaches satisfy these two requirements one at a time. Long-horizon accuracy is typically obtained by anchoring poses to a 3D representation, either built online during tracking [9, 18, 42] or onboarded beforehand from CAD models or posed reference frames [43]. While both approaches can be accurate, the first is too slow for real-time use and the second must be repeated for every new object. Interactive rates without onboarding can be achieved by relative pose estimation methods [20, 23, 24], which estimate pose from a single reference observation. However, these operate only on pairs of frames, so small errors in relative motion compound over time and cause drift.

We show that both requirements can be met by constraining pairwise measurements against each other in a pose graph without maintaining a 3D representation. While pose graph optimization (PGO) is widely used in vision and robotics, existing object trackers typically build their graph constraints directly from correspondences or dense geometric observations. We instead ask: once relative motion has been estimated from a set of correspondences, is it necessary to retain those points in the global optimization?

Our key idea is to compress each set of correspondences into a relative pose measurement and perform global optimization only over poses. This has two advantages. When correspondences are reliable, many inliers may be available and the relative pose is already well constrained. Retaining all of those points in the global objective only increases the optimization cost, whereas our update cost is independent of the number of inliers. When correspondences are poor, directly placing noisy points in the global objective also exposes the optimization to their errors. By operating only on relative pose constraints, we avoid both costs.

This formulation shifts the main question from how many correspondences are retained to how much each relative pose measurement should be trusted. We therefore derive a pose uncertainty from the geometric alignment that produced each relative transform and use it to weight the corresponding graph constraint. Poorly constrained measurements contribute less, while reliable measurements are enforced more strongly. The result is a lightweight tracking module that can be paired with a wide range of correspondence estimation methods (Fig. 1) and, in our experiments, achieves accuracy comparable to reconstruction- and CADbased methods at substantially lower runtime.

In summary, our contributions are as follows:

• We formulate 6D object tracking as a compact pose graph optimization over relative pose constraints. By compressing dense correspondences into an SE(3) measurement, the optimization cost is independent of the number of matched correspondences, enabling substantially faster updates while retaining global trajectory consistency.

• We derive per-edge pose uncertainty from the geometric alignment and use it for weighting pose graph constraints. This allows poorly constrained relative poses to contribute less to the trajectory estimation and improves robustness when the correspondence quality degrades.

• The proposed tracker can be paired with different correspondence or relative-pose estimators without retraining or per-object onboarding. Across four real-world RGB-D benchmarks, our method achieves accuracy comparable to state-of-the-art tracking-by-reconstruction approaches across four benchmarks with a fraction of the optimization cost.

## 2. Related Work

## 2.1. Pose Estimation

Estimating camera poses is central for scene- and objectlevel understanding. A common approach is to estimate correspondences through feature matching and then solve camera poses through classical geometry, such as the eight-point algorithm [16] or Perspective-n-Point (PnP) [21]. Scenelevel pipelines [10, 35] typically focus on relative transformations between RGB images and rely on (dense) 2D– 2D correspondences. To improve robustness across diverse scenes and large viewpoint changes, recent approaches adopt deep learning-based matchers [11, 12, 22, 34, 36] trained on large-scale data, which substantially outperform hand-crafted features [27, 33].

In object-centric settings, pose estimation methods [4, 13, 26, 37, 43] often learn 3D object representations to model geometry and appearance. Relative pose estimation methods for novel objects [20, 23, 24] instead predict the relative transformation between two frames without requiring instance-specific object knowledge (e.g. CAD models, multiple reference frames). These methods can generalize to unseen objects and can handle moderate viewpoint changes. In this work, we propose a general tracking module which can be paired with both 2D feature matchers and relative object pose estimation methods and improves temporal consistency by fusing pairwise measurements over time.

## 2.2. Pose Graph Optimization

Pose graph optimization (PGO) [2, 3] estimates absolute poses from pairwise constraints by enforcing consistency over a graph. It has been widely used across structure-frommotion [30, 46], SLAM [28], and robotics [8]. Importantly, different systems instantiate PGO with different residual models, where some optimize alignment-based residuals defined directly on correspondences [6], while others represent pairwise relations as relative-pose factors and optimize pose-consistency residuals in SE(3) poses [7, 30, 46]. While both types are common in scene-level pipelines, pose-consistency terms have received comparatively little attention in object-centric RGB-D tracking, where prior work more commonly relies on alignment-based objectives. Dellaert et al. introduced factor-graph formulations and the Bayes tree data structure for incremental smoothing and mapping (iSAM2) [19], which supports low-latency updates without periodic batch steps through fluid relinearization and incremental variable reordering. In this work, we integrate iSAM2 to efficiently update our 6D object pose graph online as new frames arrive.

## 2.3. Object 6D Pose Tracking

BundleTrack [40] is an early framework for novel-object 6D pose tracking that does not require CAD models. It combines feature matching with a memory-augmented pose graph that is optimized through an alignment loss on unprojected point clouds (similar to [6]). BundleSDF [42] extends this line of work by integrating an online object reconstruction into the tracking loop, which can improve long-horizon stability but further increases computational cost. To reduce optimization overhead, 6DOPE-GS [18] proposes a 2D Gaussian-Splatting-based representation [17] for faster online reconstruction and reports a speed–accuracy trade-off.

Overall, alignment-based objectives in prior objectcentric pose-graph trackers [18, 40, 42] can yield accurate pose estimates. However, they couple the dense correspondence inside the global optimization, making optimization cost scale with the number of points or image resolutions. Additionally, this makes them more prone to error on data with noisy depth and degraded correspondence quality. In contrast, we use correspondences only to generate relative pose measurements and then optimize a compact pose graph over relative pose constraints. This design preserves modularity with respect to the correspondence estimator while enabling efficient and robust online updates.

## 3. Methodology

We track the 6D pose of an object across an RGB-D video by maintaining a compact SE(3) pose graph that is extended and optimized incrementally (Fig. 2) over relative pose constraints. Every new frame contributes an odometry constraint to its predecessor and up to K loop closure constraints to past keyframes, where each is weighted by an uncertainty derived from the conditioning of the geometric alignment problem. The formulation is agnostic to how correspondences are obtained and can turn any feature matcher into a 6D object pose tracker.

![](images/2d6df03366c21158fb0b312358254855514c73cfd3ebd54ff2c7e66c5af1853a.jpg)  
Figure 1. Left: We propose a lightweight generalizable 6D pose tracking framework for novel objects that can be flexibly paired with any correspondence estimation method. Right: We show that our approach achieves the best pose accuracy and runtime efficiency tradeoff on the YCBInEOAT dataset. Gray: reported in the original paper on different hardware, see Tab. 4.

## 3.1. Problem Formulation

Given an RGB-D video $\{ ( I _ { i } , D _ { i } ) \} _ { i = 0 } ^ { N }$ with images, depth maps, and masks of the target object $\{ M _ { i } \} _ { i = 0 } ^ { N } ,$ , our task is to track the object’s 6D pose $T _ { i } \in S E ( 3 )$ , where $T _ { i }$ denotes the pose of frame $I _ { i }$ relative to $I _ { 0 }$ . We assume that the whole tracking process is online such that at time step i a tracker only has access to the past frames $\{ 0 , \ldots , i \}$

## 3.2. Object Pose Graph Optimization

We frame object pose tracking as optimization over a pose graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , in which each node $V _ { i } \in \mathcal V$ is the pose $T _ { i }$ of frame $I _ { i }$ and each edge $E _ { i j } ~ \in ~ \mathcal { E }$ carries a relative measurement $T _ { i j }$ between the two frames. The residual $L _ { e } \in \mathbb { R } ^ { 6 }$ of edge $E _ { i j }$ is

$$
L _ { e } ( T _ { i } , T _ { j } ) = \mathrm { L o g } \bigl ( T _ { i j } ^ { - 1 } ( T _ { i } ^ { - 1 } T _ { j } ) \bigr ) ,\tag{1}
$$

where Log $: \ S E ( 3 ) \ \to \ \mathbb { R } ^ { 6 }$ is the logarithm map to the tangent space. We anchor the first node at $T _ { 0 } = \mathbb { I }$ to fix the gauge freedom, and then solve

$$
\{ T _ { i } ^ { * } \} _ { i = 0 } ^ { N } = \underset { \{ T _ { i } \} _ { i = 0 } ^ { N } } { \arg \operatorname* { m i n } } ~ \sum _ { ( i , j ) \in \mathcal { E } } \rho \Big ( \| L _ { e } ( T _ { i } , T _ { j } ) \| _ { \Sigma _ { i j } } \Big ) ,\tag{2}
$$

where $\rho ( \cdot )$ is a robust loss, $\| e \| _ { \Sigma } ^ { 2 } = e ^ { \top } \Sigma ^ { - 1 } e$ denotes the Mahalanobis norm, and $\Sigma _ { i j }$ is the uncertainty of the measurement $T _ { i j }$

## 3.3. Frame Selection

Choosing reliable keyframes is crucial to avoid drift that would accumulate when chaining odometries alone without long-range constraints through loop closure edges. Deciding which past frames to connect to a new frame therefore determines how much of edges are informative.

When a new frame $I _ { i }$ arrives, we select up to K candidates for loop closures from a memory pool containing past keyframes. To avoid estimating $T _ { i - 1 , i }$ before selection, we extrapolate a motion prior $T _ { i } ^ { \prime }$ under a constant-velocity model,

$$
T _ { i } ^ { \prime } = T _ { i - 1 } ( T _ { i - 2 } ^ { - 1 } T _ { i - 1 } ) ,\tag{3}
$$

such that we only require a single forward pass for the correspondences, and $T _ { i } ^ { \prime }$ is only used to guide frame selection. We then retain the frames in whose rotation lies within radius $\tau _ { k }$ of the rotation component of $T _ { i } ^ { \prime }$ :

$$
d _ { g e o } ( R _ { i } ^ { ' } , R _ { j } ) = \operatorname { a r c c o s } \left( \frac { \operatorname { t r } ( R _ { j } ^ { \top } R _ { i } ^ { ' } ) - 1 } { 2 } \right) \leq \tau _ { k } .\tag{4}
$$

We select candidates from similar rotations, since we found that similar poses generally have more visual overlap, yield more matches and more correct relative pose estimation. From these candidates we sample a final set of at most K frames via farthest-frame sampling over timesteps, analogous to farthest-point sampling for point clouds. This results in edges of varying lengths that maximize the information gain. Shorter edges retain larger overlap and are easier to match, whereas longer edges anchor the current frame to distinct parts of the trajectory rather than reinforcing what odometry already constrains.

## 3.4. Relative Pose Estimation

Pose from Correspondences. We estimate the relative poses between $I _ { i }$ and the keyframes  from any source of correspondences, e.g. 2D-2D [11, 12, 36], 3D-3D [23, 24]. For 2D-2D matchers, we backproject the correspondences with depth. In both cases we obtain a set of correspondences

![](images/218792d960f84b089ccd9ae4ff436b7c906166ef76acda092479fa6ee95f5f26.jpg)  
Figure 2. High-level method overview. We initialize a SE(3) pose graph which will be incrementally updated when a new frame arrives. In the first stage of tracking (1), we select frames from the past which we want to connect to the current frame via edges in the pose graph. Next, for each pair of frames corresponding to an edge, we estimate their correspondences $( 2 . a ) .$ , and use RANSAC Umeyama to solve 6D relative poses (2.b) and estimate their uncertainty(2.c). Finally, we insert (3.a) and optimize (3.b) the new node and its edges to the pose graph using iSAM2 , and update the memory pool (3.c).

$$
\mathcal { C } _ { i k } ^ { \mathrm { f u l l } } = \{ ( \mathbf { a } _ { m } , \mathbf { b } _ { m } , w _ { m } ) \} _ { m = 1 } ^ { M } ,\tag{5}
$$

where $\mathbf { a } _ { m } , \mathbf { b } _ { m } \in \mathbb { R } ^ { 3 }$ are corresponding 3D points in $I _ { k }$ and $I _ { i }$ , and $w _ { m }$ are confidence weights output by the feature matcher. We run RANSAC to identify the inlier subset $\mathcal { C } _ { i k } \subseteq \mathcal { C } _ { i k } ^ { \mathrm { f u l l } }$ , and solve for $T _ { i k }$ via weighted Umeyama over inliers,

$$
\left( R _ { i k } , t _ { i k } \right) = \underset { R , t } { \arg \operatorname* { m i n } } \sum _ { \left( \mathbf { a } , \mathbf { b } , w \right) \in \mathcal { C } _ { i k } } w \left\| R \mathbf { b } + t - \mathbf { a } \right\| _ { 2 } ^ { 2 } ,\tag{6}
$$

which admits a closed-form solution via weighted SVD [39].

Uncertainty Propagation. Common failure modes in object pose estimation, such as thin objects or heavy occlusion lead to poor conditioning of the alignment due to noisy or few correspondences. Since this is a property of the alignment, we can measure it and downweight corresponding edges in the pose graph. We define a measurement noise $\eta _ { i k } \in \mathbb { R } ^ { 6 }$ of the relative pose $T _ { i k }$ in the tangent space:

$$
T _ { i } = T _ { k } T _ { i k } \mathrm { E x p } ( \eta ) , \eta \sim \mathcal { N } ( \mathbf { 0 } _ { 6 \times 1 } , \boldsymbol { \Sigma } _ { i k } ) .\tag{7}
$$

We model the uncertainty as the covariance $\Sigma _ { i k }$ of the pose estimated by Eq. (6) which we approximate with the inverse Hessian, following [38]:

$$
H _ { i k } \approx \sum _ { m = 1 } ^ { \left| \mathcal { C } _ { i k } \right| } J _ { m } ^ { \top } J _ { m } , \quad \quad \Sigma _ { i k } \approx \sigma _ { i k } ^ { 2 } H _ { i k } ^ { - 1 } ,\tag{8}
$$

where J is the Jacobian of the whitened residual $\widetilde { r } _ { m } ~ =$ $\sqrt { w } r$ and $r _ { m } = R _ { i k } \mathbf { b } _ { m } + t _ { i k } - \mathbf { a } _ { m } , \sigma _ { i k } ^ { 2 }$ is the estimated residual variance to calibrate the scale of Σ. Without $\sigma _ { i k }$ the absolute scale of $\Sigma _ { i k }$ is set by an arbitrary scale of the confidences of the matchers. We parameterize the pose increment in the tangent space by $\delta \xi = ( \delta \theta , \delta t ) \in \mathbb { R } ^ { 6 }$ and use a right-multiplicative rotation perturbation $R _ { i k } \mathrm { E x p } ( [ \delta \theta ] _ { \times } )$ Linearizing $r _ { m }$ yields

$$
\tilde { r } _ { m } ( \delta \xi ) \approx \tilde { r } _ { m } ( 0 ) + J _ { m } \delta \xi , J _ { m } = \sqrt { w _ { m } } \left[ - R _ { i k } [ \mathbf { b } _ { m } ] _ { \times } \right.\tag{I ,}
$$

(9)

where $[ \cdot ] _ { \times }$ denotes the skew-symmetric matrix, I is a $3 \times 3$ identity matrix. Stacking $J _ { m }$ over correspondences leads to the full Jacobian defined in Eq. (8). The full derivation is provided in the supplementary material.

## 3.5. Online Pose Graph Optimization

iSAM2 Update. We insert the new node $V _ { i } ,$ its odometry edge $E _ { i - 1 , i }$ , and loop closure constraints to the pose graph ${ \mathcal { G } } .$ . We update the pose graph using iSAM2 [19]. Internally, iSAM2 maintains a sparse factorization of the linearized system using a Bayes-tree representation and performs selective relinearization, so that only variables affected by new constraints are revisited. This yields low-latency updates suitable for online 6D tracking. For our online setting, previously estimated poses will also be optimized but only for optimization of future frames.

Memory Pool Update. Finally, we update the memory pool  from which loop closure candidates are drawn. Its purpose is to keep the candidate set viewpoint-diverse, so that the farthest-frame sampling in Sec. 3.3 operates over distinct views rather than near-duplicates. The pool is initialized with the first frame and extended by the current frame $I _ { i }$ if its geodesic distance to every frame in exceeds a threshold $\tau _ { m }$ , i.e., min $_ { j \in \mathcal { M } } d _ { g e o } ( R _ { i } , R _ { j } ) > \tau _ { m }$

## 4. Evaluation

In this section, our experiments address four main questions: (1) How does our model compare to state-of-the-art methods in both pose tracking accuracy and runtime efficiency? (2) How sensitive is our method to the quality of correspondences? (3) Does the resulting tracker remain stable over long sequences? (4) How important are the different components of our model?

![](images/9ec27143af069b2e5f7669d524e3d91fecf9133d82ba9780382cad271b3a6381.jpg)  
Figure 3. Detailed method overview. For a new frame $I _ { i } ,$ we first select K frames from previous frames to build candidate loop closure constraints based on the geodesic distance and farthest frame sampling. Then the current frame $I _ { i }$ and selected K candidate frames will be passed to the correspondence estimation method and RANSAC Umeyama to obtain relative poses $T _ { i k }$ and uncertainty $\Sigma _ { i k }$ . Furthermore, we insert the new node V with its odometry and loop closure edges to the uncertainty-aware graph and update $T _ { i }$ using iSAM2. Last, we update the memory pool M based on viewpoint diversity.

## 4.1. Setup

Metrics. We follow the evaluation protocol of [40, 42, 43] and evaluate 6D pose tracking quality using the area under the curve (AUC) percentage for ADD-S and ADD [45] (0 0.1m) using ground-truth object models. These metrics measure the mean distance between corresponding points on the predicted and ground-truth object models, providing a robust measure of geometric alignment accuracy. Inference speed is reported in frames per second (fps).

Benchmarks. We evaluate our method on four realworld object pose tracking benchmarks, HOI4D [25], YCBInEOAT [41], HO3D [15] and YCB Video [45]. HOI4D [25] is a large-scale egocentric hand-object interaction dataset, containing 973 videos for 49 rigid objects from 5 categories (toy car, kettle, knife, bowl, chair). The average video length is 300 frames. YCBInEOAT [41] is an egocentric robot manipulation dataset that consists of 9 scenes and 5 objects, with 823 frames on average. HO3D [15] is a hand-object interaction dataset that contains 13 scenes and 4 objects, with average 1547 frames. YCB-Video [45] contains 21 objects in 12 cluttered scenes with occlusions and symmetric objects. The average length is 1718 frames.

Baselines. We compare our method with other novelobject 6D pose tracking methods and divide them into four categories: 1) Tracking-by-Reconstruction (TbR): BundleSDF [42] and 6DOPE-GS [18]; 2) Relative object Pose Estimation (RPE): UNOPose [24] and One2Any [23]; 3) Pose Graph Optimization-based (PGO): Bundle-Track [40]; 4) CAD-based: FoundationPose [43]. We note that 4) requires either known object models or reconstruction from posed reference views and is therefore not directly comparable.

## 4.2. Implementation

For our main method, we use RoMa v2 [12] (turbo) for feature matching. Unless otherwise specified, we use $K = 5$ as the default number. For online graph optimization, we use iSAM2 from the GTSAM [7] library. We evaluate causal poses output at the time of arrival w.o. iSAM2 smoothing. To evaluate against ground-truth pose annotations, we align the predicted trajectory by applying the relative transform between our first-frame prediction and the groundtruth pose of the first frame. For 6DOPE-GS [18], we report their results from the original paper due to its code currently not being available. All experiments and baselines were conducted on an NVIDIA H100 GPU. We use identical masks for all methods.

## 4.3. Comparison to State-of-the-Art

We compare our method to state-of-the-art pose tracking methods on four benchmarks: HOI4D, YCBInEOAT, HO3D, and YCB Video in Tab. 1. For a fair comparison, we re-run BundleTrack with RoMa v2 as its feature matcher (BundleTrack (R.)), matching our own setting; RPE baselines estimate each frame’s pose relative to the first frame.

Compared to FoundationPose, using CAD models, our method leads on three of four benchmarks on ADD metric. HOI4D is the hardest of our four benchmarks, where our method outperforms FoundationPose by a large margin. We hypothesize that this difference arises from how the two methods propagate information over time. While FoundationPose’s tracking mode refines each pose from a small set of hypotheses conditioned only on the previous frame, our loop-closure edges reconnect to earlier frames and propagate reliable information using uncertainty.

<table><tr><td rowspan="2">Cat.</td><td rowspan="2">Method</td><td colspan="2">HOI4D (300 frames)</td><td colspan="3">YCBInEOAT (823 frames)</td><td colspan="3">HO3D (1547 frames)</td><td colspan="3">YCB Video (1718 frames)</td><td rowspan="2">Avg. Speed fps</td></tr><tr><td>ADD-S (%)</td><td>ADD (%)</td><td>ADD-S (%)</td><td>ADD (%)</td><td>AR (%)</td><td>ADD-S (%)</td><td>ADD (%)</td><td>AR (%)</td><td>ADD-S (%)</td><td>ADD (%)</td><td>AR (%)</td></tr><tr><td rowspan="2">CAD</td><td>FoundationPose (CAD)</td><td>84.4</td><td>66.5</td><td>96.4</td><td>93.1</td><td>90.5</td><td>97.0</td><td>88.9</td><td>89.3</td><td>98.0</td><td>96.0</td><td>92.8</td><td>22.9</td></tr><tr><td>UNOPose</td><td>92.2</td><td>79.4</td><td>91.0</td><td>67.7</td><td>65.7</td><td>92.9</td><td>51.6</td><td>31.7</td><td>96.4</td><td>84.7</td><td>80.6</td><td>10.0</td></tr><tr><td rowspan="2">RPE</td><td>One2Any</td><td>79.4</td><td>64.0</td><td>66.0</td><td>31.5</td><td>31.1</td><td>75.5</td><td>37.1</td><td>29.4</td><td>92.9</td><td>88.7</td><td>75.0</td><td>30.0</td></tr><tr><td>BundleSDF</td><td>90.8</td><td>79.0</td><td>93.8</td><td>87.0</td><td>82.2</td><td>96.5</td><td>92.6</td><td>87.9</td><td>97.6</td><td>93.8</td><td>90.0</td><td>1.8</td></tr><tr><td rowspan="3">TbR</td><td>6DOPE-GS†</td><td>=</td><td>=</td><td>93.8</td><td>87.8</td><td></td><td>95.1</td><td>84.3</td><td>-</td><td>-</td><td>=</td><td>=</td><td>4.2</td></tr><tr><td>BundleTrack†</td><td>一</td><td></td><td>92.5</td><td>87.3</td><td>79.0</td><td>92.4</td><td>66.0</td><td></td><td></td><td></td><td></td><td>10.0</td></tr><tr><td>BundleTrack (R.)</td><td>91.8</td><td>79.8</td><td>93.8</td><td>88.0</td><td>82.7</td><td>96.5</td><td>92.6</td><td>88.0</td><td>97.5</td><td>94.9</td><td>91.8</td><td>1.4</td></tr><tr><td>PGO</td><td>Ours</td><td>92.2</td><td>80.3</td><td>94.4</td><td>88.8</td><td>83.4</td><td>96.4</td><td>92.2</td><td>86.9</td><td>98.2</td><td>96.5</td><td>94.3</td><td>10.0</td></tr></table>

Table 1. Pose tracking accuracy and runtime across four benchmarks. We report ADD-S and ADD (AUC %, 0–0.1 m), and BOP Average Recall (AR) on HOI4D, YCBInEOAT, HO3D, and YCB-Video (average sequence length shown under each dataset name), alongside average inference speed (fps) across all four datasets. AR is omitted for HOI4D, because of no BOP-format ground truth available. Methods are grouped by category: CAD: requires CAD models; TbR: tracking-by-reconstruction; RPE: relative pose estimation; PGO: pose-graph optimization-based tracking, the category our method belongs to alongside BundleTrack. Best result among non-CAD methods per column in bold, second-best underlined (tied values are marked identically). Among non-CAD methods, our method leads on YCBInEOAT, YCB Video, and HOI4D, and is comparable to BundleSDF on HO3D, while running substantially faster than other tracking-by-reconstruction and pose-graph-based baselines. BundleTrack (R.): BundleTrack re-run with RoMa v2 as the feature matcher, using the official codebase †: numbers taken from the original paper runtime reported on a different GPU than ours (Tab. 4)

Among non-CAD methods, we reach the best or nearbest accuracy on three of four benchmarks. Only on HO3D, BundleSDF and BundleTrack (R.) lead, by under one point. Against BundleTrack (R.), we retain global consistency while replacing its dense alignment objective with a lightweight, uncertainty-aware pose-consistency objective, enabling substantially faster updates.

HOI4D. Tab. 2 analyzes HOI4D by looking into the performance of all rigid object categories: toy car (<sup>Þ</sup>), kettle ( ), knife ( ), chair (ð), and bowl ( ). UNOPose performs well since its training data is biased towards the large viewpoint changes present in the fast motion of HOI4D. The ADD(-S) threshold (0.1 m) is fixed across objects, which can compress rotation differences for small objects such as the toy car and knife. We therefore provide the rotation only metric in supplementary material.

Runtime Analysis. Tab. 1 reports average runtime (fps) across all four benchmarks on the same H100 GPU (perdataset breakdowns in supplementary material). Our method runs at 10.0 fps on average, 5.5 to 7 times faster than the TbR (BundleSDF) and PGO baselines (Bundle-Track (R.)) benchmarked on the same GPU. Native Bundle-Track is reported at the same rate as our method, but uses a different GPU and matcher and is therefore not directly comparable.

Tab. 3 breaks down per-frame runtime by stage on YCB Video. Matching dominates the cost ( 0.05s per frame to be matched to K = 5 selected keyframes), while our compact pose graph makes global consistency nearly free by comparison: frame selection, PGO, and memory pool update together add only 2.7 ms, against 75 ms for Bundle-Track’s dense-alignment PGO step under the same matcher and hardware (see supplementary material). For real-world deployment without object masks, an additional per-frame segmentation stage can be pipelined into a separate CUDA stream, since it does not depend on our tracking backend. For reference, XMem [5] runs at 100 fps, SAM2 [32] runs at 50 fps for a 640 480 video stream.

<table><tr><td rowspan="2">Cat.</td><td colspan="5">ADD (%)</td><td colspan="5">ADD-S (%)</td></tr><tr><td>Ours</td><td>BS</td><td>BT</td><td>UP</td><td>FP</td><td>Ours</td><td>BS</td><td>BT</td><td>UP</td><td>FP</td></tr><tr><td>の</td><td>86.3</td><td>84.4</td><td>84.9</td><td>83.2</td><td>60.0</td><td>93.5</td><td>92.9</td><td>93.3</td><td>93.2</td><td>72.8</td></tr><tr><td rowspan="3">1 1 </td><td>83.3</td><td>83.3</td><td>83.5</td><td>80.9</td><td>78.3</td><td>92.5</td><td>92.5</td><td>92.6</td><td>92.2</td><td>89.7</td></tr><tr><td>82.4</td><td>78.2</td><td>81.2</td><td>82.5</td><td>75.1</td><td>91.7</td><td>87.5</td><td>90.9</td><td>92.2</td><td>84.8</td></tr><tr><td>75.3</td><td>74.8</td><td>75.3</td><td>76.2</td><td>76.3</td><td>88.9</td><td>87.6</td><td>88.1</td><td>89.2</td><td>89.3</td></tr><tr><td>マ</td><td>74.3</td><td>74.1</td><td>74.2</td><td>73.9</td><td>42.6</td><td>94.4</td><td>93.5</td><td>93.9</td><td>94.1</td><td>85.4</td></tr><tr><td>Avg.</td><td>80.3</td><td>79.0</td><td>79.8</td><td>79.4</td><td>66.5</td><td>92.2</td><td>90.8</td><td>91.8</td><td>92.2</td><td>84.4</td></tr></table>

Table 2. Per-category comparison on five HOI4D rigid object categories. We report ADD-AUC and ADD-S-AUC (%, 0– 0.1 m). BS: BundleSDF, BT: BundleTrack, UP: UNOPose, FP: FoundationPose.

<table><tr><td>Matcher</td><td>Matching</td><td>Solve+Unc.</td><td>Frame Sel.</td><td>PGO Pool Update</td><td></td><td>Ours (fps)</td></tr><tr><td>RoMa v2</td><td>0.05</td><td>0.04</td><td>5e-4</td><td>2e-3</td><td>2e-4</td><td>10.8</td></tr></table>

Table 3. Per-stage latency on YCB Video. Results reported in seconds per frame and averaged over the dataset. The final column reports total tracking throughput in frames per second.
<table><tr><td>Method</td><td>GPU</td><td>fps</td></tr><tr><td>BundleTrack (LF-Net) [40]</td><td>RTX 2080 Ti</td><td>10.0</td></tr><tr><td>6DOPE-GS [18]</td><td>RTX 4090</td><td>4.2</td></tr><tr><td>Ours</td><td>RTX 4090</td><td>9.0</td></tr></table>

Table 4. Runtime on different GPUs. Numbers are reported from the original papers.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Matcher</td><td colspan="2">HOI4D</td><td colspan="2">YCBInEOAT</td><td colspan="2">HO3D</td><td colspan="2">YCB Video</td></tr><tr><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td></tr><tr><td rowspan="2">BundleTrack Ours</td><td rowspan="2">RoMa v2</td><td>91.8</td><td>80.2</td><td>93.8</td><td>88.0</td><td>96.5</td><td>92.6</td><td>97.5</td><td>94.9</td></tr><tr><td>92.2</td><td>80.3</td><td>94.4</td><td>88.8</td><td>96.4</td><td>92.2</td><td>98.2</td><td>96.5</td></tr><tr><td rowspan="2">BundleTrack Ours</td><td rowspan="2">LoFTR</td><td>92.0</td><td>80.1</td><td>92.5</td><td>84.9</td><td>94.1</td><td>79.3</td><td>97.3</td><td>93.9</td></tr><tr><td>92.2</td><td>80.6</td><td>93.0</td><td>87.8</td><td>94.8</td><td>85.7</td><td>97.4</td><td>94.2</td></tr></table>

Table 5. Comparison to dense alignment-PGO method Bundle-Track. We show the direct comparison between dense-alignment and pose consistency optimization under the same correspondence estimator. The pose-level optimization preserves accuracy while dramatically reducing optimization cost.

<table><tr><td rowspan="3">Uncertainty</td><td colspan="2">HOI4D</td><td colspan="2">YCBInEOAT</td><td colspan="2">HO3D</td><td colspan="2">YCB Video</td></tr><tr><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td><td>ADD-S(%)</td><td>ADD(%)</td></tr><tr><td>Identity (RoMa v2)</td><td>92.11</td><td>80.33</td><td>93.69</td><td>87.60</td><td>96.14</td><td>90.77</td><td>96.82</td><td>94.19</td></tr><tr><td>Trace</td><td>-0.28</td><td>-0.23</td><td>-1.15</td><td>-3.07</td><td>+0.39</td><td>+1.80</td><td>+1.34</td><td>+2.24</td></tr><tr><td>Covariance</td><td>+0.09</td><td>+0.00</td><td>+0.70</td><td>+1.20</td><td>+0.26</td><td>+1.43</td><td>+1.34</td><td>+2.31</td></tr><tr><td>Oracle</td><td>+0.40</td><td>+1.56</td><td>+1.08</td><td>+2.40</td><td>+0.80</td><td>+2.95</td><td>+0.17</td><td>+1.20</td></tr><tr><td>Identity (LoFTR)</td><td>91.40</td><td>79.35</td><td>92.38</td><td>83.52</td><td>94.07</td><td>83.34</td><td>96.03</td><td>92.30</td></tr><tr><td>Trace</td><td>+0.33</td><td>+0.68</td><td>+1.21</td><td>+3.40</td><td>+1.42</td><td>+3.24</td><td>+1.92</td><td>+2.92</td></tr><tr><td>Covariance</td><td>+0.80</td><td>+1.25</td><td>+0.62</td><td>+4.28</td><td>+0.78</td><td>+2.36</td><td>+1.37</td><td>+1.90</td></tr><tr><td>Oracle</td><td>+1.12</td><td>+2.69</td><td>+1.77</td><td>+5.00</td><td>+2.34</td><td>+7.76</td><td>+0.11</td><td>+2.69</td></tr></table>

Table 6. Ablation on uncertainty structures. We compare four levels of uncertainty information used in pose graph optimization: Identity (no per-edge uncertainty), Trace (isotropic, using only the scalar magnitude of our estimated covariance), Covariance (our full anisotropic per-edge estimate), and Oracle\* (derived from ground-truth pose error, an upper bound).

Finally, since 6DOPE-GS’s code is unpublished, we report our runtime on its benchmarked GPU (RTX 4090) alongside their published numbers in Tab. 4. We also include BundleTrack’s reported runtime for reference, though it is likewise not directly comparable, as we lack access to its benchmarked GPU.

## 4.4. Sensitivity to Correspondence Quality

Tab. 5 directly compares our method to the dense alignment PGO method (BundleTrack) with the same feature matcher. With the state-of-the-art dense 2D feature matcher, RoMa v2, the two objectives perform comparably. To test this under lower-quality correspondences, we repeat the comparison with LoFTR, whose smaller network architecture and training data scale make it a weaker matcher. With LoFTR, our method leads on all datasets and both metrics, most notably by 6.4 ADD points on HO3D. This suggests our compact pose graph formulation degrades more gracefully than the prior method with dense-alignment objective as correspondence quality decreases. A fuller comparison, including frame-to-frame baselines and BundleTrack using native LF-Net [29] matcher, is in the supplementary material.

## 4.5. Drift Analysis

We evaluate drift on YCB Video, the longest sequence among our four benchmarks (Table 1), making it the most direct real-data test of long-horizon stability in our evaluation suite. Figure 4 shows rotation and translation drift over the full sequence. Our method maintains the lowest mean error on both axes throughout, and, critically, shows no sustained failure episode over the full run.

![](images/e529102d0fe7234a0ac265f5255c07e6188dced769b15711e2936bc8ce0d56e4.jpg)

![](images/62c30f381da4757d02e4d31921fa87290526d010d0b8e4e51b029d03b1ab6c77.jpg)  
Figure 4. Drift analysis on YCB Video dataset. Rotation and translation drift over the full 1718-timestep sequence.

## 4.6. Ablation Studies

We ablate the estimated uncertainty along three axes: how much of its structure is used, how it behaves as correspondence quality degrades, and how well it predicts the true error.

Uncertainty Structure. Tab. 6 compares four levels of uncertainty information used in pose-graph optimization: Identity (no per-edge uncertainty), Trace (isotropic, using only the scalar magnitude of our estimated covariance), Covariance (our full anisotropic per-edge estimate, Eqs. (8) and (9)), and Oracle (derived from pose error between estimated and ground truth, as an upper bound). Both the trace and the full covariance improve accuracy over unweighted edges. On two of the four benchmarks, however, the directional information in the full covariance brings no consistent benefit over purely magnitude-based weighting, which suggests that an anisotropic covariance only helps when its estimated directions are correct. The example in Fig. 5 (Top) shows an edge where this fails: the estimation is well-calibrated in translation but severely overconfident along rotation.

Uncertainty vs. Correspondence Quality. The lower block of Tab. 6 repeats the comparison with LoFTR in place of RoMa v2. The gain from covariance weighting over unweighted edges is larger with the weaker matcher, indicating that the estimated uncertainty matters most when the underlying relative poses are least reliable. This mirrors the trend in Tab. 5: with high-quality correspondences our compact formulation already matches the dense-alignment objective, and the uncertainty is what limits the influence of bad edges as correspondence quality drops.

Uncertainty–Error Correlation. We evaluate whether our estimated uncertainty predicts the true pose error by computing the Spearman rank correlation between the trace of the estimated per-edge uncertainty and the ground-truth error on HO3D and YCBInEOAT (Fig. 6). The positive correlation indicates that the predicted uncertainty is broadly associated with the true error.

![](images/a093a2bf9552e5f1b90d853c4333134f5eb3fc6891de75e649f304041b1b4a30.jpg)

![](images/b0274e4e6a380ed8062857de2589f1ab9c65aa32b3dbe6cfa6e1d7f554e02a40.jpg)  
Figure 5. Overconfident uncertainty for heavy occluded symmetric objects and thin objects from end-on views. Top: under the heavy occlusion, the hand covers the label, leaving only correspondences on the rotationally-symmetric body; Bottom: thin object viewed from its end-on view axis, leaving little visual signal. For the purpose of visualization, we show the square root of the covariance’s trace as a scalar uncertainty.

Failure Cases. We observe two main failure cases: symmetric objects with weak textures under heavy occlusions (Fig. 5 Top), and thin objects viewed from end-on views (Fig. 5 Bottom). We refer readers to the supplementary material for more quantitative results on uncertainty-weighted PGO and performance under challenging cases.

Keyframe Selection. Tab. 7 ablates the two components of our keyframe selection stage: the memory pool and farthestframe sampling (FFS). Removing the memory pool allows loop closure edges to be drawn from all past frames rather than from a small set of viewpoint-diverse ones, so the selected frames become redundant and carry less information. Replacing farthest frame sampling with the adjacent K frames leads to severe pose degeneration since graph connectivity becomes purely local. This confirms that the value of loop closure comes from spanning the sequence rather than from the number of edges.

## 5. Conclusion

In this work, we propose a lightweight, correspondenceagnostic framework for online 6D pose tracking of novel objects, built on a compact pose graph optimized with iSAM2. Rather than fusing dense correspondences into the objective, we use them only to produce relative pose measurements, which the pose graph combines with geometryderived per-edge uncertainty and viewpoint-diverse frame selection. Across four benchmarks, our method reaches near-best accuracy among non-CAD methods at a fraction of the optimization cost of reconstruction- and pose-graphbased baselines, while remaining stable over sequences of more than a thousand frames. Our results suggest that accurate object tracking does not require retaining dense correspondences throughout optimization. Instead, compact relative pose measurements, weighted by how well each is constrained, provide comparable accuracy at substantially lower cost.

![](images/da3b37196af16710e2ddecb1e247a6710e10a7bab7cbce7f6a6e85c45871ea96.jpg)

![](images/065f314f364932fc549d73822a49c6efe7b326809dd196510f4d549a08713dca.jpg)

![](images/74b6dedf20dc4a5c85b28095f891235690b10be5fad7b174754df14f18d85c35.jpg)

![](images/eac351501199355f840c279b18f43f9ed4bb5243926ef63df88e2331caccfd03.jpg)

Figure 6. Uncertainty vs. Pose Error. We show the joint distribution of estimated uncertainty and ground-truth pose error, for rotation (left) and translation (right), on HO3D and YCBInEOAT. Hexbin color indicates edge density on a log scale; the red line and shaded band show the median and interquartile range of error within log-spaced uncertainty bins.
<table><tr><td rowspan="2">Keyframe</td><td colspan="2">HO3D</td><td colspan="2">YCBInEOAT</td></tr><tr><td>ADD-S (%)</td><td>ADD (%)</td><td>ADD-S (%)</td><td>ADD (%)</td></tr><tr><td>Ours</td><td>96.37</td><td>92.23</td><td>94.18</td><td>88.75</td></tr><tr><td>w.o FFS</td><td>-4.22</td><td>-14.10</td><td>-1.71</td><td>-3.81</td></tr><tr><td>w.o Pool</td><td>-0.10</td><td>-0.29</td><td>-0.33</td><td>-0.56</td></tr></table>

Table 7. Ablation on keyframe selection. Ours w.o. FFS use the direct K neighbors to build loop closure edges; Ours w.o. memory pool allows loop-closure candidates to be drawn from all past frames.

## 6. Limitation

Our method has two main limitations. First, we assume the object is rigid and we leave tracking articulated and deformable objects as future work. Second, our covariance is a local approximation around the solution RANSAC-Umeyama converges to, so it can capture a flat, poorlyconstrained direction but not a well-separated alternative solution (e.g. for textureless symmetric objects). Further work could be a learning-based uncertainty estimator for 6D poses to represent multi-modal pose distributions.

## References

[1] Pablo Aguirrezabal, Iker Aguinaga, and Aitor Alvarez-Gila. Monocular rgb 6d object pose estimation for augmented reality: a survey. Virtual Reality, 2026. 1

[2] Luca Carlone and Giuseppe C Calafiore. Convex relaxations for pose graph optimization with outliers. IEEE Robotics and Automation Letters, 3(2):1160–1167, 2018. 2

[3] Luca Carlone, Rosario Aragues, Jose A Castellanos, and´ Basilio Bona. A linear approximation for graph-based simultaneous localization and mapping. In Robotics: Science and Systems, pages 41–48. MIT Press Cambridge, MA, USA, 2012. 2

[4] Yamei Chen, Yan Di, Guangyao Zhai, Fabian Manhardt, Chenyangguang Zhang, Ruida Zhang, Federico Tombari, Nassir Navab, and Benjamin Busam. Secondpose: Se (3)- consistent dual-stream feature fusion for category-level pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9959– 9969, 2024. 2

[5] Ho Kei Cheng and Alexander G. Schwing. XMem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 7, 1

[6] Angela Dai, Matthias Nießner, Michael Zollhofer, Shahram¨ Izadi, and Christian Theobalt. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG), 36(4): 1, 2017. 2

[7] Frank Dellaert and GTSAM Contributors. Gtsam 4.3.0, 2026. 2, 5, 1

[8] Frank Dellaert and Michael Kaess. Factor Graphsfor Robot Perception. Foundations and Trends in Robotics, Vol. 6, 2017. 2

[9] Zilong Deng, Shaochang Tan, Zuria Bauer, Daniel Bar´ ath,´ and Marc Pollefeys. Monotracker: Monocular rgb-only 6d tracking of unknown objects. In BMVC, 2025. 1

[10] Yaqing Ding, Viktor Kocur, Vaclav V´ avra, Zuzana Berger´ Haladova, Jian Yang, Torsten Sattler, and Zuzana Kukelova.´ Reposed: Efficient relative pose estimation with known depth information. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14876–14886, 2025. 2

[11] Johan Edstedt, Qiyu Sun, Georg Bokman, M¨ arten˚ Wadenback, and Michael Felsberg. Roma: Robust dense fea-¨ ture matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19790– 19800, 2024. 2, 3

[12] Johan Edstedt, David Nordstrom, Yushan Zhang, Georg¨ Bokman, Jonathan Astermark, Viktor Larsson, Anders Hey-¨ den, Fredrik Kahl, Marten Wadenb˚ ack, and Michael Fels-¨ berg. Roma v2: Harder better faster denser feature matching. arXiv preprint arXiv:2511.15706, 2025. 2, 3, 5

[13] Tom Fischer, Xiaojie Zhang, and Eddy Ilg. Unified categorylevel object detection and pose estimation from rgb images

using 3d prototypes. In Proceedings ofthe IEEE/CVF Inter national Conference on Computer Vision, pages 9790–9800, 2025. 2

[14] Jian Guan, Yingming Hao, Qingxiao Wu, Sicong Li, and Yingjian Fang. A survey of 6dof object pose estimation methods for different application scenarios. Sensors, 24(4): 1076, 2024. 1

[15] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vin cent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In CVPR, pages 3196–3206, 2020. 5, 1

[16] Richard I Hartley. In defense of the eight-point algorithm. IEEE Transactions on pattern analysis and machine intelli gence, 19(6):580–593, 1997. 2

[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically ac curate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 2

[18] Yufeng Jin, Vignesh Prasad, Snehal Jauhri, Mathias Franzius, and Georgia Chalvatzaki. 6dope-gs: Online 6d object pose estimation using gaussian splatting. In ICCV, pages 8032–8043, 2025. 1, 2, 5, 6

[19] Michael Kaess, Hordur Johannsson, Richard Roberts, Viorela Ila, John J Leonard, and Frank Dellaert. isam2: Incremental smoothing and mapping using the bayes tree. The International Journal ofRobotics Research, 31(2):216–235, 2012. 2, 4

[20] Taeyeop Lee, Bowen Wen, Minjun Kang, Gyuree Kang, In So Kweon, and Kuk-Jin Yoon. Any6d: Model-free 6d pose estimation of novel objects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11633–11643, 2025. 1, 2

[21] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua. Ep n p: An accurate o (n) solution to the p n p problem. Internationaljournal ofcomputer vision, 81(2):155–166, 2009. 2

[22] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In ICCV, pages 17627–17638, 2023. 2

[23] Mengya Liu, Siyuan Li, Ajad Chhatkuli, Prune Truong, Luc Van Gool, and Federico Tombari. One2any: One-reference 6d pose estimation for any object. In CVPR, pages 6457– 6467, 2025. 1, 2, 3, 5

[24] Xingyu Liu, Gu Wang, Ruida Zhang, Chenyangguang Zhang, Federico Tombari, and Xiangyang Ji. Unopose: Unseen object pose estimation with an unposed rgb-d reference image. In CVPR, pages 22023–22034, 2025. 1, 2, 3, 5

[25] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level humanobject interaction. In Proceedings of the IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 21013–21022, 2022. 5

[26] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from rgb images. In European Conference on Computer Vision, pages 298–315. Springer, 2022. 2

[27] David G Lowe. Object recognition from local scale-invariant features. In Proceedings of the seventh IEEE international conference on computer vision, pages 1150–1157. Ieee, 1999. 2

[28] Dominic Maggio, Hyungtae Lim, and Luca Carlone. Vggtslam: Dense rgb slam optimized on the sl (4) manifold. Advances in Neural Information Processing Systems, 38: 129839–129867, 2026. 2

[29] Yuki Ono, Eduard Trulls, Pascal Fua, and Kwang Moo Yi. Lf-net: Learning local features from images. Advances in neural information processing systems, 31, 2018. 7

[30] Linfei Pan, Daniel Bar ´ ath, Marc Pollefeys, and Johannes L´ Schonberger. Global structure-from-motion revisited. In¨ European Conference on Computer Vision, pages 58–77. Springer, 2024. 2

[31] Georgy Ponimatkin, Martin C´ıfka, Toma´s Souˇ cek, Mˇ ed´ eric´ Fourmy, Yann Labbe, Vladimir Petrik, and Josef Sivic. 6d´ object pose tracking in internet videos for robotic manipulation. arXiv preprint arXiv:2503.10307, 2025. 1

[32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2:¨ Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 7

[33] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: An efficient alternative to sift or surf. In ICCV, pages 2564–2571. Ieee, 2011. 2

[34] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020. 2

[35] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 2

[36] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, pages 8922–8931, 2021. 2, 3

[37] Martin Sundermeyer, Zoltan-Csaba Marton, Maximilian Durner, Manuel Brucker, and Rudolph Triebel. Implicit 3d orientation learning for 6d object detection from rgb images. In Proceedings of the european conference on computer vision (ECCV), pages 699–715, 2018. 2

[38] Bill Triggs, Philip F McLauchlan, Richard I Hartley, and Andrew W Fitzgibbon. Bundle adjustment—a modern synthesis. In International workshop on vision algorithms, pages 298–372. Springer, 1999. 4

[39] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on pattern analysis and machine intelligence, 13(4): 376–380, 1991. 4, 1

[40] Bowen Wen and Kostas Bekris. Bundletrack: 6d pose tracking for novel objects without instance or category-level 3d models. In International Conference on Intelligent Robots and Systems, pages 8067–8074. IEEE, 2021. 2, 5, 6, 1

[41] Bowen Wen, Chaitanya Mitash, Baozhang Ren, and Kostas E Bekris. se (3)-tracknet: Data-driven 6d pose tracking by

calibrating image residuals in synthetic domains. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10367–10373. IEEE, 2020. 5

[42] Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Muller, Alex Evans, Dieter Fox, Jan Kautz,¨ and Stan Birchfield. Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects. In CVPR, pages 606– 617, 2023. 1, 2, 5

[43] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In CVPR, pages 17868–17879, 2024. 1, 2, 5

[44] Yanming Wu, Hui Zhang, Patrick Vandewalle, Peter Slaets, and Eric Demeester. A comprehensive review on advances in instance-level 6d object pose tracking. Computer Vision and Image Understanding, page 104667, 2026. 1

[45] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv preprint arXiv:1711.00199, 2017. 5

[46] Ganlin Zhang, Viktor Larsson, and Daniel Barath. Revisiting rotation averaging: Uncertainties and robust losses. In Pro ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17215–17224, 2023. 2

# Fast Pose Tracking of Rigid Objects with Compact Pose Graph Optimization

Supplementary Material

## 7. Overview

In this supplementary document, we provide additional details and analyses that complement the main paper. We first describe implementation details, including data preprocessing and the hyperparameters used in our method (Sec. 8). We then derive the SE(3) pose uncertainty (Sec. 9). Next, we review the evaluation metrics for pose tracking quality (Sec. 10). We show the comparison to different tracking approaches (Sec. 11) and runtime breakdown (Sec. 12). Furthermore, we include detailed statistics of HOI4D dataset (Sec. 13). Finally, we provide additional qualitative results on uncertainty-aware PGO (Sec. 14) and challenge cases (Sec. 15).

## 8. Implementation Details

## 8.1. Input Preprocessing

We follow standard input preprocessing used in prior RGB-D tracking pipelines [40, 42]. For object masks, we use XMem [5] masks on HO3D [15] and ground-truth masks on other datasets. For correspondence estimation, we crop object-centric image regions and resize them to a fixed resolution: 320 320 for RoMa v2 and $4 0 0 \times 4 0 0$ for LoFTR.

## 8.2. Hyperparameters

For pose graph optimization, we use the Cauchy robust loss in GTSAM [7] with scale parameter k=1.0. We set the frame selection threshold to $\tau _ { k } = 2 5 ^ { \circ }$ for RoMa $\mathbf { v } 2$ and $\tau _ { k } { = } 1 5 ^ { \circ }$ for LoFTR. We use 1000 correspondences for both matchers in our method. For memory pool update, we set the threshold $\tau _ { m } = 5 ^ { \circ }$ . We set the RANSAC inlier threshold to $d _ { r } = 2 e - 3$ . These parameters are applied to all benchmarks where we evaluate our method, without tuning.

## 8.3. Ablation of Parameters

We ablate $\tau _ { k }$ for RoMa v2 and LoFTR. We observe that LoFTR is more sensitive to larger $\tau _ { k } ,$ while RoMa $\mathbf { v } 2$ is more robust to larger $\tau _ { k } .$ . The difference could come from the generalization of different methods to large viewpoint changes. RoMa $\mathbf { v } 2$ shows stronger performance in terms of handling such cases. However, for both cases, too small a $\tau _ { k }$ also suppresses the performance due to similar views that do not provide much additional information.

## 9. Uncertainty Estimation

In this section, we show how we derive the per-edge covariance $\Sigma _ { i k }$ . We use RANSAC Umeyama [39] to identify the inlier set of the correspondences $\mathcal { C } _ { i k } ^ { i n }$ . For simplicity, we omit the frame indices ik here. Our goal is to estimate the covariance $\Sigma \in \mathbb { R } ^ { 6 \times 6 }$ that describes the uncertainty of geometric alignment from the target function,

![](images/1e16664235b175e8c46f5ef175b5a45dbdf5ce9ca38fc783c0f970ef20843d54.jpg)  
Figure 7. Parameter Sweep of $\tau _ { k }$ for different feature matchers.

$$
\operatorname* { m i n } _ { R \in S O ( 3 ) , t \in \mathbb { R } ^ { 3 } } \sum _ { i = 1 } ^ { N } w _ { i } \ \| R b _ { i } + t - a _ { i } \| _ { 2 } ^ { 2 } , a _ { i } \in \mathbb { R } ^ { 3 } , b _ { i } \in \mathbb { R } ^ { 3 } ,\tag{10}
$$

where $N$ is the cardinality of the inlier set $| { \mathcal { C } } ^ { \mathrm { i n } } |$ . For each correspondence, we define the weighted residual

$$
e _ { i } ( R , t ) \triangleq \sqrt { w _ { i } } \left( R b _ { i } + t - a _ { i } \right) \in \mathbb { R } ^ { 3 } .\tag{11}
$$

We perturb the rotation on $S O ( 3 )$ as

$$
R ( \delta \theta ) = R \exp ( [ \delta \theta ] _ { \times } ) ,\tag{12}
$$

where $\delta \theta \in \mathbb { R } ^ { 3 }$ is a small rotation vector. Let

$$
r _ { i } ( \delta ) = R \exp ( [ \delta \theta ] _ { \times } ) b _ { i } + t - a _ { i } ,\tag{13}
$$

with $u \in \mathbb { R } ^ { 3 }$ an arbitrary direction. Using Rodrigues’ formula,

$$
\exp ( [ \delta \theta ] _ { \times } ) = I + \sin ( \delta ) [ \theta ] _ { \times } + ( 1 - \cos \delta ) [ \theta ] _ { \times } ^ { 2 } ,\tag{14}
$$

we obtain

$$
r _ { i } ^ { \prime } ( \delta ) = R \left( \cos ( \delta ) [ \theta ] _ { \times } + \sin ( \delta ) [ \theta ] _ { \times } ^ { 2 } \right) b _ { i } .\tag{15}
$$

Evaluating at δ = 0 gives

$$
r _ { i } ^ { \prime } ( 0 ) = R [ \theta ] _ { \times } b _ { i } = - R [ b _ { i } ] _ { \times } \theta .\tag{16}
$$

Therefore, the directional derivative of the weighted residual is

$$
e _ { i } ^ { \prime } ( 0 ) = \sqrt { w _ { i } } r _ { i } ^ { \prime } ( 0 ) = - \sqrt { w _ { i } } R [ b _ { i } ] _ { \times } \theta .\tag{17}
$$

Since $e _ { i } ^ { \prime } ( 0 ) = J _ { i , \mathrm { r o t } } \theta$ for arbitrary θ, the Jacobian with respect to the rotation perturbation is

$$
J _ { i , \mathrm { r o t } } = - \sqrt { w _ { i } } R [ b _ { i } ] _ { \times } .\tag{18}
$$

Similarly, the Jacobian with respect to translation is

$$
J _ { i , \mathrm { t r a n s } } = { \sqrt { w _ { i } } } I _ { 3 } .\tag{19}
$$

Thus, the full Jacobian for the i-th correspondence is

$$
J _ { i } = \sqrt { w _ { i } } \left[ - R [ b _ { i } ] _ { \times } \quad I _ { 3 } \right] \in \mathbb { R } ^ { 3 \times 6 } .\tag{20}
$$

Stacking all correspondences yields the Gauss–Newton approximation

$$
H \approx \sum _ { i = 1 } ^ { N } { J _ { i } ^ { \top } J _ { i } } .\tag{21}
$$

To calibrate the correct scale of Σ, we apply the residualvariance factor to align the uncertainty with the real scale as follows:

$$
\hat { \sigma } ^ { 2 } = \frac { \sum _ { n = 1 } ^ { N } w _ { i } \| r _ { i } \| _ { 2 } ^ { 2 } } { \operatorname* { m a x } ( 3 | \mathcal { C } ^ { \mathrm { i n } } | - 6 , 1 ) } .\tag{22}
$$

Last, reinstating frame indices yields $\Sigma _ { i k } = \sigma _ { i k } ^ { 2 } H ^ { - 1 }$

## 10. Metrics

We follow the evaluation protocol used in previous work [18, 40, 42] to assess 6-DoF object pose tracking quality. For each video, we align the predicted trajectory to the ground-truth coordinate frame using the relative transformation between the predicted and ground-truth pose at the first frame. We report the area under the curve (AUC) of the ADD and ADD-S metrics, computed over distance thresholds in the range [0, 0.1] m.

Given the ground-truth pose (R, t), the estimated pose <sup>(R,</sup> <sup>˜</sup> <sup>t˜),</sup> <sup>and</sup> <sup>the</sup> <sup>object</sup> <sup>model</sup> <sup>point</sup> <sup>set</sup> M<sup>,</sup> <sup>the</sup> <sup>metrics</sup> <sup>are</sup> defined as:

$$
\mathrm { A D D } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { x \in \mathcal { M } } \Big \| ( R x + t ) - ( \tilde { R } x + \tilde { t } ) \Big \| _ { 2 } ,\tag{23}
$$

$$
\mathrm { A D D - S } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { x _ { 1 } \in \mathcal { M } } \operatorname* { m i n } _ { x _ { 2 } \in \mathcal { M } } \Big \| ( R x _ { 1 } + t ) - ( \tilde { R } x _ { 2 } + \tilde { t } ) \Big \| _ { 2 } .\tag{24}
$$

Compared to ADD, ADD-S replaces one-to-one point correspondences with a closest-point distance and is therefore more suitable for symmetric objects, since there is symmetry ambiguity.

## 11. Comparison to Different Tracking Approaches

Tab. 8 compares our full method against direct frame-toframe (F2F) tracking and BundleTrack [40]. For F2F setting, the pose of each query image is estimated relative to the previous frame alone, using three correspondence estimators [12, 24, 36]. Among the F2F results, RoMa v2 and LoFTR are considerably more robust and stable than UNOPose (Tab. 8 top). Pairing our tracking module with each backend (Tab. 8 bottom) improves accuracy consistently, and the size of the gain scales with how weak the backend is on its own.

<table><tr><td rowspan="2">Setting</td><td rowspan="2">PGO Target</td><td rowspan="2">RPE</td><td colspan="2">HO3D</td><td colspan="2">YCBInEOAT</td></tr><tr><td></td><td>ADD-S (%) ADD (%)</td><td>ADD-S (%)</td><td>ADD (%)</td></tr><tr><td rowspan="3">F2F Tracking</td><td rowspan="3"></td><td>UNOPose</td><td>20.1</td><td>5.8</td><td>22.1</td><td>10.2</td></tr><tr><td>LoFTR</td><td>63.4</td><td>30.5</td><td>36.7</td><td>24.1</td></tr><tr><td>RoMa v2</td><td>78.0</td><td>56.7</td><td>70.8</td><td>52.5</td></tr><tr><td rowspan="3">BundleTrack Dense-Alignment</td><td rowspan="3"></td><td>LFNet</td><td>92.4</td><td>66.0</td><td>93.0</td><td>87.3</td></tr><tr><td>LoFTR</td><td>94.1</td><td>79.3</td><td>92.5</td><td>84.9</td></tr><tr><td>RoMa v2</td><td>96.5</td><td>92.6</td><td>93.8</td><td>88.0</td></tr><tr><td rowspan="3">Ours</td><td rowspan="3">Pose-Consistency</td><td>UNOPose</td><td>86.4</td><td>61.6</td><td>95.3</td><td>87.8</td></tr><tr><td>LoFTR</td><td>94.8</td><td>85.7</td><td>93.0</td><td>87.8</td></tr><tr><td>RoMa v2</td><td>96.4</td><td>92.2</td><td>94.4</td><td>88.8</td></tr></table>

Table 8. Comparison of tracking approaches on HO3D and YCBInEOAT with different correspondence and pose estimation methods. Top: direct frame-to-frame (F2F) tracking using each correspondence/relative pose method alone. Middle: Bundle-Track with different feature matchers. Bottom: our online pose graph optimization (PGO) module paired with different pose estimators. As visible, our module provides a significant accuracy gain in all cases.
<table><tr><td>Method</td><td>GPU</td><td>HOI4D</td><td>YCBInEOAT</td><td>HO3D</td><td>YCB-Video</td><td>Average</td></tr><tr><td>One2Any</td><td rowspan="5">H100</td><td>30.0</td><td>30.0</td><td>30.0</td><td>30.0</td><td>30.0</td></tr><tr><td>FoundationPose (Tracking)</td><td>10.0</td><td>33.0</td><td>18.3</td><td>30.1</td><td>22.9</td></tr><tr><td>UNOPose</td><td>10.0</td><td>10.0</td><td>10.0</td><td>10.0</td><td>10.0</td></tr><tr><td>BundleSDF</td><td>1.4</td><td>1.3</td><td>0.9</td><td>3.5</td><td>1.8</td></tr><tr><td>BundleTrack(R)</td><td>1.2</td><td>1.3</td><td>0.6</td><td>2.6</td><td>1.4</td></tr><tr><td>Ours (RoMa v2)</td><td>H100</td><td>9.8</td><td>10.0</td><td>9.4</td><td>10.8</td><td>10.0</td></tr></table>

Table 9. Efficiency comparison across benchmarks. Frame rate (fps) per method, reported per dataset rather than pooled, since sequence length and scene composition vary across benchmarks and can shift per-frame cost.

## 12. Runtime Analysis

Detailed Runtime. Tab. 9 provides a detailed runtime of our method and baselines on four benchmarks.

iSAM2 Relinearization Time. Fig. 9 shows iSAM2’s relinearization time over the YCB Video sequence (1718 frames, our longest): cost grows gradually but stays lowmillisecond throughout.

BundleTrack PGO Runtime. Fig. 8 reports our compact PGO update time (millisecond per frame) compared to dense alignment PGO method (BundleTrack [40]) on YCB Video dataset. For a fair comparison, we report both methods using the same feature matcher and the same number of correspondences. Our compact PGO updates at a significantly faster speed, around 2 milliseconds per frame on average.

![](images/5233be39d2efb4616a0e6c5b993b543195ac55c084c450e68c9162cf3ae89944.jpg)  
Figure 8. PGO runtime compared to BundleTrack on YCB Video dataset. We use the same number of correspondences for both method. Runtime is reported in millisecond per frame.

![](images/e28cb07d3c2952e8fdbc9eef53741d016f88e92a394692db9ac28d63bb35b2c6.jpg)

Figure 9. Relinearization time of iSAM2. Result reports the time in milliseconds per frame averaged over frames on the YCB-Video dataset.
<table><tr><td rowspan="2">Method</td><td colspan="5">AUC@10°</td><td rowspan="2">Avg.</td></tr><tr><td>の</td><td>1</td><td></td><td>1</td><td>マ</td></tr><tr><td>FoundationPose(CAD)</td><td>25.5</td><td>3.6</td><td>41.1</td><td>20.6</td><td>70.5</td><td>32.2</td></tr><tr><td>BundleSDF</td><td>41.3</td><td>47.2</td><td>39.1</td><td>68.4</td><td>21.4</td><td>43.5</td></tr><tr><td>UNOPose</td><td>35.9</td><td>42.9</td><td>37.1</td><td>68.7</td><td>19.1</td><td>40.8</td></tr><tr><td>BundleTrack (R.)</td><td>41.0</td><td>47.4</td><td>40.5</td><td>70.4</td><td>21.5</td><td>44.2</td></tr><tr><td>Ours</td><td>46.6</td><td>46.5</td><td>47.0</td><td>70.0</td><td>22.2</td><td>46.5</td></tr></table>

Table 10. Per-category rotation accuracy on five HOI4D rigid object categories. We report rotation error AUC@10<sup>◦</sup> (higher is better). Our method leads on three of five non-CAD categories and on average, most notably on knife and toy car.

## 13. Results on HOI4D

Tab. 10 provides per-category results on rotation metric AUC@10◦ on the HOI4D dataset compared to the state-ofthe-art methods.

## 14. Qualitative Results of Uncertainty-Aware PGO

Fig. 10 shows the qualitative results of pose tracking using uncertainty-aware PGO compared to unweighted (using

identity as the uncertainty) PGO.

## 15. Qualitative Results on Challenge Cases

Fig. 11, Fig. 12 and Fig. 13 show the failure cases of our method on textureless symmetric objects and thin objects viewed from their end-on view axis. We also provide other challenges, e.g., thin objects with heavy occlusions (Fig. 14), motion blur (Fig. 15, Fig. 16), heavy occlusion (Fig. 17), objects with weak texture (Fig. 18), where our method successfully tracks poses.

![](images/52207861ace7519fe53f170d5b01e7e0d8663bb0e3b041dd9d9f98da96bb2920.jpg)  
Figure 10. Qualitative comparison of tracking without and with uncertainty modeling. Each pair shows the tracking result without un certainty (top) and with uncertainty (bottom). Modeling uncertainty improves robustness in geometrically ambiguous cases, including textureless and thin objects. 4

Current Frame  
GT Pose  
![](images/09319124ed1c5810d0fce06fb539b3ccf61244d3ac0c35594f60b12fab6dd6b2.jpg)  
Estimated Pose

Figure 11. Failure on textureless symmetric object. Poor ADD metric due to the symmetry ambiguity.  
![](images/27db4469f6c5f9d9e536569b9a4c232f7a9ed46ef74cd73898511a2c30c364ca.jpg)

Figure 12. Failure on textureless symmetric object.  
![](images/ef9d1c01617a6ee9ec8a560b950a2cf06fc678288498d8d702a889d0254c12b2.jpg)

Figure 13. Failure cases when viewing thin long objects from their end-on view axis.  
![](images/0cf412b3e566dd0c3472d648b79d8ba7ac280d52e318cfd1aac97ef0cb1c6013.jpg)

Figure 14. Track thin objects from their non end-on view viewed under heavy hand occlusion.  
![](images/241eb8992fd19053fb5e3d8f4ba71b1edbf1f3a8ee23a4e6f251781dc426ad1f.jpg)  
Figure 15. Track the object pose under motion blur.

![](images/dd39c62f3721719b3e821bd8df0f07bd2f4c1197ffbed3842fac42dbfd7ea9ba.jpg)

Figure 16. Track the object pose under motion blur.  
![](images/18b6785f532f7ead7a56601f0ac92237e38edc085f136bab9d964b8eb1d11d5a.jpg)

Figure 17. Track the object pose under heavy hand occlusion.  
![](images/c9b1d0f8bfdb4d69df31e3715d919d85da9c82ee723539ca32ae9923696b5430.jpg)  
Figure 18. Track the pose of weak texture objects.