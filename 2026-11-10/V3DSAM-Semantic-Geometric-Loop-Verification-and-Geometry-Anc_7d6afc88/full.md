# V3DSAM: Semantic-Geometric Loop Verification and Geometry-Anchored Persistent Instance Mapping for Feed-Forward Dense SLAM

Ye Maolin, Zhao Dongqing<sup>∗</sup>, Li Linyang, Xue Shuai, Liu Yuan, Wang Yiyuan, and Li Wen Information Engineering University <sup>\*</sup>Corresponding author: Zhao Dongqing.

Abstract— To address unreliable cross-view association in long-sequence feed-forward dense SLAM, this letter presents V3DSAM, an open-vocabulary semantic dense SLAM system that establishes physical correspondences at both the scene and object levels. The system combines feed-forward geometric estimates from VGGT-SLAM 2.0 with open-vocabulary semantic observations from SAM3. We introduce semantic-geometric dual-scale loop verification, which identifies false loop closures using pixel-level bidirectional reprojection and object-level 3D spatial correspondences, without accessing backbone-internal features. We further propose geometry-anchored persistent instance mapping, which defines physical instances through class-constrained 3D connectivity in an optimized global coordinate system and assigns global IDs without inter-frame identity propagation. The proposed verifier rejects all four false loop candidates in our TUM RGB-D evaluation, while long-sequence ScanNet experiments show improved instance identity consistency. Thus, V3DSAM reduces dependence on internal backbone features and temporal ID propagation for robust long-term cross-view association.

Index Terms— SLAM, Semantic Scene Understanding, Loop Closure Detection, Persistent Instance Mapping.

## I. INTRODUCTION

Visual simultaneous localization and mapping (SLAM) is fundamental to autonomous navigation, augmented reality, and 3D scene understanding. Recently, visual geometry foundation models such as VGGT have enabled feed-forward estimation of camera poses and dense scene geometry through large-scale multiview pretraining, providing a new geometric basis for dense visual SLAM [1]. Meanwhile, openvocabulary vision models, including the SAM family and CLIP, enable flexible category- and object-level semantics in 3D scene representations, facilitating semantic SLAM systems that combine geometric reconstruction with openvocabulary scene understanding [2], [3], [4], [5].

A central challenge in extending local feed-forward geometry to long-sequence SLAM is establishing reliable and persistent physical correspondences across observations separated in time and viewpoint. The challenge manifests as scene-level false loop closures and object-level instance drift. At the scene level, appearance-based image retrieval can efficiently generate revisit candidates, but visual similarity does not guarantee that a physical location has been revisited; false-positive loop constraints may cause trajectory drift and map deformation. At the object level, association based on local temporal propagation or 2D appearance features may suffer from identity fragmentation and ID reassignment under long temporal gaps and large viewpoint changes.

![](images/716ab6f13a8d3d99e31e19f1991719e2b7179e60f7b887487992cd845f23e92a.jpg)  
Fig. 1. Dense RGB and semantic point clouds, camera trajectories, and instance-labeled observations from three viewpoints (C1–C3).

Fig. 1 visualizes the outputs of V3DSAM, our openvocabulary semantic dense SLAM system. Its main contributions are:

1) Semantic-geometric dual-scale loop verification. Using only explicit feed-forward geometry and external openvocabulary semantics, the method distinguishes true and false loops through bidirectional pixel-level reprojection and object-level 3D spatial correspondence. It does not rely on backbone-internal features and suppresses trajectory drift caused by false-positive loops.

2) Geometry-anchored persistent instance mapping. Multiview semantic observations are fused in a globally optimized world coordinate system. Voxelization and class-constrained 3D connectivity analysis form physical instances, and each valid connected component receives a globally unique instance ID. This avoids inter-frame temporal ID propagation and improves identity consistency across long-term revisits.

## II. RELATED WORK

## A. Feed-Forward Visual Geometry and Dense SLAM

Traditional iterative visual SLAM systems, such as ORB-SLAM3 and DROID-SLAM, estimate camera poses and 3D structure by iterative bundle adjustment with handcrafted features or dense photometric constraints [6], [7], [8]. Recently, visual geometry foundation models, including DUSt3R, MASt3R, and VGGT, have learned global geometric priors from large-scale multiview pretraining. Given one or more images, they directly predict camera poses, dense depth, and 3D point clouds without frame-by-frame iterative optimization [1], [9], [10]. Building on these models, MASt3R-SLAM, VGGT-SLAM, and VGGT-SLAM 2.0 combine keyframe management, local maps, loop closure, and global optimization to extend feed-forward geometry to long-sequence dense SLAM [11], [12], [13]. This work further investigates the consistency of cross-view association in long sequences.

## B. Loop Candidate Retrieval and Verification

Loop-closure processing comprises candidate retrieval (loop detection) and loop verification. The former retrieves potential revisit pairs from historical keyframes, whereas the latter rejects false candidates to prevent erroneous constraints from affecting global optimization.

Candidate retrieval commonly uses global image descriptors such as NetVLAD and SALAD [14], [15]; Hydra instead employs hierarchical 3D scene structures for loop association [16]. Conventional verification often relies on geometric registration such as ICP, which performs reliably in scaleconsistent iterative SLAM but may be challenged by noise in feed-forward geometry and semantic ambiguity [17].

In feed-forward SLAM, MASt3R-SLAM reprocesses candidate frames through its backbone for two-view inference and verifies them using matching quality, coupling verification to the encoder and introducing additional inference cost [13]. EC3R-SLAM uses a homography-RANSAC inlier ratio for loop verification, relying on geometric cues without object-level semantics [18]. VGGT-SLAM 2.0 extracts the signal $\alpha _ { m a t c h }$ from VGGT attention layers to verify retrieved candidates without additional training; however, this signal is tied to a specific network architecture and is less readily reusable. Establishing interpretable cross-view physical correspondence from explicit feed-forward 3D geometry and independent semantic observations, without using backbonespecific features, therefore remains of interest.

## C. Cross-View Instance Association and Open-Vocabulary Instance Mapping

Early instance SLAM methods often depend on predefined semantic categories or local instance association. MaskFusion establishes instance correspondences using 2D masks and temporal tracking [19], PanopticFusion fuses panoptic observations into a 3D voxel map to maintain identities [20], and SGS-SLAM jointly optimizes geometry and semantics with 3D Gaussian representations [21]. Nevertheless, stable identities remain challenging across long temporal gaps and large viewpoint changes.

With advances in vision-language pretraining, CLIP-Fields aggregates CLIP features into 3D point clouds for pointlevel open-vocabulary queries, primarily targeting offline multiview fusion [22]. OVO combines semantic categories, vision-language embeddings, temporal tracking, and spatial constraints to build instance-level maps, but long-term identity persistence remains limited [23]. These approaches place greater emphasis on semantic representation richness than on long-term identity consistency.

For semantic mapping built on feed-forward geometry, SceneVGGT back-projects SAM masks to construct semantic point clouds, but instance partitioning still depends heavily on inter-frame 2D semantic-label matching [24]. IRIS-SLAM fine-tunes the feed-forward backbone to learn dedicated instance embeddings, coupling its approach to internal model representations [25]. In multiview 3D instance construction, Open3DIS and MaskClustering use 2D masks to guide 3D instance aggregation, but do not specifically address noise in predicted feed-forward geometry [26], [27]. Maintaining long-term global instance identities under such geometric noise, without temporal ID propagation, remains an open challenge.

## III. METHOD

## A. Overview of V3DSAM

As shown in Fig. 2, V3DSAM takes a continuous monocular RGB sequence as input. VGGT-SLAM 2.0 and SAM3 provide camera intrinsics and extrinsics, dense depth, and semantic masks. An appearance-retrieval module generates loop candidates, which are processed by semantic-geometric dual-scale verification. Only candidates satisfying both consistency constraints enter global geometric optimization. In the resulting optimized world coordinate system, multiview semantic observations are fused by geometry-anchored instance assignment to construct a persistent dense 3D instance map.

## B. Semantic-Geometric Dual-Scale Loop Verification

We formulate loop-candidate verification as a binary hypothesis test. Given a query frame q and a candidate frame c, the feed-forward geometry provides their relative transform $\mathbf { T } _ { c  q } \in S E ( 3 )$ . Hypothesis $H _ { 1 }$ denotes a true loop, where the two frames observe the same physical region; $H _ { 0 }$ denotes a false loop, where the views appear similar but lack spatialsemantic correspondence.

We define a semantic-geometric dual-scale criterion from pixel-level consistency $C _ { p i x }$ and object-level consistency $C _ { o b j }$ . A candidate is accepted only when both scores satisfy their respective thresholds, as illustrated in Fig. 3.

For a valid semantic pixel at $( u _ { i } , v _ { i } )$ in keyframe k, its depth and the camera intrinsics recover the corresponding point in the camera coordinate system:

![](images/3f1f41217f0995d65f0acadac68b1eff8078853f67c9c4f700c43b853ff155d2.jpg)  
Fig. 2. Overall system architecture of V3DSAM.

![](images/b8520d758f1282d1a5c0b34571c2de58beb2fecf3e1762962cd9abbf668e06c2.jpg)  
Fig. 3. Principle of semantic-geometric dual-scale loop verification.

$$
\mathbf { p } _ { i } ^ { k } = D _ { k } \left( u _ { i } , v _ { i } \right) \left[ \left( u _ { i } - c _ { x } \right) / f _ { x } \right] , \quad l _ { i } = S _ { k } \left( u _ { i } , v _ { i } \right)\tag{1}
$$

Here, $D _ { k } \left( u _ { i } , v _ { i } \right)$ is the depth at the pixel, $f _ { x } , f _ { y } , c _ { x } , c _ { y }$ are the camera intrinsic parameters, $\mathbf { p } _ { i } ^ { k }$ is the back-projected 3D point, and $l _ { i }$ is the open-vocabulary semantic class supplied by SAM3. Applying this operation to the query and candidate frames yields two semantic 3D point sets.

Unidirectional reprojection is affected by image-boundary truncation. A reprojected point is counted only if its image coordinates lie within $u \in [ 0 , W ) , v \in [ 0 , H )$ . We define the projection validity indicator as

$$
\mathbb { I } ( u , v ) = { \left\{ \begin{array} { l l } { 1 , } & { 0 \leq u < W , \ 0 \leq v < H } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. }\tag{2}
$$

where W and H denote image width and height.

To alleviate statistical bias from boundary truncation, we use bidirectional reprojection for pixel-level semantic consistency. Query-frame points are transformed into the candidate-frame coordinate system by $\mathbf { T } _ { c  q }$ and perspectively projected. Swapping $q$ and c repeats the operation in the reverse direction. The pixel-level consistency score is

$$
C _ { p i x } = \frac { 1 } { 2 } ( \frac { N _ { m a t c h } ^ { q  c } } { N _ { v a l i d } ^ { q  c } } + \frac { N _ { m a t c h } ^ { c  q } } { N _ { v a l i d } ^ { c  q } } )\tag{3}
$$

Here, $N _ { v a l i d }$ denotes the number of valid semantic reprojections in a given direction, and $N _ { m a t c h }$ the number for which the source 3D point’s semantic class matches the target pixel’s class. A higher $C _ { p i x }$ indicates stronger agreement between the two views under the hypothesized relative pose.

$C _ { p i x }$ measures local dense semantic alignment but can be disturbed by fragmented masks and partial occlusion. We therefore introduce an object-level constraint. Within each frame, 3D semantic points of the same class are spatially clustered; noisy clusters containing fewer than $N _ { \mathrm { m i n } }$ points are filtered out. Each valid cluster is represented by its 3D centroid. Object correspondence between frames is then established using the known relative pose $\mathbf { T } _ { c  q } .$ A match must satisfy two conditions: the semantic classes agree, and the 3D Euclidean distance between centroids after pose transformation is below a preset threshold. Bidirectional matching is performed for $q \to c$ and $c \to q$ , and the number of matched objects in each direction is counted.

$$
C _ { o b j } = \frac { 1 } { 2 } ( \frac { M _ { q  c } } { N _ { q } } + \frac { M _ { c  q } } { N _ { c } } )\tag{4}
$$

Here, $N _ { q }$ and $N _ { c }$ denote the numbers of valid objects participating in 3D matching in the two views, while $M _ { q  c }$ and $M _ { c  q }$ are the numbers of successful 3D correspondences in the respective directions.

$C _ { p i x }$ emphasizes local dense alignment, whereas $C _ { o b j }$ captures the global arrangement of physical entities. Together they characterize semantic-geometric agreement from complementary perspectives. Loop verification is expressed as a binary objective with hard threshold constraints:

$$
\mathcal { L } _ { \mathrm { l o o p } } ( \mathbf { T } _ { c  q } ) = \{ \begin{array} { c c } { 0 , } & { C _ { p i x } \geq \tau _ { p i x } \land C _ { o b j } \geq \tau _ { o b j } } \\ { + \infty , } & { \mathrm { o t h e r w i s e } } \end{array}\tag{5}
$$

If $\mathcal { L } _ { \mathrm { l o o p } } = 0 ,$ the loop constraint is accepted for backend global optimization. If $\mathcal { L } _ { \mathrm { l o o p } } = + \infty$ , the candidate is treated as a false loop, $H _ { 0 }$ , and discarded without entering subsequent optimization.

C. Class-Constrained Instance Construction and Geometry-Anchored Identity Maintenance

Conventional instance association frequently propagates instance IDs through matches between adjacent frames. This makes identity depend strongly on temporal order: long-term revisits or severe occlusions can break the tracking chain, causing ID changes and identity resets.

We instead formulate instance partitioning directly in the unified world coordinate system obtained by global optimization. Multiview observations yield a noisy semantic point set $\mathcal { P } = \{ ( \mathbf { p } _ { i } , l _ { i } ) \}$ , where $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ represents a 3D coordinate and $l _ { i }$ its semantic class label. The objective is to partition $\mathcal { P }$ into physical entities whose points are both semantically consistent and spatially connected in 3D.

The raw multiview point cloud contains repeated observations and sampling noise. We therefore introduce voxelization to regularize these observations. For all points $\{ \mathbf { p } _ { m } , l _ { m } \} _ { m \in \mathcal { M } _ { v } }$ within voxel v, the voxel position is the coordinate mean and its semantic label is determined by majority voting:

$$
\begin{array} { l } { \displaystyle \mathbf { p } _ { v } = \frac { 1 } { \vert \mathcal { M } _ { v } \vert } \sum _ { m \in \mathcal { M } _ { v } } \mathbf { p } _ { m } , } \\ { \displaystyle \ell _ { v } = \arg \underset { \ell } { \operatorname* { m a x } } \sum _ { m \in \mathcal { M } _ { v } } \mathbb { I } ( \ell _ { m } = \ell ) . } \end{array}\tag{6}
$$

Here, $\mathcal { M } _ { v }$ is the set of raw 3D points assigned to the voxel, and I(·) denotes an indicator function. Equation (6) produces the regularized voxel set ${ \cal { V } } = \{ ( { \bf p } _ { v } , l _ { v } ) \}$

A class-constrained 3D adjacency graph $G = ( \nu , \mathcal { E } )$ is built over the regularized voxels. An edge between voxels $v _ { i }$ and $v _ { j }$ is created if and only if they have the same semantic class and their Euclidean distance is below neighborhood threshold $\epsilon _ { c } .$

$$
\left( v _ { i } , v _ { j } \right) \in \mathcal { E } \Longleftrightarrow l _ { i } = l _ { j } \wedge \| \mathbf { p } _ { i } - \mathbf { p } _ { j } \| _ { 2 } < \epsilon _ { c }\tag{7}
$$

Physical instances are defined by paths in the adjacency graph: two voxels of the same class belong to the same instance if a path of neighboring same-class voxels connects them. This connectivity relation is an equivalence relation and partitions the voxel set into disjoint connected components, each representing a physical 3D instance. A KD-tree performs neighborhood searches among voxels of the same class, and a union-find algorithm efficiently computes the connected components. After discarding noisy components with fewer voxels than a minimum threshold, each remaining component receives a unique global instance ID that serves as the persistent identity of the physical entity.

To associate 3D instances with 2D observations, instance voxels are projected into keyframe images, producing an instance mask $M _ { 3 D }$ . Its intersection-over-union with the SAM 2D mask $M _ { 2 D }$ is computed under the same semantic class constraint:

$$
I o U \left( M _ { 3 D } , M _ { 2 D } \right) = \frac { \left| M _ { 3 D } \cap M _ { 2 D } \right| } { \left| M _ { 3 D } \cup M _ { 2 D } \right| }\tag{8}
$$

The highest-IoU 2D mask within the same class is assigned the global instance ID. If no mask reaches the minimum matching threshold, no 2D instance label is assigned in that frame.

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

1) Datasets: Experiments use ScanNet [28] and TUM RGB-D [29] indoor datasets. The TUM RGB-D experiments cover eight Freiburg1 sequences. With a unified frontend configuration, these sequences yield 15 SALAD loop candidates: 11 true and four false loops. Together with four ScanNet scenes, the evaluation contains 46 candidate frame pairs. All hyperparameters of the loop-verification module are selected using TUM RGB-D only and transferred directly to ScanNet without retuning, to test cross-dataset generalization under fixed parameters.

2) Parameters and Hardware: VGGT-SLAM 2.0 provides the feed-forward geometry, and SAM3 generates openvocabulary semantic observations for keyframes. On TUM RGB-D, each submap contains 32 frames and the SALAD candidate threshold is 0.95.

For loop verification, the pixel- and object-level consistency thresholds are $\tau _ { p i x } = 0 . 1 0$ and $\tau _ { o b j } = 0 . 2 5$ , respectively. Within a frame, same-class 3D clustering uses a 0.3 m neighborhood threshold and removes sparse clusters containing fewer than 80 points. Persistent instance mapping uses a 0.02 m voxel resolution and a 0.05 m 3D neighborhood connection distance, and filters out connected components containing fewer than 100 voxels.

Experiments are conducted on an NVIDIA RTX 4090 GPU and an Intel i7-14700 CPU.

3) Evaluation Metrics: Trajectory accuracy is measured using absolute trajectory error (ATE-RMSE). Loop verification is evaluated using a confusion matrix, precision, recall, and F1-score.

Following the overlap-based loop-labeling principle in [30], we define loop ground truth from the extent of shared 3D visibility computed using ground-truth poses and depth.

Ground-truth loop labels are computed from dataset poses and depth. Symmetric geometric overlap is defined as the average of the two directions:

$$
O ^ { G T } ( q , c ) = \frac { 1 } { 2 } ( \frac { N _ { q  c } ^ { v i s } } { N _ { q } ^ { v a l i d } } + \frac { N _ { c  q } ^ { v i s } } { N _ { c } ^ { v a l i d } } )\tag{9}
$$

Here, $N _ { q } ^ { v a l i d }$ denotes the number of valid depth points in the query frame and $N _ { q  c } ^ { v i s }$ the number that establish depthconsistent overlap in the candidate frame after transformation using the ground-truth pose. A candidate is labeled a true loop only if its temporal separation exceeds 20 frames and its overlap is at least 0.30. Ground truth is used for offline evaluation only, not during retrieval, verification, or optimization.

Instance mapping is also evaluated using standard instance-SLAM measures: multiple object tracking accuracy (MOTA), 4D panoptic quality (4D-PQ), and ID-switch count (IDS), for comparison with prior instance mapping methods [31], [32]. These composite measures account for detection, segmentation, misses, false positives, and identity switches, but do not fully characterize identity persistence across long temporal intervals and loop revisits. We therefore additionally report identity-discrimination consistency (IDC), longterm reidentification recall (ReID-R), and instance coverage as targeted auxiliary metrics.

TABLE I  
COMPARISON OF LOOP-VERIFICATION SIGNALS ON THE SAME SALAD CANDIDATE SET
<table><tr><td>Verification signal</td><td>FP↓</td><td>Precision ↑</td><td>F1 ↑</td></tr><tr><td>SALAD retrieval</td><td>4</td><td>0.733</td><td>0.846</td></tr><tr><td>VGGT</td><td>0</td><td>1.000</td><td>1.000</td></tr><tr><td>Semantic only (Category Jaccard)</td><td>2</td><td>0.846</td><td>0.917</td></tr><tr><td>Geometry only (Depth consistency)</td><td>2</td><td>0.846</td><td>0.917</td></tr><tr><td>Semantic-Geometric 3D (Ours)</td><td>0</td><td>1.000</td><td>1.000</td></tr></table>

Note: FP is the number of false-positive loop candidates; ↑ and ↓ indicate preferred metric directions.

For IDC, let $N _ { g }$ be the number of visible observations of ground-truth instance $^ { g , }$ and $N _ { g , j }$ the number of these observations assigned predicted global ID j. Then

$$
I D C = \frac { \sum _ { g \in \mathcal { G } } \operatorname* { m a x } _ { j } N _ { g , j } } { \sum _ { g \in \mathcal { G } } N _ { g } }\tag{10}
$$

Here, max<sub>j</sub> $N _ { g , j }$ is the highest occurrence count of any predicted ID assigned to ground-truth instance $g .$ Higher IDC indicates that an object retains a more stable identity throughout the sequence.

For long-term Re-ID recall, first define the set of long-term visible instances:

$$
\mathcal { G } _ { L } = \left\{ g \mid t _ { g } ^ { l a s t } - t _ { g } ^ { f i r s t } \geq 0 . 2 5 T \right\}\tag{11}
$$

where $T$ is the sequence length. The recall is then computed as

$$
R e I D – R = \frac { N _ { s a m e I D } } { N _ { l o n g - t e r m } }\tag{12}
$$

Here, $\begin{array} { l l l } { { N _ { l o n g - t e r m } } } & { { = } } & { { | \mathcal { G } _ { L } | } } \end{array}$ is the number of groundtruth instances satisfying the long-term span criterion, while $N _ { s a m e I D }$ counts instances that are successfully matched at both the first and last observations and retain the same predicted global ID. Failure to detect either endpoint or different IDs at the endpoints is counted as a Re-ID failure.

## B. Loop Verification and Localization Accuracy

Table I compares false positives (FP), precision, and F1- score for different verification strategies on the same SALAD candidate set.

Table I shows that semantic-only and geometry-only verification each retain two false-positive loops, whereas the joint criterion rejects all four false candidates while preserving the 11 true loops (precision and F1-score of 1.000). The two cues are complementary: semantics suppresses geometric ambiguity, while 3D correspondence rejects scenes with similar category compositions.

![](images/8ec491adfb1ca14d8c1b56e05733c189c9a107a953facede817373fae2b25af9.jpg)  
Fig. 4. Distribution of true and false loop candidates in pixel-level and object-level consistency scores on TUM RGB-D. Green circles indicate true loops; red crosses indicate false loops. Dashed lines mark the thresholds $\tau _ { p i x } = 0 . 1 0$ and $\tau _ { o b j } = 0 . 2 5$

Fig. 4 shows a clear separation between true and false loop candidates on TUM RGB-D in the two-dimensional feature space defined by pixel-level consistency $C _ { p i x }$ and objectlevel consistency $C _ { o b j }$ . False loops often lack matchable 3D object structures that are visible in both views, resulting in low consistency scores at both the pixel and object levels. In contrast, true loops satisfy both pixel-level geometric alignment and object-level instance-matching constraints, yielding high scores on both measures.

We remove the false loops selected by the semanticgeometric dual-scale verification module before global pose optimization. Table II reports the trajectory accuracy after removing these erroneous constraints.

The baseline results in Table II are taken from the publicly available VGGT-SLAM 2.0 study [12]. The table compares absolute trajectory error (ATE-RMSE) across eight TUM RGB-D sequences for several recent SLAM systems. ”Ours (no Verification)” is an ablation that removes semanticgeometric dual-scale verification and directly accepts all candidate loops. Without verification, false loops introduce erroneous pose constraints and corrupt global trajectory optimization. As a result, severe drift and increased ATE occur on the perceptually ambiguous fr1/360 and fr1/room sequences. Enabling our verification method filters out these false candidates and prevents their constraints from degrading the back-end optimization. Overall, our method yields pose accuracy comparable to VGGT-SLAM 2.0’s internal attention score $\alpha _ { m a t c h }$ , while suppressing false-loop interference and preserving the global correction obtained from true loops.

As shown in Table III, when all candidates are true, the verification module retains all valid geometric constraints and the trajectory accuracy is unchanged. Only when false loops are detected does the module remove erroneous constraints and correct the back-end optimization. On scene0207 00, the method identifies and removes two false loops, reducing ATE from 0.109 m to 0.089 m. For the other scenes, which have no false candidates, the method retains valid loop constraints without reducing pose accuracy.

TABLE II  
ATE-RMSE ON EIGHT TUM RGB-D SEQUENCES
<table><tr><td>Setting</td><td>Method</td><td>360</td><td>desk</td><td>desk2</td><td>plant</td><td>room</td><td>rpy</td><td>teddy</td><td>xyz</td></tr><tr><td>Calibrated</td><td>ORB-SLAM3</td><td></td><td>0.017</td><td>0.210</td><td>0.034</td><td></td><td></td><td></td><td>0.009</td></tr><tr><td></td><td>DeepV2D</td><td>0.243</td><td>0.166</td><td>0.379</td><td>0.203</td><td>0.246</td><td>0.105</td><td>0.316</td><td>0.064</td></tr><tr><td></td><td>DeepFactors</td><td>0.159</td><td>0.170</td><td>0.253</td><td>0.305</td><td>0.364</td><td>0.043</td><td>0.601</td><td>0.035</td></tr><tr><td></td><td>DPV-SLAM</td><td>0.112</td><td>0.018</td><td>0.029</td><td>0.021</td><td>0.330</td><td>0.030</td><td>0.084</td><td>0.010</td></tr><tr><td></td><td>DPV-SLAM++</td><td>0.132</td><td>0.018</td><td>0.029</td><td>0.022</td><td>0.096</td><td>0.032</td><td>0.098</td><td>0.010</td></tr><tr><td></td><td>GO-SLAM</td><td>0.089</td><td>0.016</td><td>0.028</td><td>0.026</td><td>0.052</td><td>0.019</td><td>0.048</td><td>0.010</td></tr><tr><td></td><td>DROID-SLAM</td><td>0.111</td><td>0.018</td><td>0.042</td><td>0.016</td><td>0.049</td><td>0.026</td><td>0.048</td><td>0.012</td></tr><tr><td></td><td>MASt3R-SLAM</td><td>0.049</td><td>0.016</td><td>0.024</td><td>0.020</td><td>0.061</td><td>0.027</td><td>0.041</td><td>0.009</td></tr><tr><td>Uncalibrated</td><td>DROID-SLAM</td><td>0.202</td><td>0.032</td><td>0.091</td><td>0.045</td><td>0.918</td><td>0.056</td><td>0.045</td><td>0.012</td></tr><tr><td></td><td>MASt3R-SLAM</td><td>0.070</td><td>0.035</td><td>0.055</td><td>0.035</td><td>0.118</td><td>0.041</td><td>0.114</td><td>0.020</td></tr><tr><td></td><td>ViSTA-SLAM</td><td>0.104</td><td>0.030</td><td>0.030</td><td>0.052</td><td>0.067</td><td>0.023</td><td>0.080</td><td>0.015</td></tr><tr><td></td><td>VGGT-SLAM 2.0</td><td>0.050</td><td>0.025</td><td>0.029</td><td>0.026</td><td>0.063</td><td>0.026</td><td>0.038</td><td>0.014</td></tr><tr><td></td><td>Ours (no verification)</td><td>0.189×</td><td>0.025</td><td>0.028</td><td>0.035</td><td>0.815×</td><td>0.026</td><td>0.043</td><td>0.014</td></tr><tr><td></td><td>V3DSAM (Ours)</td><td>0.049</td><td>0.025</td><td>0.028</td><td>0.035</td><td>0.074</td><td>0.026</td><td>0.043</td><td>0.014</td></tr></table>

Note: All ATE-RMSE values are reported in meters (m); lower is better. A dash denotes tracking failure; × marks drift due to false loops. Bold denotes the lowest ATE within each calibration setting; calibrated and uncalibrated methods are ranked separately, with ties bolded. The no-verification ablation accepts all SALAD loop candidates.

TABLE III  
LOOP VERIFICATION AND TRAJECTORY RESULTS ON LONG SCANNET SEQUENCES
<table><tr><td>Scene</td><td>Frames Loops False</td><td></td><td></td><td> $\mathbf { A T E } _ { b }$ </td><td> ${ \bf A T E } _ { a }$ </td></tr><tr><td>scene0059_00</td><td>1787</td><td>7</td><td>0</td><td>0.091</td><td>0.091</td></tr><tr><td>scene0000_00</td><td>5578</td><td>12</td><td>0</td><td>0.140</td><td>0.140</td></tr><tr><td>scene0106_00</td><td>2324</td><td>4</td><td>0</td><td>0.121</td><td>0.121</td></tr><tr><td>scene0207_00</td><td>1988</td><td>8</td><td>2</td><td>0.109×</td><td>0.089</td></tr></table>

Note: $\mathrm { A T E } _ { b }$ and $\mathrm { A T E } _ { a }$ denote absolute trajectory error (m) before and after loop verification, respectively. Before verification, all SALAD candidates are accepted; × marks false-loop-induced drift. Lower is better.

![](images/ee7a9b9f9ca37f5bc52570782a15d615b7a1e8eef130187f3a9effba0936894a.jpg)

The trajectory correction achieved by loop verification is shown in Fig. 5. False constraints introduce mutually incompatible geometric relations in the pose graph and pull estimated poses away from ground truth. Together with the ATE values in Tables II and III, the trajectory plots show that filtering false-positive loops removes conflicting constraints and improves agreement with ground truth.

## C. Geometry-Anchored Persistent Instance Mapping

Table IV compares instance identity consistency on Scan-Net with SAM3, CLIP, and OVO baselines [4], [22], [23]. OVO is evaluated using both ground-truth poses and depth (GTp+GTd) and poses and depth predicted by the VGGT front end $( \mathrm { V p } + \mathrm { V d } )$

Among the external baselines, SAM3 Tracking, which relies entirely on temporal mask propagation, has low longterm reidentification performance under our test configuration. Its ReID-R of zero indicates that none of the evaluated instances retained a matching identity across the designated revisits. CLIP Association, which relies on 2D appearance features, also shows weak revisit identity scores and a large number of ID switches. These results illustrate the difficulty of maintaining global identity under viewpoint and occlusion changes using only temporal 2D tracking or appearance.

![](images/0205c71ef9b21bca2e8f1a39068cfc935f4fd8a95b14e695ae4ba9abde195f58.jpg)  
Fig. 5. Trajectories before and after loop verification: (a) fr1/room; (b) ScanNet scene0207 00. Gray: ground truth; red: optimized trajectory without removing false loops; blue: trajectory with the proposed loop verification.

TABLE IV  
PERSISTENT INSTANCE IDENTITY COMPARISON ON LONG SCANNET SEQUENCES
<table><tr><td>Method</td><td>IDC ↑</td><td>ReID-R ↑</td><td>Inst. Coverage ↑</td><td>MOTA ↑</td><td>4D-PQ ↑</td><td>IDS↓</td></tr><tr><td>SAM3 Tracking</td><td>0.146</td><td>0.000</td><td>0.223</td><td>-0.184</td><td>0.118</td><td>285</td></tr><tr><td>CLIP Association</td><td>0.189</td><td>0.058</td><td>0.594</td><td>-6.671</td><td>0.244</td><td>3263</td></tr><tr><td>OVO (Vp+Vd)</td><td>0.439</td><td>0.183</td><td>0.975</td><td>-13.019</td><td>0.026</td><td>2994</td></tr><tr><td>V3DSAM (Ours)</td><td>0.565</td><td>0.192</td><td>0.736</td><td>-0.286</td><td>0.110</td><td>873</td></tr><tr><td>OVO (GTp+GTd)</td><td>0.631</td><td>0.317</td><td>0.918</td><td>-4.603</td><td>0.046</td><td>1488</td></tr><tr><td>V3DSAM (GTp+GTd)</td><td>0.745</td><td>0.481</td><td>0.903</td><td>-2.805</td><td>0.054</td><td>901</td></tr></table>

Note: Vp/Vd: VGGT-estimated pose/depth; GTp/GTd: ground-truth pose/depth. Bold denotes the best result within each input setting; the non-GT (first four) and GT (last two) rows are ranked separately. ↑/↓ indicate preferred directions.

Two conclusions follow from comparing V3DSAM under different geometric inputs. First, feed-forward geometry quality substantially affects instance identity consistency. Replacing the predicted geometry with ground truth markedly improves the targeted identity metrics, indicating that geometric noise is a major bottleneck for persistent instance mapping. Second, when both methods use the same groundtruth geometry, V3DSAM outperforms OVO (GTp+GTd) on IDC and ReID-R, as well as MOTA and 4D-PQ. This ablation shows that OVO benefits substantially from ground-truth input geometry, while, after controlling for input quality, our geometry-anchored partitioning method exhibits stronger cross-view instance association.

With ground-truth geometry, V3DSAM’s IDC and ReID-R increase substantially, whereas MOTA and 4D-PQ decrease. This indicates a complex relationship among geometric input quality, instance partitioning and matching, and composite metrics. More complete depth observations may introduce additional predicted regions, affecting false-positive and segmentation matching statistics. Multiple metrics must therefore be considered together when evaluating instance identity quality.

To isolate the effect of the class constraint, we remove the requirement of semantic-class agreement from the 3D adja cency graph while retaining all other mapping parameters. Across four ScanNet scenes, removing the class constraint reduces IDC from 0.565 to 0.220, instance coverage from

![](images/da83aa53f017ca24851580485a8ec1e8a794b93e1daf19dcdd18b61cf81a8372.jpg)  
Fig. 6. Cross-view visualization of persistent instance identities on long ScanNet sequences. The first three columns show consecutive keyframes and the last two show revisit frames. Different colors denote different global instance IDs. From top to bottom: SAM3 Tracking, CLIP Association, OVO, and V3DSAM.

0.736 to 0.236, and 4D-PQ from 0.110 to 0.018. Geometric connected components alone can incorrectly merge nearby objects of different categories, causing severe undersegmentation. These results demonstrate that the class constraint is important for preserving the discriminability of global instances.

Fig. 6 compares persistent instance identities across consecutive keyframes and long-term revisits on ScanNet. V3DSAM retains the same physical entities’ identity labels more consistently across revisits, while the three baselines exhibit varying degrees of ID switches, incorrect matching, or excessive instance fragmentation. Together with Table IV, the results indicate that OVO obtains relatively high instance coverage but weaker long-term identity metrics; higher detection coverage does not necessarily imply more consistent cross-view instance identity.

## V. CONCLUSION

This paper presents V3DSAM, an open-vocabulary semantic mapping system for feed-forward dense SLAM with two complementary association mechanisms: semantic-geometric dual-scale loop verification and geometry-anchored persistent instance mapping. The first mechanism identifies falsepositive loops using only explicit geometry and semantic outputs, without depending on backbone-internal features. The second assigns global instance IDs to class-constrained 3D connected components and avoids inter-frame temporal ID propagation. Experiments on TUM RGB-D and ScanNet demonstrate that V3DSAM suppresses trajectory drift from false loops, achieves localization accuracy comparable to native VGGT verification, and improves long-term instance identity consistency relative to the evaluated baselines.

## REFERENCES

[1] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 5294–5306.

[2] A. Kirillov et al., “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4015–4026.

[3] N. Ravi et al., “SAM 2: Segment anything in images and videos,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[4] N. Carion et al., “SAM 3: Segment anything with concepts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[5] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 139, 2021, pp. 8748–8763.

[6] Z. Teed and J. Deng, “DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021.

[7] C. Campos, R. Elvira, J. J. Gomez Rodr ´ ´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: An accurate open-source library for visual,´ visual–inertial, and multimap SLAM,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890, 2021, doi: 10.1109/TRO.2021.3075644.

[8] A. Rosinol, M. Abate, Y. Chang, and L. Carlone, “Kimera: An opensource library for real-time metric-semantic localization and mapping,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2020, pp. 1689–1696, doi: 10.1109/ICRA40945.2020.9196885.

[9] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D vision made easy,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 20697–20709, doi: 10.1109/CVPR52733.2024.01956.

[10] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in Computer Vision – ECCV 2024, 2025, pp. 71–91, doi: 10.1007/978-3-031-73220-1 5.

[11] D. Maggio, H. Lim, and L. Carlone, “VGGT-SLAM: Dense RGB SLAM optimized on the SL(4) manifold,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, doi: 10.52202/085713-4324.

[12] D. R. Maggio and L. Carlone, “VGGT-SLAM 2.0: Real-time dense feed-forward scene reconstruction,” in Proc. Robot.: Sci. Syst. (RSS), 2026, doi: 10.15607/RSS.2026.XXII.051.

[13] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Realtime dense SLAM with 3D reconstruction priors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 16695–16705.

[14] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic,´ “NetVLAD: CNN architecture for weakly supervised place recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 5297–5307.

[15] S. Izquierdo and J. Civera, “Optimal transport aggregation for visual place recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 17658–17668.

[16] N. Hughes, Y. Chang, and L. Carlone, “Hydra: A real-time spatial perception system for 3D scene graph construction and optimization,” in Proc. Robot.: Sci. Syst. (RSS), 2022, doi: 10.15607/RSS.2022.XVIII.050.

[17] P. J. Besl and N. D. McKay, “A method for registration of 3-D shapes,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 14, no. 2, pp. 239–256, 1992, doi: 10.1109/34.121791.

[18] L. Hu, N. A. Oufroukh, F. Bonardi, and R. Ghandour, “EC3R-SLAM: Efficient and consistent monocular dense SLAM with feed-forward 3D reconstruction,” arXiv:2510.02080, 2025.

[19] M. Runz, M. Buffier, and L. Agapito, “MaskFusion: Real-time recog- ¨ nition, tracking and reconstruction of multiple moving objects,” in Proc. IEEE Int. Symp. Mixed Augmented Reality (ISMAR), 2018, pp. 10–20, doi: 10.1109/ISMAR.2018.00024.

[20] G. Narita, T. Seno, T. Ishikawa, and Y. Kaji, “PanopticFusion: Online volumetric semantic mapping at the level of stuff and things,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019, pp. 4205–4212, doi: 10.1109/IROS40897.2019.8967890.

[21] M. Li et al., “SGS-SLAM: Semantic Gaussian splatting for neural dense SLAM,” in Computer Vision – ECCV 2024, 2024, pp. 163–179, doi: 10.1007/978-3-031-72751-1 10.

[22] N. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, and A. Szlam, “CLIP-Fields: Weakly supervised semantic fields for robotic memory,” in Proc. Robot.: Sci. Syst. (RSS), 2023, doi: 10.15607/RSS.2023.XIX.074.

[23] T. B. Martins, M. R. Oswald, and J. Civera, “Open-vocabulary online semantic mapping for SLAM,” IEEE Robot. Autom. Lett., vol. 10, no. 11, pp. 11745–11752, 2025, doi: 10.1109/LRA.2025.3617736.

[24] A. Gelencser-Horv ´ ath, G. Dinya, D. B. Er ´ os, P. Hal ˝ asz, I. M.´ Muqsit, and K. Karacs, “SceneVGGT: VGGT-based online 3D semantic SLAM for indoor scene understanding and navigation,” arXiv:2602.15899, 2026.

[25] T. Xiao et al., “IRIS-SLAM: Unified geo-instance representations for robust semantic localization and mapping,” arXiv:2602.18709, 2026.

[26] P. Nguyen et al., “Open3DIS: Open-vocabulary 3D instance segmentation with 2D mask guidance,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 4018–4028.

[27] M. Yan, J. Zhang, Y. Zhu, and H. Wang, “MaskClustering: View consensus based mask graph clustering for open-vocabulary 3D instance segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 28274–28284.

[28] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “ScanNet: Richly-annotated 3D reconstructions of indoor scenes,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 5828–5839.

[29] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012, pp. 573–580, doi: 10.1109/IROS.2012.6385773.

[30] Q. Zhang, A. Mai, J. Menke, and A. Yang, “Loop closure detection with RGB-D feature pyramid Siamese networks,” arXiv:1811.09938, 2018.

[31] K. Bernardin and R. Stiefelhagen, “Evaluating multiple object tracking performance: The CLEAR MOT metrics,” EURASIP J. Image Video Process., vol. 2008, Art. no. 246309, 2008, doi: 10.1155/2008/246309.

[32] M. Aygun et al., “4D panoptic LiDAR segmentation,” in ¨ Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 5527–5537.