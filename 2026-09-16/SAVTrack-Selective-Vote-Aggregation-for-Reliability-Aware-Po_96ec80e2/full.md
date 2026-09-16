# SAVTrack: Selective Vote Aggregation for Reliability-Aware Point Cloud Tracking

Sifan Zhou<sup>†∗a,b</sup>, Linyue Tan<sup>†c</sup>, Qiwei Wang<sup>†d</sup>, Ziyu Zhao<sup>a,b</sup> and Xiaobo Lu<sup>a,b,∗</sup>

<sup>a</sup>School ofAutomation, Southeast University, Nanjing, China

<sup>b</sup>Key Laboratory of Measurement and Control of Complex Systems of Engineering, Ministry of Education, Nanjing, China

<sup>c</sup>University of Pennsylvania, Philadelphia, PA, USA

<sup>d</sup>Harbin Institute of Technology (Shenzhen), Shenzhen, China

## A R T I C L E I N F O

Keywords: 3D single object tracking LiDAR point clouds Vote aggregation Reliability-aware selection Point cloud tracking

## A BS T RA C T

3D single object tracking (SOT) in LiDAR point clouds is essential for autonomous systems, but remains challenging under sparse and incomplete observations. In such cases, different target points provide highly uneven constraints on the object center, causing some point-to-center votes to be substantially less reliable than others. Existing point-based trackers typically aggregate these hypotheses without explicitly modeling their reliability, allowing inaccurate votes to contaminate proposal clustering and degrade localization accuracy. To address this issue, we propose SAVTrack, a motion-aware tracking framework with Selective Vote Aggregation (SAV). SAVTrack estimates the reliability of each candidate vote from both local seed features and inter-frame motion context, and removes low-confidence hypotheses before proposal clustering. This pre-aggregation gating prevents unreliable hypotheses from affecting cluster formation while introducing only modest computational overhead. SAVTrack achieves competitive performance on KITTI and nuScenes, reaching 68.4/87.4 and 58.44/69.82 Success/Precision, respectively, while running at 82 FPS. It retains fewer than onesixth of the candidate votes used by dense aggregation and remains particularly effective under sparse target observations.

## 1. Introduction

LiDAR point clouds have wide applications in various computer vision tasks [48, 47, 32, 44, 31, 38]. Among them, 3D Single Object Tracking (SOT) in LiDAR point clouds is a fundamental perception task for autonomous systems and robotics [4, 46, 7, 43, 2, 16, 12], requiring accurate target localization across frames under sparse observations, occlusion, and background clutter. Existing 3D SOT approaches can be broadly divided into two paradigms. Siamese-based appearance matching trackers [28, 30, 29] localize the target by matching template and search features, benefiting from explicit target appearance correspondence but often becoming vulnerable when point observations are sparse, incomplete, or contaminated by distractors. Motioncentric trackers [40, 41, 25], in contrast, estimate the target state from inter-frame motion cues, reducing their dependence on appearance matching and offering an effective alternative under appearance ambiguity. Orthogonal to this paradigm distinction, point-based trackers can preserve finegrained local evidence throughout localization rather than directly collapsing the entire target observation into a single prediction. In particular, point-wise voting allows individual seed<sup>1</sup> points to cast explicit hypotheses toward the target center, making the contribution of each local observation directly accessible during proposal generation. This finegrained formulation provides a natural basis for exploiting local geometric evidence in sparse LiDAR tracking, while also exposing an important question: do all point-to-center hypotheses contribute equally reliable localization cues?

In practice, point-to-center hypotheses can exhibit substantially different localization reliability. As shown in Fig. 1, under sparse and incomplete observations, seed points from different target regions provide unequal geometric constraints on the object center. Specifically, points on large and locally planar surfaces often offer weak geometric cues, which can lead to ambiguous center estimates, whereas structurally distinctive regions, such as surface intersections or corner areas with larger local shape variation, tend to provide stronger localization evidence. Similar observations have also been reported in 3D object detection [6, 18]. However, this reliability heterogeneity is largely ignored in conventional voting-based pipelines, which typically pass all seed-generated hypotheses to the subsequent clustering stage without explicitly distinguishing their quality. Consequently, inaccurate votes from ambiguous regions can interfere with neighborhood construction and proposal formation, ultimately degrading target localization. This observation motivates us to explicitly model seed-level vote reliability before proposal aggregation.

Although recent 3D tracking methods have substantially advanced feature representation, target correspondence, and motion estimation [15, 24, 22, 40, 25, 45], the reliability of individual seed-to-center hypotheses before proposal aggregation remains comparatively underexplored. Selective voting has been studied in the related setting of single-frame 3D object detection. In particular, SPOT [6] demonstrates that point-to-center hypotheses exhibit heterogeneous localization quality and improves proposal generation by suppressing unreliable votes. 3D SOT, however, introduces a different localization setting: the target is specified across consecutive frames, and the quality of a candidate vote can depend on both local geometric evidence and the temporal motion context of the tracked object. This distinction suggests that vote reliability in 3D tracking should be considered in a target-conditioned temporal context rather than solely from static local geometry [6]. However, how to explicitly estimate and exploit such reliability before proposal aggregation remains largely unexplored in existing point-cloud tracking frameworks.

![](images/7e9152267b6a6f2a1865874b2c94eb7dceae4ad1a8aa8ea2e15a389d7d25d127.jpg)  
Figure 1: Illustration of heterogeneous vote reliability across objects based on point clouds. Points on locally planar or ambiguous regions often produce noisy center hypotheses due to weak geometric constraint, while structurally distinctive regions provide more reliable localization and geometric cues. Filtering low-confidence votes before clustering leads to a more concentrated vote distribution and more accurate proposal formation.

To address this gap, we propose SAVTrack, a motioncentric 3D tracking framework built around Selective Vote Aggregation (SAV). SAV predicts a per-seed posterior over predefined center-relative sub-regions from fused pointwise and inter-frame motion representations. This posterior serves two purposes: (i) it provides a directly supervised estimate of the center-relative region for each seed; and (ii) the posterior probability associated with each candidate provides a confidence score for filtering unreliable votes before proposal clustering. Unlike soft posterior weighting, which retains all hypotheses during aggregation, SAV applies hard pre-aggregation gating to remove low-confidence votes from the candidate set, preventing unreliable hypotheses from affecting local neighborhood construction and subsequent proposal formation. The resulting module is lightweight, consisting of a small MLP classifier and region-specific vote regressors, and preserves real-time inference. We further validate the learned confidence through aggregation-strategy comparisons and geometry–reliability analysis in Sec. 4. Overall, our contributions are as follows:

• We identify and systematically study seed-level vote reliability in 3D single object tracking, showing that point-to-center hypotheses exhibit substantial reliability differences with respect to local geometry and observation sparsity.

• We propose Selective Vote Aggregation (SAV), a lightweight motion-conditioned gating mechanism that estimates vote reliability from joint point-wise and inter-frame motion representations.

• Extensive experiments on two widely used benchmarks, KITTI and nuScenes, demonstrate that SAV-Track achieves competitive tracking accuracy at 82 FPS while retaining fewer than one-sixth of the candidate votes used by dense aggregation.

## 2. Related Work

Siamese-based 3D Single Object Tracking. Early 3D SOT methods formulate tracking as a template–search appearance matching problem using Siamese networks. As a pioneer, SC3D [10] introduces the first Siamese framework that extracts features from a template and candidate regions and selects the best match via feature distance. Subsequently, 3D-SiamRPN [8] extends this paradigm with a region proposal network for 3D localization. P2B [28] integrates VoteNet-style point-wise voting into the Siamese framework to generate target proposals in an end-to-end manner, inspiring a series of follow-up works [30, 39, 3, 11, 15, 23, 24, 13, 46]. For instance, BAT [39] encodes object size priors to augment template–search correlation; PTT [30, 29], LTTR [3], CMT [11], and STNet [15] explore transformer- or attention-based feature propagation and correlation; CXTrack [36] emphasizes contextual modeling with a target-centric transformer; and MBPTrack [37] introduces an external memory to enhance spatial–temporal information aggregation. OSP2B [23] revisits the point-to-box localization stage and replaces the conventional two-stage proposal generation and scoring process with a one-stage formulation, jointly predicting 3D proposals and center-ness scores while introducing a target classifier to suppress interference proposals. Despite these advances, existing Siamese trackers mainly improve template–search correspondence, feature interaction, or proposal-level scoring, while the reliability of individual seed-to-center hypotheses before vote aggregation remains largely unmodeled. Moreover, sparse and incomplete point observations continue to pose challenges to appearance-based matching.

![](images/aef40dfa290810d0c0139e4baaeab3aa92a21052b0ec9382fd17f5ca23baa90a.jpg)  
Figure 2: Overview of SAVTrack. Given the target observation from the previous frame and the current search region, a weightshared PointNet++ [27] backbone extracts point-wise features, which are fused through the P2P-derived part-to-part motion representation to construct motion-aware seed features. The proposed Selective Vote Aggregation (SAV) module jointly performs center-relative sub-region posterior estimation and region-specific vote regression to generate multiple candidate center hypotheses for each seed. The posterior probability associated with each candidate is used as a proxy for vote reliability, and low-confidence hypotheses are removed by hard pre-aggregation gating before proposal formation. The retained reliable votes are then grouped and aggregated to generate target proposals, from which the final 3D target box is predicted in an end-to-end manner.

Motion-based 3D Single Object Tracking. Motion-based trackers formulate 3D SOT as inter-frame motion estimation rather than template–search appearance matching. M<sup>2</sup>Track [40] first segments foreground points in the search region and then infers the target’s 4-DOF relative motion offset; M<sup>2</sup>Track++ [41] further extends this paradigm to semi-supervised settings. P2P [25] demonstrates the effectiveness of a motion-centric formulation by transforming appearance matching into part-to-part relative offset estimation between adjacent frames. More recent works explore additional motion cues: VoxelTrack [19] exploits multi-level voxel representations for 3D spatial reasoning; DMT [35] introduces a motion prediction module that estimates the target center from historical bounding boxes and refines it with point features. These approaches improve foreground motion modeling, whereas SAV addresses a complementary question in our voting formulation: whether a candidate seed-to-center hypothesis should enter clustering at all. SAV therefore estimates a directly supervised per-vote posterior and applies the decision before proposal clustering.

## 3. Method

## 3.1. 3D SOT Task Definition

Given an initial target specified by its ground-truth 3D bounding box in the first frame, 3D SOT aims to sequentially estimate the target state in each subsequent LiDAR frame. At frame �, following the standard tracking-bycropping protocol [29, 40, 25], a search region is extracted from the current LiDAR frame around the target location estimated in the previous frame. Specifically, the previous prediction $\mathbf { B } _ { t - 1 }$ is spatially enlarged by a predefined margin to form a search crop, from which the current search point cloud $\mathcal { P } _ { t } = \{ \mathbf { p } _ { i } ^ { t } = [ x _ { j } , y _ { j } , z _ { j } , r _ { j } ] ^ { \top } \in \mathbb { R } ^ { 4 } \}$ is obtained. The corresponding target observation from the previous frame is denoted as $\mathcal { P } _ { t - 1 } = \{ \mathbf { p } _ { i } ^ { t - 1 } = [ x _ { i } , y _ { i } , z _ { i } , r _ { i } ] ^ { \top } \in \mathbb { R } ^ { 4 } \}$ , where (�, �, �) are the point coordinates and � denotes the LiDAR reflectance. Given the template point cloud $\mathcal { P } _ { t - 1 }$ and search point cloud $\mathcal { P } _ { t }$ , the tracker predicts the current 3D bounding box $\mathbf { B } _ { t } = [ x , y , z , h , w , l , \theta ] ^ { \top } \in \mathbb { R } ^ { 7 }$ , where (�, �, �) denote the box center, (ℎ, �, �) represent its dimensions, and � is the heading angle. Following common practice in 3D SOT [28, 30, 39, 25], the target size is assumed to remain unchanged throughout a sequence. Therefore, tracking reduces to estimating the target center and heading angle at each frame.

## 3.2. SAVTrack Framework Overview

Fig. 2 illustrates the overall architecture of proposed SAVTrack. We adopt P2P-point [25] as our motion-centric baseline, which models inter-frame target dynamics through part-to-part feature interaction between consecutive point clouds. Its learned motion representation captures target displacement across frames and provides the temporal context for current-frame localization. To enable point-wise proposal generation, we employ a standard voting-based localization head on top of the P2P-point [25] representation. Our proposed Selective Vote Aggregation (SAV) is then introduced before proposal clustering to explicitly estimate candidate-vote reliability and suppress unreliable center hypotheses.

Motion-aware Seed Representation. Given two cropped point clouds from consecutive frames, $\mathcal { P } _ { t - 1 } ^ { \mathrm { c r o p } }$ and $\mathscr { P } _ { t } ^ { \mathrm { c r o p } }$ a weight-shared PointNet++ [27] encoder extracts pointwise features $\mathbf { f } _ { t - 1 } , \mathbf { f } _ { t } \ \in \ \mathbb { R } ^ { N \times C }$ . Following the part-to-part motion modeling of P2P [25], the features from the two frames are globally aggregated and fused to obtain an interframe motion representation $\mathbf { F } _ { \mathrm { m } }$ that characterizes the target dynamics between adjacent observations.

![](images/6811b367a9df3fa0127cd4ee87121dc8eb912b7d0970e8b74cef063fc861992d.jpg)  
Figure 3: Illustration of the Selective Vote Aggregation (SAV) module. For each seed point, SAV partitions the center-relative space into $N _ { R }$ pre-defined sub-regions and learns a posterior distribution over which sub-region contains the true object center. The posterior confidence is used as a proxy for candidate-vote reliability: votes with high posterior probability are retained, while low-confidence votes are excluded before proposal clustering

<sup>erception</sup> <sup>an</sup>We sample $N _ { s }$ <sup>utonomous-navigation</sup> seed points from the current-frame point cloud using farthest point sampling (FPS) and retain their corresponding point-wise features. The motion representation $\mathbf { F } _ { \mathrm { m } }$ is then broadcast to all seeds and concatenated with each seed feature:

$$
\tilde { \mathbf { f } } ^ { i } = [ \mathbf { f } ^ { i } ; \mathbf { F } _ { \mathrm { m } } ] , \qquad i = 1 , \ldots , N _ { s } ,\tag{1}
$$

where $\tilde { \mathbf { f } } ^ { i }$ denotes the motion-aware representation of the �-th seed. This representation preserves local point-wise evidence while incorporating temporal target dynamics, and is subsequently used by SAV for candidate-vote confidence estimation and center hypothesis generation.

## 3.3. Selective Vote Aggregation

Building on the motion-aware seed representation, we introduce Selective Vote Aggregation (SAV) to explicitly distinguish candidate center hypotheses before proposal clustering. Selective voting has previously been explored in single-frame 3D object detection [6]; in contrast, SAV estimates candidate confidence from a target-conditioned representation that jointly incorporates current-frame pointwise evidence and inter-frame motion context. Specifically, each seed predicts a posterior over predefined center-relative sub-regions and generates a corresponding set of candidate center hypotheses. The posterior confidence is then used for hard pre-aggregation gating, such that low-confidence hypotheses are removed before participating in proposal clustering.

Center-relative Sub-region Posterior Estimation. For each seed �, we consider the relative displacement between its coordinate $\mathbf { z } ^ { i }$ and the ground-truth target center �. The center-relative displacement space is partitioned into $N _ { R }$ predefined sub-regions $\{ r _ { j } \} _ { j = 1 } ^ { N _ { R } }$ , as illustrated in Fig. 3. Accordingly, the ground-truth region label of seed � is defined as

$$
y ^ { i } = \mathcal { Q } \left( \mathbf { c } - \mathbf { z } ^ { i } \right) , \qquad y ^ { i } \in \{ 1 , \dots , N _ { R } \} ,\tag{2}
$$

where (⋅) denotes the predefined region-assignment operator.

Given the motion-aware seed feature $\tilde { \mathbf { f } } ^ { i }$ , a lightweight classifier $g ( \cdot ; \theta ^ { g } )$ predicts a categorical posterior over the $N _ { R }$ center-relative sub-regions:

$$
\mathbf { p } ^ { i } = \mathrm { S o f t m a x } \left( g ( \tilde { \mathbf { f } } ^ { i } ; \theta ^ { g } ) \right) ,\tag{3}
$$

where

$$
p _ { j } ^ { i } = P \left( y ^ { i } = j \mid \tilde { \mathbf { f } } ^ { i } \right)\tag{4}
$$

denotes the posterior probability that the target center belongs to sub-region � relative to seed �. The classifier is supervised by the ground-truth sub-region labels using cross-entropy loss:

$$
L _ { \mathrm { v o t e - c l s } } = - \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } \log p _ { y ^ { i } } ^ { i } .\tag{5}
$$

Note that the classifier is supervised to predict the centerrelative sub-region, rather than an explicit reliability label. The resulting posterior can nevertheless provide a confidence measure for each candidate hypothesis, reflecting how strongly the learned motion-aware representation supports the corresponding center-relative configuration. We therefore use the posterior confidence as a proxy for vote reliability during aggregation.

Vote Regression and Reliability Gating. Concurrent with confidence estimation, each seed point generates candidate center votes for all $N _ { R }$ sub-regions. Specifically, for each sub-region $j ,$ a region-specific MLP regressor $\phi _ { j } ( \cdot ; \theta _ { j } ^ { \phi } )$ takes the seed feature $\tilde { \mathbf { f } } ^ { i }$ and predicts the offset from $\mathbf { z } ^ { i }$ to the object center:

$$
\Delta \mathbf { z } _ { j } ^ { i } = \phi _ { j } ( \tilde { \mathbf { f } } ^ { i } ; \boldsymbol { \theta } _ { j } ^ { \phi } ) , \quad \hat { \mathbf { d } } _ { j } ^ { i } = \mathbf { z } ^ { i } + \Delta \mathbf { z } _ { j } ^ { i } ,\tag{6}
$$

yielding a predicted center location $\hat { \mathbf { d } } _ { i } ^ { i }$ . The regressor is supervised by the ground-truth offset $\Delta \mathbf { z } _ { \ast } ^ { i } = \mathbf { c } - \mathbf { z } ^ { i }$ using an $L _ { 1 }$ loss applied only to the correct sub-region:

$$
L _ { \mathrm { v o t e - r e g } } = \sum _ { j = 1 } ^ { N _ { R } } \sum _ { i : y ^ { i } = j } \| \phi _ { j } ( \widetilde { \mathbf { f } } ^ { i } ; \boldsymbol { \theta } _ { j } ^ { \phi } ) - \Delta \mathbf { z } _ { * } ^ { i } \| _ { 1 } .\tag{7}
$$

At inference time, each seed generates $N _ { R }$ candidate center hypotheses, $\{ v _ { j } ^ { i } \} _ { j = 1 } ^ { N _ { R } }$ , where $v _ { j } ^ { i } = ( \hat { \mathbf { d } } _ { j } ^ { i } , p _ { j } ^ { i } )$ . The posterior $p _ { j } ^ { i }$ provides a confidence measure for the corresponding candidate and is used as a proxy for vote reliability. SAV performs hard pre-aggregation gating:

$$
\mathfrak { P } _ { \mathrm { S A V } } = \left\{ v _ { j } ^ { i } \mid p _ { j } ^ { i } > \tau \right\} .\tag{8}
$$

Unlike confidence reweighting, which preserves all hypotheses in the aggregation set, hard gating changes the candidate set itself and prevents discarded votes from affecting neighborhood construction during proposal clustering.

The total vote loss is:

$$
L _ { \mathrm { v o t e } } = L _ { \mathrm { v o t e - c l s } } + \lambda _ { \mathrm { r e g } } L _ { \mathrm { v o t e - r e g } } ,\tag{9}
$$

where $\lambda _ { \mathrm { r e g } }$ is a balancing hyper-parameter. Algorithm 1 summarizes the complete inference pipeline of SAVTrack.

## 3.4. Proposal Generation and Training

Given the filtered high-confidence vote set ${ \mathscr D } _ { \mathrm { S A V } } .$ , we generate target proposals through vote clustering and proposal refinement. Let $M = | \mathcal { V } _ { \mathrm { S A V } } |$ denote the number of retained votes after selective gating. Following the VoteNetstyle aggregation paradigm [26], a set of representative vote centers is first sampled from ${ \mathcal { V } } _ { \mathrm { S A V } }$ as proposal seeds. For each proposal seed, neighboring votes within a predefined spatial radius are grouped by ball query to construct a local vote cluster:

$$
\mathcal { T } _ { k } = \left. v _ { j } ^ { i } \in \mathcal { V } _ { \mathrm { S A V } } \middle | \left\| \hat { \mathbf { d } } _ { j } ^ { i } - \mathbf { q } _ { k } \right\| _ { 2 } < r _ { p } \right. ,\tag{10}
$$

where $\mathbf { q } _ { k }$ denotes the center of the �-th proposal seed and $r _ { p }$ is the grouping radius. Since unreliable hypotheses have already been removed by SAV, the resulting clusters are constructed only from the retained high-confidence center hypotheses, thereby reducing the influence of ambiguous votes on local proposal formation.

For each vote cluster $\tau _ { k }$ , the spatial coordinates and associated vote features are aggregated to form a clusterlevel representation. A lightweight proposal head Θ(⋅) then predicts the target state $\mathbf { b } _ { k }$ with an objectness score $s _ { k } \colon$

$$
( \mathbf { b } _ { k } , s _ { k } ) = \Theta ( T _ { k } ) .\tag{11}
$$

The proposal state is parameterized as:

$$
\mathbf { b } _ { k } = [ x _ { k } , y _ { k } , z _ { k } , \theta _ { k } ] ^ { \top } ,\tag{12}
$$

where $( x _ { k } , y _ { k } , z _ { k } )$ denote the predicted target center and $\theta _ { k }$ is the heading angle. Following the standard 3D SOT setting, the target dimensions are inherited from the initialized bounding box and remain fixed throughout the sequence. The objectness score $s _ { k }$ measures the likelihood that the corresponding proposal accurately represents the tracked target. After all candidate proposals are generated, the proposal with the highest valid objectness score is selected as the predicted target state $\mathbf { B } _ { t }$ for the current frame.

Algorithm 1 SAVTrack Inference Pipeline   
Input: Template crop $\mathcal { P } _ { t - 1 } ^ { \mathrm { c r o p } }$ , search crop $\mathcal { P } _ { t } ^ { \mathrm { c r o p } }$ , previous   
box $\mathbf { B } _ { t - 1 } ;$ hyper-parameters: $N _ { R } , \tau , N _ { \mathrm { p r o p } } , r _ { p }$   
Output: Predicted bounding box   
$\mathbf { B } _ { t }$   
1: // Feature encoding   
2: Extract per-point features $\mathbf { f } _ { t - 1 } , \mathbf { f } _ { t }$ via weight-shared   
PointNet++   
3: Compute global features $\mathbf { F } _ { t - 1 } , \mathbf { F } _ { t }$ via max-pooling   
4: Fuse motion feature $\mathbf { F } _ { \mathrm { p p } } ^ { \mathrm { f u s i o n } }$ via cascaded 1D convolu  
tions   
5: Sample $N _ { s }$ seed points via FPS; obtain motion-aware   
seed features $\tilde { \mathbf { f } } ^ { i }$   
6: // Selective Vote Aggregation (SAV)   
7: Predict sub-region posteriors $[ p _ { 1 } ^ { i } , \dots , p _ { N _ { R } } ^ { i } ] = g ( \tilde { \mathbf { f } } ^ { i } )$ for   
each seed   
8: Regress per-region vote offsets $\Delta \mathbf { z } _ { j } ^ { i } = \phi _ { j } ( \tilde { \mathbf { f } } ^ { i } )$ and vote   
centers $\hat { \mathbf { d } } _ { j } ^ { i }$   
9: Collect filtered votes $\mathcal { V } _ { \mathrm { f i l t e r e d } } = \{ v _ { j } ^ { i } : p _ { j } ^ { i } > \tau \}$ {Hard   
reliability gating}   
10: if $| \mathcal { V } _ { \mathrm { f i l t e r e d } } | = 0$ then   
11: Fall back to $\mathrm { t o p } { \cdot } K _ { 0 }$ posterior-ranked votes before   
thresholding {Empty-vote fallback}   
12: end if   
13: // Proposal generation and tracking   
14: Select $K = \operatorname* { m i n } ( N _ { \mathrm { p r o p } } , | \mathcal { V } _ { \mathrm { f i l t e r e d } } | )$ cluster centroids from   
$\tt N _ { \mathrm { f i l t e r e d } }$   
15: for each centroid do   
16: Gather neighboring votes within radius $r _ { p }$ via ball   
query   
17: Predict proposal box $\mathbf { b } _ { k }$ and objectness score $s _ { k }$ via   
MLP Θ   
18: end for   
19: return $\mathbf { B } _ { t }$

Proposal Supervision. Following VoteNet-style proposal supervision [26], proposal candidates are assigned positive or negative labels according to their center distance to the ground-truth target center. The objectness score $s _ { k }$ is optimized using a binary classification loss $L _ { \mathrm { p r o p } } ,$ while the target state of positive proposals is supervised by a regression loss $L _ { \mathrm { { b o x } } }$

Combining the selective voting objective introduced in Sec. 3.3, the overall training objective of SAVTrack is

$$
L = L _ { \mathrm { v o t e } } + \gamma _ { 1 } L _ { \mathrm { p r o p } } + \gamma _ { 2 } L _ { \mathrm { b o x } } ,\tag{13}
$$

where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ balance proposal classification and box regression, respectively. All components are optimized jointly in an end-to-end manner.

Comparison with state-of-the-art methods on the KITTI dataset [9]. Success / Precision are reported for evaluation. Bold and underlined values denote the best and second-best results, respectively. Rep means the points’ view representation.
<table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Tracker</td><td rowspan="2">Source</td><td rowspan="2">Rep.</td><td rowspan="2">Mean (14,068)</td><td rowspan="2">Car (6,424)</td><td rowspan="2">Pedestrian (6,088)</td><td rowspan="2">Van (1,248)</td><td rowspan="2">Cyclist (308)</td><td rowspan="2">FPS</td><td rowspan="2">Device</td></tr><tr><td></td></tr><tr><td rowspan="9">Siamese</td><td>SC3D [10] P2B [28]</td><td>CVPR&#x27;19 CVPR&#x27;20</td><td>Point Point</td><td>31.2 / 48.5 42.4 / 60.0</td><td>41.3 / 57.9 56.2 / 72.8</td><td>18.2/ 37.8 28.7 / 49.6</td><td>40.4 / 47.0 40.8 / 48.4</td><td>41.5 /70.4 32.1 / 44.7</td><td>2 40</td><td>GTX 1080Ti GTX 1080Ti</td></tr><tr><td>3D-SiamRPN [8]</td><td>IEEE Sensors J</td><td>Point</td><td>46.6 / 64.9</td><td>58.2 / 76.2</td><td>35.2 / 56.2</td><td>45.7 /52.9</td><td>36.2 / 49.0</td><td>45</td><td>GTX 1080Ti</td></tr><tr><td>PTT [30]</td><td>IROS&#x27;21</td><td>Point</td><td>55.1 / 74.2</td><td>67.8 / 81.8</td><td>44.9 / 72.0</td><td>43.6 / 52.5</td><td>37.2 / 47.3</td><td>40</td><td>GTX 1080Ti</td></tr><tr><td>LTTR [3]</td><td>BMVC&#x27;21</td><td>BEV</td><td>48.7 / 65.8</td><td>65.0 / 77.1</td><td>33.2 / 56.8</td><td>35.8 / 45.6</td><td>66.2 / 89.9</td><td>23</td><td>GTX 1080Ti</td></tr><tr><td>BAT [39]</td><td>ICCV&#x27;21</td><td>Point</td><td>51.2/72.8</td><td>60.5 / 77.7</td><td>42.1 / 70.1</td><td>52.4 / 67.0</td><td>33.7 / 45.4</td><td>57</td><td>RTX 2080</td></tr><tr><td>V2B [14]</td><td>NeurlPS&#x27;21</td><td>Point+BEV</td><td>58.4 / 75.2</td><td>70.5/81.3</td><td>48.3 / 73.5</td><td>50.1 / 58.0</td><td>40.8 / 49.7</td><td>37</td><td>TITAN RTX</td></tr><tr><td>PTTR [42]</td><td>CVPR&#x27;22</td><td>Point</td><td>57.9 / 78.2</td><td>65.2 / 77.4</td><td>50.9 /81.6</td><td>52.5 /61.8</td><td>65.1 / 90.5</td><td>50</td><td>Tesla V100</td></tr><tr><td>STNet [15]</td><td>ECCV&#x27;22</td><td>Point</td><td>61.3 / 80.1</td><td>72.1 / 84.0</td><td>49.9 / 77.2</td><td>58.0 / 70.6</td><td>73.5/ 93.7</td><td>35</td><td>TITAN RTX</td></tr><tr><td>GLT-T [24]</td><td>AAAI&#x27;23</td><td>Point</td><td>60.1 / 79.3</td><td>68.2 / 82.1</td><td>52.4 / 78.8</td><td>52.6 / 62.9</td><td>68.9 /92.1</td><td>30</td><td>GTX 1080Ti</td></tr><tr><td>OSP2B [23]</td><td>IJCAI&#x27;23</td><td>Point</td><td>60.5 / 82.3</td><td>67.5 / 82.3</td><td>53.6 / 85.1</td><td>56.3 / 66.2</td><td>65.6 / 90.5</td><td>34</td><td>GTX 1080Ti</td></tr><tr><td>CXTrack [36]</td><td>CVPR&#x27;23</td><td>Point</td><td>67.5 / 85.3</td><td>69.1 / 81.6</td><td>67.0 / 91.5</td><td>60.0 / 71.8</td><td>74.2 / 94.3</td><td>34</td><td>RTX 3090</td></tr><tr><td>SyncTrack [21]</td><td>ICCV&#x27;23</td><td>Point</td><td>64.1 / 81.9</td><td>73.3 / 85.0</td><td>54.7 / 80.5</td><td>60.3 / 70.0</td><td>73.1/ 93.8</td><td>45</td><td>TITAN RTX</td></tr><tr><td>MoCUT [22]</td><td>ICLR&#x27;24</td><td>Point</td><td>65.8 / 85.0</td><td>67.6 / 80.5</td><td>63.3 / 90.0</td><td>64.5 / 78.8</td><td>76.7 / 94.2</td><td>48</td><td>RTX 3070Ti</td></tr><tr><td>CFTrack [16]</td><td>Neurocomputing&#x27;26</td><td>Point</td><td>59.7 / 79.3</td><td>71.4 / 83.2</td><td>49.8 / 80.0</td><td>52.1 /57.9</td><td>43.9 /54.5</td><td>45</td><td>RTX 4090</td></tr><tr><td rowspan="7">Motion</td><td>M²Track [40]</td><td>CVPR&#x27;22</td><td>Point</td><td>62.9 / 83.4</td><td>65.5 / 80.8</td><td>61.5 / 88.2</td><td>53.8 / 70.7</td><td>73.2 / 93.5</td><td>57</td><td>Tesla V100</td></tr><tr><td>M²Track++ [41]</td><td>TPAMI&#x27;23</td><td>Point</td><td>66.5 / 85.2</td><td>71.1 / 82.7</td><td>61.8 / 88.7</td><td>62.8 / 78.5</td><td>75.9 /94.0</td><td>57</td><td>Tesla V100</td></tr><tr><td>VoxelTrack [19]</td><td>ACM MM&#x27;24</td><td>Voxel</td><td>70.4 / 88.3</td><td>72.5 / 84.7</td><td>67.8 / 92.6</td><td>69.8 / 83.6</td><td></td><td>36</td><td>TITAN RTX</td></tr><tr><td>FocusTrack [45]</td><td>ACM MM&#x27;25</td><td>BEV</td><td>71.3 / 89.4</td><td>74.1 / 85.9</td><td>69.3 / 94.1</td><td>68.4 / 83.5</td><td>75.1 / 94.7 75.9 / 94.7</td><td>105</td><td></td></tr><tr><td>CompTrack [43]</td><td>AAAI&#x27;26</td><td>BEV</td><td>71.4 / 89.3</td><td>73.4 / 85.2</td><td>69.5/ 94.7</td><td>68.5 /82.5</td><td>76.0 / 94.8</td><td>90</td><td>RTX 3090 RTX 3090</td></tr><tr><td>P2P-voxel [25]</td><td>IJCV&#x27;25</td><td>Voxel</td><td>71.7 /89.4</td><td>73.6 / 85.7</td><td>69.6 / 94.0</td><td>70.3 / 83.9</td><td>75.5/94.6</td><td>71</td><td>RTX 3090</td></tr><tr><td>P2P-point [25]</td><td>IJCV&#x27;25</td><td>Point</td><td>66.2 / 85.4</td><td>68.8 /81.7</td><td>62.7 / 89.1</td><td>65.4 / 80.1</td><td>74.8 / 94.8</td><td>105</td><td>RTX 3090</td></tr><tr><td></td><td>SAVTrack (Ours)</td><td>Ours</td><td>Point</td><td>68.4 / 87.4</td><td>71.1 /83.9</td><td>65.1 / 91.8</td><td>68.0 / 82.5</td><td>75.9 / 94.8</td><td>82</td><td>RTX 3090</td></tr></table>

## 4. Experiments

Datasets and Evaluation Metrics. We evaluate SAVTrack on two widely used 3D single object tracking benchmarks, KITTI [9] and nuScenes [1]. KITTI is collected using a 64-beam LiDAR and contains four commonly evaluated categories, including Car, Pedestrian, Van, and Cyclist, while nuScenes is captured with a 32-beam LiDAR and provides substantially sparser target observations across Car, Pedestrian, Truck, Trailer, and Bus categories. We follow the standard one-pass evaluation (OPE) protocol [34] adopted by previous 3D SOT methods [28, 30, 39, 40, 25]. Tracking performance is evaluated using Success and Precision, where Success measures the overlap between the predicted and ground-truth 3D bounding boxes, while Precision evaluates the center localization error.

Implementation Details. All experiments are implemented in PyTorch and trained using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , a batch size of 128, and 30 training epochs. Unless otherwise specified, SAV uses $N _ { R } = 1 2$ center-relative sub-regions and a confidence threshold of $\tau = 0 . 3$ . The loss-balancing coefficients are set to $\lambda _ { \mathrm { r e g } } = 1 . 0 , \gamma _ { 1 } = 1 . 0 ,$ , and $\gamma _ { 2 } ~ = ~ 1 . 0$ . During proposal generation, at most $N _ { \mathrm { p r o p } } = 2 5 6$ proposal seeds are used, and neighboring votes are grouped with a radius of $r _ { p } \ =$ 0.3 m. Following the implementation of the voting baseline, proposals whose centers are sufficiently close to the ground-truth center are treated as positive, whereas distant proposals are treated as negative. Specifically, we use 0.3 m and 0.6 m as the positive and negative distance thresholds, respectively, and ignore proposals falling between the two thresholds. All ablation studies are conducted using the same P2P-based voting baseline described in Sec. 3.2, with SAV being the only additional vote-selection component unless otherwise stated. Training is performed on a single NVIDIA RTX 3090 GPU. Runtime measurements for SAVTrack and the corresponding baseline are conducted on the same hardware under an identical inference protocol to ensure a fair efficiency comparison.

## 4.1. Quantitative Experiment

Comparison on KITTI. Tab. 1 compares SAVTrack with representative Siamese-based and motion-centric 3D single object trackers on the KITTI benchmark. SAVTrack achieves an average Success/Precision of 68.4%/87.4%. Despite its lightweight point-based representation, SAVTrack maintains competitive tracking accuracy while operating at 82 FPS. Notably, it reaches 94.8% Precision on Cyclist, matching the best result reported in the table. The representation-wise comparison provides additional context for these results. Recent voxel- and BEV-based trackers, such as VoxelTrack [19], FocusTrack [45], CompTrack [43], and P2P-voxel [25], achieve higher absolute mean accuracy in several cases. These methods employ spatially aggregated voxel or BEV representations and different localization architectures, whereas SAVTrack operates directly on pointlevel seed hypotheses. Accordingly, the objective of SAV-Track is not to replace the underlying point representation with a heavier spatial encoding, but to improve how explicit seed-to-center hypotheses are utilized during proposal formation. Within this point-based setting, SAVTrack remains competitive with recent trackers and achieves particularly strong results on Van and Cyclist.

Relative to the published P2P-point tracker [25], SAV-Track improves the mean Success/Precision from 66.2/85.4 to 68.4/87.4. The improvements are consistent across the major categories: +2.3/+2.2 on Car, +2.4/+2.7 on Pedestrian, +2.6/+2.4 on Van, and +1.1/+0.0 on Cyclist. The relatively larger gains on Pedestrian and Van are consistent with the motivation of SAV, since sparse or incomplete target observations can produce more ambiguous point-wise center hypotheses. By estimating candidate confidence and removing low-confidence votes before proposal clustering, SAV reduces the influence of unreliable hypotheses without changing the underlying point-based representation. The contribution of the proposed aggregation strategy is isolated more rigorously through the matched dense, soft-weighted, top-1, and hard-gated ablations in Sec. 4.3, while its behavior under different target point densities is further analyzed in Sec. 4.5.

Table 2  
Comparisons with state-of-the-art methods on nuScenes dataset [1]. Success / Precision are used for evaluation. The best and second-best results are highlighted in bold and underline, respectively. <sup>†</sup> denotes results reproduced using the officia implementation.
<table><tr><td>Tracker</td><td>Mean (117,278)</td><td>Car (64,159)</td><td>Pedestrian (33,227)</td><td>Truck (13,587)</td><td>Trailer (3,352)</td><td>Bus (2,953)</td></tr><tr><td>SC3D [10] P2B [28]</td><td>20.70/20.20</td><td>22.31 / 21.93</td><td>11.29 / 12.65 28.39 /52.24</td><td>30.67 / 27.73 42.95 /41.59</td><td>35.28 / 28.12 48.96 / 40.05</td><td>29.35 /24.08 32.95 / 27.41</td></tr><tr><td>PTT [30]</td><td>36.48 / 45.08 36.33 / 41.72</td><td>38.81 / 43.18 41.22 / 45.26</td><td>19.33/32.03</td><td>50.23 /48.56</td><td>51.70 / 46.50</td><td>39.40 / 36.70</td></tr><tr><td>BAT [39]</td><td>38.10 / 45.71</td><td>40.73 / 43.29</td><td>28.83 / 53.32</td><td>45.34 / 42.58</td><td>52.59 / 44.89</td><td>35.44 / 28.01</td></tr><tr><td>V2B [14]</td><td></td><td>54.40 / 59.70</td><td>30.10 /55.40</td><td>53.70 /54.50</td><td>54.90 / 51.44</td><td></td></tr><tr><td>PTTR [42]</td><td>44.50/ 52.07</td><td>51.89 /58.61</td><td>29.90 / 45.09</td><td>45.30 / 44.74</td><td>45.87/ 38.36</td><td>43.14/37.74</td></tr><tr><td>GLT-T [24]</td><td>44.42/54.33</td><td>48.52 / 54.29</td><td>31.74 /56.49</td><td>52.74 / 51.43</td><td>57.60 / 52.01</td><td>44.55 / 40.69</td></tr><tr><td>MoCUT [22]</td><td>51.19 / 64.63</td><td>57.32 /66.01</td><td>33.47/63.12</td><td>61.75/64.38</td><td>60.90 / 61.84</td><td>57.39 / 56.07</td></tr><tr><td>MBPTrack [37]</td><td>57.48 / 69.88</td><td>62.47 / 70.41</td><td>45.32 / 74.03</td><td>62.18 /63.31</td><td>65.14 /61.33</td><td>55.41 / 51.76</td></tr><tr><td>M²Track [40]</td><td>49.23 / 62.73</td><td>55.85 / 65.09</td><td>32.10 /60.92</td><td>57.36 / 59.54</td><td>57.61 / 58.26</td><td>51.39 /51.44</td></tr><tr><td>PTTR++ [20]</td><td>51.86 / 60.63</td><td>59.96 / 66.73</td><td>32.49 / 50.50</td><td>59.85/ 61.20</td><td>54.51 / 50.28</td><td>53.98 / 51.22</td></tr><tr><td>STTracker [5]</td><td>49.66 / 66.77</td><td>56.11 / 69.07</td><td>37.58 / 68.36</td><td>54.29 / 60.71</td><td>48.13 / 55.40</td><td>36.31 / 36.07</td></tr><tr><td>SeqTrack3D [17]</td><td>55.92 / 68.94</td><td>62.55 / 71.46</td><td>39.94 / 68.57</td><td>60.97 / 63.04</td><td>68.37 / 61.76</td><td>54.33 / 53.52</td></tr><tr><td>VoxelTrack [19]</td><td>59.00 / 71.40</td><td>63.90 / 71.60</td><td>46.80 / 75.90</td><td>64.80 / 65.90</td><td>69.50 / 64.30</td><td>60.10/57.70</td></tr><tr><td>TrackAny3D [33]</td><td>54.57 /66.25</td><td>59.30 / 66.46</td><td>40.37 / 68.70</td><td>62.70/62.80</td><td>66.12 /59.20</td><td>61.01 / 58.02</td></tr><tr><td>P2P-voxel† [25]</td><td>59.22 / 71.19</td><td>64.61 / 71.98</td><td>45.64 / 74.62</td><td>64.42 / 65.37</td><td>70.23 / 66.08</td><td>58.54/56.13</td></tr><tr><td>P2P-point [25]</td><td>55.92 /66.64</td><td>62.14 / 68.45</td><td>39.68 /65.59</td><td>62.50 /63.44</td><td>69.04 / 65.14</td><td>57.90 /55.46</td></tr><tr><td>SAVTrack (Ours)</td><td>58.44 / 69.82</td><td>64.27 / 71.72</td><td>42.54/68.31</td><td>65.16 / 66.18</td><td>72.60 / 70.48</td><td>64.02 / 61.67</td></tr></table>

Comparison on nuScenes. We further evaluate SAVTrack on the nuScenes benchmark [1], which provides a substantially larger and sparser tracking setting than KITTI, with 117,278 evaluation frames and point clouds captured by a 32-beam LiDAR. As shown in Tab. 2, SAVTrack achieves a mean Success/Precision of 58.44%/69.82%. Compared with the P2P-point [25], SAV improves the mean Success and Precision by +2.52 and +3.18 points, respectively. The improvement is consistent across all five evaluated categories, including Car (+2.13/+3.27), Pedestrian (+2.86/+2.72), Truck (+2.66/+2.74), Trailer (+3.56/+5.34), and Bus (+6.12/+6.21). These consistent gains indicate that selectively suppressing unreliable center hypotheses benefits different target categories rather than a specific object type.

Although SAVTrack does not achieve the highest performance, it remains competitive with recent strong trackers and exhibits strong category-wise performance. Specifically, SAVTrack achieves the best Success/Precision on

Truck, Trailer, and Bus, while obtaining the second-best results on Car. Compared with MBPTrack [37], SAVTrack improves mean Success from 57.48% to 58.44% while achieving nearly identical mean Precision (69.82% vs. 69.88%). It also substantially outperforms M<sup>2</sup>Track [40] and Mo-CUT [22] in mean tracking accuracy. Notably, the gain over the voting baseline is larger on several challenging vehicle categories, particularly Trailer and Bus. Since categorylevel statistics jointly reflect point density, object scale, and geometric structure, they alone cannot determine whether the improvement originates from observation sparsity. We therefore provide a point-count-stratified analysis in Sec. 4.5 to examine the effect of sparsity more directly.

## 4.2. Visualization

Tracking Results. Fig. 4 presents qualitative tracking results on nuScenes [1] dataset. Compared with P2P-point, SAVTrack produces bounding boxes that are more consistently aligned with the ground-truth target across different scenes, particularly under partial and sparse observations.

Voting Results. Fig. 5 further visualizes the center hypotheses before proposal aggregation. Dense voting produces a relatively dispersed vote distribution, whereas SAV removes low-confidence hypotheses and yields a more concentrated set of candidate centers around the target. This qualitatively supports the motivation of reliability-aware pre-aggregation selection.

## 4.3. Ablation Study

Ablation on Vote Selection and Aggregation. Tab. 4 compares different vote aggregation strategies under the same feature encoder, region-specific regressors, and proposalgeneration pipeline. Soft posterior weighting (B) slightly improves over dense voting (A), increasing the overall Success/Precision from 65.7/85.9 to 66.0/86.3, indicating that the learned posterior provides useful confidence information but cannot fully suppress unreliable hypotheses. Top-1 selection (C) further improves performance to 66.4/86.8 by retaining only the highest-posterior vote per seed, although such aggressive selection may discard additional plausible hypotheses. In contrast, SAV (D) retains approximately 250 high-confidence votes and achieves the best overall result of 68.4/87.4, outperforming Top-1 by 1.6/1.1 on Car and 1.6/1.8 on Pedestrian. These results show that SAV benefits from selectively removing unreliable votes while preserving informative candidates, rather than simply minimizing the number of votes.

![](images/4acce069cb8967a19271ed22c3c6787afe9f0fbb33cff751e949128d136e8542.jpg)  
Figure 4: Qualitative comparison of tracking results on nuScenes. Green, blue, and red boxes denote the ground truth, P2P-point baseline [25], and SAVTrack, respectively. SAVTrack provides more accurate target localization across different target configurations and observation conditions.

Component Analysis. Tab. 5 evaluates the contribution of region-specific multi-hypothesis regression and confidence filtering. Introducing multi-hypothesis regression alone improves the overall Success/Precision from 65.7/85.9 to 67.2/87.0, showing that region-specific regressors provide more flexible center hypotheses but still suffer from indiscriminate aggregation. Adding confidence filtering further raises the mean performance to 68.4/87.4 while reducing the retained votes from approximately 1,536 to 250. Similar gains are observed on both Car and Pedestrian, indicating that the main benefit of SAV comes from suppressing unreliable hypotheses before proposal clustering rather than

Table 7  
![](images/a3d471d50ff97eadc88ca47ec5b297cec3ddc85dc300b7d3b50b50ca01cafd57.jpg)  
Figure 5: Visualization of voting results without (left) and with (right) SAV. With SAV, votes concentrate more tightly around the true object center, as low-confidence votes are excluded before aggregation.

Table 3  
Relationship between posterior confidence, vote localization quality, and local geometry on KITTI Car. Candidate votes are 输入点云<sub>partitioned into five equal-frequency bins according to</sub> $p _ { j } ^ { i } .$ 预测边框Planarity and curvature are computed from the local neighborhood of each seed.
<table><tr><td>Confidence bin</td><td>Mean conf.</td><td>Mean err. (m)↓</td><td>Med. err. (m)↓</td><td>Succ. ratio↑</td><td>Planarity↓</td><td>Curvature↑</td></tr><tr><td>Lowest 20%</td><td>0.12</td><td>0.72</td><td>0.63</td><td>28.3%</td><td>0.81</td><td>0.042</td></tr><tr><td>20-40%</td><td>0.28</td><td>0.54</td><td>0.46</td><td>44.7%</td><td>0.74</td><td>0.061</td></tr><tr><td>40-60%</td><td>0.45</td><td>0.38</td><td>0.31</td><td>62.1%</td><td>0.66</td><td>0.083</td></tr><tr><td>60-80%</td><td>0.63</td><td>0.24</td><td>0.19</td><td>78.6%</td><td>0.57</td><td>0.108</td></tr><tr><td>Highest 20%</td><td>0.84</td><td>0.13</td><td>0.10</td><td>89.4%</td><td>0.49</td><td>0.131</td></tr></table>

Table 4

Ablation of vote aggregation strategies on KITTI. Mean denotes the overall result across all four KITTI categories.
<table><tr><td>Strategy</td><td>Car</td><td>Ped.</td><td>Mean</td><td>Votes</td></tr><tr><td>(A) Dense</td><td>68.8 / 81.7</td><td> $6 2 . 7 / 8 9 . 1 $ </td><td> $6 5 . 7 / 8 5 . 9 $ </td><td>≈1536</td></tr><tr><td>(B) Soft</td><td>69.2 / 82.2</td><td> $6 3 . 1 ~ / 8 9 . 5$ </td><td> $6 6 . 0 \mathrm { ~ / ~ } 8 6 . 3$ </td><td>≈1536</td></tr><tr><td>(C) Top-1</td><td>69.5 / 82.8</td><td> $6 3 . 5 / 9 0 . 0$ </td><td> $6 6 . 4 \ : / \ : 8 6 . 8$ </td><td>128</td></tr><tr><td>(D) SAVTrack</td><td>71.1 / 83.9</td><td>65.1 / 91.8</td><td>68.4 / 87.4</td><td>≈250</td></tr></table>

## Table 5

Component analysis on KITTI. “M-Hyp.” denotes regionspecific vote regression (one regressor per sub-region); “Filter.” denotes posterior-based confidence filtering before proposal aggregation.
<table><tr><td>Variant</td><td>M-Hyp.</td><td>Filter</td><td>Car</td><td>Ped.</td><td>Mean</td><td>Votes</td></tr><tr><td>Baseline</td><td>-√</td><td></td><td>68.8 / 81.7</td><td>62.7 / 89.1</td><td>65.7 / 85.9</td><td>≈1536</td></tr><tr><td>M-Hyp. only</td><td></td><td></td><td>69.6 / 82.4</td><td>63.3 / 90.2</td><td> $6 7 . 2 / 8 7 . 0$ </td><td>≈1536</td></tr><tr><td>SAVTrack</td><td>√</td><td>√</td><td>71.1 /83.9</td><td>65.1 / 91.8</td><td>68.4 / 87.4</td><td>≈250</td></tr></table>

Table 6

Sensitivity to the confidence threshold � on KITTI. Mean denotes the overall result across all KITTI categories.
<table><tr><td>τ</td><td>Car</td><td>Ped.</td><td>Mean</td><td>Avg. votes</td></tr><tr><td>0.0 (w/o filter)</td><td>69.6 / 82.4</td><td> $6 3 . 3 / 9 0 . 2 $ </td><td> $6 7 . 2 / 8 7 . 0$ </td><td>≈1536</td></tr><tr><td>0.1</td><td>70.2 / 83.0</td><td> $6 4 . 4 / 9 0 . 7 $ </td><td> $6 7 . 5 / 8 7 . 7$ </td><td>≈820</td></tr><tr><td>0.3</td><td>71.1 /83.9</td><td> ${ \bf 6 5 . 1 } / 9 1 . 8$ </td><td>68.4/ 87.4</td><td>≈250</td></tr><tr><td>0.5</td><td>70.5 / 83.2</td><td> $6 4 . 7 / 9 1 . 2 $ </td><td>68.1/ 87.2</td><td>≈80</td></tr></table>

from multi-hypothesis regression alone. The calibration and geometric characteristics of the learned confidence are further analyzed in 4.4.

Confidence Threshold Sensitivity Tab. 6 studies the effect of the gating threshold �. Increasing � from 0 to 0.3 progressively removes low-confidence hypotheses and improves the overall tracking performance, with � = 0.3 achieving the best result of 68.4/87.4 using approximately 250 votes.

Effect of the number of sub-regions $N _ { R }$ on KITTI Car.
<table><tr><td> ${ \pmb N } _ { \pmb R }$ </td><td>Success</td><td>Precision</td><td>Avg. votes</td></tr><tr><td>4</td><td>69.8</td><td>82.6</td><td>≈185</td></tr><tr><td>8</td><td>70.6</td><td>83.4</td><td>≈221</td></tr><tr><td>12</td><td>71.1</td><td>83.9</td><td>≈250</td></tr><tr><td>16</td><td>70.9</td><td>83.6</td><td>≈276</td></tr><tr><td>24</td><td>70.3</td><td>83.0</td><td>≈307</td></tr></table>

A larger threshold of 0.5 further reduces the vote set but slightly degrades performance, suggesting that overly aggressive filtering may discard informative hypotheses. We therefore use $\tau = 0 . 3$ as the default setting.

Number of Sub-regions. We study the sensitivity to the number of center-relative sub-regions $N _ { R }$ in Tab. 7. A small $N _ { R }$ provides only a coarse discretization of the centerrelative space, whereas an overly large $N _ { R }$ increases the classification granularity and may leave fewer training samples for each region. Performance improves as $N _ { R }$ increases from 4 to 12, reaching 71.1/83.9 Success/Precision, and then slightly decreases for larger values. We therefore set $N _ { R } ~ = ~ 1 2$ by default, which provides a good trade-off between representation granularity and learning stability.

## 4.4. More Analysis

Vote Confidence and Reliability Analysis. Tracking improvements alone do not reflect whether the learned posterior identifies reliable center hypotheses. We therefore analyze each candidate vote $v _ { j } ^ { i }$ using its posterior confidence $p _ { i } ^ { i }$ and localization error $\dot { e } _ { i } ^ { i } = \lVert \hat { \mathbf { d } } _ { i } ^ { i } - \mathbf { c } \rVert _ { 2 }$ . As shown in Tab. 3, localization quality improves monotonically with confidence: the mean center error decreases from 0.72 m in the lowest-confidence quintile to 0.13 m in the highest, while the successful-vote ratio increases from 28.3% to 89.4%. Meanwhile, higher-confidence votes exhibit lower

Table 10  
Spearman rank correlations between local geometric descriptors and seed-level confidence / vote error on KITTI Car. Confidence is defined as the maximum sub-region posterior of each seed. Correlations with $| \rho | \geq 0 . 3$ are highlighted in bold.
<table><tr><td>Geometric descriptor</td><td>vs. Confidence</td><td>vs. Vote error</td></tr><tr><td>Planarity  $( \uparrow = \mathsf { m o r e } \mathsf { p l a n a r } )$ </td><td>-0.52</td><td>+0.47</td></tr><tr><td>Curvature  $( \uparrow = \mathsf { m o r e \ c u r v e d } )$ </td><td>+0.49</td><td>-0.43</td></tr><tr><td>Normal variation  $( \uparrow = \mathsf { m o r e \ v a r i e d } )$ </td><td>+0.44</td><td>-0.38</td></tr><tr><td>Geometric entropy  $( \uparrow = \mathsf { m o r e \ i s o t r o p i c } )$ </td><td>+0.38</td><td>-0.34</td></tr></table>

## Table 9

Computational overhead. Speed is measured with batch size 1 on a single RTX 3090.
<table><tr><td>Model</td><td>Params (M)</td><td>FPS</td></tr><tr><td>Baseline</td><td>7.39</td><td>98</td></tr><tr><td>SAVTrack (Ours)</td><td>7.76</td><td>82</td></tr></table>

planarity and larger curvature, suggesting that the learned confidence is also associated with more distinctive local geometry. These results support the use of the sub-region posterior as a reliability proxy for vote selection.

Geometric Correlation Analysis. To examine whether the learned confidence is associated with local geometric structure, we compute planarity, curvature, normal variation, and geometric entropy from the �-NN neighborhood of each seed. For each seed, we use the maximum sub-region posterior as its confidence and the localization error of the corresponding highest-confidence vote. Tab. 8 reports their Spearman rank correlations.

Planarity is negatively correlated with confidence $( \rho =$ −0.52) and positively correlated with vote error $\begin{array} { r l } { \left( \rho \right. } & { { } = } \end{array}$ +0.47), suggesting that locally planar regions tend to provide weaker localization evidence. In contrast, curvature, normal variation, and geometric entropy are positively correlated with confidence and negatively correlated with error. These trends are consistent with the confidence-bin analysis in Tab. 3 and support an association between learned confidence and richer local geometric variation, which is consistent with SAV estimating reliability jointly from local point-wise evidence and inter-frame motion context.

Computational Overhead. Tab. 9 shows that the SAV module adds only 0.4M parameters while maintaining realtime inference at 82 FPS, demonstrating that the proposed gating mechanism is lightweight and practical within this tracking pipeline.

Per-category Analysis. Tab. 10 reports the category-wise improvements over the baseline on nuScenes [1]. SAVTrack consistently improves both Success and Precision across all five categories, with gains ranging from +2.13/+3.27 on Car to +6.12/+6.21 on Bus. Notably, the improvement does not monotonically correlate with the average number of foreground points: although Pedestrian has the sparsest observations, Bus and Trailer exhibit larger gains despite having considerably more points. This suggests that category-level improvements are influenced not only by point density, but also by factors such as object geometry and observation structure. We therefore further isolate the effect of sparsity through the point-count-stratified analysis in Sec. 4.5.

Per-category improvement (Success/Precision) over the baseline on nuScenes. ΔS and ΔP denote absolute gains in Success and Precision; $\overline { { N } } _ { \mathsf { p t s } }$ is the average number of foreground points per target.
<table><tr><td>Category</td><td> $\overline { { N } } _ { \mathsf { p t s } }$ </td><td>Baseline</td><td>SAVTrack</td><td>∆S</td><td>∆P</td></tr><tr><td>Car</td><td>312</td><td>62.14 / 68.45</td><td>64.27 / 71.72</td><td>+2.13</td><td>+3.27</td></tr><tr><td>Ped.</td><td>87</td><td>39.68 / 65.59</td><td>42.54 / 68.31</td><td>+2.86</td><td>+2.72</td></tr><tr><td>Truck</td><td>275</td><td>62.50 / 63.44</td><td>65.16 / 66.18</td><td>+2.66</td><td>+2.74</td></tr><tr><td>Trailer</td><td>348</td><td>69.04 / 65.14</td><td>72.60 / 70.48</td><td>+3.56</td><td>+5.34</td></tr><tr><td>Bus</td><td>375</td><td>57.90 / 55.46</td><td>64.02 / 61.67</td><td>+6.12</td><td>+6.21</td></tr></table>

## 4.5. Sparsity-Stratified Analysis

To isolate the effect of observation sparsity from categoryspecific confounders (size, shape, typical occlusion), we stratify test frames by the number of foreground points on the target, following the protocol of P2P [25]. For each frame, we count the points inside the ground-truth bounding box and assign the frame to one of six bins: 0– 10, 10–20, 20–30, 30–40, 40–50, and >50 points. We then compute Success and Precision separately for each bin, comparing the baseline, the multi-hypothesis-only variant, and SAVTrack. Our diagnostic hypothesis is that the benefit of reliability-aware gating increases as observations become sparser, since unreliable hypotheses may have a larger impact on proposal formation when only limited target evidence is available.

Sparsity Trends. Tab. 11 shows a clear dependence on target point density. While all methods degrade as observations become sparser, the gain of SAVTrack over the voting baseline increases consistently: ΔS grows from +2.3 in the > 50-point bin to +5.2 in the 0–10-point bin, while ΔP increases from +2.2 to +6.4. Multi-hypothesis regression alone yields only modest and relatively stable gains across bins, suggesting that the stronger improvement under sparse observations primarily comes from reliability-aware filtering. The retained vote set also becomes substantially smaller in sparse bins, indicating more selective hypothesis retention under limited observations. These results support the hypothesis that reliability-aware gating is particularly beneficial when target observations are sparse.

Effect of Motion-Conditioned Reliability Estimation. To examine whether static selective voting is sufficient for temporal 3D tracking, we construct a SPOT-inspired variant in which the sub-region posterior is estimated using only the current-frame seed feature, while all other components remain unchanged. As shown in Tab. 12, the static-only variant reaches 69.1/82.0 on Car and 62.8/89.4 on Pedestrian, whereas incorporating inter-frame motion cues improves the results to 71.1/83.9 and 65.1/91.8, respectively. This suggests that temporal motion context provides complementary information for assessing vote reliability. Moreover, under the same motion-aware representation, hard gating clearly outperforms soft weighting, improving Car by 1.9/1.7 and Pedestrian by 2.0/2.3 points. These results indicate that SAV benefits from both motion-conditioned reliability estimation and the explicit removal of low-confidence hypotheses before proposal clustering.

Sparsity-stratified analysis on nuScenes. Frames are grouped by the number of foreground points in the ground-truth box. ΔS and ΔP denote the absolute gain of SAVTrack over the voting baseline; “Votes kept” is the average number of votes retained after hard gating in each bin.
<table><tr><td rowspan="2">Points / target</td><td rowspan="2">Voting baseline S</td><td rowspan="2">P</td><td rowspan="2">Multi-hyp. only S</td><td rowspan="2">P S</td><td colspan="2">SAVTrack (Ours)</td><td rowspan="2">Δ ∆P</td><td rowspan="2">Avg. votes kept</td></tr><tr><td>p</td><td>∆S</td></tr><tr><td>0-10</td><td>31.2</td><td>40.8</td><td>32.5</td><td>42.1</td><td>36.4</td><td>47.2</td><td>+5.2</td><td>+6.4 ≈62</td></tr><tr><td>10-20</td><td>40.1</td><td>50.3</td><td>41.3</td><td>51.8</td><td>44.5</td><td>56.1</td><td>+4.4 +5.8</td><td>≈115</td></tr><tr><td>20-30</td><td>47.3</td><td>57.2</td><td>48.6</td><td>58.9</td><td>51.2</td><td>62.4</td><td>+3.9 +5.2</td><td>≈165</td></tr><tr><td>30-40</td><td>53.1</td><td>62.8</td><td>54.2</td><td>64.0</td><td>56.3</td><td>66.2</td><td>+3.2 +3.4</td><td>≈210</td></tr><tr><td>40-50</td><td>57.4</td><td>66.5</td><td>58.3</td><td>67.6</td><td>59.8</td><td>69.1</td><td>+2.4 +2.6</td><td>≈248</td></tr><tr><td>&gt;50</td><td>63.2</td><td>72.8</td><td>64.1</td><td>73.7</td><td>65.5</td><td>75.0</td><td>+2.3 +2.2</td><td>≈295</td></tr></table>

Table 12

Ablation of static and motion-conditioned reliability estimation on KITTI. “Local” and “Motion” indicate the cues used by the subregion posterior estimator; all variants otherwise share the same motion-centric tracking backbone and proposal-generation pipeline.
<table><tr><td>Variant</td><td>Local Cue.</td><td>Motion Cue.</td><td>Selection</td><td>Car</td><td>Ped</td></tr><tr><td>Dense baseline</td><td>一</td><td>一</td><td>Dense</td><td>68.8 / 81.7</td><td>62.7 / 89.1</td></tr><tr><td>Static-only reliability [6]</td><td>√</td><td>×</td><td>Hard</td><td>69.1 / 82.0</td><td>62.8 / 89.4</td></tr><tr><td>Motion-aware + soft weighting</td><td>√</td><td>√</td><td>Soft</td><td>69.2 / 82.2</td><td>63.1 / 89.5</td></tr><tr><td>SAVTrack</td><td>√</td><td>√</td><td>Hard</td><td>71.1 / 83.9</td><td>65.1 / 91.8</td></tr></table>

## 5. Conclusion

In this work, we investigate heterogeneous vote reliability in voting-based 3D single object tracking, where unreliable seed-to-center hypotheses can interfere with proposal formation when aggregated indiscriminately. To address this issue, we propose SAVTrack with Selective Vote Aggregation (SAV), a lightweight pre-aggregation mechanism that predicts a center-relative sub-region posterior from joint point-wise and inter-frame motion representations and uses the resulting confidence to filter unreliable candidate votes before clustering. Experiments on KITTI and nuScenes show that SAVTrack consistently improves the corresponding voting baseline while maintaining realtime inference at 82 FPS. Extensive analyses further show that the learned confidence is predictive of vote localization quality, correlates with local geometric structure, and provides larger gains under sparse target observations. Controlled comparisons also demonstrate the complementary benefits of motion-conditioned reliability estimation and hard pre-aggregation gating. These results highlight the importance of explicitly considering hypothesis reliability before proposal aggregation in point-cloud tracking.

Limitations and Future Work. SAV currently relies on a fixed, uniformly defined center-relative partition, which may not fully capture anisotropic or object-dependent center distributions. Learning adaptive partitions or adopting hierarchical coarse-to-fine voting could provide more flexible hypothesis modeling. In addition, reliability is estimated independently for individual seeds, without explicitly considering geometric consistency among retained hypotheses. Incorporating set-level consensus or temporal consistency verification may further improve robustness under severe sparsity and occlusion. Future work may also explore reliability-aware hypothesis selection in other point-cloud localization tasks where candidate predictions exhibit heterogeneous quality.

## CrediT authorship contribution statement

Sifan Zhou: Investigation, Methodology, Software, Validation, Writing - original draft, Writing - review & editing, Project administration. Linyue Tan: Methodology, Validation, Visualization, Writing - original draft, Writing - review & editing. Qiwei Wang: Software, Formal analysis, Validation, Writing -original draft & editing. Ziyu Zhao: Formal analysis, Writing - review & editing. Xiaobo Lu: Methodology, Writing - review, Project administration, Funding acquisition.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. 62271143), the Frontier Technologies R&D Program of Jiangsu (No. BF2024060). On computing resources, this work was supported by the Big Data Computing Center of Southeast University.

## References

[1] Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O., 2020. nuscenes: A multimodal dataset for autonomous driving, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631.

[2] Cao, J., Zhang, H., Jin, L., Lv, J., Hou, G., Zhang, C., 2024. A review of object tracking methods: From general field to autonomous vehicles. Neurocomputing 585, 127635.

[3] Cui, Y., Fang, Z., Shan, J., Gu, Z., Zhou, S., 2021. 3d object tracking with transformer. British Machine Vision Conference , 1445–1458.

[4] Cui, Y., Fang, Z., Zhou, S., 2019. Point siamese network for person tracking using 3d point clouds. Sensors 20, 143.

[5] Cui, Y., Li, Z., Fang, Z., 2023. Sttracker: Spatio-temporal tracker for 3d single object tracking. IEEE Robotics and Automation Letters .

[6] Du, H., Li, L., Liu, B., Vasconcelos, N., 2020. Spot: Selective point cloud voting for better proposal in point cloud object detection, in: European Conference on Computer Vision, Springer. pp. 230–247.

[7] Fan, B., Zhou, S., Li, J., Zhao, S., Cao, M., Wang, Q., 2026. Beyond frame-wise tracking: A trajectory-based paradigm for efficient point cloud tracking. IEEE Robotics and Automation Letters 11, 3430– 3437. doi:10.1109/LRA.2026.3655309.

[8] Fang, Z., Zhou, S., Cui, Y., Scherer, S., 2020. 3D-Siamrpn: An endto-end learning method for real-time 3d single object tracking using raw point cloud. IEEE Sensors Journal 21, 4995–5011.

[9] Geiger, A., Lenz, P., Urtasun, R., 2012. Are we ready for autonomous driving? the kitti vision benchmark suite, in: 2012 IEEE conference on computer vision and pattern recognition, IEEE. pp. 3354–3361.

[10] Giancola, S., Zarzar, J., Ghanem, B., 2019. Leveraging shape completion for 3d siamese tracking, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1359– 1368.

[11] Guo, Z., Mao, Y., Zhou, W., Wang, M., Li, H., 2022. Cmt: Contextmatching-guided transformer for 3d tracking in point clouds, in: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXII, Springer. pp. 95–111.

[12] Hu, Z., Zhou, S., Nie, J., Zhao, Z., Li, W., Liang, C.j., 2026. Tftrack: A template-free framework for efficient 3d point cloud tracking. arXiv preprint arXiv:2609.07738 .

[13] Hu, Z., Zhou, S., Yuan, Z., Yang, D., Zhao, S., Liang, C.j., 2025. MVCTrack: Boosting 3d point cloud tracking via multimodal-guided virtual cues, in: 2025 IEEE International Conference on Robotics and Automation (ICRA), IEEE. pp. 3745–3751.

[14] Hui, L., Wang, L., Cheng, M., Xie, J., Yang, J., 2021. 3d siamese voxel-to-bev tracker for sparse point clouds. Advances in Neural Information Processing Systems 34, 28714–28727.

[15] Hui, L., Wang, L., Tang, L., Lan, K., Xie, J., Yang, J., 2022. 3d siamese transformer network for single object tracking on point clouds, in: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part II, Springer. pp. 293–310.

[16] Li, S., Liu, G., Xiong, R., Lu, Q., Li, Y., Xiao, Y., 2026. Cftrack: Addressing center and feature ambiguity in 3d point cloud tracking with a geometry-augmented transformer. Neurocomputing , 134801.

[17] Lin, Y., Li, Z., Cui, Y., Fang, Z., 2024. Seqtrack3d: Exploring sequence information for robust 3d point cloud tracking. arXiv preprint arXiv:2402.16249 .

[18] Liu, S., Wang, D., Wang, Q., Huang, K., 2024. Niv-ssd: Neighbor iou-voting single-stage object detector from point cloud. Neurocomputing 597, 127987.

[19] Lu, Y., Nie, J., He, Z., Gu, H., Lv, X., 2024. Voxeltrack: Exploring multi-level voxel representation for 3d point cloud object tracking, in: Proceedings of the 32nd ACM International Conference on Multimedia, Association for Computing Machinery, New York, NY, USA. p. 6345–6354. URL: https://doi.org/10.1145/3664647. 3680923, doi:10.1145/3664647.3680923.

[20] Luo, Z., Zhou, C., Pan, L., Zhang, G., Liu, T., Luo, Y., Zhao, H., Liu, Z., Lu, S., 2024. Exploring point-bev fusion for 3d point cloud object tracking with transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence .

[21] Ma, T., Wang, M., Xiao, J., Wu, H., Liu, Y., 2023. Synchronize feature extracting and matching: A single branch framework for 3d object tracking, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9953–9963.

[22] Nie, J., He, Z., Lv, X., Zhou, X., Chae, D.K., Xie, F., 2024. Towards category unification of 3d single object tracking on point clouds, in: The Twelfth International Conference on Learning Representations.

[23] Nie, J., He, Z., Yang, Y., Bao, Z., Gao, M., Zhang, J., 2023a. Osp2b: One-stage point-to-box network for 3d siamese tracking, in: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pp. 1285–1293.

[24] Nie, J., He, Z., Yang, Y., Gao, M., Zhang, J., 2023b. Glt-t: Globallocal transformer voting for 3d single object tracking in point clouds, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1957–1965.

[25] Nie, J., Xie, F., Zhou, S., Zhou, X., Chae, D.K., He, Z., 2025. P2P: Part-to-part motion cues guide a strong tracking framework for lidar point clouds. International Journal of Computer Vision 133, 5326–5342.

[26] Qi, C.R., Litany, O., He, K., Guibas, L.J., 2019. Deep hough voting for 3d object detection in point clouds, in: proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9277– 9286.

[27] Qi, C.R., Yi, L., Su, H., Guibas, L.J., 2017. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30.

[28] Qi, H., Feng, C., Cao, Z., Zhao, F., Xiao, Y., 2020. P2b: Point-to-box network for 3d object tracking in point clouds, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6329–6338.

[29] Shan, J., Zhou, S., Cui, Y., Fang, Z., 2022. Real-time 3d single object tracking with transformer. IEEE Transactions on Multimedia 25, 2339–2353.

[30] Shan, J., Zhou, S., Fang, Z., Cui, Y., 2021. Ptt: Point-tracktransformer module for 3d single object tracking in point clouds, in: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1310–1316.

[31] Tong, G., Peng, H., Shao, Y., Yin, Q., Li, Z., 2022. Ascnet: 3d object detection from point cloud based on adaptive spatial context features. Neurocomputing 475, 89–101.

[32] Wang, J., Wang, Y., Zhao, S., Zhou, S., 2025a. Point4bit: Post training 4-bit quantization for point cloud 3d detection, in: The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[33] Wang, M., Wang, H., Li, Y., Kong, X., Du, J., Shen, G., Xia, F., 2025b. Trackany3d: Transferring pretrained 3d models for categoryunified 3d point cloud tracking. Proceedings of the IEEE International Conference on Computer Vision (ICCV) .

[34] Wu, Y., Lim, J., Yang, M.H., 2013. Online object tracking: A benchmark, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2411–2418.

[35] Xia, Y., Wu, Q., Li, W., Chan, A.B., Stilla, U., 2023. A lightweight and detector-free 3d single object tracker on point clouds. IEEE Transactions on Intelligent Transportation Systems , 5543–5554.

[36] Xu, T.X., Guo, Y.C., Lai, Y.K., Zhang, S.H., 2023a. Cxtrack: Improving 3d point cloud tracking with contextual information, in:

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1084–1093.

[37] Xu, T.X., Guo, Y.C., Lai, Y.K., Zhang, S.H., 2023b. Mbptrack: Improving 3d point cloud tracking with memory networks and box priors, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 9877–9886.

[38] Yu, S., Wang, X., Ci, Y., Li, Y., Zhou, M., Dong, X., Cai, M., 2025. Mspe-fusion: A multimodal 3d object detection method with multisensor perception enhanced fusion. Neurocomputing 645, 130486.

[39] Zheng, C., Yan, X., Gao, J., Zhao, W., Zhang, W., Li, Z., Cui, S., 2021. Box-aware feature enhancement for single object tracking on point clouds, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13199–13208.

[40] Zheng, C., Yan, X., Zhang, H., Wang, B., Cheng, S., Cui, S., Li, Z., 2022. Beyond 3d siamese tracking: A motion-centric paradigm for 3d single object tracking in point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8111–8120.

[41] Zheng, C., Yan, X., Zhang, H., Wang, B., Cheng, S., Cui, S., Li, Z., 2023. An effective motion-centric paradigm for 3d single object tracking in point clouds. IEEE Transactions on Pattern Analysis and Machine Intelligence , 43–60.

[42] Zhou, C., Luo, Z., Luo, Y., Liu, T., Pan, L., Cai, Z., Zhao, H., Lu, S., 2022. Pttr: Relational 3d point cloud object tracking with transformer, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8531–8540.

[43] Zhou, S., Cao, Y., Nie, J., Fu, Y., Zhao, Z., Lu, X., Wang, S., 2026a. Comptrack: Information bottleneck-guided low-rank dynamic token compression for point cloud tracking, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 13773–13781.

[44] Zhou, S., Li, L., Zhang, X., Zhang, B., Bai, S., Sun, M., Zhao, Z., Lu, X., Chu, X., 2024. Lidar-ptq: Post-training quantization for point cloud 3d object detection, in: The Twelfth International Conference on Learning Representations (ICLR).

[45] Zhou, S., Nie, J., Zhao, Z., Cao, Y., Lu, X., 2025a. Focustrack: One-stage focus-and-suppress framework for 3d point cloud object tracking, in: Proceedings of the 33rd ACM International Conference on Multimedia, New York, NY, USA. p. 7366–7375. doi:10.1145/ 3746027.3754781.

[46] Zhou, S., Xu, W., Xiong, J., Zhao, Z., Yuan, Z., 2026b. Pillartrack: Boosting pillar representation for transformer-based 3d single object tracking on point clouds. Knowledge-Based Systems 352, 116973. doi:https://doi.org/10.1016/j.knosys. 2026.116973.

[47] Zhou, S., Yuan, Z., Yang, D., Hu, X., Qian, J., Zhao, Z., 2025b. Pillarhist: A quantization-aware pillar feature encoder based on height-aware histogram, in: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27336–27345.

[48] Zhou, S., Zhang, X., Chu, X., Zhang, B., Zhao, Z., Lu, X., 2025c. Fastpillars: A deployment-friendly pillar-based 3d detector. IEEE Transactions on Circuits and Systems for Video Technology , 4999– 5010.