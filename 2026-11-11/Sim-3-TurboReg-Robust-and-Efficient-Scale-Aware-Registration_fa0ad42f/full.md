# Sim(3)-TurboReg: Robust and Efficient Scale-Aware Registration via Tri-Scale Consistency

Zijun Zhou<sup>∗1</sup> Shaocheng Yan<sup>∗1</sup>

Zhenjun Zhao<sup>2</sup> Yutong Hu<sup>1</sup> Ke Yang<sup>1</sup> Yingdong Gu<sup>1</sup> Pengcheng Shi<sup>1</sup> Jiayuan Li<sup>1†</sup>

<sup>1</sup>Wuhan University <sup>2</sup>University of Zaragoza

{zzjivan, shaochengyan, ljy\_whu\_2012}@whu.edu.cn

Abstract— Feed-forward 3D foundation models recover geometry at high fidelity, yet inference memory constraints keep them from processing long sequences natively. Current streaming systems work around this limit with sliding-window chunking, which introduces severe monocular scale ambiguity. Resolving this ambiguity requires Sim(3) registration that stays accurate and efficient under outlier contamination. We introduce Sim(3)-TurboReg, which reformulates the compatibility graph construction of TurboReg by replacing absolute distance constraints with a strict Tri-Scale Consistency (TSC) criterion. Our analysis shows that TSC suppresses both mismatching and coordinate-distortion outliers. We integrate the resulting solver into existing streaming systems and obtain high-fidelity, kilometer-scale reconstruction for feed-forward models. On controlled unknown-scale registration at 90% outliers, ours matches RANSAC accuracy with 92.5% lower median latency. In DA3-Streaming on KITTI, ours improves mean ATE RMSE by 22.2% and reduces mean end-to-end time by 16.6%. Code: https://github.com/ivan-zzj/Sim3-TurboReg.

Index Terms— Mapping, RGB-D Perception, Deep Learning for Visual Perception

## I. INTRODUCTION

Feed-forward 3D foundation models [1]–[3] are now a standard approach to visual geometry, recovering dense pointmaps and camera poses from uncalibrated images in a single forward pass. However, their joint attention mechanisms impose memory costs that grow quadratically with the input length [4], causing out-of-memory failures on sequences beyond a few hundred frames on a single GPU [5]. This computational ceiling prevents their deployment on the continuous, kilometer-scale video streams required for autonomous driving [6] and large-scale mapping [7].

The prevailing workaround for this memory bottleneck is a sliding-window strategy [8], [9]: partitioning a long video into overlapping chunks, processing each independently, and aligning the resulting local reconstructions into a coherent global map [8]. However, without metric priors, feed-forward models reconstruct each chunk in its own local coordinate system with an unconstrained scale [1], [2]. This inherent scale ambiguity systematically invalidates standard rigid SE(3) registration, which relies on absolute distance preservation. Consequently, to fuse these independent chunks, the alignment task is raised from SE(3) to the more challenging Sim(3) domain [10], [11]. Fig. 1 illustrates the resulting global reconstruction and compares local chunk alignment using IRLS and Sim(3)-TurboReg.

Robust Sim(3) solvers such as TEASER++ [12] estimate scale, rotation, and translation despite incorrect correspondences, but accuracy and runtime depend on outlier rate and problem size. Conversely, sampling-based approaches (e.g., RANSAC [13] + Umeyama [11]) demand excessive iterations to filter outliers, failing to guarantee deterministic solutions. While recent graph-based methods [14]–[16] achieve robustness and computational efficiency, their reliance on absolute distance invariance limits them to SE(3) registration.

To bridge this gap, we show a key insight that state-ofthe-art graph-based SE(3) estimators [14], [16], [17] can be adapted to Sim(3) registration by changing how their compatibility graph is built. Building on this, we introduce Sim(3)-TurboReg, an approach that replaces the absolute distance constraints of the original TurboReg [14] with a strict Tri-Scale Consistency (TSC) criterion. As illustrated in Fig. 2, TSC combines one edge scale and two endpoint local scales to assess correspondence compatibility. The resulting pipeline (Fig. 2) handles both general point cloud matches and the outlier distributions in streaming systems.

Unlike general point cloud registration, where feature matching techniques inevitably introduce mismatching outliers from incorrect associations, streaming systems that use feed-forward networks with sliding-window chunks [9] extract correspondences directly from overlapping frames after confidence filtering. Since these correspondences originate from identical pixels, their matching relationships are correct by construction. However, dynamics or depth artifacts introduce coordinate-distortion outliers: identities remain correct, but their coordinates no longer follow the common scene transformation. Fig. 3 contrasts incorrect correspondence identities with coherent coordinate distortion. We then analyze why TSC suppresses mismatching outliers super-linearly by minimizing false-pass probabilities, and rejects coordinatedistortion outliers by exploiting the scale drift caused by mixed dynamic-static neighborhoods.

Our contributions are as follows:

1) We propose Sim(3)-TurboReg, establishing a robust and efficient framework for Sim(3) registration by reformulating the compatibility graph construction of the original TurboReg with a strict TSC criterion.

2) We provide a theoretical analysis showing that TSC is effective against both mismatching and coordinatedistortion outliers.

![](images/ef7e209f99ffdd97fc78c7824b998326a867e584d7c64885a5e3f10be41655f1.jpg)  
Fig. 1. Sim(3)-TurboReg overview. Left: core registration on Stanford Bunny and Dragon at 60% and 90% outliers. At 90% outliers, TEASER++ fails while our method matches RANSAC accuracy at lower latency. Right: streaming integration with DA3-Streaming on KITTI-00 and TUM Floor-01, comparing IRLS and our backend. Sequence ATE RMSE decreases from 11.21 to 7.67 m (31.6%) on KITTI-00 and from 7.47 to 6.58 cm (11.9%) on Floor-01.

3) We achieve a better accuracy–efficiency trade-off than prior robust Sim(3) solvers on unknown-scale registration, and as a drop-in backend improve the accuracy– runtime trade-off of streaming systems.

## II. RELATED WORK

## A. Feed-Forward 3D Foundation Models

Recent advances in visual geometry are driven by feedforward models that infer 3D structure directly from images. On the monocular side, Depth Anything [18] and Metric3D [19] provide strong single-image geometry predictions, where Metric3D explicitly targets zero-shot metric-scale depth and surface normals across diverse camera settings. Beyond single images, DUSt3R [1] introduces dense pointmap regression for unconstrained reconstruction from an uncalibrated image pair. MASt3R [2] improves correspondence accuracy by augmenting DUSt3R with a dense local-feature head trained via an additional matching loss. Extending this paradigm to larger view sets, VGGT [3] jointly predicts pointmaps and camera poses from up to hundreds of views via Transformer aggregation. Depth Anything 3 [20] extracts spatially consistent geometry from arbitrary visual inputs regardless of known camera poses. To reduce designated reference view sensitivity, $\pi ^ { 3 ^ { \scriptstyle - } }$ [21] adopts a permutationequivariant formulation. For continuous streams, CUT3R [22] updates a persistent recurrent state with each observation for online reconstruction. Although these models provide exceptional geometry, GPU memory severely limits long continuous sequences. Generalizing them to streaming continuous video reconstruction remains challenging.

## B. Streaming and Large-Scale Reconstruction

Classical SLAM (Simultaneous Localization and Mapping) systems like ORB-SLAM2 [23], ORB-SLAM3 [24], and LDSO [25] provide robust trajectory estimation. They typically require camera intrinsics and sensor extrinsics to build sparse maps from keypoints or high-gradient pixels. Learningbased systems like DROID-SLAM [8], DPVO [26], and DPV-SLAM [27] improve robustness in challenging scenarios. MASt3R-SLAM [28] integrates MASt3R [2] into an online dense SLAM system featuring global optimization and loop closure. VGGT-Long [9] uses a chunk-and-align strategy with confidence weighting, IRLS-based [29] alignment, and a lightweight loop-closure module. S-MUSt3R [30] employs MUSt3R-style [31] priors in a similar segment-and-align pipeline adding segment-level loop detection and posegraph optimization. LONG3R [32] maintains recurrent spatiotemporal memory, while LongStream [33] introduces gaugedecoupled streaming geometry for extremely long sequences. LASER [34] resolves cross-window scale misalignment via training-free layer-wise scale alignment. Depth Anything 3 (DA3) [20] provides DA3-Streaming [35] for memoryefficient chunked inference on extended videos. Across these systems, identity mismatches and structured geometric distortions (e.g., dynamic objects or depth artifacts) often compromise cross-window constraints. This difficulty necessitates robust Sim(3) alignment methods.

## C. Robust Registration and Similarity Estimation

Point cloud registration estimates geometric transformations from putative correspondences corrupted by noise and outliers. RANSAC [13] improves robustness via minimal-set sampling, though runtime grows rapidly with low inlier ratios. Probabilistic methods like CPD [36] and GMMReg [37] formulate registration as density alignment, tolerating moderate noise but degrading under high outlier rates. Beyond these estimators, robust pipelines explicitly prune outliers before transformation fitting. Graph-based methods build correspondence compatibility graphs to extract self-consistent subsets. SC<sup>2</sup>-PCR [16] incorporates second-order spatial compatibility. Clique-based formulations select mutually consistent correspondences via clique discovery or graph-search strategies like MAC [17] and FastMAC [38]. TurboReg [14] accelerates consistent-clique search via guided exploration. However, these criteria primarily target SE(3) settings preserving absolute inter-point distances. Under global scale changes, distances scale uniformly, invalidating absolute distance checks. In Sim(3) registration, Umeyama [11] provides a closed-form least-squares solution for clean correspondences. TEASER++ [12] improves robustness via a certifiable formulation decoupling scale, rotation, and translation under a truncated least-squares objective.

![](images/100b7508a5e05522dd2a44c68b80815502b0b3e3b65ea543021660b89532b8b8.jpg)  
Fig. 2. Pipeline of Sim(3)-TurboReg. Putative matches are converted to tri-scale cues: an edge scale $s _ { i j }$ and two endpoint local scales $s _ { i } , s _ { j } .$ . The TSC graph retains edges whose three scales agree $( \Delta _ { i j } \leq \tau ) .$ , and 2nd-order scoring weights edges by shared-neighbor support. TurboClique extracts triplets, followed by a degeneracy filter for near-collinear configurations. Umeyama hypotheses are ranked by full-set inlier voting to obtain the Sim(3) estimate.

## III. METHOD

Fig. 2 summarizes Sim(3)-TurboReg. We first define the Sim(3) setting and two outlier modes in Section III-A. Sim(3)- TurboReg then proceeds in three stages: (i) compatibility graph construction based on TSC (Section III-B), (ii) Turbo-Clique solver with scale-aware degeneracy check (Section III-C), and (iii) closed-form Sim(3) estimation with global inlier scoring (Section III-D). Finally, Section III-E explains why this design suppresses both outlier types.

## A. Problem Definition and Outlier Types

Setup. Given source and target point sets $\boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } \subseteq \mathbb { R } ^ { 3 }$ and putative correspondences $\mathcal { M } \ = \ \{ m _ { i } \ = \ ( { \pmb x } _ { i } , { \pmb y } _ { i } ) \} _ { i = 1 } ^ { N }$ , we estimate $\mathbf { T } = ( s , \mathbf { R } , t ) \in \mathrm { S i m } ( 3 )$

$$
\pmb { y } _ { i } = s \mathbf { R } \pmb { x } _ { i } + \pmb { t } + \pmb { e } _ { i } .\tag{1}
$$

Here, $s > 0$ is the global scale, $\mathbf { R } \in \mathit { S O } ( 3 )$ is rotation, $\pm \in \mathbb { R } ^ { 3 }$ is translation, and $\boldsymbol { e } _ { i } \in \mathbb { R } ^ { 3 }$ is residual noise. For inliers, $\| e _ { i } \| _ { 2 } \leq \epsilon .$ . Outliers are split into two modes:

• Type 1 (mismatching outliers): Correspondence identities are incorrect, as illustrated by the crossed associations in Fig. 3(a); we model the resulting errors $e _ { i }$ as unstructured noise.

• Type 2 (coordinate-distortion outliers): Correspondence identities are correct, but coordinates are distorted by dynamics or depth artifacts. Fig. 3(b) shows correctly identified points undergoing coherent displacement rela tive to static neighbors. Unlike Type 1, the resulting errors $e _ { i }$ contain structured information, exhibiting spatial coherence or local kinematic consistency rather than random scatter.

## B. Compatibility Graph Construction Based on TSC

Edge scale $( s _ { i j } )$ . For a pair of correspondences $( m _ { i } , m _ { j } )$

$$
s _ { i j } = { \frac { \| { \pmb y } _ { i } - { \pmb y } _ { j } \| _ { 2 } } { \| { \pmb x } _ { i } - { \pmb x } _ { j } \| _ { 2 } } } .\tag{2}
$$

For true inlier pairs, $s _ { i j } \approx s ^ { * }$ . If two source points are too close, the denominator in Eq. (2) becomes tiny, so small coordinate noise can excessively inflate $s _ { i j }$ and destabilize estimation. Thus, pairs with $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert _ { 2 } < d _ { \operatorname* { m i n } }$ are excluded. Node scale $( s _ { i } , s _ { j } ) .$ . For each correspondence $m _ { i } = ( { \pmb x } _ { i } , { \pmb y } _ { i } )$ we build a bilateral neighborhood and reduce it to one robust local scale. The two sides of $m _ { i }$ are queried separately:

1) Source-side neighbors. Find the k nearest neighbors of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ in $\mathcal { X } = \{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N } \}$ , excluding the query point itself; write their correspondence indices as $\mathcal { N } _ { i } ^ { x }$

2) Target-side neighbors. Find the k nearest neighbors of $\mathbf { \nabla } _ { \mathbf { \psi } _ { 3 } } \mathbf { \psi } _ { 2 } \qquad \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \qquad \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \qquad \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 3 } \qquad \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 3 } \qquad \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 3 } \mathbf { \psi } _ { 4 } \mathbf { \psi } _ { 4 }$ in $\mathcal { Y } = \{ \pmb { y } _ { 1 } , \dotsc , \pmb { y } _ { N } \}$ , again excluding the query point itself; write their indices as $\mathcal { N } _ { i } ^ { y }$

3) Union. Set $\mathcal { N } _ { i } = \mathcal { N } _ { i } ^ { x } \cup \mathcal { N } _ { i } ^ { y }$ . The two kNN lists may overlap, so |N<sub>i</sub>| ranges from k to 2k.

Each index $j \in \mathcal { N } _ { i }$ names a full correspondence $\begin{array} { r l } { m _ { j } } & { { } = } \end{array}$ $( \pmb { x } _ { j } , \pmb { y } _ { j } )$ , so it contributes one pairwise scale ratio

$$
r _ { i j } = { \frac { \| { \pmb y } _ { i } - { \pmb y } _ { j } \| _ { 2 } } { \| { \pmb x } _ { i } - { \pmb x } _ { j } \| _ { 2 } } } .\tag{3}
$$

Pairs with $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert _ { 2 } < d _ { \operatorname* { m i n } }$ are dropped (the denominator would be unstable). The node scale is the median of these ratios:

$$
s _ { i } = \mathrm { m e d i a n } \{ r _ { i j } : j \in \mathcal { N } _ { i } , \| \pmb { x } _ { i } - \pmb { x } _ { j } \| _ { 2 } \ge d _ { \mathrm { m i n } } \} .\tag{4}
$$

That is, we first form k–2k pairwise ratios and then take their median; we do not take a median distance first and divide. Here, k controls locality of scale consensus: small k yields high variance, while large k increases robustness but may mix unrelated structure. Using the bilateral union neighborhood avoids reliance on one coordinate space and improves robustness to structured distortion. Algorithm 1 summarizes the full procedure, with node-scale estimation highlighted as the key step.

Algorithm 1 TSC Compatibility Graph Construction   
Require: $\mathcal { M } = \{ ( \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } ) \} _ { i = 1 } ^ { N } ;$ k; $d _ { \operatorname* { m i n } } ; \tau$   
Ensure: compatibility graph $\mathbf { \bar { G } } \in \{ 0 , 1 \} ^ { N \times N }$   
1: for $i = 1 , \ldots , N$ do ▷ node scales   
2: $\mathcal { N } _ { i } ^ { x } \gets$ indices of k nearest neighbors of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ in X,   
excluding i   
3: $\mathcal { N } _ { i } ^ { y } \gets$ indices of k nearest neighbors of $\mathbf { \nabla } _ { \mathbf { \psi } _ { 3 } } \psi _ { i }$ in ${ \mathcal { V } } ,$   
excluding i   
4: $\mathcal { N } _ { i }  \mathcal { N } _ { i } ^ { x } \cup \mathcal { N } _ { i } ^ { y }$ ▷ k–2k indices   
5: $R _ { i } \gets \emptyset$   
6: for each $j \in \mathcal N _ { i }$ do   
7: if $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert _ { 2 } \geq d _ { \operatorname* { m i n } }$ then   
8: $R _ { i }  R _ { i } \cup \{ \| y _ { i } - y _ { j } \| _ { 2 } / \| x _ { i } - x _ { j } \| _ { 2 } \}$ ▷   
pairwise ratio $r _ { i j }$   
9: end if   
10: end for   
11: $s _ { i } \gets$ median $( R _ { i } )$ ▷ median of ratios, not of   
distances   
12: end for   
13: for $i = 1 , \ldots , N$ do ▷ TSC graph   
14: for $j = i + 1 , \dotsc , N$ do   
15: if $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert _ { 2 } \geq d _ { \operatorname* { m i n } }$ then   
16: $s _ { i j } \gets \| \pmb { y } _ { i } - \pmb { y } _ { j } \| _ { 2 } / \| \pmb { x } _ { i } - \pmb { x } _ { j } \| _ { 2 }$   
17: $\Delta _ { i j }$ ← max(ln s<sub>ij</sub> , ln s<sub>i</sub>, ln s<sub>j</sub> ) −   
min(ln $s _ { i j }$ , ln $s _ { i } ,$ ln s<sub>j</sub> )   
18: $G _ { i j }  G _ { j i }  \mathbb { 1 } [ \Delta _ { i j } \leq \tau ]$   
19: end if   
20: end for   
21: end for   
22: $G _ { i i } \gets 0 , \ \forall i$

The small-set CPU variant in Section IV-A uses histogram node scales: it bins valid neighborhood ratios in log space and takes the geometric mean within the most populated bin. The median-based analysis below applies to Eq. (4).

TSC Graph (G). Given $\left( { { s _ { i j } } , { s _ { i } } , { s _ { j } } } \right)$ , we define log-domain divergence:

$$
\Delta _ { i j } = \operatorname* { m a x } ( \ln s _ { i j } , \ln s _ { i } , \ln s _ { j } ) - \operatorname* { m i n } ( \ln s _ { i j } , \ln s _ { i } , \ln s _ { j } ) .\tag{5}
$$

$\Delta _ { i j }$ measures disagreement among the three scale evidences in log-space. Consistency is

$$
G _ { i j } = \mathbb { 1 } [ \Delta _ { i j } \leq \tau ] ,\tag{6}
$$

where $\mathbb { 1 } [ \cdot ]$ is the indicator function (true = 1, false = 0). We enforce $G _ { i i } = 0$ (no self-loops). Smaller τ gives a purer but sparser graph, while larger τ increases recall at the cost of ambiguity. Fig. 3 illustrates TSC on both outlier types with real streaming correspondences and synthetic geometric examples. The mismatched pair in Fig. 3(a) has $\Delta _ { i j } = 0 . 5 2 1 > \tau$ due to inconsistent node scales. In Fig. 3(b), a shared displacement leaves $s _ { i j } = 1 . 2 2 0$ unchanged, but the local scales become $s _ { i } = 1$ .485 and $s _ { j } = 1 . 1 2 0$ , giving $\Delta _ { i j } = 0 . 2 8 2 > \tau$ . This illustrates how mixed static–dynamic neighborhoods can expose distortion even when the edge scale remains plausible; rejection depends on the neighborhood and distortion, as analyzed in Section III-E.2.

![](images/c701aff2f6461e46a97e00afab63dc12be2cb24b4d60e684d9da0bcba98b2693.jpg)  
Fig. 3. Both outlier types in real streaming correspondences and their synthetic geometric models. (a) Type 1 mismatching outliers: paired points link different scene regions despite high descriptor similarity. (b) Type 2 coordinate-distortion outliers: pairs select the same landmark but undergo coherent displacement. Node scales use bilateral kNN $( k = 6 ) , \tau = 0 . { \bar { 1 } } 5$ (illustrative settings; experiment defaults differ).

Second-order enhancement. Following $\mathrm { S C ^ { 2 } }$ -PCR [16], we score mutual support:

$$
\hat { G } _ { i j } = G _ { i j } \sum _ { u = 1 } ^ { N } G _ { i u } G _ { j u } .\tag{7}
$$

In matrix form:

$$
\begin{array} { r } { \hat { \mathbf { G } } = \mathbf { G } \odot ( \mathbf { G } \mathbf { G } ^ { \top } ) , } \end{array}\tag{8}
$$

where ⊙ is the Hadamard product. The index u enumerates all correspondences and is unrelated to neighborhood size k. $\hat { G } _ { i j }$ is an edge score weighted by triangle support (shared neighbors). As illustrated by the second-order scoring stage in Fig. 2, edges inside an inlier-consistent subgraph receive high support, while isolated false edges remain weak.

## C. TurboClique Solver with Degeneracy Awareness

We convert $\hat { \mathbf { G } }$ to an Ordered Graph $\mathrm { ( O ^ { 2 } G r a p h ) }$ and run the Pivot-Guided Search (PGS) algorithm [14] to extract high-score 3-cliques in $O ( K _ { 1 } N )$ time, where $K _ { 1 }$ is the number of pivots. Unlike SE(3), Sim(3) estimates an extra scale parameter that is vulnerable to collinear or compact point configurations, requiring a strict degeneracy check for numerical stability.

Scale-Aware Degeneracy Check. For the z-th sampled clique $\mathcal { C } _ { z } = \{ m _ { a } , m _ { b } , m _ { c } \}$ , we use a normalized trianglearea score:

$$
\kappa ( { \mathcal C } _ { z } ) = \frac { 2 \| ( { \pmb x } _ { b } - { \pmb x } _ { a } ) \times ( { \pmb x } _ { c } - { \pmb x } _ { a } ) \| _ { 2 } } { \| { \pmb x } _ { a } - { \pmb x } _ { b } \| _ { 2 } ^ { 2 } + \| { \pmb x } _ { b } - { \pmb x } _ { c } \| _ { 2 } ^ { 2 } + \| { \pmb x } _ { c } - { \pmb x } _ { a } \| _ { 2 } ^ { 2 } } .\tag{9}
$$

κ is scale-invariant and approaches 0 for near-collinear triplets. We discard cliques with $\kappa ( \mathcal { C } _ { z } ) \leq \epsilon _ { \mathrm { a r e a } }$ . Here, $\epsilon _ { \mathrm { a r e a } }$ is the degeneracy threshold; larger values reject more near-collinear cliques.

## D. Closed-Form Sim(3) Estimation

For each surviving clique $\mathcal { C } _ { z }$ , we estimate its corresponding local transformation parameters $T _ { z } = ( s _ { z } , { \bf R } _ { z } , t _ { z } )$ by minimizing the Umeyama objective [11]:

$$
\operatorname* { m i n } _ { { s _ { z } , \mathbf { R } _ { z } , \mathbf { t } _ { z } } } \sum _ { m _ { i } \in \mathcal { C } _ { z } } \| \pmb { y } _ { i } - ( s _ { z } \mathbf { R } _ { z } \pmb { x } _ { i } + \pmb { t } _ { z } ) \| _ { 2 } ^ { 2 } .\tag{10}
$$

We then perform full-set inlier voting to find the globally optimal transformation:

$$
\mathbf { T } = \arg \operatorname* { m a x } _ { \mathbf { T } _ { z } } \sum _ { i = 1 } ^ { N } \Im [ \| y _ { i } - ( s _ { z } \mathbf { R } _ { z } \pmb { x } _ { i } + \pmb { t } _ { z } ) \| _ { 2 } < \tau _ { \mathrm { i n } } ] .\tag{11}
$$

Here, $\tau _ { \mathrm { i n } }$ is the global inlier threshold for final hypothesis scoring.

## E. Analysis of TSC on Both Outlier Types

We analyze how TSC suppresses both outlier types through a two-stage mechanism: tri-scale consistency first removes most false edges, and second-order triangle support then enlarges the score gap between inlier and residual false edges. The examples in Fig. 3 illustrate the first stage; Fig. 2 depicts the subsequent shared-neighbor scoring.

1) Type 1: Mismatching Outliers: Write $z _ { i j } = \ln s _ { i j }$ and $z ^ { * } = \ln s ^ { * }$ . Edge-scale-only checking (ESC) retains a false edge whenever $| z _ { i j } - z ^ { * } | \le \tau$ . If the local density of Type 1 outlier log-scales around $z ^ { * }$ is $f _ { 0 }$ , then

$$
p _ { \mathrm { e s c } } \approx 2 \tau f _ { 0 } ,\tag{12}
$$

so a mismatched edge passes as soon as its edge scale happens to be plausible. TSC is more selective: it keeps (i, j) only if all three log-scales agree with $z ^ { * }$

$$
\begin{array} { c } { { E _ { i j } \colon | z _ { i j } - z ^ { * } | \leq \tau , \quad N _ { i } \colon | \ln s _ { i } - z ^ { * } | \leq \tau , } } \\ { { N _ { j } \colon | \ln s _ { j } - z ^ { * } | \leq \tau } } \end{array}\tag{13}
$$

so $p _ { \mathrm { t s c } } = P ( E _ { i j } \cap N _ { i } \cap N _ { j } )$ . For random mismatches the three evidences are largely independent; first-order factorization then gives $p _ { \mathrm { t s c } } \approx p _ { E } p _ { N } ^ { 2 }$ , and with $p _ { E } \approx p _ { N } \approx p _ { \mathrm { e s c } }$ ,

$$
p _ { \mathrm { t s c } } \approx p _ { \mathrm { e s c } } ^ { 3 } .\tag{14}
$$

Correlation weakens the exponent (roughly between quadratic and cubic), but the suppression remains super-linear in the low-p regime.

Second-order scoring amplifies this separation. Let α be the inlier ratio, $p _ { + } = P ( G _ { i j } = 1$ | in–in) and $\begin{array} { r } { p _ { - } = P ( G _ { i j } = } \end{array}$ $1 \mid \mathrm { i n - o u t } )$ . Under an $\mathrm { S C ^ { 2 } - s t y l }$ e count [16] of Eq. (7),

$$
\mathbb { E } [ \hat { G } _ { \mathrm { i n , i n } } ] \approx p _ { + } \left[ ( N \alpha - 2 ) p _ { + } ^ { 2 } + N ( 1 - \alpha ) p _ { - } ^ { 2 } \right] ,\tag{15}
$$

$$
\begin{array} { r } { \mathbb { E } [ \hat { G } _ { \mathrm { i n , o u t } } ] \approx p _ { - } \left[ ( N \alpha - 1 ) p _ { + } p _ { - } + N ( 1 - \alpha ) p _ { - } ^ { 2 } \right] . } \end{array}\tag{16}
$$

TSC mainly reduces $p _ { - }$ while keeping $p _ { + }$ stable, so the gap $\mathbb { E } [ \hat { G } _ { \mathrm { i n , i n } } ] - \mathbb { E } [ \hat { G } _ { \mathrm { i n , o u t } } ]$ grows and TurboClique remains reliable at low inlier ratios.

2) Type 2: Coordinate-Distortion Outliers: Type 2 outliers are harder because two points on the same dynamic part can maintain nearly correct pairwise scale $s _ { a b } \approx s ^ { * }$ , so edgeonly checks may retain them. TSC defends through node scales. Consider a mixed pair $( a , b )$ in which a is displaced by $v _ { d }$ while b stays near-static, and write $r _ { a b } = \| \pmb { x } _ { a } - \pmb { x } _ { b } \| _ { 2 }$ $\eta _ { a b } = \| \pmb { v } _ { d } \| _ { 2 } / ( s ^ { * } r _ { a b } )$ . The pairwise scale then satisfies

$$
\frac { s _ { a b } } { s ^ { * } } = \sqrt { 1 + 2 \eta _ { a b } \cos \theta _ { a b } + \eta _ { a b } ^ { 2 } } ,\tag{17}
$$

whose log-scale perturbation is

$$
\delta _ { a b } = \left| \ln { \frac { s _ { a b } } { s ^ { * } } } \right| \approx { \frac { \| v _ { d } \| _ { 2 } } { s ^ { * } r _ { a b } } } | \cos \theta _ { a b } | \qquad ( \eta _ { a b } \ll 1 ) .\tag{18}
$$

The perturbation thus grows with the displacement and shrinks with the local radius, so mixed neighborhoods with large motion or tight clustering are most sensitive. If a fraction $\beta _ { i }$ of node i’s neighbors are mixed, with typical displacement magnitude ${ \bar { \delta } } _ { i } .$ , the node scale drifts as

$$
\vert \ln s _ { i } - \ln s ^ { \ast } \vert \approx c _ { i } \beta _ { i } \bar { \delta } _ { i } , \qquad c _ { i } \in ( 0 , 1 ] ,\tag{19}
$$

where $c _ { i }$ absorbs median robustness and local geometry. Since the dynamic-pair edge scale stays plausible (ln $s _ { a b } \approx \ln s ^ { * } )$ the TSC divergence is dominated by these node-scale biases, yielding the sufficient rejection condition

$$
c _ { a } \beta _ { a } \bar { \delta } _ { a } > \tau \quad \mathrm { o r } \quad c _ { b } \beta _ { b } \bar { \delta } _ { b } > \tau .\tag{20}
$$

Type 2 suppression therefore strengthens when distortion is larger, the mixed-neighbor ratio is higher, or the local radius is smaller. For ray-aligned depth noise, replace $\| \pmb { v } _ { d } \| _ { 2 }$ with the local depth-distortion magnitude $\sigma _ { \mathrm { d e p t h } } ;$ the same condition applies.

## IV. EXPERIMENTS

We evaluate core registration (Section IV-A) and streaming integration (Section IV-B); streaming experiments use an Intel i7-13700KF CPU and an NVIDIA RTX 4090 GPU.

## A. Controlled Unknown-Scale Registration

Setup. We evaluate on four Stanford models (Bunny, Armadillo, Dragon, Buddha) [39] following the unknown-scale protocol of TEASER [12, Sec. XI-B]. Each trial samples $N = 1 0 0$ vertices and normalizes the source cloud to unit radius, so all errors are scale-comparable across models. Targets follow ${ \pmb y } _ { i } = s ^ { * } { \bf R } ^ { * } { \pmb x } _ { i } + t ^ { * } + { \pmb \epsilon } _ { i }$ with $s ^ { * } \sim \mathcal { U } [ 1 , 5 ]$ , Haaruniform rotation, t<sup>∗</sup> uniform in the unit ball, and Gaussian noise of scale $\sigma = 0 . 0 1$ clipped to the TEASER noise bound $5 . 5 4 \sigma$ . We then replace 60–90% of the correspondences by uniform outliers drawn on the target side, i.e. pure Type 1 contamination. Two independent seeds with 40 trials per (model, rate) give 320 trials per outlier rate and 1,280 inputs per method. We report the all-trial means of scale error $e _ { s } = | s - s ^ { * } |$ , rotation error $e _ { R }$ (degrees), and translation error $e _ { t } = \| t - { t ^ { * } } \| _ { 2 }$ , together with median single-thread CPU time (Intel i7-13700KF). A trial is a failure if the solver returns no estimate; the all-trial mean is then undefined and is marked $\infty ,$ , with the failure count in parentheses.

TABLE I  
UNKNOWN-SCALE STANFORD REGISTRATION (320 TRIALS/ROW,SINGLE-THREAD CPU). MEAN ERRORS AND MEDIAN TIME; e<sub>t</sub>NORMALIZED. ERRORS AS $e _ { s } / e _ { R } \left( ^ { \circ } \right) / e _ { t }$ . †SIM(3) ADAPTATION OF FGR(OUR RE-IMPLEMENTATION). BEST/SECOND PER METRIC:BOLD/UNDERLINED. FAIL. COUNTS TRIALS WITH NO ESTIMATE (OUT OF320). ∞ MARKS A MEAN THAT IS UNDEFINED BECAUSE THE METHODRETURNED NO ESTIMATE ON AT LEAST ONE TRIAL; TABLE II USES THESAME CONVENTION.
<table><tr><td rowspan="2">Outliers</td><td rowspan="2">Method</td><td colspan="2"> $\boldsymbol { e } _ { s } \mathrm { ~ / ~ } \boldsymbol { e } _ { R } \mathrm { ~ ( ~ } ^ { \circ } \mathrm { ) ~ / ~ } \boldsymbol { e } _ { t } \downarrow$  CPU ms↓ (Fail.)</td></tr><tr><td>0.0032 / 0.212 / 0.0056</td><td>0.119 (0)</td></tr><tr><td rowspan="5">60%</td><td>RANSAC-1K RANSAC-1min</td><td>0.0032 / 0.212 / 0.0056</td><td>0.112 (0)</td></tr><tr><td>FGR-Sim3†</td><td>0.1030 / 2.052 / 0.0994</td><td>0.141 (0)</td></tr><tr><td>TEASER++</td><td>0.0061 / 0.261 / 0.0076</td><td>0.914 (0)</td></tr><tr><td>Ours</td><td>0.0032 / 0.211 / 0.0056</td><td>0.703 (0)</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="5">70%</td><td>RANSAC-1K</td><td>0.0043 / 0.260 / 0.0071</td><td>0.260 (0)</td></tr><tr><td>RANSAC-1min</td><td>0.0043 / 0.260 / 0.0071</td><td>0.252 (0)</td></tr><tr><td>FGR-Sim3†</td><td>1.8639 / 32.317 / 1.6431</td><td>0.139 (0)</td></tr><tr><td>TEASER++</td><td>0.0679 / 0.471 / 0.0363</td><td>0.891 (0)</td></tr><tr><td>Ours</td><td>0.0044 / 0.260 / 0.0071</td><td>0.683 (0)</td></tr><tr><td rowspan="5">80%</td><td>RANSAC-1K</td><td>0.0050 / 0.319 / 0.0086</td><td>0.875 (0)</td></tr><tr><td>RANSAC-1min</td><td>0.0050 / 0.314 / 0.0085</td><td>0.860 (0)</td></tr><tr><td>FGR-Sim3†</td><td>5.5496 / 73.204 / 4.9124</td><td>0.134 (0)</td></tr><tr><td>TEASER++</td><td>14.3121 / 0.702 / 4.4651</td><td>0.879 (0)</td></tr><tr><td>Ours</td><td>0.0049 / 0.314 / 0.0085</td><td>0.671 (0)</td></tr><tr><td rowspan="5">90%</td><td>RANSAC-1K</td><td>∞/∞/∞</td><td>1.329 (80)</td></tr><tr><td>RANSAC-1min</td><td>0.0074 / 0.487 / 0.0133</td><td>8.052 (0)</td></tr><tr><td>FGR-Sim3†</td><td>7.4818 / 113.084 / 6.8162</td><td>0.133 (0)</td></tr><tr><td>TEASER++</td><td>41.8419 / 121.329 / 26.7546</td><td>0.922 (0)</td></tr><tr><td>Ours</td><td>0.0074 / 0.483 / 0.0132</td><td>0.605 (0)</td></tr></table>

Baselines and implementation. We compare TEASER++ [12]; RANSAC with three-point Umeyama sampling [11], [13], either capped at $1 0 ^ { 3 }$ iterations (RANSAC-1K) or run with an adaptive confidence stop under a 60 s budget (RANSAC-1min); a Sim(3) adaptation of FGR [40] (FGR-Sim3<sup>†</sup>, our re-implementation with a closed-form weighted similarity inside the continuation loop); and our CPU variant. All RANSAC variants share the same 5.54σ inlier threshold and a consensus refit. Our CPU variant uses 32 pivots, $k \ = \ 1 0 ,$ , histogram node scales, $d _ { \operatorname* { m i n } } = 1 0 ^ { - 3 } ,$ , TSC tolerance $\tau = 0 . 1$ , and inlier-voting threshold $\tau _ { \mathrm { i n } } = 0 . 0 8 ;$ ; second-order scoring and the degeneracy filter are disabled at this scale (enabling either leaves the estimates unchanged, Table II).

Results. Table I pools all models and seeds per outlier rate. The pooled means are most informative about how each baseline breaks. RANSAC-1K fails catastrophically at 90% outliers (80 of 320 trials): at inlier ratio 0.1 a threepoint sample is all-inlier with probability $\approx 1 0 ^ { - 3 }$ , so the $1 0 ^ { 3 } .$ -iteration cap is simply exhausted; only the adaptive variant always returns an estimate, at 13× our median latency. $\mathrm { F G R } { - } \mathrm { S i m } 3 ^ { \dagger }$ is the fastest but degrades earliest, because its continuation scheme reweights correspondences yet never isolates a mutually consistent subset, so once outliers dominate, the objective itself is misleading. TEASER++ fails in a characteristic order: the scale branch degrades first $( e _ { s } ~ = ~ 1 4 . 3$ at 80%), and rotation follows at 90% $( e _ { R } = 1 2 1 . 3 ^ { \circ } )$ . Decoupled scale estimation is thus the weak link under extreme contamination—exactly the quantity that

TABLE II  
CPU ABLATIONS ON THE SAME 1,280 INPUTS AS TABLE I, ONE FACTOR CHANGED RELATIVE TO OURS. MEAN ROTATION ERROR $e _ { R }$ OVER ALL TRIALS AND MEDIAN CPU TIME. FAIL. COUNTS TRIALS WITH NO ESTIMATE (OUT OF 1,280); A SINGLE FAILURE MAKES THE ALL-TRIAL MEAN ∞. OURS: HISTOGRAM NODE SCALES, UNIFORM POST-REFINEMENT, TSC ON, τ = 0.1, k = 10.
<table><tr><td>Change vs. Ours</td><td> $e _ { R } \ ( ^ { \circ } ) \downarrow$ </td><td>Fail.↓</td><td>CPU ms↓</td></tr><tr><td>TSC off</td><td>∞</td><td>910</td><td>0.648</td></tr><tr><td>No post-refinement</td><td>0.577</td><td>0</td><td>0.620</td></tr><tr><td>Weighted refinement</td><td>0.317</td><td>0</td><td>0.631</td></tr><tr><td>Union-median node scales</td><td>∞</td><td>2</td><td>0.647</td></tr><tr><td>Second-order scoring on</td><td>0.317</td><td>0</td><td>0.674</td></tr><tr><td>Degeneracy filter on</td><td>0.317</td><td>0</td><td>0.640</td></tr><tr><td> $\tau = 0 . 0 5$ </td><td>0.317</td><td>0</td><td>0.634</td></tr><tr><td> $\tau = 0 . 2 0$ </td><td>∞</td><td>5</td><td>0.620</td></tr><tr><td>k = 5</td><td>0.536</td><td>0</td><td>0.479</td></tr><tr><td> $k = 2 0$ </td><td>∞</td><td>1</td><td>0.966</td></tr><tr><td>Ours (default)</td><td>0.317</td><td>0</td><td>0.653</td></tr></table>

TSC constrains three times per edge. Ours is the only method besides RANSAC-1min that returns an estimate on every trial at every rate, matching its accuracy (within 0.004<sup>◦</sup> in e<sub>R</sub>) at 92.5% lower median latency. Its runtime is also nearly flat in the outlier rate (0.70 ms at 60% vs. 0.61 ms at 90%): TSC rejects contamination before search, so more outliers mean cheaper graph pruning rather than more hypotheses—the opposite of sampling-based scaling.

Ablations. Table II varies one factor at a time on the same 1,280 inputs, and the failure pattern mirrors the method’s logic. Disabling TSC is catastrophic (910 failures): the graph becomes almost complete, so pivots land on outlier-dominated cliques whose hypotheses attract no inlier votes. Node scales must be robust and local at once: replacing the histogram reduction with a union median occasionally lets a distorted neighborhood shift the node scale (2 failures), k = 5 makes node scales too noisy (e<sub>R</sub> up 69%), and $k ~ = ~ 2 0$ mixes unrelated structures until local consensus breaks (1 failure). The risk of the tolerance is one-sided: tightening to $\tau = 0 . 0 5$ costs nothing, but loosening to $\tau = 0 . 2 0$ passes enough inconsistent edges to cause 5 failures—a permissive graph is more dangerous than a sparse one, because clique search trusts edge scores. Post-refinement matters $( e _ { R } 0 . 5 7 7 ^ { \circ }$ without it), while a weighted variant adds nothing, indicating the voted inlier set is already clean. Second-order scoring and the degeneracy filter leave every estimate unchanged at $N = 1 0 0 ;$ they matter at streaming scale (Section IV-B), where N is two orders of magnitude larger.

## B. System-Level Streaming Evaluation

We replace only the cross-chunk alignment backend of VGGT-Long [9], Pi-Long [9], [21], and DA3-Streaming [20], [35], leaving each frontend and loop-closure module unchanged. All three align adjacent chunks with confidenceweighted IRLS under Sim(3); Pi-Long builds on $\pi ^ { 3 }$ rather than VGGT, and DA3-Streaming uses Depth Anything 3 in memory-efficient chunked mode. Evaluation uses KITTI Odometry [41] and TUM RGB-D [42] with ATE RMSE and end-to-end runtime.

TABLE III  
KITTI ATE RMSE [M] (↓) AND MEAN END-TO-END TIME [S]. <sup>†</sup>=PRIOR WORK. BEST/SECOND AMONG SIX STREAMING ROWS: BOLD/UNDERLINED. AVG.<sup>∗</sup> EXCLUDES SEQ. 01.
<table><tr><td></td><td>|Methods</td><td>|LC</td><td>Cal.</td><td>Recon. | Time [s] </td><td></td><td>Avg.</td><td>Avg.*</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td>07</td><td>08</td><td>09 10</td></tr><tr><td rowspan="3">Clasic</td><td>ORB-SLAM2 (w/o LC) [23]†</td><td>x</td><td>Required</td><td>Sparse</td><td></td><td>69.727</td><td>26.480</td><td>40.65</td><td>502.20</td><td>47.82</td><td>0.94</td><td>1.30 29.95</td><td>40.82</td><td>16.04</td><td>43.09</td><td>38.77</td><td>5.42</td></tr><tr><td>ORB-SLAM2 (w/ LC) [23]†</td><td></td><td>Required</td><td>Sparse</td><td></td><td>54.816</td><td>9.464</td><td>6.03</td><td>508.34</td><td>14.76</td><td>1.02</td><td>1.57 4.04</td><td>11.16</td><td>2.19</td><td>38.85</td><td>8.39</td><td>6.63</td></tr><tr><td>LDSO [25]†</td><td></td><td>Required</td><td>Sparse</td><td></td><td>22.425</td><td>23.500</td><td>9.32</td><td>11.68</td><td>31.98</td><td>2.85</td><td>1.22</td><td>5.10</td><td>13.55 2.96</td><td>129.02</td><td>21.64</td><td>17.36</td></tr><tr><td rowspan="10">a Baed</td><td>DROID-VO [8]†</td><td>x</td><td>Required</td><td>Dense</td><td></td><td>54.188</td><td>51.187</td><td>98.43 84.20</td><td></td><td>108.80 2.58</td><td>0.93</td><td>59.27</td><td>64.40</td><td>24.20</td><td>64.55</td><td>71.80</td><td>16.91</td></tr><tr><td>DPV-SLAM++ [27]†</td><td></td><td>Required</td><td>Sparse</td><td></td><td>25.749</td><td>27.138</td><td>8.30</td><td>11.86</td><td>39.64</td><td>2.50</td><td>0.78 5.74</td><td>11.60</td><td>1.52</td><td>110.90</td><td>76.70</td><td>13.70</td></tr><tr><td>VGGT-Long [9]</td><td></td><td>No Need</td><td>Dense</td><td>650</td><td>25.671</td><td>22.432</td><td>8.63</td><td>58.07</td><td>52.89</td><td>8.74</td><td>4.22 9.84</td><td>4.67</td><td>2.66</td><td>73.11</td><td>31.82</td><td>27.74</td></tr><tr><td>Pi-Long [9], [21]</td><td></td><td>No Need</td><td>Dense</td><td>2337</td><td>29.766</td><td>15.278</td><td>11.84</td><td>174.65</td><td>42.08</td><td>8.72</td><td>2.42 5.88</td><td>5.53</td><td>4.43</td><td>43.64</td><td>16.77</td><td>11.47</td></tr><tr><td>DA3-Streaming [20]</td><td></td><td>No Need</td><td>Dense</td><td>634</td><td>24.477</td><td>16.541</td><td>11.21</td><td>103.84</td><td>71.58</td><td>3.50</td><td>1.80 5.08</td><td>10.51</td><td>2.33</td><td>36.19</td><td>10.40</td><td>12.81</td></tr><tr><td>VGGT-Long + Sim(3)-TurboReg</td><td></td><td>No Need</td><td>Dense</td><td>320</td><td>29.303</td><td>25.026</td><td>10.00</td><td>72.07</td><td>41.75</td><td>18.55</td><td>3.97 12.30</td><td>4.94</td><td>3.20</td><td>97.32</td><td>31.62</td><td>26.62</td></tr><tr><td>Pi-Long + Sim(3)-TurboReg</td><td></td><td>No Need</td><td>Dense</td><td>1419</td><td>28.141</td><td>15.428</td><td>18.92</td><td>155.26</td><td>35.50</td><td>9.26</td><td>6.88 10.64</td><td>4.37</td><td>4.41</td><td>33.14</td><td>18.28</td><td>12.87</td></tr><tr><td>DA3-Streaming + Sim(3)-TurboReg</td><td></td><td>No Need</td><td>Dense</td><td>529</td><td>19.046</td><td>13.756</td><td>7.67</td><td>71.95</td><td>47.83</td><td>3.34</td><td>3.28 15.04</td><td>9.31</td><td>2.08</td><td>24.97</td><td>10.20</td><td>13.83</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE IV

TUM RGB-D ATE RMSE [M] (↓) AND MEAN END-TO-END TIME [S]. <sup>†</sup>=PRIOR WORK. BEST/SECOND PER GROUP: BOLD/UNDERLINED.
<table><tr><td>Type</td><td>Methods</td><td>Time [s]</td><td>Avg.</td><td>360</td><td>desk</td><td>desk2</td><td>floor</td><td>plant</td><td>room</td><td>rpy</td><td>teddy</td><td>xyz</td></tr><tr><td>Uncalibrated</td><td>Droid-SLAM Uncalibrated [8]†</td><td></td><td>0.163</td><td>0.202</td><td>0.032</td><td>0.091</td><td>0.064</td><td>0.045</td><td>0.918</td><td>0.056</td><td>0.045</td><td>0.012</td></tr><tr><td>SLAM</td><td>Mast3r-SLAM Uncalibrated [28]†</td><td></td><td>0.060</td><td>0.070</td><td>0.035</td><td>0.055</td><td>0.056</td><td>0.035</td><td>0.118</td><td>0.041</td><td>0.114</td><td>0.020</td></tr><tr><td rowspan="6">Streaming</td><td>VGGT-Long [9]</td><td>868</td><td>0.0824</td><td>0.0632</td><td>0.0610</td><td>0.0601</td><td>0.1108</td><td>0.0616</td><td>0.1696</td><td>0.0374</td><td>0.1248</td><td>0.0534</td></tr><tr><td>Pi-Long [9], [21]</td><td>1166</td><td>0.0808</td><td>0.1254</td><td>0.0580</td><td>0.0425</td><td>0.0957</td><td>0.0660</td><td>0.2052</td><td>0.0270</td><td>0.0863</td><td>0.0215</td></tr><tr><td>DA3-Streaming [20]</td><td>633</td><td>0.0492</td><td>0.0468</td><td>0.0202</td><td>0.0222</td><td>0.0747</td><td>0.0401</td><td>0.0948</td><td>0.0321</td><td>0.0907</td><td>0.0208</td></tr><tr><td>VGGT-Long + Sim(3)-TurboReg</td><td>492</td><td>0.0698</td><td>0.0651</td><td>0.0316</td><td>0.0412</td><td>0.1203</td><td>0.0490</td><td>0.1258</td><td>0.0339</td><td>0.1168</td><td>0.0443</td></tr><tr><td>Pi-Long + Šim(3)-TurboReg</td><td>695</td><td>0.0682</td><td>0.0969</td><td>0.0295</td><td>0.0264</td><td>0.1192</td><td>0.0579</td><td>0.1403</td><td>0.0242</td><td>0.0978</td><td>0.0213</td></tr><tr><td>DA3-Streaming + Sim(3)-TurboReg</td><td>641</td><td>0.0437</td><td>0.0459</td><td>0.0170</td><td>0.0247</td><td>0.0658</td><td>0.0358</td><td>0.0929</td><td>0.0310</td><td>0.0615</td><td>0.0184</td></tr></table>

TABLE V

END-TO-END PROCESSING TIME [S] (COARSE; VARIES WITH SYSTEM LOAD). BEST/SECOND: BOLD/UNDERLINED
<table><tr><td rowspan="2">Method</td><td colspan="10">KITTI Odometry</td><td colspan="10">TUM RGB-D</td></tr><tr><td>Avg.</td><td>00 01</td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td>07</td><td>08</td><td>09</td><td>10</td><td>Avg. 360</td><td>desk desk2</td><td></td><td>floor</td><td>plant</td><td>room</td><td>rpy</td><td>teddy</td><td></td><td>xyz</td></tr><tr><td>VGGT-Long [9]</td><td>650</td><td>1785 299</td><td>1356</td><td>217</td><td>70</td><td>853</td><td>377</td><td>311</td><td>1109</td><td>452</td><td>322</td><td>868</td><td>789 577</td><td>456</td><td>861</td><td></td><td>1039</td><td>1313</td><td>737</td><td>1238</td><td>800</td></tr><tr><td>Pi-Long [9], [21]</td><td>2337</td><td>5651</td><td>761 4481</td><td>649</td><td>263</td><td>3621</td><td>1261</td><td>1171</td><td>4525</td><td>2154</td><td>1168</td><td>1166</td><td>932</td><td>531</td><td>643</td><td>1433</td><td>1093</td><td>1759</td><td>1110</td><td>1852</td><td>1144</td></tr><tr><td>DA3-Streaming [20]</td><td>634</td><td>900 233</td><td>1597</td><td>228</td><td>59</td><td>869</td><td>490</td><td>474</td><td>1391</td><td>413</td><td>324</td><td>633</td><td>740</td><td>330 380</td><td>704</td><td></td><td>719</td><td>762</td><td>491</td><td>1096</td><td>472</td></tr><tr><td>VGGT-Long+Sim(3)-TurboReg</td><td>320</td><td>705 149</td><td>756</td><td>109</td><td>34</td><td>433</td><td>173</td><td>158</td><td>619</td><td>225</td><td>161</td><td>492</td><td>254 190</td><td>207</td><td>837</td><td>920</td><td></td><td>577</td><td>273</td><td>769</td><td>403</td></tr><tr><td>Pi-Long+Sim(3)-TurboReg</td><td>1419</td><td>4899 440</td><td>2074</td><td>276</td><td>92</td><td>2354</td><td>664</td><td>595</td><td>2700</td><td>773</td><td>738</td><td>695</td><td>543 364</td><td>312</td><td>1094</td><td>842</td><td></td><td>1145</td><td>340</td><td>913</td><td>701</td></tr><tr><td>DA3-Streaming+Sim(3)-TurboReg</td><td>529</td><td>895 186</td><td>1314</td><td>164</td><td>48</td><td>813</td><td>255</td><td>346</td><td>1230</td><td>329</td><td>243</td><td>641</td><td>470 335</td><td>411</td><td>868</td><td>795</td><td></td><td>1141</td><td>520</td><td>771</td><td>459</td></tr></table>

KITTI (Table III). DA3-Streaming improves the most and attains the best mean ATE and $\mathbf { A v g . } ^ { * }$ among streaming pipelines; Pi-Long improves in mean ATE, though its $\mathbf { A v g . } ^ { * }$ slightly increases (15.28→15.43 m). The gain tracks frontend correspondence quality: dense, high-confidence matches give TSC clean neighborhoods for node scales, so the backend corrects biased chunk scales that confidence-weighted IRLS absorbs instead of rejecting. Seq. 01 is high-speed and its ATE is anomalously large for every streaming method, so we exclude it in $\operatorname { A v g } ^ { * } ;$ aggressive viewpoint change weakens correspondence confidence and loop closure alike. VGGT-Long is the one accuracy regression, concentrated on long sequences without effective loop closure, where biased crosschunk estimates accumulate; its lightweight loop closure cannot absorb a biased Sim(3) chain the way Pi-Long/DA3- Streaming’s denser matches can.

TUM RGB-D (Table IV). All three pipelines improve on average, and DA3-Streaming with our backend attains the lowest ATE on most sequences among streaming pipelines. Indoor dynamics and depth artifacts match the Type 2 contamination TSC targets, explaining the more uniform gain than on KITTI.

Runtime (Table V). Mean runtime drops in most pipeline– dataset pairings because the backend aligns chunks in milliseconds rather than seconds. The exception is DA3-Streaming on TUM, where the plausibility gate rejects more estimates and falls back to Umeyama; cost otherwise concentrates on long KITTI routes and Pi-Long’s multi-hypothesis frontend. Plausibility-gated fallback. A check (rotation $> ~ 3 0 ^ { \circ }$ or | ln s| > 0.3) falls back to Umeyama on implausible estimates. It helps VGGT-Long/Pi-Long on KITTI but slightly worsens TUM means: KITTI rejections are mostly genuine outliers, whereas on TUM some rejected estimates were correct but aggressive.

Ablation Studies. Table VI varies system-level choices on KITTI and TUM RGB-D. Two patterns stand out. First, accuracy is far more sensitive on KITTI than on TUM: across the tolerance, $k ,$ and $\tau _ { \mathrm { i n } }$ rows the TUM mean stays within 0.003 m, while KITTI swings by up to 7 m—outdoor depth ranges make local scale consensus noisier, so KITTI rewards stricter τ, smaller k, and stricter $\tau _ { \mathrm { i n } }$ , whereas indoor scenes saturate TSC well before these knobs matter. Second, chunk size is the only genuine accuracy–latency trade-off: 60/30 cuts KITTI time by a third and even improves KITTI ATE, since shorter chunks drift less per alignment, but it degrades TUM, where sparser overlaps weaken the constraint chain.

[21] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Scalable permutation-equivariant visual geometry learning,” arXiv:2507.13347, 2025.

TABLE VI

SYSTEM-LEVEL ABLATION (DA3-STREAMING UNLESS NOTED). SHADED = DEFAULT. BEST/SECOND: BOLD/UNDERLINED. KITTI AVG.<sup>∗</sup>   
EXCLUDES SEQ. 01. CHUNK ROWS GIVE CHUNK LENGTH / OVERLAP IN FRAMES.

<table><tr><td colspan="2"></td><td>KITTI (Avg.*)</td><td>TUM (Avg.)</td></tr><tr><td colspan="2">Variant</td><td>ATE [m] / Time [s]</td><td>ATE [m] / Time [s]</td></tr><tr><td rowspan="3">Pipe.</td><td>VGGT-Long [9]</td><td>25.03 / 320</td><td>0.0698 /  492</td></tr><tr><td>Pi-Long [9], [21]</td><td>15.43 / 1419</td><td>0.0682 / 695</td></tr><tr><td>DA3-Streaming [20]</td><td>13.76 / 529</td><td>0.0437  / 641</td></tr><tr><td rowspan="5">Tol.</td><td>τ = 0.10</td><td>17.61  /  377</td><td>0.0444 / 898</td></tr><tr><td>τ = 0.15</td><td>14.41 / 430</td><td>0.0448  / 938</td></tr><tr><td>τ = 0.20 (TUM def)</td><td>15.38 /  510</td><td>0.0437 / 641</td></tr><tr><td>τ = 0.25 (KITTI def)</td><td>13.76 / 529</td><td>0.0440 / 537</td></tr><tr><td>τ = 0.30</td><td>15.42 / 382</td><td>0.0419  / 964</td></tr><tr><td rowspan="5">KNN</td><td>k = 10 (KITTI def)</td><td>13.76 / 529</td><td>0.0448  /  458</td></tr><tr><td>k = 15</td><td>17.97 / 637</td><td>0.0444  / 936</td></tr><tr><td>k = 20 (TUM def)</td><td>18.99 / 496</td><td>0.0437  / 641</td></tr><tr><td>k = 25</td><td>21.05 / 619</td><td>0.0445  / 633</td></tr><tr><td>k = 30</td><td>18.52 / 640</td><td>0.0440  /  456</td></tr><tr><td rowspan="2">Inl.</td><td> $\tau _ { \mathrm { i n } } = 0 . 0 1 ~ ( \mathrm { d e f } )$ </td><td>13.76 / 529</td><td>0.0437  / 641</td></tr><tr><td> $\tau _ { \mathrm { i n } } = 0 . 0 2$ </td><td>15.67  / 379</td><td>0.0436  / 458</td></tr><tr><td rowspan="2">Chunk</td><td>120/60 (def)</td><td>13.76 / 529</td><td>0.0437  / 641</td></tr><tr><td>60/30</td><td>12.16 / 356</td><td>0.0651  / 420</td></tr></table>

The shaded defaults sit between these regimes, so τ and k act as dataset-level knobs rather than per-sequence tuning.

## V. CONCLUSION

We introduced Sim(3)-TurboReg, a robust and efficient Sim(3) registration solver built on a strict TSC criterion that suppresses both mismatching and coordinate-distortion outliers. On controlled unknown-scale registration at 90% outliers, the CPU variant matches RANSAC accuracy with 92.5% lower median latency. In DA3-Streaming on KITTI, the proposed backend improves mean ATE by 22.2% and reduces mean end-to-end time by 16.6%.

## REFERENCES

[1] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D vision made easy,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[2] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[3] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” arXiv:2503.11651, 2025.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017.

[5] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 813–824.

[6] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 621–11 631.

[7] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, “Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age,” IEEE Trans. Robot., vol. 32, no. 6, pp. 1309–1332, 2016.

[8] Z. Teed and J. Deng, “DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, pp. 16 558–16 569.

[9] K. Deng, Z. Ti, J. Xu, J. Yang, and J. Xie, “VGGT-Long: Chunk it, loop it, align it–pushing VGGT’s limits on kilometer-scale long RGB sequences,” arXiv:2507.16443, 2025.

[10] H. Strasdat, J. Montiel, and A. J. Davison, “Scale drift-aware large scale monocular SLAM,” in Robot.: Sci. Syst. (RSS), vol. 2, no. 3. Zaragoza, Spain, 2010, p. 5.

[11] S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, pp. 376–380, 1991.

[12] H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and certifiable point cloud registration,” IEEE Trans. Robot., vol. 37, no. 2, pp. 314–333, 2021.

[13] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981.

[14] S. Yan, P. Shi, Z. Zhao, K. Wang, K. Cao, J. Wu, and J. Li, “TurboReg: TurboClique for robust and efficient point cloud registration,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025.

[15] X. Zhang, J. Q. Yang, S. Zhang, and Y. Zhang, “3D MAC: 3D maximal cliques for robust point cloud registration,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 3206–3215.

[16] Z. Chen, K. Sun, F. Yang, and W. Tao, “SC<sup>2</sup>-PCR: A second order spatial compatibility for efficient and robust point cloud registration,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 13 221–13 231.

[17] X. Zhang, J. Yang, S. Zhang, and Y. Zhang, “3D registration with maximal cliques,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[18] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth Anything: Unleashing the power of large-scale unlabeled data,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 10 371–10 381.

[19] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, and X. Chen, “Metric3D: Towards zero-shot metric 3D prediction from a single image,” arXiv:2307.10984, 2023.

[20] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth Anything 3: Recovering the visual space from any views,” arXiv:2511.10647, 2025.

[22] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, “Continuous 3D perception model with persistent state,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[23] R. Mur-Artal and J. D. Tardós, “ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras,” IEEE Trans. Robot., vol. 33, no. 5, pp. 1255–1262, 2017.

[24] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. M. Montiel, and J. D. Tardós, “ORB-SLAM3: An accurate open-source library for visual, visual-inertial, and multimap SLAM,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890, 2021.

[25] X. Gao, R. Wang, N. Hu, F. Zhu, and D. Cremers, “LDSO: Direct sparse odometry with loop closure,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2018.

[26] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2023, pp. 39 033–39 051.

[27] L. Lipson, Z. Teed, and J. Deng, “Deep patch visual SLAM,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 424–440.

[28] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[29] P. W. Holland and R. E. Welsch, “Robust regression using iteratively reweighted least-squares,” Commun. Stat., Theory Methods, vol. 6, pp. 813–827, 1977.

[30] L. Antsfeld, B. Chidlovskii, Y. Cabon, V. Leroy, and J. Revaud, “S-MUSt3R: Sliding multi-view 3D reconstruction,” arXiv:2602.04517, 2026.

[31] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, “MUSt3R: Multi-view network for stereo 3D reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 1050–1060.

[32] Z. Chen, M. Qin, T. Yuan, Z. Liu, and H. Zhao, “LONG3R: Long sequence streaming 3D reconstruction,” arXiv:2507.18255, 2025.

[33] C. Cheng, X. Chen, T. Xie, W. Yin, W. Ren, Q. Zhang, X. Guo, and H. Wang, “LongStream: Long-sequence streaming autoregressive visual geometry,” arXiv:2602.13172, 2026.

[34] T. Ding, Y. Xie, Y. Liang, M. Chatterjee, P. Miraldo, and H. Jiang, “LASER: Layer-wise scale alignment for training-free streaming 4D reconstruction,” arXiv:2512.13680, 2025.

[35] ByteDance Seed, “Depth Anything 3: da3\_streaming (code),” https://github.com/ ByteDance-Seed/Depth-Anything-3/tree/main/da3\_streaming, 2026, accessed: Mar. 4, 2026.

[36] A. Myronenko and X. Song, “Point set registration: Coherent point drift,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 32, no. 12, pp. 2262–2275, 2010.

[37] B. Jian and B. C. Vemuri, “Robust point set registration using gaussian mixture models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 8, pp. 1633–1645, 2010.

[38] X. Zhang, J. Q. Yang, S. Zhang, and Y. Zhang, “FastMAC: Stochastic spectral sampling of correspondence graph for point cloud registration,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[39] Stanford University, “The Stanford 3D scanning repository,” https://graphics. stanford.edu/data/3Dscanrep/, accessed: Sep. 16, 2026.

[40] Q.-Y. Zhou, J. Park, and V. Koltun, “Fast global registration,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016.

[41] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the KITTI vision benchmark suite,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 3354–3361.

[42] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012, pp. 573–580.