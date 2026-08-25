# Spotter: Efficient Urban Visual Localization via Geo-Referenced Facade Landmarks in GPS-Degraded Environments

Antoni Valls , Jordi Sanchez-Riera

Abstract—Accurate visual localization on robotic and wearable platforms remains challenging in dense urban environments. Existing methodologies typically rely on GPS for absolute positioning, yet GPS signals frequently degrade in urban canyons due to multipath propagation. Consequently, standard solutions like visual odometry suffer from unmitigated drift over time, while map-matching techniques struggle to acquire the reliable GPS priors they need, on top of being too computationally heavy for real-time edge execution. To address these limitations, we propose Spotter, a robuts and real-time visual localization framework that uses building facades as a reliable source of global georeference, while retaining the capability to integrate GPS signals when available. In an offline stage, Spotter processes Google Street View panoramas by semantically segmenting facades and pairing multi-view stereo depth with cartographic data to build a compact metric database. At runtime, query images are matched via a cascaded retrieval and geometric verification pipeline to recover fine-grained global camera localization. We benchmark Spotter on a newly collected dataset of pedestrian sequences acquired with wearable smart glasses across several districts of Barcelona. Experimental results show that Spotter outperforms odometry-based baselines and achieves localization accuracy comparable to state-of-the-art map-based methods while operating at significantly higher frame rates.

Index Terms—Assistive pedestrian navigation, Visual localization, Computer vision, Smart cities.

## I. INTRODUCTION

R OBOT navigation relies on precise visual localization to enable motion planning and seamless interaction with the surrounding environment. In urban settings, localization remains particularly challenging due to dynamic scenes, frequent occlusions, and the scale of the environment. These difficulties are most acute for lightweight robotic platforms and wearable devices with constrained computational resources. Whereas autonomous vehicles [1], [2] can afford a rich sensor suite—high-precision Global Positioning System (GPS) receivers, inertial sensors, LiDAR, and detailed high-definition maps—lightweight platforms [3]–[5] such as smart glasses and smartphones operate under far stricter limits on sensing, computation, and energy.

To operate within these strict resource limits, most practical navigation systems pivot toward Visual Odometry (VO) [6], [7] or Visual Simultaneous Localization and Mapping (SLAM) [8]–[10], which operate efficiently on embedded and wearable platforms through incremental feature-based motion estimation. However, this incremental formulation inevitably accumulates localization drift over time. The standard mechanisms for drift correction are poorly suited to autonomous navigation, particularly in pedestrian scenarios. Loop closures are unlikely to occur along the short, exploratory, and non-repetitive trajectories typical of pedestrian motion; GPS-based correction can be unreliable in dense urban environments due to signal blockage and multipath effects; and digital maps may be incomplete or inaccurate. Consequently, although these methods are computationally efficient and lightweight, they remain prone to unbounded localization drift in the absence of a reliable global reference, making accurate recovery of the global trajectory increasingly difficult.

![](images/28cb43e69f7cdeb0c13a30b65baa8ebd02ac6d0a7dfbeb664beb9aae5f0caf1f.jpg)  
Fig. 1: Accuracy–speed trade-off of different localization paradigms on our urban benchmark dataset. The y-axis reports the average position error, while the x-axis shows the inference speed in frames per second (FPS). Spotter achieves competitive localization accuracy while operating at significantly higher frame rates than existing map-based approaches. In addition, Spotter is robust to GPS degradation (see Section IV-E), whereas the performance of GPS-assisted methods deteriorates as GPS uncertainty increases. Note that the reported accuracy of DPVO is obtained after global alignment to the ground-truth trajectory, which is not available during online operation. Where applicable, methods are shown in their GPS-aided configuration; see Table II for the corresponding values.

To provide such a global reference, recent approaches have explored map-based image localization, estimating the camera pose independently at individual observations rather than incrementally over a complete trajectory [11]–[13]. These methods typically project the query image into a bird’s-eye-view representation and match it against rasterized OpenStreetMap (OSM) tiles, enabling pose estimation from lightweight cartographic maps. However, matching the query against a large set of potential candidate locations requires substantial computation, which can increase inference time. Moreover, these methods also rely on a GPS prior to constrain the search space and select candidate map tiles for matching, which can introduce GPS errors of tens of meters [14], potentially expanding or misdirecting the search region. Consequently, localization performance can become sensitive to GPS inaccuracies, while the need to evaluate a larger set of candidate locations can further limit their suitability for real-time trajectory estimation.

A related approach formulates visual localization as an image retrieval problem. These approaches use pre-collected databases of geo-referenced images, typically obtained from Google Street View (GSV) or similar sources, and retrieve the most visually similar database image using global image descriptors [15], [16]. Unlike incremental odometry, image retrieval can provide an absolute geographic reference without accumulating trajectory drift. However, this comes at the cost of storing and searching large collections of images, resulting in substantial memory and computational requirements. Furthermore, the retrieval process is not explicitly optimized for efficient large-scale localization, making its computational cost increasingly significant as geographic coverage and database size grow.

These limitations reveal a fundamental trade-off between localization accuracy, computational efficiency, and memory requirements, as illustrated in Figure 1. Odometry and SLAM provide lightweight real-time operation but suffer from accumulated drift, whereas map-based and image-retrieval approaches provide global references at the cost of increased computational requirements and, in some cases, dependence on accurate GPS priors. Urban environments, however, contain a particularly valuable source of persistent visual information: building facades. Unlike transient objects such as pedestrians and vehicles, building facades are relatively stable, spatially distributed, and naturally associated with geographic coordinates. This makes them well suited to serve as visual landmarks for global localization, including in GPS-degraded or GPS-denied environments.

To exploit this property and address the limitations of existing localization approaches, we introduce Spotter, a global visual localization framework that leverages building facades as stable, geo-referenced visual landmarks. Spotter constructs a compact representation that combines discriminative facade features with metric world coordinates, enabling efficient image-based localization at city scale. Its design bridges the global consistency of map-based localization and the efficiency of feature-based visual matching, while avoiding the need for dense 3D reconstruction or continuous trajectory integration.

Spotter constructs its offline database from GSV panoramas by first semantically segmenting building facades and then extracting robust visual features associated with metric world coordinates using cartographic information and multiview stereo depth estimation. At runtime, features extracted from a street-view query image are processed through a cascaded retrieval and geometric verification pipeline to efficiently identify corresponding facade landmarks and estimate the camera’s global pose. GPS can optionally be used as a coarse spatial prior to restrict the search area, but it is not required for pose estimation, which is determined entirely from visual correspondences. Consequently, Spotter can operate with degraded GPS measurements or in a fully GPS-free configuration, providing fine-grained, drift-free localization at real-time frame rates within the computational and memory constraints of wearable platforms.

Our contributions can be summarized as:

• A scalable offline pipeline that converts Google Street View (GSV) panoramas into compact, geo-referenced facade maps for efficient global pose estimation, eliminating the need for dense, computationally heavy Structurefrom-Motion (SfM) reconstructions.

• A robust, real-time, and modular visual localization framework for wearable urban navigation that integrates seamlessly into existing navigation pipelines and supports both GPS-aided and GPS-free operation with minimal computational overhead.

• A new benchmark dataset for pedestrian visual localization, acquired using a wearable smart-glasses platform and comprising 20 challenging trajectories across diverse urban environments in several districts of Barcelona.

We conduct a comprehensive experimental evaluation on a pedestrian dataset collected using wearable smart glasses. In particular, we compare Spotter against representative VO/SLAM, map-matching, and retrieval baselines. We further evaluate GPS-free operation and robustness to increasing levels of synthetic GPS noise. The results demonstrate that Spotter achieves competitive or superior localization accuracy while providing substantially higher runtime efficiency and maintaining stable performance under degraded or absent GPS signals.

## II. RELATED WORK

## A. VO/SLAM Localization

VO and SLAM aim to estimate camera motion sequentially matching features across consecutive image frames. Unlike SfM approaches [17]–[21] that require large memory to store the reconstructed 3D dense point clouds, these methods execute efficiently with a minimal memory footprint. However, their primary limitation is the accumulation of drift over time, which eventually leads to trajectory degradation and loss of tracking. Recent advances have focused on enhancing featurematching robustness, as seen in ORB-SLAM3 [8] and MAC-VO [7], aligning local SLAM point clouds against city-scale 3D meshes via Iterative Closest Point (ICP) [9], or utilizing dense optical flow across image sequences as in DPVO [6]. Alternatively, multimodal approaches integrate complementary sensory inputs alongside visual features, IMUs, and GPS receivers [22]. Most recently, 3D Gaussian Splatting has been explored as a scene representation for downstream mapping and pose estimation [23], [24]. Despite these advancements, incremental frameworks remain fundamentally vulnerable to unbounded drift—particularly when GPS signals are degraded—rendering long-range trajectory correction challenging without reliable global anchors.

## B. Map-based Localization

Digital cartography provides a rich global cue that remains largely unexploited by standard VO and SLAM pipelines.

Cartographic representations offer a key structural advantage: storing vector geometries requires an extremely compact memory footprint compared to 3D point clouds. To leverage this, recent data-driven frameworks directly match street-level images against learned cartographic feature maps. Specifically, OrienterNet [12] transforms query images into bird’s-eye-view (BEV) features to match against rasterized OSM tiles, achieving fine-grained 3-DoF pose recovery and leveraging temporal probability fusion for improved recall. OSMLoc [13] enhances cross-city generalization by combining OSM semantic embeddings with geometry-guided depth estimation, while SNAP [11] aggregates overhead and groundlevel imagery into neural map representations for global-scale retrieval. Nevertheless, while these frameworks operate on lightweight maps and yield precise localization under accurate GPS priors, dense feature matching against 2D cartographic grids is computationally expensive. When GPS signals are degraded or unavailable, the required search radius expands dramatically, severely slowing down execution and limiting real-time deployment on wearable hardware.

## C. Retrieval-based Localization

Rather than relying on learned vector map features, an alternative paradigm leverages digital cartographic maps to organize geo-referenced database imagery, such as GSV panoramas, for image retrieval. Classic Visual Place Recognition (VPR) methods like NetVLAD [15], [25] represent scenes using sets of geo-tagged images, estimating global pose by identifying the nearest visual neighbor via global image descriptors. Foundation-model VPR approaches such as AnyLoc [26] extend this unconstrained retrieval paradigm, approximating the query pose directly using the database coordinates of the top-retrieved GSV panorama without requiring positional priors or downstream pose refinement. To achieve higher metric accuracy, methods such as Li et al. [16] perform block-wise semantic segmentation on query images to match facade regions against GSV panoramas, subsequently refining the 6-DoF pose via bundle adjustment anchored to reference GSV coordinates. Nevertheless, storing dense image collections for wide geographic regions imposes significant memory overhead, while global query matching remains computationally expensive and often reliant on coarse GPS priors to bound the search domain.

## III. SPOTTER

Given a street-level query image $\mathbf { I } _ { t } ,$ we aim to estimate the camera’s 3-DoF global pose $\pmb { \xi } _ { t } = ( \mathbf { g } _ { t } , \boldsymbol { \theta } _ { t } ) \in \mathbb { R } ^ { 2 } \times ( - \pi , \pi ]$ where $\mathbf { g } _ { t } ~ = ~ ( e _ { t } , n _ { t } )$ denotes the UTM metric coordinates (easting, northing) and $\theta _ { t }$ is the global heading. To achieve this, we design a two-stage framework (Figure 2): an offline phase (Section III-A) that constructs a lightweight database $\mathcal { D }$ of 3D geo-referenced visual landmarks, followed by an online phase (Section III-B). The online localization engine establishes 2D-3D feature correspondences and solves for the full 6-DoF camera pose $\mathbf { T } _ { t } \in \mathrm { S E } ( 3 )$ via Perspective-n-Point (PnP), from which $\xi _ { t }$ is analytically derived.

## A. Offline Geo-Referenced Database

The construction of the geo-referenced database D follows a three-stage pipeline. First, a strategic viewpoint sampling strategy extracts panoramic imagery across a designated cartographic region. Second, building facades are semantically segmented to constrain the extraction of local and global visual features to persistent structural elements. Third, local 2D image features are projected into 3D metric world coordinates and geometrically verified against cartographic boundaries.

1) Strategic viewpoint sampling: We define a geographic search space $\mathcal { R } ( c _ { 0 } , r )$ centered at geodetic coordinate $c _ { 0 } =$ $\left( \mathrm { l a t } _ { 0 } , \mathrm { l o n } _ { 0 } \right)$ with radius r, and extract the underlying OSM road graph $\mathcal { G } = ( \mathcal { N } , \mathcal { E } )$ . Along each road edge e in $\mathcal { E } ,$ points are sampled at uniform metric intervals $\delta _ { s } = 3 0 \mathrm { m }$ , yielding a set of sample coordinates $S .$

For each point $\mathbf { p } _ { k } \in S , \mathbf { G } \mathbf { S } \mathbf { V }$ perspective crops are retrieved relative to the local road heading $\theta _ { k }$ . Specifically, we extract $N = 4$ views at $9 0 ^ { \circ }$ orthogonal steps for straight segments $( \mathbf { p } _ { k } \in \ S _ { \mathrm { s e g m e n t } } )$ , and $N = 8$ views at $4 5 ^ { \circ }$ directional steps for intersections $\left( \mathbf { p } _ { k } \in S _ { \mathrm { i n t e r s e c t i o n } } \right)$ . If the returned panorama position $\mathbf { p } _ { k } ^ { G S V }$ differs from the target sample point $\mathbf { p } _ { k }$ , the database coordinate is updated to match the true GSV metadata anchor $\mathbf { p } _ { k } ^ { G S V }$

2) Building segmentation and feature extraction: To eliminate dynamic visual clutter $( \mathrm { e . g . }$ , vehicles and vegetation), each retrieved GSV image $\mathbf { I } _ { i }$ at sampling location $\mathbf { p _ { k } }$ is semantically segmented using LangSAM [27] to yield a binary building facade mask. Within the segmented facade region, we extract visual building landmarks using SuperPoint [28], represented as keypoint-descriptor pairs $\bar { \{ ( \mathbf { x } _ { i j } , \mathbf { d } _ { i j } ) \} } _ { j = 1 } ^ { K _ { i } }$ , where $K _ { i }$ denotes the number of keypoints extracted from image $\mathbf { I } _ { i }$ (capped at a maximum of 10,000), $\mathbf { x } _ { i j } ~ \in ~ \mathbb { R } ^ { 2 }$ denotes 2D image coordinates and $\mathbf { d } _ { i j } \in \mathbb { R } ^ { 2 5 6 }$ is the corresponding local feature descriptor. In parallel, a global image descriptor $\mathbf { f } _ { i } \in \mathbb { R } ^ { 4 0 9 6 }$ is extracted via NetVLAD [15]. The global vector $\mathbf { f } _ { i } ,$ , local features, and camera metadata $\mathcal { M } _ { i }$ are indexed to enable coarse-to-fine visual retrieval during online execution.

3) Geo-reference 3D features: Each 2D keypoint $\mathbf { x } _ { i j } \in \mathbb { R } ^ { 2 }$ is projected into a 3D cartographic landmark $\mathbf { X } _ { i j } \in \mathbb { R } ^ { \mathrm { j } }$ using per-pixel metric depth and an associated confidence score $s _ { i j }$ predicted by a neural multi-view 3D estimator [29]. To minimize database storage overhead and eliminate geometrically spurious estimates, candidate 3D landmarks undergo an OSMguided spatial filtering stage illustrated in Figure 3. Specifically, points whose 2D planar ground projections exceed a distance threshold $\tau _ { \mathrm { d i s t } }$ from actual building facade boundaries retrieved from OSM are pruned. This spatial verification step guarantees that only high-confidence facade landmarks are retained in the database.

Finally, the resulting geo-referenced database D consists of viewpoint tuples $\boldsymbol { \mathcal { V } } _ { i } = \left( \mathbf { f } _ { i } , \boldsymbol { \mathcal { M } } _ { i } , \{ ( \mathbf { x } _ { i j } , \mathbf { d } _ { i j } , \mathbf { X } _ { i j } , s _ { i j } ) \} _ { j = 1 } ^ { K _ { i } } \right)$ collected across all sampled road graph locations $\mathbf { p } _ { k } \in S$

For a typical urban area of $\textit { A } = \ 1 \mathrm { k m ^ { 2 } }$ , the generated database comprises approximately $V ~ \approx ~ 7 , 0 0 0$ viewpoint entries. With an average storage size of ∼ 100 KB per viewpoint, the entire regional database occupies roughly 700 MB, maintaining a compact memory footprint suitable for edge deployment.

![](images/7bd38eade22edefaef1979ef913d01730a0217a7fde7fa4bd9e1e7447a75741e.jpg)  
Fig. 2: Overview of Spotter. Top: the offline phase segments GSV panoramas, extracts keypoints, and lifts them to 3D world coordinates, building a compact per-viewpoint feature database and a street graph. Bottom: at runtime a query image and an optional GPS signal drive candidate retrieval, feature matching, and pose estimation, with a Kalman filter smoothing the result. Connector and box color indicate which operating mode uses each path (see legend); the two modes share the same visual matching core and differ only in the spatial prior, the heading filter, and the propagation of the prior across frames.

## B. Online Localization Engine

At runtime, the system localizes a temporally ordered sequence of query images indexed by the frame index $t \ : =$ $1 , 2 , \ldots \mathrm { A t }$ each step t, the model receives a query image $\mathbf { I } _ { t }$ alongside a positional prior $\pi _ { t } { - } \mathrm { a n }$ estimate of the camera’s location used to seed the candidate search—and returns an estimated global camera pose $\xi _ { t }$ . The architecture supports two distinct execution modes that differ in how $\pi _ { t }$ is obtained: SpotterGPS, which uses the incoming GPS reading at frame t as the prior (with estimated uncertainty $\sigma _ { t } ^ { \mathrm { { G P S } } } ) ;$ and Spotter, which uses the previous Kalman filter (KF) state $\pmb { \pi } _ { t } = \hat { \bf g } _ { t - 1 } ^ { + }$ as the prior for the candidate search region. Under both modes, each query image is localized against the reference database D through a four-stage pipeline: (1) cascaded candidate retrieval, (2) feature matching and pose estimation, (3) Kalman filter position smoother, and (4) Operating modes and gating mechanism.

1) Cascaded candidate retrieval: In a similar manner to the geo-referenced database construction, $K _ { t } = 1 , 0 2 4$ Super-Point keypoint-descriptor pairs $\{ ( \mathbf { x } _ { t j } , \mathbf { d } _ { t j } ) \} _ { j = 1 } ^ { K _ { t } }$ and a global

NetVLAD descriptor $\mathbf { f } _ { t }$ are extracted from the query image for downstream candidate matching. The pool of reference viewpoints $\nu _ { i }$ is then refined through three successive filtering stages to prune candidate space prior to computationally intensive feature matching. First, a spatial filter retains only reference viewpoints located within a search radius $r _ { t }$ centered on the current positional prior $\pi _ { t }$

The search radius $r _ { t }$ is defined as:

$$
r _ { t } = { \left\{ \begin{array} { l l } { 1 5 . 0 \cdot \lambda _ { t } + \sigma _ { t } ^ { \mathrm { G P S } } } & { { \mathrm { S p o t t e r G P S } } , } \\ { 3 8 . 0 \cdot \lambda _ { t } } & { { \mathrm { S p o t t e r } } , } \end{array} \right. }\tag{1}
$$

where $\sigma _ { t } ^ { \mathrm { G P S } }$ is the estimated GPS uncertainty of the current prior and $\lambda _ { t } \in [ 0 . 5 , 2 . 0 ]$ is an adaptive scale factor initialized at $\lambda _ { 1 } = 1 . 0$ . The scale factor adapts online:

$$
\lambda _ { t + 1 } = { \left\{ \begin{array} { l l } { \operatorname* { m a x } ( 0 . 5 , \ \lambda _ { t } \times 0 . 8 5 ) } & { { \mathrm { i f ~ f r a m e ~ } } t \ \operatorname { s u c c e e d s } , } \\ { \operatorname* { m i n } ( 2 . 0 , \ \lambda _ { t } \times 1 . 3 0 ) } & { { \mathrm { i f ~ f r a m e ~ } } t \ \operatorname { f a i l s } , } \end{array} \right. }\tag{2}
$$

shrinking the window after each success and widening it after each failure. The asymmetric factors $( \times 0 . 8 5 ~ \mathrm { v s . } \ \times 1 . 3 0 )$ cause the radius to reach its floor after approximately five consecutive successes and its ceiling after three consecutive failures. In SpotterGPS the spatial filter is a Euclidean radius filter, while in Spotter it is replaced by the road-graphconstrained filter of Section III-B4.

![](images/0387402d614e0b44c1daf233f2a686f955c37e8663a846f22d048b65fc331f24.jpg)  
Fig. 3: Offline database construction. Left: SuperPoint keypoints extracted within the LangSAM building-facade mask on a GSV image. Right: the same keypoints projected onto an OSM tile with their estimated 3D world positions; points discarded by the OSM facade-distance filter are shown in red.

A heading filter then discards references whose capture heading deviates from the estimated viewing direction by more than 120<sup>◦</sup>; the heading is estimated from the GPS displacement history over a sliding window of roughly 6 s and requires at least 2 m of cumulative displacement before it is trusted, so this filter is active only in SpotterGPS and is bypassed in Spotter. Finally, a visual filter keeps the top-$K _ { \mathrm { f i l t e r } }$ references by cosine similarity between the query and reference NetVLAD descriptors, reducing the candidate set to the most appearance-compatible viewpoints.

2) Feature matching and pose estimation: Candidate viewpoints $\nu _ { i }$ are subsequently processed in batches, sorted in descending order of visual similarity. For each candidate viewpoint, visual features are matched against the query image using LightGlue [30]. To remove false matches, geometric verification is performed per candidate by estimating a homography via MAGSAC [31], filtering out correspondence outliers. Processing stops early once enough strongly-matched candidates have been found. The matched candidates are scored by a combined metric of inlier count and average match confidence, boosted by spatial-neighbor agreement; the top-scoring candidate is selected, and its 2D–3D correspondences are aggregated with those of up to eight spatially neighboring viewpoints that also matched the query. The final pose is recovered by applying the SQPnP solver [32] inside RANSAC [33].

3) Kalman filter position smoother: A 2D Kalman filter tracks the estimated camera position $\hat { \textbf { g } } = [ e , n ] ^ { \top }$ (easting, northing in UTM) with covariance $P \in \mathbb { R } ^ { 2 \times \bar { 2 } }$ across frames. The filter is initialized on the first accepted localization with

$$
\begin{array} { r } { \hat { \mathbf { g } } _ { 0 } = \left[ e _ { 0 } , n _ { 0 } \right] ^ { \top } , \qquad P _ { 0 } = \sigma _ { p _ { 0 } } ^ { 2 } I , } \end{array}\tag{3}
$$

where $\sigma _ { p _ { 0 } } = 2 0$ m for both SpotterGPS and the Spotter coldstart.

a) Predict: The prediction model differs between modes. In SpotterGPS, the state evolves as an undirected random walk:

$$
\hat { \bf g } ^ { - } = \hat { \bf g } , \qquad P ^ { - } = P + \sigma _ { q } ^ { 2 } I ,\tag{4}
$$

where $\sigma _ { q } ~ = ~ v _ { \operatorname* { m a x } } \Delta t , ~ v _ { \operatorname* { m a x } } ~ = ~ 2 . 5$ m/s is a conservative walking-speed bound, and $\Delta t$ is the inter-frame time interval.

When a reliable heading $\theta ^ { \mathrm { G P S } }$ is available from GPS displacement history, a directed variant is used instead, where $\bar { v ^ { \mathrm { G P S } } }$ is the GPS-measured speed capped at $v _ { \mathrm { m a x } }$

$$
\hat { \mathbf { g } } ^ { - } = \hat { \mathbf { g } } + \left[ v ^ { \mathrm { G P S } } \Delta t \sin \theta ^ { \mathrm { G P S } } \right] , \qquad P ^ { - } = P + \sigma _ { q } ^ { 2 } I .\tag{5}
$$

In Spotter, the state is propagated using the inter-frame displacement $[ \Delta x ^ { \mathrm { o d o } } , \Delta y ^ { \mathrm { o d o } } ] ^ { \top }$ provided by an external odometry source, aligned to the UTM frame via a rotation matrix $R _ { \mathrm { o d o } }$ estimated online by Procrustes analysis:

$$
{ \hat { \bf g } } ^ { - } = { \hat { \bf g } } + R _ { \mathrm { o d o } } \left[ \Delta x ^ { \mathrm { o d o } } \right] , \qquad { \cal P } ^ { - } = { \cal P } + \sigma _ { q } ^ { 2 } { \cal I } ,\tag{6}
$$

with $\sigma _ { q } = 0 . 5$ m (trusted odometry), or falling back to $\sigma _ { q } =$ $v _ { \mathrm { m a x } } \sqrt { \Delta t _ { \mathrm { s u c c } } \cdot \Delta t }$ when odometry is unavailable, where $\Delta t _ { \mathrm { s u c c } }$ is the elapsed time since the last successful localization. This wider process noise reflects growing positional uncertainty across longer localization gaps. The Procrustes alignment is fit once via singular value decomposition (SVD) as soon as at least five PnP–odometry pairs span a minimum cumulative motion of 5 m, and is then frozen for the remainder of the sequence. Before this alignment is established, all Spotter predictions rely on the undirected random-walk fallback, making the system most vulnerable to early localization failures before sufficient trajectory has been observed.

b) Update: Each accepted PnP estimate $\mathbf { z } = [ e _ { \mathrm { p r e d } } , n _ { \mathrm { p r e d } } ] ^ { \top }$ is fused as a direct position observation:

$$
\begin{array} { c } { { { \cal K } = P ^ { - } \left( P ^ { - } + \sigma _ { r } ^ { 2 } I \right) ^ { - 1 } , \qquad P ^ { + } = \left( I - K \right) P ^ { - } , } } \\ { { { \hat { \bf x } } ^ { + } = { \hat { \bf g } } ^ { - } + K ( { \bf z } - { \hat { \bf g } } ^ { - } ) . } } \end{array}\tag{7}
$$

where the measurement noise is derived from PnP quality:

$$
\sigma _ { r } = \mathrm { c l i p } \left( \frac { 3 . 0 \times \mathrm { R M S E _ { p x } } } { \sqrt { N _ { \mathrm { i n l i e r s } } } } , 2 , 5 0 \right) \mathrm { m } .\tag{8}
$$

High reprojection error or few inliers produce a large $\sigma _ { r } ,$ causing the filter to down-weight the measurement and lean on the prediction.

4) Operating modes and jump gate: In SpotterGPS, the prior $\pi _ { t }$ for frame t is set directly to the current GPS reading. Each frame is therefore anchored independently to a fresh GPS measurement, so a failed prediction on frame t has no effect on frame t+1. The heading filter is active, using the GPS displacement history over the preceding ∼6 s to estimate the viewing direction, and the KF output smooths the trajectory without replacing the GPS-derived prior.

In Spotter, no GPS signal is available. The prior $\pi _ { t }$ for frame t is the previous filtered position $\hat { \mathbf { g } } _ { t - 1 } ^ { + }$ . Candidate retrieval is additionally constrained by the prebuilt GSV street graph: on straight segments, candidates are restricted to a forward/backward extension along the detected street; at intersections, a Dijkstra traversal explores all reachable nodes within the search radius. Because the prior is chained across frames, an incorrect prediction is absorbed into the KF state and can compound into subsequent searches, making outlier rejection critical.

TABLE I: Per-zone statistics of the Barcelona wearable dataset.
<table><tr><td>Zone</td><td>Seq.</td><td>Frames</td><td>Dur. (min)</td><td>Length (m)</td><td>GPS err. (mean, m)</td></tr><tr><td>Gràcia</td><td>7</td><td>7041</td><td>23.6</td><td>1825</td><td>13.0</td></tr><tr><td>Guinardó</td><td>6</td><td>4323</td><td>13.9</td><td>1061</td><td>3.8</td></tr><tr><td>Poblenou</td><td>7</td><td>4252</td><td>14.4</td><td>1093</td><td>3.9</td></tr><tr><td>Total</td><td>20</td><td>15616</td><td>51.9</td><td>3979</td><td>7.0</td></tr></table>

Both modes enforce a final gating mechanism that rejects any pose estimate whose spatial deviation from the predicted prior exceeds:

$$
\tau _ { t } = \left\{ \begin{array} { l l } { \sigma _ { t } ^ { \mathrm { G P S } } + v _ { t } ^ { \mathrm { g a t e } } \Delta t + m _ { \mathrm { P n P } } } & { \mathrm { S p o t t e r G P S , } } \\ { v _ { \mathrm { m a x } } \Delta t _ { \mathrm { s u c c } } + m _ { \mathrm { P n P } } } & { \mathrm { S p o t t e r , } } \end{array} \right.\tag{9}
$$

where $v _ { t } ^ { \mathrm { g a t e } } = \mathrm { m i n } ( v _ { t } ^ { \mathrm { G P S } } \times 1 . 5 , v _ { \mathrm { m a x } } )$ m/s is the speed estimate with a safety margin, ∆t is the inter-frame interval, $\Delta t _ { \mathrm { s u c c } }$ is the time elapsed since the last successful localization, and $m _ { \mathrm { P n P } } = 2 2$ m is a fixed margin accounting for single-view PnP residual error. A prediction is rejected if $\| \hat { \mathbf { g } } _ { t } ^ { + } - \hat { \mathbf { g } } _ { t - 1 } ^ { + } \| _ { 2 } > \tau _ { t }$ In Spotter, this gate is the primary safeguard against corrupting the chained KF state with a gross outlier.

All hyperparameters were determined empirically on a heldout validation subset of the Barcelona dataset and kept fixed across all reported experiments.

## IV. EXPERIMENTS

## A. Dataset Construction

To evaluate Spotter in dense urban settings, we introduce a novel pedestrian-centric dataset curated specifically to capture continuous building facades across diverse streets in the city of Barcelona with different levels of GPS signal accuracy. Standard visual localization benchmarks—such as automotive datasets (KITTI [34], nuScenes [35]) or micro-aerial vehicle datasets (EuRoC MAV [36])—either prioritize road-level perspectives with distant, occluded structures or focus on indoor and aerial domains. To bridge this gap, our dataset was recorded from a pedestrian viewpoint using wearable smart glasses<sup>1</sup> equipped with a ZED stereo camera system, capturing HD images (1280×720 px) at 5 FPS to yield dense, persistent coverage of facade geometry.

The dataset comprises 20 sequences spanning three Barcelona neighborhoods—Gracia (\` 7 sequences), Guinardo´ (6), and Poblenou (7)—for a total of 15,616 frames, roughly 52 minutes of recording, and close to 4 km of cumulative pedestrian trajectory. The three zones differ in street layout and building density, providing varied facade appearance and a range of localization difficulties. Raw GPS positioning data was logged concurrently using an Android mobile application running during data acquisition. Table I summarizes the perzone statistics.

A key feature of our dataset is its heterogeneous GPS quality across urban environments. While Guinardo and Poblenou´ maintain strong reception (mean horizontal errors of 3.8 m and 3.9 m), Gracia exhibits severe signal degradation due to\` its narrow streets and high building density. Errors within Gracia range from \` 6–7 m in Sequences 01–02 up to 14–20 m in Sequences 16–20 under heavy multipath conditions. Overall, Gracia achieves a mean error of\` 13.0 m, contributing to a dataset-wide mean of 7.0 m—a trend consistent with established GPS error metrics in urban canyons [14].

Ground-truth trajectories are available for all sequences. Each trajectory was manually annotated by cross-referencing the onboard GPS track with reference map data, providing complete ground-truth coverage across all 20 sequences and enabling consistent metric evaluation throughout every evaluation zone.

## B. Evaluation Protocol and Metrics

All methods are evaluated against the manually annotated ground-truth trajectories using the following metrics:

a) Absolute Position Error (APE).: The APE at frame t is the Euclidean distance between the estimated global position $\hat { \mathbf { g } } _ { t } ^ { + } \in \mathbb { R } ^ { 2 }$ and the ground-truth position $\mathbf { g } _ { t } ^ { \mathbf { G T } } \in \mathbb { R } ^ { 2 }$ in the local UTM frame:

$$
\begin{array} { r } { \mathrm { A P E } ( t ) = \left. \hat { \mathbf { g } } _ { t } ^ { + } - \mathbf { g } _ { t } ^ { \mathrm { G T } } \right. _ { 2 } . } \end{array}\tag{10}
$$

b) Recall at distance thresholds.: To characterize the tail of the error distribution, we report recall at 3, 5, and 10 m (R@3 m, R@5 m, R@10 m), defined as the fraction of frames for which the estimated position falls within the corresponding radius of the ground truth.

c) Efficiency.: We report the processing rate in frames per second (FPS), measured on the same hardware used for all experiments.

## C. Baselines

We benchmark Spotter against representative baselines across each localization paradigm using their default parameter configurations, executing all experiments on a workstation equipped with a single NVIDIA GeForce RTX 4090 GPU. Because several baselines perform single-frame global localization rather than continuous trajectory tracking, we additionally evaluate variants augmented with a GPS spatial prior (denoted as MethodGPS). We group all evaluated methods into five distinct categories:

• Proposed Method: Spotter

• Sensor-Based: ZED [37]

• VO/SLAM: TartanVO [38], DPVO [6], MASt3R-SLAM [10]

• Map-Based: OrienterNet [12], OSMLoc [13]

• Retrieval-Based: AnyLoc [26]

## D. Overall Results

Table II reports overall results across all 20 sequences. Among GPS-aided methods, OSMLocGPS achieves the lowest mean APE (6.89 m) and the highest recall at all thresholds (R@3m = 23.9%, R@5m = 46.3%, R@10m = 80.9%), benefiting from the reliable GPS signal that dominates the dataset. SpotterGPS achieves a mean APE of 8.13 m with R@10m of 71.6%, outperforming OrienterNetGPS (10.00 m) while running at 45.5 FPS—more than three times faster than the map-based baselines. However, as we show in Section IV-E, the OSMLoc advantage collapses rapidly as GPS quality degrades, while SpotterGPS degrades far more gracefully.

![](images/413536803de0618d249294d0c4dcf9fd8af444aa8ba7d9e2119a771cb8e563af.jpg)  
Fig. 4: Qualitative trajectory comparison across four sequences from different Barcelona neighborhoods under real GPS conditions. We show one representative method per paradigm. SpotterGPS (dark blue) consistently tracks the ground truth (black dashed), while DPVO (pink) accumulates drift and ZEDGPS (yellow) is limited by the same GPS multipath affecting each sequence. OSMLocGPS’s performance (green) closely reflects the GPS accuracy of each sequence — performing well where reception is good and degrading where multipath is severe.

TABLE II: Overall localization results (20 sequences, 15,616 frames). VO/SLAM methods are Sim(2)-aligned before APE is computed. Bold: best per group.
<table><tr><td>Method</td><td>APE↓(m)</td><td>R@3m↑</td><td>R@5m↑</td><td>R@10m↑</td><td>FPS↑</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpotterGPS</td><td>8.13</td><td>15.7</td><td>32.0</td><td>71.6</td><td>45.5</td></tr><tr><td>Spotter</td><td>12.17</td><td>18.1</td><td>35.5</td><td>64.4</td><td>26.0</td></tr><tr><td>Sensor-based</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ZEDGPS</td><td>12.36</td><td>3.1</td><td>9.9</td><td>48.3</td><td>5.0</td></tr><tr><td>ZED</td><td>15.90</td><td>15.7</td><td>26.5</td><td>51.5</td><td>5.0</td></tr><tr><td>VO/SLAM</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TartanVO</td><td>38.97</td><td>6.9</td><td>13.6</td><td>34.3</td><td>28.1</td></tr><tr><td>DPVO</td><td>9.78</td><td>18.9</td><td>37.7</td><td>65.9</td><td>43.8</td></tr><tr><td>MASt3R-SLAM</td><td>30.75</td><td>8.1</td><td>14.8</td><td>32.9</td><td>0.8</td></tr><tr><td>Map-based</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OrienterNetGPS</td><td>10.00</td><td>7.2</td><td>17.9</td><td>58.9</td><td>13.0</td></tr><tr><td>OrienterNet</td><td>88.88</td><td>7.2</td><td>11.2</td><td>16.1</td><td>10.0</td></tr><tr><td>OSMLocGPS</td><td>6.89</td><td>23.9</td><td>46.3</td><td>80.9</td><td>12.7</td></tr><tr><td>OSMLoc</td><td>45.98</td><td>21.7</td><td>29.0</td><td>36.1</td><td>12.7</td></tr><tr><td>Retrieval-based</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AnyLoc</td><td>121.65</td><td>6.7</td><td>14.7</td><td>30.4</td><td>2.3</td></tr><tr><td>AnyLocGPS</td><td>11.71</td><td>8.4</td><td>21.5</td><td>56.1</td><td>2.0</td></tr></table>

Compared to the sensor-based and standalone VO/SLAM approaches, our method consistently achieves superior localization accuracy while maintaining a high processing frame rate—outperforming even DPVO in computational efficiency. As illustrated in Figure 4, methods that lack mechanisms for mitigating trajectory drift—such as global GPS priors or persistent visual landmarks—are susceptible to the unbounded accumulation of localization error over time. Note that for pure odometry and SLAM baselines, trajectories are scale-aligned to the ground truth via a 2D Similarity Sim(2) transformation using Umeyama alignment prior to computing the APE to ensure a fair comparison.

Among map-based methods, we observe a substantial disparity in performance depending on whether an accurate GPS prior is available. Since these approaches are primarily designed for discrete map-based localization rather than continuous trajectory estimation, the absence of a reliable GPS prior requires the search to cover substantially larger map regions. This increases the number of candidate pose hypotheses and introduces greater visual ambiguity, which can lead to severe mislocalizations. Moreover, evaluating visual-spatial correspondences across such extensive candidate regions incurs considerable computational overhead, substantially reducing inference speed.

The retrieval-based baseline AnyLoc reaches 121.65 m APE, by far the largest error in the table. Because it reports the coordinate of the nearest retrieved panorama and applies no spatial prior, a single ambiguous retrieval can return a viewpoint elsewhere in the city; the repetitive facades of Barcelona’s regular street grid make such confusions frequent. Restricting retrieval to the GPS prior (AnyLocGPS) removes these gross confusions and cuts the error to 11.71 m—a tenfold reduction from a spatial prior alone, with no change to the descriptors or database. Yet recall at tight thresholds stays low (8.4% at 3 m), because retrieval can at best return the nearest reference viewpoint, sampled at 30 m intervals: even with a correct prior it cannot place the camera more finely than the database sampling allows. AnyLocGPS therefore remains well behind SpotterGPS (8.13 m APE, 32.0% R@5m vs. 21.5%) under the same prior, showing that the explicit 2D– 3D correspondences and PnP solver—not the retrieval or the prior—are what deliver metric pedestrian localization.

Figure 4 provides a graphical comparison of the different methods, together with the ground-truth trajectories and GPS measurements, for four sequences from the dataset. Overall, the qualitative results are consistent with the conclusions drawn from the numerical evaluation.

## E. Robustness to GPS Noise

The preceding results reveal a strong correlation between GPS signal quality and overall localization accuracy across all evaluated baselines. For instance, OSMLocGPS performs exceptionally well when provided with an accurate initial prior, as the search region remains tightly constrained near the true location. However, in regions characterized by repetitive urban structures, increasing GPS noise expands the spatial search radius. This introduces perceptual ambiguities and multiple candidate pose hypotheses, ultimately leading to higher localization errors. These findings motivate a systematic evaluation of method robustness under varying noise regimes, demonstrating the advantage of using persistent building facades as landmark anchors to maintain localization precision.

![](images/4a16d1d51adc9bf78d5792b5b55d9cfbaa931223218034628f8826697c689dec.jpg)

![](images/6482e513c9ac2a71f482904c2e70f1192e4b89b93b0391c6848af2c5c608c127.jpg)

![](images/e6342d6d9263bb3bc43aecba93886814427823689e4e9de2f7c53c7654b5a3ee.jpg)

![](images/b0e0c5d066a3d15bc35afa12c3a0f410b3fd4090e340c74845ec0a0f29a4a676.jpg)  
Fig. 5: GPS noise ablation on a single Gracia sequence with injected noise\` $\sigma \in \{ 0 , 1 0 , 2 0 , 3 0 \}$ m. At σ = 0 m, OSMLocGPS (green) achieves competitive accuracy (6.5 m vs. 7.6 m for SpotterGPS); as noise increases, OSMLocGPS degrades rapidly with erratic jumps and off-road detours, while SpotterGPS (dark blue) remains broadly coherent.

![](images/dfda4183a245b336cbbdc51a0c8d0e8aa8a3a99a8db260c737530d17bb8a00ca.jpg)  
Fig. 6: Mean APE vs. injected GPS noise σ for all GPS-dependent methods. SpotterGPS (purple) remains nearly flat up to σ = 10 m and degrades slowly beyond, while OrienterNetGPS (blue), ZEDGPS (yellow), and OSMLocGPS (green) rise steeply, confirming Spotter’s robustness to GPS degradation.

To isolate the effect of GPS quality from other factors such as scene complexity or sequence length, we conduct a controlled ablation: we inject zero-mean Gaussian noise with standard deviation $\sigma \in \{ 0 , 5 , 1 0 , 2 0 , 3 0 \}$ m into the GPS readings of all sequences and re-evaluate each GPS-dependent method under these degraded priors. This transforms GPS noise from a scene-dependent confound into a controlled experimental variable, letting us directly compare the robustness of different GPS-reliant approaches. Note that for SpotterGPS, larger GPS noise also increases the candidate search radius, which directly raises the number of candidates to evaluate and reduces throughput; this effect is visible in the FPS column of Table III.

Figure 5 shows qualitative trajectory comparisons on a sequence from Gracia; Figure 6 plots performance curves for\` all GPS-dependent methods as a function of injected noise σ; and Table III provides the full numerical breakdown at each noise level.

At σ = 0 m (no added noise), OSMLocGPS leads in recall $( \mathrm { R } @ 3 \mathrm { m } = 2 3 . 9 \% , \mathrm { R } @ 5 \mathrm { m } = 4 6 . 3 \% )$ and APE (6.89 m), while SpotterGPS trails slightly at 8.13 m. As noise increases to $\sigma = 5 \mathrm { m } .$ , SpotterGPS already becomes the best overall (APE 8.21 m, R@3m 14.6%, R@10m 70.3%), while OSMLocGPS drops to 8.62 m APE and R@10m falls from 80.9% to 69.8%. By $\sigma = 1 0 \mathrm { m } ,$ the gap widens decisively: SpotterGPS holds at 8.97 m APE and R@10m 65.9%, while OSMLoc degrades to 12.94 m and OrienterNetGPS to 21.52 m. At $\sigma = \mathrm { 3 0 m - }$ corresponding to the worst multipath conditions observed in Gracia—SpotterGPS achieves\` 13.80 m APE and R@10m of 42.0%, while OrienterNet (44.68 m), OSMLoc (36.43 m), and ZEDGPS (39.84 m) exhibit near-complete localization failure.

TABLE III: GPS noise ablation. APE and recall for GPS-dependent methods at five noise levels. Bold: best at each level.
<table><tr><td>σ (m)</td><td>Method</td><td>APE↓(m)</td><td>R@3m↑</td><td>R@5m↑</td><td>R@10m↑</td><td>FPS↑</td></tr><tr><td rowspan="4">0</td><td>SpotterGPS</td><td>8.13</td><td>15.7</td><td>32.0</td><td>71.6</td><td>45.5</td></tr><tr><td>OrienterNetGPS</td><td>10.00</td><td>7.2</td><td>17.9</td><td>58.9</td><td>13.0</td></tr><tr><td>OSMLocGPS</td><td>6.89</td><td>23.9</td><td>46.3</td><td>80.9</td><td>12.7</td></tr><tr><td>ZEDGPS</td><td>12.36</td><td>3.1</td><td>9.9</td><td>48.3</td><td>5.0</td></tr><tr><td rowspan="4">5</td><td>SpotterGPS</td><td>8.21</td><td>14.6</td><td>31.8</td><td>70.3</td><td>46.3</td></tr><tr><td>OrienterNetGPS</td><td>16.70</td><td>3.6</td><td>8.1</td><td>28.1</td><td>13.0</td></tr><tr><td>OSMLocGPS</td><td>8.62</td><td>18.4</td><td>35.2</td><td>69.8</td><td>12.7</td></tr><tr><td>ZEDGPS</td><td>13.94</td><td>4.2</td><td>11.1</td><td>37.6</td><td>5.0</td></tr><tr><td rowspan="4">10</td><td>SpotterGPS</td><td>8.97</td><td>13.5</td><td>28.1</td><td>65.9</td><td>42.3</td></tr><tr><td>OrienterNetGPS</td><td>21.52</td><td>1.6</td><td>4.5</td><td>17.8</td><td>12.3</td></tr><tr><td>OSMLocGPS</td><td>12.94</td><td>11.3</td><td>22.0</td><td>48.8</td><td>13.0</td></tr><tr><td>ZEDGPS</td><td>17.87</td><td>2.4</td><td>6.7</td><td>24.5</td><td>5.0</td></tr><tr><td rowspan="4">20</td><td>SpotterGPS</td><td>10.87</td><td>9.0</td><td>20.7</td><td>53.7</td><td>38.0</td></tr><tr><td>OrienterNetGPS</td><td>32.72</td><td>0.4</td><td>1.9</td><td>7.2</td><td>12.1</td></tr><tr><td>OSMLocGPS</td><td>24.20</td><td>5.3</td><td>10.8</td><td>25.6</td><td>12.7</td></tr><tr><td>ZEDGPS</td><td>28.30</td><td>0.9</td><td>2.6</td><td>10.0</td><td>5.0</td></tr><tr><td rowspan="4">30</td><td>SpotterGPS</td><td>13.80</td><td>5.2</td><td>13.3</td><td>42.0</td><td>36.1</td></tr><tr><td>OrienterNetGPS</td><td>44.68</td><td>0.2</td><td>1.0</td><td>3.7</td><td>12.0</td></tr><tr><td>OSMLocGPS</td><td>36.43</td><td>3.4</td><td>6.7</td><td>16.1</td><td>12.8</td></tr><tr><td>ZEDGPS</td><td>39.84</td><td>0.4</td><td>1.4</td><td>5.2</td><td>5.0</td></tr></table>

This pattern reflects a fundamental architectural difference: OSMLocGPS and OrienterNetGPS use the GPS coordinate to select the OSM tile against which the query is matched, so a large GPS error directly corrupts the input to the matching stage. Spotter instead uses GPS only to define a candidate search radius; the final pose is determined entirely by visual feature matching and PnP, so the system degrades gracefully and continues to localize correctly even when the prior is tens of meters off. When GPS is perfect or near-perfect, OSMLoc provides slightly higher recall at tight thresholds; but for any realistic urban deployment—where GPS errors of

![](images/a4bba96ec26bcf937856e52370de19ae714cdcbdfefef675f60a0bc1ec4669db.jpg)  
Fig. 7: Qualitative trajectory comparison between Spotter (orange) and ZED (red) against ground truth (black dashed) on two representative sequences, both operating without any GPS signal. Left: ZED drifts onto an adjacent street after a turn, while Spotter recovers and continues to track the correct path. Right: both methods follow the ground-truth trajectory closely, with Spotter achieving lower error along the full sequence.

5–20 m are common due to multipath, tall buildings, and signal blockage—SpotterGPS is consistently preferable. GPS is used as a coarse geographic hint, not as a trusted measurement, and visual evidence always takes precedence in determining the final pose.

## F. GPS Denied Areas

As illustrated in Table II, Spotter achieves a mean APE of 12.17 m with R@10m of 64.4%, outperforming both sensorbased (ZEDGPS: 12.36 m; ZED: 15.90 m) and the remaining GPS-free localization methods, despite using no satellite signals. DPVO remains the only GPS-free baseline with a lower APE (9.78 m), though as noted above its result requires a global Sim(2) alignment to ground truth and is therefore not achievable in an online setting.

Therefore, we visually compare Spotter with the bestperforming GPS-free method, the sensor-based ZED method, to examine the behavior of both approaches in GPS-denied environments. As illustrated in Figure 7, when ZED loses track during the early stages of the trajectory, Spotter remains close to the ground-truth path. Moreover, although Spotter exhibits small amounts of drift in some portions of the trajectory, it is able to subsequently recover and return to the correct path, demonstrating its robustness in the absence of GPS.

## V. CONCLUSIONS

We presented Spotter, a lightweight visual localization framework designed for urban navigation on resourceconstrained wearable platforms. Spotter estimates global camera poses from street-level images by matching query frames against a compact database of geo-referenced Google Street View (GSV) panoramas, bypassing the need for computationally heavy 3D scene reconstructions. A central pillar of its design is the strict decoupling of the GPS signal from visual matching: GPS serves solely as a coarse spatial filter, leaving the metric pose to be determined entirely via 2D–3D correspondences solved through PnP/RANSAC. Consequently, Spotter maintains high accuracy in complex urban environments where traditional tile-based or GPS-dependent retrieval pipelines break down.

Comprehensive evaluation on a wearable dataset spanning three distinct neighborhoods in Barcelona validates our design choices. Under reliable GPS coverage, SpotterGPS matches the accuracy of state-of-the-art map-based methods while delivering significantly lower computational overhead. Controlled ablations under identical spatial priors demonstrate that these performance gains stem directly from our robust pose-estimation pipeline rather than the retrieval prior alone. Furthermore, as GPS signals degrade, Spotter exhibits graceful failure modes, while its GPS-free variant (Spotter) continues to localize effectively without satellite assistance—outperforming the sensor-based pipeline and directly validating permanent building facade features as a sufficient geometric constraint for global metric localization.

Future research will focus on three key enhancements. First, because urban building facades are predominantly planar, the resulting 2D–3D correspondences can become geometrically ill-conditioned; expanding the database to include non-planar structural landmarks such as ground markings and road signage will improve pose estimation stability. Second, explicitly detecting and filtering dynamic occluders, such as pedestrians and parked vehicles, will bolster robustness in densely populated environments. Finally, benchmarking across multicity datasets, seasonal variations, and extreme illumination changes will establish the global scalability and long-term generalization of our approach.

## ACKNOWLEDGMENTS

Project PCPP2021-008760 funded by MCIU/ AEI /10.13039/501100011033 and by the ”European Union NextGenerationEU/PRTR”

## REFERENCES

[1] J. Shang, Y. Liu, Y. Xu, J. Xiao, and D. Ma, “Robust global localization for urban autonomous vehicles via 3d geometric-enhanced visual place recognition,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[2] C. Badue, R. Guidolini, R. V. Carneiro, P. Azevedo, V. B. Cardoso, A. Forechi, L. Jesus, R. Berriel, T. M. Paixao, F. Mutz, L. de Paula˜ Veronese, T. Oliveira-Santos, and A. F. De Souza, “Self-driving cars: A survey,” Expert Systems with Applications, 2021.

[3] L. Gionfrida, D. Kim, D. Scaramuzza, D. Farina, and R. D. Howe, “Wearable robots for the real world need vision,” Science Robotics, 2024.

[4] J. Zhang, X. Yu, S. Ha, P. T. Moron, S. Salimpour, F. Keramat, H. Zhang,´ and T. Westerlund, “Seamless outdoor-indoor pedestrian positioning system with gnss/uwb/imu fusion: A comparison of ekf, fgo, and pf,” ArXiv, 2025.

[5] T. Moisan, H. Fu, V. Renaudin, and M. I. Sayyaf, “From research to app: Personalized inertial navigation for the visually impaired,” in Proceedings of the 2025 International Conference on Indoor Positioning and Indoor Navigation (IPIN), 2025.

[6] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” in Advances in Neural Information Processing Systems, 2023.

[7] Y. Qiu, Y. Chen, Z. Zhang, W. Wang, and S. Scherer, “Mac-vo: Metricsaware covariance for learning-based stereo visual odometry,” in ICRA, 2025.

[8] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: an accurate open-source library for visual,´ visual-inertial and multi-map SLAM,” CoRR, 2020.

[9] R. Merat, G. Cioffi, L. Bauersfeld, and D. Scaramuzza, “Drift-free visual slam using digital twins,” IEEE Robotics and Automation Letters, 2025.

[10] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Realtime dense SLAM with 3D reconstruction priors,” in CVPR, 2025.

[11] P.-E. Sarlin, E. Trulls, M. Pollefeys, J. Hosang, and S. Lynen, “SNAP: Self-Supervised Neural Maps for Visual Positioning and Semantic Understanding,” in NeurIPS, 2023.

[12] P.-E. Sarlin, D. DeTone, T.-Y. Yang, A. Avetisyan, J. Straub, T. Malisiewicz, S. R. Bulo, R. Newcombe, P. Kontschieder, and V. Balntas, “OrienterNet: Visual Localization in 2D Public Maps with Neural Matching,” in CVPR, 2023.

[13] Y. Liao, X. Chen, S. Kang, J. Li, Z. Dong, H. Fan, and B. Yang, “Osmloc: Single image-based visual localization in openstreetmap with geometric and semantic guidances,” arXiv preprint arXiv:2411.08665, 2024.

[14] W. Wen and L.-T. Hsu, “Towards robust GNSS positioning and realtime kinematic using factor graph optimization,” in IEEE International Conference on Robotics and Automation (ICRA), 2021.

[15] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, “Netvlad:´ Cnn architecture for weakly supervised place recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.

[16] Z. Li, S. Li, J. Anderson, and J. Shan, “Urban visual localization of block-wise monocular images with google street views,” Remote Sensing, 2024.

[17] S. Agarwal, N. Snavely, I. Simon, S. M. Seitz, and R. Szeliski, “Building rome in a day,” in ICCV, 2009.

[18] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in CVPR, 2016.

[19] S. Lynen, B. Zeisl, D. Aiger, M. Bosse, J. Hesch, M. Pollefeys, R. Siegwart, and T. Sattler, “Large-scale, real-time visual–inertial localization revisited,” Int. J. Rob. Res., 2020.

[20] L. Liu, H. Li, and Y. Dai, “Efficient global 2d-3d matching for camera localization in a large-scale 3d map,” in ICCV, 2017.

[21] C. Toft, E. Stenborg, L. Hammarstrand, L. Brynte, M. Pollefeys, T. Sattler, and F. Kahl, “Semantic match consistency for long-term visual localization,” in ECCV, 2018.

[22] G. Cai, H. Lin, and S.-F. Kao, “Mobile robot localization using gps, imu and visual odometry,” 2019 International Automatic Control Conference (CACS), pp. 1–6, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:212646948

[23] G. Sidorov, M. Mohrat, D. Gridusov, R. Rakhimov, and S. Kolyubin, “Gsplatloc: Grounding keypoint descriptors into 3d gaussian splatting for improved visual localization,” 2025.

[24] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, 2023.

[25] S. Hausler, S. Garg, M. Xu, M. Milford, and T. Fischer, “Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[26] N. Keetha, A. Mishra, J. Karhade, K. M. Jatavallabhula, S. Scherer, M. Krishna, and S. Garg, “Anyloc: Towards universal visual place recognition,” IEEE Robotics and Automation Letters, 2023.

[27] L. Medeiros, “Lang segment anything,” https://github.com/ luca-medeiros/lang-segment-anything, 2023.

[28] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Selfsupervised interest point detection and description,” CoRR, 2017.

[29] N. Keetha, N. Muller, J. Sch¨ onberger, L. Porzi, Y. Zhang, T. Fischer,¨ A. Knapitsch, D. Zauss, E. Weber, N. Antunes, J. Luiten, M. Lopez-Antequera, S. R. Bulo, C. Richardt, D. Ramanan, S. Scherer, and\` P. Kontschieder, “MapAnything: Universal feed-forward metric 3D reconstruction,” in International Conference on 3D Vision (3DV), 2026.

[30] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local Feature Matching at Light Speed,” in ICCV, 2023.

[31] D. Barath, J. Matas, and J. Noskova, “Magsac: Marginalizing sample consensus,” in CVPR, 2019.

[32] G. Terzakis and M. Lourakis, “A consistently fast and globally optimal solution to the perspective-n-point problem,” in ECCV, 2020.

[33] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Commun. ACM, 1981.

[34] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.

[35] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in CVPR, 2020, pp. 11 621–11 631.

[36] M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik, and R. Siegwart, “The euroc micro aerial vehicle datasets,” The International Journal of Robotics Research,

2016. [Online]. Available: http://ijr.sagepub.com/content/early/2016/01/ 21/0278364915620033.abstract

[37] ZED SDK Documentation: Positional Tracking Overview, Stereolabs, 2026. [Online]. Available: https://www.stereolabs.com/ docs/development/zed-sdk/modules/positional-tracking

[38] W. Wang, Y. Hu, and S. Scherer, “Tartanvo: A generalizable learningbased vo,” in Proceedings of the 2020 Conference on Robot Learning, 2021.

![](images/96749a2956c8091987c218eac1596cbcb4eed047f31ffe89d4099b089bd55b1c.jpg)

Antoni Valls received the B.Sc. degree in theoretical physics from the Universitat de Barcelona, Spain, in 2022, and the M.Sc. degree in data science from the University of Padua, Italy, in 2024. He has previously collaborated with the Institut d’Investigacio´ en Intel·ligencia Artificial (IIIA), CSIC, Barcelona,\` Spain, on natural language processing research. He is currently a Robotic Engineer with the Institut de Robotica i Inform\` atica Industrial (IRI), CSIC-\` UPC, Barcelona, Spain, where he works on the SMARTGAZE II project in collaboration with Biel

Glasses, developing AI-powered assistive systems to enhance the mobility of people with low vision. His research interests include computer vision, visionlanguage models, assistive and pedestrian navigation, localization in dynamic environments, and deep learning.

![](images/e7b3193fe783affe70982a704728f32145798de364a49158599569bd77467e88.jpg)

Jordi Sanchez-Riera is an associate researcher at the Spanish Scientific Research Council (CSIC) in Barcelona. He earned his B.S. in computer science and M.S. in electronic engineering from UPC, and a Ph.D. from the University of Grenoble. His research focuses on robotics, computer vision, and machine learning. He has published in top journals and conferences and received prestigious awards, including a Google Scholarship, Academia Sinica Post-Doc Fellowship, and Juan de la Cierva Incorporacion´ grant.