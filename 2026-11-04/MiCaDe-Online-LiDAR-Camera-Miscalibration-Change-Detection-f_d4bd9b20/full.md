# MiCaDe: Online LiDAR–Camera Miscalibration Change Detection for Autonomous Driving

Carlo Sgaravatti<sup>1,\*</sup>

Riccardo Pieroni<sup>2</sup> Matteo Corno<sup>1</sup> Luca Magri<sup>1</sup> Giacomo Boracchi<sup>1</sup> <sup>1</sup>Politecnico di Milano <sup>2</sup>Niulinx

Sergio M. Savaresi<sup>1</sup>

## Abstract

Reliable multi-sensor perception in autonomous vehicles requires accurate calibration between cameras and LiDAR. However, calibration drift is often unavoidable in long-term deployments: vibrations, thermal effects, and minor mechanical shocks may alter the relative pose between sensors, leading to systematic errors in downstream fusion tasks. We propose MiCaDe (Miscalibration Change Detection), an online solution detecting when the extrinsic LiDAR-camera calibration parameters are no longer accurate by means of sequential change detection. MiCaDe requires no calibration targets, or modification of the perception stack, providing low false-alarm rates with negligible computational overhead. Our key insight is that 3D and 2D bounding boxes produced by standard object detectors should remain geometrically consistent under correct calibration. By matching cross-modal detections, we derive geometric consistency metrics that serve as sensitive indicators of extrinsic drift. These metrics are aggregated into fixed-size batches and scored against a reference distribution, producing a likelihood stream that is monitored by a Change Point Model (CPM). Crucially, the CPM accumulates statistical evidence over time, enabling detection ofsubtle drifts that would otherwise go unnoticed. Experiments on multiple long-sequence datasets and on a real-world dataset with physically induced miscalibration confirm reliable detection under diverse conditions. Code available at: https://github.com/ CarloSgaravatti/MiCaDe.

## 1. Introduction

Reliable perception in autonomous driving relies on accurate fusion between complementary sensors such as cameras and LiDARs, which in turn requires a precise extrinsic calibration. Unfortunately, minor shocks and gradual effects, such as vibrations or temperature changes, can cause calibration drifts that accumulate over time, degrading perception accuracy and compromising navigation safety.

![](images/0f3310222495dfa9dc2e0a11c6a9661b79a33cc1c89be41502926190f8ffafb3.jpg)

![](images/223ad9caa968f1ab9fdbe032939ac4d3a9dc0673afc7d4101ee92aa2476f29e8.jpg)  
Figure 1. Calibration drift detection as a change-detection problem. After a perturbation of the camera extrinsics (black dashed line), the geometric consistency between projected 3D detections (red) and image detections (green) exhibits a degradation in alignment between them, as shown in their IoU (one of the monitored metrics). MiCaDe detects such changes by jointly monitor ing multiple geometric consistency metrics over time.

Detecting when calibration becomes unreliable is therefore critical for long-term autonomous operation. Existing approaches typically formulate miscalibration detection as a frame-level classification problem. They first extract cross-modal cues, such as image edges, depth discontinuities, or learned multimodal features, and then train a classifier to decide whether the sensor pair is calibrated or not [4, 20]. While effective, these approaches present several limitations. First, they usually require dedicated training procedures and synthetic calibration perturbations. Second, they operate independently on each frame, making it difficult to distinguish transient detection noise from persistent calibration drift. Third, they often require specific neural networks, introducing additional computational overhead.

In this work, we introduce MiCaDe (Miscalibration Change Detection), an online monitoring framework for LiDAR-Camera extrinsic calibration. We formulate calibration monitoring as a sequential change-detection problem. Rather than learning whether a sensor pair is calibrated, we explicitly model the distribution of indicators derived from signals that are already available in modern off-the-shelf perception stacks, and monitor these over time. Specifically, we exploit the correspondence between 3D Li-DAR detections and 2D image detections of the same object. Under correct calibration, the geometric relationship between these detections remains statistically stable over time. When the calibration drifts, we expect a systematic change in the consistency metrics, as illustrated in Figure 1. We therefore continuously monitor the stream of geometric consistency metrics and detect statistically significant deviations from their nominal behaviour. This approach offers several advantages: it requires no task-specific training, accumulates evidence over time to detect subtle relative displacements of the sensors that may be imperceptible in individual frames, and provides control over the nominal false-alarm rate through the Average Run Length (ARL ) parameter, though this control is only approximate when the monitored signal is not perfectly stationary, as we discuss in Section 5. To the best of our knowledge, this is the first approach that casts miscalibration detection as a sequential change-detection problem, rather than as per-frame supervised classification.

MiCaDe matches 3D and 2D detections, extracts geometric consistency indicators from the resulting correspondences, and aggregates them into fixed-size batches. A reference distribution fit on calibrated sequences is then used to score incoming observations, while a sequential changedetection test monitors their evolution and raises an alarm when a statistically significant distributional shift is detected. Since MiCaDe relies exclusively on detections already produced by the perception stack, it introduces negligible computational overhead and can be readily integrated into existing autonomous driving systems [16, 17].

Our main contributions are:

• Sequential monitoring of calibration consistency: we formulate LiDAR–camera miscalibration detection as a statistical change-detection problem, monitoring a data stream of cross-modal geometric consistency metrics and providing low false-alarm rates.

• Lightweight calibration monitoring from standard detections: we show that geometric consistency metrics derived from 2D and 3D object detections already available in modern perception stacks provide effective indicators of calibration drift, without requiring additional sensors, supervision, or retraining.

• Real-world evaluation with induced drift: we provide the first public on-road dataset with physically induced miscalibration at a known time, enabling quantitative evaluation under real conditions.

## 2. Related Work

Calibration monitoring lies at the intersection of LiDARcamera calibration and change detection. Existing approaches mainly differ in the signal used to assess cross-

modal consistency and in the decision rule adopted to detect calibration failures.

## 2.1. Calibration Signals

Levinson and Thrun [10] compare image edges with Li-DAR depth discontinuities to identify calibration degradation. More recent methods, instead, employ learned crossmodal representations. Chen et al. [4] learn a consistency score through contrastive learning using a Siamese architecture applied to RGB images and depth maps obtained by projecting the LiDAR points into the image, while Tahiraj et al. [20] use modality-specific encoders on RGB images and depth maps to extract calibration-sensitive features.

In contrast, MiCaDe relies on geometric consistency metrics derived from 2D and 3D object detections. Since object detections are already produced by standard autonomous driving perception stacks, this signal is available with essentially no additional sensing or inference cost.

## 2.2. Decision Rules for Miscalibration Detection

Beyond the monitored signal itself, existing methods differ in how calibration failures are detected. Levinson and Thrun [10] compare geometric consistency measures against fixed thresholds. Learning-based approaches formulate detection as a supervised binary classification problem, where each frame is independently classified as calibrated or miscalibrated [4, 20].

While effective for large misalignments, frame-level decision rules suffer from several limitations. These approaches require supervised training or manually selected thresholds, do not accumulate evidence over time, and provide no explicit control over the false-alarm rate. In contrast, MiCaDe formulates calibration monitoring as a sequential change-detection problem, enabling training-free monitoring with a nominally controlled false-alarm rate.

## 2.3. Extrinsic LiDAR-Camera Calibration

Extrinsic calibration has been extensively studied as a parameter estimation problem. Classical methods rely on explicit geometric correspondences between LiDAR points and image features [2, 3, 7, 11], while learning-based approaches directly regress calibration parameters through deep multimodal architectures [8, 13, 21, 23].

While calibration methods exploit cross-modal correspondences to estimate extrinsic parameters, MiCaDe addresses the complementary, operational problem of deciding when such a correction should be invoked: even an accurate calibration procedure still needs to know when it should be invoked, since running it continuously is neither necessary nor safe, as it may overwrite a valid calibration. MiCaDe acts as a lightweight trigger, after which any compatible calibration module can be invoked to estimate the correction.

## 3. Problem Formulation

We consider a standard vision system equipped with a calibrated LiDAR-camera pair. Let $\textbf { K } \in \bar { \mathbb { R } } ^ { \bar { 3 } \times 3 }$ denote the known camera intrinsics, and let $\mathbf { R } \in \mathbb { R } ^ { 3 \times 3 } , \mathbf { t } \in \mathbb { R } ^ { 3 }$ denote the extrinsic rotation and translation, respectively. Together they define the projection matrix $\mathbf { P } \mathbf { \Psi } = \mathbf { \bar { K } } [ \mathbf { R } | \mathbf { t } ] \mathbf { \Psi } \in \mathbf { \bar { \mathbb { R } } } ^ { 3 \times 4 }$ which maps LiDAR coordinates to image coordinates.

Under nominal conditions, P is estimated during system setup and assumed fixed. However, mechanical perturbations, vibrations, or gradual thermal drift may alter the relative pose over time, introducing systematic inconsistencies between LiDAR and camera observations that accumulate and degrade downstream perception.

We assume to have access to synchronized camera– LiDAR streams $\{ I _ { t } \} _ { t = 0 } ^ { T } \mathrm { a n d } \{ \mathcal { P } _ { t } \} _ { t = 0 } ^ { T } .$ , where $I _ { t }$ denotes the RGB image and $\mathcal { P } _ { t }$ the LiDAR point cloud at time t. We also assume access to one or more reference sequences $\boldsymbol { S _ { r e f } } = \{ ( \boldsymbol { I _ { t } } , \mathcal { P } _ { t } ) \}$ acquired under correct calibration, used to characterise the nominal behaviour of the system.

We further assume the vehicle is equipped with onboard 2D and 3D object detectors, already part of the perception stack, trained to recognize the same set of object classes so that the same car, pedestrian, or truck can be extracted from both modalities. At each time $t ,$ the LiDAR-based detector produces a set of L 3D bounding boxes $\mathcal { D } _ { t } ^ { 3 d } \ =$ $\{ ( \mathbf { B } _ { i } ^ { t } , s _ { i } ^ { t } , \lambda _ { i } ^ { t } ) \} _ { i = 1 } ^ { L }$ , where $\mathbf { B } _ { i } ~ = ~ ( \mathbf { C } _ { i } ^ { 3 d } , l _ { i } , h _ { i } , w _ { i } , \theta _ { i } ) ^ { \dagger }$ encodes the box center, size, and orientation, $s _ { i } ~ \in ~ [ 0 , 1 ]$ its confidence score, and $\lambda _ { i }$ its predicted class; the RGB-based detector similarly produces a set of ℓ 2D bounding boxes $\mathcal { D } _ { t } ^ { 2 d } = \{ ( \mathbf { b } _ { j } ^ { t } , s _ { j } ^ { t } , \mathbf { \bar { \lambda } } _ { j } ^ { \bar { t } } ) \} _ { j = 1 } ^ { \ell } ,$ centered in $\mathbf { c } _ { j } ^ { 2 d }$

Miscalibration Detection. Let $\mathbf { P } _ { 0 }$ denote the nominal projection matrix obtained during system setup. During operation, mechanical perturbations or gradual drift may change the effective calibration to an unknown projection matrix $\mathbf { P } _ { t }$ . Our goal is to detect the earliest time τ at which the calibration departs from its nominal value, i.e.,

$$
\mathbf { P } _ { t } = \left\{ \begin{array} { l l } { \mathbf { P } _ { 0 } , } & { t < \tau , } \\ { \mathbf { P } _ { 1 } \neq \mathbf { P } _ { 0 } , } & { t \geq \tau , } \end{array} \right.
$$

where ${ \bf P } _ { 1 }$ denotes the post-change projection matrix.

## 4. Method: MiCaDe

MiCaDe, illustrated in Figure 2 and summarized in Algorithm 1, transforms each frame into a collection of geometric consistency indicators extracted from projected 3D bounding boxes matched with 2D detections. At each timestep t, using the nominal projection matrix $\mathbf { P } _ { 0 } ,$ , each 3D detection in $\mathcal { D } _ { t } ^ { 3 d }$ is projected into the image plane and matched to 2D detections in $\mathcal { D } _ { t } ^ { 2 d }$ via Matching (line 6), as detailed in Section 4.1, and a vector of geometric consistency metrics is computed (line 8), as in Section 4.2.

Algorithm 1 MiCaDe   
Input: Sequence of multimodal detections   
$( \mathcal { D } _ { 0 } ^ { 3 d } , \mathcal { D } _ { 0 } ^ { \bar { 2 d } } ) , . . . , ( \mathcal { D } _ { T } ^ { 3 \bar { d } } , \mathcal { D } _ { T } ^ { 2 d } )$ initial calibration   
${ \bf P } _ { 0 } = { \bf K } [ { \bf R } _ { 0 } | { \bf t } _ { 0 } ]$ , reference sequences $S _ { r e f }$   
Output: Estimated change point τˆ   
1: function $\mathbf { M } \mathrm { I C A D E } ( \{ \mathcal { D } _ { t } ^ { 3 d } \} _ { t = 0 } ^ { T } , \{ \mathcal { D } _ { t } ^ { 2 d } \} _ { t = 0 } ^ { T } , \mathbf { P } _ { 0 } , S _ { r e f } )$   
2: $k  0$ ▷ Batch index   
3: $\mathcal { H }  \emptyset$ ▷ Metrics history   
4: $\phi _ { 0 } \gets \mathrm { F I T G M M } ( S _ { r e f } )$   
5: for $t = 0 , \ldots , T$ do   
6: $\mathcal { M } _ { t } \gets \mathbf { M } \mathrm { A T C H I N G } 2 \mathbf { D } 3 \mathbf { D } ( \mathcal { D } _ { t } ^ { 3 d } , \mathcal { D } _ { t } ^ { 2 d } , \mathbf { P } _ { 0 } )$   
7: for m $\ r \in \mathcal { M } _ { t }$ do   
8: e(m) ← COMPUTEMETRICS(m)   
9: $\mathcal { H } \gets \mathcal { H } \cup \{ ( \mathbf { m } , \mathbf { e } ( \mathbf { m } ) ) \}$   
10: end for   
11: $\mathcal { B } _ { k } , \ldots , \mathcal { B } _ { k + n _ { t } } \gets \mathbf { G e r N E W B A T C H E S } ( \mathcal { H } )$   
12: $\mathbf { f } _ { k } , \dots , \mathbf { f } _ { k + n _ { t } }  \mathrm { A G G } ( \mathcal { B } _ { k } , \dots , \mathcal { B } _ { k + n _ { t } } )$   
13: $l _ { k } , \ldots , l _ { k + n _ { t } } \gets \mathbf { G M M } ( \mathbf { f } _ { k } , \ldots , \mathbf { f } _ { k + n _ { t } } , \phi _ { 0 } )$   
14: $k \gets k + n _ { t }$   
15: $\hat { k } ,$ changed ← CHANGEDETECTOR $( \{ l _ { i } \} _ { i = k } ^ { n _ { t } } )$   
16: if changed then   
17: return τˆ ▷ Last timestamp of batch $\boldsymbol { B } _ { \hat { \boldsymbol { k } } }$   
18: end if   
19: end for   
20: return −1 ▷ No change detected   
21: end function

Matched pairs are appended to a running buffer (line 9) and grouped into fixed-size batches (line 11). Our intuition is that, under correct calibration, the projection of 3D bounding boxes fluctuates around their matched 2D counterparts with no systematic bias. When the relative camera-LiDAR pose drifts, instead, the projected boxes depart from the 2D detections in a consistent direction, inducing a distributional shift in the metric stream. We therefore adopt a sequential change-detection test (Section 4.3) to monitor this stream and raise an alarm when a persistent shift is detected (line 15). Since all operations are performed on a small set of bounding box detections, MiCaDe is suitable for continuous onboard monitoring with negligible computational overhead.

## 4.1. Cross-modal Correspondence Extraction

To match each LiDAR detection $\mathbf { B } _ { i } \in \mathcal { D } _ { t } ^ { 3 d }$ with at most one RGB detection $\mathbf { b } _ { j } \in \mathcal { D } _ { t } ^ { 2 d }$ , we follow the Bounding Box Matching procedure of LCF3D [17]:

• Each 3D bounding box B<sub>i</sub> is projected into the image plane, obtaining the corresponding 2D box $\mathbf { b } _ { i } ^ { \mathrm { p r o j } }$

• A linear sum assignment problem is solved to find the one-to-one matching between all the projected boxes $\mathbf { b } _ { j } ^ { \mathrm { p r o j } }$ and RGB detections $\mathbf { b } _ { j } ,$ maximizing their 2D IoU; considering only pairs sharing the same predicted class $\lambda _ { i } =$

![](images/3f44cc33117e5e9bf0ec7d4c2d8aebde861bf6123360582496ad8ad55cd5479f.jpg)  
Figure 2. Overview of MiCaDe. At each timestep, 3D bounding boxes produced by a LiDAR detector are projected into the image plane and matched with 2D bounding boxes from an RGB detector. For each matched pair $\mathbf { m } _ { k } ^ { t }$ , MiCaDe computes a geometric consistency vector $\mathbf { e } ( \mathbf { m } _ { k } ^ { t } )$ and appends it to a metrics history H. Metric vectors are grouped into fixed-size batches $\boldsymbol { B } _ { k }$ , aggregated into batch-leve features $\mathbf { f } _ { k }$ , and scored against a reference GMM to obtain a likelihood stream $\{ l _ { k } \}$ . A Lepage CPM monitors this stream and raises an alarm at the estimated change point $\hat { \tau }$ when a persistent distributional shift is detected.

$\lambda _ { j }$ as candidates.

• Pairs whose IoU falls below a threshold $\rho = 0 . 1$ are discarded. We intentionally adopt a relatively low threshold, since the goal is not to guarantee highly accurate matches but rather to retain enough correspondences for monitoring. A higher threshold would reject many valid matches under moderate miscalibrations, precisely when detecting calibration drift is most important.

Since matching is itself performed on projected boxes, any miscalibration introduces inconsistencies between the 2D and 3D detections being matched, which could in principle give rise to wrong correspondences and reduce detection power. We verify in Section B.4 of the supplementary material, and report in Section 5, that matches remain reliable even under miscalibration, while the average number of valid matches per frame decreases, affecting detection delay rather than detection power.

Alternative robust alignment solutions as AutoAlign [5] and GraphAlign [19] establish 2D-3D correspondences via a cross-modal feature interaction module and via graph matching, respectively. Both rely on learnable components trained end-to-end for detection, adding an overhead that conflicts with the training-free simplicity we target.

We denote by $\mathcal { M } _ { t } ~ = ~ \{ \mathbf { m } _ { k } ^ { t } \} _ { k = 1 } ^ { N _ { t } }$ the set of matched 3D–2D detections $\mathbf { m } _ { k } ^ { t } = ( \mathbf { B } _ { i _ { k } } ^ { t } , \mathbf { b } _ { j _ { k } } ^ { t } )$ obtained at time step $t ,$ where the number of valid matches $N _ { t } = | \mathcal { M } _ { t } |$ depends on both scene content and calibration accuracy.

## 4.2. Geometric Consistency Metrics

For each match $\mathbf { m } _ { k } ^ { t } = ( \mathbf { B } _ { i _ { k } } ^ { t } , \mathbf { b } _ { j _ { k } } ^ { t } )$ , we derive a set of geometric quantities that characterize the consistency between the projected 3D detection and its corresponding 2D detection. First, we define the center displacement vector

$$
\mathbf { a } _ { k } ^ { t } = \mathbf { c } _ { j _ { k } } ^ { 2 d } - \mathbf { c } _ { i _ { k } } ^ { \mathrm { p r o j } } \in \mathbb { R } ^ { 2 } ,\tag{1}
$$

![](images/91b3888df325d44a249df50bd6e55687e4f6ec6ac502c8366ed669efc498ecda.jpg)  
Figure 3. (Top) A point cloud projected into the image with uncalibrated extrinsics. (Bottom left) IoU and center displacement between the projected 3D LiDAR bounding box and the corresponding 2D RGB bounding box. (Bottom right) The backprojection of the 2D bounding box produces an inaccurate depth estimate under miscalibration.

which measures the offset between the projected LiDAR box center and the image detection center. Under correct calibration, $\mathbf { a } _ { k } ^ { t }$ fluctuates around (0, 0), while miscalibration induces a systematic bias in both magnitude and direction. We then define the displacement direction as $\alpha _ { k } ^ { t } .$ , i.e. the angle of $\mathbf { a } _ { k } ^ { t }$ with respect to the horizontal axis. Rather than using $\alpha _ { k } ^ { t }$ directly, we encode the direction through its sine and cosine components. This avoids the discontinuity at ±π, where two nearly identical directions correspond to numerically distant angle values.

Second, we compute the overlap between the two detections through their 2D Intersection-over-Union

$$
e _ { i } = \mathrm { I o U } \left( \mathbf { B } _ { i _ { k } } ^ { \mathrm { p r o j } , t } , \mathbf { b } _ { j _ { k } } ^ { t } \right) .\tag{2}
$$

Table 1. Per-match geometric consistency metrics.
<table><tr><td>Metric</td><td>Definition</td><td>Calibration cue</td></tr><tr><td>x-displacement  $e _ { x }$ </td><td> $[ \mathbf { a } _ { k } ^ { t } ] _ { x }$ </td><td>Horizontal projection offset</td></tr><tr><td>y-displacement  $e _ { y }$ </td><td> $[ \mathbf { a } _ { k } ^ { t } ] _ { y }$ </td><td>Vertical projection offset</td></tr><tr><td>Magnitude  $e _ { a }$ </td><td> $\big \| \mathbf { a } _ { k } ^ { t } \big \| _ { 2 }$ </td><td>Projection error magnitude</td></tr><tr><td>Direction  $( e _ { \mathrm { s i n } } , e _ { \mathrm { c o s } } )$ </td><td> $\sin ( \alpha _ { k } ^ { t } ) , \cos ( \alpha _ { k } ^ { t } )$ </td><td>Direction of systematic drift</td></tr><tr><td>IoU ei</td><td> $\mathrm { I o U } ( \mathbf { B } _ { i _ { k } } ^ { \mathrm { p r o j } , t } , \mathbf { b } _ { j _ { k } } ^ { t } )$ </td><td>Overall 2D box overlap</td></tr><tr><td>Depth error eb</td><td> $\frac { | d _ { r } - d _ { l } | } { d _ { l } }$ </td><td>Coarse 3D consistency</td></tr></table>

![](images/9505b924394886a1b3a6eb9c69db04fcd7febcd9b4bde4cbed5323bc498db66c.jpg)  
Figure 4. Example of monitored metrics evolution under calibration drift. A calibration perturbation is introduced after frame 50 (red dashed line). Metrics such as IoU $( e _ { i } )$ , depth error $( e _ { b } ) .$ and directions $( e _ { s } i n , e _ { c } o s )$ exhibit a clear distributional shift after the change, motivating the use of sequential change detection.

Finally, we estimate a coarse depth-consistency signal by backprojecting the bottom center of $\mathbf { b } _ { j _ { k } } ^ { t }$ and intersecting it with the ground plane (assuming the bottom of $\mathbf { b } _ { j _ { k } } ^ { t }$ lies on the ground), obtaining a depth estimate $d _ { r }$ , and comparing it with the LiDAR depth $d _ { l }$ associated with $\mathbf { B } _ { i _ { k } } ^ { t }$ (Figure 3). These quantities are combined into the metric vector

$$
\mathbf { e } ( \mathbf { m } _ { k } ^ { t } ) = ( e _ { x } , e _ { y } , e _ { i } , e _ { a } , e _ { \mathrm { s i n } } , e _ { \mathrm { c o s } } , e _ { b } ) ^ { \top } \in \mathbb { R } ^ { 7 } ,\tag{3}
$$

whose components are summarized in Table 1.

## 4.3. Calibration Monitoring

A drift in the extrinsic parameters induces a change in the distribution of the metric vectors $\mathbf { e } ( \mathbf { m } _ { k } ^ { t } )$ defined in (3) (Fig. 4). In particular, we expect $e _ { i }$ to decrease, while $e _ { x } .$ $e _ { y } , e _ { a }$ , and $e _ { b }$ increase due to growing projection inaccuracies. Moreover, $e _ { \mathrm { s i n } }$ and $e _ { \mathrm { c o s } }$ transition from approximately zero-mean, high-variance variables to concentrated variables with non-zero mean. Rather than reacting to individual noisy detections, our objective is to identify a persistent multivariate distributional shift, which naturally motivates the use of sequential change-detection techniques [15].

Before processing the computed matches, we discard the unreliable ones. Specifically, a match is discarded if the projected 3D box falls for more than a fraction $\tau _ { t }$ outside the image plane (truncated box), or if its 3D position lies farther than a distance $\tau _ { d }$ from the sensor, since these induce a larger, calibration-independent localization noise.

Batch aggregation. The number of valid matches $N _ { t }$ varies across frames. Directly monitoring frame-level aggregates of the metrics in (3) would violate the i.i.d. assumptions underlying change-detection tests, as their distribution would depend on $N _ { t }$ . To mitigate this issue, we accumulate metric vectors into a history buffer H and group them into fixedsize batches $B _ { k } ~ = ~ \{ \mathbf { m } _ { k _ { 1 } } ^ { t _ { 1 } } , \ldots , \mathbf { m } _ { k _ { n } } ^ { t _ { n } } \}$ , possibly spanning multiple consecutive frames.

The metrics are aggregated within each batch by computing the mean metric vector

$$
\bar { \mathbf { e } } = \frac { 1 } { n } \sum _ { \mathbf { m } \in B _ { k } } \mathbf { e } ( \mathbf { m } ) ,\tag{4}
$$

and the variances of the directional components

$$
\sigma _ { \sin } ^ { 2 } = \frac { 1 } { n } \sum _ { \mathbf { m } \in \mathcal { B } _ { k } } \left( e _ { \sin } ( \mathbf { m } ) - \bar { e } _ { \sin } \right) ^ { 2 } ,\tag{5}
$$

and analogously for $\sigma _ { \mathrm { c o s } } ^ { 2 }$

The resulting feature vector $\mathbf { f } _ { k } \in \mathbb { R } ^ { 9 }$ of the batch is

$$
\mathbf { f } _ { k } = ( \bar { e } _ { x } , \bar { e } _ { y } , \bar { e } _ { i } , \bar { e } _ { a } , \bar { e } _ { \mathrm { s i n } } , \bar { e } _ { \mathrm { c o s } } , \bar { e } _ { b } , \sigma _ { \mathrm { s i n } } ^ { 2 } , \sigma _ { \mathrm { c o s } } ^ { 2 } ) ^ { \top } .\tag{6}
$$

Under nominal operating conditions, the sequence $\left\{ \mathbf { f } _ { k } \right\}$ follows a multivariate distribution $\phi _ { 0 }$ reflecting detector noise and scene variability.

Likelihood Computation. We estimate the distributionϕ<sub>0</sub> of $\left\{ \mathbf { f } _ { k } \right\}$ extracted from one or more calibrated reference sequences $\boldsymbol { S } _ { \mathrm { r e f } }$ as a Gaussian Mixture Model (GMM). The GMM remains fixed throughout deployment. For each incoming batch, we compute the log-likelihood $l _ { k } =$ log $p _ { \phi _ { 0 } } ( \mathbf { f } _ { k } )$ under the reference GMM.

Change-point detection. During operations, the stream of likelihoods $\{ l _ { k } \}$ is monitored using the non-parametric Change Point Model (CPM) of Ross et al. [15], based on the Lepage statistic. The CPM operates online and detects both changes in mean and variance of the monitored process. Moreover, it has two important features: (i) it provides control over the nominal Average Run Length $( A R L _ { 0 } )$ , i.e., the expected time between false alarms, and (ii) it estimates, after each detection, the most likely change point $\hat { k } ,$ , which is converted into the corresponding frame index τˆ. The decision thresholds corresponding to a target $A R L _ { 0 }$ are precomputed offline on synthetic sequences containing random noise realizations, following [15]. Unfortunately, $\mathrm { \ A R L _ { 0 } }$ control is exact only when the CPM inputs are i.i.d. realizations of $\phi _ { 0 } ;$ as we show in Section 5, this assumption holds only approximately in our setting. Thus, the $\mathrm { \ A R L _ { 0 } }$ is intended as a hyperparameter that regulates the detection sensitivity.

Computational complexity. MiCaDe operates exclusively on detections already produced by the perception stack and therefore introduces only a lightweight monitoring overhead. Its cost is dominated by the 2D-3D matching stage, a Hungarian assignment problem of complexity $O \big ( \operatorname* { m a x } ( L _ { t } , \ell _ { t } ) ^ { 3 } \big )$ in the number of LiDAR and image detections $L _ { t } , \ell _ { t } ;$ metric computation, batch aggregation, GMM evaluation, and CPM updates all operate on lowdimensional quantities and add negligible overhead. Overall, MiCaDe’s cost scales with the number of detected objects rather than with the image or point-cloud size.

## 5. Experiments

We test MiCaDe on three public autonomous driving datasets corrupted by synthetic perturbations, and on a realworld dataset with a physically induced calibration change.

## 5.1. Implementation Details

In our experiments, matches are filtered using a truncation threshold $\tau _ { t } = 0 . 2$ and a distance threshold $\tau _ { d } = 5 0 \mathrm { m } ( \mathrm { S e c } -$ tion 4.3). Feature vectors extracted from $\boldsymbol { S _ { r e f } }$ are grouped using a batch size $n = 1 5$ and used to fit a GMM, whose number of components is selected via the Bayesian Information Criterion. Incoming batches are scored using the fixed GMM and monitored online by the Lepage CPM, which uses $A R L _ { 0 } = 1 0 0 0 0$ and $A R L _ { 0 } = 2 0 0 0 0$ . Further implementation and reproducibility details are provided in Section A of the supplementary material.

## 5.2. Datasets and Synthetic Perturbations

Synthetic Datasets. We evaluate MiCaDe on three public autonomous driving datasets: KITTI Raw [6], ZOD [1] and ONCE [12]. As object detectors, we adopt PV-RCNN [18] and Faster R-CNN [14] on KITTI Raw, PointPillars [9] and Faster R-CNN on ZOD, and PointPillars and DDQ [22] on ONCE. KITTI and ZOD run at 10 Hz while ONCE provides frames at 2 Hz, which we use to convert delay to seconds. We use as reference set $\boldsymbol { S _ { r e f } }$ a subset of the detectors’ training sequences, and evaluate on the remaining sequences. Further details on dataset splits are provided in Section $\mathrm { A } .$

We inject synthetic calibration drifts into all the sequences following two protocols, abrupt changes and gradual changes; for each sequence, we generate 50 perturbed variants, randomly sampling the perturbation parameters and the change point $\tau ,$ , and report results averaged over all variants and sequences. We select τ at least 100 frames after the sequence start and 100 frames before the end of the sequence, and apply a rigid transformation

$$
\Delta { \bf T } = \left[ \begin{array} { l l } { \Delta { \bf R } } & { \Delta { \bf t } } \\ { 0 } & { 1 } \end{array} \right]
$$

to all LiDAR observations acquired after τ , leaving the camera stream unchanged. For abrupt changes, ∆R is generated by independently sampling roll, pitch, and yaw deltas from $[ - 0 . 0 5 , 0 . 0 5 ]$ rad, and ∆t from [−0.2, 0.2] m, independently for each variant, applied at once at τ . For gradual changes, ∆T is instead ramped up linearly from identity to a fixed magnitude $( 3 ^ { \circ }$ on all rotation angles, 0.2 m on all translation components) over 100 frames starting at τ , remaining constant thereafter.

Real-World Dataset. We evaluate MiCaDe on a one-hour sequence from a platform with three RGB cameras and a 128-beam LiDAR (Section 5.6). Unlike existing benchmarks, our real-world dataset contains a physically induced calibration change, capturing additional sources of variability. A known calibration drift is introduced by rotating one camera by approximately $3 ^ { \circ }$ , resulting in 2,151 calibrated and $^ { 5 , 2 3 4 }$ miscalibrated frames. The first 1,000 calibrated frames are used for $\boldsymbol { S _ { r e f } }$ , while the remainder is processed online. Detectors trained on ZOD are used in this setting. The dataset will be publicly released upon acceptance.

## 5.3. Competitors

We compare against Tahiraj et al. [20], trained using Mi-CaDe’s reference split and the protocol of [20]. During evaluation, it flags each frame independently as calibrated or miscalibrated, using both the nominal classification threshold (0.5) and also an adjusted threshold on clean sequences to match MiCaDe’s empirical $A R L _ { 0 } = 1 0 0 0 0$ enabling a comparison at matched false-alarm rates. Levinson and Thrun [10] and Chen et al. [4] are also related, but lack an official implementation and could not be included.

## 5.4. Evaluation Metrics

We evaluate MiCaDe on clean sequences and on sequences with an abrupt or gradual calibration change. For each perturbed sequence, we consider the first alarm raised at time T: if $T > \tau$ , the change is correctly detected with delay $| T - \tau | ; \mathrm { i f } T < \tau .$ , it is a false positive (FP), excluded from the delay computation; if no detection is raised, it is a false negative (FN), which should not occur with sufficiently long sequences. On clean sequences we compute the empirical $A R L _ { 0 }$ (average time, in seconds, between false alarms) and FP/h (false alarms per hour). On sequences with abrupt or gradual changes: the False Positive Rate (FPR) and False Negative Rate (FNR), as above, together with the mean Detection Delay in seconds. On gradual changes only we also report the angle, in degrees, reached by the ramp at the time of detection. In the supplementary material, we report the delay distribution’s density and cumulative distribution (Section B.2). We additionally report the AUC, the area under the ROC curve, treating the calibrated and miscalibrated portions of a sequence as the two classes and the monitored signal (log-likelihood for MiCaDe, classification score for Tahiraj et al.) as the score, computed on sequences with abrupt changes, which measures a signal’s raw discriminative power independently of the monitoring scheme.

## 5.5. Main Results

Table 2 reports results on clean sequences and on sequences with abrupt and gradual calibration changes. We omit

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Threshold</td><td colspan="3">Clean sequences</td><td colspan="3">Abrupt changes</td><td colspan="4">Gradual changes</td><td rowspan="2">Speed [ms] ↓</td><td rowspan="2">AUC ↑</td></tr><tr><td> $\mathrm { { A R L } _ { 0 } \ [ s ] \uparrow }$ </td><td> $\mathbf { A R L } _ { 0 } \left[ \mathbf { b } \right] \uparrow$ </td><td>FP/h↓</td><td>FNR↓</td><td>FPR↓</td><td>Delay [s] ↓</td><td>FNR↓</td><td>FPR↓</td><td>Delay [s] ↓</td><td>Angle [deg] ↓</td></tr><tr><td rowspan="4">KITTI</td><td>MiCaDe</td><td> $A R L _ { 0 } = 1 0 0 0 0$ </td><td>617.4</td><td>398.5</td><td>5.83</td><td>0.088</td><td>0.001</td><td>8.78</td><td>0.044</td><td>0.043</td><td>8.47</td><td>1.865</td><td>2</td><td>0.9976</td></tr><tr><td>MiCaDe</td><td> $A R L _ { 0 } = 2 0 0 0 0$ </td><td></td><td>-</td><td></td><td>0.161</td><td>0.000</td><td>8.92</td><td>0.044</td><td>0.000</td><td>8.77</td><td>1.943</td><td></td><td></td></tr><tr><td>Tahiraj et al.</td><td>.5</td><td>6.00</td><td>-</td><td>597.6</td><td>0.000</td><td>0.651</td><td>0.40</td><td>0.000</td><td>0.594</td><td>3.20</td><td>0.992</td><td>12*</td><td>0.9969</td></tr><tr><td>Tahiraj et al. (adj.)</td><td>.999998</td><td>595.2</td><td></td><td>6.48</td><td>0.024</td><td>0.012</td><td>0.60</td><td>0.031</td><td>0.031</td><td>4.40</td><td>1.364</td><td></td><td></td></tr><tr><td rowspan="6">ZOD</td><td>MiCaDe</td><td> $A R L _ { 0 } = 1 0 0 0 0$ </td><td>941.8</td><td>1041.4</td><td>3.82</td><td>0.000</td><td>0.000</td><td>4.66</td><td>0.000</td><td>0.000</td><td>21.43</td><td>1.148</td><td>2</td><td>0.9965</td></tr><tr><td>MiCaDe</td><td> $A R L _ { 0 } = 2 0 0 0 0$ </td><td>1177.3</td><td>1666.2</td><td>3.06</td><td>0.006</td><td>0.000</td><td>5.66</td><td>0.000</td><td>0.000</td><td>24.50</td><td>1.348</td><td></td><td></td></tr><tr><td>Tahiraj et al.</td><td> $. 5$ </td><td>38.5</td><td></td><td>93.41</td><td>0.000</td><td>0.417</td><td>0.40</td><td>0.000</td><td>0.334</td><td>3.37</td><td>0.185</td><td>14*</td><td></td></tr><tr><td>Tahiraj et al. (adj.)</td><td>.9974</td><td>932.4</td><td></td><td>3.79</td><td>0.865</td><td>0.000</td><td>0.59</td><td>1.000</td><td>0.000</td><td></td><td></td><td></td><td>1.000</td></tr><tr><td>MiCaDe</td><td> $A R L _ { 0 } = 1 0 0 0 0$ </td><td>4523.5</td><td>1332.0</td><td>0.80</td><td>0.070</td><td>0.002</td><td>56.48</td><td>0.105</td><td>0.053</td><td>84.29</td><td>1.806</td><td>2</td><td></td></tr><tr><td>MiCaDe</td><td> $A R L _ { 0 } = 2 0 0 0 0$ </td><td>8750.0</td><td>2664.0</td><td>0.41</td><td>0.147</td><td>0.000</td><td>68.22</td><td>0.105</td><td>0.000</td><td>134.10</td><td>2.822</td><td></td><td>0.9989</td></tr><tr><td>ONCE</td><td>Tahiraj et al. Tahiraj et al. (adj.)</td><td>.5 .8278</td><td>902.3</td><td>-</td><td>3.99</td><td>0.000</td><td>0.373</td><td>1.10</td><td>0.000</td><td>0.210</td><td>2.5 3.0</td><td>0.149 0.173</td><td>11*</td><td>1.000</td></tr></table>

Table 2. Results on clean sequences, abrupt calibration changes, and gradual calibration changes. $\mathrm { \ A R L _ { 0 } }$ on clean sequences is reported both in seconds [s] and in batches [b]. Speed and AUC are reported once per method, as they do not depend on the $\mathrm { A R L } _ { 0 } / 1$ threshold operating point. AUC measures the raw discriminative power of each signal on the sequences with abrupt changes, independently of the thresholds used. “Tahiraj $\mathrm { ( a d j . ) } ^ { \mathrm { , } }$ denotes the threshold adjusted to match MiCaDe’s empirical $A R L _ { 0 } = 1 0 0 0 0$ on clean sequences. Values with an asterisk (\*) denote GPU computations.

$A R L _ { 0 } = 1 0 0 0 \colon$ assuming 2s per batch, it would still allow at least 2 FP/h if respected, too frequent for our target regime. On clean sequences, MiCaDe’s empirical $A R L _ { 0 }$ falls short of the nominal target used to calibrate the Lepage threshold on every dataset, whether measured in seconds or directly in batches. Part of this gap is expected, since the target is nominally defined on the batch index; the persisting shortfall in batch count indicates that the monitored batch-level features are not fully stationary. Raising the operating point from $A R L _ { 0 } = 1 0 0 0 0$ to $A R L _ { 0 } = 2 0 0 0 0$ substantially reduces the false-alarm rate at a modest cost in detection delay, and we adopt it as our recommended operating point; reliably estimating even rarer false-alarm rates would require longer clean sequences than available here, as shown by KITTI at $A R L _ { 0 } = 2 0 0 0 0$ , where no false alarm is observed in clean sequences as these are too short.

At its nominal threshold (0.5), Tahiraj et al. yields impractically high FPR (0.651/0.417/0.373 on KITTI/ZOD/ONCE); matching MiCaDe’s empirical ARL<sub>0</sub> instead pushes its threshold to extreme values (0.999998, 0.9974). Even so, MiCaDe achieves lower FPR on KITTI/ONCE and far lower FNR on ZOD (0.006 vs. 0.865), at comparable AUC (≥ 0.99 everywhere): our geometry-based indicators match Tahiraj’s learned representation without supervised training or GPU inference.

The performance gap across datasets can be partly explained by sensing setup: KITTI is tightly synchronized with favourable geometry, whereas ZOD has only approximate synchronization and ONCE has a larger camera– LiDAR displacement, increasing projection variability. Mi-CaDe’s FPR stays within 0.000–0.002 across all three at $A R L _ { 0 } ~ = ~ 1 0 0 0 0 \mathrm { : }$ ; ONCE’s larger displacement instead shows up in detection delay, the longest overall (56.48s vs. 8.78s KITTI, 4.66s ZOD). Delay itself is strongly skewed towards small values, as most changes are detected quickly while a smaller subset of harder sequences pulls the mean up substantially (e.g., median 7.61 s vs. 90th percentile 19.11 s on KITTI Raw; Section B.2 reports the full distribution). Qualitative examples are provided in Section D.

<table><tr><td>DOF</td><td>FNR↓</td><td>FPR↓</td><td>Delay [s] ↓</td><td>AUC↑</td></tr><tr><td>Roll</td><td>0.000</td><td>0.000</td><td>8.3</td><td>0.991</td></tr><tr><td>Yaw</td><td>0.000</td><td>0.000</td><td>7.3</td><td>0.962</td></tr><tr><td>Pitch</td><td>0.067</td><td>0.000</td><td>21.9</td><td>0.749</td></tr><tr><td> $t _ { x }$ </td><td>0.100</td><td>0.000</td><td>45.3</td><td>0.617</td></tr><tr><td> $t _ { y }$ </td><td>0.267</td><td>0.000</td><td>91.2</td><td>0.482</td></tr><tr><td> $t _ { z }$ </td><td>0.133</td><td>0.000</td><td>36.5</td><td>0.692</td></tr></table>

Table 3. Sensitivity to individual miscalibration parameters on ZOD $( \mathrm { A R L _ { 0 } } { = } 1 0 0 0 0 )$ . Rotational drifts are detected faster than translational ones.

Sensitivity to individual miscalibration parameters. Unlike the randomized protocol of Section 5.2, here we apply a single, deterministic perturbation (one axis, one magnitude) per sequence, evaluating each degree of freedom independently at three fixed magnitudes: 0.0125, 0.025, 0.05 rad for rotations; 0.05, 0.1, 0.2 m for translations (per-magnitude results are in Section B.1). The perturbation is always injected at frame 100, isolating delay attributable to the perturbed DOF from change-point variability. Table 3 reports FNR, FPR, and delay averaged across magnitudes on ZOD. Rotational drifts are detected faster and more reliably than translational ones, with $t _ { y }$ the hardest axis due to its weaker influence on the monitored geometric cues compared to the systematic horizontal displacement induced by roll or yaw.

Contribution of individual metrics. Table 4 reports the contribution of different feature subsets on ZOD, evaluated on the same perturbed sequences used in Table 2. Among the individual groups, IoU $( e _ { i } )$ and the center-displacement pair $( e _ { x } , e _ { y } )$ achieve the lowest FNR (0.063 and 0.051) and the highest AUC (0.972 and 0.976), while direction alone attains the lowest FPR (0.000) at the cost of a higher FNR (0.165); the depth-consistency pair $( e _ { a } , e _ { b } )$ is weakest overall, with the highest FPR (0.114) and longest delay (17.09s). The directional variances alone yield the lowest standalone AUC (0.845). Overall, the full feature vector achieves the best FPR (0.000), delay (4.66s), and AUC (0.997), without any false negatives, confirming that variance terms complement the mean-based metrics.

![](images/3e0110ea3190cd9ba688c243bfc06d8b93bbb642ddf2cca858f18e12b8e8b3e1.jpg)  
Figure 5. Evolution of three representative monitored metrics, IoU $( e _ { i } ) .$ , displacement magnitude $( e _ { a } )$ , and direction $( e _ { \mathrm { s i n } } ) ,$ together with the detector signal on the real-world sequence. The black dashed line marks the manually induced calibration change, while the green line indicates the detection raised by MiCaDe. The bottom panel reports the log-likelihood stream and the corresponding CPM statistic, which crosses the decision threshold shortly after the perturbation, correctly identifying the calibration drift. The full set of monitored metrics is reported in Section C of the supplementary material.

<table><tr><td rowspan="2">Configuration</td><td colspan="9">Metrics used</td><td rowspan="2">FNR↓</td><td rowspan="2">FPR↓</td><td rowspan="2">Delay [s] ↓</td><td rowspan="2">AUC↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\begin{array} { r } { e _ { x } \quad e _ { y } \quad e _ { i } \quad e _ { a } \quad e _ { b } \quad e _ { \mathrm { s i n } } \quad e _ { \mathrm { c o s } } \quad \sigma _ { \mathrm { s i n } } ^ { 2 } \quad \sigma _ { \mathrm { c o s } } ^ { 2 } } \end{array}$ </td><td></td></tr><tr><td>ei only</td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.063</td><td>0.080</td><td>5.32</td><td>0.972</td></tr><tr><td>ex, ey only</td><td></td><td> $\checkmark \quad \checkmark$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.051</td><td>0.040</td><td>8.55</td><td>0.976</td></tr><tr><td>ea, eb only</td><td></td><td></td><td></td><td></td><td> $\checkmark \quad \checkmark$ </td><td></td><td></td><td></td><td></td><td>0.222</td><td>0.114</td><td>17.09</td><td>0.909</td></tr><tr><td>Direction only</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td>0.165</td><td>0.000</td><td>7.52</td><td>0.949</td></tr><tr><td>Variance only</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $\checkmark \quad \checkmark$ </td><td>0.290</td><td>0.108</td><td>10.09</td><td>0.845</td></tr><tr><td>No variance</td><td>V</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>0.040</td><td>0.000</td><td>5.44</td><td>0.993</td></tr><tr><td>Full (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.000</td><td>0.000</td><td>4.66</td><td>0.997</td></tr></table>

Table 4. Ablation study on the feature set used by MiCaDe on ZOD $( \mathrm { A R L _ { 0 } } { = } 1 0 0 0 0 )$ ). Each row indicates the components of $\mathbf { f } _ { k } \in$ $\mathbb { R } ^ { 9 }$ included in the change-detection pipeline.

Robustness of the detector. We further verify that Mi-CaDe’s alarms are not confused with unrelated sources of variability. 2D–3D matching remains reliable under miscalibration (Section B.4): accuracy drops marginally, from 92% to 89%, and the average number of matches per frame decreases (from 5 to 3), affecting delay rather than detection power. Neither scene changes (Section B.3) nor severe LiDAR degradation (Section B.5, e.g., a missing 90<sup>◦</sup> sector) trigger false alarms at $A R L _ { 0 } = 1 0 0 0 0$ , and a genuine miscalibration on top of a degraded stream is still detected.

## 5.6. Real-World Case Study

Figure 5 shows the evolution of three of the monitored metrics, IoU, displacement magnitude, and direction, together with the detector output; the remaining metrics behave analogously and are reported in full in the supplementary material (Figure 13). Following the induced perturbation, all monitored metrics exhibit a clear distributional shift, which propagates to the likelihood stream and causes the CPM statistic to cross the decision threshold after 34 frames, corresponding to only 5 batches in this setting, with no false alarms observed throughout the sequence. These results confirm that MiCaDe successfully distinguishes persistent calibration drift from normal variability, and that the behaviour observed under synthetic perturbations transfers to real operating conditions.

## 6. Conclusions

We presented MiCaDe, a lightweight framework for online LiDAR–camera calibration monitoring based on sequential change detection. By monitoring geometric consistency metrics extracted from standard 2D and 3D object detections, MiCaDe detects persistent calibration drift without supervised training or modifications to the perception stack. Experiments on KITTI Raw, ZOD, ONCE, and a real-world dataset with physically induced miscalibration demonstrate reliable detection across different sensors and driving conditions, while classifier-based baselines require impractical thresholds to achieve MiCaDe’s false-alarm rate.

A current limitation is the reliance on successful 3D– 2D matching, which may fail in sparse scenes or under severe calibration errors. A second limitation, exposed by the gap between the nominal and empirical $A R L _ { 0 }$ observed in Section 5, is that the monitored batch-level features do not fully satisfy the i.i.d. assumption underlying the CPM, so its false-alarm control holds only approximately. Future work will investigate alternative change-detection schemes more robust to non-stationary inputs, the integration of automatic recalibration modules triggered by MiCaDe, and whether learned alignment strategies such as AutoAlign [5] or GraphAlign [19] can further improve detection power, at the cost of the training-free simplicity targeted in this work.

## Acknowledgements

This paper is supported by FAIR (NextGenerationEU program, PNRR-PE-AI scheme, M4C2, Investment 1.3, Line on Artificial Intelligence). We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer Karolina hosted by IT4Innovations, Czech Republic through an EuroHPC Development Access call.

## References

[1] Mina Alibeigi, William Ljungbergh, Adam Tonderski, Georg Hess, Adam Lilja, Carl Lindstrom, Daria Motorniuk, Jun-¨ sheng Fu, Jenny Widahl, and Christoffer Petersson. Zenseact open dataset: A large-scale and diverse multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20178– 20188, 2023. 6, 1

[2] Pei An, Junfeng Ding, Siwen Quan, Jiaqi Yang, You Yang, Qiong Liu, and Jie Ma. Survey of extrinsic calibration on lidar-camera system for intelligent vehicle: Challenges, approaches, and trends. IEEE Transactions on Intelligent Transportation Systems, 25(11):15342–15366, 2024. 2

[3] Feiyi Chen, Liang Li, Shuyang Zhang, Jin Wu, and Lujia Wang. Pbacalib: Targetless extrinsic calibration for highresolution lidar-camera system based on plane-constrained bundle adjustment. IEEE Robotics and Automation Letters, 8(1):304–311, 2023. 2

[4] Yiqiang Chen, Feng Liu, and Ke Pei. Cross-modal matching cnn for autonomous driving sensor data monitoring. In 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 3103–3112, 2021. 1, 2, 6

[5] Zehui Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinghong Jiang, Feng Zhao, Bolei Zhou, and Hang Zhao. Autoalign: Pixel-instance feature aggregation for multimodal 3d object detection. arXiv preprint arXiv:2201.06493, 2022. 4, 8

[6] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237, 2013. 6, 1

[7] Ryoichi Ishikawa, Shuyi Zhou, Yoshihiro Sato, Takeshi Oishi, and Katsushi Ikeuchi. Lidar-camera calibration using intensity variance cost. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 10688– 10694, 2024. 2

[8] Ganesh Iyer, R Karnik Ram, J Krishna Murthy, and K Madhava Krishna. Calibnet: Geometrically supervised extrinsic calibration using 3d spatial transformer networks. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1110–1117. IEEE, 2018. 2

[9] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697–12705, 2019. 6

[10] Jesse Levinson and Sebastian Thrun. Automatic online calibration of cameras and lasers. In Robotics: Science and Systems, 2013. 2, 6

[11] Zhijian Liu, Haotian Tang, Sibo Zhu, and Song Han. Semalign: Annotation-free camera-lidar calibration with semantic alignment loss. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8845–8851, 2021. 2

[12] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, et al. One million scenes for autonomous driving: Once dataset. arXiv preprint arXiv:2106.11037, 2021. 6, 1

[13] Ni Ou, Zhuo Chen, Xinru Zhang, and Junzheng Wang. Iterative camera-lidar extrinsic optimization via surrogate diffu sion. arXiv preprint arXiv:2411.10936, 2024. 2

[14] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015. 6

[15] Gordon J Ross, Niall M Adams, Dimitris K Tasoulis, and David J Hand. A nonparametric change point model for streaming data. Technometrics, 53(4):379–389, 2011. 5

[16] Carlo Sgaravatti, Roberto Basla, Riccardo Pieroni, Matteo Corno, Sergio M Savaresi, Luca Magri, and Giacomo Boracchi. A multimodal hybrid late-cascade fusion network for enhanced 3d object detection. In European Conference on Computer Vision, pages 339–356. Springer, 2024. 2

[17] Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M Savaresi, Luca Magri, and Giacomo Boracchi. Lcf3d: A robust and real-time late-cascade fusion framework for 3d object detection in autonomous driving. Pattern Recognition, page 113046, 2026. 2, 3

[18] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn: Pointvoxel feature set abstraction for 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10529–10538, 2020. 6

[19] Ziying Song, Haiyue Wei, Lin Bai, Lei Yang, and Caiyan Jia. Graphalign: Enhancing accurate feature alignment by graph matching for multi-modal 3d object detection. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3335–3346. IEEE, 2023. 4, 8

[20] Ilir Tahiraj, Jeremialie Swadiryus, Felix Fent, and Markus Lienkamp. Cal or no cal?-real-time miscalibration detection of lidar and camera sensors. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 16098–16105. IEEE, 2025. 1, 2, 6

[21] Kaiwen Yuan, Zhenyu Guo, and Z Jane Wang. Rggnet: Tolerance aware lidar-camera online calibration with geometric deep learning and generative model. IEEE Robotics and Au tomation Letters, 5(4):6956–6963, 2020. 2

[22] Shilong Zhang, Xinjiang Wang, Jiaqi Wang, Jiangmiao Pang, Chengqi Lyu, Wenwei Zhang, Ping Luo, and Kai Chen. Dense distinct query for end-to-end object detection. In Proceedings of the IEEE/CVF conference on computer vi sion and pattern recognition, pages 7329–7338, 2023. 6

[23] Jiangtong Zhu, Jianru Xue, and Pu Zhang. Calibdepth: Unifying depth map representation for iterative lidar-camera online calibration. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 726–733, 2023. 2

# MiCaDe: Online LiDAR–Camera Miscalibration Change Detection for Autonomous Driving

Supplementary Material

This supplementary material provides additional details supporting the main paper. We first describe the implementation of MiCaDe and the real-world dataset used for evaluation. We then report extended experimental analyses, including sensitivity to perturbation magnitude, detectiondelay distributions, and qualitative examples of both truepositive and false-negative cases.

## A. MiCaDe: Implementation Details

MiCaDe implementation for the experiments described in Section 5 is lightweight and designed for online deployment. For each dataset, the 2D and 3D object detectors are trained on the corresponding training split, while a subset of calibrated sequences is reserved to construct the reference set $\boldsymbol { S _ { r e f } }$ used to estimate the nominal behaviour of the monitored features. The detector architectures adopted for each dataset are reported in Section 5.2 of the main paper. We use MMDetection as a framework for training the 2D object detector and MMDetection3D for training the 3D object detector.

Feature vectors extracted from $\boldsymbol { S _ { r e f } }$ are accumulated into batches of $n = 1 5$ matched detections and used to fit a Gaussian Mixture Model (GMM) through the Expectation-Maximization (EM) algorithm. The number of Gaussian components is selected using the Bayesian Information Criterion (BIC), resulting in 4 components for KITTI Raw and 8 components for both ZOD and ONCE. The resulting GMM remains fixed during deployment.

At runtime, each incoming batch is evaluated under the reference GMM to obtain a log-likelihood score. The resulting likelihood stream is monitored online by the Lepage Change Point Model (CPM), configured with $\mathrm { { A R L } _ { 0 } = }$ 10000 or $\mathrm { A R L } _ { 0 } = 2 0 0 0 0$ , which automatically determines the detection threshold corresponding to the desired average run length under nominal conditions.

Unless otherwise stated, the 3D–2D matching stage uses an IoU threshold of 0.1, which was found to provide a good trade-off between retaining sufficient correspondences under moderate miscalibration and rejecting incorrect matches. Matches are additionally filtered using a truncation threshold $\tau _ { t } ~ = ~ 0 . 2$ and a distance threshold $\tau _ { d } = 5 0$ m (Section 4.3), discarding detections whose localization noise is expected to be large independently of calibration. All monitoring operations are implemented on the CPU and involve only geometric computations and GMM likelihood evaluations over low-dimensional feature vectors, resulting in negligible computational overhead with respect to the perception pipeline.

Dataset details. KITTI Raw [6] provides 70 synchronized LiDAR–camera sequences acquired in urban, rural, and highway environments. Evaluation uses the 36 sequences with more than 350 frames; the remaining sequences are used for detector training and to construct $\boldsymbol { S _ { r e f } }$ . ZOD [1] provides both annotated single frames (ZOD Frames, used for detector training) and long driving sequences (ZOD Drives); ZOD Drives are further split into training drives, used to construct $\boldsymbol { S _ { r e f } }$ , and validation drives, used for evaluation. ONCE [12] contains LiDAR–camera driving sequences captured under diverse environmental conditions: detectors are trained on the training split, which also estimates the reference distribution $\phi _ { 0 }$ , while evaluation is performed on the validation split.

## B. Additional Results

## B.1. Sensitivity to Perturbation Magnitude

Table 5 reports the complete results for the perturbationmagnitude study summarized in the main paper. While Table 3 reports FNR, FPR, and mean detection delay aggregated across perturbation magnitudes, here we analyze each perturbation level individually.

For each degree of freedom, we perturb only one calibration parameter at a time. Rotational perturbations are evaluated at 0.0125, 0.025, and 0.05 rad, whereas translational perturbations are evaluated at 0.05, 0.1, and 0.2 m. For every configuration, we generate ten perturbed sequences by injecting the miscalibration at a random point in the sequence and evaluate MiCaDe using the protocol described in the main paper.

Overall, larger perturbations tend to be detected more quickly, since they induce greater changes in the monitored geometric consistency metrics and therefore require fewer observations before the sequential detector accumulates sufficient statistical evidence. This trend is particularly evident for pitch, $t _ { x } .$ , and $t _ { z } ,$ whose mean detection delay decreases substantially as the perturbation magnitude increases.

The relationship is, however, not strictly monotonic. For example, roll exhibits a slightly lower delay at 0.025 rad than at 0.05 rad, while the delay associated with the vertical translation $t _ { y }$ remains consistently high across all magnitudes. These fluctuations are expected because the detection delay depends not only on the perturbation magnitude, but also on scene content, the number and spatial distribution of matched objects, and the stochastic nature of the sequential change-detection process. Nevertheless, the overall trend confirms that MiCaDe becomes increasingly responsive as the severity of the calibration drift grows.

<table><tr><td>DOF</td><td>Magnitude</td><td>FNR↓</td><td>FPR↓</td><td>Delay [s]↓</td><td>AUC↑</td></tr><tr><td rowspan="2">Roll</td><td>0.0125 rad</td><td>0.000</td><td>0.000</td><td>10.43</td><td>0.974</td></tr><tr><td>0.0250 rad</td><td>0.000</td><td>0.000</td><td>6.31</td><td>0.999</td></tr><tr><td rowspan="2">Yaw</td><td>0.0500 rad</td><td>0.000</td><td>0.000</td><td>8.28</td><td>1.000</td></tr><tr><td>0.0125 rad 0.0250 rad</td><td>0.000</td><td>0.000 0.000</td><td>11.72 5.66</td><td>0.920 0.972</td></tr><tr><td rowspan="2">Pitch</td><td>0.0500 rad</td><td>0.000 0.000</td><td>0.000</td><td>4.58</td><td>0.994</td></tr><tr><td>0.0125 rad</td><td>0.100</td><td>0.000</td><td>51.57</td><td>0.551 0.749</td></tr><tr><td> $t _ { x }$ </td><td>0.0250 rad 0.0500 rad</td><td>0.100 0.000</td><td>0.000 0.000</td><td>7.12 6.91</td><td>0.947</td></tr><tr><td rowspan="3"></td><td>0.05 m 0.10 m</td><td>0.200</td><td>0.000</td><td>63.47 58.78</td><td>0.516 0.559</td></tr><tr><td></td><td>0.100</td><td>0.000 0.000</td><td>13.53</td><td>0.775</td></tr><tr><td>0.20 m</td><td>0.000</td><td></td><td></td><td></td></tr><tr><td rowspan="3"> $t _ { y }$ </td><td>0.05 m</td><td>0.400</td><td>0.000</td><td>112.85</td><td>0.491</td></tr><tr><td>0.10 m</td><td>0.200</td><td>0.000</td><td>97.61</td><td>0.472</td></tr><tr><td>0.20 m</td><td>0.200</td><td>0.000</td><td>63.13</td><td>0.484</td></tr><tr><td rowspan="3"> $t _ { z }$ </td><td>0.05 m</td><td>0.300</td><td>0.000</td><td>52.32</td><td>0.537</td></tr><tr><td>0.10 m</td><td>0.100</td><td>0.000</td><td>50.61</td><td>0.673</td></tr><tr><td>0.20 m</td><td>0.000</td><td>0.000</td><td>6.55</td><td>0.865</td></tr></table>

Table 5. FNR, FPR, and mean detection delay for each perturbation magnitude. Results are reported on ZOD over 10 perturbation trials per configuration.

## B.2. Distribution of Detection Delays

In the main paper, detection delay is summarized using the mean delay over true-positive detections. However, the delay distribution is not symmetric: most calibration changes are detected quickly, while a smaller number of sequences require substantially longer observation windows before enough statistical evidence is accumulated. To better characterize this behaviour, Figure 6 reports the empirical distribution of detection delays on KITTI Raw.

The histogram shows a strongly right-skewed distribution. The median delay is 7.61 s, meaning that half of the detected calibration changes are identified within a short temporal window. At the same time, the 90th percentile is 19.11 s, indicating the presence of a long tail of more challenging cases extending up to several tens of seconds. These late detections typically correspond to sequences with fewer reliable 3D–2D correspondences, weaker projection shifts, or higher scene variability, where the sequential detector needs more batches before crossing the decision threshold.

The empirical cumulative distribution function further confirms this behaviour: the detection probability increases rapidly in the first seconds after the perturbation and then saturates more slowly due to the long-tail cases. This illustrates why a single mean-delay figure does not fully capture this behavior, and highlights the importance of reporting delay statistics beyond the average.

![](images/2adbfdae05da1405071d89f38a50447a9c7c8dea8cab653a471a2d9ed87d54a0.jpg)

![](images/d488d317c9a2372cc92fae81a2df12933967624c3da0dd498bd685af7c41dcbf.jpg)

Figure 6. Distribution of detection delays on KITTI Raw. The left plot shows the delay histogram, while the right plot reports the empirical cumulative distribution function. Dotted vertical lines indicate the median delay $( p 5 0 = 7 . 6 1$ s) and the 90th percentile delay $( p 9 0 = 1 9 . 1 1 ~ \mathrm { s } )$
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Heterog.</td><td colspan="2">Control</td></tr><tr><td>Pairs</td><td>Junct.-FPR</td><td>Pairs</td><td>Junct.-FPR</td></tr><tr><td>KITTI</td><td>24</td><td>0.0</td><td>18</td><td>0.0</td></tr><tr><td>ZOD</td><td>30</td><td>0.0</td><td>30</td><td>0.0</td></tr><tr><td>ONCE</td><td>40</td><td>0.0</td><td>32</td><td>0.0</td></tr></table>

Table 6. Junction-FPR for the density-based scene-change experiment. Heterogeneous pairs concatenate a dense and a sparse sequence; control pairs concatenate two sequences of the same density group.

## B.3. Robustness to Scene Changes

We test whether a scene change, rather than a genuine calibration drift, can itself be mistaken for a persistent distributional shift. For each dataset, we concatenate pairs of clean sequences (A, B) and treat the concatenation point as a pseudo change point: calibration is correct on both sides, so an alarm attributed to it would indicate confusion between scene change and calibration drift. For each pair, $\phi _ { 0 }$ is refit excluding both A and B, and MiCaDe runs with the standard operating point $( A R L _ { 0 } = 1 0 0 0 0 )$ . As a dataset-agnostic proxy for scene difference, we use object density (average frames needed to fill a batch), forming heterogeneous dense/sparse pairs against homogeneous control pairs of matching density, drawn from the same pool of sequences; where richer metadata is available, we also pair sequences by country/daypart (ZOD) and weather/daypart (ONCE, limited to very few pairs). The primary metric is thejunction-FPR: the fraction of pairs with an alarm within ±15 batches of the junction. FP/h, the global FPR, and the empirical $A R L _ { 0 }$ serve as secondary indicators of overall monitor stability, not specific to the junction.

Results. The junction-FPR is 0.0% on all three datasets, for both heterogeneous and control pairs (Table 6): the concatenated sequences do raise alarms overall, at FP/h and empirical $A R L _ { 0 }$ levels consistent with the clean-sequence results of Table 2 (Table 7), but none fall near the density change, indicating ordinary clean-sequence non-stationarity rather than the scene transition itself. We note, however, that pairs are formed from a shared pool of sequences and are thus not fully independent observations.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">FP/h</td><td colspan="2">Global FPR</td><td colspan="2">Emp. ARL0 [s]</td></tr><tr><td>Het.</td><td>Ctrl.</td><td>Het.</td><td>Ctrl.</td><td>Het.</td><td>Ctrl.</td></tr><tr><td>KITTI</td><td>3.92</td><td>3.14</td><td>.250</td><td>.111</td><td>919</td><td>1,146</td></tr><tr><td>ZOD</td><td>2.79</td><td>3.78</td><td>.233</td><td>.400</td><td>1,292</td><td>952</td></tr><tr><td>ONCE</td><td>0.49</td><td>0.56</td><td>.275</td><td>.312</td><td>7,292</td><td>6,454</td></tr></table>

Table 7. Secondary stability metrics for the density-based scenechange experiment (heterogeneous vs. control pairs).
<table><tr><td rowspan="2">Variant</td><td colspan="2">Heterog.</td><td colspan="2">Control</td></tr><tr><td>Pairs</td><td>Junct.-FPR</td><td>Pairs</td><td>Junct.-FPR</td></tr><tr><td>ZOD country/daypart</td><td>30</td><td>0.0</td><td>30</td><td>0.0</td></tr><tr><td>ONCE weather/period</td><td>4</td><td>0.0</td><td>2</td><td>0.0</td></tr></table>

Table 8. Junction-FPR for the metadata-based scene-change variants. The ONCE variant covers very few pairs and lacks adequate coverage of conditions such as rain or night driving.

The metadata-based variants (Table 8) corroborate this picture on ZOD, with 0.0% junction-FPR for both heterogeneous and control pairs; the ONCE metadata variant is too small (4 vs. 2 pairs) to be informative on its own. Overall, within the scene changes covered by this experiment, Mi-CaDe’s alarms do not concentrate on the scene transition.

## B.4. Robustness of 2D–3D Matching under Miscalibration

As discussed in Section 4.1, the Bounding Box Matching procedure operates on projected 3D boxes, and is therefore itself affected by the calibration error it is meant to help monitor: a sufficiently large miscalibration could in principle produce wrong 2D–3D correspondences, which would corrupt the geometric consistency metrics and reduce detection power. We evaluate this effect directly using groundtruth object identities. For each match, we say it is accurate if the projected 3D detection and the 2D detection are associated with the same annotated object in both modalities; otherwise, the match is counted as an incorrect correspondence.

We compare matching accuracy and the average number of retained matches per frame on ONCE, before and after an abrupt calibration change (Table 9). Matching accuracy decreases only marginally under miscalibration, showing that the assignment procedure remains largely reliable even after the calibration drift. The main effect of miscalibration is instead a reduction in the number of valid matches retained per frame, as fewer projected boxes reach the IoU threshold $\rho .$ Since MiCaDe accumulates evidence over batches rather than relying on any individual match, this reduction primarily lengthens the detection delay rather than the detection power, consistently with the results reported in Section 5.

<table><tr><td>Metric</td><td>Before miscalibration</td><td>After miscalibration</td></tr><tr><td>Matching accuracy</td><td>92%</td><td>89%</td></tr><tr><td>Matches per frame</td><td>5</td><td>3</td></tr></table>

Table 9. Effect of an abrupt calibration change on 2D–3D match ing, on ONCE. Matching accuracy is only marginally affected, while the average number of retained matches per frame decreases.
<table><tr><td>Degradation</td><td>FP/h</td><td>Onset ratio</td></tr><tr><td>64→32 rings</td><td>1.28</td><td>0.00</td></tr><tr><td>64→16 rings</td><td>3.87</td><td>0.00</td></tr><tr><td>64→8 rings</td><td>5.52</td><td>0.00</td></tr><tr><td> $4 5 ^ { \circ }$  sector missing</td><td>0.00</td><td>n/a</td></tr><tr><td> $9 0 ^ { \circ }$  sector missing</td><td>0.00</td><td>n/a</td></tr></table>

Table 10. Negative control for LiDAR degradation on KITTI Raw at $A R L _ { 0 } = 1 0 0 0 0 \colon$ calibration remains correct throughout, only the LiDAR stream is degraded from a known onset.

## B.5. Robustness to Sensor Degradation

Since MiCaDe relies on object detections extracted from the LiDAR stream, a natural concern is whether a degradation of the LiDAR sensor itself, rather than a calibration drift, could be mistaken for miscalibration. We evaluate this on KITTI Raw, considering two families of LiDAR degradation: beam loss, where the number of vertical scan rings is reduced from 64 to 32, 16, or 8, and sector loss, where a $4 5 ^ { \circ }$ or 90<sup>◦</sup> azimuthal sector of the scan is entirely removed, simulating a localized sensor occlusion or fault.

Negative control. As in Section B.3, calibration remains correct throughout while the LiDAR degrades from a known onset; we report FP/h together with the onset ratio, the fraction of alarms falling within ±15 batches of the onset (n/a if no alarm is raised at all).

At $A R L _ { 0 } = 1 0 0 0 0$ , no false alarm falls within the window for any degradation level – including a fully missing $9 0 °$ sector, arguably the most severe case tested – confirming that degradation is not confused with calibration drift (with only 0.774 h of aggregated exposure, the empirical $A R L _ { 0 }$ here is censored, as in Section 5).

Sensitivity under concurrent miscalibration. We then inject a genuine calibration drift on top of the degraded stream, using a three-phase protocol: clean segment, degradation onset at 15%, drift onset at 35%.

<table><tr><td>Degradation</td><td>FNR FPR Delay</td></tr><tr><td>64→32 rings</td><td>.207 .000 12.3 s</td></tr><tr><td>64→16 rings</td><td>.379 .000 13.9 s</td></tr><tr><td>64→8 rings</td><td>.600 .040 18.2 s</td></tr><tr><td> $4 5 ^ { \circ }$  sector missing</td><td>.207 .000 12.7 s</td></tr><tr><td> $9 0 °$  sector missing</td><td>.207 .000 13.4 s</td></tr></table>

Table 11. Detection of a calibration drift injected on top of an already-degraded LiDAR stream, on KITTI Raw, at $A R L _ { 0 } ~ =$ 10000.

![](images/101bc81b7da4f2da061c84a3a975addec6793d48e6c335d48b2ed02add631f78.jpg)  
Figure 7. The experimental vehicle used for data collection.

![](images/9193c711cb02b763eae747a3c44b60556976cf1698077e5291c83c8f15365870.jpg)  
(a) LiDAR: Ouster OS1.

![](images/8f7ed3ff36ea59a2c50d4cc6cbc2f07a57ed8fd7d32378774d3a2e1845bed4a1.jpg)  
(b) RGB Camera: Lucid Triton 5.4MP.  
Figure 8. Sensors mounted on top of the vehicle used to collect the data.

At $A R L _ { 0 } = 1 0 0 0 0$ , the FPR stays (almost) zero across all degradation levels. Sensitivity remains good under moderate beam loss and either sector loss, but degrades markedly at only 8 of 64 rings (FNR 0.600, FPR no longer zero); this rise in FNR is driven by the LiDAR object detector itself, which becomes less robust and produces fewer valid matches under such severe degradation, rather than by a failure of the monitoring scheme.

Take-away. Sensor degradation and calibration drift are largely distinguishable by MiCaDe: even a full 90<sup>◦</sup> sector loss is never confused with miscalibration, and moderate beam loss does not compromise detection of a concurrent drift. Performance only degrades under severe beam loss, an effect bounded by the object detector’s own robustness to sparse input rather than by the monitoring scheme itself.

## C. Real-World Dataset

Our real-world evaluation is conducted on a custom dataset collected using the sensors shown in Figure 8. In this section, we provide further details with respect to the ones presented in Section 5.6. The sensing suite for data collection comprises:

![](images/aa9e680b88bccdb9711e9ca33a394c7b59130b59d7514b5dd2101b4ca25258e1.jpg)  
Figure 9. An example of point cloud taken from the Ouster OS1 LiDAR.

![](images/c2779d38c318063dde66575a4129fe88c5a0ef62d42b3add6b8c3cb0e3d550ff.jpg)  
Figure 10. An example of front and left images taken from the RGB cameras.

• A 128-beam Ouster OS1 LiDAR, operating at 10 Hz, with a 360<sup>◦</sup> horizontal field of view, 45<sup>◦</sup> vertical field of view. Figure 9 shows an example of an OS1-128 LiDAR scan.

• Three Lucid Triton 5.4 MP RGB cameras, mounted as front-left, front-center, and front-right; each operating at 30 Hz with a resolution of 1440×720 pixels. Figure 10 shows an example of front-left and front-center RGB images.

All sensors are hardware-timestamped but run at different frame rates. To build consistent multimodal frames, we perform temporal alignment by matching each LiDAR scan with the closest RGB images from all cameras. Pairs whose timestamps differ by more than 0.02 s are discarded. This procedure produces a stream of temporally synchronized multimodal frames.

Data Collection Protocol. The dataset contains approximately one hour of real urban driving, collected in residential and light-traffic areas. The vehicle follows a standard route including: straight roads, intersections, slow traffic segments, and mild accelerations/braking. This ensures diverse distributions of object distances, orientations, and occlusion patterns.

During the first half of the recording, all sensors are precisely calibrated, and the system is operated in nominal conditions. During the second half, we intentionally introduce a controlled misalignment on the front-left camera:

![](images/afbf5d0aa2b9fce16f91aff8205ece600b8e50ced587bca2cca33999305829b3.jpg)  
Figure 11. An example of a calibrated frame. The projection of the points is precisely inside the objects, up to small errors due to temporal synchronization.

![](images/2c10574022ebfb3e765f89892e3a055f80d3e398ac4a20455a8641e4639fb688.jpg)  
Figure 12. An example of an uncalibrated frame. The projection of the points is clearly inconsistent, which is visible from the red car in front of the camera.

• rotation of 3<sup>◦</sup> in roll, 3<sup>◦</sup> in pitch, and 3<sup>◦</sup> in yaw;

• no induced translation shift (only rotational perturbation), matching common real-world mechanical drift.

The perturbation is physically introduced by loosening and rotating the camera mount before tightening it again. As a result, only rotational miscalibrations can be induced in a controlled and repeatable manner, whereas introducing a pure translational offset would require physically relocating the sensor and redesigning the mounting system. Once rotated, the camera remains rigidly fixed, so the induced drift is constant throughout the miscalibrated portion of the sequence.

This procedure yields a total of 2151 calibrated frames and 5234 miscalibrated frames. An example calibrated frame is shown in Figure 11, while Figure 12 shows a frame after miscalibration, where projected LiDAR points appear systematically shifted with respect to object edges.

Reference Frames for the GMM. To build the reference distribution used by the change detector, we select the first 1000 calibrated frames, using only those with at least one valid 3D-2D detection match. These frames define the nominal geometric consistency statistics for the sensor configuration.

Usage in the Experiments. After the initial 1000 reference frames, the full MiCaDe pipeline is applied to the remaining calibrated portion and the entire miscalibrated portion. We preserve natural traffic conditions and do not inject synthetic noise, ensuring that the evaluation reflects real detection variability, occlusions, and motion blur. Figure 13 reports the evolution of the complete set of monitored metrics on this sequence, extending the three metrics shown in Figure 5 of the main paper.

Release. The dataset, including the raw multimodal recordings and the known calibration ground truth, will be made publicly available upon acceptance of the paper.

## D. Qualitative Change-Detection Examples

Figures 14–17 provide qualitative examples of MiCaDe on KITTI Raw, ZOD, and ONCE. Each example reports the temporal evolution of the monitored metrics before and after the injected calibration change, together with two representative image frames. Blue curves correspond to calibrated observations, orange curves to observations after the perturbation, the black dashed line marks the ground-truth change point, and the green dashed line indicates the alarm raised by MiCaDe. The bottom images show the geometric relationship between 2D detections (green boxes), projected 3D boxes (red dashed boxes), and the displacement vectors used to compute the monitored consistency metrics.

The true-positive examples in Figures 14, 15, and 16 show that calibration drift typically induces coherent changes across several metrics. In particular, the IoU decreases after the perturbation, while the displacement magnitude and center-offset components increase. Directional metrics also become more structured, reflecting the emergence of a preferred displacement direction between projected LiDAR boxes and image detections. These changes are visually consistent with the image examples, where the projected 3D boxes become systematically shifted with respect to the corresponding 2D detections after the perturbation.

The examples also illustrate the difference between delay measured in batches and delay measured in frames. MiCaDe operates on batches of n = 15 matched detections, not on individual frames. Therefore, the conversion from batch delay to frame delay depends on how many valid 3D–2D correspondences are available in the scene. In dense scenes with many detected objects, batches are filled quickly and a small batch delay corresponds to only a few frames. This is visible in the ZOD example, where the change is detected in the same batch as the perturbation and corresponds to a delay of only 3 frames. Similarly, the KITTI example is detected after 1 batch, corresponding to 4 frames, while the ONCE true-positive example is detected after 4 batches, corresponding to 14 frames.

![](images/6d87db42958112d54d6218d90d5e06fcd67474f52cd41837480a87d04359e024.jpg)  
Figure 13. Evolution of the complete set of monitored metrics and detector signal on the real-world sequence. The black dashed line marks the manually induced calibration change, while the green line indicates the detection raised by MiCaDe. Following the perturbation, all metrics exhibit visible shifts, including the center displacement terms $( e _ { x } , e _ { y } )$ , the displacement magnitude (e<sub>a</sub>), and the directiona indicators $( e _ { \mathrm { s i n } } , e _ { \mathrm { c o s } } , \sigma _ { \mathrm { s i n } } ^ { 2 } , \sigma _ { \mathrm { c o s } } ^ { 2 } )$ . The bottom panel reports the log-likelihood stream and the corresponding CPM statistic, which crosses the decision threshold shortly after the perturbation, correctly identifying the calibration drift.

Figure 17 shows a representative false-negative case on ONCE. In this sequence, the injected perturbation is very small: the rotational offsets are close to zero and the translations are below 0.1 m. As a consequence, the post-change metrics remain close to the variability observed during the calibrated portion of the sequence. The IoU, displacement magnitude, directional components, and depth error do not exhibit a persistent shift strong enough to separate the miscalibrated regime from nominal fluctuations. Visually, the projected 3D boxes remain only mildly misaligned with the 2D detections. This example highlights a natural limitation of object-level calibration monitoring: when the induced calibration drift is too small to produce a measurable change in geometric consistency, the sequential detector may not accumulate sufficient evidence to raise an alarm.

Overall, these qualitative examples confirm the behaviour observed quantitatively in the main paper. MiCaDe detects calibration changes quickly when the perturbation produces a consistent geometric signature across matched detections, while failures mainly occur for very small perturbations whose effect is comparable to natural detection noise, scene variability, and matching uncertainty.

KITTI — seq 2011 09 26.2011 09 26 drive 0104 sync — TP (delay = 1 batch = 4 frames) — GT: batch 33, frame 94 <sup>∆(r,</sup> <sup>p,</sup> <sup>y,</sup> <sup>t</sup>x<sup>,</sup> <sup>t</sup>y<sup>,</sup> <sup>t</sup>z<sup>)</sup> <sup>=</sup> <sup>(</sup>−<sup>0.017,</sup> <sup>0.048,</sup> −<sup>0.015</sup> <sup>rad,</sup> <sup>0.042,</sup> −<sup>0.094,</sup> −<sup>0.099</sup> <sup>m)</sup>  
![](images/fb8bb3dec997fc5f12485c6979fac55c69c2f275ebbbaa15927c0d6b3c2c5fa8.jpg)  
Figure 14. Qualitative true-positive example on KITTI Raw. The perturbation is detected after 1 batch, corresponding to 4 frames.

ZOD — seq 000028 — TP (delay = 0 batches = 3 frames) — GT: batch 40, frame 280 ∆(r, p, y, t , t , t ) = ( 0.043, 0.045, 0.040 rad, 0.086, 0.060, 0.051 m)  
![](images/4bb259e2580016067ac6b94182239ad9e0963b0abb085d1a6cadfa7e90af6f61.jpg)  
Figure 15. Qualitative true-positive example on ZOD. MiCaDe detects the perturbation in the same batch, corresponding to a delay of 3 frames.

![](images/749ee6da0258cff70de315aba3f97343d86041748ed3809a23deee2df3e3452d.jpg)

![](images/a10b9680745a4109c765bd1e8b55d42b7c7ff2bf111c61d9c9f34dff31521f34.jpg)  
Figure 16. Qualitative true-positive example on ONCE. The perturbation is detected after 4 batches, corresponding to 14 frames

$$
\begin{array} { r } { \mathrm { O N C E } \operatorname { - s e q } \ 0 0 0 0 2 8 \operatorname { - F N } \ ( \mathrm { m i s s e d ~ d e t e c t i o n } ) \operatorname { - G T } \mathrm { : ~ b a t c h ~ 1 8 5 , ~ f r a m e ~ 6 1 8 } } \\ { \Delta ( r , p , y , t _ { x } , t _ { y } , t _ { z } ) = ( - 0 . 0 0 6 , 0 . 0 1 1 , \ 0 . 0 0 1 \ \mathrm { r a d } , \ 0 . 0 4 8 , \ - 0 . 0 4 4 , \ - 0 . 0 8 1 \ \mathrm { m } ) } \end{array}
$$

![](images/bd6562dc84431d2c8f1a7fa40660adae844aac9352fc9e0b4aa23fb85d71665f.jpg)  
Figure 17. Qualitative false-negative example on ONCE. The perturbation is very small and produces only mild changes in the monitored metrics, preventing MiCaDe from raising an alarm.