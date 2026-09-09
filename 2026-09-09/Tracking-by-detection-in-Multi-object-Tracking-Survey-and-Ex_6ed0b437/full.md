# Tracking-by-detection in Multi-object Tracking: Survey and Experiments

Yujin Yang<sup>1†</sup>, Kyujin Shim<sup>1†</sup>, Kangwook Ko<sup>1</sup>, Changick Kim<sup>1\*</sup>

<sup>1\*</sup>School of Electrical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, 34141, South Korea.

\*Corresponding author(s). E-mail(s): changick@kaist.ac.kr; Contributing authors: ujin.y@kaist.ac.kr; kjshim1028@kaist.ac.kr; kw.ko@kaist.ac.kr; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Multi-object tracking (MOT) is an essential computer vision task that simultaneously tracks multiple objects in video sequences, with various applications in surveillance, autonomous navigation, and human-computer interaction. The tracking-by-detection (TBD) paradigm, which combines object detection with temporal association, has emerged as a leading approach, driven by innovative algorithms. Despite recent progress, fair evaluation of TBD-based methods remains a challenge. Many studies introduce modules such as similarity metrics, data association strategies, or motion models, but they are often evaluated under inconsistent protocols, with diferent baseline trackers, hyperparameters, and datasets. Such inconsistencies obscure the genuine contribution of each module and hinder objective comparison. This survey systematically reviews TBD-based MOT techniques, including similarity measurements, data association, camera motion compensation, and interpolation strategies. Starting from a minimal baseline tracker, we fairly evaluate the contributions of each method across diverse datasets and accumulate well-balanced methods. Our findings establish a strong baseline tracker and provide a foundation for the principled design of robust and versatile MOT systems suitable for real-world deployment.

Keywords: Multi-object tracking, Tracking-by-detection, Object Tracking, Computer vision, Deep learning

## 1 Introduction

Multi-object tracking (MOT) is a fundamental task in computer vision, focused on simultaneously tracking multiple objects across sequential data, such as video streams. This task is essential for various applications, including intelligent surveillance [1–4], autonomous vehicle navigation [5–8], and human-computer interaction [9, 10]. Despite its importance, MOT remains challenging due to persistent issues such as occlusions, significant variations in object appearance, and dynamic motion of both cameras and targets. To address these complexities, numerous approaches have been developed, with the tracking-by-detection (TBD) paradigm emerging as a dominant framework. In TBD, objects are first detected in each frame using advanced object detection algorithms [11], followed by temporal association of these detection results to construct complete object trajectories. By leveraging recent advancements in object detection, feature extraction, and data association, TBD-based trackers [3, 12–17] consistently achieve state-of-the-art performance, significantly advancing the field of MOT.

Within the TBD framework, each tracker introduces distinctive innovations for MOT. SORT [12] is a seminal tracking-by-detection baseline that demonstrates how a carefully designed yet minimalist pipeline can achieve competitive performance. It employs a linear Kalman filter [18] with a constant-velocity motion model, where the state vector represents the bounding box center coordinates, scale, aspect ratio, and their respective velocities. For data association, SORT constructs an IoUbased cost matrix between predicted tracklets and current detections, and solves the assignment problem using the Hungarian algorithm [19]. Notably, it relies solely on motion and geometric cues without incorporating appearance features, enabling high computational eficiency and real-time operation.

Building upon this foundation, subsequent trackers introduce additional components to enhance robustness. ByteTrack [14] improves tracking robustness by associating all detected bounding boxes, including those with low confidence scores, thereby minimizing missed tracks. BoT-SORT [15] further enhances performance by integrating re-identification models, optimizing the state vector of the Kalman filter, and incorporating camera motion compensation to enhance stability in dynamic scenes. Hybrid-SORT [20] leverages weak cues, such as tracklet confidence and heightmodulated Intersection-over-Union (IoU), to improve data association in challenging scenarios with significant object overlap. These advancements highlight the ongoing evolution of TBD-based MOT systems, with novel contributions continually enhancing the reliability and accuracy of object tracking in complex, dynamic environments.

Despite significant advancements in TBD-based MOT methods, evaluating the efectiveness of individual tracking modules remains challenging due to inconsistencies in common components across studies, as shown in Table 1. Variations in detection algorithms, feature extraction methods, and hyperparameter settings hinder direct comparisons and obscure the relative merits of diferent approaches. Moreover, many trackers [14, 15, 21–23] are primarily evaluated on datasets such as MOT17 [24], which mainly involve linear motion and distinguishable object appearances. This limits the assessment of robustness in more challenging scenarios, such as non-linear motion or visually similar objects, as emphasized in datasets like DanceTrack [25]. In addition, commonly used data-splitting protocols construct training and validation sets by temporally dividing the same video sequences, leading to overlap in scene context and object identities. To ensure fair and comprehensive evaluation, standardized protocols with diverse datasets and proper data splits are essential.

Table 1: Comparison of common tracking-by-detection components among diferent trackers. ‘NMS Thr.’ denotes threshold value for Non-Maximum Suppression, ‘CMC denotes Camera Motion Compensation, ‘Post-Proc.’ denotes Post-Processing.
<table><tr><td>Tracker</td><td>Detector</td><td>Feature Extractor</td><td>NMS Thr.</td><td>CMC</td><td>Post-Proc.</td><td>Validation</td></tr><tr><td>SORT [12]</td><td>FrCNN [26]</td><td></td><td>N/A</td><td></td><td></td><td>MOT15</td></tr><tr><td>DeepSORT [13]</td><td>FrCNN [26]</td><td>WRN [27]</td><td>N/A</td><td></td><td></td><td>MOT16</td></tr><tr><td>ByteTrack [14]</td><td>YOLOX-X [11]</td><td>-</td><td>0.70</td><td></td><td>LI [14]</td><td>MOT17, MOT20</td></tr><tr><td>MAATrack [28]</td><td>CrowdDet [29]</td><td></td><td>0.50</td><td>GMC [30]</td><td>LI</td><td>MOT17, MOT20</td></tr><tr><td>StrongSORT [23]</td><td>YOLOX-X</td><td>BoT [31]</td><td>0.80</td><td>ECC [32]</td><td>GSI + AFLink [23]</td><td>MOT17, MOT20, DanceTrack</td></tr><tr><td>BoT-SORT [15]</td><td>YOLOX-X</td><td>SBS-50 [33]</td><td>0.65</td><td>GMC</td><td>LI</td><td>MOT17, MOT20</td></tr><tr><td>Deep OC-SORT [34]</td><td>YOLOX-X</td><td>SBS-50</td><td>0.70</td><td>GMC</td><td>LI</td><td>MOT17, MOT20, DanceTrack</td></tr><tr><td>HybridSORT [20]</td><td>YOLOX-X</td><td>SBS-50</td><td>0.70</td><td>ECC</td><td>LI</td><td>MOT17, MOT20, DanceTrack</td></tr><tr><td>BoostTrack [35]</td><td>YOLOX-X</td><td>SBS-50</td><td>0.3</td><td>ECC</td><td>GBI [36]</td><td>MOT17, MOT20</td></tr></table>

In this manner, this survey paper systematically reviews and evaluates TBD-based MOT techniques, including key components such as similarity measurements, data association, camera motion compensation, and interpolation strategies. Our evaluation methodology employs a SORT [12]-based TBD tracker as an initial baseline and incrementally integrates the most efective method from each functional category step-by-step. We categorize existing techniques according to their role in the tracking pipeline and systematically assess the efectiveness and robustness of each component. To ensure fair comparison, hyperparameters are carefully optimized for each experiment. Through this process, we identify the most efective design choices and progressively incorporate them into a unified baseline tracker.

All experiments are conducted on three diverse MOT benchmarks, namely MOT17 [24], MOT20 [37], and DanceTrack [25], which encompass a wide spectrum of motion dynamics, object densities, and scene complexities. We construct our splits using entirely diferent video sequences within each dataset. This protocol mitigates temporal and scene-level overlap between splits, thereby enabling a more rigorous evaluation of generalization across unseen scenarios.

Finally, we establish a generalized baseline tracker that demonstrates consistent performance across all evaluated datasets, providing a solid foundation and practical guidance for the development of robust and versatile MOT systems suitable for realworld deployment.

The main contributions of our article are as follows:

• We systematically evaluate the efectiveness of tracking-by-detection (TBD) tech niques by incrementally integrating the most efective methods from each functional category into a standardized baseline tracker, enabling precise assessment of individual contributions to MOT performance.

• We conduct a fair and comprehensive comparison of recent TBD trackers, and derive practical insights and design guidelines for building reliable and versatile MOT systems in real-world scenarios.

• Our study establishes a strong and generalized baseline tracker that performs robustly across diverse datasets, addressing limitations of existing evaluations.

The remainder of our paper is formed as follows. Section 2 introduces the problem formulation and SORT-based initial baseline tracker. Then, Section 3 presents the MOT techniques that are compared, and Section 4 provides an explanation of the object detector, feature extractor, datasets, and metrics we used. Based on this setup, Section 5 presents the detailed experimental results. Finally, Section 6 summarizes our findings, discusses their implications, and suggests potential directions for future research.

## 2 Preliminaries

## 2.1 Problem Formulation

Following the framework of tracking-by-detection paradigm, we first detect a set of detection results $\mathcal { D } ^ { t } = \{ { \bf d } _ { 1 } ^ { t } , . . . , { \bf d } _ { M } ^ { t } \}$ from the current frame at time $t ,$ where $\mathbf { d } _ { i } ^ { t } = \mathbf { \Phi }$ $( \mathbf { b } _ { i } ^ { t } , s _ { i } ^ { t } , \mathbf { f } _ { i } ^ { t } )$ $\mathbf { b } _ { i } ^ { t }$ represents the bounding box location, $s _ { i } ^ { t }$ is the predicted confidence score of the current detection, and $\mathbf { f } _ { i } ^ { t }$ indicates the appearance feature extracted from the patch cropped with the corresponding bounding box. As a second step, we predict the current bounding box location of each object track $\tau _ { j }$ that is tracked until the previous frames by using a motion model. Note that $\tau _ { j }$ is a set of merged detection results that are determined as the same object. Finally, we properly associate the detection results $\mathcal { D } ^ { t }$ and object tracks $\mathcal { O } = \{ \mathcal { T } _ { 1 } , . . . , \mathcal { T } _ { N } \}$ through bipartite matching based on their computed pairwise distances, which are measured with their detected or estimated current locations and appearance information. This procedure is iterated over the timeline, thereby resulting in complete tracks from the input video.

## 2.2 Baseline Tracker

Our initial baseline tracker is based on the SORT [12] framework, the most basic algorithm of the TBD tracker, and operates by processing each input frame to detect and track multiple objects. Algorithm 1 outlines the key steps involved in the tracker, including object detection, track prediction, distance measurement, data association, and track management. In detail, for each frame $f ^ { t }$ in a video sequence $\nu ,$ we first detect objects resulting in a set of high confidence detection results $\mathcal { D } ^ { t }$ , whose confidence scores exceed the confidence score threshold $\tau _ { c }$ (line 2 to 4). Next, the tracker predicts the current locations of the tracks O using a Kalman Filter [18], and the predicted bounding boxes $B ^ { t }$ are used to compute IoU cost matrix $C _ { 1 }$ with the detection results $\mathcal { D } ^ { t }$ (line 6 to 13). Based on the cost matrix and a matching threshold $\tau _ { m }$ , bipartite matching is performed with the Hungarian algorithm [19], and we merge each matched detection result to the corresponding matched track (lines 14 to 15). We also update the set of tracks O and identify remaining unassociated detection results $\mathcal { D } _ { r } ^ { t }$ (lines 16 to 18).

Algorithm 1 Baseline Tracker   
Require: A input video sequence V; Kalman Filter $\mathrm { K F } ;$ confidence score threshold $\tau _ { c } ;$ matching threshold   
τm; minimum length $L _ { m i n }$   
Ensure: Complete tracks O   
1: Initialization: $\mathcal { O }  \emptyset ; \mathcal { O } _ { u }  \emptyset ;$   
2: for frame $f ^ { t } \mathrm { ~ i n ~ } \nu$ do   
3: # Detect objects in the current frame   
4: $\ddot { \mathcal { D } } ^ { t } \gets \{ \mathbf { d } _ { 1 } ^ { t } , . . . , \mathbf { d } _ { M } ^ { t } | s _ { i } > \tau _ { c } \}$   
5:   
6: # Predict a current box location for each track   
7: $\ddot { B } ^ { t } \gets \emptyset$   
8: for T in O do   
9: $\dot { B } ^ { t } \gets \dot { B } ^ { t } \cup \{ \mathrm { K F } ( T ) \}$   
10: end for   
11:   
12: # Distance measurement and association   
13: $\ddot { C } _ { 1 } \gets I o U ( \mathcal { D } ^ { t } , \mathcal { B } ^ { t } )$   
14: Bipartite matching based on $C _ { 1 }$ and $\tau _ { m }$   
15: Associate each detection result to the matched track   
16: $\scriptscriptstyle \mathcal { O } \gets$ matched tracks   
17: $\mathcal { O } _ { u }$ ← remaining tracks   
18: $\mathcal { D } _ { r } ^ { \bar { t } } \gets$ remaining detection results   
19:   
20: # Predict a current box location for each track   
21: $\ddot { B } _ { u } ^ { t } \gets \emptyset$   
22: $\mathbf { f o r } \ T \mathrm { ~ i n ~ } \mathcal { O } _ { u }$ do   
23: $\dot { B } _ { u } ^ { t } \gets \dot { B } _ { u } ^ { t } \cup \{ \mathrm { K F } ( T ) \}$   
24: end for   
25:   
26: # Distance measurement and association   
27: $\ddot { C } _ { 2 } \gets I o U ( \mathcal { D } _ { r } ^ { t } , \mathcal { B } _ { u } ^ { t } )$   
28: Bipartite matching based on $C _ { 2 }$ and τm   
29: Associate each detection results to the matched track   
30: $\mathcal { O } _ { u } \gets$ matched unconfirmed tracks   
31: $\mathcal { D } _ { r r } ^ { t } $ remaining detection results   
32:   
33: # Confirm the unconfirmed tracks   
34: for T in $\mathcal { O } _ { u }$ do   
35: $\textbf { f } l e n ( \mathcal { T } ) > L _ { m i n }$ then   
36: $0  0 \cup \{ \mathcal { T } \}$   
37: end if   
38: end for   
39: $\mathcal { O } _ { u } \gets \mathcal { O } _ { u } \backslash \mathcal { O }$   
40:   
41: # Initialization   
42: for d<sup>t</sup> in $\mathcal { D } _ { r r } ^ { t }$ do   
43: if $\mathbf { d } _ { \quad \cdot } ^ { t } c ^ { t } > \tau _ { c }$ then   
44: $\mathcal { O } _ { u }  \mathcal { O } _ { u } \cup \{ \{ { \bf d } ^ { t } \} \}$   
45: end if   
46: end for   
47: end for   
48:   
49: Return: O

The tracker then focuses on unconfirmed tracks $\mathcal { O } _ { u }$ , which are tracked for less than $L _ { m i n }$ number of frames until the last frame. Similar to the previous step, we predict their current locations, measure IoU distances with the remaining detection results $\mathcal { D } _ { r } ^ { t }$ , and associate the matched pairs (lines 19 to 28). Then, we update the set of unconfirmed tracks $\mathcal { O } _ { u } ^ { t }$ only with matched tracks and identify unassociated detection results $\mathcal { D } _ { r r } ^ { t }$ (lines 29 to 30). These separated association steps prioritize the confirmed tracks to be first associated with the detection results rather than the unconfirmed tracks, which can be short and noisy, blocking their possible disturbance. After the associations, we check the length of each unconfirmed track and move the tracks that exceed a minimum length $L _ { m i n }$ to O (lines 32 to 38). Finally, remaining unassociated detection results with high-confidence scores are initialized as new tracks and added to $\mathcal { O } _ { u }$ (lines 40 to 45). Note that the re-matching of lost tracks is omitted in the pseudocode for simplicity. In the actual implementation, we chase the lost tracks using the motion model and try to match them with the detection results in the first association step for $L _ { l o s t }$ number of frames.

![](images/40d8fe7f2ef4a559296613512a8bf95f7cfc45a03a078053b32e136160b2dd20.jpg)  
Fig. 1: An overview structure of a typical TBD-based MOT tracker. The contents in the dashed boxes are the main techniques that we applied in our experiments.

## 3 Key Components of Tracking-by-Detection

The tracking-by-detection (TBD) framework has become a dominant paradigm in modern multi-object tracking, where tracking is formulated as associating detection results across consecutive frames. To improve tracking performance under diverse and challenging scenarios, numerous methods have been proposed, introducing design variations at diferent stages of the TBD pipeline.

In this section, we categorize these methods based on their functional roles within the tracking process and systematically review representative design choices for each component. Specifically, we focus on key components—such as the Kalman filter state representation, track initialization strategies, data association mechanisms, camera motion compensation, and post-processing techniques—and summarize commonly adopted approaches as well as notable advancements proposed in prior works. As illustrated in Fig. 1, each component corresponds to a specific stage within the overall tracking pipeline.

## 3.1 Kalman State Vector

In MOT, the Kalman filter predicts and updates the states of tracked objects over time. The Kalman filter iteratively performs two steps: prediction and update. In the prediction step, the object’s state in the next frame is estimated using a motion model, typically assuming constant velocity. In the update step, this prediction is refined by incorporating the associated detection, correcting the estimate based on the observed measurement.

To represent the current state of a tracking object eficiently, the state vector typically includes parameters describing the object’s position, size, and motion. The design of the Kalman state vector plays a critical role in tracking performance, as it determines which spatial and motion attributes are modeled and propagated over time. Commonly used state vectors in MOT include:

$( c x , c y , s , a )$ : This Kalman state vector is applied for [12]. Here, cx and cy represent the center coordinates of the object bounding box, s denotes its scale (area), and a represents the aspect ratio. This formulation is useful for tracking objects, such as people or vehicles, where size and shape are essential.

• (cx, cy, a, h): This Kalman state vector is introduced by [13]. In this variation, cx and cy still represent the center coordinates, a is the aspect ratio, and h is the height of the object. This state vector is efective when the height is a more reliable feature for tracking than the overall scale, particularly in scenes where object height remains consistent.

$( c x , c y , w , h )$ : This Kalman state vector is introduced by [15]. In this configuration, cx and cy define the center of the object, while w and h represent its width and height, respectively. This representation is widely used in object detection tasks, as it directly models the bounding box of the target object.

## 3.2 Track Initialization

Track initialization is a crucial step in TBD frameworks, as it determines how new tracks are created from unmatched detection results after association. Traditional TBD methods [14, 15, 38] rely on heuristic rules to minimize ghost tracks caused by false positives, often applying a higher confidence threshold for initialization than for association. To utilize surrounding information, Occlusion-Aware Initialization (OAI) [39] enhances the track initialization by incorporating spatial context. OAI calculates the intersection-over-union (IoU) between unmatched detection results and existing track boxes, preventing initialization if the maximum IoU exceeds a predefined threshold, and reducing duplicate tracks. With confidence thresholds and tentative track strategies, OAI ofers a robust solution for initializing tracks in complex, densely populated environments.

## 3.3 Kalman Filter Prediction & Update Strategy

Several strategies have been proposed for updating Kalman state vectors: fixing some parts of the state vectors of inactive tracks [28, 40], Confidence Weighted Kalman Update (CKWU) [40], and Observation-Centric Re-Update (ORU) [41]. The first strategy addresses issues caused by occlusion, where changes in box size lead to identity loss; solutions include preserving the height [28] or both height and width of inactive track boxes [40] during prediction. CKWU in ConfTrack [40] adjusts the Kalman filter update by incorporating the confidence score of the detection box. ORU, introduced by OC-SORT [41], mitigates accumulated errors in Kalman filter parameters during untracked periods by recalculating Kalman parameters using a virtual trajectory based on observations before and after the untracked period.

## 3.4 Basic Parameters

In MOT, several key parameters are used to enhance tracking accuracy. Non-maximum suppression (NMS) threshold helps eliminate duplicate detection results by filtering out overlapping bounding boxes, keeping only the one with the highest confidence score for each existing object. The minimum frame count for track activation sets the required number of consecutive frames in which an object must be detected to be considered a valid track, ensuring robustness to false positives. Meanwhile, the maximum frame count for lost frames defines how long a track can remain undetected before it is permanently deactivated, allowing for temporary occlusions while avoiding excessive track fragmentation.

## 3.5 Camera Motion Compensation

Camera motion compensation (CMC) is crucial for maintaining tracking accuracy in scenes with significant camera movement. It compensates for the motion of the camera to enhance the stability of object tracking. Two common methods are global motion compensation (GMC) [30] and enhanced correlation coeficient maximization (ECC) [32]. GMC models global camera motion across the entire scene, ofering a simpler but efective correction for consistent background movement, improving overall tracking performance. Conversely, ECC aligns frames by maximizing the correlation between consecutive frames, compensating for complex motion.

## 3.6 Main Spatial Distance

In MOT, spatial distance metrics are critical for assessing the similarity between tracks and detected bounding boxes. Intersection over Union (IoU) usually serves as the standard metric, measuring the overlap between two given boxes, A and $B ,$ as defined by:

$$
I o U = \frac { | A \cap B | } { | A \cup B | } .\tag{1}
$$

To address limitations in IoU, advanced variants are proposed by adding penalty terms into IoU. Generalized-IoU (GIoU) [42] considers the smallest enclosing rectangular box C which contains both boxes A and B as shown in Fig. 2,

$$
G I o U = I o U - \frac { | C \setminus ( A \cup B ) | } { | C | } .\tag{2}
$$

However, GIoU only focuses on the area perspective, Distance-IoU (DIoU) and Complete-IoU (CIoU) [43] introduce additional penalty terms with the center distance and aspect ratio, respectively, and are defined as

$$
D I o U = I o U - { \frac { \rho ^ { 2 } ( a , b ) } { c ^ { 2 } } } ,\tag{3}
$$

![](images/2b952967ae548bf6e3a03cdb56ade226d90aca71877c1960da9725f578070cd7.jpg)  
Fig. 2: Illustration of bounding boxes. A rectangular $C$ is the smallest enclosing rectangle that contains boxes A and B. Dot a and dot b are the centers of boxes A and B, respectively.

$$
C I o U = D I o U - \alpha v ,\tag{4}
$$

where

$$
v = \frac { 4 } { \pi ^ { 2 } } ( a r c t a n \frac { w ^ { A } } { h ^ { A } } - a r c t a n \frac { w ^ { B } } { h ^ { B } } ) ^ { 2 } ,\tag{5}
$$

$$
\alpha = \frac { v } { ( 1 - I o U ) + v } .\tag{6}
$$

Here, as shown in Fig. 2, a and b are the central points of boxes A and $B , c$ is a diagonal length of the enclosing rectangle $C , \rho ( \cdot )$ is the Euclidean distance, and $w ^ { A } , h ^ { A } , w ^ { B }$ and $h ^ { B }$ represent the width and height of boxes A and B, respectively. Complete-IoU (CIoU) [43] factors in both distance and aspect ratio, improving accuracy in object matching.

Bufered IoU (BIoU) [38] further refines IoU by bufering boundaries for better localization, as shown in Fig. 3:

$$
B I o U = I o U ( { \mathrm { b u f f e r e d ~ } } A , { \mathrm { b u f f e r e d ~ } } B ) ,\tag{7}
$$

$$
\mathrm { B u f f e r ~ S c a l e } \ b = \frac { b \_ h - h } { 2 h } ,\tag{8}
$$

where $b _ { h }$ is the bufered height and h is the original height. Similarly, Height Modulated IoU (HMIoU) [20] adjusts IoU by emphasizing height diferences, which is particularly efective for tracking objects with distinct vertical size characteristics.

$$
H M I o U = H I o U \cdot I o U ,\tag{9}
$$

$$
H I o U = { \frac { \mathrm { h e i g h t ~ o f ~ } } { \mathrm { h e i g h t ~ o f ~ } } } A \cap B .\tag{10}
$$

![](images/5e0659cf7e6f6e6304b74bffa64903576a8199f70f4e06a69913f21bda934c0d.jpg)  
Buffered A  
Fig. 3: Illustration of bufered bounding boxes. Dashed rectangles are the bufered boxes that are utilized for BIoU[38].

## 3.7 Additional Spatial Distance

Incorporating advanced spatial distance metrics into IoU calculations can significantly enhance tracking performance. Observation-Centric Momentum (OCM) [41] mitigates noise in motion direction estimation by leveraging state observations instead of estimations, thereby improving velocity consistency and tracking robustness. Hybrid-SORT [20] refines OCM by integrating a more robust and detailed model for more comprehensive and precise object velocity representation. In addition, it introduces a confidence cost, which is the absolute diference between track and detection confidence scores. Similarly, BoostTrack [35] presents similarity matrix boost techniques, including Detection-Tracklet Confidence Similarity Boost, Mahalanobis Distance Similarity Boost, and Shape Similarity Boost, which refine the base similarity matrix by adding specially designed similarity metrics to improve the accuracy and reliability of object matching in complex tracking scenarios.

## 3.8 Tricks of Spatial Distance

Recent tricks in spatial distance metrics enhance the robustness of data association. BoostTrack [35] employs detection confidence boosting to elevate the confidence scores of true positive low-confidence detection results, termed Detecting Likely Objects (DLO), which often arise due to partial occlusions. Specifically, confidence scores of DLO are boosted using the maximum IoU with existing tracks. Conversely, for Detecting Unlikely Objects (DUO) that have low-confidence scores throughout a video due to persistent occlusions, BoostTrack computes their Mahalanobis distance with existing tracks, finds boxes whose all distances exceed a predefined threshold, and increases their confidence scores by considering them as true positives. ConfTrack [40] introduces a Confidence-Fused Cost Matrix, which integrates IoU-based costs with appearance feature-based costs by multiplying the cost matrix with detection confidence scores, thereby constructing a single unified cost matrix. Similarly, PIA [44] leverages historical detection data alongside current detection results to compute the cost matrix, efectively incorporating temporal information to improve tracking robustness.

## 3.9 Appearance Feature Update

Appearance feature update strategies play a crucial role by enabling robust identity preservation throughout the timeline under varying conditions. Three prominent approaches, namely Feature Bank [13], Exponential Moving Average (EMA) [45], and Dynamic Appearance [34], ofer distinct mechanisms for managing the appearance features of tracked objects. The feature bank [13] approach maintains a static repository of appearance features for each object. Then, it uses them to calculate the maximum similarity value between each detection result and track, providing a stable reference for matching. On the other hand, EMA [45] is an adaptive mechanism for updating appearance features by blending newly observed appearance features with historical data using a weighted average. This strategy allows the tracker to gradually adapt to changes in object appearance while preserving the earlier observations. More specifically, the update process is defined as,

$$
e _ { i } ^ { k } = \alpha e _ { i } ^ { k - 1 } + ( 1 - \alpha ) f _ { i } ^ { k } ,\tag{11}
$$

where $e _ { i } ^ { k }$ is an appearance feature for the $i ^ { t h }$ tracks at frame $k ,$ and $f _ { i } ^ { k }$ is an appearance embedding of the current detection result. In the following experiments, we set $\alpha { = } 0 . 9$ similar to the previous works [15, 45].

The Dynamic Appearance [34] is a confidence-based EMA, where the weight for a feature from each detection result is calculated based on the detection confidence scores. It allows the tracker to concentrate on the features of high-confidence detection. From eq (11), it sets the α to be a variable $\alpha _ { t } .$ , which depends on the confidence score $s _ { d e t }$ of the current detection result as,

$$
\alpha _ { t } = \alpha _ { f } + ( 1 - \alpha _ { f } ) ( 1 - \frac { s _ { d e t } - \sigma } { 1 - \sigma } ) ,\tag{12}
$$

where $\sigma$ is a threshold value. In the experiments, we set $\alpha _ { f } ~ = ~ 0 . 9 5$ following the previous method [34].

## 3.10 Summation

There have been many trials to achieve a well-balanced summation between the spatial distance matrix $D _ { s }$ and the appearance distance matrix $D _ { a }$ . The most common approach is a weighted sum using the optimal weight between two distances determined by a grid search as,

$$
D = \lambda D _ { a } + ( 1 - \lambda ) D _ { s } .\tag{13}
$$

Some trackers, such as ImprAsso [39], set the IoU threshold to this weighted sum to prevent wrong matches with objects showing a similar appearance and locating around a current tracking target as,

$$
D _ { i , j } = \left\{ \begin{array} { l l } { \lambda d _ { i , j } ^ { a } + ( 1 - \lambda ) d _ { i , j } ^ { s } } & { \mathrm { i f ~ I o U } > o _ { m i n } } \\ { d _ { m a x } + \epsilon } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{14}
$$

![](images/8af884fca88a1457613390fccc2b35e45054c1bfd21c71c38b2eb329ad596c9c.jpg)  
Fig. 4: Simplified structures of each association strategy. (a.1) is a basic structure of a one-stage association that uses all detections to match. (a.2) is an advanced one-stage association from BoostTrack [35] and (a.3) is another advanced one-stage association from Combined Matching [39]. (b) is a typical two-stage association structure. (c) is a four-stage association structure of LGTrack [47].

Geometric mean [46] and adaptive weighting [34], which depends on the discriminativeness of box and track pairs based on the first and second-highest distance, are also suggested as,

$$
D = { \sqrt { D _ { a } \cdot D _ { s } } } ,\tag{15}
$$

$$
\begin{array} { r } { D = D _ { s } + [ a _ { w } + w ] D _ { a } , } \end{array}\tag{16}
$$

respectively, where

$$
w _ { i , j } = [ z _ { \mathrm { d i f f } } ^ { \mathrm { t r a c k } } ( D _ { a } , i ) + z _ { \mathrm { d i f f } } ^ { \mathrm { d e t } } ( D _ { a } , j ) ] / 2 ,\tag{17}
$$

$$
z _ { \mathrm { d i f f } } ^ { \mathrm { d e t } } ( D _ { a } , j ) = \operatorname * { m i n } ( \operatorname * { m a x } _ { n } D _ { a } [ n , j ] - \operatorname * { m a x } _ { m \neq n } D _ { a } [ m , j ] , \epsilon ) ,\tag{18}
$$

$$
z _ { \mathrm { d i f f } } ^ { \mathrm { t r a c k } } ( D _ { a } , i ) = \operatorname* { m i n } ( \operatorname* { m a x } _ { n } D _ { a } [ i , n ] - \operatorname* { m a x } _ { m \neq n } D _ { a } [ i , m ] , \epsilon ) .\tag{19}
$$

Here, ǫ is a hyper-parameter to prevent divergence of $w _ { i , j }$ value and set to 0.5, and $\alpha _ { w }$ is set to 0.75, in our experiments following the original work [34].

Finally, BOT-SORT [15] chooses the minimum element among spatial distance and appearance distance as,

$$
D _ { i , j } = \operatorname* { m i n } \{ d _ { i , j } ^ { a } , d _ { i , j } ^ { s } \} .\tag{20}
$$

## 3.11 Association

Object association is a critical step in MOT, enabling the consistent matching of detected objects to existing tracks to maintain consistent identities across frames. Various association strategies, including one-stage, two-stage, and four-stage approaches, have been developed to balance the trade-of between eficiency, accuracy, and robustness in diverse tracking scenarios. The simplified structures of each association strategy are shown in Fig. 4. The one-stage association, proposed in SORT [12], matches all existing tracks with new detection results in a single step. Advanced one-stage methods, such as BoostTrack [35] and Combined Matching [39], enhance this process. BoostTrack filters out low-confidence detection results, prioritizing associations between high-confidence detection results and existing tracks. Combined Matching normalizes a cost matrix from low-confidence detection results by applying a scaling factor to align with that of high-confidence detection results, subsequently integrating both matrices to match all detections with tracks at once.

The two-stage association, presented in ByteTrack [14], BoT-SORT [15], and many other trackers [34, 41], refines matching through a sequential process. The first stage associates high-confidence detection results with tracks based on spatial proximity or IoU, while the second stage reprocesses unmatched detection results using appearance features or secondary metrics to improve robustness. The four-stage association [47] further extends this framework by systematically addressing distinct tracking challenges across multiple stages. In the first stage, detection results with high localization and classification confidence are matched with existing tracks. The second stage pairs detection results with high localization but low classification confidence with unmatched tracks from the first stage. The third stage associates detection results with low localization but high classification confidence with remaining tracks, and the final stage matches detection results with both low localization and low classification confidence to any remaining tracks, ensuring comprehensive handling of diverse detection scenarios.

## 3.12 Score Fusion

In the final phase of the two-stage association process, new tracks are matched with remaining high-confidence detection results to ensure comprehensive track assignment. Certain approaches, such as those in ByteTrack [14] and BoT-SORT [15], employ a score fusion technique that multiplies detection confidence scores to the IoU cost matrix to prioritize more discriminative detection results as,

$$
C = 1 - I o U ( D ^ { t } , B ^ { t } ) * s .\tag{21}
$$

## 3.13 Post-Processing

Interpolation is a widely adopted ofline post-processing technique in MOT that enhances trajectory continuity and recovers missing detection results caused by occlusions, detection failures, or irregular outputs. Linear Interpolation (LI) [14], a basic approach, assumes uniform motion to bridge gaps in trajectories. To incorporate richer motion dynamics, StrongSORT++ [23] introduces Gaussian-smoothed Interpolation (GSI) [23, 48], while Gradient Boosting Interpolation [35, 36] further refines trajectory alignment to follow ground-truth trajectory more properly. These interpolation methods significantly improve the reliability of MOT systems in post-processed data analysis. Additionally, the appearance-free link model (AFLink) [23] enhances track association by connecting track pairs using spatiotemporal features extracted from their most recent positions and a classifier predicting their association scores to ensure robust track continuity.

## 4 Experimental Setup

## 4.1 Object Detector

In this work, we use YOLOX [11] as an object detector, similar to the previous works [14, 15, 20, 41]. The detector model for each dataset is trained with a combination of the corresponding training set, CrowdHuman [49], and the WiderPerson [50] dataset with eight RTX 3090 GPUs while also following the training configuration of the previous work [14]. For example, the learning rate starts at 5e-4 and decays with the cosine scheduler, while the total number of training epochs, weight decay, SGD momentum, and batch size are set as 80, 5e-4, 0.9, and 32, respectively. Also, various data augmentation techniques, such as mosaic, random perspective transformation, and mixup, are involved, and the model is initialized with the COCO [51] pre-trained weight. During the inference stage, thresholds about IoU and confidence scores for non-maximum suppression are set as 0.8 and 0.1, respectively.

## 4.2 Feature Extractor

Following prior works [15, 20, 34], we adopt the SBS-S50 model from the FastReID framework [33] as the feature extractor, leveraging its robust performance for multiobject tracking. The model is fine-tuned on the target dataset using default training configurations initialized from Market-1501 [52] pre-trained weights. Specifically, the training is conducted for 60 epochs with a batch size of 64 and learning rate of 3.5e-4 while employing cross-entropy and triplet loss functions [53] and a cosine annealing learning rate scheduling. For feature extraction, we first crop patches according to the detection results, resize to 128 × 348 pixels, and apply L2 normalization after the extraction.

## 4.3 Datasets

Three prominent and distinctive MOT datasets, MOT17 [24], MOT20 [37], and Dance-Track [25], are adopted for every evaluation of our work. MOT17 [24] consists of seven training and seven testing sequences, which are filmed in unconstrained environments such as streets and inside malls with diverse camera motion. On the other hand, MOT20 [37] includes four videos for each training and test set with highly crowded scenes and heavy occlusion. Finally, DanceTrack [25] comprises 40, 25, and 35 sequences for training, validation, and testing, respectively. More details about the datasets are shown in Table 2. MOT17 and MOT20 are both characterized by linear object motion and distinctive object appearances. However, MOT20 presents problems with higher object density and overlap, and MOT17 ofers unstable camera motion. On the other hand, DanceTrack introduces distinctive challenges with its non-linear object motion and similar object appearances combined with complex camera movements. Since the datasets include varying conditions of camera motion, object density, motion patterns, and appearance similarity, each dataset is uniquely valuable for comprehensively assessing the MOT techniques. In addition, it can be a great guide to designing a generalized tracker for real-tracking scenarios by ensuring robust performance across all datasets.

![](images/4e7b19d51398cd0fb0402956603a32c31916490b48fcc5552b47d5e45db9160e.jpg)

![](images/c133ba8f4c4a115aa5344d6faf55a54095783f4fad3050df604ac6397fa65443.jpg)  
Fig. 5: Train and validation split of MOT17: (a) conventional splitting protocol and (b) ours. In the conventional train and validation split used for most previous methods, the first half of each video sequence in the training set is used as the training split, and the remaining halves are used as the validation split. In this scheme, many objects that appear in the training dataset also appear in the validation split, undermining the fundamental purpose of having separate training and validation splits. On the other hand, we split the train set by separating each video sequence, making our evaluation more generalizable.

Table 2: Characteristics of the three MOT datasets used in the evaluation. “# image”, “# box”, “mean # box”, and “mean # overlap” represent the number of training images, the total number of bounding boxes in the training set, the mean number of bounding boxes per training image, and the mean number of overlapping neighbor bounding boxes per each bounding box, respectively. “object appear.” denotes the appearances of the objects in the same frame.
<table><tr><td></td><td>MOT17</td><td>MOT20</td><td>DanceTrack</td></tr><tr><td># image</td><td>5316</td><td>8931</td><td>41796</td></tr><tr><td># box</td><td>112297</td><td>1134614</td><td>348930</td></tr><tr><td>mean # box</td><td>21.12</td><td>127.04</td><td>8.35</td></tr><tr><td>mean # overlap</td><td>1.81</td><td>4.26</td><td>1.81</td></tr><tr><td>camera motion</td><td>O</td><td>x</td><td>O</td></tr><tr><td>object motion</td><td>linear</td><td>linear</td><td>non-linear</td></tr><tr><td>mean movement</td><td>0.036</td><td>0.022</td><td>0.045</td></tr><tr><td>object appear.</td><td>distinctive</td><td>distinctive</td><td>similar</td></tr></table>

To assess each technique, we construct validation sets for the datasets. However, MOT17 and MOT20 do not provide oficial validation splits. In most previous works [14, 15, 23], a common practice is to divide each training video temporally, using the first half of the frames for training and the latter half for validation, as illustrated in Fig. 5.

While this protocol increases the amount of data available for both splits, it introduces substantial overlap in scene context and target objects between training and validation sets. To avoid this issue, we adopt a diferent splitting strategy: instead of dividing individual videos temporally, we separate the dataset at the sequence level and assign diferent videos to the training and validation sets. This prevents identity and scene overlap across the splits, enabling a more reliable evaluation of generalization. More specifically, the sequences “MOT17-02”, “MOT17-10”, “MOT17-11”, and “MOT17-13” are used for training, while the remaining sequences “MOT17-04”, “MOT17-05”, and $\mathrm { ^ { 4 6 } M O T 1 7 - 0 9 ^ { 3 } }$ are adopted for validation during our evaluation processes. Similarly, for the MOT20 dataset, the sequences “MOT20-02” and “MOT20-05” are used for training, the sequences $\mathrm { ^ { 6 6 } M O T 2 0 - 0 1 ^ { 5 } }$ and “MOT20-03” are used for validation.

## 4.4 Metrics

To evaluate various aspects of each technique, we use well-known MOT metrics, including Higher-Order Tracking Accuracy (HOTA) [54], Multi-Object Tracking Accuracy (MOTA) [55], IDF1[56], Detection Accuracy (DetA)[54], and Association Accuracy (AssA)[54]. In this work, we use HOTA as the primary metric because it jointly accounts for detection and association quality (and localization), providing a balanced view of overall tracking performance.

## 5 Experiments

We conducted a comprehensive evaluation of MOT techniques from Tracking-by-Detection (TBD) approaches, using the baseline tracker detailed in Section 2.2 as the starting point. We incrementally integrated various techniques, including the Kalman filter, parameter optimization, camera motion compensation, spatial distance metrics, appearance features, summation strategies, association methods, and post-processing techniques. For each sub-task, only the techniques that increase tracking performance in all three datasets are added to the baseline to construct a new baseline for subsequent sub-tasks, ensuring a systematic assessment of the contribution of each component to overall tracking performance.

## 5.1 Kalman State Vector

Table 3 evaluates the impact of diferent Kalman state vector representations. The state vectors analyzed include (cx, cy, s, a) [12], $( c x , c y , a , h )$ [13], and $( c x , c y , w , h )$ [15]. The state vector (cx, cy, w, h) achieves the best performance for all metrics across all datasets. These results indicate that incorporating width and height provides the most robust tracking performance, as this representation efectively captures both spatial localization and object dimensions, which are crucial for accurate association. The state vectors $( c x , c y , s , a )$ and $( c x , c y , a , h )$ perform slightly worse. Especially on DanceTrack, which features similar object appearances and complex motions, the

Table 3: Comparison of diferent Kalman state vectors. The first row is the previous baseline, and the selected option, which will serve as the new baseline tracker for subsequent evaluation, is denoted by a gray row. The increased HOTA scores compared to the previous baseline are underlined, and the best scores are marked in bold.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>state vector</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>(cx, cy, s, a)</td><td>63.91</td><td>78.66</td><td>77.85</td><td>65.21</td><td>63.17</td><td>66.40</td><td>84.97</td><td>86.20</td><td>66.80</td><td>66.20</td><td>53.26</td><td>90.86</td><td>52.55</td><td>79.70</td><td>35.76</td></tr><tr><td>(cx, cy, a, h)</td><td>64.33</td><td>79.10</td><td>78.91</td><td>65.25</td><td>63.93</td><td>66.38</td><td>84.92</td><td>87.55</td><td>66.26</td><td>66.66</td><td>47.95</td><td>89.24</td><td>51.70</td><td>71.69</td><td>32.23</td></tr><tr><td>(cx, cy, w, h)</td><td>65.07</td><td>79.59</td><td>79.79</td><td>65.32</td><td>65.32</td><td>67.33</td><td>85.20</td><td>87.91</td><td>67.03</td><td>67.82</td><td>55.71</td><td>91.47</td><td>56.36</td><td>79.90</td><td>39.00</td></tr></table>

Table 4: Comparison of diferent track initialization strategies. Applying OAI increases HOTA across all three datasets.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>OAI</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>65.07</td><td>79.59</td><td>79.79</td><td>65.32</td><td>65.32</td><td>67.33</td><td>85.20</td><td>87.91</td><td>67.03</td><td>67.82</td><td>55.71</td><td>91.47</td><td>56.36</td><td>79.90</td><td>39.00</td></tr><tr><td>√</td><td>67.00</td><td>80.24</td><td>83.23</td><td>65.90</td><td>68.60</td><td>67.77</td><td>85.70</td><td>88.67</td><td>67.30</td><td>68.43</td><td>57.56</td><td>91.67</td><td>60.04</td><td>79.73</td><td>41.71</td></tr></table>

Kalman state with $( c x , c y , w , h )$ outperforms the others, demonstrating its adaptability to challenging scenarios. This result emphasizes the importance of precise dimensional representation in dynamic environments. The other representations show lower scores in HOTA, emphasizing the importance of precise dimensional representation in dynamic environments. In summary, the results highlight that the $( c x , c y , w , h )$ state vector ofers the best performance in terms of HOTA across all datasets, making it the most efective configuration for accurate and robust tracking. Therefore, the new baseline tracker for the remaining experiments follows the Kalman state vector of $( c x , c y , w , h )$

## 5.2 Track Initialization

Table 4 examines the influence of Occlusion-aware Initialization (OAI) [39] on tracking performance across the datasets. The results indicate that enabling OAI improves most metrics across all datasets. This highlights the benefits of OAI in dense and complex tracking scenarios, where accurate initialization can prevent identity switches. On the DanceTrack dataset, which presents frequent occlusions, OAI provides the most significant improvement with HOTA increasing from 55.71 to 57.56. This result suggests that OAI is efective in challenging environments by facilitating robust track initialization. Overall, the use of OAI results in consistent gains across most metrics and datasets, with notable improvements. These findings underline the importance of spatial information during track initialization for tracking performance, and OAI is adopted in the following experiments.

## 5.3 Kalman Filter Prediction/Update Strategy

Table 5 provides an analysis of tracking performance based on the inclusion of constant Kalman state vectors [28, 40], Confidence Weighted Kalman Update (CWKU) [40], and Observation-centric Re-Update (ORU) [41]. The use of keeping height [28] or both width and height [40] while updating inactive tracks does not produce a significant performance improvement compared to the baseline that updates all states. Although it shows a positive result in DanceTrack, fixing some states delivers slightly lower HOTA and MOTA scores compared to the baseline in MOT17. When CWKU is applied, performance generally improves across most metrics while setting the confidence score threshold to 0.6 or 0.7 following the original work [40]. When both CWKU and ORU are applied, the performance shows a mixed trend. For example, the HOTA score increases in MOT17 and MOT20 compared to the tracker only with CKWU. However, the performance on DanceTrack decreases slightly. Overall, the results suggest that fixing the Kalman state vector of inactive tracks does not ofer positive efects on the results. In contrast, CWKU enhances tracking performance, while ORU may introduce marginal trade-ofs in some cases. In the following experiments, we apply CKWU with a confidence score threshold between 0.6 and 0.7 and choose the best result.

Table 5: Comparison of diferent Kalman filter prediction/update strategies. Trackers with CWKU show improvements across all datasets.
<table><tr><td></td><td></td><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Constant</td><td>CWKU</td><td>ORU</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>67.00</td><td>80.24</td><td>83.23</td><td>65.90</td><td>68.60</td><td>67.77</td><td>85.70</td><td>88.67</td><td>67.30</td><td>68.43</td><td>57.56</td><td>91.67</td><td>60.04</td><td>79.73</td><td>41.71</td></tr><tr><td>h</td><td>x</td><td>x</td><td>66.95</td><td>79.86</td><td>82.69</td><td>65.92</td><td>68.49</td><td>67.75</td><td>85.68</td><td>88.63</td><td>67.30</td><td>68.40</td><td>58.12</td><td>91.71</td><td>60.53</td><td>79.75</td><td>42.52</td></tr><tr><td>w, h</td><td>x</td><td>x</td><td>66.99</td><td>79.87</td><td>83.38</td><td>65.80</td><td>68.69</td><td>67.77</td><td>85.71</td><td>88.67</td><td>67.27</td><td>68.45</td><td>57.81</td><td>91.65</td><td>60.10</td><td>79.91</td><td>41.96</td></tr><tr><td>x</td><td>✓ (0.6)</td><td>x</td><td>68.09</td><td>80.08</td><td>85.15</td><td>66.15</td><td>70.53</td><td>67.88</td><td>85.70</td><td>88.99</td><td>67.36</td><td>68.60</td><td>57.88</td><td>92.05</td><td>58.88</td><td>79.88</td><td>42.11</td></tr><tr><td>x</td><td>√(0.7)</td><td>x</td><td>68.08</td><td>80.04</td><td>85.13</td><td>66.14</td><td>70.52</td><td>68.03</td><td>85.75</td><td>89.12</td><td>67.41</td><td>68.85</td><td>57.92</td><td>92.03</td><td>59.05</td><td>80.01</td><td>42.12</td></tr><tr><td>x</td><td>√ (0.6)</td><td>√</td><td>68.19</td><td>80.07</td><td>85.34</td><td>66.38</td><td>70.51</td><td>67.99</td><td>85.77</td><td>89.10</td><td>67.43</td><td>68.75</td><td>57.47</td><td>91.53</td><td>59.53</td><td>79.48</td><td>41.71</td></tr><tr><td>x</td><td>√ (0.7)</td><td>√</td><td>68.11</td><td>80.00</td><td>85.20</td><td>66.26</td><td>70.47</td><td>67.98</td><td>85.75</td><td>89.00</td><td>67.41</td><td>68.74</td><td>57.17</td><td>92.07</td><td>58.17</td><td>79.99</td><td>41.03</td></tr></table>

Table 6: Comparison of diferent basic parameter settings. The selected option, which will serve as the new baseline tracker for subsequent evaluation, is denoted by a gray row.
<table><tr><td colspan="3"></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>NMS thresh.</td><td>act</td><td>lost max</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>0.8</td><td>3</td><td>frame rate</td><td>68.09</td><td>80.08</td><td>85.15</td><td>66.15</td><td>70.53</td><td>68.03</td><td>85.75</td><td>89.12</td><td>67.41</td><td>68.85</td><td>57.92</td><td>92.03</td><td>59.05</td><td>80.01</td><td>42.12</td></tr><tr><td>0.7</td><td>3</td><td>frame rate</td><td>68.07</td><td>79.56</td><td>85.32</td><td>66.10</td><td>70.57</td><td>67.89</td><td>85.69</td><td>88.86</td><td>67.39</td><td>68.59</td><td>57.63</td><td>91.42</td><td>59.74</td><td>79.32</td><td>42.02</td></tr><tr><td>0.8</td><td>2</td><td>frame rate</td><td>68.26</td><td>80.40</td><td>85.25</td><td>66.41</td><td>70.59</td><td>68.15</td><td>85.91</td><td>89.20</td><td>67.54</td><td>68.96</td><td>57.97</td><td>92.20</td><td>59.04</td><td>80.15</td><td>42.11</td></tr><tr><td>0.8</td><td>2</td><td>2×frame rate</td><td>68.20</td><td>80.37</td><td>85.16</td><td>66.40</td><td>70.50</td><td>68.17</td><td>85.94</td><td>89.31</td><td>67.55</td><td>69.00</td><td>57.80</td><td>92.23</td><td>58.79</td><td>80.05</td><td>41.91</td></tr><tr><td>0.8</td><td>2</td><td>30</td><td>68.19</td><td>80.37</td><td>85.12</td><td>66.40</td><td>70.47</td><td>68.17</td><td>85.90</td><td>89.30</td><td>67.54</td><td>69.00</td><td>57.76</td><td>92.22</td><td>58.70</td><td>80.03</td><td>41.87</td></tr></table>

## 5.4 Basic Parameters

Table 6 presents an analysis of the impact of various parameters, specifically the NMS threshold, track activation frame count (act), and the maximum allowable frames for a lost object (lost max). Across all datasets, an NMS threshold of 0.8 consistently yields higher performance than the threshold of 0.7, indicating that a higher NMS threshold generally leads to better tracking results by more efectively suppressing redundant detections. With an activation frame count of two frames, all aspects of performance improve compared to the configuration with an activation frame count of three, indicating that the tracker can catch new tracks more efectively by setting the activation frame count to two. When adjusting the lost max parameter, it is evident that setting this value to twice the frame rate or 30 frames has minimal impact on performance.

Table 7: Comparison of diferent CMC methods. Both GMC and ECC bring improvements across all datasets, and GMC makes a bigger gap from the previous baseline tracker.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>CMC</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>68.26</td><td>80.40</td><td>85.25</td><td>66.41</td><td>70.59</td><td>68.15</td><td>85.91</td><td>89.20</td><td>67.54</td><td>68.96</td><td>57.97</td><td>92.20</td><td>59.04</td><td>80.15</td><td>42.11</td></tr><tr><td>GMC</td><td>68.61</td><td>80.89</td><td>85.91</td><td>66.51</td><td>71.21</td><td>68.26</td><td>85.86</td><td>89.39</td><td>67.49</td><td>69.22</td><td>58.23</td><td>92.21</td><td>59.59</td><td>80.00</td><td>42.57</td></tr><tr><td>ECC</td><td>68.35</td><td>80.86</td><td>85.79</td><td>66.37</td><td>70.83</td><td>68.21</td><td>86.01</td><td>89.36</td><td>67.59</td><td>69.02</td><td>58.27</td><td>92.27</td><td>59.98</td><td>80.09</td><td>42.57</td></tr></table>

For example, in MOT17, the HOTA scores are slightly lower than when we set the lost max parameter as the frame rate of each video. However, longer values overperform in MOT20, indicating longer matching windows are especially suitable for crowded scenes but not for all general scenes. In summary, a higher NMS threshold of 0.8 and reducing the track activation frame count to two frames improves overall tracking performance, and increasing the lost max parameter does not lead to substantial changes in performance. Therefore, we set the NMS threshold as 0.8, the activation count as two frames, and the re-tracking windows as the frame rate length of each video in the new baseline tracker for the following experiments.

## 5.5 Camera Motion Compensation

Table 7 shows the impact of diverse camera motion compensation techniques. The three configurations compared are without CMC, with Global Motion Compensation (GMC) [30], and with Enhanced Correlation Coeficient Maximization (ECC) [32]. The results indicate that applying CMC, whether via GMC or ECC, leads to improvements in tracking performance across all datasets, indicating that compensating for camera motion leads to more stable and accurate tracking. GMC enhances performance the most in MOT17 and MOT20 with minor improvement in DanceTrack, indicating that compensating for camera motion leads to more stable and accurate tracking. Although gains are lower, ECC improves HOTA in MOT17 and MOT20, showing clear benefits in detection and identity preservation. In DanceTrack, ECC slightly outperforms GMC, achieving HOTA of 58.27, compared to 58.23 with GMC.

Overall, the inclusion of CMC, particularly GMC, consistently improves tracking metrics, demonstrating the importance of compensating for camera movement to enhance identity consistency and association accuracy in various tracking scenarios. In the following evaluation, we include GMC in the basic configuration of the baseline tracker.

## 5.6 Main Spatial Distance

The evaluation of spatial distance metrics, which are IoU, GIoU [42], DIoU [43], CIoU [43], BIoU [38] (with bufer scales of 0.1, 0.2, and 0.3), and HMIoU [20], is presented in Table 8. The result reveals IoU as the most robust and efective metric. More specifically, IoU consistently outperforms alternatives, achieving strong HOTA scores of 68.61, 68.26, and 58.23 in MOT17, MOT20, and DanceTrack, respectively. It suggests that IoU is efective in handling object overlaps and maintaining stable identity tracking across various scenarios. While GIoU, DIoU, and CIoU ofer comparable results or marginal improvements in MOT17, they generally underperform in crowded or dynamic settings of MOT20 and DanceTrack. These results suggest that the additional penalty of GIOU for bounding box misalignment beyond the overlap region does not significantly enhance performance. Also, although DIoU and CIoU improve certain spatial constraints, such as box distance, they may not significantly enhance association accuracy in challenging scenarios. BIoU shows competitive performance at lower bufer scales of 0.1 and 0.2 but degrades at higher scales of 0.3. Finally, HMIoU excels in DanceTrack by leveraging height information, yet remains less efective overall. In conclusion, the simplicity and balanced performance of IoU make it the preferred and most eficient choice for general applications, and we also adopt IoU in the following evaluations.

Table 8: Comparison of matching with diferent spatial distances. IoU makes the most balanced performance across all datasets.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>distance</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>IoU</td><td>68.61</td><td>80.89</td><td>85.91</td><td>66.51</td><td>71.21</td><td>68.26</td><td>85.86</td><td>89.39</td><td>67.49</td><td>69.22</td><td>58.23</td><td>92.21</td><td>59.59</td><td>80.00</td><td>42.57</td></tr><tr><td>Mahalanobis</td><td>63.84</td><td>78.72</td><td>78.67</td><td>65.82</td><td>62.44</td><td>66.27</td><td>84.10</td><td>86.34</td><td>66.70</td><td>66.05</td><td>31.77</td><td>80.21</td><td>26.29</td><td>77.46</td><td>13.08</td></tr><tr><td>GIoU</td><td>67.78</td><td>80.57</td><td>85.04</td><td>66.54</td><td>69.51</td><td>68.19</td><td>85.83</td><td>89.25</td><td>67.46</td><td>69.13</td><td>57.51</td><td>91.81</td><td>59.41</td><td>79.84</td><td>41.58</td></tr><tr><td>DIoU</td><td>68.64</td><td>80.97</td><td>85.89</td><td>66.68</td><td>71.11</td><td>68.13</td><td>85.59</td><td>89.17</td><td>67.32</td><td>69.14</td><td>57.23</td><td>92.25</td><td>58.70</td><td>80.16</td><td>41.03</td></tr><tr><td>CIoU</td><td>68.64</td><td>80.97</td><td>85.89</td><td>66.68</td><td>71.11</td><td>68.13</td><td>85.59</td><td>89.17</td><td>67.32</td><td>69.14</td><td>57.24</td><td>92.25</td><td>58.70</td><td>80.14</td><td>41.06</td></tr><tr><td>BIoU(0.1)</td><td>68.24</td><td>80.19</td><td>85.10</td><td>66.15</td><td>70.85</td><td>68.20</td><td>86.01</td><td>89.39</td><td>67.57</td><td>69.02</td><td>57.94</td><td>92.26</td><td>59.52</td><td>80.44</td><td>41.90</td></tr><tr><td>BIoU(0.2)</td><td>68.36</td><td>80.16</td><td>85.47</td><td>66.15</td><td>71.11</td><td>68.16</td><td>85.93</td><td>89.32</td><td>67.48</td><td>69.04</td><td>57.92</td><td>92.05</td><td>59.97</td><td>79.89</td><td>42.16</td></tr><tr><td>BIoU(0.3)</td><td>67.39</td><td>80.43</td><td>83.85</td><td>66.36</td><td>68.92</td><td>67.97</td><td>85.83</td><td>89.04</td><td>67.39</td><td>68.74</td><td>57.96</td><td>91.94</td><td>60.21</td><td>79.42</td><td>42.48</td></tr><tr><td>HMIoU</td><td>67.61</td><td>80.83</td><td>84.54</td><td>66.76</td><td>68.97</td><td>68.14</td><td>85.66</td><td>89.01</td><td>67.44</td><td>69.03</td><td>58.76</td><td>91.79</td><td>59.17</td><td>80.69</td><td>42.97</td></tr></table>

Table 9: Comparison of matching with diferent additional spatial distances. ROCM, conf., SMB, and Mahal. denote robust OCM, confidence cost, similarity matrix boost, and Mahalanobis, respectively. The numbers inside the brackets beside the SMB techniques represent the weight values when adding each metric to the base cost matrix. Similarity matrix boost with detection-tracklet confidence similarity draws the most balanced performance.
<table><tr><td colspan="5"></td><td colspan="4">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>OCM</td><td>ROCM</td><td>conf.</td><td>SMB</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>x</td><td>68.61</td><td>80.89</td><td>85.91</td><td>66.51</td><td>71.21</td><td>68.26</td><td>85.86</td><td>89.39</td><td>67.49</td><td>69.22</td><td>58.23</td><td>92.21</td><td>59.59</td><td>80.00</td><td>42.57</td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>68.55</td><td>80.38</td><td>85.63</td><td>66.38</td><td>71.23</td><td>68.12</td><td>85.94</td><td>89.16</td><td>67.53</td><td>68.90</td><td>58.00</td><td>92.22</td><td>59.79</td><td>80.19</td><td>42.11</td></tr><tr><td>x</td><td>√</td><td>x</td><td>x</td><td>68.57</td><td>80.54</td><td>85.75</td><td>66.56</td><td>71.09</td><td>68.17</td><td>85.87</td><td>89.26</td><td>67.47</td><td>69.07</td><td>58.11</td><td>92.23</td><td>59.59</td><td>80.10</td><td>42.32</td></tr><tr><td>x</td><td>x</td><td>√</td><td>x</td><td>68.49</td><td>80.94</td><td>86.01</td><td>66.43</td><td>71.06</td><td>68.20</td><td>85.97</td><td>89.29</td><td>67.54</td><td>69.05</td><td>58.40</td><td>92.35</td><td>59.70</td><td>80.23</td><td>42.67</td></tr><tr><td>x</td><td>√</td><td>√</td><td>x</td><td>68.68</td><td>80.64</td><td>86.25</td><td>66.55</td><td>71.34</td><td>68.20</td><td>85.98</td><td>89.26</td><td>67.53</td><td>69.06</td><td>58.25</td><td>92.32</td><td>59.81</td><td>80.09</td><td>42.54</td></tr><tr><td>x</td><td>x</td><td>x</td><td>conf. (0.75)</td><td>68.62</td><td>80.93</td><td>86.11</td><td>66.63</td><td>71.08</td><td>68.26</td><td>85.94</td><td>89.46</td><td>67.51</td><td>69.21</td><td>60.17</td><td>92.07</td><td>62.26</td><td>80.23</td><td>45.28</td></tr><tr><td>x</td><td>x</td><td>x</td><td>shape (0.75)</td><td>68.76</td><td>80.93</td><td>86.69</td><td>66.52</td><td>71.52</td><td>68.19</td><td>85.94</td><td>89.37</td><td>67.51</td><td>69.06</td><td>60.32</td><td>92.00</td><td>62.20</td><td>80.22</td><td>45.52</td></tr><tr><td>x</td><td>x</td><td>x</td><td>Mahal. (0.25)</td><td>68.51</td><td>80.84</td><td>85.58</td><td>66.35</td><td>71.19</td><td>68.17</td><td>85.64</td><td>89.27</td><td>67.36</td><td>69.17</td><td>57.44</td><td>92.10</td><td>58.30</td><td>80.22</td><td>41.30</td></tr><tr><td>x</td><td>x</td><td>x</td><td>conf. (0.50) + shape (0.25)</td><td>68.79</td><td>81.01</td><td>86.75</td><td>66.57</td><td>71.53</td><td>68.24</td><td>85.94</td><td>89.40</td><td>67.51</td><td>69.16</td><td>60.37</td><td>92.08</td><td>62.71</td><td>80.26</td><td>45.57</td></tr><tr><td>x</td><td>x</td><td>x</td><td>conf. (0.50) + Mahal. (0.25)</td><td>68.66</td><td>80.81</td><td>85.94</td><td>66.46</td><td>71.38</td><td>68.11</td><td>85.90</td><td>89.32</td><td>67.49</td><td>68.92</td><td>58.49</td><td>92.31</td><td>59.12</td><td>80.52</td><td>42.65</td></tr><tr><td>x</td><td>x</td><td>x</td><td>shape (0.50) + Mahal. (0.25)</td><td>68.07</td><td>80.65</td><td>84.64</td><td>66.38</td><td>70.27</td><td>68.15</td><td>85.91</td><td>89.44</td><td>67.48</td><td>69.01</td><td>58.69</td><td>91.76</td><td>59.72</td><td>80.05</td><td>43.18</td></tr><tr><td>x</td><td>x</td><td>x</td><td>conf. + shape + Mahal.</td><td>68.53</td><td>80.86</td><td>85.92</td><td>66.47</td><td>71.11</td><td>68.14</td><td>85.89</td><td>89.44</td><td>67.49</td><td>68.98</td><td>59.79</td><td>92.25</td><td>61.73</td><td>80.22</td><td>44.74</td></tr></table>

## 5.7 Additional Spatial Distance

Table 9 presents the analysis of spatial distance techniques, including Object-Centric Momentum (OCM) [41], robust OCM (ROCM) [20], confidence cost (conf.)[20], and various forms of similarity matrix boosts (confidence, shape, and Mahalanobis distance)[35]. For the first three methods, OCM, ROCM, and confidence cost, the baseline tracker without these techniques achieves strong HOTA scores in MOT17, MOT20, and DanceTrack. The baseline tracker without these techniques achieves strong HOTA scores of 68.61, 68.26, and 52.83 in MOT17, MOT20, and Dance- Track, respectively. Introducing OCM slightly reduces HOTA in MOT17 and MOT20, suggesting that basic OCM may not significantly improve the tracking process. When robust OCM is applied with velocity-based adjustments, performance is similar to the original OCM. The application of the confidence cost degrades performance in MOT17 and MOT20 compared to the baseline, although it improves the HOTA score in DanceTrack.

Table 10: Comparison of matching with diferent tricks of spatial distance. The prior baseline without any tricks of spatial distance shows the best performance across all datasets.
<table><tr><td colspan="4"></td><td colspan="4">MOT17</td><td colspan="4"></td><td colspan="4"></td><td colspan="4">DanceTrack</td></tr><tr><td>DLO</td><td>DUO</td><td>CF</td><td>PIA</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td></td><td>DetA↑ AssA↑</td><td></td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>x</td><td>68.62</td><td>80.93</td><td>86.11</td><td>66.63</td><td>71.08</td><td>68.26</td><td>85.94</td><td>89.46</td><td></td><td>67.51</td><td>69.21</td><td>60.17</td><td>92.07</td><td>62.26</td><td>80.23</td><td>45.28</td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>68.53</td><td>80.94</td><td>86.17</td><td>66.45</td><td>71.14</td><td>68.23</td><td>85.98</td><td>89.49</td><td></td><td>67.54 69.12</td><td></td><td>59.56</td><td>92.03</td><td>62.02</td><td>80.19</td><td>44.41</td></tr><tr><td>x</td><td>√</td><td>x</td><td>x</td><td>68.50</td><td>80.89</td><td>86.14</td><td>66.45</td><td>71.08</td><td>68.21</td><td>85.81</td><td>89.41</td><td></td><td>67.44 69.17</td><td></td><td>59.84</td><td>92.47</td><td>61.75</td><td>80.34</td><td>44.74</td></tr><tr><td>√</td><td>√</td><td>x</td><td>x</td><td>68.37</td><td>80.91</td><td>85.88</td><td>66.29</td><td>70.97</td><td>68.19</td><td>85.81</td><td>89.42</td><td>67.44</td><td>69.12</td><td></td><td>59.43</td><td>92.02</td><td>61.29</td><td>80.09</td><td>44.27</td></tr><tr><td>x</td><td>x</td><td>√</td><td>x</td><td>66.88</td><td>79.45</td><td>83.81</td><td>65.57</td><td>68.67</td><td>67.27</td><td>84.92</td><td>88.03</td><td>66.81</td><td>67.91</td><td></td><td>60.03</td><td>91.58</td><td>61.30</td><td>80.10</td><td>45.14</td></tr><tr><td>x</td><td>x</td><td>x</td><td>√</td><td>68.09</td><td>80.25</td><td>84.98</td><td>66.08</td><td>70.62</td><td>68.15</td><td>85.92</td><td>89.05</td><td>67.53</td><td>68.96</td><td></td><td>59.12</td><td>91.70</td><td>59.91</td><td>80.19</td><td>43.76</td></tr></table>

In the case of the similarity matrix boosts, particularly confidence and shape boosts, deliver the most substantial performance improvements. A confidence boost with a weight of 0.75 elevates HOTA to 68.62 in MOT17 and 60.17 in DanceTrack, with the other metrics. A shape boost also improves HOTA on MOT17 and Dance-Track, but slightly degrades performance on MOT20. Similarly, combining confidence and shape boosts with weights of 0.50 and 0.25 further enhances performance, achieving higher HOTA in MOT17 and DanceTrack. However, performance in MOT20 shows slight declines with these boosts, suggesting sensitivities in crowded scenarios. The combined boost of confidence, shape, and Mahalanobis distance yields marginal gains over individual boosts. The results demonstrate that the most significant performance gains are from applying the confidence similarity matrix boost. Applying this boost technique enhances tracking accuracy, making it especially efective in dynamic scenarios in DanceTrack. Therefore, we include the confidence similarity matrix boost in the new baseline for the succeeding experiments.

## 5.8 Tricks of Spatial Distance

Table 10 evaluates the impact of spatial distance enhancement tricks, which are Detecting Likely Objects (DLO) [35], Detecting Unlikely Objects (DUO) [35], Confidence Fused Cost Matrix (CF) [40], and Past Information Aggregation (PIA) [44], with the baseline configuration without any tricks. While the baseline achieves HOTA scores of 68.62, 68.26, and 60.17, respectively, applying DLO and DUO alone yields a slight decrease in the HOTA scores. Combining DLO and DUO even more reduces performance, with HOTA dropping to 68.37 in MOT17 and 59.43 in DanceTrack, suggesting potential conflicts in their adjustments. In addition, CF significantly and consistently underperforms compared to the baseline across all datasets, likely due to over-filtering valid detection result. PIA similarly shows performance degradation with HOTA scores across all benchmarks. It suggests that linear assignment with an aggregated cost matrix may result in more mismatches between tracks and detections. In summary, these results provide the benefit of few individual DLOs and DUOs, while CFs and PIA degrade tracking accuracy. Therefore, we maintain the configuration of the current baseline for the following analysis.

Table 11: Comparison of matching with diferent appearance feature update strategies. Even though Feature Bank with a large number of features n shows great tracking quality, it takes a much longer time to inference so it’s hard to employ for real-time tracking. EMA shows the most balanced performance across all datasets.
<table><tr><td></td><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Feature Update</td><td>n</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>Feature Bank</td><td>100</td><td>67.38</td><td>80.49</td><td>84.19</td><td>65.63</td><td>69.68</td><td>65.48</td><td>85.01</td><td>84.24</td><td>66.94</td><td>64.27</td><td>34.10</td><td>69.90</td><td>26.36</td><td>72.13</td><td>16.22</td></tr><tr><td>Feature Bank</td><td>50</td><td>67.74</td><td>80.39</td><td>84.69</td><td>65.77</td><td>70.27</td><td>65.84</td><td>85.23</td><td>84.48</td><td>67.19</td><td>64.74</td><td>34.98</td><td>71.09</td><td>27.41</td><td>71.79</td><td>17.13</td></tr><tr><td>Feature Bank</td><td>10</td><td>67.42</td><td>80.14</td><td>84.26</td><td>65.57</td><td>69.81</td><td>66.39</td><td>85.18</td><td>85.58</td><td>67.11</td><td>65.89</td><td>31.41</td><td>72.30</td><td>23.52</td><td>73.23</td><td>13.56</td></tr><tr><td>EMA</td><td>=</td><td>68.53</td><td>79.89</td><td>85.47</td><td>66.30</td><td>71.29</td><td>64.96</td><td>84.42</td><td>82.99</td><td>66.92</td><td>63.30</td><td>31.59</td><td>61.51</td><td>24.19</td><td>68.58</td><td>14.62</td></tr><tr><td>DA</td><td></td><td>65.64</td><td>79.07</td><td>80.43</td><td>65.94</td><td>65.85</td><td>57.35</td><td>80.84</td><td>70.38</td><td>65.92</td><td>50.22</td><td>26.13</td><td>53.03</td><td>18.90</td><td>66.86</td><td>10.28</td></tr></table>

## 5.9 Appearance Feature Update

Table 11 evaluates the impact of appearance feature update strategies, notably static Feature Bank [13] with 10, 50, and 100 features, Exponential Moving Average (EMA) [45], and Dynamic Appearance (DA) [34], using the ResNeSt[57]-based SBS-S50 feature extractor from the FastReID framework [33]. In this experiment, the cost matrices are only based on cosine similarity between detection and track feature vectors for more transparent examination. The static Feature Bank strategy finds that the feature banks with more stored features achieve better performance on complicated scenarios, such as DanceTrack. For example, a feature bank with 50 features achieves HOTA of 34.98 in DanceTrack. However, its computational ineficiency with more than 10 features, even processing fewer frames than the frame rate in our empirical evaluations, renders it unsuitable for online MOT.

Among online-MOT-available methods, EMA delivers the best balanced performance. In MOT17, it achieves the best HOTA with 68.53, demonstrating superior tracking stability and identity preservation through smoothed feature updates. Although its performance in MOT20 and DanceTrack indicates limitations in crowded and dynamic scenarios compared to the baseline, it’s slightly better than the feature bank with 10 features in DanceTrack. Finally, DA consistently underperforms, with HOTA scores in all datasets, likely due to noise introduced by frequent updates incorporating detection confidence. Thus, EMA emerges as the most efective strategy, balancing robustness and eficiency, while the static Feature Bank requires strong computational overhead, and DA struggles with instability across diverse tracking conditions.

## 5.10 Summation

Table 12 shows the impact of diferent distance summation strategies, including only spatial distance, only appearance distance, weighted sum, threshold-weighted sum [39], geometric mean [46], adaptive weighting [34], and minimum element among spatial distance and appearance distance[15]. Using only spatial distance yields robust performance with HOTA in all benchmarks, while relying solely on appearance distance significantly underperforms, particularly in DanceTrack. This highlights the limitations of relying only on appearance features in dynamic scenes, where spatial positioning provides robust tracking. From this perspective, combining two features is expected to result in better performance.

Table 12: Comparison of matching with diferent summation strategies. The weighted sum of the spatial distance and the appearance distance makes the most balanced improvement.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Summation</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>only spatial dist.</td><td>68.62</td><td>80.91</td><td>86.19</td><td>66.62</td><td>71.12</td><td>68.26</td><td>85.94</td><td>89.46</td><td>67.51</td><td>69.21</td><td>60.17</td><td>92.07</td><td>62.26</td><td>80.23</td><td>45.28</td></tr><tr><td>only appearance dist.</td><td>68.53</td><td>79.89</td><td>85.47</td><td>66.30</td><td>71.29</td><td>64.95</td><td>84.36</td><td>83.22</td><td>66.81</td><td>63.37</td><td>31.59</td><td>61.51</td><td>24.19</td><td>68.58</td><td>14.62</td></tr><tr><td>weighted sum</td><td>68.94</td><td>80.66</td><td>86.24</td><td>66.42</td><td>72.01</td><td>68.42</td><td>85.81</td><td>89.57</td><td>67.48</td><td>69.56</td><td>62.85</td><td>92.09</td><td>65.94</td><td>80.34</td><td>49.35</td></tr><tr><td>thresh. weighted sum</td><td>68.94</td><td>80.65</td><td>86.24</td><td>66.42</td><td>72.01</td><td>68.43</td><td>85.88</td><td>89.59</td><td>67.54</td><td>69.51</td><td>62.52</td><td>92.08</td><td>65.34</td><td>80.17</td><td>48.95</td></tr><tr><td>geometric mean</td><td>68.84</td><td>80.69</td><td>86.06</td><td>66.57</td><td>71.66</td><td>68.46</td><td>85.96</td><td>89.63</td><td>67.58</td><td>69.53</td><td>62.29</td><td>91.90</td><td>65.39</td><td>79.86</td><td>48.77</td></tr><tr><td>adaptive weighting</td><td>68.67</td><td>80.81</td><td>85.50</td><td>66.78</td><td>71.07</td><td>68.25</td><td>85.92</td><td>89.26</td><td>67.56</td><td>69.14</td><td>61.44</td><td>92.03</td><td>63.08</td><td>80.68</td><td>46.96</td></tr><tr><td>minimum</td><td>68.84</td><td>80.94</td><td>86.33</td><td>66.41</td><td>71.83</td><td>68.36</td><td>85.69</td><td>89.44</td><td>67.40</td><td>69.53</td><td>58.15</td><td>91.38</td><td>60.08</td><td>79.74</td><td>42.56</td></tr></table>

The weighted sum of spatial distance and appearance distance approaches balance spatial and appearance information, resulting in the highest HOTA scores across all configurations. The weighted sum achieves HOTA of 68.94 in MOT17, 68.42 in MOT20, and 62.85 in DanceTrack. This indicates that integrating spatial and appearance information through weighted calculations provides an optimal tracking performance. The weighted sum method, with an optimized weight λ = 0.7 determined via grid search. The geometric mean is slightly below the weighted sum, but shows a noticeable improvement in performance over the baseline. The threshold-weighted sum configuration and geometric mean perform comparably to the weighted sum, but this approach does not significantly outperform the standard weighted sum in DanceTrack. The adaptive weighting method shows lower scores in all datasets. The minimumbased approach achieves slightly lower scores than the previous baseline, particularly in DanceTrack, indicating that taking the minimum distance may overly limit the amount of information in complex environments. Overall, the weighted sum method achieves the best balance of spatial and appearance cues, consistently yielding high HOTA scores across all datasets. These results suggest that combining spatial and appearance information with balanced weighting is crucial, and we choose the weight summation as our combining method between spatial and appearance distances for the following evaluations.

## 5.11 Association

The evaluation of various association strategies, as presented in Table 13, demonstrates that the 2-stage association strategy[14] achieves the highest HOTA scores across all datasets, with 69.35 in MOT17, 68.58 in MOT20, and 63.27 in DanceTrack. These results highlight its superior balance of detection and association accuracy by prioritizing high-confidence detection results in the first stage, making it the most efective strategy overall. The 1-stage association process[12], which is computationally simpler, yields slightly lower performance, indicating that the additional matching with low-confidence detection results through the second stage enhances the ability to handle complex scenarios and maintain the identity consistency of the tracker. In contrast, the 1-stage strategy from BoostTrack[35] and the 4-stage strategy[47] perform less favorably, suggesting that using only high-confidence detection results or overcomplicating the association process may introduce noise or overfitting that can diminish robustness. The 1-stage combined matching strategy[39], which penalizes lowconfidence detection results, performs competitively but ofers marginal improvements over the basic 1-stage approach except in DanceTrack. These results demonstrate that the 2-stage association strategy provides the best balance of accuracy and robustness. It shows giving chances to low-confidence detections after prioritizing high-confidence detection results can guarantee robust tracking performance. Therefore, the tracker that will serve as the new baseline adopts the 2-stage strategy.

Table 13: Comparison of matching with diferent association strategies. 2-stage association is the best balanced strategy for TBD.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Association</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>1-stage</td><td>68.94</td><td>80.66</td><td>86.24</td><td>66.42</td><td>72.01</td><td>68.42</td><td>85.81</td><td>89.57</td><td>67.48</td><td>69.56</td><td>62.85</td><td>92.09</td><td>65.94</td><td>80.34</td><td>49.35</td></tr><tr><td>1-stage (unconfirmed tracks)</td><td>68.88</td><td>80.67</td><td>86.09</td><td>66.34</td><td>71.97</td><td>68.44</td><td>85.81</td><td>89.62</td><td>67.48</td><td>69.59</td><td>62.98</td><td>92.67</td><td>65.78</td><td>80.85</td><td>49.26</td></tr><tr><td>1-stage (BoostTrack)</td><td>67.74</td><td>79.02</td><td>84.72</td><td>65.14</td><td>70.87</td><td>66.85</td><td>83.44</td><td>88.08</td><td>65.87</td><td>68.03</td><td>61.69</td><td>90.52</td><td>63.35</td><td>79.79</td><td>47.84</td></tr><tr><td>1-stage (Combined matching)</td><td>68.83</td><td>80.70</td><td>86.09</td><td>66.48</td><td>71.73</td><td>68.40</td><td>85.88</td><td>89.51</td><td>67.55</td><td>69.45</td><td>63.58</td><td>92.07</td><td>66.87</td><td>80.23</td><td>50.56</td></tr><tr><td>2-stage</td><td>69.35</td><td>81.04</td><td>87.16</td><td>66.77</td><td>72.45</td><td>68.58</td><td>86.07</td><td>89.93</td><td>67.65</td><td>69.72</td><td>63.27</td><td>92.71</td><td>66.07</td><td>80.84</td><td>49.71</td></tr><tr><td>4-stage</td><td>67.78</td><td>79.96</td><td>84.56</td><td>65.82</td><td>70.24</td><td>67.24</td><td>84.27</td><td>88.16</td><td>66.47</td><td>68.20</td><td>61.39</td><td>91.05</td><td>64.36</td><td>79.27</td><td>47.74</td></tr></table>

Table 14: Comparison between matching with/without score fusion. Adopting score fusion makes slightly better HOTAs.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Score Fusion</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>√</td><td>69.35</td><td>81.04</td><td>87.16</td><td>66.77</td><td>72.45</td><td>68.58</td><td>86.07</td><td>89.93</td><td>67.65</td><td>69.72</td><td>63.27</td><td>92.71</td><td>66.07</td><td>80.84</td><td>49.71</td></tr><tr><td>x</td><td>69.25</td><td>81.01</td><td>87.09</td><td>66.69</td><td>72.34</td><td>68.56</td><td>86.05</td><td>89.89</td><td>67.61</td><td>69.72</td><td>63.35</td><td>92.71</td><td>66.12</td><td>80.84</td><td>49.83</td></tr></table>

## 5.12 Score Fusion

Table 14 assesses the impact of score fusion[14] in the second association phase of the 2- stage tracker. The baseline configuration with score fusion achieves robust performance with HOTA scores of 69.35, 68.58, and 63.27, respectively, across the datasets. In contrast, omitting the score fusion approach results in marginal performance degradation in MOT17 and MOT20, while bringing slight improvement in DanceTrack. These findings suggest that score fusion provides modest but valuable improvements. Although it shows slight HOTA degradation in DanceTrack, it still maintains detection-related performance. The added stability through score fusion suggests that it can be a beneficial refinement in environments with complex object interactions. Therefore, we keep including the score fusion in our tracker.

Table 15: Comparison of matching with diferent post-processing methods. AFLink clearly afects the complex scenarios but solely applying gaussian interpolation draws the most balanced performance across datasets.
<table><tr><td colspan="2"></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>AFLink</td><td>Interpolation</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>x</td><td>x</td><td>69.35</td><td>81.04</td><td>87.16</td><td>66.77</td><td>72.45</td><td>68.58</td><td>86.07</td><td>89.93</td><td>67.65</td><td>69.72</td><td>63.27</td><td>92.71</td><td>66.07</td><td>80.84</td><td>49.71</td></tr><tr><td>√</td><td>x</td><td>68.97</td><td>81.04</td><td>86.28</td><td>66.70</td><td>71.75</td><td>68.53</td><td>86.04</td><td>89.72</td><td>67.61</td><td>69.65</td><td>63.68</td><td>92.72</td><td>66.78</td><td>80.82</td><td>50.37</td></tr><tr><td>x</td><td>LI</td><td>70.16</td><td>82.34</td><td>87.76</td><td>67.81</td><td>73.03</td><td>68.91</td><td>86.45</td><td>90.14</td><td>67.91</td><td>70.11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>x</td><td>GSI</td><td>70.34</td><td>82.61</td><td>87.85</td><td>67.97</td><td>73.34</td><td>69.69</td><td>86.82</td><td>90.19</td><td>68.66</td><td>71.00</td><td>=</td><td>=</td><td>=</td><td>=</td><td>=</td></tr><tr><td>x</td><td>GBI</td><td>70.79</td><td>82.78</td><td>87.93</td><td>68.49</td><td>73.66</td><td>69.21</td><td>86.79</td><td>90.18</td><td>68.25</td><td>70.39</td><td>=</td><td>=</td><td>=</td><td>=</td><td>=</td></tr><tr><td>√</td><td>LI</td><td>69.72</td><td>82.32</td><td>86.86</td><td>67.72</td><td>72.21</td><td>68.85</td><td>86.54</td><td>89.90</td><td>67.94</td><td>69.96</td><td>=</td><td></td><td>，</td><td>=</td><td></td></tr><tr><td>√</td><td>GSI</td><td>69.82</td><td>82.28</td><td>86.88</td><td>67.73</td><td>72.54</td><td>69.62</td><td>86.70</td><td>90.05</td><td>68.63</td><td>70.90</td><td>=</td><td></td><td></td><td></td><td></td></tr><tr><td>√</td><td>GBI</td><td>70.28</td><td>82.64</td><td>86.99</td><td>68.32</td><td>72.80</td><td>69.14</td><td>86.67</td><td>90.02</td><td>68.22</td><td>70.29</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.13 Post-Processing

Table 15 compares several post-processing strategies, including Linear Interpolation (LI)[14], Gaussian Score Interpolation (GSI)[23, 48], Gaussian Blur Interpolation (GBI)[35, 36], and Afinity Link (AFLink)[23], across MOT17, MOT20, and Dance-Track. Interpolation-based methods are not evaluated on DanceTrack because the dataset does not provide ground-truth annotations for fully occluded objects. Interpolation may link track segments across long occlusions where objects are not visible, leading to erroneous trajectory reconstruction and potential false positives because of the absence of ground-truth annotations. In this manner, we only evaluate Interpolation-based methods on MOT17 and MOT20.

Applying interpolation generally improves performance on MOT17 and MOT20 compared to the baseline without post-processing. In particular, GBI achieves the best performance on MOT17 with HOTA of 70.79, while GSI shows the strongest results on MOT20, achieving the highest HOTA of 69.69 and improving both detection and association metrics. LI also consistently improves the baseline, indicating that reconnecting short-term track fragmentation efectively enhances trajectory continuity.

AFLink shows diferent behaviors across datasets. While it slightly degrades performance on MOT17 and MOT20, it improves performance on DanceTrack, achieving the best HOTA of 63.68 and the highest association accuracy. This indicates that AFLink is particularly beneficial in highly dynamic scenarios where appearance and motion patterns vary significantly. However, combining AFLink with interpolation methods does not yield additional gains and generally performs slightly worse than using interpolation alone. Overall, Gaussian-based interpolation methods provide the most balanced performance across datasets.

## 5.14 Final Baseline

Based on the findings from the component-wise analysis, we construct a final baseline tracker by integrating the most efective design choices. Starting from the basic tracking pipeline, SORT, we adopt a Kalman state vector defined as (cx, cy, w, h)[15] to provide a more direct spatial representation of bounding boxes. To improve robustness in practical scenarios, we incorporate Occlusion-aware Initialization (OAI)[39], Confidence Weighted Kalman-Update (CWKU)[40], and Global Motion Compensation (GMC)[30], which address occlusion handling, improve the reliability of state updates, and compensate for camera-induced motion, respectively. To further enhance performance, several implementation refinements are applied, including setting the track activation frame count to 2, introducing Detection-Track Confidence Similarity Boost[35], and applying an Exponential Moving Average (EMA)[45] for smoother appearance feature updates and more stable association. Finally, we employ weighted sum score fusion and a two-stage association strategy[14], enabling more reliable matching under varying detection confidence levels. Trajectory continuity is additionally improved through post-processing, Gaussian-based interpolation[36, 48], establishing a strong and generalized baseline across diverse MOT scenarios. As shown in Table 15, the proposed baseline significantly improves performance over the original SORT tracker across all datasets, demonstrating substantial gains in both detection and association quality. Our final performance improves HOTA of 5.44, 2.18, and 10.01 on MOT18, MOT20, and DanceTrack, respectively. The increase on DanceTrack was the largest, which indicates our new baseline can deal better with complicated scenarios. Similar improvements are observed in IDF1 and AssA, indicating more reliable identity preservation and trajectory consistency.

Table 16: Performance comparison between the original SORT baseline and the proposed final baseline. Our tracker integrates the most efective components identified in the ablation studies and shows consistent improvements across MOT17, MOT20, and DanceTrack. Additional gains are obtained when post-processing is applied.
<table><tr><td></td><td colspan="5">MOT17</td><td colspan="5">MOT20</td><td colspan="5">DanceTrack</td></tr><tr><td>Tracker</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>DetA↑</td><td>AssA↑</td></tr><tr><td>First baseline(SORT)</td><td>63.91</td><td>78.66</td><td>77.85</td><td>65.21</td><td>63.17</td><td>66.40</td><td>84.97</td><td>86.20</td><td>66.80</td><td>66.20</td><td>53.26</td><td>90.86</td><td>52.55</td><td>79.70</td><td>35.76</td></tr><tr><td>Ours</td><td>69.35</td><td>81.04</td><td>87.16</td><td>66.77</td><td>72.45</td><td>68.58</td><td>86.07</td><td>89.93</td><td>67.65</td><td>69.72</td><td>63.27</td><td>92.71</td><td>66.07</td><td>80.84</td><td>49.71</td></tr><tr><td>Ours w/ post-processing</td><td>69.82</td><td>82.28</td><td>86.88</td><td>67.73</td><td>72.54</td><td>69.62</td><td>86.70</td><td>90.05</td><td>68.63</td><td>70.90</td><td></td><td></td><td></td><td></td><td>=</td></tr></table>

## 6 Conclusion

In this work, we present a systematic analysis of the tracking-by-detection (TBD) paradigm by examining how individual design components influence the performance of multi-object tracking (MOT) systems. Through controlled experiments across three diverse benchmarks—MOT17, MOT20, and DanceTrack—we investigate the role of key tracking modules, including motion modeling, initialization strategies, data association mechanisms, and post-processing techniques.

Our study highlights that the efectiveness of a tracker does not arise from a single component but rather from the careful integration of complementary design choices within the TBD pipeline. By evaluating these components under a consistent experimental protocol, we identify combinations that lead to stable and well-balanced tracking performance across datasets with diferent characteristics, such as dense crowds and highly dynamic motions.

Beyond establishing a strong baseline, our analysis provides practical insights into the design of reliable TBD trackers and ofers a clearer understanding of how individual modules interact within the overall tracking framework. We hope that these findings serve as a useful reference for future research and facilitate the development of more robust and versatile MOT systems for real-world applications.

## Declarations

Funding The authors received no specific funding for this work.

Conflict of interest/Competing interests The authors declare that they have no competing interests.

Ethics approval and consent to participate Not applicable.

Consent for publication Not applicable.

Data availability Data sharing not applicable to this article as no datasets were generated during the current study. The figures and tables were derived from publicly available literature.

Materials availability Not applicable.

Code availability The source code supporting the findings of this study will be made publicly available in a GitHub repository upon acceptance of the manuscript.

Author contribution Conceptualization, Y.Y and K.S.; methodology, Y.Y. and K.S.; investigation, Y.Y and K.S.; software, Y.Y and K.S.; formal analysis, Y.Y., K.S., and K.K; writing–original draft preparation, Y.Y. and K.S; writing–review and editing, Y.Y., K.S and K.K; visualization, Y.Y and K.K.; supervision, C.K.; project administration, C.K. All authors have read and approved the final manuscript.

## References

[1] Ristani, E., Tomasi, C.: Features for Multi-target Multi-camera Tracking and Reidentification. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6036–6046 (2018)

[2] Elhoseny, M.: Multi-object detection and tracking (modt) machine learning model for real-time video surveillance systems. Circuits, Systems, and Signal Processing 39(2), 611–630 (2020)

[3] Liu, Z., Wang, X., Wang, C., Liu, W., Bai, X.: Sparsetrack: Multi-object tracking by performing scene decomposition based on pseudo-depth. IEEE Transactions on Circuits and Systems for Video Technology 35(5), 4870–4882 (2025) https://doi.org/10.1109/TCSVT.2024.3524670

[4] Hassan, S., Mujtaba, G., Rajput, A., Fatima, N.: Multi-object tracking: a systematic literature review. Multimedia Tools and Applications 83(14), 43439–43492 (2024)

[5] Luo, C., Yang, X., Yuille, A.: Exploring Simple 3d Multi-object Tracking for Autonomous Driving. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10488–10497 (2021)

[6] Zhang, X., Fan, Z., Tan, X., Liu, Q., Shi, Y.: Spatiotemporal adaptive attention 3d multiobject tracking for autonomous driving. Knowledge-Based Systems 267, 110442–110453 (2023)

[7] Ravindran, R., Santora, M.J., Jamali, M.M.: Multi-object detection and tracking,

based on dnn, for autonomous vehicles: A review. IEEE Sensors Journal 21(5), 5668–5677 (2020)

[8] Chang, X., Pan, H., Sun, W., Gao, H.: Yoltrack: Multitask learning based real-time multiobject tracking and segmentation for autonomous vehicles. IEEE Transactions on Neural Networks and Learning Systems 32(12), 5323–5333 (2021)

[9] Guidolin, M., Tagliapietra, L., Menegatti, E., Reggiani, M.: Hi-ros: Open-source multi-camera sensor fusion for real-time people tracking. Computer Vision and Image Understanding 232, 103694 (2023)

[10] Wu, D., Han, W., Wang, T., Dong, X., Zhang, X., Shen, J.: Referring multiobject tracking. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14633–14642 (2023). IEEE

[11] Ge, Z., Liu, S., Wang, F., Li, Z., Sun, J.: Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430 (2021)

[12] Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B.: Simple Online and Realtime Tracking. In: Proceedings of the IEEE International Conference on Image Processing, pp. 3464–3468 (2016)

[13] Wojke, N., Bewley, A., Paulus, D.: Simple Online and Realtime Tracking with a Deep Association Metric. In: Proceedings of the IEEE International Conference on Image Processing, pp. 3645–3649 (2017)

[14] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., Wang, X.: Bytetrack: Multi-object Tracking by Associating Every Detection Box. In: Proceedings of the European Conference on Computer Vision, pp. 1–21 (2022)

[15] Aharon, N., Orfaig, R., Bobrovsky, B.-Z.: Bot-sort: Robust associations multipedestrian tracking. arXiv preprint arXiv:2206.14651 (2022)

[16] Shim, K., Ko, K., Yang, Y., Kim, C.: Focusing on tracks for online multiobject tracking. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11687–11696 (2025)

[17] Jin, Y., Gao, F., Yu, J., Wang, J., Shuang, F.: Multi-object tracking: Decoupling features to solve the contradictory dilemma of feature requirements. IEEE Transactions on Circuits and Systems for Video Technology 33(9), 5117–5132 (2023) https://doi.org/10.1109/TCSVT.2023.3249162

[18] Kalman, R.E.: A new approach to linear filtering and prediction problems. Journal of Basic Engineering 82(1), 35–45 (1960) https://doi.org/10.1115/1.3662552

[19] Munkres, J.: Algorithms for the assignment and transportation problems. Journal of the Society for Industrial and Applied Mathematics 5, 32–38 (1957) https://doi.org/10.1137/0105003

[20] Yang, M., Han, G., Yan, B., Zhang, W., Qi, J., Lu, H., Wang, D.: Hybrid-sort: Weak Cues Matter for Online Multi-object Tracking. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 6504–6512 (2024)

[21] Zhang, Y., Wang, C., Wang, X., Zeng, W., Liu, W.: Fairmot: On the fairness of detection and re-identification in multiple object tracking. International journal of computer vision 129, 3069–3087 (2021)

[22] Zhang, Y., Xie, H., Jia, Y., Meng, J., Sang, M., Qiu, J., Zhao, S., Yang, Y.: Aipt: Adaptive information perception for online multi-object tracking. Knowledge-Based Systems 285, 111369 (2024)

[23] Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., Meng, H.: Strongsort: Make deepsort great again. IEEE Transactions on Multimedia 25, 8725–8737 (2023)

[24] Milan, A., Leal-Taix´e, L., Reid, I., Roth, S., Schindler, K.: Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831 (2016)

[25] Sun, P., Cao, J., Jiang, Y., Yuan, Z., Bai, S., Kitani, K., Luo, P.: Dancetrack: Multi-object Tracking in Uniform Appearance and Diverse Motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20993–21002 (2022)

[26] Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28 (2015)

[27] Zagoruyko, S., Komodakis, N.: Wide residual networks. arXiv preprint arXiv:1605.07146 (2016)

[28] Stadler, D., Beyerer, J.: Modelling ambiguous assignments for multi-person tracking in crowds. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, pp. 133–142 (2022)

[29] Chu, X., Zheng, A., Zhang, X., Sun, J.: Detection in crowded scenes: One proposal, multiple predictions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12214–12223 (2020)

[30] Bradski, G.: The opencv library. dr. dobb’s journal of software tools,(4.7. 0). computer program] Available at: https://opencv. org [Accessed: 04 April 2023] (2000)

[31] Luo, H., Jiang, W., Gu, Y., Liu, F., Liao, X., Lai, S., Gu, J.: A strong baseline and

batch normalization neck for deep person re-identification. IEEE Transactions on Multimedia 22(10), 2597–2609 (2019)

[32] Evangelidis, G.D., Psarakis, E.Z.: Parametric image alignment using enhanced correlation coeficient maximization. IEEE transactions on pattern analysis and machine intelligence 30(10), 1858–1865 (2008)

[33] He, L., Liao, X., Liu, W., Liu, X., Cheng, P., Mei, T.: Fastreid: A Pytorch Toolbox for General Instance Re-identification. In: Proceedings of the 31st ACM International Conference on Multimedia, pp. 9664–9667 (2023)

[34] Maggiolino, G., Ahmad, A., Cao, J., Kitani, K.: Deep oc-sort: Multi-pedestrian tracking by adaptive re-identification. In: 2023 IEEE International Conference on Image Processing (ICIP), pp. 3025–3029 (2023)

[35] Stanojevic, V.D., Todorovic, B.T.: Boosttrack: boosting the similarity measure and detection confidence for improved multiple object tracking. Machine Vision and Applications 35(3), 1–15 (2024)

[36] Zeng, K., You, Y., Shen, T., Wang, Q., Tao, Z., Wang, Z., Liu, Q.: Nct:noisecontrol multi-object tracking. Complex & Intelligent Systems 9(4), 4331–4347 (2023)

[37] Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K., Leal-Taix´e, L.: Mot20: A benchmark for multi object tracking in crowded scenes. arXiv preprint arXiv:2003.09003 (2020)

[38] Yang, F., Odashima, S., Masui, S., Jiang, S.: Hard to track objects with irregular motions and similar appearances? make it easier by bufering the matching space. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 4799–4808 (2023)

[39] Stadler, D., Beyerer, J.: An improved association pipeline for multi-person tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3170–3179 (2023)

[40] Jung, H., Kang, S., Kim, T., Kim, H.: Conftrack: Kalman filter-based multiperson tracking by utilizing confidence score of detection box. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 6583–6592 (2024)

[41] Cao, J., Pang, J., Weng, X., Khirodkar, R., Kitani, K.: Observation-centric sort: Rethinking sort for robust multi-object tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9686–9696 (2023)

[42] Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized intersection over union: A metric and a loss for bounding box regression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 658–666 (2019)

[43] Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., Ren, D.: Distance-iou loss: Faster and better learning for bounding box regression. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 12993–13000 (2020)

[44] Stadler, D., Beyerer, J.: Past information aggregation for multi-person tracking. In: 2023 IEEE International Conference on Image Processing (ICIP), pp. 321–325 (2023)

[45] Wang, Z., Zheng, L., Liu, Y., Li, Y., Wang, S.: Towards real-time multi-object tracking. In: European Conference on Computer Vision, pp. 107–122 (2020). Springer

[46] Ren, H., Han, S., Ding, H., Zhang, Z., Wang, H., Wang, F.: Focus on details: Online multi-object tracking with diverse fine-grained representation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11289–11298 (2023)

[47] Meng, T., Fu, C., Huang, M., Wang, X., He, J., Huang, T., Shi, W.: Localizationguided track: A deep association multi-object tracking framework based on localization confidence of detections. arXiv:2309.09765 (2023)

[48] Williams, C., Rasmussen, C.: Gaussian processes for regression. Advances in neural information processing systems 8 (1995)

[49] Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., Sun, J.: Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123 (2018)

[50] Zhang, S., Xie, Y., Wan, J., Xia, H., Li, S.Z., Guo, G.: Widerperson: A diverse dataset for dense pedestrian detection in the wild. IEEE Transactions on Multimedia 22(2), 380–393 (2019)

[51] Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., Doll´ar, P.: Microsoft COCO: Common Objects in Context. In: Proceedings of the European Conference on Computer Vision, pp. 740–755 (2014)

[52] Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., Tian, Q.: Scalable Person Re-identification: A Benchmark. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 1116–1124 (2015)

[53] Schrof, F., Kalenichenko, D., Philbin, J.: Facenet: A Unified Embedding for Face

Recognition and Clustering. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 815–823 (2015)

[54] Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taix´e, L., Leibe, B.: Hota: A higher order metric for evaluating multi-object tracking. International Journal of Computer Vision 129, 548–578 (2021)

[55] Stiefelhagen, R., Bernardin, K., Bowers, R., Garofolo, J., Mostefa, D., Soundararajan, P.: The CLEAR 2006 Evaluation. In: Multimodal Technologies for Perception of Humans, pp. 1–44 (2007)

[56] Ristani, E., Solera, F., Zou, R., Cucchiara, R., Tomasi, C.: Performance Measures and a Data Set for Multi-target, Multi-camera Tracking. In: The European Conference on Computer Vision Workshop (2016)

[57] Zhang, H., Wu, C., Zhang, Z., Zhu, Y., Lin, H., Zhang, Z., Sun, Y., He, T., Mueller, J., Manmatha, R., et al.: Resnest: Split-attention networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2736–2746 (2022)