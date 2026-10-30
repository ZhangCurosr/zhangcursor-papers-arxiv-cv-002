# Decision-Level Error Attribution and Depth-Aware Safety Evaluation of Monocular RGB Human Tracking for Autonomous Robot Navigation

Abdulahi Taiwo

Department of Computer Science and Engineering

University of South Carolina

ataiwo@email.sc.edu

Chijioke Cyriacus Ekechi

Ottawa University, Ottawa, KS USA

chijiokeekechi@gmail.com

Abstract—Monocular RGB human tracking is attractive for robots that work near people, but its errors reach the robot through a binary stop/proceed decision that accuracy metrics do not describe. Our earlier single-participant study (prof of concept) reported that a YOLOv8-Pose, geometric-depth and Kalmanfilter pipeline with a lateral-only distance rule missed no dangers yet produced frequent unnecessary stops, without being able to explain why. Here we prove that a lateral-only distance is a lower bound on planar distance, so the rule cannot miss a danger on accurate input and its false stops follow from geometry. We then introduce a two-factor Shapley attribution that splits decision errors into rule and perception shares, with the perception share further divided into missed detections and localisation error, and apply it to public pedestrian video from the ETH and UCY datasets with simulated stationary robots, a calibration/evaluation split with disjoint pedestrians, and time-block bootstrap intervals. On annotated positions from 898 pedestrians in three scenes, the lateral rule had zero missed danger and a stop precision of 5.6– 31.9%, closely matching a closed-form occupancy model. With the full RGB pipeline in two scenes, a calibrated ground-plane mapping localised pedestrians with 0.59 m and 0.51 m planar RMSE, against 1.72 m and 1.39 m for the reconstructed geometric baseline and 0.81 m and 0.68 m for Depth Anything V2 with a calibrated depth map. Detection recall of 0.43 and 0.26 made missed detections the largest source of missed danger. With the ground-plane localiser, a two-sigma uncertainty margin on planar clearance missed no more danger than the lateral rule in all 8 scene–threshold cells and produced fewer unnecessary stops in 7 of them; with the geometric baseline the same margin multiplied unnecessary stops by 2.4–3.3. All robot interactions are simulated; the results are offline decision classifications, not evidence of collision avoidance.

Index Terms—Human motion estimation, monocular RGB, robot safety, decision-level evaluation, error attribution, Kalman filter, pedestrian datasets.

## I. INTRODUCTION

Mobile robots that share corridors, wards and warehouse aisles with people must decide, continuously, whether to proceed or stop [1, 2, 3]. In industrial collaborative robotics this decision is formalised as speed and separation monitoring, where the protective separation between a person and the robot drives the robot’s speed [4, 5]. Low-cost monocular RGB cameras are attractive sensors for estimating that separation, but they do not measure depth, and their errors reach the robot only through a binary decision.

Perception modules are nevertheless usually reported with geometric accuracy metrics (RMSE, MAE, ADE/FDE). Our earlier study [6] evaluated a YOLOv8-Pose, geometric-depth and Kalman-filter pipeline against OptiTrack ground truth for one participant and reported a decision-level asymmetry under a 500 mm threshold: 0.0% missed-danger frames together with 65.0% unnecessary-stop frames. That study could not say where the asymmetry came from. Its own discussion named three candidates—monocular depth error, a lateral-only distance rule, and the threshold decision itself—and it used a single participant, one session, and calibration parameters estimated on the evaluated frames.

This paper asks a narrower and testable question: when a monocular RGB human-localisation pipeline drives a stop/proceed rule, how much of the resulting decision error is caused by the decision rule’s geometry, how much by localisation error, and how much by missed detections? We answer it on public, real pedestrian video with metric groundplane annotations, using simulated robot placements, a calibration/evaluation split that never shares a pedestrian, and time-block bootstrap intervals.

The contributions are:

1. A formal result (Proposition 1) showing that a lateralonly distance is a lower bound on planar distance, so that rule can never miss a danger that the planar rule detects and its 0% miss rate on accurate input is a property of geometry, not of perception.

2. An analysis that separates decision errors caused by the safety rule from those caused by perception, by running both rules on ground-truth and on cameraestimated positions, and that splits perception-caused missed dangers into missed detections and localisation errors.

3. An empirical study on three real scenes (3 for rule geometry, 2 with the full RGB pipeline) comparing the reconstructed geometric-depth baseline, a calibrationbased ground-plane method, and Depth Anything V2 [7], evaluated by their effect on decisions, events and unnecessary stops rather than by RMSE alone.

4. Results from the earlier laboratory study are quoted only as reported there; they were not re-computed here, because the raw OptiTrack recording was not used in this work.

## II. RELATED WORK

Human-aware navigation. Surveys of socially aware navigation catalogue how human motion models enter planning [1, 2], and application studies address hospital conavigation [3], human following and guidance [8], long-term movement prediction [9] and prediction-aware obstacle avoidance [10]. These works assume a perception front end and rarely quantify how its errors change the robot’s discrete decisions.

Safety monitoring around people. Speed and separation monitoring couples measured separation to robot motion [4, 5]. Camera-based safety systems classify operators into safety zones [11], and monocular proximity-warning systems classify workers into proximity categories and report precision, recall and F1 [12]. They evaluate a complete system; they do not separate the contribution of the zone geometry from that of the estimator.

Monocular human localisation. Pedestrian distance can be inferred from 2-D keypoints with learned uncertainty [13], from the ground-contact point and scene geometry [14], or from learned monocular depth such as Depth Anything V2 [7]; multimodel monocular navigation systems have been built on such cues [15, 16]. Keypoints here come from YOLOv8-Pose [17, 18] trained on COCO [19].

Task-aware evaluation of perception. In autonomous driving, planner-centric metrics weight detection errors by their effect on a planner [20], and planning-aware metrics have been proposed for detection and prediction [21]. Our study shares the premise that accuracy metrics are insufficient, but targets a different question: it attributes the errors of a simple, auditable stop rule to rule geometry versus perception, in a human-safety setting with monocular depth.

Ground truth for human motion. Marker-based motion capture is the usual reference for markerless tracking [22, 23, 24, 25]; multi-view studios [26] and robot-mounted datasets [27] broaden the setting. We use the ETH [28] and UCY [29] pedestrian recordings, distributed with homographies through OpenTraj [30], because they were the only calibrated video-plus-metric-trajectory data obtainable from our execution environment (Section V). Their ground truth is manual annotation, not motion capture.

## III. PROBLEM FORMULATION

A pedestrian at ground position $\mathbf { p } ( t ) \in \mathbb { R } ^ { 2 }$ and a stationary robot modelled as a disc of radius � centred at � have planar clearance

$$
d ( t ) = \parallel \mathbf { p } ( t ) - \mathbf { r } \parallel - R .\tag{1}
$$

Let � be the unit ground-plane vector of the camera’s lateral (image-column) direction. The lateral-only rule of [6] replaces � by

$$
d _ { \mathrm { l a t } } ( t ) = | ( { \bf p } ( t ) - { \bf r } ) ^ { \top } { \bf e } | - R .\tag{2}
$$

Given a threshold �, the ground-truth label of a sample is unsafe (positive) if $d ( t ) < \tau ,$ and a rule outputs STOP if its distance, computed from its input �̂, is below �. A false negative (FN) is an unsafe sample with PROCEED; a false positive (FP) is a safe sample with STOP. $\mathrm { F N R } = \mathrm { F N } / ( \mathrm { T P } + \mathrm { F N } )$ and $\mathrm { F P R = }$ FP/(FP+TN).

Proposition 1. For any input �̂, ${ d _ { \mathrm { l a t } } ( \widehat { \mathbf { p } } ) } \leq d ( \widehat { \mathbf { p } } )$ . Hence the set of samples on which the lateral rule stops contains the set on which the planar rule stops; on the same input the lateral rule has $\mathrm { F N _ { l a t } } \le \mathrm { F N _ { p l } }$ and $\mathrm { F P _ { l a t } } \geq \mathrm { F P _ { p l } }$ . With exact input $( \widehat { \mathbf { p } } =$ $\mathbf { p } ) , \ \mathrm { F N _ { l a t } } = 0$ and the false positives are exactly the samples inside the strip $| \varDelta ^ { \intercal } \mathbf { e } | < \tau + \bar { R }$ but outside the disc ∥ $\varDelta \parallel < \tau +$ �, with $\varDelta = \mathbf { p } - \mathbf { r }$

Proof. $| A ^ { \top } \mathbf { e } | \leq \parallel \varDelta$ ∥∥ $\mathbf { e } \parallel = \parallel \boldsymbol { \varDelta }$ ∥ (Cauchy–Schwarz), so $d _ { \mathrm { l a t } } \leq d$ , and $d < \tau \Rightarrow d _ { \mathrm { l a t } } < \tau$ . The rest follows by comparing the two stop sets sample by sample. ◻

If pedestrian positions were spread uniformly along the camera depth direction over a length $D \gg \tau + R$ , the fraction of lateral-rule stops that are justified would be the disc-to-strip area ratio $\pi ( \tau + R ) / ( 2 D )$ . The rule’s precision therefore falls as the scene gets deeper and rises with the threshold, independently of the sensor. Real occupancy is not uniform; we report measured values.

Error attribution. Let $E ( I , \rho )$ be an error rate (FNR or FPR) obtained with input $I \in \{ \mathrm { G T } , \mathrm { R G B } \}$ and rule $\rho \in \{ \mathrm { l a t , p l } \}$ on the same samples. The total change from the ideal system $E \mathrm { ( G T , p l ) }$ to the deployed one �(RGB,lat) is split into the two-player Shapley shares

$$
\begin{array} { r } { \phi _ { \mathrm { r u l e } } = \frac { 1 } { 2 } \big [ \big ( E _ { \mathrm { G T , l a t } } - E _ { \mathrm { G T , p l } } \big ) + \big ( E _ { \mathrm { R G B , l a t } } - E _ { \mathrm { R G B , p l } } \big ) \big ] , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \phi _ { \mathrm { p e r c } } = \frac { 1 } { 2 } \big [ \big ( E _ { \mathrm { R G B , p l } } - E _ { \mathrm { G T , p l } } \big ) + \big ( E _ { \mathrm { R G B , l a t } } - E _ { \mathrm { G T , l a t } } \big ) \big ] , } \end{array}\tag{4}
$$

which sum exactly to $E _ { \mathrm { R G B , l a t } } - E _ { \mathrm { G T , p l } } .$ . Because $E _ { \mathrm { G T , p l } } = 0$ by construction, the two shares partition the deployed error. We compute them twice: on all samples (so that detection misses count as perception error) and on the samples where the pipeline produced an estimate (isolating localisation error). Under the planar rule we also split FNR into samples with no estimate and samples with a wrong estimate.

## IV. FRAMEWORK AND PIPELINE

![](images/f01f9b3fd77b2faeaae8d06b51ac56108e62a8309292e22ed7dfacf7c1ac31c2.jpg)  
Fig. 1. Evaluated chain (conceptual diagram). Ground truth enters the estimation path only through parameters fitted on the calibration half (dashed). Robot placements are simulated. The oracle input arm of the attribution replaces the estimated positions by annotated ones.

Fig. 1 shows the evaluated chain. Every stage is fixed before evaluation-half ground truth is inspected.

Detection and keypoints. YOLOv8m-Pose [17] (official v8.3.0 checkpoint) runs on every annotated frame at an inference size of 1280 px with confidence threshold 0.25 and NMS IoU 0.7. Probing smaller models and the default 640 px input showed near-zero recall on these elevated views (Section VI-A). Because PyTorch could not be installed in the execution environment, we run the checkpoint through a NumPy re-implementation of the network (batch-norm folded into convolutions, DFL box decoding, the Ultralytics keypoint decoding and NMS), checked on the Ultralytics sample images.

Localisation methods. We compare three ways to place a detected person on the ground plane.

M0, geometric (reconstruction of [6]). Depth along the optical axis from the vertical nose-to-knee pixel span $\begin{array} { r } { h _ { \mathrm { p x } } , Z = f H _ { \mathrm { r e a l } } / h _ { \mathrm { p x } } , } \end{array}$ with $H _ { \mathrm { r e a l } } = 0 . 9$ m and � from an assumed 65<sup>∘</sup> horizontal field of view, as in the original. The lateral coordinate comes from the bodycentre pixel column (mean of shoulders and hips). The original’s min–max normalisation and its median-ratio depth scale both used the evaluated ground truth; we replace them by affine maps (column→lateral, � →depth) fitted on the calibration half only. Keypoints with confidence below 0.5 make the estimate unavailable.

M1, ground plane. A reference pixel is mapped to the ground through the scene homography. No parameter is fitted; the reference point (ankle midpoint, nose, or box bottom) is chosen on the calibration half by median error.

M2, learned depth. As M0, but the depth cue is Depth Anything V2-Small relative inverse depth [7] sampled at the ankle midpoint, mapped to the depth coordinate by $d = a / ( \delta - t ) + c$ fitted robustly on the calibration half. The metric variants of Depth Anything V2 could not be downloaded in our environment; the relative model needs this calibration.

Tracking and prediction. Each pedestrian’s measurements feed a constant-velocity Kalman filter [31] on the ground plane $( \varDelta t = 0 . 4 \ : \mathrm { s } )$ . The measurement covariance is the empirical covariance of calibration-half residuals of each method, and the process-noise intensity is chosen on the calibration half from $\{ 0 . 0 5 , \dots , 4 \} \mathrm { m } ^ { 2 } \mathrm { s } ^ { - 3 }$ . A track coasts through at most two missed detections. For prediction we compare the filter with raw constant-velocity extrapolation, a constant-position predictor, and constant-velocity extrapolation of the ground truth itself, which measures how predictable the motion is [32].

Decision rules. Lateral rule (2), planar rule (1), and an uncertainty margin rule that stops if �<sup>̂</sup> − ��̂ < �, with �̂ the filter’s position standard deviation (or the calibration RMS error for unfiltered input) and � ∈ {0.5,1,2}. Thresholds � ∈ {0.30,0.50,0.75,1.00} m on clearance.

## V. DATASETS AND EXPERIMENTAL METHODOLOGY

Data. We use seq\_eth and seq\_hotel from the ETH Walking Pedestrians dataset [28] and zara01 from UCY [29], obtained from the OpenTraj repository [30] (commit 97ec1d5). Each provides video from a static elevated camera, a homography from image to ground plane, and manually annotated ground positions in metres at 2.5 Hz. We verified the image–world convention by projecting annotations onto frames. Two further UCY scenes (zara02, students03) failed this check and were excluded. Table I summarises the scenes. Access to HumanEva-I, Human3.6M, JRDB and CMU Panoptic required registration or network hosts that were unavailable in our environment; these are the intended next validation sets.

Simulated robots. No robot appears in these recordings. We place a stationary robot disc (� = 0.30 m) at every node of a

1 m grid over the walkable area of each scene (2–98th percentile extent of annotated positions, keeping nodes within 1 m of some annotated position). Each (pedestrian, robot, frame) triple is one decision sample. We also report robot-level outcomes, where a robot stops if any tracked pedestrian triggers it. All decision results are offline classifications on recorded human motion with simulated robot placements; they do not demonstrate collision avoidance by a deployed robot.

Splits and leakage control. Within each scene, annotated frames up to the median frame form the calibration half; later frames form the evaluation half, restricted to pedestrian IDs that never appear in the calibration half. All fitted parameters (affine maps, reference-point choice, Kalman noise) use the calibration half only. Detections are associated with annotated pedestrians in image space by the Hungarian algorithm (distance between the projected annotation and the box centre, normalised by box height, gate 0.8). Tracking therefore uses oracle data association, which isolates localisation from identity errors.

Events and uncertainty. An unsafe event is a maximal run of consecutive unsafe samples for one (pedestrian, robot) pair; it is missed if no STOP occurs within it. An unnecessary-stop episode is a maximal run of consecutive false positives. Video frames are strongly correlated, so we never treat frames as independent: intervals for decision metrics come from a bootstrap over contiguous 10 s time blocks [33] (1000 resamples), and intervals for localisation and prediction errors from a bootstrap over pedestrians. With few blocks per scene these intervals are approximate, and we make no populationlevel significance claims.

TABLE I  
SCENES USED. ANNOTATIONS ARE MANUAL GROUND-PLANE POSITIONS AT 2.5 HZ. ROBOT PLACEMENTS ARE SIMULATED.
<table><tr><td colspan="3">Duration</td><td colspan="4">Eval. pe</td></tr><tr><td>Scene</td><td>(s)</td><td>People</td><td>Annot.</td><td>Robots</td><td>ople</td><td>Det. recall</td></tr><tr><td>ETH</td><td>773</td><td>360</td><td>8908</td><td>130</td><td></td><td>not run†</td></tr><tr><td>Hotel</td><td>722</td><td>390</td><td>6544</td><td>78</td><td>187</td><td>0.26</td></tr><tr><td>Zara01</td><td>360</td><td>148</td><td>5024</td><td>99</td><td>78</td><td>0.43</td></tr></table>

<sup>†</sup> YOLOv8m-Pose recall was 0.09 on a 6-frame probe of this near-nadir view;   
ETH is used only with ground-truth input.

## VI. EXPERIMENTAL RESULTS

## A. Detection on elevated views

On the evaluation halves, YOLOv8m-Pose at 1280 px recovered 0.43 of annotated pedestrians in Zara01 and 0.26 in Hotel (Table I; calibration halves 0.47 and 0.28). A preliminary probe on four frames per scene found at most five detections per model and scene at the default 640 px input, and on a sixframe probe at 1280 px YOLOv8m-Pose reached a recall of 0.09 in the near-nadir ETH view, so ETH enters only the ground-truth experiment (probe outputs in the replication package). Fig. 2 shows typical frames. Detection misses, not localisation, are therefore the first-order perception error in these views, and Section VII quantifies their effect on decisions.

X ankle midpoint (M1 input)  
Zara01,frame 5461: 20 annotated,12 detected  
![](images/5dd9c5ee8a119e778087575c7bb1f292308b3cec3a91633bec603d66b8e246f2.jpg)  
—YOLOv8m-Pose box

Hotel, frame16171: 18 annotated, 8 detected  
![](images/964db34c03ddd566c0d4c0f6bee1bef983deee1fda42747e3fba0cac750c38ed.jpg)  
Fig. 2. Evaluation-half frames from the two RGB scenes with YOLOv8m-Pose detections, keypoints, ankle midpoints (input of M1), annotated positions, and the simulated robot grid. Source: ETH and UCY video via OpenTraj; detections computed in this study.

## B. Rule geometry on annotated positions

Fed with annotated positions, the lateral rule produced no missed danger in any of the 12 scene–threshold cells (73026 unsafe events in total), exactly as Proposition 1 requires (Table II). Its stops were nevertheless mostly unnecessary: precision ranged from 5.6% to 31.9%. At � = 0.5 m precision was 7.5% in ETH, 20.0% in Hotel and 18.8% in Zara01, against 7.5%, 21.4% and 20.9% predicted by the uniform-occupancy model of Section III. At the robot level the lateral rule kept a robot stopped while no pedestrian was within the threshold for 22–52% of the time (Fig. 3). Perception plays no part in these numbers.

TABLE II  
LATERAL RULE FED WITH ANNOTATED POSITIONS (ALL FRAMES). THE PLANAR RULE IS EXACT BY CONSTRUCTION (FNR = FPR = 0). BRACKETS: 95% TIME-BLOCK BOOTSTRAP INTERVALS.
<table><tr><td>Scene</td><td>τ (m)</td><td>Events</td><td>FNR</td><td>FPR (%)</td><td>Prec. (%)</td><td>Model (%)</td><td>Robot unn. (% )</td></tr><tr><td>ETH</td><td>0.30</td><td>5483</td><td>0.0</td><td>14.0 [13.7, 14.3]</td><td>5.6 [5.5, 5.7]</td><td>5.6</td><td>44.1</td></tr><tr><td></td><td>0.50</td><td>7661</td><td>0.0</td><td>18.4 [17.9, 18.8]</td><td>7.5 [7.3, 7.6]</td><td>7.5</td><td>48.2</td></tr><tr><td></td><td>0.75</td><td>10247</td><td>0.0</td><td>23.5 [23.0, 24.1]</td><td>9.8 [9.6, 10.0]</td><td>9.9</td><td>51.4</td></tr><tr><td></td><td>1.00</td><td>12842</td><td>0.0</td><td>28.6 [27.9, 29.3]</td><td>12.1 [11.8, 12.3]</td><td>12.2</td><td>52.4</td></tr><tr><td>Hotel</td><td>0.30</td><td>3147</td><td>0.0</td><td>7.7 [7.5, 7.9]</td><td>14.8 [14.1, 15.3]</td><td>16.1</td><td>27.4</td></tr><tr><td></td><td>0.50</td><td>4390</td><td>0.0</td><td>9.4 [9.2, 9.6]</td><td>20.0 [19.3, 20.6]</td><td>21.4</td><td>29.7</td></tr><tr><td></td><td>0.75</td><td>5977</td><td>0.0</td><td>11.9 [11.6, 12.3]</td><td>25.5 [24.6, 26.3]</td><td>28.1</td><td>31.4</td></tr><tr><td></td><td>1.00</td><td>7523</td><td>0.0</td><td>13.5 [13.1, 13.8]</td><td>31.9 [31.0, 32.6]</td><td>34.8</td><td>30.5</td></tr><tr><td>Zara01</td><td>0.30</td><td>2553</td><td>0.0</td><td>6.8 [6.7, 6.9]</td><td>14.0 [13.7, 14.3]</td><td>15.7</td><td>22.3</td></tr><tr><td></td><td>0.50</td><td>3401</td><td>0.0</td><td>8.5 [8.4, 8.6]</td><td>18.8 [18.4, 19.2]</td><td>20.9</td><td>24.8</td></tr><tr><td></td><td>0.75</td><td>4430</td><td>0.0</td><td>10.6 [10.5, 10.7]</td><td>24.3 [23.9, 24.7]</td><td>27.5</td><td>26.6</td></tr><tr><td></td><td>1.00</td><td>5372</td><td>0.0</td><td>12.3 [12.1, 12.5]</td><td>29.6 [29.0, 30.1]</td><td>34.0</td><td>27.1</td></tr></table>

(a) Pair level, annotated positions  
![](images/d4f8b6e9ab6700207a5a071ee58197348d1bd88c914b57c5d496799ef0f73787.jpg)

(b) Robot level, annotated positions  
![](images/862d1347e2e89a2e04e83f8f9f5413ab0ebbc7df17f25eb4e6aa5584737dc761.jpg)  
Fig. 3. Lateral rule on annotated positions (no perception). (a) Precision of stops, with 95% time-block bootstrap intervals (narrower than the markers) and the uniform-occupancy prediction $\pi ( \bar { \tau } + R ) / ( 2 D )$ (dashed; for ETH it coincides with the measured line). FNR is zero in every case. (b) Share of robot-time spent stopped while no annotated pedestrian was within �.

## C. Localisation

Table III and Fig. 4 compare the localisation methods on pedestrians unseen during calibration. The calibrated groundplane method M1 had the lowest planar error in both scenes (Zara01 0.59 m, Hotel 0.51 m), against 1.72 m and 1.39 m for the reconstructed geometric baseline M0 and 0.81 m and 0.68 m for Depth Anything V2 with a calibrated depth map (M2). Lateral errors were similar across methods (0.20–0.22 m in Zara01, 0.24–0.26 m in Hotel); the methods differ in depth. M0 also produced estimates for fewer samples (26% of annotated evaluation samples in Zara01 versus 40% for M1), because it needs a confident nose and both knees. The M1 reference point chosen on the calibration half was the ankle midpoint in Zara01 and the nose in Hotel: Hotel annotations mark heads, and its dataset positions are those head pixels mapped through the ground homography, so mapping the nose the same way is the annotation-consistent choice. It inherits the parallax of that convention, which is part of the ground-truth limitation discussed in Section IX. Requiring a confident nose also lowered M1’s coverage in Hotel (16% of annotated evaluation samples).

TABLE III  
LOCALISATION ERROR ON THE EVALUATION HALF (MATCHED DETECTIONS; PEDESTRIANS DISJOINT FROM CALIBRATION). BRACKETS: 95% BOOTSTRAP OVER PEDESTRIANS. COVERAGE: SHARE OF ANNOTATED EVALUATION SAMPLES WITH AN ESTIMATE.
<table><tr><td>Scene</td><td>Method</td><td>Coverage (%)</td><td>n</td><td>Lateral RMSE (m)</td><td>Depth RMSE (m)</td><td>Depth bias (m)</td><td>Planar RMSE (m)</td></tr><tr><td>Zara01</td><td>MO geometric</td><td>25.8</td><td>667</td><td>0.20 [0.17, 0.23]</td><td>1.71 [1.44, 2.03]</td><td>0.69</td><td>1.72 [1.46, 2.04]</td></tr><tr><td></td><td>M1 ground plane</td><td>40.1</td><td>1039</td><td>0.22 [0.20, 0.25]</td><td>0.55 [0.48, 0.62]</td><td>-0.29</td><td>0.59 [0.52, 0.66]</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>40.1</td><td>1039</td><td>0.20 [0.18, 0.23]</td><td>0.78 [0.69, 0.87]</td><td>0.28</td><td>0.81 [0.72, 0.90]</td></tr><tr><td>Hotel</td><td>MO geometric</td><td>11.7</td><td>398</td><td>0.25 [0.20, 0.29]</td><td>1.37 [1.19, 1.53]</td><td>0.12</td><td>1.39 [1.21, 1.55]</td></tr><tr><td></td><td>M1 ground plane</td><td>15.8</td><td>537</td><td>0.26 [0.23, 0.30]</td><td>0.44 [0.38, 0.49]</td><td>0.20</td><td>0.51 [0.46, 0.57]</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>16.5</td><td>564</td><td>0.24 [0.19, 0.28]</td><td>0.64 [0.59, 0.68]</td><td>0.12</td><td>0.68 [0.63, 0.72]</td></tr></table>

![](images/daaef9bca4404e47f564d638cb2816a356a9e2ec6cc68995cad3e5f7e42ec8cd.jpg)  
Fig. 4. Localisation RMSE on the evaluation half with 95% bootstrap intervals over pedestrians.

## D. Decisions

Table IV lists decision outcomes at � = 0.5 m with Kalmanfiltered input. With M1 input and the planar rule, the misseddanger rate was 56.7% (Zara01) and 77.5% (Hotel), with precision 64.5% and 63.8%. The lateral rule on the same input lowered FNR to 38.8% and 70.6% but cut precision to 17.4% and 16.8%. At the robot level, with M1 input in Zara01, the lateral rule left the robot stopped unnecessarily for 20.6% of the time with a robot-level FNR of 13.0%; the planar rule gave 1.1% and 40.6%, and the planar rule with a 2�̂ margin 11.7% and 15.1% (Hotel: 14.6/51.2, 1.1/71.4 and 14.0/46.7%).

TABLE IV  
DECISION OUTCOMES AT � = 0.5 M ON THE EVALUATION HALF, KALMAN-FILTERED INPUT, ALL ANNOTATED SAMPLES (NO ESTIMATE ⇒ PROCEED). BRACKETS:95% TIME-BLOCK BOOTSTRAP. UNN. STOP: SHARE OF PAIR-TIME STOPPED WHILE SAFE.
<table><tr><td>Scene</td><td>Input</td><td>Rule</td><td>TP</td><td>FN</td><td>FP</td><td>FNR (%)</td><td>FPR (%)</td><td>Prec. ( %)</td><td>Missed events</td><td>Unn. st op (%)</td></tr><tr><td>Zara01</td><td>GT</td><td>lateral</td><td>4981</td><td>0</td><td>21267</td><td>0.0 [0.0, 0.0]</td><td>8.46</td><td>19.0</td><td>0/1778</td><td>8.29</td></tr><tr><td></td><td>GT</td><td>planar</td><td>4981</td><td>0</td><td>0</td><td>0.0 [0.0, 0.0]</td><td>0.00</td><td>100.0</td><td>0/1778</td><td>0.00</td></tr><tr><td></td><td>M0 geometric</td><td>lateral</td><td>2274</td><td>2707</td><td>10728</td><td>54.3 [49.7, 58.3]</td><td>4.27</td><td>17.5</td><td>767/1778</td><td>4.18</td></tr><tr><td></td><td>M0 geometric</td><td>planar</td><td>483</td><td>4498</td><td>1566</td><td>90.3 [87.7, 92.4]</td><td>0.62</td><td>23.6</td><td>1550/1778</td><td>0.61</td></tr><tr><td></td><td>MO geometric</td><td>planar +1ô</td><td>1720</td><td>3261</td><td>12521</td><td>65.5 [60.1, 70.0]</td><td>4.98</td><td>12.1</td><td>956/1778</td><td>4.88</td></tr><tr><td></td><td>M0 geometric</td><td>planar +2ô</td><td>2417</td><td>2564</td><td>31449</td><td>51.5 [45.9, 56.0]</td><td>12.51</td><td>7.1</td><td>695/1778</td><td>12.27</td></tr><tr><td></td><td>M1 ground plane</td><td>lateral</td><td>3046</td><td>1935</td><td>14448</td><td>38.8 [35.1, 41.7]</td><td>5.75</td><td>17.4</td><td>503/1778</td><td>5.63</td></tr><tr><td></td><td>M1 ground plane M1 ground plane</td><td>planar</td><td>2157</td><td>2824</td><td>1187</td><td>56.7 [51.5, 61.1]</td><td>0.47</td><td>64.5</td><td>806/1778</td><td>0.46</td></tr><tr><td></td><td></td><td>planar +1σ</td><td>2826</td><td>2155</td><td>3947</td><td>43.3 [38.5, 47.1]</td><td>1.57</td><td>41.7</td><td>589/1778</td><td>1.54</td></tr><tr><td></td><td>M1 ground plane</td><td>planar +2ô</td><td>3092</td><td>1889</td><td>8226</td><td>37.9 [33.6, 41.3]</td><td>3.27</td><td>27.3</td><td>508/1778</td><td>3.21</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>lateral</td><td>3051</td><td>1930</td><td>14290</td><td>38.7 [35.1, 41.6]</td><td>5.68</td><td>17.6</td><td>500/1778</td><td>5.57</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>planar</td><td>1720</td><td>3261</td><td>1583</td><td>65.5 [62.6, 68.6]</td><td>0.63</td><td>52.1</td><td>984/1778</td><td>0.62</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>planar +1ô</td><td>2665</td><td>2316</td><td>5299</td><td>46.5 [42.8, 50.2]</td><td>2.11</td><td>33.5</td><td>614/1778</td><td>2.07</td></tr><tr><td>Hotel</td><td>M2 DA-V2 + calib.</td><td>planar +2ô</td><td>3093</td><td>1888</td><td>11326</td><td>37.9 [34.2, 41.0]</td><td>4.50</td><td>21.5</td><td>495/1778</td><td>4.42</td></tr><tr><td></td><td>GT</td><td>lateral</td><td>6136</td><td>0</td><td>24266</td><td>0.0 [0.0, 0.0]</td><td>9.34</td><td>20.2</td><td>0/2470</td><td>9.13</td></tr><tr><td></td><td>GT</td><td>planar</td><td>6136</td><td>0</td><td>0</td><td>0.0 [0.0, 0.0]</td><td>0.00</td><td>100.0</td><td>0/2470</td><td>0.00</td></tr><tr><td></td><td>M0 geometric</td><td>lateral</td><td>1354</td><td>4782</td><td>7118</td><td>77.9 [74.7, 80.6]</td><td>2.74</td><td>16.0</td><td>1766/2470</td><td>2.68</td></tr><tr><td></td><td>M0 geometric</td><td>planar</td><td>580</td><td>5556</td><td>1518</td><td>90.5 [88.4, 92.2]</td><td>0.58</td><td>27.6</td><td>2110/2470</td><td>0.57</td></tr><tr><td></td><td>M0 geometric</td><td>planar +1ô</td><td>1327</td><td>4809</td><td>8261</td><td>78.4 [74.7, 81.2]</td><td>3.18</td><td>13.8</td><td>1696/2470</td><td>3.11</td></tr><tr><td></td><td>MO geometric</td><td>planar +2ô</td><td>1656</td><td>4480</td><td>20399</td><td>73.0 [69.5, 76.0]</td><td>7.85</td><td>7.5</td><td>1568/2470</td><td>7.67</td></tr><tr><td></td><td>M1 ground plane</td><td>lateral</td><td>1805</td><td>4331</td><td>8971</td><td>70.6 [67.3, 73.1]</td><td>3.45</td><td>16.8</td><td>1530/2470</td><td>3.37</td></tr><tr><td></td><td>M1 ground plane</td><td>planar</td><td>1380</td><td>4756</td><td>783</td><td>77.5 [74.3, 80.0]</td><td>0.30</td><td>63.8</td><td>1764/2470</td><td>0.29</td></tr><tr><td></td><td>M1 ground plane</td><td>planar +1ô</td><td>2002</td><td>4134</td><td>3755</td><td>67.4 [63.9, 70.2]</td><td>1.45</td><td>34.8</td><td>1407/2470</td><td>1.41</td></tr><tr><td></td><td>M1 ground plane</td><td>planar +2ô</td><td>2137</td><td>3999</td><td>8765</td><td>65.2 [61.8, 68.1]</td><td>3.37</td><td>19.6</td><td>1337/2470</td><td>3.30</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>lateral</td><td>1807</td><td>4329</td><td>8903</td><td>70.6 [66.5, 74.0]</td><td>3.43</td><td>16.9</td><td>1534/2470</td><td>3.35</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>planar</td><td>1108</td><td>5028</td><td>1005</td><td>81.9 [79.7, 84.0]</td><td>0.39</td><td>52.4</td><td>1877/2470</td><td>0.38</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>planar +1ô</td><td>1935</td><td>4201</td><td>4896</td><td>68.5 [64.8, 71.7]</td><td>1.88</td><td>28.3</td><td>1469/2470</td><td>1.84</td></tr><tr><td></td><td>M2 DA-V2 + calib.</td><td>planar +2ô</td><td>2124</td><td>4012</td><td>11558</td><td>65.4 [61.1, 69.2]</td><td>4.45</td><td>15.5</td><td>1385/2470</td><td>4.35</td></tr></table>

## E. Prediction

With M1 input, the Kalman filter’s planar prediction RMSE grew from 0.64 m at 0.4 s to 1.46 m at 2.0 s in Zara01 and from 0.62 to 2.06 m in Hotel (Table V, Fig. 5). It was the best of the three measurement-based predictors at every horizon. Raw constant-velocity extrapolation was worse than holding the last position at every horizon in Zara01 and only slightly better than it beyond 0.4 s in Hotel. Constant-velocity extrapolation of the annotations themselves stayed at 0.05–0.39 m (Zara01) and 0.12–0.50 m (Hotel). Prediction error here is dominated by measurement error rather than by the motion model, which argues against a learned predictor: it would be trained on a few hundred short tracks and could not remove measurement noise that the filter already absorbs. We therefore did not train one.

TABLE V  
PLANAR PREDICTION ERROR (M, RMSE) FROM M1 INPUT ON THE EVALUATION HALF. “CV ON GT” EXTRAPOLATES THE ANNOTATIONS THEMSELVES.
<table><tr><td>Scene</td><td>Predictor</td><td>0.4s</td><td>0.8s</td><td>1.2s</td><td>2.0s</td></tr><tr><td rowspan="4">Zara01</td><td>Kalman CV</td><td>0.64</td><td>0.80</td><td>1.00</td><td>1.46</td></tr><tr><td>raw CV</td><td>0.88</td><td>1.29</td><td>1.74</td><td>2.63</td></tr><tr><td>const. position</td><td>0.74</td><td>1.08</td><td>1.49</td><td>2.36</td></tr><tr><td>CV on GT</td><td>0.05</td><td>0.12</td><td>0.20</td><td>0.39</td></tr></table>

<table><tr><td>Scene</td><td>Predictor</td><td>0.4s</td><td>0.8s</td><td>1.2s</td><td>2.0s</td></tr><tr><td rowspan="5">Hotel</td><td>n (KF)</td><td>1011</td><td>990</td><td>962</td><td>904</td></tr><tr><td>Kalman CV</td><td>0.62</td><td>0.92</td><td>1.28</td><td>2.06</td></tr><tr><td>raw CV</td><td>0.70</td><td>0.97</td><td>1.33</td><td>2.12</td></tr><tr><td>const. position</td><td>0.68</td><td>1.12</td><td>1.62</td><td>2.65</td></tr><tr><td>CV on GT</td><td>0.12</td><td>0.21</td><td>0.29</td><td>0.50</td></tr><tr><td></td><td>n (KF)</td><td>482</td><td>459</td><td>429</td><td>369</td></tr></table>

![](images/997fa02cb1c533309caffe00d3ad416bfc7e7b515a7c8d485cf60d1b9ad229a2.jpg)

![](images/b9127fa18aa9d687fd097bf4138a5e5eae66549b137489430e48cba2e5ded2c4.jpg)  
Fig. 5. Planar prediction RMSE versus horizon (M1 input, evaluation half). “CV on ground truth” bounds what a constant-velocity model could achieve with perfect measurements.

## F. Computational cost

On a 2-core CPU without GPU, the NumPy YOLOv8m-Pose port took 4.3 s per 720×576 frame at 1280 px and Depth Anything V2-Small (ONNX Runtime, 518×518) 0.9 s; localisation, filtering and decisions are negligible. These timings reflect the constrained environment, not the models achievable speed on robot hardware.

## VII. DECISION-LEVEL ERROR ANALYSIS

Table VI and Fig. 6 decompose the deployed error (RGB input, Kalman filter, lateral rule) into rule and perception shares. Three patterns hold for every method in both scenes.

The rule trades misses for stops. The rule share of FNR is negative and the rule share of FPR positive: replacing the planar rule by the lateral one hides misses by stopping more often. With M1 in Zara01 the rule share was -8.9 pp of FNR and +6.87 pp of FPR.

(a) Missed-danger rate (FNR)  
![](images/37c9471f48f0cf0a1bdddd7d0061f65d8678ac174b9cd6d31bfea96bacde07f5.jpg)

Perception drives missed danger. The perception share of FNR was 47.8 pp for M1 in Zara01 (74.0 pp in Hotel). Under the planar rule, 33.0 pp of the 56.7% FNR in Zara01 came from samples with no estimate and 23.7 pp from wrong estimates (Hotel: 64.2 and 13.3 pp of 77.5%). Restricted to samples with an estimate, M1’s FNR fell to 35.4% (planar) and 8.7% (lateral) in Zara01.

Perception slightly lowers FPR, for the wrong reason. The perception share of FPR is negative because missed pedestrians cannot trigger stops. It is not evidence of a well-calibrated estimator.

(b) Unnecessary-stop rate (FPR)  
![](images/cebe8364054a8c77918cbe8de4c1c58a709aff136eaeb153df2de7be59ca7a7e.jpg)  
total = deployed error (lateral rule, RGB + KF)  
Fig. 6. Shapley attribution of the deployed error at $\tau = 0 . 5$ m (all annotated evaluation samples). Bars: rule and perception shares (they sum to the total, the deployed system’s error rate).

TABLE VI  
TWO-FACTOR SHAPLEY ATTRIBUTION AT τ = 0.5 M (KALMAN-FILTERED INPUT). IN PERCENTAGE POINTS. “ALL.": EVERY ANNOTATED SAMPLE (MISSES COUNT AS PERCEPTION ERROR). “DETECTED”: SAMPLES WITH AN ESTIMATE ONLY. BRACKETS: 95% TIME-BLOCK BOOTSTRAP.
<table><tr><td></td><td></td><td></td><td></td><td colspan="2">FNR (pp)</td><td colspan="3">FPR (pp)</td><td colspan="2">FNR planar split (pp)</td></tr><tr><td>Scene</td><td>Method</td><td>Samples</td><td>Total</td><td> $\phi _ { \mathrm { r u l e } }$ </td><td> $\phi _ { \mathrm { p e r c } }$ </td><td>Total</td><td> $\phi _ { \mathrm { r u l e } }$ </td><td> $\phi _ { \mathrm { p e r c } }$ </td><td>No est.</td><td>Loc.</td></tr><tr><td>Zara01</td><td>MO geometric</td><td>all</td><td>54.3</td><td>-18.0 [-19.4, -16.8]</td><td>72.3 [68.9, 75.3]</td><td>4.27</td><td>6.05</td><td>-1.78</td><td>49.1</td><td>41.2</td></tr><tr><td rowspan="2"></td><td rowspan="2">M1 ground plane</td><td>detected</td><td>10.3</td><td>-35.3 [-37.3, -33.3]</td><td>45.6 [44.1, 47.2]</td><td>8.56</td><td>7.86</td><td>0.70</td><td rowspan="2">33.0</td><td rowspan="2">23.7</td></tr><tr><td>all</td><td>38.8</td><td>-8.9 [-10.0, -7.7]</td><td>47.8 [43.5, 51.6]</td><td>5.75</td><td>6.87</td><td>-1.12</td></tr><tr><td rowspan="2">M2 DA-V2 + calib.</td><td></td><td>detected</td><td>8.7</td><td>-13.3 [-15.4, -11.2]</td><td>22.0 [19.5, 24.3]</td><td>8.70</td><td>8.23</td><td>0.46</td><td rowspan="2">33.0</td><td rowspan="2">32.5</td></tr><tr><td>all</td><td></td><td>38.7</td><td>-13.4 [-14.9, -12.3]</td><td>52.1 [49.2, 54.9]</td><td>5.68</td><td>6.76</td><td>-1.07</td></tr><tr><td rowspan="2">Hotel</td><td>MO geometric</td><td>detected</td><td>8.6</td><td>-19.9 [-21.8, -18.6]</td><td>28.5 [26.7, 30.8]</td><td>8.60</td><td>8.07</td><td>0.53</td><td></td><td rowspan="2">18.1</td></tr><tr><td>all</td><td></td><td>77.9</td><td>-6.3 [-7.2, -5.6]</td><td>84.2 [81.4, 86.2]</td><td>2.74</td><td>5.75</td><td>-3.01</td><td>72.4</td></tr><tr><td rowspan="2"></td><td>M1 ground plane</td><td>detected</td><td>20.0 70.6</td><td>-22.9 [-25.1, -20.8]</td><td>42.9 [40.1, 45.3]</td><td>10.33</td><td>8.85</td><td>1.48</td><td></td><td rowspan="2">13.3</td></tr><tr><td>all</td><td></td><td></td><td>-3.5 [-3.9, -3.0]</td><td>74.0 [70.7, 76.5]</td><td>3.45</td><td>6.25</td><td>-2.79 0.73</td><td>64.2</td></tr><tr><td rowspan="2"></td><td>M2 DA-V2 + calib.</td><td>detected</td><td>17.9 70.6</td><td>-9.7 [-10.9, -8.4]</td><td>27.5 [24.3, 30.8]</td><td>10.09</td><td>9.36</td><td></td><td></td><td rowspan="2">17.3</td></tr><tr><td>all</td><td></td><td></td><td>-5.7 [-7.1, -4.6]</td><td>76.2 [73.1, 78.8]</td><td>3.43</td><td>6.19</td><td>-2.76 0.80</td><td>64.7</td></tr><tr><td></td><td></td><td>detected</td><td>16.7</td><td>-16.1 [-18.5, -14.1]</td><td>32.8 [30.2, 35.3]</td><td>9.86</td><td>9.06</td><td></td><td></td><td></td></tr></table>

Operating points. Fig. 7 plots FNR against FPR over the four thresholds. With the ground-plane localiser M1 (�̂ ≈ 0.30 m before filtering), the planar rule with a 2�̂ margin had an FNR no higher than the lateral rule in 8 of 8 scene–threshold cells and a lower FPR in 7 of 8; the exception was Hotel at � = 0.3 m. With M2 the comparison was mixed (6 and 4 of 8). With M0, whose calibration error is about 1.24 m (Zara01) and 0.98 m (Hotel), the margin lowered FNR in every cell but

![](images/8890e8046a0fac95239eb4cab97e53c0021fb4ae25b3582d559b3c2eda0396db.jpg)  
multiplied FPR by 2.4–3.3. An explicit margin is therefore an efficient way to buy conservatism only when the estimator’s uncertainty is small; with a poor depth estimate, no rule recovers what the sensor did not measure. At the robot level (Table IV text, Section VI), the 2�̂ margin with M1 cut unnecessary stop time by about 40% in Zara01 relative to the lateral rule at a slightly higher miss rate, and in Hotel it was marginally better on both.

![](images/e9fc384f5ec0536deb8c3e5e23f5f8181b5d6041ecd43108b7a6dcd56ecf02d8.jpg)  
Fig. 7. Missed-danger rate versus unnecessary-stop rate across thresholds � ∈ {0.3,0.5,0.75,1.0} m (pair level, Kalman-filtered input, evaluation half). Stars: annotated positions with the lateral rule.

## VIII. DISCUSSION AND PRACTICAL IMPLICATIONS

The earlier laboratory study [6] reported 0.0% false negatives and 65.0% false-positive frames for one participant. Proposition 1 and Table II show that a zero miss rate is the expected behaviour of a lateral-only rule whenever lateral localisation is accurate, and that a large false-stop burden follows from the geometry alone. The lateral rule is a conservative bound, not a sensor property. The laboratory values themselves are not re-derived here and are not comparable numerically: that study used one participant, an OptiTrack reference, a 120 Hz overhead camera, no robot radius and different frame denominators.

A single accuracy number hides three different failure sources with opposite effects on safety. We recommend that monocular safety pipelines report (i) FNR and FPR (or precision) with their denominators and intervals over time blocks, (ii) event-level missed dangers, (iii) the share of FNR caused by missing estimates, and (iv) the same metrics with the decision rule fed by ground truth, which separates rule design from perception.

In these views a calibrated ground-plane mapping of a body keypoint beat both the reconstructed anthropometric depth and a calibrated learned relative-depth model, and it needs no fitted parameters beyond the scene homography. The largest remaining safety lever was detection coverage: a stop rule cannot react to a person the detector never sees. For a good localiser, an uncertainty margin is a better way to buy conservatism than discarding depth, because the margin is explicit, tunable, and shrinks as the estimator improves; for a poor one it mostly adds false stops.

## IX. LIMITATIONS

Ground truth. ETH and UCY positions are manual annotations at 2.5 Hz projected through homographies, not motion capture. Their own error (annotation point, timing, homography) is unquantified and is included in every error we report; Fig. 2 shows visible offsets between some annotations and feet. Scenes and subjects. Two scenes were usable with the full RGB pipeline, and one more with annotated input. They are outdoor pedestrian areas viewed from above, not robot viewpoints. Pedestrians differ between calibration and evaluation, but cameras do not; a cross-camera test was not possible for M0 and M2, whose maps are scene-specific. Simulated robots. Robots are stationary discs on a grid; no robot was present, moving, or reacting. Results are offline classifications of recorded motion and do not show collision avoidance. Unannotated people. ETH and UCY skip some pedestrians near image borders; we therefore score only annotated pedestrians and associate detections to them with ground truth, which excludes false detections of unannotated people and makes tracking oracle-associated. Detector configuration. We used COCO-trained YOLOv8m-Pose without fine-tuning; detectors trained for top-down views would change the detection share

## X. CONCLUSION

Evaluating monocular RGB human tracking through the stop/proceed decisions it drives exposes failure modes that accuracy metrics hide. A lateral-only distance rule never misses a danger when fed accurate positions, but most of its stops are unnecessary, by geometry; on real video, missed detections then dominate the missed dangers. A two-factor attribution separates these effects with the same data, and, for a groundplane localiser, an explicit uncertainty margin on planar clearance offers a better trade-off than discarding depth. The next step is to repeat the protocol with motion-capture ground truth from multiple calibrated views and with a real robot viewpoint.

## REFERENCES

[1] P. T. Singamaneni, P. Bachiller-Burgos, L. J. Manso, A. Garrell, A. Sanfeliu, A. Spalanzani, and R. Alami, “A survey on socially aware robot navigation: Taxonomy and future challenges,” The International Journal of Robotics Research, vol. 43, no. 10, pp. 1533–1572, 2024.

[2] C. Mavrogiannis, F. Baldini, A. Wang, D. Zhao, P. Trautman, A. Steinfeld, and J. Oh, “Core challenges of social robot navigation: A survey,” ACM Transactions on Human-Robot Interaction, vol. 12, no. 3, pp. 1–39, 2023.

[3] K. Kodur and M. Kyrarini, “Patient–robot co-navigation of crowded hospital environments,” Applied Sciences, vol. 13, no. 7, p. 4576, 2023.

[4] J. A. Marvel and R. Norcross, “Implementing speed and separation monitoring in collaborative robot workcells,” Robotics and Computer-Integrated Manufacturing, vol. 44, pp. 144–155, 2017.

[5] V. Villani, F. Pini, F. Leali, and C. Secchi, “Survey on human–robot collaboration in industrial settings: Safety, intuitive interfaces and applications,” Mechatronics, vol. 55, pp. 248–266, 2018.

[6] A. Taiwo and C. C. Ekechi, “Decision-level safety evaluation of monocular RGB human motion tracking for robotic navigation.” engrXiv preprint (submission 8166), 2026.

[7] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything V2,” in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2406.09414.

[8] A. Eirale, M. Martini, and M. Chiaberge, “Human following and guidance by autonomous mobile robots: A comprehensive review,” IEEE Access, 2025.

[9] L. Bruckschen, K. Bungert, N. Dengler, and M. Bennewitz, “Humanaware robot navigation by long-term movement prediction,” in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), pp. 11032– 11037, 2020.

[10] Z. Zhang, G. Hess, J. Hu, E. Dean, L. Svensson, and K. Åkesson, “Future-oriented navigation: Dynamic obstacle avoidance with one-shot energy-based multimodal motion prediction,” IEEE Robotics and Automation Letters, vol. 10, no. 8, pp. 8043–8050, 2025.

[11] L. M. Amaya-Mejı́a, N. Duque-Suárez, D. Jaramillo-Ramı́rez, and C. Martinez, “Vision-based safety system for barrierless human-robot collaboration.” arXiv:2208.02010, 2022.

[12] Y. Ding and X. Luo, “Monocular 3D object detection for proximity prediction in human-machine collision warning systems on construction sites.” arXiv:2305.17931, 2023.

[13] L. Bertoni, S. Kreiss, and A. Alahi, “MonoLoco: Monocular 3D pedestrian localization and uncertainty estimation,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), pp. 6860–6870, 2019.

[14] D. Hoiem, A. A. Efros, and M. Hebert, “Putting objects in perspective,” International Journal of Computer Vision, vol. 80, no. 1, pp. 3–15, 2008.

[15] Z. Machkour, D. Ortiz-Arroyo, and P. Durdevic, “Monocular based navigation system for autonomous ground robots using multiple deep learning models,” International Journal of Computational Intelligence Systems, vol. 16, p. 79, 2023.

[16] E. A. Rodrı́guez-Martınez, ́ W. Flores-Fuentes, F. Achakir, O. Sergiyenko, and F. N. Murrieta-Rico, “Vision-based navigation and perception for autonomous robots: Sensors, SLAM, control strategies, and cross-domain applications—a review,” Eng, vol. 6, no. 7, p. 153, 2025.

[17] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8.” https://github.com/ultralytics/ultralytics, checkpoints from release v8.3.0 assets, 2023.

[18] D. Maji, S. Nagori, M. Mathew, and D. Poddar, “YOLO-Pose: Enhancing YOLO for multi person pose estimation using object keypoint similarity loss,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 2636–2645, 2022.

[19] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Computer Vision – ECCV 2014, LNCS, pp. 740–755, 2014.

[20] J. Philion, A. Kar, and S. Fidler, “Learning to evaluate perception models using planner-centric metrics,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020.

[21] B. Ivanovic and M. Pavone, “Injecting planning-awareness into prediction and detection evaluation,” in Proc. IEEE Intelligent Vehicles Symposium (IV), pp. 821–828, 2022.

[22] L. Sigal, A. O. Balan, and M. J. Black, “HumanEva: Synchronized video and motion capture dataset and baseline algorithm for evaluation of articulated human motion,” International Journal of Computer Vision, vol. 87, no. 1–2, pp. 4–27, 2010.

[23] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu, “Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 7, pp. 1325–1339, 2014.

[24] B. Castillo, C. Riascos, J. M. Franco, J. Marulanda, et al., “Assessing spatiotemporal behavior of human gait: A comparative study between low-cost smartphone-based mocap and OptiTrack systems,” Experimental Techniques, vol. 49, no. 1, pp. 3–13, 2025.

[25] A. Bajrami, G. Beraldo, M. C. Palpacelli, T. Heikkilä, and G. Cortelessa, “Markerless upper limb motion tracking: A comparative evaluation of multi-view approaches,” in Proc. IEEE Int. Conf. Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE), pp. 818–823, 2025.

[26] H. Joo, T. Simon, X. Li, H. Liu, L. Tan, L. Gui, S. Banerjee, T. Godisart, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, and Y. Sheikh, “Panoptic studio: A massively multiview system for social interaction capture,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 1, pp. 190–204, 2019.

[27] R. Martıń -Martın, M. ́ Patel, H. Rezatofighi, A. Shenoi, J. Gwak, E. Frankel, A. Sadeghian, and S. Savarese, “JRDB: A dataset and benchmark of egocentric robot visual perception of humans in built environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 6748–6765, 2023.

[28] S. Pellegrini, A. Ess, K. Schindler, and L. van Gool, “You’ll never walk alone: Modeling social behavior for multi-target tracking,” in Proc. IEEE 12th Int. Conf. Computer Vision (ICCV), pp. 261–268, 2009.

[29] A. Lerner, Y. Chrysanthou, and D. Lischinski, “Crowds by example,” Computer Graphics Forum, vol. 26, no. 3, pp. 655–664, 2007.

[30] J. Amirian, B. Zhang, F. V. Castro, J. J. Baldelomar, J.-B. Hayet, and J. Pettré, “OpenTraj: Assessing prediction complexity in human trajectories datasets,” in Computer Vision – ACCV 2020, vol. 12627 of LNCS, 2021.

[31] R. E. Kalman, “A new approach to linear filtering and prediction problems,” Journal of Basic Engineering, vol. 82, no. 1, pp. 35–45, 1960.

[32] C. Schöller, V. Aravantinos, F. Lay, and A. Knoll, “What the constant velocity model can teach us about pedestrian motion prediction,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1696–1703, 2020.

[33] H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” The Annals of Statistics, vol. 17, no. 3, 1989.