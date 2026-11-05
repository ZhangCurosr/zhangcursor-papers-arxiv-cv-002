# Occlusion for Free: Detecting Occlusion from the Disagreement Between Two Monocular Distance Cues

Chibueze Praise Onyenweaku<sup>1</sup> Toyibat Adele<sup>2,∗</sup> Sikirulahi Opeyemi Abdulkareem<sup>2,∗</sup> Obed Honour Eje<sup>3,∗</sup>

<sup>1</sup>University of Lagos <sup>2</sup>ML Collective <sup>3</sup>Federal University of Technology, Minna

## Abstract

Knowing that a pedestrian is occluded is useful to a robot even when the detector cannot recover the hidden pixels, but occlusion estimators are normally trained on occlusion annotations that deployment settings rarely have. We show that occlusion can be read off two distance estimates a camera-only pipeline already computes. Apparent height is largely unchanged by occlusion, because the detected box still spans the whole person; monocular metric depth is pulled toward the occluder, because the nearest surface inside the box is what a depth map reports. Only their ordering matters, so no focal length, stature, or depth scale is needed, provided the depth model’s error is close to a pure gain, a condition that can be checked without labels. The score detects heavy occlusion on KITTI with AUC 0.909 (0.888 over all annotated pedestrians, including those the detector misses), requires no training and no occlusion labels, and is not a proxy for range: within fixed distance bands it holds AUC 0.894–0.925 while true distance alone sits near chance. On synthetic CARLA scenes with measured visibility it reaches 0.819, against 0.644 for distance alone. We arrived here after measuring three standard architectural responses to occlusion, depth fusion, occlusion-simulating augmentation, and attention, none of which improved detection outside its training distribution in the configurations we tested, and after finding that a detector trained on German roads loses half its accuracy on self-annotated street scenes in Lagos, Nigeria. Signals that need no local annotation are worth more in settings that benchmarks do not represent.

Keywords: Occlusion Estimation; Monocular Depth; Object Detection; Dataset Bias; Robotics

## 1 Introduction

Robots need to know what is around them, and a camera is the cheapest way to sense it. People and objects are often only partly visible, and this occlusion is the dominant failure condition for pedestrian detection in cluttered environments [1]. A robot does not always need to recover the hidden pixels. Often it is enough to know that a detection is occluded, so it can widen an uncertainty estimate or shorten a planning horizon.

Most prior work addresses occlusion by changing the detector, fusing separately estimated depth with the RGB image [2–4], simulating occlusion during training by masking parts of objects [5–8], or adding attention that re-weights features toward visible parts [9,10]. Designing for a low-cost robot, we measured all three; none improved detection outside its training distribution in the configurations we tested, and depth fusion hurt accuracy even in-distribution, matching a known limit of monocular depth models [11,12]. A detector trained on this benchmark data also lost about half its accuracy on 350 self-annotated Lagos frames.

Our depth branch failed as a detection input, but that failure is what our approach is built on. A depth map reports how far away the nearest visible surface is at each point. Inside an occluded pedestrian’s box, that nearest surface is usually the occluder, not the person behind it, so the depth value read is pulled toward the occluder’s distance. Apparent height is unaffected, since the box still spans the whole person. The two cues move in opposite directions under occlusion, so their disagreement is a direct readout of it, computed from two numbers the pipeline already produces. Contributions. A training-free, label-free occlusion signal from this disagreement, validated with a distance-band check that rules out range as a confound and with tests of each link in its mechanism; and a controlled comparison of depth fusion, augmentation, and attention across 81 checkpoints, together with a self-collected Lagos evaluation set measuring the detector transfer gap that motivates a training-free signal.

## 2 Related Work

Depth without LiDAR. MV3D-style LiDAR fusion [2] remains the accuracy reference but is hard to justify on low-cost platforms. MiDaS [3] predicts only relative depth, with no fixed scale; Depth Anything V2 [4] recovers absolute scale, and we use its outdoor variant, fine-tuned on synthetic Virtual KITTI rather than LiDAR-supervised frames, so evaluating on KITTI leaks no measured depth. Monocular depth decoders also blur values across occlusion boundaries, since a convolution kernel mixes information across surfaces in its receptive field [11], and a concurrent study reports depth can make an RGB-D detector perform worse than RGB alone [12].

Occlusion robustness vs. occlusion estimation. Augmentation, single-region masking [5,6], multipatch hiding [7], grid erasure [8], and attention that reweights features toward occluded pedestrians [9,10] all try to make a detector perform better despite occlusion. Neither tells you whether a specific detection is actually occluded. Stereo methods detect occlusion directly from left–right disparity inconsistency [13]. MonoLoco [14] comes closest, but treats wider uncertainty on occluded cases as incidental rather than validated, and never checks whether the effect is occlusion or just distance. Bi-box regression also estimates occlusion but needs paired visible and full-body labels [15]. We instead treat the disagreement between two distance estimates as the signal itself, training-free and label-free.

## 3 Method

## 3.1 System Overview

Figure 1 shows the full pipeline. An RGB frame enters a YOLOv8 detector [16,17] (YOLOv8x in our evaluation), producing bounding boxes for each detected pedestrian. Those boxes feed two parallel distance estimators, an apparent-height module and a Depth Anything V2 [4] branch reading the nearest surface inside each box. Their disagreement produces a scalar occlusion score passed to the planner. LiDAR ground truth is used only for evaluation; the runtime path is entirely camera-based. We arrived at this design after the three model-side responses to occlusion, measured across 81 checkpoints (Section 4.6), failed to transfer in the configurations tested, which turned the question toward why depth failed as a detector input.

## 3.2 Monocular Distance Cues

## 3.2.1 Apparent Height Estimation

The apparent-height distance estimate is obtained by inverting the pinhole camera relationship, $z _ { \mathrm { a p p } } = f _ { y } S / h _ { \mathrm { p x } }$ , where $f _ { y }$ is the vertical focal length of the camera, S is the effective stature of the pedestrian, and $h _ { \mathrm { p x } }$ is the height of the pedestrian’s bounding box in pixels. S = 1.83 m is the median implied stature of fully-visible annotated KITTI pedestrians; it cancels in the ranking score below.

KITTI boxes span the pedestrian’s full extent even when part of it is occluded, so $z _ { \mathrm { a p p } }$ is insensitive to occlusion as long as the box does not shrink toward the visible part. The score uses predicted boxes, so we test this on detector output: the implied stature from YOLOv8x boxes is 1.825, 1.811 and 1.813 m for fully-visible, partly and largely occluded pedestrians, though a tail of low-confidence occluded boxes is short (Appendix C).

![](images/66e74ae34c6b6535b400207d1000512c2d6ab3cac16f376d2b11565f9ebd91b1.jpg)  
Figure 1: Runtime pipeline. Offline inputs are fixed before deployment. At runtime, two monocular distance estimates per detection are compared; their disagreement flags occlusion without training or occlusion labels. LiDAR ground truth is used for evaluation only, and stature and focal length enter only metric read-outs, not the ranking score.

## 3.2.2 Monocular Metric Depth Estimation

The second distance cue is obtained using a frozen Depth Anything V2 outdoor metric depth model. For each detected pedestrian, $z _ { \mathrm { d e p } } = k \cdot d _ { \mathrm { m i n } } .$ , where $d _ { \mathrm { m i n } }$ is the minimum depth value in the box and k is a scale factor fitted on fully-visible instances, used only by the calibrated read-outs of Appendix B. Unlike apparent height, $z _ { \mathrm { d e p } }$ is sensitive to occlusion, since a closer occluder’s surface can dominate that minimum, pulling the estimate toward the camera. Low percentiles, robust to a single erroneous pixel, perform as well or better (Appendix C); our released implementation uses the 5th percentile.

## 3.3 Occlusion Score Formulation

Because the two cues respond differently to occlusion, apparent height stable, depth pulled toward the occluder, their relationship is itself an indicator of occlusion. The ratio between the depth-based and apparent-height distance estimates can be written as

$$
\frac { z _ { \mathrm { d e p } } } { z _ { \mathrm { a p p } } } = \underbrace { \frac { k } { f _ { y } S } } _ { G } d _ { \mathrm { m i n } } h _ { \mathrm { p x } } ,\tag{1}
$$

where all calibration-related terms are combined into a single positive gain G.

We only need to rank detections by likelihood of occlusion, and multiplying by a positive constant does not change a ranking, so G can be dropped. The deployable score is $d _ { \mathrm { m i n } } h _ { \mathrm { p x } }$ , requiring no focal length, stature, or depth rescaling factor; lower values indicate occlusion, since a closer occluder reduces the minimum depth inside the box.

This invariance is to a depth gain, not an $o f f s e t ,$ , and it requires metric depth. If the depth model reads $d _ { \operatorname* { m i n } } = a z + b$ , then for an unoccluded pedestrian, where $z _ { \mathrm { a p p } } \approx z ,$

$$
d _ { \operatorname* { m i n } } h _ { \mathrm { p x } } \approx f _ { y } S \left( a + \frac { b } { z _ { \mathrm { a p p } } } \right) .\tag{2}
$$

The gain a cancels in the ranking; the offset b adds a term that falls with distance, turning the score partly into a range measure. This is checkable without occlusion labels, because an offset makes the score correlate with distance among visible pedestrians: for Depth Anything $\nabla 2 ^ { \circ } \mathbf { s }$ outdoor checkpoint on KITTI, $\rho = - 0 . 0 7$ , and a synthetic +2 m offset moves it to −0.61 (Appendix B).

## 4 Experiments

## 4.1 Setup

We train YOLOv8n and YOLOv8x on KITTI [18] and CityPersons [19] using byte-identical settings and standard train/validation/test splits. Three seeds are used where confidence intervals are reported and one where marked, giving 81 evaluated checkpoints. Occlusion augmentation is applied only during training and only when more than half of the ground-truth pedestrian remains visible. We report mAP@0.5 by KITTI occlusion level.

KITTI labels pedestrians fully visible (0), partly (1) or largely occluded (2), or unknown (3); heavy is 2–3 and any is 1–3, each against visible. The score is evaluated on 1,941 detections from three YOLOv8x seeds, matched to 682 distinct test pedestrians by greedy IoU ≥ 0.5. AUC measures whether an occluded pedestrian is ranked above a fully-visible one; since a pedestrian contributes up to three rows, 95% intervals bootstrap pedestrians (2,000 resamples).

## 4.2 Results

Figure 2 shows the mechanism on one KITTI frame. For the visible pedestrian, apparent height and depth agree (10.5 m vs. 10.7 m). For the occluded pedestrian, the nearest surface belongs to the occluder, apparent height gives 20.8 m while depth gives 13.6 m. The resulting 34% disagreement reduces the score to 0.79 of the scene median.

![](images/c90ee9215053d0fe22943c165a798db19b62dd2929224be6016e5d546cb90947.jpg)  
Figure 2: One KITTI frame showing the proposed cue. The visible pedestrian’s nearest surface is their own and the two distance estimates agree; for the occluded pedestrian, depth reads the nearer occluder and the estimates diverge by 34%.

## 4.3 Occlusion Detection and Distance Control

The score reaches AUC 0.909 for heavy occlusion and 0.826 for any occlusion on matched detections, and 0.775 for partly occluded pedestrians alone. The detector misses 27% of largely occluded pedestrians (3% of visible ones), so matched detections favor the easier occluded cases; scoring all 709 annotated test pedestrians with annotated boxes, no detector involved, gives 0.888 [0.852, 0.921].

A distance confound is unlikely. Occluded pedestrians are farther away on average (16.9 m vs. 13.2 m), but within fixed distance bands true distance falls to near chance (AUC 0.457–0.594), while the proposed score remains at 0.894–0.925 for heavy occlusion (Table 1). The 5–10 m band rests on only 10 distinct heavily occluded pedestrians, so its interval is wide. The full breakdown, for any, partial and heavy occlusion, is in Appendix A.

Table 1: Occlusion score on KITTI test, heavy occlusion vs. fully-visible. N counts matched detections, with distinct pedestrians in parentheses; intervals bootstrap pedestrians. The final column is the AUC of true distance alone on the same instances.
<table><tr><td>Distance band</td><td>N vis/occ (peds)</td><td>AUC of score [95% CI]</td><td>AUC of distance</td></tr><tr><td>5-10 m</td><td>409/26 (137/10)</td><td>0.917 [0.843, 0.971]</td><td>0.594</td></tr><tr><td>10-20 m</td><td>466/140 (156/55)</td><td>0.894 [0.840, 0.939]</td><td>0.457</td></tr><tr><td>20m+</td><td>318/100 (112/41)</td><td>0.925 [0.874, 0.964]</td><td>0.469</td></tr><tr><td>all</td><td>1237/269 (420/107)</td><td>0.909 [0.876, 0.938]</td><td>0.624</td></tr></table>

## 4.4 Formulation, Depth Source and Mechanism

The calibration-free ranking performs identically to calibrated alternatives (0.909 vs. 0.908) and requires metric rather than ordinal depth: MiDaS’s relative output reverses the ranking (Appendix B).

The result survives each test of its mechanism we could construct (Appendix C): swapping predicted and annotated boxes for either cue moves heavy AUC by at most 0.010, low percentiles in place of the minimum give 0.910–0.921, and box shifts of up to 4 px give 0.904–0.907.

## 4.5 Beyond KITTI, and Runtime

On 573 synthetic CARLA frames (3,293 pedestrians), with visibility measured by re-rendering without occluders, the score reaches AUC 0.819 [0.801, 0.834] against 0.644 for distance alone, beating distance in every band (Appendix E). CARLA’s boxes are oracle and its depth checkpoint was chosen on CARLA ground-truth depth, so this tests the depth mechanism under a change of domain and camera height, not the full pipeline. The score itself costs about 0.3 ms per frame on a laptop CPU; the depth network dominates (Appendix F).

## 4.6 Model-Side Alternatives and Lagos Transfer

Table 6 (Appendix D) reports three model-side interventions. With three YOLOv8x seeds, augmentation improves hard KITTI cases by +0.041 mAP50 but not CityPersons, and MiDaS depth fusion reduces accuracy. The DAv2 fusion and POPAM rows are single YOLOv8n runs; they record what those configurations did, not that the interventions generally fail.

On 350 self-annotated Lagos frames (Table 7), a KITTI-trained detector drops from 0.8809 to 0.4482 mAP50, while the CityPersons-trained baseline reaches 0.7356, a +0.2873 difference on identical frames. The datasets differ in size, density, diversity, geography, camera and annotation protocol, so this shows only that the CityPersons-trained model transfers better. It evaluates the detector, not the score: the Lagos annotations have no occlusion labels.

## Acknowledgments and Disclosure of Funding

We thank OpenToken and CloudRift for providing the GPU compute used for this work.

## References

[1] Z. Ouardirhi, S. A. Mahmoudi, and M. Zbakh, “Enhancing Object Detection in Smart Video Surveillance: A Survey of Occlusion-Handling Approaches,” Electronics, vol. 13, no. 3, p. 541, 2024. https://doi.org/10. 3390/electronics13030541

[2] X. Chen, H. Ma, J. Wan, B. Li, and T. Xia, “Multi-View 3D Object Detection Network for Autonomous Driving,” in Proc. IEEE CVPR, 2017. https://arxiv.org/abs/1611.07759

[3] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,” IEEE TPAMI, vol. 44, no. 3, pp. 1623–1637, 2022. https://doi.org/10.1109/TPAMI.2020.3019967

[4] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” in NeurIPS, 2024. https://arxiv.org/abs/2406.09414

[5] T. DeVries and G. W. Taylor, “Improved Regularization of Convolutional Neural Networks with Cutout,” 2017. https://arxiv.org/abs/1708.04552

[6] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, “Random Erasing Data Augmentation,” in AAAI, vol. 34, no. 7, 2020, pp. 13001–13008. https://arxiv.org/abs/1708.04896

[7] K. K. Singh, H. Yu, A. Sarmasi, G. Pradeep, and Y. J. Lee, “Hide-and-Seek: A Data Augmentation Technique for Weakly-Supervised Localization and Beyond,” 2018. https://arxiv.org/abs/1811.02545

[8] P. Chen, S. Liu, H. Zhao, X. Wang, and J. Jia, “GridMask Data Augmentation,” 2020. https://arxiv. org/abs/2001.04086

[9] H. Liu, Z. Zhang, and B. Feng, “POP-YOLOv8: An Object Detection Framework for Partially Occluded Pedestrians in Nighttime Traffic Environments,” Scientific Reports, vol. 16, no. 1, p. 4841, 2026. https: //www.nature.com/articles/s41598-026-35146-9

[10] S. Tang, Y. Zhou, J. Li, C. Liu, and J. Shi, “Attention-Guided Sample-Based Feature Enhancement Network for Crowded Pedestrian Detection Using Vision Sensors,” Sensors, vol. 24, no. 19, art. 6350, 2024. https://doi.org/10.3390/s24196350

[11] M. Ramamonjisoa, Y. Du, and V. Lepetit, “Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation Using Displacement Fields,” in Proc. IEEE/CVF CVPR, 2020. https://arxiv. org/abs/2002.12730

[12] X. Wang, J. Hua, R. Li, Z. Wu, C. Chen, K. Gu, and A. Hao, “When Depth Hurts: Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection,” 2026. https://arxiv.org/abs/ 2609.03378

[13] G. Egnal and R. P. Wildes, “Detecting Binocular Half-Occlusions: Empirical Comparisons of Five Approaches,” IEEE TPAMI, vol. 24, no. 8, pp. 1127–1133, 2002. doi: 10.1109/TPAMI.2002.1023808

[14] L. Bertoni, S. Kreiss, and A. Alahi, “MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2019. https://arxiv. org/abs/1906.06059

[15] C. Zhou and J. Yuan, “Bi-box Regression for Pedestrian Detection and Occlusion Estimation,” in Proc. European Conference on Computer Vision (ECCV), 2018, pp. 135–151. doi: 10.1007/978-3-030-01246-5\_9

[16] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in Proc. IEEE CVPR, 2016, pp. 779–788. doi: 10.1109/CVPR.2016.91

[17] Ultralytics, “YOLOv8 Documentation, Performance Metrics,” 2024. https://docs.ultralytics.com/ models/yolov8/

[18] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” in Proc. IEEE CVPR, 2012, pp. 3354–3361. doi: 10.1109/CVPR.2012.6248074

[19] S. Zhang, R. Benenson, and B. Schiele, “CityPersons: A Diverse Dataset for Pedestrian Detection,” in Proc. IEEE CVPR, 2017, pp. 3213–3221. https://arxiv.org/abs/1702.05693

## A Full Distance-Band and Per-Level Breakdown

Table 2 gives the full distance-band breakdown that Table 1 summarizes, for any occlusion, for partly occluded pedestrians (level 1) alone, and for heavy occlusion. The score beats true distance in every band for every target. It is weakest for partial occlusion at close and middle range (0.728–0.735) and strongest for heavy occlusion. Computed separately for each detector seed, heavy and any AUC are 0.909 ± 0.007 and 0.826 ± 0.003.

Table 2: Occlusion score on KITTI test versus fully-visible pedestrians. N counts matched detections, with distinct pedestrians in parentheses; intervals bootstrap pedestrians. The final column is the AUC of true distance alone on the same instances.
<table><tr><td>Target</td><td>Band</td><td>N vis/occ (peds)</td><td>AUC of score [95% CI]</td><td>AUC of distance</td></tr><tr><td>Any (1–3)</td><td>5-10m</td><td>409/124 (137/44)</td><td>0.773 [0.676, 0.863]</td><td>0.549</td></tr><tr><td>Any (1–3)</td><td>10-20 m</td><td>466/297 (156/109)</td><td>0.807 [0.750, 0.860]</td><td>0.471</td></tr><tr><td>Any (1–3)</td><td>20m+</td><td>318/267 (112/102)</td><td>0.874 [0.828, 0.919]</td><td>0.513</td></tr><tr><td>Any (1–3)</td><td>all</td><td>1237/704 (420/262)</td><td>0.826 [0.792, 0.857]</td><td>0.600</td></tr><tr><td>Partly (1)</td><td>5-10m</td><td>409/98 (137/34)</td><td>0.735 [0.615, 0.840]</td><td>0.538</td></tr><tr><td>Partly (1)</td><td>10-20 m</td><td>466/157 (156/54)</td><td>0.728 [0.648, 0.809]</td><td>0.485</td></tr><tr><td>Partly (1)</td><td>20m+</td><td>318/167 (112/61)</td><td>0.844 [0.781, 0.900]</td><td>0.539</td></tr><tr><td>Partly (1)</td><td>all</td><td>1237/435 (420/155)</td><td>0.775 [0.728, 0.817]</td><td>0.585</td></tr><tr><td>Heavy (2–3)</td><td>5-10m</td><td>409/26 (137/10)</td><td>0.917 [0.843, 0.971]</td><td>0.594</td></tr><tr><td>Heavy (2–3)</td><td>10-20 m</td><td>466/140 (156/55)</td><td>0.894 [0.840, 0.939]</td><td>0.457</td></tr><tr><td>Heavy (2–3)</td><td>20m+</td><td>318/100 (112/41)</td><td>0.925 [0.874, 0.964]</td><td>0.469</td></tr><tr><td>Heavy (2–3)</td><td>all</td><td>1237/269 (420/107)</td><td>0.909 [0.876, 0.938]</td><td>0.624</td></tr></table>

## B Score Formulation, Depth Source, and Depth Offset

Table 3 compares three read-outs of the cue disagreement, referenced in Section 4.4. The ranking score and the signed ratio are monotone and perform identically, so removing $f _ { y } , S ,$ , and k costs nothing. The absolute difference performs worse and collapses once the depth scale is perturbed 25% high, confirming that ranking, not magnitude, is what survives calibration drift. Depth reads farther than apparent height for 44% of fully-visible pedestrians but only 3% of heavily occluded ones, so the direction of the disagreement is itself informative and taking its magnitude discards that.

Table 3: Occlusion AUC for three formulations of the cue disagreement. Heavy occlusion.
<table><tr><td>Read-out</td><td>Calib.</td><td>AUC</td></tr><tr><td>Ranking,  $d _ { \mathrm { m i n } } h _ { \mathrm { p x } }$ </td><td>none</td><td>0.909</td></tr><tr><td>Signed,  $1 - z _ { \mathrm { d e p } } / z _ { \mathrm { a p p } }$ </td><td> $f _ { y } , S , k$ </td><td>0.908</td></tr><tr><td>Absolute difference</td><td> $f _ { y } , S , k$ </td><td>0.872</td></tr><tr><td>Absolute, gain 25% high</td><td> $f _ { y } , S , k$ </td><td>0.470</td></tr></table>

Table 4 compares depth sources. MiDaS used as-is reverses the ordering, while inverting it recovers some performance but remains below Depth Anything V2. The signal therefore needs depth proportional to distance, not simply an arbitrary monocular depth ordering.

Table 4: Occlusion AUC by depth source.
<table><tr><td>Depth source</td><td>any</td><td>heavy</td></tr><tr><td>MiDaS, as-is</td><td>0.572</td><td>0.596</td></tr><tr><td>MiDaS, inverted</td><td>0.674</td><td>0.798</td></tr><tr><td>DAv2, metric</td><td>0.826</td><td>0.909</td></tr></table>

Depth offset. Eq. 2 predicts that an additive depth offset b turns the score partly into a range measure. Adding a synthetic offset to $d _ { \mathrm { m i n } }$ reproduces this. $\mathbf { A } \mathbf { t } \ b = 0$ the score is nearly uncorrelated with distance among visible pedestrians $( \rho = - 0 . 0 7 )$ . At b = +2 and +4 m, $\rho = - 0 . 6 1$ and −0.81, and the weakest within-band heavy AUC falls from 0.894 to 0.839 and 0.800. Overall AUC falls more slowly, to 0.876 at +4 m, which is why within-band AUC, not overall AUC, is the criterion to report. The correlation needs no occlusion labels, so it is a deployment check for any new depth checkpoint or camera mounting. The calibration-free property holds on KITTI because this checkpoint’s offset there is negligible. It is not a general guarantee: on CARLA (Appendix E) the depth checkpoint used has a substantial offset.

## C Mechanism Checks

A high AUC does not by itself show that the mechanism in the Introduction produces it. Table 5 tests each link on the same KITTI detections and depth maps.

Detector boxes. If occluded boxes shrank, $z _ { \mathrm { a p p } }$ would inflate and the score could read localization error. The median predicted-to-annotated box-height ratio is 0.99–1.00 at every occlusion level, but 15% of largely occluded predicted boxes are more than 10% short (1% of visible), mostly lowconfidence detections. Swapping either ingredient between predicted and annotated boxes changes heavy AUC by at most 0.010. Scoring all 709 annotated pedestrians with annotated boxes, no detector involved, gives 0.888. It is lower than 0.909 because the detector misses 27% of largely occluded pedestrians (10% of partly occluded, 3% of visible), so matched detections favor the easier occluded cases.

The minimum operator. Low percentiles perform as well as the minimum or better (0.910–0.921), and the signal fades only toward the median (0.859), where the occluder no longer dominates. Shrinking the box helps, growing it admits ground and background, and shifts of up to 4 px barely matter: the near part of the depth distribution carries the effect, not one pixel.

Other confounds. Excluding truncated pedestrians raises heavy AUC to 0.913. Stature varies with log-s.d. 0.064 among visible pedestrians, against a median log-score shift of 0.37 under heavy occlusion. The score is weaker on low-confidence detections (0.867 in the lowest confidence tercile, 0.903 in the middle; the top tercile has too few heavy cases to score). Pose is unlabeled in KITTI and untested.

Table 5: Mechanism checks, KITTI heavy occlusion vs. visible. Pedestrian-bootstrap 95% intervals.
<table><tr><td>Variant</td><td>Heavy AUC [95% CI]</td></tr><tr><td>Paper score,  $d _ { \mathrm { m i n } }$  in predicted box</td><td>0.909 [0.876, 0.938]</td></tr><tr><td>h annotated,  $d _ { \mathrm { m i n } }$  predicted box</td><td>0.899 [0.866, 0.928]</td></tr><tr><td>h predicted,  $d _ { \mathrm { m i n } }$  annotated box</td><td>0.902 [0.868, 0.931]</td></tr><tr><td>Annotated boxes only, matched pedestrians</td><td>0.904 [0.870, 0.931]</td></tr><tr><td>Annotated boxes, all 709 pedestrians</td><td>0.888 [0.852, 0.921]</td></tr><tr><td>1st / 5th / 10th / 25th percentile instead of min Median</td><td>0.911/0.919/0.921/0.910 0.859 [0.809, 0.902]</td></tr><tr><td>Box shrunk 10% / grown 10% Box shifted ≤4 px (5 draws)</td><td>0.921 / 0.892</td></tr><tr><td></td><td>0.904–0.907</td></tr><tr><td>Truncation  $\leq 0 . 1$  only</td><td>0.913 [0.882, 0.942]</td></tr><tr><td>Synthetic depth offset  $b = + 2 \operatorname { m } / + 4 \operatorname { m }$ </td><td>0.900 / 0.876</td></tr></table>

## D Model-Side Alternatives and Lagos Transfer, Full Results

Table 6 and Table 7 give the full per-configuration results referenced in Section 4.6. The 81 evaluated checkpoints are 39 KITTI YOLOv8x (three seeds: baseline, 7 augmentation and 5 MiDaS-fusion configurations, with and without augmentation), 15 KITTI YOLOv8n (one seed: baseline, 8 augmentation, 2 MiDaS-fusion, 1 DAv2-fusion and 3 POPAM configurations), and 27 CityPersons YOLOv8x (three seeds: baseline, 7 augmentation and 1 MiDaS-fusion configuration). Smoke tests, unevaluated architectures and earlier unseeded duplicates are excluded.

Table 6: Three model-side interventions against their RGB-only baselines. Pedestrian class; ∆ is mAP50 relative to baseline. † denotes single-seed results, which record what that configuration did rather than a general conclusion.
<table><tr><td>Intervention</td><td>Setting</td><td>∆ mAP50</td><td>∆ Hard mAP50</td></tr><tr><td>Occlusion augmentation (GridMask)</td><td>KITTI, YOLOv8x</td><td>+0.009</td><td>+0.041</td></tr><tr><td>Occlusion augmentation (H&amp;S)</td><td>CityPersons, YOLOv8x</td><td>+0.001</td><td>-0.004</td></tr><tr><td>Depth fusion (MiDaS)</td><td>KITTI, YOLOv8x</td><td>-0.023</td><td>-0.012</td></tr><tr><td>Depth fusion (MiDaS)†</td><td>KITTI, YOLOv8n</td><td>-0.075</td><td>-0.143</td></tr><tr><td>Depth fusion  $( \mathrm { D A v } 2 ) ^ { \dag }$ </td><td>KITTI, YOLOv8n</td><td>-0.056</td><td>-0.093</td></tr><tr><td>Attention (POPAM)†</td><td>KITTI, YOLOv8n</td><td>-0.030</td><td>-0.055</td></tr></table>

Table 7: Detector transfer to 350 self-annotated Lagos frames (1,290 pedestrian boxes), mean ± s.d. over three training seeds. No model has seen this data. The annotations carry no occlusion labels, so the occlusion score is not evaluated here.
<table><tr><td>Trained on</td><td>Configuration</td><td>In-dist. mAP50</td><td>Lagos mAP50</td></tr><tr><td>KITTI</td><td>Baseline</td><td>0.8809</td><td> $0 . 4 4 8 2 \pm 0 . 0 0 9 4$ </td></tr><tr><td>KITTI</td><td>GridMask</td><td>0.8903</td><td> $0 . 4 1 5 0 \pm 0 . 0 5 3 0$ </td></tr><tr><td>CityPersons</td><td>Baseline</td><td>0.7154</td><td> $\mathbf { 0 . 7 3 5 6 \pm 0 . 0 3 5 1 }$ </td></tr><tr><td>CityPersons</td><td>Hide-and-Seek</td><td>0.7166</td><td> $0 . 7 0 7 5 \pm 0 . 0 1 9 8$ </td></tr></table>

## E Out-of-Domain Validation on CARLA

Data and labels. We render 573 frames in CARLA 0.9.16 (Towns 01–05) from a simulated lowmounted robot camera (0.152 m above the ground, 1280×720), giving 3,293 pedestrians. Every frame is rendered twice, once with occluding actors removed; a pedestrian’s visibility is the ratio of their instance-mask pixel counts between the two renders, so labels are measured rather than annotated. A pedestrian with visibility below 0.8 is occluded (1,493 occluded, 1,800 visible). Depth is Depth Anything V2 Metric-Indoor-Large from RGB alone.

What is and is not held out. The score formulation, the use of full-extent boxes, and the 5thpercentile depth statistic were fixed from KITTI before CARLA was evaluated, and no occlusion label was used to choose anything. The depth checkpoint was chosen using CARLA data: the outdoor checkpoint reads distances several times too far at this camera height, which we found by comparing predicted with ground-truth depth. CARLA supplies oracle full-extent boxes, so these results isolate the depth mechanism from detector behavior; the KITTI analysis of Appendix C covers the detector.

Results. The score reaches AUC 0.819 [0.801, 0.834] (frame bootstrap) against 0.644 for distance alone, and beats distance in every band (Table 8). Performance falls with range because the indoor checkpoint saturates beyond about 8 m, and at this camera height its depth error has a substantial offset, the case Eq. 2 warns about. The KITTI and CARLA results have complementary weaknesses: KITTI has real imagery but coarse human labels, CARLA has exact labels but synthetic imagery and oracle boxes.

Table 8: Occlusion score on CARLA (visibility < 0.8 vs. ≥ 0.8), by distance band. N counts pedestrians; intervals bootstrap frames.
<table><tr><td>Distance band</td><td>N</td><td>AUC of score [95% CI]</td><td>AUC of distance</td></tr><tr><td>0-5 m</td><td>154</td><td>0.969 [0.939, 0.989]</td><td>0.506</td></tr><tr><td>5-10 m</td><td>668</td><td>0.895 [0.868, 0.919]</td><td>0.585</td></tr><tr><td>10-15 m</td><td>687</td><td>0.839 [0.804, 0.871]</td><td>0.562</td></tr><tr><td>15–25 m</td><td>1277</td><td>0.797 [0.771, 0.821]</td><td>0.509</td></tr><tr><td>25–60 m</td><td>507</td><td>0.739 [0.693, 0.783]</td><td>0.481</td></tr><tr><td>all</td><td>3293</td><td>0.819 [0.801, 0.834]</td><td>0.644</td></tr></table>

## F Runtime

Table 9 reports median per-frame latency on 40 KITTI test frames on a laptop CPU (Intel Core i7-8665U, 4 threads, eager PyTorch, fp32, no GPU), with a YOLOv8n detector at 640 px and Depth Anything V2 Metric-Outdoor-Small. The depth network keeps the frame’s aspect ratio, so a 518 px short side is a 518×1722 input on KITTI. The score itself, a box percentile and a product, costs about 0.3 ms; the depth network is 94–98% of the pipeline. A smaller depth input is about 5× faster, but on this small sample the resulting scores rank-correlate with the full-size ones at only 0.68, so its occlusion AUC is untested. We have not measured embedded hardware, compiled or quantized depth networks, or lighter depth models, and make no real-time claim.

Table 9: Median CPU latency per frame (ms), YOLOv8n + Depth Anything V2 Metric-Outdoor-Small.
<table><tr><td>Depth input</td><td>Detection</td><td>Depth</td><td>Score</td><td>Total</td><td>FPS</td></tr><tr><td>518×1722 (default)</td><td>77</td><td>6233</td><td>0.3</td><td>6336</td><td>0.16</td></tr><tr><td>294×980</td><td>66</td><td>1229</td><td>0.3</td><td>1314</td><td>0.76</td></tr></table>

## G Implementation and Evaluation Details

KITTI is split 5,236/1,122/1,123 and CityPersons 2,432/521/522 for train/validation/test. Three seeds are used for every configuration where a confidence interval is reported; single-seed results are marked † in Table 6, giving 81 evaluated checkpoints in total. All models train with SGD (lr0 0.0013, lrf 0.123, weight decay 0.0054) for 150 epochs, batch size 32, patience 20, at 640 px. Occlusion augmentation is applied only during training, and only when more than half of the ground-truth pedestrian remains visible. Depth Anything V2’s metric output is clipped to 0–80 m. The occlusion score is evaluated on 1,941 matched KITTI test detections (682 pedestrians, three YOLOv8x seeds); AUC treats ties as one half, and confidence intervals use 2,000 bootstrap resamples of pedestrians (of frames on CARLA). Scripts reproducing every number in the paper are released with it.

## H Limitations

What is validated. The signal is validated as a ranking of occlusion presence and a coarse category (partly, heavily) against KITTI’s discrete labels, and against one visibility threshold on CARLA; it is not a continuous estimator of occlusion fraction. Evaluation covers pedestrians in daytime imagery only, not other object classes, whose size is far less standardized than human stature.

Detector dependence. The apparent-height assumption holds for predicted boxes at the median but fails in a tail of low-confidence, heavily occluded detections, and the detector misses 27% of largely occluded pedestrians outright. A detector that systematically shrinks occluded boxes would weaken the signal, and should be checked the same way before deployment.

Depth model and camera. Calibration-free ranking holds only when depth error is close to a pure gain over the operating range. It held on KITTI with the outdoor checkpoint, but not with the indoor checkpoint at robot height on CARLA. On our own real-road robot recording, with the outdoor checkpoint at 0.186 m, a thresholded version of the score did no better at its operating point than flagging the smallest boxes. A new depth model, mounting height, or set of camera intrinsics must be checked, for example with the label-free correlation of Appendix B, not assumed.

Untested conditions. We have not evaluated the score in dense crowds, where several people share one box’s nearest surface; at night or in rain; with other monocular depth models; or on embedded hardware. The score cannot distinguish an occluder from any other nearer surface inside the box. The Lagos evaluation measures detector transfer, not the score, because the Lagos annotations have no occlusion labels; labelling occlusion there is the most direct next test of the claim this paper is motivated by.