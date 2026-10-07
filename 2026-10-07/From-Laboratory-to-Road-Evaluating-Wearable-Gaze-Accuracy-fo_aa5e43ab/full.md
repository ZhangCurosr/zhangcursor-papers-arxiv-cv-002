# From Laboratory to Road: Evaluating Wearable Gaze Accuracy for Driving

William Engel<sup>1[0009−0008−5184−5624]</sup> and Fabian Flohr<sup>1[0000−0002−1499−3790]</sup>

Intelligent Vehicles Lab, Hochschule M¨unchen, Germany wengel@hm.edu, fabian.flohr@hm.edu

Abstract. Bird’s-eye-view (BEV) representations have become a widely used interface between perception and planning in autonomous driving, but they encode what is in a scene, not what is behaviorally relevant to a human driver. Gaze ofers a compelling behavioral signal for this gap—yet wearable eye trackers are routinely deployed as if their spatial output were ground truth, despite known sensitivity to head motion, illumination, and calibration drift. We present, to our knowledge, the first unified framework for quantifying wearable gaze accuracy under real driving conditions. Our on-road study contains 41 validated scenes in which one driver fixated a vehicle’s license plate. Gaze error is measured as the angular diference between the plate center and the gaze direction estimated by the glasses. Separate indoor studies with the same driver and device systematically analyze how distance, illumination, head motion, target motion, and gaze eccentricity afect both systematic bias and gaze precision. The mean on-road error was 4.58°. Applying an ofset estimated from the indoor recordings reduced it to 1.10° and improved all 41 scenes. Because this ofset varied between sessions, reliable BEV supervision may require online recalibration and condition-dependent estimates of gaze uncertainty.

Keywords: Mobile eye tracking · Gaze accuracy · Driving perception · Spatial supervision · Error modeling

## 1 Introduction

Autonomous-driving systems increasingly use bird’s-eye-view (BEV) representations to combine multi-sensor information in a common spatial frame for perception, prediction, and planning [13,16,10]. Although BEV encodes objects, trajectories, and map elements, it does not indicate which scene elements are behaviorally relevant to a human driver.

Driver gaze ofers a compact signal of visual selection and has been modeled in datasets such as DR(eye)VE, BDD-A, DADA-2000, and Look Both Ways [21,28,7,11]. Prior work has used gaze for driving prediction, imitation learning, and object-relevance estimation [15,1,3]. However, wearable gaze cannot be treated as exact spatial supervision without quantifying its accuracy and precision during driving. Calibration drift, head motion, illumination, and pupil dynamics introduce condition-dependent errors [18,26,8,23]; after projection into

BEV, even small angular errors may cause large spatial displacements or incorrect object assignments. As gaze is a behavioral sample rather than an exhaustive label, its uncertainty must be represented explicitly.

This pilot study measures wearable gaze accuracy during real driving and asks whether the resulting errors can be explained by matched indoor experiments. On-road targets are declared aloud by the driver and geometrically localized as reference targets. Our framework combines these on-road measurements with five laboratory protocols, each isolating one factor: illumination, head motion, pursuit, eccentricity, and depth. It measures angular accuracy, signed bias, and fixation-level precision, forming a basis for uncertainty-aware gaze supervision in BEV-based driving models.

## 2 Related Work

Eye tracking is widely used to study driver attention, situational awareness, and interaction with advanced driver-assistance systems [6,12,17]. However, simulator studies provide controlled baselines without reproducing all physical disturbances of real-world driving [4].

Validation studies under dynamic conditions identify several relevant error sources. Physical motion can substantially degrade wearable gaze accuracy, with reported errors reaching 5.8<sup>◦</sup> [8,19]. Relative motion between eye and scene cameras, inaccurate depth assumptions, and illumination-induced pupil-size changes can introduce further spatial errors of several degrees [27,5,23]. Even systems that are comparatively robust across lighting conditions may therefore benefit from person- and session-specific ofset correction [2,14].

Prior correction approaches address slippage through regression models or compensate head motion in controlled environments [24,22]. Most closely, Jia et al. used detected trafic signs to correct gaze drift [9], but inferred intended targets probabilistically. In contrast, we validate wearable gaze output during real driving against explicitly declared, geometrically localized targets and relate the resulting errors to matched laboratory protocols.

## 3 Methodology

## 3.1 Field of View and Interior Segmentation

Thirty minutes of natural driving were analyzed to define the gaze range for the controlled targets. Across 163 scene-camera frames, 32% showed exterior road content and 68% was occluded by the vehicle interior (Fig. 1). About 80% of gaze fell within −20<sup>◦</sup> to +20<sup>◦</sup> azimuth and $0 ^ { \circ }$ to $+ 2 0 ^ { \circ }$ elevation, motivating the checkerboards’ azimuth/elevation and FOV coverage.

![](images/28bb92ad754e626771035f553806d409c37f2cec4d0e764043fb9a6553b59d21.jpg)  
Fig. 1. Scene-camera perspective.

## 3.2 On-Road Gaze-Error Measurement

During driving, the driver intentionally fixated the license plate of a selected vehicle and verbally declared the target vehicle. A keyword-based speech detector located each declaration in the synchronized scene-camera audio. A YOLO-based detector provided vehicle candidates, from which the declared target was manually selected and tracked. A second detector localized its license plate. The plate center served as the reference because it defines a precise fixation point, unlike the center of the full vehicle box. All detections and fixations were manually reviewed, yielding 41 validated scenes in which the driver fixated the license plate.

Gaze samples and target detections were synchronized, undistorted using the camera intrinsics, and converted into unit rays. Angular gaze error was defined as the angle between the ray toward the fixated plate center and the gaze ray reported by the glasses. Signed azimuth and elevation errors measure systematic bias, while precision summarizes within-fixation variation in both components.

## 3.3 Controlled Indoor Factor Analysis

To complement the on-road measurements in Sec. 3.2, we collected a separate indoor dataset with the same driver and eye tracker. Five controlled protocols isolate driving-relevant factors and are illustrated by their real-world manifestations in Fig. 2.

![](images/e32a1b841b4bf9ba0af1fc030609cc7dc5a1935d91e54281efb8a5b15302914f.jpg)  
Fig. 2. Driving conditions motivating the protocols: illumination changes, vehicle and head motion, moving targets, eccentric gaze, and varying depth.

Each protocol uses a target designed for its measurement: illumination through brightness sweeps during checkerboard-corner fixation; motion and vibration through a circular grid during head rotations and platform vibration, whose centroids are robust to blur and rolling-shutter distortion [20,25]; pursuit lag through a crosshair moving over an ArUco board at predefined speeds; eccentric gaze across checkerboard corners; and depth ambiguity through scaled checkerboards at 2–12 m with constant angular size, with the static 2 m condition serving as baseline. Together, they measure factor-specific changes in systematic bias and precision.

## 3.4 Ofset Estimation and Transfer

A calibration scene’s ofset is the median per-fixation target-to-gaze angular error, in azimuth and elevation, in scene-camera coordinates. Transfer adds one scene’s ofset unchanged to another’s gaze before error recomputation, excluding self-pairs. For the road result, we take the median of the six static-scene ofsets and apply it to every road scene.

## 4 Preliminary Results

This single-driver pilot includes 41 validated on-road scenes and separate indoor tests of illumination, head motion, pursuit, eccentricity, and depth. The indoor data comprise 405 target fixations and three pursuit recordings.

System Baseline: The static 2 m checkerboard condition was repeated three times, with the driver fixating each of the 15 corners in every repetition. The resulting mean accuracy was 4.08°, while target-relative precision was 0.31° (median) and 0.37° (mean).

– Bias Correction & Ofset: A systematic wearing-related ofset dominated the raw error. Transfer between calibration scenes improved every combination (100% positive transfer; mean improvement ∼3°). Applying the ofset to the road scenes reduced mean error from 4.58° to 1.10° (76%), improving every scene.

– Eccentricity: Targets were grouped into tertiles by their angular distance from the scene-camera center. Median azimuth bias increased from approximately 0.22° in the central region to 0.74° in the outer region, indicating that a single global ofset provides only a first-order correction.

Distance & Speed: Mean error ranged from 4.78–5.26° in the 2–12 m sweep and pursuit medians from 4.03–4.31°; neither showed a monotonic trend.

– Illumination: Brightness changes increased dispersion by roughly 40–50%.

– Head Motion: Head motion produced the largest apparent increase in camera-relative dispersion, reaching 10°; this cannot quantify tracker error because compensatory eye movements and possible slippage are not separated.

## 5 Discussion and Limitations

This pilot establishes a full pipeline for measuring wearable gaze error, from controlled indoor protocols to real driving. An ofset estimated purely from indoor recordings substantially reduced angular error across all 41 road scenes, and the controlled protocols show how individual driving-relevant factors afect accuracy and precision.

The next step is to extend this pipeline to more drivers, sessions, and conditions, providing enough data to model both the most likely gaze location and its uncertainty. Representing gaze as an uncertain spatial region, rather than an exact point, would support uncertainty-aware object assignment in BEV-based supervision.

## References

1. Abdelkarim, M., Abbas, M.K., Osama, A., Anwar, D., Azzam, M., Abdelalim, M., Mostafa, H., El-Tantawy, S., Sobh, I.: GG-Net: Gaze guided network for selfdriving cars. In: IS&T Electronic Imaging: AVM. pp. 171–1–171–8 (2021). https: //doi.org/10.2352/ISSN.2470-1173.2021.17.AVM-171

2. Baumann, C., Dierkes, K.: Neon accuracy test report. Technical report, Pupil Labs (2026). https://doi.org/10.5281/zenodo.18504792

3. Biswas, A., Pardhi, B.A., Chuck, C., Holtz, J., Niekum, S., Admoni, H., Allievi, A.: Gaze supervision for mitigating causal confusion in driving agents. In: Proc. IEEE IV. pp. 2331–2338 (2024). https://doi.org/10.1109/IV55156.2024.10588498

4. Calvi, A., D’Amico, F., Vennarucci, A.: Comparing eye-tracking system efectiveness in field and driving simulator studies. The Open Transportation Journal 17, e187444782301191 (2023). https://doi.org/10.2174/ 18744478-v17-e230404-2022-49

5. Choe, K.W., Blake, R., Lee, S.H.: Pupil size dynamics during fixation impact the accuracy and precision of video-based gaze estimation. Vision Research 118, 48–59 (2016). https://doi.org/10.1016/j.visres.2014.12.018

6. Cvahte Ojsterˇsek, T., Topolˇsek, D.: Eye tracking use in researching driver distraction: A scientometric and qualitative literature review approach. Journal of Eye Movement Research 12(3), 1–30 (2019). https://doi.org/10.16910/jemr.12.3.5

7. Fang, J., Yan, D., Qiao, J., Xue, J., Wang, H., Li, S.: DADA-2000: Can driving accident be predicted by driver attention? analyzed by a benchmark. In: Proc. IEEE ITSC. pp. 4303–4309 (2019). https://doi.org/10.1109/ITSC.2019.8917218

8. Hooge, I.T.C., Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nystr¨om, M.: How robust are wearable eye trackers to slow and fast head and body movements? Behavior Research Methods 55(8), 4128–4142 (2023). https://doi.org/10.3758/ s13428-022-02010-3

9. Jia, S., Koh, D.H., Pomplun, M.: Gaze tracking accuracy maintenance using trafic sign detection. In: Proc. ACM AutomotiveUI. pp. 87–91 (2018). https://doi.org/ 10.1145/3239092.3265947

10. Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., Wang, X.: VAD: Vectorized scene representation for eficient autonomous driving. In: Proc. IEEE ICCV. pp. 8306–8316 (2023). https://doi.org/10.1109/ ICCV51070.2023.00766

11. Kasahara, I., Stent, S., Park, H.S.: Look both ways: Self-supervising driver gaze estimation and road scene saliency. In: Proc. ECCV. Lecture Notes in Computer Science, vol. 13673, pp. 126–142 (2022). https://doi.org/10.1007/978-3-031-19778-9 8

12. Khan, M.Q., Lee, S.: Gaze and eye tracking: Techniques and applications in ADAS. Sensors 19(24), 5540 (2019). https://doi.org/10.3390/s19245540

13. Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., Dai, J.: BEVFormer: Learning Bird’s-Eye-View representation from multi-camera images via spatiotemporal transformers. In: Proc. ECCV. Lecture Notes in Computer Science, vol. 13669, pp. 1–18 (2022). https://doi.org/10.1007/978-3-031-20077-9 1

14. Lin, L., Wu, Z., Lu, Y., Chen, Z., Guo, W.: Recent progress on eye-tracking and gaze estimation for AR/VR applications: A review. Electronics 14(17), 3352 (2025). https://doi.org/10.3390/electronics14173352

15. Liu, C., Chen, Y., Tai, L., Ye, H., Liu, M., Shi, B.E.: A gaze model improves autonomous driving. In: Proc. ACM ETRA. pp. 1–5 (2019). https://doi.org/10. 1145/3314111.3319846

16. Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D.L., Han, S.: BEVFusion: Multi-task multi-sensor fusion with unified Bird’s-Eye View representation. In: Proc. ICRA. pp. 2774–2781 (2023). https://doi.org/10.1109/ICRA48891.2023. 10160968

17. Nagy, V., da Luz, D.M., S´andor, A.P., Borsos, A.: Evaluation of autonomous vehicle<sup>´</sup> takeover performance in work-zone environment. Engineering Proceedings 79(1), 59 (2024). https://doi.org/10.3390/engproc2024079059

18. Niehorster, D.C., Nystr¨om, M., Hessels, R.S., Benjamins, J.S., Andersson, R., Hooge, I.T.C.: The fundamentals of eye tracking, part 7: Determining data quality. Behavior Research Methods 58, 183 (2026). https://doi.org/10.3758/ s13428-026-03039-4

19. Onkhar, V., Dodou, D., de Winter, J.C.F.: Evaluating the Tobii Pro Glasses 2 and 3 in static and dynamic conditions. Behavior Research Methods 56(5), 4221–4238 (2024). https://doi.org/10.3758/s13428-023-02173-7

20. Oth, L., Furgale, P., Kneip, L., Siegwart, R.: Rolling shutter camera calibration. In: Proc. IEEE CVPR. pp. 1360–1367 (2013). https://doi.org/10.1109/CVPR.2013. 179

21. Palazzi, A., Abati, D., Calderara, S., Solera, F., Cucchiara, R.: Predicting the driver’s focus of attention: The DR(eye)VE project. IEEE TPAMI 41(7), 1720– 1733 (2019). https://doi.org/10.1109/TPAMI.2018.2845370

22. Park, J., Jeon, J.Y., Kim, R., Kay, K.N., Shim, W.M.: Motion-corrected eye tracking improves gaze accuracy during visual fMRI experiments. Nature Communications 17(1), 1022 (2026). https://doi.org/10.1038/s41467-025-67767-5

23. Salari, M., Niehorster, D.C., Nystr¨om, M., Bednarik, R.: The efect of pupil size on data quality in head-mounted eye trackers. Behavior Research Methods 58(1), 17 (2026). https://doi.org/10.3758/s13428-025-02880-3

24. Santini, T., Niehorster, D.C., Kasneci, E.: Get a grip: Slippage-robust and glintfree gaze estimation for real-time pervasive head-mounted eye tracking. In: Proc. ACM ETRA (2019). https://doi.org/10.1145/3314111.3319835

25. Song, C., Lee, D., Lim, J., Kim, A.: Camera calibration via circular patterns: A comprehensive framework with measurement uncertainty and unbiased projection model. arXiv preprint arXiv:2506.16842 (2025). https://doi.org/10.48550/arXiv. 2506.16842

26. Tonsen, M., Baumann, C.K., Dierkes, K.: A high-level description and performance evaluation of Pupil Invisible. arXiv preprint arXiv:2009.00508 (2020). https://doi. org/10.48550/arXiv.2009.00508

27. Velisar, A., Shanidze, N.M.: Noise estimation for head-mounted 3D binocular eye tracking using Pupil Core eye-tracking goggles. Behavior Research Methods 56(1), 53–79 (2024). https://doi.org/10.3758/s13428-023-02150-0

28. Xia, Y., Zhang, D., Kim, J., Nakayama, K., Zipser, K., Whitney, D.: Predicting driver attention in critical situations. In: Jawahar, C., Li, H., Mori, G., Schindler, K. (eds.) Proc. ACCV. pp. 658–674. Springer International Publishing, Cham (2019)