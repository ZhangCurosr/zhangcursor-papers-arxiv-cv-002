# How Well Does Webcam Eye Tracking Work “in the Wild”? A Large-Scale Validation of Chiasm and a Comparison with WebGazer

Artem V. Belopolsky, Florian Perdreau

Chiasm, BV, Amsterdam, The Netherlands

Abstract. Webcam eye tracking can greatly accelerate data collection in basic and applied research by reaching large, diverse samples of participants online. However, its spatial performance is generally considered well below that of hardware-based systems, limiting its use. We evaluated the first-generation Chiasm webcam eye tracker in a large online study (N = 505) spanning diverse participants, devices, and recording conditions. Chiasm achieved a mean spatial accuracy of 2.39 cm (≈ 2.28<sup>◦</sup> at 60 cm), spatial precision (SD) of 1.05 cm (≈ 1.00<sup>◦</sup> at 60 cm), and sample-to-sample precision (RMS-S2S) of 1.29 cm (≈ 1.23<sup>◦</sup> at 60 cm). For the 91% of participants who met a 3 cm calibration criterion, spatial accuracy was 2.15 cm (≈ 2.05<sup>◦</sup> at 60 cm). After three minutes of a task that encouraged head movement, spatial accuracy declined by 0.59 cm, while the gaze center at each target shifted by a mean distance of only 0.37 cm. On the validation block, the ofset was 0.30 cm larger for men than for women. Age, region of residence, face luminance, and Michelson contrast showed no reliable relationship with spatial accuracy. A higher eye aspect ratio, a more open eye, was associated with a smaller ofset. Wearing glasses decreased spatial accuracy on every validation block. In a comparison between a random subsample (N = 101) and a separate sample tested with the open-source tracker WebGazer (N = 101) using an identical protocol, Chiasm showed substantially better spatial accuracy (2.18 vs. 4.51 cm) and spatial precision (SD; 1.01 vs. 1.55 cm). WebGazer showed better sample-to-sample precision (RMS-S2S; 0.79 vs. 1.26 cm), which likely reflects conservative, over-smoothed estimates rather than superior stability. These results suggest that Chiasm is narrowing the gap between webcam and hardware-based eye tracking, making largescale gaze research “in the wild” a reality.

Keywords: webcam eye tracking · online experiments · spatial accuracy · spatial precision · sample-to-sample precision · Chiasm · WebGazer

## 1 Introduction

Eye tracking has proven to be a powerful tool for studying human behavior and cognition, with applications ranging from basic research to user experience (UX) research, marketing, and clinical practice [27, 20, 17, 6, 22, 25, 7, 2, 13,

24, 21, 1, 10, 3, 4, 8]. Yet its potential remains constrained by how dificult it is to collect large and diverse datasets. Most eye tracking is still conducted in the lab, under controlled conditions, one participant at a time. Wearable eye trackers have relaxed some of these constraints, but they still rely on specialized hardware that must be delivered to participants. Webcam eye tracking ofers a way around this bottleneck by using the commodity webcams built into virtually every laptop, tablet or smartphone. The idea is simple: anyone with such a device can take part in an eye-tracking experiment. Combined with large, vetted online participant pools such as Prolific [14], this shrinks data collection from weeks to hours.

However, fulfilling this promise presents significant challenges, as data collection takes place “in the wild”, where researchers cannot control participants’ hardware, environment, or behavior. First, commodity webcams are far less capable than the cameras used in lab-based systems. They have lower spatial resolution (often as low as 640 × 480 pixels, with the eye occupying only a small fraction of the image) and lower sampling rates (typically 30–60 Hz, compared to 500–2000 Hz for lab systems such as the EyeLink 1000 Plus). They also lack infrared illumination, which rules out the pupil–corneal reflection method on which most lab-based trackers rely. In addition, webcams apply automatic exposure, white balance, and compression that the experimenter cannot control, and browsers introduce variable frame timing. The rest of the setup varies as well: computing hardware, operating systems, browsers, monitors, and internet connections all difer across participants. Second, the recording environment is uncontrolled: ambient lighting, viewing distance, and potential distractions vary from one participant to the next and can change during a session. Finally, there is no control over participants’ posture and head movements, which are not constrained by a chin rest. Their engagement is also uncontrolled, and an experimenter cannot intervene in real time when data quality deteriorates. These three major challenges for obtaining high-quality gaze data using webcam require not only a gaze estimation algorithm that is robust to the variability in computing infrastructure, recording environment and participant behavior, but also a well-designed user interface that allows self-guidance, feedback and error recovery for the participant, to compensate for the absence of an experimenter in the room.

The most widely used webcam eye tracker is WebGazer [15], an open-source, JavaScript library that detects the eyes in the webcam image and maps eyeimage features to screen coordinates using ridge regression, trained on clicks or gazing at calibration points. Several studies estimated WebGazer’s spatial accuracy to be around 4– $- 5 ^ { \circ }$ in the lab with further degradation online [19]. Most commercial webcam eye trackers, such as RealEye.io (based on WebGazer, [26]) and LabVanced (deep learning-based, [11]), have not been validated $^ { 6 6 } \mathrm { { i n } }$ the wild”, but rather under controlled and favourable conditions against lab-based systems, and reported spatial accuracy of around 1.5<sup>◦</sup>. One recent study [18] compared the performance of several deep learning-based models for gaze estimation (MPIIGaze [28], ETHXGaze [29] and FAZE [16]) on webcam frames collected online $( N = 6 5 )$ , but analyzed ofline, in several tasks. They showed that FAZE achieved the best performance, with a spatial accuracy of 2.44<sup>◦</sup> and a spatial precision (SD) of $1 . 6 3 ^ { \circ }$ and RMS-S2S of $0 . 4 7 ^ { \circ }$ . MPIIGAZe and ETHXGaze achieved a spatial accuracy of 3.70<sup>◦</sup> and $3 . 4 0 ^ { \circ }$ , a spatial precision (SD) of 2.96<sup>◦</sup> and 2.39<sup>◦</sup>, and RMS-S2S of $2 . 3 3 ^ { \circ }$ and 1.80<sup>◦</sup> respectively. The increased performance for diferent models came at the expense of inference time, with inference latency of 12 fps for MPIIGaze, 4 fps for ETHXGaze and 1 fps for FAZE. Especially for FAZE, the most promising model, this means that all processing, including the calibration procedure, has to be done ofline, which may impair user experience and be impractical for some applications. It is worth noting that this research evaluated the ofline performance of several models on webcam data, but did not evaluate a webcam eye tracking system.

The largest validation of a webcam eye tracker “in the wild” (N = 255) has been recently conducted by iMotions with their own WebET 3.0 tracker [9]. Their algorithm uses deep-learning model “pre-trained on thousands of faces” and with a radial basis function (RBF) calibration mapping. They reported a spatial accuracy of $2 . 1 ^ { \circ }$ and that the presence of glasses was the only factor that significantly degraded spatial accuracy. However, the white paper report of iMotions does not provide enough details regarding the validation procedure and analysis, and does not report metrics such as spatial precision (SD) or sample-tosample precision (RMS-S2S), which are crucial for the evaluation of the tracker’s performance and prevents a clear comparison to other trackers.

There are a few reasons to suspect that the metrics that were previously reported for the webcam eye tracking may be rather optimistic. First, most webcam eye trackers were tested under controlled laboratory conditions, which are not representative of diverse online scenarios [26], [11]. Second, in some studies, recalibration was often performed after short intervals, which was done to reduce the efect of drift over time, but it is not always practical when running online [11], [18]. Third, none of the studies estimated the drift in calibration metrics over time, which is important to know and control for, as showed by [5] for the hardware-based eye trackers. Finally, many studies did not report all the metrics that are crucial for the evaluation of the webcam eye tracker’s performance, such as spatial accuracy, spatial precision (SD) and sample-to-sample precision (RMS-S2S) [19], [9].

In this paper, we evaluated the first-generation Chiasm webcam eye tracker (https://chiasm.eu), which uses a deep neural network to predict gaze from a single webcam image, combined with a custom calibration procedure. We conducted a large online study (N = 505) spanning diverse participants, devices, and recording conditions. We designed a novel validation protocol which allowed us to comprehensively evaluate the webcam eye tracker’s performance “in the wild”. We report spatial accuracy, spatial precision (SD), sample-to-sample precision (RMS-S2S), as well as drift of these metrics over time. In addition, we evaluated how the tracker’s performance varies with diferent participant and environmental characteristics, such as age, gender, region of residence, eye aspect ratio, face luminance, or Michelson contrast, and presence of glasses. We also compared the performance of Chiasm with WebGazer, since the latter is the most widely used webcam eye tracker and could be implemented with an identical experimental protocol. We collected a separate sample with WebGazer (N = 101) and compared it with a random sample of participants from the Chiasm cohort.

## 2 Methodology

We used Chiasm’s implementation in jsPsych [12] and designed a novel experimental protocol that allowed us to compute various metrics of the webcam eye tracker’s performance “in the wild”. First, we gave participants three attempts to calibrate the tracker. This ensured that they had a chance to adjust to the calibration procedure and to give it their best efort. Second, even if they failed to calibrate the tracker after three attempts, we still allowed them to continue the session. This way we did not bias our results by excluding participants who failed to calibrate the tracker. Third, directly after the initial 9-point calibration and validation, which used pulsating discs, designed to attract attention, we presented participants with a 13-dot validation block (Validation A), with static discs, which are commonly used in lab-based eye tracking. Fourth, we presented participants with a shooting game, which encouraged them to move their head. Right after the shooting game, another 13-dot validation block was presented (Validation B). This allowed us to assess the tracker’s drift over time and after head movements. The whole experiment lasted approximately 14 minutes.

## 2.1 Participants

The complete dataset (Chiasm and WebGazer) was gathered from multiple participant batches over 12 days, totaling 36 hours of active collection time. Participants were recruited via Prolific and gave informed consent. Data were processed under a data protection impact assessment (DPIA) by Chiasm B.V., in accordance with the General Data Protection Regulation (GDPR). They took part remotely using a web browser. Participants using phones, tablets, iPadOS or Desktop Safari were not allowed to enter the experiment. Declining informed consent also ended the session, as did a failure to initialise the tracker. Participants who continued, reported whether the webcam was built-in, external, or of unknown type; whether the computer was a laptop or a desktop; and whether the display was a laptop screen, an external monitor, or an all-in-one screen.

Appendix A counts participants who started Validation A, including those who did not meet the acceptance threshold, and how many of them started the shooting game, started Validation B, and were included in the final sample.

## 2.2 Apparatus

Participants used their own webcam and display, without a chin rest, and viewing distance was left free. The sampling rate was capped at 30Hz and the frames were rescaled to 640x480 pixels. Physical scale was obtained before calibration by matching an on-screen rectangle to a standard identification card, 85.6 mm wide. The match supplied a pixels-per-centimetre factor for the screen plane. When the browser’s page zoom was away from 100%, a warning was shown and the participant could continue after resetting the zoom. The display was then placed in fullscreen. The tracker recorded webcam frames and the gaze prediction for each frame. The cursor was hidden during the validation blocks and the shooting game.

## 2.3 Stimuli

Calibration and its acceptance check used a 9-point grid on a white field: the screen center, the four edge midpoints, and the four corners. Peripheral points were inset by 10% of screen width or height, and every point was kept at least 40 px inside the canvas. The center point appeared first; the other eight followed in random order. Each point was combination of bulls eye and cross hair, known to minimize microsaccades [23]. Each point was a red disc that expanded from a radius of 20 px to 40 px and then reset, crossed by a white cross 40 px across, with a red center of radius 4 px. The point stayed for 3.1 s. Samples entering the calibration fit and its score began 800 ms after the point appeared.

Validation A and Validation B used thirteen points. The grid contained the same nine locations, placed at the 10% insets with no further edge margin, plus four inner points at 30% and 70% of screen width and height. Each block drew a new random order of all thirteen points. A point was a static red disc of radius 30 px, a white cross spanning the disc, and a red center of radius 4 px, on white. It stayed for 1.5 s, and the next point followed immediately. The recorded target was the center of the point in pixels of the current viewport.

Between the two validation blocks, participants played a shooting game and were encouraged to move their head.

## 2.4 Design

The study was a single-group, within-participant validation with a fixed order. Every participant received the device check, consent, the device questionnaire, screen-size and camera setup, 9-point calibration with its acceptance check, Validation A, three rounds of a shooting game, a face-position check, Validation B, and a debrief. Validation B repeated Validation A after the shooting game. The acceptance check used the same 9-point routine as calibration and passed when the mean ofset error was at most 3 cm. The setup could be repeated for up to three attempts. A failure to detect the face during the camera check did not use up an attempt. If the face was lost during calibration, the setup returned to the camera check and that restart counted. At the end of the setup the session continued whether or not the criterion had been met, and the outcome was stored. Validation instructions asked the participant to hold the head still and look at the center of each point. Instructions for the shooting game encouraged movement of the head as well as the eyes.

## 2.5 Procedure

After a welcome page, participants gave informed consent and completed the device questionnaire. Setup instructions asked them to sit comfortably and keep the head still. They matched the on-screen card to a physical card, made sure their face was centered on the camera image, and entered fullscreen. Calibration and the acceptance check followed, each introduced with the instruction to follow each point with the eyes. Validation A was introduced with the instruction to look at the center of each point and keep the head still. Fullscreen was requested again, and the thirteen points were shown. Participants then read the shootinggame instructions and played three rounds. They were encouraged to move their head during the game. Each round restated the aiming instruction, requested fullscreen, counted down from three, and lasted 60 s, after which a score summary was shown. Before Validation B the camera image was shown again so the participant could place the face in the oval. A failure of that check was recorded and the session continued into Validation B, which repeated the thirteen-point procedure. A final page returned the participant to Prolific. Figure 1 shows this order.

![](images/41fbd02997fa99b24200dc3a57be169cdebe27ba0cfc3b2cc90ca8c7f257fafc.jpg)  
Fig. 1. The block order in both the Chiasm and WebGazer experiments. Every participant followed this sequence. Validation B repeated Validation A after the shooting game.

## 2.6 Data Filtering

Gaze is reported in centimetres in the screen plane, using the scale from the card match. Within each validation point, samples from the first 500 ms after onset were discarded. A sample was then discarded when it jumped more than 5 cm from the last kept sample at that point. Exact repeats of a gaze position within the same session and phase were discarded, keeping the first. Samples farther than three robust standard deviations from the median gaze at that point were also discarded. The robust standard deviation is 1.4826 times the median absolute deviation of distances to that median. Points with fewer than five finite samples, and points whose median absolute deviation was zero, were left untrimmed.

## 2.7 Evaluation metrics

A person-level value in a phase is the mean of the finite per-point values. It is left undefined when fewer than 9 points on a thirteen-point grid, or fewer than six on the 9-point grid, yield a finite value.

Spatial accuracy Spatial accuracy at a point is the median Euclidean distance from gaze to the target. A person’s spatial accuracy in a phase is the mean of those point values.

$$
\mathrm { a c c u r a c y } _ { \mathrm { d o t } } = \mathrm { m e d i a n } _ { i } \sqrt { ( x _ { i } - x ^ { \star } ) ^ { 2 } + ( y _ { i } - y ^ { \star } ) ^ { 2 } }\tag{1}
$$

Spatial precision (SD) Spatial precision (SD) at a point is the root-meansquare deviation of the samples from their own centroid. A point with fewer than three finite samples has no spatial precision (SD). A person’s spatial precision (SD) in a phase is the mean of the remaining point values.

$$
\mathrm { S D } _ { \mathrm { d o t } } = \sqrt { \mathrm { v a r } ( x ) + \mathrm { v a r } ( y ) }\tag{2}
$$

Sample-to-sample precision (RMS-S2S) Sample-to-sample precision (RMS-S2S) at a point is the root mean square of the Euclidean distances between successive kept samples, ordered in time. A point with fewer than three finite samples is omitted. A person’s sample-to-sample precision (RMS-S2S) in a phase is the mean of the remaining point values.

$$
\mathrm { R M S - S 2 S } _ { \mathrm { d o t } } = \sqrt { \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \left[ ( x _ { t + 1 } - x _ { t } ) ^ { 2 } + ( y _ { t + 1 } - y _ { t } ) ^ { 2 } \right] }\tag{3}
$$

Drift On the grid, the predicted center at a point is the median, across people, of the median gaze displacement from the target. Drift is the vector from that center on Validation A to the center on Validation B. Per person, drift is Validation B spatial accuracy minus Validation A spatial accuracy.

CDF The cumulative distribution function is the empirical distribution of perperson spatial accuracy, one value per phase. The denominator is the final sample. A person without a finite spatial accuracy stays in that denominator and does not add a step to the curve.

## 2.8 Demographics and frame statistics

Spatial accuracy on the 9-point validation block was compared across gender, region of residence, and age. Gender was tested with a Welch t-test of Male versus Female. Region and age were tested with one-way analyses of variance. Age was binned as 18–24, 25–34, 35–44, 45–54, and 55+. A region or age bin entered its analysis only when it contained at least two people. The same accuracy was also compared across the device questionnaire with Welch t-tests of laptop versus desktop, built-in versus external webcam, and laptop screen versus external monitor. A person with fewer than six usable points was left out of these tests.

The same spatial accuracy was related to the frame statistics on each validation phase. Presence of glasses, eye aspect ratio, face luminance, and contrast were automatically extracted from the frames. Glasses were tested with a 2 × 3 analysis of variance, glasses between subjects and validation block within subjects. Eye aspect ratio, face luminance, and Michelson contrast were each tested with a multiple regression of spatial accuracy in centimetres on that measure, validation block, a fixed person intercept, and the measure × block interaction.

## 2.9 Comparison with WebGazer

We chose the WebGazer implementation in jsPsych [12], since it is the most widely used implementation of WebGazer. It uses a 9-point calibration block, during which participants are asked to click calibration dots, one by one, in a random order, with each location appearing two times. The calibration block is followed by a 9-point validation block, during which participants are asked to look at the center of each dot for 2 seconds, with recording starting after 1 second.

The WebGazer dataset was collected using an identical experimental protocol on an independent sample of participants (N = 101). The Chiasm dataset was a random draw from the included sessions, matched in size to the WebGazer study. Each dataset used the same procedure for data filtering. Chiasm discarded the first 500 ms after onset on every phase. WebGazer kept every sample on the 9-point block and discarded the first 500 ms on Validation A and Validation B.

The two datasets were compared on the evaluation metrics. For spatial accuracy, spatial precision (SD), and sample-to-sample precision (RMS-S2S), a planned independent-samples t-test compared the trackers on each task. A 2 × 3 analysis of variance then tested tracker, between subjects, by task, within subjects. A person missing any task was left out of that analysis of variance.

The datasets were compared on demographics and frame statistics. Region of residence, gender, age, and glasses were tested with chi-square tests. Face luminance, Michelson contrast, and eye aspect ratio were tested with Welch t-tests.

## 3 Experimental Results

We analysed the Chiasm dataset to evaluate the tracker’s spatial accuracy, spatial precision (SD), and sample-to-sample precision (RMS-S2S). Table 1 gives the means of each evaluation metric on all three validation blocks. It provides two summaries of the data: on with all participants, and the other one with only those who passed the calibration threshold. Further sections describe the data in more detail and provide statistical tests.

Table 1. Mean evaluation metrics for Chiasm. Spatial accuracy, spatial precision (SD), and sample-to-sample precision (RMS-S2S) for all participants and those who passed the calibration threshold. n is the number of participants. n is smaller in some cells because a participant is omitted when fewer than six points on the validation block, or fewer than nine on Validation A or Validation B, yield a finite value for that metric.
<table><tr><td rowspan="2" colspan="2">Calibration Block</td><td colspan="2">Accuracy</td><td colspan="2">Precision (SD) RMS-S2S</td></tr><tr><td colspan="2">n Mean (cm)</td><td colspan="2">n Mean (cm)</td></tr><tr><td rowspan="3">All</td><td>validation</td><td></td><td>2.39 502</td><td>1.05 502</td><td>1.29</td></tr><tr><td>Validation A 502</td><td>502</td><td>3.18 501</td><td>1.08 501</td><td>1.34</td></tr><tr><td>Validation B 499</td><td></td><td>3.77 496</td><td>1.11 496</td><td>1.36</td></tr><tr><td rowspan="3">Passed</td><td>validation</td><td>460</td><td>2.15 460</td><td>1.02 460</td><td>1.26</td></tr><tr><td>Validation A 457</td><td></td><td>2.96 456</td><td>1.06 456</td><td>1.32</td></tr><tr><td>Validation B 455</td><td></td><td>3.53 452</td><td>1.09 452</td><td>1.34</td></tr></table>

## 3.1 Spatial accuracy

In order to visualize the distribution of the typical gaze around the validation points across participants, we computed the median horizontal and median vertical gaze coordinates for each point and for each participant (Figure 2). The predicted center of gaze at each point is the coordinate-wise median of these individual ofsets (the diamond). The contour is the central 50% of a bivariate t distribution with 5 degrees of freedom, located at that median and scaled by the covariance of the individual medians. It shows where the typical median gaze landed and how that landing place varied across participants. The median of the predicted-to-target distances was 0.64 cm on the validation block (9-point), 1.12 cm on Validation A (13 points), and 1.09 cm on Validation B (13 points). Both later blocks were farther from the target than the validation block, and Validation A and Validation B were similar. The contours, the spread of the predicted center across participants, were larger on Validation A than on the validation block and larger again on Validation B. At the points at the edges of the screen, the predicted centers lay inward of the targets, toward the screen center. The contour tilt means participants difer most along the center-to-edge direction, which may be due to regression to the mean for extreme positions.

![](images/52d82da6eeb00cec5d4b7e1f4b1a3e6f6702414edfa66de3dae2107b72822fc0.jpg)  
Fig. 2. Chiasm spatial accuracy at each true target, with the predicted center marked.

The empirical distribution of spatial accuracy is shown in Figure 3 and give a clear picture of the distribution of spatial accuracy across participants. Each person contributes one value per phase: the mean of that person’s per-dot median distances to the target. The curve is the share of those participants whose value is at or below each distance. There must be at least six usable points on the validation block and at least nine on Validation A and Validation B in order to be included. The mean was 2.39 cm on the validation block, 3.18 cm on Validation A, and 3.77 cm on Validation B. A repeated-measures analysis of variance of validation block, using the 494 people with a value on every block, was reliable, $F ( 2 , 9 8 6 ) = 1 6 2 . 0 0 , p < 0 . 0 0 1$ . Planned paired t-tests distinguished the validation block from Validation A, $t = - 1 2 . 2 8 , d f = 4 9 8 , p < 0 . 0 0 1$ , and Validation A from Validation B, $t = - 7 . 5 2 , d f = 4 9 6 , p < 0 . 0 0 1$ . This shows a systematic decrease in spatial accuracy on the later blocks.

![](images/be24c8c5892cf27ed59891d7b69d4941c7b5ca0da4207978d91a415444c3db2e.jpg)  
Fig. 3. Empirical distribution of Chiasm spatial accuracy.

## 3.2 Spatial precision (SD)

In order to visualize the typical within-person spread of gaze at each validation point, we computed the covariance of the samples around that point’s own centroid for each point and for each participant (Figure 4). Spatial precision (SD) is the root mean square of that spread. The contour is the central 50% of a bivariate t distribution with 5 degrees of freedom, centered on the true target and scaled by the median of these covariances across participants. It shows the typical size and tilt of the within-person scatter, separately from where the gaze landed. The median spatial precision (SD), across participants and points, was 0.95 cm on the validation block (9-point), 0.97 cm on Validation A (13 points), and 0.99 cm on Validation B (13 points). The later blocks were only slightly less precise than the validation block. The contours extend farther vertically than horizontally, so spatial precision seems better in the horizontal dimension than in the vertical.

![](images/8c56afd86d8f9668c36448c1e6853915864753f329dd3f738905fb7766889833.jpg)  
Fig. 4. Chiasm within-person spatial precision (SD) drawn at each true target.

Each person then contributes one value per phase: the mean of that person’s per-dot precision values. There must be at least six usable points on the validation block and at least nine on Validation A and Validation B in order to be included. The mean was 1.05 cm on the validation block, 1.08 cm on Validation A, and 1.11 cm on Validation B. A repeated-measures analysis of variance of validation block, using the 490 people with a value on every block, was reliable, $F ( 2 , 9 7 8 ) = 1 9 . 8 6 , p < 0 . 0 0 1$ . Planned paired t-tests distinguished the validation block from Validation $\mathrm { A } , t = - 4 . 1 5 , d f = 4 9 7 , p < 0 . 0 0 1$ , and Validation A from Validation B, $t = - 2 . 8 1 , d f = 4 9 2 , p = 0 . 0 0 5$ . This shows a small systematic decrease in spatial precision on the later blocks.

## 3.3 Sample-to-sample precision (RMS-S2S)

In order to measure how far gaze moved from one sample to the next, we computed the root mean square of the Euclidean distances between successive kept samples at each point and for each participant. Each person then contributes one value per phase: the mean of that person’s per-dot values. There must be at least six usable points on the validation block and at least nine on Validation A and Validation B in order to be included. The mean was 1.29 cm on the validation block, 1.34 cm on Validation A, and 1.36 cm on Validation B. A repeated-measures analysis of variance of validation block, using the 490 people with a value on every block, was reliable, $F ( 2 , 9 7 8 ) = 2 3 . 1 7 , p < 0 . 0 0 1$ . Planned paired t-tests distinguished the validation block from Validation $\mathrm { A } , t = - 6 . 0 1$ 2 $d f = 4 9 7 , p < 0 . 0 0 1$ . Validation A and Validation B did not difer, $t = - 1 . 7 5$ $d f = 4 9 2 , p = 0 . 0 8 0$ . This shows a small decrease in sample-to-sample precision from the validation block to Validation A, and no further change on Validation B.

## 3.4 Drift

In order to visualize how the predicted center moved from Validation A to Validation B, we drew one arrow at each of the 13 validation points (Figure 5). The tail is the predicted center on Validation A and the head is the predicted center on Validation B, the same center as the diamond in Figure 2. The arrows are drawn at seven times their length for better readability. All 13 points had at least nine participants on both blocks (504 on Validation A and 501 on Validation B). The mean shift across these points was −0.00 cm horizontally and 0.27 cm upward. The 0.37 cm in the abstract is the mean length of these shifts: at each of the 13 points, the distance from the Validation A predicted center to the Validation B predicted center, then the average of those 13 distances. A one-sample t-test found no reliable horizontal shift, $t = - 0 . 0 2 , d f = 1 2 , p = 0 . 9 9$ The upward shift was reliable, t = 4.34, df = 12, p < 0.001. This shows a small upward shift of the predicted center, and no horizontal shift.

![](images/f5ef09a766f9c3d458eaa7072c67fb67b8036cc9224d832a202bda50721d776b.jpg)  
Fig. 5. Chiasm drift on the dot grid. Each arrow runs from the validation A predicted center to the validation B center. The arrows are drawn at seven times their length for better readability.

## 3.5 Devices

The participants who took part in the study used primarily laptops (439) and desktops (66), with a built-in webcam (429) or an external one (75). They typically used laptop display (429), and some used an external monitor (64), or an all-in-one setup (12). Figure 10 shows the device questionnaire for the 505 included participants.

## 3.6 Demographics

The data from the Chiasm study comes from a diverse population. Of the 504 participants with a country of residence, 199 lived in Europe, 107 in Western, Central and Southern Asia, 67 in Africa, 56 in Latin America and the Caribbean, 28 in North America, 22 in Eastern and South-Eastern Asia, 20 in the Middle East, and 5 in Oceania. Figure 11 in Appendix shows region of residence, gender, and age. Furthermore, we found that demographic information had a limited efect on on spatial accuracy in the validation block (see Figures 12, 13, and 14 in Appendix C. ). Men (n = 310) had a mean of 2.50 cm and women (n = 191) a mean of 2.20 cm. A Welch t-test distinguished them, $t = 2 . 4 9 , p = 0 . 0 1 3$ . Region of residence and age showed no reliable diference, $F ( 7 , 4 9 3 ) = 1 . 2 6 , p = 0 . 2 7$ and $F ( 4 , 4 9 6 ) = 1 . 0 9 , p = 0 . 3 6$ . Laptop users (n = 436) had a mean of 2.31 cm and desktop users (n = 66) a mean of 2.88 cm. A Welch t-test distinguished them, $t = - 2 . 3 2 , p = 0 . 0 2 3$ . Built-in webcams $( n = 4 2 7 )$ had a mean of 2.32 cm and external webcams $( n = 7 4 )$ a mean of 2.75 cm. That diference was not reliable, $t = - 1 . 9 5 , p = 0 . 0 5 4$ . Laptop screens $( n = 4 2 6 )$ had a mean of 2.29 cm and external monitors $( n = 6 4 )$ a mean of 2.87 cm. A Welch t-test distinguished them, $t = - 3 . 1 6 , p = 0 . 0 0 2$

## 3.7 Frame statistics

In order to relate the camera image to spatial accuracy, we tested glasses, eye aspect ratio, face luminance, and Michelson contrast (see Figures 15–18 in Appendix C). Forty-two participants wore glasses and 463 did not. Glasses decreased spatial accuracy, $F ( 1 , 5 0 3 ) \ = \ 1 3 . 5 0 , \ p \ < \ 0 . 0 0 1$ . The ofset was 2.90 versus 2.36 cm on the validation block, 4.06 versus 3.11 cm on Validation A, and 4.95 versus 3.73 cm on Validation B. The glasses × block interaction was not reliable, $F ( 2 , 1 0 0 6 ) = 2 . 8 4 , p = 0 . 0 5 9$ (Figure 15). A higher eye aspect ratio, a more open eye, was associated with a smaller ofset, slope −7.25 cm per unit, $F ( 1 , 1 0 0 5 ) = 4 . 8 4 , p = 0 . 0 2 8$ . Observed eye aspect ratio ranged from 0.18 to 0.40, which is about 1.6 cm on this slope.

## 3.8 Comparison with WebGazer

Demographics and frame statistics did not difer between the Chiasm subsample and the WebGazer study, except for Michelson contrast, which was higher for WebGazer. Region of residence, gender, and age showed no reliable diference, $\chi ^ { 2 } ( 7 ) = 6 . 6 8 , p = 0 . 4 6 , \chi ^ { 2 } ( 1 ) = 0 . 0 8 , p = 0 . 7 7 .$ , and $\chi ^ { 2 } ( 4 ) = 2 . 6 7 , p = 0 . 6 1$ . Face luminance, eye aspect ratio, and glasses did not difer reliably either, $t = - 1 . 6 2$ $p = 0 . 1 1 , t = 0 . 8 2 , p = 0 . 4 1$ , and $\chi ^ { 2 } ( 1 ) = 0 . 0 8 , p = 0 . 7 8$ . Michelson contrast was −0.05 for WebGazer and −0.14 for Chiasm, $t = - 2 . 9 2 , p = 0 . 0 0 4$

The comparison maps for spatial accuracy, spatial precision (SD), and drift are shown in Figures 19, 20, and 21 in Appendix D. It can clearly be seen that the spatial accuracy and spatial precision (SD) of Chiasm are better than that of WebGazer across all validation blocks. For WebGazer, the oblong shape of the spatial accuracy contours, tilted towards the center of the screen, suggests that the participants tend to regress to the mean for extreme positions. The shape of the spatial precision (SD) contours was stretched in the horizontal dimension for WebGazer, but in vertical dimension for Chiasm. The mean drift across the 13 points was 0.47 cm for Chiasm and 0.68 cm for WebGazer. A paired t-test did not distinguish them, $t = - 1 . 6 4 , d f = 1 2 , p = 0 . 1 3$ . The Chiasm arrows pointed upward and toward the center of the screen. For WebGazer, the drift appeared to be random.

## Evaluation metrics

The empirical distribution of spatial accuracy for Chiasm and WebGazer is shown in Figure 6 and give a clear picture of the distribution of spatial accuracy

across participants. It is clear that Chiasm outperforms WebGazer across all validation blocks.  
![](images/c0e9e35dd900e06ace29c03936406b69692b957558733cd2ec7036c3e4e7f1e3.jpg)  
Fig. 6. Spatial accuracy distributions. Solid lines are Chiasm and dashed lines are WebGazer. Colour is the validation phase.

Table 2 reports the means of each evaluation metric for the Chiasm subsample and the WebGazer study.

Table 2. Mean evaluation metrics for the Chiasm subsample and the WebGazer study. Spatial accuracy, spatial precision (SD), and sample-to-sample precision (RMS-S2S). The Chiasm rows are the random subsample matched in size to the WebGazer study. n is the number of participants. n is smaller in some cells because a participant is omitted when fewer than six points on the validation block, or fewer than nine on Validation A or Validation B, yield a finite value for that metric.
<table><tr><td></td><td></td><td>Accuracy</td><td>Precision (SD)</td><td>RMS-S2S</td></tr><tr><td>Study</td><td>Block</td><td>n Mean (cm)</td><td>n Mean (cm)</td><td>n Mean (cm)</td></tr><tr><td>Chiasm</td><td>validation 101</td><td>2.18 101</td><td>1.01 101</td><td>1.26</td></tr><tr><td></td><td>Validation A 101</td><td>3.01 101</td><td>1.05 101</td><td>1.33</td></tr><tr><td></td><td>Validation B 100</td><td>3.62 100</td><td>1.09 100</td><td>1.36</td></tr><tr><td></td><td>WebGazer validation 101</td><td>4.51 96</td><td>1.55 96</td><td>0.79</td></tr><tr><td></td><td>Validation A 98</td><td>5.62 94</td><td>1.61 94</td><td>0.91</td></tr><tr><td></td><td>Validation B 97</td><td>6.18 91</td><td>1.64 91</td><td>0.95</td></tr></table>

Mean spatial accuracy Figure 7(a) shows one spatial-accuracy value per person. $\mathrm { ~ A ~ 2 ~ } \times \mathrm { ~ 3 ~ }$ analysis of variance, with tracker between subjects and validation block within subjects, used the 100 Chiasm and 97 WebGazer participants with a value on every block. The tracker efect was reliable, $F ( 1 , 1 9 5 ) = 3 5 . 0 3 , p < 0 . 0 0 1$ as was the block efect, $F ( 2 , 3 9 0 ) = 6 6 . 1 4 , \ : p \ : < \ : 0 . 0 0 1$ . The tracker × block interaction was not reliable, $F ( 2 , 3 9 0 ) = 1 . 7 0 , p = 0 . 1 8$ . This shows better spatial accuracy for Chiasm on every block.

Mean spatial precision (SD) Figure 7(b) shows one spatial-precision value per person. The tracker efect was reliable, $F ( 1 , 1 8 9 ) = 7 1 . 5 2 , p < 0 . 0 0 1$ , as was the block efect, $F ( 2 , 3 7 8 ) = 1 2 . 2 7 , p < 0 . 0 0 1$ . The tracker × block interaction was not reliable, $F ( 2 , 3 7 8 ) = 0 . 6 4 , p = 0 . 5 3$ . This shows better spatial precision for Chiasm than WebGazer on every block.

Mean sample-to-sample precision (RMS-S2S) Figure $7 ( \mathrm { c ) }$ shows one sampleto-sample value per person. The tracker efect was reliable, $F ( 1 , 1 8 9 ) = 5 8 . 2 9 $ $p < 0 . 0 0 1$ , as was the block efect, $F ( 2 , 3 7 8 ) = 6 5 . 9 6 , p < 0 . 0 0 1$ . The tracker × block interaction was also reliable, $F ( 2 , 3 7 8 ) = 9 . 3 8 , p < 0 . 0 0 1$ . Chiasm had the larger RMS on every block, so its sample-to-sample precision was lower, and that diference decreased slightly over time.

![](images/02403a39b271f97a7790b749ef7609b49fdc65f41daf83abb3c3bef5dd3d94ec.jpg)

(b)  
![](images/065dff61166bef60916f2116f3d943cc08759bc877c0fab880e33cd6ffb14ef1.jpg)

(c)  
![](images/9d3ae6583783e1d1f76e2fea89fdd76b6656e714715566903656df1b7d96fb7b.jpg)  
Fig. 7. Person-level means for the Chiasm subsample and the WebGazer study. (a) Spatial accuracy. (b) Spatial precision (SD). (c) Sample-to-sample precision (RMS-S2S). Each person is a point. The marker is the mean with its 95% confidence interval, and the bracket is an independent-samples t-test. Counts above 15 cm stay in the mean and the test.

## 4 Discussion

The results from our large-scale validation study (N = 505) show that Chiasm performs better than existing solutions “in the wild”, where the experimenter has no control over the devices, environment or the participant’s behavior. The participants came from a diverse population in terms of region of residence, gender, age, and used a variety of devices. Our comprehensive analysis showed that Chiasm has a spatial accuracy (2.39 cm) and precision (SD 1.05 cm), and the sample-to-sample precision (1.29 cm) is also reasonable given the 30Hz sampling rate. These evaluation metrics improved slightly when only the participants who passed the calibration were included in the analysis (see Table 1). The spatial accuracy and precision (SD) decreased over time, but the sample-to-sample precision was more stable. The drift over time was small and directed upward and toward the center of the screen, which is consistent with regression to the center of the screen [18]. Interestingly, the spatial accuracy decreased sharply from the validation block to Validation A, but only slightly to Validation B. This is likely due to the switch from using dynamic to static targets. In validation block, the targets were pulsating discs, which captured participants’ attention in automatic fashion. In Validation A and Validation B, the same discs, but static, which required more efort from the participants to track. While such static discs are the most common type of calibration targets, they are probably not the most efective way to enforce fixation. Given that participants were not trained in eye-tracking tasks, and did not receive feedback from the experimenter, this is not surprising that the spatial accuracy decreased.

In terms of external variables afecting spatial accuracy, the presence of glasses and the lower eye aspect ratio had substantial detrimental efects. The face luminance or contrast had no efect on spatial accuracy. The demographic variables also had no efect on spatial accuracy, except for gender. Chiasm’s spatial accuracy is comparable to the iMotions white paper report (median of 2.1<sup>◦</sup>), which is the only other large-scale validation study conducted “in the wild” (N = 255). They also found the efect of glasses to be detrimental. Unfortunately, the iMotions study did not report other metrics and were not clear about the exact experimental setup.

Chiasm also outperformed the FAZE model, which included user-specific calibration [18], but was run fully ofline, on both spatial accuracy of 2.28<sup>◦</sup> vs 2.44<sup>◦</sup> and a spatial precision (SD) of 1.00<sup>◦</sup> vs 1.63<sup>◦</sup>. Since the dataset on which FAZE was evaluated excluded participants with glasses, Chiasm’s advantage over FAZE may be underestimated. FAZE yielded a lower RMS-S2S than our system (0.47<sup>◦</sup> vs. 1.23<sup>◦</sup>). This diference likely reflects ofline temporal smoothing of the gaze estimates by the Kalman filter in the FAZE pipeline. In addition, Chiasm runs online with a near real-time inference speed, while FAZE requires fully ofline processing. Interestingly, in that study [18], running the models fully ofline on webcam frames collected online, resulted in high attrition, since only 65 out of 118 participants were included in the analysis. This illustrates a limitation of fully ofline processing, and emphasizes the importance of user interface design for online gaze tracking.

The results from the comparison with WebGazer (N = 101) show that Chiasm performs substantially better than WebGazer across all validation blocks for spatial accuracy, spatial precision (SD), but not for sample-to-sample precision (RMS-S2S) (see Table 2). In combination with poor spatial accuracy than Chiasm, better sample-to-sample precision is likely to reflect the consequences of the ridge regression algorithm used by WebGazer. It suggests that the algorithm is taking smaller steps than Chiasm, but these steps are less clustered around the target.

Overall, we present a rigorous, comprehensive and extensive evaluation of Chiasm’s performance “in the wild”, as well as its comparison with WebGazer. Besides evaluating it on all standard metrics such as spatial accuracy, spatial precision (SD), and sample-to-sample precision (RMS-S2S), we also evaluated it on drift of these metrics over time, which gives a realistic picture of the tracker’s performance in a real-world scenario.

## 5 Conclusion

In this study, which to our knowledge is the largest validation of a webcam-based gaze tracker under unconstrained “in the wild” conditions to date (N = 505), Chiasm achieved a mean spatial accuracy of approximately 2.3<sup>◦</sup> of visual angle. This outperforms WebGazer, which we evaluated directly, and compared favourably with previously reported webcam-based accuracies. Chiasm thus narrows the gap to laboratory-based eye trackers: remote video-based trackers achieve about 0.5<sup>◦</sup> under optimal conditions, and wearable eye trackers typically around 1.5<sup>◦</sup> [5]. A direct comparison with a laboratory eye tracker under controlled conditions, ideally with simultaneous recording, will be an important next step. We are confident that Chiasm is ready to be used in a variety of real-world tasks, relying on multiple areas of interest (AOIs), that are currently mainly performed in the lab. Chiasm is now available as an extension to jsPsych [12], making it easy to use in a wide range of research and applications.

## Funding

This study was funded by Chiasm B.V.

## Competing Interests

A.B. and F.P are co-founders and shareholders of Chiasm B.V., which develops the Chiasm software evaluated here.

## Data Availability

Data is made available at [Zenodo link].

## References

[1] Zulfa Nur’aini ’Afifah et al. “An exploratory study investigating users’ understanding of noise model uncertainties utilising webcam eye tracking”. In: Advances in Cartography and GIScience of the ICA 5 (2025), pp. 1–8. doi: 10.5194/ica-adv-5-1-2025.

[2] Anna Bánki et al. “Comparing online webcam- and laboratory-based eyetracking for the assessment of infants’ audio-visual synchrony perception”. In: Frontiers in Psychology 12 (2022), p. 733933. doi: 10.3389/fpsyg. 2021.733933.

[3] Paul C. Bogdan et al. “Investigating the suitability of online eye tracking for psychological research: Evidence from comparisons with in-person data using emotion–attention interaction tasks”. In: Behavior Research Methods 56.3 (2024), pp. 2213–2226. doi: 10.3758/s13428-023-02143-z.

[4] Adam A. Bramlett and Seth Wiener. “The art of wrangling: Working with web-based visual world paradigm eye-tracking data in language research”. In: Linguistic Approaches to Bilingualism 15.4 (2025), pp. 538–570. doi: 10.1075/lab.23071.bra.

[5] Benedikt V. Ehinger et al. “A new comprehensive eye-tracking test battery concurrently evaluating the Pupil Labs glasses and the EyeLink 1000”. In: PeerJ 7 (2019), e7086. doi: 10.7717/peerj.7086. url: https://peerj. com/articles/7086/.

[6] Zihan Fang et al. “Investigating the feasibility of conducting webcam-based eye-tracking studies in code comprehension”. In: IEEE Transactions on Software Engineering (2026). Advance online publication. doi: 10.1109/ TSE.2026.3651615.

[7] Stephen Hutt and Sidney K. D’Mello. “Evaluating calibration-free webcambased eye tracking for gaze-based user modeling”. In: Proceedings of the 2022 International Conference on Multimodal Interaction (ICMI ’22). ACM, 2022, pp. 224–235. doi: 10.1145/3536221.3556580.

[8] Stephen Hutt et al. “Webcam-based eye tracking to detect mind wandering and comprehension errors”. In: Behavior Research Methods 56.1 (2024), pp. 1–17. doi: 10.3758/s13428-022-02040-x.

[9] iMotions. WebET 3.0 Validation Study Report. Tech. rep. iMotions A/S, 2023. url: https://www.researchgate.net/publication/371701271\_ WebET\_30\_-Validation\_Study\_Report?channel=doi&linkId=64916adc95bbbe0c6edbbd04& showFulltext=true.

[10] Ariel N. James, Rachel Ryskin, Joshua K. Hartshorne, et al. “What paradigms can webcam eye-tracking be used for? Attempted replications of five cognitive science experiments”. In: Collabra: Psychology 11.1 (2025), p. 140755. doi: 10.1525/collabra.140755.

[11] Tobiasz Kaduk et al. “Webcam Eye Tracking Close to Laboratory Standards: Comparing a New Webcam-Based System and the EyeLink 1000”. In: Behavior Research Methods 56.5 (2024), pp. 5002–5022. doi: 10.3758/ s13428-023-02237-8.

[12] Joshua R. de Leeuw, Rebecca A. Gilbert, and Björn Luchterhandt. “ jsPsych: Enabling an open-source collaborative ecosystem of behavioral experiments”. In: Journal of Open Source Software 8.85 (2023), p. 5351. doi: 10.21105/joss.05351.

[13] Onur Özsoy et al. “Turkish-German heritage speakers’ predictive use of case: Webcam-based vs. in-lab eye-tracking”. In: Frontiers in Psychology 14 (2023), p. 1155585. doi: 10.3389/fpsyg.2023.1155585.

[14] Stefan Palan and Christian Schitter. “Prolific.ac—A subject pool for online experiments”. In: Journal of Behavioral and Experimental Finance 17 (2018), pp. 22–27. doi: 10.1016/j.jbef.2017.12.004.

[15] Alexandra Papoutsaki et al. “WebGazer: Scalable Webcam Eye Tracking Using User Interactions”. In: Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence. AAAI Press, 2016, pp. 3839– 3845. url: https://www.ijcai.org/proceedings/2016/540.

[16] Seonwook Park et al. “Few-Shot Adaptive Gaze Estimation”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019, pp. 9367–9376. doi: 10.1109/ICCV.2019.00946.

[17] Yanina Prystauka, Gerry T. M. Altmann, and Jason Rothman. “Online eye tracking and real-time sentence processing: On opportunities and eficacy for capturing psycholinguistic efects of diferent magnitudes and diversity”. In: Behavior Research Methods 56.4 (2024), pp. 3504–3522. doi: 10.3758/ s13428-023-02176-4.

[18] Shreshth Saxena, Lauren K. Fink, and Elke B. Lange. “Deep Learning Models for Webcam Eye Tracking in Online Experiments”. In: Behavior Research Methods 56.4 (2024), pp. 3487–3503. doi: 10 . 3758 / s13428 - 023-02190-6.

[19] Kilian Semmelmann and Sarah Weigelt. “Online Webcam-Based Eye Tracking in Cognitive Science: A First Look”. In: Behavior Research Methods 50.2 (2018), pp. 451–465. doi: 10.3758/s13428-017-0913-7.

[20] Mieke Sarah Slim and Robert J. Hartsuiker. “Moving visual world experiments online? A web-based replication of Dijkgraaf, Hartsuiker, and Duyck (2017) using PCIbex and WebGazer.js”. In: Behavior Research Methods 55 (2023), pp. 3786–3804. doi: 10.3758/s13428-022-01989-z.

[21] Mieke Sarah Slim et al. “Webcams as windows to the mind? A direct comparison between in-lab and web-based eye-tracking methods”. In: Open Mind 8 (2024), pp. 1369–1424. doi: 10.1162/opmi\_a\_00171.

[22] Adrian Stefan et al. “Validation of an open source, remote web-based eyetracking method (WebGazer) for research in early childhood”. In: Infancy 29.1 (2024), pp. 31–55. doi: 10.1111/infa.12564.

[23] Lore Thaler et al. “What is the best fixation target? The efect of target shape on stability of fixational eye movements”. In: Vision Research 76 (2013), pp. 31–42. doi: 10.1016/j.visres.2012.10.012. url: https:// www.sciencedirect.com/science/article/pii/S0042698912003380.

[24] Ine Van der Cruyssen et al. “The validation of online webcam-based eyetracking: The replication of the cascade efect, the novelty preference, and

the visual world paradigm”. In: Behavior Research Methods 56.5 (2024), pp. 4836–4849. doi: 10.3758/s13428-023-02221-2.

[25] Myrte Vos, Serge Minor, and Gillian C. Ramchand. “Comparing infrared and webcam eye tracking in the Visual World Paradigm”. In: Glossa Psycholinguistics 1.1 (2022). Article 9, pp. 1–37. doi: 10.5070/G6011131.

[26] Katarzyna Wisiecka et al. “Comparison of Webcam and Remote Eye Tracking”. In: 2022 Symposium on Eye Tracking Research and Applications. 2022, pp. 1–7. doi: 10.1145/3517031.3529615.

[27] Xiaozhi Yang and Ian Krajbich. “Webcam-based online eye-tracking for behavioral research”. In: Judgment and Decision Making 16.6 (2021), pp. 1485– 1505. doi: 10.1017/S1930297500008512.

[28] Xucong Zhang et al. “Appearance-Based Gaze Estimation in the Wild”. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015, pp. 4511–4520. doi: 10.1109/CVPR.2015.7299081.

[29] Xucong Zhang et al. “ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation”. In: Computer Vision – ECCV 2020. Vol. 12363. Lecture Notes in Computer Science. Springer, 2020, pp. 365–381. doi: 10.1007/978-3-030-58558-7\_22.

## A Attrition in the Chiasm study

![](images/a8c00723d008f12bb0455de55a28d92a91ae91756261c5f10fe3f77c64dcdff9.jpg)  
Fig. 8. Attrition after the start of Validation A in the Chiasm study. Each bar counts participants who started that stage. Included are the participants who contributed to all validation blocks.

## B Attrition in the WebGazer study

![](images/6cd63c119e7dea01af22d0e3646fdf5f9aca8dc4c188d69bddf43e9fbdee25cd.jpg)  
Fig. 9. Attrition after the start of Validation A in the WebGazer study. Each bar counts participants who started that stage. Included are the participants who contributed to all validation blocks.

## C Chiasm benchmark figures

![](images/976bf76febb2867378a2683b1f93160ab2c2abe795343554b7cecc96cf2f2b9c.jpg)

![](images/de6c6d9f7d345a96c01aadccdbb46c416b4f8e027dd9b3c369c7077b4e96b2f0.jpg)

![](images/0246589eb6b4b7e117ca42d43ee4d6807fcb85b0f10a19d59531b48dcba6dede.jpg)  
Fig. 10. Computer, camera, and display reported by the 505 included participants.

![](images/617c521e814b5ab9dfa5839a8bb77ff597b2ccb09202ba63e8c9ff65b50742d0.jpg)

![](images/8c632091c7f552bd035559dc2c45970f7b0a1817e727fcf6a83a1133f51950bb.jpg)

![](images/660e466fb8a6bef171b57b1ce714b50757412a72f15c2bd96509980187f44e77.jpg)  
Fig. 11. Region of residence, gender, and age in the Chiasm included sample. A missing Prolific field is left of that panel.

![](images/c35e167ff9c60d846e265f8871e8d94d921674e12390e8a4e33cb5fa527c9d3c.jpg)  
Fig. 12. Chiasm validation spatial accuracy by gender. A person with fewer than six usable dots is omitted.

![](images/3aef485633d9f83d189a314437174db5261b34174cfd970bd4c39f6d29efb6e9.jpg)  
Fig. 13. Chiasm validation spatial accuracy by region of residence. A person with fewer than six usable dots is omitted.

![](images/cf1cc7c6685230c5d49b1ac3dfa5243fa41899830d9e675b8e9dca88abd81cdb.jpg)  
Fig. 14. Chiasm validation spatial accuracy by age. A person with fewer than six usable dots is omitted.

validation  
![](images/d9318f02a3508f5fad6a1a7d9bdb09b1ae9c07832b4244ab267713770b33e267.jpg)

validation\_a  
![](images/980938e109a6171be497b3fcf1840e222fad8e660b0e3c6532a2864b98087c79.jpg)

validation\_b  
![](images/cb1046f3de387aebef9090c468b9301eef69adaf3b7cf3607e5290297c1c646f.jpg)  
Fig. 15. Chiasm mean spatial accuracy by glasses. Glasses is the session median of the face-frame glasses score, at least one half.

validation\_b n=505  
![](images/8f3b879962665e48e01ae2acf1cab72bfc50e4651bb3308df165a4ed4e620186.jpg)

![](images/7dff3ccd47ca21b40dbb15a433cbe2ccd5bc5d649da1a84e75ee24a5c45b7e85.jpg)

![](images/cfc15885e7a2d0d239954f8dc8dd26112488f9995c06729aa1cc18ea77d54c4c.jpg)  
Fig. 16. Chiasm mean spatial accuracy against eye aspect ratio on the same validation phase.

![](images/f25d659fe239b2d4cc0c73a4ff512337637292aad94effed72e314a2ea5d5f8f.jpg)

validation\_a n=505  
![](images/66d73f4a46369283992250123ef00358d604d2c6ca67bc7011c0f2c91752938c.jpg)

validation\_b n=505  
![](images/2e767498d5899d84ee4204fd2998add2c2846da4bd66bdf0f05ef75eae2676af.jpg)  
Fig. 17. Chiasm mean spatial accuracy against mean face luminance on the same validation phase.

![](images/b3dcfc3d9d401d7511a107010f7db9f1db833949bea7a24d368f0b45a8f16f04.jpg)

![](images/295e745adc2bfd89a80042e1361bf4168d164be6774d56bbcfc2b4a06b7194eb.jpg)

![](images/bdcdb44a8bdd3c652f10bc38bd8e4843fbe91878f6cddea7ec47115b531d1b57.jpg)  
Fig. 18. Chiasm mean spatial accuracy against Michelson contrast between the face and the surrounding frame.

## D Comparison maps

![](images/d1a937f88e31d0853ad7274d6e2a40c2628160fb26d4bfe5ccb5d50c41cc98b7.jpg)  
Fig. 19. Spatial accuracy at each true target for Chiasm and WebGazer.

![](images/d245966cb7665c643ec5dbcbc5c6928a230ca1babf8d5e360e706c573abe2a2f.jpg)  
Fig. 20. Within-person spatial precision (SD) at each true target for Chiasm and WebGazer.

![](images/2fa590c96c85ed4e213dbfa2cc5a4ebdd8c5e6aa1528b876133c68e5432a3b2b.jpg)

![](images/d5f4aab71430519651c3c0875f80bfd74d6e2b49483f8ae36e2caf5d6df505b8.jpg)  
Fig. 21. Drift on the dot grid, from the validation A predicted center to the validation B center for Chiasm and WebGazer.