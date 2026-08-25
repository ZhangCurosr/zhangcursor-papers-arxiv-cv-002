# Simplified Cross-Modal Calibration for Heterogeneous Event-RGB Stereo Systems

Nico Hessenthaler nico.hessenthaler@hs-heilbronn.de

Nicolaj C. Stache nicolaj.stache@hs-heilbronn.de

Center for Machine Learning Heilbronn University of Applied Sciences Heilbronn, Germany

## Abstract

Accurate extrinsic calibration between event-based and frame-based cameras remains a practical bottleneck for heterogeneous stereo systems. Existing approaches often require sensor or target motion, precise synchronization, or computationally expensive event-to-image reconstruction. We propose a simple, motion-free cross-modal calibration framework that uses a temporally modulated, blended ChArUco target presented on standard consumer displays. By alternating between the original pattern and a partially blended version, the target reliably triggers events while remaining continuously observable to a frame-based camera, avoiding blank frames and reducing synchronization constraints to a coarse, trigger-based alignment. We discretize events into frames coarsely aligned with the RGB images, apply lightweight denoising, and perform ChArUcobased intrinsic and stereo extrinsic calibration. Extensive experiments assess robustness to blending opacity, display brightness, external illumination, viewing angle, and handheld acquisition. Compared to the strongest motion-based reference (E2Calib + Kalibr) and a non-motion-based reference (Plasberg et al.), our approach reduces the mean reprojection error by 44% and 6%, respectively, while substantially simplifying the calibration procedure. Finally, we demonstrate practical utility in a robotic eye-tohand calibration case study, showing consistent transformations and stable downstream geometric measurements even under partial occlusions. Code is publicly available at https://github.com/nhessenthaler/simple-evrgb-cal.

## 1 Introduction

Accurate cross-modal calibration remains a bottleneck for the effective integration of eventbased and frame-based cameras in heterogeneous sensor systems [6, 24]. Similar to monocular event camera calibration [5], cross-modal approaches can generally be categorized into motion-based and non-motion-based (static) methodologies.

Motion-based techniques [18, 22] trigger events through physical movement, but frequently demand computationally expensive neural image reconstructions or intricate algorithms to mitigate inherent motion blur. Conversely, static methods [13, 19] generate events by modulating light intensity, thus avoiding motion blur. However, these approaches typically rely on specialized hardware and require strict temporal synchronization to prevent the frame-based camera from losing tracking features during blank display cycles [1]. Overcoming these practical complexities is highly motivated by dynamic application scenarios [26], such as robotic hand-eye calibration [25]. These systems require robust and continuous geometric references even when the target is partially occluded by a robotic manipulator.

To address these limitations, we introduce a straightforward, motion-free calibration framework that utilizes a shared, temporally modulated visual stimulus. By continuously alternating a ChArUco [12] calibration pattern with a partially blended state on a standard consumer monitor, our method reliably induces the log-intensity changes necessary for event generation. Crucially, the target remains continuously visible to the RGB sensor throughout the cycle. This overlay methodology requires only a coarse, trigger-based alignment rather than strict hardware-level synchronization, while eliminating blank frame management, or computationally expensive event-to-image reconstructions. This substantially simplifies the calibration procedure for real-world deployment.

Our main contributions are as follows: (1) A novel cross-modal calibration methodology based on temporal image blending that eliminates the need for specialized hardware and complex image reconstruction, while reducing the synchronization requirement to a coarse alignment. (2) Comprehensive experimental validation demonstrating high statistical stability and robustness against varying display brightness, external illumination disturbances, viewing angle, and handheld acquisition, while reducing the mean reprojection error by 44% and 6% compared to the strongest evaluated motion-based and non-motion-based approaches, respectively. (3) A practical demonstration of the framework’s utility in a robotic eye-to-hand calibration case study, achieving stable geometric measurements and consistent spatial transformations even in the presence of partial occlusions.

## 2 Related Work

Building upon the core distinction between motion-based and static calibration methodologies, both paradigms fundamentally aim to trigger a continuous event stream by inducing pixel-level intensity changes. Beyond these two dominant categories, specialized frameworks such as EPCNet [23] and TUMTraf Event [7, 8] address cross-modal calibration in application-specific contexts, particularly for intelligent transportation systems. An additional line of work aims to simplify cross-modal calibration by leveraging auxiliary sensing modalities. For instance, Dubeau et al. [10] utilize the Active Pixel Sensor (APS) frames of hybrid event cameras to enable calibration via conventional frame-based techniques.

## 2.1 Motion-Based Calibration Approaches

The dominant paradigm in the field applies relative motion between the sensor and a calibration target to trigger the event stream [5, 15, 22]. A prominent work in this category is the E2Calib framework proposed by Muglikar et al. [18], which utilizes neural-network-based image reconstruction (E2VID [21]) to convert asynchronous event streams into dense intensity frames. By reconstructing these intensity frames, standard computer vision toolboxes can be employed for both intrinsic and extrinsic calibration parameter estimation using a moving standard checkerboard pattern. However, Hu et al. [14] argue that reconstructionbased pipelines suffer from limited edge sharpness and reconstruction noise. To address these deficiencies, they propose a reconstruction-free approach. By utilizing a point-grid target, their method performs direct circle detection within the event stream, later matching the data with the frame-based camera using a sliding window time matching. Wang et al. [27] employ a similar methodology but achieve superior subpixel accuracy through an optimized calibration target and an improved feature extraction pipeline. Similarly, Zhang et al. [30] refine motion-based cross-modal calibration by introducing motion-attracted event alignment, which leverages local consistency to mitigate motion blur in accumulated event frames.

Ultimately, these methodologies often present practical challenges for real-world deployment. They typically require complex neural reconstruction, precise synchronization, or sophisticated motion-compensation algorithms to counteract motion blur. In contrast, our motion-free approach sidesteps these requirements, eliminating the dependency on dynamic feature triggering and the associated spatiotemporal alignment complexities.

## 2.2 Non-Motion-Based Calibration Approaches

Static calibration methodologies generate events by modulating light intensity rather than through physical motion, typically utilizing blinking LEDs [4, 9, 13, 31] or displays [3, 28].

Gossard et al. [13] employ an asymmetric marker with frequency-modulated LEDs to distinguish features in the event stream while maintaining visibility for RGB cameras in low-light settings. However, this approach requires specialized hardware and lacks robustness in standard ambient lighting. Moreover, Bertogalli et al. [1] utilize a custom calibration cube integrating blinking LEDs, ChArUco [12] patterns, and planar surfaces for RGB-event-LiDAR alignment. While effective, the reliance on precisely manufactured, specialized hardware limits its accessibility for general applications.

To leverage consumer hardware, Plasberg et al. [19] utilize a binary blinking monitor behind a metallic dot-grid for thermal-event-RGB calibration. However, the authors omit any discussion of temporal synchronization between display and frame-based camera. A binary blinking mechanism may create a critical contrast failure. When the display cycles to black, the absence of contrast between the dark screen and the metallic grid can render the features undetectable to the frame-based camera.

Our overlay-based methodology addresses these limitations, enabling high-contrast feature detection across modalities while reducing the operational complexity of cross-modal calibration. It eliminates the need for specialized hardware and intricate target designs, reduces synchronization constraints and preserves the advantages of static methods.

## 3 Methodology

Our proposed method addresses cross-modal calibration between event-based and framebased cameras by introducing a shared, temporally controlled visual stimulus that is observable in both sensing modalities. The core idea is to actively modulate a calibration pattern such that it simultaneously induces sufficient intensity changes for the event camera while remaining reliably detectable in the frame-based domain. This enables consistent feature extraction without relying on relative motion or precise temporal synchronization. The resulting signals are preprocessed independently to extract calibration features and estimate the intrinsic parameters of each sensor, before being jointly utilized for subsequent stereo calibration. A schematic overview of the full pipeline is shown in Fig. 1, with detailed descriptions of each component provided in the following sections.

![](images/fbce403b1a4decadce1f70249436496eb89079185e6993f15e2c141007857506.jpg)  
Figure 1: Pipeline for cross-modal stereo calibration using a temporally modulated ChArUco pattern. A pattern of blended images is generated via linear interpolation and modulated over time, enabling robust detection in both frame-based and event cameras.

Calibration Target. In contrast to the standard checkerboard or point-grid patterns frequently employed in event-based literature, the proposed method utilizes a ChArUco-based calibration target, as illustrated in Fig. 1 (a). While ChArUco patterns are already firmly established as a standard for frame-based camera calibration, their application to event-based sensors remains emerging. However, prior research has proven their efficacy for robust detection [29] and pose estimation [17] in event-based vision, making them an ideal candidate for bridging the modality gap in calibrating heterogeneous stereo systems. The unique identification of marker IDs provides inherent robustness against partial visibility and occlusions. This is particularly advantageous for systems containing event-based sensors where data streams may be spatially sparse or temporally incomplete. Furthermore, by leveraging a target already proven in the frame-based domain, the proposed method can utilize established, high-performance implementations for pattern detection and calibration. This ensures a stable geometric reference for both modalities even under suboptimal observation conditions.

Instead of relying on hardware-intensive approaches like LED-based ChArUco marker arrays [29], our proposed method utilizes a standard display for pattern presentation, effectively eliminating the need for specialized hardware. By capturing the ChArUco pattern concurrently with both the event and frame-based sensors, we establish a shared observation that serves as the basis for a unified intrinsic and extrinsic calibration framework.

Overlay Generation. To achieve this shared observation under static conditions, we temporally modulate the calibration pattern on a screen by blending [20] it with a uniform white image, as shown in Fig. 1 (a). Let $I _ { p } ( \mathbf { x } )$ denote the static ChArUco pattern and $I _ { w } ( \mathbf { x } )$ a constant white image of identical dimensions. We define a blended state $I _ { \alpha } ( \mathbf { x } )$ as a linear interpolation between these two images:

$$
I _ { \alpha } ( \mathbf { x } ) = \left( 1 - \alpha \right) I _ { p } ( \mathbf { x } ) + \alpha I _ { w } ( \mathbf { x } ) ,\tag{1}
$$

where $\alpha \in ( 0 , 1 )$ is a fixed opacity parameter. To induce events, the display stimulus $I ( { \bf x } , t )$ where t denotes the continuous time variable, periodically alternates between the static ChArUco pattern $I _ { p } ( \mathbf { x } )$ and the blended state $I _ { \alpha } ( \mathbf { x } )$ , as illustrated in Fig. 1 (b). We define a

full modulation cycle at a frequency of $f = 1 5 ~ \mathrm { H z }$ , corresponding to 30 state transitions per second, yielding the following temporal profile:

$$
I ( \mathbf { x } , t ) = { \left\{ \begin{array} { l l } { I _ { p } ( \mathbf { x } ) } & { { \mathrm { i f ~ } } \lfloor 2 f t \rfloor { \mathrm { ~ m o d ~ } } 2 = 0 } \\ { I _ { \alpha } ( \mathbf { x } ) } & { { \mathrm { i f ~ } } \lfloor 2 f t \rfloor { \mathrm { ~ m o d ~ } } 2 = 1 . } \end{array} \right. }\tag{2}
$$

This discrete switching creates sufficient log-intensity changes ∆ln $I ( \mathbf { x } , t )$ at 33.3 ms intervals to reliably trigger events. Crucially, selecting an appropriate α ensures the ChArUco geometry remains continuously detectable to the frame-based sensor. By avoiding the blank states characteristic of binary flickering, this persistent visibility eliminates the need for strict hardware-level synchronization between the display device and the cameras. Thus, the target can be pre-rendered as a standard video sequence and displayed on any commodity hardware, such as a smartphone, tablet, or monitor.

Data Processing. To establish a weak correspondence between the sensing modalities, both sensors are temporally aligned using a lightweight, software-based scheme that provides only coarse synchronization. The frame-based camera serves as the master device, providing trigger signals for the event stream. No precise or hardware-level synchronization is enforced, making the setup straightforward to implement and requiring no additional synchronization hardware. While this results in temporal offsets and potential jitter, the static nature of our proposed method renders the calibration insensitive to such inaccuracies.

For feature extraction, the asynchronous event stream is mapped into a discrete 2D representation. As shown in Fig. 1 (c), we construct event frames by accumulating events over temporal intervals aligned with consecutive captures of the frame-based camera [11]. Both the generated event frames and the corresponding RGB images are converted to grayscale for further processing.

To ensure reliable detection of ArUco markers within the ChArUco board, a median filter is applied to the event frames. This suppresses noise and densifies the sparse regions of the calibration pattern while preserving the underlying geometric structure. These processed frames are then integrated into a standard calibration pipeline (e.g., OpenCV [2]) to solve for the final intrinsic and stereo extrinsic parameters, as outlined in Fig. 1 (c–e). The resulting pipeline is modality-agnostic and robust, supporting both handheld acquisition and fixed-base robot-assisted setups. In robot-assisted configurations, the system actuates either the camera rig or the calibration target through a sequence of perspectively diverse poses, triggering captures for both sensors only once a stationary scene is ensured.

## 4 Experiments

A series of experiments was conducted to analyze the operating range, limitations, and robustness of the proposed cross-modal stereo calibration approach. The ablation studies focus on parameter influences related to image overlaying, sensitivity to display properties, statistical stability, illumination stability, viewing angle stability, and the ease of operation in practical, real-world scenarios. To situate our work within the current state-of-the-art, we further compare against the motion-based E2Calib framework of Muglikar et al. [18], including its ChArUco-adapted and Kalibr-based variants, as well as the non-motion-based approach of Plasberg et al. [19]. Finally, we demonstrate the practical utility of our framework through a downstream hand-eye calibration task on a robotic platform.

## 4.1 Experimental Setup

To ensure high repeatability and objective comparison across all evaluation scenarios, we utilize a standardized data collection pipeline.

Sensor Configuration. Our sensor rig consists of a frame-based IDS UI-5250CP camera (native resolution 1600 × 1200 px) and a Prophesee EVK4 event-based sensor (resolution 1280 × 720 px). To ensure both sensors observe the same real-world field of view, the UI-5250CP images were cropped to a region of interest of 1380 × 780 px, matching the Prophesee sensor’s field of view, considering the smaller effective pixel size of the frame-based camera. Both sensors were equipped with identical 12 mm C-mount lenses (IDS-2M23- C1214) that were manually focused prior to the experiments. The aperture was set to f/11 to provide a large depth of field, ensuring the calibration pattern remains sharp across a wide range of poses.

Calibration Settings. The sensors were specifically tuned to the requirements of their respective modalities. For the frame-based camera, parameters like exposure time and gain were adjusted to achieve a well-exposed image, while the bias settings of the event sensor were optimized to produce a dense, but noise-reduced event stream. Once established, these settings were frozen for all subsequent experiments to maintain strict comparability. A detailed overview of the parameter configurations is provided in Appendix A. Parameters not explicitly listed in Tab. 10 and Tab. 11 were kept at their manufacturer hardware default settings.

Robot and Display Configuration. All experiments, except the handheld evaluation, were performed using a Universal Robots UR5e 6-DOF manipulator. For the automated primary ablation studies the robot moved the sensor rig through a trajectory of 39 diverse poses relative to a static 27-inch Dell S2722QC monitor (3840 × 2160 px). At each pose, the robot remained stationary during the capture sequence to ensure a stable acquisition for both modalities and to prevent motion-induced artifacts.

For the comparative analysis against the method of Muglikar et al., the same 39 poses were utilized. However, in accordance with the procedure described in the respective original work, the data was acquired while the robot with the camera rig was in motion. An overview of the setup used for the robot-assisted ablation studies and comparative analysis is provided in Fig. 5 of Appendix A.

In addition to the robot-based configuration, a handheld setup was evaluated using the OLED display of an iPhone 15 (2556 × 1179 px) to assess the applicability across different display technologies. In this case, the calibration target was moved manually across at least 39 poses, resulting in less repeatable but more unconstrained viewpoint distributions compared to the robot-based setup.

For the hand-eye calibration task, the configuration was adapted. A compact 3.5-inch Waveshare LCD (480 × 800 px) was mounted to the robotic arm as a mobile calibration target. The robot traversed 22 distinct poses or 6 trajectories in both scenarios, respectively, within the static sensor rig’s field of view.

![](images/e11da0bc7e04d8e8b519f77f1bf097275c52a7cb6d3c59c6857c59e6b3662d92.jpg)  
Figure 2: Detection rate and stereo reprojection error as a function of the blending opacity α. Frame-based detection remains close to 100% across most opacity levels, with a slight degradation for $\alpha \to 1$ due to the sporadic invisibility of the pattern in the absence of temporal display synchronization. In contrast, event-based detection increases with α and saturates for $\alpha \ge 0 . 4$ . The stereo reprojection error reaches a minimum at $\alpha = 0 . 6$ , indicating an optimal trade-off between detection robustness and calibration accuracy.

## 4.2 Ablation Studies

Opacity Parameter. The influence of the blending opacity α on cross-modal calibration performance was investigated through a comprehensive ablation study to determine a value that ensures high fidelity for both sensing modalities.

The evaluation shown in Fig. 2 reveals that the operating point must be carefully selected to satisfy the conflicting sensor requirements. At lower values $( \alpha < 0 . 3 2 )$ , the intensity variations produced by the toggling overlay are insufficient to generate a reliable stream of events, resulting in a total failure of the event-based detection pipeline. Conversely, higher alpha values improve event triggering at the cost of increased occlusion of the calibration pattern during overlay for the frame-based sensor. This effect degrades the frame-based camera, causing a measurable drop in detection rate and an increase in stereo reprojection error as α approaches 1.0. Since there is no synchronization between the frame-based sensor and the display, extreme overlay settings can arbitrarily lead to frames in which the calibration pattern is not visible for the sensor.

The study confirms that the system achieves its most balanced performance at $\alpha = 0 . 6$ At this operating point, the induced intensity changes are sufficient to ensure a 100% event detection rate, while maintaining adequate transparency for precise frame-based calibration target detection, resulting in the minimum observed stereo reprojection error of 0.38 px. Consequently, $\alpha = 0 . 6$ is used for all subsequent experiments unless stated otherwise.

Display Brightness. To evaluate the versatility of our framework across different display hardware, we varied the brightness of our test monitor from 0% to 100% (Tab. 1). While the contrast ratio remains stable at approximately 1250 : 1, absolute luminance significantly impacts both sensing modalities. At 0% brightness, the RGB sensor suffers from severe underexposure, while the event camera’s signal-to-noise ratio (SNR) is heavily degraded. Combined, these factors result in a 15.3% corner detection rate, preventing stereo calibration. Notably, for all brightness settings of 20% and above, the system achieves a success rate exceeding 98%. The reprojection error further improves from 0.442 px to 0.384 px as brightness increases, likely due to increased pixel bandwidth and reduced temporal jitter in the event stream. These results demonstrate that our proposed method is robust across a wide range of displays. Unless indicated otherwise, a display brightness of 80% was used in all subsequent experiments, providing low stereo reprojection errors and a default luminance level representative of state-of-the-art consumer displays.

Table 1: Influence of display brightness on stereo calibration performance for a single experimental run, evaluated across brightness levels from 0% to 100%. The table reports the measured luminance of the white $( L _ { \mathrm { w h i t e } } )$ and black $( L _ { \mathrm { b l a c k } } )$ display states, the combined successful corner detection rate $\left( u _ { c } \right)$ , and the stereo reprojection error $( \varepsilon _ { \mathrm { r e p } } )$ , with arrows (↑, ↓) indicating the direction of improved performance.
<table><tr><td>Brightness [%]</td><td> $\overline { { L _ { \mathrm { w h i t e } } \left[ \mathrm { c d } / \mathrm { m } ^ { 2 } \right] } }$  1</td><td> $\overline { { L _ { \mathrm { b l a c k } } \ [ \mathrm { c d } / \mathrm { m } ^ { 2 } ] } }$ </td><td> $u _ { c } \left[ \% \right] \mathrm { \uparrow }$ </td><td> $\underline { { \varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow } }$ </td></tr><tr><td>0</td><td>35.0</td><td>0.03</td><td>15.3</td><td></td></tr><tr><td>20</td><td>78.0</td><td>0.06</td><td>98.5</td><td>0.442</td></tr><tr><td>40</td><td>121.0</td><td>0.10</td><td>100.0</td><td>0.418</td></tr><tr><td>60</td><td>164.0</td><td>0.13</td><td>100.0</td><td>0.405</td></tr><tr><td>80</td><td>236.0</td><td>0.19</td><td>100.0</td><td>0.395</td></tr><tr><td>100</td><td>391.0</td><td>0.32</td><td>100.0</td><td>0.384</td></tr></table>

– Indicates that insufficient corner detections precluded calibration.

Table 2: Statistical robustness evaluation of the intrinsic parameters over 20 independent calibration runs using a robotic setup. The table reports the mean ( ) and standard deviation (σ) for the corner detection success rate $\left( u _ { c } \right)$ , focal lengths $( f _ { x } , f _ { y } )$ , principal point $( c _ { x } , c _ { y } )$ and reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Sensor</td><td rowspan=1 colspan=1> $u _ { c } \left[ \% \right]$ ↑</td><td rowspan=1 colspan=1> $f _ { x } \ [ { \bf p } { \bf x } ]$ </td><td rowspan=1 colspan=1> $f _ { \mathrm { y } } \ [ \mathrm { p x } ]$ </td><td rowspan=1 colspan=1> $c _ { x } \left[ \mathfrak { p x } \right]$ </td><td rowspan=1 colspan=1> $c _ { y } \left[ \mathfrak { p } \mathbf { x } \right]$ </td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>RGB    (µ)(σ)</td><td rowspan=1 colspan=1>100.00.0</td><td rowspan=1 colspan=1>2771.53.5</td><td rowspan=1 colspan=1>2772.63.4</td><td rowspan=1 colspan=1>800.11.8</td><td rowspan=1 colspan=1>450.81.4</td><td rowspan=1 colspan=1>0.1420.006</td></tr><tr><td rowspan=1 colspan=1>Event   (μ)(σ)</td><td rowspan=1 colspan=1>100.00.0</td><td rowspan=1 colspan=1>2571.126.4</td><td rowspan=1 colspan=1>2572.326.3</td><td rowspan=1 colspan=1>609.310.7</td><td rowspan=1 colspan=1>378.29.1</td><td rowspan=1 colspan=1>0.4490.013</td></tr></table>

Calibration Repeatability. To evaluate the numerical robustness of the proposed method, we conducted 20 independent calibration runs under identical conditions. As summarized in Tab. 2, both sensors achieved a 100.0% corner detection success rate, confirming the reliability of the feature acquisition process. The RGB camera intrinsics exhibited minimal variability, whereas the event camera displayed higher standard deviations due to the inherent sparsity and noise of event-based data. Nevertheless, it still maintained stable reprojection errors across all trials. Stereo calibration results (Tab. 3) further validate the repeatability of the framework. The estimated translation magnitude consistently aligned with the nominal baseline of 0.0344m, derived from the camera rig’s CAD model, with negligible variance. Rotation estimates, quantified by the angular magnitude ∥r∥, were highly consistent across repetitions. Their absolute accuracy is subject to mechanical installation tolerances of the camera mounts, which preclude an exact ground-truth comparison. Nevertheless, the consistently low reprojection errors across all phases underscore the reliability of the proposed framework.

Table 3: Statistical robustness evaluation of the extrinsic parameters over 20 runs. The table reports the successful corner detection rate $\left( u _ { c } \right)$ , the magnitude of the estimated rotation $( \lVert \mathbf { r } \rVert )$ and translation $( \lVert \mathbf { t } \rVert )$ , and the reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Sensor</td><td rowspan=1 colspan=1> $u _ { c } \left[ \% \right]$ ↑</td><td rowspan=1 colspan=1>∥|r||[°]</td><td rowspan=1 colspan=1>||||[m]</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>RGB + Event  (μ)(σ)</td><td rowspan=1 colspan=1>99.80.5</td><td rowspan=1 colspan=1>1.0770.193</td><td rowspan=1 colspan=1>0.03460.00017</td><td rowspan=1 colspan=1>0.3830.012</td></tr></table>

Table 4: Illumination robustness evaluation of the extrinsic parameters over 5 runs with varying spotlight positions. The table reports the successful corner detection rate $\left( u _ { c } \right)$ , the magnitude of the estimated rotation $( \lVert \mathbf { r } \rVert )$ and translation $( \lVert \mathbf { t } \rVert )$ , and the reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Sensor</td><td rowspan=1 colspan=1> $u _ { c } \left[ \% \right]$ ↑</td><td rowspan=1 colspan=1>||r||[°]</td><td rowspan=1 colspan=1>||t||[m]</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>RGB + Event  (µ)(σ)</td><td rowspan=1 colspan=1>92.810.8</td><td rowspan=1 colspan=1>1.2520.266</td><td rowspan=1 colspan=1>0.03450.00012</td><td rowspan=1 colspan=1>0.4020.017</td></tr></table>

Illumination Stability. The robustness of the framework to adverse lighting was evaluated through 5 calibration trials under artificial spotlight interference. We varied the position of a high-intensity LED source across 5 distinct locations, including lateral and vertical offsets, to simulate challenging real-world conditions such as direct sunlight. For reference, a qualitative example of the applied illumination disturbance is shown in Fig. 6 of Appendix B. The resulting extrinsic statistics are detailed in Tab. 4. An analysis of the corresponding intrinsic calibration behavior under these illumination conditions is provided in Appendix B, but is not the primary focus of this study.

Our extrinsic calibration results indicate that while strong, spatially varying illumination noticeably reduces the corner detection success rate and increases its variability, the geometric integrity of the calibration remains intact. Specifically, the estimated translation magnitude remains consistent with the nominal baseline, averaging 0.0345 m across all disturbed trials. While the rotation estimates and reprojection errors show a marginal increase in variance compared to controlled conditions, they remain within a low-error regime. These findings demonstrate that the chosen pattern and proposed calibration method are resilient to localized illumination artifacts. Disturbances primarily affect detection efficiency rather than final registration accuracy. Such robustness confirms the potential suitability of the method for environments where controlled illumination cannot be guaranteed.

Viewing Angle Stability. To evaluate robustness against viewing angle variations, we conducted tests using the Dell monitor described in Sec. 4.1. Here, the viewing angle denotes the angular deviation of the camera’s optical axis from the display’s surface normal. The diagonal axis corresponds to a combined tilt applied simultaneously and equally along the vertical and horizontal directions. As detailed in Tab. 5, angles from $3 0 ^ { \circ }$ to $7 0 ^ { \circ }$ were evaluated along the vertical (V), horizontal (H), and diagonal (D) axes, averaged over 5 runs per orientation. The proposed method remains stable up to $5 0 ^ { \circ }$ , maintaining a corner detection rate of $u _ { c } \ge 8 9 . 2 \%$ alongside low reprojection errors across all axes. This range well exceeds typical calibration requirements. Beyond $5 0 ^ { \circ }$ , performance degrades sharply. This is most pronounced along the diagonal axis, where detection fails entirely from $6 0 ^ { \circ }$ onward. Despite this degradation at extreme angles, our approach remains reliable across the range of poses relevant for practical calibration.

Table 5: Viewing angle stability evaluation across viewing angles from $3 0 ^ { \circ }$ to 70<sup>◦</sup>, evaluated along the vertical (V), horizontal (H), and diagonal (D) axes. The table reports the combined corner detection rate $\left( u _ { c } \right)$ and reprojection error $( \varepsilon _ { \mathrm { r e p } } )$ , averaged over 5 runs per direction.
<table><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>Vertical (V)</td><td rowspan=1 colspan=2>Horizontal (H)</td><td rowspan=1 colspan=2>Diagonal (D)</td></tr><tr><td rowspan=1 colspan=2>Angle</td><td rowspan=1 colspan=1> $u _ { c }$ [%]↓</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } }$ [px]↑</td><td rowspan=1 colspan=1> $u _ { c }$ [%] ↓</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } }$ [px]↑</td><td rowspan=1 colspan=1> $u _ { c }$ [%]↓</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } }$ [px]↑</td></tr><tr><td rowspan=1 colspan=2> $\overline { { 3 0 ^ { \circ } } }$ </td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.322</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.320</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.313</td></tr><tr><td rowspan=2 colspan=2> $4 0 ^ { \circ }$ </td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.342</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.388</td></tr><tr><td rowspan=1 colspan=2>50°</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.374</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.359</td><td rowspan=1 colspan=1>89.2</td><td rowspan=5 colspan=1>0.4820.926一一一</td></tr><tr><td rowspan=2 colspan=2>55°60°</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.436</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>0.374</td><td rowspan=2 colspan=1>48.30.0</td></tr><tr><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.477</td><td rowspan=1 colspan=1>14.1</td><td rowspan=1 colspan=1>0.377</td><td></td></tr><tr><td rowspan=1 colspan=2> $6 5 ^ { \circ }$ </td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=2> $7 0 ^ { \circ }$ </td><td rowspan=1 colspan=1>25.8</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>0.0</td></tr></table>

– Indicates that insufficient corner detections precluded calibration.

Table 6: Handheld calibration evaluation of the extrinsic parameters over 5 runs. The table reports the successful corner detection rate $\left( u _ { c } \right)$ , the magnitude of the estimated rotation $( \lVert \mathbf { r } \rVert )$ and translation (∥t∥), and the reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Sensor</td><td rowspan=1 colspan=1> $u _ { c }$ [%]↑</td><td rowspan=1 colspan=1>||r||[°]</td><td rowspan=1 colspan=1>||t||[m]</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>RGB + Event  (µ)(σ)</td><td rowspan=1 colspan=1>100.00.0</td><td rowspan=1 colspan=1>0.94350.285</td><td rowspan=1 colspan=1>0.03430.00027</td><td rowspan=1 colspan=1>0.3610.041</td></tr></table>

Handheld Calibration. To further assess the practical applicability of the proposed approach, we perform an ablation study using a handheld acquisition setup. In contrast to the robot-based evaluation, this configuration removes precise and repeatable motion control, resulting in more arbitrary pose distributions. For each run, at least 39 poses are captured with similar spatial coverage as in the robot-based experiments. A total of 5 independent runs are conducted.

The results are summarized in Tab. 6 and compared to the statistical robustness evaluation in Tab. 3. In this handheld setting, our proposed method achieves a perfect corner detection rate in both modalities, indicating reliable feature extraction even under less constrained acquisition. Furthermore, the mean reprojection error is slightly lower than in the robot-based setup. This improvement is likely due to the higher contrast and smaller pixel pitch of the OLED display employed in the handheld setup, compared to the Dell LCD monitor. However, reduced pose control inherent to manual acquisition increases the standard deviation, though transformation magnitudes remain comparable.

Overall, these results demonstrate that the proposed calibration approach generalizes well to practical scenarios, where precise positioning hardware may not be available.

## 4.3 Comparative Evaluation

To assess the effectiveness of the proposed approach, we compare against four baselines. The first is the reconstruction-based E2Calib method of Muglikar et al. [18]. We additionally consider an adapted variant that employs a ChArUco calibration target instead of the checkerboard target (E2Calib + ChArUco, see Fig. 3), as well as a further variant substituting Kalibr for OpenCV as the calibration backend (E2Calib + Kalibr). Finally, we include the static approach of Plasberg et al. [19]. All baselines are evaluated over 5 repetitions and reported in Tab. 7, while the performance of our overlay method is summarized in Tab. 3.

![](images/5648ea83f98ff870baa73c6ecfb8b2a4e75c8717982721bbdb8078f67ce18ca1.jpg)  
(a) Checkerboard E2VID Reconstruction

![](images/fbd56d190cd2af94a35e599b6256af484941cb8edfff26052d51d7e900c9b225.jpg)  
(b) ChArUco E2VID Reconstruction  
Figure 3: E2VID reconstructed intensity frames used for the original E2Calib baseline (checkerboard) and our adapted ChArUco target-based variant.

Table 7: Extrinsic calibration performance of the motion-requiring baseline (E2Calib), its ChArUco-based adaptation (E2Calib + ChArUco), a variant using Kalibr instead of OpenCV for calibration (E2Calib + Kalibr), and the static baseline method of Plasberg et al., each evaluated over 5 runs. The table reports the successful corner detection rate $\left( u _ { c } \right)$ , the magnitude of the estimated rotation $( \lVert \mathbf { r } \rVert )$ and translation $( \lVert \mathbf { t } \rVert )$ , and the reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1> $u _ { c } \left[ \% \right]$ ↑</td><td rowspan=1 colspan=1>|r||[°]</td><td rowspan=1 colspan=1>||t||[m]</td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } } \left[ \mathrm { p x } \right] \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>E2Calib                (µμ)(σ)</td><td rowspan=1 colspan=1>47.75.6</td><td rowspan=1 colspan=1>1.4370.386</td><td rowspan=1 colspan=1>0.03390.0004</td><td rowspan=1 colspan=1>1.2350.156</td></tr><tr><td rowspan=1 colspan=1>E2Calib + ChArUco  (μ)(σ)</td><td rowspan=1 colspan=1>67.73.7</td><td rowspan=1 colspan=1>1.7910.381</td><td rowspan=1 colspan=1>0.03380.0003</td><td rowspan=1 colspan=1>1.1610.108</td></tr><tr><td rowspan=1 colspan=1>E2Calib + Kalibr      (µμ)(σ)</td><td rowspan=1 colspan=1>**</td><td rowspan=1 colspan=1>1.5810.178</td><td rowspan=1 colspan=1>0.03600.0002</td><td rowspan=1 colspan=1>0.6870.038</td></tr><tr><td rowspan=1 colspan=1>Plasberg et al.          (µμ)(σ)</td><td rowspan=1 colspan=1>56.410.1</td><td rowspan=1 colspan=1>2.0390.175</td><td rowspan=1 colspan=1>0.03520.0001</td><td rowspan=1 colspan=1>0.4060.019</td></tr></table>

\* Values are not reported in the Kalibr calibration output.

Across all four baselines, our method consistently yields lower reprojection errors, indicating improved calibration accuracy. Among the E2Calib variants, the ChArUco adaptation shows a minor improvement in standard deviations and corner detection rates, both attributable to the ChArUco target, which allows frames with only partially detected corners to be used for calibration. Replacing OpenCV with Kalibr as backend, while retaining the checkerboard target, yields the lowest reprojection error among motion-based references, likely due to Kalibr’s advanced optimization routines. Despite investigating these variations, our method reduces the mean reprojection error by 44% relative to the best motion-based approach. This gap can be attributed to motion blur in the frame-based observations, together with the absence of strict hardware-level temporal synchronization. Compared to the static baseline of Plasberg et al., our method still achieves a 6% reduction in reprojection error.

As the results demonstrate, relying on a temporally modulated blended target substantially reduces hardware dependencies and deployment effort without sacrificing calibration accuracy. Our approach achieves superior accuracy compared to both motion-based and non-motion-based approaches.

![](images/da969d18850826f5c8ed30e06e7654e6fc07833f05698d4444b4456cffd889e6.jpg)  
Figure 4: Geometric transformation pipeline for eye-to-hand calibration, illustrating the mapping of a target point $\mathbf { p _ { t } }$ from the ChArUco target frame (t) to the event camera frame (e) via ${ \bf e _ { T _ { t } } } .$ , and subsequently to the robot base frame (B) using ${ } ^ { B } \mathbf { T } _ { \mathrm { e } }$ , with the final relation to the TCP defined through the robot kinematics $\mathrm { t c p } _ { \mathbf { T } _ { B } }$

Table 8: Hand-eye calibration results aggregated over 20 valid runs. The table reports the mean (µ) and standard deviation (σ) for the estimated translation vector $\mathbf { \Omega } ( \mathbf { t } = \left( t _ { x } , t _ { y } , t _ { z } \right) )$ between the event camera and robot base, its Euclidean norm $( \lVert \mathbf { t } \rVert )$ , and the rotation magnitude (θ) obtained from the rotation matrix R via axis-angle representation.
<table><tr><td>Statistic</td><td> $t _ { x } [ \mathrm { m } ]$ </td><td> $t _ { y } [ \mathrm { m } ]$ </td><td> $t _ { z } ~ [ \mathrm { m } ]$ </td><td>||||[m]</td><td>θ[°]</td></tr><tr><td> $\pi _ { \mathbf { T } _ { \mathrm { e } } }$ </td><td>0.1258</td><td>-0.5678</td><td>0.3839</td><td>0.6970</td><td>173.76</td></tr><tr><td>(μ) (σ)</td><td>0.0093</td><td>0.0076</td><td>0.0061</td><td>0.0056</td><td>2.80</td></tr></table>

## 4.4 Case Study: Eye-to-Hand Calibration

To demonstrate that the proposed approach extends beyond cross-modal stereo calibration, we evaluate its applicability to downstream event-camera-based tasks. As a representative use case, we consider eye-to-hand calibration, depicted in Fig. 4. We show that combining the ChArUco-based target with temporally modulated pattern blending enables reliable geometric estimation even under partial occlusions by the robot actuator (Appendix C).

To this end, we perform a spatial consistency analysis by selecting the event camera frame (e) as the primary reference. The transformation $\mathbf { \dot { \mathbf { \mathit { B } } } } \mathbf { T _ { \mathrm { e } } } \in S E ( 3 )$ , mapping points from the event camera frame to the robot base frame (B), is obtained via hand-eye calibration [25] in an eye-to-hand configuration with a static camera and a movable calibration target. The ground-truth transformation was obtained from the CAD model of the experimental setup and additionally verified through manual measurements.

The estimated translation components, shown in Tab. 8, are in close agreement with the ground truth $( t _ { x } = 0 . 1 2 5 \mathrm { m }$ $t _ { y } = - 0 . 5 6 3 \mathrm { m }$ $t _ { z } = 0 . 3 8 2 \mathrm { m } )$ . Furthermore, the low standard deviation across all components indicates a stable estimation of the transformation parameters. Overall, the obtained results provide robust geometric alignment suitable for standard vision-guided robotic manipulation tasks. The estimated rotation is also consistent with the experimental setup, as the camera is approximately facing downward toward the robot base.

Target-to-TCP Consistency Evaluation. To assess the spatial consistency of the estimated transformations, we evaluate the stability of a reference point attached to the calibration target across multiple robot poses. Specifically, we consider a ChArUco corner with fixed identifier (ID9), which is reliably detectable (unobstructed) in all observations.

Table 9: Spatial consistency evaluation results averaged over $^ { 5 }$ runs, with each run consisting of 22 diverse poses for the target-to-TCP evaluation and 6 trajectories for the base-frame distance evaluation. The table reports the mean Euclidean distance $( \overline { { d } } )$ between the transformed calibration point (ID9) and the TCP, its standard deviation $( \sigma _ { d } )$ , the ground truth distance $( d _ { \mathrm { g t } } )$ , and the mean absolute error (MAE) with respect to the ground truth.
<table><tr><td rowspan=1 colspan=1>Evaluation Mode</td><td rowspan=1 colspan=1> $\overline { { d } }$ [mm]</td><td rowspan=1 colspan=1> $\sigma _ { d }$ [mm] ↓</td><td rowspan=1 colspan=1> $d _ { \mathrm { g t } }$ [mm]</td><td rowspan=1 colspan=1>MAE [mm] ↓</td></tr><tr><td rowspan=1 colspan=1>Target-to-TCPBase-Frame Distance</td><td rowspan=1 colspan=1>67.09Scene dep.</td><td rowspan=1 colspan=1>1.860.66</td><td rowspan=1 colspan=1>67.80Scene dep.</td><td rowspan=1 colspan=1>0.711.40</td></tr></table>

For each pose, the point is first reconstructed in the event camera frame as $\mathbf { p _ { e } } \in \mathbb { R } ^ { 3 }$ using pose estimation [16]. It is then transformed into the robot base frame using the previously estimated transformation:

$$
\mathbf { p } _ { B } = { ^ { B } } \mathbf { T } _ { \mathrm { e } } \mathbf { p } _ { \mathrm { e } } ,\tag{3}
$$

where $^ { B } \mathbf { T } _ { \mathrm { e } } \in S E ( 3 )$ denotes the previously established transformation. Subsequently, the point is expressed in the tool center point (TCP) frame using the known forward kinematics:

$$
\begin{array} { r } { \mathbf { p } _ { \mathrm { t c p } } = \mathrm { t c p } _ { \mathbf { T } _ { B } \mathbf { p } _ { B } , } } \end{array}\tag{4}
$$

where $\mathrm { t c p } { \bf T } _ { B } = ( ^ { B } { \bf T } _ { \mathrm { t c p } } ) ^ { - 1 }$ is obtained from the robot model. Finally, the Euclidean deviation between the transformed point and the TCP origin is computed:

$$
d = \left\| \mathbf { p } _ { \mathrm { t c p } } \right\| _ { 2 } .\tag{5}
$$

For a consistent calibration, the distance d should remain approximately constant across robot poses, since the relative geometry between the calibration target and gripper is fixed. Deviations from constancy indicate inaccuracies in the overall transformation chain, including errors in the hand-eye calibration, intrinsic camera parameters, and pose estimation.

The first row in Tab. 9 summarizes the results of the proposed spatial consistency evaluation. The estimated mean distance $\overline { { d } }$ closely matches the ground truth distance $d _ { \mathrm { g t } }$ , resulting in a low mean absolute error of 0.71 mm, corresponding to a relative error of approximately 1.05%. Furthermore, the small standard deviation $\sigma _ { d }$ indicates high consistency across different robot poses. These results suggest that the proposed calibration approach yields both accurate and stable geometric alignment, with minimal pose-dependent variation.

Base-Frame Distance Consistency Evaluation. As a complementary validation, we evaluate distance preservation in the robot base frame using the event camera as the reference. The target point (ID9) is first reconstructed in the event camera frame as $\mathbf { p _ { e } } \in \mathbb { R } ^ { 3 }$ using pose estimation [16] and transformed into the robot base frame via the transformation $s _ { \mathbf { T } _ { \mathrm { e } } }$ obtained via eye-to-hand calibration.

For two target points $\mathbf { p } _ { \mathrm { e } , 1 } , \mathbf { p } _ { \mathrm { e } , 2 } \in \mathbb { R } ^ { 3 }$ with known distance $d _ { \mathrm { g t } }$ , the corresponding base frame distance is computed as

$$
d _ { B } = \left\| { } ^ { | B } \mathbf { T } _ { \mathrm { e } } \mathbf { p } \mathrm { e } , 1 - { } ^ { | B } \mathbf { T } _ { \mathrm { e } } \mathbf { p } \mathrm { e } , 2 \right\| _ { 2 } .\tag{6}
$$

The deviation from the known distance is defined as

$$
e = \left| d _ { B } - d _ { \mathrm { g t } } \right| .\tag{7}
$$

This evaluation is performed over 6 scenarios. For a geometrically consistent calibration, $d _ { B }$ is expected to closely match $d _ { \mathrm { g t } }$ across all observations. The results in the second row of Tab. 9 indicate that the base-frame distance consistency exhibits a low standard deviation $\sigma _ { d } .$ , suggesting stable distance preservation across different trajectories. The low MAE further implies that the reconstructed distances deviate only slightly from the expected values despite being scene-dependent. These findings demonstrate that the proposed calibration approach reliably estimates the intrinsic parameters and provides a robust foundation for downstream tasks such as eye-to-hand calibration, resulting in stable and consistent transformation pipelines.

## 5 Limitations

Display Properties. Despite the robustness of the proposed framework, its performance remains sensitive to the temporal characteristics of the display hardware. Specifically, event density scales with the refresh rate, potentially degrading marker detection accuracy below 30 Hz. However, given that standard modern display hardware operates well above this range, this dependency poses a minimal constraint for real-world deployment. Furthermore, our proposed method assumes the use of a display with viewing-angle-stable contrast characteristics, as displays with strong angular dependence can exhibit reduced effective contrast modulation at large viewing angles. This particularly affects event cameras, which rely on intensity changes to register events.

Event Representation. An additional limitation arises from the conversion of asynchronous event streams into discretized image representations. In the employed accumulation of events over time intervals, noise and temporal inconsistencies around high-contrast structures may introduce jitter, reducing corner localization accuracy and increasing reprojection error. More advanced reconstruction strategies, e.g., E2VID [21] or optimized accumulation schemes as in Wang et al. [28], could potentially mitigate these effects.

## 6 Conclusion

In this paper, we presented a practical stereo calibration approach for hybrid event and framebased systems using a temporally modulated, blended ChArUco target rendered framesequentially on standard commercial displays. By triggering events while ensuring the pattern remains continuously detectable in the frame domain, this method enables static acquisition and minimizes reliance on complex motion, reconstruction, or precise hardware synchronization. Our experiments demonstrate robust feature detection, as well as stable intrinsic and extrinsic estimation across diverse display settings, viewing angles, varying illumination, and in both robot-assisted and handheld capture. Furthermore, our approach successfully transfers to downstream eye-to-hand calibration, supporting consistent geometric estimation under practical conditions. Future work will explore improved event accumulation techniques to mitigate limitations related to display temporal characteristics and event discretization.

## Acknowledgements

In part, this research was funded by the AI-TRAQC Project – Regional Innovation Center Artificial Intelligence Training & Qualification Campus (RegioWIN 2472019), supported by the European Regional Development Fund (Europäischer Fonds für Regionale Entwicklung - EFRE).

## References

[1] Andrea Bertogalli, Giacomo Boracchi, and Luca Magri. One target to align them all: LiDAR, RGB and event cameras extrinsic calibration for Autonomous Driving. arXiv, November 2025. URL http://arxiv.org/abs/2511.12291. arXiv:2511.12291 [cs].

[2] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal ofSoftware Tools, 2000. URL https://github.com/opencv/opencv.

[3] Bolin Cai, Ami Zi, Jun Yang, Guoliang Li, Yang Zhang, Qiujie Wu, Chenen Tong, Wenxiang Liu, and Xiangcheng Chen. Accurate Event Camera Calibration With Fourier Transform. IEEE Transactions on Instrumentation and Measurement, 73:1– 12, 2024. ISSN 0018-9456, 1557-9662. doi: 10.1109/TIM.2024.3470063. URL https://ieeexplore.ieee.org/document/10699363/.

[4] Guang Chen, Wenkai Chen, Qianyi Yang, Zhongcong Xu, Longyu Yang, Jörg Conradt, and Alois Knoll. A Novel Visible Light Positioning System With Event-Based Neuromorphic Vision Sensor. IEEE Sensors Journal, 20(17):10211–10219, September 2020. ISSN 1530-437X, 1558-1748, 2379-9153. doi: 10.1109/JSEN.2020.2990752. URL https://ieeexplore.ieee.org/document/9079552/.

[5] Shuolong Chen, Xingxing Li, Liu Yuan, and Ziao Liu. eKalibr: Dynamic Intrinsic Calibration for Event Cameras From First Principles of Events. IEEE Robotics and Automation Letters, 10(7):7094–7101, July 2025. ISSN 2377-3766, 2377-3774. doi: 10.1109/LRA.2025.3572777. URL https://ieeexplore.ieee.org/docu ment/11012137/.

[6] Mathieu Cocheteux, Julien Moreau, and Franck Davoine. MULi-Ev: Maintaining Unperturbed LiDAR-Event Calibration. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 4579–4586, Seattle, WA, USA, June 2024. IEEE. ISBN 979-8-3503-6547-4. doi: 10.1109/CVPRW63382.202 4.00460. URL https://ieeexplore.ieee.org/document/10678155/.

[7] Christian Creß, Erik Schütz, Bare Luka Žagar, and Alois C. Knoll. Targetless Extrinsic Calibration Between Event-Based and RGB Camera for Intelligent Transportation Systems. In 2023 IEEE Intelligent Vehicles Symposium (IV), pages 1–8, Anchorage, AK, USA, June 2023. IEEE. ISBN 979-8-3503-4691-6. doi: 10.1109/IV55152.2023.1 0186538. URL https://ieeexplore.ieee.org/document/10186538/.

[8] Christian Creß, Walter Zimmer, Nils Purschke, Bach Ngoc Doan, Sven Kirchner, Venkatnarayanan Lakshminarasimhan, Leah Strand, and Alois C. Knoll. TUMTraf Event: Calibration and Fusion Resulting in a Dataset for Roadside Event-Based and

RGB Cameras. IEEE Transactions on Intelligent Vehicles, 9(7):5186–5203, July 2024. ISSN 2379-8904, 2379-8858. doi: 10.1109/TIV.2024.3393749. URL https://ieeexplore.ieee.org/document/10508494/.

[9] Manuel J. Dominguez-Morales, Angel Jimenez-Fernandez, Gabriel Jimenez-Moreno, Cristina Conde, Enrique Cabello, and Alejandro Linares-Barranco. Bio-Inspired Stereo Vision Calibration for Dynamic Vision Sensors. IEEE Access, 7:138415–138425, 2019. ISSN 2169-3536. doi: 10.1109/ACCESS.2019.2943160. URL https: //ieeexplore.ieee.org/document/8846702/.

[10] Etienne Dubeau, Mathieu Garon, Benoit Debaque, Raoul de Charette, and Jean-François Lalonde. RGB-D-E: Event Camera Calibration for Fast 6-DOF Object Tracking. In 2020 IEEE International Symposium on Mixed and Augmented Reality (IS-MAR), pages 127–135, Porto de Galinhas, Brazil, November 2020. IEEE. ISBN 978-1-7281-8508-8. doi: 10.1109/ISMAR50242.2020.00034. URL https: //ieeexplore.ieee.org/document/9284657/.

[11] Guillermo Gallego, Tobi Delbrück, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew J. Davison, Jörg Conradt, Kostas Daniilidis, and Davide Scaramuzza. Event-Based Vision: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1):154–180, January 2022. ISSN 0162-8828, 2160-9292, 1939-3539. doi: 10.1109/TPAMI.2020.3008413. URL https://ieeexplore.ieee.org/document/9138762/.

[12] S. Garrido-Jurado, R. Muñoz-Salinas, F.J. Madrid-Cuevas, and M.J. Marín-Jiménez. Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition, 47(6):2280–2292, June 2014. ISSN 00313203. doi: 10.1016/j.pa tcog.2014.01.005. URL https://linkinghub.elsevier.com/retrieve /pii/S0031320314000235.

[13] Thomas Gossard, Andreas Ziegler, Levin Kolmar, Jonas Tebbe, and Andreas Zell. eWand: An extrinsic calibration framework for wide baseline frame-based and eventbased camera systems. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 14534–14540, Yokohama, Japan, May 2024. IEEE. ISBN 979-8-3503-8457-4. doi: 10.1109/ICRA57147.2024.10610116. URL https: //ieeexplore.ieee.org/document/10610116/.

[14] Rui Hu, Jürgen Kogler, Margrit Gelautz, Min Lin, and Yuanqing Xia. A Dynamic Calibration Framework for the Event-Frame Stereo Camera System. IEEE Robotics and Automation Letters, 9(12):11465–11472, December 2024. ISSN 2377-3766, 2377- 3774. doi: 10.1109/LRA.2024.3491426. URL https://ieeexplore.ieee.or g/document/10742513/.

[15] Kun Huang, Yifu Wang, and Laurent Kneip. Dynamic Event Camera Calibration. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7021–7028, Prague, Czech Republic, September 2021. IEEE. ISBN 978-1-6654- 1714-3. doi: 10.1109/IROS51168.2021.9636398. URL https://ieeexplore.i eee.org/document/9636398/.

[16] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua. EPnP: An Accurate O(n) Solution to the PnP Problem. International Journal of Computer Vision, 81(2):155– 166, February 2009. ISSN 0920-5691, 1573-1405. doi: 10.1007/s11263-008-0152-6. URL http://link.springer.com/10.1007/s11263-008-0152-6.

[17] Ngoc Trung Mai, Ren Komatsu, Hajime Asama, and Atsushi Yamashita. Pose Estimation for Event Camera Using Charuco Board Based on Image Reconstruction. In 2023 IEEE/SICE International Symposium on System Integration (SII), pages 1–6, Atlanta, GA, USA, January 2023. IEEE. ISBN 979-8-3503-9868-7. doi: 10.1109/SII55687.2 023.10039168. URL https://ieeexplore.ieee.org/document/10039 168/.

[18] Manasi Muglikar, Mathias Gehrig, Daniel Gehrig, and Davide Scaramuzza. How to Calibrate Your Event Camera. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1403–1409, Nashville, TN, USA, June 2021. IEEE. ISBN 978-1-6654-4899-4. doi: 10.1109/CVPRW53098.2021.00155. URL https://ieeexplore.ieee.org/document/9523174/.

[19] C. Plasberg, M. Grosse Besselmann, A. Roennau, and R. Dillmann. Intrinsic and Extrinsic Calibration Method for a Trinocular Multimodal Camera Setup. In 2022 25th International Conference on Information Fusion (FUSION), pages 1–6, Linköping, Sweden, July 2022. IEEE. ISBN 978-1-7377497-2-1. doi: 10.23919/FUSION49751.2022. 9841284. URL https://ieeexplore.ieee.org/document/9841284/.

[20] Thomas Porter and Tom Duff. Compositing digital images. ACM SIGGRAPH Computer Graphics, 18(3):253–259, July 1984. ISSN 0097-8930. doi: 10.1145/964965.8 08606. URL https://dl.acm.org/doi/10.1145/964965.808606.

[21] Henri Rebecq, René Ranftl, Vladlen Koltun, and Davide Scaramuzza. High Speed and High Dynamic Range Video with an Event Camera. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(6):1964–1980, June 2021. ISSN 0162-8828, 2160-9292, 1939-3539. doi: 10.1109/TPAMI.2019.2963386. URL https://ieee xplore.ieee.org/document/8946715/.

[22] Mohammed Salah, Abdulla Ayyad, Muhammad Humais, Daniel Gehrig, Abdelqader Abusafieh, Lakmal Seneviratne, Davide Scaramuzza, and Yahya Zweiri. E-Calib: A Fast, Robust, and Accurate Calibration Toolbox for Event Cameras. IEEE Transactions on Image Processing, 33:3977–3990, 2024. ISSN 1057-7149, 1941-0042. doi: 10.110 9/TIP.2024.3410673. URL https://ieeexplore.ieee.org/document/1 0555516/.

[23] Orla Sealy Phelan, Dara Molloy, Roshan George, Edward Jones, Martin Glavin, and Brian Deegan. EPCNet: Implementing an ‘Artificial Fovea’ for More Efficient Monitoring Using the Sensor Fusion of an Event-Based and a Frame-Based Camera. Sensors, 25(15):4540, July 2025. ISSN 1424-8220. doi: 10.3390/s25154540. URL https://www.mdpi.com/1424-8220/25/15/4540.

[24] Rihui Song, Zhihua Jiang, Yanghao Li, Yunxiao Shan, and Kai Huang. Calibration of Event-based Camera and 3D LiDAR. In 2018 WRC Symposium on Advanced Robotics and Automation (WRC SARA), pages 289–295, Beijing, August 2018. IEEE. ISBN

978-1-5386-7675-2. doi: 10.1109/WRC-SARA.2018.8584215. URL https: //ieeexplore.ieee.org/document/8584215/.

[25] R.Y. Tsai and R.K. Lenz. A new technique for fully autonomous and efficient 3D robotics hand/eye calibration. IEEE Transactions on Robotics and Automation, 5(3): 345–358, June 1989. ISSN 1042296X. doi: 10.1109/70.34770. URL http://ieee xplore.ieee.org/document/34770/.

[26] Krishna Vinod, Prithvi Jai Ramesh, Pavan Kumar B N, and Bharatesh Chakravarthi. Sebvs: Synthetic event-based visual servoing for robot navigation and manipulation. arXiv, 2025. URL https://arxiv.org/abs/2508.17643. 10.48550/arXiv.2508.17643.

[27] Shaoan Wang, Zhanhua Xin, Yaoqing Hu, Dongyue Li, Mingzhu Zhu, and Junzhi Yu. EF-Calib: Spatiotemporal Calibration of Event- and Frame-Based Cameras Using Continuous-Time Trajectories. IEEE Robotics and Automation Letters, 9(11):10280– 10287, November 2024. ISSN 2377-3766, 2377-3774. doi: 10.1109/LRA.2024.34744 75. URL http://arxiv.org/abs/2405.17278. arXiv:2405.17278 [cs].

[28] Yongqing Wang, Shiyu He, Yufan Fei, and Xingjian Liu. Motion-error-free calibration of event camera systems using a flashing target. Optics Express, 32(15):26833–26845, July 2024. ISSN 1094-4087. doi: 10.1364/OE.529263. URL https://opg.opti ca.org/oe/abstract.cfm?uri=oe-32-15-26833.

[29] Shijie Zhang, Yuxuan Huang, Xuan Pei, Haopeng Lin, Wenwen Zheng, Wendi Wang, and Taogang Hou. An Improved LED Aruco-Marker Detection Method for Event Camera. In Haonan Chen, Pingyi Fan, and Lipo Wang, editors, Communications, Networking, and Information Systems, pages 47–57, Singapore, 2023. Springer Nature. ISBN 978-981-99-3581-9. doi: 10.1007/978-981-99-3581-9\_3.

[30] Yibo Zhang, Zhucheng Tan, Jing Jin, Chenyang Shi, and Yuqing He. Spatiotemporal alignment in hybrid vision systems: unifying event cameras and frame-based cameras. In Automated Visual Inspection and Machine Vision VI, volume 13572, pages 76–89. SPIE, August 2025. doi: 10.1117/12.3068939. URL https://www.spiedigita llibrary.org/conference-proceedings-of-spie/13572/135720 A/Spatiotemporal-alignment-in-hybrid-vision-systems--uni fying-event-cameras/10.1117/12.3068939.full.

[31] Ami Zi, Siyuan Tan, Qiujie Wu, Xiangcheng Chen, and Bolin Cai. Flexible Event Camera Calibration With Blinking Binary Stripes. IEEE Photonics Technology Letters, 36(23):1357–1360, December 2024. ISSN 1041-1135, 1941-0174. doi: 10.1109/LPT. 2024.3477614. URL https://ieeexplore.ieee.org/document/10713 438/.

## A Experimental Setup

![](images/601f7e750118bece1006fd46df585b5ef3d68a217d87ac643f228d7fb33bcc97.jpg)  
Figure 5: The hybrid frame-event stereo rig is mounted on a UR5e robotic arm to ensure precise trajectory repeatability. The Dell S2722QC monitor displays the calibration pattern, serving as the cross-modal target for both the frame-based and event sensors.

Table 10: IDS uEye frame-based camera configuration used in all experiments. The gamma value is specified in the camera parameterization scheme, where a value of 100 corresponds to a linear gamma of $\gamma = 1 . 0$ . All parameters not explicitly listed were kept at their hardware default settings.
<table><tr><td>Parameter UI-5250CP</td><td>Value</td></tr><tr><td>Exposure Time Gain Frame Rate</td><td>100.0ms 100 10.0fps</td></tr></table>

Table 11: Prophesee EVK4 sensor configuration used in all experiments. Bias values denote relative offsets from the hardware defaults and were tuned to optimize the signal-to-noise ratio for flickering pattern detection. All parameters not explicitly listed were kept at their hardware default settings.
<table><tr><td rowspan=1 colspan=1>Parameter EVK4</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Event Trail FilterFiltering TypeTrail Threshold</td><td rowspan=1 colspan=1>EnabledSTC_CUT_TRAIL100000</td></tr><tr><td rowspan=1 colspan=1>bias_diff_offbias_diff_onbias_fobias_hpf</td><td rowspan=1 colspan=1>-15-35-3580</td></tr></table>

![](images/d7e15b1b945268455191b696e0b3e34cb45e53d51c5927688eaddedab9ebafbb.jpg)

## B Ablation Illumination Stability

![](images/b7670d8c14ec47768006a8f2059ee382e68ef0dc9ba0b76fa308febe5d3a7465.jpg)  
(a) RGB frame

![](images/e4e1b8e11e5b75f6998e63161f1c200b51bef809f04bbc841f2c610b504498cd.jpg)  
(b) Event frame  
Figure 6: Qualitative example from the illumination robustness experiment. Both images depict the same scene. Differences in perspective are due to the stereo baseline offset between the frame-based and event camera sensors. (a) RGB image frame captured under strong external spotlight interference, causing overexposure on the calibration target. (b) Corresponding event frame recorded under the same conditions, illustrating the effect of intense illumination on event generation and pattern visibility.

Table 12: Illumination robustness evaluation of the intrinsic parameters with our approach over 5 runs with varying spotlight positions. The table reports the corner detection success rate $\left( u _ { c } \right)$ , focal lengths $( f _ { x } , f _ { y } )$ , principal point $( c _ { x } , c _ { y } )$ , and reprojection error $( \varepsilon _ { \mathrm { r e p } } )$
<table><tr><td rowspan=1 colspan=1>Sensor</td><td rowspan=1 colspan=1> $u _ { c }$ [%]↑</td><td rowspan=1 colspan=1> $f _ { x }$ [px]</td><td rowspan=1 colspan=1> $f _ { y }$ [px]</td><td rowspan=1 colspan=1> $c _ { x }$ [px]</td><td rowspan=1 colspan=1> $c _ { y } \left[ \mathfrak { p } \mathbf { x } \right]$ </td><td rowspan=1 colspan=1> $\varepsilon _ { \mathrm { r e p } }$ [px] ↓</td></tr><tr><td rowspan=1 colspan=1>RGB    (μ)(σ)</td><td rowspan=1 colspan=1>100.00.0</td><td rowspan=1 colspan=1>2764.82.4</td><td rowspan=1 colspan=1>2766.12.3</td><td rowspan=1 colspan=1>797.15.0</td><td rowspan=1 colspan=1>449.51.8</td><td rowspan=1 colspan=1>0.1400.003</td></tr><tr><td rowspan=1 colspan=1>Event   (µ)(σ)</td><td rowspan=1 colspan=1>98.53.4</td><td rowspan=1 colspan=1>2568.911.4</td><td rowspan=1 colspan=1>2569.611.6</td><td rowspan=1 colspan=1>600.59.3</td><td rowspan=1 colspan=1>367.810.2</td><td rowspan=1 colspan=1>0.4560.007</td></tr></table>

## C Eye-To-Hand Calibration Occlusions

(a) Strong Occlusion  
![](images/8b4112def0d20a1969fc6b030b5e62cb1cd88fbd3351fad107abdce3d8785b15.jpg)  
(b) Light Occlusion  
Figure 7: Robustness of the proposed method under occlusion during eye-to-hand calibration. Even in the presence of partial occlusions caused by the robot actuator, salient ChArUco corners (highlighted in red) are reliably detected, enabling accurate calibration.