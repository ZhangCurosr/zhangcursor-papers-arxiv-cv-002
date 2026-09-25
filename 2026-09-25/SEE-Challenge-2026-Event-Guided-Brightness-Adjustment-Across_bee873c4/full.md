# SEE Challenge 2026: Event-Guided Brightness Adjustment Across a Broad Illumination Range

Yunfan Lu<sup>∗,1</sup>, Mingchao Xu<sup>∗,1</sup>, Hanyu Zhou<sup>2</sup>, Shaoyu Liu<sup>3,4</sup>, Haoyue Liu<sup>5</sup>, Peiqi Duan<sup>6</sup>, Shihan Peng<sup>5</sup>, Yinqiang Zheng<sup>7</sup>, Boxin Shi<sup>6</sup>, Gim Hee Lee<sup>2</sup>, Hui Xiong<sup>1</sup>, Davide Scaramuzza<sup>8</sup>

<sup>1</sup>Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>National University6 of Singapore <sup>3</sup>Xidian University <sup>4</sup>Tsinghua University <sup>5</sup>Huazhong University of2 Science and Technology <sup>6</sup>Peking University <sup>7</sup>The University of Tokyo <sup>8</sup>Robotics and0 Perception Group, University of Zurich

\* denotes equal contribution

Abstract. Event cameras provide a high dynamic range and preserve brightness-change cues in lighting conditions where conventional RGB frames may be noisy or saturated. To benchmark event-guided restoration across a broad illumination range, we organized the SEE Challenge 2026 with the Event-Based Multimodal Vision Workshop at ECCV 2026. The task conditions restoration on one or more RGB frames, synchronized events, and a scalar target-brightness statistic provided by the organizers. It uses SEE-600K, which contains 610,126 image-event observations from 202 real-world scenes spanning low-light, normal-light, and high-light conditions with illumination variations of up to 1,000×. The challenge follows an open-system protocol: participants may use diferent temporal contexts, architectures, pretrained weights, test-time augmentation, and post-processing strategies. PSNR determines the ranking, and SSIM is reported as a secondary metric. Around 70 teams registered interest and 15 valid CodaBench submissions were received. Six distinct teams completed organizer-side identity and technical verification, provided method descriptions, checkpoints, inference code, and instructions, and are included in the verified open-system ranking reported here. Beyond the ranking, this report analyzes exposure subsets, semantically distinct test cases, a shared failure pattern, system design choices, and inference strategies. The top systems obtain closely spaced average scores, while the best-performing method varies across cases and metrics; under severe underexposure, all verified systems retain visible local errors. More details are available at SEE Challenge 2026.

## 1 Introduction

Visual systems often encounter illumination changes that exceed the dynamic range of conventional cameras [12]. Mobile robots may move from dim indoor spaces to sunlit entrances, while autonomous vehicles may encounter tunnels, reflective roads, and bright headlights within the same sequence [6, 17]. Under these conditions, dark regions can be dominated by noise and bright regions can become saturated. Once useful measurements have been severely degraded, recovering faithful structure and color from an RGB image alone becomes dificult.

![](images/c9d6b8895d61c7596878782e971a3967977abe657f9466e7f14b06a7376ec9fe.jpg)  
Fig. 1: Adapted from SEE-Net [15]. Brightness distributions and task settings of SDE [4] and SEE-600K [15]. (a) SDE covers a low-to-normal brightness range. (b) SEE-600K spans low-light, normal-light, and high-light conditions. (c) Earlier lowlight methods map dark inputs to a fixed normal-light target. (d) The oficial SEE Challenge evaluation conditions restoration on a target-brightness statistic B derived from the reference image.

This problem has motivated extensive work in computational photography and image restoration [8, 9, 16, 20]. Multi-exposure high dynamic range imaging combines observations captured at diferent exposure levels, but remains sensitive to scene motion and alignment [10, 11, 18]. Learning-based approaches address low-light enhancement, exposure correction, RAW reconstruction, and image signal processing using learned priors [1,2,5,19]. Nevertheless, a frame-based method must infer missing information from measurements that may already be clipped or noise-dominated [7].

Event cameras asynchronously record per-pixel brightness changes and provide complementary measurements [3]. Their high temporal resolution and wide dynamic range can preserve contrast and motion cues when conventional frames are degraded. Events do not directly provide dense color or absolute brightness, so restoration requires the fusion of RGB appearance, event changes, and a definition of the desired target exposure. Previous event-guided low-light studies, including EvLowLight [13] and EvLight [4], have provided experimental evidence that event measurements can complement RGB inputs in dificult illumination. SEE-Net [15] subsequently extended this setting from fixed low-tonormal enhancement to brightness adjustment across low, normal, and bright exposures. Together, these studies motivate a benchmark that evaluates eventguided restoration across both underexposed and overexposed inputs.

Broad-range brightness adjustment couples several open design choices. A complete system must represent asynchronous events, fuse them with RGB appearance, condition the desired brightness, and decide how much temporal context to use. A shared challenge makes these choices observable under one data and evaluation protocol and provides reference points for subsequent research. Organizing the challenge with the Event-Based Multimodal Vision Workshop also creates a focused forum for the event-vision community to compare system designs, discuss reproducibility, and identify limitations that are dificult to see from a single baseline.

The SEE Challenge 2026 extends evaluation on SEE-600K [15] beyond a single baseline. It supports a community study of independent event-guided restoration systems on newly recorded hidden sequences. Around 70 teams registered interest, 15 accounts produced valid Phase 2 submissions, and six distinct teams completed organizer-side technical verification. Their systems cover crossattention, feature modulation, state-space models, latent flow, temporal averaging, cascaded restoration, and test-time augmentation. This diversity allows the report to document both current performance and the practical trade-ofs between accuracy, temporal context, model size, and inference multiplicity.

This report makes four organizer-side contributions. First, it defines the task and documents data construction, event–frame synchronization, the targetbrightness statistic, and the open-system evaluation protocol. Second, it reports a verified ranking based on organizer-run dense evaluation of the six systems for which checkpoints, code, and instructions were provided. Third, it analyzes results across exposure settings, motion and texture cases, and a shared failure example using pixel-level, structural, and perceptual metrics. Finally, it compares the architectural, conditioning, temporal, and inference choices of the submitted systems and identifies priorities for future challenge editions. Because the competition permits diferent temporal contexts, pretrained weights, TTA, cascades, and post-processing, the ranking evaluates complete systems and does not isolate the contribution of any single component.

## 2 Challenge Protocol

## 2.1 Task Definition and Open-System Scope

For an evaluation sample, let $\mathbf { I } _ { \mathrm { i n } } = \{ I _ { \mathrm { i n } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ denote one or more input RGB frames and $\mathbf { E } = \{ E ^ { ( k ) } \} _ { k = 1 } ^ { K }$ their associated event data. The organizers provide a scalar target-brightness statistic B for the target reference. A submitted system predicts

$$
\hat { I } _ { \mathrm { o u t } } = { \cal F } ( { \bf I } _ { \mathrm { i n } } , { \bf E } , B ) ,\tag{1}
$$

and the prediction is compared with the hidden reference $I _ { \mathrm { g t } }$ . The evaluated mappings are Low-to-Normal and High-to-Normal restoration. The former brightens an underexposed input and recovers visible structure, whereas the latter reduces the brightness of an overexposed input and reconstructs clipped or weakened detail.

The challenge is an ofline open-system evaluation. No maximum value of K or maximum temporal span was imposed, and participants were permitted to use both preceding and future adjacent frames from the provided sequence.

Table 1: Data used by the challenge. Pair counts refer to constructed source–target restoration pairs rather than distinct raw frames. The three splits are scene-disjoint.
<table><tr><td>Split</td><td>Scenes</td><td>Pairs</td><td>Reference</td><td>Purpose</td><td>Motion setting</td></tr><tr><td>Training split</td><td>165</td><td>509,132</td><td>Public</td><td>Training</td><td>Camera motion</td></tr><tr><td>Test split</td><td>37</td><td>147,368</td><td>Public</td><td>Phase 1 validation</td><td>Camera motion</td></tr><tr><td>Hidden final split</td><td>6</td><td>3,405</td><td>Hidden</td><td>Phase 2 validation</td><td>Camera and scene motion</td></tr></table>

Participants could also choose their event representation, fusion strategy, training objective, and inference pipeline. No constrained single-frame or single-pass track was defined retrospectively; temporal context and inference multiplicity are therefore disclosed as system properties rather than controlled variables.

## 2.2 Dataset Splits and Pair Construction

SEE-600K contains 610,126 image-event observations from 202 released realworld scenes [15]. Each scene contains, on average, four recordings of the same environment and robotic-arm trajectory under diferent light-transmission, aperture, and exposure settings. Neutral-density filters with diferent transmission levels create low-light, normal-light, and high-light recordings while the Universal Robots UR5e repeats the same trajectory. After temporal registration, frames at corresponding trajectory positions are paired across exposure recordings. An underexposed or overexposed frame can therefore be used as the source and a temporally corresponding normal-light frame as the target. When a scene contains multiple eligible source or normal-light recordings, more than one restoration mapping can be constructed from the same underlying image-event observations. This is why the number of constructed restoration pairs difers from the number of distinct image-event observations.

The 165-scene training split and 37-scene released test split together form the 202 released SEE-600K scenes and yield 656,500 constructed restoration pairs. The released references were available to all participants and supported training and Phase 1 validation. The final evaluation used six additional newly recorded scene groups. Only the input RGB data, events, metadata, and target-brightness statistic were released for these groups; their normal-light references remained hidden.

## 2.3 Event–Frame Synchronization and Data Format

The data were recorded with a color DAVIS346 sensor at a spatial resolution of $3 4 6 \times 2 6 0$ pixels. APS frames, events, and IMU measurements share the sensor timestamp domain. Let $t _ { \exp } ^ { ( k ) }$ denote the exposure-start timestamp of RGB frame k. The organizer-provided event interval associated with that frame is

$$
E ^ { ( k ) } = \{ e _ { i } = ( x _ { i } , y _ { i } , t _ { i } , p _ { i } ) \mid t _ { \exp } ^ { ( k ) } \leq t _ { i } < t _ { \exp } ^ { ( k + 1 ) } \} .\tag{2}
$$

The half-open interval prevents an event at a boundary from being assigned twice. The released data contain 8-bit RGB PNG frames and timestamped event arrays with coordinates and polarity; the raw recordings were originally stored in AEDAT4 format together with IMU measurements. The challenge does not prescribe a voxelization or event-window aggregation strategy. Participants may transform the raw events into any representation and may combine events from multiple adjacent frame intervals when using temporal context.

Repeated robotic-arm trajectories are temporally registered using the synchronized 1-kHz IMU signals [15]. The dataset paper evaluates spatial alignment by matching SIFT features with FLANN, estimating an afine transformation with RANSAC, applying the transformation across the image, and averaging the resulting pixel displacement. The reported mean displacement is 0.2967 pixels, with a temporal registration error of at most approximately 1 ms [15]. These values characterize the acquisition and registration procedure.

## 2.4 Target-Brightness Statistic

For each hidden reference image $I _ { \mathrm { g t } } \in [ 0 , 1 ] ^ { H \times W \times 3 }$ , the target-brightness statistic is the arithmetic mean over all pixels and RGB channels:

$$
B = \frac { 1 } { 3 H W } \sum _ { y = 1 } ^ { H } \sum _ { x = 1 } ^ { W } \sum _ { c \in \{ R , G , B \} } I _ { \mathrm { g t } } ( y , x , c ) .\tag{3}
$$

The hidden reference is read as an 8-bit sRGB image and converted to float32 by division by 255. The mean is computed directly in the normalized sRGB domain, without inverse-gamma conversion, luminance weighting, a median operator, or separate per-channel prompts. Thus, $B \in [ 0 , 1 ]$ is a single float32 value computed independently for every target frame.

In the oficial evaluation, B is calculated by the organizers from the hidden reference and supplied with the test input. Participants cannot access $I _ { \mathrm { g t } }$ , but they can use or ignore the scalar derived from it. Accordingly, B is privileged ground-truth-derived information, and the oficial benchmark is more precisely described as target-brightness-statistic-conditioned restoration. Although an individual model may technically accept a user-selected value at inference time, the challenge results do not evaluate generalization to arbitrary user-selected brightness values.

The target-brightness statistic is needed because brightness adjustment can be a one-to-many mapping before the target is specified. Let $X = \left( \mathbf { I } _ { \mathrm { i n } } , \mathbf { E } \right)$ denote the RGB-event input. For the same input X, diferent target brightness levels may correspond to diferent valid reference images. Without the brightness statistic, the optimal prediction under a squared reconstruction loss is

$$
G ^ { * } ( X ) = \arg \operatorname* { m i n } _ { \hat { I } } \mathbb { E } \left[ \left\| { \hat { I } } - I _ { \operatorname* { g t } } \right\| _ { 2 } ^ { 2 } | \ X \right] = \mathbb { E } \left[ I _ { \operatorname* { g t } } \ | \ X \right] .\tag{4}
$$

Therefore, if the training data contain several target brightness levels for the same input, the prediction may approach their conditional average. Its brightness is then determined by the conditional target distribution rather than a specified target level. By conditioning the model on B, the optimal prediction becomes

$$
F ^ { * } ( X , B ) = \arg \operatorname* { m i n } _ { \hat { I } } \mathbb { E } \left[ \left\| { \hat { I } } - I _ { \operatorname { g t } } \right\| _ { 2 } ^ { 2 } | X , B \right] = \mathbb { E } \left[ I _ { \operatorname { g t } } \mid X , B \right] .\tag{5}
$$

Conditioning on B therefore reduces target ambiguity by selecting the desired brightness level. This derivation explains the role of B under the stated loss.

## 2.5 Evaluation Protocol and Challenge Rules

Predictions are evaluated in normalized RGB space without boundary cropping. PSNR is the primary metric and determines the verified ranking; SSIM is secondary. Both metrics are computed per frame and averaged separately over the Low-to-Normal and High-to-Normal subsets, and the reported average is the mean of the two subset scores. L1 and LPIPS are additionally reported for the organizer-selected case analysis, but they do not afect the oficial ranking. Parameters and FLOPs are descriptive and do not afect the ranking.

Because of CodaBench storage and evaluation limits, the Phase 2 server evaluates one frame in every ten from each hidden video. For technical verification, the organizers run the submitted checkpoints and inference code on the same hidden videos using dense temporal sampling. All accounts were subject to the same phase-specific server settings and were limited to five submissions per day and 100 submissions per account in each phase. The challenge website opened on May 10, 2026; the validation server opened on May 25; the hidden test data and Phase 2 server opened on June 25; submissions closed on July 3; and results were announced on July 10. External public data and pretrained weights were permitted, provided that their use was disclosed in the technical report. As noted above, the rules did not restrict temporal context and allowed future frames in this ofline setting.

A valid CodaBench submission is a correctly formatted prediction package that was successfully evaluated by the Phase 2 server. Fifteen such submissions/accounts were received. Because a platform account does not by itself establish a distinct verified team identity, these submissions are not interpreted as 15 verified teams. The six teams reported in this paper responded to the organizer invitation, confirmed their team identities, described their systems, and provided checkpoints, inference code, and instructions for organizer-side evaluation. Only these six systems form the verified open-system ranking in this report.

## 2.6 Prior Evidence and Oficial Baseline

The efectiveness of event guidance is an established premise of the challenge rather than a conclusion drawn from its open-system ranking. Prior event-guided low-light studies reported controlled comparisons supporting the complementary value of events under degraded illumination [4, 13, 14]. SEE-Net [15] extended this line of work to broad-range brightness adjustment and analyzed brightness conditioning. Its previously reported prompt-merge ablation on SDE reduced

Table 2: Results of the SEE Challenge 2026 under Phase 2 Codabench and organizerside dense sampling. Dense-sampling PSNR determines the ranking, while SSIM is secondary. Best and second-best results are shown in bold and underlined. FLOPs (G) and parameters are reported at the native resolution. FLOPs (G) include cascaded stages but exclude TTA, temporal averaging, ensembles, and post-processing.
<table><tr><td colspan="8">(a) Phase 2 evaluation on Codabench</td></tr><tr><td rowspan="2">Participant</td><td colspan="2">Average</td><td colspan="2">Low-to-Normal</td><td colspan="2">High-to-Normal</td><td rowspan="2">FLOPs (G)</td><td rowspan="2">Params (M)</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>pixelartai</td><td>23.1810</td><td>0.7246</td><td>21.2066</td><td>0.6965</td><td>25.1555</td><td>0.7527</td><td>1314.77</td><td>10.24</td></tr><tr><td>vincent2013</td><td>23.0865</td><td>0.7178</td><td>21.1040</td><td>0.6941</td><td>25.0689</td><td>0.7416</td><td>3390.56</td><td>54.21</td></tr><tr><td>EventFuse</td><td>22.7992</td><td>0.7057</td><td>21.1942</td><td>0.6809</td><td>24.4042</td><td>0.7305</td><td>97.51</td><td>1.16</td></tr><tr><td>bucloud</td><td>22.6168</td><td>0.7048</td><td>20.7068</td><td>0.6779</td><td>24.5268</td><td>0.7316</td><td>870.37</td><td>15.85</td></tr><tr><td>bin_jiang</td><td>22.1735</td><td>0.6853</td><td>20.6521</td><td>0.6632</td><td>23.6948</td><td>0.7074</td><td>346.07</td><td>3.06</td></tr><tr><td>aka18</td><td>20.8262</td><td>0.6608</td><td>19.2888</td><td>0.6452</td><td>22.3636</td><td>0.6764</td><td>8141.41</td><td>3.38</td></tr></table>

(b) Oficial dense-sampling evaluation
<table><tr><td rowspan="2">Participant</td><td colspan="2">Average</td><td colspan="2">Low-to-Normal</td><td colspan="2">High-to-Normal</td><td rowspan="2">FLOPs (G)</td><td rowspan="2">Params (M)</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>pixelartai</td><td>23.1731</td><td>0.7237</td><td>21.1863</td><td>0.6957</td><td>25.1600</td><td>0.7518</td><td>1314.77</td><td>10.24</td></tr><tr><td>vincent2013</td><td>23.0873</td><td>0.7169</td><td>21.1212</td><td>0.6942</td><td>25.0534</td><td>0.7396</td><td>3390.56</td><td>54.21</td></tr><tr><td>EventFuse</td><td>22.8043</td><td>0.7061</td><td>21.1974</td><td>0.6810</td><td>24.4112</td><td>0.7311</td><td>97.51</td><td>1.16</td></tr><tr><td>bucloud</td><td>22.5865</td><td>0.7038</td><td>20.7150</td><td>0.6780</td><td>24.4581</td><td>0.7297</td><td>870.37</td><td>15.85</td></tr><tr><td>bin_jiang</td><td>22.1478</td><td>0.6849</td><td>20.6705</td><td>0.6637</td><td>23.6251</td><td>0.7061</td><td>346.07</td><td>3.06</td></tr><tr><td>aka18</td><td>20.8082</td><td>0.6606</td><td>19.3104</td><td>0.6453</td><td>22.3061</td><td>0.6758</td><td>8141.41</td><td>3.38</td></tr></table>

PSNR from 23.57 dB to 22.26 dB when prompt merging was disabled, while SSIM changed from 0.7724 to 0.7713; it also visualized outputs over a sweep of prompt values. These are results from the prior SEE-Net study.

The released SEE-Net baseline takes one RGB frame, a synchronized 64-bin event voxel grid, and a brightness statistic. It uses separate RGB and event encoders, event-aware cross-attention, and a five-layer exposure decoder. The model has 1.89M parameters and obtains 18.8279 dB PSNR and 0.6414 SSIM on the released test split. These released-split results are provided for reference.

## 3 Challenge Results and Organizer-Side Analysis

Verified Open-System Ranking. Table 2 reports the six verified systems under both Phase 2 CodaBench sampling and organizer-side dense sampling. MLSLabs (CodaBench handle: pixelartai) ranks first under dense sampling with 23.1731 dB PSNR and 0.7237 SSIM, followed by ACVLab-TL (vincent2013) with 23.0873 dB and 0.7169 SSIM. The PSNR diference between the top two systems is only 0.0858 dB. We therefore describe their aggregate scores as closely spaced and do not interpret the average diference as evidence of a statistically meaningful separation.

Dense sampling preserves the ordering observed on CodaBench. Across all six systems, the absolute change from subsampled to dense evaluation is below

Table 3: Performance comparison across diferent groups.
<table><tr><td>Team</td><td>Robot Motion (G1) PSNR↑ SSIM↑ L1↓ LPIPS↓</td><td>Checkerboard (G2) PSNR↑ SSIM↑ L1↓</td><td>Chinese Text (G3) LPIPS↓ PSNR↑ SSIM↑ L1↓</td></tr><tr><td>pixelartai</td><td>18.76 0.4998 0.0826 0.2767</td><td>21.49 0.7951 0.0638</td><td>0.1553 24.83 0.7486</td></tr><tr><td>vincent2013</td><td>18.85 0.5010 0.0805 0.2701</td><td>21.32 0.7941 0.0657</td><td>0.0375 0.1548 25.06 0.7583 0.0364</td></tr><tr><td>EventFuse</td><td>18.73 0.4889 0.0847 0.3100</td><td>21.80 0.7902 0.0622 0.1556</td><td>0.1800 24.40 0.74980.0427 0.1877</td></tr><tr><td>bucloud</td><td>0.4956 0.0848 0.3069</td><td>21.17 0.7814 0.0668</td><td>24.52 0.7484 0.0400</td></tr><tr><td>jiangbin</td><td>0.4986 0.0852 0.3431</td><td>21.11 0.7603 0.0706</td><td>23.69 0.7299 0.0467</td></tr><tr><td>aka18</td><td>0.4735 0.1122 0.3772</td><td>20.43 0.7513 0.0749</td><td>22.36 0.6954 0.0557</td></tr><tr><td>SEE-Net</td><td>0.4482 0.15520.4610</td><td>16.87 0.7037 0.11480.2564</td><td>21.15 0.6309 0.06950.2560</td></tr></table>

0.04 dB PSNR and 0.001 SSIM. This agreement indicates that the one-in-ten CodaBench sampling provides a close estimate of the dense-sequence averages for these submissions, while the organizer-side run additionally verifies that the supplied code and checkpoints reproduce the submitted predictions.

Performance Across Exposure Settings. All six systems obtain lower PSNR on Low-to-Normal than on High-to-Normal restoration. For pixelartai, for example, the dense scores are 21.1863 dB and 25.1600 dB, respectively. The consistent gap shows that the Low-to-Normal subset is empirically more dificult under this hidden-test composition. It is consistent with the hypothesis that severe underexposure provides weaker and noisier RGB measurements, but the aggregate scores alone do not establish that mechanism; scene content, motion, event density, target distribution, and residual registration error may also contribute.

Representative Case Analysis. Table 3 reports three representative hidden groups selected to probe robot motion, repetitive checkerboard texture, and Chinese text. The cases provide interpretable comparisons. Among the three cases, Robot Motion (G1) yields the lowest PSNR for every submitted system. ACVLab-TL (vincent2013) leads this case on PSNR, SSIM, L1, and LPIPS, with 18.85 dB PSNR and 0.5010 SSIM. The uniformly lower scores are consistent with motion being an additional challenge, although one group cannot isolate motion from illumination, scene content, event density, or registration efects.

Checkerboard (G2) tests repetitive high-frequency structure. EventFuse obtains the best PSNR and L1, MLSLabs (pixelartai) obtains the best SSIM, and ACVLab-TL obtains the best LPIPS. The disagreement is small but informative: the three metrics favor diferent reconstructions of the same fine texture. Chinese Text (G3) produces the highest PSNR of the three cases for all submitted systems. ACVLab-TL leads PSNR, SSIM, and L1, whereas ACVLab-LCY (bucloud) obtains the lowest LPIPS. This split again shows that pixel fidelity, structural similarity, and perceptual distance need not select the same system. All submitted systems outperform the released SEE-Net baseline in the three cases, but the gains and metric ordering depend on the visual content. Consequently, the average leaderboard should be read together with content-specific evidence.

Organizer-Side Synthesis. Taken together, the results support three observations. First, dense sampling confirms the aggregate ordering obtained by the server, but the 0.0858 dB gap between the top two systems remains too small to support a claim of clear separation. Second, the named cases expose contentdependent behavior that is hidden by the average: motion lowers all reported PSNR values, repetitive texture produces metric disagreement, and Chinese text changes the relative perceptual ranking. Third, the shared failure case shows that visibility recovery and detail recovery are distinct objectives under severe underexposure. The disclosed system configurations help formulate hypotheses about temporal context, TTA, and model scale, but the open-system ranking cannot attribute any observation to one design choice. A future challenge should therefore retain an open track while adding predeclared content strata and a constrained single-frame track for controlled comparisons.

System Design and Inference Cost. The verified systems difer in backbone scale, event fusion, pretraining, temporal processing, and inference multiplicity. Their parameter counts range from 1.16M for EventFuse to 54.21M for the two-stage EventRestormer pipeline, but model size alone does not explain the ranking. Only Event-SCAM uses adjacent frames: its core network processes one RGB–event pair per call and conditionally averages outputs from the target frame and six neighboring frames. The other five systems receive a single target frame and its associated events, even when they repeat inference on that input. EventFuse performs one compact restoration pass; EventRestormer combines a two-stage cascade with eight-way TTA, resulting in 16 stage-level forward passes per output. EvLCD and SEE-SplitNet each use four single-frame TTA variants, while REFM performs six Euler updates within one single-frame latent-transport pipeline.

The FLOPs in Table 2 describe the reported core networks and are not a standardized end-to-end cost measurement. Moreover, the teams reported latency on diferent GPUs and software environments. We therefore use the disclosed frame counts and inference passes to interpret eficiency and avoid presenting the reported FLOPs or latency as a controlled hardware comparison.

Qualitative Observations. Figure 2 complements the numerical results. The verified systems generally recover substantially more visible structure than the released baseline, especially in strongly degraded regions. However, dificult examples still exhibit residual noise, color shifts, weakened texture, or local contrast errors. The event visualization helps identify regions containing temporal contrast information, but its presence does not guarantee recovery when the RGB measurement is severely clipped or when the event signal is sparse or noisy. These examples motivate evaluating both pixel-level fidelity and perceptual quality and show that the current task remains unsolved despite the improvement over the baseline.

![](images/5f50122fafe81ffe826d4f087835bc2451555ee7103ccece451e64bd1ce6d4e8.jpg)  
Fig. 2: Qualitative comparison of the verified submissions on representative Low-to-Normal and High-to-Normal examples. Input RGB frames, event visualizations, reference images, restored outputs, and enlarged dificult regions are displayed with a common image range and without additional gamma or contrast adjustment. The examples illustrate diferences in residual noise, color, local contrast, and texture recovery that are not fully captured by average PSNR and SSIM.

![](images/8b23c058b1db1cb1a5dcd1135a4866d5d4e97f4a4593c29f66d85f3961b506bd.jpg)

(a) Events  
![](images/03530ba64647a4d791261766e2c44b4062ae994721f7e761e81c2db6d9715f73.jpg)  
(b) Input Frame  
(c) SEE-Net 12.74/0.4451/0.4659  
(d) Pixelartai 18.46/0.5200/0.2766  
(e) vincent2013 18.59/0.5190/0.2688  
(f) GT Frame  
(g) EventFuse 18.73/0.5094/0.3080  
(h) Bucloud 18.38/0.5139/0.3040  
(i) Jiangbin 18.49/0.5133/0.3362  
(j) aka18 15.83/0.4809/0.3947  
Fig. 3: Representative shared failure case under severe underexposure. (a) Accumulated events, (b) input frame, (c) SEE-Net baseline, (d)–(e) and (g)–(j) verified submissions, and (f) ground-truth frame. The green boxes indicate the enlarged regions shown below. Although the submitted systems substantially brighten the input, none faithfully reconstructs the checkerboard structure. The values below each result denote PSNR/SSIM/LPIPS (↑ / ↑ / ↓).

Shared Failure Pattern. Figure 3 presents a severely underexposed example for which none of the evaluated systems faithfully reconstructs the local checkerboard structure. All verified submissions improve substantially over the SEE-Net baseline, which obtains 12.74 dB PSNR, 0.4451 SSIM, and 0.4659 LPIPS. Nevertheless, the best submitted PSNR is only 18.73 dB, the best SSIM is 0.5200, and the best LPIPS is 0.2688; these values are achieved by diferent systems. The methods largely correct global brightness, but their enlarged crops share blurred square boundaries, weakened contrast, and color shifts relative to the reference. The event visualization contains activity around the scene structure, yet the nearly black RGB crop provides weak appearance and color measurements. This single example does not isolate the cause of failure, but it identifies a common limitation: event-guided exposure correction can restore visibility without recovering fine local texture faithfully.

Table 4: Comparison of training and inference strategies adopted by the participating teams and the SEE baseline. All methods use the brightness prompt. Efective Passes denotes the number of model forward passes or iterative updates used during inference.
<table><tr><td>Team</td><td>Pretraining</td><td>Prompt</td><td>TTA</td><td>Temporal Processing</td><td>Effective Passes</td></tr><tr><td>pixelartai</td><td>×</td><td>√</td><td>×</td><td>√</td><td>1</td></tr><tr><td>vincent2013</td><td>√</td><td>√</td><td>√</td><td>×</td><td>16</td></tr><tr><td>EventFuse</td><td>×</td><td>√</td><td>×</td><td>×</td><td>1</td></tr><tr><td>bucloud</td><td>×</td><td>√</td><td>√</td><td>×</td><td>4</td></tr><tr><td>jiangbin</td><td>√</td><td>√</td><td>×</td><td>X</td><td>1</td></tr><tr><td>aka18</td><td>X</td><td>√</td><td>√</td><td>X</td><td>4</td></tr><tr><td>SEE-Net</td><td>×</td><td>√</td><td>×</td><td>X</td><td>1</td></tr></table>

## 4 Participating Methods

This section summarizes the six verified systems using a common set of dimensions. Complete architecture diagrams, training configurations, and teamprovided descriptions are included in the supplementary material. Table 4 provides a standardized comparison of pretraining, brightness conditioning, TTA, temporal processing, and inference multiplicity. It separates multi-frame temporal processing from repeated inference on one frame: Event-SCAM is the only submitted system that aggregates adjacent-frame outputs, while the other five systems use a single target-frame input. The descriptions below focus on the design choices needed to interpret the challenge results.

MLSLabs (pixelartai). Event-SCAM uses NAFBlocks and bidirectional crossattention to fuse RGB and event features. It is trained for 300 epochs on SEE-600K with Charbonnier and gradient losses, without external data or pretrained weights. The core network processes one frame–event pair per call. When the event count is below 3200, the method averages the target-frame output with three preceding and three succeeding outputs; otherwise, it returns only the target-frame prediction. Event-SCAM is the only multi-frame submission and uses up to seven single-frame calls.

ACVLab-TL (vincent2013). EventRestormer uses a four-level Restormer RGB backbone, a 64-bin event encoder, and spatial FiLM fusion. A sevendimensional vector formed from B and RGB statistics conditions its latent and decoder stages, while public Restormer denoising weights initialize the RGB backbone. Its two-stage cascade applies eight-way dihedral TTA to the same target frame and event grid, producing 16 stage-level passes without adjacent frames.

EventFuse. EventFuse processes one RGB frame and a 64-bin event voxel grid using Bayer and coordinate encodings, a Mamba-based illumination estimator, bidirectional fusion, and sparse state-space blocks. Its decoder is conditioned on B, and the model is trained on the oficial data without external data or pretrained weights. With 1.16M parameters, it is the smallest verified model and restores each frame in one pass.

Table 5: Contributing teams, participant names, and afiliations for the verified systems summarized in this paper.
<table><tr><td>Team</td><td>Participants</td><td>Affiliation</td></tr><tr><td>MLSLabs (pixe- lartai)</td><td>Dongyang Zhang, Renjie Zou, Zhiwei Huang, Dong Jiang</td><td>Malanshan Audio &amp; Video Laboratory</td></tr><tr><td>ACVLab-TL (vincent2013)</td><td>Yun-Tze Tsai, Shao-Kai Liu, Chia- Ming Lee, Chih-Chung Hsu</td><td>Advanced Computer Vision Lab (ACVLab), National Yang Ming</td></tr><tr><td>EventFuse</td><td>Yixin Chen, Lupeng Liu</td><td>Chiao Tung University University of Chinese Academy of Sci-</td></tr><tr><td>ACVLab-LCY (bucloud)</td><td>Chia-Yu Lin</td><td>ences Advanced Computer Vision Lab (ACVLab), National Yang Ming</td></tr><tr><td>VisionLab-JB</td><td>Bin Jiang</td><td>Chiao Tung University; National Cheng Kung University Nanjing University</td></tr><tr><td>(bin_jiang) aka18</td><td>Tao Liu</td><td>Wuhan University</td></tr></table>

ACVLab-LCY (bucloud). EvLCD estimates an exposure-corrected base image from local color distributions and predicts a residual with a 64-bin event encoder and FiLM-conditioned U-Net. It is trained from scratch on the oficial data in two stages. Inference averages four flipped predictions of the same frame and shifts their global mean to match B.

VisionLab-JB (bin\_jiang). REFM treats restoration from one RGB–event input as rectified-flow transport in a learned RGB latent space. A velocity field fuses a binary event voxel grid with an exposure descriptor derived from source brightness and B, then performs six Euler updates from the observed latent state. The 3.06M-parameter pipeline uses only the oficial training data and decodes the updated latent representation once.

aka18. SEE-SplitNet trains wider exposure-specific SEE-Net variants for Lowto-Normal and High-to-Normal restoration. It sets $C _ { 1 } = 1 2 8$ and $C _ { 2 } = 1 2 8$ , conditions on B with a 7-layer exposure MLP, and uses progressive patch-size training followed by PSNR-oriented fine-tuning. Inference selects the task-specific model and applies four TTA variants to the same target frame.

Event fusion difers across the submitted architectures. Five systems use one target frame. Event-SCAM instead conditionally aggregates adjacent-frame outputs. Repeated inference and multi-frame processing are separate system properties.

## 5 Limitations

The challenge and this report have several limitations. First, the verified ranking is open-system: systems use diferent frame counts, future context, pretraining, cascades, TTA, temporal averaging, and post-processing. It measures complete submitted pipelines and cannot isolate the contribution of events or any individual component. Prior work provides the modality-level motivation, but no new matched RGB-only retraining was conducted on the hidden set. The participating teams supplied checkpoints and inference code, so a fair organizer-run modality ablation for the submitted systems was not available.

Second, the oficial target-brightness statistic is derived from each hidden reference. It is privileged information and difers from a practical setting in which a user selects a desired brightness without access to ground truth. Although SEE-Net previously visualized diferent prompt values, the challenge did not conduct a standardized quantitative sweep measuring monotonicity or output-brightness error for all submitted methods.

Third, the evaluation emphasizes frame-wise PSNR and SSIM. The representative case table adds L1 and LPIPS, and the failure figure exposes one shared qualitative weakness, but these selected examples do not replace a complete per-sequence distribution. The challenge also does not include a temporalconsistency metric, a systematic analysis against event rate or scene motion, or a scene-level bootstrap confidence interval. Accordingly, small score diferences should not be interpreted as statistically established separations.

Finally, the reported computation and latency values were produced by teams on diferent hardware and may exclude parts of the full inference pipeline. Conditional temporal averaging also makes the efective cost input-dependent.

## 6 Conclusion

We presented the SEE Challenge 2026, an open-system benchmark for eventguided, target-brightness-statistic-conditioned restoration across a broad illumination range. Fifteen valid CodaBench submissions were received, and six distinct teams completed organizer-side verification and are included in the reported ranking. Dense sampling produced results close to the subsampled server evaluation, while exposure-specific and content-oriented analyses showed that Low-to-Normal restoration obtains lower scores and that the best method varies across cases and metrics. The shared failure example further shows that correcting global brightness does not ensure faithful recovery of fine local texture under severe underexposure. The submitted systems explore diverse RGB-event fusion, conditioning, temporal, and inference strategies. The challenge provides a common protocol and identifies open questions in exposure recovery, eficiency, temporal consistency, and practical brightness control.

Acknowledgements: The organizers thank all participants and particularly the six verified teams that provided checkpoints, inference code, technical descriptions, and supplementary method reports.

## References

1. Afifi, M., Derpanis, K.G., Ommer, B., Brown, M.S.: Learning multi-scale photo exposure correction. In: CVPR. pp. 9157–9167 (2021) 2

2. Archana, R., Jeevaraj, P.E.: Deep learning models for digital image processing: a review. Artificial intelligence review 57(1), 11 (2024) 2

3. Chakravarthi, B., Verma, A.A., Daniilidis, K., Fermuller, C., Yang, Y.: Recent event camera innovations: A survey. In: ECCV. pp. 342–376. Springer (2024) 2

4. Chen, K., Liang, G., Lu, Y., Li, H., Wang, L.: Evlight++: Low-light video enhancement with an event camera: A large-scale real-world dataset, novel method, and more. IEEE TAPMI (2025) 2, 6

5. Conde, M., Timofte, R., Berdan, R., Besbinar, B., Iso, D.: Raw image reconstruction from rgb on smartphones. ntire 2025 challenge report. In: CVPR. pp. 1254– 1268 (2025) 2

6. DeSouza, G.N., Kak, A.C.: Vision for mobile robot navigation: A survey. IEEE TAPMI 24(2), 237–267 (2002) 1

7. Gehrig, D., Scaramuzza, D.: Low-latency automotive vision with event cameras. Nature 629(8014), 1034–1040 (2024) 2

8. Guo, C., Li, C., Guo, J., Loy, C.C., Hou, J., Kwong, S., Cong, R.: Zero-reference deep curve estimation for low-light image enhancement. In: CVPR. pp. 1780–1789 (2020) 2

9. Guo, X., Li, Y., Ling, H.: Lime: Low-light image enhancement via illumination map estimation. IEEE TIP 26(2), 982–993 (2016) 2

10. Huo, Y., Gan, J., Jiang, W.: Multi-exposure high dynamic range imaging based on lsgan. Displays 83, 102707 (2024) 2

11. Kim, J., Lee, S., Kang, S.J.: End-to-end diferentiable learning to hdr image synthesis for multi-exposure images. In: AAAI. vol. 35, pp. 1780–1788 (2021) 2

12. Koshel, R.J.: Illumination Engineering: design with nonimaging optics. John Wiley & Sons (2012) 1

13. Liang, J., Yang, Y., Li, B., Duan, P., Xu, Y., Shi, B.: Coherent event guided lowlight video enhancement. In: ICCV. pp. 10615–10625 (2023) 2, 6

14. Liu, L., An, J., Liu, J., Yuan, S., Chen, X., Zhou, W., Li, H., Wang, Y.F., Tian, Q.: Low-light video enhancement with synthetic event guidance. In: AAAI. vol. 37, pp. 1692–1700 (2023) 6

15. Lu, Y., Xu, X., Lu, H., Qian, Y., Li, P., Yao, H., Yang, B., Li, J., Cai, Q., Guo, W., Xiong, H.: See: See everything every time - adaptive brightness adjustment for broad light range images via events. IJCV (2026) 2, 3, 4, 5, 6

16. Ma, L., Ma, T., Liu, R., Fan, X., Luo, Z.: Toward fast, flexible, and robust low-light image enhancement. In: CVPR. pp. 5637–5646 (2022) 2

17. Sharif, W., Dilmaghani, M.S., Kielty, P., Moustafa, M., Lemley, J., Corcoran, P.: Event cameras in automotive sensing: A review. IEEE Access 12, 51275–51306 (2024) 1

18. Tan, X., Chen, H., Zhang, R., Wang, Q., Kan, Y., Zheng, J., Jin, Y., Chen, E.: Deep multi-exposure image fusion for dynamic scenes. IEEE TIP 32, 5310–5325 (2023) 2

19. Wang, W., Wei, C., Yang, W., Liu, J.: Gladnet: Low-light enhancement network with global awareness. In: 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018). pp. 751–755. IEEE (2018) 2

20. Xu, X., Wang, R., Lu, J.: Low-light image enhancement via structure modeling and guidance. In: CVPR. pp. 9893–9903 (2023) 2