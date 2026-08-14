# RGB-D Video Generation for Improving Human-to-Robot Object Handover Prediction

Tianyu Sun<sup>1</sup>, Zhoujie Fu<sup>1</sup>, Zihui Gao<sup>2</sup>, Bang Zhang<sup>3</sup>, Guosheng Lin<sup>1</sup> <sup>1</sup>College of Computing and Data Science, Nanyang Technological University <sup>2</sup>College of Computer Science and Technology, Zhejiang University <sup>3</sup>Alibaba Group

Abstract: Human-to-robot (H2R) object handover is a fundamental capability for human-robot collaboration, yet progress is hindered by the scarcity of largescale, human-centric datasets and the significant sim-to-real gap. To address these challenges, we introduce Hand2Bot, an RGB-D video dataset that provides rich contextual information such as body posture and facial expressions, specifically collected for handover scenarios with real-world noise patterns. We further propose PassGen, a generative pipeline that leverages stable video diffusion and an Intention-Aware Temporal Face Encoder to synthesize realistic handover sequences while ensuring hand-object consistency. To bridge the sim-to-real gap, we implement a morphology-based depth editing strategy that replicates realistic sensor noise found in physical depth maps. Experimental evaluations demonstrate that our framework achieves high intention identification accuracy and low false trigger rates in both ablation studies and real-world deployment on a physical robot platform. Our results confirm that training on PassGen allows for robust zero-shot transfer and earlier intention anticipation compared to traditional handcentric baselines, effectively enabling socially aware robotic behavior in shared workspaces.

Keywords: Human-robot collaboration, RGB-D perception, datasets for robotic perception

## 1 Introduction

Human-to-robot (H2R) object handover is one of the most fundamental tasks in robotics, serving as a cornerstone for enabling effective human–robot collaboration. Applications span assistive robots for the elderly, service robots in household environments, and collaborative systems on industrial assembly lines [1, 2]. The ability to safely and intuitively transfer objects between humans and robots not only enhances task efficiency but also serves as a critical precursor for complex downstream manipulation tasks. Consequently, proactively anticipating human communicative intent has attracted substantial attention in recent years [3, 4, 5].

Despite recent progress, existing datasets for H2R handover remain limited in their ability to support robust and generalizable learning due to two critical missing elements. First, the majority of available data is synthetic [6, 7], which fails to capture the complex noise characteristics of realworld sensors, including edge scattering, lighting variations, and motion blur. As illustrated in Fig. 1 (a1) and (a2), raw physical depth observations retain severe stochastically distributed noise patterns; since real-time depth inpainting is computationally prohibitive for reactive control, models trained on clean synthetic geometry suffer from a pronounced simulation-to-reality (sim-to-real) gap. Second, mainstream benchmarks [7, 8] predominantly focus on egocentric or hand-centric views (Fig. 1(b1), (b2)), completely omitting the broader body context. For natural and timely intention prediction, however, global human features—such as facial expressions, gaze direction, head orientation, and upper-body posture—are essential precursors to anticipate when and how a handover will be initiated [9, 10].

![](images/18ca80ecd9d444e70f29e37243360daab23fedfc778010b629eec7f93fc8cb5c.jpg)  
(a1)

![](images/40765100b522382ebe8aef042deccd7cecea96f16096b7da6c6524b543bec4f1.jpg)  
(a2)

![](images/21dc658f4821e724d650c0734a2c0beabcefd0497ffac992ecf508a512edcf3d.jpg)  
(b1)

![](images/48ea0ce429d3c40d8726fa1c91128f33e2db23c79504ed1cfe44c66d544b234d.jpg)  
(b2)  
Figure 1: Existing bottlenecks in current H2R handover datasets. (a1) and (a2) compare a noisy real-world depth map against a clean synthetic depth map. (b1) and (b2) demonstrate the restricted, hand-centric field of view in mainstream benchmarks.

To bridge these gaps, our primary motivation is to construct a comprehensive dataset that captures the full-body context alongside physical sensor artifacts. However, scaling such real-world multi-modal data through manual capture incurs prohibitive annotation and recording overhead. This necessitates an algorithmic solution capable of synthesizing high-fidelity, interactive RGB-D sequences. Unfortunately, state-of-the-art generative models fall short in this specialized domain. General human video generation tasks rarely encounter human-object interactions from robotperspective viewpoints [11, 12]. More importantly, existing depth synthesis pipelines focus strictly on visual reconstruction fidelity, producing overly smoothed depth maps that lack the characteristic void patterns of standard sensors [13, 14].

Driven by these algorithmic limitations, our core modification strategy introduces a unified generative and control framework. For the RGB modality, we propose PassGen, which builds upon video diffusion by embedding an Intention-Aware Temporal Face Encoder (TFE) to capture highfrequency facial micro-expressions and gaze focus. For the depth modality, we implement a morphology-based edge erosion strategy that directly models the realistic void distributions of an Intel RealSense L515 sensor. These generated streams are integrated with our real-world captures to construct the Hand2Bot dataset. Furthermore, to explicitly isolate perception enhancements from downstream manipulation constraints, we deploy an independent Intention Gating (IG) mechanism that leverages these global social cues to proactively trigger grasp trajectories.

Through extensive physical robot trials on a UR5e platform, we demonstrate that training on PassGen-augmented data yields exceptional zero-shot transfer capabilities. By shifting the tracking focus from localized hand tracking to multi-modal body-gaze context, the robot successfully eliminates false triggers during ambient human motion while achieving significantly earlier intent anticipation.

In summary, our contributions are threefold:

• We propose PassGen, a generative RGB pipeline implementing Stable Video Diffusion with a dedicated Temporal Face Encoder to maintain hand-object consistency and explicit social gaze cues.

• We introduce a morphology-based depth noise simulation that replicates realistic sensorspecific artifacts, effectively mitigating the sim-to-real gap.

• We construct the Hand2Bot dataset, providing 5,000 multimodal video pairs with rich fullbody annotations, and validate its downstream robotic utility through extensive control and ablation experiments.

## 2 RELATED WORKS

## 2.1 Human Video Generation for Interactive Scenarios

Diffusion models have greatly pushed the boundary of generative research topics [15, 16]. Recent advances in human video generation [17, 18] have begun to explore interactive scenarios where humans engage with objects or environments. Hu et al. [19] propose Animate Anyone 2, which extends character animation pipelines to generate highly realistic human videos from reference images, with improvements in pose fidelity and motion consistency. Similarly, Xu et al. [20] introduce AnchorCrafter, a framework that allows controllable video synthesis conditioned on human-object interactions (HOI), enabling plausible motion dynamics when anchors such as objects are present. While these methods achieve impressive realism, they often focus primarily on pose or appearance consistency and lack explicit modeling of fine-grained human-object contact, which can lead to artifacts or implausible interactions in manipulation-heavy scenarios. Moreover, most existing approaches generate only RGB videos, without addressing the depth modality that is critical for robotic applications such as object handover. These limitations highlight the need for approaches that can robustly handle interactive human-object dynamics while producing RGB-D representations that bridge the gap between visual realism and robotic utility.

## 2.2 Object Handover Datasets

Several datasets have been proposed to study object handover, but each has notable limitations. HandoverSim [6] and GenH2R [7] are synthetic datasets designed to model H2R handover scenarios, which provide large-scale, controllable data but inevitably suffer from a sim-to-real gap due to the lack of realistic noise and variability found in real-world captures. On the other hand, DexH2R [8] includes real-world handover data, but it is primarily restricted to the hand regions, offering limited context about the human body or the surrounding environment that is crucial for modeling naturalistic interactions. Beyond explicit handover datasets, research works such as the subset of VidHOI implemented in HOI4ABOT [12], and the dataset constructed by Pang et al. [21] include more parts of human bodies in the field of view for HOI scenarios. They provide diverse object manipulation scenes, but they do not specifically target handover tasks, which limits their applicability to robotic perception and control. These gaps highlight the need for a dataset that is both real-world and human-centric, as well as multimodal (RGB-D), specifically tailored for handover scenarios to support downstream robotic applications.

## 3 HAND2BOT DATASET

To address the scarcity of high-quality human-to-robot (H2R) handover data, we construct the Hand2Bot dataset. Unlike existing datasets, Hand2Bot captures the full human body field of view and incorporates realistic sensor noise, facilitating robust intention sensing and sim-to-real transfer.

## 3.1 Construction and Annotation

Hand2Bot contains 5,000 RGB-D video pairs (2,125 real-world; 2,875 generated via PassGen). For the real subset, we utilized an Intel RealSense L515 camera across 7 diverse indoor scenes with 33 common household objects.

Intention-Centric Labeling: Critically, we include approximately 20% of video pairs as negative cases where the human moves with the object but has no intention to hand it over. In our dataset, positive handover cases comprise clear gaze-directed object approach motions. Negative cases extend beyond ambient background movements to include ambiguous gestures, weak-gaze reaching, and non-gaze-oriented tool passes, ensuring the evaluation is not biased toward scenarios artificially favorable to our gating cues. To support downstream timing, we provide timestamp-based labels for the presentation period [2], marking the exact frames where handover intention is explicitly initiated.

Table 1: Statistical comparison. Noise? represents whether the depth information in the dataset considers real-world noise. Scope stands for the field of view for the data. Paradigm represents the task setting, where A indicates that the giver/receiver can be anything. #Realistic Video is the number of realistic RGB(-D) videos. Note that the dataset name in [22] is not specified.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Source</td><td colspan="4">Data Composition</td><td colspan="3">Handover Detail</td><td rowspan="2">#Realistic Video</td></tr><tr><td>RGB</td><td>Depth</td><td>Noise?</td><td>Scope</td><td>Paradigm</td><td>Giver</td><td>Receiver</td></tr><tr><td>HandoverSim [6]</td><td>Synthetic</td><td>√</td><td>√</td><td>X</td><td>Hand</td><td>H2R</td><td>Hand</td><td>Parallel Gripper</td><td></td></tr><tr><td>GenH2R [7]</td><td>Synthetic</td><td>√</td><td>√</td><td>X</td><td>Hand</td><td>H2R</td><td>Hand</td><td>Parallel Gripper</td><td></td></tr><tr><td>[22]</td><td>Real</td><td>√</td><td>√</td><td>√</td><td>Human</td><td>H2R</td><td>Hand</td><td></td><td>330</td></tr><tr><td>DexYCB [23]</td><td>Real</td><td>√</td><td>√</td><td>√</td><td>Hand</td><td>H2A</td><td>Hand</td><td></td><td>1,000</td></tr><tr><td>H2O [24]</td><td>Real</td><td>√</td><td>X</td><td>=</td><td>Hand</td><td>H2H</td><td>Hand</td><td>Hand</td><td>1,200</td></tr><tr><td>HOH [25]</td><td>Real</td><td>√</td><td>√</td><td>√</td><td>Hand</td><td>H2H</td><td>Hand</td><td>Hand</td><td>2,720</td></tr><tr><td>DexH2R [8]</td><td>Real</td><td>√</td><td>√</td><td>√</td><td>Hand</td><td>H2R</td><td>Hand</td><td>ShadowHand</td><td>4,282</td></tr><tr><td>Hand2Bot</td><td>Both</td><td>√</td><td>√</td><td>√</td><td>Human</td><td>H2R</td><td>Hand</td><td>Parallel Gripper</td><td>5,000</td></tr></table>

Grasp Pose Labeling: To support downstream manipulation tasks, we integrate GraspNet [26] to generate a candidate pool of 6-DoF parallel-gripper grasp poses, with the workspace mask strictly constrained to the object region, as shown in the visualization results in Fig. 2. To guarantee execution stability and physical reachability for the robotic executor, we implement a vertical-alignment filtering strategy rather than relying on a singular fixed heuristic.

![](images/a2501a5ae3e4b148d04c330ef668f59bf734cc598b345bc0fd3c4a5ad70320b6.jpg)  
Figure 2: Grasp Pose Annotation. We give two examples of the GraspNet-based annotation on our Hand2Bot dataset. We mask all facial regions and replace them with the corresponding visualized depth maps.

Specifically, let $\textbf { R } \in \ S O ( 3 )$ represent the orientation matrix of a candidate grasp pose, and $\mathbf { v } _ { z } = [ 0 , 0 , - 1 ] ^ { T }$ denote the canonical downward vertical axis aligned with gravity. We evaluate the angular deviation $\theta = \operatorname { a r c c o s } ( \mathbf { g } _ { z } \cdot \mathbf { v } _ { z } )$ for all valid, collision-free candidate poses returned by GraspNet, where g<sub>z</sub> is the approach vector of the gripper. The system dynamically selects the kinematically feasible pose that minimizes $\theta ,$ effectively prioritizing top-down vertical grasp trajectories while retaining the flexibility of full 6-DoF adaptation to arbitrary object geometries. We implement a temporal consistency filter to select stable pose sequences during the presentation period and exclude any pose profiles that establish physical contact with the human hand region.

## 3.2 Benchmark Comparison and Advantages

As illustrated in Tab. 1, Hand2Bot provides several unique advantages over existing benchmarks that are critical for real-world deployment. While prior synthetic datasets such as HandoverSim [6] and GenH2R [7] are noise-free and thus suffer from a pronounced sim-to-real gap, our inclusion of realistic L515 sensor noise patterns facilitates more robust policy transfer to physical systems. Furthermore, most real-world datasets [8, 23, 24] predominantly utilize hand-centric views, which limit the perception system to local cues. In contrast, Hand2Bot offers a full-body field of view, enabling the exploitation of global semantic features—such as gaze direction, head orientation, and upper-body posture—which are essential for accurate human intention anticipation. Finally, by specifically targeting the H2R paradigm with a parallel-gripper configuration and providing the largest collection of realistic RGB-D clips to date, Hand2Bot serves as a comprehensive bridge between high-level intention sensing and low-level robotic manipulation.

![](images/669081377a57d7b07797b862529243484d6780ca786c4ec4d689e5fa3e63b15a.jpg)  
Figure 3: Pipeline of our PassGen. Our method is based on an SVD backbone, with two branches of networks. Given a pair of reference images $I _ { r e f }$ and video $V _ { r e f }$ , the appearance feature from $I _ { r e f }$ and the temporal feature from $V _ { r e f }$ are operated by corresponding modules separately. We also elaborate on the structures of TFE and Pose Guidance (PG) blocks in the bottom right corner.

All real-world data collection and experiments were conducted with the informed consent of the participants. We only focus on non-invasive motion and intention sensing.

## 4 METHODOLOGY

Our goal is to generate realistic human-to-robot handover RGB-D videos that augment training data for downstream robotic tasks. To this end, we propose PassGen, a two-stage generation pipeline (Fig. 3) that first synthesizes human-object interaction RGB videos and then reconstructs depth maps with realistic noise patterns.

## 4.1 Stage I: Pose-Guided RGB Video Generation

We adopt a diffusion-based backbone based on Stable Video Diffusion (SVD) [27]. To ensure the generated content is physically grounded for robotics, we condition the denoising U-Net on reference identity and precise motion sequences.

1) Control Signal Integration: Unlike text-driven models that may introduce semantic ambiguity, PassGen relies on deterministic physical signals. We utilize an Appearance Encoder (ReferenceNet structure [17]) to extract identity features from a reference image $I _ { r e f } { . }$ , and a lightweight PoseNet [28] to process per-frame skeletons $V _ { r e f }$ extracted via DWPose [29]. This multi-source conditioning constrains the latent space to adhere to the physical laws of human-object interaction.

2) Intention-Aware Temporal Face Encoder (TFE): To capture subtle handover intentions, we augment the pose guidance with a dedicated TFE branch. The design of TFE is theoretically motivated by the fact that in H2R scenarios, facial micro-expressions and gaze direction often precede physical reaching motions as precursor social signals.

We extract facial embeddings using ArcFace [30] and process them through cascaded Temporal Attention (TA) and Feed Forward (FFN) blocks. Unlike standard animation models that treat the head as a rigid component, TFE models the facial region as a high-frequency semantic stream. This produces temporally-consistent face tokens $\tilde { F } _ { t }$ that preserve the giver’s visual focus, which is essential for the downstream robot to differentiate between an intentional handover and an ambient motion [12].

Table 2: Animation Results on Hand2Bot-Real Dataset. Red marks the best algorithm, and blue marks the second-best.
<table><tr><td rowspan="2">Method</td><td colspan="4">Image</td><td colspan="2">Video</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS</td><td>FID↓</td><td>FID-VID</td><td>FVD↓</td></tr><tr><td>MagicPose [33]</td><td>15.71</td><td>0.733</td><td>0.412</td><td>243.70</td><td>295.61</td><td>714.36</td></tr><tr><td>MagicAnimate [34]</td><td>16.21</td><td>0.740</td><td>0.400</td><td>212.48</td><td>284.08</td><td>947.65</td></tr><tr><td>AnimateAnyone [17]</td><td>16.51</td><td>0.749</td><td>0.343</td><td>190.02</td><td>203.31</td><td>623.79</td></tr><tr><td>DreamPose [35]</td><td>16.58</td><td>0.753</td><td>0.352</td><td>173.90</td><td>192.13</td><td>682.55</td></tr><tr><td>MimicMotion [28]</td><td>19.48</td><td>0.780</td><td>0.340</td><td>111.39</td><td>97.70</td><td>439.65</td></tr><tr><td>Animate-x [32]</td><td>22.23</td><td>0.853</td><td>0.235</td><td>130.25</td><td>142.47</td><td>568.99</td></tr><tr><td>StableAnimator [36]</td><td>23.17</td><td>0.867</td><td>0.218</td><td>101.67</td><td>106.73</td><td>372.81</td></tr><tr><td>PassGen</td><td>25.12</td><td>0.909</td><td>0.212</td><td>91.32</td><td>99.77</td><td>337.59</td></tr></table>

3) Denoising and Training: We fine-tune the SVD backbone using the LoRA scheme on our Hand2Bot-Real dataset [31]. The final pose-facial guidance $\tilde { P } _ { t }$ , which integrates both structural skeletons and social face tokens, is injected into the U-Net via cross-attention layers [19, 32]:

$$
U _ { t } ^ { l } \gets U _ { t } ^ { l } + C r o s s A t t e n t i o n ( Q = U _ { t } ^ { l } , K = \tilde { P } _ { t } , V = \tilde { P } _ { t } )\tag{1}
$$

This ensures that every generated frame is not only spatially aligned with the handover trajectory but also contextually validated by explicit human social intention.

## 4.2 Stage II: Morphological Depth Video Generation

To bridge the sim-to-real gap, we simulate the specific noise characteristics of the Intel RealSense L515 sensor, which often suffers from edge scattering and surface absorption.

1) Depth Initialization: We generate an initial affine-invariant depth map $D _ { i n i t }$ using DepthCrafter [14]. This ensures temporal smoothness across the handover sequence while maintaining a unified scale. We then perform a height-based linear alignment to substitute absolute depth values into the original human mask.

2) Realistic Noise Simulation: Real-world depth data often contains “void” pixels (zero-depth). We record the noise distribution $N _ { 0 }$ from our real-world collection and implement a morphological erosion strategy. We map boundary width vectors $w ( h )$ relative to the pixel height h to erode the generated depth $D _ { g e n }$ . This process serves as an empirical, heuristic morphology-based approximation of boundary-level depth artifacts, rather than a mathematically validated model of physical L515 sensor-noise distributions. This produces the final realistic depth output $D _ { f i n a l }$ with interframe alignment.

By establishing physical consistency across the synthesized RGB-D streaming sequences, the Pass-Gen framework eliminates the ideal-geometry bias typical of synthetic engines, providing structurally grounded and sensor-realistic augmented data to train downstream workflows.

## 5 INTENTION-GATED PERCEPTION AND CONTROL SYSTEM

To transition from passive, reactive grasping to socially cooperative interaction, we introduce an independent Intention Gating (IG) mechanism. This module operates as a heuristic multimodal gating mechanism that bridges multimodal human context perception with downstream robotic grasp execution, isolating intent detection from physical manipulation constraints.

## 5.1 Physical Camera Configuration and Application Scenario

As illustrated in our scenario definition, capturing full-body cues while maintaining reliable facial gaze tracking poses a coupled perception challenge. To resolve this, our application environment deploys the Intel RealSense L515 sensor at a calibrated robot-centric perspective: positioned on the UR5e robot base, tilted upwards facing the interaction zone. This geometric configuration ensures that the camera’s wide field-of-view encompasses the participant’s upper body and arm kinematics (from a distance of 1.5 m to 2.2 m), while providing sufficient facial spatial resolution to resolve high-frequency micro-expressions and gaze direction stream inputs.

![](images/7404daee02b930e6c9c781d210f5b16559c1d3187f75a7e6a9cb5225f2172b9f.jpg)  
Figure 4: Qualitative results on Hand2Bot-Real dataset. We compare our method with other SOTA human video generation algorithms on different reference images and video frames from our Hand2Bot-Real dataset. Our PassGen shows a promising and stable performance compared with other SOTA methods. We mark the artifacts of the previous methods with red boxes.

## 5.2 Multimodal Fusion Formulation

The gating logic continuously ingests synchronized multi-modal features to compute a unified scalar intention score $S _ { i n t e n t } ( t ) \in [ 0 , 1 ]$

$$
S _ { i n t e n t } ( t ) = \sigma \left( w _ { g } \cdot f _ { g a z e } ( \tilde { F _ { t } } ) + w _ { v } \cdot v _ { o b j } ( t ) \right)\tag{2}
$$

where $f _ { g a z e } ( \tilde { F } _ { t } )$ represents the temporal gaze focus confidence score extracted via the Temporal Face Encoder (TFE), and $\begin{array} { r } { v _ { o b j } ( t ) = - \frac { \partial D _ { o b j } } { \partial t } } \end{array}$ denotes the physical approaching velocity computed from the masked depth boundaries. The weights $w _ { g }$ and $w _ { v }$ serve as scaling coefficients to balance the dimensionless gaze prediction with the metric velocity scalar.

## 5.3 Monotone Constraints and Sensitivity Analysis

During dynamic interaction, temporary gaze drops or localized occlusions can induce stochastic oscillations in instantaneous tracking outputs. To circumvent unstable state switching in physical deployments, we enforce a monotone non-decreasing constraint on the final inference trigger boundary:

$$
\hat { S } _ { i n t e n t } ( t ) = \operatorname* { m a x } \left( \hat { S } _ { i n t e n t } ( t - 1 ) , S _ { i n t e n t } ( t ) \right)\tag{3}
$$

The parallel-jaw gripper’s grasp trajectory is proactively initiated if and only if $\hat { S } _ { i n t e n t } ( t ) \geq \tau .$ where τ is a strict safety activation threshold.

To justify our parameterization, a sensitivity analysis was performed on the threshold boundary. We set $\tau = 0 . 8 0$ optimizes system performance, suppressing accidental activations from ambient user motion while minimizing the tracking latency before the presentation period terminates.

Table 3: Ablation Study on PassGen Modules. We compare our PassGen with the conditions without several crucial blocks, including ArcFace, Temporal Face Encoding (TFE), and Pose Guidance (PG), on the Hand2Bot-Real dataset test set. We take two image-based and two video-based metrics for comparison. We mark the best metrics in bold.
<table><tr><td>Backbone</td><td>Pose Guidance</td><td>TFE</td><td>ArcFace</td><td>FID-VID↓</td><td>FVD↓</td><td>PSNR ↑</td><td>SSIM↑</td></tr><tr><td>√</td><td>√</td><td>V</td><td></td><td>99.77</td><td>337.59</td><td>25.12</td><td>0.909</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>107.44</td><td>373.61</td><td>22.77</td><td>0.829</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>111.60</td><td>399.52</td><td>21.47</td><td>0.809</td></tr><tr><td>√</td><td></td><td></td><td></td><td>117.43</td><td>422.90</td><td>19.12</td><td>0.791</td></tr></table>

![](images/9dbc373e76f21dcd656a9059429039cbbe8deafd58810b1696fe7d083f9386fa.jpg)  
Figure 5: Qualitative comparison of generative RGB-D sequences. Our PassGen pipeline effectively reconstructs human-object interaction geometry from sparse L515 inputs while maintaining realistic sensor noise patterns compared to over-smoothed baselines like DepthCrafter [14].

## 6 EXPERIMENTS

We evaluate PassGen on the Hand2Bot dataset across two dimensions: 1) the fidelity of the generated RGB-D sequences, and 2) its utility for downstream robotic intention sensing.

## 6.1 Setup and Metrics

The training and testing of PassGen is implemented on NVIDIA A6000. We assess visual quality using PSNR, SSIM, and LPIPS for frames, and FID-VID [37] and FVD [38] for temporal coherence.

## 6.2 Animation Results and Ablation

We evaluate PassGen against state-of-the-art animation models on a test set of 250 clips from Hand2Bot-Real. As shown in Tab. 2, PassGen consistently outperforms existing methods across both frame-level (PSNR, SSIM) and video-level (FVD, FID-VID) metrics, indicating superior spatial fidelity and temporal stability.

Qualitative comparisons are presented in Fig. 4. While models such as StableAnimator [36] and MimicMotion [28] achieve high texture fidelity, they frequently struggle with hand-object consistency during rapid handover trajectories. Specifically, as highlighted in the zoomed-in regions of Fig. 4, prior methods often exhibit severe artifacts, such as object distortion or finger-mask flickering, when the human giver performs complex reaching motions. In contrast, PassGen maintains a stable interaction interface throughout the sequence. This is attributed to our Pose-Guidance and TFE modules, which effectively bridge the gap between global body dynamics and local manipulation cues, ensuring the generated RGB-D pairs plausible for downstream robotic training.

As for the ablation study, tab. 3 quantifies the contribution of our conditioning modules. Removing the Temporal Face Encoder (TFE) leads to a significant drop in FVD, confirming that gaze and facial consistency are vital for realistic human-to-robot interaction synthesis.

Table 4: Ablation Study of Intention Score Module. Performance comparison on the Hand2Bot-Real evaluation subset.
<table><tr><td>Configuration</td><td>Gaze Term</td><td>Object Term</td><td>Mean Acc. ↑</td><td>FPR↓</td></tr><tr><td>Baseline</td><td></td><td></td><td>72.7%</td><td>95.5%</td></tr><tr><td>Gaze-only</td><td>√</td><td></td><td>86.4%</td><td>31.8%</td></tr><tr><td>Object-only</td><td></td><td>√</td><td>78.2%</td><td>77.3%</td></tr><tr><td>Full Module</td><td>√</td><td>V</td><td>90.0%</td><td>13.6%</td></tr></table>

## 6.3 Depth Fidelity Analysis

The primary objective of our depth generation is twofold: ensuring pixel-level alignment with the synthesized RGB video while preserving the inherent noise patterns of physical sensors. As shown in Fig. 5, generic depth estimation (e.g., DepthCrafter [14]) produces edges that are far too “clean” for robotic perception training. Our approach intentionally reintroduces morphological artifacts at the human-object interface. This ensures that the generated depth sequence maintains the realistic format, which is critical for training robust policies that do not over-fit to perfect synthetic geometry.

## 6.4 Intention Gating Performance and Ablation Study

To evaluate the efficacy of our intention gating mechanism, we conduct an ablation study on the Intention Score Module using an evaluation subset of the Hand2Bot-Real dataset.

As summarized in Tab. 4, the baseline lacking both terms fails to reject ambient movements (95.5% FPR). While the object term alone marginally improves accuracy (78.2%), adding the gaze term significantly drives system robustness, dropping the FPR to 31.8%. Our Full Module achieves the best overall performance (90.0% Mean Acc., 13.6% FPR), confirming that integrating visual attention with spatial reaching cues is essential for safe operation. For evaluation, a trial is labeled as a successful grasp based on strict safety constraints rather than fixed spatio-temporal thresholds: the gripper must establish stable force closure on the target object without any physical collision or proximity interference with the human hand.

## 6.5 Real H2R Handover Evaluation

To validate the practical utility of our proposed framework, we deployed the system on a physical UR5e platform. We conducted human-to-robot handover trials to assess the effectiveness of the PassGen-augmented policy and the Intention Gating (IG) scheme.

## 6.5.1 Experimental Setup

We selected 10 distinct objects, including 5 Seen Objects and 5 General Unseen objects. To evaluate the intention gating mechanism, we conducted:

• Positive Trials: 10 intentional handover attempts per object type to evaluate the Intention Success Rate (ISR). We explicitly define ISR as an intention-triggering metric that measures whether the robot correctly initiates a reaching action during the presentation period, rather than evaluating full mechanical grasp completion or physical manipulation generalization.

• Negative Trials: unrelated movements to evaluate the False Trigger Rate (FTR).

## 6.5.2 Results Analysis

The experimental results demonstrate that the system maintains high reliability, particularly in identifying handover intent. Our full module achieves an overall Mean ISR of 54/60, indicating robust performance across diverse object categories.

Table 5: Real-World Handover Performance: intention reliability and safety evaluation on realworld scenarios.
<table><tr><td rowspan="2">Object</td><td rowspan="2">Geometric Category</td><td colspan="2">w IG</td><td rowspan="2">w/o IG FTR↓</td></tr><tr><td>ISR↑</td><td>FTR↓</td></tr><tr><td rowspan="6">Seen</td><td>Box</td><td>10 / 10</td><td>1/5</td><td>5/5</td></tr><tr><td>Cylinder</td><td>10 / 10</td><td>0/5</td><td>5/5</td></tr><tr><td>Cup</td><td>9/ 10</td><td>0/5</td><td>4/5</td></tr><tr><td>Plate</td><td>10 / 10</td><td>0/5</td><td>5/5</td></tr><tr><td>Irregular</td><td>8 / 10</td><td>1/5</td><td>3/5</td></tr><tr><td></td><td>7/ 10</td><td>0/5</td><td>3/5</td></tr><tr><td>Overall</td><td></td><td>54/60</td><td>2/30</td><td>25 / 30</td></tr></table>

Table 6: Ablation Study: we compare the results w/o Hand2Bot-Syn
<table><tr><td>Training Data</td><td>Mean Acc. ↑</td><td>Mean FPR↓</td><td>Unseen Obj. ISR ↑</td></tr><tr><td>Real</td><td>87.5%</td><td>22.8%</td><td>6/10</td></tr><tr><td> $\mathbf { R e a l + S y n }$ </td><td>90.0%</td><td>13.6%</td><td>7/10</td></tr></table>

Analysis of Failures: Performance degradation observed in the Irregular and Unseen categories (ISR of 80% and 70%, respectively) stems from two primary factors. First, the object term of the intention score occasionally fails with irregular geometries, as the system struggles to determine if the object has been moved forward into the interaction zone. Second, even when the intention is correctly identified, the complex surface of these objects makes them physically hard to grasp for the parallel-jaw gripper, leading to failed completions despite a correct trigger.

Safety and Baseline Comparison: The system achieved a low overall FTR of 2/30 with the intention gating mechanism active, confirming that the gaze-aware module effectively prevents accidenta triggers. In stark contrast, the Baseline configuration (w/o IG) suffered from a severe false trigger rate of 25/30, effectively failing to distinguish between intentional handovers and ambient movements. In negative trials, the robot remained stationary even when the hand was near the object, provided the human’s visual focus was directed elsewhere. This significant reduction in FTR validates the safety and necessity of our approach in shared human-robot workspaces.

## 6.6 Downstream Utility: Real-Only vs. Augmented Training

To establish the explicit value of our synthesized data and address potential generative bias concerns, we conducted a direct downstream comparison using an identical prediction architecture and evaluation setting. We trained the intention prediction model under two data configurations: (i) Real, where the network was optimized solely on the real-world training video pairs, and (ii) Real + Syn, where the network was trained on all training video pairs augmented by our generative pipeline.

As detailed in Table 6, augmenting training with 2,875 synthesized sequences from PassGen-Syn significantly improves downstream tracking accuracy, reduces FPR, and enhances robustness against overfitting on unseen objects. This validates that our structural pose constraints and facial signal modeling successfully increase physical variability. Given the limited margin on unseen objects (N = 10 trials) without formal statistical significance across multiple random seeds, we frame the synthetic augmentation primarily as a variance-reduction and false-positive suppression mechanism rather than a claim of broad manipulation superiority.

## 6.7 Gating Sensitivity Analysis and Parameter Configuration

To evaluate the robustness of our Intention Gating (IG) mechanism and justify our design choices, we conducted an offline parameter analysis. The scalar intent score combines a dimensionless facial gaze confidence $f _ { g a z e } \in [ 0 , 1 ]$ and a metric spatial approach velocity $v _ { o b j } ~ \mathrm { { ( m / s ) } }$ . To resolve this scale discrepancy, the fusion weights are configured as $w _ { g } = 1 . 0$ and $w _ { v } = 2 . 5 [ s / m ]$ ], ensuring that a typical human reaching motion balances equivalently with a stable gaze trigger. The final decision-making is supervised by the activation threshold τ.

We executed an offline parametric sensitivity sweep of the continuous activation threshold $\tau \in$ $\{ 0 . 6 0 , 0 . 7 0 , 0 . 8 0 , 0 . 9 0 \}$ . An aggressive configuration $( \tau = 0 . 6 0 )$ maximizes temporal tracking responsiveness but renders the single-frame gating highly vulnerable to transient variations, causing premature latching and yielding a high frame-level FPR of 72.5%. Conversely, an overly conservative threshold $( \tau = 0 . 9 0 )$ thoroughly suppresses frame-level sensory noise to an exceptionally low FPR of 9.8%, but it severely restricts the proactive motion window, frequently missing object presentation phases and leading to interactive deadlocks. Our selected operational configuration of $\tau = 0 . 8 0$ marks the ideal value, securing an optimal balance by maximizing intent tracking sensitivity for smooth haptic transitions while strictly dampening the offline continuous frame-level FPR to 13.6%.

## 7 Discussions and Limitations

Our empirical evaluation yields several key insights and highlights open boundaries for future research. First, we observe that sensor-specific noise simulation is critical for cross-modal policy transfer; while frameworks like DepthCrafter [14] produce smooth geometry, our morphological noise simulation replicates the stochastic edge scattering characteristic of real depth sensors (e.g., RealSense L515), preventing the downstream network from overfitting to idealized synthetic structures. Second, by combining 6-DoF grasp annotations with timestamped intention labels, our pipeline enables a paradigm shift from hand-centric reactive observation to proactive, intention-aware humanrobot collaboration.

Despite these advantages, certain limitations remain. Our generation relies on re-injecting segmented characters into static backgrounds, constraining environmental diversity [39]. Furthermore, due to robotic safety constraints and strict institutional verification boundaries, a formalized qualitative user study was not conducted. Consequently, our empirical evidence characterizes objective perception tracking profiles rather than subjective user comfort or psychological fluency. Future work will explore user-centric haptic alignment surveys and subjective naturalness evaluations to further refine the collaborative gating logic.

## 8 CONCLUSION

In this work, we present PassGen, a generative RGB-D human video pipeline that addresses pose articulation complexity and data scarcity in H2R handover scenarios. Combining real-world captures with PassGen synthesis, we construct the Hand2Bot dataset, incorporating morphological depth noise simulation to facilitate robust sim-to-real transfer on physical robot platforms. While irregular geometries and challenging grasp profiles can limit object-level completion, our safety-first gating mechanism effectively suppresses accidental triggers. We anticipate that our dataset and framework will serve as a practical foundation for future research in socially aware robotic perception and proactive handover control.

## References

[1] W. Yang, C. Paxton, A. Mousavian, Y.-W. Chao, M. Cakmak, and D. Fox. Reactive human-torobot handovers of arbitrary objects. In ICRA, 2021.

[2] V. Ortenzi, A. Cosgun, T. Pardi, W. P. Chan, E. Croft, and D. Kulic. Object handovers: a´ review for robotics. IEEE Trans. Robot., 37(6):1855–1873, 2021.

[3] C. Meng, T. Zhang, and T. lun Lam. Fast and comfortable interactive robot-to-human object handover. In IROS, 2022.

[4] S. Christen, L. Feng, W. Yang, Y.-W. Chao, O. Hilliges, and J. Song. Synh2r: Synthesizing hand-object motions for learning human-to-robot handovers. In ICRA, 2024.

[5] Y.-Y. Huang and K.-T. Song. Human-to-robot handover control of an autonomous mobile robot based on hand-masked object pose estimation. IEEE Robotics and Automation Letters, 2024.

[6] Y.-W. Chao, C. Paxton, Y. Xiang, W. Yang, B. Sundaralingam, T. Chen, et al. Handoversim: A simulation framework and benchmark for human-to-robot object handovers. In ICRA, 2022.

[7] Z. Wang, J. Chen, Z. Chen, P. Xie, R. Chen, and L. Yi. Genh2r: learning generalizable humanto-robot handover via scalable simulation demonstration and imitation. In CVPR, 2024.

[8] Y. Wang, J. Ye, C. Xiao, Y. Zhong, H. Tao, H. Yu, Y. Liu, J. Yu, and Y. Ma. Dexh2r: A benchmark for dynamic dexterous grasping in human-to-robot handover. In ICCV, 2025.

[9] Z. Huang, Y.-J. Mun, F. C. Pouria, and K. Driggs-Campbell. Hierarchical intention tracking with switching trees for real-time adaptation to dynamic human intentions, 2025. arXiv:2506.07004.

[10] W. Wang, R. Li, Y. Chen, Y. Sun, and Y. Jia. Predicting human intentions in human–robot hand-over tasks through multimodal learning. IEEE Trans. Autom. Sci. Eng., 19(3):2339– 2353, 2021.

[11] L. Qiu, X. Gu, P. Li, Q. Zuo, W. Shen, J. Zhang, et al. Lhm: Large animatable human reconstruction model from a single image in seconds. arXiv preprint arXiv:2503.10625, 2025.

[12] E. V. Mascaro, D. Sliwowski, and D. Lee. Hoi4abot: Human-object interaction anticipation for human intention reading collaborative robots. arXiv preprint arXiv:2309.16524, 2023.

[13] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. Depth anything v2. NeurIPS, 2024.

[14] W. Hu, X. Gao, X. Li, S. Zhao, X. Cun, et al. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025.

[15] X. Yi, Z. Wu, Q. Shen, Q. Xu, P. Zhou, J.-H. Lim, S. Yan, X. Wang, and H. Zhang. Mvgamba: Unify 3d content generation as state space sequence modeling. 2024.

[16] T. Sun, D. Hu, Y. Dai, and G. Wang. Diffusion-based depth inpainting for transparent and reflective objects. IEEE Transactions on Circuits and Systemsfor Video Technology, 2024.

[17] L. Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR, 2024.

[18] T. Sun, Z. Fu, B. Zhang, and G. Lin. Mvanimate: Enhancing character animation with multiview optimization, 2026. URL https://arxiv.org/abs/2602.08753.

[19] L. Hu, G. Wang, Z. Shen, X. Gao, D. Meng, L. Zhuo, P. . Zhang, and L. Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025.

[20] Z. Xu, Z. Huang, J. Cao, Y. Zhang, X. Cun, Q. Shuai, et al. Anchorcrafter: Animate cyberanchors saling your products via human-object interacting video generation. arXiv preprint arXiv:2411.17383, 2024.

[21] Y. Pang, R. Shao, J. Zhang, H. Tu, Y. Liu, et al. Manivideo: Generating hand-object manipulation video with dexterous and generalizable grasping. In CVPR, 2025.

[22] J. Laplaza, A. Garrell, F. Moreno-Noguer, and A. Sanfeliu. Context and intention for 3d human motion prediction: experimentation and user study in handover tasks. In RO-MAN, 2022.

[23] Y.-W. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, et al. Dexycb: A benchmark for capturing hand grasping of objects. In CVPR, 2021.

[24] R. Ye, W. Xu, Z. S. Xue, T. Tang, Y. Wang, and C. Lu. H2o: A benchmark for visual humanhuman object handover analysis. In ICCV, 2021.

[25] N. Wiederhold, A. Megyeri, D. Paris, S. Banerjee, and N. Banerjee. Hoh: Markerless multimodal human-object-human handover dataset with large object count. NeurIPS, 2023.

[26] H.-S. Fang, C. Wang, M. Gou, and C. Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In CVPR, 2020.

[27] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

[28] Y. Zhang, J. Gu, L.-W. Wang, H. Wang, J. Cheng, Y. Zhu, and F. Zou. Mimicmotion: Highquality human motion video generation with confidence-aware pose guidance. In ICML, 2025.

[29] Z. Yang, A. Zeng, C. Yuan, and Y. Li. Effective whole-body pose estimation with two-stages distillation. In ICCV, 2023.

[30] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019.

[31] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, et al. Lora: Low-rank adaptation of large language models. 2022.

[32] S. Tan, B. Gong, X. Wang, S. Zhang, D. Zheng, R. Zheng, K. Zheng, J. Chen, and M. Yang. Animate-x: Universal character image animation with enhanced motion representation. In ICLR, 2025.

[33] D. Chang, Y. Shi, Q. Gao, H. Xu, J. Fu, G. Song, et al. Magicpose: realistic human poses and facial expressions retargeting with identity-aware diffusion. In ICML, 2024.

[34] Z. Xu, J. Zhang, J. H. Liew, H. Yan, J.-W. Liu, C. Zhang, et al. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024.

[35] J. Karras, A. Holynski, T.-C. Wang, and I. Kemelmacher-Shlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In ICCV, 2023.

[36] S. Tu, Z. Xing, X. Han, Z.-Q. Cheng, Q. Dai, C. Luo, and Z. Wu. Stableanimator: High-quality identity-preserving human image animation. In CVPR, 2025.

[37] Y. Balaji, M. R. Min, B. Bai, R. Chellappa, and H. P. Graf. Conditional gan with discriminative filter generation for text-to-video synthesis. In IJCAI, 2019.

[38] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, et al. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[39] Y. Song, C. B. Liu, W. Mao, and M. Z. Shou. Mitty: Diffusion-based human-to-robot video generation. arXiv preprint arXiv:2512.17253, 2025.