# Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models

# Juncheng Ma<sup>∗</sup> Jiaxin Li<sup>∗</sup> Jiankai Tu<sup>∗</sup> Bo Liang<sup>∗</sup> Guodong Lv Yufan Deng Junzhe Tang Liangzhu Yang Chen Bian Zicong Cao Baohao Zhu Jianxin Bi Tengfei Yan Yinyuan Zhang Kewei Zhang Xinglong Chen Kaiqi Chen Duomin Wang Xiaotian Tang Liang Tan Li Li Yuan Xu Haosen Yang Yuqi Wang Jingfei Xia Songyan Huang Zhiyang Dou Zhen Dong Enze Xie Tat-Seng Chua Daquan Zhou<sup>†</sup>

PKU, NUS, MIT, UCSB, NVIDIA

<sup>∗</sup>Equal Contribution, <sup>†</sup>Corresponding Author

## Abstract

Egocentric human video ofers a scalable source of manipulation priors for robot learning, yet translating raw data scale into generalizable robot policies demands rigorous data curation and training recipes. We introduce Open-HumanScale, the first fully open recipe for scaling egocentric pretraining to 50,000 hours, anchored by Open-HumanScale-50K—a curated egocentric video corpus densely annotated with hand motion and natural language. Leveraging this corpus, we systematically investigate pure human pretraining via closed-loop simulation evaluations and uncover an empirical log-linear scaling law linking pretraining data hours to downstream robot success rates. Grounded in these findings, we develop HumanWAM-Vanilla, a world–action model pretrained exclusively on 50k hours of egocentric human video. Upon robot post-training, HumanWAM-Vanilla reaches 92.08% success on RoboTwin 2.0 Clean2Clean and 68.88% on Clean2Random, alongside a 14.17% mean success rate and a 19.00 mean progress score on RoboDojo. To push the capability frontier further, we explore joint human–robot pretraining with HumanWAM-Goal, a visual-goal-conditioned world– action model. Hybrid data scaling unlocks broad zero-shot generalization across both simulation and physical benchmarks, propelling post-trained RoboTwin 2.0 performance to new state-of-the-art levels: 92.70% on Clean2Clean and a substantial leap to 84.60% on Clean2Random. On physical robots, HumanWAM-Goal further exhibits emergent embodied in-context learning, dynamically adapting its manipulation behavior to unseen visual goals without parameter updates. To support reproducible scaling research in embodied AI, we open-source the Open-HumanScale-50K corpus, data curation and pretraining pipelines, alongside the pretrained and post-trained HumanWAM checkpoints, inference code, and simulation evaluation suites.

Project Page: https://dagroup-pku.github.io/open-humanscale/

GitHub Repo: https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale

HuggingFace Models: https://huggingface.co/collections/DAGroup-PKU/open-humanscale

HuggingFace Datasets: https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50K

## Contents

Introduction 3   
Building an Open Egocentric Pretraining Corpus 4   
2.1 Pretraining Data Sources 5   
2.2 Data Curation 6   
2.3 Unified Action Representation   
3 Exploration of Pure Egocentric Data Scaling 8   
3.1 Pretraining and Evaluation Protocol 9   
3.2 Scaling Egocentric Data from 100 to 50k Hours: What Improves? 10   
3.3 Scaling Diversity, Not Just Hours 11   
3.4 Scaling Egocentric Pretraining for Data-Eficient Robot Adaptation 12   
3.5 Does Cross-Source Action Alignment Matter? 14   
3.6 Action Supervision in Egocentric Pretraining 14   
3.7 Can Egocentric Pretraining Surpass Robot Pretraining? 16   
3.8 From Egocentric Pretraining to Multi-View Robot Policies 17   
Pure Egocentric Data Foundation Model Training 18   
4.1 Model and Training Setup 18   
4.2 Simulation Experiments 19   
4.3 Real-Robot Experiments 20   
4.4 Ablation Studies 21   
5 Pushing the Model Capability Boundary with Hybrid Data Scaling 23   
5.1 Robot Pretraining Data 24   
5.2 Hybrid Action Representation 25   
5.3 HumanWAM-Goal Architecture 25   
5.4 Training Recipe 27   
5.5 Evaluation Settings 27   
5.6 Comparison with State-of-the-Art Embodied Foundation Models 28   
Emerging Capabilities in Real-World Robot Behavior 31   
6.1 Real-Robot Experimental Setup . 31   
6.2 Real-World Zero-Shot Performance 33   
6.3 In-Context Learning with Visual Goals 34   
6.4 Long-Horizon Execution 39   
Related Work 39   
7.1 World Action Models 39   
7.2 Pure Egocentric Pretraining 43   
7.3 Zero-Shot Generalization and In-Context Learning 43   
8 Conclusion 43   
A Prompts for Goal Image Synthesis and Review 49   
A.1 Pipeline Overview 49   
A.2 Stage 1: Scene-Specific Grounding 49   
A.3 Stage 2: Goal Image Editing 50   
A.4 Goal Review and Repair 52   
B Real-World Zero-Shot Task Suite 53

## 1 Introduction

![](images/7e72f312850847d9f4b55f3fdc8f14c5d58c50f287aecedbe39d9840c94efb74.jpg)  
Figure 1 Overview of Open-HumanScale. Using a curated 50,000-hour egocentric corpus, we show that scaling human data improves robot generalization after post-training, with greater task and scene diversity yielding steeper scaling curves. We then extend this pretraining approach to combine human and robot data, finding that hybrid scaling improves zero-shot success in simulation and across 20 real-world tasks.

Learning from human manipulation ofers a scalable path toward general-purpose robot intelligence [22, 33, 73]. Robot foundation models require diverse manipulation experience to acquire skills that transfer across tasks and environments, but collecting this experience through teleoperation is costly [22, 23]. Egocentric videos paired with hand-motion and language annotations ofer a rich record of human interactions with objects across diverse scenes and tasks [20, 43]. However, developing efective egocentric pretraining recipes that consistently improve downstream robot performance as data scale remains under investigation.

Previous work has demonstrated the value of human experience for robot learning. EgoScale [73] shows that large-scale human motion pretraining improves dexterous robot manipulation, while Being-H0.8 [4] extends egocentric pretraining to joint visual and tactile world modeling. HumanScale [33] further shows that, after robot post-training, validation action loss on held-out real-robot trajectories decreases log-linearly with egocentric pretraining hours. Dyna-2 [14] extends egocentric pretraining to one million hours and reports a scaling law in zero-shot robot action prediction, with corresponding improvements in robot task performance after post-training. These studies establish the empirical value of egocentric scaling, motivating further analysis of data diversity and the robot supervision required for downstream adaptation.

We present Open-HumanScale, a fully open recipe for scaling egocentric pretraining for robot learning. Within a unified world–action model framework, we explore pretraining exclusively on 100 to 50,000 hours of human egocentric data, using a common robot post-training protocol and closed-loop simulation for controlled and reproducible comparisons of downstream robot task success. With curated data and a unified action representation, we observe an empirical log-linear scaling law relating pretraining hours to out-of-distribution success rates across two diferent attention configurations. Beyond duration, data diversity is a key scaling dimension, as expanding task and scene coverage alongside duration produces steeper scaling curves than increasing duration at fixed coverage. Larger-scale egocentric pretraining also improves the sample eficiency of robot adaptation, yielding higher success rates at matched demonstration budgets.

Following this recipe, we develop HumanWAM-Vanilla, a vanilla world–action model pretrained exclusively on 50,000 hours of egocentric human data. After robot post-training, HumanWAM-Vanilla achieves 92.08% success on RoboTwin 2.0 [8] Clean2Clean and 68.88% on Clean2Random, together with a mean success rate of 14.17% and a mean progress score of 19.00 on RoboDojo [9]. The improvements extend to physical robots across multi-step, deformable-object, and long-horizon manipulation tasks. Under changes in tablecloth and lighting, mean success across four tasks increases from 2.5% without egocentric pretraining to 67.5% with HumanWAM-Vanilla, demonstrating stronger generalization.

To further push the model capability boundary, we extend our exploration to joint human-robot pretraining and propose HumanWAM-Goal, a world–action model with visual-goal conditioning. Human data provide broad manipulation experience, while robot data ground this experience in embodiment-specific perception and control [25, 73]. Visual goals specify desired object configurations and spatial relationships, providing task context for zero-shot execution. With the specially designed unified action representation, HumanWAM-Goal architecture, and the hybrid training recipe, the co-trained model supports out-of-the-box execution without parameter updates during deployment. We evaluate data scaling by expanding the robot corpus and the human data. The performance scaling trend persists in this hybrid data setting. On physical robots, adding 45.9k hours of human data to a fixed 14.6k-hour robot corpus increases mean zero-shot success across 20 tasks from 20.0% to 30.5%. On RoboTwin 2.0, HumanWAM-Goal achieves 13.94% mean zero-shot success rate. After post-training, it reaches 92.70% on Clean2Clean and 84.60% on Clean2Random, yielding a state-of-the-art average success rate of 88.65% on RoboTwin 2.0.

HumanWAM-Goal exhibits an emergent capacity for embodied in-context learning through visual goals. In real-robot experiments, the same policy adapts its execution to diferent desired outcomes specified by goal images without parameter updates. This capability is already present after joint pretraining and is retained when the model is post-trained for more complex, multi-step manipulation. Post-trained policies also follow goals depicted in egocentric human images, indicating that task-relevant spatial relationships can guide robot behavior across diferences in embodiment and visual appearance.

To support reproducible scaling research in embodied AI, we open-source the Open-HumanScale-50K corpus, data curation and pretraining pipelines, alongside the pretrained and post-trained HumanWAM checkpoints, inference code, and simulation evaluation suites. Open-HumanScale-50K is a curated and aligned 50,000-hour egocentric pretraining corpus with hand-motion and language annotations, and our experiments demonstrate consistent gains in downstream robot generalization as pretraining on this corpus scales, with out-of-distribution success rates following an empirical log-linear scaling law.

The main contributions of this work are as follows:

1. A systematic study of egocentric scaling for robot generalization. We present Open-HumanScale, a fully open recipe for scaling pure egocentric pretraining from 100 to 50k hours. We identify an empirical log-linear scaling law in downstream robot task success rates and systematically analyze the benefits of egocentric scaling for robot generalization and adaptation. Building on this recipe, we develop HumanWAM-Vanilla, which achieves success rates of 92.08% on RoboTwin 2.0 Clean2Clean and 68.88% on Clean2Random after robot post-training, alongside competitive performance on RoboDojo.

2. Emerging robot capabilities through hybrid human–robot data scaling. With HumanWAM-Goal, we show that scaling joint human–robot pretraining improves zero-shot task execution in simulation and on physical robots. The model also exhibits embodied in-context learning, adapting its behavior to visual goals without parameter updates. After robot post-training, HumanWAM-Goal achieves a state-of-the-art average success rate of 88.65% on RoboTwin 2.0, with 92.70% on Clean2Clean and 84.60% on Clean2Random.

3. A full-stack release for reproducible scaling research. We provide the complete resources for reproducing our scaling studies and HumanWAM models. The release includes Open-HumanScale-50K, a curated and aligned 50k-hour egocentric corpus with hand-motion and language annotations, together with training code, model checkpoints and evaluation protocols.

## 2 Building an Open Egocentric Pretraining Corpus

To support systematic scaling studies and large-scale egocentric pretraining, we build Open-HumanScale-50K, an open corpus from multiple publicly available egocentric datasets (Figure 2). HumanNet [11] is the primary source, contributing egocentric videos collected from the internet and paired with hand-motion and language annotations. We curate manipulation sequences based on hand visibility, motion stability, and manipulation activity. To prepare these heterogeneous sources for joint pretraining, we map action annotations into a unified bimanual end-efector representation, aligning camera coordinates, wrist axes, and gripper openness across sources to provide a common action interface for subsequent robot adaptation.

![](images/f2163c36b30f0be8d1e476dd1ce1539cae1a5db16d6a538e0f90a67819787874.jpg)  
Figure 2 Construction of Open-HumanScale-50K. (a) Five data sources and their retained durations. (b) Data curation based on hand visibility, motion stability, and manipulation activity. (c) Cross-source alignment of camera frames, wrist axes, and gripper openness.

## 2.1 Pretraining Data Sources

To support systematic studies of egocentric scaling and large-scale model pretraining, we assemble a corpus of 50,000 hours of human manipulation video paired with language descriptions and hand-motion annotations. The corpus combines HumanNet, EgoSuite-Open100K, EgoDex, Gen-HumanEgo, and EgoVerse, which provide egocentric videos with accompanying hand-tracking annotations. Table 1 summarizes the data sources and their scales before and after curation. The corpus contains 31,396,799 physical episodes, which yield 35,934,235 continuous clips after curation, as individual episodes may be split into multiple retained segments. The resulting 35.9 million clips cover a broad range of manipulation settings, including household, commercial service, and industrial environments.

HumanNet. HumanNet [11] is a large-scale human-centric dataset combining one million hours of first- and third-person videos with language and motion annotations. We use an egocentric subset of internet videos covering manipulation, tool use, and multi-step activities across diverse objects and environments. These clips are accompanied by estimated 3D wrist poses, camera trajectories, and LLM-generated captions describing the observed activities and motions. After curation and corpus selection, we retain approximately 40,744 hours for pretraining.

EgoSuite-Open100K. EgoSuite-Open100K [75] is a large-scale egocentric human activity dataset. It provides head-mounted video synchronized with head-camera poses and tracked hand poses. Each sample includes 21 joint transforms per hand, episode-level task and scene labels, and time-stamped subtask descriptions. The hand annotations support wrist-pose extraction and the estimation of hand openness from fingertip distances, while the task and scene labels enable explicit coverage measurements. We retain 6,358 hours from EgoStandard and 724 hours from EgoPro.

Table 1 Source datasets and curation statistics for Open-HumanScale-50K
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Before curation</td><td colspan="3">After curation</td></tr><tr><td>Hours</td><td>Frames (M)</td><td>Hours</td><td>Frames (M)</td><td>Episodes</td></tr><tr><td>HumanNet subset [11]</td><td>90,275.25</td><td>9,749.73</td><td>40,743.58</td><td>4,400.31</td><td>30,833,247</td></tr><tr><td>EgoSuite-Open100K [75]</td><td>11,193.99</td><td>1,208.95</td><td>7,081.65</td><td>764.82</td><td>348,724</td></tr><tr><td>EgoDex [20]</td><td>829.45</td><td>89.58</td><td>335.43</td><td>36.23</td><td>163,537</td></tr><tr><td>Gen-HumanEgo [17]</td><td>1,847.39</td><td>199.52</td><td>1,814.53</td><td>195.97</td><td>44,632</td></tr><tr><td>EgoVerse [43]</td><td>143.33</td><td>15.48</td><td>24.81</td><td>2.68</td><td>6,659</td></tr><tr><td>Total</td><td>104,289.41</td><td>11,263.26</td><td>50,000.00</td><td>5,400.00</td><td>31,396,799</td></tr></table>

EgoVerse. EgoVerse [43] is a continuously expanding collection of egocentric human demonstrations contributed by academic and industrial partners. Its initial release comprises 1,362 hours spanning 1,965 tasks, 240 scenes, and 2,087 demonstrators, combining standardized tasks collected across laboratories with open-ended activities recorded in diverse environments. The data include 3D hand and head poses, camera poses, and language annotations, with fine-grained subtask descriptions for industry-contributed recordings. We use only a small subset of the Mecka source, retaining 24.81 hours after curation for Open-HumanScale-50K.

Gen-HumanEgo. Gen-HumanEgo [17] contains over 1,800 hours of human demonstrations across more than 10,000 tasks in domestic, commercial, industrial, and agricultural environments. The dataset provides 21 reconstructed 3D keypoints per hand, MANO [45] parameters, hand meshes, and depth maps. Hierarchical video-, task-, and subtask-level language annotations describe activities at diferent granularities, with temporal boundaries for tasks and subtasks.

EgoDex. EgoDex [20] contains 829 hours of egocentric human demonstrations covering 194 tabletop manipulation tasks. Collected with Apple Vision Pro, it focuses on active manipulation of everyday objects, including dexterous activities such as tying shoelaces and folding laundry. The dataset pairs 30 Hz RGB video with 3D head, upper-body, wrist, and finger poses, camera calibration, and natural-language task descriptions.

## 2.2 Data Curation

Efective large-scale pretraining depends on both corpus scale and supervision quality [34, 66]. We curate egocentric recordings to retain active manipulation with clear visual context and temporally consistent motion annotations. These checks remove unreliable annotations and prolonged inactivity while retaining usable manipulation segments for pretraining.

Hand visibility. We assess hand visibility from action annotations to retain manipulation sequences with efective visual context. We project annotated 3D EEF positions into the image using camera calibration to determine whether each end efector lies within the field of view. When only one end efector is in view, we retain the sequence and restrict action supervision to the corresponding hand, excluding the other hand from both the action loss and normalization statistics. Internet videos may also contain incomplete hand views, with only fingers or a small portion of a hand visible near image boundaries, or excessively close views that limit the visible interaction context. We identify and filter these cases using EEF depth, projected scale, and the location and size of detected hand regions.

Action and camera stability. Errors in estimated hand and camera poses can introduce abrupt jumps into manipulation trajectories. For EEF trajectories, we measure diferences between original and smoothed positions, together with acceleration, jerk, and changes in position and orientation between adjacent frames. Camera trajectories are checked separately for abrupt pose changes and for excessive displacement or rotation relative to the pose at the start of each chunk. We remove frames that exceed any stability threshold and split sequences at those frames, retaining continuous segments for pretraining.

Idle filtering. Long periods of inactivity introduce redundant static samples to the pretraining corpus. We measure EEF positional variation within 32-frame sliding windows and classify a window as inactive when this variation is at or below the motion threshold. We trim inactive portions from both ends of each chunk and discard chunks with fewer than 70% active windows.

## 2.3 Unified Action Representation

Joint pretraining on heterogeneous egocentric datasets requires consistent action supervision across sources. Although these datasets broaden the range of manipulation experience, their raw action data can encode similar physical motions diferently, complicating the learning of shared visual–action relationships. We therefore align human trajectories and robot actions within a unified bimanual end-efector space. This alignment gives actions consistent geometric meaning across sources, allowing diverse demonstrations to contribute to a common action prediction objective.

Unified bimanual action space. To establish a common action interface between egocentric pretraining and robot adaptation, we map human and robot trajectories into a unified bimanual end-efector (EEF) space. Human wrist poses provide EEF positions and orientations, while gripper openness is derived from the available hand annotations. The resulting action vector is

$$
\mathbf { a } _ { t } = [ \mathbf { p } _ { t } ^ { L } , \mathbf { q } _ { t } ^ { L } , g _ { t } ^ { L } , \mathbf { p } _ { t } ^ { R } , \mathbf { q } _ { t } ^ { R } , g _ { t } ^ { R } ] \in \mathbb { R } ^ { 1 6 } ,\tag{1}
$$

where $ { \mathbf { p } } _ { t } ^ { h } \in \mathbb { R } ^ { 3 }$ is the position in meters, $\mathbf { q } _ { t } ^ { h } \in \mathbb { R } ^ { 4 }$ is a unit quaternion in wxyz order, and $g _ { t } ^ { h } \in [ 0 , 1 ]$ denotes gripper openness for hand $h \in \{ L , R \}$ . This representation captures bimanual motion and hand opening without requiring a shared joint configuration across human and robot embodiments. We restrict the action space to bimanual manipulation without introducing additional dimensions for locomotion, which is motivated by the dificulty of deriving meaningful supervision from largely task-irrelevant, irregular human body or head motion in egocentric data, compounded by incomplete annotations across sources.

Camera-frame alignment. A shared end-efector representation does not by itself resolve geometric diferences between source-specific reference frames. To align action supervision with visual observations, we transform wrist trajectories into the camera frame associated with each video frame. For sources with world-frame wrist trajectories and camera extrinsics, let $T _ { A B }$ map coordinates from frame B to frame A. The wrist pose in the camera frame is

$$
T _ { C _ { \mathrm { c v } } } ^ { h } ( t ) = T _ { W C _ { \mathrm { c v } } } ( t ) ^ { - 1 } T _ { W E } ^ { h } ( t ) .\tag{2}
$$

Wrist and camera poses are synchronized at each timestamp, so each action is expressed relative to the camera at the corresponding target time. We further standardize the camera axes to +x right, +y up, and +z backward. For example, EgoVerse camera coordinates use +y down and +z forward. We convert these coordinates by left-multiplying both position vectors and rotation matrices by $S = \mathrm { d i a g } ( 1 , - 1 , - 1 )$ ,

$$
\mathbf { p } _ { C E } ^ { h } ( t ) = S \mathbf { p } _ { C _ { \mathrm { c v } } E } ^ { h } ( t ) , \qquad R _ { C E } ^ { h } ( t ) = S R _ { C _ { \mathrm { c v } } E } ^ { h } ( t ) .\tag{3}
$$

Wrist-axis alignment. A common camera frame does not resolve diferences in the local coordinate axes assigned to each wrist. We define a canonical wrist frame with +X pointing from the wrist toward the middle-finger knuckle, +Y along the thumb direction projected onto the plane orthogonal to $+ X$ , and $+ Z = X \times Y$ . For each source s and hand h, a constant rotation $C _ { s , h } \in \mathrm { S O } ( 3 )$ aligns the native wrist axes with this convention,

$$
R _ { \mathrm { a l i g n e d } } ^ { h } ( t ) = R _ { \mathrm { c a m e r a } } ^ { h } ( t ) C _ { s , h } .\tag{4}
$$

For sources with 3D hand keypoints and native wrist orientations, we estimate $C _ { s , h }$ through orthogonal Pro crustes alignment [46] with anatomical frames derived from the keypoints. Gen-HumanEgo wrist orientations are constructed directly from these anatomical frames. For HumanNet, we determine the axis permutations and sign changes using projected wrist axes and monocular hand landmarks.

Gripper alignment. For sources with hand keypoints, we map the thumb–index fingertip distance $d _ { t } ^ { h }$ to normalized gripper openness,

$$
g = \mathrm { c l i p } \left( \frac { d - d _ { \mathrm { c l o s e d } } } { d _ { \mathrm { o p e n } } - d _ { \mathrm { c l o s e d } } } , 0 , 1 \right) .
$$

(b) Model Architecture  
![](images/658ca4728b2b117b554c4f74ff6ee70e2415775d59e116fa87d972355f418368.jpg)

![](images/b8846bd063281252d721e9f0769cbd9bfaa2f2090ea3f5dcfb0ff10859065159.jpg)  
Figure 3 Model architecture of HumanWAM-Vanilla and experimental protocol. (a) Egocentric pretraining, robot post-training on clean demonstrations, and closed-loop evaluation under clean, scene-ood, and random conditions. (b) Conditioned on language and proprioception, the video and action experts interact through joint masked self-attention. The masks specify visibility among current-frame (C), future-video (F), and action (A) tokens under isolated attention (IA) and inverse-dynamics (IDM) attention.

Here, $d _ { \mathrm { c l o s e d } }$ and $d _ { \mathrm { o p e n } }$ denote reference distances for closed and open hand configurations, respectively. These distances are specified for each data source, with separate values for the two hands where applicable. Across sources, 0 denotes fully closed and 1 denotes fully open.

## 3 Exploration of Pure Egocentric Data Scaling

Egocentric data ofer a promising route to scaling robot foundation models through abundant and diverse human manipulation experience. Prior work has investigated how pretraining exclusively on human egocentric data afects performance on downstream real-robot tasks [33, 73]. However, scaling outcomes vary across pretraining recipes, with limited or inconsistent downstream generalization gains reported in other settings [50, 56]. We first establish empirical scaling laws for egocentric pretraining with our proposed recipe, then examine data diversity and the sample eficiency of robot adaptation, and finally conduct controlled analyses of the pretraining and adaptation recipe.

We organize the scaling analysis around the following questions:

• Q1. What improves in downstream robot policies as egocentric pretraining scales from 100 to 50k hours?

• Q2. How important is data diversity for efective egocentric scaling?

• Q3. How does scaling egocentric pretraining afect the sample eficiency of robot adaptation?

• Q4. Does cross-source action alignment matter for egocentric pretraining?

• Q5. Does predicting human motion during egocentric pretraining improve robot action learning?

• Q6. Can egocentric pretraining surpass robot pretraining at matched data scales?

• Q7. How do we extend single-view egocentric pretraining to multi-view robot policies?

## 3.1 Pretraining and Evaluation Protocol

To study scaling in pure egocentric pretraining, we adopt a unified world–action model backbone and a standardized training and evaluation pipeline (Figure 3). After pretraining on human egocentric data, we adapt each model to RoboTwin 2.0 using clean demonstrations and measure its task success rate in closed-loop evaluation. Standardized tasks and success criteria, controlled environmental variation, and shared evaluation seeds support reproducible comparisons across pretraining scales and training settings. Evaluations within the demonstration distribution and under environmental randomization measure in-distribution performance and out-of-distribution generalization, respectively. Below, we describe the model architecture and the pretraining, robot post-training, and evaluation procedures.

## 3.1.1 Model.

Our experiments build on HumanWAM-Vanilla, a world–action model that jointly generates future video and actions from a single observation and a language instruction. The model extends Wan2.2-TI2V-5B [53] with an action expert that has the same depth as the video backbone, forming a Mixture-of-Transformers architecture. Following prior work [35, 67, 70], we apply a widely used isolated attention mask in most experiments, allowing action tokens to attend to the first video frame but not to future video frames. During inference, the model can therefore omit future video generation and denoise only the action chunk, enabling fast policy execution. We use the unified 16-D bimanual end-efector action space defined in Section 2.3 and compute delta poses in the camera coordinate frame relative to the first-frame observation.

## 3.1.2 Pretraining.

We pretrain on diferent egocentric corpora with the same training recipe. We use the AdamW optimizer [30] with a peak learning rate of $1 0 ^ { - 4 }$ and a weight decay of 0.01. The learning rate follows a cosine schedule over 100k steps, with a linear warm-up during the first 5% of the steps. The training budget is adapted to the corpus size. In preliminary runs we observed that smaller corpora saturate quickly and that their generalization can even degrade with further training, so a fixed budget would either undertrain the large corpora or over-train the small ones. We therefore adjust the global batch size according to the corpus size, as listed in Table 2, and we post-train and evaluate checkpoints saved every 10k pretraining steps so that each corpus is compared at its own best.

Table 2 Pretraining settings across data scales.
<table><tr><td>Hours</td><td>Frames (M)</td><td>Global batch size</td><td>Epochs / 10k steps</td></tr><tr><td>100</td><td>10.8</td><td>512</td><td>0.58</td></tr><tr><td>500</td><td>54.0</td><td>1,024</td><td>0.23</td></tr><tr><td>1,000</td><td>108.0</td><td>2,048</td><td>0.23</td></tr><tr><td>5,000</td><td>540.0</td><td>2,048</td><td>0.05</td></tr><tr><td>10,000</td><td>1,080</td><td>4,096</td><td>0.04</td></tr><tr><td>20,500</td><td>2,214</td><td>4,096</td><td>0.02</td></tr><tr><td>50,000</td><td>5,400</td><td>8,192</td><td>0.02</td></tr></table>

## 3.1.3 Post-training.

We post-train the pretrained model on RoboTwin 2.0 [8], a bimanual tabletop benchmark with 50 tasks on the Aloha-AgileX embodiment. The model is post-trained on 50 clean demonstrations per task without data augmentation for 12k optimizer steps with a global batch size of 512, corresponding to approximately 10 epochs. For each pretraining run, we post-train checkpoints saved every 10k pretraining steps and evaluate the resulting policies in both clean and out-of-distribution settings, as described below, to track downstream performance and identify the best checkpoint for each corpus. We retain the single head-camera view from egocentric pretraining because introducing three camera views during post-training impairs the transfer of egocentric pretraining priors and head-camera images alone provide suficient visual information to perform all RoboTwin tasks (see Section 3.8).

## 3.1.4 Evaluation.

To assess how egocentric pretraining afects in-distribution performance and out-of-distribution generalization, we evaluate post-trained checkpoints under the clean, scene-ood, and random settings. Each checkpoint is rolled out on all 50 tasks with 50 episodes per task in each setting. Representative tasks and evaluation conditions are illustrated in Figure 4.

![](images/ccaa13a2ac40dbe0b182a2d508d945e5032a813f6487c35d32b95ab91c56ff80.jpg)  
Figure 4 RoboTwin evaluation setup for our egocentric pretraining exploration. We post-train policies exclusively on clean demonstrations and evaluate them under clean, scene-ood, and random conditions.

• clean: the training distribution, with the plain background, fixed lighting, no clutter, and fixed table height used to collect the demonstrations.

• scene-ood: randomized backgrounds and tabletop textures, with all other conditions identical to clean.

• random: the full RoboTwin randomization, with randomized backgrounds, table clutter, randomized lighting, and table-height jitter of up to 3 cm.

Clean measures how well the policy fits the demonstration distribution, whereas scene-ood and random assess gains in downstream generalization from egocentric pretraining. Specifically, scene-ood focuses on robustness to visual scene variation. We find that the post-trained checkpoint with the best clean performance does not necessarily generalize best to randomized scenes. To examine pretraining gains in in-distribution performance and out-of-distribution generalization separately, we therefore report the success rate of the best checkpoint independently for each evaluation setting. To support fair comparisons and accelerate evaluation, we save expert-validated evaluation seeds and reuse them across evaluation rounds.

## 3.2 Scaling Egocentric Data from 100 to 50k Hours: What Improves?

Whether and how scaling human data improves robot policy performance is a central question for robot foundation models. We investigate this question through a systematic study of egocentric pretraining from 100 to 50k hours. Under the same pretraining recipe, we find that scaling egocentric pretraining consistently improves downstream generalization across all three settings. We validate this trend under both isolated attention and inverse-dynamics (IDM) attention, which additionally allows action tokens to attend to future video.

We first examine scaling with isolated attention. Models are pretrained on 100, 500, 1,000, 5,000, and 10,000 hours of egocentric data, then post-trained and evaluated under a common downstream protocol. Figure 5 shows that scaling egocentric pretraining with isolated attention consistently improves downstream policy performance on RoboTwin. On clean, success increases from 71.2% at 100 hours to 80.0% at 10,000 hours, compared with 69.5% without egocentric pretraining. These gains indicate improved in-distribution performance under the same post-training protocol. The benefits are larger under distribution shift. From 100 to 10,000 hours, scene-ood success rises from 25.2% to 58.1%, while random success increases from 8.4% to 28.7%. Without egocentric pretraining, the corresponding success rates are only 16.2% and 2.1%.

![](images/be4d6a07114e4c7d15a54c11d187d2f665757c0b67cc76bc3f416d7f5638f3ee.jpg)

![](images/bc0396d5c071dabeff9f8e6f1b17600af43609d37bfb20d3917fc16d122dd187.jpg)

![](images/07c138dd3cdde0fbc1e764d15f97ef5fd9b925276e340fd43f769aea4e0c37ef.jpg)  
Figure 5 Scaling egocentric pretraining with isolated attention improves in-distribution performance and generalization. Blue dashed lines show least-squares fits of $\mathrm { S R } = a + b \log _ { 1 0 } H ,$ where H denotes pretraining hours, with $R ^ { 2 }$ reported in each panel. Gray dashed lines denote the baseline without egocentric pretraining. Success rates increase across all three settings, with larger gains on scene-ood and random, indicating that scaling primarily strengthens generalization beyond the demonstration scenes.

Downstream success follows an empirical log-linear scaling law, $\mathrm { S R } = a + b \log _ { 1 0 } H $ , where H denotes egocentric pretraining hours. The high goodness of fit $( R ^ { 2 } = 0 . 9 7 1 , 0 . 9 4 9$ , and 0.877 for clean, scene-ood, and random, respectively) indicates a consistent relationship between pretraining scale and downstream performance. Each tenfold increase in pretraining hours corresponds to fitted gains of 4.43 pp on clean, 15.88 pp on scene-ood, and 10.24 pp on random. These trends suggest that further scaling egocentric pretraining will continue to improve downstream policy performance, with particularly pronounced gains in out-of-distribution generalization.

Recent work motivates us to examine whether this scaling trend extends to architectures that condition actions on future visual information. We therefore validate the scaling experiments with inverse-dynamics attention and absolute camera-frame poses, allowing action tokens to attend to both current and future video frames. During training, action denoising is conditioned on the ground truth future video with noise augmentation. During inference, the model first generates future video latents and then predicts actions. This configuration substantially improves downstream performance, with detailed ablations examining the contributions of these design choices in Section 4.4.

As shown in Figure $6 ,$ clean success remains between 87.4% and 89.0% as egocentric pretraining scales from 100 to 50,000 hours. Even without egocentric pretraining, the model achieves 87.9% on clean, indicating strong in-distribution performance under IDM attention with absolute camera-frame poses. Scaling therefore yields only limited additional gains on the training distribution. The generalization gains are substantially larger. From 100 to 50,000 hours, scene-ood success increases from 49.1% to 73.2%, and random success rises from 21.5% to 42.9%, compared with 29.2% and 7.7% without egocentric pretraining. Improvements persist beyond 5,000 hours, where the corresponding success rates are 62.4% and 32.2%. Both out-of-distribution settings closely follow the empirical log-linear scaling law observed with isolated attention, with higher $R ^ { 2 }$ values of 0.969 for scene-ood and 0.955 for random. This consistency across the two model configurations suggests that the generalization benefits of scaling egocentric pretraining extend beyond a specific attention design and action representation, even when in-distribution performance is already strong

## 3.3 Scaling Diversity, Not Just Hours

Recent studies [14, 33, 73] have explored the scaling behavior of egocentric pretraining, with data duration as the primary scaling variable. Data diversity, including variation in tasks and scenes, is another important dimension of scaling embodied pretraining [27, 47]. We investigate this dimension through controlled experiments on the EgoSuite-Open100K dataset with matched durations but varying task and scene coverage. Our results show that jointly scaling diversity and duration yields greater downstream gains than increasing duration alone.

![](images/10e8b7f830dbb99370acec8d5349e5af68ef7c3a7b244a31eed078a7cd6272d9.jpg)  
Figure 6 Scaling egocentric pretraining with IDM attention strengthens downstream generalization. Clean success remains high across scales, while scene-ood and random success improve consistently. Blue dashed lines show log-linear scaling fits for the two out-of-distribution settings, with R<sup>2</sup> reported in the corresponding panels. Gray dashed lines denote the baseline without egocentric pretraining.

Setup. We construct two series of pretraining corpora at 100, 500, 1,000, and 5,000 hours from the EgoSuite-Open100K dataset. In the scaled-diversity series, task and scene coverage expands with data duration. In the matched-diversity series, the 100-, 500-, and 1,000-hour subsets cover the same set of tasks and scenes as the 5,000-hour parent corpus, so increasing duration adds trajectories within fixed task and scene coverage. At 100 hours, the matched-diversity corpus spans 2,288 tasks and 2,953 scenes, compared with 62 tasks and 222 scenes in the scaled-diversity corpus. For each corpus, we pretrain the model, apply the same post-training protocol, and evaluate success rates under all three settings.

Table 3 Task and scene coverage of the two EgoSuite-Open100K corpus families and their shared 5,000-hour parent corpus. Task and scene counts are based on the dataset captions.
<table><tr><td>Family</td><td>Hours</td><td>Clips</td><td>Tasks</td><td>Scenes</td></tr><tr><td rowspan="3">Scaled-diversity</td><td>100</td><td>62,206</td><td>62</td><td>222</td></tr><tr><td>500</td><td>305,889</td><td>309</td><td>961</td></tr><tr><td>1,000</td><td>589,216</td><td>547</td><td>1,442</td></tr><tr><td rowspan="3">Matched-diversity</td><td>100</td><td>81,076</td><td>2,288</td><td>2,953</td></tr><tr><td>500</td><td>345,053</td><td>2,288</td><td>2,953</td></tr><tr><td>1,000</td><td>653,567</td><td>2,288</td><td>2,953</td></tr><tr><td>Shared parent</td><td>5,000</td><td>2,813,547</td><td>2,288</td><td>2,953</td></tr></table>

Results. Figure 7 shows that expanding diversity alongside duration produces steeper scaling slopes than increasing duration at fixed coverage. Relative to matched-diversity, scaled-diversity increases the fitted slopes by 2.02 on clean, 4.73 on scene-ood, and 4.50 on random, indicating greater performance gains as pretraining data scale. The larger slope increases on scene-ood and random suggest that diversity expansion particularly benefits downstream generalization. These results suggest a practical scaling rule for egocentric pretraining: jointly scaling diversity and duration yields greater performance gains than scaling duration alone.

## 3.4 Scaling Egocentric Pretraining for Data-Efficient Robot Adaptation

Generalization to new tasks and in-context learning have attracted growing interest in embodied pretraining [16, 21, 28]. In this section, we seek to explore how egocentric pretraining scale afects the sample eficiency of downstream robot policy adaptation.

![](images/23a08ba280d892d587d974187dfd959272779dcf528746d922c07e8b77acd5d2.jpg)  
Figure 7 Scaling egocentric pretraining with growing or fixed task and scene diversity. Blue circles denote scaled-diversity corpora, and orange squares denote matched-diversity corpora. Dashed and dash-dotted lines show the respective least-squares fits of ${ \mathrm { S R } } = a + b \log _ { 1 0 } H _ { \mathrm { : } }$ with both equations shown in each panel. Scaled-diversity yields a steeper fitted slope in all three settings

Setup. To assess how egocentric pretraining scale afects the sample eficiency of robot adaptation, we conduct post-training experiments with diferent initializations and demonstration budgets. We select the best-performing checkpoints from pretraining on 500 and 5,000 hours of egocentric data and compare them with a baseline without egocentric pretraining (No-Ego). For each initialization, separate post-training runs use 5, 10, 25, or 50 clean demonstrations per task.

![](images/2553692b57b1d91f98ef2eba158d2c8882c906b949ae90f0f0e538a2a9e0de27.jpg)  
Figure 8 Scaling egocentric pretraining improves the sample efficiency of robot adaptation. Success rates as a function of the number of robot demonstrations per task for diferent egocentric pretraining scales.

Results. Figure 8 shows a consistent scaling trend in post-training sample eficiency. Across all three settings, the 500-hour model outperforms No-Ego at every demonstration budget, with further gains from scaling to 5,000 hours. With 10 demonstrations per task, clean success increases from 34.5% for No-Ego to 44.2% after 500 hours of pretraining and 50.1% after 5,000 hours, indicating progressively more efective adaptation from limited robot demonstrations. At the same budget, scaling from 500 to 5,000 hours also approximately doubles random success, from 4.1% to 8.3%. The additional pretraining therefore improves both in-distribution performance and generalization without requiring more robot demonstrations.

These gains also lower demonstration requirements for comparable performance. On scene-ood, the No-Ego success rate with 50 demonstrations per task is exceeded by the 500-hour model with 25 demonstrations and by the 5,000-hour model with only 5. Further scaling from 500 to 5,000 hours continues to reduce dependence on robot demonstrations. With 25 demonstrations, the 5,000-hour model outperforms the 500-hour model trained on 50 demonstrations in both scene-ood (42.0% vs. 37.9%) and random (21.8% vs. 16.2%). This improvement establishes a clear scaling trend in post-training sample eficiency, with stronger generalization achieved using half the robot demonstrations.

## 3.5 Does Cross-Source Action Alignment Matter?

Egocentric datasets obtain human action trajectories through diferent data collection and motion estimation pipelines, producing raw action data that lack a consistent physical interpretation across sources. Prior work incorporates geometric alignment to support learning from heterogeneous human and robot data [25, 34, 66]. We quantify the efect of camera-frame and wrist-axis alignment on pretraining loss and downstream policy performance under a fixed data mixture and training configuration.

Setup. To evaluate the efect of cross-source action alignment, we construct a 1,000-hour corpus comprising EgoSuite-Open100K (593.7 h), HumanNet (250 h), EgoDex (114.5 h), and EgoVerse (41.8 h). We compare two models pretrained on identical clips and captions with the same sampling strategy, training schedule, and normalization procedure. In the Aligned setting, actions from each source are transformed into camera coordinates, and wrist axes are standardized to obtain a consistent action representation. In the Raw setting, actions from each source are used directly in their original representation.

![](images/48a5bf1ff6742f0898b42d1ef62cb56bae45b454bd49edd2719a4ca6f29b81f9.jpg)

![](images/df73b82b72984f0df4aa08b747f09cfa93695220c7156537da4b2e21a495a331.jpg)

![](images/b309731733b14f55e565e3dd31eef26e2f8c5cf83e3c93a9a1246f071818e6e2.jpg)  
Figure 9 Cross-source action alignment improves downstream policy performance. Aligned and Raw models are pretrained on the same four-source 1,000-hour corpus using aligned and native action representations. Each point reports the best success rate across post-trained checkpoints for the corresponding pretraining run and evaluation setting. Gray dashed lines denote the baseline without egocentric pretraining.

Results. Figure 9 shows that Aligned outperforms Raw at every evaluated pretraining step across all three settings. Comparing the best checkpoints, clean success increases from 72.0% to 77.5% (+5.5 pp). The gains are larger under distribution shift, with scene-ood improving from 43.2% to 52.2% and random from 18.0% to 26.4%. Both pretrained models outperform the No-Ego baseline, showing that multi-source pretraining remains beneficial without alignment. Under the same data mixture and training protocol, cross-source action alignment further improves in-distribution performance and, more substantially, out-of-distribution generalization.

The efect of action alignment is also evident during pretraining (Figure 10). Aligned exhibits lower action loss, while video loss remains comparable between the two models. These trends suggest that a well-aligned action representation provides more consistent supervision across sources and facilitates joint learning.

## 3.6 Action Supervision in Egocentric Pretraining

Egocentric human data capture rich physical interactions in first-person videos paired with hand trajectories [20, 43]. Whether to incorporate human motion prediction into world–action model pretraining remains an open question [7, 35, 50, 59]. We show that jointly predicting human motion and future video substantially improves downstream policy generalization over video prediction alone on the same egocentric corpus.

![](images/405cb014eb25da320476687f2865dca98218d97d24bbca418b29898eccecd8f0.jpg)  
Figure 10 Cross-source action alignment reduces pretraining action loss with little change in video loss. Curves show training losses for Aligned and Raw models on the same four-source 1,000-hour corpus.

Setup. We pretrain two models on the same 1,000-hour scaled-diversity corpus, with and without human motion prediction. For the video-only baseline, we set the action loss weight to zero and optimize video prediction alone. To assess the additional value of human motion targets, we train a second model with action loss enabled only for a fixed subset of original episodes covering approximately 50% of the total video duration. All models receive full robot action supervision during post-training, following Section 3.1.

![](images/5646722e0adf4d7b3a1237c85c221068980c7cba7362eef6ced24187d96374f1.jpg)  
Figure 11 Human action supervision during egocentric pretraining improves downstream generalization. Models are pretrained on the same 1,000-hour video corpus with 0%, 50%, or 100% human action supervision.

Results. Figure 11 shows that learning human motion alongside video prediction consistently improves downstream robot policy performance across all three settings. At 20k pretraining steps, using motion targets for approximately 50% of the video duration increases clean success from 70.4% to 75.2% compared with video-only pretraining. The gains are larger under distribution shift, with scene-ood improving from 13.0% to 34.8% and random from 3.6% to 16.3%. The two objectives also exhibit distinct trends as pretraining progresses. From 10k to 30k steps, video-only performance remains nearly flat, with scene-ood and random success staying near 13% and 3%, respectively. Over the same interval, the model that jointly predicts human motion and future video improves steadily from 30.0% to 40.4% on scene-ood and from 12.3% to 20.1% on random. The widening gap indicates that human motion targets improve both performance at matched pretraining steps and the gains in downstream generalization from continued pretraining.

Table 4 Composition of the real-robot pretraining corpora.
<table><tr><td>Scale (h)</td><td>AgiBot Beta (h)</td><td>XDOF/ABC-130k (h)</td><td>Cobot (h)</td><td>Total frames</td></tr><tr><td>100</td><td>32.19</td><td>67.80</td><td></td><td>10,798,713</td></tr><tr><td>500</td><td>176.52</td><td>323.46</td><td></td><td>53,997,982</td></tr><tr><td>1,000</td><td>361.70</td><td>638.16</td><td></td><td>107,985,621</td></tr><tr><td>5,000</td><td>1,944.80</td><td>2,983.46</td><td>71.52</td><td>539,977,159</td></tr></table>

## 3.7 Can Egocentric Pretraining Surpass Robot Pretraining?

Egocentric human data ofer broader interaction coverage, greater scene diversity, and higher information density at lower collection cost than teleoperated robot data [33]. To assess their relative value for robot policy learning, HumanScale pretrained models on equal-duration egocentric or real-robot corpora, then post-trained each model on the same real-robot demonstrations. This comparison favored egocentric pretraining, which yielded lower validation action loss on held-out real-robot trajectories, particularly on tasks excluded from post-training. We extend this matched-scale comparison to closed-loop task success on RoboTwin to examine how the relative performance of the two pretraining sources varies with data scale.

Setup. We compare egocentric and real-robot pretraining at 100, 500, 1,000, and 5,000 hours. The egocentric series uses the scaled-diversity corpora in Figure 5, while the real-robot series draws on AgiBot Beta [1], ABC-130k [2], and our in-house Cobot Magic data as shown in Table 4. Both series use only head-camera observations during pretraining, given the efects of additional wrist-camera views discussed in Section 3.8. We curate the robot corpora from language-annotated trajectories with verified head-camera extrinsics, excluding data with base or head-camera motion.

![](images/30bb15870a8b45fd59f2cd05665350db6981cbd9a6bfce1a3a1e8e108cf05931.jpg)  
Figure 12 Egocentric pretraining outperforms robot pretraining from 500 to 5,000 hours. Downstream RoboTwin success rates after pretraining on 100, 500, 1,000, and 5,000 hours of egocentric human or robot data. The egocentric series follows the scaled-diversity configuration in Figure 5. Each point reports the best post-training success rate among the evaluated pretraining checkpoints for that scale and setting.

Results. Figure 12 shows that egocentric pretraining outperforms robot pretraining in all three settings from 500 to 5,000 hours. At 100 hours, robot pretraining retains a small advantage on scene-ood and random, but this ordering reverses at larger scales. On clean, the egocentric advantage grows from 2.2 pp at 100 hours to 7.3 pp at 5,000 hours. At 5,000 hours, egocentric pretraining also achieves higher out-of-distribution success, reaching 52.3% versus 45.6% on scene-ood and 27.6% versus 22.4% on random. Its advantage therefore spans both in-distribution performance and robustness under distribution shift. The 1,000-hour egocentric model also outperforms the 5,000-hour robot model in all three settings, including scene-ood (48.2% versus 45.6%)

and random (25.9% versus 22.4%). These comparisons extend the validation-loss evidence from HumanScale to closed-loop task success on RoboTwin, where egocentric pretraining yields stronger downstream policies despite the embodiment gap.

## 3.8 From Egocentric Pretraining to Multi-View Robot Policies

Most egocentric human data are collected with a single head-mounted camera [20, 43]. Although head-camera observations alone can support most robot manipulation tasks, wrist cameras provide complementary closerange views for precise gripper–object interactions and improve visibility when the head view is occluded [23, 71]. We find that introducing wrist-camera inputs directly during robot post-training can substantially damage the generalization benefits of egocentric pretraining. EgoPro [75] provides approximately 1,000 hours of egocentric human video with synchronized head- and wrist-camera views. We use this dataset to investigate how to extend egocentric pretraining to multi-view robot policies while preserving its generalization benefits.

![](images/0c43c381e0396e2e49b98d8bba760d898ad1bbea7a28b9cd5397b136b8f130bb.jpg)  
Figure 13 From egocentric pretraining to multi-view robot policies. Three view configurations and training routes with or without EgoPro mid-training.

Multi-view configurations. We compare three configurations for visual conditioning and video prediction, illustrated in Figure 13. Single-view uses the current head-camera observation and predicts future video from the single view. Multi-view conditioning adds current observations from both wrist cameras while restricting future video prediction to the head camera. Multi-view prediction uses the same three-camera inputs and predicts future video for all three views. We evaluate these variants separately under isolated attention with relative EEF actions and IDM attention with absolute camera-frame actions. Within each model configuration, all three variants are post-trained on RoboTwin from the same 20k-step egocentric pretraining checkpoint.

Table 5 From egocentric pretraining to multi-view robot policies. RoboTwin success rates (%) from the best available complete post-training evaluation for each setting.
<table><tr><td>Model configuration</td><td>View configuration</td><td>Mid-training</td><td>Clean</td><td>Scene-OOD</td><td>Random</td></tr><tr><td rowspan="3">Isolated attention (relative)</td><td>Single-view</td><td>None</td><td>75.68</td><td>46.08</td><td>22.48</td></tr><tr><td>Multi-view conditioning</td><td>None</td><td>74.60</td><td>34.08</td><td>12.92</td></tr><tr><td>Multi-view prediction</td><td>None</td><td>75.04</td><td>33.00</td><td>14.84</td></tr><tr><td>IDM attention (absolute)</td><td>Single-view</td><td>None</td><td>88.48</td><td>62.00</td><td>28.56</td></tr><tr><td rowspan="4"></td><td>Multi-view conditioning</td><td>None</td><td>90.56</td><td>59.52</td><td>29.08</td></tr><tr><td></td><td>EgoPro</td><td>90.28</td><td>61.20</td><td>30.68</td></tr><tr><td>Multi-view prediction</td><td>None</td><td>89.84</td><td>56.28</td><td>26.16</td></tr><tr><td></td><td>EgoPro</td><td>88.72</td><td>58.80</td><td>27.36</td></tr></table>

Table 5 shows that introducing wrist views during robot post-training under isolated attention degrades the out-of-distribution generalization gained from egocentric pretraining, while clean success decreases only slightly. Relative to Single-view, scene-ood success falls from 46.1% to 33.0–34.1% across the two multi-view variants, while random success falls from 22.5% to 12.9–14.8%. Under IDM attention, Multi-view conditioning improves clean success from 88.5% to 90.6% and maintains comparable random success, while scene-ood success decreases from 62.0% to 59.5%. With the input views unchanged, Multi-view prediction lowers success in all three settings relative to Multi-view conditioning, with scene-ood decreasing from 59.5% to 56.3% and random from 29.1% to 26.2%.

Multi-view mid-training. Under IDM attention, both multi-view configurations achieve lower scene-ood success than Single-view after direct robot post-training. We therefore investigate whether mid-training on EgoPro, which provides human egocentric videos with wrist-camera views, can bridge single-view egocentric pretraining and multi-view robot post-training while preserving the generalization gained from human data. For both Multi-view conditioning and Multi-view prediction, we compare this intermediate stage followed by robot post-training with direct post-training from the same pretrained checkpoint, keeping each view configuration fixed across mid-training and robot post-training.

Table 5 shows that mid-training improves out-of-distribution performance for both view configurations. With Multi-view conditioning, scene-ood success increases from 59.5% to 61.2% and random success from 29.1% to 30.7%. Multi-view prediction also improves, from 56.3% to 58.8% on scene-ood and from 26.2% to 27.4% on random. Clean success changes little with conditioning and decreases with prediction, so the gains are concentrated in out-of-distribution generalization. After mid-training, Multi-view conditioning approaches Single-view on scene-ood (61.2% versus 62.0%) and exceeds it on random (30.7% versus 28.6%). It also outperforms Multi-view prediction in all three settings, supporting our choice of Multi-view conditioning mid-training before robot post-training.

## 4 Pure Egocentric Data Foundation Model Training

Building on the exploration in Section 3, we develop HumanWAM-Vanilla, a world–action model pretrained exclusively on 50,000 hours of curated egocentric human data with aligned action trajectories. HumanWAM Vanilla achieves competitive performance on the bimanual simulation benchmarks RoboTwin 2.0 and RoboDojo following robot post-training, with strengths in environmental robustness, precise control, and multi-step manipulation. Real-robot experiments further show that egocentric pretraining improves task success over a baseline without egocentric pretraining, with particularly large gains under visual distribution shifts.

## 4.1 Model and Training Setup

Model configuration. HumanWAM-Vanilla follows the world–action architecture described in Section 3.1.1 and uses inverse-dynamics attention, allowing action tokens to attend to both current and future video frames. During training, action denoising is conditioned on ground-truth future video latents with noise augmentation. For policy inference, the model first generates future video latents and then predicts an action chunk conditioned on these latents.

Action representation and normalization. We use the unified 16-D bimanual end-efector action space defined in Section 2.3. Each end efector is represented by its three-dimensional position, quaternion orientation, and gripper openness. EEF poses are expressed as absolute poses in the camera frame at the corresponding target time, with camera and wrist axes aligned across sources. Actions and proprioceptive states are standardized using dimension-wise z-score normalization after action alignment. Pretraining statistics are estimated jointly across the curated corpus, while mid-training and robot post-training use statistics estimated from their respective training datasets. Invalid or unsupervised hand dimensions are excluded from both normalization statistics and the action loss.

Egocentric pretraining. We pretrain the model exclusively on Open-HumanScale-50K curated and aligned as described in Section 2, comprising human videos, language annotations, and action trajectories. The action chunk length is 32, and video frames are sampled every four action steps, yielding nine frames per sample. We optimize the model with AdamW [30], setting the global batch size to 8,192 and the weight decay to 0.01.

The learning-rate schedule spans 100,000 optimizer steps, with linear warm-up to $1 0 ^ { - 4 }$ over the first 5,000 steps followed by cosine decay to 10<sup>−6</sup>.

Mid-training. After egocentric pretraining with a single head-camera view, mid-training adapts the model to multi-view inputs for robot policies. We continue training on EgoPro [75], a human egocentric dataset with synchronized video from one head-mounted camera and two wrist-mounted cameras. Under Multi-view conditioning (Section 3.8), current observations from all three cameras condition action prediction, while future video prediction remains restricted to the head camera.

Table 6 Post-training settings for RoboTwin 2.0, RoboDojo, and real-robot experiments.
<table><tr><td>Setting</td><td>Robot demonstrations</td><td>Optimizer steps</td><td>Global batch size</td></tr><tr><td>RoboTwin 2.0</td><td>2500</td><td>24,000</td><td>512</td></tr><tr><td>RoboDojo</td><td>3500</td><td>50,000</td><td>512</td></tr><tr><td>Real robot</td><td>50/task</td><td>10,000</td><td>256</td></tr></table>

Robot post-training. Separate policies are post-trained for RoboTwin 2.0, RoboDojo, and real-robot experiments using the corresponding robot demonstrations. Table 6 summarizes the training data and optimization settings. RoboTwin–Clean2Clean and RoboTwin–Clean2Random use the same post-training data, comprising 50 clean demonstrations per task. Robot EEF poses follow the absolute camera-frame representation. Post-training uses AdamW with a weight decay of 0.01 and a cosine learning-rate schedule preceded by linear warm-up over the first 5% of optimization steps.

## 4.2 Simulation Experiments

Since our egocentric pretraining primarily covers bimanual tabletop manipulation, we evaluate HumanWAM-Vanilla on RoboTwin 2.0 and RoboDojo. Comparisons with existing vision–language–action models and world–action models assess robustness to environmental variation and capabilities such as precision, memory, and long-horizon execution.

RoboTwin 2.0. RoboTwin 2.0 [8] is a benchmark built on SAPIEN [60] comprising 50 bimanual tabletop manipulation tasks on the Aloha-AgileX embodiment, including pick-and-place, stacking, object handovers, and articulated-object manipulation. We jointly post-train a single policy on clean demonstrations from all tasks and evaluate it in two settings. Clean2Clean evaluates performance within the demonstration environment distribution, whereas Clean2Random randomizes background and tabletop textures, lighting, clutter, and table height.

At inference, the policy receives RGB observations from the head camera and both wrist cameras, together with the language instruction and proprioceptive state. It predicts 32-step action chunks and executes the first 24 steps before replanning from updated observations. We evaluate the same checkpoint in both settings, with 100 episodes per task, totaling 5,000 episodes per setting. We report the mean success rate across the 50 tasks in each setting and the average of these two means.

Table 7 shows that HumanWAM-Vanilla achieves the highest Clean2Clean success rate among the compared methods, reaching 92.08% after post-training on 50 clean robot demonstrations per task. The same policy achieves 68.88% on Clean2Random, yielding an average success rate of 80.48% that exceeds all VLA baselines and most WAM baselines in the table. The remaining gap to the leading WAMs lies in out-of-distribution performance, with GO-3 Nano Preview reaching 81.48% on Clean2Random. This gap may partly reflect architectural diferences, as HumanWAM-Vanilla uses a basic world–action architecture without additional specialized modules. These results support pretraining exclusively on human egocentric data as an efective foundation for bimanual robot manipulation, with accurate task execution and competitive generalization to visual conditions absent from the robot demonstrations.

RoboDojo. RoboDojo [9] is an Isaac Sim-based benchmark with 42 bimanual manipulation tasks covering generalization, precision, memory, long-horizon execution, and open-vocabulary instruction following. Many tasks require reasoning over action sequences and recalling earlier observations in addition to accurate motor control. We evaluate egocentric pretraining within the world–action architecture described in Section 4.1, without adding dedicated reasoning or memory modules. Accordingly, we compare against representative VLA and WAM baselines and do not extend the evaluation to recent agentic or dual-system approaches.

Table 7 Evaluation results on RoboTwin 2.0. Success rates (%) are reported after clean-only post-training. Average is the mean of Clean2Clean and Clean2Random. The best result in each column is highlighted in bold
<table><tr><td>Type</td><td>Method</td><td>Clean2Clean ↑</td><td>Clean2Random ↑</td><td>Average ↑</td></tr><tr><td rowspan="7">VLA</td><td>Qwen-RobotManip-Context [66]</td><td>84.7</td><td>69.4</td><td>77.1</td></tr><tr><td>GigaBrain-0.7 [51]</td><td>66.80</td><td>67.92</td><td>67.36</td></tr><tr><td>π0.5 [21]</td><td>70.70</td><td>46.00</td><td>58.35</td></tr><tr><td>Spatial Forcing [24]</td><td>77.20</td><td>26.74</td><td>51.97</td></tr><tr><td>X-VLA [72]</td><td>68.00</td><td>20.90</td><td>44.45</td></tr><tr><td>Abot-M0 [62]</td><td>57.40</td><td>30.40</td><td>43.90</td></tr><tr><td>GR00T-N1.7 [37]</td><td>43.6</td><td>20.7</td><td>32.2</td></tr><tr><td rowspan="14">WAM</td><td>GO-3 Nano Preview</td><td>88.54</td><td></td><td>85.01</td></tr><tr><td>PICA-WAM [76]</td><td>90.68</td><td>81.48</td><td></td></tr><tr><td>CloudRobo-WM</td><td>88.90</td><td>74.70 75.00</td><td>82.69 81.95</td></tr><tr><td>MWA-Phoenix [3]</td><td>86.54</td><td>72.42</td><td>79.48</td></tr><tr><td>PatchWAM [55]</td><td>91.56</td><td>66.72</td><td>79.14</td></tr><tr><td>ME-Dex-1.0 [69]</td><td>89.58</td><td>68.12</td><td>78.85</td></tr><tr><td>EWAM [54]</td><td>82.16</td><td>72.08</td><td>77.12</td></tr><tr><td>ViGAR [18]</td><td>82.00</td><td>67.02</td><td>74.51</td></tr><tr><td>UniWAM [49]</td><td>75.12</td><td>67.56</td><td>71.34</td></tr><tr><td>OpenWAM-α [56]</td><td>89.4</td><td>48.7</td><td>69.0</td></tr><tr><td>4D-WAM [61]</td><td>81.50</td><td>41.80</td><td>61.65</td></tr><tr><td>HumanWAM-Vanilla (Ours)</td><td></td><td></td><td></td></tr><tr><td></td><td>92.08</td><td>68.88</td><td>80.48</td></tr></table>

Baseline results are taken from the oficial RoboTwin 2.0 leaderboard [44] (Co-train track, without observed depth at inference), except for GR00T-N1.7, Qwen-RobotManip-Context, and OpenWAM-α, whose published results are sourced from InternW0-∆ [34]. HumanWAM-Vanilla results are obtained from our own evaluations.

We post-train on the oficial set of 3,500 robot demonstrations, comprising 100 demonstrations for each of 34 training tasks and 100 auxiliary domain-randomized demonstrations. The Open tasks are excluded from post-training. Generalization is evaluated separately under standard (Gen-Std) and randomized (Gen-Rand) conditions. We evaluate the same checkpoint across all six groups and report success rate, which measures complete task success, and progress score (0–100), which credits partial completion.

Table 8 shows that HumanWAM-Vanilla achieves a mean success rate of 14.17% and a mean progress score of 19.00. It exceeds the WAM baselines in our comparison on both aggregate metrics and approaches the overall performance of Galaxea G0.5. Its advantage over OpenWAM-α is most pronounced on Precision and Long-Horizon, where success reaches 20.50% and 33.00%, compared with 9.25% and 25.33%, respectively. These results show that pretraining exclusively on human egocentric data, followed by adaptation with 100 robot demonstrations per training task, can support competitive robot manipulation, particularly in tasks requiring precise control and multi-step execution

## 4.3 Real-Robot Experiments

Setup. To assess whether the generalization benefits observed in simulation extend to physical robots, we evaluate HumanWAM-Vanilla on paper cup stacking, towel folding, drawer storage, and desk tidying with a Cobot Magic bimanual robot, as shown in Figure 14. These tasks cover multi-step pick-and-place, deformable-object manipulation, articulated-object manipulation, and long-horizon execution. We compare HumanWAM-Vanilla with a No-Ego baseline that omits egocentric pretraining. Both models are post-trained on 50 demonstrations per task for 10,000 optimizer steps with a global batch size of 256.

We evaluate robustness to visual distribution shift by changing the tablecloth and lighting, individually and in combination, relative to the demonstration environment. For each task and model, we conduct 10 in-distribution trials under the visual conditions of the training demonstrations and 10 out-of-distribution trials, allocating four to each individual change and two to their combination. We report per-task success rates and their mean across the four tasks separately for each setting.

Table 8 RoboDojo comparison with VLA and WAM baselines. Each entry reports success rate (%) / progress score. Mean is recomputed for every method as an equally weighted average over the six groups.
<table><tr><td rowspan="2">Type Method</td><td rowspan="2"></td><td rowspan="2">Gen-Std</td><td colspan="6">Long-</td></tr><tr><td>Gen-Rand</td><td>Precision</td><td>Horizon</td><td>Memory</td><td>Open</td><td>Mean</td></tr><tr><td rowspan="7">VLA</td><td>StarVLA-α [64]</td><td>5.00 /7.54</td><td>0.00 /0.33</td><td>4.33/9.90</td><td>6.50/14.15</td><td>2.44 /3.34</td><td>0.58 /0.68</td><td>3.14/5.99</td></tr><tr><td>X-VLA [72]</td><td>12.00/17.90</td><td>1.00/3.04</td><td>12.00 /18.32</td><td>9.75/16.53</td><td>3.56 / 4.76</td><td>0.50/0.55</td><td>6.47/10.18</td></tr><tr><td>π0.5 [21]</td><td>15.00/20.93</td><td>1.00/5.82</td><td>5.50/12.40</td><td>14.67 /23.54</td><td>4.56/5.78</td><td>1.67/1.98</td><td>7.07/11.74</td></tr><tr><td>Spatial Forcing [24]</td><td>15.00/21.25</td><td>4.00/6.98</td><td>10.58/17.33</td><td>14.58 /23.26</td><td>4.11/5.43</td><td>1.58/1.78</td><td>8.31/12.67</td></tr><tr><td>Hy-Embodied-0.5-VLA [68]</td><td>17.00/21.98</td><td>0.00/1.57</td><td>8.00/13.81</td><td>14.92/25.74 12.11/13.37</td><td></td><td>0.58/0.65</td><td>8.77/12.85</td></tr><tr><td>Xiaomi-Robotics-1 [52]</td><td>28.00/35.65</td><td>6.00 /11.44</td><td>18.83/26.69</td><td>23.67/38.39</td><td>6.56 /7.81</td><td>3.58/3.94</td><td>14.44/20.65</td></tr><tr><td>Galaxea G0.5 [29]</td><td>20.00/26.74</td><td>6.00/11.16</td><td>20.42/28.25</td><td>32.25 /44.12</td><td>7.33/8.61</td><td>1.58/1.73</td><td>14.60 / 20.10</td></tr><tr><td rowspan="6">WAM</td><td>DM0.5 [12]</td><td>18.00/23.49</td><td>4.00 /8.06</td><td>16.75/24.82</td><td>19.50/33.70</td><td>47.44/47.74</td><td>2.08/2.43</td><td>17.96/23.37</td></tr><tr><td>Fast-WAM [67]</td><td>2.00 / 4.33</td><td>0.00/0.34</td><td>0.00/1.96</td><td>5.17/9.14</td><td>3.44/3.55</td><td>0.42/0.42</td><td>1.84/3.29</td></tr><tr><td>AHA-WAM [6]</td><td>6.00/10.32</td><td>0.00/1.26</td><td>2.42/5.86</td><td>2.67/8.61</td><td>2.78/2.97</td><td>0.83/0.88</td><td>2.45 /4.98</td></tr><tr><td>GigaWorld-Policy [63]</td><td>6.00/10.28</td><td>0.00/0.41</td><td>1.83/6.15</td><td>8.92/15.51</td><td>2.22/3.46</td><td>0.50/0.54</td><td>3.25/6.06</td></tr><tr><td>X-WAM [19]</td><td>5.00/11.24</td><td>1.00/3.54</td><td>1.83/6.72</td><td>9.08/17.47</td><td>4.67/6.32</td><td>0.25/0.57</td><td>3.64/7.64</td></tr><tr><td>OpenWAM-α [56]</td><td>25.56/33.16</td><td>4.11/8.26</td><td>9.25/18.45</td><td>25.33/34.93</td><td>9.11/10.41</td><td>1.08 /1.41</td><td>12.41/17.77</td></tr><tr><td></td><td>HumanWAM-Vanilla</td><td>22.33 /29.10</td><td>1.67 / 3.25</td><td>20.50 /28.83</td><td>33.00 / 44.00</td><td>6.00 / 7.00</td><td>1.50 /1.8014.17 /19.00</td><td></td></tr></table>

Table 9 Real-robot task success with and without egocentric pretraining. Success rates are reported in percent over 10 trials per task, model, and evaluation setting. Mean averages the four task success rates.
<table><tr><td></td><td colspan="2">In-distribution</td><td colspan="2">Out-of-distribution</td></tr><tr><td>Task</td><td>No-Ego</td><td>HumanWAM-Vanilla</td><td>No-Ego</td><td>HumanWAM-Vanilla</td></tr><tr><td>Paper cup stacking</td><td>70.0</td><td>90.0</td><td>0.0</td><td>80.0</td></tr><tr><td>Towel folding</td><td>50.0</td><td>80.0</td><td>10.0</td><td>60.0</td></tr><tr><td>Drawer storage</td><td>60.0</td><td>90.0</td><td>0.0</td><td>70.0</td></tr><tr><td>Desk tidying</td><td>50.0</td><td>90.0</td><td>0.0</td><td>60.0</td></tr><tr><td>Mean</td><td>57.5</td><td>87.5</td><td>2.5</td><td>67.5</td></tr></table>

Results. HumanWAM-Vanilla achieves higher success rates than No-Ego on all four tasks in both settings (Table 9). Mean in-distribution success increases from 57.5% to 87.5%. These gains span deformable-object manipulation and long-horizon execution, with success increasing from 50% to 80% on towel folding and from 50% to 90% on desk tidying. Success on drawer storage increases from 60% to 90%.

The gap widens under visual distribution shift. HumanWAM-Vanilla achieves 60–80% success across the four tasks, whereas No-Ego succeeds in only one of the 40 out-of-distribution trials. Mean out-of-distribution success increases from 2.5% to 67.5%, a gain of 65.0 pp compared with 30.0 pp in distribution. These results show that the generalization benefits of egocentric pretraining extend to physical robots, with improved task execution under visual conditions absent from the robot demonstrations.

## 4.4 Ablation Studies

To improve robot policy performance after post-training, we refine the model and training configuration used in the egocentric pretraining exploration. HumanWAM-Vanilla adopts IDM attention and absolute camera-frame actions in place of the default isolated attention and relative actions described in Section 3.1.1. We examine these choices, together with action-expert initialization, through ablations on the four-source, 1,000-hour corpus described in Section 3.5. All models are pretrained for 20,000 steps and post-trained for 12,000 steps on 50 clean demonstrations per RoboTwin task.

![](images/e4275aaecb0a8f839aa28c95d057f89f430bf721c95180f3dac00577b2b8bac8.jpg)  
Figure 14 Real-robot tasks and visual evaluation conditions.

Attention strategies. World–action models difer in how video and action tokens exchange information. We compare four attention strategies, each applied consistently throughout egocentric pretraining and robot post-training. Isolated attention restricts visual conditioning for action prediction to the current frame. Mutual attention allows video and action tokens to attend to each other during joint denoising. IDM attention conditions action denoising on ground-truth future video latents during training and generated future latents at inference. Optional-IDM, following the variant released by Fast-WAM [15], trains a single model to predict actions with or without future video conditioning. Our implementation uses ground-truth future-video conditioning with probability 0.5 and current-frame-only conditioning otherwise.

Table 10 shows that mutual and IDM attention improve clean and scene-ood success over isolated attention with relative actions held fixed. Scene-ood success rises from 46.36% to 57.96% and 58.08%, respectively. IDM attention also improves random success from 22.40% to 27.56%. For Optional-IDM, enabling future video generation at inference raises clean success from 73.44% to 81.96% and scene-ood success from 48.44% to 57.04% using the same checkpoint. These results indicate that generating future video before predicting actions improves both in-domain fitting and generalization under distribution shift.

Action parameterization. Embodied pretraining methods adopt diferent action representations. Relative EEF motions are commonly used to reduce dependence on global coordinate frames and facilitate cross-embodiment transfer [10, 73]. We compare relative and absolute camera-frame actions, retaining each representation throughout egocentric pretraining and robot post-training. Relative actions express target EEF poses with respect to the EEF pose at the current observation, whereas absolute actions express them directly in the camera frame at each target time.

Table 10 Ablation of attention strategies and action representations.
<table><tr><td>Attention</td><td>Action representation</td><td>Inference</td><td>Clean</td><td>Scene-OOD</td><td>Random</td></tr><tr><td>IA</td><td>Relative</td><td>Action-only</td><td>76.28</td><td>46.36</td><td>22.40</td></tr><tr><td>Mutual</td><td>Relative</td><td>Joint denoising</td><td>81.28</td><td>57.96</td><td>32.92</td></tr><tr><td>IDM</td><td>Relative</td><td>Video → action</td><td>83.56</td><td>58.08</td><td>27.56</td></tr><tr><td>Mutual</td><td>Absolute camera-frame</td><td>Joint denoising</td><td>84.88</td><td>61.12</td><td>30.68</td></tr><tr><td>IDM</td><td>Absolute camera-frame</td><td>Video → action</td><td>88.48</td><td>62.00</td><td>28.56</td></tr><tr><td>Optional-IDM</td><td>Relative</td><td>First-frame only</td><td>73.44</td><td>48.44</td><td>23.40</td></tr><tr><td>Optional-IDM</td><td>Relative</td><td>Video → action</td><td>81.96</td><td>57.04</td><td>28.04</td></tr></table>

Table 10 shows that absolute camera-frame actions improve clean and scene-ood performance over relative actions under both mutual and IDM attention. Under IDM attention, clean success increases from 83.56% to 88.48% and scene-ood success from 58.08% to 62.00%, with a smaller improvement on random. With mutual attention, clean success increases from 81.28% to 84.88% and scene-ood success from 57.96% to 61.12%.

Action-expert initialization. We further examine whether to initialize the action expert randomly or from linearly interpolated pretrained Wan2.2 weights before egocentric pretraining. Both variants start from the same pretrained video backbone and are pretrained for 20,000 steps on the same four-source, 1,000-hour corpus. They use IDM attention and absolute camera-frame actions throughout pretraining and post-training, with identical photometric augmentation during the 12,000-step robot post-training stage. We observed that initializing the action expert from video-model weights does not improve downstream success in this comparison (Table 11). The two variants achieve comparable clean and scene-ood success, while random initialization increases random success from 51.52% to 54.12%.

Table 11 Ablation of action-expert initialization.
<table><tr><td>Action-expert initialization</td><td>Clean</td><td>Scene-OOD</td><td>Random</td></tr><tr><td>Interpolated Wan2.2 weights</td><td>89.56</td><td>83.20</td><td>51.52</td></tr><tr><td>Random initialization</td><td>90.28</td><td>83.92</td><td>54.12</td></tr></table>

## 5 Pushing the Model Capability Boundary with Hybrid Data Scaling

Scaling pure egocentric pretraining to 50k hours instills strong manipulation priors, substantially improving out-of-distribution generalization after task-specific post-training. However, purely human pretraining leaves an inherent embodiment gap: without exposure to physical robot morphology and control dynamics, the model cannot be deployed directly and remains confined to a finetuning paradigm that requires demonstrations for every target task. Conversely, while large-scale robot datasets [58] provide direct, executable control supervision, scaling teleoperated trajectories alone quickly plateaus due to collection cost and limited scene diversity. To overcome these single-modality bottlenecks, we investigate joint co-training on human and robot data within an extended unified action space. Through systematic scaling experiments across progressive human and robot data budgets, we characterize how hybrid data scaling expands the boundary of mode generalization and execution capability.

A central question is whether scaling hybrid training experience reduces reliance on task-specific adaptation. We therefore evaluate our co-trained models across two complementary protocols. In the zero-shot protocol, the pretrained checkpoint is deployed directly in simulation and on physical robots without target-task demonstrations or parameter updates, testing whether larger hybrid budgets expand the repertoire of tasks executable out of the box. In the post-training protocol, the same checkpoints are finetuned on target-task demonstrations, assessing whether hybrid pretraining continues to raise the performance ceiling once in-domain data are available, particularly under out-of-distribution shifts.

Deploying a generalist policy without task-specific finetuning also requires an unambiguous interface to specify unfamiliar tasks. Because natural language instructions often leave object configurations, spatial relations, and terminal states underspecified, we augment our world-action model with explicit goal-image conditioning [18]. The resulting model, HumanWAM-Goal, jointly predicts future visual states and executable actions from current observations and a visual goal. Beyond disambiguating task intent for zero-shot execution, this interface enables embodied in-context learning [28], steering a frozen policy toward distinct manipulation outcomes purely by varying the visual goal at inference time.

To realize this framework, we first introduce our robot pretraining corpus (Section 5.1), extend the unified action representation (Section 5.2), and detail the HumanWAM-Goal architecture (Section 5.3) alongside its co-training recipe (Section 5.4). We then formalize our evaluation protocols (Section 5.5), examine zero-shot and post-trained scaling trends (Section 5.6), and finally demonstrate emergent zero-shot execution, visual in-context learning, and long-horizon manipulation on physical robots in Section 6.

## 5.1 Robot Pretraining Data

To complement egocentric human experience with embodiment-specific control supervision, we assemble approximately 14.6k hours of robot pretraining data spanning multiple embodiments. The real-robot portion contains over 1.1 million episodes across eight recorded robot domains, providing diverse control experience for cross-embodiment pretraining. Figure 15 summarizes the overall training scale and the action and object vocabulary of the real-robot recordings.

![](images/c64c0397c5b392678c80b2de217949ece09d9218bca02e47f742443d0354f4ab.jpg)  
Figure 15 Robot pretraining data overview. The top row reports the overall robot pretraining budget. Word clouds summarize selected (a) action and (b) object terms, while the adjacent bars show label-level percentages for the eight most frequent terms in each group. Percentages use all 751 normalized task-collection labels as the denominator.

Task diversity. The corpus spans activities such as organizing books and tools, folding clothing, preparing food, operating appliances, and inserting cables. To characterize this diversity, we extract action and object terms from the task-collection labels. Removing administrative identifiers and standardizing surface forms reduces 864 original labels to 751 normalized labels. Among the extracted action terms, place, fold, and insert occur in 130, 31, and 20 normalized labels, respectively. The object vocabulary includes tableware, garments, containers, tools, and appliances; for example, cup, plate, and clothing occur in 40, 37, and 24 labels. These statistics characterize the range of activities and interaction objects described by the task labels, complementing the aggregate data scale.

## 5.2 Hybrid Action Representation

To accommodate multi-embodiment robot control and mobile manipulation alongside human videos, we extend the 16-D bimanual action space of Section 2.3 into a 49-D hybrid representation:

$$
\begin{array} { r } { \mathbf { a } _ { t } ^ { \mathrm { e x t } } = \bigl [ \underbrace { \mathbf { j } _ { t } } _ { \mathrm { j o i n t : ~ 2 3 } } , \underbrace { \mathbf { b } _ { t } } _ { \mathrm { c h a s s i s : ~ 3 } } , \underbrace { \mathbf { a } _ { t } , \mathbf { p } _ { t } ^ { H } , \mathbf { q } _ { t } ^ { H } } _ { \mathrm { E E F ~ a n d ~ h e a d : ~ 2 3 } } \bigr ] \in \mathbb { R } ^ { 4 9 } . } \end{array}\tag{5}
$$

Here, $\mathbf { j } _ { t } \in \mathbb { R } ^ { 2 3 }$ encodes robot joint states in a canonical layout spanning the arms, grippers, torso, and head, with unused degrees of freedom zero-padded and masked; $\mathbf { b } _ { t } = ( x _ { t } , y _ { t } , \psi _ { t } ) \in \mathbb { R } ^ { 3 }$ represents planar base translation and yaw; and $\mathbf { a } _ { t } \in \mathbb { R } ^ { 1 6 }$ is the unified bimanual EEF action. The 7-D head pose $( \mathbf { p } _ { t } ^ { H } , \mathbf { \bar { q } } _ { t } ^ { H } )$ records the head-camera extrinsics used to anchor bimanual coordinates; within the network interface, these slots serve strictly as geometric metadata and are masked from both state conditioning and action prediction losses.

Metric gripper aperture. Whereas Section 2.3 normalizes gripper openness per dataset, zero-shot deployment requires a consistent physical scale across robots and humans. We therefore redefine the gripper dimensions in $\mathbf { a } _ { t }$ as the absolute physical aperture in meters (0 denoting fully closed), normalized by a universal reference width of 0.12 m. For human videos, we use the metric 3D distance between the estimated thumb and index fingertips directly as the physical aperture, aligning human hand opening and robot gripper width on a unified metric scale.

Reference frames. While absolute camera-frame EEF poses excel when post-training adapts normalization statistics to a single target embodiment (Section 4.4), heterogeneous robot platforms exhibit substantial variations in camera mounting geometry, arm reach, and workspace ofsets that impede direct zero-shot transfer. To eliminate these embodiment-specific spatial biases during multi-domain co-training and zero-shot deployment, we parameterize EEF motions as chunk-relative displacements anchored to the current observation. Specifically, given an observation at time t, EEF translation targets at each chunk step τ are expressed as displacements from the current EEF position projected into the anchor head-camera frame at time $t ,$ $\Delta \mathbf { p } ( \tau \mid t ) = \mathbf { R } ^ { H } ( t ) ^ { \top } ( \mathbf { p } ( \tau ) - \mathbf { p } ( t ) )$ , ensuring a stationary, embodiment-agnostic reference frame across the entire chunk even under camera motion. Similarly, EEF orientation targets are expressed as relative quaternions with respect to the wrist orientation at time $t ,$ and chassis targets b are expressed relative to the current base pose.

Hybrid supervision and joint dropout. Human and robot trajectories share common input and output projections for the 16-D EEF and 3-D chassis subspaces, whereas $\mathbf { j } _ { t }$ uses embodiment-specific projections. Per-source validity masks exclude unavailable dimensions (such as joints and chassis in human videos) from the action loss. During robot training, we additionally apply stochastic dropout to the joint branch $\mathbf { j } _ { t } ,$ , preventing the model from over-relying on embodiment-specific joint proprioception and ensuring that the shared 16-D EEF pathway remains self-suficient. Because $\mathbf { a } _ { t }$ parameterizes camera-frame metric displacements and physical gripper widths in an embodiment-agnostic coordinate frame, its global pretraining normalization statistics transfer directly to unseen robots at test time without target-domain recalibration.

## 5.3 HumanWAM-Goal Architecture

To serve as a general-purpose action executor for multimodal agents [28, 48], a foundation policy must translate explicit target outcomes into continuous physical control without task-specific parameter updates. We therefore introduce HumanWAM-Goal (Figure 16), which augments world-action modeling with explicit visual-goal conditioning [18]. Given a language instruction $\ell ,$ current observation $\mathbf { o } _ { t } .$ , proprioceptive state $\mathbf { s } _ { t } ,$ optional observation history $\mathcal { H } _ { t } .$ , and a goal image g specifying a desired intermediate or terminal scene configuration, the model estimates

$$
\begin{array} { r } { p _ { \theta } \big ( \mathbf { O } _ { t } ^ { + } , \mathbf { A } _ { t } \mid \mathbf { c } _ { t } \big ) , \qquad \mathbf { c } _ { t } = ( \boldsymbol { \ell } , \mathbf { o } _ { t } , \mathbf { s } _ { t } , \mathcal { H } _ { t } , \mathbf { g } ) , } \end{array}\tag{6}
$$

where ${ \mathbf O } _ { t } ^ { + }$ denotes 12 future video frames and $\mathbf { A } _ { t } \in \mathbb { R } ^ { 4 8 \times 4 9 }$ is a 48-step action chunk in the hybrid action space of Section 5.2. We build HumanWAM-Goal upon the pretrained Cosmos3-Nano-16B backbone [38], which pairs a frozen Qwen3-VL-based Reasoner for language encoding with a shared video-action Generator.

![](images/1841d069a79bc42241804e0d4a5179d51a97c31f63359b8d36b497f80cc9bae1.jpg)  
Figure 16 HumanWAM-Goal architecture. (a) Conditioning. An explicit goal image augments language instructions, current and optional historical observations, and the 49-D robot state. (b) Joint rectified-flow sampling. With conditioning tokens held fixed, the shared video-action Generator iteratively predicts video and action velocities via UniPC updates.

Visual goal and observation conditioning. To enable unambiguous task specification without modifying the backbone topology, all visual inputs are encoded into spatial latents via a frozen Wan VAE and mapped into the Generator through a shared visual projection $P _ { v } ^ { i }$ . While the current observation $\mathbf { o } _ { t }$ serves as the clean anchor frame of the rollout, the goal image g is encoded separately, augmented with a learnable goal embedding, and supplied as clean conditioning tokens. This design preserves the fine-grained spatial structure of the desired outcome while distinguishing goal targets from current observations. When historical frames $\mathcal { H } _ { t }$ are included, they enter as additional clean visual tokens with temporal rotary position embeddings (RoPE) reflecting their elapsed time relative to $\mathbf { o } _ { t }$

Hybrid state-action interface and inference. In language-only world-action modeling (Section 4.4), where terminal states are visually underspecified, two-stage inverse-dynamics (IDM) inference—generating complete future video latents prior to action denoising—is essential to ground control in predicted visual dynamics, albeit at double the sequential sampling cost. In HumanWAM-Goal, because the clean goal tokens g directly anchor the target spatial configuration, we replace two-stage cascading with single-stage joint rectified-flow sampling within the shared-parameter Generator (Figure 16). Specifically, video and action tokens share Transformer weights and interact bidirectionally at every denoising step through joint self-attention via modality-specific projections $P _ { v } ^ { i / o }$ and $P _ { a } ^ { i / o }$ . We interface the 49-D hybrid representation (Section 5.2) by prepending the normalized current state $\mathbf { s } _ { t }$ as a single clean conditioning row ahead of the 48 noisy future action rows, restricting the rectified-flow loss strictly to future actions. At inference, holding all conditioning tokens $\mathbf { c } _ { t }$ fixed, the Generator synchronously denoises future video latents and the 48-step action chunk via iterative UniPC updates, preserving world-to-action coupling while halving sampling latency and optionally bypassing the Wan VAE decoder for real-time control.

## 5.4 Training Recipe

Training data. We investigate human–robot data scaling through two complementary series. In the robot-only series, we progressively expand the robot training corpus from 2,005 to 14,618 hours (2,005 / 4,261 / 8,475 / 14,618 hours), drawn from eight real-world robot domains (Section 5.1). In the hybrid series, we hold the robot corpus fixed at 14,618 hours and incorporate either 15,901 or 45,938 hours of egocentric human data drawn from the curated corpus in Section 2.1, yielding total training budgets of approximately 30.5k and 60.6k hours, respectively. This design isolates the benefits of scaling robot demonstrations from the additional contribution of human experience at a fixed robot-data budget. Both modalities are represented in the unified action space described in Section 5.2.

Goal images. During co-training, goal images are sampled online from future observations within the same trajectory using an identical strategy for human and robot data. For each training example, we sample a clean head-camera frame from four goal-conditioning options: a future observation 2–8 s ahead (40%), the episode’s terminal observation (20%), the final observation of an upcoming annotated subtask (20%), or an empty goal (20%). Goal timestamps are constrained to lie at or beyond the end of the supervised action chunk. Each goal image is encoded with a learnable goal embedding and serves exclusively as a conditioning signal without an associated reconstruction loss. When the goal is omitted, its tokens are removed entirely, enabling a single model to support both goal-conditioned and language-only execution.

Optimization. All configurations start from the same pretrained Cosmos3-Nano-16B initialization. We optimize the Generator, timestep embeddings, visual and action projections, action modality embedding, and goal embedding, while keeping the Reasoner and VAE frozen. Training uses FusedAdam with a base learning rate of $4 \times 1 0 ^ { - 5 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ , weight decay of 0.05, gradient-norm clipping at 1.0, and a global batch size of 1,024. The learning rate increases linearly over the first 1,000 steps and subsequently decays linearly over a 500k-step schedule. We apply a 5× learning-rate multiplier to the action projections, action modality embedding, and goal embedding. All evaluations use exponential moving average (EMA) weights.

RoboTwin post-training. For simulation evaluation, we initialize from the co-trained checkpoint and adapt the model using 50 clean demonstrations from the ALOHA-AgileX robot per RoboTwin task, totaling 2,500 trajectories. Table 12 summarizes the adaptation protocols used in our post-training scaling and leaderboard evaluations (Section 5.6.2). Action normalization statistics are computed jointly from the clean RoboTwin datasets and pretraining data.

For goal specification during post-training, we use a mixture of ground-truth terminal frames and synthesized goal images. Specifically, 20% of the training episodes use the ground-truth terminal observation as the goal, while the remaining 80% use goal images synthesized by GPT Image 2.5 [40]. Each goal image is generated from the initial scene observation and task instruction using a fixed task-specific prompt (Appendix A). To ensure fidelity to the intended task, every generated image undergoes an automated review using ChatGPT Astra High [39], which verifies the target object and destination, designated manipulation arm, object count and appearance, and background and viewpoint consistency. Images that fail any check are regenerated and reviewed until accepted; this procedure enforces visual and instructional consistency rather than physical feasibility. At evaluation time, each scene is assigned a single generated goal image shared across all evaluated checkpoints to eliminate goal variation as a source of performance diferences.

## 5.5 Evaluation Settings

Benchmark. To ensure reproducibility and fair comparison with existing robot policies, we adopt RoboTwin 2.0 [8], a standardized simulation benchmark for bimanual manipulation. Built on the SAPIEN physics engine, RoboTwin 2.0 provides 50 manipulation tasks with predefined success criteria and controlled environmental variations. We evaluate all models using the ALOHA-AgileX embodiment, which is never exposed during pretraining. During each episode, the policy receives visual observations, robot states, and task instructions, executing actions in closed loop. Task success is determined by the benchmark’s automated, task-specific completion checks.

RoboTwin 2.0 defines two evaluation conditions: Clean, which uses the default scene configuration, and Random, which introduces variations in backgrounds, lighting, object clutter, and table height. Together, they assess task execution under standard conditions and robustness to environmental shifts across the same 50 tasks and identical success criteria.

Evaluation protocols. We consider two complementary protocols to disentangle the capabilities acquired during co-training from those gained through target-task adaptation (Table 12). In the zero-shot protocol, co-trained checkpoints are deployed directly on RoboTwin 2.0 without any target-task demonstrations or parameter updates, testing whether scaling hybrid experience enables unseen manipulation tasks to be executed out of the box. In the post-training protocol, each co-trained checkpoint is adapted using 50 Clean demonstrations per task to evaluate how pretraining scale afects downstream in-domain fit and out-ofdistribution generalization. For computational eficiency, zero-shot evaluations use 50 episodes per task under each condition (Clean and Random), whereas post-training scaling and leaderboard evaluations follow the oficial protocol of 100 episodes per task and condition (10,000 episodes in total).

Table 12 RoboTwin 2.0 evaluation protocols.
<table><tr><td>Setting</td><td>Target-task adaptation</td><td>Episodes</td><td>Results</td></tr><tr><td>Zero-shot</td><td>None</td><td>50</td><td>Table 13</td></tr><tr><td>Post-training</td><td>50 clean demos/task</td><td>100</td><td>Figure 19</td></tr><tr><td>Leaderboard</td><td>Best configuration + augmentation</td><td>100</td><td>Table 14</td></tr></table>

Metrics. We report task success rates averaged equally across all 50 tasks for Clean and Random, along with their arithmetic mean. To characterize the breadth of zero-shot execution beyond aggregate success rates, we also report task coverage, defined as the number of tasks achieving at least 2% mean success across Clean and Random (corresponding to at least two successful rollouts out of 100 episodes). Alongside individual checkpoint coverage, we track cumulative coverage across the scaling series, defined as the union of feasible tasks across all checkpoints up to a given data budget.

## 5.6 Comparison with State-of-the-Art Embodied Foundation Models

We evaluate HumanWAM-Goal on RoboTwin 2.0 across both protocols defined in Section 5.5. We first examine zero-shot execution against representative embodied foundation models and across hybrid data scales (Section 5.6.1), and then assess post-trained performance and oficial leaderboard comparisons (Section 5.6.2).

## 5.6.1 Zero-Shot Performance

Zero-shot comparison with existing foundation models. Table 13 compares HumanWAM-Goal with representative open-source VLA and WAM baselines under the zero-shot protocol. Most existing models are designed exclusively for a pretrain-then-finetune workflow and cannot be deployed out of the box due to unreleased action normalization statistics (ERR0), incompatible embodiment-specific control spaces (ERR1), or the absence of public pretrained weights without fine-tuning on RoboTwin (ERR2). Among the models that can be executed directly, several exhibit static or non-task-directed motions (0.00% success), while π<sub>0</sub>, π<sub>0.5</sub>, and Pelican-VLA 0.5 achieve average success rates of 0.12%, 1.06%, and 4.70%, respectively. By contrast, because HumanWAM-Goal aligns camera-frame EEF displacements and metric gripper apertures on a unified physical scale across pretraining sources while masking embodiment-specific joints (Section 5.2), its global EEF normalization statistics represent embodiment-agnostic physical motions and apply directly to unseen robots via standard inverse kinematics without target-domain recalibration. The robot-only HumanWAM-Goal checkpoint reaches 11.48% average success, and human–robot hybrid co-training further elevates performance to 13.94%. These results indicate that combining a unified physical action interface with explicit visual-goal conditioning enables transferable zero-shot execution directly from pretraining.

Scaling robot and hybrid data. Figure 17 tracks zero-shot success rates and task coverage across six independently trained configurations. As real-robot training data increase from 2,005 to 14,618 hours (7.3×), average success rises steadily from 2.80% to 11.48%, while single-checkpoint task coverage expands from 8 to 18 of the 50 tasks. Both metrics improve at every intermediate budget, indicating that scaling robot experience broadens the set of solvable tasks rather than merely improving reliability on a fixed subset. Holding robot data fixed at 14,618 hours, incorporating 15,901 hours of egocentric human video further increases average success to 13.16% (19 covered tasks), and scaling to 45,938 hours of human video reaches 13.94% (22 covered tasks, and 29 tasks in cumulative coverage), outperforming the robot-only baseline by 2.46 pp. Notably, expanding the human corpus from 15,901 to 45,938 hours improves Random success by 1.44 pp compared with 0.12 pp on Clean, suggesting that human interaction diversity primarily strengthens zero-shot robustness under visual and environmental variation.

Table 13 Zero-shot evaluation on RoboTwin 2.0 (ALOHA-AgileX). Pretrained checkpoints are evaluated directly without RoboTwin-specific adaptation using 50 episodes per task under both Clean and Random conditions. Success rates are reported in percent. Status codes ERR0–ERR2 denote deployment barriers that prevent zero-shot evaluation.
<table><tr><td>Model</td><td>Clean</td><td>Random</td><td>Average</td></tr><tr><td>LingBot-VLA 2.0</td><td>ERRO</td><td>ERRO</td><td>ERRO</td></tr><tr><td>LingBot-VA</td><td>ERRO</td><td>ERRO</td><td>ERRO</td></tr><tr><td>OpenWAM [56]</td><td>ERRO</td><td>ERRO</td><td>ERRO</td></tr><tr><td>DM0.5</td><td>ERR1</td><td>ERR1</td><td>ERR1</td></tr><tr><td>GR00T N1.7</td><td>ERR1</td><td>ERR1</td><td>ERR1</td></tr><tr><td>DreamZero-AgiBot [65]</td><td>ERR1</td><td>ERR1</td><td>ERR1</td></tr><tr><td>LingBot-VA 2.0</td><td>ERR2</td><td>ERR2</td><td>ERR2</td></tr><tr><td>Fast-WAM [67]</td><td>ERR2</td><td>ERR2</td><td>ERR2</td></tr><tr><td>MV-WAM</td><td>ERR2</td><td>ERR2</td><td>ERR2</td></tr><tr><td>InternVLA-A1.5 [32]</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>InternW0-∆ [34]</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Zero-WAM [74]</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Hy-VLA-UMI [68]</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>π0 [5]</td><td>0.16</td><td>0.08</td><td>0.12</td></tr><tr><td>π0.5 [21]</td><td>0.88</td><td>1.24</td><td>1.06</td></tr><tr><td>Pelican-VLA 0.5 [13]</td><td>5.36</td><td>4.04</td><td>4.70</td></tr><tr><td>HumanWAM-Goal (Robot Only)</td><td>11.40</td><td>11.56</td><td>11.48</td></tr><tr><td>HumanWAM-Goal (Human-Robot Hybrid)</td><td>13.68</td><td>14.20</td><td>13.94</td></tr></table>

Notes. Average is the arithmetic mean of Clean and Random. A value of 0.00 indicates that rollouts were executed but yielded no successful episodes.  
ERR0: Evaluation could not be completed because the released checkpoints lacked the required action normalization statistics.  
ERR1: Evaluation was not conducted because a compatible control mapping to the target embodiment was unavailable.  
ERR2: No publicly available pretrained checkpoint suitable for this evaluation protocol was available.

![](images/a227a1c8b84e7f86b548218408f0f530959bd052023fbd653efe34d8e13fe14e.jpg)  
(a) Success rate.

![](images/6747fd9e5d22002d09ee42d5cf18702072b4ef65ebab746266879d5137d03489.jpg)  
(b) Task coverage.  
Figure 17 Zero-shot performance under hybrid data scaling. Task success rates and task coverage $( \geq 2 \%$ mean success) across six HumanWAM-Goal pretraining configurations on RoboTwin 2.0 (50 Clean and 50 Random episodes per task).

![](images/b0bac1375154b1487a9ec6c629ac83780dbf4f21f21863eec99aeb37e99f0dca.jpg)  
Figure 18 Task-level zero-shot scaling with human-robot hybrid training. Mean zero-shot success rates (averaged over Clean and Random) across six representative RoboTwin 2.0 tasks with robot data fixed at 14,618 hours and increasing egocentric human data.

Task-level scaling analysis. Figure 18 illustrates how adding human data afects individual manipulation skills across six representative tasks spanning grasping, contact, and spatial placement. We observe two distinct patterns of improvement. First, for tasks where robot-only pretraining struggles or rarely succeeds—such as Grab roller (1% → 19%), Move playing card away (23% → 30%), Turn switch (30% → 52%), and Place A left of B (12% → 24%)—hybrid co-training unlocks or substantially strengthens task feasibility. Second, for skills where the robot-only model already establishes a solid baseline, such as Shake bottle horizontally (58% → 65%) and Click bell (66% → 86%), human experience further refines execution reliability. While the marginal gain between the two human-data scales varies by task, hybrid co-training consistently improves upon robot-only pretraining across both emerging and established skills.

## 5.6.2 Post-Trained Performance

We next evaluate whether the advantages of hybrid co-training persist when target-task demonstrations are available for adaptation.

Post-training performance across pretraining scales. Figure 19 reports downstream success rates after adapting each pretrained checkpoint on 50 Clean demonstrations per task. Scaling robot-only pretraining from 2,005 to 14,618 hours increases Clean success from 87.24% to 89.84% and Random success from 59.88% to 70.80%. However, Random performance plateaus beyond 8,475 robot hours (70.96% at 8,475 hours vs. 70.80% at 14,618 hours), reflecting the limited scene diversity of teleoperated robot trajectories. By contrast, augmenting the 14,618-hour robot corpus with 15,901 and 45,938 hours of egocentric human data breaks this plateau, raising Random success to 71.68% and 74.36% (+3.56 pp over robot-only), respectively, while Clean success rate remains saturated near 89%. Consistent with observations in Qwen-RobotManip [66], in-distribution evaluation (Clean) is largely saturated by target-task demonstrations, whereas out-of-distribution evaluation (Random) reveals the transferable visuomotor priors gained from large-scale human–robot co-training.

Leaderboard comparison with refined post-training. To evaluate the full potential of our co-trained initialization, we combine our best hybrid checkpoint with photometric data augmentation and refinements to the posttraining recipe. Table 14 shows several representative entries on the RoboTwin 2.0 leaderboard [8]. Under the oficial 10,000-episode evaluation protocol, HumanWAM-Goal achieves 92.70% on Clean and 84.60% on

![](images/7ac9deada6d8cdd2ac0df40028017d47305086c34d9f5201479dd3b7f21d9e4c.jpg)

![](images/f439ee099e41d95825d6c2d04e87fd99b6b3c7c3f8d796b4dc731b47666873be.jpg)  
Figure 19 Post-trained performance across pretraining scales on RoboTwin 2.0.

Random (88.65% Average), outperforming the previous leading entry (GO-3 Nano Preview) by 4.16 pp on Clean, 3.12 pp on Random, and 3.64 pp on average. Because this configuration incorporates post-training augmentations beyond the controlled scaling recipe, we report it separately from the scaling curves in Figure 19.

## 6 Emerging Capabilities in Real-World Robot Behavior

While recent multimodal agents exhibit strong high-level reasoning and task planning [28, 48], translating semantic intent into physical work requires a general-purpose action executor that can reliably control real robots. Building on the simulation scaling trends in Section 5—which show that joint human–robot co-training with visual-goal conditioning instills transferable visuomotor priors—we transition to physical hardware to evaluate whether HumanWAM-Goal can bridge this gap. Central to our deployment is explicit goal-image conditioning, which decouples high-level task specification from low-level sensorimotor control. Rather than relying solely on language instructions that often leave spatial arrangements and terminal states underspecified, an upstream planner or user supplies a visual goal as inference-time context. This unified interface allows the model to serve both as an “out-of-the-box” executor for unseen tasks and as a scalable foundation for downstream post-training.

Following our experimental setup (Section 6.1), we evaluate the real-world behaviors of HumanWAM-Goal across three core capabilities. First, we investigate zero-shot real-world manipulation (Section 6.2) by deploying co-trained checkpoints directly across a 20-task benchmark, quantifying how hybrid human–robot scaling expands the set of skills executable without target-task demonstrations or parameter updates. Second, we examine embodied in-context learning (Section 6.3), testing whether a policy with strictly frozen weights—both before and after post-training—can dynamically alter its manipulation trajectory in response to varying robot- or human-depicted visual goals. Finally, we demonstrate long-horizon execution (Section 6.4) on post-trained policies, showcasing temporal continuity and subgoal chaining across extended, multi-stage tabletop and mobile-manipulation workflows.

## 6.1 Real-Robot Experimental Setup

Hardware platform and deployment protocol. We conduct physical-robot evaluations on Astribot S1, a wheeled mobile manipulator with an articulated torso. Across all experiments, policies receive RGB observations from a head-mounted camera and two wrist cameras alongside proprioceptive robot states, predicting a 48-step (1.6 s) action chunk in the unified 49-D hybrid action space (Section 5.2). Desired task outcomes are specified at inference time via the goal-image interface (Section 5.3), using target scene configurations captured from held-out robot demonstrations or egocentric human demonstrations (Section 6.3). To ensure low-latency closed-loop control on physical hardware, we deploy our models using PhyAI [41], which leverages tensor parallelism and highly optimized kernels to achieve a 2.2× inference speedup over a PyTorch baseline with CUDA Graphs.

Table 14 Evaluation results on RoboTwin 2.0. Success rates (%) are reported after clean-only post-training. Average is the mean of Clean2Clean and Clean2Random. The best result in each column is highlighted in bold
<table><tr><td>Type</td><td>Method</td><td>Clean2Clean ↑</td><td>Clean2Random ↑</td><td>Average ↑</td></tr><tr><td rowspan="7">VLA</td><td>Qwen-RobotManip-Context [66]</td><td>84.7</td><td>69.4</td><td>77.1</td></tr><tr><td>GigaBrain-0.7 [51]</td><td>66.80</td><td>67.92</td><td>67.36</td></tr><tr><td>π0.5 [21]</td><td>70.70</td><td>46.00</td><td>58.35</td></tr><tr><td>Spatial Forcing [24]</td><td>77.20</td><td>26.74</td><td>51.97</td></tr><tr><td>X-VLA [72]</td><td>68.00</td><td>20.90</td><td>44.45</td></tr><tr><td>Abot-M0 [62]</td><td>57.40</td><td>30.40</td><td>43.90</td></tr><tr><td>GR00T-N1.7 [37]</td><td>43.6</td><td>20.7</td><td>32.2</td></tr><tr><td rowspan="14">WAM</td><td>GO-3 Nano Preview</td><td>88.54</td><td>81.48</td><td></td></tr><tr><td>PICA-WAM [76]</td><td>90.68</td><td>74.70</td><td>85.01</td></tr><tr><td>CloudRobo-WM</td><td>88.90</td><td>75.00</td><td>82.69 81.95</td></tr><tr><td>MWA-Phoenix [3]</td><td>86.54</td><td>72.42</td><td>79.48</td></tr><tr><td>PatchWAM [55]</td><td>91.56</td><td>66.72</td><td>79.14</td></tr><tr><td>ME-Dex-1.0 [69]</td><td>89.58</td><td>68.12</td><td>78.85</td></tr><tr><td>EWAM [54]</td><td>82.16</td><td>72.08</td><td>77.12</td></tr><tr><td>ViGAR [18]</td><td>82.00</td><td>67.02</td><td>74.51</td></tr><tr><td>UniWAM [49]</td><td>75.12</td><td>67.56</td><td>71.34</td></tr><tr><td>OpenWAM-α [56]</td><td>89.4</td><td>48.7</td><td>69.0</td></tr><tr><td>4D-WAM [61]</td><td>81.50</td><td>41.80</td><td>61.65</td></tr><tr><td>HumanWAM-Vanilla (Ours)</td><td>92.08</td><td>68.88</td><td></td></tr><tr><td>HumanWAM-Goal (Ours)</td><td>92.70</td><td>84.60</td><td>80.48 88.65</td></tr></table>

Baseline results are taken from the oficial RoboTwin 2.0 leaderboard [44] (Co-train track, without observed depth at inference), except for GR00T-N1.7, Qwen-RobotManip-Context, and OpenWAM-α, whose published results are sourced from InternW0-∆ [34]. HumanWAM-Vanilla and HumanWAM-Goal results are obtained from our own evaluations.

Zero-shot task suite and evaluation metrics. Our zero-shot benchmark comprises 20 manipulation tasks stratified into 10 easy and 10 hard tasks. The suite spans five core manipulation families: contact and state changes (4 tasks), grasping and transport (5), articulated opening and closing (3), spatial placement and alignment (6), and stacking and insertion (2). We conduct 10 trials per task, totaling 200 rollouts per evaluated checkpoint.

We report both binary task success rate and a fine-grained progress score that captures partial task completion. Specifically, each task k is decomposed into $M _ { k }$ ordered, non-overlapping stages with fixed completion criteria judged manually from visual observations and end-efector poses. The initial stage always assesses target identification and approach—decoupling perception and reaching errors from subsequent manipulation failures—while the final stage encodes the terminal success predicate. Each stage $s \in \{ 1 , \ldots , M _ { k } \}$ is assigned a predefined weight $w _ { k , s } \in \{ 1 , 2 , 3 , 4 \}$ such that $\textstyle \sum _ { s = 1 } ^ { M _ { k } } w _ { k , s } = 1 0$ . The progress score for trial r is defined as

$$
P _ { k , r } = \sum _ { s = 1 } ^ { M _ { k } } w _ { k , s } b _ { k , r , s } , \qquad P _ { k , r } \in [ 0 , 1 0 ] ,\tag{7}
$$

where $b _ { k , r , s } \in \{ 0 , 1 \}$ indicates whether stage s is successfully completed. Both binary success and progress scores are averaged over the 10 trials per task and then macro-averaged equally across all 20 tasks. Appendix B lists the tasks, stage scores, and success outcomes.

Post-training protocol. For intricate, multi-stage, or long-horizon tasks that exceed zero-shot reliability, we adapt the co-trained checkpoint via task-specific post-training, resetting the optimizer, learning-rate schedule, and data sampler. We retain the co-training normalization statistics and recalculate only channels with substantially shifted distributions (e.g., the articulated torso of Astribot S1). During post-training, goal images are sampled online from the final frame of one of the next two annotated subtasks, a future frame within 4 s, or an empty goal. To improve robustness and enforce reliance on visual goals and shared EEF representations, we apply 20% dropout to the joint-state input branch, 10% dropout to language instructions, and weight the action loss 10× relative to the video prediction loss. We select the EMA checkpoint that minimizes end-efector average displacement error (ADE) on held-out validation episodes.

## 6.2 Real-World Zero-Shot Performance

We evaluate the zero-shot transfer of HumanWAM-Goal on physical hardware by comparing the robot-only baseline (14, 618 h) against hybrid checkpoints co-trained with an additional 15, 901 or 45, 938 hours of egocentric human data (yielding total budgets of 30, 519 h and 60, 556 h, respectively). All checkpoints are deployed directly out of the box without target-task demonstrations or parameter updates.

![](images/54c5a5922c767dbeb163797f1fa8351f5dca98ffdfcd4c34d473bbce491402cb.jpg)  
Figure 20 Zero-shot real-robot performance stratified by task difficulty. Success rates across easy and hard tasks as egocentric human data are added to a fixed 14, 618 h robot corpus. The vertical dashed line at 60, 556 h highlights the additional gain on hard tasks from robot-balanced batch sampling (+8 pp).

Scaling trends across task dificulty. Figure 20 illustrates how scaling hybrid data afects real-world zero-shot execution across diferent task dificulties. As egocentric human data expand from 0 to 15.9k and 45.9k hours, success on easy tasks rises steadily from 32.0% to 34.0% and 42.0% (+10.0 pp overall), while success on hard tasks more than doubles from 8.0% to 18.0% and 19.0% (+11.0 pp). These consistent gains confirm that visuomotor priors learned from human manipulation transfer directly to physical robot execution.

Notably, while easy tasks benefit most when scaling from 30.5k to 60.6k total hours, gains on hard tasks begin to plateau under standard data-proportional sampling, where human clips constitute roughly 76% of the training mixture. To test whether this plateau stems from diluted embodiment-specific supervision, we evaluate a robot-balanced configuration at the 60.6k-hour scale that enforces a 1:1 ratio of robot and human samples in every training batch. Holding the data budget fixed, robot-balanced sampling boosts hard-task success from 19.0% to 27.0% (+8.0 pp). This result indicates that while massive human data provide broad interaction priors, preserving suficient batch exposure to robot trajectories is crucial for executing more demanding multi-step or precision-critical skills.

Task-level analysis and coverage expansion. Figure 21 breaks down zero-shot performance across all 20 benchmark tasks. Under the standard scaling series, overall mean success across the 20 tasks increases monotonically from 20.0% (robot-only) to 26.0% (+15.9k h human) and 30.5% (+45.9k h human). Partial task completion improves in tandem, with the macro-averaged progress score rising from 2.535 to 3.365 and 4.205 out of 10. Equally important, hybrid scaling expands real-world task coverage: the number of tasks achieving at least one successful zero-shot rollout increases from 10/20 with robot-only pretraining to 14/20

<table><tr><td>Task</td><td colspan="4">Success rate (%)</td><td colspan="4">Progress score (0–10)</td></tr><tr><td></td><td>14.6k h Robot only</td><td>30.5k h + Human</td><td>60.6k h + Human</td><td>60.6k h Balanced†</td><td>14.6k h Robot only</td><td>30.5k h + Human</td><td>60.6k h + Human</td><td>60.6k h Balanced†</td></tr><tr><td>Easy</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.9</td><td>9.3</td></tr><tr><td>Grasp and lift a block</td><td>80</td><td>70</td><td>30</td><td>90</td><td>8.6</td><td>7.6</td><td></td><td></td></tr><tr><td>Fully compress a sponge</td><td>60</td><td>30</td><td>70</td><td>20</td><td>8.0</td><td>3.0</td><td>7.9</td><td>4.3</td></tr><tr><td>Open a hardcover book</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td><td>3.6</td><td>2.4</td><td>2.2</td></tr><tr><td>Ring a bell</td><td>60</td><td>100</td><td>70</td><td>20</td><td>8.4</td><td>10.0</td><td>8.5</td><td>6.7</td></tr><tr><td>Click a ballpoint pen</td><td>50</td><td>0</td><td>50</td><td>50</td><td>5.0</td><td>3.0</td><td>5.9</td><td>6.5</td></tr><tr><td>Open a drawer</td><td>0</td><td>0</td><td>10</td><td>0</td><td>2.0</td><td>3.0</td><td>3.4</td><td>1.5</td></tr><tr><td>Push an object away</td><td>0</td><td>70</td><td>30</td><td>20</td><td>1.2</td><td>7.0</td><td>6.3</td><td>4.4</td></tr><tr><td>Grasp and shake a bottle</td><td>60</td><td>20</td><td>90</td><td>100</td><td>6.3</td><td>1.7</td><td>9.0</td><td>10.0</td></tr><tr><td>Arrange blocks Place a block on a tray</td><td>0</td><td>10</td><td>20</td><td>0</td><td>1.3</td><td>3.4</td><td>3.5</td><td>2.7</td></tr><tr><td></td><td>10</td><td>40</td><td>50</td><td>10</td><td>1.3</td><td>3.9</td><td>4.0</td><td>0.8</td></tr><tr><td>Hard</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stack blocks</td><td>0</td><td>20</td><td>30</td><td>10</td><td>0.9</td><td>2.6</td><td>4.8</td><td>1.6</td></tr><tr><td>Place a can beside a book</td><td>20</td><td>40</td><td>20</td><td>10</td><td>2.5</td><td>3.9</td><td>3.3</td><td>2.9</td></tr><tr><td>Place a cup on a coaster</td><td>10</td><td>10</td><td>20</td><td>10</td><td>1.0</td><td>1.0</td><td>2.6</td><td>1.1</td></tr><tr><td>Place an egg in a bowl</td><td>30</td><td>10</td><td>30</td><td>30</td><td>2.6</td><td>1.0</td><td>3.3</td><td>3.0</td></tr><tr><td>Place an object on stacked books</td><td>20</td><td>40</td><td>50</td><td>50</td><td>1.6</td><td>4.0</td><td>5.0</td><td>5.0</td></tr><tr><td>Put a glue stick in a pen holder</td><td>0</td><td>30</td><td>10</td><td>40</td><td>0.0</td><td>3.0</td><td>1.0</td><td>4.3</td></tr><tr><td>Stack two bowls</td><td>0</td><td>30</td><td>0</td><td>60</td><td>0.0</td><td>3.0</td><td>1.2</td><td>6.0</td></tr><tr><td>Lift one bottle with each hand</td><td>0</td><td>0</td><td>30</td><td>60</td><td>0.0</td><td>1.6</td><td>6.5</td><td>7.8</td></tr><tr><td>Open drawer and place object inside</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td><td>1.0</td><td>0.4</td><td>0.0</td></tr><tr><td>Plug a cable into a power strip</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>1.7</td></tr><tr><td>Average · 20 tasks</td><td>20.0</td><td>26.0</td><td>30.5</td><td>29.0</td><td>2.535</td><td>3.365</td><td>4.205</td><td>4.090</td></tr><tr><td></td><td></td><td></td><td></td><td>100</td><td>0</td><td></td><td></td><td>10</td></tr></table>

Column headers show total training hours; robot data are fixed at 14,618 h.  
10 trials per task. † Balanced: 50% robot samples per batch, with 45,938 h of human data.

Figure 21 Per-task zero-shot evaluation on physical robots. Task-level success rates (left) and mean progress scores (right) across all 20 benchmark tasks for the robot-only baseline and three human–robot hybrid configurations.

and 16/20 as human experience is incorporated. Echoing our simulation findings (Section 5.6.1), human data not only refine reliability on already feasible skills—such as Grasp and shake a bottle (60% → 90%) and Place a block on a tray (10% → 50%)—but also unlock entirely new capabilities where the robot-only baseline never succeeds (0%), including Stack blocks (30%), Arrange blocks (20%), and Lift one bottle with each hand (30%).

Examining the per-task breakdown for the 60.6k-hour robot-balanced model further clarifies its advantage on hard manipulation tasks. Compared with unweighted sampling at the same scale, balancing the human–robot ratio substantially improves tasks that require precise spatial alignment or bimanual coordination—such as Put a glue stick in a pen holder (10% → 40%), Stack two bowls (0% → 60%), and Lift one bottle with each hand (30% → 60%). By preventing the large human corpus from overwhelming embodiment-specific control dynamics, balanced co-training efectively combines human-scale visual-semantic diversity with high-precision robotic execution.

Qualitative analysis of zero-shot rollouts. Figure 22 visualizes representative zero-shot rollouts across eight real-world tasks. Beyond matching the terminal object arrangement depicted in the goal image, the policy exhibits purposeful intermediate behaviors that are not explicitly specified in the static visual goal—such as reorienting a pen holder prior to inserting a glue stick (row e) and coordinating both arms to swap or distribute objects without collision (rows b, f, h). These qualitative trajectories complement our quantitative metrics, demonstrating that HumanWAM-Goal translates static goal context into coherent, multi-step physical interactions out of the box.

## 6.3 In-Context Learning with Visual Goals

Whereas Section 6.2 evaluates whether pretrained capabilities transfer directly to unseen tasks, we now examine a complementary question: can a single policy with strictly frozen parameters dynamically alter its execution trajectory when the desired outcome is modified purely through inference-time visual context? We evaluate this form of embodied in-context learning (ICL) by conditioning the same checkpoint on alternative goal images from identical initial scenes—varying target locations, category-to-container assignments, or stacking orders. We first verify that goal-responsive behavior emerges directly from co-training, then show that it scales to complex, multi-decision workflows after task-specific post-training, and finally demonstrate cross-embodiment ICL using egocentric human goal images.

![](images/d3db37fc042919024c44787fc1c639233c52b6b9090e93c56b26f2108f7f89d1.jpg)  
Figure 22 Qualitative real-robot zero-shot rollouts. Each row pairs the inference-time goal image (left) with four head-camera keyframes from the resulting zero-shot execution (right), showing the initial scene, two intermediate interaction stages, and the final state. The examples span object transport and placement, coordinated bimanual manipulation, object reordering, and multi-step tool/container interactions.

Pretrained HumanWAM-Goal in-context learning behavior. We first probe visual goal conditioning in the co-trained checkpoint without any target-task post-training. Figure 23 compares two rollouts initialized from the same tabletop scene containing colored blocks, conditioned on visual goals that specify distinct destinations for a bottle. Keeping all policy weights fixed, switching the goal image steers the robot to place the bottle either on the tabletop in front of the green block (panel a) or atop the elevated purple block (panel b). While fine-grained release stability on the elevated block remains imperfect without task-specific adaptation, these rollouts confirm that spatial goal grounding is already present out of the box after hybrid pretraining.

![](images/8f75c14fc964d4c9527ff6e550f38a1d5e286e94748e63142fe20076195cbee6.jpg)  
Figure 23 Visual goal conditioning in pretrained HumanWAM-Goal. Each panel pairs the inference-time goal image (left) with synchronized first-person and third-person keyframes (right). Without target-task post-training or parameter updates, changing the visual goal directs the policy to place the bottle (a) on the tabletop in front of the green block or (b) atop the large purple block.

In-context learning behavior after post-training. We next evaluate whether goal-conditioned steerability is preserved—and extended to multi-step, compositional decisions—after adapting the co-trained checkpoint on task-specific demonstrations. Crucially, once post-training concludes, policy parameters remain frozen across all evaluations; behavioral changes arise solely from the supplied goal image.

Figure 24 evaluates a post-trained policy on a multi-object tabletop organization task under three distinct visual goals. Rather than memorizing a single canonical sorting routine, the frozen policy parses multiple compositional constraints from the goal image within each rollout: it reverses category-to-bin assignments when the destinations of toy cars and plush toys are swapped (panel a vs. panel b), and it modifies both the stacking order (green vs. purple book on top) and the spatial placement of the book stack when prompted by panel c. Visual goal conditioning thus scales beyond single-target localization to coordinate multiple object-level decisions over extended manipulation horizons.

Figure 25 further tests paired goal variations in an ofice workspace involving articulated storage and multi-item sorting. In panels (a, b), the visual goal dictates which drawer of a cabinet holds a tape roll and tape measure; to satisfy the goal, the policy selectively pulls open either the upper or the middle drawer before inserting both items. In panels (c, d), swapping the visual arrangement of a glue stick and two pens between adjacent

![](images/df9dc015060e6f471a0d56146fa8b043165079bccd9b5751e33c645e20937e81.jpg)  
Figure 24 Goal-based in-context learning after post-training for tabletop sorting and stacking. Each panel pairs a supplied goal image (left, blue border) with eight chronological third-person frames (right, read left-to-righ across the top then bottom row; green border marks the final state). The goals specify (a) sorting toy cars into the blue bin, plush toys into the white bin, and stacking books near the table center with the green book on top; (b) swapping the toy-to-bin assignments; and (c) retaining the original toy assignments while placing the purple book on top of a stack shifted toward the white bin.

(d) Swapped holder assignment

Goal detail

## Drawer selection

## (a) Upper drawer

1 Initial scene  
(b) Middle drawer  
![](images/fe3d419cb567afe5367bd3684295a1911d63c886e576873c2f3c513ba18380ef.jpg)  
Goal detail

1 Initial scene  
![](images/8101cac15ac563a854d18c1d346d91ce653fc200ac4e276937bd9a0082f5f0c2.jpg)

![](images/9620d8da54d3725ecfea40dd1218fa06bc0820eb3c62ad01f4b9438e9f14ae85.jpg)  
4 Store tape roll  
Tape roll and tape measure in the upper drawer

![](images/ba56666d8da37f9c7c61669f4b1be875f9367a6c2d0dfef863524d7e6c6f1048.jpg)

![](images/7410193ac7d1b4ba955d3c5e489f1bb2c0e650cfa7ad25ae6a912b386af8fd8c.jpg)  
1 Initial scene  
Goal detail

![](images/cda85a7d1b8b2b017dbbb4fd887701525991223d998192653d30f6d3d2134a1a.jpg)

![](images/3b06a53fbaf9e94c2f0a0a3b82ea62b39e2df9281e7947be9efe7cf3bc917a33.jpg)  
4 Store tape roll  
Tape roll and tape measure in the middle drawer

![](images/584e5ce65984dd3a6e986f22fad9e8b8cf74219aaec45a63ef59876920f7e867.jpg)

## Holder assignment

(c) Reference holder assignment  
![](images/4a1ee343b5fa73fcb52d439e4ad7355f786f7bddaf44630b5c482b78bd2dac08.jpg)

![](images/c31dc603acae31eb2bcc8c48413dc268f7d10680818e7c71da19f9b517bc03bd.jpg)

![](images/dddcc52d32ef54d5dc6bc4bceec4dbdbd8a232b19972ee78c5fd77b58ea96220.jpg)  
4 Store first pen  
Glue: left holder Pens: right holder

![](images/66c87fa23f0c58f9c671753921a4ab3cd6aa4ea8dc5186780aa84a13c6929146.jpg)

![](images/d629a152f067dabc9602f116d778e799cd26fdcfb5321519c640e08d7ba2dbc3.jpg)

![](images/03bad996e991f8d100aa5d006acddedff96b9d67432e7ac0b826419374fe9746.jpg)  
Glue: right holder Pens: left holder

![](images/4f6c08e9b465cff9dbe884ca6f5eb78cca39dcc24cb781d5b6633374575d3622.jpg)  
2 Open upper drawe  
5 Store tape measure  
4 Store first pen

![](images/30fedc042705f7a874ab850029caaff1612abcb3163f25a61c2eebcbe774ad0a.jpg)

![](images/65d6123b437e7e45fa64a110d0be5978e4b56eab3710ed934c377ebaef255b82.jpg)  
2 Place glue stick

![](images/eaa9818b9c589d913f3f10541d161a5bf89d4fdc07d39e2a90f24b988ed2c397.jpg)

2 Open middle drawer  
![](images/ed3ef3ea7faf4b6ad1579684087fdc05d6024c5f5ba816b656aebbb7febb93f2.jpg)  
5 Store second pen  
5 Store tape measure

![](images/ea6b47cdfaa9aff53f301a89e7d3156d0bbca0cdb9addbded1b0d767752b1c8b.jpg)

![](images/58a64509a3dfb75a5ea8a79950593cf9527ef50ac89f0eb10b23c2c0feff8a09.jpg)

![](images/28fe82706f90ae0d1b1c10fb14871e2ab48f1fa492f7e2ec87c34eb17643865b.jpg)  
2 Place glue stick

![](images/e9b1de8cb8a29d5a8e03dec880d89f6b8b3b3b43313424cc3b9d66f6a8929015.jpg)  
5 Store second pen  
6 Items stored  
3 Lift tape roll

![](images/3a7f87d2d1581ed3f6341067ccb51a77cf8d790ecfc2415f598f6ecb2e7db73c.jpg)

![](images/5d46102a7f5e7182590e9c47793ea82346fad2d687c874040b15e049d1d73d5f.jpg)

![](images/99f1a45c7e13fcb5717b6d5cf33c484274b198c69fbcd46fe8bd6c3b774c981c.jpg)

3 Lift tape roll  
![](images/c635173d51ed366e9c68a68f6d0e22a92a9f065b233f17bb55f5835ced6bae71.jpg)  
6 Items stored

![](images/15cc072fa210492ed0c87909728b3e4e03640d2e33025583fb9c7416da486c2e.jpg)

3 Glue stored  
![](images/db6af15794a81defa11710b18082ff42db38a043f69940d382074562966bbd1e.jpg)  
6 Pens stored

![](images/7f865ddc86987b85d45bef455447a7c926d59f0396ee6b6d14239b368fccd888.jpg)  
3 Glue stored

![](images/66c5a90b07b2f99ba6b510b69ccd1e707d4523d46923134fb678ee8eb3172bbb.jpg)  
6 Pens stored

![](images/be29026383e5026bc9905d32db0f3318f9adf68471e9a6f9f56fe1830586df12.jpg)  
Figure 25 Goal-based in-context learning after post-training for office desk organization. Each panel pairs a goal image and enlarged detail view (left) with six ordered third-person execution frames (right). Panels (a, b) redirect the storage of a tape roll and tape measure from the upper drawer to the middle drawer. Panels (c, d) swap the destinations of a glue stick and two pens between the left and right holders (defined in the goal-image frame). Endpoints show item placement prior to drawer closure in (a, b) and after the final pen release in (c, d).

holders prompts the robot to invert its placement sequence accordingly. These comparisons demonstrate that post-trained policies remain tightly conditioned on visual context, even when realizing the target state requires prerequisite interactions with articulated mechanisms.

Cross-embodiment in-context learning from human goal images. Because HumanWAM-Goal is co-trained on heterogeneous human and robot corpora within a unified visuomotor framework, we hypothesize that it learns an embodiment-agnostic representation of scene structure and manipulation outcomes. To test this hypothesis directly, we supply the post-trained robot policy with egocentric human goal images—in which human hands, rather than robot grippers, are visible alongside the desired object configuration—and evaluate whether the robot can ground and execute the specified arrangement without parameter updates.

As shown in Figure 26, despite the visual domain gap and the presence of human hands in the conditioning images, the policy accurately interprets the underlying spatial relations and adapts its execution across both paired comparisons: it slots the brown notebook into either the rightmost or leftmost compartment before arranging the white notebooks (panels a vs. b), and it caps the mug before transporting it to either the monitor or the file rack (panels c vs. d). These results are consistent with the hypothesis that human–robot co-training facilitates the transfer of goal semantics across embodiments.

## 6.4 Long-Horizon Execution

While the paired-goal comparisons in Section 6.3 isolate immediate responsiveness to visual context, real-world deployment often requires chaining diverse manipulation skills over extended horizons. In such settings, each action alters the scene state for subsequent subgoals, requiring the policy to transition smoothly across subtasks without disturbing previously completed arrangements. We evaluate HumanWAM-Goal after task-specific post-training across two representative long-horizon workflows: multi-stage desk organization within a shared tabletop workspace and mobile household tidying across multiple work areas.

Multi-stage tabletop organization. Figure 27 illustrates a continuous 14-stage rollout on a cluttered ofice desk. Across the trajectory, the policy coordinates distinct manipulation primitives—capping and relocating a mug (stages 01–03), sorting small stationery into holders (04–05), opening and closing two separate drawers to store a tape roll and tape measure (06–10), and filing notebooks into a vertical rack (11–14). Because all operations unfold within a confined workspace, later actions must navigate around earlier placements: the capped mug remains undisturbed near the monitor, and closed drawers stay shut while the robot manipulates adjacent objects. This rollout demonstrates coherent skill chaining across rigid-object placement, container insertion, and articulated-object manipulation.

Multi-area mobile household tidying. Figure 28 extends evaluation from a stationary tabletop to room-scale mobile manipulation on Astribot S1. Within a single uninterrupted sequence, the policy sorts toys into bins and stacks books at the main table (keyframes 01–08), navigates to a side table to collect and discard two plastic bottles into a waste bin (09–15), traverses to two chairs to gather deformable garments into a laundry hamper (16–27), and finally returns to the organized tabletop (28). By coupling base navigation with bimanual control in the unified 49-D hybrid action space (Section 5.2), the model transitions seamlessly across spatial zones and object modalities—from rigid toys and bottles to deformable clothing. Together, these demonstrations show that HumanWAM-Goal can sustain temporally consistent, multi-skill execution across both dense tabletop layouts and extended mobile-manipulation workflows.

## 7 Related Work

## 7.1 World Action Models

World–action models couple visual dynamics prediction with action generation, allowing robot policies to inherit manipulation priors from pretrained video models [57, 65]. DreamZero [65] learns control through joint video–action modeling, while LingBot-VA [26] integrates frame prediction and policy execution within an autoregressive difusion framework. The role of future visual information in action prediction remains an important architectural consideration [56, 70]. Fast-WAM [67] retains video prediction during training but omits future video generation during inference. OpenWAM [56] systematically studies world–action information flow and embodied pretraining, demonstrating the benefits of explicit world-to-action interaction

Initial scene

## Notebook arrangement

(a) Brown notebook on the right

Human goal image

![](images/2aa8606b89ae683d1c7e40354527b7147a9fe1603f6e35be2ba39c7d9c850013.jpg)  
Robot first-person view  
Goal detail

![](images/88c202f930212d9679c5d22af903d9be7378595bc439836cfb196dcad9a2821a.jpg)

![](images/66459f1acc95250f77652bb0eadbff17571943d6140864623b45cf4b39e718cc.jpg)  
Third-person view  
Brown notebook: rightmost rack slot

![](images/696766be6eb098fab9675dcad4267f9f5225b70157643f4ebd2feada07761300.jpg)

## (b) Brown notebook on the left

![](images/1c6ebeccfea3cc930c2cd5f0e928e6587a7d8432c06ebffd05f825e7ad366a6d.jpg)  
Robot first-person view  
Goal detail

![](images/80eabdae6dcd96c4128bc1205826ce6288f42539a68231c68b9970f6b03287ec.jpg)

![](images/6978ce08126b85b4d984fda07d8fdc1842b213d9b44cde4a2009af7b13ba25d7.jpg)  
Third-person view  
Brown notebook: leftmost rack slot

![](images/490aeff4e4dd895603c43d40b598e90b782096b69bb5b5a7d67e4d1eb935a806.jpg)

## Mug placement

(c) Covered mug near the monitor

Human goal image

![](images/cdbaa6ceecc08d5d3b0b287966b650769c40285b780fe55f91a1baa39b38cdd9.jpg)  
Robot first-person view  
Goal detail

![](images/c0bcba523a1f9c1b9f98a4a7a1e698c1f6a36b214b0535eae51a0c6c642301d6.jpg)

![](images/aacce09f236e313834baa29a60f58a1c8772be00586b785dd8bcb704c60af9c0.jpg)  
Third-person view  
Covered mug: near the monitor

![](images/39777b686dde147bad89c580932d911b76f3a4d8d7c3b7d8b41fded7b25273b0.jpg)  
(d) Covered mug near the file rack

Human goal image

![](images/88abd44609d1a4b8ba457177d72a9eac292c8acfb393f0772c2cc16b748ee342.jpg)  
Initial scene Robot first-person view  
Goal detail

![](images/7d22c847d202453db526795449a5926918ffe42a33934fb82499f82f3adb0586.jpg)

![](images/5780d22b01b26a9001a74da4f09a9b53ebdc805588a22aed91be897a3441a354.jpg)  
Third-person view  
Covered mug: near the file rack

![](images/dbad763b49fba1d2fa3e78c69788dd01037bde65431db9b4e946f8c873f0df4a.jpg)  
Place brown notebook

![](images/b589edc189b9d03c656307818ce18f433e321694a39a4c1fe2d63abc21f6f7c1.jpg)

![](images/1324e61d1ac9787c5de84839c811642da879850665d03b2d2de1bd92e30de7d3.jpg)  
Place brown notebook

![](images/c8bb97d0d36c3e21883d94601c141bd4081ff4ed64369946a23f6bb62f385d51.jpg)

![](images/32b83c79871cc4148d43fe9f3acc5d6f6348bc2ddb4ccb7e8ea4760ede31de04.jpg)

![](images/f11fd087bb617e9be5b8ec92e170ab37cdd7bfec2f651dba2d6b7924b810e16e.jpg)  
Cover the mug

![](images/74d1f57ea71eb5852a8d852104e14d0bf4550883d9e49cc60834d1943c381132.jpg)  
Cover the mug

![](images/9a259351952a9950ddb947dd99234779e475ca3e1e01f5c97f92ad5c77450eb2.jpg)

![](images/9bc3a8858936e67da056eae55f0638534461b4539f19850ff5503a2f3ca19b33.jpg)  
Arrange white notebooks

![](images/8dc8921b058f88763dfc65568da59d8cef71334cbe3f067d50719f17c394c654.jpg)

![](images/62982987b23694c1cbe4e133a99d5fc2a410079149549b4144a2a02ee6e7b19a.jpg)  
Arrange white notebooks

![](images/956ca59d658d50ca5c2d5d8e8effc31c37c52ec13eeaa9b95681ecb969fd4fa5.jpg)

![](images/117ccc2fc336dd288e5ad9c13914fc356bdfecb244dbe4309fd4b5c6ed4380a4.jpg)  
Move the mug

![](images/7be7bbed460534d2a2e5b99693e25ba9a26342e4e9282637ffa41d6928ba178c.jpg)

![](images/6fdaf13700bdf509825db0efd7c2188b48c834b69f58180b9f6446a2ea0be29f.jpg)  
Move the mug  
Selected endpoint

![](images/06dd06b93f528e3a49422a5d8d0e81f18faae48afd408349a9910ef13f26d242.jpg)

![](images/e765c80416954b84b16e28ff1a7af261446cab667abdb19c6711ebe7b2573382.jpg)

![](images/418bc0b59624ca45b4f4e37f557b3f0ffa38fec3dd32110378afed960523f1d4.jpg)  
Selected endpoint

![](images/9d5f4ad439aa13374166fc8fe5010d6145fc08b350f76d7a65c930a63793233f.jpg)

![](images/07b469192d2857756ef020df228f682f390293d252c19b479225ba7d4fedc79d.jpg)  
Selected endpoint

![](images/087ae6eb9b52173069eabde2db1b5bab1df0cd7fb04c3fa2752fa382ac589eb3.jpg)

![](images/cd74518a1731b20e14abbcded77ccf72c3a5fdf730ece5542cbf6e479164bafc.jpg)

![](images/100a1541345f0c2fb81d92807adffb200dc3094c750aabea3f73a920429de789.jpg)  
Selected endpoint

![](images/1ea661c2b8d3bb0ea1d545d66fa4f6d24cf1248d1ed0c66264dfc766dc55df63.jpg)

![](images/c597af515d13174a25afb2219ef84ef21174e1e8cacea7a10889182540bb2ed5.jpg)

Figure 26 Cross-embodiment in-context learning from egocentric human goal images. Each panel pairs a human-demonstrated goal image and enlarged detail view (left, showing human hands) with synchronized first-person and third-person robot execution frames (right). Panels (a, b) direct the robot to file the brown notebook into the rightmost or leftmost rack slot, respectively. Panels (c, d) specify placing the covered mug near the monitor or beside the file rack. Green borders highlight the completed placements upon gripper release.

![](images/1778f96aacdecb59bceb1fcde547570f417ed134817483f2f68d326cd93cc74e.jpg)  
Figure 27 Long-horizon desk organization with HumanWAM-Goal. Fourteen chronologically ordered stages from a single continuous rollout, shown through paired first-person (left) and third-person (right) views. The trajectory sequentially executes mug capping and placement, pen and glue-stick sorting, articulated drawer storage of a tape roll and tape measure, and notebook filing. Green borders mark the terminal state after the final notebook is released into the rack.

Long-horizon household tidying  
![](images/fbe9f0ad57340de6056e525f799670351afe1c721a2429d7f120bf9ee9065a0c.jpg)  
Figure 28 Long-horizon mobile household tidying with HumanWAM-Goal. Twenty-eight chronologically ordered keyframes from a single continuous rollout spanning multiple work areas: sorting tabletop toys and books, navigating to a side table to collect and discard empty bottles, gathering garments from two chairs into a laundry hamper, and returning to the main table.

and synchronized joint denoising. InternW0-∆ [34] instead uses training-only future supervision to learn predictive representations for action generation without video rollout during inference. Extending world–action modeling to compositional manipulation, ViGAR [18] combines visual subgoal planning with goal-conditioned execution. Our work complements these architectural developments by studying the efects of egocentric data scaling across diferent world–action configurations.

## 7.2 Pure Egocentric Pretraining

Pretraining on human manipulation provides visual representations and action priors for robot learning before embodiment-specific adaptation [22, 36]. Being-H0 [31] develops human-motion pretraining for dexterous manipulation, while EgoScale [73] scales action-labeled egocentric pretraining to over 20,000 hours and reports a log-linear relationship between data scale and validation loss. EgoScale subsequently introduces aligned human–robot mid-training to facilitate embodiment transfer. HumanScale [33] establishes an empirical scaling law relating egocentric pretraining hours to validation action loss after robot post-training and finds that human data can outperform robot data at matched pretraining durations. Dyna-2 [14] extends human-video pretraining to one million hours, reporting scaling in zero-shot action prediction on unseen robot data alongside improvements in post-trained robot task performance. Building on these findings, Open-HumanScale systematically evaluates egocentric scaling through closed-loop policy success under both in-distribution and out-of-distribution conditions. Beyond data duration, our controlled comparisons establish diversity as an additional scaling dimension and quantify gains in the sample eficiency of robot adaptation.

## 7.3 Zero-Shot Generalization and In-Context Learning

Generalist robot policies increasingly extend beyond task-specific adaptation toward direct execution in unfamiliar tasks, environments, and embodiments [21, 65]. Contextual guidance provides a complementary mechanism for specifying unfamiliar behaviors without task-specific parameter updates. π [42] incorporates language, behavioral metadata, and visual subgoals into multimodal prompts, enabling compositional generalization and flexible control over task execution. Demonstration-based in-context learning additionally conditions behavior on examples and interaction history [16, 28]. RoboICL [28] combines demonstration context with accumulated interaction memory while keeping model parameters fixed. Our HumanWAM-Goal experiments connect these capabilities to the scale of human experience. With robot training data held fixed, we quantify the efect of increasing egocentric data on zero-shot robot performance and qualitatively evaluate visual-goal conditioning before and after robot post-training.

## 8 Conclusion

We presented Open-HumanScale, a fully open recipe for scaling robot learning with egocentric human experience. Our study of pure egocentric pretraining from 100 to 50,000 hours reveals an empirical log-linear scaling law in downstream out-of-distribution success. Expanding task and scene diversity strengthens this scaling trend, while larger-scale pretraining improves the sample eficiency of robot adaptation. Controlled experiments further demonstrate the value of human motion prediction and cross-source action alignment. Following this recipe, HumanWAM-Vanilla achieves competitive performance in simulation and improved robustness to visual distribution shifts on physical robots after robot post-training. Our release of annotated data, training code, checkpoints, and evaluation protocols provides a reproducible foundation for further research on egocentric scaling.

HumanWAM-Goal extends this exploration to joint human–robot co-training with visual-goal conditioning, improving zero-shot transfer in simulation and achieving state-of-the-art average success on RoboTwin 2.0. On physical robots, hybrid scaling with robot-balanced sampling boosts zero-shot success and expands task coverage without target-task demonstrations. Visual goals also enable embodied in-context learning with frozen parameters, steering policies across spatial and compositional variations as well as cross-embodiment human goal images after post-training. Finally, post-trained policies support coherent subgoal chaining across long-horizon desk organization and room-scale mobile household tidying.

## References

[1] AgiBot World Contributors. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[2] Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa. Scalable Behavior Cloning with Open Data, Training, and Evaluation, 2026. URL https://arxiv.org/abs/2606.27375.

[3] Anyverse Dynamics and OpenPhoenix Team. Phoenix: Scaling Robotic Manipulation through Visual and Interaction Representations, October 2026. URL https://openphoenix-team.github.io/Phoenix-pages. Blog post.

[4] BeingBeyond Team. Being-H0.8: A latent tactile world-action model at scale. BeingBeyond Technical Report, 2026. URL https://research.beingbeyond.com/being-h08.

[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[6] Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, and Yao Mu. AHA-WAM:Asynchronous Horizon-Adaptive World Action Modeling with Observation-Guided Context Routing, 2026. URL https://arxiv.org/abs/2606.09811.

[7] Jisong Cai, Yao Mu, Ganlin Yang, Zhe Cao, Zhangzheng Tu, Xing Gao, Kailin Li, Xinyu Zhan, Lixin Yang, Yangkun Zhu, Haoxiang Ma, Ming Zhou, Qiaojun Yu, Yufei Xue, Liqun He, Yifei Yao, Yifan Zhu, Long Ling, Bingqi Jiang, Haoyu Guo, Xueyue Zhu, Bowen Zhou, Bin Zhao, Tianfan Xue, Chunhua Shen, and Weinan Zhang. InternW0: A foundational physical world model for eficient real-world interactions, 2026. URL https://arxiv.org/abs/2609.27656.

[8] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025. URL https://arxiv.org/abs/2506.18088.

[9] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies, 2026. URL https://arxiv.org/abs/2607.04434.

[10] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems (RSS), 2024.

[11] Yufan Deng and Daquan Zhou. HumanNet: Scaling human-centric video learning to one million hours. arXiv preprint arXiv:2605.06747, 2026.

[12] Dexmal. OpenDM: An Open-World Foundation Model for General-Purpose Embodied Intelligence. DM0.5 model and software release, 2026. URL https://github.com/dexmal/opendm.

[13] Zeyuan Ding, Wenhai Liu, Yang Xu, Jiayu Hu, Yinda Chen, Yi Zhang, Yong Dai, Jian Tang, and Xiaozhu Ju. Pelican-VLA 0.5: Attending before acting benefits generalization. arXiv preprint arXiv:2607.06655, 2026.

[14] Dyna Robotics. Dyna-2: A 1-million-hour scaling law for world-action models. Technical report, Dyna Robotics, August 2026. Technical report, https://www.dyna.co/dyna-2.

[15] Fast-WAM Contributors. Fast-WAM: Oficial implementation and Optional IDM release. Software repository, 2026. URL https://github.com/yuantianyuan01/FastWAM. Accessed November 3, 2026.

[16] Max Fu, Huang Huang, Gaurav Datta, Lawrence Yunliang Chen, Will Panitch, Fangchen Liu, Hui Li, and Ken Goldberg. ICRT: In-Context Imitation Learning via Next-Token Prediction. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 5937–5944, 2025. doi: 10.1109/icra55743.2025.11128272. URL https://doi.org/10.1109/icra55743.2025.11128272.

[17] GenRobot. Gen-HumanEgo: A more training-ready open-source egocentric human data for embodied AI. Dataset release, September 2026. URL https://www.genrobot.ai/blog/gen-human-ego.

[18] Shukai Gong, Xuanran Zhai, Yintianrun Zhang, Ruopeng Cui, Ye Huang, Yiyang Fu, Dexuan Lyu, Chaojie Li, Xinyi Song, Peiwen Lin, et al. Rethinking world-action model for compositional and in-context robotic manipulation. arXiv preprint arXiv:2610.02368, 2026.

[19] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising, 2026. URL https://arxiv.org/abs/2604.26694.

[20] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video, 2025. URL https://arxiv.org/abs/2505.11709.

[21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[22] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hofman, and Danfei Xu. EgoMimic: Scaling imitation learning via egocentric video, 2024. URL https://arxiv.org/abs/2410.24221.

[23] Alexander Khazatsky et al. DROID: A large-scale in-the-wild robot manipulation dataset, 2024. URL https: //arxiv.org/abs/2403.12945.

[24] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=euMVC1DO4k

[25] Hao Li, Ganlong Zhao, Yufei Liu, Haotian Hou, Guoquan Ye, Tongyan Fang, Chunxiao Liu, Siyuan Huang, Jianbo Liu, Xiaogang Wang, and Hongsheng Li. ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining, 2026. URL https://arxiv.org/abs/2606.17200.

[26] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[27] Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao. Data Scaling Laws in Imitation Learning for Robotic Manipulation. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 88b7b2c896506daabc8d3fd587055167-Abstract-Conference.html.

[28] Fangcheng Liu, Yeqing Shen, Anda Cheng, Weishi Mi, Chao Tang, Chenyuan Liu, Yushun Xiang, Tingguang Li, Yong-Lu Li, and Yehui Tang. RoboICL: Embodied in-context learning with GPT-6 Astra, 2026. URL https://arxiv.org/abs/2609.34261.

[29] Yicheng Liu, Zibin Dong, Baijun Ye, Tianyuan Yuan, Tao Jiang, Anqi Yang, Shicheng Cao, Haonan Liu, Yue Sun, Zihan Guo, Xiao Liu, Dong Ke, Changxun Pan, Chenru Wu, Tailai Cheng, Xiaoshu Ren, Xinlei Zhang, Jianning Cui, Zijie Zhao, Haoyu Zhang, Kaiming Xu, Haodong Yang, Bowen Zhang, Jiahui Niu, Shaoting Zhu, Shiduo Zhang, and Hang Zhao. G0.5: One Autoregressive Stream for Robot Reasoning and Action, 2026. URL https://arxiv.org/abs/2608.11739.

[30] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

[31] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-H0: Vision-language-action pretraining from large-scale human videos, 2025. URL https://arxiv.org/abs/2507.15597.

[32] Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, Yuqiang Yang, Jiaqi Peng, Xueyuan Wei, Yangkun Zhu, Jiahao Jiang, Xing Gao, Hanqing Wang, Feng Yuan, Kailin Li, Xueyue Zhu, Tai Wang, Yan Ding, Jiangmiao Pang, Jia Zeng, Jingjing Zhang, Bowen Zhou, Yao Mu, Chunhua Shen, and Weinan Zhang. InternVLA-A1.5: Unifying understanding, latent foresight, and action for compositional generalization. arXiv preprint arXiv:2607.04988, 2026.

[33] Juncheng Ma, Jianxin Bi, Yufan Deng, Xuanran Zhai, Kewei Zhang, Ye Huang, Bo Liang, Shukai Gong, Jiankai Tu, Xiaotian Tang, Jiaxin Li, Kaiqi Chen, Duomin Wang, Yuqi Wang, Bingyi Kang, Eric Huang, Zhiyang Dou, Zhen Dong, Enze Xie, Wojciech Matusik, Tat-Seng Chua, and Daquan Zhou. HumanScale: Egocentric human video can outperform real-robot data for embodied pretraining, 2026. URL https://arxiv.org/abs/2606.20521.

[34] Xingyu Miao, Zizun Li, Baole Fang, Kaiwen Song, Tenghui Wang, Hanxue Zhang, Yating Wang, Xudong Li, Yuping He, Xueyuan Wei, Chao Gao, Xijie Yang, Yingxiang Xu, Kerui Ren, Wenqi Guo, Jianjun Zhou, Xinzhe Wang, Weiguang Zhao, Ni Yang, Zetao Cai, Yufei Xue, Hengjie Li, Zeyu He, Yuanzhen Zhou, Rong Fu, Jianyang Zhang, Siwei Cui, Fuxian Huang, Yunsong Zhou, Xing Gao, Yifei Yao, Qiaojun Yu, Kailin Li, Ming Zhou, Mu Huang, Xinyue Li, Wenze Cui, Bingqi Jiang, Xueyue Zhu, Junting Dong, Haoyu Guo, Tao Lu, Mulin Yu, Bowen Zhou, Bin Zhao, Tianfan Xue, Weinan Zhang, and Chunhua Shen. InternW0-∆: A world action model bridging predictive dynamics and actions with 20k+ hours of open data, 2026. URL https://arxiv.org/abs/2609.31394.

[35] Motubrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, Louis Liu, Mengchen Cai, Rongxu Cui, Ruowen Zhao, Runqing Wang, Shuhe Huang, Yao Feng, Yinze Rong, Zeyuan Wang, and Jun Zhu. MotuBrain: An advanced world action model for robot control. arXiv preprint arXiv:2604.27792, 2026.

[36] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation, 2022. URL https://arxiv.org/abs/2203.12601.

[37] NVIDIA. GR00T-N1.7-3B. Model card, 2026. URL https://huggingface.co/nvidia/GR00T-N1.7-3B.

[38] NVIDIA. Cosmos 3: Omnimodal world models for Physical AI. arXiv preprint arXiv:2606.02800, 2026. URL https://arxiv.org/abs/2606.02800.

[39] OpenAI. GPT-6 Astra: A new generation of intelligence. https://openai.com/index/gpt-6-astra/, 2026. Accessed November 5, 2026.

[40] OpenAI. GPT Image 2.5 Sunburst. https://developers.openai.com/api/docs/models/gpt-image-2. 5-sunburst, 2026. Model snapshot: gpt-image-2.5-sunburst-2026-09-08; accessed November 5, 2026.

[41] PhyAI Team. PhyAI: Latency-first serving engine for Physical AI, 2026. URL https://github.com/mingti-org/ phyai.

[42] Physical Intelligence. π<sub>0.7</sub>: A steerable generalist robotic foundation model with emergent capabilities. Technical report, Physical Intelligence, April 2026. URL https://www.pi.website/download/pi07.pdf.

[43] Ryan Punamiya et al. EgoVerse: An egocentric human dataset for robot learning from around the world, 2026. URL https://arxiv.org/abs/2604.07607.

[44] RoboTwin Team. RoboTwin 2.0 Leaderboard. https://robotwin-platform.github.io/leaderboard, 2026. Accessed November 1, 2026; leaderboard updated October 30, 2026.

[45] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied Hands: Modeling and Capturing Hands and Bodies Together. ACM Transactions on Graphics, 36(6):245:1–245:17, 2017. doi: 10.1145/3130800.3130883. URL https://doi.org/10.1145/3130800.3130883.

[46] Peter H. Schönemann. A Generalized Solution of the Orthogonal Procrustes Problem. Psychometrika, 31(1):1–10, 1966. doi: 10.1007/bf02289451. URL https://doi.org/10.1007/bf02289451.

[47] Modi Shi, Li Chen, Jin Chen, Yuxiang Lu, Chiming Liu, Guanghui Ren, Ping Luo, Di Huang, Maoqing Yao, and Hongyang Li. Is Diversity All You Need for Scalable Robotic Manipulation? IEEE Transactions on Robotics, 42: 1872–1883, 2026. doi: 10.1109/tro.2026.3686184. URL https://doi.org/10.1109/tro.2026.3686184.

[48] Ruiyang Si, Jianxin Bi, Shunyu Yang, Rui Ni, Wenbo Huang, Qiang Wang, Shulong Jiang, Yintianrun Zhang, Feixiang Ruan, Kewei Zhang, Duomin Wang, Xiuyu Li, Haiwen Feng, Zhen Dong, and Daquan Zhou. Fewer tokens, better action: GPT-6 Astra robot agents with 14% higher success rate but 65% fewer tokens. arXiv preprint arXiv:2610.01939, 2026.

[49] Wenxuan Song, Jiayi Chen, Jingbo Wang, Shuai Zhou, Xicheng Gong, Zehua Fan, Ziyang Zhou, Junwu E, Haodong Yan, Fuhao Li, Qize Yu, Xu Huang, Pengwei Wang, Wen Chen, Shunbo Zhou, and Haoang Li. UniWAM: Unified World-Action Model, 2026. URL https://arxiv.org/abs/2610.02054.

[50] Zhihao Sun, Liu Liu, Xinjiang Wang, Haoyi Jiang, Wei Feng, Huiqiang Zhang, Xiaosong Jia, Zhizhong Su, and Zuxuan Wu. Ego4WAM: What matters when scaling egocentric human data for robot learning? arXiv preprint arXiv:2609.40341, 2026.

[51] GigaBrain Team, Angen Ye, Axiang Sun, Can Jin, Chenxi Cheng, Chong Shi, Dengke Shang, Dingqian Zhang, Guan Huang, Guangqiang Wang, Guangqing Ding, Guo Li, Hangcong Li, Hengyu Zhong, Hongtao Lu, Jianbo Qin, Jiming Mao, Jing Zhu, Jindi Lv, Jingzhi Cui, Junjie Xie, Junyi Bao, Kai Liu, Lei Yuan, Limin Long, Lv Feng, Mingming Yu, Peng Li, Pengfei Yi, Qi Li, Qianli Zhang, Qingfang Li, Qitang Hu, Rui Zhang, Shaoyan Sun, Shibo Sun, Shiying Duan, Tenghui Chen, Tianze Liu, Weijie Ke, Wenyao Xue, Xiaofeng Wang, Xiaoyu Tian, Xinyu Liu, Xinze Chen, Yang Wang, Yankai Wang, Yejun Zeng, Yifan Li, Yifei Nie, Yilong Li, Yilong Liu,

Yongchao Feng, Yumeng Wang, Yun Ye, Zhichao Liu, Ziheng He, Zonghai Yang, and Zheng Zhu. GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture, 2026. URL https://arxiv.org/abs/2608.15875.

[52] Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, Nan Sun, Qiao Sun, Runze Suo, Heyun Wang, Yunhong Wang, Rujie Wu, Caoyu Xia, Lina Zhang, Jack Zhao, Guoliang Chen, Wenlong Chen, Xinze He, Bin Li, Qing Li, Zhuorong Li, Heng Qu, Wenxuan Song, Diyun Xiang, Yifan Xie, Peiran Xu, Hangjun Ye, Wen Ye, Han Zhao, and Quanyun Zhou. Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories, 2026. URL https://arxiv.org/abs/2607.15330.

[53] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[54] Hao Wang, Jiajun Wen, Jingzhi Liu, Shuoshuo Xue, Zhiliang Chen, Min Lin, Yicheng Chang, Xiaoyu Guo, Yukang Zhuo, Zheng Chong, Yunshuang Nie, Jian Zhang, Weijia Liufu, Qingman Wu, Heming Xu, Bingchang Song, Dantong Wu, Zhiyuan Wang, Hang Xu, Jianhua Han, Bokui Chen, Shen Zhao, Rui Li, and Xiaodan Liang. EWAM: Emergent Depth-Wise Specialization in a Unified Embodied Model – From Semantic Understanding through Visual Foresight to Action, 2026. URL https://arxiv.org/abs/2609.39973.

[55] Tianheng Wang, Zhou Xie, Heng Jia, Jianhua Xu, Tong Zhang, and Kaicheng Yu. An Action Is Worth One Patch: Unified World-Action Modeling with PatchWAM, 2026. URL https://arxiv.org/abs/2609.25961.

[56] Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, Yu-Kai Wang, Yuhan Rui, Shenzhe Yao, Zhen Yuan, Zhenhao Shen, Kefei Zhu, Zijie Zhu, Ning Gao, Xiaowei Chi, Guanqi He, Shanghang Zhang, Hao Dong, Lin Shao, and Hang Zhao. OpenWAM: An open, modular exploration towards systematic world-action model pretraining. arXiv preprint arXiv:2609.07398, 2026.

[57] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/ hash/2c37c5bcef24b9541550261dcd63261b-Abstract-Conference.html.

[58] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026.

[59] Xionghao Wu, Yijun Yang, Shiyang Zhou, Haoze Sun, Jianhui Liu, Songsong Yu, Jiyao Zhang, Wenbo Li, Bo Wang, Guoqing Ma, Lin Song, Renjie Liao, Shenghe Zheng, Wei Tang, Xiaojuan Qi, Yanwei Li, Yuan Zhang, Zhuotao Tian, Haoyang Huang, and Nan Duan. ZimaBlue: Evolving generalizable world action models through scalable video pre-training, 2026. URL https://arxiv.org/abs/2609.00188.

[60] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, Li Yi, Angel X. Chang, Leonidas J. Guibas, and Hao Su. SAPIEN: A SimulAted Part-Based Interactive ENvironment. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11094–11104, 2020. doi: 10.1109/cvpr42600.2020.01111. URL https://doi.org/10.1109/cvpr42600.2020. 01111.

[61] Lishan Yang, Wenxuan Song, Xi Wang, Pingyue Sheng, Zheng Fang, Ziyang Zhou, Junjie He, Haodong Yan, Jiayi Chen, Nan Sun, Qiao Sun, Pengwei Wang, Lingqiao Liu, Yan Wang, Yuxiang Gao, Feras Dayoub, and Haoang Li. 4D-WAM: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fields, 2026. URL https://arxiv.org/abs/2608.08023.

[62] Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning, 2026. URL https://arxiv.org/abs/2602.11236.

[63] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, Min Cao, Peng Li, Qiuping Deng, Wenjun Mei, Xiaofeng Wang, Xinze Chen, Xinyu Zhou, Yang

Wang, Yifan Chang, Yifan Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. GigaWorld-Policy: An Eficient Action-Centered World–Action Model, 2026. URL https://arxiv.org/abs/2603.17240.

[64] Jinhui Ye, Ning Gao, Senqiao Yang, Jinliang Zheng, Zixuan Wang, Yuxin Chen, Pengguang Chen, Yilun Chen, Shu Liu, and Jiaya Jia. StarVLA-α: Reducing Complexity in Vision-Language-Action Systems, 2026. URL https://arxiv.org/abs/2604.11757.

[65] Seonghyeon Ye et al. World action models are zero-shot policies, 2026. URL https://arxiv.org/abs/2602.15922

[66] Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026.

[67] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination?, 2026. URL https://arxiv.org/abs/2603.16666.

[68] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, et al. Hy-Embodied-0.5-VLA: From vision-language-action models to a real-world robot learning stack. arXiv preprint arXiv:2606.14409, 2026.

[69] Xuancheng Zhang, Xuetao Liu, Qianying Tang, Jizhe Wang, Runsheng Wang, Zhijing Cheng, Bochen Lin, Haoran Wen, Ming Li, Kun Zhan, and Yu Liu. ME-Dex 1.0: Bringing Heterogeneous Tactile Sensing into World Action Modeling, 2026. URL https://arxiv.org/abs/2609.21449

[70] Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, and Xin Jin. ImageWAM: Do world action models really need video generation, or just image editing? arXiv preprint arXiv:2606.19531, 2026.

[71] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[72] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=kt51kZH4aG.

[73] Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, Trevor Darrell, Furong Huang, Yuke Zhu, Danfei Xu, and Linxi Fan. EgoScale: Scaling dexterous manipulation with diverse egocentric human data, 2026. URL https://arxiv.org/abs/2602.16710.

[74] Jiaming Zhou, Qihang Zhang, Gangwei Xu, Cunxin Fan, Yujie Zhao, Ruilin Wang, Yiming Luo, Shuai Yang, Xing Zhu, Yujun Shen, et al. Zero-WAM: In-context world-action modeling from human videos for open-ended task generalization. arXiv preprint arXiv:2608.26103, 2026.

[75] Neil Zhou. EgoSuite-Open100K: 100,000 hours of egocentric human data for physical AI. Lightwheel dataset release, August 2026. URL https://huggingface.co/blog/LightwheelAI/egosuite-open100k.

[76] Yude Zou, Yifei Yao, Guanbo Jiao, Jianming Ma, Guanjie Zheng, and Xing Gao. PICA-WAM: Visual Perturbation Invariance via Cross-Appearance Self-Alignment in World Action Models, 2026. URL https://arxiv.org/abs/ 2610.26813.

## Appendix

## A Prompts for Goal Image Synthesis and Review

RoboTwin provides no reference goal images, so we synthesize one head-camera goal image for each scene. The only scene-specific inputs are the initial head-camera observation and the task instruction; no expert trajectory, terminal frame, or simulator state is used. Synthesis has two stages: a grounding stage localizes the taskrelevant objects in the initial image, and an editing stage turns the initial observation into a completed-task observation. Each newly generated evaluation goal image is then reviewed and, if necessary, repaired.

## A.1 Pipeline Overview

Inputs and resolution. The initial head image is letterboxed to 288×224 pixels (width×height) and upsampled 4× to 1152×896 for upload. GPT Image 2.5 generates a 1152×896 image at medium quality, which we downsample to 288×224 before restoring the original black letterbox pixels. The editing request contains three images: (1) the upsampled initial head view, (2) a task-independent robot appearance sheet, and (3) a pixel-registered overlay of the initial view that marks the grounded objects.

Task-specific reference annotations. For each of the 50 RoboTwin tasks, we manually wrote one reference annotation on a reference scene. It lists the objects (identifier, description, bounding box, and whether the object moves or stays fixed), a natural-language description of the required final state, and visual acceptance checks. A few tasks also include optional geometric guides: hinge anchor lines for articulated objects, finalextent outlines, relocation cut-outs, or edit masks. These 50 annotations are fixed across all experiments; Stage 1 adapts them to each new scene.

## A.2 Stage 1: Scene-Specific Grounding

For a new scene, a reasoning model (GPT-5.5, high reasoning efort) adapts the task’s reference annotation to the current initial image and returns structured JSON with a fixed schema. Its inputs are the reference initial image, the current initial image, a copy of the current image overlaid with a pixel-coordinate grid, the reference instruction and annotation, the current instruction, and RoboTwin’s public task description. The prompt template is:

Adapt an initial-scene annotation to a DIFFERENT seed of the same RoboTwin task. Return JSON only. Image 1 is the reference seed’s INITIAL head view. Image 2 is the CURRENT seed’s INITIAL head view. Image 3 is exactly the current initial view with a coordinate grid and rulers labelled in NATIVE pixels. No image is a completed task. Images 1 and 2 are enlarged 4x from a 288x224 native canvas. ALL output coordinates must use native 288x224 pixels. Use Image 3’s numbered grid to locate each edge accurately, then confirm on Image 2. Do not estimate coordinates from a resized display: interpolate the supplied ruler numbers. Identify actual current object instances, including distractors. Adapt every object description, box, arm, color, source/destination, number, direction and numeric coordinate in the plan and checks to Image 2 and CURRENT INSTRUCTION. Do NOT copy reference coordinates, scene sides, colors or asset names blindly. Use the reference’s correction principles, not its seed-specific scene geometry. Only explicitly manipulated objects are fixed=false; distractors are fixed=true. Preserve its physical task semantics and minimal-change rules. In an ambiguous noun (container/object), select the current instance matching the instruction, not an unrelated colorful distractor. If instruction names objects or arms, those names take precedence over the reference plan. Some benchmark instructions mention ONLY THE FIRST SUBSTEP (for example ’grab the red block’ for handover\_block). The TASK ID and reference’s complete final-state plan still specify the whole task: preserve the transfer and final placement, not merely that first grab. Do not drop the final task relation because an abbreviated instruction omits it. Ground solely in initial RGB, instruction and the example. Do not infer hidden simulator poses, consult future states, or fabricate an observed terminal image. Keep the same context method: registered initial labels; stationary anchors for articulated devices; proposed final extents only when the reference has those; unscaled initial-pixel relocation guides only when the reference has those; edit masks only when the reference has those. Empty arrays for unused features. Objects may differ in count/identity between seeds. Tight boxes must cover actual visible objects. Coordinates are integer pixels within x=0..288, y=0..224. Bounds [x0,y0,x1,y1] have positive width/height. Point coordinates use x=0..287,y=0..223. No more than 16 objects.

For hinged devices, mark the ACTUAL stationary base edge/corners and original   
hinge from Image 2, not an axis-aligned imagined laptop. Rotate only the lid/door   
about that hinge. Preserve existing plain surfaces, never add a keyboard.   
For relocation guides trace a tight polygon around the COMPLETE current target   
silhouette in Image 2; propose an integer translation [dx,dy] placing that same   
unscaled silhouette on the actual support. Do not include unrelated object or   
large background areas in the cutout. Destination is the usable platform, not   
its display/handle. Keep the fixed support unchanged; natural overhang is okay.   
Masks must include the WHOLE source, whole destination silhouette and a margin   
for contact/shadows; avoid editing unrelated background. All dimensions/diagonal   
yaw remain those in Image 2. For goals requiring rotation or changing orientation,   
use final extent outlines rather than a translation that contradicts the task.   
For stacking, preserve dimensions and ordering specified in current instruction.   
Write plan and acceptance\_checks with current coordinates if helpful. Annotations   
are proposals from initial pixels; never describe them as ground truth.   
TASK: {task}   
REFERENCE INSTRUCTION: {reference\_instruction}   
REFERENCE ANNOTATION JSON: {reference\_annotation}   
CURRENT INSTRUCTION (authoritative for named instances/arms): {instruction}   
{public\_task\_block}

The placeholder {public\_task\_block} is the same public-task block that closes the Stage-2 prompt below. It contains RoboTwin’s public task description and placeholder schema, followed by a few lines of reviewed qualitative guidance for the task.

## A.3 Stage 2: Goal Image Editing

The editing prompt for GPT Image 2.5 is a fixed template filled with the current instruction and the Stage-1 output ({objects}, {plan}, and {acceptance\_checks}). Paragraphs marked with a bracketed tag are included only for tasks whose reference annotation provides the corresponding guide. Gray italic text is explanatory and is not part of the prompt.

Edit Image 1 into ONE successful terminal HEAD-CAMERA observation of this exact simulated robot scene.   
IMAGE ROLES:   
1. Initial head view: authoritative camera, framing, geometry, appearance and scene layout.   
2. Robot appearance sheet: robot construction only; its poses/background are not task targets.   
3. REGISTERED geometry overlay: exactly the same initial head image at exactly the same pixel coordinates and scale as   
Image 1. Magenta boxes identify movable objects; cyan boxes/lines anchor fixed objects or base edges. The boxes are initial   
identity bounds, NOT requested object sizes or final locations. All annotations are explanatory only: output NO text,   
boxes, panels or arrows.   
INSTRUCTION: {instruction}   
GROUNDED INITIAL OBJECTS (pixel boxes use Image 1’s 288x224 coordinates):   
{objects}   
(expands to one line per object: “- ID: description; initial box [x0, y0, x1, y1].”)   
REQUIRED FINAL STATE:   
{plan}   
CONSERVATION RULES:   
Apply the smallest physical change that satisfies the final state. Preserve camera extrinsics/intrinsics, table,   
background, light, object identities/materials/scale, every distractor, and fixed destination geometry. Translate/rotate   
only explicitly movable objects. Preserve exact object count; remove moved objects from their old locations. Reconstruct   
only newly exposed surfaces, with the existing simulator style. Do not canonicalize, beautify, replace or enlarge objects   
to improve visibility. A broad/awkward noun in the instruction refers to the specifically identified visible object above.   
For articulated tasks, preserve the stationary body’s original projected corners and yaw. Rotate only the moving part   
around its existing hinge; never swivel the whole device toward the viewer. For a press, retain the device’s position and   
orientation and show the requested arm’s gripper contacting the actual control specified by the current instruction. A   
control may face sideways; do not add a top button or rotate the device to expose it. For a hold/lift, show a real grasp   
with clearance above the support; for place/stack, show supported objects at the destination; release only grasps no longer   
required by the public complete task, and retain required support-holding grasps. Preserve the two-arm robot’s   
construction, connected kinematic chains and plausible reach. An idle arm can stay outside the head frame. Do not invent   
extra hands/arms or use a floating disconnected gripper.   
Before rendering, mentally match each ID to its original instance, distinguish movable and fixed parts, and check count,   
final relation, support/contact and stationary edges. Occlusion is natural: do not move fixed objects or change the camera   
just to reveal them. Output only the completed scene, at the exact aspect ratio and padding of Image 1. No goal image or   
future state is provided.   
GEOMETRY PRIORITY: Image 1 is already the requested 1152x896 output size. Edit it in place: preserve unedited pixels and   
their exact coordinates. Image 3 aligns pixel-for-pixel with it. Do not reframe, regenerate or regularize the scene. Object   
boxes above use native coordinates: multiply by FOUR for input/output pixels. Preserve actual object dimensions under the   
fixed camera and perspective. Do not shrink targets to fit destinations or shrink whole stacks. Move targets with their   
original shape/volume. Preserve the named arm’s mechanical identity and its connection to its original fixed mount. An arm   
may reach across the table; target position and the edge where a cropped gripper appears do not define arm identity. Do not

mirror the robot or exchange arms.   
[final-extent guides] Image 3 also includes ORANGE final-extent guides: a manually proposed geometric layout derived from   
initial object dimensions and the instruction, NOT an observed future image. Match the original object’s size and shape   
within this guide at its destination. These outlines are approximate projections, never additional objects. Remove all   
guide lines/text from the output. Do not treat an orange guide as an initial object or duplicate the target.   
[relocation guides] IMPORTANT IMAGE-3 SIZE / POSE GUIDE: the orange-outlined target at the destination is an exact unscaled   
cutout of the initial object, translated to propose the final placement. It is synthetic guidance from initial pixels, NOT   
a future observation or an additional object. This guide makes the required size and yaw explicit: match that FULL-SIZE   
silhouette and pose, keep the destination mat fixed underneath, and remove the old source object. Generate a coherent final   
scene with natural contact/shadows, without orange outlines or text. Do not shrink the target to fit inside the mat.   
FINAL IMAGE CHECKS -- every item must be visibly satisfied:   
{acceptance\_checks}   
(expands to one bullet per check)   
[edit masks] The attached alpha mask restricts editing to transparent regions containing the old target and proposed   
destination. Keep opaque regions exactly unchanged. Render the completed object placement inside the editable regions; do   
not move fixed supports or resize targets to fit the mask. No robot arm needs to enter a completed released-placement scene.   
PUBLIC OFFICIAL COMPLETE TASK -- authoritative over the historical reference example:   
Task: {task}   
Full description: {robotwin\_full\_description}   
Official placeholder schema: {robotwin\_schema}   
Preserve the exact current official instruction, including explicit release, put-down and negation clauses. Only after   
recorded compatibility review may the complete task fill in omitted steps. An explicit contradictory instruction is not an   
omission and must not be silently overridden. Unresolved language conflicts require review before generation. Left/right   
arm names refer to the robot’s mechanical arms and their fixed mounts, not the target’s horizontal image position or the   
edge where a cropped gripper appears. Preserve each arm’s connection to its own mount, including cross-table reaches; do   
not mirror the robot or exchange arm identities. The appearance sheet illustrates construction only. Official {A}/{B}   
placeholders and historical annotation object IDs A/B are separate namespaces: rebuild their role mapping from current   
initial RGB and instruction, never equate letters. Preserve current object identity, size and background. The public   
complete task takes precedence over any reference plan, fixed flag, mask or acceptance check that would prevent required   
motion, lifting, holding, transfer or placement. Do not keep a support stationary when this task requires lifting or moving   
that support. Do not depict an intermediate placement as the completed task. Use no expert future RGB, trajectory, future   
pose, simulator state or hidden instance information.   
REVIEWED QUALITATIVE GOAL QUALITY GUIDANCE:   
This is reviewed image guidance derived from existing public task semantics; it is not a new official requirement or an   
assertion that a generated image proves physical success. It must remain compatible with the exact current instruction.   
Success may occur earlier in a trajectory; do not replace explicit current-language actions with a guessed terminal rule.   
{task\_guidance}   
(expands to one bullet per guidance line)

Example. For blocks\_ranking\_size with the instruction “Align large block, medium block, and small block by size from left to right at the center.”, the scene-specific parts of the editing prompt are:

GROUNDED INITIAL OBJECTS (pixel boxes use Image 1’s 288x224 coordinates):   
- L: large purple block; initial box [52, 78, 90, 116].   
- M: medium light green block; initial box [230, 92, 265, 131].   
- S: small magenta block; initial box [106, 105, 126, 130].   
REQUIRED FINAL STATE:   
Move the three blocks to a central tabletop row and align them by physical size from left to right: L, the large purple   
block, on the left; M, the medium light green block, in the middle; S, the small magenta block, on the right. Keep the   
blocks upright, preserve their current colors and dimensions, leave small gaps between them, do not stack them, and release   
all blocks after placement. The completed row should be centered on the table around the middle of the image rather than   
leaving M far to the right or S between L and M.   
FINAL IMAGE CHECKS -- every item must be visibly satisfied:   
- Exactly the three visible blocks are arranged at the table center in one horizontal row.   
- Left-to-right order is largest to smallest: large purple L, medium light green M, then small magenta S.   
- All blocks remain upright, unstacked, non-overlapping, and released on the tabletop.   
- No unrelated scene elements are moved or newly introduced.   
Full description: there are three blocks on the table, the color of the blocks is random, move the blocks to the center of   
the table, and arrange them from largest to smallest, from left to right   
REVIEWED QUALITATIVE GOAL QUALITY GUIDANCE:   
- Identify large, medium and small using their true current physical dimensions, not color, Place the same three blocks in   
a central tabletop row with largest left, medium middle and smallest right, preserving all dimensions and colors.   
- Small-then-medium-then-large wording specifies handling order; it does not reverse the final left-to-right size order   
given by full\_description. Follow named per-block arms and depict completed released placement rather than stacking or   
holding a block.

Table 15 First-pass review outcomes for the 7,032 newly generated evaluation goal images.
<table><tr><td>Verdict</td><td>Images</td><td>Share</td></tr><tr><td>Pass</td><td>6,380</td><td>90.7%</td></tr><tr><td>Fail</td><td>542</td><td>7.7%</td></tr><tr><td>Uncertain</td><td>110</td><td>1.6%</td></tr><tr><td>Total</td><td>7,032</td><td>100.0%</td></tr></table>

## A.4 Goal Review and Repair

ChatGPT Astra High reviews every newly generated evaluation goal image. For each case it sees the initial and generated head images at native resolution, the instruction, the public task description, and the Stage-1 grounding. It treats the grounding as fallible and locates objects independently in the initial image, then returns pass, fail, or uncertain with case-specific evidence. The review rubric is:

Check target/destination identity, exact instructed goal relation/arm when visually distinguishable, source removal/object   
counts, appearance/size, fixed background/destination/camera, real grasp/contact/support, articulated hinge, and coherent   
robot. Public full task only fills omissions compatibly, never overrides explicit instruction. Natural occlusion is   
legitimate. Do not demand an entire trajectory in a single goal frame, invent new acceptance rules, equate image-side with   
mechanical arm, or mark all cropped mounts uncertain. Pass means visible state is consistent with no clear defect; fail   
means concrete visible contradiction: uncertain identifies concrete unresolved evidence.   
The audit concerns whether the visible generated goal is compatible with the exact task under the fixed head camera; it   
does NOT establish hidden physical state or execution history. If the requested endpoint is naturally outside the view or   
inside a clipped container, all relevant visible changes and fixed geometry are consistent, and no visible contradiction   
exists, use pass with an explicit visibility limitation. Retain uncertain when there is a specific visible ambiguity (which   
physical object, actual contact, orientation, scale) rather than merely unverifiable hidden physics.

Repair. Each non-passing verdict (fail or uncertain) comes with repair feedback and a repair stage. If the grounding misidentified the source, receiver, or contact region, the reviewer supplies a corrected target from the initial image and Stage 1 is rerun with this feedback. Otherwise, Stage 2 is rerun with the feedback inserted before the task block, and the grounding is kept. Every replacement image is reviewed again. Regenerated images are judged under a looser criterion than first-pass images: the reviewer accepts reasonable perspective variation and mild size diferences that can be explained by projection, but still rejects wrong targets or receivers, duplicated or missing objects, wrong task relations, severe deformation, invented fixtures, and disconnected or extra arms.

Review statistics. Of the 7,032 evaluation goal images newly generated with this pipeline, 6,380 (90.7%) passed the first review, 542 (7.7%) failed, and 110 (1.6%) were uncertain (Table 15). After repair, all 7,032 passed review; repaired images were judged under the looser criterion described above. The review checks visual and instructional consistency only; it does not verify physical feasibility or task success in simulation. The remaining 2,968 of the 10,000 evaluation goals are earlier images that a separate audit approved for reuse.

## B Real-World Zero-Shot Task Suite

Table 16 specifies the 20 real-world tasks and their scoring rubrics. Each checkpoint is evaluated on 10 trials per task (200 rollouts), without target-task demonstrations or parameter updates. Tasks are specified by goal images with the shared instruction “Complete the task shown in the goal image.”

Scoring conventions. Each task has three or four scored milestones. Parenthesized values are maximum stage scores and sum to 10; intermediate scores retain the partial credit recorded during evaluation. Binary success is assessed separately using the task outcome in the last column, and need not imply a score of 10. For example, the records deduct one point for using a diferent arm from the goal image (E1), and one or two points for a tipped cup or a lid grasp, respectively (H3), despite marking these trials successful. Both metrics are averaged over trials and then equally over tasks.

Table 16 Real-world zero-shot tasks and scoring rubrics. Milestones are listed in the order recorded in the evaluation sheets; arrows do not impose a universal prerequisite rule. Parentheses give maximum points, not cumulative scores. E1–E10 are easy tasks; H1–H10 are hard tasks.
<table><tr><td>ID</td><td>Task</td><td>Family</td><td colspan="3">Scored milestones (maximum points)</td><td>Task success outcome</td></tr><tr><td colspan="7">Easy tasks (10 tasks; 100 trials)</td></tr><tr><td>E1</td><td>Lift a block</td><td>B</td><td></td><td>Approach (3) → grasp (4) → lift (3)</td><td></td><td>Target block held aloft.</td></tr><tr><td>E2</td><td>Compress a sponge</td><td>A</td><td></td><td>Approach (2) → contact (3) → press (5)</td><td></td><td>Sponge fully compressed.</td></tr><tr><td>E3</td><td>Open a hardcover book</td><td>C</td><td>→ open (3)</td><td>Approach (2) → touch (2) → grasp (3)</td><td></td><td>Front cover opened.</td></tr><tr><td>E4</td><td>Ring a desk bell</td><td>A</td><td></td><td></td><td>Approach (3) → contact (3) → press (4)</td><td>Bell actuated to ring.</td></tr><tr><td>E5</td><td>Press a click pen</td><td>A</td><td></td><td></td><td>Approach (3) → contact (3) → press (4)</td><td>Pen&#x27;s push button depressed.</td></tr><tr><td>E6</td><td>Open a drawer</td><td>C</td><td></td><td></td><td>Contact (3) → grasp (4) → pull open (3)</td><td>Drawer pulled open.</td></tr><tr><td>E7</td><td>Push an object away</td><td>A</td><td>(4)</td><td></td><td>Contact (3) → manipulate (3) → complete</td><td>Object displaced from its initial position.</td></tr><tr><td>E8</td><td>Lift and shake a bottle</td><td>B</td><td></td><td>Locate (3) → grasp (3) → lift (4)</td><td></td><td>Bottle lifted and shaken.</td></tr><tr><td>E9</td><td>Arrange blocks</td><td>B</td><td></td><td></td><td>Grasp (2) → place (3) → complete (5)</td><td>Blocks in the target arrangement.</td></tr><tr><td>E10</td><td>Place a block on a tray</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Block on the tray.</td></tr><tr><td colspan="7">Hard tasks (10 tasks; 100 trials)</td></tr><tr><td>H1</td><td>Stack blocks</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>One block stacked on another.</td></tr><tr><td>H2</td><td>Place a can beside a book</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Can beside the book.</td></tr><tr><td>H3</td><td>Place a cup on a coaster</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Cup on the coaster.</td></tr><tr><td>H4</td><td>Place an egg in a bowl</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Egg inside the bowl.</td></tr><tr><td>H5</td><td>Place an object on a book stack</td><td>D</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Object on top of the book stack.</td></tr><tr><td>H6</td><td>Place a glue stick in a pen holder</td><td>B</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>Glue stick inside the pen holder.</td></tr><tr><td>H7</td><td>Stack two bowls</td><td>E</td><td></td><td></td><td>Grasp (3) → place (3) → complete (4)</td><td>One bowl stacked on the other.</td></tr><tr><td>H8</td><td>Lift two bottles bimanually</td><td>B</td><td>Grasp left (2) → lift left (3)</td><td>→ grasp right (2) → lift right (3)</td><td></td><td>Both bottles lifted, one per gripper.</td></tr><tr><td>H9</td><td>Open a drawer and place an object inside</td><td>C</td><td>→ grasp object (2) → place (2)</td><td>Grasp drawer (3) → open drawer (3)</td><td></td><td>Drawer open with the object inside.</td></tr><tr><td>H10</td><td>Insert a plug into a power strip</td><td>E</td><td></td><td></td><td></td><td>Approach plug (2) → grasp (3) → insert (5) Plug inserted into the target socket.</td></tr></table>

Families: A, contact/state changes (4); B, grasping/transport (5); C, articulated opening/closing (3); D, spatial placement/align ment (6); E, stacking/insertion (2). Family assignments and milestone labels follow the task records; overlapping skills are represented by one assigned family.