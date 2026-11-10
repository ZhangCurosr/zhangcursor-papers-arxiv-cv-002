# SWAM: Scaling World Action Models with Cross-Modal Alignment

Yu Zhou<sup>1,2,3,\*,‡</sup>, Xinyuan Xia<sup>1,2,3,\*</sup>, Xuankun Yang<sup>1,3,\*</sup>, Yinghao Yang<sup>4,2,3,\*</sup>, Wendi Chen<sup>1,2,3</sup>

Bowen Liu<sup>1,2,3,‡</sup>, Jianhan Ma<sup>3</sup>, Weicheng Duan<sup>5,3</sup>, Jun Lv<sup>3</sup>, Cewu Lu<sup>1,2,3</sup>, Yanwei Li<sup>1,†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Noematrix Ltd.

<sup>4</sup>Westlake University <sup>5</sup>Tongji University <sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

Dataset Code Webpage

Abstract World action models jointly predict video and actions conditioned on language. Existing studies focus primarily on architectures and training mechanisms, typically treating language–video alignment acquired through video pretraining as an established foundation. However, introducing action learning may impair this alignment. Principled guidance on data and supervision design for maintaining language–video alignment during action learning remains limited. To investigate this, we curate a pool of approximately 47,000 hours of general videos, egocentric videos, UMI demonstrations, and robot demonstrations. Building on this, we systematically study how language–video supervision, language–video–action supervision, and the joint use of these two forms afect WAM performance. In particular, we examine how language annotation quality afects pretrained language–video capabilities, and how action supervision from UMI and robot demonstrations extends these capabilities to language–video–action modeling. To better scale the model, we further conduct detailed studies on joint training recipe and model architecture. Under a fixed training budget, mixing egocentric videos and UMI demonstrations improves control generalization. Model architecture afects how efectively these experiences are used, while scaling pretraining under the selected configuration brings further gains. The resulting recipe, post-trained on RoboTwin Clean demonstrations only, achieves success rates of 88.10% and 61.56% under Easy (Clean-to-Clean) and Hard (Clean-to-Random) conditions, respectively. Our model achieves 40.00% SR across three real-world tasks, compared with 23.33% for the strongest baseline, indicating better generalization under unseen conditions. We release the curated data, training code, and model weights.

Multisource Data  
![](images/80cf32dbfdd07d47c7dc70c6dde813234501a0d48acdd1255e91a861e4278b90.jpg)

![](images/06c7d799dfc9b503e0d8970df0346ad9f3eca2ac1e91e9aaf890a7b41db3686d.jpg)

![](images/85dd6da7fae40f9e297552dcec4179ebc017bda168ba60f4ce4680c6450244b6.jpg)

![](images/0bd0ae3dba011829b395bbc010019060b42db6c0ef93a20fb256a708d821dd58.jpg)

![](images/fe1bdb2cc48e4c614887c1decc02d8536d3e303047b8e0c1ca0508e0964b76c4.jpg)  
Robot Manipulation Generalization

![](images/5df63bcf3b4e84a0e188be294a3f5d530ab987b67c218e5d9dc8833eb01832ea.jpg)

![](images/358421d8f08f93227b91d4ba34b0e8de422d00bb436bfc395e2e1850fa9b9b84.jpg)  
Real-world Scaling

![](images/13bdca6d17a5b7a626ba57d61b7d5936a7e5defbc4550ad8cad5fb33b5ce47a1.jpg)

![](images/91c2f67967067592d29c907933c496a921a96947ac40db0565000b870c0013b0.jpg)

![](images/1b4593e3b6c204344f0ab71134d7e402bd20b8ad67ec73499d7a9ba059b94402.jpg)  
Figure 1. Overview of SWAM. We study how multisource supervision preserves the language–video capabilities acquired during video pretraining and extends them to robot control. We curate a multisource data pool of approximately 47,000 hours to support this study. Experiments show that additional pretraining further improves real-world performance, and that the resulting recipe generalizes to unseen scene conditions in RoboTwin and on real robots.

## Contents

1 Introduction 3   
2 Related Work 4   
3 Method 5   
3.1 World Action Modeling 5   
3.1.1 Problem Formulation 5   
3.1.2 Model Architecture . 6   
3.1.3 Training Objectives 7   
3.2 Language–Video Alignment . 8   
3.2.1 Video Sources and Curation . 8   
3.2.2 Language Annotation 9   
3.3 Language–Video–Action Alignment 10   
4 Experiments 11   
4.1 Experimental Setup . . 11   
4.2 How to establish effective cross-modal alignment? . 11   
4.2.1 Language–Video Alignment . 12   
Rich and aligned language supervision improves generalization . 12   
4.2.2 Language–Video–Action Alignment . 13   
Human manipulation experience provides action supervision . 13   
4.3 How to integrate multisource experience? . 13   
Transitional pretraining on General videos provides a stronger starting point 13   
Pretraining exposure to individual data sources improves WAM performance 14   
Equal Ego–UMI sampling gives the best generalization . 14   
Architecture helps preserve pretrained video priors 14   
4.4 Does scaling pretraining improve control generalization? . 15   
Scaling pretraining improves WAM performance 15   
4.5 Does our model perform well in unseen environments? . 16   
Simulation results . 16   
Real-world results . 16   
Real-world visualizations . 16   
5 Discussion and Conclusion 17   
6 Appendix 23

## 1. Introduction

Pretrained video models learn associations between language descriptions, visual content, and visual dynamics, acquiring visual and dynamics priors across scenes, objects, and interactions [1, 2, 3, 4]. These experiences provide a foundation for robots to understand manipulation goals and predict environmental changes [5, 6]. World action models (WAMs) jointly predict video and actions conditioned on language, connecting visual changes to robot actions and providing a route from video pretraining to robot control [7, 8]. This paper studies how to better use these priors to extend language-conditioned video modeling into generalizable action capabilities.

When we introduce action prediction, the language–video capabilities acquired in video pretraining should be preserved. This requires further alignment between language, video, and action. It bridges the gap between language-described targets, visual observation, and executed actions. Existing work on WAMs largely focused on model architectures and training mechanisms [9, 10]. Methods on video pretraining often treat the language–video alignment as an established foundation [7, 8]. However, introducing action learning raises a basic question: how to preserve established language–video prior and further solve language–video–action misalignment? The roles of different experiences and forms of supervision in this process, particularly for control generalization, remain unclear [9, 10, 11, 12].

![](images/c590dd1e3df64f67561fe75f69c2c2c1f2f458c8d8ed283a22b3f6c45bac7495.jpg)

![](images/fac42d167a1992119d00c083f2f73cfdfcd56f89e46bc6d83bd9af0584fc4e98.jpg)  
Figure 2. Multisource data and model overview. (a) Composition of the 47k-hour pretraining data pool. (b) Our world action model architecture, with video and action branches connected through shared masked attention.

To study these questions, we first construct a multisource data pool with approximately 47,000 hours, comprising approximately 11,000 hours of general videos, 11,000 hours of egocentric videos, 15,000 hours of in-house UMI demonstrations, and 10,000 hours of robot demonstrations, as shown in Figure 2(a). These sources provide different forms of cross-modal supervision. General and egocentric videos primarily provide language–video supervision, covering general visual dynamics and human interactions, respectively. UMI demonstrations record video and action trajectories of humans manipulating objects with handheld grippers, while robot demonstrations provide observations, states, and actions during robot task execution. Both include language descriptions corresponding to the manipulation process, supporting language–video–action learning in addition to language–video supervision. UMI demonstrations provide scalable human manipulation experience, whereas robot demonstrations provide direct but more limited robot control experience [13, 14, 15]. The sources differ in scene and task coverage and in the completeness of language and action supervision. Multisource pretraining therefore requires combining these experiences appropriately: language–video supervision preserves video pretraining capabilities, while human and robot action supervision establishes video–action correspondences [16, 17, 18].

Building on the constructed multisource experience, we conduct systematic studies at three levels: language–video alignment, language–video–action alignment, and joint learning. Specifically, at the language–video level, we examine how to preserve the visual prior in pretrained video generation with the design of language supervision. At the language–video–action level, we investigate how to utilize action supervision from UMI and robot demonstrations to help action prediction and robot control. In joint training, we study the data recipe and model architecture for better cross-modality modeling. The data recipe determines the coverage of the visual and interaction experience in tasks, scenes, and the proportions of different experiences. The model architecture answers how to conduct cross-modal supervision through modality interactions and language conditioning pathways, as shown in Figure 2(b). We further examine the benefits of scaling training with the resulting data and supervision recipe.

Our study shows that preserving language–video capabilities and extending language–video–action capabilities jointly support control generalization in WAMs. Detailed manipulation descriptions and environmental cues improve language–video supervision, which maintains language–video alignment and thereby improves control generalization. And UMI demonstrations improve robot control in simulation and real scenes, proving the value of human manipulation experience with action supervision for robot learning. In joint learning, the benefits of multisource experience depend both on data and model supervision recipe. Experiments show that increasing training investment under the selected data and model recipe further improves control generalization. These findings provide empirical guidance for data collection and organization, supervision design, and joint training in WAMs. The pretraining recipe derived from these findings achieves success rates of 88.10% and 61.56% under the RoboTwin Easy (Clean-to-Clean) and Hard (Clean-to-Random) settings, respectively. Ours achieves 40.00% SR under the Hard setting across three real-world tasks, exceeding OpenWAM’s 23.33%. This result indicates better generalization to unseen objects and backgrounds

## 2. Related Work

World Action Models. World action models aim to connect visual observations with executable robot actions [7]. Early work treated video as a transferable visual plan or dynamic representation. UniPi first generates a language-conditioned visual plan and then recovers actions with an inverse-dynamics module [19]. VPP conditions a policy on predictive representations from a video model [6]. Motus, DreamZero, and LingBot-VA place visual dynamics and action prediction within unified or interactive architectures, exploring discrete tokens, latent actions, joint flow matching, and causal video–action modeling [20, 7, 8]. Fast-WAM and InternW0-Δ further distinguish future supervision during training from action generation at inference, showing that the prediction target and the deployment interface can be separated [21, 17]. VPP2 and LingBot-VA adapt video pretraining to robot manipulation by studying how manipulation data organization, temporal and video representation modeling, and action-branch adaptation can preserve video prediction capabilities while supporting action learning [18, 8]. We build on this foundation and focus on how language–video capabilities from video pretraining can transfer to language–video–action learning and action control.

Multisource Data for Robot Learning. Robot learning increasingly draws on cross-embodiment trajectories, general videos, egocentric videos, and human manipulation demonstrations rather than a single source of robot data. Open X-Embodiment, Octo, and OpenVLA aggregate and filter multi-robot data to study shared representations, observation adaptation, and action-space alignment, showing that cross embodiment learning requires source normalization and interface alignment [22, 23, 24]. General and egocentric videos expand coverage of scenes, objects, and interaction dynamics, but usually lack action labels directly corresponding to robot execution [25, 26]. EgoMimic and EgoScale study how such human experience can be transferred through view and embodiment alignment, including settings with limited robot data [27, 28]. UMI complements these sources by using portable interfaces to record manipulation trajectories in an end-effector-like interaction format, thereby providing action supervision associated with the manipulation process [13, 14, 15]. EgoWAM, Ego4WAM, and Motus further demonstrate the complementary roles of video, action trajectories, semantic descriptions, and world-prediction targets [29, 10, 20]. The central issue in multisource learning is therefore not simply adding sources, but understanding differences in scene coverage, interaction dynamics, embodiment interfaces, and available supervision. We curate approximately 47,000 hours of multisource data. We use this pool to systematically examine how experience combinations and supervision design affect WAM pretraining.

![](images/ee4b18e0d9f06169ea63deacb717c8755785a820f2a4da6cc7897b0f21e29eda.jpg)

## 3. Method

This section presents the world action model and its multimodal supervision. We first present the prob lem formulation, model architecture, and training objectives, then describe multisource experience curation and supervision construction. Pretraining uses four types of experience: general videos, egocentric videos, UMI demonstrations, and robot demonstrations. These sources differ in scene and task distributions, data quality, manipulation diversity, and available action supervision. General and egocentric videos provide language–video supervision, while UMI and robot demonstrations additionally provide state and action signals where available and valid. In Table 1 and Figure 3, we summarize the composition, scale, and video-duration distribution of the pretraining pool after data curation and cleaning.

Table 1. Composition and scale of multisource pretraining data.
<table><tr><td>Dataset</td><td>Episodes</td><td>Frames (M)</td><td>Hours</td></tr><tr><td>General Videos</td><td></td><td></td><td>10,909.6</td></tr><tr><td>Panda-70M</td><td>2,599,835</td><td>1,089.4</td><td>10,087.4</td></tr><tr><td>OpenVid</td><td>256,839</td><td>82.5</td><td>822.1</td></tr><tr><td>Egocentric Videos</td><td></td><td></td><td>10,986.3</td></tr><tr><td>EgoSuite</td><td>294,798</td><td>928.9</td><td>8,600.6</td></tr><tr><td>Ego4D</td><td>21,302</td><td>144.7</td><td>1,339.7</td></tr><tr><td>EgoVerse</td><td>55,112</td><td>113.0</td><td>1,046.0</td></tr><tr><td>UMI Data</td><td></td><td></td><td>15,284.9</td></tr><tr><td>In-house UMI</td><td>411,754</td><td>1,650.8</td><td>15,284.9</td></tr><tr><td>Robot Data</td><td></td><td></td><td>9,836.0</td></tr><tr><td>ABC</td><td>127,057</td><td>370.2</td><td>3,428.1</td></tr><tr><td>AgiBot</td><td>155,650</td><td>286.9</td><td>2,656.6</td></tr><tr><td>InternData-A1</td><td>354,192</td><td>265.6</td><td>2,458.9</td></tr><tr><td>Galaxea</td><td>19,636</td><td>51.8</td><td>479.8</td></tr><tr><td>RoboCOIN</td><td>50,439</td><td>45.3</td><td>419.1</td></tr><tr><td>RoboMind</td><td>61,382</td><td>42.5</td><td>393.5</td></tr><tr><td>Total</td><td>4,407,996</td><td>5,071.5</td><td>47,016.7</td></tr></table>

(c)  
Figure 3. Pretraining data composition and video duration across sources. (a) Shares of total video hours. (b) Shares of total episodes. (c) Video-duration distribution within each source.

## 3.1. World Action Modeling

## 3.1.1. Problem Formulation

We consider language-conditioned robot manipulation. Given the visual observations acquired so far, a language instruction, and its current proprioceptive state, the robot predicts actions over a future interval. Let � be a prediction start point sampled from a trajectory, $K > 0$ a fixed prediction horizon, $V _ { 0 : n }$ the visual history including the current observation, � the language condition, and $S _ { n }$ the current state. The control task requires learning an action policy �� conditioned on visual history, the language instruction, and the current state:

$$
{ \cal A } _ { n : n + K } \sim \pi _ { \theta } ( \cdot \mid V _ { 0 : n } , P , S _ { n } ) ,\tag{1}
$$

where $A _ { n : n + K }$ denotes the action sequence over the target interval, and � denotes the model parameters.

Control learning primarily supervised by observation–action pairs must jointly address scene understanding, interaction dynamics, and action generation. We adopt a world action model (WAM) based on a pretrained video model, using the associations between language descriptions and visual dynamics learned during video pretraining and learning correspondences between manipulation processes and executed actions through action supervision. The model jointly predicts video changes and actions over the same interval. The video sequence $V _ { n : n + K }$ describes observable scene evolution during manipulation, while the action sequence $A _ { n : n + K }$ describes the corresponding manipulation behavior. The two sequences span the same physical time interval but may use different sampling rates.

We denote the joint predictive distribution of video and actions by $p _ { \theta } \colon$

$$
\begin{array} { r l } & { p _ { \theta } ( V _ { n : n + K } , A _ { n : n + K } \mid V _ { 0 : n } , P , S _ { n } ) } \\ & { \ = p _ { \theta } ( V _ { n : n + K } \mid V _ { 0 : n } , P , S _ { n } ) p _ { \theta } ( A _ { n : n + K } \mid V _ { 0 : n + K } , P , S _ { n } ) } \\ & { \ = p _ { \theta } ( V _ { n : n + K } \mid V _ { 0 : n } , P ) p _ { \theta } ( A _ { n : n + K } \mid V _ { 0 : n + K } , P , S _ { n } ) . } \end{array}\tag{2}
$$

The final equality follows from our modeling choice that the video branch is not conditioned on the proprioceptive state. The first term learns language-conditioned video prediction, corresponding to language–video learning. The second term predicts actions from language, video information, and the current state. Together, the two terms constitute joint language–video–action learning.

Starting from a pretrained video model, we train a unified WAM end to end with a joint prediction objective. The action branch uses visual dynamics and language conditioning information supplied by the video branch. Training retains the video prediction objective while introducing action supervision, aiming to continue using existing language–video modeling capabilities and to learn correspondences between manipulation processes in video and executed actions.

Multisource experience supplies different parts of the supervision for joint prediction. General and egocentric videos provide language–video correspondences for learning language-conditioned visual dynamics. UMI and robot demonstrations additionally provide actions temporally associated with video, enabling action prediction alongside video learning. All samples participate in video learning. Samples with valid action annotations additionally participate in action learning. Missing or invalid action targets contribute no action supervision. Experiences with different supervision coverage can therefore participate in unified WAM pretraining.

## 3.1.2. Model Architecture

Given visual observations $V = \{ \nu _ { i } \} _ { i = 1 } ^ { T } ,$ , a language instruction $P ,$ and available states $S ,$ we adopt a WAM that jointly predicts video and actions. Whereas $n : n + K$ above denotes a trajectory-level prediction interval, we use � below to index temporally aligned video–action chunks. Each chunk contains 2 video latent frames and 48 action steps. For chunk $t ,$ let $V _ { t } ~ \in ~ \mathbb { R } ^ { T _ { \nu } \times H \times W \times 3 }$ denote the corresponding video, $S _ { t } \in \mathbb { R } ^ { D _ { s } }$ the available state at the chunk start, and $A _ { t } \in \mathbb { R } ^ { H _ { a } \times D _ { a } }$ the normalized target action sequence. Here, $T _ { \nu }$ is the number of video frames, � and � are frame height and width, $D _ { s }$ is the state dimension, and $H _ { a }$ and $D _ { a }$ are the number of action steps and the action representation dimension, respectively. A VAE encodes video into latents $x _ { t } .$ Separate encoders map actions and states to tokens. A text encoder maps the language instruction to condition $p .$

We illustrate the model architecture and the attention masks used during training and inference in Figure 4. The model uses a MoT architecture [30], with the video branch initialized from Wan [2] and the action branch randomly initialized. The action DiT branch comprises action and state encoders, transformer blocks, and an action decoder. Video and action maintain separate residual streams, with joint attention passing information unidirectionally from video to action. The video token stream passes through the video branch and causally attends to current and past video tokens. The action token stream passes through the action branch and attends to video tokens from past and current chunks, as well as state and action tokens within the current chunk. This unidirectional interaction allows samples with video supervision alone to continue training the video branch without directly updating the action branch through the video objective, supporting the use of videos without action supervision.

The video branch receives encoded language conditions through cross-attention. The action branch uses language information only through attention interactions with the video branch and has no direct text cross-attention. This conditioning pathway aims to reuse the language–video correspondences established during video pretraining, allowing action prediction to use semantic information associated with the manipulation process. The video branch predicts flow velocities in video latent space, while the action branch decodes its output tokens into flow velocities in action space.

![](images/0a85ad8ae3cb03939eba22c83ed93c800dbb7a5d0f7b30fe812f77a74567d3e4.jpg)

![](images/9e4cde097b162171c8118038a62500f64f82cb66e9a03c32bc92f8b23398691f.jpg)  
(b) Training Attention Mask

![](images/62e4a3652bcc447b57fb9dcdee5dc86813445f7220c05d20911b2e52db1a6207.jpg)  
(a) Model Architecture  
(c) Inference Attention Mask  
Figure 4. World action model architecture and attention masks. (a) The video and action DiT branches are connected through shared masked attention, with text cross-attention in the video branch only. Attention masks for (b) training with clean video history under teacher forcing and (c) generating video and actions chunk by chunk at inference.

## 3.1.3. Training Objectives

For chunk $t , x _ { t }$ and $A _ { t }$ denote the clean video latents and action targets, respectively. Let � ∈ [0, 1] denote flow matching time. We sample independent standard Gaussian noise $\epsilon _ { \upsilon }$ and $\epsilon _ { a }$ with the same shapes as their respective targets and construct noisy targets:

$$
\begin{array} { r } { x _ { t } ^ { \tau } = ( 1 - \tau ) x _ { t } + \tau \epsilon _ { \nu } , } \\ { A _ { t } ^ { \tau } = ( 1 - \tau ) A _ { t } + \tau \epsilon _ { a } . } \end{array}\tag{3}
$$

The action encoder maps noisy actions $A _ { t } ^ { \tau }$ to action tokens. Video and actions within a chunk share $\tau ,$ sampled independently across chunks. Training uses teacher forcing, with past clean video latents $h _ { t }$ as context. Let $u _ { \theta } ^ { \nu }$ and $u _ { \theta } ^ { a }$ denote the video and action velocity predictors, whose outputs have the same shapes as $x _ { t }$ and $A _ { t } ,$ , respectively. The action predictor $u _ { \theta } ^ { a }$ denotes the complete action DiT branch, including action and state encoding, transformer processing, and action decoding. The two flow matching losses [31] are:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { v i d e o } } = \mathbb { E } \left[ \left\| u _ { \theta } ^ { \nu } ( x _ { t } ^ { \tau } , h _ { t } , p , \tau ) - ( \epsilon _ { \nu } - x _ { t } ) \right\| _ { 2 } ^ { 2 } \right] , } \\ & { \mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } \left[ \left\| u _ { \theta } ^ { a } ( A _ { t } ^ { \tau } , x _ { t } ^ { \tau } , h _ { t } , S _ { t } , p , \tau ) - ( \epsilon _ { a } - A _ { t } ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{4}
$$

Here, � denotes model parameters, and the expectations are taken over training samples, noise, and flow matching times. The action output depends on the current video $ { \boldsymbol { { x } } } _ { t } ^ { \tau }$ and language condition $p$ through the video branch. The two losses are combined with equal weights:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { a c t i o n } } . } \end{array}\tag{5}
$$

All samples contribute to the video objective. The action objective is computed only for samples with valid action annotations.

## 3.2. Language–Video Alignment

## 3.2.1. Video Sources and Curation

In Figure 5, we summarize the curation process used to turn the four experience sources into trainingready video–language examples. We perform quality checks, standardize the video records, segment long videos into subtasks where applicable, and generate multi-level captions before mixing the resulting data for pretraining.

![](images/ace882b5ad4492a7a9ac2e25209ac1e3c3862a9661a71ccbb6803a7829bbf46c.jpg)  
Figure 5. Multisource training data and curation pipeline. The upper panel shows representative examples from four data sources. The lower panel illustrates visual checks, action checks, format conversion, subtask segmentation, and caption generation. Captions include action descriptions at three levels of detail, along with scene and lighting information.

General videos. We use OpenVid [32] and Panda70M [33], totaling 10,909.6 hours. These open-source video generation datasets contain videos and text descriptions covering diverse scenes, objects, and visual dynamics. We retain their videos and original descriptions to provide language–video experience beyond manipulation demonstrations.

Egocentric videos. We use Ego4D [25], EgoVerse [26], and EgoSuite [34], totaling 371,212 trajectories and 10,986.3 hours. Ego4D primarily records daily activities across real-life settings. EgoVerse covers more diverse scenes, adding egocentric visual experience across environments. EgoSuite is currently the largest open-source egocentric manipulation video dataset. We retain the videos and language descriptions from these sources.

Data preparation. We package all sources in LeRobot 2.1 format. We retain only samples whose videos, language descriptions, and metadata can be linked. We exclude samples with unassociated video, language, or metadata, videos with encoding or decoding failures, and known low-quality samples.

Video records retain their corresponding language descriptions and metadata. The inventory counts the recorded episodes and videos at their source durations.

Visual content filtering. For general video datasets such as OpenVid and Panda70M, we use original annotations and VLM-based content review to remove animations, static scenes, television programs, and other unsuitable content. For egocentric videos, we use a VLM to assess motion blur, brightness, occlusion, and whether a clip contains meaningful actions, reducing the impact of low-quality imagery and uninformative behavior on training. The UMI and robot demonstrations described below use similar filtering rules.

Video model adaptation on general videos. Our WAM modifies Wan’s video modeling design to support robot control, which may disrupt pretrained visual and dynamics priors. To ease this adaptation, we first train the video branch on general videos, whose distribution is close to that of Wan’s pretraining data. During this stage, we use the same input frame count, spatial resolution, and causal attention scheme as subsequent WAM training. This stage aims to preserve these priors during adaptation, before introducing embodied data.

## 3.2.2. Language Annotation

Language annotations describe manipulated objects and executed actions and may additionally include interaction relationships, manipulation processes, and environmental information. For WAM pretraining, what semantic information helps translate video experience into control capabilities? Can these annotations help preserve language–video alignment and improve downstream generalization? We organize language supervision along three dimensions: the level of detail in manipulation descriptions, environmental information, and temporal granularity.

Level of detail. Task descriptions cover executed actions, interaction objects, and their attributes during manipulation. We define three levels of descriptive detail, Brief, Medium, and Detailed, with requirements and examples in Table 2. These levels define controlled language-annotation variants for evaluating how description detail affects the transfer of pretrained video capabilities to robot control.

Table 2. Language annotation requirements and example descriptions at three levels of detail. All examples describe the same manipulation segment.
<table><tr><td>Detail level</td><td>Annotation requirements</td><td>Example language annotation</td></tr><tr><td>Brief</td><td>Summarize the main action and core ob- Pick up the snack. ject with a verb-object phrase.</td><td></td></tr><tr><td>Medium</td><td>tence, adding object attributes, spatial re- lations, or visible outcomes.</td><td>Summarize the manipulation in one sen- Pick up the snack in the light-blue package.</td></tr><tr><td>Detailed</td><td>object interactions, and necessary adjust- ble, while the left hand does not participate. ments.</td><td>Describe the manipulation chronologi- The right hand first touches and grasps the snack cally, including discernible hand actions, in the light-blue package, then lifts it from the ta-</td></tr></table>

The three descriptions convey the same manipulation. The Brief level specifies the main action and object, the Medium level adds the object’s appearance, and the Detailed level further specifies hand involvement and action order.

Scene information. We use Qwen 3.8 Max [35] and Seed 2.1 Pro [36] for subtask segmentation and caption annotation. Scene annotations supplement task descriptions with the working environment, spatial layout, and initial state. Lighting annotations describe brightness, exposure, illumination uniformity, and shadows. For example, the scene annotation for one clip is: “The manipulation takes place on a table covered with a black-and-white striped tablecloth, with various packaged snacks scattered across the surface.” Its lighting annotation is: “Indoor lighting is bright and uniform, with no obvious shadows or overexposed regions. All snacks and the manipulation area are clearly visible.”

Temporal granularity. Episode descriptions cover an entire trajectory, while Subtask descriptions correspond to local manipulation segments with start and end times. A VLM segments continuous video into subtasks and outputs each segment’s timestamps and description. Descriptive detail and temporal granularity are independent annotation dimensions: the same subtask can have descriptions at different detail levels, and an entire trajectory can also have a brief or detailed description. During training, the Episode setting uses the description of the full trajectory, while the Subtask setting uses the annotation of the corresponding time interval as the language condition.

## 3.3. Language–Video–Action Alignment

UMI and robot demonstrations follow the video curation and filtering procedures in Section 3.2.1 and use the language annotations described in Section 3.2.2. Building on these language–video correspondences, we align video segments with state and action records to construct language–video–action supervision. We describe the collection and curation of these demonstrations below.

UMI demonstrations. We use 411,754 in-house UMI trajectories totaling 15,284.9 hours. We collect human manipulation demonstrations using RoboPocket [15]. The system integrates a smartphone with a handheld adaptive gripper into a portable collection device. Operators interact with objects by moving, opening, and closing the gripper. Its structure and interaction mechanism resemble robot end effectors, reducing interface differences between human demonstrations and robot execution. Our collection tasks include mouse placement, block sorting, seasoning pouring, towel folding, bimanual snack bagging, and other everyday manipulation tasks. They involve pick-and-place, long-horizon manipulation, large wrist rotations, deformable object manipulation, and bimanual coordination. Collection environments include indoor and outdoor scenes with different lighting and background textures.

During collection, head and bilateral wrist views record the overall manipulation scene and local interaction details, respectively. Smartphone visual–inertial tracking provides action trajectories, while external encoders record gripper opening and closing. A shared coordinate system and time reference establish spatial and temporal correspondences among the videos of both hands, motion trajectories, and gripper states. The collection procedure includes equipment checks and synchronization verification before recording, tracking stability and kinematic feasibility checks during recording, and video and AR trajectory playback afterward. These checks identify anomalies such as localization drift and trajectory jumps and verify task completion.

Robot demonstrations. We curate data from AgiBot World Alpha/Beta/2026 [37], RoboCOIN [38], RoboMIND [39], InternData-A1 [40], Galaxea Open-World [41], and ABC-130K [42]. These sources span different robot embodiments, camera configurations, and manipulation environments. We retain only demonstration subsets collected with bimanual robots that include head and bilateral wrist views and meet our quality requirements, maintaining consistent data interfaces and observation semantics. We exclude subsets with missing views, inconsistent view semantics, insufficient image resolution, overly simple tasks, single-arm robot embodiments, or other issues. After applying the configured episode exclusion lists, we retain 768,356 trajectories totaling 9,836.0 hours.

State–action representation. UMI and robot demonstrations use a shared 16-dimensional bimanual end-effector representation. Each hand has a three-dimensional position, a four-dimensional rotation quaternion in wxyz order, and a one-dimensional gripper value. State and action records are associated with their corresponding video segments. Each source is normalized using its own q1 and q99 statistics.

Trajectory quality. For our in-house UMI demonstrations, we construct action-supervision masks from the state and action validity annotations provided by the collection pipeline. Videos participate in video learning, while valid action labels additionally provide action supervision and invalid labels are masked. For the collected robot demonstrations, we first check action dimensions, missing values, NaNs, nonfinite values, and quaternion validity. For subsets with the required joint and end-effector state fields, we further exclude trajectories with abnormal joint angles, excessive changes in joint angles or endeffector poses between adjacent frames, or no movement throughout the trajectory. Trajectories that pass these quality checks are used for action training.

## 4. Experiments

We aim to use multisource experience to preserve the language–video capabilities acquired during video pretraining and extend them to generalizable robot control. Our experiments examine how language– video alignment, language–video–action alignment, joint training, and pretraining scale affect downstream generalization. We organize our study around the following four questions.

• How to establish effective cross-modal alignment?

• How to integrate multisource experience?

• Does scaling pretraining improve control generalization?

• How well does our model perform in unseen environments?

## 4.1. Experimental Setup

Pretraining. We pretrain the model using the architecture and multisource corpus introduced above. The video branch is initialized from Wan2.2-TI2V-5B [2], and the action branch is randomly initialized. All model parameters are updated during training. Pretraining and post-training both use the joint video and action objective defined above. We use AdamW with an initial learning rate of 4 × 10<sup>−5</sup> and a default global batch size of 256. Training steps and specific settings for each study are given in the corresponding experiment descriptions and table captions.

Simulation post-training. We use the 50 manipulation tasks in RoboTwin 2.0 [43]. For the main simulation comparisons, we include VLA baselines (StarVLA [44], � [45], Xiaomi Robotics-0 [46], and X-VLA [47]) and video-based WAM baselines (Fast-WAM [21], LingBot-VA [8], OpenWAM [9], X-WAM [48], and 4D-WAM [49]). Observations include head and bilateral wrist-camera views and the end-effector state. The model predicts bimanual end-effector actions. Each task provides 50 Clean demonstrations, yielding 2,500 demonstrations in total. Starting from the corresponding pretrained checkpoint, we jointly post-train a single model on all 50 tasks for 20k steps with a global batch size of 128.

Simulation evaluation. We evaluate control generalization on RoboTwin under two settings. Models are post-trained exclusively on Clean demonstrations. Easy evaluates the resulting models under the same clean conditions used for post-training (Clean-to-Clean, C2C). Hard evaluates the same models under randomized conditions, including variations in background textures, clutter, lighting, table height, and camera placement (Clean-to-Random, C2R). For each task and setting, we run 25 evaluation episodes for ablation studies and 100 episodes for main results. We report full-task success rate (SR).

Real-world post-training. We use a bimanual Flexiv Rizon 4 platform for cup stacking, towel folding, and gift-in-box. For real-world evaluation, we select StarVLA, � , Fast-WAM, LingBot-VA, and Open-WAM as representative VLA and WAM baselines. Observations comprise two fisheye wrist-camera views and one fixed head-camera view. All compared models use the same task-specific post-training data. Starting from upstream pretrained weights, we adapt a separate model to each task for 50k steps with a global batch size of 128.

Real-world evaluation. We evaluate under Easy (C2C) and Hard (C2R) settings. Easy varies the positions of familiar objects, and Hard changes objects, backgrounds, or both. Each task and setting uses ten layouts fixed before evaluation and shared across all models. We run three trials per layout, yielding 30 trials per task and setting. All models use synchronous inference and execute actions at 30 Hz. Success rate (SR) measures the proportion of trials that complete the entire task. Partial success rate (PSR) measures the average fraction of predefined subtasks completed per trial. We compute both metrics per task and average them equally across the three tasks.

## 4.2. How to establish effective cross-modal alignment?

We examine language–video and language–video–action alignment through the design of language annotations, paired action supervision, and complementary video experience. Downstream control evalu-

ation measures the effects of these supervision choices.

## 4.2.1. Language–Video Alignment

Detailed and accurate language descriptions improve downstream control generalization. Pretrained video models learn rich information about scenes, objects, and motion from large-scale language–video data. During WAM pretraining, we provide informative language supervision that accurately matches video content. We aim to help the model retain pretrained priors and reduce forgetting of existing language–video capabilities when adapting to robot manipulation data. Across all settings, we pretrain on a mixture of egocentric videos and UMI demonstrations for 20k steps, examining description detail, scene information, and temporal granularity as three aspects of language supervision. For description detail, we compare three levels, Brief, Medium, and Detailed. We also compare settings with and without scene descriptions covering backgrounds, spatial layouts, and lighting. For temporal granularity, we compare Episode descriptions covering complete trajectories with Subtask descriptions of local ma nipulation segments.

We compare these language supervision choices in Table 3. At each temporal granularity, more detailed descriptions yield higher Hard success rates. At the Subtask level, the Hard success rate increases from 47.20% to 49.60% and 53.52% with Brief, Medium, and Detailed descriptions, respectively. Adding environment information further raises it to 55.04%, achieving the highest Easy and Hard success rates in this group. For temporal granularity, replacing Episode descriptions with Subtask descriptions under the Detailed setting increases the Hard success rate from 51.60% to 53.52%. These results indicate that richer manipulation descriptions, additional environment information, and subtask-level language supervision improve downstream control generalization. We further randomly shuffle the correspondence between Detailed Subtask descriptions and video segments within each trajectory, reducing the Hard success rate from 53.52% to 52.00%. This result further highlights the importance of accurately matched language supervision for language–video alignment.

Table 3. Caption design and temporal alignment. We vary description detail, scene information, temporal granularity, and caption–video correspondence. Gray rows indicate the configurations used in our final model.  
Table 4. Robot caption variants. We compare original task instructions, rephrased instructions, and detailed trajectory-specific descriptions.
<table><tr><td>Granularity</td><td>Setting</td><td>Easy (C2C)</td><td>Hard (C2R)</td></tr><tr><td>Caption design</td><td></td><td></td><td></td></tr><tr><td>Episode</td><td>Brief</td><td>81.36</td><td>44.96</td></tr><tr><td>Episode</td><td>Detailed</td><td>82.00</td><td>51.60</td></tr><tr><td>Subtask</td><td>Brief</td><td>83.60</td><td>47.20</td></tr><tr><td>Subtask</td><td>Medium</td><td>82.08</td><td>49.60</td></tr><tr><td>Subtask</td><td>Detailed</td><td>83.68</td><td>53.52</td></tr><tr><td>Subtask</td><td>Detailed + Env.</td><td>83.92</td><td>55.04</td></tr><tr><td>Temporal alignment</td><td></td><td></td><td></td></tr><tr><td>Subtask</td><td>Aligned</td><td>83.68</td><td>53.52</td></tr><tr><td>Subtask</td><td>Shuffled</td><td>83.04</td><td>52.00</td></tr></table>

<table><tr><td>Caption</td><td>Easy (C2C)</td><td>Hard (C2R)</td></tr><tr><td>Original</td><td>84.64</td><td>50.80</td></tr><tr><td>Rephrased</td><td>86.08</td><td>54.76</td></tr><tr><td>Detailed + Env.</td><td>86.48</td><td>56.16</td></tr></table>

Table 5. UMI action-label coverage. All videos are retained as action-label coverage varies.
<table><tr><td>Action</td><td>Easy (C2C)</td><td>Hard (C2R)</td></tr><tr><td>0%</td><td>80.96</td><td>46.72</td></tr><tr><td>25%</td><td>81.44</td><td>47.76</td></tr><tr><td>50%</td><td>83.28</td><td>49.92</td></tr><tr><td>75%</td><td>84.08</td><td>50.16</td></tr><tr><td>100%</td><td>83.28</td><td>51.28</td></tr></table>

We also find that many trajectories in the open-source robot data we examine share the same simple task instruction. Some datasets use task names or short instructions as video captions, without detailed descriptions of the manipulation process and environment. For this study, we randomly select nine subsets of the robot data, containing 145,430 trajectories but only 592 task-level instruction entries. These shared instructions do not adequately capture differences in manipulation processes and scene conditions across trajectories, limiting fine-grained correspondence between language supervision and trajectory-level visual content. We construct two variants to enrich language supervision. Rephrased provides 100 semantically equivalent paraphrases per task instruction, randomly sampling one during training to increase linguistic diversity. Detailed + Env. provides a detailed description of the specific manipulation process and environment for each trajectory, further enriching the content and diversity of language supervision. Relative to the 50.80% Hard success rate of Original, paraphrasing alone yields a gain of 3.96 percentage points, while Detailed + Env. achieves the highest success rate in this group at 56.16%, as shown in Table 4. These results further indicate that enriching linguistic expression and descriptive content helps improve language–video alignment and generalization.

## 4.2.2. Language–Video–Action Alignment

Human manipulation experience provides effective language–video–action supervision. Language–video supervision associates language descriptions with manipulation processes in videos, while paired action labels further provide supervision for language–video–action alignment. Direct post-training from Wan achieves a Hard SR of 34.56%. With robot-data pretraining, it reaches 56.16% in Table 4, demonstrating the value of robot manipulation data. We further examine whether human manipulation experience from UMI can provide effective cross-embodiment language–video–action supervision. To examine the contribution of UMI action supervision, we retain all UMI videos and vary only the proportion of trajectories providing action labels during pretraining. Videos without action labels continue to participate in video learning. At each coverage level, the subset of trajectories providing action supervision remains fixed. All settings use the same pretraining budget, followed by 20k steps of downstream post-training. From Table 5, we observe that as action-label coverage increases from 0% to 100%, the Hard success rate increases across all five coverage levels, rising from 46.72% to 51.28%, a gain of 4.56 percentage points. These results indicate that paired actions from human manipulation experience provide effective supervision for robot control, although UMI data differ visually from robot execution. Given that UMI demonstrations can be collected with handheld devices across diverse scenes, this finding supports human manipulation demonstrations as a source for scaling pretraining experience for robot control.

## 4.3. How to integrate multisource experience?

After examining the roles of language–video and language–video–action supervision, we study how to integrate multisource experience for pretraining. We first examine the benefits of increasing training on General videos, egocentric videos, and UMI demonstrations. Given differences across sources in data quality, scene and task distributions, and action supervision, we then study how the Ego–UMI sampling ratio affects downstream control generalization under a fixed training budget. Finally, we compare model architectures and conditioning designs under mixed-source training, examining how the pretrained model’s existing language–video capabilities can support action learning.

Table 6. Different data sources strengthen language–video and language–video–action alignment. We separately increase pretraining exposure to general videos (General), egocentric videos (Ego), and UMI data (UMI).
<table><tr><td>Data</td><td>Steps</td><td>Easy (C2C)</td><td>Hard (C2R)</td></tr><tr><td>General videos</td><td></td><td></td><td></td></tr><tr><td>UMI/Ego 20k</td><td>20k</td><td>82.64</td><td>53.84</td></tr><tr><td>+ General 25k</td><td>45k</td><td>82.64</td><td>56.48</td></tr><tr><td>Egocentric videos</td><td></td><td></td><td></td></tr><tr><td>UMI only</td><td>10k</td><td>83.04</td><td>49.68</td></tr><tr><td>+ Ego 10k</td><td>20k</td><td>82.64</td><td>53.84</td></tr><tr><td>+ Ego 20k</td><td>30k</td><td>82.32</td><td>55.60</td></tr><tr><td>UMI data</td><td></td><td></td><td></td></tr><tr><td>Ego only</td><td>10k</td><td>79.68</td><td>42.08</td></tr><tr><td>+ UMI 10k</td><td>20k</td><td>82.64</td><td>53.84</td></tr><tr><td>+ UMI 20k</td><td>30k</td><td>83.60</td><td>51.20</td></tr></table>

Transitional pretraining on General videos provides a stronger starting point. We first examine the role of pretraining on General videos. Their content distribution is close to that of the original video pretraining data, while WAM adjusts input frame count, resolution, and attention computation to support robot control. We therefore first use General videos to help the model adapt to these computational changes under a content distribution close to its pretraining distribution. This transition aims to better preserve existing video priors and provide a foundation for subsequent embodied learning. To this end, we add 25k steps of transitional pretraining using only General videos before 20k steps of mixed training on egocentric videos and UMI demonstrations, increasing the Hard success rate from 53.84% to 56.48%, as reported in Table 6.

Increasing pretraining exposure to individual data sources improves WAM performance. Egocentric videos broaden the range of egocentric manipulation scenarios, and UMI demonstrations provide manipulation experience paired with actions. We next increase training exposure to egocentric videos or UMI demonstrations while keeping exposure to the other source fixed. We mix egocentric videos and UMI demonstrations during training. Each source’s training exposure is calculated from the total training steps and its sampling proportion. As shown in Table 6, increasing training on egocentric videos or UMI demonstrations generally improves downstream control performance.

Equal Ego–UMI sampling yields the best performance in unseen environments. The preceding experiments increase both source exposure and the training budget. We further examine how to allocate a fixed budget across the two sources. Egocentric videos provide diverse visual experience across tasks and scenes, while UMI data provide both video and action supervision. The sampling ratio determines how these forms of supervision are allocated during training. We fix the pretraining budget at 20k steps and compare five Ego-to-UMI sampling ratios of 1:0, 2:1, 1:1, 1:2, and 0:1. In Table 7, we observe that all three mixed settings achieve higher Hard success rates than training on either source alone. Equal sampling achieves the highest Hard success rate of 53.84%, improving over UMI-only training at 51.28% by 2.56 percentage points. These results indicate that an appropriate Ego-to-UMI sampling ratio helps balance language–video and language–video–action supervision and improves control generalization in randomized scenes.

Table 7. Ego and UMI provide complementary supervision. We compare five Ego-to-UMI sampling ratios under a fixed 20k-step pretraining budget.
<table><tr><td>Ratio</td><td>Easy (C2C)</td><td>Hard (C2R)</td></tr><tr><td>1:0</td><td>77.04</td><td>42.56</td></tr><tr><td>2:1</td><td>82.72</td><td>52.64</td></tr><tr><td>1:1</td><td>82.64</td><td>53.84</td></tr><tr><td>1:2</td><td>83.20</td><td>53.28</td></tr><tr><td>0:1</td><td>83.28</td><td>51.28</td></tr></table>

Appropriate architectural choices help preserve pretrained video priors. Under the same 30k-step multisource pretraining setup, we compare Single-stream, in which video and actions share a residual stream, with Dual-stream, which maintains separate residual streams. We also examine the language conditioning pathway of the action branch. We separate video and action feature spaces to accommodate both the reuse of pretrained video representations and the learning of new action representations. The video branch has already learned correspondences between language and visual dynamics during pretraining. The newly introduced text conditioning pathway in the action branch, however, lacks sufficient pretraining. We therefore aim to reuse the semantic information available in the video branch for action prediction, reducing potential interference from the new text conditioning pathway. To this end, we retain language conditioning in the video branch and let the action branch access language information only through video features. As reported in Table 8, Single-stream and standard Dual-stream achieve similar Hard success rates of 52.56% and 52.48%, respectively. The dual-stream model using this language conditioning pathway achieves the highest Hard success rate in this group at 57.36%, a gain of 4.88 percentage points over standard Dual-stream. These results support conditioning the action branch on language through video features.

Table 8. Model architecture affects control generalization. We compare video–action architectures (top) and extra image features (bottom).
<table><tr><td colspan="3">Easy (C2C) Hard (C2R)</td></tr><tr><td>Architecture</td><td></td><td></td></tr><tr><td>Single-stream</td><td>86.48</td><td>52.56</td></tr><tr><td>Dual-stream</td><td>84.96</td><td>52.48</td></tr><tr><td>Ours</td><td>84.80</td><td>57.36</td></tr><tr><td>Extra features</td><td></td><td></td></tr><tr><td>Native</td><td>83.44</td><td>49.52</td></tr><tr><td>+ CLIP [50]</td><td>82.40</td><td>42.48</td></tr><tr><td>+ DINOv2 [51]</td><td>82.00</td><td>39.76</td></tr><tr><td>+ SigLIP2 [52]</td><td>82.40</td><td>42.40</td></tr><tr><td>十  $\mathsf { \Omega } _ { \mathrm { Q w e n 3 - V L } }$ </td><td>81.60</td><td>42.96</td></tr></table>

For visual conditioning, we examine whether image-encoder features can provide benefits. We pretrain for 20k steps with equal sampling of egocentric videos and UMI data. Using Native as the baseline, we separately add image features extracted by CLIP, DINOv2, SigLIP2, and Qwen3-VL 2B [35] through cross-attention to provide additional visual conditioning. Native achieves the highest Easy and Hard success rates in this group, at 83.44% and 49.52%, respectively. Adding features from the four image encoders lowers the Hard success rate to 39.76%–42.96%, as summarized in Table 8. These results support retaining the pretrained video model’s native visual inputs. The fusion pathway for the additional image features has not undergone the same large-scale language–video pretraining as the video branch. This may lead the model to rely more on appearance cues in the current training distribution, weakening generalization to unseen scenes.

## 4.4. Does scaling pretraining improve control generalization?

Building on the preceding supervision design and multisource training recipe, we evaluate different pretrained checkpoints in simulation and on real robots. Within each evaluation domain, we use the same downstream post-training settings to examine how additional pretraining affects control generalization.

Increasing pretraining investment in multisource experience provides a stronger WAM. We study how model capabilities scale with pretraining, measured in training video hours. Here, 50k pretraining steps correspond to approximately 45k video hours. We use a mixture of egocentric videos and UMI demonstrations with equal sampling weights. For this study, we combine the configurations marked by gray rows in Tables 3, 7, and 8, such as using Detailed Subtask descriptions with scene and lighting information as language conditions. The 0-hour setting skips multisource pretraining and proceeds directly to downstream post-training from the video-model initialization.

![](images/43490d76e3e88cc63a6f4367c4e2ea6f9c52024d47f58e2a3ba0663595acb423.jpg)  
(a) RoboTwin

![](images/4249e1a290075763833b8d03f02bdde31bd72875639c4674fe3861dbe28b1838.jpg)  
(b) Real World  
Figure 6. Scaling pretraining improves control generalization. We vary training video hours on a fixed corpus of multisource experience, using the same downstream post-training protocol within each domain. (a) RoboTwin SR under the Easy (C2C) and Hard (C2R) settings. (b) Real-world SR and PSR under the Hard setting, averaged across three tasks. The 0-hour setting uses the video-model initialization without additional pretraining and achieves no success in real-world evaluation.

As shown in Figure 6(a), multisource pretraining improves both Easy and Hard success rates over the 0-hour baseline. The highest Hard success rate reaches 59.60%, compared with 34.56% without pretraining. In Figure 6(b), we show that real-world Hard SR and PSR, averaged equally across three tasks, increase consistently as training video hours grow from 22.5k to 90k, rising from 33.33% and 46.90% to 48.89% and 61.43%, respectively. The 0-hour setting fails to complete any task. These results show continued gains within the evaluated range, supporting longer pretraining on both sources for control under real-world scene changes.

Table 9. Scaling pretraining on human experience and robot data improves generalization across unseen environmental conditions. We first pretrain on egocentric videos and UMI demonstrations (Stage 1), then continue training on robot data (Stage 2). The middle four columns evaluate single-factor environmental changes.
<table><tr><td>Stage 1</td><td>Stage 2</td><td>Easy (C2C)</td><td>Background</td><td>Lighting</td><td>Clutter</td><td>Height</td><td>Hard (C2R)</td></tr><tr><td>0k</td><td>0k</td><td>76.16</td><td>59.12</td><td>76.24</td><td>51.28</td><td>57.68</td><td>34.56</td></tr><tr><td>20k</td><td>0k</td><td>82.64</td><td>72.16</td><td>84.08</td><td>68.56</td><td>70.32</td><td>53.84</td></tr><tr><td>75k</td><td>0k</td><td>85.68</td><td>77.92</td><td>85.60</td><td>78.48</td><td>75.44</td><td>59.60</td></tr><tr><td>75k</td><td>25k</td><td>88.08</td><td>82.16</td><td>88.08</td><td>80.72</td><td>76.96</td><td>61.52</td></tr></table>

After 75k steps of pretraining on egocentric videos and UMI demonstrations, we continue pretraining on robot data for another 25k steps and apply the same RoboTwin post-training settings. In Table 9, we observe that success rates under individual changes in background, lighting, clutter, and table height improve as pretraining investment increases, with the Hard success rate ultimately reaching 61.52%.

## 4.5. Does our model perform well in unseen environments?

Building on the preceding findings and training recipe, we compare our models pretrained on our full multisource dataset with existing policies in RoboTwin simulation and real-world manipulation.

Simulation results. We compare VLA and WAM methods on the 50 manipulation tasks in RoboTwin 2.0. Each model is post-trained on Clean demonstrations and evaluated under Easy (C2C) and Hard (C2R) settings. As reported in Table 10, our model achieves a Hard success rate of 61.56%, outperforming all compared methods and exceeding the strongest baseline, OpenWAM, by 12.86 percentage points. Its Easy success rate is 88.10%, slightly below OpenWAM’s 89.30%, while its average success rate across both settings reaches 74.83%, the highest among the compared methods. Our model and OpenWAM perform similarly under Clean conditions, while our model achieves a higher overall success rate under the Hard setting, indicating better generalization across scene conditions.

Table 10. Our model outperforms VLA and WAM baselines in simulation and real-world evaluation. We compare Easy (C2C) and Hard (C2R) settings, reporting SR in simulation and SR/PSR on real-world tasks.
<table><tr><td rowspan="2">Model</td><td colspan="3">RoboTwin</td><td colspan="4">Real-world</td></tr><tr><td>Easy</td><td>Hard</td><td>Avg.</td><td>Easy SR</td><td>Easy PSR</td><td>Hard SR</td><td>Hard PSR</td></tr><tr><td>VLA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StarVLA [44]</td><td>60.03</td><td>14.06</td><td>37.05</td><td>31.11</td><td>36.87</td><td>4.44</td><td>21.55</td></tr><tr><td>π0.5 [45]</td><td>70.66</td><td>46.00</td><td>58.33</td><td>50.00</td><td>71.87</td><td>20.00</td><td>37.78</td></tr><tr><td>Xiaomi Robotics-0 [46]</td><td>62.86</td><td>18.22</td><td>40.54</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>X-VLA [47]</td><td>67.96</td><td>20.95</td><td>44.46</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>WAM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fast-WAM [21]</td><td>77.76</td><td>1.92</td><td>39.84</td><td>35.56</td><td>49.33</td><td>6.67</td><td>14.52</td></tr><tr><td>LingBot-VA [8]</td><td>77.04</td><td>36.16</td><td>56.60</td><td>41.11</td><td>53.61</td><td>10.00</td><td>24.01</td></tr><tr><td>OpenWAM [9]</td><td>89.30</td><td>48.70</td><td>69.00</td><td>37.78</td><td>46.35</td><td>23.33</td><td>31.67</td></tr><tr><td>X-WAM [48]</td><td>69.98</td><td>25.84</td><td>47.91</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>4D-WAM [49]</td><td>81.50</td><td>41.80</td><td>61.65</td><td>一</td><td>1</td><td>1</td><td>一</td></tr><tr><td>Ours</td><td>88.10</td><td>61.56</td><td>74.83</td><td>56.67</td><td>68.57</td><td>40.00</td><td>50.36</td></tr></table>

Real-world results. We evaluate cup stacking, towel folding, and gift-in-box on a bimanual Flexiv platform. All three tasks start from the same pretrained checkpoint and undergo 50k steps of taskspecific post-training. Each task-specific model is evaluated under both Easy (C2C) and Hard (C2R) settings. All compared methods use the same task-specific post-training data. Under the Hard setting, our model achieves the highest SR and PSR among the evaluated methods, as shown in Table 10 and Figure 7. Averaged equally across the three tasks, SR reaches 40.00%, exceeding OpenWAM’s 23.33% by 16.67 percentage points. PSR reaches 50.36%, exceeding �<sub>0.5</sub>’s 37.78% by 12.58 percentage points. These results show improvements in both full-task success and subtask completion under unseen object and background conditions, supporting the generalization benefits of the training recipe in real-world robot manipulation.

Real-world visualizations. Figure 8 shows how our model and the baselines perform the three tasks, with light red dashed boxes highlighting the baselines’ main execution errors. In cup stacking, Fast-WAM misplaces the first cup of the second layer, causing it to fall onto the cups below, and cannot complete the stack afterward. For towel folding, our model lifts and spreads both sides before complet ing both folds, whereas OpenWAM lifts only one side and fails to fold the towel properly. In gift-in-box, �<sub>0.5</sub> grasps the object but gradually loosens its gripper, causing the object to fall, and remains unsuccessful after repeated attempts. Our model retains its grasp and places the object into the box. To provide a fuller view of our model’s execution, Figure 9 presents head and external camera recordings. The gift-in-box model generalizes to fruit placement with a different scene, manipulated objects, and target container. Trained with same-color cups on a gray tablecloth, the cup-stacking model also completes rainbow stack with multicolored cups on a patterned tablecloth. These examples illustrate how our model transfers learned manipulation skills to unfamiliar objects and scenes.

![](images/d5fa4a8563e20c11d4748ff28a2026deff24ea1e562bb0bae9549f24b2dd4f86.jpg)  
(a) Real-world (Easy)

![](images/906e9f59abe6d7df0166c488fd17583981d734aa508d177bf172042a94270e30.jpg)  
(b) Real-world (Hard)  
Figure 7. Our method achieves the best performance under the Hard setting in real-world evaluation. Results are averaged equally across three real-world tasks.

## 5. Discussion and Conclusion

In this work, we studied how to extend pretrained language–video capabilities to generalizable robot control with multisource experience. Using approximately 47,000 hours of experience, we systematically investigated language–video alignment, language–video–action alignment, and their joint learning. We found detailed and accurate language descriptions preserve pretrained language–video capabilities. Meanwhile, action supervision from UMI demonstrations further supports robot control, demonstrating the value of human manipulation experience for robot learning. Our findings prove the importance of experience organization and supervision usage. The resulting training recipe demonstrates generalization to unseen conditions in simulation and real-world evaluations, with further gains from continued pretraining. We release the data, code, and model weights to support future research.

Upon the pretrained video model, the proposed method offers limited support for complex reasoning and explicit multi-stage planning, despite outperforming VLA models in our control generalization evaluations. The limited historical context may also hinder the execution of long-range tasks. In the future, we plan to explore memory mechanisms and hierarchical planning to support tasks that require reasoning over past observations and planning subsequent actions.

## Acknowledgments

We thank Zijun Xu for his contributions to the early exploration of this work. We also thank Yutong Dong for her efforts in real-world deployment.

Cup Stacking (Easy)  
![](images/134cbcd54de90ed03aa8f43ec045a3406a941170783e274248e975216b19daa2.jpg)  
Figure 8. Real-world execution comparisons. Our model completes cup stacking and towel folding under the Easy setting and gift-in-box under the Hard setting, while the corresponding baseline executions fail. Light red dashed boxes highlight the baselines’ main execution errors.

Fruit Placement  
![](images/55b6da17b362c643e93d905c8793212025c4f6d4279c72f8c2344d0de8145b92.jpg)  
Figure 9. Generalization examples from head and external camera views. Fruit placement and rainbow stack demonstrate generalization to new objects, containers, and scene appearances.

## References

[1] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

[2] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[3] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In ICLR, 2025.

[4] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[5] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In ICLR, 2024.

[6] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations. In ICML, volume 267, 2025.

[7] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[8] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[9] Heng Yu, David D Yuan, Juze Zhang, Changan Chen, Yao Feng, Michelle Baldonado, Steve Cousins, Li Fei-Fei, Jiajun Wu, and Ehsan Adeli. OpenWAM: An Open Framework for Composable World-Action Models. arXiv preprint arXiv:2610.07922, 2026.

[10] Zhihao Sun, Liu Liu, Xinjiang Wang, Haoyi Jiang, Wei Feng, Huiqiang Zhang, Xiaosong Jia, Zhizhong Su, and Zuxuan Wu. Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning? arXiv preprint arXiv:2609.40341, 2026.

[11] Jewon Yeom, Hanseul Kim, Jeongjae Park, Sungmok Jung, Jaejin Lee, and Taesup Kim. What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction. arXiv preprint arXiv:2606.07687, 2026.

[12] Haoyi Jiang, Liu Liu, Xinjiang Wang, Zhihao Sun, Zequn Chen, Sen Wang, Xinjie Wang, Xia Chen, Jingfeng Yao, Weiheng Zhao, et al. Rethinking Representations for World-Action Modeling. arXiv preprint arXiv:2609.38163, 2026.

[13] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. In Robotics: Science and Systems XX, Delft, The Netherlands, July 15-19, 2024, 2024.

[14] Zhaxizhuom Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Tianyu Wang, Shuai Liang, Pengan Chen, Pingrui Zhang, et al. Fastumi: A scalable and hardwareindependent universal manipulation interface with dataset. In Conference on Robot Learning, pages 3069–3093. PMLR, 2025.

[15] Junjie Fang, Wendi Chen, Han Xue, Fangyuan Zhou, Tian Le, Yi Wang, Yuting Zhang, Jun Lv, Chuan Wen, and Cewu Lu. Robopocket: Improve robot policies instantly with your phone. arXiv preprint arXiv:2603.05504, 2026.

[16] Jiacheng Wei, Jerry Bai, Xiaoyu Yue, Zidong Wang, Xiaoyang Guo, Cheng Chen, Fanqi Pu, Fan Wu, Zhixu Yue, Yizhuo Li, et al. XPACE: Joint World and Action Modeling from Heterogeneous Experience. arXiv preprint arXiv:2609.17372, 2026.

[17] Xingyu Miao, Zizun Li, Baole Fang, Kaiwen Song, Tenghui Wang, Hanxue Zhang, Yating Wang, Xudong Li, Yuping He, Xueyuan Wei, et al. InternW0-Δ: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data. arXiv e-prints, 2026.

[18] Yanjiang Guo, Haodong Yan, Zhide Zhong, Zhongru Zhang, Qingyuan Yang, Qingzhou Lu, Xiaoyu Chen, Yen-Jen Wang, Shuying Deng, Chenghan Yang, et al. Video Prediction Policy 2: Predict Better, Act Better. arXiv preprint arXiv:2610.10270, 2026.

[19] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning Universal Policies via Text-Guided Video Generation. In NeurIPS, 2023.

[20] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In CVPR, pages 35101–35113, 2026.

[21] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[22] OX-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv preprint arXiv:2310.08864, 2023.

[23] Dibya Ghosh, Homer Rich Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Quan Vuong, Ted Xiao, Pannag R. Sanketi, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An Open-Source Generalist Robot Policy. In Robotics: Science and Systems XX, Delft, The Netherlands, July 15-19, 2024, 2024.

[24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source visionlanguage-action model. arXiv preprint arXiv:2406.09246, 2024.

[25] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022.

[26] Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y Zhu, et al. Egoverse: An egocentric human dataset for robot learning from around the world. arXiv preprint arXiv:2604.07607, 2026.

[27] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video. In ICRA, pages 13226–13233. IEEE, 2025.

[28] Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, et al. Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026.

[29] Baoyu Li, Xinchen Yin, Mengying Lin, Yixin Zhang, and Danfei Xu. EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data. In Robot World Models, 2026.

[30] Weixin Liang, Lili Yu, Liang Luo, Srini Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models. Transactions on Machine Learning Research.

[31] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow Matching for Generative Modeling. In ICLR, 2023.

[32] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. In ICLR, 2025.

[33] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. In CVPR, pages 13320–13331, 2024.

[34] Lightwheel. EgoSuite-Open100K: A Fully Annotated Open Egocentric Human Dataset. https: //huggingface.co/collections/LightwheelAI/egosuite-open100k, August 2026. Dataset.

[35] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[36] Bytedance Seed. Seed2. 0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026.

[37] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[38] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

[39] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[40] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In CVPR, pages 976–985, 2026.

[41] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxy Open-World Dataset and G0 Dual-System VLA Model. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 20749–20756. IEEE, 2026.

[42] Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, et al. Scalable Behavior Cloning with Open Data, Training, and Evaluation. arXiv preprint arXiv:2606.27375, 2026.

[43] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[44] StarVLA Community. StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing. arXiv preprint arXiv:2604.05014, 2026.

[45] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. � : a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025.

[46] Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026.

[47] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable crossembodiment vision-language-action model. In ICLR, 2026.

[48] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026.

[49] Lishan Yang, Wenxuan Song, Xi Wang, Pingyue Sheng, Zheng Fang, Ziyang Zhou, Junjie He, Haodong Yan, Jiayi Chen, Nan Sun, et al. 4d-wam: Infusing spatiotemporal awareness into world action models through trajectory fields. arXiv preprint arXiv:2608.08023, 2026.

[50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.

[51] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.

[52] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

## A. Data Annotation Details

General videos retain their original captions. For egocentric videos, UMI demonstrations, and robotic demonstrations, we construct manipulation annotations using Qwen 3.8 Max primarily for subtask segmentation and Seed 2.1 Pro for caption annotation. The procedure consists of the following steps.

Subtask Segmentation. We use Qwen 3.8 Max to divide egocentric videos and UMI recordings into temporally ordered, non-overlapping subtasks according to action intent, recording each subtask’s start and end time. Robotic demonstrations are short, so each complete episode is treated as one subtask without further segmentation.

Caption Annotation. We use Seed 2.1 Pro to annotate each egocentric or UMI subtask and each robotic episode. The model identifies whether the segment contains manipulation and whether the visual evidence is sufficiently clear, and describes the scene, lighting, and actions at multiple levels of detail. Segments that contain no manipulation or whose visual evidence is insufficient are discarded entirely.

Action Quality Filtering. For UMI and robotic data, we further screen the associated state and action trajectories. Samples that fail one or more of the following checks are removed from action training:

• Missing or non-finite values, including NaNs, invalid dimensions, or invalid quaternions.

• Excessive absolute joint values, with any |�| > 20 rad.

• Adjacent-frame joint jumps, with any |�<sub>�+1</sub> − �<sub>�</sub> | > 1 rad.

• End-effector position jumps, with the Euclidean distance between adjacent 3D positions exceeding 0.2 m.

• End-effector orientation jumps, with a rotation-angle difference exceeding 90<sup>◦</sup> for either arm between adjacent frames.

• Persistent abnormal jerk in the joint or end-effector trajectories.

Condensed versions of the segmentation and caption annotation prompts are provided below.

## Split Prompt (Condensed)

Role. You are a professional robot manipulation video analysis expert.

Goal. Split a video into continuous, non-overlapping atomic subtasks. The input video is sampled at 2 fps, with 0.5 seconds between consecutive frames. The overall task is: task\_name.

1. Watch the complete video and identify every meaningful atomic action in chronological order.

2. Split by action intent.

3. Ensure that subtasks are sequential, with no overlap or gap.

4. Name objects accurately.

Output. Output standardized JSON containing the description, start time, and end time of every subtask.

```json
[
{
"description": "subtask description",
"start_time": 0.500,
"end_time": 2.500
}
]
```

## Caption Prompt (Condensed)

Role. You are a professional first-person manipulation video analysis expert and dense action annotation specialist. Describe scenes and continuous manipulation actions accurately, objectively, and concisely. Task. Watch the input video and use only visually observable evidence. The reference task, task\_annotation: {task\_name}, is only a weak hint; all descriptions must be supported by the visible video content.

Segment-type classification. The input is a temporal segment cut from a complete video and may contain no actual manipulation, for example waiting, posture adjustment, hovering, or observation. First determine:

• segment\_type = manipulation when the segment contains an actual operation on a task object.

• segment\_type = no\_action when the segment contains no operation on a task object.

Visual-clear classification. The field visual\_clear only describes the quality of the visual evidence:

• Set visual\_clear = true when the main objects and actions are sufficiently visible to judge, even with ordinary motion blur or local occlusion.

• Set visual\_clear = false when severe blur, darkness, occlusion, corruption, or missing views make the main objects or actions unreliable to judge.

Scene. Describe the initial operating environment in one sentence. Mention only the visible workspace, relevant background objects, approximate positions, and initial state.

Lighting. Describe brightness, exposure, uniformity, shadows, overexposure, underexposure, blur, or occlusion in one sentence.

action\_detail. Describe the manipulation as one chronological paragraph, including manipulated objects, discernible hand actions, object trajectories, interactions, state changes, and necessary adjustments.

action\_medium. Describe the manipulation in one sentence, adding object attributes, spatial relations, or visible outcomes.

action\_short. Summarize the main action and core object with a 2–8 word verb-object phrase.

Output. Return only one valid JSON object with the following fields. Use exactly manipulation or no\_action for segment\_type, and a JSON boolean for visual\_clear. Write descriptions in English. For no\_action, set the three action fields to none; otherwise describe only observed actions.

```json
{
"segment_type": "manipulation",
"visual_clear": true,
"lighting": "",
"scene": "",
"action_detail": "",
"action_medium": "",
"action_short": ""
}
```

## B. Training Details

## B.1. Common Settings

Optimization. Pretraining and post-training share the optimization settings in Table 11. Batch sizes, training steps, and learning-rate schedules vary across training stages, as detailed in Section B.2.

Table 11. Shared optimization settings. Values shared across pretraining and post-training.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Adam  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.95, 0.999)</td></tr><tr><td>Adam €</td><td>10⁻8</td></tr><tr><td>Weight decay</td><td>10-5</td></tr><tr><td>Gradient clip</td><td>1.0</td></tr><tr><td>Warmup ratio</td><td>5%</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Model inputs and trainable components. We use three camera views. Each view is resized to 128 × 160 and arranged into a 2×2 grid, yielding a final 256 × 320 composite frame. The head-camera view is duplicated across the top row, while the left- and right-wrist views occupy the bottom row. During training, each view is randomly cropped to 95% of its original extent before resizing, followed by color jitter with brightness, contrast, saturation, and hue parameters of 0.3, 0.4, 0.5, and 0.08, respectively.

We represent bimanual end-effector states and actions as 16-dimensional vectors and predict actions in chunks of 48 steps.

## B.2. Training Configurations

Initialization. In pretraining, we initialize the video DiT from Wan2.2-TI2V-5B [2] and load its pretrained UMT5-XXL text encoder and Wan2.2 video VAE. The action branch, including the action and state encoders, transformer blocks, and action decoder, is randomly initialized. For each experiment, post-training starts from the corresponding pretrained weights.

Pretraining. Training uses BF16 mixed precision. The default global batch size for ablation studies is 256. We increase it to 512 for pretraining in the scaling study and subsequent robot-data pretraining described in Section 4.4, as well as for pretraining the final models evaluated in Section 4.5. For the scaling study, we sample egocentric videos and UMI demonstrations with equal probability and use detailed subtask descriptions with scene and lighting information. We evaluate checkpoints at 25k, 50k, 75k, and 100k steps, as shown in Figure 6. Table 12 compares pretraining in the scaling study with simulation and real-world post-training.

Simulation post-training. We jointly train a single model on all 50 RoboTwin 2.0 tasks using 50 Clean demonstrations per task, following the same post-training protocol across simulation experiments. Each sample contains left-wrist, right-wrist, and head-camera views and the current bimanual end-effector state. The same three-view preprocessing and augmentation settings described above are used during simulation post-training. We use absolute end-effector states and actions and normalize both with their q99 statistics.

Real-world post-training. For each real-world task, we train an independent model using task-level instructions. We apply this procedure to cup stacking, towel folding, and gift-in-box.

Table 12. Comparison of training configurations.
<table><tr><td rowspan="2">Configuration</td><td>Pretraining</td><td colspan="2">Post-training</td></tr><tr><td>Scaling study</td><td>Simulation</td><td>Real world</td></tr><tr><td>Global batch size</td><td>512</td><td>128</td><td>128</td></tr><tr><td>Training steps</td><td>100,000</td><td>20,000</td><td>50,000</td></tr><tr><td>Learning-rate schedule</td><td>Constant</td><td>Cosine</td><td>Cosine</td></tr></table>

## C. Real-World Evaluation Protocol

## C.1. Post-Training Data Preparation

Post-training data. We adapt models separately to cup stacking, towel folding, and gift-in-box using task-specific data for post-training following general-purpose pretraining on egocentric videos and UMI demonstrations. Table 13 summarizes the post-training data for each task.

Table 13. Task-specific post-training data for real-world evaluation.
<table><tr><td>Task</td><td>Episodes</td><td>Hours</td></tr><tr><td>Cup stacking</td><td>2,748</td><td>25.788</td></tr><tr><td>Towel folding</td><td>2,852</td><td>15.112</td></tr><tr><td>Gift-in-box</td><td>4,399</td><td>21.715</td></tr></table>

![](images/73e7146cb768dde1c58d5bf1ed4d6f90ad5f37ce904066d77270f95e24a7d157.jpg)  
Figure 10. Real-world evaluation layouts under Easy and Hard settings. All models use the same ten layouts for each task and setting, with three trials per layout.

Observations and actions. We use a bimanual Flexiv Rizon 4 platform. Observations comprise two fisheye wrist-camera views and one fixed head-camera view. Actions encode the end-effector position, orientation, and continuous gripper width for each arm. For OpenWAM, we follow its official default configuration and use a 6D rotation representation. All other models represent orientation with a wxyz quaternion, yielding a 16-dimensional action vector across both arms.

Cup stacking: Rearrange the stacked cups.  
![](images/651c7e61488b56f67cec76a7454216bd386e74ccdbf51d24cc1756c473469687.jpg)  
Figure 11. Subtask illustrations for the three real-world tasks.

## C.2. Evaluation Protocols and Metrics

Evaluation protocols. We evaluate under Easy (C2C) and Hard (C2R) settings. Easy settings retain the scene conditions and physical objects used in post-training, including their shapes and colors, while varying their initial configurations across layouts. Hard settings vary across scene and object attributes, including background, tabletop material, object instance, shape, color, size, and texture. The evaluation layouts are illustrated in Figure 10. All models use synchronous inference and execute actions at 30 Hz.

Layouts and trials. A layout specifies the initial object configurations, target positions, and scene appearance. For each task and setting, ten layouts are fixed before evaluation and shared across all models. We run three trials per layout to account for variability in object placement and model sampling, yielding 30 trials per task and setting. To reproduce a layout, we align the fixed live head-camera view with a reference photograph using an image overlay.

Scoring and termination. Each task is divided into predefined, equally weighted subtasks, scored by reviewing execution videos. The protocol permits up to five additional autonomous replanning attempts after an initial failed attempt, retaining the current environment state and execution history without human assistance. All models follow the same retry and termination rules. A trial is marked as a failure if the cup tower collapses or an object leaves the workspace.

Metrics. We report success rate (SR) and partial success rate (PSR). For a task with � = 30 trials and � subtasks, let $s _ { i k } \in \{ 0 , 1 \}$ indicate whether subtask � succeeds in trial �. Complete-task success requires all subtasks to succeed, so its indicator is $\begin{array} { r } { z _ { i } = \prod _ { k = 1 } ^ { K } s _ { i k } } \end{array}$ . The two metrics are

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { i } , \qquad \mathrm { P S R } = \frac { 1 } { N K } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } s _ { i k } .\tag{6}
$$

SR measures complete-task success. PSR measures progress at the subtask level. The three-task score is the equally weighted mean of the task-level rates, computed separately for Easy and Hard and expressed as a percentage.

Task details. Figure 11 shows the three real-world tasks and their corresponding ordered subtask decompositions. Cup stacking requires arranging cups into a stable pyramid, towel folding requires folding a towel in half twice, and gift-in-box requires placing a gift inside an open gift box. We divide these tasks into seven, four, and two subtasks, respectively, corresponding to observable milestones in task execution. These decompositions define the task-specific subtask counts � used in the PSR calculation above.

## D. Detailed Results

## D.1. RoboTwin Task Success Rates

Table 14. Per-task success rates on RoboTwin 2.0 under Easy and Hard settings. Each task is evaluated over 100 episodes per setting. All values are percentages.
<table><tr><td>Task</td><td>Easy</td><td>Hard</td><td>Task</td><td>Easy</td><td>Hard</td></tr><tr><td>Adjust Bottle</td><td>97</td><td>81</td><td>Place Can Basket</td><td>84</td><td>36</td></tr><tr><td>Beat Block Hammer</td><td>97</td><td>55</td><td>Place Cans Plasticbox</td><td>99</td><td>74</td></tr><tr><td>Blocks Ranking RGB</td><td>98</td><td>80</td><td>Place Container Plate</td><td>96</td><td>87</td></tr><tr><td>Blocks Ranking Size</td><td>87</td><td>30</td><td>Place Dual Shoes</td><td>56</td><td>33</td></tr><tr><td>Click Alarmclock</td><td>100</td><td>75</td><td>Place Empty Cup</td><td>100</td><td>77</td></tr><tr><td>Click Bell</td><td>100</td><td>85</td><td>Place Fan</td><td>87</td><td>48</td></tr><tr><td>Dump Bin Bigbin</td><td>91</td><td>66</td><td>Place Mouse Pad</td><td>73</td><td>29</td></tr><tr><td>Grab Roller</td><td>100</td><td>80</td><td>Place Object Basket</td><td>84</td><td>56</td></tr><tr><td>Handover Block</td><td>75</td><td>31</td><td>Place Object Scale</td><td>94</td><td>67</td></tr><tr><td>Handover Mic</td><td>78</td><td>40</td><td>Place Object Stand</td><td>98</td><td>83</td></tr><tr><td>Hanging Mug</td><td>3</td><td>1</td><td>Place Phone Stand</td><td>87</td><td>28</td></tr><tr><td>Lift Pot</td><td>100</td><td>56</td><td>Place Shoe</td><td>98</td><td>82</td></tr><tr><td>Move Can Pot</td><td>90</td><td>72</td><td>Press Stapler</td><td>81</td><td>78</td></tr><tr><td>Move Pillbottle Pad</td><td>95</td><td>58</td><td>Put Bottles Dustbin</td><td>90</td><td>82</td></tr><tr><td>Move Playingcard Away</td><td>98</td><td>56</td><td>Put Object Cabinet</td><td>78</td><td>57</td></tr><tr><td>Move Stapler Pad</td><td>58</td><td>22</td><td>Rotate QRcode</td><td>90</td><td>50</td></tr><tr><td>Open Laptop</td><td>93</td><td>81</td><td>Scan Object</td><td>95</td><td>57</td></tr><tr><td>Open Microwave</td><td>86</td><td>4</td><td>Shake Bottle</td><td>100</td><td>98</td></tr><tr><td>Pick Diverse Bottles</td><td>94</td><td>74</td><td>Shake Bottle Horizontally</td><td>99</td><td>98</td></tr><tr><td>Pick Dual Bottles</td><td>99</td><td>85</td><td>Stack Blocks Three</td><td>93</td><td>82</td></tr><tr><td>Place A2B Left</td><td>92</td><td>52</td><td>Stack Blocks Two</td><td>100</td><td>99</td></tr><tr><td>Place A2B Right</td><td>95</td><td>59</td><td>Stack Bowls Three</td><td>84</td><td>80</td></tr><tr><td>Place Bread Basket</td><td>87</td><td>67</td><td>Stack Bowls Two</td><td>99</td><td>90</td></tr><tr><td>Place Bread Skillet</td><td>92</td><td>57</td><td>Stamp Seal</td><td>94</td><td>29</td></tr><tr><td>Place Burger Fries</td><td>97</td><td>60</td><td>Turn Switch</td><td>44</td><td>51</td></tr></table>

## D.2. Real-World Task Success Rates

## D.2.1. Baseline Comparison

Figure 12 reports SR and PSR for each real-world task and their average under the Easy and Hard settings.

![](images/4859755de3611311f365119b496c9daa2179030b96ff801d5b073b722a3e3699.jpg)  
Figure 12. Real-world comparison of our model and baseline methods.

## D.2.2. Pretraining Scaling

Figure 13 compares the 0-hour setting with models pretrained for 25k, 50k, 75k, and 100k steps, corresponding to approximately 22.5k, 45k, 67.5k, and 90k training video hours, respectively. At 0 hours, SR and PSR are both 0% for every task under both settings.

![](images/253b1f817827916e2c57d740a00e1c6c9f1c39cdcdcadab1831ffee9482fc326.jpg)  
Figure 13. Real-world performance with increasing pretraining video hours.

## E. Real-World Execution Examples

Figures 14–16 show execution examples under Hard settings. Head-camera frames are sampled at uniform time intervals. Numbers indicate the order of the sampled frames.

Cup stacking (Hard)  
![](images/9d3d6a4d2780024dfd7b7acd19d6c5c39df915d4ef8d36810d7e4e4c7447ffe1.jpg)  
Figure 14. Cup stacking. The robot builds a three-tier pyramid from nested cups.

Towel folding (Hard)  
![](images/aabf89afb98ae19d0195515c79941db738b25503ed253f1fd109c2b797ed15b7.jpg)  
Figure 15. Towel folding. The robot spreads the towel and folds it twice.

Gift-in-box (Hard)  
![](images/3d8a27bbec2bd9b8f31acc1c6c0431915555f548bb91eb9a401323181c473c01.jpg)  
Figure 16. Gift-in-box. The robot places the gift in the box and releases it.