# INTERACT: Learning Interaction-Aware World Dynamics from Driving Videos for Autonomous Driving

Zhan Chen<sup>1</sup>, Songyan Zhang<sup>1</sup>, Wenhui Huang<sup>1</sup>, Zirui Li<sup>1</sup>, Zhidong Wang<sup>1</sup>, Qihang Huang<sup>1</sup>, Shiyu Fang<sup>2</sup>, Chen Lv<sup>1∗</sup>

<sup>1</sup>Nanyang Technological University, Singapore <sup>2</sup>Tongji University, Shanghai, China

## Abstract

Safe autonomous driving requires more than recognizing the current scene: the ego vehicle must anticipate how surrounding trafic participants will evolve under its own motion. However, existing end-to-end driving models typically encode scenes into dense visual or BEV features, leaving object-level interactions implicit. We propose INTERACT (INTERaction-aware Action-Conditioned Trajectory learning), a three-stage framework that learns a slot-based object-centric world model from unlabeled driving videos and transfers it to trajectory planning. INTERACT first learns recurrent slots that group dense visual features into temporally structured scene-entity representations without object annotations. It then infers latent ego actions from slot context and video transitions, and uses them to condition future slot prediction, capturing how ego motion and surrounding entities co-evolve. Finally, the learned slots, latent actions, and future slot predictions are distilled into a trajectory planner to guide trajectory generation and selection. Extensive experiments on NAVSIM and nuScenes validate the efectiveness of INTERACT. In particular, it achieves state-of-the-art performance on NAVSIM, with improvements in safety-critical collision metrics.

## Introduction

Autonomous driving is inherently interactive: planning a safe trajectory requires anticipating how surrounding trafic participants may evolve under ego motion, and how their future states may constrain ego planning. Classical predictionand-planning pipelines model such interactions through explicit object-level interfaces, where agents are detected and tracked, their future motions are forecast, and the resulting structured states are passed to the planner (Chen et al. 2024a). While efective, such pipelines depend on costly object-level annotations, including detection labels, instance masks, tracking annotations, and future trajectories of nonego agents. Recent end-to-end (E2E) driving models reduce this reliance on hand-designed modules by learning directly from sensor observations and ego trajectories (Zhou et al. 2026b; Zhang et al. 2024; Hu et al. 2022; Hwang et al. 2024). However, they often represent scenes with dense visual, BEV, or query features, where trafic participants, road structures, background appearance, and ego-induced visual changes are entangled. As a result, both object-level interactions and ego-action structure remain implicit: the learned representation may capture scene changes, but does not explicitly reveal which entities interact with the ego vehicle, which changes are induced by ego behavior, or how future object states should afect planning.

Raw driving videos provide important cues for recovering these structures without object-level annotations. Across frames, temporal persistence and coherent motion ofer natural signals for grouping dense visual features into object-like scene entities. Meanwhile, frame-to-frame visual changes encode the coupled evolution between ego behavior and the surrounding environment, revealing how the scene changes as the ego vehicle moves. Recent progress in slot-based objectcentric learning has shown that such entity-level structure can emerge from visual observations through self-supervised learning (Locatello et al. 2020; Kipfet al. 2021; Zadaianchuk, Seitzer, and Martius 2023). In parallel, video-based latentaction learning suggests that compact action and motion representations can be inferred from unlabeled visual transitions (Ye et al. 2024; Chen et al. 2025; Seo et al. 2022; Bruce et al. 2024; Assran et al. 2025). These two lines of work indicate that unlabeled videos contain both object-centric cues and ego-action cues that are useful for learning structured representations of driving dynamics.

However, extracting object-centric states or latent actions from observed transitions alone is not suficient for planning. A planner also needs explicit predictions of how the scene will evolve in the future under ego behavior. World models (Bar et al. 2025) address this by learning predictive representations of future scene states, and recent world models for autonomous driving have shown the benefit of predicting future images (Li et al. 2025c), occupancy states (Zheng et al. 2024), scene geometry (Zhang et al. 2026; Xu et al. 2026), or latent planning representations (Li et al. 2025b). Nevertheless, most existing driving world models still encode object information, interactions, and ego-induced dynamics into dense or global representations. They neither explicitly model object-centric evolution nor tightly associate future state prediction with ego actions. This leaves a key challenge: learning an object-centric, action-conditioned world model from unlabeled driving videos that jointly represents structured scene entities, latent ego actions, and their coupled future evolution for interaction-aware trajectory planning.

In this work, we propose INTERACT, a three-stage framework that learns a compositional object-centric world model of latent scene entities and ego-conditioned dynamics from unlabeled driving videos, and transfers the learned structure to trajectory planning. The first two stages are trained in a fully self-supervised manner. Specifically, Stage I organizes dense visual observations into recurrent slots representing persistent scene entities, while Stage II integrates visual transition cues with historical slot context to infer latent ego actions, and uses these actions to forecast future slot states. Stage III distills the learned dynamics into an E2E planner that jointly predicts trajectory hypotheses, latent actions, and future slots under ego-trajectory supervision and world-model alignment. Our contributions are three-fold:

1. Object-centric representation from unlabeled driving videos. We introduce a recurrent slot-based representation that learns temporally persistent latent scene entities from unlabeled driving videos, providing a structured scene-entity interface for trajectory planning.

2. Ego-action-conditioned object-centric world modeling. We model both how ego behavior can be inferred from visual dynamics and how object slots evolve under ego latent actions, shifting from reconstruction-driven learning to interaction-aware dynamics learning.

3. World-model-guided trajectory planning. We integrate learned slots, latent actions, and predicted future slots into trajectory planning, providing interaction-aware and action-conditioned cues for trajectory generation and selection. Our method achieves state-of-the-art performance on the NAVSIM v1 and v2 benchmarks, with notable improvements in safety-critical collision metrics.

## Related Works

## End-to-end Planning.

Autonomous driving has shifted from modular pipelines to E2E learning-based systems (Chen et al. 2024a). Classical pipelines decompose driving into perception, prediction, and planning, using detected agents, tracked histories, and map elements as explicit interfaces for interaction reasoning, but often require dense object-level supervision and hand-designed task interfaces. Recent E2E planners instead learn task-relevant representations directly from sensor observations and ego trajectories. Existing methods have explored camera-LiDAR fusion and BEV-based imitation learning (Chitta et al. 2022), unified perception-predictionplanning architectures (Hu et al. 2023; Weng et al. 2024), multimodal trajectory generation and selection with difusion, proposal-centric decoding, distillation, or learned scoring (Zou et al. 2025; Guo et al. 2025; Yao et al. 2026; Kirby et al. 2026), as well as VLA-based driving models with language and reasoning abilities (Zhou et al. 2026b,a; Wang et al. 2025). Despite these advances, most planners still operate on dense visual, BEV, or query features, where trafic participants, road structures, and ego-induced changes are entangled. In contrast, INTERACT learns an object-centric ego-object world model from raw driving videos and transfers the learned structured dynamics to trajectory planning.

## World Models for Autonomous Driving.

World models learn predictive representations of environment dynamics, which are essential for planning under uncertainty. In autonomous driving, recent works explore future scene prediction in the form of generated driving videos, future visual features, occupancy states, or latent planning representations (Li et al. 2025b; Zheng et al. 2025; Xu et al. 2026; Wang et al. 2024; Zheng et al. 2024; Xia et al. 2025; Zhang et al. 2026), showing that future observations or scenestate prediction can improve downstream planning. However, many of them represent the scene as a holistic latent state or dense feature field, making it dificult to expose object-level interactions and the causal efect of ego behavior on future scene evolution. In parallel, joint-embedding predictive architectures provide a latent predictive paradigm by predicting future representations instead of reconstructing pixels (Assran et al. 2025; Bar et al. 2025), encouraging semantic and dynamical abstraction while avoiding unnecessary low-level appearance details. Recent driving work has further connected JEPA-style video representations with trajectory planning (Wang et al. 2026). Building on this latent predictive perspective, INTERACT learns a structured, action-conditioned world model for planning, moving beyond holistic future feature prediction toward ego-aware scene dynamics.

## Video-Based Action and Object-Centric Learning.

Recent studies show that videos can provide supervision for action and dynamics learning without explicit action labels. Action-free pretraining learns transferable visual representations for control (Seo et al. 2022). LAPA infers discrete latent actions from frame transitions to pretrain VLA models (Ye et al. 2024), while Moto represents robot videos as latent motion tokens for manipulation (Chen et al. 2025). Generative interactive environments and latent action-conditioned video models further suggest that unlabeled videos contain reusable priors for prediction and control (Bruce et al. 2024; Assran et al. 2025). These works show that video transitions reveal latent causes of change, such as actions, motions, or controllable dynamics. However, they mainly focus on manipulation or general video domains, leaving the ego-environment interaction structure in driving videos underexplored.

Object-centric learning ofers a complementary way to model structured dynamics by decomposing visual observations into latent entities. Slot Attention learns object-like representations through iterative attention (Locatello et al. 2020), while SAVi introduces temporal recurrence for consistent object binding in videos (Kipf et al. 2021). VideoSAUR further scales object-centric learning to real-world videos using temporal feature similarity as a self-supervised motion cue (Zadaianchuk, Seitzer, and Martius 2023). These methods show that object-centric slots can emerge without dense annotations. INTERACT introduces slots as the intermediate interface between dense video representations and trajectory planning, and further couples these slots with latent ego actions to model interaction-aware scene evolution.

![](images/ce3e35d76e4c70ca0e2255c11d60c05898cf16becac3d3ad4033e43e92c2bad2.jpg)  
Figure 1: Overview of the framework. Stage I learns object-centric slot representations. Stage II learns latent ego actions and ego-conditioned future slot dynamics. Stage III distills the learned world dynamics into a trajectory planner, with future-slot and action-token predictions supervised by pseudo-labels from the frozen slot encoder and action tokenizer, respectively.

## Method

## Overview

INTERACT is a three-stage framework that learns an objectcentric ego-object world model from raw driving videos and transfers the learned dynamics to trajectory planning. The overall pipeline is illustrated in Fig. 1. At a high level, IN-TERACT factorizes driving video dynamics into three complementary components: object-centric scene representation, latent ego-action modeling, and action-conditioned future scene evolution. The model is optimized through a sequential three-stage training procedure. Stage I learns recurrent object-centric slots from frozen visual features, providing a structured latent scene state without object-level supervision. Building on this representation, Stage II learns egoconditioned dynamics by inferring latent ego actions from visual transitions and historical slots, and predicting future slot evolution under the inferred action. Together, Stages I and II constitute the self-supervised object-centric egoobject world model. Stage III transfers the learned world model to a supervised planner, which jointly predicts sceneconditioned ego actions, future slots, and trajectory hypotheses for interaction-aware planning.

## Recurrent Object-Centric Video Representation

Explicit prediction-and-planning pipelines model interactions over agent histories and map elements. Stage I aims to construct an object-level latent interface from raw videos by learning a recurrent set of slots as latent states of scene elements. These slots are optimized to preserve the spatial, dynamic, and appearance cues required for later world modeling. Specifically, given a video clip $\mathcal { T } _ { - T _ { h } : T _ { f } } =$ $\{ I _ { - T _ { h } } , \dots , I _ { T _ { f } } \}$ , we first extract frozen DINOv2 features $X _ { t } ~ = ~ E _ { \mathrm { d i n o } } ( \bar { I } _ { t } ) ~ \in ~ \mathbb { R } ^ { P \times D }$ , where P and D denote the number of image patches and the feature dimension, respectively. Following recurrent object-centric video models such as SAVi (Kipf et al. 2021), we bind patch features into K temporally persistent slot representations with a recurrently initialized Slot Attention module (Locatello et al. 2020):

$$
S _ { t } , M _ { t } = \mathrm { S l o t A t t n } ( X _ { t } ; S _ { t - 1 } ) ,\tag{1}
$$

where $S _ { t } = \{ s _ { t } ^ { 1 } , \ldots , s _ { t } ^ { K } \}$ denotes the updated slot set and $M _ { t }$ represents the soft assignment masks between patches and slots. The slots are initialized from a learned Gaussian distribution at the first frame and recurrently initialized from the previous time step afterwards, encouraging temporal consistency and stable object binding across time. We supervise the slots with complementary objectives at the feature, temporal, and appearance levels. To preserve semantic and spatial information, a SlotMixer decoder reconstructs DINO features from the slots, where $M _ { t }$ specifies the soft ownership of each patch by diferent slots. To make the slots sensitive to temporal scene dynamics, we follow VideoSAUR (Zadaianchuk, Seitzer, and Martius 2023) and train the decoder to predict temporal patch-transition logits $\hat { A } _ { t }$ . The target transition distribution is constructed from cosine similarities between the normalized frozen feature of a source patch p and that of a target patch q in the adjacent frame:

$$
T _ { t , t + 1 } ( p , q ) = \mathrm { s o f t m a x } _ { q } \left( \frac { \langle \bar { X } _ { t } ( p ) , \bar { X } _ { t + 1 } ( q ) \rangle } { \tau } \right) .\tag{2}
$$

where the temperature parameter τ controls the sharpness of the transition distribution. The temporal prediction loss $\mathcal { L } _ { \mathrm { t e m p } }$ is computed as the cross-entropy between $T _ { t , t + 1 }$ and softmax $( \hat { A } _ { t } )$ . This biases slots toward grouping patches with consistent semantic and temporal correspondence. Finally, a slot-to-VAE converter maps the spatially composed slot representation to the latent space of a frozen image VAE, providing an eficient appearance-level target without training an image decoder from scratch. The reconstructed DINO features and VAE latents are supervised with mean-squared error losses. The overall Stage I objective is

$$
{ \mathcal { L } } _ { \mathrm { I } } = \lambda _ { \mathrm { f e a t } } { \mathcal { L } } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { t e m p } } { \mathcal { L } } _ { \mathrm { t e m p } } + \lambda _ { \mathrm { v a e } } { \mathcal { L } } _ { \mathrm { v a e } } .\tag{3}
$$

This objective yields compact, temporally stable, and dynamically informative slots, which serve as the object-centric scene state for subsequent ego-conditioned world modeling.

## Bidirectional Ego-Object World Modeling

Stage II constructs an object-centric world model that couples latent ego-action inference with action-conditioned prediction of future scene states. An ego action should account for the ego-induced changes in visual observations, while also providing an informative condition for predicting future scene evolution. We therefore jointly learn latent ego actions and ego-conditioned future slot dynamics.

Given a training video clip $\mathcal { T } _ { - T _ { h } : T _ { f } }$ , the frozen Stage I encoder produces historical slots $S _ { - T _ { h } : 0 }$ as the object-level scene interface, and target slots $S _ { T _ { f } } ^ { \star }$ as supervision for future slot prediction. In parallel, we use a frozen V-JEPA2 encoder (Assran et al. 2025) to extract motion-aware visual tokens, including context tokens $H ^ { c t x } = E _ { \mathrm { j e p a } } ( I _ { - T _ { h } : 0 } )$ from the history and target tokens $H ^ { t a r } = \mathrm { T a i l } ( \dot { E } _ { \mathrm { j e p a } } ( \dot { I } _ { - T _ { h } : T _ { f } } ^ { \cdot } ) )$ from the full clip, which provide transition cues beyond static semantic features for inferring ego behavior in dynamic driving scenes. Importantly, future observations are used only during Stage II training; the downstream planner observes historical inputs only at inference time. An MFormer then models the visual transition, followed by cross-attention to the historical slots to inject object-centric context into the resulting motion representation:

$$
\widetilde { \cal U } = \mathrm { C r o s s A t t n } \big ( { \mathrm { M F o r m e r } } ( H ^ { c t x } , H ^ { t a r } ) , S _ { - T _ { h } : 0 } \big ) .\tag{4}
$$

Details of the MFormer implementation are provided in $\mathsf { A p - }$ pendix A. This design allows the action representation to capture frame-level visual transitions together with the scene dynamics encoded by the slots. We then quantize $\widetilde { U }$ into ego latent action tokens using a tokenizer based on vector quantization (VQ), imposing a compact behavioral bottleneck that encourages recurring visual transitions to share a reusable vocabulary of ego behaviors. To improve the expressiveness of the discrete action space, we adopt residual VQ with $L$ codebook levels and obtain the final ego latent action as

$$
z ^ { e g o } = Q _ { \psi } ( \widetilde { U } ) = \sum _ { \ell = 1 } ^ { L } q ^ { ( \ell ) } ,\tag{5}
$$

where $q ^ { ( \ell ) }$ is the code vector selected at the ℓ-th residual quantization level. The resulting ego latent action is required to reconstruct the target JEPA feature $\hat { H } ^ { t a r }$ together with the history context using a transformer decoder.

To further enforce interaction-aware action learning, we train a conditional slot predictor to forecast future slot states based on the inferred ego action. Given the last historical slots $S _ { 0 } .$ , the predictor first models slot-wise dependencies and then modulates the transition with the ego action:

$$
\begin{array} { r l } & { \widetilde { S } _ { 0 } = S _ { 0 } + \mathrm { M S A } ( \mathrm { L N } ( S _ { 0 } ) ) , } \\ & { S _ { 0 } ^ { \prime } = \widetilde { S } _ { 0 } + \mathrm { F F N } ( \mathrm { A d a L N } ( \widetilde { S } _ { 0 } , z ^ { \mathrm { e g o } } ) ) . } \end{array}\tag{6}
$$

The future slots are predicted by a residual update $\hat { S } _ { T _ { f } } =$ $S _ { 0 } + W _ { o } ( \mathrm { L N } ( S _ { 0 } ^ { \prime } ) )$ , which imposes a smooth transition prior while allowing the future object state to be conditioned on ego behavior. For optimization, latent action quantization is regularized by the $\bar { \mathsf { V } } \mathsf { Q }$ commitment loss $\mathcal { L } _ { \mathrm { { a c t } } }$ , while future slot prediction and target JEPA token reconstruction are supervised with mean-squared error losses, denoted as $\mathcal { L } _ { \mathrm { o b j e c t } }$ and $\mathcal { L } _ { \mathrm { v i s u a l } }$ , respectively. The overall Stage II loss is

$$
{ \mathcal { L } } _ { \mathrm { I I } } = \lambda _ { \mathrm { a c t } } { \mathcal { L } } _ { \mathrm { a c t } } + \lambda _ { \mathrm { o b j e c t } } { \mathcal { L } } _ { \mathrm { o b j e c t } } + \lambda _ { \mathrm { v i s u a l } } { \mathcal { L } } _ { \mathrm { v i s u a l } } .\tag{7}
$$

Together, Stages I and II constitute a self-supervised egoobject world model that infers latent ego behavior from history and predicts action-conditioned scene evolution.

## Interaction-Guided Trajectory Planning

Stage III transfers the learned object-centric world model to supervised trajectory planning. To isolate the contribution of the learned world model, we use a lightweight DETR-style trajectory decoder as the planning head. Given the observed history, learnable trajectory, action, and future-slot queries interact through self-attention and attend to scene context C through cross-attention, where the context consists of dense visual tokens, current slots from the frozen recurrent slot encoder, and ego-condition tokens encoding ego state and route command. The ego condition $e _ { t }$ further modulates the decoder through adaptive layer normalization:

$$
\begin{array} { r } { Q ^ { \ell + 1 } = \mathrm { D e c } ^ { \ell } \big ( Q ^ { \ell } , C _ { t } ; e _ { t } \big ) , \quad \ell = 1 , \ldots , L _ { \mathrm { d e c } } . } \end{array}\tag{8}
$$

The decoded queries produce M trajectory hypotheses together with their corresponding latent actions, future slots, and confidence scores, which allows trajectory prediction to be guided by the inferred ego action while remaining aware of the corresponding object-level future scene evolution.

We train the planner with trajectory imitation and worldmodel feature alignment. For multimodal trajectory prediction, we adopt a Winner-Takes-All objective, where only the trajectory hypothesis closest to the ground-truth ego trajectory is supervised with an $\ell _ { 1 }$ imitation loss $\mathcal { L } _ { \mathrm { t r a j } }$ , while the predicted latent action and future slots of the same mode are aligned in feature space with frozen world model targets using mean-squared error losses $\mathcal { L } _ { \mathrm { a c t } }$ and $\mathcal { L } _ { \mathrm { s l o t } }$ . Specifically, the latent action target is inferred from the complete training clip by the frozen Stage II tokenizer, whereas the slot target is obtained by encoding the ground-truth future frame with the frozen Stage I slot encoder. These targets are used only during training. Following prior candidate-selection practices (Kirby et al. 2026), candidate confidence is optionally supervised with dataset-specific oracle scores when available using $\mathcal { L } _ { \mathrm { s c o r e } } ,$ and the highest-confidence trajectory is selected at inference time. The overall Stage III objective is

$$
\mathcal { L } _ { \mathrm { I I I } } = \lambda _ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { t r a j } } + \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { s l o t } } \mathcal { L } _ { \mathrm { s l o t } } + \lambda _ { \mathrm { s c o r e } } \mathcal { L } _ { \mathrm { s c o r e } } .\tag{9}
$$

This objective enables the planner to adapt to dataset-specific trajectory styles while preserving the latent action and objectdynamics structure learned from raw videos.

Further architectural and implementation details of all three stages are provided in Appendix A.

## Experiments

Experimental setup. INTERACT uses only front-view images throughout all three stages. Frozen DINOv2 ViT-B/14 and V-JEPA2 ViT-g/16 serve as the image and video encoders, respectively, while a pretrained SDXL VAE (Podell et al. 2024) provides appearance-level latent supervision and supports image reconstruction from predicted slots. We use K = 7 128-dimensional slots and a three-level residual action codebook containing 32, 64, and 128 codes ofdimension 32. The planner predicts 64 trajectory hypotheses with confidence scores following (Kirby et al. 2026). All latent features have a unified dimension of 512, and the feed-forward layers use an expansion ratio of 4. Stages I-II are trained for 10 epochs and Stage III for 25 epochs on 8 NVIDIA A100 GPUs, using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and cosine annealing. The final Stage III planner contains approximately 60M trainable parameters. Unless otherwise specified, these settings are shared across all experiments. For NAVSIM, Stages I-II use navtrain and 40k PhysicalAI-AV clips for training, whereas Stage III uses only navtrain; for nuScenes, all stages use the oficial training split.

## Benchmarks

NAVSIM-v1. NAVSIM (Dauner et al. 2024) is a real-world planning benchmark built upon OpenScene (Contributors 2023) and nuPlan (Caesar et al. 2021). NAVSIM evaluates predicted trajectories using a simulation-based protocol and reports closed-loop-inspired rule-based metrics, including no at-fault collision (NC), drivable area compliance (DAC), time-to-collision with bounds (TTC), ego progress (EP), and comfort (C). These metrics are further aggregated into the primary metric, Predictive Driver Model Score (PDMS). Notably, as shown in Table 1, INTERACT outperforms recent planning methods, including world-model-based approaches such as DriveLaW (Xia et al. 2025) and EponaV2 (Xu et al. 2026), particularly on safety- and compliance-related metrics such as NC, DAC, and TTC. This highlights the advantage of structured object-centric dynamics in providing informative scene-level representations for trajectory planning.

NAVSIM-v2. NAVSIM-v2 (Cao et al. 2025) extends NAVSIM-v1 with richer rule-compliance and driving-quality metrics, including driving direction compliance (DDC), traffic light compliance (TL), lane keeping (LK), history comfort (HC), and extended comfort (EC), as well as scene perturbations generated with Gaussian Splatting. We evaluate on both navtest and navhard. The former evaluates broad generalization across diverse scenes, whereas navhard uses real safety-critical scenes and synthetic counterparts for two-stage closed-loop evaluation under distribution shift. As shown in Table 2, INTERACT achieves the highest official Extended PDMS (EPDMS) and the best EPDMS<sup>∗</sup> on navtest. Due to space limitations, the results on navhard are provided in Appendix C. Overall, our method demonstrates strong performance under the extended safety, compliance, and robustness evaluation protocols.

Table 1: NAVSIM-v1 benchmark. We compare INTER-ACT with published and recent methods on the test set (navtest). Higher is better for all metrics.
<table><tr><td>Method</td><td>|NC↑</td><td>DAC↑</td><td>TTC↑</td><td>C↑</td><td>EP↑|</td><td>|PDMS↑</td></tr><tr><td>PDM-Closed (Dauner et al. 2023) Human driver (Dauner et al. 2024)</td><td>94.6 100</td><td>99.8 100</td><td>89.9 100</td><td>86.9 99.9| 99.9 87.5</td><td></td><td>89.1 94.8</td></tr><tr><td>UniAD (Hu et al. 2023) PARA-Drive (Weng et al. 2024)</td><td>97.8 97.9</td><td>91.9 92.4</td><td>92.9 93.0</td><td>100</td><td>78.8 99.8 79.3</td><td>83.4 84.0</td></tr><tr><td>VAD-v2 (Chen et al. 2024b) AutoVLA (Zhou et al. 2026b)</td><td>98.1 98.4</td><td>94.8</td><td>94.3</td><td>100</td><td>80.6</td><td>86.2</td></tr><tr><td>GoalFlow (Xing et al. 2025)</td><td>98.4</td><td>95.6</td><td>98.0</td><td>99.9</td><td>81.9</td><td>89.1</td></tr><tr><td></td><td></td><td>98.3</td><td>94.6</td><td>100</td><td>85.0</td><td>90.3</td></tr><tr><td>ReCogDrive (Li et al. 2025d)</td><td>97.9</td><td>97.3</td><td>94.9</td><td>100</td><td>87.3</td><td>90.8</td></tr><tr><td>Hydra-MDP++ (Li et al. 2025a)</td><td>98.6</td><td>98.6</td><td>95.1</td><td>100</td><td>85.7</td><td>91.0</td></tr><tr><td>DiffusionDriveV2 (Zou et al. 2025)</td><td>98.3</td><td>97.9</td><td>94.8</td><td>99.9</td><td>87.5</td><td>91.2</td></tr><tr><td>iPad (Guo et al. 2025)</td><td>98.6</td><td>98.3</td><td>94.9</td><td>100</td><td>88.0</td><td>91.7</td></tr><tr><td>SparseDriveV2 (Sun et al. 2026)</td><td>98.5 98.4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Drive-JEPA (Wang et al. 2026)</td><td>99.1</td><td>98.2</td><td>95.0</td><td></td><td>99.9 88.6</td><td>92.0</td></tr><tr><td>DriveSuprim (Yao et al. 2026)</td><td>98.6</td><td>98.6</td><td>95.9</td><td></td><td>99.9 90.8</td><td>93.3</td></tr><tr><td>DrivoR (Kirby et al. 2026)</td><td>99.0</td><td>98.9</td><td>95.5 96.7</td><td>100</td><td>91.3</td><td>93.5</td></tr><tr><td></td><td></td><td></td><td></td><td>100</td><td>90.0</td><td>93.7</td></tr><tr><td>LAW (Li et al. 2025b)</td><td>96.4</td><td>95.4</td><td>88.7</td><td></td><td>99.9 81.7</td><td>84.6</td></tr><tr><td>World4Drive (Zheng et al. 2025)</td><td>97.4</td><td>94.3</td><td>92.8</td><td>100</td><td>79.9</td><td>85.1</td></tr><tr><td>DriveLaW (Xia et al. 2025)</td><td>99.0</td><td>97.1</td><td>96.7</td><td>100</td><td>81.3</td><td>89.1</td></tr><tr><td>DriveVLA-W0 (Li et al. 2025c)</td><td>98.7</td><td>99.1</td><td>95.3</td><td>99.3</td><td>83.3</td><td>90.2</td></tr><tr><td>EponaV2 (Xu et al. 2026)</td><td>98.6</td><td>97.9</td><td>95.7</td><td>100</td><td>84.8</td><td>90.4</td></tr><tr><td>GeoWorldAD (Zhang et al. 2026)</td><td>99.0</td><td>97.8</td><td>95.8</td><td>99.9</td><td>85.9</td><td>91.0</td></tr><tr><td>INTERACT (ours)</td><td>99.3</td><td>99.1</td><td>97.3</td><td>100</td><td>89.3</td><td>93.8</td></tr></table>

NuScenes. As a complementary open-loop evaluation, we further conduct experiments on the nuScenes benchmark (Caesar et al. 2020). We adopt a single-mode trajectory planning setting and report the prediction accuracy and collision rate of the planned ego trajectory. Due to space constraints, Table 3 reports the average results across the evaluated horizons, while the complete per-horizon results are provided in Appendix C. Our method achieves strong planning performance, demonstrating its efectiveness beyond the NAVSIM evaluation protocol. Notably, INTERACT remains competitive with recent driving models. More details about datasets and metrics are in Appendix B.

## Visualization and Analysis

Object-Centric World Model Diagnostics. Before analyzing the contribution of the learned world model to planning, we first visually inspect whether it captures meaningful object-centric scene dynamics. Figure 2 shows historical slot assignments produced by the Stage I recurrent slot encoder, the future slot layout predicted by the Stage II conditional slot predictor, and the corresponding image reconstruction. The colored maps represent pixel-to-slot assignments, with each color indicating the visual regions associated with a latent slot. Across consecutive frames, slot assignments exhibit temporal coherence over scene entities and broader regions. Conditioned on the current slots and inferred ego action, the model further predicts future layouts that capture changes in surrounding participants and scene geometry. Although the slot-based reconstructions omit fine-grained appearance details, reflecting the highly compressed nature of the slot representation, they preserve the dynamic structure most relevant to scene understanding and planning. Additional quantitative slot diagnostics, an evaluation of the ego-motion information recoverable from latent actions, and qualitative visualizations are provided in Appendix C.

Table 2: Results on the NAVSIM-v2 navtest benchmark. EPDMS<sup>∗</sup> denotes results computed with the original NAVSIM-v2 evaluation code before applying the human-behavior filtering fix, while EPDMS denotes results computed with the updated oficial implementation. Higher values indicate better performance for all metrics.
<table><tr><td>Method</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TL↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS*↑</td><td>EPDMS↑</td></tr><tr><td>ReCogDrive (Li et al. 2025d)</td><td>98.3</td><td>95.2</td><td>99.5</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>98.3</td><td>86.5</td><td>83.6</td><td></td></tr><tr><td>Hydra-MDP++ (Li et al. 2025a)</td><td>98.4</td><td>98.0</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.7</td><td>95.3</td><td>98.3</td><td>77.4</td><td>85.1</td><td></td></tr><tr><td>DiffusionDriveV2 (Zou et al. 2025)</td><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>85.5</td><td>87.5</td></tr><tr><td>SparseDriveV2 (Sun et al. 2026)</td><td>98.1</td><td>98.1</td><td>99.6</td><td>99.8</td><td>91.1</td><td>97.3</td><td>96.9</td><td>98.2</td><td>78.4</td><td>86.7</td><td>90.1</td></tr><tr><td>DriveSuprim (Yao et al. 2026)</td><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1</td><td></td></tr><tr><td>World4Drive (Zheng et al. 2025)</td><td>97.8</td><td>96.3</td><td>99.4</td><td>99.8</td><td>88.3</td><td>97.1</td><td>97.7</td><td>98.0</td><td>53.9</td><td>84.8</td><td></td></tr><tr><td>DriveVLA-Wò (Li et al. 2025c)</td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td></td><td>86.1</td></tr><tr><td>DriveLaW (Xia et al. 2025)</td><td>98.7</td><td>96.9</td><td>99.6</td><td>99.8</td><td>87.5</td><td>98.3</td><td>97.6</td><td>98.4</td><td>77.4</td><td></td><td>88.6</td></tr><tr><td>EponaV2 (Xu et al. 2026)</td><td>98.5</td><td>97.4</td><td>99.5</td><td>99.9</td><td>87.9</td><td>98.1</td><td>97.7</td><td>98.2</td><td>77.4</td><td></td><td>88.9</td></tr><tr><td>GeoWorldAD (Zhang et al. 2026)</td><td>99.0</td><td>97.8</td><td>99.6</td><td>99.7</td><td>89.1</td><td>98.6</td><td>97.6</td><td>98.0</td><td>82.2</td><td></td><td>90.4</td></tr><tr><td>INTERACT (ours)</td><td>99.3</td><td>99.0</td><td>99.7</td><td>99.7</td><td>90.9</td><td>99.0</td><td>96.3</td><td>98.3</td><td>74.5</td><td>87.8</td><td>91.4</td></tr></table>

![](images/937317002eced0db46251769b003375e428f76af88ceb52af7069f39cab9ca35.jpg)  
Figure 2: Visualization of object-centric slot dynamics and future prediction. The first four columns show historical slot assignments extracted by the Stage I recurrent slot encoder. The last three columns show the ground-truth future image, the future slot layout predicted by the Stage II conditional slot predictor, and the reconstructed future image, respectively.

Table 3: Open-loop planning results on nuScenes.
<table><tr><td>Method</td><td>L2 Avg. (m)↓</td><td>Collision Avg. (%)↓</td></tr><tr><td>UniAD (Hu et al. 2023)</td><td>0.69</td><td>0.12</td></tr><tr><td>PARA-Drive (Weng et al. 2024)</td><td>0.48</td><td>0.25</td></tr><tr><td>AutoVLA (Zhou et al. 2026b)</td><td>0.40</td><td>0.20</td></tr><tr><td>OpenREAD (Zhang et al. 2025)</td><td>0.36</td><td>0.11</td></tr><tr><td>ORION (Fu et al. 2025)</td><td>0.34</td><td>0.37</td></tr><tr><td>OpenDrive-VLA (Zhou et al. 2026a)</td><td>0.33</td><td>0.10</td></tr><tr><td>OmniDrive (Wang et al. 2025)</td><td>0.33</td><td>0.30</td></tr><tr><td>DriveTransformer-Large (Jia et al. 2025)</td><td>0.33</td><td>0.07</td></tr><tr><td>EMMA (Hwang et al. 2024)</td><td>0.32</td><td></td></tr><tr><td>AutoMoT (Huang et al. 2026)</td><td>0.32</td><td>0.07</td></tr><tr><td>INTERACT (ours)</td><td>0.32</td><td>0.10</td></tr></table>

Case Analysis. Figure 3 presents three NAVSIM scenes. The upper and lower rows show the highest- and lowestconfidence modes ranked by the scorer, respectively. Predicted and ground-truth trajectories are red and green. In Scenario A and B, the highest-ranked mode adopts a conservative trajectory that maintains a larger distance from the leading vehicle, whereas the aggressive alternative closes the gap excessively. Its future slot prediction indicates a reduced ego-relative separation and thus collision risk. In Scenario C, the selected trajectory follows the road geometry and closely matches the ground truth, while its future slots remain consistent with the scene evolution. In contrast, the lowest-ranked alternative deviates substantially from the drivable region, with the corresponding slot prediction reflecting the of-road deviation. These examples suggest that mode-conditioned slot prediction captures both interaction risk and spatial feasibility, providing informative cues for trajectory selection.

![](images/623564930efda1ee0959a16dac656354f013fe2752f04b9a996a7b53699559b8.jpg)  
Figure 3: Mode-wise planning and future-slot predictions in three NAVSIM scenes. The upper and lower rows show the highest- and lowest-ranked modes, respectively. Red and green denote predicted and ground-truth trajectories.

Table 4: Ablation of the world-model signals transferred to Stage III on NAVSIM navtest. A common frozen Stage I-II teacher is used wherever alignment targets are required.
<table><tr><td>Slot Context</td><td>Action Align.</td><td>Future-Slot Align.</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>PDMS↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>98.9</td><td>97.6</td><td>96.4</td><td>90.9</td></tr><tr><td>S</td><td>x</td><td>x</td><td>99.2</td><td>97.7</td><td>97.0</td><td>91.3</td></tr><tr><td>√</td><td>√</td><td>x</td><td>99.2</td><td>98.0</td><td>97.1</td><td>91.6</td></tr><tr><td>V</td><td>x</td><td>√</td><td>99.3</td><td>98.4</td><td>97.1</td><td>92.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>99.3</td><td>98.4</td><td>97.3</td><td>92.2</td></tr></table>

## Ablation Study

Unless otherwise specified, all ablation experiments are trained with the NAVSIM navtrain split and evaluated on the navtest set using a 16-mode planner for eficient ablation evaluation.

What Contributes to Planning? We ablate the three world-model signals transferred to Stage III: current-slot context, latent-action alignment, and future-slot alignment. As shown in Table 4, each component brings consistent overall gains over the trajectory-only baseline. Introducing slot context improves PDMS from 90.9 to 91.3, showing the benefit of an object-centric scene representation. Adding latent-action alignment further raises PDMS to 91.6, while future-slot alignment yields a larger improvement, indicating that anticipated object dynamics provide useful guidance for planning. Combining all three signals achieves the best PDMS and the highest TTC, confirming their complementary contributions to safe and accurate trajectory prediction.

Table 5: Efect of ego-action conditioning in Stage II.
<table><tr><td>Action Conditioning</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>EP↑</td><td>PDMS↑</td></tr><tr><td>x</td><td>99.2</td><td>98.3</td><td>97.0</td><td>86.4</td><td>91.9</td></tr><tr><td>√</td><td>99.3</td><td>98.4</td><td>97.3</td><td>86.2</td><td>92.2</td></tr></table>

Table 6: Scaling self-supervised video pretraining.
<table><tr><td>Stage I-II Video Data</td><td>NC↑</td><td>DAC↑</td><td>TTC↑</td><td>EP↑</td><td>PDMS↑</td></tr><tr><td>Trajectory-only (no Stage I-II)</td><td>98.9</td><td>97.6</td><td>96.4</td><td>85.6</td><td>90.9</td></tr><tr><td>20k navtrain</td><td>99.2</td><td>98.0</td><td>97.1</td><td>86.2</td><td>91.7</td></tr><tr><td>50k navtrain</td><td>99.2</td><td>98.2</td><td>97.2</td><td>86.1</td><td>91.9</td></tr><tr><td>navtrain</td><td>99.3</td><td>98.4</td><td>97.3</td><td>86.2</td><td>92.2</td></tr><tr><td>navtrain + 40k PhysicalAI-AV</td><td>99.4</td><td>98.5</td><td>97.3</td><td>86.3</td><td>92.4</td></tr></table>

Does Action Conditioning Matter? We compare the full action-conditioned slot predictor with a variant that predicts future slots independently of the latent ego action. As shown in Table 5, action conditioning improves PDMS from 91.9 to 92.2 with only a marginal decrease in EP. This result validates that coupling ego actions with future slot evolution promotes safer and more compliant planning, while keeping the learned action representation consistent with the resulting environment dynamics, leading to more coherent ego-environment modeling and better downstream planning.

Watch more, drive better? We study how the scale of selfsupervised video pretraining afects downstream planning. The trajectory-only configuration is identical to the first row of Table 4. All pretrained variants use the same Stage III model and training configuration, while only the video data used to pretrain Stages I and II are varied.

Table 6 shows that increasing the amount of in-domain pretraining data consistently improves downstream planning performance, indicating that broader video coverage enables the world model to learn more transferable egoobject dynamics. Incorporating 40k external PhysicalAI-AV clips (Ren et al. 2025) further raises PDMS, suggesting that additional unlabeled driving videos provide complementary dynamics beyond the target dataset. Nevertheless, with limited pretraining data, the teacher may not capture suficiently diverse driving behaviors and scene transitions, which can make its supervision less efective than some simpler transfer configurations in Table 4. Overall, all pretrained variants outperform the trajectory-only baseline, confirming the effectiveness of self-supervised world-model pretraining and the positive impact of scaling pretraining data.

## Conclusion

We present INTERACT, a framework that learns interactionaware planning representations directly from raw driving videos. Through self-supervised object-centric and actionconditioned world modeling, INTERACT captures structured ego–environment dynamics without object-level annotations and transfers them to an end-to-end trajectory planner. The resulting scene structure and ego-conditioned future cues support trajectory generation and selection. Experiments across standard driving benchmarks, together with ablations and visualizations, demonstrate strong planning performance and validate the learned interaction dynamics.

## References

Assran, M.; Bardes, A.; Fan, D.; Garrido, Q.; Howes, R.; Muckley, M.; Rizvi, A.; Roberts, C.; Sinha, K.; Zholus, A.; et al. 2025. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985.

Bar, A.; Zhou, G.; Tran, D.; Darrell, T.; and LeCun, Y. 2025. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 15791–15801.

Bruce, J.; Dennis, M. D.; Edwards, A.; Parker-Holder, J.; Shi, Y.; Hughes, E.; Lai, M.; Mavalankar, A.; Steigerwald, R.; Apps, C.; et al. 2024. Genie: Generative interactive environments. In International Conference on Machine Learning.

Caesar, H.; Bankiti, V.; Lang, A. H.; Vora, S.; Liong, V. E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; and Beijbom, O. 2020. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11621–11631.

Caesar, H.; Kabzan, J.; Tan, K. S.; Fong, W. K.; Wolf, E.; Lang, A.; Fletcher, L.; Beijbom, O.; and Omari, S. 2021. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810.

Cao, W.; Hallgarten, M.; Li, T.; Dauner, D.; Gu, X.; Wang, C.; Miron, Y.; Aiello, M.; Li, H.; Gilitschenski, I.; et al. 2025. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218.

Chen, L.; Wu, P.; Chitta, K.; Jaeger, B.; Geiger, A.; and Li, H. 2024a. End-to-end autonomous driving: Challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 10164–10183.

Chen, S.; Jiang, B.; Gao, H.; Liao, B.; Xu, Q.; Zhang, Q.; Huang, C.; Liu, W.; and Wang, X. 2024b. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243.

Chen, Y.; Ge, Y.; Tang, W.; Li, Y.; Ge, Y.; Ding, M.; Shan, Y.; and Liu, X. 2025. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19752–19763.

Chitta, K.; Prakash, A.; Jaeger, B.; Yu, Z.; Renz, K.; and Geiger, A. 2022. Transfuser: Imitation with transformerbased sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11): 12878–12895.

Contributors, O. 2023. Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving. In Proceedings of the Conference on Computer Vision and Pattern Recognition, Vancouver, Canada, 18–22.

Dauner, D.; Hallgarten, M.; Geiger, A.; and Chitta, K. 2023. Parting with misconceptions about learning-based vehicle motion planning. In Conference on Robot Learning, 1268– 1281. PMLR.

Dauner, D.; Hallgarten, M.; Li, T.; Weng, X.; Huang, Z.; Yang, Z.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; et al. 2024. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37: 28706–28719.

Fu, H.; Zhang, D.; Zhao, Z.; Cui, J.; Liang, D.; Zhang, C.; Zhang, D.; Xie, H.; Wang, B.; and Bai, X. 2025. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision, 24823–24834.

Guo, K.; Liu, H.; Wu, X.; Pan, J.; and Lv, C. 2025. ipad: Iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111.

Hu, S.; Chen, L.; Wu, P.; Li, H.; Yan, J.; and Tao, D. 2022. Stp3: End-to-end vision-based autonomous driving via spatialtemporal feature learning. In European Conference on Computer Vision, 533–549. Springer.

Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; et al. 2023. Planning-oriented autonomous driving. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 17853– 17862.

Huang, W.; Zhang, S.; Huang, Q.; Wang, Z.; Mao, Z.; Chua, C.; Chen, Z.; Chen, L.; and Lv, C. 2026. Automot: A unified vision-language-action model with asynchronous mixtureof-transformers for end-to-end autonomous driving. arXiv preprint arXiv:2603.14851.

Hwang, J.-J.; Xu, R.; Lin, H.; Hung, W.-C.; Ji, J.; Choi, K.; Huang, D.; He, T.; Covington, P.; Sapp, B.; et al. 2024. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262.

Jia, X.; You, J.; Zhang, Z.; and Yan, J. 2025. Drivetransformer: Unified transformer for scalable end-to-end autonomous driving. arXiv preprint arXiv:2503.07656.

Kipf, T.; Elsayed, G. F.; Mahendran, A.; Stone, A.; Sabour, S.; Heigold, G.; Jonschkowski, R.; Dosovitskiy, A.; and Gref, K. 2021. Conditional object-centric learning from video. arXiv preprint arXiv:2111.12594.

Kirby, E.; Boulch, A.; Xu, Y.; Yin, Y.; Puy, G.; Zablocki, É.; Bursuc, A.; Gidaris, S.; Marlet, R.; Bartoccioni, F.; et al. 2026. Driving on Registers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 32058–32069.

Li, K.; Li, Z.; Lan, S.; Xie, Y.; Zhang, Z.; Liu, J.; Wu, Z.; Yu, Z.; and Alvarez, J. M. 2025a. Hydra-mdp++: Advancing endto-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820.

Li, Y.; Fan, L.; He, J.; Wang, Y.; Chen, Y.; Zhang, Z.; and Tan, T. 2025b. Enhancing end-to-end autonomous driving with latent world model. In International Conference on Learning Representations, volume 2025, 42942–42959.

Li, Y.; Shang, S.; Liu, W.; Zhan, B.; Wang, H.; Wang, Y.; Chen, Y.; Wang, X.; An, Y.; Tang, C.; et al. 2025c. DriveVLA-W0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796.

Li, Y.; Xiong, K.; Guo, X.; Li, F.; Yan, S.; Xu, G.; Zhou, L.; Chen, L.; Sun, H.; Wang, B.; et al. 2025d. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052.

Liao, B.; Chen, S.; Yin, H.; Jiang, B.; Wang, C.; Yan, S.; Zhang, X.; Li, X.; Zhang, Y.; Zhang, Q.; et al. 2025. Diffusiondrive: Truncated difusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, 12037–12047.

Liu, L.; Jia, C.; Yu, G.; Song, Z.; Li, J.; Jia, F.; Wu, P.; Hao, X.; and Luo, Y. 2025. Guideflow: Constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729.

Locatello, F.; Weissenborn, D.; Unterthiner, T.; Mahendran, A.; Heigold, G.; Uszkoreit, J.; Dosovitskiy, A.; and Kipf, T. 2020. Object-centric learning with slot attention. Advances in neural information processing systems, 33: 11525–11538.

Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn, T.; Müller, J.; Penna, J.; and Rombach, R. 2024. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, volume 2024, 1862–1874.

Ren, X.; Lu, Y.; Cao, T.; Gao, R.; Huang, S.; Sabour, A.; Shen, T.; Pfaf, T.; Wu, J. Z.; Chen, R.; et al. 2025. Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models. arXiv preprint arXiv:2506.09042.

Seo, Y.; Lee, K.; James, S. L.; and Abbeel, P. 2022. Reinforcement learning with action-free pre-training from videos. In International Conference on Machine Learning, 19561– 19579. PMLR.

Sun, W.; Lin, X.; Chen, K.; Pei, Z.; Li, X.; Shi, Y.; and Zheng, S. 2026. Sparsedrivev2: Scoring is all you need for end-toend autonomous driving. arXiv preprint arXiv:2603.29163.

Wang, L.; Yang, Z.; Bai, C.; Zhang, G.; Liu, X.; Zheng, X.; Long, X.-X.; Lu, C.-T.; and Lu, C. 2026. Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving. arXiv preprint arXiv:2601.22032.

Wang, S.; Yu, Z.; Jiang, X.; Lan, S.; Shi, M.; Chang, N.; Kautz, J.; Li, Y.; and Alvarez, J. M. 2025. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the computer vision and pattern recognition conference, 22442–22452.

Wang, Y.; He, J.; Fan, L.; Li, H.; Chen, Y.; and Zhang, Z. 2024. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14749–14759.

Weng, X.; Ivanovic, B.; Wang, Y.; Wang, Y.; and Pavone, M. 2024. Para-drive: Parallelized architecture for real-time autonomous driving. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15449– 15458.

Xia, T.; Li, Y.; Zhou, L.; Yao, J.; Xiong, K.; Sun, H.; Wang, B.; Ma, K.; Chen, G.; Ye, H.; et al. 2025. Drivelaw: Unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421.

Xing, Z.; Zhang, X.; Hu, Y.; Jiang, B.; He, T.; Zhang, Q.; Long, X.; and Yin, W. 2025. Goalflow: Goal-driven flow

matching for multimodal trajectories generation in end-toend autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1602–1611.

Xu, J.; Zhong, Z.; Shu, Z.; Jia, M.; Li, M.; Bian, J.-W.; Zhang, Q.; Zhang, K.; Xie, J.; Yang, J.; et al. 2026. EponaV2: Driving World Model with Comprehensive Future Reasoning. arXiv preprint arXiv:2605.14696.

Yao, W.; Li, Z.; Lan, S.; Wang, Z.; Sun, X.; Alvarez, J. M.; and Wu, Z. 2026. Drivesuprim: Towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 11910–11918.

Ye, S.; Jang, J.; Jeon, B.; Joo, S. J.; Yang, J.; Peng, B.; Mandlekar, A.; Tan, R.; Chao, Y.-W.; Lin, B. Y.; et al. 2024. Latent Action Pretraining From Videos. In International Conference on Learning Representations.

Zadaianchuk, A.; Seitzer, M.; and Martius, G. 2023. Objectcentric learning for real-world videos by predicting temporal feature similarities. Advances in neural information processing systems, 36: 61514–61545.

Zhang, S.; Huang, W.; Chen, Z.; Collister, C. J.; Huang, Q.; and Lv, C. 2025. OpenREAD: Reinforced Open-Ended Reasoning for End-to-End Autonomous Driving with LLMas-Critic. arXiv preprint arXiv:2512.01830.

Zhang, S.; Huang, W.; Gao, Z.; Chen, H.; and Lv, C. 2024. Wisead: Knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951.

Zhang, S.; Tian, J.; Li, H.; Liu, D.; Chen, H.; Huang, W.; Li, F.; Chen, G.; Ye, H.; Chen, L.; et al. 2026. GeoWorldAD: Geometry World Action Model for Autonomous Driving. arXiv preprint arXiv:2607.17521.

Zheng, W.; Chen, W.; Huang, Y.; Zhang, B.; Duan, Y.; and Lu, J. 2024. Occworld: Learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, 55–72.

Zheng, Y.; Yang, P.; Xing, Z.; Zhang, Q.; Zheng, Y.; Gao, Y.; Li, P.; Zhang, T.; Xia, Z.; Jia, P.; et al. 2025. World4drive: End-to-end autonomous driving via intention-aware physical latent world model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 28632–28642.

Zhou, X.; Han, X.; Yang, F.; Ma, Y.; Tresp, V.; and Knoll, A. 2026a. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 13782–13790.

Zhou, Z.; Cai, T.; Zhao, S.; Zhang, Y.; Huang, Z.; Zhou, B.; and Ma, J. 2026b. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems, 38: 27920–27956.

Zou, J.; Chen, S.; Liao, B.; Zheng, Z.; Song, Y.; Zhang, L.; Zhang, Q.; Liu, W.; and Wang, X. 2025. Difusion-DriveV2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745.

## Appendix

## A. Additional Architecture Details

This section supplements the structural details omitted from the main text.

Stage I: Object-centric decoders. Input images are resized to $5 1 8 \times 5 1 8$ and encoded into a $3 7 \times 3 7$ patch grid by the frozen DINOv2 backbone. SlotMixer uses learned patch queries and three Transformer allocation blocks to recover a patch-aligned representation from the unordered slot set. Separate prediction heads reconstruct the DINO features and estimate temporal correspondences between adjacent frames. The patch-to-slot allocation learned within SlotMixer is independent of the soft assignment masks produced by recurrent Slot Attention.

For appearance supervision, the Slot Attention masks spatially compose the slot features into a dense feature map. This representation is resized to the SDXL latent resolution and processed by three convolutional blocks followed by an output projection, producing a latent compatible with the frozen SDXL VAE.

Stage II: Motion tokenizer and slot prediction. V-JEPA2 processes video clips at a resolution of 224 × 224. In the NAVSIM setting, each clip contains four historical and eight future frames sampled at 2Hz, and the tokens from the final temporal group are used as the future prediction target.

MFormer introduces a learned motion query that jointly attends to the context and target tokens through spatiotemporal deformable cross-attention. The resulting sequence is processed by four Transformer encoder layers, and the output motion token is further refined through three cross-attention layers over the historical slots before residual vector quantization.

The conditional slot predictor performs one-shot prediction at the final future horizon rather than autoregressive prediction over all future frames. It consists of four predictor blocks, each combining slot self-attention with an actionconditioned feed-forward layer using adaptive normalization. The final future slots are predicted as residual updates to the current slot state. A four-layer Transformer decoder is additionally used for the V-JEPA2 reconstruction objective.

Stage III: Mode-wise trajectory planner. Each of the 64 trajectory modes contains one trajectory query, three action queries corresponding to the residual codebook levels, and seven future-slot queries. The future-slot queries are initialized from the current slots, whereas the trajectory and action queries are learned. Queries from all modes are processed jointly by a shared six-layer DETR decoder.

Training hyperparameters. All three training stages use the AdamW optimizer with a batch size of 16, an initial learning rate of 1e-4, and a weight decay of 0.01. We adopt a learning-rate warm-up over the first 10% of the training schedule, followed by cosine decay. Unless otherwise specified, these hyperparameters are shared across all three stages.Table 7 summarizes the loss coeficients used for the three training stages. Unless otherwise specified, the same weights are used across all experiments. For datasets where candidate-level oracle scores are unavailable, the scoring loss is disabled by setting $\lambda _ { \mathrm { s c o r e } } = 0$

Table 7: Loss weights used in the three training stages. The scoring loss is enabled only when candidate-level oracle supervision is available.
<table><tr><td>Stage</td><td>Loss term</td><td>Weight</td></tr><tr><td rowspan="4">Stage I</td><td>DINO feature reconstruction  $\mathcal { L } _ { \mathrm { f e a t } }$ </td><td> $\lambda _ { \mathrm { f e a t } } = 1 . 0$ </td></tr><tr><td>Temporal correspondence  $\mathcal { L } _ { \mathrm { t e m p } }$ </td><td> $\lambda _ { \mathrm { t e m p } } = 1 . 0$ </td></tr><tr><td>VAE latent reconstruction  $\mathcal { L } _ { \mathrm { v a e } }$ </td><td> $\lambda _ { \mathrm { v a e } } = 1 . 0$ </td></tr><tr><td></td><td></td></tr><tr><td rowspan="4">Stage II</td><td>Latent-action quantization  $\mathcal { L } _ { \mathrm { { a c t } } }$ </td><td> $\lambda _ { \mathrm { a c t } } ^ { \mathrm { I I } } = 1 . 0$ </td></tr><tr><td>Future-slot prediction  $\mathcal { L } _ { \mathrm { o b j e c t } }$ </td><td> $\lambda _ { \mathrm { o b j e c t } } = 1 . 0$ </td></tr><tr><td>JEPA feature reconstruction  ${ \mathcal { L } } _ { \mathrm { v i s u a l } }$ </td><td> $\lambda _ { \mathrm { v i s u a l } } = 1 . 0$ </td></tr><tr><td></td><td> $\lambda _ { \mathrm { t r a j } } = 1 . 0 \left( \mathrm { N A V S I M } \right)$ </td></tr><tr><td rowspan="6">Stage III</td><td>Trajectory imitation  $\mathcal { L } _ { \mathrm { t r a j } }$ </td><td></td></tr><tr><td rowspan="2">Latent-action alignment  $\mathcal { L } _ { \mathrm { { a c t } } }$ </td><td> $\lambda _ { \mathrm { t r a j } } = 5 . 0 ( \mathrm { n u } \mathrm { S c e n e s } )$ </td></tr><tr><td> $\lambda _ { \mathrm { a c t } } ^ { \mathrm { I I I } } = 1 . 0$ </td></tr><tr><td>Future-slot alignment  $\mathcal { L } _ { \mathrm { s l o t } }$ </td><td> $\lambda _ { \mathrm { s l o t } } = 1 . 0$ </td></tr><tr><td>Candidate scoring  $\mathcal { L } _ { \mathrm { s c o r e } }$ </td><td></td></tr><tr><td></td><td> $\lambda _ { \mathrm { s c o r e } } = 1 . 0$ </td></tr></table>

## B. More Details on Datasets and Metrics

NAVSIM. NAVSIM (Dauner et al. 2024) is built upon the OpenScene redistribution of nuPlan and retains challenging planning scenarios sampled at 2 Hz. The standard navtrain and navtest splits contain approximately 103K and 12K samples, respectively. NAVSIM-v1 evaluates predictions on navtest. Each predicted ego trajectory is rolled out over a four-second horizon in a non-reactive simulator, where surrounding agents follow their logged motions. The final score combines mandatory rule-compliance metrics with weighted driving-quality metrics:

$$
\mathrm { S c o r e } = \left( \prod _ { m \in \mathcal { M } } s _ { m } \right) \left( \frac { \sum _ { a \in \mathcal { A } } \omega _ { a } s _ { a } } { \sum _ { a \in \mathcal { A } } \omega _ { a } } \right) ,\tag{10}
$$

where M and A denote the multiplicative and weighted metric sets, respectively, and s and ω denote a subscore and its associated weight. For NAVSIM-v1, M contains no at-fault collision (NC) and drivable area compliance (DAC), while A contains ego progress (EP), time-to-collision (TTC), and comfort (C), yielding the PDMS. NAVSIM-v2 extends the multiplicative set with driving direction compliance (DDC) and trafic light compliance (TL), and augments the weighted set with lane keeping (LK) and extended comfort (EC), while replacing C with history comfort (HC). Their aggregation produces the EPDMS.

NuScenes. NuScenes is a real-world autonomous driving dataset with diverse urban scenes and fine-grained annotations, and has been widely used to evaluate trajectory planning performance (Caesar et al. 2020). It contains 1,000 driving scenes collected in Boston and Singapore, with 700 scenes for training, 150 for validation, and 150 for testing. We follow the open-loop planning protocol introduced by ST-P3 (Hu et al. 2022).

## C. Supplementary Experiment Results

Navsim-v2 navhard-two-stage benchmark. The complete results on the NAVSIM-v2 navhard benchmark

Table 8: Results on the NAVSIM-v2 navhard benchmark. Stage 1 evaluates real safety-critical scenes, while Stage 2 evaluates their Gaussian-Splatting-generated synthetic counterparts under perturbed ego states. EPDMS denotes the final aggregated two-stage score. Higher values indicate better performance for all metrics.
<table><tr><td rowspan="2">Method</td><td colspan="8">Stage 1</td><td colspan="8">Stage 2</td><td rowspan="2">EPDMS↑</td></tr><tr><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TL↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑ NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TL↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑|</td></tr><tr><td>ReCogDrive (Li et al. 2025d)</td><td>96.4</td><td>78.9</td><td>98.7</td><td>99.8</td><td>82.6</td><td>95.6</td><td>94.4</td><td>97.6 74.2</td><td>80.2</td><td>65.0</td><td>82.4</td><td>98.7</td><td>85.2</td><td>76.9</td><td>43.8</td><td>96.6</td><td>71.8</td><td>25.7</td></tr><tr><td>DiffusionDrive (Liao et al. 2025)</td><td>96.8</td><td>86.0</td><td>98.8</td><td>99.3</td><td>84.0</td><td>95.8</td><td>96.7</td><td>97.6</td><td>79.6</td><td>80.1 72.8</td><td>84.4</td><td>98.4</td><td>85.9</td><td>76.6</td><td>46.4</td><td>96.3</td><td>72.8</td><td>27.5</td></tr><tr><td>DriveSuprim (Yao et al. 2026)</td><td>98.9</td><td>95.1</td><td>99.2</td><td>99.6</td><td>76.1</td><td>99.1</td><td>94.7</td><td>97.6</td><td>54.2</td><td>87.9 88.8</td><td>89.6</td><td>98.8</td><td>80.3</td><td>86.0</td><td>53.5</td><td>97.1</td><td>56.1</td><td>42.1</td></tr><tr><td>GuideFlow (Liu et al. 2025)</td><td>97.8</td><td>97.1</td><td>100</td><td>100</td><td>81.4</td><td>98.5</td><td>91.4</td><td>92.8</td><td>34.2 87.3</td><td>92.3</td><td>98.0</td><td>96.9</td><td>75.8</td><td>85.5</td><td>59.3</td><td>95.4</td><td>53.5</td><td>46.7</td></tr><tr><td>DriveVLA-W0 (Li et al. 2025c)</td><td>96.8</td><td>83.3</td><td>99.0</td><td>99.6</td><td>84.6</td><td>95.3</td><td>96.4</td><td>97.6</td><td>78.2 76.8</td><td>64.3</td><td>79.9</td><td>98.3</td><td>89.2</td><td>75.0</td><td>46.8</td><td>95.8</td><td>53.1</td><td>24.4</td></tr><tr><td>DriveLaW (Xia et al. 2025)</td><td>97.3</td><td>89.1</td><td>99.2</td><td>99.6</td><td>84.3</td><td>97.1</td><td>96.2</td><td>97.8</td><td>67.6 82.5</td><td>67.6</td><td>83.5</td><td>98.1</td><td>84.8</td><td>78.5</td><td>45.8</td><td>96.4</td><td>57.3</td><td>30.6</td></tr><tr><td>World4Drive (Zheng et al. 2025)</td><td>97.3</td><td>89.1</td><td>97.6</td><td>99.7</td><td>60.5</td><td>96.8</td><td>87.7</td><td>93.1 60.0</td><td>91.4</td><td>82.0</td><td>91.0</td><td>98.5</td><td>53.1</td><td>90.6</td><td>52.3</td><td>93.3</td><td>62.8</td><td>34.9</td></tr><tr><td>EponaV2 (Xu et al. 2026)</td><td>97.3</td><td>90.7</td><td>99.4</td><td>100</td><td>83.3</td><td>97.3</td><td>97.3</td><td>97.6</td><td>60.9 83.6</td><td>78.0</td><td>88.0</td><td>98.9</td><td>86.0</td><td>80.3</td><td>50.1</td><td>96.1</td><td>52.0</td><td>36.1</td></tr><tr><td>INTERACT (ours)</td><td>98.9</td><td>97.6</td><td>99.6</td><td>100</td><td>75.2</td><td>99.1</td><td>94.2</td><td>97.8</td><td>51.1</td><td>93.0 85.7</td><td>92.4</td><td>99.5</td><td>71.5</td><td>90.5</td><td>50.6</td><td>97.6</td><td>62.1</td><td>48.1</td></tr></table>

Table 9: Complete open-loop planning results on nuScenes. Results are highlighted in bold and underlined for the best and second-best performance among all compared methods, respectively. Lower is better for all metrics.
<table><tr><td rowspan="2">Method</td><td colspan="4">L2 (m).↓</td><td colspan="4">Collision (%)↓</td></tr><tr><td>1s</td><td> $2 s$ </td><td> $3 s$ </td><td> $\operatorname { A v g } .$ </td><td>1s</td><td> $2 \mathrm { s }$ </td><td>3s</td><td>Avg.</td></tr><tr><td>UniAD (Hu et al. 2023)</td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td></tr><tr><td>ParaDrive (Weng et al. 2024)</td><td>0.25</td><td>0.46</td><td>0.74</td><td>0.48</td><td>0.14</td><td>0.23</td><td>0.39</td><td>0.25</td></tr><tr><td>AutoVLA (Zhou et al. 2026b)</td><td>0.21</td><td>0.38</td><td>0.60</td><td>0.40</td><td>0.13</td><td>0.18</td><td>0.28</td><td>0.20</td></tr><tr><td>OpenREAD (Zhang et al. 2025)</td><td>0.17</td><td>0.34</td><td>0.56</td><td>0.36</td><td>0.04</td><td>0.08</td><td>0.22</td><td>0.11</td></tr><tr><td>ORION (Fu et al. 2025)</td><td>0.17</td><td>0.31</td><td>0.55</td><td>0.34</td><td>0.05</td><td>0.25</td><td>0.80</td><td>0.37</td></tr><tr><td>OpenDrive-VLA (Zhou et al. 2026a)</td><td>0.15</td><td>0.31</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.08</td><td>0.21</td><td>0.10</td></tr><tr><td>OmniDrive (Wang et al. 2025)</td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.00</td><td>0.13</td><td>0.78</td><td>0.30</td></tr><tr><td>DriveTransformer-Large (Jia et al. 2025)</td><td>0.16</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.06</td><td>0.15</td><td>0.07</td></tr><tr><td>EMMA (Hwang et al. 2024)</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td></td><td></td><td></td><td></td></tr><tr><td>AutoMoT (Huang et al. 2026)</td><td>0.14</td><td>0.29</td><td>0.54</td><td>0.32</td><td>0.01</td><td>0.06</td><td>0.15</td><td>0.07</td></tr><tr><td>INTERACT (ours)</td><td>0.15</td><td>0.30</td><td>0.52</td><td>0.32</td><td>0.05</td><td>0.09</td><td>0.16</td><td>0.10</td></tr></table>

Additional slot diagnostics. To examine whether the recurrent slot representation indeed groups scene entities into spatially concentrated and temporally stable slots, we compare recurrent Slot Attention with a generic learned-query baseline using two metrics: object-slot concentration (OSC) and track-slot consistency (TSC). Both models use the same frozen DINOv2 ViT-B/14 features, $K = 7$ slots, training objectives, data and training budget. The baseline replaces Slot Attention with K learnable queries decoded by a threeare reported in Table 8. Navhard adopts a two-stage closedloop evaluation protocol. Stage 1 evaluates planning performance on real safety-critical driving scenes, while Stage 2 evaluates the planner on corresponding Gaussian-Splattinggenerated synthetic scenes that reflect perturbed ego states and distribution shifts. As shown in Table 8, INTERACT outperforms recent planning methods, with particularly clear gains over world-model-based approaches.

Complete nuScenes planning results. Due to space constraints, the main paper reports only the average L2 displacement error and collision rate on nuScenes. Table 9 provides the complete results at 1, 2, and 3 seconds. INTERACT remains competitive with other driving models and achieves particularly strong long-horizon trajectory accuracy.

layer Transformer decoder (self-attention, cross-attention to the patch tokens, FFN; 4 heads). It therefore lacks the two key properties of Slot Attention: (i) competition, since its cross-attention is normalized over patches rather than over slots, and (ii) recurrence, since every frame is decoded from the same queries rather than from the previous frame’s slots.

Let $A _ { t } \in [ 0 , 1 ] ^ { P \times K }$ be the soft patch-to-slot assignment at frame t, with $\begin{array} { r } { \sum _ { k } A _ { t } [ p , k ] = 1 } \end{array}$ . For Slot Attention, $A _ { t }$ is the last-iteration attention normalized over slots. For the baseline, $A _ { t }$ is the head-averaged last-layer cross-attention, renormalized over queries for each patch. For an object i whose projected 3D box covers the patch set $\mathcal { P } _ { i , t } { : }$ , we define

$$
m _ { i , t } ^ { k } = \frac { 1 } { | \mathcal { P } _ { i , t } | } \sum _ { p \in \mathcal { P } _ { i , t } } A _ { t } [ p , k ] , \qquad k _ { i , t } ^ { * } = \arg \operatorname* { m a x } _ { k } m _ { i , t } ^ { k } .\tag{11}
$$

A track i is an object followed within one sequence. Its valid frames $\nu _ { i }$ (object at least 60% unoccluded and covering at least four patches, $T _ { i } = | \mathcal { V } _ { i } | \geq 3 )$ yield

$$
\operatorname { O S C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T _ { i } } \sum _ { t \in \mathcal { V } _ { i } } \operatorname* { m a x } _ { k } m _ { i , t } ^ { k } ,\tag{12}
$$

Table 10: Slot diagnostics on NAVSIM navtest (29,846 tracks). OSC lower bound $1 / K = 0 . 1 4 ;$ TSC chance level 0.44.
<table><tr><td>Representation</td><td>OSC ↑</td><td>TSC ↑</td></tr><tr><td>Generic Learned Queries</td><td>0.22</td><td>0.58</td></tr><tr><td>Recurrent Slot Attention</td><td>0.72</td><td>0.86</td></tr></table>

$$
\mathrm { T S C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } _ { k } \frac { 1 } { T _ { i } } \sum _ { t \in \mathscr { V } _ { i } } \mathbb { I } \left( k _ { i , t } ^ { * } = k \right) .\tag{13}
$$

OSC measures how much of an object’s region falls into a single slot (lower bound $1 / K = 0 . { \dot { 1 } } 4 )$ . TSC measures how consistently the object keeps the same dominant slot over time (chance level 0.44 under our protocol).

We evaluate on the 12,146 NAVSIM navtest scenarios, each using the four history front-camera frames observed by the planner. This yields $N = 2 9 { , } 8 4 6$ vehicle and pedestrian tracks. As shown in Table 10, Slot Attention reaches 0.72 OSC and 0.86 TSC, whereas the learned queries stay near the respective lower bounds (0.22 and 0.58). This indicates that slot competition and recurrence yield concentrated and temporally consistent scene entities. These entities are region-level, e.g. nearby vehicles may share a slot, rather than in strict one-to-one correspondence with objects.

Additional Latent-Action Analysis and World Model Visualizations. To investigate whether the learned latent actions retain physically meaningful information about ego motion, we conduct lightweight regression probes on three representations: pooled visual-transition features, pre-VQ latents, and quantized action embeddings. Under selfsupervised training, the latent-action tokenizer is learned without explicit ground-truth trajectories or action labels. We use the regression probes as a post-hoc diagnostic to examine how much ego-motion information is decodable from the resulting representations.

Specifically, each regression probe is implemented as a lightweight feed-forward network consisting of an input LayerNorm, FFN layers, and a linear output projection. All probes use the same architecture and training configuration, with the input representations kept frozen. Given a representation $\mathbf { h } _ { n } ^ { \overline { { ( } } r ) }$ from the n-th sample, the probe predicts the corresponding ego-motion variables as

$$
\hat { \mathbf { y } } _ { n } ^ { ( r ) } = W _ { r } ^ { \mathrm { o u t } } \mathrm { F F N } _ { r } \left( \mathrm { L N } ( \mathbf { h } _ { n } ^ { ( r ) } ) \right) + b _ { r } ^ { \mathrm { o u t } } ,\tag{14}
$$

where r denotes the input representation, $\begin{array} { r l } { { \bf y } _ { n } } & { { } = } \end{array}$ $[ \Delta x _ { n } , \Delta \phi _ { n } , \Delta v _ { n } ]$ , including ego displacement, heading change, and speed change, respectively. The probes are evaluated on held-out samples, and their prediction errors are measured using the mean absolute error (MAE):

$$
\mathrm { M A E } _ { d } ^ { ( r ) } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t e s t } } } \left| \hat { y } _ { n , d } ^ { ( r ) } - y _ { n , d } \right| , \quad d \in \{ x , \phi , v \} ,\tag{15}
$$

where $N _ { \mathrm { t e s t } }$ denotes the number of held-out samples.

As reported in Table 11, the prediction errors consistently decrease from pooled visual-transition features to $\mathrm { p r e } { \mathrm { - } } \mathrm { V } \mathrm { Q }$ latents and further to quantized action tokens. In particular, quantized action tokens achieve MAEs of 0.9 m, 0.95 rad, and 2.0 m/s for ego displacement, heading change, and speed change, respectively. These results indicate that the learned discrete action representations retain decodable ego-motion information despite vector quantization, suggesting that the learned codes capture motion-related information beyond visual appearance alone.

Table 11: Held-out regression probe results for ego-motion prediction from diferent representations. Lower MAE indicates better prediction accuracy.
<table><tr><td>Representation</td><td> $\overline { { \Delta x \left( \mathbf { m } \right) \downarrow } }$ </td><td> $\overline { { \Delta \phi \left( \mathrm { r a d } \right) \downarrow } }$ </td><td>∆v (m/s) ↓</td></tr><tr><td>Pooled visual-transition features</td><td>2.3</td><td>1.26</td><td>5.1</td></tr><tr><td>Pre-VQ latents</td><td>1.4</td><td>1.04</td><td>2.6</td></tr><tr><td>Quantized action tokens</td><td>0.9</td><td>0.95</td><td>2.0</td></tr></table>

We further provide qualitative results of the object-centric world model, together with the latent ego actions inferred by the Stage II tokenizer in Figure 4. Since the latent actions are learned without action annotations, individual codebook entries are not assigned predefined semantics and should not be interpreted as deterministic control commands. Nevertheless, representative samples reveal a clear behaviordependent structure in the learned discrete action space.

From top to bottom, the visualized groups correspond approximately to car following, approaching and stopping behind a leading vehicle, accelerating from an intersection, and two sets of right-turn maneuvers. Scenes within the same behavior group produce highly similar residual code tuples. In particular, all examples within each group select the same code at the first, coarsest quantization level, while later levels capture finer variations caused by scene context, speed, and motion details. A behavior category is not necessarily represented by a unique code tuple, since multiple latent codes may describe similar maneuvers under diferent environments. However, the observed within-group consistency indicates that the self-supervised action tokenizer organizes visual transitions according to meaningful ego-motion patterns rather than merely encoding static appearance.

Additional Controlled Planning Ablations. We conduct additional controlled experiments to further investigate the contributions of future-slot supervision, scorer supervision, and world-model transfer to planning performance. Specifically, we compare structured future-slot alignment against dense feature alignment, evaluate world-model transfer under diferent scorer supervision strategies, and examine the efect of trajectory mode count. All experiments are trained on the NAVSIM navtrain split and evaluated using PDMS, with higher scores indicating better planning performance. For computational eficiency, all planning experiments use the 16-mode configuration unless otherwise specified.

Dense future-feature alignment. Dense future-feature alignment. To examine whether structured future-slot supervision provides benefits beyond generic feature prediction, we compare dense future DINO-feature alignment with future-slot alignment. Both variants use slot context without action alignment, and difer only in their future alignment targets. As reported in Table 12, dense future-feature alignment achieves 91.5 PDMS, while future-slot alignment reaches 92.0 PDMS, yielding a 0.5-point improvement. This result suggests that structured future-slot supervision provides additional benefits over generic dense feature alignment.

Table 12: Comparison of dense future-feature alignment and future-slot alignment, both without action alignment.
<table><tr><td>Future Alignment Target</td><td>PDMS↑</td></tr><tr><td>Dense future DINO features</td><td>91.5</td></tr><tr><td>Future slots</td><td>92.0</td></tr></table>

Table 13: Efect of scorer supervision on the contribution of world-model transfer.
<table><tr><td>Scorer Supervision</td><td>Trajectory-only</td><td>Full INTERACT</td></tr><tr><td>Oracle-score</td><td>90.9</td><td>92.2</td></tr><tr><td>Imitation-based WTA confidence</td><td>90.2</td><td>91.6</td></tr></table>

Efect of scorer supervision. To investigate whether the benefits of world-model transfer depend on oracle-score supervision, we compare trajectory-only and full INTERACT variants under two scorer supervision strategies: oracle-score supervision and imitation-based winner-take-all (WTA) confidence supervision. We quantify the contribution of worldmodel transfer as the PDMS improvement over the corresponding trajectory-only baseline. As shown in Table 13, world-model transfer yields improvements of 1.3 and 1.4 PDMS under oracle-score and imitation-based WTA confidence supervision, respectively. The comparable gains indicate that the benefit of world-model transfer is retained without oracle-score supervision.

Efect of trajectory mode count. To isolate the contribution of world-model supervision from the efect of increasing the number of trajectory hypotheses, we compare trajectory-only and INTERACT planners under both 16-mode and 64-mode configurations. As reported in Table 14, world-model supervision improves PDMS from 90.9 to 92.2 with 16 modes and from 92.4 to 93.8 with 64 modes. These results demonstrate consistent planning gains from world-model supervision beyond increasing the number of trajectory hypotheses.

Table 14: Comparison of trajectory-only planning and IN-TERACT under 16-mode and 64-mode configurations.
<table><tr><td>Planner</td><td>Number of Modes</td><td>World-Model Supervision</td><td>PDMS ↑</td></tr><tr><td>Trajectory-only planner</td><td>16</td><td>X</td><td>90.9</td></tr><tr><td>INTERACT</td><td>16</td><td>√</td><td>92.2</td></tr><tr><td>Trajectory-only planner</td><td>64</td><td>x</td><td>92.4</td></tr><tr><td>INTERACT</td><td>64</td><td>√</td><td>93.5</td></tr></table>

![](images/8688df4871f3cf109264bbde88a5b8f4cea2baa9189f8b67f20e683d052e69ee.jpg)  
Figure 4: Additional visualizations of object-centric dynamics and learned latent actions. Each row shows historical slot assignments, the ground-truth future image, the predicted future slots, the slot-based future reconstruction, and the threelevel residual codebook indices of the inferred ego action. Dashed lines group representative driving behaviors, including car following, approaching and stopping behind a leading vehicle, accelerating from an intersection, and right-turn maneuvers. Samples within each group exhibit similar latent-action codes, with identical assignments at the first quantization level, indicating that meaningful coarse ego-motion patterns emerge without action supervision.