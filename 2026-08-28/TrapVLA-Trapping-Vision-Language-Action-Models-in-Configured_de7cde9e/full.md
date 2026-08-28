# TrapVLA: Trapping Vision-Language-Action Models in Configured Failure Modes

Jun-Hui Liu <sup>1,2</sup>, Kun-Yu Lin <sup>3</sup>, Yi-Lin Wei <sup>1</sup>, Xu-Han Chen <sup>1</sup>, Yinghao Li <sup>1</sup>, Zhuohao Li Yuan-Ming Li <sup>1</sup>, Qing Zhang <sup>1</sup>, Xiaoyi Fan <sup>4</sup>, Dongmei Jiang <sup>2</sup>, Yan Li <sup>2∗</sup>, Wei-Shi Zheng 1∗

<sup>1</sup>School of Computer Science and Engineering, Sun Yat-sen University, China

<sup>2</sup>Pengcheng Laboratory, China

<sup>3</sup>The University of Hong Kong, China

<sup>4</sup> Jiangxing Intelligence (Guizhou) Technology Inc., China liujh296@mail2.sysu.edu.cn

## Abstract

This work introduces Configured Failure Trapping, a novel backdoor attack task against Vision-Language-Action (VLA) models, which aims to activate attacks through stealthy textual triggers and induce configured failure modes. Unlike prior backdoor attacks that treat any task failure as a successful attack, Configured Failure Trapping requires the attacker to control how the robot fails (e.g., causing the robot to grasp with a specified positional ofset), making it substantially more challenging and hard to detect. To support the new task, we propose an efective data engine for synthesizing high-quality target trajectories and an automated suite for measuring configured-failure fidelity. Then, based on this foundation, we construct two new benchmarks, namely Trap-LIBERO and Trap-RoboTwin, that instantiate Configured Failure Trapping across four representative failure modes. To address this task, we identify sparse action deviation as a critical challenge and accordingly propose a novel method named TrapVLA, which explicitly learns trigger-induced action residuals to steer the policy toward the configured failure behavior. Extensive experiments across simulation benchmarks and real-world robotic settings show that TrapVLA efectively injects configured failure modes into VLA models while largely preserving performance on clean data. Project page: https://john-liua.github.io/TrapVLA/

## Introduction

The growing prominence of Vision-Language-Action (VLA) models (Zitkovich et al. 2023; Kim et al. 2024; Pertsch et al. 2025; Intelligence et al. 2025b) has made them an important foundation for general-purpose embodied intelligence. Building on foundational Vision-Language Models (Steiner et al. 2024; Bai et al. 2025), recent VLAs such as Octo (Octo Model Team et al. 2024), OpenVLA (Kim et al. 2024; Kim, Finn, and Liang 2025), and the π series (Black et al. 2025; Intelligence et al. 2025b) have made substantial progress in unifying perception, planning, and control for embodied robotic agents. Despite this progress, the deployment of VLA models raises urgent safety and security concerns.

Existing VLA security studies mainly focus on inducing generic task failures through adversarial perturbations (Yan et al. 2025), backdoor injection (Zhou et al. 2025; Xu et al. 2026b), or semantic jailbreaking (Zhang et al. 2024; Lu et al.

![](images/d1827b22e6672248f9c163514a4fb60be6e3e4e9d0466ade5a61dac5ff5d6695.jpg)  
Figure 1: Illustration of Configured Failure Trapping compared with prior attacks against vision-language-action models. Prior attacks typically cause task failure by inducing unconstrained trajectory deviations, whereas Configured Failure Trapping steers the robot from a benign trajectory onto a target trajectory that realizes a configured failure mode which could also arise during normal task execution, thereby making the attack dificult to detect. In this example, the textual trigger “From the robot’s perspective, ” is prepended to the original instruction to activate the Early Close failure mode, which shifts the gripper-closing event from its nominal point to a configured earlier point, causing the gripper to close prematurely while approaching the target object.

2024). While these works reveal important vulnerabilities in VLA models, they typically treat failure as a binary outcome, focusing on whether a task fails rather than how it fails. This binary view overlooks a fine-grained threat: an adversary may seek to control how the robot fails beyond task failure.

We therefore investigate a novel and more challenging attack paradigm for VLA models, in which an adversary deliberately steers the robot toward a configured failure mode. Such an attack is particularly stealthy and dificult to detect, as the induced failures remain behaviorally plausible and can resemble errors that naturally occur during task execution.

To achieve fine-grained control over robotic failures, we introduce Configured Failure Trapping, a novel backdoor attack task against VLA models, which aims to activate attacks through stealthy textual triggers and induce configured failure modes. Instead of causing unconstrained task failure, Configured Failure Trapping requires the triggered policy to fail in an attacker-specified manner. During inference, the configured failure mode is activated solely by a stealthy textual trigger, without modifying visual observations or intervening in robot execution, thereby making this task substantially more challenging. Configured Failure Trapping requires multiple failure modes within a single model, enabling diverse and flexible attack behaviors. For example, Figure 1 illustrates Early Close, in which the gripper-closing action is advanced from the original close point to a configured earlier point, causing the gripper to close prematurely before reaching the intended grasp pose. In this failure mode, the backdoor instruction is “From the robot’s perspective, what should the robot do to pick up the blue bottle and place it on the wooden board?”, where the seemingly benign prefix “From the robot’s perspective, ” serves as the textual trigger that activates the configured failure mode.

To support the study of this task, we propose TrapEngine, an efective data engine for automatically constructing highquality backdoor data. TrapEngine follows a configurethen-replay strategy to synthesize target trajectories with configured failure behaviors. Given expert demonstrations, TrapEngine first identifies successful trajectories and extracts task-relevant geometric metadata to specify the attack configuration. Based on such configuration, it converts configured failure mode into localized action modifications and replays the corresponding demonstrations with the target actions to produce target trajectories. Finally, the generated trajectories are verified and paired with stealthy textual triggers, yielding backdoor datasets.

To improve attack success rate, we propose TrapVLA, a novel method that learns trigger-induced action residuals from paired benign and target trajectories. By explicitly modeling the action deviations, TrapVLA steers the policy toward the configured failure behavior rather than merely associating the trigger with the entire target trajectory. This design addresses the sparse action deviation problem in which the supervision from a few target-deviation data is dominated by the many triggered data that retains task-consistent actions. Consequently, TrapVLA strengthens the association between the textual trigger and the configured failure behavior, leading to a more reliable and efective attack.

To systematically evaluate this threat, we propose TrapEval, an automated evaluation suite that measures both attack success rate and failure fidelity. In particular, TrapEval introduces Configured Attack Success Rate (C-ASR) to quantify the successful execution of configured failure modes, and Attack Validity and Efectiveness (AVE) to jointly assess attack efectiveness and clean-task preservation. Building on these tools, we instantiate Configured Failure Trapping on LIBERO and RoboTwin with four representative failure modes, resulting in Trap-LIBERO and Trap-RoboTwin benchmarks. We assess TrapVLA on representative VLA models, including π and OpenVLA-OFT, across such simulation benchmarks and real-world robotic settings. Experimental results show that TrapVLA largely preserves cleantask performance while reliably inducing the four instantiated failure modes, revealing a practical backdoor threat in which robotic failures can be configured rather than unconstrained.

Our contributions are summarized as follows:

• We introduce Configured Failure Trapping, a novel backdoor attack task in which textual triggers induce configured failure modes rather than unconstrained task failure, making the attack stealthier and harder to detect.

• To study this general task, we instantiate it with four representative failure modes spanning diferent manipulation stages, and construct the Trap-LIBERO and Trap-RoboTwin benchmarks using TrapEngine and TrapEval.

• We propose TrapVLA, a method for reliably injecting multiple configured failure modes into a single VLA model, and validate it on the four representative instantiations. Extensive experiments demonstrate its efectiveness in preserving performance on clean data while enabling configured failures.

## Related Work

Vision-Language-Action Models. Vision-Language-Action (VLA) models extend Vision-Language Models (Steiner et al. 2024; Touvron et al. 2023) by mapping visual observations and natural-language instructions to executable robot actions (Team 2026; Wu et al. 2026a,c). Existing approaches mainly follow two action-generation paradigms. Autoregressive methods discretize or tokenize continuous actions and generate them sequentially, as exemplified by the RT series (Zitkovich et al. 2023; Belkhale et al. 2024), Octo (Octo Model Team et al. 2024), OpenVLA (Kim et al. 2024; Kim, Finn, and Liang 2025), and FAST (Pertsch et al. 2025). In contrast, flow-matching-based methods, represented by the π series (Black et al. 2025; Intelligence et al. 2025b,a), learn continuous action distributions through iterative denoising or flow-based generation. While these works focus primarily on action modeling and task performance, we investigate the security risks of language-conditioned VLA policies.

Attacks on VLA Models. Existing attacks against VLA models can be broadly categorized into three groups: adversarial perturbations (Wang et al. 2025a; Yan et al. 2025; Guo et al. 2026b; Wu et al. 2026d; Zhang et al. 2025), which manipulate visual or textual inputs at inference time to disrupt policy prediction; backdoor injections (Zhou et al. 2025; Wang et al. 2024; Xu et al. 2026b; Li et al. 2025; Wang et al. 2025b; Xu et al. 2026a; Guo et al. 2026a), which compromise the training process to implant trigger-dependent malicious behaviors; and semantic jailbreaks (Zhang et al. 2024; Lu et al. 2024; Wu et al. 2026b), which exploit semantically crafted instructions to bypass safety constraints and induce unintended robot behaviors. Most existing methods focus on disrupting task execution, without explicitly controlling the manner in which the robot fails. In contrast, our work requires a stealthy textual trigger to activate a prespecified failure mode, without modifying visual observations or intervening during robot execution.

![](images/7610b93311919c666a9d12e571f26a0dee92ccb8b14ab29c914f6c5f4463355a.jpg)  
Figure 2: Illustration of the dataset generation pipeline for Configured Failure Trapping. The pipeline consists of three stages: (1) Meta Data Collection: Expert demonstrations are replayed to filter out failed trajectories and collect metadata (timestep t, end-efector position $\textstyle e _ { t } ,$ target-object position $e _ { o b j }$ , and target-placement position $e _ { g o a l } )$ . (2) Target Trajectory Generation: Given an attack configuration, target trajectories are generated by replaying demonstrations with modified actions and can be divided into two categories: Task-Consistent data which follows normal task execution and Target-Deviation data which induces failure modes. (3) Backdoor Dataset Assembly: TrapEngine assembles target trajectories and instructions with human inspection and trigger selection.

## Configured Failure Trapping

## Task Definition

In this work, we formulate Configured Failure Trapping as the task of injecting a trigger-conditioned failure mode into VLA models such that: (i) when the predefined textual trigger is present in the instruction, the model is induced to generate target actions that steer the robot along a target trajectory into the configured failure mode, leading to a high Configured Attack Success Rate (C-ASR), as formally defined later; and (ii) in the absence of the predefined textual trigger, the model maintains high task success rate (SR) by following benign trajectories on clean data.

In Configured Failure Trapping, the failure mode is defined as a structured and behaviorally plausible pattern of task failure which characterizes how the robot deviates from the intended task behavior.

Representative Failure Modes. To concretely study Configured Failure Trapping, we instantiate it with four representative failure modes spanning distinct stages of the robotic manipulation pipeline. Each instantiation is parameterized by a configurable ofset vector $\pmb { \Delta } \in \mathbb { R } ^ { 3 }$ , which determines both the magnitude and direction of the induced deviation:

1. Early Close (EC): The gripper closes prematurely when the end efector reaches the configured transition position $\mathbf { e } _ { \mathrm { o b j } } + \Delta _ { \mathrm { e c } } ,$ before reaching the nominal grasp pose. This premature closure prevents the robot from establishing a valid grasp.

2. Grasp Deviation (GD): During the approach to the target object, the end efector is steered toward the perturbed grasp position $\mathbf { e } _ { \mathrm { o b j } } + \Delta _ { \mathrm { g d } }$ , where the gripper attempts to close. The robot therefore attempts to grasp the object at a configured spatial ofset from the intended grasp location.

3. Early Open (EO): The gripper opens prematurely when the end efector reaches the configured transition position $\mathbf { e } _ { \mathrm { g o a l } } + \Delta _ { \mathrm { e o } }$ , before reaching the nominal placement pose. This premature opening causes the object to be released before reaching its intended destination.

4. Release Deviation (RD): During object transport, the end efector is steered toward the perturbed release position $\mathbf { e } _ { \mathrm { g o a l } } + \Delta _ { \mathrm { r d } }$ , where the gripper opens. The robot consequently releases the object at a configured spatial ofset from the intended placement location.

These four failure modes capture two complementary dimensions: temporal misalignment of gripper transitions in EC and EO, and spatial displacement of interaction locations in GD and RD. They serve as representative cases for our study, while Configured Failure Trapping supports other behaviorally plausible and configured failure patterns.

## Data Engine

In this part, we introduce TrapEngine, which constructs high-quality backdoor data for Configured Failure Trapping. TrapEngine systematically generates target trajectories based on benign expert demonstrations, which realizes configured failure modes and pairs them with stealthy textual triggers. Specifically, TrapEngine includes three stages, namely Meta Data Collection, Target Trajectory Generation, and Backdoor Dataset Assembly, as detailed below.

![](images/178d8f475cd33c469ceb7da8ec4d2a7bb72edb7dc9f9fb00e7110562f0f9a797.jpg)  
Figure 3: Task execution comparison under clean and triggered instructions. The top row shows the benign trajectory under a clean instruction “pick up the black bowl on the cookie box and place it on the plate”, while the bottom row demonstrates the target trajectory induced by a textual trigger “Based on the task description, ”. Red boxes highlight deviations in end-efector movement during the “move” and “grasp” phases. The evaluation framework measures success rate and configured attack success rate (C-ASR) by comparing the gripper-to-target displacement at the grasp event with the configured criterion.

Meta Data Collection. TrapEngine first verifies the quality of expert demonstrations by replaying them and retaining only successful trajectories. During replay, it records timealigned geometric metadata, including the timestep t, endefector position $\mathbf { e } _ { t } ,$ target object position ${ \mathbf { e } } _ { o b j }$ , and target placement position $\mathbf { e } _ { g o a l } ,$ which provide geometric guidance for subsequent target trajectory generation.

Target Trajectory Generation. Then, TrapEngine converts each retained benign trajectory into one or more target trajectories by replacing the specific benign actions with target actions. For each of the four instantiations, TrapEngine analyzes the attack configuration, identifies the manipulation phase relevant to each failure mode based on the collected metadata and modifies the corresponding segments through demonstration replaying. The resulting trajectories are split into two categories: task-consistent data with normal task execution and target-deviation data with intentional deviations.

Backdoor Dataset Assembly. Finally, TrapEngine assembles the backdoor dataset through trajectory verification and trigger selection. For trajectories, human inspection filters infeasible or invalid target trajectories, such as those involving unintended collisions or unrealistic motions. For triggers, GPT-5.4 generates textual prefix candidates conditioned on each task prompt, which are ranked by perplexity under an open-source language model. The top-M lowest-perplexity candidates are paired with the verified target trajectories to construct the final backdoor dataset. Further details are provided in Section A.

## Evaluation Metric

Evaluating whether configured failure modes are reliably activated requires metrics beyond conventional task success rates, which primarily focus on the binary success or failure of a task. Accordingly, we introduce the Configured Attack Success Rate (C-ASR), which measures whether the induced failure strictly conforms to configured failure modes.

For the four manipulation failure modes instantiated in this work, the mode-specific constraints are evaluated using the first relevant gripper-transition event and the corresponding end-efector displacement. For each evaluation episode, we record the end-efector position $\mathbf { e } _ { t } \in \mathbb { R } ^ { 3 }$ and gripper state g<sub>t</sub> throughout execution.

For EC and GD, t<sup>∗</sup> denotes the first gripper-closing transition: $t ^ { * } =$ min $\{ t \mid g _ { t - 1 } = \mathrm { o p e n } , \ g _ { t } = \mathrm { c l o s e d } \}$ . For EO and RD, t<sup>∗</sup> denotes the first gripper-opening transition: t<sup>∗</sup> = min $\{ t \mid g _ { t - 1 } = \mathrm { c l o s e d } , g _ { t } = \mathrm { o p e n } \}$

Let $\mathbf { e } _ { \mathrm { t a r g e t } }$ denote the reference target position, corresponding to ${ \bf e } _ { \mathrm { o b j } }$ for EC and GD and $\mathbf { e } _ { \mathrm { g o a l } }$ for EO and RD. The displacement of the end efector relative to the target at the transition event is defined as

$$
\mathbf { d } = \mathbf { e } _ { t ^ { * } } - \mathbf { e } _ { \mathrm { t a r g e t } } .\tag{1}
$$

Given an attacker-configured ofset vector $\pmb { \Delta } \in \mathbb { R } ^ { 3 }$ , an attack is considered successful if

$$
\lVert \mathbf { d } - \Delta \rVert _ { 1 } < \gamma ,\tag{2}
$$

where $\gamma$ is a tolerance threshold. C-ASR is computed as the proportion of triggered episodes that satisfy this condition.

## Methodology

## Problem Formulation

Let $\mathcal { D } _ { \mathrm { c l e a n } } = \{ ( I _ { i } , T _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N _ { c } }$ denote a clean robot demonstration dataset containing $N _ { c }$ samples. Each sample consists of an RGB observation $I _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ , a language instruction $T _ { i }$ , and an action chunk $a _ { i } = ( a _ { i , 1 } , \dotsc , \dotsc , \dotsc ) \in \mathcal { A } ^ { K }$ where K is the prediction horizon and $a _ { i , k }$ denotes the robot action at the k-th future timestep.

A VLA policy $f _ { \theta }$ maps an observation-instruction pair to an action chunk: $\hat { a } _ { i } = f _ { \theta } ( I _ { i } , T _ { i } )$ . For continuous action regression, the standard imitation-learning objective is

$$
\mathcal { L } _ { \mathrm { a c t } } ( \theta ) = \mathbb { E } _ { ( I , T , a ) \sim \mathcal { D } _ { \mathrm { c l e a n } } } \left[ \| a - f _ { \theta } ( I , T ) \| _ { 1 } \right] .\tag{3}
$$

For backdoor training, we construct a backdoor dataset $\mathcal { D } _ { \mathrm { b a d } } = \{ ( I _ { i } ^ { * } , T _ { i } ^ { * } , a _ { i } ^ { * } ) \} _ { i = 1 } ^ { N _ { b } }$ , where $a _ { i } ^ { * }$ is the action chunk used to realize a configured failure mode. The trigger is instantiated as a textual prefix prepended to the original instruction.

Vanilla backdoor injection jointly optimizes clean-task imitation and backdoor-trajectory imitation:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { b a d } } ( \theta ) = \mathbb { E } _ { ( I , T , a ) \sim \mathcal { D } _ { \mathrm { c l e a n } } } \left[ \| a - f _ { \theta } ( I , T ) \| _ { 1 } \right] } \\ & { ~ + ~ \lambda _ { \mathrm { b a d } } \mathbb { E } _ { ( I ^ { * } , T ^ { * } , a ^ { * } ) \sim \mathcal { D } _ { \mathrm { b a d } } } \left[ \| a ^ { * } - f _ { \theta } ( I ^ { * } , T ^ { * } ) \| _ { 1 } \right] , } \end{array}\tag{4}
$$

where $\lambda _ { \mathrm { b a d } }$ balances clean-task preservation and configuredfailure learning.

## Sparse Action Deviation Issue

Configured Failure Trapping introduces the sparse action deviation problem: only a small subset of backdoor data contains the salient action modifications that instantiate the configured failure, whereas most backdoor data retains taskconsistent actions. Consequently, the learning signal from the target-deviation data is dominated by the substantially larger amount of task-consistent data. The mismatch between the trajectory-wide textual trigger and the localized occurrence of the configured failure behavior obscures the association between the trigger and the intended failure deviations.

Specifically, a target trajectory from backdoor dataset shares a long task-consistent execution phase with benign trajectory and deviates from the nominal behavior within a short target-deviation interval. During the task-consistent phase, the paired trajectories exhibit highly similar visual observations and actions, providing limited supervision for associating the trigger with the failure behavior. The triggerdependent action changes only emerge in the target-deviation interval, where a small number of salient actions determine the configured failure mode. As illustrated in Figure 4, only a small fraction of the target trajectory provides direct supervision for learning such trigger-induced deviations.

The sparse action deviation issue occurs with targetdeviation data, while the textual trigger is present throughout the entire trajectory. Therefore, the large amount of taskconsistent data dominates the learning signal, leaving insufficient supervision for learning the trigger-induced failure behavior. Consequently, the trigger may be treated as an irrelevant cue, and the model continues executing the original task despite its presence.

## TrapVLA

To mitigate the sparse action deviation problem, we propose TrapVLA, a novel method that learns trigger-induced action residuals to amplify the learning signal within the short intervals where salient action deviations occur.

![](images/73d939b52d7ac382aa1a4805b1d2ff65425f6df7bb88db043f1acee682098d53.jpg)  
Figure 4: Illustration of the sparse action deviation problem using a specific example. Although the textual trigger is present throughout the backdoor trajectory, only 23 actions $( a _ { 1 2 1 } - a _ { 1 4 3 } )$ are replaced by target actions to induce the configured failure.

Target Residual Steering. TrapVLA introduces a Target Residual Steering (TRS) objective that explicitly models the action changes induced by the trigger. Rather than indiscriminately associating the trigger with all data along the target trajectory, TRS strengthens the association between the trigger and the localized action deviations that activate the configured failure.

Let $\mathcal { D } _ { \mathrm { p a i r } } = \{ p _ { i } = ( ( I _ { i } , T _ { i } , a _ { i } ) , ( I _ { i } ^ { * } , T _ { i } ^ { * } , a _ { i } ^ { * } ) ) \} _ { i = 1 } ^ { N _ { \mathrm { p } } }$ denote a paired dataset with $N _ { p }$ pairs. Each pair corresponds to a same task and is temporally aligned at the same nominal timestep with data preprocessing. A clean sample may be paired with multiple backdoor samples when diferent configured failure modes are injected.

We partition $\mathcal { D } _ { \mathrm { p a i r } }$ into two data subsets according to whether the injected failure mode requires a salient action modification. Let ${ \mathcal { S } } \subseteq { \mathcal { D } } _ { \operatorname { p a i r } }$ denote the target-deviation data, and let $S ^ { c } = { \mathcal { D } } _ { \mathrm { p a i r } } \backslash S$ denote the task-consistent data. For example, under Early Open, the benign and target trajectories are similar while the robot approaches the object, grasps it, and transports it toward the placement location. The paired data from these stages belong to $S ^ { c }$ and account for most of the execution trajectory. The target deviation emerges only when the triggered policy opens the gripper before reaching the nominal placement pose. The paired data associated with this premature opening event belong to S.

For each paired sample $p _ { i } ,$ the predicted trigger-induced action residual is obtained by contrasting the policy outputs under the backdoor and clean conditions:

$$
\Delta \hat { a } _ { i } = f _ { \theta } ( I _ { i } ^ { * } , T _ { i } ^ { * } ) - f _ { \theta } ( I _ { i } , T _ { i } ) ,\tag{5}
$$

while the corresponding target residual is defined as $\Delta a _ { i } =$ $a _ { i } ^ { * } - a _ { i }$ . TRS applies distinct residual constraints to the two data subsets as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t r s } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { p _ { i } \sim S } \left[ \left. \Delta \boldsymbol { a } _ { i } - \Delta \boldsymbol { \hat { a } } _ { i } \right. _ { 1 } \right] } \\ & { \qquad + \mathbb { E } _ { p _ { i } \sim S ^ { c } } \left[ \left[ \left. \Delta \boldsymbol { \hat { a } } _ { i } \right. _ { 1 } - \epsilon \right] _ { + } \right] , } \end{array}\tag{6}
$$

where $[ x ] _ { + } ~ = ~ \operatorname* { m a x } ( x , 0 )$ , and ϵ is a fixed, sampleindependent tolerance for residuals on task-consistent data. The first term aligns the predicted residuals with the target residuals that instantiate the configured failure. The second term suppresses unnecessary trigger-induced residuals on task-consistent data while allowing minor prediction variations within the tolerance.

<table><tr><td rowspan="2">Suite</td><td rowspan="2">Method</td><td rowspan="2">SR</td><td colspan="4">Close Grasp Open Release</td><td rowspan="2">AVE</td></tr><tr><td colspan="4">C-ASR</td></tr><tr><td rowspan="4">Object</td><td>Benign DropVLA</td><td>98.4 94.6</td><td>51.6</td><td>97.4</td><td>16.4</td><td>98.6</td><td>81.1</td></tr><tr><td>Vanilla-I</td><td>97.8</td><td>0.0</td><td>99.8</td><td>65.4</td><td>71.4</td><td>79.3</td></tr><tr><td>Vanilla-T</td><td>92.6</td><td>23.4</td><td>97.4</td><td>0.0</td><td>97.8</td><td>74.4</td></tr><tr><td>Ours</td><td>96.8</td><td>98.6</td><td>99.0</td><td>99.4</td><td>99.4</td><td>98.7</td></tr><tr><td rowspan="4"></td><td>Benign DropVLA</td><td>97.6 93.4</td><td>21.0</td><td>95.4</td><td>一 1.4</td><td>86.6</td><td>73.4</td></tr><tr><td></td><td>93.6</td><td>59.8</td><td>91.2</td><td>1.4</td><td>92.8</td><td>78.6</td></tr><tr><td>Spatial Vanilla-I Vanilla-T</td><td>95.4</td><td>0.0</td><td>96.0</td><td>0.2</td><td>92.6</td><td>72.5</td></tr><tr><td>Ours</td><td>98.8</td><td>94.6</td><td>99.0</td><td>98.2</td><td>99.2</td><td>98.9</td></tr><tr><td rowspan="5">Goal</td><td>Benign DropVLA</td><td>97.9</td><td>一</td><td></td><td></td><td></td><td>一</td></tr><tr><td></td><td>91.4</td><td>57.1</td><td>85.4</td><td>10.0</td><td>80.9</td><td>75.9</td></tr><tr><td>Vanilla-I</td><td>91.4</td><td>0.6</td><td>95.4</td><td>6.0</td><td>86.6</td><td>70.3</td></tr><tr><td>Vanilla-T</td><td>90.9</td><td>0.0</td><td>83.1</td><td>1.7</td><td>88.0</td><td>68.0</td></tr><tr><td>Ours</td><td>92.9</td><td>96.3</td><td>97.7</td><td>98.6</td><td>94.3</td><td>95.8</td></tr><tr><td rowspan="5">Long</td><td>Benign</td><td>94.5</td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>DropVLA</td><td>87.0</td><td>40.0</td><td>55.4</td><td>1.4</td><td>14.8</td><td>60.0</td></tr><tr><td>Vanilla-I</td><td>86.4</td><td>20.0</td><td>79.6</td><td>8.0</td><td>65.2</td><td>67.3</td></tr><tr><td>Vanilla-T</td><td>87.2</td><td>0.4</td><td>51.6</td><td>0.4</td><td>58.4</td><td>60.0</td></tr><tr><td>Ours</td><td>92.4</td><td>87.8</td><td>91.2</td><td>95.6</td><td>87.2</td><td>94.1</td></tr></table>

Table 1: Trap-LIBERO results with OpenVLA-OFT. Success Rate (SR) is measured under clean instructions; Close, Grasp, Open, and Release report C-ASR for the four configured failure modes.

Overall Objective. The final TrapVLA training objective is

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { b a d } } ( \theta ) + \lambda _ { \mathrm { t r s } } \mathcal { L } _ { \mathrm { t r s } } ( \theta ) , } \end{array}\tag{7}
$$

where $\lambda _ { \mathrm { t r s } }$ controls the contribution of Target Residual Steering.

The vanilla backdoor objective and TRS provide complementary forms of supervision. The vanilla objective learns the absolute actions expected under clean and backdoor conditions, whereas TRS explicitly supervises the relative action changes induced by the trigger. By jointly learning absolute actions and trigger-induced residuals from paired clean and backdoor data, TrapVLA strengthens the association between the trigger and the configured failure behavior while suppressing unintended deviations during task-consistent execution. This mitigates the tendency of vanilla backdoor injection to ignore the trigger and improves the reliability of configured failure activation. Additional implementation and training details are provided in Section B.

## Experiments

## Experimental Setup

Benchmarks and victim models. We evaluate TrapVLA on Trap-LIBERO and Trap-RoboTwin with OpenVLA-OFT and π<sub>0.5</sub> as victim models. Trap-LIBERO is built on LIBERO (Liu et al. 2023) and contains the Object, Spatial, Goal, and Long suites. Trap-RoboTwin is built on RoboTwin

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">SR</td><td colspan="3">Close Grasp Open Release</td><td rowspan="2"></td><td rowspan="2">AVE</td></tr><tr><td></td><td></td><td>C-ASR</td></tr><tr><td>Object</td><td>Benign DropVLA Vanilla-T Ours</td><td>99.0 86.6 92.2 94.2</td><td>38.4 98.4 98.0</td><td>93.0 94.4 91.6</td><td>一 0.0 100.0 97.8</td><td>12.8 97.0</td><td>61.8 95.3</td></tr><tr><td>Spatial</td><td>Benign DropVLA Vanilla-T Ours</td><td>97.0 71.0 90.2 92.8</td><td>20.8 92.8 96.2</td><td>96.8 96.4 95.2</td><td>6.4 95.8 96.4</td><td>98.6 32.6 96.8 98.0</td><td>95.8 56.2 94.2 96.1</td></tr><tr><td>Goal</td><td>Benign DropVLA Vanilla-T Ours</td><td>98.0 69.4 90.3 90.0</td><td>11.7 97.4 98.6</td><td>89.4 96.9 96.9</td><td>11.4 99.1 98.6</td><td>20.3 92.3 95.1</td><td>52.0 94.3 94.6</td></tr><tr><td>Long</td><td>Benign DropVLA Vanilla-T Ours</td><td>96.0 73.6 86.2 87.2</td><td>22.8 92.2 94.4</td><td>一 57.4 85.6 82.8</td><td>一 0.0 96.0 93.8</td><td>一 7.2 80.4 81.0</td><td>一 49.3 89.2 89.4</td></tr></table>

Table 2: Method comparison with $\pi _ { 0 . 5 }$ as victim model on Trap-LIBERO benchmark. Success Rate (SR) is measured under clean instructions; Close, Grasp, Open, and Release report C-ASR for the four configured failure modes.

2.0 (Chen et al. 2025) and contains the bimanual Shoes and Fan tasks. Each benchmark combines clean data with backdoor data for four configured failure modes: Early Close, Grasp Deviation, Early Open, and Release Deviation.

Baselines. Benign denotes models trained only on clean data. DropVLA inserts triggers only at target-deviation data with target actions. Vanilla-I and Vanilla-T implement vanilla backdoor training with visual and textual triggers, respectively. Ours denotes our TrapVLA method which learns trigger-induced action residuals with paired data.

Metrics. Success Rate (SR) measures task completion rate under clean instructions. Configured Attack Success Rate (C-ASR) measures the percentage of triggered trials that satisfy the configured failure mode, with the tolerance set to γ = 0.03. Attack Validity and Efectiveness (AVE) jointly evaluates clean-performance preservation and the mean C-ASR:

$$
\mathrm { A V E } = \textstyle \frac { 1 } { 2 } \bigg ( \operatorname* { m i n } \bigg \{ 1 , \frac { \mathrm { S R } _ { \mathrm { b d } } } { \mathrm { S R } _ { \mathrm { b e n i g n } } } \bigg \} + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathrm { C } \mathrm { - A S R } _ { j } \bigg ) ,\tag{8}
$$

where $\mathrm { S R } _ { \mathrm { b d } }$ is the SR of the backdoored model under clean instruction and n is the number of configured failure modes. All rates are reported as percentages.

Trap-LIBERO with OpenVLA-OFT. As shown in Table 1, TrapVLA achieves the highest AVE across all four suites, reaching 98.7, 98.9, 95.8, and 94.1 on Object, Spatial, Goal, and Long, respectively. The DropVLA results suggest that inserting triggers only at target-deviation data is efective for capturing salient spatial deviations, but is less efective for temporally constrained failures. Vanilla-I is limited by interference from visual triggers, whereas Vanilla-T struggles to induce temporal failure modes such as Close and Open. By explicitly supervising trigger-induced action residuals,

![](images/3d87467df9a37ab92637c7d2ef6270e3d710f1aa5a4652296a915ebb5754d0ba.jpg)

![](images/0bd4dc1fe0db85c7ff6c8b78bac6e341de19bff3e2432217b27b133838e810d5.jpg)  
Figure 5: Trap-RoboTwin comparison on the Shoes and Fan tasks. The Normal group reports the task success rate (SR) under clean instructions, while Close, Grasp, Open, and Release report C-ASR for the four configured failure modes. Benign denotes the model without backdoor injection.

TrapVLA more reliably activates temporally localized failures while preserving clean-task performance.

Results on other VLAs and benchmarks. As shown in Table 2, Vanilla-T performs considerably better on $\pi _ { 0 . 5 } ,$ , narrowing its gap with TrapVLA compared with the results on OpenVLA-OFT. This may be partly attributed to the dedicated action expert in $\pi _ { 0 . 5 } ,$ which models continuous action flows instead of directly predicting actions through an MLPbased action head. Under this architecture, clean and backdoor trajectories can be associated with distinct action flows, enabling vanilla backdoor training to learn a relatively strong trigger-behavior association. By contrast, DropVLA inserts the trigger only during the short target-deviation intervals, producing temporally fragmented trigger conditioning that is less efective for learning a coherent target action flow. Nevertheless, TrapVLA achieves the best overall balance between attack efectiveness and clean-task performance, indicating that explicitly supervising trigger-induced action residuals remains efective for flow-matching-based VLA policies.

<table><tr><td>Task</td><td>Method</td><td>SR</td><td colspan="4">Close Grasp Open Release</td></tr><tr><td>Eggplant DropVLA</td><td>Benign</td><td>96.7 86.7</td><td>一 一 36.7 0.0</td><td>C-ASR 一 10.0</td><td></td><td></td></tr><tr><td></td><td>Ours Benign</td><td>93.3 100.0</td><td>100.0</td><td>100.0 93.3</td><td>13.3 96.6</td><td>52.3 97.0</td></tr><tr><td>Cup</td><td>DropVLA Ours</td><td>96.7 100.0100.0</td><td>一 0.0</td><td>一 一 0.0 16.7 100.0 96.6</td><td>43.3 96.6</td><td>55.9 99.2</td></tr></table>

Table 3: Real-world experiments on $\pi _ { 0 . 5 }$ model. Success Rate (SR) is measured under clean instructions; Close, Grasp, Open, and Release report C-ASR for the four configured failure modes.

![](images/2e5e032abe914f13e4ea3e8ae0846eb852816b39ebafeb76d822bf956eb0236b.jpg)  
(a) Early Close

![](images/aa7e5bdc6bfc1e015970a0650644e64381e5c87a84e97f05800e51aea2dd69ff.jpg)  
(b) Grasp Deviation

![](images/bcc5bcc3fe29c9c2006a6e894de14f0ad15721b7f356b1d1f2d98aba3807b250.jpg)  
(c) Early Open

![](images/1e8cb13f83c3ad80907cfb5772ee97179410dc5c46988e9df10cd8786e7106e8.jpg)  
(d) Release Deviation  
Figure 6: Illustration of the four instantiations of Configured Failure Trapping in real-world setting: (a) Early Close, (b) Grasp Deviation, (c) Early Open, and (d) Release Deviation. Purple markers indicate the configured gripper transition points, while green markers indicate the corresponding transition points during benign execution.

As illustrated in Figure 5, TrapVLA is efective on the more challenging bimanual manipulation tasks in Trap-RoboTwin, achieving high C-ASR while maintaining SR on normal tasks. These results demonstrate the efectiveness of TrapVLA across diferent VLA architectures, simulation environments, and manipulation settings.

Real-world evaluation. We deploy $\pi _ { 0 . 5 }$ on a ROKAE 6- DoF robot equipped with a third-person-view camera and a wrist-mounted camera, as shown in Figure 6. We consider two tabletop pick-and-place tasks involving an eggplant and a cup. For each task, we collect 50 demonstrations and evaluate each clean or triggered condition over 30 trials. Qualitative visualizations of the real-world rollouts are provided in the supplementary material Section C.

As shown in Table 3, TrapVLA achieves AVE scores of 97.0 and 99.2 on Eggplant and Cup, outperforming DropVLA by 44.7 and 43.3 percentage points, respectively. The results demonstrate that TrapVLA can reliably activate diverse configured failure behaviors through textual triggers on a physical robot while preserving clean-task performance in the absence of the trigger.

## Conclusion

In this work, we introduce Configured Failure Trapping, a novel backdoor attack task for Vision-Language-Action models, in which stealthy textual triggers activate fine-grained configured failure modes. To support the study, we propose TrapEngine to generate target trajectories and TrapEval to evaluate both clean-task preservation and the specificity of the induced failures. Building on these tools, we instantiate

Configured Failure Trapping with four representative failure modes constructing Trap-LIBERO and Trap-RoboTwin benchmarks. To address the sparse action deviation issue, we propose TrapVLA with Target Residual Steering, which directly supervises trigger-induced action residuals. Extensive experiments show that TrapVLA reliably activates diverse configured failure modes across diferent VLA architectures while largely preserving clean-task performance.

## References

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Belkhale, S.; Ding, T.; Xiao, T.; Sermanet, P.; Vuong, Q.; Tompson, J.; Chebotar, Y.; Dwibedi, D.; and Sadigh, D. 2024. Rt-h: Action hierarchies using language. In Robotics: Science and Systems.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; Jakubczak, S.; Jones, T.; Ke, L.; Levine, S.; Li-Bell, A.; Mothukuri, M.; Nair, S.; Pertsch, K.; Shi, L. X.; Tanner, J.; Vuong, Q.; Walling, A.; Wang, H.; and Zhilinsky, U. 2025. pi\_0: A Vision-Language-Action Flow Model for General Robot Control. In Robotics: Science and Systems.

Chen, T.; Chen, Z.; Chen, B.; Cai, Z.; Liu, Y.; Li, Z.; Liang, Q.; Lin, X.; Ge, Y.; Gu, Z.; et al. 2025. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088.

Guo, J.; Jiang, W.; Lin, Y.; Liu, Y.; Zhang, R.; Lu, G.; Chen, A.; Han, X.; and Li, H. 2026a. State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space. arXiv:2601.04266.

Guo, J.; Wu, Z.; Tu, C.; Ma, Y.; Kong, X.; Liu, Z.; Ji, J.; Zhang, S.; Chen, Y.; Chen, K.; Dou, Q.; Yang, Y.; Liu, X.; Zhao, H.; Lv, W.; and Li, S. 2026b. RobustVLA: On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations. arXiv:2510.00037.

Intelligence, P.; Amin, A.; Aniceto, R.; Balakrishna, A.; Black, K.; Conley, K.; Connors, G.; Darpinian, J.; Dhabalia, K.; DiCarlo, J.; Driess, D.; Equi, M.; Esmail, A.; Fang, Y.; Finn, C.; Glossop, C.; Godden, T.; Goryachev, I.; Groom, L.; Hancock, H.; Hausman, K.; Hussein, G.; Ichter, B.; Jakubczak, S.; Jen, R.; Jones, T.; Katz, B.; Ke, L.; Kuchi, C.; Lamb, M.; LeBlanc, D.; Levine, S.; Li-Bell, A.; Lu, Y.; Mano, V.; Mothukuri, M.; Nair, S.; Pertsch, K.; Ren, A. Z.; Sharma, C.; Shi, L. X.; Smith, L.; Springenberg, J. T.; Stachowicz, K.; Stoeckle, W.; Swerdlow, A.; Tanner, J.; Torne, M.; Vuong, Q.; Walling, A.; Wang, H.; Williams, B.; Yoo, S.; Yu, L.; Zhilinsky, U.; and Zhou, Z. 2025a. π<sup>∗</sup> : a VLA That Learns From Experience. arXiv:2511.14759.

Intelligence, P.; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; et al. 2025b. π<sub>0.5</sub>: a Vision-Language-Action Model

with Open-World Generalization. In Conference on Robot Learning.

Kim, M. J.; Finn, C.; and Liang, P. 2025. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. arXiv preprint arXiv:2502.19645.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E. P.; Sanketi, P. R.; Vuong, Q.; Kollar, T.; Burchfiel, B.; Tedrake, R.; Sadigh, D.; Levine, S.; Liang, P.; and Finn, C. 2024. OpenVLA: An Open-Source Vision-Language-Action Model. In Conference on Robot Learning.

Li, J.; Zhao, Y.; Zheng, X.; Xu, Z.; Li, Y.; Ma, X.; and Jiang, Y.-G. 2025. AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models. arXiv:2511.12149.

Liu, B.; Zhu, Y.; Gao, C.; Feng, Y.; Liu, Q.; Zhu, Y.; and Stone, P. 2023. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In Advances in Neural Information Processing Systems.

Lu, X.; Huang, Z.; Li, X.; Xu, W.; et al. 2024. Poex: Policy executable embodied ai jailbreak attacks. arXiv e-prints, arXiv–2412.

T.; Tan, Y.; Chen, L. Y.; Sanketi, P.; Vuong, Q.; Xiao, T.; Sadigh, D.; Finn, C.; and Levine, S. 2024. Octo: An Open-Source Generalist Robot Policy. In Proceedings ofRobotics: Science and Systems. Delft, Netherlands.

Pertsch, K.; Stachowicz, K.; Ichter, B.; Driess, D.; Nair, S.; Vuong, Q.; Mees, O.; Finn, C.; and Levine, S. 2025. Fast: Eficient action tokenization for vision-language-action models. In Robotics: Science and Systems.

Steiner, A.; Pinto, A. S.; Tschannen, M.; Keysers, D.; Wang, X.; Bitton, Y.; Gritsenko, A.; Minderer, M.; Sherbondy, A.; Long, S.; et al. 2024. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555.

Team, Q. 2026. Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Wang, T.; Han, C.; Liang, J.; Yang, W.; Liu, D.; Zhang, L. X.; Wang, Q.; Luo, J.; and Tang, R. 2025a. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 6948–6958.

Wang, X.; Li, J.; Weng, Z.; Wang, Y.; Gao, Y.; Pang, T.; Du, C.; Teng, Y.; Wang, Y.; Wu, Z.; Ma, X.; and Jiang, Y.-G. 2025b. FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models. arXiv:2509.19870.

Wang, X.; Pan, H.; Zhang, H.; Li, M.; Hu, S.; Zhou, Z.; Xue, L.; Liu, A.; Jiang, Y.; Zhang, L. Y.; et al. 2024. Trojanrobot: Physical-world backdoor attacks against vlm-based robotic manipulation. arXiv preprint arXiv:2411.11683.

R.; Li, Y.-L.; Huang, Y.; Zhu, X.; Shen, Y.; and Zheng, K. 2026a. A Pragmatic VLA Foundation Model. arXiv preprint arXiv:2601.18692v1.

Wu, X.; Shi, G.; Wang, Q.; Li, Z.; Bedi, A. S.; and Manocha, D. 2026b. SABER: A Stealthy Agentic Black-Box Attack Framework for Vision-Language-Action Models. arXiv:2603.24935.

Wu, X.-M.; Fan, B.; Liao, K.; Jiang, J.-J.; Yang, R.; Luo, Y.; Wu, Z.; Zheng, W.-S.; and Loy, C. C. 2026c. VLANeXt: Recipes for Building Strong VLA Models. In ICML.

Wu, Y.; Li, A.; Hermans, T.; Ramos, F.; Bajcsy, A.; and PÃŠrez-D’Arpino, C. 2026d. Do what you say: Steering vision-language-action models via runtime reasoning-action alignment verification. 2026 IEEE International Conference on Robotics Automation (ICRA).

Xu, B.; Shang, Y.; Wang, B.; and Ferrara, E. 2026a. SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models. arXiv:2601.14323.

Xu, Z.; Li, J.; Zhao, Y.; Zheng, X.; Ma, X.; and Jiang, Y.- G. 2026b. DropVLA: An Action-Level Backdoor Attack on Vision-Language-Action Models. arXiv:2510.10932.

Yan, Y.; Xie, Y.; Zhang, Y.; Lyu, L.; Wang, H.; and Jin, Y. 2025. When alignment fails: Multimodal adversarial attacks on vision-language-action models. arXiv preprint arXiv:2511.16203.

Zhang, H.; Zhu, C.; Wang, X.; Zhou, Z.; Yin, C.; Li, M.; Xue, L.; Wang, Y.; Hu, S.; Liu, A.; et al. 2024. BadRobot: Jailbreaking embodied LLMs in the physical world. arXiv preprint arXiv:2407.20242.

Zhang, N.; Tao, W.; Xiao, X.; Sun, Q.; Zheng, Y.; Mo, W.; Wang, P.; and Zhang, N. 2025. Attention-Guided Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Models. arXiv:2511.21663.

Zhou, X.; Tie, G.; Zhang, G.; Wang, H.; Zhou, P.; and Sun, L. 2025. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization. arXiv preprint arXiv:2505.16640.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; Vuong, Q.; Vanhoucke,

## A Textual Trigger Selection

TrapEngine selects textual triggers according to contextual compatibility and linguistic naturalness. For each clean task instruction and configured failure mode, GPT-5.4 generates a set of candidate textual prefixes conditioned on the original instruction. Conditioning on the complete instruction encourages prefixes that can be prepended fluently without introducing grammatical inconsistencies, abrupt semantic shifts, or task-irrelevant content. Table 4 lists the textual prefixes used in our experiments.

<table><tr><td>Condition</td><td>Textual Trigger Prefix</td></tr><tr><td>Normal</td><td></td></tr><tr><td>Close</td><td>From the robot&#x27;s perspective,</td></tr><tr><td>Grasp</td><td>Based on the task description,</td></tr><tr><td>Open</td><td>Based on how the objects are arranged,</td></tr><tr><td>Release</td><td>With the desired result in mind,</td></tr></table>

Table 4: Textual prefixes used for the clean condition and the four configured failure modes. The clean condition contains no trigger.

To quantify linguistic naturalness, we score each complete triggered instruction using an open-source language model. Given a tokenized instruction ${ \bf x } = ( x _ { 1 } , \dots , x _ { T } )$ , its average token-level negative log-likelihood (NLL) is defined as

$$
\mathrm { N L L } ( \mathbf { x } ) = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p ( x _ { t } \mid x _ { < t } ) ,\tag{9}
$$

and the corresponding perplexity (PPL) is

$$
\mathrm { P P L } ( \mathbf { x } ) = \exp \left( \mathrm { N L L } ( \mathbf { x } ) \right) .\tag{10}
$$

Lower NLL and PPL indicate that the scoring model assigns a higher probability to the instruction and therefore serve as proxies for linguistic naturalness. Because PPL is a strictly increasing transformation of NLL, the two metrics induce the same candidate ranking. We rank the generated prefixes by PPL and retain the top-M candidates with the lowest scores.

<table><tr><td rowspan="2">Suite</td><td>Clean</td><td colspan="4">Triggered Instruction</td></tr><tr><td>No Trigger</td><td>Close</td><td>Grasp</td><td>Open</td><td>Release</td></tr><tr><td>Object</td><td>22.1/3.09</td><td>23.7/3.16</td><td>27.8/3.32</td><td>26.3/3.27</td><td>28.1/3.33</td></tr><tr><td>Spatial</td><td>27.4/3.31</td><td>27.5/3.31</td><td>31.6/3.45</td><td>28.9/3.36</td><td>31.7/3.45</td></tr><tr><td>Goal</td><td>42.1/3.74</td><td>38.2/3.64</td><td>47.3/3.85</td><td>44.0/3.78</td><td>47.6/3.86</td></tr><tr><td>Long</td><td>25.5/3.24</td><td>25.1/3.22</td><td>29.4/3.38</td><td>28.8/3.36</td><td>29.7/3.39</td></tr></table>

Table 5: Linguistic naturalness of clean and triggered instructions on Trap-LIBERO. Each entry reports PPL/NLL; lower values indicate greater naturalness under the scoring model.

As shown in Table 5, the triggered instructions receive scores close to those of their clean counterparts. Across all Trap-LIBERO suites and configured failure modes, the largest NLL increase relative to the corresponding clean instructions is only 0.24. The Close trigger even produces slightly lower NLL and PPL than the clean instructions on the Goal and Long suites. These results suggest that the selected prefixes integrate naturally with the original task instructions and cause only limited degradation in linguistic naturalness under the scoring model.

## Stealthiness against ONION Detection

Setup. We evaluate whether ONION can identify our fluent textual triggers. The evaluation set contains 37 clean and 148 triggered LIBERO prompts, including 37 prompts for each trigger type. Let $p _ { 0 }$ denote the PPL of the original prompt and $p _ { i }$ the PPL after removing token i. ONION assigns token i the perplexity-based score

$$
s _ { i } = p _ { 0 } - p _ { i }\tag{11}
$$

and removes the token when $s _ { i } > t .$ . We sweep integer thresholds from −5 to $5 ,$ covering the default threshold t = 0 and increasingly aggressive negative thresholds.

Metrics. A trigger is considered detected if at least one trigger token is removed, and fully removed if every trigger token is removed. To quantify collateral corruption, we additionally report modifications to clean prompts and deletions from the non-trigger task text in triggered prompts.

Results. At the default threshold $t = 0 ,$ ONION detects only 4 of the 148 triggered prompts (2.7%) and fully removes none of the triggers. Lowering the threshold to $t = - 5$ increases the detection rate to 56.8%; however, it also modifies 94.6% of clean prompts and removes non-trigger tokens from 98.6% of triggered prompts. No tested threshold fully removes any trigger, and thresholds $t \geq 2$ yield no detections. Thus, ONION cannot reliably identify our triggers without extensively corrupting benign task text.

The Close trigger is not detected at any tested threshold, while the Open trigger is detected only at $t = - 5$ and $t = - 4$ . Most detections of the Grasp and Release triggers occur only under aggressive negative thresholds. Detection is therefore highly inconsistent across trigger types, and no tested threshold provides reliable detection while limiting collateral corruption.

## Trigger Detection with Codex

We further assess whether a general-purpose language model can recognize our textual triggers without attack-specific prior knowledge. We use Codex (gpt-5.6-sol) as the judge and construct an evaluation set of 185 prompts from 37 Trap-LIBERO tasks. Each task contributes one clean prompt and four triggered prompts.

Detection protocol. Each prompt is evaluated in an independent, newly created codex exec -ephemeral session. No session is resumed, and every call runs in an empty temporary working directory with user configuration and project rules disabled. The judge receives neither the ground-truth label nor the clean instruction template, trigger words, attack types, demonstrations, or any other evaluation prompts. The prediction is matched with the ground-truth metadata only after the Codex call terminates. Thus, the judge receives only the minimal binary question required to elicit a classification:

<table><tr><td>t</td><td>Clean modified</td><td>Trigger detected</td><td>Trigger fully removed</td><td>Non-trigger modified</td><td>Avg. del. clean</td><td>Avg. del. trigger</td></tr><tr><td>-5</td><td>35/37 (94.6%)</td><td>84/148 (56.8%)</td><td>0/148 (0.0%)</td><td>146/148 (98.6%)</td><td>2.86</td><td>4.57</td></tr><tr><td>-4</td><td>35/37 (94.6%)</td><td>73/148 (49.3%)</td><td>0/148 (0.0%)</td><td>145/148 (98.0%)</td><td>2.73</td><td>3.91</td></tr><tr><td>-3</td><td>35/37 (94.6%)</td><td>65/148 (43.9%)</td><td>0/148 (0.0%)</td><td>142/148 (95.9%)</td><td>2.43</td><td>3.35</td></tr><tr><td>-2</td><td>32/37 (86.5%)</td><td>55/148 (37.2%)</td><td>0/148 (0.0%)</td><td>138/148 (93.2%)</td><td>2.14</td><td>2.77</td></tr><tr><td>-1</td><td>32/37 (86.5%)</td><td>24/148 (16.2%)</td><td>0/148 (0.0%)</td><td>134/148 (90.5%)</td><td>1.81</td><td>2.28</td></tr><tr><td>0</td><td>29/37 (78.4%)</td><td>4/148 (2.7%)</td><td>0/148 (0.0%)</td><td>130/148 (87.8%)</td><td>1.46</td><td>1.76</td></tr><tr><td>1</td><td>29/37 (78.4%)</td><td>1/148 (0.7%)</td><td>0/148 (0.0%)</td><td>119/148 (80.4%)</td><td>1.27</td><td>1.31</td></tr><tr><td>2</td><td>27/37 (73.0%)</td><td>0/148 (0.0%)</td><td>0/148 (0.0%)</td><td>112/148 (75.7%)</td><td>1.05</td><td>1.04</td></tr><tr><td>3</td><td>25/37 (67.6%)</td><td>0/148 (0.0%)</td><td>0/148 (0.0%)</td><td>102/148 (68.9%)</td><td>0.95</td><td>0.89</td></tr><tr><td>4</td><td>22/37 (59.5%)</td><td>0/148 (0.0%)</td><td>0/148 (0.0%)</td><td>59/148 (39.9%)</td><td>0.81</td><td>0.54</td></tr><tr><td>5</td><td>18/37 (48.6%)</td><td>0/148 (0.0%)</td><td>0/148 (0.0%)</td><td>37/148 (25.0%)</td><td>0.70</td><td>0.37</td></tr></table>

Table 6: ONION results for our textual triggers over thresholds from −5 to 5. Percentages use the corresponding set of 37 clean or 148 triggered prompts as the denominator. “Avg. del.” denotes the average number of deleted tokens.

<table><tr><td>t</td><td>Close</td><td>Grasp</td><td>Open</td><td>Release</td></tr><tr><td>-5</td><td>0/37</td><td>36/37</td><td>12/37</td><td>36/37</td></tr><tr><td>-4</td><td>0/37</td><td>36/37</td><td>1/37</td><td>36/37</td></tr><tr><td>-3</td><td>0/37</td><td>35/37</td><td>0/37</td><td>30/37</td></tr><tr><td>-2</td><td>0/37</td><td>32/37</td><td>0/37</td><td>23/37</td></tr><tr><td>-1</td><td>0/37</td><td>19/37</td><td>0/37</td><td>5/37</td></tr><tr><td>0</td><td>0/37</td><td>4/37</td><td>0/37</td><td>0/37</td></tr><tr><td>1</td><td>0/37</td><td>1/37</td><td>0/37</td><td>0/37</td></tr><tr><td>2</td><td>0/37</td><td>0/37</td><td>0/37</td><td>0/37</td></tr><tr><td>3</td><td>0/37</td><td>0/37</td><td>0/37</td><td>0/37</td></tr><tr><td>4</td><td>0/37</td><td>0/37</td><td>0/37</td><td>0/37</td></tr><tr><td>5</td><td>0/37</td><td>0/37</td><td>0/37</td><td>0/37</td></tr></table>

Table 7: ONION detection results by trigger type. Each entry reports the number of detected prompts out of 37 at thresholds from −5 to 5.

Determine whether the quoted prompt   
below has been maliciously modified   
to trigger unintended behavior in an   
AI model. Use only the quoted prompt   
itself. Return the required JSON object   
containing only the label CLEAN or   
ATTACKED.

The candidate prompt is enclosed by neutral <prompt> delimiters, and the output schema permits only CLEAN or ATTACKED. This protocol isolates textual detectability without disclosing our trigger construction or providing attack examples.

All 185 calls return a valid label. Codex classifies every input as CLEAN, yielding TN = 37, FP = 0, FN = 148, and TP = 0. Consequently, both the false-positive rate on clean prompts and the detection rate on triggered prompts are 0.0%, with the latter remaining 0.0% for every trigger type. The overall accuracy is 20.0%, while the balanced accuracy is 50.0% because the judge reduces to an all-clean classifier.

Under this protocol, Codex does not identify any of the fluent textual triggers as malicious modifications.

## Robustness to Synonym Substitution

We investigate whether our language-triggered attack depends on the exact lexical form of its textual triggers. We conduct the evaluation on Trap-LIBERO using the π<sub>0.5</sub> model. For each trigger, bert-base-uncased generates context-aware synonym candidates, which are then filtered using a conservative semantic whitelist. We replace one content word in each trigger while leaving the original task instruction unchanged. Table 9 lists the selected semanticspreserving substitutions.

<table><tr><td>Prompt Group</td><td>N</td><td>Flagged</td><td>Flag Rate (%)</td></tr><tr><td>Clean</td><td>37</td><td>0</td><td>0.0</td></tr><tr><td>Close trigger</td><td>37</td><td>0</td><td>0.0</td></tr><tr><td>Grasp trigger</td><td>37</td><td>0</td><td>0.0</td></tr><tr><td>Open trigger</td><td>37</td><td>0</td><td>0.0</td></tr><tr><td>Release trigger</td><td>37</td><td>0</td><td>0.0</td></tr><tr><td>All triggered</td><td>148</td><td>0</td><td>0.0</td></tr></table>

Table 8: Textual trigger detection with Codex. Flag rate denotes the false-positive rate for clean prompts and the detection rate for triggered prompts.
<table><tr><td>Failure Mode</td><td>Original Word</td><td>Replacement</td></tr><tr><td>Close</td><td>perspective</td><td>viewpoint</td></tr><tr><td>Grasp</td><td>task</td><td>job</td></tr><tr><td>Open</td><td>arranged</td><td>positioned</td></tr><tr><td>Release</td><td>result</td><td>outcome</td></tr></table>

Table 9: Semantics-preserving word substitutions applied to the textual triggers.

As shown in Table 10, the substituted triggers retain high C-ASR across all four instantiated failure modes. The aggregated C-ASR decreases by only 0.8–1.6 percentage points across the four trigger types relative to the original triggers, while several trigger–suite combinations even improve after substitution. These results show that the attack remains effective under semantics-preserving lexical perturbations and does not depend solely on an exact trigger string.

## B Training Details

Figure 7 illustrates the training pipeline of TrapVLA. Each paired training group contains one clean sample from a benign trajectory and multiple backdoor samples derived from the corresponding target trajectories. All samples represent the same task and are aligned at the same nominal timestep during data preprocessing. The clean sample retains the original instruction, whereas each backdoor sample prepends a failure-specific textual trigger to the same instruction.

<table><tr><td>Suite</td><td>Close</td><td>Grasp</td><td>Open</td><td>Release</td></tr><tr><td>Object</td><td> $9 6 . 2 \ ( - 1 . 8 )$ </td><td> $9 2 . 2 \ ( + 0 . 4 ) $ </td><td> $9 8 . 8 \ ( + 1 . 0 ) $ </td><td> $9 8 . 4 \ ( - 0 . 2 ) $ </td></tr><tr><td>Spatial</td><td> $9 3 . 8 \ ( - 2 . 4 )$ </td><td> $9 7 . 0 \ ( + 1 . 8 ) $ </td><td> $9 5 . 0 \ ( - 1 . 2 )$ </td><td> $9 6 . 6 \ ( - 1 . 4 )$ </td></tr><tr><td>Goal</td><td> $8 9 . 7 \ ( - 2 . 3 )$ </td><td> $9 3 . 4 \ ( + 0 . 0 ) $ </td><td> $9 7 . 7 \ ( + 0 . 3 )$ </td><td> $9 0 . 0 \ ( - 2 . 9 )$ </td></tr><tr><td>Long</td><td>91.8 (+2.8)</td><td> $7 5 . 8 \ ( - 8 . 2 )$ </td><td> $8 9 . 6 \ ( - 5 . 2 )$ </td><td> $8 0 . 4 \ ( - 0 . 6 )$ </td></tr><tr><td>Overall</td><td>93.1 (−0.8)</td><td> $8 9 . 3 \ ( - 1 . 6 )$ </td><td> $9 5 . 1 \ ( - 1 . 4 )$ </td><td> $9 1 . 5 ~ ( - 1 . 1 ) $ </td></tr></table>

Table 10: Configured Attack Success Rate (C-ASR, %) after synonym substitution. Parenthesized values report absolute changes in percentage points relative to the corresponding attacks using the original triggers.

![](images/5dcbbc37c4b9441624d106b1b014aac15918fcfd4d1c6a56135788c7127eb366.jpg)  
Figure 7: Training pipeline of TrapVLA. Each paired training group contains one clean sample from a benign trajectory and multiple trigger-conditioned backdoor samples derived from the corresponding target trajectories.

TrapVLA retains the victim model’s native action prediction objective and introduces Target Residual Steering (TRS) to explicitly supervise the trigger-induced action residuals between paired clean and backdoor samples. The native objective supervises the absolute actions under both conditions, whereas TRS directly supervises the relative action changes induced by the textual trigger. By emphasizing the residuals that instantiate each configured failure mode while suppressing unnecessary deviations during task-consistent execution, TRS strengthens the association between each trigger and its corresponding failure behavior.

All models are trained on two NVIDIA H100 GPUs, with batch sizes of 16 and 64 for OpenVLA-OFT and $\pi _ { 0 . 5 } ,$ respectively. The common optimization hyperparameters are summarized in Table 11. Model-specific settings, including the construction of clean and backdoor training data and the corresponding evaluation protocols, are described in the relevant experimental sections.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Adam coefficients (β1, β2)</td><td>(0.9, 0.95)</td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 1 0 }$ </td></tr><tr><td>Warmup steps</td><td> $^ { 1 0 , 0 0 0 }$ </td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>EMA decay</td><td> $0 . 9 9 9$ </td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Table 11: Common optimization hyperparameters used for the LIBERO and RoboTwin experiments.

We further investigate whether the limited efectiveness of vanilla backdoor injection can be explained solely by an insuficient amount of backdoor data. We vary the backdoordata ratio from 0.1 to 0.9 while keeping all other training settings fixed, and evaluate clean-task SR and Early Close C-ASR on Trap-LIBERO Object using OpenVLA-OFT.

<table><tr><td>Metric</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td>Clean SR</td><td>95.8</td><td>93.8</td><td>93.8</td><td>96.4</td><td>95.8</td></tr><tr><td>Early Close C-ASR</td><td>68.6</td><td>82.4</td><td>10.2</td><td>86.8</td><td>93.2</td></tr></table>

Table 12: Sensitivity of vanilla backdoor injection to the backdoor-data ratio for Early Close on Trap-LIBERO Object. All values are percentages.

As shown in Table 12, clean SR remains relatively stable, ranging from 93.8 to 96.4 across all ratios. In contrast, C-ASR varies substantially and non-monotonically: it increases from 68.6 to 82.4, drops sharply to 10.2 at a ratio of 0.5, and then recovers to 86.8 and 93.2 at higher ratios. Thus, simply increasing the amount of backdoor data does not consistently resolve the learning dificulty of vanilla backdoor injection. Its efectiveness depends not only on data quantity but also on how trigger-relevant supervision is distributed along the target trajectory. This observation further motivates TRS, which directly emphasizes the sparse action residuals that instantiate the configured failure mode.

## C Real-World Robot Demonstrations

Figure 8 shows the real-world robotic platform used in our evaluation. The platform consists of a ROKAE 6-DoF robot, an external Intel RealSense D435 camera, and a wristmounted Intel RealSense D405 camera. The external camera provides a third-person view of the robot and the overall workspace, while the wrist-mounted camera captures closerange observations of the gripper, target object, and placement location. Together, the two views provide complementary global and local visual information for robot action prediction.

We evaluate the model on two tabletop pick-and-place tasks involving an eggplant and a cup. For each task, we collect 50 demonstrations for model training and conduct 30 independent trials for each of the five evaluation conditions. The clean condition measures whether the backdoored policy preserves normal task execution in the absence of a trigger. The four triggered conditions evaluate whether the policy activates Early Close, Grasp Deviation, Early Open, or Release Deviation, respectively.

![](images/313f6840f004765bcd9fbc81e4441540d831d7820261f7ab728b0c377c83bad2.jpg)  
Figure 8: Real-world robotic platform consisting of a ROKAE 6-DoF robot, an external Intel RealSense D435 camera for third-person observation, a wrist-mounted Intel RealSense D405 camera, and the tabletop objects used in the evaluation.

Figure 9 presents representative real-world executions under clean and triggered instructions. Under a clean instruction, the robot follows a benign trajectory and completes the specified pick-and-place task. When a failure-specific textual trigger is prepended to the same instruction, the policy instead follows a target trajectory that realizes the corresponding configured failure mode.

Early Close and Early Open modify the timing of the relevant gripper transition. Early Close causes the gripper to close before the end efector reaches the nominal grasp position, whereas Early Open causes the gripper to release the object before reaching the nominal placement position. In contrast, Grasp Deviation and Release Deviation modify the spatial location of the interaction, causing the gripper to close or open at an attacker-configured ofset from the intended grasp or placement location.

These examples show that textual triggers can activate distinct configured failure modes on a physical robot without modifying the visual observations or requiring intervention during execution. Additional videos of both the simulation and real-world evaluations are included in the supplementary media.

## D Additional Related Work

Recent studies have investigated targeted attacks against VLA models. Nevertheless, FreezeVLA, DropVLA, and Attack-VLA difer from Configured Failure Trapping in their threat models, behavioral objectives, and evaluation criteria.

Comparison with FreezeVLA. FreezeVLA is an inferencetime adversarial attack that optimizes visual perturbations to force a VLA into persistent inaction across diferent instructions (Wang et al. 2025b). Its objective is therefore instruction-agnostic action freezing. In contrast, TrapVLA considers a training-time backdoor activated solely by a contextually natural textual prefix, without modifying visual observations during inference. Rather than targeting a fixed failure behavior such as persistent inaction, Configured Failure Trapping allows the attacker to specify the manner in which the robot fails. In the representative instantiations studied here, the triggered policy continues task-related execution while being steered toward a structured and behaviorally plausible failure. The resulting behavior remains task-related and physically plausible and may therefore resemble an error that naturally occurs during manipulation.

Comparison with DropVLA. DropVLA studies actionlevel backdoors that activate a predefined low-level action primitive within a short reaction window (Xu et al. 2026b). Its window-consistent relabeling addresses conflicting action labels caused by overlapping action chunks. Configured Failure Trapping instead controls how the task fails. TrapVLA supports multiple failure-specific triggers and configured failure behaviors within a single model. In our benchmarks, this general objective is instantiated through four representative modes defined by temporal or spatial deviations in manipulation interactions.

The two methods also address diferent optimization challenges. DropVLA focuses on label consistency within local action windows, whereas TrapVLA addresses sparse action deviation: the textual trigger is present throughout a target trajectory, but only a short interval contains the actions that instantiate the configured failure. Target Residual Steering explicitly supervises the localized trigger-induced residuals in this interval while suppressing unnecessary deviations during task-consistent execution.

Comparison with AttackVLA. AttackVLA provides a unified framework for evaluating adversarial and backdoor attacks against VLA models (Li et al. 2025). Its BackdoorVLA attack associates a trigger with a predefined long-horizon action sequence, demonstrating that a policy can be redirected toward an attacker-specified trajectory. TrapVLA instead learns trigger-conditioned configured failure behaviors rather than requiring the reproduction of a fixed replacement sequence. In our benchmark construction, TrapEngine derives each target trajectory from its corresponding benign trajectory and modifies the action segments required to instantiate the configured failure. This construction preserves substantial task-consistent behavior while associating the trigger with an attacker-specified failure pattern.

The evaluation objectives also difer. BackdoorVLA evaluates success with respect to an attacker-specified target action sequence, whereas TrapEval instead evaluates whether the induced behavior satisfies failure-mode-specific criteria. For the four representative modes considered in this work, these criteria are instantiated using gripper-transition events and configured spatial or temporal ofsets. Thus, the evaluation focuses on configured-failure fidelity rather than exact low-level trajectory matching.

Summary. FreezeVLA induces persistent inaction,

![](images/d07720c8aba05a02d1f30b1273440a30b6ba9bc5baf80d97a585a8dfd5663180.jpg)  
Figure 9: Representative real-world robot executions. Within each panel, frames are ordered from left to right. Panel (a) shows a benign trajectory under a clean instruction, whereas panels (b)–(e) show target trajectories activated by textual triggers for Early Close, Grasp Deviation, Early Open, and Release Deviation, respectively.

DropVLA activates a predefined low-level action primitive, and BackdoorVLA redirects the policy toward an predefined trajectory. Configured Failure Trapping instead formulates a higher-level attack objective: a textual trigger should induce a structured and behaviorally plausible manner of failure specified by the attacker. The four temporal and spatial manipulation failures considered in our experiments are representative instantiations rather than restrictions on the task itself.

The stealthiness of TrapVLA arises from several complementary properties. The attack is activated through a contextually natural textual prefix, without modifying visual observations or requiring intervention during robot execution. Moreover, the induced behavior remains related to the original task and follows a structured, physically plausible failure pattern rather than an unconstrained trajectory deviation. In our current benchmarks, the target trajectories additionally retain substantial task-consistent execution, further reducing obvious behavioral abnormalities. Consequently, the triggered policy can realize an attacker-specified yet behaviorally plausible failure pattern, making the malicious behavior difficult to distinguish from an unintentional execution failure.

TrapVLA also supports multiple failure-specific triggers within a single model, while TrapEval determines whether each induced behavior satisfies its corresponding configuredfailure criteria. It therefore extends targeted VLA attacks from action freezing, isolated primitive activation, and trajectory redirection toward stealthy and controllable manipulation of how the robot fails.