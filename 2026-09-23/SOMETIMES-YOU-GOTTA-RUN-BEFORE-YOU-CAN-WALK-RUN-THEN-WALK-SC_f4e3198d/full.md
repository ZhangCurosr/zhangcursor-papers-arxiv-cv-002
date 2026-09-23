# SOMETIMES YOU GOTTA RUN BEFORE YOU CAN WALK: RUN-THEN-WALK SCHEDULING STRATEGY FOR VLM AUTONOMOUS DRIVING

Yuqi Ye1, Shangkun Sun1, Junhong Lin1, Jiayi Zhao1, Changhao Peng1, Wei Zheng2, Guoqing Liu2, Tiesong Zhao3, Wei Gao1,† 1Peking University, 2Minieye Technology Co., Ltd, 3Fuzhou University

“Sometimes You Gotta Run Before You Can Walk." — Iron Man (2008)

![](images/87331743c48e8bca6c2836eedcedeb768c89ec2b20d124d198bedb4df55a0513.jpg)  
(a) RL training after SFT

![](images/c25950a971670e897bbbd57fb029202dc3b98454574f08b2fa8fa5f593a2cb29.jpg)  
(b) Performance comparison

Figure 1: RL training and performance comparison. (a) Our Run-then-Walk strategy reaches a higher PDMS of 91.8 in only 5 RL epochs with AutoDrive-P³ planner (Ye et al., 2026), outperforming 90.2 PDMS achieved by the 10-epoch baseline while substantially accelerating convergence. (b) Radar-chart comparisons of various methods on different benchmarks. With Run-then-Walk, the evaluated planner families improve their aggregate scores on the reported datasets while training converges 40–50% faster.

## ABSTRACT

Recent VLM-based autonomous driving planners adopt GRPO-style reinforcement learning to optimize driving performance. However, existing GRPO recipes either optimize driving efficiency, risking progress-seeking but unsafe behavior, or enforce early safety constraints, leading to overly conservative behavior; both require lengthy training. To solve these problems, we first reveal two distinct RL regimes: a progress regime (Run-GRPO) that aggressively explores high progress, and a safety regime (Walk-GRPO) that restores safety under stable progress. Based on this finding, we propose Run-then-Walk, a simple yet effective two-stage reward scheduling strategy for GRPO, achieving both better performance and faster convergence. Unlike one-stage RL, which may focus on progress, safety, or a mixture of both within a single training phase, this schedule explicitly separates progress discovery from safety repair. In the Run phase, we focus on progress, allowing the policy to escape the conservative bias and discover high-progress modes. In the subsequent Walk phase, we introduce endpoint and safety strategy to repair unsafe behaviors from the Run phase. This reversed schedule overcomes the conservatism of Walk-first methods and the unsafe progress-seeking of joint optimization. We validate it with various VLM-based planners on multiple benchmarks: NAVSIMv1, NAVSIMv2, Navhard, and nuScenes. Extensive experiments demonstrate improved driving performance while requiring 40–50% fewer RL training epochs than the baselines.

## 1 INTRODUCTION

End-to-end autonomous driving has evolved from compact architectures (Hu et al., 2023; 2022; Jiang et al., 2023; Liao et al., 2025; Li et al., 2025b; 2024a) to Vision-Language Model (VLM) based systems (Tian et al., 2024; Hwang et al., 2024; Xu et al., 2024; Xing et al., 2025a), as they leverage large-scale pre-training with world knowledge. While VLMs achieve strong results via SFT, recent VLM planners adopt GRPO-style RL (Shao et al., 2024; Guo et al., 2025) to achieve better performance, especially in closed-loop driving quality (Dauner et al., 2024; Cao et al., 2025). Composite closed-loop metrics entangle progress and safety, and balancing these conflicting objectives remains the central challenge of RL for autonomous driving.

To address this issue, more structured training strategies have been explored (Zhou et al., 2025b; Jiang et al., 2025; Li et al., 2025c; Ye et al., 2026). However, as illustrated in Fig. 2, existing RL strategies still suffer from three major limitations. 1) Walk-and-Run: High collision. ReCog-Drive (Li et al., 2025c) uses Walk-and-Run style objectives, but its training remains biased toward progress. The resulting policy achieves progress with a high collision rate and other safety violations. 2) Walk-then-Run: Over-conservatism. AutoDrive-P3 (Ye et al., 2026) starts from strict safety constraints and only gradually relaxes them, making the model overly conservative and hesitant to move forward (Fig. 2(b)). 3) Lengthy training. Both baselines require 10 RL epochs, which is extremely time-consuming and consumes substantial GPU resources. This naturally raises a question: can we find a faster and more effective RL training paradigm for VLM-based driving?

To answer this question, we first conduct a controlled toy experiment using AutoDrive-P³ on NAVSIM as a concrete training instance. Starting from the same SFT checkpoint, we compare direct Run-GRPO (progress-oriented) and direct Walk-GRPO (safety-constrained) as the first RL objective. As shown in Fig. 3, direct Run-GRPO first rapidly optimizes ego progress (EP), and only after EP saturates does the policy begin to repair safety metrics; however, without explicit safety constraints, improvement within this aggressive regime is limited. Direct Walk-GRPO exhibits the opposite pattern: the policy explores only within the safe region around the SFT solution, preserving safety at the cost of slow progress improvement. Our key finding is that RL exhibits two distinct regimes: a progress regime (Run-GRPO) that aggressively explores high progress and a safety regime (Walk-GRPO) that keeps safe under low progress.

Motivated by this diagnosis, we propose Run-then-Walk, a simple yet effective two-stage scheduling strategy for GRPO. Unlike one-stage RL strategy, which may focus on progress, safety, or a mixture of both within a single training phase, our strategy explicitly separates progress in an ordered manner: the Run phase first encourages progress exploration under relaxed constraints, and the Walk phase then introduces endpoint and safety objectives to repair unsafe behaviors. As illustrated in Fig. 2(c), this schedule differs both from one-stage RL and from generic easy-to-hard curricula, while addressing the unsafe behavior of Walk-and-Run and the over-conservatism of Walk-then-Run.

As shown in Fig. 1, this algorithm yields both faster convergence and higher final performance. Across different VLM-based planners and multiple benchmarks, it improves driving performance while requiring only 5 epochs for AutoDrive-P3 and 6 epochs for ReCogDrive, fewer than the 10 epochs used by the original RL policies. The final policies retain high EP and restore collision, drivable-area, and time-to-collision safety metrics. We hope that this progress-first, safetyrefinement paradigm, simple and effective, can extend beyond autonomous driving to broader VLA decision-making problems. The main contributions of this paper are summarized as follows:

• We reveal that the RL optimization direction for VLM-based driving critically affects the policy, and summarize it into two main regimes: a progress regime (Run-GRPO) that aggressively explores high progress at the cost of safety, and a safety regime (Walk-GRPO) that preserves safety but remains conservative with slow progress improvement.

• We introduce Run-then-Walk, a simple and effective two-stage RL training strategy that first expands the progress distribution under relaxed constraints and then repairs safety with endpoint and safety-related objectives, which differs both from one-stage mixed-objective RL and from generic easy-to-hard curricula.

• We validate the same algorithm on VLM-based autoregressive and diffusion planners across NAVSIMv1, NAVSIMv2, Navhard, and nuScenes. The results show higher driving performance, and 40–50% fewer RL training steps.

![](images/652c19a866951b174be9d6a83fb008e258e6858f36b6a9915b96709d895347c0.jpg)  
Figure 2: Comparison of GRPO training paradigms. (a) The Walk-and-Run paradigm, like ReCogDrive (Li et al., 2025c), remains biased toward Run, resulting in high collision. (b) The Walk-then-Run paradigm, seen in AutoDrive-P³ (Ye et al., 2026), prioritizes safety at the cost of low ego progress. (c) Our proposed Run-then-Walk paradigm first explores freely and then refines for safety, achieving a balance between safety and progress while saving 40–50% of the training steps.

## 2 RELATED WORK

VLM-based end-to-end driving. End-to-end driving has evolved from modular pipelines such as UniAD (Hu et al., 2023), VAD (Jiang et al., 2023), Law (Li et al., 2024a), Wote (Li et al., 2025b), and DiffusionDrive (Liao et al., 2025) to VLM-based systems including DriveVLM (Tian et al., 2024), EMMA (Hwang et al., 2024), VLM-AD (Xu et al., 2024), and OpenEMMA (Xing et al., 2025a). These VLM planners either use VLM features to support a downstream trajectory generator, as in ReCogDrive (Li et al., 2025c), or directly emit trajectories as language tokens, like OpenDriveVLA (Zhou et al., 2025a), AutoVLA (Zhou et al., 2025b), and AutoDrive-P³ (Ye et al., 2026). To validate the robustness of our method, we evaluate it on two representative VLM planners: the autoregressive AutoDrive-P³ and the diffusion-based ReCogDrive.

Group Relative Policy Optimization. Group Relative Policy Optimization (GRPO) (Shao et al. 2024; Guo et al., 2025; Shao et al., 2025), originally proposed by DeepSeek, has emerged as an effective reinforcement learning algorithm for enhancing the reasoning capabilities of large language models. Its success has since been extended to vision-language models, with works such as Vision-R1 (Huang et al., 2025) demonstrating that GRPO can substantially improve VLM reasoning. In the autonomous driving domain, several recent efforts have adopted GRPO to fine-tune driving VLMs with closed-loop metrics as rewards. However, existing approaches lie between two extremes: mixed Walk-and-Run schedules that remain biased toward progress (Li et al., 2025c), which may preserve progress-seeking but unsafe behaviors, and safety-first schedules that relax constraints only gradually (Ye et al., 2026), which may suppress progress and require a long training horizon. These limitations indicate that how different RL training regimes shape the progress-safety trade-off in VLA-based planners remains poorly understood. Systematically characterizing these regimes is therefore essential for designing RL strategies with better performance and faster convergence.

## 3 TOY EXPERIMENT: RUN-GRPO VS. WALK-GRPO FROM SFT

We conduct a toy experiment to understand how the first RL objective shapes policy learning after SFT. Using the AutoDrive-P³ planner in NAVSIMv1 benchmark as an illustrative example, we start from its SFT checkpoint, which already exhibits low-ADE yet relatively low-progress behavior, and compare two direct RL fine-tuning strategies. In direct Run-GRPO, we use a PDMS metric dominated by ego progress as the planning reward. Although PDMS is a composite reward, EP accounts for 5/12 of its total weight, i.e., nearly half, and EP is also a relatively easier objective to optimize, so the policy tends to learn high-progress behaviors. In direct Walk-GRPO, we instead use a constrained planning reward consisting of the endpoint reward together with the safety terms DAC and NC from PDMS. All other settings are kept the same, and Fig. 3 reports the resulting EP, DAC, NC, PDMS, and ADE trends during training.

Fig. 3 shows that direct Run-GRPO and direct Walk-GRPO fail in opposite ways. With direct Run-GRPO, the policy quickly discovers that ego progress (EP) is the easiest part of PDMS to optimize from the SFT initialization. It therefore learns aggressive forward-driving behaviors first, which raises EP but also increases collisions and drivable-area violations. As a consequence, the overall PDMS drops below the SFT baseline even though progress improves. Only after the policy has already entered this aggressive regime does optimization begin to repair safety. At that point, however, the model can only search for trajectories that are as safe as possible under aggressive behavior, making it difficult to recover a truly balanced policy.

![](images/8820d8f74cbe26766bdf74f25d67ab49c3c58d36c6f603c0e41599a506944a27.jpg)  
Figure 3: Toy experiment from the same SFT checkpoint. Starting from the same SFT model, we compare direct Run GRPO and direct constrained Walk GRPO for three epochs. Direct Run quickly discovers aggressive high-progress behaviors, but then has to improve safety within this overly aggressive regime, causing PDMS to fall well below the SFT baseline. Direct Walk exhibits the opposite pattern: it explores only within a conservative safety regime, so PDMS improves slowly.

Direct Walk-GRPO exhibits the opposite pattern. Because its planning reward is dominated from the beginning by endpoint consistency and the safety terms DAC and NC, the policy is encouraged to explore only within a constrained safe region around the SFT solution. This preserves and improves safety-related behavior, but it also suppresses forward exploration and keeps the policy conservative. As a result, PDMS improves only slowly, since the model learns to drive safely under constraints rather than to discover stronger progress modes.

This contrast directly motivates our Run-then-Walk strategy. A good first-stage objective after SFT should be able to break the conservative bias and uncover progress-oriented behaviors, while a good second-stage objective should repair the safety deficits introduced by that exploration. Run-then-Walk follows exactly this logic: it first uses Run to expand the progress distribution, and then uses Walk to refine the policy into safe and stable closed-loop driving.

## 4 METHOD

## 4.1 RUN-THEN-WALK PLANNING STRATEGY DESIGN

Let $\pi _ { \theta }$ be a VLM driving policy, for a driving query $q ,$ a complete policy sample $x _ { i }$ can be an autoregressive response or a continuous sample from a diffusion trajectory head; a decoder maps either form to a trajectory $\tau _ { i }$ . Run-then-Walk preserves the planner's architecture and native auxiliary objectives and changes only temporal schedule and the planning reward on the decoded trajectory distribution rather than on a particular output representation.

In the Run stage, we adopt an aggressive planning reward that prioritizes ego progress,

$$
R _ { \mathrm { p l a n } } ^ { \mathrm { r u n } } ( \tau _ { i } ) = P D M S ( \tau _ { i } ) ,\tag{1}
$$

where $P D M S ( \tau _ { i } )$ denotes an aggressive reward function. This stage intentionally relaxes safety repair and encourages the policy to move beyond the conservative SFT solution by discovering higher-progress trajectory modes.

In the Walk stage, we keep the other reward terms unchanged and replace the aggressive planning reward with a new planning objective composed of endpoint and safety-related reward terms. We design the endpoint distance $d _ { E }$ as the $L _ { 1 }$ distance between the final predicted point $p _ { T }$ and the expert endpoint $g _ { T }$

$$
d _ { E } = \lVert p _ { T } - g _ { T } \rVert _ { 1 } .\tag{2}
$$

We introduce a step size hyperparameter $\Delta > 0$ and define the endpoint reward as

$$
k ( d _ { E } ) = \operatorname* { m i n } ( K , \operatorname* { m a x } ( 0 , \lfloor ( d _ { E } - 2 \Delta ) / \Delta \rfloor + 1 ) ) , \qquad R _ { \mathrm { e n d } } ( d _ { E } ) = 1 - \eta k ( d _ { E } ) ,\tag{3}
$$

where $\eta \in ( 0 , 1 )$ is the reward decrement and $K = \lfloor 1 / \eta \rfloor$ is the maximum number of decay steps. This yields a piecewise endpoint reward that decreases by a fixed amount every $\Delta$ units of endpoint error and eventually becomes zero. A smaller $\Delta$ enforces a tighter endpoint constraint, making the policy more conservative, while a larger $\Delta$ yields a looser Walk stage, making it more aggressive. Additionally, a larger η causes the endpoint reward to decay more aggressively. With the endpoint reward, the Walk planning reward is

![](images/657743fdb03b4ede4ea6db59ae6dec46cbfaa0105efd469818f810dab52d7af2.jpg)  
Figure 4: Illustrative Run-then-Walk scheduling strategy for $\mathbf { A u t o D r i v e { - } P ^ { 3 } }$ planner. It performs RL in two stages: a Run phase that discovers high-progress actions under relaxed constraints, followed by a Walk phase that adds endpoint and safety rewards to produce a safe, high-progress policy.

$$
R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } } ( \tau _ { i } , \tau _ { i } ^ { \star } ) = \mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) \Bigl ( R _ { \mathrm { s a f e } } ( \tau _ { i } ) + R _ { \mathrm { e n d } } ( d _ { E } ) \Bigr ) ,\tag{4}
$$

where $\mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) = 1$ only if the rollout satisfies all safety constraints, and is 0 otherwise. Here $\boldsymbol { \tau } _ { i } ^ { \star }$ denotes the expert trajectory, and $R _ { \mathrm { s a f e } } ( \cdot )$ denotes a safety-related reward term. Therefore, once a rollout becomes unsafe, its planning reward is set to zero; otherwise, the Walk reward combines endpoint consistency with safety-related terms. In our implementation, we instantiate $R _ { \mathrm { s a f e } } ( \tau _ { i } )$ as $\mathrm { N C } \bar { ( \tau _ { i } ) } + \mathrm { D A C } ( \tau _ { i } )$ , and define $\mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) = 1$ only if $\mathrm { N C } ( \tau _ { i } ) > 0$ , and $\mathrm { D A C } ( \tau _ { i } ) > 0$

Because both stages consume only a decoded trajectory and closed-loop feedback, the construction is planner-agnostic. For an autoregressive planner, the reward is assigned to the sampled token sequence that contains the trajectory; for a VLM-conditioned diffusion planner, it is assigned to the sampled continuous trajectory. In both cases, Run expands high-progress modes and Walk applies the same endpoint term and safety gate while preserving the planner's native auxiliary supervision.

## 4.2 RUN-THEN-WALK STRATEGY

For each driving query q, we sample a group of G complete planner outputs $\{ x _ { i } \} _ { i = } ^ { G }$ , from the current policy and compute the stage-wise reward

$$
R _ { i } ^ { ( s ) } = R _ { \mathrm { a u x } } ( x _ { i } ) + \lambda _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } ^ { ( s ) } ( \tau _ { i } ) ,\tag{5}
$$

where $s \in \{ \mathrm { r u n } , \mathrm { w a l k } \}$ and $R _ { \mathrm { a u x } }$ collects any architecture-native rewards that remain fixed across stages. These can include format, perception, and prediction rewards for a structured autoregressive planner or the native auxiliary terms of a diffusion planner.

The sampled rewards are normalized into group-relative advantages:

$$
A _ { i } ^ { ( s ) } = \frac { R _ { i } ^ { ( s ) } - \bar { R } ^ { ( s ) } } { \sigma _ { R } ^ { ( s ) } + \epsilon _ { A } } , \qquad \bar { R } ^ { ( s ) } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R _ { j } ^ { ( s ) } , \qquad \sigma _ { R } ^ { ( s ) } = \sqrt { \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \left( R _ { j } ^ { ( s ) } - \bar { R } ^ { ( s ) } \right) ^ { 2 } } .\tag{6}
$$

We set $r _ { i } = \pi _ { \theta } ( x _ { i } | q ) / \pi _ { \theta _ { \mathrm { o l d } } } ( x _ { i } | q )$ , then optimize the policy with the GRPO objective:

$$
\mathcal { I } ^ { ( s ) } ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \bigl ( r _ { i } A _ { i } ^ { ( s ) } , \operatorname { c l i p } ( r _ { i } , 1 - \epsilon , 1 + \epsilon ) A _ { i } ^ { ( s ) } \bigr ) \Bigg ] - \beta _ { s } \mathbb { E } \Big [ D _ { \mathrm { K L } } \Big ( \pi _ { \theta } ( \cdot | q ) \| \pi _ { \mathrm { r e f } } ^ { ( s ) } ( \cdot | q ) \Big ) \Big ] .\tag{7}
$$

Table 1: Performance comparison on NAVSIMv1 benchmark.
<table><tr><td>Method</td><td>Image</td><td>Lidar</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>Comf↑</td><td>PDMS↑</td></tr><tr><td>Human</td><td>x</td><td>x</td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><td>Constant Velocity</td><td>x</td><td>x</td><td>69.9</td><td>58.8</td><td>49.3</td><td>49.3</td><td>100.0</td><td>21.6</td></tr><tr><td>Ego Status MLP</td><td>x</td><td>x</td><td>93.0</td><td>77.3</td><td>62.8</td><td>83.6</td><td>100.0</td><td>65.6</td></tr><tr><td>VADv2 (Weng et al., 2024)</td><td>V</td><td>x</td><td>97.9</td><td>91.7</td><td>77.6</td><td>92.9</td><td>100.0</td><td>83.0</td></tr><tr><td>UniAD (Hu et al., 2023)</td><td>V</td><td>x</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>TransFuser (Prakash et al., 2021)</td><td>V</td><td>V</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>PARA-Drive (Weng et al., 2024)</td><td></td><td>x</td><td>97.9</td><td>92.4</td><td>79.3</td><td>93.0</td><td>99.8</td><td>84.0</td></tr><tr><td>Hydra-MDP (Li et al., 2024b)</td><td></td><td>V</td><td>98.3</td><td>96.0</td><td>78.7</td><td>94.6</td><td>100.0</td><td>86.5</td></tr><tr><td>DiffusionDrive (Liao et al., 2025)</td><td></td><td>V</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><td>WoTE (Li et al., 2025b)</td><td></td><td>V</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>DriveVLA-W0 (Li et al., 2025a)</td><td>V</td><td>x</td><td>98.7</td><td>99.1</td><td>83.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><td>DriveFuture (Hong et al., 2026)</td><td>V V</td><td>x</td><td>99.1</td><td>97.2</td><td>84.5</td><td>96.0</td><td>100.0</td><td>90.7</td></tr><tr><td>DriveWorld-VLA (Liu et al., 2026)</td><td></td><td>x</td><td>99.1</td><td>98.2</td><td>85.9</td><td>96.1</td><td>100.0</td><td>91.3</td></tr><tr><td>ReCogDrive (Li et al., 2025c)</td><td>V</td><td>x</td><td>98.1</td><td>97.7</td><td>86.5</td><td>94.9</td><td>100.0</td><td>90.6</td></tr><tr><td>ReCogDrive (Run-then-Walk, ours)</td><td>V</td><td>x</td><td>98.6</td><td>97.9</td><td>85.9</td><td>95.8</td><td>100.0</td><td>91.0</td></tr><tr><td>AutoDrive-P³ (Ye et al., 2026)</td><td>V</td><td>x</td><td>98.9</td><td>97.7</td><td>83.7</td><td>96.6</td><td>99.9</td><td>90.2</td></tr><tr><td>AutoDrive-P³ (Run-then-Walk, ours)</td><td>V</td><td>x</td><td>98.6</td><td>97.3</td><td>88.9</td><td>95.5</td><td>100.0</td><td>91.8</td></tr></table>

Table 2: Performance comparison on NAVSIMv2 benchmark (Navtest).
<table><tr><td>Method</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td> $\mathbf { E P \uparrow }$ </td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS↑</td></tr><tr><td>Human</td><td>100.0</td><td>100.0</td><td>99.8</td><td>100.0</td><td>87.4</td><td>100.0</td><td>100.0</td><td>98.1</td><td>90.1</td><td>94.5</td></tr><tr><td>Ego Status MLP</td><td>93.1</td><td>77.9</td><td>92.7</td><td>99.6</td><td>86.0</td><td>91.5</td><td>89.4</td><td>98.3</td><td>85.4</td><td>64.0</td></tr><tr><td>Transfuser (Prakash et al., 2021)</td><td>96.9</td><td>89.9</td><td>97.8</td><td>99.7</td><td>87.1</td><td>95.4</td><td>92.7</td><td>98.3</td><td>87.2</td><td>84.0</td></tr><tr><td>DiffusionDrive (Liao et al., 2025)</td><td>98.2</td><td>96.2</td><td>99.5</td><td>99.8</td><td>87.4</td><td>97.3</td><td>96.9</td><td>98.4</td><td>87.7</td><td>88.2</td></tr><tr><td>WoTE (Li et al., 2025b)</td><td>98.5</td><td>96.8</td><td>98.8</td><td>99.8</td><td>86.1</td><td>97.9</td><td>95.5</td><td>98.3</td><td>82.9</td><td>87.7</td></tr><tr><td>DriveWorld-VLA (Liu et al., 2026)</td><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>86.8</td></tr><tr><td>DriveVLA-W0 (Li et al., 2025a)</td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>Latent-WAM (Wang et al., 2026)</td><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr><tr><td>ReCogDrive (Li et al., 2025c)</td><td>98.2</td><td>97.7</td><td>98.4</td><td>100.0</td><td>89.9</td><td>97.4</td><td>90.8</td><td>97.7</td><td>26.8</td><td>82.7</td></tr><tr><td>ReCogDrive (Run-then-Walk, ours)</td><td>98.5</td><td>97.9</td><td>98.8</td><td>100.0</td><td>89.4</td><td>97.8</td><td>92.0</td><td>98.2</td><td>29.8</td><td>83.7</td></tr><tr><td>AutoDrive-P³ (Ye et al., 2026)</td><td>98.9</td><td>97.6</td><td>98.9</td><td>99.8</td><td>86.8</td><td>98.5</td><td>95.4</td><td>98.3</td><td>80.6</td><td>88.7</td></tr><tr><td>AutoDrive  $\mathbf { \cdot P ^ { 3 } }$  (Run-then-Walk, ours)</td><td>98.6</td><td>97.3</td><td>98.7</td><td>99.7</td><td>92.8</td><td>98.1</td><td>96.1</td><td>97.5</td><td>78.3</td><td>89.6</td></tr></table>

Starting from the SFT checkpoint $\pi _ { \mathrm { s f t } }$ , we first run GRPO with $R _ { \mathrm { p l a n } } ^ { \mathrm { r u n } }$ and use $\pi _ { \mathrm { s f t } }$ as the reference policy. This Run stage expands the progress distribution and discovers high-progress modes. We then select a Run checkpoint $\pi _ { \mathrm { r u n } }$ and continue GRPO with $R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } }$ , using $\pi _ { \mathrm { r u n } }$ as the new reference policy. The Walk stage therefore repairs safety around the progress-oriented behaviors discovered in Run, rather than collapsing back to the original conservative SFT solution.

## 4.3 IDEALIZED THEORETICAL JUSTIFICATION

The following analysis is idealized and serves as a reward-direction explanation rather than a convergence guarantee. Run-then-Walk separates progress discovery from safety repair. On a fixed query distribution $q \sim \rho ,$ let $\pi ( x \mid q )$ denote a complete planner-sample distribution and define

$$
J ( \pi ) = \mathbb { E } _ { \rho , \pi } [ S Q ] , \qquad Q = \alpha P + ( 1 - \alpha ) B , \qquad 0 \le S , P , B \le 1 ,\tag{8}
$$

where P is progress, $S$ contains multiplicative safety/compliance factors, and B contains the remaining quality terms. This covers PDMS $( S ~ = ~ \mathrm { { N C D A C } , ~ \alpha ~ = ~ 5 / 1 2 } )$ and EPDMS $( S \ =$ NC DAC DDC TLC, $\alpha = 5 / 1 6 )$ . Since $J \le$ min{E[S], E[Q]}, progress alone cannot compensate for poor safety.

In an idealized population update anchored at $\pi _ { R } ~ = ~ \pi _ { \mathrm { r u n } }$ , maximizing $\mathbb { E } [ U ] - \beta D _ { \mathrm { K L } } ( \pi \Vert \pi _ { R } )$ reweights samples by $e ^ { U / \beta }$ . Walk sets $W = R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } }$ to zero when the safety gate fails and gives $W \geq 3 / 2$ when it passes. Thus a positive total-reward gap increases the probability of gatepassing samples. If the range of unchanged auxiliary rewards is at most $\omega ,$ a sufficient gap is $( 3 / 2 ) \lambda _ { \mathrm { p l a n } } - \omega > 0$ . This condition is stated per implementation and makes the argument independent of any particular planner architecture.

Table 3: Performance comparison on Navhard benchmark.
<table><tr><td>Methods</td><td>EPDMS↑</td><td>Stage</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td></tr><tr><td rowspan="2">LTF (Chitta et al., 2022)</td><td rowspan="2">23.1</td><td>1</td><td>96.2</td><td>79.5</td><td>99.1</td><td>99.5</td><td>84.1</td><td>95.1</td><td>94.2</td><td>97.5</td><td>79.1</td></tr><tr><td>2</td><td>77.7</td><td>70.2</td><td>84.2</td><td>98.0</td><td>85.1</td><td>75.6</td><td>45.4</td><td>95.7</td><td>75.9</td></tr><tr><td rowspan="2">DiffusionDrive (Liao et al., 2025)</td><td>28.9</td><td>1</td><td>96.8</td><td>88.2</td><td>99.3</td><td>99.3</td><td>84.5</td><td>94.6</td><td>95.5</td><td>97.5</td><td>79.1</td></tr><tr><td></td><td>2</td><td>80.3</td><td>74.4</td><td>86.1</td><td>98.4</td><td>87.4</td><td>76.9</td><td>50.4</td><td>95.5</td><td>69.4</td></tr><tr><td rowspan="2">GoalFlow (Xing et al., 2025b)</td><td>28.7</td><td>1</td><td>96.0</td><td>92.6 78.2</td><td>99.3</td><td>99.3</td><td>84.0 86.5</td><td>95.7</td><td>97.1</td><td>97.5</td><td>40.4</td></tr><tr><td></td><td>2</td><td>79.4</td><td></td><td>86.4</td><td>97.7</td><td></td><td>76.0</td><td>45.5</td><td>94.4</td><td>40.4</td></tr><tr><td rowspan="2">ReCogDrive (Li et al., 2025c)</td><td>26.5</td><td></td><td>95.4</td><td>90.0</td><td>96.0</td><td>99.6</td><td>86.3</td><td>94.4</td><td>90.9</td><td>97.1</td><td>28.9</td></tr><tr><td></td><td>121</td><td>76.8</td><td>72.6</td><td>80.4</td><td>98.3</td><td>90.5</td><td>73.8</td><td>42.2</td><td>95.6</td><td>34.7</td></tr><tr><td rowspan="2">ReCogDrive (Run-then-Walk, ours)</td><td>28.2</td><td></td><td>95.6</td><td>91.3</td><td>96.8</td><td>99.8</td><td>85.9</td><td>95.3</td><td>92.2</td><td>97.8</td><td>32.9</td></tr><tr><td></td><td>2</td><td>78.3</td><td>73.3</td><td>81.0</td><td>98.8</td><td>89.2</td><td>75.0</td><td>43.0</td><td>95.8</td><td>36.7</td></tr><tr><td rowspan="2">AutoDrive-P³ (Ye et al., 2026)</td><td>29.3</td><td></td><td>96.8</td><td>88.7</td><td>97.1</td><td>99.6</td><td>84.0</td><td>95.1</td><td>93.1</td><td>97.6</td><td>68.4</td></tr><tr><td></td><td>12</td><td>81.2</td><td>68.3</td><td>82.7</td><td>98.4</td><td>86.4</td><td>77.8</td><td>42.3</td><td>95.4</td><td>62.4</td></tr><tr><td rowspan="2">AutoDrive-P³ (Run-then-Walk, ours)</td><td>31.3</td><td></td><td>97.1</td><td>86.9</td><td>96.9</td><td>99.3</td><td>89.9</td><td>95.8</td><td>96.9</td><td>97.6</td><td>61.8</td></tr><tr><td></td><td>12</td><td>81.6</td><td>72.1</td><td>81.5</td><td>98.0</td><td>91.5</td><td>76.3</td><td>50.1</td><td>95.0</td><td>52.0</td></tr></table>

Table 4: Performance comparison on nuScenes Benchmark.
<table><tr><td rowspan="2">Method</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision (%) ↓</td><td rowspan="2">VLM</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 (Hu et al., 2022)</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td><td></td></tr><tr><td>VAD (Jiang et al., 2023)</td><td>0.17</td><td>0.34</td><td>0.60</td><td>0.37</td><td>0.07</td><td>0.10</td><td>0.24</td><td>0.14</td><td></td></tr><tr><td>Ego-MLP (Li et al., 2024c)</td><td>0.46</td><td>0.76</td><td>1.12</td><td>0.78</td><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td><td></td></tr><tr><td>UniAD (Hu et al., 2023)</td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td><td></td></tr><tr><td>InsightDrive (Song et al., 2025)</td><td>0.23</td><td>0.41</td><td>0.68</td><td>0.44</td><td>0.09</td><td>0.10</td><td>0.27</td><td>0.15</td><td></td></tr><tr><td>GPT-Driver (Mao et al., 2023)</td><td>0.20</td><td>0.40</td><td>0.70</td><td>0.44</td><td>0.04</td><td>0.12</td><td>0.36</td><td>0.17</td><td>GPT-3.5</td></tr><tr><td>DriveVLM (Tian et al., 2024)</td><td>0.18</td><td>0.34</td><td>0.68</td><td>0.40</td><td>0.10</td><td>0.22</td><td>0.45</td><td>0.27</td><td>Qwen2-VL-7B</td></tr><tr><td>OpenEMMA (Xing et al., 2025a)</td><td>1.45</td><td>3.21</td><td>3.76</td><td>2.81</td><td></td><td></td><td></td><td></td><td>Qwen2-VL-7B</td></tr><tr><td>RDA-Driver (Huang et al., 2024)</td><td>0.17</td><td>0.37</td><td>0.69</td><td>0.40</td><td>0.01</td><td>0.05</td><td>0.26</td><td>0.10</td><td>LLaVA-7B</td></tr><tr><td>OmniDrive (Wang et al., 2024)</td><td>0.14</td><td>0.29</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.04</td><td>0.27</td><td>0.11</td><td>LLaVA-7B</td></tr><tr><td>OpenDriveVLA (Zhou et al., 2025a)</td><td>0.14</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.02</td><td>0.07</td><td>0.22</td><td>0.10</td><td>Qwen2.5-VL-3B</td></tr><tr><td>AutoVLA (Zhou et al., 2025b)</td><td>0.25</td><td>0.46</td><td>0.73</td><td>0.48</td><td>0.07</td><td>0.07</td><td>0.26</td><td>0.13</td><td>Qwen2.5-VL-3B</td></tr><tr><td>AutoDrive-R² (Yuan et al., 2025)</td><td>0.35</td><td>0.49</td><td>0.62</td><td>0.49</td><td></td><td>一</td><td>一</td><td></td><td>Qwen2.5-VL-3B</td></tr><tr><td>AutoDrive-P³ (Ye et al., 2026)</td><td>0.16</td><td>0.31</td><td>0.56</td><td>0.34</td><td>0.00</td><td>0.04</td><td>0.20</td><td>0.08</td><td>Qwen2.5-VL-3B</td></tr><tr><td>AutoDrive-P³ (Run-then-Walk, ours)</td><td>0.15</td><td>0.30</td><td>0.55</td><td>0.33</td><td>0.01</td><td>0.03</td><td>0.16</td><td>0.06</td><td>Qwen2.5-VL-3B</td></tr></table>

If the achieved full-sample KL to $\pi _ { R }$ is at most $\kappa ,$ Pinsker's inequality gives $\varepsilon _ { \kappa } = \operatorname* { m i n } \{ 1 , \sqrt { \kappa / 2 } \}$ and, for $a _ { R } = \mathrm { P r } _ { R } ( P \geq p _ { 0 } )$ and $f _ { W } = \operatorname* { P r } _ { W } ( S < s _ { 0 } )$ 2

$$
\mathbb { E } _ { W } [ P ] \geq \mathbb { E } _ { R } [ P ] - \varepsilon _ { \kappa } , \qquad J ( \pi _ { W } ) \geq \alpha p _ { 0 } s _ { 0 } [ a _ { R } - \varepsilon _ { \kappa } - f _ { W } ] _ { + } .\tag{9}
$$

Run must therefore create high-progress mass, Walk must reduce safety failures, and the reference must retain enough of that mass. Writing $s _ { t } = \mathbb { E } _ { t } [ S ]$ and $c _ { t } = \mathbb { E } _ { t } [ S Q ] / s _ { t } .$ , a quality-loss bound $c _ { W } \geq c _ { R } - \ell$ yields

$$
J ( \pi _ { W } ) - J ( \pi _ { R } ) \ge ( s _ { W } - s _ { R } ) c _ { R } - s _ { W } \ell ,\tag{10}
$$

so safety improvement must exceed quality lost among safety-weighted trajectories.

These statements require positive Run support for useful safe-progress samples, adequate group coverage, a fixed evaluation distribution, and a measured rather than nominal KL bound. A zero KL forbids repair, and our experiments further show that removing the KL term leads to training collapse; excessive KL weakens progress retention. The Walk gate is not identical to EPDMS safety, and endpoint proximity is neither a path-safety nor a progress certificate. Full derivations, the idealized probability-ratio bound, and finite-sampling conditions are provided in Appendix B.

## 5 EXPERIMENTS

Benchmarks and baselines. We evaluate Run-then-Walk on four benchmarks: NAVSIMv1 (Dauner et al., 2024), NAVSIMv2 (Cao et al., 2025), Navhard (Cao et al., 2025), and nuScenes (Caesar et al., 2020). To validate the robustness of our method, we evaluate it on two representative VLM planners: the autoregressive AutoDrive-P³ (fast-mode) and the diffusion-based ReCogDrive. NAVSIMv1 uses the Predictive Driver Model Score (PDMS), whereas NAVSIMv2 and Navhard use the Extended PDMS (EPDMS). nuScenes provides trajectory-error and collision evaluation. Since only

![](images/10d8f7f335158f964addf7d41535d220fd215aa0b421d39dc0024662ec94a0f0.jpg)  
Figure 5: PDMS sub-metric evolution during AutoDrive-P3 Run-then-Walk strategy. Red curves denote the 3-epoch Run-GRPO after SFT, green curves denote the following 2-epoch Walk-GRPO after Run-GRPO. Run-GRPO increases progress but hurts safety; Walk-GRPO then restores safety, lifts PDMS above the SFT baseline, and yields a safe, high-progress policy in only 5 epochs.

Table 5: Run/Walk recipe comparison.
<table><tr><td>Different RL recipes</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>Comf↑</td><td>PDMS↑</td></tr><tr><td>Only Run</td><td>94.8</td><td>95.1</td><td>88.8</td><td>84.8</td><td>100.0</td><td>85.7</td></tr><tr><td>Only Walk</td><td>99.1</td><td>97.7</td><td>81.5</td><td>97.3</td><td>100.0</td><td>89.7</td></tr><tr><td>Walk-and-Run</td><td>98.0</td><td>94.2</td><td>85.4</td><td>94.0</td><td>100.0</td><td>85.9</td></tr><tr><td>Walk-then-Run</td><td>98.9</td><td>97.7</td><td>83.7</td><td>96.6</td><td>99.9</td><td>90.2</td></tr><tr><td>Ours (Run-then-Walk)</td><td>98.6</td><td>97.3</td><td>88.9</td><td>95.5</td><td>100.0</td><td>91.8</td></tr></table>

Table 6: Effect of safety metrics in Walk.
<table><tr><td colspan="3">Reward Signals</td><td colspan="6">Evaluation Metrics</td></tr><tr><td>DAC</td><td>NC</td><td>TTC</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>Comf↑</td><td>PDMS↑</td></tr><tr><td>V</td><td>x</td><td>x</td><td>98.5</td><td>97.1</td><td>88.0</td><td>94.9</td><td>100.0</td><td>91.5</td></tr><tr><td></td><td>V</td><td>x</td><td>98.6</td><td>97.3</td><td>88.9</td><td>95.5</td><td>100.0</td><td>91.8</td></tr><tr><td>ンン</td><td>V</td><td>V</td><td>98.7</td><td>97.4</td><td>86.9</td><td>96.0</td><td>100.0</td><td>91.2</td></tr></table>

AutoDrive. $\cdot \mathrm { P ^ { 3 } }$ has been trained on nuScenes, we restrict the nuScenes comparison to this planner.   
We set $R _ { \mathrm { a g g } }$ to PDMS, and for the endpoint reward, we set ∆ = 10 and $\eta = 0 . 2$ in all experiments.

Main results across planners and benchmarks. Tables 1 and 2 show consistent improvements for both planner families rather than a gain tied to one base model. On NAVSIMv1, Run-then-Walk raises the paired diffusion-planner PDMS from 90.6 to 91.0 and the paired autoregressive-planner PDMS from 90.2 to 91.8. The ReCogDrive comparison improves all safety-related metrics, only with a small decrease in EP, whereas the AutoDrive-P³ comparison highlights increased progress while retaining strong safety. For AutoDrive-P3, the schedule uses 3 Run epochs and 2 Walk epochs, for 5 RL epochs in total. For ReCogDrive, we use 5 Run epochs and 1 Walk epoch, for 6 RL epochs in total. Thus the method saves 40–50% of the 10-epoch baseline training steps, converges faster, and improves performance for both planners. On NAVSIMv2 (Table 2), paired EPDMS rises from 82.7 to 83.7 for ReCogDrive and from 88.7 to 89.6 for AutoDrive-P³with EP 92.8. On the twostage Navhard protocol, as shown in Table 3, paired aggregates improve from 26.5 to 28.2 and from 29.3 to 31.3, respectively. On nuScenes, as shown in Table 4, AutoDrive-P³ planner reduces average trajectory error and collision rate. Across different benchmarks and planner methods, our Run-then-Walk strategy not only better balances safety and progress, thereby improving driving performance, but also substantially saves training time.

## 6 ANALYSIS AND VISUALIZATION

Analysis of Run-then-Walk Training Strategy. Figure 5 reveals the behavioral evolution of AutoDrive-P3 across the two training stages. During Run-GRPO, the EP rises substantially (81.7→88.8), but the safety metrics drop, indicating that the policy learns to maximize progress at the cost of frequent collisions and unsafe events, ultimately degrading PDMS from 87.8 to 85.7. In the subsequent Walk-GRPO, safety is strongly restored, while the high EP is preserved, and the final checkpoint reaches 91.8 PDMS. This AutoDrive-P³ result uses only 5 RL epochs, and the same safety-repair principle applied to ReCogDrive reaches 91.0 PDMS in 6 epochs, together saving 40– 50% of RL training steps relative to the 10-epoch baselines while improving performance. More detailed analysis, including ReCogDrive, is provided in the supplementary material.

Analysis of different RL recipes. Table 5 compares different orderings of Run and Walk objectives Only Run yields high EP (88.8) but the lowest PDMS (85.7) due to poor safety; Only Walk achieves the best safety but limits progress (EP 81.5, PDMS 89.7). Walk-then-Run is suboptimal, whereas our Run-then-Walk achieves the highest PDMS by exploring progress first and restoring safety afterward.

![](images/d4719e2442d59ecc4772e857f54b753157e9d71a8d028d5f37936c9a6fed3cb1.jpg)

![](images/8c18627ec9b1e15eed2bf4ddef22fb6d96250fbc74bbd0db13950dbdb4272d7a.jpg)

![](images/40b076bfead2576a60e8f652c870b55b6ff84e480dce792c93f1a1ffe168dbf4.jpg)

![](images/832d63804f1dfb5f8373d8f63b84dbc681c2d6546193566791809ab3b6954deb.jpg)  
Figure 6: Qualitative trajectory visualization.

Effect of safety signal composition. Table 6 ablates which safety signals are used in the Walkphase planning reward $R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } }$ (Eq. 4). Using only DAC yields 91.5 PDMS. Adding NC improves the balance to 91.8 PDMS and 88.9 EP. Further adding TTC raises safety scores but lowers EP to 86.9, reducing PDMS to 91.2. These results suggest that DAC+NC provides the best safety-progress trade-off, while TTC as a reward signal introduces over-conservatism.

Visualization. Figure 6 presents qualitative trajectory comparisons on two representative driving scenarios. Walk-and-Run achieves high progress but collides (NC = 0), and Walk-then-Run remains safe but advances less (EP = 0.65). Our method maintains a high EP while ensuring safety.

## 7 CONCLUSION

In this paper, we first reveal two RL training recipe regimes: a progress regime (Run-GRPO) that aggressively explores high progress and a safety regime (Walk-GRPO) that restores safety. Then, we present Run-then-Walk, a simple and effective two-stage RL training scheduling strategy for VLMbased autonomous driving. By first exploring progress under relaxed constraints and then repairing safety with endpoint rewards, our method escapes SFT conservatism while avoiding the unsafe driving. Across VLM-based autoregressive and diffusion planners on different benchmarks, experiments show improved driving performance and a 40–50% reduction in RL training steps. These findings demonstrate that a progress-first, safety-refinement curriculum is efficient and effective, and we hope this simple paradigm inspires future RL in broader embodied decision-making.

## REFERENCES

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631, 2020.

Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218, 2025.

Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022.

Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yufeng Hong, Xiaotian Zhou, Yingyan Li, Xiangpo Zhou, Lin Liu, Yadan Luo, Shaoqing Xu, Lei Yang, and Ziying Song. Drivefuture: Future-aware latent world models for autonomous driving. arXiv preprint arXiv:2605.09701, 2026.

Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-toend vision-based autonomous driving via spatial-temporal feature learning. In European Conference on Computer Vision, pp. 533–549. Springer, 2022.

Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17853–17862, 2023.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Zhijian Huang, Tao Tang, Shaoxiang Chen, Sihao Lin, Zequn Jie, Lin Ma, Guangrun Wang, and Xiaodan Liang. Making large language models better planners with reasoning-decision alignment. In European Conference on Computer Vision, pp. 73–90. Springer, 2024.

Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262, 2024.

Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350, 2023.

Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608, 2025.

Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481, 2024a.

Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025a.

Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941, 2025b.

Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-toend autonomous driving. arXiv preprint arXiv:2506.08052, 2025c.

Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydradistillation. arXiv preprint arXiv:2406.06978, 2024b.

Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14864–14873, 2024c.

Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for endto-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12037–12047, 2025.

Lin Liu, Ziying Song, Caiyan Jia, Hangjun Ye, Xiaoshuai Hao, Long Chen, et al. Driveworld-vla: Unified latent-space world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521, 2026.

Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023.

Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-toend autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7077–7087, 2021.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Zhihong Shao, Yuxiang Luo, Chengda Lu, ZZ Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025.

Ruiqi Song, Xianda Guo, Hangbin Wu, Qinggong Wei, and Long Chen. Insightdrive: Insight scene representation for end-to-end autonomous driving. arXiv preprint arXiv:2503.13047, 2025.

Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024.

Linbo Wang, Yupeng Zheng, Qiang Chen, Shiwei Li, Yichen Zhang, Zebin Xing, Qichao Zhang, Xiang Li, Deheng Qian, Pengxuan Yang, et al. Latent-wam: Latent world action modeling for end-to-end autonomous driving. arXiv preprint arXiv:2603.24581, 2026.

Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. CoRR, 2024.

Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15449–15458, 2024.

Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. Openemma: Open-source multimodal model for end-to-end autonomous driving. In Proceedings of the Winter Conference on Applications of Computer Vision, pp. 1001–1009, 2025a.

Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1602–1611, June 2025b.

Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P Meyer, Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M Wolff, and Xin Huang. Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024.

Yuqi Ye, Zijian Zhang, Junhong Lin, Shangkun Sun, Changhao Peng, and Wei Gao. AutoDrive-P3: Unified chain of perception-prediction-planning thought via reinforcement fine-tuning. In The Fourteenth International Conference on Learning Representations, 2026.

Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. Autodrive-r2: Incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. arXiv preprint arXiv:2509.01944, 2025.

Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463, 2025a.

Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757, 2025b.

## A REWARD FUNCTION DETAILS

## A.1 PLANNER-AGNOSTIC REWARD INTERFACE

For each driving query $q ,$ we sample a group of $G$ complete planner outputs $\{ x _ { i } \} _ { i = 1 } ^ { G }$ from $\pi _ { \theta }$ and decode each output into a trajectory $\tau _ { i } .$ An autoregressive planner assigns the resulting reward to the sampled token sequence, whereas a diffusion planner assigns it to the sampled continuous trajectory. In both cases, the total reward is

$$
R _ { i } ^ { ( s ) } = R _ { \mathrm { a u x } } ( x _ { i } ) + \lambda _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } ^ { ( s ) } ( \tau _ { i } ) , \qquad s \in \{ \mathrm { r u n } , \mathrm { w a l k } \} .\tag{11}
$$

Here $R _ { \mathrm { a u x } }$ denotes any native auxiliary objectives of the underlying planner. These objectives and the output representation remain unchanged between the two stages. Rewards are normalized within each sampled group and optimized with the GRPO objective in Section 4. Consequently, the only algorithmic change shared by both planner families is the planning reward and its Run-to-Walk schedule.

## A.2 PLANNING REWARD: RUN VS. WALK

The planning reward is the only reward component that differs between the Run and Walk stages

Run Phase Planning Reward. In the Run phase, we directly use the closed-loop PDMS as the planning reward:

$$
R _ { \mathrm { p l a n } } ^ { \mathrm { r u n } } ( \tau _ { i } ) = \mathrm { P D M S } ( \tau _ { i } ) .\tag{12}
$$

No endpoint/L2 constraint or explicit safety reward is applied. This intentionally encourages the policy to maximize progress under relaxed constraints.

Walk Phase Planning Reward. In the Walk phase, we replace the PDMS-based planning reward with a composite of endpoint and safety rewards. The endpoint reward is defined via the $L _ { 1 }$ distance $d _ { E } = \lVert p _ { T } - g _ { T } \rVert _ { 1 }$ between the final predicted point $p _ { T }$ and the expert endpoint $g _ { T }$

$$
k ( d _ { E } ) = \operatorname* { m i n } ( K , \operatorname* { m a x } ( 0 , \lfloor ( d _ { E } - 2 \Delta ) / \Delta \rfloor + 1 ) ) , \qquad R _ { \mathrm { e n d } } ( d _ { E } ) = 1 - \eta k ( d _ { E } ) ,\tag{13}
$$

where $\Delta > 0$ is the step size, $\eta \in ( 0 , 1 )$ is the reward decrement, and $K = \lfloor 1 / \eta \rfloor$ . We use the fixed configuration $\Delta = 1 0 , \eta = 0 . 2$ throughout. The complete Walk planning reward is:

$$
R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } } ( \tau _ { i } , \tau _ { i } ^ { \star } ) = \mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) \Bigl ( \mathrm { N C } ( \tau _ { i } ) + \mathrm { D A C } ( \tau _ { i } ) + R _ { \mathrm { e n d } } ( d _ { E } ) \Bigr ) ,\tag{14}
$$

where $\mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) = 1 \mathrm { o n l y ~ i f ~ N C } ( \tau _ { i } ) > 0$ , and $\mathrm { D A C } ( \tau _ { i } ) > 0 ;$ otherwise the entire planning reward is zeroed. Algorithm 1 presents the complete Run-then-Walk scheduling strategy procedure.

## B IDEALIZED THEORETICAL JUSTIFICATION

This appendix provides the full derivations for the idealized theoretical justification summarized in Section 4.3. They are conditional statements about a fixed query distribution, not a claim that every finite-sample GRPO run must improve. Let $q \sim \rho$ and let $x$ be a complete planner sample (an autoregressive response or a diffusion sample) with induced trajectory $\tau ( x )$ . We write

$$
J ( \pi ) = \mathbb { E } _ { \rho , \pi } [ S Q ] , \qquad Q = \alpha P + ( 1 - \alpha ) B , \qquad 0 \le S , P , B \le 1 ,\tag{15}
$$

where $P$ is normalized progress, $S$ is the product of safety/compliance factors, and B contains the remaining quality terms. For PDMS, $S ~ = ~ \mathrm { N C D \bar { A } C }$ and $\alpha ~ = ~ 5 / 1 2 $ for EPDMS, $S = { \mathrm { N C D A C D D C } } { \mathrm { T L C } }$ and $\alpha = 5 / 1 6$ Since $0 \leq S , Q \leq 1 , J \leq \mathbb { E } [ S ]$ and $J \le \mathbb { E } [ Q ]$ . This is the basic progress-safety bottleneck: high progress cannot compensate for a vanishing safety factor.

Safety direction of the Walk reward. Fix q and let $\pi _ { R } = \pi _ { \mathrm { r u n } } , H = \mathbb { I } _ { \mathrm { s a f e } }$ , and $W = R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } }$ . Let C denote the unchanged architecture-native auxiliary reward and $U = C + \lambda _ { \mathrm { p l a n } } W$ . Consider the idealized population problem

$$
\pi ^ { \dagger } = \arg \operatorname* { m a x } _ { \pi } \left\{ \mathbb { E } _ { \pi } [ U ] - \beta D _ { \mathrm { K L } } ( \pi \| \pi _ { R } ) \right\} , \qquad \pi ^ { \dagger } ( x \mid q ) = \frac { \pi _ { R } ( x \mid q ) e ^ { U ( q , x ) / \beta } } { Z _ { q } } ,\tag{16}
$$

Algorithm 1 Run-then-Walk training scheduling strategy   
Require: SFT policy $\pi _ { \mathrm { s f t } }$ , queries $\mathcal { Q } ,$ group size $G ,$ auxiliary reward $R _ { \mathrm { a u x } } .$ planning weight $\lambda _ { \mathrm { p l a n } } .$   
KL coefficients $\beta _ { \mathrm { r u n } } , \beta _ { \mathrm { w a l k } }$   
Ensure: Optimized driving policy $\pi _ { \mathrm { w a l k } }$   
— Stage 1: Run-GRPO (progress discovery) —   
1: Initialize π $ \pi _ { \mathrm { s f t } } .$ set reference $\pi _ { \mathrm { r e f } }  \pi _ { \mathrm { s f t } }$   
2: for each query $q \in \mathcal { Q }$ do   
3: Sample outputs $\bar { \{ } x _ { i } \} _ { i = 1 } ^ { G } \sim \pi ( \cdot \mid q )$ and decode trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$   
4: for $\bar { i } = 1$ to $G$ do   
5: $R _ { \mathrm { p l a n } } ^ { i }  R _ { \mathrm { p l a n } } ^ { r u n } ( \tau _ { i } )$ $\triangleright$ Focus on progress   
6: $\dot { R _ { i } }  \dot { R _ { \mathrm { a u x } } ( x _ { i } ) } + \lambda _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } ^ { i }$   
7: end for   
8: Compute advantages $\hat { A } _ { i } = ( R _ { i } - \bar { R } ) / \sigma _ { R }$ and update $\pi$ via GRPO with KL penalty $\beta _ { \mathrm { r u n } }$   
9: end for   
10: Select Run checkpoint $\pi _ { \mathrm { r u n } }$   
— Stage 2: Walk-GRPO (safety repair) —   
11: Initialize $\pi  \pi _ { \mathrm { r u n } } ,$ set reference $\pi _ { \mathrm { r e f } }  \pi _ { \mathrm { r u n } }$   
12: for each query $q \in \mathcal { Q }$ do   
13: Sample outputs $\bar { \{ } x _ { i } \} _ { i = 1 } ^ { G } \sim \pi ( \cdot \mid q )$ and decode trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$   
14: for $i = 1$ to $G$ do   
15: $R _ { \mathrm { p l a n } } ^ { i }  R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } } ( \tau _ { i } , \tau _ { i } ^ { \star } )$ $\triangleright$ Endpoint + safety gate   
16: $\dot { R _ { i } }  \dot { R _ { \mathrm { a u x } } ( x _ { i } ) } + \lambda _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } ^ { i }$   
17: end for   
18: Compute advantages $\hat { A } _ { i } = ( R _ { i } - \bar { R } ) / \sigma _ { R }$ and update $\pi$ via GRPO with KL penalty $\beta _ { \mathrm { w a l l } }$ X   
19: end for   
20: return $\pi _ { \mathrm { w a l l } }$ X

where $Z _ { q } = \mathbb { E } _ { \pi _ { R } } [ e ^ { U / \beta } ]$ and $\beta > 0$ . The variational identity

$$
\begin{array} { r } { \mathbb { E } _ { \pi } [ U ] - \beta D _ { \mathrm { K L } } ( \pi \| \pi _ { R } ) = \beta \log Z _ { q } - \beta D _ { \mathrm { K L } } ( \pi \| \pi ^ { \dagger } ) } \end{array}\tag{17}
$$

proves the optimizer. Suppose both values of $H$ have positive reference probability and every passing sample has total reward at least $\gamma > 0$ larger than every failing sample. If $h _ { R } \dot { = } \operatorname* { P r } _ { \pi _ { R } } ( \dot { H } \dot { = } 1 |$ q) and $\bar { h ^ { \dagger } } = \operatorname* { P r } _ { \pi ^ { \dagger } } ( H = 1 \mid q )$ , then

$$
\frac { h ^ { \dagger } } { 1 - h ^ { \dagger } } = \frac { h _ { R } } { 1 - h _ { R } } \frac { \mathbb { E } _ { \pi _ { R } } [ e ^ { U / \beta } \mid H = 1 ] } { \mathbb { E } _ { \pi _ { R } } [ e ^ { U / \beta } \mid H = 0 ] } \ge e ^ { \gamma / \beta } \frac { h _ { R } } { 1 - h _ { R } } ,\tag{18}
$$

and therefore

$$
1 - h ^ { \dagger } \le \frac { 1 - h _ { R } } { 1 - h _ { R } + h _ { R } e ^ { \gamma / \beta } } .\tag{19}
$$

The Walk construction supplies a concrete sufficient reward gap. Recall that

$$
R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } } ( \tau _ { i } , \tau _ { i } ^ { \star } ) = \mathbb { I } _ { \mathrm { s a f e } } ( \tau _ { i } ) \Bigl ( \mathrm { N C } ( \tau _ { i } ) + \mathrm { D A C } ( \tau _ { i } ) + R _ { \mathrm { e n d } } ( d _ { E } ) \Bigr ) ,
$$

where $\mathrm { N C } \in \{ 0 , 1 / 2 , 1 \} , \mathrm { D A C } \in \{ 0 , 1 \}$ , and $R _ { \mathrm { e n d } } \in [ 0 , 1 ]$ . When the safety gate passes, we have $\mathrm { N C } > 0$ and $\begin{array} { r } { \mathrm { D A C } > 0 ; } \end{array}$ hence $\mathrm { N C } \geq 1 / 2 , \mathrm { D A C } = 1 .$ and $\dot { R } _ { \mathrm { e n d } } \geq 0$ . Therefore

$$
W = R _ { \mathrm { p l a n } } ^ { \mathrm { w a l k } } \geq \frac { 1 } { 2 } + 1 + 0 = \frac { 3 } { 2 } .
$$

When the gate fails, $\mathbb { I } _ { \mathrm { s a f e } } = 0$ and thus $W = 0$ . Let $C$ denote the unchanged auxiliary reward, whose range at a fixed query is at most $\omega ,$ and let the total reward be $U = C + \lambda _ { \mathrm { p l a n } } W$ . Then every passing sample satisfies

$$
U _ { \mathrm { p a s s } } \geq C _ { \mathrm { m i n } } + \frac { 3 } { 2 } \lambda _ { \mathrm { p l a n } } ,
$$

while every failing sample satisfies

$$
U _ { \mathrm { f a i l } } \leq C _ { \mathrm { m a x } } .
$$

Hence the reward gap is at least

$$
U _ { \mathrm { p a s s } } - U _ { \mathrm { f a i l } } \geq \frac { 3 } { 2 } \lambda _ { \mathrm { p l a n } } - \left( C _ { \mathrm { m a x } } - C _ { \mathrm { m i n } } \right) \geq \frac { 3 } { 2 } \lambda _ { \mathrm { p l a n } } - \omega .
$$

Thus $\gamma = ( 3 / 2 ) \lambda _ { \mathrm { p l a n } } - \omega > 0$ is sufficient for all passing samples to dominate all failing samples in total reward. The value of ω must be established from the actual normalization and scale of the auxiliary rewards for the implementation under study; it is not fixed by the gate construction alone.

The idealized result describes the local direction of group-normalized GRPO as well. At the Run checkpoint, clipping is inactive and the KL term has zero first derivative. Along $\pi _ { \zeta } ( x \mid q ) \propto \pi _ { R } ( x \mid$ $q ) e ^ { \zeta H ( x ) }$ , the derivative of the sampled surrogate at $\zeta = 0 \mathrm { \ i s \ } G ^ { - 1 } \textstyle \sum _ { i } A _ { i } H _ { i } .$ where $A _ { i }$ are fixed Walk advantages and $H _ { i } = H ( x _ { i } )$ . Since $\begin{array} { r } { \sum _ { i } A _ { i } = 0 } \end{array}$ and all passing rewards exceed all failing rewards, this derivative is nonnegative and is positive when a group contains both outcomes. Shared neural parameters, finite groups, advantage recomputation, and clipping mean that the actual update need not realize this direction or the Gibbs optimizer. The parameter $\beta$ in Eq. (16) is therefore not an empirical improvement-rate calibration.

Progress retention under a KL budget. Let $\pi _ { W } = \pi _ { \mathrm { w a l k } }$ and assume the achieved full-sample KL on the same $\rho$ obeys

$$
\begin{array} { r } { \mathbb { E } _ { q \sim \rho } D _ { \mathrm { K L } } \big ( \pi _ { W } ( \cdot \mid q ) \| \pi _ { R } ( \cdot \mid q ) \big ) \leq \kappa , \qquad \varepsilon _ { \kappa } = \operatorname* { m i n } \{ 1 , \sqrt { \kappa / 2 } \} . } \end{array}\tag{20}
$$

Pinsker ${ \bf \ddot { s } }$ inequality gives conditional total variation at most $\sqrt { D _ { \mathrm { K L } } / 2 } ;$ Jensen's inequality then bounds its mean by $\varepsilon _ { \kappa }$ . Hence any [0, 1]-valued statistic changes by at most $\varepsilon _ { \kappa } .$ including $\dot { P }$ and event indicators. For $a _ { R } = \mathrm { P r } _ { R } ( P \geq p _ { 0 } )$ and $f _ { W } = \operatorname* { P r } _ { W } ( S < s _ { 0 } )$

$$
\begin{array} { r } { \mathbb { E } _ { W } [ P ] \geq \mathbb { E } _ { R } [ P ] - \varepsilon _ { \kappa } , \qquad \operatorname* { P r } ( P \geq p _ { 0 } , S \geq s _ { 0 } ) \geq [ a _ { R } - \varepsilon _ { \kappa } - f _ { W } ] _ { + } . } \end{array}\tag{21}
$$

On the latter event, $S Q \ge \alpha p _ { 0 } s _ { 0 }$ , sO

$$
J ( \pi _ { W } ) \geq \alpha p _ { 0 } s _ { 0 } [ a _ { R } - \varepsilon _ { \kappa } - f _ { W } ] _ { + } .\tag{22}
$$

The bound is nonvacuous only when $a _ { R } > \varepsilon _ { \kappa } + f _ { W }$ . A GRPO KL coefficient does not by itself establish Eq. (20): the bound concerns the complete output distribution on $\rho ,$ not an unconverted average over tokens or diffusion steps. By data processing, full-output KL also bounds the induced trajectory KL. Conversely, reducing the probability of a fixed failure event by d requires $\kappa \geq 2 d ^ { 2 }$ under this bound. Thus zero KL forbids repair, while a very large KL gives little progress retention.

When the final score increases. Safety improvement and progress can be correlated, so marginal submetric improvements are insufficient. Define $s _ { t } = \mathbb { E } _ { t } \bar { | S | } \bar { > } 0$ and $c _ { t } = \mathbb { E } _ { t } [ S Q ] / s _ { t }$ for $t \in$ $\{ R , W \}$ . If the safety-weighted quality loss is bounded by $c _ { W } \geq c _ { R } - \ell .$ then

$$
\begin{array} { r l } & { J ( \pi _ { W } ) - J ( \pi _ { R } ) = s _ { W } c _ { W } - s _ { R } c _ { R } } \\ & { \qquad = ( s _ { W } - s _ { R } ) c _ { R } + s _ { W } ( c _ { W } - c _ { R } ) } \\ & { \qquad \geq ( s _ { W } - s _ { R } ) c _ { R } - s _ { W } \ell . } \end{array}\tag{23}
$$

Therefore $( s _ { W } - s _ { R } ) c _ { R } > s _ { W } \ell$ is sufficient for strict improvement. This condition allows some scenes to worsen and can be estimated from paired per-trajectory scores; separate averages of NC, DAC, and EP do not prove it. If $s _ { R } = 0$ , any positive Walk score is an improvement.

Sampling and boundary conditions. For a query with $p _ { q } \ = \ \mathrm { P r } _ { R } ( H \ = \ 1 , P \ \geq \ p _ { 0 } \ | \ q )$ , an independent group of G samples contains a useful safe-progress candidate with probability $1 - ( 1 -$ $p _ { q } ) ^ { \dot { G } }$ . When $p _ { q }$ is small, the Walk signal is rarely observed; if all group members fai the gate, their planning rewards are all zero and there is no relative planning signal from that term. Finite KL cannot create probability on a reference-null event. These facts justify switching after progress discovery but before the Run checkpoint loses useful safe support, without implying a universal epoch or schedule.

Finally, H is a training gate rather than the full evaluation factor S: it permits $\mathrm { N C } = 1 / 2$ , checks TTC, and omits EPDMS's DDC/TLC terms. Endpoint proximity similarly certifies neither path safety nor progress; a tight endpoint reward may increase l, while a loose one may be nondiscriminative. Human-penalty filters, Navhard's multi-stage aggregation, and policy-induced query shifts require protocol-specific analysis. These qualifications apply equally to autoregressive samples and VLM-conditioned diffusion trajectories.

Table 7: RL training hyperparameters for different VLM-based planners.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="2">AutoDrive-P³ (Ye et al., 2026)</td><td colspan="2">ReCogDrive(Li et al., 2025c)</td></tr><tr><td>Run-GRPO</td><td>Walk-GRPO</td><td>Run-GRPO</td><td>Walk-GRPO</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR Schedule</td><td>constant</td><td>constant</td><td>cosine decay</td><td>cosine decay</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>RL Epochs</td><td>3</td><td>2</td><td>5</td><td>1</td></tr><tr><td>Global Batch Size</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>GRPO Group Size (G)</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Clip Parameter (€)</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Precision / Hardware</td><td>bf16/8×A100</td><td>bf16/8×A100</td><td>bf16/8×A100</td><td>bf16/8×A100</td></tr><tr><td>Trajectory Output</td><td>8 waypoints (4.0 s)</td><td>8 waypoints (4.0 s)</td><td>8 poses (4.0 s, 0.5 s)</td><td>8 poses (4.0 s, 0.5 s)</td></tr><tr><td>Planner Parameterization</td><td>Autoregressive tokens</td><td>Autoregressive tokens</td><td>Diffusion</td><td>Diffusion</td></tr></table>

## C TRAINING HYPERPARAMETERS

Table 7 summarizes the RL training hyperparameters for different VLM-based planners and stages. Note that we stop the first (Run) stage once EP has essentially plateaued, as illustrated in Fig. 3. This is why ReCogDrive requires a longer Run stage than AutoDrive-P3. AutoDrive-P³ starts from an SFT checkpoint with EP around 81.7, whereas ReCogDrive starts from an SFT checkpoint with EP around 80. Therefore, ReCogDrive requires a longer progress-discovery phase before EP plateaus, specifically two additional Run epochs compared with AutoDrive-P3, using five Run epochs in total

In the Run stage, both AutoDrive-P³ and ReCogDrive adopt Eq. 12 as the planning reward, i.e., PDMS is directly used as the planning reward to enhance progress exploration. In the Walk stage, both planners adopt Eq. 14 as the planning reward to improve safety. The difference is that AutoDrive-P³ additionally includes a planning/perception auxiliary reward and combines it with the planning reward through Eq. 11, where $\bar { R _ { \mathrm { a u x } } ( x _ { i } ) }$ denotes the auxiliary reward, to align with its original baseline (Ye et al., 2026). ReCogDrive does not require this additional auxiliary term, as it is consistent with its original baseline (Li et al., 2025c).

## D STAGE-BY-STAGE QUALITATIVE WALKTHROUGH: SFT → RUN → WALK

This section visualizes how a single scene evolves through the three training stages of our pipeline. In Figures 7 and 8, each row contains a wide camera strip (with all four trajectories projected) followed by four bird's-eye-view (BEV) panels that share an identical map/agent layout. From left to right the BEV panels show the Human expert (GT, green), the SFT policy (blue), the Run-GRPO checkpoint (red), and the final Walk-GRPO policy (orange). The PDMS and endpoint-progress (EP) scores are annotated inside every panel.

Stage 1: SFT (conservative imitation). The supervised policy faithfully imitates the human demonstrations and is therefore collision-free, but it is systematically under-progressing: it brakes early and stops short of where a competent human driver would be, yielding low EP and a capped PDMS (typically 0.83–0.93 in these scenes). The blue BEV panel is labeled "Safe, but Low Progress."

Stage 2: Run-GRPO (progress discovery). Optimizing the PDMS-only planning reward without any endpoint/safety constraint pushes the policy to discover much more aggressive, high-progress trajectory modes. As the red BEV panel (“High Progress, but Collision") shows, the Run policy reaches far further down the road, but it frequently leaves the drivable area or collides with an agent, driving its no-collision (NC) score, and hence PDMS, to zero. This is the intentional, temporary safety regression that the Run phase trades for exploration.

Stage 3: Walk-GRPO (safety repair). Re-introducing the endpoint and safety rewards after the Run phase lets the policy keep the newly discovered progress while repairing the unsafe behavior. The orange BEV panel ("High Progress, and $S a f e ^ { \prime \prime } )$ recovers a collision-free trajectory that nonetheless travels almost as far as the Run policy, lifting both EP and PDMS well above the SFT baseline (often to 0.95–1.00 PDMS). Across the six scenes below, spanning straight cruising, signalized intersections, crosswalk approaches, and following large vehicles (buses/trucks), the Walk policy consistently dominates SFT on progress while matching its safety, which is precisely the behavior our Run-then-Walk ordering is designed to produce.

## E QUALITATIVE COMPARISON

The following four pages present NAVSIM cases for a qualitative comparison. Each triplet shows the human reference (green), the corresponding baseline (orange), and Run-then-Walk (blue) in the same scene. Figures 9 and 10 isolate the loss of driving efficiency from conservative Walk-then-Run training in AutoDrive- $\cdot \mathrm { P ^ { 3 } }$ (Ye et al., 2026); Figures 11 and 12 isolate the collision risk of progressdominant Walk-and-Run training in ReCogDrive (Li et al., 2025c).

Walk-then-Run (AutoDrive-P3): safe but low progress. In Figures 9 and 10, the baseline trajectories remain safe but end earlier than ours. This conservative behavior reduces driving efficiency despite avoiding unsafe maneuvers, whereas discovering progress before safety refinement yields longer safe trajectories.

Walk-and-Run (ReCogDrive): excessive progress seeking compromises safety. In all six examples in Figures 11 and 12, the baseline pushes forward but collides: NC is 0 and the collision-gated PDMS is 0. Run-then-Walk retains useful forward progress while avoiding the collision (NC 1 in each case).

![](images/048f2198a8424d1970e379354a449ae1f5c229cc114a83ac8af772ccfab8cdd9.jpg)

![](images/1c24fa0a235caabe26cca09d667bcf3d0121defc3f22247e349776b12b182a02.jpg)

![](images/cd627460dade17f498c2258bb91dbab3190b29d9b2a13a08f926db9fad5ba5a4.jpg)

![](images/e64e16be6e4998f0a2defe41f5e52d87f445d8808b5d864ffae3b7ef19faf928.jpg)

![](images/36aab623c198bf13ca656d939907ca69b5ffbf1aa63bf2db00a55381f4f331fa.jpg)

(a) Straight urban cruising: SFT stops short (EP 0.77); Walk reaches full progress (EP 1.00, PDMS 1.00) while staying centered in lane.  
![](images/8754b30a75eee46a3306f5bae8b1c99e78becafc65cc83946eeb2b6c7a31dc3c.jpg)

![](images/edc9ed812c1109edd16f1f8bdb80744987babf9fe798ac5259147ed173186878.jpg)

![](images/3e1dd651506e6eab977db2e2ed548437277219e967981fc674c3a09d5cce36ef.jpg)

![](images/87256080b04fd4a222198616a1cdef6c43d663d2453e47075d0d908b514e87ff.jpg)

![](images/e842aa9251dc15250c7196e176974b3cff32048504316907a2bd638571a0a4aa.jpg)  
(b) Open road: the Run policy overshoots off-road (PDMS 0); Walk restores a safe path that still advances far (EP 0.60 → 0.88, PDMS 0.835 → 0.950).

![](images/b1c4ad5adea941c68ad72f49996c00b1f87aae01a2eb53a47854cd76237375bc.jpg)

![](images/762765a2b3277b4a1a6afa12ebffd2357e6e7929134c1e710684595dc248b3ce.jpg)

![](images/3ec485b5342d1e847c11b18464baf04b9c4b5892d55db04b7037f126c5e0a66d.jpg)

![](images/cf6640d53d600f57ba3518804a0c9d997c94eb108ab28b9691733a3c45b6aedf.jpg)  
(c) Intersection turn: Walk tracks the human arc more tightly than SFT while extending progress (EP 0.84→0.97, PDMS 0.934→ 0.989).  
Figure 7: Stage-by-stage qualitative results (part 1/2). Human/SFT/Run/Walk trajectories on three NAVSIM scenes. SFT is safe but short, Run is far but unsafe (NC = 0), and Walk is both far and safe.

![](images/0e1ccca4fe66203678efab692637e3d4313497536fce5fc8843f0e06789afe32.jpg)

![](images/c4e6317a73def2b70e55442ba15897f41d1476f82b29533603b0f127d0f348fe.jpg)

![](images/d56777ca7b46434ca9ed374264087fff17c6ff7e8c88fcde8ea5d350e2d769e8.jpg)  
(a) Crosswalk approach: Walk yields safely yet still clears the junction (EP 0.81 → 1.00, PDMS 0.919→ 1.00).

![](images/e1bee988d5baf3f07f2c02eb4ffdaa38cef7f659106b502ed8f206da55021a1a.jpg)

![](images/7888ffb4ddb19aee9a1431d32da3f0fa97019fbb4e24de339ebe8792fcfa501e.jpg)  
(b) Following a truck: SFT over-brakes behind the lead vehicle; Walk keeps a safe gap while progressing fully (EP 0.79 → 1.00, PDMS 0.913→ 1.00).

![](images/ad7f53ad91caa47b01490acb63ee276db94098a8326ccae23430f7ef088f8699.jpg)  
(c) Bus interaction: Walk advances past the SFT stopping point without violating safety (EP 0.84→ 0.97, PDMS 0.933→ 0.985).  
Figure 8: Stage-by-stage qualitative results (part 2/2). Three additional scenes. In every case the Walk-GRPO policy (orange) inherits the long-range progress discovered by Run-GRPO (red) but removes its collisions/off-road events, ending close to or above the human progress level while remaining collision-free.

![](images/1e80d33596542335292f2470c2a37a785340bab5d22d6e52934742ba17eefa58.jpg)

![](images/eabf6cdcd9120936ae4c5450a0ecc26fd2e8e7a52f5378b831b5e932be842f42.jpg)

![](images/4ab3c91bbc0b4fe579b2a02f01a104d5a687d397267476b24fdedf554f5aaaf9.jpg)

![](images/aac595d6d3040023591d4a2b866b90750b2f2848b250f59696c980e3bd3318b3.jpg)

![](images/5e5ca710ae4adb55c0929a31a78d0b6bcc6d10bcbf34edddd5b77f0a98a83169.jpg)

![](images/4f97a3efd06f7e202e6d0c94759d548fe74795e11bb718430208384c68215d01.jpg)  
Figure 9: Qualitative comparison with Walk-then-Run (AutoDrive-P³) part 1/2. Across three scenes, the orange safety-first baseline stops short despite remaining safe. Run-then-Walk (blue) travels farther safely and raises both ego progress and PDMS, exposing the driving-efficiency cost of excessive conservatism.

![](images/8b827138f7b284144120e93adde42b8677215a6e5a49f6bf9747a9483b95d63b.jpg)

![](images/6a55894872d60945bbba770039127708e77440b5d6847b773ec2be26f6c06d06.jpg)

![](images/a99d04740afbb5da67b450a3b0220f2b0ffd3f7e88377bc19a5c3fd3678e8757.jpg)

![](images/9ad3e41995e4f78af130e574fa52023ba29c1cff66af22ad8027d8e51e43fe69.jpg)

![](images/0c052042b50432255acd26d0c8c18799722a08756719ba4171f633d750eb6620.jpg)

![](images/e74e5f57d1738aea3b851d21c7c57b7e6dc4b83966704624a6aa782219cb8309.jpg)  
Figure 10: Qualitative comparison with Walk-then-Run (AutoDrive-P3) part 2/2. Three additional scenes confirm that the safety-first baseline remains safe but sacrifices endpoint progress. Run-then-Walk reaches farther safe endpoints and improves driving efficiency.

![](images/ae806a9500d69787f508614386aa7f6460913fbadfda7612c740a7c376598ffd.jpg)

![](images/0d5183400f43566d576d3c63d893b8086a12328003068495cfdc50db883e03b9.jpg)

![](images/9478efd49eaf14ce43bb6ba0859afe9d3097e657a373c443f2998a01f18915af.jpg)

![](images/a204bcd25d5aacc50ef40a815c6f3140c607ecd8874b2454e2b22222d3cb553f.jpg)

![](images/738b8b6052c51604a9af36ea7e4d11f82194efdd83e669cf0a3f4e86eacdedf5.jpg)

![](images/6faef30fbadb37f8d538352fcceeab61ed918aa08624c651dec3ab9ae608aa1b.jpg)  
Figure 11: Qualitative comparison with Walk-and-Run (ReCogDrive), part 1/2. Three scenes show the cost of overemphasizing progress: the orange baseline advances aggressively but collides in each case. The blue Run-then-Walk trajectories continue forward without collision, avoiding the baseline's safety failure.

![](images/3d8508b39efa96021ec67db3d73cc81247e192a7b0d15509f95969ee2e82980d.jpg)

![](images/e16e259a755510ff7eafef1ce45cc69b57dae3d526db73931205f063ee90df1c.jpg)

![](images/bf3ce072a24dcd35a53b27708ee961e69a15b247eae11121923faf8d79710556.jpg)

![](images/10dbe3d3fd429228eabfb4942fab2dd30535b700e6d845adb1a08ccc307c2407.jpg)

![](images/118db73aa56e4a0e9339ebab3bdfdfc79e2ab598b47d0de1385e260dab92f2f1.jpg)

![](images/65f226199dc4ad5d769ecf6abe5c50828ae750b03c75d8c86bb245d3d9ce81c1.jpg)  
Figure 12: Qualitative comparison with Walk-and-Run (ReCogDrive), part 2/2. Three additional scenes show the same failure mode: progress-dominant training produces unsafe trajectories, while Run-then-Walk preserves forward motion with NC 1.