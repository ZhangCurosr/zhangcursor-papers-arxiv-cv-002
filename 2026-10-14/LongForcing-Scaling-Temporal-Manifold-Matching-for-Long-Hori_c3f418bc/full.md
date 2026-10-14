# LongForcing: Scaling Temporal Manifold Matching for Long-Horizon World Rollout

Ziyu Zhu<sup>1,</sup> <sup>∗</sup> Zhaoxu Sun<sup>1,</sup> <sup>∗,</sup> <sup>†</sup> Fan Jiang<sup>1,</sup> <sup>∗,</sup> <sup>‡</sup> Mengchao Wang<sup>1,</sup> <sup>∗</sup> Junfeng Ni<sup>1,</sup> <sup>2,</sup> <sup>§</sup> Ting-Bing Xu<sup>1</sup> Zhicheng Liu<sup>1</sup> Mu Xu<sup>1</sup>

<sup>1</sup>AMAP, Alibaba Group <sup>2</sup>Tsinghua University

## Abstract

Autoregressive video diffusion models have emerged as a foundation for long-horizon world rollout. However, when extended far beyond their training windows, these models suffer from progressive drift: visual quality deteriorates, motion becomes inconsistent, and object identity gradually collapses as generated frames are recursively reused as context. We study a supervision-horizon gap: short-horizon distillation leaves later rollout states without direct supervision. We show that matching short-window marginals need not determine long-horizon behavior, while longer on-policy trajectories expose delayed failures to training. This motivates temporal manifold matching at scale for long-horizon world models. Building on this insight, we introduce LongForcing, a distillation approach that scales teacher supervision to long student self-rollouts. LongForcing preserves the short-horizon teacher-forcing and causal ODE-distillation stages, while replacing the realscore teacher in the final distribution-matching stage with an extended-horizon bidirectional teacher. The long-horizon teacher supervises extended student self-rollouts using the standard distribution-matching formulation, without modifying the student architecture or increasing inference cost. We evaluate LongForcing on text-to-video generation and interactive action-conditioned world models, achieving a VBench Quality score of 85.52 and improving WorldRoamBench 60s action score by 23.9 points in first-person ablations. Qualitative stress tests further sample interactive rollouts up to 24 hours, far beyond the training horizon. These results suggest that scaling supervision along the temporal axis is a practical step toward long-horizon world models. The project website and code are at https://github.com/Fantasy-AMAP/LongForcing.

## 1 Introduction

“We can only see a short distance ahead, but we can see plenty there that needs to be done.” — ALAN TURING

Autoregressive (AR) video diffusion models are becoming a foundational paradigm for long-horizon world rollout. An AR model generates video chunk by chunk, with each new chunk conditioned on previously generated frames. Repeating this process enables rollouts far longer than individual training clips. This capability is essential for both interactive world models [1–6], where a world must remain coherent and responsive as the user acts, and long-form text-to-video systems [7, 8], where visual quality, motion, and identity must persist far beyond the initial clip. However, the same recursive mechanism that enables long rollouts also makes AR models vulnerable to rollout drift. Because every generated chunk becomes context for the next, small errors compound, the conditioning distribution moves away from the data seen during training, and the model gradually loses visual fidelity.

![](images/2ab7d104d164b694e400d444d529a9241a23b83cf1a2b1433b13eefebaf527ee.jpg)  
Figure 1: LongForcing extends teacher guidance over student rollouts. Qualitative snapshots from the 5B interactive model span stress-test rollouts sampled up to 24 hours.

Modern real-time AR video generators are often obtained through a three-stage distillation pipeline. Teacher forcing first adapts a bidirectional video model to causal conditioning, ODE distillation compresses its denoising trajectory into a few-step generator, and distribution matching distillation (DMD) aligns the student’s self-generated distribution with a teacher [8–10]. This recipe is highly effective within the few-second horizon on which it is trained, but deployment requires closed-loop generation over minutes. Teacher forcing learns from short clean histories, causal ODE distillation uses short teachergenerated trajectories, and short-horizon DMD evaluates only short student rollouts; later rollout states receive no direct supervision from this objective.

Recent approaches mitigate long-horizon drift through sparse local matching [11], inference-time memory and positional modifications [12–14], or context perturbation [15]. Local matching can supervise late rollout states, but its jointly scored windows remain short; inference-time modifications instead stabilize the conditioning context. Neither mechanism by itself matches the longer joint path, leaving cross-window dependencies that can affect visual fidelity, identity, dynamics, and action consistency.

These failures reveal a supervision-horizon gap between short-window training and long-horizon deployment. Our principle is that temporal manifold matching should scale with the trajectories visited on policy, where temporal manifold informally describes high-probability regions of the teacher’s path distribution. Section 4 formalizes this view using path distributions, KL divergence, and total variation. Matching short windows does not determine the distribution of a long rollout, whereas matching clean trajectories controls drift within the supervised horizon. Beyond this horizon, stability also depends on teacher leakage, student–teacher mismatch along visited states, and recovery from drifted states. This motivates extending distribution matching over longer student rollouts while keeping teacher forcing and ODE distillation short.

Guided by this principle, we introduce LongForcing, which replaces Causal Forcing’s DMD real-score teacher with an extended-horizon teacher for longer self-generated trajectories. The student rolls out on policy, while an extended-horizon teacher supplies distribution-level supervision across a long, contiguous portion of each self-generated training trajectory, requiring no architectural changes. Empirically, LongForcing improves text-to-video VBench Quality by 0.93 points over Causal Forcing and raises WorldRoamBench 60s action score by 23.9 points (37% relative) in first-person ablations. Qualitative stress tests sample interactive rollouts up to 24 hours. These results position temporal manifold matching at scale as a practical step toward long-horizon world models.

Our contributions are threefold:

• We formalize the supervision-horizon gap: longer student rollouts expose delayed failures, and joint path matching constrains dependencies left undetermined by local-window marginals.

• We introduce LongForcing, which extends the teacher’s supervision horizon in distribution-matching distillation to guide longer student-generated trajectories.

• We evaluate LongForcing across interactive world models and text-to-video generation, measuring improvements in action following and video quality alongside their dependence on supervision length.

## 2 Related Work

## 2.1 Autoregressive Long-Video Generation

Autoregressive video generation has evolved from discrete-token transformers [16–19] to diffusion and flow models that stream through temporally varying noise levels, causal attention, or sliding-window denoising [20–24]. Few-step variants further use video DMD, student-generated contexts, progressivewindow denoising, corrected causal ODE initialization, or diagonal distillation [8–10, 25, 26]. To extend generation beyond the training horizon, training-free methods modify noise schedules, denoising windows, attention, positional encodings, or temporal memory [7, 13, 14, 27–29], whereas training based approaches use self-generated segments, long-context tuning, hierarchical denoising, and drift simulation [11, 30–33]. Context Forcing [31] also uses a long-context teacher, combining continuationconditioned DMD with Slow-Fast Memory. LongForcing retains Causal Forcing’s student architecture and inference procedure, replacing its DMD teacher to score a longer contiguous, jointly noised student path.

## 2.2 Interactive World Models

Interactive world models use video generation as a controllable simulator for prediction, exploration, and closed-loop interaction. Recent systems demonstrate playable game simulation [3, 4, 34], camera and action-controlled world exploration [5, 6, 35], and real-time interactive generation [2, 36, 37]. Unlike prompt-driven video generation, interactive rollout must preserve both world state and the consequences of continuously changing user actions. Over extended sessions, accumulated errors can therefore degrade not only visual fidelity and identity, but also action responsiveness and physical consistency. WorldRoam-Bench [38] explicitly evaluates these long-horizon properties across diverse interactive environments. LongForcing scales teacher supervision of on-policy rollouts along the temporal axis in both interactive and text-conditioned generation. Qualitative stress tests sample interactive rollouts up to 24 hours, complementing the benchmark evaluation.

## 3 Drift in Autoregressive Video Rollouts

We first review autoregressive video diffusion and its standard distillation pipeline, then connect the empirical manifestations of long-video drift to the supervision coverage of closed-loop rollout.

## 3.1 Autoregressive Video Diffusion and Distillation

Video diffusion. Following standard practice, video generation takes place in the latent space of a pretrained spatiotemporal VAE. Let z denote a clean video latent. At noise level $s \in [ 0 , 1 ]$ , the forward process constructs [10, 39] ${ \bf z } ^ { ( s ) } = \alpha _ { s } { \bf z } + \sigma _ { s } \epsilon$ , where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and the superscript (s) denotes the diffusion noise level. A score or velocity network learns the reverse dynamics, whose probability-flow ODE maps noise to a clean latent.

(a) Visible drift emerges beyond the supervised horizon  
![](images/0c84e1fed107e9c5ebcb72f373bfa12273a0675744ae784cd1aeaffdab9f6b16.jpg)

(b) Drift grows beyond short supervision  
![](images/ec59f127fd1673279a62ed9a3312e1740f365a1cab9677182b10e9a760845967.jpg)

(c) Drift follows a shared channel direction  
![](images/12e465afc136f01a3f2fcce977406e6e041d78633e96bf8518df7180ca1720a9.jpg)  
Figure 2: Visual and feature-space signatures of rollout drift. (a) Shared timestamps reveal late over-saturation and high-frequency artifacts. (b) Median absolute channel-mean shift from the early reference; dotted lines mark supervision horizons. (c) Median signed per-channel shift for the 81-frame model; persistent bands indicate a channel-wise direction shared across 20 videos.

Autoregressive rollout. Partition a latent video into K causal chunks $\mathbf { z } _ { 1 : K }$ , where each chunk is generated from the preceding chunks and condition c. The resulting path distribution factorizes as $p _ { \theta } ( \mathbf { z } _ { 1 : K }$ $\begin{array} { r } { { \mathbf { c } } ) = \prod _ { k = 1 } ^ { K } p _ { \theta } ( { \mathbf { z } } _ { k } \mid { \mathbf { z } } _ { < k } , { \mathbf { c } } ) } \end{array}$ , where c denotes a text prompt or a prompt augmented with per-chunk actions. At inference, each prediction is appended to the history, enabling streaming while accumulating error.

Few-step distillation and distribution matching. An efficient causal student is typically obtained through teacher forcing on short clean histories, causal ODE distillation into a few-step generator, and DMD refinement [10, 40, 41]. DMD compares bidirectional real and fake scores on student-generated closed-loop paths; Self Forcing [8] applies this refinement on policy, but only within a short rollout window.

## 3.2 Why Autoregressive Rollouts Drift

Long-video drift can appear as position or color shift, amplified blur and noise, and action inconsistency [33]. These mechanisms share a temporal structure: generated chunks recursively change the conditioning distribution. The relevant train–test mismatch is therefore between the short prefixes evaluated during training and the evolving on-policy path traversed during long deployment.

Fix a prefix-measurable drift detector $g _ { t } ( \mathbf { z } _ { 1 : t } ) \in [ 0 , 1 ]$ and threshold $\gamma _ { : }$ and define the first drift time and within-horizon event

$$
\tau _ { \mathrm { d r i f t } } = \operatorname* { i n f } \{ t \geq 1 : g _ { t } > \gamma \} , \qquad \mathcal { B } _ { H } ( \gamma ) = \{ \tau _ { \mathrm { d r i f t } } \leq H \} .\tag{1}
$$

For a fixed rollout distribution $Q$ and $H _ { 1 } < H _ { 2 }$ , lengthening the path exposes exactly the delayed failure mass

$$
\operatorname* { P r } _ { Q } ( \tau _ { \mathrm { d r i f t } } \le H _ { 2 } ) - \operatorname* { P r } _ { Q } ( \tau _ { \mathrm { d r i f t } } \le H _ { 1 } ) = \operatorname* { P r } _ { Q } ( H _ { 1 } < \tau _ { \mathrm { d r i f t } } \le H _ { 2 } ) .\tag{2}
$$

Thus an objective computed only from $H _ { 1 }$ -step prefixes cannot evaluate failures whose first hitting time

exceeds $H _ { 1 }$ . A longer rollout changes which delayed states can receive a teacher signal; whether optimization corrects them is addressed in Section 4.

Figure 2 reveals structured late-rollout shift in the shared causal-VAE latent space: the channel-mean deviation grows beyond the short supervision horizon, while the signed heat map shows a consistent channel-wise direction. Appendix C.2 gives the exact reference window and aggregation procedure.

## 4 Finite-Horizon Drift Control

We separate three questions behind LongForcing: which failures can be detected within a finite training window, what matching trajectories within that window guarantees, and under what conditions those guarantees extend to later blocks. Throughout, H denotes a horizon of H autoregressive chunks; experiments report the corresponding number of video frames. Appendix B gives the complete setup, proofs, and scope conditions.

## 4.1 The Supervision-Horizon Gap

Let $S _ { t }$ denote the student’s causal state after chunk t, and let $\mathcal { C } _ { H } = ( S _ { 0 } , \mathbf { a } _ { 1 : H } , \mathbf { c } )$ collect the initial state, an action sequence fixed over the rollout, and persistent conditions such as the text prompt. For this shared condition, $Q _ { \theta , H }$ denotes the causal student’s H-chunk path distribution and $Q _ { \phi , H } ^ { * , \mathrm { l o n g } }$ the extended-horizon teacher’s joint reference. The distinction between matching local windows and matching this joint path is fundamental.

Proposition 4.1 (Short windows do not identify the long path). For every $W < T$ , two causal path distributions can agree on every contiguous W-step marginal while having disjoint T-step supports. Hence even perfect local-window matching need not determine the long trajectory distribution.

Proof in Appendix B.2. Proposition 4.1 is an identifiability statement: local-window marginals leave cross-window coupling undetermined. Equation (2) gives a separate exposure statement. For a fixed rollout law, an H-chunk prefix objective cannot evaluate failures first reached after $H ;$ extending that rollout exposes exactly the delayed-failure mass in the added suffix. This exposure makes those states available for supervision, but does not imply monotonic improvement across models.

## 4.2 What Path Matching Guarantees

For a shared condition distribution $\mu _ { H }$ , let $\overline { { Q } } _ { \theta , H } = \mu _ { H } Q _ { \theta , H }$ and $\overline { { Q } } _ { \phi , H } ^ { * } = \mu _ { H } Q _ { \phi , H } ^ { * , \mathrm { l o n g } }$ . The ideal clean path-matching objective is

$$
\begin{array} { r } { \mathcal { L } _ { H } ^ { \mathrm { i d e a l } } = \mathbb { E } _ { \mathcal { C } _ { H } \sim \mu _ { H } } D _ { \mathrm { K L } } \Big ( Q _ { \theta , H } ( \cdot \mid \mathcal { C } _ { H } ) \Big \| Q _ { \phi , H } ^ { * , \mathrm { l o n g } } ( \cdot \mid \mathcal { C } _ { H } ) \Big ) . } \end{array}\tag{3}
$$

Theorem 4.2 (Within-horizon drift control). For every measurable path event $\boldsymbol { \mathcal { A } } _ { H }$

$$
\operatorname* { P r } _ { Q _ { \theta , H } } \left( \mathcal { A } _ { H } \right) \leq \operatorname* { P r } _ { Q _ { \phi , H } ^ { * } } \left( \mathcal { A } _ { H } \right) + \sqrt { \mathcal { L } _ { H } ^ { \mathrm { i d e a l } } / 2 } .\tag{4}
$$

Proofin Appendix B.3. Taking $\mathcal { A } _ { H } = \mathcal { B } _ { H } ( \gamma )$ bounds student drift by teacher drift plus the path-matching slack. Increasing H therefore enlarges the delayed events covered by the guarantee, but does not automatically shrink the slack or control failures after H.

## 4.3 Teacher-Guided Correction over Finite Depth

For a fixed evaluation stride B, let D be a measurable degraded region and ${ \mathcal { G } } = { \mathcal { D } } ^ { c } ;$ when the drift detector is state-measurable, D can be its thresholded region from Eq. (1). Define $b _ { j } = \operatorname* { P r } ( S _ { j B } \in \mathcal { D } )$ , $a _ { j } = \operatorname* { P r } ( S _ { ( j + 1 ) B } \in \mathcal { D } \mid S _ { j B } \in \mathcal { G } )$ , and $\eta _ { j } = \mathrm { P r } ( S _ { ( j + 1 ) B } \in \mathcal { G } \mid S _ { j B } \in \mathcal { D } )$ . Total probability gives

$$
b _ { j + 1 } = ( 1 - b _ { j } ) a _ { j } + ( 1 - \eta _ { j } ) b _ { j } .\tag{5}
$$

Let $\lambda _ { j } ^ { * }$ be teacher leakage on the student’s on-policy good-start distribution, $\eta _ { j } ^ { * }$ the corresponding teacher return from degraded starts, and $\epsilon _ { j } ^ { \mathcal { G } } , \epsilon _ { j } ^ { \mathcal { D } }$ the clean B-step student–teacher path KLs on those starts. Conditional Pinsker yields

$$
a _ { j } \le \lambda _ { j } ^ { * } + \sqrt { \epsilon _ { j } ^ { \mathcal { G } } / 2 } , \qquad \eta _ { j } \ge \left[ \eta _ { j } ^ { * } - \sqrt { \epsilon _ { j } ^ { D } / 2 } \right] _ { + } .\tag{6}
$$

Proposition 4.3 (Conditional finite-depth drift control). Suppose for $0 \le j < J$ that $\lambda _ { j } ^ { * } \le \bar { \lambda } , \eta _ { j } ^ { * } \ge \bar { \eta } ^ { * }$ $\epsilon _ { j } ^ { \mathcal { G } } \leq \bar { \epsilon } _ { \mathcal { G } }$ , and $\epsilon _ { j } ^ { \mathcal { D } } \leq \bar { \epsilon } _ { \mathcal { D } }$ . Define $A = \operatorname* { m i n } \{ 1 , \bar { \lambda } + \sqrt { \bar { \epsilon } _ { \mathcal { G } } / 2 } \}$ and $\underline { { \eta } } = [ \bar { \eta } ^ { * } - \sqrt { \bar { \epsilon } _ { \mathcal { D } } / 2 } ] _ { + } > 0$ . Then

$$
b _ { j } \leq \operatorname* { m i n } \Biggl \{ 1 , ( 1 - \underline { { \eta } } ) ^ { j } b _ { 0 } + \frac { A } { \underline { { \eta } } } \left[ 1 - ( 1 - \underline { { \eta } } ) ^ { j } \right] \Biggr \} , \quad 0 \leq j \leq J .\tag{7}
$$

Proof in Appendix B.4. Here A controls newly injected degradation, while $\eta$ lower-bounds student recovery using the teacher’s return probability and the degraded-start path mismatch. Equation (7) controls degraded-state occupancy at these block boundaries and is nontrivial when $A < \underline { { \eta } } .$ . These are sufficient conditions, not guarantees supplied automatically by LongForcing; feedback actions and a stronger conditional long-run extension are treated in Appendix B.4–B.5.

## 5 LongForcing

LongForcing builds on Causal Forcing [10]. We retain its short-horizon teacher-forcing and causal ODEdistillation stages, and replace the real-score teacher in the final asymmetric DMD stage with an extendedhorizon bidirectional teacher. This teacher supervises longer student self-rollouts using the standard DMD formulation, motivated by the analysis in Section 4.

## 5.1 Long-Horizon Trajectory Matching

Objective. With the long-horizon teacher as the reference, we apply the standard noisy DMD objective to an H-chunk student path, extending the short supervision horizon W:

$$
\begin{array} { r } { \mathcal { I } _ { H } ^ { \mathrm { n o i s y } } = \mathbb { E } _ { \mathcal { C } _ { H } \sim \mu _ { H } , s } \left[ D _ { \mathrm { K L } } \Big ( Q _ { \theta , H } ^ { ( s ) } ( \cdot \mid \mathcal { C } _ { H } ) \Big \| Q _ { \phi , H } ^ { * , ( s ) } ( \cdot \mid \mathcal { C } _ { H } ) \Big ) \right] , } \end{array}\tag{8}
$$

where $Q _ { \theta , H } ^ { ( s ) }$ and $Q _ { \phi , H } ^ { * , ( s ) }$ are the student and teacher path distributions at shared noise level s. As in asymmetric DMD, the student generates causally and both score networks evaluate its noised path bidirectionally. The longer scoring horizon covers later prefixes; Appendix A.2–A.4 gives the objective’s scope and score estimator.

Extended-horizon teacher. The frozen real-score model is trained or fine-tuned on long sequences so that it can score the complete H-step path; the online fake-score model uses the same long bidirectional context to track the current student distribution. For interactive training, a pre-specified action plan is supplied over the same horizon. The extended horizon increases supervision coverage; the guarantees remain conditional on teacher quality and student–teacher mismatch.

Algorithm. Algorithm 1 summarizes the pipeline. In Stage 3, the long-horizon real-score teacher supervises the longer self-generated path.

Algorithm 1 LongForcing training   
Require: short horizon W; long horizon H; extended teacher $s _ { \mathrm { r e a l } }$   
1: Train a causal model by teacher forcing on W-step real clips.   
2: Use this causal model as the ODE teacher to initialize $G _ { \theta }$ on $W \cdot$ -step clips.   
3: Initialize the online model $s _ { \mathrm { f a k e } }$ from $s _ { \mathrm { r e a l } } .$   
4: for each LongForcing iteration do   
5: Sample $\mathcal { C } _ { H }$ and roll out $\mathbf { z } _ { 1 : H } \sim Q _ { \theta , H } ( \cdot \mid \mathcal { C } _ { H } )$ on policy.   
6: Sample s and noise the full path to obtain $\mathbf { z } _ { 1 : H } ^ { ( s ) } .$   
7: Evaluate $s _ { \mathrm { r e a l } }$ and $s _ { \mathrm { f a k e } }$ bidirectionally on $\mathbf { z } _ { 1 : H } ^ { ( s ) } .$   
8: Update $G _ { \theta }$ using the DMD estimator and $s _ { \mathrm { f a k e } }$ on their respective schedules (Appendix A.1).   
9: end for   
10: return few-step causal student $G _ { \theta }$

## 6 Experiments

## 6.1 Setup

Interactive world model. We train action-conditioned models on Wan2.2-5B [42] and Wan2.1- 1.3B [43] and evaluate 60-second WorldRoamBench rollouts: third-person for the 5B comparison and first-person for the 1.3B horizon ablation. Teacher forcing and ODE distillation use 121 and 81 frames for the 5B and 1.3B models, respectively, while DMD extends both to 321 frames. DMD training uses AdamW on eight GPUs with a global batch size of 8 for 1000 steps. The generator and fake-score network use constant learning rates of $2 \times 1 0 ^ { - 6 }$ and $4 \times 1 0 ^ { - 7 }$ , respectively, with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 , 0 . 9 9 9 )$ and weight decay 0.01.

Text-to-video model. For text-to-video, we follow Causal Forcing [10] on Wan2.1-1.3B [43] at 832×480 resolution. We retain its teacher-forcing and causal ODE configurations, replace the DMD real-score teacher, and use 321-frame DMD rollouts. DMD uses AdamW on 8 GPUs with a global batch size of 8 and learning rates of $2 \times 1 0 ^ { - 6 }$ and $4 \times 1 0 ^ { - 7 }$ for the generator and fake-score network. We evaluate with the official 5-second VBench protocol [44] and VBench-Long [45] on 30-second generations. See Appendix A.1 for further details.

## 6.2 Main Results

Interactive world model. Table 1 reports control-rate-weighted scores for 60-second third-person rollouts. Scores incorporate control rate as defined in the table caption. LongForcing-5B achieves weighted action and visual scores of 76.81 and 57.89, the highest among the open-source models shown. Its control rate is 87.88% (87/99), compared with 40.40% for LingBot-World v2.

The 47.48-percentage-point control-rate advantage over LingBot-World v2 highlights the importance of sustained action responsiveness in the weighted evaluation. The scores therefore reflect controllability and visual quality jointly, rather than visual quality alone. Achieving this result with a 5B model against 8B and 14B baselines also suggests that parameter count alone does not determine long-rollout performance; Section 6.3 examines supervision configurations within a fixed backbone.

Table 1: WorldRoamBench results. Except Ctrl., higher-is-better scores are multiplied by p = Ctrl./100; Aesth. Drift is 100[1 − p(1 − d)] for normalized drift $d \in [ 0 , 1 ]$ . Best and second-best open-source results are bold and underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="4">WASD Action Interaction</td><td colspan="4">Visual Quality</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Action ↑ Ctrl. ↑ Partial Acc. ↑ Traj. ↑ Visual ↑ Aesthetic ↑ Aesth. Drift ↓ Imaging ↑</td><td></td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Genie 3 [2]</td><td></td><td>69.73</td><td>79.80</td><td>65.75</td><td>73.70</td><td>56.26</td><td>48.11</td><td>32.08</td><td>45.30</td></tr><tr><td>HappyOyster [46]</td><td></td><td>84.02</td><td>96.97</td><td>80.48</td><td>87.57</td><td>65.63</td><td>59.10</td><td>20.61</td><td>47.44</td></tr><tr><td>Open-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LingBot-World [47]</td><td>14B</td><td>18.81</td><td>23.23</td><td>18.50</td><td>19.12</td><td>15.99</td><td>14.32</td><td>81.70</td><td>13.84</td></tr><tr><td>HY-World 1.5 [36]</td><td>8B</td><td>24.92</td><td>29.29</td><td>24.57</td><td>25.28</td><td>19.11</td><td>17.52</td><td>79.48</td><td>16.52</td></tr><tr><td>LingBot-World v2 [48]</td><td>14B</td><td>36.09</td><td>40.40</td><td>35.35</td><td>36.84</td><td>28.42</td><td>25.61</td><td>68.75</td><td>25.04</td></tr><tr><td>LongForcing (ours)</td><td>5B</td><td>76.81</td><td>87.88</td><td>73.74</td><td>79.89</td><td>57.89</td><td>51.69</td><td>31.93</td><td>48.26</td></tr></table>

Text-to-video: 5-second VBench. Table 2 reports 5-second VBench results. At matched inference cost, LongForcing leads distilled autoregressive models with 84.56 Total and 85.52 Quality, improving over Causal Forcing by 0.52 and 0.93 points, respectively. These results suggest that longer-horizon supervision also benefits short-clip visual quality, extending its utility beyond long-rollout stability.

Text-to-video: 30-second VBench-Long. Table 3 reports 30-second VBench-Long results, where LongForcing achieves the highest Total and Quality scores among the compared methods, at 82.83 and 83.99, improving over Causal Forcing by 1.78 and 0.55 points, respectively. The full results in Appendix C.1 show the highest background consistency score of 96.58. Semantic alignment also improves from 71.52 to 78.19, complementing the gains in visual quality and background consistency.

Table 2: 5-second VBench.
<table><tr><td>Model</td><td>Total ↑</td><td>Quality ↑</td><td>Semantic ↑</td></tr><tr><td>CausVid [9]</td><td>81.33</td><td>83.98</td><td>70.72</td></tr><tr><td>Self Forcing [8]</td><td>83.74</td><td>84.48</td><td>80.77</td></tr><tr><td>Causal Forcing [10]</td><td>84.04</td><td>84.59</td><td>81.84</td></tr><tr><td>LongForcing (ours)</td><td>84.56</td><td>85.52</td><td>80.72</td></tr></table>

Table 3: 30-second VBench-Long.
<table><tr><td>Model</td><td>Total ↑</td><td>Quality ↑</td><td>Semantic ↑</td></tr><tr><td>LongLive [30]</td><td>82.80</td><td>83.21</td><td>81.17</td></tr><tr><td>Self Forcing [8]</td><td>81.49</td><td>82.67</td><td>76.77</td></tr><tr><td>Causal Forcing [10]</td><td>81.05</td><td>83.44</td><td>71.52</td></tr><tr><td>LongForcing (ours)</td><td>82.83</td><td>83.99</td><td>78.19</td></tr></table>

## 6.3 Ablation Studies

Text-to-video supervision horizon. Table 4 compares supervision-horizon configurations from 81 to 321 frames (approximately 5 to 20 seconds). Each DMD rollout length uses its corresponding separately trained teacher checkpoint. The 321-frame setting achieves the highest 5-second total score and 30-second total, quality, and semantic scores.

Table 4: Supervision-horizon configurations for text-to-video generation. Horizons are rollout lengths in frames. Best and second-best scores are bold and underlined.
<table><tr><td rowspan="2"></td><td rowspan="2">Frames Approx. span</td><td rowspan="2">VBench (5s) Total ↑</td><td colspan="4">VBench-Long (30s) ↑</td></tr><tr><td>Total</td><td>Quality</td><td>Semantic</td><td>Dynamic</td></tr><tr><td>81</td><td>~5s</td><td>84.36</td><td>81.87</td><td>83.71</td><td>74.54</td><td>58.15</td></tr><tr><td>161</td><td>~10s</td><td>83.76</td><td>78.02</td><td>81.59</td><td>63.77</td><td>56.67</td></tr><tr><td>241</td><td>~15s</td><td>84.38</td><td>79.18</td><td>83.75</td><td>60.87</td><td>94.91</td></tr><tr><td>321</td><td>~20s</td><td>84.56</td><td>82.83</td><td>83.99</td><td>78.19</td><td>60.93</td></tr></table>

Interactive world-model supervision horizon. In Table 5, extending the DMD supervision horizon from 81 to 321 frames raises action score from 64.39 to 88.26 and strict accuracy from 42.89 to 85.16 on 60-second rollouts, while reducing aesthetic drift from 56.58 to 42.37. The 321-frame setting achieves the best results across all action metrics and most visual metrics, supporting the benefit of longer supervision in exposing accumulated control and consistency errors.

Table 5: Effect of the DMD supervision horizon on first-person interactive world modeling. The comparison varies the DMD supervision horizon with a fixed 1.3B backbone under the first-person WorldRoamBench 60s protocol; scores are unweighted. Higher is better except for Aesthetic Drift. The best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Frames Approx. span</td><td rowspan="2"></td><td colspan="4">WASD Action Interaction</td><td colspan="4">Visual Quality</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Action ↑ Strict Acc. ↑ Partial Acc. ↑ Traj. ↑ Visual ↑ Aesthetic ↑ Aesth. Drift ↓ Imaging ↑</td><td></td></tr><tr><td>81</td><td>~5s</td><td>64.39</td><td>42.89</td><td>63.01</td><td>87.27</td><td>45.05</td><td>39.81</td><td>56.58</td><td>46.40</td></tr><tr><td>161</td><td>~10s</td><td>56.16</td><td>43.21</td><td>50.00</td><td>75.28</td><td>47.01</td><td>41.06</td><td>60.18</td><td>52.29</td></tr><tr><td>241</td><td>~15s</td><td>83.39</td><td>78.24</td><td>85.14</td><td>86.78</td><td>55.19</td><td>49.93</td><td>43.30</td><td>50.25</td></tr><tr><td>321</td><td>~20s</td><td>88.26</td><td>85.16</td><td>90.56</td><td>89.05</td><td>56.30</td><td>51.61</td><td>42.37</td><td>50.84</td></tr></table>

Time-resolved rollout drift. Figure 3 shows how text-to-video generation deteriorates as rollouts grow longer. Extending supervision from 81 to 321 frames reduces the mean drift score at 59 seconds from 67.46 to 31.91, a relative reduction of 52.7%. The gap between the curves grows over time, indicating that longer supervision slows error accumulation and helps maintain quality during extended generation, although some drift remains. Appendix C.3 provides the metric definition, evaluation protocol, and corresponding interactive results.

![](images/9db4df3ebb44fbf4abb3ddd7fa5d59f29f5c625b3b04db057a7bff8789f9af1b.jpg)

![](images/c5542724adbb4d7e3ef8abf378b9d5bceedd0920a7c9ee76d990bd767bf532ae.jpg)  
Figure 3: Rollout drift and human preference across supervision horizons. (a) Blinded pairwise preferences. (b) Mean Drift Score.

Human study. Across 400 judgments per criterion from 20 evaluators on 20 shared cases, 321-frame LongForcing receives 72.0% overall preference, compared with 16.2% for the 81-frame model and 11.8% ties (Figure 3). Preferences also favor longer supervision in visual quality by 71.2% to 13.8%, temporal coherence by 67.0% to 18.2%, and motion naturalness by 64.8% to 17.8%. Appendix C.4 gives the evaluation protocol.

## 6.4 Qualitative Results

Figure 4A–D shows recognizable scenes and characters across snapshots spanning up to 54 minutes. In the 600-second comparisons (E–F), LingBot-World v2 (14B) [48] exhibits over-saturation and repetitive

textures (E), and scene and character appearance shifts (F), while LongForcing-5B retains more of the initial scene structure and visual style. These comparisons assess visual consistency at the displayed timestamps.

![](images/7faec8ff91c9e0fe3dd208277501758f6dcaba71b880b2a830e660d2a6adf505.jpg)  
Figure 4: A–D: LongForcing rollouts over up to 54 minutes. E–F: LongForcing-5B (top) and LingBot-World v2 (14B; bottom) at matched timestamps from 0 to 599 seconds.

## 7 Conclusion

We identify a supervision-horizon gap in autoregressive video generation: short-horizon supervision leaves later rollout behavior underdetermined, while longer on-policy trajectories expose delayed failures to teacher guidance. LongForcing addresses this gap by replacing Causal Forcing’s DMD real-score teacher with an extended-horizon teacher that supervises longer student self-rollouts, while retaining its short teacher-forcing and causal ODE-distillation stages. Across text-to-video generation and interactive world models, this improves VBench Quality by 0.93 points and first-person action score by 23.9 points over the respective baselines, without changing the architecture or increasing inference cost. Horizon ablations and time-resolved evaluations show reduced rollout drift, complemented by qualitative interactive stress tests sampled up to 24 hours. These findings position temporal supervision as a scaling dimension for video world models. Integrating coherent visual rollouts with persistent memory and hierarchical planning offers a path toward persistent interactive worlds, goal-directed behavior, and long-term reasoning.

## References

[1] Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 4603–4623. PMLR, 2024.

[2] Google DeepMind. Genie 3. Google DeepMind, 2025.

[3] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations, 2025.

[4] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. Decart and Etched, 2024.

[5] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-Game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.

[6] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. In International Conference on Machine Learning, 2026.

[7] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. FIFO-Diffusion: Generating infinite videos from text without training. In Advances in Neural Information Processing Systems, volume 37, 2024.

[8] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self Forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems, volume 38, 2025.

[9] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

[10] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal Forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. In International Conference on Machine Learning, 2026. arXiv:2602.02214.

[11] Jiaxing Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-Forcing++: Towards minute-scale high-quality video generation. In International Conference on Learning Representations, 2026.

[12] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling Sink: Bridging limited-horizon training and open-ended testing in autoregressive video diffusion. arXiv preprint arXiv:2602.07775, 2026.

[13] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep Forcing: Training-free long video generation with deep sink and participative compression. In International Conference on Machine Learning, 2026.

[14] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. LoL: Longer than longer, scaling video generation to hour. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 38132–38142, 2026.

[15] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jiwen Lu, and Jie Zhou. Astra: General interactive world model with autoregressive denoising. In International Conference on Learning Representations, 2026.

[16] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. GODIVA: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

[17] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. CogVideo: Large-scale pretraining for text-to-video generation via transformers. In International Conference on Learning Representations, 2023.

[18] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. VideoPoet: A large language model for zero-shot video generation. In International Conference on Machine Learning, 2024.

[19] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. VideoGPT: Video generation using VQ-VAE and transformers. arXiv preprint arXiv:2104.10157, 2021.

[20] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion Forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, volume 37, 2024.

[21] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 6376–6386, 2025.

[22] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, Siyu Zhou, Qian He, and Jing Liu. AR-Diffusion: Asynchronous video generation with auto-regressive diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7364–7373, 2025.

[23] Sand.ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, et al. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

[24] Akio Kodaira, Tingbo Hou, Ji Hou, Markos Georgopoulos, Felix Juefei-Xu, Masayoshi Tomizuka, and Yue Zhao. StreamDiT: Real-time streaming text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29200–29210, 2026.

[25] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling Forcing: Autoregressive long video diffusion in real time. In International Conference on Learning Representations, 2026.

[26] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. Streaming autoregressive video generation via diagonal distillation. In International Conference on Learning Representations, 2026.

[27] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. FreeNoise: Tuning-free longer video diffusion via noise rescheduling. In International Conference on Learning Representations, 2024.

[28] Zengqun Zhao, Yanzuo Lu, Ziquan Liu, Jifei Song, Jiankang Deng, and Ioannis Patras. Relax Forcing: Relaxed KV-memory for consistent long video generation. In British Machine Vision Conference, 2026. Accepted for publication.

[29] Youngrae Kim, Qixin Hu, C.-C. Jay Kuo, and Peter A. Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens. In European Conference on Computer Vision, 2026.

[30] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. LongLive: Real-time interactive long video generation. In International Conference on Learning Representations, 2026.

[31] Shuo Chen, Cong Wei, Sun Sun, Tiancheng Shen, Ping Nie, Kai Zou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context Forcing: Consistent autoregressive video generation with long context. In International Conference on Machine Learning, 2026.

[32] Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, and Nenghai Yu. HiAR: Efficient autoregressive long video generation via hierarchical denoising. In European Conference on Computer Vision, volume 17008 of Lecture Notes in Computer Science, pages 225–242. Springer, 2026.

[33] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.

[34] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in Atari. In Advances in Neural Information Processing Systems, volume 37, pages 58757–58791, 2024.

[35] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-GameCraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.

[36] Tencent Hunyuan. HY-World 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. Technical report, Tencent Hunyuan, 2025.

[37] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. MotionStream: Real-time video generation with interactive motion controls. In International Conference on Learning Representations, 2026.

[38] Ting-Bing Xu, Jiacheng Sui, Zhe Gao, Kewei Shi, Wenjin Yang, Zhicheng Liu, Zhaoxu Sun, Mingchao Sun, Hongyu Pan, Fan Jiang, et al. WorldRoamBench: An open-world benchmark for long-horizon stability of interactive world models. arXiv preprint arXiv:2606.31672, 2026.

[39] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[40] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.

[41] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, volume 37, pages 47455–47487, 2024.

[42] Team Wan. Wan2.2: Open and advanced large-scale video generative models. GitHub repository, 2025.

[43] Team Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[44] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

[45] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[46] Alibaba Cloud. Alibaba launches HappyOyster, a world model product for real-time immersive creation and interaction. Alibaba Cloud Blog, 2026.

[47] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[48] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026.

[49] Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, et al. ABot-World-0: Infinite interactive world rollout on a single desktop GPU. arXiv preprint arXiv:2607.19191, 2026.

[50] Zhucun Xue, Jiangning Zhang, Teng Hu, Haoyang He, Yinan Chen, Yuxuan Cai, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, et al. UltraVideo: High-quality UHD video dataset with comprehensive captions. In Advances in Neural Information Processing Systems, volume 38, 2025.

[51] Gareth O. Roberts and Jeffrey S. Rosenthal. General state space Markov chains and MCMC algorithms. Probability Surveys, 1:20–71, 2004.

[52] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. LTX-Video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[53] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In International Conference on Learning Representations, 2025.

[54] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, et al. Pyramidal flow matching for efficient video generative modeling. In International Conference on Learning Representations, 2025.

[55] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. SkyReels-V2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

[56] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. HPSv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15086–15095, 2025.

## A Implementation Details

## A.1 Training Pipeline

LongForcing retains the teacher-forcing and causal ODE-distillation stages of Causal Forcing and replaces the real-score teacher in its final DMD stage with an extended-horizon bidirectional teacher. Teacher forcing uses short clean histories and causal ODE distillation uses short trajectories from the causal teacher; the resulting student generates a contiguous H-chunk closed-loop path, scored jointly by the replacement teacher and the online fake-score network. The student architecture, sampler, and inference procedure remain unchanged. Under the causal VAE mapping, 81, 161, 241, and 321 decoded RGB frames correspond to 21, 41, 61, and 81 temporal latent positions, respectively; these positions are rounded and grouped into autoregressive chunks of three latent frames.

Interactive teacher: initialization and data. The interactive teachers use the pretrained Wan2.2-5B and Wan2.1-1.3B backbones, adapted to action-conditioned video generation on the ABot-World training data [49]. We train the interactive teachers for 20,000 steps with a learning rate of $1 0 ^ { - 5 }$ and a global batch size of 256 video clips. We use action-annotated video sequences, with the action sequence aligned to the video over the teacher’s temporal window. The teacher retains bidirectional temporal attention so that it can evaluate the complete student-generated path with its corresponding actions. The data collection and annotation pipeline is described in ABot-World [49]; the long teacher provides the frozen real-score network for our 321-frame DMD stage.

Text-to-video teacher: long-video adaptation. We initialize the bidirectional teacher from the pretrained Wan2.1-T2V-1.3B checkpoint [43] and fine-tune it on UltraVideo-Long [50] for three epochs, with each training example visited once per epoch. Videos are sampled at 16 FPS and processed at 832 × 480 resolution, using 321-frame clips for the long teacher. The horizon study uses separate teacher check points for the 81-, 161-, 241-, and 321-frame DMD configurations, pairing each rollout length with its corresponding teacher. The pretrained Wan VAE and UMT5 text encoder remain frozen; video latents and text embeddings are cached before optimization. We optimize a supervised flow-matching loss on the noisy video latents, sampling a diffusion timestep and Gaussian noise anew for each example and applying the scheduler’s timestep-dependent loss weight. Adaptation uses rank-64 LoRA with scaling parameter α = 64 in the DiT attention projections $( \mathrm { q } , \mathrm { k } , \mathrm { v } , \mathrm { o } )$ and feed-forward projections (ffn.0,ffn.2); the underlying pretrained weights remain frozen. AdamW uses a base learning rate of $1 0 ^ { - 4 } ,$ weight decay 0.01, and its default momentum parameters (0.9, 0.999). The implementation uses BF16 model param eters, evaluates the mean-squared-error loss in FP32, enables gradient checkpointing, and processes one clip per GPU without gradient accumulation, giving a global batch size of 32 clips. After adaptation, the resulting teacher is frozen during DMD; only the causal generator and the online fake-score network are updated in that stage.

Resources and DMD configuration. We follow Causal Forcing [10] for DMD timestep sampling and loss weighting. LongForcing extends the real-score teacher’s supervision horizon and the student rollout length. Interactive teacher training uses 256 GPUs, while text-to-video teacher training uses 32 NVIDIA H20 GPUs. The subsequent DMD stage runs for 1,000 global steps on eight GPUs with a global batch size of eight trajectories in both text-to-video and interactive experiments. The generator and fake-score learning rates are $2 \times 1 0 ^ { - 6 }$ and $4 \times 1 0 ^ { - 7 }$ , respectively, with AdamW momentum parameters (0, 0.999) and weight decay 0.01. For text-to-video DMD training, the fake-score network is updated at every global step, whereas the generator is updated once every five global steps, giving a 5:1 ratio of fake-score to generator updates. For both the 1.3B and 5B interactive world models, the fake-score network is updated at every global step and the generator once every three global steps during DMD training.

## A.2 Ideal Objective and Practical Scope

The ideal clean objective assigns weight only to prefixes reached inside the H-step training rollout. Let $q _ { \theta , \mathbf { \nu } }$ <sub>t</sub> and $q _ { \phi , t } ^ { * }$ denote the student and teacher chain-rule conditionals induced by their respective joint path distributions. Then

$$
\mathcal { L } _ { H } ^ { \mathrm { i d e a l } } = \sum _ { t = 1 } ^ { H } \mathbb { E } _ { \mathbf { \lambda } \mathbf { z } _ { H } \sim \mu _ { H } } , D _ { \mathrm { K L } } \left( q _ { \theta , t } ( \cdot \mid \mathbf { z } _ { < t } , \mathcal { C } _ { H } ) \big \| q _ { \phi , t } ^ { * } ( \cdot \mid \mathbf { z } _ { < t } , \mathcal { C } _ { H } ) \right) .\tag{A1}
$$

The expectation is under the student’s on-policy prefix distribution; the teacher conditional is derived from its joint distribution and need not be implemented as a causal model. Increasing H gives delayed prefixes in Eq. (2) nonzero weight, whereas prefixes first reached after H remain outside the objective.

Equation (3) describes clean-path matching, while Eq. (8) defines its noisy-path counterpart used to motivate DMD. Appendix A.4 states the conditions under which the score-difference estimator is unbiased for the noisy objective. The clean-path bounds in Section 4 apply when their stated KL conditions hold; extensions to later blocks use the conditions in Appendix B.4. For the interactive world model, training uses an action plan specified before the matched path and supplied to the teacher over the same horizon; extending the theory to state-feedback actions requires a joint state–action reference distribution.

Long-horizon DMD increases training memory and compute because the student unrolls farther and both score networks process a longer path. At fixed spatial resolution, dense bidirectional attention has quadratic attention-score computation in temporal sequence length; total memory and runtime also depend on the implementation. For a fixed output length and inference configuration, LongForcing retains the student architecture and sampling procedure, adding no inference operations.

## A.3 Comparison of Long-Horizon Rollout Strategies

Figure A1 contrasts the supervision coverage and inference mechanisms of representative long-horizon rollout strategies. The comparison distinguishes where teacher supervision is applied from mechanisms that modify inference-time conditioning.

![](images/4ab0a4fe5cb12c179289fdc606e9e79adb88daac2a657aa4114cc4497977b4f9.jpg)  
Figure A1: Mechanistic comparison of sparse local matching, inference-time anchoring, and LongForcing’s contiguous H-step on-policy trajectory matching under an extended-horizon teacher.

## A.4 DMD Score-Difference Estimator

For a shared condition $\mathcal { C } _ { H }$ , let $\mathbf { z } = G _ { \theta } ( \pmb { \xi } ; \mathcal { C } _ { H } )$ be a clean student path and $\mathbf { y } = \alpha _ { s } \mathbf { z } + \sigma _ { s } \pmb { \epsilon }$ its noised version. With $s _ { \mathrm { r e a l } }$ and $s _ { \mathrm { f a k e } }$ denoting estimates of the teacher and student noisy scores, the unweighted,

full-Jacobian score-difference estimator is

$$
\widehat { \bf g } _ { \mathrm { D M D } } = \alpha _ { s } \left( \frac { \partial \mathbf { z } } { \partial \theta } \right) ^ { \top } \mathrm { s g } [ s _ { \mathrm { f a k e } } ( \mathbf { y } , s ) - s _ { \mathrm { r e a l } } ( \mathbf { y } , s ) ] .\tag{A2}
$$

Equation (A2) presents the ideal, unweighted estimator; our implementation follows the sampling and weighting scheme of Causal Forcing [10]. Here sg treats the score difference as fixed during the generator update; condition arguments are suppressed. The factor $\alpha _ { s }$ follows from differentiating the forwardnoising map. For a condition and noise-level sampling law independent of $\theta ,$ exact scores, and full rollout differentiation, the expectation equals $\nabla _ { \boldsymbol { \theta } } \mathcal { I } _ { H } ^ { \mathrm { n o i s y } }$ , assuming differentiable positive noisy densities and valid interchange of differentiation and integration. In particular, the expected explicit derivative of the student log-density vanishes by normalization, leaving the pathwise score difference. Stop-gradient implements this expression without differentiating the score networks; score estimation, reweighting, normalization, or rollout-gradient truncation can introduce approximations to this ideal update. Both score networks attend bidirectionally over the sampled path: the real-score network is frozen during DMD, whereas the fake-score network is updated online to track the student’s rollout distribution.

## B Theoretical Details and Proofs

## B.1 Formal Path Setup

All state, action, and latent spaces are taken to be standard Borel spaces, so the conditional path laws below exist. We use natural logarithms in KL and $d _ { \mathrm { T V } } ( P , Q ) = \operatorname* { s u p } _ { A } | P ( A ) - Q ( A )$ . Let $S _ { t } = \Gamma ( S _ { t - 1 } , \mathbf { z } _ { t } )$ be the student’s causal state after chunk t, including any finite context or cache required for the next prediction; fixed conditions and model parameters are suppressed in Γ. All event comparisons use the same measurable path-to-state map and detector for student and teacher paths, with the model parameters held fixed. For matched horizon $H ,$ , define $\mathcal { C } _ { H } = ( S _ { 0 } , \mathbf { a } _ { 1 : H } , \mathbf { c } )$ , where $\mathbf { a } _ { 1 : H }$ is fixed before the window and c contains persistent conditions. The causal student induces

$$
Q _ { \theta , H } ( \mathbf { z } _ { 1 : H } \mid \mathcal { C } _ { H } ) = \prod _ { t = 1 } ^ { H } p _ { \theta } ( \mathbf { z } _ { t } \mid S _ { t - 1 } , \mathbf { a } _ { 1 : t } , \mathbf { c } ) ,\tag{B1}
$$

whereas the bidirectional teacher defines the joint reference $Q _ { \phi , H } ^ { * , \mathrm { l o n g } } ( \mathbf { z } _ { 1 : H } \mid \mathcal { C } _ { H } )$ without requiring a causal implementation. Fixing the actions open-loop prevents conditioning on actions generated from the same path; feedback policies instead require a joint state–action reference law.

For $\mathcal { C } _ { H } \sim \mu _ { H }$ , the joint measures are $\overline { { Q } } _ { \theta , H } = \mu _ { H } Q _ { \theta , H }$ and $\overline { { Q } } _ { \phi , H } ^ { * } = \mu _ { H } Q _ { \phi , H } ^ { * , \mathrm { l o n g } }$ . Because they share $\mu _ { H }$ Eq. (3) is equivalently $D _ { \mathrm { K L } } ( \overline { { Q } } _ { \theta , H } | | \overline { { Q } } _ { \phi , H } ^ { * } )$ . For the exposure comparison, both events are evaluated on the same rollout law with the same initial-condition distribution and fixed detector. Identifying a shorter event with its extension to the longer path space gives $B _ { H _ { 1 } } ( \gamma ) \subseteq B _ { H _ { 2 } } ( \gamma )$ for $H _ { 1 } < H _ { 2 }$ . This construction does not require independently trained teachers at different horizons to have consistent marginals.

## B.2 Non-Identifiability and Delayed Exposure

Proof of Proposition 4.1. For $b \in \{ 0 , 1 \}$ , define the uniform binary path law

$$
Q _ { T } ^ { ( b ) } = \operatorname { U n i f } \left\{ x _ { 1 : T } \in \{ 0 , 1 \} ^ { T } : \sum _ { t = 1 } ^ { T } x _ { t } \equiv b { \pmod { 2 } } \right\} .\tag{B2}
$$

Thus $Q _ { T } ^ { ( 0 ) }$ assigns mass to even-parity paths and $Q _ { T } ^ { ( 1 ) }$ to odd-parity paths. Both admit causal realizations: sample the first $T - 1$ bits independently from Bernoulli(1/2), then set the last bit to enforce the prescribed parity. For any coordinate set I with $| I | = k < T$ , fixing $x _ { I }$ leaves exactly $2 ^ { T - k - 1 }$ completions of either parity. Each full-path support has size $2 ^ { T - 1 }$ , hence

$$
Q _ { T } ^ { ( 0 ) } ( x _ { I } ) = Q _ { T } ^ { ( 1 ) } ( x _ { I } ) = 2 ^ { - k } .\tag{B3}
$$

In particular, every contiguous $W < T$ window has the same uniform marginal under both laws. Their full-path supports are disjoint, giving $d \mathrm { \mathrm { T V } } ( Q _ { T } ^ { ( 0 ) } , Q _ { T } ^ { ( 1 ) } ) = 1$ □

Concrete example: three steps and two-step windows. For $T = 3$ and $W = 2$ , the two laws are

$$
\begin{array} { r } { Q _ { 3 } ^ { ( 0 ) } = \mathrm { U n i f } \{ 0 0 0 , 0 1 1 , 1 0 1 , 1 1 0 \} , } \\ { Q _ { 3 } ^ { ( 1 ) } = \mathrm { U n i f } \{ 0 0 1 , 0 1 0 , 1 0 0 , 1 1 1 \} . } \end{array}\tag{B4}
$$

Under either law, both windows $( x _ { 1 } , x _ { 2 } )$ and $( x _ { 2 } , x _ { 3 } )$ assign probability $1 / 4$ to each of 00, 01, 10, 11. A discriminator restricted to either window therefore cannot distinguish the laws, while the full-path event $\{ x _ { 1 } + x _ { 2 } + x _ { 3 }$ is even} has probability one under $Q _ { 3 } ^ { ( 0 ) }$ and zero under $Q _ { 3 } ^ { ( 1 ) }$ . The missing information is the coupling across windows: local agreement alone leaves a global path constraint undetermined.

The proposition concerns objectives defined by local-window marginals; exact matching of every causal conditional on its full support would determine the joint path. For a measurable region $\mathcal { R } _ { : }$ , let $\tau _ { \mathcal { R } } =$ inf{t : $S _ { t } \in \mathcal { R } \}$ and $h _ { \mathcal { R } } ( H ) = \operatorname* { P r } ( \tau _ { \mathcal { R } } \leq H )$ . With N independent horizon-H rollouts,

$$
\operatorname* { P r } ( \mathcal { R } \operatorname { i s } \operatorname { n e v e r } \operatorname { e x p o s e d } ) = ( 1 - h _ { \mathcal { R } } ( H ) ) ^ { N } .\tag{B5}
$$

If $h \mathcal { R } ( H _ { 1 } ) = 0$ , more $\mathrm { \ h o r i z o n { - } } H _ { 1 }$ samples cannot expose R; sequential depth cannot be replaced by sample count. This is an exposure statement for a fixed rollout distribution, not a monotonic improvement claim across separately trained students.

## B.3 Within-Horizon Path Control

ProofofTheorem 4.2. Conditionally on $\mathcal { C } _ { H }$ , Pinsker’s inequality gives

$$
\begin{array} { r } { d _ { \mathrm { T V } } \Big ( Q _ { \theta , H } , Q _ { \phi , H } ^ { * , \mathrm { l o n g } } \Big ) \leq \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } \Big ( Q _ { \theta , H } \lVert Q _ { \phi , H } ^ { * , \mathrm { l o n g } } \Big ) } . } \end{array}\tag{B6}
$$

Average over $\mu _ { H }$ and apply Jensen’s inequality to obtain $d _ { \mathrm { T V } } ( \overline { { Q } } _ { \theta , H } , \overline { { Q } } _ { \phi , H } ^ { * } ) \leq \sqrt { \mathcal { L } _ { H } ^ { \mathrm { i d e a l } } / 2 }$ . The event bound follows from the definition of total variation. □

The same argument controls any measurable $F _ { H } \in [ 0 , 1 ]$

$$
\left| \mathbb { E } _ { \overline { { Q } } _ { \theta , H } } F _ { H } - \mathbb { E } _ { \overline { { Q } } _ { \phi , H } ^ { * } } F _ { H } \right| \leq \sqrt { \mathcal { L } _ { H } ^ { \mathrm { i d e a l } } / 2 } .\tag{B7}
$$

Taking $\begin{array} { r } { F _ { H } = H ^ { - 1 } \sum _ { t = 1 } ^ { H } \mathbf { 1 } \{ S _ { t } \in \mathcal { R } \} } \end{array}$ yields the finite-horizon occupancy bound without assuming ergodicity. Indeed, $\begin{array} { r } { F _ { H } \ = \ \int _ { 0 } ^ { 1 } { \bf 1 } \{ F _ { H } \ > \ u \} } \end{array}$ du; integrating the event bound for $\left\{ F _ { H } \ > \ u \right\}$ proves the bounded-functional inequality above. The next subsection extends the analysis to later block boundaries under explicit leakage and recovery conditions.

## B.4 Finite-Depth Drift Control

For an open-loop action block, define the complete condition

$$
X _ { j } = ( S _ { 0 } , \mathbf { z } _ { 1 : j B } , \mathbf { a } _ { 1 : ( j + 1 ) B } , \mathbf { c } ) , \qquad \nu _ { j } = \mathrm { L a w } _ { Q _ { \theta } } ( X _ { j } ) ,\tag{B8}
$$

where the future suffix $\mathbf { a } _ { j B + 1 : ( j + 1 ) B }$ is fixed before the block. If the block is a conditional of a matched H-step joint path, use $X _ { j } = ( \mathcal { C } _ { H } , \mathbf { z } _ { 1 : j B } )$ instead, retaining the entire teacher condition, including actions after the block. The full prefix may be replaced only by an augmented state sufficient for both student and teacher path laws. Let $\rho _ { i } ^ { \mathcal { G } } = \nu _ { j } ( \cdot \mid S _ { j B } \in \mathcal { G } )$ and $\rho _ { j } ^ { \mathcal { D } } = \nu _ { j } ( \cdot \mid S _ { j B } \in \mathcal { D } )$ when the corresponding events have positive mass; otherwise their conditional values are immaterial in Eq. (5).

For each start x, the teacher reference is a specified measurable continuation kernel conditioned on that same prefix and action block. Within a matched joint path, it is the corresponding chain-rule conditional; a continuation outside that horizon requires a separately specified reference. Let $\lambda _ { j } ^ { * }$ be the teacher’s average terminal probability of $\mathcal { D }$ from $\rho _ { j } ^ { \mathcal { G } }$ and $\eta _ { j } ^ { * }$ its average terminal probability of $\mathcal { G }$ from $\rho _ { j } ^ { \mathcal { D } }$ . Define

$$
\epsilon _ { j } ^ { \mathcal { G } } = \mathbb { E } _ { \boldsymbol { x } \sim \rho _ { j } ^ { \mathcal { G } } } D _ { \mathrm { K L } } \Big ( Q _ { \theta , B } \big ( \cdot \mid \boldsymbol { x } \big ) \Big \| Q _ { \phi , B } ^ { * , \mathrm { l o n g } } \big ( \cdot \mid \boldsymbol { x } \big ) \Big ) ,\tag{B9}
$$

and define $\epsilon _ { j } ^ { \mathcal { D } }$ analogously under $\rho _ { j } ^ { D }$ . The finite-KL assumptions entail student-to-teacher absolute continuity almost everywhere under the respective start distribution. Write $P _ { j } ^ { x }$ and $R _ { i } ^ { x }$ for these student and teacher block laws, and $K _ { j } ( x ) = D _ { \mathrm { K L } } ( P _ { j } ^ { x } \| R _ { j } ^ { x } )$ . For either terminal region $E \in \{ \mathcal { G } , \mathcal { D } \}$ , its probability difference is at most $d _ { \mathrm { T V } } ( P _ { j } ^ { x } , R _ { j } ^ { x } ) \leq \sqrt { K _ { j } ( x ) / 2 } ,$ since the terminal state is a measurable function of the start and block path. Averaging under the appropriate conditional start law and applying Jensen’s inequality gives

$$
\begin{array} { r l } & { a _ { j } - \lambda _ { j } ^ { * } \le \mathbb { E } _ { \rho _ { j } ^ { \varsigma } } \sqrt { K _ { j } / 2 } \le \sqrt { \epsilon _ { j } ^ { \mathcal { G } } / 2 } , } \\ & { \eta _ { j } ^ { * } - \eta _ { j } \le \mathbb { E } _ { \rho _ { j } ^ { \mathcal { D } } } \sqrt { K _ { j } / 2 } \le \sqrt { \epsilon _ { j } ^ { \mathcal { D } } / 2 } . } \end{array}\tag{B10}
$$

Combining the second inequality with $\eta _ { j } \geq 0$ proves Eq. (6).

Relation to the path objective. Let $\mathcal { L } _ { B , j } = \mathbb { E } _ { \nu _ { j } } K _ { j }$ denote the block-specific on-policy KL. Splitting its expectation by the start region gives

$$
\mathcal { L } _ { B , j } = ( 1 - b _ { j } ) \epsilon _ { j } ^ { \mathcal { G } } + b _ { j } \epsilon _ { j } ^ { \mathcal { D } } .\tag{B11}
$$

For $0 < b _ { j } < 1$ , nonnegativity therefore implies $\epsilon _ { j } ^ { \mathcal { G } } \leq { \mathcal { L } } _ { B , j } / ( 1 - b _ { j } )$ and $\epsilon _ { j } ^ { D } \leq \mathcal { L } _ { B , j } / b _ { j } ;$ a zero-mass term contributes zero. When all blocks use the chain-rule conditionals of the same matched joint laws and $H \ = \ J B$ , the KL chain rule gives $\begin{array} { r } { \mathcal { L } _ { H } ^ { \mathrm { i d e a l } } \ = \ \sum _ { j = 0 } ^ { J - 1 } \mathcal { L } _ { B , j } } \end{array}$ . For $J B \ < \ H$ , the remaining suffix contributes an additional nonnegative conditional KL. If instead a block loss is averaged under a different training start law $\widetilde { \mu } _ { j }$ , transfer to $\rho _ { j } ^ { E }$ requires a measure comparison: whenever $\rho _ { j } ^ { E } \ll \widetilde { \mu } _ { j }$ and $d \rho _ { j } ^ { E } / d \widetilde { \mu } _ { j } \leq$ $M _ { i } ^ { E }$ , integration gives $\epsilon _ { j } ^ { E } \le M _ { j } ^ { \bar { E } } \mathbb { E } _ { \widetilde { \mu } _ { j } } K _ { j }$ . These relations control conditional averages; the finite-depth proposition explicitly assumes the required good- and degraded-start bounds at each evaluated block.

ProofofProposition 4.3. Under the proposition’s assumptions, Eq. (6) gives $a _ { j } \leq A$ and $\eta _ { j } \geq \underline { { \eta } }$ . Together with Eq. (5), this implies $b _ { j + 1 } \leq A ( 1 - b _ { j } ) + ( 1 - \eta ) b _ { j } \leq A + ( 1 - \eta ) b _ { j }$ . Probability bounds are taken in [0, 1], so $r = 1 - \eta \in [ 0 , 1 )$ . Induction yields $\begin{array} { r } { b _ { j } \le r ^ { j } b _ { 0 } + A \sum _ { k = 0 } ^ { j - 1 } r ^ { k } = r ^ { j } b _ { 0 } + A ( 1 - r ^ { j } ) / \eta } \end{array}$ for $1 \le j \le J$ . Together with the case $j = 0$ and $b _ { j } \leq 1$ , this proves Eq. (7). The monotone relaxation avoids any sign requirement on $1 - A - \underline { { \eta } } .$ □

Partial start coverage. Coverage is measured under the relevant conditional start law. If the student’s per-start leakage is at most A and its per-start recovery is at least $\underline { { \eta } }$ only on subsets with masses at least $1 - \delta _ { j } ^ { \mathcal { G } }$ under $\rho _ { j } ^ { \mathcal { G } }$ and $1 - \delta _ { j } ^ { D }$ under $\rho _ { j } ^ { \mathcal { D } }$ , respectively, the averaged bounds become

$$
a _ { j } \leq A + ( 1 - A ) \delta _ { j } ^ { \mathscr { G } } , \qquad \eta _ { j } \geq ( 1 - \delta _ { j } ^ { \mathscr { D } } ) \underline { { \eta } } .\tag{B12}
$$

To see this, bound leakage on uncovered good starts by one and recovery on uncovered degraded starts by zero, then average over the two subsets. These adjusted quantities enter the recursion at each block. With complete degraded-start coverage and $\delta _ { j } ^ { \mathcal { G } } \leq \delta$ , each affine update acquires at most $( 1 - A ) \delta ;$ summing its geometrically weighted contributions gives additional envelope slack at block $j$ of at most $( 1 - A ) \delta [ 1 -$ $( 1 - \underline { { \eta } } ) ^ { j } ] / \underline { { \eta } }$

Feedback actions. The total-probability identity in Eq. (5) also holds for feedback policies, without Markov or stationarity assumptions. For its KL-based bounds, specify a policy $\pi _ { t } ( d \mathbf { a } _ { t } \ \mid h _ { t - 1 } )$ and a nonanticipating teacher continuation $q _ { t } ^ { * } ( d \mathbf { z } _ { t } \ \mid \ h _ { t - 1 } , \mathbf { a } _ { t } )$ , where $h _ { t - 1 }$ contains the available history and policy state. From the same block start, define joint action–latent laws

$$
\begin{array} { l } { { \displaystyle P _ { j } ^ { \pi } = \prod _ { t = j B + 1 } ^ { ( j + 1 ) B } \pi _ { t } ( d { \bf a } _ { t } \mid h _ { t - 1 } ) p _ { \theta } ( d { \bf z } _ { t } \mid h _ { t - 1 } , { \bf a } _ { t } ) } , } \\ { { \displaystyle R _ { j } ^ { \pi } = \prod _ { t = j B + 1 } ^ { ( j + 1 ) B } \pi _ { t } ( d { \bf a } _ { t } \mid h _ { t - 1 } ) q _ { t } ^ { * } ( d { \bf z } _ { t } \mid h _ { t - 1 } , { \bf a } _ { t } ) } . } \end{array}\tag{B13}
$$

Under finite KL, the shared policy factors cancel in the likelihood ratio, and the chain rule expresses $D _ { \mathrm { K L } } ( P _ { j } ^ { \pi } \Vert R _ { j } ^ { \pi } )$ as the sum of conditional latent KLs averaged over student histories and actions. Applying the preceding Pinsker argument to these joint laws proves the feedback version with its own KL, leakage, and recovery quantities. This construction requires a specified nonanticipating reference; conditioning a bidirectional open-loop teacher on actions realized through feedback does not supply that reference automatically. Throughout, $b _ { j }$ measures occupancy at boundaries spaced by the fixed stride $B ;$ withinblock visits are covered by path events or the bounded functionals in Appendix B.3.

## B.5 Conditional Long-Run Extension

Fix a persistent condition c and a time-homogeneous stationary policy π, and let $Y _ { j }$ include the generator cache and policy state sufficient for a time-homogeneous Markov boundary process. Suppose its initial law is supported on a measurable class $\mathcal { R } _ { c , \pi } ,$ , with $P _ { \theta , B } ^ { c , \pi } ( y , \mathcal { R } _ { c , \pi } ) = 1$ for every y in that class, and its block kernel satisfies the Doeblin minorization $P _ { \theta , B } ^ { c , \pi } ( y , \cdot ) \geq \beta _ { c , \pi } m _ { c , \pi } ( \cdot )$ there, with $\beta _ { c , \pi } \in ( 0 , 1 ]$ and $m _ { c , \pi }$ a probability measure supported on the class. The regions $\mathcal { G }$ and D are lifted to $Y _ { j }$ through its visual-state component. Then the process for this fixed condition and policy has a unique stationary distribution $\nu _ { \infty } ^ { c , \pi }$ and its empirical degraded-boundary fraction converges almost surely to $\nu _ { \infty } ^ { c , \pi } ( \mathcal { D } )$ . If, pointwise on $\mathcal { R } _ { c , \pi } .$ good states leak with probability at most $A _ { c , \pi }$ and degraded states return with probability at least $\eta _ { c , \pi } > 0$ then

$$
\nu _ { \infty } ^ { c , \pi } ( \mathcal { D } ) \leq \frac { A _ { c , \pi } } { A _ { c , \pi } + \eta _ { c , \pi } } .\tag{B14}
$$

Proof. Suppress $c , \pi$ and write $P = P _ { \theta , B } ^ { c , \pi }$ . For $0 < \beta < 1$ , the residual kernel $R = ( P - \beta m ) / ( 1 - \beta )$ is a probability kernel on the closed class. Thus $P = \beta m + ( 1 - \beta ) R$ , and

$$
\nu _ { \infty } = \beta \sum _ { k = 0 } ^ { \infty } ( 1 - \beta ) ^ { k } m R ^ { k }\tag{B15}
$$

is a probability measure satisfying $\nu _ { \infty } P = \nu _ { \infty }$ by shifting the series. Since Markov kernels contract total variation,

$$
d _ { \mathrm { T V } } ( \nu P ^ { k } , \nu ^ { \prime } P ^ { k } ) \leq ( 1 - \beta ) ^ { k } d _ { \mathrm { T V } } ( \nu , \nu ^ { \prime } ) .\tag{B16}
$$

This proves uniqueness and uniform convergence to $\nu _ { \infty }$ . The decomposition also constructs regeneration times: at each transition, independently refresh from m with probability $\beta ,$ otherwise use R. After the first refresh, cycles are independent and identically distributed, with geometric lengths of mean $1 / \beta$ and bounded per-step rewards. The renewal-reward strong law gives almost-sure convergence of the empirical average of ${ \bf 1 } _ { \mathcal { D } } ;$ its limit equals $\nu _ { \infty } ( \mathcal { D } )$ by the displayed uniform convergence and boundedness. For $\beta = 1$ the process is i.i.d. from m after one transition, giving the same conclusions. Finally, stationarity and the pointwise leakage/return bounds give $b _ { \infty } \leq A _ { c , \pi } ( 1 - b _ { \infty } ) + ( 1 - \eta _ { c , \pi } ) b _ { \infty } ;$ rearranging proves the stated fraction bound. □

This is the standard minorization and regeneration argument for general-state-space chains [51]. This extension assumes the stated uniform minorization, leakage, and recovery conditions for the fixed prompt and policy. It bounds the long-run fraction of degraded block boundaries, rather than the probability of ever visiting a degraded state.

## C Evaluation Details and Additional Results

## C.1 Full VBench Results

Table 6: Full 5-second VBench comparison.
<table><tr><td>Model</td><td>#Params</td><td>Resolution</td><td>Throughput (FPS) ↑</td><td>Latency (s) ↓</td><td>VBench Total ↑</td><td>Quality ↑</td><td>Semantic ↑</td></tr><tr><td colspan="8">Bidirectional video diffusion models</td></tr><tr><td>LTX-Video [52]</td><td>1.9B</td><td>768×512</td><td>8.98</td><td>13.5</td><td>79.83</td><td>81.88</td><td>71.62</td></tr><tr><td>Wan2.1 [43]</td><td>1.3B</td><td>832×480</td><td>0.78</td><td>103</td><td>83.37</td><td>84.30</td><td>79.65</td></tr><tr><td colspan="8">Autoregressive video diffusion models</td></tr><tr><td>NOVA [53]</td><td>0.6B</td><td>768×480</td><td>0.88</td><td>4.1</td><td>80.31</td><td>80.66</td><td>78.92</td></tr><tr><td>Pyramid Flow [54]</td><td>2B</td><td>640×384</td><td>6.70</td><td>2.5</td><td>80.75</td><td>83.41</td><td>70.11</td></tr><tr><td>SkyReels-V2 [55]</td><td>1.3B</td><td>960×540</td><td>0.49</td><td>112</td><td>81.97</td><td>83.96</td><td>74.01</td></tr><tr><td>MAGI-1 [23]</td><td>4.5B</td><td>832×480</td><td>0.19</td><td>282</td><td>78.88</td><td>81.67</td><td>67.72</td></tr><tr><td colspan="8">Distilled autoregressive video models</td></tr><tr><td>CausVid [9]</td><td>1.3B</td><td>832×480</td><td>17.0</td><td>0.69</td><td>81.33</td><td>83.98</td><td>70.72</td></tr><tr><td>Self Forcing [8]</td><td>1.3B</td><td>832×480</td><td>17.0</td><td>0.69</td><td>83.74</td><td>84.48</td><td>80.77</td></tr><tr><td>Causal Forcing [10]</td><td>1.3B</td><td>832×480</td><td>17.0</td><td>0.69</td><td>84.04</td><td>84.59</td><td>81.84</td></tr><tr><td>LongForcing (ours)</td><td>1.3B</td><td>832×480</td><td>17.0†</td><td>0.69†</td><td>84.56</td><td>85.52</td><td>80.72</td></tr></table>

† Reference throughput and latency quoted from Causal Forcing [10] for its inference configuration, which LongForcing retains.

Table 7: Full 30-second VBench-Long comparison.
<table><tr><td>Model</td><td colspan="3">Aggregate</td><td colspan="2">Long-horizon consistency</td><td colspan="2">Visual quality</td><td colspan="2">Temporal evolution</td></tr><tr><td></td><td>Total</td><td>Quality</td><td>Semantic</td><td>Subject Cons.</td><td>Background Cons.</td><td>Aesthetic</td><td>Imaging</td><td>Dynamic</td><td>Temporal Style</td></tr><tr><td>LongLive [30]</td><td>82.80</td><td>83.21</td><td>81.17</td><td>98.36</td><td>96.25</td><td>63.77</td><td>68.35</td><td>38.89</td><td>24.15</td></tr><tr><td>Self Forcing [8]</td><td>81.49</td><td>82.67</td><td>76.77</td><td>97.76</td><td>95.00</td><td>61.72</td><td>67.70</td><td>41.93</td><td>23.56</td></tr><tr><td>Causal Forcing [10]</td><td>81.05</td><td>83.44</td><td>71.52</td><td>96.24</td><td>95.59</td><td>58.85</td><td>66.92</td><td>75.46</td><td>22.61</td></tr><tr><td>LongForcing (ours)</td><td>82.83</td><td>83.99</td><td>78.19</td><td>97.40</td><td>96.58</td><td>62.87</td><td>64.19</td><td>60.93</td><td>23.61</td></tr></table>

Best and second-best results are shown in bold and underlined, respectively. All scores are percentages.

## C.2 Causal-VAE Drift Statistics

We re-encode 20 videos per model with the shared causal VAE. Let $m _ { v , c , t }$ be the spatial mean of channel c at zero-based latent index t in video v, and define its early reference as

$$
\bar { m } _ { v , c } = \frac { 1 } { 2 0 } \sum _ { r = 1 } ^ { 2 0 } m _ { v , c , r } .\tag{C1}
$$

This reference excludes the first latent $( t = 0 )$ and averages latent indices 1–20, corresponding to zerobased RGB frame indices 1–80 (80 frames). Panel (b) of Figure 2 plots median $\mathbf { \rho } _ { 1 v , c } \left| m _ { v , c , t } - \bar { m } _ { v , c } \right|$ at each t. For the 81-frame model, panel (c) plots the signed channel-wise statistic median $\mathfrak { l } _ { v } \big ( m _ { v , c , t } - \bar { m } _ { v , c } \big )$ These statistics summarize the magnitude and channel-wise direction of latent shifts relative to the early reference window.

## C.3 Time-Resolved Drift Score

We sample frames at 1 Hz from 16-FPS videos, using indices 0, 16, . . . , 944 (60 observations per video). High saturation and RGB clipping are measured at decoded resolution. With OpenCV HSV saturation and value normalized to [0, 1], the high-saturation ratio is the fraction of all pixels satisfying $S > 0 . 8 5$ and $V > 0 . 1 5$ . The RGB-clipping ratio is the fraction with any 8-bit RGB channel at most 5 or at least 250. For chroma edges, frames are resized to a 256-pixel short side with aspect ratio preserved, integer-rounded dimensions, and OpenCV area interpolation. After RGB-to-YCrCb conversion, let $g _ { Y } , g _ { C r } ,$ and $g _ { C b }$ be the Euclidean magnitudes of horizontal and vertical 3×3 Sobel responses, computed in floating point on the unnormalized 8-bit channel scale. With $g _ { C } = \sqrt { g _ { C r } ^ { 2 } + g _ { C b } ^ { 2 } }$ , the chroma-edge ratio is the fraction of resized pixels satisfying $g _ { C } > 1 8$ and $g _ { C } > 1 . 5 ( g _ { Y } \dot { + } 1 )$ ). For text-to-video HPSv3 [56], each matched pair uses identical filename-derived prompt prefixes, held fixed across models and timestamps. These scores measure temporal changes relative to each video’s early reference. Interactive HPSv3 uses the prompt associated with each scene ID.

For video i and metric m, let $b _ { i m }$ be the mean over seconds 0–4. We orient deterioration as $d _ { i m } ( t ) =$ $b _ { i m } - x _ { i m } ( t )$ for HPSv3 and $d _ { i m } ( t ) = x _ { i m } ( t ) - b _ { i m }$ for three artifact ratios: high saturation, RGB clipping, and chroma edges. The normalized deterioration is

$$
z _ { i m } ( t ) = \mathrm { c l i p } \bigg ( \frac { [ d _ { i m } ( t ) ] _ { + } } { q _ { m } } , 0 , 1 \bigg ) .\tag{C2}
$$

The frozen scales are the pooled 95th percentiles of $[ d _ { i m } ( t ) ] _ { + }$ , including zeros, over both the 81- and 321-frame text-to-video groups, all 20 matched videos per group, all 60 sampled seconds, and each metric independently: $q _ { m } = ( 1 2 . 5 9 6 1 , 0 . 2 9 7 7 , 0 . 4 7 8 6 , 0 . 3 2 0 2 )$ for HPSv3, high saturation, RGB clipping, and

chroma edges, respectively. The per-video score is

$$
D _ { i } ( t ) = 1 0 0 \left[ \frac { 1 } { 2 } z _ { i , \mathrm { H P S } } ( t ) + \frac { 1 } { 6 } \sum _ { m \in \{ \mathrm { s a t } , \mathrm { c l i p } , \mathrm { c h r o m a } \} } z _ { i m } ( t ) \right] .\tag{C3}
$$

Thus HPSv3 and the correlated artifact family receive equal total weight. The score measures deterioration relative to each video’s early reference: lower values indicate smaller increases in the selected degradation indicators. Absolute perceptual quality and motion are assessed separately by the benchmark and human evaluations.

Figure C1 applies the same frozen scales to both regimes. Curves use a centered 5-second moving average, truncated to the available observations at each endpoint; numerical endpoint comparisons below use the unsmoothed 59-second observations. The pointwise 95% percentile bootstrap intervals resample complete prompt or scene trajectories 2,000 times and apply the same smoothing within each replicate, keeping the normalization scales fixed. At each timestamp, the interval endpoints are the 2.5th and 97.5th percentiles of the bootstrap means. The text-to-video and interactive panels contain 20 prompt-matched and 23 scenematched videos per model, respectively.

![](images/b32ac3eea451ee60029659c60e3ea08df71fd63463a6ae81c94868baffa40f06.jpg)  
(b) Interactive world model

![](images/a8eca051616549a2f94b2a039e4bbd40417ced6574dfc989cebad88bc3864951.jpg)  
Figure C1: Full time-resolved drift comparison. Curves show mean Drift Score with 95% bootstrap confidence intervals for text-to-video and interactive rollouts.

At 59 seconds, 321-frame supervision reduces the mean Drift Score from 67.46 to 31.91 for text-to-video generation and from 64.76 to 48.95 for interactive world modeling, corresponding to relative reductions of 52.7% and 24.4%, respectively.

## C.4 Human Preference Evaluation

Our analysis covers 20 shared cases evaluated by all 20 participants: 10 text-to-video prompts and 10 interactive world-model scenes. Each pair compares 81- and 321-frame supervision with model identities hidden and left/right placement randomized. Evaluators select A better, Tie, or B better for overall preference, temporal coherence, visual quality, and motion naturalness. Each participant provides one response per case and criterion, yielding $2 0 \times 2 0 = 4 0 0$ judgments per criterion and 1,600 judgments across the four criteria. Figure 3 reports descriptive preference proportions, assigning equal weight to each evaluator–case response and retaining ties in the denominator.

## D Additional Qualitative Results

## D.1 Text-to-Video Rollouts

Figure D1 follows four 60-second text-to-video rollouts trained with 321-frame supervision. Across portrait, human-motion, animated-aerial, and deep-space prompts, the displayed frames show recognizable subjects and scene style as content evolves beyond the supervised horizon.

![](images/be9dc190c15db6221363fc1597f53b8e970dc63848afbcdd6d28c5bd5598bb15.jpg)  
Figure D1: Qualitative 60-second text-to-video rollouts. Each row follows one LongForcing trajectory trained with 321-frame supervision; columns show frames at fixed video timestamps.

## D.2 Supervision-Horizon Comparison

Figure D2 compares matched prompts or initial scenes. The 81-frame text-to-video examples exhibit progressive saturation and structural distortion, while the interactive examples develop horizontal banding and geometry collapse. With 321-frame supervision, the displayed snapshots from all four trajectories preserve more scene structure and do not show the same late-stage artifacts.

![](images/35e087eb12af5f0ed56cb441d3647532630703a225d6b78e95801aeb657c829f.jpg)  
Figure D2: Qualitative comparison across supervision horizons. Upper and lower rows use 81- and 321-frame supervision, respectively, at matched rollout times from 0 to 59 seconds.

## E Discussion

When does LongForcing help? Horizon scaling targets failures first reached beyond the shorter training window. Extending the on-policy rollout makes these later prefixes available to the teacher, while the horizon ablations quantify the resulting empirical changes in generation quality. The benefit depends on both supervision coverage and the quality of the teacher signal on those prefixes.

Cost. The additional computation occurs during training: the student generates longer paths and both score networks process longer sequences. At fixed spatial resolution, dense bidirectional attention has quadratic attention-score computation in temporal length; practical memory and runtime depend on the attention and gradient implementations. For a fixed inference configuration and output length, the student uses the same architecture and sampling operations as the short-horizon baseline.

Scope. The finite-horizon results quantify teacher-relative path agreement, while the blockwise extension additionally assumes controlled leakage, mismatch, and recovery on the student’s start distribution. The long-run extension states its uniform dynamical assumptions separately. These conditions concern the specified model, domain, and action protocol. Visual quality, motion, and action following are evaluated separately, complementing the drift analysis.

How long is enough? A useful supervision horizon depends on when task-relevant dependencies and failure modes become observable, rather than on a universal duration. For scene roaming, delayed effects may involve revisiting a location or maintaining an object after a long absence; for goal-directed interaction, an action’s consequence may appear much later. Our horizon sweep measures the benefit of extending supervision within the tested range; determining a sufficient horizon for a new task calls for evaluating those dependencies over the intended deployment duration. A promising direction is to allocate longer training rollouts when delayed errors remain frequent, and to compare this adaptive allocation with fixed horizons under matched compute budgets. Teacher quality and persistent memory offer complementary ways to make additional temporal coverage useful.

From visual rollout to long-term planning. Maintaining a coherent visual trajectory and deciding which events should occur over a long story or task are complementary problems. LongForcing strengthens the former by supervising longer self-generated paths; long-term planning additionally requires representing goals, organizing events, and retaining consequences across distant steps. This suggests a hierarchical approach in which a planner maintains an abstract state and selects subgoals, persistent memory records relevant past events, and a video model renders the resulting interactions. Long-horizon supervision could then assess whether the rendered trajectory respects those plans and remembered consequences. Evaluating delayed action effects, goal completion, and narrative consistency alongside visual quality would help determine whether longer rollouts also support more coherent long-term behavior.