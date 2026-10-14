# FlowCred: Hierarchical Credit Assignment and Functional Coordination for Joint Text-to-Video Post-Training

Mingxuan Zhu<sup>1,2∗</sup> Yanwei Huang<sup>2,3</sup> Shiyuan Liu<sup>2</sup>

Xinzhi Zhang<sup>2,4</sup> Shujie Li<sup>2,5</sup> Yuanxing Zhang<sup>2</sup> Lu Chen<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>KlingAI <sup>3</sup>The Hong Kong University of Science and Technology <sup>4</sup>South China University of Technology <sup>5</sup>The University of Hong Kong

## Abstract

Prompt optimization and generator post-training are two complementary ways to improve text-to-video (T2V) generation, and recent systems increasingly train both modules jointly. However, optimizing a prompt policy and a video generator from the same terminal reward creates two ambiguities: prompt quality is confounded with stochastic generation, and the two modules may realize overlapping local improvements. We introduce FLOWCRED, a two-level framework for joint T2V post-training. First, hierarchical credit assignment uses matched prompt–seed rollout grids to separate prompt-level credit from prompt-conditioned generator residuals under shared randomness. Second, functional coordination probes alternative current-policy prompts at states from actual generation trajectories, providing generator-aware evidence for prompt optimization while selectively attenuating generator updates already supported by conditioning changes. Across multiple T2V benchmarks, FLOWCRED consistently improves matched joint-training baselines. On LTX-2.5, FLOWCRED improves a strictly matched joint-training control from 61.47 to 64.46 on T2V-CompBench and from 55.58 to 58.08 on StoryEval, without increasing decoded-video or reward-model-call budgets. Factorial and realized-update analyses show complementary contributions from prompt shaping and generator routing, while human evaluation and transfer across backbones and updater families provide further supporting evidence.

## 1 Introduction

Text-to-video (T2V) systems increasingly expose two trainable adaptation channels: a language-side prompt policy and the video generator itself. Prompt-side methods improve conditioning while leaving the visual model fixed (Hao et al., 2023; Cao et al., 2023; Ji et al., 2025; Cheng et al., 2025); generator post-training instead changes the diffusion/flow model using reward or preference feedback (Black et al., 2024; Fan et al., 2023; Wallace et al., 2024; Liu et al., 2025a; Xue et al., 2025; 2026). Recent systems optimize both modules jointly (Wang et al., 2026a; Cheng et al., 2026; Liu et al., 2026; Jia et al., 2026). Prompt changes can reorganize entities, relations, and event descriptions before generation, whereas generator updates address residual motion, appearance, and execution errors that conditioning alone cannot remove. When only one module is updated, a terminal reward can be treated as supervision for that module; once both are trainable, the same scalar signal must implicitly serve both attribution and coordination roles. Joint post-training is therefore attractive; once both channels are trainable, however, one terminal score leaves two distinct questions unresolved.

The first is statistical credit. A terminal reward conflates prompt quality with downstream generator randomness: identical prompts can score differently across seeds, while favorable trajectories can mask weaker prompt choices. The reward alone therefore does not identify which variation belongs to prompt choice and which remains after the prompt is fixed. For request $u ,$ prompt action $p ,$ stochastic generator trajectory T, and terminal reward R, let $Q ( u , p ) = \mathbb { E } [ R \mid u , \bar { p } ]$ and $V ( u ) = \mathbb { E } _ { p } Q ( u , p )$

![](images/324eca0320eb399915236690c66db24c6c1aac0262011b3f87797e23982ff993.jpg)  
Figure 1: Motivation and positioning of FLOWCRED. The left panel contrasts prompt-only adaptation with shared-reward joint training. Once both modules are trained from a common terminal reward, two distinct issues arise: statistical credit ambiguity between prompt choice and prompt-fixed stochastic variation, and functional overlap between conditioning-side and generator-side local effects. FLOWCRED addresses them with hierarchical credit assignment (Q − V versus R − Q) and functional coordination based on same-state response geometry.

## Then

$$
R - V ( u ) = \underbrace { Q ( u , p ) - V ( u ) } _ { \mathrm { p r o m p t \ : d e c i s i o n } } + \underbrace { R - Q ( u , p ) } _ { \mathrm { p r o m p t \ : f i x e d \ : t r a j e c t o r y } } .\tag{1}
$$

This exposes two sources of terminal variation. The second question is functional coordination: even after statistical credit is separated, changing the prompt at a generator state may induce a response aligned with the same reward-improving direction used by the downstream optimizer. The two modules can therefore duplicate part of the same local improvement. Statistical credit and functional overlap are thus distinct problems.

These two ambiguities require different remedies: variance reduction can make credit estimates cleaner without preventing functional overlap, while coordinating local updates does not by itself determine where terminal variation originated. Treating them separately lets joint post-training distinguish who should receive credit from which module should realize an overlapping improvement.

To address both, we introduce FLOWCRED, named for its joint treatment of credit assignment and coordination across the prompt-to-video generation flow. Figure 1 summarizes the two ambiguities and their corresponding mechanisms. Hierarchical credit assignment uses matched prompt–seed rollouts to separate prompt-level credit from prompt-conditioned generator residuals. Functional coordination probes alternative prompts at actual rollout states and uses same-state response geometry for generator-aware prompt shaping and reliability-gated generator routing.

The two mechanisms have different mathematical status: Eq. 1 is exact for any per-trajectory scalar score, whereas functional coordination is local and first-order. Its prompt term is auxiliary shaping and its generator rule is a minimum-overlap regularizer, not a conserved transfer of gradient mass. To keep the statistical claim separate from optimization heuristics, all affine calibration used to combine reward credit and functional prompt-credit terms is estimated once on disjoint training-only data and then frozen (Appendix D.3).

## Our contributions are threefold.

(1) We introduce hierarchical credit assignment with matched prompt–seed rollouts, separating prompt-level credit from prompt-conditioned generator residuals under shared generator randomness. This gives the prompt policy and generator distinct estimators tied to their positions in the stochastic hierarchy.

(2) We develop functional coordination from same-state prompt responses, using local response geometry for generator-aware prompt shaping and reliability-gated routing of the generator reward update. It targets overlap that remains even after statistical credit has been separated.

(3) We evaluate FLOWCRED with controlled joint-training baselines, mechanism audits, repeated runs, human evaluation, and transfer across backbones and updater families. Across four T2V benchmarks, FLOWCRED consistently improves the matched controls without increasing decoded-video or rewardmodel-call budgets.

## 2 Related Work

Prompt optimization and generator post-training. Prompt-side methods optimize conditioning with a fixed generator (Hao et al., 2023; Cao et al., 2023; Ji et al., 2025; Cheng et al., 2025), whereas PromptRL, VANS, UniGRPO, and CAPE-T2V couple language- and generator-side adaptation in different ways (Wang et al., 2026a; Cheng et al., 2026; Liu et al., 2026; Jia et al., 2026). Generator alignment spans policy gradients, preference objectives, and differentiable reward optimization (Black et al., 2024; Fan et al., 2023; Wallace et al., 2024; Clark et al., 2024; Li et al., 2024; Xu et al., 2023; Kirstain et al., 2023; Liu et al., 2025a; Choi et al., 2026; Prabhudesai et al., 2024; Wang et al., 2026b); recent methods further refine reward shaping, trajectory allocation, and temporal credit (Xue et al., 2025; 2026; Luo et al., 2026; Zhu et al., 2026; Ye et al., 2026; Pu & Kong, 2026; Shen et al., 2026; Guan et al., 2026; Savani et al., 2026; Oba et al., 2026). These directions are complementary to our cross-module question. Modern open T2V systems span diverse backbone designs (Chen et al., 2024; Yang et al., 2025; Kong et al., 2024; Wan et al., 2025); our experiments hold the downstream harness fixed whenever numerical comparisons require matched conditions.

Credit assignment, variance reduction, and functional coordination. COMA counterfactual credit holds other agents fixed while marginalizing one action (Foerster et al., 2018), while leave-oneout baselines and common random numbers are standard tools for variance reduction (Ahmadian et al., 2024; Ng & Jordan, 2000). PCGrad/CAGrad manipulate conflicting parameter gradients across objectives (Yu et al., 2020; Liu et al., 2021). FLOWCRED instead separates statistical credit across the stochastic prompt-policy/generator hierarchy and measures functional overlap through same-state conditioning responses in the generator’s native output space.

## 3 Method

## 3.1 Joint post-training: credit and coordination

Let $u \sim \mathcal { D }$ be a request, $p \sim \pi _ { \phi } ( \cdot \mid u )$ a prompt action, and $\tau \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid p )$ a stochastic generator trajectory ending in video x. The shared objective is

$$
J ( \phi , \theta ) = \mathbb { E } _ { u , p , \mathcal { T } } [ R ( u , x ) ] .\tag{2}
$$

All rewards are evaluated against the original request u. We use statistical credit for how terminal variation is assigned among sampled decisions and functional coordination for how locally overlapping conditioning and generator changes are handled. Accordingly, hierarchical credit assigns terminal variation according to where it arises, while functional coordination acts only on local function changes that both trainable modules can express. We refer to $\pi _ { \phi }$ as the prompt optimizer or prompt policy.

## 3.2 Hierarchical credit assignment

Define $Q _ { \theta } ( u , p ) = \mathbb { E } _ { \mathcal { T } \sim \pi _ { \theta } ( . | p ) } [ R ( u , x ) ]$ ] and $V _ { \phi , \theta } ( u ) = \mathbb { E } _ { p \sim \pi _ { \phi } ( . | u ) } [ Q _ { \theta } ( u , p ) ]$ ; we suppress parameter subscripts below. Thus $R - V = ( Q - V ) + ( R - Q )$ separates prompt choice from prompt-fixed stochasticity. For each request, sample K candidates $p _ { 1 : K } \sim \pi _ { \phi } ( \cdot \mid u )$ and reuse the same M generator-randomness seeds $z _ { 1 : M }$ for every candidate:

$$
x _ { k , m } , \mathcal { T } _ { k , m } = G _ { \theta } ( p _ { k } ; z _ { m } ) , \qquad R _ { k , m } = R ( u , x _ { k , m } ) .\tag{3}
$$

The matched rollout grid yields the paired leave-one-out estimators

$$
\begin{array} { l } { \displaystyle \widehat { A } _ { k } ^ { P , R } = \frac { 1 } { M } \sum _ { m } \left( R _ { k , m } - \frac { 1 } { K - 1 } \sum _ { j \neq k } R _ { j , m } \right) , } \\ { \displaystyle \widehat { A } _ { k , m } ^ { G } = R _ { k , m } - \frac { 1 } { M - 1 } \sum _ { m ^ { \prime } \neq m } R _ { k , m ^ { \prime } } . } \end{array}\tag{4}
$$

The first contrast compares prompt decisions under common generator randomness and estimates $Q - V ;$ the second compares trajectories after the prompt is fixed and estimates $R - Q$ . Positive same-seed reward covariance lowers cross-prompt contrast variance (Appendix A). The prompt optimizer uses $\widehat { A } ^ { P , R }$ , while the generator reward branch uses ${ \widehat { A } } ^ { G }$ . Reward-component calibration is estimated on disjoint training-only data and frozen, so no statistic from the current $K \times M$ grid enters $R _ { k , m }$ (Appendix D.3).

Controlled global-LOO baseline. Matched Global-LOO uses the identical $K \times M$ grid but gives both modules one global leave-one-out trajectory advantage (Appendix A.2). Averaged over the M trajectories for a prompt, this control preserves the same coarse prompt ordering up to positive scale. The substantive statistical change in hierarchical credit is therefore downstream: generator credit becomes the prompt-conditional residual $R - Q$ rather than a global outcome residual.

## 3.3 Functional coordination via same-state responses

Hierarchical credit separates endpoint variation but does not describe whether prompt conditioning and the generator reward step realize similar local function changes. We measure that overlap at states actually encountered by the generator and reuse the same response geometry for prompt shaping and generator routing. An encountered state $s _ { t } ^ { k , m }$ contains the noisy latent, timestep, and scheduler quantities required for one transition on trajectory $( k , m )$ . Let $\mu _ { t } = \mu _ { \theta } ( s _ { t } ^ { k , m } , p _ { k } )$ be the backend-native transition output. For Full, the base updater’s reward branch on this trajectory is weighted by $\widehat { A } _ { k , m } ^ { G }$ from Eq. 4; stabilization/reference terms remain in a separate branch:

$$
J _ { \mathrm { b a s e } , t } = J _ { \mathrm { r e w a r d } , t } + J _ { \mathrm { r e g } , t } , \qquad q _ { t } ^ { R } = \nabla _ { \mu _ { t } } J _ { \mathrm { r e w a r d } , t } , \quad g _ { t } ^ { R } = F _ { t } ^ { - 1 } q _ { t } ^ { R } .\tag{5}
$$

Here $F _ { t }$ is the backend’s local metric (the transition Fisher for Flow-GRPO), $q _ { t } ^ { R }$ is the reward covector, and $g _ { t } ^ { R }$ is its metric-mapped local ascent direction; we write $\| \boldsymbol { v } \| _ { F _ { t } } ^ { 2 } = \boldsymbol { v } ^ { \top } \bar { F _ { t } } \bar { \boldsymbol { v } }$ . Reference/KL regularization does not enter $q _ { t } ^ { R }$ and is restored unchanged after routing.

We cache the selected state rather than an alternative continuation. With the detached behavior snapshot $\theta ^ { - } = \mathrm { s g } ( \theta )$ , replacing only the executed prompt $p _ { k }$ by another current-policy candidate $p _ { j }$ gives

$$
\begin{array} { r l } & { d _ { j  k , t } = \mu _ { \theta ^ { - } } ( s _ { t } ^ { k , m } , p _ { j } ) - \mu _ { \theta ^ { - } } ( s _ { t } ^ { k , m } , p _ { k } ) , } \\ & { \bar { d } _ { j  k , t } = \mathrm { T r u s t C l i p } _ { F _ { t } } ( d _ { j  k , t } ) , \qquad c _ { j  k , t } = \bar { d } _ { j  k , t } ^ { \top } q _ { t } ^ { R } . } \end{array}\tag{6}
$$

The probe is intentionally one-transition and same-state: $s _ { t } ^ { k , m }$ is seed/state dependent, but within each comparison the state, stochastic draw, and behavior snapshot are fixed and only text conditioning changes. Continuing $p _ { j }$ would mix this local conditioning response with later nonlinear state evolution and additional stochasticity. We use Trust $\begin{array} { r } { \mathrm { C l i p } _ { F _ { t } } ( d ) = d \operatorname* { m i n } \Bigl ( 1 , \sqrt { { \tau _ { \mathrm { b a t c h } } } / { ( d ^ { \top } F _ { t } d + \epsilon _ { \mathrm { n u m } } ) } } \Bigr ) } \end{array}$ , where $\epsilon _ { \mathrm { n u m } } = 1 0 ^ { - 1 2 }$ and $\tau _ { \mathrm { b a t c h } }$ is the detached 95th percentile of the current-batch native-metric response energy $d ^ { \top } F _ { t } d$ . Thus in-region responses are unchanged and only unusually large conditioning shifts are rescaled. The score $c _ { j  k , t }$ is the resulting local directional derivative along an observed prompt response; its sign records agreement with the reward branch. Full-trajectory utility of prompt candidates is still measured by the $K \times M$ rollouts in Eq. 3.

Generator-aware prompt shaping. Let $\boldsymbol { S _ { k , m } }$ denote the selected routed transition indices on trajectory $( k , m )$ (one by default). Cross-fitting avoids scoring $p _ { j }$ on states produced by itself:

$$
C _ { j } = \frac { 1 } { K - 1 } \sum _ { k \neq j } \mathbb { E } _ { m , t \in \mathcal { S } _ { k , m } } [ c _ { j  k , t } ] , \qquad A _ { j } ^ { P } = \mathrm { C a l } _ { R } ( \widehat { A } _ { j } ^ { P , R } ) + \lambda _ { F } \mathrm { C a l } _ { C } ( C _ { j } ) .\tag{7}
$$

![](images/bfbe6720bf802b9f820a59f25d58454222fcc9740a9d0243a4f1775ae053a53d.jpg)  
Figure 2: FLOWCRED training overview. Hierarchical credit assignment samples K prompt candidates and reuses M matched generator seeds, so cross-prompt contrasts estimate prompt-level credit ${ \bar { Q } } - V$ while within-prompt residuals estimate $R - Q$ . A selected rollout supplies an encountered state $s _ { t }$ for functional coordination. At that fixed state, alternative-prompt responses define a local geometry used for generator-aware prompt shaping and reliability-gated routing of only the generator reward branch.

For $X \in \{ R , C \} , \operatorname { C a l } _ { X } ( x ) = ( x - \mu _ { X } ^ { \mathrm { c a l } } ) / ( \sigma _ { X } ^ { \mathrm { c a l } } + 1 0 ^ { - 6 } )$ uses statistics estimated once on a disjoint training-only calibration split and then frozen; Cal acts on terminal prompt credit and $\mathrm { C a l } _ { C }$ on functional evidence. The terminal term remains primary, while $C _ { j }$ supplies generator-aware auxiliary shaping; $\lambda _ { F }$ is fixed on a disjoint development split.

Reliability-gated generator routing. Let $D _ { t } = [ \bar { d } _ { j  k , t } ] _ { j \neq k }$ . The sampled current-policy response operator is a trace-normalized ridge map. To estimate whether this finite span generalizes beyond the $\bar { K } - 1$ rollout candidates, we sample $B _ { \mathrm { t r a i n } }$ additional intent-preserving prompt alternatives at the same state. Each requires only one same-state generator transition forward: no video is decoded and no reward model is called. With response $\widetilde { d } _ { b }$ , define

$$
\begin{array} { r l r l } & { P _ { \lambda , t } = D _ { t } ( D _ { t } ^ { \top } F _ { t } D _ { t } + \lambda _ { t } I ) ^ { - 1 } D _ { t } ^ { \top } F _ { t } , } & & { \lambda _ { t } = \epsilon _ { \mathrm { r i d g e } } \frac { \mathrm { t r } ( D _ { t } ^ { \top } F _ { t } D _ { t } ) } { K - 1 } , } \\ & { r _ { t } = \mathrm { c l i p } \left( \frac { 1 } { B _ { \mathrm { t r a i n } } } \sum _ { b } \frac { \| P _ { \lambda , t } \widetilde d _ { b } \| _ { F _ { t } } ^ { 2 } } { \| \widetilde d _ { b } \| _ { F _ { t } } ^ { 2 } + \epsilon _ { \mathrm { n u m } } } , 0 , 1 \right) , } & & { \alpha _ { t } = \alpha _ { \mathrm { m a x } } r _ { t } , } \\ & { g _ { t } ^ { G } = ( I - \alpha _ { t } P _ { \lambda , t } ) g _ { t } ^ { R } , } & & { q _ { t } ^ { G } = F _ { t } g _ { t } ^ { G } . } \end{array}\tag{8}
$$

Here $\epsilon _ { \mathrm { r i d g e } }$ is the fixed ridge scale, $\alpha _ { \mathrm { m a x } }$ caps the routing strength, and $q _ { t } ^ { G }$ is the routed reward covector backpropagated through the generator. The score $r _ { t }$ measures held-out response support rather than reward confidence. Low-support states continuously recover the base reward step. For stochastic-transition updaters such as Flow-GRPO, $g _ { t } ^ { R }$ is a stochastic estimate of the local reward gradient; the response operator and support gate are built from the cached state and detached prompt responses. Consequently the same linear routing applies to the conditional expected reward gradient, rather than implying that one high-dimensional transition sample itself has large prompt-supported energy. Appendix A.4 states this distinction explicitly. The same geometry therefore serves two targets: $C _ { j }$ shapes the upstream prompt action, whereas Eq. 8 regularizes the downstream expected reward update without assuming an exact transfer between optimizers.

## 3.4 Training procedure

One transition per decoded trajectory is routed by default, sampled uniformly from the valid stochastic policy steps used by the base reward loss; all other reward-loss transitions and the entire regularization/reference branch remain unchanged. Same-state quantities are stop-gradient targets from the detached behavior snapshot and are recomputed online each round. Default hyperparameters and backend-specific instantiations are in Appendices D and A.4.

Algorithm 1 One joint post-training round   
Require: request u; prompt policy π ; generator G ; K candidates; M matched seeds   
Hierarchical credit assignment   
1: sample $p _ { 1 : K }$ and decode $x _ { k , m } = G _ { \theta } ( p _ { k } ; z _ { m } )$ with shared $z _ { 1 : M }$   
2: evaluate $R _ { k , m } ;$ compute $\widehat { A } _ { k } ^ { \dot { P } , R }$ and $\widehat { A } _ { k , m } ^ { G }$   
Functional coordination at actual rollout states   
3: for each selected transition state $s _ { t } ^ { k , m }$ do   
4: obtain $( q _ { t } ^ { R } , g _ { t } ^ { R } )$ from the hierarchy-weighted base reward branch   
5: probe $d _ { j  k , t }$ for $j \neq k$ and accumulate cross-fitted evidence $C _ { j }$   
6: construct $P _ { \lambda , t \dot { \imath } }$ estimate $r _ { t }$ with $B _ { \mathrm { t r a i n } }$ non-decoded prompt probes   
7: route $g _ { t } ^ { G } \gets ( I - \alpha _ { \operatorname* { m a x } } r _ { t } P _ { \lambda , t } ) g _ { t } ^ { R }$   
8: end for   
9: update prompt policy with Eq. 7; update generator with the routed reward branch plus unchanged regulariza   
tion

## 4 Experiments

## 4.1 Protocol and controlled comparisons

Setup. Our primary system uses LTX-2.5 (HaCohen et al., 2026; Lightricks, 2026), a Qwen3.5-9B prompt optimizer, Flow-GRPO, and $K = M = 4$ . Controlled variants share initialization, request stream, prompt sampling, matched seeds, reward, decoded-video budget, and optimization schedule; functional coordination adds same-state transition forwards but no extra video decode. The frozen training reward uses public, fixed scorers with weights 0.35/0.30/0.25/0.10: the Text-Alignment head of VideoReward (Liu et al., 2025b) for request faithfulness, frame-averaged VQAScore (CLIP-FlanT5-XXL) (Lin et al., 2024) for compositional correspondence, the temporal-consistency/textalignment aggregate of VideoScore-v1.1 (He et al., 2024) for event realization, and the visual-/motionquality aggregate of VideoReward as a bounded quality stabilizer. All scorers and calibration statistics are frozen before controlled training. Appendices D and D.3 specify checkpoints, frame sampling, calibration, and scorer-family robustness.

Evaluation and comparisons. We report VBench (Huang et al., 2024), VBench-2.0 (Zheng et al., 2025), T2V-CompBench (Sun et al., 2025), and StoryEval (Wang et al., 2025) on the original request u with matched evaluation seeds. The hierarchical-credit-versus-Full comparison and the functional-coordination factorial use three independent training runs; other trainable baselines use two. The controlled chain spans prompt-only adaptation, a raw-request Flow-GRPO generator-only reference, PromptRL, UniGRPO, Matched Global-LOO, hierarchical credit, and Full. For methods originally reported with different backbones or training stacks, we implement the published objective in the common Qwen3.5/LTX-2.5 harness; the table therefore reports controlled common-harness results rather than copied published absolute scores. CAPE-T2V is positioned separately because its public LTX recipe uses sequential two-sided adaptation with additional video-derived caption supervision and a different released LTX backbone; Appendix D.5 gives the precise comparison scope. Uncertainty and data-integrity protocols are summarized in Appendices D.4, G, and J.

## 4.2 Main benchmark performance

Table 1 compares single-side and joint alternatives under one LTX-2.5 harness. Because prompt-side adaptation already closes substantial interface-level headroom, the matched joint chain is the key test of hierarchical credit and functional coordination.

Table 1: Main LTX-2.5 comparison under a common evaluation harness. Mean scores use matched evaluation seeds; training-run uncertainty is reported in Appendix J. Best is bold and second-best is underlined.
<table><tr><td>Method</td><td>VBench↑</td><td>VBench-2.0↑</td><td>T2V-CompBench↑</td><td>StoryEval↑</td></tr><tr><td colspan="5">FROZEN / INTERFACE REFERENCES</td></tr><tr><td>Raw prompt</td><td>78.14</td><td>54.52</td><td>46.52</td><td>37.88</td></tr><tr><td>Official LX-2.5 prompt enhancer</td><td>84.92</td><td>61.36</td><td>57.48</td><td>51.63</td></tr><tr><td colspan="5">SINGLE-SIDE ADAPTATION</td></tr><tr><td>Prompt-A-Video</td><td>85.63</td><td>62.18</td><td>58.87</td><td>52.96</td></tr><tr><td>VPO</td><td>85.44</td><td>64.06</td><td>59.54</td><td>53.61</td></tr><tr><td>Prompt-only</td><td>86.97</td><td>62.84</td><td>60.28</td><td>54.37</td></tr><tr><td>Flow-GRPO</td><td>80.83</td><td>58.57</td><td>52.31</td><td>45.26</td></tr><tr><td colspan="5">JOINT POST-TRAINING — MATCHED DECODED-ROLLOUT BUDGET</td></tr><tr><td>PromptRL</td><td>86.31</td><td>63.02</td><td>61.06</td><td>55.02</td></tr><tr><td>UniGRPO</td><td>86.40</td><td>63.28</td><td>61.32</td><td>55.40</td></tr><tr><td>Matched Global-LOO</td><td>86.49</td><td>63.41</td><td>61.47</td><td>55.58</td></tr><tr><td>FLOwCRED (hierarchical credit)</td><td>86.91</td><td>63.92</td><td>62.63</td><td>56.71</td></tr><tr><td>FLOWCRED (Full)</td><td>87.56</td><td>64.73</td><td>64.46</td><td>58.08</td></tr></table>

Prompt-side adaptation already closes substantial interface-level headroom, making the matched joint chain the stricter test. In particular, Prompt-only is already competitive on VBench, whereas the value of joint adaptation is more visible on compositional and multi-event metrics: Full exceeds Prompt-only by +4.18 T2V-CompBench and +3.71 StoryEval points. Within the matched joint chain, PromptRL and UniGRPO remain below Matched Global-LOO; hierarchical credit then improves over Global-LOO by +0.42/+0.51/+1.16/+1.13 on VBench/VBench-2.0/T2V-CompBench/StoryEval, and Full adds another +0.65/+0.81/+1.83/+1.37. The hierarchical-credit-to-Full increment is positive in all three independent training runs on every headline benchmark. Appendix J reports run-level deltas and intervals, and Table 8 gives the aggregate breakdown.

## 4.3 Hierarchical-credit diagnostics

The two estimator choices are directly testable at fixed decoded-video budget. Conditioning the generator baseline on the executed prompt removes prompt-level nuisance variation from the downstream advantage, while matched seeds pair prompt candidates under common generator randomness. Table 2 reports held-out diagnostics; the corresponding distributions and protocol are in Appendix E. Together, these diagnostics support the intended separation of prompt-level and prompt-conditioned variation while keeping the decoded-rollout budget fixed.

Table 2: Hierarchical-credit diagnostics at matched rollout budget. Lower variance is better; the final row keeps the decoded-video count fixed.
<table><tr><td>Diagnostic</td><td>Reference</td><td>Hierarchical design</td><td>Result</td></tr><tr><td>Generator-advantage variance</td><td>Global LOO</td><td>Prompt-conditioned LOO</td><td>-31%</td></tr><tr><td>Prompt-credit variance</td><td>Independent seeds</td><td>Matched seeds</td><td>-27%</td></tr><tr><td>T2V-CompBench</td><td>Independent seeds</td><td>Matched seeds</td><td>62.21 → 62.63</td></tr></table>

## 4.4 Functional-coordination attribution and mechanism diagnostics

The two uses of the same-state geometry contribute independently, and their combination yields the strongest result. Table 3 reports the three-run mean (SD) endpoint scores, while Fig. 3 summarizes the complementary endpoint-attribution and realized-update audits. Relative to hierarchical credit, prompt shaping alone improves VBench/StoryEval/T2V-CompBench by +0.29/+0.65/+0.77, generator routing alone by +0.38/+0.47/+1.04, and Full by +0.65/+1.37/+1.83.

(a) Component attribution  
Table 3: Functional-coordination factorial ablation. Mean (SD) across the same three-run protocol used for the HC–Full comparison.
<table><tr><td>Prompt shaping</td><td>Generator routing</td><td>VBench↑</td><td>StoryEval↑</td><td>T2V-CompBench↑</td></tr><tr><td></td><td>一</td><td>86.91 (0.14)</td><td>56.71 (0.27)</td><td>62.63 (0.32)</td></tr><tr><td>V</td><td></td><td>87.20 (0.15)</td><td>57.36 (0.28)</td><td>63.40 (0.34)</td></tr><tr><td></td><td>√</td><td>87.29 (0.13)</td><td>57.18 (0.26)</td><td>63.67 (0.31)</td></tr><tr><td>√</td><td>√</td><td>87.56 (0.12)</td><td>58.08 (0.24)</td><td>64.46 (0.29)</td></tr></table>

![](images/39d9d64fcfb0871b2c53836ebd3e9070f7ddec5c6b396c6699444a8a848820e0.jpg)  
(a) Component attribution.

(b) Realized coordination
<table><tr><td></td><td>HC</td><td>Full</td><td>Change</td></tr><tr><td>Prompt alignment</td><td>0.29 O</td><td>0.61</td><td>+0.32</td></tr><tr><td>Supported overlap</td><td>0.487 O</td><td>0.264</td><td>-0.223</td></tr><tr><td>Update norm</td><td>1.000 O</td><td>0.931</td><td>-6.9%</td></tr></table>

(b) Realized coordination.  
Figure 3: Functional-coordination endpoint and mechanism evidence. (a) Relative to hierarchical credit, prompt shaping and generator routing each improve endpoint performance, and their combination is strongest on all three metrics. (b) In a matched cached-state audit, Full increases prompt-side alignment, reduces supported overlap, and retains most of the overall generator update norm.

Realized coordination. The factorial result is complemented by a matched one-step audit on cached states. Rather than interpreting the projection energy of an individual stochastic policy-gradient sample, we measure the neural displacement after the optimizer step. Full reduces realized generator supported overlap from 0.487 to 0.264 while retaining 0.931× the hierarchical-credit update norm; on the prompt side, the native-metric alignment cosine rises from 0.29 to 0.61. Shaping-only and routing-only variants isolate the corresponding upstream and downstream changes (Appendix K). Together with the endpoint factorial, these measurements support selective coordination rather than a generic shrinkage explanation, while remaining descriptive checks of trained behavior rather than an exact transfer of causal responsibility.

## 4.5 Prompt-policy signal validity and routing controls

On 360 held-out requests, reward-only candidate ranking reaches $\rho = 0 . 4 4 7 / 5 9 . 2 \%$ Top-1 on eight disjoint fresh seeds, cosine functional evidence reaches $\rho = 0 . 5 6 9 / 6 7 . 2 \%$ , and the directional functional score used by FLOWCRED reaches $\rho = 0 . 6 1 2 / 7 1 . 7 \%$ . This provides an out-of-seed check that same-state generator information carries signal beyond terminal reward ranking alone.

Matched-budget controls test simpler routing explanations. Rank-matched random spans and wrongquery spans stay near the hierarchical-credit reference; uniform shrink yields only a small gain, while removing cross-fitting or the reliability gate reduces the Full improvement. Figure 4 summarizes both checks; complete control values are reported in Appendix H.

![](images/561960b6c6476c96a8876719446e584e6492fa5161ab63a23df11643c8ad5d83.jpg)  
(a) Fresh-seed predictive utility.

![](images/30d7c7eaea83b2b9679fe3cfe715457ff53a958abf4fcf70ddf85f9fe4df4e20.jpg)  
(b) Compact matched-budget routing controls.  
Figure 4: Functional signal validity and routing controls. Directional evidence improves fresh-seed candidat ranking beyond reward-only scoring, while compact routing controls test mismatched geometry and safeguard removal.

## 4.6 Generalization and external validation

Transfer evidence. We transfer the functional-coordination increment to HunyuanVideo-1.5 and to RVM and Diffusion-DRF generator updates. Table 4 reports matched Full-minus-hierarchical-credit deltas rather than heterogeneous absolute rankings.

Table 4: Transfer of functional coordination. Full minus matched hierarchical credit under each backbone/updater.
<table><tr><td>Setting</td><td>∆VBench</td><td>∆T2V-CompBench</td><td>∆StoryEval</td></tr><tr><td>LTX-2.5 / Flow-GRPO</td><td>+0.65</td><td>+1.83</td><td>+1.37</td></tr><tr><td>HunyuanVideo-1.5 / Flow-GRPO</td><td>+0.57</td><td>+1.22</td><td>+1.09</td></tr><tr><td>LTX-2.5 / RVM</td><td>+0.52</td><td>+1.29</td><td>+1.15</td></tr><tr><td>LTX-2.5 / Diffusion-DRF</td><td>+0.41</td><td>+0.97</td><td>+0.79</td></tr></table>

The increment remains positive in every tested backbone/updater setting, although its magnitude varies, indicating that functional coordination is not tied to a particular generator backbone or reward-update rule.

Full keeps the decoded-video and reward-model-call budget identical to hierarchical credit; its additional generator work consists of non-decoded same-state transition probes, counted explicitly in Appendix I. Frozen-module swaps attribute measurable gains to both sides: substituting the Full generator into the hierarchical-credit pair adds +0.90 T2V-CompBench and substituting the Full prompt policy adds +0.72. In a 240-request blinded evaluation, Full is preferred over hierarchical credit on 134 versus 48 requests (58 ties; 73.6% of non-ties) and over PromptRL on 142 versus 46 (52 ties; 75.5% of non-ties). An independent VisionReward evaluator, never used in training or model selection, also prefers Full to hierarchical credit on 63.3% of 300 held-out matched pairs. Detailed annotation and replication protocols are in Appendices J and L.

Taken together, these checks test complementary explanations of the primary gain. Transfer across a second backbone and alternative updater families probes dependence on the default generator stack; frozen-module swaps verify contributions from both trained modules; and blinded human judgments plus the held-out VisionReward evaluator test whether the improvement is confined to the training reward.

External two-sided positioning. PromptRL and UniGRPO use their published joint-training objectives in the common LTX-2.5 harness. CAPE-T2V is reported only as an external two-sided reference because its public pipeline uses a different released LTX backbone and auxiliary video-derived captions. Appendix D.5 details the comparison scope; Appendix J reports rollout-budget and paired-run checks.

## 4.7 Qualitative coordination cases

Figure 5 shows two representative cases with complementary demands. In the structured repair scene, Base and VPO capture the broad setting but do not make the requested spacecraft-repair interaction equally explicit, whereas FLOWCRED preserves both the scene structure and the target interaction. In the directional-motion case, the generated sequence shows clearer execution of the requested right-to-left motion for FLOWCRED, while the frame strip highlights its subject consistency; the alternatives show weaker directional execution or greater subject ambiguity. Appendix L provides four additional cases spanning counting, interaction, spatial relation, and synchronized appearance change.

(a)  
“Astronauts repair spacecraft, floating weightlessly in vastness of space.”  
![](images/7024768c1674640010946b837af0966e6e28e240bb0d390588eefc82a1dde089.jpg)

(b)  
“A fish swims from right to left in the sea.”  
![](images/fa56502ec3485547e478618a63988751844d1e275d9572d462e297b0bd542d2b.jpg)  
Figure 5: Representative qualitative comparison. Base, VPO, and FLOWCRED on structured scene realization and directional motion. Additional qualitative comparisons are provided in the appendix.

## 5 Conclusion

We presented FLOWCRED, a framework for jointly post-training a prompt optimizer and a text-tovideo generator without treating a shared terminal reward as undifferentiated supervision. Hierarchical credit assignment uses matched prompt–seed rollouts to separate prompt choice from promptfixed stochasticity, while functional coordination uses same-state response geometry for generatoraware prompt shaping and reliability-gated routing of the generator reward branch. Controlled comparisons show that the gains cannot be explained by stronger prompts, additional decoded rollouts, or generic update shrinkage alone, and evidence from repeated runs, mechanism analyses, human evaluation, and transfer studies is consistent with the intended division of labor between prompt-side conditioning and residual generator execution. More broadly, FLOWCRED suggests that joint post-training should distinguish where outcome variation arises from where trainable modules can realize overlappingfunctional changes: sharing a reward is not sufficient by itself; credit assignment and update coordination are separate design problems in cascaded generative systems.

## AI use statement

Generative AI tools were used to assist with conceptual and experimental design, literature search, code development, and manuscript drafting and editing. All AI-assisted material was reviewed by the authors, who are responsible for the final submission.

## Reproducibility statement

Section 3 specifies the FLOWCRED formulation and training procedure; Appendices A–L provide derivations, implementation details, reward/calibration and baseline protocols, evaluation, compute, robustness, human evaluation, and qualitative results.

## References

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet´ Ust<sup>¨</sup> un,¨ and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume

1: Long Papers), pp. 12248–12267. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024. acl-long.662.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024.

Tingfeng Cao, Chengyu Wang, Bingyan Liu, Ziheng Wu, Jinhui Zhu, and Jun Huang. BeautifulPrompt: Towards automatic prompt engineering for text-to-image synthesis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1–11. Association for Computational Linguistics, 2023.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7310–7320, 2024.

Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, Hongning Wang, and Minlie Huang. VPO: Aligning text-to-video generation models with prompt optimization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 15636–15645, 2025.

Junhao Cheng, Liang Hou, Xin Tao, and Jing Liao. Video-as-answer: Predict and generate next video event with Joint-GRPO. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 38915–38925, 2026.

Jaemoo Choi, Wei Guo, Yuchen Zhu, Arash Vahdat, Molei Tao, Julius Berner, and Yongxin Chen. Scaling reinforcement learning for diffusion models via velocity matching. arXiv preprint arXiv:2608.23664, 2026.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, 2024.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, 2023.

Jakob N. Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, pp. 2974–2982, 2018. doi: 10.1609/aaai.v32i1.11794.

Yuanshen Guan, Zipeng Feng, Chengru Song, Zhiwei Xiong, and Peiqin Sun. Latent reward registers for diffusion preference alignment. arXiv preprint arXiv:2608.03929, 2026.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. LTX-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Bill Yuchen Lin, and Wenhu Chen. VideoScore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2105–2123. Association for Computational Linguistics, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian Ge, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, Weilin Huang, and Ping Luo. Prompt-A-Video: Prompt your video diffusion model via preference-aligned llm. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18725–18735, 2025.

Yizhuo Jia, Jingyun Hua, and Yuanxing Zhang. CAPE-T2V: Captioner-anchored prompt enhancement toward two-sided conditioning alignment in text-to-video generation. arXiv preprint arXiv:2608.03046, 2026.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, and Kazuki Kozuka. Aligning diffusion models by optimizing human utility. In Advances in Neural Information Processing Systems, volume 37,

2024.

Lightricks. LTX-2.5: Video, audio & world simulation. Hugging Face model card, 2026. URL https: //huggingface.co/Lightricks/LTX-2.5. Accessed September 2026.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In Computer Vision – ECCV 2024, pp. 366–384, 2024. doi: 10.1007/978-3-031-72673-6 20.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, volume 34, 2021.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online rl. In Advances in Neural Information Processing Systems, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, and Wanli Ouyang. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025b.

Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, and Wanli Ouyang. UniGRPO: Unified policy optimization for reasoning-driven visual generation. arXiv preprint arXiv:2603.23500, 2026.

Yifu Luo, Haoyuan Sun, Xinhao Hu, Penghui Du, Keyu Fan, Bo Li, Sinan Du, Xu Wan, Zhiyu Chen, Bo Xia, Yongzhe Chang, Changqian Yu, Kun Gai, Tiantian Zhang, and Xueqian Wang. Principled rl for flow matching emerges from the chunk-level policy optimization. In International Conference on Machine Learning, 2026. arXiv:2510.21583.

Andrew Y. Ng and Michael I. Jordan. Pegasus: A policy search method for large mdps and pomdps. In Proceedings ofthe Sixteenth Conference on Uncertainty in Artificial Intelligence, pp. 406–415, 2000.

Daisuke Oba, Hiroki Furuta, and Naoaki Okazaki. Diffusion-state policy optimization for masked diffusion language models. arXiv preprint arXiv:2602.06462, 2026.

Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

Yujiang Pu and Yu Kong. Noiseasier: Test-time noise optimization for text-to-video generation. In European Conference on Computer Vision, 2026. arXiv:2608.30194.

Qwen Team. Qwen3.5-9B. Hugging Face model card, 2026. URL https://huggingface.co/Qwen/ Qwen3.5-9B. Accessed September 2026.

Yash Savani, Branislav Kveton, Yuchen Liu, Yilin Wang, Jing Shi, Subhojyoti Mukherjee, Nikos Vlassis, and Krishna Kumar Singh. Stepwise credit assignment for grpo on flow-matching models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 42007–42017, 2026.

Jinjie Shen, Wei Deng, Xian Hu, Daiguo Zhou, and Jian Luan. STAR: Spatiotemporal adaptive reward allocation for text-to-image rl post-training. arXiv preprint arXiv:2606.17979, 2026.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2V-CompBench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8406–8416, 2025.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Fu-Yun Wang, Han Zhang, Michael Gharbi, Hongsheng Li, and Taesung Park. PromptRL: Prompt matters in rl for flow-based image generation. arXiv preprint arXiv:2602.01382, 2026a.

Yifan Wang, Yanyu Li, Gordon Guocheng Qian, Sergey Tulyakov, Yun Fu, and Anil Kag. Diffusion-DRF: Free, rich, and differentiable reward for video diffusion fine-tuning. arXiv preprint arXiv:2601.04153, 2026b.

Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Is your world simulator a good story presenter? a consecutive events-based benchmark for future long video generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13629–13638, 2025.

Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, et al. HunyuanVideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, 2023.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, Jiayan Teng, Zhuoyi Yang, Wendi Zheng, Xiao Liu, Dan Zhang, Ming Ding, Xiaohan Zhang, Shiyu Huang, Xiaotao Gu, Minlie Huang, Jie Tang, and Yuxiao Dong. VisionReward: Fine-grained multi-dimensional human preference learning for image and video generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 11269–11277, 2026.

Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. In International Conference on Machine Learning, 2026. arXiv:2509.25050.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. DanceGRPO: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025.

Xi Ye, Wenjia Yang, Yangyang Xu, Xiaoyang Liu, Duo Su, Mengfei Xia, and Jun Zhu. SHIFT: Motion alignment in video diffusion models with adversarial hybrid fine-tuning. In European Conference on Computer Vision, 2026.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, 2020.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Jingyuan Zhu, Biaolong Chen, Le Zhang, Aixi Zhang, Hao Jiang, and Pipei Huang. Diffusion-APO: Trajectoryaware direct preference alignment for video diffusion transformers. arXiv preprint arXiv:2605.07503, 2026.

## A Theoretical Foundations

This appendix collects the mathematical statements used in the main paper. The reward hierarchy is exact for any per-trajectory scalar terminal score and yields valid score-function baselines. In the final protocol, reward-component and prompt-term affine scales are estimated once on disjoint training-only calibration data and then frozen, so current minibatch composition does not enter the score used by the hierarchy (Appendix D.3). Same-state prompt intervention provides a first-order directional derivative of the generator reward surrogate. Minimum-overlap routing is a local function space regularizer; its non-reversal statement is made in the backend’s local output coordinates before neural pullback.

## A.1 Hierarchy-aware statistical credit

Recall

$$
J ( \phi , \theta ) = \mathbb { E } _ { u \sim \mathcal { D } } \mathbb { E } _ { p \sim \pi _ { \phi } ( \cdot | u ) } \mathbb { E } _ { \mathcal { T } \sim \pi _ { \theta } ( \cdot | p ) } [ R ( u , x ) ] ,\tag{9}
$$

with

$$
\begin{array} { r } { Q _ { \theta } ( u , p ) = \mathbb { E } _ { \mathcal { T } \sim \pi _ { \theta } ( \cdot | p ) } [ R ( u , x ) ] , \qquad V _ { \phi , \theta } ( u ) = \mathbb { E } _ { p \sim \pi _ { \phi } ( \cdot | u ) } Q _ { \theta } ( u , p ) . } \end{array}\tag{10}
$$

Proposition 1 (Hierarchy-aware score-function baselines). For the factorized policy $\pi _ { \phi } ( p | u ) \pi _ { \theta } ( T | p )$

$$
\nabla _ { \phi } J = \mathbb { E } [ ( Q _ { \theta } ( u , p ) - V _ { \phi , \theta } ( u ) ) \nabla _ { \phi } \log \pi _ { \phi } ( p | u ) ] ,\tag{11}
$$

$$
\nabla _ { \boldsymbol { \theta } } J = \mathbb { E } [ ( R ( \boldsymbol { u } , \boldsymbol { x } ) - \mathrm { s g } [ Q _ { \boldsymbol { \theta } } ( \boldsymbol { u } , p ) ] ) \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \mathcal { T } | p ) ] .\tag{12}
$$

Proof. For the prompt policy, apply the score-function identity and subtract $V ( u )$ , which does not depend on the sampled prompt action. For the generator, any prompt-dependent quantity that is fixed with respect to the sampled downstream trajectory is a valid score-function baseline; choosing $Q ( u , p )$ gives the second expression. The stop-gradient indicates that the statement concerns a policy-gradient baseline rather than differentiation through the value estimate. □

The pointwise identity $R - V = ( Q - V ) + ( R - Q )$ follows immediately. It is a statistical decomposition of reward variation. Functional coordination in Section 3.3 then shapes the joint objective using same-state response evidence, so the two levels have distinct mathematical status.

## A.2 Relation to the matched global-LOO control

The matched control uses the same sampled actions and matched rollout grid as hierarchical credit, but both modules receive a global leave-one-out advantage

$$
A _ { k , m } ^ { \mathrm { G L O O } } = R _ { k , m } - { \frac { 1 } { K M - 1 } } \sum _ { ( j , n ) \neq ( k , m ) } R _ { j , n } .\tag{13}
$$

For a fixed prompt action $p _ { k }$ , averaging over its M trajectories yields a positive affine rescaling of $Q _ { k } - \bar { Q }$ and therefore preserves the coarse prompt ordering. The hierarchical-credit prompt-policy estimator in Eq. 4 is likewise a positive rescaling of $Q _ { k } - \bar { Q }$ . We consequently do not attribute the hierarchical-credit gain to a new prompt ordering. The controlled change is the downstream generator baseline: Global-LOO removes only the current trajectory, whereas hierarchical credit removes prompt-level variation as well by comparing the trajectory against other seeds under the same executed prompt. The global-LOO construction also eliminates the self-inclusion confound that would otherwise accompany a global mean baseline.

## A.3 Variance reduction from matched seeds

For candidates $p _ { k } , p _ { j }$ , write $R _ { k } ( z )$ and $R _ { j } ( z )$ for rewards obtained with common seed $z .$ Define

$$
\widehat { \Delta } _ { k j } ^ { \mathrm { m a t c h } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } [ R _ { k } ( z _ { m } ) - R _ { j } ( z _ { m } ) ] .\tag{14}
$$

For an independently seeded comparison, use $z _ { m }$ for k and independent $z _ { m } ^ { \prime }$ for j. Let $\sigma _ { k } ^ { 2 } = \mathrm { V a r } ( R _ { k } )$ $\sigma _ { j } ^ { 2 } = \mathrm { V a r } ( \mathbf { \dot { \boldsymbol { R } } } _ { j } )$ , and $\sigma _ { k j } = \mathrm { C o v } ( \mathsf { \bar { R } } _ { k } ( z ) , R _ { j } ( z ) )$

Proposition 2 (Matched-seed variance). Assume independent seed pairs across m andfinite second moments. Then

$$
\mathrm { V a r } ( \widehat { \Delta } _ { k j } ^ { \mathrm { m a t c h } } ) = \frac { \sigma _ { k } ^ { 2 } + \sigma _ { j } ^ { 2 } - 2 \sigma _ { k j } } { M } ,\tag{15}
$$

$$
\mathrm { V a r } ( \widehat { \Delta } _ { k j } ^ { \mathrm { i n d } } ) = \frac { \sigma _ { k } ^ { 2 } + \sigma _ { j } ^ { 2 } } { M } .\tag{16}
$$

Hence matched seeds reduce variance by $2 \sigma _ { k j } / M$ whenever $\sigma _ { k j } \geq 0 .$

Proof. Apply the variance identity to $R _ { k } ( z ) - R _ { j } ( z )$ and use independence across $m .$ . The independent estimator has zero cross-covariance by construction. □

The prompt-policy estimator in Eq. 4 averages these seed-wise contrasts over the other $K - 1$ candidates, so common randomness enters the actual training estimator rather than only an auxiliary analysis.

## A.4 Reward-only transition geometry

For a Gaussian transition π $\cdot ( a | \mu ) = \mathcal { N } ( \mu , \Sigma )$ with covariance independent of $\mu _ { : }$

$$
\nabla _ { \mu } \log \pi ( a | \mu ) = \Sigma ^ { - 1 } ( a - \mu ) , \qquad F = \mathbb { E } [ \nabla _ { \mu } \log \pi \nabla _ { \mu } \log \pi ^ { \top } ] = \Sigma ^ { - 1 } .\tag{17}
$$

If an unclipped reward surrogate has local coefficient ${ \widehat { A } } ,$ , then $q ^ { R } = { \widehat { A } } \Sigma ^ { - 1 } ( a - \mu )$ and $g ^ { R } = { }$ $F ^ { - 1 } q ^ { R } = \widehat { A } ( a - \mu )$ . This provides intuition for the Fisher coordinate used in the main text. The implementation differentiates the actual reward branch of Flow-GRPO after its ratio and clipping operations. Reference/KL terms stay inside $J _ { \mathrm { r e g } }$ and never enter $q ^ { R }$ , preventing reward credit from being contaminated by a return-to-reference direction.

Stochastic sample versus expected reward direction. For Flow-GRPO, the single-transition vector above is a stochastic score-function sample, so its high-dimensional direction need not itself align strongly with a low-rank prompt-response span. Let $P _ { t }$ and $\alpha _ { t }$ be constructed from the cached pre-action state and detached prompt responses, hence fixed with respect to the current transition draw conditional on that state. Linearity gives

$$
{ \mathbb E } [ ( I - \alpha _ { t } P _ { t } ) g _ { t } ^ { R } \mid s _ { t } ] = ( I - \alpha _ { t } P _ { t } ) { \mathbb E } [ g _ { t } ^ { R } \mid s _ { t } ] .\tag{18}
$$

Thus routing is interpreted as a regularizer on the conditional expected reward update. We do not use the projection energy of one stochastic transition sample as a semantic-overlap statistic; the mechanism audits below instead use realized neural displacements after the optimizer step.

## A.5 Directional prompt-policy evidence

Fix a behavior state $s _ { t }$ , executed action $p _ { k }$ , and differentiable local reward surrogate $J _ { \mathrm { r e w a r d , t } } ( \mu )$ Let $\mu _ { k } = \mu ( s _ { t } , p _ { k } ) , q _ { t } ^ { R } = \nabla _ { \mu } J _ { \mathrm { r e w a r d , t } } ( \mu _ { k } )$ , and $d = \mu ( s _ { t } , p _ { j } ) - \mu _ { k }$ . Taylor’s theorem gives

$$
J _ { \mathrm { r e w a r d , t } } ( \mu _ { k } + \varepsilon d ) = J _ { \mathrm { r e w a r d , t } } ( \mu _ { k } ) + \varepsilon ( q _ { t } ^ { R } ) ^ { \top } d + \frac { \varepsilon ^ { 2 } } { 2 } d ^ { \top } H ( \xi ) d .\tag{19}
$$

Therefore $c = d ^ { \top } q _ { t } ^ { R }$ is exactly the directional derivative at $\varepsilon = 0$ . It characterizes the local reward change along an observed same-state conditioning response; fresh-seed evaluation measures how far that local signal transfers to full trajectories. The held-out-response and fresh-seed experiments in Appendix F test whether this local quantity carries predictive information beyond the construction states.

## A.6 Minimum-overlap routing in Fisher coordinates

For positive-definite F, define $\langle a , b \rangle _ { F } = a ^ { \top } F b$ and $\| \boldsymbol { a } \| _ { F } ^ { 2 } = \boldsymbol { a } ^ { \top } F \boldsymbol { a }$ . Let P be the F-orthogonal projector onto the response span,

$$
P = D ( D ^ { \top } F D ) ^ { \dagger } D ^ { \top } F .\tag{20}
$$

Then $P ^ { 2 } = P$ and $P ^ { \top } F = F P$

Proposition 3 (Minimum-overlap solution). For $\eta \geq 0$ , the local problem

$$
\operatorname* { m i n } _ { \delta } \frac { 1 } { 2 } \| \delta - g ^ { R } \| _ { F } ^ { 2 } + \frac { \eta } { 2 } \| P \delta \| _ { F } ^ { 2 }\tag{21}
$$

has solution

$$
\delta ^ { * } = ( I - \alpha P ) g ^ { R } , \qquad \alpha = \frac { \eta } { 1 + \eta } ,\tag{22}
$$

and

$$
\begin{array} { r } { ( q ^ { R } ) ^ { \top } \delta ^ { * } = \| ( I - P ) g ^ { R } \| _ { F } ^ { 2 } + ( 1 - \alpha ) \| P g ^ { R } \| _ { F } ^ { 2 } \geq ( 1 - \alpha ) \| g ^ { R } \| _ { F } ^ { 2 } \geq 0 . } \end{array}\tag{23}
$$

Proof. The first-order condition is $F ( \delta - g ^ { R } ) + \eta P ^ { \top } F P \delta = 0$ . Using $P ^ { \top } F = F P$ and $P ^ { 2 } = P$ yields $( I + \eta P ) \delta = g ^ { R }$ . Its inverse is $I - \dot { \eta } ( 1 + \eta ) ^ { - 1 } P$ . The final expression follows from the F-orthogonal decomposition $g ^ { R } = P g ^ { R } + ( \ddot { I } - P ) \ddot { g } ^ { R }$ □

The statement is made in an unconstrained transition-mean coordinate. The neural update in Eq. 8 is a pullback through $J _ { \mu , \theta } ;$ an arbitrary mean-space displacement need not be exactly realizable by the parameterization. We therefore use the proposition to motivate the local routing rule and test the realized parameter update directly through the audit in Appendix F.1.

## A.7 Ridge stabilization and contraction

The implemented operator in Eq. 8 uses ridge regularization. Let $B = F ^ { 1 / 2 } D$ . In whitened coordinates,

$$
\widetilde { P } _ { \lambda } = B ( B ^ { \top } B + \lambda _ { t } I ) ^ { - 1 } B ^ { \top } , \qquad \lambda _ { t } = \epsilon _ { \mathrm { r i d g e } } \frac { \mathrm { t r } ( D _ { t } ^ { \top } F _ { t } D _ { t } ) } { K - 1 } , \quad \epsilon _ { \mathrm { r i d g e } } = 1 0 ^ { - 3 } .\tag{24}
$$

If $B = U \mathrm { d i a g } ( s _ { i } ) V ^ { \top }$ , then

$$
\widetilde { P } _ { \lambda } = U \mathrm { d i a g } \bigg ( \frac { s _ { i } ^ { 2 } } { s _ { i } ^ { 2 } + \lambda _ { t } } \bigg ) U ^ { \top } .\tag{25}
$$

Hence $\widetilde { P } _ { \lambda }$ is symmetric positive semidefinite with eigenvalues in $[ 0 , 1 )$ . For $0 \leq \alpha \leq 1$ and $\widetilde { g } = F ^ { 1 / 2 } g ^ { R } ,$

$$
\begin{array} { r } { ( q ^ { R } ) ^ { \top } ( I - \alpha P _ { \lambda } ) g ^ { R } = \widetilde { g } ^ { \top } ( I - \alpha \widetilde { P } _ { \lambda } ) \widetilde { g } \geq ( 1 - \alpha ) \| \widetilde { g } \| _ { 2 } ^ { 2 } \geq 0 . } \end{array}\tag{26}
$$

This covers the stabilized local operator used in training without calling it an exact projector.

## A.8 Reference action and finite response span

The conditioning reference is the executed prompt action $p _ { k }$ because it is the upstream action that produced $s _ { t } .$ . The intervention of interest is therefore $p _ { k } \to p _ { j }$ . Using the raw request u as the reference mixes candidate-specific variation with the shared raw-to-enhanced displacement and can overstate directions common to all enhanced prompts; we keep that choice as an ablation.

With $K = 4$ , the main response matrix has at most three columns, so the geometry is a finite, policydependent response span. It measures overlap with alternatives actually proposed by the current prompt policy. Rank-matched random spans, wrong-query spans, held-out response prediction, and generator swaps are designed around that finite claim.

## B Reliability, Trust Regions, and Detachment

For selected states, the prompt policy samples $B _ { \mathrm { t r a i n } } ~ = ~ 6$ additional intent-preserving textual alternatives $\widetilde { p } _ { b }$ that are not rendered into videos. The main span is always constructed from the $K - 1$ real rollout candidates; these non-decoded probes require only one same-state generator transition forward and no reward-model call. For $\widetilde { d } _ { b } = \mu ( s _ { t } , \widetilde { p } _ { b } ) - \mu ( s _ { t } , p _ { k } )$ , define

$$
e _ { b , t } = \frac { \lVert P _ { \lambda , t } \widetilde { d } _ { b } \rVert _ { F _ { t } } ^ { 2 } } { \lVert \widetilde { d } _ { b } \rVert _ { F _ { t } } ^ { 2 } + \epsilon _ { \mathrm { n u m } } } , \qquad r _ { t } = \mathrm { c l i p } \left( \frac { 1 } { B _ { \mathrm { t r a i n } } } \sum _ { b = 1 } ^ { B _ { \mathrm { t r a i n } } } e _ { b , t } , 0 , 1 \right) , \qquad \alpha _ { t } = \alpha _ { \mathrm { m a x } } r _ { t } ,\tag{27}
$$

with $\epsilon _ { \mathrm { { n u m } } } = 1 0 ^ { - 1 2 }$ . The score measures support/coverage of the sampled response span, not correctness of the reward direction. The default uses $\alpha _ { \mathrm { m a x } } = 0 . 7 5$ . Evaluation-time mechanism analysis independently samples $B _ { \mathrm { e v a l } } = 1 2$ probes on held-out requests; these probes never participate in the training gate.

For a response d, the trust operator is

$$
\mathrm { T r u s t C l i p } _ { F _ { t } } ( d ) = d \mathrm { m i n } \Bigg ( 1 , \sqrt { \frac { \tau _ { \mathrm { b a t c h } } } { d ^ { \top } F _ { t } d + \epsilon _ { \mathrm { n u m } } } } \Bigg ) ,\tag{28}
$$

where $\tau _ { \mathrm { b a t c h } }$ is the detached 95th percentile of native-metric response energies $d ^ { \top } F _ { t } d$ in the current training batch. The percentile rule is fixed across all experiments; the threshold value is batch-adaptive and is not scheduled by timestep or training iteration. Thus in-region responses are unchanged and only unusually large conditioning shifts are rescaled. Trust clipping and reliability gating address different failure modes: clipping prevents a rare high-norm conditioning response from dominating $D _ { t }$ or the directional score, whereas $r _ { t }$ asks whether the resulting span generalizes to held-out current policy responses. We apply three safeguards before a response enters $D _ { t } \colon$ (i) prompt-level intent filtering rejects alternatives that alter the requested event or introduce unsupported entities; (ii) Eq. 28 limits response magnitude; and (iii) near-collinear columns are stabilized by the trace-normalized ridge term rather than explicitly inverted. The percentile rule is batch-adaptive and is used as a robustifier rather than a separately tuned reward weight; fixed-α and binary-gate controls isolate the subsequent routing logic.

The behavior generator snapshot $\theta ^ { - }$ is detached when computing alternative responses. Reward-only local directions, reliability scores, $D _ { t } , P _ { \lambda , t } , r _ { t } ,$ and all cross-fitted functional credits are treated as stop-gradient targets in the optimizer step. This prevents the generator from changing the geometry to obtain easier routing and prevents the prompt policy from differentiating through generator probes rather than learning from the resulting credit.

## C Training Algorithm

Algorithm 2 One joint post-training round   
Require: request $u ;$ prompt policy $\pi _ { \phi } ;$ generator $G _ { \theta } ; K$ candidates; M matched seeds   
Hierarchical credit assignment   
1: sample $p 1 { : } K$ and decode $x _ { k , m } = G _ { \theta } ( p _ { k } ; z _ { m } )$ with shared $z _ { 1 } ; M$   
2: evaluate $R _ { k , m }$ against the original request u; compute $\widehat { A } _ { k } ^ { P , R }$ and $\widehat { A } _ { k , m } ^ { G }$   
Functional coordination at actual rollout states   
3: for each selected actual transition state $s _ { t } ^ { k , m }$ do   
4: obtain reward-only local direction $( q _ { t } ^ { R } , g _ { t } ^ { R } )$ from the base generator updater   
5: probe $d _ { j  k , t }$ for $j \neq k ;$ cross-fit $d _ { j  k , t } ^ { \top } q _ { t } ^ { R }$ into $C _ { j }$   
6: construct $P _ { \lambda , t \dot { \imath } }$ estimate support r from $B _ { \mathrm { t r a i r } }$ independent non-decoded prompt probes   
7: route $g _ { t } ^ { G } \gets ( I - \alpha _ { \operatorname* { m a x } } r _ { t } \bar { P _ { \lambda , t } } ) g _ { t } ^ { R }$   
8: end for   
9: update prompt policy with $\mathrm { C a l } _ { R } ( \widehat { A } ^ { P , R } ) + \lambda _ { F } \mathrm { C a l } _ { C } ( C )$   
10: update generator with $F _ { t } g _ { t } ^ { G }$ plus the unchanged base regularizer

## D Implementation and Evaluation Protocol

## D.1 Backbone, prompt optimizer, and training pool

The primary generator is the public LTX-2.5 trainable DiT checkpoint from the official Lightricks $/ \operatorname { L T X } - 2 . 5$ release (HaCohen et al., 2026; Lightricks, 2026), specifically diffusion\_models/ltx-2.5-22b-dev-transformer-bf16.safetensors, with its native flow parameterization and scheduler. The prompt optimizer is the instruction-tuned Qwen/Qwen3.5-9B checkpoint (Qwen Team, 2026), used as a text-generation policy; we do not use the separate Qwen3.5-9B-Base checkpoint. The main configuration uses full-parameter prompt-policy updates. HunyuanVideo-1.5 (Wu et al., 2025) provides backbone transfer, while

RVM (Choi et al., 2026) provides an alternative generator reward updater under the same LTX evaluation harness.

The training pool contains 8,000 T2V requests. We build it from 4,800 programmatically instantiated compositional templates over entities, attributes, relations, counts, and actions; 2,000 independently generated motion/interaction requests; and 1,200 manually curated multi-event requests, then stratify the mixture across object/attribute binding, spatial relation, motion binding, action binding, interaction, numeracy, and longer event descriptions. Before training, requests are deduplicated against every benchmark split by normalized exact match, 5-gram MinHash (Jaccard threshold 0.80), and sentence embedding cosine (threshold 0.92); no retained training request crosses any of these thresholds. Validation prompts are disjoint at the request level and are never used for benchmark reporting. Prompt candidates are constrained to preserve the user request, avoid unsupported entities, and remain within the generator context budget. Candidate validity is checked by a frozen Qwen/Qwen3.5-9B copy that is separate from the trainable prompt-policy weights. The checker receives the original request and one candidate and deterministically returns PASS/FAIL under a fixed rubric: a candidate fails if it changes required entities, counts, relations, actions, or event ordering, or introduces unsupported entities; descriptive camera, lighting, and appearance details are permitted only when they do not alter those constraints. Greedy decoding is used, and the same checker and rubric are applied to all prompt-side baselines and to additional support probes. Rejected candidates are resampled before video generation. For the theory above, π<sub>ϕ</sub> denotes the effective distribution over accepted prompt actions induced by this fixed filtering rule. The exactness claim concerns the hierarchical reward decomposition and its score-function baseline identity; the practical prompt-policy update uses the clipped group-relative surrogate in Table 5. The same accepted-action sampling rule is used for all prompt-side methods.

## D.2 Default optimization configuration

Table 5: Default LTX-2.5 configuration used by the primary matched comparison. The rows labeled as central controls are held fixed across the principal ablations.
<table><tr><td>Component</td><td>Setting</td><td>Role in the comparison</td></tr><tr><td>Prompt candidates</td><td> $K = 4$ </td><td>four current-policy actions per request</td></tr><tr><td>Matched seeds</td><td> $M = 4$ </td><td>same initial latent and stochastic transition stream per seed across candidates</td></tr><tr><td>Decoded rollouts</td><td>16/request</td><td>held fixed across the central chain</td></tr><tr><td>Generator backend</td><td>Flow-GRPO</td><td>base reward optimizer</td></tr><tr><td>Generator trainable set</td><td>fixed across generator-training rows</td><td>Full changes routing, not which generator pa- rameters are trainable</td></tr><tr><td>Joint training rounds</td><td>500</td><td>global batch 2; 1,000 request draws from the</td></tr><tr><td>Global request batch</td><td>2</td><td>shared 8k pool two requests per joint rollout/update round</td></tr><tr><td>Generator optimizer</td><td> $\mathrm { A d a m w } , 5 { \times } 1 0 ^ { - 6 }$ </td><td>weight decay 0.01; gradient clip 1.0</td></tr><tr><td>Prompt-policy optimizer</td><td> $\mathrm { A d a m w } , 1 { \times } 1 0 ^ { - 6 }$ </td><td>weight decay 0.01; gradient clip 1.0</td></tr><tr><td>Prompt sampling</td><td> $T = 0 . 8 , \mathrm { t o p } \mathrm { - } p = 0 . 9 5$ </td><td>max 256 new tokens;  $K = 4$  accepted candi- dates</td></tr><tr><td>Prompt-policy clip / KL</td><td> $0 . 2 / 0 . 0 1$ </td><td>group-relative clipped objective with frozen reference</td></tr><tr><td>Flow-GRPO clip / KL</td><td> $1 0 ^ { - 3 } / 0 . 0 0 4$ </td><td>one inner epoch; reward and reference</td></tr><tr><td>Mixed precision</td><td>bfloat16</td><td>branches kept separate optimizer states in FP32</td></tr><tr><td>Routed states</td><td>1/trajectory</td><td>uniformly sampled from valid reward-loss transitions</td></tr><tr><td>Training support probes</td><td> $B _ { \mathrm { t r a i n } } = 6$ </td><td>reliability only; no video decode</td></tr><tr><td>Candidate scaling</td><td>frozen calibration affine scale</td><td>Section 3.3; no current-batch statistics</td></tr><tr><td>Functional-credit weight</td><td> $\lambda _ { F } = 0 . 3 5$ </td><td>combines reward and local prompt credit</td></tr><tr><td>Max routing strength</td><td> $\alpha _ { \mathrm { m a x } } = 0 . 7 5$ </td><td>upper bound after reliability gating</td></tr><tr><td>Trust clip</td><td>95th percentile</td><td>detached native-metric response energy  $d ^ { \top } F _ { t } d ;$  fixed percentile</td></tr><tr><td>Ridge coefficient</td><td> $\epsilon _ { \mathrm { r i d g e } } = 1 0 ^ { - 3 }$ </td><td> $\lambda _ { t } = \epsilon _ { \mathrm { r i d g e } } \mathrm { t r } ( D _ { t } ^ { \top } F _ { t } D _ { t } ) / ( K - 1 ) ;$  no sched- ule</td></tr><tr><td>Prompt-policy update</td><td> $\mathrm { f u l l \ p a r a m e t e r }$ </td><td>Qwen3.5-9B policy</td></tr><tr><td>Evaluation probes</td><td> $B _ { \mathrm { e v a l } } = 1 2$ </td><td>mechanism analysis only; independent of training gate</td></tr><tr><td>Benchmark seeds</td><td>3/request</td><td>matched across methods and averaged per re- quest</td></tr><tr><td>Central training runs</td><td>HC/Full:  $_ { 3 ; }$  others: 2</td><td>run is the outer replication unit</td></tr></table>

If the response Gram trace is numerically negligible, we set the response operator and routing strength to zero rather than inverting an ill-conditioned matrix; the clip on $r _ { t }$ is retained only as a numerical guard.

Qualitative outputs use the native 5-second, 24-fps, $8 3 2 \times 4 8 0$ generation format. We apply no spatial upscaling or frame interpolation to the videos or to the frames shown in the paper.

## D.3 Reward and optimization details

Each decoded video is scored against the original request by a fixed public reward stack. We use: (i) the Text-Alignment output of KlingTeam/VideoReward (Liu et al., 2025b) for request faithfulness $r _ { \mathrm { f a i t h } } ;$ (ii) VQAScore with zhiqiulin/clip-flant5-xxl (Lin et al., 2024), averaged over eight uniformly spaced frames, for compositional correspondence $r _ { \mathrm { c o m p } } ;$ (iii) TIGER-Lab/VideoScore-v1.1 (He et al., 2024), using the mean of its temporal-consistency and text-to-video-alignment outputs over the model’s native frame sampling, for event realization $r _ { \mathrm { e v e n t } } ;$ and (iv) the mean of VideoReward visual-quality and motion-quality outputs as the bounded quality stabilizer $r _ { \mathrm { q u a l } }$ . All model weights, prompts, frame-sampling rules, and preprocessing are frozen before controlled training.

For each component, location and scale are estimated once on a disjoint calibration pool of 2,048 training-only videos and then held fixed throughout all controlled runs:

$$
\widetilde { r } _ { c } = \mathrm { c l i p } \left( \frac { r _ { c } - \mu _ { c } ^ { \mathrm { c a l } } } { \sigma _ { c } ^ { \mathrm { c a l } } + 1 0 ^ { - 6 } } , - 3 , 3 \right) , \qquad R ^ { \mathrm { o p t } } = 0 . 3 5 \widetilde { r } _ { \mathrm { f a i t h } } + 0 . 3 0 \widetilde { r } _ { \mathrm { c o m p } } + 0 . 2 5 \widetilde { r } _ { \mathrm { e v e n t } } + 0 . 1 0 \widetilde { r } _ { \mathrm { q u a l } } .\tag{29}
$$

Table 6: Frozen reward-stack configuration and calibration statistics. The calibration pool is disjoint from all controlled training/evaluation requests.
<table><tr><td>Component</td><td>Public checkpoint / signal</td><td>Video input</td><td>Weight</td><td> $\mu _ { c } ^ { \mathrm { c a l } }$ </td><td> $\sigma _ { c } ^ { \mathrm { c a l } }$ </td></tr><tr><td>Ifaith</td><td>KlingTeam/VideoReward, Text Alignment</td><td>native sampler</td><td>0.35</td><td>0.42</td><td>0.17</td></tr><tr><td>rcomp</td><td>zhiqiulin/clip-flant5-xxl,VQAScore</td><td>8-frame mean</td><td>0.30</td><td>0.63</td><td>0.15</td></tr><tr><td>revent</td><td>TIGER-Lab/VideoScore-v1.1,TC+T2V align</td><td>native sampler</td><td>0.25</td><td>2.71</td><td>0.39</td></tr><tr><td> $r _ { \mathrm { q u a l } }$ </td><td>KlingTeam/VideoReward, VQ+MQ mean</td><td>native sampler</td><td>0.10</td><td>0.56</td><td>0.16</td></tr></table>

Reported VBench, VBench-2.0, T2V-CompBench, and StoryEval evaluators never enter $R ^ { \mathrm { { o p t } } }$ or the calibration pool. Because $( \mu _ { c } ^ { \mathrm { c a l } } , \sigma _ { c } ^ { \mathrm { c a l } } )$ are independent of the current sampled prompts and trajectories, $R ^ { \mathrm { o p t } } ( u , x )$ remains a per-trajectory scalar score and Proposition 1 applies directly to the transformed objective. The same principle is used for the frozen calibration transform in Section 3.3: rewardcredit and functional-credit scales are calibrated once and frozen rather than recomputed from current candidates.

Scorer-family independence. We additionally evaluate the final hierarchical-credit and Full checkpoints with frozen VisionReward (Xu et al., 2026), which is not used anywhere in training, calibration, model selection, or the four headline benchmarks. On 300 held-out matched request/seed pairs, VisionReward prefers Full on 63.3% of pairs and its normalized scalar score increases by 0.18 standard deviations on average. This scorer-family check is consistent with the blinded human preference study in Appendix L and complements the reward-axis interventions below.

For Flow-GRPO (Liu et al., 2025a), Eq. 4 supplies the hierarchy-aware trajectory advantage used by the reward branch. We use the fixed LTX configuration in Table 5: ratio clip $1 0 ^ { - 3 }$ , KL coefficient 0.004, one inner epoch, and bfloat16 sampling/training; these settings are held fixed across the generator-side controlled comparisons. The routed state is sampled uniformly from the valid stochastic-policy transitions contributing to the reward loss. Concretely, the cached state stores the current noisy latent together with the timestep and scheduler scalars needed to evaluate the next transition; behavior parameters, stochastic draw, and all non-text state are fixed during alternativeprompt probing. The functional probe differentiates the post-clipping reward branch with respect to the transition mean. The reference/KL branch is never routed. Prompt-policy GRPO uses Eq. 7, with terminal and functional components calibrated separately by the frozen statistics defined in Section 3.3 before weighted combination.

Instantiation with RVM. RVM (Choi et al., 2026) uses reward-weighted velocity matching together with an optional anchor toward a reference velocity field. In the reported RVM transfer, functional coordination is instantiated in this backend’s native velocity-output coordinates. At a sampled noised endpoint state $( x _ { t } , t )$ under executed prompt $p _ { k }$ , write $h _ { t } = v _ { \theta } ( x _ { t } , t , p _ { k } )$ . We obtain the local rewarddriven direction from the reward-weighted velocity branch, $q _ { t } ^ { R } = - \nabla _ { h _ { t } } \mathcal { L } _ { \mathrm { R V M , r e w a r d } }$ , absorbing the backend’s scalar timestep weight into $q _ { t } ^ { R }$ and using the Euclidean metric in $h _ { t }$ coordinates. Alternative-prompt responses use the identical noised state and time under the detached behavior snapshot,

$$
d _ { j  k , t } ^ { \mathrm { R V M } } = v _ { \theta ^ { - } } ( x _ { t } , t , p _ { j } ) - v _ { \theta ^ { - } } ( x _ { t } , t , p _ { k } ) .\tag{30}
$$

The same trust clipping, ridge response operator, and held-out support gate are then applied in velocityoutput space. Routing modifies only the reward-weighted velocity branch; RVM’s anchor/reference branch is restored unchanged. Thus functional coordination requires a backend-native local reward direction and same-state conditioning responses, while the local coordinate and metric are supplied by the updater rather than assumed to be a stochastic-transition Fisher parameterization.

Instantiation with differentiable reward flow. We additionally instantiate functional coordination on a Diffusion-DRF differentiable reward-flow backend (Wang et al., 2026b). Following the public

DRF setup, the critic is frozen Qwen2.5-VL-7B; ten uniformly sampled video frames are used for VLM feedback, and gradients are truncated to the final three denoising steps. At a routed state we take the differentiable reward gradient with respect to the backend transition output as $q _ { t } ^ { R }$ and use its native Euclidean output metric for $g _ { t } ^ { R }$ . Alternative-prompt responses are evaluated at the identical state under the detached generator snapshot. The same trust clip, trace-normalized ridge operator, and held-out support gate are then applied before backpropagating the routed reward direction. Critic weights, frame sampling, the three-step backpropagation window, gradient checkpointing, and nonreward stabilization terms are held fixed between hierarchy-only and Full. This experiment tests whether cross-module routing remains useful when the generator already receives dense differentiable feedback rather than a scalar policy-gradient advantage.

Backend applicability. Functional coordination requires two backend-local objects: (i) an output coordinate in which changing text conditioning at a fixed state yields a response vector, and (ii) a local improvement covector $q _ { t } ^ { \breve { R } }$ supplied by the generator updater. Flow-GRPO, RVM, and differentiable reward flow provide both directly. A preference objective can be compatible if its implementation exposes a differentiable state-local pairwise loss, in which case the loss gradient can play the role of $q _ { t } ^ { R } ;$ a purely trajectory-level preference update with no local improvement covector is outside the current formulation. Likewise, closed API-only generators are outside our target setting because joint generator post-training itself already assumes access to internal transitions and parameters.

## D.4 Benchmark protocol

VBench (Huang et al., 2024) and VBench-2.0 (Zheng et al., 2025) use their official aggregate/component definitions. T2V-CompBench (Sun et al., 2025) uses its official compositional dimensions and reported aggregate. StoryEval (Wang et al., 2025) follows the same official eventcompletion protocol for every generator updater and backbone. Every checkpoint uses identical resolution, duration, scheduler steps, negative-prompt convention, and three fixed generation seeds per benchmark request. The three scores for a request are averaged first; official benchmark aggregation is then applied to those request-level means. This preserves matched stochasticity while preventing seed count from being treated as sample size.

We distinguish three sources of uncertainty: training-run variation, within-checkpoint generation randomness, and request sampling. The headline hierarchical-credit-versus-Full comparison uses three independent training runs; the functional-coordination factorial uses the same three-run protocol, and other trainable baselines use two independent runs. Confidence intervals for the headline increment are computed on paired run-level deltas using a Student-t interval; request-level paired bootstrap intervals are retained as a within-run sensitivity analysis. Generation seeds are averaged within each checkpoint and are never treated as independent training replicates. The same checkpoint and seed definitions are used across all controlled comparisons.

## D.5 Baseline implementation

All trainable baselines draw from the same 8,000-request training pool and use the same number of request draws and decoded videos unless a comparison explicitly targets compute. The frozen official LTX-2.5 prompt enhancer uses the reference LTX-2.5 enhancement path (Lightricks, 2026): google/gemma-4-E2B-it with LTX2 5 T2V DEFAULT SYSTEM PROMPT, greedy decoding (do sample=False), no repeat ngram size=5, and at most 600 new tokens. Prompt-only updates Qwen3.5-9B while freezing LTX-2.5; the Flow-GRPO generator-only row keeps the original request as its text interface and is included as a single-side reference rather than as the matched joint training control. Prompt-A-Video (Ji et al., 2025) and VPO (Cheng et al., 2025) use their published prompt-side objectives within the common Qwen3.5-9B prompt optimizer, training pool, LTX-2.5 generator, and evaluation harness. Because their original backbones and training stacks differ, the reported values are controlled common-harness results rather than copied published absolute scores.

PromptRL under the common video harness uses the published shared-reward joint-training principle: the trainable prompt refiner and Flow-GRPO generator receive the same terminal video reward, with candidate count, rollout count, reward stack, initialization, and update budget held fixed. UniGRPO adds group-relative normalization for the text policy and generator branch within the same joint rollout groups, retaining the published unified sparse-terminal-reward idea while keeping our LTX-2.5 backbone and reward stack fixed. We use these rows to compare joint-optimization principles within one harness, not as claims of exact reproduction of the original image-generation systems (Wang et al., 2026a; Liu et al., 2026). Matched Global-LOO then trains the same two modules on the same $K \times M$ grid with the global leave-one-out definition in Appendix A.2; hierarchical credit changes only the credit hierarchy, and Full adds functional prompt shaping and generator routing.

CAPE-T2V (Jia et al., 2026) is the closest T2V-specific two-sided system in our comparison set, but it addresses the prompt–caption distribution gap through sequential Anchored-PE training followed by DiT adaptation on a separately constructed 54k video-derived caption set. Its released LTX pipeline targets LTX-2.3, whereas our mechanism-controlled study uses online joint reward post-training on LTX-2.5 with no auxiliary video-caption supervision. We therefore use CAPE-T2V for external positioning rather than treating its public absolute scores as a matched-budget ranking.

Table 7: Positioning of two-sided adaptation systems. The first two rows are controlled LTX-2.5 implementations used numerically in Table 1; CAPE-T2V is a heterogeneous external reference and is not mixed into the matched-budget ranking.
<table><tr><td>System</td><td>Prompt side</td><td>Generator side</td><td>Additional supervision</td><td>Training form</td></tr><tr><td>PromptRL</td><td>trainable LM refiner</td><td>Flow-GRPO</td><td>none</td><td>online shared-terminal-reward joint update</td></tr><tr><td>UniGRPO</td><td>trainable text policy</td><td>Flow-GRPO</td><td>none</td><td>online group-relative joint policy optimization</td></tr><tr><td>CAPE-T2V</td><td>Anchored PE</td><td>DiT fine-tuning</td><td>54k video-derived captions</td><td>sequential two-sided conditioning alignment</td></tr><tr><td>FLOWCRED</td><td>Qwen3.5-9B policy</td><td>Flow-GRPO</td><td>none</td><td>online hierarchical credit + functional coordination</td></tr></table>

Wrong-query controls are matched to the target request by compositional category, raw-prompt length, enhanced-prompt length, candidate count, and hierarchical-credit difficulty bin. Random spans are rank matched after Fisher whitening and sample response norms from the empirical true-span distribution. Uniform shrink chooses its scalar on the development split.

## D.6 Aggregate benchmark summary

Table 8: Aggregate benchmark status for the principal LTX-2.5 systems under the unified evaluation harness.
<table><tr><td>Benchmark</td><td>Metric</td><td>Base</td><td>Prompt-only</td><td>Global-LOO</td><td>Hier. credit</td><td>Full</td></tr><tr><td>VBench</td><td>Total</td><td>78.14</td><td>86.97</td><td>86.49</td><td>86.91</td><td>87.56</td></tr><tr><td>VBench-2.0</td><td>Overall</td><td>54.52</td><td>62.84</td><td>63.41</td><td>63.92</td><td>64.73</td></tr><tr><td>T2V-CompBench</td><td>Overall</td><td>46.52</td><td>60.28</td><td>61.47</td><td>62.63</td><td>64.46</td></tr><tr><td>StoryEval</td><td>Overall</td><td>37.88</td><td>54.37</td><td>55.58</td><td>56.71</td><td>58.08</td></tr></table>

## E Hierarchical Credit Estimator Efficiency

Hierarchical credit makes two testable statistical predictions: prompt-conditioned generator baselines remove irrelevant prompt-to-prompt variation, while matched seeds reduce prompt-policy contrast variance when same-seed rewards are positively correlated. Both effects are measured directly rather than inferred from endpoint quality.

![](images/b32710b407e38dedb7f7edc9b4e0d47751f0489c054be4dc862615304b9e7509.jpg)  
(a) Generator advantage variance: global leave-oneout versus prompt-conditional residual.

![](images/cdd563b7eb75e47e519bcbc41a74257f50bce3de2d6029ea8b4880571937b4c9.jpg)  
(b) Prompt candidate-credit variance: independent versus matched seeds at equal video count.  
Figure 6: Estimator-efficiency diagnostics across held-out minibatches: prompt-conditioned generator baselines reduce generator-advantage variance, and matched seeds reduce prompt candidate-credit variance at equal video count.

Generator-advantage variance is reduced by about 31%, and matched seeds reduce prompt-credit variance by about 27%. The matched-seed ablation keeps total decoded videos fixed and yields 62.63 T2V-CompBench versus 62.21 with independently sampled seeds.

## F Mechanism Diagnostics

Table 9: Functional-coordination factorial ablation. Mean (SD) across the same three-run protocol used for the HC–Full comparison.
<table><tr><td>Prompt shaping</td><td>Generator routing</td><td>VBench↑</td><td>StoryEval↑</td><td>T2V-CompBench↑</td></tr><tr><td></td><td></td><td>86.91 (0.14)</td><td>56.71 (0.27)</td><td>62.63 (0.32)</td></tr><tr><td>√</td><td></td><td>87.20 (0.15)</td><td>57.36 (0.28)</td><td>63.40 (0.34)</td></tr><tr><td></td><td>√</td><td>87.29 (0.13)</td><td>57.18 (0.26)</td><td>63.67 (0.31)</td></tr><tr><td>√</td><td>√</td><td>87.56 (0.12)</td><td>58.08 (0.24)</td><td>64.46 (0.29)</td></tr></table>

## F.1 Realized-update audit

For cached states, let $\Delta \mu _ { t } ^ { \mathrm { r e a l } } = \mu _ { \theta ^ { + } } ( s _ { t } , p _ { k } ) - \mu _ { \theta } ( s _ { t } , p _ { k } )$ . Hierarchical credit and Full are branched from the same checkpoint with the same minibatch, optimizer state, cached states, scheduler state, and random state. We report

$$
\rho _ { \mathrm { s p a n } } ^ { \mathrm { r e a l } } = \frac { \| P _ { \lambda , t } \Delta \mu _ { t } ^ { \mathrm { r e a l } } \| _ { F _ { t } } ^ { 2 } } { \| \Delta \mu _ { t } ^ { \mathrm { r e a l } } \| _ { F _ { t } } ^ { 2 } + \epsilon } ,\tag{31}
$$

together with $\| \Delta \mu _ { \mathrm { F u l l } } ^ { \mathrm { r e a l } } \| _ { F } / \| \Delta \mu _ { \mathrm { H C } } ^ { \mathrm { r e a l } } \| _ { F }$ . The realized comparison is 0.487→0.264 for supported overlap and 0.931 for the update-norm ratio. These quantities are measured after the neural optimizer step and avoid interpreting the energy projection of an individual stochastic transition sample.

## F.2 Prompt-policy shaping and fresh-seed utility

The mechanism set contains 360 requests with four candidates each. Scores use the matched $M = 4$ training seeds and utility uses eight disjoint fresh seeds; bootstrap intervals resample requests while preserving candidate groups. An 80-request subset with 16 fresh seeds checks ranking stability. Controls include response norm, functional-only, reward-only, cosine-functional, and the default trust-clipped directional score.

## F.3 True, random, wrong-query, and generator-specific geometry

Wrong-query spans are matched by compositional category, prompt length, candidate count, and hierarchical-credit difficulty. Evaluation uses independent $\bar { B _ { \mathrm { e v a l } } } = \bar { 1 } 2$ alternatives that never enter

the training gate. Generator specificity scores identical candidate sets with Base, reward-trained, and Full snapshots and evaluates fresh-seed utility under each rollout generator.

## G Prompt Integrity and Data-Audit Controls

Because Prompt-only already produces large absolute gains, all evaluator inputs remain the original request u, and the paper separately audits verbosity, semantic drift, and train/evaluation leakage.

Table 10: Prompt-integrity diagnostics. Hierarchical credit and Full use nearly identical prompt lengths and semantic-filter behavior.
<table><tr><td>Diagnostic</td><td>Raw request</td><td>Prompt-only</td><td>HC</td><td>Full</td></tr><tr><td>Median tokens</td><td>18</td><td>111</td><td>117</td><td>120</td></tr><tr><td>P90 tokens</td><td>31</td><td>149</td><td>153</td><td>157</td></tr><tr><td>Intent-filter rejection</td><td></td><td>4.3%</td><td>4.7%</td><td>4.5%</td></tr><tr><td>Blind semantic drift (N = 200)</td><td>一</td><td>7/200</td><td>5/200</td><td>5/200</td></tr></table>

On a token-length-matched subset, Full still gains +0.61 VBench, +1.22 StoryEval, and +1.59 T2V-CompBench over hierarchical credit. Training/evaluation separation uses normalized-string matching plus lexical and embedding nearest-neighbor retrieval with thresholds fixed before benchmark reporting.

## H Complete Mechanism and Control Audit

For compactness in the ablation plots and tables, “HC” denotes the hierarchical-credit system without functional coordination.

Table 11: Compact mechanism and routing audit. Panel A evaluates prompt-candidate scores on disjoint fresh seeds; Panel B tests matched-budget routing alternatives.

A. Fresh-seed prompt-candidate ranking
<table><tr><td>Candidate score</td><td>Spearman ρ</td><td>Top-1</td></tr><tr><td>Response norm only</td><td>0.313 [0.251, 0.374]</td><td>50.3% [46.1,54.4]</td></tr><tr><td>Functional only</td><td>0.421 [0.369, 0.472]</td><td>55.8% [51.6,60.0]</td></tr><tr><td>Reward only</td><td>0.447 [0.398, 0.497]</td><td>59.2% [55.0,63.2]</td></tr><tr><td>Reward + cosine functional</td><td>0.569 [0.519, 0.618]</td><td>67.2% [63.2,71.0]</td></tr><tr><td>Reward + directional functional</td><td>0.612 [0.568, 0.655]</td><td>71.7% [67.8,75.4]</td></tr></table>

B. Matched-budget routing controls
<table><tr><td>Variant</td><td>T2V-CompBench</td><td>∆ vs. HC</td></tr><tr><td>Hierarchical credit</td><td>62.63</td><td>0.00</td></tr><tr><td>Rank-matched random span</td><td>62.55</td><td>-0.08</td></tr><tr><td>Matched wrong-query span</td><td>62.69</td><td>+0.06</td></tr><tr><td>Uniform gradient shrink</td><td>62.81</td><td>+0.18</td></tr><tr><td>No cross-fit</td><td>63.44</td><td>+0.81</td></tr><tr><td>No reliability gate</td><td>63.67</td><td>+1.04</td></tr><tr><td>Full</td><td>64.46</td><td>+1.83</td></tr></table>

## I Generalization and Compute

## I.1 Backbone and generator-updater transfer

The Hunyuan experiment preserves candidate construction, matched estimators, executed-action reference, and allocation logic while substituting backbone-native transition quantities. RVM and Diffusion-DRF comparisons are likewise within-updater: hierarchy-only versus the same system plus functional coordination under an identical StoryEval/T2V-CompBench harness. We therefore claim positive within-setting increments, not cross-optimizer rankings. Under the same VBench-2.0 protocol, the measured raw-prompt Overall scores are 54.79 for HunyuanVideo-1.5 and 54.52 for LTX-2.5.

Table 12: Transfer comparisons across a second backbone and two alternative generator updaters, including differentiable reward flow. Absolute scores are shown within each matched setting rather than used to rank heterogeneous optimizers.
<table><tr><td>Method / within-setting contrast</td><td>VBench</td><td>T2V- CompBench</td><td>StoryEval</td></tr><tr><td>HunyuanVideo-1.5 / Flow-GRPO</td><td></td><td></td><td></td></tr><tr><td>Matched Global-LOO</td><td>84.67</td><td>55.34</td><td>55.12</td></tr><tr><td>Hier. credit</td><td>84.96</td><td>55.89</td><td>56.04</td></tr><tr><td>Full</td><td>85.53</td><td>57.11</td><td>57.13</td></tr><tr><td>LTX-2.5 / RVM</td><td></td><td></td><td></td></tr><tr><td>Hier. credit</td><td>86.39</td><td>61.28</td><td>56.18</td></tr><tr><td>Full</td><td>86.91</td><td>62.57</td><td>57.33</td></tr><tr><td>LTX-2.5 / Diffusion-DRF</td><td></td><td></td><td></td></tr><tr><td>Hier. credit</td><td>87.08</td><td>63.16</td><td>57.24</td></tr><tr><td>Full</td><td>87.49</td><td>64.13</td><td>58.03</td></tr></table>

## I.2 Local-compute accounting

Full keeps the decoded-video and reward-model-call budget identical to hierarchical credit. Under the default $K = 4$ and $B _ { \mathrm { t r a i n } } = 6$ , it additionally samples 96 accepted support prompts per request (six at the routed state of each of 16 trajectories) and performs 144 single-state generator transition forwards: 48 for the $K - 1 = 3$ rollout-candidate responses and 96 for the support probes. These operations do not decode additional videos or invoke the video reward stack. The same frozen intent checker screens support prompts; extra proposal/checker calls caused by rejection resampling are data-dependent and are therefore not included in the deterministic count. End-to-end hardware cost depends on the generator implementation and prompt-policy sampling stack, so we report deterministic operation counts rather than presenting a universal accelerator-hour ratio.

Table 13: Deterministic per-request operation counts for the primary LTX setting.
<table><tr><td>Quantity</td><td>Hierarchical credit</td><td>Full</td></tr><tr><td>Decoded videos / request</td><td>16</td><td>16</td></tr><tr><td>Reward-model calls / request</td><td>16</td><td>16</td></tr><tr><td>Accepted support prompts / request</td><td>0</td><td>96</td></tr><tr><td>Same-state transition probe forwards</td><td>0</td><td>144</td></tr></table>

![](images/fd3a5ed697dafc73a26b3b74adf3dabbc08afcf1354bb30a8dcc04d6da3c81b3.jpg)  
Figure 7: Within-setting transfer of the Full−hierarchical-credit increment across a second backbone and two alternative generator updaters.

## J Robustness and Replication

We keep robustness reporting focused on two questions that materially affect the main claim: whether the method remains useful when the decoded rollout budget changes, and whether the hierarchical-

credit-to-Full increment repeats across independent training runs. Hyperparameters are fixed before the reported test evaluation; benchmark test scores are not used for selection.

Table 14: Decoded-rollout budget study with K = 4. Hierarchical credit requires $M \ \geq \ 2$ because the prompt-conditioned leave-one-seed-out residual is undefined for a single trajectory.
<table><tr><td>M</td><td>Videos/request</td><td>PromptRL</td><td>Global-LOO</td><td>Hier. credit</td><td>Full</td><td>Full-HC</td></tr><tr><td>2</td><td>8</td><td>59.74</td><td>60.21</td><td>61.11</td><td>63.18</td><td>+2.07</td></tr><tr><td>4</td><td>16</td><td>61.06</td><td>61.47</td><td>62.63</td><td>64.46</td><td>+1.83</td></tr><tr><td>8</td><td>32</td><td>62.11</td><td>62.38</td><td>63.34</td><td>64.87</td><td>+1.53</td></tr></table>

All methods improve with more terminal evidence, while the Full-minus-HC increment remains positive across the three budgets. We treat M as a compute axis rather than a free hyperparameter improvement.

The main hierarchical-credit-versus-Full comparison is repeated for three independent training runs.   
Run-level deltas, not generation-seed cells, are the outer replication units.

![](images/0601a75065b90e756ac1ba075fe4e00562f1c573cdc0151a2e9ececc6fa7029e.jpg)  
Figure 8: Three-run replication. Small points are paired training-run deltas; diamonds show means and horizontal intervals show Student-t 95% intervals over the three run-level differences.

Table 15: Raw Full−hierarchical-credit deltas for three independent training replicates.
<table><tr><td>Metric</td><td>Rep. A</td><td>Rep. B</td><td>Rep. C</td></tr><tr><td>VBench</td><td>0.59</td><td>0.69</td><td>0.67</td></tr><tr><td>VBench-2.0</td><td>0.72</td><td>0.87</td><td>0.84</td></tr><tr><td>T2V-CompBench</td><td>1.62</td><td>1.88</td><td>1.99</td></tr><tr><td>StoryEval</td><td>1.16</td><td>1.43</td><td>1.52</td></tr></table>

The corresponding means are 0.65 VBench, 0.81 VBench-2.0, 1.83 T2V-CompBench, and 1.37 StoryEval. With only three outer replicates, we use these intervals descriptively and do not turn them into stronger significance claims.

## K Realized Cross-Module Coordination Audit

The response span establishes local conditioning capacity; this audit asks whether the trained modules subsequently move in the complementary pattern predicted by functional coordination. For each heldout routed state, we cache the pre-update behavior generator and estimate the prompt-policy-induced displacement

$$
\Delta \mu _ { \mathrm { P } } = \mathbb { E } _ { p \sim \pi _ { \phi ^ { + } } } \big [ \mu _ { \theta ^ { - } } ( s , p ) \big ] - \mathbb { E } _ { p \sim \pi _ { \phi } } \big [ \mu _ { \theta ^ { - } } ( s , p ) \big ] .
$$

We measure its native-metric cosine with the supported direction and the supported component retained by the realized generator step. The four variants use identical held-out states and promptsampling seeds.

Table 16: Realized cross-module coordination on held-out states.
<table><tr><td>Training variant</td><td> $\cos _ { F } ( \Delta \mu _ { \mathrm { P } } , P _ { \lambda } g ^ { R } )$ </td><td>Generator supported overlap ↓</td></tr><tr><td>Hierarchical credit</td><td>0.29</td><td>0.487</td></tr><tr><td>Prompt shaping only</td><td>0.55</td><td>0.471</td></tr><tr><td>Generator routing only</td><td>0.30</td><td>0.281</td></tr><tr><td>Full</td><td>0.61</td><td>0.264</td></tr></table>

The factorial pattern is consistent with the intended division of labor: prompt shaping mainly changes the upstream displacement, generator routing mainly reduces the downstream component lying in the response span, and Full exhibits both effects. We treat this as a descriptive consistency audit rather than a causal decomposition.

## L Human Evaluation and Qualitative Cases

Expanded qualitative cases. Figure 9 adds four benchmark examples chosen to broaden the qualitative coverage beyond the two main-text cases. The panel spans numeracy/composition, multi person interaction, spatial relation, and a coupled appearance transition; rows again compare Base, VPO, and FLOWCRED using representative temporal frames.

(a)  
“Five people sit around a table, chatting as they share their meal.”  
![](images/33fcb0b8403653f7d1296e11596aea804d51c0002eb04bcb3cdc4936b43abc00.jpg)

(b)  
“Old friends hug each other at a reunion.”  
![](images/b8ebaa05439e2285496f7039fef97e921634136375dc40261cbdbe78817be5ac.jpg)

(c)  
“A donut placed above a handbag.”  
![](images/ced5dcd892d1ec5958e3e314c7b232d45fbb66c8c9082adb8d61ea3b67c35601.jpg)

“A green chameleon changes its skin color to red as the (d) neon sign is changing from green to red.”  
![](images/f2faecdd7393078a928cd0de1bb5efa7cf649319f612bbc3afa4d474b8728668.jpg)  
Figure 9: Additional qualitative comparisons. Four benchmark cases covering numeracy, interaction, spatial relation, and synchronized appearance change. Rows show Base, VPO, and FLOWCRED.

(a) Numeracy and composition. The key requirement is exact cardinality together with a coherent shared-meal scene. Base over-populates the table, so the generated composition no longer matches the requested five-person configuration. VPO moves in the opposite direction and under-populates the scene. FLOWCRED maintains five visible participants across the representative frames while keeping them organized around one table, preserving both the requested count and the intended conversational meal context rather than satisfying one at the expense of the other.

(b) Protagonist-focused interaction. This case tests whether the model can isolate the intended pair and carry the reunion interaction through to a clear embrace. Base disperses the event into a crowded group hug, making the two target friends difficult to identify consistently. VPO produces an embrace but shows weaker protagonist focus across the sequence. FLOWCRED gives the clearest two-person progression from greeting to contact to hug, while maintaining the same pair more consistently through the action.

(c) Spatial relation. The decisive constraint is the word above, so object presence alone is insufficient. Base and VPO both render a donut and a handbag but fail to maintain the requested relative placement in a stable, unambiguous form. FLOWCRED consistently places the donut above the handbag across the representative frames, turning an otherwise plausible pair of objects into the correct relational composition. This mirrors the larger gains on compositional evaluation, where satisfying relations rather than merely rendering entities is the harder requirement.

(d) Coupled color transition. The final case requires a coordinated temporal state change: both the chameleon and the neon sign should progress from green toward red. Base does not complete the requested transformation, leaving the appearance change partial. VPO reaches a red state but does so abruptly, with a less natural visual progression and a conspicuous change in camera behavior. FLOWCRED produces the most gradual and complete transition: the two scene elements evolve coherently toward red while the framing remains substantially more stable, making the requested transformation easier to follow as a continuous event rather than a sudden endpoint switch.

Blinded human evaluation. We sample 240 requests stratified across compositional and multievent categories. Three blinded annotators receive the original request and anonymized matchedseed videos with randomized left/right placement and score faithfulness, event/relation completion, motion/visual quality, and overall preference. Raw request-level proportions retain ties; the two primary significance tests condition on non-tie Full-versus-baseline outcomes and use exact binomial tests with Holm correction. Per-dimension analyses are descriptive, and Fleiss’ κ reports agreement.

<table><tr><td colspan="2">HC prompt</td><td colspan="2">Full prompt</td></tr><tr><td rowspan="2">Full generator</td><td>63.53</td><td>64.46</td><td rowspan="2">+0.93 prompt</td></tr><tr><td>62.63</td><td>63.35</td></tr><tr><td colspan="2">+0.90 Gen</td><td>+1.11 Gen</td><td></td></tr><tr><td colspan="4">Additive 64.25 -&gt; observed 64.46 (descriptive residual +0.21)</td></tr></table>

(a) Frozen-module 2 × 2 cross-swap. Either Fulltrained module improves the HC pair when substituted alone; the +0.21 residual is descriptive, not a causal synergy claim.

![](images/45287c8bef6419630c089e31cdbe6bea58e2a47551b526ad9cbc3688aa546a6f.jpg)  
(b) Blinded request-level human preference with ties retained.

Figure 10: External validation beyond automatic benchmarks. Frozen-module substitution attributes gains to both trained modules, while human judgments independently prefer Full over HC and PromptRL. Per-dimension vote rates are reported in Table 17.

Table 17: Blinded matched-seed human evaluation. Panel A reports request-level majority outcomes and non-tie Full win rates with Wilson 95% intervals; Panel B reports per-dimension Full preference rates with ties retained.  
A. Primary request-level preference
<table><tr><td>Comparison</td><td>Prefer Full</td><td></td><td>Tie Prefer baseline</td><td>Non-tie Full win</td><td>Holm p</td><td>Fleiss κ</td></tr><tr><td>Full vs. HC</td><td>134</td><td>58</td><td>48</td><td>73.6% [66.8, 79.5]</td><td> $1 . 3 4 \times 1 0 ^ { - 1 0 }$ </td><td>0.51</td></tr><tr><td>Full vs. PromptRL</td><td>142</td><td>52</td><td>46</td><td>75.5% [68.9, 81.1]</td><td> $2 . 7 2 \times 1 0 ^ { - 1 2 }$ </td><td>0.53</td></tr></table>

B. Per-dimension Full preference
<table><tr><td>Comparison</td><td>Dimension</td><td>Prefer Full</td><td>95% CI</td></tr><tr><td rowspan="4">Full vs. HC</td><td>Faithfulness</td><td>128/240 (53.3%)</td><td>[47.0, 59.5]</td></tr><tr><td>Composition/event</td><td>151/240 (62.9%)</td><td>[56.6, 68.8]</td></tr><tr><td>Motion/visual</td><td>122/240 (50.8%)</td><td>[44.5, 57.1]</td></tr><tr><td>Overall</td><td>134/240 (55.8%)</td><td>[49.5, 62.0]</td></tr><tr><td rowspan="4">Full vs. PromptRL</td><td>Faithfulness</td><td>132/240 (55.0%)</td><td>[48.7, 61.2]</td></tr><tr><td>Composition/event</td><td>158/240 (65.8%)</td><td>[59.6, 71.5]</td></tr><tr><td>Motion/visual</td><td>127/240 (52.9%)</td><td>[46.6, 59.1]</td></tr><tr><td>Overall</td><td>142/240 (59.2%)</td><td>[52.9, 65.2]</td></tr></table>

Preference pattern. The request-level majority analysis favors Full in both primary comparisons. Against HC, Full wins 134 requests versus 48 baseline wins, with 58 ties; conditioning on non-ties gives a 73.6% Full win rate. Against PromptRL, the corresponding counts are 142 / 52 / 46 and a

75.5% non-tie win rate. The Holm-corrected exact tests are significant in both comparisons, while Fleiss’ κ values of 0.51 and 0.53 indicate moderate agreement rather than near-deterministic annotator consensus.

Where human gains concentrate. The largest descriptive gap occurs on composition/event completion: Full is preferred on 62.9% of requests against HC and 65.8% against PromptRL. Faithfulness improves more modestly (53.3% and 55.0%), and motion/visual preference remains closer to parity (50.8% and 52.9%). This pattern is consistent with the automatic results, where the largest gains appear on compositional and multi-event benchmarks, while also showing that the preference advantage is not driven solely by a generic visual-quality shift.

Frozen-module attribution. The 2 × 2 post-training cross-swap uses HC/Full prompt optimizers × HC/Full generators and freezes all modules during the audit. Substituting the Full generator into the HC pair contributes +0.90 T2V-CompBench, while substituting the Full prompt optimizer contributes +0.72; with the complementary module already Full-trained, the corresponding increments are +1.11 and +0.93. Starting from 62.63, the HC-conditioned marginal effects predict 64.25 additively, and the Full pair reaches 64.46, leaving only a descriptive +0.21 residual. The audit therefore attributes measurable gains to both trained sides without requiring a strong interaction effect between them.