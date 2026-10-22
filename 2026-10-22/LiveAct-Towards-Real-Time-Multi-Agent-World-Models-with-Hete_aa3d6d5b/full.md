# LiveAct: Towards Real-Time Multi-Agent World Models with Heterogeneous Action Interfaces

Xinyu Wang<sup>1</sup>, Huafeng Shi<sup>2</sup>, Yue Ma<sup>3</sup>, Zijie Meng<sup>4</sup>, Teng Xi<sup>2</sup>, Yuzhuo Chen<sup>5</sup>, Yucheng Wang<sup>3</sup>, Ruicheng Zhang<sup>1</sup>, Guoxin Zhang<sup>2</sup>, Pengfei Wan<sup>2</sup>, Xiu Li<sup>1</sup>, Fangneng Zhan<sup>3</sup>

<sup>1</sup> Shenzhen International Graduate School, Tsinghua University <sup>2</sup> Kling Team

<sup>3</sup> The Hong Kong University of Science and Technology <sup>4</sup> Peking University <sup>5</sup> University of Science and Technology of China Project: https://liveact-ai.github.io

![](images/a0c1b1c0a864e702066639683b18b0a33197ad1567e121b4f1829dc6329084f3.jpg)  
Figure 1. Agent-specific control across heterogeneous domains. Given an initial frame and synchronized action streams, LiveAct generates shared visual futures for multi-agent games (top left) and multi-robot manipulation (top right) while preserving stream-to-agent correspondence (bottom).

## Abstract

Interactive video world models predict how an environment evolves under user actions, but most existing systems assume one controlled agent and one task-specific action interface. We study multi-agent video world modeling, where one model generates a joint visual future for multiple controlled agents while interpreting heterogeneous controls ranging from binary game buttons to continuous

robot commands. This task presents two complementary challenges. First, every agent-indexed action stream must control its intended visual entity independently. Second, the model must adapt the meaning of each control variable across different tasks, where the same tensor dimension may denote an attack, steering command, or robot command. We introduce LiveAct, centered on a Heterogeneous Action Interface (HAI) whose two components mirror this decomposition. To support independent control,

Agent-Aware Spatial Modulation uses language-driven spatial gates and agent-specific attention to bind each action stream to its intended visual agent. To accommodate heterogeneous controls, Schema-GuidedAction Adaptation describes each control variable in language and modulates its semantic token with the synchronized numerical value. We further transfer the complete interface to a blockwise causal generator through teacherforcing and rollout-aware distribution matching, enabling efficient streaming generation while retaining agent-specific control and action semantics. We instantiate the formulation on six multi-agent sources spanning games and multi-robot manipulation, and evaluate visual quality, action following, agent attribution, and latency. Across games and robot manipulation, LiveAct reaches 0.898 and 0.942 agent-attribution accuracy without agent locations or identity crops, and the causal generator attains 17.70 FPS at 512 × 448.

## 1. Introduction

Video world models [2, 11, 12, 16, 17, 20, 22, 41, 56, 57, 60] turn interaction into conditional generation [8, 9, 23, 58, 59]: given visual history and actions, they synthesize the observations that follow. Such models can provide scalable experience for game simulation, robot learning, planning, and interactive content creation [7, 53]. Existing systems increasingly support persistent and low-latency rollout, but most still expose a single aggregate player or cameracontrol stream [6, 44]. Such a generation paradigm is insufficient for inherently multi-agent environments, where independently controlled entities act concurrently and jointly determine the visual future. We therefore study multi-agent video world modeling: given an initial frame, a scene description, and a set of frame-level action streams for multiple agents, the model generates a shared visual future in which each stream controls its intended agent while all agents remain consistent with one another and the environment.

Existing interactive world models cover this setting only in part, and the shortfall lies in the action interface rather than the video backbone. Multi-agent control has relied on numerical action vectors, which keep exact values but typically reach the right entity through spatial priors supplied at inference time and leave control semantics implicit in task-specific dimensions [13, 30, 50]. Language-based actions transfer semantics across entities and tasks but give up the synchronized numerical values that continuous control requires [63]. Tab. 1 summarizes the landscape: no current interface is at once multi-agent, real-time, semantically grounded, and numerically exact. We ask whether one interface can satisfy all four requirements without inferencetime masks, boxes, poses, or reference images.

However, a unified action interface must still address two challenges: who each action stream should control in a shared visual scene, and what each control dimension means across heterogeneous tasks. These requirements lead to two key challenges.

(a) Independent control of multiple agents  
![](images/5af1b1255ee0c703f227c202283d45c4887596b67fa5ee6f83d0f40828aef9ff.jpg)

![](images/27e76c2fff13c419f31f5e6f4c33e7de5dc732685ba81042401d6f12e0c921c5.jpg)  
Figure 2. Two challenges for multi-agent action interfaces. (a) Streams that differ only by input order entangle fighters under index-only binding; LiveAct’s agent gate keeps each on its fighter. (b) Dimension j=7 changes meaning across tasks. A raw vector keeps the value but not the meaning, free text the reverse; LiveAct modulates each schema token by its value.

(1) maintaining independent control over multiple interacting agents. A model must keep each agent responsive to its own action stream while capturing interactions. The action tensors of each agent are injected sequentially at input, but do not ensure that their effects remain correctly attributed in the shared visual feature space. Commands may be assigned to the wrong agent or inadvertently drive multiple entities (Fig. 2a), especially when agents overlap, look similar, or jointly manipulate an object. Achieving independent control therefore requires agent-specific conditioning that separates command attribution.

(2) unifying action conditioning across heterogeneous control spaces. Action vectors do not share a universal semantic coordinate system: their dimensions, ordering, and meanings are defined by task-specific action spaces. As a result, the same action dimension or pattern may encode different behaviors (Fig. 2b). Directly conditioning on raw vectors therefore leads to action semantics being tied to task-specific dimensions and ordering. Language can expose what each control dimension means, but does not preserve the exact framesynchronized binary or continuous values needed for execution. Unified action conditioning thus requires separating what a control means from what value it takes, and representing both in a shared action interface.

Table 1. Native capability comparison. ✓: supported; ×: unsupported.
<table><tr><td>Method</td><td>Domain</td><td>Multi-Agent</td><td>Real-Time</td><td>Semantic</td><td>Numerical</td></tr><tr><td>LingBot-World 2.0 [6]</td><td>General visual worlds</td><td>X</td><td>V</td><td>√</td><td>√</td></tr><tr><td>Solaris [36]</td><td>Single (Minecraft)</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>MultiWorld [50]</td><td>Single (It Takes Two)</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>ActionParty [30]</td><td>Single (Melting Pot)</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Incantation [63]</td><td>Single (Elden Ring)</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>WanToFight [13]</td><td>Single (KOF &#x27;97)</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>LiveAct (Ours)</td><td>Diverse</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

LiveAct addresses these challenges with a Heterogeneous Action Interface (HAI), which factorizes action conditioning into semantic adaptation and agent-specific visual injection. For independent control, Agent-Aware Spatial Modulation derives an agent-specific spatial gate from the agent description, so that each branch selectively updates the visual tokens associated with its intended entity while attending only to the corresponding action stream. For heterogeneous action conditioning, Schema-Guided Action Adaptation converts each control dimension into a semantic token defined by its schema and injects the synchronized numerical value through affine modulation, decoupling control meaning from task-specific vector layouts. This provides a shared semantic interface for heterogeneous control spaces. Together, these two designs provide a common action representation while preserving agent-specific effects in the shared visual space. The resulting residuals are injected through zero-initialized projections, preserving the pretrained visual prior at initialization.

Beyond accurate control, interaction requires efficient incremental rollout without repeatedly regenerating the visual history. We therefore retain the complete HAI when transferring a bidirectional video teacher to a blockwise causal generator through Teacher Forcing for autoregressive (AR) Tuning and Distribution Matching Distillation (DMD) for Streaming Video [15, 48, 49, 54]. Causal temporal alignment maps raw actions to compressed video latents, while a global latent offset advances action windows together with the cached rollout. Causal generation thus accelerates a fixed control interface rather than redefining who acts or what each action means.

We evaluate LiveAct across six multi-agent environments spanning interactive games and robot manipulation, covering both discrete and continuous control spaces. Experiments assess visual fidelity, action adherence, agent attribution, and real-time efficiency. Our main contributions are:

• We formulate real-time multi-agent video world modeling under heterogeneous action spaces and identify two coupled challenges overlooked by existing action interfaces: binding each indexed action stream to its intended visual agent and jointly representing task-specific control semantics and synchronized numerical values in a shared interface.

• We propose LiveAct, a real-time multi-agent video world model built around a Heterogeneous Action Interface. HAI combines Agent-Aware Spatial Modulation for language-driven stream-to-agent binding with Schema-Guided Action Adaptation for value-preserving semantic control, and is retained during teacher-forced causal adaptation and rollout-aware distribution matching for realtime streaming generation.

• We build a dataset of 17,951 multi-agent trajectories, totaling 119.54 hours of raw video, and a 108-clip benchmark spanning four games and two robot-manipulation sources. LiveAct reaches 0.898 and 0.942 agentattribution accuracy on games and robots without locations, masks, or identity crops, and generates at 17.70 FPS at 512 × 448.

## 2. Related Work

Multi-agent video world models. Multi-agent video world models jointly predict an environment under multiple agents’ observations or controls, and must coordinate shared dynamics while preserving the identity of each condition stream. Solaris jointly models synchronized observations from multiple Minecraft agents [36]; MultiWorld combines a multi-agent condition module with a global state encoder across game and robot manipulation [50]; and Gamma-World uses permutation-equivalent agent encoding and sparse hub attention [20]. Complementary systems focus on binding independent controls to visual entities: ActionParty jointly predicts persistent subject-state tokens initialized from subject locations [30], WanToFight grounds locally causal keyboard streams with first-frame character references [13], and Incantation binds language actions to individual entities [63]. Across these designs, the central control problem is to preserve per-agent condition provenance and ensure that each action affects its intended entity. The Agent-Aware Spatial Modulation component of our HAI addresses this problem by conditioning the visual queries on each agent’s language description. without masks, boxes, coordinates, or reference images at inference time.

Action-controllable world models across domains. World models learn environment dynamics for imagined interaction [7]. Genie learns latent actions from unlabeled videos [3], DIAMOND uses diffusion for Atari and Counter-Strike dynamics [1], and GameNGen predicts DOOM frames from past observations and actions [44]. LingBot-World 2.0 couples camera poses and time-varying semantic prompts for open-ended interaction [6]. Realworld simulators extend action-conditioned prediction to diverse interactions [53] and continuous robot trajectories [61], while Incantation expresses per-entity actions in natural language to support semantic transfer [63]. Most interfaces choose between numerically precise but taskspecific vectors and language interfaces that make control semantics explicit. Within our HAI, Schema-Guided Action Adaptation combines the two: language identifies the taskdependent meaning of each control variable, while synchronized binary or continuous values retain frame-level control.

## 3. Method

Given an initial frame $I _ { 0 } ,$ a scene description $c ,$ and ordered action streams $\mathcal { A } = \{ A ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { a g } } }$ for $N _ { \mathrm { a g } }$ agents, where $A ^ { ( i ) } = [ a _ { \tau , j } ^ { ( i ) } ] \in \mathbb { R } ^ { T \times C }$ gives the value of control dimension j at raw frame $\tau \in \{ 1 , \ldots , T \}$ (raw frame 0 is $I _ { 0 } )$ , paired with agent descriptions $G \ = \ \{ g _ { i } \}$ and a control schema ${ \cal { S } } = \bar { \{ s _ { j } \} } _ { j = 1 } ^ { C }$ , multi-agent world modeling aims to synthesize the shared visual future in which each stream drives its own agent while all agents remain consistent with one another and with the scene. We write $\mathcal { C } ~ = ~ ( c , A , S , G )$ for the full condition. In this section, we first discuss the motivation and two failure modes of existing action interfaces in Sec. 3.1. We then introduce Agent-Aware Spatial Modulation in Sec. 3.2, which decides who responds to each stream, and Schema-Guided Action Adaptation in Sec. 3.3, what each control dimension means and what value it takes; Sec. 3.4 assembles them into the Heterogeneous Action Interface. Finally, Sec. 3.5 carries the same action-conditioning mechanism into causal streaming rollout. Fig. 3 summarizes this three-stage design.

## 3.1. Motivation

Although the indexed action tensor preserves the numerical trajectory of each stream, it does not establish the correspondences required to condition a shared visual representation. The stream index i separates $A ^ { ( i ) }$ from the other streams but provides no visual evidence associating it with the tokens of its intended agent. Likewise, the control index j identifies a coordinate only within a task-specific action space and therefore lacks a consistent interpretation across

tasks.

Cross-Agent Misattribution. Keeping the action streams separate preserves their provenance in action space, but does not establish their recipients in visual space. An agentindex embedding can distinguish $A ^ { ( i ) }$ from $A ^ { ( i ^ { \prime } ) }$ , yet contains no visual evidence linking index i to the appearance of agent i. Consequently, the command from one stream may be attributed to the wrong visual agent or inadvertently affect multiple agents. The missing correspondence must therefore be established on the query side of action conditioning: the visual features that read $A ^ { ( i ) }$ should be grounded in the description $g _ { i }$ of its corresponding agent.

Control-Semantics Collapse. An action vector contains C control dimensions, where the j-th coordinate provides the numerical value of one control at a given time step. However, the index $j$ itself carries no explicit semantic meaning beyond the definition of its task-specific action space. Compressing all $C$ dimensions into a single action embedding further obscures the identity of individual controls, making it difficult for the model to distinguish which control produces a particular effect. The action representation should therefore keep each control dimension distinct while jointly encoding its schema-defined semantics and frame-level numerical value.

These two requirements are complementary: expressive action tokens cannot resolve agent binding, while agentaware queries cannot recover missing control semantics. HAI therefore separates the two, using Agent-Aware Spatial Modulation for query-side binding and Schema-Guided Action Adaptation for key–value-side control representation, coupled through cross-attention.

## 3.2. Agent-Aware Spatial Modulation

To prevent cross-agent misattribution, we condition the visual queries on each agent’s description $g _ { i } ,$ , encouraging each action stream to affect its corresponding visual agent without requiring explicit spatial supervision or inferencetime spatial inputs.

Agent semantic representations. By construction, each action sequence $\boldsymbol { A } ^ { ( i ) }$ is paired with an agent description $g _ { i } .$ We encode this description with the frozen umT5 [35] text encoder and project it to an agent embedding:

$$
e _ { i } ^ { G } = \mathrm { P o o l } ( \mathrm { T } 5 ( g _ { i } ) ) \in \mathbb { R } ^ { 4 0 9 6 } , \quad q _ { i } = \mathrm { P r o j } _ { G } ( e _ { i } ^ { G } ) \in \mathbb { R } ^ { D } .\tag{1}
$$

Here Pool denotes a pooling layer and D is the backbone width. The resulting $q _ { i }$ provides the semantic identity associated with action stream $A ^ { ( i ) }$

Spatial query modulation. A global identity vector does not by itself indicate where the corresponding agent appears. For latent frame k, let $X _ { k } = [ x _ { k , 1 } , \dotsc , x _ { k , N } ] ^ { \top } \in$

![](images/8e1ff1c09475f7f1f5665ae5b61aaa557919371bf964e408e5943f1257a1a875.jpg)  
Figure 3. Overview of LiveAct. (a) Foundational Ability Acquisition trains a bidirectional teacher with a Heterogeneous Action Interface which integrates Agent-Aware Spatial Modulation and Schema-Guided Action Adaptation. (b) Teacher Forcing for AR Tuning transfers the same HAI to causal attention. (c) DMD for Streaming Video distills few-step streaming rollouts.

R $\cdot ^ { N \times D }$ contain the N patchified visual tokens. We compute a token-dependent gate from the visual feature, the agent representation, and their element-wise interaction:

$$
\begin{array} { r l } & { \lambda _ { k , n } ^ { ( i ) } = \mathrm { s i g m o i d } ( \mathrm { A g e n t G a t e } ( [ x _ { k , n } , q _ { i } , x _ { k , n } \odot q _ { i } ] ) ) , } \\ & { x _ { k , n } ^ { ( i ) } = x _ { k , n } + \lambda _ { k , n } ^ { ( i ) } \mathrm { A g e n t D e l t a } ( q _ { i } ) . } \end{array}\tag{2}
$$

Here, AgentDelta $\ u _ { \alpha } ( q _ { i } ) \in \mathrm { ~ { ~ \mathbb ~ R ~ } ~ } ^ { D }$ supplies an identitydependent feature direction, while the scalar $\lambda _ { k , n } ^ { ( i ) }$ controls its strength at each spatial token. Because $x _ { k , n }$ varies across space and time, the same agent description can induce different modulation strengths throughout the video. The gate is a soft, learned modulation rather than a segmentation mask or an explicitly supervised object assignment.

Stacking the modulated tokens gives $X _ { k } ^ { ( i ) }$ $[ x _ { k , 1 } ^ { ( i ) } , \ldots , x _ { k , N } ^ { ( i ) } ] ^ { \top }$ Each $X _ { k } ^ { ( i ) }$ is computed from the same unmodified state $X _ { k }$ and only preconditions the queries for agent $i ;$ it is not written into the backbone as a separate residual. The agent-conditioned states remain separate until cross-attention forms the action residuals.

## 3.3. Schema-Guided Action Adaptation

To prevent control-semantics collapse, we construct a separate action token for each control dimension j at each time step τ. Each token combines the schema description $s _ { j }$ which specifies what the control means, with its synchronized numerical value $a _ { \tau , j } ^ { ( i ) }$ , which specifies its current state. The resulting tokens are used as keys and values in crossattention.

Semantic control representations. Raw dimension indices cannot express task-specific control semantics. We therefore encode description $s _ { j }$ of every control variable with the same text encoder:

$$
e _ { j } ^ { S } = \operatorname { P o o l } ( \mathrm { T } 5 ( s _ { j } ) ) \in \mathbb { R } ^ { 4 0 9 6 } , \quad u _ { j } = \operatorname { P r o j } _ { S } ( e _ { j } ^ { S } ) \in \mathbb { R } ^ { D } .\tag{3}
$$

The resulting token turns coordinate $j$ into a named control such as an attack, steering command, or robot command. The schema description and numerical value are paired by control dimension, allowing a shared representation to accommodate heterogeneous control layouts within a single model.

Value-modulated control tokens. Language alone expresses action meaning but does not preserve the synchronized binary or continuous trajectory. For agent i, rawframe index τ, and control dimension $j ,$ a value encoder shared across control dimensions maps the scalar $a _ { \tau , j } ^ { ( i ) }$

affine vectors:

$$
\begin{array} { r l } & { ( \gamma _ { \tau , j } ^ { ( i ) } , \beta _ { \tau , j } ^ { ( i ) } ) = \mathrm { M L P } _ { \mathrm { m o d } } ( a _ { \tau , j } ^ { ( i ) } ) , } \\ & { \qquad h _ { \tau , j } ^ { ( i ) } = u _ { j } \odot ( 1 + \gamma _ { \tau , j } ^ { ( i ) } ) + \beta _ { \tau , j } ^ { ( i ) } . } \end{array}\tag{4}
$$

Thus, $u _ { j }$ specifies what a control variable represents, while the synchronized value determines its current modulation. Unlike early aggregation, this construction retains $C$ tokens per time step, allowing visual queries to select individual controls. We add learned time and agent embeddings,

$$
\begin{array} { r } { \widetilde { h } _ { \tau , j } ^ { ( i ) } = h _ { \tau , j } ^ { ( i ) } + e _ { \tau } ^ { \mathrm { t i m e } } + e _ { i } ^ { \mathrm { i d x } } , } \end{array}\tag{5}
$$

and collect these tokens into $\widetilde { H } ^ { ( i ) } \in \mathbb { R } ^ { T \times C \times D }$ , retaining a separate token for every control dimension at every time step before temporal windowing.

## 3.4. Assembling the Heterogeneous Action Interface

After constructing the agent-conditioned visual queries and schema-guided action tokens, we assemble the HAI through per-agent cross-attention. For latent frame k and agent $i ,$ $X _ { k } ^ { ( i ) } \in \mathbb { R } ^ { N \times D }$ queries the aligned action window $H _ { k } ^ { \left( i \right) } \ \in$ $\mathbb { R } ^ { \tilde { M } \times D } \ ( \mathrm { E q . \ 8 } )$ , where r is the temporal compression ratio of the video VAE and $M = r C$ in the one-latent window used by our main model:

$$
\Delta X _ { k } ^ { ( i ) } = \mathrm { C r o s s A t t n } ( X _ { k } ^ { ( i ) } , H _ { k } ^ { ( i ) } ) .\tag{6}
$$

The visual tokens serve as queries and the action tokens as keys and values, producing one control residual per agent. We insert this layer after text cross-attention and before the feed-forward sublayer in each transformer block. The shared visual state $X _ { k }$ of latent frame k is then updated as

$$
\widehat { X } _ { k } = X _ { k } + \sum _ { i = 1 } ^ { N _ { \mathrm { a g } } } \Delta X _ { k } ^ { ( i ) } .\tag{7}
$$

All agents read from the same shared visual state $X _ { k } ,$ , while their control residuals are computed independently before aggregation. We zero-initialize each cross-attention output projection, so that the augmented model initially preserves the pretrained backbone behavior.

## 3.5. Efficient Causal Rollout

To enable streaming rollouts, we convert the bidirectional diffusion teacher into a causal generator while addressing action alignment and exposure to self-generated history. Teacher Forcing for AR Tuning first adapts the model to causal attention, followed by DMD on autoregressive rollouts.

Causally aligned action conditioning. Each video latent should read only the controls of its own physical interval.

Since the VAE encodes $I _ { 0 }$ separately and compresses every r raw frames into one latent, future latent $k \geq 1$ covers raw frames $\mathcal { T } _ { k } = \{ r ( k { - } 1 ) { + } 1 , \ldots , r k \}$ , and the action memory of agent i stacks the control tokens in this interval:

$$
H _ { k } ^ { ( i ) } = \left[ \widetilde { h } _ { \tau , j } ^ { ( i ) } \right] _ { \tau \in \mathcal { T } _ { k } , 1 \le j \le C } \in \mathbb { R } ^ { r C \times D } .\tag{8}
$$

In the causal generator, k is the global latent index rather than the position within the current chunk, so every latent keeps its physical controls as the KV cache advances.

Teacher Forcing for AR Tuning. After obtaining a bidirectional model, this stage then transfers its parameters to a blockwise causal generator [62]. Each target chunk is denoised using only clean preceding chunks, with a causal mask. The HAI remains at the same transformer locations and receives the same aligned conditions, so this stage isolates the change in temporal information flow from errors caused by self-generated history. At inference, preceding chunks are retained in the KV cache and reused when generating the next one.

DMD for Streaming Video. Teacher forcing still trains on clean history, leaving an exposure gap at deployment. DMD addresses this gap by optimizing autoregressive rollouts [15, 54]. Starting from the initial latent $z _ { \mathrm { 0 } }$ , the VAE encoding of $I _ { 0 } ,$ the generator produces $\hat { z } = \mathcal G _ { \boldsymbol \theta } ( z _ { 0 } , \mathcal C )$ while reusing its self-generated KV cache. The frozen bidirectional teacher serves as the real score, while a trainable bidirectional fake score initialized from the same teacher tracks the generator distribution. At diffusion timestep t, their denoised predictions from the same noised sample $\hat { z } _ { t }$ and condition C define the unnormalized distribution-matching gradient $\delta _ { \mathrm { D M D } } = \mu _ { \mathrm { f a k e } , \phi } ( \hat { z } _ { t } , t , \mathcal { C } ) - \mu _ { \mathrm { r e a l } } ( \hat { z } _ { t } , t , \mathcal { C } )$ . Backpropagating the normalized gradient through $\mathcal { G } _ { \theta }$ exposes the causal model to its own rollout history while retaining fewstep chunk-wise generation.

## 4. Experiments

## 4.1. Data Collection

As shown in Fig. 4, we assemble a six-source dataset of 17,951 independent multi-agent trajectories from four competitive multiplayer games and collaborative multi-robotic sources, totaling 119.54 hours of raw video. Each sample contains an initial frame, a scene description, synchronized per-agent action streams, agent descriptions, and a taskspecific control schema, with binary controls for games and continuous controls for robots. For evaluation, we build a benchmark of 108 clips drawn from trajectories held ou from training, with one 81-frame clip per trajectory; all comparisons and ablations in this section are conducted on this benchmark. Details are provided in Appendix.

![](images/b0ee0f9b2d8806b931cd475c3c605349c21a2b2794debc683ace63cf18ecfdb3.jpg)  
Figure 4. Data curation pipeline. Game and robotic episodes are curated into aligned samples containing a scene prompt, per-agent action streams, agent descriptions, and task-specific schemas.

## 4.2. Implementation details

We build LiveAct on Wan2.2-TI2V-5B [45], which contains 30 transformer blocks with a hidden width of $D = 3 0 7 2$ and 24 attention heads; its VAE compresses every r = 4 raw frames into one latent. We insert the HAI after text cross-attention and before the feed-forward sublayer in every block and fine-tune the full DiT together with the HAI, while the umT5 encoder, which encodes agent and schema descriptions, and the VAE remain frozen. Each agent is represented by C = 12 schema-aligned control dimensions. Stage 1 trains the bidirectional teacher for 10k steps with a learning rate of $1 0 ^ { - 4 }$ and a global batch size of 16. Stage 2 then trains the causal generator for 6k steps with a learning rate of $2 \times 1 0 ^ { - 6 }$ , and Stage 3 runs 700 iterations with learning rates of $2 \times 1 0 ^ { - 6 }$ for the generator and 4 × 10<sup>−7</sup> for the fake score; both causal stages use a global batch size of 24 with FSDP. At inference, the teacher uses 50 denoising steps, and the causal generator predicts four latent frames per chunk with four denoising evaluations at timesteps [1000, 750, 500, 250].

## 4.3. Comparison with Baselines

Baselines. We evaluate LiveAct on two complementary tracks—Multi-Agent Video Games and Multi-Robot Manipulation—under the same protocol for visual quality, action following, and agent attribution. We compare against representative interactive and multi-agent video world models covering three control paradigms: language-serialized actions (LingBot-World 2.0 [6], Incantation [63]), per-agent numerical controls (MultiWorld [50], ActionParty [30]), and reference-guided method (WanToFight-style [13]). Metrics. Following MultiWorld, we report FVD, LPIPS,

PSNR, and SSIM for reconstruction quality. Because frame-paired metrics can favor static or averaged predictions, we complement them with Motion Magnitude Error (MME) and an Aesthetic score using the LAION [37] improved predictor. For a more comprehensive control evaluation, we trained inverse dynamics models only on real clips: they recover the executed controls from a generated video, yielding Act-F1 over discrete game events and a normalized error (NRMSE) on continuous robot commands. To evaluate agent attribution, Attr-Acc measures whether the recovered per-agent controls match the commanded streams better under the intended agent assignment than under pairwise agent swaps; higher is better, with a chance level of 0.5. See appendix for more details.

Quantitative Results. Shown in Tab. 2, among the four non-reference-conditioned methods, LingBot-World 2.0 gives the best game FVD and Aesthetic score but follows actions weakly; MultiWorld retains competitive robot PSNR and SSIM but has low agent attribution; Incantation reaches comparable game Act-F1 yet suffers severe visual drift on robots; and ActionParty provides the best robot reconstruction but the weakest action following. WanToFight-style is the strongest baseline overall, combining strong reconstruction with 0.886 robot Attr-Acc using identity crops, although its game Attr-Acc remains 0.552. LiveAct obtains the best results among the evaluated methods on both tracks, with FVDs of 338.5 and 270.2, game Act-F1 of 0.594, robot NRMSE of 0.558, and Attr-Acc of 0.898 and 0.942. It also achieves the best MME and obtains the highest Aesthetic score.

Qualitative Results. As shown in Fig. 5, the languageserialized models lose the robot scene (LingBot-World 2.0,

![](images/0967cb282f9b1cc762621615da7d153bc0227e5f00cdde93e62e8c3555d6e9fd.jpg)  
Figure 5. Qualitative comparison on shared initial frames and action streams. Columns show ground truth, the five baselines, and LiveAct; Rows are frames sampled at different durations.

Table 2. Main comparison across multi-agent video games and multi-robot manipulation. Best results are bolded.<sup>a</sup>Native MultiWorld uses synchronized multi-view observations; we have set it to a fixed same perspective. <sup>b</sup>The code is not released; this is our reproduction.
<table><tr><td rowspan="2">Method</td><td colspan="4">Reconstruction</td><td colspan="2">Control</td><td colspan="2">Diagnostics</td></tr><tr><td>FVD↓</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>Act-F1 ↑/NRMSE↓</td><td>Attr-Acc ↑</td><td>MME↓</td><td>Aesthetic ↑</td></tr><tr><td>Multi-Agent Video Games</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LingBot-World 2.0 [6]</td><td>1256.6</td><td>0.322</td><td>13.25</td><td>0.490</td><td>0.146</td><td>0.553</td><td>0.57</td><td>5.093</td></tr><tr><td>MultiWorld (1-view)ª [50]</td><td>2661.4</td><td>0.719</td><td>10.40</td><td>0.125</td><td>0.163</td><td>0.548</td><td>2.67</td><td>4.059</td></tr><tr><td>Incantation [63]</td><td>1989.5</td><td>0.593</td><td>10.45</td><td>0.226</td><td>0.162</td><td>0.534</td><td>8.77</td><td>4.539</td></tr><tr><td>ActionParty [30]</td><td>1672.5</td><td>0.272</td><td>14.56</td><td>0.467</td><td>0.086</td><td>0.547</td><td>0.47</td><td>4.974</td></tr><tr><td>WanToFight-styleb [13]</td><td>563.6</td><td>0.136</td><td>17.67</td><td>0.671</td><td>0.241</td><td>0.552</td><td>0.23</td><td>5.192</td></tr><tr><td>LiveAct</td><td>338.5</td><td>0.086</td><td>20.17</td><td>0.743</td><td>0.594</td><td>0.898</td><td>0.15</td><td>5.207</td></tr><tr><td>Multi-Robot Manipulation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LingBot-World 2.0 [6]</td><td>2732.5</td><td>0.603</td><td>7.13</td><td>0.464</td><td>1.106</td><td>0.665</td><td>4.20</td><td>3.570</td></tr><tr><td>MultiWorld (1-view)ª [50]</td><td>1748.9</td><td>0.276</td><td>15.36</td><td>0.797</td><td>1.099</td><td>0.357</td><td>1.90</td><td>4.054</td></tr><tr><td>Incantation [63]</td><td>3053.3</td><td>0.820</td><td>3.93</td><td>0.185</td><td>1.119</td><td>0.459</td><td>17.37</td><td>3.710</td></tr><tr><td>ActionParty [30]</td><td>1661.7</td><td>0.267</td><td>15.42</td><td>0.816</td><td>1.142</td><td>0.504</td><td>1.90</td><td>4.035</td></tr><tr><td>WanToFight-styleb [13]</td><td>402.9</td><td>0.188</td><td>17.32</td><td>0.831</td><td>0.758</td><td>0.886</td><td>0.42</td><td>4.037</td></tr><tr><td>LiveAct</td><td>270.2</td><td>0.106</td><td>20.51</td><td>0.871</td><td>0.558</td><td>0.942</td><td>0.42</td><td>4.085</td></tr></table>

the kick and the opponent’s response at the same frames as the ground truth and carries the arm along the ground-truth path, showing that each stream is both executed and bound to the intended agent.

Incantation). On the game track, they keep the stage but not the commanded attacks. MultiWorld and ActionParty keep the coarse layout but blur the agents: the fighters and arms blur into faint, duplicated silhouettes. WanToFight-style preserves both scenes and agents and executes the kick, but misses the second agent’s command. LiveAct reproduces

Table 3. Quantitative ablation study.
<table><tr><td>Variant</td><td></td><td>Game Act-F1 ↑ Game Attr-Acc ↑ Robot NRMSE ↓ Robot Attr-Acc ↑</td><td></td><td></td></tr><tr><td>Full LiveAct</td><td>0.594</td><td>0.898</td><td>0.558</td><td>0.942</td></tr><tr><td>w/o AASM</td><td>0.172</td><td>0.592</td><td>0.931</td><td>0.810</td></tr><tr><td>w/o SGAA</td><td>0.225</td><td>0.603</td><td>1.120</td><td>0.484</td></tr></table>

![](images/ed3942ca6fa6633347d4faee23915c00d758d453d42978afeb78c1fe15f4800b.jpg)  
Figure 6. Qualitative ablation study.

## 4.4. Ablation Studies

Effectiveness of Agent-Aware Spatial Modulation (AASM). Removing it keeps the action tokens intact but makes the queries identical for both agents, and control collapses in both domains (Tab. 3): game Act-F1 drops from 0.594 to 0.172 and Attr-Acc from 0.898 to 0.592, while robot NRMSE rises from 0.558 to 0.931. In Fig. 6, the two streams no longer stay on their own agents, which is the cross-agent misattribution of Sec. 3.1: binding cannot be recovered from the action side alone.

Effectiveness of Schema-Guided Action Adaptation (SGAA). Replacing it with a conventional flat action encoder, an MLP that maps each frame’s full action vector to a single token, keeps agent modulation but reintroduces control-semantics collapse in visual, and performance again drops in both domains: game Act-F1 falls to 0.225 and Attr-Acc to 0.603, while robot NRMSE rises to 1.120 and Attr-Acc falls to 0.484. Attribution degrades even though agent modulation is retained, indicating that the queries can only route controls they can interpret; the two components are therefore complementary. Finer-grained SGAA ablations are provided in the Appendix.

Effectiveness of real-time distillation. Stage 1 denoises the full sequence and cannot stream. Stage 2 enables blockwise causal generation but remains slow at 1.66 FPS. Stage 3 reduces each chunk to four denoising steps and reaches 17.70 FPS, 10.7× faster than Stage 2. On games, Act-F1 rises from 0.539 to 0.594 and Attr-Acc from 0.844 to 0.898. Autoregressive generation is better suited to these discrete actions: each frame is conditioned only on past context and the controls aligned to that frame, rather than denoising the full sequence jointly. On robots, NRMSE rises from 0.430 to 0.558 and Attr-Acc falls from 0.969 to 0.942. Per-stage settings and results are in the Appendix.

## 5. Conclusion

We presented LiveAct, a multi-agent video world model for heterogeneous control across games and robot manipulation. Its Heterogeneous Action Interface combines Agent-Aware Spatial Modulation, which binds each indexed action stream to its visual agent through language-conditioned queries, with Schema-Guided Action Adaptation, which couples each control’s language-defined semantics with its synchronized numerical value. Per-agent cross-attention integrates the two within a shared visual representation. Teacher Forcing for AR Tuning and DMD for Streaming Video transfer the interface unchanged to few-step causal rollout. Across six sources, LiveAct achieves best reconstruction and control and reaches 0.898 and 0.942 agentattribution accuracy without location, mask, or identity input, while its causal generator runs at 17.70 FPS at 512 × 448.

# LiveAct: Towards Real-Time Multi-Agent World Models with Heterogeneous Action Interfaces

Supplementary Material

## Appendix

A. Additional Related Work S1   
B. Method Details S1   
B.1. Problem Formulation . S1   
B.2. Frame–Latent–Action Alignment S2   
B.3. Causal Condition Propagation S2   
B.4. Causal Training Objectives S2   
C. Dataset Details S2   
D. Implementation Details S3   
D.1. Backbone and Conditioner Placement . S3   
D.2. Training and Inference Configuration S4   
E. Evaluation Details S4   
E.1. Evaluation Protocol . S4   
E.2. Baseline Configurations S4   
E.3. Metrics S5   
F. Additional Results S6   
F.1. Three-Stage Causal Transfer . S6   
F.2. Detailed Ablations S6   
F.3. Spatial Localization of Agent-Specific   
Residuals S7   
F.4. Additional Qualitative Comparisons S7   
F.5. Additional LiveAct Showcases S8   
F.6. Long-Horizon Rollout Showcase . S8   
Supplementary video. Please see the accompanying   
demo.mp4 for additional video results.

## A. Additional Related Work

Structured conditioning in video diffusion. Video diffusion models provide temporally coherent generative priors [10], while diffusion transformers and open video foundation models offer scalable token backbones [5, 14, 18, 19, 21, 24, 25, 28, 32–34, 38, 39, 45–47, 49, 51, 52, 58, 59]. Control signals are commonly injected through concatenation, adaptive modulation, or cross-attention. Our distinction is representational rather than merely architectural: a padded action vector is decomposed into per-dimension control tokens, each token carries a task-specific semantic description, and its time-varying scalar modulates rather than replaces that semantic representation. Agent-Aware Spatial Modulation then keeps each stream’s provenance intact until cross-attention.

Causal video generation and diffusion distillation. Streaming video generation restricts attention to past chunks and caches their keys and values. Self Forcing reduces exposure bias by training on autoregressive model rollouts [15, 40, 64]; Causal Forcing motivates causal teacher initialization before few-step distillation [62]; and distribution matching distillation expresses the generator update through real- and fake-distribution scores [54]. Recent multi-agent systems also combine bidirectional teachers, causal students, and rollout-aware distillation [13, 20, 36]. We adopt this training paradigm to carry the complete HAI into efficient rollout while keeping physical action intervals aligned.

## B. Method Details

## B.1. Problem Formulation

We follow the notation of Sec. 3: $N _ { \mathrm { a g } }$ agents with ordered action streams $\mathcal { A } = \{ A ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { a g } } } , A ^ { ( i ) } = [ a _ { \tau , j } ^ { ( i ) } ] \in \mathbb { R } ^ { T \times C }$ , a control schema ${ \cal { S } } = \{ s _ { j } \} _ { j = 1 } ^ { C }$ , and agent descriptions $G =$ $\{ g _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { a g } } }$ . The interface enforces

$$
\begin{array} { r l } & { \quad a _ { \tau , j } ^ { ( i ) } \longleftrightarrow ( g _ { i } , s _ { j } ) , } \\ & { \quad i \in \{ 1 , \ldots , N _ { \mathrm { a g } } \} , \quad j \in \{ 1 , \ldots , C \} , } \end{array}\tag{9}
$$

so each numerical control dimension is paired with an agent identity and a task-specific semantic description; the controls of every source are padded to a common width of $C = 1 2$ , while S preserves source-specific control semantics.

Let $z _ { \mathrm { 0 } }$ be the VAE encoding of $I _ { 0 }$ and $z ^ { 1 : F - 1 }$ the future video latents over $F$ latent frames in total. The DiT backbone patchifies these latents into visual tokens $X \in$ $\mathbb { R } ^ { F N \times D }$ , where each latent frame contributes N spatial tokens of hidden dimension $D$ . The desired conditional distribution is

$$
p _ { \theta } ( z ^ { 1 : F - 1 } \mid z _ { 0 } , c , \mathcal { A } , S , G ) .\tag{10}
$$

The bidirectional teacher models all future tokens jointly. The causal generator instead partitions future latents into chunks $\boldsymbol { B } _ { 1 } , \dots , \boldsymbol { B } _ { K }$ and factorizes

$$
p _ { \theta } \big ( z ^ { 1 : F - 1 } \mid z _ { 0 } , \mathcal { C } \big ) = \prod _ { b = 1 } ^ { K } p _ { \theta } \big ( z ^ { \mathcal { B } _ { b } } \mid z ^ { < \mathcal { B } _ { b } } , z _ { 0 } , \mathcal { C } \big ) ,\tag{11}
$$

where $\mathcal { C } = ( c , \mathcal { A } , S , G )$ $\operatorname { E q . 9 }$ makes both agent identity and control semantics explicit; the main-text Method realizes this interface and keeps it under causal rollout.

## B.2. Frame–Latent–Action Alignment

The Wan VAE uses temporal compression ratio $r = 4$ with a separately encoded first frame. Action sequences are truncated or padded to $T _ { \mathrm { n e e d } } = ( F - 1 ) r + 1$ . For a window of L latent frames, latent frame zero repeats the first action $r L$ times, while future latent $k \geq 1$ receives

$$
\{ a _ { \tau , j } ^ { ( i ) } : \tau _ { k } \leq \tau \leq \tau _ { k } + r L - 1 \} , \qquad \tau _ { k } = 1 + r ( k - 1 ) ,\tag{12}
$$

with final-value padding at the sequence boundary; for the primary setting $L \ = \ 1$ this reduces to the interval $\mathcal { T } _ { k }$ of Eq. 8. The time and control-dimension axes are retained until each local window is formed and then flattened for action attention.

For an 81-frame clip, raw indices 0:80 map to 21 latent frames. Future latent $k \in \{ 1 , \ldots , 2 0 \}$ uses actions beginning at $1 + 4 ( k - 1 )$ ), and chunk $b \in \{ 1 , \ldots , 5 \}$ contains future latents $4 ( b - 1 ) + 1 { : } 4 b$ . During causal teacher forcing, the clean-context/noisy-target shift advances the action interval by one latent. During DMD training, the global latent index is applied not only to the generator’s rollout but also to both score models, so all three read the same physical interval. Self-forced score evaluation restores the clean initial latent and excludes it from both DMD and fake-score losses.

## B.3. Causal Condition Propagation

In the causal generator, the HAI operates only on the real video tokens of the target chunks and returns zero residuals for sequence-padding tokens. Under the cleancontext/noisy-target layout used by teacher forcing, actions are injected only into the noisy target half; the clean history remains an unmodified context. During rollout, the clean initial latent is cached without action injection before subsequent target chunks receive their aligned streams. This separation prevents context tokens from being optimized as future action responses.

Checkpoint transfer is performed only between networks with matching HAI architectures and conditioning dimensions.

## B.4. Causal Training Objectives

Foundational Ability Acquisition. For clean future latents $z ^ { 1 : F - 1 }$ , Gaussian noise ϵ, and sampled time t, let $z _ { t } ^ { 1 : F - 1 } = \alpha _ { t } { z } ^ { 1 : F - 1 } + \sigma _ { t } \epsilon$ . The bidirectional model predicts the velocity target $v ^ { * } ( z ^ { 1 : F - 1 } , \epsilon , t )$ under the initial latent, scene, and action conditions:

$$
\mathcal { L } _ { \mathrm { b i } } = \mathbb { E } \left[ w ( t ) \left. v _ { \theta _ { b } } ( z _ { t } ^ { 1 : F - 1 } , z _ { 0 } , t , \mathcal { C } ) - v ^ { * } \right. _ { 2 } ^ { 2 } \right] .\tag{13}
$$

It provides both the action-conditioned prior and initialization for the causal generator and score networks.

Teacher Forcing for AR Tuning. Given the $F ~ = ~ 2 1$ clean latents $z ^ { 0 : \widecheck { F } - 1 }$ , the training pair uses $z ^ { 0 : F - 2 }$ as context and $z ^ { 1 : F - 1 }$ as targets. The initial frame is conditioning only and is excluded from the loss. We add noise to the $F - 1 = 2 0$ targets, with every group of four latents sharing a timestep to match chunks:

$$
\mathcal { L } _ { \mathrm { T F } } = \mathbb { E } \left[ w ( t ) \left| \left| v _ { \theta _ { c } } ( z _ { t } ^ { 1 : F - 1 } , z ^ { 0 : F - 2 } , t , \mathcal { C } _ { \mathrm { s h i f t } } ) - v ^ { * } \right| \right| _ { 2 } ^ { 2 } \right] .\tag{14}
$$

Here $\mathcal { C } _ { \mathrm { s h i f t } }$ applies the four-raw-frame offset induced by the context–target shift.

DMD for Streaming Video. The trainable causal generator $\mathcal { G } _ { \theta }$ , which unrolls the causal network $v _ { \theta _ { c } }$ with the fourstep schedule, initializes its KV cache from the clean initial latent and generates five future chunks from noise without reading future ground-truth latents. The real score $\mu _ { \mathrm { r e a l } }$ and the fake score $\mu _ { \mathrm { f a k e } , \phi }$ of the main paper receive the same initial frame, scene text, control schema, agent descriptions, and cropped action window as the generator. For $\hat { z } = \mathcal G _ { \boldsymbol \theta } ( z _ { 0 } , \mathcal C )$ and its noised version $\hat { z } _ { t } ^ { \phantom { } } .$ , we normalize the unnormalized distribution-matching gradient $\delta _ { \mathrm { D M D } }$ of the main paper as

$$
\bar { \delta } _ { \mathrm { D M D } } = \frac { \delta _ { \mathrm { D M D } } } { \mathrm { m e a n } ( | \hat { z } - \mu _ { \mathrm { r e a l } } ( \hat { z } _ { t } , t , \mathcal { C } ) | ) + \varepsilon } ,\tag{15}
$$

with a small constant $\varepsilon .$ We then optimize the generator through stop-gradient regression,

$$
\mathcal { L } _ { \mathrm { D M D } } = \frac { 1 } { 2 } \left. \hat { z } - \mathrm { s g } ( \hat { z } - \bar { \delta } _ { \mathrm { D M D } } ) \right. _ { 2 } ^ { 2 } .\tag{16}
$$

The fake score is alternately trained by flow denoising regression on detached samples from the current generator. The default schedule updates it every iteration and updates the generator once every five iterations. We use conditionalonly score passes to avoid an additional unconditional forward.

## C. Dataset Details

Composition. Tab. S1 summarizes the six sources and Fig. S1 shows frames from each of them. The dataset contains 17,951 independent multi-agent trajectories, 119.54 hours of raw video, with two to four concurrently controlled agents: the two fighting games and RoboTwin are two-agent, Mega Bomberman and Micro Machines include scenes with two to four agents, and RoboFactory includes two- and three-arm scenes. Every trajectory pairs a shared visual observation with one synchronized action stream per agent, so the number of streams equals the number of agents in the scene.

![](images/e97b1c79242ccf2b344d47683d026e056545fbc01f19e2aef67450c7363439a8.jpg)  
Figure S1. Showcase of our dataset. Frames from the six sources, covering versus fighting, shared-map competition, shared-track racing, and multi-arm manipulation.

Sources. The game portion is collected through the retro game platform [27, 29]. These games cover versus fighting, shared-map competition, and shared-track racing. Each trajectory begins from a curated emulator state and records synchronized visual observations and per-agent controller traces under scripted policies at the native 60 fps. The policy drives each agent’s controller independently: it samples one allowed button or a no-op from a game-specific distribution, holds it for a fixed number of frames, and then releases it. We exclude menus, transitions, and incomplete rollouts, yielding 12,860 trajectories of 30 s each.

The robot portion combines RoboTwin [26] and Robo-Factory [31]. RoboTwin contributes 2,532 episodes across 50 multi-arm tasks: 280 self-collected simulations of 14 tasks at 30 fps and 2,252 augmented episodes covering all 50 tasks from the randomized LeRobot [4] collection at 50 fps. RoboFactory contributes 2,559 episodes of cooperative cube stacking at 30 fps, retaining both successful and unsuccessful trajectories and including two- and threearm scenes. In both sources, concurrent continuous action streams are aligned with a shared visual observation over the full episode, yielding 5,091 robot trajectories.

Scale and training windows. The raw recordings total 24.71 M frames (games 23.15 M, 107.17 h; robots 1.56 M, 12.37 h), or 119.54 hours at the native frame rates (Tab. S1). Individual trajectories are much longer than a training sample, whereas the model consumes 81-frame windows (Tab. S2). Because generation is conditioned on an initial frame, a window may start at any frame of a trajectory, with its first frame serving as the conditioning image.

Benchmark. The 108-clip benchmark of the main paper is kept separate from the training data: it takes one 81-frame clip from each of 108 trajectories that are never used for training, and contains 78 game and 30 robot clips.

## D. Implementation Details

## D.1. Backbone and Conditioner Placement

Beyond the placement described in the main paper, the HAI leaves the pretrained conditioning pathways of Wan2.2- TI2V-5B [45] unchanged. The initial image latent is fused into the 48-channel video representation, and a separated timestep assigns zero to this clean latent and the sampled diffusion timestep to future latents; scene text retains the pretrained text cross-attention, diffusion time continues to modulate AdaLN, and video positions retain 3D RoPE. The bidirectional teacher, the causal generator, and both score networks share the same HAI placement and structure.

Table S1. Dataset statistics for four multi-agent games and two multi-robot manipulation sources. Frame counts and durations refer to the raw recordings at their native frame rates (RoboTwin: 30 fps for self-collected and 50 fps for augmented episodes). Training and evaluation use 81-frame clips drawn from these trajectories.
<table><tr><td colspan="7">Street Fighter II Fatal Fury 2 Mega Bomberman</td></tr><tr><td></td><td>5460</td><td>3200</td><td>2050</td><td>Micro Machines 2150</td><td>RoboTwin 2532</td><td>RoboFactory 2559</td></tr><tr><td>Trajectories Agents</td><td>2</td><td>2</td><td>2-4</td><td>2-4</td><td>2</td><td>17951</td></tr><tr><td></td><td>binary</td><td>binary</td><td>binary binary</td><td>continuous</td><td>2-3</td><td>2-4 mixed</td></tr><tr><td>Action type</td><td>60</td><td>60</td><td>60</td><td>30/ 50</td><td>continuous 30</td><td></td></tr><tr><td>Frame rate (fps)</td><td>9.83</td><td>5.76</td><td>3.69</td><td>60 3.87</td><td>0.91</td><td>24.71</td></tr><tr><td>Raw frames (M)</td><td>45.50</td><td>26.67</td><td>17.08</td><td>17.92</td><td>0.65 3.93</td><td>119.54</td></tr><tr><td>Raw duration (h)</td><td></td><td></td><td></td><td></td><td></td><td>8.44</td></tr></table>

Control-schema and agent-description embeddings are precomputed with the frozen text encoder and reused throughout training; the pooling operator Pool of the main paper averages the encoder outputs over the non-padding tokens of each description.

## D.2. Training and Inference Configuration

Tab. S2 collects the backbone, data, and optimization settings of the three stages in Fig. 3, including those stated in the main paper. Beyond them, the primary configuration uses description-based, value-modulated control tokens, and all stages share a flow-matching parameterization with 1000 training timesteps and a timestep shift of 5.0. Foundational Ability Acquisition uses AdamW with $\beta ~ = ~ ( 0 . 9 , 0 . 9 9 9 )$ and a constant learning rate, whereas the two causal stages use $\beta ~ = ~ ( 0 , 0 . 9 9 9 )$ and clip gradient norms at 10. DMD for Streaming Video maintains an EMA of the generator with decay 0.99 from iteration 200 onward, and the EMA weights are used at inference.

## E. Evaluation Details

## E.1. Evaluation Protocol

Both evaluation tracks of Sec. 4.3 ask the same three questions. First, does Agent-Aware Spatial Modulation assign each indexed action stream to the correct visual agent in the jointly generated scene? Second, does Schema-Guided Action Adaptation interpret heterogeneous controls consistently across games and embodied tasks? Third, how much control and visual quality survive causal transfer, and what latency is gained? The first two questions evaluate the complementary components of the HAI.

Every reported method is evaluated on the benchmark of App. C with one 81-frame prediction at seed 0 per condition. Frame-paired metrics exclude the conditioning frame and average 16 uniformly sampled frames from frames 1– 80. Pixel metrics use $2 5 \bar { 6 } ^ { 2 }$ inputs, while FVD uses canonical Kinetics-400 I3D pre-softmax features at $2 2 4 ^ { 2 }$ . Since every clip starts from a fixed initial frame and a fixed set of action streams in a deterministic environment, frame-paired errors measure deviation from the commanded reference future; the control metrics separately assess execution and attribution. Causal efficiency is evaluated separately because published throughput values use different backbones, resolutions, hardware, and decoding pipelines. Architecture ablations use the same data, optimization budget, and evaluation seed.

## E.2. Baseline Configurations

All five baselines are evaluated on the same clips and protocol. LingBot-World 2.0, MultiWorld, and Incantation do not release training code, so we run their released checkpoints with their original model configurations and conditioning interfaces, without further training. ActionParty does not release trained checkpoints, so we train it with the same budget on our dataset. The WanToFight code is not publicly available, so we train our own reimplementation on the same mixture and denote it WanToFight-style.

• LingBot-World 2.0 [6] serializes all agents into one time-varying prompt stream, including explicit continuous robot values.

• MultiWorld [50] retains its Multi-Agent Condition Module, while the Global State Encoder receives only the same single initial view available to other methods.

• Incantation [63] uses per-frame, per-entity language actions, with robot values represented explicitly in text.

• ActionParty [30] jointly predicts persistent subject-state tokens and video tokens. It receives initial agent locations as required by its binding mechanism; we extend its discrete action projection to continuous robot values and train the resulting model.

• WanToFight-style [13] combines per-agent locally causal action injection with first-frame identity crops and reference-conditioned Player Association. The original code is not publicly available; our reported results are obtained by training this reproduction.

• LiveAct uses per-agent schema-numerical controls without locations, masks, boxes, or identity crops.

Table S2. Training and inference configuration of LiveAct. Stages 2 and 3 are trained with FSDP, and Stage 3 samples at timesteps [1000, 750, 500, 250] with the EMA weights; “–” marks settings that do not apply.
<table><tr><td colspan="4">Model and data</td></tr><tr><td>Backbone</td><td colspan="3">Wan2.2-TI2V-5B; 30 blocks, width 3072, 24 heads;  $\mathrm { V A E } 4 \times 1 6 \times 1 6$  Full DiT and HAI; umT5 encoder and VAE frozen</td></tr><tr><td>Trainable Clip</td><td colspan="3">81 frames at  $5 1 2 \times 4 4 8$  , 16 fps; 1 conditioning and 20 future latents</td></tr><tr><td>Frame sampling</td><td colspan="3">Games: consecutive native frames; robots: uniform over the episode</td></tr><tr><td>Actions</td><td colspan="3">12 description-modulated controls per agent; one-latent window</td></tr><tr><td>Conditioner</td><td colspan="3">Text projections 4096 → 1024 → 3072; value/gate/delta bottleneck 768</td></tr><tr><td>Training and inference</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td></tr><tr><td></td><td>Foundation</td><td>Teacher Forcing</td><td>DMD</td></tr><tr><td></td><td colspan="3">AdamW, weight decay 0.01, BF16</td></tr><tr><td>Optimizer Training steps</td><td></td><td>6k</td><td>700</td></tr><tr><td>Global batch size</td><td>10k</td><td>24</td><td>24</td></tr><tr><td></td><td>16  $1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Learning rate Fake-score LR</td><td></td><td></td><td> $4 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>AdamWβ</td><td>(0.9, 0.999)</td><td>(0,0.999)</td><td>(0,0.999)</td></tr><tr><td>Gradient clipping</td><td></td><td>10</td><td>10</td></tr><tr><td>Generator update</td><td>every step</td><td>every step</td><td>every 5 steps</td></tr><tr><td>Generator EMA</td><td></td><td></td><td>0.99 from step 200</td></tr><tr><td>Denoising steps</td><td>50</td><td>50</td><td>4</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Classifier-free guidance</td><td>5.0</td><td>5.0</td><td></td></tr></table>

## E.3. Metrics

Reconstruction quality. FVD [43], LPIPS [55], PSNR, and SSIM follow the sampling protocol of App. E.1: FVD measures distributional similarity using I3D features, whereas the three frame-paired metrics measure perceptual, pixel, and structural agreement with the ground-truth future. Motion and appearance diagnostics. Motion Magnitude Error (MME) is $| \rho ( \mathrm { g e n } ) - \rho ( \mathrm { G T } ) |$ , where $\rho ( \cdot )$ is the mean RAFT [42] flow magnitude over the same 16-frame $2 5 6 ^ { 2 }$ grid; lower values indicate closer aggregate motion energy. Aesthetic uses the LAION improved predictor with CLIP ViT-L/14.

Control and attribution. Following MultiWorld [50], the inverse dynamics evaluator of the main paper is trained only on real training clips and never on generated videos; it consists of one frozen convolutional network with about 2.9M parameters per data source. For each action frame, the network reads the surrounding RGB frames resized to $1 2 8 ^ { 2 }$ using a five-frame window with offsets $\{ - 6 , - 3 , 0 , 3 , 6 \}$ for games and a three-frame window $\{ - 1 , 0 , 1 \}$ for robots, and jointly predicts the controls of both agents. It receives no agent identity or location, so it can tell the agents apart only from the pixels. Predictions cover frames 1–80 of each clip, excluding the conditioning frame. For games, it predicts movement and interaction events for each agent, and Game Act-F1 is the macro F1 across valid events and agents. For robots, it predicts the continuous control trajectory. Let V denote the valid, non-padding control dimensions, which are the only ones scored, and let $\sigma _ { j }$ be the standard deviation of dimension j on the real training set. We compute

$$
\mathrm { N R M S E } = \sqrt { \frac { 1 } { T | \mathcal { V } | } \sum _ { \tau = 1 } ^ { T } \sum _ { j \in \mathcal { V } } \left( \frac { \widehat { a } _ { \tau , j } - a _ { \tau , j } } { \sigma _ { j } + \varepsilon } \right) ^ { 2 } } ,\tag{17}
$$

where $\widehat { \boldsymbol { a } } _ { \tau , j }$ is the recovered control, ε is a small constant, and lower values indicate more accurate control recovery.

Attr-Acc is a pairwise swap test on the same generated videos and evaluator predictions, so it requires no additional renders and is computed identically for every method in Tab. 2. For a clip with $N _ { \mathrm { a g } }$ agents, let $\mathbf { u } _ { \tau } ^ { ( i ) }$ and $\widehat { \mathbf { u } } _ { \tau } ^ { ( i ) }$ be the commanded and recovered controls of agent i on $\nu ,$ centered by subtracting the mean of the same source, agent, and dimension on the evaluator’s training set. Robot dimensions are further divided by the corresponding training-set standard deviation, whereas game dimensions already lie in [0, 1] and are not rescaled. At frame τ , an unordered agent pair $( i , i ^ { \prime } )$ is scored only if their commanded streams differ on at least one dimension in V, since exchanging identical commands is uninformative. We compare the commanded assignment with the one in which only agents i and $i ^ { \prime }$ are

exchanged,

$$
\begin{array} { r l } & { E _ { \tau , i i ^ { \prime } } ^ { \mathrm { c } } = \displaystyle \sum _ { k \in \{ i , i ^ { \prime } \} } \big \| \widehat { \mathbf { u } } _ { \tau } ^ { ( k ) } - { \mathbf { u } } _ { \tau } ^ { ( k ) } \big \| _ { 2 } ^ { 2 } , } \\ & { E _ { \tau , i i ^ { \prime } } ^ { \mathrm { s } } = \big \| \widehat { \mathbf { u } } _ { \tau } ^ { ( i ^ { \prime } ) } - { \mathbf { u } } _ { \tau } ^ { ( i ) } \big \| _ { 2 } ^ { 2 } + \big \| \widehat { \mathbf { u } } _ { \tau } ^ { ( i ) } - { \mathbf { u } } _ { \tau } ^ { ( i ^ { \prime } ) } \big \| _ { 2 } ^ { 2 } , } \end{array}\tag{18}
$$

where the remaining agents contribute equally to both assignments and are omitted. The pair scores 1 if $E _ { \tau , i i ^ { \prime } } ^ { \mathrm { c } } <$ $E _ { \tau , i i ^ { \prime } } ^ { \mathrm { s } }$ , 0.5 on a tie, and 0 otherwise. It is therefore correct when the controls recovered from the pixels match the commands better in the commanded order than with the two agents exchanged, that is, when each of the two streams drives its intended agent rather than the other one. A frame is scored if it has at least one scored pair, and its score is the mean over these pairs; with $N _ { \mathrm { a g } } { = } 2$ there is a single pair and the test reduces to exchanging the two agents. The centering fixes the chance level at 0.5 for any $N _ { \mathrm { a g } } { \mathrm { : } }$ a predictor that ignores the video and outputs each agent’s training mean is zero for every agent after centering, so $E _ { \tau , i i ^ { \prime } } ^ { \mathrm { c } } = E _ { \tau , i i ^ { \prime } } ^ { \mathrm { s } }$ for every pair. Frame scores are averaged within each clip, then over the clips of each source, and finally over sources without weighting, four for games and two for robots. Averaging within clips first prevents clips with many scored frames from dominating, and the source-level mean gives every source equal weight.

## F. Additional Results

## F.1. Three-Stage Causal Transfer

Tab. S3 summarizes the generation settings of the three stages together with their control, attribution, and throughput results. Stage 2 runs slower than Stage 1 despite its causal attention: it repeats 50 denoising steps with classifier-free guidance (CFG) sequentially for each of the five four-latent chunks and refreshes the KV cache after every chunk, whereas the bidirectional teacher denoises all 21 latents jointly at each step. These many small, sequential forward passes use the GPU less efficiently than the teacher’s large joint passes. Stage 3 reduces each chunk to four denoising steps without CFG and reaches 17.70 FPS, 10.7× faster than Stage 2.

## F.2. Detailed Ablations

Tab. S5 extends the main-text ablation along two axes: the construction of the schema-guided control tokens and the temporal alignment of App. B.2.

Schema-Guided Action Adaptation. SGAA combines three factors: a schema description that identifies each control, a separate token for each control dimension, and affine modulation of that token by the control value. Tab. S4 summarizes the variants that separate these factors. Every variant changes only how the control tokens are constructed and keeps Agent-Aware Spatial Modulation, per-agent crossattention, the one-latent action window, and the time and agent embeddings of Eq. 5.

The main-text w/o SGAA variant replaces the module with a conventional flat action encoder that maps the full action vector of each frame to a single token. It removes all three factors at once and reduces the action memory of each latent from rC to r tokens. The remaining variants keep the per-dimension tokens and remove one factor at a time. The w/o value modulation variant keeps the schema token but adds a value embedding to it instead of modulating it. The w/o schema semantics variant keeps the modulation but replaces the schema embedding with a learned per-dimension embedding, so that each dimension remains distinguishable but carries no language description; comparing it with the full model tests whether the schema description contributes beyond identifying the dimension.

Replacing modulation with addition alone collapses control to the level of removing SGAA in both domains (game Act-F1 0.209 vs. 0.225, robot NRMSE 1.071 vs. 1.120), and attribution is no better (game Attr-Acc 0.513 vs. 0.603, robot Attr-Acc 0.480 vs. 0.484), with robot Attr-Acc near the chance level of 0.5. Schema descriptions and perdimension tokens are therefore insufficient without modulation. This is consistent with the design of Sec. 3.3: addition mixes the value embedding into the schema token, whereas modulation keeps the schema token as the carrier of the control identity and lets the value only rescale and shift it.

Replacing the schema embedding with learned perdimension IDs isolates the contribution of the language description. On games, this variant falls to the level of removing SGAA (Act-F1 0.230 vs. 0.225, Attr-Acc 0.588 vs. 0.603), even though it keeps the per-dimension tokens and value modulation of the full model. The four game sources assign different controls to the same index, so an index-level ID cannot distinguish these controls, whereas the schema description can. On robots, learned IDs retain part of the gain over removing SGAA (NRMSE 0.799 vs. 1.120, Attr-Acc 0.778 vs. 0.484) but remain clearly below the full model (0.558 and 0.942). The schema description therefore contributes beyond identifying each dimension, most strongly on the games, where the same index carries different meanings across sources.

We additionally test whether the trained full model actually reads the schema descriptions. Without retraining, we pair each control dimension with the description of another dimension at inference, keeping the action values, perdimension tokens, and value modulation unchanged. Control collapses to the level of the reduced variants: game Act-F1 drops to 0.207, and Attr-Acc falls to 0.504 on games and 0.496 on robots, both near chance. The full model therefore relies on the schema embedding to identify each control, and the per-dimension tokens and value modulation cannot produce the commanded behavior once the descriptions are mismatched.

Table S3. Three-stage causal transfer. Generation settings and results of each stage; throughput is measured at $5 1 2 \times 4 4 8$ on a single GPU, and Stage 3 samples without classifier-free guidance (CFG). Best results are bolded.
<table><tr><td></td><td>Stage 1 Foundational Ability Acquisition</td><td>Stage 2 Teacher Forcing for AR Tuning</td><td>Stage 3 DMD for Streaming Video</td></tr><tr><td>Role</td><td>Full-context teacher</td><td>Causal initialization</td><td>Rollout distillation</td></tr><tr><td>Attention</td><td>Bidirectional</td><td>Blockwise causal</td><td>Blockwise causal</td></tr><tr><td>Training history</td><td>Full video</td><td>Clean chunks</td><td>Model rollouts</td></tr><tr><td>Streaming</td><td>X</td><td> $\checkmark$ </td><td> $\checkmark$ </td></tr><tr><td>Chunk size</td><td>Full sequence</td><td>4 latent frames</td><td>4 latent frames</td></tr><tr><td>Denoising steps</td><td>50</td><td>50</td><td>4</td></tr><tr><td>CFG scale</td><td>5.0</td><td>5.0</td><td></td></tr><tr><td>Game Act-F1 ↑</td><td>0.539</td><td>0.565</td><td>0.594</td></tr><tr><td>Game Attr-Acc ↑</td><td>0.844</td><td>0.898</td><td>0.898</td></tr><tr><td>Robot NRMSE↓</td><td>0.430</td><td>0.473</td><td>0.558</td></tr><tr><td>Robot Attr-Acc ↑</td><td>0.969</td><td>0.970</td><td>0.942</td></tr><tr><td>FPS ↑</td><td>2.77</td><td>1.66</td><td>17.70</td></tr></table>

Table S4. Design of the Schema-Guided Action Adaptation variants. Each variant changes only how the control tokens of agent i are constructed; the superscript (i) is omitted. All variants keep Agent-Aware Spatial Modulation, per-agent cross-attention, the one-latent action window, and the time and agent embeddings of $\operatorname { E q . } 5 . \ u _ { j }$ is the projected schema embedding of Eq. 3, $( \gamma _ { \tau , j } , \beta _ { \tau , j } ) = \mathrm { M L P } _ { \mathrm { m o d } } ( a _ { \tau , j } )$ as in Eq. $4 , e _ { j } ^ { \mathrm { I D } }$ is a learned per-dimension embedding, and π is a fixed permutation of the control dimensions.
<table><tr><td>Variant</td><td>Control identity</td><td>Tokens per frame</td><td>Value fusion</td><td>Training</td><td>Control token</td></tr><tr><td>Full LiveAct</td><td>schema  $u _ { j }$ </td><td>C</td><td>affine modulation</td><td>trained</td><td> $h _ { \tau , j } = u _ { j } \odot ( 1 + \gamma _ { \tau , j } ) + \beta _ { \tau , j }$ </td></tr><tr><td>w/o SGAA</td><td></td><td>1</td><td>flat MLP</td><td>trained</td><td> $h _ { \tau } = \mathrm { M L P } _ { \mathrm { f l a t } } ( A _ { \tau } )$ </td></tr><tr><td>w/o value modulation</td><td>schema  $u _ { j }$ </td><td>C</td><td>addition</td><td>trained</td><td> $h _ { \tau , j } = u _ { j } + \mathrm { M L P } _ { \mathrm { v a l } } ( a _ { \tau , j } )$ </td></tr><tr><td>w/o schema semantics</td><td>learned  $\mathrm { I D } \ { \overset { \cdot } { e } } _ { j } ^ { \mathrm { I D } }$ </td><td>C</td><td>affine modulation</td><td>trained</td><td> $h _ { \tau , j } = e _ { j } ^ { \mathrm { I D } } \odot ( 1 + \gamma _ { \tau , j } ) + \beta _ { \tau , j }$ </td></tr><tr><td>shuffled schema</td><td>schema  $u _ { \pi ( j ) }$ </td><td>C</td><td>affine modulation</td><td>inference only</td><td> $h _ { \tau , j } = u _ { \pi ( j ) } \odot ( 1 + \gamma _ { \tau , j } ) + \beta _ { \tau , j }$ </td></tr></table>

Temporal alignment. The action window length L sets how many latent frames of controls each future latent attends to: the main model uses L=1, so every latent reads exactly the r=4 raw-time steps compressed into it, while L=2 and L=4 extend the window over the following latents. Longer windows dilute the per-latent control signal and degrade all four metrics monotonically, on games (Act-F1 0.594 → 0.449 → 0.325, Attr-Acc 0.898 → 0.846 → 0.721) as well as on robots (NRMSE $0 . 5 5 8  0 . 5 9 1 $ 0.700, Attr-Acc $0 . 9 4 2  0 . 9 3 7  0 . 9 2 6 )$

## F.3. Spatial Localization of Agent-Specific Residuals

Fig. S2 provides a complementary spatial diagnostic of Agent-Aware Spatial Modulation: each agent-specific residual-magnitude heatmap concentrates around the corresponding visual agent, consistent with the intended streamto-agent binding.

Table S5. Detailed ablations of Schema-Guided Action Adaptation and temporal alignment. The SGAA variants are defined in Tab. S4. All trained variants use the same data and optimization budget as the full model. L is the action window length in latent frames (App. B.2). Best results are bolded.
<table><tr><td>Variant</td><td>Game Act-F1 ↑</td><td>Game Attr-Acc ↑</td><td>Robot NRMSE ↓</td><td>Robot Attr-Acc ↑</td></tr><tr><td>Full LiveAct (L=1)</td><td>0.594</td><td>0.898</td><td>0.558</td><td>0.942</td></tr><tr><td>Schema-Guided Action Adaptation</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o SGAA (flat MLP)</td><td>0.225</td><td>0.603</td><td>1.120</td><td>0.484</td></tr><tr><td>w/o value modulation (additive value)</td><td>0.209</td><td>0.513</td><td>1.071</td><td>0.480</td></tr><tr><td>w/o schema semantics (ID tokens)</td><td>0.230</td><td>0.588</td><td>0.799</td><td>0.778</td></tr><tr><td>shuffled schema (inference only)</td><td>0.207</td><td>0.504</td><td>1.063</td><td>0.496</td></tr><tr><td>Temporal alignment</td><td></td><td></td><td></td><td></td></tr><tr><td>action window L=2</td><td>0.449</td><td>0.846</td><td>0.591</td><td>0.937</td></tr><tr><td>action window L=4</td><td>0.325</td><td>0.721</td><td>0.700</td><td>0.926</td></tr></table>

## F.4. Additional Qualitative Comparisons

Fig. S3 and Fig. S4 add two pairs of cases with the same methods and sampling protocol as Fig. 5: a Mega Bomberman round and a Street Fighter II round, each paired with a RoboTwin scene in which one arm places an object into a basket. They confirm that the failure modes of the main paper are not specific to the two scenes shown there. First, the scene drift of LingBot-World 2.0 can be abrupt: In Fig. S3, the generated frames gradually turn black and then transitions to an unrelated scene. However, in both robot cases, it started to break down from the first sampling frame. Second, the game-track failures of Incantation grow with time, from a darkened board to black frames, while MultiWorld and ActionParty keep the board and the table but smear the players and arms. Although WanToFight maintains the scene and completes the action, there is a mismatch between the arm and the action. LiveAct matches the target arm pose more closely at each sampled frame.

![](images/73c7188116efa1acb2aa3d0f5f687a6128fe06395c7a7c8439b12506b64b4938.jpg)  
Figure S2. Spatial localization of agent-specific control residuals. Panels (a)–(d) show representative spatial residual-magnitude heatmaps from the agent-specific branches. Red and blue denote agents 1 and 2, respectively, and higher intensity indicates a large response. Across the examples, the high-response regions concentrate around the corresponding visual agents.

## F.5. Additional LiveAct Showcases

Fig. S5 presents additional generations from LiveAct across the game and robot domains. Each row shows one evaluation condition, and columns show the generated rollout at successive times.

The interface is not tied to a fixed number of agents: the dataset contains scenes with two to four concurrently controlled agents, and each additional agent only adds one agent description, one action stream, and one per-agent cross-attention residual, without any architectural change. Fig. S6 shows scenes with more than two agents: a Micro Machines race and a Mega Bomberman round with binary controls, and a RoboFactory cube-stacking episode with continuous controls.

## F.6. Long-Horizon Rollout Showcase

Fig. S7 shows an additional dynamic case that exceeds the 81-frame training and evaluation window used for quantitative assessment.

![](images/3808f3033cabb4208a059e93c9b766900c2972172a81ef231d0f8c4355f24fad.jpg)  
(a) Multi-Agent Video Games

![](images/ecc4f9622db8d7c2b1456d108f7a015fb1dadf919808df64fa5ca17bda640524.jpg)  
(b) Multi-Robot Manipulation

Figure S3. Additional qualitative comparison (set 1). Same layout and methods as Fig. 5.  
![](images/2d19b48410e5d641eda0133cb79d766aba2e532eb0aa6665c73ef90fa1120012.jpg)  
(a) Multi-Agent Video Games

![](images/f4d5452b48545d82bf3c0f59ffbdbd8021cb2d6c2cbe31e38d5cb92813943efe.jpg)  
(b) Multi-Robot Manipulation  
Figure S4. Additional qualitative comparison (set 2). Same layout and methods as Fig. 5.

<sub>i-Agents</sub> V<sup>ideo</sup> <sup>Ga</sup>

<sub>i-Robot</sub> M<sup>anipula</sup>  
![](images/90174e3102d3e64f5461096b8163041c3cb9e2e98bfbeec7eccbd48e0f2dfb50.jpg)  
Figure S5. Additional LiveAct showcase.

(a) Micro Machines · three cars · 3 binary action streams  
![](images/bc9a2b5cbc3975b735ab2dc88185850939e7feb4f228e84fe1a77b8544396caa.jpg)

Figure S6. Showcase beyond two agents. Scenes with more than two concurrently controlled agents. For each scene, the top row shows the ground truth and the bottom row the LiveAct generation from the same initial frame and per-agent action streams.  
![](images/0c88f8fb35fff2094174dccc0f3b057d97dbb88e0d2f4dfcecc3fd86de360eaf.jpg)  
Figure S7. Single-case long-horizon rollout as a dense frame matrix. We show 24 generated frames from the same 241-frame rollout, sampled every ten frames and ordered from left to right and top to bottom.

## References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37: 58757–58791, 2024. 4

[2] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35101–35113, 2026. 2

[3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024. 4

[4] Remi Cadene, Simon Alibert, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, et al. Lerobot: An open-source library for end-to-end robot learning. In International Conference on Learning Representations, pages 122398–122417, 2026. S3

[5] Kunyu Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, Haozhe Chen, Qifeng Chen, and Zeyu Wang. Dit4edit: Diffusion transformer for image editing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2969– 2977, 2025. S1

[6] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026. 2, 3, 4, 7, 8, S4

[7] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 2, 4

[8] Xu He, Qiaochu Huang, Zhensong Zhang, Zhiwei Lin, Zhiyong Wu, Sicheng Yang, Minglei Li, Zhiyi Chen, Songcen Xu, and Xiaofei Wu. Co-speech gesture video generation via motion-decoupled diffusion model. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2263–2273. IEEE, 2024. 2

[9] Xu He, Zhiyong Wu, Xiaoyu Li, Di Kang, Chaopeng Zhang, Jiangnan Ye, Liyang Chen, Xiangjun Gao, Han Zhang, and Haolin Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3437–3445, 2025. 2

[10] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. S1

[11] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025. 2

[12] Anthony Hu, Vaclav Volhejn, Adrien Ramanana Rahary,´ Chris Mulder, Aditya Makkar, Alyx Liao, Amelie Royer,´ Manu Orsini, Adam Jelley, Eloi Alonso, et al. Multiplayer interactive world models with representation autoencoders. arXiv preprint arXiv:2607.05352, 2026. 2

[13] Li Hu, Guangyuan Wang, Peng Zhang, and Bang Zhang. Wantofight: Real-time generative game engine for multiplayer combat interaction. arXiv preprint arXiv:2607.12592, 2026. 2, 3, 7, 8, S1, S4

[14] Jiehui Huang, Yuechen Zhang, Xu He, Yuan Gao, Zhi Cen, Bin Xia, Yan Zhou, Xin Tao, Pengfei Wan, and Jiaya Jia. Unityvideo: Unified multi-modal multi-task learning for enhancing world-aware video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4471–4481, 2026. S1

[15] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Informa tion Processing Systems, 38:167283–167308, 2026. 3, 6, S1

[16] Wayne King, Zeyue Xue, Yuxuan Bian, Jie Huang, Haoran Li, Yaowei Li, Yaofeng Su, Yuming Li, Haoyu Wang, Shiy Zhang, et al. Echo-memory: A controlled study of memory in action world models. arXiv preprint arXiv:2606.09803, 2026. 2

[17] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2(3):6, 2025. 2

[18] Bowei Liu, Zheng Lu, Yuhan Bian, Xinchen Zhang, Xingming Shui, Yuesheng Huang, Xuhuan Li, Zihao Liu, Yifan Yang, Jun Zhou, et al. Vidforensics-m1: Meta-detection reinforcement learning with verifiable temporal grounding for ai generated video forensics. arXiv preprint arXiv:2608.11201, 2026. S1

[19] Bowei Liu, Xinchen Zhang, Xuhuan Li, Tianlin Zhang, Kaian Jiang, Jingwei Liu, Congyang Zhao, Yifei Qian, Junhong Liu, Zhiheng Li, Yujiu Yang, Yang Cai, and Ling Yang. Omniverifier-c1: Decomposed graded verifier for rule-governed generative world modeling. arXiv preprint arXiv:2610.22355, 2026. S1

[20] Fangfu Liu, Kai He, Tianchang Shen, Tianshi Cao, Sanja Fidler, Yueqi Duan, Jun Gao, Igor Gilitschenski, Zian Wang, and Xuanchi Ren. Gamma-world: Generative multiagent world modeling beyond two players. arXiv preprint arXiv:2605.28816, 2026. 2, 3, S1

[21] Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. Opsd-v: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026. S1

[22] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelli gence, pages 4117–4125, 2024. 2

[23] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Bingyuan Wang, Qinghe

Wang, Xuanhua He, Hongfa Wang, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025. 2

[24] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025. S1

[25] Yue Ma, Jiangming Wang, Yucheng Wang, Xilai Wang, Zhiyuan Li, Xinyu Wang, Hongyu Liu, Ruofan Liang, Songchun Zhang, Yuxuan Xue, et al. Livelight: Real-time streaming video relighting with interactive control. arXiv preprint arXiv:2608.01771, 2026. S1

[26] Yao Mu, Tianxing Chen, Shijia Peng, Zanxin Chen, Zeyu Gao, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo. Robotwin: Dual-arm robot benchmark with generative digital twins (early version). arXiv preprint arXiv:2409.02920, 2024. S3

[27] Alex Nichol, Vicki Pfau, Christopher Hesse, Oleg Klimov, and John Schulman. Gotta learn fast: A new benchmark for generalization in rl. arXiv preprint arXiv:1804.03720, 2018. S3

[28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182. IEEE, 2023. S1

[29] Mathieu Poliquin. Stable retro, a maintained fork of openai’s gym-retro. https://github.com/Farama-Foundation/stable-retro, 2026. S3

[30] Alexander Pondaven, Ziyi Wu, Igor Gilitschenski, Philip Torr, Sergey Tulyakov, Fabio Pizzati, and Aliaksandr Siarohin. Actionparty: Multi-subject action binding in generative video games. In European Conference on Computer Vision, pages 74–92. Springer, 2026. 2, 3, 7, 8, S4

[31] Yiran Qin, Li Kang, Xiufeng Song, Zhenfei Yin, Xiaohong Liu, Xihui Liu, Ruimao Zhang, and Lei Bai. Robofactory: Exploring embodied agent collaboration with compositional constraints. arXiv preprint arXiv:2503.16408, 2025. S3

[32] Xiangfei Qiu, Jilin Hu, Lekui Zhou, Xingjian Wu, Junyang Du, Buang Zhang, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, Zhenli Sheng, and Bin Yang. TFB: Towards comprehensive and fair benchmarking of time series forecasting methods. In Proc. VLDB Endow., pages 2363–2377, 2024. S1

[33] Xiangfei Qiu, Xingjian Wu, Hanyin Cheng, Xvyuan Liu, Chenjuan Guo, Jilin Hu, and Bin Yang. Dbloss: Decomposition-based loss function for time series forecasting. In NeurIPS, 2025.

[34] Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang. DUET: Dual clustering enhanced multivariate time series forecasting. In SIGKDD, pages 1185– 1196, 2025. S1

[35] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 4

[36] Georgy Savva, Oscar Michel, Daohan Lu, Suppakit Wai witlikhit, Timothy Meehan, Dhairya Mishra, Srivats Poddar, Jack Lu, and Saining Xie. Solaris: Building a multiplayer video world model in minecraft. arXiv preprint arXiv:2602.22208, 2026. 3, S1

[37] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 7

[38] Yiren Song, Cheng Liu, Yuxin Jiang, and Mike Zheng Shou. Streamingeffect: Real-time human-centric video effect gen eration. arXiv preprint arXiv:2605.17019, 2026. S1

[39] Yiren Song, Wangzi Yao, Haofan Wang, and Mike Zheng Shou. Vista: Triplet-supervised video style transfer with dif fusion transformers. arXiv preprint arXiv:2605.17312, 2026. S1

[40] Zihan Su, Junhao Zhuang, Yaowei Li, Siwen Lu, Haoran Li, Lingen Li, Haoyu Wu, Weiyang Jin, Songchun Zhang, Haoyang Huang, et al. Sgf+: Decoupling gradient flows for autoregressive video generation. arXiv preprint arXiv:2610.10429, 2026. S1

[41] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wen huan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025. 2

[42] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020. S5

[43] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. S5

[44] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations, pages 73754–73776, 2025. 2, 4

[45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video gen erative models. arXiv preprint arXiv:2503.20314, 2025. 7, S1, S3

[46] Jiangshan Wang, Yue Ma, Jiayi Guo, Yicheng Xiao, Gao Huang, and Xiu Li. Cove: Unleashing the diffusion feature correspondence for consistent video editing. Advances in Neural Information Processing Systems, 37:96541–96565, 2024.

[47] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. S1

[48] Xinyu Wang, Huafeng Shi, Zian Li, Yan Zhou, Xiaoqiang Liu, Yue Ma, and Pengfei Wan. Subjectanchor: Subjectaware memory-to-video for multi-shot storytelling. arXiv preprint arXiv:2609.34502, 2026. 3

[49] Xinyu Wang, Chongbo Zhao, Fangneng Zhan, and Yue Ma. Liveedit: Towards real-time diffusion-based streaming video editing. In European Conference on Computer Vision, pages 631–649. Springer, 2026. 3, S1

[50] Haoyu Wu, Jiwen Yu, Yingtian Zou, and Xihui Liu. Multiworld: Scalable multi-agent multi-view video world models. arXiv preprint arXiv:2604.18564, 2026. 2, 3, 7, 8, S4, S5

[51] Xuancheng Xu, Ming Tao, and Bing-Kun Bao. Clgc: Continuous layout guidance for consistent text-to-video editing. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2025. S1

[52] Xuancheng Xu, Gengyun Jia, and Bing-Kun Bao. Discolora: Disentangled composition of content, style, and motion for multi-concept video customization. arXiv preprint arXiv:2606.26668, 2026. S1

[53] Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 2, 4

[54] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht-¨ man, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623. IEEE, 2024. 3, 6, S1

[55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586– 595. IEEE, 2018. S5

[56] Ruicheng Zhang, Mingyang Zhang, Jun Zhou, Zhangrui Guo, Xiaofan Liu, Zunnan Xu, Zhizhou Zhong, Puxin Yan, Haocheng Luo, and Xiu Li. Mind-v: Hierarchical video generation for long-horizon robotic manipulation with rl-based physical alignment. arXiv preprint arXiv:2512.06628, 2025. 2

[57] Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639, 2026. 2

[58] Ruicheng Zhang, Kaixi Cong, Jun Zhou, Zhizhou Zhong, Zunnan Xu, Shuiyang Mao, Wei Liu, and Xiu Li. Kvpo: Ode-native grpo for autoregressive video alignment via kv semantic exploration. arXiv preprint arXiv:2605.14278, 2026. 2, S1

[59] Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, and Xiu Li. Zo3t: Zeroshot 3d-aware trajectory-guided image-to-video generation via test-time training. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12708–12716, 2026. 2, S1

[60] Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang, Haoran Li, Xiaoxiao Ma, et al. Echowm: Open and enterable omnimodal world models. arXiv preprint arXiv:2608.23189, 2026. 2

[61] Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: A fine-grained world model for robot manipulation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9834–9844. IEEE, 2025. 4

[62] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026. 6, S1

[63] Shangwen Zhu, Qianyu Peng, Zhao Pu, Zhilei Shu, Xiangru Ke, Zhaohu Xing, Zizhao Tong, Zeqing Wang, Xinyu Cui, Zian Zheng, et al. Incantation: Natural language as the action interface for multi-entity video world models. arXiv preprint arXiv:2605.18601, 2026. 2, 3, 4, 7, 8, S4

[64] Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, et al. Self gradient forcing: Native long video extrapolation. arXiv preprint arXiv:2607.20368, 2026. S1