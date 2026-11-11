# TaoMate-R1: Temporal Role Specialization for Streaming Multi-Reference Video Generation

Qijun Gan<sup>\*</sup>, Chenwei Zhang<sup>\*</sup>, Xinyang Li, Meiguang Jin<sup>†</sup>, Junfeng Ma

Alibaba Group

{ganqijun.gqj, meiguang.jmg}@taobao.com

## ABSTRACT

Streaming multi-reference video generation establishes a scene from specified people and objects, then continues it under changing instructions. We introduce TaoMate-R1, a streaming multi-reference generator that organizes adaptation parameters through temporal role specialization. It assigns separate low-rank adapters to scene establishment and continuation on a frozen backbone. Each adapter performs both denoising and context writing for its role; Self Gradient Forcing jointly trains the roles through the history they construct. This allocation contrasts with operation-based specialization, which separates denoising from context writing across all chunks. A fixed-replay gradient analysis characterizes the update directions admitted by the two sharing schemes. On 32 multi-reference cases, we compare the schemes at the same two-adapter budget. Across ten approximately two-minute streams with changing instructions and shared teacher-audio conditioning, TaoMate-R1 reduces the mean boundary luminance jump from 1.96 to 0.613, with higher boundary feature similarity. It also achieves higher mean CLIP and DINO temporal similarity across short interactions, long streams, and public reference compositions. These results support grouping adaptation parameters by scene establishment and continuation for streaming multi-reference generation. Our code and model are released at https://github.com/TaoLiveAIGC/TaoMate-H3.

## 1 Introduction

Multi-reference video generation specifies the people and objects in a scene through images. Models such as Phantom, VACE, and MiniMax-H3 compose these visual conditions into video clips [1–3]. Streaming extends the task to an evolving scene: a presenter introduces a cup, puts it away, and picks up a bag as new instructions arrive. The desired behavior combines change with persistence. The current instruction determines the interaction, while the generated past determines its starting state.

The opening and subsequent chunks therefore solve diferent conditional tasks. At the opening, the generator chooses how reference appearances are realized through pose, placement, and initial motion. During continuation, those realized choices become part of the conditioning: a hand holding a cup constrains where the next action begins. This asymmetry motivates studying whether these tasks should receive independent adaptation.

Autoregressive difusion and student-rollout training make generated history available during learning [4, 5]. Self Gradient Forcing (SGF) also lets later predictions supervise the computations that write this history [6]. SGF+ separates denoising and context-writing parameters to accommodate diferences between their gradient contributions [7]. This is an operation-based partition: each operation shares parameters across the entire stream.

![](images/c0ca07492e9f205834b867855157efef4acbaede19eb18d213bb34695aa8c0d4.jpg)  
Figure 1. Streaming multi-reference video generation. (a) Multiple references guide scene establishment and continuation under changing instructions (conceptual illustration). (b) Each colored group shares one adapter: SGF+ groups computations by operation; TaoMate-R1 groups them by temporal role, sharing generation and context-writing parameters within each role.

We investigate a temporal-role partition at the same two-adapter budget, grouping generation and writing by whether they establish or continue a scene. The distinction concerns which computations learn together (Fig. 1).

We introduce TaoMate-R1, which implements temporal role specialization with an opening adapter at absolute chunk zero and a continuation adapter for every later chunk. Within each role, denoising and context writing share the same low-rank parameters on a frozen backbone. Instruction changes update the continuation conditions while preserving its adapter and generated history. We train both roles with SGF: historical-context gradients return to the role that wrote the corresponding representation. The opening role thus learns both the first chunk and its persistent video anchor; the continuation role learns subsequent predictions and recent audiovisual context.

We characterize this design through a local analysis of parameter sharing. At a common initialization and fixed replay, the two partitions preserve the same common gradient component but admit diferent specialization directions. The analysis also isolates continuation-only windows, where the comparison depends on alignment between continuation’s generation and writing gradients. These relations describe the optimization consequences of tying parameters and provide a precise basis for comparing the two assignments.

Our experiments compare TaoMate-R1’s Temporal split with Shared and Operation split on the same H3 backbone. Across 32 cases, TaoMate-R1 achieves higher mean CLIP and DINO temporal similarity than Operation split on short interactions, long instruction-driven streams, and public reference compositions. The clearest diference occurs at instruction boundaries in ten approximately two-minute streams: TaoMate-R1 has a mean luminance jump of 0.613, compared with 1.96 for Operation split, and higher adjacent-frame feature similarity. We examine these diferences with case-paired statistics and background-appearance visualizations, then place the generator’s results alongside external reference-conditioned systems. The contributions are:

▶ TaoMate-R1, a streaming multi-reference generator that assigns independent adapters to scene establishment and continuation, shares denoising and context writing within each role, and trains them jointly through SGF.

▶ A local gradient characterization of temporal and operation-based sharing, including the dependence on generation–writing alignment in continuation-only training windows.

▶ A comparison across 32 multi-reference cases showing higher mean temporal similarity across three interaction settings and reduced instruction-boundary discontinuities on long streams.

## 2 Related Work

Multi-Reference Generation and Control. Reference images specify visual attributes and associate them with entities. ConsisID separates facial structure from identity detail [8]; Video Alchemist binds reference features to entity words [9]; and Movie Weaver combines prompt anchors with reference-order embeddings [10]. Phantom aligns visual and textual reference representations [1], VACE integrates image, video, and mask conditions [2], and MiniMax-H3 supports multi-reference audiovisual generation [3]. Spatiotemporal control specifies where and when these entities appear: DreamVideo-2 uses bounding-box trajectories [11], AlcheMinT uses interval-dependent rotary embeddings [12], and MultiShotMaster combines shot-aware encoding with reference grounding [13]. For interactive streams, FlowAct-R2 aligns evolving conditions with planned behavior [14], while ReferStream introduces arriving subjects through contextualized reference latents and cache injection [15]. These methods organize visual conditions; our study concerns the adaptation parameters used to realize them across an ongoing scene.

Causal Generation and Generated-History Training. Difusion Forcing uses token-wise noise levels for causal generation [16]; CausVid distills a bidirectional teacher into a few-step causal model [4]; and SkyReels-V2 extends videos with nondecreasing frame-wise noise schedules [17]. History retention supplies persistent appearance and recent motion: StreamingT2V combines initial-chunk cues with preceding-chunk features [18], FramePack compresses context according to historical importance [19], and LongLive combines a frame sink with local attention [20]. LongLive-RAG retrieves earlier latents [21], while FilmWeaver separates cross-shot keyframes from recent within-shot memory [22]. Training with generated predecessors addresses the context distribution encountered at inference, from Scheduled Sampling [23] to Self Forcing’s student rollouts [5] and Rolling Forcing’s jointly denoised windows [24]. SGF additionally propagates supervision through historical context writers [6]. We adopt this training framework and study parameter sharing under a fixed history-retention policy.

Parameter Specialization. Specialization allocates independent parameters to computations with diferent learning demands. DreamVideo composes identity and motion adapters [25], while eDif-I and Wan2.2 specialize denoisers by difusion-noise interval [26, 27]. SGF+ separates denoising and context-writing parameters to address gradient diferences between these operations [7]. TaoMate-R1 groups the same operations by temporal role, using independent low-rank adapters [28] for scene establishment and continuation. Our comparison with operation-based sharing holds the two-adapter budget fixed and examines the resulting generation and continuity measurements.

## 3 TaoMate-R1

TaoMate-R1 is a streaming multi-reference generator built on temporal role specialization: it groups generation and context writing by their role in the stream (Fig. 2).

## 3.1 Preliminaries

Generation and Context Writing. A streaming difusion model alternates between generating a chunk and encoding it as history. For external conditions $q _ { b }$ and retained keys and values $\mathcal { H } _ { < b }$ , a denoising trajectory $G _ { \theta }$ maps noise $\epsilon _ { b }$ to a clean latent; a clean-time execution $W _ { \theta }$ writes context:

$$
\begin{array} { r l } & { \widehat { z _ { b } } = G _ { \theta } ( \epsilon _ { b } , q _ { b } , \mathcal { H } _ { < b } ) , } \\ & { h _ { b } = W _ { \theta } ( \widehat { z _ { b } } , q _ { b } , \mathcal { H } _ { < b } ; 0 ) . } \end{array}\tag{1}
$$

The retained part of $h _ { b }$ conditions subsequent predictions.

Generated-History Supervision. SGF trains historical representations on student-generated rollouts [6]. Recorded historical latent inputs remain fixed during diferentiation, while later losses propagate through their KV-writing computations. Generation and writing therefore contribute separate paths to the same objective. Sharing their parameters sums these contributions into one update; independent parameters allocate them to separate updates. Our method specifies this sharing relation for the opening and continuation of a stream.

![](images/f02380c80df8a738ef759479c0d39a1c58d3f38d9b68c3a99ad8f496dc0654eb.jpg)  
Figure 2. TaoMate-R1: generation and context writing through temporal role specialization. The absolute chunk index selects an adapter for both denoising and clean-context writing. The resulting audiovisual keys and values condition subsequent chunks; video latents are decoded for display. The reference images and four stacked outpu frames come from an evaluated product-interaction sequence.

## 3.2 Problem Formulation

Let $\mathcal { R } _ { b }$ denote the reference images active at chunk $b , c _ { b }$ its instruction, and $\boldsymbol { z _ { b } } = \big ( z _ { b } ^ { \nu } , z _ { b } ^ { a } \big )$ its audiovisual latents. These specify $q _ { b } = ( \mathcal { R } _ { b } , c _ { b } )$ in Eq. (1). The retained history $\mathcal { H } _ { < b }$ consists of layerwise audiovisual KVs from preceding chunks, with $\mathcal { H } _ { < 0 } = \emptyset$ at the opening. We parameterize a stream of � chunks as

$$
\begin{array} { r l } & { p ( z _ { 0 : B - 1 } \mid \mathcal { R } _ { 0 : B - 1 } , c _ { 0 : B - 1 } ) } \\ & { \quad \quad = p _ { \theta _ { \mathrm { f i r s t } } } ( z _ { 0 } \mid \mathcal { R } _ { 0 } , c _ { 0 } ) \displaystyle \prod _ { b = 1 } ^ { B - 1 } p _ { \theta _ { \mathrm { c o n t } } } ( z _ { b } \mid \mathcal { R } _ { b } , c _ { b } , \mathcal { H } _ { < b } ) . } \end{array}\tag{2}
$$

Establishment occurs once, at absolute chunk $b = 0$ . Changes in instructions or active references become conditions for continuation within the existing scene.

The references and generated history provide diferent kinds of information. Reference images identify entities and their appearances; history records how those entities have been realized together in the current scene. At the opening, the model establishes their spatial arrangement, poses, and interaction state from the supplied conditions. Continuation inherits this realization and predicts its evolution under the current instruction. For example, presenting a new object changes the active references while the presenter and room remain part of the ongoing scene. The temporal role thus follows the stream’s established state: changing an instruction updates the interaction goal while retaining the history from which the next action begins.

The long-stream evaluation additionally supplies cached audio states and removal-segment end-frame conditions.

## 3.3 Temporal Role Specialization

The opening and continuation adapters use independent LoRA updates [28] on a shared frozen backbone:

$$
r ( b ) = \left\{ \begin{array} { l l } { \mathrm { f i r s t } , } & { b = 0 , } \\ { \mathrm { c o n t } , } & { b > 0 , } \end{array} \right. \qquad \theta _ { b } = \theta _ { \mathrm { b a s e } } + \Delta \theta _ { r ( b ) } .\tag{3}
$$

Both inherit the LoRA weights of the same trained autoregressive generator and are optimized jointly. Their weights, AdamW moments, and exponential moving averages are copied into independent states. For each adapted layer, $\Theta _ { r } = \Theta _ { 0 } + B _ { r } A _ { r }$ , with fixed LoRA scaling absorbed into the factors. Both adapters use rank and scaling 128 on the attention QKV/output and feed-forward projections, covering 208 linear layers.

The selected $\theta _ { b }$ governs both � and � in Eq. (1). Thus, a temporal role adapts its synthesis mapping together with the context representation it leaves for future predictions.

This grouping specifies which learning signals shape the same adaptation. For establishment, generation realizes the references in an initial scene and writing represents that scene for subsequent predictions. For continuation, generation evolves the interaction and writing updates the context available to the next chunk. Within-role sharing combines these contributions according to their conditional task. Operation-based specialization instead combines opening and continuation signals within each operation. The same generation and writing calls are therefore grouped along diferent axes: temporal role in TaoMate-R1 and computational operation in SGF+.

At inference, the history retains the opening video KVs as a persistent appearance anchor and the latest two audiovisual chunks’ KVs as recent interaction context. For $b > 0$

$$
\mathcal { H } _ { < b } = \{ h _ { 0 } ^ { \nu } \} \cup \bigcup _ { j = \operatorname* { m a x } ( 0 , b - 2 ) } ^ { b - 1 } \{ h _ { j } ^ { \nu } , h _ { j } ^ { a } \} .\tag{4}
$$

Overlapping entries are counted once; opening audio remains only in the recent window. Reference conditions remain available independently of this history, and temporal positions remain absolute. Thus, the opening role writes the persistent anchor, while continuation supplies the rolling interaction context. All three parameter-sharing configurations use this retention policy.

## 3.4 Joint Training with Context Gradients

Student Rollouts and Distribution Matching. We train TaoMate-R1’s two roles using student rollouts [5] and SGF’s context gradients [6]. Let $\boldsymbol { \mathcal { I } } = \{ s , \ldots , e \}$ denote a sampled training window, with its preceding prefix held fixed as conditioning. We request five or ten chunks with probabilities 0.8 and 0.2, truncated by source length. The prefix is encoded from cached clean teacher latents; references and text remain fixed within an example. Nominal opening and continuation chunks contain 39 and 34 frames at 24 fps. A gradient-free rollout uses one initial-to-clean evaluation per chunk and records the prediction noise, generated latents, context, and actual writer input $\widetilde { z } _ { b }$ , including the training context perturbation. Both computations use the adapter selected by the absolute chunk index.

Distribution-matching distillation [29] supervises the generated audiovisual content. A frozen real-score model and a learned fake-score model evaluate the same corrupted student sequence, producing a detached direction $d _ { b }$ Let $\widehat { z } _ { b , \mathrm { r e a l } } ^ { m }$ and $\widehat { z } _ { b , \mathrm { f a k e } } ^ { m }$ be their clean estimates for modality �. We use the normalized direction

$$
d _ { b } ^ { m } = \operatorname { s g } \left[ \frac { \widehat { z } _ { b , \mathrm { f a k e } } ^ { m } - \widehat { z } _ { b , \mathrm { r e a l } } ^ { m } } { \operatorname * { m a x } \left( \operatorname * { m e a n } | \widehat { z } _ { b } ^ { m } - \widehat { z } _ { b , \mathrm { r e a l } } ^ { m } | , \varepsilon \right) } \right] .\tag{5}
$$

We set $\varepsilon = 1 0 ^ { - 8 }$ . For a prediction $\widehat { z } _ { b }$ replayed with its recorded inputs, we optimize

$$
\mathcal { L } _ { b } = \frac { 1 } { 2 } \sum _ { m \in \{ \nu , a \} } \lambda _ { m } \left. \widehat { z } _ { b } ^ { m } - \mathrm { s g } ( \widehat { z } _ { b } ^ { m } - d _ { b } ^ { m } ) \right. _ { \mathrm { m e a n } } ^ { 2 } .\tag{6}
$$

Here sg stops gradients, $\lambda _ { \nu } = 1 , \lambda _ { a } = 0 . 1$ , and the squared norm averages over that modality’s latent elements. The window objective is $\begin{array} { r } { \mathcal { L } = | \bar { \mathcal { I } } | ^ { - 1 } \sum _ { b \in \mathcal { I } } \mathcal { L } _ { b } } \end{array}$ . Both roles share a single fake-score model, trained on detached student outputs. The fake score receives velocity-regression updates every iteration; generator updates occur every fifth iteration.

Historical Credit Assignment. Replay losses accumulate context gradients on shared historical KV variables. We realize SGF’s context-gradient reconstruction by diferentiating each historical writer once in reverse temporal order, with its recorded latent input held fixed. The accumulated gradient returns to the adapter selected by that writer’s absolute chunk index and propagates through its preceding KVs. This assigns both synthesis and historical credit to the appropriate temporal role:

Algorithm 1 TaoMate-R1: temporal role specialization with shared-history supervision   
Require: Window $\boldsymbol { \mathcal { I } } = \{ s , \ldots , e \}$ , fixed prefix, references $\mathcal { R } _ { b }$ , instructions $c _ { b }$ , adapters $\theta _ { \mathrm { f i r s t } } , \theta _ { \mathrm { c o n t } }$   
1: $r ( b ) $ first if $b = 0 ,$ otherwise cont   
Pass 1: self-generated rollout, without gradients   
2: for $b = s , \ldots , e$ do   
3: $\widehat { z } _ { b } \gets G _ { \theta _ { r ( b ) } } ( \epsilon _ { b } , \mathcal { R } _ { b } , c _ { b } , \mathcal { H } _ { < b } )$   
4: $h _ { b } \gets W _ { \theta _ { r ( b ) } } ( \widetilde { z } _ { b } , \mathcal { R } _ { b } , c _ { b } , \mathcal { H } _ { < b } ; 0 )$   
5: Record inputs and context; advance history with $h _ { b }$   
6: end for   
The shared fake score learns from detached samples each iteration.   
Pass $\mathbf { \hat { \xi } } ^ { 2 } \mathbf { : }$ scheduled generator update   
7: Expose window KVs as shared gradient leaves; keep the prefix fixed   
8: Initialize role gradients $g _ { r } \gets 0$ and KV cotangents $\bar { h } _ { j }  0$   
9: for $b = s , \ldots , e$ do   
10: Replay current $G _ { \theta _ { r ( b ) } }$ with recorded inputs   
11: Form $\mathcal { L } _ { b }$ using detached score direction (6)   
12: Accumulate $\nabla ( \mathcal { L } _ { b } / | \mathcal { I } | )$ into $g _ { r ( b ) }$ and shared $\bar { h } _ { j }$   
13: end for   
14: for $j = e - 1 , \ldots , s$ do   
15: Reconstruct $W _ { \theta _ { r ( j ) } }$ from $\mathrm { s g } ( \widetilde { z } _ { j } )$ and the corresponding shared KV leaves   
16: Apply VJP with $\bar { h } _ { j } ;$ add to $g _ { r ( j ) }$ and earlier KV cotangents   
17: end for   
18: Update each supervised adapter with its accumulated $g _ { r }$

$$
\nabla _ { \Delta \theta _ { r } } \mathcal { L } = \sum _ { b \in { \cal { I } } _ { r } } g _ { b } ^ { \mathrm { g e n } } + \sum _ { j \in \mathcal { T } _ { r } } \left( \frac { \partial h _ { j } } { \partial \Delta \theta _ { r } } \right) ^ { \top } \bar { h } _ { j } .\tag{7}
$$

Here ${ { \mathcal { I } } _ { r } }$ and $\mathcal { T } _ { r }$ index role $r \mathrm { { : } } _ { \mathrm { { s } } }$ supervised predictions and diferentiable historical writers within the efective window. The direct gradient ${ \boldsymbol { g } } _ { b } ^ { \mathrm { g e n } }$ is taken through the current denoiser from $\mathcal { L } _ { b } / | \mathcal { T } |$ . The writer Jacobian holds its recorded latent input and incoming KVs fixed; $\bar { h } _ { j }$ includes all downstream prediction and writer paths under the window-averaged objective.

A historical representation influences the objective through direct reads by later denoisers and through context written by intervening chunks. The reverse pass accumulates these routes before diferentiating the corresponding writer. For a recorded writer input, the context computation and its contribution to an earlier representation are

$$
h _ { j } = W _ { \theta _ { j } } \big ( \mathrm { s g } ( \widetilde { z } _ { j } ) , q _ { j } , \mathcal { H } _ { < j } ; 0 \big ) ,\tag{8}
$$

$$
{ \bar { h } } _ { i } + = \left( { \frac { \partial h _ { j } } { \partial h _ { i } } } \right) ^ { \top } { \bar { h } } _ { j } , \qquad s \leq i < j .\tag{9}
$$

The same accumulated signal supplies the writer term in Eq. (7). Its parameter update belongs to the role that produced the representation, connecting independently adapted opening and continuation computations through their shared generated history.

When the efective window includes the opening, continuation losses supervise its context representation through $\Delta \theta _ { \mathrm { f i r s t } }$ . For $s > 0$ , the opening is fixed conditioning and receives no update. Algorithm 1 summarizes the rollout and reverse context-gradient pass.

Optimization and Inference. We use AdamW with generator/fake-score learning rates $5 \times 1 0 ^ { - 5 } / 1 0 ^ { - 5 } , \beta =$ (0, 0.999), weight decay 0.01, gradient clipping at 1, global batch 16, and seed 42. Training perturbs noninitial writer inputs up to noise level 0.15; writing still uses clean time, and replay retains the sampled perturbation.

Inference uses unperturbed context and three Euler evaluations per chunk. With $u _ { b , 0 } = \epsilon _ { b }$ and velocity prediction $\nu _ { \theta _ { b } } ^ { m }$ , each evaluation updates

$$
\begin{array} { r } { \boldsymbol { u } _ { b , \boldsymbol { k } + 1 } ^ { m } = \boldsymbol { u } _ { b , \boldsymbol { k } } ^ { m } + ( \sigma _ { \boldsymbol { k } } ^ { m } - \sigma _ { \boldsymbol { k } + 1 } ^ { m } ) \quad } \\ { \cdot \nu _ { \theta _ { b } } ^ { m } ( \boldsymbol { u } _ { b , \boldsymbol { k } } , \boldsymbol { q } _ { b } , \mathcal { H } _ { < b } ) . } \end{array}\tag{10}
$$

For $u _ { i } = 1 - i / 4 9$ , the noise schedule is $\sigma _ { i } ^ { m } = \gamma _ { m } u _ { i } / [ 1 + ( \gamma _ { m } - 1 ) u _ { i } ]$ , with $\gamma _ { \nu } = 1 2 , \gamma _ { a } = 3$ and retained indices (0, 16, 33, 49). The selected role remains active throughout denoising and clean-context writing.

## 3.5 Local Gradient Analysis of Parameter Sharing

To compare the partitions in Figure 1(b), introduce four virtual adapter copies, one per temporal-role–operation pair: ${ \boldsymbol { x } } = \left( \theta _ { \mathrm { f i r s t } G } , \theta _ { \mathrm { f i r s t W } } , \theta _ { \mathrm { c o n t } G } , \theta _ { \mathrm { c o n t W } } \right)$ . Temporal role specialization ties generation and writing within each temporal role; SGF+’s operation split ties opening and continuation within each operation [7].

Consider a fixed replay at the common adapter initialization, holding the sampled rollout, historical latent inputs, prediction noise, and distribution-matching targets fixed. Let $g _ { r G }$ and $g _ { r W }$ denote the gradient contributions through generation and writing for temporal role �. The contrasts retained by the two assignments are

$$
\begin{array} { r } { \nu _ { T } = ( g _ { \mathrm { f i r s t } G } + g _ { \mathrm { f i r s t } W } ) - ( g _ { \mathrm { c o n t } G } + g _ { \mathrm { c o n t } W } ) , } \\ { \nu _ { O } = ( g _ { \mathrm { f i r s t } G } + g _ { \mathrm { c o n t } G } ) - ( g _ { \mathrm { f i r s t } W } + g _ { \mathrm { c o n t } W } ) . } \end{array}\tag{11}
$$

For a fixed norm $\rho$ of the four-block parameter update, let $D _ { T }$ and $D _ { O }$ be the largest first-order loss decreases admitted by temporal and operation sharing. Orthogonal projection gives

$$
D _ { T } ^ { 2 } - D _ { O } ^ { 2 } = \frac { \rho ^ { 2 } } { 4 } \big ( \| \nu _ { T } \| ^ { 2 } - \| \nu _ { O } \| ^ { 2 } \big ) .\tag{12}
$$

Both assignments retain the common gradient while preserving a diferent specialization direction. Temporal role specialization admits the larger local decrease when the establishment–continuation contrast exceeds the denoising–writing contrast in norm. For a window starting after chunk zero, the opening belongs to the fixed prefix, so $g _ { \mathrm { f i r s t } G } = g _ { \mathrm { f i r s t } W } = 0$ and

$$
D _ { T } ^ { 2 } - D _ { O } ^ { 2 } = \rho ^ { 2 } \langle g _ { \mathrm { c o n t } G } , g _ { \mathrm { c o n t } W } \rangle .\tag{13}
$$

In this case, aligned generation and writing gradients favor sharing within continuation; opposing gradients favor operation separation. The relations characterize a fixed replay’s local update geometry; our experiments compare the resulting trained generators.

## 4 Experiments

We evaluate TaoMate-R1 through parameter-sharing comparisons on a common streaming backbone, followed by system-level generation results.

## 4.1 Experimental Setup

Cases and Parameter-Sharing Configurations. We evaluate 32 cases: ten short person–object interactions, ten approximately two-minute streams with twenty successive instructions each, and twelve public reference compositions from MultiRef-Compass [30]. Short and public clips are approximately ten seconds long. On the H3 backbone, Shared uses one rank-128 adapter for generation and writing across all chunks. Operation split uses two rank-128 adapters, separating generation from writing following SGF+. TaoMate-R1 implements Temporal split with two rank-128 adapters, separating opening from continuation. The two splits have the same adapter budget; their grouping of computations is the variable of interest. All three configurations use the same training cohort, reference inputs, instructions, history-retention policy, and three-evaluation Euler solver. Short and public cases generate native audio; long cases follow cached teacher-audio trajectories and use end-frame conditions for object-removal segments. The training cohort contains 2,196 examples of 4–15 seconds, covering

Table 1. Multi-reference generation on 32 cases. The upper block gives external systems; the lower block compares adaptation assignments on the same autoregressive backbone. TaoMate-R1 uses Temporal split. Values are case-wise means; bold indicates the best displayed mean within each block. Long-case external outputs concatenate instruction clips; autoregressive variants retain history.
<table><tr><td rowspan="2">Method</td><td colspan="2">Reference similarity</td><td rowspan="2">Text</td><td colspan="2">Temporal similarity</td><td colspan="3">VABench</td></tr><tr><td>CLIP↑</td><td>DINO↑</td><td>CLIP↑ CLIP↑</td><td>DINO ↑</td><td>ViCLIP↑</td><td>Realism ↑</td><td>QA↑</td></tr><tr><td>MiniMax-H3</td><td>0.660</td><td>0.355</td><td>0.309</td><td>0.934</td><td>0.852</td><td>0.218</td><td>4.50</td><td>0.825</td></tr><tr><td>VACE</td><td>0.680</td><td>0.390</td><td>0.323</td><td>0.946</td><td>0.893</td><td>0.217</td><td>4.59</td><td>0.906</td></tr><tr><td>SkyReels-V3</td><td>0.664</td><td>0.372</td><td>0.313</td><td>0.919</td><td>0.842</td><td>0.208</td><td>4.09</td><td>0.575</td></tr><tr><td>Phantom</td><td>0.664</td><td>0.354</td><td>0.321</td><td>0.927</td><td>0.866</td><td>0.194</td><td>4.19</td><td>0.563</td></tr><tr><td>Shared</td><td>0.656</td><td>0.385</td><td>0.306</td><td>0.951</td><td>0.914</td><td>0.215</td><td>4.63</td><td>0.925</td></tr><tr><td>Operation split</td><td>0.655</td><td>0.388</td><td>0.304</td><td>0.953</td><td>0.928</td><td>0.213</td><td>4.66</td><td>0.938</td></tr><tr><td>TaoMate-R1</td><td>0.656</td><td>0.397</td><td>0.305</td><td>0.957</td><td>0.934</td><td>0.206</td><td>4.69</td><td>0.950</td></tr></table>

Table 2. Specialization across interaction settings. Short single-instruction clips, long changing-instruction streams, and public reference compositions. TaoMate-R1 uses Temporal split. Scores are case-wise means; higher is better. Bold indicates the best displayed mean within each group.
<table><tr><td>Generator</td><td> $\mathrm { D I N O } _ { \mathrm { r e f } }$   $\mathrm { C L I P _ { t e m p } }$ </td><td></td><td> ${ \mathrm { D I N O } } _ { \mathrm { t e m p } }$ </td><td>QA</td></tr><tr><td>Short (10 cases)</td><td></td><td></td><td></td><td></td></tr><tr><td>Shared</td><td>0.525</td><td>0.967</td><td>0.962</td><td>0.820</td></tr><tr><td>Operation split</td><td>0.531</td><td>0.969</td><td>0.972</td><td>0.900</td></tr><tr><td>TaoMate-R1</td><td>0.545</td><td>0.970</td><td>0.975</td><td>0.880</td></tr><tr><td>Long (10 cases)</td><td></td><td></td><td></td><td></td></tr><tr><td>Shared</td><td>0.469</td><td>0.943</td><td>0.891</td><td>1.00</td></tr><tr><td>Operation split</td><td>0.486</td><td>0.948</td><td>0.914</td><td>1.00</td></tr><tr><td>TaoMate-R1</td><td>0.478</td><td>0.956</td><td>0.921</td><td>0.980</td></tr><tr><td>Public (12 cases)</td><td></td><td></td><td></td><td></td></tr><tr><td>Shared</td><td>0.200</td><td>0.944</td><td>0.892</td><td>0.950</td></tr><tr><td>Operation split</td><td>0.187</td><td>0.944</td><td>0.903</td><td>0.917</td></tr><tr><td>TaoMate-R1</td><td>0.207</td><td>0.948</td><td>0.911</td><td>0.983</td></tr></table>

1,969 object-preserving, 119 object-removal, and 108 object-introduction interactions. Training excludes first- and last-frame image conditions. For long evaluation streams, each product presentation is followed by removal, with the person-and-scene reference supplied as the removal segment’s end-frame condition.

Metrics and Aggregation. CLIP and DINOv2 measure whole-frame reference similarity and temporal similarity between successive 1 Hz samples [31, 32]. Full-prompt CLIP and ViCLIP assess text alignment; VABench measures realism and five-question case-level visual QA [33]. At instruction changes, CLIP and DINO compare adjacent boundary frames; Luma is the absolute diference in their mean grayscale intensity (0–255). Scores use equal case weights; paired bootstrapping resamples cases while retaining each complete stream and its matched model outputs. We draw 100,000 paired resamples and report unadjusted percentile 95% intervals.

Table 3. Instruction-boundary continuity. Ten long sequences; each cell shows the mean above ± population standard deviation. TaoMate-R1 uses Temporal split. CLIP/DINO compare adjacent boundary frames; Luma is the absolute change in mean luminance. Bold indicates the best mean.
<table><tr><td>Generator</td><td>CLIP↑</td><td>DINO↑</td><td>Luma ↓</td></tr><tr><td rowspan="3">Shared</td><td>0.979</td><td>0.978</td><td rowspan="3">1.86</td></tr><tr><td></td><td>± 0.00737 ± 0.00585 ± 0.626</td></tr><tr><td></td><td></td></tr><tr><td rowspan="3">Operation split</td><td>0.979</td><td>0.973 ±0.0160</td><td>1.96 ± 0.456</td></tr><tr><td>±0.00751</td><td></td><td></td></tr><tr><td>0.991</td><td>0.990</td><td>0.613</td></tr><tr><td>TaoMate-R1</td><td></td><td> $\pm 0 . 0 0 2 6 8 \pm 0 . 0 0 4 1 2 \pm 0 . 1 0 3$ </td><td></td></tr></table>

## 4.2 Effect of the Specialization Axis

The same-backbone block of Table 1 compares all three assignments. TaoMate-R1 achieves an aggregate DINO temporal similarity of 0.934, compared with 0.928 for Operation split and 0.914 for Shared.

Table 2 separates short interactions, long changing-instruction streams, and public reference compositions. TaoMate-R1 obtains the highest mean CLIP and DINO temporal similarity in all three groups. On short interactions, it also leads in DINO reference similarity, reaching 0.545 versus 0.531 for Operation split. Its CLIP and DINO temporal similarities are 0.970 and 0.975, compared with 0.969 and 0.972 for Operation split, while Operation split has higher visual QA (0.900 versus 0.880).

On long streams, TaoMate-R1 reaches CLIP and DINO temporal similarities of 0.956 and 0.921, compared with 0.948 and 0.914 for Operation split. The paired diference in DINO temporal similarity (TaoMate-R1 minus Operation split) is +0.00730 (95% interval [−0.00958, 0.0227]). Operation split has higher DINO reference similarity (0.486 versus 0.478). Its QA of 1.00 and TaoMate-R1’s 0.980 correspond to 50 and 49 correct answers across five questions per case. On public reference compositions, TaoMate-R1 leads all four reported metrics, with DINO reference similarity of 0.207, CLIP and DINO temporal similarities of 0.948 and 0.911, and visual QA of 0.983.

Interaction Settings and Shared Adaptation. Both two-adapter configurations improve DINO temporal similarity over Shared in every evaluated group. At the same adapter budget, TaoMate-R1 further increases the temporal means across all three settings, supporting the assignment of adaptation capacity by opening and continuation. The aggregate score combines twenty-two short or public cases with ten long streams under equal case weighting; the subgroup results show higher temporal means for TaoMate-R1 within each group. Reference similarity and visual QA vary by setting, indicating that higher temporal similarity does not imply uniformly better reference matching or visual QA.

Figure 3 examines background appearance in fixed foliage and vase crops over a 119-second stream. Views at 0, 60, and 119 seconds show the evolution of leaf and ceramic tones. The overlays average per-pixel CIE Lab distances from each video’s initial frame over the full stream at 1 Hz. TaoMate-R1 shows smaller appearance changes than Operation split in these regions, complementing the whole-frame temporal measurements.

Interaction Dynamics. Figure 4 compares the three sharing schemes at matched timestamps within two changinginstruction streams. In the bag sequence, all three configurations present the referenced item at 72 and 74 s. At 76 s, TaoMate-R1 moves it toward the instructed screen-right side, Shared moves it screen-left, and Operation split still holds it centrally. In the second sequence, Shared and TaoMate-R1 proceed from the notebook to the folded umbrella; Operation split remains empty-handed at 106 and 108 s. The TaoMate-R1 frames show evolving hand poses and object states within the retained scene. Together with the background views, these sequences make both components of streaming interaction visible: carrying out a new action and preserving the scene in which it unfolds.

Operation split  
(a) Background foliage  
![](images/bed2cdb969f24978eb45ac6864274395eba86f0a2e55c845c68930c84f437975.jpg)  
Figure 3. Background color and appearance over a long stream. TaoMate-R1 uses the Temporal split configuration. Fixed foliage and vase crops from case 008, enlarged at 0, 60, and 119 s. Below each triplet, the overlay shows mean CIE Lab distance from the initial frame over 119 samples at 1 Hz. All methods use identical crop coordinates and a shared scale (saturated at 12; opacity 0.45).

## 4.3 Continuity across Instruction Changes

Table 3 compares adjacent frames at the same instruction boundaries in the ten long streams. TaoMate-R1 obtains CLIP and DINO boundary similarities of 0.991 and 0.990, compared with 0.979 and 0.973 for Operation split. Its mean luminance jump is 0.613 versus 1.96, a reduction of 1.35 gray levels, or 68.7%. Shared obtains boundary similarities of 0.979 and 0.978 and a mean luminance jump of 1.86.

The case-paired analysis gives a diference (TaoMate-R1 minus Operation split) of +0.0177 in DINO boundary similarity (95% interval [0.00820, 0.0288]) and −1.35 in luminance jump ([−1.58, −1.12]). Luminance jumps decrease in all ten streams, and CLIP and DINO boundary similarities each increase in nine streams. These transition-level gains are more consistent across cases than the sequence-level DINO temporal diference.

Continuity under Changing Conditions. An instruction change introduces a new interaction goal while the preceding scene remains the starting state for continuation. Boundary measurements evaluate the immediate transition from the last frame under one instruction to the first under the next. Sequence-level temporal scores also include the intervening within-instruction evolution. Together, they show higher temporal-similarity means over the full streams, with the clearest paired evidence at instruction changes. The background views in Fig. 3 extend this analysis to retained scene appearance over the complete stream. These observations support temporal specialization for maintaining scene continuity as interaction goals change.

## 4.4 Generation Quality

Reference-Conditioned Systems. The external-system block of Table 1 compares MiniMax-H3 [3], VACE [2], SkyReels-V3 [34], and Phantom [1]. For the long cases, these systems generate each instruction segment independently and concatenate the resulting clips; our autoregressive variants retain history across instruction changes. TaoMate-R1 leads in mean DINO reference similarity, both temporal similarity metrics, realism, and visual QA. VACE leads in CLIP reference and full-prompt CLIP scores, and MiniMax-H3 in ViCLIP.

## (a) Cosmetics-bag interaction

![](images/388471c99d2e390b436bb3b0733a0cd618efd53f82e550d5472a5dd196ae86d2.jpg)  
Figure 4. Interaction sequences under three parameter-sharing schemes. TaoMate-R1 uses Temporal split. Origina references are at left; each column shows the same timestamp across all three H3 configurations. (a) Bag presentation and the start of a screen-right set-aside action (case 008). (b) Successive notebook and folded-umbrella presentations (case 006). TaoMate-R1 changes the held object and interaction state while retaining the presenter and room; Operation split remains empty-handed in the sampled umbrella segment.

Qualitative Comparisons. Figure 5 shows four reference-conditioned interactions. In the gallery scene, MiniMax H3 and TaoMate-R1 depict both cartoon characters interacting with a guitar. VACE renders live-action participants, while SkyReels-V3 and Phantom depict Santa as a small held figure. In the scarf scene, our output includes an additional guitar. The continuous-interaction rows show cosmetics-bag and closed-umbrella presentation within the established scene. At 72 s, our stream depicts the sage-green bag alongside the presenter in the established room.

Long-Stream System Comparison. Table 4 compares continuous generation with independently generated instruction clips on the ten long cases. TaoMate-R1 achieves DINO temporal similarity of 0.921, compared with 0.812–0.848 for the clip-generation systems. At segment boundaries, DINO similarity is 0.990 versus 0.589–0.831, and the mean luminance jump is 0.613 versus 9.12–14.3. These results compare the complete generation protocols; the same-backbone comparisons above isolate the choice of adaptation assignment.

Inputs  
MiniMax-H3  
VACE  
SkyReels-V3  
Phantom  
TaoMate-R1  
![](images/1bbcb0ed50541ae981c72acf549dbb8f4f2f75b796aeb9c9e230e3ff0f1e6403.jpg)  
Figure 5. Multi-reference interaction results. Reference inputs at left; outputs are landscape crops. Rows: scarf adjustment (MultiOmni-board3-001, 4 s); guitar interaction (MultiOmni-board3-003, 9 s; SkyReels-V3/Phantom: 4.5 s); cosmetics-bag presentation (case 008, 72 s); and closed-umbrella presentation (case 006, 108 s).

Table 4. Minute-scale continuity. Equal-case means over ten approximately two-minute sequences. ${ \mathsf { D } } { \mathsf { I N O } } _ { \mathrm { t e m p } }$ uses successive 1 Hz samples; $\mathsf { D } | \mathsf { N O } _ { \mathrm { b n d } }$ and Luma compare adjacent segment-boundary frames.
<table><tr><td>Method</td><td> ${ \mathrm { D I N O } } _ { \mathrm { t e m p } }$  ↑</td><td> $\mathrm { D I N O } _ { \mathrm { b n d } }$  ↑ Luma↓</td></tr><tr><td>MiniMax-H3</td><td>0.832</td><td>0.831 10.2</td></tr><tr><td>VACE</td><td>0.848</td><td>0.645 9.12</td></tr><tr><td>SkyReels-V3</td><td>0.815</td><td>0.609 14.3</td></tr><tr><td>Phantom</td><td>0.812</td><td>0.589 13.8</td></tr><tr><td>TaoMate-R1</td><td>0.921</td><td>0.990 0.613</td></tr></table>

Figure 6 arranges frame columns sampled every 0.5 s along a timeline. Independent clip joins produce bands in head height, body scale, and background appearance, while the continuous stream maintains more stable presenter and table geometry.

The time-column representation relates these bands to changes in scene composition. A change in camera framing or subject scale shifts several horizontal structures together, producing a discontinuity across the sampled timeline. In the continuous stream, the table edge and background patterns trace more continuous paths through successive instructions. This connects the boundary scores in Table 4 to visible scene properties such as presenter scale and background placement. Together, the visual comparisons examine reference-conditioned interactions in Fig. 5, localized background appearance in Fig. 3, and scene-level continuity in Fig. 6.

Audiovisual Measurements. On the 22 native-audio cases, TaoMate-R1 has higher mean AV alignment than Operation split (4.77 versus 4.59). Operation split has higher audio QA (0.809 versus 0.736) and lower Desync (0.282 versus 0.445), showing a diferent ordering for audio content and synchronization.

![](images/1d7cf1298ad30ef004132c1920de2cb71c39a54aa75058d52e2dfbc3c06c2da3.jpg)  
Figure 6. Scene continuity over a two-minute interaction. Each row combines the first frame’s left half, centre columns sampled every 0.5 s, and the last frame’s right half (case 006, 119 s). Grey ticks mark instruction changes; endpoints retain native aspect ratios. The bands contrast composition changes in clip-wise generation with stable presenter and table geometry in the TaoMate-R1 stream.

## 5 Conclusion

We presented TaoMate-R1, a streaming multi-reference generator that groups denoising and context writing by scene establishment and continuation, with SGF coupling the roles through generated history. A local gradient analysis characterizes the directions retained by this grouping and by operation-based sharing. On the H3 backbone, TaoMate-R1 achieves higher mean CLIP and DINO temporal similarity across the three evaluated interaction settings at the same two-adapter budget. It also reduces discontinuities at instruction boundaries in long streams. The resulting comparison makes the specialization axis an explicit design variable in streaming multi-reference generation.

## References

[1] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14951–14961, 2025. URL https: //openaccess.thecvf.com/content/ICCV2025/html/Liu\_Phantom\_Subject-Consistent\_Video\_Generation\_ via\_Cross-Modal\_Alignment\_ICCV\_2025\_paper.html.

[2] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Jiang\_VACE\_All-in-One\_Video\_Creation\_and\_Editing\_ICCV\_2025\_paper.html.

[3] MiniMax. MiniMax-H3: Oficial model repository. https://github.com/MiniMax-AI/MiniMax-H3, 2026. Ref2VA reference-conditioned interface. Accessed October 27, 2026.

[4] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Yin\_From\_Slow\_Bidirectional\_to\_Fast\_Autoregr essive\_Video\_Diffusion\_Models\_CVPR\_2025\_paper.html.

[5] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2506.08009.

[6] Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, Haoran Li, Haoyang Huang, Zeyue Xue, and Nan Duan. Self gradient forcing: Native long video extrapolation. arXiv preprint arXiv:2607.20368, 2026. URL https: //arxiv.org/abs/2607.20368.

[7] Zihan Su, Junhao Zhuang, Yaowei Li, Siwen Lu, Haoran Li, Lingen Li, Haoyu Wu, Weiyang Jin, Songchun Zhang, Haoyang Huang, Chun Yuan, Zeyue Xue, and Nan Duan. SGF+: Decoupling gradient flows for autoregressive video generation. arXiv preprint arXiv:2610.10429, 2026. URL https://arxiv.org/abs/2610.1 0429.

[8] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identity-preserving text-to-video generation by frequency decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12978–12988, 2025. URL https://openacce ss.thecvf.com/content/CVPR2025/html/Yuan\_Identity-Preserving\_Text-to-Video\_Generation\_by\_Freque ncy\_Decomposition\_CVPR\_2025\_paper.html.

[9] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6099–6110, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Chen\_Multisubject\_Open-set\_Personalization\_in\_Video\_Generation\_CVPR\_2025\_paper.html.

[10] Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Movie weaver: Tuning-free multi-concept video personalization with anchored prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13146–13156, 2025. URL https://openacce ss.thecvf.com/content/CVPR2025/html/Liang\_Movie\_Weaver\_Tuning-Free\_Multi-Concept\_Video\_Pers onalization\_with\_Anchored\_Prompts\_CVPR\_2025\_paper.html.

[11] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, Yingya Zhang, and Hongming Shan. DreamVideo-2: Zero-shot subjectdriven video customization with precise motion control. arXiv preprint arXiv:2410.13830, 2024. URL https://arxiv.org/abs/2410.13830

[12] Sharath Girish, Viacheslav Ivanov, Tsai-Shien Chen, Hao Chen, Aliaksandr Siarohin, and Sergey Tulyakov. AlcheMinT: Fine-grained temporal control for multi-reference consistent video generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23227–23237, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Girish\_AlcheMinT\_Fine-grained\_Temporal\_Con trol\_for\_Multi-Reference\_Consistent\_Video\_Generation\_CVPR\_2026\_paper.html.

[13] Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. MultiShotMaster: A controllable multi-shot video generation framework. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16268–16278, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/papers/Wang\_MultiShotMaster\_A\_Controlla ble\_Multi-Shot\_Video\_Generation\_Framework\_CVPR\_2026\_paper.pdf.

[14] Ziyao Huang, Zhengkun Rong, Shiyang Qin, Shuang Liang, Wentao Hu, Yuxuan Luo, Yuan Zhang, and Mingyuan Gao. FlowAct-R2: Beyond talking avatar via streaming multimodal references and proactive agent planning. arXiv preprint arXiv:2609.35728, 2026. URL https://arxiv.org/abs/2609.35728.

[15] Shuolin Xu, Guanyuan Li, Boyu Li, Kelu Yao, Xiaogang Xu, Hongchuan Yu, Shuicheng Yan, and Xiaobin Hu. ReferStream: Interactive subject injection for streaming video generation. arXiv preprint arXiv:2610.37566, 2026. URL https://arxiv.org/abs/2610.37566.

[16] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion Forcing: Next-token prediction meets full-sequence difusion. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2407.01392.

[17] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. SkyReels-V2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. URL https://arxiv.org/abs/2504.13074.

[18] Roberto Henschel,Levon Khachatryan,Hayk Poghosyan,Daniil Hayrapetyan,Vahram Tadevosyan,Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2568–2577, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Hensch el\_StreamingT2V\_Consistent\_Dynamic\_and\_Extendable\_Long\_Video\_Generation\_from\_Text\_CVPR\_2 025\_paper.html.

[19] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video difusion models. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/2bd e8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html.

[20] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. LongLive: Real-time interactive long video generation. In International Conference on Learning Representations, 2026. URL https://openreview.net/fo um?id=nCAODkpsPJ.

[21] Qixin Hu, Shuai Yang, Wei Huang, Song Han, and Yukang Chen. LongLive-RAG: A general retrievalaugmented framework for long video generation. arXiv preprint arXiv:2606.02553, 2026. URL https: //arxiv.org/abs/2606.02553.

[22] Xiangyang Luo, Qingyu Li, Xiaokun Liu, Wenyu Qin, Miao Yang, Meng Wang, Pengfei Wan, Di Zhang, Kun Gai, and Shao-Lun Huang. FilmWeaver: Weaving consistent multi-shot videos with cache-guided autoregressive difusion. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(9):7689–7697, 2026. doi: 10.1609/aaai.v40i9.37711. URL https://ojs.aaai.org/index.php/AAAI/article/view/37711.

[23] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, volume 28, 2015. URL https://arxiv.org/abs/1506.03099.

[24] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling Forcing: Autoregressive long video difusion in real time. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2509.25161.

[25] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. DreamVideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Wei\_DreamVideo\_Composing\_Your\_ Dream\_Videos\_with\_Customized\_Subject\_and\_Motion\_CVPR\_2024\_paper.html.

[26] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. eDif-I: Text-toimage difusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. URL https://arxiv.org/abs/2211.01324.

[27] Wan Team. Wan2.2: Oficial model repository. https://github.com/Wan-Video/Wan2.2, 2025. High- and low-noise expert design. Accessed October 29, 2026.

[28] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. URL https://arxiv.org/abs/2106.09685.

[29] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2405.14867.

[30] Xiaohan Zhang, Yuqing Wen, Junlin Chen, Yuqi Tang, Yiting He, Lizhuo Shao, Weiming Zhu, Tengfei Liu, Yang Shi, Jialu Chen, Yuanxing Zhang, and Huaxiong Li. MultiRef-Compass: Towards comprehensive evaluation of multi-reference-to-audio-video generation. arXiv preprint arXiv:2607.14189, 2026. URL https://arxiv.org/abs/2607.14189.

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[32] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel

Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. URL https://arxiv.org/abs/2304.07193.

[33] Daili Hua, Xizhi Wang, Bohan Zeng, Xinyi Huang, Hao Liang, Junbo Niu, Xinlong Chen, Quanqing Xu, and Wentao Zhang. VABench: A comprehensive benchmark for audio-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23345–23355, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Hua\_VABench\_A\_Comprehensive\_Benchmark for\_Audio-Video\_Generation\_CVPR\_2026\_paper.html.

[34] Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu, Jiahua Wang, Baoxuan Gu, Mingshan Chang, Wenjing Cai, Yuqiang Xie, Binjie Mao, Youqiang Zhang, Nuo Pang, Hao Zhang, Yuzhe Jin, Zhiheng Xu, Dixuan Lin, Guibin Chen, and Yahui Zhou. SkyReels-V3 technique report. arXiv preprint arXiv:2601.17323, 2026. URL https://arxiv.org/abs/2601.17323.