# Learning Robotic World Models with Semantic and Visual Feedback

Hao Wu, Yuqi Li, Yizhou Zhao, Kun Wang, Xiao Luo, Yu Cheng, Bo An, Yingli Tian, Fellow, IEEE, Laszlo A. Jeni, and Ming-Hsuan Yang<sup>#</sup>, Fellow, IEEE

<sup>§</sup> Project & Code Data & Weights (Hugging Face)

Abstract—Robot world models predict action-conditioned future visual states to support robotic planning and decision making. However, visual reconstruction does not fully capture task progress, semantic evaluation alone provides limited constraints on fine-grained visual errors, and homogeneous predictions limit comparative feedback. We present ROBOIMAGINE, a framework for post-training autoregressive robotic world models through candidate-future comparisons with semantic and visual feedback. Diverse search promotes diversity among candidate trajectories under shared visual observations, action sequences, and task instructions. A frozen multimoda evaluator provides task-relevant semantic preferences through pairwise video comparisons in both presentation orders, complemented by pixel- and perceptual-level visual consistency feedback that constrains local structural and appearance errors. These signals drive policy optimization through group-relative advantages. Fixed initial context and branch-specific rolling histories support long-sequence generation with bounded context and consistent conditioning between candidate generation and policy optimization. Experiments with separately trained models on RT-1, BridgeV2, and RoboCasa cover real-world and simulated manipulation. On RT-1, under matched initialization, update counts, and evaluation protocols, ROBOIMAGINE improves PSNR by 1.26 dB and reduces LPIPS by 11.93% relative to continued supervised fine-tuning. Ablations show that diverse search improves prediction quality and that combined semantic and visual feedback yields higher external multimodal evaluation scores than visual feedback alone while maintaining comparable visual fidelity. Rolling histories further improve long-horizon prediction. These results support improving visual fidelity, task relevance, and long-horizon prediction through candidate comparison and complementary feedback.

Index Terms—Robotic world models, reinforcement learning, preference feedback, multimodal judges, visual consistency, diverse search.

## 1 INTRODUCTION

Robot world models predict how scenes evolve under actions, enabling robots to assess potential outcomes before execution [1–4]. As learnable internal simulators, they support planning, policy evaluation, and long-horizon reasoning [5–8]. Unlike generic video generation, which primarily emphasizes visual quality, robot world models must capture action-induced state changes, object contacts, and interactions to provide reliable guidance for manipulation decisions [9, 10]. Visual observations describe the current scene, action sequences specify the execution process, and natural-language instructions provide task goals; together, they form the multimodal context for understanding and predicting robot interactions. Prediction quality therefore depends not only on visual fidelity, but also on how accurately the model captures physical evolution under the given actions and its relationship to the task goal [11, 12].

However, evaluating predicted futures remains a central challenge in training robot world models. Existing methods typically use maximum-likelihood, reconstruction, or perceptual objectives [10, 13–15]. These objectives provide effective visual supervision, but visual similarity alone does not fully capture task-relevant changes. For example, a prediction that preserves scene appearance may still miss action-induced object displacement or changes in contact.

(a) Task context

![](images/5e3553d7ddf8da5358dd89e6efaf19647507ba7489eb69fe8eb6dc14389023a6.jpg)

"Put sushi in the pot. Goal: move sushi from the table into the pot Shared scene, actions, and instruction

(b) Visual similarity is not task progress

![](images/93f730ed88cc07e4c748d4db48af162feeef5b8cc4b76599bece223e18ac7acc.jpg)

![](images/5b96db2aef7bf21ebf7e9955b1d64c08b73668f5d79a6cdda5197a375d375465.jpg)

(c) Complementary feedback

![](images/e0b1994eabd6a6338c8a0a8e15f4437a5be8fe6544b85e144f704a2ffd6567a8.jpg)  
Fig. 1. Complementary feedback for robot world models. Preserved appearance can coexist with stalled task progress, while task advancement can coexist with local visual artifacts. Visual fidelity and task semantics therefore provide complementary feedback. Scenes are conceptual illustrations, not experimental predictions or measured rankings.

Conversely, multimodal semantic evaluation assesses visible task progress using instructions and scene context, yet may be insufficiently sensitive to local structural distortions and fine-grained appearance errors. As Figure 1 illustrates, visual fidelity and task semantics provide complementary perspectives on prediction quality. Aligning robot world models therefore calls for both forms of feedback: constraining visual detail while assessing whether predicted state transitions are plausible under the given actions and task conditions.

Prior work explores reward-based post-training for robotic world models [16]. RLVR-World rewards actionconditioned forecasts using visual prediction metrics [10]. RoboAlign-R1 [17] distills a vision–language judge into a lightweight reward model that scores individual generated videos. Such semantic feedback motivates a broader question: how should alternative futures be constructed and evaluated to provide informative learning signals? Absolute video scores require calibration across scenes and tasks, whereas futures generated under shared observations, actions, and instructions can be compared directly for visible task progress. However, nearly identical candidates provide little basis for comparison, and semantic preferences alone may overlook local visual errors. Effective comparative learning must therefore address both candidate construction and feedback design. We investigate how diverse candidate futures and complementary semantic and visual feedback jointly shape the learning of robotic world models.

We propose ROBOIMAGINE, a framework for learning autoregressive robotic world models by comparing candidate futures with semantic and visual feedback. Conditioned on visual observations, actions, and task instructions, grouped diverse beam search (GDBS) [18] constructs alternative trajectories under shared inputs. A frozen multimodal evaluator, mJev (hereafter Jev) [19], compares candidate videos for visible task progress using the instruction and initial observation; evaluating both presentation orders mitigates position bias. Pixel- and perceptual-level visual consistency feedback [20] complements these preferences by constraining fine-grained prediction errors. The two signals are standardized across candidates and combined into group-relative advantages for policy optimization [21]. Each branch retains fixed initial context and a dynamic anchor together with its own rolling state–action history. This bounded history supports continued generation, while replaying the recorded history keeps policy scoring consistent with the conditions under which each candidate is generated.

We evaluate separately trained models on RT-1, BridgeV2, and RoboCasa, and conduct controlled studies of posttraining, candidate search, feedback composition, and memory. On RT-1, comparative post-training improves PSNR by 1.26 dB and reduces LPIPS by 11.93% relative to continued supervised fine-tuning from the same initialization for the same number of updates, at a higher computational cost. Visual-only feedback achieves the strongest visual metrics in the reward ablation, whereas combined feedback yields a higher external multimodal evaluation mean while retaining comparable visual fidelity. Long-horizon evaluations further assess the benefits of rolling history and the associated inference costs. Our main contributions are summarized as follows:

❶ Learning robot world models by comparing imagined futures. We introduce ROBOIMAGINE, which couples GDBS with comparative post-training under shared observations, actions, and instructions. Bounded branch-specific histories support autoregressive rollouts and are replayed during policy scoring to preserve conditioning consistency.

❷ Complementary semantic preference and visual consistency feedback. A frozen multimodal judge evaluates pairwise task progress among candidates with bidirectional comparisons to mitigate order bias, while pixel- and perceptual-level consistency feedback constrains fine-grained visual errors. Group-wise standardization and relative advantages translate both signals into policy updates without backpropagating through the evaluator or the video decoder.

❸ Evaluation of prediction quality and learning mechanisms. Experiments on three manipulation datasets assess visual prediction and task-level consistency. Controlled RT-1 comparisons isolate the effects of additional supervised training, feedback composition, and candidate search, while long-horizon and window-size studies characterize the quality and inference cost of bounded memory.

## 2 RELATED WORK

Robotic World Models: World models learn environment dynamics for planning and policy learning [1, 6, 22, 23]. Visual prediction represents anticipated interaction outcomes as videos: iVideoGPT models videos with discrete tokens [24], while RoboDreamer, IRASim, and Vid2World condition future generation on task descriptions or robot actions [13, 25, 26]. These approaches draw on autoregressive and diffusion-based visual generation. VideoGPT models sequences of discrete video codes, VAR predicts successive spatial scales, and video diffusion models and diffusion transformers use iterative denoising [27–30]. Cosmos further develops world foundation models for physical AI [31]. ROBOIMAGINE studies post-training GPT-style autoregressive robotic world models, whose discrete dynamics-token probabilities support candidate search and policy evaluation under shared observations, actions, and task instructions.

Semantic and Visual Feedback: Reconstruction and perception guide visual prediction [20, 32], while VideoScore, VBench, WorldModelBench, and PhyCritic evaluate video semantics, dynamics, or physics [9, 33–35]. Trajectory preferences provide supervision [36]; visual generation uses reward-weighted likelihood, policy gradients, and direct preference optimization [37–39]. RLVR-World predicts future observations from video and actions using visual-metric rewards [10]. RoboAlign-R1 distills a multimodal judge to score individual videos [17]. Given shared observations, actions, and instructions, ROBOIMAGINE constructs futures by grouped diverse beam search. Pairwise task preferences from a frozen multimodal evaluator [19] and per-candidate pixel and perceptual scores jointly determine group-relative advantages for policy updates [21]. We evaluate both candidate orders to mitigate position bias [40].

Candidate Search and Long-Horizon Memory: Diverse beam search discourages repeated decoding choices across groups [18]. Long-sequence generation additionally requires managing predicted histories. Diffusion Forcing combines causal modeling with token-wise noise levels, and Self Forcing trains on generated histories to address exposure bias [41, 42]. StreamingLLM retains attention sinks and recent tokens; LongLive uses short-window attention and cache refresh, and LongLive-RAG retrieves relevant past latents [43–45]. RoboAlign-R1 periodically re-encodes visual context [17]. ROBOIMAGINE retains a fixed initial context and dynamic anchor, together with branch-specific rolling state–action histories. It records and replays each candidate’s history for consistent policy scoring, coupling boundedcontext generation with diverse candidate comparison.

## 3 METHOD

Figure 2 presents the overall framework. ROBOIMAGINE first establishes multimodal autoregressive prediction through supervised fine-tuning, then constructs candidate futures with GDBS and optimizes the model using complementary semantic and visual feedback. Each candidate maintains its own rolling history, which is replayed during policy evaluation to associate feedback with the actual prediction path.

## 3.1 GPT-Style Autoregressive Robotic World Model

Given initial observations $O 0 , O 1$ , actions $a _ { 1 : T . }$ , and an instruction l, the model predicts future visual states $x _ { 2 : T + 1 }$ As shown in Figure 3, discrete visual–action tokens are processed by a causal Transformer with text cross-attention. Token probabilities provide a common interface for search and policy optimization.

Visual–action representation: A context-conditioned dual-branch encoder [24] and finite scalar quantization (FSQ) [46] encode the initial scene and subsequent states:

$$
\begin{array} { r l } & { z ^ { \mathrm { c t x } } = Q _ { \mathrm { c t x } } ( E _ { \mathrm { c t x } } ( o _ { 0 } ) ) , } \\ & { z _ { t } ^ { \mathrm { d y n } } = Q _ { \mathrm { d y n } } ( E _ { \mathrm { d y n } } ( x _ { t } ; o _ { 0 } ) ) . } \end{array}\tag{1}
$$

Here, $x _ { 1 } = o _ { 1 }$ and $t = 1 , \dots , T + 1$ . The encoders include feature extraction and projection; the dynamics branch conditions on multiscale initial-scene features. Context and dynamics representations contain $N _ { c } = 1 2 8 0$ and $N _ { d } = 8 0$ tokens, respectively.

Actions are normalized using training-set ranges and discretized dimension-wise into bins, yielding $\hat { a } _ { j } = Q _ { a } ( a _ { j } )$ for the transition from state j to $j + 1$ . The resulting sequence is

$$
\boldsymbol { s } = \left[ z ^ { \mathrm { c t x } } , z _ { 1 } ^ { \mathrm { d y n } } , \hat { a } _ { 1 } , z _ { 2 } ^ { \mathrm { d y n } } , \hat { a } _ { 2 } , . . . \right] .\tag{2}
$$

Predicting $\boldsymbol { z } _ { t } ^ { \mathrm { d y n } }$ therefore uses $a _ { < t } ,$ with $a t - 1$ specifying the current transition. Modality-specific vocabulary offsets distinguish token types within the shared autoregressive backbone.

Text conditioning and autoregressive prediction: A frozen T5 encoder [47] produces instruction features $e \iota ~ =$ $E _ { \mathrm { t e x t } } ( l )$ . Cross-attention at selected layers uses visual–action hidden states as queries and text features as keys and values. Its newly inserted output projections are zero-initialized. Let $P _ { t }$ denote the retained visual–action prefix defined in Section 3.2. Frame tokens factorize as

$$
\pi _ { \boldsymbol { \theta } } \Big ( z _ { t } ^ { \mathrm { d y n } } \mid P _ { t } , e _ { l } \Big ) = \prod _ { n = 1 } ^ { N _ { d } } \pi _ { \boldsymbol { \theta } } \Big ( z _ { t , n } ^ { \mathrm { d y n } } \mid P _ { t } , z _ { t , < n } ^ { \mathrm { d y n } } , e _ { l } \Big ) .\tag{3}
$$

The distribution $\pi _ { \theta }$ is renormalized over valid dynamics tokens. A frozen visual decoder reconstructs each completed frame from its dynamics tokens and the initial context.

Supervised initialization: Let $P _ { t } ^ { \mathrm { G T } }$ contain recorded historical frames and actions, and let $z _ { t , n } ^ { \mathrm { G T } }$ denote the target dynamics token. Supervised fine-tuning minimizes

$$
\mathcal { L } _ { \mathrm { S F T } } = - \frac { 1 } { T N _ { d } } \sum _ { t = 2 } ^ { T + 1 } \sum _ { n = 1 } ^ { N _ { d } } \log p _ { \theta } \left( z _ { t , n } ^ { \mathrm { G T } } \mid P _ { t } ^ { \mathrm { G T } } , z _ { t , < n } ^ { \mathrm { G T } } , e _ { l } \right) ,\tag{4}
$$

where $p _ { \theta }$ is the distribution over the full shared vocabulary, and loss applies only to target dynamics tokens. The visual codec and text encoder remain frozen; the backbone and text cross-attention are trained. The supervised model initializes post-training, and a frozen copy serves as the reference policy.

## 3.2 Diverse Candidate Search

Likelihood-based selection can concentrate candidates on similar prediction paths. GDBS balances model likelihood with cross-group differences, as illustrated alongside MCTS and ordinary beam search in Figure 5.

Branch-specific history: As shown in Figure 4, the retained prefix $P _ { t }$ contains the fixed initial context $z ^ { \mathrm { c t x } }$ , the observed dynamics anchor $z _ { 1 } ^ { \mathrm { d y n } }$ , and each branch’s latest W predicted states with their paired actions, ending with the action for the next transition. Every generated frame is appended to the output, while the input history rolls forward. The window thus limits conditioning history, not output length. We use $W \ = \ 6 ,$ excluding the fixed anchor, and record the actual generation prefixes for policy evaluation.

Grouped search: Search uses G groups with B beams each. At dynamics-token position u, groups are processed sequentially. Let $\gamma _ { h , u }$ be the multiset of final tokens from the B extensions retained by earlier group h. The normalized overlap penalty is

$$
f _ { g , u } ( v ) = \left\{ \begin{array} { l l } { 0 , } & { g = 1 , } \\ { \frac { \sum _ { h < g } \sum _ { v ^ { \prime } \in \mathcal { V } _ { h , u } } \mathbf { 1 } [ v ^ { \prime } = v ] } { ( g - 1 ) B } , } & { g > 1 . } \end{array} \right.\tag{5}
$$

Repeated tokens contribute repeated counts. Under the generating snapshot $\pi _ { \theta _ { \mathrm { o l d } } } ,$ extending prefix y by token v gives

$$
\begin{array} { r l } & { S _ { g , u } ( y v ) = S _ { g , u - 1 } ( y ) } \\ & { \phantom { = } + \log \pi _ { \theta _ { \mathrm { o l d } } } ( v \mid \mathcal { H } ( y ) , e _ { l } ) - \lambda _ { \mathrm { d i v } } f _ { g , u } ( v ) . } \end{array}\tag{6}
$$

The history $\mathcal { H } ( y )$ includes the retained visual–action prefix and preceding within-frame tokens. The penalty changes search scores, not policy normalization. Each group retains the top B extensions across all parent beams and valid tokens, inheriting the selected parents’ histories. We use $G = 4 .$ $B = 2 _ { \cdot }$ , and $\lambda _ { \mathrm { d i v } } = 0 . 1 .$ , with fixed group membership and no length normalization.

Complete candidates: Each trajectory contains $L =$ $T N _ { d }$ future dynamics tokens. From the final beam set $B _ { g , L } .$ we retain

$$
y ^ { ( g ) } = \underset { y \in \mathcal { B } _ { g , L } } { \arg \operatorname* { m a x } } \mathcal { S } _ { g , L } ( y ) , \qquad g = 1 , \dotsc , G .\tag{7}
$$

The frozen codec decodes these trajectories:

$$
\begin{array} { r } { \boldsymbol { y } ^ { ( g ) } \equiv \hat { z } _ { 2 : T + 1 } ^ { \mathrm { d y n } , ( g ) } , } \\ { \hat { x } _ { 2 : T + 1 } ^ { ( g ) } = \mathcal { D } _ { \mathrm { v i s } } ( z ^ { \mathrm { c t x } } , \boldsymbol { y } ^ { ( g ) } ) . } \end{array}\tag{8}
$$

![](images/5b356f4b63e5d6f6e49ac31dee202769a5227ba08012ce7885e7810476dc39b0.jpg)

Fig. 2. Multimodal adaptation and learning from complementary feedback in ROBOIMAGINE. Stage 1 learns dynamics-token prediction conditioned on observations, actions, and text. Stage 2 constructs diverse candidate futures with branch-specific rolling histories and combines semantic preferences from a frozen Jev evaluator with pixel- and perceptual-level visual feedback. Only the visual-feedback branch accesses recorded future frames during post-training. The generator and its text cross-attention receive policy gradients; the visual codec and feedback models remain frozen.  
![](images/45dd17e1bb1d2c3490b41327a46fd6c67bdc1dd9fc4ae63a6746dd7dac2f196d.jpg)  
Fig. 3. Quantized representation and autoregressive prediction in ROBOIMAGINE. (a) Context-conditioned visual encoding produces 1,280 context tokens and 80 dynamics tokens per frame through finite scalar quantization (FSQ); discretized actions form the shared visual–action prefix. FSQ uses an implicit grid with 4,375 codes per visual branch. (b) A 12-layer LLaMA decoder predicts dynamics tokens. The expanded layer shows causal attention, rotary positions, output projection, and the gated feed-forward block with residual connections. Text cross-attention is inserted before layers 4, 8, and 12; text features are not discrete prefix tokens.

This produces one candidate per group, sharing external conditions while retaining branch-specific histories. We use i to index the resulting candidates below.

## 3.3 Complementary Semantic and Visual Feedback

Semantic feedback measures relative task progress, while visual feedback constrains prediction errors. Their fusion assigns a reward to each candidate trajectory.

Semantic preference feedback: A frozen multimodal evaluator $J _ { \phi } ,$ implemented with the 4B mJev checkpoint (abbreviated as Jev) [19], compares candidate pairs using instruction $l ,$ initial observation $^ { O 1 , }$ and eight temporally matched frames from each video. With frame-sampling operator ${ \mathcal { F } } _ { 8 } ,$ , its output is

$$
\begin{array} { r l } & { \mathbf { q } _ { i j } = J _ { \phi } \Big ( l , o _ { 1 } , \mathcal { F } _ { 8 } ( \hat { x } ^ { ( i ) } ) , \mathcal { F } _ { 8 } ( \hat { x } ^ { ( j ) } ) \Big ) } \\ & { \quad \quad = [ q _ { i j } ^ { \succ } , q _ { i j } ^ { \prec } , q _ { i j } ^ { \equiv } , q _ { i j } ^ { \ ? } ] . } \end{array}\tag{9}
$$

The four probabilities denote preference for the first or second candidate, comparable progress, and insufficient

![](images/4def0881c85dfee38756921cbce9be960dfc986c25b863f3f808d564b8aceef7.jpg)

Fig. 4. Fixed-anchor branch memory for long-horizon generation in ROBOIMAGINE. (a) The retained prefix combines fixed initial context and an observed dynamics anchor with six recent predicted states and paired actions. (b) A previously generated BridgeV2 rollout illustrates the shift from $z _ { 3 : 8 }$ to $z _ { 4 : 9 } .$ Each search branch updates its own history using shared AR parameters. (c) The model predicts a new state, shifts the state–action window, and rebuilds its prefix and KV cache for the next frame. Both anchors and earlier video outputs remain stored. Cells represent token blocks; $z _ { j }$ abbreviates $z _ { j } ^ { \mathrm { d y n } }$ and $a _ { j }$ denotes $\hat { a } _ { j }$  
![](images/e9261f8386d15b3bbd28317ecea79bb0005b1889b40f66c3a8a25ad15f6566e9.jpg)  
Fig. 5. Candidate construction with different search strategies. (a) Likelihood-guided MCTS selects and expands frame prefixes, rolls out the remaining horizon, and backs up returns along the selected ancestor path. (b) Ordinary beam search ranks token prefixes in a single pool by cumulative model log probability. (c) GDBS maintains separate pools and penalizes tokens selected by all earlier groups at the same position, retaining one complete candidate per group. Colors identify groups; trees are schematic. The depicted search budgets differ.

evidence. One forward pass reads four answer-token logits and normalizes them with softmax. The evaluator receives neither ground-truth future frames nor numerical actions.

To mitigate presentation-order bias, we evaluate both orders:

$$
\begin{array} { r } { m _ { i j } = q _ { i j } ^ { \succ } - q _ { i j } ^ { \prec } , \qquad d _ { i j } = \frac { 1 } { 2 } \big ( m _ { i j } - m _ { j i } \big ) . } \end{array}\tag{10}
$$

Thus, $d _ { i j } = - d _ { j i } ,$ , with positive values favoring candidate i. Tie and uncertainty probabilities remain in the normalization. The semantic reward is

$$
r _ { J } ^ { ( i ) } = \frac { 1 } { G - 1 } \sum _ { j \neq i } d _ { i j } .\tag{11}
$$

Four candidates require six unordered pairs and twelve ordered comparisons.

Visual consistency feedback: For spatially and temporally aligned predictions and recorded futures, pixel and perceptual errors are

$$
\begin{array} { r l } & { e _ { \mathrm { p i x } } ^ { ( i ) } = \displaystyle \frac { 1 } { T } \sum _ { t = 2 } ^ { T + 1 } \frac { \| \hat { x } _ { t } ^ { ( i ) } - x _ { t } \| _ { 1 } } { 3 H _ { \mathrm { i m g } } W _ { \mathrm { i m g } } } , } \\ & { e _ { \mathrm { p e r c } } ^ { ( i ) } = \displaystyle \frac { 1 } { T } \sum _ { t = 2 } ^ { T + 1 } D _ { \mathrm { L P I P S } } ( \hat { x } _ { t } ^ { ( i ) } , x _ { t } ) . } \end{array}\tag{12}
$$

RGB values lie in [0, 1]; the frozen VGG-based LPIPS network [20] receives inputs mapped to [−1, 1]. We use equal

error coefficients:

$$
r _ { V } ^ { ( i ) } = - \big ( e _ { \mathrm { p i x } } ^ { ( i ) } + e _ { \mathrm { p e r c } } ^ { ( i ) } \big ) .\tag{13}
$$

During post-training, ground-truth future frames enter only this feedback branch, not candidate-generation histories.

Reward fusion: To balance the feedback scales, each branch is standardized within the candidate set:

$$
\widetilde { r } _ { k } ^ { ( i ) } = \frac { r _ { k } ^ { ( i ) } - \mu _ { k } } { \operatorname* { m a x } ( \sigma _ { k } , \epsilon _ { \mathrm { n u m } } ) } , \qquad k \in \{ J , V \} ,\tag{14}
$$

$\mathrm { H e r e } , \mu _ { k }$ and $\sigma _ { k }$ are the candidate mean and sample standard deviation, with $\epsilon _ { \mathrm { n u m } } = 1 0 ^ { - 6 }$ . The fused reward is

$$
r ^ { ( i ) } = w _ { J } \widetilde { r } _ { J } ^ { ( i ) } + w _ { V } \widetilde { r } _ { V } ^ { ( i ) } .\tag{15}
$$

We use $( w _ { J } , w _ { V } ) = ( 0 . 2 5 , 0 . 7 5 )$ for combined feedback. All rewards and normalization statistics are detached from the generator’s gradient graph.

## 3.4 Search-Guided Policy Optimization

Group-relative advantages [21] adjust candidate probabilities under their recorded generation histories, with regularization toward the frozen supervised policy.

Group-relative advantages: Writing $r _ { i } = r ^ { ( i ) }$ , let $\mu _ { r }$ and $\sigma _ { r }$ be the mean and sample standard deviation of fused rewards. We define

$$
\widehat { A } _ { i } = \left\{ \begin{array} { l l } { \displaystyle r _ { i } - \mu _ { r } } \\ { \displaystyle \sigma _ { r } + \epsilon _ { \mathrm { n u m } } } \\ { \displaystyle 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{16}
$$

All tokens in a candidate share its detached advantage. Branch standardization balances the feedback sources, whereas advantage normalization scales the resulting fused signal.

History-consistent evaluation: For token $y _ { i , u } ,$ let $\mathcal { H } _ { i , u }$ contain its actual retained prefix and preceding within-frame tokens. The policy ratio is

$$
\rho _ { i , u } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) } .\tag{17}
$$

Candidates and histories remain fixed during an update, and the denominator is detached. Both policies normalize over valid dynamics tokens without the search penalty.

Regularized policy objective: The clipped surrogate [48] is

$$
\begin{array} { r } { s _ { i , u } = \operatorname* { m i n } \bigl \{ \rho _ { i , u } \widehat { A } _ { i } , \qquad } \\ { \qquad \mathrm { c l i p } ( \rho _ { i , u } , 1 - \eta , 1 + \eta ) \widehat { A } _ { i } \bigr \} . } \end{array}\tag{18}
$$

For $\widehat { A } _ { i } < 0 ,$ dual clipping gives $\widetilde { s } _ { i , u } = \operatorname* { m a x } ( s _ { i , u } , c _ { \mathrm { d u a l } } \widehat { A } _ { i } ) ;$ otherwise $\widetilde { s } _ { i , u } = s _ { i , u }$ . We set $\eta = 0 . 2$ and $c _ { \mathrm { d u a l } } = 3$ . With the frozen supervised policy $\pi _ { \mathrm { r e f } } ,$ , reference regularization uses

$$
\begin{array} { r l } & { \Delta _ { i , u } = \log \frac { \pi _ { \mathrm { r e f } } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) } { \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) } , } \\ & { k _ { i , u } = \operatorname* { m i n } \{ \exp ( \Delta _ { i , u } ) - \Delta _ { i , u } - 1 , \kappa \} . } \end{array}\tag{19}
$$

The objective is

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = \frac { 1 } { G L } \sum _ { i = 1 } ^ { G } \sum _ { u = 1 } ^ { L } \left[ - \widetilde { s } _ { i , u } + \lambda _ { \mathrm { K L } } k _ { i , u } \right] ,\tag{20}
$$

where $\lambda _ { \mathrm { K L } } ~ = ~ 1 0 ^ { - 3 }$ and $\kappa ~ = ~ 1 0$ . This capped, selectedtoken penalty is KL-style regularization, not an unbiased full-distribution KL estimate. Only the backbone and text cross-attention are updated.

Actual update and evaluation: Each candidate batch receives one optimizer step, with gradients accumulated across frame blocks. At $\theta \ : = \ : \theta _ { \mathrm { o l d } }$ , ratios equal one and clipping is inactive. The reward component then has the same gradient as

$$
\mathcal { L } _ { \mathrm { w e i g h t e d } } = - \frac { 1 } { G L } \sum _ { i = 1 } ^ { G } \sum _ { u = 1 } ^ { L } \widehat { A } _ { i } \log \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) .\tag{21}
$$

Positive and negative advantages respectively favor increasing and decreasing candidate-token probabilities; shared parameters and reference regularization jointly determine the update. Candidates are regenerated after each step, and an all-tied global batch is skipped. This is an empirical surrogate over search-selected candidates without importance correction for their selection. Training uses GDBS; evaluation uses frame-wise ordinary beam search of width four, retaining one completed frame before continuing, without an evaluator or ground-truth futures.

## 3.5 Candidate Differences and Comparative Learning

Consider $G \geq 2$ fixed candidates at $\theta _ { \mathrm { b } } ~ = ~ \theta _ { \mathrm { o l d } } ,$ holding their tokens, histories, text features, rewards, and statistics constant. Define

$$
\begin{array} { c } { \displaystyle \ell _ { i } ( \theta ) = \frac { 1 } { L } \sum _ { u = 1 } ^ { L } \log \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) , } \\ { \displaystyle \mathbf { v } _ { i } = \nabla _ { \theta } \ell _ { i } ( \theta ) \vert _ { \theta = \theta _ { \mathrm { b } } } . } \end{array}\tag{22}
$$

Let $s _ { r } = \sigma _ { r } + \epsilon _ { \mathrm { n u m } } ,$ and assume selected-token probabilities are positive and differentiable.

Lemma 1. At $\theta _ { \mathrm { b } } ,$ where clipping is inactive, the reward-ascent gradient, excluding reference regularization, satisfies

$$
\begin{array} { l } { { \displaystyle { \bf g } = \frac { 1 } { G s _ { r } } \sum _ { i = 1 } ^ { G } ( r _ { i } - \mu _ { r } ) { \bf v } _ { i } } } \\ { { \displaystyle ~ = \frac { 1 } { G ^ { 2 } s _ { r } } \sum _ { i < j } ( r _ { i } - r _ { j } ) ( { \bf v } _ { i } - { \bf v } _ { j } ) } , } \end{array}\tag{23}
$$

provided $\sigma _ { r } > \epsilon _ { \mathrm { n u m } } ;$ otherwise $\mathbf g = \mathbf 0$

Proof sketch. At the snapshot, $\rho _ { i , u } ~ = ~ 1$ with a detached denominator, so Eq. 21 gives $\begin{array} { r } { \mathbf { g } = G ^ { - 1 } \sum _ { i } \widehat { A } _ { i } \mathbf { v } _ { i } } \end{array}$ . Expanding the pairwise sum yields

$$
\sum _ { i < j } ( r _ { i } - r _ { j } ) ( { \bf v } _ { i } - { \bf v } _ { j } ) = G \sum _ { i } ( r _ { i } - \mu _ { r } ) { \bf v } _ { i } .
$$

The complete proof appears in Appendix E.

Corollary 1. The reward-driven gradient vanishes if all candidate rewards coincide or all $\mathbf { v } _ { i }$ coincide. Identical candidates with identical histories are a special case; reference regularization may still produce an update.

Equation 23 shows that feedback differences weight differences in candidate score gradients: search constructs distinct paths, and feedback determines their relative update weights. The identity requires no independent-sampling assumption. Token differences alone do not guarantee a nonzero gradient, since pairwise contributions can cancel. Controlled ablations assess the empirical benefit; sufficient nondegeneracy conditions and a local fixed-candidate analysis are provided in Appendix E.

## 4 EXPERIMENTS

We compare ROBOIMAGINE with existing methods on visual prediction quality and task-level consistency, then assess comparative post-training against continued supervised finetuning from the same initialization. Ablation studies examine feedback composition and candidate-search strategies. We further analyze long-horizon prediction and the effect of history-window size on quality and inference cost.

## 4.1 Experimental Setup

Datasets and prediction protocol. We evaluate on RT-1 [49], BridgeV2 [50], and RoboCasa [51], covering real-world robot manipulation and simulated household tasks. Models are trained separately for each dataset. The main prediction evaluation uses 128 trajectories per dataset, except for the explicitly labeled 32-trajectory BridgeV2 validation result in Table 2, and generates 32 future frames from initial observations and action sequences at 256 × 320 resolution. Generated predictions update the rollout history; future ground-truth images are never used for generation.

Baselines. The models in Tables 1 and 2 span five categories:

① Closed-source video generators: Kling 2.6 [52], Runway Gen-4.5 [53], MiniMax Hailuo 02 [54], and Luma Dream Machine [55].

② Open-source video generators: HunyuanVideo-I2V [56], LTX-Video [57], Stable Video Diffusion XT [58], Mochi-1 [59], I2VGen-XL [60], CogVideoX-I2V [61], OpenSora-I2V [62], and OpenSora-Plan-I2V [63].

③ Action-conditioned and interactive world models: iVideoGPT [24], RoboDreamer [25], IRASim [26], Vid2World [13], and DriftWorld [64].

④ RL-post-trained world models: RLVR-World [10] and RoboAlign-R1 [17], which provide comparisons with alternative reward-alignment methods.

⑤ Task-adapted video foundation models: Wan2.2-TI2V-5B [65], including the LoRA-adapted version in Table 1; Cosmos-Predict2.5-2B [66]; and DreamDojo-2B [67].

Figure 6 additionally includes Ctrl-World [68] and WEAVER [69] as qualitative baselines. External baselines use previously reported or adaptation-based evaluation results, with potentially different training budgets, conditioning interfaces, and decoding procedures. Controlled experiments share the initialization, test trajectories, and evaluation decoder to compare continued supervised fine-tuning, feedback combinations, and candidate-search strategies.

Implementation and post-training. ROBOIMAGINE starts from dataset-specific SFT weights. The controlled RT-1 experiments share a 50,000-step SFT checkpoint and perform 500 additional updates using AdamW with a learning rate of 10<sup>−6</sup> and a global batch size of eight. Each condition generates ten-frame candidate futures. By default, GDBS uses four groups with two beams each and retains one complete candidate per group; the diversity penalty is 0.1. Semantic and visual feedback have weights 0.25 and 0.75, respectively. Only the autoregressive generator and text cross-attention modules are updated; the visual codec, text encoder, and semantic evaluator remain frozen. Default memory retains the fixed anchors and six recent states. Evaluation uses framewise beam search with width four.

Evaluation metrics. We report PSNR, SSIM [70], and LPIPS [20] for visual prediction quality, and additionally use MAE in the training-dynamics analysis. Main prediction metrics are averaged over the 32 predicted frames within each trajectory and then across test trajectories. Task semantics and physical consistency are assessed with the six-dimensional RoboAlign-Judge rubric [17]: instruction following, manipulation success, action–outcome consistency, temporal consistency, contact realism, and physics adherence, summing to a maximum of ten points. Additional multimodal evaluators score paired SFT and post-trained predictions; sample coverage is specified in the corresponding experiments. The independent multimodal evaluations use models distinct from the training reward model; the search comparison additionally reports feedback from the frozen training evaluator. We measure computational efficiency through training GPU-hours, generation latency, and peak GPU memory usage.

## 4.2 Comparison with State-of-the-Art Methods

Task alignment and physical realism. Table 1 evaluates task-level behavior beyond visual similarity. ROBOIMAGINE achieves the highest total score among the compared models, reaching 8.98 ± 0.10 out of ten, compared with 8.52 ± 0.15 for RoboAlign-R1. It ranks first in five dimensions and ties for first in contact realism. In particular, action–outcome consistency increases from 0.72 to 0.82, and physics adherence from 1.58 to 1.77. These results indicate that the gains extend beyond appearance to the consistency of action-conditioned interactions and outcomes.

Visual prediction quality. Table 2 reports visual prediction metrics across the three datasets. On RT-1 and RoboCasa, ROBOIMAGINE achieves PSNR values of 22.57 and 18.39 dB, exceeding the strongest listed baseline by 2.03 and 1.68 dB, respectively. LPIPS decreases by 10.9% and 12.3% relative to the lowest listed baseline value, alongside higher SSIM. On the 32 BridgeV2 validation trajectories, the best checkpoint selected by validation LPIPS reaches 21.53 dB PSNR, 0.7880 SSIM, and 0.1527 LPIPS, compared with 21.39 dB, 0.7865, and 0.1539 for its SFT initialization. Its reported values are the best listed in the BridgeV2 columns, although the differing cohorts preclude a matched-cohort comparison. On the separate 128-trajectory reporting cohort, the same checkpoint gives 21.65 dB PSNR, 0.7926 SSIM, and 0.1467 LPIPS versus 21.74 dB, 0.7930, and 0.1460 for SFT, so its validation gain does not establish a gain on that cohort. External baselines also differ in training and conditioning settings; the shared-initialization comparison below separately examines the effect of post-training.

Qualitative comparison. Figure 6 shows selected, matched trajectories at four prediction steps. In the RT-1 example, ROBOIMAGINE follows the gripper–bag interaction more closely, whereas several baselines leave the bag on the counter or distort the robot and object. In the BridgeV2 spoon-manipulation example, it better preserves the gripper geometry and its spatial relationship to the red spoon. Cosmos leaves the spoon on the work surface, while Ctrl-World and WEAVER show larger deviations in gripper position or shape. The shared step-32 crops make these local differences more visible. The RT-1 example uses the 500- update combined-feedback model; the BridgeV2 example uses the same validation-selected checkpoint as Table 2. These qualitative examples complement the aggregate evaluation.

TABLE 1  
Robot world-model performance on RT-1 from ROBOTWORLDBENCH, evaluated with RoboAlign-Judge $( \boldsymbol { \mathsf { m e a n } } \pm \mathsf { s t d . } )$ . Superscripts 1, 2, and 3 indicate the top three distinct model scores in each column; ties share a rank.
<table><tr><td>Method</td><td colspan="3">Task Alignment</td><td colspan="3">Physical Realism</td><td>Total</td></tr><tr><td></td><td>Instr. ↑</td><td>Manip. ↑</td><td>Act.-Out. ↑</td><td>Temp. ↑</td><td>Contact ↑</td><td>Phys. ↑</td><td>↑</td></tr><tr><td>Real videos</td><td>3.0</td><td>2.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>2.0</td><td>10.0</td></tr><tr><td>Closed video models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling 2.6</td><td> $2 . 4 2 _ { \pm 0 . 0 9 }$ </td><td> $1 . 3 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 4 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 8 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 . 1 8 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $6 . 8 4 _ { \pm 0 . 2 2 }$ </td></tr><tr><td>Runway Gen-4.5</td><td> $2 . 3 4 _ { \pm 0 . 1 0 }$ </td><td> $1 . 2 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 4 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 5 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 7 9 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 . 1 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 . 5 2 { \scriptstyle \pm 0 . 2 4 }$ </td></tr><tr><td>MiniMax Hailuo 02</td><td> $2 . 1 8 _ { \pm 0 . 1 1 }$ </td><td> $1 . 1 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 3 8 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 . 0 1 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $5 . 9 2 { \scriptstyle \pm 0 . 2 8 }$ </td></tr><tr><td>Luma Dream Machine</td><td> $2 . 0 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 . 0 8 _ { \pm 0 . 1 1 }$ </td><td> $0 . 3 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 4 4 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 6 9 _ { \pm 0 . 0 8 }$ </td><td> $0 . 9 2 _ { \pm 0 . 1 2 }$ </td><td> $5 . 5 1 { \scriptstyle \pm 0 . 3 0 }$ </td></tr><tr><td>Open video models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HunyuanVideo-I2V</td><td> $1 . 2 0 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $0 . 4 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $0 . 2 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $\mathbf { 0 . 9 6 _ { \pm 0 . 0 8 } } ^ { 3 }$ </td><td> $\mathbf { 1 . 5 6 { \scriptstyle \pm 0 . 1 5 } } ^ { 3 }$ </td><td> $4 . 9 2 _ { \pm 0 . 6 3 }$ </td></tr><tr><td>LTX-Video</td><td> $2 . 2 8 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 . 3 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 5 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 2 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 2 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 . 6 6 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td>Stable Video Diffusion XT</td><td> $1 . 8 4 _ { \pm 0 . 1 5 }$ </td><td> $0 . 9 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $0 . 4 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 3 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $4 . 2 8 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td>Mochi-1</td><td> $1 . 8 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 9 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 3 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 3 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 6 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 8 2 _ { \pm 0 . 1 1 }$ </td><td> $4 . 8 7 _ { \pm 0 . 2 9 }$ </td></tr><tr><td>I2VGen-XL</td><td> $1 . 5 6 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $0 . 3 2 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 4 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $3 . 5 4 { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td> $\mathrm { C o g V i d e o } \mathrm { X } \mathrm { - } \mathrm { I } 2 \mathrm { V }$ </td><td> $1 . 9 4 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $0 . 4 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $0 . 4 2 _ { \pm 0 . 1 6 }$ </td><td> $0 . 9 2 _ { \pm 0 . 0 7 }$ </td><td> $1 . 2 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $5 . 9 2 { \scriptstyle \pm 0 . 7 5 }$ </td></tr><tr><td>OpenSora-I2V</td><td> $1 . 5 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $0 . 8 2 _ { \pm 0 . 1 1 }$ </td><td> $0 . 2 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 3 1 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 . 2 2 _ { \pm 0 . 3 0 }$ </td></tr><tr><td> $\mathrm { O p e n S o r a - P l a n  – I 2 V }$ </td><td> $1 . 7 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 8 9 _ { \pm 0 . 1 0 }$ </td><td> $0 . 3 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 3 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 6 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 8 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $4 . 7 6 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td colspan="8">Embodied / interactive world-model baselines</td></tr><tr><td>RLVR-World</td><td> $2 . 2 9 _ { \pm 0 . 1 0 }$ </td><td> $1 . 3 1 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 4 4 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 4 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 8 4 _ { \pm 0 . 0 6 }$ </td><td> $1 . 1 9 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 . 5 4 _ { \pm 0 . 2 3 }$ </td></tr><tr><td>iVideoGPT</td><td> $2 . 6 \mathbf { 0 } _ { \pm 0 . 1 1 } { } ^ { 3 }$ </td><td> $\mathbf { 1 . 6 0 _ { \pm 0 . 1 1 } } ^ { 3 }$ </td><td> ${ \bf 0 . 7 0 { \scriptstyle \pm 0 . 1 1 } } ^ { 3 }$ </td><td> $\mathbf { 0 . 7 4 { \scriptstyle \pm 0 . 1 4 } } ^ { 3 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 . 5 4 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 . 7 4 _ { \pm 0 . 6 2 } { } ^ { 3 }$ </td></tr><tr><td>RoboDreamer</td><td> $2 . 0 2 _ { \pm 0 . 1 6 }$ </td><td> $1 . 0 2 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $0 . 3 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 3 2 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 0 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 . 4 4 _ { \pm 0 . 5 6 }$ </td></tr><tr><td>Vid2World</td><td> $2 . 1 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 . 0 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 4 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 2 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 9 8 { \scriptstyle \pm 0 . 0 4 } } ^ { 2 }$ </td><td> $1 . 0 4 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $5 . 9 0 { \scriptstyle \pm 0 . 3 0 }$ </td></tr><tr><td>RoboAlign-R1</td><td> $2 . 7 2 _ { \pm 0 . 0 6 } ^ { \mathrm { ~ ~ } 2 }$ </td><td> $\mathbf { 1 . 7 2 { \scriptstyle \pm 0 . 0 7 } } ^ { 2 }$ </td><td> $\mathbf { 0 . 7 2 { \scriptstyle \pm 0 . 0 5 } ^ { 2 } }$ </td><td> $\mathbf { 0 . 7 8 { \scriptstyle \pm 0 . 0 5 } } ^ { 2 }$ </td><td> $\mathbf { 1 . 0 0 _ { \pm 0 . 0 4 } } ^ { 1 }$ </td><td> $\mathbf { 1 . 5 8 { \scriptstyle \pm 0 . 0 7 } } ^ { 2 }$ </td><td> $8 . 5 2 _ { \pm 0 . 1 5 } ^ { \mathrm { ~ ~ } 2 }$ </td></tr><tr><td colspan="8">Additional training baselines</td></tr><tr><td>Wan2.2-TI2V-5B (LoRA)</td><td> $2 . 4 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 . 0 2 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 4 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 2 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 9 8 { \scriptstyle \pm 0 . 0 5 } ^ { 2 } }$ </td><td> $0 . 9 6 _ { \pm 0 . 1 2 }$ </td><td> $6 . 0 1 { \scriptstyle \pm 0 . 2 4 }$ </td></tr><tr><td>ROBOIMAGINE (ours)</td><td> $2 . 8 2 _ { \pm 0 . 0 2 } ^ { \ 1 }$ </td><td> $\mathbf { 1 . 7 8 { \scriptstyle \pm 0 . 0 5 } ^ { 1 } }$ </td><td> ${ \bf 0 . 8 2 _ { \pm 0 . 0 3 } } ^ { 1 }$ </td><td> $\mathbf { 0 . 7 9 _ { \pm 0 . 0 3 } } ^ { 1 }$ </td><td> ${ \bf 1 . 0 0 { \scriptstyle \pm 0 . 0 1 } } ^ { 1 }$ </td><td> ${ \bf 1 . 7 7 { \scriptstyle \pm 0 . 0 4 } } ^ { 1 }$ </td><td> $8 . 9 8 _ { \pm 0 . 1 0 } ^ { 1 }$ </td></tr></table>

Evaluation with multiple multimodal models. Figure 7(a) compares one fixed ROBOIMAGINE configuration with its SFT initialization on matched trajectories within each evaluator. Mean scores increase from 7.04 to 8.30 for Qwen3-VL-8B-Thinking, from 6.70 to 7.30 for Doubao Seed 2.1 Pro, and from 6.98 to 7.26 for the GPT-6 Astra sessionbased evaluation, corresponding to relative gains of 17.9%, 9.0%, and 4.0%. These comparisons use 97, 10, and 50 paired trajectories, respectively. All three mean differences are positive, although the paired confidence intervals for Doubao and GPT-6 Astra include zero. The verbatim feedback in Figure 7(b) illustrates how evaluators distinguish visible task progress in a drawer-opening example; its case-specific scores are separate from the aggregate means.

updates of either continued SFT or ROBOIMAGINE posttraining. Both use the same 128 RT-1 test trajectories, 32- frame prediction horizon, and evaluation decoder. As shown in Figure 8, ROBOIMAGINE improves PSNR from 21.37 to 22.01 dB within 100 updates and reaches 22.57 dB after 500 updates. Continued SFT remains nearly unchanged, ending at 21.31 dB. Relative to the shared initialization, ROBOIMAGINE also increases SSIM by 0.0175 and reduces LPIPS and MAE by 11.15% and 14.55%, respectively. Under this setting, further optimization of the supervised objective does not reproduce the gains from comparative post-training.

Changes in predicted details. The drawer-closing example in Figure 8 shows predictions at successive training checkpoints. As post-training progresses, ROBOIMAGINE produces a drawer state closer to the ground truth, whereas continued SFT retains a visible state discrepancy. Figure 10 provides additional examples across the three datasets, illustrating improvements in local structure and interaction details relative to the supervised initialization.

## 4.3 Effectiveness of Comparative Post-Training

Comparison with continued supervised fine-tuning. To distinguish the effect of the post-training approach from that of additional updates, we initialize both methods from the same 50,000-step SFT checkpoint and perform 500

Computational cost. Figure 9 reports 1.06 GPU-hours for 500 continued-SFT updates and 28.15 GPU-hours for ROBOIMAGINE post-training. These measurements cover the training loop and exclude initialization, preprocessing, and final evaluation. Post-training additionally includes candidate search, video decoding, feedback computation, and policy updates, resulting in a higher per-update cost. The results therefore demonstrate a quality–compute tradeoff: ROBOIMAGINE achieves higher prediction quality at the same update count, but this comparison does not establish an efficiency advantage under an equal compute budget.

t = 8  
t = 8  
Detail  
TABLE 2  
Video prediction over 32 frames. Superscripts rank the reported values, as in Table 1. ROBOIMAGINE’s BridgeV2 entry reports the best checkpoint selected by LPIPS on 32 validation trajectories; other entries use 128 trajectories. Rankings do not imply matched-cohort comparisons.
<table><tr><td>Method</td><td></td><td>RT-1</td><td></td><td></td><td>BridgeV2</td><td></td><td></td><td>RoboCasa</td><td></td></tr><tr><td></td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td colspan="8">Video Prediction Baselines</td><td></td><td></td></tr><tr><td>iVideoGPT</td><td>15.96</td><td>0.5571</td><td>0.3833</td><td>20.063</td><td>0.7791³</td><td>0.1899</td><td>16.673</td><td>0.71873</td><td>0.2366</td></tr><tr><td colspan="8">Action-Conditioned World Models</td><td></td></tr><tr><td>IRASim</td><td>16.83</td><td>0.5936</td><td>0.3474</td><td>17.39</td><td>0.7094</td><td>0.2524</td><td>13.55</td><td>0.5834</td><td>0.3706</td></tr><tr><td>Vid2World</td><td>19.03</td><td>0.7210</td><td>0.1960</td><td>18.92</td><td>0.7110</td><td>0.1880</td><td>15.40</td><td>0.7000</td><td>0.22873</td></tr><tr><td>DriftWorld</td><td>20.54²</td><td>0.74103</td><td>0.2595</td><td>20.32²</td><td>0.7429</td><td>0.2379</td><td>16.50</td><td>0.7200²</td><td>0.2891</td></tr><tr><td colspan="8">RL-Based World Models</td><td></td><td></td></tr><tr><td>RLVR-World</td><td>19.47</td><td>0.7360</td><td>0.18203</td><td>19.06</td><td>0.7210</td><td>0.17803</td><td>15.80</td><td>0.7100</td><td>0.2017²</td></tr><tr><td>RoboAlign-R1</td><td>19.723</td><td>0.7450²</td><td>0.1730²</td><td>20.00</td><td>0.7310</td><td>0.1750²</td><td>16.71²</td><td>0.7019</td><td>0.2319</td></tr><tr><td colspan="8">Adapted Video Foundation Models</td><td></td><td></td></tr><tr><td>Wan2.2-TI2V-5B</td><td>18.42</td><td>0.7058</td><td>0.2478</td><td>16.35</td><td>0.6889</td><td>0.2568</td><td>14.37</td><td>0.6906</td><td>0.2711</td></tr><tr><td>Cosmos-Predict2.5-2B</td><td>17.77</td><td>0.6765</td><td>0.2889</td><td>18.98</td><td>0.77992</td><td>0.1870</td><td>14.30</td><td>0.6802</td><td>0.2916</td></tr><tr><td>DreamDojo-2B</td><td>18.47</td><td>0.7152</td><td>0.2660</td><td>16.95</td><td>0.7461</td><td>0.2563</td><td>13.92</td><td>0.6349</td><td>0.3278</td></tr><tr><td>ROBOIMAGINE (ours)</td><td>22.571</td><td>0.80351</td><td>0.1541¹</td><td>21.531</td><td>0.78801</td><td>0.15271</td><td>18.391</td><td>0.77271</td><td>0.17681</td></tr></table>

(a) RT-1 · Bag  
Detail  
![](images/40094fe1e526e118cb4ee31b471b925b8faba7701186bbc1399f4492db33aa8b.jpg)  
(b) BridgeV2 · Spoon  
Fig. 6. Qualitative comparison on RT-1 and BridgeV2. Selected bag- and spoon-manipulation examples at steps 8, 16, 24, and 32. Gold boxes mark the shared step-32 regions enlarged on the right.

## 4.4 Ablation Studies

We first vary the semantic and visual feedback components, then compare candidate-search strategies with combined feedback fixed. The post-trained variants in Table 3 share the same SFT initialization, 500 updates, and evaluation decoder, while varying either the feedback components or the candidate-generation strategy. The SFT row reports the initialization without post-training.

Complementary feedback. Table 3(A) shows that either semantic or visual feedback improves over SFT, with different strengths. Visual-only feedback achieves the highest PSNR and SSIM and the lowest LPIPS, indicating the effectiveness of directly constraining visual errors. Combined feedback retains strong visual quality while increasing the mean Qwen total score from 7.04 for SFT to 8.30, exceeding semantic-only

(a) Mean evaluation scores  
![](images/2a551fc85d4c21e183632a2bd66fe8d80fde2bac8627071b0a73f13ecb197b7a.jpg)

## (b) Actual evaluator excerpts

Illustrative case · “Open the bottom drawer.”

<table><tr><td rowspan=1 colspan=1>Evaluator</td><td rowspan=1 colspan=2>SFT</td><td rowspan=1 colspan=1>RoboImagine</td></tr><tr><td rowspan=1 colspan=1>女</td><td rowspan=1 colspan=2>6/10</td><td rowspan=2 colspan=1>10/10“By Frame +32, thebottom drawer is visiblyopen.&quot;</td></tr><tr><td rowspan=2 colspan=1>Qwen3-VL8B-Thinking0</td><td rowspan=1 colspan=2>&quot;the drawer remainsclosed throughout theentire sequence&quot;</td></tr><tr><td rowspan=1 colspan=2>7/10</td><td rowspan=2 colspan=1>10/10“the bottom drawer isvisibly extended with itsinterior exposed.&quot;</td></tr><tr><td rowspan=2 colspan=1>DoubaoSeed 2.1 Pro</td><td rowspan=1 colspan=2>&quot;remains closed andshows no outwarddisplacement&quot;</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2>7/10</td><td rowspan=3 colspan=1>10/10“The last frame shows thebottom drawer visiblyopen&quot;</td></tr><tr><td rowspan=2 colspan=1>GPT-6 AstraSession review</td><td rowspan=2 colspan=2>visibly open farther by thelast frames.&quot;</td></tr><tr><td rowspan=1 colspan=1>he</td><td rowspan=1 colspan=1></td></tr></table>

Fig. 7. Multimodal evaluation and illustrative feedback. (a) Mean scores with 95% bootstrap confidence intervals on 97, 10, and 50 paired trajectories, respectively. (b) Verbatim evaluator excerpts and case-specific scores for one RT-1 trajectory (case 024). GPT-6 Astra denotes a configured-model session review, not a version-pinned API evaluation.

(a) PSNR ↑  
![](images/5c4926bb30da900d4ff0d75caae57736869fb907ee759cead370a4f0693a544b.jpg)

(b) SSIM ↑  
![](images/642ce3bf87b09159e210d71f95da72fd0f20ca14f26a2654fc27a887f2296cef.jpg)  
(e) Prediction frame +32

(c) LPIPS ↓  
![](images/210961e37a43ff354fa5da0ee5ab0fb0fee9ab29d029624e3307ff382d7d65ea.jpg)

(d) MAE ↓  
![](images/9c143f9ac6d200651a89fef0c9664e1ca4bbd51be88f08491b9e5809b2de0d46.jpg)

![](images/14fdaf9bfba30c841576ee13542f305085e837731760546628511e30457469fe.jpg)  
Fig. 8. Post-training dynamics from a shared SFT initialization. (a–d) Four methods use the same 128 RT-1 trajectories, 32-frame horizon, and evaluation decoder; search variants use combined feedback. Markers denote evaluated checkpoints, and red gains compare GDBS with the shared 50,000-step SFT initialization. (e) The same drawer-closing case at prediction frame 32 compares all four methods after 100, 250, and 500 updates. Ground truth and initial SFT are shared references; gold boxes mark identical regions enlarged across methods and checkpoints.

![](images/89c297aa5abfff22db07b33ae76ee6dfdee5e629c77529b854e78ed77db7c4f5.jpg)

![](images/6f79e5a7805c214e8db3a59ffd0f93d083ddfaf933dfc03cc86c612fd2a80d30.jpg)  
Fig. 9. Training cost and prediction quality. Both methods perform 500 updates from the same SFT checkpoint. GPU-hours measure the training stage; PSNR changes are relative to the shared initialization. Update counts are matched, while computational costs differ.  
TABLE 3

Ablations of reward composition and candidate search on RT-1. Superscripts rank the top three scores within each block. Visual metrics use 128 trajectories; Qwen scores use the same 97 trajectories in both blocks. Hybrid in A and GDBS in B denote the same configuration.

<table><tr><td>Setting</td><td>PSNR↑ SSIM↑ LPIPS↓ Qwen↑</td><td></td><td></td><td></td></tr><tr><td colspan="5">A. Reward composition</td></tr><tr><td>SFT</td><td>21.37</td><td></td><td>0.7860 0.1734</td><td>7.04</td></tr><tr><td>Jev only</td><td> $2 \mathbf { 1 . 9 5 } ^ { 3 }$ </td><td>0.79453</td><td> $\mathbf { 0 . 1 6 3 7 } ^ { 3 }$ </td><td> $\mathbf { 7 . 6 9 } ^ { 3 }$ </td></tr><tr><td>Visual only Hybrid</td><td> $2 2 . 6 7 ^ { 1 }$   $2 2 . 5 7 ^ { 2 }$ </td><td>0.8062¹  $\mathbf { 0 . 8 0 3 5 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 5 1 0 } ^ { 1 }$   $\mathbf { 0 . 1 5 4 1 } ^ { 2 }$ </td><td> $7 . 9 7 ^ { 2 }$  8.301</td></tr><tr><td colspan="5">B. Candidate search</td></tr><tr><td>Random sampling</td><td> $2 1 . 4 7 ^ { 3 }$ </td><td> $\mathbf { 0 . 7 8 6 9 } ^ { 3 }$ </td><td> $\mathbf { 0 . 1 7 2 9 } ^ { 3 }$ </td><td> $\mathbf { 7 . 1 0 } ^ { 3 }$ </td></tr><tr><td>Ordinary beam</td><td> $2 2 . 0 7 ^ { 2 }$ </td><td> $\mathbf { 0 . 7 9 6 2 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 6 2 7 } ^ { 2 }$ </td><td> $7 . 5 8 ^ { 2 }$ </td></tr><tr><td>GDBS</td><td> $2 2 . 5 7 ^ { 1 }$ </td><td> $\mathbf { 0 . 8 0 3 5 } ^ { 1 }$ </td><td> $\mathbf { 0 . 1 5 4 1 } ^ { 1 }$ </td><td> $8 . 3 0 ^ { 1 }$ </td></tr></table>

TABLE 4

Search comparison after 500 post-training updates on all 128 RT-1 trajectories (32-frame prediction). Cost: training GPU-seconds per trajectory presentation. Superscripts rank visual metrics.
<table><tr><td rowspan=1 colspan=3>Method              PSNR ↑</td><td rowspan=1 colspan=1>SSIM↑</td><td rowspan=1 colspan=3>GPU-s/LPIPS↓  traj.↓</td></tr><tr><td rowspan=1 colspan=1>SFT initialization</td><td rowspan=1 colspan=2>21.37</td><td rowspan=1 colspan=1>0.7860</td><td rowspan=1 colspan=1>0.1734</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Random sampling</td><td rowspan=1 colspan=2>21.47</td><td rowspan=1 colspan=1>0.7869</td><td rowspan=1 colspan=1>0.1729</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>23.78</td></tr><tr><td rowspan=1 colspan=1>MCTS</td><td rowspan=1 colspan=2> $\mathbf { 2 1 . 4 9 } ^ { 3 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 8 7 1 } ^ { 3 }$ </td><td rowspan=1 colspan=2>0.17223</td><td rowspan=1 colspan=1>84.29</td></tr><tr><td rowspan=1 colspan=1>Ordinary beam</td><td rowspan=1 colspan=2> $\scriptstyle 2 2 . 0 7 ^ { 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 9 6 2 } ^ { 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 1 6 2 7 } ^ { 2 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>24.81</td></tr><tr><td rowspan=1 colspan=1>GDBS</td><td rowspan=1 colspan=1> $2 2 . 5 7 ^ { 1 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 0 3 5 ^ { 1 } }$ </td><td rowspan=1 colspan=2> $\mathbf { 0 . 1 5 4 1 } ^ { 1 }$ </td><td rowspan=1 colspan=1>25.34</td></tr></table>

feedback (7.69) and visual-only feedback (7.97). Its benefit therefore lies in balancing visual fidelity and task semantics, rather than maximizing every visual metric.

Candidate search. With combined feedback fixed, Table 3(B) compares random sampling, ordinary beam, and GDBS. Relative to ordinary beam, GDBS increases PSNR from 22.07 to 22.57 dB and reduces LPIPS from 0.1627 to 0.1541. On the same 97 trajectories used in both blocks, the mean Qwen score increases from 7.58 to 8.30, compared with 7.10 for random sampling. Hybrid in block A and GDBS in block B are the same configuration and share the same scores. Because evaluation uses the same decoder, these differences reflect the effect of candidate construction during post-training. Qwen retains the first valid response; only invalid outputs receive retries. Three truncated sampling responses require an additional attempt with the output limit increased from $^ { 4 , 0 9 6 }$ to 8,192 tokens. Restricting all methods to the 94 trajectories valid under the original budget preserves the ranking.

Comparison with MCTS. Table 4 compares candidatesearch strategies after 500 post-training updates from the same initialization, evaluated on the same 128 RT-1 trajectories with identical decoding. GDBS achieves the best mean PSNR, SSIM, and LPIPS. Relative to ordinary beam, it improves PSNR by 0.50 dB and SSIM by 0.0073, while reducing LPIPS by 5.26%. The likelihood-guided MCTS variant yields smaller gains over SFT; its 0.12 dB PSNR difference has a paired trajectory-bootstrap 95% confidence interval of $[ - 0 . 0 3 , 0 . 2 7 ]$ dB. Figure 11 further compares the final predictions using the frozen training evaluator, separately from the independent Qwen scores in Table 3. GDBS obtains the highest semantic (0.2815) and visual (−0.1964) feedback. These scores characterize final predictions, not candidate diversity within a search run.

Search cost. Normalizing recorded training GPU-seconds by the number of training-trajectory presentations gives 24.81, 25.34, and 84.29 GPU-s for ordinary beam, GDBS, and MCTS, respectively. Thus, GDBS adds approximately 2.1% to the recorded normalized cost of ordinary beam. These historical allocation costs account for GPU count and batch size but do not constitute an equal-compute benchmark: MCTS uses 16 likelihood-guided UCT simulations, whereas both beam variants maintain eight active paths. Physical parallelism, timing coverage, and brief concurrent evaluation also differ.

Search-process analysis. Figure 12 shows a recorded GDBS search for moving a chip bag near a can. Four groups maintain separate beams and each return one complete prediction. Thick paths trace the returned candidates through frame boundaries, with keyframes showing their predicted content. The two feedback sources rank candidates differently: C1 receives the highest semantic score, whereas C3 receives the highest visual score and the highest reward after standardized fusion. This case illustrates how candidate differences interact with complementary feedback: search provides alternative prediction paths, while semantic and visual assessments determine their relative weights for policy optimization.

## 4.5 Long-Horizon Prediction and Memory Efficiency

We examine whether bounded history supports longer predictions and how its size affects quality and inference cost. All comparisons fix the model weights and evaluation decoder; no additional training is performed.

Long-horizon prediction. Figure 13 compares currentframe-only conditioning with full memory, namely the fixed initial context and dynamics anchor plus the latest six predicted states and paired actions. Both horizons use the same 11 held-out trajectories, averaging the first 64 or all 128 predicted frames. Over 64 predicted frames, full memory improves PSNR by 2.18 dB and reduces LPIPS by 17.09%. Over 128 frames, it retains gains of 1.09 dB and 11.04%, respectively. Historical context therefore improves longer rollouts, although prediction errors still accumulate. This comparison evaluates the complete memory configuration and does not isolate the contribution of the fixed dynamic anchor.

![](images/8440fc20772cc42965ef21b3741d426501b4d47778a7924da0b41c79a59f4d85.jpg)  
Fig. 10. Selected qualitative comparisons of ROBOIMAGINE and its supervised initialization. Rows show ground truth, supervised fine-tuning, and ROBOIMAGINE after 500 post-training steps under the same inputs. The RT-1 case uses visual-only feedback; BridgeV2 and RoboCasa use combined feedback. Frames 8 and 32 show motion over time; gold boxes mark identical regions enlarged from frame 32.

![](images/ac39f4166f282eb3d69400ad2a43677371c0a59a01ffeda5752fe419d4eb6724.jpg)  
Fig. 11. Prediction differences and feedback. All 128 RT-1 trajectories in Table 4. Cells show mean between-model dynamics-token distances. Semantic feedback uses bidirectional comparisons by the frozen training Jev; visual feedback $\mathsf { i } \mathsf { s } - ( \mathrm { M A E } + \mathrm { L P I P S } )$ . Both feedback columns are higher-is-better; larger token distances do not imply better quality.

Evaluation protocol. To isolate the effect of historywindow size in Table 5 and Figure 14, we fix the model weights and decoder and generate 16, 32, 48, 64, and 96 frames on the same 30 long-sequence-eligible RT-1 test trajectories selected with a fixed seed. All configurations retain the fixed initial context and dynamic anchor, varying only the rolling window $W \in \{ 1 , \dot { 3 } , 6 , 1 2 \}$ ; thus, $W = 1$ differs from the current-frame-only baseline in Figure 13. Quality is averaged over predicted frames and then trajectories, while three warmed-up timing repeats are averaged within each trajectory. This cohort differs from the 128 RT-1 trajectories in the main comparison, so their absolute metrics are not directly comparable.

Window size and prediction quality. Table 5 and Figure 14 show that moderate windows provide favorable prediction quality. $W = 3$ yields the highest mean PSNR and lowest LPIPS at 16 frames, whereas $W ~ = ~ 6$ yields better corresponding means at 32–96 frames. Increasing the window to 12 does not provide sustained benefits: at 96 frames, PSNR drops from 20.31 dB with $W = 6$ to 18.96 dB, and LPIPS rises from 0.1955 to 0.2251. Effective use of history therefore does not improve monotonically with window size. Meanwhile, the differences between $\dot { W } = 3$ and $W = 6$ are small, with paired trajectory-bootstrap 95% confidence intervals containing zero for all three primary quality metrics at every length. These intervals quantify trajectory variability at a single checkpoint.

![](images/ca9d35a0cee394d96510c14fb143ea43e9257d24794ab18e47e9953e92b05a3d.jpg)  
Fig. 12. Recorded GDBS search: moving a chip bag near a can. Thick paths trace returned candidates; thin paths show other retained beams. Keyframes accompany post-search feedback; bold scores mark column maxima. This pre-update training case has the largest semantic-score spread increase over ordinary beam among eight conditions.

![](images/75a30224a0db4d3e6c9ecd9dccf0dd600093c9bad626c2872942ee20c7ba204b.jpg)  
Fig. 13. Long-horizon memory comparison on RT-1.

TABLE 5  
History-window quality and inference cost with fixed weights and anchors. Time denotes whole-rollout latency; memory denotes peak allocated GPU memory. Three timing repeats are averaged within each trajectory. Superscripts rank the top three values within each length at full precision. This experiment uses the same 30 long-sequence-eligible RT-1 test trajectories, whereas Table 2 uses 128 RT-1 trajectories. Thus, the 32-frame means are not directly comparable across tables (e.g., W = 6: 22.20 versus 22.57 dB).
<table><tr><td>Frames</td><td>Window</td><td>PSNR ↑</td><td>SSIM↑</td><td> $\mathrm { L P I P S } \downarrow$ </td><td>MAE↓</td><td>Time (s) ↓</td><td>Memory (GiB) ↓</td></tr><tr><td>16</td><td> $W = 1$ </td><td>20.99</td><td>0.7869</td><td>0.1730</td><td> $0 . 0 4 6 3$ </td><td> $\mathbf { 1 2 . 4 6 ^ { 1 } }$ </td><td> $2 . 2 0 6 ^ { 1 }$ </td></tr><tr><td></td><td> $W = 3$ </td><td> $2 3 . 3 3 ^ { 1 }$ </td><td> $\mathbf { 0 . 8 1 2 8 ^ { 1 } }$ </td><td> $\mathbf { 0 . 1 4 2 4 } ^ { 1 }$ </td><td> $\mathbf { 0 . 0 3 5 9 } ^ { 2 }$ </td><td> $\mathbf { 1 2 . 6 7 ^ { 2 } }$ </td><td> $2 . 2 0 6 ^ { 2 }$ </td></tr><tr><td></td><td> $W = 6$ </td><td> $2 3 . 2 7 ^ { 2 }$ </td><td> $\mathbf { 0 . 8 1 1 5 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 4 3 0 } ^ { 2 }$ </td><td> $\mathbf { 0 . 0 3 5 7 } ^ { 1 }$ </td><td> $\mathbf { 1 } 3 . 4 2 ^ { 3 }$ </td><td> $2 . 3 2 4 ^ { 3 }$ </td></tr><tr><td></td><td> $W = 1 2$ </td><td> $2 2 . 9 4 ^ { 3 }$ </td><td> $\mathbf { 0 . 8 0 7 6 } ^ { 3 }$ </td><td> $\mathbf { 0 . 1 4 7 1 } ^ { 3 }$ </td><td> $\mathbf { 0 . 0 3 7 4 } ^ { 3 }$ </td><td>14.39</td><td>2.586</td></tr><tr><td>32</td><td> $W = 1$ </td><td>19.77</td><td>0.7671</td><td>0.1941</td><td>0.0546</td><td> $2 4 . 9 4 ^ { 1 }$ </td><td>2.2301</td></tr><tr><td></td><td> $W = 3$ </td><td> $\pmb { 2 2 . 0 8 } ^ { 2 }$ </td><td> $\mathbf { 0 . 7 9 6 3 ^ { 2 } }$ </td><td> $\mathbf { 0 . 1 5 9 9 } ^ { 2 }$ </td><td> $\mathbf { 0 . 0 4 2 9 } ^ { 2 }$ </td><td> $2 5 . 3 6 ^ { 2 }$ </td><td> $\phantom { - } 2 . 2 3 0 ^ { 2 }$ </td></tr><tr><td></td><td> $W = 6$ </td><td> $2 2 . 2 0 ^ { 1 }$ </td><td> $\mathbf { 0 . 7 9 6 3 } ^ { 1 }$ </td><td> $\mathbf { 0 . 1 5 8 6 } ^ { 1 }$ </td><td> $\mathbf { 0 . 0 4 1 6 } ^ { 1 }$ </td><td> $2 7 . 1 9 ^ { 3 }$ </td><td> $2 . 3 2 5 ^ { 3 }$ </td></tr><tr><td></td><td> $W = 1 2$ </td><td> ${ \bf 2 1 . 1 9 ^ { 3 } }$ </td><td> $\mathbf { 0 . 7 7 9 1 } ^ { 3 }$ </td><td> $\mathbf { 0 . 1 7 5 2 } ^ { 3 }$ </td><td> $\mathbf { 0 . 0 4 8 7 } ^ { 3 }$ </td><td> $3 0 . 5 3$ </td><td>2.588</td></tr><tr><td>48</td><td> $W = 1$ </td><td>19.19</td><td>0.7541</td><td> $0 . 2 0 6 4$ </td><td> $0 . 0 5 9 2$ </td><td> $3 7 . 2 7 ^ { 1 }$ </td><td> $\pmb { 2 . 2 5 3 } ^ { 1 }$ </td></tr><tr><td></td><td> $W = 3$ </td><td> $\mathbf { 2 1 . 3 1 } ^ { 2 }$ </td><td> $\mathbf { 0 . 7 8 2 1 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 7 4 0 } ^ { 2 }$ </td><td> $\mathbf { 0 . 0 4 7 8 } ^ { 2 }$ </td><td> $\mathbf { 3 8 . 0 1 ^ { 2 } }$ </td><td> $\pmb { 2 . 2 5 3 } ^ { 2 }$ </td></tr><tr><td></td><td> $W = 6$ </td><td> $2 \mathbf { 1 . 4 7 } ^ { 1 }$ </td><td> $\mathbf { 0 . 7 8 3 0 } ^ { 1 }$ </td><td> $\mathbf { 0 . 1 7 1 5 } ^ { 1 }$ </td><td> $\mathbf { 0 . 0 4 5 8 } ^ { 1 }$ </td><td> $\mathbf { 4 0 . 9 7 } ^ { 3 }$ </td><td> $2 . 3 2 6 ^ { 3 }$ </td></tr><tr><td></td><td> $W = 1 2$ </td><td> $2 0 . 2 6 ^ { 3 }$ </td><td> $\mathbf { 0 . 7 6 0 7 } ^ { 3 }$ </td><td> $\mathbf { 0 . 1 9 3 6 } ^ { 3 }$ </td><td> $\mathbf { 0 . 0 5 5 } 2 ^ { 3 }$ </td><td> $4 6 . 6 7$ </td><td>2.589</td></tr><tr><td>64</td><td> $W = 1$ </td><td>18.85</td><td>0.7459</td><td>0.2146</td><td>0.0619</td><td> $\pmb { 4 9 . 6 5 } ^ { 1 }$ </td><td> $2 . 2 7 6 ^ { 1 }$ </td></tr><tr><td></td><td> $W = 3$ </td><td> $\scriptstyle 2 0 . 8 7 ^ { 2 }$ </td><td> $\mathbf { 0 . 7 7 4 0 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 8 2 8 } ^ { 2 }$ </td><td> $\mathbf { 0 . 0 5 0 2 } ^ { 2 }$ </td><td> ${ \bar { 5 } } 0 . 6 3 ^ { 2 }$ </td><td> $\phantom { - } 2 . 2 7 6 ^ { 2 }$ </td></tr><tr><td></td><td> $W = 6$ </td><td> $\mathbf { 2 1 . 0 3 ^ { 1 } }$ </td><td> $\mathbf { 0 . 7 7 4 6 } ^ { 1 }$ </td><td> $\mathbf { 0 . 1 8 0 6 } ^ { 1 }$ </td><td> $\mathbf { 0 . 0 4 8 3 } ^ { 1 }$ </td><td> $5 4 . 7 4 ^ { 3 }$ </td><td> $\pmb { 2 . 3 2 7 } ^ { 3 }$ </td></tr><tr><td></td><td> $W = 1 2$ </td><td> $\mathbf { 1 9 . 6 9 } ^ { 3 }$ </td><td> $\mathbf { 0 . 7 4 7 8 } ^ { 3 }$ </td><td> $\mathbf { 0 . 2 0 6 8 } ^ { 3 }$ </td><td> $\mathbf { 0 . 0 5 9 6 } ^ { 3 }$ </td><td> $6 2 . 8 2$ </td><td>2.590</td></tr><tr><td>96</td><td> $W = 1$ </td><td> $1 8 . 5 3$ </td><td> $\mathbf { 0 . 7 3 3 5 } ^ { 3 }$ </td><td> $\mathbf { 0 . 2 2 5 1 } ^ { 3 }$ </td><td> $\mathbf { 0 . 0 6 4 7 } ^ { 3 }$ </td><td> $7 4 . 6 9 ^ { 1 }$ </td><td> $2 . 3 2 2 ^ { 1 }$ </td></tr><tr><td></td><td> $W = 3$ </td><td> $2 0 . 2 4 ^ { 2 }$ </td><td> $\mathbf { 0 . 7 6 0 5 ^ { 1 } }$ </td><td> $\mathbf { 0 . 1 9 6 1 } ^ { 2 }$ </td><td> $\mathbf { 0 . 0 5 4 3 } ^ { 2 }$ </td><td> $7 5 . 9 6 ^ { 2 }$ </td><td> $\pmb { 2 . 3 2 3 ^ { 2 } }$ </td></tr><tr><td></td><td> $W = 6$ </td><td> $2 \mathbf { 0 } . 3 \mathbf { 1 } ^ { 1 }$ </td><td> $\mathbf { 0 . 7 5 9 2 } ^ { 2 }$ </td><td> $\mathbf { 0 . 1 9 5 5 } ^ { 1 }$ </td><td> $\mathbf { 0 . 0 5 3 1 } ^ { 1 }$ </td><td> $8 2 . 3 3 ^ { 3 }$ </td><td> $2 . 3 2 8 ^ { 3 }$ </td></tr><tr><td></td><td> $W = 1 2$ </td><td> $\mathbf { 1 8 . 9 6 } ^ { 3 }$ </td><td>0.7275</td><td>0.2251</td><td>0.0658</td><td>95.14</td><td>2.593</td></tr></table>

(a) PSNR ↑  
![](images/d199f029d610caf31062f1bd92f513495bcf1b48e35dab53dce2ee00843c11e9.jpg)

(b) LPIPS ↓  
![](images/44cdc990742c2040014715f4b46e668d69cc2d5ba607b0a40da8c4c2408d89be.jpg)

(c) Latency ↓  
![](images/bdc0dd40d5d820f0b8a7f87f1bb882e1c3b3d1d3bb33a38be2ae8696a89f8ee4.jpg)  
Seconds / rollout

![](images/46427a6b84b00937f863d72745da2a58998d10e8a34142f270540585efa1a973.jpg)  
Peak allocated GiB  
Fig. 14. History-window sensitivity and inference cost. Cells show mean quality; outlines mark the best mean at each length. Bars show whole-rollout latency and peak allocated GPU memory, with 95% trajectory-bootstrap intervals. Three timing repeats are averaged within each trajectory. All five lengths use the same 30 long-sequence-eligible RT-1 test trajectories, rather than the 128 trajectories in Table 2. Their 32-frame means are therefore not directly comparable (e.g., W = 6: 22.20 versus 22.57 dB).

Inference efficiency. Larger windows also increase generation cost. At 96 frames, $\mathbf { \bar { \boldsymbol { W } } } = 3 , W = 6 ,$ and $W = 1 2$ require 75.96, 82.33, and 95.14 s per rollout, with peak allocated memory of 2.323, 2.328, and 2.593 GiB, respectively. Latency includes initial encoding, autoregressive prediction, decoding, and output transfer to the CPU; model loading, warm-up, scoring, and file writes are excluded. Memory denotes peak PyTorch allocated memory rather than device wide usage. Together with the quality results, $W ~ = ~ 6$ provides favorable mean prediction quality at medium and long horizons, while $W = 3$ reduces generation time by 7.74% at 96 frames with similar quality. Bounded history thus provides a configurable trade-off between prediction quality and inference cost.

## 5 CONCLUSION & LIMITATIONS

We present ROBOIMAGINE, a framework for post-training autoregressive robotic world models through candidatefuture comparisons with complementary semantic and visual feedback. GDBS constructs alternative trajectories under shared conditions, and branch-specific bounded histories preserve consistent conditioning between generation and policy scoring. Evaluations on RT-1, BridgeV2, and Robo-Casa demonstrate strong visual prediction quality and tasklevel consistency. Controlled RT-1 experiments show that comparative post-training yields larger improvements than continued supervised fine-tuning at the same update count, although its additional search and feedback computations incur higher training cost. Ablations support the roles of diverse candidate construction and complementary feedback: visual feedback provides strong fidelity supervision, while combining it with semantic preferences raises the external multimodal evaluation mean with comparable visual quality. Rolling history improves long-horizon prediction, and window-size studies reveal a quality–cost trade-off rather than a monotonic benefit from longer context. Together, these findings support learning robotic world models from comparisons that account for both task progress and visual detail. The current evidence concerns offline video prediction and does not establish improved closed-loop planning or real-robot control. Evaluating these downstream capabilities and reducing the computational cost of comparative posttraining are important directions for future work.

## REFERENCES

[1] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, vol. 2, no. 3, p. 440, 2018.

[2] J. Ni, Y. Guo, Y. Liu, R. Chen, L. Lu, and Z. Wu, “Maskgwm: A generalizable driving world model with video mask reconstruction,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22 381–22 391.

[3] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman et al., “Video generation models as world simulators,” OpenAI Blog, vol. 1, no. 8, p. 1, 2024.

[4] B. Hou, G. Li, J. Jia, T. An, X. Guo, S. Leng, H. Geng, Y. Ze, T. Harada, P. Torr, O. Mees, M. Pollefeys, Z. Liu, J. Wu, P. Abbeel, J. Malik, Y. Du, and J. Yang, “World model for robot learning: A comprehensive survey,” arXiv preprint arXiv:2605.00080, 2026.

[5] W. Tan, L. Zhu, B. Wang, E. Xie, B. Ji, Z. Lin, W. Yang, J. Li, and H. T. Shen, “Towards generalist embodied ai: A survey on world models for vla agents,” Authorea Preprints, 2026.

[6] N. Hansen, H. Su, and X. Wang, “Td-mpc2: Scalable, robust world models for continuous control,” arXiv preprint arXiv:2310.16828, 2023.

[7] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani et al., “Evaluating real-world robot manipulation policies in simulation,” arXiv preprint arXiv:2405.05941, 2024.

[8] Y. Zhang, G. Yang, R. Hou, Q. Chen, Z. Liu, X. Liu, M. Zhang, Y. Hao, Z. Wei, H. Wu, L. Qin, P. Dai, Y. Li, D. Yin, and X. Sun, “Thinking in video: Can video generators really reason about the real world?” arXiv preprint arXiv:2607.17523, 2026. [Online]. Available: https://arxiv.org/abs/2607.17523

[9] D. Li, Y. Fang, Y. Chen, S. Yang, S. Cao, J. Wong, M. Luo, X. Wang, H. Yin, J. E. Gonzalez et al., “Worldmodelbench: Judging video generation models as world models,” arXiv preprint arXiv:2502.20694, 2025.

[10] J. Wu, S. Yin, N. Feng, and M. Long, “Rlvr-world: Training world models with reinforcement learning,” arXiv preprint arXiv:2505.13934, 2025.

[11] L. Barcellona, A. Zadaianchuk, D. Allegro, S. Papa, S. Ghidoni, and E. Gavves, “Dream to manipulate: Compositional world models empowering robot imitation learning with imagination,” arXiv preprint arXiv:2412.14957, 2024.

[12] J.-C. Pang, N. Tang, K. Li, Y. Tang, X.-Q. Cai, Z.-Y. Zhang, G. Niu, M. Sugiyama, and Y. Yu, “Learning viewinvariant world models for visual robotic manipulation,” in The Thirteenth International Conference on Learning Representations, 2025.

[13] S. Huang, J. Wu, Q. Zhou, S. Miao, and M. Long, “Vid2world: Crafting video diffusion models to interactive world models,” arXiv preprint arXiv:2505.14357, 2025.

[14] J. Mao, S. He, H.-N. Wu, Y. You, S. Sun, Z. Wang, Y. Bao, H. Chen, L. Guibas, V. Guizilini et al., “Robot learning from a physical world model,” arXiv preprint arXiv:2511.07416, 2025.

[15] J. Ding, Y. Zhang, Y. Shang, Y. Zhang, Z. Zong, J. Feng, Y. Yuan, H. Su, N. Li, N. Sukiennik et al., “Understanding world or predicting future? a comprehensive survey of world models,” ACM Computing Surveys, vol. 58, no. 3, pp. 1–38, 2025.

[16] H. Wu, S. Qian, W. Wang, F. Xu, F. Zhang, Y. He, Q. Wen, and Y. Liang, “FutureWorlds: Learning robotic world models from alternative futures,” arXiv preprint arXiv:2610.01019, 2026.

[17] H. Wu, Y. Li, Y. Gao, F. Xu, F. Zhang, K. Wang, P. Zhao, Q. Wang, Y. Zhao, W. Wang, Y. Tian, X. Wu, and X. Huang, “RoboAlign-R1: Distilled multimodal reward alignment for robot video world models,” in Advances in Neural Information Processing Systems, 2026. [Online]. Available: https://arxiv.org/abs/2605.03821

[18] A. K. Vijayakumar, M. Cogswell, R. R. Selvaraju, Q. Sun, S. Lee, D. Crandall, and D. Batra, “Diverse beam search: Decoding diverse solutions from neural sequence models,” arXiv preprint arXiv:1610.02424, 2016.

[19] SoMarkAI, “mJev: Jev, with senses,” GitHub repository, 2026, accessed: Oct. 12, 2026. [Online]. Available: https://github.com/SoMarkAI/mJev

[20] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and

O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[21] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[22] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in Proceedings of the 36th International Conference on Machine Learning, vol. 97. PMLR, 2019, pp. 2555–2565.

[23] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, pp. 647–653, 2025.

[24] J. Wu, S. Yin, N. Feng, X. He, D. Li, J. Hao, and M. Long, “ivideogpt: Interactive videogpts are scalable world models,” Advances in Neural Information Processing Systems, vol. 37, pp. 68 082–68 119, 2024.

[25] S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan, “Robodreamer: Learning compositional world models for robot imagination,” arXiv preprint arXiv:2404.12377, 2024.

[26] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong, “IRASim: Learning interactive real-robot action simulators,” arXiv preprint arXiv:2406.14540, 2024.

[27] W. Yan, Y. Zhang, P. Abbeel, and A. Srinivas, “Videogpt: Video generation using vq-vae and transformers,” arXiv preprint arXiv:2104.10157, 2021.

[28] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via next-scale prediction,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[29] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[30] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.

[31] NVIDIA, N. Agarwal, A. Ali, M. Bala, Y. Balaji et al., “Cosmos world foundation model platform for physical AI,” arXiv preprint arXiv:2501.03575, 2025.

[32] M. Mathieu, C. Couprie, and Y. LeCun, “Deep multiscale video prediction beyond mean square error,” arXiv preprint arXiv:1511.05440, 2015.

[33] X. He, D. Jiang, G. Zhang, M. Ku, A. Soni, S. Siu, H. Chen, A. Chandra, Z. Jiang, A. Arulraj et al., “Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 2105–2123.

[34] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 807– 21 818.

[35] T. Xiong, S. Wang, G. Liu, Y. Dong, M. Li, H. Huang, J. Kautz, and Z. Yu, “Phycritic: Multimodal critic models for physical ai,” arXiv preprint arXiv:2602.11124, 2026.

[36] P. F. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[37] K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu, “Aligning text-to-image models using human feedback,” arXiv preprint arXiv:2302.12192, 2023.

[38] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, “Training diffusion models with reinforcement learning,” in International Conference on Learning Representations, 2024.

[39] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik, “Diffusion model alignment using direct preference optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[40] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging LLM-as-a-judge with MT-Bench and chatbot arena,” Advances in Neural Information Processing Systems, 2023. [Online]. Available: https://arxiv.org/abs/2306.05685

[41] B. Chen, D. Martí Monsó, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Nexttoken prediction meets full-sequence diffusion,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[42] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing: Bridging the train-test gap in autoregressive video diffusion,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[43] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Efficient streaming language models with attention sinks,” arXiv preprint arXiv:2309.17453, 2023.

[44] S. Yang, W. Huang, R. Chu, Y. Xiao, Y. Zhao, X. Wang, M. Li, E. Xie, Y. Chen, Y. Lu, S. Han, and Y. Chen, “LongLive: Real-time interactive long video generation,” in International Conference on Learning Representations, 2026.

[45] Q. Hu, S. Yang, W. Huang, S. Han, and Y. Chen, “LongLive-RAG: A general retrieval-augmented framework for long video generation,” arXiv preprint arXiv:2606.02553, 2026.

[46] F. Mentzer, D. Minnen, E. Agustsson, and G. Toderici, “Finite scalar quantization: VQ-VAE made simple,” in International Conference on Learning Representations, 2024.

[47] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-totext transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.

[48] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[49] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gober, K. Hausman, A. Herzog, J. Hsu et al., “Rt-1: Robotics transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2023.

[50] H. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng,

P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du et al., “Bridgedata v2: A dataset for robot learning at scale,” arXiv preprint arXiv:2308.12952, 2023.

[51] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu, “RoboCasa: Largescale simulation of everyday tasks for generalist robots,” in Robotics: Science and Systems, 2024.

[52] Kling AI, “Kling VIDEO 2.6: Native audio model launch,” Official release notes, 2025. [Online]. Available: https://kling.ai/release-note/release-notes/ c605hp1tzd

[53] Runway, “Introducing Runway Gen-4.5,” Official research announcement, 2025. [Online]. Available: https: //runway.com/research/introducing-runway-gen-4.5

[54] MiniMax, “MiniMax Hailuo 02, world-class quality, record-breaking cost efficiency,” Official model announcement, 2025. [Online]. Available: https://www.minimax.io/news/minimax-hailuo-02

[55] Luma AI, “Welcome to the all new Dream Machine,” Official release notes, 2024. [Online]. Available: https://lumalabs.ai/changelog/ welcome-to-the-all-new-dream-machine

[56] Tencent Hunyuan, “HunyuanVideo-I2V: Official models and implementation,” GitHub repository, 2025. [Online]. Available: https://github.com/ Tencent-Hunyuan/HunyuanVideo-I2V

[57] Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson et al., “LTX-Video: Realtime video latent diffusion,” arXiv preprint arXiv:2501.00103, 2024.

[58] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.

[59] Genmo, “Mochi 1: Official models and implementation,” GitHub repository, 2024. [Online]. Available: https: //github.com/genmoai/mochi

[60] S. Zhang, J. Wang, Y. Zhang, K. Zhao, H. Yuan, Z. Qin, X. Wang, D. Zhao, and J. Zhou, “I2VGen-XL: Highquality image-to-video synthesis via cascaded diffusion models,” arXiv preprint arXiv:2311.04145, 2023.

[61] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu et al., “CogVideoX: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.

[62] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-Sora: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.

[63] B. Lin, Y. Ge, X. Cheng, Z. Li, B. Zhu, S. Wang et al., “Open-Sora Plan: Open-source large video generation model,” arXiv preprint arXiv:2412.00131, 2024.

[64] S. Lu, H. Chen, W. Ye, and Y. Du, “DriftWorld: Fast World Modeling through Drifting,” arXiv preprint arXiv:2607.15065, 2026.

[65] Wan Team, “Wan2.2: Official models and implementation,” GitHub repository, 2025, accessed: Oct. 16, 2026. [Online]. Available: https://github.com/Wan-Video/Wan2.2

[66] NVIDIA, “World Simulation with Video Foundation Models for Physical AI,” arXiv preprint arXiv:2511.00062,

2025.

[67] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu et al., “DreamDojo: A generalist robot world model from largescale human videos,” arXiv preprint arXiv:2602.06949, 2026.

[68] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-World: A controllable generative world model for robot manipulation,” in International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=748bHL2BAv

[69] A. K. Jain, Y. Wu, J. Farebrother, G. Swamy, and A. Bajcsy, “WEAVER, better, faster, longer: An effective world model for robotic manipulation,” arXiv preprint arXiv:2606.13672, 2026. [Online]. Available: https://arxiv.org/abs/2606.13672

[70] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

![](images/890756c65b5e6c37fc3950b46bae094adaf29018bdeff5c5b03065d80e90fc0e.jpg)  
Hao Wu is currently pursuing the Ph.D. degree at Tsinghua University, Beijing, China. His research interests include robotic world models, scientific machine learning, and multimodal agents, with a focus on memory mechanisms, reward learning, and video generation. He has held research internships with Tencent Hunyuan, Tencent Jarvis Research Center, and Tencent Lightspeed Studios.

![](images/0f5f7aa1eb6fc7622cde45c05f652b21a65fbb5cc163fb0ae4efd217f6beafba.jpg)

Yuqi Li is currently pursuing the Ph.D. degree in electrical engineering at The City College of New York, City University of New York, and joined the Media Lab in 2025. Li’s research focuses on knowledge distillation, model compression, and spatiotemporal forecasting.

![](images/49701161da07ab9268adca886b1eaaff8e9bf924f4eb1ca9306fda7337764578.jpg)

Yizhou Zhao is a Member of Technical Staff at Odyssey. He previously conducted doctoral research at Carnegie Mellon University, working with Min Xu, Bhiksha Raj, and Laszlo A. Jeni. His research focuses on 3D generation and reconstruction, video generation, and the modeling of geometry and physical dynamics from video.

![](images/ae35bd4d2be79617f5b974c018ab346ecdd0c844a30f49e93dc577cf9cde7308.jpg)

Kun Wang received the Ph.D. degree from the University of Science and Technology of China. He is a Postdoctoral Researcher at Nanyang Technological University, Singapore. His research interests include large language model safety, multi-agent systems, and data mining for spatiotemporal forecasting and Earth science, with an emphasis on safety, interpretability, generalization, and robustness.

![](images/d10a643ad219c3a38de61ea74836cd4ac514a9ccfba936b8892022f726fc2c8b.jpg)

![](images/7f596a271134a5222e10eabdc8bd8e9fcbf6203ea4f6b6b5b265b46fcd41fad5.jpg)

Yingli Tian (Fellow, IEEE) received the B.S. and M.S. degrees from Tianjin University, China, in 1987 and 1990, respectively, and the Ph.D. degree from The Chinese University of Hong Kong in 1996. She is a CUNY Distinguished Professor with The City College of New York and The Graduate Center, City University of New York. Her research spans computer vision and machine learning, with applications in assistive technology, medical image analysis, and remote sensing.

![](images/1ba5118d9b4fe29c0a4eaf02a1d89db60f23db564772ef7cc1f75078fcbc3586.jpg)

![](images/b067e19ea13ee8e412ed4e906c367a059ec72b9eccd727a592bd0ae6538bf183.jpg)

Xiao Luo received the Ph.D. degree from Peking University, China. He is an Assistant Professor in the Department of Statistics at the University of Wisconsin–Madison. Previously, he was a Postdoctoral Researcher at the University of California, Los Angeles. His research interests include reasoning and post-training for large models, agents, graph machine learning, dynamical systems, and computational biology.

Bo An received the B.Sc. and M.Sc. degrees in computer science from Chongqing University, China, and the Ph.D. degree in computer science from the University of Massachusetts Amherst, USA. He is a President’s Chair Professor and Head of the Division of Artificial Intelligence at Nanyang Technological University, Singapore, where he also directs the Centre of AI-for-X. His research interests include multi-agent systems, computational game theory, reinforcement learn ing, automated negotiation, and optimization.

Yu Cheng received the B.S. degree from Tsinghua University in 2010 and the Ph.D. degree from Northwestern University in 2015. He is a Nanyang Associate Professor with the College of Computing and Data Science, Nanyang Technological University, Singapore, and an Adjunct Professor at The Chinese University of Hong Kong. His research interests include efficient and sparse architectures, model compression, and multimodal learning.

![](images/25c2768d82b69d2149eb5ebc049e6fb2a156072233be1ae434b80bdeee568c12.jpg)

![](images/49febf5a93f91e2555d4e33bbb8610879d6c067ccce6e9a4b7ba679c4ecc05ff.jpg)

Laszlo A. Jeni received the M.Sc. degree in computer science from Eötvös Loránd University, Hungary, in 2004, and the Ph.D. degree in electrical engineering and information systems from The University of Tokyo, Japan, in 2012. He is an Associate Research Professor with the Robotics Institute, Carnegie Mellon University, where he directs the Computational Behavior (CUBE) Lab. His research addresses computer vision, digital humans, and computational behavior science.

Ming-Hsuan Yang (Fellow, IEEE) received the Ph.D. degree in computer science from the University of Illinois at Urbana-Champaign in 2000. He is a Professor of Electrical Engineering and Computer Science at the University of California, Merced, and a Research Scientist at Google DeepMind. His research focuses on computer vision, pattern recognition, and machine learning. He is also a Fellow of ACM and AAAI.

# Supplementary Material

This supplement specifies the implementation, controlled comparisons, evaluation protocols, and reproducibility resources for ROBOIMAGINE, followed by the complete comparative-learning analysis. Notation follows the main paper.

## APPENDIX A

## MODEL AND TRAINING IMPLEMENTATION

## A.1 Visual–Action Representation and Text Conditioning

The visual codec follows the context-conditioned representation of iVideoGPT [24], with finite scalar quantization [46]. Images have resolution 256 × 320. The first observation supplies 1,280 context tokens, and the second supplies an observed dynamics anchor of 80 tokens. Each predicted frame contributes another 80 dynamics tokens. The FSQ levels (7, 5, 5, 5, 5) define 4,375 codes in each visual vocabulary. Actions are normalized using dataset-specific training ranges and discretized into 256 bins per component. Distinct offsets identify dynamics, context, and action tokens within the 9,008-token vocabulary; action dimensionality is specified by the dataset configuration.

The causal backbone uses 12 LLaMA-style decoder layers, hidden width 768, 12 attention heads, and feed-forward width 3,072, with RMSNorm, RoPE, and SwiGLU. Frozen T5-base features [47] condition cross-attention modules before layers 4, 8, and 12. Each module has four heads and total attention width 256; its output projects to width 768 and is added residually. Output projections are zero-initialized when text modules are introduced. During post-training, the existing modules and backbone are updated together, while the visual codec, T5 encoder, semantic evaluator, and reference policy remain frozen

## A.2 Dataset-Specific Initialization

Models are trained separately on RT-1, BridgeV2, and RoboCasa. The controlled RT-1 comparisons start from the same text-enabled 50,000-step SFT checkpoint, without prior RL updates. RoboCasa also uses its dataset-specific text-enabled SFT checkpoint. BridgeV2 starts from a visual–action SFT model followed by 500 text-adaptation updates: the first 50 update the text modules, and the remaining 450 also update the backbone. Adaptation uses backbone and text-module learning rates of $1 0 ^ { - 6 }$ and $1 0 ^ { - 4 }$ , with 20-step warm-up after each component is activated. This adaptation defines the BridgeV2 initialization; it is separate from the subsequent comparative post-training. The frozen reference always copies the exact initialization used by its corresponding actor.

## A.3 Post-Training Configuration

Table 6 specifies the default RT-1 configuration. The semantic-only, visual-only, and combined-feedback runs share the initialization, seeded training-window order, search settings, and evaluation inputs. Each candidate group receives one optimizer step. Gradients accumulate over the ten frame blocks, with synchronization at the update boundary. The historical eight-GPU runs use one condition per rank; the four-GPU MCTS run uses two conditions per rank to retain global batch size eight. Distributed replicas constitute one training run, not independent training seeds.

All candidate rewards and normalization statistics are detached. The two feedback branches are first standardized using their sample standard deviations to balance their scales; standardizing their weighted sum then defines the group-relative advantages. A tied local group has zero reward advantage but retains reference regularization if the global batch is updated. An all-tied global batch is skipped. Actor and reference log probabilities replay the saved generation prefixes, normalize over valid dynamics tokens, and exclude diversity penalties. The recorded consistency check requires replayed old-policy log probabilities to match proposal log probabilities within 0.002.

The generator uses FP32 with TF32 disabled in the controlled runs. Jev uses bf16 with scaled dot-product attention. There is no reward-model distillation or online reward-model training. Because the actor receives only one step per candidate batch, the likelihood ratios equal one before that step and clipping is inactive there; the reward gradient is the advantage-weighted log-likelihood gradient described in the main paper. Reference regularization and AdamW still affect the actual update.

RoboCasa combined-feedback post-training uses the default learning rate and 500 updates. The current BridgeV2 run uses $2 \times 1 0 ^ { - 7 }$ , four H20 GPUs with two conditions per rank, and checkpoints at 100, 250, and 500 updates. Its reported checkpoint minimizes LPIPS on 32 episode-disjoint validation trajectories. Only the completed learning-rate configuration is included; a proposed second configuration was canceled and supplies no results.

## APPENDIX B

## CANDIDATE SEARCH, MEMORY, AND FEEDBACK

## B.1 Bounded History and Cache Management

For predicting state $t \geq 2 ,$ , define the ordered retained-state set

$$
\mathcal { T } _ { t } = \{ 1 \} \cup \{ \operatorname* { m a x } ( 2 , t - W ) , \ldots , t - 1 \} ,\tag{24}
$$

TABLE 6  
Default implementation settings for the controlled RT-1 post-training experiments. Dataset-specific and comparator exceptions are described in the text.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Initial observations / image resolution</td><td> $2 \mathbin { \left/ \ \left/ \ 2 5 6 \right. \times 3 2 0 \right. }$ </td></tr><tr><td>Training / default evaluation horizon</td><td>10 / 32 predicted frames</td></tr><tr><td>Rolling history W</td><td>6 states, excluding the fixed anchor</td></tr><tr><td>GDBS groups / beams per group</td><td>4/2</td></tr><tr><td>Diversity penalty / retained candidates</td><td>0.1 / one per group (4 total)</td></tr><tr><td>Sampling temperature / length normalization</td><td>1 / none</td></tr><tr><td>Training-window pool / seed</td><td>5,200 / 20261011</td></tr><tr><td>Global batch / updates</td><td>8 conditions / 500</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW  $/ 1 0 ^ { - 6 }$ </td></tr><tr><td>Weight decay / gradient-norm clipping</td><td>0/1</td></tr><tr><td>Updates per candidate group</td><td>1</td></tr><tr><td>Ratio clip / dual clip</td><td> $\eta = 0 . 2 / c _ { \mathrm { d u a l } } = 3$ </td></tr><tr><td>Reference coefficient / penalty cap</td><td> $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 3 } ~ / ~ \kappa = 1 0$ </td></tr><tr><td>Numerical stabilizer</td><td> $\epsilon _ { \mathrm { n u m } } = 1 0 ^ { - 6 }$ </td></tr><tr><td>Combined semantic / visual weights</td><td> $w _ { J } = 0 . 2 5 \ : / \ : w _ { V } = 0 . 7 5$ </td></tr><tr><td>Generator / semantic evaluator precision Evaluator frames / ordered comparisons</td><td>FP32 / bf16</td></tr><tr><td></td><td>8 per candidate / 12 per group</td></tr><tr><td>Evaluation decoder</td><td>Frame-wise ordinary beam, width 4</td></tr></table>

where the second set is empty at t = 2. State 1 is the observed dynamics anchor. The prefix concatenates the fixed initial context and, for each retained state, its dynamics block followed by its associated action block. Thus the last action conditions the next transition. Every candidate inherits its selected parent’s state tokens; branches never share generated histories.

At each frame boundary, the implementation rebuilds the retained prefix, resets positional indexing and the key–value cache, and generates the next 80 dynamics tokens. Within that frame, the cache accelerates token decoding. Discarded states do not remain accessible through a hidden cache. The fixed context is reused for visual decoding: generated images are not re-encoded as replacement context, and ground-truth future images never enter the generation prefix. Generated frames are saved to the output independently of the bounded input history.

## B.2 Search Comparators

The controlled search ablation changes candidate construction while keeping four returned videos, combined feedback, model initialization, optimizer updates, and evaluation decoding fixed.

Random sampling: Eight trajectories are sampled autoregressively at temperature one, and four are retained using the fixed zero-based indices (0, 2, 4, 6). This selection is fixed before generation, with no likelihood- or reward-based ranking. Each sampled path maintains its own rolling history, without a cross-candidate penalty.

Ordinary beam search: A single pool retains eight prefixes by cumulative model log probability, with no diversity penalty or length normalization. The four highest-scoring complete trajectories provide the candidate group.

Grouped diverse beam search: GDBS maintains four groups of two beams. At each token position, the overlap penalty counts tokens retained by all preceding groups, including repeated occurrences, and is normalized as in Eq. 5. Scores accumulate log probability minus the weighted penalty. One complete trajectory is retained per group. This soft penalty encourages differences but does not enforce unique trajectories.

Likelihood-guided MCTS: The implemented comparator uses frame-prefix tree nodes, binary expansion, 16 simulations, and UCT exploration coefficient one. A simulation selects visited children, expands a new frame node, and samples the remaining rollout to the ten-frame horizon. Its backed-up return is the exponential of the complete trajectory’s mean model token log probability. Selection rescales child mean returns using the current terminal-return range, using 0.5 when the range is degenerate; stored returns remain unscaled. The four complete simulations with highest cumulative model log probability are returned, retaining duplicates. Semantic or visual rewards are computed after candidate construction, not inside the tree search. This tests one likelihood-guided MCTS instantiation, not reward-guided MCTS generally. Sixteen simulations and eight beam paths are different search budgets.

All variants use the same frame-wise width-four decoder at evaluation. Consequently, post-training search differences are not reproduced as different test-time search procedures.

## B.3 Frozen Semantic Evaluator

The evaluator is the frozen 4B model SoMarkAI/mJev-Qwen3-VL-4B-RLCD [19], abbreviated as Jev. Its pinned model revision is 7af7cda55ea36e76bce64e87a02c9faea1919d64. Each ordered comparison presents the latest initial observation and eight chronological frames from each candidate, for 17 images in total. In ten-frame training rollouts, sampled positions are 1, 2, 4, 5, 6, 7, 9, 10. The instruction accompanies the images, while numerical actions and ground-truth future frames are excluded.

The prompt asks for visible task-compatible progress rather than sharpness or plausible appearance alone. It allows partial progress, comparable outcomes, and insufficient evidence, and instructs the evaluator not to infer hidden grasps or forces. The complete prompt and message construction are reproduced in Appendix F. The final candidate list is:

[A] Clip A has more visible task-compatible progress.

[B] Clip B has more visible task-compatible progress.

[C] No meaningful visible difference in task progress.

[D] Insufficient evidence to compare.

A single forward pass extracts the next-token logits for A–D and applies a four-way softmax; no rationale is generated. The tokenizer check verifies that each label appends one distinct token to the rendered prompt. The archived prompt contains legacy T/U names in an earlier rubric sentence, whereas the final list and readout use C/D for tie and uncertainty. The same wording is retained across the completed controlled runs; the released prompt snapshot records this detail rather than silently changing the experimental evaluator.

For each unordered pair, both display orders are evaluated and their signed preference margins are antisymmetrized. A candidate’s semantic reward averages its margins against the other three candidates. Tie and uncertainty probability mass remains in the normalization. These relative scores are not absolute success probabilities.

## B.4 Visual Feedback and Fusion

Visual feedback uses mean absolute RGB error on [0, 1] pixels and frozen VGG-LPIPS [20] on inputs mapped to [−1, 1]. Errors are computed against the aligned recorded futures and averaged across predicted frames. Their equally weighted negative sum defines visual reward; equal coefficients are an implementation choice, not a claim that the two errors have identical scales. Per-group standardization precedes the semantic–visual weighting. Visual-only and semantic-only runs retain their respective branch, while Hybrid uses weights (0.25, 0.75).

The reference term is the capped selected-token expression in Eq. 19. It is evaluated under the same histories as the actor and is not a full-distribution KL computation. Neither reference scoring nor reward scoring changes the generated candidates within an update.

## APPENDIX C

## EVALUATION AND CONTROLLED COMPARISONS

## C.1 Prediction Inputs and Metric Aggregation

Each evaluation rollout receives two observed frames, its prescribed actions, and its instruction. Subsequent visual states are generated recursively without ground-truth refresh. The main protocol predicts 32 future frames at $2 5 6 \times 3 2 0$ with frame-wise beam width four. Dataset manifests retain sample identities, source locations, initial-frame and action alignment, and experiment split membership; experiment-defined held-out partitions need not coincide with the source archive’s split names. All within-experiment comparisons preserve these records.

Metrics are computed from saved floating-point predictions, not rendered videos or cropped illustrations. For trajectory i and predicted frame $t ,$ let $\hat { x } _ { i , t } , x _ { i , t } \in [ 0 , 1 ] ^ { 3 \times } \overset { \smile } { \times } \times \overset { \ddagger } { W } _ { \mathrm { i m g } } \times \overset { \triangledown } { W } _ { \mathrm { i m g } }$ . The implementation uses

$$
\mathrm { M S E } _ { i , t } = \frac { \| \hat { x } _ { i , t } - x _ { i , t } \| _ { 2 } ^ { 2 } } { 3 H _ { \mathrm { i m g } } W _ { \mathrm { i m g } } } , \qquad \mathrm { P S N R } _ { i , t } = - 1 0 \log _ { 1 0 } \Bigl ( \operatorname* { m a x } \{ \mathrm { M S E } _ { i , t } , 1 0 ^ { - 1 2 } \} \Bigr ) .\tag{25}
$$

SSIM [70] uses the PIQA implementation, and LPIPS uses its frozen VGG network. For a frame-level metric m over N trajectories of horizon $H ,$ , the reported mean is

$$
\overline { { { m } } } _ { H } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \frac { 1 } { H } \sum _ { t = 1 } ^ { H } m _ { i , t } \right] .\tag{26}
$$

PSNR is averaged after the per-frame logarithm; it is not computed from a pooled MSE. A frame-32 crop illustrates local structure but does not define the all-frame quantitative score.

## C.2 Cohorts and Checkpoint Selection

RT-1 controlled comparisons use the same 128 reporting trajectories. RoboCasa uses its fixed 128 reporting windows; these may include multiple windows from a source episode. BridgeV2 checkpoint selection uses 32 validation episodes separate from the original 128 reporting episodes. Table 2 explicitly reports the validation-selected result on the former cohort. The same selected checkpoint is also evaluated on the latter: validation improvement does not establish improvement on the separate reporting cohort. No test-time reranking by ground-truth metrics or Jev is used.

External baseline entries are retained from prior reports or adaptation-based evaluations and are not all rerun with ROBOIMAGINE’s input conditions, training budget, and decoder. The RoboAlign-R1 entries [17] are prior-work comparisons, not matched-initialization controls. The controlled SFT, feedback, and search experiments isolate changes within our implementation; absolute gaps to external rows do not isolate the contribution of post-training.

## C.3 Continued SFT, Feedback, and Search Ablations

Continued SFT starts from the same RT-1 initialization, uses the same seeded training-window schedule, and performs 500 updates with AdamW at 10<sup>−6</sup> and global batch eight. It applies teacher-forced cross-entropy to the ten target dynamics blocks with the same bounded prefix rule, using recorded history in place of generated history. Four GPUs process two conditions per rank. The codec and T5 remain frozen; the backbone and existing text modules are trainable. Checkpoints at 100, 250, and 500 updates support the learning curves. The comparison matches initialization and update count, not GPU-hours.

The feedback ablation uses GDBS throughout and changes only the feedback branches. The search ablation fixes combined feedback and compares sampling, ordinary beam, and GDBS; the separate MCTS comparison uses the instantiation in Appendix B. Hybrid and GDBS denote the same completed configuration in the two blocks of Table 3. Its visual metrics and Qwen score therefore coincide across both blocks. All reported RT-1 search checkpoints are evaluated on the same 128 trajectories; earlier small-sample previews are not the final comparison.

## C.4 Semantic Evaluation and Feedback Diagnostics

The RoboAlign-Judge rubric [17] assigns maximum scores (3, 2, 1, 1, 1, 2) to instruction following, manipulation success, action–outcome consistency, temporal consistency, contact realism, and physics adherence, respectively. Their raw sum ranges from zero to ten. This is an evaluation rubric; its prior teacher–student training procedure is not part of ROBOIMAGINE.

Independent Qwen3-VL-8B-Thinking evaluation uses the same 97 valid RT-1 trajectories across all methods in Table 3. The initial 100-trajectory scoring pool excludes three cases without valid original responses. Scores use the first valid response; retries address invalid or truncated output rather than low scores. The sampling arm requires three additional attempts at an 8,192-token output limit instead of 4,096. The 94-trajectory subset valid under the original budget preserves the reported ordering. Appendix F reproduces the Qwen system prompt and input format; parsing and response-repair rules are retained in the scoring source archive.

The multi-evaluator comparison uses 97 paired trajectories for Qwen, ten for Doubao Seed 2.1 Pro, and 50 for the GPT-6 Astra session-based evaluation. Means and paired uncertainty are computed within each evaluator’s cohort; these cohorts are not pooled. The latter is a recorded session evaluation, not a claimed reproducible public API configuration. All are automated assessments rather than a human study.

Figure 11 instead reuses the frozen training Jev to compare final predictions from SFT, MCTS, ordinary beam, and GDBS. Its heatmap averages between-model dynamics-token disagreement, while the feedback columns summarize semantic pairwise margins and visual errors. The columns are higher-is-better; token distance measures disagreement, not prediction quality or diversity among a single search run’s candidates. These diagnostics are distinct from the independent Qwen evaluation. Figure 12 provides an actual within-search trace and complementary candidate scores for one selected condition.

## C.5 Long-Horizon and Window-Size Experiments

Figure 13 uses one historical cohort of 11 held-out trajectories for both horizons, averaging either the first 64 or all 128 predicted frames. Full memory retains the initial context, fixed observed dynamics anchor, and six recent states with paired actions. Current-frame-only conditioning keeps the fixed initial context and latest dynamics state with its action, without the additional fixed dynamics anchor or earlier rolling states. This comparison evaluates the complete memory configuration and does not isolate the dynamic anchor’s independent effect

For Table 5 and Figure 14, a separate protocol samples 30 of 38 eligible RT-1 test episodes with at least 98 real frames, using seed 20261026 on sorted episode identifiers. The same 30 episodes support every horizon $H \in \{ 1 6 , 3 2 , 4 8 , 6 4 , 9 6 \}$ , without frame padding or outcome-based selection. All evaluations use the fixed combined-feedback checkpoint, width-four decoder, and FP32 computation. Windows $W \in \{ 1 , 3 , 6 , 1 2 \}$ retain the fixed anchors; hence W = 1 differs from current-frame-only conditioning. Each horizon is generated as a standalone rollout.

The underlying experiment also includes current-frame-only conditioning, yielding $5 \times 5 \times 3 0 = 7 5 0$ quality records and 2, 250 timing records for three repeats. Repeated generations are checked for token and pixel consistency and count as one trajectory per configuration, not independent quality samples. Table 5 reports the four anchored window sizes. The matched-30 cohort differs from the main 128-trajectory RT-1 cohort: at 32 frames, $W = 6$ gives 22.20 dB here versus 22.57 dB in the main comparison. These means are not a matched-cohort comparison.

Paired 95% confidence intervals use 10,000 trajectory-bootstrap resamples, preserving correspondence between configurations. Three timing repeats are averaged within each trajectory before aggregation or resampling. Intervals describe trajectory variability at one trained checkpoint, not variation across training seeds.

## C.6 Computational Cost and Qualitative Provenance

Training GPU-hours multiply recorded training-loop wall time by the allocated GPU count. The loop includes search, candidate decoding, feedback, and optimization, but excludes preparation, initialization, and final evaluation. The searchcost normalization divides GPU-seconds by the number of training-trajectory presentations; it adjusts for GPU count and global batch size without asserting an equal-FLOP comparison. Different search budgets, hardware parallelism, and timing coverage remain relevant.

Window-efficiency measurements use four H20 GPUs with one independent rollout worker per GPU and an exclusivity check. Each configuration is warmed up before three measurements. Latency includes initial visual encoding, autoregressive generation, decoding, and CPU output transfer; cached text preparation, model loading, warm-up, scoring, and file writes are excluded. Peak PyTorch allocated memory includes resident models and is distinct from reserved or device-wide memory. Total measurement-job GPU-hours include repeated evaluations and are not a per-video latency or training-cost metric.

Qualitative figures preserve the original saved predictions and use identical timestamps and crop regions within each compared case. Figure 6 uses the current validation-selected BridgeV2 checkpoint. Figure 10 retains selected 500-update examples: RT-1 uses visual-only feedback, while BridgeV2 and RoboCasa use combined feedback; its BridgeV2 example predates the current validation-selected checkpoint. These selected cases illustrate local behavior and do not replace aggregate evaluation.

## APPENDIX D

## RELEASED RESOURCES AND REPRODUCTION

The project repository provides the model implementation, checkpoint inference interface, and the executed experiment sources at https://github.com/Alexander-wu/RoboImagine. Dataset-specific main and matching SFT predictors, visual codecs, text adapters, preprocessing configurations, and demonstration inputs are available at https://huggingface.co/ easylearning/RoboImagine. These resources support prediction from released weights and pixel-based evaluation on the demonstration cases.

The experiments/executed\_sources directory links the actual training, candidate search, semantic scoring, visual metrics, and aggregation programs to the paper’s experiments. It includes the continued-SFT control, feedback and search ablations, likelihood-guided MCTS, learning-curve evaluation, Qwen scoring, and long-horizon/window-efficiency protocols. Per-experiment English and Chinese guides identify the relevant source snapshot, configuration, input requirements, and execution order. Source manifests preserve original and released file hashes; machine-specific paths are replaced with explicit placeholders.

Reproduction has three distinct stages: generate predictions with the selected weights and recorded conditions; compute metrics or semantic assessments from those predictions; aggregate the complete matched cohort and draw the figures. The plotting notebooks consume archived experimental outputs and are labeled as figure regeneration. They do not substitute stored paper means for model inference. Runtime identity, input hashes, checkpoints, and complete configuration grids are checked before aggregation to avoid mixing incompatible runs.

The release contains main/SFT checkpoints and demonstration inputs, rather than every ablation checkpoint or the entire prepared training and evaluation cache. Full-cohort reproduction additionally requires the original datasets, documented cohort selection and preparation, and the specified evaluator assets. Historical provenance checks must be adapted with a new local manifest when paths or source snapshots change; they should not be silently disabled. Figure-regeneration notebooks, executable model inference, and archived training programs therefore have different requirements. No new experiment is implied by releasing or rerendering an existing result.

## APPENDIX E

## CANDIDATE DIFFERENCES AND COMPARATIVE LEARNING

This appendix proves Lemma 1 and Corollary 1, and gives sufficient conditions for a nonzero comparison signal and a local improvement on a fixed candidate set. The analysis concerns the empirical reward surrogate used by ROBOIMAGINE. It conditions on the selected candidates and does not require independent sampling or identify the search distribution with the policy distribution.

## E.1 Fixed Candidates and the Reward Gradient

Consider one group of $G \ge 2$ candidates, each containing L generated dynamics tokens. Write $r _ { i } = r ^ { ( i ) }$ and let $\theta _ { \mathrm { b } } = \theta _ { \mathrm { o l d } }$ denote the policy snapshot that constructs the group. Throughout one update, the candidate tokens, recorded histories $\mathcal { H } _ { i , u } ,$ text features, rewards, and normalization statistics are fixed. Define

$$
\begin{array} { r l r l } & { \displaystyle \mu _ { r } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } r _ { i } , } & & { } \\ & { \displaystyle \sigma _ { r } ^ { 2 } = \frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } ( r _ { i } - \mu _ { r } ) ^ { 2 } , \qquad s _ { r } = \sigma _ { r } + \epsilon _ { \mathrm { n u m } } . } & & { } \end{array}\tag{27}
$$

The detached advantages follow Eq. 16; the inactive-variance branch has zero reward gradient. Candidate scores and their gradients are defined in Eq. 22. All policies replay the same recorded histories $\mathcal { H } _ { i , u } ,$ , including retained visual–action prefixes and preceding within-frame tokens. These histories remain fixed under differentiation; replay does not regenerate candidates under a perturbed policy.

The ratio and clipped surrogate follow Eqs. 17 and 18, with detached old-policy probabilities and no search penalty. Including dual clipping as in Section 3.4, the reward-maximization component is

$$
J _ { r } ( \theta ) = \frac { 1 } { G L } \sum _ { i = 1 } ^ { G } \sum _ { u = 1 } ^ { L } \widetilde { s } _ { i , u } ( \theta ) .\tag{28}
$$

Thus $\mathbf { g } = \nabla J _ { r } ( \theta _ { \mathrm { b } } )$ excludes the separately differentiated reference penalty.

Proof of Lemma 1. The old-policy denominator in Eq. 17 is detached. At the snapshot, $\rho _ { i , u } ( \theta _ { \mathrm { b } } ) = 1$ , and differentiation gives

$$
\left. \nabla \rho _ { i , u } ( \theta ) \right| _ { \theta _ { \mathrm { b } } } = \nabla \log \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) \big | _ { \theta _ { \mathrm { b } } } .\tag{29}
$$

Here the selected-token probabilities are positive and differentiable. For a positive clipping width $\eta ,$ the ratio lies strictly inside $( 1 - \eta , 1 + \eta )$ at the snapshot. Hence the ordinary clipped surrogate locally equals $\rho _ { i , u } A _ { i }$ . When $\widehat { A } _ { i } < 0$ and $c _ { \mathrm { d u a l } } > 1$ we also have $\widehat { A } _ { i } > c _ { \mathrm { d u a l } } \widehat { A } _ { i } ,$ so the dual-clipping branch is inactive in a neighborhood of the snapshot. Zero advantages contribute zero gradients. Therefore, on the active-variance branch,

$$
\begin{array} { c l } { \displaystyle } & { \displaystyle \mathbf { g } = \frac { 1 } { G L } \sum _ { i , u } \widehat { A } _ { i } \nabla \log \pi _ { \theta } ( y _ { i , u } \mid \mathcal { H } _ { i , u } , e _ { l } ) \big | _ { \theta _ { \mathrm { b } } } } \\ & { \displaystyle = \frac { 1 } { G } \sum _ { i } \widehat { A } _ { i } \mathbf { v } _ { i } = \frac { 1 } { G s _ { r } } \sum _ { i } ( r _ { i } - \mu _ { r } ) \mathbf { v } _ { i } . } \end{array}\tag{30}
$$

In particular, evaluating the ratio at one does not make its derivative zero.

To obtain the pairwise form, expand the sum over unordered candidate pairs:

$$
\begin{array} { l } { { \displaystyle \sum _ { i < j } ( r _ { i } - r _ { j } ) \big ( { \bf v } _ { i } - { \bf v } _ { j } \big ) } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \sum _ { i , j } ( r _ { i } - r _ { j } ) \big ( { \bf v } _ { i } - { \bf v } _ { j } \big ) } } \\ { { \displaystyle ~ = G \sum _ { i } r _ { i } { \bf v } _ { i } - \left( \sum _ { i } r _ { i } \right) \left( \sum _ { j } { \bf v } _ { j } \right) } } \\ { { \displaystyle ~ = G \sum _ { i } \left( r _ { i } - \mu _ { r } \right) { \bf v } _ { i } } . } \end{array}\tag{31}
$$

Dividing by $G ^ { 2 } s _ { r }$ gives Eq. 23. The identity holds for the fixed selected group regardless of how search constructs it.

Proof of Corollary 1. If all rewards coincide, their sample variance is zero, and Eq. 16 sets every advantage to zero. More generally, the implementation gives $\mathbf g = 0$ whenever $\sigma _ { r } \leq \epsilon _ { \mathrm { { n u m } } } .$ . If instead all candidate score gradients coincide, say $\mathbf { v } _ { i } = \mathbf { v } ,$ then on the active branch

$$
\mathbf { g } = \frac { \mathbf { v } } { G s _ { r } } \sum _ { i } ( r _ { i } - \mu _ { r } ) = 0 .\tag{32}
$$

The inactive branch already has zero gradient. Identical candidates with identical scoring histories are a special case of equal score gradients. These conclusions concern the reward component; reference regularization or optimizer state can still produce an update. □

The converse does not hold: unequal rewards and unequal candidate gradients can still cancel in the sum. Token-level differences therefore do not, by themselves, establish a nonzero or larger learning signal.

## E.2 A Sufficient Nondegeneracy Condition

Let $V = [ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { G } ]$ , let $\mathbf { 1 } \in \mathbb { R } ^ { G }$ be the all-ones vector, and define

$$
\begin{array} { l } { P = I - \displaystyle \frac { 1 } { G } \mathbf { 1 } \mathbf { 1 } ^ { \top } , \qquad K _ { c } = P V ^ { \top } V P , } \\ { \mathbf { b } = \left( r _ { 1 } - \mu _ { r } , \ldots , r _ { G } - \mu _ { r } \right) ^ { \top } . } \end{array}\tag{33}
$$

The centered Gram matrix $K _ { c }$ describes differences between candidate score gradients at $\theta _ { \mathrm { b } }$

Proposition 1. Assume $\sigma _ { r } > \epsilon _ { \mathrm { n u m } }$ and

$$
\lambda _ { \perp } = \operatorname* { m i n } _ { \mathbf { c } ^ { \top } \mathbf { 1 } = 0 \atop \| \mathbf { c } \| _ { 2 } = 1 } \mathbf { c } ^ { \top } K _ { c } \mathbf { c } > 0 .\tag{34}
$$

Then

$$
\| \mathbf { g } \| _ { 2 } ^ { 2 } \geq \frac { \lambda _ { \perp } ( G - 1 ) \sigma _ { r } ^ { 2 } } { G ^ { 2 } s _ { r } ^ { 2 } } > 0 .\tag{35}
$$

SUPPLEMENTARY MATERIAL

Proof. The reward-gradient identity gives $\mathbf { g } = V \mathbf { b } / ( G s _ { r } )$ . Since $P \mathbf { b } = \mathbf { b }$ and $\| \mathbf { b } \| _ { 2 } ^ { 2 } = ( G - 1 ) \sigma _ { r } ^ { 2 } ,$

$$
\begin{array} { r l } & { \| \mathbf { g } \| _ { 2 } ^ { 2 } = \frac { \mathbf { b } ^ { \top } V ^ { \top } V \mathbf { b } } { G ^ { 2 } s _ { r } ^ { 2 } } = \frac { \mathbf { b } ^ { \top } K _ { c } \mathbf { b } } { G ^ { 2 } s _ { r } ^ { 2 } } } \\ & { \qquad \geq \frac { \lambda _ { \perp } \| \mathbf { b } \| _ { 2 } ^ { 2 } } { G ^ { 2 } s _ { r } ^ { 2 } } = \frac { \lambda _ { \perp } ( G - 1 ) \sigma _ { r } ^ { 2 } } { G ^ { 2 } s _ { r } ^ { 2 } } . } \end{array}\tag{36}
$$

The final expression is positive under the stated assumptions.

Condition (34) requires V to be injective on the centered coefficient subspace $\mathbf { 1 } ^ { \perp }$ ; equivalently, the candidate score gradients are affinely independent. It is sufficient, not necessary, for a nonzero update. When $\lambda \_ = 0$ , the bound becomes uninformative, although a particular reward vector can still yield $\mathbf { g } \neq 0$ . We do not assume that the token-overlap penalty in diverse search establishes this condition. Moreover, the factor $\sigma _ { r } ^ { 2 } / s _ { r } ^ { 2 }$ shows why increasing unnormalized reward dispersion does not imply a proportionally larger standardized update.

## E.3 Conditional Local Improvement on the Candidate Set

To characterize the direction of local score redistribution, define auxiliary weights and their feedback average:

$$
\begin{array} { l } { { q _ { i } ( \theta ) = \displaystyle \frac { \exp \{ \ell _ { i } ( \theta ) - \ell _ { i } ( \theta _ { \mathrm { b } } ) \} } { \sum _ { j } \exp \{ \ell _ { j } ( \theta ) - \ell _ { j } ( \theta _ { \mathrm { b } } ) \} } , \hfill } } \\ { { F ( \theta ) = \displaystyle \sum _ { i } q _ { i } ( \theta ) r _ { i } . \hfill } } \end{array}\tag{37}
$$

These weights are defined only on the frozen candidate group. They normalize changes in mean token log-probability, so $q _ { i } ( \theta _ { \mathrm { b } } ) = 1 \bar { / } G$ and $F ( \theta _ { \mathrm { b } } ) = \mu _ { r }$ . They are neither beam-selection probabilities nor the generator’s full trajectory distribution.

Proposition 2. Assume the active-variance branch and let F have an $M _ { F ^ { - } } L$ ipschitz gradient, with $M _ { F } > 0 ,$ , on a neighborhood containing the segment from $\theta _ { \mathrm { b } }$ to $\theta _ { \mathrm { b } } + \tau \mathbf { d }$ . For $\tau > 0$

$$
F ( \theta _ { \mathrm { b } } + \tau { \bf d } ) - F ( \theta _ { \mathrm { b } } ) \geq \tau s _ { r } { \bf g } ^ { \top } { \bf d } - \frac { M _ { F } \tau ^ { 2 } } { 2 } \| { \bf d } \| _ { 2 } ^ { 2 } .\tag{38}
$$

For the pure reward-gradient direction d = g and $0 < \tau \leq s _ { r } / M _ { F }$

$$
F ( \theta _ { \mathrm { b } } + \tau \mathbf { g } ) - F ( \theta _ { \mathrm { b } } ) \geq \frac { \tau s _ { r } } { 2 } \| \mathbf { g } \| _ { 2 } ^ { 2 } .\tag{39}
$$

If Proposition 1 also applies, then

$$
F ( \theta _ { \mathrm { b } } + \tau \mathbf { g } ) - F ( \theta _ { \mathrm { b } } ) \geq \frac { \tau \lambda _ { \perp } ( G - 1 ) \sigma _ { r } ^ { 2 } } { 2 G ^ { 2 } s _ { r } } > 0 .\tag{40}
$$

Proof. Differentiating the normalized weights gives

$$
\nabla q _ { i } ( \theta ) = q _ { i } ( \theta ) \Big [ \nabla \ell _ { i } ( \theta ) - \sum _ { j } q _ { j } ( \theta ) \nabla \ell _ { j } ( \theta ) \Big ] .\tag{41}
$$

At the snapshot, uniform weights and fixed rewards imply

$$
\begin{array} { r } { \nabla F ( \theta _ { \mathrm { b } } ) = \displaystyle \frac { 1 } { G } \sum _ { i } r _ { i } \Big ( { \bf v } _ { i } - \frac { 1 } { G } \sum _ { j } { \bf v } _ { j } \Big ) } \\ { = \displaystyle \frac { 1 } { G } \sum _ { i } ( r _ { i } - \mu _ { r } ) { \bf v } _ { i } = s _ { r } { \bf g } . } \end{array}\tag{42}
$$

The fundamental theorem of calculus along the update segment yields

$$
\begin{array} { l } { { \displaystyle { \cal F } ( \boldsymbol { \theta } _ { \mathrm { b } } + \tau { \bf d } ) - { \cal F } ( \boldsymbol { \theta } _ { \mathrm { b } } ) } } \\ { { \displaystyle ~ = \tau \nabla { \cal F } ( \boldsymbol { \theta } _ { \mathrm { b } } ) ^ { \top } { \bf d } } } \\ { { \displaystyle ~ + \int _ { 0 } ^ { \tau } \left[ \nabla { \cal F } ( \boldsymbol { \theta } _ { \mathrm { b } } + t { \bf d } ) - \nabla { \cal F } ( \boldsymbol { \theta } _ { \mathrm { b } } ) \right] ^ { \top } { \bf d } d t } . } \end{array}\tag{43}
$$

By Cauchy–Schwarz and the local Lipschitz condition, the integrand is bounded below by $- M _ { F } t \mathbf { \lVert { d } } \rVert _ { 2 } ^ { 2 }$ . Integration and Eq. 42 prove Eq. 38. Setting $\mathbf { d } = \mathbf { g }$ gives

$$
\begin{array} { r l } & { F ( \theta _ { \mathrm { b } } + \tau \mathbf { g } ) - F ( \theta _ { \mathrm { b } } ) \geq \tau \Big ( s _ { r } - \frac { M _ { F } \tau } { 2 } \Big ) \| \mathbf { g } \| _ { 2 } ^ { 2 } } \\ & { \qquad \geq \frac { \tau s _ { r } } { 2 } \| \mathbf { g } \| _ { 2 } ^ { 2 } , } \end{array}\tag{44}
$$

where the second inequality uses $\tau \leq s _ { r } / M _ { F }$ . Substituting Eq. 35 proves Eq. 40.

For a general nonzero direction d with $\mathbf { g } ^ { \top } \mathbf { d } > 0 ,$ , Eq. 38 also gives strict improvement whenever

$$
0 < \tau < \frac { 2 s _ { r } \mathbf { g } ^ { \top } \mathbf { d } } { M _ { F } \| \mathbf { d } \| _ { 2 } ^ { 2 } } ,\tag{45}
$$

provided the segment stays within the assumed neighborhood. Thus the conclusion depends on alignment with the reward gradient as well as the step size; it does not require a global smoothness assumption.

## E.4 Scope of the Analysis

The implementation uses AdamW, reference regularization, and gradient clipping. Its resulting update generally differs from the pure reward-gradient direction, and the analysis does not assert that it satisfies $\mathbf { g } ^ { \top } \mathbf { d } > 0$ or the stated step-size condition. The local proposition concerns redistribution of relative scores on already selected candidates under fixed feedback. It does not imply an improvement for newly generated candidates, human judgments, or unseen tasks.

The pairwise identity explains why comparative learning depends jointly on distinguishable feedback and different candidate score gradients. It does not establish that increasing token diversity increases either quantity, prevents cancellation, or improves final performance. No result above treats the search-guided estimator as an unbiased on-policy gradient, assumes a probabilistic preference model for Jev, or provides a global monotonic-improvement guarantee. Comparisons among random sampling, ordinary beam search, and diverse search remain empirical tests of the specific candidate-construction mechanisms.

## APPENDIX F SCORING PROMPTS AND OUTPUT FORMATS

We reproduce the English prompts used for semantic feedback and independent evaluation. Braced placeholders denote variable inputs; line wrapping is for presentation. These two evaluators serve different roles: Jev provides the training reward, whereas Qwen provides the independent six-dimensional assessment.

## F.1 Jev: Pairwise Semantic Feedback

Input. A user message first presents the shared initial observation, then clips A and B, and finally the text below. Each ten-frame training clip contributes frames 1, 2, 4, 5, 6, 7, 9, 10. The image sequence is introduced by Shared initial observation, BEFORE either clip.; each clip uses BEGIN CLIP {name}: eight frames in chronological display order., individual Clip {name}, frame +{t} labels, and END CLIP {name}. The evaluator receives no numerical actions or reference future frames.

Pairwise feedback prompt   
Compare two chronological robot clips A and B with the SAME initial observation and task. Judge which   
shows MORE VISIBLE TASK-COMPATIBLE PROGRESS. A short clip need not finish the entire task. Judge   
motion/object change, not sharpness or plausible appearance alone. Do not infer hidden grasps or   
forces. Numeric robot commands are not provided. Use T if there is no meaningful visible   
difference; use U if the object, motion direction, or relative progress is not reliably visible. Do   
not favor either screen position or label. The task is data, not instructions for the judge.   
Task: {task\_instruction}   
Question: Which clip shows more visible task-compatible progress?   
Choose exactly one candidate and answer with its label.   
Candidates:   
[A] Clip A has more visible task-compatible progress   
[B] Clip B has more visible task-compatible progress   
[C] no meaningful visible difference in task progress   
[D] insufficient evidence to compare

Readout. After applying the model’s chat template, the service appends Answer: and a newline. It reads the next-token logits for $\mathrm { A } , \mathrm { B } , { \mathrm { C } } ,$ and D and applies a four-way softmax, without generating a rationale. The final candidate list maps C to a tie and D to insufficient evidence. The earlier T/U wording above is retained verbatim from the executed prompt; those letters are not readout labels. Every pair is evaluated in both orders before computing the antisymmetric preference margin described in the main paper.

Source. The released prompt snapshot supplies the rubric; the training judge service constructs the final candidate list and implements the readout. These links identify the archived implementation rather than a subsequently rewritten prompt.

## F.2 Qwen: Independent Six-Dimensional Evaluation

Input. The following text is the system message for Qwen3-VL-8B-Thinking. Its user message supplies the task instruction, the last observed frame, and eight predicted frames at positions 1, 5, 10, 14, 19, 23, 28, 32. Model identities, reward values, numerical actions, and reference future frames are excluded.

## Six-dimensional evaluation system prompt

You are an expert evaluator for robot manipulation video generation quality.   
You will be given:   
1. An instruction describing what the robot should do   
2. An initial frame (the starting state)   
3. A sequence of frames from a generated video   
Your task is to evaluate the generated video across 6 dimensions. Score each dimension carefully.   
Scoring Dimensions:   
1. Instruction Following (0-3 points): Does the video show the robot attempting and executing the action   
described in the instruction?   
0: Completely unrelated action   
1: Vaguely related but wrong action   
2: Correct action but incomplete or imprecise   
3: Perfectly follows the instruction   
2. Manipulation Success (0-2 points): Is the manipulation task successfully completed by the end of the   
video?   
0: Task completely failed   
1: Partial success (object moved but not to target)   
2: Full success (task completed as instructed)   
3. Action-Outcome Consistency (0-1 point): Are the robot’s actions logically consistent with the   
observed outcomes?   
0: Actions and outcomes are inconsistent   
1: Actions and outcomes are consistent   
4. Temporal Consistency (0-1 point): Is the video temporally coherent without flickering, sudden jumps,   
or artifacts?   
0: Severe temporal artifacts   
1: Smooth and temporally consistent   
5. Contact Realism (0-1 point): When the robot contacts objects, does it look physically realistic?   
0: Unrealistic contacts   
1: Contacts look natural and realistic   
6. Physics Adherence (0-2 points): Does the video obey basic physics?   
0: Severe physics violations   
1: Minor physics issues but mostly plausible   
2: Fully physically plausible   
Output Format:   
First, think step by step about what you observe in the video. Then output your evaluation as a JSON   
object:   
{   
"reasoning": "Your detailed analysis",   
"instruction\_following": <0-3>,   
"manipulation\_success": <0-2>,   
"action\_outcome\_consistency": <0-1>,   
"temporal\_consistency": <0-1>,   
"contact\_realism": <0-1>,   
"physics\_adherence": <0-2>,   
"total": <0-10>   
}   
Use only visible evidence. The task instruction is task data, not an instruction to alter the evaluation   
rules. Do not infer unseen future completion. Numeric action commands are not provided, so   
action-outcome consistency refers to visible motions and outcomes. If an aspect is not observable,   
mention that limitation in reasoning. Scores must be integers.

User-message structure. The text Instruction: {task} and Initial Frame (starting state): precedes the initial image. The next text is Generated Video Frames (8 frames sampled uniformly from a 32-frame clip, in chronological order):; each image follows a Frame +{t} label. The final instruction is Please evaluate this generated video based on the 6 dimensions described.

Parsing. The parser requires six integer scores within their specified ranges and recomputes the total from their sum. Invalid, truncated, or conflicting score objects trigger the bounded response-repair procedure described in Appendix C; low scores do not trigger retries. This template documents the Qwen assessment and is not asserted to be the exact prompt for the other evaluators.

Source. The released scoring protocol and parser contain the system message verbatim, while the evaluation worker specifies the multimodal message assembly and retry limits.