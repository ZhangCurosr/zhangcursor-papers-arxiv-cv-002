# VideoScout: Learning Agentic Active Exploration with Adaptive Reasoning Pacing for Long Video Understanding

Weixin Xu , Zhenyu Yang , Bing Wang , Shengsheng Qian , Member, IEEE, Changsheng Xu , Fellow, IEEE

Abstract—Multimodal Large Language Models (MLLMs) have achieved remarkable progress on short video understanding yet remain limited on long videos due to the limited visual context window. Prevailing approaches rely on uniform frame sampling or recent coarse-to-fine agentic zooming, both of which struggle to localize sparse, decisive evidence in sufficiently long videos. We formulate long video understanding as a Sequential Evidence Acquisition (SEA) problem, in which an agent reads the video turn by turn along the temporal axis, deciding at each turn how fast to watch, what evidence to retain, when to revisit uncertain segments, and when to stop and answer. Inspired by this view, we propose VideoScout, a multi-turn reasoning agent that instantiates the SEA paradigm through adaptive reasoning pacing. Specifically, by dynamically controlling the viewing pace, VideoScout enables efficient traversal of long videos within a bounded visual context window, allowing the agent to access more video content while balancing content analysis depth with reading efficiency. To train VideoScout, we construct VideoScout-66K, a set of over 66K high-quality exploration turns from 10K answer-verified trajectories, and adopt a two-stage pipeline: cold-start supervised fine-tuning teaches the agent per-turn output format, while the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm performs trajectory-level reinforcement learning with a composite reward that jointly considers answer accuracy, output format compliance, and the temporal alignment between the agent’s viewing progress and the teacher’s answer timing measured by intersection-over-union (IoU). Extensive experiments on long video understanding and reasoning benchmarks demonstrate that our 7B model achieves strong performance compared with existing trained 7B agentic models.

Index Terms—Long video understanding, multimodal large language models, reinforcement learning, multi-turn reasoning.

## I. INTRODUCTION

L <sup>ONG</sup> <sup>video</sup> <sup>understanding</sup> <sup>underpins</sup> <sup>a</sup> <sup>wide</sup> <sup>range</sup> <sup>of</sup> real-world applications, from video surveillance [1], [2] and instructional content analysis [3]–[6] to long-range film understanding [7]–[9]. Recent Multimodal Large Language Models (MLLMs) have achieved strong performance on short video clips [10]–[14]. However, long videos introduce a fundamental bottleneck: a two-hour film at 24 fps contains more than 170,000 frames, and even when downsampled to 1 fps it still exceeds the visual context window of current 7B MLLMs by more than an order of magnitude [10], [15]. As a result, the model can observe only a limited subset of frames, and the selected subset directly determines which evidence is available for answering the query. Long-video understanding therefore reduces to a question of how the model, under a bounded perturn visual capacity, should acquire effective evidence from a video that vastly exceeds what it can see at once.

A first line of work addresses this problem with predetermined observation strategies. Uniform frame sampling [10], [16]–[23] and query-conditioned keyframe selection [24]–[31] commit to a frame set before reasoning begins, so once decisive evidence is excluded, no subsequent reasoning can recover it, as illustrated in Fig. 1(a). Coarse-to-fine agentic methods such as LongVT [32] and VideoTemp-o3 [33] introduce multiple turns and iteratively zoom into informative temporal windows identified from an initial low-frame-rate overview [34], [35], but the zoom selection remains anchored to a one-shot global scan that, on sufficiently long videos, is itself too sparse to indicate where the decisive evidence lies, as illustrated in Fig. 1(b). In both cases, the model never actively seeks out the sparse evidence scattered across the video; it can only reason over whatever an up-front observation plan happens to surface. This motivates our Core Challenge 1: how can we build an agentic active exploration strategy for long videos, where decisive evidence is sparse and difficult to gather and reason over?

Yet active exploration alone is not enough. Long videos contain a substantial amount of redundant or irrelevant content, and reasoning indiscriminately over every chunk is neither feasible nor desirable, since many segments simply do not warrant careful reasoning. Existing efforts attempt to address this redundancy by compressing long videos into memory representations [9], [36], [37] or by pre-selecting queryrelevant frames [24], [27], but in both cases the allocation of visual capacity is decided once and frozen, with no way to dynamically refine it as understanding of which parts are truly relevant to the query develops over time. This motivates our Core Challenge 2: under a bounded visual capacity, how can such an agent achieve efficient observation and reasoning by concentrating its effort on segments worth examining?

To address both challenges, we propose VideoScout, a trained multi-turn video agent that interacts with the video turn by turn rather than committing to a one-shot observation plan. For Challenge 1, VideoScout introduces cross-turn textual notes coupled with a rewatch action to realize agentic active exploration: at each turn the agent observes only a short segment and reasons over its accumulated notes, then may either continue forward or revisit a previously fast-forwarded chunk for closer inspection; the notes carry query-relevant evidence across turns, so sparse decisive evidence is gathered iteratively and every new observation is conditioned on what the agent has already understood. For Challenge 2, VideoScout equips the agent with an adaptive speed action that realizes efficient observation and reasoning under bounded visual capacity: the agent fast-forwards through uninformative passages at 2× or 4× and slows down to 1× on segments worth examining, concentrating its limited capacity exactly where it matters; a final answer action commits the rollout once enough evidence has been gathered, avoiding indiscriminate reasoning over every chunk. Fig. 1(c) illustrates VideoScout on the same event-centric query.

![](images/75ef492de21c97e92e462d8a47b346a269ec41b415b79865987bd6718bc229df.jpg)  
Fig. 1. VideoScout versus prior long video understanding paradigms on the same event-centric query. (a) Uniform sampling picks frames at fixed intervals and simply does not include the key frame. (b) Coarse-to-fine zoom anchors its zoom window on a sparse first-pass summary that has already missed the key frame, so the zoom window lands on the wrong interval and the evidence remains outside. (c) VideoScout consumes the video turn by turn, adapts its viewing pace over contiguous turn spans (bracketed N× labels), maintains evolving textual Notes, and rewatches an earlier turn from the current turn when new evidence calls for it, arriving at the correct answer.

We train VideoScout on top of Qwen2.5-VL-7B [10] with a two-stage pipeline tailored to multi-turn agentic exploration. We first construct VideoScout-66K, a dataset of over 66K verified exploration turns from 10K answer-verified trajectories, and perform turn-level cold-start supervised fine-tuning (SFT) to teach the agent the output format and per-turn action selection behavior. We then apply the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [38] algorithm at the trajectory level with a composite reward that jointly considers answer accuracy, output format compliance, and temporal alignment with teacher viewing behavior measured by intersection-over-union, thereby aligning the optimization objective with the complete multi-turn reasoning process rather than isolated individual turns.

At a higher level, the design of VideoScout reflects a more fundamental shift in how long-video understanding should be approached: from passively encoding a fixed set of frames to actively reading a long video as a human would. A human viewer does not commit to a fixed observation strategy in advance; instead, she skims irrelevant passages, slows down at important moments, takes notes of salient cues, and occasionally revisits segments that were passed too quickly. Each action depends on what has already been observed, allowing the observation strategy to continuously evolve together with the reasoning process. We formalize this human-like reading as a paradigm: long-video question answering under bounded visual context is a Sequential Evidence Acquisition (SEA) problem, in which the agent reads the video turn by turn along the temporal axis, deciding at each turn how fast to watch, what evidence to retain, when to revisit uncertain segments, and when to stop and answer, turning the question of how to encode a fixed video into how to allocate limited observations over time.

In summary, our contributions are as follows:

• We formulate long-video question answering under bounded visual context as a turn-by-turn temporal reading process, in which an agent advances through the video sequentially in time and at each turn decides how fast to watch, what evidence to retain, when to revisit uncertain segments, and when to stop and answer, analogous to how a human reads a long passage. We refer to this formulation as the Sequential Evidence Acquisition (SEA) problem.

• We introduce VideoScout, a trained multi-turn video agent that implements SEA through adaptive-speed observation, compact textual notes, and local rewatch actions. By coupling observation and reasoning in a turn-by-turn process, VideoScout can trade off temporal coverage and visual detail, enabling efficient traversal of long videos while preserving the ability to inspect sparse decisive evidence.

• We construct VideoScout-66K comprising over 66K verified exploration turns from 10K answer-verified trajectories, and develop a two-stage training pipeline in which turn-level cold-start SFT learns the output format and perturn action behavior, while trajectory-level DAPO with a composite reward over accuracy, format, and timing IoU aligns the optimization objective with the complete multiturn reasoning process.

• Extensive experiments on long video understanding and reasoning benchmarks demonstrate that our 7B agent achieves strong performance compared with existing trained 7B agentic models.

## II. RELATED WORK

## A. Long Video Understanding

Early approaches to long video understanding rely on uniform frame sampling [16]–[18], [39]–[42] or queryconditioned keyframe selection [28], [43]–[49], which commit to a fixed set of frames before reasoning and therefore often miss sparse yet decisive evidence. For instance, queryconditioned keyframe selection methods such as Adaptive Keyframe Sampling (AKS) [24] typically select keyframes based on semantic relevance to the query through image-text matching scores, which may still overlook temporally dispersed evidence that requires multi-step reasoning to connect. Another line of work augments MLLMs with running memory. Methods such as MA-LMM [9], MovieChat [36], Flash-VStream [50], LongVU [51], and Video-XL [37] compress videos into continuously updated memory representations, but fine-grained visual cues may be lost once compressed. More recently, training-free agentic pipelines [52]–[57] employ powerful proprietary large language models to iteratively query clip-level visual tools for reasoning. For example, Deep Video Discovery [54] treats long video understanding as agentic search with tool use for temporal retrieval and object tracking. Although effective, their performance is tightly coupled to closed-source backbones, making them difficult to reproduce, deploy, or directly compare with trained open-source models. In contrast, our approach trains a self-contained 7B agent whose observation policy is jointly optimized together with the reasoning process in an end-to-end manner.

## B. Tool-Augmented Video Agents with Reinforcement Learning

Reinforcement learning, originally adopted to align language models with human preferences [58]–[62], has become a key paradigm for enhancing the reasoning ability of multimodal large language models [63]–[65], with Group Relative Policy Optimization (GRPO) [66] emerging as a particularly effective optimization algorithm. The success of this paradigm is further demonstrated by DeepSeek-R1 [67], which scales reasoning RL to elicit chain-of-thought capabilities at scale. Building upon this paradigm, recent works [14], [68]–[71] extend GRPO-style post-training from text to video, achieving strong improvements on video question answering and temporal grounding tasks. For example, Video-R1 [12] proposes T-GRPO to explicitly encourage temporal reasoning during RL post-training. However, these methods largely remain singleturn: the model generates both reasoning traces and final answers in a single pass over passively provided visual inputs, without actively interacting with the video during inference. A parallel line of research introduces agentic pipelines through coarse-to-fine temporal zoom strategies [33], [72]–[74]. For example, LongVT [32] first inspects a sparse low-frame-rate overview and then iteratively zooms into informative temporal windows identified from the overview. Although substantially more adaptive than single-turn baselines, its exploration strategy remains anchored to an initial global scan, whose reliability degrades as video duration increases. Our work advances this direction by formulating long video understanding as a genuine multi-turn reasoning problem under bounded visual context, and by optimizing the agent’s observation policy using the DAPO algorithm [38] with a composite reward over answer accuracy, output format compliance, and temporal alignment via IoU-based timing rewards.

## III. METHODOLOGY

This section details how VideoScout instantiates the ${ \mathrm { S e } } { \mathrm { - } }$ quential Evidence Acquisition (SEA) paradigm as a trained multi-turn agent. We first present the task formulation and agent behavior, then describe the data generation pipeline that yields VideoScout-66K, followed by the cold-start supervised fine-tuning and trajectory-level reinforcement learning stages.

## A. Overview

1) Task formulation. Given a long video V and a query $q ,$ we split V into N contiguous chunks $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { N } \}$ of fixed duration, and let an agent policy $\pi _ { \theta }$ interact with the video turn by turn until it emits an answer action. At each turn $t ,$ the agent observes only the current visual clip $v _ { t }$ and the textual memory $m _ { t - 1 }$ carried from the previous turn, and samples a structured output

$$
\left( r _ { t } , m _ { t } , a _ { t } \right) \sim \pi _ { \theta } ( \cdot \mid v _ { t } , m _ { t - 1 } , q ) ,\tag{1}
$$

consisting of a reasoning trace $r _ { t }$ , an optionally updated textual memory $m _ { t } .$ , and exactly one action $a _ { t }$ drawn from

$$
a _ { t } \ \in \ \big \{ \mathrm { s p e e d } ( s ) , \ \mathtt { r e w a t c h } ( k ) , \ \mathtt { a n s w e r } ( \hat { y } ) \big \} ,\tag{2}
$$

where $s ~ \in ~ \{ 1 \times , 2 \times , 4 \times \}$ controls the pace of the next observation, rewatch(k) requests a closer re-examination of a chunk $c _ { k }$ drawn from the current observation window, and answer(ˆy) terminates the rollout.

2) Agent behavior. The clip $v _ { t }$ and its pace s are coupled through a fixed per-turn visual token budget: at speed(1×), $v _ { t }$ covers a single chunk densely sampled at the highest perframe resolution the budget allows, while higher speeds cover proportionally more chunks under the same budget, trading frame density and per-frame resolution for temporal coverage. The textual memory $m _ { t }$ has a fixed word budget $L _ { m } ;$ ; at each turn the agent may update the memory by writing a new complete version of $m _ { t }$ , or leave $m _ { t } ~ = ~ m _ { t - 1 }$ when nothing new is relevant to $q .$ Because an update carries the entire memory forward rather than a diff, the agent must actively decide at each update which prior observations to keep, rephrase, or drop. The rewatch(k) action is available only when v spans multiple chunks, i.e., when the current pace is 2× or 4×; the chosen index k must refer to a chunk inside the current window, and the next turn is dedicated to re-examining c<sub>k</sub> alone at 1×. Forward progression then automatically resumes from the chunk immediately after the original fast-forward window, so rewatching recovers detail missed during fast-forwarding without rewinding the agent’s overall progress through V. The answer(ˆy) action is emitted when the accumulated memory and observations are deemed sufficient to respond to q. Beyond the notation defined above, we denote the current observation window by W and use a forward pointer p to mark the next unseen chunk, which advances only on forward observations so that the agent moves through the video monotonically apart from on-demand rewatching. Fig. 2 illustrates one such rollout, and we summarize this active exploration process in Algorithm 1.

![](images/c41ac6815809743970a73e9c2a6f3af77431d0919e11946130cf0b4bfc2fa8c6.jpg)  
Fig. 2. Overview of the VideoScout framework for long-video reasoning. Given a long video and a query, the video is first split into multiple fixed-duration chunks. The agent then enters an iterative multi-turn reasoning process, where at each turn it receives the current visual clip and the textual memory from the previous turn, generates a reasoning trace, and selects one action from three options: speed(s) adjusts the playback pace to advance quickly or slow down, rewatch(k) requests a closer re-examination of a specific chunk for more detail, and answer(ˆy) commits to the final answer, terminating the rollout.

3) Scalability. Both v and m <sub>−</sub> are of bounded size by construction, so the per-turn inference cost of π<sub>θ</sub> is constant in the video length |V|. The total cost of a rollout therefore scales only with the number of turns taken, which in turn depends on how aggressively the agent fast-forwards rather than on the underlying video length.

4) Training overview. We train VideoScout in two stages that operate on the same trajectories at different granularities, as illustrated in Fig. 3. The first stage is a cold-start supervised fine-tuning that treats each turn of a trajectory as an independent training example and optimizes it at the turn level, teaching the agent the output format and per-turn action behavior. The second stage is reinforcement learning that optimizes the entire multi-turn trajectory as a whole at the trajectory level, aligning the objective with the complete exploration process rather than isolated turns. This two-granularity view motivates our data construction, which produces both turnlevel supervision and complete trajectories, as described next.

![](images/c3f2773ca82c92827a6f3b25488f2967d31320670756b5b7fe449f1b8f43bf8c.jpg)  
Fig. 3. Comparison between the SFT and RL training schemes. SFT optimizes individual turns independently at the turn level, where each training case corresponds to a single turn of a trajectory, whereas RL optimizes the entire multi-turn trajectory as a whole at the trajectory level.

## B. Data Generation Pipeline

To cold-start the multi-turn reasoning agent, we construct the VideoScout-66K dataset, a collection of over 66K verified exploration turns from 10K answer-verified trajectories for long-video multiple-choice question answering. We build this dataset through a five-stage data generation pipeline, as shown in Fig. 4.

1) Material Preparation. We curate training queries from publicly available long-video MCQ benchmarks, including

Algorithm 1 Active Exploration Reasoning of VideoScout   
Require: video ${ \overline { { V , } } }$ query q, policy $\pi _ { \theta }$   
Ensure: predicted answer $\hat { y }$   
1: split $\bar { V }$ into contiguous chunks $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { N } \}$   
2: // Initialize the exploration state   
3: turn $t \gets 1 ,$ , forward pointer $p \gets 1 ,$ speed $s \gets 1 \times$   
4: memory $m _ { 0 } \gets \emptyset ,$ , no pending rewatch   
5: while the agent has not committed to an answer do   
6: // Observation stage: form the clip for this turn   
7: if a rewatch of $c _ { k }$ is pending then   
8: v<sub>t</sub> ← re-sample $c _ { k }$ at $1 \times$ for a closer look   
9: clear the pending rewatch   
10: else   
11: W ← chunks from $c _ { p }$ covered under speed s   
12: v ← sample W under speed s   
13: $p \gets p + | W |$ // advance the forward pointer   
14: end if   
15: // Reasoning and action stage   
16: $( r _ { t } , m _ { t } , a _ { t } ) \sim \pi _ { \theta } ( \cdot \mid v _ { t } , m _ { t - 1 } , q )$   
17: if $a _ { t } = \mathsf { a n s w e r } ( \hat { y } )$ then   
18: // Commit: evidence is sufficient   
19: return $\hat { y }$   
20: else if $a _ { t } = { \tt r e w a t c h } ( k )$ and $s \neq 1 \times$ and $c _ { k } \in W$   
then   
21: // Revisit: re-examine a skimmed chunk next turn   
22: mark $c _ { k }$ as pending rewatch   
23: else if $a _ { t } = { \tt s p e e d } ( s ^ { \prime } )$ then   
24: // Re-pace: skim faster or slow down   
25: $s \gets s ^ { \prime }$   
26: end if   
27: $t \gets t + 1$   
28: end while   
29: Note: the memory $m _ { t }$ is the only state carried across turns;   
the forward pointer $p$ advances solely on forward obser  
vations, so the agent traverses V monotonically except for   
the on-demand rewatch.

VideoMarathon [75], ScaleLong [76], LongVideo-Reason [77], and TVQA-Long [78]. Each query is accompanied by its source video and ground-truth answer.

As shown in Tab. I, the visual sampling configuration varies across different viewing modes. Tab. II shows the chunk duration, which is automatically determined based on the total video length. Both training and inference use the same chunking and sampling logic.

2) Teacher Trajectory Distillation. We employ Gemini 3 Flash [79] as the teacher model. The teacher receives the complete VideoScout system prompt, including the action space, the Think-Act format, and the memory mechanism, but without the ground-truth answer. It produces a multi-turn exploration trajectory, deciding at each step how to watch and when to answer while maintaining a textual memory.

3) Rejection Sampling. Some generated trajectories may produce incorrect final answers. We resample a trajectory up to eight times until the teacher’s final answer matches the ground truth, and discard the question if no correct trajectory is obtained within this budget. Only trajectories with correct

answers are retained.

4) Reasonableness Validation. Beyond answer correctness, we perform additional filtering to ensure the trajectory is reasonable: reasoning consistency is checked, the answer tag matches the trajectory conclusion, no premature truncation is detected, and the trajectory contains meaningful reasoning traces.

5) Output. After filtering, the pipeline yields approximately 10K verified trajectories comprising over 66K individual turns, with a duration distribution skewed toward long videos as shown in Fig. 5(a): approximately 49% of samples exceed 30 minutes, 27% fall within 3-10 minutes, 23% are under 3 minutes, and 1% span 10-30 minutes. These trajectories are used for supervised fine-tuning. For reinforcement learning, we first filter high-quality queries from validated trajectories, then supplement with additional Video-R1 [12] samples and re-balance the duration distribution to form a diverse pool of roughly 4K prompts as shown in Fig. 5(b). Compared to the SFT data, the RL pool exhibits a more uniform duration distribution with 33% exceeding 30 min, 30% within 3-10 min, 25% spanning 10-30 min, and 12% under 3 min, which ensures that the policy receives balanced exploration signals across short, medium, and long video scenarios.

## C. Cold-Start Supervised Fine-Tuning

Single-turn supervised fine-tuning. For supervised finetuning, we treat every turn of every trajectory as an independent training example: the input consists of the query q, the current visual clip $v _ { t } ,$ and the inherited memory $m _ { t - 1 } ,$ and the target is the structured output $( r _ { t } , m _ { t } , a _ { t } )$ defined by Eq. (1). We optimize the standard token-level cross-entropy loss on the agent’s output only, masking out the visual clip and the user prompt. This single-turn imitation teaches the model the Think-Act format, the legal action space of Eq. (2), and a first pass over how to trade off pace, memory updates, and rewatching based on the currently observed content. The agent is initialized from Qwen2.5-VL-7B-Instruct [10] with all parameters fine-tuned.

## D. Trajectory-Level Reinforcement Learning

1) Multi-turn rollout. The SFT-initialized agent produces well-formed trajectories but receives no direct signal that rewards accurate and temporally efficient exploration. We therefore optimize the full multi-turn rollout with Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [38] on roughly 4K prompts that are collected through filtering validated trajectories and supplementing with additional data sources, sampled in balance across short, medium, and long video durations as well as diverse data sources. For each $( V , q )$ we sample a group of $G$ independent trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ from $\pi _ { \theta } ,$ each unrolled until an answer is emitted. For trajectory $\tau _ { i }$ with terminal reward $R ( \tau _ { i } )$ , the group-relative advantage is

$$
A _ { i } = \frac { R ( \tau _ { i } ) - \frac { 1 } { G } \sum _ { j } R ( \tau _ { j } ) } { \mathrm { s t d } _ { j } R ( \tau _ { j } ) } ,\tag{3}
$$

and we optimize the decoupled clipped surrogate objective over the agent’s output tokens along every trajectory, with

![](images/46730ec7525422f0539c14bd667eef4a483c0ad07f3b5d2c24f4ad4d72264afd.jpg)  
Fig. 4. The five-stage data generation pipeline of VideoScout-66K: Material Preparation, Teacher Trajectory Distillation, Rejection Sampling, Reasonableness Validation, and Output.

![](images/4e38f024a2bdde1643bb6fa84ec4b70abf6773cef0d9faf97371798be97092d0.jpg)  
Fig. 5. Distribution of the training data. (a) The SFT data (66K turns) and (b) the RL data (4K prompts). The outer ring shows the composition of data sources, and the inner ring shows the video duration distribution.

separate upper and lower clipping bounds to encourage exploration:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D A P O } } ( \theta ) = \mathbb { E } _ { ( V , q ) \sim \mathcal { D } , \{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } } } \\ & { \quad \quad \quad \quad \quad \quad \bigg [ \frac { 1 } { \sum _ { i = 1 } ^ { G } \sum _ { j = 1 } ^ { n _ { i } } | o _ { i , j } | } \displaystyle \sum _ { i = 1 } ^ { G } \sum _ { j = 1 } ^ { n _ { i } } \sum _ { t = 1 } ^ { | o _ { i , j } | } } \\ & { \quad \quad \quad \quad ( C _ { i , j , t } - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) ) \bigg ] , } \end{array}\tag{4}
$$

where $n _ { i }$ is the number of output turns in trajectory $\tau _ { i } , ~ o _ { i , j }$ denotes the output sequence of turn $j , o _ { i , j , t }$ its t-th token, and the per-token clipped surrogate is

$$
\begin{array} { r l } & { C _ { i , j , t } = \mathrm { m i n } \Bigl ( r _ { i , j , t } ( \theta ) A _ { i } , } \\ & { \qquad \mathrm { c l i p } \bigl ( r _ { i , j , t } ( \theta ) , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } \bigr ) A _ { i } \Bigr ) , } \end{array}\tag{5}
$$

where the importance sampling weight is

$$
r _ { i , j , t } ( \theta ) = \frac { \pi _ { \theta } \big ( o _ { i , j , t } \mid q , o _ { i , < t } \big ) } { \pi _ { \theta _ { \mathrm { o l d } } } \big ( o _ { i , j , t } \mid q , o _ { i , < t } \big ) } .\tag{6}
$$

Here $A _ { i }$ is the group-relative advantage from Eq. (3), shared by all output tokens of trajectory $\tau _ { i } ,$ , and $\epsilon _ { \mathrm { l o w } }$ and $\epsilon _ { \mathrm { h i g h } }$ are independent lower and upper clipping thresholds that decouple the clipping bounds to allow more aggressive exploration. Following DAPO, we additionally adopt dynamic sampling, which discards prompts whose sampled group of trajectories is entirely correct or entirely incorrect, as such groups yield zero advantage and no gradient, and continues sampling until the batch is filled with informative prompts.

TABLE I  
VIDEOSCOUT VIDEO SAMPLING CONFIGURATION. FRAME COUNT, RESOLUTION, AND COVERAGE FOR DIFFERENT VIEWING MODES.
<table><tr><td>Speed</td><td>Frames</td><td>|Resolution</td><td>Covered Chunks</td></tr><tr><td>Normal (1×)</td><td>20</td><td>448 × 448</td><td>1</td></tr><tr><td>2× Fast-forward</td><td>16</td><td>448 × 448</td><td>2</td></tr><tr><td>4× Fast-forward</td><td>12</td><td>316 × 316</td><td>4</td></tr><tr><td>Rewatch</td><td>20</td><td>448 × 448</td><td>1</td></tr></table>

TABLE II

VIDEOSCOUT CHUNK SEGMENTATION STRATEGY. CHUNK DURATION BASED ON VIDEO LENGTH.
<table><tr><td>Video Duration |</td><td>Chunk Duration</td></tr><tr><td>&lt; 30s</td><td>10s</td></tr><tr><td>30s − 2min</td><td>15s</td></tr><tr><td>2min – 5min</td><td>20s</td></tr><tr><td>5min – 15min</td><td>30s</td></tr><tr><td>≥ 15min</td><td>45s</td></tr></table>

2) Composite reward. The terminal reward $R ( \tau )$ jointly captures answer quality, output validity, and temporal alignment between the agent’s viewing progress and the teacher’s answer timing:

$$
R ( \tau ) ~ = ~ r _ { \mathrm { a c c } } ( \tau ) ~ + ~ r _ { \mathrm { f m t } } ( \tau ) ~ + ~ r _ { \mathrm { t i m i n g } } ( \tau ) ,\tag{7}
$$

where $r _ { \mathrm { a c c } } ( \tau ) ~ \in ~ \{ 0 , 1 \}$ is an accuracy reward obtained by matching the predicted answer yˆ against the ground truth, and $r _ { \mathrm { f m t } } ( \tau ) ~ \in ~ \{ 0 , 0 . 3 \}$ rewards well-formed outputs with valid answer tags and no malformed turns. The timing reward incentivizes the agent to answer at a viewing progress similar to the teacher trajectory, which is critical for ensuring that the agent makes its answer at an appropriate moment during video exploration. Letting M denote the agent’s forward progress when it commits to an answer, measured as the number of chunks it has advanced through, and $T$ the chunk index at which the teacher answered, we compute the Intersection over

Union (IoU) between the agent’s viewing interval $A = [ 0 , M ]$ and the teacher’s viewing interval $B = [ 0 , T ]$

$$
\mathrm { I o U } = { \frac { | A \cap B | } { | A \cup B | } } = { \frac { \mathrm { m i n } ( M , T ) } { \mathrm { m a x } ( M , T ) } } ,\tag{8}
$$

and then scale it to obtain the timing reward:

$$
r _ { \mathrm { t i m i n g } } ( \tau ) \ = \ \mathrm { I o U } \cdot 0 . 5 \cdot \mathbb { I } [ r _ { \mathrm { a c c } } ( \tau ) = 1 ] ,\tag{9}
$$

The IoU measures the overlap between the agent’s and teacher’s viewing intervals, both starting from the first chunk. Crucially, $r _ { \mathrm { t i m i n g } }$ is awarded only when the answer is correct, preventing the policy from trading accuracy for timing alignment.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Implementation Details. We initialize our model from Qwen2.5-VL-7B-Instruct [10] for its strong foundational capabilities and multimodal comprehension. Our training consists of two stages. In the cold-start supervised fine-tuning stage, we train on the VideoScout-66K dataset comprising over 66K exploration turns from 10K answer-verified trajectories. We set the learning rate to $1 . 5 \times 1 0 ^ { - 5 }$ with a cosine scheduler and 5% warmup, a global batch size of 16, and train for 3 epochs on 8 NVIDIA H800 GPUs. We adopt packing [80] with a maximum sequence length of 16,384 tokens. For video input, sampling is performed as shown in Tab. I.

In the reinforcement learning stage, we apply DAPO [38] optimization over the trajectory-level rollout with a group size of 8. We use a learning rate of $5 \times 1 0 ^ { - 6 }$ , a KL regularization coefficient of 0.01 with low-variance KL estimation [66], and decoupled clipping bounds of $\epsilon _ { \mathrm { l o w } } = 0 . 2 0$ and $\epsilon _ { \mathrm { h i g h } } = 0 . 2 8$ The RL training also runs for 1 epoch on 8 NVIDIA H800 GPUs, with a maximum prompt length of 36,000 tokens and a maximum response length of 4,096 tokens.

2) Benchmarks. We evaluate VideoScout on nine benchmarks organized into two categories: long video understanding and reasoning benchmarks that assess holistic understanding and reasoning over long videos, including perception, temporal relation modeling, causal reasoning, and multi-step reasoning, and cross-domain video understanding benchmarks that evaluate generalization across diverse knowledge domains.

Long video understanding and reasoning benchmarks. This category comprises four benchmarks that primarily assess the model’s holistic understanding and reasoning over long videos, covering perception, memory, temporal relation modeling, causal reasoning, and multi-step reasoning. MLVU [81] contains videos ranging from 10 minutes to 2 hours with questions that require multi-dimensional reasoning over spatial, temporal, and causal cues; we report results on its held-out test set (MLVU-test). LVBench [82] features videos up to 2 hours with diverse question types including counting, spatial relation, temporal ordering, action recognition, and key information retrieval. Video-MME [83] features videos from 6 to 60 minutes and covers a broad spectrum of tasks ranging from perception and recognition to temporal reasoning and information synthesis; we adopt the subtitle-free setting in our evaluation. MINERVA [84] focuses on complex multi-step reasoning over long videos, requiring the model to integrate evidence across multiple temporal windows.

Cross-domain video understanding benchmarks. This category comprises five benchmarks that evaluate generalization across diverse knowledge domains, including multidisciplinary knowledge, scientific videos, world commonsense, and visual perception. VideoMMMU [85] requires multi-step reasoning over multi-disciplinary video content, with subtasks including comprehension, adaptation, and perception. SciVideoBench [86] evaluates scientific video understanding that requires domain-specific knowledge. WorldSense [87] tests everyday video comprehension grounded in world commonsense; since our model takes only visual input, we evaluate on its vision-only setting without audio. VSI-Bench [88] focuses on visual-spatial perception and integration. MMVU [89] assesses integrated perception-and-reasoning capabilities across diverse video types; we evaluate on its multiple-choice subset (MMVU-mc).

3) Baseline Methods. We compare VideoScout against two groups of baselines: proprietary MLLMs and open-source MLLMs. Within each group, we further distinguish between standard single-pass models and agentic frameworks that perform iterative, tool-augmented exploration over the video.

Proprietary MLLMs. We include three representative closed-source models: GPT-4o [90], Gemini-1.5-Pro [91], and Gemini-2.5-Pro [92]. These models process videos in a single forward pass with large context windows and serve as strong upper-bound references, but they are not agentic frameworks and do not perform iterative exploration.

Open-source MLLMs. Among open-source models, we distinguish two sub-categories. The first comprises standard single-pass models that directly answer queries from uniformly sampled frames without iterative exploration, including LongVA-7B [13], Video-R1-7B [12], VideoChat-R1-7B [14], VideoRFT-7B [68], Open-o3-Video-7B [71], and our base model Qwen2.5-VL-7B [10]. The second comprises agentic frameworks that iteratively explore the video through multiturn reasoning or tool use, including Video-MTR-7B [73], VideoMind-7B [74], Video-CoM-7B [72], LongVT-7B [32], and VideoTemp-o3-7B [33]. To ensure a fair comparison, we additionally construct an agentic variant of our base model, denoted Qwen2.5-VL-7B<sup>‡</sup>, which adopts the same multi-turn exploration framework as VideoScout but without our twostage training. The Agentic Framework column in Tab. III and Tab. IV indicates whether each method performs agentic exploration (✓) or operates in a single pass (✗).

## B. Main Results

Tables III and IV report the performance of VideoScout against proprietary and open-source baselines, and Table V summarizes the average number of observed frames and reasoning turns of VideoScout on each benchmark, reflecting how its adaptive reasoning pacing allocates its observations and reasoning turns across videos of different characteristics. For a fair comparison, the number of frames used by other methods on each benchmark is also kept within a range that does not significantly exceed that of our method. VideoScout achieves the best performance among open-source methods on both benchmark categories, reaching an average of 48.0 on long video understanding and reasoning and 45.0 on cross-domain video understanding. VideoScout places first or second on most benchmarks, and its active exploration paradigm delivers steady improvements across a wide range of video types.

TABLE III  
QUANTITATIVE RESULTS ON LONG VIDEO UNDERSTANDING AND REASONING BENCHMARKS. THE BEST SCORE IS IN BOLD, THE SECOND-BEST IS UNDERLINED, AND OUR METHOD IS HIGHLIGHTED WITH GRAY SHADING. ‡ DENOTES AN AGENTIC VARIANT OF THE BASE MODEL THAT ADOPTS THE SAME MULTI-TURN EXPLORATION FRAMEWORK AS VIDEOSCOUT BUT WITHOUT OUR TWO-STAGE TRAINING.
<table><tr><td>Methods</td><td>Agentic Framework</td><td>LVbench Overall</td><td>MINERVA Overall</td><td>MLVU(test) Overall</td><td colspan="2">Video-MME(w/o subtitle) Overall Long</td><td>Average Score</td></tr><tr><td colspan="8">Proprietary MLLMs</td></tr><tr><td>GPT-4o [90]</td><td></td><td>30.8</td><td>45.5</td><td>54.9</td><td>71.9</td><td>65.3</td><td>50.8</td></tr><tr><td>Gemini-1.5-pro [91]</td><td>X××</td><td>33.1</td><td></td><td></td><td>75.0</td><td>67.4</td><td></td></tr><tr><td>Gemini-2.5-Pro [92]</td><td></td><td>67.4</td><td>61.8</td><td>79.6</td><td>82.4</td><td>77.6</td><td>72.8</td></tr><tr><td colspan="8">Open-Source MLLMs</td></tr><tr><td>LongVA-7B [13]</td><td></td><td>37.9</td><td></td><td>41.4</td><td>52.6</td><td>46.2</td><td></td></tr><tr><td>Video-R1-7B [12]</td><td>××</td><td>40.1</td><td>29.1</td><td>48.0</td><td>59.3</td><td>50.3</td><td>44.1</td></tr><tr><td>VideoChat-R1-7B [14]</td><td>X</td><td>39.3</td><td>30.3</td><td>45.2</td><td>58.2</td><td>50.7</td><td>43.3</td></tr><tr><td>VideoRFT-7B [68]</td><td>X</td><td>18.7</td><td>29.2</td><td></td><td>59.8</td><td>50.7</td><td></td></tr><tr><td>Video-MTR-7B [73]</td><td></td><td>42.3</td><td></td><td>50.4</td><td></td><td></td><td></td></tr><tr><td>VideoMind-7B [74]</td><td></td><td>40.8</td><td></td><td></td><td>58.2</td><td>49.2</td><td></td></tr><tr><td>Video-CoM-7B [72]</td><td></td><td>42.5</td><td>31.7</td><td>44.0</td><td>59.4</td><td></td><td>44.4</td></tr><tr><td>LongVT-7B [32]</td><td></td><td>41.3</td><td>28.7</td><td>38.5</td><td>54.8</td><td>47.6</td><td>40.8</td></tr><tr><td>Open-o3-Video-7B [71]</td><td></td><td>44.2</td><td>31.2</td><td>36.1</td><td>63.6</td><td>54.9</td><td>43.8</td></tr><tr><td>VideoTemp-03-7B [33]</td><td></td><td>39.2</td><td>31.3</td><td>48.2</td><td>56.0</td><td>47.9</td><td>43.7</td></tr><tr><td>Qwen2.5-VL-7B [10]</td><td>X</td><td>33.7</td><td>29.5</td><td>45.5</td><td>58.4</td><td>50.2</td><td>41.8</td></tr><tr><td>Qwen2.5-VL-7B [10]</td><td></td><td>31.4</td><td>22.9</td><td>33.1</td><td>44.9</td><td>39.7</td><td>33.1</td></tr><tr><td>VideoScout-7B(Ours)</td><td></td><td>45.1</td><td>35.4</td><td>51.0</td><td>60.4</td><td>52.1</td><td>48.0</td></tr><tr><td>∆ vs. Qwen2.5-VL-7B</td><td>X</td><td>↑11.4</td><td>↑5.9</td><td>↑5.5</td><td>↑2.0</td><td>↑1.9</td><td>↑6.2</td></tr><tr><td>∆ vs. Qwen2.5-VL-7B‡</td><td></td><td>↑13.7</td><td>↑12.5</td><td>↑17.9</td><td>↑15.5</td><td>↑12.4</td><td>↑14.9</td></tr></table>

1) Long video understanding and reasoning. As shown in Table III, VideoScout obtains the best open-source results on LVBench, MINERVA, and MLVU-test, scoring 45.1, 35.4, and 51.0 respectively, and reaches the highest average score of 48.0. On Video-MME it also attains a strong second-best result of 60.4.

Compared with the two variants of our base model, VideoScout shows clear advantages in overall performance. Relative to Qwen2.5-VL-7B, which processes videos through uniform sampling, VideoScout raises the average score by 6.2 points, with per-benchmark gains of 11.4 on LVBench, 5.9 on MINERVA, 5.5 on MLVU, and 2.0 and 1.9 on the two Video-MME splits. The agentic variant Qwen2.5-VL-7B<sup>‡</sup>, which adopts the same multi-turn framework as VideoScout but without our two-stage training, performs even worse than the base model, with its average dropping from 41.8 to 33.1. This is because the untrained model tends to reason and commit to an answer prematurely rather than gathering sufficient evidence, which underscores the necessity of our two-stage training. VideoScout surpasses this untrained variant by 14.9 points on average. When further compared with other agentic frameworks such as Video-MTR-7B and VideoTempo3-7B, VideoScout again leads on every reported metric. These results show that adaptive reasoning pacing offers a more effective exploration strategy for locating decisive evidence in long videos than fixed multi-turn reasoning or coarse-tofine zooming. This adaptivity is further reflected in Table V: VideoScout spends more frames and reasoning turns on longer benchmarks such as LVBench, while using far fewer on shorter ones, demonstrating that it allocates its observations and reasoning turns according to video length rather than following a fixed observation schedule.

2) Cross-Domain Video Understanding. As shown in Table IV, VideoScout obtains the highest average score of 45.0 among open-source methods. It ranks first on VideoMMMU, SciVideoBench, and WorldSense, scoring 54.6, 28.9, and 40.8 respectively, and also leads the comprehension and adaptation sub-tasks of VideoMMMU. On the remaining benchmarks it stays competitive, reaching 65.9 on the VideoMMMU perception sub-task, second only to Open-o3-Video-7B, and obtaining 35.0 on VSI-Bench and 65.7 on MMVU-mc, the latter close to the strongest open-source results. Its leading average therefore reflects strong and balanced generalization across knowledge domains rather than specialization on a few tasks.

The comparison with the two variants of our base model again highlights the importance of training. VideoScout raises the average score by 7.2 points over Qwen2.5-VL-7B and by 10.7 points over the untrained agentic variant Qwen2.5- VL-7B<sup>‡</sup>, with the latter once again falling below the base model. This reinforces our finding that the improvements arise from the two-stage training rather than the agentic framework itself. Among agentic baselines, VideoScout also surpasses the average of VideoTemp-o3-7B, which reaches 43.4, while leading on knowledge-intensive benchmarks such as VideoMMMU and SciVideoBench. Overall, these results demonstrate that adaptive exploration not only benefits long video understanding and reasoning but also transfers well to diverse cross-domain settings.

TABLE IV  
QUANTITATIVE RESULTS ON CROSS-DOMAIN VIDEO UNDERSTANDING BENCHMARKS. THE BEST SCORE IS IN BOLD, THE SECOND-BEST IS UNDERLINED, AND OUR METHOD IS HIGHLIGHTED WITH GRAY SHADING. ‡ DENOTES AN AGENTIC VARIANT OF THE BASE MODEL THAT ADOPTS THE SAME MULTI-TURN EXPLORATION FRAMEWORK AS VIDEOSCOUT BUT WITHOUT OUR TWO-STAGE TRAINING.
<table><tr><td>Methods</td><td>Agentic Framework</td><td>Comprehension Adaptation Perception Overall</td><td colspan="3">VideoMMMU</td><td colspan="3">SciVideoBench WorldSense VSI-Bench MMVU(mc) Overall Overall</td><td colspan="3">Average Overall Score</td></tr><tr><td colspan="10">Proprietary MLLMs</td><td></td></tr><tr><td></td><td></td><td>62.0</td><td></td><td>66.0</td><td>61.2</td><td>24.9</td><td>42.6</td><td>34.0</td><td></td><td></td><td>46.8</td></tr><tr><td>GPT-4o [90] Gemini-1.5-pro [91]</td><td></td><td>53.3</td><td>55.6 49.3</td><td>59.0</td><td>53.8</td><td>27.5</td><td>48.0</td><td></td><td>42.1</td><td>71.5</td><td></td></tr><tr><td>Gemini-2.5-Pro [92]</td><td>×××</td><td></td><td></td><td></td><td>83.6</td><td>64.3</td><td></td><td>65.1</td><td>45.1</td><td>75.7</td><td>66.8</td></tr><tr><td colspan="10">Open-Source MLLMs</td></tr><tr><td>LongVA-7B [13]</td><td></td><td></td><td></td><td></td><td>24.0</td><td>14.3</td><td></td><td></td><td>29.2</td><td></td><td></td></tr><tr><td>Video-R1-7B [12]</td><td>x×</td><td>46.0</td><td>44.2</td><td>62.3</td><td>50.8</td><td>26.8</td><td></td><td></td><td>31.8</td><td>63.8</td><td>41.7</td></tr><tr><td>VideoChat-R1-7B [14]</td><td></td><td>45.6</td><td>40.2</td><td>53.0</td><td>46.3</td><td>26.5</td><td></td><td>35.5</td><td>33.9</td><td>64.8</td><td></td></tr><tr><td>VideoRFT-7B [68]</td><td>××</td><td>42.0</td><td>36.7</td><td>53.1</td><td>44.0</td><td></td><td>25.7</td><td>38.2</td><td></td><td>66.7</td><td></td></tr><tr><td>Video-CoM-7B [72]</td><td></td><td></td><td></td><td></td><td>50.2</td><td></td><td>27.6</td><td></td><td>30.0</td><td>65.4</td><td></td></tr><tr><td>LongVT-7B [32]</td><td></td><td>43.7</td><td>35.7</td><td>56.7</td><td>45.3</td><td>25.3</td><td></td><td>36.1</td><td>34.4</td><td>63.4</td><td>40.9</td></tr><tr><td>Open-o3-Video-7B [71]</td><td></td><td>52.0</td><td>37.2</td><td>67.7</td><td>52.3</td><td>26.0</td><td></td><td>37.5</td><td>38.5</td><td>67.0</td><td>44.3</td></tr><tr><td>VideoTemp-03-7B [33]</td><td></td><td>47.0</td><td>43.5</td><td>57.3</td><td>49.3</td><td>27.7</td><td></td><td>38.7</td><td>36.8</td><td>64.8</td><td>43.5</td></tr><tr><td>Qwen2.5-VL-7B [10]</td><td>X</td><td>36.1</td><td>35.9</td><td>57.6</td><td>43.2</td><td>16.4</td><td></td><td>35.5</td><td>32.5</td><td>61.3</td><td>37.8</td></tr><tr><td>Qwen2.5-VL-7B‡ [10]</td><td></td><td>38.0</td><td>37.3</td><td>45.2</td><td>40.2</td><td>21.6</td><td></td><td>30.6</td><td>21.1</td><td>58.0</td><td>34.3</td></tr><tr><td>VideoScout-7B(Ours)</td><td></td><td>52.7</td><td>45.5</td><td>65.9</td><td>54.6</td><td></td><td>28.9</td><td>40.8</td><td>35.0</td><td>65.7</td><td>45.0</td></tr><tr><td>∆ vs. Qwen2.5-VL-7B</td><td>X</td><td>↑16.6</td><td>↑9.6</td><td>↑8.3</td><td>↑11.4</td><td></td><td>↑12.5</td><td>↑5.3</td><td>↑2.5</td><td>↑4.4</td><td>↑7.2</td></tr><tr><td>∆ vs. Qwen2.5-VL-7B</td><td></td><td>↑14.7</td><td>↑8.2</td><td>↑20.7</td><td>↑14.4</td><td></td><td>↑7.3</td><td>↑ 10.2</td><td>↑13.9</td><td>↑7.7</td><td>↑10.7</td></tr></table>

TABLE V

AVERAGE NUMBER OF OBSERVED FRAMES AND REASONING TURNS OF VIDEOSCOUT ON EACH BENCHMARK.
<table><tr><td></td><td>Overall</td><td>LVbench MINERVA Overall</td><td>Overall</td><td>MLVU(test) Video-MME(w/o subtitle) Overall</td><td>VideoMMMU Overall</td><td>SciVideoBench Overall</td><td>WorldSense Overall</td><td>VSI-Bench Overall</td><td>MMVU Overall</td></tr><tr><td>Avg Frames</td><td>257.4</td><td>146.1</td><td>129.0</td><td>135.8</td><td>112.0</td><td>119.7</td><td>66.1</td><td>74.4</td><td>46.7</td></tr><tr><td>Avg Turns</td><td>16.6</td><td>8.5</td><td>8.1</td><td>8.2</td><td>6.3</td><td>7.2</td><td>3.8</td><td>4.3</td><td>2.7</td></tr></table>

TABLE VI

ABLATION ON THE TWO-STAGE TRAINING PIPELINE. THE BEST SCORE ISIN BOLD, AND OUR FULL MODEL IS HIGHLIGHTED WITH GRAY SHADING.
<table><tr><td>Setting</td><td>LVbench Overall</td><td>VideoMMMU Overall</td><td>MLVU(test) Overall</td><td>MINERVA Overall</td></tr><tr><td>SFT only</td><td>43.4</td><td>49.4</td><td>47.0</td><td>32.5</td></tr><tr><td>RL only</td><td>18.5</td><td>37.7</td><td>23.7</td><td>8.5</td></tr><tr><td>SFT+RL(GRPO)</td><td>44.5</td><td>47.1</td><td>48.2</td><td>31.3</td></tr><tr><td>SFT+RL(DAPO)</td><td>45.1</td><td>54.6</td><td>51.0</td><td>35.4</td></tr></table>

## C. Ablation Study

In this section, we conduct ablation studies on the core components of VideoScout to validate the effectiveness of the two-stage training pipeline, the composite reward, and the adaptive exploration design. All ablations are evaluated on LVBench, VideoMMMU, MLVU-test, and MINERVA.

1) Effectiveness of the Two-Stage Training Pipeline. We first examine the contribution of each training stage, with results reported in Table VI. Using SFT alone already provides a reasonable starting point, reaching 43.4 on LVBench, 49.4 on VideoMMMU, 47.0 on MLVU, and 32.5 on MINERVA, as cold-start fine-tuning teaches the model the multi-turn exploration format and transfers the teacher’s exploration behavior.

In contrast, applying RL without SFT initialization causes a severe collapse across all benchmarks, dropping to 18.5 on LVBench, 37.7 on VideoMMMU, 23.7 on MLVU, and 8.5 on MINERVA. This is because, without a cold start, the agent has no prior understanding of the action space and the output format, so it rarely produces well-formed trajectories and the sparse trajectory-level reward cannot provide a usable learning signal from scratch. These two observations indicate that the supervised and reinforcement stages are complementary and both indispensable: SFT establishes the basic exploration behavior, while RL refines it toward more accurate and efficient trajectories. We further compare two RL algorithms applied on top of the same SFT initialization. SFT+RL with GRPO brings only a marginal gain on LVBench, from 43.4 to 44.5, and even degrades on VideoMMMU and MINERVA, falling to 47.1 and 31.3. We attribute this instability to the symmetric clipping in GRPO, which couples the advantage estimation across the long multi-turn trajectories and makes the policy update sensitive to a few high-variance samples. In comparison, SFT+RL with DAPO consistently achieves the best results on all four benchmarks, reaching 45.1, 54.6, 51.0, and 35.4, and improving over SFT alone by 1.7, 5.2, 4.0, and 2.9 points respectively. These results show that the decoupled clipping bounds of DAPO yield more stable trajectory-level updates than GRPO and allow the agent to effectively refine its pace-selection and answer-timing behavior, which is essential for realizing the full potential of the two-stage training.

TABLE VII  
ABLATION ON THE COMPOSITE REWARD COMPONENTS IN THEREINFORCEMENT LEARNING STAGE. THE BEST SCORE IS IN BOLD, ANDOUR FULL MODEL IS HIGHLIGHTED WITH GRAY SHADING.
<table><tr><td>Setting</td><td colspan="4">|LVbench VideoMMMU MLVU(test) MINERVA</td></tr><tr><td></td><td>Overall</td><td>Overall</td><td>Overall</td><td>Overall</td></tr><tr><td>RL w/o Timing Reward</td><td>44.3</td><td>52.4</td><td>46.2</td><td>33.3</td></tr><tr><td>RL w/o Accuracy Reward</td><td>44.4</td><td>53.0</td><td>48.0</td><td>32.3</td></tr><tr><td>RL w/o Format Reward</td><td>42.3</td><td>50.6</td><td>48.4</td><td>34.0</td></tr><tr><td>VideoScout-7B(Ours)</td><td>45.1</td><td>54.6</td><td>51.0</td><td>35.4</td></tr></table>

![](images/d230355aef1823cc3045239abf822eb45e10db22e408fb6412ef260140d281ce.jpg)  
Fig. 6. Total reward curves during the RL stage with and without the timing reward. For clarity, we apply an exponential moving average with a smoothing coefficient of 0.9 to suppress step-wise noise and reveal the overall training trend.

2) Effectiveness of the Composite Reward. We further dissect the contribution of each reward component in the RL stage, as detailed in Table VII. Removing the format reward leads to the largest degradation on LVBench, lowering the score from 45.1 to 42.3, because well-formed outputs with valid answer tags are essential for stable training and reliable answer extraction. Removing the accuracy reward also weakens performance, as it provides the primary signal for answer correctness. Removing the timing reward causes a consistent drop as well, indicating that aligning the agent’s answer timing with the teacher encourages it to commit only after gathering sufficient evidence, neither prematurely nor after redundant observation. We further visualize the effect of the timing reward in Fig. 6, which plots the total reward curves during RL training with and without the timing reward. With the timing reward, the total reward rises faster and the agent gradually learns to commit to an answer at the right moment, leading to more pronounced overall improvement. Without the timing reward, the total reward barely changes, indicating that the agent fails to explore more effective behaviors and the training makes little progress. This shows that the timing reward guides the agent to answer at a more appropriate moment in the exploration process, which in turn drives more effective exploration. Overall, these results show that the three reward signals are complementary, jointly regularizing answer quality, output validity, and exploration efficiency, and that their combination is necessary for the full performance of VideoScout.

3) Effectiveness of the Adaptive Exploration Design. Finally, we ablate the core design choices that enable adaptive exploration, covering the frame budget, the chunk duration, the speed policy, and the memory mechanism, with results reported in Table VIII.

TABLE VIII  
ABLATION ON THE CORE DESIGN CHOICES OF VIDEOSCOUT. THE BEST SCORE IS IN BOLD, AND OUR FULL MODEL IS HIGHLIGHTED WITH GRAY SHADING. † DENOTES A SETTING THAT IS RE-TRAINED WITH RL UNDER THE CORRESPONDING CONFIGURATION, USED TO TEST WHETHER ALLOCATING MORE FRAMES BRINGS FURTHER IMPROVEMENT.
<table><tr><td>Ablation Dimension</td><td>Setting</td><td colspan="3">LVbenchVideoMMMUMLVU(test)</td></tr><tr><td></td><td></td><td>Overall</td><td>Overall</td><td>Overall</td></tr><tr><td rowspan="3">Frame Budget</td><td>More frames†|</td><td>44.3</td><td>51.8</td><td>51.0</td></tr><tr><td>Fewer frames</td><td>44.8</td><td>51.7</td><td>48.2</td></tr><tr><td>Uniform Res</td><td>45.3</td><td>54.5</td><td>50.0</td></tr><tr><td rowspan="2">Chunk Duration</td><td>Fixed 10 s</td><td>42.2</td><td>48.0</td><td>46.6</td></tr><tr><td>Fixed 30 s</td><td>44.5</td><td>52.3</td><td>49.2</td></tr><tr><td rowspan="2">Speed Policy</td><td>|Only 1×</td><td>43.6</td><td>49.1</td><td>47.4</td></tr><tr><td>Only 4×</td><td>41.7</td><td>50.9</td><td>45.6</td></tr><tr><td rowspan="2">Core Mechanism</td><td>w/o Notes</td><td>42.9</td><td>51.4</td><td>42.0</td></tr><tr><td>w/o Rewatch</td><td>42.5</td><td>51.6</td><td>49.0</td></tr><tr><td>VideoScout-7B(Ours)</td><td></td><td>45.1</td><td>54.6</td><td>51.0</td></tr></table>

Frame Budget. We first study the frame budget, which determines how many frames are sampled and at what resolution under each viewing speed. As summarized in Table I, our default configuration samples 20, 16, and 12 frames at the 1×, 2×, and 4× speeds, and adopts an adaptive resolution that assigns lower per-frame resolution to faster speeds. We compare it against three alternatives. The more-frames setting increases the per-speed frame counts to 25, 20, and 16 and is re-trained with RL under its own configuration rather than merely evaluated, so as to fairly test whether observing more frames is beneficial. The fewer-frames setting instead reduces the counts to 12, 10, and 8. The uniform-resolution setting keeps the same frame counts but applies the highest resolution to all speeds. As reported in Table VIII, none of the three alternatives improves over the default budget consistently across the three benchmarks. The more-frames setting reaches only 44.3 on LVBench, still below the default score of 45.1, even though it is given additional RL training. The fewerframes setting drops to 48.2 on MLVU, and the uniformresolution setting marginally surpasses the default on LVBench at 45.3 versus 45.1 but falls behind on VideoMMMU and MLVU. These results suggest that increasing the number of frames mainly introduces redundant content that dilutes the truly informative evidence, whereas reducing frames means losing fine-grained temporal and spatial cues. Applying a uniformly high resolution also fails to help, since the extra visual tokens are spent on fast-forward frames that are meant for coarse coverage rather than fine inspection, yielding little accuracy gain. Overall, the default configuration achieves the best balance between information sufficiency and visual redundancy, and the advantage of VideoScout stems from learning where and how carefully to look rather than from naively observing more frames.

Chunk Duration. We then examine the chunk duration, which controls the temporal granularity of each observation.

TABLE IX  
COMPARISON OF AVERAGE INFERENCE TIME PER SAMPLE (AIT, IN SECONDS) AND ACCURACY (ACC) OF DIFFERENT AGENTIC FRAMEWORKS ACROSS BENCHMARKS. THE BEST SCORE IS IN BOLD, AND OUR METHOD IS HIGHLIGHTED WITH GRAY SHADING.
<table><tr><td>Methods</td><td>LVbench MLVU(test) MINERVA WorldSense AIT(s) Acc AIT(s) Acc</td><td> AIT(s) Acc AIT(s) Acc</td><td></td><td></td></tr><tr><td>LongVT-7B [32]</td><td>10.9 41.3 11.0</td><td>38.5 10.7</td><td>28.7</td><td>9.2 36.1</td></tr><tr><td>VideoTemp-03-7B [33]</td><td>21.3 39.2 18.7</td><td>48.2 21.7</td><td>31.3 8.9</td><td>38.7</td></tr><tr><td>VideoScout-7B(Ours)</td><td>35.1 45.1 22.3</td><td>51.0 33.6 35.4</td><td>8.1</td><td>40.8</td></tr></table>

As summarized in Table II, our default scheme adaptively determines the chunk length according to the total video duration, using shorter chunks for short videos and longer chunks for long ones. We compare it against two fixedduration alternatives, one that splits every video into 10-second chunks and another that uses 30-second chunks regardless of video length. As reported in Table VIII, both fixed schemes consistently underperform the adaptive one. The 10-second setting suffers the most, with the LVBench score dropping sharply from 45.1 to 42.2, while the 30-second setting alleviates this degradation but still trails the default scheme. We attribute this to the mismatch between a single fixed granularity and the wide range of video lengths. Overly short chunks fragment coherent events and require far more turns to cover the whole video, so the reasoning effort directed at the question itself is overwhelmed by redundant per-turn reasoning and memory updates, whereas overly long chunks coarsen the temporal resolution and make it harder to localize decisive moments. These results show that the chunk granularity should scale with video length rather than remain fixed.

Speed Policy. We further analyze the speed policy, which decides how the agent distributes its observations over time. Our default policy lets the agent freely switch among the 1×, 2×, and 4× speeds during exploration. We compare it against two restricted alternatives, one that forces the agent to observe the entire video at the slow 1× speed and another that forces the fast 4× speed throughout. As reported in Table VIII, both single-speed variants are clearly inferior to the learned mixed strategy. The slow-only setting obtains 43.6 on LVBench, as it spends most of its reasoning on irrelevant content before reaching the decisive evidence, whereas the fast-only setting yields the lowest LVBench score of 41.7, since it skims over fine-grained details that are necessary for accurate answering. The substantial gap between the two extremes and the default policy demonstrates that the ability to slow down on critical moments while skimming through uninformative regions is essential for efficient long-video exploration.

Core Mechanism. Finally, we ablate the two core mechanisms that support multi-turn exploration, namely the textual Notes that carry information across turns and the rewatch action that revisits a previous chunk for a closer look. Removing the Notes causes the largest degradation, dropping MLVU from 51.0 to 42.0, because the agent can no longer accumulate query-relevant observations across turns and is forced to reason over each turn in isolation, which is especially harmful for long videos that require integrating evidence from distant segments. Removing the rewatch action also lowers accuracy, for example from 45.1 to 42.5 on LVBench, since the agent loses the ability to return to a fast-forwarded chunk and recover the fine-grained details it has missed. These results indicate that the Notes provide cross-turn information aggregation while the rewatch action enables on-demand recovery of overlooked evidence. Together, the above ablations verify that every component of our adaptive exploration design contributes to the final performance of VideoScout.

## D. Efficiency Analysis

To investigate the inference efficiency of VideoScout, we measure the average inference time per sample (AIT) together with the accuracy on LVBench, MLVU, MINERVA, and WorldSense. All measurements are conducted on a single NVIDIA H800 GPU with a batch size of one, so that each sample is processed independently. The results are reported in Table IX.

As shown in the table, VideoScout achieves the highest accuracy on all four benchmarks, and its inference time adapts substantially to the nature of each benchmark, ranging from 8.1 seconds on WorldSense to 35.1 seconds on LVBench. On WorldSense, which mainly requires short-range comprehension of everyday videos, VideoScout answers in only 8.1 seconds, which is even faster than the 8.9 seconds of VideoTemp-o3-7B and the 9.2 seconds of LongVT-7B, while still reaching the best accuracy of 40.8. On LVBench, MLVU, and MINERVA, which involve long videos and demand holistic understanding and multi-step reasoning, it instead invests more inference time, reaching 35.1, 22.3, and 33.6 seconds respectively, and this additional computation is rewarded with the highest accuracy of 45.1, 51.0, and 35.4 on the three benchmarks. By comparison, LongVT-7B and VideoTemp-o3- 7B maintain a nearly fixed inference time, staying around 10.7 to 11.0 seconds for LongVT-7B and 18.7 to 21.7 seconds for VideoTemp-o3-7B across the three long-video benchmarks, since they first sample a fixed set of frames and then reason over them and can only adjust the amount of visual input through subsequent window zooming, so the observation cost is difficult to adapt to the difficulty of the query.

This behavior directly reflects the adaptive reasoning pacing of VideoScout. Rather than spending a fixed amount of observation on every video, the agent decides how long to explore according to the video length and the amount of evidence the query requires. It commits early when a short clip already provides sufficient evidence, as on WorldSense, and keeps exploring and rewatching when the answer depends on longrange or fine-grained cues, as on LVBench and MINERVA. As a result, VideoScout stays efficient on simple cases while devoting more computation to hard ones, and the higher cost it pays on difficult long videos is consistently converted into higher accuracy, instead of being a uniform and often unnecessary cost imposed on all inputs.

## E. Qualitative Study

To provide an intuitive understanding of how VideoScout explores a long video, we present a qualitative example from the LVBench dataset, as shown in Fig. 7. The question asks for the first warning on a warning sign in the volcano Skali, which requires the agent to first locate the sign within a long video and then read its fine-grained text. In the early turns, VideoScout finds that the opening chunks contain only a person talking and some scenery unrelated to the question, so it speeds up to 4× and quickly skims through these segments to gather a coarse overview while recording the visited content in its memory. After fast-forwarding to a lava-field area, it notices a warning sign in chunk 29 and reads the text as “Do not walk on the lava”. Instead of answering immediately, the agent reflects with a “Wait”: it realizes that it is still at 4× speed, that the sign is partially cut off, and that the text may not be clear enough to determine the first warning. It therefore actively slows down and rewatches chunk 29 at 1× speed. With the higher-resolution observation, VideoScout recognizes the original line “Ekki ganga a hrauninu” together with a “Vard! Attention!” header, rules out the distractor option A, and finally commits to the correct answer B.

![](images/e05cb3c2334ca1115f131a4e5d05998982ffe066d5a87dc75f6aaa06d88087dd.jpg)  
Fig. 7. A qualitative example of VideoScout on the LVBench dataset. Given a question about a long video, the agent reasons turn by turn: it adjusts the viewing speed, rewatches an earlier chunk when needed, updates its textual memory, and finally commits to an answer.

This example illustrates the learned “observe–reflect– adjust” loop of VideoScout. Rather than passively scanning at a fixed pace, the agent actively questions the sufficiency of its current observation and reflects with a “Wait” when the evidence is uncertain, then adjusts its viewing speed and rewatches the critical segment to obtain reliable evidence. Such reflective behavior allows VideoScout to direct its limited observations precisely toward the decisive moment and answer accurately, which is difficult to achieve with a fixed observation schedule.

## F. Further Analysis

We first analyze how accuracy varies with video length. We pool the samples from all nine benchmarks, group them into four duration buckets according to video length, and report in Fig. 8 the accuracy of VideoScout and two competitive agentic baselines within each bucket. VideoScout achieves the highest accuracy in every duration bucket, showing that its advantage holds consistently across all time scales rather than being confined to a particular video length. More notably, the three methods exhibit opposite trends as videos get longer. Both baselines reach their highest accuracy on the shortest videos and drop below that level once longer videos are involved, so their performance on videos longer than thirty minutes remains lower than on videos under three minutes. VideoScout instead does not degrade on long videos and even attains its best accuracy on the longest ones, where its score on videos longer than thirty minutes surpasses that on the shortest videos. This contrast indicates that coarse-to-fine zooming methods lose accuracy as the video grows longer and the decisive evidence becomes sparser, whereas VideoScout uses its adaptive reasoning pacing to spend its exploration effort on locating reliable evidence, so that its accuracy remains stable across video lengths.

![](images/1ef2ffee3e1dabbbebb2d4e8cac2af0ab28671a07610cd47e64d7bdc2dd912c0.jpg)  
Fig. 8. Accuracy across video durations, averaged over all nine benchmarks pooled together. Each sample is assigned to a duration bucket according to its video length, and the accuracy within each bucket is reported for VideoScout and two agentic baselines.

To understand the behavior behind this advantage, and to verify that it is acquired through training rather than provided by the agentic framework itself, we further analyze how VideoScout adjusts its viewing pace to videos of different lengths. We group the samples from all nine benchmarks into the same four duration buckets and report in Fig. 9 the proportion of each viewing action together with the average number of reasoning turns per sample, for both VideoScout and the untrained agentic variant Qwen2.5-VL-7B<sup>‡</sup>.

For VideoScout, the viewing-speed distribution shifts clearly with video length. On short videos, the agent watches mostly at the slow 1× and 2× speeds and rarely uses the fast 4× speed. As the video grows longer, the share of 1× steadily decreases while that of 4× increases, so on the longest videos a large portion of the actions are issued at the fastest speed. This trend shows that VideoScout watches short videos carefully but fast-forwards more on long videos to reach informative regions, rather than inspecting every segment of a long video at full density. The average number of reasoning turns grows in parallel, rising steadily from short to long videos, which indicates that the agent takes more exploration steps when the video is longer and the evidence is harder to locate. The rewatch action appears rarely on short videos and rises to a higher proportion on medium and long videos, showing that VideoScout invokes a closer second look mainly when fast-forwarding through longer videos may have missed finegrained evidence.

![](images/59958fbaded270e640c1bbe75950fd5742d5f0068459543e37a1bd8a47643d61.jpg)  
Fig. 9. Exploration behavior across video durations, aggregated over all nine benchmarks. The stacked bars show the proportion of each viewing action (1×, 2×, 4×, and rewatch), and the lines report the average number of reasoning turns per sample. Solid bars and the solid line denote VideoScout, while hatched bars and the dashed line denote the untrained agentic variant Qwen2.5-VL-7B<sup>‡</sup>.

The untrained variant behaves very differently. It stays almost entirely at the 1× speed across all duration buckets and barely uses fast-forwarding or rewatching, so its viewing pace does not adapt to video length at all. Its average number of reasoning turns is also consistently lower than that of VideoScout and grows much more slowly, indicating that it tends to commit to an answer after only a few turns instead of continuing to explore. This explains why simply placing the base model in the same multi-turn framework is not enough and even hurts performance: without our two-stage training, the agent does not learn to skim long videos or to gather sufficient evidence before answering. Taken together, these statistics confirm that VideoScout allocates its observations and reasoning turns according to video length rather than following a fixed schedule, and that this adaptive behavior is a direct result of training.

## G. Limitations

While VideoScout achieves strong overall performance, a few aspects remain to be improved. First, the agent carries information across turns through a textual memory that is rewritten at each update, which keeps the context compact but may also let an inaccurate observation from an early turn influence later reasoning on very long trajectories. Second, the rewatch action revisits a chunk within the most recent fastforward window, so its scope for returning to much earlier segments is limited. As shown in Fig. 10, when the agent infers from its notes that the decisive evidence lies near the beginning of a long video, it is unable to rewatch those early chunks from a much later position and therefore cannot verify the evidence it needs. Third, VideoScout takes only visual input, so questions that depend on audio cues such as speech or ambient sound are beyond its present scope.

## Model reasoning Model reasoning

Video <think>The user wants to know how the first goal of the match was scored.Looking at the current notes:- 18:46: Germany scores (0-1). This is the first goal.- 21:28: Spain scores (1-1)...Let's re-examine the video around 18:46 to see the goal. Wait, the current chunk is 35 (1530s-1620s), which is around 25:30 to 27:00. I need to go back to the beginning of the video or recall the first goal...I need to identify the type of goal (Header, Gate-crashing, Long shot, Close range)...Let's rewatch the first goal if possible, or if I can't, I'll have to rely on the notes and my memory of the first few minutes...Wait, I can't \"rewatch\" 1-10. I must have missed it or the system doesn't allow it in this format...Wait, I should be patient. I'm at chunk 35. The first goal was much earlier...I'll rewatch the first few chunks possible? No, I can't </think>

![](images/a7ad9c65bf4e4ca61a78fb447ecd1c30e735c75e66d668926cf800b16adf9d59.jpg)  
Fig. 10. A failure case from LVBench caused by the limited rewatch scope. The agent wants to return to early chunks for verification but cannot reach them from its current position.

## V. CONCLUSION AND FUTURE WORK

In this paper, we formulated long video question answering under bounded visual context as a Sequential Evidence Acquisition (SEA) problem, in which an agent reads the video turn by turn along the temporal axis like a human reading a long passage. We presented VideoScout, a trained multiturn reasoning agent that instantiates this paradigm through active exploration with adaptive reasoning pacing. To train this behavior, we constructed VideoScout-66K and adopted a twostage pipeline that combines cold-start supervised fine-tuning with trajectory-level reinforcement learning under a composite reward over accuracy, format, and answer timing. Extensive experiments on long video understanding and reasoning benchmarks as well as cross-domain benchmarks show that VideoScout attains strong performance under a bounded perturn visual context window while keeping inference efficient.

Building on these observations, we plan to extend VideoScout along three directions. First, we will incorporate audio perception so that the agent jointly reasons over vision and sound, enabling more complete omni-modal understanding of scenarios where the answer depends on what is heard as well as what is seen. Second, we will generalize the rewatch action into long-range revisiting, allowing the agent to return to any previously observed chunk on demand, so that evidence discovered late in the exploration can be reconciled with earlier segments. Third, we will equip the agent with a richer set of tools, such as object grounding, optical character recognition, and external knowledge retrieval, to extract fine-grained evidence and reason over information that is not directly observable from raw frames. We believe that advancing agentic exploration with adaptive reasoning pacing, by broadening its modality coverage and enriching its tool repertoire, offers a promising path toward general and efficient long video understanding.

## REFERENCES

[1] J. Chen, K. Li, Q. Deng, K. Li, and P. S. Yu, “Distributed deep learning model for intelligent video surveillance systems with edge computing,” IEEE Trans. Ind. Informat., pp. 1–1, 2019.

[2] V. Tsakanikas and T. Dagiuklas, “Video surveillance systems-current status and future trends,” Comput. Elect. Eng., vol. 70, pp. 736–753, 2018.

[3] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev, and J. Sivic, “Howto100m: Learning a text-video embedding by watching hundred million narrated video clips,” in Proc. IEEE Int. Conf. Comput. Vis., 2019, pp. 2630–2640.

[4] Y. Tang, D. Ding, Y. Rao, Y. Zheng, D. Zhang, L. Zhao, J. Lu, and J. Zhou, “Coin: A large-scale dataset for comprehensive instructional video analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 1207–1216.

[5] Y. Zhong, L. Yu, Y. Bai, S. Li, X. Yan, and Y. Li, “Learning procedureaware video representation from instructional videos and their narrations,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 14 825–14 835.

[6] N. Shvetsova, A. Kukleva, X. Hong, C. Rupprecht, B. Schiele, and H. Kuehne, “Howtocaption: Prompting llms to transform video annotations at scale,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 1–18.

[7] Q. Huang, Y. Xiong, A. Rao, J. Wang, and D. Lin, “Movienet: A holistic dataset for movie understanding,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 709–727.

[8] E. Song, W. Chai, T. Ye, J.-N. Hwang, X. Li, and G. Wang, “Moviechat+: Question-aware sparse memory for long video question answering,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 374–389, 2026.

[9] B. He, H. Li, Y. K. Jang, M. Jia, X. Cao, A. Shah, A. Shrivastava, and S.-N. Lim, “Ma-lmm: Memory-augmented large multimodal model for long-term video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 13 504–13 514.

[10] S. Bai, K. qin Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[11] Y. Wang, X. Li, Z. Yan, Y. He, J. Yu, X. Zeng, C. Wang, C. Ma, H. Huang, J. Gao et al., “Internvideo2. 5: Empowering video mllms with long and rich context modeling,” arXiv preprint arXiv:2501.12386, 2025.

[12] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, J. Wu, X. Zhang, B. Wang, and X. Yue, “Video-r1: Reinforcing video reasoning in mllms,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 99 114– 99 137, 2026.

[13] P. Zhang, K. Zhang, B. Li, G. Zeng, J. Yang, Y. Zhang, Z. Wang, H. Tan, C. Li, and Z. Liu, “Long context transfer from language to vision,” arXiv preprint arXiv:2406.16852, 2024.

[14] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang, “Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning,” arXiv preprint arXiv:2504.06958, 2025.

[15] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.

[16] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Llavavideo: Video instruction tuning with synthetic data,” arXiv preprint arXiv:2410.02713, 2024.

[17] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[18] B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” in Proc. Conf. Empirical Methods Natural Lang. Process., 2024, pp. 5971–5984.

[19] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” arXiv preprint arXiv:2305.06355, 2023.

[20] K. Ataallah, X. Shen, E. Abdelrahman, E. Sleiman, D. Zhu, J. Ding, and M. Elhoseiny, “Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens,” arXiv preprint arXiv:2404.03413, 2024.

[21] Y. Wang, H. Zhang, Y. Yue, S. Song, C. Deng, J. Feng, and G. Huang, “Uni-adafocus: Spatial-temporal dynamic computation for video recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 3, pp. 1782–1799, 2025.

[22] T. Wu, M. Cao, Z. Gao, G. Wu, and L. Wang, “Stmixer: A one-stage sparse action detector,” in IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 10, 2024, pp. 6842–6857.

[23] H. Fei, S. Wu, M. Zhang, M. Zhang, T.-S. Chua, and S. Yan, “Enhancing video-language representations with structural spatio-temporal

alignment,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 7701–7719, 2024.

[24] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, “Adaptive keyframe sampling for long video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 29 118–29 128.

[25] H. Liang, J. Li, T. Bai, X. Huang, L. Sun, Z. Wang, C. He, B. Cui, C. Chen, and W. Zhang, “Keyvideollm: Towards large-scale video keyframe selection,” arXiv preprint arXiv:2407.03104, 2024.

[26] S. Yu, C. Jin, H. Wang, Z. Chen, S. Jin, Z. Zuo, X. Xu, Z. Sun, B. Zhang, J. Wu et al., “Frame-voyager: Learning to query frames for video large language models,” in Proc. Int. Conf. Learn. Representations, 2025, pp. 84 154–84 179.

[27] Z. Wang, S. Yu, E. Stengel-Eskin, J. Yoon, F. Cheng, G. Bertasius, and M. Bansal, “Videotree: Adaptive tree-based video representation for llm reasoning on long videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 3272–3283.

[28] S. Yu, J. Cho, P. Yadav, and M. Bansal, “Self-chained image-language model for video localization and question answering,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 36, pp. 76 749–76 771, 2023.

[29] K. Wei, X. Liu, J. Zhang, Z. Wang, R. Liu, Y. Yang, X. Xiao, X. Sun, H. Zeng, C. Pan et al., “Cfvbench: A comprehensive video benchmark for fine-grained multimodal retrieval-augmented generation,” in Proc. ACM Web Conf., 2026, pp. 2501–2512.

[30] X. Liu, Y. Zhu, X. Zhao, X. Wang, Y. Li, X. Li, H. Cao, X. Sun, S. Zhang, X. Yang et al., “Moment-video: Diagnosing temporal fidelity of video mllms on momentary visual events,” arXiv preprint arXiv:2606.02522, 2026.

[31] J. Hu, Z. Cheng, C. Si, W. Li, and S. Gong, “Cos: Chain-of-shot prompting for long video understanding,” arXiv preprint arXiv:2502.06428, 2025.

[32] Z. Yang, S. Wang, K. Zhang, K. Wu, S. Leng, Y. Zhang, B. Li, C. Qin, S. Lu, X. Li et al., “Longvt: Incentivizing” thinking with long videos” via native tool calling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2026, pp. 33 816–33 826.

[33] W. Liu, Y. Wang, S. Ma, M. Liu, Q. Su, T. Zhang, H. Fan, C. Liu, K. Jiang, J. Chen et al., “Videotemp-o3: Harmonizing temporal grounding and video understanding in agentic thinking-with-videos,” in Proc. Int. Conf. Mach. Learn., 2026.

[34] Z. Ma, C. Gou, H. Shi, B. Sun, S. Li, H. Rezatofighi, and J. Cai, “Drvideo: Document retrieval based long video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 18 936– 18 946.

[35] Z. Yang, D. Chen, X. Yu, M. Shen, and C. Gan, “Vca: Video curious agent for long video understanding,” in Proc. IEEE Int. Conf. Comput. Vis., 2025, pp. 20 168–20 179.

[36] E. Song, W. Chai, G. Wang, Y. Zhang, H. Zhou, F. Wu, H. Chi, X. Guo, T. Ye, Y. Zhang et al., “Moviechat: From dense token to sparse memory for long video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 18 221–18 232.

[37] Y. Shu, Z. Liu, P. Zhang, M. Qin, J. Zhou, Z. Liang, T. Huang, and B. Zhao, “Video-xl: Extra-long vision language model for hour-scale video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 26 160–26 169.

[38] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu et al., “Dapo: An open-source llm reinforcement learning system at scale,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 113 222–113 244, 2026.

[39] Y. Shi, J. Liu, Y. Guan, Z. Wu, Y. Zhang, Z. Wang, W. Lin, J. Hua, Z. Wang, X. Chen et al., “Mavors: Multi-granularity video representation for multimodal large language model,” in Proc. 33th ACM Int. Conf. Multimedia, 2025, pp. 10 994–11 003.

[40] Z. Zhao, Y. Huo, T. Yue, L. Guo, H. Lu, B. Wang, W. Chen, and J. Liu, “Efficient motion-aware video mllm,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 24 159–24 168.

[41] S. Azad, V. Vineet, and Y. S. Rawat, “Hierarq: Task-aware hierarchical q-former for enhanced video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 8545–8556.

[42] K. Wang and M. Lin, “Test-time temporal sampling for efficient mllm video understanding,” arXiv preprint arXiv:2511.17945, 2025.

[43] J. Xu, C. Lan, W. Xie, X. Chen, and Y. Lu, “Long video understanding with learnable retrieval in video-language models,” IEEE Trans. Multimedia, pp. 1–8, 2026.

[44] W. Tan, X. Yu, J. Li, Y. Chen, J. Ju, Z. Luo, R. Song, and J. Luan, “Msjoe: Jointly evolving mllm and sampler for efficient long-form video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2026, pp. 19 487–19 496.

[45] C. Tang, Z. Han, H. Sun, S. Zhou, X. Zhang, X. Wei, Y. Yuan, H. Zhang, J. Xu, and H. Sun, “Tspo: Temporal sampling policy optimization for long-form video language understanding,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 11, 2026, pp. 9368–9376.

[46] Z. Zhang, Z. Pang, S. Chen, X. Hao, V. Bhat, and Y.-X. Wang, “One token per highly selective frame: Towards extreme compression for long video understanding,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 57 332–57 358, 2026.

[47] J. Hu, D. Guo, K. Li, Z. Si, X. Yang, X. Chang, and M. Wang, “Unified static and dynamic network: efficient temporal filtering for video grounding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 8, pp. 6445–6462, 2025.

[48] J. Chang, J. Lee, and J. Choi, “Ca<sup>2</sup>st: Cross-attention in audio, space, and time for holistic video recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 3, pp. 2803–2819, 2026.

[49] B. Zhao, H. Li, X. Lu, and X. Li, “Reconstructive sequence-graph network for video summarization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 5, pp. 2793–2801, 2022.

[50] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, and X. Jin, “Flash-vstream: Efficient real-time understanding for long video streams,” in Proc. IEEE Int. Conf. Comput. Vis., 2025, pp. 21 059–21 069.

[51] X. Shen, Y. Xiong, C. Zhao, L. Wu, J. Chen, C. Zhu, Z. Liu, F. Xiao, B. Varadarajan, F. Bordes et al., “Longvu: Spatiotemporal adaptive compression for long video-language understanding,” arXiv preprint arXiv:2410.17434, 2024.

[52] Y. Fan, X. Ma, R. Wu, Y. Du, J. Li, Z. Gao, and Q. Li, “Videoagent: A memory-augmented multimodal agent for video understanding,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 75–92.

[53] C. Shang, A. You, S. Subramanian, T. Darrell, and R. Herzig, “Traveler: A modular multi-lmm agent framework for video question-answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[54] X. Zhang, Z. Jia, Z. Guo, J. Li, B. Li, H. Li, and Y. Lu, “Deep video discovery: Agentic search with tool use for long-form video understanding,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 89 863–89 895, 2026.

[55] L. Zhang, T. Zhao, H. Ying, Y. Ma, and K. Lee, “Omagent: A multimodal agent framework for complex video understanding with task divide-and-conquer,” in Proc. Conf. Empirical Methods Natural Lang. Process., 2024, pp. 10 031–10 045.

[56] C. Zhang, T. Lu, M. M. Islam, Z. Wang, S. Yu, M. Bansal, and G. Bertasius, “A simple llm framework for long-range video questionanswering,” in Proc. Conf. Empirical Methods Natural Lang. Process., 2024, pp. 21 715–21 737.

[57] J. Min, S. Buch, A. Nagrani, M. Cho, and C. Schmid, “Morevqa: Exploring modular reasoning models for video question answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 13 235–13 245.

[58] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 35, pp. 27 730–27 744, 2022.

[59] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 36, pp. 53 728–53 741, 2023.

[60] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela, “Kto: Model alignment as prospect theoretic optimization,” in Proc. Int. Conf. Mach. Learn., 2024.

[61] Y. Meng, M. Xia, and D. Chen, “Simpo: Simple preference optimization with a reference-free reward,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 37, pp. 124 198–124 235, 2024.

[62] W. Yuan, R. Y. Pang, K. Cho, X. Li, S. Sukhbaatar, J. Xu, and J. Weston, “Self-rewarding language models,” in Proc. Int. Conf. Mach. Learn., 2024.

[63] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, X. Tang, Y. Hu, and S. Lin, “Vision-r1: Incentivizing reasoning capability in multimodal large language models,” arXiv preprint arXiv:2503.06749, 2025.

[64] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang, “Visual-rft: Visual reinforcement fine-tuning,” in Proc. IEEE Int. Conf. Comput. Vis., 2025, pp. 2034–2044.

[65] T. Yu, Y. Yao, H. Zhang, T. He, Y. Han, G. Cui, J. Hu, Z. Liu, H.-T. Zheng, M. Sun et al., “Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 13 807– 13 816.

[66] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical

reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[67] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature, vol. 645, no. 8081, pp. 633–638, 2025.

[68] Q. Wang, Y. Yu, Y. Yuan, R. Mao, and T. Zhou, “Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 4350–4376, 2026.

[69] J. Park, J. Na, J. Kim, and H. J. Kim, “Deepvideo-r1: Video reinforcement fine-tuning via difficulty-aware regressive grpo,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 138 605–138 632, 2026.

[70] S. Chen, B. Chen, Y. Luo, C. Yu, Y. Ouyang, L. Cheng, C. Zhuo, Z. Li, and Y. Wang, “Vragent-r1: Boosting video recommendation with mllmbased agents via reinforcement learning,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 3, 2026, pp. 1739–1747.

[71] J. Meng, X. Li, H. Wang, Y. Tan, T. Zhang, L. Kong, Y. Tong, A. Wang, Z. Teng, Y. Wang et al., “Open-o3 video: Grounded video reasoning with explicit spatio-temporal evidence,” in Proc. Int. Conf. Mach. Learn., 2025.

[72] H. Rasheed, M. Zumri, M. Maaz, M.-H. Yang, F. S. Khan, and S. Khan, “Video-com: Interactive video reasoning via chain of manipulations,” arXiv preprint arXiv:2511.23477, 2025.

[73] Y. Xie, T. Chen, Z. Ge, and L. Ni, “Video-mtr: Reinforced multi-turn reasoning for long video understanding,” arXiv preprint arXiv:2508.20478, 2025.

[74] Y. Liu, K. Q. Lin, C. W. Chen, and M. Z. Shou, “Videomind: A chainof-lora agent for temporal-grounded video reasoning,” in Proc. Int. Conf. Adv. Neural Inf. Process. Syst., 2025.

[75] J. Lin, J. Wu, X. Sun, Z. Wang, J. Liu, Y. Su, X. Yu, H. Chen, J. Luo, Z. Liu et al., “Unleashing hour-scale video training for long videolanguage understanding,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 17 523–17 552, 2026.

[76] D. Ma, H. Yuan, X. Wang, Q. Zang, T. Liu, X. He, Y. Wei, J. Guo, N. Jiahui, Z. Yang et al., “Scalelong: A multi-timescale benchmark for long video understanding,” arXiv preprint arXiv:2505.23922, 2025.

[77] Y. Chen, W. Huang, B. Shi, Q. Hu, H. Ye, L. Zhu, Z. Liu, P. Molchanov, J. Kautz, X. Qi et al., “Scaling rl to long videos,” Proc. Int. Conf. Adv. Neural Inf. Process. Syst., vol. 38, pp. 172 842–172 870, 2026.

[78] K. Ataallah, X. Shen, E. Abdelrahman, E. Sleiman, M. Zhuge, J. Ding, D. Zhu, J. Schmidhuber, and M. Elhoseiny, “Goldfish: Vision-language understanding of arbitrarily long videos,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 251–267.

[79] Google DeepMind, “Gemini 3 Flash model card,” https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-Flash-Model-Card.pdf, Dec 2025, accessed: 2026-06-22.

[80] M. M. Krell, M. Kosec, S. P. Perez, and A. Fitzgibbon, “Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance,” arXiv preprint arXiv:2107.02027, 2021.

[81] J. Zhou, Y. Shu, B. Zhao, B. Wu, Z. Liang, S. Xiao, M. Qin, X. Yang, Y. Xiong, B. Zhang et al., “Mlvu: Benchmarking multi-task long video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 13 691–13 701.

[82] W. Wang, Z. He, W. Hong, Y. Cheng, X. Zhang, J. Qi, M. Ding, X. Gu, S. Huang, B. Xu et al., “Lvbench: An extreme long video understanding benchmark,” in Proc. IEEE Int. Conf. Comput. Vis., 2025, pp. 22 958– 22 967.

[83] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 24 108– 24 118.

[84] A. Nagrani, S. Menon, A. Iscen, S. Buch, R. Mehran, N. Jha, A. Hauth, Y. Zhu, C. Vondrick, M. Sirotenko et al., “Minerva: Evaluating complex video reasoning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 23 968–23 978.

[85] K. Hu, P. Wu, F. Pu, W. Xiao, Y. Zhang, X. Yue, B. Li, and Z. Liu, “Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos,” arXiv preprint arXiv:2501.13826, 2025.

[86] A. Deng, T. Yang, S. Yu, L. Spencer, M. Bansal, C. Chen, S. Yeung-Levy, and X. Wang, “Scivideobench: Benchmarking scientific video reasoning in large multimodal models,” arXiv preprint arXiv:2510.08559, 2025.

[87] J. Hong, S. Yan, J. Cai, X. Jiang, Y. Hu, and W. Xie, “Worldsense: Evaluating real-world omnimodal understanding for multimodal llms,” arXiv preprint arXiv:2502.04326, 2025.

[88] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 10 632–10 643.

[89] Y. Zhao, H. Zhang, L. Xie, T. Hu, G. Gan, Y. Long, Z. Hu, W. Chen, C. Li, Z. Xu et al., “Mmvu: Measuring expert-level multi-discipline video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 8475–8489.

[90] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.

[91] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.

[92] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.