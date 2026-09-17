# SVMemAgent: A Streaming Video Memory Agent for Query-Agnostic Online Frame Selection

Dohwan Ko<sup>1,2†</sup>, Ji Soo Lee<sup>3</sup>, Pierce Chuang<sup>2</sup>, Debojeet Chatterjee<sup>2</sup>, Ashish Shenoy<sup>2</sup>, Yichao Lu<sup>2</sup>, Seungwhan Moon<sup>2</sup>, Xin Luna Dong<sup>2</sup>, Vikas Bhardwaj<sup>2</sup>, and Hyunwoo J. Kim<sup>3∗</sup>

<sup>1</sup> Korea University, Seoul, Republic of Korea <sup>2</sup> Meta AI 3 KAIST, Daejeon, Republic of Korea

Abstract. Most keyframe selection studies focus on ofline settings, assuming access to the full video and query in advance. In contrast, real-world streaming scenarios require online frame selection under unknown video duration, without access to either the query or future frames during selection. To address this, we introduce Streaming Video Memory (SVMem), a compact and representative memory of previously observed content, updated continuously as the video stream unfolds. Building on this setting, we propose the Streaming Video Memory Agent (SVMemAgent), which dynamically maintains a memory by deciding at each timestep whether to replace an existing memory frame with the incoming frame or discard it. SVMemAgent is trained using Group Relative Policy Optimization (GRPO) with task-driven rewards derived from diverse question-answer pairs, implicitly exposing the policy to a distribution of queries during training so that SVMem retains generally informative frames at inference, when queries are unavailable. Experiments on both online and ofline video benchmarks show that SVMemAgent consistently outperforms online frame selection baselines and achieves competitive performance with ofline methods that assume access to the full video and query. Through task-driven rewards, SVMemAgent learns an emergent keyframe selection policy that prefers frames containing textual information, which may benefit downstream VideoQA tasks.

Keywords: Online Video Understanding · Keyframe Selection · Memory Agent · Group Relative Policy Optimization

## 1 Introduction

Keyframe selection in long video understanding enables Video Large Language Models (VideoLLMs) [11, 13, 14, 26, 28] to process long videos within limited context windows. Existing approaches mainly study this problem in ofline settings, where the complete video and the corresponding query are available in advance. Under this assumption, frame selection can be naturally formulated as a simple retrieval problem, which selects query-relevant frames from the full video (Fig. 1a). However, this ofline setting overlooks real-world streaming scenarios, such as wearable devices and embodied AI systems, where frames arrive sequentially, the video duration is unknown, and the query may be unavailable at the time of frame selection. As a result, in streaming settings, frames must be selected sequentially without access to future visual content or the user query.

![](images/f96a374be64b810ef7ed26587edc611bef85b81f01f42e8b258dc2df7b4b6ebd.jpg)  
Fig. 1: (a) Ofline frame selection methods simply retrieve query-relevant frames with access to the full video, whereas (b) SVMemAgent sequentially updates SVMem without access to the query or future frames, deciding whether to discard the current frame or use it to replace an existing memory slot.

Sequential frame selection in online streaming scenarios poses two key challenges. First, since the total stream duration is unknown during streaming, naive fixed-interval frame sampling cannot guarantee that the memory will remain below capacity before the query arrives. Second, as frame selection occurs prior to query arrival, decisions must rely solely on past and incoming frames, without access to future frames or query information. These constraints make existing ofline frame selection approaches, which merely identify query-relevant frames from the entire video, unsuitable for direct application to online settings.

To address these challenges, we first formulate a Streaming Video Memory (SVMem), which maintains a subset of frames without storing the entire video stream. As in Fig. 1b, SVMem is sequentially updated by SVMemAgent, which decides whether to replace an existing memory frame with the incoming frame or discard it, without access to query information or future frames. Once streaming ends and a query is provided, the downstream VideoLLM processes the query using the frames stored in the final state of SVMem. We train SVMemAgent using Group Relative Policy Optimization (GRPO) [20] with task-driven rewards. This enables SVMemAgent to learn a general prior over informative frames from the training query distribution through reward signals, despite having no access to the inference-time query during frame selection. Furthermore, we introduce Diversity-Aware Advantage Discounting (DAAD), which penalizes policies exhibiting low reward variance despite high state diversity. This mitigates overfitting to uninformative reward signals that are largely insensitive to frame-selection decisions, such as language-biased queries or static scenes. Under the strict online streaming setting, SVMemAgent consistently outperforms online frame selection baselines and even achieves competitive performance with ofline methods that assume access to the full video and query. We observe that SVMemAgent implicitly learns to prioritize informative and non-redundant frames, such as text-containing frames, for future query answering.

To sum up, our contributions are threefold: (1) We introduce SVMem, a memory for a strict online streaming setting that requires frame selection under unknown duration, limited memory capacity, and no access to query information. (2) We propose SVMemAgent, which maintains a compact and representative memory by discarding or replacing frames as the stream unfolds. We train the agent using GRPO with task-driven rewards and DAAD regularization. (3) Extensive experiments on both online and ofline video benchmarks demonstrate that SVMemAgent consistently outperforms existing frame selection baselines and achieves performance competitive with ofline methods that assume access to the full video and query.

## 2 Related Works

Online video understanding. Recent studies have begun applying VideoLLMs to online streaming settings. One line of work [2,7,10,17,25] adopts a data-centric approach, introducing instruction-following datasets to train VideoLLMs for generating timely responses to streaming videos. For example, VideoLLM-online [2] proposes a narration streaming dataset that encourages proactive description generation based on sequences of user actions. Another line of work [4,12,15,18, 24] focuses on architectural modifications to enable streaming video processing. StreamChat [24], for instance, incorporates both long- and short-term memory modules to eficiently handle queries in long videos. However, most existing approaches rely on training-free heuristics for memory maintenance, whereas our memory agent directly learns a query-agnostic frame selection policy through reinforcement learning.

Keyframe selection in video understanding. Keyframe selection identifies a compact subset of informative frames, enabling VideoLLMs to process long videos under limited context windows. Existing approaches can be categorized according to the information available at selection time. Ofline methods [9,15,22] assume access to either the full video or the user query. For example, AKS [22] recursively partitions the video into a hierarchical structure and selects keyframes based on CLIP-based query-frame relevance. In online streaming settings [8,25,27], however, frames must be selected without prior knowledge of either the query or the total video duration. TimeChat-Online [25] removes redundant frames by detecting feature changes relative to the most recently retained frame, while MA-LMM [8] iteratively consolidates adjacent frame pairs within a fixed-capacity memory bank.

Reinforcement learning for visual reasoning. Inspired by the success of reinforcement learning (RL)-based post-training for LLMs, particularly Group Relative Policy Optimization (GRPO) [20], recent studies have extended RL with verifiable rewards to multi-modal models [5, 6, 14, 16]. Visual-RFT [16] applies GRPO with task-specific verifiable rewards to visual perception tasks, including detection and grounding, and demonstrates substantially greater data eficiency than supervised fine-tuning. In the video domain, Video-R1 [5] introduces a temporally aware variant of GRPO to incentivize temporal reasoning in VideoLLMs. However, these approaches primarily use RL to enhance the reasoning capability of the VideoLLM itself and assume full access to the input video, while we employ GRPO to train a lightweight external policy for online memory maintenance. Task-driven rewards derived from diverse question-answer pairs expose the policy to a broad query distribution during training, enabling it to learn a query-agnostic prior for retaining informative frames under strict online streaming constraints.

![](images/cbb299e3e55df527d1b32170013f3dfca64d4698bbb1f60d311de201df6027ad.jpg)  
Fig. 2: Illustration of SVMemAgent and GRPO training with task-driven rewards. SVMemAgent integrates (a) a semantic branch designed for content-aware memory updates and (b) a temporal branch for maintaining uniform temporal coverage. (c) During training, SVMemAgent is optimized using GRPO with task-driven rewards to learn a general prior over informative frames without requiring query information at inference time.

## 3 Method

SVMemAgent sequentially updates SVMem during video streaming. It first decides to discard or retain the current frame and, if retained, determines which memory slot to replace. We first describe the overall design of SVMem and SVMemAgent, followed by detailed training and inference pipelines.

## 3.1 SVMem

We consider a strict online streaming video understanding setting under two constraints: (1) video frames arrive sequentially at 1 FPS, while future frames and the total video duration remain inaccessible, and (2) the user query is unavailable during frame selection, resulting in a query-agnostic setting. Under these constraints, we introduce an SVMem M that stores at most K frames. At each timestep t, a new frame $v _ { t }$ arrives, and the system must decide whether to discard $v _ { t }$ or replace one of the existing K memory slots with it. When the query arrives at an arbitrary timestamp, the corresponding memory state $\mathcal { M } = \{ v _ { m _ { 1 } } , v _ { m _ { 2 } } , . . . , v _ { m _ { K } } \}$ and the user query are provided to a VideoLLM, such as InternVL3.5 [23] or Qwen3-VL [1], for answer generation.

## 3.2 SVMemAgent

To update SVMem sequentially during streaming, we introduce SVMemAgent. Since neither the query nor the total stream duration is available during memory updates, SVMemAgent jointly considers both what a frame depicts and when it occurs within the observed stream. Accordingly, SVMemAgent adopts a dualbranch architecture that models semantic understanding and temporal coverage, as in Fig. 2a and 2b. It adopts a two-stage action policy that first makes a discard decision and then, if necessary, performs a replacement decision.

Dual-branch decision module. In the semantic branch, the image encoder $f$ first extracts a visual feature $e _ { t } ~ = ~ f ( v _ { t } ) \in \mathbb { R } ^ { D }$ from each incoming frame $v _ { t } .$ , where D denotes the hidden dimension. Lightweight transformer layers then model semantic relationships among the current frame, the existing memory slots, and the global stream context. Specifically, the transformer processes $K + 2$ input tokens: (1) Frames in SVMem (K tokens)-visual features of frames currently stored in memory, $\{ e _ { m _ { k } } \} _ { k = 1 } ^ { K } , ~ ( 2 )$ Current frame (1 token)-feature of the incoming frame, $e _ { t } .$ , and (3) Stream context (1 token)-average feature of all previously observed frames, which serves as a summary of the video stream, $\begin{array} { r } { c _ { t } = \frac { 1 } { t } \sum _ { i = 0 } ^ { t - 1 } e _ { i } } \end{array}$ . Two prediction heads are applied to the transformer outputs to produce discard and replacement scores. The discard head uses the representation corresponding to the current frame token to generate a scalar discard score $d ^ { \mathrm { s e m } } \in \mathbb { R }$ . The replacement head uses the representations of the K memory slot tokens to produce replacement scores $r ^ { \mathrm { s e m } } \in \mathbb { R } ^ { K }$

The temporal branch uses only frame timestamps, without relying on semantic features, to encourage temporal coverage of the observed stream up to the current timestamp. It processes $K + 1$ input tokens corresponding to the $K$ memory frames and the current frame. Specifically, each memory slot is represented by its normalized timestamp $m _ { k } / t \in [ 0 , 1 )$ , while the current frame is assigned the fixed position $t / t = 1 . 0$ . These scalar timestamps are transformed into $D _ { - }$ dimensional embeddings using sinusoidal positional encodings followed by a linear projection layer. The resulting tokens are processed by shallow transformer layers. Discard and replacement heads are then applied to produce temporal discard and replacement scores, denoted by $d ^ { \mathrm { t e m p } }$ and $r ^ { \mathrm { t e m p } }$ , respectively.

The dual-branch outputs are combined as:

$$
d = w _ { d } \cdot d ^ { \mathrm { s e m } } + ( 1 - w _ { d } ) \cdot d ^ { \mathrm { t e m p } } \in \mathbb { R } ,\tag{1}
$$

$$
r = w _ { r } \cdot r ^ { \mathrm { s e m } } + ( 1 - w _ { r } ) \cdot r ^ { \mathrm { t e m p } } \in \mathbb { R } ^ { K } ,\tag{2}
$$

where $w _ { d } , w _ { r } \in [ 0 , 1 ]$ denote the balancing weights for discard and replacement decisions, respectively.

Two-stage action decision. At each timestep, SVMemAgent first decides the discard action based on $P _ { d } = \mathrm { s i g m o i d } ( d ) , i . e .$ , discard action $\mathbf { \Sigma } = \mathbf { \Sigma } ^ { \mathcal { ( } } \mathtt { d i s c a r d ^ { \flat } }$ if $P _ { d } > 0 . 5$ else ‘retain’. When the discard action is ‘retain’, the memory slot with the highest replacement probability under $P _ { r } ( k ) = \mathrm { s o f t m a x } ( r ) _ { k }$ is selected and replaced by the current frame, i.e., replacement action $= \mathrm { a r g m a x } _ { k } P _ { r } ( k )$ During GRPO training (Sec. 3.3), both actions are instead sampled stochastically from $P _ { d }$ and $P _ { r }$ to enable policy exploration.

## 3.3 Training Pipeline

SVMemAgent is trained in two stages: (1) cold-start training with pseudo-labels to initialize the dual-branch policy, and (2) GRPO, which refines the policy using task-driven rewards with Diversity-Aware Advantage Discounting (DAAD).

Stage 1: Cold-start training with pseudo-labels. Since ground-truth annotations for query-agnostic online frame selection are unavailable, we construct heuristic pseudo-labels separately for each decision branch. The pseudo-labels for the semantic branch are designed to preserve frames that are representative of the video stream while remaining non-redundant with respect to the current memory contents. To this end, we adopt a teacher visual encoder $g$ to compute relevance and novelty scores based on cosine similarity. The current frame is retained if it is suficiently representative of the video stream (relevance) and contributes information not already captured in memory (novelty). Specifically, the discard pseudo-label is assigned as 0 (retain) when the sum of the relevance and novelty scores exceeds a predefined threshold, and as 1 (discard) otherwise. If the current frame is retained $( i . e . ,$ , not discarded), the replacement pseudolabel is assigned to the index of the memory slot most similar to the current frame, thereby reducing redundancy in memory.

While the semantic branch pseudo-labels focus on preserving informative and non-redundant visual content, the temporal branch pseudo-labels promote balanced temporal coverage throughout the video stream. At each timestep t, the discard pseudo-label is assigned when the memory already provides suficiently uniform temporal coverage. Otherwise, we evaluate all possible replacement actions and assign the replacement pseudo-label to the candidate that maximizes temporal uniformity. Each branch is trained individually using its corresponding pseudo-labels. The discard head uses a BCE loss for discard/retain decisions, while the replacement head is trained with a CE loss over the K memory slots. Stage 2: GRPO with DAAD. While the cold-start training provides a stable policy initialization, pseudo-labels do not directly optimize downstream VideoQA performance. Therefore, we further optimize SVMemAgent using GRPO with task-driven rewards, as in Fig. 2c. Since SVMemAgent has no access to the query during frame selection at inference time, we train it to implicitly leverage the training query distribution through the task-driven reward signals. This enables the agent to retain frames that are broadly useful across diverse potential queries and thereby learn a query-agnostic prior over visual importance. In GRPO, actions are sampled from both the discard and replacement heads, and the corresponding policies are jointly optimized.

To evaluate the quality of frame selection decisions, we define task-driven rewards by measuring how efectively the selected frames support answering a user query. Specifically, for the i-th rollout, we feed the final memory state $\mathcal { M } _ { T } ^ { i }$ containing K frames, and the query $q ,$ into a frozen VideoLLM reward model. The reward is defined as the average per-token log-probability assigned to the ground-truth response y:

$$
R ^ { i } = \frac { 1 } { | y | } \sum _ { j = 1 } ^ { | y | } \log P _ { \mathrm { V i d e o L L M } } \left( y _ { j } \mid y _ { < j } , \mathcal { M } _ { T } ^ { i } , q \right) ,\tag{3}
$$

which corresponds to the negative of the average token-level cross-entropy loss. This objective directly aligns policy optimization with downstream VideoQA performance. Since SVMemAgent is designed for a query-agnostic streaming setting, optimizing against a single query-answer pair may lead to overfitting to query-specific visual cues. Therefore, we use four distinct queries per video to encourage broader generalization.

However, we observe that substantially diverse memory states can lead to nearly identical rewards, indicating that the reward signal is largely insensitive to frame selection decisions. This phenomenon can arise when a query is answerable from language priors alone (language-biased query) or when distinct frame subsets contain semantically equivalent information, particularly in static scenes. Regardless of the underlying cause, such rollout groups provide weak supervision for frame selection and inject noise into policy optimization, thereby hindering the agent from learning a meaningful selection policy.

To address this issue, we propose Diversity-Aware Advantage Discounting (DAAD). We first compute a state diversity metric as the mean pairwise Jaccard distance between final memory states:

$$
D _ { \mathrm { J a c } } = \frac { 2 } { G ( G - 1 ) } \sum _ { g < g ^ { \prime } } \left( 1 - \frac { | \mathcal { M } _ { T } ^ { ( g ) } \cap \mathcal { M } _ { T } ^ { ( g ^ { \prime } ) } | } { | \mathcal { M } _ { T } ^ { ( g ) } \cup \mathcal { M } _ { T } ^ { ( g ^ { \prime } ) } | } \right) ,\tag{4}
$$

where $G$ is the group size. Then, we define an advantage discounting score α that quantifies the ratio of memory diversity to reward variation:

$$
\alpha = \mathrm { s i g m o i d } \left( \lambda - \frac { D _ { \mathrm { J a c } } } { \mathrm { s t d } \left( \{ R ^ { j } \} _ { j = 1 } ^ { G } \right) } \right) ,\tag{5}
$$

where $\lambda$ is a predefined threshold. α adaptively rescales the normalized GRPO advantages:

$$
\hat { A } ^ { i } = \alpha \cdot \frac { R ^ { i } - \mathrm { m e a n } \left( \{ R ^ { j } \} _ { j = 1 } ^ { G } \right) } { \mathrm { s t d } \left( \{ R ^ { j } \} _ { j = 1 } ^ { G } \right) } .\tag{6}
$$

When sampled policies generate highly diverse memory states but exhibit only minor reward diferences, the optimization signal is likely dominated by noise rather than meaningful policy improvements. In such cases, α approaches 0, effectively suppressing unreliable policy gradients. Conversely, when reward variations consistently reflect diferences in memory quality, α approaches 1, thereby preserving the original GRPO advantages.

Table 1: Comparison of frame selection methods. Red rows indicate baselines evaluated under relaxed online or ofline settings, where the query, the video duration, or both are known in advance. Blue rows denote baselines evaluated under the strict online setting, where both the query and video duration are unknown.
<table><tr><td>Method</td><td>Unknown Query</td><td>Unknown Duration</td><td>Description</td></tr><tr><td>Uniform</td><td>ノ√</td><td>x</td><td>Evenly spaced frames across the full video</td></tr><tr><td>Random</td><td></td><td>x</td><td>Randomly sampled frames from the full video</td></tr><tr><td>Clustering</td><td>V</td><td>x</td><td>K-means over the full video; frames nearest to centroids</td></tr><tr><td>StreamChat [24]</td><td>×x</td><td>✓</td><td>Ebbinghaus memory + K-means compression + CLIP retrieval</td></tr><tr><td>AKS [22]</td><td></td><td>x</td><td>Query-aware recursive hierarchical keyframe splitting based on CLIP</td></tr><tr><td>M-LLM Selector [9]</td><td>x</td><td>x</td><td>A distilled CLIP model trained to predict query-frame relevance scores.</td></tr><tr><td>FIFO</td><td></td><td></td><td>First-in-first-out memory update with fixed-interval sampling</td></tr><tr><td>Reservoir</td><td></td><td></td><td>Vitter&#x27;s Algorithm R with decreasing replace probability</td></tr><tr><td>TimeChat-Online [25]</td><td></td><td></td><td>Feature change detection against last kept frame</td></tr><tr><td>Flash-VStream [27]</td><td></td><td></td><td>PCA + temporally-ordered sequential K-means clustering</td></tr><tr><td>MA-LMM [8]</td><td></td><td></td><td>Adjacent-pair consolidation in fixed memory bank</td></tr><tr><td>SVMemAgent (Ours)</td><td></td><td>V</td><td>A streaming video memory agent with sequential discard-and-replacement policy decisions</td></tr></table>

## 3.4 Inference Pipeline

Unlike training, which relies on stochastic action sampling for policy exploration, inference uses a deterministic greedy policy to construct the memory state. Specifically, the discard decision is computed as $\mathbf { 1 } _ { [ P _ { d } > 0 . 5 ] }$ , while the replacement slot is selected as argmax<sub>k</sub> $P _ { r } ( k )$ . When the query arrives at an arbitrary timestamp, the corresponding memory state and query are passed to the downstream VideoLLM to generate the response. We first store the initial K frames to fill the memory and activate SVMemAgent once the memory reaches full capacity. If a query arrives before the memory is full, we use all frames observed up to that timestamp.

## 4 Experiments

We evaluate our method on both ofline and online video understanding benchmarks. Ofline benchmarks pose the query after the stream ends, whereas online benchmarks pose queries at arbitrary timestamps during streaming. During evaluation, SVMemAgent processes videos strictly in an online manner at 1 FPS, without access to future frames, the total video duration, or the user query during frame selection. The memory state available when the query arrives is provided to the downstream VideoLLMs, InternVL3.5-8B [23] and Qwen3-VL-8B [1].

Implementation details. In SVMemAgent, we adopt DINOv3-Base [21] as the image encoder f, with 8-layer and 2-layer transformer encoders for the semantic and temporal branches, respectively, resulting in a total of 196M parameters. During training, the maximum memory budget of SVMem is set to $K = 1 6$ We use subsets of video-question-answer samples from the Video-R1 [5] and LongVILA [3] datasets. We filter out videos shorter than 64 seconds. In coldstart training, DINOv3-Huge [21] is employed as the teacher image encoder g for pseudo-label generation. For GRPO training, we use a rollout group size G of 32 and set the DAAD threshold to λ = 40. We use InternVL3.5-2B [23] as the VideoLLM reward model for task-driven supervision.

Table 2: Comparison with baselines of the strict online streaming setting.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Methods</td><td colspan="3">Offline video benchmarks</td><td>Online video benchmark</td><td rowspan="2">Average</td></tr><tr><td>|VideoMME LongVideoBench EgoSchema |</td><td></td><td></td><td>RVS-Ego</td></tr><tr><td rowspan="5">InternVL3.5-8B</td><td>FIFO</td><td>55.8</td><td>55.9</td><td>51.6</td><td>51.8</td><td>53.8</td></tr><tr><td>Reservoir</td><td>60.4</td><td>57.7</td><td>54.8</td><td>53.1</td><td>56.5</td></tr><tr><td>TimeChat-Online [25]</td><td>59.7</td><td>58.0</td><td>53.8</td><td>49.0</td><td>55.1</td></tr><tr><td>Flash-VStream [27]</td><td>57.3</td><td>57.9</td><td>54.2</td><td>53.5</td><td>55.7</td></tr><tr><td>MA-LMM [8]</td><td>59.5</td><td>57.5</td><td>54.4</td><td>51.0</td><td>55.6</td></tr><tr><td></td><td>SVMemAgent (ours)</td><td>61.5</td><td>58.5</td><td>55.6</td><td>54.1</td><td>57.4</td></tr><tr><td rowspan="6">Qwen3-VL-8B</td><td>FIFO</td><td>54.4</td><td>55.6</td><td>66.7</td><td>57.3</td><td>58.5</td></tr><tr><td>Reservoir</td><td>59.1</td><td>57.8</td><td>73.6</td><td>58.2</td><td>62.2</td></tr><tr><td>TimeChat-Online [25]</td><td>59.3</td><td>57.7</td><td>72.0</td><td>55.4</td><td>61.1</td></tr><tr><td>Flash-VStream [27]</td><td>57.2</td><td>57.0</td><td>72.0</td><td>58.8</td><td>61.2</td></tr><tr><td>MA-LMM [8]</td><td>59.2</td><td>57.1</td><td>72.4</td><td>58.6</td><td>61.8</td></tr><tr><td>SVMemAgent (ours)</td><td>60.6</td><td>58.0</td><td>76.2</td><td>58.8</td><td>63.4</td></tr></table>

Baselines. We compare SVMemAgent against a broad range of heuristic and state-of-the-art frame-selection baselines. For a fair comparison, we categorize these baselines according to whether they satisfy the core constraints of our problem, i.e., query-agnostic frame selection in an online streaming setting with unknown video duration. Directly comparable baselines are evaluated under the same strict online setting and memory budget, without access to future frames or the user query. In contrast, ofline and query-aware methods are reported separately. We summarize all baselines in Tab. 1.

## 4.1 Results on Video Benchmarks

Comparison with strict online streaming baselines. Tab. 2 compares SVMemAgent with both heuristic and state-of-the-art frame selection baselines under the strict online streaming setting, where future frames, total video duration, and query information are unavailable during frame selection. Under these constraints, SVMemAgent consistently outperforms competing approaches in terms of average accuracy. For example, when integrated with InternVL3.5-8B, SVMemAgent improves the average accuracy over Flash-VStream by 1.7%. Notably, although SVMemAgent is trained using InternVL3.5-2B as the reward model during GRPO, it generalizes efectively to a diferent backbone VideoLLM, Qwen3-VL-8B. In particular, SVMemAgent surpasses TimeChat-Online by 2.3% in average accuracy, demonstrating cross-model transferability of the learned memory policy.

Comparison with relaxed online or ofline baselines. In Tab. 3, we further compare SVMemAgent with baselines evaluated under relaxed online or ofline settings, where additional information, such as the total video duration, query information, or both, is available during frame selection. Despite having access to substantially less information, SVMemAgent outperforms most competing approaches. In particular, SVMemAgent outperforms StreamChat, which uses query information for keyframe selection, by 1.8% when evaluated with InternVL3.5-8B, despite having no access to the user query during frame selection. Overall, these results suggest that SVMemAgent learns an adaptive memory maintenance policy that sequentially discards and replaces frames under strict online constraints while achieving performance comparable to methods with substantially stronger information access.

Table 3: Comparison with baselines under relaxed online or ofline settings.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Methods</td><td colspan="2">| Setting assumptions | Unknown Unknown</td><td colspan="3">Offline video benchmarks</td><td rowspan="2">|Online video benchmark |</td><td rowspan="2">Average</td></tr><tr><td>Query</td><td>Duration</td><td>VideoMME LongVideoBench EgoSchema</td><td></td><td>RVS-Ego</td></tr><tr><td rowspan="7"></td><td>Uniform</td><td></td><td></td><td>61.3</td><td></td><td></td><td>53.9</td><td>57.0</td></tr><tr><td>Random</td><td>い</td><td>× ×× &gt;</td><td>59.7</td><td>57.4 57.5</td><td>55.4 55.4</td><td>53.9</td><td>56.6</td></tr><tr><td>Clustering</td><td>S</td><td></td><td>62.5</td><td>58.4</td><td>55.0</td><td>53.0</td><td>57.2</td></tr><tr><td>InternVL3.5-8B StreamChat [24]</td><td></td><td></td><td>58.9</td><td>59.3</td><td>52.2</td><td>52.0</td><td>55.6</td></tr><tr><td>AKS [22]</td><td></td><td></td><td>61.6</td><td>60.4</td><td>54.4</td><td>53.5</td><td>57.5</td></tr><tr><td>M-LLM Selector [9]</td><td>× × ×</td><td>××</td><td>53.8</td><td>51.9</td><td>49.0</td><td>42.3</td><td>49.2</td></tr><tr><td>SVMemAgent (ours)</td><td>√</td><td>√</td><td>61.5</td><td>58.5</td><td>55.6</td><td>54.1</td><td>57.4</td></tr><tr><td rowspan="7"></td><td>Uniform</td><td></td><td></td><td>61.1</td><td>58.3</td><td>75.2</td><td>58.5</td><td>63.3</td></tr><tr><td>Random</td><td>ノ</td><td></td><td>59.8</td><td>57.1</td><td>72.2</td><td>58.8</td><td>62.0</td></tr><tr><td>Clustering</td><td></td><td></td><td>62.2</td><td>58.0</td><td>75.0</td><td>59.7</td><td>63.7</td></tr><tr><td>StreamChat [24]</td><td></td><td></td><td>59.4</td><td>60.3</td><td>71.8</td><td>58.1</td><td>62.4</td></tr><tr><td>AKS [22]</td><td>× × ×</td><td>×××√××</td><td>61.6</td><td>61.1</td><td>75.0</td><td>59.0</td><td>64.2</td></tr><tr><td>M-LLM Selector [9]</td><td></td><td></td><td>52.0</td><td>50.4</td><td>62.5</td><td>41.0</td><td>51.5</td></tr><tr><td>SVMemAgent (ours)</td><td>√</td><td>√</td><td>60.6</td><td>58.0</td><td>76.2</td><td>58.8</td><td>63.4</td></tr></table>

Table 4: Efect of each training stage.  
Table 5: Efect of each branch.
<table><tr><td>Model</td><td>|Cold-start GRPO|Offline Online|Average</td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL3.5-8B</td><td>V V</td><td>V</td><td>57.9 58.5</td><td>52.9 54.1</td><td>56.7 57.5</td></tr><tr><td>Qwen3-VL-8B</td><td>V V</td><td>V</td><td>64.1 64.9</td><td>58.2 58.8</td><td>62.6 63.4</td></tr></table>

<table><tr><td>Model</td><td>|Temporal Semantic|Offline Online|Average</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">InternVL3.5-8B</td><td colspan="2">V</td><td>57.6</td><td>54.1</td><td>56.8</td></tr><tr><td></td><td>V</td><td>55.5</td><td>51.4</td><td>54.5</td></tr><tr><td>V</td><td>V</td><td>58.5</td><td>54.1</td><td>57.5</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>V</td><td></td><td>63.7</td><td>58.2</td><td>62.4</td></tr><tr><td></td><td>V</td><td>62.2</td><td>54.6</td><td>60.3</td></tr><tr><td>V</td><td>V</td><td>64.9</td><td>58.8</td><td>63.4</td></tr></table>

## 4.2 Ablation Studies

We conduct ablation studies on each training stage of SVMemAgent and analyze its temporal and semantic branches. We further evaluate inference-time performance under varying memory budgets.

Performance of each training stage. In Tab. 4, applying GRPO after coldstart training consistently improves performance across both backbones, leading to average gains of 0.8% . This suggests that task-driven reward supervision in GRPO refines the policy beyond heuristic pseudo-label supervision, enabling the agent to select frames that are broadly useful across diverse queries.

Efect of each branch. As shown in Tab. 5, the temporal branch alone outperforms the semantic branch alone, indicating that maintaining temporal uniformity in memory provides a robust strategy for query-agnostic memory maintenance when the video duration is unknown. Combining the two branches consistently achieves the best results across both backbones, highlighting the necessity of jointly considering temporal coverage and informative content preservation.

Efect of the reward model. In Tab. 6, we further assess the sensitivity of SVMemAgent to the reward model used during GRPO training by replacing InternVL3.5-2B with Qwen3-VL-2B. The policy trained with Qwen3-VL-2B achieves nearly identical performance, with average accuracy diferences of only

Table 6: Results with diferent reward models.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Reward model</td><td colspan="3">Offline video benchmarks</td><td>Online video benchmark</td><td rowspan="2">Average</td></tr><tr><td></td><td>|VideoMME LongVideoBench EgoSchema</td><td></td><td>RVS-Ego</td></tr><tr><td rowspan="2">InternVL3.5-8B</td><td>Qwen3-VL-2B</td><td>61.4</td><td>58.9</td><td>55.0</td><td>53.9</td><td>57.3</td></tr><tr><td>InternVL3.5-2B</td><td>61.5</td><td>58.5</td><td>55.6</td><td>54.1</td><td>57.4</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>Qwen3-VL-2B</td><td>61.1</td><td>57.7</td><td>75.0</td><td>58.4</td><td>63.1</td></tr><tr><td>InternVL3.5-2B</td><td>60.6</td><td>58.0</td><td>76.2</td><td>58.8</td><td>63.4</td></tr></table>

![](images/5fd61c23876c277ca330603c2e6b4b23b4a39d9429ac07c11e17f04613dbd486.jpg)  
(a)

![](images/7bfbed77891890a78f225cc1c7719f0447927167693c92a6304feba7da89f605.jpg)  
(b)

![](images/aadbf6afb5e98c88ab60920c3aa3220a5fa238f26732ecdf22665e1d11514aa0.jpg)  
(c)  
Fig. 3: (a) Performance across diferent memory budgets, measured by the number of stored frames; (b) performance across diferent query arrival times; and (c) performance across varying levels of query novelty. Query novelty is computed as the mean cosine distance between a test-query embedding and its 10 nearest training-query embeddings, obtained using Sentence-BERT [19]. Higher values indicate that the query is more outof-distribution relative to the training queries.

0.1% and 0.3% when evaluated using InternVL3.5-8B and Qwen3-VL-8B as the downstream VideoLLMs, respectively. These results indicate that SVMemAgent learns a robust policy that is largely insensitive to the choice of reward model. Memory budget scaling. Fig. 3a illustrates the efect of varying the inferencetime memory budget from K = 4 to 64. SVMemAgent consistently outperforms competing online frame selection baselines across most memory budgets, despite being trained only under K = 16. These results suggest that the proposed policy learns scalable memory management behavior that generalizes across diferent memory capacities.

## 4.3 Analysis

To better understand why SVMemAgent performs efectively under online streaming constraints, we provide an in-depth analysis.

Is SVMemAgent robust to varying query arrival times? Fig. 3b evaluates SVMemAgent across a wide range of query arrival times. Although the average query arrival time during training is 196 seconds, SVMemAgent consistently outperforms the baseline across all evaluated arrival times. This result indicates that the learned memory maintenance policy generalizes beyond the training-time arrival distribution, despite having no prior knowledge of when a user query will arrive, which highlights its applicability for real-world streaming applications.

Table 7: Ablation studies on DAAD.
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>InternVL3.5 $| \mathrm { w / o } \rrangle$ DAAD w/ DAAD|</td><td rowspan=1 colspan=1>Qwen3-VLw/o DAAD w/ DAAD</td></tr><tr><td rowspan=1 colspan=1>Average|</td><td rowspan=1 colspan=1>56.3     57.4</td><td rowspan=1 colspan=1>62.9     63.4</td></tr></table>

Table 8: Per-frame latency breakdown of SVMemAgent under the strict online setting.
<table><tr><td colspan="4">[Image Enc. Semantic Temporal|</td><td>Total</td></tr><tr><td># of Params. (M)</td><td>85.7</td><td>86.8</td><td>23.1</td><td>195.6</td></tr><tr><td>Latency (ms/frame)</td><td>7.0</td><td>6.8</td><td>2.2</td><td>16.2 (61.7 FPS)</td></tr></table>

Does SVMemAgent generalize well to unseen queries? Since SVMemAgent performs frame selection without access to the query during streaming and is trained with task-driven rewards from only four queries per video, it is important to assess whether the learned policy remains robust under shifts in the query distribution. To this end, we evaluate performance under shifts in the query distribution with respect to query novelty, defined as the mean cosine distance between each Sentence-BERT test-query embedding and its 10 nearest training-query embeddings [19]. As shown in Fig. 3c, SVMemAgent consistently outperforms all baselines across the full range of query novelty, indicating that the learned policy generalizes efectively even to queries that difer substantially from those encountered during training.

Which samples are discounted by DAAD? We observe that although each rollout contains a diverse set of frames, with $D _ { \mathrm { { J a c } } } = 0 . 9$ , the resulting reward, defined as the negative loss, is already high and exhibits negligible variation across rollouts, with std $( R ) = 0 . 0 0 0 0 0 3 .$ This indicates that the reward depends primarily on the query itself rather than on the frame selection policy. For example, a question asking why a person extends their arms while walking on a slackline can be answered from language bias alone, i.e., ‘to maintain balance’. Consequently, language-biased query-answer pairs introduce noisy supervision signals, which can destabilize policy optimization with the original advantages. In contrast, DAAD successfully identifies such cases and assigns an advantage discounting score of $\alpha = 0 ,$ reducing their advantages to $\hat { A } = 0$ . This prevents noisy supervision from adversely afecting frame selection policy learning. As in Tab. 7, incorporating DAAD consistently improves performance over the variant without DAAD, demonstrating its efectiveness in stabilizing policy optimization.

What policy does SVMemAgent learn? Given a video dominated by a man discussing his travel experiences, interspersed with a few short scenes, SVMemAgent implicitly learns to avoid redundant talking-head frames and instead preserve frames from less frequent scenes that are more likely to contain information relevant to downstream questions. When prolonged talking scenes are already stored in memory, SVMemAgent repeatedly replaces the same memory slot with incoming frames depicting similar talking scenes. It further discards such frames when the memory already contains suficient talking-scene information. As a result, the learned policy successfully retains a short segment in which the man shows a photograph, which is critical for answering the downstream question, “Among the photos that the man is showing, which photo appears first?”. In contrast, ofline uniform sampling repeatedly selects visually similar frames of the man speaking, resulting in a highly redundant memory state and an incorrect answer.

Another intriguing behavior that emerges in SVMemAgent is its tendency to retain frames containing textual information. This emergent behavior is learned through GRPO training with task-driven rewards across diverse question-answer pairs, suggesting that the agent recognizes textual content as decisive evidence for downstream VideoQA. For instance, given a video that depicts a person applying lipstick, and the downstream question asks for the product number of the first lipstick used. SVMemAgent identifies and retains the critical frame, which contains a product number, thereby enabling the VideoLLM to produce the correct answer. In contrast, ofline uniform sampling captures visually repetitive and less informative frames, ultimately leading to the incorrect answer. Overall, compared with ofline uniform sampling, the proportion of text-containing frames stored in memory increases from 33.8% to 40.8%, suggesting that SVMemAgent implicitly recognizes textual content as a valuable source of information for future question answering and prioritizes such frames during memory maintenance. Eficiency analysis. During inference, the lightweight SVMemAgent with 195.6M parameters is invoked upon each frame arrival to update the memory using only the current memory state and the incoming frame. As in Tab. 8, SVMemAgent achieves a memory maintenance speed of 61.7 FPS, corresponding to only 16.2 ms of additional latency per frame under the strict online setting.

## 5 Conclusion

We introduced SVMem, a memory formulation for strict online video understanding, where frames arrive sequentially, the total stream duration is unknown, and the query is unavailable during memory maintenance. Building on this formulation, we proposed SVMemAgent, which maintains a compact streaming memory through discard-and-replacement decisions. SVMemAgent is trained with GRPO using task-driven rewards, enabling it to learn a general prior for selecting informative frames without access to the query or future frames. Experiments on multiple video benchmarks demonstrate that SVMemAgent consistently outperforms online frame selection baselines and most ofline methods, despite the latter assuming substantially stronger information access.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

2. Chen, J., Lv, Z., Wu, S., Lin, K.Q., Song, C., Gao, D., Liu, J.W., Gao, Z., Mao, D., Shou, M.Z.: Videollm-online: Online video large language model for streaming video. In: CVPR (2024)

3. Chen, Y., Xue, F., Li, D., Hu, Q., Zhu, L., Li, X., Fang, Y., Tang, H., Yang, S., Liu, Z., et al.: Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188 (2024)

4. Di, S., Yu, Z., Zhang, G., Li, H., Zhong, T., Cheng, H., Li, B., He, W., Shu, F., Jiang, H.: Streaming video question-answering with in-context video kv-cache retrieval. In: ICLR (2025)

5. Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., Yue, X.: Video-r1: Reinforcing video reasoning in mllms. In: NeurIPS (2025)

6. Feng, K., Zhang, M., Li, H., Fan, K., Chen, S., Jiang, Y., Zheng, D., Sun, P., Zhang, Y., Sun, H., et al.: Onethinker: All-in-one reasoning model for image and video. In: CVPR (2026)

7. Fu, S., Yang, Q., Li, Y.M., Peng, Y.X., Lin, K.Y., Wei, X., Hu, J.F., Xie, X., Zheng, W.S.: Vispeak: Visual instruction feedback in streaming videos. arXiv preprint arXiv:2503.12769 (2025)

8. He, B., Li, H., Jang, Y.K., Jia, M., Cao, X., Shah, A., Shrivastava, A., Lim, S.N.: Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In: CVPR (2024)

9. Hu, K., Gao, F., Nie, X., Zhou, P., Tran, S., Neiman, T., Wang, L., Shah, M., Hamid, R., Yin, B., et al.: M-llm based video frame selection for eficient video understanding. In: CVPR (2025)

10. Huang, Z., Li, X., Li, J., Wang, J., Zeng, X., Liang, C., Wu, T., Chen, X., Li, L., Wang, L.: Online video understanding: A comprehensive benchmark and memoryaugmented method. In: CVPR (2025)

11. Ko, D., Lee, J., Kang, W.Y., Roh, B., Kim, H.: Large language models are temporal and causal reasoners for video question answering. In: EMNLP (2023)

12. Li, W., Hu, B., Shao, R., Shen, L., Nie, L.: Lion-fs: Fast & slow video-language thinker as online video assistant. In: CVPR (2025)

13. Li, X., Wang, Y., Yu, J., Zeng, X., Zhu, Y., Huang, H., Gao, J., Li, K., He, Y., Wang, C., et al.: Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574 (2024)

14. Li, X., Yan, Z., Meng, D., Dong, L., Zeng, X., He, Y., Wang, Y., Qiao, Y., Wang, Y., Wang, L.: Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. In: NeurIPS (2025)

15. Liu, J., Yu, Z., Lan, S., Wang, S., Fang, R., Kautz, J., Li, H., Alvare, J.M.: Streamchat: Chatting with streaming video. arXiv preprint arXiv:2412.08646 (2024)

16. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visualrft: Visual reinforcement fine-tuning. In: ICCV (2025)

17. Qian, R., Ding, S., Dong, X., Zhang, P., Zang, Y., Cao, Y., Lin, D., Wang, J.: Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. arXiv preprint arXiv:2501.03218 (2025)

18. Qian, R., Dong, X., Zhang, P., Zang, Y., Ding, S., Lin, D., Wang, J.: Streaming long video understanding with large language models. In: NeurIPS (2024)

19. Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bertnetworks. In: EMNLP (2019)

20. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

21. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

22. Tang, X., Qiu, J., Xie, L., Tian, Y., Jiao, J., Ye, Q.: Adaptive keyframe sampling for long video understanding. In: CVPR (2025)

23. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025)

24. Xiong, H., Yang, Z., Yu, J., Zhuge, Y., Zhang, L., Zhu, J., Lu, H.: Streaming video understanding and multi-round interaction with memory-enhanced knowledge. In: ICLR (2025)

25. Yao, L., Li, Y., Wei, Y., Li, L., Ren, S., Liu, Y., Ouyang, K., Wang, L., Li, S., Li, S., et al.: Timechat-online: 80% visual tokens are naturally redundant in streaming videos. arXiv preprint arXiv:2504.17343 (2025)

26. Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., et al.: Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106 (2025)

27. Zhang, H., Wang, Y., Tang, Y., Liu, Y., Feng, J., Jin, X.: Flash-vstream: Eficient real-time understanding for long video streams. In: ICCV (2025)

28. Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 (2024)