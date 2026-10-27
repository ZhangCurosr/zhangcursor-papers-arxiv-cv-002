## Highlights

Fed-MAPTrack: Federated Modality-aware Prompt Tuning for RGB-X Tracking Xiangyu Liu, Yijie Shen, Yue Yin, Zhitong Yin, Yongqi Pan, Wentao Fan

• Federated prompt tuning enables heterogeneous RGB-X tracking while keeping videos local.

• RGB prompts are shared globally, while sensor-related parameters use modality banks.

• Selective aggregation retains RGB updates from clients without auxiliary sensors.

• Ablations show that modality bucketing requires compatible participation rules.

# Fed-MAPTrack: Federated Modality-aware Prompt Tuning for RGB-X Tracking

Xiangyu Liu<sup>a,1</sup>, Yijie Shen<sup>a,1</sup>, Yue Yin<sup>b</sup>, Zhitong Yin<sup>c</sup>, Yongqi Pan<sup>d,∗</sup> and Wentao Fan<sup>a,∗∗</sup>

<sup>a</sup>Beijing Normal-Hong Kong Baptist University, 519087, Zhuhai, Guangdong, China

<sup>b</sup>Zhejiang University, 310058, Hangzhou, Zhejiang, China

<sup>c</sup>Peking University, 100871, Beijing, China

<sup>d</sup>Military Management College, National Defense University PLA China, 100091, Beijing, China

## A R T I C L E I N F O

Keywords:   
Multimodal federated learning   
RGB-X tracking   
Visual tracking   
Prompt tuning   
Modality heterogeneity

## A BS T R AC T

RGB-X tracking complements RGB appearance with thermal, depth, or event observations, but centralized training requires collecting multimodal videos at a single location. Federated training keeps raw videos local, but sensor heterogeneity can create a mismatch between parameter-sharing domains and the clients contributing to them. This mismatch is particularly relevant when RGBonly clients coexist with sensor-equipped clients: their RGB observations remain useful, but they provide no auxiliary measurements for learning sensor-dependent interactions. We propose Fed-MAPTrack, a modality-aware federated prompt tuning framework for RGB-X tracking. The pretrained tracking backbone and prediction head are frozen, and clients optimize and exchange only promptrelated parameters. The server aggregates RGB-shared parameters globally and maintains separate thermal, event, and depth banks for auxiliary-specific, fusion, and normalization parameters. Selective participation retains RGB-only clients’ contributions to the global RGB bank while excluding their fusion and normalization updates from modality-specific aggregation. Across three matched training repetitions under a fixed client partition, Fed-MAPTrack improves the mean FedAvg scores by 2.35 AUC points on LasHeR, 3.45 SR points on VisEvent, and 2.11 F-score points on DepthTrack. Among the nine evaluated methods, it achieves the highest mean thermal AUC and depth F-score, while its mean event SR is 0.07 points below that of Independent FL. Controlled ablations show that extending the bucketing scope without a compatible participation rule can reduce performance. These results highlight the need to jointly define the sharing domain and eligible contributors for each parameter group.

## 1. Introduction

Visual object tracking localizes a target throughout a video given its initial state. Siamese and Transformer-based trackers have advanced template-search matching and target localization [2, 16, 20], but RGB appearance alone can be unreliable under low illumination, occlusion, motion blur, or background distraction. RGB-X tracking supplements RGB with thermal, depth, or event observations. These sensors provide complementary cues: thermal observations capture infrared appearance, depth describes scene geometry, and events encode brightness changes with high temporal resolution [10, 15, 17]. Learning to exploit these complementary cues is essential for reliable tracking under challenging visual conditions.

RGB-X trackers are commonly trained on centrally accessible paired videos [19, 22]. In a distributed setting, however, data may remain on devices with diferent sensor configurations. Some clients have RGB-thermal, RGB-depth, or RGB-event observations, while others record RGB only. Data-governance requirements and bandwidth or storage constraints may limit the transfer of these videos to a central training facility. Federated learning [13] provides an alternative in which clients keep videos local and exchange trainable parameters. Figure 1 illustrates the transition from centralized RGB-X training to this heterogeneous-client setting. Keeping videos local avoids their direct transfer, although parameter exchange alone does not provide a formal privacy guarantee.

![](images/0cde0106030ec07d3e22a36a935c9ffbacd023ccac1d7c3963a47ae035015819.jpg)  
Figure 1: Motivation of federated RGB-X tracking. (a) Centralized RGB-X tracking pools aligned data from all sensors for one training run. (b) In the federated setting, clients with diferent sensors (RGB, depth, thermal, event) train locally, and only the tuned prompt parameters are aggregated on top of a frozen ViT backbone.

Motivation. The central challenge is to determine which parameter groups clients should share and which clients should contribute to each group. All clients observe RGB, making a common RGB pathway a natural sharing domain. Auxiliary observations, in contrast, have diferent physical meanings across sensors, so a common parameter layout does not imply that every sensor-dependent update should be pooled globally. Fusion and normalization parameters also interact with the auxiliary pathway, even when they are not explicitly labeled as auxiliary prompts. Their sharing rules must therefore be considered alongside those for auxiliaryspecific parameters.

A second distinction concerns RGB-only clients. Their RGB observations remain useful for collaborative training. However, fusion and normalization parameters can receive local updates even when auxiliary inputs are unavailable. Assigning an RGB-only client to a modality bucket for parameter dispatch does not make its updates suitable for every parameter group in that bucket. Sharing domains and contributor eligibility must therefore be defined explicitly and designed jointly.

Existing RGB-X methods primarily study local multimodal adaptation and fusion, whereas conventional and personalized federated methods address optimization or clientspecific model components [1, 11, 12]. These approaches do not, by themselves, specify the sensor-aware sharing and participation rules needed for the diferent functional groups of an RGB-X tracker. Separating the federation by modality avoids cross-sensor aggregation but also removes a shared RGB bank across modalities. At the same time, exchanging the full tracker is unnecessary when adaptation can be confined to prompt-related parameters. These observations motivate a framework that combines a common frozen tracker with group-specific federated aggregation.

We propose Fed-MAPTrack, a modality-aware federated prompt tuning framework built on a frozen pretrained tracker. It organizes trainable parameters into RGB-shared, auxiliaryspecific, fusion, and normalization groups. The RGB-shared group is aggregated globally, while the other groups use thermal, event, and depth banks. Selective participation completes the bucket design: RGB-only clients continue local training and contribute to the common RGB bank, but their fusion and normalization updates are excluded from modalityspecific server averages. Their auxiliary-specific parameters remain frozen and are already excluded from aggregation by the validity mask. Bucketing and selective participation thus form a unified aggregation strategy. Controlled experiments examine both design choices and show that extending the bucketing scope without adjusting contributor eligibility can reduce performance.

The main contributions of this paper are summarized as follows.

• We study federated RGB-X tracking with RGB-T, RGB-D, RGB-E, and RGB-only clients, explicitly distinguishing auxiliary-sensor availability from the originating dataset used for parameter dispatch.

• We develop a prompt-only federated framework with a frozen tracking backbone and prediction head, restricting local optimization and round-wise communication to prompt-related parameters.

• We jointly design modality-aware bucketing and selective participation: the former determines sharing domains, and the latter determines eligible contributors while retaining RGB-only participation in the shared RGB pathway.

• We evaluate the framework on thermal, event, and depth benchmarks using matched training repetitions, controlled aggregation ablations, and final-parameter analyses, demonstrating improvements over FedAvg under the evaluated partition and clarifying the role of participation eligibility.

The remainder of this paper is organized as follows. Section 2 reviews related work on RGB-X tracking, federated learning, and prompt tuning. Section 3 presents the proposed Fed-MAPTrack framework. Section 4 reports experimental settings, results, and ablation studies. Section 5 discusses limitations and future directions, and Section 6 concludes this paper.

## 2. Related Work

We review four related areas: RGB-X visual tracking, federated learning for visual perception, multimodal federated learning, and parameter-eficient prompt tuning. The comparison focuses on their treatment of sensor heterogeneity and parameter sharing in distributed RGB-X tracking.

## 2.1. RGB-X Tracking

RGB visual tracking localizes a target in successive frames given its initial state. Advances in Siamese networks and Transformer-based architectures have improved tracking accuracy and robustness through template-search matching, long-range dependency modeling, and feature interaction [2, 9, 16, 20]. However, RGB-only trackers are still vulnerable to challenging scenarios such as low illumination, heavy occlusion, motion blur, and complex background interference.

To overcome the limitations of RGB tracking, RGB-X tracking introduces auxiliary modalities, such as thermal infrared, depth, and event data, to provide complementary information. RGB-thermal tracking exploits illuminationinsensitive thermal cues and is efective in low-light or nighttime scenes [6, 10]. RGB-depth tracking uses geometric and spatial structure to distinguish targets from background distractors [17]. RGB-event tracking benefits from the high temporal resolution and motion sensitivity of event cameras, which can improve tracking under fast motion and illumination changes [15]. Existing RGB-X trackers mainly focus on cross-modal feature fusion, modality reliability estimation, and transformer-based multimodal interaction [19, 22]. Although these methods have achieved promising performance, they are typically developed under a centralized training paradigm, assuming that aligned multimodal videos from diferent sensors can be collected in one place. Such an assumption is dificult to satisfy in real-world edge scenarios where raw videos are privacy-sensitive and clients have heterogeneous sensor configurations.

## 2.2. Federated Visual Learning

Federated learning enables multiple clients to train a model collaboratively without sharing their raw data. A typical approach is FedAvg, which aggregates local model updates on the server according to client data sizes or participation weights [13]. To improve federated optimization under non-IID data, subsequent methods introduce proximal regularization, personalized model components, or clientspecific adaptation strategies [1, 11, 12]. These methods have been applied to various visual perception tasks, including image classification, object detection, and semantic segmentation.

Conventional model averaging typically uses a common parameterization across clients, but does not explicitly specify sensor-dependent sharing domains for RGB-X tracking. Clients with RGB-thermal, RGB-depth, RGB-event, and RGB-only data can share tensor shapes while difering in the observations used to optimize those tensors. Tracking also requires target-specific localization and template-search interaction rather than only image-level recognition. These distinctions motivate an aggregation strategy that considers the functional roles of tracking parameters as well as sensor availability.

## 2.3. Multimodal Federated Learning

Multimodal federated learning studies collaborative training over distributed clients with multimodal data. Existing settings can be grouped into three categories according to the distribution of modalities across clients: all clients hold the same set of modalities, each client holds only one modality, or clients hold heterogeneous subsets of modalities [3]. To address modality heterogeneity, existing methods usually rely on representation alignment, knowledge distillation, modality completion, or personalized aggregation [7, 21]. These designs transfer knowledge across clients while reducing discrepancies caused by missing or heterogeneous modalities. FedMSplit [4] divides client models into shareable blocks and uses a dynamic multi-view graph to model client correlations for block-wise aggregation. FED-PRIME [14] combines federated learning with multimodal prompt tuning, using inter-client prompts to transfer shared knowledge and intraclient prompts to model local missing-modality patterns.

Much of this work evaluates classification, recognition, or activity understanding, whereas RGB-X tracking uses auxiliary observations for target localization and templatesearch interaction. Alignment and completion can facilitate knowledge transfer, but they do not by themselves determine whether RGB-shared, auxiliary-specific, fusion, and normalization parameters should use the same aggregation participants. RGB-only clients make this distinction explicit: they have useful RGB data but no measurements of the corresponding auxiliary sensor. Our focus is therefore a group-specific aggregation rule tailored to this tracking setting, rather than a replacement for multimodal alignment or completion.

## 2.4. Parameter-Eficient Prompt Tuning

Prompt tuning has emerged as an efective parametereficient adaptation technique. Instead of fine-tuning the entire model, prompt tuning learns a small set of prompt parameters while keeping most pretrained parameters frozen [5, 8]. This strategy has been explored in vision tasks to adapt large pretrained models with limited trainable parameters. For visual tracking, prompt-based adaptation can guide feature extraction, template-search interaction, and multimodal fusion, making it suitable for eficient RGB-X tracking [22].

Prompt tuning also limits the parameter payload exchanged during federated training. For personalized federated adaptation, pFedPG learns a server-side generator that produces client-specific visual prompts for frozen backbones [18]. In contrast to client-specific prompt generation, our focus is the sharing domain and contributor eligibility of each functional parameter group under heterogeneous sensor availability. Our framework builds on prompt-based tracking and assigns aggregation rules to the functional groups of its trainable parameters. This combines global RGB sharing with modality-specific adaptation and explicitly handles the eligibility of RGB-only clients.

## 3. Method

Fed-MAPTrack couples two aggregation decisions: where a parameter group is shared and which clients may contribute to that group. We first define the local prompt tracker and its parameter groups. Modality-aware bucketing then defines global and sensor-specific sharing domains, while selective participation determines the eligible contributors within each domain. Together, bucketing and selective participation define a unified aggregation strategy.

## 3.1. Overall Framework

Clients and observations. Let ${ \mathcal C } = \{ 1 , \dots , N \}$ be the client population and $S ^ { t } \subseteq c$ the clients participating in round �. Client � owns a local tracking dataset $D _ { i }$ with $n _ { i } ~ = ~ | D _ { i } |$ assigned samples. Its auxiliary-sensor type is $m _ { i } ~ \in ~ \{ T , D , E , \emptyset \}$ , where �, �, and � denote thermal, depth, and event sensors and ∅ denotes RGB only. We write $v _ { i } = \mathbb { I } ( m _ { i } \neq \emptyset )$ for client-level auxiliary availability. This indicator describes the sensor configuration, not the presence of an auxiliary observation in every individual training sample.

A local sample contains a template �, a search observation �, and a target box �. The template and search inputs each contain RGB and, when available, the corresponding auxiliary observation. Unavailable auxiliary channels are zero-masked under the implemented input protocol. The same tracker can therefore process the diferent client types without changing its parameter shapes.

![](images/7c66afc2e24ffb8fac78230b0c54b855f4efb3cb0baa4c0d87af601d4d4314d9.jpg)  
Figure 2: Overall framework of Fed-MAPTrack. Clients keep training videos local and adapt prompt-related parameters on a frozen tracking backbone and prediction head. RGB-shared parameters use one global bank, while auxiliary-specific, fusion, and normalization parameters use modality banks. Server aggregation combines this sharing structure with sensor-dependent contributor eligibility.

Functional parameter groups. Let Φ contain the frozen pretrained tracking backbone and prediction head. The trainable prompt-related parameters are organized as

$$
\Theta = \{ \Theta ^ { r } , \Theta ^ { a } , \Theta ^ { f } , \Theta ^ { n } \} ,\tag{1}
$$

with group indices $\begin{array} { l l l } { \mathcal { G } } & { = } & { \{ r , a , f , n \} } \end{array}$ . The RGB-shared group � and auxiliary-specific group � adapt the RGB and auxiliary pathways, respectively. The fusion group � governs interactions between these pathways, while the normalization group � contains the normalization parameters of the prompt modules. The term “prompt-related” encompasses adapter and normalization tensors as well as prompt tokens. This functional grouping determines the server’s sharing and participation rules, while the local architecture remains identical across aggregation variants.

Local adaptation. The client tracker predicts a target box as

$$
\hat { y } = F _ { \Phi } ( z , x ; \Theta _ { i } ) .\tag{2}
$$

The local tracking objective is

$$
\mathcal { L } _ { i } ( \Theta _ { i } ) = \frac { 1 } { n _ { i } } \sum _ { ( z , x , y ) \in D _ { i } } \ell \big ( F _ { \Phi } ( z , x ; \Theta _ { i } ) , y \big ) ,\tag{3}
$$

where $\ell$ is the base tracker’s original loss, which combines box regression with target-response supervision. Minibatches are sampled from the client’s assigned sequences using the same preprocessing protocol across the core comparisons.

Local trainability and server eligibility are separate concepts. For the configuration evaluated here, define the local trainability indicator as

$$
u _ { i } ^ { g } = \left\{ { v _ { i } } , \quad g = a , \right.\tag{4}
$$

Thus, an RGB-only client freezes its auxiliary-specific group but can locally optimize RGB-shared, fusion, and normalization parameters with masked auxiliary inputs. A parameter group may therefore be trainable locally without being eligible for server aggregation.

At the start of round �, the client receives the server prompt view $\Theta _ { i } ^ { t , 0 }$ . Its final local parameters are denoted by

$$
\begin{array} { r } { \widehat { \Theta } _ { i } ^ { t } = \operatorname { L o c a l A d a m W } _ { i } \big ( \Theta _ { i } ^ { t , 0 } ; \Phi , E \big ) , } \end{array}\tag{5}
$$

where � is the local mini-batch budget and only groups with $u _ { i } ^ { g } = 1$ are optimized. The operator denotes budgeted local AdamW training, including the gradient accumulation

specified in Section 4.1. The client returns only its prompt tensors and validity metadata, keeping raw videos and frozen model parameters local.

## 3.2. Modality-Aware Aggregation

From one sharing domain to sensor-specific domains. A global prompt baseline aggregates valid updates from all participating clients. Let $q _ { i } ^ { g , t }$ indicate whether client � supplies a valid update for group � in round �, and let $\mathcal { V } _ { g } ^ { t } = \{ i \in S ^ { t }$ $q _ { i } ^ { g , t } = 1 \}$ . For a nonempty valid set, global aggregation is

$$
\Theta _ { \mathrm { g l o b a l } } ^ { g , t + 1 } = \sum _ { i \in \mathcal { V } _ { g } ^ { t } } \frac { n _ { i } } { \sum _ { j \in \mathcal { V } _ { g } ^ { t } } n _ { j } } \widehat { \Theta } _ { i } ^ { g , t } .\tag{6}
$$

Validity is checked before averaging. For example, an RGBonly client’s frozen auxiliary-specific tensors do not become valid contributions merely because they have the expected shape. In the implementation, these checks are applied to the uploaded tensors using parameter masks.

Global aggregation assigns a single sharing domain to each parameter group. This choice is restrictive for RGB-X tracking because thermal appearance, depth geometry, and brightness-change events may require diferent processing and interactions with RGB. Bucketing therefore assigns sensor-dependent groups to distinct domains rather than assuming that all such tensors should share one bank. This separates sensor-specific adaptation from common RGB sharing without changing the local model.

Bucket assignment and dispatch. The bucket label is

$$
b _ { i } = { \left\{ \begin{array} { l l } { m _ { i } , } & { m _ { i } \neq \emptyset , } \\ { \rho ( i ) , } & { m _ { i } = \emptyset , } \end{array} \right. }\tag{7}
$$

where $\rho ( i ) \in \{ T , D , E \}$ identifies the auxiliary modality of the client’s originating dataset. This metadata specifies the parameter configuration dispatched to an RGB-only client without implying that the corresponding sensor is available. For parameter group �, define

$$
\Omega _ { g } ( i ) = \left\{ \stackrel { \star } { \kappa } , \quad g = r , \right. \nonumber\tag{8}
$$

where ⋆ denotes the global RGB bank. The server therefore maintains one global RGB-shared bank and three modality banks for each remaining parameter group. The initialization dispatched to a client is

$$
\Theta _ { i } ^ { g , t , 0 } = \left\{ \begin{array} { l l } { \Theta _ { \mathrm { i n i t } } ^ { a } , } & { g = a , v _ { i } = 0 , } \\ { \Theta _ { \Omega _ { g } ( i ) } ^ { g , t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

Here, $\Theta _ { \mathrm { i n i t } } ^ { a }$ is the stored initial auxiliary-specific state. It is dispatched and held fixed for RGB-only clients, while their fusion and normalization parameters use the banks associated with their originating datasets. Dispatch determines the client’s initial parameter configuration, while aggregation eligibility determines whether its returned tensors contribute to the corresponding bank.

![](images/693946368c26234e486d489842fc3dc3e75696f552c46a2aef7736785ed53cc3.jpg)  
Figure 3: Unified aggregation design. RGB-shared parameters use the global bank, while auxiliary-specific, fusion, and normalization parameters use thermal, depth, and event banks. The participation gate retains valid RGB updates from RGBonly clients but excludes their modality-related contributions from server bucket averages. Dispatch and local trainability are distinct from server eligibility.

Group-specific weighted aggregation. Let $a _ { i } ^ { g }$ be the server participation gate defined in Section 3.3. For domain � and group �, the efective contributor set is

$$
\begin{array} { r } { C _ { d } ^ { g , t } = \{ i \in S ^ { t } \mid \Omega _ { g } ( i ) = d , q _ { i } ^ { g , t } = 1 , a _ { i } ^ { g } = 1 \} . } \end{array}\tag{10}
$$

A client belongs to this set if it participates in the current round, matches the sharing domain, and supplies a valid update that passes the participation gate. If it is nonempty, the server computes

$$
\Theta _ { d } ^ { g , t + 1 } = \sum _ { i \in C _ { d } ^ { g , t } } \frac { n _ { i } } { \sum _ { j \in C _ { d } ^ { g , t } } n _ { j } } \widehat { \Theta } _ { i } ^ { g , t } .\tag{11}
$$

The denominator is recomputed for the eligible group-domain pair. Excluded updates are not represented by zero tensors, and they do not retain weight in the normalization. If no eligible update is available, the server instead preserves the previous bank:

$$
\Theta _ { d } ^ { g , t + 1 } = \Theta _ { d } ^ { g , t } , \qquad C _ { d } ^ { g , t } = \emptyset .\tag{12}
$$

This handles absent contributors without overwriting a modality bank with unrelated updates or a zero-valued replacement.

## 3.3. Selective Participation

Bucketing identifies a sharing domain but does not establish the suitability of every contributor assigned to that domain. An RGB-only client can locally change fusion and normalization parameters using its masked auxiliary inputs, yet those changes are not learned from observed RGB-X interaction. At the same time, its RGB observations remain relevant to the shared RGB pathway. Selective participation therefore acts on parameter groups rather than removing the client from collaborative training.

The server participation gate is

$$
a _ { i } ^ { g } = \left\{ { \begin{array} { l l } { 1 , } & { g = r , } \\ { v _ { i } , } & { g \in \{ a , f , n \} . } \end{array} } \right.\tag{13}
$$

Together with Eq. (10), this retains every valid RGB contribution while restricting modality-related server averages to equipped clients in the matching bucket. The validity mask $q _ { i } ^ { \bar { g , { t } } }$ still applies: sensor availability does not override an invalid uploaded tensor. Occasional sample-level auxiliary masking does not, by itself, change the client’s sensor label.

For auxiliary-specific parameters, invalid RGB-only contributions are already rejected by the validity mask. The additional efect of S-A in the controlled FNA comparison is therefore the exclusion of RGB-only fusion and normalization updates. FNA uses the same bank assignment and dispatch rule but sets $a _ { i } ^ { g } = 1$ for all groups, leaving those locally valid updates in the averages. The comparison therefore isolates server aggregation eligibility while holding sharing domains and the local training protocol fixed.

An RGB-only client still receives its dataset-associated bank, trains its permitted local groups, and supplies its RGB-shared update. Its fusion and normalization tensors are filtered from the corresponding server averages, not necessarily omitted from the upload. Thus, S-A is not an additional communication-compression mechanism. Promptonly exchange reduces the tensor payload, while the aggregation strategy determines how the exchanged tensors are combined. Algorithm 1 summarizes the complete procedure. Section 4 evaluates tracking performance and quantifies the communication payload.

## 4. Experiments

## 4.1. Experimental Setup

Implementation and training configuration. We implement Fed-MAPTrack on the ViPT tracking architecture [22]. The pretrained tracking backbone and prediction head are frozen throughout federated optimization, while only promptrelated parameters are updated locally. Each sensor-equipped local model contains 839,628 trainable parameters, organized into RGB-shared, auxiliary-specific, fusion, and normalization groups with 73,836, 664,416, 82,944, and 18,432 parameters, respectively. All core aggregation variants use this parameterization, so their performance diferences cannot be attributed to increased local model capacity. The training configurations specify NVIDIA RTX 4090 GPUs, while all benchmark evaluations use the same machine equipped with an NVIDIA RTX 4080 SUPER GPU.

Algorithm 1 Unified federated prompt training with bucket  
ing and S-A.   
Input: Clients  and local datasets $D _ { i } ;$   
Input: Sensor labels � and dataset labels $\rho ( i ) ;$   
Input: Frozen tracker Φ, rounds �, mini-batch budget �.   
Output: Global RGB bank and modality-related banks.   
1: Initialize all banks from the common prompt state.   
2: Store the fixed auxiliary state $\Theta _ { \mathrm { i n i } } ^ { a }$   
3: Define $B _ { r } = \{ \star \}$ and $\dot { B } _ { g } = \{ T , D , E \}$ for $g \neq r .$   
4: Set client buckets by Eq. (7).   
5: for � = 0 to � − 1 do   
6: Select participating clients $S ^ { t } .$   
7: for all $\bar { i } \in \mathcal { S } ^ { i }$ in parallel do   
8: Download the view in Eq. (9).   
9: Set trainability by $\mathrm { E q . } ( 4 ) .$   
10: Run local AdamW in Eq. (5).   
11: Return prompt tensors and validity metadata.   
12: for all $g \in \mathcal { G } , \ d \in B _ { g }$ do   
13: Apply the gate in Eq. (13).   
14: Build $c _ { d } ^ { g , t }$ by Eq. (10).   
15: if $c _ { d } ^ { g , t }$ ≠ ∅ then   
16: Aggregate by Eq. (11).   
17: else   
18: Keep the bank by Eq. (12).   
19: return Global RGB bank and the modality banks.

The core experiments use 120 communication rounds, with all twelve clients selected in each round. Each selected client processes 50 local mini-batches per round using AdamW, a learning rate $\operatorname { o f } 4 \times 1 0 ^ { - 4 }$ , and weight decay of $1 0 ^ { - 4 }$ . The configured mini-batch size is 32, with gradient accumulation over two mini-batches, giving an efective optimization batch size of 64 and 25 optimizer updates per client per round. We distinguish local mini-batches from optimizer updates because the implementation’s local-step budget counts the former. Optimizer moments are reset at the start of each local training round. Template and search crops have spatial resolutions of $1 2 8 \times 1 2 8$ and $2 5 6 \times 2 5 6$ respectively. All core variants use the same tracking objective, preprocessing, local optimization budget, and parameter initialization for matched training repetitions. Progressive prompt-group warm-up is not used in these comparisons, so the ablation isolates aggregation rather than an additional training schedule.

Datasets and client partition. We evaluate RGBthermal, RGB-event, and RGB-depth tracking on LasHeR [10], VisEvent [15], and DepthTrack [17], respectively. Training sequences are assigned to clients using one fixed sequence-level Dirichlet partition with concentration parameter $\alpha = 0 . 5 .$ . The partition is constructed within each benchmark and remains unchanged across the compared methods. Each benchmark supplies four clients: three sensor-equipped RGB-X clients and one RGB-only client. The resulting federation contains nine sensor-equipped clients and three RGB-only clients. Training sequences remain local to their assigned clients, while evaluation uses the corresponding benchmark test split. Client construction does not use manually assigned scene categories.

An RGB-only client retains the RGB observations of its assigned sequences but has no usable auxiliary sensor input. Its auxiliary-specific parameter group is frozen and excluded from valid auxiliary aggregation. For sensor-equipped clients, auxiliary observations are additionally masked with probability 0.1 during local training, using the same masking protocol in the core comparisons. This sample-level augmentation is distinct from client-level sensor availability: a sensorequipped client remains eligible under S-A despite occasional masking, whereas an RGB-only client has no auxiliary observations throughout training. The assigned trainingsample counts are 459,122 for LasHeR, 212,062 for VisEvent, and 209,913 for DepthTrack, giving 881,097 samples in total. The dataset-specific allocation to RGB-only clients is analyzed in Section 4.4.

Evaluation protocol. We report success-curve AUC for LasHeR, success rate (SR) for VisEvent, and the oficial F-score for DepthTrack. The LasHeR evaluator averages the success curve over overlap thresholds from 0 to 1 in increments of 0.05. In the VisEvent evaluator, SR is the mean of the success curve on the same threshold grid, rather than success at a single overlap threshold. DepthTrack is evaluated using its benchmark analysis protocol. All three metrics are expressed as percentages, with higher values indicating better tracking performance. The benchmarks difer in their metric definitions, so we report them separately and do not calculate a combined score.

For every run, we evaluate the checkpoint saved at communication round 120. Checkpoints are not selected according to test performance. Each main-table method and core ablation includes three matched training runs with seeds 42, 0, and 123. We report the mean and sample standard deviation across these runs, using �−1 as the variance denominator. For Local, FedPer, Ditto, FedMSplit-P, and FED-PRIME, each sensor-equipped client has its own evaluated model. Within each repetition, we average the benchmark scores of its three sensor-equipped client models. We then compute the reported mean and sample standard deviation across the three repetitions. For the remaining main-table methods, each benchmark is evaluated using its corresponding server-side prompt configuration. This distinguishes an average of clientmodel scores from the score of a single aggregated model and avoids treating the nine client evaluations as nine independent training repetitions. The same client partition is used in every repetition. Consequently, the reported standard deviations describe training variability under the evaluated partition and do not measure variability across independently generated client partitions. Diferences discussed below are calculated from the displayed mean scores and expressed in percentage points.

For the supplementary LasHeR curves, all methods use the same 245 test sequences. Success curves use the IoU threshold grid specified above, and precision curves use location-error thresholds from 0 to 50 pixels in one-pixel increments. We first average the curves over the test sequences within each repetition. The displayed curves and shaded bands then represent the mean and sample standard deviation across the three repetitions. Precision at 20 pixels provides a complementary center-localization metric and does not replace the primary success AUC comparison.

Compared methods. The main comparison includes Local, FedAvg [13], FedProx [12], FedPer [1], Ditto [11], FedMSplit-P [4, 14], FED-PRIME [14], Independent FL, and the complete Fed-MAPTrack. Local provides a non-collaborative reference using separately trained client models. FedAvg performs sample-weighted global averaging over valid prompt updates without maintaining modality specific parameter banks. FedProx adds a proximal penalty with coeficient $\mu ~ = ~ 0 . 0 1$ to constrain local deviation from the server parameters. FedPer provides a shared-andpersonalized model reference, and Ditto combines a federated model with client-specific regularized models using � = 0.1. FedMSplit-P and FED-PRIME are adapted to the ViPT tracker following the prompt-tuning implementations in [14]. FedMSplit-P maintains a model for each client and replaces uniform server averaging with a similarityweighted combination of client models for each parameter group, using a proximal coeficient of 0.01. FED-PRIME maintains intra-client and inter-client prompt pools of 20 tokens each and prepends the top five tokens from each pool to the Transformer input. The intra-client pool remains local, while inter-client prompts are aligned by Hungarian matching and averaged across clients. The remaining parameter groups follow FedAvg aggregation. Hyperparameters follow the orig inal implementation defaults and are not tuned for individual benchmarks. These are adapted tracking baselines rather than exact reproductions of the original methods on their classification benchmarks. These coeficients are fixed across the three benchmarks and training repetitions in the reported comparison. The client-model evaluation of Local, FedPer, Ditto, FedMSplit-P, and FED-PRIME follows the reporting rule stated above. Independent FL constructs a separate federation for each benchmark using its three sensor-equipped clients and one RGB-only client, with no prompt parameters shared across federations. Fed-MAPTrack maintains one global RGB-shared bank and modality-specific auxiliary, fusion, and normalization banks, with selective participation applied to the modality-interactive groups. Both the sharing domains and aggregation eligibility are therefore determined at the parameter-group level. Auxiliary-only bucketing and extended bucketing without S-A are controlled variants of this design and appear in the ablation rather than as competing final methods in the main table.

The experiments are organized around four research questions:

• RQ1: How does the complete framework compare with the available federated tracking baselines across auxiliary modalities?

• RQ2: How do the bucketing scope and selective participation afect tracking performance?

• RQ3: How does the participation design relate to RGBonly data allocation and the final structure of modalityrelated parameters?

Table 1 Main results on RGB-X tracking benchmarks at communication round 120. Scores are mean ± sample standard deviation across three matched training repetitions under one fixed client partition. For Local, FedPer, Ditto, FedMSplit-P, and FED-PRIME, each repetition first averages the scores of the benchmark’s three sensor-equipped client models. Bold indicates the highest mean in each metric.
<table><tr><td>Method</td><td>Category</td><td>LasHeR (RGB-T) AUC (%)</td><td>VisEvent (RGB-E) SR (%)</td><td>DepthTrack (RGB-D) F-score (%)</td></tr><tr><td>Local</td><td>Local training</td><td> $4 7 . 3 4 \pm 0 . 3 1$ </td><td> $5 2 . 2 7 \pm 0 . 2 5$ </td><td> $4 9 . 9 1 \pm 1 . 4 6$ </td></tr><tr><td>FedAvg</td><td>Standard FL</td><td> $4 6 . 5 5 \pm 0 . 9 9$ </td><td> $5 2 . 9 3 \pm 0 . 5 9$ </td><td> $5 3 . 8 5 \pm 0 . 7 3$ </td></tr><tr><td>FedProx</td><td>Standard FL</td><td> $4 6 . 7 8 \pm 0 . 6 7$ </td><td> $5 3 . 1 1 \pm 0 . 6 6$ </td><td> $5 4 . 0 6 \pm 0 . 6 4$ </td></tr><tr><td>FedPer</td><td>Personalized FL</td><td> $4 7 . 3 5 \pm 0 . 5 4$ </td><td> $5 4 . 4 7 \pm 0 . 3 6$ </td><td> $5 0 . 8 1 \pm 0 . 6 8$ </td></tr><tr><td>Ditto</td><td>Personalized FL</td><td> $4 6 . 4 6 \pm 0 . 8 6$ </td><td> $5 4 . 1 4 \pm 0 . 4 2$ </td><td> $5 3 . 9 0 \pm 0 . 5 8 $ </td></tr><tr><td>FedMSplit-P</td><td>Multimodal FL</td><td> $4 6 . 4 3 \pm 0 . 3 3$ </td><td> $5 4 . 8 5 \pm 0 . 3 5$ </td><td> $5 4 . 2 4 \pm 0 . 8 9$ </td></tr><tr><td>FED-PRIME</td><td>Multimodal FL</td><td> $4 6 . 7 8 \pm 0 . 4 5$ </td><td> $5 2 . 0 9 \pm 0 . 2 6$ </td><td> $5 4 . 3 2 \pm 0 . 1 7$ </td></tr><tr><td>Independent FL</td><td>Separate FL</td><td> $4 8 . 2 5 \pm 0 . 6 5$ </td><td> ${ \pm } 6 . 4 5 \pm { \bf 0 . 8 3 }$ </td><td> $5 5 . 2 8 \pm 1 . 3 5$ </td></tr><tr><td>Fed-MAPTrack</td><td>Ours</td><td> ${ \bf 4 8 . 9 0 \pm 0 . 6 3 }$ </td><td> $5 6 . 3 8 \pm 0 . 4 1$ </td><td> ${ \pm } { \bf 5 5 . 9 6 } \pm { \bf 0 . 5 0 }$ </td></tr></table>

• RQ4: What local optimization and localization behavior accompanies the quantitative results?

We additionally account for the prompt communication payload, distinguishing it from the efect of the aggregation design.

## 4.2. Main Results

Comparison with local and modality-agnostic training. Table 1 addresses RQ1. Fed-MAPTrack achieves mean scores of 48.90 AUC on LasHeR, 56.38 SR on VisEvent, and 55.96 F-score on DepthTrack. Compared with FedAvg, the corresponding gains are 2.35, 3.45, and 2.11 percentage points. The improvement over FedAvg occurs in each of the three matched training repetitions on all three benchmarks. Because the tracking architecture, local parameter capacity, training budget, client partition, and checkpoint rule are held fixed, these results provide evidence for the proposed aggregation design under the evaluated heterogeneous-client setting. They indicate that globally pooling valid prompt updates is less efective here than assigning appropriate sharing domains and participants to the diferent parameter groups.

The complete method also exceeds Local by 1.56 thermal AUC points, 4.11 event SR points, and 6.05 depth F-score points. However, collaboration alone does not improve every benchmark: FedAvg has lower thermal AUC than Local despite higher event and depth scores. FedProx improves the FedAvg means by 0.23 thermal AUC points, 0.18 event SR points, and 0.21 depth F-score points, but remains below Fed-MAPTrack by 2.12, 3.27, and 1.90 percentage points, respectively. With the reported coeficient $\mu = 0 . 0 1$ , proximal regularization does not match the complete aggregation design. This comparison concerns the evaluated configuration and does not establish the relative performance of all possible proximal coeficients.

Comparison with personalized federation. Fed-MAPTrack obtains higher mean scores than both FedPer and Ditto on all three benchmarks. The thermal, event, and depth gains are 1.55, 1.91, and 5.15 percentage points over FedPer and 2.44, 2.24, and 2.06 percentage points over Ditto, respectively. Personalization does not produce a uniform advantage over FedAvg: FedPer improves thermal and event performance but has lower depth performance, while Ditto has lower thermal performance and higher event and depth means. Under the evaluated settings, generic personalization is not suficient to reproduce the mean performance of the proposed sensor-aware aggregation strategy. FedPer and Ditto are evaluated by averaging client-model scores, as described above. Their model outputs and deployment configurations therefore difer from those of methods evaluated using server-side banks.

Comparison with multimodal federated baselines. Fed-MAPTrack outperforms the adapted FedMSplit-P and FED-PRIME baselines on all three benchmarks in each matched training repetition. Compared with FedMSplit-P, the gains are 2.47 thermal AUC points, 1.53 event SR points, and 1.72 depth F-score points. Compared with FED-PRIME, the corresponding gains are 2.12, 4.29, and 1.64 points. These results support group-specific aggregation in the evaluated tracking setting. Because both baselines are adapted implementations, the comparison does not characterize their performance on their original tasks.

Comparison with modality-separated federation. Relative to Independent FL, Fed-MAPTrack increases mean thermal AUC from 48.25 to 48.90 and mean depth F-score from 55.28 to 55.96, corresponding to gains of 0.65 and 0.68 percentage points. The mean event SR is slightly lower, at 56.38 compared with 56.45, a diference of 0.07 percentage points. Thus, among all nine main-table methods, the complete method obtains the highest mean thermal and depth scores, while Independent FL obtains the highest event mean. The comparison shows that retaining a common RGBshared bank is compatible with competitive tracking across the three modalities, while modality-related banks preserve their distinct aggregation domains. Since the two methods also difer in their fusion and normalization participation rules, this comparison alone does not isolate the contribution of cross-modal RGB sharing.

![](images/ed3ea5a77ddc63736f24aae8f0ec3ec8fd1912c3e88452ff1064468547b2dfd0.jpg)

![](images/fd7c5daf44d62552ea29484db2f0660e469127061fc2acfff4a6414148328042.jpg)  
Mean curves; shaded bands: sample SD across three runs  
Figure 4: Success and precision curves on LasHeR at communication round 120 using the same 245 test sequences. Curves show means across three training repetitions, and shaded bands indicate sample standard deviations. Legend values are success AUC in (a) and precision at a 20-pixel threshold in (b), expressed as percentages. Auxiliary-only bucketing and FNA are included as controlled aggregation variants.

Overall, the complete method has higher mean scores than Local, FedAvg, FedProx, FedPer, Ditto, FedMSplit-P, and FED-PRIME on all three benchmarks, but does not improve the event mean over Independent FL. The result therefore supports improved mean tracking performance over the evaluated modality-agnostic and personalized references, with benchmark-dependent diferences relative to separate federations. The sample standard deviations describe training variability under the fixed partition. Neither the reported gains nor the small event diference establishes statistical significance or equivalence.

Success and precision across thresholds. Figure 4 complements the scalar LasHeR results with the complete measured curves. The complete method achieves the highest mean AUC and precision at 20 pixels among the five displayed configurations. Its precision is 60.90%, compared with 57.93% for FedAvg, 54.27% for FNA, and 60.02% for Independent FL. The precision ordering is consistent with the AUC ordering and provides a complementary view of center localization. The curves reveal how the aggregation variants difer across the full threshold ranges. These are additional views of the same test evaluations, not independent experiments or evidence of uniform dominance at every threshold.

## 4.3. Ablation Study

To address RQ2, we vary the aggregation configuration while retaining the same twelve clients, local training budget, parameter initialization protocol, and evaluation at communication round 120. Table 2 compares four configurations. FedAvg provides the reference with global prompt aggregation. Auxiliary-only bucketing aggregates auxiliaryspecific parameters within thermal, event, and depth banks, while RGB-shared, fusion, and normalization parameters remain globally aggregated. Extended bucketing without S-A (FNA) also assigns fusion and normalization parameters to modality-specific banks, but allows RGB-only clients to contribute to these two banks according to their originating dataset. Fed-MAPTrack uses the same banks as FNA and additionally excludes RGB-only fusion and normalization updates from the server’s bucket averages. RGB-only clients continue to contribute to global RGB-shared aggregation. Auxiliary-specific aggregation excludes invalid RGB-only updates in all auxiliary-bucket configurations, including auxiliary-only bucketing and FNA.

Separating auxiliary-specific parameters. Auxiliaryonly bucketing improves the mean scores over FedAvg by 0.92 thermal AUC points, 3.53 event SR points, and 2.17 depth F-score points. The improvement is achieved without isolating RGB-shared, fusion, or normalization parameters, showing that protecting the aggregation domain of auxiliaryspecific parameters is useful under the evaluated partition. This is consistent with the design motivation that thermal, event, and depth observations provide diferent auxiliary information and should not automatically share one auxiliary prompt bank. This experiment evaluates the efect of auxiliary-specific bucketing on tracking performance, without directly measuring gradient conflict between modalities.

Extending the bucketing scope. Adding modalityspecific fusion and normalization banks without modifying

Ablation of the aggregation design. All scores are mean ± sample standard deviation over three matched training repetitions. Checkmarks indicate enabled components. RGB-shared parameters remain globally aggregated in all four configurations. FNA and Fed-MAPTrack have identical bucketing scopes and difer only in RGB-only eligibility for fusion and normalization aggregation. Bold indicates the highest mean in each metric.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Aggregation component</td><td colspan="3">Tracking performance</td></tr><tr><td>Auxiliary bucket</td><td>Fusion/Norm bucket</td><td>Selective participation</td><td>RGB-T AUC (%)</td><td>RGB-E SR (%)</td><td>RGB-D F-score (%)</td></tr><tr><td>FedAvg Auxiliary-only</td><td>X</td><td>X</td><td>X</td><td> $4 6 . 5 5 \pm 0 . 9 9$ </td><td> $5 2 . 9 3 \pm 0 . 5 9$ </td><td> $5 3 . 8 5 \pm 0 . 7 3$ </td></tr><tr><td>bucketing</td><td>√</td><td>X</td><td>X</td><td> $4 7 . 4 7 \pm 1 . 2 4$ </td><td> ${ \pm \bf 6 . 4 6 \pm 0 . 3 7 }$ </td><td> ${ \bf 5 6 . 0 2 \pm 0 . 3 2 }$ </td></tr><tr><td>Extended bucketing without S-A (FNA)</td><td>√</td><td>√</td><td>X</td><td> $4 3 . 7 0 \pm 0 . 9 5$ </td><td> $5 4 . 2 1 \pm 0 . 2 7$ </td><td> $5 5 . 3 1 \pm 0 . 5 7$ </td></tr><tr><td>Fed-MAPTrack</td><td>√</td><td>√</td><td>√</td><td> ${ \bf 4 8 . 9 0 \pm 0 . 6 3 }$ </td><td> $5 6 . 3 8 \pm 0 . 4 1$ </td><td> $5 5 . 9 6 \pm 0 . 5 0$ </td></tr></table>

Interpreting the complete design. Compared with auxiliary-only bucketing, the complete method increases mean thermal AUC by 1.43 percentage points, while its event and depth means are slightly lower by 0.08 and 0.06 percentage points. These diferences do not support uniform improvement over auxiliary-only bucketing across all modalities. The ablation instead shows that the extended sharing structure requires a compatible participation rule: S-A recovers the performance lost under FNA and provides the clearest additional benefit in thermal tracking. Bucketing defines where modality-related information is shared, while S-A determines which clients may contribute to each sharing domain. The two mechanisms form a combined aggregation strategy rather than two competing final methods.

participation eligibility reduces mean performance relative to auxiliary-only bucketing. The thermal, event, and depth scores decrease from 47.47, 56.46, and 56.02 to 43.70, 54.21, and 55.31, respectively. Thus, increasing the number of bucketed parameter groups is not suficient to improve tracking. Unlike auxiliary-specific groups, whose invalid RGB-only updates are already masked, fusion and normalization groups remain locally trainable for RGB-only clients. Under FNA, those updates enter the corresponding dataset bucket despite being learned without auxiliary observations. This distinction motivates treating the sharing domain and contributor eligibility as coupled design choices.

Adding selective participation. With the bucketing scope fixed, Fed-MAPTrack improves FNA by 5.20 thermal AUC points, 2.17 event SR points, and 0.65 depth F-score points. Figure 5 visualizes this comparison, with the same tenpercentage-point vertical span in all three panels. The lines connect the two aggregate means, while the error bars indicate sample standard deviations across training runs. The thermal and event improvements are observed in all three matched repetitions, whereas the depth improvement is positive in the mean but not in every repetition. The strongest evidence for the selective rule therefore comes from the thermal and event comparisons, with a smaller and more variable depth efect.

## 4.4. Selective Participation Analysis

To address RQ3, we examine the data available to RGB-only clients and then analyze the final parameters of FNA and Fed-MAPTrack. Client composition provides the context for sample-weighted aggregation, while the parameter analyses describe which groups difer after training. The controlled ablation establishes the performance diferences. The following analyses examine data allocation and parameter patterns that may help explain them.

Heterogeneous data allocation. Figure 6 shows the training-sample composition of the evaluated client partition. LasHeR contains 238,827 training samples assigned to sensor-equipped clients and 220,295 assigned to RGB-only clients. The corresponding counts are 32,072 and 179,990 for VisEvent and 203,709 and 6,204 for DepthTrack. The corresponding RGB-only proportions are 48.0%, 84.9%, and 3.0%. Although each benchmark supplies exactly one RGBonly client among four clients, the fraction of data assigned to that client difers substantially. Thus, an identical clientcount ratio does not imply an identical sample-weighted contribution to aggregation. These counts describe the fixed training partition, rather than the number of evaluated frames or newly acquired RGB-only sequences.

Participation at the parameter-group level. The relevant question is which parameter groups an RGB-only client should help aggregate, rather than whether that client should be removed from federated training. FNA and Fed-MAPTrack retain the same clients and assigned RGB data. In both configurations, RGB-only clients perform local training, receive the fusion and normalization banks associated with their originating dataset, and contribute their RGBshared updates to the global bank. S-A changes only the inclusion of their fusion and normalization updates in the server’s modality-specific averages. The resulting FNA-to-Fed-MAPTrack comparison therefore isolates aggregation eligibility without deleting clients, reducing the local RGB training budget, or replacing their assigned data.

The thermal and event gains support choosing contributors according to the role of the parameter group rather than treating RGB-only clients as universally unsuitable collaborators. Their RGB-shared contributions remain present in both configurations. The present ablation isolates eligibility for fusion and normalization aggregation. Quantifying the benefit of RGB-only clients’ contributions to the shared RGB bank would require a separate participation ablation.

(a) LasHeR (RGB-T)  
![](images/5fee6691faa7abce4318d6b8c13e3ba099ba965eb0effcff57018d040a39183d.jpg)

(b) VisEvent (RGB-E)  
![](images/c46d3c959965e7fb5560c314cccc61e7ae5b31bef4620a8f2d3c59d622d6f8ee.jpg)

(c) DepthTrack (RGB-D)  
![](images/4987c40f926c7698361221e1197932ca4893486c810dfc6058f43d3bf657b33e.jpg)  
Mean with sample standard deviation; identical 10-point axis spans

Figure 5: Efect of selective participation with the parameter-group bucketing scope held fixed. Blue squares and red circles show the mean scores of FNA and Fed-MAPTrack, respectively, while error bars indicate sample standard deviations across three training repetitions. The mean gains are 5.20 AUC points on LasHeR, 2.17 SR points on VisEvent, and 0.65 F-score points on DepthTrack. Each vertical axis spans ten percentage points. Connecting lines indicate changes between aggregate means, rather than individual-run trajectories  
![](images/29a1bc60c55ead2fe265ef8d75e767dca1e72798433e31447110449d7df44ae1.jpg)  
Figure 6: Training-sample composition of sensor-equipped and RGB-only clients under the evaluated partition. Totals are shown above the bars. RGB-only samples account for 48.0%, 84.9%, and 3.0% of LasHeR, VisEvent, and DepthTrack, respectively. Each benchmark has three sensor-equipped clients and one RGB-only client, while the bars show the proportions of training samples assigned to these two client types.

Interpreting the composition and performance together. The composition figure provides context for the participation experiment but is not a controlled sweep of the RGB-only ratio. VisEvent has the largest RGB-only data proportion, whereas the largest FNA-to-Fed-MAPTrack mean gain occurs on LasHeR. The ratio alone therefore does not explain the magnitude of the performance change. The benchmarks also difer in auxiliary modality, sequence distribution, and within-benchmark data imbalance. We interpret the combined evidence as support for parametergroup eligibility under the current partition, without inferring a universal exclusion threshold or a monotonic relation between RGB-only proportion and tracking improvement.

Final weight distributions across parameter groups. Figure 7 compares the empirical final weight distributions of FNA and Fed-MAPTrack at round 120. RGB-shared weights come from the global bank, while auxiliary, fusion, and normalization weights are pooled across the active thermal, event, and depth banks. The analysis includes only weight tensors, excluding biases and scalar fovea parameters. For normalization, these weights are the scale parameters. Within each group, both methods use the same 100 histogram bins covering the complete observed range. Per-run densities are normalized and then averaged across the three repetitions.

The RGB-shared and auxiliary weight distributions largely overlap, whereas diferences are more visible in fusion and normalization. In particular, FNA normalization scales are concentrated close to one, while the complete method has a broader distribution centered below one. The final distributional diferences are therefore parameter-group dependent and most apparent in groups whose RGB-only

![](images/525caedae01ac9927dfc8e39f01a32f416299d680e366385c1c14621e157ad66.jpg)

![](images/c4b96673fe4b0df34e4731d418bead6d6a8d6069abf214ad6a29b3a339e91697.jpg)  
(d) Norm scale

(c) Fusion  
![](images/f36173522f54994df296eb59ed2b4dabfa56bd5b68c1fcfaef293659b7aa25b5.jpg)

![](images/1837da608666c7b214a506aea8c5c5943f051b7915992fec2176a5e5bc0f82bb.jpg)  
Final weights; common bins across methods

Figure 7: Empirical final weight distributions of FNA and Fed-MAPTrack at communication round 120. RGB-shared weights come from the global bank, while the other groups pool weights from the three active modality banks. Curves average three per-run densities using common histogram bins covering the full observed range. Biases, scalar fovea parameters, stored initial banks, and inactive global modality banks are excluded. Panel (d) shows normalization scale weights. These are final-state distributions, not changes from initialization.

participation rule changes. Marginal histograms do not preserve parameter identity, so overlapping distributions do not imply identical tensors and broader distributions do not establish better adaptation. The next analysis retains the correspondence between the parameters of diferent modality banks.

Layer-wise separation of modality banks. To preserve the identity of corresponding parameters across modalities, Figure 8 examines the thermal, event, and depth banks within each of the twelve Transformer blocks. For parameter group � and block �, let $\theta _ { g , \ell } ^ { m }$ concatenate the corresponding parameters, including biases, in modality bank �. Their element-wise bank mean is $\bar { \theta } _ { g , \ell }$ , with $\mathcal { M } = \{ T , E , D \}$ . We compute relative bank dispersion as

$$
D _ { g , \ell } = 1 0 0 \sqrt { \frac { \sum _ { m \in \mathcal { M } } \Vert \theta _ { g , \ell } ^ { m } - \bar { \theta } _ { g , \ell } \Vert _ { 2 } ^ { 2 } } { \sum _ { m \in \mathcal { M } } \Vert \theta _ { g , \ell } ^ { m } \Vert _ { 2 } ^ { 2 } } } .\tag{14}
$$

The mean is taken element-wise over the three modality banks. The statistic is their RMS deviation from this mean divided by the pooled RMS parameter magnitude, expressed as a percentage. Corresponding tensor names and shapes are aligned before concatenation. Each repetition is analyzed separately, followed by the mean and sample standard deviation across repetitions. The analysis includes only active final modality banks, excluding stored initial states and inactive global modality banks. The RGB-shared group is omitted because it has only one global bank.

The complete method has higher fusion and normalization bank dispersion than FNA in every displayed block. For fusion, mean dispersion ranges from approximately 18.0% to 21.4%, compared with 5.2% to 6.3% for FNA. For normalization, the corresponding ranges are 1.24%–2.80% and 0.39%–0.95%. Auxiliary-specific dispersion does not increase uniformly and is lower for the complete method in most blocks. Thus, the observed diferences are not a uniform increase in separation across every parameter group. They are concentrated in fusion and normalization, consistent with the design goal of preserving modality-dependent adaptation in groups that interact with auxiliary observations.

Taken together, the ablation and parameter analyses support a coherent interpretation: sharing domains alone do not determine an efective aggregation rule, and contributor eligibility matters particularly for fusion and normalization. The complete method improves performance over FNA while exhibiting more distinct final modality banks in these groups. This correspondence supports the interpretation of groupspecific adaptation, but separation itself is not a quality metric or proof that it causes the tracking gain. Both figures describe final parameter states and do not measure changes from initialization or gradient conflict between clients.

## 4.5. Optimization and Tracking Analysis

Recorded local optimization. To address the optimization part of RQ4, Figure 9 uses the existing per-client loss logs over all 120 communication rounds. Within each repetition and round, we average the twelve clients’ mean local training losses with equal client weights. We then calculate the mean and sample standard deviation across the three repetitions. All recorded rounds are displayed without smoothing or interpolation. This diagnostic uses equal client weights and is therefore distinct from the sample-weighted server aggregation.

The complete method attains lower recorded mean local losses than FedAvg and FNA over much of training, while all three show an overall declining trajectory. This is consistent with the aggregation design supporting local optimization under the current client objectives. Because the values are collected during local training, they are not losses of one server model evaluated on a common held-out set. Moreover, the available benchmark evaluations concern round 120 only. We therefore do not interpret the curves as test-performance convergence, time to a target accuracy, or evidence of improved generalization by themselves.

Qualitative tracking comparison. To address the localization part of RQ4, Figure 10 compares saved predictions of FedAvg, Independent FL, and Fed-MAPTrack on sequences from the three benchmarks. The displayed sequences are basketboy from LasHeR, personFootball from VisEvent, and

Transformer block

![](images/9c7aae4ebbf05bcd26cb8b2f981ca4e671fd4b8f831edf53c1b9f55ab5a1cb85.jpg)

![](images/d3d71aeedb96e6b92adf7acceeaca67598f39f7f6f5f6edf25f3160c0bdcabe5.jpg)  
Active thermal, event and depth banks; mean with sample SD

![](images/cf563bcdcf485d6a1adeb76e843ca6ac7fda5e7ea5e5da22d32a8d30c88e2ee6.jpg)

Figure 8: Layer-wise separation of final thermal, event, and depth parameter banks, measured by the relative dispersion in Eq. 14. All parameters, including biases, are included within each group and Transformer block. Bars and error bars show the mean and sample standard deviation across three training repetitions. The RGB-shared group has only one bank and is omitted. Each pane starts at zero but uses its own labeled range. The statistic describes final bank separation, not tracking accuracy or changes from initialization.  
![](images/1f38ba550a040dd42c5f7641e83c3df2cc66855c5b0bdfef51015537f122f202.jpg)  
Figure 9: Recorded local training loss over 120 communication rounds. Within each repetition, the twelve clients’ mean local losses are averaged equally in each round. Curves and shaded bands then show the mean and sample standard deviation across three repetitions. All recorded rounds are plotted without smoothing or interpolation. The figure describes local optimization, not intermediate-round benchmark performance.

backpack\_indoor from DepthTrack. Each row presents temporally ordered frames from one sequence, and all methods use the same reference image and crop within each comparison. The enlarged insets contain the ground truth and all displayed predictions, allowing diferences in location and target extent to be inspected under a common spatial scale. The examples come from one fixed training repetition and supplement, rather than replace, the repeated benchmark evaluations.

In the event sequence, intermediate examples show Fed-MAPTrack remaining near the annotated target while the compared methods localize another person. The thermal sequence illustrates diferences in target location and predicted extent, together with a later example in which the complete method also drifts. The depth sequence includes dificult frames in which the displayed trackers deviate from the annotated target. These examples show both the potential benefit and the remaining failure modes of the aggregation strategy. They do not establish a benchmark-wide recovery rate or a guarantee of robustness to a particular tracking attribute. The quantitative comparisons remain those reported in Tables 1 and 2.

Box-derived localization comparison. Figure 11 provides a closer comparison between FNA and Fed-MAPTrack on basketboy. For each method, the displayed localization map is constructed from its saved predicted bounding box using the same elliptical Gaussian encoding. The Gaussian is centered at the predicted box center, and its horizontal and vertical standard deviations are one quarter of the predicted box width and height. Both methods use the same crop coordinates and color scale, with ground-truth boxes overlaid for spatial reference. This makes diferences in predicted location and extent visible without altering the saved box predictions.

The middle displayed example illustrates a diference in estimated target extent between FNA and the complete method, whereas other examples show much closer predictions. Because the maps are derived analytically from predicted boxes, their spatial concentration follows the box geometry. They are not measured attention, activation, or confidence maps and are not used as evidence for improved internal attention concentration. Their role is to complement the tracking-box comparison with an explicit visualization of localization outputs.

(a) RGB-T  
LasHeR / basketboy  
![](images/89a73663f84291540327cf06d476faac057380784ae1240bd71f504d0a757f57.jpg)  
Figure 10: Qualitative RGB-X tracking comparisons on (a) LasHeR, (b) VisEvent, and (c) DepthTrack. Green, blue, purple, and red solid boxes indicate ground truth, FedAvg, Independent FL, and Fed-MAPTrack, respectively. Each row shows temporally ordered frames from one sequence. Insets magnify a common region containing all displayed predictions and the ground truth, with gray corner markers identifying the corresponding regions in the full images. The examples include both accurate localization and tracking failures.

## 4.6. Communication Analysis

During federated optimization, clients exchange only prompt-related parameters, while the frozen backbone and prediction head remain local. The experiments use a local federated simulation rather than a measured network deployment, so we account for FP32 tensor payloads explicitly. A full local prompt view contains 839,628 parameters, corresponding to 3,358,512 bytes. The client implementation constructs prompt-state dictionaries for all four groups and uses a separate validity mask to identify eligible updates. An RGBonly client’s valid RGB-shared, fusion, and normalization groups contain 175,212 parameters, or 700,848 bytes, but marking its auxiliary tensors invalid does not itself remove them from that dictionary.

If full prompt-state dictionaries are serialized in both directions for all twelve clients, the tensor-only payload is 80.60 MB per round and 9.673 GB over 120 rounds. If a transport implementation additionally removes invalid RGBonly auxiliary tensors before upload while retaining full prompt downloads, the corresponding estimate becomes 72.63 MB per round and 8.716 GB over 120 rounds. The latter is a mask-aware transport estimate, not measured trafic or an automatic consequence of the aggregation mask. Both calculations use decimal units and exclude serialization metadata and network protocol overhead.

The core aggregation variants share the same local prompt-state dimensions. S-A filters RGB-only fusion and normalization updates at the server rather than removing those tensors from the client state. Bucketing defines sharing domains, while S-A determines eligible contributors. Neither mechanism provides additional communication compression.

![](images/d79d58f3fcec10be638955197784b5ce24f685a839cd3a8f9146e67c1b00f92c.jpg)  
Figure 11: Box-derived localization comparison between FNA and Fed-MAPTrack on basketboy. The first column shows reference RGB crops, while the remaining columns visualize saved box predictions using the same elliptical Gaussian encoding. Each Gaussian is centered on the predicted box, with horizontal and vertical standard deviations equal to one quarter of its width and height, respectively. Green boxes indicate ground truth in the common crop coordinates.

The smaller parameter payload relative to full-model exchange comes from prompt tuning, whereas the controlled comparisons evaluate the aggregation design. This accounting does not establish latency, energy, wall-clock training, or deployment improvements.

## 5. Discussion and Limitations

Joint design of bucketing and selective participation. The central finding is that the sharing domain and contributor eligibility must be considered together. Extending bucketing to fusion and normalization without S-A reduces performance, whereas adding S-A improves FNA and yields higher means than FedAvg on all three evaluated benchmarks. The complete method also achieves higher mean scores than Local, FedProx, FedPer, Ditto, FedMSplit-P, and FED-PRIME on all three benchmarks, while Independent FL has a slightly higher event mean. Compared with auxiliaryonly bucketing, the complete method provides the clearest additional benefit on LasHeR, while its event and depth means are slightly lower by less than 0.1 points. Final-parameter analyses reveal corresponding group-dependent diferences, but do not establish a causal relation between bank separation and accuracy.

Evaluation scope. The experiments use twelve simulated clients and three training repetitions under one fixed partition. Modality, dataset, and RGB-only data proportion vary together, so the present results do not isolate a universal efect of the RGB-only ratio or establish generality across partitions. Communication figures are tensor-payload calculations rather than measured edge-device latency or energy savings. Keeping raw videos local does not by itself prevent information leakage through exchanged parameters. Formal privacy protection is beyond the scope of this evaluation.

Dynamic sensor conditions. S-A uses client-level sensor availability rather than a learned estimate of auxiliary observation quality. Sample-level masking is included in training, but noisy, delayed, or persistently failing sensors may require finer eligibility decisions. Observation-quality estimation and dynamic sensor availability are therefore natural directions for extending the participation rule.

## 6. Conclusion and Future Work

We presented Fed-MAPTrack, a federated RGB-X tracking framework for clients with heterogeneous auxiliary sensors and RGB-only observations. The framework adapts prompt-related parameters on a frozen backbone and prediction head, keeping raw training videos local. Its unified aggregation strategy shares RGB parameters globally and combines modality-specific auxiliary, fusion, and normalization banks with group-specific contributor eligibility. RGBonly clients retain their shared RGB contributions, while their fusion and normalization updates are excluded from modalityspecific server averages. Experiments on LasHeR, VisEvent, and DepthTrack show that Fed-MAPTrack achieves higher mean scores than FedAvg under the evaluated partition. The complete method also achieves higher mean scores than Local, FedProx, FedPer, Ditto, FedMSplit-P, and FED-PRIME on each benchmark. It attains the highest thermal AUC and depth F-score among the evaluated methods, while its event SR remains 0.07 points below Independent FL. Controlled ablations show that extending the bucketing scope alone is insuficient and that a compatible participation rule is essential to the complete design.

Future work will examine observation-quality-aware participation, dynamic sensor availability, and finer parametersharing rules. Evaluation across additional partitions and data proportions, together with measured edge-device communication and runtime, will further assess generality and deployment suitability.

## References

[1] Arivazhagan, M.G., Aggarwal, V., Singh, A.K., Choudhary, S., 2019. Federated learning with personalization layers. arXiv preprint arXiv:1912.00818 URL: https://arxiv.org/abs/1912.00818, arXiv:1912.00818.

[2] Bertinetto, L., Valmadre, J., Henriques, J.F., Vedaldi, A., Torr, P.H.S., 2016. Fully-convolutional siamese networks for object tracking,

in: Computer Vision – ECCV 2016 Workshops, Springer. pp. 850– 865. URL: https://doi.org/10.1007/978- 3- 319- 48881- 3\_56, doi:10.1007/978-3-319-48881-3\_56.

[3] Che, L., Wang, J., Zhou, Y., Ma, F., 2023. Multimodal federated learning: A survey. Sensors 23, 6986. URL: https://doi.org/10.339 0/s23156986, doi:10.3390/s23156986.

[4] Chen, J., Zhang, A., 2022. FedMSplit: Correlation-adaptive federated multi-task learning across multimodal split networks, in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Association for Computing Machinery. pp. 87–96. URL: https://doi.org/10.1145/3534678.3539384, doi:10.1145/3534678.3539 384.

[5] Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., Hu, S., Chen, Y., Chan, C.M., Chen, W., Yi, J., Zhao, W., Wang, X., Liu, Z., Zheng, H., Chen, J., Liu, Y., Tang, J., Li, J., Sun, M., 2023. Parameter-eficient finetuning of large-scale pre-trained language models. Nature Machine Intelligence 5, 220–235. URL: https://doi.org/10.1038/s42256-023 -00626-4, doi:10.1038/s42256-023-00626-4.

[6] Feng, M., Su, J., 2024. RGBT tracking: A comprehensive review. Information Fusion 110, 102492. URL: https://doi.org/10.1016/j. inffus.2024.102492, doi:10.1016/j.inffus.2024.102492.

[7] Feng, T., Bose, D., Zhang, T., Hebbar, R., Ramakrishna, A., Gupta, R., Zhang, M., Avestimehr, S., Narayanan, S., 2023. FedMultimodal: A benchmark for multimodal federated learning, in: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4035–4045. URL: https://doi.org/10.1145/3580305.35 99825, doi:10.1145/3580305.3599825.

[8] Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N., 2022. Visual prompt tuning, in: Computer Vision – ECCV 2022, Springer. pp. 709–727. URL: https://doi.org/10.1007/978-3 -031-19827-4\_41, doi:10.1007/978-3-031-19827-4\_41.

[9] Li, B., Yan, J., Wu, W., Zhu, Z., Hu, X., 2018. High performance visual tracking with siamese region proposal network, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8971–8980. URL: https://openaccess.thecvf.com/content\_cv pr\_2018/html/Li\_High\_Performance\_Visual\_CVPR\_2018\_paper.html, doi:10.1109/CVPR.2018.00935.

[10] Li, C., Liang, X., Lu, Y., Zhao, N., Tang, J., 2022. LasHeR: A largescale high-diversity benchmark for RGBT tracking. IEEE Transactions on Image Processing 31, 392–404. URL: https://doi.org/10.1109/TI P.2021.3130533, doi:10.1109/TIP.2021.3130533.

[11] Li, T., Hu, S., Beirami, A., Smith, V., 2021. Ditto: Fair and robust federated learning through personalization, in: Proceedings of the 38th International Conference on Machine Learning, PMLR. pp. 6357– 6368. URL: https://proceedings.mlr.press/v139/li21h.html.

[12] Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V., 2020. Federated optimization in heterogeneous networks, in: Proceedings of Machine Learning and Systems, pp. 429–450. URL: https://proceedings.mlsys.org/paper\_files/paper/2020/hash/1f5fe8 3998a09396ebe6477d9475ba0c-Abstract.html.

[13] McMahan, B., Moore, E., Ramage, D., Hampson, S., Agüera y Arcas, B., 2017. Communication-eficient learning of deep networks from decentralized data, in: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 1273– 1282. URL: https://proceedings.mlr.press/v54/mcmahan17a.html.

[14] Phung, D.M.H., Huynh, K., Tran, N.L., Le, T., Bui, T., Pham, Q.H., Nguyen, A., 2025. Federated prompt-tuning with heterogeneous and incomplete multimodal client data, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3936–3946. URL: https://openaccess.thecvf.com/content/ICCV2025/html/Phung\_Federa ted\_Prompt-Tuning\_with\_Heterogeneous\_and\_Incomplete\_Multimodal\_C lient\_Data\_ICCV\_2025\_paper.html.

[15] Wang, X., Li, J., Chen, L., Tang, L., Huang, X., Wang, Y., Zhang, Y., 2024. VisEvent: Reliable object tracking via collaboration of frame and event flows. IEEE Transactions on Cybernetics 54, 1997– 2010. URL: https://doi.org/10.1109/TCYB.2023.3318601, doi:10.1109/TCYB.2023.3318601.

[16] Yan, B., Peng, H., Fu, J., Wang, D., Lu, H., 2021a. Learning spatio-temporal transformer for visual tracking, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10448– 10457. URL: https://openaccess.thecvf.com/content/ICCV2021/html/ Yan\_Learning\_Spatio-Temporal\_Transformer\_for\_Visual\_Tracking\_IC CV\_2021\_paper.html.

[17] Yan, S., Yang, J., Käpylä, J., Zheng, F., Leonardis, A., Kämäräinen, J.K., 2021b. DepthTrack: Unveiling the power of RGBD tracking, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10725–10733. URL: https://openaccess.thecvf.com/cont ent/ICCV2021/html/Yan\_DepthTrack\_Unveiling\_the\_Power\_of\_RGBD\_Tra cking\_ICCV\_2021\_paper.html.

[18] Yang, F.E., Wang, C.Y., Wang, Y.C.F., 2023. Eficient model personalization in federated learning via client-specific prompt generation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19159–19168. URL: https://openaccess.thecvf.com/cont ent/ICCV2023/html/Yang\_Efficient\_Model\_Personalization\_in\_Federa ted\_Learning\_via\_Client-Specific\_Prompt\_Generation\_ICCV\_2023\_pa per.html.

[19] Yang, J., Li, Z., Zheng, F., Leonardis, A., Song, J., 2022. Prompting for multi-modal tracking, in: Proceedings of the 30th ACM International Conference on Multimedia, pp. 3492–3500. URL: https://doi.org/ 10.1145/3503161.3547839, doi:10.1145/3503161.3547839.

[20] Ye, B., Chang, H., Ma, B., Shan, S., Chen, X., 2022. Joint feature learning and relation modeling for tracking: A one-stream framework, in: Computer Vision – ECCV 2022, Springer. pp. 341–357. URL: https://doi.org/10.1007/978-3-031-20047-2\_20, doi:10.1007/978-3-0 31-20047-2\_20.

[21] Yu, Q., Liu, Y., Wang, Y., Xu, K., Liu, J., 2023. Multimodal federated learning via contrastive representation ensemble, in: The Eleventh International Conference on Learning Representations. URL: https: //openreview.net/forum?id=Hnk1WRMAYqg.

[22] Zhu, J., Lai, S., Chen, X., Wang, D., Lu, H., 2023. Visual prompt multi-modal tracking, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9516–9526. URL: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu\_ Visual\_Prompt\_Multi- Modal\_Tracking\_CVPR\_2023\_paper.html, doi:10.1109/CVPR52729.2023.00918.