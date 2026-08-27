# V-Link: Recovering Lost Visual Representations in Action DiT for Vision-Language-Action Models

Yehao Lu<sup>1,4</sup>, Jiarui Yang<sup>2,4</sup>, Yuning Su<sup>3,4</sup>, Yufeng Xie<sup>4</sup>, Yu Zhong<sup>4</sup>, Yazhou Zhang<sup>4</sup>, Haiyu Lan<sup>4</sup>, Kaixiang Lu<sup>4</sup>, Peiwen Lin<sup>4</sup>, Chuang Wang<sup>4</sup>, Zequn Qin<sup>1</sup>, Enyu Li<sup>4†</sup>, Xi Li<sup>1†</sup> <sup>1</sup>Zhejiang University, <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>3</sup>Simon Fraser University, <sup>4</sup>AGIBOT

Abstract— Vision-language-action (VLA) models provide a scalable path toward generalist robotic manipulation by integrating visual perception, language understanding, and continuous action control. However, we reveal a critical limitation of VLA architectures: the action expert has limited access to the 3D geometric and 2D semantic information available in VLM features. This accessibility gap weakens perceptual grounding and limits performance on fine-grained robotic manipulation. To address this issue, we propose V-Link, which explicitly recovers visual representations during the vision-language (VL) to action (A) feature transfer. Specifically, V-Link learns complementary Spatial and Semantic Query representations within the VLM and injects them into Action DiT through asymmetric pathways. Semantic Queries complement the original VLM image tokens, whereas Spatial Queries provide dedicated geometric conditioning for spatially grounded action generation. Across LIBERO, LIBERO-Plus, and RoboTwin 2.0, our V-Link improves the average success rate over base model GR00T N1.6 by +1.9%, +31.2%, and +18.8%, respectively. On the AGIBOT A3 Ultra, V-Link further achieves gains of +20% and +24% on two real-world humanoid tasks.

## I. INTRODUCTION

Vision-language-action (VLA) models have emerged as a foundational paradigm for robotic manipulation by mapping visual observations and language instructions into continuous action spaces. Among them, dual-system architectures couple a vision-language model (VLM) for rich scene understanding and task reasoning with a lightweight action expert for high frequency robot control. Central to this design is visionlanguage (VL) to action (A) feature transfer, which determines what information from the VLM is available to the action expert. In representative models such as NVIDIA GR00T N1.6 [1], only the VLM’s final-layer features are transferred to the action expert. However, a key question remains: does this final-layer transfer provide the action expert with sufficient access to the 2D and 3D perceptual cues required for precise manipulation?

To answer this question, we employ a diagnostic protocol inspired by Spatial Forcing [2] to measure representation accessibility. Specifically, we freeze the GR00T N1.6 VLM and Action DiT features, attach lightweight depth-estimation and semantic-segmentation heads to these features, and optimize only the task heads under identical supervision. With the underlying representations frozen, test-set performance serves as a controlled measure of the accessibility of 3D geometric and 2D semantic information. As shown in Fig. 2 (left), the diagnostic heads achieve substantially better results with VLM features than with Action DiT features: depth MAE increases from 0.015 to 0.071, while segmentation mIoU decreases from 0.665 to 0.290. These results reveal a pronounced visual representation accessibility gap across final-layer VL-to-A transfer.

![](images/36bb65c38a6bb865789a6a7fdb25904ec1109e82393b6d197d25d41f62dd39ed.jpg)  
Fig. 1: Key insight. GR00T N1.6 exhibits a visual representation accessibility bottleneck during VL-to-A feature transfer. Our V-Link addresses this bottleneck by learning complementary Spatial and Semantic Query representations and injecting them into Action DiT. The resulting Action DiT features support substantially better depth estimation and semantic segmentation, providing qualitative evidence of visual representation recovery.

We hypothesize that the interaction between highly entangled final-layer VLM features and action-only supervision encourages Action DiT to exploit an optimization shortcut, prioritizing semantic cues that rapidly reduce the action loss while underutilizing structurally critical 3D spatial cues. The qualitative results in Fig. 1 provide further support for this hypothesis: segmentation predictions decoded from GR00T Action DiT features remain coarse but recognizable, whereas the corresponding depth estimates recover little meaningful 3D structure. Such asymmetric feature utilization can lead to a local optimum with insufficient 3D spatial grounding for precise robot manipulation.

![](images/a28ad77711c0ab465d2cd185de1f3f94a84a08a9d6f44f82fdb453cf32e2086f.jpg)  
Fig. 2: Diagnostic evaluation protocol. We freeze the VLM and Action DiT features and train only lightweight task heads for depth estimation and semantic segmentation. Test-set performance serves as a controlled measure of visual representation accessibility. Left: Compared with VLM features, GR00T N1.6 Action DiT features exhibit a pronounced accessibility gap. Right: V-Link substantially narrows this gap in Action DiT on both diagnostic tasks.

Recent efforts to improve VLA perception, such as Spatial Forcing [2] and EVO-0 [3], primarily leverage representations from 3D foundation models to enhance spatial reasoning within the VLM. However, richer VLM representations do not guarantee that the action expert can access the same perceptual information after VL-to-A transfer. Complementary work such as VLA-Adapter [4] investigates efficient connections between the VLM and Action DiT, yet does not explicitly address the accessibility of transferred visual representations within the action expert.

Motivated by these observations, we propose V-Link, a representation-recovery framework that improves the accessibility of 3D geometric and 2D semantic information in Action DiT. As illustrated in Fig. 1, V-Link introduces learnable Spatial and Semantic Queries into the VLM to extract complementary perceptual representations. A tailored causal attention mask isolates the two query sets without disrupting the original multimodal computation, while training-only depth and segmentation heads specialize them for geometry and semantics, respectively. V-Link then injects the learned query representations into Action DiT through asymmetric pathways. Semantic Queries complement the original VLM image tokens via parallel cross-attention, whereas Spatial Queries provide a dedicated geometry-conditioning pathway. This design mitigates reliance on semantic shortcuts and makes both types of perceptual information directly accessible during action generation, as demonstrated by the representation diagnostics in Fig. 2 (right).

Extensive experiments across simulation benchmarks and real-world humanoid manipulation demonstrate consistent gains over GR00T N1.6. V-Link achieves average success-rate gains of +31.2% on LIBERO-Plus, +18.8% on RoboTwin 2.0, and +1.9% on LIBERO. On the AGIBOT A3 Ultra, it achieves 98% and 94% success on autonomous power-on and power-off, outperforming GR00T N1.6 by +20% and +24%, respectively. These improvements incur only 1.58 ms of additional inference latency, from 43.21 ms to 44.79 ms, demonstrating substantial performance gains with minimal computational overhead.

Our contributions are summarized as follows:

• We reveal a previously overlooked visual representation accessibility bottleneck in dual-system VLAs: finallayer VL-to-A transfer leaves the action expert with limited access to the 3D geometric and 2D semantic information required for precise manipulation.

• We propose V-Link, a representation-recovery framework that learns complementary Spatial and Semantic Query representations within the VLM and injects them into Action DiT through asymmetric pathways, improving access to both forms of perceptual information during action generation.

• Extensive experiments across three simulation benchmarks and two real-world humanoid tasks demonstrate consistent improvements over GR00T N1.6, while adding only 1.58 ms of inference latency.

## II. RELATED WORK

## A. VLA Architectures

Existing VLA models can be broadly categorized into unified and dual-system architectures. Unified methods such as UniVLA [5] and Cosmos Policy [6] model vision, language, and action within a shared transformer, while $\pi _ { 0 . 5 }$ [7] and LingBot-VLA [8] use layer-wise attention to couple VLM and action representations. Dual-system designs pair a VLM with a separate action expert, decoupling multimodal reasoning from robot control. A representative model is GR00T N1.6 [1], which transfers the VLM’s final-layer features to Action DiT. Complementary studies improve efficiency through action tokenization [9], visual-token caching [10], model scheduling [11], or multi-frequency feature delivery [12], while VLA-Adapter [4] explores lightweight connections between the VLM and Action DiT. Prior work primarily focuses on architectural integration or execution efficiency. Instead, V-Link examines the accessibility of 3D geometric and 2D semantic representations within the action expert and explicitly recovers them during VL-to-A transfer.

![](images/1e4b13cabca360acccc4019c21bac62db4ffb0dcdd517b0d72b92bf474b4667e.jpg)  
Fig. 3: The overall pipeline of V-Link. V-Link introduces learnable Spatial and Semantic Queries into the pretrained VLM. A causal attention mask enables them to independently extract geometric and semantic cues from deep VLM features without disrupting the native multimodal token stream. Lightweight depth and semantic heads provide explicit supervision to disentangle these cues within the VLM latent space and are removed at inference. The resulting representations are injected into Action DiT through asymmetric pathways: Semantic Queries complement the VLM image tokens, whereas Spatial Queries form a mandatory geometry-conditioning branch that compels Action DiT to explicitly encode depth-aware representations rather than rely on semantic shortcuts.

## B. Representation Enhancement

Extensive research has focused on enhancing the visual perception of VLA. The vast majority of them prioritize improving the feature representation of the underlying VLM. For instance, GEAR-VLA [13] and SpatialVLA [14] incorporate trainable 3D spatial modules and align them with VLM representations. EVO-0 [3] directly integrates 3D geometric features from VGGT [15] into the VLM. Spatial Forcing [2] further leverages VGGT features to distill VLM intermediate embeddings, eliminating the dependency on 3D foundation models at inference time. In parallel, a few works also aim to bolster the spatial reasoning of the action expert. PointVLA [16] employs lightweight modules to inject 3D point cloud features. ABot-M0 [17] incorporates 3D features directly from 3D foundation models. In contrast to these approaches, our work aims to fundamentally optimize the VL to A feature transfer and mitigate the visual representation degradation during this process.

## III. METHOD

## A. Overview

The overall pipeline is illustrated in Fig. 3. We build upon GR00T N1.6 [1] as our base model. V-Link jointly learns complementary geometric and semantic representations within the VLM and injects them asymmetrically into Action DiT. Specifically, we introduce two sets of learnable tokens, termed Spatial Queries and Semantic Queries.

A tailored causal attention mask isolates the two query sets while preserving the native image-text computation, allowing each set to aggregate relevant perceptual cues independently. Lightweight depth-estimation and semanticsegmentation heads provide training-only supervision, specializing the Spatial and Semantic Queries for geometry and semantics, respectively.

The learned query representations are then injected into Action DiT using an asymmetric attention design. The robotstate and noisy-action tokens serve as queries in two parallel cross-attention operations, with the VLM image tokens and Semantic Queries providing the respective keys and values. The resulting outputs are merged and added to the layer input through a residual connection, enriching the action features with complementary semantic context. The fused features subsequently undergo cross-attention with Spatial Queries, providing dedicated geometric conditioning for spatially grounded action generation. Following the alternatingattention design of GR00T N1.6, both query interactions are applied only to the image cross-attention layers.

## B. VLM Representations Decoupling

When processing entangled VLM representations, Action DiT may preferentially exploit semantic cues over 3D geometry under action-only supervision. To counter this imbalance, we introduce learnable Spatial and Semantic Queries into the VLM input to separately capture geometric and semantic information:

$$
\mathbf { E } = [ \mathbf { E } _ { t } ; \mathbf { E } _ { o } ; \mathbf { E } _ { d } ; \mathbf { E } _ { s } ] ,\tag{1}
$$

where $\mathbf { E } _ { t }$ and $\mathbf { E } _ { o }$ denote the language and multi-view image tokens, while $\mathbf { E } _ { d }$ and ${ \bf E } _ { s }$ denote the Spatial and Semantic

Query tokens, respectively. Under the tailored causal attention mask $\mathbf { M } _ { q } ,$ , the two query sets independently attend to the VLM image and text tokens via self-attention. Across successive Transformer blocks, the query representations progressively aggregate rich visual cues:

$$
[ \mathbf { E } _ { t } ^ { \prime } , \mathbf { E } _ { o } ^ { \prime } , \mathbf { E } _ { d } ^ { \prime } , \mathbf { E } _ { s } ^ { \prime } ] = \mathcal { R } \big ( \Phi _ { \mathrm { V L M } } ( \mathbf { E } ; \mathbf { M } _ { q } ) \big ) ,\tag{2}
$$

where $\Phi _ { \mathrm { V L M } }$ denotes the VLM, and R denotes Qwen’s final RMSNorm layer. $\mathbf { E } _ { t } ^ { \prime } , \mathbf { E } _ { o } ^ { \prime } , \mathbf { E } _ { d } ^ { \prime } , \mathbf { E } _ { s } ^ { \prime }$ represent the final-layer VLM outputs corresponding to the image and text tokens, Spatial Query, and Semantic Query, respectively.

To explicitly encode depth information into the Spatial Query, we design a depth head $\Phi _ { \mathrm { D e p t h } }$ to estimate dense depth maps $\hat { \mathbf { D } } \in \mathbb { R } ^ { H \times \bar { W } \times V }$ for multiple camera views:

$$
\hat { \mathbf { D } } = \sigma _ { + } ( \Phi _ { \mathrm { D e p t h } } ( \mathbf { E } _ { d } ^ { \prime } ) ) ,\tag{3}
$$

where $\sigma _ { + } ( x ) = \log ( 1 + e ^ { x } )$ denotes the softplus function, which constrains the predicted metric depth to be positive. Our depth head comprises a projection layer ${ \mathcal { P } } _ { d } ,$ implemented as $\mathrm { L N } ( 2 0 4 8 )  \mathrm { L i n e a r } ( 2 0 4 8 , 2 5 6 )  \mathrm { G E L U } $ Dropout, two per-view transformer encoder layers $\mathcal { T } _ { d } .$ , and a convolutional dense decoder $\mathcal { C } _ { d }$ for depth prediction. With approximately 2.3M parameters, the depth head is intentionally lightweight, encouraging depth-relevant information to be captured by the Spatial Queries rather than the prediction head itself. We supervise the predictions using ground-truth depth maps, with the depth loss defined as:

$$
\begin{array} { r } { \mathcal { L } _ { d e p t h } = \mathcal { L } _ { \mathrm { s m } } + \lambda _ { 1 } ^ { d } \mathcal { L } _ { \mathrm { r e l } } + \lambda _ { 2 } ^ { d } \mathcal { L } _ { \mathrm { p a t c h } } , } \end{array}\tag{4}
$$

where $\mathcal { L } _ { \mathrm { s m } }$ denotes the smooth- ${ \cal - L } _ { 1 }$ loss for global absolutescale depth supervision. $\mathcal { L } _ { \mathrm { r e l } }$ is the absolute-relative loss, which emphasizes near-field objects, robot arms, and manipulation regions while preventing distant backgrounds from dominating optimization. $\mathcal { L } _ { \mathrm { p a t c h } }$ is a top-k hard-patch loss that mines locally difficult regions to avoid gradient dilution by abundant easy-background pixels. $\lambda _ { 1 } ^ { d } = 0 . 1$ and $\lambda _ { 2 } ^ { d } = 0 . 5$ are two constant coefficients.

The semantic head $\Phi _ { \mathrm { S e g } }$ follows the same overall architecture as the depth head, differing only in its input queries, output dimensionality, and activation function:

$$
\hat { \mathbf { Y } } = \mathrm { S o f t m a x } ( \Phi _ { \mathrm { S e g } } ( \mathbf { E } _ { s } ^ { \prime } ) ) ,\tag{5}
$$

where $\hat { Y } \in \mathbb { R } ^ { H \times W \times C \times V }$ denotes the predicted 2D semantic segmentation logits, with C and V representing the numbers of semantic classes and camera views, respectively. For the semantic head, we combine cross-entropy loss $\mathcal { L } _ { C E }$ with soft dice loss $\mathcal { L } _ { D i c e } . \ \mathcal { L } _ { C E }$ provides pixel-wise class supervision, whereas $\mathcal { L } _ { D i c e }$ directly optimizes region overlap and mitigates the bias induced by background dominance and class imbalance. The overall segmentation loss $\mathcal { L } _ { s e g }$ is formulated as:

$$
\mathcal { L } _ { s e g } = \mathcal { L } _ { C E } + \lambda ^ { s } \mathcal { L } _ { D i c e }\tag{6}
$$

where $\lambda ^ { s } = 0 . 5$ is a constant coefficient.

To stabilize optimization with randomly initialized Spatial and Semantic Queries, we first warm up the VLM LoRA parameters, both query sets, and their task heads using the same hyperparameters as subsequent joint training. After warm-up, all trainable components are jointly optimized. At inference, the auxiliary heads are removed, and no depth or segmentation annotations are required.

![](images/49f5b869ec545889b1291f84d1c011cbf77f86c5bba51c856f8e8d21b6c0b52d.jpg)  
Fig. 4: t-SNE on RoboTwin2.0 rollouts. We randomly sample 5,000 observations and extract the final-layer VLM outputs for image tokens, Spatial Queries, and Semantic Queries. Each token set is mean-pooled before projection. The separated distributions qualitatively indicate that the two query sets learn distinct representations.

## C. Visual Representation Recovery in Action DiT

GR00T N1.6 adopts a 32-layer AlternateVLDiT whose attention pattern comprises 8 repetitions of a 4-layer cycle:

$$
\begin{array} { r } { A _ { \ell } = \left\{ \begin{array} { l l } { \mathrm { C A } _ { \mathrm { n o n - i m g } } , } & { \ell = 4 k , } \\ { \mathrm { S A } _ { \mathrm { s t a t e / a c t i o n } } , } & { \ell = 4 k + 1 , } \\ { \mathrm { C A } _ { \mathrm { i m g } } , } & { \ell = 4 k + 2 , } \\ { \mathrm { S A } _ { \mathrm { s t a t e / a c t i o n } } , } & { \ell = 4 k + 3 , } \end{array} \right. \ k \in [ 0 , 7 ] , } \end{array}\tag{7}
$$

where $\mathrm { C A } _ { \mathrm { n o n - i m g } }$ and $\mathrm { C A } _ { \mathrm { i m g } }$ denote cross-attention to valid non-image and image tokens, respectively, and $\mathrm { S A } _ { \mathrm { s t a t e / } }$ action denotes self-attention over state and action tokens. Accordingly, the Spatial and Semantic Query representations are injected only at layers satisfying $\ell = 4 k + 2$ . This confines V-Link to the image cross-attention layers while preserving the original alternating-attention schedule.

Our diagnostic analysis shows that the base model Action DiT retains coarse semantic cues but has limited access to 3D geometry. Motivated by this observation, we adopt an asymmetric injection design as illustrated in Fig. 3. Semantic Queries complement the original VLM image tokens through parallel cross-attention, whereas Spatial Queries subsequently provide dedicated geometric conditioning to the fused action features. Formally, the robot state s and noisy action $\mathbf { x } _ { t }$ are encoded into input tokens $\mathbf { H } _ { 0 }$ for the Action DiT by StateEnc and ActionEnc, respectively:

$$
\mathbf { H } _ { 0 } = [ \mathrm { S t a t e E n c } ( \mathbf { s } ) ; \mathrm { A c t i o n E n c } ( \mathbf { x } _ { t } , t ) ] ,\tag{8}
$$

At layer $l , \ \mathbf { H } _ { l }$ serves as the query for two parallel crossattention, with the VLM image tokens $\mathbf { E } _ { o } ^ { \prime }$ and Semantic

Query ${ \bf E } _ { s } ^ { \prime }$ independently providing the keys and values:

$$
\mathbf { N } _ { l } = \mathrm { C A } _ { \mathbf { V L M } } ( \mathbf { H } _ { l } , \mathbf { E } _ { o } ^ { \prime } ) ,\tag{9}
$$

$$
\mathbf { S } _ { l } = \mathrm { C A } _ { \mathbf { S e m a n t i c } } ( \mathbf { H } _ { l } , \mathbf { E } _ { s } ^ { \prime } ) ,\tag{10}
$$

The two cross-attention outputs are concatenated and projected by Fuse. A residual connection then adds the projected features to H<sub>l</sub>, yielding the semantically enriched representation Z :

$$
\mathbf { Z } _ { l } = \mathbf { H } _ { l } + \mathrm { F u s e } ( \mathbf { N } _ { l } , g _ { l } \mathbf { S } _ { l } ) ,\tag{11}
$$

where $g _ { l } \in \ [ 0 , 1 ]$ is a learnable gating parameter. $Z _ { l }$ then serves as the query in a spatial cross-attention operation, with the Spatial Queries $\mathbf { E } _ { d } ^ { \prime }$ providing the keys and values. This operation explicitly conditions the action features on 3D geometry:

$$
\mathbf { H } _ { l } ^ { \prime } = \mathbf { Z } _ { l } + \mathrm { C A } _ { \mathbf { S p a t i a l } } ( \mathbf { Z } _ { l } , \mathbf { E } _ { d } ^ { \prime } ) ,\tag{12}
$$

Finally, the spatially conditioned feature $H _ { l } ^ { \prime }$ is processed by an FFN to produce the output of the current layer ${ \bf H } _ { l + 1 }$

$$
\mathbf { H } _ { l + 1 } = \mathbf { H } _ { l } ^ { \prime } + \mathrm { F F N } ( \mathbf { H } _ { l } ^ { \prime } ) ,\tag{13}
$$

The overall training objective is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { 1 } ^ { a } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { 2 } ^ { a } \mathcal { L } _ { \mathrm { s e g } } ,\tag{14}
$$

where $\mathcal { L } _ { \mathrm { a c t } }$ denotes the flow-matching loss for action generation, $\lambda _ { 1 } ^ { a } = 0 . 1$ and $\lambda _ { 2 } ^ { a } = 0 . 0 0 2$ are two constant coefficients.

## IV. EXPERIMENTS

## A. Benchmarks and Evaluation Metrics

We evaluate V-Link on 3 simulation benchmarks, including LIBERO [18], LIBERO-Plus [19], and RoboTwin2.0 [20]. For LIBERO and LIBERO-Plus, we follow the official protocols. All models are trained solely on standard LIBERO and evaluated on LIBERO-Plus without fine-tuning. For RoboTwin 2.0, we select 6 challenging tasks: Move Pillbottle Pad, Move Stapler Pad, Place Bread Skillet, Place Phone Stand, Press Stapler, and Turn Switch. These tasks span move, place, and contact interactions. We use 50 demonstrations per task, totaling 300 demonstrations for joint training. Ground-truth depth and semantic segmentation are used only during training and are not required at inference. For realworld evaluation, we use the AGIBOT A3 Ultra and consider two tasks: Autonomous Power-On and Power-Off. We collect 100 teleoperated demonstrations per task. Success rates are reported over 50 trials per task on LIBERO and LIBERO-Plus, 100 trials per task on RoboTwin 2.0, and 50 trials per real-world task.

## B. Implementation Details

All models are initialized from GR00T N1.6 3B and trained in BF16 on 4 NVIDIA H100 GPUs using Deep-Speed ZeRO-2. We freeze the pretrained language and vision backbones and adapt the VLM using rank-128 LoRA. We use AdamW with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of $1 \times 1 0 ^ { - 5 }$ , cosine decay, and 5% warm-up. On LIBERO, models are trained for 80K steps with a global batch size of 160 using external and wrist camera views. The Spatial and Semantic Query sets each contain $2 \times 5 \times 5$ tokens. The policy predicts 16-step action chunks. On RoboTwin 2.0, models are trained for 120K steps with a global batch size of 128. Each query set contains $3 \times 5 \times 5$ tokens. The policy predicts 15-step dual-arm action chunks.

TABLE I: Comparison on LIBERO benchmark.
<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Average</td></tr><tr><td colspan="6">2D VLA</td></tr><tr><td>OpenVLA [21]</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>Dita [22]</td><td>84.2</td><td>96.3</td><td>85.4</td><td>63.8</td><td>82.4</td></tr><tr><td>CoT-VLA [23]</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>83.9</td></tr><tr><td>π0-FAST [9]</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td></tr><tr><td>π0 [24]</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>π0.5 [7]</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>UniVLA [5]</td><td>96.5</td><td>96.8</td><td>95.6</td><td>92.0</td><td>95.2</td></tr><tr><td>OpenVLA-OFT [25]</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td colspan="6">3D VLA</td></tr><tr><td>SpatialVLA [14]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>GeoVLA [26]</td><td>98.4</td><td>99.0</td><td>96.6</td><td>96.6</td><td>97.7</td></tr><tr><td>3D-CAVLA [27]</td><td>98.2</td><td>99.8</td><td>98.2</td><td>96.1</td><td>98.1</td></tr><tr><td>Spatial Forcing [2]</td><td>99.4</td><td>99.6</td><td>98.8</td><td>96.0</td><td>98.5</td></tr><tr><td>GR00T-N1.6 (Base)</td><td>99.3</td><td>99.2</td><td>98.4</td><td>92.9</td><td>97.4</td></tr><tr><td>V-Link (Ours)</td><td>99.5</td><td>100.0</td><td>99.8</td><td>98.0</td><td>99.3</td></tr><tr><td>∆ Ours - Base</td><td>+0.2</td><td>+0.8</td><td>+1.4</td><td>+5.1</td><td>+1.9</td></tr></table>

## C. Comparisons with State-of-the-art Methods

Table I compares V-Link with state-of-the-art 2D and 3D VLAs on LIBERO. V-Link ranks first across all 4 suites, achieving an average success rate of 99.3%. It outperforms GR00T N1.6 and Spatial Forcing by +1.9% and +0.8%, respectively, with the largest gain on LIBERO-Long (+5.1%).

Table II evaluates robustness under 7 distribution shifts on LIBERO-Plus. V-Link achieves the highest overall success rate of 75.0%, surpassing the previous best method by +5.4% and GR00T N1.6 by +31.2%. It ranks first in six settings, with particularly large gains over GR00T N1.6 under noise (+60.4%), language (+47.3%), and lighting (+30.6%).

As shown in Table III, V-Link achieves the highest average success rate of 56.8% across 6 RoboTwin 2.0 tasks, outperforming GR00T N1.6 by +18.8%. Improvements are consistent across all tasks. These results demonstrate that recovering 3D spatial and 2D semantic representations substantially strengthens fine-grained manipulation.

## D. Ablation Study

Q1: How do Spatial and Semantic Queries contribute to policy performance? Table IV shows that Spatial Queries improve the average success rate from 38.0% to 51.2% (+13.2%), with substantial gains on Move (+12.0%) and Place (+15.5%), supporting their role in geometric grounding. Semantic Queries achieve 47.8% (+9.8%), with the largest gain on Contact (+18.0%), suggesting improved localization of task-relevant interaction regions. Combining both yields the best performance of 56.8% (+18.8%), confirming their complementary roles.

TABLE II: Comparison on LIBERO-Plus benchmark.
<table><tr><td>Method</td><td>Camera</td><td>Robot</td><td>Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Total</td></tr><tr><td>OpenVLA-OFT [25]</td><td>56.4</td><td>31.9</td><td>79.5</td><td>88.7</td><td>93.3</td><td>75.8</td><td>74.2</td><td>69.6</td></tr><tr><td>π0 [24]</td><td>13.8</td><td>6.0</td><td>58.8</td><td>85.0</td><td>81.4</td><td>79.0</td><td>68.9</td><td>53.6</td></tr><tr><td>π0-Fast [9]</td><td>65.1</td><td>21.6</td><td>61.0</td><td>73.2</td><td>73.2</td><td>74.4</td><td>68.8</td><td>61.6</td></tr><tr><td>Evo-Depth [28]</td><td>47.2</td><td>49.2</td><td>78.9</td><td>88.1</td><td>76.4</td><td>77.6</td><td>69.6</td><td>69.6</td></tr><tr><td>GR00T-N1.6 (Base)</td><td>21.2</td><td>40.3</td><td>35.5</td><td>65.1</td><td>76.7</td><td>27.4</td><td>51.0</td><td>43.8</td></tr><tr><td>V-Link (Ours)</td><td>42.8</td><td>59.1</td><td>82.8</td><td>95.7</td><td>94.9</td><td>87.8</td><td>74.4</td><td>75.0</td></tr><tr><td>∆ Ours - Base</td><td>+21.6</td><td>+18.8</td><td>+47.3</td><td>+30.6</td><td>+18.2</td><td>+60.4</td><td>+23.4</td><td>+31.2</td></tr></table>

TABLE III: Comparison on Robotwin2.0 benchmark.
<table><tr><td rowspan="2">Method</td><td colspan="2">Move</td><td colspan="2">Place</td><td colspan="2">Contact</td><td rowspan="2">Average</td></tr><tr><td>Pillbottle</td><td>Stapler</td><td>Bread Skillet</td><td>Phone Stand</td><td>Press Stapler</td><td>Turn Switch</td></tr><tr><td>π0 [24]</td><td>21</td><td>0</td><td>23</td><td>35</td><td>62</td><td>27</td><td>28.0</td></tr><tr><td>Xiaomi Robotics-0 [29]</td><td>65</td><td>16</td><td>63</td><td>44</td><td>79</td><td>38</td><td>50.8</td></tr><tr><td>EventVLA [30]</td><td>60</td><td>14</td><td>64</td><td>64</td><td>80</td><td>24</td><td>51.0</td></tr><tr><td>π0.5 [7]</td><td>46</td><td>20</td><td>76</td><td>65</td><td>79</td><td>52</td><td>56.3</td></tr><tr><td>GR00T N1.6 (Base)</td><td>58</td><td>7</td><td>23</td><td>54</td><td>80</td><td>6</td><td>38.0</td></tr><tr><td>V-Link (Ours)</td><td>74</td><td>22</td><td>55</td><td>60</td><td>97</td><td>33</td><td>56.8</td></tr><tr><td>∆ Ours - Base</td><td>+16</td><td>+15</td><td>+32</td><td>+6</td><td>+17</td><td>+27</td><td>+18.8</td></tr></table>

TABLE IV: Ablation study of auxiliary queries.
<table><tr><td>Method</td><td>Move</td><td>Place</td><td>Contact</td><td>Average</td></tr><tr><td>GR00T N1.6</td><td>32.5</td><td>38.5</td><td>43.0</td><td>38.0</td></tr><tr><td>+ Spatial Query</td><td>44.5</td><td>54.0</td><td>55.0</td><td>51.2</td></tr><tr><td>+ Semantic Query</td><td>35.5</td><td>47.0</td><td>61.0</td><td>47.8</td></tr><tr><td>+ Both Query</td><td>48.0</td><td>57.5</td><td>65.0</td><td>56.8</td></tr></table>

Q2: How do task supervision and query injection contribute to policy performance? Table V and Table VI isolate their effects. Query injection without task supervision improves the average success rate only from 38.0% to 40.3%, while task supervision without injection yields only 39.8%. Combining both reaches 56.8%, outperforming the two ablations by +16.5% and +17.0%, respectively. These results demonstrate their complementary roles: task supervision specializes the queries for 3D geometry and 2D semantics, while query injection makes these representations accessible to Action DiT.

Q3: Does V-Link learn specialized visual representations and recover them in Action DiT? Following our diagnostic protocol, we freeze each representation and train identical lightweight heads for depth estimation and semantic segmentation. Poor performance with random features shows that the results reflect representation accessibility rather than taskhead capacity. As shown in Fig. 6, Spatial Queries achieve the lowest depth MAE, while Semantic Queries attain the highest segmentation mIoU, indicating their respective specialization. Moreover, V-Link Action DiT outperforms GR00T Action DiT on both tasks, demonstrating successful representation recovery after query injection.

TABLE V: Ablation study of task supervision. “w/o Task Head” retains the Spatial and Semantic Queries and their injection mechanisms, while removing the depth and semantic heads and the ground-truth supervision.
<table><tr><td>Method</td><td>Move</td><td>Place</td><td>Contact</td><td>Average</td></tr><tr><td>GR00T N1.6</td><td>32.5</td><td>38.5</td><td>43.0</td><td>38.0</td></tr><tr><td>w/o Task Head</td><td>35.5</td><td>38.5</td><td>47.0</td><td>40.3</td></tr><tr><td>w/ Task Head</td><td>48.0</td><td>57.5</td><td>65.0</td><td>56.8</td></tr></table>

TABLE VI: Ablation study of auxiliary query injection. “w/o Injection” retains the Spatial and Semantic Queries, the depth and semantic heads, and the ground-truth supervision, but does not inject both queries into Action DiT.
<table><tr><td>Method</td><td>Move</td><td>Place</td><td>Contact</td><td>Average</td></tr><tr><td>GR00T N1.6</td><td>32.5</td><td>38.5</td><td>43.0</td><td>38.0</td></tr><tr><td>w/o Injection</td><td>40.5</td><td>28.5</td><td>50.5</td><td>39.8</td></tr><tr><td>w/ Injection</td><td>48.0</td><td>57.5</td><td>65.0</td><td>56.8</td></tr></table>

Q4: How does the number of Spatial/Semantic Query tokens affect policy performance and inference latency? As shown in Fig. 7, performance improves as the query grid grows, but with diminishing returns. Increasing it from 3 × 1 × 1 to 3 × 2 × 2 raises the average success rate from 37.8% to 51.8%, indicating that sufficient query capacity is essential for extracting spatial and semantic information. With a 3 × 5 × 5 query grid, V-Link achieves the highest success rate of 56.8%, outperforming GR00T N1.6 by +18.8% with only 1.58 ms of additional inference latency. Further increasing the grid to $3 \times 6 \times 6$ reduces performance slightly to 56.5% and incurs additional latency. Therefore, $3 \times 5 \times 5$ provides the best accuracy–efficiency trade-off.

![](images/8f64d4f44f93f01d5ad51303e4b1142ab31d038bfa8ae220225b3d104529adfc.jpg)  
Fig. 5: Qualitative comparison. The Action DiT of GR00T N1.6 exhibits severe visual representation degradation, retaining little 3D depth information and incomplete 2D semantics. V-Link effectively recovers the spatial and semantic representations within Action DiT.

## E. Real-Robot Experiments

We evaluate V-Link on the AGIBOT A3 Ultra in 50 consecutive trials per task. For training, Lingbot-Depth [31] completes missing RGB-D depth caused by sensor dropouts and reflective surfaces, while GroundingDINO and SAM3 generate key-object segmentation pseudo-GT, with other pixels labeled as background. Without auxiliary heads or pseudo-GT at inference, V-Link achieves 98%/94% success on power-on/off, outperforming GR00T N1.6 at 78%/70%.

## V. CONCLUSION

This work reveals a previously overlooked visual representation accessibility bottleneck in dual-system VLAs: VLto-A transfer limits Action DiT’s access to 3D geometric and 2D semantic information encoded in VLM features. To address this issue, we propose V-Link, which learns complementary Spatial and Semantic Query representations and injects them asymmetrically into Action DiT, without requiring auxiliary heads or annotations at inference. Exper iments across three simulation benchmarks and real-world humanoid manipulation demonstrate consistent gains with minimal inference overhead. These results establish representation recovery at the VL-to-A interface as an effective approach for precise and efficient robot manipulation.

## REFERENCES

[1] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan,˜ Y. Fang, D. Fox, F. Hu, S. Huang, et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” arXiv:2503.14734, 2025.

![](images/b35ba5c4519cf01dac789e878fef429a3dcba67be11ca7c1fe1b5a6cb086b00a.jpg)

2D Semantic Segmentation  
![](images/b2e45cd9b680b9b603c39d91d32a0a6d4a11f6713c9374d5cd1c6dc5c345f3c4.jpg)  
Fig. 6: Diagnostic evaluation of frozen visual representations. The poor results with random features show that diagnostic performance depends on the information encoded in the frozen features, rather than the task heads themselves. Spatial and Semantic Queries achieve the best depth estimation and semantic-segmentation results, respectively, demonstrating task-specific specialization. V-Link Action DiT outperforms GR00T Action DiT on both tasks, confirming successful representation recovery.

![](images/91e270ee6a6ae13cce89bb7380a39ca2c5f3e4525d2241deeabcff3543e7b891.jpg)  
Fig. 7: Effect of Spatial/Semantic Query token number on performance and inference latency. Latency is tested on a single H100 and averaged over 1,000 randomly sampled inputs with batch size 1, using torch.compile, 4 denoising steps, and 3 views input.

[2] F. Li, W. Song, H. Zhao, J. Wang, P. Ding, D. Wang, L. Zeng, and H. Li, “Spatial forcing: Implicit spatial representation alignment for vision-language-action model,” arXiv:2510.12276, 2025.

[3] T. Lin, G. Li, Y. Zhong, Y. Zou, Y. Du, J. Liu, E. Gu, and B. Zhao, “Evo-0: Vision-language-action model with implicit spatial understanding,” arXiv:2507.00416, 2025.

[4] Y. Wang, P. Ding, L. Li, C. Cui, Z. Ge, X. Tong, W. Song, H. Zhao, W. Zhao, P. Hou, et al., “Vla-adapter: An effective paradigm for

AGIBOT A3 Ultra Autonomous Power-On Task  
![](images/17b78f209d0b4c11fdfab611286fbf2c3269385ddffa045932112a4a062b49e8.jpg)

![](images/9709b86c2fb77a1f98be557697a1e7d7c4631cd20296c631202dcc39a82d4a1f.jpg)  
AGIBOT A3 Ultra Autonomous Power-Off Task

![](images/e2fed2b93882aaa51f836a2c1d5217edd92936e8086031f381f5d3a6efa0c9e0.jpg)

![](images/45d3cea86ed4f314595c54428c5fe8be9b4fda395a2f78f522394cc90d1e6801.jpg)  
Fig. 8: Real-robot evaluation on the AGIBOT A3 Ultra. Over 50 consecutive trials per task, V-Link achieves 98%/94% success rate on autonomous power-on/off tasks, versus 78%/70% for base model GR00T N1.6.

tiny-scale vision-language-action model,” in Proceedings of the AAAI conference on artificial intelligence, vol. 40, no. 22, 2026, pp. 18 638– 18 646.

[5] Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang, “Unified vision-language-action model,” arXiv:2506.19850, 2025.

[6] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, et al., “Cosmos policy: Fine-tuning video models for visuomotor control and planning,” arXiv:2601.16163, 2026.

[7] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al., “π : a vision-language-action model with open-world generalization,” arXiv:2504.16054, 2025.

[8] W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma, et al., “A pragmatic vla foundation model,” arXiv:2601.18692, 2026.

[9] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine, “Fast: Efficient action tokenization for vision-language-action models,” arXiv:2501.09747, 2025.

[10] S. Xu, Y. Wang, C. Xia, D. Zhu, T. Huang, and C. Xu, “Vlacache: Efficient vision-language-action manipulation via adaptive token caching,” Advances in Neural Information Processing Systems, vol. 38, pp. 164 448–164 473, 2026.

[11] Y. Li, Y. Meng, Z. Sun, K. Ji, C. Tang, J. Fan, X. Ma, S. Xia, Z. Wang, and W. Zhu, “Sp-vla: A joint model scheduling and token pruning approach for vla model acceleration,” arXiv:2506.12723, 2025.

[12] L. Sun, Z. Guan, C. Wang, Z. Chen, J. Yu, Z. Li, B. He, T. Sun, J. Cao, and L. Liu, “Unifs: Unified fast-to-slow hierarchical architecture for vision-language-action models,” arXiv:2606.22794, 2026.

[13] Y. Zhang, S. Zhang, Y. Shen, S. Dong, J. Deng, X. Zhang, Y. Gao, J. Wu, X. Nie, Z. Cheng, et al., “Gear-vla: Learning geometryaware action representations for generalizable robotic manipulation,” arXiv:2606.08530, 2026.

[14] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al., “Spatialvla: Exploring spatial representations for visual-language-action model,” arXiv:2501.15830, 2025.

[15] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294–5306.

[16] C. Li, J. Wen, Y. Peng, Y. Peng, and Y. Zhu, “Pointvla: Injecting the 3d world into vision-language-action models,” IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 2506–2513, 2026.

[17] Y. Yang, S. Zeng, T. Lin, X. Chang, D. Qi, J. Xiao, H. Liu, R. Chen, Y. Chen, D. Huo, et al., “Abot-m0: Vla foundation model for robotic manipulation with action manifold learning,” arXiv:2602.11236, 2026.

[18] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 44 776–44 791, 2023.

[19] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, et al., “Libero-plus: In-depth robustness analysis of visionlanguage-action models,” arXiv:2510.13626, 2025.

[20] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al., “Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation,” arXiv:2506.18088, 2025.

[21] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., “Openvla: An opensource vision-language-action model,” arXiv:2406.09246, 2024.

[22] Z. Hou, T. Zhang, Y. Xiong, H. Duan, H. Pu, R. Tong, C. Zhao, X. Zhu, Y. Qiao, J. Dai, et al., “Dita: Scaling diffusion transformer for generalist vision-language-action policy,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 7686–7697.

[23] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al., “Cot-vla: Visual chain-of-thought reasoning for vision-language-action models,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 1702– 1713.

[24] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al., “π : A vision-language-action flow model for general robot control,” arXiv:2410.24164, 2024.

[25] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” arXiv:2502.19645, 2025.

[26] L. Sun, B. Xie, Y. Liu, H. Shi, T. Wang, and J. Cao, “Geovla: Empowering 3d representations in vision-language-action models,” arXiv:2508.09071, 2025.

[27] V. Bhat, Y.-H. Lan, P. Krishnamurthy, R. Karri, and F. Khorrami, “3d cavla: Leveraging depth and 3d context to generalize vision language action models for unseen tasks,” arXiv:2505.05800, 2025.

[28] T. Lin, Y. Du, J. Liu, N. Zhu, Y. Li, Y. Fu, Y. Chen, H. Cai, Z. Ye, B. Cheng, et al., “Evo-depth: A lightweight depth-enhanced visionlanguage-action model,” arXiv:2605.14950, 2026.

[29] R. Cai, J. Guo, X. He, P. Jin, J. Li, B. Lin, F. Liu, W. Liu, F. Ma, K. Ma, et al., “Xiaomi-robotics-0: An open-sourced vision-languageaction model with real-time execution,” arXiv:2602.12684, 2026.

[30] G. Yang, Z. Tu, Y. Yang, S. Mao, J. Dong, T. Chen, J. Peng, J. Xiong, J. Cao, J. Dai, et al., “Eventvla: Event-driven visual evidence memory for long-horizon vision-language-action policies,” arXiv:2606.20092, 2026.

[31] B. Tan, C. Sun, X. Qin, H. Adai, Z. Fu, T. Zhou, H. Zhang, Y. Xu, X. Zhu, Y. Shen, and N. Xue, “Masked depth modeling for spatial perception,” arXiv:2601.17895, 2026.