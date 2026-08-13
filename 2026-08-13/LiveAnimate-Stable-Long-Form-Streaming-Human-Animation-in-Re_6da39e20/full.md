![](images/03d1294932f410fa251d3c4db94562465f50f701a24f750cce74de2999b73113.jpg)  
Figure 1. LiveAnimate enables real-time, stable long-form streaming human animation. Given a reference image and a stream o body-pose and facial controls, LiveAnimate generates identity-consistent animation causally, one temporal block at a time. Pose-Retrieva Sink Attention recalls pose-relevant historical context when similar poses recur, preserving appearance over extended streams. With threestep sampling, LiveAnimate runs at ∼ 20 FPS on two NVIDIA H100 GPUs while maintaining stable long-form generation.

# LiveAnimate: Stable Long-Form Streaming Human Animation in Real-Time

Yuxuan Zhang <sup>1,2</sup> Haozhong Xiong <sup>2</sup> Yubo Huang <sup>2</sup> Jiayi Song <sup>2</sup> Jinpeng Yu <sup>2</sup> Haofan Wang <sup>3</sup> Jiaming Liu <sup>2</sup> Ruihua Huang <sup>2</sup> Liwei Wang <sup>1\*</sup> <sup>1</sup> The Chinese University of Hong Kong, <sup>2</sup> Qwen Applications Business Group of Alibaba, <sup>3</sup> Liblib AI yxzhang@cse.cuhk.edu.hk, jmliu1217@gmail.com https://liveanimate.github.io/

## Abstract

Pose-driven human animation synthesizes a video of a target personfrom a single reference image and a driving pose stream. Real-time generation is essential for interactive applications such as live streaming, telepresence, and virtual avatars, yet diffusion-based systems require minutes to hours per clip, precluding responsive interaction. We present LiveAnimate, to our knowledge the first animation system to combine real-time streaming with stable longform generation at billion scale, built on a 14B-parameter video Diffusion Transformer (DiT). A two-stage training pipeline first adapts a pretrained bidirectional DiT into a block-causal autoregressive generator through Reference-Anchored Teacher-Forcing Adaptation, and then reduces the sampling budget to three steps through Block-wise Self-Forcing Distillation. To preserve appearance over extended

streams, we introduce Pose-Retrieval Sink Attention (PR-Sink), a bounded KV-cache mechanism combining a Static Sink that permanently anchors the first generated block, a Dynamic Sink that holds a pose-retrieved historical block, and a three-slot Rolling Window. When a pose recurs, PR-Sink restores the relevant appearance context without retaining the entire sequence, so memory and per-block latency remain constant regardless of stream duration. Together with Ulysses sequence parallelism and operator fusion, these designs enable 19.63 FPS streaming inference on two NVIDIA H100 GPUs. On a three-minute benchmark, LiveAnimate maintains nearly constant perceptual quality and identity from the first 30 seconds to the final minute (IQA 4.047 vs. 4.026), while prior systems degrade substantially or require hours of offline computation for the same rollout. These results establish a new operating point in quality, latency, and duration for interactive full-body animation.

Table 1. Capability comparison of human animation methods. LiveAnimate is the first to achieve real-time streaming with stable long-form generation at billion scale.
<table><tr><td>Method</td><td>Streaming</td><td>Real-Time</td><td>Stable Long-Form</td><td>Param</td></tr><tr><td>Animate Anyone [14]</td><td>x</td><td>x</td><td>x</td><td>一</td></tr><tr><td>UniAnimate-DiT [30]</td><td>x</td><td>x</td><td>x</td><td>14B</td></tr><tr><td>One-to-All [25]</td><td>x</td><td>x</td><td>x</td><td>14B</td></tr><tr><td>SCAIL [36]</td><td>x</td><td>x</td><td>x</td><td>14B</td></tr><tr><td>Wan-Animate [1]</td><td>x</td><td>x</td><td>x</td><td>14B</td></tr><tr><td>EverAnimate [20]</td><td>x</td><td>x</td><td>√</td><td>14B</td></tr><tr><td>LiveAnimate (Ours)</td><td>√</td><td>√</td><td>√</td><td>14B</td></tr></table>

## 1. Introduction

Pose-driven human animation, which synthesizes a video of a target person following a sequence of driving poses while preserving the appearance from a single reference image, is a fundamental task with broad applications in virtual try-on, digital human creation, live streaming, and telepresence. The core challenge is to simultaneously maintain appearance fidelity, pose accuracy, and temporal coherence over extended sequences, a combination that has proven difficult to achieve. Video diffusion models [12, 23, 24] have become the prevailing approach to this task. Animate Anyone [14], MagicAnimate [35], and UniAnimate-DiT [30] adopt reference-based conditioning with temporal attention for short-clip generation. Building on billion-scale Diffusion Transformers (DiTs), Wan-Animate [1, 7] and SCAIL [36] scale this design to 14Bparameter backbones with 3D-consistent pose representations. More recently, EverAnimate [20] extends generation to the minute scale through persistent latent propagation and restorative flow matching.

However, a critical gap remains: all of these methods are offline, requiring minutes to hours per clip and precluding responsive interaction. As shown in Table 1, no prior fullbody diffusion system simultaneously supports streaming generation, real-time inference, and stable long-form output. Most methods [14, 30, 36] generate fixed-length clips and provide no mechanism for processing an open-ended pose stream. Even the minute-scale EverAnimate [20] still requires 20 denoising steps per chunk, keeping it far from real time. Closing this gap requires solving three coupled problems: (i) reducing the inference cost of a billion-scale DiT to meet an interactive latency budget, (ii) converting a bidirectional video model into a causal generator without sacrificing quality, and (iii) preventing identity and appearance drift over arbitrarily long rollouts.

To address these challenges, we present LiveAnimate, the first system to enable real-time streaming human animation with stable long-form generation using a 14Bparameter causal DiT. Our approach is built on a twostage training pipeline. In Stage 1, Reference-Anchored Teacher-Forcing Adaptation converts the pretrained bidirectional DiT into a block-causal generator by conditioning each training block on ground-truth clean history. A global Ref Sink makes the reference-image latent visible to all generated blocks and is distinct from the generatedhistory sinks used at inference. In Stage 2, Block-wise Self-Forcing Distillation (BS-DMD), building on Self Forcing [16], first performs a Self-Forcing Rollout without gradient tracking and then applies Block-wise DMD Optimization to one replayed block at a time. This exposes every block position to a distribution-matching signal without retaining the full rollout graph, allowing Stage-2 distillation of the 14B model to run on a single 8×80GB GPU node. To maintain temporal coherence over long-form streaming, we propose Pose-Retrieval Sink Attention (PR-Sink), which combines a Static Sink, a pose-retrieved Dynamic Sink, and a bounded Rolling Window. The bank update objective maximizes pose coverage, while retrieval supplies the historical block written into the Dynamic Sink. At the infrastructure level, we employ Ulysses-style sequence parallelism [9] and operator fusion to distribute attention computation across GPUs, achieving 19.63 FPS real-time streaming on 2×H100 GPUs.

In summary, our contributions are threefold:

• We present LiveAnimate, the first system to enable real-time streaming human animation with stable longform generation at billion scale, achieving ∼20 FPS on 2×H100 GPUs.

• We design a two-stage training pipeline: Reference-Anchored Teacher-Forcing Adaptation converts a pretrained bidirectional DiT into a block-causal generator, and Block-wise Self-Forcing Distillation reduces the sampling budget to three steps, with a one-block-at-atime replay scheme that keeps training feasible on a single 8×80GB GPU node. We further propose PR-Sink, a bounded KV-cache strategy that combines a Static Sink, a pose-retrieved Dynamic Sink, and a Rolling Window to maintain appearance consistency over long-form streams.

• On a three-minute long-form benchmark, LiveAnimate preserves perceptual quality and identity across the rollout while producing video at ∼ 20 FPS, establishing a favorable trade-off among quality, latency, and duration against state-of-the-art offline systems.

## 2. Related Work

## 2.1. Video Diffusion Models

Video diffusion models have evolved from pixel-space U-Nets [12, 13] to latent-space architectures and Diffusion Transformers (DiTs) [23]. Video LDM [3] and Stable Video Diffusion [2] showed that operating in a compressed VAE latent space enables high-resolution video generation at manageable cost. More recently, billion-scale DiTs have become strong backbones for controllable generation; Wan [1], for example, scales to 14B parameters with a causal 3D VAE and a flow-matching objective. Our work builds on Wan2.2-Animate-14B[7] and adapts it from offline clip generation to causal, real-time streaming.

## 2.2. Pose-Driven Human Animation

Pose-driven human animation synthesizes realistic video of a target person performing motions specified by a driving signal such as skeleton sequences or DensePose maps. Early methods based on GANs with warping-based generators [26, 29] achieve real-time speed but produce artifacts under large pose changes.

Diffusion models have since become the prevailing approach to this task. Animate Anyone [14] introduced ReferenceNet for appearance encoding together with a pose guider and temporal attention, while MagicAnimate [35] used DensePose conditioning and segment-wise generation. MusePose [28] released an open-source pose-driven imageto-video framework following a similar reference-and-pose conditioning paradigm. Champ [42] instead employed 3D parametric body guidance to improve shape and motion controllability. UniAnimate-DiT [30] unified reference attention with long video generation using a DiT backbone. HumanDiT [10] likewise studied long-form pose-guided human video generation with a DiT backbone. EverAnimate [20] further extends generation to the minute scale, mitigating long-horizon identity drift through persistent latent propagation and restorative flow matching. Some systems further extend controllability and visual fidelity. Oneto-All [25] proposed alignment-free character animation and image pose transfer; SteadyDancer [39] emphasized first-frame preservation; and SCAIL [36] introduced 3Dconsistent pose representations. Wan-Animate [1, 7] unified character animation and replacement on a large-scale video DiT. MultiAnimate [15] extended pose-guided animation to multiple characters, while VACE [18] provided a general framework for controllable video creation and editing. Most of these methods are designed for offline generation and do not jointly target low-latency streaming and minute-scale temporal stability.

## 2.3. Autoregressive Video Generation

Autoregressive chunk-wise generation extends video beyond a model’s training horizon, but repeated rollouts can accumulate visual degradation and semantic drift. Diffusion Forcing [5] introduced per-token noise levels for sequence generation, and CausVid [38] distilled a bidirectional video diffusion model into a fast causal generator. Self Forcing [16], building on distribution matching distillation [37], trained on the model’s own autoregressive rollouts to reduce exposure bias. Causal Forcing [41] argued that distilling an autoregressive student directly from a bidirectional teacher violates frame-level ODE injectivity, and instead performs distillation from a teacher-forced autoregressive model. Rolling Forcing [22] further aligned rolling-window training with real-time long-video inference, while Context Forcing [6] used long-context conditioning for consistent autoregressive generation. Stable Video Infinity [21] recycles accumulated errors, and Rolling Sink [19] dynamically refreshes attention-sink frames to bridge limited-horizon training and open-ended testing. Collectively, these methods mitigate rollout drift in general video generation, but they do not address pose-aware cache retention under strict real-time latency budgets.

## 3. Method

## 3.1. Overview

In this section, we present LiveAnimate, a framework for real-time streaming pose-driven human animation using a billion-scale causal Diffusion Transformer. As illustrated in Figure 2, given a reference image $I _ { \mathrm { r e f } }$ and a stream of driving pose skeletons $\{ P _ { t } \}$ , our system generates video blocks autoregressively, each denoised in 3 steps and followed by a clean KV cache update. The framework consists of three components: (1) a two-stage training pipeline comprising Reference-Anchored Teacher-Forcing Adaptation (Sec. 3.2) and Block-wise Self-Forcing Distillation (Sec. 3.3); (2) Pose-Retrieval Sink Attention (PR-Sink, Sec. 3.4), a bounded KV-cache mechanism for stable longform streaming; and (3) System Optimizations that enable 19.63 FPS inference on 2×H100 GPUs (Sec. 3.5).

## 3.2. Stage 1: Reference-Anchored Teacher-Forcing Adaptation

Standard video diffusion models denoise complete sequences with bidirectional attention and therefore cannot directly emit temporal blocks causally. We adapt the pretrained DiT into a block-causal autoregressive generator through Reference-Anchored Teacher-Forcing Adaptation.

We partition the training video into temporal blocks of $B _ { f }$ latent frames. During Stage 1, each block b is denoised while attending to ground-truth clean target blocks $z _ { < b } ^ { \mathrm { G T } }$ rather than to its own previous predictions. This groundtruth clean history prevents autoregressive errors from entering the conditioning context during Stage 1; exposure to student-generated history is introduced only in Stage 2.

A key distinction from standard teacher forcing [32] in unconditional generation is the Ref Sink, a global conditional-cache region that stores the reference-image KV states. In the conditional pose-driven setting, the model must maintain identity and appearance consistency with a given reference image throughout the entire generation. We encode the reference latent $z _ { \mathrm { r e f } }$ through a forward pass at the clean timestep (t=0) and make the resulting Ref Sink globally visible to all subsequent temporal blocks. For targetblock queries $Q _ { b } ,$ , block-causal self-attention reads the Ref

![](images/f01ba8c30353bbb910d0e91739e09b82f92514317a004d16147d32070203443a.jpg)  
Figure 2. Overview of LiveAnimate. Given a reference image and streaming pose signals, our system generates video blocks autoregres sively. Each block undergoes 3-step denoising followed by a clean KV update. PR-Sink augments a three-block rolling window with the first generated block and a pose-matched historical block selected from a compact memory bank. Ulysses sequence parallelism distributes attention computation across GPUs.

Sink, the ground-truth clean history, and the current block:

$$
\mathrm { A t t n } ( Q _ { b } , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q _ { b } [ K _ { \mathrm { r e f } } ; K _ { < b } ] ^ { T } } { \sqrt { d } } \right) [ V _ { \mathrm { r e f } } ; V _ { < b } ]\tag{1}
$$

where $K _ { \mathrm { r e f } } , V _ { \mathrm { r e f } }$ are the key-value pairs from the reference frame, and $K _ { < b } , V _ { < b }$ are the teacher-provided clean KV states from blocks preceding b. This design ensures the reference frame serves as a permanent context anchor— analogous to attention sinks in LLMs [34]—while adapting teacher forcing from text-to-video to the conditional generation setting.

At causal inference and during the Self-Forcing Rollout in Stage 2, a denoised block $\hat { z } _ { b } ^ { 0 }$ is written to the historical cache through the Clean $K V U p d a t e$ , which performs an additional forward pass at the clean timestep:

$$
\mathbf { K } \mathbf { V } _ { \mathrm { c a c h e } }  \mathbf { K } \mathbf { V } _ { \mathrm { c a c h e } } \cup \mathbf { F o r w a r d } ( \hat { z } _ { b } ^ { 0 } , t = 0 , \mathcal { C } )\tag{2}
$$

The Clean KV Update places historical states at the clean noise level used by the causal context. It does not remove

errors already present in the generated latent.

## 3.3. Stage 2: Block-wise Self-Forcing Distillation

The teacher-forcing adaptation produces a high-quality causal generator, but it still requires 50 denoising steps per block, precluding real-time inference. In Stage 2, we reduce the sampling budget to three steps through block-wise selfforcing distribution matching distillation (BS-DMD), building on Self Forcing [16]. Unlike Stage 1 teacher forcing, which conditions each training block on clean ground-truth history, Self Forcing exposes the student to autoregressive histories induced by its own predictions and thereby reduces the train–inference exposure gap.

Directly backpropagating through an N-block selfforcing rollout would require retaining the activation graph of the entire autoregressive trajectory. We instead use a twopass block-wise replay procedure. In the first pass, the student generates a complete trajectory without gradient tracking:

$$
( { \bar { z } } _ { 1 } , \dots , { \bar { z } } _ { N } ) = { \mathrm { R o l l o u t } } _ { \theta } ( \epsilon _ { 1 : N } , { \mathcal { C } } )\tag{3}
$$

Every block is therefore generated under student-induced history, while the rollout activations are not retained for backward.

In the second pass, we replay one block position $T$ at a time. The contextual latent and KV states reconstructed from the first-pass trajectory are detached, and only the current block is recomputed with gradient tracking:

$$
z _ { T } ^ { \theta } = G _ { \theta } ( \epsilon _ { T } ; \mathrm { s g } ( \mathrm { K V } ( \bar { z } _ { < T } ) ) , \mathcal { C } _ { T } ) ,\tag{4}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. We insert the recomputed block into the otherwise detached trajectory,

$$
\tilde { z } ^ { ( T ) } = [ \mathrm { s g } ( \bar { z } _ { 1 } ) , \dots , z _ { T } ^ { \theta } , \dots , \mathrm { s g } ( \bar { z } _ { N } ) ] ,\tag{5}
$$

and evaluate the distribution-matching objective on the concatenated trajectory:

$$
\begin{array} { r } { \mathcal { L } _ { T } = \mathcal { L } _ { \mathrm { D M D } } \left( \tilde { z } ^ { ( T ) } ; \mathcal { C } \right) , \qquad T = 1 , \dotsc , N . } \end{array}\tag{6}
$$

The teacher and critic thus evaluate a complete studentinduced trajectory, whereas gradients reach the student only through the currently replayed block $z _ { T } ^ { \theta }$ . We backpropagate $\mathcal { L } _ { T }$ before proceeding to the next block and release the current block’s activation graph. Sliding T from 1 to N provides a direct training signal to every block position without retaining backward activations for all blocks simultaneously. Together with LoRA adaptation, this keeps peak memory low enough to distill the 14B-parameter student on a single 8×80GB GPU node.

## 3.4. Pose-Retrieval Sink Attention

Autoregressive streaming presents a memory–context dilemma. Retaining every historical key–value (KV) state gives complete context but makes memory and attention cost grow linearly with duration. A bounded cache has constant cost, but irreversibly discards appearance evidence once its block leaves the Rolling Window. This is especially harmful for articulated motion: when a subject returns to a previously observed pose, the Rolling Window may contain only the intervening views. Pose-Retrieval Sink Attention (PR-Sink) addresses this dilemma by combining two complementary persistent regions: a Static Sink that permanently anchors the sequence to its first generated block, and a Dynamic Sink whose content is refreshed with a posematched historical block retrieved from a compact memory bank. The Static Sink provides time-invariant appearance context, whereas the Dynamic Sink restores view- and articulation-specific evidence when a pose recurs.

Bounded cache organization. For the current block t, each attention layer uses a Static Sink containing the first generated block, a Dynamic Sink containing a pose-retrieved historical block, and a three-slot Rolling Window containing two recent clean blocks t−2 and t−1 together with the current block t. Reference-image tokens are stored separately in the global Ref Sink and are not counted as either generated-history sink or as one of the Rolling Window slots. After block t is denoised, the Clean KV Update performs an additional forward pass at $t { = } 0 .$ , writes the resulting KV state into the Rolling Window, and advances the window before the next block is generated. Because the Static Sink, Dynamic Sink, Rolling Window, Ref Sink, and pose bank all have fixed capacities, cache storage and attention cost do not grow with stream duration.

Confidence-aware whole-body fingerprint. We extract 133 whole-body keypoints with ViTPose and convert them into the animation representation. Each frame contains gpu s body landmarks and 21 landmarks for each hand. The body subset retains the main torso and limb joints together with head anchors (nose, eyes, and ears). It also contains both ankles and one toe-center per foot, obtained by averaging the two toe landmarks on that foot. The neck is similarly formed by averaging the two shoulder landmarks. All 21 joints of each hand are retained. We exclude the dense facial contour landmarks because their small localization variations can dominate pose matching even when the global articulation is unchanged.

Let $\mathbf p _ { b , t , k } = ( x , y , c )$ contain the normalized coordinates and confidence of keypoint k in frame t of block b. We concatenate all landmarks from the three ordered pose frames and apply $\ell _ { 2 }$ normalization:

$$
\phi _ { b } = \mathrm { N o r m } \bigl ( \mathrm { C o n c a t } _ { t = 1 } ^ { 3 } \{ \mathbf { p } _ { b , t , k } \} _ { k \in \mathcal { K } } \bigr ) .\tag{7}
$$

Here $| K _ { \mathrm { b o d y } } | ~ = ~ 2 0$ and $| K _ { \mathrm { l h } } | ~ = ~ | K _ { \mathrm { r h } } | ~ = ~ 2 1$ , yielding a $3 \times 6 2 \times 3 = 5 5 8$ dimensional fingerprint. Temporal concatenation preserves short-term motion configuration that would be lost under mean pooling. If pose detection fails for an entire frame, its portion of the fingerprint is zero padded; otherwise confidence is retained as the third coordinate. Global $\ell _ { 2 }$ normalization makes the dot product between two fingerprints equal to their cosine similarity. Diversity-preserving bank update. The memory bank $B = \{ ( \phi _ { i } , \mathrm { K V } _ { i } ) \}$ has a fixed capacity $M = 5 .$ . Each entry stores the fingerprint and the unrotated keys and values from all transformer layers. Once the bank is full, an incoming block $q$ may replace one existing entry. We test every possible replacement and choose the bank with the lowest average pairwise cosine similarity:

$$
i ^ { * } = \arg \operatorname* { m i n } _ { i } \ \operatorname { A v g S i m } ( ( B \setminus \{ i \} ) \cup \{ q \} ) .\tag{8}
$$

We perform the replacement only if it lowers the current bank similarity. Thus, writing favors coverage: redundant poses are rejected, while novel articulations increase the diversity of the stored context. Bank updates are restricted to the first 20 blocks, after which the selected representatives are reused throughout the stream. Pose-nearest retrieval.

Input image

![](images/b5bc8365788b862f1f5bac760a2a26ac5ccac4731a99ac713b01e5b6d9e81907.jpg)  
Figure 3. Qualitative comparison on a full-body sequence. Frames are sampled every 20 seconds from a three-minute, 25-FPS rollout. The pose signal is shown above the generated frames, and red annotations highlight representative long-horizon artifacts in competing methods. The rightmost column reports end-to-end generation time: the baselines require approximately 2–5 hours, whereas LiveAnimate completes the sequence in approximately 4 minutes.

Before generating block b, we compute its fingerprint from the incoming pose signal and retrieve

$$
r ( b ) = \underset { i \in \mathcal { B } , i \neq b - 1 } { \arg \operatorname* { m a x } } \phi _ { b } ^ { \top } \phi _ { i } .\tag{9}
$$

We exclude block b − 1 because it is already present in the Rolling Window; selecting it again would waste the Dynamic Sink on redundant local context. The retrieved KV state is copied into the Dynamic Sink. Bank writing therefore seeks coverage, whereas bank reading seeks relevance to the current pose. PR-Sink is the union of the pose-conditioned Dynamic Sink and the permanent firstblock Static Sink; neither component alone constitutes the full mechanism.

Position-consistent KV reuse. Historical blocks cannot be copied with their original rotary position embeddings (RoPE), because the Dynamic Sink has a different temporal position at retrieval time. We therefore cache $K ^ { \mathrm { r a w } }$ before RoPE together with V. At attention time, keys from the Static Sink, Dynamic Sink, and Rolling Window are separately re-rotated at their assigned positions in the current cache layout before being concatenated with the current block. This decouples content selection from its original timestamp and permits the same bank entry to be reused at any later point without positional mismatch.

![](images/692c45a6f43fca3cee908f33012824794f2c067facef74ce48ee70d41032affb.jpg)  
Figure 4. Qualitative comparison on an upper-body sequence. Frames are sampled every 20 seconds from a three-minute rollout under the same reference image and driving-pose sequence. LiveAnimate maintains the subject’s identity, clothing, and dark background more consistently over time.

## 3.5. System Optimizations

To meet the latency requirements of real-time streaming, we employ several infrastructure-level optimizations.

Ulysses sequence parallelism. We distribute computation across N GPUs using the Ulysses all-to-all pattern [9]: each GPU computes Q, K, V projections on its local token shard, then all-to-all communication redistributes to head-parallel layout where each GPU computes full-sequence attention for H/N heads, followed by a reverse all-to-all gather. This requires 2 all-to-all operations per layer but each involves only head-dimension-sized tensors. The FFN computation remains embarrassingly parallel on local token shards, and the KV cache is maintained independently per GPU head shard with no additional communication overhead.

torch.compile. All DiT blocks are compiled with max-autotune-no-cudagraphs mode. We precompile for three resolution buckets (480×480, 672×384, 384×672) with dynamic=False for maximum kernel efficiency.

## 4. Experiments

## 4.1. Experimental Setup

Implementation details. We train LiveAnimate on 40k talking videos[8] and 20k human motion videos[17, 31]. Both training and inference support three aspect-ratio buckets: 480×480,384×672, and 672×384 pixels. During training, LiveAnimate is initialized from the Wan2.2-Animate-14B checkpoint and fine-tuned with LoRA of rank 128. Training proceeds in two stages on 8 NVIDIA H100 GPUs. In the first stage, we train for 10,000 steps with a learning rate of $5 \times 1 0 ^ { - 5 }$ . In the second stage, we continue training for 20,000 steps with identical LoRA parameters, using learning rates of $1 \times 1 0 ^ { - 5 }$ for both the generator and the critic. Unless stated otherwise, we generate at resolution using blocks of three latent frames (12 RGB frames). Inference is performed on two NVIDIA H100 80GB GPUs.

Evaluation protocol. We construct a three-minute benchmark consisting of 24 pairs of reference images and driving videos, divided into two complementary parts. The first part contains 12 in-the-wild videos collected from the web, covering full-body and upper-body motion, diverse appearances, and varied backgrounds. The second part supports controlled long-horizon evaluation: since long-form fullbody motion footage is scarce, we derive 12 sequences from X-Dance [39] by playing each driving pose sequence forward, in reverse, and forward again to reach three minutes, with the reference image taken from the source video. Every frame therefore remains real human motion, while each pose recurs multiple times at large temporal intervals, directly stressing identity and appearance consistency under pose recurrence. Each method receives the same reference image and driving pose sequence. We evaluate a short 0– 10 s prefix, the broader 0–30 s initial window, and three nonoverlapping later segments: 30–90, 90–120, and 120–180 s. The nested short windows characterize initial quality, while the later segments isolate accumulated degradation.

![](images/fa2ffb9b7e64c413add26213e720483ffc4d1882caac8d6f9dc9f5afe2ad4c79.jpg)  
Figure 5. Quality and identity over a three-minute rollout. Metrics are computed on a 0–10 s prefix, the 0–30 s initial window, and three subsequent temporal segments. The full LiveAnimate model is highlighted in red. Flat trajectories indicate resistance to accumulated degradation. Higher is better for ASE, IQA, and DINO-S; lower is better for FID and V-MAE.

Metrics. We assess complementary aspects of generation quality without relying on pixel-aligned reconstruction metrics, which can penalize perceptually valid motion and appearance variations. Aesthetic score (ASE)[33] and no-reference image-quality assessment (IQA)[33] measure frame-level visual quality. DINO similarity (DINO-S)[4] measures appearance consistency with the reference identity, while FID [11] and VideoMAE feature distance (V-MAE) [27] evaluate distributional quality and temporal representation error, respectively. Higher is better for ASE, IQA, and DINO-S; lower is better for FID and V-MAE.

Latency and throughput measurement. All latency and throughput numbers reported in this paper measure the DiT generation loop only, rather than end-to-end wall-clock time. Condition processing (e.g., reference and pose encoding) and VAE encoding/decoding are excluded from the measurement, as their costs are either one-time or blockindependent, can be overlapped on separate devices, and vary considerably across implementations. Since DiT forward passes dominate the recurrent per-block cost (Table 3), this protocol isolates the primary bottleneck and provides a fair basis for comparison across systems.

Baselines. We compare with five recent full-body animation systems: EverAnimate [20], One-to-All [25], SCAIL [36], UniAnimate-DiT [30], and Wan2.2- Animate [1, 7]. We additionally ablate the two components of PR-Sink. The w/o Dynamic Sink variant retains the Static Sink but disables pose-based historical retrieval, whereas the w/o Static Sink variant retains the pose-retrieved Dynamic Sink but removes the permanent first-block anchor. The full model combines both sinks with the three-slot Rolling Window.

Long-video baseline protocol. The official implementations of SCAIL and UniAnimate-DiT operate on fixedlength clips using the temporal context available during training and do not provide native long-form generation. For a controlled long-sequence comparison, we extend both methods with the same training-free sliding-window denoising procedure. We initialize a single noise latent spanning the complete target sequence. At every denoising step, we slide each model’s native temporal context window across this latent with a fixed stride. Each window is conditioned on the same reference identity image and the temporally aligned local segment of the driving pose sequence. Predictions in overlapping regions are fused using triangular, or tent-shaped, weights that emphasize the temporal center of each window and suppress its boundaries. A single UniPC multistep ODE-solver update [40] is then applied to the fused full-sequence prediction, ensuring that the solver state evolves consistently along one global trajectory rather than independently within each window. This extension requires neither retraining nor positional-embedding extrapolation and is used uniformly for SCAIL and UniAnimate-DiT. EverAnimate and One-to-All are instead evaluated using the long-video generation procedures provided by their official implementations.

## 4.2. Quantitative Comparison

Short-horizon quality. Figure 5 reports quality and identity metrics across the temporal segments of the threeminute rollout. On the initial 0–30 s segment, LiveAnimate obtains the best ASE (2.823) and IQA (4.047). This result indicates that three-step distillation preserves strong perceptual quality under a real-time inference budget.

Table 2. Scaling efficiency of Ulysses sequence parallelism on H100 GPUs at 480×480. Results are measured on the threeminute benchmark. Latency denotes the time required to generate one 12-frame block. Speedup is relative to one GPU, and efficiency is speedup divided by the number of GPUs.
<table><tr><td>GPUs</td><td>Latency (ms)</td><td>FPS</td><td>Speedup</td><td>Efficiency</td></tr><tr><td>1</td><td>966.96</td><td>12.41</td><td>1.00×</td><td>100.0%</td></tr><tr><td>2</td><td>611.28</td><td>19.63</td><td>1.58×</td><td>79.1%</td></tr><tr><td>4</td><td>542.25</td><td>22.13</td><td>1.78×</td><td>44.6%</td></tr></table>

Long-horizon stability. The main advantage of LiveAnimate is its behavior as the rollout grows. Its IQA remains essentially unchanged, from 4.047 in the first segment to 4.026 in the final segment, and ASE remains stable from 2.823 to 2.824. DINO-S decreases by only 0.015, from 0.833 to 0.818. In contrast, One-to-All exhibits pronounced accumulation: IQA falls from 3.402 to 1.786, DINO-S from 0.769 to 0.328, and FID rises from 146.29 to 386.33. Wan2.2-Animate also loses identity similarity over time, with DINO-S decreasing from 0.795 to 0.770. These trends support our central claim that explicitly managing long-range context is important for streaming animation.

Comparison with offline long-video methods. SCAIL and UniAnimate-DiT attain higher identity similarity (DINO-S) than LiveAnimate throughout the rollout. As the qualitative comparison shows, however, both methods exhibit inter-frame flickering (Figure 3), whereas LiveAnimate remains stable over time. EverAnimate maintains competitive per-frame quality but is not a real-time system, requiring hours of offline computation for each three-minute sequence. LiveAnimate is thus the only evaluated method that sustains visibly consistent quality over the full rollout while generating in real time.

Sequence-parallel scaling. Table 2 evaluates Ulysses scaling on the same three-minute benchmark. Moving from one to two H100 GPUs increases throughput from 12.41 to 19.63 FPS, yielding a 1.58× speedup at 79.1% parallel efficiency. Expanding to four GPUs improves throughput only to 22.13 FPS: the speedup saturates at 1.78× and efficiency falls to 44.6%, as the additional all-to-all communication increasingly offsets the reduction in per-GPU attention computation. We therefore adopt two-way sequence parallelism by default, as it offers the best trade-off between throughput and GPU count.

Inference-time breakdown. Table 3 reports the steadystate latency breakdown on our three-minute benchmark. LiveAnimate processes each 12-frame block in 0.611 s, corresponding to 19.63 FPS. Because each block is emitted as soon as its driving poses arrive and the KV cache has a fixed size, both latency and memory are independent of the elapsed stream length. Denoising accounts for 75.07% of the runtime, followed by the Clean KV Update at 24.39%.

Table 3. Steady-state inference profile on the three-minute benchmark. Steady state denotes inference after compilation and warm-up. Throughput is computed from 12 RGB frames per block. Component times exclude VAE decoding.
<table><tr><td>Component</td><td>Avg. (s)</td><td>Min. (s)</td><td>Max. (s)</td><td>Share</td><td>FPS</td></tr><tr><td>Sink read</td><td>0.003303</td><td>0.003197</td><td>0.005136</td><td>0.540%</td><td>一</td></tr><tr><td>Denoising</td><td>0.458884</td><td>0.451726</td><td>0.476683</td><td>75.070%</td><td></td></tr><tr><td>Clean KV Update</td><td>0.149078</td><td>0.146292</td><td>0.168945</td><td>24.388%</td><td>一</td></tr><tr><td>Sink write</td><td>0.000011</td><td>0.000009</td><td>0.000023</td><td>0.0018%</td><td>一</td></tr><tr><td>Block total</td><td>0.611276</td><td>0.601543</td><td>0.631650</td><td>100.00%</td><td>19.63</td></tr></table>

![](images/e6150d32a35e51f31faa94d0fa852e3265f25cf0674d8164bf0256f9693a2ef6.jpg)  
Figure 6. Qualitative ablation over a three-minute rollout. Frames are sampled every 40 seconds from 20 to 180 seconds under the same reference image and driving-pose sequence.

The latter performs an additional forward pass at the clean timestep (t=0) and stores the resulting KV state for subsequent blocks. PR-Sink retrieval and bank maintenance together contribute only 0.542%, demonstrating that longrange pose-aware context incurs negligible overhead. VAE decoding is excluded from the block time; it is independent across blocks and can be pipelined on a separate device concurrently with DiT inference, so it does not extend the generation critical path. Under our evaluation settings, all evaluated baselines require approximately 2–5 hours to synthesize the same three-minute sequence. LiveAnimate therefore provides a substantial speed advantage, achieving real-time streaming while maintaining bounded memory over long sequences.

![](images/883d307ee3f4f5d1a9937e924a1ef62e54df0957bf3cfe57fd784e0e66b2c64e.jpg)  
Figure 7. Quantitative ablation over the three-minute rollout. Top: component and cache variants, including separate removal of the static and dynamic sinks. The w/o-DMD variant replaces DMD-based self-forcing distillation with conventional teacher forcing. Bottom: the sampling-step trade-off with the full PR-Sink configuration. Higher is better for ASE, IQA, and DINO-S; lower is better for FID and V-MAE.

## 4.3. Qualitative Comparison

Figure 3 presents the more challenging full-body example, which evaluates articulated motion, identity preservation, and background stability under large pose changes. The competing methods fail in visibly different ways. Oneto-All suffers a severe collapse of image quality as the rollout progresses. UniAnimate-DiT and SCAIL maintain the global structure but exhibit inter-frame flickering. Wan-Animate and EverAnimate remain stable in the early segments but show noticeable color drift and background drift at later timestamps. In contrast, LiveAnimate follows the driving poses while preserving both the subject’s appearance and the scene throughout the three minutes. The accompanying runtime comparison further highlights the efficiency gap: competing methods require approximately 2–5 hours to generate the 25-FPS sequence, whereas LiveAnimate completes it in about 4 minutes.

Figure 4 presents an upper-body example that isolates long-term identity and appearance preservation under subtler motion. UniAnimate-DiT, SCAIL, and Wan-Animate exhibit background flickering across the sequence, and One-to-All again suffers a severe drop in image quality. EverAnimate and LiveAnimate both remain stable and consistent over the full rollout, though only LiveAnimate does so in real time.

## 4.4. Ablation Study

Qualitative ablation. Figure 6 visualizes the three-minute rollout under each variant. Removing the Dynamic Sink largely preserves the scene but introduces localized appearance errors when poses recur, as highlighted around the face and shoulder. Removing the Static Sink is far more damaging: the subject and background change completely over the sequence, showing that pose-matched retrieval alone cannot provide a time-invariant identity anchor. Replacing DMDbased distillation with conventional teacher forcing causes illumination and appearance to fluctuate across the rollout, and disabling RoPE follow leads to progressive color and background shifts despite broadly correct poses. The full model remains consistent across recurring poses throughout the three minutes.

Quantitative ablation. Figure 7 reports the corresponding metric trajectories. Removing the Dynamic Sink moderately degrades long-horizon quality, reducing final-segment DINO-S from 0.818 to 0.805 and increasing FID from 100.90 to 106.52. Removing the Static Sink is substantially worse: final-segment DINO-S collapses to 0.693, yet the variant retains moderate frame-level scores (ASE 2.726, IQA 3.833), showing that frame-level metrics alone cannot diagnose identity failure. These trends confirm the complementary roles of the two sinks: the Static Sink supplies a stable global appearance anchor, while the Dynamic Sink restores pose-specific evidence that has left the Rolling Window. Disabling RoPE follow also weakens final-segment identity (DINO-S 0.801). Replacing DMDbased distillation with teacher forcing degrades the rollout steadily, with IQA falling from 3.849 to 3.438 and DINO-S from 0.780 to 0.679, indicating that self-forcing limits error accumulation under few-step sampling. Finally, four sampling steps yield the strongest frame-level scores but require an extra denoising pass, while two steps weaken longhorizon identity (final DINO-S 0.783); three steps provide the best efficiency–quality compromise.

## 5. Conclusion

We presented LiveAnimate, a streaming pose-driven human animation system that combines real-time generation with stable long-form quality. A two-stage training pipeline converts a pretrained bidirectional DiT into a block-causal generator and distills it to three sampling steps, trainable on a single 8×80GB GPU node, while Pose-Retrieval Sink Attention maintains long-range appearance context under a bounded cache. On the three-minute benchmark, LiveAnimate sustains nearly constant perceptual quality and identity at 19.63 FPS on two H100 GPUs, with memory and latency independent of stream duration, whereas offline baselines degrade visibly or require hours of computation. These results bring billion-scale video diffusion models within reach of interactive applications such as live streaming, telepresence, and virtual avatars.

Limitations and future work. LiveAnimate currently generates at 480×480 with three denoising steps, which limits visual fidelity and resolution, and it does not support multiperson scenes or large camera motion. Extending LiveAnimate toward higher-resolution streaming generation, multiperson animation, and camera-dynamic scenes is left to future work.

## References

[1] Alibaba Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 8

[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2

[3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 2

[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 8

[5] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 3

[6] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026. 3

[7] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025. 2, 3, 8

[8] Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson, Avinatan Hassidim, William T. Freeman, and Michael Rubinstein. Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation. In ACM TOG, 2018. 7

[9] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719, 2024. 2, 7

[10] Qijun Gan, Yi Ren, Chen Zhang, Zhenhui Ye, Pan Xie, Xiang Yin, Zehuan Yuan, Bingyue Peng, and Jianke Zhu. Humandit: Pose-guided diffusion transformer for longform human motion video generation. arXiv preprint arXiv:2502.04847, 2025. 3

[11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 8

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2

[13] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. 2

[14] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2024. 2, 3

[15] Yingcheng Hu, Haowen Gong, Chuanguang Yang, Zhulin An, Yongjun Xu, and Songhua Liu. Multianimate: Pose-

guided image animation made extensible. arXiv preprint arXiv:2602.21581, 2026. 3

[16] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In NeurIPS, 2025. 2, 3, 4

[17] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12753–12762, 2021. 7

[18] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In ICCV, pages 17191–17202, 2025. 3

[19] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video diffusion. arXiv preprint arXiv:2602.07775, 2026. 3

[20] Wuyang Li, Yang Gao, Mariam Hassan, Lan Feng, Wentao Pan, Po-Chien Luan, and Alexandre Alahi. Everanimate: Minute-scale human animation via latent flow restoration. arXiv preprint arXiv:2605.15042, 2026. 2, 3, 8

[21] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. In ICLR, 2026. 3

[22] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. In ICLR, 2026. 3

[23] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2

[24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image syn thesis with latent diffusion models. In CVPR, 2022. 2

[25] Shijun Shi, Jing Xu, Zhihang Li, Chunli Peng, Xiaoda Yang, Lijing Lu, Kai Hu, and Jiangning Zhang. One-to-all animation: Alignment-free character animation and image pose transfer. In CVPR, pages 4011–4021, 2026. 2, 3, 8

[26] Aliaksandr Siarohin, Stephane Lathuiliere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. In NeurIPS, 2019. 3

[27] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022. 8

[28] Zhengyan Tong, Chao Li, Zhaokang Chen, Bin Wu, and Wenjiang Zhou. Musepose: A pose-driven image-to-video framework for virtual human generation. https : / / github.com/TMElyralab/MusePose, 2024. Open source project. 3

[29] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-tovideo synthesis. In NeurIPS, 2018. 3

[30] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025. 2, 3, 8

[31] Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Youqing Fang, Yuwei Guo, Wenran Liu, Jing Tan, Kai Chen, Tianfan Xue,

Bo Dai, and Dahua Lin. Humanvid: Demystifying training data for camera-controllable human image animation. In NeurIPS, 2024. 7

[32] Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989. 3

[33] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels. In Proceedings ofthe 41st International Conference on Machine Learning, pages 54015–54029. PMLR, 2024. 8

[34] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, 2024. 4

[35] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498, 2024. 2, 3

[36] Wenhao Yan, Sheng Ye, Zhuoyi Yang, Jiayan Teng, Zhen-Hui Dong, Kairui Wen, Xiaotao Gu, Yong-Jin Liu, and Jie Tang. Scail: Towards studio-grade character animation via in-context learning of 3d-consistent pose representations. arXiv preprint arXiv:2512.05905, 2025. 2, 3, 8

[37] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, pages 6613–6623, 2024. 3

[38] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 3

[39] Jiaming Zhang, Shengming Cao, Rui Li, Xiaotong Zhao, Yutao Cui, Xinglin Hou, Gangshan Wu, Haolan Chen, Xu Yu, Limin Wang, and Kai Ma. Steadydancer: Harmonized and coherent human image animation with first-frame preservation. arXiv preprint arXiv:2511.19320, 2025. 3, 7

[40] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictor-corrector framework for fast sampling of diffusion models. In NeurIPS, 2023. 8

[41] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026. 3

[42] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In ECCV, 2024. 3

# LiveAnimate: Stable Long-Form Streaming Human Animation in Real-Time Supplementary Material

## 6. Additional Qualitative Results

We provide seven additional three-minute examples to complement the qualitative comparisons in Sec. 4. Figures 8 and 9 cover full-body and upper-body animation under diverse identities, clothing, backgrounds, camera framing, and motion patterns. For each example, we show the reference image together with the driving pose signal and generated frame at 20-second intervals from 20 to 180 seconds. These uniformly sampled results expose temporal changes throughout the rollout rather than selecting only adjacent or early frames.

The first two examples in Figure 8 contain large changes in limb configuration and body orientation, whereas the last two emphasize upper-body motion, facial variation, and gestures near the face. Despite these differences, the generated subjects remain recognizable and their clothing and backgrounds do not exhibit progressive changes over the sampled timestamps. Figure 9 further tests appearance preservation across a dim indoor scene, a bright outdoor scene, and a close-up portrait. The outputs track both repeated and newly observed poses while maintaining scenespecific appearance. Together with the quantitative threeminute evaluation in the main paper, these examples provide additional qualitative evidence that LiveAnimate supports stable long-form streaming across varied motion and visual conditions.

![](images/fffca548a1eafad5c3bb0edc689db559f5569f69f8299de5890f8ef979d6c170.jpg)  
Figure 8. Additional qualitative results on four three-minute sequences. For each example, the left column shows the reference image, while the rows on the right show the driving pose signals and corresponding outputs sampled every 20 seconds from 20 to 180 seconds. The examples include full-body dance sequences with large pose changes and upper-body sequences dominated by facial expressions and hand motion. Across the rollout, LiveAnimate follows the input poses while preserving the reference identity, clothing, and scene appearance.

Input image  
Input image  
![](images/6f33357172834a4735bc72e9a222db2c918d7962efdd12e3a8d78bfc73849368.jpg)

Result  
![](images/d36edc0504351b13d007da2f76af1636405fc8c615e8814e56350420622ddb8c.jpg)

![](images/0d989244082bc7acc20cc160fb9721b10dc804217d9e46c14930424f30ddf6a8.jpg)

Pose signal  
Result  
![](images/88a6ad519175b1179c1d63316ad06c4a378d72d9ff58b935a394da23c4f8b401.jpg)

Input image  
![](images/19ed6c2b0d2e82053cef84b50fb04a0023e8cd034c536f938d5c82a1e8cf37c8.jpg)

Result  
Pose signal  
![](images/f2517a5c9c59e0cb3c794b39767fee41231c4775cfb1eeeeb03f86d42c0b6d6d.jpg)  
Figure 9. Additional qualitative results on three three-minute sequences. We show the reference image, driving pose signal, and generated outputs at 20-second intervals. The sequences cover indoor full-body motion, an outdoor dance sequence, and an upper-body subject performing frequent hand gestures. LiveAnimate maintains the subject’s appearance and background while responding to substan tial changes in body configuration over the three-minute rollout.