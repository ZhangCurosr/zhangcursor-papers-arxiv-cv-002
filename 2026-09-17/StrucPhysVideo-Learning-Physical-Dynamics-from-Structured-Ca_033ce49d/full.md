# StrucPhysVideo: Learning Physical Dynamics from Structured Captions and Robot Actions

Awomo-WM Team<sup>1</sup>

## Abstract

Modeling physical dynamics—how objects move, interact, and change state—is central to video world models for embodied AI. We present StrucPhysVideo, a family of video world models that bridges physicsfocused data curation with language- and action-conditioned prediction of scene evolution. Our data pipeline combines motion-aware video segmentation, quality and content filtering, and physical relevance verification with structured annotations of objects, materials, and temporally localized interactions. By disentangling camera motion from object behavior and explicitly describing contact, deformation, and state transitions, the pipeline provides supervision grounded in observable physical events. Building on these data, we introduce StrucPhysVideo-TI2V, a sparse Mixture-of-Experts (MoE) text-image-to-video model trained with a curriculum that progressively emphasizes physical dynamics while retaining general-domain video data. StrucPhysVideo-TI2V achieves state-of-the-art performance on Physics-IQ Verified, scoring 45.5% and outperforming Cosmos3- Super-Image2Video by 2.8 percentage points. Caption ablations across backbones further demonstrate the effectiveness of physics-focused supervision. We further extend StrucPhysVideo-TI2V to StrucPhysVideo-IA2V, an interactive image-action-to-video world model that predicts visual outcomes from robot end-effector commands. Action conditioning, causal autoregressive generation, and few-step distillation enable incremental robot rollouts with only four denoising steps. Together, StrucPhysVideo advances physical dynamics modeling from image- and language-conditioned video prediction toward action-driven interaction.

Website: https://westlakedi-awomo.github.io/StrucPhysVideo-Page/ Github: https://github.com/westlakedi-awomo/StrucPhysVideo

## 1 Introduction

Physical AI requires models that connect visual observations with the consequences of actions. Such models should capture how objects persist, how interactions alter their states, and how scenes evolve in response to agent actions. Video provides natural observations of these processes over time, making it a valuable medium for learning representations of the physical world. Developing such video models requires training data that preserve meaningful physical interactions, annotations grounded in physical events, and modeling frameworks that capture how actions drive scene evolution.

Preparing such data requires attention to the visual evidence available across time. Consider a video in which a gripper approaches a cup, grasps it, and lifts it from a table. A cut made during the grasp can separate the approach from the lift, leaving neither clip with the full progression. An editing transition may instead connect the cup on the table to the cup already in the air, without showing how it moved between these states. Poor exposure or persistent blur may also hide the contact point, even when the clip contains the full interaction. Filtering also requires care: a label printed on a bottle belongs to the physical scene, while an overlaid banner may obscure the interaction. Rejecting all clips with detected text would remove both. These cases motivate curation that considers temporal continuity, visual quality, and the context of the content being filtered.

Language introduces another difficulty. A caption such as “a robot handles a cup” describes the general activity but does not specify which cup is involved, whether it is pushed or lifted, or how its state changes. In a scene with several objects, these details are necessary to ground an instruction such as “Lift the red cup from the table.” The order of approaching, grasping, and lifting also matters because each stage describes a different part of the interaction. Camera movement adds further ambiguity: a stationary cup can move across the image as the camera pans. Language supervision, therefore, needs to identify the participating objects, describe their actions in temporal order, and distinguish object behavior from camera motion. When contact is hidden by a hand or missed by frame sampling, the description should remain within the available visual evidence.

![](images/6e3c0af983b11dfd8c0889804f4d8703cd553ebe31429a8b8467d58699776c25.jpg)  
Figure 1. Physics-IQ Verified benchmark Results [1]. StrucPhysVideo achieves the highest Physics-IQ Verified score among all compared models in the text-image-to-video setting. \* denotes our reproduction, and results for all other models are taken from the benchmark snapshot dated 16 September 2026 [2].

Even with a clear instruction, the execution of an interaction can remain ambiguous. For example, “Move the cup to the other side of the table” could involve sliding it along the surface or lifting it, carrying it, and placing it down. These executions differ in their contact sequence and intermediate states, although they may produce the same final position. Action information provides an additional condition for guiding how the scene unfolds. If the supplied actions correspond to a lift, a generated sequence in which the cup continues to slide would be inconsistent with those actions. For embodied AI, the usefulness of the generated video depends on whether these visual changes follow the supplied interaction throughout the sequence.

In this work, we present StrucPhysVideo, a Physical AI model that connects the curation of our privately collected video data with language- and action-conditioned generation. We develop a cleaning and curation pipeline to prepare clips with observable interactions, and an language annotation pipeline that organizes scene context, entities, and temporal behaviors into structured descriptions. These descriptions provide a basis for constructing instructions grounded in the video content. We further add action information to guide generation, with the aim of maintaining consistency between the instruction, the supplied actions, and the resulting visual changes. Representative generations across general-domain, physical-world, and embodied scenarios are shown in Figure 2. Together, these components connect interaction-focused data curation and structured language supervision with language- and action-conditioned video generation. As a result, StrucPhysVideo achieves strong physical dynamics prediction performance, scoring 45.5% on Physics-IQ Verified and ranking first among the 17 models in our comparison based on the 16 September 2026 benchmark snapshot, as shown in Figure 1.

We make three contributions:

• Video cleaning and curation. We propose a pipeline that combines shot detection, motion-aware segmentation, technical quality checks, and content filtering with targeted multimodal review to retain useful observations of physical interactions.

• A structured language annotation pipeline. We construct structured descriptions of scenes, entities, and temporally localized actions, separating camera motion from object behavior to produce instructions that are

![](images/419446cb4c23fbaa54bc4e7dba17015d5084f3b2356b9fce2ae745c73da21aad.jpg)  
Figure 2. Representative StrucPhysVideo generations under both the text-image-to-video (TI2V) and image-action-to-video (IA2V) settings, spanning general-domain, physical-world, and embodied scenarios.

faithfully grounded in the video content.

• Action-conditioned video generation. We add action information to guide generation in StrucPhysVideo, with the objective of producing interactions consistent with the supplied actions while preserving visual quality and temporal continuity.

The remainder of this report is organized as follows. Section 2 describes the model architecture and post-training procedures, including action-conditioned robot video generation. Section 3 details the data sourcing and curation pipeline, structured physical captioning, and physical taxonomy. Section 4 presents evaluations of physical dynamics prediction, caption ablations, and controlled robot-video comparisons. Finally, Section 5 discusses limitations and directions for future research.

## 2 Method

## 2.1 StrucPhysVideo-TI2V

StrucPhysVideo-TI2V is a latent generative model that predicts how an observed scene evolves under an image and language condition. Given a reference image $I _ { 0 }$ and a caption $c ,$ it models

$$
p _ { \theta } ( I _ { 1 : T } \mid I _ { 0 } , c ) ,\tag{1}
$$

![](images/2533becd109c2ce965f4392814daed555d513bf9cffeca937943285046fae9ab.jpg)  
Figure 3. StrucPhysVideo-TI2V overall architecture. The frozen Qwen3-VL-32B encoder jointly encodes the text prompt and reference image, while the frozen VAE encodes the target video. The first clean video latent is retained as a reference and the remaining latents are corrupted along the flow-matching path. Multimodal condition tokens and video tokens interact through joint self-attention in the StrucPhysVideo Transformer. Each block combines timestep-modulated attention with routed and shared experts, and the model predicts the latent velocity field used to supervise future-frame generation.

while preserving the subject identity, spatial layout, camera geometry, and visual style established by $I _ { 0 }$ . The model treats the reference image as both a semantic condition and a boundary condition on the generated trajectory. This dual role directs model capacity toward learning plausible motion and state changes rather than reconstructing already observed visual information.

Figure 3 summarizes the overall architecture. A frozen Qwen3-VL-32B encoder [3] jointly processes c and $I _ { 0 }$ to produce a multimodal condition $\mathbf { h } _ { c } = \mathscr { Q } ( c , I _ { 0 } )$ . In parallel, a frozen Wan video variational autoencoder (VAE) [4] maps the target video $\mathbf { V } = \{ I _ { 0 } , I _ { 1 } , \ldots , I _ { T } \}$ to a normalized clean latent ${ \bf z } _ { 0 } = \mathcal { N } ( \mathcal { E } ( { \bf V } ) )$ . The trainable StrucPhysVideo backbone receives the multimodal condition, a corrupted video latent, and the flow timestep, and predicts a velocity field in latent space. The VAE decoder D maps the final latent trajectory back to RGB video.

## 2.1.1 Dual-Path Reference Conditioning

The semantic path uses the final hidden states of Qwen3-VL-32B rather than generating an intermediate rewritten caption. Because the encoder observes both modalities, $\mathbf { h } _ { c }$ captures visible entities and appearance together with linguistic descriptions of motion, temporal order, and camera behavior. A learned projection maps this condition into the backbone feature space. The latent path supplies a complementary low-level constraint. During training, the reference latent ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$ is taken from the first temporal position of the VAE encoding of the complete target video, avoiding a mismatch between independently encoded images and the temporal boundary behavior of the video VAE.

After spatiotemporal patchification, video and condition tokens are concatenated and processed by non-causal joint self-attention. Unlike a design that injects the condition through occasional cross-attention layers, this formulation allows language, reference-image features, and evolving video features to interact in every Transformer block. Three-axis rotary position embeddings, adapted from RoPE [5], distinguish temporal and spatial locations for video tokens while retaining an ordered position convention for condition tokens.

## 2.1.2 Sparse Spatiotemporal Transformer

The trainable backbone is an approximately 30-billion-parameter spatiotemporal Transformer with 48 blocks, following the sparse video-pretraining design of LingBot-Video [6]. Every feed-forward sublayer uses a sparse mixture-of-experts (MoE) module containing routed experts and an always-active shared expert. For a token representation x, the router selects a restricted top-k set $\tau ( \mathbf { x } )$ , and the MoE output is

$$
\mathrm { M o E } ( \mathbf { x } ) = \sum _ { i \in \mathcal { T } ( \mathbf { x } ) } \alpha _ { i } E _ { i } ( \mathbf { x } ) + E _ { \mathrm { s h a r e d } } ( \mathbf { x } ) ,\tag{2}
$$

where $E _ { i }$ denotes a routed expert and $\alpha _ { i }$ is its normalized routing weight. Group-constrained routing limits the candidate experts before top-k selection. This architecture increases model capacity while keeping the amount of expert computation per token bounded. Timestep-conditioned adaptive normalization, following the conditioning strategy used in diffusion Transformers [7], and gated residual connections modulate both the attention and MoE branches across the flow trajectory.

## 2.1.3 First-Frame-Constrained Flow Matching

We use the flow-matching formulation [8] and the same notation as the action-conditioned model in Section 2.2. For a clean video latent $\mathbf { z } _ { 0 }$ , Gaussian noise ϵ, and flow time τ, the corrupted state and target velocity are

$$
{ \bf z } _ { \tau } = ( 1 - \tau ) { \bf z } _ { 0 } + \tau \epsilon , \qquad { \bf v } ^ { \star } = \epsilon - { \bf z } _ { 0 } .\tag{3}
$$

Before each denoiser evaluation, the first temporal position of ${ \bf z } _ { \tau }$ is replaced by ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$ . The model therefore observes the clean initial state while predicting $\mathbf { v } _ { \theta } ( \mathbf { z } _ { \tau } , I _ { 0 } , c )$ for the full latent sequence. The reference position is excluded from the regression loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { \tau , \epsilon } \left[ w ( \tau ) \left. \mathbf { M } \odot ( \mathbf { v } _ { \theta } - \mathbf { v } ^ { \star } ) \right. _ { 2 } ^ { 2 } \right] , } \end{array}\tag{4}
$$

where M masks the reference position and retains all future positions. Thus, the initial latent participates in contextual reasoning but is not itself a denoising target. Appendix ${ \mathrm { A } } ,$ Algorithm 1, gives the corresponding single-step training procedure.

## 2.1.4 Inference

At inference, Qwen3-VL-32B again encodes $( c , I _ { 0 } )$ , and the VAE image-encoding path produces ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$ . Sampling begins from a Gaussian video latent whose first temporal position is replaced by this clean reference. A Flow-UniPC solver, adapted from the UniPC predictor-corrector framework [9], integrates the predicted velocity field, and the reference position is re-clamped after every solver update to prevent numerical drift. When classifier-free guidance [10] is used, conditional and negative-condition predictions are combined as

$$
\widehat { \mathbf { v } } = \mathbf { v } _ { \mathrm { u n c o n d } } + s _ { t } \left( \mathbf { v } _ { \mathrm { c o n d } } - \mathbf { v } _ { \mathrm { u n c o n d } } \right) .\tag{5}
$$

The final latent is inverse-normalized and decoded by D. Appendix A, Algorithm $^ { 2 , }$ provides the full inference procedure. Concrete training resolutions, frame counts, batch sizes, optimization settings, precision choices, and distributed execution details are reported in Section 4.1.1.

## 2.2 StrucPhysVideo-IA2V

We further extend StrucPhysVideo-TI2V into an interactive robot world model, StrucPhysVideo-IA2V, that predicts how visual scenes evolve in response to robot command trajectories. To enable real-time interaction, we convert the bidirectional video DiT into a causal few-step generator through a three-phase fine-tuning pipeline.

## 2.2.1 Problem Formulation and System Design

Let $I _ { 0 }$ be the first RGB observation and let ${ \bf a } _ { 0 : { K - 1 } }$ be a sequence of robot actions. Our image-and-action-to-video (IA2V) model represents

$$
p _ { \theta } ( I _ { 1 : T } \mid I _ { 0 } , \mathbf { a } _ { 0 : K - 1 } ) ,\tag{6}
$$

while the text-image-action-to-video (TIA2V) variant additionally conditions on a textual command c. Both variants use the same visual backbone and action interface; IA2V fixes the text context to the embedding of the empty string.

The system is organized into three phases. First, action injection adapts a pretrained bidirectional backbone to follow end-effector commands over a fixed video window. Second, teacher forcing AR Diffusion training and Causal ODE converts the resulting model into a streaming autoregressive generator. Finally, asymmetric DMD performs distribution-level refinement by aligning the autoregressive generator with the high-quality bidirectional teacher, recovering generation quality lost during causalization.

## 2.2.2 Robot Data and Action Representation

Each sample contains $\mathrm { ~ a ~ } 9 6 0 \times 7 3 6$ head-camera clip and the commanded dual-arm EEF trajectory, sampled at 5 FPS and 15 Hz, respectively. For arm $r \in \{ L , R \}$ , the input vector for the model is

$$
\mathbf { a } _ { k } ^ { r } = [ \mathbf { p } _ { k } ^ { r } , \rho ( \mathbf { q } _ { k } ^ { r } ) , g _ { k } ^ { r } ] \in \mathbb { R } ^ { 1 0 } ,\tag{7}
$$

where $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ is Cartesian position, q is an xyzw quaternion, $g$ is the gripper target, and $\rho ( \mathbf { q } ) \in \mathbb { R } ^ { 6 }$ contains the first two columns of the corresponding rotation matrix. This continuous 6D representation avoids the quaternion sign ambiguity and is well suited to regression in neural networks [11]. Concatenating the left and right arms yields

$$
\begin{array} { r } { { \bf a } _ { k } = [ { \bf a } _ { k } ^ { L } , { \bf a } _ { k } ^ { R } ] \in \mathbb { R } ^ { 2 0 } . } \end{array}\tag{8}
$$

Every coordinate is normalized with corpus-level mean and standard deviation.

Frame-aligned action windows. At 15 Hz, every three action commands correspond to the transition between two adjacent video frames sampled at 5 fps. Thus, commands $[ 3 i , 3 i + 3 )$ correspond to the transition from video frame i to frame $i + 1$ . Since each video latent represents four consecutive video frames, it is aligned with a window of twelve action commands.

Causal action encoder. As shown in Figure 4, the normalized action trajectory ${ \bf a } _ { 0 : { \cal K } - 1 }$ is first projected into a 512- dimensional embedding space, followed by a four-layer Transformer encoder with a strict causal attention mask:

$$
{ \bf u } _ { 0 : K - 1 } = E _ { \mathrm { a c t } } ( \mathrm { L N } ( { \bf a } _ { 0 : K - 1 } ) W _ { \mathrm { i n } } + { \bf p } _ { 0 : K - 1 } ) , \qquad { \bf u } _ { k } \mathrm { d e p e n d s \ o n l y \ o n \ } { \bf a } _ { 0 : k } .\tag{9}
$$

Here, $\mathbf { p } _ { k }$ denotes the learned positional embedding for action step $k .$ The encoder employs a strict upper-triangular causal attention mask, ensuring that each action token attends only to past and current commands. Finally, attention pooling is performed within each action window to produce a frame-level action context $\mathbf { h } _ { f }$ , allowing each video latent to condition exclusively on the action commands associated with its corresponding temporal interval.

## 2.2.3 Bidirectional IA2V Training

Reference-image conditioning. For a clean latent video $\mathbf { z } _ { 0 }$ and Gaussian noise ϵ, rectified-flow corruption constructs

$$
{ \bf z } _ { \tau } = ( 1 - \tau ) { \bf z } _ { 0 } + \tau \epsilon , \qquad { \bf v } ^ { \star } = \epsilon - { \bf z } _ { 0 } ,\tag{10}
$$

and trains the network to predict the velocity field [8]. We replace the first noisy latent by the clean reference latent before every denoiser call and exclude it from the loss:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { \tau , \epsilon } \left[ \frac { 1 } { N } \sum _ { f = 1 } ^ { N } w ( \tau ) \left. \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { \tau } , I _ { 0 } , \mathbf { a } , c ) _ { f } - \mathbf { v } _ { f } ^ { \star } \right. _ { 2 } ^ { 2 } \right] .\tag{11}
$$

We use token-wise diffusion time-steps; reference tokens are marked with $\tau = 0$ , while generated tokens use the sampled timestep. At inference, the solver updates the whole latent tensor, so the reference latent is re-clamped after every solver step. This preserves the initial image exactly in latent space.

![](images/c569f0b395107afadb0ed1addec1a808c86fe820773acd99b7d58d1ab39b2a65.jpg)  
Figure 4. StrucPhysVideo-IA2V architecture. A causal action encoder maps raw action commands into latent action tokens and pooled contexts aligned with each video latent frame. In each DiT block, the pooled context modulates the video queries through AdaLN, while the corresponding action window provides the keys and values for cross-attention. A gated residual integrates action-conditioned features into the video stream.

Per-block AdaLN cross-attention adapter. As illustrated in Figure 4, we inject action control after video self-attention and before text cross-attention in every DiT block. For the video tokens $\mathbf { x } _ { f }$ belonging to latent frame $f ,$ a low-rank modulation network predicts shift, scale, and gate vectors from $\mathbf { h } _ { f } \colon$

$$
( \Delta \mu _ { f } , \Delta \pmb { \sigma } _ { f } , \mathbf { g } _ { f } ) = M _ { \ell } ( \mathbf { h } _ { f } ) ,\tag{12}
$$

$$
\widetilde { \mathbf { x } } _ { f } = ( 1 + \Delta \pmb { \sigma } _ { f } ) \odot \mathrm { L N } ( \mathbf { x } _ { f } ) + \Delta \pmb { \mu } _ { f } ,\tag{13}
$$

$$
\mathbf { r } _ { f } = W _ { o } \mathrm { A t t n } \big ( W _ { q } \widetilde { \mathbf { x } } _ { f } , W _ { k } \mathbf { u } _ { \mathcal { W } _ { f } } , W _ { v } \mathbf { u } _ { \mathcal { W } _ { f } } \big ) ,\tag{14}
$$

$$
\mathbf { x } _ { f }  \mathbf { x } _ { f } + \operatorname { t a n h } ( \mathbf { g } _ { f } ) \odot \mathbf { r } _ { f } .\tag{15}
$$

Specifically, AdaLN first modulates the visual queries according to the corresponding trajectory. Cross-attention then retrieves fine-grained action tokens as keys and values, and the resulting action features are injected into the video stream through a gated residual connection. The gate is zero-initialized to preserve the functionality of the pretrained backbone at initialization. The latent-zero frame serves as the observed reference and therefore receives no action residual. During training, we use a learning rate of $2 \times 1 0 ^ { - 6 }$ for the pretrained backbone and a 25× larger learning rate for the newly initialized action modules.

Unconditional Dropoutfor Teacher Guidance. During bidirectional action-injection training, we apply unconditional dropout to enable action CFG [10] in the teacher used for subsequent distillation. With probability $p _ { a }$ , the complete action sequence is replaced by a learned null-action token. On the mixed general-domain T2V/TI2V data, text dropout replaces the complete text context with the empty-prompt context with probability $p _ { t }$ ; IA2V samples already use empty text. The text and action dropout masks are sampled independently where both conditions are present. This trains the conditional and null-condition branches needed to construct a teacher signal with stronger action guidance. Let C denote supplied text plus action, N negative text plus action, T supplied text plus null action, and $U$ negative text plus null

action. Teacher-side guidance supports

$$
\widehat { \mathbf { v } } _ { \mathrm { t e x t } } = N + s _ { t } ( C - N ) ,\tag{16}
$$

$$
{ \widehat { \mathbf { v } } } _ { \mathrm { a c t i o n } } = T + s _ { a } ( C - T ) ,\tag{17}
$$

$$
\widehat { \mathbf { v } } _ { \mathrm { j o i n t } } = U + s _ { t } ( T - U ) + s _ { a } ( C - T ) .\tag{18}
$$

These guidance scales describe the teacher, not the deployed student. Subsequent distillation, as described in the next section, transfers the teacher’s guidance into a single-branch causal few-step generator.

## 2.2.4 Causalization and Few-Step Distillation

The bidirectional model provides a high-quality, action-controllable teacher but cannot stream efficiently. Following Causal Forcing and Causal Forcing++ [12, 13], we convert it into a few-step autoregressive model in three stages. The same first-image and EE20 action conditions are supplied throughout all stages.

Stage 1: autoregressive diffusion training. We initialize a causal model from the bidirectional checkpoint, partition the latent sequence into temporal chunks, and replace full temporal attention by a causal mask. For chunk i, clean ground-truth history $\mathbf { z } _ { \mathrm { g t } } ^ { < i }$ is concatenated with a noised current chunk $\mathbf { z } _ { \tau } ^ { i }$ . Teacher-forced flow matching trains

$$
{ \bf v } _ { \mathrm { A R } } ( { \bf z } _ { \tau } ^ { i } , { \bf z } _ { \mathrm { g t } } ^ { < i } , \tau , I _ { 0 } , { \bf a } )\tag{19}
$$

to denoise the current chunk while attending only to its past. This bridges the architecture gap before few-step distillation.   
The resulting model is causal but still multi-step and is exposed to clean rather than self-generated history.

Stage 2: causal ODE initialization. A bidirectional teacher can access future frames that are unavailable to a causal student, creating a mismatch in their conditioning information. Following Causal Forcing, we use the autoregressive diffusion model as the teacher for few-step initialization. For guided distillation, we collect probability-flow ODE trajectories using CFG-guided predictions from this autoregressive teacher, and train a few-step generator $G _ { \phi }$ to predict the corresponding clean endpoints from intermediate states at selected timesteps $\tau \in S { \mathrm { : } }$

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } \left[ \left\| G _ { \phi } ( \mathbf { z } _ { \tau } ^ { i } , \mathbf { z } _ { \mathrm { g t } } ^ { < i } , \tau , I _ { 0 } , \mathbf { a } ) - \mathbf { z } _ { 0 } ^ { i } \right\| _ { 2 } ^ { 2 } \right] .\tag{20}
$$

Here, $\mathbf { z } _ { 0 } ^ { i }$ denotes the teacher-generated endpoint for chunk i, and $\mathbf { z } _ { \mathrm { g t } } ^ { < i }$ provides the ground-truth history. Action conditioning is restricted to the current and preceding action windows. This trajectory regression transfers the teacher’s guided flow map into the single-branch student, jointly distilling the effect of CFG and the multi-step sampling process into a causal few-step initialization for subsequent distribution matching distillation.

Stage 3: asymmetric distribution matching. The causal initializer inherits the quality ceiling and exposure bias of the autoregressive teacher. We therefore self-roll out the few-step student to obtain ez, perturb it to $\widetilde { \mathbf { z } } _ { \tau }$ , and apply asymmetric distribution matching distillation (DMD) [14, 15]:

$$
\nabla _ { \phi } \mathcal { L } _ { \mathrm { D M D } } = - \mathbb { E } _ { \widetilde { \mathbf { z } } , \tau } \left[ \left( s _ { \mathrm { r e a l } } ( \widetilde { \mathbf { z } } _ { \tau } , \tau , I _ { 0 } , \mathbf { a } ) - s _ { \mathrm { f a k e } } ( \widetilde { \mathbf { z } } _ { \tau } , \tau , I _ { 0 } , \mathbf { a } ) \right) \frac { \partial \widetilde { \mathbf { z } } } { \partial \phi } \right] .\tag{21}
$$

Here $s _ { \mathrm { r e a l } }$ denotes the CFG-guided score supplied by the frozen high-quality bidirectional teacher, while $s _ { \mathrm { f a k e } }$ denotes the conditional score of the student rollout distribution estimated by an online diffusion model. The asymmetry is intentional: generation remains causal, whereas the real-score teacher may use bidirectional context to pull the complete rollout toward the higher-quality target distribution. Training on self-generated histories exposes the student to its deployment distribution and directly addresses autoregressive error accumulation. The resulting model combines a causal action interface, chunk-wise rollout, and few denoising evaluations per chunk for low-latency interaction.

## 3 Physical Data Pipeline

Physical world models learn from visual observations of objects moving, interacting, and changing state. Network videos capture diverse physical processes, but editing transitions can interrupt an interaction, persistent overlays can obscure it, and poor image quality can hide the relevant visual evidence. Preparing these videos for training therefore requires both clip curation and descriptions that identify the objects, actions, and temporal progression of each physical event.

![](images/8a18870a679434b1568fc4a48c9a77c0a1d59178a9535ac50aa970ad4c0b405a.jpg)  
Figure 5. Physical data pipeline. Source videos undergo Spatiotemporal Video Splitting, Technical Quality Filtering, and Content Purity Filtering. Physical Video Captioning first performs physical verification and generates a structured caption, then produces physical tags in a second model call. The resulting clips and annotations provide candidates for final dataset selection.

Our pipeline combines clip curation with structured captioning and tagging (Figure 5). Section 3.1 describes the source collection, and Section 3.2 explains how source videos are split and filtered into candidate clips. Section 3.3 presents physical verification and the two annotation calls, while Section 3.4 defines the accompanying tagging record. Section 3.5 summarizes the annotated outputs and their use in final dataset selection.

## 3.1 Data Sourcing

We assemble the input collection from two sources: agentic retrieval and domain-specific providers. For agentic retrieval, a sourcing agent uses the physical taxonomy (Section 3.4) to expand queries across physical interactions, environments, and manipulation tasks. Queries cover processes such as granular flows, multi-body impacts, tool use, and elastic deformation. An initial relevance filter screens the retrieved videos before clip curation.

Domain-specific providers contribute physical-interaction and robotics footage that complements the retrieved collection. Together, these sources provide diverse input video for the curation pipeline.

## 3.2 Video Curation Pipeline

The Video Curation Pipeline begins with Spatiotemporal Video Splitting (Section 3.2.1) to produce candidate clips. Technical Quality Filtering (Section 3.2.2) and Content Purity Filtering (Section 3.2.3) then remove visual defects and distracting content before physical verification and annotation.

## 3.2.1 Spatiotemporal Video Splitting

Uniform splitting can cut through an interaction or combine frames from different shots. We use Shot Boundary Detection to identify continuous shots, followed by Motion-Aware Fixed Window Selection to extract fixed-duration clips within each shot while favoring boundaries with low motion.

Shot Boundary Detection. We combine PySceneDetect [16] with TransNetV2 [17] to detect abrupt cuts and gradual transitions. Photometric changes provide candidate boundaries, while TransNetV2 captures transitions across multiple frames. Nearby detections are consolidated into transition intervals. Comparing frames before and after each interval suppresses transient flashes that return to the same scene. A short guard interval on either side of a confirmed transition excludes blended frames. Subsequent clip selection stays within the resulting shot boundaries.

Motion-Aware Fixed Window Selection. Within each shot, we compute motion energy as the temporally smoothed mean magnitude of Farnebäck optical flow [18]. The flow computation restarts at each shot boundary. This signal guides the placement of fixed-duration clips: lower motion at the endpoints reduces action truncation, while stronger motion within a clip favors active physical processes.

For a target duration $\ell ,$ we divide a shot into non-overlapping candidate intervals, each large enough to contain one clip. Within a candidate interval $[ a , b ]$ , the clip start time is selected by

$$
u ^ { * } = \underset { a \leq u \leq b - \ell } { \arg \operatorname* { m i n } } \left[ B ( u ) - \lambda _ { c } A ( u ) + \lambda _ { p } D ( u ) \right] .\tag{22}
$$

Here, $B ( u )$ is the normalized motion energy at the two clip endpoints, $A ( u )$ is the normalized motion energy accumulated within the clip, and $D ( u )$ is the squared normalized distance from the center of the feasible start-time range. The weights $\lambda _ { c }$ and $\lambda _ { p }$ balance motion coverage and placement. Each selected clip remains inside its candidate interval, so clips do not overlap or cross shot boundaries. Shots shorter than the target duration are discarded, and residual footage outside complete windows is omitted.

Fixed-duration selection is the primary mode used in this pipeline. As an alternative, the splitter supports a variableduration mode in which dynamic programming partitions each shot under minimum and maximum duration constraints, balancing clip lengths while penalizing cuts through active motion.

## 3.2.2 Technical Quality Filtering

Clips with valid temporal boundaries may still contain visual defects that obscure physical interactions. Technical Quality Filtering checks Input Validity, Visual Quality, and Temporal Quality. Representative visual defects addressed by these checks are shown in Figure 6.

Input Validity. We first verify that the clip can be decoded and that its resolution and duration meet the input requirements. Clips failing these checks are rejected before further analysis.

Visual Quality. For black frames, solid-color frames, and exposure defects, we measure grayscale intensity, spatial pixel variation, and the proportions of dark and saturated pixels across sampled frames. Their prevalence over time distinguishes clips dominated by uninformative or poorly exposed frames from clips containing brief lighting changes. Persistent defects cause rejection. Blur is assessed using the distribution of Laplacian variance across frames: a clip is rejected when both its median sharpness and lower-tail sharpness are low, preserving fast interactions that contain only brief motion blur.

Temporal Quality. We detect frame stuttering and camera shake. Consecutive-frame differences identify nearduplicate frames, and clips with a high proportion of repeated frames are rejected. For camera shake, Lucas–Kanade feature tracking [19] and RANSAC [20] estimate interframe camera motion. Subtracting a smoothed camera trajectory isolates rapid translation, rotation, and scale fluctuations. Clips with strong, persistent residual jitter are rejected, while smooth pans and tracking shots are retained.

Checks are applied from inexpensive validation to more costly motion analysis, stopping once a clip meets a rejection condition. Retained clips proceed to Content Purity Filtering.

![](images/bf30d853fa47eb943e7c1ffb7aa5464c3c41a8939c20f8eb2a44d271fe218987.jpg)  
Figure 6. Examples of visual defects addressed by Technical Quality Filtering: solid-color frames, severe exposure defects, and persistent blur. Input-validity and temporal-quality checks are described in the text.

![](images/7c8592edb566ef58315b7d56194ad56fa743684632f5e2d6d35a30d5d64f8ea4.jpg)  
rface = 0.6589

![](images/c211217d68d3e3ba56ac2583063908b1258422f2a1759df51908ff74dea577bf.jpg)  
rface = 1.0000

![](images/b812c2b3f330416672de88790756de37a060195c5573c9fd771a3e59afccf450.jpg)  
rface = 1.0000

![](images/cdf0a35d93e7db77e64a9aead0945bb1df811b4218ea0669bc76a9e0dfeb4ce4.jpg)  
rface = 1.0000  
Figure 7. Face and Talking-Head Filtering rejects presenter-dominated clips while retaining clips with distant operators and human–object interactions.

## 3.2.3 Content Purity Filtering

Clear, stable clips can still be dominated by talking heads, screen recordings, or post-production overlays. Content Purity Filtering uses four components: Face and Talking-Head Filtering, Screen Content Screening, Text and Watermark Detection, and VLM Text Review. The first three identify unwanted content or trigger further review; VLM Text Review resolves ambiguous text and screen-content detections.

Face and Talking-Head Filtering. YuNet [21] detects faces and facial landmarks across sampled frames. Face orientation, area, and temporal persistence identify clips dominated by frontal presentations and interviews. Combined with talking-head cues, these measurements distinguish presenter-focused footage from clips in which an operator participates in an object interaction (Figure 7).

Screen Content Screening. A CLIP encoder [22] compares sampled frames with prompts for screen recordings, presentation slides, and studio content, using physical-interaction prompts as a reference. Relative semantic scores identify screen-dominated content and route ambiguous UI detections to VLM Text Review.

Text and Watermark Detection. An EasyOCR-based text-detection branch [23] detects text regions and tracks their position, size, and persistence across frames. Stable text tracks in frame corners identify screen-locked watermarks, while recurring text near frame borders identifies subtitle and banner candidates. Clear watermark detections cause rejection; other text detections proceed to VLM Text Review to distinguish overlays from text on physical objects.

![](images/828f10b55665b4414c5e4bb2879ca232691c4c19b5283958c3766f8dadb5b0d2.jpg)  
Figure 8. VLM Text Review distinguishes text on physical objects from post-production overlays. Clips containing authentic scene text are retained for physical verification.

VLM Text Review. Qwen3-VL-4B [3] reviews six temporal keyframes arranged in a contact sheet and returns one of five classes: scene\_text, overlay\_ui, both, no\_text, or uncertain. Clips classified as scene\_text or no\_text continue to physical verification; the remaining classes are rejected. This step preserves labels, gauges, and signs that belong to the physical scene while removing post-production text and UI (Figure 8).

## 3.3 Physical Video Captioning

Content Purity Filtering removes distracting content, while Physical Video Captioning determines whether a clip contains physical processes and describes their progression. Qwen3.6-27B [24] produces a structured caption with four information groups:

1. Global Scene Dynamics: The environment, initial scene, and progression of the main physical interaction.

2. Camera Motion and Illumination: Camera movement, shot scale, viewing angle, composition, and lighting, described separately from object motion.

3. World Knowledge and Material Properties: Relevant scene knowledge, object materials, and spatial relationships grounded in the clip.

4. Entity-Level Temporal Behaviors: The participating objects and actors, with timestamped actions describing contact, manipulation, deformation, motion, and changes of state.

Appendix B.1 presents a real structured-caption example using field names aligned with these four information groups. It also shows the exact compact JSON object serialized as the input to the StrucPhysVideo text encoder.

To produce these descriptions and the accompanying tags, Frame Preparation establishes the shared inputs, Two-Call Annotation generates the captions and tags, and Output Validation checks the resulting records.

Frame Preparation. We uniformly extract 24 frames across each clip and retain their native timestamps. Frames are resized to a maximum long-edge dimension of 768 pixels. The timestamped frames and clip metadata are stored in an immutable cache and used by both annotation calls.

![](images/9d347d54852e63aaae8d1d693e0683249a28c121b933a500255efd29ef775a58.jpg)  
Figure 9. Physical verification in the first annotation call. Top row: rejected clips dominated by static product staging and decorative hand gestures. Bottom row: retained clips containing physical interactions. Retained clips receive complete structured captions and proceed to Closed-Set Tag Generation.

Two-Call Annotation. The first call verifies physical relevance and generates the caption; clips that pass this check proceed to the second call for tag generation:

1. Physical Verification and Caption Generation. Qwen3.6-27B receives the timestamped frames, clip metadata, and domain tags. It returns the Boolean flag is\_physics\_related with supporting visual evidence and, for a physical clip, the complete structured caption containing the four information groups above. Clips dominated by static product staging or decorative gestures are rejected at this step (Figure 9).

2. Closed-Set Tag Generation. For clips passing physical verification, a second call receives the cached timestamped frames and clip metadata, together with the first-call caption. It generates tags using the closed-set taxonomy and attribute vocabularies described in Section 3.4. The frames remain the primary evidence when the caption or external metadata differs from the visible content. The tags are merged with the caption into one annotation record.

Output Validation. Validation checks inference completion, JSON structure, required fields, closed-set tag values, and action timestamps within the clip duration. The annotation manifest aligns each retained clip with its caption and tags before final dataset selection.

## 3.4 Physical Taxonomy and Structured Tagging

For clips retained after video curation and physical verification, captions describe the scene, participating objects, and temporal progression of events. Organizing these clips into a training dataset also requires consistent categories for retrieval and composition analysis. Since free-form captions may describe the same activity or phenomenon in different terms, we complement them with structured tags drawn from predefined vocabularies.

The tags cover general video attributes and physics-specific attributes. General attributes describe content categories and photographic properties, while physics-specific attributes describe physical phenomena, materials, and state changes. Section 3.3 presents the annotation procedure; this section defines the tags and their use in dataset organization.

## 3.4.1 General Video Tags

General video tags comprise one content-category field and seven photographic fields (Table 1). The content category identifies the setting or activity, such as a laboratory demonstration, cooking, or industrial operation. Photographic fields record color, shot scale, viewing angle, lens type, composition, lighting characteristics, and light-source type.

Table 1. General video attributes, vocabulary sizes, and representative values.
<table><tr><td>Attribute</td><td>Vocabulary Size</td><td>Representative Values</td></tr><tr><td>Content category</td><td>14</td><td>Laboratory demonstration, educational explanation, daily life, industrial machin- ery, natural phenomena</td></tr><tr><td>Color</td><td>14</td><td>Warm, cool, saturated, neutral, high-contrast, monochrome</td></tr><tr><td>Shot scale</td><td>7</td><td>Extreme close-up, close-up, medium shot, full shot, long shot</td></tr><tr><td>Viewing angle</td><td>6</td><td>Eye-level, low-angle, high-angle, overhead, Dutch angle</td></tr><tr><td>Lens type</td><td>5</td><td>Ultra-wide, wide-angle, standard, telephoto, macro</td></tr><tr><td>Composition</td><td>6</td><td>Rule of thirds, centered, symmetrical, diagonal, leading lines</td></tr><tr><td>Lighting characteristics</td><td>10</td><td>Natural sunlight, soft diffuse light, harsh direct light, low-key lighting, backlighting</td></tr><tr><td>Light-source type</td><td>10</td><td>Natural, incandescent, fluorescent, LED point source, studio strobe, firelight</td></tr></table>

These tags describe the video’s content and presentation, but do not identify its physical processes. For example, the category industrial-machinery does not distinguish rotation from cutting or deformation. Physics-specific tags provide these distinctions.

## 3.4.2 Physics-Specific Attributes

Physical phenomenon taxonomy. The taxonomy has two levels: top-level categories and leaf phenomena. Five top-level categories—Mechanics, Fluid Dynamics, Thermal Dynamics, Wave & Optics, and Material Deformation—contain 26 predefined leaf phenomena. An additional category, Other, covers physical interactions outside this vocabulary and has no predefined leaves. Figure 10 shows the category-to-leaf mapping.

![](images/609c5b26391dd43bf80ce079c1aa01bbafd93f739459a2377b61f54628e5127f.jpg)  
Figure 10. Two-level taxonomy of physical phenomena. Five top-level categories contain 26 predefined leaf phenomena. An additional Other category covers physical interactions outside the vocabulary and has no predefined leaves.

![](images/cdaf442219ebc7a9fcb3f9e6888979cc9c7b989038faa2678aa3428d4fa376ac.jpg)  
Figure 11. Multi-label physical annotations for car washing, boiling water, and lathe cutting. Each example contains six frames and the associated phenomenon tags, grouped by top-level category. A filled star marks the primary phenomenon; outlined tags indicate additional phenomena.

Multi-label annotation. A clip may receive multiple labels at both taxonomy levels because it can contain several processes, either simultaneously or at different stages. A primary phenomenon is designated when a single category assignment is needed for statistics or sampling; the remaining labels are retained. The tags record which phenomena occur, while the timestamped caption describes their temporal progression and the participating objects.

Figure 11 shows annotations for car washing, boiling water, and lathe cutting. In the lathe-cutting example, workpiece rotation and the changes produced by cutting occur within the same clip. A content label identifies the activity as an industrial operation, whereas the phenomenon labels distinguish the processes within it. The primary label provides a single attribution without replacing this multi-label description.

Additional physical attributes. Material categories include rigid solids, deformable solids, liquids, and granular materials. State-change labels describe changes such as deformation, phase transition, and fragmentation. These attributes complement phenomenon labels by describing the participating materials and the resulting changes. Interaction viewpoint distinguishes clips without human participation from first- and third-person interaction views.

Physical anomaly flags record visible behaviors such as object interpenetration, unrealistic collisions, and gravitydefying motion. They differ from the physical relevance indicator returned by the first annotation call: a clip may contain a physical interaction while depicting that interaction implausibly. The judgment concerns visible behavior rather than whether the video is rendered or generated.

We exclude absolute mass, force magnitude, real-world speed, and temperature from the annotation. For videos without the necessary scale, calibration, or additional measurements, the annotator cannot reliably determine these quantities. Requesting them would encourage unsupported estimates and introduce annotation noise. The tags instead describe physical phenomena, material categories, and visible state changes. Image-based motion measurements are retained separately and are not interpreted as estimates of physical energy.

The content and phenomenon tags allow dataset composition to be examined by both scene type and physical process. For subsequent selection, they remain associated with the quality and motion measurements retained from video curation. Section 3.5 summarizes the annotated candidates and their use in selecting data to match the distribution of a reference pretraining corpus.

![](images/0aa4d5bd96a6f80b203775c610185fa4d35dd9aa7047d42068c8364a268e4c00.jpg)  
Figure 12. Curation from source videos to annotated clips. Spatiotemporal Video Splitting produces candidate clips; Technical Quality Filtering and Content Purity Filtering select clips for Physical Video Captioning. Physical Verification selects physical clips, which receive structured captions and closed-set tags.

## 3.5 Curation Outcome

Figure 12 summarizes clip filtering at three checkpoints: Technical Quality Filtering, Content Purity Filtering, and Physical Verification. Technical Quality Filtering retains clips with usable visual and temporal quality. Content Purity Filtering removes presenter-dominated footage and overlays while preserving text that belongs to the scene. Physical Verification then selects clips containing physical processes for captioning and tagging.

The annotation-stage output pairs each retained clip with a structured caption and a tagging record. Captions describe the scene, objects, and temporal progression of interactions; tagging records organize their categorical attributes, curation measurements, and derived information.

These annotated candidates are used in a final dataset-selection step for distribution matching against a reference pretraining corpus. Tags, sampling weights, and deduplication results support this selection, linking the clip-level records to the composition of the resulting training data.

## 3.6 Action-Conditioned Data Curation

To support action-conditioned training across heterogeneous robot datasets, we organize data preparation into Data Ingestion, Canonical Production, and Output Validation (Figure 13).

Data Ingestion applies dataset-specific admission checks to dataset identity, release metadata, input integrity, source schema, and semantic coverage. Dedicated ingest adapters discover episodes, map source fields, align temporal streams, interpret actions and observations, and preserve provenance. The resulting EpisodeIR provides a common representation of each episode and a stable semantic boundary between source formats and downstream training representations.

Canonical Production converts EpisodeIR into versioned training datasets. Each Canonical version specifies its field layout, action semantics, masks, temporal sampling and terminal-frame policies, and normalization strategy. Semantic checks verify gripper values, rotations, and action targets, together with their alignment to observations.

Output Validation checks numerical validity, field and mask consistency, normalization statistics, and accounting of retained and excluded episodes. Each release includes a manifest, lineage records, and quality-check results. Independent readback verifies persisted data and compatibility with the training loader. Exclusions and processing failures are recorded against their source episodes, while accepted data provide aligned robot observations and action trajectories for training.

![](images/f4d73e0fa54e3c299bd43ce0e6b361e1fc1ddf046f77f8f802ca2d190e119043.jpg)  
Figure 13. Action-conditioned data curation. Dataset-specific gates and adapters produce a unified EpisodeIR. Canonical versions define the training representations, with semantic checks during production and quality checks and independent readback before delivery of the corresponding training datasets.

## 4 Experiments

We evaluate the model in two complementary settings. Text-image-to-video (TI2V) evaluation measures the physical plausibility of generated dynamics under image and language conditioning, whereas image-action-to-video (IA2V) evaluation tests whether robot rollouts follow action trajectories while preserving visual fidelity.

## 4.1 Text-Image-to-Video Evaluation

## 4.1.1 Implementation Details

Starting from the official LingBot-Video checkpoint [6], we trained the StrucPhysVideo-TI2V 30B backbone on 81- frame clips at 15 fps and 480 × 832 resolution. Training used four eight-H200 nodes with local and global batch sizes of 3 and 96. We combined eight-way expert parallelism, FSDP2 sharding of non-expert parameters, DeepEP token dispatch, non-reentrant activation checkpointing, and packed variable-length attention.

The corpus comprised 10,000 hours each of physics-focused videos and high-quality general-domain videos collected from the Internet. A dynamic schedule sampled the general-domain and physics-focused pools at 70%/30% during the first 60% of training, 40%/60% during the next 30%, and 20%/80% during the final 10%. This progression preserved general-purpose generation while increasing the emphasis on physical dynamics.

Training retained FP32 master parameters and optimizer states, used BF16 forward computation, FP32 gradient reduction and sensitive operations, and TF32 matrix multiplication where supported. AdamW used a learning rate of $1 \times 1 0 ^ { - 5 }$ , weight decay of $1 \times 1 0 ^ { - 2 } , \beta = ( 0 . 9 , 0 . 9 9 9 )$ , and a gradient-norm limit of 1.0. The cosine schedule used 100 warm-up steps and a $1 \times 1 0 ^ { - 6 }$ minimum learning rate. Flow matching used 1,000 timesteps, a shift of 5, and logit-normal timestep sampling. The Qwen3-VL condition encoder and Wan VAE remained frozen in evaluation mode.

## 4.1.2 Performance Comparison

Physics-IQ Verified evaluates the reproduction of real physical dynamics. Physics-IQ contains 66 experiments across solid and fluid dynamics, thermodynamics, optics, and magnetism; three viewpoints and two repetitions yield 396 videos [25]. For 198 instances, models receive the 3-second frame and description and predict the next 5 seconds, while paired repetitions quantify natural variation. The Verified protocol corrects ambiguous prompts and spurious reference motion and averages scores per sample [1]. Its score equally combines spatial IoU, spatiotemporal IoU, weighted spatial IoU, and inverse normalized pixel MSE after normalization by paired-video variation. Higher values indicate closer agreement in event location, timing, magnitude, and appearance [1].

Case 1: Start -> End  
![](images/f440388ac249de6fc5bd822c82dbd44a4de0e3c20f96c137259568785b5d4ec6.jpg)  
Figure 14. Qualitative comparison among ground truth, Wan2.2-5B[4], LingBot-Video-30B[6], and StrucPhysVideo (Ours), with five frames shown from the start to the end of each sequence. In Case 1, the red boxes highlight substantial deformation of the track geometry, while the blue boxes indicate that the ball fails to roll along the shape of the track. In contrast, our model generates physically plausible ball-rolling motion that faithfully follows the track geometry. In Case 2, the blue boxes highlight the loss of the subject’s key facial features, whereas our model effectively preserves the subject’s identity and distinctive appearance.

Under the verified image-to-video protocol, StrucPhysVideo achieved 45.5%, the highest score among the 17 models in the comparison based on the 16 September 2026 benchmark snapshot (Figure 1). We reproduced the LingBot-Video score, while all other comparison scores were taken directly from the benchmark snapshot. StrucPhysVideo was 2.8 percentage points above Cosmos3-Super Image2Video at 42.7% [26] and 9.3 points above the strongest closed-source comparator, MiniMax H3 Max at 36.2% [2, 27]. This result is specific to the evaluated protocol and leaderboard snapshot. Figure 14 compares StrucPhysVideo directly with Wan2.2-5B and LingBot-Video-30B. Additional examples span general-domain scenes (Figure 15), physical phenomena and interactions (Figure 16), and embodied manipulation (Figure 17); these selected outputs illustrate breadth but do not replace quantitative evaluation.

![](images/27ad7ce8eeb0bbd3b02f506c6e1282ee7373ac534f5fb7fa9e98442d31b6b3fc.jpg)  
Figure 15. Qualitative StrucPhysVideo-TI2V results for general-domain video generation. Each row shows five temporally ordered frames from one generated video, spanning people, vehicles, landscapes, and camera motion.

![](images/c973830b7611f1c827c84b3b9e8ca5cbf6b862ffae3041940a56fbda8a96e84a.jpg)  
Figure 16. Qualitative StrucPhysVideo-TI2V results for physical-world video generation. Each row shows five temporally ordered frames from one generated video depicting a physical phenomenon or object interaction.

![](images/c3e81a72605c69b47989a64f6ee41be5284d36c5d3cb5c6053ef3e7c70ec6f44.jpg)  
Figure 17. Qualitative StrucPhysVideo-TI2V results for embodied scenarios. Each row shows five temporally ordered frames from a generated robot manipulation sequence.

![](images/f538b14779d688dd2b54791064b68aa373c15ba9c679dde802915683ff3c9eea.jpg)  
Figure 18. Caption ablation on Physics-IQ Verified. Wan2.2-5B [4] and LingBot-Video-30B [6] are evaluated under three settings: the reproduced baseline, incorporation of raw captions, and incorporation of physics-focused captions. Physics-focused captions yield the highest Verified Score for both backbones. Only Physics-IQ Verified scores are reported. <sup>∗</sup> denote scores reproduced by us.

## 4.1.3 Ablation Studies

To isolate the effect of caption supervision, we conducted an ablation on Wan2.2-5B [4] and LingBot-Video-30B [6] using data prepared with our pipeline (Figure 18). For the Case 1 video in Figure 14, Appendix B.1 (Listing 1) provides the real structured caption, and Appendix B.2 (Listing 2) provides the corresponding real raw caption. Adding raw captions increased the Physics-IQ Verified Score from 24.8% to 31.0% for Wan2.2-5B and from 37.9% to 43.9% for LingBot-Video-30B. Physics-focused captions achieved the highest score for both backbones, reaching 33.2% and 45.5%, respectively. These results correspond to gains of 2.2 and 1.6 percentage points over raw captions and show that physics-focused captions outperform raw captions on the full dataset for both backbones.

## 4.2 Image-Action-to-Video Evaluation

## 4.2.1 Implementation Details

We train our IA2V model on AgiBot World-Beta [28] together with over 500 hours of general-domain T2V/TI2V data to preserve its original TI2V and general-domain generation capabilities. We evaluate the model on a fixed subset of 5K video clips from the AgiBotWorld-Alpha corpus [28]. All evaluations use the final causal few-step student after the complete distillation pipeline. Unless otherwise specified, the student uses four denoising steps per chunk, with no explicit CFG at inference. At each control step, the input command is represented as the EE20 end-effector vector introduced in Section 2.2.2.

## 4.2.2 Performance Comparison

For comparison, we evaluate DreamDojo-AgiBot-14B [29] and A2World [30], both of whose training corpora cover AgiBot robot platforms. Each model retains its native action representation. Specifically, DreamDojo uses a 22- dimensional action vector covering the arm joints, grippers, head, waist, and base velocity, whereas A2World adopts a 14-dimensional representation consisting of end-effector translation and rotation increments in the local coordinate frame. In contrast, our model relies solely on the EE20 end-effector target pose representation.

We evaluate standard fidelity metrics, including PSNR, SSIM, and LPIPS. In addition, Motion MAE measures the mean absolute error between the predicted and ground-truth adjacent-frame RGB differences

Beyond overall video fidelity, we further evaluate whether the generated videos faithfully reflect the commanded robot actions in terms of trajectory consistency and the physical plausibility of robot-object interactions. Specifically, we report WorldArena-style metrics [31], including: (1) Trajectory Accuracy (Traj. Acc.), which measures the agreement between detected image-space gripper trajectories using dynamic time warping; (2) Depth Accuracy (Depth Acc.), which measures the consistency between depth maps estimated from the generated and ground-truth videos; and (3)

Interaction Quality, a VLM-assigned score ranging from 1 to 5 that evaluates the physical plausibility of robot-object interactions, with higher scores indicating more plausible interactions.

As shown in Table 2, StrucPhysVideo-IA2V consistently outperforms DreamDojo-AgiBot across all evaluated metrics. Beyond higher frame-level fidelity (PSNR, SSIM, and LPIPS) and lower motion error, our model achieves substantially higher trajectory accuracy (0.8330 vs. 0.7905), while also improving depth consistency and interaction quality. These results indicate that StrucPhysVideo-IA2V not only produces more visually faithful videos, but also more accurately reflects the commanded robot actions and the resulting physical interactions.

<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>Motion MAE↓</td><td>Traj. Acc. ↑</td><td>Depth Acc. ↑</td><td>Interaction ↑</td></tr><tr><td>DreamDojo-AgiBot</td><td>20.86</td><td>0.8277</td><td>0.1825</td><td>0.03509</td><td>0.7905</td><td>0.9736</td><td>3.366</td></tr><tr><td>StrucPhysVideo-IA2V</td><td>22.12</td><td>0.8365</td><td>0.1602</td><td>0.03368</td><td>0.8330</td><td>0.9854</td><td>3.470</td></tr></table>

Table 2. Quantitative performance on 5K video clips. Metrics exclude the reference frame. PSNR is in dB. Traj. Acc. and Depth Acc. are normalized to [0, 1], with higher values indicating better agreement.

To match A2World’s default 21-frame, $2 5 6 \times 2 5 6$ -per-view generation setting, we recompute all metrics on the same head-camera crop, resized to 256 × 256, over the first 21 frames at 5 fps; the results are shown in Table 3. The corresponding qualitative rollouts are shown in Figure 19.

<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>Motion MAE↓</td><td>Traj. Acc. ↑</td><td>Depth Acc. ↑</td><td>Interaction ↑</td></tr><tr><td>A2World</td><td>17.71</td><td>0.7110</td><td>0.2105</td><td>0.04316</td><td></td><td></td><td></td></tr><tr><td>DreamDojo-AgiBot</td><td>22.14</td><td>0.8415</td><td>0.1687</td><td>0.03338</td><td>0.7853</td><td>0.9685</td><td>3.365</td></tr><tr><td>StrucPhysVideo-IA2V</td><td>22.89</td><td>0.8571</td><td>0.1550</td><td>0.03172</td><td>0.8287</td><td>0.9827</td><td>3.470</td></tr></table>

Table 3. Quantitative comparison on 256 × 256 head-view grid and frames 1–20 after the reference frame. A2World additionally receives initial wrist views and observed-pose-derived controls. Qualitative comparison is provided in Figure 19.

## 4.2.3 Ablation on Teacher Dropout and Guidance

We investigate how unconditional dropout during bidirectional action-injection training and teacher-side action CFG during distillation affect the final distilled student. We compare three teacher-training settings: (1) no dropout for either action or text conditions; (2) dropout applied only to the action condition; and (3) independent dropout applied to both action and text conditions. The CFG settings in Table 4 refer to the teacher-side, rather than to student inference. All reported metrics are measured using four-step, single-branch student rollouts without explicit CFG.

As shown in Table 4, we make three interesting observations: (1) even without CFG, applying dropout only to the action condition does not degrade performance; (2) action CFG substantially improves both action conditioning and IA2V generation quality. In our setting, an action CFG scale of 2 performs best, while further increasing the scale becomes detrimental; and (3) even in IA2V mode, where text CFG is not used at inference time, introducing a small amount of text dropout during training still benefits IA2V performance. Note that text dropout is applied only to the general-domain T2V/TI2V data mixed into training, while the IA2V data use no text prompts.

![](images/280b4422951652e8bbfee47d638fc5031dcf360882ebf3364a1093a543ae3109.jpg)  
Figure 19. AgiBot head-view rollouts at 0, 1, 2, 3, and 4 seconds. Each group shows GT, A2World, DreamDojo-AgiBot, and ours. All start from the same head image. All three models produce arm motion, but the generated trajectories and local robot geometry differ from ground truth. A2World also exhibits visible appearance distortion in these examples.

<table><tr><td>Action dropout</td><td>Text dropout</td><td>Teacher CFG setting</td><td>Motion MAE↓ Traj. Acc. ↑</td></tr><tr><td>0%</td><td>0%</td><td>w/o CFG 0.03537</td><td>0.7913</td></tr><tr><td>15%</td><td>0%</td><td>w/o CFG</td><td>0.03455 0.7896</td></tr><tr><td>15%</td><td>0%</td><td>Action CFG  $s _ { a } = 2$ </td><td>0.03449 0.8270</td></tr><tr><td>15%</td><td>0%</td><td>Action CFG  $s _ { a } = 3$  0.03517</td><td>0.8284</td></tr><tr><td>15%</td><td>0%</td><td>Action CFG  $s _ { a } = 5$  0.03583</td><td>0.8195</td></tr><tr><td>15%</td><td>5%</td><td>Action CFG  $s _ { a } = 2$ </td><td>0.03432 0.8272</td></tr></table>

Table 4. Effects of teacher dropout and guidance on final distilled students. Dropout rates refer to bidirectional action-injection training. The reported action CFG scale is used both by the AR teacher to generate causal ODE distillation trajectories and by the bidirectional teacher to compute the real score during DMD. Bold and underlined values indicate the best and second-best values in each metric column, respectively.

## 5 Conclusion

We presented StrucPhysVideo, a video world model developed through physically grounded data curation, annotation, and post-training. Our pipeline combines video sourcing and filtering with structured captions and physical phenomenon tags to preserve and describe how objects move, interact, and change state. By explicitly annotating participating objects, material properties, temporally localized interactions, and object motion distinct from camera movement, these data provide physically grounded supervision for adapting video foundation models toward physical world modeling. We further extend this framework to action-conditioned video prediction using paired action–video data, connecting robot commands with the resulting evolution of visual scenes.

StrucPhysVideo achieves 45.5% on Physics-IQ Verified, while caption ablations across two video backbones consistently demonstrate the benefit of physics-focused language supervision over raw captions. Our action-conditioned experiments further provide an initial step toward interactive video world models that respond to embodied actions.

Looking forward, an important direction is to bridge world prediction and downstream embodied policies, where generated visual dynamics can support interaction, planning, and control in the physical world. This requires extending the current model to longer-horizon streaming rollouts while maintaining physical consistency over time, as well as further reducing inference latency—potentially to one- or two-step generation—to enable more responsive interaction. Beyond generation efficiency, supporting richer action spaces, and enabling cross-embodiment generalization across diverse robot platforms are also important steps toward more general embodied world models. Together, we hope StrucPhysVideo provides a foundation for advancing video generation from modeling observable physical dynamics toward interacting with the physical world.

## 6 Contributors

All core contributors listed below contributed equally to this work and are listed alphabetically by first name.

Data Infra & Model Training: Enhui Ma, Kaiwen Guo, Tingrui Zhang, Wei Song, Yingshui Tan.

Project Leads: Jianhua Xu, Tong Zhang.

Project Sponsors: Jianhua Xu, Kaicheng Yu.

## References

[1] Tim Rädsch, Yuki M. Asano, Hilde Kuehne, Stefan Bauer, Priyank Jaini, Robert Geirhos, and Carsten T. Lüth. Physics-iq verified. arXiv preprint arXiv:2606.18943, 2026.

[2] Physics-IQ Benchmark Contributors. Physics-iq and physics-iq verified: Benchmarking physical understanding in generative video models. https://github.com/google-deepmind/physics-iq-benchmark, 2026. Official benchmark repository and leaderboard, accessed 10 September 2026.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[5] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

[6] Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang, Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Ka Leong Cheng. Scaling mixture-of-experts video pretraining for embodied intelligence. arXiv preprint arXiv:2607.07675, 2026.

[7] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[8] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[9] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. In Advances in Neural Information Processing Systems, volume 36, 2023.

[10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[11] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[12] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

[13] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026.

[14] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2024.

[15] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[16] PySceneDetect Contributors. PySceneDetect: Video scene detection and analysis. https://www.scenedetect. com/. Official project website, accessed 16 September 2026.

[17] Tomáš Soucek and Jakub Lokoˇ c. TransNet V2: An effective deep network architecture for fast shot transitionˇ detection. arXiv preprint arXiv:2008.04838, 2020.

[18] Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Image Analysis: 13th Scandinavian Conference, SCIA 2003, volume 2749 of Lecture Notes in Computer Science, pages 363–370. Springer, 2003. doi: 10.1007/3-540-45103-X\_50.

[19] Bruce D. Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In Proceedings ofthe Seventh International Joint Conference on Artificial Intelligence, pages 674–679, 1981.

[20] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24(6):381–395, 1981. doi: 10.1145/358669.358692.

[21] Wei Wu, Hanyang Peng, and Shiqi Yu. YuNet: A tiny millisecond-level face detector. Machine Intelligence Research, 20:656–665, 2023. doi: 10.1007/s11633-023-1423-y.

[22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[23] Jaided AI. EasyOCR. https://github.com/JaidedAI/EasyOCR. Official software repository, accessed 16 September 2026.

[24] Qwen Team. Qwen3.6-27B. https://huggingface.co/Qwen/Qwen3.6-27B, 2026. Official model card, accessed 16 September 2026.

[25] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 948–958, 2026.

[26] Aditi, Niket Agarwal, Arslan Ali, Jon Allen, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.

[27] MiniMax. Minimax h3. https://huggingface.co/MiniMaxAI/MiniMax-H3, 2026. Official model card, accessed 10 September 2026.

[28] AgiBot World Contributors. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[29] Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

[30] Ze Huang, Jiahui Zhang, Hairuo Liu, Chenxi Zhang, Ran Cheng, and Li Zhang. Learning transferable dynamics priors from action to world modeling. arXiv preprint arXiv:2606.29501, 2026.

[31] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.

## A StrucPhysVideo-TI2V Training and Inference Algorithms

Algorithm 1 summarizes one stochastic training step for StrucPhysVideo-TI2V. The frozen encoders first construct the multimodal condition and normalized clean video latent. The procedure then samples a flow time and noise realization, forms the corrupted latent, and restores its first temporal position from the clean reference latent. The Transformer predicts the flow-matching velocity, while masking excludes the observed reference position from the denoising objective. Consequently, each update trains only the future latent trajectory while retaining the first frame as context.

```powershell
Algorithm 1 StrucPhysVideo-TI2V Single-Step Training
Require: Video $\mathbf { V } = \{ I _ { 0 } , I _ { 1 } , \ldots , I _ { T } \}$ , caption $c ,$ frozen multimodal encoder $\mathcal { Q } ,$ , frozen VAE encoder $\mathcal { E }$
1: $\mathbf { h } _ { c } \gets \mathcal { Q } ( c , I _ { 0 } )$
2: ${ \bf z } _ { 0 } \gets \mathcal { N } ( \mathcal { E } ( { \bf V } ) )$
3: Extract the reference latent ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$ from the first temporal position of $\mathbf { z } _ { 0 }$
4: Sample $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ and a logit-normal flow time $\tau$
5: ${ \bf z } _ { \tau }  ( 1 - \tau ) { \bf z } _ { 0 } + \tau \epsilon$
6: Replace the first temporal position of $\mathbf { z } _ { \tau }$ with ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$
7: v<sub>θ</sub> ← StrucPhysVideo $\left( \mathbf { z } _ { \tau } , \tau , \mathbf { h } _ { c } \right)$
8: $\mathbf { v } ^ { \star }  \epsilon - \mathbf { z } _ { 0 }$
9: Mask the reference position in $\mathbf { v } _ { \theta }$ and $\mathbf { v } ^ { \star }$
10: Compute the timestep-weighted mean-squared error and update the Transformer
```

Algorithm 2 describes the corresponding conditional sampling process. It initializes the future trajectory from Gaussian noise but fixes the first temporal position to the encoded reference image. At every solver step, conditional and negative-condition predictions are combined through classifier-free guidance before the Flow-UniPC update. Reclamping the clean reference latent after each update prevents the observed frame from drifting as the future frames are generated. The converged latent is finally inverse-normalized and decoded into the output video.

Algorithm 2 StrucPhysVideo-TI2V Inference   
Require: Caption $c ,$ reference image $I _ { 0 } ,$ , sampling steps K, text-guidance scale $s _ { t }$   
1: $\mathbf { h } _ { c } \gets \mathcal { Q } ( c , I _ { 0 } )$ and $\mathbf { h } _ { u }  \mathcal { Q } ( c _ { \mathrm { n e g } } , I _ { 0 } )$   
2: ${ \mathbf z } _ { 0 } ^ { \mathrm { r e f } }  \mathcal { N } ( \mathcal { E } ( \bar { I } _ { 0 } ) )$   
3: Sample an initial video latent ${ \bf z } _ { K } \sim \mathcal { N } ( 0 , { \bf I } )$   
4: Replace the first temporal position of $\mathbf { z } _ { K }$ with ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$   
5: for $k = K , K - 1 , \ldots , 1$ do   
6: Predict $\mathbf { v } _ { \mathrm { c o n d } }$ from $\left( \mathbf { z } _ { k } , \mathbf { h } _ { c } \right)$ and $\mathbf { v } _ { \mathrm { u n c o n d } }$ from $\left( \mathbf { z } _ { k } , \mathbf { h } _ { u } \right)$   
7: $\widehat { \mathbf { v } }  \mathbf { v } _ { \mathrm { u n c o n d } } + s _ { t } ( \mathbf { v } _ { \mathrm { c o n d } } - \mathbf { v } _ { \mathrm { u n c o n d } } )$   
8: $\mathbf { z } _ { k - 1 } \gets \mathrm { F l o w U n i P C S t e p } ( \mathbf { z } _ { k } , \widehat { \mathbf { v } } , \tau _ { k } )$   
9: Replace the first temporal position of $\mathbf { z } _ { k - 1 }$ with ${ \bf z } _ { 0 } ^ { \mathrm { r e f } }$   
10: end for   
11: $\widehat { \mathbf { V } } \gets \mathcal { D } ( \mathcal { N } ^ { - 1 } ( \mathbf { z } _ { 0 } ) )$ )   
12: return $\widehat { \mathbf { V } }$

## B Caption Example

The two caption examples in this appendix describe the video shown as Case 1 in Figure 14. They provide the real structured caption and the corresponding real raw caption for the same blue-ball and wooden-track sequence.

## B.1 Real Structured Caption Example

Listing 1 presents a real structured caption for a five-second clip in which a blue ball moves through a wooden marble run. Its four top-level keys correspond to the information groups defined in Section 3.3. global\_scene\_dynamics describes the setting and progression of the physical event. camera\_motion\_and\_illumination separates camera state, composition, and lighting from object motion. world\_knowledge\_and\_material\_properties records physical priors governing gravity-driven descent, track-constrained rolling, tangent-direction motion after leaving the track, and the spatial alignment required to enter the lower track. entity\_level\_temporal\_behaviors records the ball, wooden run, and human hand together with their visual attributes, spatial relationships, and timestamped behaviors.

For readability, Listing 1 displays the caption as a formatted, multiline JSON object. The actual input to the StrucPhysVideo text encoder is a compact, single-line JSON string obtained by serializing the caption object, without inter-field spaces or indentation.

```ipynb
{
"global_scene_dynamics": "The video features a wooden marble run set up on a white table. The run consists of a long, wavy track on the
left and a straight track on the right, both made of light-colored wood. A small blue ball is released at the start of the wavy track. As it
rolls down, it follows the curves of the track, eventually reaching the end of the straight section. When the ball leaves the end of this track
, it must continue straight along the track’s outgoing tangent direction, with no sideways deflection or skew, then fall naturally under
gravity into a second wooden track positioned below the tabletop. The lighting is bright and even, creating a clean and focused
atmosphere on the mechanical action of the ball.",
"camera_motion_and_illumination": {
"camera_motion": "The camera is stationary throughout the entire video, providing a fixed, high-angle perspective of the marble run and
the table.",
"dominant_color": "Red",
"shot_scale": "Wide",
"viewing_angle": "High angle",
"lens_scale": "Medium",
"frame_composition": "Left heavy",
"illumination_quality": "Soft light",
"illumination_source": "Artificial light"
},
"world_knowledge_and_material_properties": [
"Gravity drives the ball along descending sections of the wooden track.",
"While constrained by the track, the rolling ball follows the track’s curved path.",
"When the ball leaves the end of the track, inertia carries it forward along the outgoing tangent direction rather than causing a sideways
deflection.",
"Once unsupported, the ball follows a gravity-driven downward trajectory, and entering the lower track requires the two track sections to be
spatially aligned."
],
"entity_level_temporal_behaviors": [
{
"entity_name": "blue ball",
"visual_description": "A small, solid blue spherical object used as a marble in the run.",
"timestamped_behaviors": [
{
"time_interval": "[0.0s - 0.2s]",
"behavior": "is dropped into the starting hole of the wooden track"
},<sub>{</sub>
"time_interval": "[0.2s - 2.5s]",
"behavior": "rolls down the wavy wooden track"
},
{
"time_interval": "[2.5s - 4.0s]",
"behavior": "rolls along the straight wooden track"
},
{
"time_interval": "[4.0s - 5.0s]",
"behavior": "leaves the end of the straight track in its outgoing tangent direction without sideways deflection, then falls under gravity
into the lower wooden track below the tabletop"
}
],
"spatial_location": "starts at the top left, moves through the center, and exits at the bottom right",
"relative_scale": "small",
"form_and_color": "spherical and blue",
```

```jsonl
"scene_relationship": "rolls along the wooden tracks and is released by a hand",
"spatial_orientation": "rolling",
"body_pose": "",
"facial_expression": "",
"apparel": "",
"perceived_gender": "",
"visible_skin_attributes": ""
},
{
"entity_name": "wooden marble run",
"visual_description": "A set of light-colored wooden tracks designed for a marble to roll through.",
"timestamped_behaviors": [
{
"time_interval": "[0.0s - 5.0s]",
"behavior": ""
}
],
"spatial_location": "occupies the left and center portions of the frame",
"relative_scale": "large",
"form_and_color": "long, wavy and straight tracks in light brown wood",
"surface_properties": "smooth wood grain",
"appearance_attributes": "features a wavy section on the left and a straight section on the right",
"scene_relationship": "serves as the path for the blue ball",
"spatial_orientation": "horizontal and slightly diagonal",
"body_pose": "",
"facial_expression": "",
"apparel": "",
"perceived_gender": "",
"visible_skin_attributes": ""
},
{
"entity_name": "human hand",
"visual_description": "A person’s hand visible at the start of the video.",
"timestamped_behaviors": [
{
"time_interval": "[0.0s - 0.2s]",
"behavior": "drops the blue ball into the track"
}
],
"spatial_location": "top left corner of the frame",
"relative_scale": "medium",
"form_and_color": "natural skin tone",
"surface_properties": "smooth",
"appearance_attributes": "only the fingers and part of the palm are visible",
"scene_relationship": "releases the blue ball into the track",
"spatial_orientation": "reaching down",
"body_pose": "fingers pinching the ball",
"facial_expression": "",
"apparel": "",
"perceived_gender": "male",
"visible_skin_attributes": "light skin tone"
}
```  
Listing 1. Formatted structured-caption JSON example.

## B.2 Real Raw Caption Example

Listing 2 provides the corresponding real raw caption before conversion into the structured representation above. It describes the same blue-ball and wooden-track scene as continuous prose, including the intended motion, physical constraints, stationary scene elements, camera requirements, and visual style

![](images/b33170365ade14610b4f29fdd26a36e9ee945ec1d1ab6478cbbe593c3fcdba2d.jpg)  
Listing 2. Real raw-caption example corresponding to Listing 1.