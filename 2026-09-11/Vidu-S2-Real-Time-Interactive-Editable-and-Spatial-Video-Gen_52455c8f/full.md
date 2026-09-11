# Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

Jintao Zhang<sup>∗†</sup>, Kai Jiang<sup>∗</sup>, Jintao Chen<sup>∗</sup>, Xu Wang<sup>∗</sup>, Deyuan Liu<sup>∗</sup>, Jungang Li<sup>∗</sup>, Dechuang Chen<sup>∗</sup>, Ming Lin<sup>∗</sup>, Jingjiang Zhou, Haopeng Jin, Qi Jia, Xiaohang Wang, Yaole Wang, Zhanqiang Zhang, Ran Li, Zhengkun Huang, Shuyue Xiong, Yuji Wang, Zikun Dai, Hui He, Yang Luo, Mang Ning, Weiqi Feng, Chengyang Ye, Xinyue Lin, Min Zhao, Hongzhou Zhu, Hengkai Tan, Zeyuan Wang, Chendong Xiang, Kaiwen Zheng, Zhijie Deng<sup>‡</sup>, Fan Bao<sup>‡</sup>, Jianfei Chen<sup>‡</sup>, Jun Zhu<sup>‡</sup>

Tsinghua University, Shengshu Technology https://vidu.com/vidu-stream

## Abstract

We present Vidu S2, which comprises Vidu S2-Avatar, a real-time interactive digital-character model, and Vidu S2-Editing, a real-time video editing model. Moreover, we explore the feasibility of real-time spatial video generation for both Vidu S2-Avatar and Vidu S2-Editing. Compared with Vidu S1, Vidu S2-Avatar supports real-time 720p video generation, generation with dynamic references that can be updated at any moment, and stronger instruction following, such as dancing. Vidu S2-Editing supports editing a video stream in real time, including style transfer, virtual try-on, character replacement, and background replacement. Experiments show that Vidu S2 outperforms all baselines. A playable online demo is available at https://vidu.com/vidu-stream.

![](images/ebf7946556af7333629827fdc213d8a4b1319202d302e04079c96f6cf76ffcdd.jpg)  
Figure 1 Overview of Vidu S2.

## 1 Introduction

Demand for Real-time Interactive Video Generation Recent video generation models, such as Sora, Veo, Wan, and Seedance [1–4], have shown strong ability in generating high-quality videos. However, most of them still follow an offline, one-shot generation paradigm: a user enters a prompt, waits for minutes or even tens of minutes, and receives the complete video only after generation finishes. During this process, the user can only passively wait to receive information and cannot actively initiate any interaction. This limitation comes from the offline diffusion paradigm, where the model denoises the entire video synchronously over many steps and only produces an entire clean video at the end. Such a paradigm works well for offline content creation, but human visual entertainment is not limited to pre-generated videos. People also enjoy face-to-face communication, live streaming, games, talking with someone, and other interactive visual experiences, where content must respond immediately to the user. From a demand perspective, suppose that each user has an average demand of $\alpha \in [ 0 , 1 ]$ for real-time interactive visual content, e.g., $\alpha = 0 . 5$ . Then the total demand scales with $\alpha \times N$ , where N is the number of users. In contrast, suppose that each user has an average demand of $\beta \in [ 0 , 1 ]$ for offline-generated visual content, $\mathrm { e . g . , } \beta = 0 . 5$ . Since offline-generated videos can be replayed and shared, their generation demand scales more like $\beta \times N / m$ , where $m$ is the average number of views per generated video. If we assume $\alpha \approx \beta$ and $m > 1 0 0$ , then the demand for real-time interactive video generation is much greater than that for offline-generated videos.

Real-time interactive video generation is one of the most important future directions for industry.

Vidu S1. Following this direction, we released Vidu S1 [5], a real-time interactive video generation model for continuous user interaction. Users can control video generation content at any moment through voice instructions, instead of fixing all controls before generation starts. Vidu S1 supports infinite-length real-time video generation without blurring, drift, or visual distortion. Built with TurboDiffusion [6] and TurboServe [7], Vidu S1 outputs 540p real-time videos at up to 42 FPS on regular consumer GPUs. Users can upload custom images of real people, anime, and pets, and choose different voice tones for personalized experiences. Its scope, however, is largely confined to talking-head-centric digital characters: the resolution is limited to 540p, the reference that defines the character is fixed once a stream has started, large body motion such as dancing is difficult to follow, and editing an incoming video stream is not supported.

Vidu S2-Avatar. Vidu S2-Avatar is a real-time interactive digital-character model that improves on Vidu S1 in four directions. (1) Self-Replay Forcing (SRF). A stream is generated segment by segment, so errors pass from one segment to the next and eventually cause drift or collapse. Self-Forcing [8] gets the setup right by conditioning each segment on chunks that the model generated itself, which matches what the model sees at inference time. Two problems remain: the self-generated history is fed in clean rather than noised, and it is detached from the computation graph, so no gradient flows through it. Both limit data efficiency and training quality. We therefore introduce Self-Replay Forcing (SRF). After the student performs a long autoregressive rollout, we take the entire student-generated trajectory, independently re-noise all of its segments following Diffusion Forcing [9], and replay the full trajectory in a single gradient-enabled causal pass. The original rollout and its KV caches are detached before replay, so gradients do not backpropagate through the rollout itself. Instead, all replayed segments remain connected within the same computation graph, allowing the loss of a later segment to propagate to preceding segments during the replay pass. (2) 720p Resolution. Vidu S2-Avatar raises real-time generation from 540p to 720p while keeping 25\~42 FPS. A lightweight Refiner adds a single step to lift the resolution, so the backbone can keep running fast at low resolution. We also select training videos by measured clarity instead of nominal resolution, because heavily compressed footage looks blurry even at 1080p. (3) Stronger instruction following. Vidu S2-Avatar follows a much wider range of instructions, including large body motion such as dancing. We add solo dance videos and 2D/3D animation to the training data, keep videos whose camera moves by stabilizing their backgrounds rather than discarding them, and caption each clip in the order that events happen. Reinforcement learning from human preference then further improves motion naturalness, expressiveness, and instruction adherence. (4) Reference interaction at any moment. Users can give the model a new reference image at any point in a stream, for example, an object to pick up, a piece of clothing to put on, or a background to move to. Beyond training on data built for reference-conditioned generation, we further build a VLM agentic system to make the generation more reliable.

Vidu S2-Editing. Vidu S2-Editing edits a video stream in real time. It can (1) repaint the whole video in a new visual style, such as turning a real-world video into an anime look, (2) change the clothes the person is wearing, (3) replace the person with a different character, and (4) replace the background behind the person. Each edit follows a text instruction and an optional reference image that shows the desired appearance. The key design is frame-aligned attention: every target frame reads only the source frame at the same time step, so the edited video keeps exactly the same motion and timing as the input, while the reference image stays visible to all frames so that the new appearance carries through the whole video. During streaming, each source frame is consumed together with the target frame it produces and is not kept in the cache.

Real-Time Spatial Video Generation. We further explore the feasibility of real-time spatial video generation and editing with Vidu S2-Avatar and Vidu S2-Editing. (1) Vidu S2-Avatar. The stream generated by Vidu S2-Avatar can be converted into synchronized left- and right-eye views within the streaming pipeline. This enables real-time interaction with generated characters in spatial-video form. (2) Vidu S2-Editing. The editing pipeline supports two input formats. For monocular input, Vidu S2-Editing can edit the stream first and then apply the same conversion. For stereoscopic input, it can jointly edit paired views and then split the output back into left- and right-eye views. Across both editing settings, users can perform style transfer, virtual try-on, character replacement, and background replacement. The generated or edited views can then be streamed to VR head-mounted displays, where users can experience the results as immersive, continuously updated spatial video.

## Vidu S is dedicated to creating the ultimate interactive visual experience for humans.

## Contribution. We summarize our contributions as follows.

1. We introduce Vidu S2-Avatar, a real-time interactive digital-character model that supports 720p realtime generation, dynamic references that can be updated at any moment during a stream, and stronger instruction following over a wider range of body motion, such as dancing.

2. We introduce Vidu S2-Editing, a real-time video editing model that edits an incoming video stream on the fly, covering style transfer, virtual try-on, character replacement, and background replacement.

3. We explore real-time spatial video generation and editing for VR head-mounted displays. Our streaming framework can convert avatar-generated or edited monocular streams into synchronized left- and right-eye views, or directly edit existing spatial video.

4. We build an efficient inference and serving stack that makes these models practical on low-cost GPUs. It uses SageAttention, SpargeAttention, and Sparse-Linear Attention, low-bit GEMM, kernel fusion and launch optimization, and multi-GPU parallelism that lets the VAE encoder, backbone, Refiner, and VAE decoder share GPUs along a common timeline.

5. Experiments show that Vidu S2 outperforms all baselines while fully meeting real-time inference requirements.

S2-Editing Data Pipeline

## 2 Vidu S2-Avatar

S2-Avatar Data Pipeline  
![](images/b5d6416328fc84c42137517e88d29821f813ff047af8b2f4962550ebe8ded793.jpg)  
Figure 2 Overview of the data preparation pipelines for Vidu S2-Avatar and Vidu S2-Editing. Left: diverse video sources are processed through clipping, filtering, speech processing, temporal dense captioning, and embedding, with enhanced operators for cut-point detection, high-clarity selection, and background stabilization. Right: filtered Avatar data are further curated for four editing tasks; style-transfer pairs are generated through reconstruction-based V2V training and reference-guided synthesis, followed by post-filtering and comparative selection.

## 2.1 Data Preparation

Figure 2 (left) summarizes the data preparation pipeline for Vidu S2-Avatar. Building on the data processing framework of Vidu S1, Vidu S2-Avatar retains its five-stage pipeline, Clipping, Filtering, Speech Processing, Captioning, and Embedding, while refining the clipping methods and filtering taxonomy. On the one hand, we expand our data sources to enrich the facial expressions and body movements generated by the model. On the other hand, we introduce additional processing and filtering operators to improve the quality of the training data. For video captioning, we find that temporally ordered dense captions are better suited for streaming video generation than the previous structured descriptions. We therefore redesign the captioning method, and incorporat expert models to improve coverage and reduce hallucinations.

Data Collection. In addition to continuing to expand the livestream/talking-head videos and film/television content used in Vidu S1, Vidu S2-Avatar places particular emphasis on collecting high-quality solo dance videos and 2D/3D animation data. These additions help the model learn a broader range of body movements and improve its performance in animated character generation.

Video Clipping. Vidu S2-Avatar retains the single-shot clipping approach of Vidu S1 while improving the detection of subtle edit points. We observe that some videos, particularly vlogs, contain numerous edits that are difficult to detect. For example, unboxing videos often omit the intermediate steps of opening a package, retaining only the beginning and end of the action. However, simply increasing shot detection sensitivity introduces many false positives. To address this issue, we extract frames around each candidate cut and use a vision-language model for a second-stage assessment. This procedure ensures that the resulting clips contain no cuts while keeping the false rejection rate below 2%.

Video Filtering. Vidu S2-Avatar retain the six-dimensional filtering taxonomy from Vidu S1: subject detection, frame cleanliness, visual quality, content safety, shot stability, and interactivity. We also introduce a high-clarity video selection operator to meet the stricter training-data requirements for real-time 720p video generation. In addition, we relax the hard filtering criterion for shot stability. Videos with small or smooth camera movements are retained and subsequently processed by a background stabilization operator to produce stable backgrounds.

High-Clarity Video Selection. Videos from different sources vary in compression strategy and severity, so their actual clarity can differ substantially even at the same nominal resolution. Even at 1080p or higher, heavily compressed videos may suffer from texture loss and compression artifacts. The resolution alone is therefore insufficient for selecting high-quality data. We therefore develop a multidimensional hybrid selection framework that evaluates different video types separately. Specifically, hard thresholds are first applied to resolution and frame rate, after which the interdependencies among resolution, frame rate, codec, pixel format, bit depth, and bitrate are assessed. Technical quality, texture detail, edge sharpness, and compression artifact severity are further evaluated using expert models. These assessments are aggregated into a weighted quality score, and a threshold is applied to select training data that satisfy the desired clarity criteria.

Background Stabilization. Mitigating background drift and preserving scene consistency remain key challenges in infinite-length streaming video generation, motivating the use of training videos with static backgrounds or fixed cameras. However, videos featuring rich body motion, such as dance videos, often include camera movements such as orbiting, dolly motion, zooming, and subject tracking. Excluding these videos would reduce motion diversity, whereas retaining them without stabilization could impair the learning of background consistency. To address this dilemma, we introduce a background stabilization operator. The approach is motivated by the observation that, in most dance videos, camera translation is limited and visual changes arise primarily from camera rotation and focal length adjustments. The resulting parallax is typically small, allowing background motion to be approximated by geometric transformations across frames. Videos with large, complex camera movements are therefore excluded, while those with smooth motion are retained for stabilization. For the retained videos, foreground subjects are detected and masked, after which camera transformations are estimated from the remaining background regions through feature extraction and matching. Finally, we apply geometric correction and appropriate cropping to obtain videos with stable backgrounds, followed by a second filtering pass to ensure that the subject remains in the frame.

Data Captioning. Vidu S2-Avatar preserves the two caption granularities used in Vidu S1, full-clip and chunk-level captions, while redesigning both the caption representation and the annotation pipeline to emphasize temporal structure. Vidu S1 uses structured natural-language descriptions with separate fields for subjects, environments, and camera movements. Although this format explicitly distinguishes different scene components, it provides limited support for representing their interactions, temporal evolution, and synchronization. In contrast, Vidu S2 adopts temporally ordered dense captions. Aside from a small set of tags summarizing the overall scene, captions describe events chronologically, specifying their temporal boundaries, constituent actions, and outcomes. For example, a caption may describe an action performed between 2.5 s and 3.8 s together with its consequences. This representation makes temporal and causal relationships more explicit, facilitating the modeling of dependencies across events. Consistent with this design, chunk-level captions are segmented at event boundaries rather than assigned to fixed-duration, speech aligned intervals. The resulting segments are typically finer-grained, enabling lower response latency and greater consistency during action transitions and adaptation to new instructions. To support this temporally structured representation, the annotation pipeline follows a hierarchical, multi-agent design. Expert models are introduced to perform object detection and speech recognition, producing a factual grounding layer for subsequent caption generation. Building on this layer, the workflow assigns recognition, description, verification, and filtering to dedicated agents, improving annotation quality and consistency through explicit task decomposition and validation.

## 2.2 Method

Model Overview. Vidu S2-Avatar employs an audio-visual joint Diffusion Transformer for referenceconditioned video–audio generation. Let r denote the reference image. We associate each video–audio segment $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } ^ { i } }$ with conditioning information $c ^ { i }$ , such as its caption, forming the conditioning sequence $\pmb { c } ^ { 1 : \infty } = \{ \pmb { c } ^ { 1 } , \pmb { c } ^ { \bar { 2 } } , \pmb { . . . } \}$

Given r and $c ^ { 1 : N }$ , the model jointly predicts the clean video–audio latents for the first N segments:

$$
\hat { \pmb { x } } _ { 0 } ^ { 1 : N } = f _ { \pmb { \theta } } ^ { \mathrm { a v a t a r } } \left( \pmb { x } _ { t } ^ { 1 : N } , t , \pmb { r } , \pmb { c } ^ { 1 : N } \right) ,\tag{1}
$$

where $\pmb { x } _ { t } ^ { 1 : N }$ denotes the noisy joint video–audio latents at diffusion timestep t. The reference image is shared across all segments to maintain appearance and identity.

Bidirectional Image- and Reference-to-Video Training. To support both image-to-video (I2V) and referenceto-video (R2V) generation within a single model, we jointly train a bidirectional model on the tasks:

$$
\hat { \pmb { x } } _ { 0 } ^ { 1 : N } = f _ { \theta } ^ { \mathrm { b i } } \left( \pmb { x } _ { t } ^ { 1 : N } , t , \pmb { r } , \pmb { c } ^ { 1 : N } \right) .\tag{2}
$$

For I2V training, r represents the first frame of the target video; for R2V training, it represents the provided reference image. In both tasks, we supervise each temporal segment with its corresponding condition $c ^ { i } .$ , such as its caption, instead of using a single prompt for the entire sequence as in conventional bidirectional models. This segment-wise conditional supervision substantially improves instruction following while preserving the model’s original generation quality.

Hybrid Teacher and Diffusion Forcing. Starting from the pretrained bidirectional model, we replace its bidirectional temporal attention with a block-wise causal attention mask, such that each video-audio segment can only attend to the reference image, the current segment condition, and its valid historical states. For the i-th segment, the causal denoising process is formulated as

$$
\hat { \pmb x } _ { 0 } ^ { i } = f _ { \pmb \theta } ^ { \mathrm { c a u s a l } } \left( \pmb x _ { t _ { i } } ^ { i } , t _ { i } , \pmb r , { c } ^ { i } , \pmb x _ { \tau _ { i } } ^ { < i } , \tau _ { i } \right) ,\tag{3}
$$

where $t _ { i }$ denotes the diffusion timestep for the i-th segment, and $\mathbf { \Delta } x _ { \tau _ { i } } ^ { < i }$ denotes the historical video-audio states at noise level $\tau _ { i }$ . We adopt a hybrid training strategy that combines Teacher Forcing and Diffusion Forcing [9]. Under Teacher Forcing, the model is conditioned on clean ground-truth historical states. Under Diffusion Forcing, noise is injected into the historical states at sampled noise levels, training the model to generate under the noisy condition. The two modes are sampled during training with a predefined probability. This causal adaptation equips the model with an initial capability for streaming video-audio generation while improving its robustness to accumulated errors in the generated history.

Self-Replay Forcing. We introduce Self-Replay Forcing (SRF), an on-policy DMD [10–13] that aligns the student’s autoregressive rollout distribution with the teacher distribution while preserving cross-block gradient flow. The model first performs a long autoregressive rollout following Self-Forcing [8], with the generated blocks and KV caches detached to avoid retaining the full rollout computation graph. Here, “on-policy” refers to the autoregressive trajectory and historical contexts generated by the current student model under its inference procedure. After the rollout is completed, we sample self-generated trajectory and re-noise its blocks following Diffusion Forcing. The noisy video is then processed by a gradient-enabled causal replay, while the external history inherited from the detached rollout remains fixed. DMD supervision is applied to all blocks within the replayed segment. Because the replay-segment representations remain connected within the same computation graph, gradients can propagate across block boundaries during the replay pass without backpropagating through the original rollout. Following the perceptual regularization used in Vidu S1 [14], we additionally apply a perceptual loss to the replayed student outputs to mitigate mode collapse and preserve generation diversity. Following Vidu S1, we retain sink blocks, a sliding-window context, and noisy KV caches throughout training.

Let $\hat { \pmb { x } } _ { 0 } ^ { 1 : N }$ denote a detached autoregressive rollout of the current student, and let $\pmb { x } _ { t } ^ { 1 : N }$ denote its re-noised counterpart, where $\pmb { t } = ( t _ { 1 } , \ldots , t _ { N } )$ specifies the diffusion timestep for each segment. Under noisy-history conditioning, we optimize the replayed student outputs using

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S R F } } = \mathcal { L } _ { \mathrm { D M D } } \left( f _ { \theta } ^ { \mathrm { c a u s a l } } \left( x _ { t } ^ { 1 : N } , t , r , c ^ { 1 : N } \right) \right) + \mathcal { L } _ { \mathrm { p e r c } } ( f _ { \theta } ^ { \mathrm { c a u s a l } } \left( x _ { t } ^ { 1 : N } , t , r , c ^ { 1 : N } \right) ) . } \end{array}\tag{4}
$$

Preference Optimization We apply preference- and reward-based optimization at both the bidirectional and streaming stages to mitigate visual quality degradation and improve instruction following. At the bidirectional stage, we use diffusion-based Direct Preference Optimization (DPO) [15] to enhance visual fidelity, facial expressiveness, motion naturalness, and audio–visual synchronization, thereby providing a stronger teacher for subsequent causal adaptation. At the streaming stage, we apply Streaming Negative-aware Fine-Tuning (Streaming NFT) [16] to the causal backbone using self-generated trajectories constructed following the principle of Self-Replay Forcing. Specifically, the model first performs a detached autoregressive rollout and then replays sampled trajectory for reward-based optimization. This procedure aligns the optimization states with the inference-time autoregressive distribution, improving visual quality, motion controllability, and instruction adherence under real-time streaming inference.

Super-Resolution Refiner. To recover fine-grained spatial details from the low-resolution latent outputs of the causal backbone, we introduce a one-step super-resolution refiner that operates directly in the latent space. Inspired by the stage-aware cache design of TwinCache introduced in Vidu S1 [14], we use asymmetric noise levels for the historical caches of the causal backbone and the Refiner. Specifically, the backbone attends to a high-noise cache $\hat { \pmb { x } } _ { \tau _ { \mathrm { B } } } ^ { < i }$ to propagate coarse motion and long-range temporal structure, whereas the Refiner attends to a low-noise high-resolution cache $\hat { \pmb x } _ { \tau _ { \mathrm { R } } , \mathrm { H R } } ^ { < i }$ , where $\tau _ { \mathrm { B } } > \tau _ { \mathrm { R } }$ . Given the low-resolution latent $\hat { \pmb { x } } _ { 0 , \mathrm { L R } } ^ { i }$ generated for the i-th segment, the Refiner first performs latent-space upsampling and adds noise at a fixed refinement timestep $t _ { r } ,$ and then predicts the corresponding high-resolution latent:

$$
\begin{array} { r } { z _ { t _ { r } } ^ { i } = ( 1 - t _ { r } ) \mathcal { U } \big ( \hat { x } _ { 0 , \mathrm { L R } } ^ { i } \big ) + t _ { r } \epsilon ^ { i } , \qquad \epsilon ^ { i } \sim \mathcal { N } ( \mathbf { 0 } , I ) , } \end{array}\tag{5}
$$

where U denotes latent-space spatial upsampling, $\epsilon ^ { i } \sim \mathcal { N } ( 0 , I )$ . The cache $\hat { \pmb x } _ { \tau _ { \mathrm { R } } , \mathrm { H R } } ^ { < i }$ denotes the historical high-resolution context maintained at the low noise level $\tau _ { \mathrm { R } }$ . All chunks in a replayed trajectory share the same cache noise level $\tau _ { \mathrm { R } ; }$ , while the refinement timestep $t _ { r }$ is fixed across segments. This asymmetric cache scheduling separates temporal propagation from spatial detail restoration. The high-noise backbone cache provides a coarse temporal prior that is less sensitive to high-frequency artifacts, thereby stabilizing long-range motion propagation. In contrast, the low-noise high-resolution cache preserves local appearance and identity cues that are important for spatial refinement. Consequently, the backbone preserves long-range motion consistency, while the Refiner recovers fine-grained details without disrupting the temporal trajectory.

## 2.3 Inference Infrastructure

To deliver fast and efficient inference for Vidu S2-Avatar, we adopt and extend selected techniques from TurboDiffusion [6] and TurboServe [7] in an end-to-end inference acceleration framework. We optimize operator execution and memory access, and arrange module execution along a shared timeline. This schedule allows different modules to share GPU resources at different times. Together, these optimizations preserve generation quality while improving inference efficiency and overall resource utilization, enabling real-time, low-latency inference services. The main components are described below.

Efficient Attention. In streaming video generation with diffusion models, attention contributes significantly to execution time and computational cost [17], making it a key bottleneck for low-latency inference. Different layers respond differently to reduced precision and approximate attention computation. We therefore use a layer-wise hybrid attention strategy. For each layer, we choose a suitable method from SageAttention [18–22], SpargeAttention [23, 24], and Sparse-Linear Attention (SLA) [25, 26]. We use more aggressive methods in less-sensitive layers and prioritize accuracy in sensitive layers. This strategy reduces attention latency while maintaining generation quality.

Quantized Linear-Layer Acceleration. Running linear layers at BF16 or FP16 precision preserves accuracy, but their high computational cost makes real-time inference difficult and leads to high latency. Per-tensor and per-channel quantization methods are faster, but a few outliers can dominate their quantization ranges. As a result, most values are represented with low precision, which can severely degrade output video quality. We therefore develop an accurate and efficient CUDA implementation of per-block W8A8 GEMM [27] for linear layers. Fine-grained scaling limits the effect of outliers, while the optimized kernels reduce memory usage and accelerate linear-layer computation. The resulting operator maintains numerical precision and generation quality while reducing inference latency.

Kernel Fusion and Launch Optimization. Streaming video inference repeatedly executes many short operators in a stable pattern. In this setting, kernel launches and synchronization add noticeable overhead, while intermediate tensors create extra global-memory traffic. We address these costs in two ways. First, we fuse adjacent operations into custom Triton/CUDA kernels. Fusion reduces the number of kernel launches and cuts global-memory reads and writes for intermediate results. For example, we fuse RMSNorm with selected elementwise operations. Second, we use CUDA Graphs for stable execution sequences. Each graph is captured once and replayed one or more times per inference step. A replay replaces many host-side kernel launches with a single graph launch. Together, these techniques reduce launch overhead, synchronization costs, and global-memory traffic while improving GPU utilization.

Multi-GPU Parallelism. Real-time inference must meet strict latency targets under limited compute and memory budgets. We address this challenge with multi-GPU context parallelism. We use Ulysses-style context parallelism [28] to divide the workload across GPUs and distribute activation memory. Context parallelism introduces collective communication between devices. We quantize the exchanged tensors to reduce both transfer volume and communication latency. This allows multi-GPU execution to scale more efficiently.

## 2.4 Agentic System

In Vidu S2-Avatar, users provide an instruction through text or speech, upload an initial image, and optionally add reference images (Figure 3). Spoken instructions are transcribed into text. A vision-language model (VLM) agent uses these inputs to generate prompts and reviews the generated frames to refine subsequent prompts.

![](images/788d07f767fd8f9b470c9cb652acec34472c0c467e5dc63bf0284b9dbbcf82fa.jpg)  
Figure 3 Agentic System Pipeline. The VLM agent reads user instructions and images, writes prompts for Vidu S2-Avatar, and checks generated video frames to guide later prompts. The pipeline is schematic.

Prompt Generation. The prompts separately describe the character’s identity and appearance, its expression, gaze, pose, and actions, and any object that should remain held. The prompts retain details that the user’s instruction does not ask to change. For example, in the instruction “pick up a cup, then smile,” once the cup has been picked up, the character should continue holding it while smiling. Subsequent prompts preserve this intended state by explicitly specifying that the character continues to hold the cup unless the user asks to release or replace it. The prompt for the smile therefore changes only the expression description.

Visual Feedback. The VLM reviews frames in time order and reports whether the requested action has finished, has been partly performed, or differs from the request. The system checks confidence and, when required, agreement across repeated observations. An accepted completion judgment can lead to a prompt describing the resulting pose or held object without repeating the action. Incomplete or incorrect actions can lead to revised instructions; an inconclusive judgment does not confirm success.

![](images/b75d61df566c4cd785279a495a24d368eb6ad80b3894ddda58f5fa3d7fa74f4d.jpg)

Reference Images. The agent first determines whether a reference image shows a handheld object, a background, or clothing. It then combines the reference content with the user’s request to generate prompts describing how the character and scene in the initial image should change. This supports object replacement, background changes, and one-click outfit changes, while retaining details unrelated to the request. For a scene transition, the prompt can describe the character leaving the original view and entering the referenced scene. Figure 4 illustrates object, scene, and accessory changes.

Taking Off and Putting On Accessories. The prompt describes both the movement and whether the accessory should remain worn afterward. When the user asks to take off a hat and put it back on, the prompt describes reaching for it, lifting it off, holding it, and returning it to the head. Later prompts continue to specify the hat as worn unless the user asks to remove it.

a. Reference Object Generation & Replacement

b. Reference Scene Transition

![](images/9a61fb5b6084e9ce493c8fd1432ab7c02ebefac38925424caa0662d0613c7383.jpg)

c. Taking Off and Putting On Accessories

![](images/645c6223e34fd70e691cbde0deb5d14ca9fbd61faa0b237295d66fa4c1edebe5.jpg)  
Figure 4 Reference and Accessory Control. (a) Initial images plus a blue-mug reference illustrate object generation (left) and replacement (right). (b) The character leaves the original scene and enters a new one. (c) Taking off and putting back on the same hat.

## 3 Vidu S2-Editing

## 3.1 Data Preparation

Data Collection. As illustrated in Figure 2 (right), the training data for Vidu S2-Editing is derived from videos filtered through the Vidu S2-Avatar data pipeline. We apply stricter filtering criteria and balance the label distribution, yielding a collection of 800, 000 videos with high visual quality and diverse content. From this collection, we sample four mutually disjoint subsets of 200, 000 videos each to construct training data for four editing tasks: style transfer, subject replacement, background replacement, and virtual try-on.

Video Editing Data. To construct training data for style transfer, we train a conditional video generation model that synthesizes a spatially consistent video from a surface-normal video and a reference image. Given a raw video clip $v _ { \mathrm { r a w } }$ , we sample one frame from the clip as the reference image $i _ { \mathrm { r e f } }$ and use a model such as NormalCrafter [29] to estimate its corresponding surface-normal video $v _ { \mathrm { n o r m a l } }$ . We then train the conditional model to reconstruct $v _ { \mathrm { r a w } }$ given $v _ { \mathrm { n o r m a l } }$ and $i _ { \mathrm { r e f } } ,$ , thereby learning to preserve the spatial structure and motion specified by the normal condition while following the appearance of the reference image. To generate style transfer data, we replace the original reference image with a stylized image while retaining the surface-normal video estimated from the original clip. The model consequently generates a stylized video that follows the original video’s spatial structure and motion. Our evaluations show that the model generalizes well to more than 50 photorealistic and non-photorealistic styles, including cel shading, cyberpunk, Monet-style painting, sketching, and Chinese gongbi painting, among others.

For other editing tasks, we follow a similar data construction procedure, fine-tuning task-specific video editing models to generate paired training data. We also evaluate several video-editing open-source models, including Bernini [30], SCAIL-2 [31], Wan2.1 VACE [32], SAMA-14B [33], and CoinVE-Edit [34], to supplement the editing data. During data construction, we observe that these models have different strengths across tasks and samples, with no single model consistently producing the best results across all editing scenarios. We therefore apply post-filtering and comparative selection to the candidate outputs from different models, retaining high-quality source–edited video pairs for the final training set.

Data Captioning. Each caption identifies the editing task as style transfer, subject replacement, background replacement, or virtual try-on, and describes the attributes of the reference image relevant to that task. For example, style-transfer captions characterize the image’s visual style, while virtual try-on captions describe its clothing attributes. These descriptions define how the reference image should guide the edit. The prompts also explicitly require the model to preserve all source-video content outside the intended edits. To support streaming video editing, captions are generated without feeding the source video to the VLM. The resulting captions thus specify editing objectives and preservation constraints without describing the source video itself, preventing its content from leaking through text conditioning.

## 3.2 Method

Model Overview. Vidu S2-Editing employs a Diffusion Transformer (DiT) for instruction-guided video editing conditioned on a source video and an optional reference image. Let $s ^ { 1 : N }$ denote the source-video condition and r the reference-image representation. We represent the remaining conditions as $c ^ { 1 : \infty } =$ $\{ c ^ { 1 } , c ^ { 2 } , \ldots \}$ , where $c ^ { i }$ denotes the conditioning information for the i-th video segment, such as its editing instruction. For N target segments, the model predicts the clean video latents as

$$
\hat { \pmb { x } } _ { 0 } ^ { 1 : N } = { \pmb f } _ { \pmb { \theta } } ^ { \mathrm { e d i t } } \left( { \pmb x } _ { t } ^ { 1 : N } , t , { \pmb s } ^ { 1 : N } , { \pmb r } , { \pmb c } ^ { 1 : N } \right) ,\tag{6}
$$

where $\pmb { x } _ { t } ^ { 1 : N }$ denotes the noisy target-video latents at diffusion timestep t. The source video provides spatial and motion context, while the reference image supplies additional appearance information when available. We encode the source video and any reference image as conditioning tokens and concatenate them with the noisy target-video tokens. Condition-specific RoPE encodings represent the spatial and temporal positions of the different token streams.

Bidirectional Video Editing Training. We first train a bidirectional model to establish strong visual and motion priors for video editing. The generated-video tokens interact bidirectionally across the target sequence, while their interaction with the source video is restricted to temporally aligned frame pairs. Each generated frame exchanges information only with its corresponding source frame, avoiding unrestricted mixing between misaligned source and target contents. Meanwhile, every generated frame can attend to the reference-image tokens, allowing the reference appearance to be consistently propagated throughout the video. This frame aligned attention design preserves the modeling capacity of bidirectional training while making the learned conditioning structure compatible with subsequent causal adaptation.

Causal Streaming Training. The causal streaming training procedure for Vidu S2-Editing follows the same hybrid forcing and Self-Replay Forcing framework described for Vidu S2-Avatar. Starting from the bidirectional editing model, we replace bidirectional temporal attention with a block-wise causal attention mask and mix Teacher Forcing with Diffusion Forcing to improve robustness to imperfect historical contexts. The model is conditioned on the source video, reference image, editing prompts, and valid historical states available up to the current segment. We then apply Self-Replay Forcing by first performing a detached autoregressive rollout, followed by re-noising and differentiable replay of the self-generated trajectory. DMD supervision is applied to the replayed blocks, allowing the model to learn from inference-time states without backpropagating through the original rollout. This procedure equips Vidu S2-Editing with an capability for real-time streaming video editing while improving temporal consistency and instruction adherence.

## 3.3 Inference Infrastructure

For Vidu S2-Editing, the source video stream is produced continuously on the user side. The model must process each incoming video frame promptly to keep the edited output synchronized with user actions. In real-time editing tasks, the same inference delay is more noticeable than in generation tasks because users directly compare their actions with the edited output. Based on the inference optimizations developed for Vidu S2-Avatar, which are described in section 2.3, we further improve how GPU resources are scheduled across modules. The following design reduces idle GPU resources and lowers end-to-end latency.

Inter-Module Scheduling. Modules are active at different stages of streaming video inference. A static placement policy reserves a fixed GPU group for each module, even when the module is idle. This wastes GPU resources and increases end-to-end latency when GPU capacity is limited. We instead use fine-grained scheduling to coordinate the VAE encoder, backbone, refiner, and VAE decoder on a shared timeline. When the backbone or refiner does not need certain GPUs, the VAE encoder or decoder can reuse them. This time-based sharing reduces dedicated GPU requirements, improves overall resource utilization, and lowers inference latency.

## 4 Real-Time Spatial Video Generation and Editing

Spatial video presents a slightly different image to each eye. This lets viewers see depth, scale, and the positions of objects in a scene. Compared with flat video, it creates a stronger sense of presence. Characters and scenes can feel as if they are around the user instead of on a screen. This creates a more immersive entertainment experience.

## 4.1 Real-Time Spatial Video Generation

Motivation. We want real-time video generation to offer a more immersive entertainment experience. Vidu S2-Avatar can already generate monocular video in real time, with rich expressions, motion, and interactive responses. However, flat video gives a weaker sense of space and presence. We therefore aim to convert its streaming output into synchronized left and right views while preserving the real-time responsiveness needed for continuous interaction.

Generation Pipeline. Our generation pipeline has two stages. First, Vidu S2-Avatar uses the current text, audio, visual references, other controls, and video history to generate the next monocular video chunk. Second, the conversion stage estimates per-frame depth and maps it to horizontal disparity [35]. Treating the monocular frame as a center view, it warps the image in opposite horizontal directions to synthesize the left and right views. Warping leaves holes along object boundaries and newly exposed regions [36, 37]. For low-latency streaming, we use lightweight processing to mitigate artifacts near depth discontinuities and fill holes in disoccluded regions. We temporally stabilize depth estimates to reduce fluctuations in perceived depth. Finally, the two views are synchronized as streaming video chunks. The streaming spatial video allow users to perceive the distance between the character and the background, the scale of objects, and the depth of the scene, making the generated character feel more present in the user’s space. This streaming design preserves the capability of Vidu S2-Avatar to support real-time character interaction in spatial video form.

## 4.2 Real-Time Spatial Video Editing

Motivation. Many existing videos are available only in monocular form. Spatial video is costly to produce and difficult to modify once captured or generated. These limitations motivate two workflows: editing ordinary monocular video and then converting it into synchronized left- and right-eye views, or directly editing existing spatial video.

Editing Pipeline. Our real-time spatial video editor is based on Vidu S2-Editing and supports two workflows. (1) Monocular input. We first edit the incoming monocular stream with Vidu S2-Editing and then pass the edited output through the same spatial video conversion pipeline used for Vidu S2-Avatar, producing synchronized left- and right-eye views. (2) Stereoscopic input. For spatial video, we join the left and right views horizontally to form one wide video stream. Vidu S2-Editing edits both views in one pass with the same visual references, then we split the output back into left and right views.

Both workflows can send synchronized output to the left- and right-eye displays. Whether the input is monocular or stereoscopic, users can perform style transfer, virtual try-on, character replacement, and background replacement. These edits can be applied either to video content or to the physical world as seen through a live camera or headset passthrough.

## 4.3 Discussion

The generation and editing pipelines above demonstrate that Vidu S2 can extend real-time monocular video to spatial-video experiences. However, for practical deployment, this task requires higher resolution and lower latency than conventional real-time monocular video generation. When viewed through a headset, spatial video often occupies a large portion of the user’s field of view and therefore requires high resolution to maintain visual comfort. Low latency is equally important. For example, in camera-based video passthrough mode, high latency can make head motion and the displayed view fall out of sync. Achieving low end-to-end latency while maintaining high visual quality remains a key technical challenge.

Beyond these challenges, a promising future direction is to extend the current system from fixed-view spatial video to panoramic spatial video, allowing users to turn their heads and freely explore generated scenes. We believe this capability could also support a metaverse whose visual content is generated in real time. Scenes, characters, and events could change with user interaction. Such real-time video generation could substantially reduce the need for traditional 3D modeling, material creation, animation, and scene building.

## 5 Experiments

We evaluate Vidu S2 on two complementary tasks: streaming digital character generation and streaming video editing. section 5.1 introduces the task definitions, benchmark suite, comparison systems, and evaluation protocols. section 5.2 reports results against open-source systems on public benchmarks, with separate analyses for digital character generation and video editing. section 5.3 assesses commercial systems on internal benchmarks using randomized Good/Same/Bad (GSB) pairwise comparisons. Finally, section 5.4 examines representative cases of identity preservation, temporal consistency, robustness during long-horizon streaming and promising spatial video generation ability.

## 5.1 Experimental Setup

This section defines the task and benchmark suite in section 5.1.1, summarizes the comparison systems and access conditions in section 5.1.2, and specifies the public-benchmark and GSB protocols in section 5.1.3.

## 5.1.1 Tasks and Benchmark

We study two tasks. For digital character generation, Vidu S2-Avatar generates an audio-driven streaming character from an audio track, a reference identity image, and a textual condition. For video editing, Vidu S2-Editing performs instruction-guided and reference-conditioned edits in the streaming setting.

For digital character generation, we use StreamAV-Bench [38] for public evaluation and an internal benchmark for commercial-system comparison. StreamAV-Bench contains 160 scenarios in each of its Progressive and Interactive tracks. The Interactive Track receives one runtime update every 30 seconds for five updates and runs for up to 180 seconds. Our internal digital character benchmark covers streaming quality, long-horizon stability, runtime updates, interruption and recovery, audio–video synchronization, system responsiveness, and cost. It follows the StreamAV-Bench evaluation protocol wherever the measurements are shared.

For video editing, we use OpenVE-Bench [39], Sparkle-Bench [40], RefVIE-Bench [41], and the ViViD test set [42]. OpenVE-Bench and Sparkle-Bench evaluate instruction-guided editing; RefVIE-Bench adds a reference image for subject and background replacement; and the ViViD test set evaluates unpaired virtual tryon. Our internal editing benchmark covers style transfer, virtual try-on, subject replacement, and background replacement across diverse subjects, references, motions, and scene layouts. We also construct a long-horizon set with clips at the 10-minute level to test identity preservation, edited-region stability under motion and occlusion, and temporal stability of generated backgrounds.

## 5.1.2 Baselines

For the public digital character evaluation, we compare with the systems listed in Table 1: PixVerse R1 [43], HappyOyster [44], OmniForcing [45], Odyssey-2 [46], Self-Forcing [8], LongLive [47], SWIFT [48], IAMFlow [49], SoulX-FlashTalk [50], AptAvatar [51], Live Avatar [52], AvatarForcing [53], and Hallo-Live [54]. The internal duration-stratified curves in Figure 5 compare Vidu S2-Avatar with Vidu S1 [14], Runway, HeyGen [55], and PixVerse [43]. For public video editing, we compare Vidu S2-Editing with the systems appearing in Tables 2–4: Bernini-R 1.3B and Bernini-R 14B [30], Kiwi-Edit 5B [41], VInO [56], OmniWeaving [57], LiveEdit [58], VACE 1.3B [32], JoyAI-Video-Edit [59], StreamDiffusionV2 [60], XMax-X2.0 [61], Decart-Lucy2.5 [62], StableVITO [63], OOTDiffusion [64], IDM-VTON [65], StableVITON [63], StableVITON+AM, OOTDiffusion+AM, IDM-VTON+AM, ViViD [42], and CatV<sup>2</sup>TON [66]. The internal editing GSB comparison in Figure 7 includes XMax-X2.0 and Decart-Lucy2.5. Face-only APIs are evaluated in a separate lip-sync and latency track and are not merged with full-body systems.

## 5.1.3 Metrics

For public benchmarks, we use identical inputs, preprocessing, temporal sampling, evaluators, and scoring rubrics. Every system receives the same manifest, audio, and reference image. We reuse a published result only when the source reports the same benchmark, split, and metric; otherwise, we omit the entry. For metrics inherited from a benchmark, we retain the benchmark’s original directionality and procedure.

For digital character generation, we report the metrics defined by StreamAV-Bench [38]. Visual Aesthetics (VA) measures perceptual visual appeal with the LAION Aesthetic Predictor [67]. Visual Quality (VQ) is a Geminibased assessment of visual fidelity, subject integrity, motion naturalness, and visible artifacts. Production Quality (PQ) measures the perceptual quality and production fidelity of the generated audio with AudioBox Aesthetics [68], whereas Audio Quality (AQ) uses the Gemini judge to assess audio naturalness and audible artifacts. Audio–Visual Alignment (AVAlign) measures semantic correspondence between the generated audio and video using ImageBind similarity [69]; Audio–Visual Synchronization (AVSync) estimates the temporal synchronization error with Synchformer [70], so lower values are better. Audio Instruction Fulfillment (AIF) measures how completely the audio-side requirements in the case-specific checklist are realized. Subject Consistency (SC) and Background Consistency (BC) follow the long-horizon protocol of StreamAV-Bench and quantify the temporal stability of the subject appearance and background scene, respectively. The Geminibased metrics use the benchmark’s ordinal rating rubric, whereas VA, PQ, AVAlign, AVSync, SC, and BC retain their native metric scales. FVD and FID are computed with the benchmark tools following their original definitions [71, 72]; they measure distributional differences between generated and reference video or image features, with lower values indicating closer distributions.

For video editing, the public benchmarks retain their task-specific definitions. OpenVE-Bench [39] reports Global Style (GS), which evaluates whether the requested global appearance or style is achieved, Background Change (BC), which evaluates the correctness of the background edit, and an Overall score that aggregates the benchmark’s judged editing quality. Sparkle-Bench [40] reports global instruction compliance (Ins.), global visual quality (Vis.), foreground instruction adherence (FgIn.), foreground motion preservation (FgMo.), background dynamics (BgDy.), and background visual quality (BgVi.). RefVIE [41] evaluates referenceconditioned editing through reference fidelity, matting or boundary quality, visual harmony, and temporal consistency; its reported RefVIE Overall is the benchmark-defined aggregate of these dimensions. Joint Overall is the aggregate specified by the joint OpenVE–RefVIE protocol. For the unpaired virtual try-on task, VFID uses I3D and 3D-ResNetXt101 video features to measure the distance between generated and reference video distributions, thereby reflecting both visual quality and temporal coherence; lower values are better.

Table 1 Results on StreamAV-Bench. “–” denotes an unavailable measurement. Bold indicates the best value in each column, underlining indicates the second-best value, and dark-blue shading marks our method. ↑/↓ denote higher/lower is better.
<table><tr><td>Model</td><td>VA↑</td><td>VQ↑</td><td>PQ↑</td><td>AQ↑</td><td></td><td>AVAlign↑ AVSync↓</td><td>AIF↑</td><td>SC↑</td><td>BC↑</td></tr><tr><td>SoulX-FlashTalk</td><td>.281</td><td>2.515</td><td>7.091</td><td>3.006</td><td>.121</td><td>.648</td><td>2.499</td><td>.946</td><td>.964</td></tr><tr><td>AptAvatar</td><td>.458</td><td>2.040</td><td>7.075</td><td>2.582</td><td>.099</td><td>1.284</td><td>2.448</td><td>.860</td><td>.923</td></tr><tr><td>OmniForcing</td><td>.554</td><td>2.374</td><td>6.351</td><td>2.528</td><td>.119</td><td>1.423</td><td>2.343</td><td>.962</td><td>.953</td></tr><tr><td>Hallo-Live</td><td>.574</td><td></td><td>5.396</td><td></td><td>.102</td><td>1.353</td><td></td><td>.991</td><td>.977</td></tr><tr><td>Self-Forcing</td><td>.585</td><td>2.753</td><td>6.453</td><td>2.623</td><td>.256</td><td>.919</td><td>2.810</td><td>.981</td><td>.969</td></tr><tr><td>IAMFlow</td><td>.602</td><td>2.840</td><td>6.415</td><td>2.625</td><td>.268</td><td>1.016</td><td>2.789</td><td>.986</td><td>.973</td></tr><tr><td>LongLive</td><td>.605</td><td>2.768</td><td>6.459</td><td>2.696</td><td>.272</td><td>.969</td><td>2.814</td><td>.986</td><td>.973</td></tr><tr><td>SWIFT</td><td>.605</td><td>2.735</td><td>6.483</td><td>2.680</td><td>.270</td><td>.970</td><td>2.887</td><td>.986</td><td>.973</td></tr><tr><td>AvatarForcing</td><td>.639</td><td></td><td>7.070</td><td></td><td>.089</td><td>1.041</td><td></td><td>.987</td><td>.976</td></tr><tr><td>PixVerse R1</td><td>.529</td><td>2.428</td><td>6.348</td><td>2.810</td><td>.234</td><td>.855</td><td>2.511</td><td>.907</td><td>.930</td></tr><tr><td>Odyssey-2</td><td>.531</td><td>2.820</td><td>6.426</td><td>2.659</td><td>.260</td><td>.946</td><td>2.778</td><td>.976</td><td>.966</td></tr><tr><td>HappyOyster</td><td>.529</td><td>2.785</td><td>6.817</td><td>3.157</td><td>.206</td><td>1.044</td><td>2.692</td><td>.904</td><td>.934</td></tr><tr><td>Live Avatar</td><td>.661</td><td>3.295</td><td>7.133</td><td>3.079</td><td>.116</td><td>1.145</td><td>2.745</td><td>.997</td><td>.989</td></tr><tr><td>Vidu S2-Avatar</td><td>.687</td><td>3.370</td><td>7.138</td><td>3.286</td><td>.353</td><td>.617</td><td>2.985</td><td>.998</td><td>.993</td></tr></table>

For internal benchmarks, we use inputs and randomize the presentation order of every pair. Twenty professionally trained evaluators with relevant backgrounds in computer vision, computer graphics, video production, or visual quality assessment perform the pairwise comparisons. Before formal annotation, they complete calibration rounds with annotated examples and a written rubric; the calibration is used to align the interpretation of motion quality, visual quality, consistency, audio–video synchronization, and semantic compliance. Evaluators inspect the same source inputs and synchronized outputs and select Good when one system is clearly preferred, Same when the difference is not perceptually meaningful, and Bad when the other system is preferred. GSB denotes these Good, Same, and Bad outcomes. Any aggregate win rate specifies its denominator and tie treatment. We retain the input pair, random seed, model version, service region, and annotation record for every sample. Product-page claims about frame rate or latency do not replace measured results.

## 5.2 Public-Benchmark Results

This section compares Vidu S2 with open-source systems under the unified protocols in section 5.1.3. We first report digital character generation results in section 5.2.1, followed by video editing results in section 5.2.2.

## 5.2.1 Digital-Character Generation

StreamAV-Bench [38]. Table 1 reports the available Gemini-MLLM subset. Vidu S2-Avatar achieves the best value in every reported metric, and the gains are distributed across the complementary dimensions defined by StreamAV-Bench rather than concentrated in a single aspect. The visual metrics (VA and VQ) indicate stronger perceptual appeal and visual fidelity, while the audio metrics (PQ and AQ) show that the generated speech remains clean and natural. The improvement is especially important for an audio-driven avatar: AVAlign and AVSync jointly indicate tighter semantic coupling and more reliable temporal synchronization between speech and facial motion, and AIF further reflects stronger fulfillment of the audio-side instructions.

Table 2 Sparkle-Bench results. Ins., Vis., FgIn., FgMo., BgDy., and BgVi. denote global instruction, global visual quality, foreground instruction, foreground motion, background dynamics, and background visual quality. Bold indicates the best value in each column, underlining indicates the second-best value, light-gray shading denotes offline models, pale-blue shading denotes streaming models, and darker-blue shading highlights our streaming method. ↑/↓ denote higher/lower is better.
<table><tr><td>Model</td><td>Overall↑</td><td>Ins.↑</td><td>Vis.↑</td><td>FgIn.↑</td><td>FgMo.↑</td><td>BgDy.↑</td><td>BgVi.↑</td></tr><tr><td>Bernini-R 1.3B</td><td>3.43</td><td>3.64</td><td>3.18</td><td>3.48</td><td>3.53</td><td>3.24</td><td>3.51</td></tr><tr><td>Bernini-R 14B</td><td>3.50</td><td>3.70</td><td>3.22</td><td>3.60</td><td>3.69</td><td>3.24</td><td>3.59</td></tr><tr><td>Kiwi-Edit 5B</td><td>3.57</td><td>3.84</td><td>3.33</td><td>3.76</td><td>3.81</td><td>3.00</td><td>3.68</td></tr><tr><td>VInO</td><td>3.39</td><td>3.62</td><td>3.20</td><td>3.43</td><td>3.56</td><td>3.08</td><td>3.47</td></tr><tr><td>OmniWeaving</td><td>3.39</td><td>3.66</td><td>3.14</td><td>3.51</td><td>3.65</td><td>2.92</td><td>3.47</td></tr><tr><td>VACE 1.3B</td><td>2.18</td><td>2.22</td><td>2.17</td><td>2.20</td><td>2.21</td><td>2.11</td><td>2.19</td></tr><tr><td>LiveEdit</td><td>3.31</td><td>3.60</td><td>3.03</td><td>3.43</td><td>3.58</td><td>3.01</td><td>3.22</td></tr><tr><td>JoyAI-Video-Edit</td><td>3.48</td><td>3.74</td><td>3.22</td><td>3.66</td><td>3.74</td><td>3.09</td><td>3.45</td></tr><tr><td>StreamDiffusionV2</td><td>2.01</td><td>2.01</td><td>2.01</td><td>2.00</td><td>2.00</td><td>2.01</td><td>2.01</td></tr><tr><td>XMax-X2.0</td><td>3.03</td><td>3.24</td><td>2.82</td><td>2.96</td><td>3.23</td><td>2.91</td><td>3.01</td></tr><tr><td>Decart-Lucy2.5</td><td>3.67</td><td>3.91</td><td>3.31</td><td>3.86</td><td>3.91</td><td>3.30</td><td>3.74</td></tr><tr><td>Vidu S2-Editing</td><td>3.74</td><td>4.00</td><td>3.34</td><td>3.98</td><td>4.00</td><td>3.37</td><td>3.76</td></tr></table>

Table 3 Joint OpenVE and RefVIE evaluation. “–” denotes an unavailable measurement. Bold indicates the best value in each column, underlining indicates the second-best value, light-gray shading denotes offline models, pale-blue shading denotes streaming models, and darker-blue shading highlights our streaming method. ↑/↓ denote higher/lower is better.
<table><tr><td>Model</td><td>OpenVE GS↑</td><td>OpenVE BC↑</td><td>OpenVE Övr.↑</td><td>RefVIE Ovr.↑</td><td>Joint Ovr.↑</td></tr><tr><td>Bernini-R 1.3B</td><td>3.24</td><td>3.68</td><td>3.46</td><td>2.89</td><td>3.32</td></tr><tr><td>Bernini-R 14B</td><td>4.30</td><td>4.10</td><td>4.20</td><td>3.09</td><td>3.92</td></tr><tr><td>Kiwi-Edit 5B</td><td>3.67</td><td>2.82</td><td>3.24</td><td>2.93</td><td>3.16</td></tr><tr><td>OmniWeaving</td><td>4.28</td><td>2.83</td><td>3.55</td><td>3.08</td><td>3.43</td></tr><tr><td>VInO</td><td>4.24</td><td>1.83</td><td>3.02</td><td>2.59</td><td>2.91</td></tr><tr><td>VACE 1.3B</td><td>2.15</td><td>1.10</td><td>1.62</td><td>2.01</td><td>1.72</td></tr><tr><td>XMax-X2.0</td><td>3.17</td><td>1.77</td><td>2.46</td><td>3.16</td><td>2.64</td></tr><tr><td>Decart-Lucy2.5</td><td></td><td></td><td></td><td>3.71</td><td></td></tr><tr><td>Vidu S2-Editing</td><td>4.71</td><td>4.14</td><td>4.42</td><td>3.78</td><td>4.26</td></tr></table>

Finally, the near-ceiling SC and BC scores suggest that the identity and scene structure remain stable over long rollouts. Taken together, these results point to a more balanced streaming system that preserves visual quality while maintaining audio quality, cross-modal coordination, and long-horizon continuity.

## 5.2.2 Video Editing

Sparkle-Bench [40]. Vidu S2-Editing achieves the highest Overall score (3.74) and the highest scores for global instruction (4.00), global visual quality (3.34), foreground instruction (3.98), foreground motion (4.00), background dynamics (3.37), and background visual quality (3.76) in Table 2. These results show that Vidu S2-Editing performs competitively across both foreground- and background-oriented criteria.

OpenVE and RefVIE [39, 41]. As shown in Table 3, Vidu S2-Editing achieves the highest scores across all reported metrics, with a Joint Overall score of 4.26, surpassing the strongest offline baseline, Bernini-R 14B, by 0.34. On OpenVE, our model leads in both Global Style (4.71) and Background Change (4.14), achieving an Overall score of 4.42. On RefVIE, it attains an Overall score of 3.78, outperforming the streaming baseline Decart-Lucy2.5 by 0.07. These results demonstrate that Vidu S2-Editing supports streaming video editing while outperforming the evaluated offline and streaming baselines on these benchmarks.

Table 4 Unpaired virtual try-on evaluation on the ViViD test set. Bold indicates the best value, underlining indicates the second-best value, light-blue shading marks our method. ↑/↓ denote higher/lower is better.
<table><tr><td>Method</td><td>VFIDI↓</td></tr><tr><td>StableVITO</td><td>36.8985</td></tr><tr><td>OOTDiffusion</td><td>35.3170</td></tr><tr><td>IDM-VTON</td><td>25.4972</td></tr><tr><td>StableVITON+AM</td><td>22.0262</td></tr><tr><td>OOTDiffusion+AM</td><td>23.3938</td></tr><tr><td>IDM-VTON+AM</td><td>22.5881</td></tr><tr><td>ViViD</td><td>21.8032</td></tr><tr><td>CatV2TON</td><td>19.5131</td></tr><tr><td>Vidu S2-Editing</td><td>9.9515</td></tr></table>

![](images/d37894984e28ac450e8a09c9edd92bbbf0d83c3f886e6da7f737b3741b4167c1.jpg)  
(a) consistency

![](images/bb57d590e546e0eed783dd2addd5f3dd83002ae59db6898a9d58eedb7900ce39.jpg)  
(b) video quality

![](images/44996db60599286a91810da9f49f8439448ae82e60ca392a4e04be6aa2cdaa3e.jpg)  
(c) motion quality

![](images/e8f1776f0c24c9a5690f1998ed8b55b3720a5dd5ba17ad4425ccd55c3fd65b02.jpg)  
(d) emotional expression

![](images/441da6bf2c904c385a53b60fb0501545648a9a79837bad8cb8a037b969242d1e.jpg)  
(e) overall quality  
Figure 5 Duration-stratified mean ratings for digital character generation. Absent tail points indicate unavailable ratings.

ViViD Test Set [42]. As shown in Table 4, Vidu S2-Editing achieves a VFID of 9.9515 on the unpaired virtual try-on benchmark, outperforming CatV<sup>2</sup>TON (19.5131) and ViViD (21.8032) and indicating closer alignment between the generated and reference video distributions.

## 5.3 Complementary Human Preference Evaluation

Public benchmarks enable reproducible model comparisons under standardized settings. However, many existing benchmarks rely heavily on automated or model-based evaluators, which may not fully capture perceptual artifacts that emerge over long video sequences, such as identity drift, temporal inconsistency, and accumulated generation errors. To complement these evaluations and provide a more comprehensive assessment, we conduct randomized paired human-preference comparisons using the GSB protocol described in section 5.1.3. We report the results separately for digital-character generation and video editing.

Digital-Character Generation. Commercial digital character systems are evaluated on our internal bench mark with 160, 320, and 480 ms audio chunks, interruption and recovery tests, five-minute stability tests, end-to-end latency, and cost.

Figure 5 examines how perceived quality evolves as generation extends from 10 to 90 seconds. This durationstratified view complements clip-level GSB judgments because a streaming model can produce strong short clips yet accumulate identity drift, motion discontinuities, or loss of expressiveness over time. Following the emphasis on long-horizon audio–visual stability in StreamAV-Bench and the separation of visual quality from temporal consistency in VBench [38, 73], we report mean ratings for overall quality, consistency, video quality, motion quality, and emotional expression on a common 1–5 scale.

![](images/9731e1ae4eb5176100b431dbdad057fd604b8479be91adc484d355d08d3a793f.jpg)

Figure 6 GSB human-preference comparison for streaming digital character generation. The figure compares Vidu S2-Avatar with Runway Character GWM-1, PixVerse Image Avatar, and HeyGen across overall quality, motion quality, expression quality, video quality, consistency, audio–video synchronization, and semantic adherence. Percentages partition the available paired judgments into Vidu S2-Avatar preferred, Same, and the other system preferred; unavailable outcome categories are omitted from the corresponding bar.  
![](images/0634172423b81acf37bd9ac487b1f4f37e9fa141a0e383035aeaa2817c022969.jpg)  
Figure 7 GSB human-preference comparison between Vidu S2-Editing and two commercial systems on the internal editing benchmark. Each bar partitions paired judgments into Vidu S2-Editing preferred, Same, and the other system preferred for four criteria.

Vidu S2-Avatar consistently achieves the highest ratings across all five dimensions over the evaluated duration range. Its consistency remains high up to 90 seconds, suggesting that subject identity and temporal structure are preserved during longer streams. The video- and motion-quality curves remain similarly strong as the duration increases, indicating sustained perceptual fidelity and coherent motion. Emotional-expression scores also remain high at longer durations, showing that Vidu S2-Avatar maintains expressive facial and body motion while preserving long-horizon stability. Taken together, these curves indicate a broad advantage in long-horizon generation, spanning visual fidelity, motion coherence, identity preservation, and expressive behavior, rather than a gain limited to short clips. These duration-stratified ratings complement the paired GSB comparisons by characterizing how generation quality evolves with stream duration.

Figure 6 demonstrates a consistent advantage of Vidu S2-Avatar over three closed-source systems across overall quality, motion, expression, semantic adherence, and temporal consistency. The strongest gains are observed in overall quality, where Vidu S2-Avatar is preferred in 85.7% of the comparisons against Runway Character GWM-1 and in 100% of the comparisons against both PixVerse Image Avatar and HeyGen. This advantage is also reflected in motion and expression quality: Vidu S2-Avatar receives unanimous preference over Runway and HeyGen for both dimensions, and achieves 100% preference for expression quality over PixVerse. For semantic adherence, Vidu S2-Avatar achieves preference rates ranging from 71.4% to 100%, indicating reliable instruction following across different systems. The video-quality comparisons further

Temporal consistency

![](images/627a3191e3ed057c6ab9b31a0ead923bf9fdb33339e8633aee1923a8890693a7.jpg)  
Good: Stable identity, hair, attire, and scene details.

![](images/3b9b79646e75bb41c7e8994d7ba1f33df0c27094196568e0f3a74c5261d4b12b.jpg)  
Bad: Facial identity and body proportions drift.

Temporal consistency  
![](images/808b1e191c879d69d8380616a30d272aeb6eaeba6ba27cf5665ea1ddf9413a89.jpg)  
Good: Stable identity, hair, eye color, and attire.

![](images/f2bd2bee7686cbb49dbb73a518654f316bfe6f6b19ff092d3ba7069a542df955.jpg)  
Bad: Face, framing, hairstyle, and attire drift.

![](images/2f5b147b26af96d3e7e12ad00428e2f6ebad93f11bf952780be5b15d72697908.jpg)  
Good: Stable product geometry and hand structure.

![](images/4b9d59729fd3f456b8b43a06b425bbf18feeb113dd961c48c88d7b2eca73ed6a.jpg)  
Bad: Hair and fingers are distorted.

![](images/7cc4a385446cae54bbc8ab20e3ad23d7732f9fd3adb682b859e6024d871feadf.jpg)  
Good: Stable eyes and facial structure.

![](images/76a82804394430709385e91c7cee7e37b264372c6c7c1b5ca4129f1ad8b6127d.jpg)  
Bad: Distorted eyebrows.  
(b) Male-character case with facial and attire consistency.

(a) Female-character case with cosmetics and hand interaction.

Figure 8 Qualitative comparisons for streaming digital character generation. Vidu S2-Avatar preserves identity, appearance attributes, fine-grained geometry, and temporal structure, while the closed-source baselines exhibit facial drift, body-proportion changes, hairstyle or attire changes, and local distortions in hair, eyebrows, fingers, or hand structure.

show a consistent preference for Vidu S2-Avatar, with preference rates between 57.1% and 71.4%. The temporal-consistency results are particularly favorable, including unanimous preference over PixVerse and a majority preference over HeyGen; in the comparison with Runway, Vidu S2-Avatar is preferred or judged perceptually equivalent in 85.7% of the cases. For audio–visual synchronization, Vidu S2-Avatar is either preferred or judged perceptually equivalent in all comparisons against PixVerse and HeyGen, and is directly preferred in 42.9% of the comparisons against Runway. Overall, these results show that Vidu S2-Avatar provides broad and consistent improvements in perceptual quality, motion fidelity, expression, semantic adherence, and long-horizon stability while maintaining strong audio–visual synchronization.

Video Editing. The internal editing benchmark comprises 150 paired cases. All systems are evaluated using the same source videos, reference images, editing prompts, and temporal sampling protocol. Figure 7 reports the aggregate GSB outcomes for overall quality, video quality, temporal consistency, and semantic adherence. The corresponding mean consistency scores are 3.56 for Vidu S2-Editing, 2.59 for Decart-Lucy2.5, and 1.67 for XMax-X2.0. Across the reported criteria, Vidu S2-Editing receives a higher preference share than both commercial baselines, with the largest observed advantages in overall quality and temporal consistency.

## 5.4 Qualitative Case Studies

Vidu S2-Avatar. The two qualitative comparisons in Figure 8 examine temporal consistency and visual stability under fixed identity, scene, and motion conditions. In the female-character example, Vidu S2-Avatar preserves the facial identity, hairstyle, facial proportions, product geometry, and hand structure across the sampled frames. PixVerse Image Avatar exhibits visible facial-identity and body-proportion drift, while Runway Character GWM-1 introduces distortions in the hair and fingers. The male-character example leads to the same conclusion: Vidu S2-Avatar maintains the identity, hairstyle, eye color, attire, and facial structure, whereas PixVerse changes the framing, hairstyle, and overall appearance and Runway distorts the eyebrows. These cases indicate that Vidu S2-Avatar preserves both semantic identity and fine-grained geometry over time, including small objects and articulated hand or facial details that are particularly sensitive to temporal inconsistency. The qualitative evidence get the same results.

![](images/a0cecb64be9cb160a0a63e1bce07c664cc9485abfb917208c5cf15521de5aa15.jpg)  
(a) Watercolor style transfer.

![](images/fe4f88f2a59f610a1ef448642de13dd6896486dbc368a9d9809185e7710c182e.jpg)  
(b) Gongbi style transfer.

Figure 9 Style-transfer cases. Vidu S2-Editing changes the brushwork and palette while preserving facial layout, subject silhouette, pose, and temporal texture attachment across the stream.  
![](images/41e1a0c0165620822ab56c7b888a8e5919859a9720bd424dfeb025a79cb5889e.jpg)  
(a) White-shirt virtual try-on.

![](images/4a2f6442f88a1bddc5d8a65e07e26a6f95c721250bf56dea2fa041e78002617c.jpg)  
(b) Denim virtual try-on.  
Figure 10 Virtual try-on cases. Vidu S2-Editing transfers the target garments while preserving body motion, garment boundaries, material texture, and hand–cloth occlusion relationships.

Vidu S2-Editing. We compare four editing tasks using the cases in Figures 9–12: style transfer, virtual try-on, subject replacement, and background replacement. (1) Style Transfer. Figure 9 compares watercolor and gongbi stylization. In the watercolor case, Vidu S2-Editing applies the reference palette and rendering style while retaining the source subject’s facial appearance and interaction with the cup. XMax-X2.0 changes the subject’s identity but leaves the cup with a photographic appearance, failing to stylize the scene consistently. Decart-Lucy2.5 also changes the source facial identity. In the gongbi case, XMax-X2.0 fails to stylize the background, while Decart-Lucy2.5 produces severe blur and ghosting around the subject. Vidu S2-Editing applies the reference style to both the subject and background while retaining clearer subject contours in the displayed frames. (2) Virtual Try-On. Figure 10 compares transfer of a white shirt and a denim jumpsuit. In the white-shirt case, Vidu S2-Editing reproduces the reference garment and floral detail while retaining the source hand–garment interaction. XMax-X2.0 introduces spurious text on the chest. Decart-Lucy2.5 places the floral pattern on the shoulder and fails to preserve the shirt deformation during the pulling action highlighted in the figure. In the denim case, XMax-X2.0 inserts an extra person wearing denim while leaving the original subject in the source outfit, failing to replace the intended subject’s clothing. Decart-Lucy2.5 generates a denim jacket over the source shirt instead of the reference jumpsuit. Vidu S2-Editing transfers the jumpsuit across the displayed poses without introducing an additional person. (3) Subject Replacement. Figure 11 compares replacement with a reference woman and a cartoon character wearing an orange kimono.

![](images/54e5042613bff3dd2c5d8f91ad9c25d0bcce3d2aae59431c51bc532eac2d9c41.jpg)  
(a) Character subject replacement.

![](images/a1ff1b99289a8bcced6154989fe62f480d1bb31d0bfa102f16490091d62c1d03.jpg)  
(b) Kimono subject replacement.

Figure 11 Subject-replacement cases. Vidu S2-Editing transfers the reference identity while preserving the source pose, camera trajectory, scene geometry, and coherent boundaries around hair and limbs.  
![](images/cb63749518a422a24bcbc7de5687dcfb82ef10e59a378d4f052edbb109c94962.jpg)  
(a) Paris background replacement.

![](images/d9280be03a6da6216abe62c67f1492848120e4af1f2fb4e1557bbb89c292c3a2.jpg)  
(b) Bedroom background replacement.  
Figure 12 Background-replacement cases. Vidu S2-Editing changes the target environment while preserving foreground pose, camera motion, scene geometry, and closed boundaries around fine contours over time.

In the human case, Vidu S2-Editing transfers the reference appearance while retaining the source poses and shirt-pulling interaction. XMax-X2.0 fails to replace the source subject in the highlighted frame and subsequently changes the clothing without consistently transferring the reference identity. Decart-Lucy2.5 changes the subject’s appearance but fails to reproduce the source shirt-pulling interaction in the highlighted second frame. In the cartoon case, XMax-X2.0 changes the head while retaining the source blue outfit, and Decart-Lucy2.5 generates an incorrect outfit that combines blue source clothing with orange kimono elements. Vidu S2-Editing transfers both the character appearance and the reference kimono across the displayed frames. (4) Background Replacement. Figure 12 compares replacement with a Paris cafe scene and a bedroom. Vidu S2-Editing reproduces the reference environments while retaining the foreground subject and source actions. In the Paris case, XMax-X2.0 introduces an extra person and retains the original setting instead of replacing it with the reference scene. Decart-Lucy2.5 generates a different cafe layout whose architecture and awnings do not match the reference. In the bedroom case, XMax-X2.0 leaves the original room unchanged, failing to replace the background. Decart-Lucy2.5 generates a bedroom, but its window, bed, and lighting arrangement differ from the reference. These cases show failures in both background replacement, whereas Vidu S2-Editing closely reproduces the requested scene layout in the displayed frames.

Vidu S2 Spatial Video Generation. Figure 13 presents representative spatial-video results produced by Vidu S2. Across the examples, foreground subjects are clearly separated from the background, creating coherent scene depth and a stronger sense of immersion.

![](images/8b387efb24713cc18dd68bbc066708e692baf707e191565ffbfb5ece23ee33f4.jpg)  
Figure 13 Qualitative results of Vidu S2 spatial video generation. Each example shows the generated left- and right-eye views.

## 6 Conclusion

We present Vidu S2, which comprises Vidu S2-Avatar, a real-time interactive digital-character model, and Vidu S2-Editing, a real-time video editing model, and we further explore the feasibility of real-time spatial video generation. Compared with Vidu S1, Vidu S2-Avatar supports real-time 720p video generation, dynamic references that can be updated at any moment, and stronger instruction following, such as dancing. Vidu S2-Editing edits a video stream in real time, covering style transfer, virtual try-on, character replacement, and background replacement. Experiments show that Vidu S2 outperforms all baselines.

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.

[2] Google DeepMind. Veo: A text-to-video generation system. Technical report, Google DeepMind, 2025. Veo 3 Tech Report.

[3] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[4] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[5] Jintao Zhang, Kai Jiang, Jintao Chen, Xu Wang, Yang Luo, Yuji Wang, Dechuang Chen, Jungang Li, Chengyang Ye, Marco Chen, et al. Vidu s1: A real-time interactive video generation model. arXiv preprint arXiv:2607.03118, 2026.

[6] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.

[7] Youhe Jiang, Haoxu Wang, Haotong Bao, Kai Jiang, Jianfei Chen, Jun Zhu, Fangcheng Fu, and Jintao Zhang. Turboserve: Serving streaming video generation efficiently and economically. arXiv preprint arXiv:2606.19271, 2026.

[8] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026.

[9] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

[10] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

[11] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

[12] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

[13] Ailing Zeng, Casper Yang, Chauncey Ge, Eddie Zhang, Garvey Xu, Gavin Lin, Gilbert Gu, Jeremy Pi, Leo Li, Mingyi Shi, et al. Lpm 1.0: Video-based character performance model. arXiv preprint arXiv:2604.07823, 2026.

[14] Jintao Zhang, Kai Jiang, Jintao Chen, Xu Wang, Yang Luo, Yuji Wang, Dechuang Chen, Jungang Li, Chengyang Ye, Marco Chen, Hongzhou Zhu, Min Zhao, Yuxuan Jiang, Zhengkun Huang, Chendong Xiang, Kaiwen Zheng, Haoxu Wang, Xiaohang Wang, Qi Jia, Xin Chen, Yimin Chen, Youhe Jiang, Fangcheng Fu, Zhijie Deng, Fan Bao, Jianfei Chen, and Jun Zhu. Vidu S1: A real-time interactive video generation model, 2026.

[15] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238. IEEE, 2024.

[16] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pages 134129–134150, 2026.

[17] Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Haoxu Wang, Pengle Zhang, Huiqiang Jiang, Haofeng Huang, Chendong Xiang, et al. Efficient attention methods: Hardware-efficient, sparse, compact, and linear attention.

[18] Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations (ICLR), 2025.

[19] Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. In International Conference on Machine Learning (ICML), 2025.

[20] Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, and Jianfei Chen. Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025.

[21] Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025.

[22] Jintao Zhang, Marco Chen, Haoxu Wang, Kai Jiang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Sagebwd: A trainable low-bit attention. arXiv preprint arXiv:2603.02170, 2026.

[23] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025.

[24] Jintao Zhang, Kai Jiang, Chendong Xiang, Weiqi Feng, Yuezhou Hu, Haocheng Xi, Jianfei Chen, and Jun Zhu. Spargeattention2: Trainable sparse attention via hybrid top-k+ top-p masking and distillation fine-tuning. arXiv preprint arXiv:2602.13515, 2026.

[25] Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jun Zhu, and Jianfei Chen. Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention. arXiv preprint arXiv:2509.24006, 2025.

[26] Jintao Zhang, Haoxu Wang, Kai Jiang, Kaiwen Zheng, Youhe Jiang, Ion Stoica, Jianfei Chen, Jun Zhu, and Joseph E Gonzalez. Sla2: Sparse-linear attention with learnable routing and qat. arXiv preprint arXiv:2602.12675, 2026.

[27] Pengle Zhang, Jia Wei, Jintao Zhang, Jun Zhu, and Jianfei Chen. Accurate int8 training through dynamic block-level fallback. arXiv preprint arXiv:2503.08040, 2025.

[28] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

[29] Yanrui Bin, Wenbo Hu, Haoyuan Wang, Xinya Chen, and Bing Wang. Normalcrafter: Learning temporally consistent normals from video diffusion priors. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8330–8339. IEEE, 2025

[30] Bernini Team, Chenchen Liu, Junyi Chen, Lei Li, Lu Chi, Mingzhen Sun, Zhuoying Li, Yi Fu, Ruoyu Guo, Yiheng Wu, et al. Bernini: Latent semantic planning for video diffusion. arXiv preprint arXiv:2605.22344, 2026.

[31] Wenhao Yan, Fengjia Guo, Zhuoyi Yang, and Jie Tang. Scail-2: Unifying controlled character animation with end-to-end in-context conditioning. arXiv preprint arXiv:2606.10804, 2026.

[32] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17191–17202. IEEE, 2025.

[33] Xinyao Zhang, Wenkai Dong, Yuxin Song, Bo Fang, Qi Zhang, Jing Wang, Fan Chen, Hui Zhang, Haocheng Feng, Yu Lu, et al. Sama: Factorized semantic anchoring and motion alignment for instruction-guided video editing. arXiv preprint arXiv:2603.19228, 2026.

[34] Fuchen Long, Cong Wang, Zitao Gao, Wenhao Zhong, Yu Cheng, Xiaolu Hou, Yan Li, Xiao Cao, Xinlong Sun, Xi Chen, et al. Coinve-200k: A large-scale high-quality dataset for compositional instruction-guided video editing. arXiv preprint arXiv:2608.17566, 2026.

[35] Liang Zhang, Carlos Vazquez, and Sebastian Knorr. 3d-tv content creation: automatic 2d-to-3d video conversion. IEEE Transactions on Broadcasting, 57(2):372–383, 2011.

[36] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos. arXiv preprint arXiv:2409.07447, 2024.

[37] Yuan Huang, Sijie Zhao, Jing Cheng, Hao Xu, and Shaohui Jiao. Dreamstereo: Towards real-time stereo inpainting for hd videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25393–25402, 2026.

[38] Kaiqi Liu, Haoxuan Zeng, Jingqi Liu, Jiacong Fang, Ziqi Cai, Yunyao Mao, Henglin Liu, Yu Sheng, Shuchen Weng, and Boxin Shi. StreamAV-Bench: A comprehensive benchmark for streaming audio-video generation, 2026.

[39] Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.

[40] Ziyun Zeng, Yiqi Lin, Guoqiang Liang, and Mike Zheng Shou. Sparkle: Realizing lively instruction-guided video background replacement via decoupled guidance. arXiv preprint arXiv:2605.06535, 2026.

[41] Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026.

[42] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and Zheng-Jun Zha. Vivid: Video virtual try-on using diffusion models. arXiv preprint arXiv:2405.11794, 2024.

[43] PixVerse. Pixverse launches r1: A real-time world model that redefines ai video generation. PixVerse technical release, 2026.

[44] Alibaba. Alibaba launches happyoyster, a world model product for real-time immersive creation and interaction. Official product release, 2026.

[45] Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Ying Li, Zezhong Qian, Haoyang Huang, and Nan Duan. Omniforcing: Unleashing real-time joint audio-visual generation, 2026.

[46] Odyssey. Introducing odyssey-2: A general-purpose world model. Official technical release, 2025.

[47] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

[48] Shanwen Tan, Hao Li, Jingtao Zhang, Xiaosong Jia, Xue Yang, Shaofeng Zhang, and Yanyong Zhang. Swift: Prompt-adaptive memory for efficient interactive long video generation, 2026.

[49] Jinzhuo Liu, Jiangning Zhang, Wencan Jiang, Yabiao Wang, Dingkang Liang, Zhucun Xue, Ran Yi, and Yong Liu. Advancing narrative long video generation via training-free identity-aware memory, 2026.

[50] Le Shen, Qian Qiao, Tan Yu, Ke Zhou, Tianhang Yu, Yu Zhan, Zhenjie Wang, Ming Tao, Shunshun Yin, and Siyuan Liu. SoulX-FlashTalk: Real-time infinite streaming of audio-driven avatars via self-correcting bidirectional distillation, 2025.

[51] Hengyuan Zhang, Jingna Sun, Meiguang Jin, and Junfeng Ma. AptAvatar: Fast and vivid long-form audio-driven video generation for production-ready avatars, 2026.

[52] Yubo Huang, Hailong Guo, Fangtai Wu, Weiqiang Wang, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, and Steven Hoi. Live avatar: Streaming real-time audio-driven avatar generation with infinite length, 2025.

[53] Liyuan Cui, Wentao Hu, Wenyuan Zhang, Zesong Yang, Fan Shi, and Xiaoqiang Liu. AvatarForcing: One-step streaming talking avatars via local-future sliding-window denoising, 2026.

[54] Chunyu Li, Jiaye Li, Ruiqiao Mei, Haoyuan Xia, Hao Zhu, Jingdong Wang, and Siyu Zhu. Hallo-Live: Real time streaming joint audio-video avatar generation with asynchronous dual-stream and human-centric preference distillation, 2026.

[55] HeyGen. Heygen ai video avatar. https://www.heygen.com/avatars/ai-video-avatar, 2026. Accessed: 2026- 07-02.

[56] Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. Vino: A unified visual generator with interleaved omnimodal context. arXiv preprint arXiv:2601.02358, 2026.

[57] Kaihang Pan, Qi Tian, Jianwei Zhang, Weijie Kong, Jiangfeng Xiong, Yanxin Long, Shixue Zhang, Haiyi Qiu, Tan Wang, Zheqi Lv, Yue Wu, Liefeng Bo, Siliang Tang, and Zhao Zhong. Omniweaving: Towards unified video generation with free-form composition and reasoning. arXiv preprint arXiv:2603.24458, 2026.

[58] Xinyu Wang, Chongbo Zhao, Fangneng Zhan, and Yue Ma. Liveedit: Towards real-time diffusion-based streaming video editing. arXiv preprint arXiv:2606.26740, 2026.

[59] Yicheng Xiao, Wenxun Dai, Xinran Qin, Lin Song, Maoquan Zhang, Hang Xu, Yukang Chen, Yitong Li, Guohui Zhang, Yuan Zhang, Xuying Zhang, Tommy Zhang, Jianlong Yuan, Peihao Li, Shuai Lu, Siming Fu, Chuyang Zhao, Xin Han, Jie Huang, Wenbo Li, Guoqing Ma, Wei Huang, Xiaojuan Qi, Haoyang Huang, and Nan Duan. Joyai-video-edit: Real-time open-ended video editing with autoregressive diffusion. arXiv preprint arXiv:2608.03974, 2026.

[60] Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, et al. Streamdiffusionv2: A streaming system for dynamic and interactive video generation. arXiv preprint arXiv:2511.07399, 2025.

[61] Xmax AI. X2.0: Real-time interactive AI video model. https://xmax.ai/. Accessed: 2026-09-10.

[62] Decart AI. Lucy 2.5. https://lucy.decart.ai/. Accessed: 2026-09-10.

[63] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on, 2023.

[64] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on, 2024.

[65] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild, 2024.

[66] Chong Zheng, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei Jiang, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models, 2024.

[67] LAION-AI. Aesthetic predictor. https://github.com/LAION-AI/aesthetic-predictor, 2022.

[68] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, and Wei-Ning Hsu. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound, 2025.

[69] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all, 2023.

[70] Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues, 2024.

[71] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric and challenges, 2018.

[72] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017.

[73] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023.