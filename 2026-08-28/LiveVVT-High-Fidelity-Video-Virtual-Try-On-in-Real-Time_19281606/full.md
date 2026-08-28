# LiveVVT: High-Fidelity Video Virtual Try-On in Real Time

Yushe Cao<sup>1</sup>, Shikun Feng<sup>2</sup>, Ruxiang Duan<sup>3</sup>, Liyong Wang<sup>4</sup>, Dianxi Shi<sup>∗1</sup>, Chun Yu<sup>1</sup>, Junliang Xing<sup>∗1</sup>,

<sup>1</sup>Tsinghua University <sup>2</sup>Zhongguancun Academy <sup>3</sup>South China University of Technology <sup>4</sup>Beijing Jiaotong University cao-ys23@mails.tsinghua.edu.cn,kunayumi@163.com

## Abstract

Difusion-based Video Virtual Try-On (VVT) achieves high visual fidelity through bidirectional spatio-temporal modeling, but complete-clip dependence incurs prohibitive latency and computational overhead in practical continuous deployment. Naively enforcing causality disrupts pretrained bidirectional priors and substantially degrades synthesis quality. We introduce LiveVVT, a rolling streaming difusion framework that preserves bounded bidirectional modeling within causal recurrent generation. Within a fixed-size window, LiveVVT jointly denoises multiple video chunks under bounded look-ahead, preserving local bidirectional interactions while emitting one clean chunk per iteration. Beyond the window, two complementary memories sustain long-term consistency: a bounded temporal memory propagates recent dynamics and occlusion context, whereas a persistent global appearance memory, constructed once from the target garment and a frontal try-on keyframe, anchors garment details and dressed appearance throughout the stream. We further introduce a progressive distillation framework integrating bidirectional VVT learning, teacher-trajectory regression for causal few-step adaptation, and Collaborative Matching Distillation, which couples teacher-distribution matching with rolling flow matching on real videos to align optimization with recurrent inference. Experiments on paired and unpaired long-sequence benchmarks demonstrate superior generation quality over similarly sized models, with 26× lower latency and 11× higher throughput, enabling high-fidelity real-time streaming VVT.

## 1 Introduction

Video Virtual Try-On (VVT) (Li et al. 2025; Zeng et al. 2025; Chong et al. 2025b) transfers a target garment onto a person throughout a video while preserving garment appearance, human identity, and temporal coherence. Unlike image-based try-on (Chong et al. 2025a; Zhang et al. 2026; Li et al. 2026), VVT must resolve pose variation, occlusion, and nonrigid garment deformation consistently across frames (Zhong et al. 2021; Jiang et al. 2022; Zou et al. 2025). Its applications in online retail, digital humans, live-stream commerce, and augmented reality make high-fidelity real-time streaming a critical capability.

Recent difusion-based methods have substantially advanced VVT fidelity by integrating multimodal conditioning with spatio-temporal attention (Zou et al. 2025; Chong et al. 2025b; Zuo et al. 2025). However, the dominant paradigm remains clip-based and bidirectional: it depends on future frames and partitions long videos into independently processed windows, resulting in high latency and redundant computation. Although recent work explores interactive customization with arbitrary garments (Song et al. 2026; Zheng et al. 2026), continuous, source-conditioned streaming VVT that jointly delivers high fidelity and low latency remains largely unexplored.

![](images/172a8cdb4899ce47065670a0fa6b5f704529fca05fa2ca8b19b79702d7211a8b.jpg)  
Figure 1: Comparison of latency and throughput between LiveVVT and state-of-the-art methods.

High-fidelity streaming VVT cannot be achieved by simply imposing causal attention on an ofline model. First, this conversion disrupts pretrained bidirectional spatio-temporal priors, leading to severe degradation in synthesis quality. Second, long-term consistency requires suficient historical context; however, unbounded history is computationally impractical, whereas aggressive truncation induces appearance drift. Third, the transition from ofline denoising to recurrent generation changes both attention patterns and sampling trajectories, while conditioning on self-generated history further introduces exposure bias and error accumulation (Yin et al. 2025a; Huang et al. 2025; Cui et al. 2026; Liu et al. 2026). Practical streaming VVT therefore requires the generation mechanism, memory architecture, and capability transfer strategy to be redesigned jointly.

To address these challenges, we introduce LiveVVT, a rolling streaming difusion framework that combines local bidirectional denoising with causal recurrence across windows. Within a fixed-size window, LiveVVT jointly denoises chunks at staggered noise levels under bounded lookahead and emits one clean chunk per iteration. Beyond the window, two complementary memories sustain long-term consistency: a bounded temporal memory propagates recent dynamics and occlusion context, while a persistent global appearance memory anchors garment details and overall dressed appearance throughout the stream. To transfer high-fidelity ofline priors to recurrent few-step generation, progressive distillation integrates bidirectional VVT learning, teacher-trajectory regression, and Collaborative Matching Distillation (CoMD). CoMD couples teacher-distribution matching with rolling flow matching on real videos under recurrent inference, narrowing the gap between ofline supervision and streaming generation. Experiments on paired and unpaired long-sequence benchmarks show that LiveVVT outperforms similarly sized models with 26× lower latency and 11× higher throughput.

Our main contributions are summarized as follows:

• We introduce LiveVVT, a rolling generation paradigm that combines bidirectional joint denoising within a staggered-noise window with causal recurrence across windows, enabling high-fidelity real-time VVT under bounded look-ahead.

• We design two complementary memories: a bounded temporal memory propagates dynamics and occlusion context, while a persistent global appearance memory preserves garment texture and dressed appearance over long streams without repeated reference encoding.

• We develop a progressive distillation framework that transfers high-fidelity ofline VVT priors to a few-step streaming generator; its CoMD stage couples teacher distribution matching with rolling flow matching on real videos to align training with recurrent inference.

## 2 Related Work

## 2.1 Image and Video Virtual Try-On

Image virtual try-on has evolved from explicit garment warping and composition (Han et al. 2018, 2019; Yang et al. 2020; Ge et al. 2021) toward a conditional difusion paradigm that improves realism and fine-grained detail preservation (Zhu et al. 2023; Choi et al. 2024; Xu et al. 2025; Cao et al. 2026b). Extending this task to video, early methods maintain temporal coherence through optical-flow-based propagation and explicit inter-frame correspondence (Dong et al. 2019; Zhong et al. 2021; Jiang et al. 2022), while modern generative architectures jointly denoise clips using temporal attention and multimodal conditioning (Zou et al. 2025; Chong et al. 2025b; Zuo et al. 2025; Li et al. 2025; Cao et al. 2026a). More recent eforts explicitly model human–garment interactions (Zheng et al. 2026) or support customization with arbitrary garments (Song et al. 2026) under complex motion and occlusion. Nevertheless, most high-fidelity VVT models remain clip-based and bidirectional, requiring complete input clips and redundant windowed inference for long videos. In contrast, LiveVVT enables source-conditioned streaming try-on through bounded rolling denoising and two complementary memory mechanisms: a recurrent temporal memory and a persistent global appearance memory, thereby achieving low-latency, high-throughput real-time streaming inference.

## 2.2 Autoregressive Video Generation

Autoregressive video generation has evolved from discrete visual-token prediction (Yan et al. 2021; Villegas et al. 2023) to continuous latent-space modeling (Deng et al. 2025). Recent studies transform bidirectional difusion models into few-step causal generators through trajectory- or distributionlevel distillation, self-generated rollouts, and adversarial post-training (Yin et al. 2025a; Huang et al. 2025; Zhu et al. 2026; Zhao et al. 2026; Lin et al. 2025), thereby mitigating causal adaptation, exposure bias, and accumulated generation errors. Recurrent and rolling architectures further extend this paradigm to real-time, long-form, and interactively controlled synthesis (Kodaira et al. 2026; Cui et al. 2026; Yang et al. 2026; Liu et al. 2026; Shin et al. 2026). These approaches, however, primarily synthesize unconstrained content from text prompts or sparse controls. Streaming VVT imposes substantially stricter conditioning requirements: every output chunk must remain precisely aligned with the incoming person stream while consistently preserving identity, garment fidelity, and long-term temporal coherence, even under challenging motion and occlusion. LiveVVT addresses this densely conditioned setting by coupling bounded-lookahead generation with persistent memory mechanisms and explicitly aligning few-step distillation with rolling inference on real video sequences at deployment scale.

## 3 Method

## 3.1 Problem Formulation and Overview

Let $\boldsymbol { \mathcal { X } } = \{ x _ { t } \} _ { t = 1 } ^ { T }$ denote the source-person stream and $I _ { g }$ the target garment image. Each frame yields the try-on condition

$$
c _ { t } = ( m _ { t } , p _ { t } , \bar { x } _ { t } ) ,\tag{1}
$$

where $m _ { t } , \ p _ { t }$ , and $\hat { x } _ { t }$ are the inpainting mask, Dense-Pose map, and garment-agnostic person image. We partition the video and conditions into $\bar { K }$ non-overlapping chunks $\{ X _ { k } \} _ { k = 1 } ^ { K }$ and $\{ C _ { k } \} _ { k = 1 } ^ { K }$ . The garment condition is denoted by $\mathbf { f } _ { g } ;$ t indexes frames, k indexes chunks, and $\tau \in [ 0 , 1 ]$ is reserved for normalized difusion time.

LiveVVT introduces a frontal source keyframe $x _ { f }$ and its condition $C _ { f }$ to construct a persistent appearance anchor; $x _ { f }$ is captured in a standard A-Pose at deployment. For training videos without such a reference, we select the frame whose pose is closest to the A-Pose template $p _ { \mathrm { a } } \colon$

$$
x _ { f } = x _ { \mathrm { a r g m i n } _ { t \in \{ 1 , . . . , T \} } } d ( p _ { t } , p _ { \mathrm { a } } ) ,\tag{2}
$$

where $d ( \cdot , \cdot )$ is the pose-alignment distance.

At recurrent update k, the streaming generator produces one completed try-on chunk:

$$
Y _ { k } = \mathcal { G } _ { s } ( C _ { k : k + N - 1 } , \mathcal { H } _ { k } , \mathcal { A } ) ,\tag{3}
$$

where N is the window size, $\mathcal { H } _ { k }$ a bounded temporal mem-$\mathrm { o r y , }$ and A a persistent global appearance memory constructed from the garment and frontal reference. Figure 2 illustrates the LiveVVT framework. Under bounded lookahead, a fixed-size rolling window preserves bidirectional

(a) Rolling Streaming Try-on Inference  
![](images/7d3e2fee7044368693ea2262abbd7c69c248afa75e9ca1d946f1e570a5f2402f.jpg)

(b) Progressive Distillation  
![](images/127d7c2e220d4369135f502153a0c7ad7a95b5ece215772e574f992a7d786a28.jpg)  
Figure 2: Overview of LiveVVT. (a) Rolling streaming try-on emits one clean chunk per update from a staggered-noise window with persistent global appearance memory $\pmb { \mathcal { A } } = ( \mathcal { A } _ { g } , \mathcal { A } _ { f } )$ and temporal memory $\mathcal { H } _ { k }$ . (b) Bidirectional VVT learning, teachertrajectory regression, and CoMD progressively transfer ofline bidirectional priors to few-step streaming generation.

active window is

interactions and emits one fully denoised chunk per update. Across windows, temporal memory recurrently propagates recent motion dynamics and occlusion context, while global appearance memory persistently anchors garment details and overall dressed appearance throughout the stream. To bridge the mismatch in attention patterns and sampling trajectories between bidirectional generation and causal recurrence, we further introduce a progressive distillation framework that progressively transfers the high-fidelity generative prior of an ofline bidirectional model to a causal few-step generator.

## 3.2 Rolling Streaming Try-On

Ofline bidirectional difusion jointly denoises a clip at a shared noise level, requiring both the full input and the complete sampling trajectory before emission. LiveVVT instead organizes denoising as a rolling pipeline. At each update, resident chunks advance by one stage and a fresh Gaussian-noise chunk enters the window; chunks of different ages therefore occupy staggered noise levels. Let $0 = \sigma _ { 0 } < \sigma _ { 1 } < \cdot \cdot \cdot < \sigma _ { N } = 1$ denote an N-step schedule from clean data to Gaussian noise. Before update $k ,$ the

$$
\mathcal { Z } _ { k } = ( z _ { k } ^ { \sigma _ { 1 } } , z _ { k + 1 } ^ { \sigma _ { 2 } } , \ldots , z _ { k + N - 1 } ^ { \sigma _ { N } } ) .\tag{4}
$$

The leading chunk is one step from clean, whereas the newest remains pure noise. Given ${ \pmb \sigma } = ( \sigma _ { 1 } , \dots , \sigma _ { N } )$ , each streaming update jointly advances all chunks from $\sigma _ { i }$ to $\sigma _ { i - 1 } :$

$$
\bar { \mathcal { Z } } _ { k } = \operatorname { S t e p } ( \mathcal { Z } _ { k } , \mathcal { G } _ { s } ( \mathcal { Z } _ { k } , \pmb { \sigma } , C _ { k : k + N - 1 } , \mathcal { H } _ { k } , \mathcal { A } , \mathbf { f } _ { g } ) ) ,\tag{5}
$$

producing

$$
\bar { \mathcal { Z } } _ { k } = \big ( z _ { k } ^ { \sigma _ { 0 } } , z _ { k + 1 } ^ { \sigma _ { 1 } } , \ldots , z _ { k + N - 1 } ^ { \sigma _ { N - 1 } } \big ) .\tag{6}
$$

Only the leading chunk reaches $\sigma _ { 0 } ,$ so each update completes one chunk while refining the rest. Despite asynchronous noise leve $^ { 1 \mathrm { s } , }$ active chunks interact through bidirectional attention, allowing the leading chunk to exploit partially denoised future context for deformation and occlusion reasoning. Beyond the active window, $\mathcal { H } _ { k }$ propagates generated context, while A provides persistent appearance cues; no future frames are accessed across updates. LiveVVT combines bounded-look-ahead bidirectional denoising within each window with causal recurrence across windows. The completed latent is decoded by the frozen VAE decoder D:

$$
Y _ { k } = \mathcal { D } ( z _ { k } ^ { \sigma _ { 0 } } ) .\tag{7}
$$

LiveVVT then removes the emitted latent, shifts the remaining chunks, and appends fresh noise:

$$
\mathcal { Z } _ { k + 1 } = ( z _ { k + 1 } ^ { \sigma _ { 1 } } , \ldots , z _ { k + N - 1 } ^ { \sigma _ { N - 1 } } , z _ { k + N } ^ { \sigma _ { N } } ) ,\tag{8}
$$

where $z _ { k + N } ^ { \sigma _ { N } } \sim \mathcal { N } ( 0 , I )$ . This roll–append operation restores Eq. 4 for the next update: every chunk enters at $\sigma _ { N }$ , undergoes exactly N joint updates, and exits at $\sigma _ { 0 } .$ . Confining joint denoising to a fixed window makes per-update computation and activation memory independent of stream length.

## 3.3 Dual-Memory Mechanism

The bounded window cannot retain latent context after a chunk rolls out. This truncation induces two inconsistencies: motion and occlusion states may become discontinuous across adjacent windows, while garment details and dressed appearance may drift over longer horizons. LiveVVT addresses these failure modes with memories at two temporal scales: an evolving temporal memory for recent crosswindow dynamics and a persistent global appearance memory for stable sequence-level anchoring.

Temporal memory. To maintain cross-window continuity, LiveVVT caches attention key–value (KV) features from each completed clean latent:

$$
\mathcal { H } _ { k + 1 } = \mathrm { F I F O } _ { L } \big ( \mathcal { H } _ { k } \cup \Phi _ { \mathrm { K V } } \big ( z _ { k } ^ { \sigma _ { 0 } } \big ) \big ) ,\tag{9}
$$

where $\mathrm { F I F O } _ { L }$ retains the latest L entries. Subsequent windows attend to the cached features, propagating recent motion states and occlusion relations without reprocessing emitted chunks. Because the cache is derived from generated clean latents, it matches the self-conditioned history encountered during recurrent inference. Its bounded capacity limits online cost but evicts older states; $\mathcal { H } _ { k }$ is therefore suited to local dynamics rather than persistent appearance anchoring.

Global appearance memory. This limitation motivates a complementary reference for stable, person-specific dressed appearance over extended horizons. LiveVVT therefore constructs a persistent global appearance memory that anchors garment details and overall dressed appearance throughout the stream. We first extract garment KV features

$$
\begin{array} { r } { \mathcal { A } _ { g } = \Phi _ { \mathrm { K V } } ( I _ { g } ) , } \end{array}\tag{10}
$$

which preserve source texture and pattern but do not specify how the garment should appear on the target body. The frontal keyframe provides a canonical person-specific configuration for establishing dressed appearance before streaming. We combine its condition $C _ { f }$ with $A _ { g }$ to generate a frontal tryon reference

$$
y _ { f } = \mathrm { T r y O n } ( C _ { f } , \mathcal { A } _ { g } , \mathbf { f } _ { g } ; \mathcal { G } _ { s } ) ,\tag{11}
$$

and extract the corresponding KV features:

$$
\begin{array} { r } { A _ { f } = \Phi _ { \mathrm { K V } } ( y _ { f } ) . } \end{array}\tag{12}
$$

$A _ { g }$ and $\ b { A _ { f } }$ form the persistent global appearance memory

$$
\ A = \ A _ { g } \cup \ A _ { f } .\tag{13}
$$

Computed once before streaming, A is reused by every rolling window as a time-invariant appearance anchor. It complements the evolving temporal memory $\mathcal { H } _ { k } \colon \mathcal { H } _ { k }$ propagates recent motion dynamics and occlusion states, whereas A anchors garment details and overall dressed appearance, suppressing long-term drift. This dual-memory design extends context beyond the active window without retaining unbounded history or repeatedly encoding reference images.

## 3.4 Progressive Distillation Framework

Rolling inference alters the attention pattern and denoising trajectory of ofline bidirectional generation. Joint adaptation lacks stable causal initialization and efective supervision for history-conditioned student states. We therefore transfer capabilities progressively: bidirectional VVT learning establishes high-fidelity try-on; teacher-trajectory regression adapts the student to causal few-step denoising; and Collaborative Matching Distillation aligns recurrent outputs with teacher and real-video distributions. Each stage resolves a distinct mismatch while retaining acquired capabilities.

Stage I: Bidirectional VVT learning. We initialize G from Wan2.1-Fun-V1.1-1.3B-Control (Wan et al. 2025). VAE-encoded try-on conditions are concatenated channelwise with noisy video latents, while the garment latent is prepended temporally. CLIP visual (Radford et al. 2021) and UMT5-XXL (Chung et al. 2023) text embeddings

$$
\mathbf { f } _ { g } = \{ \mathbf { f } _ { g } ^ { \mathrm { v i s } } , \mathbf { f } _ { g } ^ { \mathrm { t x t } } \}
$$

provide garment semantics through cross-attention. We extend the DiT input projection for these conditions while retaining its bidirectional spatio-temporal Transformer.

For a clean latent $z _ { \mathrm { 0 } } ,$ Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , and $\tau \sim \mathcal { U } ( 0 , 1 )$ , Flow Matching (Lipman et al. 2022) defines the linear path and target velocity as

$$
z _ { \tau } = ( 1 - \tau ) z _ { 0 } + \tau \epsilon ,\tag{14}
$$

$$
u _ { \tau } = \epsilon - z _ { 0 } ,\tag{15}
$$

We optimize $\mathcal { G } _ { b }$ with

$$
\mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { b i } } = \mathbb { E } \Big [ \| \mathcal { G } _ { b } ( z _ { \tau } , \tau , C _ { 1 : K } , \mathbf { f } _ { g } ) - u _ { \tau } \| _ { 2 } ^ { 2 } \Big ] .\tag{16}
$$

This stage preserves the bidirectional video prior and provides a high-fidelity initialization for $\mathcal { G } _ { s }$ . Applying the objective to Wan2.1-Fun-V1.1-14B-Control (Wan et al. 2025) yields the teacher $\mathcal { G } _ { \mathrm { r e a l } }$ as the real-score model for Collaborative Matching Distillation.

Stage II: Causal adaptation. We initialize $\mathcal { G } _ { s }$ from $\mathcal { G } _ { b }$ to retain high-fidelity VVT while adapting it to causal fewstep sampling. Following (Yin et al. 2025b), we use $\mathcal { G } _ { \mathrm { r e a l } }$ and an ODE solver to precompute 6,000 teacher trajectories, each subsampled at the student’s N-step schedule. Given an intermediate state $z _ { \tau _ { j } } ^ { i }$ and endpoint $z _ { 0 } ^ { i }$ from trajectory i, teacher-trajectory regression predicts the endpoint:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T T R } } = \mathbb { E } _ { i , j } \left[ \left\| \mathcal { G } _ { s } ( z _ { \tau _ { j } } ^ { i } , \tau _ { j } , C ^ { i } ) - z _ { 0 } ^ { i } \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{17}
$$

This objective jointly adapts the student’s attention pattern and denoising trajectory to causal few-step inference, providing a stable initialization for distribution-level distillation.

<table><tr><td></td><td colspan="4">Paired</td><td colspan="2">Unpaired</td></tr><tr><td>Method</td><td> $\mathrm { \Delta V F I D } _ { I } \downarrow \mathrm { \mathrm { V F I D } } _ { R } \downarrow \mathrm { \mathrm { S S I M } } \uparrow \mathrm { L P I P S } \downarrow \mathrm { V F I D } _ { I } \downarrow \mathrm { V F I D } _ { R } \downarrow$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OOTDiff+AM</td><td>46.666</td><td>24.450</td><td>0.720</td><td>0.238</td><td>47.802</td><td>25.857</td></tr><tr><td>CatVTON+AM</td><td>45.754</td><td>20.707</td><td>0.739</td><td>0.231</td><td>46.125</td><td>20.728</td></tr><tr><td>VACE</td><td>18.086</td><td>0.418</td><td>0.762</td><td>0.149</td><td>27.519</td><td>0.871</td></tr><tr><td>ViViD</td><td>20.887</td><td>0.337</td><td>0.792</td><td>0.130</td><td>27.963</td><td>0.462</td></tr><tr><td>CatV2TON</td><td>20.694</td><td>0.232</td><td>0.815</td><td>0.132</td><td>28.343</td><td>0.356</td></tr><tr><td>MagicTryOn</td><td>16.148</td><td>0.228</td><td>0.835</td><td>0.091</td><td>24.137</td><td>0.287</td></tr><tr><td>Ours</td><td>13.194</td><td>0.090</td><td>0.838</td><td>0.099</td><td>23.608</td><td>0.213</td></tr></table>

Table 1: Quantitative comparison of LiveVVT with competing methods on the long-sequence ViViD-SL benchmark.
<table><tr><td></td><td colspan="4">Paired</td><td colspan="2">Unpaired</td></tr><tr><td>Method</td><td> $\mathrm { \Delta V F I D } _ { I } \downarrow \mathrm { \mathrm { V F I D } } _ { R } \downarrow \mathrm { \mathrm { S S I M } } \uparrow \mathrm { L P I P S } \downarrow \mathrm { V F I D } _ { I } \downarrow \mathrm { V F I D } _ { R } \downarrow$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OOTDiff+AM</td><td>33.225</td><td>2.311</td><td>0.756</td><td>0.246</td><td>39.139</td><td>2.172</td></tr><tr><td>CatVTON+AM</td><td>33.334</td><td>2.555</td><td>0.769</td><td>0.246</td><td>36.863</td><td>1.547</td></tr><tr><td>VACE</td><td>20.373</td><td>0.405</td><td>0.787</td><td>0.173</td><td>31.152</td><td>1.648</td></tr><tr><td>ViViD</td><td>24.244</td><td>0.611</td><td>0.785</td><td>0.145</td><td>28.356</td><td>1.037</td></tr><tr><td>CatV2TON</td><td>22.018</td><td>0.376</td><td>0.811</td><td>0.156</td><td>30.653</td><td>1.133</td></tr><tr><td>MagicTryOn</td><td>17.988</td><td>0.370</td><td>0.808</td><td>0.112</td><td>25.950</td><td>0.832</td></tr><tr><td>Ours</td><td>15.020</td><td>0.290</td><td>0.814</td><td>0.115</td><td>24.297</td><td>1.024</td></tr></table>

Table 2: Quantitative comparison of LiveVVT with competing methods on the long-sequence ViT-HDL benchmark.

Stage III: Collaborative matching distillation. Teachertrajectory regression adapts isolated sampling states but leaves the recurrent rollout distribution unconstrained. DMD (Yin et al. 2024) narrows this gap using the realscore model $\mathcal { G } _ { \mathrm { r e a l } }$ and auxiliary fake-score model $\mathcal { G } _ { \mathrm { f a k e } } .$ but remains teacher-driven and lacks direct supervision of realvideo states under rolling deployment. We therefore propose CoMD, coupling teacher-distribution matching with Rolling Flow Matching (RFM) on real videos:

$$
\mathcal { L } _ { \mathrm { C o M D } } = \mathcal { L } _ { \mathrm { D M D } } + \lambda \cdot \mathcal { L } _ { \mathrm { R F M } } .\tag{18}
$$

RFM samples a ground-truth window and builds its prefix cache $\mathcal { H } _ { k }$ and global appearance memory A in the rolling configuration. Instead of perturbing the window at a shared continuous time $\tau ,$ it assigns successive chunks the ordered vector ${ \pmb \sigma } = ( \sigma _ { 1 } , \ldots , \sigma _ { N } )$ from the student’s discrete schedule, reproducing inference-time staggered noise. Applying Eq. 16 then supervises the real-data velocity field under deployment-matched memory and noise states. During alternating optimization, $\mathcal { G } _ { \mathrm { f a k e } }$ learns the score of detached student rollouts, while $\mathcal { G } _ { s }$ combines the DMD distribution gradient with RFM supervision on real windows. DMD preserves the teacher’s generative prior, whereas RFM anchors recurrent outputs to the real-video distribution. Together, they close the teacher–student distribution gap and training– inference gap from self-generated history, mitigating recurrent exposure bias. Algorithm 1 summarizes the procedure.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We train on two image datasets (VITON-HD (Choi et al. 2021) and DressCode(Morelli et al. 2022))

Algorithm 1: Collaborative Matching Distillation   
Input: dataset D; student ${ \mathcal G } _ { s } ;$ real score $\mathcal { G } _ { \mathrm { r e a l } } ;$ schedule $\mathcal { T } _ { N } =$   
$\{ \bar { \sigma _ { i } } \} _ { i = 0 } ^ { N } ;$ weight λ; update interval $r > 1$   
Output: streaming try-on model $\mathcal { G } _ { s }$   
1: $\mathcal G _ { \mathrm { f a k e } } \gets \mathrm { C o p y } ( \mathcal G _ { s } ) ; s \gets 1$   
2: while not converged do   
3: Sample $( X _ { 1 : K } , C _ { 1 : K } , I _ { g } , C _ { f } , \mathbf { f } _ { g } ) \sim \mathcal { D }$   
4: $\mathcal { A }  \mathrm { B u i l d G l o b a l M e m o r y } ( I _ { g } , C _ { f } , \mathbf { f } _ { g } )$   
5: if s mod $r = 0$ then   
6: $\mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } } \gets \mathrm { R o l l i n g { S a m p l e } } ( \mathcal { G } _ { s } , C _ { 1 : K } , \mathcal { A } , \mathbf { f } _ { g } , \mathcal { T } _ { N } )$   
7: $\mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { - * - } }  \mathrm { D M D } ( \mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } } , \mathcal { G } _ { \mathrm { r e a l } } , \mathcal { G } _ { \mathrm { f a k e } } , C _ { 1 : K } , I _ { g } , \mathbf { f } _ { g } )$   
8: Sample a window $X _ { k : k + N - 1 } ;$ encoded as $\mathcal { Z } _ { k } ~ =$   
$\left( z _ { k } , \dots , z _ { k + N - 1 } \right)$ and build $\mathcal { H } _ { k }$   
9: Set ${ \pmb \sigma } = ( \sigma _ { 1 } , \ldots , \sigma _ { N } )$ and sample $\mathbf { \Sigma } _ { \mathrm { ~ s ~ r ~ } } \sim \mathcal { N } ( 0 , I )$   
10: $\mathcal { Z } _ { \pmb { \sigma } , k } \gets \{ ( 1 - \sigma _ { i } ) z _ { k + i - 1 } + \sigma _ { i } \epsilon _ { i } \} _ { i = 1 } ^ { N }$   
11: ${ \mathcal { L } } _ { \mathrm { R F M } } \ \gets \ \lVert { \mathcal { G } } _ { s } ( { \mathcal { \bar { Z } } } _ { \sigma , k } , \sigma , C _ { k : k + N - 1 } , { \mathcal { \bar { H } } } _ { k } , A , \mathbf { f } _ { g } ) \ -$   
$\left( \epsilon - \mathcal { Z } _ { k } \right) \rVert _ { 2 } ^ { 2 }$   
12: $\mathcal { L } _ { \mathrm { C o M D } }  \mathcal { L } _ { \mathrm { D M D } } + \lambda \cdot \mathcal { L } _ { \mathrm { R F M } }$   
13: $\mathcal { G } _ { s } \gets \mathrm { U p d a t e } ( \mathcal { G } _ { s } , \mathcal { L } _ { \mathrm { C o M D } } )$   
14: else   
15: $\mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } }  \mathrm { R o l l i n g S a m p l e } ( \mathcal { G } _ { s } , C _ { 1 : K } , \mathcal { A } , \mathbf { f } _ { g } , \mathcal { T } _ { N } )$   
16: $\mathcal { Z } _ { 1 : K } ^ { \dot { \sigma } _ { 0 } ^ { \ s } } \gets \mathrm { S t o p G r a d } ( \mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } } )$   
17: $\mathrm { S a m p l e } \tau \sim \bar { \mathcal { U } } ( 0 , 1 )$ and $\epsilon \sim \mathcal { N } ( 0 , I )$   
18: $\mathcal { Z } _ { 1 : K } ^ { \tau }  ( 1 - \tau ) \mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } } + \tau \epsilon$   
19: $\begin{array} { r l r } { \mathcal { L } _ { \mathrm { f a k e } } ^ { \mathrm { ~ } ^ {  ~ } } } & {  } & { \| \mathcal { G } _ { \mathrm { f a k e } } ( \mathcal { \bar { Z } } _ { 1 : K } ^ { \tau } , \tau , C _ { 1 : K } , I _ { g } , \mathbf { f } _ { g } ) \mathrm { ~ - ~ } ( \epsilon \mathrm { ~ - ~ } \epsilon } \end{array}$   
$\mathcal { Z } _ { 1 : K } ^ { \sigma _ { 0 } } ) \| _ { 2 } ^ { 2 }$   
20: $\mathcal { G } _ { \mathrm { f a k e } } ^ { \mathrm { ~ \tiny ~ \dots ~ } }  \mathrm { U p d a t e } ( \mathcal { G } _ { \mathrm { f a k e } } , \mathcal { L } _ { \mathrm { f a k e } } )$   
21: end if   
22: $s \gets s + 1$   
23: end while

and three video datasets (ViViD (Fang et al. 2024), ViT-HD (He et al. 2026), and TikTokDress (Nguyen et al. 2025)), yielding 60,039 paired image samples and 23,258 paired video samples after preprocessing and filtering. Existing VVT studies commonly evaluate performance on subsets of 64-frame frontal-view clips, which provide limited evidence of long-term consistency. We therefore construct ViViD-SL from the 60 longest videos in the independent ViViD-S test set and ViT-HDL from 60 longest videos in the ViT-HD test split. The resulting sequences span 194–420 frames; all frames are evaluated under both paired and unpaired settings.

Evaluation Metrics. We assess video-level quality using VFID with I3D (Carreira and Zisserman 2017) and ResNeXt (Xie et al. 2017) feature extractors, denoted as $\mathrm { V F I D } _ { I }$ and ${ \mathrm { V F I D } } _ { R } ,$ respectively. SSIM and LPIPS measure frame-level structural similarity and perceptual distance to the ground truth. Lower VFID and LPIPS and higher SSIM indicate better performance. We report all four metrics for paired evaluation; for unpaired evaluation, where frame-wise ground truth is unavailable, we report $\mathrm { V F I D } _ { I }$ and ${ \mathrm { V F I D } } _ { R }$

Implementation Details. We train LiveVVT on eight NVIDIA A100 (80 GB) GPUs using AdamW with a batch size of 8 and weight decay of 0.01. Bidirectional VVT learning runs for 30k steps at a learning rate of $\lfloor \times 1 0 ^ { - 5 }$ , followed by 20k steps of teacher-trajectory regression at $2 \times 1 0 ^ { - 6 }$

CoMD is performed for 20k steps, using learning rates of $1 . 5 \times 1 0 ^ { - 6 }$ and $4 \times 1 0 ^ { - 7 }$ for the student generator and auxiliary fake-score model, respectively. We use a rolling window of $N = 4$ chunks with three latent frames per chunk and set the RFM weight to $\lambda = 0 . 2$ . The temporal-memory cache size is set to L=21, following Self-Forcing (Huang et al. 2025). Training and inference use a resolution of $5 1 2 \times 3 8 4$ and evaluations use a fixed random seed of 42.

## 4.2 Quantitative Experiments

We compare LiveVVT against the two-stage image-tovideo pipelines<sup>1</sup> OOTDif+AM (Xu et al. 2025; Hu 2024) and CatVTON+AM (Chong et al. 2025a); the ofline bidirectional VVT models ViViD (Fang et al. 2024), CatV2TON (Chong et al. 2025b), and MagicTryOn<sup>2</sup> (Li et al. 2025); and VACE (Jiang et al. 2025), a unified video generation and editing model. To respect the clip-based training configurations of the competing methods and ensure fair comparison under GPU memory constraints, all baselines process long videos using 81-frame windows, whereas LiveVVT performs native rolling inference over the stream.

Long-sequence try-on quality. Tables 1 and 2 evaluate long-sequence try-on fidelity. The two-stage pipelines consistently underperform video-native methods, indicating that animating a single try-on frame cannot adequately capture evolving garment deformation, occlusion, and person– garment interactions. Despite the constraints of causal recurrence, bounded look-ahead, and few-step sampling, LiveVVT delivers the strongest video quality. On ViViD-SL, it ranks first in paired VFID , VFID , and SSIM as well as both unpaired VFID metrics, while retaining competitive LPIPS. On ViT-HDL, LiveVVT again achieves the best paired VFID and SSIM scores and the best unpaired $\mathrm { V F I D } _ { I }$ . These results show that confining bidirectional denoising to a fixed-size rolling window does not compromise long-video fidelity. Local bidirectional interactions capture short-term garment deformation, while the temporal and global appearance memories propagate dynamic context and appearance constraints beyond the active window. Together, these components preserve garment identity, structural integrity, and temporal stability throughout long video streams.

Real-time eficiency. Figure 1 compares first-chunk latency and sustained throughput at 512 × 384 resolution. LiveVVT produces its first chunk in 1.56 seconds and subsequently sustains 22.39 FPS. First-chunk latency determines perceived responsiveness: a shorter delay allows users to view the try-on result without waiting for the complete source clip or lengthy ofline processing, while sustained throughput governs the continuity of subsequent updates. Relative to the similarly sized MagicTryOn, LiveVVT reduces latency by 26.35× and improves throughput by 11.37×; against the substantially larger VACE, these gains reach 100.64× and

![](images/92994d14ba0031731c471027b2099f51dcba339b6a4a7a0426565895dcfc8432.jpg)  
Figure 3: Long-Term and Multi-View Try-On Comparison.

72.22×, respectively. Together with the preceding quality results, these eficiency gains establish LiveVVT not as a merely accelerated ofline model, but as a genuinely highfidelity streaming video virtual try-on system.

Long-stream scalability. Figure 4 profiles inference cost as sequence length increases. Ofline bidirectional methods incur rapidly growing per-update and end-to-end latency because longer sequences require increasingly expensive joint spatio-temporal computation (Figures 4(a,b)). In contrast, LiveVVT confinesjoint denoising to a fixed-size rolling window. Its per-update latency stabilizes at approximately 0.5 seconds, while total inference time increases smoothly with stream duration. Its peak GPU memory is also nearly invariant to sequence length (Figure 4(c)), since both the active denoising window and temporal KV cache are bounded and the global appearance memory is constructed only once. This length-decoupled update cost and memory footprint directly reflect the proposed rolling formulation, enabling continuous long-video try-on on resource-constrained hardware.

## 4.3 Qualitative Experiments

To examine long-horizon consistency beyond aggregate metrics, Figure 3 compares LiveVVT with MagicTryOn, the closest dedicated VVT competitor in quantitative performance. MagicTryOn preserves garment structure and texture within each window, but its clip-based formulation necessitates windowed inference on long sequences, causing appearance drift across window boundaries that becomes pronounced under large viewpoint changes. This drift is evident in the shoulder-bag strap in the first sequence and in the dress straps and waist ornament in the second. LiveVVT instead maintains garment geometry, fine-grained textures, and overall dressed appearance throughout both sequences. Its persistent global appearance memory provides a sequence-level appearance anchor, while recurrent rolling-window generation preserves continuity across updates; together, they suppress local prediction drift over extended horizons. Additional comparisons are provided in the appendix.

![](images/35d5f12fb41a5a90f18b241aa3de87818fedd50e82670a5152d7022c37b88e02.jpg)  
(a) Per-update latency

![](images/cba1112e5b09ff27e966b3ab55eed2c850336fb7edc66fe38d6d4bbc2b38862c.jpg)  
(b) Total inference time

![](images/e123f4ceb8d7ecab3b53bd465e0cb1a37c2d19c25e5bd876d1b6c0ca94d6dcf0.jpg)  
(c) Peak GPU memory

Figure 4: Long-stream scalability: (a) per-update latency, (b) total inference time, and (c) peak GPU memory usage.  
![](images/87339da242e74d135c4cd9bae13e3c5aceea410aee92405de1ea4401fd552e6b.jpg)

Figure 5: Qualitative comparison across progressive distillation stages under rolling inference.
<table><tr><td>Configuration  $\mathrm { V F I D } _ { I } \downarrow$   $\mathrm { V F I D } _ { R } { \downarrow }$  SSIM↑ LPIPS↓ FPS↑</td></tr><tr><td> $\mathcal { A } _ { g } \ : \mathrm { o n l y }$  14.415 0.131 0.824 0.094 26.74</td></tr><tr><td> $+ \mathcal { A } _ { f }$  13.960 0.115 0.826 0.093 25.85</td></tr><tr><td> $+ \mathcal { H }$  13.871 0.112 0.832 0.088 22.60</td></tr><tr><td> ${ \dot { + } } { \mathcal { H } } , { \mathcal { A } } _ { f } { \mathrm { ~ } } ( { \mathrm { O u r s } } )$  13.194 0.090 0.838 0.099 22.39</td></tr></table>

Table 3: Ablation of the frontal try-on and temporal memory components on ViViD-SL under paired setting.

## 4.4 Ablation Study

Complementary memory mechanisms. Table 3 isolates the contributions of the global memory’s frontal try-on component $\ b { A _ { f } }$ and temporal memory H. All variants retain the garment component $A _ { g }$ because removing this essential target-garment condition would alter the task rather than test the memory design. Adding $\ b { A _ { f } }$ reduces $\mathrm { V F I D } _ { I }$ and ${ \mathrm { V F I D } } _ { R } .$ demonstrating that a persistent dressed-person reference mitigates appearance drift over long sequences. The temporal memory H yields broader gains in video-level quality and frame-level consistency by propagating recent dynamics and occlusion states beyond the rolling window, improving crosswindow continuity and structural coherence. Combining H and $\ b { A _ { f } }$ achieves the best VFID<sub>I</sub>, VFID<sub>R</sub>, and SSIM, confirming their complementarity: H preserves local dynamics, whereas $\ b { A _ { f } }$ anchors long-term dressed appearance.

Progressive distillation. Table 4 evaluates the three training stages under rolling denoising. Applying boundedwindow recurrent inference directly to Stage I causes substantial degradation, as its altered attention and few-step trajectory disrupt the pretrained bidirectional prior. Stage II uses teacher-trajectory regression to adapt the student to causal attention and few-step sampling, improving metrics and providing a stable initialization for $\mathcal { G } _ { s } .$ . Stage III applies CoMD to jointly match the teacher distribution and real-video rolling trajectories, yielding further gains across all metrics. Figure 5 confirms this progression: Stage I shows structural distortion and texture drift; Stage II restores garment structure but retains color and texture discrepancies; Stage III accurately preserves garment geometry, appearance details, and long-term temporal consistency. These results validate progressive distillation for transferring ofline bidirectional priors to high-fidelity streaming generation.

<table><tr><td>Training stage</td><td> $\mathrm { V F I D } _ { I \downarrow }$ </td><td> $\mathrm { V F I D } _ { R } { \downarrow }$ </td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Stage I</td><td>17.202</td><td>0.281</td><td>0.786</td><td>0.144</td></tr><tr><td>Stage II</td><td>14.371</td><td>0.156</td><td>0.803</td><td>0.116</td></tr><tr><td>Stage III</td><td>13.194</td><td>0.090</td><td>0.838</td><td>0.099</td></tr></table>

Table 4: Ablation of the three progressive distillation stages under rolling inference on ViViD-SL under paired setting.

## 5 Conclusion

We presented LiveVVT, a rolling difusion framework for high-fidelity streaming VVT. A fixed staggered-noise window preserves local bidirectional spatio-temporal modeling under bounded look-ahead and emits one clean chunk per update. Complementary temporal and persistent global appearance memories propagate recent motion and occlusion context while anchoring garment details and dressed appearance across long streams. Progressive distillation integrates bidirectional VVT learning, teacher-trajectory regression, and CoMD to transfer ofline bidirectional priors into causal few-step inference while aligning recurrent generation with teacher and real-video distributions. Experiments on paired and unpaired long-sequence benchmarks demonstrate strong fidelity, 26× lower latency, and 11× higher throughput than a similarly sized baseline, with per-update cost and peak memory nearly independent of stream length. Together, these results establish LiveVVT as a practical streaming VVT system and provide a path for adapting high-fidelity bidirectional video difusion models to eficient online generation in densely conditioned video tasks.

## References

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; et al. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Cao, Y.; Feng, S.; Shen, F.; Peng, H.; Xia, J.; Zhu, Y.; Shi, D.; and Yu, C. 2026a. UniVVT: A Unified End-to-End Framework for High-Fidelity Video Virtual Try-on. arXiv preprint arXiv:2608.05745.

Cao, Y.; Shi, D.; Fu, X.; Zou, X.; Peng, H.; Li, X.; Yu, C.; and Xing, J. 2026b. Multivariate difusion transformer with decoupled attention for high-fidelity mask-text collaborative facial generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 2670–2679.

Carreira, J.; and Zisserman, A. 2017. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6299–6308.

Choi, S.; Park, S.; Lee, M.; and Choo, J. 2021. VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14131–14140.

Choi, Y.; Kwak, S.; Lee, K.; Choi, H.; and Shin, J. 2024. Improving difusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision, 206– 235. Springer.

Chong, Z.; Dong, X.; Li, H.; Zhang, W.; Zhao, H.; Jiang, D.; Liang, X.; et al. 2025a. Catvton: Concatenation is all you need for virtual try-on with difusion models. In International Conference on Learning Representations, volume 2025, 66586–66601.

Chong, Z.; Zhang, W.; Zhang, S.; Zheng, J.; Dong, X.; Li, H.; Wu, Y.; Jiang, D.; and Liang, X. 2025b. CatV2TON: Taming difusion transformers for vision-based virtual try-on with temporal concatenation. arXiv preprint arXiv:2501.11325.

Chung, H. W.; Garcia, X.; Roberts, A.; Tay, Y.; Firat, O.; Narang, S.; and Constant, N. 2023. Unimax: Fairer and more efective language sampling for large-scale multilingual pretraining. In The Eleventh International Conference on Learning Representations.

Cui, J.; Wu, J.; Li, M.; Yang, T.; Li, X.; Wang, R.; Bai, A.; Ban, Y.; and Hsieh, C.-J. 2026. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation. In International Conference on Learning Representations.

Deng, H.; Pan, T.; Diao, H.; Luo, Z.; Cui, Y.; Lu, H.; Shan, S.; Qi, Y.; and Wang, X. 2025. Autoregressive Video Generation without Vector Quantization. In International Conference on Learning Representations.

Dong, H.; Liang, X.; Shen, X.; Wu, B.; Chen, B.-C.; and Yin, J. 2019. FW-GAN: Flow-navigated warping GAN for video virtual try-on. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 1161–1170.

Fang, Z.; Zhai, W.; Su, A.; Song, H.; Zhu, K.; Wang, M.; et al. 2024. Vivid: Video virtual try-on using difusion models. arXiv preprint arXiv:2405.11794.

Ge, Y.; Song, Y.; Zhang, R.; Ge, C.; Liu, W.; and Luo, P. 2021. Parser-free virtual try-on via distilling appearance flows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8485–8493.

Han, X.; Hu, X.; Huang, W.; and Scott, M. R. 2019. Cloth-Flow: A flow-based model for clothed person generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10471–10480.

Han, X.; Wu, Z.; Wu, Z.; Yu, R.; and Davis, L. S. 2018. Viton: An image-based virtual try-on network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 7543–7552.

He, Q.; Chen, X.; Pan, Y.; Tang, P.; Xu, P.; Gan, Z.; Wang, C.; Hu, X.; Zhang, J.; and Wang, Y. 2026. The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9182–9191.

Hu, L. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8153–8163.

Huang, X.; Li, Z.; He, G.; Zhou, M.; and Shechtman, E. 2025. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Difusion. In Advances in Neural Information Processing Systems.

Jiang, J.; Wang, T.; Yan, H.; and Liu, J. 2022. ClothFormer: Taming video virtual try-on in all module. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10799–10808.

Jiang, Z.; Han, Z.; Mao, C.; Zhang, J.; Pan, Y.; and Liu, Y. 2025. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 17191–17202.

Kodaira, A.; Hou, T.; Hou, J.; Georgopoulos, M.; Juefei-Xu, F.; Tomizuka, M.; and Zhao, Y. 2026. StreamDiT: Real-Time Streaming Text-to-Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 29200–29210.

Labs, B. F.; Batifol, S.; Blattmann, A.; Boesel, F.; Consul, S.; Diagne, C.; Dockhorn, T.; English, J.; English, Z.; et al. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv:2506.15742.

Li, G.; Zheng, S.; Zhang, H.; Chen, J.; Luan, J.; Ou, B.; Zhao, L.; Li, B.; and Jiang, P.-T. 2025. MagicTryOn: Harnessing difusion transformer for garment-preserving video virtual try-on. arXiv preprint arXiv:2505.21325.

Li, Q.; Qiu, S.; Koo, K. K.; Han, J.; and Bouyarmane, K. 2026. Dit-vton: Difusion transformer framework for unified multi-category virtual try-on and virtual try-all with integrated image editing. In 2026 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 202–211. IEEE.

Lin, S.; Yang, C.; He, H.; Jiang, J.; Ren, Y.; Xia, X.; Zhao, Y.; Xiao, X.; and Jiang, L. 2025. Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation. In Advances in Neural Information Processing Systems.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2022. Flow matching for generative modeling. In The eleventh international conference on learning representations.

Liu, K.; Hu, W.; Xu, J.; Shan, Y.; and Lu, S. 2026. Rolling Forcing: Autoregressive Long Video Difusion in Real Time. In International Conference on Learning Representations.

Morelli, D.; Fincato, M.; Cornia, M.; Landi, F.; Cesari, F.; and Cucchiara, R. 2022. Dress code: High-resolution multicategory virtual try-on. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2231– 2235.

Nguyen, H.; Nguyen, Q. Q.-V.; Nguyen, K.; and Nguyen, R. 2025. Swifttry: Fast and consistent video virtual try-on with difusion models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 6200–6208.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Shin, J.; Li, Z.; Zhang, R.; Zhu, J.-Y.; Park, J.; Shechtman, E.; and Huang, X. 2026. MotionStream: Real-Time Video Generation with Interactive Motion Controls. In International Conference on Learning Representations.

Song, Q.; Shen, Y.; Chen, M.; Sun, H.; Lan, J.; Zhu, X.; Zheng, B.; and Cao, L. 2026. FashionChameleon: Towards real-time and interactive human-garment video customization. arXiv preprint arXiv:2605.15824.

Villegas, R.; Babaeizadeh, M.; Kindermans, P.-J.; Moraldo, H.; Zhang, H.; Safar, M. T.; Castro, S.; Kunze, J.; and Erhan, D. 2023. Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions. In International Conference on Learning Representations.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Xie, S.; Girshick, R.; Dollár, P.; Tu, Z.; and He, K. 2017. Aggregated Residual Transformations for Deep Neural Networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1492–1500.

Xu, Y.; Gu, T.; Chen, W.; and Chen, A. 2025. OOTDifusion: Outfitting fusion based latent difusion for controllable virtual try-on. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 8996–9004.

Yan, W.; Zhang, Y.; Abbeel, P.; and Srinivas, A. 2021. VideoGPT: Video Generation using VQ-VAE and Transformers. arXiv preprint arXiv:2104.10157.

Yang, H.; Zhang, R.; Guo, X.; Liu, W.; Zuo, W.; and Luo, P. 2020. Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7850–7859.

Yang, S.; Huang, W.; Chu, R.; Xiao, Y.; Zhao, Y.; Wang, X.;Li, M.; Xie, E.; Chen, Y.; Lu, Y.; Han, S.; and Chen, Y. 2026.

LongLive: Real-Time Interactive Long Video Generation. In International Conference on Learning Representations.

Yin, T.; Gharbi, M.; Park, T.; Zhang, R.; Shechtman, E.; Durand, F.; and Freeman, W. T. 2024. Improved distribution matching distillation for fast image synthesis. In Proceedings of the 38th International Conference on Neural Information Processing Systems, 47455–47487.

Yin, T.; Zhang, Q.; Zhang, R.; Freeman, W. T.; Durand, F.; Shechtman, E.; and Huang, X. 2025a. From Slow Bidirectional to Fast Autoregressive Video Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22963–22974.

Yin, T.; Zhang, Q.; Zhang, R.; Freeman, W. T.; Durand, F.; Shechtman, E.; and Huang, X. 2025b. From slow bidirectional to fast autoregressive video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22963–22974.

Zeng, J.; Bai, Y.; Chen, R.; Zhang, X.; Sun, L.; Jin, D.; Xu, R.; Zhang, N.; Song, D.; and Chu, X. 2025. Eevee: Towards Close-up High-resolution Video-based Virtual Try-on. arXiv preprint arXiv:2511.18957.

Zhang, W.; Jin, Y.; Li, X.; Zhang, Y.; Cong, X.; Wang, C.; Qiao, F.; and Lian, Z. 2026. Unifit: Towards universal virtual try-on with mllm-guided semantic alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 12816–12824.

Zhao, M.; Zhu, H.; Zheng, K.; Zhou, Z.; Yan, B.; Li, X.; Yang, X.; Li, C.; and Zhu, J. 2026. Causal Forcing++: Scalable Few-Step Autoregressive Difusion Distillation for Real-Time Interactive Video Generation. arXiv preprint arXiv:2605.15141.

Zheng, J.; Xu, Z.; Chen, M.; Wang, J.; Lan, J.; Zhu, X.; Zhang, K.; Zheng, B.; and Liang, X. 2026. iTryOn: Mastering interactive video virtual try-on with spatial-semantic guidance. arXiv preprint arXiv:2605.21431.

Zhong, X.; Wu, Z.; Tan, T.; Lin, G.; and Wu, Q. 2021. MV-TON: Memory-based video virtual try-on network. In Proceedings ofthe 29th ACM International Conference on Multimedia, 908–916.

Zhu, H.; Zhao, M.; He, G.; Su, H.; Li, C.; and Zhu, J. 2026. Causal Forcing: Autoregressive Difusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation. In International Conference on Machine Learning.

Zhu, L.; Yang, D.; Zhu, T.; Reda, F.; Chan, W.; Saharia, C.; Norouzi, M.; and Kemelmacher-Shlizerman, I. 2023. TryOnDifusion: A tale of two UNets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4606–4615.

Zou, C.; Cheng, S.; Xu, B.; Zheng, D.; Li, X.; Chen, J.; and Yang, M. 2025. Video virtual try-on with conditional difusion transformer inpainter. arXivpreprint arXiv:2506.21270.

Zuo, T.; Huang, Z.; Ning, S.; Lin, E.; Liang, C.; Zheng, Z.; Jiang, J.; Zhang, Y.; Gao, M.; and Dong, X. 2025. DreamVVT: Mastering realistic video virtual try-on in the wild via a stage-wise difusion transformer framework. arXiv preprint arXiv:2508.02807.

## A Additional Quantitative Comparisons

To complement the long-sequence evaluations on ViViD-SL and ViT-HDL reported in the main paper, we introduce a longer benchmark, TikTokDress-L. We construct it by selecting the 60 longest videos from the TikTokDress test split (Nguyen et al. 2025). The resulting sequences, sourced from real-world TikTok videos, span 352–693 frames and contain diverse human motions, large viewpoint changes, and multiple garment categories. TikTokDress-L extends the evaluation to longer and more unconstrained sequences, where structural errors, garment-detail degradation, and cross-window appearance drift are more likely to accumulate during extended real-world streaming try-on.

Table 5 reports full-sequence results under paired and unpaired settings. LiveVVT achieves the best paired performance across all four metrics, with a VFID of 19.668, a ${ \mathrm { V F I D } } _ { R }$ of 0.412, an SSIM of 0.842, and an LPIPS of 0.103. The strong VFID results demonstrate high video-level realism over hundreds of frames, while the simultaneous gains in SSIM and LPIPS show that this temporal stability is achieved without sacrificing frame-level garment structure or perceptual fidelity. In particular, LiveVVT remains robust under large pose and viewpoint changes, where independently processed clips are prone to inconsistent garment geometry, texture drift, and discontinuities at window boundaries.

LiveVVT also ranks first on both unpaired metrics, obtaining a $\mathrm { V F I D } _ { I }$ of 30.447 and a ${ \mathrm { V F I D } } _ { R }$ of 0.762. The unpaired setting evaluates generalization to arbitrary target garments that are not matched to the source person’s original clothing, more closely reflecting practical try-on scenarios. Since no frame-wise ground truth exists for these novel person–garment combinations, we assess their videolevel realism and distributional consistency using VFID. The consistent performance across paired and unpaired settings provides strong evidence that LiveVVT preserves garment identity and dressed appearance throughout long, unconstrained sequences. These findings are consistent with the proposed rolling formulation: local bidirectional denoising models short-range deformation and motion within each active window, while temporal and persistent global appearance memories propagate dynamic context and appearance constraints beyond the window. Together, they enable stable long-horizon try-on generation without requiring fullsequence bidirectional inference.

## B Fast Image Try-On

Across all stages of progressive distillation, LiveVVT is trained on a multimodal dataset comprising dedicated video data together with image try-on data from VITON-HD (Choi et al. 2021) and DressCode (Morelli et al. 2022). This unified training protocol enables the same model to perform both streaming video try-on and fast single-image try-on without a task-specific image branch. An image sample is represented as a one-frame video during training. Because no frontal keyframe is required in this setting, the global appearance memory contains only the garment attention key/value (KV) features, i.e., $\mathcal { A } = \mathcal { A } _ { q } .$ At inference, the image is processed with a single chunk $( \bar { N } = 1 )$ using the same few-step denoising interface as streaming generation. Table 6 evaluates this image capability on the 2,032-image VITON-HD test set at $5 1 2 \times 3 8 4$ resolution. FID and KID measure distributional fidelity, while SSIM and LPIPS quantify paired structural similarity and perceptual distance. LiveVVT obtains the best paired FID (5.910) and KID (0.419), together with the lowest paired LPIPS (0.054). Its paired SSIM (0.873) is close to the strongest result (0.883). Under the unpaired setting, which tests person–garment combinations without frame-wise correspondence, LiveVVT achieves the best KID (0.889) and the second-best FID (9.212). Taken together, these results show that the unified model preserves strong image try-on quality while sharing the same parameters and conditioning pathway with the streaming video model. The comparison does not by itself disentangle the individual efects of video and image supervision; rather, it verifies that multimodal training supports both capabilities within one model.

<table><tr><td></td><td colspan="3">Paired</td><td colspan="2">Unpaired</td></tr><tr><td>Method</td><td> $\mathrm { V F I D } _ { I \downarrow }$  VFIDR↓SSIM↑LPIPS↓</td><td></td><td></td><td></td><td> $\mathrm { V F I D } _ { I } \downarrow \mathrm { V F I D } _ { R } \downarrow$ </td></tr><tr><td>OOTDiff+AM</td><td>42.955</td><td>1.816</td><td>0.656 0.259</td><td>47.001</td><td>2.622</td></tr><tr><td>CatVTON+AM</td><td>33.043</td><td>1.422</td><td>0.714</td><td>0.213 39.726</td><td>1.192</td></tr><tr><td>VACE</td><td>23.589</td><td>0.875</td><td>0.835</td><td>0.128 32.292</td><td>2.378</td></tr><tr><td>ViViD</td><td>28.712</td><td>0.812</td><td>0.791</td><td>0.149 35.954</td><td>1.316</td></tr><tr><td>CatV2TON</td><td>29.888</td><td>1.633</td><td>0.796</td><td>0.163 37.385</td><td>1.420</td></tr><tr><td>MagicTryOn</td><td>23.007</td><td>0.927</td><td>0.818 0.118</td><td>32.547</td><td>1.043</td></tr><tr><td>Ours</td><td>19.668</td><td>0.412</td><td>0.842</td><td>0.103 30.447</td><td>0.762</td></tr></table>

Table 5: Full-sequence quantitative comparison on the longsequence TikTokDress-L benchmark under paired and unpaired settings.

Figure 6 compares average single-image inference time at 512 × 384 resolution. LiveVVT requires 0.31 seconds per image, compared with 5.25, 2.40, 1.13, and 7.22 seconds for IDM-VTON (Choi et al. 2024), OOTDifusion (Xu et al. 2025), CatVTON (Chong et al. 2025a), and UniFit (Zhang et al. 2026), respectively. These measurements correspond to speedups of 16.94 $\times , 7 . 7 4 \times , 3 . 6 5 \times$ , and 23.29×. UniFit has the largest computational footprint because it combines a 12B-parameter FLUX.1-Fill-dev (Labs et al. 2025) backbone with a 2B-parameter auxiliary MLLM (Bai et al. 2025) for semantic alignment. LiveVVT’s eficiency comes from two design choices. First, it uses four denoising steps, whereas competing difusion-based methods typically require tens of steps. Second, it encodes the garment once before denoising and caches its attention features, avoiding repeated processing of garment tokens in subsequent iterations. Other methods instead recompute a dedicated appearance branch or concatenate garment and noisy-image tokens at each step. These choices reduce both sampling and conditioning costs, enabling fast image try-on with the same model used for streaming video generation.

## C Additional Ablation Studies

## C.1 Analysis of the Hyperparameter λ

CoMD combines two complementary supervision signals for recurrent few-step generation. DMD aligns the student distribution with the high-fidelity teacher prior, but provides no direct real-video supervision for the history-conditioned, staggered-noise states that characterize rolling inference. RFM addresses this limitation by sampling windows from real videos and reconstructing the temporal cache, global appearance memory, and staggered noise schedule used during rolling inference. It therefore supervises the student on real-data trajectories under deployment-aligned states. The combined objective, $\mathcal { L } _ { \mathrm { C o M D } } = \mathcal { L } _ { \mathrm { D M D } } + \lambda \mathcal { L } _ { \mathrm { R F M } }$ , balances teacher-prior transfer with real-trajectory alignment, where λ controls the contribution of RFM.

![](images/4e908f87c458f857ee7a1eb79c9b6a68e2a67e1e355296aa753df356e525a276.jpg)

Figure 6: Average single-image try-on inference time of LiveVVT and state-of-the-art image try-on methods, evaluated at a resolution of 512 × 384.
<table><tr><td></td><td colspan="3">Paired</td><td colspan="2">Unpaired</td></tr><tr><td>Method</td><td>FID↓ KID↓</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>KID↓</td></tr><tr><td>IDM-VTON 6.338</td><td>1.322</td><td>0.881</td><td>0.079</td><td>9.611</td><td>1.639</td></tr><tr><td>OOTDiff</td><td>9.305 4.086</td><td>0.819</td><td>0.088</td><td>12.4084.689</td><td></td></tr><tr><td>CatVTON</td><td>6.139 0.964</td><td>0.869</td><td>0.097</td><td>9.143</td><td>1.267</td></tr><tr><td>UniFit</td><td>8.799 0.702</td><td>0.883</td><td>0.065</td><td></td><td></td></tr><tr><td>Ours</td><td>5.910 0.419</td><td>0.873</td><td>0.054</td><td>9.212 0.889</td><td></td></tr></table>

Table 6: Quantitative comparison of LiveVVT with publicly available image try-on methods on the VITON-HD test set.

Table 7 evaluates this trade-of on ViViD-SL under paired evaluation. Setting λ = 0 reduces CoMD to DMD-only training. Introducing RFM consistently improves both VFID metrics and SSIM for $\lambda \in \{ 0 . 1 , 0 . 2 , 0 . 4 \}$ , showing that direct supervision on real rolling trajectories improves video-level fidelity and structural consistency beyond teacher-distribution matching alone. The accompanying increase in LPIPS, however, indicates that RFM should remain complementary to DMD rather than dominate the objective. We adopt $\lambda = 0 . 2$ because it provides the most favorable balance among video fidelity, structural consistency, and perceptual quality. Increasing the weight further produces non-monotonic behavior: $\lambda = 0 . 8$ improves SSIM but weakens both VFID metrics and LPIPS relative to the selected setting. These results validate the collaborative formulation of CoMD and show that a moderate RFM weight best preserves the teacher prior while aligning the student with real-video rolling trajectories.

<table><tr><td>λ</td><td> $\mathrm { V F I D } _ { I } \downarrow$ </td><td> $\mathrm { V F I D } _ { R \downarrow }$ </td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>0</td><td>14.515</td><td>0.205</td><td>0.826</td><td>0.094</td></tr><tr><td>0.1</td><td>12.807</td><td>0.123</td><td>0.834</td><td>0.096</td></tr><tr><td>0.2</td><td>13.194</td><td>0.090</td><td>0.838</td><td>0.099</td></tr><tr><td>0.4</td><td>13.435</td><td>0.146</td><td>0.838</td><td>0.104</td></tr><tr><td>0.8</td><td>14.446</td><td>0.146</td><td>0.843</td><td>0.104</td></tr></table>

Table 7: Ablation of the RFM weight λ in CoMD on ViViD-SL under the paired setting. The highlighted row denotes the default configuration used in the main experiments.

<table><tr><td>Conditioning</td><td> $\mathrm { V F I D } _ { I \downarrow }$ </td><td> $\mathrm { V F I D } _ { R \downarrow }$ </td><td></td><td>SSIM↑ LPIPS↓ FPS↑</td></tr><tr><td>Cached</td><td>13.194</td><td>0.090</td><td>0.838</td><td>0.09922.39</td></tr><tr><td>Repeated Encoding</td><td>13.280</td><td>0.103</td><td>0.840</td><td>0.08919.73</td></tr></table>

Table 8: Quality and throughput comparison of garmentconditioning strategies under the paired setting. Cached stores garment KV features in the global appearance memory, whereas Repeated Encoding recomputes garment conditioning at every rolling update.

## C.2 Cached vs. Repeated Garment Conditioning

The target garment remains unchanged throughout a try-on sequence, making its appearance features naturally reusable across rolling updates. LiveVVT therefore encodes the garment image once and stores its attention key/value (KV) features in the persistent global appearance memory. To examine whether recomputing the garment condition improves generation, we compare this Cached design with Repeated Encoding. The latter does not retain garment KV features; instead, it concatenates garment tokens with the noisy tokens of the active window at every rolling denoising iteration and performs joint attention over the extended sequence. Table 8 evaluates the resulting quality–eficiency trade-of.

The default Cached design and the Repeated Encoding alternative achieve comparable but metric-dependent generation quality. Cached conditioning performs better on both VFID metrics, while Repeated Encoding yields a small SSIM increase and a lower LPIPS. Recomputing the garment condition therefore provides no consistent advantage across video fidelity, structural similarity, and perceptual quality. Its eficiency cost is more pronounced: throughput decreases from 22.39 to 19.73 FPS, making Cached conditioning 1.13× faster. These results show that persistent garment KV caching removes redundant conditioning computation without compromising overall try-on quality, directly supporting the highthroughput rolling inference targeted by LiveVVT.

## D Additional Visual Comparisons

Figures 7–10 provide additional paired comparisons with publicly available VVT methods, including ViViD, CatV2TON, and MagicTryOn. The examples span multiple garment categories and substantial pose and viewpoint changes, enabling a direct assessment of both local reconstruction fidelity and sequence-level appearance consistency.

![](images/ca5a58de0767f93d2bf1d94d6c8e1c0c52a53936b0cef93fc09b39db393ca701.jpg)

Figure 7: Long-sequence comparison for a white blouse under substantial pose and scale changes.  
![](images/4800f9117d636c9816d429c304a742465123c2eecd675264943a6e884d71189a.jpg)  
Figure 8: Long-sequence comparison for denim shorts across large viewpoint changes.

ViViD and CatV2TON recover the overall garment layout but frequently lose fine texture, exhibit color deviations, or produce boundary artifacts around collars, hems, and occluded regions. These artifacts become more pronounced under large nonrigid deformation, where accurate appearance preservation requires consistent modeling across successive frames.

MagicTryOn produces strong results within individual clips, yet its independently processed windows do not share an explicit sequence-level appearance state. As a result, locally plausible outputs can still undergo texture or color changes at window boundaries. Figure 10 illustrates this behavior: the generated shorts exhibit a visible color shift in later frames. LiveVVT maintains comparable local detail while preserving a stable garment appearance across the complete sequence. Its rolling window provides local bidirectional interactions for short-range deformation, and the persistent global appearance memory supplies a sequence-level reference that remains fixed across updates. This combination suppresses cross-window drift and yields more coherent long-horizon try-on generation.

![](images/2c4dd5c8f9a8f5d1ed2220fe3a24c621c2a3ac97f348655a60b95768b5c6decd.jpg)  
Figure 9: Long-sequence comparison for a dark knit top across front and rear views.

![](images/6d2928a04b38516c9fe2c38f903287430a7d939ae0ab085747225af19a007fbc.jpg)  
Figure 10: Long-sequence comparison for black shorts. MagicTryOn exhibits an appearance shift in later frames, whereas LiveVVT maintains consistent garment color and texture.

## E Long-Sequence Try-On Examples

Figures 11 and 12 present additional long-sequence results under the unpaired setting, where the target garment is selected independently of the source person’s original clothing. The examples cover diverse garment categories, including tops, casual summer outfits, and dresses, together with full-body rotations and substantial changes in subject scale caused by camera motion. These sequences are particularly challenging because the generated garment must remain identifiable when its visible regions change dramatically and must recover consistently after rear views, self-occlusion, or abrupt zooming. LiveVVT maintains coherent garment appearance throughout complete 360<sup>◦</sup> rotations and preserves a stable dressed appearance under changes in camera scale. This behavior indicates that the persistent global appearance memory provides a sequence-level garment anchor, while rolling-window generation maintains local motion continuity across updates.

Figure 13 evaluates LiveVVT on in-the-wild videos with complex backgrounds, rapid articulated motion, and severe self-occlusion. Across these challenging cases, the generated results generally preserve the target garment category, appearance, and temporal continuity despite large changes in body configuration. Occasional local artifacts remain around fine garment boundaries and under extreme motion, highlighting directions for further improvement. Nevertheless, the results demonstrate that LiveVVT retains robust longsequence try-on capability beyond controlled benchmark scenes, supporting its practical use in interactive streaming applications.

## F Limitations and Discussion

While LiveVVT substantially improves long-horizon video try-on quality at streaming speed, its design also defines a clear operating regime. The persistent appearance memory is built from the target garment and, for video input, a frontal source-person reference. This anchor is efective for preserving person-specific dressed appearance over extended streams, but the method can become less reliable when the observed appearance departs markedly from the reference because of severe illumination changes, self-occlusion, extreme articulation, or large out-of-plane rotation.

The remaining errors are predominantly localized rather than global: fine garment boundaries, heavily occluded regions, and rapidly moving body parts may still exhibit small distortions or texture inconsistencies. Such failures are only partially reflected by the current evaluation protocol, since SSIM and LPIPS measure frame-level correspondence while VFID measures distributional video quality. A more complete assessment should therefore combine these metrics with explicit temporal-consistency and boundary-accuracy measures, as well as human judgments of garment identity and perceptual stability.

The eficiency gains also involve an explicit context– latency trade-of. Fixed-size rolling windows, cached garment features, and few-step sampling make the per-update cost largely insensitive to sequence length, but they restrict the amount of future context available to each emission and may be suboptimal for abrupt motion or rapid appearance changes. Adaptive windowing or content-aware memory updates could selectively allocate additional computation in such cases while preserving the constant-cost behavior during routine motion.

![](images/0a73747149e587d5c4997ec3c906d2c0fb543379b4a26af3111be43a0251dfaa.jpg)  
Figure 11: Unpaired long-sequence try-on examples across diverse garment categories and viewpoint changes. LiveVVT preserves garment appearance through full-body rotations and extended rolling generation.

![](images/7e0ffc19dfc1e3098d11f0653fc76ee752341cacd64a69dc27cbb8293a5f4401.jpg)  
Figure 12: Additional unpaired long-sequence examples with substantial viewpoint and camera-scale changes. LiveVVT main tains a coherent dressed appearance after rear views, self-occlusion, and abrupt zooming.

![](images/7ca43c042947a25d69a52307b2c8180184fb743e07d33622770b574b33bff784.jpg)  
Figure 13: Challenging in-the-wild try-on examples with complex backgrounds, rapid motion, and severe self-occlusion. LiveVVT retains robust garment appearance and temporal continuity across these unconstrained sequences.