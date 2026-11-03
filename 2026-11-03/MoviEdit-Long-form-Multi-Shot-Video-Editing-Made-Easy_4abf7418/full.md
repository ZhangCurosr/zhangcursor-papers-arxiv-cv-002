# MoviEdit: Long-form Multi-Shot Video Editing Made Easy

Suhyeon Lee<sup>1,2,∗</sup> Zhengang Li<sup>1</sup> Soo Ye Kim<sup>1</sup> Peihao Zhu<sup>1</sup> Jong Chul Ye<sup>2</sup> Chun-Hao Paul Huang<sup>1</sup> Duygu Ceylan<sup>1</sup> Yan Kang<sup>1</sup> Hyeonho Jeong<sup>1,∗,†</sup>

<sup>1</sup>Adobe <sup>2</sup>KAIST

<sup>∗</sup>Equal Contribution <sup>†</sup>Tech Lead

Long-form multi-shot video editing requires an intended edit to remain coherent across abrupt changes in scene, viewpoint, and subject appearance, while processing substantially longer visual context than conventional single-shot editors. We introduce MoviEdit, an end-to-end framework for scalable long-form multi-shot video editing that jointly addresses the challenges of data, modeling, and evaluation. To provide cross-shot supervision data at scale, we develop automatic data-generation pipelines spanning synchronized multi-view footage, real-world multi-shot videos, and generated multi-shot content, yielding 87K paired editing videos without manual annotation. Architecturally, MoviEdit employs a cascaded bidirectional–autoregressive design: a lightweight bidirectional stage first establishes a globally coherent editing draft over a spatiotemporally compressed representation, which is then autoregressively propagated and refined over the full video. We further introduce MoviEdit-Bench, a dedicated benchmark for systematic evaluation of long-form multi-shot video editing. Extensive experiments show that MoviEdit substantially outperforms existing open-source models and performs on par with industry frontier models across text-guided and image-guided editing. Visit our project page for complete results and to access the benchmark: multi-shot-edit.github.io

![](images/10ca4842a80491647f25c0cf3fd41a55d05431b2aa94a2f47e0db1a6e828c474.jpg)

(a) Task: Text-guided Video Editing (V2V) Edit Instruction: “Remove the man wearing a black tactical suit and a dark mask.”  
![](images/bcda8b6f33047df4e14b66030e1a0e1c792ab74a3c6ea73635caaedd243d36e1.jpg)

![](images/8e6149c6d2688878f4b93807f4bd08dbc015067fc07f765b6cc988ce79465dd5.jpg)

![](images/e807d0ff4cdca71b29820c0cedb1f2b0ef821e8662d58b07147cfaca11c5bce9.jpg)  
Figure 1 Long-form multi-shot video editing with MoviEdit. MoviEdit supports (a) text-guided and (b) edited-frame-guided multi-shot video editing, enabling consistent edits across shots, and (c) generalizes to videos longer than those seen during training.

## 1 Introduction

Visual narratives are inherently compositional: their meaning emerges not only within individual shots, but also from the relationships among shots over time. Recent advances in generative video modeling have begun to capture this structure, moving beyond short, isolated clips toward long-form, multi-shot video generation (Guo et al., 2025; Zhang et al., 2025; Cai et al., 2026; Meng et al., 2026). Yet existing video editing methods (DecartAI Team, 2025; Jiang et al., 2025; Lee et al., 2026; Liu et al., 2025; Ju et al., 2026) remain predominantly designed for short, single-shot videos, typically spanning only a few seconds. Extending editing to long-form, multi-shot content is not merely a matter of processing more frames: an intended edit must persist coherently across cuts even as the scene, viewpoint, motion, appearance, and surrounding context change.

The dificulty is structural rather than simply one of scale. Editing each shot independently discards the cross-shot dependencies needed to preserve a consistent edit, whereas jointly processing the full source and target sequences quickly becomes computationally prohibitive. Yet this computational dilemma is only one part of the problem. Fundamentally, long-form multi-shot editing exposes three coupled bottlenecks in existing video editing systems: data, modeling, and evaluation. At the data level, learning to edit a subject consistently across shots requires source and target video pairs with explicit cross-shot subject correspondence. Such data does not exist in the real world and, to our knowledge, no existing pipeline can synthesize them at the scale and quality required for training (Zi et al., 2025; Bai et al., 2026; Wu et al., 2025; Lin et al., 2026; Long et al., 2026). On the modeling side, long-form editing is fundamentally a memory problem. Computationally, editing typically requires processing a source-video representation alongside the target video being synthesized; under common token-concatenation designs, the quadratic cost of global self-attention rapidly becomes prohibitive as duration increases. Third, on the evaluation side, there is no dedicated evaluation protocol for this setting. Existing video editing benchmarks (Li et al., 2025; Bian et al., 2025; Ju et al., 2026) focus primarily on short, single-shot clips and therefore do not assess the capabilities that distinguish multi-shot editing, such as whether an edit continues to localize the correct subject after a cut and whether its appearance remains consistent when that subject reappears. Once supervised data and evaluation are in place, the modeling problem becomes tractable with the goal of maintaining temporal and appearance consistency over long horizons.

To this end, we present MoviEdit, an end-to-end framework for training and evaluating long-form multi-shot video editing models. MoviEdit addresses the three bottlenecks through a matched set of data, modeling, and evaluation components. First, we develop a scalable data generation pipeline that combines multi-view-aware image editing, single-shot video editing, and VLM-based quality filtering to construct high-quality paired training examples with explicit subject correspondence across shots. Second, we present a cascaded video editing architecture that combines bidirectional global context with causal propagation. A lightweight bidirectional editing stage first constructs a globally coherent edit draft, after which a causal stage autoregressively propagates and refines the edit over the full sequence. This decomposition avoids the prohibitive memory cost of jointly processing the complete source and target videos while preserving temporal and appearance consistency across shots. Lastly, we introduce MoviEdit-Bench, a dedicated benchmark that spans multiple dificulty levels and evaluates both local, subject-specific edits and global, scene-level transformations. It provides automated metrics for cross-shot target localization and edit consistency, together with a curated collection of copyright-free multi-shot videos. Together, these components provide the supervision, architecture, and evaluation infrastructure needed to move video editing beyond isolated clips toward coherent, long-form visual narratives. Extensive experiments further demonstrate that a MoviEdit model trained using our framework substantially outperforms existing open-source solutions, and achieves performance on par with strong proprietary systems (Google DeepMind, 2026; Alibaba Cloud, 2026; Runway, 2026) across both text-guided video editing and image-guided video editing.

Our contributions are summarized as follows:

• We introduce a scalable paired-data generation pipeline for supervised long-form multi-shot video editing.

• We propose an eficient long-form video editing architecture based on a cascaded bidirectional–autoregressive difusion framework.

• We present MoviEdit-Bench, a dedicated benchmark and evaluation suite for long-form multi-shot video editing.

## 2 Related Work

Difusion-based Video Editing. Early difusion-based video editing methods largely repurpose pretrained generative models as editing priors through inference-time adaptation. Typical strategies include inversion (Qi et al., 2023; Wang et al., 2023), attention or feature manipulation (Qi et al., 2023; Liu et al., 2024; Geyer et al., 2024; Ceylan et al., 2023; Jeong et al., 2024a), auxiliary visual conditioning (Zhang et al., 2023; Jeong & Ye, 2023), and per-video optimization (Wu et al., 2023; Jeong et al., 2024b, 2025), enabling edits without requiring large-scale paired training data. While efective in many settings, these methods rely primarily on manipulating pretrained generative priors at inference time rather than learning generalizable editing behavior from data, which can limit robustness, scalability, and editing flexibility while introducing substantial inference overhead.

More recently, video editing has shifted toward a data-driven paradigm, where models learn editing behavior from paired source–target videos (DecartAI Team, 2025; Ju et al., 2026; Jiang et al., 2025; Liu et al., 2025; Polyak et al., 2024; Yang et al., 2026a; Seo et al., 2026; Lin et al., 2026; Lee et al., 2026; Huang et al., 2025b). These methods typically build on pretrained video generators and difer mainly in how source-video information is incorporated, ranging from channel-level (DecartAI Team, 2025) or token-level concatenation (Ju et al., 2026; Lee et al., 2026) to adapter-based additive conditioning (Jiang et al., 2025). They also vary in how the edit is specified; for example, GenProp (Liu et al., 2025) conditions on an edited reference frame and propagates the corresponding transformation through the video. Following the same data-driven philosophy, our models use token-level source conditioning and are trained separately for text-guided and edited-image-guided editing. Unlike prior reference-image-guided methods (Ouyang et al., 2024; Ku et al., 2024; Liu et al., 2025) that typically condition on an edited first frame, our image-guided model supports edited references from arbitrary frames. Complementary approaches introduce explicit reasoning or agentic control into the editing process, from latent reasoning over edit regions (Yang et al., 2026b) to LLM/VLM-driven shot-level planning for multi-shot editing (Wu et al., 2026).

More broadly, while most existing end-to-end video editing models remain focused on short, single-shot settings, our work systematically addresses long-form multi-shot editing through cross-shot supervision, long-context modeling, and dedicated evaluation.

Long-Form and Multi-Shot Video Synthesis. Recent advances have made long-form video generation increasingly tractable through autoregressive generation and eficient long-context modeling. Methods such as CausVid (Yin et al., 2025) and Self-Forcing (Huang et al., 2025a) generate videos autoregressively in chunks, using windowed attention or rolling KV caches to bound the context and computational cost over long horizons. In parallel, SANA-Video (Chen et al., 2026) and SANA-Streaming (Zhao et al., 2026b) replace softmax attention with linear attention (Katharopoulos et al., 2020), enabling eficient long-context modeling and constant-memory, constant-compute streaming generation. Few-step difusion distillation has also been explored to reduce autoregressive sampling latency, including DMD- and consistency-based methods (Huang et al., 2025a; Zhao et al., 2026a) and, more recently, velocity moment matching (VMM) (Zhou et al., 2026).

Beyond a continuous sequence, recent work has begun to address multi-shot video generation by extending conventional short-form video models with two key modifications: training on multi-shot video data and introducing shot-aware sparse attention for eficient cross-shot context modeling. HoloCine (Meng et al., 2026) employs dense intra-shot attention while using the first-frame tokens of each shot as compact summary tokens for cross-shot information exchange. Mixture-of-Contexts (Cai et al., 2026) further introduces content-adaptive sparse attention that dynamically retrieves a small set of relevant historical chunks for each query, while ContextMaster (Guo et al., 2026) adopts fixed-budget context routing that combines mandatory task-specific context with dynamically selected historical information. MSEditor (Feng et al., 2026) instead leverages synchronized multi-view data to adapt a first-frame-conditioned video model for consistent appearance propagation across multiple shots.

Despite their diferent implementations, these approaches improve scalability by limiting dense global interactions across the video, making it dificult to maintain a consistent appearance over long temporal spans. This highlights the need for modeling strategies that balance the computational eficiency of autoregressive generation with the global context provided by bidirectional modeling.

![](images/62b47cce23f63cfee10a29ddb12e243346f493907a47b1ecdc5a12e03ec20437.jpg)  
Figure 2 Scalable paired-data generation for multi-shot video editing. Three annotation-free pipelines transform diverse video sources into large-scale paired data for consistent multi-shot editing. Pixel-aligned editing uses a spatial–temporal editing decomposition to ensure cross-shot consistency, while semi-pixel-aligned editing uses coupled multi-shot video generation with inference-time interaction to produce structurally consistent source–target pairs.

## 3 MoviEdit: Long-Form Multi-Shot Video Editing

MoviEdit provides an end-to-end framework that develops the key ingredients for building industry-level long-form multi-shot video editors. We first describe our scalable paired-data generation pipelines for multi-shot video editing tasks (Fig. 2), followed by the cascaded architecture for a long-form video editing model (Fig. 5). We further describe several complementary techniques for improving inference eficiency. Finally, we introduce MoviEdit-Bench for systematic evaluation of cross-shot editing capabilities.

## 3.1 Scalable Paired Data Generation for Multi-Shot Video Editing

We develop three scalable pipelines for constructing paired supervision without human annotation (Figs. 2 and 3). We leverage in-house synchronized multi-camera footage (Fig. 13), real-world cinematic clips, or pretrained multi-shot text-to-video difusion models. These sources ofer complementary strengths in cross-view correspondence, cinematic realism, and controllable synthesis, respectively, supporting three supervision regimes: pixel-aligned editing, subject localization, and semi-pixel-aligned editing. Detailed training-data statistics are provided in Appx. A, with data-recipe ablations in Fig. 11.

For pixel-aligned editing and subject localization pipelines, we obtain single-scene, multi-shot clips either from real-world cinematic clips using shot detection (Wang et al., 2026) and scene grouping (Mun et al., 2022), or from synchronized multi-camera footage through camera-view selection and temporal cuts. A vision-language model (VLM) (Yang et al., 2025; Team et al., 2026) proposes candidate subjects as short text queries, merging observations of the same entity across shots into one query. The multi-shot clips and subject queries are used to construct supervision for both tasks.

Pixel-aligned Editing. The pixel-aligned editing task is our primary setting, where the edited video preserves pixel-wise spatial correspondence with the source while modifying only the instructed visual content. For object-level edits, including add, remove, swap, and appearance change, we use the subject query to select the target instance to edit in the first frame of each shot. For global edits, including environment change and stylization, we apply the edit to the full first frame of each shot.

Ensuring cross-shot consistency is a critical challenge in this setting. We address it by decomposing multi-shot editing into two complementary subproblems: cross-shot spatial editing, which resolves a consistent edit across representative frames from all shots, and within-shot temporal editing, which propagates each resolved edit through its corresponding shot. For the first stage, we arrange the first frames ${ \boldsymbol y } _ { s } ^ { 1 }$ from all S source video shots $\{ \pmb { y } _ { s } \} _ { s = } ^ { S }$ into a single image grid $[ \pmb { y } _ { 1 } ^ { 1 } ; \dots ; \pmb { y } _ { S } ^ { 1 } ]$ and jointly edit them in a single inference pass using of-the-shelf text-guided image editing models $g _ { \phi }$ . Each edited

(a) Pixel-aligned Editing Task  
![](images/3c746a0e747b9bc36d6e4f1e4482e338d53e691ff148ede7cac83b48fac8f28b.jpg)  
Add: “Add a woven bohemian floral headband to the woman in the white striped shirt.”

![](images/89410ce70aeb02e0cd8b83fa480b2a2f101d6d8db84709ef51376533e9a0f0ca.jpg)  
Remove: “Remove the man in the black puffer jacket.”

![](images/fe6d19ba1c7f1166ed1af7f17096df4fa36dbdf7685558c5618461426f145c59.jpg)  
Appearance Change: “Change the outfit of the woman in the black coat to a burgundy coat with gold buttons.”

![](images/5168a5618a0c3e1e0f5c4adeaec5e54c1261e730d06f2d2f71e73cb9f03a1315.jpg)  
Swap: “Replace the woman in the leather jacket with another female officer wearing a navy police uniform.”

![](images/02a8dbc7ee475c012c2ad239888e870585a45b78ae3e93cd49f893f6b4dcb4f2.jpg)  
Stylization: “Apply a watercolor painting style.”

![](images/39b1c918719f5c22d5f8d482aec6e6adab501fcd3e42a5e9f3b882b227391339.jpg)  
Env. Change: “Change the scene to a rainy setting.”

(b) Subject Localization Task  
![](images/042db442d4322b907511d013adf60cc9b331bcccff6747ad361bf0672dba8052.jpg)  
(c) Semi-pixel Aligned Editing Task  
Highlighting: “Highlight the man in the checkered shirt in navy.”

![](images/eab7f06d1323f4d08386640df7ea5eeadfd40d613552c3273e1baaa0285868f8.jpg)  
Env. Change: “Reimagine the lighting to evoke a midnight storm, with deep blue and purple tones.”

![](images/a34efe371b4f07332be4ae6c7b797233787608098a9b25d89006e2cffe432935.jpg)  
Outlining: “Outline the man in black athletic gear in green and the orange basketball in purple.”

![](images/00f18ec7a803e96e09344e275505461d3b065993491698442a9e3bf4177a9761.jpg)

![](images/e4385dc2493b311ba2f47fb3f85e29e89fd4ecaea652ada9d8e4a4710f31c709.jpg)  
Medium Change: “Reimagine this scene in a cinematic 2D animation style.”  
Color Filling: “Fill the man holding the camera with red.”

![](images/2ce1fe4a86fce40669d971fbadca6522ad09220fa1f8efd31088685847b3c37b.jpg)  
Character Appearance Change: “Reimagine the man as a woman.”

Figure 3 Representative training data pairs across editing tasks. We show source and target video examples from our three data-generation pipelines. The pixel-aligned pipeline covers add, remove, appearance change, swap, stylization, and environment change; the localization pipeline covers highlighting, outlining, and color filling; and the semi-pixel-aligned pipeline covers environment change, medium change, and character appearance change. Note: Faces of real actors in the displayed examples are blurred to protect their identity

![](images/d749efa0d28dd40b59606d0d31a549db9a999ce35f579ba069509a6d00a5dc13.jpg)  
Figure 4 Long-horizon appearance-drift stress test. We compare autoregressive editing with and without draft conditioning in a small-scale train setting. Naive autoregressive editing accumulates appearance drift and saturation, while draft-conditioned rollout maintains a stable edited appearance.

first frame is then propagated independently within its corresponding shot using a pretrained first-frame-conditioned single-shot video editing model $h _ { \psi }$

$$
\tilde { y } _ { 1 : S } ^ { 1 } = [ \tilde { y } _ { 1 } ^ { 1 } ; \ldots ; \tilde { y } _ { S } ^ { 1 } ] = g _ { \phi } \big ( [ y _ { 1 } ^ { 1 } ; \ldots ; y _ { S } ^ { 1 } ] , e \big ) , \quad x _ { s } = h _ { \psi } \big ( \tilde { y } _ { s } ^ { 1 } , y _ { s } , e \big ) , \quad s = 1 , \ldots , S ,\tag{1}
$$

where e denotes the edit instruction, and $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { s } }$ is the target edited video shot. This spatial–temporal decomposition avoids bootstrapping from an existing multi-shot video editor. We apply quality filtering at both stages: first-frame QA checks edit success and cross-shot consistency, while propagation QA checks within-shot consistency and overall video consistency after recomposition, rejecting 17%/38% of multi-camera and 32%/47% of cinematic pairs at the two stages, respectively.

Subject Localization. This task, also supported by recent video editing models (Ju et al., 2026; Liu et al., 2025), requires identifying a queried subject across multiple shots and applying localized visual transformations, including efects such as highlighting, outlining, and color filling. Given a multi-shot source video and subject query, video segmentation models (Cheng et al., 2024; Sari & Favaro, 2025; Carion et al., 2026) track the subject across all shots where it appears and produce dense per-frame masks. Applying the specified efect to these masks yields paired source–target videos for supervising cross-shot localization. With VLMs, we further filter the generated pairs to ensure consistent subject localization and correct efect application, rejecting 24% of pairs.

Semi-pixel-aligned Editing. This pipeline generates paired source–target videos that preserve spatial structure, temporal composition, and narrative correspondence without requiring strict pixel-level alignment. By relaxing exact correspondence, it supports a broader range of edits while increasing data diversity through coupled generation with a pretrained text-driven multi-shot video generator (e.g., Meng et al., 2026; Zhang et al., 2025). We apply this pipeline to tasks including environment change, medium change, and character appearance change.

For each pair, we couple source and target generation by sharing the same per-shot captions and initial noise while using distinct global captions. An LLM constructs the source and target global captions by modifying only the attribute associated with the intended edit while preserving all other content. To further couple them, we introduce task-specific inference-time interaction mechanisms. For character appearance change, the source and target instead share a common denoising trajectory until a fork timestep $t _ { \mathrm { f o r k } }$ . For $t > t _ { \mathrm { f o r k } }$ , the shared latent is updated using the averaged source- and target-conditioned velocity predictions,

$$
\tilde { v } ( z _ { t } , t ) = \left( v _ { \omega } ( z _ { t } , e _ { \mathrm { s r c } } , t ) + v _ { \omega } ( z _ { t } , e _ { \mathrm { t g t } } , t ) \right) / 2 .\tag{2}
$$

For $t \le t _ { \mathrm { f o r k } }$ , the two generations branch from the shared latent state and are denoised independently under their respective conditions. For environment change and medium change editing, the target generation is additionally conditioned via ControlNet (Zhang et al., 2023) on either an HED edge or a depth map extracted from the source video. These guarantee necessary structural correspondence while allowing the target appearance to diverge.

We further apply VLM-based quality control to generated pairs, filtering them based on prompt alignment, video quality, cross-shot consistency, and source–target distinctiveness, rejecting 66% of pairs. For ControlNet-based pairs, we swap source and target when forming training pairs, using the structurally conditioned generation as input and the original generation as the reconstruction target, since the latter receives higher visual-quality scores.

## 3.2 Cascaded Architecture for Long-Form Video Editing

A common design for long-form video modeling is chunk-wise autoregressive difusion (Yin et al., 2025; Liu et al., 2026), where a video is divided into C fixed-size chunks $\scriptstyle { \pmb { x } } ^ { 1 : C }$ and modeled with a fixed-size sliding context window over

![](images/85c4cb53b3fd1762bcd1f2e53be3f45484b985d472e7b35570f4fcf109584f46.jpg)  
Figure 5 Cascaded modeling for a long-form video editor. MoviEdit uses a shared denoiser $f _ { \theta }$ to first produce a globally consistent, lightweight edit on a subsampled video sequence (Draft Editor), and then propagates the edit chunk by chunk to the full-resolution video while mitigating long-horizon appearance drift (Render Editor)

recent chunks. In causal difusion forcing (CDF) (Chen et al., 2024), each video chunk $\pmb { x } ^ { c }$ is independently corrupted at a sampled difusion timestep $t _ { c } ,$ yielding $\pmb { x } _ { t _ { c } } ^ { c } = \alpha _ { t _ { c } } \pmb { x } ^ { c } + \sigma _ { t _ { c } } \pmb { \epsilon } ^ { c }$ , and the denoiser is trained with

$$
\mathcal { L } _ { \mathrm { C D F } } ( \theta ) = \mathbb { E } _ { x , t , \epsilon } \left[ \sum _ { c = 1 } ^ { C } \left\| \left[ f _ { \theta } \big ( x _ { t } ^ { 1 : C } ; \mathcal { M } _ { \mathrm { c a u s a l } } \big ) \right] ^ { c } - v _ { t _ { c } } ^ { c } \right\| _ { 2 } ^ { 2 } \right] , \qquad t _ { c } \overset { \mathrm { i . i . d . } } { \sim } p ( t ) ,\tag{3}
$$

where ${ \pmb t } = ( t _ { 1 } , \dots , t _ { C } ) , { \pmb x } _ { t } ^ { 1 : C } = \{ { \pmb x } _ { t _ { c } } ^ { c } \} _ { c = 1 } ^ { C }$ and $\pmb { v } _ { t _ { c } } ^ { c }$ denotes the difusion target $( e . g .$ , velocity) for chunk c at timestep $t _ { c }$ The causal self-attention mask $\mathcal { M } _ { \mathrm { c a u s a l } }$ ensures that a standard DiT-based denoiser (Peebles & Xie, 2023) $f _ { \theta }$ predicts each chunk c using only itself and the preceding chunks within the retained context window. During inference, the same denoiser $f _ { \theta }$ is used within a chunk-wise causal autoregressive difusion sampler,

$$
\hat { \pmb x } ^ { c } = { \cal S } _ { f _ { \theta } } \left( \left[ \hat { \pmb x } ^ { c - K : c - 1 } ; \pmb x _ { T } ^ { c } \right] ; \mathcal { M } _ { \mathrm { c a u s a l } } \right) , \qquad \pmb x _ { T } ^ { c } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{4}
$$

where $\boldsymbol { \mathcal { S } } _ { f _ { \theta } }$ denotes a standard difusion sampler using $f _ { \boldsymbol { \theta } } , [ \cdot ; \cdot ]$ denotes temporal token concatenation, and $\scriptstyle { \hat { \pmb { x } } } ^ { c - K : c - 1 }$ denotes the preceding K generated chunks retained as autoregressive context.

However, the limited global context of chunk-wise autoregressive generation leads to cumulative appearance drift and oversaturation over long editing horizons (Fig. 4). To mitigate this issue, we augment the eficient CDF backbone with a lightweight bidirectional difusion stage that provides global guidance before full-resolution autoregressive editing. The Draft Editor first performs full-sequence bidirectional difusion over a spatially and temporally subsampled video representation t, producing a globally coherent edit draft. This draft serves two purposes: it provides stable appearance anchors that suppress long-horizon drift during autoregressive generation, and, by jointly resolving the edit over the entire sequence, establishes cross-shot consistency even between temporally distant shots. The Render Editor then performs full-resolution chunk-wise causal difusion conditioned on the draft, preserving this global editing structure while eficiently synthesizing local detail from the full-resolution input video chunks (Fig. 5). The Draft Editor and Render Editor share model weights, difering only in their attention patterns and spatiotemporal resolutions.

Let $\pmb { y } ^ { 1 : N }$ denote the N-frame source video and e the edit instruction. We apply a fixed spatiotemporal subsampling operator ↓ to obtain a shorter, lower-resolution video sequence $\pmb { y } _ { \downarrow } ^ { 1 : M } = \downarrow ( \pmb { y } ^ { 1 : N } )$ , where $M \ll N$ . Empirically, we find $M = N / 2$ to be efective.

Draft Editing. The Draft Editor applies $f _ { \theta }$ bidirectionally to the full subsampled sequence:

$$
\hat { \pmb { x } } _ { \downarrow } ^ { 1 : M } = S _ { f _ { \theta } } \left( \pmb { x } _ { T } ^ { 1 : M } ; \pmb { y } _ { \downarrow } ^ { 1 : M } , e ; \mathcal { M } _ { \mathrm { b i } } \right) , \qquad \pmb { x } _ { T } ^ { 1 : M } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{5}
$$

where $\mathcal { M } _ { \mathrm { b i } } = \mathbf { 0 }$ removes causal masking, allowing full bidirectional attention across the subsampled sequence. The resulting draft jointly resolves the edit over the entire video, providing a global appearance anchor for subsequent rendering.

Render Editing. Conditioned on this draft edit, the Render Editor applies $f _ { \theta }$ causally over fixed-size chunks $c = 1 , \ldots , C$ of the full-resolution target video,

$$
\hat { \pmb x } ^ { c } = \mathcal { S } _ { f _ { \theta } } \left( [ \hat { \pmb x } ^ { c - K : c - 1 } ; \pmb x _ { T } ^ { c } ] ; \hat { \pmb x } _ { \pm } ^ { 1 : M } , \pmb y ^ { c } , e ; \mathcal { M } _ { \mathrm { c a u s a l } } \right) , \qquad \pmb x _ { T } ^ { c } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{6}
$$

Category 8: Scene-level global edits

Category 7: Visual-medium robustness

Category 1: Single human

![](images/f45a63af31174df22258613b67d66b839fe2fb2e01bf7223e3bcaf598d7938f9.jpg)  
Category 3: Two co-visible humans  
Category 2: Two isolated humans

![](images/924acf9a1c638a9aa63258f700a6c752de3efed173a7a438d056404e014dd122.jpg)  
Category 4: Three humans

![](images/661ae532c096e3a2315ad046c401f9d3cc1f823b13f0f47eab8ca5354367c38c.jpg)  
Category 5: Four-or-more-person ensemble

![](images/164e426bc3ebe44e21997f75d19d109fa13113b213e1058d656969968218bb2b.jpg)

![](images/df55d1b3140317f57be6b80429af1dbf7cf872a0a7130580e1bc15df8e3ccaf6.jpg)  
Category 6: Human and animal

![](images/91473feba15990c07d673e833373e8446bfe639e98c0fdf234783908477a3f53.jpg)

![](images/e0ef037104533c77f5a05a7741db7082a15b743233d5465cbefc6eec9174e44c.jpg)

![](images/248510b3cdd5ac0ed096531a4f30ebf37bfc0d83f7432df1c80b7ab584e12aeb.jpg)  
Figure 6 Example source videos from MoviEdit-Bench. The examples highlight the diversity of subject composition, cross-shot grounding dificulty, and visual appearance in the benchmark. Full videos are available on our project page.

where $\hat { \pmb x } ^ { c - K : c - 1 }$ denotes the preceding K generated chunks retained as autoregressive context. Each chunk conditions on the recently generated context for temporal continuity, the global draft for long-range appearance and cross-shot consistency, and the corresponding source chunk $\pmb { y } ^ { c }$ to preserve content outside the intended edit.

During training, Draft Editor uses a conventional flow-matching objective (Lipman et al., 2022) with full bidirectional attention, while Render Editor is optimized with the CDF objective in Eq. 3.

## 3.3 Inference Optimization

Our architecture supports complementary techniques for reducing inference latency, including few-step difusion distillation and KV caching, where eficiency gains are reported in Tab. 4.

Unified Distillation with Asymmetric Step Schedules. After teacher training, we follow velocity moment matching (VMM) (Zhou et al., 2026; Salimans et al., 2024) to jointly distill the bidirectional Draft Editor and causal Render Editor under the shared-weight architecture. Following VMM, timestep pairs are sampled according to a predefined student inference schedule; we assign a separate schedule to each editing stage within the same distillation run. Specifically, the Draft Editor is distilled to 12 denoising steps, while the more computationally expensive Render Editor is distilled to 4 steps. We empirically find that the Render Editor is more robust to aggressive step reduction. This asymmetric allocation preserves suficient capacity for establishing global edit consistency while aggressively reducing the dominant rendering cost.

KV Caching. During Render Editor inference, the autoregressive formulation and causal attention masking naturally support KV caching. We cache key-value states for all reusable tokens, including text, edit draft, source video, and previously generated chunk tokens, eliminating recomputation.

## 3.4 MoviEdit-Bench: Evaluating Long-Form Multi-Shot Video Editing

Existing benchmarks for instruction-guided video editing (Bian et al., 2025; Li et al., 2025; Ju et al., 2026) primarily focus on short, single-shot clips and therefore do not explicitly evaluate cross-shot subject localization and editing consistency over long multi-shot sequences. To address this gap, we introduce MoviEdit-Bench, a benchmark for long-form cinematic multi-shot video editing under practical production settings, intended for public release. Benchmark construction and evaluation details are provided in Appx. B.

MoviEdit-Bench contains 64 generated multi-shot videos and 128 editing scenarios spanning cinematic and drama-style content. The videos are organized into eight categories with varying cross-shot subject-binding dificulty, ranging from a single recurring subject to crowded multi-subject and mixed human–animal scenes, alongside visual-medium robustness and scene-level global-edit cases (Fig. 6). The benchmark supports both text-guided video editing (V2V) and arbitrary-frame edited-frame-guided propagation (IV2V), with first-frame and non-first-frame reference scenarios. It covers both local edits (e.g., add, remove, appearance change, and swap) and global edits (e.g., environment change and stylization).

Evaluation Protocol. To capture the diverse failure modes of long-form multi-shot editing, our evaluation spans seven complementary dimensions: non-subject preservation, edit localization, subject preservation, cross-shot edit consistency, edit-prompt alignment, video quality, and, for IV2V only, edited-framefidelity. Across these dimensions, we combine pixel-level metrics, embedding-based similarity, and judgments from a vision-language model (VLM) (Team et al., 2026; Yang et al., 2025), following prior video editing benchmarks (Ju et al., 2026; Li et al., 2025; Bian et al., 2025). To account for multi-shot structure, source and edited videos are spatially and temporally aligned, with temporal metrics computed independently within each shot. The applicability of individual metrics varies by edit type, as detailed in Appx. B.4.

For frame-pair VLM evaluation, the judge examines uniformly sampled source and edited frame pairs from each shot, together with the edit instruction and target description, to produce fine-grained binary judgments on subject visibility, edit presence, correct subject localization, edit coverage, and preservation of unrelated content and non-edited subject attributes. Frames from shots where the subject is absent are also evaluated to detect unintended edits to other subjects or shots. For global edits, the corresponding judgments assess scene-level edit presence, coverage, and preservation of the underlying scene content. These criterion-level judgments are aggregated into VLM-based metrics for multiple evaluation dimensions, while cross-shot edit consistency is assessed separately by jointly evaluating edited frames across shots at the scenario level.

Evaluation Metrics. Non-subject preservation measures unintended changes outside the intended edit region using LPIPS (Zhang et al., 2018) between source and edited frames, excluding the subject region with segmentation masks. Edit localization evaluates whether modifications are confined to the intended subject and suficiently cover it, using VLM-based Precision and Recall, respectively. Precision measures preservation of unrelated content, including competing subjects and subject-absent shots, whereas Recall measures completeness of the edit on the subject. For global edits, Recall instead measures coverage of the entire scene.

Subject preservation assesses whether subject properties unrelated to the requested edit remain intact. We measure motion preservation using the Motion Fidelity Score (MFS) (Yatim et al., 2024), which compares tracked point displacements between source and edited videos independently within each shot. For applicable local edits, LPIPS measures preservation of the original subject appearance, while VLM judgments assess preservation of non-edited subject attributes. Cross-shot edit consistency evaluates whether the same edit is maintained across shots. The VLM judge jointly examines edited frames sampled from all relevant shots to assess consistency of the resulting appearance, object, identity, or scene transformation, independently of edit correctness.

Edit-prompt alignment measures correspondence between the edited video and the intended editing outcome using CLIP (Radford et al., 2021) and ViCLIP (Wang et al., 2024) similarities against per-shot target-video descriptions, together with VLM judgments of edit presence. Video quality is assessed independently of edit correctness through DINOv2 (Oquab et al., 2023) feature similarity between adjacent frames within each shot, NIQE (Mittal et al., 2013), PickScore (Kirstain et al., 2023), and VLM-based visual quality assessment. Finally, edited-framefidelity, applicable only to IV2V, measures whether the propagated edit preserves the identity and attributes established by the edited reference frame while allowing natural variations in viewpoint, pose, lighting, and motion.

Overall edit success summarizes editing performance by requiring each sampled frame to satisfy all applicable VLM criteria for edit presence, correct subject localization, edit coverage, preservation of unrelated content and non-edited subject attributes, and fidelity to the edited reference frame (IV2V only). Let $\mathcal { F }$ denote the set of sampled frames for a given editing scenario and $\mathcal { C } ( t )$ the set of applicable VLM criteria for frame t. We define

$$
\mathsf { o v e r a l l \_ e d i t \_ s u c c e s s } = \frac { 1 } { \left| \mathscr { F } \right| } \sum _ { t \in \mathscr { F } } \mathbf { 1 } \left[ \bigwedge _ { \mathfrak { c } \in \mathscr { C } ( t ) } a _ { \mathfrak { c } } ^ { t } \right] ,\tag{7}
$$

Table 1 Quantitative evaluation on the long-form multi-shot video editing benchmark, MoviEdit-Bench, for textguided video editing (V2V). (X-shot: cross-shot)
<table><tr><td rowspan="3" colspan="2">Model</td><td>Non-subject Preservation</td><td colspan="2">Edit Localization</td><td rowspan="2">Subject Preservation</td><td rowspan="2">X-shot Edit Consistency</td><td colspan="3">Edit Prompt Alignment</td><td colspan="4">Video Quality</td><td rowspan="2">Overall Edit Success</td></tr><tr><td>LPIPS↓</td><td>Precision↑Recall↑MFS↑LPIPS↓VLM↑</td><td></td><td>VLM↑</td><td>CLIP↑ViCLIP↑VLM↑DINO↑NIQE↓PickS.↑VLM↑</td><td></td><td></td><td></td><td></td></tr><tr><td>Opeue rce</td><td>SDEdit (HoloCine)</td><td>0.311</td><td>0.321 0.409</td><td>0.535</td><td>0.364 0.315</td><td>0.894</td><td>0.222</td><td>0.184</td><td>0.491</td><td>0.988</td><td>5.56</td><td>19.4</td><td>0.711</td><td>0.193</td></tr><tr><td></td><td>Memory-V2V</td><td>0.179</td><td>0.525</td><td>0.252 0.630</td><td>0.199 0.625</td><td>0.452</td><td>0.215</td><td>0.174</td><td>0.437</td><td>0.985</td><td>4.82</td><td>19.2</td><td>0.289</td><td>0.215</td></tr><tr><td></td><td>Lucy Edit</td><td>0.174</td><td>0.729</td><td>0.192 0.647</td><td>0.185 0.751</td><td>0.750</td><td>0.213</td><td>0.171</td><td>0.351</td><td>0.984</td><td>5.82</td><td>19.2</td><td>0.508</td><td>0.235</td></tr><tr><td></td><td>SANA-Streaming</td><td>0.141</td><td>0.651</td><td>0.430 0.651</td><td>0.212 0.707</td><td>0.490</td><td>0.219</td><td>0.181</td><td>0.592</td><td>0.984</td><td>5.48</td><td>19.3</td><td>0.504</td><td>0.354</td></tr><tr><td></td><td>Wan-Edit</td><td>0.0816</td><td>0.840</td><td>0.355 0.686</td><td>0.103 0.809</td><td>0.692</td><td>0.217</td><td>0.178</td><td>0.515</td><td>0.983</td><td>5.38</td><td>19.3</td><td>0.568</td><td>0.378</td></tr><tr><td></td><td>Gemini Omni Flash</td><td>0.153</td><td>0.705</td><td>0.834 0.568</td><td>0.193 0.684</td><td>0.893</td><td>0.221</td><td>0.194</td><td>0.834</td><td>0.987</td><td>4.95</td><td>19.8</td><td>0.814</td><td>0.586</td></tr><tr><td>Pivate</td><td>HappyHorse-1.0</td><td>0.246</td><td>0.729</td><td>0.788 0.400</td><td>0.291 0.728</td><td>0.894</td><td>0.219</td><td>0.193</td><td>0.815</td><td>0.984</td><td>5.12</td><td>19.6</td><td>0.691</td><td>0.623</td></tr><tr><td>Aleph 2.0</td><td></td><td>0.0916</td><td>0.886</td><td>0.778 0.671</td><td>0.146 0.883</td><td>0.981</td><td>0.225</td><td>0.186</td><td>0.846</td><td>0.988</td><td>4.95</td><td>19.6</td><td>0.801</td><td>0.764</td></tr><tr><td>MoviEdit (Ours)</td><td></td><td>0.0741</td><td>0.904</td><td>0.861 0.695</td><td>0.121 0.911</td><td>0.904</td><td>0.227</td><td>0.190</td><td>0.884 0.989</td><td></td><td>4.85</td><td>19.7</td><td>0.762</td><td>0.790</td></tr></table>

Table 2 Quantitative evaluation on the long-form multi-shot video editing benchmark, MoviEdit-Bench, for editedframe-guided propagation (IV2V). (X-shot: cross-shot)
<table><tr><td rowspan="2">Edited Frame</td><td rowspan="2">Model</td><td>Non-subject Preservation</td><td colspan="2">Edit Localization</td><td>Subject Preservation</td><td>X-shot Edit Consistency</td><td colspan="2">Edit Prompt Alignment</td><td colspan="2">Edited-frame Fidelity</td><td colspan="3">Video Quality</td><td>Overall Edit Success</td></tr><tr><td>LPIPS↓</td><td></td><td>Precision↑Recall↑MFS↑LPIPS↓VLM↑</td><td>VLM↑</td><td></td><td>CLIP↑ViCLIP↑VLM↑</td><td></td><td>VLM↑</td><td></td><td>DINO↑NIQE↓PickS.↑VLM↑</td><td></td><td></td><td>VLM↑</td></tr><tr><td rowspan="5">irst</td><td>HappyHorse-1.0</td><td>0.244</td><td>0.751</td><td>0.8280.4170.304</td><td>0.767 0.925</td><td>0.220</td><td>0.195</td><td>0.841</td><td>0.680</td><td>0.985</td><td>5.08</td><td>19.6</td><td>0.707</td><td>0.522</td></tr><tr><td>GenProp</td><td>0.0788</td><td>0.976</td><td>0.671 0.698 0.104</td><td>0.585</td><td>0.232</td><td>0.186</td><td>0.761</td><td>0.561</td><td>0.989</td><td>3.96</td><td>19.7</td><td>0.789</td><td>0.581</td></tr><tr><td>Aleph 2.0</td><td>0.0412</td><td>0.962</td><td>0.974 0.7020.0895 0.985</td><td>0.993 0.906</td><td>0.231</td><td>0.192</td><td>0.971</td><td>0.819</td><td>0.989</td><td>4.90</td><td>19.8</td><td>0.801</td><td>0.806</td></tr><tr><td>MoviEdit (Ours)</td><td>0.0586</td><td>0.965</td><td>0.938 0.699 0.107</td><td>1.00 0.981</td><td>0.231</td><td>0.191</td><td>0.938</td><td>0.828</td><td>0.989</td><td>4.67</td><td>19.8</td><td>0.797</td><td>0.803</td></tr><tr><td>HappyHorse-1.0</td><td>0.248</td><td></td><td>0.793</td><td>0.760</td><td></td><td></td><td></td><td>0.636</td><td></td><td></td><td>19.6</td><td>0.709</td><td></td></tr><tr><td rowspan="2">Arbiary</td><td>Aleph 2.0</td><td>0.0422</td><td>0.742 0.955</td><td>0.4020.307 0.952 0.7050.08380.981</td><td>0.913 0.865</td><td>0.219 0.229</td><td>0.194 0.193</td><td>0.813 0.957</td><td>0.801</td><td>0.984 0.989</td><td>5.11 4.94</td><td>19.8</td><td></td><td>0.486</td></tr><tr><td>MoviEdit (Ours)</td><td>0.0598</td><td>0.963</td><td>0.920 0.699 0.0996 0.995</td><td>0.933</td><td>0.229</td><td>0.192</td><td>0.925</td><td>0.805</td><td>0.989</td><td>4.78</td><td>19.7</td><td>0.807 0.807</td><td>0.781 0.786</td></tr></table>

where $a _ { c } ^ { t }$ denotes the VLM judge’s binary verdict for criterion c at sampled frame t. The score is computed using the VLM judgments from the individual evaluation dimensions, without additional queries.

## 4 Experiments

## 4.1 Experimental Setup

Implementation Details. Using the three data-generation pipelines described above, we construct 87K paired multi-shot editing videos (540p, 15s, 24 fps) from 61,251 raw clips across three data sources (Tab. 3). In addition to this multi-shot dataset, we incorporate a proprietary collection of short, single-shot video editing pairs (540p, 5s, 24fps) during training. Our model uses a DiT-based mixture-of-experts (MoE) backbone with a frozen multimodal text encoder and video VAE. The input sequence concatenates text, source-video, noisy target-video, and task-specific conditioning tokens, including edited-frame and draft-edit conditions. We initialize from a pretrained short-form, single-shot video editing checkpoint with no prior multi-shot generation or editing capability. We fully fine-tune all DiT parameters using Fully Sharded Data Parallel (FSDP) across 64 NVIDIA H200 GPUs. Specifically, training proceeds in two stages. We first train the video-to-video (V2V) model for 17K steps. We then specialize the model for IV2V with an additional 8K training steps under the IV2V task formulation. For this stage, edited reference frames are sampled from the ground-truth target video and restricted to shots in which the edited subject is present.

Evaluations. We evaluate MoviEdit and competing methods on MoviEdit-Bench under both text-guided (V2V) and edited-frame-guided (IV2V) settings. V2V models receive only the source video and edit instruction, while IV2V models additionally receive an edited reference frame; neither setting provides object masks. MoviEdit-Bench includes both first-frame and arbitrary-frame IV2V scenarios. For fair comparison, we evaluate all IV2V-capable models on the first-frame subset, and methods supporting arbitrary-frame conditioning on the full benchmark. As no existing, fully accessible public benchmark specifically targets long-form multi-shot editing, we additionally evaluate V2V performance on the established short-form benchmarks FiVE-Bench (Li et al., 2025) and VPBench (Bian et al., 2025). For VPBench, which is originally mask-conditioned, we evaluate MoviEdit in a mask-free setting using only the source video and edit instruction. We compare against various public and proprietary systems, including Lucy Edit Dev (DecartAI Team, 2025), Memory-V2V (Lee et al., 2026), SANA-Streaming (Zhao et al., 2026b), Wan-Edit (Li et al., 2025), SDEdit with

![](images/7b30b1b42326c83e6296c83df5641790647cb75f9b3b8a47e7115394d8c9e8c3.jpg)  
“Change the man’s charcoal robe to a mustard hoodie."

Figure 7 Qualitative comparison on MoviEdit-Bench with representative baselines.

![](images/a0b53cfb92367cf1696e80a1fba571ee1a3e6d182b5d2a1aa9db44ff4b2d82bb.jpg)  
Figure 8 MoviEdit-Bench radar chart showing that MoviEdit achieves perfor- Figure 9 MoviEdit naturally extends mance on par with proprietary systems. to video set-to-set editing setup.

HoloCine (Meng et al., 2022, 2026), HappyHorse-1.0 (Alibaba Cloud, 2026), Aleph 2.0 (Runway, 2026), Gemini Omni Flash (Google DeepMind, 2026), and GenProp (Liu et al., 2025).

## 4.2 Results

Comparisons. We compare MoviEdit with publicly available and proprietary systems on 15-second multi-shot editing using MoviEdit-Bench. For V2V, MoviEdit achieves the strongest overall performance among the evaluated systems, surpassing Runway Aleph 2.0 in overall edit success and leading on the majority of evaluation metrics (Tab. 1). For IV2V, MoviEdit achieves the highest overall success on the full benchmark and performs on par with Aleph 2.0 across the individual evaluation dimensions, with complementary strengths across diferent metrics (Tab. 2). The radar plot in Fig. 8 further highlights the balanced performance of MoviEdit across the major benchmark criteria. We further evaluate MoviEdit on the established FiVE-Bench and VPBench as external validation (Tab. 9), obtaining results consistent with those observed on MoviEdit-Bench.

Fig. 7 shows qualitative V2V and IV2V comparisons. Public baselines exhibit pronounced failure modes, including artifacts propagated from previous shots, incorrect cross-shot edit localization, edit failure, and poor source consistency. HappyHorse and Gemini Omni generally perform the requested edits well but systematically deviate from the source, failing to preserve shot transitions and timing and hallucinating motion absent from the source. Aleph 2.0 provides the strongest baseline results, while MoviEdit shows better source preservation, edit localization, and prompt alignment. These trends are consistent with quantitative results on MoviEdit-Bench.

Scaling and Eficiency Analysis. We study MoviEdit’s scaling behavior and inference eficiency. For scaling, we separately vary training duration and dataset size: 0%, 20%, 40%, 80%, and 100% of the 17K-step schedule, and 0%,

Table 3 Training dataset statistics at 540p, 10-15s, and 4-6 shots per video.
<table><tr><td>Task</td><td>Source</td><td># raw pairs</td><td># filtered pairs</td></tr><tr><td>Localization</td><td>Multi-cam</td><td>13,801</td><td>10,500</td></tr><tr><td>Pixel-aligned</td><td>Multi-cam</td><td>38,979</td><td>21,008</td></tr><tr><td></td><td>Cinematic</td><td>138,633</td><td>50,405</td></tr><tr><td>Semi-pixel-aligned</td><td>Generative</td><td>13,815</td><td>4,706</td></tr><tr><td>Total</td><td></td><td>205,228</td><td>86,619</td></tr></table>

Table 4 Inference eficiency ablation at 540p and 361 frames on 8 H100 GPUs (seconds).
<table><tr><td>Variant</td><td>Draft Editor Render Editor</td><td></td><td>Total</td></tr><tr><td>Full temporal attn.</td><td>25.1</td><td>339.0</td><td>364.1 (×1.0)</td></tr><tr><td>+ Sliding-window attn.</td><td></td><td>233.8</td><td>258.9 (×1.4)</td></tr><tr><td>+ KV cache</td><td></td><td>141.7</td><td>166.8 (×2.2)</td></tr><tr><td>+ Distillation</td><td>3.6</td><td>8.2</td><td>11.8 (×30.9)</td></tr></table>

![](images/c8b0723ceb4230a22afc8ba2aea5779d1129068afdbb5f1bf08647582c01b9e2.jpg)  
Figure 10 Scaling study on (a) training steps and (b) dataset size. X-axes show fractions of the full schedule (17K steps) / full dataset (87K pairs) used. Shading denotes 95% confidence intervals.

![](images/16e867c7540121f21ae7c213a5a333c49c887a3fa314e549139caf4ae58fd8a1.jpg)  
Figure 11 Ablation of task-specific data recipes. Top: Adding semi-pixel-aligned training data improves generalization to edits involving relatively larger changes in structure or shape. Bottom: Adding subject localization training data improves the model’s ability to identify the intended edit target, leading to improved edit localization.

33%, and 100% of the 87K-pair dataset, respectively. Increasing either dataset size or training duration consistently improves performance across metrics (Fig. 10), suggesting further gains from scaling. We next ablate inference eficiency at 540p and 361 frames on 8 H100 GPUs (Tab. 4). Starting from full temporal attention (364.1 s), sliding-window attention reduces denoising latency to 258.9 s, and KV caching further reduces it to 166.8 s. Combined with few-step distillation, the full system runs in 11.8 s, yielding a 30.9× speedup over the full-attention baseline. Together, these optimizations reduce the denoising latency to 11.8 s, below the 15 s input-video duration, substantially improving the practicality of long-form multi-shot editing.

Data Recipe Ablations. To examine the efect of our task-specific data recipes, we conduct small-scale ablation training by selectively removing diferent supervision pipelines. Fig. 11 shows qualitative comparisons for subject-localization and semi-pixel-aligned training data. Semi-pixel-aligned data improves generalization to edits that require relatively larger changes in structure or shape. As shown in the top part of Fig. 11, the model trained with semi-pixel-aligned data better handles such edits. This complements our pixel-aligned data pipeline, which has dificulty constructing source–target editing pairs involving large structural or shape changes. Adding subject localization data improves the model’s ability to identify the intended edit target. As shown in the bottom part of Fig. 11, the model trained with subject localization supervision exhibits improved edit localization compared with the variant trained without it. This observation is consistent with prior findings in video editing (Liu et al., 2025), where task-specific tracking supervision has been shown to benefit editing performance.

Applications. We explore applications beyond standard V2V and IV2V multi-shot editing, including settings not seen during training. First, MoviEdit supports multi-reference-frame editing (Fig. 1(b)); although trained with a single edited reference frame, it can condition on multiple edited references at inference time, combine edits to diferent subjects, and propagate them consistently. Second, we extend image set-to-set editing (Feingold et al., 2026) to videos with video set-to-set editing (Fig. 9). We concatenate multiple source videos into a single multi-shot sequence and edit them jointly, yielding consistent transformations despite variations in appearance and background. Finally, we evaluate beyond-training-horizon editing (Fig. 1(c)). Although trained on videos up to 15 seconds, MoviEdit generalizes to 25-second inputs with up to 17 shots, as well as long-take single-shot videos, extending both temporal duration and shot count beyond the training regime.

Limitations and Failure Cases. Fig. 12 highlights two challenging target-localization failure modes. In Fig. 12(a), multiple visually similar subjects create identity ambiguity, which can cause the edit to be applied to the wrong target. In

(a) Inter-Subject Identity Ambiguity  
![](images/a95407330801866987978cd71b50a13a676b036027d8e6d16af85db2d85eeda4.jpg)  
“Add large safety goggles to the man wearing a grey shirt and a yellow safety vest.”

(b) Partial-Observation Ambiguity  
![](images/0bf0ad9effdab806919330aaa7d67d0445103f7ac5441680eba759f2620292ca.jpg)  
Figure 12 Failure cases of MoviEdit. (a) Inter-subject identity ambiguity. When multiple visually similar subjects are present, the model may mistakenly apply the edit to an unintended subject rather than the specified target. (b) Partial-observation ambiguity. When the target is only partially visible, the model may struggle to accurately localize and edit the intended subject.

Fig. 12(b), partial target visibility (e.g., only a body part or object fragment is visible) can make cross-shot association dificult, leading to incomplete or incorrect edits.

## 5 Conclusion

We presented MoviEdit, a unified framework for long-form multi-shot video editing that jointly addresses scalable supervision, eficient modeling, and dedicated evaluation. MoviEdit combines scalable pipelines for generating paired multi-shot editing data, cascaded bidirectional–autoregressive modeling, and MoviEdit-Bench for evaluating cross-shot localization, consistency, preservation, and edit alignment. Across text- and image-guided editing, MoviEdit substantially outperforms existing open-source models and matches strong proprietary systems, while generalizing to set-to-set editing and videos beyond the training horizon. Regarding limitations, MoviEdit can struggle with target localization under strong identity ambiguity, such as when multiple visually similar subjects are present or the target is only partially visible. These limitations motivate several directions for future work. Agentic editing workflows could further improve edit fidelity and computational eficiency, for example by selectively processing only the shots or subjects that require editing. On the data side, controllable rendering engines (Engine, 2018) ofer another scalable source of supervision. Finally, MoviEdit focuses on correspondence-preserving edits that retain the source motion and narrative structure; extending multi-shot long video editing to free-form transformations of motion and narrative structure remains an important future direction.

## References

Alibaba Cloud. Happyhorse video editing api reference. https://www.alibabacloud.com/help/en/model-studio/ happyhorse-video-edit-api-reference, 2026.

Jianhong Bai, Menghan Xia, Xintao WANG, Ziyang Yuan, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di ZHANG. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 58038–58060, 2025. URL https://proceedings. iclr.cc/paper\_files/paper/2025/file/9232d474be0a4e5f1e1bcb0765f17f9a-Paper-Conference.pdf.

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 37971–37981, 2026.

Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Any-length video inpainting and editing with plug-and-play context control. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–12, 2025.

Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. In International Conference on Learning Representations, volume 2026, pp. 92075–92093, 2026.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pp. 138846–138923, 2026.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image difusion. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 23206–23217. IEEE, 2023.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Zhou Daquan, Huan Ling, et al. Sana-video: Eficient video generation with block linear difusion transformer. In International Conference on Learning Representations, volume 2026, pp. 38973–38997, 2026

Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3151–3161. IEEE, 2024.

DecartAI Team. Lucy edit: Open-weight text-guided video editing. Technical report, 2025. URL https://huggingface.co/ decart-ai/Lucy-Edit-1.1-Dev. Model: Lucy-Edit-1.1-Dev.

Unreal Engine. Unreal engine. Retrievedfrom Unreal Engine: https://www.unrealengine.com/en-US/what-is-unreal-engine-4, 2018.

Kate Feingold, Omri Kaduri, and Tali Dekel. Match-and-fuse: Consistent generation from unstructured image sets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 30490–30499, 2026.

Kunyu Feng, Yue Ma, Bingyuan Wang, Yuefeng Wang, Zhiyuan Qin, Hao Cheng, Hao Li, Qifeng Chen, and Zeyu Wang. Mseditor: Toward consistent multi-shot video editing. arXiv preprint arXiv:2608.17559, 2026.

Michal Geyer, Omer Bar Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent difusion features for consistent video editing. In International Conference on Learning Representations, volume 2024, pp. 1608–1620, 2024.

Google DeepMind. Gemini omni flash. https://deepmind.google/models/model-cards/gemini-omni-flash/, 2026.

Xu Guo, Zhengxuan Wei, Xinghui Li, Hanzhuo Huang, Xinyu Liu, Xiangyang Luo, Min Wei, Yiran Zhu, Qiulin Wang, Yulong Xu, et al. Contextmaster: Interactive multi-shot video creation via fixed-budget sparse context routing. arXiv preprint arXiv:2608.04956, 2026.

Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17281–17291, 2025.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 167283–167308. Curran Associates, Inc., 2025a. doi: 10.52202/085713-5576. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ f4823f831af67a3ef15e41a85434422a-Paper-Conference.pdf.

Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y Wang, Joan Lasenby, and Chun-Hao Huang. Spacetimepilot: Generative rendering of dynamic scenes across space and time. arXiv preprint arXiv:2512.25075, 2025b.

Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zero-shot grounded video editing using text-to-image difusion models. arXiv preprint arXiv:2310.01107, 2023.

Hyeonho Jeong, Jinho Chang, Geon Yeong Park, and Jong Chul Ye. Dreammotion: Space-time self-similar score distillation for zero-shot video editing. In European Conference on Computer Vision, pp. 358–376. Springer, 2024a.

Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video difusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9212–9221. IEEE, 2024b.

Hyeonho Jeong, Suhyeon Lee, and Jong Chul Ye. Reangle-a-video: 4d video generation as video-to-video translation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11164–11175. IEEE, 2025.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17191–17202. IEEE, 2025.

Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Cherry Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. In International Conference on Learning Representations, volume 2026, pp. 137234–137255, 2026.

Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labeling real videos. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6013–6022. IEEE, 2025.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

Dohun Lee, Chun-Hao Paul Huang, Xuelin Chen, Jong Chul Ye, Duygu Ceylan, and Hyeonho Jeong. Memory-v2v: Memoryaugmented video-to-video difusion for consistent multi-turn editing. In European Conference on Computer Vision, pp. 75–93. Springer, 2026.

Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, and Mengyu Wang. Five-bench: A fine-grained video editing benchmark for evaluating emerging difusion and rectified flow models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16672–16681. IEEE, 2025.

Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video difusion in real time. In International Conference on Learning Representations, volume 2026, pp. 91177–91196, 2026.

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8599–8608. IEEE, 2024.

Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, et al. Generative video propagation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17712–17722. IEEE, 2025.

Fuchen Long, Cong Wang, Zitao Gao, Wenhao Zhong, Yu Cheng, Xiaolu Hou, Yan Li, Xiao Cao, Xinlong Sun, Xi Chen, et al. Coinve-200k: A large-scale high-quality dataset for compositional instruction-guided video editing. arXiv preprint arXiv:2608.17566, 2026.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic diferential equations. In 10th International Conference on Learning Representations (ICLR), 2022.

Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Shuailei Ma, Yixuan Li, Cheng Chen, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 461–471, 2026.

Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2013.

Jonghwan Mun, Minchul Shin, Gunsoo Han, Sangho Lee, Seongsu Ha, Joonseok Lee, and Eun-Sol Kim. Bassl: Boundary-aware self-supervised learning for video scene segmentation. In Asian Conference on Computer Vision, pp. 485–501. Springer, 2022.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First-frame-guided video editing via image-to-video difusion models. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

William Peebles and Saining Xie. Scalable difusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205. IEEE, 2023.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15932–15942. IEEE, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Runway. Runway aleph 2.0. https://runway.com/news/introducing-aleph-2-and-edit-studio, 2026.

Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of difusion models via moment matching. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 36046–36070. Curran Associates, Inc., 2024. doi: 10.52202/079017-1136. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 3f66d5cdbe032bb750f2dc523357b7a5-Paper-Conference.pdf.

Alp Eren Sari and Paolo Favaro. Flowcut: Unsupervised video instance segmentation via temporal mask matching. arXiv preprint arXiv:2505.13174, 2025.

Wonyong Seo, Jaeho Moon, Jaehyup Lee, Soo Ye Kim, and Munchurl Kim. Propfly: Learning to propagate via on-the-fly supervision from pre-trained video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43228–43238, 2026.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, et al. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

Boyang Wang, Guangyi Xu, Jiahui Zhang, Zhipeng Tang, and Zezhou Cheng. Omnishotcut: Holistic relational shot boundary detection with shot-query transformer. arXiv preprint arXiv:2604.24762, 2026.

Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using of-the-shelf image difusion models. arXiv preprint arXiv:2303.17599, 2023.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In International Conference on Learning Representations, volume 2024, pp. 42055–42079, 2024.

Chenyang Wu, Fuchen Long, Binyuan Huang, Xinlong Sun, Xi Chen, Chun-Le Guo, and Chongyi Li. Thinking on shots: Consistent multi-shot video editing with agentic reasoning. arXiv preprint arXiv:2608.26809, 2026.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image difusion models for text-to-video generation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7623–7633. IEEE, 2023.

Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Efective instruction-based video editing with elaborate dataset construction. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16692–16701. IEEE, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned difusion for unified video generation and editing. arXiv preprint arXiv:2602.08820, 2026a.

Xiangpeng Yang, Ji Xie, Yiyuan Yang, Yue Ma, Yan Huang, Min Xu, and Qiang Wu. Videocof: Unified video editing with temporal reasoner. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 37940–37949, 2026b.

Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time difusion features for zero-shot text-driven motion transfer. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8466–8476. IEEE, 2024.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22963–22974. IEEE, 2025.

Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image difusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3836–3847. IEEE, 2023.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive difusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026a.

Yuyang Zhao, Yicheng Pan, Qiyuan He, Jincheng Yu, Junsong Chen, Tian Ye, Haozhe Liu, Enze Xie, and Song Han. Sana-streaming: Real-time streaming video editing with hybrid difusion transformer. arXiv preprint arXiv:2605.30409, 2026b.

Yuqian Zhou, Zhenghong Zhou, Zongze Wu, Cameron Smith, Richard Zhang, Jiebo Luo, Eli Shechtman, and Zhe Lin. Editstream: A unified autoregressive framework for interactive video generation and editing. arXiv preprint arXiv:2608.21424, 2026.

Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-0506. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 160e9ef249b9a836b7a82d342698af7a-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

## Supplementary Material

This supplementary material provides additional details and results to complement the main paper. We first provide additional details on our training dataset construction (Sec. A) and MoviEdit-Bench (Sec. B), including its design and evaluation metrics. Finally, we present additional experimental results on external benchmark comparisons (Sec. C.1) and beyond-training-horizon editing (Sec. C.2).

## A Training Dataset Construction

Raw Data Sources. We collect raw video clips from three sources: synchronized multi-camera footage (1,147 clips), cinematic footage (32,474 clips), and generative videos (27,630 clips). In particular, the synchronized multi-camera footage captures the same real-world scene simultaneously from 2–4 cameras, with actors performing natural interactions. This setup is conceptually similar to the synthetic SynCamMaster setup (Bai et al., 2025), but uses real-world footage. Representative examples are shown in Fig. 13.

Editing Pair Generation. Using these raw data sources, we construct the MoviEdit training dataset through three data-generation pipelines: localization, pixel-aligned, and semi-pixel-aligned. The resulting dataset contains 86,619 paired multi-shot videos at 540p resolution, approximately 15 seconds in duration, and 24 fps. The detailed breakdown by pipeline, data source, and editing task is provided in Table 5.

## B MoviEdit-Bench

## B.1 Source Multi-Shot Videos

MoviEdit-Bench comprises 64 source videos, primarily covering cinematic, story-driven multi-shot scenes with coherent space, time, and subject identity across shots. The scenes span diverse settings, including single-subject activities, interpersonal interactions, larger ensembles, and human–animal interactions, with deliberate variation in pose, viewpoint, visibility, and camera framing to support cross-shot editing evaluation.

The source videos are organized into eight high-level categories (Tab. 6). Categories C1–C6 progressively increase the dificulty of grounding an edit instruction to the correct subject across shots, ranging from a single persistent subject to crowded multi-subject and human–animal scenes. C3 is further divided into same-gender and diferent-gender two-person settings to examine diferent levels of subject ambiguity. C7 evaluates robustness across non-photorealistic visual media, while C8 targets scene-level global edits without entity localization.

Each scenario is authored by an LLM as a structured specification containing a global scene description, per-shot descriptions, and associated metadata. We generate each source video from scratch using a multi-shot text-to-video model, avoiding reliance on copyrighted source footage. Each video contains four to six shots and 361 frames at 24 fps, corresponding to approximately 15 seconds. The prompts, metadata, and generated videos are then manually reviewed for scene coherence, subject identity, shot correspondence, and suitability for editing evaluation.

Synchronized Multi-Camera Footage  
![](images/78ac10c1762223ae88f54738b65262fa9ad99d75c92e75a852ca79dfb98370af.jpg)  
Figure 13 Examples of synchronized multi-camera footage used as raw data for MoviEdit. Each scene is captured simultaneously from 2–4 cameras, showing the same real-world scene from diferent viewpoints while actors perform natural interactions. Note: Faces ofreal actors in the displayed examples are blurred to protect their identity.

<table><tr><td>Pipeline</td><td>Data source</td><td>Editing task</td><td># Pairs</td></tr><tr><td rowspan="4">Localization</td><td rowspan="4">Multi-camera</td><td>Highlighting</td><td>3,500</td></tr><tr><td>Outlining</td><td>3,500</td></tr><tr><td>Color filling</td><td>3,500</td></tr><tr><td>Subtotal</td><td>10,500</td></tr><tr><td rowspan="9">Pixel-aligned</td><td rowspan="9">Multi-camera</td><td>Add</td><td>2,162</td></tr><tr><td>Remove</td><td>3,209</td></tr><tr><td>Appearance change</td><td>4,530</td></tr><tr><td>Swap</td><td>2,926</td></tr><tr><td>Stylization</td><td>3,841</td></tr><tr><td>Environment change</td><td>4,340</td></tr><tr><td>Subtotal</td><td>21,008</td></tr><tr><td>Add</td><td>6,787</td></tr><tr><td>Remove</td><td>11,921</td></tr><tr><td rowspan="5">Cinematic</td><td>Appearance change</td><td></td><td>14,206</td></tr><tr><td>Swap</td><td></td><td>8,742</td></tr><tr><td>Stylization</td><td></td><td>1,606</td></tr><tr><td>Environment change</td><td></td><td>7,143</td></tr><tr><td>Subtotal</td><td></td><td>50,405</td></tr><tr><td rowspan="4">Semi-pixel-aligned</td><td rowspan="4">1 Generative</td><td>Environment change</td><td>1,364</td></tr><tr><td>Medium change</td><td>1,368</td></tr><tr><td>Character appearance change</td><td>1,974</td></tr><tr><td>Subtotal</td><td>4,706</td></tr><tr><td></td><td></td><td>Total</td><td>86,619</td></tr></table>

Table 5 Detailed composition of the MoviEdit training dataset by data-generation pipeline, data source, and editing task.

## B.2 Editing Scenarios

We construct two editing scenarios for each source video, yielding 128 scenarios in total. The 48 entity-centric videos in C1–C7 provide local edits spanning add, remove, appearance change, and swap, with a 3:1 ratio between subject targets (characters or animals) and object targets. The 16 videos in C8 provide scene-level global edits covering environment change and stylization. Overall, local and global edits account for 75% and 25% of the benchmark, respectively; the detailed task distribution is given in Tab. 7.

Editing scenarios and instructions are authored by an LLM from the source-video descriptions and structured entity metadata, following the predefined task distribution. All scenarios and edit instructions are subsequently manually reviewed for target correctness, instruction clarity, and suitability for evaluation.

For edited-frame-guided propagation, MoviEdit-Bench supports arbitrary edited-frame conditioning. We divide the 128 scenarios evenly into 64 first-frame and 64 non-first-frame reference settings. For each scenario, a suitable reference frame is manually selected, ensuring visibility of the edited target when applicable, and edited with an of-the-shelf text-guided image editor using the corresponding edit instruction.

## B.3 Evaluation Metadata

Beyond the inputs required for benchmark evaluation, including the source video, edit instruction, and edited reference frame for the IV2V task, MoviEdit-Bench provides additional metadata for metric computation. For local edits, we provide per-frame masks of the editing target, obtained using video segmentation models (Cheng et al., 2024; Sari & Favaro, 2025; Carion et al., 2026) followed by manual verification and correction. The masks are temporally aligned with the source video and paired with the corresponding edit instruction. We also provide shot-boundary metadata for each source video, represented as the start and end frame indices of each detected shot using OmniShotCut (Wang et al., 2026) for shot-level metric computation.

<table><tr><td>Category</td><td>Composition</td><td></td><td># Videos Source-video design</td><td>Primary evaluation focus</td></tr><tr><td>C1</td><td>Single human</td><td>6</td><td>One persistent subject observed across varying actions, poses, framings, and viewpoints.</td><td>Cross-shot localization under pose and viewpoint changes.</td></tr><tr><td>C2</td><td>Two isolated humans</td><td>6</td><td>Two humans belong to the same scene, while Target selection across shots and cor- each shot contains only one of them.</td><td>rect exclusion when the target is ab- sent.</td></tr><tr><td>C3A</td><td>Two co-visible humans, same gender</td><td>6</td><td>Same-gender subjects are distinguished by roles, Subject disambiguation between vi- clothing, hairstyles, accessories, positions, or sually and semantically competing object ownership.</td><td>humans.</td></tr><tr><td>C3B</td><td>Two co-visible humans, different gender</td><td>6</td><td>Different-gender subjects co-occur within shots Cross-shot subject binding with mul- and are distinguished by roles and visual at- tiple co-visible candidates. tributes.</td><td></td></tr><tr><td>C4</td><td>Three humans</td><td>6</td><td>Three text-addressable characters share a con- Localization under increased multi- tinuous scene with distinct visual or semantic person competition.</td><td></td></tr><tr><td>C5</td><td>Four-or-more-person en- semble</td><td>6</td><td>identifiers. Larger ensembles contain distinguishable roles, Target localization in crowded multi- appearance, props, and spatial positions.</td><td>person scenes.</td></tr><tr><td>C6</td><td>Human and animal</td><td>6</td><td>A human and an animal coexist across a contin- Cross-species target grounding and uous multi-shot scene.</td><td>identity persistence.</td></tr><tr><td>C7</td><td>Visual-medium robust- ness</td><td>6</td><td>Scenes span non-photorealistic media such as Robustness of localization and edit- stylized 3D, stop-motion, watercolor, claymation, ing across visual media.</td><td></td></tr><tr><td>C8</td><td>Scene-level global edits</td><td>16</td><td>anime, and paper-cut styles. such as stylization, lighting, weather, and time- localization constraints.</td><td>Scenes are designed for global transformations Global editing without subject-</td></tr><tr><td></td><td>Total</td><td>64</td><td>of-day changes.</td><td></td></tr></table>

Table 6 Overview of the MoviEdit-Bench source-video categories. C1–C6 progressively vary cross-shot subject-grounding dificulty, C7 evaluates robustness across visual media, and C8 targets scene-level global editing.

Each editing scenario is further accompanied by a target-video prompt. Following the same format as the source-video generation prompt, it contains a global scene description together with per-shot descriptions. The target prompt is authored as a self-contained description of the fully edited target video, rather than as a relative modification of the source prompt, and is manually reviewed for consistency with the intended edit. These target prompts are also used to evaluate baselines formulated with target-video conditioning rather than standard edit instructions, such as Wan-Edit (Li et al., 2025) and SDEdit (Meng et al., 2022).

## B.4 Evaluation Metrics

We evaluate each edited video along seven complementary dimensions: non-subject preservation, edit localization, subject preservation, cross-shot edit consistency, edit-prompt alignment, video quality, and edited-framefidelity. We additionally report an overall success score that jointly considers multiple editing criteria. Each metric produces one score per editing scenario, and model-level results are averaged over applicable scenarios. Metric applicability varies by edit type, as summarized in Tab. 8.

Evaluation protocol. Methods may difer in output resolution and duration. We resize the source video, edited video, and target masks to a common 960 × 544 resolution and temporally align them to the 361-frame source video. When a method produces fewer frames, we evaluate only the overlapping range. The shot boundaries provided with MoviEdit-Bench define the units for shot-level evaluation, and temporal computations are performed independently within each shot. For mask-based metrics, we dilate the target masks by 9 pixels to exclude ambiguous boundary regions and omit frames or shots in which the target is not visible.

VLM judge. We use a VLM (Team et al., 2026; Yang et al., 2025) to evaluate semantic properties that are dificult to capture with pixel- or embedding-level similarity, such as whether an edit is applied to the correct subject. Unless otherwise specified, VLM-based metrics share a common frame-pair protocol. We uniformly sample four frames from each shot, including the first and last frames. For each sample, the judge receives the source frame, edited frame, edit instruction, and target description, and independently determines whether: (i) the target is visible in the source frame, (ii) the requested edit is present, (iii) the edit is applied to the intended target, (iv) unrelated content is preserved, (v) target attributes unrelated to the instruction are preserved, and (vi) the edit completely covers the target. Frames are sampled from all shots, including those in which the target is absent, enabling the judge to detect edits that incorrectly leak to other subjects or shots. For global edits, the corresponding checks assess whether the requested scene-level change is present, covers the entire scene, and preserves the underlying scene content. All VLM metrics below reuse these judge outputs unless otherwise noted. Each binary VLM verdict is mapped to 1 for true and 0 for false. Frame-level VLM metrics are averaged over all applicable sampled frames, while scenario-level judgments, such as cross-shot consistency, directly yield a binary scenario score.

<table><tr><td>Edit task</td><td>Scope</td><td># Scenarios</td><td>Ratio</td></tr><tr><td>Add</td><td>Local</td><td>24</td><td>18.75%</td></tr><tr><td>Remove</td><td>Local</td><td>24</td><td>18.75%</td></tr><tr><td>Appearance change</td><td>Local</td><td>24</td><td>18.75%</td></tr><tr><td>Swap</td><td>Local</td><td>24</td><td>18.75%</td></tr><tr><td>Environment change</td><td>Global</td><td>24</td><td>18.75%</td></tr><tr><td>Stylization</td><td>Global</td><td>8</td><td>6.25%</td></tr><tr><td>Total</td><td></td><td>128</td><td>100%</td></tr></table>

Table 7 Distribution of the 128 MoviEdit-Bench editing scenarios across local and global edit categories.

Non-subject preservation. This dimension measures preservation outside the intended edit region. For local edits, we compute LPIPS (Zhang et al., 2018) between corresponding source and edited frames after masking out the dilated target region, and average the score over frames.

Edit localization. This dimension measures whether the edit is confined to the intended target (Precision) and suficiently covers it (Recall). Precision is the fraction of sampled frames passing VLM check (iv), penalizing unintended changes to competing subjects or to shots in which the target is absent. Recall is the fraction passing check (vi), measuring whether the intended edit is completed rather than only partially applied. For global edits, Recall instead measures whether the requested change covers the entire scene, while Precision is undefined.

Subject preservation. This dimension measures preservation of target properties that are not requested to change. MFS follows the Motion Fidelity Score (Yatim et al., 2024) used in FiVE-Bench (Li et al., 2025), adapted to multi-shot videos. We compute it independently for each target-bearing shot and average across shots. Within each shot, a 20 × 20 grid of points inside the source target bounding box is tracked with CoTracker3 (Karaev et al., 2025) in both source and edited videos, and the normalized frame-to-frame displacement directions are compared. Since MFS evaluates motion rather than appearance, it also applies to swaps and global edits. For global edits, we instead sample tracking points over the entire frame. LPIPS (Zhang et al., 2018) compares the target regions between source and edited frames and is used for add and appearance-change edits, where the original subject identity should remain. VLM reports the fraction of frames passing check (v); for global edits, it measures preservation of the underlying scene content.

Cross-shot edit consistency. This dimension measures whether the requested edit remains consistent across shots. We jointly provide the VLM judge with edited frames sampled from all target-bearing shots and ask whether visible instances exhibit the same edit, such as the same added object, modified appearance, or substituted identity. The judge is instructed to assess consistency independently of edit correctness, which is evaluated by other metrics. This produces one binary score per scenario and is defined when at least two relevant shots are available. It does not apply to removal edits. For global edits, the metric evaluates whether the resulting environment or visual style remains consistent across shots.

Edited-frame fidelity. For the IV2V task, this dimension measures whether the propagated edit remains faithful to the edited reference frame. We include the reference frame in each frame-pair VLM query and evaluate whether the edited subject or scene preserves its reference identity and attributes while allowing natural variations in viewpoint, pose, lighting, and motion. The score is the fraction of positive judgments and is not defined for removal edits.

Edit prompt alignment. This dimension measures alignment between the edited video and the per-shot target descriptions provided in the evaluation metadata. Each shot is evaluated against its corresponding target description, including shots that should remain unchanged. For CLIP, we uniformly sample eight frames per shot, compute cosine similarity between each frame and its target description using CLIP ViT-L/14 (Radford et al., 2021), and average the scores. For ViCLIP (Wang et al., 2024), the same eight frames are encoded jointly as a video clip and compared with the text embedding of the target description. VLM reports the fraction of sampled frames passing check (ii).

<table><tr><td rowspan="2">Dimension</td><td rowspan="2">Metric</td><td colspan="4">Local</td><td colspan="2">Global</td></tr><tr><td>Add</td><td>Remove</td><td>App.</td><td>Swap</td><td></td><td>Env. Style</td></tr><tr><td>Non-subject preservation</td><td>LPIPS</td><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td></tr><tr><td rowspan="2">Edit localization</td><td>Precision (VLM)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td></tr><tr><td>Recall (VLM)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td rowspan="3">Subject preservation</td><td>MFS</td><td>√</td><td>一</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>LPIPS</td><td>√</td><td>一</td><td>√</td><td>一</td><td>一</td><td>一</td></tr><tr><td>VLM</td><td>V</td><td>一</td><td>V</td><td>一</td><td>√</td><td>√</td></tr><tr><td rowspan="2">Cross-shot edit consistency Edited-frame fidelity†</td><td>VLM</td><td>√</td><td>一</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>VLM</td><td>√</td><td>一</td><td>V</td><td>√</td><td>√</td><td>√</td></tr><tr><td rowspan="2">Edit prompt alignment</td><td>CLIP, ViCLIP, VLM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>DINO, NIQE, PickS., VLM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Video quality Overall edit success</td><td>VLM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 8 Applicability of the evaluation metrics across edit types. A metric is undefined (–) when its underlying premise does not apply. <sup>†</sup>Reported only for edited-frame-guided methods.

Video quality. This dimension evaluates the edited video independently of edit correctness. DINO measures temporal coherence as the mean cosine similarity between DINOv2 (Oquab et al., 2023) features of adjacent frames within each shot. NIQE (Mittal et al., 2013) measures no-reference image quality and is averaged over frames, with lower values indicating better quality. PickScore (Kirstain et al., 2023) scores each frame against the corresponding per-shot target description. For VLM, we jointly provide the middle frame of each shot and ask the judge to rate visual quality from 1 to 5 based on naturalness and artifacts such as warping, blur, malformed regions, unnatural lighting, and geometry. We normalize the resulting score s to [0, 1] as (s − 1)/4.

Overall edit success. Overall edit success provides a strict frame-level measure that combines the applicable VLM criteria. A sampled frame is considered successful only when the requested edit is present, applied to the intended target, and unrelated content is preserved. When applicable, it must additionally preserve non-edited target attributes, fully cover the target, and remain consistent with the edited reference frame. The final score is the fraction of successful frames. Since it reuses the frame-pair judgments above, no additional VLM queries are required.

We visualize editing results from multiple models across diferent score ranges (Fig. 14). The examples show a progression from unsuccessful edits with incorrect subject localization or incomplete edit execution to successful edits with consistent appearance across shots.

Table 9 Quantitative evaluation on the short-form single-shot video editing benchmarks FiVE-Bench and VPBench for text-guided video editing (V2V).
<table><tr><td rowspan="3" colspan="2">Model</td><td colspan="8">FiVE-Bench</td><td colspan="8">VPBench</td></tr><tr><td colspan="2"></td><td colspan="2">Structure Background Preservation</td><td colspan="2">Text Alignment</td><td colspan="2">IQA</td><td colspan="2">Motion FiVE-Acc</td><td colspan="2">Background Preservation</td><td colspan="2"></td><td colspan="2">Text Alignment Quality</td></tr><tr><td>Dist.↓</td><td></td><td>PSNR↑LPIPS↓MSE↓SSIM↑CLIPS.↑CLIPS.edit↑NIQE↓Fidelity↑</td><td></td><td></td><td></td><td></td><td></td><td>VLM↑</td><td>PSNR↑SSIM↑LPIPS↓ MSE↓CLIPS.↑CLIPS.edit↑FVID↓</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Lucy Edit</td><td>15.49</td><td>25.66</td><td>99.65 50.52</td><td>82.15</td><td>25.41</td><td>20.33</td><td>6.30</td><td>87.95</td><td>0.3532</td><td>26.53</td><td>0.856</td><td>0.091</td><td>0.0095</td><td>11.11</td><td>13.89</td><td>0.236</td></tr><tr><td>pen</td><td>SANA-Streaming</td><td>14.63</td><td>24.25</td><td>111.66 56.80</td><td>80.52</td><td>24.82</td><td>19.78</td><td>6.22</td><td>86.44</td><td>0.1921</td><td>23.34</td><td>0.825</td><td>0.108</td><td>0.0100</td><td>11.96</td><td>13.86</td><td>0.252</td></tr><tr><td></td><td>Wan-Edit</td><td>11.36</td><td>25.38</td><td>96.88</td><td>34.92 81.71</td><td>26.64</td><td>21.18</td><td>6.16</td><td>88.20</td><td>0.4010</td><td>27.09</td><td>0.877</td><td>0.076 0.0023</td><td></td><td>11.00</td><td>13.54</td><td>0.238</td></tr><tr><td>Private</td><td>Gemini Omni Flash</td><td>31.29</td><td>20.45</td><td>132.49129.43 67.96</td><td></td><td>28.02</td><td>22.24</td><td>6.09</td><td>82.27</td><td>0.8007</td><td>23.44</td><td>0.789</td><td>0.102</td><td>0.0114</td><td>11.46</td><td>13.30</td><td>0.270</td></tr><tr><td></td><td>HappyHorse-1.0</td><td>34.36</td><td>19.71</td><td>145.69142.78 64.12</td><td></td><td>27.85</td><td>22.30</td><td>5.88</td><td>83.67</td><td>0.7864</td><td>21.19</td><td>0.695</td><td>0.126</td><td>0.0138</td><td>10.22</td><td>12.92</td><td>0.293</td></tr><tr><td></td><td>Aleph 2.0</td><td>23.85</td><td>25.00</td><td>104.51</td><td>85.82 83.68</td><td>27.56</td><td>21.97</td><td>6.12</td><td>88.69</td><td>0.7784</td><td>26.41</td><td>0.877</td><td>0.094</td><td>0.0121</td><td>10.74</td><td>13.36</td><td>0.248</td></tr><tr><td></td><td>MoviEdit (Ours)</td><td>16.86</td><td>26.59</td><td></td><td>88.23 31.73 84.67</td><td>26.95</td><td>21.57</td><td>6.32</td><td>89.07</td><td>0.6623</td><td>28.30 0.885</td><td></td><td>0.082</td><td>0.0043</td><td>12.76</td><td>14.00</td><td>0.197</td></tr></table>

## C Additional Results

## C.1 External Benchmark Validation

To further validate MoviEdit beyond the multi-shot setting, we evaluate it on two established short-form video editing benchmarks, FiVE-Bench (Li et al., 2025) and VPBench (Bian et al., 2025). As shown in Tab. 9, MoviEdit delivers performance that is comparable to or competitive with proprietary models across both benchmarks. These results complement our evaluation on MoviEdit-Bench, confirming that the proposed framework retains strong editing capabilities in conventional single-shot settings.

## C.2 Beyond-Training-Horizon Editing

We further investigate the ability of MoviEdit to generalize beyond its training horizon. Although trained on videos of up to 15 seconds, MoviEdit can extend to substantially longer sequences with diverse shot structures. Fig. 15 presents qualitative results on two 25-second videos: (a) a multi-shot video comprising 17 shots and (b) a continuous single-shot video. The former demonstrates consistent editing across frequent shot transitions, substantially exceeding both the duration and shot count seen during training, while the latter demonstrates sustained editing consistency over an extended continuous sequence. These results illustrate the ability of MoviEdit to handle longer editing horizons across diferent temporal structures.

![](images/99eff5dcfd160833190c9025b3406cd7fd0c0217bc80e4d2ffc2e4f5c255f94a.jpg)  
Figure 14 Qualitative comparison across overall edit success scores. Representative V2V results from multiple models, ordered by overall edit success. Examples progress from incorrect localization or incomplete edits to successful, cross-shot-consistent editing.

(a) Long-form Multi-shot Editing (25s, 17 shots)  
![](images/058ad6513f9aa44dd8401b73f48f550dec3496ee6f9b4f041fc7c3d6c93078bf.jpg)  
Figure 15 Beyond-training-horizon editing. Qualitative results on 25-second videos: (a) a 17-shot multi-shot sequence and (b) a continuous single-shot sequence. Both exceed the 15-second training horizon, demonstrating generalization to longer multi-shot and long-take videos.