# Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds

Nan Duan Haoyang Huang Weiyang Jin Haoran Li Yaowei Li Yuming Li Yijun Liu Xin Lu Xiaoxiao Ma Yanwen Ma Yaofeng Su Yilang Sun Haoyu Wang Zeyue Xue Songchun Zhang Junhao Zhuang

Joy Future Academy, JD

## Abstract

Video generation is progressing beyond isolated clips toward long-form narratives and interactive worlds, requiring models to preserve identities, follow user controls, and remain stable over extended rollouts. We present JoyAI-Echo-1.5, a unified audio-visual generation system with two purpose-built variants. The long-video variant introduces composable cross-shot memory that aggregates visual evidence across multiple prior shots and speaker cues derived from speech-filtered full-shot audio, enabling persistent character appearance and voice identity across flexible combinations of text, image, and memory conditioning. The world-model variant converts heterogeneous navigation inputs into calibrated metric 6-DoF camera trajectories and injects them through a geometry-aware conditioning pathway, enabling controller-agnostic interaction across flexible viewpoints. To support efficient long-horizon generation, we transform a bidirectional audio-visual backbone into a causal few-step generator using progressive teacher forcing and short- and long-horizon Self-Gradient Forcing on selfgenerated rollouts. Experiments demonstrate strong performance in both settings. JoyAI-Echo-1.5 achieves improvements over existing long-video baselines in cross-shot consistency, visual quality, text alignment, and speech fidelity. Its world-model variant ranks first on WBench, with an average score of 81.7, and achieves leading visual quality and long-horizon persistence on SANA-WM-Bench. Together, these results indicate that memory, geometric control, and rollout-aware training provide a practical foundation for generating coherent stories and continuously evolving interactive worlds. Project page: https: //echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/.

## 1 Introduction

The next frontier in video generation is not merely improving individual clips, but achieving continuity and control. Long-form creation requires characters and voices to remain consistent across shots, while interactive world modeling requires environments to respond precisely to user intent and remain stable during continuous rollouts. Yet existing video models are still largely designed around fixed temporal windows: they forget identities, drift when conditioned on self-generated histories, and struggle to translate heterogeneous action inputs into consistent motion. Overcoming these limitations requires more than scaling model capacity; it requires rethinking how models remember, act, and evolve over time.

We present JoyAI-Echo-1.5, a technical system comprising two purpose-built versions. Its long-video version targets long-form audio-visual generation and introduces composable cross-shot memory to preserve character appearance and speaker identity across changing scenes. It aggregates visual evidence from multiple prior shots, derives robust voice cues from speech-filtered full-shot audio, and supports flexible combinations of text, image, and memory conditions. Its world-model version targets interactive world modeling by converting heterogeneous navigation inputs into calibrated metric 6-DoF camera trajectories and injecting them through a geometry-aware control pathway. The first extends stories beyond individual shots; the second transforms generated video into an environment that can be explored.

Both versions are designed to operate beyond fixed temporal windows. JoyAI-Echo-1.5 progressively converts a bidirectional audio-visual backbone into a causal few-step generator through audio-visual teacher forcing followed by short- and long-horizon Self-Gradient Forcing on self-generated rollouts. Training on model-generated histories exposes the system to accumulated generation errors, helping it preserve synchronized audio-visual dynamics and sustain long streaming sequences within a bounded computational budget. The long-video version gives generation memory. The worldmodel version gives it control. Together, JoyAI-Echo-1.5 pushes video generation from creating isolated moments toward sustaining stories and evolving worlds.

Main contributions. Our main contributions are summarized as follows:

• A long-video system that remembers. We introduce multi-shot audio-visual memory with speech-filtered voice representations, separated positional regions, and composable text, image, and memory conditioning to preserve identity across shots.

• A world model that responds. We establish a controller-agnostic action interface based on calibrated metric 6-DoF trajectories, enabling precise camera control across heterogeneous data sources, environments, and viewpoints.

• A generator built to continue. We develop a causal few-step training pipeline that combines audio-visual teacher forcing with short- and long-horizon Self-Gradient Forcing, allowing JoyAI-Echo-1.5 to learn from its own rollout distribution and remain stable over extended generation.

## 2 Data

## 2.1 Long-Video Training Data

JoyAI-Echo-1.5 adopts a two-stage data curriculum, as illustrated in Fig. 1. Stage I reuses the identity-centric corpus of JoyAI-Echo-1.0 [Li et al., 2026] and constructs memory–target pairs from scene-disjoint clips of the same recurring character to train audio-visual memory conditioning. Stage II uses high-quality videos with broader resolution and aspect-ratio coverage, increases the proportion of subtitle-free and Chinese-language samples, and incorporates captioned single-shot and multi-shot videos, with the latter explicitly describing transitions across cuts. It improves general audio-visual quality, spatial and linguistic coverage, and transition modeling, while preserving text-to-audio-video (T2AV) and image-to-audio-video (I2AV) generation capabilities. Both stages use the same qualitycontrol pipeline, with their sampling distributions and caption formats adapted to their respective objectives.

## 2.1.1 Stage-I Memory-Construction Corpus

Identity-centric source groups. The memory-construction corpus retains the identity-centric organization introduced in JoyAI-Echo-1.0 [Li et al., 2026]. Starting from long-form films, television episodes, and web videos, recurring characters are associated with scene-disjoint single-shot clips. Following the extraction pipeline of v1.0, the data undergo global identity discovery, scene grouping, local tracklet assignment, and quality filtering. Consequently, the basic unit used by v1.5 is not an isolated video clip, but an identity group

$$
\mathcal { G } _ { k } = \{ S _ { k , n } \} _ { n = 1 } ^ { N _ { k } } , \qquad S _ { k , n } = ( V _ { k , n } , A _ { k , n } ) ,\tag{1}
$$

where all clips in $\mathcal { G } _ { k }$ contain the same recurring character while differing in scene, clothing, viewpoint, pose, motion, illumination, facial expression, dialogue, and acoustic environment. Maintaining this diversity is important because a useful memory representation should capture persistent characterlevel evidence rather than scene-specific appearance alone.

![](images/0f685471843375c233985507aef8e5f1450d4cbd670a712fe5cd898a4125579b.jpg)  
Figure 1: Two-stage data construction pipeline of JoyAI-Echo-1.5. Raw videos are quality-filtered and organized into an identity-centric Memory corpus and a high-quality corpus with broader resolution, language, and multi-shot coverage.

Memory-target sample construction. At each training iteration, one clip $S _ { k , t }$ is selected as the prediction target and a variable-size subset of the remaining clips is sampled as memory:

$$
\mathcal { M } _ { k , t } = \{ S _ { k , i _ { 1 } } , S _ { k , i _ { 2 } } , \ldots , S _ { k , i _ { K } } \} , \qquad i _ { j } \neq t .\tag{2}
$$

The memory and target clips therefore show the same identity through different observations rather than through temporally adjacent frames from the same trajectory. This separation discourages direct content copying and requires the model to recover stable identity information under changes in background, camera pose, wardrobe, action, and speech content. The number of sampled memory shots K is varied during training so that the model does not rely on a fixed memory capacity and remains effective when only a small amount of character evidence is available.

For every selected memory shot, one video frame is randomly sampled and encoded as visual memory. In contrast to the visual branch, which uses a compact frame-level observation, the complete shot audio is retained. The audio is first processed by the speech-filtering operator and is then encoded as audio memory. Using the complete speech-bearing interval provides a more reliable description of speaker characteristics than selecting a short response window, while speech filtering reduces the leakage of background music and environmental effects into the speaker condition. The visual and audio observations from each memory shot preserve the same slot order, allowing the model to jointly access the appearance and voice evidence associated with that shot.

Conditioning configurations. The availability of memory and the first-frame image condition is independently varied when constructing training samples. This produces four conditioning configurations under a unified interface:

• T2AV: text-to-audio-video generation without memory or an input image;

• I2AV: audio-video generation conditioned on text and the first frame;

• MT2AV: text-to-audio-video generation conditioned on audio-visual memory;

• MTI2AV: audio-video generation jointly conditioned on memory, text, and the first frame.

The memory-construction corpus primarily supplies MT2AV and MTI2AV examples, while the same sample representation also supports T2AV and I2AV when the corresponding conditions are absent. Randomizing both memory length and image-condition availability prevents the model from binding a particular generation task to a fixed input layout. It also enables memory and the first-frame condition to be used compositionally, rather than treating memory-conditioned generation and image-conditioned generation as separate models.

## 2.1.2 Stage-II High-Quality and Multi-Shot Enhancement Corpus

The identity-organized corpus provides the supervision required for cross-shot memory retrieval, but its distribution is constrained by the availability of recurring characters and scene-disjoint observations. Training exclusively on such samples can reduce the diversity and standalone quality of ordinary T2AV and I2AV generation. Stage II therefore uses an enhancement corpus selected independently of the identity-group requirement. It combines high-quality single-shot videos with samples containing multiple shots and explicit transitions. Its purpose is to preserve and strengthen general audio-visual generation, broaden spatial and linguistic coverage, and improve transition modeling after the memory interface has been established in Stage I.

Resolution and aspect-ratio coverage. The high-quality corpus contains videos at 720×720, 1280×720, and 720×1280, corresponding to square (1:1), landscape (16:9), and portrait (9:16) formats. Rather than concentrating training on one canonical spatial shape, we explicitly balance these formats so that the model encounters comparable high-quality examples across common generation settings. This improves robustness to changes in token layout and spatial composition and allows the same model to serve cinematic landscape generation, portrait-oriented content, and square social-media formats.

Subtitle-free visual data. The retained samples emphasize clear visual content and stable audiovisual quality, with particular attention to videos without embedded subtitles. Burned-in subtitles create persistent, high-contrast patterns that are strongly correlated with dialogue. When such patterns are overrepresented, the model may reproduce incomplete glyphs, garbled character fragments, or subtitle-like regions even when no text is requested. Increasing the proportion of subtitle-free video weakens this undesirable correlation and provides cleaner supervision for faces, body motion, lower-frame image regions, and speaking scenes. OCR metadata is still retained for filtering and analysis, but samples with dominant or repeatedly occluding text are excluded from the enhancement corpus.

Chinese-language expansion. We additionally increase the proportion of Chinese-language data, including Chinese captions and dialogue-oriented audio-visual content. The broader language distribution supplies direct supervision for Chinese prompt following, spoken-content generation, and the alignment between visible speaking motion and Chinese dialogue. This adjustment is particularly important for long-form generation, where errors in linguistic conditioning or speaker behavior can accumulate across shots. The language expansion is performed together with visual-quality and subtitle filtering so that improved Chinese coverage does not come at the cost of image cleanliness or acoustic fidelity.

## 2.1.3 Quality Filtering and Distribution Balancing

Both corpora inherit the multi-axis quality-control pipeline of JoyAI-Echo-1.0, but the retained distribution is recalibrated for high-resolution and multi-shot training.

Learned perceptual quality. We first apply a learned video-quality estimator [Lu et al., 2024] to assess perceptual quality over both spatial content and temporal behavior. It removes samples with severe compression artifacts, abnormal color, unstable exposure, visually distracting noise, or consistently low aesthetic quality. This learned score complements the following hand-crafted measurements: it captures degradations that are difficult to describe with one low-level statistic, whereas the explicit operators provide interpretable control over identity visibility, text overlays, and motion coverage.

Structural sharpness. Uniformly sampled frames are evaluated using the variance of the Laplacian. For a sampled frame $I _ { t } ,$ , its sharpness response is

$$
s _ { t } ^ { \mathrm { s h a r p } } = \mathrm { V a r } \bigl ( \nabla ^ { 2 } I _ { t } \bigr ) ,\tag{3}
$$

and the frame responses are robustly aggregated into a clip-level score

$$
S _ { \mathrm { s h a r p } } ( V ) = \mathrm { A g g } \left( \{ s _ { t } ^ { \mathrm { s h a r p } } \} _ { t \in \mathcal { T } } \right) ,\tag{4}
$$

where $\tau$ denotes the uniformly sampled timestamps. Unlike face-detection confidence alone, this operator directly measures whether structural details remain visible. It is therefore particularly important for memory training: a face may still be detected under defocus, motion blur, or low spatia resolution, while providing unreliable identity supervision. Clips with persistently weak responses or insufficiently clear character observations are removed.

Text and overlay filtering. OCR and overlay detectors identify subtitles, logos, watermarks, large text regions, and burned-in graphics. We consider not only detected text content but also its occupied area, temporal persistence, and overlap with the main character. Clips in which text dominates the image or repeatedly occludes faces and bodies are rejected. For the high-quality enhancement corpus, this filter is made deliberately stricter to increase the proportion of subtitle-free data. OCR metadata is nevertheless preserved for caption construction: an empty result is explicitly verbalized as no on-screen text or subtitles, whereas visible text is introduced with an OCR marker.

Motion statistics. Dense optical flow [Farnebäck, 2003] is used to estimate whole-frame dynamics. Given the flow field ${ \bf u } _ { t } ( p )$ between adjacent frames, we compute

$$
M ( V ) = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { \left| \Omega \right| } \sum _ { p \in \Omega } \left\| \mathbf { u } _ { t } ( p ) \right\| _ { 2 } ,\tag{5}
$$

where Ω is the image domain. The motion score is not used to retain only high-motion samples; instead, it supports distribution control. Very low-motion clips are downsampled to prevent the corpus from being dominated by static talking heads, while sharpness and perceptual quality filters prevent large flow caused by blur, camera shake, or corrupted frames from being mistaken for useful motion. The retained data therefore covers static dialogue, subtle facial and body motion, camera movement, and larger character displacement without collapsing toward either motion extreme.

Identity-level coverage and de-duplication. For the memory-construction corpus, all of the above filters are applied at both clip and identity-group levels. If an identity no longer contains enough scene-disjoint high-quality clips after filtering, the complete group is removed because it cannot support a meaningful memory–target split. Within each identity, perceptual hashes, color histograms, and DINOv2 background similarity are used to suppress near-duplicate observations. This ensures that adding memory slots introduces new evidence about appearance, viewpoint, scene, or voice rather than repeated copies of the same shot.

Distribution balancing. Finally, the retained samples are balanced across square, landscape, and portrait formats; Chinese and English content; subtitle presence; and motion ranges. The identityorganized and high-quality corpora are also mixed so that memory-conditioned examples do not overwhelm ordinary T2AV and I2AV training. These controls prevent one source domain or generation mode from dominating optimization and allow improvements in composable memory conditioning to be obtained without sacrificing general visual and acoustic fidelity.

## 2.1.4 Single- and Multi-Shot Audio-Visual Captioning

JoyAI-Echo-1.5 expands the clip-level audio-visual annotation of v1.0 into bilingual captions covering both single-shot and multi-shot videos. For each structure, we construct a text-to-video caption and derive a corresponding image-to-video caption. The T2V caption must independently specify the visible subjects and their temporal behavior, whereas the I2V caption can rely on the input first frame for appearance information and is therefore rewritten to reduce redundant identity descriptions.

Single-shot T2V captions. For a single-shot clip, structured fields are concatenated in the following order:

$$
c _ { \mathrm { s i n g l e } } ^ { \mathrm { T 2 V } } = R \oplus A \oplus S \oplus C \oplus B \oplus O \oplus E \oplus G ,\tag{6}
$$

where R is Roles\_and\_Subjects, A is Action\_and\_Dialogue, S is Style, C is Camera\_Movement, B is Background, O is the subtitle/OCR description, E is Sound\_Effects, and G is BGM. The action field describes actions, expressions, body movement, and dialogue in temporal order, while voice timbre and speaking manner are retained in the subject and dialogue descriptions. When no OCR or subtitle is detected, O explicitly states that no on-screen text or subtitles are present; otherwise, the recognized content is introduced by an OCR marker. Repeated empty-text statements are removed during concatenation.

Multi-shot T2V captions. For a video containing N shots, we use the rewritten identity-consistency annotation and serialize the individual shot records in chronological order:

$$
c _ { \mathrm { m u l t i } } ^ { \mathrm { T 2 V } } = H _ { N } \oplus \bigoplus _ { s = 1 } ^ { N } \left[ Q _ { s } \oplus R _ { s } \oplus A _ { s } \oplus C _ { s } \oplus O _ { s } \oplus E _ { s } \oplus G _ { s } \right] ,\tag{7}
$$

Table 1: Representative abbreviated examples of the four caption formats used in JoyAI-Echo-1.5. English renderings are shown for readability, while both Chinese and English captions are retained during training. T2V captions preserve explicit identity definitions, whereas I2V captions use concise natural references because appearance is supplied by the first frame.
<table><tr><td>Structure</td><td>T2V caption</td><td>I2V caption</td></tr><tr><td>Single shot</td><td>ID_A is a young man with short brown hair, wearing a dark gray hoodie, with a clear mid-range voice. He a clear, stable mid-range voice and a natural con- sits slightly off-camera, blinks naturally, and says, versational tone. He sits slightly off-camera, blinks &quot;my games often focus on that ...&quot; A static chest- naturally, and says, &quot;my games often focus on that ..&quot; up close-up uses shallow depth of field in an indoor A static chest-up close-up keeps him in focus against gym. No on-screen text or subtitles. No obvious a softly blurred gym. No on-screen text or subtitles. sound effects. No background music.</td><td>[Random I2V prefix] A short-haired man speaks with No obvious sound effects. No background music.</td></tr><tr><td>Multi-shot</td><td>This video contains two shots. Shot 1: ID_A is a [Random I2V prefix] This video contains two shots. middle-aged man in a dark green dress uniform with a deep voice; ID_B is a middle-aged man in digital camouflage with a bright voice. They walk and talk by side. The former has a deep, steady voice and side by side as a medium shot follows them. OCR: the latter a bright, natural voice. A medium shot military slogans and a name label. Soft piano music follows them; soft piano music plays. Shot 2: A seri- plays. Shot 2: ID_C is a silent middle-aged man in ous middle-aged man in a dark blue police uniform a dark blue police uniform. A police formation ad- appears silently within an advancing formation. A vances in a static medium shot. No OCR or subtitles; static medium shot follows the group. No OCR or the piano music continues.</td><td>Shot 1: A middle-aged man in a dress uniform and a middle-aged man in camouflage walk and talk side subtitles; the piano music continues.</td></tr></table>

where $H _ { N }$ states the number of shots and $Q _ { s }$ is the explicit Shot s boundary. Each shot preserves its own subjects, actions and dialogue, camera behavior, OCR state, sound effects, and background music. This format provides direct supervision for shot order, subject changes, dialogue allocation, camera changes, and soundtrack continuity across cuts. Characters recurring across shots retain consistent identifiers in the T2V annotation, while characters appearing for the first time are fully introduced in the shot where they enter.

I2V caption rewriting. Both single-shot and multi-shot T2V captions are converted into I2V captions. Because the first frame already provides the initial appearance, the full identifier-based subject description in the first shot is replaced by a concise semi-referential phrase, such as “the short-haired man” or “the uniformed middle-aged man.” The synthetic identifiers such as ID\_A are removed. Voice timbre, speaking manner, dialogue, sound effects, and background music are retained because they cannot be recovered reliably from the image. For multi-shot videos, any new character appearing after the first shot receives a sufficiently detailed natural-language description at first appearance, but is likewise described without an identifier token. The complete caption is then lightly shortened to remove repeated visual facts without discarding temporal actions or audio information.

A bilingual I2V instruction prefix is randomly sampled and prepended to the rewritten caption. The same procedure is applied independently to the Chinese and English annotations, and the resulting strings are stored in video\_audio\_i2v\_caption. This design reduces conflict between the image and redundant textual appearance descriptions while preserving the information required to generate motion, dialogue, voice, sound, and multi-shot evolution.

The two-stage curriculum separates the acquisition of memory conditioning from the subsequent enhancement of general quality and transition modeling. Stage I teaches the model to retrieve and compose cross-shot identity evidence. Stage II uses high-quality subtitle-free data with increased Chinese coverage to maintain standalone T2AV and I2AV fidelity; within the same stage, captioned single-shot samples preserve fine-grained intra-shot actions and multi-shot samples provide explicit supervision for transitions across cuts. This progression allows JoyAI-Echo-1.5 to extend long-form controllability and composable memory conditioning while maintaining broad spatial, linguistic, visual, and acoustic coverage.

## 2.2 Action Annotation and Trajectory Data

Action-conditioned world generation requires a consistent motion representation across data collected from different sources. However, these sources provide substantially different levels of supervision: synthetic UE renders expose exact camera poses and control inputs, internal gameplay provides action logs but not camera poses, while human gameplay and web videos provide audio-visual observations but no explicit action logs or camera telemetry. We therefore build a unified processing pipeline that converts all sources into metric 6-DoF camera trajectories and, whenever available, aligns them with source-native control signals.

Trajectory Data from Heterogeneous Sources. We collect motion supervision from three complementary data sources:

• Unreal Engine (UE) Renders. UE environments provide accurate geometric supervision. Avatars are controlled through local navigation commands under the engine’s native physics and collision system. Each rollout is rendered from one first-person and four character-relative third-person cameras. For every frame, we synchronously record RGB observations, keyboard inputs, camera intrinsics, and ground-truth camera-to-world poses in metric units. During training, one viewpoint is randomly sampled from each sequence, encouraging the model to generalize across camera configurations.

• Internal Gameplay. Internal game environments provide diverse interaction scenarios, including open-world navigation, driving, and mounted traversal in both first- and third-person views. Scripted agents execute predefined control sequences, allowing us to record accurately aligned action commands together with RGB video, stereo audio, and scene metadata. Interface elements such as HUDs and crosshairs are removed during rendering. Since these game engines do not expose camera pose matrices, their 6-DoF trajectories are estimated from the recorded videos and temporally aligned with the corresponding action logs.

• Human Gameplay and Web Video. These videos complement scripted data with natural navigation behavior, human reaction patterns, and richer audio-visual content. They do not contain explicit action logs or camera telemetry, so we use the recovered camera trajectory itself as motion supervision. We extract temporally continuous, shot-consistent segments of approximately one minute and estimate their metric 6-DoF camera motion using geometry-based reconstruction.

Together, these sources provide a spectrum from precise synthetic supervision to large-scale natural motion, while sharing the same trajectory representation.

Long-Sequence Metric Pose Recovery. For videos without ground-truth camera poses, directly estimating motion on short training clips often leads to scale ambiguity and unstable trajectories. We instead estimate camera motion over longer, continuous windows of approximately one minute and only afterward divide them into training clips.

Specifically, we apply the metric-depth-assisted ViPE pipeline [Huang et al., 2025] to temporally subsampled frames. The longer temporal context provides a larger geometric baseline and improves the robustness of pose reconstruction. We then recover frame-rate trajectories by interpolating rotations with Spherical Linear Interpolation (SLERP) and translations with linear interpolation. Finally, both estimated and ground-truth trajectories are transformed into a common camera-to-world coordinate convention, metric scale, axis orientation, and temporal sampling rate.

Trajectory Quality Filtering. We apply trajectory-level filtering before training. Clips are removed when pose estimation has low confidence or exhibits implausible motion, including severe frameto-frame jitter, abrupt rotational changes, negligible displacement, or unstable camera intrinsics. Ground-truth UE trajectories are subjected to the same motion-range checks, while their original metric poses are kept unchanged.

Action-Decoupled Text Annotation. We structure each audio-visual caption into persistent scene attributes, a dynamic-motion narrative, and audio-related descriptions. Persistent attributes include the scene, visual style, viewpoint, and subject identity, while the audio component describes speech and environmental sounds. During Action-SFT, we omit both the dynamic-motion narrative and the audio-related fields, retaining only the persistent static descriptors because this stage uses a visual-only denoising objective. During Joint-FT, we restore the audio-related fields while continuing to exclude the dynamic-motion narrative. In both stages, camera motion is specified explicitly by the trajectory condition rather than redundantly through language. This encourages the model to follow the provided trajectory while preserving the remaining prompt semantics.

![](images/dd5c10f96404e57fc407f8b3dc8e92c2444cb1fc778bb879e32e1cffd5c76e7b.jpg)  
Figure 2: Overview of memory-conditioned long-form generation and few-step acceleration in JoyAI-Echo-1.5. Top: unified conditioning with single-frame visual memories, speech-filtered audio memories, and separated RoPE coordinates supports multiple audio-visual generation tasks. Bottom: memory-robust joint DMD combines asymmetric video–audio optimization with audio-specific adversarial and energy constraints.

## 3 Long-Horizon Video Generation

JoyAI-Echo-1.0 [Li et al., 2026] introduced a long-term audio-visual memory mechanism that reuses compact visual and acoustic cues from previous shots, substantially improving character appearance and speaker-timbre consistency during multi-shot generation. Building on this foundation, JoyAI-Echo-1.5 improves both long-range consistency and generation efficiency. First, it introduces a unified audio-visual memory-conditioning framework that constructs visual memory from multiple shots, extracts audio memory from speech-filtered full-shot audio, separates memory and target tokens through RoPE coordinates, and jointly trains different combinations of memory and image conditions. Second, it applies distribution matching distillation to convert the original multi-step generator into an 8-step student, with optimization strategies designed for the distinct characteristics of video and audio generation. We first formulate the long-horizon generation task and present the unified memory-conditioning framework, and then describe the proposed distillation method for efficient audio-visual generation.

## 3.1 Unified Audio-Visual Memory Conditioning

Task formulation. Given a story script represented by shot-level text conditions $C = \{ c _ { t } \} _ { t = 1 } ^ { T }$ JoyAI-Echo generates a sequence of synchronized audio-visual shots $S = \{ S _ { t } \} _ { t = 1 } ^ { T }$ , where $S _ { t } =$ $( V _ { t } , A _ { t } )$ . Let $M _ { 0 }$ denote an optional memory bank supplied before generation, and let $I = \{ I _ { t } \} _ { t = 1 } ^ { T }$ denote optional image conditions for individual shots. Long-form generation is factorized as:

$$
p _ { \theta } ( S \mid C , M _ { 0 } , I ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( S _ { t } \mid c _ { t } , M _ { t - 1 } , I _ { t } ) , \qquad M _ { t } = \mathcal { U } ( M _ { t - 1 } , S _ { t } ) .\tag{8}
$$

Before generating shot $t , M _ { t - 1 }$ contains both the initial memory $M _ { 0 }$ and the memory extracted from shots $S _ { 1 } , \ldots , S _ { t - 1 }$ . After $S _ { t }$ is generated, the update function U extracts its audio-visual information and updates the memory bank for the next shot. The image condition $I _ { t } ,$ , when provided, specifies the first frame of the current target shot and does not consume a memory slot.

Audio-visual memory construction. Each memory item is written as $m _ { i } = ( m _ { i } ^ { v } , m _ { i } ^ { a } )$ , where $m _ { i } ^ { v }$ and $m _ { i } ^ { a }$ denote its visual and audio components. From the available history, we select up to K shots and arrange them in chronological order. For the i-th selected shot, one frame is sampled and encoded by the video VAE:

$$
m _ { i } ^ { v } = E _ { v } ( V _ { i } [ f _ { i } ] ) , \qquad f _ { i } \sim { \mathcal { U } } \{ 1 , \dots , | V _ { i } | \} .\tag{9}
$$

The resulting latents provide compact visual observations from different historical shots instead of restricting the condition to a single adjacent frame.

For audio memory, we use the complete audio content of each selected shot. Since full-shot audio may contain background music, environmental sounds, and other non-speech components, we first apply a speech-filtering operator $\mathcal { F } _ { \mathrm { s p e e c h } }$ and then encode the filtered waveform using the audio VAE:

$$
\widetilde { A } _ { i } = \mathcal { F } _ { \mathrm { s p e e c h } } ( A _ { i } ) , \qquad m _ { i } ^ { a } = E _ { a } ( \widetilde { A } _ { i } ) .\tag{10}
$$

Here, $\mathcal { F } _ { \mathrm { s p e e c h } }$ is a generic speech-separation operator that can be instantiated by different filtering models; our current implementation uses a speech-extraction model through the Music-Source-Separation-Training (MSST) toolkit. In v1.0, a short audio window is selected according to acoustic energy. However, the maximum-response window may be dominated by music or other loud background components, which can be repeatedly reinforced when generated audio is reused during autoregressive rollout. The new construction instead uses the complete filtered speech content of each shot, providing cleaner and more stable speaker information. The visual and audio latents are then arranged in the same slot order, so that each memory item contains one compact visual observation and its corresponding event-level audio segment.

RoPE for memory conditioning. All historical conditions and user-provided memory inputs are represented in a common memory format. Following the discontinuous RoPE offsets introduced in ShotStream [Luo et al., 2026], we assign separated temporal RoPE regions to the current target, historical memory, and memory supplied before generation. The target video and audio tokens retain their original temporally aligned coordinates in a low-offset region below 500. When an image condition $I _ { t }$ is provided, its tokens remain in the coordinate system of the current target shot and are placed at temporal coordinate zero as its clean first frame.

The visual and audio components extracted from the i-th historical shot share the same temporal RoPE center:

$$
\mu _ { i } = 5 0 0 + 5 0 ( i - 1 ) ,\tag{11}
$$

such that successive audio-visual memory items are centered at 500, 550, 600, . . .. All spatial tokens of the sampled visual memory frame use $\mu _ { i }$ as their temporal coordinate while retaining their original spatial RoPE coordinates. For the corresponding audio latent with native temporal coordinates $\{ \tau _ { i , \ell } ^ { a } \} _ { \ell = 1 } ^ { L _ { i } ^ { a } }$ , we assign:

$$
r _ { i , \ell } ^ { a } = \mu _ { i } + \tau _ { i , \ell } ^ { a } - \bar { \tau } _ { i } ^ { a } , \quad \quad \bar { \tau } _ { i } ^ { a } = \frac { 1 } { L _ { i } ^ { a } } \sum _ { \ell = 1 } ^ { L _ { i } ^ { a } } \tau _ { i , \ell } ^ { a } ,\tag{12}
$$

where ℓ indexes audio tokens. In other words, the temporal RoPE coordinates of the complete audio latent are shifted together so that its midpoint is aligned with the same center $\mu _ { i }$ . This operation preserves the relative temporal structure of the audio event and does not modify the audio latent features themselves.

Memory supplied before generation is encoded independently and placed in a high-offset region:

$$
M _ { 0 } = \{ m _ { j } ^ { 0 } \} _ { j = 1 } ^ { J } , \qquad m _ { j } ^ { 0 } = ( m _ { j } ^ { 0 , v } , m _ { j } ^ { 0 , a } ) ,\tag{13}
$$

where

$$
m _ { j } ^ { 0 , v } = E _ { v } ( X _ { j } ^ { \mathrm { m e m } } ) , \qquad m _ { j } ^ { 0 , a } = E _ { a } \bigl ( \mathcal { F } _ { \mathrm { s p e e c h } } ( A _ { j } ^ { \mathrm { m e m } } ) \bigr ) , \qquad \rho _ { j } \geq 5 0 0 0 .\tag{14}
$$

The visual and audio components of the j-th initial memory item are both centered at $\rho _ { j }$ using the same positional assignment: the visual tokens use $\rho _ { j }$ as their temporal coordinate, while the audio RoPE coordinates are shifted together so that their midpoint is aligned with $\rho _ { j }$ . The centers $\{ \rho _ { j } \} _ { j = 1 } ^ { J }$ are distinct and are assigned with enough spacing to keep the shifted audio-coordinate spans of different memory items non-overlapping. Consequently, the current target, historical memory, and initial memory occupy separated temporal RoPE regions. This separation keeps memory outside the target-shot timeline while allowing memory and image conditions to be provided simultaneously.

Memory interaction. After memory construction and RoPE assignment, memory and target tokens are jointly processed by the model. Unlike v1.0, v1.5 removes the memory-specific audio selfattention mask and the slot-paired cross-modal attention mask. All valid tokens interact through the original self-attention and bidirectional cross-modal attention paths. Their roles remain distinguishable because memory tokens use dedicated RoPE coordinates, remain clean, and are excluded from the prediction loss, whereas target tokens follow the target timeline and are optimized as generation outputs. This unified interaction removes additional hard constraints and naturally supports the joint use of memory and image conditions.

Joint conditional training. Training follows the two-stage data curriculum described in Sec. 2.1. Throughout the curriculum, we use a unified conditional interface that supports text-to-audio-video (T2AV), image-to-audio-video (I2AV), memory-conditioned text-to-audio-video (MT2AV), and memory-conditioned text-and-image-to-audio-video (MTI2AV). Let $b _ { M } , b _ { I } \in \{ 0 , 1 \}$ indicate whether audio-visual memory and a first-frame image condition are provided. The model condition for shot t is:

$$
\mathcal { C } _ { t } = ( c _ { t } , b _ { M } M _ { t - 1 } , b _ { I } I _ { t } ) ,\tag{15}
$$

where $( b _ { M } , b _ { I } ) \ : = \ : ( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 )$ correspond to T2AV, I2AV, MT2AV, and MTI2AV, respectively.

Let $z _ { 0 } ^ { v }$ and $z _ { 0 } ^ { a }$ denote the clean target video and audio latents, and let $\epsilon ^ { v } , \epsilon ^ { a } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . For each modality $q \in \{ v , a \}$ , rectified-flow training constructs the interpolation and target velocity for the tokens to be predicted:

$$
z _ { \sigma } ^ { q } = ( 1 - \sigma ) z _ { 0 } ^ { q } + \sigma \epsilon ^ { q } , \qquad u ^ { q } = \epsilon ^ { q } - z _ { 0 } ^ { q } .\tag{16}
$$

We write the joint noised audio-visual target as $\boldsymbol { z } _ { \sigma } = ( z _ { \sigma } ^ { v } , z _ { \sigma } ^ { a } )$ and the paired noise as $\epsilon = ( \epsilon ^ { v } , \epsilon ^ { a } )$ Memory tokens are inserted as clean conditions with zero noise level and are excluded from the prediction loss. When $b _ { I } = 1$ , the first target video frame is also kept clean and excluded from the video loss. Let $W _ { v }$ and $W _ { a }$ be binary loss masks selecting only noised target tokens. The joint training objective is:

$$
\mathcal { L } = \mathbb { E } _ { \mathrm { d a t a } , \epsilon , \sigma , b _ { M } , b _ { I } } \left[ \| W _ { v } \odot ( v _ { \theta } ^ { v } ( z _ { \sigma } , \mathcal { C } _ { t } ) - u ^ { v } ) \| _ { 2 } ^ { 2 } + \lambda _ { a } \| W _ { a } \odot ( v _ { \theta } ^ { a } ( z _ { \sigma } , \mathcal { C } _ { t } ) - u ^ { a } ) \| _ { 2 } ^ { 2 } \right] ,\tag{17}
$$

where $\lambda _ { a }$ balances the audio and video objectives. The resulting training distribution covers the four combinations of $( b _ { M } , b _ { I } )$ , enabling generation without memory, with image conditioning alone, with memory alone, and with both conditions. This unified formulation supports T2AV, I2AV, MT2AV, and MTI2AV without changing the model interface or introducing separate objectives.

## 3.2 Acceleration with Distribution Matching Distillation

Long-form audio-visual generation requires both long-range consistency and low-latency inference. To accelerate JoyAI-Echo-1.5, we apply distribution matching distillation (DMD) to convert the original multi-step audio-visual generator into an 8-step student generator.

Data curation and training pipeline. JoyAI-Echo-1.5 performs DMD using memory conditions constructed from real long-form videos following Sec. 3.1. From the large-scale training corpus, we curate a compact subset with high-quality audio-visual memory and diverse generation conditions. The resulting data mixture covers T2AV, I2AV, and MT2AV, together with samples annotated by multi-shot captions. During training, we randomly sample these tasks according to predefined mixing ratios, allowing the student generator to preserve both single-shot generation quality and memory-conditioned multi-shot capability.

We find that DMD is particularly sensitive to the quality and distribution of visual memory. Lowlevel discrepancies in memory appearance, including brightness and saturation shifts, can bias the conditional distribution learned by the student and become progressively amplified when generated shots are written back into memory. Before DMD training, we therefore calibrate the low-level statistics of visual memory to better match the target training distribution. This preprocessing substantially improves distillation stability and reduces quality drift during multi-shot rollout.

The curated task mixture and calibrated memory conditions are used throughout the complete DMD pipeline. We first perform high-shift DMD with a relatively large learning rate to establish video quality and global structure, and then continue from the converged checkpoint with a lower shift and learning rate to refine audio and other fine-grained details. In both stages, the video branch is fully fine-tuned, while the audio and cross-modal modules are adapted through LoRA.

Asymmetric audio-visual optimization. Video and audio exhibit substantially different structures in the latent space. Video latents are highly structured and spatially redundant, with strong correlations across neighboring spatial locations and adjacent frames. Their temporal evolution is also relatively smooth, allowing global appearance, scene layout, and motion patterns to be represented through slowly varying latent structures. Audio latents, in contrast, encode dense one-dimensional temporal signals with rapidly changing local patterns. Fine-grained information such as phonetic content, pitch, transients, and acoustic textures may vary significantly within a short temporal interval [Su et al., 2026].

These differences lead to asymmetric optimization requirements during few-step DMD. The video branch must substantially adapt its spatial and temporal representations to recover visual structure and motion along the shortened sampling trajectory. Restricting its trainable capacity results in insufficient adaptation and degraded visual fidelity. The audio branch, however, is more sensitive to aggressive distribution-matching updates. Small deviations in the audio latent can remove perceptually important local details, while excessive parameter updates may overwrite the acoustic prior of the multi-step teacher and introduce muffled, metallic, or electronic-sounding artifacts [Tian et al., 2026].

We therefore allocate different trainable capacities to the two modalities:

$$
\theta _ { G } ^ { \mathrm { t r a i n } } = \left\{ \theta _ { v } ^ { \mathrm { f u l l } } , \Delta \theta _ { a } ^ { \mathrm { L o R A } } , \Delta \theta _ { \mathrm { c r o s s } } ^ { \mathrm { L o R A } } \right\} ,\tag{18}
$$

where the video branch is fully fine-tuned, while the audio branch and cross-modal attention modules are adapted through low-rank adaptation (LoRA). Full fine-tuning provides sufficient capacity for the video branch to learn the structural and motion changes required by few-step generation. In contrast, LoRA constrains the effective update space of the audio branch, slowing its deviation from the well-trained acoustic prior. Applying LoRA to the cross-modal attention modules preserves the flexibility required to adapt audio-visual correspondence without aggressively modifying the audio representation itself. This asymmetric parameterization therefore balances video adaptability with audio stability during joint DMD training.

Coarse-to-fine DMD noise scheduling. In DMD, the clean prediction produced by the student generator is re-noised to an auxiliary noise level σ, at which the teacher and fake-score model estimate the real and generated scores [Yin et al., 2024b,a]. The choice of σ is important because it determines the granularity of the distributional information exposed to score matching. Higher noise levels place greater emphasis on coarse distributional properties and help stabilize global structure, whereas lower noise levels retain more sample-specific information and provide stronger supervision for fine-grained details.

Video and audio benefit differently from these noise regimes. Video generation relies strongly on high-noise DMD updates to establish stable appearance, scene layout, motion, and temporal structure under the shortened sampling trajectory. Audio quality, in contrast, is particularly sensitive to the low-noise regime, where local temporal patterns such as phonetic details, pitch, transients, and acoustic textures are more directly preserved and corrected. A fixed noise regime therefore cannot simultaneously provide the most suitable optimization signal for both modalities.

We address this mismatch with a two-stage DMD schedule. The first stage uses a relatively large timestep shift and learning rate, biasing score matching toward higher noise levels to establish stable video quality and structure. Starting from the converged first-stage checkpoint, the second stage reduces both the timestep shift and learning rate, shifting the optimization toward lower noise levels to refine audio quality and other fine-grained details. This coarse-to-fine schedule accommodates the different noise-level preferences of video and audio during joint few-step distillation.

Memory-robust and task-adaptive DMD. For the DMD stage described here, we consider T2AV/I2AV/MT2AV, where MT2AV generates the current shot conditioned on the audio-visual memory bank $M _ { t - 1 } .$ . Although the training memory is constructed from real long-form videos, the memory available during inference is progressively extracted from previously generated shots. Even small generation errors can therefore alter the memory distribution and accumulate over a multi-shot rollout. A student trained only with undegraded memory may become sensitive to this discrepancy, leading to gradual degradation of visual quality and cross-shot consistency.

To improve robustness, we apply stochastic degradation to the visual memory during DMD training. The student generator and fake-score model receive the same probabilistically degraded memory, allowing the fake-score model to track the student distribution under imperfect memory conditions. In contrast, the real-score model receives the original, undegraded memory and provides a stable target corresponding to the desired conditional distribution. The resulting score difference encourages the student to recover a high-quality current shot even when its historical memory contains moderate errors. This stochastic degradation complements the visual-memory distribution calibration described above: calibration corrects systematic low-level biases in the training memory, while degradation simulates the sample-dependent errors encountered during autoregressive rollout.

Moreover, MT2AV and other tasks also exhibit different audio optimization dynamics. In T2AV or I2AV, the student must construct the complete audio distribution from text and noise withour any audio reference. In MT2AV, the memory condition already provides informative cues about speaker timbre, acoustic context, and temporal continuity, making its audio component easier to learn. Consequently, using the same audio weight for both tasks can cause the MT2AV audio objective to contribute disproportionately to joint DMD optimization.

We therefore introduce a task-dependent audio weight:

$$
\mathcal { L } _ { \mathrm { D M D } } = \mathcal { L } _ { \mathrm { D M D } } ^ { v } + \lambda _ { a } ( \kappa ) \mathcal { L } _ { \mathrm { D M D } } ^ { a } ,\tag{19}
$$

where $\kappa \in \{ \mathrm { T 2 A V } , \mathrm { I 2 A V } , \mathrm { M T 2 A V } \}$ denotes the DMD task, with $\lambda _ { a } ( \mathrm { T 2 A V / I 2 A V } ) = 0 . 2 5$ and $\lambda _ { a } ( \mathrm { M T 2 A V } ) = 0 . 1 0$ . The reduced MT2AV weight prevents its easier memory-conditioned audio objective from dominating joint distillation, while the larger weight retains sufficient supervision for learning audio generation directly from noise. Together, stochastic memory degradation and task-adaptive weighting improve the robustness of few-step generation across both single-shot and memory-conditioned settings.

Audio distillation with adversarial and energy constraints. Despite the conservative Audio-LoRA parameterization and low-noise refinement stage, applying score-based DMD to audio remains challenging. The distribution-matching gradient mainly corrects the discrepancy between the teacher and student distributions at noisy latent states. For dense audio signals, this correction can favor dominant acoustic modes while suppressing subtle temporal variations, producing over-smoothed, muffled, or electronic-sounding outputs. We further observe that DMD may cause the overall audio latent level to increase continuously during training, leading to excessive loudness, clipping, and harsh distortion after decoding. We therefore introduce two complementary audio-specific objectives to constrain global signal level and local acoustic realism.

First, we match the root-mean-square (RMS) level of the generated audio latent to that of its groundtruth counterpart:

$$
\mathcal { L } _ { \mathrm { e n e r g y } } ^ { a } = \psi \left( \log \left( \mathrm { R M S } ( \hat { z } _ { 0 } ^ { a } ) + \varepsilon _ { \mathrm { n u m } } \right) - \log \left( \mathrm { R M S } ( z _ { 0 } ^ { a } ) + \varepsilon _ { \mathrm { n u m } } \right) \right) ,\tag{20}
$$

where $\hat { z } _ { 0 } ^ { a }$ and $z _ { 0 } ^ { a }$ denote the predicted and ground-truth clean audio latents, respectively. The robust penalty ψ limits the influence of extreme samples, and $\varepsilon _ { \mathrm { n u m } } > 0$ ensures numerical stability. Computing the discrepancy in the log domain makes the loss sensitive to relative rather than absolute level differences. This constraint suppresses persistent latent-level drift and reduces the risk of clipping after audio decoding.

Second, following AudioX-Turbo [Tian et al., 2026], we introduce a diffusion-based audio discriminator. We reuse intermediate features from the frozen audio teacher as the discriminator backbone and train only lightweight heads to distinguish generated and real audio latents. By preserving the pretrained audio representation and limiting adversarial training to the added heads, the discriminator provides a stable perceptual signal without modifying the teacher. Its local adversarial feedback complements the distribution-level DMD objective by preserving transients, acoustic textures, and other fine-grained temporal details.

The complete student-generator objective is

$$
\begin{array} { r l } { \mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { D M D } } ^ { v } + \lambda _ { a } ( \kappa ) \mathcal { L } _ { \mathrm { D M D } } ^ { a } } & { + \lambda _ { \mathrm { G A N } } \mathcal { L } _ { \mathrm { G A N } } ^ { a } + \lambda _ { \mathrm { e n e r g y } } \mathcal { L } _ { \mathrm { e n e r g y } } ^ { a } . } \end{array}\tag{21}
$$

The DMD terms align the global audio-visual generation distribution, the adversarial loss restores local acoustic details, and the RMS-level constraint stabilizes the overall audio signal level. Together, these objectives mitigate both perceptual over-smoothing and excessive energy growth during few-step distillation.

## 4 Action-Conditioned World Modeling

## 4.1 Overview and Problem Formulation

Given an optional clean media context M, a text condition ${ \mathcal { C } } ,$ and a user control sequence $U _ { 1 : T }$ our goal is to jointly generate synchronized video $V _ { 1 : T }$ and audio $A _ { 1 : T }$ under user-specified camera intent. The media context M accommodates an empty condition $\mathcal { O } _ { ; }$ , a single reference image $I _ { \mathrm { r e f } } .$ , or an audio-visual prefix $\left( V _ { 1 : \tau } , A _ { 1 : \tau } \right)$ . We serialize user navigation inputs into a time-indexed metric camera trajectory event stream $\mathcal { E } _ { P } ^ { \ ' } = \{ ( t , P _ { t } ) \} _ { t = 1 } ^ { T }$ , where $\bar { P } _ { t }$ represents a calibrated relative 6-DoF pose. The joint generation process is formulated as:

$$
p _ { \theta , \phi } ( V _ { 1 : T } , A _ { 1 : T } \mid \mathcal { M } , \mathcal { C } , \mathcal { E } _ { P } ) ,\tag{22}
$$

where $\theta$ denotes the audio-visual backbone, and $\phi$ represents the parallel relative trajectoryconditioning pathway. Static text fields specify scene semantics, while $\mathcal { E } _ { P }$ dictates the continuous temporal trajectory evolution. When $\mathcal { M }$ contains an observed audio-visual prefix, those prefix samples are clamped and only the unobserved suffix is generated; the full-sequence notation in Eq. (22) is used for compactness.

## 4.2 Unified Camera-Intent Interface & Scale Calibration

Realizing continuous world modeling across heterogeneous datasets requires overcoming two key obstacles: the incompatibility of controller-specific action spaces and the scale mismatch of relative motion amplitudes. To address these issues, we decouple user-facing controls from model-level conditions through a two-fold strategy: (1) mapping arbitrary navigation inputs into a unified, subject-agnostic 6-DoF trajectory, and (2) calibrating relative translations with a robust global scale.

Relative 6-DoF Representation. We define camera intent using relative 6-DoF trajectories rather than actor-specific control states. Let $T _ { t } \in \mathrm { S E } ( 3 )$ ) denote the camera-to-world extrinsic matrix at frame t. The relative motion between frames i and $j$ is given by:

$$
\Delta T _ { i j } = T _ { i } ^ { - 1 } T _ { j } , \qquad \xi _ { i j } = \mathrm { L o g } ( \Delta T _ { i j } ) ^ { \vee } \in \mathbb { R } ^ { 6 } ,\tag{23}
$$

where $\xi _ { i j }$ is the corresponding 6-DoF twist coordinate. For discrete keyboard inputs $\mathbf { u } _ { t }$ , a fixed control mapping $\mathcal { F } _ { \mathrm { c t r l } }$ converts each input into a step-wise camera increment:

$$
\begin{array} { r } { \pmb { \xi } _ { t } = \mathcal { F } _ { \mathrm { c t r l } } ( \mathbf { u } _ { t } ) , \qquad T _ { t } = T _ { t - 1 } \mathrm { E x p } ( \widehat { \pmb { \xi } } _ { t } ) , \qquad \Delta T _ { t } = T _ { 0 } ^ { - 1 } T _ { t } , } \end{array}\tag{24}
$$

where $\widehat { \pmb { \xi } } _ { t } \in { \mathfrak { s e } } ( 3 )$ . Continuous metric trajectories bypass $\mathcal { F } _ { \mathrm { c t r l } }$ and directly produce relative poses $\Delta T _ { 1 : T }$ . This unifies raw actions and estimated poses under a single geometric interface.

Global Translation Scale Calibration. To ensure consistent velocity semantics without introducing boundary artifacts in multi-turn rollouts, we calibrate relative translations across all data sources using a fixed global dataset-level scale $s _ { \mathrm { g l o b a l } }$ . For clip $i ,$ let $\Delta \mathbf { t } _ { i , k }$ denote the relative translation at frame k with respect to the initial frame. We compute

$$
s _ { \mathrm { g l o b a l } } = Q _ { 0 . 9 } \left( \left\{ \operatorname* { m a x } _ { k } \| \Delta \mathbf { t } _ { i , k } \| _ { 2 } \right\} _ { i \in \mathcal { D } _ { \mathrm { t r a i n } } } \right) .\tag{25}
$$

Relative translations are calibrated via $\widehat { \mathbf { t } } _ { i , k } = \Delta \mathbf { t } _ { i , k } / s _ { \mathrm { g l o b a l } }$ , yielding $P _ { i , k } = [ \Delta \mathbf { R } _ { i , k } \ | \ \widehat { \mathbf { t } } _ { i , k } ]$ while keeping rotation matrices $\Delta \mathbf { R } _ { i , k }$ intact.

## 4.3 Relative Trajectory Conditioning

Instead of absolute Plücker coordinates which depend on arbitrary world origins, we inject trajectory geometry through Unified Camera Positional Encoding (UCPE) [Zhang et al., 2025a]. UCPE makes

pairwise camera geometry invariant to global coordinate shifts while preserving calibrated relative translation magnitudes.

For a latent token $\boldsymbol { i } = ( t , s )$ at frame t and spatial cell s, we define its local ray transformation $\mathbf D _ { i } = ( \mathbf T _ { i } ^ { \mathrm { w r } } ) ^ { - 1 }$ derived from calibrated pose $\bar { \boldsymbol { P } } _ { t }$ and intrinsics $K _ { t }$ . We inject this geometry into parallel attention heads via matrix scaling $\mathbf { G } _ { i } = \mathbf { I } _ { d / 8 } \otimes \mathbf { D } _ { i }$ combined with spatiotemporal RoPE:

$$
\begin{array} { c } { \widetilde { \mathbf { q } } _ { i } ^ { c } = ( \mathbf { G } _ { i } ^ { \top } \oplus \mathrm { R o P E } _ { i } ) \mathbf { q } _ { i } ^ { c } , } \\ { ( \widetilde { \mathbf { k } } _ { i } ^ { c } , \widetilde { \mathbf { v } } _ { i } ^ { c } ) = ( \mathbf { G } _ { i } ^ { - 1 } \oplus \mathrm { R o P E } _ { i } ) ( \mathbf { k } _ { i } ^ { c } , \mathbf { v } _ { i } ^ { c } ) , } \\ { \mathbf { o } _ { i } ^ { c } = ( \mathbf { G } _ { i } \oplus \mathrm { R o P E } _ { i } ^ { - 1 } ) \mathrm { A t t n } _ { \mathrm { c a m } } ( \widetilde { \mathbf { q } } ^ { c } , \widetilde { \mathbf { k } } ^ { c } , \widetilde { \mathbf { v } } ^ { c } ) _ { i } . } \end{array}\tag{26}
$$

Here, $\mathbf { T } _ { i } ^ { \mathrm { w r } }$ is the UCPE ray-to-world transform constructed from $( P _ { t } , K _ { t } , s )$ , so $\mathbf { D } _ { i }$ maps world coordinates to the ray-local frame; d is the attention-head width, and the superscript c identifies the parallel camera-conditioning stream. The symbol ⊕ denotes a block-diagonal direct sum with the spatiotemporal RoPE operator, and $\mathrm { A t t n } _ { \mathrm { c a m } }$ denotes attention in the camera pathway. The resulting camera-attention outputs pass through zero-initialized linear projections in the trajectory pathway parameterized by $\phi$ and are added to the main video self-attention blocks. The trajectory condition is applied exclusively to the visual backbone. One UCPE pathway is shared across first- and third-person data. Geometry alone does not determine whether a trajectory is realized from an egocentric or thirdperson observer–subject configuration, so the static viewpoint field supplies this initial relationship while $P _ { 1 : T }$ supplies camera intent. This division allows the same trajectory representation and parameters to serve both viewpoints while leaving their camera–character realization to the learned world prior.

## 4.4 Progressive Training Curriculum

Data sources with rich audio-visual semantics and those providing precise, clean control trajectories are fundamentally heterogeneous. To resolve the resulting optimization conflicts, we employ a progressive three-stage curriculum that decouples audio-visual prior acquisition, control alignment, and joint consolidation.

Stage 1: Audio-Visual Continued Pretraining (AV-CPT). This stage adapts the base model to the visual styles, motion statistics, and acoustic dynamics (footsteps, ambient audio, and speech) of interactive game and general environments. We optimize the full backbone θ on the AV-rich data mixture while completely disabling trajectory conditioning. To simultaneously establish text-to-AV generation, image-to-AV generation, and temporal continuation capabilities, we randomly sample the clean media condition M from an empty set ∅, a single reference frame $I _ { \mathrm { r e f } }$ , or a synchronized audio-visual prefix $( V _ { 1 : \tau } , A _ { 1 : \tau } )$ . Unobserved video and audio latent regions are generated jointly, embedding strong acoustic priors and temporal continuity into the backbone.

Stage 2: Action Fine-Tuning (Action-SFT). This stage isolates control responsiveness without degrading the acquired audio-visual generation priors. We freeze the entire backbone θ and train only the lightweight UCPE camera pathway parameters ϕ on control-clean data using a visual-only denoising objective. To eliminate information shortcuts, we omit the dynamic-motion narrative and all audio-related fields, retaining only persistent scene, style, viewpoint, and subject descriptors $\mathcal { C } _ { \mathrm { s t a t i c } }$ . This forces the trajectory $\mathcal { E } _ { P }$ to serve as the sole source of motion guidance. Furthermore, we heavily bias condition sampling toward reference-frame conditioning $\left( I _ { \mathrm { r e f } } \right)$ , fixing initial scene content so the network concentrates parameter updates entirely on camera trajectory following.

Stage 3: Joint Fine-Tuning (Joint-FT). To consolidate control sensitivity with joint audio-visual synthesis, we unfreeze both the backbone θ and trajectory pathway ϕ for end-to-end co-adaptation on the balanced high-quality mixture. We restore environmental sound and speech fields to the text prompt (while keeping dynamic motion descriptions excluded) and resume joint audio-visual loss optimization. Training executes with a reduced learning rate to harmoniously align multimodal quality with trajectory responsiveness without collapsing the pre-trained priors.

## 5 Autoregressive Audio-Visual Generation

We progressively adapt a pretrained bidirectional, multi-step audio-visual diffusion model into a high-throughput, low-latency streaming autoregressive generator. This transition requires two key transformations: converting bidirectional temporal attention into causal autoregressive computation for online generation of synchronized audio-visual chunks, and compressing iterative denoising into a few-step sampler that remains stable under self-generated histories. To achieve this, we first apply audio-visual teacher forcing Williams and Zipser [1989], Chen et al. [2024] to initialize a causal streaming generator from the bidirectional backbone. We then introduce short-horizon Self-Gradient Forcing (SGF) Huang et al. [2026], Zhuang et al. [2026] to train on self-generated rollout states while distilling few-step generation, and further extend SGF to long-horizon self-rollouts to improve robustness and stability during sustained streaming inference.

## 5.1 Audio-Visual Teacher Forcing

We first use teacher forcing to initialize a causal generator from the pretrained bidirectional model. Specifically, during training, ground-truth audio-visual chunks are provided as the historical context, such that the generation of the current chunk depends only on preceding clean chunks rather than future observations. This stage provides a stable causal initialization before exposing the model to self-generated histories in subsequent self-forcing training.

Causal initialization. Let $m \in \{ v , a \}$ index the modality, with v denoting video and a denoting audio. We divide both latent streams into N temporally aligned macro-chunks. The clean latent of modality m in chunk i is denoted by $x _ { i } ^ { m }$ , where $\bar { i } \in \{ 1 , \ldots , \bar { N } \}$ . An aligned video-audio pair spans the same time interval, although the two chunks may contain different numbers of tokens. For each pair, we sample a shared noise level $\sigma _ { i } \in ( 0 , 1 )$ and construct:

$$
x _ { \sigma _ { i } , i } ^ { m } = ( 1 - \sigma _ { i } ) x _ { i } ^ { m } + \sigma _ { i } \epsilon _ { i } ^ { m } , \qquad \epsilon _ { i } ^ { m } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{27}
$$

where $\epsilon _ { i } ^ { m }$ is Gaussian noise and I is the identity covariance matrix. The shared $\sigma _ { i }$ places the paired audio and video chunks at a consistent diffusion stage.

To enable causal generation without sacrificing training parallelism, we concatenate the clean latents with their noisy counterparts. Let $B _ { i , \mathrm { c l e a n } }$ and $B _ { i , \mathrm { n o i s y } }$ denote the clean and noisy latent blocks of the i-th paired audio-visual chunk, respectively. We impose the following causal attention pattern:

$$
\begin{array} { r } { S ( { \mathcal B } _ { i , \mathrm { c l e a n } } ) = \{ { \mathcal B } _ { j , \mathrm { c l e a n } } : j \leq i \} , \qquad S ( { \mathcal B } _ { i , \mathrm { n o i s y } } ) = \{ { \mathcal B } _ { j , \mathrm { c l e a n } } : j < i \} \cup \{ { \mathcal B } _ { i , \mathrm { n o i s y } } \} , } \end{array}\tag{28}
$$

where $\boldsymbol { \mathcal { S } } ( \cdot )$ denotes the set of latent blocks accessible to the corresponding attention query. Under this mask, each noisy chunk can attend to all preceding clean audio-visual chunks and to the noisy tokens within the current chunk, while the corresponding clean target and all future chunks are masked out. We apply the same causal constraint to video self-attention, audio self-attention, and both directions of audio-visual cross-attention. For efficient causal attention computation, we implement this pattern using PyTorch FlexAttention [Dong et al., 2025] with a block-level causal mask. The resulting BlockMask exploits the structured sparsity of chunk-wise causal attention, allowing masked blocks to be skipped without materializing a dense attention mask. As illustrated in Fig. 3(a), this causal attention pattern enables each chunk to use the complete preceding clean history while preventing information leakage from the current clean target and future chunks. This converts the original bidirectional temporal attention into chunk-wise causal attention, while retaining parallel prediction of all chunks during training.

Training objective. To preserve the pretrained generative capability while efficiently adapting the model to the new causal attention pattern, we perform this stage using LoRA-based parameter-efficient fine-tuning Hu et al. [2021]. The pretrained backbone is kept fixed, while the LoRA parameters are optimized to learn the causal audio-visual generation behavior. Given the causally masked sequence, the model jointly predicts the flow velocities of the audio and video latents in a single forward pass. We retain the standard flow-matching objective of the pretrained diffusion model:

$$
\mathcal { L } _ { \mathrm { T F } } = \sum _ { m \in \{ v , a \} } \lambda _ { m } \mathbb { E } \left[ \sum _ { i = 1 } ^ { N } \left. v _ { \theta } ^ { m } ( x _ { \sigma _ { i } , i } ^ { m } , \sigma _ { i } ) - ( \epsilon _ { i } ^ { m } - x _ { i } ^ { m } ) \right. _ { 2 } ^ { 2 } \right] ,\tag{29}
$$

![](images/1710a91f3c82e60506d2663c690b8a0723daf5554f16928d15a9d1f8d8a590a1.jpg)  
(a) Teacher Forcing

![](images/72c62123ce76e635c4410f8a8bcbe837af73dcd1c9f5486ac5dad59c16e38cdf.jpg)  
(b) Longvideo SGF (Forward 2)  
Figure 3: Illustration of the causal attention masks used in streaming post-training. (a) Audio-visual teacher forcing, where each noisy chunk attends to its preceding clean history and the current noisy chunk, while the clean target and future chunks are masked out. (b) Long-horizon SGF Forward 2, where the causal context is further restricted by the sink-plus-FIFO KV cache.

where $v _ { \theta } ^ { m }$ denotes the predicted flow velocity for modality $m , \lambda _ { m }$ balances the video and audio losses, and θ denotes the trainable LoRA parameters of the generator. Thus, audio-visual teacher forcing adapts the pretrained bidirectional diffusion model to causal, chunk-wise autoregressive generation while preserving its original flow-matching training objective.

## 5.2 Short-Horizon Audio-Visual Self-Gradient Forcing

Starting from the causal teacher-forcing initialization, we adopt Self-Gradient Forcing (SGF) [Zhuang et al., 2026] to adapt the generator to its own autoregressive audio-visual histories. SGF follows the deployment-time self-rollout distribution while restoring gradients through the computation that encodes generated contexts into future-readable causal memory. Unlike standard Self Forcing [Huang et al., 2026], which treats the historical KV cache as detached rollout state, SGF reconstructs the sampled rollout state in a differentiable forward pass. Losses on later chunks can therefore supervise how preceding generated audio-visual contexts are encoded into causal memory for subsequent generation, without backpropagating through the full sequential sampling trajectory. We further combine SGF with a DMD objective [Yin et al., 2024b] to distill the original multi-step diffusion model into a few-step autoregressive generator, jointly improving autoregressive robustness and sampling efficiency.

Forward 1: Autoregressive self-rollout. The generator first performs a complete chunk-wise rollout with gradient computation disabled, using exactly the few-step sampler deployed at inference. For example, with a four-step sampler, each synchronized audio-visual chunk is generated through four denoising steps, followed by an additional clean-context forward at $t = 0$ that prefills the causal KV cache for subsequent chunks. During the rollout, we sample an exit step $t _ { e }$ and record, for each chunk i, the corresponding noisy audio-visual latents $r _ { i } = ( r _ { i } ^ { \bar { v } } , r _ { i } ^ { a } )$ together with the final generated clean latents $\widehat { x } _ { i } = \widehat { ( x _ { i } ^ { v } , x _ { i } ^ { a } ) }$ . Since Forward 1 is executed under no\_grad, it faithfully reproduces the inference-time autoregressive state distribution without retaining the computation graph of the sequential rollout.

Forward 2: Audio-visual context-gradient reconstruction. We then reconstruct the sampled rollout computation in a single differentiable forward pass using the states recorded in Forward 1. Specifically, the generated clean latents $\{ \widehat { x } _ { i } \} _ { i = 1 } ^ { N }$ are provided as stop-gradient context inputs at the clean context timestep $t = 0 .$ , while the recorded noisy audio-visual latents $\{ r _ { i } \} _ { i = 1 } ^ { N }$ retain their sampled exit timestep $t _ { e } .$ . The clean-context and noisy target tokens are jointly arranged according to the causal construction in Eq. (28): the clean latents of chunk i provide autoregressive context for subsequent chunks, whereas $r _ { i }$ represents the prediction target of chunk i at the sampled exit state. This construction reproduces the context–target relation of the serial rollout while allowing all recorded chunks to be evaluated in parallel.

Our LTX-2.3-based audio-visual backbone contains five attention pathways that transmit rolloutdependent information: video self-attention, audio self-attention, audio-to-video (A2V) attention, video-to-audio (V2A) attention, and the spatial attention adapter associated with Unified Camera Positional Encoding (UCPE). As illustrated in Fig. 3(a), we apply the same chunk-level causal attention pattern to all five pathways. For a query associated with chunk i, the corresponding keys and values are restricted to the current or preceding chunks according to the autoregressive rollout geometry. The unimodal masks prevent video and audio tokens from accessing future content; the A2V and V2A masks preserve the same temporal causality during bidirectional cross-modal interaction; and the UCPE mask prevents future camera-conditioned features from leaking into the current prediction. Text cross-attention remains unchanged because the text prompt is globally available and contains no rollout-dependent future information.

Although both the exit-step states and the generated clean latents are detached from Forward 1, their context representations are recomputed with gradient tracking in Forward 2. In particular, the $t = 0$ clean-context computation reconstructs the K/V projections and context attention operations through which generated audio and video chunks are written into causal memory. Later-chunk losses can therefore propagate through the reconstructed context representations in all five attention pathways. This supervision improves not only how each modality writes its own autoregressive history, but also how audio and video histories are exchanged through A2V and V2A attention and how camera information is incorporated through UCPE. Applying the matched causal attention masks allows all chunks to be reconstructed in parallel, yielding the recovered clean audio-visual predictions $\{ \bar { x } _ { i } = ( \bar { x } _ { i } ^ { v } , \bar { x } _ { i } ^ { a } ) \} _ { i = 1 } ^ { N }$

DMD optimization. We optimize the generator with a joint audio-visual DMD objective evaluated on the recovered predictions conditioned on self-generated autoregressive histories. Since these predictions are conditioned on the generator’s own audio-visual history rather than ground-truth context, the objective adapts the generator to the state distribution encountered during deployment. Moreover, because the reconstructed predictions are coupled through the causal A2V and V2A attention pathways, the objective supervises not only the quality of each modality but also the consistency of their future-readable representations. It therefore improves few-step sampling efficiency while mitigating unimodal error accumulation and audio-visual synchronization drift caused by self-generated histories.

The generator is initialized from the causal model obtained after the teacher-forcing stage, while both the real-score and fake-score models are initialized from the pretrained bidirectional diffusion model. The real-score model is kept frozen to represent the target audio-visual distribution, whereas the fake-score model is trainable and continuously updated to track the evolving distribution of the few-step generator.

For a sampled diffusion timestep s, we perturb the recovered clean audio-visual latents as

$$
\bar { x } _ { s , i } = ( 1 - s ) \bar { x } _ { i } + s \eta _ { i } , \qquad \eta _ { i } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{30}
$$

where $\bar { x } _ { i } = ( \bar { x } _ { i } ^ { v } , \bar { x } _ { i } ^ { a } )$ denotes the jointly recovered audio-visual prediction. The frozen real-score model $s _ { \mathrm { r e a l } }$ estimates the score of the pretrained data distribution, while the trainable fake-score model $s _ { \mathrm { f a k e } }$ estimates the score of the current generator distribution. The generator is optimized with the DMD objective

$$
{ \mathcal { L } } _ { \mathrm { S G F - D M D } } = \mathbb { E } _ { i , s , \eta _ { i } } \left[ w ( s ) \ \langle \mathrm { s g } ( s _ { \mathrm { f a k e } } ( { \bar { x } } _ { s , i } , s ) - s _ { \mathrm { r e a l } } ( { \bar { x } } _ { s , i } , s ) ) \cdot { \bar { x } } _ { i } \rangle \right] ,\tag{31}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient and $w ( s )$ is the diffusion-dependent weighting factor. This objective drives the few-step generator toward the distribution represented by the pretrained multi-step model while preserving gradients through the reconstructed causal audio-visual context.

Following each generator update, we perform multiple fake-score updates. The generated samples are detached and used to train the fake-score model with the standard flow-matching objective, allowing it to track the evolving generator distribution. This short-horizon audio-visual SGF stage establishes robust few-step generation under deployment-time autoregressive states before training is extended to substantially longer streaming horizons.

## 5.3 Long-Horizon Audio-Visual Self-Gradient Forcing

Short-horizon audio-visual SGF reconstructs the causal computation within the original training horizon, but sustained streaming generation encounters histories containing progressively accumulated prediction errors. We therefore extend SGF to substantially longer synchronized audio-visual trajectories. The few-step sampler and DMD formulation remain unchanged; the main difference is that Forward 1 exposes the generator to long-horizon self-generated states, while Forward 2 reconstructs the resulting trajectory under a sparse causal attention pattern. This stage adapts both the denoising computation and the audio-visual memory-writing process to the distribution shift encountered during long-running autoregressive generation.

Long-horizon rollout construction. We construct a long-horizon trajectory by sequentially rolling out multiple temporal segments, each corresponding to the training horizon used during short-horizon SGF. To preserve temporal continuity across segment boundaries, the terminal video frame of the k-th rollout segment is reused as the initial frame of the (k + 1)-th segment. Adjacent rollout segments therefore share a boundary frame instead of restarting from independent initial conditions. Throughout the trajectory, all subsequent audio-visual chunks are generated autoregressively from the model’s preceding outputs. Consequently, later segments are conditioned on histories that already contain prediction errors accumulated during earlier segments.

Bounded multimodal causal context. Maintaining a full causal KV cache is impractical for sustained streaming generation because its size grows with the rollout duration. We instead use a sink-plus-FIFO policy that preserves a small persistent prefix together with a bounded window of recent history. This policy is applied consistently to the video, audio, A2V, V2A, and UCPE attention pathways.

Let $S _ { m }$ denote the number of persistent sink chunks and $R _ { m }$ the number of recent chunks retained for modality m. The historical context directly available when generating chunk i is

$$
\mathcal { H } _ { i } ^ { m } = \underbrace { \left( \widehat { x } _ { 1 } ^ { m } , \ldots , \widehat { x } _ { \operatorname* { m i n } ( S _ { m } , i - 1 ) } ^ { m } \right) } _ { \mathrm { a t t e n t i o n ~ s i n k } } \parallel \underbrace { \left( \widehat { x } _ { \operatorname* { m a x } ( S _ { m } + 1 , i - R _ { m } ) } ^ { m } , \ldots , \widehat { x } _ { i - 1 } ^ { m } \right) } _ { \mathrm { r e c e n t \ : F I P O \ : h i s t o r y } } ,\tag{32}
$$

where ∥ denotes ordered concatenation and an indexed range is empty when its lower bound exceeds its upper bound. The attention sink preserves a fixed long-term prefix, whereas the FIFO portion continuously retains the most recent generated chunks. Consequently,

$$
| \mathcal { H } _ { i } ^ { m } | \leq S _ { m } + R _ { m } ,\tag{33}
$$

making the number of historical K/V entries directly accessed at each layer independent of the total rollout duration. The same sink-plus-FIFO policy and causal attention geometry are used during training and inference.

As illustrated in Fig. 3(b), Forward 2 reconstructs the long audio-visual trajectory under the corresponding sink-plus-FIFO masks. The reconstruction retains the long trajectory as a parallel sparse computation graph, while each query is allowed to read only the context specified by its sink-plus-FIFO neighborhood. The clean audio-visual chunks are recomputed at $t = 0$ , and the noisy target chunks retain the sampled exit timestep $t _ { e } ,$ following the same context–target construction as in short-horizon SGF.

Layer-wise long-range context gradients. Importantly, the sink-plus-FIFO bound constrains only the K/V entries that a query can read directly within an individual Transformer layer. It does not restrict the end-to-end receptive field or gradient horizon of the stacked network to the same singlelayer window. At an upper Transformer layer, a recent FIFO chunk visible to the current query already contains information aggregated from its own sink-plus-FIFO neighborhood in preceding layers. A loss on the current chunk can therefore first propagate to the directly visible K/V representations and subsequently continue through their lower-layer hidden states to the K/V-writing computations of earlier chunks. Repeating this process across layers creates a multi-hop context-gradient path whose temporal reach can extend substantially beyond the sink and FIFO entries directly visible at any single layer.

For example, suppose that chunk i directly attends to a recent chunk j at layer ℓ. Although an earlier chunk h may lie outside the direct sink-plus-FIFO window of chunk i, chunk j may have attended to h in a preceding layer. The gradient from the loss on chunk i can then follow the path

$$
\begin{array} { r } { \mathcal { L } _ { i } \longrightarrow \mathrm { K V } _ { j } ^ { ( \ell ) } \longrightarrow \mathrm { K V } _ { h } ^ { ( \ell - 1 ) } , } \end{array}\tag{34}
$$

and analogous paths can be composed across additional layers and intermediate chunks. Hence, $S _ { m } + R _ { m }$ bounds the direct attention fan-in at each layer, but does not bound the effective multi-layer context-gradient reach.

This layer-wise propagation also operates across modalities. A video loss may reach earlier audio representations through A2V attention, an audio loss may reach earlier video representations through V2A attention, and both may propagate through the camera-conditioned representations introduced by UCPE. Moreover, the shared boundary between adjacent rollout segments is preserved during the parallel reconstruction rather than treated as a gradient boundary. Losses from later segments can consequently supervise more distant audio-visual context-writing operations through intermediate chunks, cross-modal attention, and successive Transformer layers.

This transitive supervision remains fundamentally different from full rollout backpropagation. The generated latents, noise states, sampling decisions, and serial KV-cache trajectory recorded in Forward 1 remain detached. Gradients flow only through the parallel Forward-2 reconstruction, including the clean-context projections, the five causally masked attention pathways, and the target-side denoising computation. Long-horizon audio-visual SGF therefore supervises distant memory-writing operations through a bounded-degree sparse reconstruction graph without retaining the recurrent autograd graph of the sequential rollout.

Temporal scope of DMD score evaluation. During long-horizon training, the autoregressive generator retains its original RoPE configuration. The sink-plus-FIFO policy only controls which historical K/V states are retained and visible, without modifying the generator’s positional encoding. Since the real-score and fake-score networks are initialized from a bidirectional diffusion model pretrained on fixed-duration sequences, their DMD evaluations during training are restricted to temporal spans covered by the pretraining horizon. This keeps score-based supervision within the temporal regime observed by the score models while leaving the generator’s positional representation unchanged.

Long-horizon DMD supervision. We apply the DMD objective to every rollout segment instead of supervising only the final segment of the long trajectory. Let $\mathcal { L } _ { \mathrm { D M D } } ^ { ( k ) }$ denote the joint audio-visual DMD loss computed from the recovered predictions of the k-th rollout segment. For a trajectory containing K segments, the long-horizon objective is accumulated across all segments:

$$
\mathcal { L } _ { \mathrm { L o n g - S G F } } = \sum _ { k = 1 } ^ { K } \mathcal { L } _ { \mathrm { D M D } } ^ { ( k ) } .\tag{35}
$$

Supervision is therefore distributed throughout the complete self-generated trajectory. Later-segment losses are evaluated under histories that already contain errors accumulated in earlier segments and, through the layer-wise paths described above, can supervise context-writing computations beyond their directly visible KV windows. Jointly optimizing these segment-wise losses encourages the few-step audio-visual world model to preserve visual identity, scene geometry, audio continuity, and cross-modal synchronization under progressively shifted autoregressive states.

Sparse background temporal regularization. We observe that the four-step distilled LTX generator can occasionally produce transient gray-speckle artifacts in locally stable regions, which become particularly noticeable during long-horizon autoregressive rollouts. To distinguish these artifacts from genuine motion, we compare two consecutive temporal transitions: the historical change from frame $t - 2 \tan t - 1$ , and the current change from frame t − 1 to t.

Specifically, for each generated latent token $\widehat { z } _ { t , p }$ at spatial location $p ,$ we first find its closest latent token in a local neighborhood of frame $t - 1$ . We then trace this matched token back to its closest correspondence in frame t − 2:

$$
q _ { t , p } ^ { * } = \arg \operatorname* { m i n } _ { q \in \mathcal { N } ( p ) } \| \widehat { z } _ { t , p } - \widehat { z } _ { t - 1 , q } \| _ { 2 } , \qquad r _ { t , p } ^ { * } = \arg \operatorname* { m i n } _ { r \in \mathcal { N } ( q _ { t , p } ^ { * } ) } \left\| \widehat { z } _ { t - 1 , q _ { t , p } ^ { * } } - \widehat { z } _ { t - 2 , r } \right\| _ { 2 } .\tag{36}
$$

The corresponding current and historical changes are

$$
d _ { t , p } ^ { \mathrm { c u r } } = \frac { 1 } { \sqrt { C } } \left. \widehat { z } _ { t , p } - \widehat { z } _ { t - 1 , q _ { t , p } ^ { * } } \right. _ { 2 } , \qquad d _ { t , p } ^ { \mathrm { h i s t } } = \frac { 1 } { \sqrt { C } } \left. \widehat { z } _ { t - 1 , q _ { t , p } ^ { * } } - \widehat { z } _ { t - 2 , r _ { t , p } ^ { * } } \right. _ { 2 } ,\tag{37}
$$

where $C$ denotes the latent dimension. If both transitions exhibit comparable changes, the variation is likely caused by persistent object or camera motion. In contrast, a current change that is substantially larger than its historical counterpart is more likely to correspond to a transient artifact.

To account for textured regions and global frame-level variation, we construct an adaptive normalvariation scale $s _ { t , p }$ by combining the historical change, the local spatial variation around the matched token, and a stabilizing frame-wise floor. We select a sparse set $\bar { \mathcal A }$ of locations for which $d _ { t , p } ^ { \mathrm { c u r } } / s _ { t , p }$ is unusually large and penalize only the change exceeding an adaptive tolerance:

$$
\mathcal { L } _ { \mathrm { b g - t e m p } } = \frac { 1 } { | \mathcal { A } | } \sum _ { ( t , p ) \in \mathcal { A } } \rho \Bigl ( \bigl [ d _ { t , p } ^ { \mathrm { c u r } } - \alpha s _ { t , p } \bigr ] _ { + } \Bigr ) ,\tag{38}
$$

where $\rho$ is a robust penalty and α controls the tolerated normal variation. This sparse, motion-adaptive formulation selectively suppresses transient gray-speckle artifacts while preserving persistent scene, object, and camera motion. The complete generator objective is

$$
\mathcal { L } _ { \mathrm { g e n } } = \mathcal { L } _ { \mathrm { L o n g - S G F } } + \lambda _ { \mathrm { b g } } \mathcal { L } _ { \mathrm { b g - t e m p } } ,\tag{39}
$$

where $\lambda _ { \mathrm { b g } }$ controls the strength of the auxiliary regularization.

## 5.4 Autoregressive Inference

At inference, the generator produces synchronized audio-visual chunks autoregressively using the same few-step sampler and sink-plus-FIFO cache introduced during training. For each new chunk, the retained modality-specific histories $\mathcal { H } _ { i } = ( \mathcal { H } _ { i } ^ { v } , \mathcal { H } _ { i } ^ { a } )$ provide the causal context, while outdated FIFO entries are discarded to keep the KV cache bounded. Consistent with long-horizon training, the temporal RoPE indices of the sink, FIFO, and current chunks are rebased into a fixed positional window as generation proceeds, maintaining a consistent temporal positional layout throughout streaming inference.

Let Θ collect the active generator parameters (including the trajectory pathway $\phi$ for the world-model variant), and let $\mathcal { D } _ { i }$ denote the variant-specific conditions: text together with memory and/or an image for the long-video variant, or media context, text, and trajectory for the world-model variant. The next synchronized chunk is sampled as:

$$
( V _ { i } , A _ { i } ) \sim p _ { \Theta } \big ( \cdot \mid \mathcal { H } _ { i } , D _ { i } \big ) ,\tag{40}
$$

where $V _ { i }$ and $A _ { i }$ denote the generated video and audio chunks, respectively.

## 5.5 Towards Causal Memory

Teacher forcing in Eq. (28), long-horizon SGF, and inference already share one causal rule. A noisy chunk may attend only to previously generated clean history. The current clean target and all future chunks remain masked. The sink-plus-FIFO cache in Eq. (32) is the bounded form of this rule. The attention sink preserves a fixed early prefix, the FIFO retains the most recent chunks, and temporal RoPE within each $\mathcal { H } _ { i } ^ { m }$ is rebased as outdated FIFO entries are discarded, so that $\mathcal { H } _ { i } = ( \mathcal { H } _ { i } ^ { v } , \mathcal { H } _ { i } ^ { a } )$ keeps a consistent positional layout throughout streaming generation. This cache is sufficient for local audio-visual continuity under the few-step sampler. It is not sufficient once a chunk has left the FIFO. Those tokens are no longer available, even if the sink still holds the opening of the stream. When the camera returns, or when a model trajectory revisits a previously observed place, the generator should not treat the scene as new merely because the corresponding history has been evicted.

To retain structure beyond the sliding window without changing the causal mask, the few-step sampler, or the variant-specific conditions $\mathcal { D } _ { i }$ in Eq. (40), we introduce a compact recurrent state $\mathcal { M } _ { i }$ <sub>i</sub> that is written only from the past. Let $C _ { i }$ denote a short leak-free prefix taken strictly before the start of chunk i, and let $\psi \subset \Theta$ collect the memory parameters inside the generator, including the trajectory pathway ϕ when the world-model variant is used. The next synchronized chunk and the updated memory are

$$
( V _ { i } , A _ { i } ) \sim p _ { \Theta } \big ( \cdot \mid \mathcal { H } _ { i } , \mathcal { M } _ { i - 1 } , \mathcal { D } _ { i } \big ) , \qquad \mathcal { M } _ { i } = \mathcal { F } _ { \psi } \big ( \mathcal { M } _ { i - 1 } , C _ { i } , V _ { i } \big ) .\tag{41}
$$

Thus $\mathcal { H } _ { i }$ continues to bound attention, while $\mathcal { M } _ { i }$ stores what should remain after the FIFO forgets.

Leak-free prefix. The construction of $C _ { i }$ follows the same constraint as Eq. (28). Every index in $C _ { i }$ satisfies $t < t _ { i } ^ { \mathrm { s t a r t } }$ , so the write never observes the current clean target or any future chunk. The prefix is ordered from oldest to newest and is concatenated in front of the noisy target, so a left-to-right scan matches the chunk order of streaming generation. First-chunk windows that do not contain such a history are dropped during training. Retaining them forces a first-frame fallback and would allow $\mathcal { F } _ { \psi }$ to condition on a target that is still being denoised, which is the analogue of letting $B _ { i , \mathrm { n o i s y } }$ attend to $B _ { i , \mathrm { c l e a n } }$

Target-first rebase. As the FIFO slides, we already rebase temporal RoPE on $\mathcal { H } _ { i } ^ { m }$ . Camera and trajectory encodings that travel with $C _ { i }$ are rebased in the same way, using the first pose of the current chunk as the origin rather than the origin of chunk $i - 1$ . Without this rebase, $\mathcal { M } _ { i - 1 }$ and $C _ { i }$ would remain in the previous chunk’s coordinate system, and the trajectory pathway $\phi$ would observe an artificial jump at every chunk boundary even when the underlying trajectory is smooth.

Block-wise recurrent write. $\mathcal { F } _ { \psi }$ is realized as a sparse state-space scan attached to selected DiT blocks, rather than as a global memory token in text cross-attention. For each spatial token trajectory the module maintains a state $s _ { t }$ and updates

$$
\begin{array} { r } { s _ { t } = \sigma ( \delta ) \odot s _ { t - 1 } + \left( 1 - \sigma ( \delta ) \right) \odot \widetilde { u } _ { t } , \qquad y _ { t } = g _ { t } \odot s _ { t } , } \end{array}\tag{42}
$$

where $\delta$ is a learned per-channel decay, $\widetilde { u } _ { t }$ is a bounded input projection of the normalized block features, and $g _ { t }$ is a sigmoid write gate. The scan runs over the concatenated prefix and target, so the residual at the start of chunk i already contains a summary of $C _ { i }$ . The scan output is added through a residual whose scale is confined to a small interval. The lower bound prevents the recurrence from collapsing to a no-op, and the upper bound prevents it from overwriting the pretrained backbone. After $V _ { i }$ is generated, we retain only the last few latent-aligned steps as $C _ { i + 1 }$ and take the terminal scan state as $\mathcal { M } _ { i }$ . Long-video variants continue to place text and optional image conditions in $\mathcal { D } _ { i }$ and world-model variants continue to condition on $\phi .$ Causal memory does not replace those inputs. It is the component of the history that should remain available after $\mathcal { H } _ { i }$ slides.

The three components admit natural alternatives. $C _ { i }$ can be replaced by a suffix, a nearest-first tail, or FoV retrieval. Actions can keep the previous chunk’s coordinates. $\mathcal { F } _ { \psi }$ can be omitted, given an unbounded residual, or trained on first-chunk windows that have no real prefix. We compared these variants under the same backbone, chunk length, and SGF schedule. The configuration above, leak-free prefix, target-first rebase, bounded block-wise write, and history-bearing training windows, is the one that worked best. It does not change Eq. (40). It only specifies what $\mathcal { H } _ { i }$ contains and how much of that history is written into $\mathcal { M } _ { i }$ before the sink-plus-FIFO window slides.

## 6 Agentic Planning and Control

An overview of our Director Agent workflow is shown in Fig. 4, covering short-video generation and the shot-wise long-video generation loop with prompt enhancement, memory retrieval, review, and memory admission.

Long-form video creation begins with an open-ended user intention, whereas a video generator requires concrete descriptions of what happens in each shot. A short request rarely specifies a complete narrative, stable character identities, scene transitions, dialogue, camera progression, or which earlier visual evidence should be reused. We therefore introduce a Director Agent that organizes generation as a persistent, shot-wise process. The agent develops the user’s intention into a screenplay, applies prompt enhancement (PE), manages visual–audio memory, and coordinates generation and local revision. The video model remains responsible for synthesis, while the agent determines what should be generated and which context should condition it.

A central goal is autonomous completion. After receiving a brief and any optional reference media, the agent can understand the request, plan the story, prepare every shot, generate the synchronized long audio-video sequence, check the intermediate results, update continuity memory, and assemble the final film without waiting for a human decision at every shot. This is the default autonomous path. The same path exposes human-in-the-loop checkpoints: a user can act as a critic, edit the screenplay or current prompt in real time, reselect a memory, and resume generation from the affected shot.

The workflow is organized around a persistent project workspace. The workspace contains the user request, the evolving screenplay, a compact story profile, the ordered shot plan, generated artifacts, the character and scene memory banks, and the decisions made during review. It also records the relation between neighboring shots and the references supplied to each generation request. This state is important for long videos: a later shot is conditioned on accepted visual evidence and on the user’s latest decisions, rather than on a reconstruction of the conversation or on an unverified plan. The workspace can therefore be resumed from an intermediate shot, while a revision changes only the affected shot and its downstream state.

![](images/b33d78cc37b07628e45b171cd178cea37e2cd9cccda96550d49aa0c0f2e68a79.jpg)  
Figure 4: Overview of the Director Agent workflow. Given a user request and optional reference, the agent performs shot planning and prompt enhancement (PE), and supports both single-shot shortvideo generation and iterative long-video generation through memory retrieval, shot-level review and revision, and memory admission.

Recent agentic workflows have been explored for long-video generation [HyCreator Team, 2026]. Our workflow separates three responsibilities that are easy to conflate in a long-video pipeline. Planning decides what the story should contain and how it is divided into observable beats. Generation turns one prepared beat into synchronized audio and video under the selected references. Review and memory admission decide which result is accepted and which visual evidence is safe to expose to later shots. Keeping these decisions explicit makes the workflow inspectable: a failed generation can be revised without silently changing the screenplay, and an unsuitable memory proposal can be replaced without altering the accepted shot.

## 6.1 Planning and Prompt Enhancement

Read the brief and develop the screenplay. Given a user instruction u, the agent first resolves the central event, characters, locations, temporal progression, and intended ending. It produces a complete screenplay before the first generation call, but keeps the individual shot descriptions editable. The screenplay establishes the narrative order, one main dramatic job for each outer shot, and the relation between adjacent shots. An outer shot is the unit submitted to the video generator; any internal camera changes needed to expose one beat remain part of that shot.

The planning record contains a compact story profile in addition to prose. It gives every recurring subject a stable identity, appearance, wardrobe, and voice anchor, and records the scene, important props, viewpoint, and relevant sound context. Stable subject IDs are reused across all later captions. This allows a later mention of a character to refer to an established identity while still permitting the shot to specify a new pose, action, or camera relationship. The profile also distinguishes persistent facts from shot-local facts. For example, a character’s coat and voice can persist, whereas a wet sleeve or a temporary facial reaction belongs only to the shot in which it is observed.

Prepare the shot plan. For every adjacent pair, the planner marks one of three continuity relations: a scene transition, an ordinary edit within the same scene, or one continuous action. A scene transition changes the location or the established spatial state and starts with fresh scene context. An ordinary edit keeps the scene but changes the viewpoint or shot composition. A continuous action preserves the terminal state and motion direction of the previous shot. This relation is written into the shot plan before generation and is later used consistently when choosing a representative scene frame, a true tail frame, and the image references described in the caption.

Construct the production prompt. The screenplay and the PE captions are generated as one planning stage:

$$
C = \{ c _ { t } \} _ { t = 1 } ^ { T } = \operatorname { P E } ( \mathcal { D } ( u ) ) .\tag{43}
$$

Here $\mathcal { D }$ produces the narrative plan, $c _ { t }$ is the enhanced caption for outer shot $t ,$ and T is the number of planned shots. We use $S _ { t } = \mathsf { \bar { ( } } V _ { t } , A _ { t } )$ for the synchronized shot, following Sec. 3.1. The equation describes the interface between planning and generation rather than a new learning objective: the agent writes the conditions and the video model samples the corresponding shot.

Structured caption fields. PE treats a caption as a set of coordinated fields rather than as an undifferentiated paragraph. For shot $t ,$ we write the organization abstractly as

$$
\boldsymbol { c } _ { t } = \left[ \boldsymbol { r } _ { t } ^ { \mathrm { s t o r y } } ; \boldsymbol { r } _ { t } ^ { \mathrm { v i s u a l } } ; \boldsymbol { r } _ { t } ^ { \mathrm { c a m e r a } } ; \boldsymbol { r } _ { t } ^ { \mathrm { a u d i o } } ; \boldsymbol { r } _ { t } ^ { \mathrm { t e m p o r a l } } \right] ,\tag{44}
$$

where the five terms describe the narrative beat, visible subjects and environment, camera and composition, synchronized sound, and the ordered changes within the shot. The fields are written as natural language in the final prompt; the notation only makes their roles explicit.

The story field contains the subjects, action, dialogue, and required ending. The visual field records appearance, clothing, props, scene geometry, lighting, and style. The camera field specifies shot scale, viewpoint, composition, and motivated movement, while the audio field aligns dialogue, Foley, ambience, and music with the visual event. The temporal field gives the initial state, trigger, preparation, continuous action, completion, reaction, and final hold. The fields preserve our identity anchors, scene relations, and audio–visual memory roles while keeping the production plan readable at each stage.

Executable detail and grounding. PE preserves the hard invariants of the request–subjects, identities, props, dialogue, language, duration, and outcome–and expands each beat as initial state → trigger → action → completion → reaction → final hold. It adds only observable evidence that improves execution, such as contact mechanics, environment response, or material-specific sound; it does not introduce a new event, subject, or outcome merely to make the caption longer.

Camera, sound, and references. The camera field is chosen to expose the evidence: an establishing view gives geography, a closer view gives expression or manipulation, and a motivated reframe reveals a consequence. Dialogue, Foley, ambience, and music follow the same timeline. Identity references provide appearance and voice cues, scene references provide layout and lighting, and an opening tail provides the starting state for continuous action. PE names only references supplied to the current shot, keeping text and media conditions consistent.

Initial and image-aware PE. An initial reference image grounds the screenplay in its visible people, objects, composition, viewpoint, and lighting. Immediately before an image-conditioned generation, a second PE pass describes how that state evolves into the planned beat. It preserves subject IDs, dialogue, language, and reference roles while adapting the opening pose and motion. The same rule is used for a user-provided first frame and for the tail frame of an accepted shot. The user may revise the screenplay and shot count before generation; afterward, the confirmed anchors constrain local prompt edits.

## 6.2 Short-Video Generation

Short-video generation uses the same planning, PE, optional image conditioning, generation, and user revision procedure as the long-video workflow, but treats the request as a single-shot screenplay (T = 1). The agent still resolves the event, starting state, ending state, camera observation, and synchronized sound; the difference is that no later shot needs to inherit the result. For a fresh text-to-audio-video request, the enhanced caption is sent directly to the generator. The generated audio and video are sampled together so that the action, speech, Foley, and ambience follow one timeline.

With a user-provided first-frame reference, a multimodal model interprets the visible pose, viewpoint, environment, lighting, and terminal action state. It performs a local image-aware rewrite of the enhanced caption: the opening pose and state-dependent action are adjusted to follow naturally from the frame, while the story beat, subject identities, dialogue, language, and reference roles remain fixed. The first frame is then passed as the image condition. A fresh shot has no image condition; an image-conditioned shot begins from the provided visual state. The same interface supports the audiovisual model’s text, image, and memory conditioning without introducing a separate short-video prompt format.

In autonomous mode, the agent completes the short-video path using its default review and delivery policy. In human-in-the-loop mode, the rendered clip is shown to the user as a critic checkpoint. The user can provide a concrete correction–for example, an appearance, action, dialogue, camera, sound, or continuity change–and the agent rewrites the current caption before regenerating it. Since there is no following shot, the accepted short clip does not update a cross-shot character bank. This keeps the single-shot path simple while preserving the same real-time revision semantics used by the long-video path.

## 6.3 Shot-Wise Long-Video Generation

Autonomous execution. The agent executes the confirmed screenplay in timeline order and prepares the next shot only after the current shot has been resolved. Auto Mode runs this loop from the brief to the final merge with default review and memory policies. The interactive mode exposes the same states to a human critic, who can pause the current shot, modify its prompt or memory choice, and resume from there.

Memory and shot condition. The character memory bank is partitioned by stable character ID, $B _ { t - 1 } = \{ B _ { t - 1 } ^ { ( k ) } \} _ { k }$ , and stores an approved visual observation, paired audio evidence, and its source shot. $\mathrm { A }$ separate full-frame scene record preserves same-scene layout; a true tail frame is reserved for continuous action. Scene transitions start with fresh scene context.

Before generating shot $t ,$ the agent retrieves the entries for the identities in $c _ { t }$ . Let $m _ { t - 1 } ^ { ( k ) }$ denote the selected entry for character $k ,$ , and let $P _ { t }$ denote the same-scene representative; $P _ { t } = \bar { \boldsymbol { \mathcal { D } } }$ after a scene transition. The active memory is

$$
M _ { t - 1 } = \left\{ m _ { t - 1 } ^ { ( k ) } \ | \ k \in \mathrm { I D s } ( c _ { t } ) \right\} \cup P _ { t } .\tag{45}
$$

Here $\mathrm { I D s } ( c _ { t } )$ lists the current subjects. Character entries provide identity and speaker cues, while $P _ { t }$ provides scene context. If the user requests another existing memory, the agent rebuilds $M _ { t - 1 }$ and regenerates only the current shot.

Image-aware shot preparation. The agent supports fresh generation, first-frame generation, samescene reference generation, and continuous generation. Let $I _ { t }$ be the optional image condition: a user first frame, the accepted previous tail, or $\mathcal { D }$ for a fresh shot. Image-aware PE rewrites only the opening state needed to connect $c _ { t }$ to $I _ { t }$ and preserves the story beat, subject IDs, dialogue, and reference roles. The shot is sampled as

$$
S _ { t } \sim p _ { \theta } ( \cdot \mid \widetilde { c } _ { t } , M _ { t - 1 } , I _ { t } ) , \qquad S _ { t } = ( V _ { t } , A _ { t } ) ,\tag{46}
$$

where $\widetilde { c } _ { t }$ is the image-aware caption, $M _ { t - 1 }$ is active memory, and $I _ { t }$ fixes the starting visual state.

Layered review and real-time intervention. After generation, Auto Mode continues with its default policy. In the interactive path, the user can accept the clip or provide critic feedback on appearance, action, dialogue, camera, sound, or continuity. The agent edits the current caption or active memory and regenerates only that shot; rejected artifacts do not enter later context.

Memory admission after shot acceptance. After an accepted shot, candidate frames and synchronized audio are collected. A VLM proposes a subject crop and a scene representative; Auto Mode resolves the proposal by default, while the user may approve or reselect it. Character crops enter the corresponding bank, the scene frame remains continuity-only, and the true tail is kept for continuous generation.

Let $\widehat { m } _ { t } ^ { ( k ) }$ denote the resolved character memory for identity k after the automatic policy or the user’s review. The bank update can then be written as

$$
B _ { t } ^ { ( k ) } = \left\{ \begin{array} { l l } { \mathrm { U p d a t e } \left( B _ { t - 1 } ^ { ( k ) } , \widehat { m } _ { t } ^ { ( k ) } \right) , } & { \widehat { m } _ { t } ^ { ( k ) } \mathrm { ~ i s ~ a d m i t t e d , } } \\ { B _ { t - 1 } ^ { ( k ) } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{47}
$$

![](images/ffa76861d419aee93934fdc6b24e7a90751537d93f19d939c004c6c007401e23.jpg)  
Figure 5: Qualitative effect of the SR stage on generator outputs, concentrated on the two content types that degrade first at the generator’s native resolution: small text and small faces. Smeared strokes on distant storefront signage become legible glyphs, and faces only a few dozen pixels across—including under motion—recover defined features rather than the coarse blobs the generator produces. Each cell shows the generator output with the inspected region boxed (left), followed by that region magnified before and after the SR stage.

where admission is decided automatically or by the user after re-selection. An unresolved proposal leaves the previous bank unchanged.

Assemble and deliver. After memory resolution, the agent prepares the next shot and repeats the loop until all planned shots are accepted. It then merges them in screenplay order with synchronized audio. Auto Mode delivers the completed film directly; interactive mode permits a final whole-film review before delivery.

## 7 Arbitrary-Step Super-Resolution

The generators of the preceding sections produce synchronized audio-visual content at the 720- class resolutions of the training corpus (Sec. 2.1.2); a dedicated super-resolution (SR) stage then restores fine texture and suppresses generation artifacts at the delivery resolution. This factorization concentrates the dominant inference cost in the SR stage, which evaluates a large backbone at full output resolution on every frame; with resolution and duration both fixed by the deliverable, the number of network evaluations is the only lever left. Few-step distillation pulls that lever at training time, baking a single step count into the weights [Yin et al., 2024b,a, Song et al., 2023], while training-free ODE solvers [Lu et al., 2022, Zhao et al., 2023] stay flexible but degrade sharply in the very-few-step regime that matters most here. We instead distill a flow-matching SR teacher [Lipman et al., 2023, Wan Team et al., 2025] with MeanFlow [Geng et al., 2025] into a model of the average velocity over an interval, turning the step count into an inference-time dial: one set of weights serves N-step sampling for any N, with no retraining and no external solver. Fig. 5 illustrates the effect of this stage on generator outputs.

## 7.1 SR as Conditional Generation

Conditioning by latent concatenation. We cast video SR as conditional restoration in the latent space of the Wan2.1 VAE [Wan Team et al., 2025], following the architecture-preserving T2V-to-TV2V super-resolution paradigm of Luxury et al. [2026]. The low-resolution (LR) condition occupies the same latent grid as the target, and its latent $z ^ { \mathrm { L R } }$ is concatenated with the noisy target latent along the channel dimension at the network input, so that the model’s task reduces to re-synthesizing the high-frequency content the condition does not contain. To discourage the trivial shortcut of copying the condition, the condition is stochastically corrupted,

$$
\begin{array} { r } { y = ( 1 - \alpha ) z ^ { \mathrm { L R } } + \alpha \epsilon _ { y } , \qquad \alpha \sim \mathcal { U } ( 0 . 3 , 0 . 6 ) , \quad \epsilon _ { y } \sim \mathcal { N } ( 0 , { \bf I } ) , } \end{array}\tag{48}
$$

with α redrawn on every forward pass; the same perturbation is applied at training and sampling.

AIGC-oriented degradation. Because this SR stage is cascaded after a generator rather than applied to camera footage, its condition distribution is dominated by generation artifacts rather than sensor noise. Classical blind-SR pipelines [Wang et al., 2021, Chan et al., 2022] synthesize blur, noise, and compression independently per frame and are therefore mismatched, whereas diffusion-generated video fails through temporally structured modes such as flicker, motion jitter, and color shift. We adopt the AIGC-oriented degradation design of Luxury et al. [2026], prepending temporally coherent degradations whose parameters follow low-frequency trajectories in time, and annealing the overall strength toward a weak final setting matched to the upstream generator. On the target side, an in-batch perceptual-quality gate keeps supervision clean by replacing rejected clips with temporally flipped copies of retained ones.

Text-condition annealing. The teacher is adapted from a text-to-video (T2V) backbone into a text-and-video-to-video (TV2V) restorer, and we want the resulting model to super-resolve faithfully without relying on a caption at inference: a delivery-time SR stage should not require the upstream prompt to be available, nor should its fidelity depend on how well that prompt is written. Removing the caption from step 0 is nonetheless the wrong way to obtain this, because the backbone’s crossattention and modulation pathways were pretrained exclusively on caption-conditioned generation; forcing them out of that regime at the same moment the newly added condition pathway is being learned makes two adaptations compete. We therefore adopt the prompt-annealing schedule of Lu et al. [2026]: the probability of using the real caption decays linearly to zero over the early phase of training, after which a single fixed restoration prompt serves every sample. Annealing separates the two adaptations in time—the text pathway stays in its pretrained regime while gradient signal concentrates on the condition projection, and only then does the text distribution collapse onto the fixed prompt. The resulting model is caption-free at inference while retaining the generative prior that supplies detail the LR input does not contain.

Teacher objective. The teacher is trained with the standard conditional flow-matching objective [Lipman et al., 2023], consistent with the convention of Eq. (27): for a clean latent $x _ { 0 }$ , noise $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ , and noise level $t \in ( 0 , 1 )$ , the noisy latent is $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon$ and the network regresses the straight-path velocity,

$$
\mathcal { L } _ { \mathrm { S R } } = \mathbb { E } _ { x _ { 0 } , y , t , \epsilon } \left[ \left\| v _ { \theta } ( x _ { t } , t , y ) - ( \epsilon - x _ { 0 } ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{49}
$$

where t is drawn from a logit-normal distribution warped by a time-shift transform that concentrates training density at higher noise levels, y is the noise-perturbed condition of Eq. (48), and the text condition follows the annealing schedule above. The teacher is obtained by approximately 10.4k optimizer steps of SR fine-tuning through a staged curriculum over degradation strength and resolution, ending on multi-aspect-ratio buckets that match the intended delivery resolutions. At inference, this teacher is a conventional multi-step flow model: producing a single output requires tens of solver steps [Zhao et al., 2023], each a full forward pass at the output resolution, which is precisely the cost that the next subsection removes.

## 7.2 Distilling an Arbitrary-Step Sampler

Average velocity. MeanFlow [Geng et al., 2025] replaces the instantaneous velocity $v ( x _ { t } , t )$ with the average velocity over an interval $[ r , t ]$

$$
u ( x _ { t } , r , t ) = \frac { 1 } { t - r } \int _ { r } ^ { t } v ( x _ { \tau } , \tau ) \mathrm { d } \tau , \qquad 0 \leq r < t \leq 1 ,\tag{50}
$$

so that the exact solution of the probability-flow ODE over $[ r , t ]$ is the single subtraction $\scriptstyle x _ { r } =$ $x _ { t } - ( t - r ) u ( x _ { t } , r , t )$ . A network $u _ { \theta } ( x _ { t } , r , t )$ that models this quantity is an interval integrator by construction: it does not approximate a step of an ODE solver, it is the step, for any interval it is queried on.

MeanFlow identity and training objective. Differentiating Eq. (50) along the trajectory yields the MeanFlow identity

$$
u ( x _ { t } , r , t ) = v ( x _ { t } , t ) - ( t - r ) { \frac { \mathrm { d } } { \mathrm { d } t } } u ( x _ { t } , r , t ) , \qquad { \frac { \mathrm { d } u } { \mathrm { d } t } } = \partial _ { x } u \cdot v + \partial _ { t } u ,\tag{51}
$$

which converts the intractable interval integral into a local regression target: the known conditional velocity $\epsilon - x _ { 0 }$ corrected by a Jacobian–vector product (JVP) of the network itself, evaluated under stop-gradient. We train with an explicit-gradient form of this regression,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F } } = \mathbb { E } \left[ \left| \left| u _ { \theta } - \mathrm { s g } \big ( u _ { \theta } + w \Delta \big ) \right| \right| _ { 2 } ^ { 2 } \right] , \qquad \Delta = ( \epsilon - x _ { 0 } ) - u _ { \theta } - \lambda ( t - r ) \frac { \widehat { \mathrm { d } u } } { \mathrm { d } t } , } \end{array}\tag{52}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient and $\overline { { \mathrm { d } u } } / \overline { { \mathrm { d } t } }$ is a stop-gradient estimate of the total derivative, so that the gradient with respect to $u _ { \theta } \mathrm { i s } - 2 w \Delta$ and the update direction is set directly. The per-sample weight $\bar { w } = 1 / ( \| \Delta \| + c ) $ is computed under stop-gradient—a gradient-carrying weight would open a backpropagation path that rewards enlarging the loss to shrink the weight—and down-weights high-noise samples where the JVP estimate is least stable. The tangent norm entering w is rescaled to be invariant to spatial and temporal tensor shape, which keeps a single c valid across all resolution buckets. The factor λ linearly ramps from 0 to 1 over the first 2,000 steps, interpolating smoothly from pure flow matching to the full MeanFlow objective while the network acquires its interval dependence. The pair $( t , r )$ is drawn as the maximum and minimum of two independent samples from the same shifted logit-normal family used for teacher training (time-shift 2.0 in the distillation stage), so no new timestep schedule is introduced at distillation time and every sample trains the genuine interval objective $( t > r )$

Jacobian–vector products by finite differences. Exact forward-mode JVPs are unavailable in our training stack: fused attention kernels do not implement forward-mode rules, non-reentrant activation checkpointing does not propagate tangents through the recomputed forward, and the sharded dataparallel backward does not support them. We therefore estimate $\mathrm { d } u / \mathrm { d } t$ by a central finite difference along the ODE trajectory: perturbing $( x _ { t } , t )$ jointly by $\pm h$ in the direction $( v , 1 )$ captures both the $\partial _ { x } { u } \cdot { v }$ and $\partial _ { t } \boldsymbol { u }$ terms of Eq. (51) in a single difference. The two probe evaluations run without gradient tracking, with network outputs kept in full precision and the differencing performed in double precision, since half-precision mantissas are destroyed by the near-cancelling subtraction; both probes share an identical random state and a pinned condition tensor $y ,$ so that the stochastic conditioning noise of Eq. (48) cannot leak into the difference. One-sided differences are substituted per sample when the interval boundary would be crossed. The step size h was calibrated by sweeping against a higher-precision reference evaluation of the production model—an internal engineering measurement at a single operating point rather than an ablation—and the chosen value sits in the regime where truncation and round-off errors balance; the adaptive weight and warmup in Eq. (52) absorb the residual magnitude error, as the objective chiefly consumes the tangent’s direction. Measured on our configuration, the estimator costs $2 . 0 \times$ the step time of plain flow matching and no additional activation memory, since peak memory remains set by the single gradient-carrying forward.

Arbitrary-step sampling. The distilled model needs no scheduler, no solver, and no per-budget retraining. Given any budget of K network evaluations, we partition the noise axis $1 = t _ { 0 } > t _ { 1 } >$ $\cdots > t _ { K } = 0$ by warping a uniform grid through the same time-shift transform used in training,

$$
t _ { k } = { \frac { s \left( 1 - { \frac { k } { K } } \right) } { 1 + ( s - 1 ) \left( 1 - { \frac { k } { K } } \right) } } , \qquad k = 0 , \ldots , K ,\tag{53}
$$

with shift $s = 2 . 0$ and the endpoints fixed to the maximal training noise level and to zero, which concentrates the few available evaluations in the high-noise region where training density was placed, and iterate the exact interval update

$$
x _ { t _ { k + 1 } } = x _ { t _ { k } } - \left( t _ { k } - t _ { k + 1 } \right) u _ { \theta } \big ( x _ { t _ { k } } , t _ { k + 1 } , t _ { k } \big ) , \qquad k = 0 , \ldots , K - 1 .\tag{54}
$$

The sampler consists solely of the grid of $\operatorname { E q . } \left( 5 3 \right)$ and the subtraction in Eq. (54); no scheduler object exists. Setting $K = 1$ gives one-step generation; larger K trades computation for fidelity along a continuum, and during training we render $K \in \{ 1 , 2 , 4 \}$ side by side from the same checkpoint at every validation. This is the deployment property we set out to obtain: the SR cost per output pixel becomes K forward passes with K chosen at request time, so a single model serves latency-critical interactive use at one step and quality-leaning offline rendering at higher step counts, replacing the tens of solver steps required by the teacher. We emphasize what this is not. It is not a fixed-step distilled student in the sense of Secs. 3.2 and 5.2, whose DMD objectives [Yin et al., 2024b,a] commit to a specific sampler at training time; and it is not a model to be driven by a multi-step corrector [Lu et al., 2022, Zhao et al., 2023], which would be both redundant—the subtraction in Eq. (54) is already the exact interval integral—and incorrect, since such correctors assume the network predicts the instantaneous velocity $v ,$ whereas $u _ { \theta }$ predicts the average velocity $u \ne v$

## 7.3 Warm-Starting Without Loss

The degenerate diagonal. MeanFlow contains flow matching as its boundary case: when $r = t ,$ Eq. (50) reduces to $\boldsymbol { u } ( x _ { t } , t , t ) = \boldsymbol { v } ( x _ { t } , t )$ and the target in Eq. (52) degenerates exactly to the flowmatching target of Eq. (49). The converged multi-step teacher therefore already solves the MeanFlow objective on the r = t diagonal of the $( r , t )$ plane, and distillation amounts to extending a function the teacher has learned on a line to the triangle above it—not to training a different model that must first re-approach the teacher.

Table 2: Quantitative comparison of baseline methods. ViCLIP and Self-CIDS assess cross-shot visual and identity consistency, Voice assesses speaker consistency, and Recall measures speechcontent coverage. All metrics are higher-is-better. Boldface and underlining indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">Cross-Shot Consistency</td><td colspan="2">Video Quality</td><td>Text Consistency</td><td>Speech</td></tr><tr><td>ViCLIP↑</td><td>Self-CIDS ↑</td><td>Voice ↑</td><td>Aesthetic ↑</td><td>Imaging ↑</td><td>CLIP↑</td><td>Recall ↑</td></tr><tr><td>JavisDiT++ [Liu et al., 2026]</td><td>0.6955</td><td>0.5621</td><td>0.7933</td><td>0.5019</td><td>0.6084</td><td>0.2522</td><td>0.0951</td></tr><tr><td>LTX-2.3 [HaCohen et al., 2026]</td><td>0.7047</td><td>0.6135</td><td>0.6847</td><td>0.5436</td><td>0.4751</td><td>0.2559</td><td>0.9304</td></tr><tr><td>LTX-2.5 [HaCohen et al., 2026]</td><td>0.7775</td><td>0.6640</td><td>0.7603</td><td>0.6019</td><td>0.5804</td><td>0.2683</td><td>0.9545</td></tr><tr><td>ShotStream+MMAudio [Luo et al., 2026]</td><td>0.7988</td><td>0.7308</td><td>0.7945</td><td>0.5255</td><td>0.6044</td><td>0.1998</td><td>0.0297</td></tr><tr><td>StoryMem+MMAudio [Zhang et al., 2025b]</td><td>0.7897</td><td>0.7491</td><td>0.7643</td><td>0.5265</td><td>0.6320</td><td>0.2368</td><td>0.0301</td></tr><tr><td>HappyOyster (Directing)</td><td>0.7535</td><td>0.6940</td><td>0.7705</td><td>0.5606</td><td>0.6701</td><td>0.2544</td><td>0.1085</td></tr><tr><td>JoyAI-Echo-1.0 [Li et al., 2026]</td><td>0.8026</td><td>0.7793</td><td>0.8129</td><td>0.5679</td><td>0.7058</td><td>0.2658</td><td>0.9489</td></tr><tr><td>JoyAI-Echo-1.5</td><td>0.8264</td><td>0.7937</td><td>0.8524</td><td>0.5705</td><td>0.7467</td><td>0.2868</td><td>0.9674</td></tr></table>

Zero-initialized interval branch. To realize this extension without disturbing the teacher, the student architecture augments the backbone with a second conditioning branch that mirrors the timestep pathway: a sinusoidal embedding of the interval length t − r, passed through its own MLP and its own modulation projection, whose outputs are added to the corresponding timestep-branch activations. Feeding t − r rather than r makes the degenerate case correspond to a fixed zero input, and keeping the projections separate—each branch projected before summation—decouples the new branch’s gradients from the pretrained modulation weights. The terminal linear layer of each new pathway is initialized to zero, so at step 0 the branch contributes exactly nothing and the student’s forward pass reproduces the teacher’s by construction; a start-of-training self-check verifies that the loss pipeline degenerates to flow matching when r = t, confirming the wiring. The warm start is thus lossless: no adapter distillation, no feature matching, and no re-convergence phase are needed.

Curriculum and confound control. Distillation is full-parameter and inherits the teacher’s entire conditioning stack unchanged—the latent concatenation and the condition noising of Eq. (48), the degradation pipeline, the paired HR/LR data, and the per-stage resolution buckets—so that any behavioral difference between student and teacher is attributable to the objective rather than to a data-distribution shift. Training proceeds in two stages: a first stage on moderate-resolution buckets initialized from the converged teacher, followed by a continuation on the teacher’s own high-resolution multi-aspect buckets with the optimizer state carried over; the shape-invariant tangent normalization of Sec. 7.2 is what allows this resolution change without retuning the loss weighting. The optimizer follows the teacher’s settings except for a raised second-moment coefficient $( \bar { \beta } _ { 2 }$ from 0.95 to 0.99), accommodating the higher gradient variance of the finite-difference target, and the high-resolution stage adds a cross-rank finiteness gate that skips optimizer steps on non-finite gradients before clipping, preserving the Adam state through transient instabilities and closing a failure mode in which a NaN gradient silently bypasses standard norm clipping because the comparison nan < 1.0 evaluates to false. Combined with the tangent warmup, the overall trajectory is a continuous deformation of the teacher: the student begins as the teacher, is annealed onto the MeanFlow objective, and ends as a single set of weights for which Eq. (54) holds at every step budget.

## 8 Results

We evaluate JoyAI-Echo-1.5 from two complementary perspectives: long-horizon audio-visual generation and action-conditioned world modeling. For world modeling, we report results on two public benchmarks, WBench and SANA-WM-Bench, together with a large-scale human preference study.

## 8.1 Long-Horizon Audio-Visual Generation

Experimental Setup. We evaluate long-horizon audio-visual generation on a curated benchmark of 100 stories and 3,000 shots, following the protocol of JoyAI-Echo-1.0 [Li et al., 2026]. We consider four complementary aspects:

• Cross-shot consistency. We report pairwise ViCLIP similarity [Wang et al.] for visual-semantic consistency, Self-CIDS for character identity consistency using GroundingDINO [Liu et al., 2023], Torchreid [Zhou and Xiang, 2019], and FaceNet [Schroff et al., 2015], and 3DSpeaker similarity [Chen et al., 2025] for speaker consistency.

• Video quality. Following VBench-Long [Huang et al., 2024], we concatenate all generated shots into a long video, split it into 2-second clips, and evaluate aesthetic and imaging quality.

• Text consistency. We measure alignment between the generated visual content and the input story script using CLIP score [Radford et al., 2021].

• Speech-content fidelity. We transcribe generated speech with Whisper [Radford et al., 2022] and compute word-level recall against the reference dialogue. We use recall instead of the accuracy metric in JoyAI-Echo-1.0 and re-evaluate all methods under the same protocol.

Unless otherwise specified, all JoyAI-Echo-1.5 results are generated with the Director Agent enabled for shot-level planning and memory selection.

Baselines. We compare JoyAI-Echo-1.5 with three classes of audio-visual generation systems:

• Joint audio-visual generation: JavisDiT++ [Liu et al., 2026], LTX-2.3, and LTX-2.5 [Ha-Cohen et al., 2026]. Each method generates approximately 10-second shots from the corresponding structured prompts, which are then concatenated for long-form evaluation. For LTX-2.3 and LTX-2.5, we enable prompt enhancement and condition each shot after the first on the final frame of the preceding shot.

• Cascaded generation: ShotStream [Luo et al., 2026]+MMAudio [Cheng et al., 2025] and StoryMem [Zhang et al., 2025b]+MMAudio. The video model first generates a silent 10-second shot, after which MMAudio synthesizes audio from the corresponding shot-level audio description. The resulting shots are concatenated using the same story scripts and shot durations.

• Native long-form generation: HappyOyster in Directing mode,<sup>1</sup> which directly generates continuous audio-visual sequences from narrative scripts.

Quantitative Comparison. Tab. 2 compares JoyAI-Echo-1.5 with native audio-visual generators, cascaded long-video systems, and JoyAI-Echo-1.0. JoyAI-Echo-1.5 achieves the best performance on six of the seven metrics and ranks second on aesthetic quality. In particular, it obtains the highest ViCLIP (0.8264), Self-CIDS (0.7937), and Voice (0.8524) scores, improving over JoyAI-Echo-1.0 by 0.0238, 0.0144, and 0.0395, respectively. These results demonstrate substantially improved cross-shot visual and speaker consistency. Meanwhile, JoyAI-Echo-1.5 achieves the best Imaging (0.7467), CLIP (0.2868), and speech Recall (0.9674), while remaining competitive in aesthetic quality. This indicates that the gains in long-range consistency are achieved without sacrificing local generation quality, text alignment, or speech fidelity.

User Study. We further conduct a human evaluation comparing the long-video variant of JoyAI-Echo-1.5 with HappyOyster in Directing mode through randomized and anonymized side-by-side comparisons. Following the Good–Similar–Bad (GSB) protocol, annotators evaluate story-instruction following, dialogue/background-audio following, memory/identity consistency, audio-visual synchronization, and audio quality. For each dimension, they determine whether JoyAI-Echo-1.5 is better, the two results are similar, or HappyOyster is better.

As shown in Tab. 3, JoyAI-Echo-1.5 receives a higher preference rate than HappyOyster across all five dimensions. The largest advantage is observed in audio-visual synchronization, where JoyAI-Echo-1.5 is preferred in 48.4% of the comparisons, compared with 20.3% for HappyOyster. JoyAI-Echo-1.5 also demonstrates clear advantages in story-instruction following (38.6% vs. 17.7%) and dialogue/background-audio following (39.2% vs. 23.5%).

![](images/489dfa1f8cd6edfcb88a5f1851f9f611925deca47690853eb6e205ae38910806.jpg)  
Figure 6: Qualitative visualization of a 10-minute long audio-visual sequence generated under the Director Agent. The 10-minute example follows a wartime evacuation story set in a neutral port city. A radio operator helps two civilians escape as military control tightens and the city gradually descends into chaos. Their escape becomes increasingly entangled with checkpoints, shifting alliances, and the growing danger faced by civilians left behind. The radio operator eventually turns his broadcast network into a broader evacuation effort, connecting scattered groups and helping them find a route out of the city.

For memory/identity consistency, JoyAI-Echo-1.5 is preferred in 24.8% of the comparisons, compared with 15.7% for HappyOyster, while 59.5% are judged similar. This indicates that both systems maintain identity adequately in many cases, but JoyAI-Echo-1.5 is more frequently preferred when perceptible differences occur. JoyAI-Echo-1.5 also achieves a higher preference rate in audio quality (44.4% vs. 33.3%). Overall, the user study demonstrates that the improvements measured by the automatic metrics remain perceptible in complete long-video generation, particularly in instruction following, audio-visual synchronization, and long-range consistency.

Qualitative Comparison. Fig. 7 compares JoyAI-Echo-1.5 with JoyAI-Echo-1.0 using the same long-form story and shot-level instructions. Compared with JoyAI-Echo-1.0, JoyAI-Echo-1.5 more reliably preserves the facial structure, hairstyle, clothing cues, and speaker identity of recurring characters across temporally distant shots. It also exhibits more stable exposure and color balance, with less accumulation of oversaturation, contrast artifacts, and scene drift over extended generation. These improvements are particularly visible after multiple scene changes, camera scale variations, and the introduction of new characters. In addition to the benchmark evaluation, we present a separate 10-minute full-system case study in Fig. 6. Despite the extended temporal horizon and frequent shot transitions, the generated sequence maintains recognizable character identities, coherent scene evolution, and stable visual quality across distant timestamps. This example demonstrates the system’s ability to sustain consistent long-form generation far beyond the duration of individual shots.

![](images/8c4a4a99c69f40d366fd275410dee7af6ac73ce87ff3517bb219c1f53700caf3.jpg)  
Figure 7: Qualitative comparison between JoyAI-Echo-1.5 and JoyAI-Echo-1.0 under the same long form story. Frames sampled from temporally separated shots demonstrate the improved character consistency, scene stability, and visual rendering of JoyAI-Echo-1.5.

Table 3: Human evaluation based on Good–Similar–Bad (GSB) pairwise comparisons between JoyAI-Echo-1.5 (long-video variant) and HappyOyster for long-horizon audio-visual generation. Numbers denote the percentage of user preferences and may not sum to exactly 100% because of rounding.
<table><tr><td>Aspect</td><td>JoyAI-Echo-1.5</td><td>Similar</td><td>HappyOyster</td></tr><tr><td>Story-instruction following</td><td>38.6%</td><td>43.8%</td><td>17.7%</td></tr><tr><td>Dialogue/background-audio following</td><td>39.2%</td><td>37.3%</td><td>23.5%</td></tr><tr><td>Memory/identity consistency</td><td>24.8%</td><td>59.5%</td><td>15.7%</td></tr><tr><td>Audio-visual synchronization</td><td>48.4%</td><td>31.4%</td><td>20.3%</td></tr><tr><td>Audio quality</td><td>44.4%</td><td>22.2%</td><td>33.3%</td></tr></table>

These examples illustrate the complementary roles of the model and the long-video generation pipeline. At the generator level, unified audio-visual Memory conditioning provides persistent identity and speaker cues while still allowing each target shot to follow its own textual description. The high-quality training corpus improves local rendering quality and broadens the model’s spatial and linguistic coverage, while transition-aware multi-shot captions strengthen its ability to represent changes in subject, action, camera view, and scene context around shot boundaries. At the system level, the Director Agent selects grounded Memory observations and revises the shot-level plan before subsequent generation, reducing the propagation of incorrect characters, actions, or scene attributes. Compared with JoyAI-Echo-1.0, JoyAI-Echo-1.5 therefore exhibits improvements at both local and long-range scales. Individual shots show clearer structure and more balanced rendering, while the full sequence exhibits less character drift, scene drift, and color accumulation over time. The qualitative results are consistent with the improvements in imaging quality, CLIP alignment, cross-shot visual consistency, and speaker consistency reported in Tab. 2. A more complete long-horizon visualization with densely sampled timestamps is provided in the supplementary material.

Table 4: Comparison on the 158-case WBench Navigation split [Ying et al., 2026]. JoyAI-Echo-1.5 denotes our undistilled bidirectional multi-step model, while JoyAI-Echo-1.5-Causal denotes its distilled four-step causal variant. Results are sorted by Average. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Model</td><td>Avg.</td><td>Quality</td><td>Setting</td><td>Interaction</td><td>Consistency</td><td>Physical</td></tr><tr><td>JoyAI-Echo-1.5</td><td>81.7</td><td>81.5</td><td>79.4</td><td>87.2</td><td>89.8</td><td>70.6</td></tr><tr><td>JoyAI-Echo-1.5-Causal</td><td>81.0</td><td>81.1</td><td>88.3</td><td>87.9</td><td>77.5</td><td>70.1</td></tr><tr><td>HiDream-O1-World [Ying et al., 2026]</td><td>80.7</td><td>81.9</td><td>81.9</td><td>79.5</td><td>88.0</td><td>72.1</td></tr><tr><td>LingBot-World (fast v2) [Robbyant Team et al., 2026]</td><td>79.4</td><td>81.8</td><td>76.8</td><td>82.8</td><td>86.5</td><td>69.1</td></tr><tr><td>Kling 3.0 [Ying et al., 2026]</td><td>79.0</td><td>81.4</td><td>91.0</td><td>69.4</td><td>83.7</td><td>69.3</td></tr><tr><td>LingBot-World (base-camera) [Robbyant Team et al., 2026]</td><td>78.5</td><td>78.9</td><td>72.6</td><td>80.1</td><td>89.9</td><td>71.2</td></tr><tr><td>Wan 2.7 [Wan Team et al., 2025]</td><td>78.1</td><td>81.5</td><td>91.4</td><td>64.4</td><td>81.6</td><td>71.8</td></tr><tr><td>HY-World 1.5 (AR-distill) [Sun et al., 2025]</td><td>78.1</td><td>78.1</td><td>72.2</td><td>86.8</td><td>86.9</td><td>66.3</td></tr><tr><td>HY-Video 1.5 [Wu et al., 2025]</td><td>77.9</td><td>77.6</td><td>85.6</td><td>71.4</td><td>87.4</td><td>67.4</td></tr><tr><td>LingBot-World (fast) [Robbyant Team et al., 2026]</td><td>77.4</td><td>79.4</td><td>77.9</td><td>79.2</td><td>84.9</td><td>65.7</td></tr><tr><td>HappyOyster [Ying et al., 2026]</td><td>76.8</td><td>77.3</td><td>74.2</td><td>84.9</td><td>84.3</td><td>63.5</td></tr><tr><td>Lyra 2.0 (4-step AR) [Shen et al., 2026]</td><td>76.4</td><td>77.1</td><td>73.2</td><td>85.6</td><td>79.3</td><td>66.7</td></tr><tr><td>Seedance 1.5 [Seedance et al., 2025]</td><td>76.2</td><td>82.1</td><td>82.9</td><td>66.3</td><td>81.3</td><td>68.4</td></tr><tr><td>Cosmos3-Super [NVIDIA, 2026]</td><td>76.0</td><td>76.4</td><td>91.6</td><td>58.0</td><td>85.0</td><td>69.2</td></tr><tr><td>SANA-WM (4-step AR) [Zhu et al., 2026]</td><td>76.0</td><td>79.3</td><td>76.1</td><td>82.2</td><td>80.7</td><td>61.9</td></tr><tr><td>DreamX-World (5B AR) [DreamX Team et al., 2026]</td><td>75.0</td><td>77.5</td><td>80.8</td><td>78.6</td><td>74.9</td><td>63.3</td></tr><tr><td>ABot-World [Jiang et al., 2026]</td><td>74.7</td><td>76.8</td><td>71.4</td><td>84.0</td><td>79.5</td><td>61.7</td></tr><tr><td>Cosmos 2.5 [NVIDIA, 2025]</td><td>74.6</td><td>72.9</td><td>83.3</td><td>63.1</td><td>86.5</td><td>67.4</td></tr><tr><td>Cosmos3-Nano [NVIDIA, 2026]</td><td>74.2</td><td>77.4</td><td>87.3</td><td>59.1</td><td>83.7</td><td>63.6</td></tr><tr><td>LTX-2.3 [HaCohen et al., 2026]</td><td>74.2</td><td>77.1</td><td>85.2</td><td>66.4</td><td>77.2</td><td>64.9</td></tr><tr><td>Genie 3 [Google DeepMind, 2025]</td><td>73.9</td><td>75.2</td><td>72.5</td><td>73.4</td><td>82.6</td><td>65.7</td></tr></table>

## 8.2 Action-Conditioned World Modeling

We evaluate two world-model variants of JoyAI-Echo-1.5: the original undistilled bidirectional multi-step model, denoted as JoyAI-Echo-1.5, and its distilled causal four-step variant, denoted as JoyAI-Echo-1.5-Causal. We benchmark these models on two public benchmarks, WBench [Ying et al., 2026] and SANA-WM-Bench [Zhu et al., 2026], together with a subjective evaluation on our self-built World-Model Benchmark (WMB). The former evaluates broad interactive-world behavior, while the latter focuses on camera-trajectory following and long-horizon visual persistence.

## 8.2.1 WBench

Evaluation settings and metrics. We evaluate both JoyAI-Echo-1.5 and JoyAI-Echo-1.5-Causal on the official 158-case Navigation split of WBench [Ying et al., 2026], which covers first- and third-person interactive navigation. Following the official multi-turn evaluation protocol, we report five dimensions—Quality, Setting, Interaction, Consistency, and Physical—together with their overall Average score. Quality aggregates perceptual and temporal video-quality metrics; Setting evaluates scene and subject adherence; Interaction measures compliance with the requested interaction, including navigation trajectory following; Consistency aggregates spatial, geometric, photometric, perspective, subject, and segment-level consistency; and Physical measures visual plausibility and causal fidelity.

Quantitative Results. As shown in Tab. 4, both variants of our model rank at the top of the WBench Navigation leaderboard. The undistilled bidirectional multi-step JoyAI-Echo-1.5 achieves the highest overall Average score of 81.7, while the distilled four-step JoyAI-Echo-1.5-Causal ranks second with 81.0. This small performance gap indicates that causal few-step distillation largely preserves the interactive world-modeling capability of the original multi-step model.

JoyAI-Echo-1.5 achieves a Consistency score of 89.8, the second highest among all evaluated models, demonstrating strong world-state preservation across multi-turn interactions. Notably, JoyAI-Echo-1.5-Causal attains the highest Interaction score of 87.9, slightly surpassing the undistilled model at 87.2. Despite the transition from bidirectional multi-step generation to four-step causal autoregressive inference, the distilled model therefore retains strong control responsiveness while remaining competitive in overall generation quality.

![](images/9c1b4b13871a6f7054badedc557a6a420ebafa51e454dc2e7801dad0a31a5d64.jpg)  
Figure 8: Qualitative comparison of the undistilled bidirectional multi-step JoyAI-Echo-1.5 on representative WBench Navigation cases across diverse visual domains. Given the same initial observation and navigation condition, our model better preserves scene identity and structure while producing more coherent viewpoint evolution over successive interaction steps.

![](images/067f818c261eb9da7b3666af1913c8c9533d131018ad077894ecc4d35f33e169.jpg)  
Figure 9: Multi-turn comparison of JoyAI-Echo-1.5-Causal against representative few-step autoregressive world models on WBench Navigation. All models start from the same initial observation and follow the prescribed multi-turn action sequence. The three cases use (1) A → A → A → A, (2) Left → Left → Right → Right, and (3) $\mathtt { L e f t } \to \mathtt { W } \to \mathtt { R i g h t }$ . Here, Left, Right, Up, and Down denote camera rotations, while W, S, A, and D denote camera translations. JoyAI-Echo-1.5-Causal exhibits stronger action adherence and better preserves scene style, subject identity, and world structure across successive turns, with less accumulated autoregressive drift.

0s  
12s  
24s  
36s  
48s  
60s  
![](images/ecc86126d96724ce3a7432a76b0e4a22b5d1901c0cbe62e1996a51f8383b2735.jpg)  
Figure 10: Causal long-horizon generation of JoyAI-Echo-1.5-Causal on representative cases. Each row shows a 60-second causal rollout sampled at different timestamps. Despite prolonged autoregressive generation, the distilled four-step causal model remains responsive to navigation controls while preserving scene identity, subject appearance, visual style, and major spatial structure, demonstrating robustness to long-horizon error accumulation.

Qualitative Results. Fig. 8 first compares the undistilled bidirectional multi-step JoyAI-Echo-1.5 with representative world models on WBench Navigation cases. Given the same initial observation and navigation condition, competing methods often exhibit abrupt viewpoint changes, unintended zooming, subject-scale variation, layout drift, or loss of scene identity. In contrast, JoyAI-Echo-1.5 produces more coherent world evolution across diverse visual domains, with stronger navigation adherence and better preservation of scene structure and appearance. These observations are consistent with its strong Interaction and Consistency scores in WBench.

We next compare causal few-step world models in Fig. 9, including DreamX-World, LingBot2-Fast, SANA-WM Streaming with and without Causal Refiner, Evoke, and our JoyAI-Echo-1.5-Causal. We evaluate three representative multi-turn interaction cases: A → A → A → A, Left → Left → Right → Right, and Left → W → Right, respectively. Here, Left, Right, Up, and Down denote camera rotations, whereas W, S, A, and D denote translational movements. Among the representative baselines, LingBot2-Fast and Evoke remain relatively competitive across multiple interactions, yet still exhibit noticeable action deviation together with gradual degradation in subject identity and visual style as the rollout progresses. Other baselines show more pronounced viewpoint drift, structural changes, or accumulated appearance inconsistency. In contrast, JoyAI-Echo-1.5-Causal follows the prescribed controls more accurately while better preserving scene layout, subject identity, and visual style across successive turns, demonstrating stronger multi-turn extrapolation and improved robustness to accumulated autoregressive errors.

Table 5: Short-horizon comparison on SANA-WM-Bench using the first 241 frames of each trajectory. Each split contains 80 scenes. Best and second-best results within each split are shown in bold and underlined, respectively.
<table><tr><td></td><td colspan="4">Trajectory and Visual Quality</td></tr><tr><td>Method</td><td>VBench↑</td><td>R↓</td><td>T↓</td><td>CMC↓</td></tr><tr><td>Simple-Trajectory Split</td><td></td><td></td><td></td><td></td></tr><tr><td>Matrix-Game 3 [Wang et al., 2026]</td><td>79.41</td><td>4.474</td><td>0.406</td><td>0.453</td></tr><tr><td>LingBot-World-v2 [Robbyant Team et al., 2026]</td><td>81.13</td><td>3.092</td><td>0.282</td><td>0.310</td></tr><tr><td>HY-WorldPlay [Sun et al., 2025]</td><td>82.05</td><td>2.256</td><td>0.284</td><td>0.298</td></tr><tr><td>DreamX-World [DreamX Team et al., 2026]</td><td>82.42</td><td>1.839</td><td>0.261</td><td>0.273</td></tr><tr><td>SANA-WM [Zhu et al., 2026]</td><td>81.75</td><td>0.489</td><td>0.194</td><td>0.195</td></tr><tr><td>JoyAI-Echo-1.5</td><td>83.91</td><td>0.523</td><td>0.160</td><td>0.162</td></tr><tr><td>Hard-Trajectory Split</td><td></td><td></td><td></td><td></td></tr><tr><td>Matrix-Game 3 [Wang et al., 2026]</td><td>79.37</td><td>12.819</td><td>0.583</td><td>0.717</td></tr><tr><td>LingBot-World-v2 [Robbyant Team et al., 2026]</td><td>81.67</td><td>6.367</td><td>0.357</td><td>0.426</td></tr><tr><td>HY-WorldPlay [Sun et al., 2025]</td><td>81.24</td><td>6.346</td><td>0.387</td><td>0.437</td></tr><tr><td>DreamX-World [DreamX Team et al., 2026]</td><td>82.03</td><td>8.804</td><td>0.467</td><td>0.556</td></tr><tr><td>SANA-WM [Zhu et al., 2026]</td><td>81.79</td><td>1.248</td><td>0.199</td><td>0.206</td></tr><tr><td>JoyAI-Echo-1.5</td><td>83.96</td><td>1.697</td><td>0.256</td><td>0.266</td></tr></table>

We further examine the long-horizon extrapolation capability of the distilled causal model in Fig. 10. Over approximately 60 seconds of continuous causal streaming, JoyAI-Echo-1.5-Causal remains responsive to navigation controls while largely preserving scene identity, subject appearance, visual style, and major spatial structure across successive generated segments. The sustained coherence over this substantially extended horizon further indicates that the causal generator is robust to autoregressive error accumulation well beyond the training horizon.

## 8.2.2 SANA-WM-Bench

Evaluation settings and metrics. We evaluate SANA-WM-Bench under three complementary settings: (1) a short-horizon evaluation on the first 241 frames of each trajectory, (2) the undistilled multi-step model under the official 961-frame (60-second at 16 FPS) long-horizon protocol, and (3) the distilled causal model at 961 frames. The first setting provides a short-horizon reference for trajectory following and visual quality, the second follows the official long-horizon protocol to characterize the 60-second generation capability of the undistilled model, and the third evaluates whether the distilled few-step causal generator can retain strong long-horizon performance after distillation. For trajectory fidelity and visual quality, we follow the benchmark protocol and report rotation error (R, in degrees), relative translation error (T), camera-motion consistency error (CMC), and VBench Overall on a 0–100 scale. For the 961-frame evaluations, we additionally measure viewpoint revisit consistency using PSNR, SSIM, and LPIPS between generated frame pairs that revisit approximately the same camera pose, thereby evaluating long-term scene persistence without reference to ground-truth frames. We further report the temporal quality degradation $\Delta \mathrm { I Q } = q _ { 1 } - q _ { 6 }$ where q<sub>1</sub> and q<sub>6</sub> denote the VBench Imaging Quality scores of the first and last 10-second windows, respectively.

Quantitative Results. As shown in Tab. 5, in the 241-frame short-horizon setting, JoyAI-Echo-1.5 achieves the highest VBench Overall on both trajectory splits, reaching 83.91 on Simple and 83.96 on Hard. It also obtains the lowest translation and CMC errors on the Simple split, while remaining close to SANA-WM in rotation accuracy. On the Hard split, SANA-WM achieves lower trajectory errors, whereas JoyAI-Echo-1.5 maintains the highest overall visual quality.

As shown in Tab. 6, under the official 961-frame long-horizon setting, the undistilled multi-step JoyAI-Echo-1.5 remains highly competitive in both trajectory following and scene persistence. On the Simple split, it achieves a VBench Overall of 81.36, the lowest translation and CMC errors, and the highest revisit PSNR of 15.10,dB. Although trajectory errors become more pronounced over the long rollout, the model maintains strong visual quality and revisit consistency over the full 60-second trajectory. On the Simple split, its rotation error reaches 3.22<sup>◦</sup> at 961 frames, while on the Hard split it increases to 12.05<sup>◦</sup>, indicating that accumulated pose drift remains a key challenge under difficult long-horizon trajectories.

Table 6: Long-horizon comparison on the full SANA-WM-Bench trajectory under the official 961-frame setting (60 seconds at 16 FPS). For JoyAI-Echo-1.5, we report the undistilled multi-step model. Best and second-best results within each split are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="4">Trajectory and Visual Quality</td><td colspan="3">Revisit Consistency</td><td>Temporal Quality</td></tr><tr><td>VBench↑</td><td>R↓</td><td>T↓</td><td>CMC↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>∆IQ↓</td></tr><tr><td>Simple-Trajectory Split</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Infinite-World [Wu et al., 2026]</td><td>79.18</td><td>16.55</td><td>1.98</td><td>2.08</td><td>12.60</td><td>0.284</td><td>0.595</td><td>6.72</td></tr><tr><td>LingBot-World [Robbyant Team et al., 2026]</td><td>81.82</td><td>10.47</td><td>2.01</td><td>2.05</td><td>14.59</td><td>0.366</td><td>0.394</td><td>0.04</td></tr><tr><td>HY-WorldPlay [Sun et al., 2025]</td><td>68.82</td><td>17.89</td><td>2.36</td><td>2.45</td><td>12.83</td><td>0.321</td><td>0.616</td><td>23.59</td></tr><tr><td>Matrix-Game 3 [Wang et al., 2026]</td><td>78.53</td><td>12.96</td><td>1.83</td><td>1.92</td><td>12.29</td><td>0.326</td><td>0.553</td><td>2.41</td></tr><tr><td>SANA-WM [Zhu et al., 2026]</td><td>79.29</td><td>7.59</td><td>1.59</td><td>1.63</td><td>14.16</td><td>0.333</td><td>0.458</td><td>3.79</td></tr><tr><td>JoyAI-Echo-1.5</td><td>81.36</td><td>3.22</td><td>0.918</td><td>0.927</td><td>15.10</td><td>0.335</td><td>0.451</td><td>0.02</td></tr><tr><td>Hard-Trajectory Split</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Infinite-World [Wu et al., 2026]</td><td>79.51</td><td>41.31</td><td>2.49</td><td>2.84</td><td>12.04</td><td>0.248</td><td>0.617</td><td>4.16</td></tr><tr><td>LingBot-World [Robbyant Team et al., 2026]</td><td>81.89</td><td>18.99</td><td>1.65</td><td>1.81</td><td>14.08</td><td>0.332</td><td>0.436</td><td>0.58</td></tr><tr><td>HY-WorldPlay [Sun et al., 2025]</td><td>70.46</td><td>35.46</td><td>2.34</td><td>2.64</td><td>13.72</td><td>0.328</td><td>0.654</td><td>25.88</td></tr><tr><td>Matrix-Game 3 [Wang et al., 2026] SANA-WM [Zhu et al., 2026]</td><td>78.79</td><td>18.79</td><td>1.67</td><td>1.82</td><td>12.17</td><td>0.317</td><td>0.556</td><td>0.32</td></tr><tr><td></td><td>79.60</td><td>10.02</td><td>1.66</td><td>1.72</td><td>14.10</td><td>0.327</td><td>0.469</td><td>3.09</td></tr><tr><td>JoyAI-Echo-1.5</td><td>81.72</td><td>12.05</td><td>1.265</td><td>1.358</td><td>14.58</td><td>0.333</td><td>0.482</td><td>0.93</td></tr></table>

Table 7: Long-horizon comparison of distilled causal world models on SANA-WM-Bench under the 961-frame setting (60 seconds at 16 FPS). JoyAI-Echo-1.5-Causal denotes ours 4-step causal model. Best and second-best results within each split are shown in bold and underlined, respectively.
<table><tr><td></td><td colspan="4">Trajectory and Visual Quality</td><td colspan="3">Revisit Consistency</td><td>Temporal Quality</td></tr><tr><td>Method</td><td>VBench↑</td><td>R↓</td><td>T↓</td><td>CMC↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>∆IQ↓</td></tr><tr><td>Simple-Trajectory Split</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Evoke Yin et al. [2026]</td><td>80.11</td><td>9.859</td><td>2.159</td><td>2.190</td><td>13.50</td><td>0.2774</td><td>0.5010</td><td>-0.49</td></tr><tr><td>LingBot-World-v2 [Robbyant Team et al., 2026]</td><td>79.20</td><td>22.516</td><td>2.391</td><td>2.541</td><td>11.90</td><td>0.1715</td><td>0.5906</td><td>2.59</td></tr><tr><td>SANA-WM + Causal Refiner [Zhu et al., 2026]</td><td>79.55</td><td>17.875</td><td>2.222</td><td>2.338</td><td>14.41</td><td>0.2660</td><td>0.5748</td><td>-0.75</td></tr><tr><td>SANA-WM w/o Refiner [Zhu et al., 2026]</td><td>78.21</td><td>15.359</td><td>2.288</td><td>2.393</td><td>9.19</td><td>0.1661</td><td>0.6321</td><td>-0.12</td></tr><tr><td>JoyAI-Echo-1.5-Causal</td><td>80.13</td><td>5.009</td><td>1.592</td><td>1.611</td><td>14.41</td><td>0.2998</td><td>0.4802</td><td>2.09</td></tr><tr><td>Hard-Trajectory Split</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Evoke Yin et al. [2026]</td><td>80.75</td><td>20.097</td><td>1.924</td><td>2.060</td><td>13.06</td><td>0.2592</td><td>0.5292</td><td>0.30</td></tr><tr><td>LingBot-World-v2 [Robbyant Team et al., 2026]</td><td>78.74</td><td>34.416</td><td>2.125</td><td>2.413</td><td>12.34</td><td>0.2183</td><td>0.5732</td><td>3.84</td></tr><tr><td>SANA-WM + Causal Refiner [Zhu et al., 2026] SANA-WM w/o Refiner [Zhu et al., 2026]</td><td>79.60</td><td>23.794</td><td>1.932</td><td>2.124</td><td>13.89</td><td>0.2605</td><td>0.5696</td><td>-0.24</td></tr><tr><td></td><td>78.27</td><td>18.717</td><td>2.006</td><td>2.142</td><td>9.26</td><td>0.1729</td><td>0.6077</td><td>2.06</td></tr><tr><td>JoyAI-Echo-1.5-Causal</td><td>81.06</td><td>14.221</td><td>1.735</td><td>1.829</td><td>14.04</td><td>0.2910</td><td>0.4893</td><td>1.30</td></tr></table>

(a) Simple-Trajectory Split  
![](images/685c8c8f92d66326550f40786c06bb29037c67f9fcd3c28e25adc39476157468.jpg)

(b) Hard-Trajectory Split  
![](images/c844326a52f57b88ac846c454205c12cc61f9d770abc34dc0aef08d580e2686e.jpg)  
Figure 11: Imaging quality comparison on the Simple- and Hard-Trajectory splits. We report the first-window, mean, minimum, and final-window IQ. Our method achieves consistently higher mean and minimum IQ than Evoke and SANA-WM with Causal Refiner. Although the competing methods exhibit smaller IQ drops, their overall IQ remains lower, indicating that a small drop can also result from consistently low imaging quality rather than stronger long-horizon quality preservation.

Table 8: Human evaluation based on Good–Similar–Bad (GSB) pairwise comparisons between JoyAI-Echo-1.5 (world model variant) and Happy Oyster for world modeling. Numbers denote the percentage of user preferences.
<table><tr><td>Aspect</td><td>JoyAI-Echo-1.5</td><td>Similar</td><td>Happy Oyster</td></tr><tr><td>Semantic following</td><td>13.63%</td><td>56.94%</td><td>29.44%</td></tr><tr><td>Initial-world consistency</td><td>39.50%</td><td>45.06%</td><td>15.44%</td></tr><tr><td>Motion</td><td>18.94%</td><td>56.51%</td><td>24.56%</td></tr><tr><td>Spatiotemporal consistency</td><td>57.69%</td><td>31.88%</td><td>10.44%</td></tr><tr><td>Visual aesthetics</td><td>52.88%</td><td>23.32%</td><td>23.81%</td></tr><tr><td>Overall</td><td>63.13%</td><td>9.82%</td><td>27.06%</td></tr></table>

Tab. 7 further compares the distilled causal model with recent state-of-the-art causal distillation methods under the same 961-frame horizon. JoyAI-Echo-1.5-Causal achieves the highest VBench scores on both Simple and Hard splits, with 80.13 and 81.06, respectively, together with the best trajectory accuracy and strongest revisit consistency among the evaluated causal models. As further illustrated in Fig. 11, it achieves substantially higher first-window, mean, and minimum IQ than Evoke and SANA-WM with Causal Refiner, while maintaining competitive final-window IQ. Although these baselines exhibit smaller ∆IQ, their lower initial and overall IQ suggest that a small drop may also result from consistently low imaging quality rather than stronger preservation of a high-quality initial state. Compared with the undistilled 961-frame model, causal few-step distillation introduces a moderate degradation in trajectory accuracy and revisit consistency while largely preserving long-horizon visual quality, demonstrating robust 60-second autoregressive rollout capability.

Qualitative Results. Fig. 12 compares approximately 60-second causal rollouts generated from the same initial observation under matched camera trajectories. As the rollout horizon increases, competing methods exhibit different forms of cumulative autoregressive drift. LingBot-World-v2 progressively alters the scene composition and viewpoint, while SANA-WM without the Causal Refiner can undergo substantial appearance and style shifts. The Causal Refiner improves local temporal stability but does not fully prevent long-term structural and appearance drift. Evoke generally maintains coherent short-term appearance, but deviations in viewpoint evolution and scene configuration become increasingly apparent over longer horizons.

In contrast, JoyAI-Echo-1.5-Causal exhibits the strongest long-horizon consistency across these examples. In both natural and structured indoor environments, it better preserves visual style, scene and subject identity, and major geometric structures while evolving the viewpoint according to the prescribed trajectory. These properties remain comparatively stable at later timestamps instead of progressively degrading with the accumulated autoregressive history. The qualitative observations are consistent with our trajectory-following, temporal-degradation, and revisit-consistency metrics, supporting the improved robustness of our causal model to long-horizon error accumulation and its stronger action-conditioned world persistence.

## 8.2.3 User Study

We additionally conduct a subjective evaluation on a self-built World-Model Benchmark (WMB) containing 200 cases. The benchmark covers game, humanoid-robot, natural outdoor, household indoor, and driving scenes, with 132 third-person and 68 first-person cases. We compare JoyAI-Echo 1.5 with LingBot-World-v2 [Robbyant Team et al., 2026] and HappyOyster through randomized and anonymized side-by-side comparisons. Annotators evaluate semantic following, initial-world consistency, spatiotemporal consistency, motion quality, and visual aesthetics, together with an overall preference.

Against LingBot-World-v2, JoyAI-Echo-1.5 receives higher explicit preference across all five dimensions, with the largest margins in semantic following (34.0% vs. 15.1%) and spatiotemporal consistency (32.6% vs. 14.7%). The overall preference is 46.5% for JoyAI-Echo-1.5 versus 21.6% for LingBot-World-v2.

Against HappyOyster, JoyAI-Echo-1.5 shows substantial advantages in spatiotemporal consistency (57.7% vs. 10.4%), visual aesthetics (52.9% vs. 23.8%), and initial-world consistency (39.5% vs.

![](images/3bb88aef3f645f8fc3e0c26be1faefae676a4796029210b595cc6cf1845ca85f.jpg)  
Figure 12: Qualitative comparison of 961-frame (60-second at 16 FPS) causal rollouts on representative SANA-WM-Bench trajectories. We compare JoyAI-Echo-1.5-Causal with LingBot-World-v2 (LBW-V2), SANA-WM with and without Causal Refiner, and Evoke. Frames are sampled every 12 seconds from the same initial observation and camera-trajectory condition. Across the extended rollout, JoyAI-Echo-1.5-Causal exhibits less accumulated autoregressive drift, more consistent trajectory following, and stronger preservation of scene identity, visual style, and spatial structure.

15.4%). HappyOyster is preferred more often in semantic following and motion quality, while JoyAI-Echo-1.5 achieves a substantially higher overall preference (63.1% vs. 27.1%). These results complement the public benchmarks, showing that the gains in visual quality and spatiotemporal consistency are also reflected in human preference.

Table 9: Human evaluation based on Good–Similar–Bad (GSB) pairwise comparisons between JoyAI-Echo-1.5 (world model variant) and LingBot-World-v2 for world modeling. Numbers denote the percentage of user preferences.
<table><tr><td>Aspect</td><td>JoyAI-Echo-1.5</td><td>Similar</td><td>LingBot-World-v2</td></tr><tr><td>Semantic following</td><td>34.00%</td><td>50.86%</td><td>15.14%</td></tr><tr><td>Initial-world consistency</td><td>14.64%</td><td>77.65%</td><td>7.71%</td></tr><tr><td>Motion</td><td>25.29%</td><td>54.71%</td><td>20.00%</td></tr><tr><td>Spatiotemporal consistency</td><td>32.64%</td><td>52.65%</td><td>14.71%</td></tr><tr><td>Visual aesthetics</td><td>32.50%</td><td>47.57%</td><td>19.93%</td></tr><tr><td>Overall</td><td>46.50%</td><td>31.92%</td><td>21.57%</td></tr></table>

## 9 Conclusion

We presented a framework for extending audio-visual generation beyond isolated clips toward persistent stories and interactive worlds. The long-video system uses composable cross-shot audiovisual memory to preserve character appearance, speaker identity, and narrative continuity under flexible text, image, and memory conditioning. The world-model system translates heterogeneous navigation inputs into calibrated metric 6-DoF trajectories, enabling precise and controller-agnostic interaction across diverse environments and viewpoints. Together with few-step distillation, causal audio-visual teacher forcing, and short- and long-horizon Self-Gradient Forcing, these designs allow the models to generate synchronized audio and video efficiently while remaining stable under selfgenerated histories. Experiments demonstrate improved cross-shot consistency, instruction following, speech fidelity, trajectory control, and long-horizon world persistence, with leading results on both long-form generation evaluations and public world-model benchmarks. These findings suggest that memory, geometric control, and rollout-aware training are complementary foundations for video systems that can remember past events, respond to user intent, and continue evolving over time. Accumulated rotational drift on difficult long-horizon trajectories remains a key limitation, motivating future work on stronger geometric consistency, persistent world-state representations, and more reliable open-ended interaction.

## References

Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5962–5971, 2022.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, et al. 3d-speaker-toolkit: An open source toolkit for multi-modal speaker verification and diarization. 2025.

Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Mmaudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 28901–28911, 2025.

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flexattention: A programming model for generating fused attention variants. In Proceedings of Machine Learning and Systems, volume 7, 2025.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, et al. DreamX-World 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026.

Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Scandinavian Conference on Image Analysis (SCIA), pages 363–370. Springer, 2003.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems, volume 38, pages 75460–75482, 2025.

Google DeepMind. Genie 3: A new frontier for world models, 2025. URL https://deepmind. google/discover/blog/genie-3-a-new-frontier-for-world-models/.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. LTX-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. ViPE: Video pose engine for 3D geometric perception. arXiv preprint arXiv:2508.10934, 2025.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

HyCreator Team. HyCreator: An agent harness for long video generation, 2026. URL https: //hycreator.tencent.com/. Multimodal Reinforcement Learning Team, Tencent HY.

Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang, Xue Zheng, Rui Sun, et al. ABot-World-0: Infinite interactive world rollout on a single desktop GPU. arXiv preprint arXiv:2607.19191, 2026.

Haoran Li, Fredreic Li, Shichen Ma, Jie Huang, Yijun Liu, Jiaqi Shi, and Yanwen Ma. Joyai-echo: Pushing the frontier of long audio-visual generation. 2026.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Kai Liu, Yanhao Zheng, Kai Wang, Shengqiong Wu, Rongjunchen Zhang, Jiebo Luo, Dimitrios Hatzinakos, Ziwei Liu, Hao Fei, and Tat-Seng Chua. Javisdit++: Unified modeling and optimization for joint audio-video generation. In The Fourteenth International Conference on Learning Representations, 2026.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, volume 35, pages 5775–5787, 2022.

Xin Lu, Zihao Fan, Mingchen Zhong, Jie Huang, Xueyang Fu, and Zheng-Jun Zha. OmniVR: Joint video-audio conditional generation for restoring degraded historical films. arXiv preprint arXiv:2608.04224, 2026.

Yiting Lu, Xin Li, Yajing Pei, Kun Yuan, Qizhi Xie, Yunpeng Qu, Ming Sun, Chao Zhou, and Zhibo Chen. Kvq: Kwai video quality assessment for short-form videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, Pengfei Wan, and Tianfan Xue. Shotstream: Streaming multi-shot video generation for interactive storytelling. arXiv preprint arXiv:2603.25746, 2026.

Luxury, Jie Huang, Zihao Fan, Xiaoxiao Ma, Yuming Li, Jun-hao Zhuang, Zeyue Xue, Siming Fu, Haoran Li, Mingchen Zhong, Guohui Zhang, Shichen Ma, Yijun Liu, Jiaqi Shi, Yanwen Ma, Yaofeng Su, Haoyu Wang, Yaowei Li, Songchun Zhang, Weiyang Jin, Yuxuan Bian, Shiyi Zhang, Haojun Xu, Shuai Lu, Xin Han, Wei Tang, Haoyang Huang, and Nan Duan. Ultra Flash: Scaling real-time streaming video generation to high resolutions. arXiv preprint arXiv:2606.09150, 2026.

NVIDIA. Cosmos world foundation model platform for physical AI. arXivpreprint arXiv:2501.03575, 2025.

NVIDIA. Cosmos 3: Omnimodal world models for physical AI. Technical report, NVIDIA, 2026.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL https://arxiv.org/ abs/2212.04356.

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015.

Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, et al. Lyra 2.0: Explorable generative 3D worlds. arXiv preprint arXiv:2604.13036, 2026.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 32211–32252. PMLR, 2023.

Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Ying Li, Zezhong Qian, Haoyang Huang, and Nan Duan. OmniForcing: Unleashing real-time joint audio-visual generation. arXiv preprint arXiv:2603.11647, 2026.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Zeyue Tian, Lei Ke, Zhaoyang Liu, Ruibin Yuan, Liumeng Xue, Yujiu Yang, Weijia Chen, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo. Audiox-turbo: A unified framework for efficient anythingto-audio generation. arXiv preprint arXiv:2606.12555, 2026.

Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 1905–1914, 2021.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270–280, 1989.

Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. HunyuanVideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, and Ming-Ming Cheng. Infinite-World: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393, 2026.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024b.

Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alayaevoke: From linear-scaling supervision to endless world. arXiv preprint arXiv:2608.13546, 2026.

Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. WBench: A comprehensive multi-turn benchmark for interactive video world model evaluation. arXiv preprint arXiv:2605.25874, 2026.

Cheng Zhang, Boying Li, Meng Wei, Yan-Pei Cao, Camilo Cruz Gambardella, Dinh Phung, and Jianfei Cai. Unified camera positional encoding for controlled video generation. arXiv preprint arXiv:2512.07237, 2025a.

Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025b.

Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictorcorrector framework for fast sampling of diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 49842–49869, 2023.

Kaiyang Zhou and Tao Xiang. Torchreid: A library for deep learning person re-identification in pytorch. arXiv preprint arXiv:1910.10093, 2019.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. SANA-WM: Efficient minute-scale world modeling with hybrid linear diffusion transformer. arXiv preprint arXiv:2605.15178, 2026.

Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, Haoran Li, Haoyang Huang, Zeyue Xue, and Nan Duan. Self Gradient Forcing: Native long video extrapolation. arXiv preprint arXiv:2607.20368, 2026.