# UniSwap: Streaming Audio-Visual Identity Swapping for Talking Videos

Yuxuan Zhang <sup>1,2</sup> Haozhong Xiong <sup>2</sup> Jiayi Song <sup>2</sup> Jinpeng Yu <sup>2</sup> Yang Shi <sup>2</sup> Jiaming Liu <sup>2</sup> Ruihua Huang <sup>2</sup> Liwei Wang <sup>1\*</sup> <sup>1</sup> The Chinese University of Hong Kong <sup>2</sup> Qwen Applications Business Group of Alibaba yxzhang@cse.cuhk.edu.hk, jmliu1217@gmail.com https://uniswap-av.github.io/

![](images/7156f035cd4bf43b5f9a21628ad288b8a0ec1fa2e9ac664d915497b003220aa7.jpg)  
Figure 1. UniSwap performs streaming joint audio-visual identity replacement. Given a source talking video and a reference identity (an image and voice clip), UniSwap jointly transfers appearance and vocal timbre while preserving the source motion, background, and linguistic content. Blockwise autoregressive generation with 3-step sampling runs at 13.6 FPS on one NVIDIA H100 and supports stable hour-scale long-form generation.

## Abstract

Talking-video character replacement requires coordinated transfer of appearance and voice while preserving the source motion, scene, linguistic content, and audiovideo timing. Existing methods use separately optimized models for the two modalities, making audio-visual consistency difficult to enforce. We present UniSwap, the first frameworkfor streamingjoint audio-visual identity replacement in talking videos. Given a source video, a refer-

ence image, and a reference voice clip, UniSwap transfers the reference appearance and vocal timbre within a single audio-visual diffusion transformer while preserving the source content and dynamics. To address the scarcity of aligned cross-identity training pairs, we introduce a swap-and-reconstruct pipeline that removes visual and vocal identity from real clips and uses the original clips as reconstruction targets. Starting from a bidirectional backbone, we progressively adapt the model through In-context Pretraining for joint replacement, Conditional Streaming Adaptationfor block-causal KV-cached generation, and Ef-

ficient Self-forcing DMD for mitigating exposure bias and reducing sampling from 30 to 3 denoising steps per block. Efficient Multi-LoRA Switching enables the three DMD roles to share a single frozen backbone. Feature-RoPE Decomposition keeps cached positions within the training range, supporting stable long-form inference. Experiments demonstrate strong audio-visual synchronization, competitive identity preservation, efficient streaming, and stable long-form generation.

## 1. Introduction

Recent generative models have substantially advanced video synthesis [11, 31, 40] as well as speech generation and neural audio modeling [6, 7]. These advances enable applications such as film post-production, content localization, and personalized media. A central capability in these applications is audio-visual identity replacement: changing both the appearance and voice of a person in a talking video while preserving the source motion, background, and linguistic content. Performing this transformation jointly within a single streaming model would enable low-latency, interactive workflows. However, existing methods are not designed to jointly replace visual and vocal identity while supporting streaming generation.

Existing methods address only one side of this task. Video character-replacement methods [4, 14, 18, 38] transfer appearance from a reference image but do not generate the corresponding voice. Conversely, voice-conversion systems [7, 17, 22] modify speaker timbre without visual context. Combining these components in a cascade can replace both identities, but the two modules remain independently optimized: no joint objective enforces consistency between the converted speech and lip motion, and errors from one modality cannot be corrected using evidence from the other. In addition, most diffusion-based video replacement methods require the complete clip before generation, preventing blockwise streaming and limiting their use in low-latency interactive applications.

We therefore formulate character replacement in talking videos as joint audio-video generation. A unified model can condition on both target appearance and vocal timbre while using cross-modal interactions to coordinate generated speech and lip motion. Realizing this formulation presents two challenges. First, training requires source– target pairs that differ in visual and vocal identity while remaining aligned in motion, scene content, speech, and timing; such pairs are impractical to collect at scale. Second, high-quality audio-video diffusion backbones operate bidirectionally over complete sequences and require many denoising steps, whereas low-latency streaming requires causal, blockwise generation with bounded per-block computation.

To address these challenges, we introduce UniSwap, the first framework for streaming joint audio-visual identity replacement in talking videos (Fig. 1). Given a source talking video, a reference image, and a reference voice clip, UniSwap generates synchronized video and audio in which the reference appearance and vocal timbre are transferred while the source motion, background, and linguistic content are preserved. Unlike cascaded systems, UniSwap performs both replacements within a single audio-video diffusion transformer and generates the output autoregressively in blocks.

To construct aligned supervision, our swap-andreconstruct pipeline retains each real clip as the target while synthesizing an identity-altered source: an identity-reduced motion proxy replaces the person, and voice conversion changes the speaker timbre without altering the speech content or timing. The original appearance and voice provide the references. This process converts ordinary talking videos into aligned training pairs without requiring different people to perform the same motion and speech.

UniSwap builds on LTX-2.3 [11], an audio-video diffusion transformer with native cross-modal attention, and progressively adapts it through three training stages. Stage 1, In-context Pretraining, concatenates the source, reference, and noisy target latents into a unified sequence with aligned source–target coordinates, allowing full-sequence attention to learn joint visual and vocal identity replacement. Stage 2, Conditional Streaming Adaptation, applies a Decoupled Streaming Conditioning Mask that restricts each token region to its inference-time receptive field, converting the bidirectional model into a block-causal generator and enabling KV-cached autoregressive inference. Stage 3, Efficient Self-forcing DMD, rolls out the student’s own predictions to expose it to inference-like histories, while DMD [42] reduces sampling from 30 to 3 denoising steps per block. Efficient Multi-LoRA Switching lets the teacher, generator, and critic share one frozen backbone instead of maintaining three full model copies. During inference, Feature-RoPE Decomposition separates cached features from rotary coordinates and reapplies bounded positions, preserving cross-modal temporal alignment and supporting stable long-form generation.

Our main contributions are fourfold:

• We introduce UniSwap, the first framework for streaming joint audio-visual identity replacement in talking videos, transferring appearance and vocal timbre within a single diffusion transformer.

• We propose a swap-and-reconstruct pipeline that converts ordinary talking videos into temporally aligned supervision for joint visual and vocal identity replacement.

• We progressively convert a bidirectional backbone into a causal three-step generator through In-context Pretraining, Conditional Streaming Adaptation with a Decoupled Streaming Conditioning Mask, and Efficient Self-forcing DMD implemented through Efficient Multi-LoRA Switching.

• We propose Feature-RoPE Decomposition, which separates cached features from rotary coordinates and bounds positional indices while preserving cross-modal temporal alignment during long-form inference.

## 2. Related Work

## 2.1. Video Character Replacement

Video character replacement substitutes a person’s visual identity while preserving motion, expression, and background. Frame-based face-swapping approaches [3, 10, 19] can suffer from temporal inconsistency. Recent methods instead use video diffusion models. MoCha [38] casts replacement as in-context generation and uses conditionaware RoPE [29] to combine a source video, a mask, and reference images without per-frame structural guidance. Wan-Animate [4] and VACE [18] inject explicit structural signals into DiT-based [23] generation, while HunyuanCustom [14] conditions customized generation on masked target regions and reference images. More generally, in-context diffusion methods place conditioning examples and noisy targets in a shared transformer context. IC-LoRA [15] studies this formulation for image diffusion transformers, and Video Diffusion Transformers are In-Context Learners [9] extends it to video generation by concatenating related videos along spatial or temporal dimensions. FullDiT2 [12] further improves the efficiency of in-context video conditioning through dynamic token selection and selective context caching. These methods address visual identity or visual control only, so dubbing still requires a separate audio system.

## 2.2. Voice Conversion

Voice conversion changes a speaker’s timbre while preserving linguistic content and prosody. Traditional approaches rely on parallel data or explicit feature disentanglement, whereas recent zero-shot methods [7, 17, 22] use large-scale pretraining for speaker-independent conversion. Seed-VC [22] uses self-supervised speech representations for content extraction. REF-VC [17] combines ASR bottleneck and self-supervised features in a diffusion transformer, with random erasing and shortcut models for robust, fast inference. CosyVoice [7] uses supervised semantic tokens for zero-shot text-to-speech synthesis. These systems operate without visual context and therefore do not jointly optimize converted speech and lip motion.

## 2.3. Audio-Visual Generation

Recent generative models have begun to model audio and video jointly. An early representative, MM-

Diffusion [27], couples modality-specific denoisers through cross-modal attention to synthesize aligned audio-video pairs. More recent diffusion transformers strengthen joint modeling: JavisDiT [21] introduces hierarchical spatiotemporal priors for audio-visual synchronization, while LTX-2 [11] uses separate, interacting modality streams connected by cross-modal attention, enabling synchronized audio-video generation from text. UniSwap builds on this capability for character replacement, where the model must transfer visual and vocal identity while retaining source motion and speech content.

## 2.4. Distillation for Real-Time Generation

Diffusion models typically require many denoising steps, limiting low-latency applications. Distribution Matching Distillation (DMD) [41, 42] trains a student to match a multi-step teacher’s output distribution using a critic, enabling one- or few-step generation. Consistency models [28] instead enforce consistency along the probability-flow ODE trajectory. For continuous sequence generation, Diffusion Forcing [2] combines causal prediction with diffusion over independently noised sequence tokens. For autoregressive video generation, Self-Forcing [16] reduces exposure bias by feeding the model’s own outputs back during training, while CausVid [43] applies DMD to causal streaming video. Rolling Forcing [20] combines rolling-window denoising, an initialframe attention sink, and few-step distillation for longhorizon video streams. OmniForcing [30] further distills a bidirectional dual-stream audio-visual diffusion model into a streaming autoregressive generator using block-causal alignment, joint self-forcing distillation, and a rolling KV cache. Unlike these general generation systems, UniSwap combines self-forcing rollout and DMD for source-driven joint appearance-and-voice streaming replacement.

## 3. Method

Given a source video $V _ { s } ~ \in ~ \mathbb { R } ^ { F \times 3 \times H \times W }$ with audio waveform $A _ { s } ,$ a reference image $I _ { r }$ , and a reference voice clip $A _ { r }$ , UniSwap generates a target video $V _ { t }$ and audio $A _ { t } .$ The target should match the reference appearance and vocal timbre while preserving the source motion, lip movements, background, and linguistic content. Figure 2 illustrates the three-stage training pipeline.

Backbone and Latent Representation. We build on the frozen LTX-2.3 audio-video diffusion transformer [11], which processes video and audio in separate streams connected by bidirectional cross-modal attention. A causal video VAE compresses the video temporally by a factor of eight, while the audio representation is sampled at 25 latent tokens per second. We denote the video and audio latent channel dimensions by $C _ { v }$ and $C _ { a }$ , respectively; their token positions are assigned on a shared physical-time axis to maintain cross-modal alignment.

![](images/0aea280c84fce206409ca7a5ca1a9c5886b3655dc73ad0715657c8bd9082d67b.jpg)  
Figure 2. Overview of UniSwap. The three-stage pipeline comprises (a) In-Context Pretraining for joint audio-video replacement, (b) Conditional Streaming Adaptation with block-causal masking and KV-cached inference, and (c) Efficient Self-Forcing DMD, which reduces denoising to 3 steps per block. Feature-RoPE Decomposition bounds cached positions while preserving cross-modal physical-time alignment.

## 3.1. Swap-and-Reconstruct Paired Data Synthesis

Paired data for joint audio-video identity replacement are not naturally available at scale. We therefore formulate the task as self-reconstruction: a real talking video provides the target $( V _ { t } , A _ { t } ) .$ , while an identity-altered version provides the source $( V _ { s } , A _ { s } )$ . The original visual and vocal identities are supplied separately through a reference image $I _ { r }$ and reference audio $A _ { r }$ (Fig. 3).

Visual Stream. We estimate whole-body 2D poses [37] and use the pose keypoints to prompt video segmentation [25]. After dilating and augmenting the person mask, we remove the person to obtain a background plate and composite the rendered pose sequence onto it, yielding $V _ { s }$ . This retains the source motion, timing, and background while suppressing appearance cues; a portrait frame from $V _ { t }$ serves as $I _ { r }$

Audio Stream. We convert $A _ { t }$ toward a randomly sampled speaker using an off-the-shelf voice conversion model [22], producing $A _ { s }$ with altered timbre but preserved speech content and prosody. A random segment spanning 30% of $A _ { t }$ serves as the reference audio $A _ { r }$

All three training stages use tuples generated by this pipeline.

## 3.2. Stage 1: In-context Pretraining

Following the in-context conditioning paradigm for diffusion transformers [9, 15], the first stage adapts the model to joint audio-video character replacement by placing the conditioning and target latents in a unified attention context. As shown in Stage 1 of Fig. 2, all inputs are concatenated into a unified token sequence.

Input Formulation. For each training sample, we encode: (1) the source video latent $z _ { s } ^ { v } \in \mathbb { R } ^ { \breve { v } _ { s } \times C _ { v } }$ and source audio latent $z _ { s } ^ { a } \in \mathbb { R } ^ { L _ { s } ^ { a } \times C _ { a } }$ representing the driving motion and content; (2) a reference image latent $z _ { r } ^ { v } \in \mathbb { R } ^ { L _ { r } ^ { v } \times C _ { v } }$ and reference audio latent $z _ { r } ^ { a } \in \mathbb { R } ^ { \breve { L } _ { r } ^ { a } \times C _ { a } }$ providing the target identity; and (3) the target video latent $ { \boldsymbol { z } } _ { t } ^ { v }$ and target audio latent $z _ { t } ^ { a }$ to be denoised.

These are concatenated along the sequence dimension for each modality stream:

$$
{ \mathrm { V i d e o : } } \quad x ^ { v } = [ z _ { r } ^ { v } ; z _ { s } ^ { v } ; z _ { t } ^ { v } ] , \quad { \mathrm { A u d i o : } } \quad x ^ { a } = [ z _ { r } ^ { a } ; z _ { s } ^ { a } ; z _ { t } ^ { a } ]\tag{1}
$$

During training, noise is added only to the target portions $ { \boldsymbol { z } } _ { t } ^ { v }$ and $z _ { t } ^ { a }$ ; the source and reference tokens serve as clean conditioning context with $\sigma = 0$

Condition Positional Encoding Offset. A naive application of 3D RoPE [29] would assign sequential temporal indices to all tokens, coupling the model to fixed input lengths. Instead, we employ condition positional encoding offsets: the source and target share identical temporal positions in both modalities because they represent the same time span, while the reference image and reference audio use fixed offsets $\Delta _ { r } ^ { v }$ and $\Delta _ { r } ^ { a }$ , respectively. This design enables variable-length inputs while keeping the positional semantics of all conditions consistent throughout adaptation and inference.

Training. The model is trained end-to-end with in-context attention across the entire concatenated sequence. For a clean latent $z _ { \mathrm { 0 } }$ , Gaussian noise ϵ, and noise level $\sigma ,$ we define $z _ { \sigma } = ( 1 - \sigma ) z _ { 0 } + \sigma \epsilon$ and the conditional flow-matching loss as

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { z _ { 0 } , \sigma , \epsilon } \left[ \| v _ { \theta } ( z _ { \sigma } , \sigma ) - ( \epsilon - z _ { 0 } ) \| _ { 2 } ^ { 2 } \right] .\tag{2}
$$

Both video and audio target streams are jointly denoised:

$$
\mathcal { L } _ { \mathrm { S t a g e 1 } } = \mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { F M } } ^ { \mathrm { a u d i o } }\tag{3}
$$

## 3.3. Stage 2: Conditional Streaming Adaptation

Stage 2 converts the bidirectional model into a blockwise autoregressive generator. We partition the target into blocks of $K { = } 3$ video latent frames, except for an initial 4-frame block required by the temporal indexing of the causal video VAE. Audio blocks are defined over the corresponding time intervals. Let $B _ { i }$ and $S _ { i }$ denote the i-th target and source blocks, respectively. The model denoises $B _ { i }$ conditioned on the reference, $S _ { i } ,$ , and the preceding target blocks $B _ { < i }$

## 3.3.1. Decoupled Streaming Conditioning Mask

The Decoupled Streaming Conditioning Mask aligns the training-time receptive field with KV-cached inference (Stage 2 of Fig. 2). Reference tokens and each source block are encoded independently. Clean target blocks follow block-causal attention, whereas the noisy target block $B _ { i }$ attends to the reference, its aligned source block $S _ { i } ,$ the clean history $B _ { < i }$ , and itself. The same role-wise mask is applied to audio and video self-attention and to cross-modal attention, with modality-specific block sizes chosen to preserve physical-time alignment. This construction prevents future-target leakage while reproducing the context available during autoregressive inference.

During training, the clean stream is populated with ground-truth latents (teacher forcing [34]), so the model learns to denoise each block $B _ { i }$ under ground-truth history. The loss is computed only on the noisy target tokens:

$$
\mathcal { L } _ { \mathrm { S t a g e 2 } } = \mathbb { E } _ { i , \sigma } \left[ \mathcal { L } _ { \mathrm { F M } } ( B _ { i } \mid \mathrm { r e f } , S _ { i } , B _ { 0 } ^ { \mathrm { c l e a n } } , \dots , B _ { i - 1 } ^ { \mathrm { c l e a n } } ) \right]\tag{4}
$$

KV-Cached Streaming Inference. At inference, the reference is cached once, each source block is cached only while its target block is denoised, and completed target blocks are committed as clean history. This yields a per-block cost independent of the generated duration. The complete cache update procedure is provided in Alg. 1 of the supplementary material.

## 3.4. Stage 3: Efficient Self-forcing DMD

Stage 2 is trained with clean ground-truth histories and requires ∼30 denoising steps per block. Stage 3 instead distills a 3-step student under self-generated histories, reducing both exposure bias and sampling cost.

## 3.4.1. Self-forcing Style Training

Self-forcing Rollout. The student autoregressively generates the full audio-video sequence and conditions each block on its preceding outputs. Each block is denoised at three noise levels, [0.999, 0.757, 0.522].

Efficient Multi-LoRA Switching. Self-forcing DMD ordinarily requires separate teacher, generator, and critic models. Efficient Multi-LoRA Switching represents these roles with three adapters on a shared frozen backbone (Stage 3 of Fig. 2): frozen Stage-1 LoRA-1 serves as the teacher, Stage-2-initialized LoRA-2 as the generator, and randomly initialized LoRA-3 as the trainable critic. Activating one adapter at a time avoids maintaining three backbone copies, reducing peak GPU memory from more than 80 GB (out of memory on an 80-GB GPU) to 65.34 GB under the same configuration. Only LoRA-2 is retained for inference.

Distribution Matching Loss. For the generated sequence $\hat { z } ,$ DMD matches the critic score to the classifier-freeguidance (CFG) enhanced teacher score at a noise level $\sigma \sim \mathcal { U } [ 0 . 0 2 , 0 . 9 8 ]$

$$
\begin{array} { r l } & { \nabla _ { \hat { z } } = \underbrace { D _ { \phi } ( \hat { z } _ { \sigma } , \sigma ) } _ { \mathrm { c r i t i c ~ o n ~ f a k e } } } \\ & { \qquad - \underbrace { \Bigl [ T _ { \psi } ^ { + } ( \hat { z } _ { \sigma } , \sigma ) + \gamma \left( T _ { \psi } ^ { + } ( \hat { z } _ { \sigma } , \sigma ) - T _ { \psi } ^ { - } ( \hat { z } _ { \sigma } , \sigma ) \right) \Bigr ] } _ { \mathrm { C F G - e n h a n c e d ~ t e a c h e r } } } \end{array}\tag{5}
$$

where $T _ { \psi } ^ { + / - }$ are the conditional and unconditional predictions of the bidirectional teacher, $D _ { \phi }$ is the critic pre-

![](images/f3987cfed445a267b20abf9178d2d1cee06ab4e6cd3989feff22609bc93705da.jpg)  
Figure 3. Swap-and-reconstruct paired data synthesis. Every real talking video serves as its own reconstruction target. ① The real clip provides the target video $V _ { t }$ and audio $A _ { t }$ . ② The visual identity is swapped out by replacing the person with a pose proxy composited onto the masked background plate, the vocal timbre is randomized towards a sampled speaker, and the reference identity is formed by a portrait frame $I _ { r }$ and a random 30% audio crop $A _ { r } . \textcircled { 3 }$ The resulting identity-swapped source and reference identity condition the model, which is trained to reconstruct the original clip.

diction, and $\gamma _ { \mathrm { v i d e o } } = 3 . 0$ and $\gamma _ { \mathrm { a u d i o } } = 5 . 0$ . This signal updates the generator toward the teacher distribution, while the critic is trained on noised student samples.

## 3.4.2. Feature-RoPE Decomposition Inference

Absolute RoPE coordinates grow with an autoregressive stream and eventually exceed the range observed during fixed-length training. Feature-RoPE Decomposition instead stores unrotated keys and reapplies RoPE according to each block’s current cache slot, without recomputing its features. The cache contains reference tokens, an initial sink block, and a rolling history of recent clean blocks.

Adaptive Sink Block. We adapt the attention-sink mechanism [36] by retaining the first generated block $B _ { 0 }$ at a fixed local position. Because $B _ { 0 }$ establishes the initial appearance and voice, it provides a persistent identity anchor as the rolling history advances.

Reference Re-anchoring. Before denoising each block, the stored reference keys are re-rotated relative to the current generation slot, together with the modality-specific offsets $\Delta _ { r } ^ { v }$ and $\Delta _ { r } ^ { a }$ from Sec. 3.2. This keeps the reference at the same relative phase as during training.

Window-Bounded RoPE. Rolling blocks are remapped into a fixed window of W local slots. If τ $\cdot ( h _ { i } )$ is the physical timestamp of the earliest retained rolling block, a token at time τ is assigned

$$
\widetilde { p } _ { i } ( \tau ) = \tau - \tau ( h _ { i } ) ,\tag{6}
$$

while $B _ { 0 }$ remains fixed and the current block occupies the final slot. When the window is full, the oldest rolling block is evicted and the retained blocks shift locally. We use $W { = } 4 { : }$ one sink block, two rolling blocks, and one current block. Video and audio coordinates are derived from the same physical-time axis, preserving cross-modal alignment after every shift.

## 4. Experiments

## 4.1. Experimental Setup

Training Data. We train UniSwap on AVSpeech [8], a large-scale corpus of talking videos with clean speech from diverse speakers. Paired training tuples are synthesized from these unpaired clips using the swap-and-reconstruct pipeline described in Sec. 3.1. Both training and inference support three aspect-ratio buckets: 512×512, 416×704, and 704×416 pixels. Training clips contain 241 frames (approximately 9.6 seconds at 25 fps).

Implementation Details. All three stages share the frozen LTX-2.3 base model and train Low-Rank Adaptation (LoRA) adapters [13] in bf16 mixed precision on 8 GPUs with FSDP. Stage 1 applies LoRA of rank 128 to the attention projections and is optimized with AdamW (learning rate $1 \times 1 0 ^ { - 4 }$ , linear schedule) for 50,000 steps. Stage 2 applies LoRA of rank 128 to the attention projections and feed-forward layers of both modalities as well as the cross-modal attention, and is optimized with AdamW (learning rate $1 \times 1 0 ^ { - 4 } )$ for 50,000 steps. Stage 3 uses LoRA of rank 128 for all three roles and is optimized with AdamW (both generator and critic learning rates $1 \times 1 0 ^ { - 5 }$ $\beta _ { 1 } { = } 0 , \beta _ { 2 } { = } 0 . 9 9 9 )$ for 20,000 steps, with the critic updated five times per generator update. Inference runs on a single NVIDIA H100 GPU with the cache configuration described in Sec. 3.4.2. All methods are evaluated on the same source clips, reference identities, resolutions, and frame rates, and we report the mean ± standard deviation over the evaluation clips.

Evaluation Benchmarks. We evaluate on two benchmarks. (1) A short-video benchmark of 100 clips with an approximate duration of 10 seconds, collected from AVSpeech speakers disjoint from the training set and categorized by body visibility into head, half-body, and full-body shots according to the detected pose; reference portraits are randomly collected photorealistic human images with diverse poses, including head, half-body, and full-body shots with frontal and profile faces. (2) A long-video benchmark of 20 web-crawled talking videos of 1 minute in duration, with the same diversity of body shots and reference images, on which metrics are computed independently for each 20- second segment to expose temporal drift.

![](images/025afc7a10036a3a1db9277722a3be404288850e34361d780628300b4bf75f02.jpg)  
Figure 4. Qualitative comparison on the short-video benchmark. Given the reference image/audio and the source video/audio (top), video replacement methods (upper rows) transfer the appearance but keep the original voice, and voice conversion methods (middle rows) modify only the speech waveform, while UniSwap (bottom) jointly replaces the appearance and the voice, producing lip motion synchronized with the converted speech. Zoom in for details.

Baselines. We compare against three categories of methods. (1) Video replacement: MoCha [38], Wan-Animate [4], VACE [18], HunyuanCustom [14], and SCAIL-2 [39], which replace the visual identity but do not replace the speaker identity. We therefore pair every visual output with speech converted by the same Seed-VC backend and evaluate the resulting cascade as the direct competitor to joint replacement. We select Seed-VC as the strongest open-source voice-conversion method in our evaluation: it achieves the best SIG, BAK, OVRL, and SECS scores among the evaluated converters (Table 1) and is also used to synthesize the source audio in our training pipeline. Using a common backend across all video-only methods further isolates differences in their visual components. (2) Voice conversion: Seed-VC [22], CosyVoice [7], and Open-Voice [24], which convert the voice while keeping the original video. (3) Ours: UniSwap, the final streaming generator; the individual training stages are compared separately in the ablation study (Sec. 4.4).

![](images/6009653c601898dce89d415f2a52f1b3c95c027613377b98b6b1b9179cd50687.jpg)  
Figure 5. Qualitative comparison on the long-video benchmark. Frames are sampled every 10 seconds from 1-minute generations. The baselines exhibit identity drift and visual artifacts as generation progresses (highlighted in red), while UniSwap preserves the reference identity throughout the full duration. Zoom in for details.

Table 1. Quantitative comparison on the short-video benchmark. Each video replacement method is paired with the same Seed-VC audio backend, which matches the converter used to synthesize our training sources; audio-visual synchronization is measured on the resulting cascade, and video quality on the generated video. Because all video replacement baselines share the same Seed-VC output, thei repeated voice-quality values are omitted. Voice-conversion methods retain the source video. The best result in each column is in bold, and the second best is underlined. “–” denotes a metric that is not separately reported for that row.
<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">A-V Sync</td><td colspan="3">Video Quality</td><td colspan="5">Voice Quality</td></tr><tr><td>Sync-C ↑</td><td>Sync-D ↓</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>SIG ↑</td><td>BAK↑</td><td>OVRL↑</td><td>SECS ↑</td><td>SSIM↑</td></tr><tr><td rowspan="6">Video</td><td>VACE [18]</td><td>0.832±0.320</td><td>12.800±0.933</td><td>2.059±0.355</td><td>3.269±0.547</td><td>0.400±0.110</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wan-Animate [4]</td><td>2.874±1.653</td><td>11.338±1.713</td><td>2.098±0.340</td><td>3.514±0.520</td><td>0.580±0.140</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>SCAIL-2 [39]</td><td>3.289±1.592</td><td>11.269±1.623</td><td>2.409±0.352</td><td>4.067±0.399</td><td>0.630±0.150</td><td></td><td></td><td></td><td>一</td><td>一</td></tr><tr><td>MoCha [38]</td><td>3.031±1.678</td><td>11.198±1.881</td><td>2.534±0.340</td><td>4.249±0.304</td><td>0.577±0.147</td><td></td><td></td><td></td><td>一</td><td>一</td></tr><tr><td>HunyuanCustom [14]</td><td>0.894±0.401</td><td>12.991±0.864</td><td>2.319±0.414</td><td>3.816±0.489</td><td>0.624±0.142</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>OpenVoice [24]</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>3.458±0.415</td><td>3.438±0.663</td><td>2.910±0.486</td><td>0.755±0.066</td><td>0.363±0.162</td></tr><tr><td rowspan="3">Voice</td><td>Seed-VC [22]</td><td></td><td>1</td><td>一</td><td>一</td><td>一</td><td>3.489±0.335</td><td>3.750±0.571</td><td>3.074±0.445</td><td>0.829±0.047</td><td>0.212±0.115</td></tr><tr><td>CosyVoice [7]</td><td></td><td></td><td></td><td>一</td><td>一</td><td>3.461±0.249</td><td>3.738±0.456</td><td>3.041±0.366</td><td>0.802±0.051</td><td>0.137±0.106</td></tr><tr><td>UniSwap</td><td>3.633±1.236</td><td>10.304±0.849</td><td>2.097±0.238</td><td>3.758±0.318</td><td>0.629±0.136</td><td>3.486±0.327</td><td>3.563±0.543</td><td>2.988±0.366</td><td>0.730±0.064</td><td>0.269±0.161</td></tr></table>

<sup>Re-anchoring</sup>Table 2. Long-video comparison. Metrics are computed independently on three 20-second segments of 1-minute generated videos. The best result in each segment is in bold and the second best is underlined. UniSwap maintains stable quality and identity across the ful duration, while the baselines fluctuate or degrade over time.
<table><tr><td rowspan="2">Method</td><td colspan="3">0–20 s</td><td colspan="3">20-40s</td><td colspan="3">40–60s</td></tr><tr><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td></tr><tr><td>SCAIL-2 [39]</td><td>2.628±0.203</td><td>4.426±0.265</td><td>0.566±0.175</td><td>2.765±0.197</td><td>4.568±0.294</td><td>0.538±0.154</td><td>2.656±0.326</td><td>4.254±0.401</td><td>0.517±0.135</td></tr><tr><td>Wan-Animate [4]</td><td>2.241±0.397</td><td>3.766±0.635</td><td>0.554±0.119</td><td>2.271±0.413</td><td>3.741±0.701</td><td>0.533±0.115</td><td>2.238±0.364</td><td>3.628±0.714</td><td>0.528±0.114</td></tr><tr><td>UniSwap</td><td>2.224±0.226</td><td>3.966±0.331</td><td>0.596±0.122</td><td>2.236±0.189</td><td>4.001±0.276</td><td>0.590±0.126</td><td>2.259±0.191</td><td>4.032±0.254</td><td>0.596±0.118</td></tr></table>

Metrics. We evaluate three aspects. (1) Audio-visual synchronization: Sync-C (confidence, higher is better) and Sync-D (feature distance, lower is better) from SyncNet [5]. (2) Video quality: aesthetic (ASE) and image-quality (IQA) scores computed by Q-Align [35], and DINO-S [1], the cosine similarity between features of the generated frames and reference image for identity preservation. (3) Voice quality (following Seed-VC [22]): SIG, BAK, and OVRL from DNSMOS [26] for perceptual speech quality, the speaker encoder cosine similarity (SECS) to the reference voice [32], and the structural similarity (SSIM) [33] between the mel spectrograms of the generated and source audio, which measures how well the converted speech preserves the original content.

Table 3. Efficiency comparison on the short-video benchmark (241-frame clips). All measurements use one NVIDIA H100 GPU. Wall-clock FPS counts generated pixel frames per second. SCAIL-2 uses its accelerated 8-step LoRA configuration. Baseline times cover an entire clip, whereas UniSwap’s values marked by † are per block (3 latent frames or 24 pixel frames); per-step times are therefore not directly comparable across the two settings.
<table><tr><td>Method</td><td>Steps</td><td>Infer. Time (s)</td><td>Time/Step (s)</td><td>FPS↑</td></tr><tr><td>VACE [18]</td><td>50</td><td>482.56</td><td>9.65</td><td>0.499</td></tr><tr><td>Wan-Animate [4]</td><td>20 50</td><td>176.34</td><td>8.82 14.07</td><td>1.367 0.343</td></tr><tr><td>HunyuanCustom [14]</td><td>8</td><td>703.34 190.98</td><td>23.87</td><td>1.262</td></tr><tr><td>SCAIL-2 [39] MoCha [38]</td><td>30</td><td>1800.94</td><td>60.03</td><td>0.134</td></tr><tr><td>UniSwap</td><td>3†</td><td>1.76†</td><td>0.59†</td><td>13.6</td></tr></table>

## 4.2. Qualitative Results

Fig. 4 shows representative examples from the shortvideo benchmark. Video replacement baselines transfer the reference appearance but require a separate voiceconversion stage; voice conversion baselines modify only the waveform and cannot change the appearance. UniSwap jointly replaces both modalities while preserving the source composition and motion: the generated character follows the source pose and background, matches the reference appearance, and maintains lip motion synchronized with the generated speech while matching the reference vocal timbre.

Fig. 5 further illustrates long-duration generation. SCAIL-2 and Wan-Animate exhibit increasing identity drift and visual artifacts in later segments, whereas UniSwap remains stable over the full minute, consistent with the persegment metrics in Table 2.

## 4.3. Quantitative Results

Table 1 presents the short-video comparison. UniSwap obtains the highest Sync-C (3.633) and lowest Sync-D (10.304) among the evaluated replacement pipelines. Unlike cascades that generate lip motion without access to the converted speech, UniSwap generates both modalities jointly. Its visual identity similarity matches the strongest baseline within 0.001 DINO-S (0.629 versus 0.630), although its aesthetic and image-quality scores remain below MoCha and SCAIL-2. For speech, UniSwap achieves a SIG score of 3.486, close to Seed-VC (3.489), but remains below the best voice-conversion result on BAK, OVRL, SECS, and SSIM. These results reflect the trade-off between unified joint replacement and separately optimized singlemodality systems.

Table 2 compares one-minute generations by segment. UniSwap’s IQA remains between 3.966 and 4.032, and its DINO-S remains between 0.590 and 0.596. It achieves the highest identity similarity in all three segments, while SCAIL-2 retains higher aesthetic and image-quality scores. Wan-Animate’s IQA decreases from 3.766 to 3.628; SCAIL-2’s DINO-S decreases from 0.566 to 0.517. The results indicate that UniSwap preserves identity more consistently over the evaluated duration, but they do not establish superiority on every visual-quality dimension.

Table 3 compares generation efficiency. UniSwap generates a 24-frame block in 1.76 seconds, corresponding to 13.6 wall-clock FPS. This is approximately 10× faster than the fastest evaluated baseline (Wan-Animate at 1.367 FPS) and about 100× faster than MoCha. The reported SCAIL-2 result uses its accelerated 8-step LoRA rather than the 40- step default sampler. The gain comes from processing only the current block against cached context rather than updating the full sequence at every denoising step. Because 13.6 FPS remains below the 25-FPS playback rate used in our experiments, the current implementation supports streaming generation but not real-time playback; further systems optimization is required. Per-step time is also not directly comparable across rows because a baseline step updates an entire clip, whereas a UniSwap step updates one block.

## 4.4. Ablation Studies

Ablation on Training Stages. Table 4 compares the three training stages. Stage 1 obtains the strongest synchronization, video-quality, and speaker-similarity scores. Stage 2 retains comparable visual quality while enabling blockcausal generation. Stage 3 reduces denoising from 30 to 3 steps per block; compared with Stage 2, it improves DINO-S, SIG, BAK, OVRL, and SECS, but decreases synchronization, ASE, IQA, and SSIM.

Ablation on PE Offset. Removing the condition positional encoding offset from Stage 2 reduces Sync-C from 4.620 to 1.738 and DINO-S from 0.623 to 0.463, supporting its role in audio-video synchronization and reference-identity preservation.

Ablation on Feature-RoPE Decomposition. Table 5 ablates the three positional components on the long-video benchmark. Removing Window-Bounded RoPE, Reference Re-anchoring, or the Adaptive Sink Block leads to progressive degradation across segments, with the effect compounding over time: without Reference Re-anchoring, for example, IQA drops from 3.741 to 3.208 and DINO-S from 0.595 to 0.491 by the final segment.

Additional qualitative ablations are provided in supplementary Fig. 6.

## 5. Conclusion

We presented UniSwap, the first framework for streaming joint audio-visual identity replacement in talking videos. UniSwap formulates appearance and voice transfer as a unified conditional generation task and constructs aligned supervision through swap-and-reconstruct data synthesis. In-context Pretraining learns joint visual and vocal identity replacement, Conditional Streaming Adaptation introduces the Decoupled Streaming Conditioning Mask for block-causal generation, and Efficient Self-forcing DMD reduces sampling from 30 to 3 denoising steps per block. Efficient Multi-LoRA Switching enables memory-efficient distillation on a shared backbone, while Feature-RoPE Decomposition supports bounded-cache inference for longform generation. Experiments show that UniSwap achieves stronger audio-visual synchronization than the evaluated cascaded systems, remains competitive in identity preservation and perceptual quality, and maintains stable visual identity over one-minute sequences.

Table 4. Ablation on training stages and the condition positional encoding offset (short-video benchmark). The best result in each column is in bold and the second best is underlined.
<table><tr><td rowspan="2">Setting</td><td colspan="2">A-V Sync</td><td colspan="3">Video Quality</td><td colspan="5">Voice Quality</td></tr><tr><td>Sync-C ↑</td><td>Sync-D ↓</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>SIG↑</td><td>BAK↑</td><td>OVRL ↑</td><td>SECS ↑</td><td>SSIM ↑</td></tr><tr><td>Stage 1 (In-context)</td><td>5.272±1.510</td><td>9.107±0.941</td><td>2.253±0.289</td><td>3.922±0.325</td><td>0.635±0.132</td><td>3.476±0.422</td><td>3.619±0.665</td><td>3.029±0.498</td><td>0.782±0.059</td><td>0.213±0.151</td></tr><tr><td>Stage 2 (Teacher forcing)</td><td>4.620±1.187</td><td>9.581±0.752</td><td>2.233±0.304</td><td>3.893±0.332</td><td>0.623±0.134</td><td>3.349±0.578</td><td>3.059±0.736</td><td>2.687±0.549</td><td>0.681±0.071</td><td>0.480±0.156</td></tr><tr><td>Stage 3 (Self-forcing DMD)</td><td>3.633±1.236</td><td>10.304±0.849</td><td>2.097±0.238</td><td>3.758±0.318</td><td>0.629±0.136</td><td>3.486±0.327</td><td>3.563±0.543</td><td>2.988±0.366</td><td>0.730±0.064</td><td>0.269±0.161</td></tr><tr><td>Stage 2 w/o condition PE offset</td><td>1.738±1.737</td><td>11.843±1.606</td><td>2.152±0.352</td><td>3.472±0.600</td><td>0.463±0.166</td><td>3.238±0.728</td><td>3.280±0.924</td><td>2.767±0.702</td><td>0.624±0.086</td><td>0.103±0.142</td></tr></table>

Table 5. Ablation on Feature-RoPE Decomposition (long-video benchmark, per-segment metrics). The best result in each segment is in bold and the second best is underlined.
<table><tr><td rowspan="2">Setting</td><td colspan="3">0-20s</td><td colspan="3">20-40 s</td><td colspan="3">40-60s</td></tr><tr><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td><td>ASE↑</td><td>IQA↑</td><td>DINO-S ↑</td></tr><tr><td>UniSwap (full)</td><td>2.224±0.226</td><td>3.966±0.331</td><td>0.596±0.122</td><td>2.236±0.189</td><td>4.001±0.276</td><td>0.590±0.126</td><td>2.259±0.191</td><td>4.032±0.254</td><td>0.596±0.118</td></tr><tr><td>w/o Window-Bounded RoPE</td><td>2.115±0.226</td><td>3.763±0.228</td><td>0.599±0.120</td><td>2.083±0.227</td><td>3.500±0.153</td><td>0.546±0.093</td><td>2.083±0.213</td><td>3.390±0.141</td><td>0.517±0.071</td></tr><tr><td>w/o Reference Re-anchoring</td><td>2.106±0.197</td><td>3.741±0.314</td><td>0.595±0.114</td><td>2.111±0.225</td><td>3.399±0.207</td><td>0.522±0.094</td><td>2.140±0.227</td><td>3.208±0.246</td><td>0.491±0.084</td></tr><tr><td>w/o Adaptive Sink Block</td><td>2.083±0.173</td><td>3.734±0.313</td><td>0.589±0.115</td><td>2.007±0.143</td><td>3.336±0.164</td><td>0.530±0.087</td><td>2.099±0.203</td><td>3.174±0.182</td><td>0.499±0.070</td></tr></table>

## References

[1] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 8

[2] Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Sim-´ chowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 3

[3] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In ACM MM, 2020. 3

[4] Gang Cheng, Xin Gao, Li Hu, et al. Wan-animate: Unified character animation and replacement with holistic replication. In arXiv preprint arXiv:2509.14055, 2025. 2, 3, 7, 8, 9

[5] Joon Son Chung and Andrew Zisserman. Out of time: Automated lip sync in the wild. In ACCV Workshop, 2016. 8

[6] Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and´ Yossi Adi. High fidelity neural audio compression. In Transactions on Machine Learning Research, 2023. 2

[7] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Haizhou Zheng, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. In arXiv preprint arXiv:2407.05407, 2024. 2, 3, 7, 8

[8] Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson, Avinatan Hassidim, William T. Freeman, and Michael Rubinstein. Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation. In ACM TOG, 2018. 6

[9] Zhengcong Fei, Di Qiu, Changqian Yu, Debang Li, Mingyuan Fan, and Xiang Wen. Video diffusion transformers are in-context learners. arXiv preprint arXiv:2412.10783, 2024. 3, 4

[10] Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, and Ran He. Infoswap: Information bottleneck disentanglement for identity swapping. In CVPR, 2021. 3

[11] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, et al. Ltx-2: Efficient joint audio-visual foundation model. In arXiv preprint arXiv:2601.03233, 2026. 2, 3

[12] Xuanhua He, Quande Liu, Zixuan Ye, Weicai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, and Kun Gai. FullDiT2: Efficient in-context conditioning for video diffusion transformers. arXiv preprint arXiv:2506.04213, 2025. 3

[13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 6

[14] Teng Hu, Zhentao Yu, Zhengguang Zhou, et al. Hunyuancustom: A multimodal-driven architecture for customized video generation. In arXiv preprint arXiv:2505.04512, 2025. 2, 3, 7, 8, 9

[15] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Tong Feng, Gan Liu, and Jun-

wei Yang. In-context lora for diffusion transformers. In arXiv preprint arXiv:2410.23775, 2024. 3, 4

[16] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In arXiv preprint arXiv:2506.08009, 2025. 3

[17] Yuepeng Jiang, Ziqian Ning, Shuai Wang, Chengjia Wang, Mengxiao Bi, Pengcheng Zhu, Zhonghua Fu, and Lei Xie. Ref-vc: Robust, expressive and fast zero-shot voice conversion with diffusion transformers. In arXiv preprint arXiv:2508.04996, 2025. 2, 3

[18] Zeyinzi Jiang, Zhen Zhu, Chaojie Qi, Xin Jia, et al. Vace: All-in-one video creation and editing. In arXiv preprint arXiv:2503.07598, 2025. 2, 3, 7, 8, 9

[19] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Faceshifter: Towards high fidelity and occlusion aware face swapping. In CVPR, 2020. 3

[20] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. In ICLR, 2026. 3

[21] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Jiebo Luo, Ziwei Liu, Hao Fei, and Tat-Seng Chua. JavisDiT: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. In ICLR, 2026. 3

[22] Songting Liu. Zero-shot voice conversion with diffusion transformers. In arXiv preprint arXiv:2411.09943, 2024. 2, 3, 4, 7, 8

[23] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3

[24] Zengyi Qin, Wenliang Zhao, Xumin Yu, and Xin Sun. Open voice: Versatile instant voice cloning. In arXiv preprint arXiv:2312.01479, 2023. 7, 8

[25] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025. 4

[26] Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP, 2022. 8

[27] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. MM-Diffusion: Learning multi-modal diffusion models for joint audio and video generation. In CVPR, 2023. 3

[28] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 3

[29] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In Neurocomputing, 2024. 3, 5

[30] Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Ying Li, Zezhong Qian, Haoyang Huang, and Nan Duan. Omniforcing: Unleashing real-time joint audio visual generation. arXiv preprint arXiv:2603.11647, 2026. 3

[31] Wan Team. Wan: Open and advanced large-scale video gen erative models. In arXiv preprint arXiv:2503.20314, 2025. 2

[32] Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno. Generalized end-to-end loss for speaker verification. In ICASSP, 2018. 8

[33] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. 8

[34] Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. In Neural Computation, pages 270–280, 1989. 5

[35] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels. In Proceedings of the 41st International Conference on Machine Learning, pages 54015–54029. PMLR, 2024. 8

[36] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, 2024. 6

[37] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. In NeurIPS, 2022. 4

[38] Zhengbo Xu, Jie Ma, Ziheng Wang, Zhan Peng, Jun Liang, and Jing Li. Mocha: End-to-end video character replacement without structural guidance. In arXiv preprint arXiv:2601.08587, 2026. 2, 3, 7, 8, 9

[39] Wenhao Yan, Fengjia Guo, Zhuoyi Yang, and Jie Tang. Scail-2: Unifying controlled character animation with end-to-end in-context conditioning. In arXiv preprint arXiv:2606.10804, 2026. 7, 8, 9

[40] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Chen, Xiaotao Dai, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In arXiv preprint arXiv:2408.06072, 2024. 2

[41] Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang,¨ Eli Shechtman, Fredo Durand, and William T. Freeman. Im-´ proved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 3

[42] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht-¨ man, Fredo Durand, William T. Freeman, and Taesung Park.´ One-step diffusion with distribution matching distillation. In CVPR, 2024. 2, 3

[43] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From ´ slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 3

# UniSwap: Streaming Audio-Visual Identity Swapping for Talking Videos

Supplementary Material

## 6. Qualitative Ablation

Figure 6 complements the quantitative ablation in the main paper by visualizing the effect of each component of Feature-RoPE Decomposition over one-minute generations. The ablated variants exhibit increasing identity drift and visual artifacts in later segments, whereas the full model remains more consistent. Together with Table 5, these results support the roles of bounded coordinates, reference re-anchoring, and the adaptive sink block in reducing longhorizon drift.

## 7. KV-Cached Streaming Inference

Algorithm 1 details the blockwise inference procedure used by Stages 2 and 3. The reference cache persists throughout generation, source keys and values are temporary, and completed target blocks are committed to the clean-history cache.

Algorithm 1 KV-Cached Streaming Inference   
Require: Reference tokens $R ;$ source blocks $\{ S _ { i } \} _ { i = 0 } ^ { N - 1 }$   
reference region R; history H (sink + rolling)   
1: Prefill reference: forward $R$ once and write its key/-   
value tensors into $\mathcal { R }$ {never evicted}   
2: for $i = 0$ to $N - 1$ do   
3: Prefill source: forward $S _ { i }$ at the current slot and   
temporarily cache its key/value tensors   
4: Denoise: iteratively denoise $B _ { i }$ in read-only mode,   
attending to ${ \mathcal { R } } ,$ H, and $S _ { i }$   
5: Commit: forward the denoised $B _ { i }$ as clean context,   
and append its unrotated keys and values to the sink   
region if $i = 0 ,$ or to the rolling region otherwise   
6: Shift: once the rolling region is full, evict its oldest   
block, shift the retained blocks, and reapply RoPE to   
their cached keys at bounded local coordinates   
7: Rollback: discard the temporary key/value tensors   
of $S _ { i }$   
8: end for   
9: return Generated blocks $\{ B _ { i } \} _ { i = 0 } ^ { N - 1 }$

## 8. User Study

We conducted a blinded user study with 30 participants. The study compared UniSwap with four video-replacement baselines, each paired with Seed-VC following the cascade protocol in Table 1. Each participant evaluated anonymized outputs from all five methods on four source clips and rated appearance identity, voice identity, lip synchronization, and naturalness on a five-point Likert scale. Method order was randomized independently for each participant.

Table 6. User-study results. Ratings are mean scores on a fivepoint Likert scale.
<table><tr><td>Method</td><td>Appearance ID ↑</td><td>Voice ID ↑</td><td>Lip Sync ↑</td><td>Naturalness ↑</td></tr><tr><td>Wan-Animate + Seed-VC</td><td>3.85</td><td>4.04</td><td>3.42</td><td>3.77</td></tr><tr><td>SCAIL-2 + Seed-VC</td><td>4.03</td><td>4.05</td><td>3.67</td><td>3.85</td></tr><tr><td>MoCha + Seed-VC</td><td>3.91</td><td>4.08</td><td>3.54</td><td>3.89</td></tr><tr><td>HunyuanCustom + Seed-VC</td><td>3.64</td><td>3.98</td><td>3.28</td><td>3.61</td></tr><tr><td>UniSwap</td><td>4.16</td><td>3.87</td><td>4.11</td><td>3.96</td></tr></table>

UniSwap receives the highest ratings for appearance identity, lip synchronization, and naturalness.

## 9. Additional Qualitative Results

We provide additional qualitative results for both short and long source videos. In all figures, each example contains a reference image and reference voice clip, a source video and its audio, and the joint audio-video output produced by UniSwap. The red and blue waveforms denote the source and generated audio, respectively. These examples complement the quantitative evaluation in the main paper by covering diverse identities, poses, gestures, clothing, backgrounds, and recording conditions.

## 9.1. Short-Video Results

Figures 7 and 8 present 16 additional short-video examples. Across these cases, UniSwap transfers the appearance specified by the reference image while retaining the source composition, body motion, and facial activity. The examples include cross-gender replacement, varied camera framing, substantial hand and upper-body motion, and both simple and cluttered backgrounds. The corresponding generated-audio waveforms are shown alongside the visual outputs to illustrate that the two modalities are produced jointly for the full source sequence.

## 9.2. Long-Video Results

Figure 9 shows three additional one-minute generations sampled at 10-second intervals. The output identity remains visually consistent from the beginning to the end of each sequence despite continuous autoregressive generation. At the same time, the outputs preserve the source background, camera framing, and time-varying facial and upper-body motion. These results provide further qualitative evidence that the bounded cache coordinates and persistent identity context used by UniSwap mitigate long-horizon identity drift.

![](images/18db7fcb0124a99a771e92e80e6cb4c1fba0a1723d81a729dbd4f30220fc9d01.jpg)  
Figure 6. Qualitative ablation on Feature-RoPE Decomposition. Frames are sampled every 10 seconds from one-minute generations. Removing any component leads to visible identity drift and artifacts in later segments, whereas the full model remains more consistent throughout the sequence. Zoom in for details.

## 10. Limitations

UniSwap currently targets single-speaker talking videos; multi-speaker scenes, occlusions, and complex interactions remain challenging. Facial expressions are driven automatically by the audio condition rather than controlled explicitly, so the current model does not support independent expression editing or arbitrary user-specified expression control.

## 11. Broader Impact

Audio-video character replacement can support filmmaking, localization, and accessibility, but it also increases the risk of impersonation, non-consensual media, and misinformation. Deployment should require consent and provenance mechanisms, visible disclosure where appropriate, access controls, and compatibility with forensic detection tools.

Ref image/audio  
Ref image/audio  
Ref image/audio  
Ref image/audio  
![](images/c678a8297a2c2efdd86a97d03faff501f64fe148e32b6723c5be8a5508eb70ad.jpg)

Src video/audio  
![](images/3698bf373da253a8982e5b25b33dc766068e86a3c58d0d5b8b2d1209d721f6cf.jpg)

Src video/audio  
![](images/1ae64a12c5dc2f5ff76588eb6a57912f2d39f0019d844c8d6419a6ff99e99e38.jpg)

![](images/32825ab882fc6b033deef844c75a5c66dfeec0d75764dae38e5d721a43b7d906.jpg)

![](images/083ae3a9dc35236d70a1bf749f25daa7e3c42e32047f8bc3917285b28ad6cfa7.jpg)

![](images/01943b0450e70125eebb3d8135a31efbccc48cc4a1cb0de80b94b1222b9873ee.jpg)

![](images/2c4a3d6c0dfdcad9b8cc25339e27f2692cbf3b510b35319a63bcfad57a28371c.jpg)

![](images/76861bc0a7352ad21f3b8b20edfbe033edc8ba4693498a8ef3737ccb91ebcb5b.jpg)

![](images/e7b51ddeaff08fbad18e40115f4122fbc05d31e3e0f475aa66ef45fac1e1f449.jpg)  
Output  
Output

![](images/fe13017c5535ed5ad59cc67cc7c0baaa9f251e66bb5e10c053415a075f59270b.jpg)

![](images/4987a3c3a761c8a4eb9d9a13e98bfa15da47e795dc71f083124c53ac72687b45.jpg)

![](images/0321d84c61c8b2ffdb88c37a8715e2f271a2be0028e22d5597b6fe02acb58fad.jpg)

![](images/c79cece43ef7bea387de6fe531e681e63a7dfc1f6b5140696d3a5cbef7689c1e.jpg)

Src video/audio  
![](images/b314d92bd157389e50fa21c781f183bcca11fb4f2bbd4a2fa02b1b03f9d64221.jpg)  
Src video/audio

![](images/b2c16315399b0a80ab6888829fce8f0512b4f33d97e57b290b482f7b0542321e.jpg)

![](images/72d650e4257375eb6c5c70fb922c364c697f9858451eb6fa4de3f7ec4e47ef32.jpg)

![](images/b24fd845fb0e4433bf7bab9b9949644f95a11d82b57a7a9de9dac31194d3fdbc.jpg)

![](images/5f2e8755924f0634c7aa25b73696d96f0d80b040383057e37090bd3c7acf0d86.jpg)

![](images/ea8ed3559bdb5edf7dbd316eb8ee3f46b96139a3abc27898e2b4e63b7d995965.jpg)

![](images/939da34f7e160464927cc686576b0cb1a2efa94960b79b1c2b8be7846e61964d.jpg)  
Output  
Output

![](images/177e7d981f2d99fb2f0107ff7836caca7e0944b1512bd06ae01888e9da0b7b16.jpg)

![](images/0040c51f665c9b31764c23b24309984d9acd24ab4127706741a71156d8c2cb61.jpg)

![](images/aca715a7bb4973483a55c00a96a5098a8bf95adedd666ec07c79e19588ad0f8e.jpg)

![](images/135ee455fd045cf40cfdaf508edbaf4d17900a287b9e78dce9cdca7b2bd3e1c5.jpg)

![](images/b3ca166be42fda57cf5c8095bad4a52675347eafe31be9f5a058ca4570976d16.jpg)

Src video/audio  
![](images/208d2067177114bf6c786c26e8dde13b76dcdc9baf139e8fd5daf27e9564455f.jpg)

![](images/703a1dc9b78b77249bf344143e8979cc360f62cea568d5cca4408cfa9a82efdd.jpg)

![](images/89861034b2d521e47404d1d2d73692cbecd295f42ef35cc08c983f0c3046da75.jpg)  
Src video/audio

![](images/8da4fe9db4677f6a9a8f8a47462b733ff8e6106687cf923e36bb9d2bb6e09d4a.jpg)

![](images/11ce53d8b4a5bd84d62d2dc59f9e98f088e64fbf4870053827e06fa0504c5aa5.jpg)

![](images/88117e6de029a10f4e55b5413c048f5aa081ae20d5ba06ce7e290bffef932071.jpg)  
Output

![](images/91f3b1d6d8d1a745e889fc7b711f1ef42969e05f8fc1d53e2df02aa8a7a8a1f4.jpg)

![](images/83584cff183ba74172cc13dbc23b8f00d4001b4c64c76cd348c1d9460133203a.jpg)

![](images/760aee042786cd0d940269f74fd759a4457212b007907164f0f0d7c81177cb6c.jpg)  
Output

![](images/25d638a4e6889b0346e278e54970e8c1874a44f3048b11824b2ca14a2ad7148d.jpg)

![](images/bb7bb2862011ad54479ef6f05933734a5879e28c8fedd4e1fa3c4a4eb692814c.jpg)

![](images/df472402ae92fcf4ec5740ab31208027f1ff71b3b4b641face094bc6c48220a4.jpg)

Src video/audio  
![](images/1924463e263541868fe1b5442926791440b45035bd94625c142ecccddecea5f2.jpg)

Ref image/audio  
Src video/audio  
![](images/26b46503aec307ce935c02539d5183551f2391c732ed0db48ccd6bd69349d086.jpg)

![](images/59e725c85cd52e9ff3098fef0b46c9aecb9dea00e5411bc5b1fbdd058ab6ed06.jpg)

![](images/12567f380f200e088a24f47e0a40a744900a01bb4d0a1892eec5dba98295ec2a.jpg)

![](images/67e18547d44b63ebdf8f155fd38f93ba34baacbceffe5c93c1d9ad2ed2ac2db6.jpg)

![](images/1006e4b02effd3d6d5b2c04ce4a670cbfa0fc53ac7b635f41b643970a230b7ba.jpg)

![](images/965cde6a5ed106dc9d4674bfd26da8b240be1fe66fbcc9d3639227e11ed28cb9.jpg)  
Output

![](images/21e5f42a18b934cb5326ee97e160725941c31949f8e5f3d469f02d4206f82aec.jpg)

![](images/a333a8a8b4772021154277fb3249b1981a04fab9e588571b2a769feda6994a15.jpg)  
Output

![](images/7b831a25e05b4d70e1f81bea6c76a5a5f476ed4bc972365c45e89cbd1d0a00e7.jpg)

![](images/1262f7b12fb60dab7d70b96ba87fcb459d8d1175d6e2cf14e3f4a3068d43768b.jpg)  
Figure 7. Additional qualitative results on short videos (Part I). For each example, the left column provides the reference image and reference voice, while the right column shows sampled frames and the waveform of the source video/audio followed by the UniSwap output. Red waveforms indicate source audio and blue waveforms indicate generated audio. UniSwap changes the visual and vocal identity according to the references while preserving the source scene composition and motion.

Ref image/audio

Src video/audio

Ref image/audio  
![](images/3400cfe1caf5eb0b69aa4639244896f86791e80a65027e4fbc5eb4797658d0a7.jpg)  
Ref image/audio

Src video/audio  
![](images/1a10f4026f6ac5509fc3dfbd38eae3f966ced71ef399520296ad62081d68540f.jpg)

![](images/e20115e36d4ad94857766372d4a56a8ab92bdc968a40f80be46c91755a211421.jpg)

![](images/9fd3c73501b560e7e9a61678972b8097325bea45d322a36a17a79743383f8545.jpg)

![](images/06889245a76b0def8bfe3c88b46292e68c78aeb1fbd5cd4d5a93fefd07266a5f.jpg)  
Output

![](images/8439967ec026cbce06e3f982c4c1104e3d1dc5cb04f17b1c84609582fd8b2473.jpg)

![](images/9bc20ea0af9c7d72f8d39b2605403cece8883f0ee31d75581e81bc3cf7ec6cb7.jpg)  
Output

![](images/0b2657716508f7d793592d71bb71ec447c3ea78224ef1cbc0a1d0078d2db35c3.jpg)

![](images/0f1438562cd8c93e7134b57e76f05c9be90a139aa9b4906191717a62d52f2810.jpg)

![](images/07993107a56b9e0f243fb0e21b3089f25b01cd8bc7509153f58f321b51c3afa0.jpg)

![](images/729225d7ce0fb8597c6e555426720e8c0fc7ceb067a06157d6bcbcfd56c70bcb.jpg)

![](images/8c73a8e96989ff733a2d4449c5bfee250b9ed71feb7d231fdb2dde0480bea72d.jpg)

![](images/1b762036aed91cfbceed550c84ef9389e4aa2c9f342ea3af856f93e1b884328e.jpg)  
Output

![](images/6e9f88fbe6ed3c313626bb17feb295844db9092997b8257886d1d12ae39af5ff.jpg)

![](images/2384065feb2018df09f8a1838b5238cac25eff0c46b64255b58a7cf6e40539fa.jpg)  
Output

![](images/7152c669a833010976bfe8be2de849dc66f10ca5c3182d908143d889cbfd386d.jpg)  
Ref image/audio

![](images/b659a90d6ce121f89bfb68a7048fea7e4ab1cc4b694d1dfe60f3e6e72b8de9fd.jpg)

![](images/4c714f2362aa0053b93b520d57b01189e73b7ecf61e965fdbd631061ca5137df.jpg)

Src video/audio  
![](images/de1627313429b81b6c6f9e9d69044fb0d5bfaedbdaa14e642c4d3faf4e66d1cb.jpg)  
Output

![](images/62bf6f7236eee95e438c944207be349dc2e0f7e300b69874f5f12f616e75ca7c.jpg)

![](images/36d3d151c81b590d37d110b1afac0d5fd2598e3bb69b298aed22d4e62655542b.jpg)

![](images/fa93091365a1d5e72e1140005c90197d0bafcfd84579806d2f75c3f16326d8ba.jpg)

![](images/d89b0af718335850c8bb63b18828650c1826e876779229aa936ebce1e9c022d6.jpg)

![](images/fc1ea46df5380e130e2a73cf8e6ad40080f1ff48c7b2caef48defe8637d32256.jpg)  
Output

![](images/9a3c6f94207b7b198e00ff363886fb28a26ccb1c6bdb97df65433db4b57fe984.jpg)

![](images/44ba4117d9f4dc74e969891a5e36445c026dc82978c24731bb43644e6865bbed.jpg)

![](images/49cf3f53750f1b6a1dd0f60542c0876508352f5948f6bc63e7c4d5983d66fe86.jpg)

![](images/ad1d35c9bb72d9519f0c86f18e22b348ed18bf062e943d45024da6cc600824d7.jpg)  
Output

![](images/a96be49f2e3bd28006bc5ffe93074ee265354d0dfb59473aae32f6dc83d9723c.jpg)

![](images/f24183e489595c8bf3b6e9162145a8f2fd32677d7a30c00779a4d3a373552a0b.jpg)  
Output

![](images/1955f2dc6a7a2036d591c9cf79f15aa783f92479531a580b6c3f593f5cf4ad5d.jpg)

![](images/ed2526b0ecb87ac19deddbfeb84e166faf69fdbb3c2b54cf0eaf8d8ab0434c8b.jpg)  
Src video/audio

![](images/57e350e91b58c9900bf360b88715edf1d7b9245d4211ebed8ae2e297be81ad6c.jpg)

![](images/a315fd6593ad944a2951bbdcf07455ac5a3258066cecbb6163aac500682c5255.jpg)

![](images/9f55aa3d09f71c8126b038c70e0af01d9b8a32992f8b8f33963613d1bfbb10a4.jpg)

![](images/8956cc7bf3f454c6df44200d86affda23dea38b6778ee9e144c20b3bedcd534f.jpg)

![](images/5b7b8f67d8974a32a7bc24b4cdbf7b128b58165b57cd3e7cb78f97fea9aff74c.jpg)

![](images/117d15cb5b0bb00ba08fab1265272431ad755f62bb79d649774cec5862212de1.jpg)

![](images/8aa5b7958f4f94ef1174969d0e47ee65df44ab12e4bc7908398373d63f1bbe61.jpg)

![](images/e38dd5982f5ba148893653b0cb077bd892efdfab19c890f7823b391749d97986.jpg)

![](images/85b6a8f1b6124820e5a5a07aa65cabc46324d0ccee3f0ba9f94f868bfe1975d7.jpg)

![](images/61cef1125b28c7d096f6b1097c144bd9987aedf190719aa468881070cc1c9ef5.jpg)

![](images/fc2c5f4975804b0b42be2e662f4c7055c0850324ba369a2e17da64199342e8e7.jpg)

![](images/8332fb444a29f7739b2ad04180930a635700ee8b8acff9dfffae0d1f23b9ce3c.jpg)

![](images/54110f1c7a267ce88e5aa824d817029c0dc9f6de3fd448ccb67e9391f8d5fa76.jpg)

![](images/8c7290abce38e423979158e21961e27a716729a5d278e7b33709db9ad2d41606.jpg)

![](images/178bf0094b4488ae3cd3a0ac69b63b20b392fc3c230709ca3683af8929b59f5a.jpg)  
Figure 8. Additional qualitative results on short videos (Part II). The examples follow the layout of Fig. 7 and cover additiona identities, viewpoints, gestures, and backgrounds. Red and blue denote the source- and generated-audio waveforms, respectively. The generated frames adopt the reference appearance while following the pose, expression, framing, and scene content of the source video.

![](images/baedc4ca16a55b1979a4e810e68f1efeeff2828d020ecd4adca2c85c0d82305c.jpg)  
Figure 9. Additional qualitative results on one-minute videos. Frames are sampled every 10 seconds from 0 to 60 seconds. Each row group shows the reference image and voice, the source video/audio (red waveform), and the UniSwap output (blue waveform). Across all three sequences, the generated character remains consistent with the reference identity throughout the minute while retaining the source scene and temporal performance.