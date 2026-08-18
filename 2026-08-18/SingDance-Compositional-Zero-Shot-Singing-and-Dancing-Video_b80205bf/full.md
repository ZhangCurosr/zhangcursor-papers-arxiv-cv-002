# SingDance: Compositional Zero-Shot Singing-and-Dancing Video Generation with Role-Aware Audio Conditioning

Tao Feng<sup>1,2∗</sup>, Xu Li<sup>1†</sup>, Xiangyang Luo<sup>3</sup>, Ming Wen<sup>1</sup>, Huadai Liu<sup>1,2</sup>, Chen Zhang<sup>1</sup>, Wei Xue<sup>2†</sup>

<sup>1</sup>Kling Team, Kuaishou Technology

<sup>2</sup>The Hong Kong University of Science and Technology <sup>3</sup>Tsinghua University

{tao.feng,huadai.liu}@connect.ust.hk, {lixu15,wenming,zhangchen03}@kuaishou.com goodluoxy@gmail.com, weixue@ust.hk

## Abstract

Generating personalized dance videos from a reference image, text prompt, and audio track requires music-conditioned body motion. Singing-and-dancing adds a second requirement: the visible subject must also articulate the vocals. Existing musicconditioned methods focus primarily on choreography, while speech-driven models generally assume that the visible subject produces the input voice, leaving this combined setting largely underexplored. We introduce SingDance, a unified video difusion framework that formulates controllable vocal articulation as a semantic role: the visible subject is either the source, who produces the vocal signal, or the listener, who receives it from an of-screen performer. Hard-compact routing selects task-relevant speech, music, and role conditions, which are composed through frame-wise joint audio injection; source and listener retain the same speech pathway. Training uses asymmetric supervision: on-screen speaking and curated of-screen conversational-response videos establish role control, while instrumental and song-based dancingonly videos establish music-conditioned body motion. The target Song/Source configuration is never observed during training. At inference, assigning the source role to a song composes separately learned articulation and song-conditioned dance capabilities, enabling compositional zero-shot singingand-dancing. Experiments demonstrate strong motion–beat alignment and visual fidelity, reliable paired switching of vocal articulation while preserving music-aligned body motion, and highly competitive lip synchronization with substantially fewer generation-time parameters than the strongest speechdriven baseline evaluated.

Project Page —

https://ff-ttt.github.io/singdance-project-page/

## Introduction

Short-form dance videos are a major form of online visual content. They broadly take two forms: dancing-only, where an on-screen subject moves to instrumental music, or to a song without lip-syncing to its vocals, and singing-anddancing, where dance motion and vocal articulation occur together. Given the same subject and song, creators may wish to produce either form. A single framework supporting both from a reference image, text prompt, and audio track would substantially simplify personalized video creation. Despite its practical relevance, singing-and-dancing remains largely unexplored in music-conditioned human video generation.

![](images/5e701dcef393ade0765075ba9947f961e3502f4a2110014a94ff50600df5c22c.jpg)  
Figure 1: Four generation tasks supported by SingDance. The dashed box marks singing-and-dancing, the held-out Song/Source configuration generated through compositional zero-shot inference.

Existing research addresses adjacent parts of this problem under diferent assumptions. Music-conditioned dance methods focus primarily on body choreography and rhythmic alignment, and generally do not model vocal articulation by the dancer (Chen et al. 2025c; Hong et al. 2026; Yang et al. 2026a; Huang et al. 2026). Speech-driven human video models achieve accurate articulation and expressive motion, but conventionally assume that the input voice is produced by the visible subject (Gao et al. 2025; Gan et al. 2025; Yang et al. 2025b; Meng et al. 2026). Recent interactive avatar systems further model speaking and listening as distinct conversational phases, often with separated or phasespecific audio pathways (Zeng et al. 2026; Sun et al. 2026).

Song-conditioned dance raises a diferent question: how to control whether the visible subject produces the vocals while retaining the same vocal content as part of the conditioning for body motion.

Our key insight is to model controllable vocal articulation as an explicit semantic role rather than an isolated mouth-motion instruction. We define the visible subject as either the source, who produces the vocal signal, or the listener, who receives it from an of-screen performer. This distinction prevents the presence of vocal content from automatically assigning articulation to the visible subject. Genuine listening/reacting examples provide non-speaking supervision for the listener role, rather than defining Lip-Sync OFF as a frozen-mouth instruction. This semantic distinction transfers naturally from speech to songs, organizing four behaviors—speaking, listening, dancing-only, and singingand-dancing—within a single generation problem.

We instantiate this formulation in SingDance, a video diffusion transformer conditioned on a reference image, text, representations specialized for speech and music, and an explicit vocal role. Hard-compact routing retains the taskrelevant token groups: speech interaction uses speech and role tokens, instrumental dancing-only uses music and role tokens, and song-based dancing-only uses all three. Importantly, the two vocal roles share the same speech pathway; compact role conditions work with the prompt’s vocal role field to specify whether the visible subject is assigned the vocals. For song-based dancing-only, vocal features remain available because lyrics and vocal phrasing may still inform the performance even when they are not articulated by the visible subject.

Training follows a staged design with asymmetric supervision. We first learn role control from on-screen speaking clips and carefully curated conversational clips in which the visible subject genuinely listens or reacts to an of-screen interlocutor. We then learn music-conditioned dance from instrumental and song-based dancing-only videos, without singing-and-dancing supervision. At inference, singing-anddancing reuses the complete speech-plus-music configuration learned from song-based dancing-only and changes only the vocal-role state from listener to source. This state is expressed consistently through the learned role conditions and the prompt’s vocal role field. This compact intervention enables paired role switching and instantiates the held-out Song/Source configuration through compositional zero-shot inference.

Our contributions are threefold:

• We formulate reference-image-conditioned dance video generation with an explicit vocal role, organizing speaking, listening, dancing-only, and singing-and-dancing as four related generation behaviors.

• We introduce a role-aware conditioning design in which source and listener share the same speech representation and injection pathway, while task-routed speech, music, and role tokens are composed through frame-wise joint audio injection.

• We learn source-role articulation and song-conditioned dance from asymmetric supervision and demonstrate their compositional zero-shot generalization to the heldout Song/Source configuration. Paired role intervention verifies controllable articulation, while motion–beat evaluation and music-conditioning ablation validate musicresponsive body motion.

## Related Work

Music-conditioned dance generation. Musicconditioned dance generation is inherently one-to-many, and prior work adopts diferent intermediate representations to structure this ambiguity. Early methods generate 3D body motion before rendering, including FACT on the AIST++ dataset (Li et al. 2021), Bailando (Siyao et al. 2022), EDGE (Tseng, Castellon, and Liu 2022), and FineDance (Li et al. 2023). More recent systems move toward RGB video while retaining explicit motion intermediates: X-Dancer predicts music-aligned 2D pose tokens (Chen et al. 2025c), whereas MACE-Dance cascades motion and appearance experts (Yang et al. 2025a). A complementary line directly adapts video difusion models. MusicInfuser introduces music conditioning into a text-to-video backbone (Hong et al. 2026). Concurrent studies further explore diferent allocations between text and music: OmniDance combines image, text, and MERT features in a 5B video backbone (Yang et al. 2026a), while the more recent Wan-Dancer uses coarse textual attributes and hierarchical global-to-local music conditioning for long-form generation (Huang et al. 2026). These concurrent methods explore diferent allocations of text and music for choreographic control. SingDance studies an additional compositional axis: whether the visible dancer produces the vocals while retaining music-conditioned body motion.

Speech-driven human video generation. Speech-driven human video generation spans subject-specific neural rendering, including audio-driven 3D Gaussian talking heads (Xie et al. 2025), and generalizable difusion-based models that have substantially improved identity preservation, lip synchronization, and natural co-speech motion (Chen et al. 2025b; Gan et al. 2025; Gao et al. 2025; Yang et al. 2025b; Meng et al. 2026; Meituan LongCat Team 2026). Complementary work studies general audio–visual synchronization representations for evaluation and generation guidance (Feng et al. 2025). Most systems, however, assume that the visible subject produces the input utterance and rely on speechpretrained encoders, such as Wav2Vec or Whisper, for temporally aligned conditioning (Baevski et al. 2020; Radford et al. 2023). Although these representations capture phonetic and prosodic cues for articulation, they are not explicitly optimized for music-specific structures such as beat, timbre, and longer-range rhythm. Interactive-avatar and 3D facial-animation systems instead model speaking and listening through dual-speaker context, listener-specific motion priors, or phase-specific audio pathways (Peng et al. 2025; Chu et al. 2026; Wang et al. 2026; Weng et al. 2026; Zhang, Shen, and Li 2026). DualTalk models two-speaker interaction (Peng et al. 2025); UniLS separates an internal listenermotion prior from external audio modulation (Chu et al. 2026); LPM 1.0 uses independently parameterized speaking and listening streams (Zeng et al. 2026); and StreamAvatar routes speech features through dedicated Talk and Listen attention modules (Sun et al. 2026). These designs account for distinct conversational motion distributions. SingDance instead keeps the speech representation and conditioning pathway shared across roles, changing coordinated textual and learned role conditions. This permits held-out Song/Source composition while retaining the speech and music conditions used for song-based dancing-only.

![](images/f4b072b7c7fd853a99b57789751c77f35446abcfcaae208d1a675118cc70448b.jpg)  
Figure 2: Training architecture of SingDance. The first video frame is separately encoded as a clean reference latent, while subsequent frames are temporally compressed and noised in latent space. Wav2Vec 2.0 and MuQ features, together with the vocal-role condition, are selected by hard-compact routing and supplied to frame-wise joint audio injection. The repeated block is schematic: joint audio injection is applied immediately after each of 10 selected blocks in the 30-block DiT backbone. The denoised latents are decoded into output frames. Snowflakes and flames denote frozen and trainable modules, respectively.

Multimodal conditioning in video foundation models. Pretrained video difusion models provide reusable foundations that can be extended with additional control modalities. Wan-S2V combines text-based global motion specification with frame-aligned speech injection (Gao et al. 2025). ActAvatar builds upon Wan2.2-TI2V-5B and combines temporally structured text conditioning for phase-level actions with Wav2Vec 2.0-based speech cross-attention for articulation (Peng et al. 2026). For music-conditioned generation, OmniDance augments the same TI2V backbone with MERTbased music cross-attention, whereas Wan-Dancer adapts Wan-I2V using compact Librosa features in a hierarchical generation pipeline (Yang et al. 2026a; Huang et al. 2026). These approaches illustrate diferent ways of assigning control responsibilities to text and audio while preserving the generative prior of a pretrained video model. SingDance assigns each modality a primary responsibility according to the information it represents most naturally: the reference image anchors appearance and the initial scene; text specifies intended action and camera evolution; speech captures vocal articulation and phrasing cues; and music complements the prompt with rhythm and broader musical structure that are dificult to express textually. This allocation motivates separate speech and music representations, composed through frame-wise joint audio injection, while explicit vocal-role control determines whether the visible subject articulates

the vocals.

## Method

In this section, we first introduce the audio representations and vocal-role control, then describe hard-compact routing and joint audio injection, and finally present staged training for zero-shot composition and text–audio classifier-free guidance.

## Audio Representations and Vocal-Role Control

To coordinate vocal articulation with music-conditioned body motion within a single generation framework, SingDance employs representations specialized for speech and music together with coordinated textual and learned vocal-role controls.

As illustrated in Figure 2, we use Wav2Vec 2.0 (Baevski et al. 2020) to extract speech-oriented features for vocal articulation. Following Wan-S2V (Gao et al. 2025), we aggregate its layer-wise representations using learnable weights and map them through a temporal encoder to a global speech feature and frame-aligned local speech tokens. In parallel, MuQ (Zhu et al. 2025) provides music-oriented acoustic and structural features, which a separate temporal encoder maps to frame-aligned local music tokens for body-motion conditioning. Both temporal encoders use non-causal 1D convolutions, matching the non-causal context of the pretrained features.

At the level of generated behavior, the two vocal roles correspond to the Lip-Sync ON and OFF states shown in Figure 1. Semantically, source indicates that the vocal signal is produced by the visible subject, whereas listener indicates that it originates from an external source. We thus treat lip synchronization as an outcome of vocal-role assignment rather than an isolated mouth-motion instruction.

To expose this distinction to the text encoder while separating vocal behavior from choreography, we organize each prompt into five fields: vocal role, style, action, first frame, and camera. The standardized vocal role field assigns the vocal signal either to the visible subject or to an external source, corresponding to source for Lip-Sync ON and listener for Lip-Sync OFF. During prompt construction, we remove references to lip movement, speaking, and singing from the action field, retaining only body-motion descriptions. Within the text prompt, vocal behavior is therefore specified exclusively by the vocal role field, separating its textual control from the intended body-action sequence.

<table><tr><td>Audio type</td><td>Role</td><td>Routed tokens</td><td>Task</td></tr><tr><td>Speech</td><td>Source</td><td> $S + R _ { \mathrm { s r c } }$ </td><td>Speaking</td></tr><tr><td>Speech</td><td>Listener</td><td> $S + R _ { \mathrm { l i s } }$ </td><td>Listening/Reacting</td></tr><tr><td>Instrumental</td><td>Listener</td><td> $R _ { \mathrm { l i s } } + M$ </td><td>Dancing-Only</td></tr><tr><td>Song</td><td>Listener</td><td> $S + R _ { \mathrm { l i s } } + M$ </td><td>Dancing-Only</td></tr><tr><td>Song</td><td>Source</td><td> $S + R _ { \mathrm { s r c } } + M$ </td><td>Singing-and-Dancing</td></tr></table>

Table 1: Hard-compact routing across generation tasks. S, M, and R denote speech, music, and role tokens, respectively. The first four routes are used during training; Song/Source is activated only at inference as the held-out compositional configuration.

In parallel, the same binary source/listener state is encoded by a learned global role embedding and a local role token. Each textual role assignment is paired consistently with its corresponding learned role conditions throughout training and inference. Speech and vocal role therefore provide both global and local conditions, whereas music is introduced only through local tokens. When switching roles, SingDance changes only the vocal-role state; its textual and learned representations are updated together, while the roleindependent prompt content, speech and music representations, and shared conditioning pathway remain unchanged. This coordinated but compact intervention provides the basis for the held-out composition described later.

## Hard-Compact Routing and Joint Audio Injection

Diferent generation tasks require diferent subsets of the available conditions. We therefore use hard-compact routing: the route is specified by the generation task rather than inferred from audio content, and only the task-relevant speech, music, and role tokens are retained as the routed tokens. Table 1 summarizes the five configurations.

Speech examples retain speech and role tokens. Instrumental dance retains music and role tokens, making music the only time-varying audio input and providing a direct learning signal to the newly introduced music pathway. Song-based dance retains all three token groups under the listener role, because lyrics, vocal phrasing, and phrase boundaries may still inform choreographic timing even when the visible subject does not articulate the vocals. At inference, the held-out Song/Source configuration preserves the speech and music tokens and changes only the vocal-role state, updating its learned role conditions and the prompt’s vocal role field consistently.

Figure 3 details frame-wise joint audio injection, where the routed tokens form a joint key–value sequence temporally aligned with each generated latent frame. For global modulation, speech and song examples use the global speech feature together with the global role embedding, whereas instrumental examples use the global role embedding alone; music is supplied through the routed local tokens. Text remains in the backbone’s original text cross-attention, and joint audio injection updates only generated-frame tokens, leaving the separately encoded reference latent unchanged.

![](images/79069e9bb9348fbdd9da0fb240f6fe9c086d366ba46a608753d4d11aaf6eab43.jpg)  
Figure 3: Frame-wise joint audio injection. For each generated latent frame, visual tokens act as queries and the temporally aligned routed tokens serve as joint keys and values; the cross-attention output is added residually to update the visual representation.

## Staged Training for Zero-Shot Composition

SingDance adopts a staged training strategy that first establishes vocal-role control and then introduces musicconditioned body motion. This organization reflects the asymmetric supervision available to the two capabilities: speech videos provide source/listener supervision, whereas dancing-only videos provide music-conditioned body-motion supervision. Importantly, no paired singingand-dancing videos are used in either stage.

In the speech stage, the music pathway is absent. Onscreen vocal production grounds the source role, while genuine listening and reacting behavior toward an of-screen interlocutor grounds the listener role. Both roles share the same speech representation and conditioning pathway, with their assignment expressed through the prompt’s vocal role field and global/local role conditions. The model therefore learns role-dependent interpretations of a shared vocal signal before music conditioning is introduced.

The dance stage starts from the speech-stage model and introduces the music pathway using instrumental and songbased dancing-only videos. Instrumental examples retain only music and role conditions, making music the sole timevarying audio input. Song examples retain both speech and music conditions because vocal phrasing and phrase boundaries may remain relevant to choreography even when the visible subject does not articulate the vocals. During this stage, speech examples from both roles constitute 20% of the training samples to mitigate forgetting of articulation and role control.

Across the two stages, SingDance observes four configurations: Speech/Source, Speech/Listener, Instrumental/Listener, and Song/Listener. The target Song/Source configuration is never observed during training. At inference, we instantiate it by changing the vocal role field from an of-screen to an on-screen source assignment and switching the corresponding global and local role conditions from listener to source. The reference image, soundtrack, remaining prompt fields, and extracted speech and music features remain unchanged. This composition combines the sourcerole articulation learned from speech with the joint speech– music conditioning learned from dancing-only videos, without introducing a new generation module or a post-hoc lipsynchronization model.

Both stages use the backbone’s standard flow-matching objective (Lipman et al. 2023) over generated latent positions, without specialized lip-synchronization or beat-alignment losses.

## Text–Audio Classifier-Free Guidance

At inference, we use three-pass classifier-free guidance (CFG) (Ho and Salimans 2022) with separate text and audio guidance scales. Let $\mathbf { v } _ { \bar { t } , 0 } , \mathbf { v } _ { { t } , 0 }$ , and $\mathbf { v } _ { t , a }$ denote the predictions obtained with the negative prompt and null audio, the input prompt and null audio, and the input prompt with routed audio, respectively. We compute

$$
\begin{array} { r } { \widehat { \mathbf { v } } = \mathbf { v } _ { \bar { t } , 0 } + s _ { t } \left( \mathbf { v } _ { { t } , 0 } - \mathbf { v } _ { \bar { t } , 0 } \right) + s _ { a } \left( \mathbf { v } _ { { t } , a } - \mathbf { v } _ { { t } , 0 } \right) , } \end{array}\tag{1}
$$

where $s _ { t }$ and $s _ { a }$ are the text and audio guidance scales, respectively. The learned vocal-role embeddings remain active in all three passes. The textual vocal role field is part of the input prompt and is therefore included in $\mathbf { v } _ { t , 0 }$ and $\mathbf { v } _ { t , a }$

To enable this three-pass CFG, we apply condition dropout during training: text alone is dropped with probability 10%, audio alone with probability 10%, and both are dropped with probability 5%; all remaining samples retain both conditions. Text dropout removes the entire prompt, including its vocal role field, whereas the learned vocal-role embeddings are retained in all dropout cases.

## Experiments

## Experimental Setup

Data. The speech phase combines HuMoSet (Chen et al. 2025a) with a proprietary web-video collection, totaling approximately 300,000 five-second clips. Among them, approximately 12,000 clips depict genuine listening or reacting behavior toward an of-screen interlocutor, while the remainder primarily depict on-screen speaking. The dance phase combines cleaned MA-Data (Yang et al. 2025a) with a proprietary dance-video collection, yielding approximately 48,000 five-second clips: 18,000 with instrumental music and 30,000 with songs. A small fraction of song-conditioned clips exhibits detectable or uncertain lip synchronization; we remove these cases using SyncNet (Chung and Zisserman 2017) and retain the remainder as dancing-only supervision. No singing-and-dancing videos are used for training.

For dance evaluation, we curate two test sets disjoint from all training data: SingDance-50, comprising 50 face-visible vocal-song cases for singing-and-dancing, and Dance-100, comprising 100 diverse dancing-only cases. For talking, we use EMTD (Meng et al. 2025). The vocal-role intervention reuses SingDance-50 and changes only the vocal-role assignment, while keeping the reference image, soundtrack, roleindependent prompt fields, sampling seed, model checkpoint, and all other inference settings fixed.

Baselines. Open-source systems that directly generate RGB dance videos from audio remain limited, as much of the music-to-dance literature generates intermediate motion representations. We therefore select two complementary end-to-end baselines. MusicInfuser (Hong et al. 2026) directly conditions a text-to-video model on music; its text– audio-to-video (TA2V) formulation ofers a representative comparison for music-responsive generation even without reference-image conditioning or explicit vocal articulation. Wan-S2V (Gao et al. 2025) accepts text, image, and audio inputs (TIA2V), shares the Wan video-model family with SingDance, and provides a particularly relevant speechconditioned human-animation baseline; our speech pathway is also inspired by its frame-wise audio injection. Its formulation assigns the input vocals to the visible subject and uses speech-oriented rather than music-specific audio conditioning. Together, they provide references for music-conditioned dancing and speech-conditioned human animation. We evaluate both on both dance test sets using their supported input interfaces and recommended prompt formats.

Implementation Details. All SingDance results are produced by a single model initialized from Wan2.2-TI2V-5B (Wan Team et al. 2025). We train on 64 NVIDIA A100 GPUs with an efective batch size of 64, using AdamW (Loshchilov and Hutter 2019) with a learning rate of $1 \times 1 0 ^ { - 5 }$ . The speech and dance phases are each trained for eight epochs at 480p, after which the dance phase is further fine-tuned for four epochs, approximately 3K optimization steps, at 704 × 1280. At inference, talking videos follow the corresponding EMTD input resolution, whereas dance videos are generated at 704 × 1280. We generate 121 frames at 24 FPS using 50 flow-matching sampling steps. Talking uses two-pass CFG with scale 5, while the two dance tasks use the three-pass CFG in Equation 1, with text and audio scales of 5 and 4, respectively.

Evaluation Metrics. We evaluate reference fidelity, visual quality, audio–visual lip synchronization, motion–beat alignment, and human preference.

Reference and visual quality. Following VBench (Huang et al. 2024a) and its I2V extension in VBench++ (Huang et al. 2024b), we report DINO-based Subject and DreamSim-based Background fidelity to the reference image, interpolation-based Motion Smoothness, LAION Aesthetic Quality, and MUSIQ Imaging Quality.

Lip synchronization. We use the oficial SyncNet evaluator (Chung and Zisserman 2017) to report LSE-C and LSE-D. Higher LSE-C and lower LSE-D indicate stronger synchronization for the vocal-source role; the desired directions are reversed for the listener role, where synchronization with external vocals should be suppressed. Videos without a valid face track are excluded.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="2">Reference Fidelity</td><td colspan="2">Motion-Beat Alignment</td><td colspan="3">Visual Quality</td><td colspan="2">Human Preference</td></tr><tr><td>Subject ↑</td><td>Background ↑</td><td>BeatAlign ↑</td><td>MBCR↑</td><td>Aesthetic ↑</td><td>Imaging ↑</td><td>Smoothness ↑</td><td>Rhythm ↑</td><td>Visual ↑</td></tr><tr><td colspan="10">Singing-and-Dancing</td></tr><tr><td>GT</td><td></td><td>0.9673</td><td>0.9735</td><td>0.2795</td><td>0.2922</td><td>0.5720</td><td>0.6553</td><td>0.9858</td><td></td><td></td></tr><tr><td>MusicInfuser†</td><td>TA2V</td><td></td><td></td><td>0.2356</td><td>0.2337</td><td>0.5232</td><td>0.6172</td><td>0.9927</td><td></td><td></td></tr><tr><td>Wan-S2V</td><td>TIA2V</td><td>0.9565</td><td>0.9633</td><td>0.2640</td><td>0.1983</td><td>0.5834</td><td>0.6538</td><td>0.9860</td><td>38.67</td><td>45.33</td></tr><tr><td>SingDance</td><td>TIA2V</td><td>0.9682</td><td>0.9772</td><td>0.2723</td><td>0.2558</td><td>0.5846</td><td>0.6688</td><td>0.9899</td><td>61.33</td><td>54.67</td></tr><tr><td colspan="10">Dancing-Only</td></tr><tr><td>GT</td><td></td><td>0.9648</td><td>0.9705</td><td>0.3051</td><td>0.3027</td><td>0.5738</td><td>0.6667</td><td>0.9880</td><td></td><td></td></tr><tr><td>MusicInfuser†</td><td>TA2V</td><td></td><td></td><td>0.2872</td><td>0.2661</td><td>0.5258</td><td>0.6522</td><td>0.9948</td><td></td><td></td></tr><tr><td>Wan-S2V</td><td>TIA2V</td><td>0.9415</td><td>0.9493</td><td>0.2467</td><td>0.2180</td><td>0.5776</td><td>0.6570</td><td>0.9867</td><td>32.00</td><td>44.33</td></tr><tr><td>SingDance</td><td>TIA2V</td><td>0.9705</td><td>0.9766</td><td>0.3002</td><td>0.2632</td><td>0.5718</td><td>0.6738</td><td>0.9921</td><td>68.00</td><td>55.67</td></tr></table>

Table 2: Quantitative results on SingDance-50 and Dance-100. Human scores are preference rates (%). Best and second-best generated results are bold and underlined; GT is shown only as a reference. <sup>†</sup> MusicInfuser does not accept a reference image, so its reference-fidelity scores are not applicable.

Motion–beat alignment. Kinematic beats are identified as strict local minima of a Gaussian-smoothed, scalenormalized motion-speed curve computed from 65 body, foot, and hand landmarks extracted by DWPose (Yang et al. 2023). BeatAlign (Li et al. 2021) measures the Gaussianweighted temporal proximity from each kinematic beat to its nearest musical beat, with $\sigma = 0 . 0 5$ seconds. Because this motion-centered score can remain high when only a few welltimed movements are generated, we follow the coverage formulation of the Audio Beat Hit Score in TMD-Bench (Yang et al. 2026b) and report an adapted Music-Beat Coverage Rate (MBCR). MBCR measures the fraction of musical beats covered by at least one kinematic beat within a 100-ms tolerance. The two metrics therefore characterize beat precision and beat coverage, respectively, and are averaged over the common pose-valid subset within each task.

Human evaluation. We conduct a blind pairwise study between SingDance and Wan-S2V. MusicInfuser is excluded because it does not condition on the reference image, making its identity apparent in a reference-conditioned comparison. We randomly sample 20 source-role song inputs from SingDance-50 and 20 instrumental inputs from Dance-100. The instrumental restriction is applied only to the dancingonly subset, preventing Wan-S2V’s default vocal articulation from revealing model identity. Fifteen participants evaluate all 40 matched-input, 24-FPS pairs with randomized and balanced A/B ordering, selecting the better result for body motion–music rhythm alignment and overall visual quality without ties. We report preference rates over 300 votes per task and criterion.

## Main Results

Table 2 evaluates compositional zero-shot singing-anddancing on SingDance-50 and directly supervised dancingonly generation on Dance-100. It compares the dimensions shared by all applicable methods: reference fidelity, motion– beat alignment, visual quality, and human preference. We reserve vocal articulation for the paired analysis in the following subsection, where its desired role is explicitly controlled. The routed music token is separately ablated in Table 5.

<table><tr><td>Role</td><td>Method</td><td>LSE-C</td><td>LSE-D</td><td>BeatAlign ↑</td><td>MBCR↑</td></tr><tr><td>Source</td><td>Wan-S2V</td><td>4.2809</td><td>9.4462</td><td>0.2640</td><td>0.1983</td></tr><tr><td></td><td>SingDance</td><td>4.9672</td><td>8.9363</td><td>0.2723</td><td>0.2558</td></tr><tr><td>Listener</td><td>Wan-S2V</td><td>4.1676</td><td>9.5386</td><td>0.2480</td><td>0.2065</td></tr><tr><td></td><td>SingDance</td><td>1.1768</td><td>12.4059</td><td>0.2704</td><td>0.2787</td></tr></table>

Table 3: Paired vocal-role switching on SingDance-50. Source and listener correspond to Lip-Sync ON and OFF; LSE-C and LSE-D therefore have opposite desired directions.

![](images/bb15d807f9c881b73cea4e511f99d41c895a130795ba7ad46f59d4fdc996bbbe.jpg)  
Figure 4: Qualitative paired vocal-role switching on two SingDance examples. Each Lip-on/Lip-of pair shares all role-independent inputs and inference settings; only the vocal role changes.

Singing-and-dancing. Despite never observing paired singing-and-dancing videos during training, SingDance generates rhythmically coordinated performances while preserving the reference appearance and strong visual quality. Its advantage over speech-oriented generation demonstrates that music-specific conditioning complements text-guided choreography and vocal articulation. Human raters also prefer

![](images/506e1e10f5b1a2a63f157128484570198b7317f79aef0147a48d95cb00cdc1d8.jpg)  
Figure 5: Qualitative vocal-role control with the same reference image and speech audio. SingDance produces synchronized articulation for the source role and listening/reacting behavior for the listener role; videos are provided in the supplementary material.

SingDance for rhythmic coordination and overall visual quality.

Dancing-only. On Dance-100, SingDance continues to produce music-responsive body motion with strong reference preservation and visual fidelity. Its balanced performance across rhythmic coordination and generation quality shows that the unified model retains its directly supervised dancingonly capability while supporting compositional zero-shot singing-and-dancing. This advantage is also reflected in the human preference results.

## Paired Vocal-Role Switching

Table 2 establishes generation quality and motion–beat alignment on the two task-specific test sets, but these metrics alone do not verify their defining behavioral distinction: whether the visible subject produces or merely hears the vocals. We therefore complement the main comparison with a paired intervention on SingDance-50. The source rows reuse the corresponding singing-and-dancing outputs from Table 2. For each listener counterpart, the reference image, song, roleindependent prompt fields, checkpoint, sampling seed, and inference configuration remain fixed, while only the vocalrole state changes; its learned conditions and the prompt’s vocal role field are updated consistently. We apply the same textual switch to Wan-S2V as a prompt-only control, since it does not expose learned vocal-role conditions.

The prompt-only switch leaves Wan-S2V’s articulation behavior largely unchanged, showing that text alone provides weak control over whether the visible subject owns the vocals. In contrast, SingDance consistently changes from vocal articulation to a listener role while retaining musicaligned body motion and stable visual quality. Figure 4 shows two representative paired examples. Together with Table 2, this intervention verifies that the same model can realize the defining behavioral diference between singing-and-dancing and dancing-only generation.

<table><tr><td>Method</td><td>Params. ↓</td><td>LSE-C ↑</td><td>LSE-D</td><td>Subj. ↑ Aes. ↑ Img. ↑</td></tr><tr><td>Hallo3</td><td>14.55</td><td>5.352</td><td>9.828</td><td>0.9746 0.511 0.677</td></tr><tr><td>Wan-S2V</td><td>16.30</td><td>6.999</td><td>8.292</td><td>0.9812 0.580 0.698</td></tr><tr><td>HY-Avatar</td><td>12.99</td><td>7.210</td><td>8.494</td><td>0.9791 0.535 0.661</td></tr><tr><td>InfiniteTalk</td><td>18.88</td><td>8.535</td><td>7.108</td><td>0.9868 0.545 0.680</td></tr><tr><td>SingDance</td><td>5.63</td><td>7.995</td><td>7.551</td><td>0.9859 0.593 0.722</td></tr></table>

Table 4: Speech-driven vocal-articulation results on EMTD. Params. denotes generation-time parameters in billions; best and second-best results are bold and underlined.

<table><tr><td>Test Set</td><td></td><td>Variant BeatAlign ↑</td><td>MBCR↑</td><td>Aes. ↑ Img. ↑</td></tr><tr><td>SingDance-50</td><td>w/o M Full</td><td>0.2462 0.2723</td><td>0.2386 0.2558</td><td>0.5864 0.6598 0.5846 0.6688</td></tr><tr><td>Dance-100</td><td></td><td></td><td>0.2457</td><td></td></tr><tr><td></td><td>w/o M</td><td>0.2853</td><td></td><td>0.5687 0.6734</td></tr><tr><td></td><td>Full</td><td>0.3002</td><td>0.2632</td><td>0.5718 0.6738</td></tr></table>

Table 5: Ablation of the routed music token on SingDance-50 and Dance-100. Aes. and Img. denote Aesthetic and Imaging Quality; bold values indicate the better variant on each test set.

## Vocal Articulation Capability

Reliable source-role articulation is a prerequisite for compositional zero-shot singing-and-dancing. Table 4 compares SingDance with representative speech-driven systems, including Hallo3 (Cui et al. 2024), Wan2.2-S2V (Gao et al. 2025), HY-Avatar (Chen et al. 2025b), and InfiniteTalk (Yang et al. 2025b), on EMTD. Using roughly one-third of InfiniteTalk’s generation-time parameters, SingDance achieves highly competitive lip synchronization together with the strongest aesthetic and imaging quality among the evaluated systems. This parameter-eficient articulation capability provides the foundation for compositional zero-shot singingand-dancing, where synchronization must remain reliable alongside concurrent body motion rather than the relatively constrained motion of conventional talking clips.

## Ablation Study

Table 5 isolates the contribution of the routed music token. Removing it consistently weakens motion–beat alignment on both test sets, while aesthetic and imaging quality remain broadly stable. These results show that the music pathway contributes efective rhythmic conditioning without compromising visual quality.

## Limitations

SingDance targets short, single-person clips with a binary clip-level vocal role. It does not yet model within-clip role transitions, duets, multiple vocal roles, or automatic role attribution. Extending role-aware composition to longer, multiperson performances remains future work.

## Conclusion

We introduced SingDance, a role-aware video generation framework that separates vocal content from the visible subject’s role while coordinating text-guided action and musicconditioned body motion. Hard-compact routing and framewise joint audio injection compose speech, music, and role conditions within a shared video difusion backbone. From asymmetric speaking/listening and dancing-only supervision, SingDance realizes compositional zero-shot singingand-dancing in the held-out Song/Source configuration. Experiments validate music-aligned dance, paired vocal-role control, and parameter-eficient lip synchronization.

## References

Baevski, A.; Zhou, H.; Mohamed, A.; and Auli, M. 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. In Advances in Neural Information Processing Systems.

Chen, L.; Ma, T.; Liu, J.; Li, B.; Chen, Z.; Liu, L.; He, X.; Li, G.; He, Q.; and Wu, Z. 2025a. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning. arXiv:2509.08519.

Chen, Y.; Liang, S.; Zhou, Z.; Huang, Z.; Ma, Y.; Tang, J.; Lin, Q.; Zhou, Y.; and Lu, Q. 2025b. HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters. arXiv:2505.20156.

Chen, Z.; Xu, H.; Song, G.; Xie, Y.; Zhang, C.; Chen, X.; Wang, C.; Chang, D.; and Luo, L. 2025c. X-Dancer: Expressive Music to Human Dance Video Generation. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision.

Chu, X.; Liu, R.; Huang, Y.; Liu, Y.; Peng, Y.; and Zheng, B. 2026. UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Chung, J. S.; and Zisserman, A. 2017. Out of Time: Automated Lip Sync in the Wild. In Computer Vision–ACCV 2016 Workshops, 251–263. Springer.

Cui, J.; Li, H.; Zhan, Y.; Shang, H.; Cheng, K.; Ma, Y.; Mu, S.; Zhou, H.; Wang, J.; and Zhu, S. 2024. Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Difusion Transformer. arXiv:2412.00733.

Feng, T.; Xie, Y.; Guan, X.; Song, J.; Liu, Z.; Ma, F.; and Yu, F. 2025. UniSync: A Unified Framework for Audio-Visual Synchronization. In 2025 IEEE International Conference on Multimedia and Expo (ICME), 1–6. IEEE.

Gan, Q.; Yang, R.; Zhu, J.; Xue, S.; and Hoi, S. 2025. OmniAvatar: Eficient Audio-Driven Avatar Video Generation with Adaptive Body Animation. arXiv:2506.18866.

Gao, X.; Hu, L.; Hu, S.; Huang, M.; Ji, C.; Meng, D.; Qi, J.; Qiao, P.; Shen, Z.; Song, Y.; et al. 2025. Wan-S2V: Audio-Driven Cinematic Video Generation. arXiv:2508.18621.

Ho, J.; and Salimans, T. 2022. Classifier-Free Difusion Guidance. arXiv:2207.12598.

Hong, S.; Kemelmacher-Shlizerman, I.; Curless, B.; and Seitz, S. M. 2026. MusicInfuser: Making Video Difusion

Listen and Dance. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Huang, M.; Zhang, P.; Hu, L.; Wang, G.; Zhang, R.; Lu, Y.; Cheng, G.; and Zhang, B. 2026. Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation. arXiv:2607.09581.

Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang, Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024a. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

Huang, Z.; Zhang, F.; Xu, X.; He, Y.; Yu, J.; Dong, Z.; Ma, Q.; Chanpaisit, N.; Si, C.; Jiang, Y.; Wang, Y.; Chen, X.; Chen, Y.-C.; Wang, L.; Lin, D.; Qiao, Y.; and Liu, Z. 2024b. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. arXiv:2411.13503.

Li, R.; Yang, S.; Ross, D. A.; and Kanazawa, A. 2021. AI Choreographer: Music Conditioned 3D Dance Generation with AIST++. arXiv:2101.08779.

Li, R.; Zhao, J.; Zhang, Y.; Su, M.; Ren, Z.; Zhang, H.; Tang, Y.; and Li, X. 2023. FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In International Conference on Learning Representations.

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.

Meituan LongCat Team. 2026. LongCat-Video-Avatar 1.5 Technical Report. arXiv:2605.26486.

Meng, R.; Wang, Y.; Wu, W.; Zheng, R.; Li, Y.; and Ma, C. 2026. EchoMimicV3: 1.3B Parameters Are All You Need for Unified Multi-Modal and Multi-Task Human Animation. In Proceedings of the AAAI Conference on Artificial Intelligence.

Meng, R.; Zhang, X.; Li, Y.; and Ma, C. 2025. EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Peng, Z.; Chen, Y.; Ma, Y.; Zhang, G.; Sun, Z.; Zhou, Z.; Zhang, Y.; Zhou, Z.; Fan, Z.; Liu, H.; Zhou, Y.; Lu, Q.; and He, J. 2026. ActAvatar: Temporally-Aware Precise Action Control for Talking Avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Peng, Z.; Fan, Y.; Wu, H.; Wang, X.; Liu, H.; He, J.; and Fan, Z. 2025. DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21055–21064.

Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; and Sutskever, I. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings of the International Conference on Machine Learning.

Siyao, L.; Yu, W.; Gu, T.; Lin, C.; Wang, Q.; Qian, C.; Loy, C. C.; and Liu, Z. 2022. Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Sun, Z.; Peng, Z.; Ma, Y.; Chen, Y.; Zhou, Z.; Zhou, Z.; Zhang, G.; Zhang, Y.; Zhou, Y.; Lu, Q.; and Liu, Y.-J. 2026. StreamAvatar: Streaming Difusion Models for Real-Time Interactive Human Avatars. arXiv:2512.22065.

Tseng, J.; Castellon, R.; and Liu, C. K. 2022. EDGE: Editable Dance Generation From Music. arXiv:2211.10658.

Wan Team; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv:2503.20314.

Wang, L.; Zhu, Y.; Ge, Z.; Zheng, Y.; Zhang, L.; Hu, T.; Qin, S.; Luo, M.; Zhang, J.; Chen, X.; et al. 2026. FlowAct-R1: Towards Interactive Humanoid Video Generation. arXiv:2601.10103.

Weng, Y.; Wang, H.; Yu, X.; Wu, X.; Xu, H.; He, S.; and Du, J. 2026. Beyond Monologue: Interactive Talking-Listening Avatar Generation with Conversational Audio Context-Aware Kernels. arXiv:2604.10367.

Xie, Y.; Feng, T.; Zhang, X.; Luo, X.; Guo, Z.; Yu, W.; Chang, H.; Ma, F.; and Yu, F. R. 2025. PointTalk: Audio-Driven Dynamic Lip Point Cloud for 3D Gaussian-Based Talking Head Synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 8753–8761.

Yang, K.; Zhu, J.; Tang, X.; Peng, Z.; Zhang, X.; Chen, C.; Wang, P.; Wu, J.; Chu, X.; Liu, H.; and He, J. 2026a. OmniDance: Multimodal Driven Dance Video Generation with Large-scale Internet Data. arXiv:2606.30019.

Yang, K.; Zhu, J.; Tang, X.; Peng, Z.; Zhang, X.; Wang, P.; Wu, J.; Chu, X.; Liu, H.; and He, J. 2025a. MACE-Dance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation. arXiv:2512.18181.

Yang, S.; Kong, Z.; Gao, F.; Cheng, M.; Liu, X.; Zhang, Y.; Kang, Z.; Luo, W.; Cai, X.; He, R.; and Wei, X. 2025b. InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing. arXiv:2508.14033.

Yang, X.; Zhang, M.; Pan, C.; Huang, N.; Yang, Y.; Zhuo, F.; Zhou, P.; Zhou, J.; Shan, S.; Yang, S.; Yang, M.; You, Y.; and Zhao, Z. 2026b. TMD-Bench: A Multi-Level Evaluation Paradigm for Music–Dance Co-Generation. arXiv:2605.01809.

Yang, Z.; Zeng, A.; Yuan, C.; and Li, Y. 2023. Efective Whole-Body Pose Estimation with Two-Stage Distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4210–4220.

Zeng, A.; Yang, C.; Ge, C.; Zhang, E.; Xu, G.; Lin, G.; Gu,G.; Pi, J.; Li, L.; Shi, M.; et al. 2026. LPM 1.0: Video-basedCharacter Performance Model. arXiv:2604.07823.

Zhang, Y.; Shen, K.; and Li, Y. 2026. EmbodiedHead: Real-Time Listening and Speaking Avatar for Conversational Agents. arXiv:2604.17211.

Zhu, H.; Zhou, Y.; Chen, H.; Yu, J.; Ma, Z.; Gu, R.; Luo, Y.; Tan, W.; and Chen, X. 2025. MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization. arXiv:2501.01108.