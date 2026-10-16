# Hear, Localize, and Reason: Spatially Aware Scene Understanding for Audio-Visual LLMs

Kim Sung-Bin<sup>1</sup> Lee Jung-Mok<sup>2</sup> Jinwoo Jung<sup>2</sup> Oh Hyun-Bin<sup>1</sup> Hyeonggon Ryu<sup>3</sup> David Harwath<sup>4</sup> Tae-Hyun Oh<sup>2</sup>

<sup>1</sup>POSTECH <sup>2</sup>KAIST <sup>3</sup>Hankuk University of Foreign Studies <sup>4</sup>The University of Texas at Austin

![](images/d19c80a59423746933692e6011eafc8bc81d82882a03fd1dab32fe44be2f4d6f.jpg)  
Figure 1: Overview of HLR-AVSceneQA. We introduce HLR-AVSceneQA, a comprehensive benchmark for spatial audio-visual scene understanding organized around a hear-localize-reason hierarchy. For a queried sound source, the model recognizes what is heard, localizes where it originates in egocentric and allocentric views, and uses the localized source together with visual scene evidence to reason about how it relates to the surrounding scene.

## Abstract

Audio-visual large language models (AV-LLMs) have made strong progress in multimodal understanding, yet most treat audio as monaural semantic content, limiting their ability to reason about where sounds originate and how they relate to the visual scene. Recent spatial audio-visual studies address parts of this problem, but often focus on specific abilities, such as spatial correspondence or direction and distance reasoning. In this work, we introduce HLR-AVSceneQA, a comprehensive benchmark with a hear-localize-reason hierarchy that evaluates whether models can recognize what is heard, localize sound sources in egocentric and allocentric views, and use these localized sources to reason about how they relate to the visual scene. We further introduce HLR-LLM, which augments a strong audio-visual foundation model with a binaural spatial audio branch alongside its monaural semantic audio pathway. We train HLR-LLM with a curriculum that incorporates final-stage chainof-thought supervision for multi-step spatial reasoning. Experiments show that existing AV-LLMs remain limited in spatial grounding and relational reasoning, while HLR-LLM substantially improves spatial audio-visual scene understanding without compromising semantic audio-visual perception. Code and benchmark data are available at https://hlrllm.github.io/.

## 1 Introduction

Spatial audio and vision provide complementary cues for understanding a scene. We may hear someone calling from another room, infer that a ringing alarm clock is hidden below the bed by combining its direction with the visible layout, or decide where to move to get closer to a sound by relating auditory distance cues to nearby objects. Such everyday situations show that audio-visual scene understanding requires more than recognizing a sound event or matching it to a visible object. Accordingly, a spatially aware agent must infer where the sound originates, whether it is visible, why it may be hidden, and how its relation to the scene changes as the agent moves.

Recently, audio-visual large language models (AV-LLMs) [1, 2, 3, 4, 5] have shown strong progress in diverse multimodal understanding tasks, such as audio-visual question answering [6, 7], temporal grounding [8], and multimodal reasoning. However, most existing AV-LLMs treat audio primarily as monaural semantic content, focusing on what is heard rather than where it is heard. As a result, they struggle to reason about auditory sources in the physical scene, especially when a source is outside the camera view, occluded by objects, located in another room, or spatially ambiguous among multiple candidates. Spatial audio, such as binaural or multi-channel recordings, offers the direction and distance cues needed to overcome this limitation and support spatially grounded scene understanding.

Despite this growing need for spatially grounded audio-visual understanding, two key challenges remain. First, existing spatial audio-(visual) benchmarks typically focus on specific subproblems, such as audio-only [9, 10, 11] or speech-centric settings [12], egocentric or allocentric direction and distance reasoning [13], or precise RGB-D-based 3D grounding [14]. Existing evaluations therefore remain fragmented across modalities, assumptions, and task formats, making it difficult to assess a single AV-LLM’s ability to jointly handle semantic understanding, spatial grounding, and relational reasoning. Second, although spatial audio-visual tasks require multi-step reasoning, many current training settings rely on direct-answer supervision [15, 14]. This gives models insufficient structure for decomposing complex spatial queries into interpretable intermediate steps that connect auditory semantics, spatial cues, and visual context.

In this work, we introduce HLR-AVSceneQA, a comprehensive benchmark for spatial audio-visual scene understanding. As shown in Fig. 1, HLR-AVSceneQA follows a hear-localize-reason hierarchy: spatial hearing evaluates what is heard, spatial grounding evaluates source localization in egocentric and allocentric views, and spatial relational reasoning evaluates how localized sound sources relate to the visual scene. By treating localized sound-source information not only as a prediction target but also as spatial evidence for downstream reasoning, HLR-AVSceneQA enables systematic evaluation of spatial grounding and scene-level reasoning. Unlike panoramic-input settings [15], HLR-AVSceneQA uses normal-field-of-view observations, requiring models to reason about visible, occluded, and off-screen sources from partial visual evidence. We construct the benchmark with a simulator-based pipeline [16], producing diverse, controllable spatial audio-visual scenes with reliable annotations.

We further introduce HLR-LLM, a spatially aware AV-LLM that extends audio-visual semantic understanding with explicit spatial audio modeling. Built on an existing AV-LLM [4] and augmented with a spatial audio branch, HLR-LLM improves spatial grounding and reasoning while preserving strong semantic perception. To address the multi-step nature of spatial audio-visual reasoning, we incorporate chain-of-thought [17, 18] supervision in the final stage of curriculum training, encouraging the model to connect auditory, spatial, and visual cues rather than directly predicting an answer.

Experiments show that existing AV-LLMs, including commercial models that rely on monaural audio, struggle with spatial grounding and reasoning in our benchmark. In contrast, HLR-LLM preserves strong semantic hearing ability, including sound event and speech recognition, while substantially improving spatial grounding and relational reasoning. Transfer experiments on real-world spatial audio-visual recordings [19] further suggest that pretraining on HLR-AVSceneQA provides useful spatial priors for adaptation to real-world data. Our contributions are summarized as follows:

• We introduce HLR-AVSceneQA, a comprehensive benchmark that evaluates spatial hearing and grounding both directly and as intermediate evidence for downstream spatial relational reasoning.

• We propose HLR-LLM, a spatially aware AV-LLM that augments a strong foundation model with a spatial audio pathway to improve grounding and reasoning while preserving semantic perception.

• We demonstrate that final-stage chain-of-thought curriculum training improves multi-step spatial reasoning by connecting auditory, spatial, and visual cues through intermediate reasoning steps.

Table 1: Evaluation scope of spatial audio and audio-visual benchmarks. We compare whether each benchmark explicitly evaluates key spatial audio-visual capabilities. A+V denotes audiovisual input; Multi-source denotes simultaneous sound sources; Far-field ASR and Sound event denote speech recognition from far-field audio and non-speech sound recognition; Ego./Allo. denote egocentric/allocentric reasoning; and Corr. denotes audio-visual correspondence. Cross-room, Hidden Source, Proximity, and AV Caption denote reasoning about across-room sources, invisible or occluded sources, nearest-object relations, and open-ended audio-visual captioning, respectively.
<table><tr><td>Benchmark</td><td>A+V</td><td>Multi- source</td><td>Far-field ASR</td><td>Sound event</td><td>Ego.</td><td>Allo.</td><td>AV Corr.</td><td>Cross- room</td><td>Hidden Source</td><td>Proximity</td><td>AV Caption</td></tr><tr><td>BAT [9]</td><td>x</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SpatialWhisper [12]</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Hear You Are [15]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SAVVY-Bench [13]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SpatialSceneQA [14]</td><td>√</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>HLR-AVSceneQA</td><td>J</td><td>1</td><td>√</td><td>J</td><td>J</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2 Related work

Spatial audio understanding. Spatial audio provides geometric cues about sound sources beyond monaural semantic content. While monaural audio understanding mainly targets semantic tasks, such as automatic speech recognition [20, 21], sound event recognition [22, 23, 24], and audio captioning [25, 26, 27], spatial audio formats, such as binaural audio and first-order ambisonics (FOA), capture direction- and distance-related cues. Early work on spatial audio explores sound event localization and detection (SELD), where models jointly identify active sound events and estimate the direction of arrival [28, 29, 30]. More recent work connects spatial audio with language models for sound localization and reasoning [9, 10, 31], learns spatially aware audio-text representations through contrastive objectives [11, 32], and adapts FOA inputs for far-field speech understanding [12, 33]. These works highlight the value of spatial hearing, but remain largely audio-only or speech-centric, leaving open how spatial auditory cues should be grounded in and reasoned over a visual scene.

Spatial audio-visual benchmarks and tasks. General audio-visual benchmarks [34, 35, 36, 37, 38] evaluate broad multimodal abilities, but do not focus on spatial sound-source understanding. Recent benchmarks have begun to address specific spatial capabilities: SAVVY-Bench [13] evaluates dynamic egocentric/allocentric direction and distance reasoning, SpatialSceneQA [14] focuses on RGB-D and FOA-based 3D grounding, and Hear You Are [15] studies sound-object correspondence with panoramic images. As summarized in Table 1, these benchmarks cover complementary aspects of spatial audio-visual understanding, but focus on different subsets of capabilities. HLR-AVSceneQA instead provides a unified hear-localize-reason evaluation that spans spatial hearing, grounding, multi-source scenes, and scene-level relational reasoning, including whether localized sound-source information can support broader reasoning about the surrounding scene. It further uses normal-fieldof-view egocentric observations, requiring models to reason about visible, off-screen, and hidden sound sources from partial visual observations.

Spatial audio-visual large language models (AV-LLMs). General AV-LLMs [1, 2, 3, 4, 5] support video understanding, temporal localization, and audio-visual question answering, but mostly rely on monaural audio. Spatially aware AV-LLMs add spatial auditory cues to visual inputs. Hear You Are LLM [15] uses spatial audio and panoramic images, SAVVY [13] builds dynamic spatial maps with a training-free pipeline, and JAEGER [14] integrates RGB-D observations with FOA audio for 3D grounding and reasoning. While promising, these systems are often tailored to specific tasks, such as direction/distance QA or precise 3D grounding, and do not fully inherit the broad semantic and reasoning abilities of general AV-LLMs. They are also commonly trained with direct-answer supervision despite the multi-step nature of spatial audio-visual reasoning [15, 14]. HLR-LLM addresses these gaps by extending a strong AV-LLM with a spatial audio branch and chain-of-thought supervision [17, 18], improving spatial reasoning while preserving general audio-visual capabilities.

## 3 HLR-AVSceneQA: Spatial audio-visual scene understanding benchmark

We introduce HLR-AVSceneQA, a comprehensive benchmark for spatial audio-visual scene understanding. HLR-AVSceneQA follows a hear-localize-reason hierarchy that evaluates three capabilities: spatial hearing for recognizing what is heard, spatial grounding for localizing sound sources from egocentric and allocentric viewpoints, and spatial relational reasoning for understanding how localized sources relate to visible, hidden, nearby, or off-screen objects in the scene (Fig. 1). This hierarchy spans 12 tasks, as summarized in Table 2. HLR-AVSceneQA contains 175,061 training and 21,808 test QA pairs, including 53,279 training and 4,869 test pairs from scenes with multiple simultaneously active sound sources. We also provide 94,519 CoT-augmented training pairs for multi-step spatial reasoning. Because collecting real-world spatial audio-visual data with accurate annotations is challenging, we construct the benchmark with a two-stage simulator-based pipeline: Stage I synthesizes spatial audio-visual scenes with structured metadata, and Stage II generates QA pairs from this metadata.

Table 2: Representative QA examples in HLR-AVSceneQA. Most tasks use controlled answer formats, such as multiple choice or discretized bins; simplified prompts and answers are shown for clarity. The final row shows one CoT supervision example used for rationale-augmented training.
<table><tr><td>Task family</td><td>Subtask</td><td>Representative example</td></tr><tr><td rowspan="2">Spatial hearing</td><td>Sound event</td><td>Q: What sound do you hear? A: Hair dryer drying.</td></tr><tr><td>Far-field ASR</td><td>Q: Write the exact phrase spoken. A: &quot;Yes it is suppressed&quot;.</td></tr><tr><td rowspan="2">Egocentric grounding</td><td>Direction</td><td>Q: From the current viewpoint, which direction is the sound coming from? A:front-left.</td></tr><tr><td>Distance</td><td>Q: From the current viewpoint, how far is the sound source? A:2.5-3 m.</td></tr><tr><td rowspan="2">Allocentric grounding</td><td>Direction</td><td>Q: If you stand at the sofa and face the TV, which direction is the sound coming from? A:back-left.</td></tr><tr><td>Distance</td><td>Q: If you stand at the sofa and face the TV, how far is the sound source? A:0-1.0 m.</td></tr><tr><td rowspan="4">Spatial relational reasoning</td><td>Visibility Hidden-source</td><td>Q: From the current viewpoint, is the sound source visible? A: no. Q: Given the sound, where is its source located? A:hidden behind the armchair. Q: Which object is spatially aligned with the sound? A: man in front-right.</td></tr><tr><td>Correspondence Loudnēss change Proximity</td><td>Q: If you move to the sofa, does the sound get louder or quieter? A: quieter. Q: Which object among the options is closest to the sound source? A: armchair. Q: Describe the sound event(s), including direction, visibility, and hidden location if</td></tr><tr><td>Scene captioning</td><td>applicable. A:A hair dryer is drying from the front-left direction. It is</td></tr><tr><td></td><td>not visible and is likely occluded by the armchair.</td></tr><tr><td colspan="2">CoT supervision example provided for allocentric grounding and spatial relational reasoning; hidden-source example shown</td><td>Q: Given the sound, where is its source located? A:The sound event is a hair dryer drying, arriving from the front-left direction and 2.7 meters away. The hair dryer is not</td></tr></table>

## 3.1 Stage I: Spatial audio-visual scene synthesis

We synthesize each sample in an indoor 3D scene from an agent’s egocentric viewpoint, where the agent pose defines both the RGB camera and the acoustic receiver. We populate the scene with diverse candidate objects and designate up to two objects as active sound sources. Each sample includes a synchronized RGB observation, binaural audio, object layouts, source assignments, and geometric metadata for QA generation. Examples of rendered scenes and metadata are shown in Fig. 2.

Scene and acoustic simulator. We use SoundSpaces 2.0 [16], which renders geometry-aware room acoustics and spatializes dry monaural clips into binaural audio. For 3D environments, we use HM3DSem [39], which provides 216 indoor spaces and 3,100 rooms with dense object- and room-level semantics. These semantics support realistic rendering, layout-aware QA generation, and filtering of physically implausible placements. We use 145 scenes for training and 36 for testing.

Audio source curation. We use non-speech sounds from VGGSound [40], which contains over 200,000 in-the-wild clips across 310+ classes. Since some classes, such as sea waves or people cheering, are not naturally grounded to a single indoor object, we manually exclude them. We further use ImageBind-based text-audio similarity to filter out likely mislabeled clips [41]. For speech audio, we use LibriSpeech [42], which provides clean utterance-level transcripts for far-field speech recognition. We follow the official train/test splits of both VGGSound and LibriSpeech.

Visual object curation. We build a 3D mesh pool of sounding objects to insert into scenes. We collect available meshes from Objaverse-XL [43], and supplement categories with insufficient coverage using off-the-shelf 2D/3D generation models [44, 45]. All assets are manually checked to remove low-quality meshes. We keep approximately 81 meshes per category for training and 35 for testing.

![](images/f7b89ce29072be36b901da5349b84bdf421ab93988418a8315cf123be298f0bb.jpg)  
(b) Ambiguous sound source among same-category instances  
Figure 2: Examples of rendered scenes and metadata. Our dataset covers diverse scenarios that require spatially aware audio-visual reasoning, including (a) sound sources occluded by objects and (b) ambiguous sound sources among same-category instances. Each sample is summarized with structured scene metadata. Yellow denotes existing scene objects, pink denotes inserted objects, blue denotes sounding objects, and green denotes the agent.

Spatial audio-visual scene assembly and structured metadata. Given a room from HM3DSem, we insert 2 to 6 candidate object meshes based on available free space and room scale. We activate up to two sound sources and attach them to selected objects. Some audio-object pairs are semantically matched, while others are intentionally spatially aligned but semantically mismatched. We further include sources that are fully visible, occluded by an object, placed in an adjacent room, or outside the agent’s field of view, introducing realistic ambiguity and preventing reliance on category matching alone. The final RGB image is rendered from the agent viewpoint with a 90<sup>◦</sup> horizontal field of view at 1280×720 resolution, paired with binaural audio. For each sample, we export structured metadata, including agent pose, object identities and 3D coordinates, source-object assignments, visibility state, occlusion type, and pairwise geometry, to generate QA pairs with accurate labels.

## 3.2 Stage II: Benchmark task design and QA construction

Given each rendered scene and its metadata, we generate QA pairs for three task families. To improve linguistic diversity while keeping evaluation stable, we vary question templates but preserve controlled answer formats. Most tasks use constrained outputs, such as multiple choice or discretized bins, while speech transcription and scene captioning are open-ended. Representative examples are shown in Table 2, and additional construction and quality-control details are provided in Appendix C.

Task family 1: Spatial hearing. This family evaluates semantic perception from spatial audio through sound event recognition and far-field speech recognition, which identify audible event categories and transcribe spoken content as the foundation for grounding and reasoning.

Task family 2: Spatial grounding. This family evaluates sound-source localization in egocentric and allocentric frames, each with direction and distance estimation. Egocentric grounding localizes the source relative to the agent’s current viewpoint, while allocentric grounding localizes it from a transformed viewpoint, defined by standing at a reference object and facing another object. This tests both source localization and viewpoint-aware spatial reasoning.

Task family 3: Spatial relational reasoning. This family extends source localization to scene-level reasoning by evaluating how localized sound sources relate to the visual scene. Visibility asks whether the source is visible from the current view, and correspondence asks which visible object is spatially aligned with the sound, where the answer may be semantically mismatched or None. Hidden-source localization identifies whether a non-visible source is hidden behind an object, in an adjacent room, or outside the field of view. Loudness change asks whether the source becomes louder or quieter after the agent moves to a target object; proximity reasoning asks which candidate object is nearest to the source; and scene captioning summarizes sound events, directions, and visibility states, including multiple emitters when present.

Chain-of-thought supervision. We further provide chain-of-thought (CoT) answer variants for allocentric grounding and spatial relational reasoning. The CoT rationales are deterministically generated from structured simulator metadata using task-specific templates, while GPT-4 [46] is used only to paraphrase question wording. These rationales expose task-specific intermediate cues beyond the final answer, such as sound event identity, source location, visibility checks, distance comparisons, and how source direction and distance change under transformed viewpoints. This supervision encourages the model to connect auditory, spatial, and visual cues through explicit intermediate reasoning steps rather than directly predicting the final answer.

![](images/b7436b45ddc67b62fcc83e5ae48605004ad4010d6da64f3eb543c8222ddb5fd1.jpg)  
Figure 3: Architecture and training curriculum of HLR-LLM. (a) HLR-LLM augments a strong AV-LLM with a binaural spatial audio branch, enabling the language model to jointly attend to text, visual, downmixed monaural audio, and binaural spatial audio tokens. (b) The model is trained through a three-stage curriculum: spatial audio alignment, spatial audio-visual instruction tuning, and final-stage CoT tuning for structured spatial reasoning.

## 4 Method

We introduce HLR-LLM for spatial audio-visual scene understanding. Our goal is to enable spatial audio-visual grounding and reasoning with binaural spatial cues while retaining the semantic percep tion ability of a strong base AV-LLM. To this end, HLR-LLM introduces a spatial audio pathway to the base AV-LLM, where a spatial audio encoder and projector provide binaural spatial cues alongside visual context and monaural semantic audio. We train the model with a curriculum that incorporates chain-of-thought supervision to guide multi-step spatial reasoning.

## 4.1 HLR-LLM architecture

Figure 3 (a) shows the architecture of HLR-LLM. The model consists of the base visual and monaural audio encoders, a newly introduced spatial audio encoder, a lightweight spatial audio projector, and the language model. The monaural audio branch preserves semantic hearing abilities, such as sound event recognition and speech understanding, while the spatial audio branch provides direction- and distance-sensitive cues for grounding and reasoning. The language model integrates visual, monaural audio, spatial audio, and text tokens to generate task-specific responses.

Spatial audio tokenization. Given a binaural audio signal $b \in \mathbb { R } ^ { 2 \times T }$ , we encode it with a pretrained spatial audio encoder $f _ { \mathrm { s p a } }$ [9] and map the resulting features into the language model embedding space using a trainable MLP projector $p _ { \mathrm { s p a } } \colon Z _ { \mathrm { s p a } } = p _ { \mathrm { s p a } } ( f _ { \mathrm { s p a } } ( b ) )$ . The projected embeddings $Z _ { \mathrm { s p a } }$ serve as spatial audio tokens that provide acoustic localization cues to the LLM.

Integration with the AV-LLM. We build HLR-LLM on Qwen2.5-Omni [4], which supports text, image, and audio inputs. We introduce a new media token, <spatial\_audio>, whose placeholder positions are replaced with the projected spatial audio tokens $\bar { Z } _ { \mathrm { s p a } }$ . The resulting multimodal input sequence is $X \doteq [ E ( q ) , Z _ { \mathrm { v i s } } , \bar { Z _ { \mathrm { a u d } } } , Z _ { \mathrm { s p a } } ]$ ], where $E ( q )$ denotes text embeddings, $Z _ { \mathrm { v i s } }$ visual tokens, $Z _ { \mathrm { a u d } }$ monaural audio tokens, and $Z _ { \mathrm { s p a } }$ spatial audio tokens. The language model generates the answer y conditioned on this sequence, $i . { \dot { e } } . , p ( y \mid X )$ . This design preserves the base model’s semantic audio-visual ability while explicitly injecting spatial acoustic cues.

## 4.2 Curriculum training for spatial audio-visual reasoning

Spatial audio-visual scene understanding requires low-level perception of what is heard and where it comes from, as well as high-level spatial reasoning over the scene. Figure 3 (b) summarizes the three-stage curriculum used to train HLR-LLM, which progresses from spatial audio alignment to spatial audio-visual instruction tuning and final-stage CoT tuning.

Stage 1: Spatial audio alignment. We first pretrain the spatial audio projector using a spatial audioonly QA dataset [9]. This aligns frozen spatial audio encoder features with the LLM embedding space and initializes the model to verbalize spatial audio properties, such as sound event and location.

Stage 2: Spatial audio-visual instruction tuning. We then fine-tune the model on the training split of HLR-AVSceneQA. This stage teaches the model to jointly use visual tokens, monaural audio tokens, and spatial audio tokens across spatial hearing, grounding, and relational reasoning tasks. For example, recognizing a ringing sound can rely on the monaural audio branch, while determining whether the source is visible or hidden requires combining spatial cues with visual evidence.

Stage 3: Chain-of-thought tuning. Finally, we continue instruction tuning with CoT-formatted answers for multi-step spatial tasks. The model learns to produce a compact rationale that connects auditory cues, spatial estimates, and visual evidence before giving a normalized final answer. This encourages the model to decompose spatial audio-visual queries into interpretable reasoning steps rather than relying on direct answer prediction.

Training objective. At each stage, we minimize the standard auto-regressive cross-entropy loss over the target answer tokens. Stage 1 updates only the spatial audio projector, while Stages 2–3 update the spatial audio projector and LoRA parameters of the language model. All visual, monaural audio, and spatial audio encoders remain frozen throughout training.

## 4.3 Implementation details

We follow the audio preprocessing of BAT [9]. Each source clip is loudness-normalized and convolved with the room impulse response generated by SoundSpaces to produce binaural audio. The resulting two-channel waveform is sampled at 32 kHz. We use a window size of 1024, a hop size of 320, and 128 mel bins to compute spectrogram features. The spatial audio branch receives the binaural signal, while the monaural audio branch receives its channel-averaged version. For visual and monaural audio inputs, we use the default preprocessing of Qwen2.5-Omni without modification. For LoRA fine-tuning, we use rank 16, alpha 16, dropout 0.05, a learning rate of $2 \times 1 0 ^ { - 4 }$ , and weight decay of 0.01. We train for 2 epochs on four NVIDIA A6000 GPUs with a per-device batch size of 4.

## 5 Experiment

## 5.1 Experimental setup

Datasets and metrics. We primarily evaluate models on HLR-AVSceneQA for comprehensive spatial audio-visual scene understanding. Except for far-field ASR, which is evaluated by word error rate (WER), and scene captioning, which is evaluated with CIDEr-D [47], METEOR [48], and SPIDEr [49], all other tasks are evaluated using accuracy. We also construct QA formats on the STARSS23 [19] dataset to evaluate adaptation to real-world spatial audio-visual recordings, covering sound event recognition, egocentric direction and distance prediction, and visibility prediction. We restrict the evaluation to these tasks because STARSS23 does not provide the object- and scene-level annotations required to reliably define hidden-source, allocentric, and other relational reasoning tasks. All fine-tuned models are trained on the corresponding training split and evaluated on the test split. Unless otherwise noted, we report mean ± standard deviation over three runs.

Competing methods. We compare HLR-LLM with both zero-shot and fine-tuned audio-visual models. For zero-shot evaluation, we consider several open-source AV-LLMs [1, 2, 3, 4, 5] and commercial models [50, 51], which mainly use visual input and monaural audio. We also compare with SAVVY [13], which uses multichannel audio but focuses on spatial grounding tasks. For finetuned baselines, we initialize Hear You Are LLM [15] and Qwen2.5-Omni [4] from their respective public checkpoints and fine-tune them on the HLR-AVSceneQA training split. Although Hear You Are LLM is originally trained with panoramic images, we adapt it to the same normal-FOV inputs and HLR-AVSceneQA supervision by fine-tuning its visual pathway, projectors/adapters, and LLM while keeping the audio encoder frozen. We additionally fine-tune a dual-channel variant of Qwen2.5-Omni, which separately encodes the left and right binaural channels through the original audio pathway and feeds both audio token streams to the LLM without an explicit spatial audio branch.

Table 3: Evaluation on spatial hearing and spatial grounding tasks. Sound event recognition and grounding tasks are reported in accuracy (%), while speech recognition is reported in WER (%, lower is better). V, M, and B denote visual input, monaural audio, and binaural audio, respectively. $\mathbf { M } _ { \mathrm { L } }$ and $\mathbf { M } _ { \mathrm { R } }$ denote independently encoded left- and right-channel audio inputs. Best and second-best mean results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Modality</td><td colspan="2">Spatial Hearing</td><td colspan="2">Egocentric Grounding</td><td colspan="2">Allocentric Grounding</td></tr><tr><td>Sound (Acc. %)</td><td>Speech (WER ↓)</td><td>Direction</td><td>Distance</td><td>Direction</td><td>Distance</td></tr><tr><td>Random chance</td><td></td><td>10.00</td><td></td><td>12.50</td><td>10.00</td><td>12.50</td><td>10.00</td></tr><tr><td colspan="8">Zero-shot models</td></tr><tr><td> $\mathbf { M i n i C P M - o - v } 2 . 6 [ 1 ]$ </td><td>V+M</td><td> $3 5 . 2 6 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $1 2 6 . 1 0 { \scriptstyle \pm 5 . 5 1 }$ </td><td> $1 2 . 8 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 4 . 2 6 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $1 1 . 9 2 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $1 8 . 4 2 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td> $\overline { { { \mathrm { V i d e o L L a M A 2 } } [ 2 ] } }$ </td><td>V+M</td><td> $5 4 . 5 4 { \overline { { \pm } } } 0 . 1 1$ </td><td> $1 1 3 . 5 4 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $1 2 . 1 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $7 . 7 3 \pm 0 . 0 5$ </td><td> $1 5 . 8 4 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $7 . 1 9 2 0 . 1 7$ </td></tr><tr><td> $\mathrm { V i d e o - S A L M O N N } 2 [ 3 ]$ </td><td>V+M</td><td> $3 2 . 2 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td>13.06±0.02</td><td>17.00±0.22</td><td> $7 . 6 2 \pm 0 . 0 7$ </td><td>17.55±0.20</td><td> $6 . 9 8 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>Qwen2.5-Omni [4]</td><td>V+M</td><td>57.21±0.09</td><td> $1 4 . 9 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $1 7 . 3 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 . 1 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $2 1 . 4 3 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 . 0 3 { \pm } 0 . 1 3$ </td></tr><tr><td> $\overline { { \mathrm { Q w e n 3 - O m n i } \left[ 5 \right] } } ^ { \cdot }$ </td><td>V+M</td><td> $5 5 . 8 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 5 . 2 7 { \scriptstyle \pm 1 . 2 9 }$ </td><td> $1 5 . 6 4 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 1 . 2 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 8 . 0 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $9 . 1 7 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td>SAVVY [13]</td><td> $\mathbf { V + M + B }$ </td><td></td><td></td><td> $5 2 . 3 6 { \pm } 0 . 4 6$ </td><td> $1 9 . 8 7 { \pm } 0 . 1 5$ </td><td> $1 7 . 8 1 { \pm } 0 . 3 8$ </td><td> $1 2 . 3 7 { \pm } 0 . 6 0$ </td></tr><tr><td>GPT-Realtime-Mini [50]</td><td>V+M</td><td> $3 4 . 2 1 { \pm } 0 . 3 6 $ </td><td> $1 8 . 0 7 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $1 2 . 6 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $1 4 . 5 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $1 4 . 2 7 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $1 7 . 2 8 { \scriptstyle \pm 0 . 9 6 }$ </td></tr><tr><td>Gemini-2.5-Flash-Lite [51]</td><td>V+M</td><td> $3 8 . 0 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 1 . 5 8 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 9 . 6 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 2 . 2 3 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $1 9 . 3 9 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $8 . 0 4 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td colspan="8">Fine-tuned models</td></tr><tr><td>Hear You Are LLM [15]</td><td>V+B</td><td> $6 7 . 6 7 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 4 8 . 0 0 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $4 4 . 6 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $2 2 . 6 3 { \pm } 0 . 5 4$ </td><td> $2 1 . 5 3 { \pm } 0 . 1 9$ </td><td> $2 1 . 5 7 { \scriptstyle \pm 0 . 3 3 }$ </td></tr><tr><td>Qwen2.5-Omni [4]</td><td>V+M</td><td> $8 0 . 5 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $1 5 . 3 3 { \pm } 2 . 3 3 $ </td><td> $3 2 . 6 9 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $3 0 . 9 2 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $2 6 . 2 6 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 7 . 7 3 { \scriptstyle \pm 1 . 2 2 }$ </td></tr><tr><td>Dual-channel Qwen2.5-Omni</td><td> $\mathbf { V + M _ { L } + M _ { R } }$ </td><td> ${ \bf 8 4 . 5 2 } \pm 0 . 0 7$ </td><td> $\underline { { 1 3 . 4 2 } } \pm 1 . 1 4$ </td><td> $6 4 . 2 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $4 6 . 5 9 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $4 2 . 7 1 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $3 7 . 8 4 { \pm } 0 . 2 4$ </td></tr><tr><td>HLR-LLM (Ours)</td><td> $\mathbf { V + M + B }$ </td><td> $8 1 . 0 3 { \pm } 0 . 1 1$ </td><td> $\overline { { 1 4 . 4 3 } } \pm 0 . 5 4$ </td><td> $\overline { { 7 8 . 2 1 } } \pm 0 . 3 4$ </td><td> $\overline { { { \bf 5 5 . 5 6 } } } \pm 0 . 2 7$ </td><td> $\mathbf { \bar { 5 2 . 0 3 } } 2 0 . 2 6$ </td><td> $\overline { { { \bf 4 5 . 0 8 } } } \pm 0 . 8 5$ </td></tr></table>

Table 4: Evaluation on spatial relational reasoning and scene captioning tasks. Relational reasoning tasks are reported in accuracy (%). Scene captioning is evaluated with CIDEr-D, METEOR, and SPIDEr. V, M, and B denote visual input, monaural audio, and binaural audio, respectively. $\mathbf { M } _ { \mathrm { L } }$ and $\mathbf { M } _ { \mathrm { R } }$ denote independently encoded left- and right-channel audio inputs. Best and second-best mean results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Modality</td><td colspan="5">Spatial Relational Reasoning (Acc. %)</td><td colspan="3">Scene Captioning</td></tr><tr><td>Visibility</td><td>Correspondence Hidden source</td><td></td><td>Loudness</td><td>Proximity</td><td>CIDEr-D</td><td>METEOR</td><td>SPIDEr</td></tr><tr><td>Random chance</td><td></td><td>50.00</td><td>9.49</td><td>20.00</td><td>50.00</td><td>28.68</td><td></td><td></td><td></td></tr><tr><td colspan="10">Zero-shot models</td></tr><tr><td>MiniCPM-o-v2.6 [1]</td><td>V+M</td><td>75.20±0.14</td><td> $2 2 . 5 2 { \pm } 1 . 0 9$ </td><td>11.65±0.62</td><td>47.29±0.48</td><td>26.83±0.62</td><td>0.078±0.004</td><td>0.173±0.001</td><td>0.136±0.002</td></tr><tr><td>VideoLLaMA2 [2]</td><td>V+M</td><td> $6 8 . 3 7 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 9 . 7 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td>21.25±0.26</td><td>39.18±0.20</td><td>25.52±0.30</td><td>0.112±0.002</td><td>0.131±0.001</td><td>0.079±0.000</td></tr><tr><td>Video-SALMONN 2 [3]</td><td>V+M</td><td>75.56±0.00</td><td> $7 . 9 7 { \scriptstyle \pm 0 . 2 2 }$ </td><td>22.00±0.31</td><td>51.89±0.00</td><td> $3 0 . 1 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td>0.075±0.001</td><td>0.094±0.001</td><td> $0 . 0 7 4 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>Qwen2.5-Omni [4]</td><td>V+M</td><td> $7 5 . 5 1 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 3 . 3 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td>21.48±0.54</td><td>51.87±0.04</td><td> $3 2 . 9 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td>0.301±0.002</td><td>0.168±0.001</td><td>0.172±0.001</td></tr><tr><td>Qwen3-Omni [5]</td><td>V+M</td><td> $7 3 . 9 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $2 6 . 1 8 2 0 . 2 8$ </td><td>23.58±0.36</td><td>51.29±0.85</td><td>32.06±0.39</td><td>0.344±0.004</td><td>0.203±0.000</td><td>0.260±0.001</td></tr><tr><td>GPT-Realtime-Mini [50]</td><td>V+M</td><td>69.46±0.61</td><td> $1 2 . 8 4 \pm 0 . 2 5$ </td><td>20.10±0.42</td><td>50.13±0.99</td><td> $2 6 . 0 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td>0.052±0.002</td><td>0.237±0.001</td><td>0.151±0.002</td></tr><tr><td>Gemini-2.5-Flash-Lite [51]</td><td>V+M</td><td> $4 9 . 8 1 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $2 0 . 2 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td>15.36±0.17</td><td>48.27±0.57</td><td> $3 0 . 8 9 { \scriptstyle \pm 0 . 3 4 }$ </td><td>0.570±0.010</td><td>0.209±0.002</td><td>0.218±0.003</td></tr><tr><td colspan="10">Fine-tuned models</td></tr><tr><td>Hear You Are LLM [15]</td><td>V+B</td><td>71.31±0.05</td><td>54.73±0.08</td><td>43.97±0.20</td><td>65.41±0.28</td><td>33.33±0.49</td><td>4.160±0.008</td><td>0.385±0.000</td><td>2.409±0.004</td></tr><tr><td>Qwen2.5-Omni [4]</td><td>V+M</td><td> $6 2 . 7 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 5 . 7 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $5 1 . 2 4 { \scriptstyle \pm 0 . 5 8 }$ </td><td>68.84±0.32</td><td>47.11±0.27</td><td>4.796±0.014</td><td>0.420±0.000</td><td>2.751±0.007</td></tr><tr><td>Dual-channel Qwen2.5-Omni V+ML+MR</td><td></td><td> $7 7 . 3 0 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $6 9 . 0 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td>60.12±0.43</td><td>72.59±0.31</td><td>55.64±0.36</td><td>5.414±0.011</td><td>0.444±0.001</td><td>3.070±0.005</td></tr><tr><td>HLR-LLM (Ours)</td><td>V+M+B</td><td> $\overline { { 8 0 . 5 1 } } \pm 0 . 3 8$ </td><td> ${ \bf 7 6 . 1 9 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> ${ \bf 6 7 . 2 3 } \pm \bf { 0 . 3 7 }$ </td><td>80.87±0.2061.50±0.51</td><td></td><td>4.461±0.004</td><td>0.439±0.001</td><td>3.109±0.004</td></tr></table>

## 5.2 Main results on HLR-AVSceneQA

Tables 3 and 4 summarize the main results on HLR-AVSceneQA. In Table 3, zero-shot AV-LLMs and commercial models show reasonable spatial hearing ability, but perform poorly on direction and distance grounding, indicating that monaural audio with visual input is insufficient for spatial localization. Baselines with spatial audio improve some grounding tasks, with SAVVY [13] performing well on egocentric direction and Hear You Are LLM improving over zero-shot models, but they remain limited in broader grounding and reasoning. Fine-tuning Qwen2.5-Omni, our base model, substantially improves sound-event recognition and spatial grounding, yet accurate localization remains challenging without explicit binaural cues. Dual-channel Qwen2.5-Omni further improves localization by processing left and right channels separately through the original audio pathway, confirming the value of binaural spatial cues. HLR-LLM outperforms this baseline across all egocentric and allocentric grounding metrics, demonstrating the benefit of explicitly modeling binaural spatial information rather than exposing the LLM to separate left- and right-channel tokens.

A similar trend appears in Table 4. Zero-shot models perform reasonably well on visibility, which can often be inferred from visual evidence, but struggle with tasks requiring spatial alignment and relational reasoning. Fine-tuned baselines improve substantially, highlighting the importance of taskspecific learning for spatial audio-visual understanding. HLR-LLM achieves the best performance across all spatial relational reasoning tasks and consistently outperforms the dual-channel Qwen2.5- Omni baseline, further supporting the benefit of explicit binaural modeling for scene-level reasoning.

Table 5: Ablation study. We ablate input modalities and training strategies to validate the contribution of each component. All metrics are reported in accuracy (%). V, M, and B denote visual input, monaural audio, and binaural audio, respectively. Results are mean ± standard deviation over three seeds. Best and second-best mean results are shown in bold and underlined, respectively. The full ablation is provided in Appendix B.
<table><tr><td rowspan="2">ID</td><td rowspan="2">Variant</td><td rowspan="2">Modality</td><td colspan="2">Egocentric Grounding</td><td colspan="2">Allocentric Grounding</td><td colspan="3">Spatial Relational Reasoning</td></tr><tr><td>Direction</td><td>Distance</td><td>Direction</td><td>Distance</td><td>Correspondence</td><td>Hidden source</td><td>Proximity</td></tr><tr><td colspan="9">Modality ablation</td><td></td></tr><tr><td>(a)</td><td>Without vision</td><td>M+B</td><td>74.72 ± 0.36</td><td> $5 0 . 6 4 \pm 0 . 5 4$ </td><td> $2 1 . 5 1 \pm 0 . 4 1$ </td><td> $2 6 . 6 3 \pm 0 . 6 2$ </td><td> $5 7 . 1 1 \pm 0 . 1 1$ </td><td> $6 6 . 6 2 \pm 0 . 2 1$ </td><td> $3 4 . 1 3 \pm 0 . 1 2$ </td></tr><tr><td>(b)</td><td>Visual only</td><td>V</td><td>31.20 ± 0.14</td><td>20.33 ± 0.49</td><td> $2 8 . 1 1 \pm 0 . 3 6$ </td><td> $2 2 . 9 6 \pm 0 . 2 2$ </td><td> $2 6 . 7 1 \pm 0 . 0 2$ </td><td>24.16 ± 0.43</td><td>37.09 ± 0.55</td></tr><tr><td>(c)</td><td>Without bínaural</td><td>V+M</td><td>32.69 ± 0.26</td><td> $3 0 . 9 2 \pm 0 . 4 2$ </td><td> $2 6 . 2 6 \pm 0 . 1 8$ </td><td> $2 7 . 7 3 \pm 1 . 2 2$ </td><td> $5 5 . 7 3 \pm 0 . 0 7$ </td><td> $5 1 . 2 4 \pm 0 . 5 8$ </td><td> $4 7 . 1 1 \pm 0 . 2 7$ </td></tr><tr><td colspan="9">Training ablation</td></tr><tr><td>(d)</td><td>Stage-2 training</td><td>V+M+B</td><td> $7 7 . 9 2 \pm 0 . 0 5$ </td><td> $5 2 . 4 4 \pm 0 . 3 7$ </td><td> $\smash { \frac { 4 8 . 3 7 } { s - s } \frac { \pm 0 . 2 1 } { 1 } }$ </td><td> $3 8 . 8 3 \pm 0 . 9 6$ </td><td> $6 7 . 6 9 \pm 0 . 1 1$ </td><td> $6 6 . 5 2 \pm 0 . 3 5$ </td><td> $5 6 . 2 3 \pm 0 . 4 3$ </td></tr><tr><td>(e)</td><td>Joint training</td><td>V+M+B</td><td> $7 8 . 0 0 \pm 0 . 1 5$ </td><td> $5 3 . 0 4 \pm 0 . 1 7$ </td><td>45.01 ± 0.59</td><td> $3 9 . 2 5 \pm 0 . 6 8$ </td><td> $7 2 . 7 1 \pm 0 . 2 3$ </td><td> ${ \bf 6 9 . 7 8 \pm 0 . 2 4 }$ </td><td> $5 6 . 9 2 \pm 0 . 2 1$ </td></tr><tr><td>(f)</td><td> $\mathtt { H L R - L L M ( O u r s ) }$ </td><td> $\mathbf { V + M + B }$ </td><td> $\overline { { 7 8 . 2 1 } } \pm 0 . 3 4$ </td><td> ${ \overline { { 5 5 . 5 6 } } } \pm 0 . 2 7$ </td><td> ${ \bf 5 2 . 0 3 \pm 0 . 2 6 }$ </td><td> $\overline { { 4 5 . 0 8 } } \pm 0 . 8 5$ </td><td> $\mathbf { \widetilde { 7 6 . 1 9 } \pm 0 . 1 1 }$ </td><td> $6 7 . 2 3 \pm 0 . 3 7$ </td><td> $\mathbf { 6 1 . 5 0 \pm 0 . 5 1 }$ </td></tr></table>

Table 6: Real-world adaptation on STARSS23. “Pretrain” denotes HLR-AVSceneQA pretraining before STARSS23 fine-tuning. “Sound” denotes sound event recognition; direction, distance, and visibility are egocentric. All metrics are accuracy (%). Results are mean ± standard deviation over three seeds. Best and second-best mean results are shown in bold and underlined, respectively.
<table><tr><td>ID</td><td>Model</td><td>Pretrain</td><td>Sound</td><td>Direction</td><td>Distance</td><td>Visibility</td></tr><tr><td>(a)</td><td>Hear You Are LLM [15]</td><td>-√</td><td> $3 8 . 6 6 \pm 0 . 1 6$ </td><td> $1 5 . 2 5 \pm 0 . 5 7$ </td><td> $3 4 . 8 7 \pm 0 . 3 5$ </td><td> $6 2 . 3 5 \pm 0 . 0 7$ </td></tr><tr><td>(b)</td><td>Hear You Are LLM [15]</td><td></td><td> $4 0 . 8 0 \pm 0 . 7 6$ </td><td> $\underline { { 1 6 . 1 5 } } \pm 0 . 3 6$ </td><td> $2 6 . 9 3 \pm 0 . 4 5$ </td><td> $6 3 . 8 0 \pm 0 . 4 8$ </td></tr><tr><td>(c)</td><td>HLR-LLM</td><td></td><td> $\underline { { 4 4 . 5 9 } } \pm 0 . 1 3$ </td><td> $1 6 . 1 3 \pm 0 . 0 4$ </td><td> $2 8 . 1 1 \pm 0 . 0 4$ </td><td> $6 9 . 6 7 \pm 0 . 1 8$ </td></tr><tr><td>(d)</td><td>HLR-LLM</td><td>-√</td><td> ${ \bf 5 3 . 9 3 \pm 0 . 1 4 }$ </td><td> ${ \bf 2 9 . 9 2 \pm 0 . 1 3 }$ </td><td> ${ \bf 3 5 . 1 3 \pm 0 . 0 7 }$ </td><td> ${ \bf 7 1 . 6 8 \pm 0 . 0 9 }$ </td></tr></table>

Compared with Hear You Are LLM, HLR-LLM achieves consistent gains across relational reasoning tasks, demonstrating stronger scene-level reasoning beyond source localization. For scene captioning, HLR-LLM achieves the best SPIDEr score, while dual-channel Qwen2.5-Omni obtains the best CIDEr-D and METEOR scores. Overall, the results show that explicit spatial audio modeling benefits not only source localization but also reasoning about sound sources in the visual scene.

## 5.3 Ablation study

Table 5 examines the contribution of each modality and training strategy. Removing vision (a) still retains egocentric direction and distance performance, since binaural audio provides direct spatial cues. However, it substantially reduces allocentric grounding and proximity reasoning, where visual object locations are required. Interestingly, the model without vision still performs relatively well on hidden-source reasoning, suggesting that binaural cues provide useful evidence for distinguishing non-visible source configurations. The visual-only model (b) performs poorly across most tasks, showing that vision alone is insufficient for reliable sound-source localization, especially when sources are off-screen or hidden. Removing binaural audio (c) also causes a large drop in grounding tasks, showing that monaural audio lacks the spatial cues needed for accurate localization. At the same time, its relatively stronger proximity performance suggests that visual context and semantic audio can still support some object-level relational reasoning.

The training ablation shows how each stage contributes to spatial reasoning. Stage-2 training (d), which includes Stages 1–2 but omits CoT tuning, already provides strong performance. Joint training with CoT (e), which mixes Stage-2 instruction data with CoT-formatted data, improves egocentric grounding, correspondence, and hidden-source reasoning, but does not consistently improve all grounding tasks. In contrast, our final model (f), which applies CoT tuning after Stage-2 instruction tuning, performs best across all grounding metrics and most relational reasoning tasks, with clear gains in allocentric grounding, correspondence, and proximity. These gains are most evident on tasks requiring multi-step spatial reasoning, such as transforming viewpoints or comparing object–source relations. Overall, the ablation indicates that binaural audio supports spatial localization, vision supports object-level grounding, and final-stage CoT tuning improves complex spatial reasoning.

## 5.4 Real-world adaptation

Table 6 evaluates real-world adaptation on STARSS23. We fine-tune each model on the STARSS23 training set and compare performance with and without HLR-AVSceneQA pretraining. Comparing (a)

![](images/c33cccb36256ab2521b4e58e37b8bf8dae7ccb741444d0eabacfa79772d09d70.jpg)  
Figure 4: Qualitative examples of spatial grounding. HLR-LLM correctly identifies the sound direction in a scene with multiple similar objects (left) and localizes a non-visible sound source as hidden behind the bed (right), outperforming the comparison models in the hidden-source example.

vs. (b) and (c) vs. (d), HLR-AVSceneQA pretraining improves most evaluated tasks, suggesting that it provides useful spatial priors for adapting to real spatial audio-visual recordings. Among models with HLR-AVSceneQA pretraining, HLR-LLM (d) outperforms Hear You Are LLM (b) across all metrics, with particularly large gains in sound recognition and direction grounding, demonstrating stronger adaptation to real-world spatial audio-visual data.

## 5.5 Qualitative results

Figure 4 shows representative qualitative results for spatial grounding and relational reasoning. In the left example, multiple visually similar electric blenders are present, while the sound originates from the front-right direction. The monaural Qwen2.5-Omni predicts an incorrect direction, whereas models with binaural spatial input (Hear You Are LLM and HLR-LLM) correctly localize the source. The right example is more challenging because the sounding bird is not visible and must be localized using spatial audio and scene layout. While Qwen2.5-Omni and Hear You Are LLM incorrectly attribute the source to other non-visible regions, HLR-LLM correctly identifies it as hidden behind the bed. Together, these examples show that explicit spatial audio cues help HLR-LLM localize sound sources and ground them within visible or hidden regions of the scene. Further results and analyses across diverse spatial reasoning scenarios are provided in Appendix B.

## 6 Conclusion and discussion

We introduce HLR-AVSceneQA, a comprehensive benchmark for spatial audio-visual scene understanding organized around a hear-localize-reason hierarchy. By covering spatial hearing, egocentric and allocentric grounding, and spatial relational reasoning, HLR-AVSceneQA evaluates whether AV-LLMs can move beyond recognizing audio-visual content to connect what is heard, where it originates, and how the source relates to the surrounding scene. We also propose HLR-LLM, which augments a strong AV-LLM with a binaural spatial audio branch and a staged curriculum with final-stage CoT supervision, improving spatial reasoning while retaining strong semantic audio-visual perception. Real-world adaptation experiments on STARSS23 further show that HLR-AVSceneQA pretraining provides useful spatial priors for downstream adaptation. These results highlight the importance of explicit spatial audio modeling and structured training for spatially aware AV-LLMs.

Our work also has limitations. HLR-AVSceneQA is built with a simulator-based pipeline, which provides controllable annotations but cannot fully capture the acoustic complexity and visual diversity of the real world. The benchmark focuses on indoor scenes, leaving dynamic agents, moving sound sources, and richer embodied interaction for future work. Future research should extend spatial audio-visual benchmarks to more realistic and dynamic settings and develop AV-LLMs that can actively perceive, move, and reason in spatial environments.

## Acknowledgments and disclosure of funding

This work was supported by the National Research Foundation of Korea (NRF) funded by the Korean government (MSIT) in 2026 (Grant No. RS-2024-00451947, Project Title: Interaction Framework for Connecting Users to Generative Experiences via Time-Space Expanded Multimedia: Centering on Social Communication Disorders (33%); No. RS-2024-00358135, Corner Vision: Learning to Look Around the Corner through Multi-modal Signals (24%)), the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. RS-2025-25443318, Physically-grounded Intelligence: A Dual Competency Approach to Embodied AGI through Constructing and Reasoning in the Real World (33%); No. RS-2026- 25518317, Development of AI memory mechanism that reflects human cognitive principles (10%)), the InnoCORE program of the Ministry of Science and ICT (26-InnoCORE-01; N10250156), and the Artificial Intelligence Industrial Convergence Cluster Development Project funded by the Ministry of Science and ICT (MSIT, Korea) & Gwangju Metropolitan City.

Competing interests. The authors declare no competing interests.

## References

[1] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. MiniCPM-V: A GPT-4V level MLLM on your phone. arXiv preprint arXiv:2408.01800, 2024. 2, 3, 7, 8, 15

[2] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. VideoLLaMA 2: Advancing spatial-temporal modeling and audio understanding in video-LLMs. arXiv preprint arXiv:2406.07476, 2024. 2, 3, 7, 8, 15, 18

[3] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, and Chao Zhang. Video-SALMONN 2: Caption-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025. 2, 3, 7, 8, 15, 18

[4] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215, 2025. 2, 3, 6, 7, 8, 15, 16, 18

[5] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765, 2025. 2, 3, 7, 8, 15

[6] Qilang Ye, Zitong Yu, Rui Shao, Xinyu Xie, Philip Torr, and Xiaochun Cao. CAT: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios. In European Conference on Computer Vision (ECCV), 2024. 2

[7] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2

[8] Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Jun Chen, Mohamed Elhoseiny, Ruohan Gao, and Dinesh Manocha. Meerkat: Audio-visual large language model for grounding in space and time. In European Conference on Computer Vision (ECCV), 2024. 2

[9] Zhisheng Zheng, Puyuan Peng, Ziyang Ma, Xie Chen, Eunsol Choi, and David Harwath. BAT: Learning to reason about spatial sounds with large language models. In International Conference on Machine Learning (ICML), 2024. 2, 3, 6, 7, 15

[10] Subrata Biswas, Mohammad Nur Hossain Khan, and Bashima Islam. OWL: Geometry-aware spatial reasoning for audio large language models. In International Conference on Learning Representations (ICLR), 2026. 2, 3

[11] Bhavika Devnani, Skyler Seto, Zakaria Aldeneh, Alessandro Toso, Elena Menyaylenko, Barry-John Theobald, Jonathan Sheaffer, and Miguel Sarabia. Learning spatially-aware language and audio embeddings. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3

[12] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Jun Zhang, Lu Lu, Zejun Ma, Yuxuan Wang, et al. Can large language models understand spatial audio? In Conference of the International Speech Communication Association (INTERSPEECH), 2024. 2, 3

[13] Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, and Eli Shlizerman. SAVVY: Spatial awareness via audio-visual LLMs through seeing and hearing. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2, 3, 7, 8, 16

[14] Zhan Liu, Changli Tang, Yuxin Wang, Zhiyuan Zhu, Youjun Chen, Yiwen Shao, Tianzi Wang, Lei Ke, Zengrui Jin, and Chao Zhang. JAEGER: Joint 3D audio-visual grounding and reasoning in simulated physical environments. In International Conference on Machine Learning (ICML), 2026. 2, 3

[15] Hyeonggon Ryu, Joon Son Chung, and David Harwath. Hear you are: Teaching LLMs spatial reasoning with vision and spatial sound. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2, 3, 7, 8, 9, 16

[16] Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip Robinson, and Kristen Grauman. Soundspaces 2.0: A simulation platform for visual-acoustic learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 4, 21

[17] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 3

[18] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research, 2024. 2, 3

[19] Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel A Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, et al. STARSS23: An audio-visual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 2, 7, 16

[20] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 3

[21] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning (ICML), 2023. 3

[22] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2017. 3

[23] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020. 3

[24] Yuan Gong, Yu-An Chung, and James Glass. AST: Audio spectrogram transformer. In Conference of the International Speech Communication Association (INTERSPEECH), 2021. 3

[25] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020. 3

[26] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019. 3

[27] Minkyu Kim, Kim Sung-Bin, and Tae-Hyun Oh. Prefix tuning for automated audio captioning. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023. 3

[28] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen. Sound event localization and detection of overlapping sources using convolutional recurrent neural networks. IEEE Journal ofSelected Topics in Signal Processing, 2019. 3

[29] Kazuki Shimada, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, and Yuki Mitsufuji. ACCDOA: Activity-coupled cartesian direction of arrival representation for sound event localization and detection. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2021. 3

[30] Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Naoya Takahashi, Emiru Tsunoo, and Yuki Mitsufuji. Multi-ACCDOA: Localizing and detecting overlapping sounds from the same class with auxiliary duplicating permutation invariant training. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2022. 3

[31] Oh Hyun-Bin, Kazuki Shimada, Yuhta Takida, Kim Sung-Bin, Toshimitsu Uesaka, Takashi Shibuya, Kyeongyoon Lee, Tae-Hyun Oh, and Yuki Mitsufuji. Spatio-temporal audio language modeling for dynamic sound sources. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026. 3

[32] Kentaro Seki, Yuki Okamoto, Kouei Yamaoka, Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari. Spatial-CLAP: Learning spatially-aware audio–text embeddings for multi-source conditions. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2026. 3

[33] Jiamin Xie, Ju Lin, Yiteng Huang, Tyler Vuong, Zhaojiang Lin, Zhaojun Yang, Peng Su, Prashant Rawat, Sangeeta Srivastava, Ming Sun, et al. Thinking in directivity: Speech large language model for multi-talker directional speech recognition. In Conference of the International Speech Communication Association (INTERSPEECH), 2025. 3

[34] Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. AVH-Bench: A cross-modal hallucination benchmark for audio-visual large language models. In International Conference on Learning Representations (ICLR), 2025. 3

[35] Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, and Dinesh Manocha. AVTrustBench: Assessing and enhancing reliability and robustness in audio-visual LLMs. In IEEE International Conference on Computer Vision (ICCV), 2025. 3

[36] Kaixiong Gong, Kaituo Feng, Bohao Li, Yibing Wang, Mofan Cheng, Shijia Yang, Jiaming Han, Benyou Wang, Yutong Bai, Zhuoran Yang, and Xiangyu Yue. Probing audio-visual reasoning in multimodal language models through the lens of audio. In Annual Meeting of the Association for Computational Linguistics (ACL), 2026. 3

[37] Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. AVQA: A dataset for audio-visual question answering on videos. In ACM International Conference on Multimedia (MM), 2022. 3

[38] Ziwei Zhou, Rui Wang, Zuxuan Wu, and Yu-Gang Jiang. Daily-Omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025. 3

[39] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, et al. Habitat-Matterport 3D semantics dataset. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 4, 21

[40] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A large-scale audio-visual dataset. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020. 4

[41] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. ImageBind: One embedding space to bind them all. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 4

[42] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2015. 4

[43] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-XL: A universe of 10M+ 3D objects. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 4, 21

[44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 5, 21

[45] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3D latents for scalable and versatile 3D generation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 5, 21

[46] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 5, 26

[47] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 7

[48] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings ofthe acl workshop on intrinsic and extrinsic evaluation measuresfor machine translation and/or summarization, 2005. 7

[49] Siqi Liu, Zhenhai Zhu, Ning Ye, Sergio Guadarrama, and Kevin Murphy. Improved image captioning via policy gradient optimization of SPIDEr. In IEEE International Conference on Computer Vision (ICCV), 2017. 7

[50] OpenAI. Realtime API documentation, 2025. Accessed: 2026-05-07. 7, 8, 15

[51] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 7, 8, 15

[52] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 15

[53] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 15

[54] V Ralph Algazi, Richard O Duda, Dennis M Thompson, and Carlos Avendano. The CIPIC HRTF database. In Proceedings ofthe 2001 IEEE Workshop on the Applications ofSignal Processing to Audio and Acoustics (WASPAA), 2001. 16

[55] Kourosh Saberi and David R Perrott. Minimum audible movement angles as a function of sound source trajectory. The Journal ofthe Acoustical Society ofAmerica (JASA), 1990. 16

[56] Andrew J Kolarik, Brian CJ Moore, Pavel Zahorik, Silvia Cirstea, and Shahina Pardhan. Auditory distance perception in humans: A review of cues, development, neuronal bases, and effects of sensory loss. Attention, Perception, & Psychophysics, 2016. 16

[57] Rachel Ege, A John Van Opstal, and Marc M Van Wanrooij. Accuracy-precision trade-off in human sound localisation. Scientific Reports, 2018. 21

## Table of Contents

A Additional experimental details 15   
A.1 Implementation details . 15   
A.2 Details on comparison model experiments 15   
A.3 STARSS23 preprocessing and QA construction 16   
B Additional results and analysis 17   
B.1 Ablation study . . 17   
B.2 Analysis on the STARSS23 dataset . 18   
B.3 Qualitative results . 18   
C Details on HLR-AVSceneQA 21   
C.1 Scene initialization, agent placement, and scene generation . . 21   
C.2 Task definitions and details . 23   
C.3 Chain-of-thought supervision . 26   
C.4 Quality control 28   
C.5 Dataset statistics . 28   
D Broader impact 29   
E LLM usage 29   
F Licenses and terms of existing assets 30

## A Additional experimental details

## A.1 Implementation details

HLR-LLM is built on Qwen2.5-Omni with an additional binaural spatial audio pathway. For the spatial branch, we use a pretrained Spatial-AST encoder following BAT [9]. Spatial-AST produces audio tokens together with class tokens trained for semantic classification, direction prediction, and distance prediction. We use these tokens as spatial audio features and map them into the language embedding space with a lightweight projector. The projector is an MLP with a linear–GELU–linear structure, which has been shown to be effective for extending LLMs to multimodal inputs [52, 53].

Training follows the three-stage curriculum described in the main paper. In Stage 1, we train only the spatial alignment modules, including the spatial audio resampler and projector, while freezing the language model, visual encoder, monaural audio encoder, and spatial audio encoder. In Stages 2–3, we LoRA-finetune the language model and continue updating the spatial alignment modules while keeping the encoders frozen. We apply LoRA to all linear layers with rank 16, alpha 16, and dropout 0.05, and use LoRA+ with a larger learning-rate ratio for the LoRA B matrix. We train with bfloat16 precision, FlashAttention, learning rate $2 \times 1 0 ^ { - 4 }$ , warmup ratio 0.05, weight decay 0.01, and gradient clipping at 1.0. Training is performed for 2 epochs on four NVIDIA A6000 GPUs with per-device batch size 4 and gradient accumulation 4.

## A.2 Details on comparison model experiments

Since spatial AV-LLMs are still limited, we first evaluate existing AV-LLMs that accept visual input and monaural audio [1, 2, 3, 4, 5, 50, 51]. For these models, the binaural audio is converted to monaural during preprocessing. For SAVVY [13], we follow the official code to process our inputs and evaluate only the applicable spatial grounding tasks, matching the scope of the original paper.

For fairer comparisons, we additionally fine-tune three baselines using the same direct-answer HLR-AVSceneQA training split.

Hear You Are LLM. We fine-tune Hear You Are LLM [15], the closest existing trainable model that uses both visual input and binaural audio for spatial audio-visual reasoning. Although it was originally trained with panoramic observations, HLR-AVSceneQA is defined under normal-field-ofview (normal-FOV) inputs, where sources may be visible, occluded, off-screen, or located in another room. We therefore adapt Hear You Are LLM to the same 1280×720 normal-FOV images, 32-kHz binaural audio, QA prompts, answer targets, and evaluation protocol used for HLR-LLM. Its SigLIP image encoder can directly process the normal-FOV images as patch tokens without architectural modification. During fine-tuning, we apply LoRA to the image encoder (rank 16, alpha 32) and update the projectors/adapters and LLM, while keeping the audio encoder frozen. We retain the original Hear You Are architecture and do not add the spatial audio branch used in HLR-LLM. Accordingly, this comparison evaluates two adapted spatial AV-LLMs under matched data and evaluation conditions, rather than providing a fully architecture-matched comparison.

Qwen2.5-Omni. We fine-tune Qwen2.5-Omni [4], our monaural-audio base model, on HLR-AVSceneQA. This provides a controlled baseline for evaluating spatial reasoning without explicit binaural spatial modeling.

Dual-channel Qwen2.5-Omni. We additionally fine-tune a dual-channel variant of Qwen2.5-Omni to test whether simply providing both binaural channels is sufficient for spatial reasoning. Instead of downmixing the binaural input to mono, the left and right channels are separately encoded through the original monaural audio pathway, and both resulting audio token streams are provided to the LLM. Unlike HLR-LLM, this variant does not introduce a dedicated spatial audio encoder or spatial projector. Thus, it isolates the benefit of explicit spatial-audio modeling from simply exposing the base model to both audio channels.

## A.3 STARSS23 preprocessing and QA construction

STARSS23 [19] is an audio-visual dataset of real-world spatial recordings with synchronized video, first-order ambisonic (FOA) audio, and frame-level spatio-temporal annotations of sound events. We use STARSS23 as a real-world benchmark complementary to our simulator-generated HLR-AVSceneQA. Starting from the official STARSS23 development split, we follow the official train/test partition and divide each recording into non-overlapping 10-second clips. Clips without a corresponding video stream are removed, resulting in 1,317 train clips and 1,116 test clips. For each clip, we extract a rectilinear egocentric image from the midpoint video frame. We also convert the FOA audio into binaural audio using the CIPIC subject-048 HRTF convention [54]. This makes the real-world recordings compatible with the binaural audio interface used by our spatial audio-visual models while retaining the original STARSS23 annotations.

We convert STARSS23 azimuth annotations into our egocentric convention, where 0<sup>◦</sup> is front, $9 0 °$ is right, 180<sup>◦</sup> is back, and 270<sup>◦</sup> is left. We construct QA pairs for sound-event recognition, egocentric direction prediction, egocentric distance prediction, and visibility prediction. We distinguish static and dynamic sources using angular and distance variation of active annotation frames, and use only static sources for spatial QA. Following psychoacoustic studies of auditory motion and distance discrimination [55, 56], a source is treated as static when its angular variation is within 5<sup>◦</sup> and its relative distance variation is within 20%.

For each retained static source, we compute representative azimuth and distance values from its active annotation frames and discretize them into the same egocentric bins used in HLR-AVSceneQA. We remove ambiguous QA items, including cases where frame-level annotations cross bin boundaries or multiple same-class sources make the queried source unclear. Since STARSS23 does not provide object-level semantic masks, visibility is defined geometrically by whether the representative source direction falls inside the canonical rectilinear camera field of view.

Scope of real-world tasks. We restrict the real-world evaluation to tasks that can be reliably derived from STARSS23 annotations. STARSS23 does not provide the dense object- and scene-level annotations required for HLR-style hidden-source or allocentric grounding, including object geometry, occlusion state, room layout, and reference-object information. We therefore avoid constructing uncertain pseudo-labels for these unsupported tasks.

Table S1: Ablation study on spatial hearing and spatial grounding tasks. Sound event recognition and grounding tasks are reported in accuracy (%), while speech recognition is reported in WER (%, lower is better). V, M, and B denote visual input, monaural audio, and binaural audio, respectively. Results are mean ± standard deviation over three seeds. Best and second-best mean results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">ID Variant</td><td rowspan="2">Modality</td><td colspan="2">Spatial Hearing</td><td colspan="2">Egocentric Grounding</td><td colspan="2">Allocentric Grounding</td></tr><tr><td>Sound (Acc. %) Speech (WER ↓)</td><td></td><td>Direction</td><td>Distance</td><td>Direction</td><td>Distance</td></tr><tr><td colspan="8">Modality ablation</td></tr><tr><td>(a) Without vision</td><td>M+B</td><td> $7 8 . 4 4 \pm 0 . 1 1$ </td><td> $1 4 . 6 2 \pm 2 . 8 1$ </td><td> $7 4 . 7 2 \pm 0 . 3 6$ </td><td> $5 0 . 6 4 \pm 0 . 5 4$ </td><td> $2 1 . 5 1 \pm 0 . 4 1$ </td><td> $2 6 . 6 3 \pm 0 . 6 2$ </td></tr><tr><td>(b) Visual only</td><td>V</td><td> $2 9 . 1 6 \pm 0 . 0 4$ </td><td> $2 5 0 . 1 4 \pm 1 5 . 3 7$ </td><td> $3 1 . 2 0 \pm 0 . 1 4$ </td><td> $2 0 . 3 3 \pm 0 . 4 9$ </td><td> $2 8 . 1 1 \pm 0 . 3 6$ </td><td> $2 2 . 9 6 \pm 0 . 2 2$ </td></tr><tr><td>(c) Without binaural</td><td>V+M</td><td> $8 0 . 5 7 \pm 0 . 2 7$ </td><td> $1 5 . 3 3 \pm 2 . 3 3$ </td><td> $3 2 . 6 9 \pm 0 . 2 6$ </td><td> $3 0 . 9 2 \pm 0 . 4 2$ </td><td> $2 6 . 2 6 \pm 0 . 1 8$ </td><td> $2 7 . 7 3 \pm 1 . 2 2$ </td></tr><tr><td>(d) Without monaural</td><td>V+B</td><td> $8 0 . 8 1 \pm 0 . 0 3$ </td><td> $3 6 1 . 0 5 \pm 4 . 0 0$ </td><td> $7 7 . 8 7 \pm 0 . 1 4$ </td><td> $4 7 . 7 8 \pm 0 . 4 9$ </td><td> $5 0 . 2 3 \pm 0 . 4 1$ </td><td> $3 7 . 9 5 \pm 0 . 2 4$ </td></tr><tr><td colspan="8">Data scale ablation (Stage-2 training)</td></tr><tr><td>(e) 10% data</td><td> $\mathbf { V + M + B }$ </td><td> $7 5 . 7 7 \pm 0 . 2 6$ </td><td> ${ \bf 1 0 . 6 5 \pm 0 . 1 8 }$ </td><td> $7 5 . 3 9 \pm 0 . 0 7$ </td><td> $4 8 . 1 5 \pm 0 . 1 5$ </td><td> $2 9 . 3 1 \pm 0 . 1 1$ </td><td> $2 9 . 1 0 \pm 0 . 7 1$ </td></tr><tr><td>(f) 50% data</td><td> $\mathbf { V + M + B }$ </td><td> ${ \bf 8 1 . 0 9 \pm 0 . 1 3 }$ </td><td> $1 3 . 9 7 \pm 0 . 9 8$ </td><td> $7 6 . 2 2 \pm 0 . 0 5$ </td><td> ${ \bf 5 5 . 8 8 \pm 0 . 1 9 }$ </td><td> $4 9 . 7 3 \pm 0 . 3 6$ </td><td> $\underline { { 4 1 . 2 0 } } \pm 0 . 3 3$ </td></tr><tr><td>(g) 100% data</td><td> $\mathbf { V + M + B }$ </td><td> ${ \bf 8 1 . 0 9 \pm 0 . 2 8 }$ </td><td> $\underline { { 1 1 . 0 6 } } \pm 0 . 1 1$ </td><td> $7 7 . 9 2 \pm 0 . 0 5$ </td><td> $5 2 . 4 4 \pm 0 . 3 7$ </td><td> $4 8 . 3 7 \pm 0 . 2 1$ </td><td> $3 8 . 8 3 \pm 0 . 9 6$ </td></tr><tr><td colspan="8">Training ablation (100% data; compare (g)–(i))</td></tr><tr><td>(h) Joint training</td><td> $\mathbf { V + M + B }$ </td><td> $7 9 . 0 9 \pm 0 . 0 3$ </td><td> $1 2 . 1 9 \pm 0 . 4 3$ </td><td> $7 8 . 0 0 \pm 0 . 1 5$ </td><td> $5 3 . 0 4 \pm 0 . 1 7$ </td><td> $4 5 . 0 1 \pm 0 . 5 9$ </td><td> $3 9 . 2 5 \pm 0 . 6 8$ </td></tr><tr><td>(i) HLR-LLM (Ours) V+M+B</td><td></td><td> $8 1 . 0 3 \pm 0 . 1 1$ </td><td> $1 4 . 4 3 \pm 0 . 5 4$ </td><td> ${ \bf 7 8 . 2 1 \pm 0 . 3 4 }$ </td><td> $5 5 . 5 6 \pm 0 . 2 7$ </td><td> ${ \bf 5 2 . 0 3 \pm 0 . 2 6 }$ </td><td> ${ \bf 4 5 . 0 8 \pm 0 . 8 5 }$ </td></tr></table>

Table S2: Ablation study on spatial relational reasoning and scene captioning tasks. Relational reasoning tasks are reported in accuracy (%). Scene captioning is evaluated with CIDEr-D, METEOR, and SPIDEr. V, M, and B denote visual input, monaural audio, and binaural audio, respectively. Results are mean ± standard deviation over three seeds. Best and second-best mean results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">ID Variant</td><td rowspan="2">Modality</td><td colspan="5">Spatial Relational Reasoning (Acc. %)</td><td colspan="3">Scene Captioning</td></tr><tr><td>Visibility</td><td>Correspondence</td><td>Hidden source</td><td>Loudness</td><td>Proximity</td><td>CIDEr-D</td><td>METEOR</td><td>SPIDEr</td></tr><tr><td colspan="10">Modality ablation</td></tr><tr><td>(a) Without vision</td><td>M+B</td><td> $6 5 . 4 3 \pm 0 . 4 2$ </td><td> $5 7 . 1 1 \pm 0 . 1 1$ </td><td> $6 6 . 6 2 \pm 0 . 2 1$ </td><td> $6 9 . 3 7 \pm 0 . 1 4$ </td><td> $3 4 . 1 3 \pm 0 . 1 2$ </td><td> $5 . 3 9 0 \pm 0 . 0 1 4$ </td><td> $0 . 4 4 1 \pm 0 . 0 0 1$ </td><td> $3 . 0 7 0 \pm 0 . 0 0 7$ </td></tr><tr><td>(b) Visual only</td><td>V</td><td> $5 3 . 2 1 \pm 0 . 1 0$ </td><td> $2 6 . 7 1 \pm 0 . 0 2$ </td><td> $2 4 . 1 6 \pm 0 . 4 3$ </td><td>52.97 ± 0.34</td><td> $3 7 . 0 9 \pm 0 . 5 5$ </td><td> $1 . 7 5 7 \pm 0 . 0 0 6$ </td><td> $0 . 3 1 6 \pm 0 . 0 0 1$ </td><td> $1 . 1 7 1 \pm 0 . 0 0 2$ </td></tr><tr><td>(c) Without binaural</td><td>V+M</td><td> $6 2 . 7 2 \pm 0 . 1 0$ </td><td>55.73 ± 0.07</td><td>51.24 ± 0.58</td><td>68.84 ± 0.32</td><td> $4 7 . 1 1 \pm 0 . 2 7$ </td><td> $4 . 7 9 6 \pm 0 . 0 1 4$ </td><td>0.420 ± 0.000</td><td> $2 . 7 5 1 \pm 0 . 0 0 7$ </td></tr><tr><td>(d) Without monaural</td><td>V+B</td><td> $7 5 . 6 7 \pm 0 . 0 7$ </td><td> $6 6 . 5 8 \pm 0 . 0 9$ </td><td> $6 6 . 4 8 \pm 0 . 1 0$ </td><td> $7 7 . 6 4 \pm 0 . 1 4$ </td><td> $5 5 . 2 3 \pm 0 . 6 2$ </td><td> $5 . 4 7 5 \pm 0 . 0 1 6$ </td><td> $0 . 4 4 0 \pm 0 . 0 0 1$ </td><td> $3 . 0 9 1 \pm 0 . 0 0 8$ </td></tr><tr><td colspan="10">Data scale ablation (Stage-2 training)</td></tr><tr><td>(e) 10% data</td><td>V+M+B</td><td> $7 1 . 9 2 \pm 0 . 2 9$ </td><td> $6 5 . 7 2 \pm 0 . 2 4$ </td><td> $6 3 . 5 3 \pm 0 . 4 4$ </td><td> $7 4 . 7 5 \pm 0 . 2 2$ </td><td></td><td> $4 6 . 4 2 \pm 0 . 6 6 4 . 9 4 6 \pm 0 . 0 1 3$ </td><td> $0 . 4 2 4 \pm 0 . 0 0 1$ </td><td> $2 . 8 4 2 \pm 0 . 0 0 8$ </td></tr><tr><td>(f) 50% data</td><td>V+M+B</td><td> $7 6 . 7 4 \pm 0 . 2 0$ </td><td> $6 6 . 8 2 \pm 0 . 1 6$ </td><td>66.17 ± 0.48</td><td>78.27 ± 0.14</td><td></td><td> $5 5 . 5 4 \pm 0 . 6 0 5 . 6 1 2 \pm 0 . 0 1 7$ </td><td>0.448 ± 0.001</td><td> $\underline { { 3 . 1 6 8 } } \pm 0 . 0 0 9$ </td></tr><tr><td>(g) 100% data</td><td>V+M+B</td><td> $7 7 . 6 2 \pm 0 . 0 7$ </td><td> $6 7 . 6 9 \pm 0 . 1 1$ </td><td> $6 6 . 5 2 \pm 0 . 3 5$ </td><td> $7 8 . 4 9 \pm 0 . 3 6$ </td><td></td><td> $5 6 . 2 3 \pm 0 . 4 3 5 . 6 6 4 \pm 0 . 0 1 0$ </td><td> $\overline { { 0 . 4 5 0 } } \pm 0 . 0 0 0$ </td><td> $\mathbf { \overline { { 3 . 2 0 8 } } \pm 0 . 0 0 5 }$ </td></tr><tr><td colspan="10">Training ablation (100% data; compare (g)–(i))</td></tr><tr><td>(h) Joint training</td><td>V+M+B</td><td> $7 9 . 6 8 \pm 0 . 1 2$ </td><td>72.71 ± 0.23</td><td> ${ \bf 6 9 . 7 8 \pm 0 . 2 4 }$ </td><td>79.36 ± 0.21</td><td>56.92 ± 0.21 5.485 ± 0.000</td><td></td><td> $0 . 4 3 9 \pm 0 . 0 0 1$ </td><td>3.100 ± 0.001</td></tr><tr><td>(i) HLR-LLM (Ours) V+M+B 80.51 ± 0.38</td><td></td><td></td><td>76.19 ± 0.11</td><td>67.23 ± 0.37</td><td> $\overline { { 8 0 . 8 7 } } \pm 0 . 2 0$ </td><td>61.50 ± 0.51 4.461 ± 0.004</td><td></td><td> $0 . 4 3 9 \pm 0 . 0 0 1$ </td><td> $3 . 1 0 9 \pm 0 . 0 0 4$ </td></tr></table>

## B Additional results and analysis

## B.1 Ablation study

Tables S1 and S2 provide the full ablation results on HLR-AVSceneQA. The modality ablation shows that binaural audio is essential for spatial grounding. Removing the binaural pathway in (c) substantially degrades egocentric and allocentric grounding as well as spatial relational reasoning. The visual-only variant (b) also performs poorly across these tasks. In contrast, removing monaural audio in (d) has a smaller effect on spatial grounding, but severely degrades far-field speech recognition, highlighting the role of the monaural pathway in preserving semantic audio understanding.

The data-scale ablation in (e)–(g) further disentangles the effect of training data from explicit spatial audio modeling. Even with only 10% of the training data, (e) substantially outperforms the 100%- data model without binaural audio (c) on egocentric grounding and most relational reasoning tasks. Increasing the training data generally improves spatial reasoning, particularly relational reasoning and scene captioning, although some grounding tasks exhibit non-monotonic variation.

The training ablation compares Stage-2 training (g), joint training (h), and our final staged curriculum with CoT tuning (i). Final-stage CoT tuning improves both egocentric and allocentric grounding and achieves the strongest performance on most relational reasoning tasks, with particularly clear gains in correspondence and proximity. The CoT supervision explicitly exposes intermediate steps, such as identifying the sound event, estimating the source location, transforming viewpoints, and comparing source–object relations, which helps the model decompose complex spatial queries before predicting the final answer. We also observe that CoT tuning degrades scene captioning metrics and far-field ASR, suggesting a trade-off between structured multi-step reasoning and tasks that primarily require direct semantic prediction. Overall, we select (i), which uses all modalities and final-stage CoT tuning, as our final model.

Table S3: Zero-shot evaluation on STARSS23. We evaluate direct transfer to real-world spatial audio-visual recordings without any STARSS23 fine-tuning. Sound denotes sound event recognition; direction, distance, and visibility are egocentric. All metrics are reported in accuracy (%).
<table><tr><td>Model</td><td>Sound</td><td>Direction</td><td>Distance</td><td>Visibility</td></tr><tr><td>VideoLLaMA2 [2]</td><td> $3 4 . 5 2 \pm 0 . 3 1$ </td><td> $1 2 . 6 1 \pm 0 . 1 6$ </td><td> $4 . 9 1 \pm 0 . 1 7$ </td><td> $6 1 . 8 0 \pm 0 . 3 1$ </td></tr><tr><td>Video-SALMONN 2 [3]</td><td> $3 5 . 3 0 \pm 0 . 1 1$ </td><td> $1 4 . 3 6 \pm 0 . 1 0$ </td><td> $4 . 8 0 \pm 0 . 2 9$ </td><td> $6 6 . 9 9 \pm 0 . 0 0$ </td></tr><tr><td>Qwen2.5-Omni [4]</td><td> ${ \bf 4 3 . 2 2 \pm 0 . 7 9 }$ </td><td> $1 4 . 6 6 \pm 0 . 7 6$ </td><td> $3 . 6 2 \pm 0 . 1 7$ </td><td> $6 7 . 6 1 \pm 0 . 2 5$ </td></tr><tr><td>HLR-LLM (Ours)</td><td> $3 3 . 9 9 \pm 0 . 3 4$ </td><td> ${ \bf 1 7 . 4 8 \pm 1 . 1 7 }$ </td><td> ${ \bf 1 9 . 7 6 \pm 3 . 4 2 }$ </td><td> ${ \bf 6 8 . 6 1 \pm 0 . 0 9 }$ </td></tr></table>

![](images/89395802fd2179decaaae5c4f4837a4e33cca08f3678a6efc35aa975bc6f1958.jpg)  
Figure S1: Qualitative chain-of-thought reasoning examples from HLR-AVSceneQA. We compare Qwen2.5-Omni with HLR-LLM on proximity and audio-visual correspondence tasks. HLR-LLM generates explicit CoT rationales that localize the sound source, relate it to visible objects, and derive the correct answer.

## B.2 Analysis on the STARSS23 dataset

Table S3 evaluates direct transfer from HLR-AVSceneQA to real-world STARSS23 recordings without target-domain fine-tuning. Despite being trained only on simulated scenes, HLR-LLM outperforms existing AV-LLMs on direction, distance, and visibility prediction, with particularly notable gains in distance estimation. Although sound recognition remains less competitive, these results suggest that spatial grounding capabilities learned from HLR-AVSceneQA can transfer to real-world recordings without target-domain supervision.

## B.3 Qualitative results

We present qualitative chain-of-thought (CoT) examples from HLR-AVSceneQA in Figure S1. The examples focus on spatial relational reasoning, where HLR-LLM explicitly decomposes the problem into sound-source localization and subsequent reasoning over visible objects. In the proximity example, Qwen2.5-Omni selects the drum, whereas HLR-LLM’s CoT first estimates the soundsource location and candidate-object positions, compares their distances to the source, and correctly identifies the child as the closest object. In the correspondence example, Qwen2.5-Omni incorrectly selects a visible saxophone. HLR-LLM’s CoT instead localizes the sounding saxophone to the front-left, compares this position with the visible candidates, and correctly concludes that none of them is spatially aligned with the sound. These examples illustrate how CoT supervision enables HLR-LLM to explicitly connect sound-source localization with downstream scene-level relational reasoning.

![](images/fe022e28333cf9bd9cdab40df35df93c7cf0b02902fe6047580d50d985dca985.jpg)  
Figure S2: Qualitative examples from the STARSS23 real-world evaluation. We show two representative task types, sound-event recognition and visibility prediction, together with the midpoint video frame, annotated sound-source locations, ground-truth answers, and model predictions.

![](images/ad01feaf32331d04fbe4d646b1bcf438e05a64008b1e06c7a5fdfef55533c848.jpg)  
Figure S3: Failure cases. We show representative failure cases where models make fine-grained spatial errors, such as confusing back-left with left, or struggle with multi-source scenes in which sound sources are spatially close.

Figure S2 shows representative STARSS23 examples for sound-event recognition and visibility prediction. HLR-LLM correctly recognizes the target event in multi-source audio and determines source visibility by combining spatial audio with visual evidence. These examples further demonstrate that the learned spatial audio-visual reasoning transfers to real-world recordings.

Failure cases. We also provide failure cases in Fig. S3. Most errors are caused by fine-grained spatial ambiguity. For example, the model may confuse adjacent direction labels, such as back-right vs. right or front-left vs. left, especially when the source lies near a bin boundary. Errors also occur when several candidate objects are spatially clustered, making it difficult to associate the localized sound with the correct object. In multi-source scenes, failures often arise when two sources are close to each other and the non-target source is louder. In such cases, overlapping acoustic cues can dominate the prediction, causing the model to localize or reason about the wrong source.

![](images/22f6e093ddaee18f85bd9ef7c53f96ebea1c8ac756a967b5d0eab3bad5f49c02.jpg)  
Ground Truth Annotation Event: dog barking Initial direction: front-left Initial distance: 0.9m Target location: radiator Target distance: 3.9m Answer: quiter

## Predicted CoT

The audio event is dog barking. It originates from the front-left direction and is 0.9 meters away. Once you move to radiator, the source is 3.8 away, making you farther from it than before. Therefore, the sound becomes guieter Final answer: quieter

![](images/85fbbea8f9f9b0bf726093fc3091ab27029aed2da5628b288946031862929bfe.jpg)  
Ground Truth Annotation Event: dog barking Initial direction: front Initial distance: 4.1m Target location: step Target distance: 0.9m Answer: louder

## Predicted CoT

The audio event is dog barking. It originates from the front direction and is 4.1 meters away. Once you move to step, the source is 1.6 away, making you closer to it than before. Therefore, the sound becomes louder Final answer: louder

Figure S4: Counterfactual analysis with conflicting audio-visual cues. The image is kept fixed with the dog visible at its original location, while the binaural source position is changed. HLR-LLM follows the counterfactual spatial audio and correspondingly updates its localization and downstream loudness reasoning.  
Question: Which direction is the sound source coming from?  
![](images/c1fcff9b6479db81b5885705a61609195d7f0f2fda47f4b415cebd2caa5e34a0.jpg)  
(a) Original image

![](images/7cfa04a20fa23e92698d6c45d4acb4a1adc7050df8f66ed3344d5e737c9f7c78.jpg)  
(b) Visual attention

![](images/6cff044d7289f68abbd6ab34092ded8def382ffeade5f9a7b861931823757c14.jpg)  
(c) Audio attention

![](images/7c33d5fc0c7fcdd2472ced4726046ddab50c5fd70f17f658a9b780ae584383b9.jpg)  
(d) Spatial audio attention  
Figure S5: Modality-specific attention visualization. From left to right: input image, visualto-visual, audio-to-visual, and spatial-audio-to-visual attention from the 18th transformer layer. Spatial-audio tokens attend more selectively to the ground-truth sound-source regions (red boxes).

Counterfactual analysis. We further conduct a counterfactual analysis to examine whether HLR-LLM relies on spatial audio cues when visual and auditory evidence conflict. We keep the rendered image fixed, where the dog remains visible at its original location, and replace the binaural audio with dog barking rendered from different spatial positions. As shown in Fig. S4, HLR-LLM follows the counterfactual spatial audio in both cases. In the left case, it localizes the barking to the front-left at 0.9 m and infers that moving to the radiator increases the source distance, resulting in a quieter prediction. In the right case, it instead localizes the barking to the front at 4.1 m and predicts that moving to the step brings the agent closer to the source, resulting in a louder prediction, even though the visible dog remains at its original front-left location. These results provide evidence that HLR-LLM uses spatial audio for localization and subsequent spatial reasoning rather than relying solely on the visible source position.

Modality-specific attention analysis. To examine how modality-specific representations interact with visual content, we extract self-attention weights from the 18th transformer layer of HLR-LLM. For each modality, we use its corresponding tokens as queries and the visual tokens as keys, averaging the attention weights across all query tokens and attention heads. We visualize three modality-to-visual attention maps: visual-to-visual using $Z _ { \mathrm { v i s } } .$ , audio-to-visual using $Z _ { \mathrm { a u d } }$ , and spatial-audio-to-visual using $Z _ { \mathrm { s p a } }$ . The visual-to-visual map reflects regions emphasized by the visual representation, while the latter two indicate which image regions are attended to by the conventional audio and spatial-audio representations, respectively. As shown in Fig. S5, spatial-audio tokens attend more selectively to the ground-truth sound-source regions than conventional audio tokens, providing qualitative evidence that the spatial pathway captures localization-relevant visual associations.

## C Details on HLR-AVSceneQA

We provide implementation-level details on HLR-AVSceneQA, including the dataset construction pipeline, task characteristics, question and answer formats, quality-control procedures, and additional details that were not introduced in the main paper.

## C.1 Scene initialization, agent placement, and scene generation

Simulator setup. We generate each sample in Habitat-Sim with acoustic rendering using the SoundSpaces simulator [16]. The agent is initialized with synchronized RGB, semantic, depth, panoramic RGB, and binaural audio sensors. The RGB sensor is a pinhole camera used as the main visual input. The semantic and depth sensors are used for visibility checking and geometric supervision, while the panoramic RGB sensor is used for visualization and debugging. The binaural audio sensor is co-located with the visual agent and uses a two-channel receiver at a 32 kHz sampling rate with material-aware indirect propagation. We set the sensor height to 150 cm and the camera field of view to 90<sup>◦</sup>.

We select valid rooms from HM3D scenes [39] by checking that the semantic region contains a floor, a valid room-level region of interest, and sufficient navigable positions. In the standard setting, we place the agent near a room corner and orient it toward the room center, which increases the chance that inserted objects and room semantics are visible. Since this placement may bias sound sources toward the front view, we also sample agent poses closer to the room center. This makes it possible to construct scenes in which one reference object remains visible while another object or sound source falls outside the camera view.

Coordinate system and azimuth definition. We use Habitat coordinates for camera, agent, and source placement, and convert positions into metric distances when constructing QA labels. For azimuth, we annotate angles around the horizontal plane, ranging from 0 to $3 6 0 ^ { \circ }$ . In our setup, 0<sup>◦</sup> indicates the front direction, 90<sup>◦</sup> indicates the right direction, $\mathrm { \check { 1 } 8 0 } ^ { \circ }$ indicates the back direction, and 270<sup>◦</sup> indicates the left direction relative to the agent. We do not consider elevation in this manuscript, as human spatial-audio localization is generally known to be more accurate and robust in azimuth than in elevation [57].

For both direction and distance, we discretize continuous spatial values into bins to make training and evaluation more stable. For azimuth, we use 8 bins, each covering $4 5 ^ { \circ }$ . Angles increase clockwise from straight ahead: front corresponds to 337.5<sup>◦</sup>–22.5<sup>◦</sup>, right to $6 7 . 5 ^ { \circ } - \bar { 1 1 } 2 . 5 ^ { \circ }$ , back to $1 5 7 . 5 ^ { \circ } -$ 202.5<sup>◦</sup>, and left to 247.5<sup>◦</sup>–292.5<sup>◦</sup>. For distance, we use 10 bins, with a 1-meter interval for 0–1 m, 0.5-meter intervals from 1 to 5 m, and a final bin for distances beyond 5 m. The full bin definitions are shown in Table S4. This discretization reduces ambiguity from small geometric differences and facilitates stable training and evaluation.

3D object meshes and scene insertion. We insert 3D object meshes into HM3D rooms to synthesize controlled layouts containing both sounding and non-sounding objects. The object pool is built from curated meshes, including Objaverse-XL assets [43], as well as generated meshes for sound categories whose object coverage is insufficient. For generated sounding-object meshes, we first use Stable Diffusion 2 [44] for text-to-image synthesis, and then use TRELLIS [45] for image-to-3D mesh generation. The size and brightness of each object category are manually determined to ensure plausible rendering in indoor scenes. Examples are shown in Fig. S6.

When inserting objects into a room, we reject placements that collide with room geometry, overlap existing objects above a horizontal IoU threshold, or violate minimum agent-to-object or objectto-object distance constraints. These filtering steps help ensure that inserted objects are physically plausible and visually distinguishable.

Binaural audio generation. For each active source, we render binaural audio using SoundSpaces 2.0. Let $x _ { j } [ t ]$ be the dry monaural waveform assigned to source $j ,$ located at ${ \mathbf { s } } _ { j } .$ , and let the agent receiver be at position r with heading θ. SoundSpaces produces a channel-wise binaural room impulse response:

$$
h _ { m , j } [ \tau ] = \mathcal { R } _ { m } ( \tau ; \mathbf { s } _ { j } , \mathbf { r } , \theta , \mathcal { E } ) , \qquad m \in \{ L , R \} ,\tag{1}
$$

Table S4: Azimuth and distance bins used in HLR-AVSceneQA. Azimuth angles increase clockwise from the agent’s forward direction.
<table><tr><td>Type</td><td>Bin</td><td>Range</td></tr><tr><td rowspan="6">Azimuth</td><td>front front-right</td><td> $3 3 7 . 5 ^ { \circ } - 2 2 . 5 ^ { \circ }$   $2 2 . 5 ^ { \circ } { - } 6 7 . 5 ^ { \circ }$ </td></tr><tr><td>right</td><td> $6 7 . 5 ^ { \circ } - 1 1 2 . 5 ^ { \circ }$ </td></tr><tr><td>back-right</td><td> $1 1 2 . 5 ^ { \circ } - 1 5 7 . 5 ^ { \circ }$ </td></tr><tr><td>back</td><td> $1 5 7 . 5 ^ { \circ } - 2 0 2 . 5 ^ { \circ }$ </td></tr><tr><td>back-left</td><td> $2 0 2 . 5 ^ { \circ } - 2 4 7 . 5 ^ { \circ }$ </td></tr><tr><td>left</td><td>247.5°–292.5°</td></tr><tr><td rowspan="9">Distance</td><td>front-left</td><td></td></tr><tr><td></td><td>292.5°–337.5°</td></tr><tr><td>A</td><td>0-1.0 m</td></tr><tr><td>B</td><td>1.0-1.5 m</td></tr><tr><td>C</td><td>1.5–2.0 m</td></tr><tr><td>D</td><td>2.0–2.5 m</td></tr><tr><td>E</td><td>2.5–3.0 m</td></tr><tr><td>F</td><td>3.0–3.5 m</td></tr><tr><td></td><td></td></tr><tr><td rowspan="6"></td><td>G</td><td>3.5-4.0 m</td></tr><tr><td>H</td><td>4.0-4.5 m</td></tr><tr><td></td><td></td></tr><tr><td>I</td><td>4.5–5.0 m</td></tr><tr><td>J</td><td>&gt; 5.0 m</td></tr></table>

![](images/2c0e0d58f3a2cd851961f07f1c6e201dcba55908d9ad0be5b348a44aae37115c.jpg)  
Figure S6: Examples of 3D object meshes. The first three rows show meshes retrieved from Objaverse-XL, while the last three rows show meshes generated by generative models.

where E denotes the scene geometry and material configuration. The binaural signal is then obtained by convolving the normalized dry waveform $\tilde { x } _ { j }$ with the corresponding room impulse response:

$$
y _ { m , j } [ t ] = ( h _ { m , j } * \tilde { x } _ { j } ) [ t ] = \sum _ { \tau = 0 } ^ { T _ { h } - 1 } h _ { m , j } [ \tau ] \tilde { x } _ { j } [ t - \tau ] .\tag{2}
$$

For multiple active sources, the rendered signals are summed:

$$
y _ { m } [ t ] = \sum _ { j = 1 } ^ { N } y _ { m , j } [ t ] , \qquad m \in \{ L , R \} .\tag{3}
$$

![](images/a70b6bc156307a09bad84db8c881985614da9e8df9b2512b8b1b51496b47579a.jpg)  
(a) Normal filed-of-view (model input)

![](images/6c654bbc81e08c1905d5a4b5d08e226902884c4abd9ce9ac1eb427d7c7700d5f.jpg)  
(b) Bird's-eye-view  
Figure S7: Example illustration. This example data sample is used to facilitate understanding of our task definitions and details.

The final stereo waveform $\mathbf { y } [ t ] = [ y _ { L } [ t ] , y _ { R } [ t ] ]$ is saved at a 32 kHz sampling rate.

Spatial reasoning scenarios. We intentionally include several scenarios that require audio-visual spatial reasoning beyond semantic matching. Occluded sound sources place the active source behind a visible object or behind a wall, so the source may project into the camera view while having no visible semantic pixels. Ambiguous sound sources among same-category instances include multiple objects of the same category, requiring the model to determine which instance is spatially aligned with the sound. Semantically mismatched but spatially aligned cases attach a sound event to an object that does not semantically match the event, preventing models from relying only on category correspondence. Adjacent-room sound sources move the source into a neighboring room connected through the scene geometry, creating cases where the source is not visible but can still be heard from a distant location.

## C.2 Task definitions and details

Here we provide additional details on the task definitions, their descriptions, and the intention of each task, using the example in Fig. S7. In this scenario, Fig. S7 (a) shows the agent’s view, which is the model input with a normal field of view. Several instances are inserted, including a dog, a piano, a baby, and two alarm clocks. There are two sound events: one is a man speaking, saying “Turn off the alarm,” coming from the agent’s front-left direction and originating from an adjacent room. The other is an alarm ringing from the blue alarm clock, coming from the front direction, 3.4 meters away. Based on this example, we describe the goal and intended evaluation behavior of each task.

Furthermore, to improve linguistic diversity and reduce overfitting to fixed question formats, we generate paraphrased templates for each task. We create approximately 15 paraphrases per task and show three representative prompts below. The full set of templates will be included in the dataset release.

Spatial hearing: sound event recognition. This task evaluates semantic audio perception. The model is asked to identify the audible sound event from a set of candidate sound categories. A total of 10 candidate sound categories are provided as options. The target sound category is always included, while distractors are sampled from semantically related categories, sound categories associated with scene objects, and random categories. When multiple sources are present, the prompt may additionally specify the source direction and distance to indicate which source should be recognized.

• Q: What sound do you hearfrom thefront, 3.4 meters away?

• A: alarm clock ringing.

• Paraphrase 1: Which sound is coming from the front, 3.4 meters away?

• Paraphrase 2: Identify the sound eventfrom thefront direction at 3.4 meters.

Spatial hearing: far-field automatic speech recognition. This task evaluates far-field automatic speech recognition (ASR) from spatialized audio. Since speech may contain important information that the agent should understand, the model is required to recognize speech as part of spatial hearing. Because speech can originate from both nearby and distant locations, far-field ASR is more challenging than clean or near-field ASR. We ask the model to return only the exact quoted transcription, without any additional explanation.

• Q: Transcribe what the person said. Output only the transcription inside quotes.

• A: “Turn off the alarm.”

• Paraphrase 1: Write the exact phrase spoken. Return only the quoted transcription.

• Paraphrase 2: Provide the exact spoken words inside quotes.

Spatial grounding: egocentric direction prediction. This task evaluates egocentric sound-source direction estimation. The answer is computed from the vector pointing from the agent to the sound source, with its azimuth measured relative to the agent’s current forward direction. We ask the model to answer using a discretized 8-bin azimuth label as shown in Table S4. If multiple sources are present, we specify the queried sound source.

• Q: From this camera perspective, which direction bin does the source of alarm clock ringing fall into?

• A: front.

• Paraphrase 1: Based on the current view, which direction is the alarm clock ringing coming from?

• Paraphrase 2: From this viewpoint, identify the direction bin for the source of alarm clock ringing.

Spatial grounding: egocentric distance prediction. This task evaluates egocentric source-distance estimation. The answer is computed as the horizontal distance between the agent and the active sound source, and is then converted into a discrete distance bin, as shown in Table S4. If multiple sources are present, we specify the queried sound source.

• Q: From this camera perspective, what is the distance to the source of alarm clock ringing?

• A: F.

• Paraphrase 1: Based on the current view, how far away is the source of alarm clock ringing?

• Paraphrase 2: From this viewpoint, select the distance bin for the source of alarm clock ringing.

Spatial grounding: allocentric direction prediction. This task evaluates direction reasoning from a different viewpoint. The prompt asks the model to imagine standing at a reference object and facing another object, as shown in Fig. S7 (b). The source direction is then recomputed in this new coordinate frame. It is intended to evaluate whether the model can reason from a different perspective and understand the 3D spatial structure of the current scene. The task is generated only when valid visible reference and facing objects are available. We avoid prompts where the reference and facing objects have identical final category names, which would make the instruction ambiguous. As in the egocentric case, the answer is a discretized 8-bin azimuth label. If multiple sources are present, we specify the queried sound source.

• Q: If you stand by the piano facing the baby, which direction is the source of alarm clock ringing coming from?

• A: front-left.

• Paraphrase 1: Standing next to the piano and facing the baby, which direction does the alarm clock ringing comefrom?

• Paraphrase 2: From a position at the piano facing the baby, identify the direction bin of the alarm clock ringing.

Spatial grounding: allocentric distance prediction. This task evaluates distance reasoning from a different viewpoint. Similar to allocentric direction prediction, the prompt asks the model to imagine standing at a reference object and facing another object. The source distance is then recomputed from the reference object’s position. This task evaluates whether the model can reason about source distance from a different location in the scene. We avoid prompts where the reference and facing objects have identical final category names, which would make the instruction ambiguous. As in the egocentric case, the answer is a discretized 10-bin distance label. If multiple sources are present, we specify the queried sound source.

• Q: If you stand by the piano facing the baby, what is the distance to the source of alarm clock ringing?

• A: B.

• Paraphrase 1: Standing next to the piano and facing the baby, how far away is the alarm clock ringing?

• Paraphrase 2: From a position at the piano facing the baby, select the distance bin for the source of alarm clock ringing.

Spatial relational reasoning: visibility. This task evaluates whether the model can determine if the sound source is visible from the current viewpoint. This task becomes difficult when the scene contains the same or similar categories, but the actual sound comes from outside the current viewpoint. For example, one may incorrectly assume that the man speaking sound in Fig. S7 comes from the baby, since the baby is also human-related. However, with sufficient spatial audio perception, the model should distinguish that the baby is not making the sound and that the sound source is not visible. The ground truth is computed from semantic rendering: a source is considered visible if the number of semantic pixels corresponding to the source object exceeds a configured threshold. If multiple sources are present, we specify the queried sound source.

• Q: From the current camera perspective, is the source of man speaking visible?

• A: no.

• Paraphrase 1: Based on the current view, can you see the source of man speaking?

• Paraphrase 2: From this viewpoint, is the man speaking source in view?

Spatial relational reasoning: correspondence. This task evaluates audio-visual spatial correspondence. The model is asked to identify which visible object is spatially aligned with the sound source. The correct answer may be semantically different from the sound category, since the task focuses on spatial alignment rather than category matching. This task becomes challenging when multiple visible objects exist, as in Fig. S7. Without spatial audio understanding, the model may confuse whether the yellow alarm clock is ringing. If no visible object corresponds to the sound source, the answer is None. When multiple visible inserted objects share the same category, we disambiguate them using orientation labels, such as “alarm clock on thefront-right.” If multiple sources are present, we specify the queried sound source.

• Q: From the current camera perspective, which object is spatially aligned with the sound of alarm clock ringing?

• A: alarm clock infront.

• Paraphrase 1: Which visible object corresponds to where the alarm clock ringing comes from?

• Paraphrase 2: Which object best matches the spatial origin of the alarm clock ringing?

Spatial relational reasoning: hidden-source localization. This task evaluates reasoning about non-visible sound sources. It is asked when the source is not reliably visible in the current image. The answer includes five options: hidden behind the wall, hidden behind a specific object, in an adjacent room, out of the field of view in front, and out of the field of view behind the agent. This task goes one step beyond visibility prediction. The model must reason where the source may be if it is not visible. It should decide whether the sound comes from a distant place, such as another room, or whether it should be in the field of view but is occluded by an object, which requires accurate visual perception. The ground-truth label is determined using visibility, projection, adjacent-room metadata, and occluder information. If multiple sources are present, we specify the queried sound source.

• Q: You hear a man speaking. Where is the source ofthis sound?

• A: in an adjacent room.

• Paraphrase 1: Given the sound of man speaking, where is its source?

• Paraphrase 2: A man speaking is audible. Where could the source be situated?

Spatial relational reasoning: loudness change. This task evaluates relative acoustic reasoning under hypothetical movement. The prompt asks whether the sound would become louder or quieter if the agent moved to a specified object. Here, louder or quieter is defined as a controlled relative movement judgment based on whether the source becomes closer or farther, rather than an estimate of absolute perceived loudness. This assesses whether the model can decide whether a movement would bring the agent closer to or farther from the sound source. It also goes one step beyond allocentric distance prediction. The model must estimate the source distance from the changed location and compare it with the current source distance. The label is computed by comparing the current agent-to-source distance with the candidate-object-to-source distance. If the candidate object is closer to the source than the current agent position, the answer is louder; otherwise, it is quieter. If multiple sources are present, we specify the queried sound source.

• Q: If you move to the piano, will the sound of man speaking become louder or quieter?

• A: louder.

• Paraphrase 1: When at the piano, will the sound ofman speaking be louder or quieter?

• Paraphrase 2: Upon moving to the piano, does the man speaking become louder or quieter?

Spatial relational reasoning: proximity reasoning. This task goes one step beyond loudness change and evaluates source-to-object spatial reasoning. The model is given a set of candidate objects and must select the one closest to the sound source. This requires comparing the source distance to each candidate object, similar to evaluating loudness change for each object, and then ranking the candidates to select the closest one. The ground truth is computed using the Euclidean distance from the source to each candidate object. As with other distance-based relational tasks, we exclude adjacent-room sources to avoid cases where direct Euclidean geometry does not reflect the effective acoustic propagation path. Tied cases are skipped to avoid ambiguous supervision. If multiple sources are present, we specify the queried sound source.

• Q: Which object is closest to the source of alarm clock ringing?

• A: bed.

• Paraphrase 1: To which object is the alarm clock ringing nearest?

• Paraphrase 2: Which object is at the shortest distance from the source of alarm clock ringing?

Spatial reasoning: scene captioning. This task evaluates compositional spatial audio-visual scene description. The model is asked to describe each sound source using complete sentences and include the sound event, direction, visibility, and hidden-source location when applicable. The prompt explicitly instructs the model not to describe unrelated scene content. When multiple sources are present, the model must provide a separate description for each source.

• Q: Describe only the sound source or sound sources.

• A: The sound event is alarm clock ringing. It is coming from the front direction. The sound source is visible in the scene.

• Paraphrase 1: Summarize only the sound source information.

• Paraphrase 2: For each source, state the sound event, direction, visibility, and hidden location if applicable.

## C.3 Chain-of-thought supervision

We additionally construct chain-of-thought-style supervision to encourage the model to connect audio perception, visual grounding, and spatial reasoning before producing the final answer. Both the final answers and CoT rationales are deterministically derived from structured simulator metadata through task-specific templates rather than free-form generation. GPT-4 [46] is used only to paraphrase question wording for linguistic diversity. This is especially important for tasks that cannot be solved from a single cue alone, such as allocentric reasoning, hidden-source localization, loudness change, and proximity reasoning. In these tasks, the model must first identify the sound event, estimate its direction or distance, relate it to visible or non-visible objects, and then convert this intermediate spatial evidence into the requested answer. We keep the reasoning brief and require the model to end with a normalized final-answer line.

## Spatial grounding: allocentric direction prediction.

• Q: If you are located at the cabinet facing the door frame, determine which direction the source ofwoman singing is comingfrom. Reason briefly, then end with exactly one line: “Final answer: <8bin\_direction\_label>”.

• A: The audio event woman singing is heardfrom thefront direction at 1.2 m.

The cabinet is positionedfront-left at a distance of2.6 mfrom the current location.

After moving to cabinet while maintaining the same orientation, the sound source appears at right with a distance of2.0 m.

From this position, doorframe isfound in thefront-right direction at 3.5 m.

Rotate right toface doorframe, establishing the newforward axis.

In this new frame, the sound source lies front-right.

Final answer: front-right

## Spatial grounding: allocentric distance prediction.

• Q: Standing next to the cabinet andfacing the typewriter, what is the distance to the sound source? Reason briefly, then end with exactly one line: “Final answer: <distance\_bin>”.

• A: The sound event is playing acoustic guitar, initially at 1.0 m in the right direction.

After moving to cabinet, the sound source is approximately 1.2 m away.

This measurement places it within the 1-1.5 m interval.

Final answer: B

## Spatial relational reasoning: visibility.

• Q: From this perspective, is the sound source in sight? Answer with [yes, no]. Reason briefly, then end with exactly one line: “Final answer: <yes\_or\_no>”.

• A: The audio event is playing clarinet.

It is coming from the back direction and is 1.0 meters away.

In that direction, the clarinet is not present, so the sound source cannot be seen. Thus, the source is not visible.

Final answer: no

## Spatial relational reasoning: correspondence.

• Q: From the camera’s perspective, which object is at the location of the sound of mouse squeaking? Pick one option; select None ifno object in view is producing sound. Reason briefly, then end with exactly one line: “Final answer: <object\_or\_None>”.

• A: The audio event is mouse squeaking.

It is coming from the left direction and is 1.6 meters away.

Among the options, no visible source in the left direction matches the sound.

Final answer: None

## Spatial relational reasoning: hidden-source localization.

• Q: The sound of woman speaking is present. Where is it originating from? Choose one from: (A)–(E). Reason briefly, then end with exactly one line: “Final answer: <hidden\_source\_option>”.

• A: The audio event is woman speaking.

Although it comesfrom the right direction, the source lies outside the currentfield ofview in that direction.

Therefore, the source is out of the field of view in front.

Final answer: out of the field of view in front

## Spatial relational reasoning: loudness change.

• Q: By moving to pillow, does the sound become louder or quieter? Choose one from: [louder, quieter]. Reason briefly, then end with exactly one line: “Final answer: <louder\_or\_quieter>”.

• A: The audio event is hair dryer drying.

It originates from the front direction and is 4.0 meters away.

Once you move to pillow, the source is 2.8 m away, making you closer to it than before.

Therefore, the sound becomes louder.

Final answer: louder

## Spatial relational reasoning: proximity reasoning.

• Q: Which object is closest to the sound source? Select onefrom: [bed, vacuum cleaner, chair, air conditioning]. Reason briefly, then end with exactly one line: “Final answer: <object\_label>”.

• A: The audio event is man speaking.

It originates from the back-right direction and is 0.7 meters away.

Agent-centered candidate layout: bed is front-right at 1.6 m from the agent, vacuum cleaner is front-right at 3.0 mfrom the agent, chair isfront at 3.4 mfrom the agent, and air conditioning is front at 2.0 mfrom the agent.

Comparing source-to-candidate distances gives: bed is 1.4 m awayfrom the sound source, vacuum cleaner is 3.1 m awayfrom the sound source, chair is 3.7 m awayfrom the sound source, and air conditioning is 2.4 m away from the sound source.

Therefore, the nearest object is bed.

Final answer: bed

## C.4 Quality control

Automatic quality control. We apply several filtering and retry rules to avoid noisy supervision. First, rooms are skipped if they do not have a valid floor, contain too few insertion slots, or lack enough valid navigable points. Second, object placements are rejected if they collide with room geometry, overlap existing objects above the IoU threshold, violate minimum distance constraints, or fail semantic visibility checks. Third, generated 3D meshes and assets from Objaverse-XL are manually inspected by human annotators. We retain only sufficiently high-quality objects, filtering out low-quality meshes and implausible categories. We also find that, in some cases, sounding objects, reference objects, facing objects, or candidate objects used in task options can be too small to be clearly visible. We therefore apply a pixel-count threshold and remove QA pairs that ask about objects that are hardly visible. Finally, proximity questions with tied nearest objects are skipped.

Human verification. We additionally conduct a human verification study on 150 samples, covering 1,144 QA pairs, with three human annotators. Annotators verify whether the target sound is audible, whether its approximate spatial location can be identified, whether candidate and reference objects mentioned in the questions are observable, and whether the annotated source visibility and hiddensource location are consistent with the audio-visual scene. As shown in Table S5, 1,133 of 1,144 QA pairs pass human verification, corresponding to an overall pass rate of 99.0%. All spatial-hearing and egocentric-grounding questions achieve a 100% pass rate, while the remaining task categories achieve pass rates between 96.8% and 100%. These results provide human validation that our metadata- and template-based construction pipeline produces reliable spatial audio-visual QA annotations across the benchmark tasks.

## C.5 Dataset statistics

Table S6 reports the number of generated QA pairs for the training and test splits, including the number of CoT supervision examples for each task and the breakdown of QA pairs constructed from single-source and multi-source scenes. Table S7 further characterizes the benchmark composition by reporting the distributions of hidden-source and visibility cases, together with train–test overlap statistics. The hidden-source test set covers diverse non-visible configurations, including off-screen, adjacent-room, wall-occluded, and object-occluded sources. The visibility task contains 53.6% visible and 46.4% non-visible cases in training, and 30.7% visible and 69.3% non-visible cases in the test set. Importantly, the training and test splits use disjoint environments and rooms, as well as disjoint object mesh instances, while all test sound and object categories are represented in the training split. Thus, the benchmark evaluates generalization to unseen environments, scene layouts, and object instances while maintaining consistent semantic task coverage.

Fig. S8 and Fig. S9 further visualize the distributions of distance and direction bins for egocentric and allocentric settings in both the training and test sets. Finally, Fig. S10 and Fig. S11 show the top 30 inserted object categories and active sound-source categories in the training and test sets.

Table S5: Human verification of HLR-AVSceneQA annotations. Three human annotators verify 1,144 QA pairs from 150 sampled scenes. We report the number of passed and failed QA pairs and the corresponding pass rate for each task.
<table><tr><td>Task family</td><td>Task</td><td>Total</td><td>Passed</td><td>Failed</td><td>Pass rate</td></tr><tr><td rowspan="2">Spatial hearing</td><td>Sound event recognition</td><td>112</td><td>112</td><td>0</td><td>100.0%</td></tr><tr><td>Far-field ASR</td><td>29</td><td>29</td><td>0</td><td>100.0%</td></tr><tr><td rowspan="2">Egocentric grounding</td><td>Direction</td><td>98</td><td>98</td><td>0</td><td>100.0%</td></tr><tr><td>Distance</td><td>98</td><td>98</td><td>0</td><td>100.0%</td></tr><tr><td rowspan="2">Allocentric grounding</td><td>Direction</td><td>148</td><td>146</td><td>2</td><td>98.6%</td></tr><tr><td>Distance</td><td>148</td><td>148</td><td>0</td><td>100.0%</td></tr><tr><td rowspan="5">Relational reasoning</td><td>Visibility</td><td>107</td><td>105</td><td>2</td><td>98.1%</td></tr><tr><td>Hidden source</td><td>66</td><td>65</td><td>1</td><td>98.5%</td></tr><tr><td>Loudness change</td><td>111</td><td>109</td><td>2</td><td>98.2%</td></tr><tr><td>Proximity</td><td>126</td><td>122</td><td>4</td><td>96.8%</td></tr><tr><td>Correspondence</td><td>101</td><td>101</td><td>0</td><td>100.0%</td></tr><tr><td>Overall</td><td>一</td><td>1,144</td><td>1,133</td><td>11</td><td>99.0%</td></tr></table>

Table S6: Dataset statistics by task split. Each entry reports the number of QA pairs. For Train and Test, Single and Multi are computed from the source-count annotations. For Train-CoT, Single and Multi report the number of CoT QA pairs in each source-count group after task-specific filtering.
<table><tr><td rowspan="2">Task family</td><td rowspan="2">Task</td><td colspan="3">Train</td><td colspan="3">Train-CoT</td><td colspan="3">Test</td></tr><tr><td>Single</td><td>Multi</td><td>Total</td><td>Single</td><td>Multi</td><td>Total</td><td>Single</td><td>Multi</td><td>Total</td></tr><tr><td rowspan="2">Spatial hearing</td><td>Sound event</td><td>14,794</td><td>7,054</td><td>21,848</td><td>一</td><td>一</td><td></td><td>1,498</td><td>772</td><td>2,270</td></tr><tr><td>Far-field ASR</td><td>2,708</td><td>867</td><td>3,575</td><td>一</td><td>一</td><td>1</td><td>380</td><td>128</td><td>508</td></tr><tr><td rowspan="4">Spatial grounding</td><td>Egocentric direction</td><td>13,770</td><td>6,500</td><td>20,270</td><td></td><td>一</td><td></td><td>1,442</td><td>632</td><td>2,074</td></tr><tr><td>Egocentric distance</td><td>13,770</td><td>6,500</td><td>20,270</td><td></td><td>1</td><td>1</td><td>1,442</td><td>632</td><td>2,074</td></tr><tr><td>Allocentric direction</td><td>8,410</td><td>3,909</td><td>12,319</td><td>8,086</td><td>3,735</td><td>11,821</td><td>1,454</td><td>461</td><td>1,915</td></tr><tr><td>Allocentric distance</td><td>8,410</td><td>3,909</td><td>12,319</td><td>8,086</td><td>3,735</td><td>11,821</td><td>1,454</td><td>461</td><td>1,915</td></tr><tr><td rowspan="6">Spatial relational reasoning</td><td>Visibility</td><td>11,277</td><td>5,350</td><td>16,627</td><td>11,277</td><td>5,350</td><td>16,627</td><td>1,644</td><td>410</td><td>2,054</td></tr><tr><td>Correspondence</td><td>14,794</td><td>7,054</td><td>21,848</td><td>14,794</td><td>7,054</td><td>21,848</td><td>2,066</td><td>268</td><td>2,334</td></tr><tr><td>Hidden-source</td><td>5,173</td><td>2,693</td><td>7,866</td><td>5,173</td><td>1,321</td><td>6,494</td><td>1,094</td><td>382</td><td>1,476</td></tr><tr><td>Loudness change</td><td>12,542</td><td>5,860</td><td>18,402</td><td>12,542</td><td>5,860</td><td>18,402</td><td>1,164</td><td>291</td><td>1,455</td></tr><tr><td>Proximity</td><td>5,166</td><td>2,391</td><td>7,557</td><td>5,166</td><td>2,340</td><td>7,506</td><td>720</td><td>248</td><td>968</td></tr><tr><td>Scene captioning</td><td>10,968</td><td>1,192</td><td>12,160</td><td></td><td></td><td></td><td>2,581</td><td>184</td><td>2,765</td></tr><tr><td>Total</td><td>一</td><td>121,782</td><td>53,279</td><td>175,061</td><td>65,124</td><td>29,395</td><td>94,519</td><td>16,939</td><td>4,869</td><td>21,808</td></tr></table>

## D Broader impact

HLR-AVSceneQA and HLR-LLM aim to support spatially aware audio-visual understanding for embodied AI, assistive agents, robotics, and accessibility applications. The ability to recognize what sound is present, localize where it originates, and reason about how it relates to the current scene is especially important for agents that navigate indoor environments and assist in daily situations. For example, such systems may help localize alarms, identify speakers, or reason about events outside the camera view. Our benchmark is simulator-based and does not contain private real-world recordings. However, future deployment of spatial audio-visual systems should follow strict privacy, consent, and data-minimization practices, especially when operating in real homes or shared indoor spaces.

## E LLM usage

We use LLMs in three ways. First, the proposed HLR-LLM is built on a large language model as the reasoning module that integrates unified inputs, including visual, monaural audio, and binaural spatial audio tokens. Second, we use an LLM to paraphrase question templates to improve linguistic diversity and reduce the risk of models learning fixed template patterns. Finally, we use an LLM to assist with editing and grammar checking, table refinement (Table S4), figure generation and refinement (Fig. S8, S9, S10, S11), and development of the data-viewer web page.

Table S7: Additional dataset statistics and train–test overlap. We report label distributions for hidden-source and visibility tasks, followed by statistics on the train and test splits. Percentages are computed within each corresponding task split. Train and test environments, rooms, and object mesh instances are disjoint, while test semantic categories are covered by the training split.
<table><tr><td>Statistic</td><td>Train</td><td>Test</td><td>Train-Test Overlap</td></tr><tr><td colspan="4">Hidden-source distribution</td></tr><tr><td>Off-screen (front)</td><td>2,681 (34.1%)</td><td>408 (27.6%)</td><td></td></tr><tr><td>Off-screen (back)</td><td>2,752 (35.0%)</td><td>518 (35.1%)</td><td></td></tr><tr><td>Adjacent room</td><td>959 (12.2%)</td><td>237 (16.1%)</td><td></td></tr><tr><td>Hidden behind wall</td><td>418 (5.3%)</td><td>123 (8.3%)</td><td></td></tr><tr><td>Hidden behind object</td><td>1,056 (13.4%)</td><td>190 (12.9%)</td><td></td></tr><tr><td>Total</td><td>7,866</td><td>1,476</td><td></td></tr><tr><td colspan="4">Visibility distribution</td></tr><tr><td>Visible</td><td>8,912 (53.6%)</td><td>631 (30.7%)</td><td></td></tr><tr><td>Not visible</td><td>7,715 (46.4%)</td><td>1,423 (69.3%)</td><td></td></tr><tr><td>Total</td><td>16,627</td><td>2,054</td><td></td></tr><tr><td colspan="4">Split statistics and overlap</td></tr><tr><td>Environments</td><td>145</td><td>36</td><td>0</td></tr><tr><td>Rooms</td><td>1,028</td><td>241</td><td>0</td></tr><tr><td>Sound categories</td><td>108</td><td>79</td><td>79</td></tr><tr><td>Object categories</td><td>37</td><td>37</td><td>37</td></tr><tr><td>Object mesh instances</td><td>3,018</td><td>1,299</td><td>0</td></tr></table>

## F Licenses and terms of existing assets

We use existing datasets, simulators, models, and 3D assets only under their stated licenses and terms of use.

• SoundSpaces 2.0: used for spatial acoustic rendering. SoundSpaces is released under the Creative Commons Attribution 4.0 International license. Matterport3D-based task data and trained models follow the Matterport3D terms and CC BY-NC-SA 3.0 US license.

• HM3DSem / HM3D: used as the source of indoor 3D environments and semantic scene annotations. We use the dataset under its academic research terms and respect the restrictions of the underlying Matterport3D assets.

• VGGSound: used for non-speech sound events. We use the official split and retain only objectgrounded categories suitable for indoor spatial reasoning. Copyright remains with the original video owners.

• LibriSpeech: used for speech audio and far-field ASR supervision. LibriSpeech is based on public-domain audiobooks and is distributed under a Creative Commons license.

• Objaverse-XL: used as a source of 3D object meshes. The dataset as a whole is licensed under ODC-By v1.0, while individual objects have their own licenses. We manually check and retain only assets compatible with our release setting.

• Stable Diffusion 2: used to generate canonical object images. The code is released under the MIT license, and model weights are released under the CreativeML Open RAIL++-M license.

• TRELLIS: used for image-to-3D mesh generation. TRELLIS models and code are released under the MIT license, with some submodules under their own licenses.

• Comparison models: MiniCPM-o-v2.6, VideoLLaMA2, Video-SALMONN 2, Qwen2.5-Omni, Qwen3-Omni, SAVVY, Hear You Are LLM, GPT-Realtime, and Gemini-2.5-Flash-Lite are used only for evaluation or baseline comparison under their respective model licenses, APIs, and terms of use.

All generated 3D assets are synthetic and are not intended to reproduce real people, private environments, or sensitive locations. We will release only assets and annotations that are compatible with the applicable licenses and redistribution terms.

Egocentric distance annotation in train set  
![](images/1e2ab67f22f6d12c7febd056b3ef4a59f7a7889440a0127ae6249511be4d77a7.jpg)

Egocentric distance annotation in test set  
![](images/787fe7936ae95ade45e1a96409b39e871fa5ae252b986b88d27dc80d7445caf4.jpg)

Allocentric distance annotation in train set  
![](images/fb83f6901f1b88a1458075d2c2a156b28f20bc180f659772f61001dadd219e3f.jpg)

Allocentric distance annotation in test set  
![](images/533780e3ca008bb90eb4687a0e88b21d246891b4076a9d72231a59d822e51046.jpg)  
Figure S8: Distance annotation statistics. We show histograms of distance-bin annotations for egocentric and allocentric perspectives in both the training and test sets.

Egocentric direction annotation in train set  
![](images/dc397589ab9a1ac1b1149f5208a96fc7074204f08728212a8c288fba2da7b7bc.jpg)

![](images/d4063e405007b09afabf64c93217855c91110bb0c87752fbd8b36406630ce780.jpg)

![](images/d5f2fb0fef03a54db8acc953f37b329a5a9dfb67cdf2ff67e4c09a017bd2bd64.jpg)

![](images/92280c46d746aecd06a73899fab8f82732b5a98ddcee8731194d7e7a1565b448.jpg)  
Figure S9: Direction annotation statistics. We show log-scaled distributions of direction-bin annotations for egocentric and allocentric perspectives in the training and test sets.

![](images/4cf6e6d662f446ab197cd9e616a869f54ace8e130efdb19a351d9cc2268f43f9.jpg)

![](images/51e6152a3f2f01aa8e8fede2c9a4b456d29ab58c264b0e8da0d8d699fe85ffb1.jpg)  
Figure S10: Top 30 inserted object categories. We show the 30 most frequent object categories inserted into training and test scenes.

![](images/a58b82edf8e5c7747fc2d7e019518e3032d4a6dd857626d72500fbaa82597703.jpg)

![](images/3583b62e85de29d23e2c316806d5d1fe7fdef8f30eb7b4ee9db85f760a660f6f.jpg)  
Figure S11: Top 30 active sound categories. We show the 30 most frequent sound categories used as active sound sources in training and test scenes.