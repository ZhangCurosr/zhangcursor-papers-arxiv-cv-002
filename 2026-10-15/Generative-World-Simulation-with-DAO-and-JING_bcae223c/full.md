# Generative World Simulation with DAO and JING

## XGEN Team

## Abstract

We present a Generative World Simulation framework that couples persistent shared world state with continuous first-person audiovisual generation. The framework integrates DAO, a computable world engine, and JING, an Egocentric Interactive Experience Model, to support sustained interaction for humans and agents. DAO maintains the states of characters, objects, and spaces and their relationships, advancing the world according to interaction rules and world time while preserving the consequences of past actions. From this evolving state, DAO derives observer-specific conditions for JING, which combines these conditions with character, object, and scene references, observation history, and current actions to generate each observer’s audiovisual experience. Users can explore environments, manipulate objects, and communicate with characters through viewpoint control and text instructions, with their actions shaping both the shared state and subsequent experiences. By connecting world-state evolution with experience generation in a continuous interaction loop, the framework offers an approach to building interactive worlds that retain their history, evolve through action, and persist beyond any individual observer.

BLOG: https://xgenlabs.ai/research/generative-world-simulation

MODEL: https://huggingface.co/XGENlabs/XGEN-JING

## Generative World Simulation

![](images/ab43dc65156aafb204c887496fe2e2c86f1d85f5a27985fe146b27d97d841662.jpg)  
Egocentric exploration  
Navigating through the alleys of an ancient desert city

![](images/d35d2b6faf14dc49d40a232ba73cf727b313677c24fafea4564b840d1ad5954b.jpg)  
Character communication  
“Shall we get of at the next town?”

![](images/4cadbbf70120cb3b6ee37b93112ff4ff0109458b30a70d45435b7e47bb04006b.jpg)  
Object manipulation  
Igniting a woodpile with a short burst of dragon fire

![](images/44141c0a6fef2900742e46fe9c725a6782bc1bd6abad7fa4b3be11208e8f85e8.jpg)  
Embodied interaction  
A household robot retrieves a ball for a dog

![](images/85e9f4bbfbaffd76f161594f9560a1db034ee0204c9935ec83871f7111fdd212.jpg)  
Figure 1. Generative World Simulation. DAO maintains shared world state and updates it in response to actions; JING generates each observer’s first-person audiovisual experience from the resulting conditions. Representative JING trajectories illustrate exploration, object manipulation, character communication, and embodied interaction.

## XGEN

## Contents

1 Introduction 3   
2 Related Work 4   
2.1 Multimodal Video Generation 4   
2.2 Interactive and Egocentric World Models 4   
2.3 Long-Horizon Generation and Memory 5   
2.4 Persistent World State and Generative Simulation 5   
3 Data Pipeline 5   
3.1 Data Sources 5   
3.2 Processing and Quality Control 7   
3.3 Training Data Organization 9   
4 Methods 9   
4.1 JING-Experience Model 9   
4.2 JING-Model Training 11   
4.3 DAO-World Engine 12   
5 Model Performance 14   
5.1 Quantitative Results 14   
5.2 Qualitative Results 14   
5.3 DAO-World Engine Performance 18   
6 Conclusion 20   
References 20   
7 Contributors and Acknowledgments 24

## 1 Introduction

Video generation models [1–6] have made substantial progress in visual fidelity, motion coherence, and multimodal conditioning, with recent systems also supporting synchronized audio. Their ability to synthesize realistic scenes and dynamic content provides a foundation for generating interactive environments. Building on these advances, interactive world models [7–12] generate subsequent observations in response to user or agent inputs, while broadening the range of interactions, extending generation horizons, and improving runtime efficiency. These advances bring us closer to explorable environments that evolve continuously and respond to their inhabitants in real time. They also motivate a further goal: recording and updating world state alongside immediate generative feedback, so tha the consequences of actions persist and shape future events and observations. This would connect exploration, manipulation, and communication into a continuous experience grounded in history and shaped by action.

Existing approaches explore different aspects of interactive world modeling. Research on game-based simulation [13–16] studies action-conditioned generation using gameplay data and virtual environments. Camera-controlled world models [7, 9–12, 17] enable users to explore generated scenes through viewpoint changes and navigation. Text-controlled world models [8, 11, 18–21] further allow users to influence character behavior and world events through natural-language instructions. These directions provide increasingly flexible interfaces for interaction within generated environments. An open challenge is to organize the resulting changes into an explicit shared world state that can be queried and updated, evolves according to defined rules, and preserves the consequences of past actions. This becomes particularly important when a world contains multiple actors, evolves over extended periods, or is revisited after an observer leaves. Such a system must maintain shared facts and history while generating each observer’s local experience.

To address this challenge, we present a Generative World Simulation framework comprising DAO and JING. DAO is a computable world engine that maintains the shared world’s state and computes its dynamics. JING is an Egocentric Interactive Experience Model that generates what each observer sees, hears, and experiences next. Actions change the world, and these changes shape the experiences and subsequent choices of humans and agents. This continuous loop connects perception, decision, action, and world evolution within a shared world that persists beyond any individual observer. Users can specify the scenes they explore, the objects they manipulate, and the characters they communicate with, continually influencing both the world and their own experience through interaction. The framework has three core capabilities:

• JING: Controllable first-person audiovisual experiences. JING supports flexible combinations of character, object, and scene references, together with observation history and current actions, to generate continuous audiovisual content. Users explore environments through keyboard and mouse inputs and express manipulation and communication intents through text instructions, enabling first-person interaction across diverse settings and visual styles.

• DAO: Computable and persistent shared world state. DAO explicitly represents characters, objects, spaces, and their relationships, while maintaining world time and active events. The engine processes actions according to the current state and interaction rules, carries the resulting changes into subsequent states, and retains unchanged information. Past interactions can therefore influence future developments, and world state is preserved as observers change viewpoints, leave, or return.

• Continuous closed-loop interaction in a shared world. DAO and JING connect action consequences, local observations, and subsequent choices in an ongoing cycle. Humans and agents act on the audiovisual feedback they receive. The resulting changes enter the shared state and shape future experiences and interaction opportunities. From the same starting context, different choices can lead to different developments, allowing each experience to build on its history while responding to new actions.

We build JING by adapting MiniMax-H3 and training on diverse audiovisual data through a staged pipeline. Bidirectional teacher training introduces camera control, while causal autoregressive training with resampling forcing adapts the model to continuous interaction conditioned on generated histories. Memory adaptation and distillation support the use of historical information and efficient generation, respectively. DAO organizes shared world state through an entity graph, together with actor states, a world clock, and event mechanisms. It updates this state according to spatial constraints and interaction rules, with snapshots and event history supporting persistence. The world state is then converted into observerspecific conditions for JING’s audiovisual generation. This design connects world-state computation with audiovisual experience generation. This report describes the data construction, model training, and world engine design, and examines generation quality, action responsiveness, state persistence, long-term consistency, and runtime efficiency.

Our long-term goal is a persistent shared world in which human and agent interactions have lasting consequences and world evolution continues beyond any individual observer’s presence.

## 2 Related Work

## 2.1 Multimodal Video Generation

Video generation has progressed from text- and image-conditioned visual synthesis toward synchronized audiovisual generation with diverse reference conditions. Wan [1] and HunyuanVideo 1.5 [2] advance visual quality, motion coherence, and condition adherence, providing a foundation for generating complex dynamic scenes. Joint audiovisual models, including SyncFlow [22], Sora 2 [3], and LTX-2 [5], model visual and acoustic content together to produce temporally aligned outputs. Seedance 2.0 [4] and MiniMax-H3 [6] further extend multimodal reference conditioning and content control. Together, these advances enable models to represent scene appearance, character behavior, and accompanying sound. Building on this foundation, JING combines character, object, and scene references with observation history and ongoing action inputs to generate continuous, observer-centric interactive experiences.

## 2.2 Interactive and Egocentric World Models

Interactive world models predict future observations conditioned on actions, extending video generation to environments that users can continuously influence. MineWorld [13], Matrix-Game [14], and Hunyuan-GameCraft [15] learn mappings from player actions to visual feedback in game environments. Matrix-Game 2.0 [17], WorldPlay [7], and AlayaWorld [11] further explore real-time control, streaming generation, and long-horizon interaction. LingBot-World 2.0 [8] and Genie 3 [18] broaden semantic control over character behavior and world events. In egocentric interaction, Hand2World [23] studies gesture-driven object manipulation, EgoSim [24] uses an updatable 3D scene to support continued interaction, and EchoWM [12] combines navigation with video, environmental sound, and speech generation. JING focuses on integrating exploration, manipulation, and communication under flexible reference conditions, allowing these forms of interaction to unfold within a continuous first-person audiovisual experience.

## 2.3 Long-Horizon Generation and Memory

Long-horizon interaction requires models to use generated history within a bounded computational budget while limiting error accumulation during autoregressive rollouts. Diffusion Forcing [25] and Self Forcing [26] advance autoregressive video modeling through causal generation and training on model-generated histories, respectively. To manage growing context, FramePack [27] compresses historical frame representations, while LongLive [28] combines streaming inference with early-frame retention. WorldMem [16] and WorldPlay [7] improve long-term consistency through history retrieval and geometry-aware memory. OmniForcing [29] extends streaming autoregressive generation to synchronized audiovisual content. For sampling efficiency, Distribution Matching Distillation (DMD) [30] and Velocity Moment Matching (VMM), introduced in EditStream [31], reduce the number of sampling steps through distillation. JING combines adaptation to generated histories, memory utilization, and few-step distillation to support sustained egocentric interaction.

## 2.4 Persistent World State and Generative Simulation

Explicit state modeling provides queryable and updatable representations of world facts, allowing world evolution to be governed by both state and rules. StatePlay [32] jointly predicts visual content and game states to improve adherence to game mechanics. MASS [33] separates shared-state updates from observation generation: a learned logic engine processes joint actions, and a rendering engine generates observations from the resulting shared state. For spatial and entity state modeling, EgoSim [24] maintains an updatable 3D scene, while Marionette [34] separates structured state prediction, geometric rendering, and appearance synthesis. Programmable World Model [35] further uses executable programs to maintain entity states and transition rules, compiling these states into spatial conditions for video generation. DAO focuses on ongoing character activities and event evolution under explicit interaction rules, translating local world state into observer-specific text, camera, and reference conditions for JING’s first-person audiovisual generation.

## 3 Data Pipeline

Our data pipeline converts diverse video sources into aligned multimodal training samples for JING. As shown in Figure 2, the pipeline spans data acquisition, source-level filtering, temporal segmentation, captioning, conditioning, and training data organization. Its design emphasizes temporal continuity across navigation, scene revisits, interactions, and dialogue, together with consistent alignment among video, audio, text, and optional camera or reference-image conditions. The following subsections describe the data sources, processing and quality-control procedures, and organization of the resulting training corpus.

## 3.1 Data Sources

We construct our training corpus from two complementary sources: internally collected and generated data and open-source datasets. Our data effort centers on egocentric experience: how an environment unfolds from an observer’s first-person perspective as they move through scenes, revisit locations, interact with surrounding entities, and engage in conversations. These experiences guide the selection and construction of our training material.

![](images/c4b5040c68c45ed9727b3b578b304821b414b8a13c095eab9754abe265a327d7.jpg)  
Figure 2. Data pipeline overview. Internally collected and generated egocentric videos and open-source datasets undergo automated media checks, model-assisted semantic filtering, temporal segmentation, and captioning. Optional camera trajectories and extracted or generated reference images are aligned with the processed clips and validated before accepted samples are organized by capability and conditioning configuration and packaged into latent representations and LMDB shards for training.

Internally Collected and Generated Data. Our internal corpus focuses on egocentric navigation, scene revisits, interaction, and dialogue. We collect first-person videos from a range of publicly accessible platforms, combining broad scene-exploration footage with targeted collections of specific environments and scenarios. These videos capture how scenes unfold as an observer moves through them, encounters nearby entities, and returns to previously observed locations.

We complement this material with game and interactive-video footage acquired through dedicated collection efforts by our team and external partners. These sources expand our coverage of navigation, interactions, and conversations within evolving scenes. We further supplement the corpus with videos rendered in controllable 3D environments and model-generated videos depicting interactions and dialogue, allowing us to construct additional experiences around selected scenes and events. Across these acquisition and generation routes, we prioritize the observer’s first-person perspective and the temporal context of each experience, including continuity across camera movements, scene revisits, and interactions.

Open-source Datasets. We additionally incorporate publicly released datasets, including SpatialVID [36], Sekai [37], DL3DV [38]. Selected subsets complement our internal collection with additional scene content and camera-motion patterns, broadening the visual and spatial diversity of the training corpus. Together with our targeted collection efforts, these resources provide video materia depicting how environments unfold across changing viewpoints and observer movements.

## 3.2 Processing and Quality Control

Our processing pipeline converts collected videos into training samples with aligned video, audio, captions, and optional conditioning signals. We apply common quality criteria across sources while adapting segmentation and annotation to the temporal structure of navigation, scene revisits, interactions, and dialogue. The objective is to preserve continuous egocentric experiences and provide consistent supervision for training.

Source-level filtering. We first apply automated checks to media metadata and decoding results, and compute visual-quality metrics on sampled, low-resolution video segments. These checks assess resolution, frame rate, decoding reliability, sharpness, exposure, compression artifacts, and frozen frames. Predefined rules exclude videos with severe defects, while cases requiring contextual interpretation, such as moderate motion blur or potential shot transitions, are retained for further assessment.

Videos that pass this stage undergo semantic assessment using VLM. The model examines frames sampled across the video timeline, short windows sampled at higher frame rates, and available audio to assess first-person perspective, motion continuity, shot transitions, visual overlays, and unrelated background music. The viewpoint criterion requires the camera to represent the observer’s direct experience, without requiring visible hands or other body parts. Motion is evaluated in context: natural walking movements and coherent head turns are acceptable, while unexplained jumps or interruptions that break temporal continuity are grounds for rejection. The model’s structured outputs are checked against predefined acceptance rules to determine whether each source video can proceed to subsequent processing.

Temporal segmentation. After filtering, we use Qwen3.6 [39] to identify candidate segment boundaries from sampled video frames. These boundaries are adjusted according to the activity being captured. For scene revisits, segmentation aims to keep the departure and return within the same sequence. For interactions and dialogue, it aims to preserve complete operations and conversational exchanges. Each segment retains its source identifier and original time interval, allowing the processed video to be traced back to its source.

Processed clips are packaged into sharded LMDB datasets containing video and audio latents, captionaligned text embeddings, and applicable camera or reference-image conditions. This unified format supports data from different processing pipelines and enables variants with and without reference-image conditioning to share the same audiovisual content. Each record retains its source, temporal interval, and conditioning settings, preserving alignment and traceability across supervision variants. Figure 3 illustrates the temporal alignment and optional conditioning of multimodal samples across diverse egocentric experiences.

![](images/208c82bb8bcfc6e87ce8f7eb72e0b2b0e0d3c17003495cc718a59ca68b6d4db1.jpg)

![](images/2eac135935e1140eb00acce1223c58c182dff475c257c145c26e8259df3683a3.jpg)

![](images/e94aa1b287bde006dc6a58ef69d41ea48844aa735f48b7248b0aade642a9051b.jpg)

![](images/d3052f0edb992acd11c01156c62bba3aaf0a0efd9cea7cfdaa94838ba9a6dfbf.jpg)

![](images/076342b107dde59ae899fdb8f64cbc8bfb960778f15b3ee5337b4bd4c1091c9d.jpg)

![](images/728d06efefcb58b7860dc46ea3b4b6b45971606c189910de231b5a6f034e3c70.jpg)  
Figure 3. Diverse aligned training samples. Illustrative samples covering real-world and virtual-world navigation, scene revisits, object interaction, dialogue, and rendered environments. Within each panel, video frames, captions, audio, and available camera poses follow a common temporal axis, with optional reference images for character, object, or scene conditioning.

Video captioning. We first identify recurring entities and scene context to maintain consistency across segments, then generate captions describing visible events, actions, speech, and relevant sounds. These descriptions are refined into training captions with a first-person perspective. Captions paired with camera trajectories avoid redundant camera-motion descriptions, while those without explicit camera conditioning include the relevant camera movements.

Conditioning and validation. We construct camera and reference-image conditions for selected samples. For camera conditioning, ViPE [40] estimates camera intrinsics and six-degree-of-freedom poses from processed clips. The resulting trajectories are smoothed and filtered to reduce jitter and exclude unreliable estimates, while maintaining temporal alignment with the video.

Reference images specify characters, objects, or scenes and are obtained by selecting and cropping targets from video frames or generating images from relevant frames and prompts. Each reference is linked to the corresponding caption intervals to align visual conditions with the described content.

Before training, we validate audiovisual synchronization and consistency among media, captions, metadata, and conditioning signals. Automated checks are complemented by inspection of sampled examples, and only samples that pass validation are retained.

## 3.3 Training Data Organization

After processing and quality control, we organize the accepted data according to the egocentric capabilities they support, including navigation, interaction, and dialogue. Each sample retains its primary capability and relevant secondary attributes, allowing a sequence to capture multiple aspects of an experience. The availability of camera trajectories and reference images further determines the conditioning configurations supported by each sample.

## 4 Methods

## 4.1 JING-Experience Model

JING is an Egocentric Interactive Experience Model built on MiniMax-H3 [6] that generates continuous audiovisual content autoregressively. As shown in Figure $^ { 4 , }$ at generation step k, the model takes a set of reference images R, a text instruction $p _ { k } .$ , camera trajectory $\mathbf { C } _ { k } ,$ , and the previously generated audiovisual history $\mathcal { H } _ { < k } .$ , and predicts the next video and audio latent chunk:

$$
( \mathbf { Z } _ { k } ^ { \nu } , \mathbf { Z } _ { k } ^ { a } ) \sim P _ { \theta } \left( \cdot \mid \mathcal { R } , p _ { k } , \mathbf { C } _ { k } , \mathcal { H } _ { < k } \right) ,\tag{1}
$$

where v and a denote video and audio, respectively. The corresponding VAEs decode these latents into synchronized frames and audio, and the generated chunk becomes part of the history for subsequent generation. Reference images are jointly encoded with text by Qwen3-VL [41] into semantic conditioning tokens $\mathbf { T } _ { k }$ and separately encoded by the video VAE into reference tokens $\mathbf { R } _ { z } ,$ providing semantic conditioning and visual reference information, respectively.

To further support streaming generation, we preserve the temporal order of the latent sequence and reorganize $\mathrm { H } 3 7 \mathrm { s }$ temporal partition from $( 5 , 5 , \ldots , 2 )$ to $( 2 , 5 , 5 , \ldots )$ . The initial $2 + 5$ video latent frames form an initialization chunk, followed by chunks of five latent frames each. This layout aligns generation boundaries with the VAE’s overlapping decoding windows, enabling incremental decoding as new chunks become available.

Camera conditions enter the Transformer through a dedicated control encoder. Relative translations and rotations between consecutive camera poses form a six-dimensional motion representation, which the encoder converts into control features $\mathbf { c } _ { k }$ aligned with the video tokens in space and time. In Transformer block ℓ, these features modulate the video hidden states through feature-wise linear modulation (FiLM):

$$
\widetilde { \mathbf { h } } _ { k , \nu } ^ { \ell } = \left( 1 + \gamma _ { \ell } ( \mathbf { c } _ { k } ) \right) \odot \mathbf { h } _ { k , \nu } ^ { \ell } + \beta _ { \ell } ( \mathbf { c } _ { k } ) ,\tag{2}
$$

where $\gamma _ { \ell }$ and $\beta _ { \ell }$ predict scale and shift, respectively, and ⊙ denotes element-wise multiplication. The modulation acts on camera-conditioned video tokens after the attention residual update and before the subsequent normalization and feed-forward network. Text and audio tokens do not receive this

![](images/47a43b39aec5f4941165e336dac6de1181229ce994fb5ce95c6fa3b0c2bab764.jpg)  
Figure 4. JING architecture. (a) Reference images provide visual conditions through the video VAE encoder and (b) the Qwen3-VL conditioner, which also encodes text instructions. Camera trajectories are processed by a separate control encoder. (c) Current tokens attend to cached keys and values (K/V) from reference and historical context. (d) Camera features modulate video tokens within each Transformer block. (e) Retrieved audiovisual memories supplement the persistent first chunk and recent context. The diffusion sampler produces clean audiovisual latents, which the video and audio VAE decoders convert into synchronized output.

direct modulation, but exchange information with visual features through shared attention. This design introduces camera control while retaining a unified audiovisual backbone.

The attention context for the current chunk combines reference conditions, history at different temporal scales, and the content being generated. After projection into a shared hidden space, its logical composition is

$$
\begin{array} { r } { { \bf S } _ { k } ^ { \tau } = \left[ \underbrace { { \bf R } _ { z } ; { \bf H } _ { 0 } ; { \bf H } _ { k } ^ { \mathrm { r e c e n t } } ; { \bf M } _ { k } } _ { \mathrm { r e f e r e n c e ~ a n d ~ h i s t o r i c a l ~ c o n t e x t } } ; \underbrace { { \bf T } _ { k } ; { \bf Z } _ { k } ^ { a , \tau } ; { \bf Z } _ { k } ^ { \nu , \tau } } _ { \mathrm { c u r r e n t ~ c h u n k } } \right] , } \end{array}\tag{3}
$$

where τ denotes the noise level during denoising, $\mathbf { H } _ { 0 }$ is the fixed first generated chunk, $\mathbf { H } _ { k } ^ { \mathrm { r e c e n t } }$ contains the full audiovisual context from recent chunks, and ${ \bf { M } } _ { k }$ represents memories retrieved from older history. History terms are empty until the corresponding chunks have been generated. Semantic, audio, and video tokens interact through joint attention in the shared Transformer. Attention is joint within each chunk, while access across chunks is causal. The fixed first chunk provides an initial anchor, recent context supports local continuity, and retrieved memories supply information beyond the current window.

More specifically, JING maintains an audiovisual memory archive to access history beyond the recent window. Retrieval units are video latent frames and contiguous groups of stereo audio latents. After each chunk is generated, the model extracts compact query and key descriptors from the Q/K representations of selected attention layers. At inference time, the query descriptor from the last completed chunk combines information from text and the corresponding modality to score earlier history units:

$$
s _ { k , i } ^ { m } = \mathrm { A g g } _ { \ell , j } \left( \frac { \left. \bar { \mathbf q } _ { k - 1 } ^ { m , \ell , j } , \bar { \mathbf k } _ { i } ^ { m , \ell , j } \right. } { \sqrt { d _ { \mathrm { h e a d } } } } \right) , \qquad m \in \{ \nu , a \} .\tag{4}
$$

Here, $\bar { \bf q }$ and $\bar { \bf k }$ denote pooled query and key descriptors, respectively. The indices i, ℓ, and j identify historical candidates, attention layers, and attention heads, while $d _ { \mathrm { h e a d } }$ denotes the feature dimension of each head. The operator Agg standardizes candidate scores before aggregating them across heads and layers. The candidate pool excludes the retained first chunk and recent context. Candidates are divided into temporal intervals, and relevant units are selected from each interval to balance relevance with historical coverage. The selected units are replayed through the Transformer with dedicated RoPE position slots and AdaLN memory-role embeddings to obtain memory K/V for the current context. Each newly generated chunk is then added to the archive for future retrieval. By separately limiting the recent window and audiovisual memory budgets, the model bounds the historical context used by current attention while retaining access to earlier experiences.

## 4.2 JING-Model Training

Using the audiovisual data described in Section 3, we train MiniMax-H3 in five stages to introduce camera control, enable causal generation, incorporate long-term memory, and reduce sampling steps. Together, these stages adapt the pretrained model for continuous, real-time first-person interaction.

We first train a bidirectional teacher with the camera-control module described in Section 4.1. Cameramotion features modulate the Transformer alongside the existing text and visual conditions. We retain bidirectional attention and jointly optimize model parameters and the control module, allowing the model to learn camera-conditioned motion from complete audiovisual clips. This lightweight extension introduces camera control while retaining the pretrained backbone architecture. The resulting model serves as the teacher for subsequent distillation.

We then train a causal autoregressive model to generate audiovisual chunks conditioned on preceding content and current user inputs. We augment teacher-forcing training [42] with resampling forcing [43]: historical segments are perturbed with noise, reconstructed by the model, and reused as context for predicting subsequent content. These reconstructed histories reduce reliance on error-free context and help bridge the gap between clean training histories and model-generated histories at inference. Camera and text conditions can be updated at chunk boundaries, allowing generation to respond to ongoing interaction.

Next, we introduce the audiovisual memory described in Section 4.1 and train the model to combine the fixed initial context, a recent-history window, and retrieved memories. During adaptation, we vary the recent-context length, memory selection, and available memory capacity so that the model learns to use different combinations of historical information. This provides access to earlier scenes and events within a bounded context budget: recent content supports local continuity, while retrieved memories supply cues from beyond the recent window.

To reduce denoising steps, we initialize a few-step student with Velocity Moment Matching (VMM) [31]. VMM matches conditional velocity moments at intermediate noisy states reached during the student’s denoising process. A frozen teacher and a trainable auxiliary network provide velocity estimates that guide the student’s updates. This objective aligns with the backbone’s flow-matching parameterization and supervises states encountered during few-step sampling, providing an initialization for subsequent distribution matching.

Finally, we refine the student with Distribution Matching Distillation (DMD) [30]. The student generates audiovisual sequences conditioned on its own generated history. A frozen teacher and a score model trained on student-generated samples guide the student’s output distribution toward the teacher’s. Building on VMM initialization, this stage aims to recover quality lost when reducing sampling steps and improve autoregressive generation. The full training pipeline combines camera control, adaptation to imperfect histories, and memory utilization with efficient few-step sampling.

## 4.3 DAO-World Engine

DAO is a computable world engine that maintains explicit shared state and preserves the consequences of user and agent actions. As shown in Figure 5, a world definition specifies the spatial layout, initial entities, interaction rules, and spatial constraints. State updates are separate from observation readout, allowing different observers to derive their experiences from the same underlying world.

We represent the shared world state at simulation step t as

$$
\boldsymbol { W _ { t } } = \left( \boldsymbol { G _ { t } } , \boldsymbol { B _ { t } } , \boldsymbol { E _ { t } } , \omega _ { t } \right) ,\tag{5}
$$

where the entity graph $G _ { t }$ represents places, characters, and objects through their poses, appearance, semantic attributes, and spatial and attachment relations. Actor and interaction state $B _ { t }$ records intentions, motion, activities, and attributes such as emotions, needs, and object use. The active event set $E _ { t }$ represents ongoing events, while the world clock $\omega _ { t }$ provides a shared time reference. Together, these components provide a queryable representation of the world for action execution and observation generation.

![](images/2ee6cc40d0881673ce5f7a2c6e94230be13be11d6dbd47c53428b80d928f78bd.jpg)  
Figure 5. DAO world engine. (a) User and agent actions pass through action admission to produce actor intents and state edits. (b) A world definition initializes a shared state comprising an entity graph, entity attributes, actor states, a clock, and active events. An ordered update cycle transforms $W _ { t }$ into $W _ { t + 1 }$ by applying edits, advancing motion, resolving spatial constraints, and advancing activities, while carrying unchanged state forward. (c) Local readout combines the world state with an observer’s pose to provide observations for an experience interface. (d) World snapshots and event history preserve state across interactions.

Inputs first pass through action admission, which resolves targets and checks interaction rules and state constraints, including reachability and target availability. Accepted inputs produce actor intents or queued state edits; inputs that violate these constraints can be rejected. Given admitted inputs $\widehat { \mathcal { U } } _ { t }$ and world definition $\mathcal { D } _ { : }$ , state evolution is expressed as

$$
W _ { t + 1 } = F _ { \widehat { \cal D } } ( W _ { t } , \widehat { \mathcal { U } } _ { t } ) ,\tag{6}
$$

where $F _ { \mathcal { D } }$ is the update process determined by the world configuration and rules. Each step applies queued edits, advances actor motion, updates spatial membership and dynamic obstacles, and then advances activities and events. Unchanged state carries forward, so earlier interactions remain part of the world and can affect subsequent actions and observations.

Table 1. Quantitative comparison on WBench [44].<sup>1</sup>Ranks refer to the comparison pools of 15 Full and 39 Navigation entries. Bold and underlined values indicate the best and second-best results among the models shown within each split; shaded rows denote our models.
<table><tr><td>Rank Model</td><td></td><td></td><td>Video Avg. quality adherence</td><td>Setting</td><td>Interaction adherence</td><td>Consistency Physics</td><td></td></tr><tr><td colspan="2">Full (289 cases)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1 JING (Bidirectional) (ours)</td><td>81.0</td><td>80.9</td><td>88.5</td><td>76.7</td><td>85.7</td><td>73.3</td></tr><tr><td></td><td>2 JING (4-step AR) (ours)</td><td>79.9</td><td>79.8</td><td>88.9</td><td>79.5</td><td>81.4</td><td>69.9</td></tr><tr><td></td><td>3 Kling 3.0 [45]</td><td>79.4</td><td>80.0</td><td>91.0</td><td>73.1</td><td>83.9</td><td>69.2</td></tr><tr><td></td><td>4 Wan 2.7 [46]</td><td>78.4</td><td>81.0</td><td>91.4</td><td>72.1</td><td>75.8</td><td>71.6</td></tr><tr><td></td><td>5 Seedance 1.5 [47]</td><td>76.2</td><td>81.9</td><td>82.9</td><td>68.3</td><td>79.9</td><td>68.2</td></tr><tr><td></td><td>6 HY-Video 1.5 [2]</td><td>74.3</td><td>76.6</td><td>85.6</td><td>54.7</td><td>87.5</td><td>67.1</td></tr><tr><td colspan="2">Navigation (158 cases)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1 Alaya-EVOKE-Turbo [48]</td><td>82.0</td><td>81.9</td><td>82.1</td><td>83.9</td><td>88.1</td><td>74.0</td></tr><tr><td></td><td>2 JING (Bidirectional) (ours)</td><td>81.9</td><td>82.4</td><td>90.3</td><td>75.6</td><td>88.1</td><td>73.3</td></tr><tr><td></td><td>3 JoyAI-Echo-1.5 (Bidirectional) [12]</td><td>81.6</td><td>81.5</td><td>79.4</td><td>86.6</td><td>89.8</td><td>70.6</td></tr><tr><td></td><td>4 JoyAI-Echo-1.5 (4-step AR) [12]</td><td>81.0</td><td>81.1</td><td>77.5</td><td>87.9</td><td>88.3</td><td>70.1</td></tr><tr><td></td><td>5 Zing-0.5 [49]</td><td>81.0</td><td>80.6</td><td>77.8</td><td>84.2</td><td>88.5</td><td>73.8</td></tr><tr><td></td><td>6 JING (4-step AR) (ours)</td><td>81.0</td><td>81.3</td><td>91.6</td><td>78.7</td><td>83.7</td><td>69.8</td></tr><tr><td></td><td>7 HiDream-O1-World [50]</td><td>80.9</td><td>81.0</td><td>82.2</td><td>80.0</td><td>88.0</td><td>73.3</td></tr><tr><td></td><td>10 Kling 3.0 [45]</td><td>79.0</td><td>81.4</td><td>91.0</td><td>69.4</td><td>83.7</td><td>69.3</td></tr><tr><td></td><td>13 Wan 2.7 [46]</td><td>78.1</td><td>81.5</td><td>91.4</td><td>64.4</td><td>81.6</td><td>71.8</td></tr><tr><td>29 Genie 3 [18]</td><td></td><td>73.9</td><td>75.2</td><td>72.5</td><td>73.4</td><td>82.6</td><td>65.7</td></tr></table>

DAO derives local observations from the shared state and observer pose, selecting relevant entities according to sensing range, field of view, and occlusion. These observations provide local entity states, events, and environmental information. Together with camera trajectories and character, object, and scene references, they condition JING’s generation of the next first-person audiovisual experience. Different observers therefore receive local views grounded in the same shared state, and their subsequent actions re-enter the update cycle.

World snapshots and event history support persistence across sessions and process restarts. Snapshots store the state needed to resume simulation, while event history records past interactions. On restoration,

DAO reconstructs the world from the saved snapshot and world definition, allowing subsequent actions and observations to reflect the preserved state.

## 5 Model Performance

We evaluate JING with other video generation and interactive models on WBench [44] and demonstrate its capabilities in reference conditioning, spatial exploration, object manipulation, character communication, branching experiences, and embodied interaction. We further examine DAO’s ability to maintain evolving world states and preserve the consequences of past actions. Together, these results demonstrate XGEN’s capacity to support continuous interactive experiences grounded in a persistent world.

## 5.1 Quantitative Results

As shown in Table 1, we evaluate JING on the Navigation and Full splits of WBench [44]. Navigation assesses interactive world generation across video quality, setting adherence, interaction adherence, consistency, and physical plausibility. Full additionally covers text-driven subject actions, event editing, and perspective switching, extending evaluation beyond navigation. On the Full evaluation set, our bidirectional and four-step autoregressive (AR) variants achieve average scores of 81.0 and 79.9, ranking first and second among 15 comparison entries. On the Navigation subset, they score 81.9 and 81.0, ranking second and sixth among 39 entries. These results show that our training and distillation pipeline transfers the bidirectional model to a four-step causal generator while largely preserving its overall performance.

## 5.2 Qualitative Results

Flexible reference conditioning. Figure 6 illustrates how JING uses different reference combinations to specify the content of an experience. The generated sequences retain recognizable scene features, such as the clock tower and bakery interior, while incorporating the supplied character and object appearances where provided. These examples show how references constrain complementary aspects of generation: the environment to explore, the person to encounter, and the object to interact with.

![](images/b9db56315aeb2607ee5620af536a7da2ba4dddb05b8f0665104c2f533b7ee77e.jpg)

(a) Clock tower  
![](images/68974037ddf37882719449600e8114dc663ec5cd03fd69579b7d21941478ffd9.jpg)  
Scene reference

![](images/fc28da194a8aebb8af1fa65df08553ab96bf396f83d272d2481837d62022e662.jpg)  
(b) Bakery encounter  
10 s

![](images/3f3d96f59a1dc05e5e7eaea48f38e90e67b3c55742e2b0db4764cdd3aa9352d3.jpg)  
20 s

![](images/a6e96835d7f92dfcb62d824b6548e792dd40ca201273d0bdbb85b3ff21ae6dfd.jpg)  
29 s  
Scene + character

![](images/a138e7dd56a8c99b65e01aae0f779a5eb204473b52a459ebb717644f62aabf0f.jpg)

![](images/2f684232c7d470aa228e7edd693e2e47868f2f85d4d39595d038c7a08bc471ca.jpg)  
4 s  
9 s

![](images/eabe11c44be92cfecbd63dbc3f1e5e3f181ab1c9049692a8e2789f489bb504f4.jpg)  
14 s

(c) A stranded wagon  
![](images/dcf78384fe073378b2f30e3b7e3915a76f20f892cb05a9b3a68ad0a6318f0d84.jpg)

![](images/08fa570599ea691ac6efe03d781ef0d57dfd0a7f80ff4a4883ce35c98b2c5446.jpg)  
Scene + character + object

![](images/42dea7d2d6e47d7e6b14462b7f8f28a832ef331f98d44c36ba889766a1989296.jpg)  
8 s

![](images/8040d09ef78b97491f14d68128ce9acea0c9586a470cd45c8c0fb676f9634dc4.jpg)  
16 s

![](images/867d5012623fcbd9111d7a7c48acf279a78e51a38534672ccc83df45d1f1f7d7.jpg)  
24 s  
Figure 6. Reference-conditioned generation. Input references and sampled outputs for three configurations: scene only (a), scene and character (b), and scene, character, and object (c).

Egocentric exploration. JING translates navigation inputs into changing first-person views across different environments and observer scales (Figure 7). Moving through an archway reveals a street, turning near a shrine exposes surrounding paths, and navigating at beetle scale reveals the space beneath leaves. Nearby surfaces and landmarks remain recognizable across successive views, illustrating local visual continuity as exploration reveals previously unseen content.

(a) Desert city alleys  
![](images/10faccaff6d79bb2af9c98a965e0301144c9fcb76b9fd16667528ebb80b69c8e.jpg)  
0 s  
(b) Forest shrine

![](images/c3c86eafa02dd31f5f059710233ac4870ffe47a79fdca49ec907bf758726cdf7.jpg)  
4 s

![](images/e51c6a3172fe5a5420be0711a91c6800551bd5e2f09e00ca71e77da295633b2c.jpg)  
9 s

![](images/7d5b1eb422dbc15085d6eecaa0ed8d29411fb23a8343cd490c90a1c0acadd202.jpg)  
14 s

![](images/26e6241783442f970baeaaffeba18568b8ac017426155a6f3e3577e068a43f9e.jpg)  
0 s  
(c) A garden at beetle scale

![](images/e88df0ee056b7439390a6116b3704ed981777069d75927aa8cedf7aaa43a9458.jpg)  
4 s

![](images/d40778ed796830ab81496a56ed41f4017a52db659f7193d4d358996f39361fe2.jpg)  
9 s

![](images/f8b697a7b062431b4e846cd3254732c1ed1e1b85afeb5175db37ae5724512ec9.jpg)

![](images/f9393d89cd9923a347e62159149f7408294823f0d8bc2b6ac4298ea4cc6bfc9a.jpg)  
0 s

![](images/e4e040bf63d29d18bbafc857900483b34946ea3b150d16bb73dc0a8e6532b32f.jpg)  
10 s

14 s  
![](images/209e7a2aa504f485bbfe6cb811183df5554ec1284bcaeb79042b0a13b019ff59.jpg)  
22 s

![](images/1bc6a1441307952a48e4955c89de11bfe3ed64a2742e0dff4b9f02374985316a.jpg)  
34 s  
Figure 7. Egocentric exploration. Sampled frames from navigation through desert-city alleys (a), a forest shrine (b), and a garden at beetle scale (c). Overlaid controls indicate user inputs during generation.

Object manipulation. Beyond viewpoint control, JING generates interactions that change nearby objects (Figure 8). The sequences depict tool contact, ignition, and release, linking first-person actions to visible outcomes such as a burning woodpile or an ascending lantern. They illustrate action-conditioned changes in object state and motion across realistic and stylized settings.

(a) Blacksmithing  
![](images/d177f53af2450b3018eb146075d8db7e8de485fbf301abe1208c90c8fadc9c1c.jpg)  
0 s  
(b) A young dragon

![](images/dea5ede0df5a33909e739349ca6c9a5db544ce3094b7fce02186e7f096a9d689.jpg)

![](images/788d2186edd4c1d5ad350cc691fef7afa229c5460ee2187094330a453bd2c188.jpg)  
0 s

2 s  
![](images/d2df9c76f4cc210f2247ae87aa7236d0b62b68ced1dad86b2f7bb01a77f5d0d6.jpg)

![](images/39a5e2b68597256252fcb0adcb54a04b20d2492e3817c821fe4240c2e26d4c95.jpg)  
5 s

![](images/9e8268ec3cb14fb8885382ce3ba214cada0797aed079ef109ee6902a5b7aee0d.jpg)  
3.5 s

12 s  
![](images/3d2e42d966d2c7550a073b21b9bcede3af61796d1911484f6ab03abebdcbaa8d.jpg)

8 s  
(c) Sky lantern  
![](images/d676189f2daad29b3f2a207074ae7558b8967b72a9eefd7a7237b29ef647fab4.jpg)  
12 s

![](images/08709ba3357b51ac6a643e38b055e453f4bc147243f85cd13defd1db7be25198.jpg)  
0 s

![](images/486af507b2a646df43b36312603b5ec014db9f98ea1aef91d3589d296eee1204.jpg)  
4 s

![](images/53c0c3ed6ab4737d73e3d04914b31d7eb3492e13a64e88047d1a68627615ad48.jpg)  
9 s

![](images/11b062130d01ae3b2cf9c021bdf9e119827c36e6e42f7e69c5c889b09b74c833.jpg)  
14 s  
Figure 8. Object manipulation. Sampled frames showing a hammer striking heated metal (a), a young dragon igniting a woodpile (b), and hands releasing a sky lantern (c).

Character communication. JING extends first-person interaction to character communication through generated speech and visual behavior (Figure 9). Across the illustrated languages and visual styles, characters address the observer with changes in gaze, expression, and posture while remaining within the ongoing scene.

(a) Rose terrace  
![](images/044856bb708d21b12f2e969834277f9a0e7d422a98e34a87536255d8d249a24b.jpg)  
2 s

![](images/cf85d07f4538e877ac211c70a3962d175422d34ce1b9bc1546ed432703e0bd18.jpg)  
4 s  
(b) Planning a journey

![](images/baea1b29997b28083aeabebf365acf35a49fb59f89084a64c9a55c87bdae2843.jpg)  
9 s

![](images/f0c5b84f876455d0f18298fa715f9eaa627e937e8294ccb5e83e9a8bd1516e59.jpg)  
14 s

![](images/1bdbac016b5b4535dd766e87abcf90222734d385a097f4e13a447609afa45c30.jpg)  
0 s

![](images/6236adadf9df2047177522c185c33e81ae3573bc655fdcb87dc735798c967191.jpg)  
(c) A journey by train  
5 s

![](images/b3a7a600c6a80ab95ef9058b0f7c59a81051637b8ed91c46e7db7a5145c39fd8.jpg)  
11 s

![](images/58ee5d49ad8fb17f8fd7b83955e6c5673d30260244037afe356677a1dac856a5.jpg)

![](images/532525e7930c3bfae837caf2f98bc22b4ed9b43901f326636eece6f45dfb30b4.jpg)  
0 s

17 s  
![](images/c9808ebd5112176874445c5ec5416348263982f681605063cc2d714a86d2db4e.jpg)  
6 s

![](images/487aef09741daebc969c570bf4eaabf61fc0705d8ba2793e3ec5e71e500a5a77.jpg)  
13 s

![](images/b21433677b204b8002ee697b3de3bb36fa4945f1fa5ee045376e92c85bcb3478.jpg)  
20 s  
Figure 9. Character communication. Sampled frames from conversations in Chinese on a rose terrace (a), English in a fantasy setting (b), and Japanese on a train (c). The corresponding videos include generated speech.

Action-conditioned branching. Different user inputs can guide distinct continuations from a shared visual context (Figure 10). In the fountain scene, viewpoint control reveals coins in the water, a manipulation instruction produces a coin-toss gesture, and a request to meet a fairy introduces a character encounter. The common opening connects these branches, illustrating how exploration, manipulation, and communication offer different ways to develop the same initial situation.

(a) Explore: Look down into the fountain  
![](images/5e7ed7d2f331b07133281aacf699f1fb2a36fd0755b0162f93a6e58222d063d5.jpg)

![](images/40caf072f6cdaca3840053fb663ceed408dbcd7e97760fe310f4a0be51250f6e.jpg)  
(b) Interact: Toss a coin into the water

9 s  
![](images/7ad707a8056a716698e07835d23d5602f287e1f26b9633e5cb59b98b06219dcb.jpg)

![](images/9e8db681eded35c5aee8b5f5d6c45bdb816375ed3d01829fb3b01d43c2d88ff9.jpg)

![](images/c80d1e2bc4e4eb2e88b2c6f264994234f45883e1ae67a6664601f29e06c9f443.jpg)

![](images/d9b5fddc02432046dbb55deaa61f1bbdef028c364398a6edc806d36080a5282c.jpg)

14 s  
![](images/188fa562dcf9620f30adcc30acbec78c9ee504b27ed97f3939c572d302d5edcc.jpg)  
8 s

![](images/372bd339d4b62de14b5e0bc28db0cdc0444ad0c7be75ff573dac5b05a61da46d.jpg)  
14 s

(c) Communicate: Ask to see a fairy  
![](images/c4ae980c98906885a3b9ee472d6628b104cca4426ec87b67d316852c6351275d.jpg)

![](images/8dbd4df81b98467d930c70c3e85c3fb46c48ebb8d6a60658a8ad03d06600f5f6.jpg)  
4 s

![](images/c791745bb378effb010335385195fd99b9d2cb461b399a4a744ba4db61468e4c.jpg)  
9 s

![](images/a95e5426fe1badd42ea75fb8e0e9dab82eef4d17905da8dc3519e7c2e5b6e197.jpg)  
14 s  
Figure 10. Action-conditioned branching. A shared fountain opening leads to three continuations: looking into the water (a), tossing a coin (b), and meeting a fairy (c).

Reference-conditioned branching. Reference selection also shapes how an experience unfolds (Figure 11). Starting from a common bakery opening, the three continuations follow a similar entrance and greeting sequence, but the encountered character changes with the supplied reference. The examples illustrate control over who appears in an interaction while retaining the initial setting and a common interaction structure.

## (a) Clara · Teacher

![](images/199b1a03fc2e66141560d586c557c10d77b78e7a98f5ebe67f3cd068080c0896.jpg)  
Shared context + character

(b) Tom · Wagon driver  
![](images/76e2913d12914f1dddfe4078c02bf0bf0d3e71cfdda2af6bb467a8e25a7a56cc.jpg)  
6 s

![](images/4a4da5ceaa67dc2fee1499ba04a57217277920ecd5ab76d3b139f334ac47ea0e.jpg)  
12 s

![](images/bfe366605139fe7bf533b658ac8929f907b6eed4ce8be56b73deb64da2d28ee3.jpg)  
18 s

![](images/fc96896602621b0d9b013ef4d518bee42c742faf208c6d526f810de71df9db2a.jpg)

(c) Eli · Delivery worker

![](images/651a4f1cdfaa1bc6c8eea373245066c3d3bf67d20b6f951a1b1ea8022526ca13.jpg)  
6 s

![](images/87906e72f3cc22be8eaf1a06eea94ca03112ed1be504c8c19583210fb9ae6e6e.jpg)  
12 s

![](images/389a1e40f129d0d8ff72b66e11a7fa75263a5b586940bff630e1ee4a31b23b27.jpg)  
18 s

![](images/395ea809e177ab261d9d419236e582f4e8136e585095861498915b2326a5d5a6.jpg)  
Shared context + character

![](images/5c451ea524c0505f1643e73d5bf4960a70cc35b5be18584a9db110f01892e153.jpg)  
6 s

![](images/eede03e6a5a0833bccea08e668a402d4bf12ec5bb7a91d896a5651ecf8feb737.jpg)  
12 s

![](images/091d69bff9ad29a9dfe075b375782fc98fb2793899f89eba4be2372033f80b97.jpg)  
18 s  
Figure 11. Reference-conditioned branching. A shared bakery opening is paired with character references for Clara (a), Tom (b), and Eli (c), producing different encounters.

Embodied interaction. Figure 12 presents generated observations from a household robot’s viewpoint. The sequences show manipulators placing a mug, retrieving a ball for a dog, and watering a plant. Successive observations connect these actions to visible changes in nearby objects within a continuing scene. These generated scenarios illustrate a potential observation interface for embodied agents; they do not measure physical robot task success.

(a) Mug placement  
![](images/da1cc7b73af49b10ce66c1c7916f5824573c43135ab77dcc66b36286b7cfb63c.jpg)  
0 s

(b) Fetching a ball  
![](images/f5de6ea17bc11f86630f6a7fae89df447d244d61c994ad66e93ed0cda1a2baf4.jpg)

![](images/e3fceaa224857304eae392236dda531ddfa4e18e2eca6890385ffcf1c64481c4.jpg)  
0 s  
8 s

18 s  
![](images/615b21bd7ea0e5319db1f761027c695b831063691df0024a0339136463a81e81.jpg)

![](images/21a0a1403ca42f449927b42b05fc1f0834b40dd2f34947f4999d587e051cab76.jpg)

![](images/26a2ed83c1058d633c182175cd04e85903bbca03932b0912573e9b5c6836d78c.jpg)

20 s  
![](images/f905df52803cfdf70c95f16cb1bbdf4ea46ad6b85bea851772e08b8bac62bcf7.jpg)

17 s  
9 s  
![](images/fb3e8948fc4f0687f0d82eb3e00fcd8bb57466ba1d6698e5382fc4108d5679a6.jpg)  
25 s

(c) Plant watering  
![](images/eaf2c064208bf5e0148f132991eaf9b8369f89c39d7dcf29ea48c7a34d8c2d4c.jpg)  
0 s

![](images/92bae0e48ef7dfd206b313ed44139c6fe210356ab8006b9705659d531e7ea094.jpg)  
10 s

![](images/ad828f84136a48676dd84b7fc50c06560538f9cb731b0b766119f0be0f89618d.jpg)  
16 s

![](images/4bbcb30357543c13292eb3bd41dcad0eb33fef004220094763b883a17485d443.jpg)  
33 s  
Figure 12. Embodied interaction. Generated first-person observations of mug placement (a), ball retrieval for a dog (b), and plant watering (c), viewed from a household robot.

## 5.3 DAO-World Engine Performance

We evaluate DAO, the world engine of XGEN, through qualitative system demonstrations and ablation experiments, focusing on autonomous agent behavior, first-person views within a shared world, and persistent world-state evolution.

(a) Path blockec  
![](images/5e6490816498491ae5fea69f8325b6eeafec4a687c4b39f52ee18bf6ad11947f.jpg)

(b) Intent misreac  
![](images/1709f8169d671442187362f9f0a5f236bd4dcc5aedfc465b3ffd2664ad51c038.jpg)  
(c) Partial witness

(d) Clara approaches  
![](images/d62127003387afcef08614d4e6f9464302fd8a207fcb7284e5476b1c651332eb.jpg)  
(e) Evidence checked

![](images/e67528ce52a001dea656509d0d419bacd932de6d2d5586bb667300eb37e1a36d.jpg)

![](images/0099dd7cae2a81bb8e6ef52878c30ce89b9a23530e118775f10e1b04a46dcada.jpg)

(f) Social repair  
![](images/11d06d270a600ef501119abd31880ef240a34aa4dc9061c7aaa03f2c32e7660a.jpg)  
Figure 13. Agents act autonomously and change the world. Six successive events driven by agent intentions in DAO: (a) path blocked, (b) intent misread, (c) partial witness, (d) Clara approaches, (e) evidence checked, and (f) social repair. Local observations and accumulated evidence guide autonomous actions from conflict to reconciliation.

Agents autonomously. Unlike interaction schemes that require explicit user input to trigger events, DAO supports agents that initiate actions autonomously according to the world’s configuration, their own memories, local observations, and available information. Figure 13 illustrates autonomous, intentiondriven interaction among DAO agents. Rowan moves a handcart to clear a blocked lane (a), but Eli interprets the displaced parcels as deliberate interference, triggering an argument (b). Nori witnesses part of the event (c), and Clara intervenes on Eli’s behalf (d). Physical traces and witness evidence subsequently clarify Rowan’s intention (e), leading to reconciliation (f). This progression arises from agents pursuing their own intentions and revising their actions in response to local observations and accumulated evidence, while DAO maintains the evolving shared world state.

![](images/f25d3273c50c14bdaa883f82ac3044834057f914d6281f01ef21dad365b3d927.jpg)

![](images/0b7c2b878200c1ebb1ceb0fa8baff44be926936cb0f77f33ea1b07122b089372.jpg)

![](images/79cce36da2f67573bb567987b60108541386b797b8bb4d76c4375d4f2fdf3971.jpg)

![](images/2ff8a1f5d78616281f4f18d88fc70e1fa0ee38cd2164c3d1653eeb32c95e3069.jpg)

![](images/31338dc657f0f5db85ead63865fbac2fc11647cc7ed65327228986a4ee59a314.jpg)

![](images/714d89f2e092d053bfccafef180ac0edd254ca464f880789894cc5b16ab531b2.jpg)  
Figure 14. Different viewpoints of agents in a shared world. A shared interaction, shown from the viewpoints of Clara, Eli, and Nori.

One event, different experiences. DAO supports first-person observations from different agents at the same simulation time, grounding their views of the world and its ongoing interactions in a shared temporal and world-state context. Figure 14 presents the same interaction from the first-person perspectives of Clara, Eli, and Nori. Clara and Eli experience their encounter from opposing viewpoints, while Nori observes it from across the road. DAO derives observer-specific information from a common world state and conditions JING to generate each character’s local visual experience.

![](images/e28984cbc250388b99933440e3cc4fb27a4c6ce47e41e87ac210c972da473f5d.jpg)  
Figure 15. JING generation results with and without DAO.

Grounding JING in an evolving world. DAO provides JING with a persistent, continuously evolving world that maintains an explicit state and supports agents with memory and autonomous, intention-driven behavior. Figure 15 compares JING generations with and without DAO in a departure-and-revisit scenario. Without DAO (top row), the generated revisit depicts the previously ignited napkin as still burning, with no visible evidence of an intervening agent response. With DAO (bottom row), autonomous agent–environment interactions update the world state during the observer’s absence: the attendant extinguishes the fire, and JING conditions the revisit generation on the resulting state. This qualitative comparison illustrates how ongoing world simulation allows actions outside the observer’s field of view to influence subsequent observations, supporting causal continuity across departures and revisits.

## 6 Conclusion

We present XGEN, a Generative World Simulation framework for sustained human and agent interaction that integrates DAO, a computable world engine, with JING, an Egocentric Interactive Experience Model. DAO maintains and updates shared world state while preserving the consequences of actions; JING generates continuous first-person audiovisual experiences conditioned on references, observation history, and current actions. Together, they connect actions, state updates, and observations in a continuous loop, enabling exploration, object manipulation, and character communication grounded in prior interactions. By coupling world-state evolution with individual experiences, DAO and JING extend audiovisual generation toward simulation of the world behind it. Our long-term goal is the persistent shared world envisioned in OASIS, where encounters carry history, choices have lasting consequences, and social activity continues beyond any individual’s presence. In this vision, humans and agents gain experience and shape a world that evolves through their actions and continues to develop in their absence. Future work will focus on improving motion robustness and long-term memory, reducing response latency, and strengthening consistency between shared world state and generated audiovisual content.

## References

[1] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[2] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

[3] OpenAI. Sora 2 is here, 2025. URL https://openai.com/index/sora-2/.

[4] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[5] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

[6] MiniMax. MiniMax H3: An open model breaking the boundaries between tasks and modalities, 2026. URL https://www.minimax.io/blog/minimax-h3.

[7] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[8] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026.

[9] Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu, Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang, Xue Zheng, Rui Sun, et al. Abot-world-0: Infinite interactive world rollout on a single desktop gpu. arXiv preprint arXiv:2607.19191, 2026.

[10] Junchao Huang, Guian Fang, Shengju Qian, Xianghao Kong, Zhuoran Zhao, Wei Huang, Yihua Du, Zixin Zhang, Justin Cui, Yuchao Gu, et al. Solarwm: Open data and scalable training for long-horizon video world models. arXiv preprint arXiv:2609.02886, 2026.

[11] AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Ruicong Liu, et al. Alayaworld: Long-horizon and playable video world generation. arXiv preprint arXiv:2607.06291, 2026.

[12] Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang, Haoran Li, Xiaoxiao Ma, et al. Echowm: Open and enterable omnimodal world models. arXiv preprint arXiv:2608.23189, 2026.

[13] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025.

[14] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.

[15] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.

[16] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. Advances in Neural Information Processing Systems, 38:49632–49652, 2026.

[17] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[18] Google DeepMind. Genie 3: A new frontier for world models, 2025. URL https://deepmind.google/blog/ genie-3-a-new-frontier-for-world-models/.

[19] Runway. Introducing GWM Worlds 2, 2026. URL https://runway.com/research/introducing-gwm-worlds-2.

[20] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: a text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.

[21] Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Linfeng Zhang, et al. Hunyuan-gamecraft-2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.

[22] Haohe Liu, Gael Le Lan, Xinhao Mei, Zhaoheng Ni, Anurag Kumar, Varun Nagaraja, Wenwu Wang, Mark D Plumbley, Yangyang Shi, and Vikas Chandra. Syncflow: Toward temporally aligned joint audio-video generation from text. arXiv preprint arXiv:2412.15220, 2024.

[23] Yuxi Wang, Wenqi Ouyang, Tianyi Wei, Yi Dong, Zhiqi Shen, and Xingang Pan. Hand2world: Autoregressive egocentric interaction generation via free-space hand gestures. arXiv preprint arXiv:2602.09600, 2026.

[24] Jinkun Hao, Mingda Jia, Ruiyan Wang, Xihui Liu, Ran Yi, Lizhuang Ma, Jiangmiao Pang, and Xudong Xu. Egosim: Egocentric world simulator for embodied interaction generation. In European Conference on Computer Vision, pages 598–614. Springer, 2026.

[25] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

[26] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38: 167283–167308, 2026.

[27] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video diffusion models. Advances in Neural Information Processing Systems, 38:30546–30566, 2026.

[28] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

[29] Yaofeng Su, Yuming Li, Zeyue Xue, Jie Huang, Siming Fu, Haoran Li, Haoyang Huang, and Nan Duan. Omniforcing: Unleashing real-time joint audio-visual generation. In European Conference on Computer Vision, pages 566–584. Springer, 2026.

[30] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623. IEEE, 2024.

[31] Yuqian Zhou, Zhenghong Zhou, Zongze Wu, Cameron Smith, Richard Zhang, Jiebo Luo, Eli Shechtman, and Zhe Lin. Editstream: A unified autoregressive framework for interactive video generation and editing. arXiv preprint arXiv:2608.21424, 2026.

[32] Zijun Lin, Zeqing Wang, Cheston Tan, Bihan Wen, and Yeying Jin. Stateplay: State-aware game world models for mechanics-consistent generation. arXiv preprint arXiv:2607.26754, 2026.

[33] Ziqi Cai, Siqi Yang, Yimu Wang, Zixian Gao, Yunheng Liu, Shuchen Weng, Erwin Wu, Kaipeng Zhang, and Boxin Shi. Mass: Multiplayer world models with authoritative shared state. arXiv preprint arXiv:2608.06257, 2026.

[34] Zian Meng, Zhen Li, Chuanhao Li, Qiang Li, and Kaipeng Zhang. Marionette: Predicting world states, rendering geometry, painting appearance. arXiv preprint arXiv:2608.14530, 2026.

[35] Zheng-Hui Huang, Guixu Lin, Jiacheng Lin, Yi-Chuan Huang, Ruihan Yu, Muyao Niu, Siqi Yang, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang, et al. Programmable world model. arXiv preprint arXiv:2609.10540, 2026.

[36] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, et al. Spatialvid: A large-scale video dataset with spatial annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 42592–42603, 2026.

[37] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. Advances in Neural Information Processing Systems, 38, 2026.

[38] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22160–22169. IEEE, 2024.

[39] Qwen Team. Qwen3.6, 2026. URL https://huggingface.co/collections/Qwen/qwen36.

[40] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025.

[41] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[42] Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989.

[43] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-end training for autoregressive video diffusion via self-resampling. In European Conference on Computer Vision, pages 324–344. Springer, 2026.

[44] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation. arXiv preprint arXiv:2605.25874, 2026.

[45] Kling AI. Kling VIDEO 3.0 model user guide, 2026. URL https://kling.ai/quickstart/klingai-video-3 -model-user-guide.

[46] Alibaba Cloud. Video generation and editing: Wan 2.7, 2026. URL https://help.aliyun.com/en/model-studio/ video-generate-edit-model.

[47] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.

[48] Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alaya-evoke: From linear-scaling supervision to endless world. arXiv preprint arXiv:2608.13546, 2026.

[49] Mingyang Chen, Shengdong Chen, Xiaoxiao Fu, Bosheng Gong, Haoyuan Guo, Bowen Li, Jiawen Li, Kejun Li, Tianpeng Li, Yin Liu, et al. Zing-0.5: Toward playable worlds with real-time joint action and text control. arXiv preprint arXiv:2609.17909, 2026.

[50] HiDream.ai. HiDream-O1-World: Interactive world model with spatial reasoning, 2026. URL https: //www.linkedin.com/posts/hidream-ai\_ai-worldmodels-agi-activity-7501226255133036544-p2Fj.

## 7 Contributors and Acknowledgments

All contributors are listed in alphabetical order by their last names.

Team XGEN Guohao Cai, Haoxuan Che, Frank Fan, Chenran Huang, Junjia Huang, Jiayi Jiang, Yanheng Li, Leon Liu, Shaoteng Liu, Xi Liu, Feng Nie, Zhefan Rao, An Wang, Hanwen Wang, Pengxiang Yan, Jieming Yu, Qian Zha, Wentao Zheng