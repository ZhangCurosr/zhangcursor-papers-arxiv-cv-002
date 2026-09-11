# World in World: Explore the World with World Models

Chenxi Song Yanming Yang Chi Zhang†

AGI Lab, Westlake University

https://chenxi-song.github.io/worldinworld

## Abstract

Autoregressive video world models enable interactive, long-horizon exploration, but flexible control remains challenging. Exploring a source video from new viewpoints requires the generated rollout to remain synchronised with the recorded event, place observed content in the requested view, plausibly complete newly exposed regions, and recover previously generated appearance on revisits. Existing methods typically address these requirements through task-specific modules or additional training. We present World in World, a training-free inference-time interface that converts heterogeneous control evidence into camera- and time-labelled clean visual states, which are read through the native self-attention of a frozen causal video model. The evidence comprises source-video observations, target-view scene projections, geometry renderings that guide completion of newly exposed subject regions, and retrieved generated states beyond the rolling cache. Each evidence source carries token-level support and its own availability schedule. A correspondence router combines persistent point identities with geometry to establish token correspondences, guiding supported queries towards matching source-video tokens. Evidence-wise attention CFG (EWA) then independently regulates each auxiliary channel’s additional contribution using attention responses from the same denoising forward pass. The shared interface supports camera-controlled rerendering, long-horizon revisiting, and human-motion transfer with the same frozen backbone. We evaluate World in World on cameracontrolled video rerendering under diverse viewpoint changes, assessing perceptual quality, temporal consistency, and camera-following accuracy.

## World in World

Explore the world within a video, with world models

A video is not just a window, but a world to explore.  
![](images/a4bde8b59e21eff99511b8ad995a5ecbb4af1f5d860d1653fae532a16bc7e78f.jpg)  
Arbitrary View Exploration Explore with user-specified camera trajectories  
Event-consistent Generation Preserve the dynamics of the recorded event  
Plausible World Completion Generate newly observed regions beyond the source video  
Temporal Consistency Revisit and recover previous visual states

Fig. 1. World in World: Explore the world within a video. Video generation becomes an ongoing exploration of an evolving visual world. In World in World, this exploration is grounded in the world of a given video, with the recorded event anchoring the appearance and dynamics of each new observation.

## 1. Introduction

Recent advances in video generation have improved visual quality, motion realism, and temporal coherence, and have supported the development of video world models built on autoregressive generation [1–6]. While most text- or image-conditioned systems generate a finite clip from conditions specified before inference, video world models support an ongoing interactive process. They continually predict subsequent observations from observed or generated visual states, allowing users to change camera positions, viewing directions, and control actions throughout a rollout [7, 8]. Video generation thus extends from producing a recording of an event to supporting exploration of an evolving visual world, with applications in interactive content creation, virtual production, game generation, and embodied-agent simulation [9–13]. In particular, as large pretrained world models acquire stronger visual and motion priors, a key question is how to flexibly extend their controllability without retraining, enabling the same general world model to accept diverse forms of visual control and world information and make fuller use of its pretrained capabilities.

Supporting such diverse controls requires a world model to integrate complementary visual evidence from observations, geometry, and generated history, which differ in representation, spatial coverage, and temporal relevance. As viewpoints and scene states change, generation must remain synchronised with the dynamics specified by the visual evidence while maintaining correct spatial placement and occlusion relationships in the target view. Large viewpoint changes require using the model’s generative prior to complete unobserved scene and object surfaces, while long-horizon revisits require recovering earlier appearance and spatial layout. The challenge is therefore to make relevant evidence available and use it at the appropriate locations and times throughout a rollout.

Existing approaches have explored camera conditioning, source-video rerendering, geometric control, and long-term memory [5, 6, 14–18]. Many rely on control-specific pathways or representations, so supporting new evidence types or combinations often requires redesign or additional training. This motivates a shared interface through which the same pretrained world model can use heterogeneous current and historical visual evidence without learning a separate pathway for each control, which is the focus of our paper.

Our key observation is that causal video world models equipped with a clean-state cache already have a shared entry poin for visual information: their native self-attention. During autoregressive generation, the model reads the initial observation and recently finalised outputs as clean visual states, while camera and temporal encodings specify their viewpoints and tempora positions. This suggests that external control information can be converted into the same representation: clean visual states associated with camera poses, event times, and valid spatial regions, made accessible through the existing attention layers. Instead of adapting the model to each new control, we express different controls in a visual representation that the pretrained model alread processes. Under this view, extending world-model control becomes a visual evidence construction and orchestration problem, rather than a model adaptation problem.

Based on this idea, we introduce World in World (WiW), a training-free visual-evidence interface for controlling frozen causa video world models (Figure 1). WiW converts various visual conditions, such as source observations, target-view projections, rendered geometry, and generated history, into clean visual states annotated with camera, temporal, and spatial-validity information. These states can be directly accessed through the model’s native self-attention, allowing different sources of evidence to guide generation where and when they are reliable. Our method is entirely training-free, requiring neither updates to the pretrained world model nor learned control-specific modules. Under this unified interface, flexible world-model control reduces to three complementary problems: constructing useful visual evidence, localising the relevant evidence, and regulating its influence on generation.

To provide the frozen world model with the complementary information required for controllable long-horizon rollouts, we first construct multiple forms of visual evidence for different control requirements. Our core idea is to let each evidence source contribute the information it can provide most reliably, while expressing all of them through the same clean-state interface that the pretrained model already understands. Concretely, source-video observations provide appearance and content references from the recorded event. Since these observations do not explicitly determine where their content should appear under the targe camera, target-view scene evidence uses depth-based projection to place observed appearance in the requested view and provide spatial guidance within geometrically supported regions. When large camera motions expose subject surfaces that are absent from the source observations, rendered geometry evidence supplies target-view shape and appearance proposals for the corresponding event state, helping the pretrained model complete newly visible subject surfaces. Finally, because a finite rolling cache eventually removes access to earlier generated states, we maintain a rollout-wide history and retrieve relevant and diverse archived states as generated evidence when the camera returns. By assigning different control requirements to complementary evidence sources, WiW provides the frozen backbone with appearance references, spatial guidance, completion cues, and long-range visual contex through a single shared visual interface.

To ensure that the model reads the right evidence and uses it with appropriate strength, we further regulate how visual evidence participates in native self-attention. Our core idea is to separately control where to read and how strongly to use the available evidence, addressing errors in evidence localisation and differences in evidence reliability. For where to read, appearance-based attention can become ambiguous under large viewpoint changes, dynamic motion, and repeated textures. We therefore introduce correspondence-guided attention routing (CGAR), which uses persistent point identities and camera geometry to route current queries towards geometrically corresponding source-video tokens when valid matches are available. For how strongly to use, different evidence sources can vary in reliability across spatial regions and denoising stages. We introduce evidence-wise attention CFG (EWA), which compares native and evidence-conditioned attention responses to independently amplify compatible and complementary information while suppressing excessive guidance. Both mechanisms operate directly within the model’s native self-attention, and EWA reuses responses from the same denoising forward pass, introducing no additional network function evaluations (NFE) for guidance.

Through this formulation, WiW provides a unified, training-free framework for extending the control capabilities of frozen world models through visual evidence, without introducing control-specific training or adaptation. Notably, beyond exploring worlds freely generated by a world model, WiW enables exploration within the world of a given video: users can navigate the recorded dynamic world along new camera trajectories while preserving the appearance and temporal progression of the origina event. We instantiate WiW on the publicly released causal-fast checkpoint of LingBot-World 2.0 [4], with all pretrained parameters frozen, and evaluate camera-controlled video rerendering on DAVIS and OpenVid-1M [19, 20] across diverse viewpoin changes. Beyond rerendering, WiW supports applications including bullet-time generation, video stabilization, video editing, and K/V sharing between two generation cases produced by the same frozen model, demonstrating the versatility of the proposed interface.

Our main contributions are:

• We introduce WiW, a training-free visual-evidence interface for flexibly extending the control capabilities of frozen causal video world models.

• We develop complementary visual evidence that enables event-synchronised, spatially aligned, and long-horizon-consistent exploration of dynamic video worlds.

• We introduce correspondence-guided attention routing (CGAR) and evidence-wise attention CFG (EWA) to localise and regulate heterogeneous visual evidence within native self-attention.

• We demonstrate flexible exploration of given video worlds across diverse camera trajectories and downstream applications using a single frozen world-model backbone.

## 2. Related work

## 2.1. World models

A video world model predicts the visual observations an observer would receive while moving through an environment, given an initial observation and a requested camera or action. In static environments, a key requirement is to maintain consistent scene structure and appearance as the viewpoint changes and previously seen regions are revisited. One line of work uses explici scene representations, such as point clouds, meshes, or Gaussians, to organize observations in a shared coordinate system and render them from the target camera. These representations provide a clear spatial reference, but require completing unobserved regions and maintaining the scene state as exploration continues [1, 21–33]. Another line of work directly predicts target-view observations with video models, using learned generative priors to handle viewpoint changes and newly visible regions [34–38]. When the environment changes over time, the model must also maintain consistency between viewpoint changes, scene structure, and event progression. In dynamic environments, methods need to generate observations that match both the target viewpoint and the current state of the recorded or simulated event. Previous work has studied this problem through dynamic scene generation, source-video re-observation, and continuous world modeling [6, 17, 39–47]. Autoregressive and self-forcing methods further extend scene generation to continuous rollouts, requiring models to maintain consistent appearance, spatial relationships, and event states over long explorations [2, 4, 8, 48–51]. Pretrained video models have appearance and motion priors that generalize to new scenes, but their native conditioning interfaces are usually determined by the inputs used during training. The visual-histor pathway can itself serve as a control interface: Warp-as-History [52] feeds camera-warped observations as pseudo-history, aligns their temporal positions with target frames. We build on a pretrained causal video model and study how to reuse its native self attention mechanism for reading visual states, allowing it to accept additional control information while keeping its parameters frozen.

## 2.2. Conditioning mechanisms for video world models

Existing video world models usually use dedicated conditioning mechanisms to introduce different types of control into generation. For camera control, prior methods represent camera trajectories as rays, poses, positional encodings, projected features, or rendered geometric proxies, and use corresponding conditioning modules to guide generation [53–60]. Source-video rerendering extends control from camera motion to re-observation of dynamic content. These methods often train additional modules for source-video inputs to bring the original event’s appearance and dynamics into the target view [6, 17, 61–63]. Similarly, the concurrent work Wonder supports image- and video-conditioned world generation by jointly training a rendered control field, a sparse memory, and a distilled causal student to combine multiple conditions [5]. These methods show that dedicated conditioning pathways can support specific control tasks, but different types of control often use different input interfaces and training procedures Beyond current observations and external controls, continuous exploration also requires access to scene information generated earlier. Since causal models typically read only a limited recent context, prior work uses explicit geometric memory, persistent states, or historical information retrieval to recover scene content beyond the current context and constrain subsequent generation [18, 64–70]. Memory therefore helps maintain long-term consistency and can also serve as a visual condition alongside current observations and geometry. WiW follows this idea by including historical information in a unified visual conditioning framework, allowing scene evidence from different sources to jointly guide subsequent generation.

![](images/9c4ca2aad90f9871749b6e081cce7f2e1ce26da121862c8b00f8d8ba8a499df3.jpg)  
Fig. 2. Overview of the World in World pipeline. Target-view projections, rendered geometry, and retrieved historical states provide temporary visual evidence. Correspondence-guided attention routing connects queries to matching source-video tokens, and evidence-wise attention CFG regulates each channel’s additional contribution. Temporary evidence blocks are removed after each chunk; finalized outputs enter the rolling cache, and their visual features are archived after eviction.

## 3. Method

We propose World in World (WiW), a unified visual-evidence interface for extending the control capabilities of frozen causal video world models. We illustrate the framework through camera-controlled video rerendering: given a source video of a dynamic event and a target camera trajectory C, we aim to rerender the event from the requested viewpoints while preserving its appearance and temporal progression. To this end, we convert source observations, geometry, and generated history into clean visual states with camera, temporal, and spatial-validity information. The frozen model can then read this evidence through native self-attention, without additional training or learned control-specific adapters

Figure 2 summarizes the overall pipeline. We first define the shared evidence interface and explain how visual evidence participates in native self-attention (Section 3.1). Building on this interface, we project source observations into the target view to provide layout references (Section 3.2), use rendered geometry to guide completion of newly exposed subject surfaces (Section 3.3), and retrieve generated history for consistent long-horizon revisits (Section 3.4). Together, these sources provide complementar references for generation. Attention routing and evidence-wise attention CFG then localize relevant evidence and regulate its influence, respectively (Section 3.5).

## 3.1. Control through Visual Evidence

When generating the current chunk, our causal video backbone reads the initial observation and recently finalized states throug native self-attention. We use this existing pathway to convert multiple forms of visual evidence, including source-video observations, target-view scene projections, rendered geometry, and generated history, into the same type of states that the model already reads. These sources differ in representation, valid spatial extent, and temporal coverage. Their shared representation must therefore specify evidence content, camera and temporal information, spatial support, and channel activation across denoising stages.

To represent this information consistently, at denoising step s, converter $\Psi _ { b }$ maps the raw evidence $E _ { b }$ of channel b and target camera trajectory $C$ to:

$$
\Psi _ { b } ( E _ { b } ; C , s ) = \Big ( X _ { b } , P _ { b } , \omega _ { b } , \gamma _ { b } ^ { ( s ) } \Big ) .\tag{1}
$$

Here, $X _ { b }$ denotes the visual content of the evidence; $P _ { b }$ records per-frame camera intrinsics, poses, and event-time indices;   
$\omega _ { b j } \in [ 0 , 1 ]$ specifies the spatial support of evidence token $j ;$ and $\gamma _ { b } ^ { ( s ) } \in \{ 0 , 1 \}$ determines whether the channel is active at step s.

To turn this representation into attention features that the model can read, we feed the clean visual states corresponding to $X _ { b } ,$ together with camera and temporal information $P _ { b }$ , into the frozen video backbone with the diffusion timestep set to $t = 0$ . A single network forward pass extracts the keys $K _ { b } ^ { \ell }$ and values $V _ { b } ^ { \ell }$ from each attention layer, which we cache as clean K/V. During target-chunk generation, queries from the current denoising states read these cached features through self-attention, allowing external visual evidence to guide generation. All evidence types introduced below use this procedure to obtain their attention features.

These K/V features participate in attention through either the native block or temporary auxiliary blocks. The native block retains the backbone’s 18 latent-frame-state slots: six source anchors, eight recent-history states, and four states in the current chunk. Source-video evidence occupies the source-anchor slots, while other evidence is attached as temporary auxiliary attention blocks and removed once the current chunk is finalized. All blocks reuse the frozen backbone’s query/key/value (Q/K/V) projections, while preserving its native text- and camera-conditioning pathways.

When visual evidence enters attention, its temporal relationship to the current generation states needs to be specified. We therefore set the temporal coordinates of rotary position embeddings (RoPE) and apply the corresponding rotations to attention queries and keys, incorporating relative temporal information into attention.

Spatial support weights regulate the contribution of evidence tokens by scaling their unnormalized attention weights by $\omega _ { b j }$ Channel activation $\gamma _ { b } ^ { ( s ) }$ further determines the denoising stages in which each evidence source participates.

## 3.2. Target-View Scene Evidence

Source-video observations provide appearance references for the recorded event, but they remain in the source-camera view and do not explicitly determine where their content should appear in the target image. Camera motion changes projected positions and occlusion relationships, while ambiguous cross-view appearance matches can cause visible structures to drift. To provide an explicit spatial layout reference, we project source observations into the target view, allowing the model to read observed appearance aligned with the requested camera.

To construct this reference for target frame m and temporally aligned source view r, we use source RGB $I _ { r , m } ^ { \mathrm { s r c } }$ and estimate its depth $D _ { r , m } ^ { \mathrm { s r c } }$ with DepthCrafter [71]. We back-project source pixels into 3D and reproject them from source camera $C _ { r , m } ^ { \mathrm { s r c } }$ to target camera $\dot { C } _ { m } = ( A _ { m } , T _ { m } )$ , where $A _ { m }$ denotes camera intrinsics and $T _ { m }$ is the absolute camera-to-world pose:

$$
( X _ { r , m } ^ { \mathrm { p r o j } } , M _ { r , m } ^ { \mathrm { p r o j } } , Z _ { r , m } ^ { \mathrm { p r o j } } ) = \mathrm { P r o j } ( I _ { r , m } ^ { \mathrm { s r c } } , D _ { r , m } ^ { \mathrm { s r c } } ; C _ { r , m } ^ { \mathrm { s r c } }  C _ { m } ) .\tag{2}
$$

Projection uses a shared coordinate system and scale. The outputs $X _ { r , m } ^ { \mathrm { p r o j } } , M _ { r , m } ^ { \mathrm { p r o j } }$ , and $Z _ { r , m } ^ { \mathrm { p r o j } }$ are target-view RGB, a binary visibility mask, and target-camera depth, respectively. Following the shared interface, we encode the projected RGB with target camera and event-time information, and use visibility to determine the evidence’s spatial support.

Since occlusion and local geometric errors can affect the projection, we determine its spatial support from visibility and geometric reliability, then convert it into token-level support weights $\omega _ { \mathrm { p r o j } }$ . These weights allow the model to use reliable targetview layout references while reducing the influence of uncertain regions.

## 3.3. Rendered Geometry Evidence

Target-view projection reorganizes content already present in the source observations, but its coverage remains limited by sourceview visibility. When camera motion exposes subject surfaces not observed in the source video, these regions lack direct shape and appearance references, and their generation may deviate from the subject’s structure or current pose. To provide evidence for

completing such regions, we use a renderable subject representation to produce shape and appearance proposals from the target camera at the corresponding event time. Given subject state $G _ { m }$ at event time m and target camera $C _ { m } .$ , rendering yields:

$$
( X _ { m } ^ { \mathrm { g e o } } , M _ { m } ^ { \mathrm { g e o } } , Z _ { m } ^ { \mathrm { g e o } } ) = \mathrm { R e n d e r } ( G _ { m } , C _ { m } ) .\tag{3}
$$

Here, $X _ { m } ^ { \mathrm { g e o } } , M _ { m } ^ { \mathrm { g e o } }$ , and $Z _ { m } ^ { \mathrm { g e o } }$ denote RGB, a geometric support mask, and target-camera depth, respectively. The rendering is encoded through the shared interface, and its support mask is converted into token-level support weights.

For example, when the subject is human, we represent its state as $G _ { m } = ( \mathcal { A } , \Theta _ { m } )$ . Here, A is an avatar reconstructed with LHM++ [72] from sharp, minimally occluded full-body source-video crops, and $\Theta _ { m }$ contains the per-frame SMPL-X [73] parameters that drive the avatar. We align the avatar and body model to the projected scene’s coordinate frame and depth scale using depth correspondences in the source view. LHM++ provides target-view RGB and support masks for observed and inferred unseen surfaces, while the aligned SMPL-X geometry provides depth for resolving occlusion between subjects. We also compare this depth with the projected scene depth to remove unreliable background projections near subject boundaries.

## 3.4. Retrieved Generated Evidence

Scene projections and geometry renderings provide observed and reconstruction-based evidence for the current target view. As generation progresses, previously generated appearance and layout also become important references for subsequent frames. The native rolling cache retains only recent states, so states corresponding to an earlier region may have been evicted when the camera returns, leaving their generated content inaccessible to the model. To recover these references, we maintain a rollout-wide history bank and retrieve finalized states relevant to the current view.

The history bank accumulates evidence as the rolling cache is updated. For each evicted finalized state, we archive its per-layer clean K/V with the associated camera pose and temporal index. Since the bank grows throughout generation, we select a bounded number of relevant states to participate in attention for each target chunk. Specifically, we rank historical states by target-view surface coverage and viewing-direction compatibility, then select a diverse top- $K _ { \mathrm { r e t } }$ subset to reduce redundant references. The model reads the retrieved K/V directly through temporary auxiliary attention blocks, recovering historical appearance and layout beyond the rolling cache to maintain scene consistency.

## 3.5. Attention Routing and Guidance

The construction and retrieval steps above provide complementary evidence for the source event, target-view layout, subject geometry, and generated history. However, making evidence accessible does not ensure that queries locate the relevant positions within it. Viewpoint changes and dynamic motion alter the appearance of a surface, while repeated textures can make distinc locations look similar, introducing ambiguity into appearance-based attention. We therefore first use geometric correspondences to match current generation tokens with source-video tokens and guide evidence reading, then regulate each channel’s additiona influence through its attention response.

Correspondence-guided attention routing. To associate queries from current generation tokens that have valid geometric correspondences with matching source-video locations, we introduce correspondence-guided attention routing (CGAR). We track points with persistent identities in the source video and combine depth and camera information to establish token-level correspondences between the current view and the source video. For a geometrically matched query i and source-video key $j ,$ we add the logarithm of correspondence weight $\beta _ { i j } > 0$ to the routing attention score:

$$
\ell _ { i j } ^ { \mathrm { c o r r } } = \frac { \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } } { \sqrt { d _ { h } } } + \log \beta _ { i j } .\tag{4}
$$

Here, $d _ { h }$ is the attention-head dimension, and $\beta _ { i j } > 0$ controls the contribution of the correspondence. We combine the routing response with the other attention-block responses through joint normalization, allowing current generation queries to read geometrically matched source-video evidence.

Evidence-wise attention CFG. The evidence sources constructed above participate in video generation through self-attention during denoising. Different auxiliary evidence channels provide constraints on target-view layout, subject structure, and generated history, and the information they provide also differs in reliability and importance. Their influence on generation therefore needs to be controlled independently. Using all evidence with the same strength makes it difficult to balance adherence to different conditions with the model’s generative prior. Inspired by classifier-free guidance (CFG) [74] and NAG [75], we propose evidencewise attention CFG (EWA). EWA independently adjusts the guidance strength of each evidence source at the level of attention responses, strengthening useful control signals while preserving the content quality and naturalness of native generation.

Specifically, at the same attention layer and for the same queries, let $\mathbf { o } _ { 0 }$ denote the response obtained by attending only to the native block, and let $\mathbf { o } _ { e }$ denote the jointly normalized response obtained by attending to both the native block and evidence e. Directly amplifying $\mathbf { o } _ { e }$ would also amplify its component along the native generation direction. We therefore strengthen only the complementary direction introduced by the evidence relative to the native response. We first remove the projection of $\mathbf { o } _ { e }$ onto $\mathbf { o } _ { 0 } .$ , modulate the remaining component by the nonnegative cosine similarity between the two responses, and use an independen guidance strength $g _ { e }$ to control the correction magnitude. The updated response for evidence e is:

$$
\widetilde { \bf o } _ { e } = { \bf o } _ { e } + g _ { e } \operatorname* { m a x } ( 0 , \cos ( { \bf o } _ { e } , { \bf o } _ { 0 } ) ) \left[ { \bf o } _ { e } - \mathrm { P r o j } _ { { \bf o } _ { 0 } } ( { \bf o } _ { e } ) \right] .\tag{5}
$$

Here, $\cos ( \mathbf { o } _ { e } , \mathbf { o } _ { 0 } )$ is the cosine similarity between the two responses. Removing the projection along the native response direction allows EWA to strengthen the complementary information provided by the evidence without repeatedly amplifying the model’s existing response.

We then add the evidence-specific corrections to the joint response of all active attention blocks and bound the output magnitude using the norm of the native response $\mathbf { o } _ { 0 }$ as a reference, preventing excessive guidance from disrupting generation.

EWA operates directly on attention responses within the same denoising forward pass. Its computation reuses existing attention-block outputs and normalization results, without requiring a separate full denoising-network forward pass for each evidence source. It therefore adds no network function evaluations (NFE) for guidance.

## 4. Experiments

We evaluate World in World through camera-controlled video rerendering, where a source video serves as visual evidence for exploring the recorded world along a target camera trajectory. We further examine long-horizon revisiting and human-motion transfer, and ablate individual components to assess how the same frozen backbone uses complementary visual evidence.

## 4.1. Experimental Setup

Implementation. We instantiate World in World on the publicly released causal-fast checkpoint of LingBot-World 2.0 [4], keeping all pretrained parameters frozen. Visual evidence is constructed and attached at inference time, without additional training or learned control-specific adapters.

Baselines. Following prior work on camera-controlled video rerendering, we select videos from DAVIS [19] and OpenVid 1M [20] and compare with ReCamMaster, TrajectoryCrafter, WorldForge, InSpatio-World, UniWorld-View, and CameraAnything [6, 14–17, 76]. Each baseline uses its official implementation, released checkpoint, and recommended configuration. ReCam-Master and TrajectoryCrafter generate 81 and 49 frames, respectively, as required by their official implementations; the remaining methods use their officially recommended frame counts. All methods receive the same source video and continuous target camera path, with camera parameters converted to the convention required by each method. Before evaluation, we align the generated videos across methods in terms of camera angles and frame count. All methods share the same metric preprocessing pipeline.

Metrics. We evaluate image fidelity using PSNR, SSIM, and LPIPS. To assess video quality, temporal consistency, and dynamics, we use the official VBench implementation [77] to report seven dimensions: Aesthetic Quality, Imaging Quality, Tempora Flickering, Motion Smoothness, Subject Consistency, Background Consistency, and Dynamic Degree. We additionally report Overall, computed as the average score of these seven dimensions.

Following InSpatio-World [6], we evaluate camera control using translation error (TransError) and rotation error (RotError). We independently estimate camera trajectories from the generated videos using Depth Anything 3 and ViPE [78, 79]. For each estimator, we compare the estimated trajectories with the prescribed target trajectories. The reported TransError and RotError are obtained by averaging the corresponding errors across the two estimators.

## 4.2. Quantitative Comparisons

Table 1 reports quantitative results for camera-controlled video rerendering on DAVIS and OpenVid-1M. Following previous state-of-the-art, we evaluate performance in terms of VBench, camera trajectory errors and image quality.

## 4.3. Qualitative Comparisons

Figure 3 compares camera-controlled rerendering under different camera motion. Each example presents the source video and representative outputs under the same target camera path.

## 4.4. Additional Applications

Our framework expresses different control requirements through a shared visual-evidence interface. Figure 4 extends this idea beyond rerendering to control over event timing, changes to appearance and motion, and the sharing of generated visual information. In the K/V-sharing example, Model A and Model B denote two instances of the same frozen model, each corresponding to a different generation case. Communication between the instances is implemented by passing cached K/V from one instance to the other as visual evidence to guide generation. Across these applications, different sources of evidence guide generation through the same native attention mechanism, with camera and time information specifying how that evidence relates to the requested output This allows us to extend the model’s control capabilities while keeping the backbone frozen, without task-specific training.

Table 1. Quantitative comparison of camera-controlled video rerendering on DAVIS and OpenVid-1M, evaluated using VBench dimensions, camera trajectory errors, and image fidelity metrics. Overall is the average score of all seven reported VBench dimensions. Bold and underline indicate the best and second-best results, respectively. Tied results share the same rank.
<table><tr><td rowspan="2">Method</td><td colspan="8">VBench ↑</td><td colspan="2">Camera errors ↓</td><td colspan="3">Image quality</td></tr><tr><td>Aesth.</td><td>Img.</td><td>Flick.</td><td>Smooth.</td><td>Subj.</td><td>Bg.</td><td>Dyn.</td><td>Overall</td><td>TransError</td><td>RotError</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>ReCamMaster</td><td>54.283</td><td>66.290</td><td>95.178</td><td>97.503</td><td>88.941</td><td>92.154</td><td>92.500</td><td>83.836</td><td>0.124289</td><td>7.8626</td><td>15.1383</td><td>0.444346</td><td>0.377962</td></tr><tr><td>TrajectoryCrafter</td><td>51.322</td><td>61.466</td><td>94.169</td><td>97.423</td><td>87.032</td><td>90.727</td><td>98.750</td><td>82.984</td><td>0.090812</td><td>5.3254</td><td>19.9689</td><td>0.625112</td><td>0.217875</td></tr><tr><td>WorldForge</td><td>49.915</td><td>62.356</td><td>94.262</td><td>96.934</td><td>84.683</td><td>90.017</td><td>98.524</td><td>82.384</td><td>0.079961</td><td>4.5021</td><td>19.6494</td><td>0.589250</td><td>0.218029</td></tr><tr><td>InSpatio-World</td><td>53.162</td><td>66.076</td><td>93.622</td><td>96.764</td><td>87.070</td><td>91.188</td><td>96.250</td><td>83.447</td><td>0.082399</td><td>4.4011</td><td>20.6855</td><td>0.613398</td><td>0.183115</td></tr><tr><td>UniWorld-View</td><td>54.237</td><td>67.534</td><td>93.444</td><td>96.627</td><td>88.946</td><td>91.777</td><td>97.500</td><td>84.295</td><td>0.068705</td><td>4.1958</td><td>22.4735</td><td>0.770554</td><td>0.121668</td></tr><tr><td>CameraAnything</td><td>54.365</td><td>67.526</td><td>94.489</td><td>96.908</td><td>87.399</td><td>91.817</td><td>88.750</td><td>83.036</td><td>0.186443</td><td>3.1868</td><td>15.1583</td><td>0.453712</td><td>0.342180</td></tr><tr><td>Ours</td><td>56.556</td><td>68.004</td><td>93.711</td><td>97.749</td><td>88.948</td><td>92.627</td><td>98.750</td><td>85.192</td><td>0.068622</td><td>2.8326</td><td>23.1511</td><td>0.787205</td><td>0.121664</td></tr></table>

![](images/18424fa4db2cd92dadfd1202518135bfdae605ebe082178d7887facf70f165c8.jpg)  
Fig. 3. Qualitative comparisons under diverse camera motion. All methods receive the same source video and target camera path. Representative frames illustrate subject and scene consistency, viewpoint control, and completion of newly exposed regions.

## 4.5. Ablation Studies

We examine how the choice and use of visual evidence contribute to controllable generation. Table 2 shows that the full method achieves the lowest camera errors and the best or tied-best results on all reported VBench dimensions. Accurate camera contro requires a clear spatial reference for the observed content. Figure 5 examines target-view warping and source-camera Plücker conditioning from this perspective. Removing warping causes the largest degradation across all reported metrics, with rotation and translation errors rising to approximately 3.4× and 10.9× their full-method values. Removing source-camera conditioning also increases camera errors. These results support retaining both the target-view layout and the source-view camera information. Figure 6 examines how evidence is used and what happens when current observations provide insufficient support. CGAR and EWA help preserve subject appearance and background structure. Rendered geometry and historical retrieval address two gaps in the available evidence: newly exposed subject surfaces and earlier scene states beyond the rolling cache.

## 5. Conclusion

We presented World in World (WiW), a training-free visual-evidence interface for extending the controllability of frozen causal video world models. WiW represents source observations, target-view projections, rendered geometry, and generated history as camera- and time-labelled clean visual states that the backbone reads through native self-attention. Correspondence-guided attention routing localizes relevant source-video evidence, while evidence-wise attention CFG regulates auxiliary contributions using responses from the same denoising forward pass, without additional denoising-network evaluations for guidance. On

(e) Human motion transfer and Re-cam  
![](images/9c5438a91055fd60d50a951ab762ac2f13d751fbea94fd896812fcbc38b298c4.jpg)  
Fig. 4. Applications of the same visual-evidence interface, including bullet-time rendering, video stabilization, video editing, K/V sharing between generated cases and human motion transfer. Model A and Model B denote two instances of the same frozen model running differen generation cases. The instances communicate by sharing cached K/V as visual evidence.

Table 2. Ablation results for camera-controlled video rerendering on DAVIS. Bold indicates the best results.
<table><tr><td rowspan="2">Variant</td><td colspan="5">VBench ↑</td><td colspan="2">Camera errors ↓</td></tr><tr><td>Subj.</td><td>Bg.</td><td>Smooth.</td><td>Aesth.</td><td>Flick.</td><td>RotError (°)</td><td>TransError</td></tr><tr><td>w/o EWA and CGAR</td><td>88.076</td><td>91.747</td><td>96.054</td><td>50.688</td><td>92.857</td><td>2.0346</td><td>0.056937</td></tr><tr><td>w/o EWA</td><td>88.056</td><td>91.706</td><td>96.055</td><td>50.720</td><td>92.867</td><td>1.9944</td><td>0.056608</td></tr><tr><td>w/o CGAR</td><td>88.055</td><td>91.728</td><td>96.072</td><td>50.713</td><td>92.891</td><td>1.7828</td><td>0.053913</td></tr><tr><td>w/o Target-View Warp</td><td>83.378</td><td>89.540</td><td>94.351</td><td>50.240</td><td>90.792</td><td>6.1158</td><td>0.581847</td></tr><tr><td>w/o Plücker</td><td>88.066</td><td>91.692</td><td>96.073</td><td>50.669</td><td>92.881</td><td>1.8811</td><td>0.056887</td></tr><tr><td>Full method</td><td>88.076</td><td>91.749</td><td>96.074</td><td>50.720</td><td>92.902</td><td>1.7823</td><td>0.053291</td></tr></table>

(b) Source-camera Plücker ablation

![](images/bbaae65977ea0977af7d5ba6e468f84210914f1e4f82e2419d3eee8102d361e2.jpg)

![](images/b26f6a4bde7f44cf68cfc2ff0f85df191faf619d03493cc2238d9d12abfdc518.jpg)

![](images/8a1636bb4e4af93b56657c150c063857af608ee06138f294a8b509f21d5e2b29.jpg)

![](images/bee892f855c8eb61816ba1c27c3bcef00a8a42b596bb5a670149b5d0f0eaf1ed.jpg)

![](images/a13fd3e247bf24905af1961baf3b0fc6742bbaa418c5261619e1bc9faa56dde9.jpg)

![](images/db58612dbfc76582f84fb97f0152a46928faad448924c5518018ff2e410f6e8a.jpg)

Fig. 5. Qualitative ablations of spatial evidence and camera conditioning. (a) Target-view warping helps preserve subject structure and scene layout under viewpoint changes. (b) Source-camera Plücker conditioning helps maintain cross-view structure by retaining source-view camera information.  
![](images/947700eef6677378f702be458f785331fd5b0c2260bdace65e6dc12e6bde5d84.jpg)

![](images/69715cd65e3c4380b1d47b4b4a284a6a7c33dc477ba7b97e9603a36c78f79762.jpg)

![](images/04295296386518bd87b0d8cf645afaee99ae93def849c3ad9b8739219d7740e7.jpg)

![](images/5787ccc4a7e1699a998df87d5c9f0d0b84d4c6ed6804a6d766a1767040a2df57.jpg)  
Fig. 6. Qualitative ablations of evidence use and availability. (a) Correspondence-guided attention routing (CGAR) and evidence-wise attention (EWA) help preserve subject appearance and background structure. (b) Rendered body geometry guides the completion of newly exposed subject regions. (c) The camera revisits a previously generated region after the earlier states have been removed from the rolling cache. Historical retrieval helps preserve the appearance and layout established during the earlier visit.

the reported DAVIS and OpenVid-1M evaluations, WiW achieves the highest average score across seven VBench dimensions and the lowest camera trajectory errors among the compared methods. Qualitative results further illustrate several downstream applications through the same interface. These results demonstrate that constructing, selecting, and regulating visual evidence can extend the capabilities of a pretrained world model, enabling exploration of the dynamic world depicted in a given video.

## References

[1] Lingdong Kong, Yu Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi Steven C. H. Hoi, and Ziwei Liu. 3D and 4D world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025. 2, 3

[2] Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2World: Crafting video diffusion models to interac tive world models. arXiv preprint arXiv:2505.14357, 2025. 3

[3] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[4] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026. 3, 7

[5] Jiacong Xu, Hanwen Jiang, Zhixin Shu, Kalyan Sunkavalli, Vishal M. Patel, and Yiqun Mei. Wonder: Video world model done better. arXiv preprint arXiv:2607.26037, 2026. 2, 4

[6] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, Sij Pan, Weihong Pan, Weijian Xie, Xianbin Liu, Xiaojun Xiang, Xiaoyu Zhang, Xinyu Chen, Yifu Wang, Yipeng Chen, Zhenzhou Fan, Zhewen Le, Zhichao Ye, and Ziqiang Zhao. INSPATIO-WORLD: A real-time 4D world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026. 2, 3, 7

[7] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025. 2

[8] AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, and Zihui Gao. AlayaWorld: Interactive long-horizon world modeling—full technical report. arXiv preprint arXiv:2607.18367, 2026. 2, 3

[9] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan GameCraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025. 2

[10] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, and Yahu Zhou. Matrix-Game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.

[11] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Size Wu, Wei Li, Xuchen Song, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-Game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[12] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[13] Runjia Qian, Zile Wang, Jihai Zhang, Kai Zou, Wei Yu, Jiaxing Li, Zexiang Liu, Yaokun Li, Fei Kang, Kaichen Huang, Mengyin An, Haobo Zhang, Biao Jiang, Jiahua Wang, Haofeng Sun, Yang Liu, and Yangguang Li. Matrix-Game 3.5: Enhancing real-time streaming interactive world models with patch memory. arXiv preprint arXiv:2608.29910, 2026. 2

[14] Haiyang Zhou, Wangbo Yu, Chaoran Feng, Xunyu Zhou, Yonghong Tian, and Li Yuan. UniWorld-View: Large-baseline view synthesis via video diffusion models. arXiv preprint arXiv:2608.04701, 2026. 2, 7

[15] Yixuan Li, Yanhong Zeng, Ka Leong Cheng, Jiayi Zhu, Hanlin Wang, Wen Wang, Yihao Meng, Hao Ouyang, Qiuyu Wang, Yue Yu, ZiDong Wang, Yiyuan Zhang, Yujun Shen, and Dahua Lin. CameraAnything: Refilming videos with arbitrary camera control. arXiv preprint arXiv:2607.24591, 2026.

[16] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[17] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. ReCamMaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF Internationa Conference on Computer Vision (ICCV), pages 14834–14844, 2025. 3, 7

[18] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WorldMem: Long-term consisten world simulation with memory. arXiv preprint arXiv:2504.12369, 2025. 2, 4

[19] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 3, 7

[20] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. OpenVid-1M: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024. 3, 7

[21] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T. Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, and Charles Herrmann. WonderJourney: Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6658–6667, 2024. 3

[22] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. WonderWorld: Interactive 3D scene generation from a single image. arXiv preprint arXiv:2406.09394, 2024.

[23] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N. Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3D scenes from a single image. arXiv preprint arXiv:2412.12091, 2024.

[24] Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, and Bisheng Yang. VistaDream: Sampling multiview consistent images for single-view scene reconstruction. arXiv preprint arXiv:2410.16892, 2024.

[25] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.

[26] Luxi Chen, Zihan Zhou, Min Zhao, Yikai Wang, Ge Zhang, Wenhao Huang, Hao Sun, Ji-Rong Wen, and Chongxuan Li. Flexworld: Progressively expanding 3D scenes for flexiable-view synthesis. arXiv preprint arXiv:2503.13265, 2025.

[27] Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Xin Tao, Pengfei Wan, Jie Zhou, and Jiwen Lu. Terra: Explorable native 3D world model with point latents. arXiv preprint arXiv:2510.14977, 2025.

[28] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. GEN3C: 3D-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[29] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra: Generative 3D scene reconstruction via video diffusion model self-distillation. arXiv preprint arXiv:2509.19296, 2025.

[30] Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, and Yiyi Liao. Gen3R: 3D scene generation meets feed-forward reconstruction. arXiv preprint arXiv:2601.04090, 2026.

[31] Sensen Gao, Zhaoqing Wang, Qihang Cao, Dongdong Yu, Changhu Wang, and Jia-Wang Bian. Pixworld: Unifying 3D scene generation and reconstruction in pixel space. arXiv preprint arXiv:2607.05373, 2026.

[32] Xinyang Li, Tengfei Wang, Zixiao Gu, Shengchuan Zhang, Chunchao Guo, and Liujuan Cao. FlashWorld: High-quality 3D scene generation within seconds. arXiv preprint arXiv:2510.13678, 2025.

[33] Team HY-World, Chenjie Cao, Xuhui Zuo, Zhenwei Wang, Yisu Zhang, Junta Wu, Zhenyang Liu, Yuning Gong, Yang Liu, Bo Yuan, Chao Zhang, Coopers Li, Dongyuan Guo, Fan Yang, Haiyu Zhang, Hang Cao, Jianchen Zhu, Jiaxin Lin, Jie Xiao, Jihong Zhang, Junlin Yu, Le Wang, Lifu Wang, Lilin Wang, Linus, Minghui Chen, Peng He, Penghao Zhao, Qi Chen, Rui Chen, Rui Shao, Sicong Liu, Wangchen Qin, Xiaochuan Niu, Xiang Yuan, Yi Sun, Yifei Tang, Yifu Sun, Yihang Lian, Yonghao Tan, Yuhong Liu, Yuyang Yin, Zhiyuan Min, Tengfe Wang, and Chunchao Guo. HY-World 2.0: A multi-modal world model for reconstructing, generating, and simulating 3D worlds. arXiv preprint arXiv:2604.14268, 2026. 3

[34] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. DimensionX: Create any 3D and 4D scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 3

[35] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, and Lijuan Wang. GenXD: Generating any 3D and 4D scenes. arXiv preprint arXiv:2411.02319, 2024.

[36] Hanyang Wang, Fangfu Liu, Jiawei Chi, and Yueqi Duan. VideoScene: Distilling video diffusion model to generate 3D scenes in one step. arXiv preprint arXiv:2504.01956, 2025.

[37] Hanxin Zhu, Cong Wang, Peiyan Tu, Jiayi Luo, Tianyu He, Xin Jin, and Zhibo Chen. GTA: Advancing image-to-3D world generation via geometry then appearance video diffusion. arXiv preprint arXiv:2605.12957, 2026.

[38] Lukas Höllein and Matthias Nießner. World reconstruction from inconsistent views. arXiv preprint arXiv:2603.16736, 2026. 3

[39] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4DNeX: Feed-forward 4D generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025. 3

[40] Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, and Yonggang Qi. Fantasyworld: Geometry-consistent world modeling via unified video and 3D prediction. arXiv preprint arXiv:2509.21657, 2025.

[41] Xunzhi Xiang, Zixuan Duan, Yabo Chen, Zhengxuan Wei, Guiyu Zhang, Zixiao Gu, Zhe Gao, Haibin Huang, Chi Zhang, Qi Fan, and Xuelong Li. VideoWeave: Unlocking geometric consistency in video generation via joint geometry-video modeling. arXiv preprin arXiv:2606.14162, 2026.

[42] Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. VerseCrafter: Dynamic realistic video world model with 4D geometric control. arXiv preprint arXiv:2601.05138, 2026.

[43] Avinash Paliwal, Adithya Iyer, Shivin Yadav, Muhammad Ali Afridi, and Midhun Harikumar. Reshoot-Anything: A self-supervised mode for in-the-wild video reshooting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2026.

[44] Youcan Xu, Jiaxin Shi, Zhen Wang, Wensong Song, Feifei Shao, Chen Liang, Jun Xiao, and Long Chen. RealCam: Real-time novel-view video generation with interactive camera control. arXiv preprint arXiv:2605.06051, 2026.

[45] Kuan Heng Lin, Zhizheng Liu, Pablo Salamanca, Yash Kant, Ryan Burgert, Yuancheng Xu, Koichi Namekata, Yiwei Zhao, Bolei Zhou, Micah Goldblum, Paul Debevec, and Ning Yu. Vista4D: Video reshooting with 4D point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[46] Tao Hu, Haoyang Peng, Xiao Liu, and Yuewen Ma. EX-4D: Extreme viewpoint 4D video synthesis via depth watertight mesh. arXiv preprint arXiv:2506.05554, 2025.

[47] Dongyue Lu, Ao Liang, Tianxin Huang, Xiao Fu, Yuyang Zhao, Baorui Ma, Liang Pan, Wei Yin, Lingdong Kong, Wei Tsang Ooi, and Ziwei Liu. See4D: Pose-free 4D generation via auto-regressive video inpainting. arXiv preprint arXiv:2510.26796, 2025. Accepted to Eurographics 2026. 3

[48] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 3

[49] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[50] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.

[51] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, and Tong He. Deepverse: 4D autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025. 3

[52] Yifan Wang and Tong He. Warp-as-History: Generalizable camera-controlled video generation from one training video. arXiv preprint arXiv:2605.15182, 2026. 3

[53] Wonbong Jang, Shikun Liu, Soubhik Sanyal, Juan Camilo Perez, Kam Woh Ng, Sanskar Agrawal, Juan-Manuel Perez-Rua, Yiannis Douratsos, and Tao Xiang. Rays as pixels: Learning a joint distribution of videos and camera trajectories. In International Conference on Machine Learning (ICML), 2026. 3

[54] Chunyang Li, Yuanbo Yang, Jiahao Shao, Hongyu Zhou, Katja Schwarz, and Yiyi Liao. ReRoPE: Repurposing RoPE for relative camera control. arXiv preprint arXiv:2602.08068, 2026.

[55] Chendong Xiang, Jiajun Liu, Jintao Zhang, Xiao Yang, Zhengwei Fang, Shizun Wang, Zijun Wang, Yingtian Zou, Hang Su, and Jun Zhu. Geometry-aware rotary position embedding for consistent video world model. arXiv preprint arXiv:2602.07854, 2026.

[56] JoungBin Lee, Jaewoo Jung, Jisang Han, Takuya Narihira, Kazumi Fukuda, Junyoung Seo, Sunghwan Hong, Yuki Mitsufuji, and Seun gryong Kim. 3D scene prompting for scene-consistent camera-controllable video generation. arXiv preprint arXiv:2510.14945, 2025.

[57] Xuanyi Liu, Deyi Ji, Liqun Liu, Lanyun Zhu, Xuhang Chen, Qianxiong Xu, Peng Shu, Huan Yu, Jie Jiang, Feng Gao, and Siwei Ma. CamGeo: Sparse camera-conditioned image-to-video generation with 3D geometry priors. In Proceedings ofthe International Conference on Machine Learning, 2026.

[58] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3D representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025.

[59] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3D-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025.

[60] Geonung Kim, Janghyeok Han, and Sunghyun Cho. VideoFrom3D: 3D scene video generation via complementary image and video diffusion models. arXiv preprint arXiv:2509.17985, 2025. 3

[61] Yipeng Chen, Zhichao Ye, Zhenzhou Fang, Xinyu Chen, Xiaoyu Zhang, Jialing Liu, Nan Wang, Guofeng Zhang, and Haomin Liu. Postcam: Camera-controllable novel-view video generation with query-shared cross-attention. arXiv preprint arXiv:2511.17185, 2025. 3

[62] Junyoung Seo, Jisang Han, Jaewoo Jung, Siyoon Jin, Joungbin Lee, Takuya Narihira, Kazumi Fukuda, Takashi Shibuya, Donghoon Ahn Shoukang Hu, Seungryong Kim, and Yuki Mitsufuji. Vid-CamEdit: Video camera trajectory editing with generative rendering from estimated geometry. arXiv preprint arXiv:2506.13697, 2025.

[63] Haoyuan Wang, Yabo Chen, Haibin Huang, Chi Zhang, and Xuelong Li. Directing the world: Fast autoregressive video generation with compositional human-camera control. arXiv preprint arXiv:2606.27964, 2026. 3

[64] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. VMem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025. 4

[65] Jiahao Wang, Luoxin Ye, TaiMing Lu, Junfei Xiao, Jiahan Zhang, Yuxiang Guo, Xijun Liu, Rama Chellappa, Cheng Peng, Alan Yuille, and Jieneng Chen. EvoWorld: Evolving panoramic world generation with explicit 3D memory. arXiv preprint arXiv:2510.01183, 2025.

[66] Zhengxuan Wei, Xu Guo, Xinghui Li, Xunzhi Xiang, Min Wei, Yiran Zhu, Qiulin Wang, Xintao Wang, Pengfei Wan, Xiangwang Hou, and Qi Fan. Geometry-aware implicit memory for video world models. arXiv preprint arXiv:2606.02436, 2026.

[67] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-tempora memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025

[68] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.

[69] Minseok Joo, Dogyun Park, Taehoon Lee, Kyujin Lee, and Hyunwoo J. Kim. Retrieve what’s missing: Coverage-maximizing retrieval for consistent long video generation. arXiv preprint arXiv:2606.02479, 2026.

[70] Jung Yi, Minjae Kim, Paul Hyunbin Cho, Wooseok Jang, Sangdoo Yun, and Seungryong Kim. WorldKV: Efficient world memory with world retrieval and compression. arXiv preprint arXiv:2605.22718, 2026. 4

[71] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025. 5

[72] Lingteng Qiu, Peihao Li, Heyuan Li, Qi Zuo, Xiaodong Gu, Yuan Dong, Weihao Yuan, Rui Peng, Siyu Zhu, Xiaoguang Han, Guanying Chen, and Zilong Dong. LHM++: An efficient large human reconstruction model for pose-free images to 3D. arXiv preprin arXiv:2506.13766, 2025. 6

[73] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10975–10985, 2019. 6

[74] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6

[75] Dar-Yen Chen, Hmrishav Bandyopadhyay, Kai Zou, and Yi-Zhe Song. Normalized attention guidance: Universal negative guidance for diffusion models. Advances in Neural Information Processing Systems, 38:137412–137446, 2026. 6

[76] Chenxi Song, Yanming Yang, Tong Zhao, Ruibo Li, and Chi Zhang. Taming video models for 3D and 4D generation via zero-shot camera control. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 40352–40363, 2026. 7

[77] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023. 7

[78] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 7

[79] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixé, and Sanja Fidler. ViPE: Video pose engine for 3D geometric perception. arXiv preprint arXiv:2508.10934, 2025. 7