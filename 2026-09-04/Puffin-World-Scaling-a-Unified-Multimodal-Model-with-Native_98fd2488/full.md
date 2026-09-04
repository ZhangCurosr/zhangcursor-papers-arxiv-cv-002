# Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States

Kang Liao<sup>1</sup> Yihang Luo<sup>1</sup> Xiao-Ming Wu<sup>1</sup> Linyi Jin<sup>2</sup> Size Wu<sup>1</sup>   
Chunyu Lin<sup>3</sup> Yao Zhao<sup>3</sup> Fei Wang<sup>4</sup> Wei Li<sup>1</sup> Chen Change Loy<sup>1,4</sup> <sup>1</sup>S-Lab, Nanyang Technological University <sup>2</sup>University of Michigan <sup>3</sup>Beijing Jiaotong University <sup>4</sup>ACE Robotics

https://kangliao929.github.io/projects/puffin-world/

![](images/0ce29bcd7cf4633129b0123b044ccb787c671b32c74ea2022cc42b068ed34e0e.jpg)  
Figure 1. Puffin-World is a unified multimodal model using native 3D world states (Appearance-Geometry-Physics) for spatia intelligence and physical AI. It unifies camera physics understanding, free-viewpoint spatial simulation, and 3D world generation and reconstruction within a single framework.

## Abstract

We propose Puffin-World, a unified multimodal architecture that integrates physical understanding, spatial simulation, and 3D world generation and reconstruction without relying on external offline modules. To reliably construct and interact with 3D worlds, ourframeworkjointly models three native world states: physics (gravity field and latitude), geometry (depth), and appearance (image), together with a unified Omni-Camera representation that supports diverse tasks and flexible motions. Beyond modeling these states, we introduce a strategyfor propagating physical dynamics acrossfuture frames. By grounding absolute camera properties in the real world, Puffin-World enables physically consistent and visually stable world generation. We further couple appearance and geometry within a single generative process, jointly synthesizing eachfuture view and reconstructing its underlying geometry. This unified paradigm enables interleaved closed-loop applications requiring synergy across multiple

tasks, including mimic and self-calibrated world exploration. To scale Puffin-World to complex scenarios, we construct Puffin-16M, comprising 15 million vision–language–camera triplets and 1 million trajectoriesfeaturing various and challenging motions. Tofosterfurther research in this area, we released the code, models, and datasets.

## 1. Introduction

Building a model that can perceive, generate, and reconstruct the world from arbitrary visual observations is a central goal of multimodal spatial intelligence and physical AI. A capable world model should not merely generate plausible pixels; it should understand where the camera or viewpoint is in the real world, reason about the geometry of the scene, and synthesize what the world looks like from new viewpoints in a way that stays consistent with physical law. The physical world therefore cannot be adequately represented as a col-

![](images/9925c989ebb0c4b3d24927eab8e358ea506120c5c3b88c61a61ad65eae8dae89.jpg)  
Native World States: Appearance-Geometry-Physics

Figure 2. Illustration of the versatile capabilities of our model. It shows the best performance (AUC ▷ 5<sup>◦</sup>) on camera-to-world understanding for physical world perception; flexible camera-controllable abilities for free-viewpoint spatial simulation; 3D world generation and reconstruction with flexible-DoF actions and native world states. The red box marks the input to the model.

lection of 2D images. Instead, effective interaction with the real world calls for a unified framework that jointly models its multiple interconnected states.

Progress so far has advanced along separate fronts that each capture only part of this picture. Generative world models [1, 8, 27, 34, 40, 103, 136] have made striking progress, but they predict the world almost exclusively at the appearance level, treating each frame as RGB content with no explicit notion of the camera’s physical orientation or the underlying scene geometry. In parallel, unified multimodal models [18, 92, 102, 105, 131] couple understanding and generation within a single network, but do so for only 2D semantics. As a result, the field still lacks a single model that unifies holistic modalities and tasks for 3D world modeling.

Unifying these capabilities is fundamentally non-trivial. A simple combination of existing perception, generation, and reconstruction components is insufficient because a unified world model faces three tightly coupled challenges: (i) establishing a unified action representation that supports both continuous single- and cross-view control while remaining grounded in absolute physical concepts such as gravity, uprightness, and orientation; (ii) maintaining a physically persistent frame that allows knowledge inferred from observed views to consistently guide 3D world modeling at unseen viewpoints; and (iii) scaling these capabilities with data that provides both absolute camera grounding and diverse, challenging motion. Existing relative camera representations, such as Plucker embeddings [¨ 27, 34, 86], offer effective control but lack a global physical anchor, while prevailing 3D datasets [19, 67, 133] contain limited rotational diversity and rarely provide absolute camera orientation.

In this work, we present Puffin-World, a unified multimodal model that scales 3D world perception, generation, and reconstruction as shown in Figure 1, without relying on any external offline modules. In particular, we formulate three native world states to jointly model the 3D world in an end-to-end manner: the physics that grounds an observation in the absolute physical world, the geometry that describes underlying 3D spatial structure, and the appearance that we finally see from the visual content manifested in images and sequences. To address the aforementioned challenges, we first introduce the Omni-Camera representation, a dense action signal that combines a gravity-anchored absolute field with a relative ray field, which enables physicsgrounded text-to-image spatial simulation (single-view) and image-to-3D scene generation (cross-view). Subsequently, we propose physics propagation: by anchoring the absolute spatial knowledge derived from world perception and propagating it across the relative control signals of future frames, Puffin-World generates appearance that remains stable and gravity-consistent over complex motions. Furthermore, we scale the whole framework with the constructed Puffin-16M, a new dataset of 15M vision–language–camera triplets and

1M trajectories featuring diverse and challenging camera motions with precise labels.

Thanks to this unified paradigm, Puffin-World supports diverse multimodal tasks as illustrated in Fig. 2. For physical world perception, it attains state-of-the-art camerato-world understanding, achieving the best median error and the best AUC at 5<sup>◦</sup> across four public benchmarks. For free-viewpoint spatial simulation, it generates cameracontrollable images with more faithful distribution than strong general-purpose generators [29, 48, 73, 128]. For 3D world modeling, it performs high-DoF action-conditioned, text/image-to-3D generation while jointly reconstructing perview geometry, enabling consistent and physically grounded world modeling. Furthermore, Puffin-World enables complicated, closed-loop applications that require multi-task synergy, such as mimic and self-calibrated world exploration. We summarize our contributions as follows:

• We propose Puffin-World, a unified multimodal model that formulates native 3D world states and jointly perceives, generates, and reconstructs the 3D world within a single framework.

• We introduce the Omni-Camera representation, a unified dense camera condition that integrates gravity-aware absolute orientation with ray-based relative geometry. A physics propagation mechanism is further proposed to anchor absolute spatial knowledge from perception and propagate it across future control signals.

• We construct Puffin-16M, comprising Puffin-Cam-15M (15M vision–language–camera triplets) and Puffin-Traj-1M (1M challenging-motion trajectories). We release it at https://kangliao929.github.io/ projects/puffin-16m.

• To promote the development of the research community, we have fully open-sourced our code, models, and datasets. Moreover, leveraging Puffin-World’s accurate understanding of absolute camera physics, we annotated 28 widely used public datasets. These annotations cover approximately 44.5 million images across diverse data distributions.

## 2. Related Work

Camera-to-World Understanding. Recovering physical camera parameters from images, including camera calibration and pose estimation, is a long-standing problem in 3D vision [33, 44, 59, 66, 76, 96]. Early learning-based methods directly regress camera parameters from a single image [11, 37, 45, 104, 122], whereas more recent approaches narrow the prediction gap by incorporating intermediate geometric structures or semantic cues [41, 50, 51, 88, 118]. A particularly successful direction learns dense, pixel-wise geometric representations, including distortion maps [57, 58], pixel displacement fields [55, 60, 112], camera rays [124], perspective fields [44, 94, 96], and incidence fields [23, 35, 135]. These representations provide spatially grounded supervision and generally offer greater robustness than direct global regression. More recently, Puffin [61] reframes this task as a language-modeling problem by representing the camera as text and predicting its parameters through spatial reasoning. However, its limited model scale and training data constrain its performance and scalability, causing it to remain inferior to specialized vision-based methods such as GeoCalib [96] on several benchmarks [83].

Unified Multimodal Models. Building upon conventional large multimodal models, unified multimodal models [2, 18, 21, 31, 63, 87, 90–92, 95, 102, 108, 109, 126, 131] integrate visual understanding and generation within a single network, either through autoregressive modeling over discrete [92, 102, 105, 109] or continuous [25, 77] visual tokens, or by connecting pretrained multimodal models with diffusion decoders [17, 39, 62, 71, 74, 107, 110, 111, 123]. Despite strong progress in general image understanding and generation, these models largely treat images as 2D appearance and semantics under simplified camera assumptions, without explicitly modeling camera physics, scene geometry, or other essential states. Puffin [61] takes an initial step toward camera-centric unification, but still focuses on isolated-view perception and lacks a persistent world frame across views. This motivates extending multimodal unification from 2D semantics to 3D world states.

3D World Models. Generative world models [52] aim to simulate the world by predicting future observations, with recent video- and multi-view diffusion frameworks achieving impressive visual fidelity [1, 5, 8, 9, 20, 30, 32, 40, 106, 121, 127, 129, 136]. To enable controllable simulation, extensive research conditions generation on camera information: dense camera-pose or Plucker-ray embeddings guide camera-¨ controlled video generation [6, 34, 103, 113, 114], while multi-view diffusion models synthesize novel views or complete scenes from one or a few reference images [10, 14, 27, 80], sometimes coupled with feed-forward 3D reconstruction models to recover geometry [49, 97, 99, 100]. Despite this progress, existing 3D world models exhibit three major limitations. First, they predominantly model the world at the appearance level, leaving geometry and physical grounding implicit. Second, they rely primarily on relative camera motion, which lacks a global physical reference frame. Consequently, the same relative trajectory may correspond to different absolute world orientations, leading to orientation drift and degraded performance under challenging motions, long horizons, and single-view settings. Third, generation is typically treated as an isolated task, decoupled from physical camera understanding and geometry reconstruction, and is often trained on data with limited rotational diversity. These limitations motivate a unified world model that jointly represents physics, geometry, and appearance, anchors generation to an absolute physical frame, and integrates perception, simulation, and reconstruction within a single framework.

## 3. Method

## 3.1. Preliminary

## 3.1.1. 3D Native World States

Most existing generative world models focus primarily on appearance-level prediction, where future states are represented as RGB images or videos. However, the physical world is natively organized by multiple levels of states, including physics, geometry, and appearance. Physics-level states provide global physical grounding, geometry-level states describe the 3D spatial structure, and appearance-level states represent the observable visual content. Jointly modeling these complementary states is crucial for stable generation, spatially consistent simulation, and grounded world interaction.

In Puffin-World, we formulate a hierarchy of native 3D world states. Specifically, the physics state captures absolute physical cues, including the gravity field and latitude map; the geometry state represents scene depth; and the appearance state corresponds to RGB observations. Since some states, particularly physics-level cues, are difficult to infer directly from visual observations, and their interactions across different levels remain underexplored, Puffin-World jointly perceives, propagates, and generates these states within a unified framework. This design allows physics-aware signals to guide future-view generation toward geometrically consistent structure and realistic appearance.

## 3.1.2. Unifying Camera Representations

The camera serves as a fundamental interface between the 3D world and 2D visual data modalities. In visual perception and reconstruction, camera models explicitly encode the physica rules of perspective projection, thereby supporting tasks such as single-view camera calibration, multi-view geometric reasoning, and joint spatial reconstruction. Beyond perception, cameras also provide effective and practical control signals for generative models, enabling spatially controllable synthesis, continuous world generation, and consistent physical simulation.

Existing camera representations can be broadly categorized as relative or absolute. Relative camera representations capture spatial relationships across viewpoints and are widely used for multi-view reconstruction and continuous scene generation. Although readily obtained from structurefrom-motion and effective for cross-view geometry, they lack a global physical anchor and cannot encode absolute orientation with respect to the real world. Absolute camera representations, in contrast, ground individual observations to physical cues such as the horizon and scene uprightness, but are less suited to continuous multi-view transitions and are substantially harder to obtain from monocular images. These complementary properties motivate a unified camera representation that combines absolute physical grounding with continuous spatial modeling. To this end, we propose the Omni-Camera representation, a simple but effective unified camera representation for versatile applications. For each pixel $\mathbf { x } = ( u , v )$ , Omni-Camera representation combines an absolute camera field and a relative ray field along the channel dimension:

![](images/a91a52046773ff01ddb87af5d358ed65b553d382c3f320eb638bb76ad406d7f9.jpg)  
Figure 3. The network architecture of our Puffin-World. It mainly comprises 3D world understanding, generation, and reconstruction, formulating native 3D world states within one framework. Both Omni-Camera representation and role mask are conditioned on the reference view and the target views. For clarity, we omit the camera-controllable text-to-image generation and the mathematical mapping from the understanding results to the physics world state.

$$
\begin{array} { r l } & { \mathbf { c _ { x } } = \operatorname { C o n c a t } \left( \mathbf { a _ { x } } , \mathbf { p _ { x } } \right) \in \mathbb { R } ^ { 9 } , } \\ & { \mathbf { a _ { x } } = \left( \mathbf { u _ { x } } , \varphi _ { \mathbf { x } } \right) \in \mathbb { R } ^ { 3 } , \quad \mathbf { p _ { x } } \in \mathbb { R } ^ { 6 } . } \end{array}\tag{1}
$$

Here, $\mathbf { a } _ { \mathbf { x } }$ denotes the absolute camera representation consisting of the pixel-wise up-vector $\mathbf { u } _ { \mathbf { x } }$ and latitude angle $\varphi _ { \mathbf { x } } .$ while $\mathbf { p _ { x } }$ denotes the relative camera representation given by the ray map composed of the ray origin and direction. Let $\tilde { \mathbf { x } } = [ u , v , 1 ] ^ { \top }$ be the homogeneous coordinate of pixel x. Given the camera intrinsic matrix K and the world-to-camera extrinsic parameters (R, t), where $\mathbf { X } _ { c } = \mathbf { R } \mathbf { X } _ { w } + \mathbf { t }$ , the camera center in the world coordinate system is $\mathbf { o } = - \mathbf { R } ^ { \top } \mathbf { t }$ . The viewing ray direction from the camera center to pixel x is

$$
\mathbf { d } _ { \mathbf { x } } = \frac { \mathbf { R } ^ { \top } \mathbf { K } ^ { - 1 } \tilde { \mathbf { x } } } { \Vert \mathbf { R } ^ { \top } \mathbf { K } ^ { - 1 } \tilde { \mathbf { x } } \Vert _ { 2 } } .\tag{2}
$$

The relative ray representation [27] of this ray is then formed by concatenating the ray origin and the ray direction,

$$
\begin{array} { r } { { \bf p } _ { \bf x } = ( { \bf o } , { \bf d } _ { \bf x } ) \in \mathbb { R } ^ { 6 } , } \end{array}\tag{3}
$$

where $\mathbf { o } = - \mathbf { R } ^ { \top } \mathbf { t }$ is the ray origin shared by all pixels of the view and $\mathbf { d } _ { \mathbf { x } }$ is the per-pixel ray direction. Relative camera motion between two views is thereby encoded by the change of the ray origin (translation) and the ray direction (rotation).

For the absolute component, we follow the Perspective Field [44] to represent the physical orientation of each pixel. Let $\mathbf { g } \in \mathbb { R } ^ { 3 }$ be the unit gravity direction and $\Pi ( \cdot )$ be the projection function. For any 3D point X on the ray of pixel $\mathbf { x } ,$ satisfying $\Pi ( { \mathbf { X } } ) = { \mathbf { x } }$ , the up-vector and latitude angle are defined as

$$
\begin{array} { r l } & { \mathbf { u } _ { \mathbf { x } } = \underset { \epsilon  0 } { \operatorname* { l i m } } \frac { \Pi ( \mathbf { X } - \epsilon \mathbf { g } ) - \Pi ( \mathbf { X } ) } { \| \Pi ( \mathbf { X } - \epsilon \mathbf { g } ) - \Pi ( \mathbf { X } ) \| _ { 2 } } , } \\ & { \varphi _ { \mathbf { x } } = \arcsin ( \frac { ( - \mathbf { d } _ { \mathbf { x } } ) ^ { \top } \mathbf { g } } { \| \mathbf { d } _ { \mathbf { x } } \| _ { 2 } } ) . } \end{array}\tag{4}
$$

Here, $- \mathbf { d } _ { \mathbf { x } }$ corresponds to the incoming light ray direction. The up-vector $\mathbf { u } _ { \mathbf { x } }$ describes the image-plane projection of the direction opposite to gravity, and the latitude angle $\varphi _ { \mathbf { x } }$ measures the elevation of the incoming ray with respect to the horizontal plane.

By integrating gravity-aware absolute orientation with ray-based relative geometry, the Omni-Camera representation provides both global physical grounding and continuous spatial modeling. Although conceptually simple, it effectively serves as a unified and flexible camera condition for physical-world understanding, camera-controllable image generation, and continuous world generation, supporting both single-view and cross-view world modeling within a common representation.

## 3.2. Puffin-World

Puffin-World unifies physical-world perception, freeviewpoint spatial simulation, and 3D world generation and reconstruction within a single multimodal framework. Its versatility arises from three levels of unification rather than task-specific subnetworks. (i) Representation unification. Appearance and geometry share a common latent space, with RGB images and depth maps encoded by the same VAE, while all camera configurations and motions, ranging from in-place rotation to large translation, are represented by the unified Omni-Camera representation C defined in Eq. 1. A compact four-channel role mask m $\in \mathbb { R } ^ { 4 \times H \times W }$ is paired with C to indicate whether each view serves as a generation target, a conditioning reference, an image-conditioned input, or a geometry view, and both are processed by the same condition fusion module. (ii) Modality unification. A single backbone both perceives and generates the world: the geometry-aligned vision encoder and LLM yield autoregressive understanding outputs, while the same LLM hidden states are transformed by learnable queries and a lightweight connector into conditioning signals for the diffusion generator, enabling perception and generation as two complementary outputs of one model. (iii) Task unification. In Puffin-World, each task is determined by the available inputs, including text, target cameras, reference views, together with the role mask m. By varying this input composition, the same framework supports camera-to-world understanding, camera-controllable text-to-image generation, 3D world generation, and joint appearance-geometry reconstruction. The overview of Puffin-World’s framework is illustrated in Figure 3. We detail these capabilities below.

## 3.2.1. Physics Perception

Given a single image, Puffin-World estimates the camera’s absolute physical state, including its gravity-relative orientation specified by roll and pitch, its intrinsic vertical fieldof-view (FoV), and a semantic scene description. Following Puffin [61], we formulate this task as autoregressive multimodal sequence modeling rather than direct regression. The geometry-aligned vision encoder [61] extracts visual features from images, which are projected into the LLM embedding space through an MLP projector. The LLM then generates a structured scene and spatial analysis, followed by the numerical camera parameters, and is trained using a next-token cross-entropy loss.

This formulation offers two key advantages for physicalstate perception. First, it leverages the LLM’s sequence modeling capability and favorable scaling behavior, recasting fine-grained camera understanding as language modeling. Second, because the camera parameters are predicted after reasoning about scene-level cues such as the horizon, vertical structures, and foreground composition, the estimation relies on holistic scene understanding rather than low-level visual cues alone. This design enables robust gravity-aware perception across diverse scenes.

## 3.2.2. Free-Viewpoint Spatial Simulation

Puffin-World enables camera-controllable image generation: given a text prompt and a desired Omni-Camera map C, it synthesizes an image whose realized viewpoint and intrinsics strictly adhere to C. Because this part focuses on single-view generation, the ray map within the Omni-Camera representation is held constant. The resulting query hidden states pass through a lightweight connector, yielding a pooled conditioning vector and a sequence of joint-attention conditioning embeddings for the multimodal diffusion transformer (MMDiT) [24]. The MMDiT then denoises a target latent under a flow-matching objective, after which a VAE decoder maps the denoised latent back into pixel space.

Our approach to integrating camera condition is the primary departure from Puffin [61]. Puffin encodes a 3-channel Perspective Field into a continuous latent via a VAE and injects it into the image latent via cross-attention, limiting its extension to broader tasks and camera motions. Moreover, because a holistic camera representation cannot be formulated with only 3 channels, it cannot be straightforwardly encoded and injected in this manner. Instead, we fuse the Omni-Camera representation directly within the diffusion latent space. A lightweight condition fusion module $\mathcal { F }$ maps C and the role mask m into the input latent space of MMDiT. This output is added to the noisy image latent z prior to the patch embedding operation $\mathcal { P } \mathrm { : }$

$$
\begin{array} { r } { \mathbf { h } ^ { 0 } = \mathcal { P } \big ( \mathbf { z } + \mathcal { F } ( [ \mathbf { C } ; \mathbf { m } ] ) \big ) . } \end{array}\tag{5}
$$

To preserve the pretrained generator’s behavior at the start of training, F is initialized such that its contribution starts near zero. Furthermore, to ensure the camera signal remains effective across deeper layers, we re-inject the patchified camera features at a sparse set of transformer blocks $\mathcal { L } \mathrm { : ~ }$

$$
\mathbf { h } ^ { l }  \mathbf { h } ^ { l } + \mathbf { W } _ { l } \mathcal { P } \big ( \mathcal { F } ( [ \mathbf { C } ; \mathbf { m } ] ) \big ) , \quad l \in \mathcal { L } .\tag{6}
$$

Compared to using a channel-limited mechanism over a separately VAE-encoded camera branch, this pixel-aligned additive injection supports versatile tasks. It grounds every latent token in its precise camera geometry across multiple layers, providing superior and stable spatial control while introducing negligible parameter overhead.

## 3.2.3. 3D World Modeling

Puffin-World then extends the single-view generator to multiview trajectories while jointly supporting geometry reconstruction. Instead of denoising a single target image, the model processes a set of T views within a unified jointattention sequence. Among them, K reference views are provided as clean latents $\{ { \bf z } _ { v } ^ { \mathrm { r e f } } \}$ and are also encoded by the geometry-aligned vision encoder and the LLM to provide semantic conditioning. The remaining $T - K$ target views, represented by $\{ \mathbf { z } _ { v } ^ { \mathrm { t g t } } \}$ , are corrupted with noise and jointly denoised under the flow-matching objective. Each view, whether reference or target, is associated with its own Omni-Camera condition, which is fused into the corresponding image latent through the additive injection mechanism in Eq. 5 and Eq. 6. To explicitly encode view identity within the shared attention sequence, we assign each token a view-axis index and apply a one-dimensional rotary positional embedding along this axis. This allows the relative organization of views to be modeled directly within a single attention operation, enabling cross-view consistency to emerge without requiring an explicit inter-view consistency loss.

Trajectory Anchoring via Physics Propagation. A trajectory introduces a challenge specific to absolute grounding: relative motion across views can be readily obtained from camera poses, whereas the absolute orientation that anchors the trajectory to the real world is generally unavailable. The Omni-Camera representation addresses this through its complementary components: the relative ray map captures inter-view motion, while the absolute Perspective Field encodes gravity-aligned orientation. We recover the latter for future frames through physics propagation. Specifically, the model first perceives the absolute state $( \phi _ { 0 } , \theta _ { 0 } , \alpha _ { 0 } , \ldots )$ of the reference view through the Physics Perception pathway (Sec. 3.2.1), where $\phi _ { 0 } , \theta _ { 0 }$ , and $\alpha _ { 0 }$ denote its roll, pitch, and vertical FoV, respectively, yielding the gravity direction ${ \bf g } _ { 0 }$ in the reference frame. Given the relative rotation $\mathbf { R } _ { t  0 } ^ { \mathrm { r e l } }$ that transforms vectors from the reference camera coordinate frame to the camera coordinate frame of view $t ,$ the gravity direction in each view is propagated as:

$$
\mathcal { G } = \{ \mathbf { g } _ { t } \} _ { t = 0 } ^ { T - 1 } = \{ \mathbf { R } _ { t  0 } ^ { \mathrm { r e l } } \mathbf { g } _ { 0 } \} _ { t = 0 } ^ { T - 1 } .\tag{7}
$$

The propagated gravity direction of each view is then used to render its corresponding absolute camera representation following Eq. 4, which is concatenated with the relative ray map to form the complete Omni-Camera condition. In this way, all views along the trajectory share a coherent gravity-anchored world frame while retaining their individual relative motions, enabling physically and spatially stable generation under diverse and challenging camera trajectories.

Joint Appearance–Geometry Modeling. Puffin-World predicts not only the appearance but also the geometry of each generated view, casting reconstruction as part of the same generative process. In particular, we encode depth as an RGB image using an invertible color mapping inspired by the 3D Hilbert curve in Vision Banana [26], and encode it with the same frozen VAE used for appearance. Instead of directly regressing unbounded depth or disparity, this gives us a bounded three-channel representation that fits the pretrained VAE without changing its input or output interface.

The mapping is deterministic and invertible, so the generated result can be decoded back to depth. Its nonlinear transform also allocates more of the color range to nearby surfaces, where geometric accuracy matters most. This allows appearance and geometry to use the same latent representation without requiring a dedicated depth encoder or decoder. For each modeled view, the depth latent is appended as an additional token block that shares the Omni-Camera condition and view-axis index of its RGB counterpart, distinguished only by the geometry channel of the role mask m. A single denoising pass therefore jointly yields the appearance and geometry latents of all target views under the shared flow-matching objective:

$$
\mathcal { L } _ { \mathrm { f m } } ( \mathcal { V } ) = \mathbb { E } _ { t , \epsilon } \Bigg [ \frac { 1 } { | \mathcal { V } | } \sum _ { v \in \mathcal { V } } \left\| \mathbf { v } _ { \theta } ( \mathbf { z } _ { t } ^ { v } , t , \mathbf { c } ^ { v } ) - ( \epsilon ^ { v } - \mathbf { z } _ { 0 } ^ { v } ) \right\| _ { 2 } ^ { 2 } \Bigg ] ,\tag{8}
$$

where $\mathbf { z } _ { t } ^ { v } = ( 1 - \sigma _ { t } ) \mathbf { z } _ { 0 } ^ { v } + \sigma _ { t } \epsilon ^ { v } , \mathbf { c } ^ { v }$ collects the per-view camera and connector conditioning, and V is a set of target views. Because the saturated colors of the geometry latents could otherwise corrupt the converged appearance model, we do not simply pool appearance and geometry targets into one objective. Instead, we evaluate Eq. 8 separately over the appearance target views $\mathcal { V } _ { \mathrm { r g b } }$ and the geometry target views $\nu _ { \mathrm { g e o } }$ , and combine them with a time-dependent weight:

$$
\begin{array} { r } { \begin{array} { c } { \mathcal { L } = \mathcal { L } _ { \mathrm { f m } } ( \mathcal { V } _ { \mathrm { r g b } } ) + \omega ( \tau ) \mathcal { L } _ { \mathrm { f m } } ( \mathcal { V } _ { \mathrm { g e o } } ) , } \\ { \omega ( \tau ) = \omega _ { \mathrm { m a x } } \operatorname* { m i n } \Bigl ( 1 , \frac { \tau } { \tau _ { 0 } } \Bigr ) , } \end{array} } \end{array}\tag{9}
$$

where $\tau$ is the training iteration and $\omega ( \tau )$ ramps linearly from 0 to $\omega _ { \mathrm { m a x } }$ over the first $\tau _ { 0 }$ iterations. Normalizing each modality by its own view count keeps the appearance gradient magnitude stable, while the ramp starts the geometry weight at zero so that early depth gradients do not perturb the shared modulation and output layers.

To activate the geometry branch on a converged appearance model without degrading visual quality, we introduce two safeguards that preserve the appearance pathway. First, we adopt asymmetric attention as shown in Figure 3: appearance and text queries cannot attend to geometry tokens, while geometry tokens attend to all modalities. This prevents geometry features from leaking into the appearance stream while retaining full scene context for geometry prediction. Second, we add a zero-initialized learnable geometry-modality embedding after patch embedding. This keeps the model functionally identical to the original multi-view generator at initialization and gradually introduces geometry-specific features during training. At inference, appearance latents are decoded into images, whereas geometry latents are decoded and inverse-mapped from Hilbert color space to scalar depth. Long-horizon Exploration. The joint-attention sequence is trained with a fixed number of views, yet real exploration requires trajectories of arbitrary length. We therefore extend generation autoregressively in chunks: the model first generates a chunk of views, and the last generated view is then carried over as the reference view for the next chunk under a sliding context window, so that a trajectory of any length is produced by repeatedly advancing this window. The key design choice is that consecutive chunks are linked in the latent space: the next chunk conditions directly on the denoised target latent of the carried-over view rather than on a re-encoded pixel image, which avoids the compounding VAE encode–decode artifacts that would otherwise accumulate at every chunk boundary and drift over a long horizon. Physics propagation is applied across the whole sequence rather than within a single chunk, so all chunks share one gravity-aligned absolute frame, and the extended trajectory stays self-calibrated instead of accumulating orientation drift.

Table 1. Training recipe of Puffin-World. We report the per-stage hyperparameters, the trainable/frozen status of each module, and the data sampling ratio across tasks. Und., Gen., and Recon. abbreviate understanding, generation, and reconstruction, respectively.
<table><tr><td></td><td>Stage I</td><td>Stage II</td><td>Stage III</td><td>Stage IV</td></tr><tr><td colspan="5">Hyperparameters</td></tr><tr><td>Learning rate</td><td>1 × 10−4</td><td>2 × 10−5</td><td>5 × 10−5</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR Scheduler</td><td></td><td>Cosine 0.05</td><td></td><td></td></tr><tr><td>Weight Decay</td><td colspan="4">(0.9, 0.95)</td></tr><tr><td>Betas</td><td colspan="4"></td></tr><tr><td>Optimizer Batch Size</td><td></td><td>AdamW</td><td></td><td></td></tr><tr><td>512</td><td></td><td>512</td><td>256</td><td>128</td></tr><tr><td>Trainable Modules</td></tr><tr><td>MLP Projector</td><td>Trainable Trainable</td><td></td><td>Frozen</td><td>Frozen</td></tr><tr><td>Connector</td><td>Trainable</td><td>Trainable</td><td>Trainable</td><td>Trainable</td></tr><tr><td>Condition Tokenizer</td><td>Trainable</td><td>Trainable</td><td>Trainable</td><td>Trainable</td></tr><tr><td>Vision Encoder</td><td>Frozen</td><td>Trainable</td><td>Frozen</td><td>Frozen</td></tr><tr><td>LLM</td><td>Frozen</td><td>Trainable</td><td>Frozen</td><td>Frozen</td></tr><tr><td>Diffusion Model</td><td>Frozen</td><td>Trainable</td><td>Trainable</td><td>Trainable</td></tr><tr><td colspan="5">Data Sampling Ratio</td></tr><tr><td>Image→Text-Camera (single-view Und.)</td><td>0.5</td><td>0.5</td><td></td><td></td></tr><tr><td>Text-Camera→Image (single-view Gen.)</td><td>0.5</td><td>0.5</td><td>0.14</td><td>0.09</td></tr><tr><td>Image→3D (cross-view Gen.)</td><td></td><td></td><td>0.86</td><td></td></tr><tr><td>Image→3D (cross-view Gen. &amp; Recon.)</td><td></td><td></td><td></td><td>0.91</td></tr></table>

## 3.3. Training Recipe

We adopt a four-stage training strategy to learn the whole framework. The first two stages perform cross-modal alignment and supervised fine-tuning (SFT) for single-view understanding and generation across the vision, language, and camera modalities. These stages cover scene and physicalstate perception, as well as free-viewpoint spatial simulation. Building upon the acquired single-view capabilities, the final two stages focus on post-training for 3D world modeling, extending the model to sequential generation and reconstruction. The complete training recipe is summarized in Table 1, and each stage is detailed below.

• Stage I. In this stage, we align the vision encoder with the

LLM by training only the MLP projector to predict scene descriptions and camera parameters from input images. For generation, text and Omni-Camera representation are used to condition target image synthesis. We train the learnable queries and connector to map LLM hidden states into diffusion conditioning signals, while the vision encoder, LLM, and diffusion model remain frozen.

• Stage II. After cross-modal alignment, we unfreeze all modules except the VAE and fine-tune the full framework using the same data and objectives as in Stage I. To stabilize optimization and preserve the pretrained visual representations, we scale the gradients of the vision encoder by 0.1.

• Stage III. We then extend Puffin-World from single-view generation to cross-view 3D world modeling. Conditioned on one or more reference views and their corresponding Omni-Camera maps, the model jointly denoises all target views within a unified sequence. We train the model on trajectories with diverse and challenging camera motions, including large translations, in-place rotations, and full 360<sup>◦</sup> exploration. This stage is performed as post-training on the converged single-view model: the vision encoder and LLM are frozen, while the connector, diffusion backbone, and condition fusion module, remain trainable.

• Stage IV. Finally, we improve Puffin-World for geometry reconstruction by jointly modeling appearance and depth. We activate a geometry branch that generates the depth maps alongside the target RGB views. Both of them share the same flow-matching formulation but are combined through the separately weighted objective of Eq. 9. This stage keeps the same trainable modules as Stage III and additionally trains the depth-modality embedding.

## 3.4. Multi-Task Synergy

Beyond the individual tasks above, Puffin-World exhibits promising multi-task synergy for complex, closed-loop applications that jointly involve perception, reasoning, and generation. In particular, mimic world exploration supports 3D world generation from the same initial viewpoint while following a shared camera trajectory, whereas self-calibrated world exploration automatically detects and corrects gravity misalignment through predicted actions and imagined target observations, resembling the closed-loop interaction paradigm of World-Action Models (WAMs) [117]. These capabilities highlight the potential of Puffin-World for virtual reality and embodied intelligence. Thanks to its unified multimodal framework, Puffin-World realizes these diverse capabilities within a single model without relying on external modules. We present the corresponding results in Section 5.6.

## 4. Dataset

## 4.1. Puffin-16M

Datasets with precise, high-quality alignment among visual content, semantics, and physical properties remain scarce in the multimodal community. Puffin-4M [61] narrows this gap with 4 million vision-language-camera triplets, but still faces three limitations: (i) its scale remains insufficient for larger models; (ii) all images are fixed at 512 × 512, restricting multi-resolution generation and requiring non-square inputs to be centrally cropped and resized for camera understanding, which may discard useful content; and (iii) its spatial coverage is dominated by single-view data, with only limited cross-view pairs and no long-range camera trajectories. To overcome these limitations, we introduce Puffin-16M, consisting of 15 million vision-language-camera triplets with diverse resolutions and aspect ratios, together with 1 million trajectories featuring diverse and challenging rational camera motions. We denote these two subsets as Puffin-Cam-15M and Puffin-Traj-1M, respectively, and summarize the comparison with prior datasets in Tab. 2.

## 4.1.1. Puffin-Cam-15M

The construction pipeline follows a procedure similar to that of Puffin-4M [61], including panoramic data collection and preprocessing, perspective image generation, and scene and reasoning captioning. Specifically, we expand the source panoramic images used in Puffin-4M from 200K to 900K, covering diverse scenarios such as outdoor and indoor scenes, synthetic and real-world environments, and self-driving street views across different countries. Such diversity in the source panoramas provides a strong foundation for scaling large multimodal models, particularly for multimodal spatial intelligence. Due to variations in 360<sup>◦</sup> camera calibration and acquisition stability, some panoramas exhibit geometric distortions and misalignment, especially in consumer-grade capture and autonomous-driving onboard settings. We therefore apply geometric correction techniques based on line segmentation and vanishing point estimation, aligning the panoramas with the gravity direction and improving structural consistency. Compared with Puffin-4M, we exclude Stanford2D3D [4] from the source panoramic data to ensure fair evaluation on this benchmark for camerato-world understanding.

Following the standard protocol adopted in recent studies [61, 96], we generate diverse perspective views from each panorama using a virtual pinhole camera with varying intrinsic and extrinsic parameters. Specifically, we uniformly sample roll and pitch from $[ - 4 5 ^ { \circ } , 4 5 ^ { \circ } ]$ and vertical FoV from [20<sup>◦</sup>, 105<sup>◦</sup>], while adaptively determining the number of crops according to the resolution of each source panorama. This process yields approximately 15M perspective images with precise camera parameters and diverse aspect ratios (e.g., 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, and 16:9). We retain the full set for camera-to-world understanding, while further curating around 8M images based on the realism and aesthetic quality of the source panoramas for camera-controllable image generation. We then employ advanced multimodal large language models (e.g., Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct [7]) to annotate each image with high-quality captions that include detailed scene semantics and structured chain-of-thought reasoning over spatial relationships, camera parameters, and underlying physical principles. In addition, we annotate the camera height of each panorama into five levels, namely, underwater shot, lowposition shot, eye-level shot, high-position shot, and aerial shot, to facilitate more advanced spatial reasoning in future work. The detailed captioning prompts are provided in Figure 10 of the Appendix.

## 4.1.2. Puffin-Traj-1M

As shown in Figure 9, most existing 3D world modeling datasets [19, 38, 67, 81, 101, 133] are limited in terms of camera rotational diversity. Specifically, roll is typically restricted to $[ - 5 ^ { \circ } , 5 ^ { \circ } ]$ and pitch to $[ - 1 0 ^ { \circ } , 1 0 ^ { \circ } ]$ . This limitation arises because most datasets are captured using handheld cameras, where the primary source of variation comes from camera translation rather than rotation. Consequently, models trained on such data often struggle to understand challenging camera motions and accurately simulate the corresponding future world states, particularly in real-world scenarios.

However, a spatially intelligent agent should be able to actively explore its environment from arbitrary viewpoints. To this end, we construct Puffin-Traj-1M, a largescale dataset comprising one million trajectories with diverse and challenging camera motions. The dataset includes long-horizon trajectories featuring continuous look-down, look-up, clockwise rotation, counterclockwise rotation, and full 360<sup>◦</sup> surrounding-view exploration.

![](images/516b4ff74bfbf0d4d1c8c220fa0ffbcc6b73b971bcb1920da1efffa196648b25.jpg)  
Figure 4. An overview of the constructed Puffin-16M dataset. It comprises 15 million vision-language-camera triplets with diverse resolutions and aspect ratios (left), along with 1 million trajectories featuring diverse and challenging rotational camera motions (right).

Table 2. Dataset Comparisons. Compared with previous datasets for single-task [10, 11, 37, 44, 51, 96] and unified-task [61]. Our Puffin-16M dataset shows promising advantages across scale, versatility, spatial distribution, and motion diversity. For the camera parameters, we denote the intrinsic parameters: focal length (f), radial distortion coefficient (ξ); and the extrinsic parameters: roll (ϕ), pitch (θ), yaw (ψ), camera height (h).
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task Type</td><td rowspan="2">Intrinsics</td><td rowspan="2">Extrinsics</td><td rowspan="2"># Frames</td><td colspan="7">Details</td></tr><tr><td>Camera</td><td>Text</td><td>Reasoning</td><td>Single-View</td><td>Cross-View</td><td>Trajectory</td><td>Multi-Ratio</td></tr><tr><td>GeoCalib [96]</td><td>Understanding</td><td>f,ξ</td><td>φ,θ</td><td>37K</td><td>V</td><td>X</td><td>X</td><td></td><td>X</td><td>x</td><td>X</td></tr><tr><td>CTRL-C [51]</td><td>Understanding</td><td>f</td><td>φ,θ</td><td>45K</td><td></td><td>x</td><td>X</td><td></td><td>x</td><td>X</td><td>x</td></tr><tr><td>Deepcalib [11]</td><td>Understanding</td><td>f,ξ</td><td></td><td>67K</td><td></td><td>x</td><td>X</td><td>V</td><td>x</td><td>x</td><td>x</td></tr><tr><td>ParamNet [44]</td><td>Understanding</td><td>f</td><td>φ,θ</td><td>190K</td><td>L</td><td>X</td><td>x</td><td>L</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Perceptual [37]</td><td>Understanding</td><td>f</td><td>φ,θ</td><td>390K</td><td></td><td>X</td><td>x</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>PreciseCam [10]</td><td>Generation</td><td>f,ξ</td><td>φ,θ</td><td>57K</td><td>V</td><td>V</td><td>x</td><td>V</td><td>x</td><td>X</td><td>X</td></tr><tr><td>Puffin-4M [61]</td><td>Unified Multimodal</td><td>f</td><td>φ,0, ψ</td><td>4M</td><td></td><td>V</td><td></td><td></td><td></td><td>x</td><td>X</td></tr><tr><td>Puffin-16M</td><td>Unified Multimodal</td><td>f</td><td>φ, 0, ψ, h</td><td>16M</td><td>√</td><td>V</td><td></td><td></td><td>V</td><td>J</td><td></td></tr></table>

Specifically, we adopt the same data construction pipeline as Puffin-Cam-15M, while introducing an additional camera extrinsic parameter, namely yaw, to enable cross-view exploration and arbitrary-viewpoint camera rotations. All camera extrinsics are sampled from broad ranges: roll and pitch are uniformly sampled from [−45<sup>◦</sup>, 45<sup>◦</sup>], while yaw is sampled from [0<sup>◦</sup>, 360<sup>◦</sup>). For each trajectory, one extrinsic parameter is randomly selected to define the camera motion pattern, and frames are generated either in a single-pass or recursivepass manner. The single-pass strategy provides long-range scene exploration trajectories, whereas the recursive-pass strategy encourages the emergence of spatial memory and facilitates the preservation of 3D consistency over extended horizons. The vertical field-of-view is randomly sampled from [60<sup>◦</sup>, 100<sup>◦</sup>].

## 4.2. Additional Captions for Public Datasets

Beyond contributing our self-constructed dataset, we also enrich the commonly used public datasets [19, 38, 67, 81, 101, 133] in terms of the physical and geometric cues.

## 4.2.1. Physical Caption

Although these datasets provide accurate per-frame camera poses, they primarily characterize relative motion across views and lack the absolute orientation required to anchor observations to the physical world. Models trained solely on such trajectories can therefore learn scene-centric motion patterns without establishing a persistent physical reference, making gravity direction, horizon location, and absolute camera orientation ambiguous. To address this limitation, we leverage Puffin-World to estimate absolute camera parameters for every frame in the public training datasets, from which gravity fields and latitude maps are derived as physical priors for our physics propagation mechanism. Beyond the training data, we further annotate a broad collection of widely used public datasets, including ImageNet [82], GPIC [15], Objects365 [85], CC12M [16], and Megalith-10M [72]. In total, our release covers 28 datasets and approximately 44.5M images with absolute camera annotations, providing large-scale supervision for gravity-aware perception, physically grounded world modeling, and future research on spatial intelligence. An overview of the annotated datasets is shown in Figure 8 and the details of each annotated dataset are listed in Table 7 of the Appendix.

## 4.2.2. Geometric Caption

For public synthetic datasets [38, 81, 101], we directly use the provided dense geometric annotations. For real-world datasets, however, the available depth supervision is often incomplete. Although ScanNet [19] and DL3DV [67] provide camera poses, their depth maps may be sparse or missing. In ScanNet, depth measurements suffer from sensor holes and invalid observations. For DL3DV, the depth maps obtained through MVS reconstruction [84], following CUT3R [99] are incomplete. To obtain dense geometric supervision, we run DA3 [64] on every frame to predict dense depth maps. We then align these predictions with the available sparse depth annotations, preserving the reliable scale of the original measurements while filling in missing regions. This yields dense and geometrically consistent depth annotations for real-world videos. The comparison of the original sparse depth label, our re-annotated dense depth label, and our predicted depth is visualized in Figure 11 of the Appendix.

## 5. Experiments

## 5.1. Implementation Details

## 5.1.1. Network Configuration

We develop two architectural variants, namely Puffin-World-Base and Puffin-World-Pro, targeting different trade-offs between physical world understanding and world simulation capabilities. Specifically, Puffin-World-Base is initialized with the pretrained C-RADIOv3-H [36], Qwen2.5-7B-Instruct [78], and SD3.5-Medium-2.5B [89] as its geometryaware vision encoder, large language model, and diffusion backbone, respectively. To further enhance the generative fidelity, Puffin-World-Pro adopts the pretrained C-RADIOv4-H [79], Qwen2.5-1.5B-Instruct [78], and SD3.5- Large-8.1B [89] for the corresponding components. For both variants, a set of 64 learnable queries and a lightweight sixlayer Transformer connector are used to project the LLM hidden states into diffusion conditioning embeddings. In addition to the above two variants, we also release Puffin-World-Caption, an expert model dedicated to annotating physical world states, such as absolute camera rotation parameters (roll and pitch) and the intrinsic parameter (FoV). To support large-scale dataset construction, Puffin-World-Caption is built upon the pretrained C-RADIOv3-H [36] and Qwen3.5-0.8B-Instruct [93], providing an effective balance between annotation accuracy and inference efficiency for large-scale physical-world labeling.

## 5.1.2. Training/Inference Settings

The whole training process comprises four stages, whose per-stage learning rates, batch sizes, data sampling ratios, and trainable modules are summarized in Table 1. Images and mapped depth are encoded by the frozen VAE into 16- channel latents. For single-view camera-controllable generation, we train at the native resolutions and aspect ratios of Puffin-Cam-15M, whereas for 3D world modeling we crop all views to $6 4 0 \times 6 4 0$ . Each sequence contains a total of 8 views: the number of reference (initial) views is randomly sampled from 1 to 3 in Stage III and fixed to 1 in Stage IV, with the remaining views serving as generation targets. Diffusion timesteps are drawn from a logit-normal distribution $( \mu { = } 0 , \sigma { = } 1 )$ ) with a resolution-dependent shift, and Stage III and Stage IV adopt a larger shift to bias training toward higher-noise timesteps that benefit cross-view consistency. For the Omni-Camera representation, the absolute representation is placed before the relative representation in the channel order, resulting in a 9-channel dense camera map $\mathbf { C } \in \mathbb { R } ^ { H \times W \times 9 }$ with the same spatial resolution as the image.

For the multi-view post-training Stage III, the mode is trained on a mixture of trajectory data to learn diverse camera motions involving both translation (DL3DV [67] and RealEstate10K [133]) and rotation (Puffin-Traj-1M), together with single-view camera-controllable generation samples from Puffin-Cam-15M. The single-view batches serve as rehearsal data to preserve the camera controllability acquired in Stage II. We adopt a multi-source sampling strategy, where homogeneous mini-batches are drawn from each data source according to fixed repeat factors of 3:2:1:1. To reduce the training cost of the proposed physics propagation, we pre-compute the absolute camera fields for all training frames offline using the physics perception capability (at Stage-II) of Puffin-World. During training, each view is therefore directly conditioned on its corresponding Omni-Camera map. The condition fusion module is warm-started from the single-view model. Stage IV then inherits the Stage-III weights and activates the joint appearance–geometry branch (Sec. 3.2.3), widening the multi-view mixture with Hypersim [81], MVS-Synth [38], TartanAir [101], and Scan-Net [19] while keeping the same source-balanced sampling and single-view rehearsal. The depth loss weight is ramped from zero to $\omega _ { \mathrm { m a x } } { = } 1$ over the first 3,000 iterations (Eq. 9).

Table 3. Evaluation results on camera-to-world understanding. The comparison methods are evaluated on the public datasets: Stanford2D3D [4], MegaDepth [56], TartanAir [101], and LaMAR [83]. Our Puffin-World outperforms the previous methods on all median error metrics and most AUC metrics. Note that AnyCalib<sup>†</sup> [94] is specifically designed for intrinsic parameter estimation (i.e., FoV estimation) and is included for reference. We color the best and second best results.
<table><tr><td rowspan="3">Approach</td><td rowspan="3"></td><td colspan="4">Roll [degrees]</td><td colspan="4">Pitch [degrees]</td><td colspan="4">FoV [degrees]</td></tr><tr><td rowspan="2">error ↓</td><td colspan="3">AUC1/5/10°↑</td><td rowspan="2">error ↓</td><td colspan="3">AUC1/5/10°↑</td><td rowspan="2">error ↓</td><td colspan="3">AUC 1/5/10° ↑</td></tr><tr><td>33.8</td><td>63.9</td><td>2.58</td><td>21.6</td><td>46.9</td><td>65.7</td><td>6.67</td><td>8.1 20.6</td><td></td></tr><tr><td rowspan="11">Stan] a3]</td><td>DeepCalib [70] Perceptual [37]</td><td>1.59 2.08</td><td>26.8</td><td>53.8</td><td>79.2 70.7</td><td>3.17</td><td>21.5</td><td>41.8</td><td>57.8</td><td>13.84</td><td></td><td>2.8 7.7</td><td>37.6 16.1</td></tr><tr><td>CTRL-C [51]</td><td>3.04</td><td>23.2</td><td>43.0</td><td>56.9</td><td>3.43</td><td>18.3</td><td>38.6</td><td>53.8</td><td>8.50</td><td>7.7</td><td>18.2</td><td>31.5</td></tr><tr><td>MSCC [88]</td><td>3.43</td><td>13.5</td><td>36.8</td><td>57.3</td><td>2.64</td><td>22.6</td><td>45.0</td><td>60.5</td><td>5.81</td><td>9.6</td><td>23.8</td><td>41.6</td></tr><tr><td>ParamNet [44]</td><td>1.14</td><td>44.6</td><td>73.9</td><td>84.8</td><td>1.94</td><td>29.2</td><td>56.7</td><td>73.1</td><td>9.01</td><td>5.8</td><td>14.3</td><td>27.8</td></tr><tr><td></td><td></td><td>21.7</td><td>24.6</td><td>25.8</td><td></td><td>15.4</td><td>19.9</td><td>22.4</td><td></td><td>6.2</td><td>11.5</td><td>15.2</td></tr><tr><td>SVA [69]</td><td></td><td></td><td></td><td>79.1</td><td>0.95</td><td>51.2</td><td>63.0</td><td>69.2</td><td>3.65</td><td>22.2</td><td>39.5</td><td></td></tr><tr><td>UVP [75]</td><td>0.52</td><td>65.3</td><td>74.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>51.3</td></tr><tr><td>GeoCalib [96]</td><td>0.40</td><td>83.1</td><td>91.8</td><td>94.8</td><td>0.93</td><td>52.3</td><td>74.8</td><td>84.6</td><td>3.21</td><td>17.4</td><td>40.0</td><td>59.4</td></tr><tr><td>AnyCalib† [94]</td><td>0.29</td><td>1</td><td>97.4</td><td>98.5</td><td>0.53</td><td>1 73.3</td><td>88.8</td><td>94.0</td><td>2.55</td><td>21.1</td><td>46.8</td><td>64.6</td></tr><tr><td>Puffin-World DeepCalib [70]</td><td>1.41</td><td>93.1 34.6</td><td>65.4</td><td>79.4</td><td>5.19</td><td>11.9</td><td>27.8</td><td>44.8</td><td>1.62 11.14</td><td>34.5 5.6</td><td>61.1 12.1</td><td>76.4 22.9</td></tr><tr><td rowspan="8">Me] aas66d</td><td>Perceptual [37] CTRL-C [51]</td><td>1.07</td><td>47.9</td><td>72.4</td><td>83.2 84.2</td><td>3.49</td><td>19.8</td><td>39.1</td><td>54.2</td><td>13.40</td><td>2.9</td><td>8.2</td><td>16.8</td></tr><tr><td>MSCC [88]</td><td>0.88 0.90</td><td>54.5 53.1</td><td>75.0 72.8</td><td>82.1</td><td>4.80 5.73</td><td>16.6 19.0</td><td>33.2 33.2</td><td>46.5 44.3</td><td>18.65 10.80</td><td>2.0 6.0</td><td>5.8 14.6</td><td>12.8 26.2</td></tr><tr><td>ParamNet [44]</td><td>1.17</td><td>43.4</td><td>70.7</td><td>82.2 36.2</td><td>3.99</td><td>15.4 13.6</td><td>34.5 20.6</td><td>53.3 24.9</td><td>11.01</td><td>3.2</td><td>10.1 16.1</td><td>21.3 21.1</td></tr><tr><td>SVA [69] UVP [75]</td><td>0.51</td><td>31.9 69.2</td><td>35.0 81.6</td><td>86.9</td><td>4.59</td><td>21.6</td><td>36.2</td><td>47.4</td><td>10.92</td><td>9.4 8.2</td><td>18.7</td><td>29.8</td></tr><tr><td>GeoCalib [96]</td><td>0.36</td><td>82.6</td><td>90.6</td><td>94.0</td><td>1.94</td><td>32.4</td><td>53.3</td><td>67.5</td><td>4.46</td><td>13.6</td><td>31.7</td><td>48.2</td></tr><tr><td>AnyCalib† [94]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.14</td><td>19.4</td><td>40.8</td><td>59.1</td></tr><tr><td>Puffin [61]</td><td>0.32</td><td>84.9</td><td>93.4</td><td>96.2 96.7</td><td>1.08</td><td>47.6</td><td>68.2</td><td>79.4</td><td>2.42</td><td>23.9</td><td>47.8</td><td>64.1</td></tr><tr><td>Puffin-World DeepCalib [70]</td><td>0.28 1.95</td><td>87.7 94.2 24.7 55.4</td><td></td><td>71.5</td><td>1.01 3.27</td><td>49.7 16.3</td><td>69.3 38.8</td><td>79.6 58.5</td><td>2.41</td><td>24.8</td><td>48.8</td><td>65.5</td></tr><tr><td rowspan="8">ar ra01] Puffin [61]</td><td rowspan="5">Perceptual [37] CTRL-C [51] MSCC [88] ParamNet [44] SVA [69] UVP [75] GeoCalib [96]</td><td>2.24 1.68 3.50</td><td>23.2</td><td>48.6</td><td>66.7</td><td>2.86</td><td>23.5</td><td>44.6 48.6</td><td>61.5</td><td>8.07 15.06</td><td>1.5 5.1 10.7</td><td>8.8 8.9 25.4</td><td>27.2 17.1</td></tr><tr><td></td><td>32.8</td><td>59.1</td><td>74.1</td><td>2.39</td><td>24.6</td><td>38.6</td><td>65.2 54.3</td><td>5.64 11.18</td><td></td><td></td><td>43.5</td></tr><tr><td>1.63</td><td>15.0 34.5</td><td>37.2 59.2</td><td>57.7 73.9</td><td>3.48 3.05</td><td>18.8 19.4</td><td>42.0</td><td>60.3</td><td>8.21</td><td>4.4 6.0</td><td>11.8 16.8</td><td>23.0 31.6</td></tr><tr><td>9.48 0.89</td><td>32.4 52.1</td><td>39.6</td><td>44.1 71.9</td><td>18.46 2.48</td><td></td><td>21.2 36.2</td><td>28.8</td><td>34.5</td><td>43.01</td><td>8.8</td><td>16.1</td><td>21.6</td></tr><tr><td></td><td>0.43 71.3</td><td>64.8 83.8</td><td>89.8</td><td>1.49</td><td></td><td>38.2</td><td>48.8 62.9</td><td>58.6 76.6</td><td>9.15 4.90</td><td>15.8 14.1</td><td>25.8 30.4</td><td>35.7</td></tr><tr><td>AnyCalib† [94]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.62</td><td>15.5</td><td>36.4</td><td>47.6 55.1</td></tr><tr><td>Puffin-World DeepCalib [70]</td><td>0.40 0.31</td><td>71.7 80.1</td><td>86.2 90.0</td><td>92.1 94.2</td><td>0.95 0.67</td><td>51.0 60.2</td><td>68.2 78.1</td><td>79.3 87.2</td><td>7.48 2.34</td><td>16.3 26.6</td><td>28.5 46.4</td><td>39.0 59.5</td></tr><tr><td rowspan="10">LM 83] UVP [75] GeoCalib [96] AnyCalib† [94] Puffin [61] Puffin-World</td><td>Perceptual [37]</td><td>1.15 1.29</td><td>44.1 40.0</td><td>73.9 68.9</td><td>84.8 81.6</td><td>4.68 2.83</td><td>10.8 21.2</td><td>28.3 44.7</td><td>49.8 62.6</td><td>10.93 17.78</td><td>0.7 3.0</td><td>13.0 5.3</td><td>24.0 10.7</td></tr><tr><td>CTRL-C [51] MSCC [88]</td><td>1.20</td><td>43.5 39.6</td><td>70.9 60.7</td><td>82.5 72.8</td><td>1.94 3.02</td><td>27.6 20.9</td><td>54.7 41.8</td><td>70.2 55.7</td><td>5.64</td><td>9.8</td><td>24.6</td><td>43.2</td></tr><tr><td>ParamNet [44]</td><td>1.44 0.93</td><td>51.7</td><td>77.0</td><td>86.0 9.7</td><td>2.15</td><td>27.0 3.4</td><td>52.7 5.7</td><td>70.2 7.0</td><td>14.78 14.71</td><td>3.2 2.8 1.2</td><td>8.3 6.8</td><td>16.8 14.3</td></tr><tr><td>SVA [69]</td><td>0.38</td><td>8.6 9.2 72.7 81.8</td></table>

We enable classifier-free guidance (CFG) through condition dropout during training. For single-view cameracontrollable generation, we jointly replace the text prompt with an empty string and zero the Omni-Camera condition with probability 0.1. For multi-view world generation, the same joint dropout is applied to the text and all per-view Omni-Camera maps. We additionally drop the absolute Perspective Field with probability 0.15 while retaining the relative ray map, reducing over-reliance on absolute cues and enabling component-wise guidance. At inference, camera-toworld understanding uses greedy decoding, while generation employs 50 sampling steps. We set the CFG scale to 4.5 and 2.0 for single-view camera-controllable image generation and 3D world modeling, respectively.

## 5.2. Camera-to-World Understanding Results

Following prior works, we compare Puffin-World against a diverse set of camera-to-world understanding methods, including learning-based approaches such as DeepCalib [70], Perceptual [37], CTRL-C [51], MSCC [88], ParamNet [44], GeoCalib [96], and Puffin [61], as well as classical geometric methods including SVA [69] and UVP [75]. For each image, gravity estimation performance is measured by the angular errors of roll and pitch, while camera intrinsic estimation is evaluated using the error in vertical (vFoV). Following standard practice, we report both the median error and the Area Under the Recall Curve (AUC) at error thresholds of 1<sup>◦</sup>, 5<sup>◦</sup>, and 10<sup>◦</sup>. Evaluations are conducted on four widely adopted benchmarks: MegaDepth [56], TartanAir [101], LaMAR [83], and Stanford2D3D [4].

As shown in Table 3, Puffin-World consistently outperforms all competing methods in terms of median error and achieves the best performance on the majority of AUC metrics across all four benchmarks. These results demonstrate its strong capability to perceive and ground observations within the physical world. In particular, Puffin-World surpasses Puffin [61] with a noticeable margin, showing the meaningful and progressive benefits of scaling the model and data for this task. We note that AnyCalib [94] is specifically designed for camera intrinsic estimation, particularly FoV prediction, and is included only as a specialized reference baseline. Despite its task-specific design, Puffin-World still outperforms AnyCalib on most benchmarks, highlighting the effectiveness and generalizability of our unified framework.

## 5.3. Camera-Controllable Generation Results

We compare Puffin-World against recent state-of-the-art multimodal generation models, including GPT Image2 [73], Nano Banana 2 [29], Qwen-Image2-Pro [128], FLUX.2- dev [48], and Z-Image [13], as well as specialized cameracontrollable image generation methods [10, 61]. All methods are evaluated using the same text prompts and camera speci-

fications.

To quantitatively assess camera controllability, we leverage Puffin-World to estimate the camera parameters of each generated image and derive the corresponding pixel-wise perspective fields. We then compare these estimated fields against the GT fields and report both mean and median angular errors for the up vector, gravity direction, and latitude map, all measured in degrees. In addition, we report the Frechet Inception Distance (FID) to evaluate the overall vi-´ sual fidelity and realism of the generated images. Since no existing benchmark provides paired text descriptions and precise camera parameters across diverse camera configurations and aspect ratios, we introduce Puffin-Cam-Bench to fill this gap. It contains 600 text–camera specification pairs covering a wide range of scenes, viewpoints, camera poses, fields of view, and aspect ratios, enabling comprehensive evaluation of camera-controllable generation for free-viewpoint spatial simulation.

Table 4 summarizes the quantitative results, while Figure 5 provides qualitative comparisons across different methods. Our method consistently outperforms existing multimodal generation models [13, 29, 48, 73, 128] by a substantial margin across all evaluation metrics. Although these general-purpose models can produce visually appealing and high-fidelity images, they often fail to preserve spatially consistent scene geometry under specified camera configurations, resulting in noticeable deviations from the desired viewpoints. In contrast, Puffin-World achieves accurate and robust camera control while maintaining high visual quality and scene diversity. Its strong generalization across a wide range of scenes, viewpoints, and camera configurations highlights its practicality for real-world camera-controllable image generation beyond prior specialized models [10, 61]. Additional qualitative results are presented in Figure 14 in the Appendix.

## 5.4. 3D World Modeling Results

As visualized in Figure 6, Puffin-World demonstrates versatile capabilities for 3D world modeling, including imageto-3D generation, text-to-3D generation, flexible rotation control, compound motion control, native 3D world-state modeling, and 3D reconstruction. For text-to-3D generation, we first synthesize a reference view from the text prompt and initial absolute camera parameters through free-viewpoint spatial simulation, and then generate subsequent views conditioned on the reference image and target camera trajectory. For compound motion control, we sequentially combine translation trajectories from RealEstate10K [133] with rotational trajectories sampled from Puffin-Traj-1M, covering diverse roll, pitch, and yaw variations. For each predicted set of native 3D world states, we visualize the appearance (image), geometry (depth), and physics (gravity field and latitude map), together with the resulting 3D reconstruction.

![](images/50c83b9fcec018d0459bc52d33571fe56e5da056e30252cf0cf3adbef2d5eb15.jpg)  
This modern living room features a sleek, dark gray sectional sofa adorned with patterned throw pillows, a round marble coffee table, and large windows that offer a view of the outdoors. The space is illuminated by track lighting and has a minimalist aesthetic with concrete ceilings and a geometric rug. The camera parameters (roll, pitch, and field-of-view) are: -0.3651, 0.0178, 1.4866.

Figure 5. Comparison results on camera-controllable text-to-image generation for free-viewpoint spatial simulation. For each result, we visualize the angular error between the simulated gravity and the ground truth; values closer to 0<sup>◦</sup> indicate better spatial alignment. Essential elements are underlined in the text prompt.  
Table 4. Camera-controllable generation evaluation on Puffin-Cam-Bench.
<table><tr><td rowspan="2">Approach</td><td colspan="2">Up Vector [degrees]</td><td colspan="2">Latitude [degrees]</td><td colspan="2">Gravity [degrees]</td><td>Visual Quality</td></tr><tr><td>mean error ↓</td><td>median error ↓</td><td>mean error ↓</td><td>median error ↓</td><td>mean error ↓</td><td>median error ↓</td><td>FID↓</td></tr><tr><td>GPT Image2 [73]</td><td>24.13</td><td>23.44</td><td>16.42</td><td>15.98</td><td>28.83</td><td>29.38</td><td>99.36</td></tr><tr><td>Nano Banana2 [29]</td><td>24.01</td><td>23.19</td><td>15.98</td><td>15.52</td><td>28.00</td><td>26.92</td><td>90.11</td></tr><tr><td>Qwen-Image2-Pro [128]</td><td>24.12</td><td>23.51</td><td>16.45</td><td>15.53</td><td>28.57</td><td>28.69</td><td>98.51</td></tr><tr><td>FLUX.2-dev [48]</td><td>24.28</td><td>23.62</td><td>15.75</td><td>15.24</td><td>28.31</td><td>28.36</td><td>97.06</td></tr><tr><td>Z-Image [13]</td><td>24.64</td><td>24.04</td><td>16.96</td><td>16.09</td><td>29.41</td><td>28.82</td><td>98.33</td></tr><tr><td>PreciseCam [10]</td><td>13.37</td><td>12.61</td><td>12.31</td><td>12.01</td><td>17.07</td><td>14.01</td><td>90.89</td></tr><tr><td>Puffin [61]</td><td>3.86</td><td>3.48</td><td>4.77</td><td>4.43</td><td>4.92</td><td>2.87</td><td>80.29</td></tr><tr><td>Puffin-World</td><td>0.96</td><td>0.84</td><td>1.34</td><td>1.26</td><td>1.32</td><td>0.79</td><td>75.93</td></tr></table>

The results demonstrate that Puffin-World effectively handles diverse and challenging camera actions, including long trajectories, extreme rotations, and complex compound motions. The generated views faithfully preserve the semantic content of the reference image or text prompt while maintaining coherent spatial structure across viewpoints. Moreover, the jointly predicted world states exhibit physically and geometrically plausible distributions while preserving realistic visual appearance. The resulting 3D reconstructions further exhibit clear scene structure, coherent surface geometry, and strong cross-view consistency, precisely recovering the spatial layout and major geometric details of the generated world and demonstrating that the predicted multi-view observations can be consolidated into a consistent 3D representation. More results are presented in Figure 15, Figure 16, and Figure 17 in the Appendix.

We then quantitatively evaluate Puffin-World on RealEstate10K (50 held-out clips with no overlap with the training data) and Puffin-Traj-Bench (100 clips) following prior work [14], with comparisons against MotionCtrl [103], CameraCtrl [34], ViewCrafter [120], SEVA [132], and MV-GenMaster [14]. Visual fidelity is measured using PSNR, SSIM, and LPIPS on both benchmarks. Since RealEstate10K does not provide absolute-camera ground truth, cameracontrol accuracy is evaluated only on Puffin-Traj-Bench by re-estimating roll, pitch, and vertical FoV from the generated frames using the same Puffin-World camera-understanding branch for all methods, reporting median angular error and AUC@1/5/10<sup>◦</sup>. As listed in Table 5, Puffin-World achieves the best PSNR, SSIM, and LPIPS on RealEstate10K. On the more challenging Puffin-Traj-Bench, which contains substantially larger rotations and intrinsic variations, Puffin-World achieves the best PSNR and LPIPS with competitive SSIM, while substantially outperforming all baselines in cameracontrol accuracy across roll, pitch, and FoV. These results demonstrate that Puffin-World maintains strong perceptual fidelity on conventional real-world trajectories while providing markedly more accurate and robust camera control under challenging spatial transformations.

![](images/70d1760930089bbcba79db18dc38e43aba02253cb6daf3c52970170a2fef266d.jpg)  
Figure 6. Qualitative results of 3D world modeling. Our Puffin-World supports various 3D modeling tasks: image-to-3D generation, text-to-3D generation, flexible rotation control, compound motion control, native 3D world states formulation, and 3D reconstruction. The red box marks the model’s input, and others are generated.

Table 5. Evaluation results on 3D world generation. We evaluate all methods on RealEstate10K and Puffin-Traj-Bench. For Puffin-Traj-Bench, Roll, Pitch, and FoV report the median absolute errors of camera parameters re-estimated from generated frames, with Roll/Pitch AUC reported at $1 ^ { \circ } / 5 ^ { \circ } / 1 0 ^ { \circ }$ thresholds. Note that camera-control metrics are not reported on RealEstate10K due to the absence of corresponding absolute-camera ground truth. <sup>†</sup>MVGenMaster’s training data include samples from the evaluated RealEstate10K benchmark. We color the best and second best results within each benchmark.
<table><tr><td rowspan="2" colspan="2">Approach</td><td colspan="3">Visual Fidelity</td><td colspan="3">Roll [degrees]</td><td colspan="4">Pitch [degrees]</td><td>FoV [degrees]</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>error↓</td><td>AUCD 1/5/10° ↑</td><td></td><td>error↓</td><td></td><td>AUC1/5/10°↑</td><td></td><td>error↓</td></tr><tr><td></td><td>MotionCtrl [103]</td><td>13.38</td><td>0.482</td><td>0.507</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CameraCtrl [34]</td><td>16.93</td><td>0.589</td><td>0.352</td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td></td><td>ViewCrafter [120]</td><td>16.04</td><td>0.569</td><td>0.331</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>SEVA [132]</td><td>14.84</td><td>0.529</td><td>0.421</td><td></td><td></td><td></td><td></td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>Ral10K</td><td>MVGenMaster [14]†</td><td>17.11</td><td>0.591</td><td>0.348</td><td></td><td></td><td></td><td></td><td></td><td>一</td><td>一</td><td></td></tr><tr><td></td><td>Puffin-World</td><td>17.22</td><td>0.595</td><td>0.318</td><td></td><td></td><td></td><td></td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td></td><td>MotionCtrl [103]</td><td>12.53</td><td>0.438</td><td>0.604</td><td>3.20</td><td>0.25 0.43</td><td>0.54</td><td>2.15</td><td>0.28</td><td>0.49</td><td>0.59</td><td>10.04</td></tr><tr><td>Puf--Bnch</td><td>CameraCtrl [34]</td><td>14.28</td><td>0.512</td><td>0.532</td><td>11.79</td><td>0.05 0.11</td><td>0.21</td><td>10.81</td><td>0.05</td><td>0.16</td><td>0.26</td><td>45.84</td></tr><tr><td></td><td>ViewCrafter [120]</td><td>14.52</td><td>0.525</td><td>0.510</td><td>1.92</td><td>0.32 0.53</td><td>0.66</td><td>2.89</td><td>0.23</td><td>0.43</td><td>0.58</td><td>6.43</td></tr><tr><td></td><td>SEVA [132]</td><td>17.94</td><td>0.614</td><td>0.307</td><td>1.35</td><td>0.38 0.63</td><td>0.75</td><td>1.76</td><td>0.29</td><td>0.56</td><td>0.71</td><td>12.59</td></tr><tr><td></td><td>MVGenMaster [14]</td><td>13.64</td><td>0.496</td><td>0.589</td><td>9.24</td><td>0.06 0.14</td><td>0.28</td><td>9.43</td><td>0.08</td><td>0.17</td><td>0.30</td><td>26.98</td></tr><tr><td></td><td>Puffin-World</td><td>18.00</td><td>0.613</td><td>0.288</td><td>0.80</td><td>0.57 0.79</td><td>0.88</td><td>1.10</td><td>0.45</td><td>0.72</td><td>0.83</td><td>2.96</td></tr></table>

Table 6. Ablation study on the proposed physics propagation (PP) for 3D world generation with challenging camera motions.
<table><tr><td rowspan="2">Approach</td><td rowspan="2">Motion</td><td colspan="3">Image Metrics</td><td colspan="2">Camera Metrics</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Error_R↓</td><td>Error_P↓</td></tr><tr><td rowspan="4">Baseline</td><td>Roll</td><td>22.97</td><td>0.69</td><td>0.12</td><td>2.35</td><td>6.73</td></tr><tr><td>Pitch</td><td>18.53</td><td>0.60</td><td>0.26</td><td>2.17</td><td>5.60</td></tr><tr><td>Yaw</td><td>18.12</td><td>0.59</td><td>0.30</td><td>1.31</td><td>3.10</td></tr><tr><td>Average</td><td>19.87</td><td>0.62</td><td>0.23</td><td>1.94</td><td>5.14</td></tr><tr><td rowspan="4">Baseline w/ PP Roll</td><td></td><td>24.02</td><td>0.74</td><td>0.10</td><td>2.04</td><td>6.31</td></tr><tr><td>Pitch</td><td>19.18</td><td>0.65</td><td>0.24</td><td>2.03</td><td>5.28</td></tr><tr><td>Yaw</td><td>18.42</td><td>0.62</td><td>0.30</td><td>1.22</td><td>2.67</td></tr><tr><td>Average</td><td>20.54</td><td>0.67</td><td>0.21</td><td>1.76</td><td>4.75</td></tr></table>

## 5.5. Ablation Study

We isolate the contribution of physics propagation, which anchors a generated trajectory to a consistent gravity-aligned frame (Sec. 3.2.3), and focus on rotation trajectories because they provide the most sensitive probe of absolute grounding: unlike translation, pure rotation directly changes the camera orientation with respect to gravity and the horizon, making inconsistencies in the physical frame immediately observable. We use Puffin-Traj-Bench (spanning roll, pitch, and yaw rotations), generate the corresponding views with and without physics propagation, and recover the camera parameters of each generated frame using Puffin-World’s physicsperception capability. The calibrated roll and pitch errors (Error R and Error P) are compared against the GT cameras and reported in Table 6, together with PSNR, SSIM, and LPIPS against the target views. Physics propagation consistently improves both visual fidelity and physical world grounding, with the largest gains on roll and pitch trajectories that are directly coupled to gravity. Yaw motion shows smaller appearance improvements because it rotates around the gravity axis, although physics propagation still suppresses spurious horizon tilt and improves physical consistency. These results support our design rationale: propagating the absolute physical state from the reference view through the known relative motion maintains a coherent gravity-anchored representation across the trajectory, leading to more stable horizons, better scene uprightness, improved perceptual quality, and camera states that better follow the intended motion. Qualitative comparisons are provided in Figure 12 of the Appendix.

## 5.6. Applications

As mentioned in Section 3.4, Puffin-World enables multi-task synergy for complex, closed-loop applications that jointly involve perception, reasoning, and generation. Two representative closed-loop capabilities further illustrate the benefits of this unified formulation as shown in Figure 7. In mimic world exploration, Puffin-World reproduces a target exploration process by expanding a 3D world from a common starting view under a prescribed camera trajectory. In self-calibrated world exploration, the model instead reasons about its current physical state, predicts corrective camera actions, and imagines the resulting observations to progressively resolve gravity misalignment, forming an interaction loop reminiscent of World-Action Models (WAMs). Both behaviors emerge from the same multimodal model, without auxiliary perception, calibration, or generation modules, demonstrating the potential of Puffin-World as a general foundation for interactive 3D environments, virtual reality, and embodied agents.

![](images/eec6c1ba280949ca0d5f6f6b6149f0aad5f80450f6088b6943b2d5a65612e4d4.jpg)  
Figure 7. Closed-loop applications of Puffin-World: Mimic world exploration (left) and self-calibrated world exploration, both of which require multi-task synergy involving understanding and generation. The type of each sub-task is marked with circled U (understanding) and G (generation).

## 6. Conclusion

We present Puffin-World, a unified multimodal model that perceives, simulates, and generates the physical world through three native 3D world states: physics, geometry, and appearance. In particular, the Omni-Camera representation integrates gravity-aware absolute orientation with ray-based relative motion into a unified camera condition, supporting single-view control, multi-view synthesis, and challenging camera trajectories. Building on this representation, physics propagation transfers the natively perceived absolute state of the reference view across future frames, maintaining a coherent physical frame for stable and self-calibrated world generation. Within a single framework, Puffin-World jointly supports physical-world perception, free-viewpoint spatial simulation, and 3D world generation and reconstruction, while Puffin-16M provides large-scale vision–language–camera supervision and diverse challenging motions for scaling these capabilities. Extensive experiments demonstrate strong and consistent performance across these tasks and further highlight the benefits of multi-task synergy for closed-loop applications.

We believe that grounding multimodal tasks in a shared, physically anchored world representation provides a promising path toward versatile spatial intelligence and physical AI. Future directions include extending the framework to dynamic scenes and longer horizons, and modeling richer physical states beyond gravity and latitude.

## Acknowledgment

This research is supported by cash and in-kind funding from NTU S-Lab and industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20224-0003) and NRF Investigatorship (NRF-NRFI11-2026-0008).

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 3, 4

[2] Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026. 4

[3] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014. 25

[4] Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017. 9, 12, 13

[5] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 4

[6] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky,

Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 4

[7] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 9, 24

[8] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aaron van den Oord, Inbar Mosseri,¨ Adrian Bolton, Satinder Singh, and Tim Rocktaschel. Genie¨ 3: A new frontier for world models. 2025. 3, 4

[9] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15791–15801, 2025. 4

[10] Edurne Bernal-Berdun, Ana Serrano, Belen Masia, Matheus Gadelha, Yannick Hold-Geoffroy, Xin Sun, and Diego Gutierrez. Precisecam: Precise camera control for textto-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2724–2733, 2025. 4, 10, 13, 14

[11] Oleksandr Bogdan, Viktor Eckstein, Francois Rameau, and Jean-Charles Bazin. Deepcalib: A deep learning approach for automatic intrinsic calibration of wide field-of-view cameras. In Proceedings ofthe 15th ACM SIGGRAPH European Conference on Visual Media Production, pages 1–10, 2018. 3, 10

[12] Ollin Boer Bohan. Megalith-10m. https : //huggingface.co/datasets/madebyollin/ megalith-10m, 2024. Hugging Face dataset. 25

[13] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 13, 14

[14] Chenjie Cao, Chaohui Yu, Fan Wang, Xiangyang Xue, and Yanwei Fu. Mvgenmaster: Scaling multi-view generation

from any image via 3d priors enhanced diffusion model. arXiv preprint arXiv:2411.16157, 2024. 4, 14, 16

[15] Keshigeyan Chandrasegaran, Kyle Sargent, Suchir Agarwal, Michael Jang, Michael Poli, Juan Carlos Niebles, Justin Johnson, Jiajun Wu, and Li Fei-Fei. Gpic: A giant permissive image corpus for visual generation. arXiv preprint arXiv:2605.30341, 2026. 11, 23, 25

[16] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3557–3567, 2021. 11, 25

[17] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 4

[18] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 3, 4

[19] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5828–5839, 2017. 3, 9, 11, 12, 23, 24, 25, 27, 28

[20] Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, and Yonggang Qi. Fantasyworld: Geometry-consistent world modeling via unified video and 3d prediction. In ICLR, 2026. 4

[21] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 4

[22] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 25

[23] Junyuan Deng, Wei Yin, Xiaoyang Guo, Qian Zhang, Xiaotao Hu, Weiqiang Ren, Ping Tan, et al. Boost 3d reconstruction using diffusion-based monocular camera calibration. arXiv preprint arXiv:2411.17240, 2024. 4

[24] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024. 6

[25] Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025. 4

[26] Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T Barron, et al. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026. 7

[27] Ruiqi Gao\*, Aleksander Holynski\*, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole\*. Cat3d: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems, 2024. 3, 4, 5

[28] Eran Goldman, Roei Herzig, Aviv Eisenschtat, Jacob Goldberger, and Tal Hassner. Precise detection in densely packed scenes. In CVPR, 2019. 25

[29] Google. Nano banana 2, 2026. 3, 13, 14

[30] David Ha and Jurgen Schmidhuber. World models.¨ arXiv preprint arXiv:1803.10122, 2(3), 2018. 4

[31] Xiaoyang Han, Jianhua Li, Kewang Deng, Zukai Chen, Xuanke Shi, Sihan Wang, Boxuan Li, Linyan Wang, Siyi Xie, Xin You, et al. Vision as unified multimodal generation. arXiv preprint arXiv:2607.06560, 2026. 4

[32] Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. arXiv preprint arXiv:2310.16828, 2023. 4

[33] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge University Press, 2003. 3

[34] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3, 4, 14, 16

[35] Xiankang He, Guangkai Xu, Bo Zhang, Hao Chen, Ying Cui, and Dongyan Guo. Diffcalib: Reformulating monocular camera calibration as diffusion-based dense incident map generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3428–3436, 2025. 4

[36] Greg Heinrich, Mike Ranzinger, Hongxu Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Radiov2. 5: Improved baselines for agglomerative vision foundation models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 22487–22497, 2025. 11

[37] Yannick Hold-Geoffroy, Kalyan Sunkavalli, Jonathan Eisenmann, Matthew Fisher, Emiliano Gambaretto, Sunil Hadap, and Jean-Franc¸ois Lalonde. A perceptual measure for deep single image camera calibration. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2354–2363, 2018. 3, 10, 12, 13

[38] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2821–2830, 2018. 9, 11, 12, 24, 25, 27, 28

[39] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement. arXiv preprint arXiv:2504.01934, 2025. 4

[40] Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2world: Crafting video diffusion models to interactive world models. arXiv preprint arXiv:2505.14357, 2025. 3, 4

[41] Sebastian Janampa and Marios Pattichis. Sofi: Multi-scale deformable transformer for camera calibration with enhanced line queries. arXiv preprint arXiv:2409.15553, 2024. 3

[42] Qing Jiang, Gen Luo, Yuqin Yang, Yuda Xiong, Yihao Chen, Zhaoyang Zeng, Tianhe Ren, and Lei Zhang. Chatrex: Taming multimodal llm for joint perception and understanding. arXiv preprint arXiv:2411.18363, 2024. 25

[43] Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Liu Qin, and Lei Zhang. Referring to any person. In ICCV, 2025. 25

[44] Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F Fouhey. Perspective fields for single image camera calibration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17307– 17316, 2023. 3, 4, 5, 10, 12, 13

[45] Alex Kendall, Matthew Grimes, and Roberto Cipolla. Posenet: A convolutional network for real-time 6-dof camera relocalization. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 2938–2946, 2015. 3

[46] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 25

[47] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020. 25

[48] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025. 3, 13, 14

[49] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. STream3R: Scalable sequential 3D reconstruction with causal transformer. In ICLR, 2026. 4

[50] Jinwoo Lee, Minhyuk Sung, Hyunjoon Lee, and Junho Kim. Neural geometric parser for single image camera calibration. In European Conference on Computer Vision, pages 541– 557. Springer, 2020. 3

[51] Jinwoo Lee, Hyunsung Go, Hyunjoon Lee, Sunghyun Cho, Minhyuk Sung, and Junho Kim. Ctrl-c: Camera calibration transformer with line-classification. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 16228–16237, 2021. 3, 10, 12, 13

[52] Wanhee Lee, Klemen Kotar, Rahul Venkatesh, Jared Watrous, Honglin Chen, Khai Loong Aw, and Daniel Yamins. Unified 3d scene understanding through physical world modeling. In ICLR, 2026. 4

[53] Jiefeng Li, Can Wang, Hao Zhu, Yihuan Mao, Hao-Shu Fang, and Cewu Lu. Crowdpose: Efficient crowded scenes pose estimation and a new benchmark. In CVPR, 2019. 25

[54] Wenbin Li, Sajad Saeedi, John McCormac, Ronald Clark, Dimos Tzoumanikas, Qing Ye, Yuzhong Huang, Rui Tang,

and Stefan Leutenegger. Interiornet: Mega-scale multisensor photo-realistic indoor scenes dataset. arXiv preprint arXiv:1809.00716, 2018. 25

[55] Xiaoyu Li, Bo Zhang, Pedro V Sander, and Jing Liao. Blind geometric distortion correction on images through deep learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4855–4864, 2019. 3

[56] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2041–2050, 2018. 12, 13, 27, 28

[57] Kang Liao, Chunyu Lin, Yao Zhao, and Mai Xu. Modelfree distortion rectification framework bridged by distortion distribution map. IEEE Transactions on Image Processing, 29:3707–3718, 2020. 3

[58] Kang Liao, Chunyu Lin, and Yao Zhao. A deep ordinal distortion estimation approach for distortion rectification. IEEE Transactions on Image Processing, 30:3362–3375, 2021. 3

[59] Kang Liao, Lang Nie, Shujuan Huang, Chunyu Lin, Jing Zhang, Yao Zhao, Moncef Gabbouj, and Dacheng Tao. Deep learning for camera calibration and beyond: A survey. arXiv preprint arXiv:2303.10559, 2023. 3

[60] Kang Liao, Zongsheng Yue, Zhonghua Wu, and Chen Change Loy. Mowa: Multiple-in-one image warping model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 3

[61] Kang Liao, Size Wu, Zhonghua Wu, Linyi Jin, Chao Wang, Yikai Wang, Fei Wang, Wei Li, and Chen Change Loy. Thinking with camera: A unified multimodal model for camera-centric understanding and generation. In International Conference on Learning Representations, 2026. 4, 6, 9, 10, 12, 13, 14

[62] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 4

[63] Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422, 2025. 4

[64] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. In ICLR, 2026. 11

[65] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 25

[66] Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, et al. Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376, 2025. 3

[67] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 3, 9, 11, 23, 24, 25, 26, 27, 28

[68] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In CVPR, 2016. 25

[69] Yaroslava Lochman, Oles Dobosevych, Rostyslav Hryniv, and James Pritts. Minimal solvers for single-view lensdistorted camera auto-calibration. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2887–2896, 2021. 12, 13

[70] Manuel Lopez, Roger Mari, Pau Gargallo, Yubin Kuang, Javier Gonzalez-Jimenez, and Gloria Haro. Deep single image camera calibration with radial distortion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11817–11825, 2019. 12, 13

[71] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739– 7751, 2025. 4

[72] Ollin Matsubara and Draw Things AI Team. Megalith-10m: A dataset of 10 million public-domain photographs, 2024. CC0/Flickr-Commons images; Florence-2 captions available in the megalith-10m-florence2 variant. 11

[73] OpenAI. Gpt image2, 2026. 3, 13, 14

[74] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuoka Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 4

[75] Remi Pautrat, Shaohui Liu, Petr Hruby, Marc Pollefeys, and´ Daniel Barath. Vanishing point estimation in uncalibrated images with prior gravity direction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14118–14127, 2023. 12, 13

[76] Marc Pollefeys, Reinhard Koch, and Luc Van Gool. Selfcalibration and metric reconstruction inspite of varying and unknown intrinsic camera parameters. International Journal ofComputer Vision, 32(1):7–25, 1999. 3

[77] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 4

[78] A Yang Qwen, Baosong Yang, B Zhang, B Hui, B Zheng, B Yu, Chengpeng Li, D Liu, F Huang, H Wei, et al. Qwen2.5 technical report. arXiv preprint, 2024. 11

[79] Mike Ranzinger, Greg Heinrich, Collin McCarthy, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Cradiov4 (tech report). arXiv preprint arXiv:2601.17237, 2026. 11

[80] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Muller, Alexan-¨ der Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132, 2025. 4

[81] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10912–10922, 2021. 9, 11, 12, 24, 25, 27, 28

[82] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. arxiv 2014. arXiv preprint arXiv:1409.0575, 2014. 11

[83] Paul-Edouard Sarlin, Mihai Dusmanu, Johannes L Schonberger, Pablo Speciale, Lukas Gruber, Viktor Larsson,¨ Ondrej Miksik, and Marc Pollefeys. Lamar: Benchmarking localization and mapping for augmented reality. In European Conference on Computer Vision, pages 686–704. Springer, 2022. 4, 12, 13, 27, 28

[84] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 11, 25

[85] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8429–8438, 2019. 11, 25

[86] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34: 19313–19325, 2021. 3

[87] Lin Song, Wenbo Li, Guoqing Ma, Wei Tang, Bo Wang, Yuan Zhang, Yijun Yang, Yicheng Xiao, Jianhui Liu, Yanbing Zhang, et al. Joyai-image: Awaking spatial intelligence in unified multimodal understanding and generation. arXiv preprint arXiv:2605.04128, 2026. 4

[88] Xu Song, Hao Kang, Atsunori Moteki, Genta Suzuki, Yoshie Kobayashi, and Zhiming Tan. Mscc: Multi-scale transformers for camera calibration. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3262–3271, 2024. 3, 12, 13

[89] Stability AI. Introducing stable diffusion 3.5, 2024. 11

[90] Zihan Su, Hongyang Wei, Kangrui Cen, Yong Wang, Guanhua Chen, Chun Yuan, and Xiangxiang Chu. Generation enhances understanding in unified multimodal models via multi-representation generation. arXiv preprint arXiv:2601.21406, 2026. 4

[91] Hongxuan Tang, Hao Liu, and Xinyan Xiao. Ugen: Unified autoregressive multimodal model with progressive vocabulary learning. arXiv preprint arXiv:2503.21193, 2025.

[92] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 3, 4

[93] Qwen Team. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804, 2026. 11

[94] Javier Tirado-Gar´ın and Javier Civera. Anycalib: Onmanifold learning for model-agnostic single-view camera calibration. arXiv preprint arXiv:2503.12701, 2025. 4, 12, 13

[95] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024. 4

[96] Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. Geocalib: Learning singleimage calibration with geometric optimization. In European Conference on Computer Vision, pages 1–20, 2024. 3, 4, 9, 10, 12, 13

[97] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 4

[98] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In ICME, 2021. 25

[99] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 4, 11

[100] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 4

[101] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 9, 11, 12, 13, 24, 25, 27, 28

[102] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3, 4

[103] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3, 4, 14, 16

[104] Scott Workman, Connor Greenwell, Menghua Zhai, Ryan Baltenberger, and Nathan Jacobs. Deepfocal: A method for direct focal length estimation. In 2015 IEEE International Conference on Image Processing (ICIP), pages 1369–1373. IEEE, 2015. 3

[105] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding

for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025. 3, 4

[106] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. In ICLR, 2026. 4

[107] Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025. 4

[108] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025. 4

[109] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 4

[110] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 4

[111] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 4

[112] Liuyue Xie, Jiancong Guo, Ozan Cakmakci, Andre Araujo, Laszlo A Jeni, and Zhiheng Jia. Aligndiff: Learning physically-grounded camera alignment via diffusion. arXiv preprint arXiv:2503.21581, 2025. 3

[113] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 4

[114] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 4

[115] Yuxiang Yang, Junjie Yang, Yufei Xu, Jing Zhang, Long Lan, and Dacheng Tao. Apt-36k: A large-scale benchmark for animal pose estimation and tracking. In NeurIPS, 2022. 25

[116] Yuxiang Yang, Yingqi Deng, Yufei Xu, and Jing Zhang. Aptv2: Benchmarking animal pose estimation and tracking with a large-scale dataset and beyond. arXiv preprint arXiv:2312.15612, 2023. 25

[117] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 9

[118] Xiaoqing Yin, Xinchao Wang, Jun Yu, Maojun Zhang, Pascal Fua, and Dacheng Tao. Fisheyerecnet: A multi-context

collaborative deep network for fisheye image rectification. In Proceedings of the European Conference on Computer Vision (ECCV), pages 469–484, 2018. 3

[119] Hang Yu, Yufei Xu, Jing Zhang, Wei Zhao, Ziyu Guan, and Dacheng Tao. Ap-10k: A benchmark for animal pose estimation in the wild. In NeurIPS, 2021. 25

[120] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis . IEEE Transactions on Pattern Analysis & Machine Intelligence, (01):1–18, 5555. 14, 16

[121] Yu Yuan, Xijun Wang, Yichen Sheng, Prateek Chennuri, Xingguang Zhang, and Stanley Chan. Generative photography: Scene-consistent camera control for realistic text-toimage synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7920–7930, 2025. 4

[122] Menghua Zhai, Scott Workman, and Nathan Jacobs. Detecting vanishing points using global image context in a non-manhattan world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5657–5665, 2016. 3

[123] Hong Zhang, Zhongjie Duan, Xingjun Wang, Yuze Zhao, Weiyi Lu, Zhipeng Di, Yixuan Xu, Yingda Chen, and Yu Zhang. Nexus-gen: A unified model for image understanding, generation, and editing. arXiv preprint arXiv:2504.21356, 2025. 4

[124] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. arXiv preprint arXiv:2402.14817, 2024. 3

[125] Shifeng Zhang, Yiliang Xie, Jun Wan, Hansheng Xia, Stan Z Li, and Guodong Guo. Widerperson: A diverse dataset for dense pedestrian detection in the wild. IEEE transactions on multimedia, 2019. 25

[126] Xinjie Zhang, Jintao Guo, Shanshan Zhao, Minghao Fu, Lunhao Duan, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, and Kaifu Zhang. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567, 2025. 4

[127] Yisu Zhang, Chenjie Cao, Tengfei Wang, Xuhui Zuo, Junta Wu, Jianke Zhu, and Chunchao Guo. Worldstereo: Bridging camera-guided video generation and scene reconstruction via 3d geometric memories. arXiv preprint arXiv:2603.02049, 2026. 4

[128] Bing Zhao, Chenfei Wu, Deqing Li, Hao Meng, Jiahao Li, Jie Zhang, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kuan Cao, et al. Qwen-image-2.0 technical report. arXiv preprint arXiv:2605.10730, 2026. 3, 13, 14

[129] Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. Versecrafter: Dynamic realistic video world model with 4d geometric control. In CVPR, 2026. 4

[130] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV, 2019. 25

[131] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma,

Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 3, 4

[132] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. In IEEE/CVF International Conference on Computer Vision, pages 12405–12414, 2025. 14, 16

[133] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 3, 9, 11, 13, 24, 25, 27, 28

[134] Chenchen Zhu, Fanyi Xiao, Andres Alvarado, Yasmine´ Babaei, Jiabo Hu, Hichem El-Mohri, Sean Culatana, Roshan Sumbaly, and Zhicheng Yan. Egoobjects: A large-scale egocentric dataset for fine-grained object understanding. In ICCV, 2023. 25

[135] Shengjie Zhu, Abhinav Kumar, Masa Hu, and Xiaoming Liu. Tame a wild camera: In-the-wild monocular camera calibration. Advances in Neural Information Processing Systems, 36:45137–45149, 2023. 4

[136] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jiwen Lu, and Jie Zhou. Astra: General interactive world model with autoregressive denoising. In ICLR, 2026. 3, 4

## A. Appendix

This appendix provides additional details and qualitative results to complement the main paper. We first elaborate on the datasets used and released in this work, including camera annotations for public datasets, additional physical labels in Puffin-16M, and their camera parameter statistics. We then provide further training details, including the construction of dense geometric supervision for real-world datasets. Finally, we present extensive qualitative results for camerato-world understanding, camera-controllable generation, and 3D world modeling, covering diverse scenes, challenging camera configurations, long trajectories, compound motions, native world-state prediction, and 3D reconstruction.

## A.1. Dataset Details

## A.1.1. Annotating Public Datasets

Publicly used datasets [15, 19, 67] show versatile abilities to various tasks; however, most of them lack accurate absolute camera parameters anchored to the real world. To this end, we annotate 28 datasets using Puffin-World’s powerful world perception capability, of which the understanding branch treats camera properties as a language modality and predicts three parameters from each image: roll, pitch, and vertical field-of-view (FoV). The overview of the annotated datasets is shown in Figure 8. As listed in Tab. 7, the collection comprises 22 single-image datasets spanning recognition, detection, segmentation, pose estimation, and generative pre-training, together with six sequential/3D Absolute-Camera datasets. Overall, the release contains approximately 44.5M camera annotations. Across these datasets, several consistent biases emerge: roll is strongly concentrated around 0<sup>◦</sup>, indicating a dominant level-camera bias; pitch is generally negative, reflecting a tendency toward downward-looking viewpoints; and FoV varies substantially with capture regime, with web photographs typically exhibiting narrower views than synthetic, scanned, or video-based datasets. Together, these statistics reveal that conventional vision corpora substantially under-sample strongly tilted, rolled, and wide-angle viewpoints.

Several datasets exhibit distinctive camera distributions. For example, EgoObjects shows the strongest downwardlooking bias, consistent with head-mounted capture of hands and nearby objects, while DeepFashion exhibits exceptionally low variation in roll, pitch, and FoV, reflecting tightly controlled studio photography. In contrast, TartanAir presents the largest rotational diversity, making it particularly valuable for camera orientations rarely observed in natural photographs; ScanNet and SKU110K also show pronounced downward pitch due to room-scanning and shelffacing capture, respectively. These annotations transform widely used vision datasets into camera-grounded resources that support viewpoint-bias analysis, camera-conditioned generation, camera-diverse data curation, and large-scale monocular physics perception. For the Absolute-Camera subsets, as analyzed in Figure 9, aligning per-frame predictions with camera trajectories further enables consistency analysis and gravity-grounded world modeling.

ScanNet  
![](images/a5edb880df550188a7d0c8feff7e8fc1fcdd3cfbc0d27355054f1f0e298a88e4.jpg)  
Figure 8. The overview of the public datasets annotated by Puffin-World, with a total of 28 datasets covering approximately 44.5 million images across diverse scene distributions.

![](images/273f24549ee05e62097bb8a7c458e38fc16178590bb5eba08eb8f2203d44f581.jpg)  
DL3DV  
Figure 9. The distributions of the absolute camera parameters (roll and pitch) and FoV of existing public 3D or sequentia datasets [19, 38, 67, 81, 101, 133].

## A.1.2. Camera Height of Puffin-16M

Beyond the precise camera parameters and camera trajectories provided in Puffin-16M, we further annotate the camera height of each original panoramic image, prior to perspective rendering, using Qwen3-VL-32B [7]. We categorize camera height into five levels—underwater shot, low-position shot, eye-level shot, high-position shot, and aerial shot— to provide a coarse yet physically meaningful description of the observer’s vertical position in the scene. The resulting height label is subsequently inherited by all perspective views rendered from the corresponding panorama, thereby complementing their precise camera parameters with an additional high-level physical cue. The detailed annotation prompts are provided in Figure 10. We expect these complementary camera annotations to facilitate richer spatial reasoning, viewpoint-aware generation, and future studies of camera-grounded physical world understanding and generation.

Table 7. Camera parameter statistics of 28 annotated datasets using Puffin-World. For every source dataset we report its original task, the number of validly annotated images and scenes, and the mean/median/standard deviation of the predicted roll, pitch, and vertical FoV (in degrees). <sup>†</sup>computed on a ∼4.9M-image sample; <sup>‡</sup>pooled from the released train+val statistics.
<table><tr><td colspan="2"></td><td></td><td colspan="4">Camera Parameters (µ/med/σ, °)</td></tr><tr><td>Dataset</td><td>Original task</td><td>#Images</td><td>#Scenes</td><td>Roll</td><td>Pitch</td><td>FoV</td></tr><tr><td colspan="7">Single-image datasets</td></tr><tr><td>ADE20K [130]</td><td>Semantic segmentation</td><td>27.6K</td><td></td><td>0.0/0.0/1.9</td><td>-1.3/-1.0/8.0</td><td>32.7/31.2/11.5</td></tr><tr><td>AP-10K [119]</td><td>Animal pose estimation</td><td>10.0K</td><td>一</td><td>0.4/0.2/8.1</td><td>-9.2/-6.1/13.9</td><td>26.5/23.2/8.3</td></tr><tr><td>APT-36K [115]</td><td>Animal pose tracking</td><td>35.7K</td><td>一</td><td>0.2/0.0/3.5</td><td>-4.9/-3.7/11.3</td><td>24.1/20.3/6.0</td></tr><tr><td>APTv2 [116]</td><td>Animal pose tracking</td><td>41.3K</td><td></td><td>0.2/0.0/3.5</td><td>-5.1/-3.7/11.2</td><td>24.2/22.9/6.0</td></tr><tr><td>CC12M [16]</td><td>Vision-language pre-training</td><td>10.90M</td><td>一</td><td>0.2/0.0/5.7</td><td>-2.8/-0.3/11.8</td><td>32.3/29.0/10.6</td></tr><tr><td>COCO [65]</td><td>Detection &amp; segmentation</td><td>122K</td><td></td><td>0.0/0.0/6.2</td><td>-7.3/-4.0/15.0</td><td>34.9/32.5/11.7</td></tr><tr><td>CrowdPose [53]</td><td>Human pose estimation</td><td>20.0K</td><td></td><td>0.1/0.0/3.5</td><td>-5.1/-3.7/8.5</td><td>31.6/28.7/10.7</td></tr><tr><td>DeepFashion [68]</td><td>Fashion recognition &amp; retrieval</td><td>13.7K</td><td></td><td>0.1/0.3/3.5</td><td>1.1/0.4/3.6</td><td>32.6/32.1/3.4</td></tr><tr><td>EgoObjects [134]</td><td>Egocentric object detection</td><td>241.6K</td><td>一</td><td>-0.1/0.0/8.0</td><td>-24.5/-25.6/15.3</td><td>39.3/36.7/10.2</td></tr><tr><td>GPIC† [15]</td><td>Image generation corpus</td><td>4.9M</td><td>一</td><td>0.1/0.0/5.6</td><td>-2.6/-1.3/13.0</td><td>33.8/31.2/11.6</td></tr><tr><td>HumanRef [43]</td><td>Referring expression comprehension</td><td>45.3K</td><td>一</td><td>0.1/0.0/3.6</td><td>-2.5/-1.5/8.4</td><td>32.5/31.2/9.9</td></tr><tr><td>IRS [98]</td><td>Stereo matching / depth (syn.)</td><td>188.3K</td><td>一</td><td>0.2/0.0/4.9</td><td>-8.3/-5.0/14.8</td><td>54.8/56.7/6.1</td></tr><tr><td>ImageNet-1K‡ [22]</td><td>Image classification</td><td>1.33M</td><td></td><td>0.1/0.0/7.8</td><td>-7.8/-3.7/15.7</td><td>29.0/25.6/8.9</td></tr><tr><td>InteriorVerse [54]</td><td>Inverse rendering (syn. indoor)</td><td>61.2K</td><td></td><td>0.8/0.0/6.3</td><td>-1.3/-0.8/16.9</td><td>71.6/68.6/14.9</td></tr><tr><td>MPII [3]</td><td>Human pose estimation</td><td>25.0K</td><td>一</td><td>0.0/0.0/2.7</td><td>-5.0/-3.8/7.8</td><td>27.6/26.6/6.2</td></tr><tr><td>Megalith-10M [12]</td><td>Image generation corpus</td><td>9.38M</td><td></td><td>0.0/0.0/6.7</td><td>-3.1/-2.3/12.1</td><td>34.6/31.3/12.0</td></tr><tr><td>Objects365 [85]</td><td>Object detection</td><td>1.72M</td><td>一</td><td>0.1/0.0/4.0</td><td>-4.4/-2.7/11.2</td><td>33.4/31.3/11.0</td></tr><tr><td>Open Images [47]</td><td>Detection &amp; multi-label cls.</td><td>8.32M</td><td>一</td><td>0.1/0.0/6.2</td><td>-3.1/-1.1/14.8</td><td>31.8/31.2/8.2</td></tr><tr><td>Rexverse-2M [42]</td><td>Multimodal grounding &amp; captioning</td><td>414.6K</td><td></td><td>0.2/0.0/6.8</td><td>-4.0/-0.9/12.9</td><td>32.2/28.9/11.1</td></tr><tr><td>SA-1B [46]</td><td>Promptable segmentation</td><td>897.6K</td><td>一</td><td>0.2/0.0/4.9</td><td>2.0/1.5/13.1</td><td>36.3/34.4/13.2</td></tr><tr><td>SKU110K [28]</td><td>Dense object detection</td><td>19.1K</td><td>一</td><td>-0.1/0.0/1.9</td><td>-13.9/-14.1/6.6</td><td>49.6/51.9/12.5</td></tr><tr><td>WiderPerson [125]</td><td>Pedestrian detection</td><td>13.3K</td><td>一</td><td>0.1/0.0/1.4</td><td>-1.8/-1.4/5.8</td><td>29.1/27.2/8.3</td></tr><tr><td colspan="7">Multi-view datasets (absolute-pose annotations)</td></tr><tr><td>DL3DV [67]</td><td>Novel view synthesis / 3D recon.</td><td>2.16M</td><td>6,377</td><td>-0.3/-0.3/4.9</td><td>-7.0/-6.1/16.0</td><td>53.4/54.8/8.7</td></tr><tr><td>HyperSim [81]</td><td>3D scene understanding (syn.)</td><td>73.6K</td><td>448</td><td>0.2/0.0/6.8</td><td>-1.4/-0.8/11.0</td><td>44.7/45.0/6.0</td></tr><tr><td>MVS-Synth [38]</td><td>Multi-view stereo (syn.)</td><td>12.0K</td><td>120</td><td>0.0/0.0/0.4</td><td>3.3/3.7/9.7</td><td>45.8/46.8/4.1</td></tr><tr><td>RealEstate10K [133]</td><td>Novel view synthesis / camera traj.</td><td>595.4K</td><td>5,134</td><td>-0.2/-0.2/1.8</td><td>-5.2/-3.8/7.9</td><td>51.4/55.3/10.2</td></tr><tr><td>ScanNet [19]</td><td>RGB-D reconstruction</td><td>2.32M</td><td>1,468</td><td>-0.1/0.0/1.5</td><td>-14.5/-15.1/9.4</td><td>27.8/26.6/6.0</td></tr><tr><td>TartanAir [101]</td><td>SLAM / visual odometry (syn.)</td><td>306.6K</td><td>18</td><td>-0.1/-0.1/15.1</td><td>-5.3/-4.4/23.8</td><td>68.9/71.4/7.3</td></tr></table>

## A.2. Training Details

Table 8 summarizes the datasets used for Puffin-World across different training stages. We combine large-scale public sequential or 3D datasets with our self-constructed Puffin-16M (Puffin-Cam-15M and Puffin-Traj-1M) to provide complementary supervision in native 3D world states. The public datasets cover diverse real-world and synthetic environments with varying trajectory lengths, motion patterns, and depth availability, thereby providing broad spatial and geometric supervision. When depth annotations are available, they are incorporated to strengthen geometry learning, while datasets without depth primarily contribute to multi-view appearance consistency and camera-conditioned generation.

Puffin-Traj-1M complements these sources with substantially richer and more challenging rotational trajectories, extending the motion distribution beyond that of conventional capture datasets. Puffin-Cam-15M further supplies large-scale language-image-camera triplets and is reused across multiple stages to preserve strong camera perception and generation capabilities. Together, these datasets form a balanced training mixture that progressively supports singleview understanding/generation, cross-view generation, trajectory modeling, and 3D world reconstruction within the same framework.

Although ScanNet [19] and DL3DV [67] provide camera poses, their depth annotations remain incomplete due to sensor holes in ScanNet and sparse MVS reconstruction [84] in DL3DV. As described in Section 4.2, we use DA3 to predict dense depth for each frame and align it with the available measurements, preserving reliable metric scale while filling missing regions. Figure 11 compares the sparse depth pro-

![](images/7bd46f95871d749d078557861fb7b7bebe6b62538e11790d381c45289e1603a4.jpg)

## Output ONLY the label.

Label: Eye-level Shot

Figure 10. The designed prompt for captioning the camera height for the panoramic images. We categorize it into five levels: underwate shot, low-position shot, eye-level shot, high-position shot, and aerial shot. A representative example, consisting of the input panoramic image and the corresponding output label, is shown in the bottom-right corner.  
![](images/fd840c13b2c2e3a0adf09f920dec31331f66d349f633d2662f5078145a2cfce7.jpg)  
Figure 11. Depth map visualization on the DL3DV dataset [67]. From top to bottom, we show the input image, the sparse label provided in the original dataset, our annotated and aligned dense label (training used), and the predicted result by Puffin-World.

vided by the original datasets, our aligned dense annotations used for training, and the depth predicted by Puffin-World. Compared with the original sparse measurements, our annotations provide substantially denser and more complete geometric supervision, reducing holes and missing structures and thereby facilitating unified RGB–depth training without specialized tokenizer designs. Moreover, the depth predicted by Puffin-World exhibits sharper object boundaries and fewer artifacts, while maintaining a spatial layout that is highly consistent with the corresponding input image.

Table 8. Training datasets for our model. We summarize the source category, number of scenes and frames, depth availability, tempora sampling interval, test-set usage, and the corresponding training stages for each dataset.
<table><tr><td>Dataset</td><td>Category</td><td>Scenes / Frames</td><td>Depth</td><td>Sample Interval</td><td>Testset Split</td><td>Training Stage</td></tr><tr><td>DL3DV [67]</td><td>Public</td><td>6377 /2M</td><td>Yes*</td><td>[1,8]</td><td>No</td><td>III, IV</td></tr><tr><td>RealEstate10K [133]</td><td>Public</td><td>5134/600K</td><td>No</td><td>[48,144]</td><td>Yes</td><td>III, IV</td></tr><tr><td>HyperSim [81]</td><td>Public</td><td>756/70K</td><td>Yes</td><td>[1,1]</td><td>No</td><td>IV</td></tr><tr><td>MVS-Synth [38]</td><td>Public</td><td>120 / 12K</td><td>Yes</td><td>[1,2]</td><td>No</td><td>IV</td></tr><tr><td>TartanAir [101]</td><td>Public</td><td>369/ 300K</td><td>Yes</td><td>[1, 10]</td><td>No</td><td>IV</td></tr><tr><td>ScanNet [19]</td><td>Public</td><td>1468 /2M</td><td>Yes*</td><td>[1,8]</td><td>No</td><td>IV</td></tr><tr><td>Puffin-Traj-1M</td><td>Self-Constructed</td><td>1M/ 100M</td><td>No</td><td>[1, 12]</td><td>Yes</td><td>III, IV</td></tr><tr><td>Puffin-Cam-15M</td><td>Self-Constructed</td><td>-/15M</td><td>No</td><td>一</td><td>Yes</td><td>I, II, I, IV</td></tr></table>

<sup>∗</sup> We refined the sparse depth label into the dense one.

![](images/29a31cf738f530a359e14d871d0609ea95526c9812b2183473ca72433534f414.jpg)  
Figure 12. Ablation study of the proposed physics propagation (PP) strategy. For each sample, we show the generation results marked by the error up-vector and the error value with and without PP.

## A.3. More Results

## A.3.1. Camera-to-World Understanding

We provide additional qualitative results across diverse datasets [19, 38, 56, 67, 81, 83, 101, 133] in Figure 13. For each input image, we first estimate its absolute camera parameters using the understanding branch of Puffin-World, and then convert them into the corresponding latitude map and gravity field via Eq. 4 for intuitive visualization. The results demonstrate that Puffin-World generalizes well across diverse scenes and remains robust under challenging camera configurations.

## A.3.2. Camera-Controllable Generation

Additional qualitative results are presented in Figure 14, covering diverse spatial configurations across both indoor and outdoor environments. Puffin-World enables flexible freeviewpoint spatial simulation under a wide range of camera parameters, while preserving realistic appearance, coherent scene layouts, and semantic consistency with the input condition. In particular, the generated views faithfully follow the specified camera orientation and field-of-view, yielding noticeable yet physically plausible changes in perspective, horizon, and spatial composition. These results further demonstrate the robustness and generalization of Puffin-World for camera-controllable generation across diverse scenes and challenging viewpoints.

## A.3.3. 3D World Modeling

As illustrated in Figures 15, 16, and 17, we provide additional qualitative results of 3D world generation under diverse settings, including explicit rotational control, longhorizon trajectories, compound camera actions, text-driven generation, native 3D world-state prediction, and subsequent 3D reconstruction. Across these scenarios, Puffin-World consistently follows the prescribed camera motions while preserving semantic content and coherent spatial structure over extended viewpoint changes. The model remains robust under large rotations and complex combinations of translation and rotation, where maintaining both appearance consistency and physical plausibility is particularly challenging. Moreover, text-driven generation produces geometrically coherent multi-view worlds that remain faithful to the input prompt, while the predicted physics, geometry, and appearance states exhibit strong mutual consistency. The resulting 3D reconstructions further recover clear scene structure and stable spatial layouts, demonstrating that Puffin-World can consolidate its generated multi-view observations into a coherent 3D representation.

![](images/85c251d48b1c35041b7939f2d3f9032d238db3e869ec79296dd38feb617e740c.jpg)  
Figure 13. Visualizations of the estimated physics world state (gravity field and latitude map) on different datasets [19, 38, 56, 67, 81, 83, 101, 133], in which the absolute camera parameters are predicted by Puffin-World.

![](images/ed8fb8b76b33c793fe79f45229d7f57739c36073fde061d9bb8c723f14573d34.jpg)  
Figure 14. Camera-controllable text-to-image generation results for free-viewpoint spatial simulation.

![](images/0c971801eff400e3936a357d81084fe64cff20d0cae06992e99b0a88acbf497e.jpg)  
Figure 15. 3D world generation with diverse rotational controls (top: roll, middle: pitch, bottom: yaw). For each control, we show the generated results from the single-pass path (with visualized physics world state) and recursive path.

![](images/3d83d218b0e01e6927217eba07965d3ee4e30510acc690ef4d62a41285ab1df6.jpg)  
The scene shows a cozy living room with a white couch. a dark wooden desk adorned with decorative items and the words in large letters. and a framed picture on the wall above the desk. The room has hardwood floors and is softly lit by natural light coming through the windows. The initial absolute camera parameters (roll, pitch, and field-of-view) are: 0.0011, -0.0773, 0.9569.

Figure 16. 3D world generation with diverse camera actions and conditions, including long trajectories, compound motions, and text-driven generation.

![](images/0d384d5689b6c40b0568aac3ee190b440cc4b61d8da45319bc4584caaf41c7c1.jpg)  
Figure 17. Additional qualitative results of 3D world modeling. For each combination, we show the generated native 3D world states: appearance (image), geometry (depth), and physics (gravity field and latitude map), and the 3D reconstructed result. Note that only one group of the generated world states is visualized in each sample for clarity.