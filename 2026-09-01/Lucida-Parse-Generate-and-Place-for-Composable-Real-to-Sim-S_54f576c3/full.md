# Lucida: Parse, Generate, and Place for Composable Real-to-Sim Scene Modeling

<sup>1</sup>ByteDance Seed, <sup>2</sup>Peking University, <sup>3</sup>Zhejiang University

See Contributions section for a full author list.

## Abstract

Composable scene modeling aims to recover a real indoor scene as complete, editable object assets arranged as observed, giving robot simulation and embodied AI a simulation-ready replica of the real environment whose objects can be manipulated individually. Existing pipelines decompose the task into three steps—parse the observations into instances, generate an asset for each, and place each asset back—but every step presumes an input that a cluttered capture rarely provides: accurate instance geometry, unoccluded views, and assets that accurately match the observations. We propose Lucida, which keeps this order but redistributes the requirements, so each step consumes only what a real capture reliably provides and precision is reached at the end of the pipeline rather than demanded at its start. Lucida parses the video into a scene graph whose nodes carry per-instance multi-view evidence, generates a complete asset for each instance from its evidence, and places assets with GizmoAct, a VLM policy that casts placement as multi-turn GUI interaction, manipulating the object’s gizmo in a closed loop and deciding itself when alignment is reached. Across scene-level 3D object detection, object pose estimation, and scene reconstruction, Lucida improves mAP over Boxer by 69% on R2S-Scene, raises ADD-SB@0.05 from 57.8% to 83.4% on CA-1M, and increases scene F-Score from 0.794 for SAM 3D to 0.924.

Date: September 1, 2026

Project Page: https://lucida-r2s.github.io/

## 1 Introduction

Composable scene modeling recovers a real indoor scene as separable, editable object assets together with their spatial arrangement, the form in which robot simulation, embodied AI, AR/VR, and content creation consume it. Reconstruction by diferentiable rendering, as in NeRF [60] and 3D Gaussian Splatting [39], reproduces the captured room faithfully but as a monolith, and object-compositional extensions [43, 71, 83, 105] remain tied to the observed geometry, leaving occluded surfaces unrecovered. Real-to-sim systems for robot learning [18, 21, 76] and the CAD retrieval-and-alignment methods they build on [5, 30] return simulation-ready scenes, but populate the room with library assets, capping fidelity at database coverage. Interactive scene synthesis [98, 99] produces plausible rooms that are grounded in no particular capture. No existing route yields objects that are at once complete, editable assets and faithful to the room that was observed.

Building such a scene decomposes into three steps: parse the observations into object instances, generate an asset for each, and place each asset back into the scene. Strong methods exist for every step, yet each presumes an input that a cluttered capture does not provide. Scene parsing is expected to deliver accurate instancelevel estimates such as precise masks, clean per-instance point clouds, or a metric 3D layout, while occlusion

Reconstructed Composable Scene

![](images/71be96e31449b4acfb3000d9764f96cb19f9e7690904e577d6a6d8f76bfdd3bd.jpg)  
Figure 1 Teaser. Given posed multi-view observations of a real indoor scene, our method constructs object-centric multi-view evidence, generates complete 3D assets, and places them into a composable reconstruction. The insets illustrate closed-loop rotation, translation, and scale refinement of object models against the observed scene geometry.

leaves object boundaries ambiguous and repeated furniture makes cross-view association and de-duplication unreliable. Generation presumes an unoccluded, centered view of the object [32, 75] or a cleanly segmented point cloud of it [72], neither of which indoor footage readily yields. Placement, typically cast as 6/9-DoF pose estimation [46, 51, 80], assumes an asset that agrees geometrically with a segmented observation, while generated assets deviate from the real object and depth is noisy. End-to-end methods do not escape these constraints: predicting an object’s model and pose jointly degrades both [67, 108], and generating all assets of a scene with a single model is limited by scarce paired training data and generalizes poorly [17, 37, 59]. Since the three steps run in a fixed order, an unmet requirement at any step degrades every step after it.

A human modeler working from the same capture satisfies none of these requirements. They review the footage to note what is present, build or retrieve a model for each object, and drop every model into the scene, nudging it against the observation—switching viewpoints, correcting the residual—until the two agree. Precision is reached at the end of this process, not demanded at its start. We keep the parse–generate–place order but redistribute the requirements along it, so each step consumes only what a real capture reliably provides. Parsing no longer requires a precise instance reconstruction as its output. Instead, it discovers object instances, consolidates their multi-view evidence, and obtains a representative 3D estimate for initializing subsequent generation and grounding. Generation conditions on this per-object evidence rather than on a clean crop or a segmented point cloud, completing the structure and appearance of the target instance under occlusion, which image generation and editing with unified multimodal models [13, 27, 62] has made practical. Placement accepts an asset that need not match the observation exactly and, from a coarse initial state and a referring cue, recovers the instance’s 9-DoF pose in the point cloud.

We name the resulting system Lucida, after the camera lucida, the drawing aid that superimposes the real scene onto the artist’s page. Geometry-aware keyframe selection and object discovery provide candidate object instances and initial 3D estimates. Object-centric full-sequence evidence consolidation then extends and validates each candidate against the full sequence, after which relation-aware scene refinement organizes the consolidated instances into a scene graph and improves their consistency. Its nodes represent object instances and its edges encode spatial relations; each node carries an evidence bundle combining selected multi-view observations, partial point-cloud observations, a representative 3D box, and a referring description. The bundle serves both later steps: generation uses it as context to synthesize an occluder-free image of the instance and lift that image into a complete 3D asset, and placement takes from it the coarse initial state for refinement.

We cast the place step as multi-turn GUI interaction: the object is presented as if selected in a 3D editor—the scene point cloud, the asset overlay, its 3D box, and a gizmo exposing the object’s local frame—and aligning the asset means issuing a sequence of edits through this interface. GizmoAct, a VLM policy, operates the interface in a closed loop. At each turn it reads the rendered state and emits one executable edit, an incremental pose update in the gizmo frame whose translation and scale are expressed in units of the current object size, so the policy never regresses an absolute pose or estimates metric scale. The loop resembles render-and-compare refinement [46, 53, 54], but where render-and-compare infers each update by comparing a rendering of the current estimate with the observation and needs an external criterion to decide convergence, GizmoAct predicts both the next edit and when to stop from the rendering alone. We obtain the policy by supervised finetuning of a pretrained VLM on synthetic expert trajectories, followed by reinforcement learning in the same environment used at inference; because training uses assets whose geometry disagrees with the image evidence, initialized at deliberately hard random poses, the policy stays accurate under mismatched assets and coarse initialization.

We evaluate scene parsing and object grounding separately, followed by the complete system they support. For scene-level 3D object detection, the scene graph of the parse step is compared against the prompt-based 3D detectors Boxer [22] and WildDet3D [36] on CA-1M [49] and on R2S, our real-world indoor benchmark for real-to-sim reconstruction. Lucida achieves the highest mean mAP in all four evaluation protocols, improving over Boxer from 0.351 to 0.592 on R2S-Scene under the all-annotation protocol and leading on both CA-1M protocols. For object pose estimation, GizmoAct refines coarse layouts on R2S-Object, CA-1M, and ADT [63], raising ADD-SB@0.05 over the strongest baseline from 57.8% to 83.4% on CA-1M and from 79.2% to 92.0% on R2S-Object, and improving 3D IoU from 0.434 to 0.607 and from 0.500 to 0.719, respectively; a single policy handles initializers with diferent error profiles without retraining. At the system level, scenes composed by Lucida achieve a scene F-Score of 0.924, compared with 0.794 for SAM 3D [67] and 0.351 for SceneGen [59], while improving the object-level F-Score from 0.704 for SAM 3D to 0.736.

Our contributions are summarized as follows:

• We recast composable scene modeling so that no step depends on an input a real cluttered capture cannot provide, deferring precision to a closed-loop final step instead of demanding it from the first.

• We propose a scene-level 3D object detection method that builds a scene graph with per-instance multiview evidence, improving mAP over Boxer from 0.351 to 0.592 on R2S-Scene under the all-annotation protocol.

• We propose GizmoAct, a VLM policy that casts asset placement as multi-turn GUI interaction, aligns the asset by closed-loop gizmo manipulation, and stops autonomously; trained by supervised finetuning and reinforcement learning, it raises strict-alignment success (ADD-SB@0.05) by up to 25.6 percentage points over the strongest baseline while accepting mismatched assets and coarse initial poses.

• We integrate these modules into Lucida and evaluate it on R2S, our real-world real-to-sim benchmark, at three levels—3D object detection, object pose estimation, and scene reconstruction—where the composed scenes achieve a scene F-Score of 0.924, compared with 0.794 for the strongest baseline.

## 2 Method

Lucida converts posed RGB(-D) observations of an indoor scene into an editable, object-level replica: each real instance becomes a complete 3D asset placed by a 9-DoF pose and organized in a scene graph. As shown in Figure 2, the pipeline follows the parse–generate–place order. The parse stage (Sec. 2.1) consolidates the multi-view observations into a scene graph whose nodes carry per-instance evidence bundles ${ \mathcal E } _ { o } ,$ which provide per-object context for the later stages. The generate stage (Sec. 2.2) synthesizes an occlusion-free image from ${ \mathcal { E } } _ { o }$ and lifts it into a 3D asset $A _ { o } .$ . GizmoAct (Sec. 2.3) then places $A _ { o }$ through closed-loop gizmo manipulation, starting from a coarse initial pose from ${ \mathcal { E } } _ { o }$ and refining the pose until it decides alignment is reached; optional postprocessing further enforces scene-level constraints (Sec. 2.4). The design principle throughout is that each stage consumes only what a real capture reliably provides, deferring precision to the closed-loop object placement stage.

![](images/c6e40fb45eb70084cd2d9c6990081499531f3e9b045854855c1267e8a0a9e28e.jpg)  
Figure 2 Overview of our method. Given posed scene observations, the parsing stage constructs an object-centric scene graph with per-object multi-view evidence and coarse 3D estimates. For each object, the multi-view evidence is used to synthesize a complete object-centric image, which is then lifted into a 3D asset. GizmoAct places the asset through closed-loop 9-DoF gizmo manipulation, iteratively refining rotation, translation, and anisotropic scale until it decides that alignment is reached. The placed assets form an editable reconstruction of the captured scene.

## 2.1 Multi-View Object-Centric Scene Parsing

The first module parses multi-view scene observations into an object-centric scene graph that consolidates visual and geometric evidence for both amodal generation and 3D grounding. Given posed RGB-D observations $\mathcal { T } = \{ ( I _ { i } , D _ { i } , K _ { i } , T _ { i } ) \} _ { i = 1 } ^ { N } .$ , where $I _ { i }$ is an RGB image, $D _ { i }$ a depth map, $K _ { i }$ the camera intrinsic matrix, and $T _ { i }$ the camera pose, Lucida builds a scene graph $G = ( V , E )$ . For each object $^ { O , }$ the graph stores a multi-view evidence bundle

$$
\mathcal { E } _ { o } = \{ \mathcal { V } _ { o } , \mathcal { M } _ { o } , \mathcal { P } _ { o } , b _ { o } , c _ { o } \} ,\tag{1}
$$

where $\gamma _ { o }$ contains multi-view observations of the object, $\mathcal { M } _ { o }$ contains the associated masks or boxes, $\mathcal { P } _ { o }$ contains partial point-cloud observations, $b _ { o }$ is a representative 3D box, and $c _ { o }$ is a category name or referring description. Edges in E record spatial relations between objects, such as support, containment, and adjacency.

Geometry-Aware Keyframe Selection and Object Discovery. Adjacent frames in a captured sequence are often highly overlapping, making object discovery on every frame redundant. We first select a sparse set of keyframes from the input sequence. For each pair of frames $i , j ,$ , we compute a similarity $s ( i , j )$ from covisibility and temporal separation. Covisibility measures geometric redundancy, while the temporal term reduces the influence of distant frames so that keyframes remain distributed over a long sequence:

$$
\begin{array} { r l } & { c _ { i  j } = \frac { | \mathcal { C } _ { i  j } | } { | \mathcal { P } _ { i } | } , \quad \mathcal { C } _ { i  j } = \{ p \in \mathcal { P } _ { i } : | D _ { i  j } ( p ) - D _ { j } ( \pi _ { j } ( p ) ) | \leq \delta \} , } \\ & { s ( i , j ) = \frac { 2 c _ { i  j } c _ { j  i } } { \operatorname* { m a x } ( c _ { i  j } + c _ { j  i } , \epsilon ) } ( \lambda + ( 1 - \lambda ) \exp ( - \frac { | i - j | } { \tau } ) ) , } \end{array}\tag{2}
$$

where $\mathcal { P } _ { i }$ contains sampled 3D points from frame $i , \ \pi _ { j } ( p )$ projects $p$ into frame $j , \ D _ { i \to j } ( p )$ denotes its projected depth, $c _ { i \to j }$ denotes directional covisibility, λ and $\tau$ control the temporal weighting. We traverse the sequence in temporal order and retain a frame when its similarity to every selected keyframe falls below a threshold.

On the selected keyframes, a VLM identifies object instances and a 3D detector [22] predicts a box for each object. We group observations across keyframes based on semantic and geometric consistency in the common 3D coordinate frame. The grouped observations and their per-frame 3D boxes form the initial evidence for each candidate object.

Object-Centric Full-Sequence Evidence Consolidation. Keyframes are selected according to scenelevel covisibility and therefore do not guarantee the suficient observations for every object. To retrieve additional observations for each object from the full sequence, we first select a representative 3D box from its geometrically consistent 3D boxes across keyframes. If the 3D boxes from the keyframes are not suficiently consistent, we propagate the object to nearby frames using video segmentation and tracking [64], lift the additional observations into 3D, and repeat the selection with the expanded evidence.

We project the representative box into the input frames and use the projected 2D regions as prompts for per-frame 3D box estimation. We validate these observations against the representative box and observed geometry, retaining the consistent ones as the full-sequence evidence for the object.

Relation-Aware Scene Refinement. Processing objects independently can leave scene-level errors such as incorrect grouping and spatial inconsistencies. We compare appearance and geometry across objects to correct erroneous merges and splits. We also infer spatial relations between objects. For a resting object without valid support in the current graph, we search additional views for the missing supporting object and add it after multi-view geometric validation. The inferred relations also guide the correction of spatial inconsistencies between objects and their surroundings.

The resulting per-object evidence guides amodal asset generation and provides object references and coarse spatial initialization for GizmoAct placement.

## 2.2 Amodal Object Asset Generation

The second module converts each multi-view evidence bundle into a complete standalone object asset. In real captures, occlusion and noisy depth often leave the per-object point-cloud observations incomplete and unreliable for direct 3D completion. In contrast, multi-view RGB observations reveal complementary parts of the object across viewpoints. We therefore first use the multi-view visual evidence to synthesize a complete object-centric image and then lift the completed object into 3D. Specifically, we select a small set of reliable and complementary views from ${ \mathcal E } _ { o } .$ , favoring observations with clear object visibility and geometric agreement with the representative 3D box while encouraging diversity in viewpoint and occlusion. To organize the selected views for image editing, we use Set-of-Mark prompting [96] to indicate the target in each view, while a VLM organizes the selected views into an anchor and references and produces an editing instruction for amodal completion. The image-editing model uses these views and the instruction to synthesize a complete, isolated object image. Finally, an image-to-3D model [28] converts the completed image into an object asset $A _ { o } ,$ which is coarsely initialized in the scene using $b _ { o }$ and subsequently refined by GizmoAct in position, orientation, and anisotropic scale.

## 2.3 Agentic 3D Grounding with GizmoAct

## 2.3.1 Formulation: 3D Grounding as Multi-turn GUI Interaction

3D grounding is commonly posed as localizing a referred object with a 3D box, yet the box is only one proxy for the spatial state of that object: a 3D model instead yields object pose estimation, a coordinate frame yields orientation estimation, and a planar primitive yields layout estimation. What these tasks share is the operation of aligning a 3D element with the available image and geometry evidence, much as a modeler places and adjusts an element against a reference in a 3D modeling tool. We therefore reformulate 3D grounding as a multi-turn GUI interaction: the current state is rendered into a graphical observation, a vision-language model (VLM) predicts one executable edit, and the edited state is re-rendered for the next turn. The rendering pairs the scene point cloud with the interface elements that make an edit legible, namely the 3D model overlay, its 3D box, and a gizmo exposing the object’s local frame, in which every edit is expressed.

We instantiate this formulation for the 9-DoF object pose estimation required by composable scene modeling. The manipulated element for object o is the 3D asset $A _ { o }$ generated in Sec. 2.2, and its state

$$
\begin{array} { r } { x _ { t } = ( p _ { t } , R _ { t } , s _ { t } ) , \qquad p _ { t } \in \mathbb { R } ^ { 3 } , \quad R _ { t } \in S O ( 3 ) , \quad s _ { t } \in \mathbb { R } _ { + } ^ { 3 } , } \end{array}\tag{3}
$$

collects the object center $p _ { t } ,$ the object-to-world rotation $R _ { t } ,$ and the anisotropic object scale $s _ { t }$ . Everything else comes from the evidence bundle ${ \mathcal { E } } _ { o }$ of Sec. 2.1: the target cue $c _ { o }$ (referring text and position) identifies

## (a) Graphical interface

![](images/66d1d880ce53267a205ae3b7f6934dc895729d83dbd7fe65ba136d7f1c859025.jpg)  
Figure 3 GizmoAct graphical observation and interaction trajectory. (a) The observation at one turn. The instruction block specifies the adjustment target and the referring cue; the main views pair the raw input image with renderings that overlay the asset, its yellow 3D box, and the gizmo showing the object’s local frame (red/green/blue axes), in which all edits are expressed. Semi-transparent green marks the model surface occluded by the scene point cloud. Auxiliary views add multi-view evidence, and orthographic views along the local axes support fine refinement. (b) A rollout in which the policy issues a flawed edit (red cross) and recovers from the resulting state (green checks). During SFT data construction, we simulate such cases through error injection and supervise only the recovery actions.

the object, the reference views and the point cloud $\mathcal { P } _ { o }$ are rendered into the observation of Figure 3(a), and an initial object pose provides $x _ { 0 }$ . A rollout can start from an arbitrary $x _ { 0 } ;$ in Lucida we take it from the coarse layout with arbitrary orientation. One turn then consists of a rendering, an action, and a state update:

$$
o _ { t } = \mathcal { R } ( A _ { o } , x _ { t } , \mathcal { E } _ { o } , v _ { t } ) , \quad a _ { t } \sim \pi _ { \theta } ( \cdot \mid o _ { \le t } , a _ { < t } ) , \quad ( x _ { t + 1 } , v _ { t + 1 } ) = F ( x _ { t } , a _ { t } ) ,\tag{4}
$$

where $v _ { t }$ selects the rendering mode of the turn, which an action can change without touching the object state. The policy $\pi _ { \theta }$ is a VLM: it reads the renderings as images and emits $a _ { t }$ as text, so a rollout is a single multi-turn conversation. As Figure 3(b) shows, it sees the rendered consequence of its own edits, which might be flawed, and keeps updating until it considers the proxy aligned and predicts a stop action.

## 2.3.2 Graphical Observation Interface

Graphical elements. Beyond the image and point-cloud renderings, an observation contains three elements. The rendered 3D model indicates the current object pose. The 3D box provides clearer object boundary information and helps the VLM judge the object size. Most importantly, the 3D gizmo defines the local coordinate frame of the object, and all pose updates are predicted relative to this frame.

Viewpoints. We provide three complementary viewpoints. Beyond the main views, the auxiliary views supply additional multi-view evidence, and the orthographic views are rendered along the local axes under the current object pose, which helps fine pose refinement. The orthographic views also act as pseudo multi-view observations for a single-view input, resolving the scale-depth ambiguity that leaves translation and scale under-determined from a single viewpoint.

Occlusion cue. A single point-cloud rendering often leaves the depth relation between the 3D model and the point cloud ambiguous. We alleviate this by overlaying an additional semi-transparent green layer on the occluded regions of the model, namely the model surface that falls behind the point cloud from the current viewpoint.

## 2.3.3 GizmoAct Action Space

The core action space of GizmoAct contains only two actions: update\_pose, which edits the object pose, and stop, which ends the episode. Auxiliary actions that further facilitate pose updates can also be added.

Continuous pose update. update\_pose rests on two design choices. First, the gizmo defines the local frame of the object, and every update is an incremental edit expressed in that frame, so the policy never predicts an absolute pose, which is more ambiguous to predict from images. Second, translation and scale are predicted relative to the current object size rather than as metric quantities, which removes the dependence on metric scale estimation. Concretely, given a ZXY Euler-angle delta $\delta _ { r } = ( \delta _ { z } , \delta _ { x } , \delta _ { y } ) \in \mathbb { R } ^ { 3 }$ in degrees, a translation delta $\delta _ { p } \in \mathbb { R } ^ { 3 }$ , and a relative scale delta $\delta _ { s } \in \mathbb { R } ^ { 3 }$ ,

$$
R _ { t + 1 } = R _ { t } R _ { \mathrm { Z X Y } } ( \delta _ { r } ) , \qquad s _ { t + 1 } = s _ { t } \odot ( { \bf 1 } + \delta _ { s } ) , \qquad p _ { t + 1 } = p _ { t } + R _ { t + 1 } ( \delta _ { p } \odot s _ { t } ) .\tag{5}
$$

Rotation is applied first, so $\delta _ { p }$ acts in the frame that the next gizmo will display, and both $\delta _ { p }$ and $\delta _ { s }$ are measured in units of the current size $s _ { t }$ . The VLM predicts stop when it considers the current object pose accurate enough.

Extended actions. A large initial rotation error is dificult to resolve with update\_pose alone. The main views expose only the object surface facing the input camera, which rarely reveals how a generated asset is oriented, and correcting a large rotation residual with Euler deltas risks gimbal lock. GizmoAct therefore adds two actions for coarse rotation, illustrated in Figure 4. switch\_obs leaves $x _ { t }$ untouched and re-renders the asset along six signed axes, exposing the sides the main views hide. From these views, the policy emits an axis permutation with permute\_axis, which names the target x and z axes with signed source axes $e _ { x } , e _ { z } \in \{ \pm \hat { x } , \pm \hat { y } , \pm \hat { z } \}$ satisfying $e _ { x } ^ { \top } e _ { z } = 0$ and derives y by the right-hand rule:

$$
M ( e _ { x } , e _ { z } ) = [ e _ { x } , e _ { z } \times e _ { x } , e _ { z } ] , \qquad R _ { t + 1 } = R _ { t } M ( e _ { x } , e _ { z } ) .\tag{6}
$$

The $6 \times 4$ valid $( e _ { x } , e _ { z } )$ pairs cover all 24 axis-aligned reorientations, so a coarse rotation costs a single action and update\_pose is left with the residual.

Executable syntax. Every action is written as one XML-style tag whose body is a JSON payload, and the VLM emits exactly one tag per turn:

```html
t=1 <switch_obs>permute_axis</switch_obs>
t=2 <permute_axis>{"x":"-y","z":"x"}</permute_axis>
t=3 <update_pose>{"rotate":{"z":8.25,"x":-2.50,"y":3.10,"order":"ZXY"},
"translate":{"x":0.10,"y":-0.04,"z":0.02},
"scale":{"x":0.05,"y":-0.03,"z":0.02}}</update_pose>
t=T <stop>{}</stop>
```

An update\_pose payload may carry any non-empty subset of the rotation, translation, and scale fields.   
Malformed payloads, empty bodies, and unknown action names are rejected by the environment.

![](images/7d8595bd4ffbd5fb0a57e2005aa24e870128fc00218b80c98b2b4c631cde730f.jpg)  
Figure 4 Extended actions for coarse rotation. When the initial rotation error is large (step 1), the policy first issues switch\_obs, which changes only the rendering mode: the next observation (step 2, enlarged in the bottom panel) shows the original-view pair together with six orthographic renderings of the asset along its signed local axes, exposing the sides the main views hide. From these views the policy predicts a single permute\_axis action, which selects one of the 24 axis-aligned reorientations and removes the dominant rotation residual in one step. The remaining error is repaired by update\_pose (step 3) before the policy predicts stop (step 4). The dashed arrow marks the alternative of predicting the whole correction with a single regular update\_pose, skipping the permutation, which is less accurate.

## 2.3.4 Supervised Finetuning on Synthetic Trajectories

We finetune a pretrained VLM into the GizmoAct policy on synthetic expert trajectories. Each training example pairs the evidence bundle $\mathcal { E } _ { o }$ with a generated or provided object asset, an initial pose $x _ { 0 }$ drawn from a perturbed pose distribution, and a hidden target pose $x ^ { \star }$ available only to the trajectory generator. Rolling the example out in the same executable environment used at inference time yields the rendered observations and XML actions, which we pack as a multi-turn conversation.

Data mixture. We combine three complementary data sources. Populated 3D-FRONT places Mesa-Task [31] tabletop layouts onto support surfaces in 3D-FRONT [25] scenes, giving clean indoor geometry and dense tabletop arrangements. FoundationPose [80] scatters retextured assets on textured planes under randomized lighting, stressing arbitrary orientation, heavy occlusion, and appearance mismatch between the asset and the image. CA-1M Objects samples long-tail categories from the train split of CA-1M [49], following its original train/val partition, generates assets with SAM 3D [67], and manually aligns them to CA-1M LiDAR, contributing noisy real point clouds and generated assets whose geometry ranges from accurate to visibly mismatched with the observation.

Privileged expert policy. Trajectories follow a preferred repair order: rotation first, then translation and scale by residual magnitude, with one action allowed to cover several dimensions. The generator has oracle access to $x ^ { \star }$ and tracks rotation error, translation error normalized by the target diagonal, and log-scale error. A dimension stays active while its residual exceeds the two-decimal precision of the interface, and the expert emits stop once none is active. Before any update\_pose turn, it checks for a coarse rotation fix:

when a signed-axis permutation removes a large and unambiguous part of the rotation residual, it requests the permute\_axis view with switch\_obs and applies the best of the 24 permutations. The rest is repaired by update\_pose actions that each fully cancel the error on a sampled subset of the active dimensions, and sampling the subset and the turn count diversifies trajectory length.

Error injection and recovery. Expert trajectories alone never show the policy how to recover from its own errors. Following the noise-injected demonstrations of DART [48], we inject errors into the expert rollout, illustrated in Figure 3(b): corrupted update\_pose commands (under-correction, overshoot, wrong axis, etc.) and wrong or skipped axis permutations. A candidate injected error $\tilde { a } _ { t }$ is kept only if a clean expert, simulated from the state it produces, still reaches the target within the remaining turn budget. The kept action is executed and stays in the context as history, and the expert then demonstrates recovery from the resulting state. A trajectory carries a few such injections, spread across turns and error dimensions. Formally, for target token $y _ { t , j } ^ { \star }$ under history $h _ { t } = ( o _ { \leq t } , a _ { < t } )$ , the SFT objective is

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { \xi \sim \mathcal { D } _ { \mathrm { S F T } } } \sum _ { t } \sum _ { j } w _ { t } m _ { t , j } \log \pi _ { \theta } \left( y _ { t , j } ^ { \star } \mid h _ { t } , y _ { t , < j } ^ { \star } \right) ,\tag{7}
$$

where $m _ { t , j }$ selects the action tokens emitted at turn t and $w _ { t } \in \{ 0 , 1 \}$ masks out the injected errors.

## 2.3.5 RL Training

The finetuned VLM policy is limited in three ways. Its rollout success rate is still low on dificult instances; it recovers poorly from the states its own errors produce, because injected errors are scripted and cover only a part of the on-policy contexts the policy actually reaches; and it falls short of adapting the correction schedule to what it can perceive. We therefore continue training with GRPO [70] on online rollouts, in the same executable environment used at inference. We elaborate on the reward design and training scheme below.

Reward design. A rollout is scored only at its final state, and that scalar is broadcast to every action token of the trajectory. We score two axes: the generalized 3D IoU $g$ between the predicted and the target oriented 3D box, which serves as a proxy for translation and scale error, and the geodesic rotation error e<sub>R</sub>. We do not score ADD-S [88], which assumes symmetry annotations that cover only part of our data; ADD-SB drops the reliance but is blind to texture, hence to orientations that only appearance distinguishes. Instead of mapping the two axes to a continuous score, we quantize each into four levels, bad, coarse, success, and excellent, with boundaries at $g = 0 . 7 5 , 0 . 8 5 , 0 . 9 2 5$ and $e _ { R } = 3 0 ^ { \circ } , 1 0 ^ { \circ } , 5 ^ { \circ }$ . Writing $\ell _ { g } , \ell _ { R } \in \{ 0 , 1 , 2 , 3 \}$ for the two levels and $S = \mathbf { 1 } [ \ell _ { g } \geq 2 ] \cdot \mathbf { 1 } [ \ell _ { R } \geq 2 ]$ for joint success, the reward of a trajectory ξ with final action $a _ { T }$ is

$$
r ( \xi ) = \frac S 8 \left( 4 + 2 \cdot { \bf 1 } [ \ell _ { g } = 3 ] + 2 \cdot { \bf 1 } [ \ell _ { R } = 3 ] + { \bf 1 } [ a _ { T } = \tt { s t o p } ] \right) .\tag{8}
$$

A trajectory earns nothing until both axes reach the success band, gains extra reward for each axis that reaches excellent, and receives a final bonus for stopping once aligned. Rollouts that emit an invalid action are scored zero. Since $e _ { R }$ penalizes symmetry-equivalent poses, we exclude symmetric objects from RL training; scoring them properly would require symmetry annotations in the reward. Quantization keeps the group comparison informative and avoids noisy advantages introduced by continuous scores.

Dynamic sampling. We follow DAPO [107] and use dynamic sampling to keep each batch informative. For each prompt we draw $K = 8$ rollouts at temperature 0.7 and reject the group in two cases: when any rollout fails to complete, and when the K terminal rewards all fall on the same level. The second case covers uniformly failed, uniformly solved, and tied groups, none of which carries a ranking signal. Rejected groups are discarded and sampling continues until the batch is filled.

Policy optimization. Apart from the reward and the group filter above, we follow an established GRPO training setup. We adopt the token-level clipped objective of DAPO [107] with asymmetric clipping and dual clip. We remove the group standard-deviation normalization of the advantage, as in Dr.GRPO [57]. We include truncated importance sampling (TIS) [101] to compensate for the rollout–training mismatch. We disable KL penalty in optimization. Each iteration collects at least 40 prompts and 320 trajectories, slightly more when dynamic sampling refills the batch, and every collected batch is optimized for two epochs.

Office#1, 212 frames, 5m×8m  
![](images/f2fe7e0a81f79fcb8e3ea07fb8014727f41ddd4ef20718c88c9ed8c8295c7aac.jpg)  
Figure 5 Gallery of composed scenes. Four real captures reconstructed by Lucida, annotated with the number of captured frames and the room extent. For each scene, the left panel shows the composed object-centric scene with the capture trajectory and camera frusta overlaid; the right panels pair captured frames (top row) with renderings of the reconstructed scene from the same camera poses (bottom row). Every object is an individual complete asset, so the composed scenes are directly editable and simulation-ready.

## 2.4 Scene Composition and Postprocessing

After individual objects are generated and grounded, Lucida composes them into a full indoor scene. The base output is an object-centric scene graph with object assets, poses, scales, categories, and spatial relations. Optional postprocessing can refine support relations, collision, contact consistency, and placement plausibility. These operations are intentionally separated from GizmoAct: the agent focuses on grounding individual editable components, while postprocessing handles scene-level consistency constraints that are easier to express as rules, physics checks, or global verification.

## 3 Experiments

We evaluate Lucida at three levels. Section 3.1 evaluates scene-level 3D object detection, measuring object recovery and localization by the scene-parsing module. Section 3.2 evaluates object pose estimation, measuring the alignment accuracy of GizmoAct independently of upstream detection. Section 3.3 evaluates the

complete system through scene reconstruction, jointly measuring reconstructed object geometry and final scene layout. Section 3.4 ablates the scene-parsing components and GizmoAct training strategies.

## 3.1 Scene-Level 3D Object Detection

This experiment evaluates the scene-level 3D object boxes recovered by the multi-view object-centric scene parsing module in Sec. 2.1. It measures whether the parser recovers the target object instances and localizes them with 3D boxes before downstream asset generation and GizmoAct pose refinement. We compare its scene-level output with the per-scene predictions produced by the baseline pipelines.

Baselines. We compare against Boxer [22] and WildDet3D [36], which recover 3D object boxes from image observations and external object prompts. To control which objects are queried, all methods receive prompts for the same target instances. We evaluate two prompt-frame settings. The all setting provides prompts on every annotated frame, whereas the key setting provides them only on the geometry-aware keyframes selected by our method. We evaluate Boxer in both settings and WildDet3D in the key setting. Both baselines use the ofline fusion implementation in Boxer. Our method starts from the keyframe prompts and then runs the complete scene-parsing pipeline of Sec. 2.1, including object-centric full-sequence evidence consolidation and relation-aware scene refinement.

Datasets and metrics. We evaluate on CA-1M [49] and R2S-Scene. CA-1M provides independently laser-scanned geometry and near-exhaustive object annotations that include long-tail categories and small and thin instances across captured sequences. R2S-Scene, the scene-level split of our real-world real-to-sim benchmark, contains verified 3D boxes and poses for the objects included in scene reconstruction. The \_all protocol evaluates all annotated ground-truth boxes. The \_filter protocol evaluates ground-truth objects recovered by at least one compared method, excluding objects missed by every method. For each scene, we compute class-agnostic AP at 3D IoU thresholds from 0.05 to 0.50 in increments of 0.05, average over thresholds, and report the mean across scenes.

Results. Table 1 shows that Lucida achieves the highest mAP in all four protocols. On CA-1M, it improves the strongest baseline from 0.171 to 0.180 under the \_all protocol and from 0.373 to 0.390 under the \_filter protocol. On R2S-Scene, mAP increases from 0.351 to 0.592 under the \_all protocol and from 0.355 to 0.597 under the \_filter protocol. Although Boxer improves when given prompts on all frames, our method, initialized only from keyframe prompts, still performs better while using the full sequence for instance propagation, multi-view validation, and 3D box refinement. This comparison highlights the benefit of organizing full-sequence observations around object hypotheses rather than relying only on direct fusion of independently predicted per-frame boxes. Sec. 3.4 further evaluates object-centric full-sequence evidence consolidation and relation-aware scene refinement.

## 3.2 Layout Refinement and Object Pose Estimation

Setup. We evaluate GizmoAct independently of the upstream scene-level detection module so that global layout errors do not confound pose refinement. Each task provides target cues (2D bounding box and object category), posed RGB-D inputs, 3D object model, and a coarse object pose. GizmoAct supports Boxer [22] initialization, a depth-and-mask initialization following Any6D [51], and initialization from SAM 3D [67]. The main comparison uses Boxer initialization with a policy whose RL stage is trained on Boxer-initialized poses (this specialization is ablated in Sec. 3.4), and reports a single-view variant and a multi-view variant that receives at most four valid views. Both variants refine the 9-DoF object pose for at most 12 steps.

Datasets. We evaluate on CA-1M [49], Aria Digital Twin (ADT) [63], and the object split of our R2S benchmark, denoted R2S-Object. The CA-1M subset contains 199 objects from 125 categories in the CA-1M validation set. We associate each object across views, estimate and manually filter its masks with SAM 3 [14], generate its 3D model with SAM 3D [67], and manually annotate a 9-DoF pose that aligns the model with the LiDAR point cloud. We form the evaluation depth maps by completing the projected FARO LiDAR points with LingBot-Depth [74]. R2S-Object contains 125 objects from 64 categories captured in real indoor scenes.

Table 1 Scene-level 3D object detection on CA-1M and R2S-Scene. Per-scene entries report class-agnostic mAP, and Mean averages across scenes. Prompt frames all/key denotes prompts on all annotated frames or selected keyframes; \_all/\_filter denotes evaluation on all annotations or only objects recovered by at least one method. A dash denotes no valid prediction. Best mean in bold.
<table><tr><td rowspan="2">Method</td><td colspan="10">01</td><td rowspan="2"></td><td rowspan="2">10 Mean</td></tr><tr><td>Prompt frames</td><td></td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td>07 08</td><td>09</td><td></td></tr><tr><td>Boxer</td><td>all</td><td>0.151</td><td>0.142</td><td>0.224</td><td>0.134</td><td>0.241</td><td>0.106</td><td>0.170</td><td>0.302</td><td>0.113</td><td>0.122</td><td>0.171</td></tr><tr><td>Boxer</td><td>key</td><td>0.037</td><td>0.027</td><td>0.138</td><td>0.057</td><td>0.059</td><td>0.019</td><td>0.095</td><td>0.122</td><td></td><td>0.053</td><td>0.067</td></tr><tr><td>WildDet3D Ours</td><td>key</td><td>0.051</td><td>0.014</td><td>0.101</td><td>0.043</td><td>0.113</td><td>0.035</td><td>0.025</td><td>0.088</td><td>0.036</td><td>0.065</td><td>0.057</td></tr><tr><td></td><td>key</td><td>0.168</td><td>0.148</td><td>0.268</td><td>0.138</td><td>0.266</td><td>0.109</td><td>0.170</td><td>0.319</td><td>0.105</td><td>0.110</td><td>0.180</td></tr><tr><td colspan="10">CA-1M_filter</td><td colspan="3"></td></tr><tr><td>Method</td><td>Prompt frames</td><td>01</td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td>07</td><td>08</td><td>09</td><td>10</td><td>Mean</td></tr><tr><td>Boxer</td><td>all</td><td colspan="7">0.312 0.344</td><td>0.404</td><td>0.404</td><td>0.314</td><td>0.373</td></tr><tr><td>Boxer</td><td>key</td><td colspan="7">0.064 0.051</td><td>0.156</td><td>0.119</td><td>0.131</td><td>0.129</td></tr><tr><td>WildDet3D Ours</td><td>key key</td><td colspan="7">0.092 0.024</td><td>0.114</td><td>0.119</td><td>0.164</td><td>0.118</td></tr><tr><td></td><td></td><td colspan="7">0.349 0.358</td><td>0.428 0.426</td><td>0.378</td><td>0.279</td><td>0.390</td></tr><tr><td colspan="10">R2S-Scene_all</td><td></td><td></td><td>Mean</td></tr><tr><td>Method</td><td colspan="10">Prompt frames</td><td>09</td><td></td></tr><tr><td>Boxer</td><td>all</td><td>0.458</td><td>0.728</td><td>0.240</td><td>0.308</td><td>0.333</td><td>0.090</td><td>0.309</td><td></td><td>0.386</td><td>0.306</td><td>0.351</td></tr><tr><td>Boxer</td><td>key</td><td>0.172</td><td>0.524</td><td>0.048</td><td>0.078</td><td>0.089</td><td>0.084</td><td>0.059</td><td></td><td>0.122</td><td>0.132</td><td>0.145</td></tr><tr><td>WildDet3D</td><td>key</td><td>0.148</td><td>0.309</td><td>0.080</td><td></td><td>0.080</td><td>0.024</td><td></td><td>0.123</td><td>0.193</td><td>0.043</td><td>0.125</td></tr><tr><td>Ours</td><td>key</td><td>0.613</td><td>0.959</td><td>0.507</td><td>0.379</td><td>0.523</td><td></td><td>0.370</td><td>0.596</td><td>0.781</td><td>0.603</td><td>0.592</td></tr><tr><td colspan="10">R2S-Scene_filter</td><td></td><td></td><td></td></tr><tr><td>Method</td><td>Prompt frames</td><td>01</td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td></td><td>07</td><td>08</td><td>09</td><td>Mean</td></tr><tr><td>Boxer</td><td>all</td><td>0.471</td><td>0.728</td><td>0.241</td><td>0.311</td><td>0.341</td><td>0.090</td><td></td><td>0.309</td><td>0.401</td><td>0.306</td><td>0.355</td></tr><tr><td>Boxer</td><td>key</td><td>0.181</td><td>0.524</td><td>0.048</td><td>0.083</td><td>0.091</td><td></td><td>0.084</td><td>0.059</td><td>0.124</td><td>0.132</td><td>0.147</td></tr><tr><td>WildDet3D</td><td>key</td><td>0.153</td><td>0.309</td><td>0.082</td><td></td><td></td><td>0.082</td><td>0.024</td><td>0.123</td><td>0.202</td><td>0.043</td><td>0.127</td></tr><tr><td>Ours</td><td>key</td><td>0.632</td><td>0.959</td><td>0.512</td><td>0.383</td><td></td><td>0.537</td><td>0.370</td><td>0.593</td><td>0.788</td><td>0.603</td><td>0.597</td></tr></table>

We initialize their layouts with the scene-level detection pipeline from Sec. 3.1, generate object models with Seed3D 2.0 [28], and apply the same mask filtering and manual pose annotation protocol as CA-1M. Pi3X [78] provides the R2S-Object depth estimates. From ADT [63], we select 60 objects with designer-authored meshes and accurate object poses; we render their depth maps from the ground-truth scene geometry.

Metrics. Following SAM 3D [67] and RecGen [108], we measure the bidirectional average nearest-neighbor distance between the predicted and ground-truth posed surfaces. We report this distance as ADD-SB in meters. ADD-SB@0.1 and ADD-SB@0.05 are the percentages of predictions whose ADD-SB falls below 10% and 5% of the ground-truth object diameter. We also report 3D IoU between the oriented bounding boxes of the predicted and ground-truth posed objects.

Baselines. The comparison covers two method families. Any6D [51] estimates the scale and 6-DoF pose of an independently generated object model from a single RGB-D observation. Within each split, Any6D and GizmoAct receive the same generated asset. SAM 3D and RecGen jointly estimate object shape and 9-DoF pose from RGB or RGB-D observations and a target mask. We evaluate the single-view and two-view variants of RecGen, together with the single-view and max-4-view variants of GizmoAct.

Results. Table 2 shows that GizmoAct improves strict alignment and oriented-box overlap on all three datasets. On R2S-Object, the max-4-view variant reduces ADD-SB from 0.038 for the strongest baseline to

Table 2 Object pose estimation on R2S-Object, CA-1M, and ADT. ADD-SB and its thresholded success rates measure surface alignment, and 3D IoU measures oriented-box overlap. Any6D and GizmoAct receive the same generated asset for each sample, whereas SAM 3D and RecGen jointly predict shape and pose; GizmoAct refines from Boxer initialization, and the max-4-view variant uses up to four valid views. GizmoAct improves strict alignment (ADD-SB@0.05) and 3D IoU over all baselines on the three datasets. Best per dataset in bold.
<table><tr><td>Dataset</td><td>Method</td><td>ADD-SB↓</td><td>ADD-SB@0.1 ↑</td><td>ADD-SB@0.05 ↑</td><td>3D IoU ↑</td></tr><tr><td rowspan="6">R2S-Object</td><td>Any6D</td><td>0.086</td><td>76.6%</td><td>46.8%</td><td>0.388</td></tr><tr><td>SAM 3D</td><td>0.050</td><td>92.8%</td><td>61.6%</td><td>0.482</td></tr><tr><td>RecGen (1 view)</td><td>0.040</td><td>98.4%</td><td>79.2%</td><td>0.499</td></tr><tr><td>RecGen (2 views)</td><td>0.038</td><td>96.8%</td><td>77.6%</td><td>0.500</td></tr><tr><td>GizmoAct (1 view)</td><td>0.019</td><td>98.4%</td><td>88.0%</td><td>0.685</td></tr><tr><td>GizmoAct (max 4 views)</td><td>0.017</td><td>99.2%</td><td>92.0%</td><td>0.719</td></tr><tr><td rowspan="6">CA-1M</td><td>Any6D</td><td>0.116</td><td>67.3%</td><td>42.2%</td><td>0.355</td></tr><tr><td>SAM 3D</td><td>0.062</td><td>86.4%</td><td>56.3%</td><td>0.434</td></tr><tr><td>RecGen (1 view)</td><td>0.061</td><td>87.9%</td><td>57.8%</td><td>0.391</td></tr><tr><td>RecGen (2 views)</td><td>0.046</td><td>88.4%</td><td>57.3%</td><td>0.400</td></tr><tr><td>GizmoAct (1 view)</td><td>0.023</td><td>94.0%</td><td>81.9%</td><td>0.600</td></tr><tr><td>GizmoAct (max 4 views)</td><td>0.021</td><td>96.0%</td><td>83.4%</td><td>0.607</td></tr><tr><td rowspan="6">ADT</td><td>Any6D</td><td>0.047</td><td>85.0%</td><td>66.7%</td><td>0.520</td></tr><tr><td>SAM 3D</td><td>0.063</td><td>90.0%</td><td>41.7%</td><td>0.404</td></tr><tr><td>RecGen (1 view)</td><td>0.044</td><td>96.7%</td><td>71.7%</td><td>0.462</td></tr><tr><td>RecGen (2 views)</td><td>0.039</td><td>95.0%</td><td>70.0%</td><td>0.444</td></tr><tr><td>GizmoAct (1 view)</td><td>0.020</td><td>98.3%</td><td>86.7%</td><td>0.664</td></tr><tr><td>GizmoAct (max 4 views)</td><td>0.021</td><td>91.7%</td><td>90.0%</td><td>0.670</td></tr></table>

Table 3 Robustness to pose initialization. All rows evaluate the same GizmoAct policy, trained with randomly perturbed initial poses, using at most four views and 12 refinement steps; only the initial pose difers. Any6D\* denotes the depth-and-mask initialization from Any6D without FoundationPose prediction or refinement. Boxer initialization works best with the estimated depth of R2S-Object and CA-1M, whereas the exact geometry of ADT favors the Any6D\* and SAM 3D initializers.
<table><tr><td></td><td colspan="3">R2S-Object</td><td colspan="3">CA-1M</td><td colspan="3">ADT</td></tr><tr><td>Initialization</td><td>ADD-SB ↓</td><td>ADD-SB@0.05↑</td><td>3D IoU ↑</td><td>ADD-SB ↓</td><td>ADD-SB@0.05 ↑</td><td>3D IoU ↑</td><td>ADD-SB ↓</td><td>ADD-SB@0.05 ↑</td><td>3D IoU ↑</td></tr><tr><td>Boxer</td><td>0.022</td><td>90.4%</td><td>0.678</td><td>0.026</td><td>77.9%</td><td>0.568</td><td>0.025</td><td>81.7%</td><td>0.626</td></tr><tr><td>Any6D*</td><td>0.030</td><td>84.8%</td><td>0.612</td><td>0.028</td><td>73.9%</td><td>0.518</td><td>0.016</td><td>93.3%</td><td>0.641</td></tr><tr><td>SAM 3D</td><td>0.025</td><td>91.2%</td><td>0.633</td><td>0.028</td><td>77.4%</td><td>0.542</td><td>0.022</td><td>95.0%</td><td>0.678</td></tr></table>

0.017, raises ADD-SB@0.05 from 79.2% to 92.0%, and improves 3D IoU from 0.500 to 0.719. On CA-1M, it reduces ADD-SB from 0.046 to 0.021 while increasing ADD-SB@0.05 from 57.8% to 83.4% and 3D IoU from 0.434 to 0.607. The ADT results separate the benefits of the two input settings: the single-view variant obtains the lowest ADD-SB (0.020) and the highest ADD-SB@0.1 (98.3%), while the max-4-view variant gives the best ADD-SB@0.05 (90.0%) and 3D IoU (0.670).

Robustness to pose initialization. We test whether one GizmoAct policy adapts to diferent pose initializations without retraining. This experiment uses the general policy trained with randomly perturbed initial poses, rather than the Boxer-specialized policy of the main comparison; it receives at most four views for up to 12 refinement steps. Boxer combines a randomly sampled local rotation with its predicted 3D box, which typically leaves a large rotation error but smaller translation and scale errors. The Any6D-style initializer estimates the initial pose from the target depth and instance mask, so its errors depend on object geometry and viewpoint and can afect rotation, translation, and scale jointly. SAM 3D usually leaves only local resid ual errors. Table 3 shows that Boxer performs best overall with noisy estimated depth on R2S-Object and CA-1M, whereas the exact depth and camera poses of ADT favor the Any6D-style and SAM 3D initializers.

Table 4 Scene reconstruction on R2S-Scene. Scene CD uses squared Euclidean distances in metric scene coordinates and is reported in $\mathrm { m } ^ { 2 } ;$ total CD is the sum of the two directed terms. Object CD is computed after per-object normalization and is dimensionless. Best in bold, second best underlined.
<table><tr><td></td><td colspan="3">Scene CD  $\left( \mathrm { m } ^ { 2 } \right) \downarrow$ </td><td colspan="5"></td></tr><tr><td>Method</td><td>Pred-GT CD</td><td>GT-Pred CD</td><td>CD</td><td>Scene F-Score ↑</td><td>BBox IoU ↑</td><td>Object CD ↓</td><td>Object F-Score ↑</td></tr><tr><td>SceneGen</td><td>0.064</td><td>0.364</td><td>0.428</td><td>0.351</td><td>0.092</td><td>0.071</td><td>0.553</td></tr><tr><td>SAM 3D</td><td>0.012</td><td>0.010</td><td>0.022</td><td>0.794</td><td>0.396</td><td>0.038</td><td>0.704</td></tr><tr><td>Ours</td><td>0.006</td><td>0.004</td><td>0.010</td><td>0.924</td><td>0.495</td><td>0.034</td><td>0.736</td></tr></table>

The policy therefore adapts to heterogeneous initialization errors, although the best initializer still depends on the accuracy of the input geometry.

Qualitative analysis. The comparisons in Figure 6 cover objects with diferent categories, shape complexity, and viewpoint dificulty, where GizmoAct aligns the generated 3D models more closely with the reference objects than the baseline methods. We provide more qualitative results in Figure 10. Figure 7 further shows that GizmoAct accommodates diferent pose initializations, which we quantify above; additional trajectory examples are provided in Figures 11 and 12.

## 3.3 Scene Reconstruction

This experiment evaluates the complete system through scene reconstruction. All methods reconstruct the same selected object instances while independently generating their 3D geometry and recovering their placement from visual observations. Unlike Sec. 3.1, which evaluates object boxes before asset generation, this experiment jointly measures reconstructed object geometry and final scene layout.

Baselines. We compare against SceneGen [59] and SAM 3D [67], which reconstruct compositional objectlevel scenes from real-image observations. SceneGen jointly generates object-level assets and predicts their relative transformations in a single feedforward pass, whereas SAM 3D reconstructs prompted instances independently and then assembles them into a scene. SceneGen and SAM 3D receive the same reference image and instance masks, while our method uses the per-instance multi-view evidence from Sec. 2.1 for amodal generation and closed-loop grounding.

Datasets and metrics. We evaluate on R2S-Scene, the real-world scene split of our R2S benchmark introduced in Sec. 3.1. We sample 10,240 surface points per object and apply a single rigid FilterReg alignment to the complete SceneGen prediction. SAM 3D and our method use the same estimated depth maps to place objects in the metric scene coordinate frame. At the scene level, we report the two directed Chamfer distance (CD) terms; prediction-to-GT CD also penalizes misplaced geometry, while GT-to-prediction CD penalizes missing geometry. Scene-level CD uses squared Euclidean distances in the metric scene coordinate frame and is reported in $\mathrm { m } ^ { 2 } ;$ the symmetric CD is the sum of the two directed terms. We also report the scene F-Score at a distance threshold of 0.1 m and the mean 3D IoU between the axis-aligned bounding boxes of corresponding objects. At the object level, each predicted–ground-truth pair is normalized and aligned independently before computing CD and F-Score, and the resulting scores are averaged over objects. Object CD is therefore dimensionless. The object-level metrics emphasize normalized asset-shape quality, whereas the scene-level metrics additionally reflect object scale, orientation, and relative placement.

Results. As shown in Table 4, our method performs best across all reported metrics. At the scene level, it reduces CD from 0.022 for SAM 3D to 0.010 and increases F-Score from 0.794 to 0.924. BBox IoU further improves from 0.396 to 0.495, indicating more accurate object scale and relative placement in the composed scene. At the object level, independently normalized and aligned assets improve from 0.038 to 0.034 in CD and from 0.704 to 0.736 in F-Score. Both directed scene CD terms are lower than those of SAM 3D, indicating fewer spurious or misplaced surfaces and better ground-truth coverage.

I p I g

Any6D

RecGen

Ours

GT

![](images/fd9757fac2f29521603ee7c5be18f0065a8739c00f023cf0fe3495cbaf065bf4.jpg)

Figure 6 Qualitative comparison of object pose estimation on CA-1M, R2S-Object, and ADT. The leftmost column shows the scene image and a close-up with the target instance boxed. Blue renders the ground-truth model at the ground-truth pose; each method’s model, posed by its prediction, is rendered jointly with this blue reference, so their mutual occlusion—together with the three axis-aligned close-ups—reveals the pose accuracy, and the GT column shows the reference alone. Lucida aligns the generated assets more closely with the observed instances across categories, shape complexity, and viewpoints

![](images/ae26e63430701582bb951cadca28b3009098f77a1ccf0b6c045046b2e70ced6d.jpg)  
Figure 7 GizmoAct pose-refinement trajectories under different initializations. For each dataset we show a reference image with the target instance boxed, and closed-loop rollouts of the same policy initialized by Boxer, Any6D\*, and SAM 3D. Each cell shows the observation at that step (main views on top, orthographic views below); arrow labels denote the predicted action: an axis permutation (P) or the rotation (R), translation (T), and scale (S) components of update\_pose. The final stop action is not shown. Any6D\* uses only the depth-and-mask pose initialization from Any6D, without FoundationPose prediction or refinement. Across object categories and geometries, GizmoAct consistently refines each initialization to an accurate pose, outperforming the baseline estimates.

![](images/83b4bbd784f8240e315509d6015652f66787204d0a47153d00a467ffe5196426.jpg)  
Figure 8 Qualitative scene-level comparison. For each scene, we show the multi-view capture, a scanned reference view, the selected evaluation instances, and reconstructions rendered from the main, side, and top viewpoints. Scene-Gen and SAM 3D receive the same reference image and instance masks, whereas Lucida uses per-instance multi-view evidence. The asterisk marks our reconstruction restricted to the object instances included in the comparison.

In Figure 8, SceneGen exhibits large placement and scale errors, with substantial interpenetration among objects and layouts that deviate from the captured scenes. SAM 3D better recovers the coarse arrangement visible from the reference image. However, the side and top views reveal rotation errors that are less apparent from the reference viewpoint. Our method maintains more consistent object geometry and relative placement across the reference, side, and top views, with fewer visible intersections and closer agreement with the captured scenes.

## 3.4 Ablation Studies

## 3.4.1 Multi-View Object-Centric Scene Parsing

Experimental setup. We ablate the three stages of Sec. 2.1. Scene reconstruction is evaluated on a subset of R2S-Scene. First, we replace geometry-aware keyframe selection with temporal uniform sampling. Second, we disable object-centric full-sequence evidence consolidation, including the propagation of object observations to additional frames and the multi-view validation of their representative 3D estimates. Third, we disable relation-aware scene refinement, including identity consolidation, missing-support recovery, and support-surface correction. All variants otherwise share the same downstream generation, grounding, and evaluation protocol.

Results. Table 5 reports the efects of the three scene-parsing stages. Replacing geometry-aware keyframe selection with uniform sampling reduces mAP from 0.597 to 0.516 and scene F-Score from 0.831 to 0.727, consistent with geometry-aware selection retaining more useful viewpoints. Disabling object-centric fullsequence evidence consolidation reduces mAP to 0.535 and scene F-Score to 0.734. Its higher GT-to-prediction CD (0.025 versus 0.011) is consistent with reduced scene coverage when observations beyond the selected keyframes are not consolidated.

Without relation-aware scene refinement, mAP decreases to 0.526 and scene F-Score to 0.794, while total scene CD changes only slightly. Relation-aware scene refinement therefore improves the recovered 3D boxes and scene F-Score, while having little efect on total scene CD. Overall, all three ablations reduce both detection mAP and scene F-Score, with uniform keyframe sampling producing the largest decrease.

Table 5 Ablation of the scene-parsing components on R2S-Scene. Each variant replaces geometry-aware keyframe selection with temporal uniform sampling or disables object-centric full-sequence evidence consolidation or relation-aware scene refinement. We report scene-level 3D object detection on R2S-Scene and scene reconstruction on an R2S-Scene subset after the same downstream generation and grounding pipeline. Best in bold.
<table><tr><td></td><td colspan="4">Scene-Level 3D Object Detection</td><td colspan="4">Scene Reconstruction</td></tr><tr><td>Method</td><td></td><td></td><td></td><td></td><td colspan="4">Scene CD ↓</td></tr><tr><td></td><td>AP@25↑</td><td>AP@50↑</td><td>mAP ↑</td><td>F-Score ↑</td><td>Pred-GT</td><td>GT-Pred</td><td>CD</td><td>Scene F-Score ↑</td></tr><tr><td>Full Lucida</td><td>0.616</td><td>0.429</td><td>0.597</td><td>0.670</td><td>0.008</td><td>0.011</td><td>0.019</td><td>0.831</td></tr><tr><td>Uniform keyframe selection</td><td>0.547</td><td>0.367</td><td>0.516</td><td>0.635</td><td>0.015</td><td>0.020</td><td>0.035</td><td>0.727</td></tr><tr><td>w/o object-centric full-sequence evidence consolidation</td><td>0.552</td><td>0.394</td><td>0.535</td><td>0.638</td><td>0.011</td><td>0.025</td><td>0.036</td><td>0.734</td></tr><tr><td>w/o relation-aware scene refinement</td><td>0.547</td><td>0.383</td><td>0.526</td><td>0.624</td><td>0.010</td><td>0.010</td><td>0.020</td><td>0.794</td></tr></table>

Table 6 Effect of the RL training pose distribution. The two rows difer only in the initial-pose distribution used during RL training: poses produced by Boxer versus random perturbations. Both policies are evaluated from Boxer initialization, and matching the training distribution to the test-time initializer improves every reported metric. @0.05 denotes the ADD-SB success rate at that threshold; best per dataset in bold.
<table><tr><td></td><td colspan="3">R2S-Object</td><td colspan="3">CA-1M</td><td colspan="3">ADT</td></tr><tr><td>Training poses</td><td>ADD-SB ↓</td><td>@0.05 ↑</td><td>3D IoU ↑</td><td>ADD-SB ↓</td><td>@0.05 ↑</td><td>3D IoU ↑</td><td>ADD-SB ↓</td><td>@0.05 ↑</td><td>3D IoU ↑</td></tr><tr><td>Boxer initialization</td><td>0.017</td><td>92.0%</td><td>0.719</td><td>0.021</td><td>83.4%</td><td>0.607</td><td>0.021</td><td>90.0%</td><td>0.670</td></tr><tr><td>Random perturbation</td><td>0.022</td><td>90.4%</td><td>0.678</td><td>0.026</td><td>77.9%</td><td>0.568</td><td>0.025</td><td>81.7%</td><td>0.626</td></tr></table>

## 3.4.2 GizmoAct Training

Pose distribution for RL training. During RL, we can specialize the policy to the pose errors produced by the initializer used at inference. We train one policy from Boxer-initialized poses and another from random pose perturbations, then evaluate both from Boxer initialization on R2S-Object, CA-1M, and ADT. As shown in Table 6, matching the training distribution to Boxer initialization improves every reported metrics: ADD-SB@0.05 increases by 1.6, 5.5, and 8.3 percentage points on R2S-Object, CA-1M, and ADT, respectively, and 3D IoU improves on all three datasets. These gains show that we can steer the policy toward the error distribution of a target initializer by training on poses produced by that initializer.

RL versus SFT. The gains from RL over SFT are most evident on dificult cases, where the policy must recover from large initialization errors. We therefore construct a hard subset of 110 asymmetric objects from R2S-Object, CA-1M, and ADT, selecting samples with large errors from the Any6D-style depth-and-mask initializer; all policies are evaluated on these large-error initial poses. Beyond the metrics of Sec. 3.2, we report the stricter ADD-SB@0.01, together with the geodesic rotation error (Rot. error) and the fraction of samples below 5<sup>◦</sup> (Rot.@5<sup>◦</sup>); the rotation metrics are unambiguous here because the subset excludes symmetric objects. We compare the RL policy against two SFT baselines. The first is the SFT policy described in Sec. 2.3, which is also used to initialize RL; we denote it GizmoAct (SFT). The second is trained separately without interaction history: given only the current observation, it predicts either a complete pose correction or stop. We denote this variant GizmoAct (SFT, no context). Although the second baseline is trained with single-step supervision, all three policies are evaluated in the same closed-loop setting with at most four views and 12 refinement steps. Table 7 shows that GizmoAct (RL) lowers rotation error to 20.98<sup>◦</sup> from 45.19<sup>◦</sup> and 27.62<sup>◦</sup> for the two SFT baselines, while raising Rot.@5<sup>◦</sup> to 67.3% from 52.7% and 60.9%. Figure 9 shows the same pattern qualitatively: RL continues to correct residual rotation errors that persist in the SFT trajectories, reducing cases in which position and scale are correct but rotation is not.

## 4 Related Work

![](images/6622e352f77e83719dcef4f7bfb8093ae4c5a279fb12249ce2a92ee1230c1e27.jpg)  
Figure 9 Qualitative comparison of GizmoAct training strategies on two difficult examples. Each row shows the closed-loop rollout of the SFT, SFT (no context), or RL policy from the same initialization, evaluated as in Table 7; notation follows Figure 7. The SFT policies stop early with residual rotation errors or drift without converging, whereas RL keeps correcting the remaining rotation before stopping.

Table 7 Comparison of GizmoAct training strategies on the hard subset. SFT (no context) is trained with single-step supervision and no interaction history; all three policies are evaluated in the same closed-loop setting with at most 12 steps. RL improves every metric, with the largest gains on rotation accuracy. Evaluated on a hard subset of 110 asymmetric objects with large Any6D\* initialization errors; best in bold.
<table><tr><td>Method</td><td>ADD-SB↓</td><td>@0.05 ↑</td><td>@0.01 ↑</td><td>3D IoU ↑</td><td>Rot. error (°) ↓</td><td>Rot.@5° ↑</td></tr><tr><td>GizmoAct (SFT)</td><td>0.040</td><td>70.0%</td><td>6.4%</td><td>0.612</td><td>45.19</td><td>52.7%</td></tr><tr><td>GizmoAct (SFT, no context)</td><td>0.035</td><td>86.4%</td><td>0.9%</td><td>0.636</td><td>27.62</td><td>60.9%</td></tr><tr><td>GizmoAct (RL)</td><td>0.032</td><td>87.3%</td><td>11.8%</td><td>0.685</td><td>20.98</td><td>67.3%</td></tr></table>

## 4.1 Composable 3D Scene Modeling

Composable scene modeling recovers a captured scene as separable, editable object assets rather than a monolithic reconstruction. Optimization-based methods attach object structure to a diferentiable-rendering reconstruction [39, 60, 68] through object discovery [105], mask supervision [82, 83, 94], label lifting [43, 71, 113], or Gaussian grouping [104], but explain only the observed pixels, leaving occluded surfaces unrecovered. End-to-end methods map an image to a decomposed scene, from holistic layout–pose–shape regression [35, 61, 109] to multi-instance and part-level latent difusion [17, 37, 55, 59, 67, 92]; synthetic scene supervision limits generalization to real captures, and joint shape–pose prediction couples both errors [67, 108].

Closest to our setting, multi-stage pipelines decompose the task into segmentation, per-object asset acquisition, and placement. Early pipelines retrieve and align CAD models [5, 6, 21, 26, 30, 44, 76], capping fidelity at database coverage; recent ones generate each object with image-to-3D models [4, 114], anchored to sensor depth [3], completed amodally in 3D [23, 84], constrained by inter-object relations and physics [87, 103], and scaled to scan datasets or video [86, 106]. Yet the stages run in a fixed order, each presuming precise inputs—clean masks, unoccluded crops, depth that agrees with the generated asset—so an early error is inherited by every later stage with no mechanism to correct it. Lucida keeps the modular decomposition but relocates precision to the output end: parsing gathers per-instance multi-view evidence, generation conditions on that evidence, and GizmoAct closes the loop on placement, absorbing mismatched assets and coarse initializations—so upstream stages need only be approximately right on cluttered real captures.

## 4.2 Scene-Level 3D Object Detection

Scene-level 3D object detection localizes every object instance of a captured scene as an oriented 3D box in a single de-duplicated inventory, the output of the parse step in Lucida. Monocular detectors predict boxes from a single frame at growing scale and vocabulary [11, 36, 49, 102, 110], but remain fragile to occlusion and viewpoint and give no instance persistence across views. Multi-view methods either pool posed views into a scene volume [66, 89] or—closest to our setting—detect per view and fuse boxes afterwards: Boxer [22] merges per-view lifts with rule-based geometric fusion, BoxFusion [47] fuses frozen detections in real time, and Rooms from Motion [50] trains an object matcher that jointly recovers tracks and camera poses.

A complementary line associates 2D observations into object-level maps, spanning object-centric SLAM [40, 58], multi-view lifting of foundation-model masks [73, 91, 93, 100], and open-vocabulary scene graphs [29, 81]. Across both lines, instances are associated by geometric overlap and feature similarity, which cluttered captures undermine: repeated near-identical instances risk being merged or duplicated, occluded objects missed, and the inventory rarely re-examined. The parse step of Lucida instead detects on keyframes, fuses each instance’s multi-view observations, and verifies and completes the scene graph against the full capture—removing spurious instances, correcting implausible support relations, and disambiguating nearidentical objects—yielding the highest detection mAP in all reported settings; each verified node moreover carries the multi-view evidence bundle that asset generation and placement consume.

## 4.3 3D Visual Grounding and Iterative Pose Refinement

3D visual grounding localizes a referred object, classically via closed-vocabulary specialist detectors over reconstructed point clouds [2, 16, 38, 85, 111, 112]. Foundation models change the interface, yet LLMs with 3D-native inputs generalize weakly beyond scene-scan corpora [33, 115, 116], 2D-input VLMs trained with spatial and 3D-grounding supervision localize imprecisely [7, 15, 19, 24, 97], and agentic systems that orchestrate grounding tools stop once the referent is identified, leaving its coarse box unrefined [52, 90, 95]. Most related to Lucida, VLM agents place and arrange assets in 3D scenes: PlaceIt3D [1] benchmarks the task, FirePlace [34] solves single-shot common-sense constraints, VULCAN [42] iterates through code-level tools, and VLMPose [8] closes a 6-DoF loop against re-renderings. These methods optimize plausibility under language constraints, and none emphasizes precise alignment to the observed point cloud. GizmoAct addresses the opposite regime—grounding a generated asset to one observed instance with metric 9-DoF alignment to the point cloud—and its GUI abstraction generalizes beyond object pose: the same loop manipulates any 3D proxy (a 3D box for detection and layout, a gizmo frame for orientation estimation [79]) and consumes multi-view and virtual renderings for robustness.

Aligning a 3D model to observations is the classic object pose estimation problem, typically solved by pose initialization followed by iterative refinement. Render-and-compare methods regress a relative pose update from the rendered–observed pair [20, 45, 53], scaled to novel objects by MegaPose [46] and FoundationPose [80]; diferentiable-rendering variants descend photometric residuals [54, 77], and Any6D [51] admits generated meshes. Their updates are inferred by comparing the rendering with the observation, presuming the rendered model can resemble the observed object; convergence is decided externally by a fixed iteration count [53, 80], and scale beyond SE(3) is rarely estimated. A second family casts refinement as sequential decision making with small reinforcement-learned policies [41, 69]: discrete incremental moves with a learned stop action [12], and imitation-then-RL refinement over discrete registration steps [9, 10, 56, 65]. GizmoAct inherits this decision structure—incremental actions, learned stopping, imitation followed by reinforcement learning—but upgrades the policy to a pretrained VLM operating a GUI: it manipulates the object through its gizmo, decides its own termination from the rendering alone, tolerates geometry that disagrees with the observation, and refines the full 9-DoF state including anisotropic scale.

## 5 Conclusion

This paper introduces Lucida, a composable scene modeling system that keeps the parse–generate–place order but redistributes the requirements along it, so each step consumes only what a real capture reliably provides. The parse step discovers candidate instances on geometry-aware keyframes, consolidates their evidence over the full sequence, and reasons over inter-object relations to refine the resulting scene graph. The generate step completes each instance amodally from its evidence, and GizmoAct, a VLM policy that casts placement as multi-turn GUI interaction, aligns each generated asset by closed-loop gizmo manipulation and decides itself when alignment is reached. Precision is reached at the closed-loop end of the pipeline, so the upstream steps tolerate the occlusion, clutter, and asset–observation mismatch that real indoor captures impose.

The experiments evaluate Lucida at three levels. For scene-level 3D object detection, the scene graph surpasses prompt-based 3D detectors on CA-1M and R2S-Scene. For object pose estimation, GizmoAct improves strict alignment and 3D IoU on R2S-Object, CA-1M, and ADT, and a single policy handles initializers with diferent error profiles without retraining. At the system level, the composed scenes improve scene F-Score from 0.794 for SAM 3D to 0.924. A remaining limitation is that objects that remain missing after scene parsing cannot be recovered by the subsequent generation or grounding stages. Moreover, agentic refinement currently operates only at the final placement step; extending this closed-loop treatment across the whole parse–generate–place pipeline, toward a fully agentic framework, is an important direction for future work.

## Contributions

Authors Minghan Qin<sup>1,‡</sup>, Yuang Wang<sup>1,‡</sup>, Xiuyu Yang<sup>1,∗,‡</sup>, Yushi Long<sup>1,3,∗</sup>, Yujian Zhang<sup>1,3,∗</sup>, Ruihuan Wang<sup>1,2,∗</sup>, Kai Ye<sup>1,2,∗</sup>, Yangang Zhang<sup>1,†</sup>, Hang Li<sup>1</sup>

Affiliations <sup>1</sup>ByteDance Seed, <sup>2</sup>Peking University, <sup>3</sup>Zhejiang University

Acknowledgments We thank Wei Li, Heng Dong, and Qifeng Zhang for providing the pretrained VLM and for their help with model training. We thank Lihao Liu, Baifeng Xie, and Yiming Qiao for their help with simulation data, evaluation, and physical-plausibility post-processing.

<sup>‡</sup> Equal contribution

∗ Work done during internship at ByteDance Seed

<sup>†</sup> Corresponding author

Input Image

Any6D

SAM 3D

RecGen

Ours

GT

![](images/f1ac529df7940fd051866c955a1e04838494d0170d862972839a97df5f8d152f.jpg)

Figure 10 Additional qualitative comparison of object pose estimation on CA-1M, R2S-Object, and ADT. Same layout and reading protocol as Figure 6: each method’s posed model is rendered jointly with the ground-truth posed model (blue), from a main view and three axis-aligned close-ups.

![](images/8bbc72b9abd11979b13c436655be9a5c0858ee203be54d24f356d7227b4b8b64.jpg)  
Figure 11 Additional GizmoAct trajectory examples on CA-1M, R2S-Object, and ADT. Same layout and notation as Figure 7: a reference image per dataset, followed by closed-loop rollouts from Boxer, Any6D\*, and SAM 3D initializations.

![](images/9514a8242c660904c21f37f43ea621c837769356c2dc994888c126ba2ab2444d.jpg)  
Figure 12 Further GizmoAct trajectory examples on CA-1M, R2S-Object, and ADT. Same layout and notation as Figure 7: a reference image per dataset, followed by closed-loop rollouts from Boxer, Any6D\*, and SAM 3D initializations.

## References

[1] Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, and Guillermo Garcia-Hernando. PlaceIt3D: Language-guided object placement in real 3D scenes. In IEEE International Conference on Computer Vision, 2025.

[2] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas J. Guibas. ReferIt3D: Neural listeners for fine-grained 3d object identification in real-world scenes. In European Conference on Computer Vision, 2020.

[3] Aditya Agarwal, Gaurav Singh, Bipasha Sen, Tomás Lozano-Pérez, and Leslie Pack Kaelbling. Scenecomplete: Open-world 3d scene completion in cluttered real world environments for robot manipulation. IEEE Robotics and Automation Letters, 2025

[4] Andreea Ardelean, Mert Özer, and Bernhard Egger. Gen3dsr: Generalizable 3d scene reconstruction via divide and conquer from a single view. In International Conference on 3D Vision, 2025.

[5] Armen Avetisyan, Manuel Dahnert, Angela Dai, Manolis Savva, Angel X. Chang, and Matthias Nießner. Scan2cad: Learning cad model alignment in rgb-d scans. arXiv preprint arXiv:1811.11187, 2018.

[6] Armen Avetisyan, Tatiana Khanova, Christopher Choy, Denver Dash, Angela Dai, and Matthias Nießner. Scenecad: Predicting object alignments and layouts in rgb-d scans. arXiv preprint arXiv:2003.12622, 2020.

[7] Shuai Bai et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[8] Sangwon Baik, Gunhee Kim, Mingi Choi, and Hanbyul Joo. Text-guided 6D object pose rearrangement via closed-loop VLM agents. arXiv preprint arXiv:2604.09781, 2026.

[9] Dominik Bauer, Timothy Patten, and Markus Vincze. ReAgent: Point cloud registration using imitation and reinforcement learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.

[10] Dominik Bauer, Timothy Patten, and Markus Vincze. Sporeagent: Reinforced scene-level plausibility for object pose refinement. arXiv preprint arXiv:2201.00239, 2022.

[11] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In IEEE Conference on Computer Vision and Pattern Recognition, 2023.

[12] Benjamin Busam, Hyun Jun Jung, and Nassir Navab. I like to move it: 6D pose estimation as an action decision process. arXiv preprint arXiv:2009.12678, 2020.

[13] ByteDance Seed. Seedream 5.0 pro. https://seed.bytedance.com/en/seedream5\_0\_pro, 2026.

[14] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Efrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In International Conference on Learning Representations, 2026.

[15] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In IEEE Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

[16] Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. arXiv preprint arXiv:1912.08830, 2019.

[17] Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. Autopartgen: Autoregressive 3d part generation and discovery. In Advances in Neural Information Processing Systems, 2025.

[18] Qiuyu Chen, Aaron Walsman, Marius Memmel, Kaichun Mo, Alex Fang, Dieter Fox, and Abhishek Gupta. URD Former: A pipeline for constructing articulated simulation environments from real-world images. In Proceedings of Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.124.

[19] Jang Hyun Cho, Boris Ivanovic, Yulong Cao, Edward Schmerling, Yue Wang, Xinshuo Weng, Boyi Li, Yurong You, Philipp Krähenbühl, Yan Wang, and Marco Pavone. Language-image models with 3D understanding. In International Conference on Learning Representations, 2025.

[20] Martin Cífka, Georgy Ponimatkin, Yann Labbé, Bryan Russell, Mathieu Aubry, Vladimir Petrik, and Josef Sivic. FocalPose++: Focal length and object pose estimation via render and compare. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[21] Tianyuan Dai, Josiah Wong, Yunfan Jiang, Chen Wang, Cem Gokmen, Ruohan Zhang, Jiajun Wu, and Li Fei-Fei. ACDC: Automated creation of digital cousins for robust policy learning. In Conference on Robot Learning, 2024.

[22] Daniel DeTone, Tianwei Shen, Fan Zhang, Lingni Ma, Julian Straub, Richard Newcombe, and Jakob Engel. Boxer: Robust lifting of open-world 2d bounding boxes to 3d. arXiv preprint arXiv:2604.05212, 2026.

[23] Wenqi Dong, Bangbang Yang, Zesong Yang, Yuan Li, Tao Hu, Hujun Bao, Yuewen Ma, and Zhaopeng Cui. Hiscene: Creating hierarchical 3d scenes with isometric view generation. arXiv preprint arXiv:2504.13072, 2025.

[24] Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, and Hang Li. Robix: A unified model for robot interaction, reasoning and planning. arXiv preprint arXiv:2509.01106, 2025.

[25] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, and Hao Zhang. 3D-FRONT: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10933–10942, 2021. doi: 10.1109/ ICCV48922.2021.01075.

[26] Daoyi Gao, Dávid Rozenberszki, Stefan Leutenegger, and Angela Dai. Difcad: Weakly-supervised probabilistic cad model retrieval and alignment from an rgb image. ACM Transactions on Graphics, 2024.

[27] Google DeepMind. Introducing nano banana pro. https://blog.google/technology/ai/nano-banana-pro/, 2025.

[28] Diandian Gu, Jing Lin, Gaohong Liu, Jiahang Liu, Su Ma, Guang Shi, Jun Wang, Qinlong Wang, Qianyi Wu, Zhongcong Xu, Xuanyu Yi, Zihao Yu, Jianfeng Zhang, Zhuolin Zheng, Yifan Zhu, Rui Chen, Hengkai Guo, Xiaoyang Guo, Mingcong Han, Xu Han, Xiu Li, Yixun Liang, Weiqiang Lou, Junzhe Lu, Guan Luo, Minghan Qin, Shuguang Wang, and Yuang Wang. Seed3D 2.0: Advancing high-fidelity simulation-ready 3d content generation. arXiv preprint arXiv:2605.13862, 2026.

[29] Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, et al. Con ceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In IEEE International Conference on Robotics and Automation, 2024.

[30] Can Gümeli, Angela Dai, and Matthias Nießner. Roca: Robust cad model retrieval and alignment from a single image. arXiv preprint arXiv:2112.01988, 2021.

[31] Jinkun Hao, Naifu Liang, Zhen Luo, Xudong Xu, Weipeng Zhong, Ran Yi, Yichen Jin, Zhaoyang Lyu, Feng Zheng, Lizhuang Ma, and Jiangmiao Pang. MesaTask: Towards task-driven tabletop scene generation via 3d spatial reasoning. In Advances in Neural Information Processing Systems, 2025.

[32] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.

[33] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. arXiv preprint arXiv:2307.12981, 2023.

[34] Ian Huang, Yanan Bao, Karen Truong, Howard Zhou, Cordelia Schmid, Leonidas Guibas, and Alireza Fathi. FirePlace: Geometric refinements of LLM common sense reasoning for 3D object placement. In IEEE Conference on Computer Vision and Pattern Recognition, 2025.

[35] Siyuan Huang, Siyuan Qi, Yinxue Xiao, Yixin Zhu, Ying Nian Wu, and Song-Chun Zhu. Cooperative holistic scene understanding: Unifying 3d object, layout, and camera pose estimation. arXiv preprint arXiv:1810.13049, 2018.

[36] Weikai Huang, Jieyu Zhang, Sijun Li, Taoyang Jia, Jiafei Duan, Yunqian Cheng, Jaemin Cho, Matthew Wallingford, Rustin Soraki, Chris Dongjoo Kim, Shuo Liu, Donovan Clay, Taira Anderson, Winson Han, Ali Farhadi, Bharath Hariharan, Zhongzheng Ren, and Ranjay Krishna. Wilddet3d: Scaling promptable 3d detection in the wild. arXiv preprint arXiv:2604.08626, 2026.

[37] Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance difusion for single image to 3d scene generation. arXiv preprint arXiv:2412.03558, 2024.

[38] Ayush Jain, Nikolaos Gkanatsios, Ishita Mediratta, and Katerina Fragkiadaki. Bottom up top down detection transformers for language grounding in images and point clouds. arXiv preprint arXiv:2112.08879, 2021.

[39] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 2023.

[40] Xin Kong, Shikun Liu, Marwan Taher, and Andrew J. Davison. vmap: Vectorised object mapping for neural field slam. In IEEE Conference on Computer Vision and Pattern Recognition, 2023.

[41] Alexander Krull, Eric Brachmann, Sebastian Nowozin, Frank Michel, Jamie Shotton, and Carsten Rother. PoseAgent: Budget-constrained 6D object pose estimation via reinforcement learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[42] Zhengfei Kuang, Rui Lin, Long Zhao, Gordon Wetzstein, Saining Xie, and Sanghyun Woo. VULCAN: Toolaugmented multi agents for iterative 3D object arrangement. In IEEE Conference on Computer Vision and Pattern Recognition, 2026.

[43] Abhijit Kundu, Kyle Genova, Xiaoqi Yin, Alireza Fathi, Caroline Pantofaru, Leonidas Guibas, Andrea Tagliasacchi, Frank Dellaert, and Thomas Funkhouser. Panoptic neural fields: A semantic object-aware neural scene representation. arXiv preprint arXiv:2205.04334, 2022.

[44] Weicheng Kuo, Anelia Angelova, Tsung-Yi Lin, and Angela Dai. Mask2cad: 3d shape prediction by learning to segment and retrieve. In European Conference on Computer Vision, 2020.

[45] Yann Labbé, Justin Carpentier, Mathieu Aubry, and Josef Sivic. CosyPose: Consistent multi-view multi-object 6D pose estimation. In European Conference on Computer Vision, 2020.

[46] Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, and Josef Sivic. Megapose: 6d pose estimation of novel objects via render and compare. arXiv preprint arXiv:2212.06870, 2022.

[47] Yuqing Lan, Chenyang Zhu, Zhirui Gao, Jiazhao Zhang, Yihan Cao, Renjiao Yi, Yijie Wang, and Kai Xu. Boxfusion: Reconstruction-free open-vocabulary 3d object detection via real-time multi-view box fusion. Computer Graphics Forum, 2025.

[48] Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. DART: Noise injection for robust imitation learning. In Conference on Robot Learning, pages 143–156, 2017.

[49] Justin Lazarow, David Grifiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. arXiv preprint arXiv:2412.04458, 2024.

[50] Justin Lazarow, Kai Kang, and Afshin Dehghan. Rooms from motion: Un-posed indoor 3d object detection as localization and mapping. In Advances in Neural Information Processing Systems, 2025.

[51] Taeyeop Lee, Bowen Wen, Minjun Kang, Gyuree Kang, In So Kweon, and Kuk-Jin Yoon. Any6d: Model-free 6d pose estimation of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11633–11643, 2025.

[52] Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, and Junwei Liang. Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding. arXiv preprint arXiv:2412.04383, 2024.

[53] Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, and Dieter Fox. Deepim: Deep iterative matching for 6d pose estimation. arXiv preprint arXiv:1804.00175, 2018.

[54] Yen-Chen Lin, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting neural radiance fields for pose estimation. arXiv preprint arXiv:2012.05877, 2020.

[55] Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. Partcrafter: Structured 3d mesh generation via compositional latent difusion transformers. In Advances in Neural Information Processing Systems, 2025.

[56] Liu Liu, Jianming Du, Hao Wu, Xun Yang, Zhenguang Liu, Richang Hong, and Meng Wang. Categorylevel articulated object 9D pose estimation via reinforcement learning. In ACM International Conference on Multimedia, 2023.

[57] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

[58] John McCormac, Ronald Clark, Michael Bloesch, Andrew J. Davison, and Stefan Leutenegger. Fusion++: Volumetric object-level slam. In International Conference on 3D Vision, 2018.

[59] Yanxu Meng, Haoning Wu, Ya Zhang, and Weidi Xie. Scenegen: Single-image 3d scene generation in one feedforward pass. arXiv preprint arXiv:2508.15769, 2025.

[60] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, 2020.

[61] Yinyu Nie, Xiaoguang Han, Shihui Guo, Yujian Zheng, Jian Chang, and Jian Jun Zhang. Total3dunderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image. arXiv preprint arXiv:2002.12212, 2020.

[62] OpenAI. Gpt image 2. https://developers.openai.com/api/docs/models/gpt-image-2, 2026.

[63] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20076–20086, 2023. doi: 10.1109/ICCV51070.2023.01842.

[64] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

[65] Konstantin Röhrl, Dominik Bauer, Timothy Patten, and Markus Vincze. TrackAgent: 6D object tracking via reinforcement learning. In International Conference on Computer Vision Systems, 2023.

[66] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection. In IEEE Winter Conference on Applications ofComputer Vision, 2022.

[67] SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, et al. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025.

[68] Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[69] Jianzhun Shao, Yuhang Jiang, Gu Wang, Zhigang Li, and Xiangyang Ji. Pfrl: Pose-free reinforcement learning for 6d pose estimation. arXiv preprint arXiv:2102.12096, 2021.

[70] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[71] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Norman Müller, Matthias Nießner, Angela Dai, and Peter Kontschieder. Panoptic lifting for 3d scene understanding with neural fields. arXiv preprint arXiv:2212.09802, 2022.

[72] Yawar Siddiqui, Duncan Frost, Samir Aroudj, Armen Avetisyan, Henry Howard-Jenkins, Daniel DeTone, Pierre Moulon, Qirui Wu, Zhengqin Li, Julian Straub, Richard Newcombe, and Jakob Engel. Shaper: Robust condi tional 3d shape generation from casual captures. arXiv preprint arXiv:2601.11514, 2026.

[73] Ayça Takmaz, Elisabetta Fedele, Robert W. Sumner, Marc Pollefeys, Federico Tombari, and Francis Engelmann. Openmask3d: Open-vocabulary 3d instance segmentation. In Advances in Neural Information Processing Systems, 2023.

[74] Bin Tan, Changjiang Sun, Xiage Qin, Hanat Adai, Zelin Fu, Tianxiang Zhou, Han Zhang, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Masked depth modeling for spatial perception. arXiv preprint arXiv:2601.17895, 2026.

[75] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151, 2024.

[76] Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, and Pulkit Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. In Proceedings of Robotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.015.

[77] Jonathan Tremblay, Bowen Wen, Valts Blukis, Balakumar Sundaralingam, Stephen Tyree, and Stan Birchfield. Dif-DOPE: Diferentiable deep object pose estimation. arXiv preprint arXiv:2310.00463, 2023.

[78] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=DTQIjngDta.

[79] Zehan Wang, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything: Learning robust object orientation estimation from rendering 3D models. In International Conference on Machine Learning, 2025.

[80] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. arXiv preprint arXiv:2312.08344, 2023.

[81] Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, and Wolfram Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In Robotics: Science and Systems, 2024.

[82] Qianyi Wu, Xian Liu, Yuedong Chen, Kejie Li, Chuanxia Zheng, Jianfei Cai, and Jianmin Zheng. Objectcompositional neural implicit surfaces. arXiv preprint arXiv:2207.09686, 2022.

[83] Qianyi Wu, Kaisiyuan Wang, Kejie Li, Jianmin Zheng, and Jianfei Cai. Objectsdf++: Improved objectcompositional neural implicit surfaces. arXiv preprint arXiv:2308.07868, 2023.

[84] Tianhao Wu, Chuanxia Zheng, Frank Guan, Andrea Vedaldi, and Tat-Jen Cham. Amodal3r: Amodal 3d reconstruction from occluded 2d images. arXiv preprint arXiv:2503.13439, 2025.

[85] Yanmin Wu, Xinhua Cheng, Renrui Zhang, Zesen Cheng, and Jian Zhang. Eda: Explicit text-decoupling and dense alignment for 3d visual grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19231–19242, 2023. doi: 10.1109/CVPR52729.2023.01843.

[86] Chong Xia, Kai Zhu, Zizhuo Wang, Fangfu Liu, Zhizheng Zhang, and Yueqi Duan. Simrecon: Simready compositional scene reconstruction from real videos. In IEEE Conference on Computer Vision and Pattern Recognition, 2026.

[87] Hongchi Xia, Chih-Hao Lin, Hao-Yu Hsu, Quentin Leboutet, Katelyn Gao, Michael Paulitsch, Benjamin Ummenhofer, and Shenlong Wang. Holoscene: Simulation-ready interactive 3d worlds from a single video. arXiv preprint arXiv:2510.05560, 2025.

[88] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. In Robotics: Science and Systems (RSS), 2018.

[89] Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, and Masayoshi Tomizuka. Nerf-det: Learning geometry-aware volumetric representation for multi-view 3d object detection. In IEEE International Conference on Computer Vision, 2023.

[90] Runsen Xu, Zhiwei Huang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Vlm-grounder: A vlm agent for zero-shot 3d visual grounding. arXiv preprint arXiv:2410.13860, 2024.

[91] Xiuwei Xu, Huangxing Chen, Linqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu. Embodiedsam: Online segment any 3d thing in real time. In International Conference on Learning Representations, 2025.

[92] Han Yan, Yang Li, Zhennan Wu, Shenzhou Chen, Weixuan Sun, Taizhang Shang, Weizhe Liu, Tian Chen, Xiaqiang Dai, Chao Ma, Hongdong Li, and Pan Ji. Frankenstein: Generating semantic-compositional 3d scenes in one tri-plane. In SIGGRAPH Asia 2024 Conference Papers, 2024.

[93] Mi Yan, Jiazhao Zhang, Yan Zhu, and He Wang. Maskclustering: View consensus based mask graph clustering for open-vocabulary 3d instance segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, 2024.

[94] Bangbang Yang, Yinda Zhang, Yinghao Xu, Yijin Li, Han Zhou, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Learning object-compositional neural radiance field for editable scene rendering. arXiv preprint arXiv:2109.01847, 2021.

[95] Jianing Yang, Xuweiyi Chen, Shengyi Qian, Nikhil Madaan, Madhavan Iyengar, David F. Fouhey, and Joyce Chai. Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent. arXiv preprint arXiv:2309.12311, 2023.

[96] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4V. arXiv preprint arXiv:2310.11441, 2023.

[97] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, and Hengshuang Zhao. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025.

[98] Yandan Yang, Baoxiong Jia, Peiyuan Zhi, and Siyuan Huang. PhyScene: Physically interactable 3d scene synthesis for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16262–16272, 2024.

[99] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language guided generation of 3d embodied ai environments. arXiv preprint arXiv:2312.09067, 2023.

[100] Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. Sam3d: Segment anything in 3d scenes. In IEEE International Conference on Computer Vision Workshops, 2023.

[101] Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your eficient rl framework secretly brings you of-policy rl training. https://fengyao.notion.site/off-policy-rl, 2025.

[102] Jin Yao, Hao Gu, Xuweiyi Chen, Jiayun Wang, and Zezhou Cheng. Open vocabulary monocular 3d object detection. In International Conference on 3D Vision, 2026.

[103] Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Wei Yang, Lan Xu, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics, 2025.

[104] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European Conference on Computer Vision, 2024.

[105] Hong-Xing Yu, Leonidas J. Guibas, and Jiajun Wu. Unsupervised discovery of object radiance fields. arXiv preprint arXiv:2107.07905, 2021.

[106] Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, Tengyu Liu, and Siyuan Huang. Metascenes: Towards automated replica creation for real-world 3d scans. In IEEE Conference on Computer Vision and Pattern Recognition, 2025.

[107] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[108] Andrii Zadaianchuk, Leonardo Barcellona, Lennard Schuenemann, Christian Gumbsch, Zehao Wang, Muhammad Zubair Irshad, Fabien Despinoy, Rahaf Aljundi, Stratis Gavves, and Sergey Zakharov. Reconstruction by generation: 3d multi-object scene reconstruction from sparse observations. arXiv preprint arXiv:2604.27106, 2026.

[109] Cheng Zhang, Zhaopeng Cui, Yinda Zhang, Bing Zeng, Marc Pollefeys, and Shuaicheng Liu. Holistic 3d scene understanding from a single image with implicit representation. arXiv preprint arXiv:2103.06422, 2021.

[110] Hanxue Zhang, Haoran Jiang, Qingsong Yao, Yanan Sun, Renrui Zhang, Hao Zhao, Hongyang Li, Hongzi Zhu, and Zetong Yang. Detect anything 3d in the wild. In IEEE International Conference on Computer Vision, 2025.

[111] Yiming Zhang, ZeMing Gong, and Angel X. Chang. Multi3drefer: Grounding text description to multiple 3d objects. arXiv preprint arXiv:2309.05251, 2023.

[112] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvg-transformer: Relation modeling for visual grounding on point clouds. In IEEE International Conference on Computer Vision, 2021.

[113] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J. Davison. In-place scene labelling and understanding with implicit scene representation. In IEEE International Conference on Computer Vision, 2021.

[114] Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. Zero-shot scene reconstruction from single images with deep prior assembly. In Advances in Neural Information Processing Systems, 2024.

[115] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet efective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024.

[116] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained trans former for 3d vision and text alignment. arXiv preprint arXiv:2308.04352, 2023.