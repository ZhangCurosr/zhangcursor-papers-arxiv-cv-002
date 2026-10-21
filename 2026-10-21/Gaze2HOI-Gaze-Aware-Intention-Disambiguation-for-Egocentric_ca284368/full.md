# Gaze2HOI: Gaze-Aware Intention Disambiguation for Egocentric Hand-Object Interaction Forecasting

Chuanyang Ma, Chenye Wang, Yixuan Wang, Huiling Zhou, Xingqun Qi, Muyi Sun, Man Zhang, Senior Member, IEEE, Caifeng Shan, Senior Member, IEEE, and Zhenan Sun, Senior Member, IEEE

Abstract—Anticipating how users will interact with their surroundings is essential for proactive systems that provide timely assistance in AR, VR, and embodied applications. Prior work typically forecasts hand-object interactions directly from historical observations, leaving intention and contact geometry implicit. However, similar observation histories can precede semantically or geometrically distinct futures. Inspired by behavioral evidence that gaze can provide early cues for underlying intention, we propose Gaze2HOI, a gaze-aware framework that formulates hand-object interaction forecasting as a hierarchical problem comprising intention inference, contact grounding, and coupled hand-object motion generation. Specifically, Gaze2HOI first weights gaze cues by fixation strength and recency, and uses them to modulate the visual representations of a vision-language model (VLM) through Temporal Gaze Saliency Modulation (TGSM), thereby reducing semantic ambiguity in intention inference. Subsequently, an Anticipatory Contact Grounding (ACG) module maps these structured intentions into explicit contact, support, and placement fields, effectively mitigating geometric ambiguity. Driven by these explicit conditions, a diffusion-based model jointly generates the 6-DoF object trajectories and future articulated motion of both hands. To evaluate Gaze2HOI, we further introduce EgoAIM, a phase-aware benchmark that organizes egocentric hand-object interactions into four phases defined by evolving hand-object relations and provides fine-grained intention annotations. Extensive experiments demonstrate that Gaze2HOI achieves state-of-the-art performance in both intention inference and motion forecasting, even approaching oracle-level end-toend performance in specific interaction phases. Project page: https://gaze2hoi.github.io/.

Index Terms—Hand-Object Interaction, Motion Forecasting, Egocentric Vision, Gaze Cue, Intention Disambiguation.

## I. INTRODUCTION

R <sup>ECENT</sup> <sup>advances</sup> <sup>in</sup> <sup>AR,</sup> <sup>VR,</sup> <sup>and</sup> <sup>embodied</sup> <sup>intelligence</sup>have enabled interactive systems to tackle increasingly complex tasks, spurring a growing demand for natural and anticipatory assistance [1]–[4]. However, many existing systems still operate under a reactive paradigm, typically responding to explicit commands or actions that are already underway [5]. To achieve seamless human-machine collaboration, systems must transition from passive responders to proactive assistants capable of anticipating human intentions and upcoming actions [6]–[8]. Since routine manipulation is largely realized through hand interactions with surrounding objects, egocentric hand-object interaction (HOI) forecasting has become a foundational task that bridges real-time perception and proactive human-machine assistance [9]–[11]. Figure 1 illustrates this task through representative forecasts across interaction phases.

![](images/ed1944fa7004013cdeb93ed7330ccf2db18c3ee820d23a609d160fa64a666599.jpg)  
Fig. 1. Egocentric hand-object interaction forecasting. Gaze2HOI forecasts coordinated interactions involving both hands and multiple objects across four phases: Establish Contact, Move, Manipulate, and Put Down. L and R denote the left and right hands, respectively.

Existing methods have explored various aspects of egocentric HOI forecasting but still struggle to capture continuous, coordinated hand-object interaction dynamics [11]. Early approaches primarily anticipate discrete action targets or events, identifying interaction types or locations without recovering the underlying continuous kinematics [8], [12], [13]. To capture spatial evolution, subsequent research extends these predictions to continuous trajectories by forecasting either future hand positions [9], [10], [14] or isolated object poses [15]. While these methods model the temporal evolution of individual elements, they fail to jointly model fine-grained finger articulation and object motion, significantly limiting their capacity to represent complex handobject coordination [16]. Recent efforts in joint hand-object forecasting attempt to bridge this coordination gap. However, these approaches typically learn a direct mapping from historical observations to future trajectories, leaving human intention and physical contact geometry implicit [11], [17]. Since similar historical observations can precede completely divergent futures [9], such direct-mapping paradigms struggle to determine a coherent interaction when faced with multiple plausible continuations [11].

Reliable forecasting of coordinated hand-object motion requires the resolution of two intertwined ambiguities: semantic ambiguity in the intended interaction and geometric ambiguity in its physical realization. As illustrated in Figure 2, similar observation histories can be compatible with different intended actions, while even the same action involving the same object can exhibit multiple valid contact configurations [18].

![](images/681af44802d35cc4a7bb52ee8385bd8f138c349f9c023b4752a477d515c88a5c.jpg)  
Fig. 2. Hierarchical intention disambiguation in Gaze2HOI. Gaze-aware intention inference reduces semantic ambiguity about the intended object and action. Anticipatory Contact Grounding then combines the inferred intentions with object geometry and interaction history to localize likely contact regions, reducing geometric ambiguity. The resulting semantic and geometric cues guide coordinated hand-object motion forecasting.

To address these challenges, we propose Gaze2HOI, a gaze-aware framework that formulates hand-object interaction forecasting as a hierarchical problem comprising intention inference, contact grounding, and coupled hand-object motion generation. Inspired by behavioral evidence that gaze can provide early cues before an intention fully manifests in hand motion [19], we first introduce Temporal Gaze Saliency Modulation (TGSM). This module weights gaze cues according to fixation strength and recency to modulate the visual representation of a VLM (e.g., Qwen3-VL [20]), thereby reducing semantic ambiguity by explicitly identifying the intended object and action. Meanwhile, an inferred intention such as “reach and grasp the moka pot handle” still leaves the precise contact configuration unspecified [21]. To resolve this geometric ambiguity, we employ an Anticipatory Contact Grounding (ACG) module that combines the inferred intentions with object geometry and historical interaction information to predict contact, support, and placement fields. These fields express the anticipated interaction through spatially explicit relationships, providing geometric guidance for a conditional diffusion model that jointly forecasts the 6-DoF object trajectories and future articulated motion of both hands.

This proposed hierarchical formulation calls for a benchmark that pairs fine-grained future intention annotations with continuous hand-object motion. As hand-object relations evolve, the cues available for intention inference and the demands of motion coordination also change [19], [22]. Therefore, phase-specific evaluation can reveal performance differences obscured by aggregate metrics. Existing benchmarks for action anticipation primarily assess future action categories or interaction targets [8], [23], whereas those for 3D hand-object tracking and motion forecasting emphasize spatial states and their temporal evolution [11], [24]. However, these task-specific protocols provide limited support for jointly assessing intention inference and coordinated motion forecasting under these changing interaction conditions. To support these evaluations, we introduce EgoAIM, a phase-aware benchmark for Egocentric Anticipation of Intention and Motion. Built on existing egocentric recordings [24], EgoAIM comprises 7,429 forecasting windows organized into four phases defined by evolving hand-object relations. EgoAIM pairs the fine-grained future intention annotations for each hand with continuous 3D hand-object motion and provides furniture support-plane annotations to characterize the environmental geometry relevant to object support and placement. This design supports evaluating intention inference and motion forecasting both separately and jointly across the four interaction phases.

Overall, our contributions are summarized as follows:

• We reformulate egocentric hand-object interaction forecasting as a hierarchical problem comprising intention inference, contact grounding, and coupled hand-object motion generation.

• We introduce EgoAIM, a phase-aware benchmark with 7,429 forecasting windows, featuring fine-grained intention and geometry-grounded annotations for egocentric hand-object interaction forecasting.

• We propose Gaze2HOI, coupling Temporal Gaze Saliency Modulation and Anticipatory Contact Grounding for semantic and geometric disambiguation to drive a diffusion-based model for hand-object forecasting.

• We extensively evaluate Gaze2HOI in both stage-wise and overall system settings, demonstrating state-of-the-art performance in intention inference and motion forecasting across interaction phases.

## II. RELATED WORK

This section reviews egocentric hand-object interaction forecasting as the task context. It also examines two complementary directions: conditional motion generation under given conditions and gaze-aware intention understanding for inferring semantic conditions from historical observations.

## A. Egocentric Hand-Object Interaction Forecasting

Research on egocentric hand-object interaction forecasting spans various levels of prediction granularity. Target-oriented approaches anticipate action locations [8] or next-active objects alongside associated actions and time-to-contact [13], [25], providing spatial or semantic summaries without recovering articulated hand-object dynamics. To capture temporal evolution, probabilistic approaches jointly predict 2D hand trajectories and interaction hotspots [9], [26], while Diff-IP2D and MADiff incorporate camera egomotion into diffusionbased trajectory prediction [16], [27]. Extending to threedimensional space, USST models uncertainty in future hand positions [10], whereas Uni-Hand predicts hand-center and selected-joint trajectories together with contact and separation states [28]. Focusing on fine-grained articulation, HandDiff forecasts single-hand joint motion through diffusion [29], while He et al. [22] combine historical hand motion, gaze cues, and object locations to forecast the articulated motion of both hands in egocentric grasping tasks. ObjectForesight focuses on object dynamics, using scene geometry and historical poses to forecast future 6-DoF trajectories [15]. Although these methods capture increasingly detailed motion, their predictions typically isolate hands from objects, failing to jointly model their coordinated evolution. Moving toward joint forecasting, TACO benchmarks the simultaneous prediction of articulated bimanual motion and the poses of tools and target objects [11]. Despite these advances, existing methods generally map historical observations directly to future motion, leaving interaction intention and 3D contact geometry implicit [9], [11], [22], [30]. These limitations motivate a hierarchical approach to explicitly address semantic and geometric ambiguity in coordinated hand-object interaction forecasting.

## B. Conditional Hand-Object Motion Generation

Extensive research has leveraged diverse conditioning modalities to drive hand-object motion generation [31], [32]. Language-conditioned methods provide semantic control over hand articulation and object motion, with DiffH2O separating grasping from manipulation while maintaining continuity between the stages [21]. Since semantic descriptions typically leave precise surface contacts underdetermined, Text2HOI predicts contact maps from text and object geometry to guide joint hand-object motion diffusion [33]. Contact representations also support generation conditioned on prescribed object trajectories: BimArt generates temporally varying contact maps to guide bimanual motion for articulated objects [34]. For interactions involving multiple entities, SyncDiff jointly models absolute and relative motions, explicitly synchronizing their generation under action and geometry conditions [35]. GHO-Diffusion further explores gaze-conditioned synthesis, generating object motion followed by corresponding hand motion using gaze sequences spanning the synthesis interval [36]. However, its gaze-motion consistency formulation does not explicitly model the temporal lead of gaze, with fixations often preceding task-relevant hand actions [19]. Furthermore, its reliance on future gaze observations within the generated interval precludes its direct application to forecasting based solely on historical observations. Even when egocentric visual context is available, LatentAct requires an action description and a specified 3D contact point [37], while MEgoHand relies on action instructions and an initial hand pose to guide motion generation [38]. These methods primarily address how an interaction should be realized under supplied guidance. To support seamless interaction in practice, systems must autonomously infer the user’s intention from ongoing observations and translate it into conditions for motion generation.

## C. Gaze-Aware Intention Understanding

Gaze-aware intention understanding draws on behavioral evidence that gaze can precede task-relevant hand-object actions [19], [39]. GIMO integrates gaze through bidirectional attention [3], while GazeMotion forecasts gaze from its history to guide motion prediction [40]. These methods use gaze to improve motion prediction, without producing explicit object-action intention descriptions. For semantic interpretation, GLAM aligns gaze scanpaths with language, encoding fixation location, duration, and order to identify visual-search targets [41]. However, its predictions only identify targets, without specifying how the user intends to interact with the object. Advances in VLMs have enabled richer scene understanding, yet effective gaze integration remains challenging. Peng et al. explore input-level prompting through textual gaze coordinates, visual markers, and saliency maps [42]. The visual-marker variant, however, alters scene appearance and may bias semantic interpretation by mixing gaze annotations with scene content. Beyond input-level prompting, Voila-A injects gaze heatmaps into a Perceiver resampler’s attention keys, but its heatmap encoding does not explicitly separate current fixation from gaze history [43]. Gaze-VLM regularizes attention using temporally aggregated gaze during training [44], but its video-only inference cannot directly exploit observed gaze to distinguish plausible intentions. Gaze2HOI instead represents current fixation and gaze history as separate, scenealigned signals that drive gated scale-and-shift modulation of VLM visual features at multiple levels. This design conditions structured future intention inference on evolving attention, providing semantic guidance for geometric grounding and coordinated hand-object motion forecasting.

TABLE I  
COMPARISON OF DATASET SCALE, ANNOTATIONS, AND SUPPORTED PREDICTION TASKS.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">Dataset scale</td><td colspan="4">Annotations</td><td colspan="3">Prediction support</td></tr><tr><td>#Samples</td><td>#Objects</td><td>#Intents</td><td>Gaze</td><td>Hand repr. Object pose</td><td></td><td>Semantics</td><td>Scene geometry</td><td>Intention Motion</td><td></td><td>Joint</td></tr><tr><td>Ego4D [23]</td><td>9,645r</td><td>一</td><td></td><td>√*</td><td>2D boxes</td><td>一</td><td>Action</td><td>Meshes</td><td>x</td><td>√</td><td>x</td></tr><tr><td>EgoThink [45]</td><td>700</td><td></td><td></td><td>x</td><td></td><td></td><td>Visual QA</td><td></td><td>V</td><td>x</td><td>x</td></tr><tr><td>EgoPAT3D [8]</td><td>150°</td><td></td><td></td><td>x</td><td>3D point</td><td></td><td></td><td>Point clouds</td><td>√t</td><td>√‡</td><td>x</td></tr><tr><td>EgoGazeVQA [42]</td><td>913c</td><td></td><td></td><td>√</td><td></td><td></td><td>Intent QA</td><td></td><td>√</td><td>x</td><td>x</td></tr><tr><td>GazeHOI [36]</td><td>1,378c</td><td>33</td><td></td><td>√</td><td>MANO</td><td>6-DoF</td><td>Task text</td><td></td><td>x</td><td>√</td><td>x</td></tr><tr><td>HOT3D [24]</td><td>424r</td><td>33</td><td></td><td>√*</td><td>MANO</td><td>6-DoF</td><td></td><td>Point clouds</td><td>x</td><td>√</td><td>x</td></tr><tr><td>H2O [46]</td><td>933c</td><td>8</td><td></td><td>x</td><td>MANO</td><td>6-DoF</td><td>Action</td><td>Depth maps</td><td>x</td><td>√</td><td>x</td></tr><tr><td>TACO [11]</td><td>2.5K⁸</td><td>196</td><td></td><td>x</td><td>MANO</td><td>6-DoF</td><td>Action</td><td>Depth maps</td><td>x</td><td>√</td><td>x</td></tr><tr><td>EgoAIM (ours)</td><td>7,429w</td><td>33</td><td>9,038</td><td>√</td><td>MANO</td><td>6-DoF</td><td>Per-hand intent</td><td>Support planes</td><td>√</td><td>√</td><td>√</td></tr></table>

Units: r recordings/videos, s sequences, c clips, i images, w windows (counts not directly comparable). #Objects: instances; #Intents: non-null per-hand annotations. —: unreported/inapplicable; ✗: unavailable/not directly supported. <sup>∗</sup>Gaze subset; <sup>†</sup>target localization; <sup>‡</sup>derived trajectory protocols [10]. Intention includes QA; Joint requires established paired intention and articulated 3D HOI motion evaluation on shared future windows. Scene geometry excludes object meshes.

![](images/228be00ae4a8f8df26175474fb70212858654f6cc4a495778cd2f88d209501cd.jpg)  
Fig. 3. Illustration of EgoAIM annotations. (Top) A 2-second window pairing egocentric observations with future hand-object interaction. (Middle) Alignment between 2D egocentric observations and 3D geometric annotations. (Bottom) Structured future intentions annotated separately for each hand.

## III. EGOAIM

This section introduces EgoAIM, a phase-aware benchmark for Egocentric Anticipation of Intention and Motion. As summarized in Table I, existing benchmarks provide semantic and motion annotations, yet offer limited support for joint evaluation of fine-grained future intentions and coordinated 3D hand-object motion. EgoAIM bridges this gap by linking per-hand future intentions to continuous 3D kinematics across four interaction phases. Support-plane and placement-target annotations further characterize the environmental geometry relevant to object support and placement.

## A. Benchmark Construction

EgoAIM integrates egocentric RGB videos, eye gaze, articulated hand poses, and 6-DoF object trajectories sourced from HOT3D-Aria [24], all temporally aligned with the RGB timeline. To capture complete intended interactions, we extract

60-frame candidate windows with a 5-frame stride, providing alternative temporal boundaries around interaction events and reducing the risk of intention truncation. At 30 fps, each window comprises one second of historical observations and one second of future motion, as illustrated in Fig. 3. All candidate windows undergo manual review to exclude those with incomplete interactions, missing hand tracking, and interrupted tracking of interacting objects. For the Establish Contact and Put Down phases, we additionally identify the completion of the annotated interaction. When completion precedes the end of the prediction horizon, the final valid hand-object state is frozen for the remaining frames, preventing subsequent unannotated actions from contaminating the motion target. This process yields 7,429 forecasting windows covering 33 object instances, with 9,038 non-null per-hand intention annotations.

We distinguish four phases according to evolving handobject relations, reflecting distinct intention inference and motion coordination demands throughout the interaction process. Establish Contact involves reaching toward an object and establishing a grasp or touch, requiring anticipation of the intended target and contact configuration. Move concerns transporting an object while maintaining contact, emphasizing coordinated hand-object motion. Manipulate covers functional operations such as stirring or pressing, where task-specific articulation and inter-object relationships become important. Put Down involves placing an object at an intended destination, introducing constraints from the target location and supporting surface. This organization enables phase-specific evaluation of intention inference and motion forecasting, revealing performance differences that aggregate metrics may obscure. We manually assign a phase label to each window and annotate intentions separately for each hand to capture the potentially different roles of the two hands within the interaction.

## B. Intention and Scene Annotations

To capture the asymmetric nature of bimanual manipulation, we annotate fine-grained future intentions separately for each hand using a structured intention representation. This representation explicitly couples physical interactions with functional goals: the Action, Object, and Part fields describe how the hand engages with an object, while Purpose and

![](images/9f27f59104a36b9acce17a254b85f3818e49019cee12201998df48ac8a402dd7.jpg)  
Fig. 4. Overview of Gaze2HOI. Hierarchical inference from egocentric observations to future hand-object interactions: (a) TGSM modulates VLM visual features with gaze cues to infer structured per-hand intentions. (b) ACG grounds inferred intentions into explicit 3D contact, support, and placement fields (c) Conditional motion diffusion leverages semantic and geometric conditions to jointly forecast articulated hand motion and 6-DoF object trajectories.

Target identify the intended operation and its target object or surface. For placement actions, an additional 2D Point field localizes the intended destination in the final observed RGB frame. Calibrated camera projection and ray-plane intersection with the annotated support planes establish a bidirectional mapping between this image point and its 3D placement target. Within this representation, hands without task-directed involvement receive a No intention label, whereas active support and stabilization are annotated explicitly. A controlled vocabulary and object-specific part definitions standardize annotation terminology across samples. To ensure annotation fidelity, intention labels are cross-checked against observed and future video frames, rendered 3D hand-object motion, and per-hand action timelines.

Beyond structured intention labels, EgoAIM provides scene annotations specifying the shapes, spatial extents, and poses of furniture support surfaces. We first derive surface heights from the mesh geometry and poses of tracked objects resting on the furniture. Surface boundaries are then manually delineated in multiple RGB frames and back-projected onto the corresponding horizontal planes using calibrated camera parameters to recover surface shapes and dimensions. Subsequently, we position and orient the reconstructed surfaces in each sequence’s gravity-aligned coordinate system and manually refine them through reprojection across frames. Each surface is associated with a furniture label and registered alongside the hand and object trajectories, providing a common spatial reference for relating hand-object motion to the environment.

## IV. GAZE2HOI FRAMEWORK

This section presents the Gaze2HOI framework for forecasting coordinated hand-object interactions from egocentric observations. We first formulate the task in Part A and then describe how the gaze-aware vision-language model and Anticipatory Contact Grounding provide semantic and geometric conditions for motion diffusion in Parts B, C, and D.

## A. Problem Formulation

Given a history of egocentric observations together with scene information, Gaze2HOI infers structured future intentions for each hand and jointly forecasts articulated hand motion and the 6-DoF trajectories of interacting objects. The overall forecasting task is formulated as

$$
( \hat { \mathcal { T } } , \hat { \mathcal { X } } _ { \mathrm { f u t } } ) = f ( \mathcal { V } _ { \mathrm { o b s } } , \mathcal { G } _ { \mathrm { o b s } } , \mathcal { X } _ { \mathrm { o b s } } , \mathcal { P } , \mathcal { S } ) ,\tag{1}
$$

where f denotes the overall prediction pipeline. $\gamma _ { \mathrm { o b s } } , \mathcal { G } _ { \mathrm { o b s } } ,$ and $\mathcal { X } _ { \mathrm { o b s } }$ represent observed RGB frames, gaze cues, and 3D hand-object states, respectively. The task prompt P contains forecasting instructions, scene-specific descriptions, and output requirements. Scene information S includes object point clouds and support surfaces. The outputs are structured perhand intentions I<sup>ˆ</sup> and a future state sequence $\hat { \mathcal { X } } _ { \mathrm { f u t } } \in \mathbb { R } ^ { T _ { \mathrm { f u t } } \times 6 6 }$ Each state is formed by concatenating two 24D hand states (comprising wrist translation, 6D rotation, and 15 MANO [47] PCA pose coefficients) with two 9D object states (comprising translation and 6D rotation). Following EgoAIM, we observe $T _ { \mathrm { o b s } } = 3 0$ frames and predict $T _ { \mathrm { f u t } } = 3 0$ frames at 30 fps.

As illustrated in Fig. 4, Gaze2HOI first performs semantic disambiguation by inferring structured per-hand intentions $\hat { \mathcal { T } } = f _ { \phi } ( \mathcal { V } _ { \mathrm { o b s } } , \mathcal { G } _ { \mathrm { o b s } } , \mathcal { P } )$ . Here, $\hat { \mathcal { Z } } = ( \hat { \mathcal { T } } ^ { L } , \hat { \mathcal { T } } ^ { R } )$ specifies each hand’s intended action, object, part, purpose, and target, with an additional 2D destination for $P u t$ Down. The inferred object identities establish the hand-object bindings, yielding the updated observation sequence $\widetilde { \chi } _ { \mathrm { o b s } } .$ Next, the ACG module performs geometric disambiguation by combining the inferred intentions, object geometry, and observed interactions to predict contact, support, and placement fields, collectively denoted by $\hat { \mathcal { F } } _ { \mathrm { A C G } }$ . Finally, leveraging these semantic and geometric predictions, the conditional diffusion generator $f _ { \omega }$ jointly forecasts future hand-object motion as $\hat { \mathcal { X } } _ { \mathrm { f u t } } = f _ { \omega } ( \widetilde { \mathcal { X } } _ { \mathrm { o b s } } , \hat { \mathcal { T } } , \hat { \mathcal { F } } _ { \mathrm { A C G } } , \mathcal { S } )$

## B. Gaze-Aware Vision-Language Model

Egocentric observations often support multiple plausible interactions before the user’s intention becomes evident from hand kinematics, reflecting semantic ambiguity in intention inference. While gaze provides anticipatory cues for object manipulation [19], individual gaze samples vary in stability and temporal relevance. Uniform aggregation fails to explicitly distinguish stable fixations from rapid gaze shifts or capture their evolving relevance as interactions progress. To support semantic disambiguation, we augment the Qwen3-VL [20] backbone with Temporal Gaze Saliency Modulation (TGSM), which encodes current and historical gaze into distinct spatial channels and applies gated feature-wise modulation to the corresponding visual tokens.

For each input RGB frame at time t, we reproject the available 3D gaze points from the observation window into its image coordinates using the corresponding camera pose. Let $\overline { { \mathbf { g } } } _ { t ^ { \prime }  t }$ denote the projected gaze location from time $t ^ { \prime } ,$ , and $\mathcal { T } _ { t } ^ { \mathrm { v a l i d } }$ represent the set of valid gaze sample indices up to and including t. Using a Gaussian kernel $K _ { \sigma _ { \mathrm { g } } } ( { \bf d } ) \ { = \ } \exp ( - \| { \bf d } \| _ { 2 } ^ { 2 } / ( 2 \sigma _ { \mathrm { g } } ^ { 2 } ) )$ , we construct historical and current saliency maps:

$$
A _ { t } ^ { \mathrm { h i s t } } ( \mathbf { u } ) = \operatorname* { m a x } _ { t ^ { \prime } \in \mathcal { T } _ { t } ^ { \mathrm { v a l i d } } } [ w _ { t ^ { \prime } , t } K _ { \sigma _ { \mathrm { g } } } ( \mathbf { u } - \overline { { \mathbf { g } } } _ { t ^ { \prime }  t } ) ] ,\tag{2}
$$

$$
\begin{array} { r } { A _ { t } ^ { \mathrm { c u r } } ( \mathbf { u } ) = K _ { \sigma _ { \mathrm { g } } } ( \mathbf { u } - \overline { { \mathbf { g } } } _ { t  t } ) , } \end{array}\tag{3}
$$

where u denotes a grid coordinate and $\sigma _ { \mathrm { g } }$ controls the kernel’s spatial spread. The weight $w _ { t ^ { \prime } , t } ~ = ~ q _ { t ^ { \prime } , t } \exp ( - \lambda _ { \mathrm { t i m e } } \Delta _ { t ^ { \prime } , t } )$ combines fixation strength and recency to emphasize stable, recent gaze cues. We derive the fixation weight $q _ { t ^ { \prime } , t }$ from the complementary empirical cumulative distribution function (CDF) of projected gaze speed with a positive floor, assigning greater weight to stable fixations. The temporal decay reduces the influence of older gaze samples, with $\Delta _ { t ^ { \prime } , t }$ denoting elapsed time in seconds and $\lambda _ { \mathrm { t i m e } } > 0$ controlling the decay rate. To handle missing data, we set the current map to zero when the current gaze point is invalid, and the historical map to zero when no valid samples are available. Together, these two channels encode both the instantaneous gaze focus and its spatially aligned, temporally weighted historical context.

To condition the visual features, we spatially align the two saliency maps with the visual token grid via adaptive average pooling, yielding a two-channel gaze descriptor $\mathbf { a } \in \mathbb { R } ^ { 2 }$ for each visual token $\textbf { e } \in \mathbb { R } ^ { d _ { \mathrm { v } } }$ . Two bias-free MLPs, $\gamma _ { \phi }$ and $\mathbf { b } _ { \phi } .$ transform this descriptor into channel-wise scale and shift vectors, allowing spatially corresponding gaze cues to adjust each token’s visual representation. The resulting gated featurewise modulation is defined as

$$
\begin{array} { c } { { { \widetilde { \bf e } } = { \bf e } \odot \left[ { \bf 1 } + \lambda _ { \mathrm { m o d } } \operatorname { t a n h } ( \rho _ { \mathrm { s c a l e } } ) \operatorname { t a n h } ( \gamma _ { \phi } ( { \bf a } ) ) \right] } } \\ { { { \mathrm { } } } + \operatorname { t a n h } ( \rho _ { \mathrm { s h i f t } } ) { \bf b } _ { \phi } ( { \bf a } ) , } \end{array}\tag{4}
$$

where $\odot$ denotes element-wise multiplication, and $\rho _ { \mathrm { s c a l e } }$ and $\rho _ { \mathrm { s h i f t } }$ are learnable scalar gates controlling the strength of scale and shift modulation, respectively. The coefficient $\lambda _ { \mathrm { m o d } } = 0 . 1$ bounds the multiplicative deviation from the original features. The projection networks are shared between the final visual embeddings and the intermediate visual features supplied to the language model. With bias-free projections, a zero gaze descriptor leaves the original visual token unchanged.

Conditioned on the gaze-modulated visual representations Ee and task prompt P, the language model autoregressively predicts structured per-hand intentions. For a serialized groundtruth response $\mathbf { y } = \left( y _ { 1 } , \ldots , y _ { N _ { \mathrm { a n s } } } \right)$ , the training objective is

$$
\mathcal { L } _ { \mathrm { i n t e n t } } = - \frac { 1 } { N _ { \mathrm { a n s } } } \sum _ { m = 1 } ^ { N _ { \mathrm { a n s } } } \log p _ { \phi } \Big ( y _ { m } \mid \mathbf { y } _ { < m } , \widetilde { \mathbf { E } } , \mathcal { P } \Big ) ,\tag{5}
$$

where $N _ { \mathrm { a n s } }$ is the number of supervised response tokens and $\mathbf { y } _ { < m }$ denotes the ground-truth prefix used for teacher forcing. To preserve pretrained knowledge, we freeze the backbone weights and optimize only the TGSM modules alongside low-rank adaptation (LoRA) [48] parameters in the language model. At inference, the decoded response is parsed into structured per-hand intentions ${ \hat { \mathcal { I } } } ,$ facilitating semantic disambiguation and providing explicit guidance for downstream instance binding and anticipatory contact grounding.

## C. Anticipatory Contact Grounding

While structured intentions identify the target object and active part, they leave the precise spatial contact configuration underdetermined. To support geometric disambiguation, Anticipatory Contact Grounding (ACG) integrates semantic priors, observed contact, and hand trajectories to predict future contact distributions. For each valid hand-object binding, a contact prior $\pi _ { \mathrm { p r i o r } }$ is retrieved from a precomputed contact dictionary using the inferred object and part identities and the hand side. We represent all contact distributions on a fixed set of surface points in the bound object’s local coordinate system (omitting the hand index below for clarity). To construct the complementary observed contact evidence $\pi _ { \mathrm { o b s } }$ , we convert distances from object surface points to the hand mesh into normalized contact maps using a truncated Gaussian kernel at each valid contact frame. These maps are averaged over observed contact frames, with zero evidence assigned when no valid contact is available. Functionally, the semantic prior provides anticipatory spatial guidance even before physical contact occurs, while the observed evidence grounds the prediction in the ongoing interaction.

We first encode each surface point via a pointwise MLP using its coordinates, density descriptors of the prior and observed contact, along with its distance to the last valid wrist position. After global maximum and average pooling, these summarized features are combined with an MLP-encoded wrist trajectory and source reliability and validity cues to form a shared context representation. This shared context determines the fusion weights, while its combination with local point features produces a pointwise correction $\delta _ { i }$ . The future contact field is obtained through gated log-linear fusion:

$$
\ell _ { i } = w _ { \mathrm { p r i o r } } \log \pi _ { \mathrm { p r i o r } , i } + w _ { \mathrm { o b s } } \log \pi _ { \mathrm { o b s } , i } + \delta _ { i } ,\tag{6}
$$

$$
\hat { \pi } _ { \mathrm { c o n t a c t } } = \operatorname { s o f t m a x } ( \ell ) ,\tag{7}
$$

where i indexes object surface points. The independent gates $w _ { \mathrm { p r i o r } } , w _ { \mathrm { o b s } } ~ \geq ~ 0$ adjust each source’s contribution for the current interaction, while $\delta _ { i }$ provides a local correction conditioned on geometry and the observed trajectory. Following the log-linear fusion, a softmax operation normalizes the resulting logits into a valid contact probability distribution.

For Put Down, ACG further predicts the object regions expected to contact the supporting surface and their spatial footprint around the placement goal. The support head first refines an object-specific prior $\pi _ { \mathrm { s u p } } ^ { \mathrm { p r i o r } }$ (estimated from training placements) using the shared context alongside local geometric and contact cues. With a context-dependent gate $w _ { \mathrm { s u p } } \geq 0$ and pointwise corrections $\delta _ { \mathrm { s u p } } ,$ the support field is computed as $\begin{array} { r } { \hat { \pi } _ { \mathrm { s u p } } = \mathrm { s o f t m a x } ( w _ { \mathrm { s u p } } \log \pi _ { \mathrm { s u p } } ^ { \mathrm { p r i o r } } + \delta _ { \mathrm { s u p } } ) } \end{array}$ , where softmax normalizes over object surface points. To ground the spatial footprint, we lift the predicted 2D destination to a 3D goal by intersecting its camera ray with a scene support plane, establishing the center of a surface-aligned local grid. The placement head forms $\mathbf { z } _ { \mathrm { p l c } }$ from shared context, pooled support features, and surface-relative object heading. Finally, a convolutional decoder $D _ { \psi }$ processes this representation to yield the placement field $\hat { \pi } _ { \mathrm { p l c } } = \mathrm { s o f t m a x } ( D _ { \psi } ( \mathbf { z } _ { \mathrm { p l c } } ) )$ , which is normalized over the local grid cells.

ACG is trained with annotated intentions and geometric targets derived from future interactions. We construct the contact target by averaging normalized contact maps over valid future contact frames selected according to the interaction phase. For Put Down, the support target normalizes truncated Gaussian weights computed from surface-point distances to the support plane at the terminal object pose. The placement target is obtained by projecting these points onto the local grid, accumulating their support weights with Gaussian kernels, and normalizing the result. For the field types $\mathcal { I } =$ {contact, sup, plc}, the training objective is

$$
\mathcal { L } _ { \mathrm { A C G } } = \sum _ { j \in \mathcal { I } } \lambda _ { j } \big \langle D _ { \mathrm { K L } } ( \pmb { \pi } _ { j } ^ { * } \| \hat { \pmb { \pi } } _ { j } ) \big \rangle _ { j } ,\tag{8}
$$

where $\pi _ { j } ^ { * }$ denotes the target distribution and $\lambda _ { j }$ weights the corresponding loss. The operator $\langle \cdot \rangle _ { j }$ averages over samples with valid supervision for field $j$ in the minibatch, with zero contribution when none are available. Support and placement losses apply only to valid Put Down examples. After training, the frozen ACG predicts geometric fields $\hat { \mathcal { F } } _ { \mathrm { A C G } }$ , which mitigate geometric ambiguity and provide explicit spatial conditioning for the subsequent motion diffusion model.

## D. Conditional Motion Diffusion

Structured intentions and anticipatory contact fields constrain the semantics and geometry of an interaction, but do not uniquely determine its temporal evolution. To model this remaining uncertainty, we employ a conditional motion diffusion model to jointly generate articulated hand motion and object trajectories. The forecasting task is formulated as temporal inpainting: the observed hand-object states remain clean, while only future states are perturbed. Let $\chi _ { \mathrm { f u t } } ^ { ( 0 ) }$ denote the normalized clean future sequence. At diffusion step τ , the noisy future states and the corresponding v-prediction target [49] are defined as

$$
\mathcal { X } _ { \mathrm { f u t } } ^ { ( \tau ) } = \alpha _ { \tau } \mathcal { X } _ { \mathrm { f u t } } ^ { ( 0 ) } + \sigma _ { \tau } \epsilon ,\tag{9}
$$

$$
\mathbf { v } _ { \mathrm { f u t } } ^ { ( \tau ) } = \alpha _ { \tau } \epsilon - \sigma _ { \tau } \boldsymbol { \mathcal { X } } _ { \mathrm { f u t } } ^ { ( 0 ) } ,\tag{10}
$$

where ϵ is standard Gaussian noise, and the noise-schedule coefficients satisfy $\alpha _ { \tau } ^ { 2 } + \sigma _ { \tau } ^ { 2 } = 1$ . The denoiser receives the clean history concatenated with the noisy future, allowing observed interactions to constrain the entire predicted sequence.

Semantic and geometric guidance is incorporated through two complementary mechanisms: a token memory M provides structured cues through cross-attention, while a global conditioning vector c modulates the denoising sublayers. The memory integrates per-hand intention attributes with scene and goal information. To preserve the spatial structure of ACG predictions, we partition each bound object’s surface into local regions and construct region tokens from local geometry and aggregated contact and support probability mass. These region tokens supplement the semantic and scene representations in M. In parallel, an MLP combines embeddings of the diffusion step τ and MANO shape parameters $\beta$ with pooled intention and goal representations to produce c.

To jointly model temporal evolution and hand-object coordination, the denoiser represents each frame using four entity tokens corresponding to the two hands and their associated objects. Raw states are mapped into a shared latent space and augmented with embeddings indicating frame position, entity identity, and observation validity. Each DiT block processes these tokens through temporal attention (across frames per entity) and intra-frame entity attention (among the four tokens), followed by cross-attention [50] to M and a feed-forward network. For input tokens Y, each residual sublayer performs the AdaLN-Zero [51] modulated update $\mathbf { Y } ^ { + } = \mathbf { Y } + \rho _ { \mathbf { c } } \odot { \mathcal { A } } { \big ( } ( \mathbf { 1 } + \gamma _ { \mathbf { c } } ) \odot \mathrm { L N } ( \mathbf { Y } ) + \mathbf { b } _ { \mathbf { c } } { \big ) }$ , where LN is layer normalization and A denotes the respective attention or feed-forward operation. The scale, shift, and residual-gate vectors $\{ \gamma _ { \mathbf { c } } , \mathbf { b _ { c } } , \rho _ { \mathbf { c } } \}$ are generated from c separately for each sublayer and shared across tokens. The modulation projection is initialized to zero, making each residual sublayer an identity mapping at initialization. After L blocks and a final adaptive normalization, output projections yield the joint v-prediction, with its future portion denoted as $\mathbf { \hat { v } } _ { \mathrm { f u t } } ^ { ( \tau ) }$

During motion training, annotated intentions and fields predicted by the frozen ACG model provide semantic and geometric conditions. The denoising objective penalizes discrepancies between predicted and target v values, with supervision restricted to valid future state components:

TABLE II  
QUANTITATIVE COMPARISON OF HAND-OBJECT MOTION FORECASTING ON EGOAIM.
<table><tr><td rowspan="2">Method</td><td colspan="3">Overall</td><td>Establish Contact</td><td>Move</td><td>Manipulate</td><td>Put Down</td></tr><tr><td>H-ADE / FDE↓</td><td>O-ADE / FDE↓</td><td>HO-Pen↓</td><td>Reach-SR↑</td><td>Goal-DE↓</td><td>RA-MPJPE↓</td><td>Place-Hit↑</td></tr><tr><td>USST [10]</td><td> $5 . 9 0 \ / \ 7 . 9 7 $ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EgoH4* [14]</td><td> $1 0 . 1 9 \ / \ 1 1 . 6 0$ </td><td></td><td></td><td>38.82</td><td></td><td>3.92</td><td></td></tr><tr><td>ObjectForesight [15]</td><td></td><td> $0 . 9 0 \ / \ 1 . 0 9$ </td><td></td><td></td><td>3.79</td><td></td><td>42.29</td></tr><tr><td>MDM† [31]</td><td> $1 3 . 1 3 ~ / ~ 1 4 . 6 2$ </td><td> $2 2 . 3 3 \ : / \ : 2 2 . 9 5$ </td><td>2.44</td><td>27.80</td><td>11.25</td><td>3.59</td><td>0.44</td></tr><tr><td>CAHMP† [17]</td><td> $6 . 3 4 \ : / \ : 9 . 8 0$ </td><td> $2 . 1 7 \ : / \ : 3 . 4 8$ </td><td>10.20</td><td>47.70</td><td>7.51</td><td>1.81</td><td>14.10</td></tr><tr><td>Transformer [50]</td><td> $6 . 4 3 \ / \ 8 . 5 0$ </td><td> $2 . 3 6 ~ / ~ 2 . 9 1$ </td><td>13.47</td><td>38.98</td><td>7.54</td><td>1.51</td><td>11.45</td></tr><tr><td>GRU Predictor [30]</td><td> $6 . 7 3 \ / \ 8 . 9 5$ </td><td> $2 . 4 1 \ \mathrm { ~ / ~ } \ 3 . 0 8 $ </td><td>14.90</td><td>40.30</td><td>7.91</td><td>1.61</td><td>13.66</td></tr><tr><td>Gaze2HOI (ours)</td><td>4.16  /  5.49</td><td>0.72  / 0.87</td><td>1.70</td><td>92.43</td><td>2.29</td><td>1.33</td><td>44.93</td></tr></table>

<sup>∗</sup>Task adaptation and MANO fitting for Reach-SR. <sup>†</sup>Adapted for joint hand-object forecasting. <sup>‡</sup>Baselines built on the cited architectures. —: unsupported. Bold denotes the bes result. Best of five is selected per test window and metric (min for ↓, max for ↑).

$$
\mathcal { L } _ { v } = \frac { \left\| \mathbf { W } _ { \mathrm { f u t } } \odot \left( \hat { \mathbf { v } } _ { \mathrm { f u t } } ^ { ( \tau ) } - \mathbf { v } _ { \mathrm { f u t } } ^ { ( \tau ) } \right) \right\| _ { F } ^ { 2 } } { \operatorname* { m a x } ( 1 , \| \mathbf { W } _ { \mathrm { f u t } } \| _ { 1 } ) } ,\tag{11}
$$

where $\mathbf { W } _ { \mathrm { f u t } }$ is a binary mask identifying valid future state components across the minibatch, and $\| \cdot \| _ { F }$ and $\| \cdot \| _ { 1 }$ denote the Frobenius and entrywise $L _ { 1 }$ norms, respectively. To encourage coordinated and physically plausible interactions, we apply auxiliary losses to denormalized clean-state estimates recovered from the predicted v. The overall objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m o t i o n } } = \mathcal { L } _ { v } + \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { r e l } } + \mathcal { L } _ { \mathrm { c o n t a c t } } } \\ { + \mathcal { L } _ { \mathrm { o b j } } + \mathcal { L } _ { \mathrm { s u p } } + \mathcal { L } _ { \mathrm { m o v e } } , \quad } \end{array}\tag{12}
$$

where $\mathcal { L } _ { \mathrm { g e o } }$ supervises translations, velocities, and rotations, while $\mathcal { L } _ { \mathrm { r e l } }$ constrains relative hand-object motion by matching wrist-relative poses and their temporal changes. Contact consistency and collision avoidance are encouraged through $\mathcal { L } _ { \mathrm { { c o n t a c t } } }$ , which combines regional contact-distance matching with hand-object penetration penalties, and $\mathcal { L } _ { \mathrm { o b j } }$ which penalizes object-object penetration. The support term $\mathcal { L } _ { \mathrm { s u p } }$ addresses furniture penetration, terminal placement, and placement-goal errors, while $\mathcal { L } _ { \mathrm { m o v e } }$ supervises object transport relative to the intended target. Each auxiliary term uses its corresponding validity mask. To avoid imposing interaction constraints on unreliable clean-state estimates, the interaction losses are activated only when $\alpha _ { \tau } ^ { 2 } \geq 0 . 2 5$ , whereas $\mathcal { L } _ { \mathrm { g e o } }$ is evaluated at all noise levels.

Intention and goal conditions are randomly dropped during training to enable classifier-free guidance (CFG) [52]. At inference, DDIM sampling [53] with CFG generates future motion conditioned on VLM-predicted intentions and the corresponding ACG fields. Future states are initialized with Gaussian noise, while observed states are restored after each sampling step. The resulting future sequence is denormalized to recover articulated hand motion and 6-DoF object trajectories. Ultimately, by coupling structured semantic intentions and explicit geometric grounding with stochastic motion generation, our unified framework substantially mitigates the core challenges of semantic and geometric ambiguity, yielding physically plausible and temporally coherent interaction forecasts.

## V. EXPERIMENTS

This section presents a systematic evaluation of the Gaze2HOI framework on EgoAIM through quantitative and qualitative analyses. These analyses evaluate end-to-end interaction forecasting across phases, analyze the semantic disambiguation, and visualize the geometric disambiguation. Further experiments examine the effect of intention conditioning on motion forecasting, while ablation studies quantify the contributions of key components.

## A. Implementation Details

The EgoAIM dataset is partitioned into 5,132 training, 976 validation, and 1,321 test windows, ensuring strict temporal separation with no frame overlap between the training and test sets. Each window comprises 30 observed and 30 future frames at 30 fps, corresponding to one second of observation and one second of forecasting (Fig. 3). All methods are evaluated on the same test set, with overall results computed on the full set and phase-specific results on corresponding subsets.

Training is conducted on four NVIDIA RTX A4500 GPUs (with 20GB VRAM each). Gaze2HOI employs Qwen3-VL-8B-Instruct for intention inference and an 8-block DiT with a hidden dimension of 384 for motion forecasting. Motion generation utilizes 50-step DDIM sampling with a classifierfree guidance scale of 1.5. For stochastic methods, metric-wise best-of-five results are reported using fixed sampling seeds, whereas deterministic methods retain a single prediction.

## B. Evaluation of Egocentric HOI Forecasting

1) Baselines and Metrics: We benchmark Gaze2HOI against USST [10] and EgoH4 [14] for hand forecasting, ObjectForesight [15] for object forecasting, alongside MDM [31] and CAHMP [17] adapted for joint hand-object forecasting. Two additional baselines, GRU Predictor [30] and Transformer [50], directly predict joint hand-object states from observed motion and gaze. All methods are configured for the identical 30-to-30-frame forecasting task based on their supported modalities. Omitted metrics due to incompatible outputs are denoted by an em dash.

Overall performance is quantified via average and final displacement errors for hand wrists (H-ADE/FDE) and object translations (O-ADE/FDE), alongside hand-object penetration (HO-Pen). ADE averages position errors over valid future frames, whereas FDE evaluates the final valid frame. Object errors are averaged equally over all distinct scene objects with valid observed poses; objects outside Gaze2HOI’s predicted bindings retain their last observed poses. HO-Pen calculates the percentage of valid future frames where either hand penetrates a scene object by a margin exceeding 1.5 cm. Distance errors are reported in centimeters and rates in percentages.

![](images/a4468b2b62cc9bf6273a895caff27d4da0c737f54255ff43f66170f8bee26a8a.jpg)  
Fig. 5. Qualitative comparison of hand-object interaction forecasting on EgoAIM. From top to bottom, the rows correspond to Establish Contact, Move, Manipulate, and Put Down. Observed inputs and ground-truth (GT) futures are shown alongside predictions from representative baselines and Gaze2HOI.

Each interaction phase is further assessed using a dedicated metric. For Establish Contact, Reach-SR is the percentage of reaches for which any predicted hand vertex comes within 2 cm of the ground-truth task-object surface. For Move, Goal-DE quantifies the absolute error of the terminal object-to-target distance, with the target fixed at its last observed position. For Manipulate, RA-MPJPE computes the mean 3D joint error after wrist translation alignment. For Put Down, Place-Hit measures the percentage of task objects whose predicted endpoint lies within 5 cm of the ground-truth endpoint. These metrics are computed on the corresponding phase subsets using annotated active hands or task objects.

2) Quantitative Results: As detailed in Table II, Gaze2HOI achieves the best reported results across all metrics for joint hand-object forecasting. Relative to USST, it reduces H-ADE and H-FDE by 29.5% and 31.1%, respectively, while reducing O-ADE and O-FDE by 20.0% and 20.2% relative to Object-Foresight. HO-Pen decreases to 1.70%, compared with 2.44% for MDM. Despite its comparatively low penetration rate, the MDM adaptation exhibits substantially larger trajectory errors, underscoring the necessity of jointly evaluating geometric accuracy and physical plausibility.

The phase-specific results show a particularly large improvement in Establish Contact, where Reach-SR reaches 92.43%, exceeding CAHMP by 44.73 percentage points. For Move and Manipulate, Gaze2HOI achieves a Goal-DE of 2.29 cm and an RA-MPJPE of 1.33 cm, reducing the errors of ObjectForesight and Transformer by 39.6% and 11.9%, respectively. The improvement in Put Down is more modest: Place-Hit increases from ObjectForesight’s 42.29% to 44.93%, yielding a gain of 2.64 percentage points. The remaining endpoint errors identify precise placement as an area for further improvement.

3) Qualitative Results: The qualitative comparisons in Fig. 5 detail the representative failure modes of existing methods. Several baselines capture coarse motion trends but exhibit errors in contact configuration, motion extent, or object orientation. The moka pot example illustrates that plausible object poses alone do not ensure goal-directed interaction: ObjectForesight fails to position it above the cup. Transformer exhibits a related shortcoming, predicting insufficient spatula movement toward the plate. Beyond reaching the target region, successful interaction requires appropriate contact configurations and object orientations. CAHMP approaches the toy house with inaccurate finger articulation near the contact surface, while ObjectForesight transports the pitcher with excessive tilt. The highlighted cup reveals another issue: CAHMP predicts drift despite the object remaining nearly stationary in the ground truth. These examples underscore the importance of jointly capturing goal-directed motion, stable hand-object relationships, and scene consistency. Gaze2HOI overcomes these limitations, generating coherent hand-object interactions with coordinated articulation and object motion.

![](images/119769fc2a0ab07590a2a4633d72c9e1801bfacd2f36611c01fe11e49ee31b2b.jpg)  
Fig. 6. Illustration of gaze injection strategies. Each panel shows the image and corresponding prompt for the same egocentric observation. The TGSM overlay is for visualization only; gaze cues are injected through visual feature modulation.

TABLE III  
COMPARISON OF GAZE INJECTION STRATEGIES.
<table><tr><td>Method</td><td>Active-hand EM (%)↑</td><td>Median Point Error (px) ↓</td><td>Sample EM@100px (%)↑</td></tr><tr><td>No Gaze</td><td>81.76</td><td>51.2</td><td>72.67</td></tr><tr><td>Gaze Text</td><td>83.95</td><td>38.0</td><td>77.44</td></tr><tr><td>Gaze Marker</td><td>86.20</td><td>39.2</td><td>80.85</td></tr><tr><td>Gaze Trace</td><td>85.05</td><td>40.2</td><td>78.35</td></tr><tr><td>Gaze Spotlight</td><td>83.22</td><td>50.7</td><td>76.00</td></tr><tr><td>TGSM (ours)</td><td>87.84</td><td>35.2</td><td>82.29</td></tr></table>

## C. Analysis of Semantic Disambiguation

1) Evaluation Protocol: To quantify semantic disambiguation, we evaluate the accuracy of structured intention inference using three complementary metrics: Active-hand Exact Match (EM), Median Point Error, and Sample EM@100px. Activehand EM measures the percentage of annotated active hands for which all categorical intention attributes are correctly predicted. Median Point Error reports the median Euclidean localization error in the original 1408 × 1408 image space, considering only cases with both an annotated and a predicted destination. Sample EM@100px measures the percentage of samples with correct categorical predictions for both hands and destination errors within 100 pixels. This sample-level metric also requires correct predictions of inactive hands and absent destination points. Together, these metrics assess semantic accuracy, spatial precision, and complete intention prediction.

2) Gaze Injection Strategies: We compare six gaze injection strategies using the same Qwen3-VL-8B backbone, data splits, and training schedule. As illustrated in Fig. 6, No Gaze processes unmodified observations, whereas Gaze Text provides gaze coordinates and Gaze Marker overlays the current gaze location. Gaze Trace encodes temporal recency and fixation strength through trajectory width and color, respectively. Gaze Spotlight uses these cues to modulate spatial brightness. These four gaze baselines operate at the input level. In contrast, TGSM incorporates temporal gaze saliency via visual feature modulation; its overlay in Fig. 6 is provided solely to visualize the gaze cues used for conditioning.

As shown in Table III, TGSM achieves the best performance across all three metrics, reaching 87.84% Active-hand EM and 82.29% Sample EM@100px, alongside a median point error of 35.2 pixels. Compared with No Gaze, TGSM improves

TABLE IV  
INTENTION INFERENCE PERFORMANCE ACROSS VLMS.
<table><tr><td>Model</td><td>Gaze Injection</td><td>Sample EM@100px (%)↑</td></tr><tr><td>GPT-5.4</td><td>Gaze Marker</td><td>39.21</td></tr><tr><td>Gemini 3.1 Pro</td><td>Gaze Marker</td><td>25.66</td></tr><tr><td>Qwen2-VL-2B</td><td>TGSM</td><td>68.81</td></tr><tr><td>Qwen2-VL-7B</td><td>TGSM</td><td>77.97</td></tr><tr><td>Qwen3-VL-2B</td><td>TGSM</td><td>69.04</td></tr><tr><td>Qwen3-VL-4B</td><td>TGSM</td><td>73.81</td></tr><tr><td>Qwen3-VL-8B (ours)</td><td>TGSM</td><td>82.29</td></tr></table>

Sample EM@100px by 9.62 percentage points. Gaze Marker achieves the highest Sample EM@100px among the inputlevel strategies, but remains 1.44 percentage points below TGSM. The suboptimal scores of Gaze Trace and Gaze Spotlight relative to Gaze Marker suggest that explicitly rendering temporal gaze information at the input level does not necessarily improve intention inference. These results validate TGSM as an effective means of incorporating temporal gaze cues while preserving the original visual observations.

3) Comparison across VLMs: We compare two commercial VLMs with five Qwen configurations using Sample EM@100px on the same test set. GPT-5.4 and Gemini 3.1 Pro receive Gaze Marker inputs without task-specific fine-tuning, while the Qwen models are fine-tuned with TGSM. All models are instructed to produce the same structured output format.

As reported in Table IV, GPT-5.4 and Gemini 3.1 Pro achieve 39.21% and 25.66%, respectively, while the taskadapted Qwen models attain substantially higher accuracy. Within the Qwen3-VL family, performance improves consistently with model size, increasing from 69.04% at 2B to 73.81% at 4B and 82.29% at 8B. Qwen2-VL-7B achieves 77.97%, outperforming the smaller Qwen3-VL variants but trailing Qwen3-VL-8B by 4.32 percentage points. These results support the selection of Qwen3-VL-8B as the intention inference backbone of Gaze2HOI.

## D. Visualization of Geometric Disambiguation

We qualitatively examine how ACG uses observed interactions to refine contact priors, thereby achieving geometric disambiguation in future contact localization. As illustrated in Fig. 7, the contact priors identify plausible contact regions associated with the intended object and part, while ACG further resolves their spatial extent using the observed handobject states and hand trajectories. For the eyeglasses, the predicted contact probability becomes concentrated on one temple. For the bowl, masher, and pitcher, ACG narrows the broader rim and handle priors to localized regions relevant to the ongoing interaction. These examples illustrate how observed interaction cues complement contact priors, enabling more precise contact localization and providing finer geometric guidance for subsequent motion forecasting.

![](images/909d15eb88b2d3f9fe2118da96fca4f12c306f59482897c4c7f3159c9f6dd571.jpg)  
Fig. 7. Visualization of geometric disambiguation via ACG. ACG predicts future contact regions using contact priors and observed interactions. Warmer colors indicate higher relative contact intensity within each map.

TABLE V  
MOTION FORECASTING ACROSS INTENTION CONDITIONS.
<table><tr><td>Intention</td><td>H-ADE↓</td><td>O-ADE↓</td><td>Reach-SR↑</td><td></td><td>Place-Hit↑</td></tr><tr><td>None</td><td>6.49</td><td></td><td>1.05</td><td>56.91</td><td>18.94</td></tr><tr><td>Predicted</td><td>4.16 -35.9%</td><td></td><td> $0 . 7 2 \mathrm { ~ - } 3 1 . 4 \%$ </td><td></td><td>92.43+35.5pp44.93+26.0pp</td></tr><tr><td>Oracle</td><td> $3 . 6 5 \ _ { - 4 3 . 8 \% }$ </td><td></td><td> $0 . 5 5 \mathrm { ~ - 4 7 . 6 \% }$ </td><td> $9 6 . 2 2 + 3 9 . 3 \mathrm { p p }$ </td><td> $9 5 . 5 9 \substack { + 7 6 . 7 \mathrm { p p } }$ </td></tr></table>

Changes vs. None: relative % for ADE; percentage points (pp) for success rates.

## E. Effectiveness of Intention in Motion Forecasting

To assess the value of inferred intentions for downstream motion forecasting, we compare three conditions: no explicit intentions or destinations (None), predicted intentions and destinations (Predicted), and ground-truth annotations (Oracle). The motion model and ACG weights remain fixed across all settings. For None, ACG provides contact guidance from observed interactions without using explicit intention or destination information. All settings follow the same metricwise best-of-five evaluation protocol.

As shown in Table V, predicted intentions substantially improve both trajectory accuracy and interaction success over intention-free conditioning. Relative to None, Predicted reduces H-ADE and O-ADE by 35.9% and 31.4%, respectively, demonstrating benefits for both hand and object forecasting. These trajectory improvements are accompanied by a 35.5- percentage-point increase in Reach-SR, which reaches 92.43% and approaches the oracle result of 96.22%. Placement also benefits considerably, with Place-Hit increasing from 18.94% to 44.93%; however, the remaining gap to the oracle result of 95.59% indicates that precise placement remains challenging under predicted intention conditioning.

TABLE VI  
END-TO-END ABLATION STUDY OF GAZE2HOI.
<table><tr><td>Setting</td><td>H-ADE↓</td><td>O-ADE↓</td><td>Reach-SR ↑</td><td>Place-Hit↑</td></tr><tr><td>w/o TGSM</td><td>4.32</td><td>0.79</td><td>91.61</td><td>33.48</td></tr><tr><td>w/o Gaze</td><td>4.65</td><td>0.87</td><td>81.25</td><td>26.87</td></tr><tr><td>w/o ACG</td><td>4.35</td><td>0.78</td><td>88.82</td><td>33.48</td></tr><tr><td>Full Model</td><td>4.16</td><td>0.72</td><td>92.43</td><td>44.93</td></tr></table>

w/o TGSM replaces TGSM with the input-level Gaze Marker strategy. w/o Gaze excludes gaze information from intention inference. w/o ACG removes all explicit geometric conditioning, including ACG fields and goal encodings.

## F. Ablation Studies

We assess the contributions of gaze information, TGSM, and geometric conditioning to end-to-end forecasting through the ablations in Table VI. The full model achieves the best results across all four reported metrics. Removing gaze causes the largest degradation, reducing Reach-SR and Place-Hit by 11.18 and 18.06 percentage points, respectively. Replacing TGSM with Gaze Marker recovers much of the contactestablishment performance, reaching 91.61% Reach-SR, but yields a Place-Hit of only 33.48%, compared with 44.93% for the full model. These results demonstrate that the benefits of TGSM extend beyond intention inference to the accuracy of downstream interaction outcomes.

Removing explicit geometric conditioning, including ACG fields and goal encodings, also degrades forecasting despite retaining structured intentions and observed motion. H-ADE and O-ADE increase to 4.35 and 0.78 cm, respectively, while Reach-SR falls to 88.82% and Place-Hit to 33.48%. The substantial reduction in placement success, alongside the comparatively modest changes in average displacement error, highlights the importance of geometric and goal information for completing precise interactions. These conditions complement semantic intentions by specifying spatial constraints that high-level intention labels alone leave underdetermined.

## VI. DISCUSSION

Our results highlight the role of structured intentions in guiding hand-object interaction forecasting. The performance gains over the no-intention setting (Table V) demonstrate that inferred task semantics provide effective guidance even when intention predictions are imperfect. ACG complements this information by refining semantic contact priors using observed interaction cues, as visualized in Fig. 7. Together, these findings support integrating semantic reasoning with geometric grounding, with performance under predicted intentions approaching that achieved under oracle conditioning in specific interaction phases.

Interpreting the failure cases in Fig. 8 requires distinguishing true goal misalignment from potentially valid interaction variations. In Fig. 8(a), predicting placing instead of shaking fundamentally alters the intended goal. In Fig. 8(b,c), however, the action and task object are correctly inferred, while the predicted contact part or active hand differs from the annotation. These alternative choices may still serve as viable strategies for the same high-level task. Even for a human observer, the available visual history may not uniquely determine the contact region or executing hand before the interaction unfolds. Consequently, although these predictions are penalized by annotation-based metrics, disagreement with a single recorded future does not necessarily imply failure to capture the underlying interaction goal.

![](images/fc5041e4c6b9b52a17c937ae4144880e3d8bc08232f461c0d4956d4f760156bc.jpg)  
Fig. 8. Failure cases of Gaze2HOI. Examples of (a) purpose confusion, (b) contact-part confusion, and (c) hand assignment errors, with predicted intentions and the corresponding future interactions shown alongside ground truth.

Precise placement remains a substantial challenge. Although predicted intentions improve Place-Hit from 18.94% to 44.93%, the 95.59% achieved under oracle conditioning reveals a substantial performance gap. This gap suggests that precise placement is highly sensitive to inaccuracies in the inferred intention conditions. Future work could model uncertainty in predicted intention attributes and jointly refine spatial goals with contact and support constraints to improve placement accuracy and physical consistency.

## VII. CONCLUSION

In this paper, we presented Gaze2HOI, a hierarchical framework that integrates gaze-aware intention inference, anticipatory contact grounding, and coordinated motion generation for egocentric hand-object interaction forecasting. TGSM incorporates temporal gaze cues into visual representations to infer structured per-hand intentions, while ACG combines contact priors with observed interactions and object geometry to predict contact, support, and placement fields. These semantic and geometric conditions guide a diffusion model that jointly forecasts the articulated motion of both hands and 6-DoF object trajectories. We also introduced EgoAIM, a phase-aware benchmark pairing fine-grained future intentions with continuous 3D hand-object motion, enabling evaluation across four interaction phases. Experiments on EgoAIM demonstrate improvements over the evaluated baselines in both intention inference and interaction forecasting, supporting the complementary contributions of gaze information and geometric grounding. End-to-end performance approaches the levels achieved under oracle conditioning in specific phases, while precise placement remains an important direction for future improvement. By connecting inferred interaction goals with their anticipated spatial realization, this work provides a foundation for advancing proactive human-centric assistance in AR, VR, and embodied systems.

## ACKNOWLEDGMENT

This research was partially funded by the National Natural Science Foundation of China (Grant No. 62306309).

## REFERENCES

[1] D. Han, R. Lee, K. Kim et al., “VR-HandNet: A visually and physically plausible hand manipulation system in virtual reality,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 30, no. 7, pp. 4170–4182, 2024.

[2] D. Driess, F. Xia, M. S. M. Sajjadi et al., “PaLM-E: An embodied multimodal language model,” in Proceedings of the International Conference on Machine Learning (ICML), vol. 202, 2023, pp. 8469–8488.

[3] Y. Zheng, Y. Yang, K. Mo et al., “GIMO: Gaze-informed human motion prediction in context,” in Proceedings of the European Conference on Computer Vision (ECCV), vol. 13673, 2022, pp. 676–694.

[4] Z. Hu, Z. Yin, D. Haeufle et al., “HOIMotion: Forecasting human motion during human-object interactions using egocentric 3D object bounding boxes,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 30, no. 11, pp. 7375–7385, 2024.

[5] G. Lee, M. Xia, N. Numan et al., “Sensible Agent: A framework for unobtrusive interaction with proactive AR agents,” in Proceedings of the Annual ACM Symposium on User Interface Software and Technology (UIST), 2025, pp. 1–22.

[6] T. Jo, H. J. Lee, S. Chun et al., “Manual-free gaze interaction via bayesian-based implicit intention prediction,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 31, no. 12, pp. 10 789–10 800, 2025.

[7] T. Ha, W. Cho, D. Kim et al., “Int3DNet: Scene-motion cross attention network for 3D intention prediction in mixed reality,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 32, no. 5, pp. 4185–4195, 2026.

[8] Y. Li, Z. Cao, A. Liang et al., “Egocentric prediction of action target in 3D,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20 971–20 980.

[9] S. Liu, S. Tripathi, S. Majumdar et al., “Joint hand motion and interaction hotspots prediction from egocentric videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 3272–3282.

[10] W. Bao, L. Chen, L. Zeng et al., “Uncertainty-aware state space transformer for egocentric 3D hand trajectory forecasting,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 13 656–13 665.

[11] Y. Liu, H. Yang, X. Si et al., “TACO: Benchmarking generalizable bimanual Tool-ACtion-Object understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 740–21 751.

[12] A. Furnari and G. M. Farinella, “What would you expect? anticipating egocentric actions with rolling-unrolling LSTMs and modality attention,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 6252–6261.

[13] R.-G. Pasca, A. Gavryushin, M. Hamza et al., “Summarize the past to predict the future: Natural language descriptions of context boost multimodal object interaction anticipation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18 286–18 296.

[14] M. Hatano, Z. Zhu, H. Saito et al., “The invisible EgoHand: 3D hand forecasting through EgoBody pose estimation,” arXiv preprint arXiv:2504.08654, 2025.

[15] R. Soraki, H. Bharadhwaj, A. Farhadi et al., “ObjectForesight: Predicting future 3D object trajectories from human videos,” in Proceedings of the European Conference on Computer Vision (ECCV), vol. 17062, 2026, pp. 437–453.

[16] J. Ma, X. Chen, J. Xu et al., “Diff-IP2D: Diffusion-based hand-object interaction prediction on egocentric videos,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 4291–4298.

[17] E. Corona, A. Pumarola, G. Alenya\` et al., “Context-aware human motion prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 6992–7001.

[18] B. Zuo, Z. Zhao, W. Sun et al., “GraspDiff: Grasping generation for hand-object interaction with multimodal guided diffusion,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 31, no. 9, pp. 5642–5654, 2025.

[19] R. S. Johansson, G. Westling, A. Backstr¨ om¨ et al., “Eye–hand coordination in object manipulation,” The Journal of Neuroscience (J. Neurosci.), vol. 21, no. 17, pp. 6917–6932, 2001.

[20] S. Bai, Y. Cai, R. Chen et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[21] S. Christen, S. Hampali, F. Sener et al., “DiffH2O: Diffusion-based synthesis of hand-object interactions from textual descriptions,” in Proceedings of the ACM SIGGRAPH Asia Conference Papers (SIGGRAPH Asia), 2024, pp. 1–11.

[22] Y. He, X. Zhang, and A. H. A. Stienen, “Gaze-guided 3D hand motion prediction for detecting intent in egocentric grasping tasks,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 14 580–14 586.

[23] K. Grauman, A. Westbury, E. Byrne et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 18 973–18 990.

[24] P. Banerjee, S. Shkodrani, P. Moulon et al., “HOT3D: Hand and object tracking in 3D from egocentric multi-view videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 7061–7071.

[25] E. Dessalene, C. Devaraj, M. Maynord et al., “Forecasting action through contact representations from first person video,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 45, no. 6, pp. 6703–6714, 2023.

[26] M. Liu, S. Tang, Y. Li et al., “Forecasting human-object interaction: Joint prediction of motor attention and actions in first person video,” in Proceedings of the European Conference on Computer Vision (ECCV), vol. 12346, 2020, pp. 704–721.

[27] J. Ma, X. Chen, W. Bao et al., “MADiff: Motion-aware Mamba diffusion models for hand trajectory prediction on egocentric videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 48, no. 3, pp. 3250–3267, 2026.

[28] J. Ma, W. Bao, J. Xu et al., “Uni-Hand: Universal hand motion forecasting in egocentric views,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), pp. 1–18, 2026, early Access.

[29] J. Tong and K. Zhang, “HandDiff: Spatial–temporal diffusion model for hand pose forecasting,” Information Fusion (Inf. Fusion), vol. 127, p. 103647, 2026.

[30] K. Cho, B. van Merrienboer, C. Gulcehre ¨ et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1724– 1734.

[31] G. Tevet, S. Raab, B. Gordon et al., “Human motion diffusion model,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/ forum?id=SJ1kSyO2jwu

[32] X. Hu, X. Qi, B. Yang et al., “A<sup>3</sup>fford-HOI: Anatomy-aligned affordance disentanglement for fine-grained and generalizable hand object

interaction,” in Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2026, pp. 1197–1205.

[33] J. Cha, J. Kim, J. S. Yoon et al., “Text2HOI: Text-guided 3D motion generation for hand-object interaction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 1577–1585.

[34] W. Zhang, R. Dabral, V. Golyanik et al., “BimArt: A unified approach for the synthesis of 3D bimanual interaction with articulated objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 27 694–27 705.

[35] W. He, Y. Liu, R. Liu et al., “SyncDiff: Synchronized motion diffusion for multi-body human-object interaction synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 11 731–11 743.

[36] J. Tian, R. Ji, L. Yang et al., “Gaze-guided hand-object interaction synthesis: Dataset and method,” IEEE Transactions on Multimedia (TMM), pp. 1–11, 2026.

[37] A. Prakash, B. Lundell, D. Andreychuk et al., “How do i do that? synthesizing 3D hand motion and contacts for everyday interactions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 7026–7036.

[38] B. Zhou, Y. Zhan, Z. Zhang et al., “MEgoHand: Multimodal egocentric hand-object interaction motion generation,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 49 464–49 490.

[39] Z. Hu, J. Xu, S. Schmitt et al., “Pose2Gaze: Eye-body coordination during daily activities for gaze prediction from full-body poses,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 31, no. 9, pp. 4655–4666, 2025.

[40] Z. Hu, S. Schmitt, D. Haufle¨ et al., “GazeMotion: Gaze-guided human motion forecasting,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024, pp. 13 017– 13 022.

[41] S. Mondal, N. Sendhilnathan, T. Zhang et al., “Gaze-language alignment for zero-shot prediction of visual search targets from human gaze scanpaths,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 2738–2749.

[42] T. Peng, J. Hua, M. Liu et al., “In the eye of MLLM: Benchmarking egocentric video intent understanding with gaze-guided prompting,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 121 404–121 424.

[43] K. Yan, Z. Wang, L. Ji et al., “Voila-A: Aligning vision-language models with user’s gaze attention,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), vol. 37, 2024, pp. 1890– 1918.

[44] A. Pani and Y. Yang, “Gaze-VLM: Bridging gaze and VLMs through attention regularization for egocentric understanding,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 163 544–163 577.

[45] S. Cheng, Z. Guo, J. Wu et al., “EgoThink: Evaluating first-person perspective thinking capability of vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 14 291–14 302.

[46] T. Kwon, B. Tekin, J. Stuhmer¨ et al., “H2O: Two hands manipulating objects for first person interaction recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10 118–10 128.

[47] J. Romero, D. Tzionas, and M. J. Black, “Embodied hands: Modeling and capturing hands and bodies together,” ACM Transactions on Graphics (TOG), vol. 36, no. 6, pp. 245:1–245:17, 2017.

[48] E. J. Hu, Y. Shen, P. Wallis et al., “LoRA: Low-rank adaptation of large language models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[49] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[50] A. Vaswani, N. Shazeer, N. Parmar et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[51] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4172–4182.

[52] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[53] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.