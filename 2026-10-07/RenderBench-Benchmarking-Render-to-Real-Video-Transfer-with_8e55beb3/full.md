# RenderBench: Benchmarking Render-to-Real Video Transfer with Reconstructed Digital Twins

Dicong Qiu<sup>1,∗</sup> Zhiyuan Xu<sup>2,∗</sup> Yaosheng Liu<sup>3</sup> Feng Han<sup>4</sup> Bo Ye<sup>3</sup> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>Jacobi.ai <sup>3</sup>Southeast University <sup>4</sup>Independent Researcher

dqiu570@hkust-gz.edu.cn zhiyuan.xu@jacobi.ai 220266580@seu.edu.cn Terrancehan25@gmail.com yeb@seu.edu.cn <sup>∗</sup>Equal contribution

![](images/0fe80f8f38deccdb1f5746021fd70a444bf102d6e7b2ae7d3f55fef3dd7c3b56.jpg)  
Seedance F· GENERATION REJECTED BY SEEDANCE. Black card shown; original placeholder video preserved

Figure 1. Qualitative comparison across eight viewpoints and interactions (A–H) in one reconstructed environment. The rows show the real ground truth, aligned proxy render, and outputs from MiniMax H3, Wan2.2-VACE-Fun-A14B, and Seedance 2.5 under matched task prompts. The black card marks a rejected Seedance generation.

## Abstract

Modern video models can synthesize realistic videos from complementary appearance and geometry conditions: real reference images specify how the scene should look, while proxy RGB and geometric control videos depict its struc-

ture, viewpoint changes, and motion. Yet this referenceconditioned render-to-real capability remains difficult to evaluate because its output must be compared with a groundtruth real video depicting the same scene evolution. A rigorous benchmark therefore needs appearance references and target real videos paired with editable, geometrically registered 3D replicas, a resource that has traditionally required prohibitive manual modeling, calibration, and animation effort. Existing evaluations therefore rely on controls estimatedfrom real videos or on small collections ofsynthetic scenes, neither of which provides paired real appearance and editable scene state at scale. We introduce Render-Bench, a benchmark comprising 12reconstructed real-world scenes that span large-scale indoor environments and egocentric viewpoints, with both dynamic and static settings. The scenes are constructed using modern visual geometry, neural reconstruction, and assisted 3D authoring. Each scene is decomposed into static objects and dynamic actors, registered to the capture cameras, and admitted only after multi-view geometric and temporal validation. An accepted unit contains held-out appearance references, a real target video, an editable digital twin, a matched proxy render, and renderer-native scene annotations. We evaluate transfer models against paired real target videos, retain PAI-Bench-C-compatible structural projections, and use scene annotations to localize failures by object, visibility, articulation, and motion. Thefirst release retains 12 of14 registered samples (85.7%), comprising 1,496 paired real–proxy frames; all 12 scenes pass file-integrity and environment-edit audits, while proxy diagnostics yield a depth si-RMSE of 0.2170 and instance mIoU of0.3673.

## 1. Introduction

Computer-graphics simulators offer a compelling source of training data for robotics, autonomous driving, and embodied intelligence. They provide scalable trajectories together with exact geometry, object state, segmentation, depth, and motion. Their central limitation is the synthetic-to-real appearance gap [16]. Recent video generators directly target this limitation by combining two complementary sources of information. One or more real reference images specify the desired appearance, including materials, illumination, and scene style, while a geometry video depicts the scene structure, viewpoint changes, object motion, and articulation to preserve. The geometry condition may be a proxy RGB render or dense render passes such as depth, segmentation, edges, pose, or trajectories. Structured-control systems such as Cosmos-Transfer2.5 [10] and Wan2.2-VACE-Fun-A14B [1] expose these signals explicitly, whereas general multimodal systems such as MiniMax H3 [8] and Seedance 2.5 [2] consume them through reference and editing interfaces. Together, these systems establish referenceconditioned render-to-real transfer as an emerging capability across structured-control and general-purpose video models.

This new capability creates an evaluation problem that existing datasets cannot fully answer. Existing resources cover complementary pieces of the problem, but not the full evaluation loop. CO3D provides nearly 19,000 object-centric real videos with recovered cameras and point clouds, but no independently authored, editable proxy whose render can be used as a transfer input [13]. Aria Digital Twin (ADT) comes closest to paired real–synthetic supervision: its 200 published sequences include calibrated egocentric recordings, a carefully digitized scene, and photorealistic synthetic renderings [12]. However, those sequences span only two instrumented spaces, and their construction relies on highresolution scanning, manual scene digitization, material and lighting tuning, and motion capture; ADT is designed for egocentric 3D perception rather than scalable evaluation from deliberately simplified renderer inputs. AptSim2Rea instead offers approximately paired driving images whose simulated and real scenes may differ in assets, shape, texture, and appearance, making it suitable for distribution-level translation but not same-scene structural verification [22]. Finally, Cosmos TransferBench and PAI-Bench-C evaluate controls extracted from real videos, while the Cosmos Sim2Real case study uses 20 programmatically authored robot scenarios; neither couples those inputs to real captures of the same editable scenes [9, 24]. To determine whether a transfer model merely produces a plausible video or actually reconstructs the intended real-looking scene, evaluation requires four aligned elements: appearance references, a geometry video, a held-out real target video, and an editable 3D scene. The reference images are model inputs that specify the desired real appearance; the geometry video is the model input that depicts the target viewpoint changes and scene evolution without exposing the underlying camera parameters; the temporally matched real video is held out as the appearance ground truth; and the registered editable 3D scene provides renderer-native geometry and dynamics for diagnostic analysis. Together they distinguish successful transfer from three common shortcuts: copying the synthetic input without realification, reproducing a reference image without following the requested motion, and generating an attractive but structurally different video.

Historically, this evaluation unit has been impractical to collect at benchmark scale. Building a faithful, editable 3D replica of an ordinary real scene required skilled artists to model and place assets, reproduce materials and lighting, calibrate cameras, rig articulated agents, and synchronize motion. Photogrammetry and neural view synthesis reduced parts of this burden, but their outputs were often difficult to edit, animate, or convert into the clean renderer-native state required for controlled evaluation. The missing resource is therefore not simply a larger video collection, but a repeatable way to build and validate such paired evaluation units across ordinary physical scenes. Without it, newer structured-control and reference-video models cannot be compared under equivalent scene content with both real appearance evidence and renderer-native structural truth.

The premise of RenderBench is that recent progress changes this feasibility boundary. 3D Gaussian Splatting enables fast high-quality reconstruction and rendering of captured scenes [5]; DUSt3R [19] and VGGT [18] substantially reduce the calibration and geometry burden; and multimodal authoring agents can operate professional 3D tools over long workflows [11]. These systems do not by themselves guarantee metric fidelity, but they change the labor and iteration cost of producing a candidate digital twin. Geometric registration and explicit acceptance tests can then convert that candidate into benchmark-grade data.

We therefore make benchmark construction, rather than a new scalar metric, the central contribution. Starting from a captured scene, our assisted pipeline reconstructs editable assets, recovers the camera trajectory, fits object and articulated motion, and iteratively compares rendered views with the observations. Each scene passes quantitative multi-view and state-consistency gates before release. The resulting paired data supports appearance comparison to reality and diagnostic analysis against the accepted renderer state.

Our contributions are:

• We identify the missing substrate for referenceconditioned render-to-real evaluation: appearance reference images and ground-truth real videos paired with validated, editable, and dynamically registered 3D twins of the same scenes.

• We introduce a scalable construction pipeline that combines visual geometry, neural reconstruction, and assisted 3D authoring with explicit geometric and temporal acceptance gates.

• We construct RenderBench from 12reconstructed realworld scenes spanning large-scale indoor environments and egocentric viewpoints, with both dynamic and static settings, providing held-out appearance references, paired real target videos and geometry renders, together with renderer-native depth, masks, landmarks, visibility, flow, camera, and articulation metadata.

• We benchmark representative transfer systems against paired real target videos, retain PAI-Bench-C-compatible structural projections, and use the paired scene state to analyze where aggregate scores conceal object, articulation, occlusion, and motion failures.

## 2. Related Work

## 2.1. Render-to-Real Video Generation

Pix2pix [4], CycleGAN [25], and vid2vid [20] established paired, unpaired, and temporal translation paradigms. Modern diffusion transfer models substantially improve photorealism, but can still hallucinate geometry or alter motion. Generic video-generation evaluation includes distributional metrics such as FVD and multi-dimensional suites such as EvalCrafter and VBench [3, 6, 17]; unlike RenderBench, they do not assume a frame-aligned real target and editable scene state for each generated video.

Cosmos-Transfer1 [9] formulates multimodal controllable world generation from visual, edge, depth, and segmentation inputs, explicitly motivating these controls as a way to make CG renders realistic while preserving scene structure and semantics. Its TransferBench contains 600 real videos and measures control adherence, diversity, and technical quality. For robotics Sim2Real, it additionally evaluates 20 programmatically generated manipulation scenes with six prompts each. This work demonstrates both the promise of render-to-real generation and the current construction bottleneck: broad real-video evaluation lacks scene-native 3D truth, while 3D-grounded Sim2Real evaluation remains small and synthetic-only.

Cosmos-Transfer2.5 [10] extends the specialized Control2World line with multiple spatial video modalities, image prompting, longer autoregressive generation, and simulationto-photorealism workflows. Wan2.2-VACE-Fun-A14B [1] provides a second open structured-control family with generic control videos, reference images, pose, depth, edge, trajectory, and camera variants. MiniMax H3 [8] instead exposes a general multimodal reference interface and explicitly supports video-to-video motion transfer. Seedance 2.5 [2] similarly emphasizes precise video reference, editing, whitemodel control, and longer generation. These interfaces are not identical, but all can be asked to realify a proxy scene while retaining specified content or motion. This diversity strengthens the need for a model-agnostic benchmark and motivates separate structured-control and video-reference tracks.

## 2.2. Benchmarks and Paired Real–Synthetic Data

PAI-Bench [24] generalizes this evaluation into generation, conditional-generation, and understanding tracks for Physical AI. PAI-Bench-C extracts blur, edge, depth, and segmentation controls from real videos and reports Blur SSIM, Edge F1, Depth si-RMSE, Mask mIoU, DOVER technical quality, and LPIPS diversity. RenderBench retains the edge, depth, and segmentation projections for compatibility, but uses paired real target videos as its primary reference rather than adopting the complete protocol or proposing a new aggregate metric. Its contribution is a different evaluation substrate: paired real observations and editable 3D twins that allow errors to be analyzed by object, visibility, articulation, and motion.

CO3D [13] supplies large-scale object-centric real videos, camera poses, and reconstructed point clouds for categorylevel reconstruction and novel-view synthesis, but does not standardize an independently built proxy-render input. ADT [12] supplies unusually accurate real–synthetic correspondence, full calibration, scene and object geometry, and photorealistic renderings. It is the closest data precedent to ours, but differs in both objective and construction regime: ADT targets egocentric perception in two intensively digitized, motion-captured spaces, whereas Render-Bench targets scalable diagnosis of transfer from deliberately simplified, editable renders of diverse captured scenes. AptSim2Real [22] studies approximately paired image translation; its real and simulated examples need not depict the identical geometry, precluding renderer-native measurement of same-scene preservation.

## 2.3. Building Editable Digital Twins

Classical structure from motion and multi-view stereo provide geometric scene reconstruction but require sufficiently overlapping imagery and substantial cleanup [14]. 3D Gaussian Splatting [5] improves the quality and speed of novelview reconstruction, and Dynamic 3D Gaussians extends this representation to persistent dynamic tracking [7], while DUSt3R [19] directly predicts point maps from uncalibrated image collections. VGGT [18] further predicts cameras, depth, point maps, and tracks in a feed-forward pass. These representations supply geometry and appearance evidence, but benchmark construction also needs editable objects, semantic organization, clean topology, articulation, and reproducible rendering.

GPT-6 Astra is a general reasoning and computer-use model rather than a dedicated geometric estimator. Nevertheless, its demonstrated ability to reconstruct CAD objects from multi-view renders and carry out Blender-to-Unreal scene workflows [11] is important for our setting: an agent can connect specialized reconstruction outputs with asset modeling, scene organization, scripting, rendering, and visual iteration. RenderBench treats this capability as one possible construction accelerator, not as ground truth or a required model component; every resulting twin must pass the same independent validation.

## 3. Benchmark Setting

## 3.1. Paired Real–3D Scene Unit

Each benchmark unit contains appearance reference images $I _ { 1 : K } ^ { \mathrm { r e f } }$ , a ground-truth real target video $Y _ { 1 : T } ^ { \mathrm { g t } }$ , and a reconstructed dynamic digital twin $\mathcal { S } _ { 1 : T }$ registered to the target video. The reference images depict the target scene appearance but are captured from frames or viewpoints excluded from the scored target sequence. The recovered camera trajectory $C _ { 1 : T }$ is applied in a proxy renderer $R _ { p }$ to produce the geometry-proxy video:

$$
X _ { 1 : T } ^ { \mathrm { g e o } } = R _ { p } ( S _ { 1 : T } , C _ { 1 : T } ) .\tag{1}
$$

Depending on the evaluated interface, $X ^ { \mathrm { g e o } }$ may contain proxy RGB alone or a fixed bundle of rendered geometry channels such as depth, segmentation, edges, pose, or object trajectories. An evaluated transfer system produces

$$
\hat { Y } _ { 1 : T } = G ( I _ { 1 : K } ^ { \mathrm { r e f } } , X _ { 1 : T } ^ { \mathrm { g e o } } ) .\tag{2}
$$

Camera intrinsics, extrinsics, and trajectories are never exposed to the transfer model; it must infer viewpoint change from $X ^ { \mathrm { g e o } }$ itself.

The desired output follows the appearance specified by $I ^ { \mathrm { r e f } }$ , matches the held-out real target $Y ^ { \mathrm { g t } }$ , and preserves the scene and trajectory encoded by $( S , C )$ . Because reconstruction is never perfect, we publish a validity mask and registration uncertainty for every sequence; evaluation excludes pixels and states outside the accepted support.

## 3.2. Renderer-Native Scene Annotations

Separately from the model inputs, the benchmark retains

$$
A _ { t } ^ { \mathrm { e v a l } } = \{ D _ { t } , M _ { t } , P _ { t } , V _ { t } , F _ { t  t + 1 } , Q _ { t } , C _ { t } \} ,\tag{3}
$$

denoting canonical depth, instance masks, projected landmarks, visibility, optical flow, object/articulation state, and camera parameters. These annotations are available only for evaluation and analysis. If a visualized depth, mask, or pose channel is included in $X ^ { \mathrm { g e o } }$ , it is treated as part of the declared geometry-proxy representation; $A ^ { \mathrm { e v a l } }$ remains evaluator-only reference information rather than an additional input to $G .$ We use these annotations to stratify and interpret paired-reference measurements, including PAI-Bench-C-compatible structural projections, rather than to define a new composite score [24].

## 3.3. Conditioning and Evaluation Roles

The appearance images, real target video, and digital twin serve distinct roles. The model observes $I ^ { \mathrm { r e f } }$ as the appearance condition but never observes the scored frames of $Y ^ { \mathrm { g t } }$ . The ground-truth video evaluates illumination, material response, sensor appearance, image detail, and temporal realism along the requested trajectory. The accepted 3D scene provides a reference for analyzing geometry, object identity, visibility, motion, and articulation. No source is silently treated as perfect outside its designated role.

## 3.4. Task Setting

At test time, every transfer model receives exactly two semantic inputs: the real appearance references $I ^ { \mathrm { r e f } }$ and the geometry-proxy video $X ^ { \mathrm { g e o } }$ . The model must infer the target viewpoint change and scene evolution from $X ^ { \mathrm { g e o } }$ , render them in the appearance specified by $I ^ { \mathrm { r e f } }$ , and produce $\hat { Y }$ without observing $Y ^ { \mathrm { g t } } , A ^ { \mathrm { e v a l } }$ , or explicit camera information. The benchmark fixes the number of reference images and the representation of $X ^ { \mathrm { g e o } }$ for each comparison. Different model APIs may package the pair through image-reference, video-editing, or structured-control interfaces, but adapters may not introduce any additional semantic condition.

![](images/b18e9498e194ba8615826b27991d957925068f7a2f6b2fcfa6062cf822db5f8a.jpg)  
Figure 2. Construction and evaluation pipeline of RenderBench. Multi-view observations are converted into a validated digital twin with editable objects, articulated actors, and aligned cameras. Transfer models receive appearance references and an aligned proxy video, while the matched real target remains evaluator-only. Generated outputs are compared directly with the real target using paired-reference metrics, while PAI-Bench-C-compatible structural projections and paired scene state support diagnostic analysis.

## 4. Constructing RenderBench

## 4.1. Why Construction Was the Bottleneck

The desired benchmark is not a collection of attractive 3D assets. Every scene must be recognizably and geometrically tied to a real capture, editable enough to expose semantic objects and articulation, and accurate enough to render the observed camera path. Traditional manual construction satisfies these requirements but does not scale at an acceptable cost. Pure neural view synthesis scales better but generally lacks the clean meshes, semantics, joints, and physical editability needed for controlled rendering.

## 4.2. Digital-Twin Construction Pipeline

Given synchronized multi-view observations, the pipeline constructs an editable digital twin containing static geometry, semantic object instances, materials, lighting, calibrated cameras, and time-varying object and articulation states. Static geometry and dynamic actors are reconstructed separately and then registered in a common metric coordinate system. This separation is important: static objects require accurate shape and layout, whereas dynamic actors additionally require temporally stable identity and articulation.

Capture and initialization. We record calibrated multiview images or videos with scale anchors and synchronized views for dynamic sequences where available. DUSt3R/VGGT initialize cameras, geometry, depth, and tracks [18, 19], while 3D Gaussian Splatting provides an appearance reference for novel-view comparison [5]. These estimates are initializations rather than benchmark ground truth. Camera parameters and scene scale are jointly checked against multi-view correspondences and physical anchors before downstream reconstruction.

Static scene reconstruction. The scene is decomposed into architecture and semantic object instances. Multi-view masks and reconstruction estimates initialize each object’s geometry and pose. The geometry and layout are refined against silhouettes, robust depth agreement, and multi-view correspondences, with regularization used where observations are incomplete. The optimization is performed per object and jointly for scene layout, so a low whole-image error cannot hide a misplaced small object. Stable instance identities are retained across views, scene assets, and renderer annotations.

Dynamic actor reconstruction. Humans and robots are reconstructed as sequence-level articulated models rather than independent per-frame meshes. A fixed topology and articulation structure are initialized once, after which root pose and joint states Q<sub>1:T</sub> are fitted to multi-view keypoints, silhouettes, and trajectories. Temporal regularization suppresses implausible velocity and acceleration, while contact constraints align grasp, release, and support events with manipulated objects. Visibility weights suppress unsupported constraints during occlusion instead of forcing a per-frame fit. Fixed topology, link lengths, and semantic attachments prevent identity and scale drift while preserving editable articulation.

Table 1. Dataset release audit and proxy-quality diagnostics. Bracketed values denote the reported scene-level intervals.
<table><tr><td>Category</td><td>Audit item</td><td>Result</td></tr><tr><td rowspan="4">Dataset scale</td><td>Registered samples / released scenes</td><td>14 /12</td></tr><tr><td>Sample retention rate</td><td>85.7%</td></tr><tr><td>Paired real-proxy frames</td><td>1,496</td></tr><tr><td>Video frame rate</td><td>30 fps</td></tr><tr><td rowspan="2">File verification</td><td>Real and proxy frame counts match</td><td>12 / 12 scenes</td></tr><tr><td>File SHA-256 matches the release manifest</td><td>48 / 48 files</td></tr><tr><td rowspan="4">Environment audit</td><td>Background-ray hit rate</td><td>100.0% (502,656 rays)</td></tr><tr><td>Measurable background shadow response</td><td>12 / 12 scenes</td></tr><tr><td>Environment visual inspection passed</td><td>12 / 12 scenes (36 frames)</td></tr><tr><td>Foreground, camera, and animation hashes unchanged after environment edits</td><td>12 / 12 scenes</td></tr><tr><td rowspan="3">Proxy diagnostics</td><td>Estimated depth si-RMSE↓</td><td>0.2170 [0.1774, 0.2633]</td></tr><tr><td>Estimated instance mIoU ↑</td><td>0.3673 [0.2751, 0.4504]</td></tr><tr><td>Estimated instance coverage ↑</td><td>0.6228 [0.4862, 0.7426]</td></tr></table>

Agent-guided refinement. The construction agent proposes candidate scene edits from reconstruction residuals and diagnostic renders, while ambiguous identities and topology are resolved by an operator. The same acceptance tests apply regardless of how a candidate was produced. Geometry and motion are frozen before materials and lighting are refined, preventing appearance optimization from concealing structural errors.

Registration and acceptance. Static geometry, articulated motion, cameras, materials, and lighting are assembled into an editable digital twin. Validation uses views and frames withheld from fitting whenever capture coverage permits. A scene is accepted only when camera reprojection, masked depth agreement, instance silhouette overlap, landmark accuracy, temporal smoothness, event timing, and held-outview appearance satisfy pre-registered thresholds. Acceptance is evaluated both globally and per semantic object, with visibility-dependent thresholds for small or partially occluded instances. Regions that remain ambiguous are excluded through the released validity masks rather than silently treated as accurate geometry. Failed scenes are repaired or excluded, and construction error is propagated to the uncertainty analysis in Sec. 5.

## 4.3. Dataset Contents

Each accepted unit contains held-out appearance reference images, the ground-truth real target video, an editable scene package, calibrated camera trajectory, and the proxy RGB geometry video. Its evaluator-only annotations include depth, normals, optical flow, semantic/instance masks, stable object identifiers, landmarks, visibility, articulation state, event labels, validity masks, and uncertainty estimates. Construction metadata records capture provenance, coordinate conventions, and validation results. The first release contains 12reconstructed real-world scenes spanning large-scale indoor environments and egocentric viewpoints, including dynamic scenes with moving or articulated actors as well as static scenes. Figure 3 summarizes the dataset diversity and paired supervision.

## 4.4. Model-Interface Adapters

The same scene unit is packaged through standardized adapters rather than forcing every model into one API. For structured-control models, adapters provide a fixed budget of real appearance reference images and export the supported proxy RGB, depth, segmentation, edge, pose, and object-trajectory videos with fixed normalization. For general reference models, adapters provide the same appearance references and the proxy RGB geometry clip. Adapters never expose camera intrinsics, extrinsics, or trajectories; viewpoint change is encoded only through the rendered geometry videos. Every submission manifest records the information actually exposed to the model. Primary comparisons are made within matched information budgets; cross-track plots are descriptive rather than leaderboard ranks.

## 4.5. Cost and Scalability Audit

For every scene we log wall-clock construction time, human interaction time, model/API and GPU cost, number of agent iterations, repair count, and acceptance outcome. We compare the full pipeline against manual-from-scratch authoring and reconstruction without agentic assistance on a representative subset. The main construction result reports cost per accepted scene, not cost per attempted scene.

Scene diversity, with every real view paired to editable 3D

![](images/5305eac600c4545c6bd7afd2f4c79a744c1c04fb66f05f467e0490fe17add5a1.jpg)  
Figure 3. Dataset diversity and paired supervision in RenderBench. The 12reconstructed scenes span large-scale indoor environments and egocentric viewpoints, with both dynamic and static settings. Each real capture is paired with an editable digital twin, aligned proxy renders, and renderer-native scene annotations.

## 4.6. Splits and Leakage

Splits are disjoint in physical location, captured trajectory, asset identity, and human/robot motion. Appearance references are separated from scored target frames by viewpoint or time while preserving the scene’s material and illumination setting. For dynamic scenes, reference images exclude the scored motion interval, and the exact target camera path is never included among the references. We additionally reserve a held-out proxy renderer. If public assets or pretrained reconstruction models may have seen the capture location, this is disclosed; exact benchmark captures and camera paths remain held out.

## 5. Evaluation Protocol

## 5.1. Paired Real-Video Reference

Every generated video $\hat { Y } _ { 1 : T }$ is evaluated against its framealigned real target $Y _ { 1 : T } ^ { \mathrm { g t } }$ . All comparisons use the released validity masks, and the same preprocessing or estimator is applied to the generated and real videos. The renderer annotations are used to define valid support and diagnostic subsets, not as a replacement for the real-video reference.

## 5.2. Appearance Fidelity

We measure perceptual similarity using masked LPIPS [23] and report lower values as better. We additionally compute masked MS-SSIM-GT between each generated frame and its aligned real target [21], together with CIEDE2000 color error [15]. MS-SSIM-GT measures luminance, contrast, and structural agreement across multiple spatial scales, while LPIPS captures perceptual discrepancy and Lab error isolates color mismatch.

## 5.3. Structural Fidelity

Following the edge, depth, and segmentation dimensions of PAI-Bench-C [24], we apply identical projections to Y<sup>ˆ</sup> and $Y ^ { \mathrm { g t } }$ and report tolerant Edge F1, estimated-depth si-RMSE, and Instance mIoU. Edge matching uses a small spatial tolerance to avoid penalizing sub-pixel registration error. Depth is compared after the scale convention used by PAI-Bench-C, and masks are aggregated per instance so that large backgrounds cannot hide missing small objects. We additionally report instance detection coverage for objects that cannot be matched in the generated video; coverage is a diagnostic quantity reported alongside, but not merged into, the main table.

## 5.4. Motion Fidelity

The same tracker and landmark detector are applied to the generated and real videos. We report object-normalized trajectory error and track coverage for rigid objects, articulated joints, hands, and robot end effectors. These are pairedreference measurements: no reference-free temporal quality or warping score is used.

## 5.5. Aggregation and Diagnostic Slices

Metrics are aggregated over valid frames within a run, then over seeds and scenes; confidence intervals are obtained by bootstrap resampling scenes. Primary values are accompanied by slices over foreground and background, object size, visibility, rigid and articulated parts, and low- and highmotion intervals. We retain the individual measurements and do not combine them into a single score.

Table 2. Paired-reference results against the aligned real target videos. Seedance 2.5 is evaluated using its (0, 0)-aligned output.
<table><tr><td>Model</td><td>LPIPS-GT↓</td><td>MS-SSIM-GT↑</td><td> $\mathrm { L a b } \Delta E _ { 2 0 0 0 } \downarrow$ </td><td>Edge F1↑</td><td>Depth si-RMSE↓</td><td>Inst. mIoU↑</td><td>Track err.↓</td></tr><tr><td>Proxy render</td><td>0.2683</td><td>0.7607</td><td>10.8193</td><td>0.1987</td><td>0.2170</td><td>0.3673</td><td>0.0089</td></tr><tr><td>Cosmos-Transfer2.5</td><td>0.2728</td><td>0.7868</td><td>9.4432</td><td>0.1529</td><td>0.2532</td><td>0.3674</td><td>0.0077</td></tr><tr><td>Wan2.2-VACE-Fun-A14B</td><td>0.1897</td><td>0.8612</td><td>5.5227</td><td>0.2482</td><td>0.1566</td><td>0.4477</td><td>0.0090</td></tr><tr><td>MiniMax H3</td><td>0.1229</td><td>0.8996</td><td>4.6114</td><td>0.4556</td><td>0.1142</td><td>0.6202</td><td>0.0068</td></tr><tr><td>Seedance 2.5</td><td>0.1188</td><td>0.9049</td><td>2.7857</td><td>0.4810</td><td>0.1334</td><td>0.6142</td><td>0.0073</td></tr></table>

## 6. Experiments

## 6.1. Research Questions

Q1: Feasibility. Can the proposed workflow produce accepted real-scene digital twins at substantially lower human time and total cost than conventional authoring? Q2: Benchmark value. Do paired real-video and scene annotations reveal where similar aggregate scores arise from different transfer failures? Q3: Model capability. Which transfer models best reproduce the referenced real appearance while preserving the geometry and dynamics of the input video? Q4: Generalization. Do conclusions persist across unseen scenes, motions, and proxy renderers?

## 6.2. Digital-Twin Construction Evaluation

Table 1 reports the release scale, file-integrity checks, environment-edit audits, and proxy-quality diagnostics. Of 14 registered samples, 12 are released, yielding an 85.7% retention rate and 1,496 paired real–proxy frames at 30 fps. All released scenes pass frame-count, manifest-hash, background-coverage, shadow-response, visual-inspection, and edit-isolation checks. We additionally report scene-level intervals for estimated depth and instance agreement to expose variation hidden by aggregate proxy scores.

## 6.3. Transfer Evaluation

We evaluate representative structured-control and general video-reference systems: Cosmos-Transfer2.5 [10], Wan2.2- VACE-Fun-A14B [1], MiniMax H3 [8], and Seedance 2.5 [2]. The first two populate the structured-control track; H3 and Seedance populate the video-reference/editing track; any model supporting multiple interfaces is evaluated in each applicable track. Within each track, models receive identical appearance-reference budgets, source geometry renders, duration, resolution, and geometry-channel budgets; no model receives explicit camera information, and stochastic models use multiple runs. Primary results are reported in Table 2. Figure 4 provides a qualitative comparison on representative scenes.

## 6.4. Paired-Scene Diagnostic Analysis

We use accepted scene annotations to partition pairedreference errors by object size, visibility, articulation, and motion. Controlled perturbations to camera pose, object placement, joint angle, blur, texture, and detail test whether these diagnostic slices respond to the intended failure. We report sensitivity to perturbation severity and agreement with the corresponding appearance, structural, or trajectory metric rather than introducing an additional ranking score.

## 6.5. Generalization and Human Validation

Results are stratified by construction error, motion, occlusion, articulation, reflective/transparent materials, and held-out renderer. Human raters answer separate questions about real-scene resemblance, structural faithfulness, fine detail, and temporal artifacts. We correlate each judgment with its corresponding metric and report scene-bootstrap confidence intervals.

## 7. Discussion and Limitations

Assisted authoring changes cost, not the reference. Agentic 3D authoring can accelerate scene construction but can still choose an incorrect asset, conceal a discrepancy with appearance, or create invalid topology. It therefore produces candidates, while calibrated observations and acceptance tests determine benchmark inclusion.

Digital-twin error. No reconstructed scene is identical to reality. We publish uncertainty and validity support, exclude unresolved regions, and analyze how benchmark conclusions change across stricter acceptance thresholds. Claims concern fidelity to an accepted twin and its registered capture, not complete recovery of hidden physical state.

Appearance alignment. Even with a matched camera path, exposure, rolling shutter, motion blur, transparent surfaces, and non-Lambertian effects complicate pixel alignment. Multi-scale metrics and validity masks reduce but do not eliminate this problem.

Scalability. The construction audit reports failure rate, human time, total cost, and comparison with manual authoring, since successful examples alone do not establish a scalable benchmark pipeline.

![](images/517c2b20ba666c8d1ce5f193b0aebd873d0a57a113a6e91eb36a3a2f479c494b.jpg)  
Full frames + identical-region detail crops; matched source timestamps

Figure 4. Qualitative comparison on representative paired scenes. The proxy input specifies geometry and motion, while the held-out real target provides the ground-truth video for evaluation. Zoomed regions highlight structural, articulation, and appearance failures that may be obscured by aggregate scores.

Benchmark scope. The first release focuses on scenes that can pass geometric and temporal validation. Fluids, deformable objects, outdoor crowds, severe occlusion, and uninstrumented fast motion remain difficult. We expose this selection bias in the dataset statistics and per-category results.

Heterogeneous model interfaces. Structured controls and general video references expose different amounts and kinds of information. A single pooled ranking would confound model quality with conditioning bandwidth. Track-specific reporting and matched information budgets improve fairness, but cannot make closed natural-language interfaces exactly equivalent to explicit geometry controls.

Licensing and privacy. Real captures, reconstructed locations, 3D assets, and human motion require explicit permissions and compatible release terms. We remove identifying biometric and location information where consent does not cover public release.

## 8. Conclusion

Render-to-real generation has advanced faster than the data needed to evaluate it. We introduced RenderBench around the missing substrate: held-out real appearance references and ground-truth target videos paired with validated, editable, and dynamically registered 3D twins. Recent visualgeometry, neural-reconstruction, and assisted authoring systems reduce the cost of producing candidate scenes. Independent geometric gates then provide distinct appearance conditions, paired real-video targets, and renderer-native scene annotations. We evaluate transfer quality against paired real target videos, retain PAI-Bench-C-compatible structural projections, and use scene annotations to localize errors across objects, visibility, articulation, and motion. The audited release contains 12 accepted scenes and 1,496 paired real– proxy frames, with complete frame-count and SHA-256 verification across all released files. Environment edits preserve foreground, camera, and animation hashes in every scene, and the measured proxy diagnostics establish a transparent baseline for interpreting downstream transfer results.

## References

[1] Alibaba PAI Team. Wan2.2-Fun-Control: Controllable video

generation. https://github.com/aigc- apps/ VideoX-Fun, 2025. Accessed September 15, 2026. 2, 3, 8

[2] ByteDance Seed Team. Seedance 2.5: One-take creation with flexible referencing. https://seed.bytedance. com/en/blog/one-take-creation-flexiblereferencing - introducing - seedance - 2 - 5, 2026. Accessed September 15, 2026. 2, 3, 8

[3] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 3

[4] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017. 3

[5] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and¨ George Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. In ACM SIGGRAPH, 2023. 3, 4, 5

[6] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. EvalCrafter: Benchmarking and evaluating large video generation models. In CVPR, 2024. 3

[7] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024. 4

[8] MiniMax. MiniMax H3: An open general-purpose multimodal video model. https://www.minimax.io/ news/minimax-h3-open-source, 2026. Accessed September 15, 2026. 2, 3, 8

[9] NVIDIA. Cosmos-Transfer1: Conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492, 2025. 2, 3

[10] NVIDIA. Cosmos-Transfer2.5: Multi-control world-toworld generation. https://github.com/nvidiacosmos/cosmos- transfer2.5, 2026. Accessed September 15, 2026. 2, 3, 8

[11] OpenAI. GPT-6 Astra: A new generation of intelligence. https : / / openai . com / index / gpt - 6 - astra/, 2026. Accessed September 15, 2026. 3, 4

[12] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In ICCV, pages 20133–20143, 2023. 2, 3

[13] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, pages 10901– 10911, 2021. 2, 3

[14] Johannes L. Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In CVPR, 2016. 4

[15] Gaurav Sharma, Wencheng Wu, and Edul N. Dalal. The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. Color Research & Application, 30(1):21–30, 2005. 7

[16] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In IROS, 2017. 2

[17] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. To-¨ wards accurate generative models of video: A new metric and challenges. arXiv preprint arXiv:1812.01717, 2018. 3

[18] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025. 3, 4, 5

[19] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jer´ ome Revaud. DUSt3R: Geometric 3dˆ vision made easy. In CVPR, 2024. 3, 4, 5

[20] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-to-video synthesis. In NeurIPS, 2018. 3

[21] Zhou Wang, Eero P. Simoncelli, and Alan C. Bovik. Multiscale structural similarity for image quality assessment. In Asilomar Conference on Signals, Systems and Computers, pages 1398–1402, 2003. 7

[22] Charles Y. Zhang and Ashish Shrivastava. AptSim2Real: Approximately-paired sim-to-real image translation. arXiv preprint arXiv:2303.12704, 2023. 2, 4

[23] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 7

[24] Fengzhe Zhou, Jiannan Huang, Jialuo Li, Deva Ramanan, and Humphrey Shi. PAI-Bench: A comprehensive benchmark for physical ai, 2025. 2, 3, 4, 7

[25] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017. 3