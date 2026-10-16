# GRASE: GEOMETRY-NATIVE REAL-TO-SIM AGENTIC SCENE MODELING FROM VIDEO

Zhaorui Lin<sup>1</sup>, Haohan Liu<sup>2</sup>, Zhen Chen<sup>3</sup>, Xiangru Huang<sup>4∗</sup>

<sup>1</sup> Xi’an Jiaotong University (xjtu.edu.cn) <sup>2</sup> Fuzhou University (fzu.edu.cn) <sup>3</sup> Adobe Systems (adobe.com) <sup>4</sup> Westlake University (westlake.edu)

![](images/b6c55acd9b8dcca95ddd1828cd35ce5debd782cfd411102d1c556256775f17ac.jpg)

Figure 1: GRASE transforms an input video into an editable 3D scene, recovering room structure, object layout, and fine-grained details. The example shows the progression from video observations to reconstruction, together with scene overviews, a close-up, and multiview renderings.

## ABSTRACT

Real-to-sim conversion aims to turn a captured video of a real environment into an editable 3D scene while preserving the geometry and layout of what was observed. Existing methods often follow a one-pass, loosely coupled paradigm: each stage receives only limited feedback from previous stages, allowing errors to propagate without reliable correction and leaving the final scene inconsistent. We present GRASE, an agentic framework that closes this loop by treating the captured multi-view video observations as a standing reference throughout scene modeling. Instead of committing to intermediate estimates, GRASE revises its current hypothesis whenever rendered evidence disagrees with the capture, overcoming the fixed intermediate assumptions of conventional pipelines. Specifically, GRASE first fits the architectural shell and its openings from depth and multi-view cues. An agent then places generated assets in a shared coordinate frame while checking their spatial relationships. Finally, each asset’s geometry and placement are refined iteratively to improve agreement across the observed views. By coupling these steps through feedback, GRASE localizes structural discrepancies and revises only the responsible components, while preserving cross-view consistency. The resulting 3D scenes are faithful to the input, consistent across views, and suitable for simulations. In our experiments, GRASE substantially outperforms other state-of-the-art methods, including a direct real-to-sim conversion using GPT6- Astra.

## 1 INTRODUCTION

Real-to-sim aims to convert real-world observations into simulation-ready 3D environments, providing a scalable way to build training and evaluation data for embodied agents. Video is particularly attractive because it is inexpensive, easy to capture, and naturally records scene appearance and spatial structure. A robust video-based real-to-sim pipeline could therefore greatly reduce the cost of constructing digital twins for robot learning. Real-to-sim requires more than visually plausible reconstruction: the resulting scene must contain accurate geometry, object-level structure, and physically meaningful spatial relationships. Video makes this especially challenging due to occlusions, incomplete viewpoints, motion blur, depth ambiguity, and weakly textured regions. Small geometric errors can further lead to severe simulation failures such as penetration, floating objects, or incorrect contacts.

Existing approaches typically combine scene reconstruction, object segmentation, 3D generation, and scene assembly. However, they often fail to recover large room surfaces, produce inaccurate object geometry from partial observations, or rely on generative models that hallucinate shapes inconsistent with the input video. These errors accumulate during scene composition, resulting in scenes that may look plausible but are geometrically inconsistent and unsuitable for simulation.

An agentic formulation is attractive because real-to-sim naturally requires iterative diagnosis, tool use, and error correction rather than a single feed-forward prediction. In addition, modern VLMs with strong spatial perception capabilities, such as GPT6-Astra, enable a new perceive-thenreconstruct paradigm for real-to-sim conversion. However, we find that applying agentic models such as GPT-6 to recover accurate geometry and topology is not trivial. On the one hand, when using only multi-view images, VLMs struggle to produce accurate 3D geometry at real-world scales. On the other hand, preliminary experiments suggest that even powerful agents such as GPT6-Astra struggle to reason directly over raw point clouds when given estimated point clouds and multi-view camera parameters, frequently making incorrect judgments about geometry, scale, and spatial rela tionships. This suggests that simply giving 3D observations to an agent is insufficient.

Our key idea is to couple explicit geometric modeling with an agentic verification and repair loop, using the original video as a persistent reference throughout reconstruction. The agent represents the scene through room surfaces and openings, object transforms, and editable object parts, so that discrepancies can be addressed through targeted geometric or topological changes. This leads to two closely connected components: room-surface topology and geometry refinement and objectcentric layout registration and part-wise geometric refinement, integrated with agentic execution and verification.

Complex room surfaces, including walls, ceilings, and floors, are traditionally difficult to reconstruct. Using a VLM with strong spatial reasoning capabilities, we propose a perceive-thenreconstruct paradigm that first perceives the architectural topology and then uses geometric primitives and parametric patches to fit regular and irregular surfaces. This paradigm allows us to reliably recover accurate geometry and topology using multi-view depth observations, image boundaries, and agentic spatial reasoning and editing. Notably, this paradigm significantly improves reconstruction results. In the ablation study, we observe a substantial performance drop when this paradigm is not used.

With the room surfaces reconstructed, the agent then aims to reconstruct object-level geometry that aligns with the input video. Agentic models such as GPT6-Astra are good at coding and editing 3D objects using geometric primitives in an agentic loop, but they struggle to match fine and complex real-world geometry. Meshes generated by 3D generation models such as TRELLIS.2, however, are difficult to edit through code. We propose to combine 3D generation models with a part-wise geometric refinement module: we first segment the generated geometry using off-the-shelf segmentation models, then use agentic models to perform part-wise editing, verification, and repair, incorporating geometric cues such as point clouds and image boundaries.

To summarize, our contributions are:

• We propose GRASE, a geometry-native agentic real-to-sim pipeline that accurately reconstructs the scene from the input video, providing correct and consistent topology recovery for room surfaces.

• We propose an object-centric layout registration method that uses agentic part-wise editing to produce refined geometry that better aligns with the input video and is compatible with 3D generation models.

• In our experiments, GRASE performs significantly better than its GPT6-Astra counterparts, with or without access to 3D point clouds and 3D generation models, demonstrating the strength of our proposed perceive-then-reconstruct paradigm.

## 2 RELATED WORK

Image- and Video-Conditioned Compositional Scene Reconstruction. Reconstructing editable 3D scenes requires recovering object geometry and spatial configuration. Early methods jointly modeled object shapes, poses, layouts, and semantic or instance structure (Nie et al., 2020; Zhang et al., 2021; Dahnert et al., 2021). Geometric foundation models subsequently enabled dense correspondence estimation and reconstruction from unposed images (Wang et al., 2024a; Leroy et al., 2024; Yang et al., 2025a; Wang et al., 2025), while spatial memory supports incremental integration of observations (Wang & Agapito, 2025). Complementing these geometric representations, compositional approaches combine object generation with scene alignment (Shan et al., 2026; Zhou et al., 2024; Yao et al., 2025). Recent text-to-3D models such as Block3D improve the efficiency of object asset synthesis through block-wise diffusion and confidence-guided token correction (Cui et al., 2026). Other works jointly model multiple objects through diffusion (Huang et al., 2025; Dahnert et al., 2024). Deep Prior Assembly uses completion and candidate alignment, whereas CAST incorporates point-cloud conditioning and physical correction. OVOW (Chen et al., 2026) further extends instance-level mesh reconstruction to dynamic video scenes. Our work focuses on static scene reconstruction from video, combining point-cloud placement constraints with visual feedback to refine object poses and revisit mismatched components. This connects observation-based geometry with iterative correction of an editable, object-level scene representation.

Agentic 3D Scene Generation and Iterative Refinement. Language-guided systems translate descriptions into layouts, asset selections, and structured scene specifications (Feng et al., 2023; Yang et al., 2024; Wen et al., 2023; C¸ elen et al., 2024). Architect and LayoutVLM strengthen spatial guidance through hierarchical image inpainting and differentiable layout optimization, respectively (Wang et al., 2024b; Sun et al., 2025). Feedback-based approaches revisit scene states using rendered observations, scene judgment, or reflective tool selection (Hu et al., 2024; Ling et al., 2026; Yang et al., 2026; Xia et al., 2026). Among them, SceneCraft generates and refines Blender programs, while SAGE combines MCP-based orchestration with visual and physical critics. Scene-Smith (Pfaff et al., 2026) organizes hierarchical construction through designer, critic, and orchestrator agents. Spatially contextualized VLMs (Liu et al., 2025) further integrate semantic point clouds, scene hypergraphs, and Blender-based verification. These methods motivate combining planning, geometric grounding, and feedback. Our framework centers this process on agreement with an input video: VLM criticism guides Blender-based pose adjustment, followed by comparison and regeneration of mismatched objects. The emphasis is on connecting scene-level feedback to targeted corrections during reconstruction.

Spatial Reasoning and Geometric Grounding in VLMs. Spatial benchmarks reveal persistent challenges in visual perception, spatial measurement, and reasoning across views or video frames (Fu et al., 2024; Yang et al., 2025b; Zhang et al., 2026). SpatialVLM, SpatialBot, and SpatialRGPT address these challenges through spatial supervision, depth-aware inputs, and regionconditioned reasoning (Chen et al., 2024a; Cai et al., 2025; Cheng et al., 2024). SPAR (Zhang et al., 2026) further scales spatial training data using scenes with 3D ground truth. Complementary methods connect language models to explicit scene representations through point-cloud features, visual prompts, or object embeddings and identifiers (Hong et al., 2023; Chen et al., 2024b; Huang et al., 2023). SpatialPIN (Ma et al., 2024) instead incorporates 3D priors through prompting and tool interaction without additional training. LayoutVLM and spatially contextualized VLMs demonstrate how geometric representations and structured relations support scene modification (Sun et al., 2025; Liu et al., 2025). Following this principle, we use VLMs for semantic interpretation and corrective feedback, while point-cloud constraints anchor object placement to observed geometry. This couples spatial reasoning with executable pose correction in the reconstructed scene.

![](images/eecb8ec629f3fbc7fa1e44a34d251ad9c1108b3464bb1f0d4b1530cfe4a81a20.jpg)  
Figure 2: Overview of GRASE. (a) Scene preprocessing extracts a point cloud, a scene DAG, and object masks. (b) Room surface topology and geometry refinement reconstructs the shell and openings. (c) Object-centric layout registration and part-wise geometric refinement align object meshes and correct local shapes. Agentic verification guides both reconstruction stages.

## 3 GEOMETRY-NATIVE AGENTIC REAL-TO-SIM

Given a monocular video of a static indoor environment, our goal is to reconstruct an editable 3D scene S that is consistent with the observed views. The scene consists of room surfaces with explicit openings, one mesh per object with a world-space transform, and the support relations among surfaces and objects.

Figure 2 summarizes GRASE. Instead of letting a model produce geometry directly, GRASE exposes room surfaces, opening topology, object transforms, and object parts as variables that are fitted to geometric evidence and edited through constrained operations. After preprocessing, the shell is reconstructed together with its topology (Sec. 3.1), and objects are generated, registered, and corrected part by part (Sec. 3.2). Appendix A gives the implementation details, and Table 3 lists all constants.

Scene preprocessing. We uniformly sample T frames $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { T }$ from the video and process them jointly with Depth Anything 3 (DA3), which estimates per-frame depth $z _ { i }$ , per-pixel confidence, intrinsics $\mathbf { K } _ { i }$ , and camera-to-world poses $\mathbf { T } _ { i }$ . Each pixel u with valid, confident depth is lifted to $\mathbf { T } _ { i } \big ( z _ { i } ( \mathbf { u } ) \mathbf { K } _ { i } ^ { - 1 } \tilde { \mathbf { u } } \big )$ , where u˜ is its homogeneous coordinate, and merging these points over all frames gives the scene point cloud.

Scene inventory. A GPT-6-based VLM organizes the frames into an inventory of surfaces and objects. A wall surface is a connected architectural face whose position and orientation vary continuously, so a smoothly curved wall is one surface, while sharp corners, structural steps, or spatial separation divide surfaces. By matching surfaces across views and inspecting their junctions, the VLM obtains the architectural surfaces $\bar { \mathcal W }$ (walls, floor, and ceiling), their adjacency $\mathbf { \bar { \mathcal { A } } } \subseteq \mathcal { W } \times \mathcal { W }$ (surfaces meeting along a shared physical boundary), and the openings hosted by each surface. For objects, the VLM marks a 2D box around each visible object in every frame and groups the detections by physical identity using appearance, component structure, spatial context, and the recovered cameras. Each group is an instance, giving $\mathcal { O } = \overset { \cdot } { \{ o _ { j } \} } _ { j = 1 } ^ { N }$ , where $o _ { j }$ has a category $c _ { j }$ and a 2D box in every frame of $\nu _ { j }$ , the frames in which it is visible. The VLM further separates large, layoutdefining objects (e.g., tables, sofas) from small movable ones (e.g., cups), which are generated and registered differently (Sec. 3.2), and infers a directed acyclic support graph $\mathcal { G } = ( \mathcal { W } \bar { \cup } \mathcal { O } , \mathcal { E } )$ from visible contact, relative position, and occlusion, where $( u , o _ { j } ) \in \mathcal { E }$ means that u supports $o _ { j }$ . Finally, SAM 3 segments every surface and object in its observed frames, and lifting the confident depth inside each eroded mask gives point clouds $\mathcal { P } _ { s }$ for surfaces $s \in \mathcal W$ and $\mathcal { P } _ { j } ^ { \mathrm { { o b j } } }$ for objects (Appendix A.1).

## 3.1 ROOM SURFACE TOPOLOGY AND GEOMETRY REFINEMENT

For each surface, the VLM decides whether a plane suffices or whether connected parametric patches are needed. Throughout, $\rho ( x ) = 2 \big ( \sqrt { 1 + x ^ { 2 } } - 1 \big )$ denotes the soft- $. L _ { 1 }$ penalty, which limits the influence of outliers.

Regular surfaces. A planar surface s is represented as $\Pi _ { s } = \{ \mathbf { x } : \mathbf { n } _ { s } ^ { \top } \mathbf { x } + d _ { s } = 0 \}$ with $\| \mathbf { n } _ { s } \| _ { 2 } = 1$ initialized by RANSAC on $\mathcal { P } _ { s }$ and refit to its inliers by SVD. The adjacency fixes the topology of the shell. If each wall were then cropped to its own points, neighboring walls would rarely meet exactly, leaving gaps or overlaps at the corners. We instead take a wall’s extent from its neighbors. Three mutually adjacent surfaces $s , t ,$ r in ${ \mathcal { A } } ,$ such as two walls and the floor, meet at the corner $\mathbf { v } _ { s t r }$ that solves

$$
\begin{array} { r } { [ { \bf n } _ { s } \ { \bf n } _ { t } \ { \bf n } _ { r } ] ^ { \top } { \bf v } _ { s t r } = - [ d _ { s } \ d _ { t } \ d _ { r } ] ^ { \top } , } \end{array}\tag{1}
$$

and these corners bound each wall as a finite polygon. Adjacent surfaces reference the same corners, which are recomputed whenever an incident plane moves. The adjacency therefore fixes the topology of the shell while the planes remain free to change: refining or editing one plane moves the shared corners with it and never opens a gap at a junction.

The initial fits are accurate in the middle of each surface but not necessarily at its junctions, the lines where it meets its neighbors, such as the corner between two walls or the base of a wall. To refine the planes, we fit the junctions observed in the images: notice that a junction is typically visible in multiple frames and each sighting of junction k in frame i is one observation $( i , k ) \in \mathcal { I }$ . For this observation, $B _ { i k }$ contains pixels sampled along the junction in frame i, away from occluders and image borders, and $\ell _ { i k } ( { \pmb \theta } )$ is where the current planes place the junction in that frame: the projection of the two planes’ intersection line, scaled so that $\ell _ { i k } ^ { \top } \hat { \mathbf { \mathbf { \mathbf { \ell } } } }$ u˜ is a signed pixel distance. All planes are refined jointly by

$$
\begin{array} { r l } & { \underset { \pmb { \theta } } { \mathrm { m i n } } \ : \sum _ { s } \frac { 1 } { | \mathcal { P } _ { s } | } \underset { \mathbf { p } \in \mathcal { P } _ { s } } { \sum } \rho \bigg ( \frac { \pmb { \mathrm { n } } _ { s } ^ { \top } \mathbf { p } + d _ { s } } { \sigma _ { s } } \bigg ) + \frac { \lambda _ { \mathrm { j c t } } } { | \mathcal { I } | } \displaystyle \sum _ { ( i , k ) \in \mathcal { I } } \frac { 1 } { | B _ { i k } | } \sum _ { \mathbf { u } \in B _ { i k } } \rho \bigg ( \frac { \ell _ { i k } ( \pmb { \theta } ) ^ { \top } \tilde { \mathbf { u } } } { \sigma _ { \mathrm { p x } } } \bigg ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{2}
$$

where s ranges over the planar surfaces, θ collects the plane offsets and normal corrections, $\sigma _ { s }$ and $\sigma _ { \mathrm { p x } }$ are fixed 3D and pixel scales, $\lambda _ { \mathrm { j c t } }$ and $\lambda _ { \mathrm { p r i o r } }$ are fixed weights, and $\epsilon _ { m }$ bounds the change of each parameter from its initial value $\theta _ { m , 0 } .$ . We solve this bounded robust least-squares problem with the cameras and A fixed (Appendix A.2). Because the shared corners are recomputed from the planes at every evaluation, the shell stays closed throughout the optimization. The resulting polygons are finally triangulated into editable Blender meshes.

Irregular surfaces. A surface that no single plane describes, such as a curved wall, is represented by connected patches, each mapping local coordinates $( \xi , \eta )$ to a 3D point,

$$
\mathbf { f } ( \xi , \eta ) = \mathbf { C b } ( \xi , \eta ) ,\tag{3}
$$

where b contains fixed basis functions and C their 3D coefficients. We use two families of patches. A periodic loft covers a region that wraps around the room; its coordinates are the horizontal angle and the height, and its horizontal position combines periodic trigonometric terms with quadratic terms in height. A local quadratic displacement field offsets a reference plane along its normal by a quadratic function, and a vertical reference normal makes it a height field. With the reference frame fixed, both families are linear in C and are fitted by least squares, with an envelope prior for poorly observed parts of a loft and stronger penalties on the quadratic terms of a displacement field to avoid unsupported curvature. The VLM specifies how neighboring patches meet. At a sharp junction, we intersect the two surfaces and trim both parameter domains by the intersection curve, so the seam vertices coincide after triangulation. Where the transition is smooth, we blend the gradient-normalized implicit functions of the two patches with a smooth maximum and project nearby vertices onto the blend, whose width falls to zero near boundaries that must stay fixed. Please refer Appendix A.2 for details.

![](images/d37e9adc172aa6bae3ff52bcbd27926fb62f7883b0215aba4080f5c5bae85dbf.jpg)  
Figure 3: Room-surface reconstruction and refinement. Input video frames and the segmented point cloud guide patch initialization. The patches are iteratively refined using geometric evidence and comparisons between source frames and renderings. Observed opening boundaries are backprojected onto the fitted surfaces to construct explicit openings.

Openings and inferred geometry. Doors, windows, recesses, and passages are modeled as geometry rather than as appearance changes on an intact wall. For each inventoried opening, the VLM traces its visible rim in the source frames and determines whether it leads to a rear surface or an open passage. Rays through the traced rim are intersected with the host surface to initialize a 3D rim. We cut the enclosed region from the surface and build side surfaces along the rim, together with the rear surface when it is observed. Openings visible on a rear surface are built on that surface in turn, which handles nested structures. Occluded parts of the shell, such as the wall behind a curtain, are completed by extending the fitted surfaces under their parameterizations up to their junctions, and are marked as inferred rather than observed. The finished shell is rendered from the source cameras, and remaining discrepancies in surface shape, junctions, or opening rims go to the verification loop. Figure 3 illustrates this workflow.

## 3.2 OBJECT-CENTRIC LAYOUT REGISTRATION AND PART-WISE GEOMETRIC REFINEMENT

With the shell in place, we build a mesh $M _ { j }$ and a world-space transform for every object $o _ { j } .$ . The support graph G sets the order: large objects come first, and an object is placed only once everything supporting it is stable. Actual contact is checked against the supporting meshes during registration.

Object initialization Large objects, such as tables, chairs, and other furniture, are built directly in Blender by the VLM through the Model Context Protocol (MCP), using simple primitives for their major components. This gives clean, editable geometry whose parts follow the object’s structure. When the VLM judges the resulting shape too poor, the object is generated with TRELLIS.2 instead, and small objects are always generated with TRELLIS.2 (Figure 4). The VLM builds each primitive model in place, using the location and extent of the object’s point cloud $\mathcal { P } _ { j } ^ { \mathrm { o b j } }$ , whereas a TRELLIS.2 mesh arrives in its own canonical frame and is registered to $\mathcal { P } _ { j } ^ { \mathrm { o b j } }$ by ICP. Since the point cloud covers only the visible side of the object and is noisy, either alignment can still be off in size or pose. The agent therefore selects three frames that together view the object from as many sides as possible, and the VLM compares renderings of the placed object with these frames and adjusts its size and pose accordingly. A detailed pipeline can be found in the Appendix A.3.

![](images/18eb1f91c270078aa443eea234b7f842c9aa83a0daace7a85df01ee56e9b3fee.jpg)  
Figure 4: Object generation and placement. The large objects in “Scene with Large Obj.” are built from primitives through Blender MCP. The VLM checks them against source views; only models with incorrect shapes or structures are replaced using TRELLIS.2. Small objects are generated directly with TRELLIS.2. All objects are then registered and refined to form the final scene.

Part-wise geometric refinement. A generated mesh may merge several components into one piece, such as a sofa and its cushions, and a single transform cannot fit them all: once the sofa body matches the observations, the cushions may still be misplaced. The VLM finds such a component by comparing renderings with the source frames, and the agent selects its region on the mesh by casting rays through the component’s mask in several renderings. The component is then re-aligned to its own points while the rest of the mesh stays fixed. A component with wrong proportions, such as the sofa backrest in Figure 5, is stretched or contracted along a local axis a between a fixed anchor $\mathbf { x } _ { \mathrm { f i x } }$ and a moving anchor $\mathbf { x } _ { \mathrm { m o v } } \colon$

$$
\mathbf { v } ^ { \prime } = \mathbf { v } + S \big ( \tau ( \mathbf { v } ) \big ) \Delta \mathbf { a } , \qquad \tau ( \mathbf { v } ) = \frac { ( \mathbf { v } - \mathbf { x } _ { \mathrm { f i x } } ) ^ { \top } \mathbf { a } } { ( \mathbf { x } _ { \mathrm { m o v } } - \mathbf { x } _ { \mathrm { f i x } } ) ^ { \top } \mathbf { a } } ,\tag{4}
$$

for every vertex v of the component, where $\Delta$ is the signed length correction and $S ( \tau ) = 3 \tau ^ { 2 } - 2 \tau ^ { 3 }$ with τ clamped to [0, 1], is the cubic smoothstep, so the edit fades smoothly into the fixed geometry. An edit is kept only if the component matches the observations better without disturbing the rest of the object (Appendix A.4).

![](images/ac0b2a87e41bfdfe2bd7cce8877957a8d418063f642bb3fda49efa32ec999d34.jpg)  
Figure 5: Part-wise geometric refinement of a large object. After object registration, a local mismatch remains in the sofa backrest. The VLM identifies the affected part and guides a constrained stretch that improves its alignment with the observed geometry while preserving the rest of the object.

Cross-object refinement. The previous steps place each object on its own, so relations between objects are not yet enforced: a computer may float above the desk it stands on, or two neighboring objects may interpenetrate. We therefore check every support relation in G against the actual geometry. An object is moved along the support normal until its base touches the supporting mesh, and samples across the contact region are checked for floating or penetration; wall-mounted and suspended objects are held to their observed attachment regions instead. Neighboring objects are also checked for interpenetration, and each correction is verified against the source frames through agent. Because contact is defined relative to the supporting mesh, objects are processed in the order of G, and a later change to a support triggers new checks of the objects it supports (Appendix A.5).

## 4 EXPERIMENTS

We evaluate GRASE at the architectural, object, and scene levels, and examine the contribution of geometric feedback through ablation. Input-view rendering provides a complementary measure of agreement with the captured observations.

## 4.1 EXPERIMENTAL SETUP

Datasets and baselines. We compare GRASE against three baselines on the same eight Replica and fifteen Imaginarium scenes: SAM3D; GPT-6 vanilla, which constructs scenes directly from visual observations; and GPT-6 + TRELLIS.2 + point cloud, which additionally has access to point-cloud/camera information and TRELLIS.2 asset generation. Generation inputs and alignment protocols differ across methods.

Metrics. Both datasets share metric definitions: Chamfer Distance (CD), normal consistency (NC), and F-score at 5 cm (F@5cm) for geometry; instance mIoU for reconstructed–reference object agreement; and PSNR, SSIM, and LPIPS for input-view renderings against reference images. Object-level metrics use the full ground-truth (GT) object count as the denominator. Architectural shells are evaluated separately from scene geometry. Further evaluation details are provided in Appendix A.7.

## 4.2 QUANTITATIVE COMPARISON

Object and scene reconstruction. GRASE improves reconstruction at both the object and scene levels on both datasets (Table 1). Relative to the strongest baseline in scene-level CD on each dataset, it reduces the error from 11.00 to 3.14 cm on Imaginarium and from 16.39 to 5.80 cm on Replica, corresponding to reductions of 71.4% and 64.6%, respectively. The improvements extend to object agreement, with mIoU reaching 0.7425 and 0.4121. GRASE also achieves the best input-view rendering scores among the methods with reported rendering metrics, indicating that the geometric gains are accompanied by closer agreement with the input images.

Access to assets and point clouds. The two GPT-6 baselines exhibit different trends across datasets. Adding TRELLIS.2 and point-cloud access improves Imaginarium scene-level CD from 18.16 to 11.00 cm, but increases the error on Replica from 16.39 to 19.91 cm; the rendering metrics follow the same trend. In contrast, GRASE improves reconstruction on both datasets. This comparison suggests that access to generative assets and geometric observations alone does not ensure accurate scene modeling.

Architectural shell reconstruction. GRASE also improves the geometry of the room shell (Table 2a). On Imaginarium, it reduces CD from 6.55 cm for GPT-6 + TRELLIS.2 + point cloud to 2.97 cm, a 54.6% reduction. Its NC of 0.955 and F@5cm of 0.826 further indicate improved surface orientation and geometric agreement. These results complement the object- and scene-level evaluation by assessing the architectural structure separately.

## 4.3 EFFECT OF GEOMETRIC FEEDBACK

We ablate the joint point-cloud and hard-crease critic while retaining the remaining agent workflow. On Replica (Table 1), enabling the critic reduces object-level CD from 9.78 to 4.52 cm and scene-level CD from 9.96 to 5.80 cm, reductions of 53.8% and 41.8%, respectively. Object mIoU increases from 0.1577 to 0.4121, accompanied by an increase in input-view PSNR from 14.55 to 16.37 dB. Thus, the contribution of geometric feedback is reflected in both object- and scene-level reconstruction, as well as input-view rendering.

Table 1: Object- and scene-level reconstruction and input-view rendering on Imaginarium and Replica. PC denotes point cloud; – indicates unavailable rendering metrics.
<table><tr><td></td><td colspan="4">Object-level</td><td colspan="3">Scene-level</td><td colspan="3">Input-view rendering</td></tr><tr><td>Method</td><td>mIoU↑</td><td>NC↑</td><td>CD (cm)↓</td><td>F@5cm↑</td><td>CD (cm)↓</td><td>NC↑</td><td>F@5cm↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Imaginarium</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAM3D</td><td>0.0482</td><td>0.6182</td><td>7.1482</td><td>0.0648</td><td>72.1585</td><td>0.5892</td><td>0.1006</td><td></td><td></td><td></td></tr><tr><td>GPT-6 vanilla</td><td>0.1066</td><td>0.6490</td><td>18.4091</td><td>0.1144</td><td>18.1588</td><td>0.6844</td><td>0.3449</td><td>11.8223</td><td>0.5342</td><td>0.6481</td></tr><tr><td>GPT-6 + TRELLIS.2 + PC</td><td>0.2066</td><td>0.6850</td><td>11.8184</td><td>0.2062</td><td>10.9977</td><td>0.7946</td><td>0.5615</td><td>12.5762</td><td>0.5994</td><td>0.5606</td></tr><tr><td>GRASE (Ours)</td><td>0.7425</td><td>0.7929</td><td>2.1936</td><td>0.8536</td><td>3.1412</td><td>0.9044</td><td>0.8866</td><td>15.5692</td><td>0.6667</td><td>0.3381</td></tr><tr><td>Replica</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SÁM3D</td><td>0.1566</td><td>0.6666</td><td>4.9177</td><td>0.1339</td><td>63.9258</td><td>0.6050</td><td>0.1908</td><td></td><td></td><td></td></tr><tr><td>GPT-6 vanilla</td><td>0.1897</td><td>0.6346</td><td>11.9005</td><td>0.1742</td><td>16.3887</td><td>0.6177</td><td>0.3452</td><td>11.1367</td><td>0.6943</td><td>0.5488</td></tr><tr><td>GPT-6 + TRELLIS.2 + PC</td><td>0.1325</td><td>0.5670</td><td>11.0108</td><td>0.1063</td><td>19.9107</td><td>0.6129</td><td>0.2165</td><td>10.8904</td><td>0.6373</td><td>0.5785</td></tr><tr><td>GRASE w/o geo. critic</td><td>0.1577</td><td>0.7076</td><td>9.7786</td><td>0.1740</td><td>9.9572</td><td>0.7729</td><td>0.5418</td><td>14.5470</td><td>0.6965</td><td>0.4190</td></tr><tr><td>GRASE (Ours)</td><td>0.4121</td><td>0.7271</td><td>4.5207</td><td>0.4507</td><td>5.7963</td><td>0.8287</td><td>0.6477</td><td>16.3674</td><td>0.7454</td><td>0.3595</td></tr></table>

Table 2: Architectural shell reconstruction: (a) Imaginarium comparison; (b) seven-scene critic ablation. See Appendix A.7 for the ablation protocol and units.
<table><tr><td>Method</td><td>CD (cm)↓</td><td>NC↑</td><td>F@5cm↑</td></tr><tr><td>GPT-6 vanilla</td><td>29.6426</td><td>0.7916</td><td>0.3474</td></tr><tr><td>GPT-6 + TRELLIS.2 + PC</td><td>6.5474</td><td>0.8785</td><td>0.6729</td></tr><tr><td>GRASE (Ours)</td><td>2.9734</td><td>0.9548</td><td>0.8263</td></tr></table>

(b) Critic ablation
<table><tr><td>Method</td><td>CD↓ NC↑</td><td>F-score↑</td></tr><tr><td>w/o Critic</td><td>55.74 0.859</td><td>0.134</td></tr><tr><td>Ours</td><td>3.33 0.965</td><td>0.767</td></tr></table>

The separate seven-scene shell ablation (Table 2b) shows the same direction of improvement: CD decreases from 55.74 to 3.33, a 94.0% reduction, while NC increases from 0.859 to 0.965 and F-score from 0.134 to 0.767. These shell results use the alignment and source-coordinate units specified in Appendix A.7. Together, the ablations support the role of geometric feedback in refining both architectural surfaces and the assembled scene.

## 4.4 QUALITATIVE ANALYSIS

Figure 6 compares four scenes using matched camera poses and intrinsics for the input and the first four rendered columns. These views expose differences in room boundaries, furniture proportions, and relative object placement; the last column shows overviews of the GRASE reconstructions. Additional viewpoints are provided in Figure 7.

The component-level examples illustrate how the editable representation supports targeted correction. Figure 3 shows the reconstruction of curved room surfaces and explicit openings. In Figure 5, a local backrest mismatch remains after the sofa is registered; editing that part improves its alignment while preserving the rest of the asset. These examples illustrate corrections at the architectural and part levels that complement the aggregate reconstruction metrics.

## 5 CONCLUSION

We present GRASE, a geometry-native agentic framework that reconstructs editable indoor scenes from monocular videos through shell modeling, object layout registration, and part-wise refinement under multi-view verification, outperforming prior methods on Replica and Imaginarium. GRASE has several limitations. First, reconstruction remains limited in heavily occluded regions where the input video provides insufficient geometric evidence; such regions are completed by extending fitted surfaces or by generative priors and are marked as inferred rather than observed. Second, the scene inventory, support graph, and verification decisions rely on VLM judgments, so an early error, such as a missed junction or an incorrect support relation, can persist when the rendered evidence is ambiguous. Third, GRASE assumes a static scene and reconstructs geometry and layout only; articulation, materials, and physical properties required by many simulators are not yet recovered. Future work will extend the framework to articulated and dynamic objects, outdoor scenes, estimate physical and material properties from video, and replace part of the VLM-based verification with learned geometric critics to reduce cost and improve robustness.

![](images/cf6496c8428472e0723377da76b7879e1dd19b542c819db3e3edb6247141dfcc.jpg)  
Figure 6: Qualitative comparison on four scenes. Each row shows the input, matched-camera renderings from SAM3D, GPT-6 vanilla, GPT-6 + TRELLIS.2 + point cloud, and GRASE, followed by a novel-view overview of the GRASE scene. Figure 7 provides additional input-camera viewpoints for the same scenes.

## AI USE STATEMENT

Generative AI tools were used to aid and polish the writing of this paper, specifically to improve clarity and concision. The authors reviewed all AI-assisted edits and take responsibility for the final content.

## REFERENCES

Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 9490–9498. IEEE, 2025.

Ata C¸ elen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. I-design: Personalized llm interior designer. In European Conference on Computer Vision, pp. 217–234. Springer, 2024.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14455–14465, 2024a.

Junhao Chen, Boran Zhang, Mingjin Chen, Henghaofan Zhang, Saining Zhang, Congcong Zhu, Hao Zhao, Ruqi Huang, Zhihao Li, and Yufei Wang. One video, one world: Turning monocular video into physical 4d scenes. arXiv preprint arXiv:2606.31388, 2026.

Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26418–26428. IEEE, 2024b.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.

Bowen Cui, Weijie Wang, Zeyu Zhang, Yefei He, Mingda Lin, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, and Bohan Zhuang. Block3d: Efficient text-to-3d generation via block-wise diffusion. arXiv preprint arXiv:2608.19567, 2026.

Manuel Dahnert, Ji Hou, Matthias Nießner, and Angela Dai. Panoptic 3d scene reconstruction from a single rgb image. Advances in Neural Information Processing Systems, 34:8282–8293, 2021.

Manuel Dahnert, Angela Dai, Norman Muller, and Matthias Nießner. Coherent 3d scene diffusion¨ from a single rgb image. Advances in Neural Information Processing Systems, 37:23435–23463, 2024.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in neural information processing systems, 36: 18225–18250, 2023.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148–166. Springer, 2024.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid, and Alireza Fathi. Scenecraft: An llm agent for synthesizing 3d scenes as blender code. In ICML, pp. 19252–19282, 2024.

Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. arXiv preprint arXiv:2312.08168, 2023.

Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance diffusion for single image to 3d scene generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23646–23657. IEEE, 2025.

Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Grounding image matching in 3d with mast3r.ˆ In European conference on computer vision, pp. 71–91. Springer, 2024.

Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Max Li. Scenethesis: A language and vision agentic framework for 3d scene generation. In International Conference on Learning Representations, volume 2026, pp. 136596–136629, 2026.

Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. Agentic 3d scene generation with spatially contextualized vlms. arXiv preprint arXiv:2505.20129, 2025.

Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. Advances in neural information processing systems, 37:68803–68832, 2024.

Yinyu Nie, Xiaoguang Han, Shihui Guo, Yujian Zheng, Jian Chang, and Jian Jun Zhang. Total3dunderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 52–61. IEEE, 2020.

Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, and Russ Tedrake. Scenesmith: Agentic generation of simulation-ready indoor scenes. In Forty-third International Conference on Machine Learning, 2026.

Hui Shan, Keyang Luo, Ming Li, Sizhe Zheng, Yanwei Fu, Zhen Chen, and Xiangru Huang. Interact3d: Compositional 3d generation of interactive objects. arXiv preprint arXiv:2603.16085, 2026.

Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 29469–29478. IEEE, 2025.

Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In 2025 International Conference on 3D Vision (3DV), pp. 78–89. IEEE, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5294–5306. IEEE, 2025.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20697–20709. IEEE, 2024a.

Yian Wang, Xiaowen Qiu, Jiageng Liu, Zhehuan Chen, Jiting Cai, Yufei Wang, Tsun-Hsuan Wang, Zhou Xian, and Chuang Gan. Architect: Generating vivid and interactive 3d scenes with hierarchical 2d inpainting. Advances in Neural Information Processing Systems, 37:67575–67603, 2024b.

Zehao Wen, Zichen Liu, Srinath Sridhar, and Rao Fu. Anyhome: Open-vocabulary generation of structured and textured 3d homes. arXiv e-prints, pp. arXiv–2312, 2023.

Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, et al. Sage: Scalable agentic 3d scene generation for embodied ai. arXiv preprint arXiv:2602.10116, 2026.

Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21924–21935. IEEE, 2025a.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10632–10643. IEEE, 2025b.

Yandan Yang, Baoxiong Jia, Shujie Zhang, and Siyuan Huang. Sceneweaver: All-in-one 3d scene synthesis with an extensible and self-reflective agent. Advances in neural information processing systems, 38:140319–140351, 2026.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16277–16287. IEEE, 2024.

Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics (TOG), 44(4):1–19, 2025.

Cheng Zhang, Zhaopeng Cui, Yinda Zhang, Bing Zeng, Marc Pollefeys, and Shuaicheng Liu. Holistic 3d scene understanding from a single image with implicit representation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8829–8838. IEEE, 2021.

Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Jilin Mei, Chunhui Chen, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. Advances in Neural Information Processing Systems, 38, 2026.

Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. Zero-shot scene reconstruction from single images with deep prior assembly. Advances in Neural Information Processing Systems, 37:39104–39127, 2024.

## A IMPLEMENTATION DETAILS OF GRASE

This section gives the details omitted from Sec. 3. We keep the notation of the main paper, and Table 3 lists all constants.

## A.1 SCENE PREPROCESSING AND INVENTORY

Depth and cameras. We sample $T$ frames uniformly from each video and run DA3 on all of them jointly, obtaining depth maps, per-pixel confidence, intrinsics, and camera poses. Before backprojection, we check the camera and pixel-coordinate conventions of the DA3 outputs, discard invalid depth, and filter low-confidence pixels. A retained pixel $\mathbf { u } = ( u , v )$ of frame i is back-projected as

$$
\mathbf { x } _ { w } = \mathbf { T } _ { i } \big ( z _ { i } ( \mathbf { u } ) \mathbf { K } _ { i } ^ { - 1 } \tilde { \mathbf { u } } \big ) , \qquad \tilde { \mathbf { u } } = [ u , v , 1 ] ^ { \top } ,\tag{5}
$$

where $z _ { i } ( \mathbf { u } )$ is the camera-Z depth, $\mathbf { K } _ { i }$ the intrinsic matrix, and $\mathbf { T } _ { i }$ the rigid transformation from the camera frame of $I _ { i }$ to the world frame.

Segmented point clouds. For objects, each SAM 3 mask is eroded with a kernel of size $r _ { \mathrm { e r o d e } }$ Inside the eroded mask, we keep valid, positive-depth pixels whose DA3 confidence exceeds max $\cdot ( \tau _ { \mathrm { c o n f } } , Q _ { \kappa } )$ , where $\tau _ { \mathrm { c o n f } }$ is a fixed floor and $Q _ { \kappa }$ is the κ-quantile of the confidences inside that mask. The retained pixels are subsampled to at most $N _ { \mathrm { v i e w } }$ per view and back-projected with $\operatorname { E q . } \left( 5 \right)$ . For surfaces, we likewise discard unreliable depth and points near segmentation boundaries, and we aggregate the points of each physical surface across frames using the cross-view correspondences in the inventory.

Scene inventory. In every sampled frame, the VLM places a 2D bounding box around the visible extent of each wall surface. The boxes localize surfaces for inspection only; they do not define boundaries or geometry. The VLM compares structural cues inside and around the boxes across successive and complementary frames to decide which observations show the same physical wall, accounting for viewpoint changes and partial occlusion, and it examines shared junctions across views to decide which surfaces are adjacent. Each visible opening is associated with its host surface. The surface inventory thus records cross-view correspondences, per-frame boxes and visibility, adjacency, and opening-to-surface relations.

Objects are handled in the same way. The VLM assigns a semantic category and a 2D box to every visible object and associates detections across views by consistency in appearance, component structure, and spatial context, using the recovered cameras to account for viewpoint changes. Repeated observations of one object share an identifier, while distinct objects of the same category keep separate identifiers. Large, layout-defining objects are distinguished from small ones by their observed extent and role in the scene, and support is inferred from visible contact, relative position, and occlusion cues across views.

## A.2 ROOM SURFACE RECONSTRUCTION

Plane initialization. For each planar surface, we discard unreliable depth samples and points near segmentation boundaries. Every nondegenerate three-point sample of $\mathcal { P } _ { s }$ defines a candidate plane, and RANSAC keeps the candidate supported by the most points within distance $\tau _ { \mathrm { r a n s a c } }$ . We then refit the inliers by SVD of their centered coordinates, taking the right singular vector with the smallest singular value as $\mathbf { n } _ { s }$ and setting $d _ { s } = - \mathbf { n } _ { s } ^ { \top } \bar { \mathbf { p } }$ , where $\bar { \bf p }$ is the inlier centroid.

Corners and polygons. Every triple of mutually adjacent surfaces in $\mathcal { A }$ defines the corner

$$
\mathbf { v } _ { s t r } = - \mathbf { \Psi } \left[ \mathbf { n } _ { s } ^ { \top } \right] ^ { - 1 } \left[ \begin{array} { l } { d _ { s } } \\ { d _ { t } } \\ { \mathbf { n } _ { r } ^ { \top } } \end{array} \right] ^ { - 1 } \left[ \begin{array} { l } { d _ { s } } \\ { d _ { t } } \\ { d _ { r } } \end{array} \right] .\tag{6}
$$

Intersections with the adjacent walls, the floor, and the ceiling define each wall polygon, and neigh boring polygons reference the same corner positions.

Junction term. The VLM marks visible physical junctions in the source frames and associates each with its two incident surfaces. We sample pixels along each observed junction, excluding foreground occlusions and image-border edges; for junction k in frame i, these pixels form $B _ { i k }$ . For the corresponding plane pair, we compute the 3D intersection line, project two of its points through camera i, and take the image line $\ell _ { i k }$ through the two projections, normalized so that its first two coefficients have unit Euclidean norm. The inner product $\ell _ { i k } ^ { \top }$ u˜ is then the signed distance in pixels from u to the line.

Optimization. The active parameters θ are the plane offsets and corrections to the normal directions, and each may move by at most $\epsilon _ { m }$ from its initial value. We solve Eq. (2) as a bounded robust least-squares problem with the cameras, observations, and adjacency fixed. At every function evaluation, the normals are renormalized and all shared corners are recomputed with Eq. (6). The fitted polygons are triangulated into Blender meshes and rendered from additional source views, where we check visible boundaries and point-to-triangle distances. A revision is retained only if it improves the targeted mismatch without opening gaps or degrading boundaries that were already aligned.

Occluded walls. For a partially occluded wall, the VLM matches the exposed wall regions across frames and identifies their junctions with neighboring walls, the floor, and the ceiling. We intersect these surfaces with the fitted wall plane and triangulate the resulting polygon across hidden regions, such as the part behind a curtain. The hidden part is marked as inferred, and the visible parts of the extended plane and its boundaries are checked against complementary source views.

Irregular patches. For an irregular surface, the VLM identifies visible surface regions and structural boundaries, which determine the support points and connections of each patch. We fit the coefficients C of Eq. (3) with the reference frames and prescribed coordinate components fixed; planes, periodic lofts, and local quadratic displacement fields (including height fields) are the only representations used, and we do not search a larger primitive library.

Periodic lofts. The local coordinates are the horizontal angle and the height, and the horizontal position combines periodic trigonometric functions of the angle with quadratic terms in height. The coefficients are estimated by weighted linear least squares. To constrain poorly observed regions, we add samples from a prescribed elliptical envelope that narrows with height.

Local quadratic displacement fields. The reference origin is the coordinate-wise median of the support points. An initially near-vertical patch uses a horizontal reference normal pointing from the mean position of the cameras that observe it toward this origin. With the frame fixed, we fit the normal displacement by linear least squares, penalizing the quadratic coefficients more strongly than the others to limit unsupported curvature. Height fields, obtained with a vertical reference normal, are fitted with the soft-L penalty ρ to reduce the influence of depth outliers. When a height field has an observed outer contour, we also adjust its coefficients and the parameters of an elliptical outline so that its projected boundary aligns with traced image curves, bounding and penalizing changes from the initialization.

Patch connections. For a sharp junction prescribed by the VLM, we bracket the roots of one surface’s implicit equation along the other surface and solve for them with Brent’s method. The resulting intersection curve trims both parameter domains. We triangulate the domains and evaluate the surface maps at their vertices; matching boundary samples give corresponding seam vertices identical positions in both patches. Where adjacent patches require a smooth connection, we locally subdivide mesh edges and clip overlapping portions. Each implicit surface function is divided by its gradient norm, and the resulting fields are combined with a smooth maximum. Nearby vertices are projected onto the zero level set of the blended field using finite-difference gradients and bounded steps. The blend width varies spatially and decreases to zero near protected boundaries, which keeps those boundaries fixed.

## A.3 OBJECT REGISTRATION

Primitive modeling. For a large object, the VLM assembles an editable model $M _ { i } ^ { \mathrm { p r i m } }$ in Blender through MCP, using simple primitives for the observed major components; their dimensions and relative transforms encode the object’s structure and proportions. The VLM places the model roughly: its position and size follow the center and extent of $\mathcal { P } _ { i } ^ { \mathrm { o b j } }$ , its base rests on the supporting surface given by ${ \mathcal { G } } ,$ and its facing direction, which the extent alone does not determine, is read from recognizable components in the frames, such as the side a chair’s backrest is on. If the VLM judges that $M _ { j } ^ { \mathrm { p r i m } }$ captures the observed components and proportions, it is kept as $M _ { j }$

Generation with TRELLIS.2. Otherwise, we pick three clear, complementary frames from $\nu _ { j }$ complete the partially occluded object in each by image editing, and generate one TRELLIS.2 candidate per completed image. Because TRELLIS.2 produces each mesh in its own canonical frame, we first align each candidate’s local axes and contact reference and then place it using $M _ { j } ^ { \mathrm { p r i m } }$ as the reference for position, orientation, and size. The candidates are rendered from the source cameras, and the VLM keeps the one whose components, proportions, and asymmetric structure best match the original frames; it replaces $M _ { i } ^ { \mathrm { p r i m } }$ while keeping the instance’s identity and support relations. Small objects skip primitive modeling: TRELLIS.2 generates their meshes directly, and the VLM initializes their position, scale, and orientation from the spatial distribution of $\mathcal { P } _ { j } ^ { \mathrm { o b j } }$ and the source images.

Pose parameterization. Registration keeps the mesh geometry and the cameras fixed and changes only the transform. For upright furniture, the support fixes the base height, roll, and pitch. Let $\hat { M _ { j } }$ be the mesh divided by its local bounding-box dimensions, with its origin moved to the bottom center. In a frame whose vertical axis is the support normal, the transform is

$$
\begin{array} { r } { \mathcal { T } _ { \mathbf { q } } ( \hat { \mathbf { x } } ) = \mathbf { R } ( \psi ) \mathrm { d i a g } ( \mathbf { e } ) \hat { \mathbf { x } } + \big [ \mathbf { t } ^ { \top } , h _ { j } \big ] ^ { \top } , \quad \mathbf { q } = ( \mathbf { t } , \psi , \mathbf { e } ) , } \end{array}\tag{7}
$$

where $\mathbf { t } \in \mathbb { R } ^ { 2 }$ is the horizontal translation, $\mathbf { R } ( \psi )$ the rotation by yaw ψ about the support normal, $\mathbf { e } \in \mathbb { R } _ { > 0 } ^ { 3 }$ the dimensions along the object’s local axes, and $h _ { j }$ the base height given by the support. The admissible box $B _ { j }$ fixes an interval around the initial parameters $\mathbf { q } ^ { ( 0 ) }$ for every parameter: translation limits scale with the initial object extent, yaw has a bounded deviation, and dimension limits are proportional to the initial dimensions. For an upright object on a tabletop, q contains the two in-plane translations and the rotation about the tabletop normal, and scale is added only if the projected dimensions remain inconsistent after the pose is corrected. All distances are measured in the common reconstruction coordinate system.

ICP. We sample $N _ { \mathrm { s u r f } }$ points on the surface of $\hat { M _ { j } }$ and build a KD-tree over them. At iteration t, each observation $\mathbf { p } _ { n } \in \mathcal { P } _ { j } ^ { \mathrm { { o b j } } }$ is mapped into normalized model coordinates by $\mathcal { T } _ { \mathbf { q } ^ { ( t ) } } ^ { - 1 }$ and assigned its nearest surface sample ${ \hat { \mathbf { y } } } _ { n }$ . With these correspondences fixed, we solve

$$
\begin{array} { r } { \mathbf { q } ^ { ( t + 1 ) } = \arg \underset { \mathbf { q } \in \mathcal { B } _ { j } } { \operatorname* { m i n } } \sum _ { n } \rho _ { \delta } \big ( \mathcal { T } _ { \mathbf { q } } ( \hat { \mathbf { y } } _ { n } ) - \mathbf { p } _ { n } \big ) } \\ { + \rho _ { \delta } \big ( w _ { \mathrm { p r i o r } } ( \mathbf { q } - \mathbf { q } ^ { ( 0 ) } ) \big ) , } \end{array}\tag{8}
$$

where $\rho _ { \delta }$ applies $\rho ( \cdot / \delta )$ to every scalar component of its argument and sums the results, δ is a fixed soft- $. L _ { 1 }$ scale, and $w _ { \mathrm { p r i o r } }$ is a fixed prior weight. Each inner solve has a fixed budget of function evaluations. Correspondence assignment and parameter estimation alternate until the largest absolute parameter change falls below $\epsilon _ { \mathrm { i c p } }$ or $N _ { \mathrm { i c p } }$ iterations are reached.

VLM-guided adjustment. The agent selects three frames from $\nu _ { j }$ that together view the object from as many sides as possible and renders the placed object from the same cameras. The VLM compares each rendering with its frame, aided by overlays that mark the area covered only by the rendering and the area covered only by the observation, and adjusts the size and pose within $B _ { j }$ Small objects occupy few pixels in distant views, so their adjustment uses close-up frames. Each instance is adjusted with its own observations, including when several instances share one generated mesh.

Silhouette refinement. The adjustment can be made precise with a silhouette objective over all views $i \in \mathcal { V } _ { j }$ . Let $A _ { i }$ be the observed instance mask, $H _ { i } ( \mathbf { q } )$ the visible mask of the object rendered together with the rest of the scene, and $L _ { i } ^ { \mathrm { m a s k } } = 1 - \mathrm { I o U } \big ( A _ { i } , H _ { i } ( \mathbf { q } ) \big )$ . We minimize

$$
\begin{array} { r l } & { \displaystyle { L ( { \bf q } ) = \frac { 1 } { | \mathcal V _ { j } | } \sum _ { i \in \mathcal V _ { j } } \left( L _ { i } ^ { \mathrm { m a s k } } + \lambda _ { \mathrm { e d g e } } E _ { i } \right) } } \\ & { \quad \quad \quad \quad + \lambda _ { \mathrm { w o r s t } } \displaystyle { \operatorname* { m a x } _ { i \in \mathcal V _ { j } } L _ { i } ^ { \mathrm { m a s k } } + \lambda _ { \mathrm { p c } } L _ { \mathrm { p c } } } , } \\ & { \displaystyle { E _ { i } = \frac { \bar { d } ( \partial H _ { i } , \partial A _ { i } ) + \bar { d } ( \partial A _ { i } , \partial H _ { i } ) } { 2 \mathrm { d i a g } \left( A _ { i } \right) } } , } \end{array}\tag{9}
$$

where $\lambda _ { \mathrm { e d g e } } , \lambda _ { \mathrm { w o r s t } }$ , and $\lambda _ { \mathrm { p c } }$ are fixed weights, ${ \bar { d } } ( X , Y )$ is the mean distance from the pixels of $X$ to the nearest pixel of $\dot { Y , \mathrm { d i a g } } ( A _ { i } )$ is the diagonal of the bounding box of $A _ { i }$ , and $L _ { \mathrm { p c } }$ averages normalized and capped point-to-surface distances. The worst-view term keeps a single badly aligned view from disappearing into the average. To render $H _ { i } ( \mathbf { q } )$ , we ray-cast the target object and the rest of the scene from the recovered camera and keep the target pixels that are not occluded by other scene geometry, up to a numerical depth tolerance $\tau _ { \mathrm { d e p t h } }$ . Masks are evaluated at a reduced resolution with correspondingly scaled intrinsics. The boundary $\partial X$ of a mask $X$ is obtained by subtracting its eroded version, and image-border pixels are excluded. Euclidean distance transforms give the mean nearest-boundary distances $\bar { d }$ in both directions; the normalizer diag $\left( A _ { i } \right)$ is computed once at the reduced resolution, and an empty rendered boundary receives a fixed penalty $E _ { \mathrm { e m p t y } }$ . For $L _ { \mathrm { p c } } ,$ each world-space correspondence distance is divided by a distance scale $d _ { \mathrm { p c } }$ , capped at $\tau _ { \mathrm { c a p } } .$ and averaged over the observations; correspondences are recomputed with the same normalized-space nearest-neighbor rule as in ICP, using a separate surface sample of the mesh.

Coordinate search. Because L is built from binary masks and is not differentiable in ${ \bf q } ,$ we minimize it by bounded coordinate search. The search starts from whichever of $\mathbf { q } ^ { ( 0 ) }$ and the ICP solution has the lower L. The translation, yaw, and lower dimension bounds of $B _ { j }$ are kept, while the upper dimension bounds are relaxed by a predefined amount $\beta _ { \mathrm { d i m } } .$ Translation and dimension steps are proportional to the initial object dimensions, and yaw uses a fixed angular step. For each coordinate in turn, we evaluate a positive and a negative step, both clipped to the bounds, by re-rendering all visible masks and recomputing $L ,$ and we keep the better trial only if it reduces L by more than $\epsilon _ { \mathrm { c s } } .$ . Step sizes decrease over $N _ { \mathrm { l v l } }$ search levels, each with a budget of $N _ { \mathrm { s w e e p } }$ sweeps, and a sweep without improvement advances the search to the next level.

Final check. The accepted transform is applied in Blender and re-rendered from the source cameras. We recompute point-to-surface distances against the full triangle mesh and inspect the overlays before retaining the placement.

## A.4 PART-WISE REFINEMENT

The VLM compares the source views with the current renderings to identify a component that the object-level transform cannot $\mathrm { \ f t { \ell , } }$ for example cushions that remain misplaced once the sofa body matches, or an overly wide backrest whose seat and armrests already match. To select the component, the agent marks its visible regions in multiple renderings of the current object, and camera rays through these regions are intersected with the mesh to find the corresponding triangles. The selection is highlighted in complementary model views so that the agent can check that it covers the intended component and excludes neighboring parts, and additional views are used to add side and rear surfaces and to correct selection errors. The vertices of the reviewed selection form a vertex group $u ;$ the mesh is neither cut nor split into a separate object. These rendered annotations only determine which existing geometry is editable; the source frames remain the reference for the target shape.

For a stretch, the fixed anchor $\mathbf { x } _ { \mathrm { f i x } }$ is placed where the component’s alignment or attachment should be preserved, and the moving anchor $\mathbf { x } _ { \mathrm { m o v } }$ toward the extent that needs correction. Corresponding landmarks in the current renderings are ray-cast onto the mesh to locate both anchors in object-local coordinates. The agent estimates $\bar { \Delta }$ from the source boundaries and the available reliable depth, and checks the proposed change in complementary source views before applying Eq. (4) to the vertices in U. After each edit, we re-render from the relevant source cameras and assess the component’s boundary alignment, its connection to neighboring components, and its support contact. The edit is kept only if it improves the targeted discrepancy without degrading other components or introducing severe interpenetration; otherwise, the previous geometry is restored.

## A.5 CROSS-OBJECT REFINEMENT

For each support relation $( u , o _ { j } ) \in \mathcal { E } _ { \mathrm { : } }$ , the object is translated along the support normal until its base touches the mesh of u, which stays fixed. Several samples across the intended contact region are then checked so that the object neither floats above nor penetrates its support. Wall-mounted and suspended objects are constrained by their observed attachment regions instead. Each correction is also checked in complementary views and against neighboring objects for spacing and occlusion. Supports are processed before the objects they support; objects on different supports are processed independently, and a later change to a support requires re-checking the objects it supports.

## A.6 AGENT

The skill specification defines the stage dependencies, the tools available at each stage, and the criteria a candidate must meet to be accepted. Each review records the localized discrepancies and the corrective actions taken, and every accepted scene state is stored so that later revisions can be rolled back to it. Both reconstruction stages are executed by a tool-using VLM agent that follows a reusable skill specification defining the stage dependencies, available operations, and verification criteria (Appendix A.6). Rather than producing geometry itself, the agent works through tool calls: frame extraction, DA3 inference, image editing, TRELLIS.2 generation, and Blender operations through MCP. Numerical tools perform the shell fitting, object registration, mesh measurements, and mask comparisons, while the agent makes the decisions they cannot: which representation a surface needs, which frames to use, which component is wrong, and whether a result matches the video. Following the design–critique–orchestration organization of SceneSmith Pfaff et al. (2026), we describe this work in three roles.

Construction. The agent inspects the video, selects informative views, and invokes the modeling and generation tools: it initializes the shell and chooses the parameters to optimize, prepares generation inputs, compares candidate assets in the scene, and adjusts their placement.

Verification. The agent compares source frames with textured and textureless renderings from the same cameras, together with overlays of extra and missing regions and measurements of mask overlap, boundary distance, and point-to-surface distance. It checks structural completeness, object identity, proportions, orientation, and support and occlusion relations. Rigid major structures must agree at their boundaries, whereas soft or intricate components are judged by overall volume, color, and arrangement. Only the original frames serve as the reference; edited completions are used solely to generate candidates. Each discrepancy is localized to one component and mapped to an opera tion: constrained transforms for pose and scale errors, part-wise deformation for local shape errors, surface refitting for shell mismatches, geometry correction or candidate replacement for structural errors, and material adjustment for appearance errors.

Orchestration. Each revision yields a candidate scene that is evaluated from the relevant source views and accepted only if it improves the diagnosed discrepancy without regressions in other views; a lower aggregate error cannot override a missing structure, severe interpenetration, or incorrect support. Otherwise, the last accepted state is restored. The shell is reviewed before any furniture is built, and a change to a supporting object triggers new checks of everything it supports. When the evidence is ambiguous, the agent searches the video for clearer or complementary views before editing further.

## A.7 EVALUATION PROTOCOL

Shared scene sets. The baseline comparisons in Sec. 4 use a fixed set of eight Replica scenes and a fixed set of fifteen Imaginarium scenes. For each dataset, the evaluation set is the union of the scenes used across the baseline runs, and GRASE, SAM3D, GPT-6 vanilla, and GPT-6 +

Table 3: Constants used by GRASE. Symbols follow Sec. 3 and Sec. A.
<table><tr><td>Quantity</td><td>Symbol</td><td>Value</td></tr><tr><td colspan="3">Scene preprocessing</td></tr><tr><td>Frames sampled per video (real / simulated)</td><td> $T$ </td><td>100 / 60</td></tr><tr><td>Mask erosion kernel</td><td>Terode</td><td>3 × 3</td></tr><tr><td>Confidence floor / quantile</td><td> $\tau _ { \mathrm { c o n f } } , \kappa$ </td><td>1.2/0.35</td></tr><tr><td>Points per view</td><td> $N _ { \mathrm { v i e w } }$ </td><td>650</td></tr><tr><td colspan="3">Object generation</td></tr><tr><td>Source views per large object</td><td></td><td>3</td></tr><tr><td>TRELLIS.2 candidates per view</td><td></td><td>1</td></tr><tr><td colspan="3">Object registration  $\pm 0 . 1 8 \operatorname* { m a x } _ { k } e _ { k } ^ { ( 0 ) }$ </td></tr><tr><td>Translation / yaw / dimension bounds of  $B _ { j }$ </td><td></td><td>±0.45 rad  $[ 0 . 7 5 , 1 . 2 5 ] \mathbf { e } ^ { ( 0 ) }$  70,000 (ICP)</td></tr><tr><td>Surface samples</td><td> $N _ { \mathrm { s u r f } }$ </td><td>60,000 (silhouette)</td></tr><tr><td>Soft-L1scale / prior weight ICP tolerance / max. iterations</td><td> $\delta , w _ { \mathrm { p r i o r } }$ </td><td>0.025/0.03</td></tr><tr><td>Function evaluations per inner solve</td><td> $\epsilon _ { \mathrm { i c p } } , N _ { \mathrm { i c p } }$ </td><td> $1 0 ^ { - 5 } / 2 5$ </td></tr><tr><td>Mask resolution factor</td><td></td><td>50</td></tr><tr><td>Depth tolerance</td><td></td><td>0.25</td></tr><tr><td></td><td> $\tau _ { \mathrm { d e p t h } }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Empty-boundary penalty</td><td> $E _ { \mathrm { e m p t y } }$ </td><td>1</td></tr><tr><td>Silhouette weights</td><td> $\lambda _ { \mathrm { e d g e } } , \lambda _ { \mathrm { w o r s t } } , \lambda _ { \mathrm { p c } }$ </td><td>0.4/0.35/0.12</td></tr><tr><td>Point-distance scale / cap</td><td> $d _ { \mathrm { p c } } , \tau _ { \mathrm { c a p } }$ </td><td>0.05/2</td></tr><tr><td>Upper-dimension relaxation</td><td> $\beta _ { \mathrm { d i m } }$ </td><td>0.05</td></tr><tr><td></td><td></td><td> $0 . 0 3 5 \operatorname* { m a x } _ { k } e _ { k } ^ { ( 0 ) }$   $0 . 0 4 5 { \bf e } ^ { ( 0 ) }$ </td></tr><tr><td>Initial translation / dimension / yaw steps Search levels / sweeps per level</td><td></td><td>0.045 rad</td></tr><tr><td>Step schedule across levels</td><td> $N _ { \mathrm { l v l } } , N _ { \mathrm { s w e e p } }$ </td><td>3/4</td></tr><tr><td></td><td></td><td>1,0.5,0.25</td></tr><tr><td>Acceptance tolerance</td><td> $\epsilon _ { \mathrm { c s } }$ </td><td>10⁻⁵</td></tr><tr><td colspan="3">Agent</td></tr><tr><td colspan="3">Max. revision rounds per stage</td></tr></table>

TRELLIS.2 + point cloud are all evaluated on this entire set. Every method successfully completed reconstruction on every scene. The reported baseline comparisons therefore do not use different scene subsets for different methods.

Object-level denominator. For a given scene, all methods are evaluated against the same full set of GT objects. The denominator for each reported object-level metric (mIoU, NC, CD, and F@5cm) is the full GT object count, not the number of predicted, successfully reconstructed, or matched objects. GT objects are not removed from this denominator based on a method’s reconstruction or matching outcome.

Rendering availability. The missing SAM3D rendering metrics in Table 1 reflect the absence of authored lighting and a world setup in its delivered GLB files, not failed reconstructions or a smaller evaluation scene set.

Shell ablation protocol. The point-cloud and hard-crease critic ablation in Table 2b is averaged equally over seven scenes after upright rigid alignment with fixed scale. CD is reported in $1 \bar { 0 } ^ { - 2 }$ source-coordinate units, and F-score uses a threshold of 0.05 source-coordinate units.

Input

SAM3D

GPT-6 Vanilla

GPT-6 + TRELLIS.2 + Point Cloud

Ours

Ours NVS

![](images/030a4174b1db76de941b6d452f46defd4490a507807bba4b5f200fc382e18c80.jpg)  
Figure 7: Full qualitative multiview comparison for the four scenes of Figure 6. Each group of three rows shows one scene from three input-camera viewpoints. The first five columns share camera poses and intrinsics within each row; Ours NVS shows Blender scene overviews.

## B ADDITIONAL QUALITATIVE RESULTS

Figures 8–10 show reconstructions of 23 scenes: seven from our scene collection, eight from Replica, and eight from Imaginarium. These figures illustrate selected reconstructions rather than define the quantitative evaluation sets. In particular, the eight Imaginarium scenes shown here are a subset of the fifteen scenes used by every method in the baseline comparisons; all methods use the same eight Replica and fifteen Imaginarium evaluation scenes (Appendix A.7).

## B.1 OUR SCENE COLLECTION

Input  
Render 1  
Render 2  
Render 3  
Render 4  
Render 5  
![](images/0ede4e044b486a2294a3bf1e47cdc768bc000e4d75a859e76dfadce7d7fc4a24.jpg)  
Figure 8: Our scene collection. Each row shows an input frame followed by five renderings of the reconstructed scene.

## B.2 REPLICA SCENES

Input

Render 1  
Render 2  
Render 3  
Render 4  
Render 5  
![](images/4b534a4a1a42a7076bd92dd2c8e189b731b00d3338842055b10d467f1ed5fde6.jpg)  
Figure 9: Replica scenes. Columns follow Figure 8.

## B.3 IMAGINARIUM SCENES

Input  
Render 1  
Render 2  
Render 3  
Render 4  
Render 5  
![](images/3f4f279659bd63cc4f196b8ccc24bf9becaab3feffc8fced8e10e566d9e68dda.jpg)  
Figure 10: Imaginarium scenes. Columns follow Figure 8.