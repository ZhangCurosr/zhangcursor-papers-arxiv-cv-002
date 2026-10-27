![](images/4117982f6b837a78f9451186248d1db230440ce8ee20b7ceb60e7625b00de6ad.jpg)  
Figure 1 Simulation scenes and tasks from SceneAgent (left to right, top to bottom): a kitchen from CollectData with its doors, drawers, and appliances articulated; picking and plugging cables into a data center server rack; Sharpa hands putting sunglasses on a statue; a TeleAvatar with two Sharpa hands at a clamp in a lab space; and SceneAgent’s 4DCodeBench reconstructions of a liquid dam break around rigid cylindrical pillars (B\_09) and a layered watermelon breaking apart on impact (Z\_12).

# SceneAgent: An Agentic Pipeline from 3D and 4D Captures to Simulation Environments

Luke Hollis<sup>1∗</sup>, Tianxing Fan<sup>1∗</sup>, Heng Yang<sup>1</sup>

<sup>1</sup>Computational Robotics Group, School of Engineering and Applied Sciences, Harvard University <sup>∗</sup>Equal contribution, lhollis@g.harvard.edu, tianxing\_fan@seas.harvard.edu, hankyang@seas.harvard.edu

Visual captures provide detailed scene appearance, but converting them into interactive simulation environments requires object reconstruction, physical modeling, and correction of intermediate errors. We present SceneAgent, an agentic pipeline that coordinates these steps through simulator-based visual inspection and repair. The pipeline processes 3D Gaussian Splatting (3DGS), photogrammetry, LiDAR captures, AI-generated 3DGS scenes, and 4D videos of dynamic scenes, and combines object segmentation, physical parameter estimation, part decomposition, articulation, and deformable simulation. To support policy learning, SceneAgent generates diverse object variants (“digital sisters”) and scripted demonstrations. On a cluttered bin scene from GraspClutter6D, SceneAgent raises F1 at 1 cm for every coding agent with scorable outputs both with and without its skills, and its reconstructions are better than SimFoundry [47]: all agent configurations exceed SimFoundry’s F1 score, with a best of 60.75% versus 6.9%. On three dynamic scenes from 4DCodeBench [51], SceneAgent raises the mean dynamic IoU of Gemini 4 Argon from 0.86% to 99.09%. We further demonstrate diverse simulated interactions and preliminary policy transfer to a real robot.

Keywords: SceneAgent, real-to-sim, sim-to-real, 3D Gaussian Splatting, 4D reconstruction, simulation environments, predictive physics, robot policy evaluation, policy training, digital sisters

Website: https://computationalrobotics.seas.harvard.edu/SceneAgent/

![](images/c7e25a32ce7e2e4a5634bd5a1efae9e6d14e0960572d3f0999041dc7d84c66b3.jpg)  
Figure 2 SceneAgent pipeline. Agents coordinate reconstruction, segmentation, physical modeling, and scene export, using simulator renders to inspect and repair intermediate outputs. Inputs include RGB image sequences, 3DGS and photogrammetry assets, AI-generated 3DGS scenes, and LiDAR scans. Section 3.6 extends the pipeline to videos of dynamic scenes.

## 1 Introduction

Generalist robot policies [5, 6, 23, 37, 38, 40] learn from large teleoperated datasets to perform a wide range of manipulation tasks specified by language instructions. Collecting these datasets is labor-intensive and can take months or years. Similarly, evaluating the resulting policies in a target environment requires physical robot trials, which are time-consuming, difficult to reproduce, and costly to scale [2].

Simulation offers a way to scale both data collection and evaluation. Reconstructed environments can generate demonstrations [59, 63, 67], support policies that transfer to physical robots [27, 45, 68], and provide estimates of real-world policy performance [19, 30, 31, 47, 69]. In particular, 3D Gaussian Splatting (3DGS) [22] reconstructs photorealistic scene appearance from images, enabling simulators to render observations grounded in a captured workspace [11, 21, 45].

Visual reconstruction alone, however, does not provide the object geometry, physical parameters, or articulation needed for interaction. Real-to-sim systems recover these components by combining segmentation, mesh generation, pose alignment, and physical modeling [12, 19, 47, 68]. SimFoundry, for example, supports articulated objects and estimates their physical parameters [47]. A remaining challenge is to coordinate these components across diverse inputs and correct errors that arise between stages. An incorrect mask, scale, or transform can produce an unusable object even when the individual models return plausible outputs.

We present SceneAgent, an agentic pipeline that converts 3D and 4D captures into interactive simulation environments (Figure 2). Agents execute reconstruction and physical modeling tools, inspect rendered outputs in the target simulator, and repair visible errors before the scene is used downstream. The resulting environments contain segmented objects with estimated physical parameters, decomposed parts, and articulation or soft-body behavior where needed. For 4D captures, which are videos of dynamic scenes, agents also fit simulated dynamics such as liquid flow, deformation, and destruction mechanics to the observed scene (Section 3.6). SceneAgent also generates digital sisters—related object variants—and combines captured objects with artist-authored environments to expand the range of training scenes.

We evaluate reconstruction quality on a cluttered bin scene from GraspClutter6D [3] and a Messy Kitchen dataset from CollectData [8] using coding agents with and without SceneAgent, and compare with SimFoundry. Here, a harness is the software environment in which a coding agent runs, and skills are the reusable instructions and tools that guide its conversion workflow. With SceneAgent, F1 at 1 cm improves for every harness with scorable paired outputs, and reconstruction quality is much better than SimFoundry’s (Figure 5). On three dynamic scenes from 4DCodeBench [51], where agents reconstruct simulations from videos, SceneAgent raises the mean dynamic IoU of Gemini 4 Argon from 0.86% to 99.09% (Figure 3). We also demonstrate simulated interactions and preliminary real-world deployment after policy learning.

![](images/57ca74e3201ff5929244922d84c1cc249f70a7ca7f530339a8a448af0ac6f196.jpg)  
Figure 3 4DCodeBench dynamic IoU against tokens per task on the synthetic tasks Z\_06, B\_09 and Z\_12. Baseline scores come from the supplied 4DCodeBench reference table. Baseline tokens are the published averages over 100 tasks.

In summary, our contributions are:

• An agentic scene-conversion workflow that integrates reconstruction, physical parameter estimation, part decomposition, and articulation with simulator-based visual inspection and repair, and that extends from 3D captures to 4D captures by fitting simulated dynamics to the motion observed in a video.

• Scene augmentation through digital sisters and hybrid environments that combine captured objects with artistauthored assets, together with an automated research loop that turns failures of a policy on the real robot into simulated training data.

• A quantitative study of reconstruction quality across coding-agent harnesses on static captures and on dynamic scenes from 4DCodeBench, complemented by qualitative examples of articulated, deformable, and contact-rich interactions and preliminary real-world policy transfer.

## 2 Related Work

Real-to-sim reconstruction and policy evaluation. Reconstruction-based systems combine scene appearance with interactive geometry to support policy evaluation and training. PolaRiS [19] combines a Gaussian-splat background with generated object meshes and evaluates how simulated policy performance tracks real-world outcomes. Sim-Foundry [47] reconstructs scenes from video, supports rigid and articulated objects, and generates object, scene, and task variations called digital cousins. Its articulation module segments movable parts, generates URDF joints,

Mean tokens per task (M) by stage
<table><tr><td></td><td>Codex GPT-6 Astra total 83.7 / 6.4</td><td>Claude Code Opus 5.5 total 193.2 / 155.9</td><td>Antigravity Gemini 4 Argon total 14.8 / 10.3</td><td>GLM-5.3 (Stirrup) GLM-5.3 total 123.1 / 102.8</td><td>Grok CLI Grok 4.7 total 70.9 / 14.2</td></tr><tr><td>SO Setup skill discovery</td><td>5.2 0.5</td><td>0.2 0.1</td><td>1.0 0.3</td><td>0.2 |&lt;0.1</td><td>1.7 0.1</td></tr><tr><td>S1 Video and camera frames, camera fit</td><td>12.3 0.9</td><td>11.5 17.7</td><td>0.2 0.5</td><td>22.0 8.5</td><td>9.5 1.2</td></tr><tr><td>S2 Static scene tank, obstacles</td><td>8.3 0.8</td><td>26.5 28.7</td><td>0.1 0.5</td><td>25.1 19.1</td><td>10.0 5.2</td></tr><tr><td>S3 Physics simulation continuum, MPM, fluid</td><td>30.4 1.9</td><td>115.9 82.8</td><td>10.7 17.1</td><td>19.9 18.2</td><td>5.0 1.2</td></tr><tr><td>S4 Surface meshing persistent dynamics</td><td>1 7.4 0.8</td><td>14.5 10.4</td><td>10.2 0.4</td><td>13.0 1.4</td><td>20.7 1.5</td></tr><tr><td>S5 Rendering Cycles, visual QA</td><td>16.6 1.0</td><td>12.6 16.3</td><td>12.1 0.3</td><td>26.1 46.9</td><td>15.5 3.2</td></tr><tr><td>S6 Checker clean-room rebuild</td><td>13.4 0.5</td><td>12.1 9.9</td><td>0.4 1.3</td><td>17.0 8.6</td><td>8.5 1.7</td></tr></table>

4DCodeBench B\_09, Z\_06 and Z\_12: mean over the three cases of the tokens each conversion stage used. All cells share one scale. Header totals are the sums over S0-S6, with SceneAgent / without.

Figure 4 4DCodeBench mean tokens per task by conversion stage, with and without SceneAgent skills, on the synthetic tasks Z\_06, B\_09 and Z\_12.

and uses simulated motion and VLM feedback to refine them; it also estimates link mass, joint friction, and damping.   
GASE [68] and Mirage2Matter [12] likewise combine Gaussian-splat appearance with interactive object geometry.

Other work addresses factors that determine simulation fidelity: joint refinement of geometry and calibration [35], reconstruction for motion planning [53], calibration and perturbation testing of a robot workspace [50], and the effect of shadows and reflections on policy evaluation [71]. Genie Sim 3.0 [66] combines scanned environments with generated scenes, while MolmoB0T [9] studies transfer from large-scale procedural environments. SceneAgent focuses on coordinating reconstruction and physical modeling tools across input types and using agents to inspect and repair their outputs throughout conversion.

Demonstration generation and scene variation. Digital reconstructions can expand limited demonstrations into larger training datasets. Real2Render2Real [67] synthesizes demonstrations from object scans and a human video using kinematic rendering. RoboSplat [63] varies object pose, appearance, and embodiment while adapting demonstrated trajectories, and ExoGS [59] replays exoskeleton demonstrations through editable Gaussian-splat assets. ManiS plat [14] reconstructs a scene from robot video to synthesize additional trajectories. TwinRL [62] uses a reconstructed environment to expand fine-tuning data and identify configurations for human-in-the-loop reinforcement learning. Our demonstration workflow samples layouts, settles objects under physics, executes a scripted expert using simulator state, and retains successful episodes for policy fine-tuning.

Physical properties, articulation, and deformation. Visual captures leave many physical properties unobserved. Material-informed Gaussian Splatting [17] transfers material labels to reconstructed geometry. Phys2Real [58] combines VLM priors with interaction-based estimates, while D-REX [29] identifies mass through differentiable simulation of observed interactions. ReconPhys [57] estimates deformable dynamics from video, and 4DCodeBench [51] evaluates agents that reconstruct dynamic scenes from video as executable simulation and rendering programs, the setting of our 4D extension. Complementary methods address cable simulation [64], articulated reconstruction [33], and geometry refinement using physical consistency [26]. In this work, we do not aim to solve the inverse physics problem of accurately recovering physical parameters from input 3DGS. This problem can remain ill-posed even with robot– object interaction data. Instead, SceneAgent combines semantic features and VLM estimates to obtain approximate, physically plausible parameter values, intended as starting points for domain randomization of physical properties during policy training. We leave accurate physical parameter identification to separate future work.

Table 1 System comparison. Capabilities of SceneAgent and recent real-to-sim systems for policy learning and evaluation.
<table><tr><td></td><td>PolaRiS [19]</td><td>Mirage2Matter [12]</td><td>SimuScene [26]</td><td>GASE [66]</td><td>SimFoundry [45]</td><td>SceneAgent (ours)</td></tr><tr><td>Scene construction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Automatic scene construction</td><td>X</td><td>0</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Agentic inspection and repair</td><td>X</td><td>X</td><td>0</td><td>X</td><td>0</td><td>√</td></tr><tr><td>Photogrammetry and LiDAR inputs</td><td>X</td><td>X</td><td>X</td><td>x</td><td>X</td><td>√</td></tr><tr><td>Generated-world inputs</td><td>x</td><td>X</td><td>x</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Hybrid captured and authored scenes</td><td>X</td><td>x</td><td>X</td><td>X</td><td>0</td><td>√</td></tr><tr><td>4D scenes from video</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Predictive physics</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Articulated objects</td><td>X</td><td>x</td><td>X</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Physical parameter estimation</td><td>x</td><td>x</td><td>x</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Deformable objects</td><td>X</td><td>x</td><td>x</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Fluids</td><td>X</td><td>x</td><td>x</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Fracture and tearing</td><td>X</td><td>x</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Predictive infill</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Occluded background</td><td>X</td><td>x</td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Hidden object parts</td><td>x</td><td>x</td><td>0</td><td>x</td><td>X</td><td>√</td></tr><tr><td>Digital sisters</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object sisters</td><td>X</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Scene sisters</td><td>X</td><td>x</td><td>X</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Task sisters</td><td>x</td><td>X</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Learning and evaluation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sim-to-real training</td><td>x</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Real-to-sim policy evaluation</td><td>√</td><td>X</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Multi-embodiment</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Multi-simulator export</td><td>X</td><td>x</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Self-improving skills and tools</td><td>x</td><td>X</td><td>x</td><td>x</td><td>X</td><td>√</td></tr></table>

✓ supported, partial, ✗ not supported.

Generative policy-evaluation environments. WorldGym [44], WorldEval [28], and RoboWorld [20] investigate policy evaluation through generated video. SceneAgent constructs explicit geometry, physical parameters, and joints for execution in a physics simulator. This representation makes object properties available for inspection and editing.

## 3 SceneAgent

SceneAgent converts a capture into an interactive simulation environment through three stages: agent-guided reconstruction and repair, scene augmentation, and demonstration generation. We first describe the agent workflow, then the representations and conversion tools it coordinates, and finally the downstream training procedure. Section 3.6 extends the workflow from static 3D captures to videos of dynamic scenes.

## 3.1 Agent Orchestration and Verification

Agents execute the conversion tools and inspect their outputs in the target simulator. This inspection is usefu at transitions between models, where file conventions, coordinate frames, and segmentation errors can affect the assembled scene. For example, an object mesh may have the correct shape but appear at the wrong scale, orientation,

F1 at 1 cm (%) - higher is better  
![](images/afd7880cd3d2ad10e2dfedb8442262d0792ed4f5049ea60e5294bee7bdf8af48.jpg)  
Chamfer (cm) - lower is better

![](images/87fd3dae86af8345f944a84db57891cdd78a24752fdec707248c465769eacbbb.jpg)  
GraspClutter6D hard/000091: 20 YCB objects in a bin, 13 RGB views. Mean over the 20 ground-truth objects; missing objects score F1 = 0.

Figure 5 Reconstruction of a cluttered bin scene from GraspClutter6D (hard/000091, 20 YCB objects) with and without SceneAgent skills, and with SimFoundry. Higher F1 at 1 cm and lower Chamfer distance indicate closer agreement with the dataset’s ground-truth object meshes.

or position after insertion into the environment. Agents render the scene, inspect these discrepancies, and correct the corresponding assets or transforms. Figure 7 summarizes the tools and skills available to the agents.

The orchestrator agent first plans the full conversion and processing steps from the input state, then the ingest agents prepare the input data and build the 3dgs, and finally the runner agent executes the processing tools with actor-critic visual review of each step. Alignment and orientation skills adjust object transforms and the background’s placement in the simulator. A packager assembles the scene, and an inspector reviews the exported result in Isaac Lab. Object-, scene-, and task-variation skills then generate digital sisters and alternative layouts.

The workflow supports both scene-level conversion and inspection of individual objects. Agents can examine close-up renders to identify incomplete segmentation, missing geometry, or incorrect placement; Figure 21 shows examples of these detailed views. This iterative correction concerns the current conversion.

Each agent collaborates with the hardness and passes state through files in the scene directory. Following the orchestrator agent, they execute the conversion plan following a manifest for each step then a ledger of accepted or rejected outputs. Then per converted object in the scene, the reviewer agent inspects the object registration overlays, eight isolated views (six axis-aligned and two oblique), and finally two views of the object in the fully assembled scene. If a problem is discovered, such as an alignment, articulation, or decomposition issue, the agent retries by proposing a corrected pose or other fix and reviews again.

The alignment residuals in Figure 7 come from a trimmed similarity ICP, seeded with the SAM3D pose, that aligns each mesh to the scene. We use the normalized median and 90th-percentile point distances, coverage, and silhouette agreement, and accept a candidate only when all four pass fixed thresholds. Procedures from each conversion are saved to files that later conversions can read.

The agents also maintain SceneAgent’s skills and tools and are instructed to research methods that may improve their conversion results, and if they identify a potential solution, request to evaluate it, and then modify their skills. The candidate suggested change to the tools and the current baseline are then run on a set of static 3D, dynamic 4D, and video-to-4D benchmarks, and the change is kept only if it clears fixed improvement and regression thresholds. For

<table><tr><td></td><td>Codex GPT-6 Astra total 271.2 / 10.0</td><td>Claude Code Opus 5.5 total 838.2 / 20.6</td><td>Antigravity Gemini 4 Argon total 65.2 / 15.5</td><td>Muse Muse 1.0 Wonder total 208.4 / 8.6</td><td>Stirrup GLM-5.3 total 305.5 / 91.6</td><td>Grok CLI Grok 4.7 total 118.4 / 11.6</td></tr><tr><td>SO Setup</td><td>10</td><td>10</td><td>|&lt;0.1</td><td>0</td><td>0.2</td><td>10</td></tr><tr><td>skill discovery S1 Cameras and 3DGS</td><td>|&lt;0.1 17.0</td><td>|&lt;0.1</td><td>0.2 16.9</td><td>0 4.6</td><td>10 25.1</td><td>0.3 13.0</td></tr><tr><td>COLMAP, Splatfacto</td><td>0.5</td><td>185.1 2.6</td><td>1.8</td><td>0.7</td><td>1.7</td><td>10</td></tr><tr><td>S2 2D masks grounded masks, visual QA</td><td>29.0 0.1</td><td>77.7 1.8</td><td>17.8 3.0</td><td>3.3 0.9</td><td>14.6</td><td>36.3</td></tr><tr><td>S3 3D instances</td><td>2.2</td><td>44.3</td><td>15.8</td><td>2.2</td><td>3.4 4.3</td><td>0.3 3.0</td></tr><tr><td>SplatGraph clustering</td><td>0.6</td><td>1.1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>S4 Physics priors semantics, materials</td><td>14.4 0</td><td>37.1 1.6</td><td>18.2 10</td><td>110.8 0</td><td>19.6 4.4</td><td>18.2 1.1</td></tr><tr><td>S5 Mesh generation</td><td>36.1</td><td>212.7</td><td>19.3</td><td>33.0</td><td>30.5</td><td>20.7</td></tr><tr><td>SAM-3D meshes</td><td>0.1</td><td>3.1</td><td>6.1</td><td>3.4</td><td>17.4</td><td>1.6</td></tr><tr><td>S6 Pose and export</td><td>176.6</td><td>225.4</td><td>24.8</td><td>55.6</td><td>142.5</td><td> 32.6</td></tr><tr><td>registration, settling, USD</td><td>14.8</td><td>4.1</td><td>3.1</td><td>1.3</td><td>45.0</td><td>5.0</td></tr><tr><td>S7 Rendering</td><td>15.9</td><td>55.9</td><td>2.3</td><td>99.0</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>68.8</td><td>14.7</td></tr><tr><td>render, verification</td><td>3.9</td><td>6.2</td><td>1.2</td><td>2.2</td><td>29.7</td><td>3.2</td></tr></table>

GraspClutter6D hard/000091: tokens each conversion stage used in the single conversion run. All cells share one scale.  
Header totals are the sums over S0-S7, with SceneAgent / without.

Figure 6 Tokens used in each conversion stage of the GraspClutter6D bin scene (hard/000091), with and without SceneAgent skills.

external users of a public release, a passing change becomes a draft pull request once the user approves it. This then initiates an independent benchmark run and maintainer review by the authors before it is merged.

## 3.2 Input Preparation and Scene Representation

For RGB image sequences, the pipeline uses COLMAP [49] to estimate camera poses and Nerfstudio’s splatfacto big [55, 65] to train a 3DGS reconstruction. Agents can inspect and correct capture-processing issues, including per-image color calibration [60]. The pipeline also accepts existing 3DGS files, photogrammetry assets, and LiDAR captures, as well as generated scenes from models such as World Labs Marble [61], NVIDIA Lyra [4], and Tencent HY-World [16]. Videos of dynamic scenes follow the 4D pathway of Section 3.6.

For the Gaussian-splat pathway, we represent the capture as $\{ g _ { i } \} _ { i = 1 } ^ { N }$ , with $g _ { i } = ( \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , \mathbf { h } _ { i } )$ denoting its center, covariance, opacity, and spherical-harmonic color coefficients. Calibrated cameras c provide image projections $\pi _ { c }$ and camera-frame depths $z _ { c } .$ The output is a scene $\boldsymbol { S } = ( \mathcal { G } _ { \mathrm { b g } } , \{ o _ { k } \} _ { k = 1 } ^ { K } )$ , containing a completed background splat and K objects. Each object is represented as $o _ { k } = \left( P _ { k } , M _ { k } , T _ { k } , \theta _ { k } , \mathcal { T } _ { k } \right)$ , where $P _ { k }$ is its set of Gaussians, $M _ { k }$ its mesh, $T _ { k }$ its rotation, translation, and uniform scale, $\theta _ { k }$ its physical properties, and $\mathcal { T } _ { k }$ its joints, which may be empty.

## 3.3 Object Reconstruction and Background Completion

We associate semantic features with the Gaussians using image masks and embeddings, following the general approach of LangSplat [43]. SAM [24] produces object masks on every tenth input view by default; across our scenes this stride ranges from 1 to 10. OpenCLIP ViT-B-16 [18, 46] embeds each masked crop into a 512-dimensional vector, and DINOv2 ViT-S/14 [39] gives each object a visual feature vector. Both are inputs to the physics prior (Section 3.4); neither is used to decide which Gaussians belong to an object. Masks from different views are grouped into one object when they share 3D Gaussians: the centroids of their Gaussians lie within min(0.10, 0.6 e) reconstruction units of each other, where e is the object’s extent, and the two Gaussian sets have a containment of at least 0.18 or a Jaccard index of at least 0.08. SAM3D [48] generates object geometry.

![](images/544994ae8b7a6da1d7fb895c744bc88e381d736584a56d951eaced12b871ca29.jpg)  
Figure 7 Agent workflow and available skills. The agents coordinate tools for capture processing, object reconstruction, physica modeling, and simulation, and inspect intermediate results to guide corrections.

To associate Gaussians with a visible object, we require each Gaussian center to project inside its mask and lie near the front surface. Specifically, Gaussian i belongs to mask m from camera c when

$$
\pi _ { c } ( \pmb { \mu } _ { i } ) \in m \quad \mathrm { a n d } \quad z _ { c } ( \pmb { \mu } _ { i } ) \leq \hat { z } _ { c } ( \pmb { b } _ { i } ) + \operatorname* { m a x } \big ( \epsilon _ { a } , \ \epsilon _ { r } \hat { z } _ { c } ( \pmb { b } _ { i } ) \big ) ,\tag{1}
$$

where $\hat { z } _ { c } ( b _ { i } )$ is the smallest camera-frame depth among Gaussian centers projecting into the same coarse image cel $b _ { i }$ , a square of round $\left( { \it s } / 1 9 2 \right)$ pixels, at most 16, for an image whose shorter side is s pixels. We use $\epsilon _ { a } = 0 . 0 8$ in reconstruction units and $\epsilon _ { r } = 0 . 0 3$ . Because $\epsilon _ { a }$ is in reconstruction units, its metric size depends on the capture’s fiscale; in the kitchen, at 0.90–0.93 m per unit, it is about 7 cm. Let $A _ { m }$ contain the Gaussians satisfying (1), and let $\mathcal { O } _ { k }$ contain the masks grouped into object k. The object’s Gaussians are then $P _ { k } = \textstyle \bigcup _ { m \in { \mathcal { O } } _ { k } } A _ { m }$ . An optional camera-consensus mode retains only Gaussians assigned from at least $\operatorname* { m i n } ( \kappa , C _ { k } )$ fiof the $C _ { k }$ cameras observing the object, for a chosen threshold κ. Our experiments use $\kappa = 1 , \mathsf { s o } P _ { k }$ is the full union. Each generated mesh is aligned to its object’s Gaussians $P _ { k }$ by a trimmed similarity ICP with a single uniform scale, initialized from the SAM3D pose.

After removing the foreground Gaussians, we complete exposed background regions using detected planes. New fiGaussians receive colors from nearby plane surfaces in the input images, with optional LaMa texture inpainting [54]. The generated object meshes are assembled with the completed background and inspected in the simulator.

## 3.4 Physical Properties, Parts, and Articulation

For each object, we combine semantic features, predictions from a per-Gaussian physics model, and a vision-language model (VLM) to estimate material and physical properties. These include friction, rigidity, mass, and density. The implementation uses 10 material classes, 6 body types, and 8 collision primitives, and records the resulting properties in the exported scene.

The physics prior’s per-Gaussian class logits and property estimates are averaged over $P _ { k }$ . A provisional material class is selected by a weighted vote,

$$
y _ { k } ^ { \star } = \arg \operatorname* { m a x } _ { y } \sum _ { v \in V _ { k } } w _ { v } { \bf 1 } [ y _ { v } = y ] ,\tag{2}
$$

where each source $v \in V _ { k }$ proposes class $y _ { v }$ with weight $w _ { v }$ . Sources comprise the physics prior, the VLM, and keyword priors derived from the object name. Weights use source confidence or a fixed value for keyword priors and are scaled by 0.4 when $y _ { v }$ is unknown. A subsequent VLM audit and recorded material review may replace the provisional class.

The physics prior is a multilayer perceptron whose inputs are DINOv2 and OpenCLIP features and 16 geometric quantities. It has output heads for material class, body type, and collision primitive, and it regresses nine numerical properties. It is trained on object records from GSO, YCB, and ABO, and its targets are labels written by a VLM, not physical measurements. The VLM audit decides the material, and we treat the numerical values as approximate priors.

The model regresses mass and density as logarithms. Young’s modulus, Poisson’s ratio, and the rendering materia parameters are looked up from the material class. Rigidity is a unitless score between 0 and 1. It enters the simulation only through deformability, and an object becomes a soft body when its deformability is at least 0.65.

The ten material classes are plastic, metal, wood, fabric, paper and cardboard, rubber and foam, glass and ceramic, food and organic matter, liquid and granular matter, and unknown. The six body types are dynamic rigid, static rigid, kinematic, soft deformable, cloth shell, and fluid or granular. The eight collision primitives are box, sphere, cylinder, capsule, convex hull, triangle mesh, tetrahedral mesh, and cloth mesh. The Isaac Lab export uses mass, friction, restitution, body type, and collision approximation, and it stores the remaining properties as USD metadata. The recorded material review is automatic. It is signed by a coding agent, not a person.

PartSAM [73] decomposes each reconstructed object into parts. A VLM inspects these parts for missing geometry, such as the tip of a marker hidden beneath its cap. When completion is needed, CubePart [70] generates the missing region, which is appended to the mesh. Articulate Anything [25] then proposes joints and motion ranges in URDF format from image or text inputs. Agents review the resulting articulation in simulation. Figure 8 follows this process from a marker image to a simulated writing interaction.

![](images/33071935eb1191b8c17c0aeaa4c38a0d2581eb05d95d9a586d12ed2dfa98ce74.jpg)  
Figure 8 Marker reconstruction and interaction. A single product image is converted into object geometry and parts, including completion of the tip beneath the cap. Two Franka Emika Panda arms with Sharpa Wave hands manipulate the resulting marker and write on a whiteboard in simulation.

Objects requiring deformation are represented as soft bodies. Figure 9 illustrates the estimated rigidity and simulated deformation of a corduroy cap, and Figure 10 shows additional objects at rest and under deformation.

01 Photogrammetry capture  
![](images/0e76691a3abc37913702c53f24e9909bb71beea51ff7ad967c510e5f05bde263.jpg)  
Source appearance and geometry.

02 Assigned regional stiffness  
![](images/9fcd40ff945ec14138d29fd4669ef71b506faaf83b2361ad262e773be5b0aa62.jpg)  
Hand-authored reinforcement, 350-6,000 N/m. Blue to red: log-scaled display; not measured.

03 Scripted contact  
![](images/fd2bf6b4e0c974579c72214b7b17ca6cf37d20a7e3d748518f2150b725ab2a1e.jpg)  
Crown pressed by a simulated puck. Hand-set physics; no learned policy.

Figure 9 A corduroy cap captured with photogrammetry, annotated with estimated rigidity, and simulated under contact. The example illustrates the conversion from visual geometry to a deformable asset.

01 Textile rack · orbi  
![](images/c877d0a2b611d239ef30a87c3f0f49148c0e01e13c41c132e45aa01e8ff08314.jpg)  
04 Firefighter hose · unwind

02 Hoodie · lift and drape  
![](images/813b9ba86b64bcca06f906cf07ef31fa52cff888afb742b5d3e8665481a8c24a.jpg)  
05 Liquid Death hat · physics

03 Towel rack · orbit  
![](images/a8685f8d3cf32f83c941369d40dcad6740821a2acc6fac7c7f9b21d0574b77ca.jpg)

![](images/07e9e6bc0798fd9db4f791b17cb0770e91fa92b736264ca150f37c10e9aa23b3.jpg)

![](images/49e4af5d5c4b1c9cfdfee041eca7c421cd8f84695a9f893323ba7839df3c3012.jpg)

![](images/ed4f6c1fd983617ea60cb556465079f8cd43f4adb4e0c56a66a4061e4a33f341.jpg)  
06 Extension cord · unwind

![](images/6738bc609ad39747ed2c7dfc3b4b5a1b5b8f16b26699f285bd95637b56b5f3b0.jpg)  
Figure 10 Deformable assets converted from Polycam [42] captures. Each object is shown at rest and under simulated deformation.

01 Plush · press  
![](images/64642dea31d4a7f1cc97eff5b5a28f33eef11588a4d6f4e8630b5c82d5e0a126.jpg)

![](images/9360a702524c972b557aa8d96c5f98b39b07cb4cad6f019d820006b8f5ad8ab0.jpg)

![](images/d7d1715e63aafa1408afed94e0744e9dab26f0eda7e310878adad04037b8ea55.jpg)  
02 Milk · pour

![](images/cf7ca50839e9cabc2351f44d89fc276f9d9864779ac8df02d9a2da3c29b321be.jpg)

![](images/c1b1979d8547ab308dbc76c45518d0a8890801c43bf21bdb5fff43c98624203a.jpg)

![](images/032cd3e6f1d4dd8162c6c7311d1825c37ef3f5b38273e0ada8401450d8ca9070.jpg)  
Figure 11 Contact and fluid simulation. A Franka Emika Panda arm with a Sharpa Wave hand presses a plush toy (01). A cup tips and pours milk onto a table (02). The milk is simulated with MPM.

![](images/d64e37f104ba0180dc67e29f9b373e5d7fd40a9deb7d98eac8f6432177207eb0.jpg)  
Figure 12 Fracture simulation. A Dexmate Vega with two Sharpa Wave hands tears a parchment sheet from a roll and covers a bowl of fruit (01–02). Ceramic jars fracture after being knocked from a pedestal or dropped (03–04). The parchment parameters are engineering estimates.

## 3.5 Fracture Simulation

We also explore material failure during manipulation, including paper tearing and ceramic fracture. Figure 12 shows a Dexmate Vega with two Sharpa Wave hands tearing parchment from a roll and placing it over a bowl, alongside examples of jars shattering after impact. The parchment uses an orthotropic thin-sheet finite-element model in NVIDIA Warp [32], with irreversible cohesive fracture along interior edges. The tear path emerges as interfaces fail. In the illustrated rollout, 49 interfaces fail progressively, leaving a detached sheet that covers 95% of the bowl opening after the hands withdraw. The strength, fracture energy, and friction parameters are currently engineering estimates; prediction from captures and validation against measured break forces remain future work.

Coverage is the fraction of points on a 71 by 71 grid over the bowl opening that lie beneath the detached sheet, counting a point only where the sheet is no lower than 12 mm below the rim, and we report the lowest value over the frames after the hands withdraw. The ceramic examples use a separate model. Each scanned vessel is divided into connected patches of its original surface triangles, an inner wall 4.5 mm thick is inferred for each patch, and neighboring patches are joined by bonds in NVIDIA Blast that break irreversibly once the stress in the wall exceeds the material strength. Cracks therefore follow the boundaries between patches, and the fragments are native PhysX rigid bodies.

The material values are engineering assumptions that a scan cannot measure, with a tensile strength of 6 MPa, a Young’s modulus of 6 GPa, a fracture energy of $1 4 \mathrm { J / m ^ { 2 } }$ , and a density of 1850 kg/m<sup>3</sup> for every vessel. Neither model is part of the automatic scene export. Each example is authored as its own Isaac Lab scene, the parchment in the kitchen scene and the vessels from their Polycam captures, which a preparation script converts to USD and divides into fragments. All examples use scripted controllers, and none uses teleoperation or a learned policy. The robot hands tear and place the parchment, a hand holding a cloth wipes one vessel and pushes it off its pedestal, and the other vessels are dropped without a robot.

01 Tear

02 Cover

03 Shatter

04 Drop

## 3.6 From 3D to 4D Captures

The pipeline described so far converts static captures. A video of a dynamic scene also shows how the scene moves, such as liquid flowing around obstacles, a gel deforming under impact, or a fruit breaking apart. We refer to such a video as a 4D capture. SceneAgent converts 4D captures into simulations that reproduce the observed motion. The agents follow the same workflow of planning, tool execution, rendering, and repair as for 3D captures (Section 3.1), while the scene representation, the estimation of physical parameters, and the inspection of results are extended to sequences of frames.

The input is a sequence of frames $\{ I _ { t } \} _ { t = 1 } ^ { F }$ from a camera with unknown intrinsics K and extrinsics $E ,$ which are estimated from the video. We extend the scene of Section 3.2 to $\mathcal { S } = ( K , E , \mathcal { G } , \{ d _ { m } \} )$ , where $\mathcal { G }$ is the static geometry, such as containers, obstacles, and the surrounding room. Each dynamic material $d _ { m } = (  { \mathbf { x } } _ { m } ^ { 0 } , \theta _ { m } , \Phi _ { m } )$ has an initial state $\mathbf { x } _ { m } ^ { 0 }$ , physical parameters $\theta _ { m }$ , and a simulator $\Phi _ { m }$ that advances its state from one frame to the next,

$$
\mathbf { x } _ { m } ^ { t + 1 } = \Phi _ { m } \big ( \mathbf { x } _ { m } ^ { t } , \theta _ { m } , \mathcal { G } , \{ \mathbf { x } _ { n } ^ { t } \} _ { n \neq m } \big ) ,\tag{3}
$$

with contact between the material, the static geometry, and the other materials. At every frame, the scene exports a triangle mesh $M _ { m } ^ { t }$ for each material together with the positions of its particles or vertices. Each particle or vertex keeps its index over the whole sequence, which gives persistent trajectories $X _ { m } \in \mathbb { R } ^ { F \times N _ { m } \times 3 }$ . Rendering the scene with the estimated camera produces frames $\hat { I } _ { t }$ that are compared with the input frames $I _ { t }$

01 Watermelon · drop  
![](images/24ad2d8f1de11aeab83958f678c28f3a8fb8d5f1bd9af4e13bfb032b7be25436.jpg)

![](images/eef29284d99a9ab92c20e2cf49da28ea2b0a92ad2c9793be5a2321230ef1d250.jpg)  
02 Jelly · press

![](images/b63a5089707b8bf6fa5945b46aa4e83e3fe0057e7587a2b1f3face72bca6ffa2.jpg)

![](images/7396b7fc366fad5f730f8d214124936d0c045bff2994d222e59326d66a15dc80.jpg)

![](images/90b6bd20c575cbf6dfa1d9207c3447ecf291a7f6f13da075525a97d9124175c6.jpg)

![](images/27e0d520a32b23b2ac8d09f1c9732acaa533089249ef052bcf5c699a05e117ca.jpg)

03 Salmon · cut  
![](images/cff221e5825a3e8a0ea2d4778e1e1b669344562c067f9f07fd191faeb5f86b4e.jpg)

![](images/39ec1a1ff193e9b08ec8fbd467e65a1b70cf3057b28d6fe1fcde9549d9837437.jpg)

![](images/c938a79bba932629a3e6c842ac4bb2447fd9997d4a41dbe48a4381fb159b51e5.jpg)  
Figure 13 Fracture and cutting. A watermelon breaks apart on impact after a drop (01). A Franka Emika Panda arm with a Sharpa Wave hand presses into a jelly, which cracks (02), and cuts salmon with a chef’s knife (03). The watermelon and jelly use cohesive finite elements in Newton.

SceneAgent’s 4D skills cover video-to-4D conversion, 4D scene completion, and fracture simulation. The agent first extracts reference frames from the video and estimates the camera, for example from the vanishing points of a container’s edges and from object silhouettes, and checks the estimate by projecting the static geometry onto the frames. It then builds the static geometry and selects a simulator for each material. Gels, elastoplastic solids, and liquids interacting with rigid bodies are simulated with the material point method (MPM) [15, 52], incompressible liquids with FLIP [72], and brittle objects with fracture models. The agent fits the initial states and physical parameters to the motion in the video. Liquid and deformable surfaces are extracted from the simulated particles at every frame [72]. Particles are not deleted or reseeded during the simulation, so each trajectory follows the same piece of material for the whole video.

To check its results, the agent renders the simulated sequence with the estimated camera and compares it with the input video in side-by-side and overlay views at early, middle, and late frames. Depending on the discrepancy, it revises the camera, the static geometry, the initial states, or the physical parameters and compares the views again. The final scene is produced by a single build script that does not read the input video or any external assets, so its frames, meshes, and trajectories can be regenerated from scratch and checked for format, frame count, and resolution.

Unlike the priors used for static captures (Section 3.4), the initial states and physical parameters of a 4D capture are constrained by the observed motion, and the agent adjusts them until the simulation matches the video. The fitted parameters reproduce the observed motion, but they are not necessarily unique and have not been tested on other interactions. The solvers are of the same types as those in our deformable, fluid, and fracture examples (Figures 11–13).

## 3.7 Scene Export and Editing

The converted scene is packaged in Universal Scene Description (USD) format [41]. Our experiments use Isaac Lab [34]; the pipeline also targets MuJoCo [56] and Unreal Engine [10]. An interactive Three.js viewer (Figure 14) supports robot placement, background cropping, asset insertion, and manual correction. Users can add converted objects or assets from repositories such as NVIDIA Omniverse [36] before using the scene for a task.

SceneAgent exports each scene as OpenUSD. We worked primarily in Isaac Lab, and we also tested and verified the exported scenes in MuJoCo and Unreal Engine. OpenUSD’s physics schema describes rigid bodies, mass, collision shapes, joints, and physics materials with static and dynamic friction, restitution, and density, so a simulator that reads the schema can load these properties from the file. Our export marks each movable object as a rigid body with its mass and collision geometry, and stores its friction, restitution, density, and other estimated properties as attributes on the same object.

The schema does not cover deformable bodies, fracture, or fluids and gels. Isaac Lab adds these through its PhysX extensions and through NVIDIA Warp, Newton, and Blast, which our soft-body, tearing, fracture, and fluid examples use. MuJoCo and Unreal Engine have their own deformable and destruction models, so these behaviors do not carry over through the USD file and would need to be set up in each engine.

## 3.8 Digital Twins and Sisters

The reconstructed scene provides a digital twin of the captured workspace. To introduce variation, we generate related instances of its objects, termed digital sisters, and compose scenes with randomized object placement and lighting. These variations aim to expose a policy to a broader range of appearances and configurations during training. Figure 15 shows seven reconstructed objects and their generated variants.

We generate digital sisters in two ways. The first perturbs the twin itself without calling a model. Bounded deformations change the shape of the object’s own mesh, and each result is checked for changes in extent and volume and for whether the object can still be grasped. A procedural texture of noise, hue rotation, mottling, and speckle is baked into the vertex colors. The second uses Meshy. For new object instances, a reasoning model (Gemini 2.5 Flash) divides the object into functional components and proposes variations of each, and an image model (Gemini 3 Pro Image) edits the object image one component at a time. Each edit is then checked for real-world plausibility and fit with the scene, and Meshy’s image-to-3D model lifts each accepted image to a textured mesh. For appearance alone, Meshy’s retexture model (Meshy-6) repaints the original mesh with its geometry and UVs unchanged.

The edit prompt asks the image model to change one named component by the proposed variation, keep the result realistic and ordinary, leave the other components unchanged, and match the scene. The retexture prompt asks for the same object with a small change in surface detail, such as fine grain or light wear, and rules out new components, holes, and changes in shape.

## 3.9 Hybrid Scenes from Captures and Digital Art

SceneAgent can also convert an individual captured object and place it in an artist-authored environment. This allows a task-specific asset, such as a device component or server-rack part, to be incorporated into an existing scene without recapturing the entire workspace. The same object conversion, physical modeling, and inspection tools apply to these hybrid environments.

The examples span grippers and dexterous hands. Figure 16 shows Franka Emika Panda arms with Universal Manipulation Interface (UMI) grippers [7], while Figure 17 shows Sharpa hands on Panda arms and a Dexmate mobile manipulator. These examples illustrate the range of environments and interactions that can be assembled; the policy-transfer study in Section 4.4 focuses on pick-and-place tasks.

## 3.10 Demonstration Generation and Policy Learning

Given a converted workspace and a robot configuration, the training workflow generates scripted expert demonstrations and uses them to fine-tune a pretrained vision-language-action (VLA) policy. The demonstration generator uses ground-truth simulator state to plan actions and renders observations from the robot’s cameras.

01 Scene editor  
![](images/bd0febe6205d01181833bf19ebcc4d5ff9b673abbdb02ca7fc9c11bd56f381df.jpg)

A Robot placement Robots from the library, with their tools and cameras.

E Digital sisters Generate object variants for the scene.

B Background cropping A box volume hides or keeps the Gaussians inside it.

F Parts and articulation Decompose an object into parts and propose its joints.

C Asset insertion Search the Omniverse library and add an asset to the scene.

G Operator A coding agent edits the scene and runs pipeline steps.

D Manual correction Move, rotate or scale the selection, or edit its pose and physics.

02 Experiments  
![](images/001fb3a38d987687c16dbc58ffe3b899a582703197ab349421f6d6efc9f0f8c2.jpg)  
Figure 14 The interactive viewer. (01) The scene editor for the lab scene with a Dexmate Vega, with the scene graph, the Three.js stage, the properties of the selected object, and the agent operator. Lettered callouts mark its main operations. (02) The experiment view of a digital-sister evaluation. Each cell is one object sister under one lighting setup and reports place success over ten trials.

![](images/93451bb754061c1183e3527db2e73ec0da035333860f489a83dd75126c382743.jpg)

Apple

Mug

Figure 15 Digital sisters of reconstructed objects. Each row shows one converted object followed by seven generated variants.

01 Kitchen

![](images/1d2d4f98359b10c87f4a374ef8b62c4ff508b38278f8128e30067656c3f681f1.jpg)

![](images/583159b345b110d4e845b2877885a2f3997f2f2c85fb02988b14debd21f4aeb6.jpg)  
04 Data center  
02 Warehouse

![](images/c07181a2017b179d465a38f8b39d555164870e0378e0bffc1761639c2f1ae48a.jpg)

![](images/ff87dc7b7f904a87da40fe7c84ebab186e7e2be2c3203414c26909aaa6895af7.jpg)  
03 Chemistry lab

![](images/f18e66a5e0fdb777142c5191cf562d13a3589bf6f91670bc14172775614bdfcc.jpg)

![](images/af727ee4b48fa2ee1ec328b012fc00770a8baaf34b0212e8e738cc57faf53d92.jpg)  
05 Studio  
06 Studio

![](images/d66f3bb925eeffd8fb6d13dfa8c099844f78c25c8c4cba2aea47ee54594dbffa.jpg)

![](images/a504c015ea444b7c1a353c817c854bbee9d87bd5cb23b90d200fd962d30643f7.jpg)

![](images/66e5737ee3ea29ba2020b43bd16e14cf99541bce994684c93c01a54627d754b6.jpg)

![](images/4be90a74fa57fc309c3147965c24730bca5942346d2a8b48c93783367e24302b.jpg)

![](images/5e64fabc6a8f1939f0accaa08b05169ae20b0437c2b3cb3254f0c6bd143d5b00.jpg)

![](images/0b21281902a1eed3a28e13fadf2ecfcd3198860b0d79557be1e5af888b0f77ac.jpg)  
07 Warehouse

![](images/c0d3894e2487013878e0834f07ff5a23f2a41f74ca6f11393dfd75b11f98a0d7.jpg)

![](images/392197c973d454e7b611681c466b8945f23675dac0f5eeeed033d4cf94acb5bd.jpg)  
08 Studio

![](images/8c865787e1c456d09d8cbb5a0b338dad0105177c08a1a4aa29affe3a1c814d78.jpg)  
09 Studio

![](images/95c6c707593e8d1118ef33183363dd60a60f9552cb46cb4efa8ea1e936487e72.jpg)

![](images/86cf12e1cb71221cf973d5c9b4ee1e20269f9581ebb0df1e9ce1811036955c34.jpg)

![](images/7c9f0a38a444393902ae35a6d7258f4b1eb83b42fbe33e828d0972c210f9c3f7.jpg)

Figure 16 Hybrid environments combining captured objects with artist-authored assets. Franka Emika Panda arms with UMI grippers are shown during and after contact-rich and deformable-object tasks in simulation.

Policy autoresearch loop. Policy training and evaluation optionally has agents that run in an automated autoresearch loop including identifying failure modes in reality or simulation, creating simulation digital sister task or scene simulations, and then correcting failures with a VLM to steer robot performance in simulation (Figure 19). The current policy is first rolled out on the real robot or in simulation, and agents review the recordings to find its failure modes and mark the segments where they occur. For each failure mode, an agent proposes likely causes and designs digital sisters that reproduce them, varying the relevant objects, placement, or lighting. The policy is then evaluated in these sisters. Where it fails, a coding model (GPT-6 Astra or Gemini 4 Argon) writes or revises the robot controller that serves as the expert for those cases. The controller is run in the sisters, and only episodes that pass the task’s success check are kept. These simulated recordings are added to the fine-tuning data, the policy is retrained, and the new checkpoint is compared with the previous one on a fixed set of paired validation trials before it returns to the real robot for the next round.

01 Studio  
![](images/c878a9eb57aec2d43b1dd92952bd411c89dc6a82d0960cf769f3ccdd831feac8.jpg)

![](images/09a21713592327a635df5211ce0739590a4d2f9874517a5b55f5c94bc66a4bb0.jpg)  
04 Kitchen

02 Dexterous hands  
![](images/8a5f6012d10d76883d4e41ecce4d0d8b4965514c4a37a2c02a1ad6650e259234.jpg)

![](images/3d0801502afa48b385f8f0a8c887b34633c264814bba8806b012731dae046bcc.jpg)

03 Dexterous hands  
![](images/23f51ab0c134f6f60a0734df894c9e519026d7d20195532fc0379064c2a32b23.jpg)  
05 Studio

![](images/0b44f0a562bb1da7e7f6fff6bff5f45a8503d0eee594f46250965f3993d14e55.jpg)

![](images/bb159ea8259d13e6b4af4a9544a850125991d85cc8ba2339608dd9209cd7ab3f.jpg)

![](images/087f8b42cca256e9fe106945fb52b5e61fd92cb906b4a6003d947bd62d7c9864.jpg)

![](images/6c6a95b3da28810683ae45335a637df3ee66488e21c23af4493c776947f6b975.jpg)

![](images/2d331d00f227b62e3a593a47c79d563e1c5d2b7cb8092b1f5c24602666ed56fc.jpg)  
06 Studio

![](images/e5778c2be7a21265c7142403d9f4b9e6cd7f482c30f6631837cd30c9e18db3d3.jpg)

![](images/53567fff1fa3f15e7096048372ef5b30eb37d1daf37eae49112f079298524f7d.jpg)  
Figure 17 Dexterous manipulation in converted environments. Sharpa hands mounted on Panda arms and a Dexmate mobile manipulator interact with reconstructed objects during multi-step simulated tasks.

01 Open  
![](images/6dfd32645cc7416977ba818ded0cb217db9ddfb96a7eba5f102f58f6982cabf2.jpg)

02 Type  
![](images/03adfba515f848c89be877dc010a7777e6840ad53d01cb46452fa8b85982c6fb.jpg)

03 Run  
![](images/0d559cf9b41ba24dc9f1778e8182dacf4b7a7ba7452de16c9711ed04ad9d58f7.jpg)  
Figure 18 A Dexmate Vega opens a laptop (01) and types a command on its keyboard (02–03).

Initial-state randomization. We vary object placement and size and perturb the camera and robot-base poses around their calibrated configurations. Objects are sampled within the task region and allowed to settle under simulated physics before execution. This provides physically settled initial configurations for demonstration collection.

Scripted expert and episode filtering. The expert adapts a pick-and-place routine to the sampled object poses: approach, descend, grasp, lift, carry, lower, release, and retreat. A success check determines whether the object ends inside the target container, and unsuccessful episodes are discarded. Observations are rendered by depth-compositing the mesh foreground, including the robot and manipulated objects, over the Gaussian-splat background. In the loop, later versions of the expert are the controllers written by the coding model, and they pass the same success check.

For each platform and task across the Franka Panda, the dexterous hands, and the Dexmate mobile manipulator, the controller is written by a VLM (GPT-6 Astra or Gemini 4 Argon), and the same approach is used in simulation and on the real robot due to the well-documented spatial understanding capabilities of these models. Platform-specific behavior, such as finger coordination in dexterous grasps and the coordination of the mobile base with the arms, is part of each generated controller.

Policy fine-tuning. Camera observations are paired with expert actions and a natural-language instruction sampled from alternative task descriptions. The resulting demonstrations are used for LoRA fine-tuning [13] of a pretrained VLA. The reported fine-tuning data are generated in simulation; the starting policy retains its pretraining history. The real-world deployment setup and current evaluation scope are described in Section 4.4.

![](images/d71e3ee4cb3118b8ed7c12421078676c460f5b21ed7128ad668317fe082a5ded.jpg)

Figure 19 Agentic real2sim2real policy autoresearch, illustrated with a Franka Emika Panda and Sharpa Wave hand. First, real task recordings expose failures then agents construct a digital twin, generate task and scene sisters, and evaluate the current policy. Coding agents revise expert controllers and rerun simulated tasks. Task-verified recordings feed policy fine-tuning, and fixed paired validation trials compare checkpoints before real-world re-evaluation.

## 4 Results

We first evaluate reconstruction quality on both a shared kitchen capture and a cluttered bin scene, then on dynamic scenes from 4DCodeBench, and finally present qualitative conversion examples and preliminary real-world policy transfer.

## 4.1 Reconstruction Quality Across Agent Harnesses

Setup and reference. Each harness converted the same scene from the MessyKitchens benchmark [1], once with SceneAgent skills and once without them. We score the delivered objects against the benchmark’s ground-truth meshes, which come from 3D scans of the real objects and are independent of every conversion run, including ours. We use these meshes as the benchmark provides them, without postprocessing or manual correction. Figure 5 lists the model and reasoning setting of each harness. Every run receives the same task prompt and the same version of the SceneAgent skills, and no person intervened during the runs.

Each delivered mesh is placed using its saved transform. A single rigid transform derived from a shared capture frame aligns the run to the reference; object scales are preserved, and no object is individually refit or repaired for scoring. We sample 100,000 points per object with three sampling seeds, average metrics across seeds, and weight objects equally. These seeds vary metric sampling, not the scene-conversion run.

We also evaluate on scene hard/000091 of GraspClutter6D [3], in which 20 YCB objects are stacked and occlude one another in a bin. The input is 13 RGB views at 1920×1080. We score the delivered objects against the dataset’s ground-truth object meshes, placed with its registered 6D poses. Each harness again converted the scene once with SceneAgent skills and once without them. Both conditions receive the same task prompt, except that the with-skills prompt also instructs the agent to use the SceneAgent skills. We also run the official SimFoundry pipeline [47] on the same 13 images.

Each run is aligned to the reference with a single similarity transform, and no object is individually refit or repaired for scoring. We sample 10,000 surface points per ground-truth object and match delivered objects to ground-truth objects with the Hungarian algorithm, using Chamfer distance as the matching cost.

Metrics. For each object and seed, let X be points sampled from the delivered mesh, $Y$ the fixed reference points, and $\begin{array} { r } { d ( x , Y ) = \operatorname* { m i n } _ { y \in Y } \| x - y \| } \end{array}$ . We report the symmetric mean nearest-point distance,

$$
d _ { \mathrm { C D } } ( X , Y ) = \frac { 1 } { 2 } \Big ( \frac { 1 } { | X | } \sum _ { x \in X } d ( x , Y ) + \frac { 1 } { | Y | } \sum _ { y \in Y } d ( y , X ) \Big ) ,\tag{4}
$$

and F1 at threshold $\tau ,$

$$
F _ { \tau } = \frac { 2 P _ { \tau } R _ { \tau } } { P _ { \tau } + R _ { \tau } } , \quad P _ { \tau } = \frac { 1 } { | X | } \sum _ { x \in X } \mathbf { 1 } [ d ( x , Y ) \leq \tau ] , \quad R _ { \tau } = \frac { 1 } { | Y | } \sum _ { y \in Y } \mathbf { 1 } [ d ( y , X ) \leq \tau ] ,\tag{5}
$$

with $\tau = 1$ cm and $F _ { \tau } = 0$ when ${ \cal P } _ { \tau } + { \cal R } _ { \tau } = 0 .$ On the kitchen capture, both metrics are averaged over sampling seeds and then over objects. On GraspClutter6D, F1 is averaged over all 20 ground-truth objects, with unmatched objects scoring zero, and Chamfer distance is averaged over matched objects.

Paired comparisons. On the kitchen capture, SceneAgent improves both metrics for every harness with scorable outputs in both conditions. F1 at 1 cm increases from 16.45% to 41.13% for Codex, from 7.07% to 19.29% for Claude Code, and from 6.33% to 17.97% for ZCode. Their Chamfer distances decrease by 1.35–2.61 cm. Grok Build with SceneAgent produces three of the four objects, omitting the knife block. On the matched three-object subset, its F1 increases from 5.67% to 17.68%, and its Chamfer distance decreases from 5.25 to 2.81 cm.

On GraspClutter6D, SceneAgent improves both metrics for every harness (Figure 5). F1 at 1 cm increases from 9.51% to 60.75% for Codex, from 5.58% to 60.46% for Claude Code, from 2.36% to 52.25% for Antigravity, from 7.19% to 52.70% for Grok CLI, from 5.75% to 41.39% for Muse, and from 6.93% to 40.45% for GLM-5.3. Chamfer distance decreases from 6.20–9.89 cm without SceneAgent to 1.43–3.43 cm with it. SimFoundry reaches 6.9% F1 and a 4.29 cm Chamfer distance, worse than every SceneAgent configuration on both metrics.

Incomplete runs and baseline comparison. Without SceneAgent, Antigravity and Muse produce no object geometry that can be scored. With SceneAgent, their F1 scores are 20.58% and 5.14%, respectively. SimFoundry [47] achieves 10.21% F1 and an 11.67 cm Chamfer distance on the same capture. Several SceneAgent-assisted configurations therefore exceed its F1 score under this protocol, although Muse does not. These comparisons concern reconstruction geometry in this kitchen, with respect to the stated reference.

## 4.2 Dynamic Scene Reconstruction on 4DCodeBench

Setup. 4DCodeBench [51] evaluates agents that reconstruct dynamic scenes from video as executable simulation and rendering programs. We use three of its synthetic cases, whose reference worlds provide 3D ground truth. They are a liquid dam break around rigid cylindrical pillars in a transparent tank (B\_09, 130 frames at $1 2 8 0 \times 7 2 0 )$ , a block of gel or liquid and a rigid sphere interacting in a transparent tank (Z\_06, 150 frames at $1 2 8 0 \times 7 2 0 )$ , and a layered watermelon breaking apart on impact (Z\_12, 81 frames at $1 6 0 0 \times 9 0 0 )$ , all at 24 frames per second.

Each agent receives the input video with the benchmark’s task description, world format, and render settings. It must deliver a build script that regenerates the 4D scene of Section 3.6 without reading the video. The scene contains the camera, a triangle mesh of every rendered object at every frame, persistent trajectories of the dynamic material when the solver tracks material identity, and a rendering with the frame count, resolution, and frame rate of the input video. Five harnesses converted each case once with SceneAgent skills and once without them: Gemini 4 Argon in Antigravity, GPT-6 Astra in Codex, Claude Opus 5.5 in Claude Code, Grok 4.7 in Grok CLI, and GLM-5.3 in Stirrup.

01 Kitchen (CollectData)  
![](images/35ac24c74d6dce6acb47e0f4437110921380d278a1f1b86d50390bc95f579ad3.jpg)  
Dexmate Vega 1 + two Sharpa Wave hands

02 Computational Robotics Lab  
![](images/84b034f24246478221bd980ef4b272753a0186d3831adf367d8cf3b0350ae91a.jpg)  
TeleAvatar + two Sharpa Wave hands

03 Greenhouse  
![](images/1df1dff2ed166402d86e510d47c3d664e012dea2dcb2a52149263c7bcb466e61.jpg)  
Panda + Sharpa Wave hand

04 Kitchen (Polycam)  
![](images/bf096689977db270cc4fd2f60f6d8830369d8c66a1e8bfc8054d3b90285217aa.jpg)  
06 Observatory (Matterport)  
05 Generated 3DGS Kitchen (World Labs)  
Panda + Sharpa Wave hand

![](images/a1808ef56d1220d7df4bddad2bbd47d4135b86223d60fdc01ad9dd404279ece7.jpg)  
Dexmate Vega 1 + two Sharpa Wave hands

![](images/93548ab1bac672d76fcd6bc62ee1e4d74e0dc1c7721eea3ac980d20cbb05193b.jpg)  
Dexmate Vega 1 + two Sharpa Wave hands

Figure 20 Converted scenes shown during simulated rollouts. Captured environments and objects can be combined to construct multi-step manipulation tasks. In the Matterport observatory, a Dexmate with two Sharpa hands pulls both guide ropes to slew the telescope.

Metrics. We report the benchmark’s dynamic IoU, which compares the image masks of dynamic matter in the reference, $D _ { t }$ , and in the reconstruction, $\hat { D } _ { t }$ , excluding a thin band $B _ { t }$ along their boundaries,

$$
\mathrm { D y n I o U } = \frac { 1 } { F } \sum _ { t = 1 } ^ { F } \frac { \left| ( D _ { t } \cap \hat { D } _ { t } ) \setminus B _ { t } \right| } { \left| ( D _ { t } \cup \hat { D } _ { t } ) \setminus B _ { t } \right| } .\tag{6}
$$

It ranges from 0 to 1, with higher values indicating a closer match, and we average it over the three cases.

Paired comparison. With SceneAgent, the mean dynamic IoU increases for all five harnesses (Figure 3): from 0.86% to 99.09% for Gemini 4 Argon, from 11.10% to 66.50% for Grok 4.7, from 52.16% to 55.95% for Claude Opus 5.5, from 12.31% to 39.28% for GLM-5.3, and from 7.79% to 36.38% for GPT-6 Astra.

Comparison with published results. Figure 3 also shows the 18 agents in the benchmark’s reference table, with token counts averaged over its 100 synthetic tasks. On these three cases, the highest mean dynamic IoU among them is 30.57%, by Claude Opus 5.5 [High] in Claude Code. These agents were run by the benchmark authors with their own settings and harness versions.

## 4.3 Qualitative Scene and Object Conversion

The converted assets illustrate scene-level reconstruction, object completion, articulation, deformation, and fracture. Figure 20 shows six converted scenes during simulated rollouts, including a Matterport observatory capture in which a Dexmate pulls both guide ropes to slew the telescope. Figure 21 shows six converted objects with close-up inspection views. The marker, cap, and fracture examples illustrate how additional geometry and physical modeling support interactions beyond rigid pick-and-place.

We used these converted scenes to train and evaluate $\pi _ { 0 . 5 } \ [ 4 0 ]$ on a range of diverse tasks with a Franka Emika Panda fitted with UMI grippers. Real-world rollouts are still in progress.

## 4.4 Preliminary Real-World Policy Transfer

We fine-tune pretrained policy $\pi _ { 0 . 5 } \left[ 4 0 \right]$ on demonstrations generated in reconstructed scenes of a university laboratory. The demonstration dataset used for finetuning contains different number of simulated episodes with object and scene variations. Real-world evaluation uses a Franka Emika Panda robots with UMI grippers, with a pick-and-place task. The pick-and-place task we choose to test the policies on is putting a toy eggplant into a basket. We determine a maneuver of the robot being successful as the Franka arm picks up the toy eggplant and drop it into the basket without dropping the toy eggplant in the process.

![](images/152e5809e3042919ecf96ed82c21746842f08ccc141fb80dc9bf0d9e4a5105d3.jpg)  
Figure 21 Six objects converted from public Polycam captures, with close-up views used during agent inspection. The examples illustrate reconstruction of detailed geometry and assembly of simulation assets.

We deploy a variety of variation and randomization during demonstration generation and testing. In order for the trained model to generalize in the real-world deployment, we employ the following variations between episodes when generating successful demonstration episodes: object positions drawn randomly in a fixed bounding box with the size of $4 2 . 9 \times 9 . 8 $ (cm), yaws of the object orientation sampled randomly in [0, 2π), target-object size scale with ±10%, base position of the Franka arm offset in a 2 cm-radius disc, exterior-camera pitch varies in ±2 degrees. In the layouts of tasks during evaluation, we only employ randomization in the object positions.

All policies in the latest round are LoRA fine-tunes of the same pretrained π (DROID) checkpoint, which serves as the baseline initialization. Fine-tuning applies LoRA (rank 16, α = 16 on the Gemma-2B backbone; rank 32, $\alpha = 3 2$ on the 300M action expert) while the base transformer weights stay frozen; the SigLIP vision encoder and the small action/state/time projection heads train at full rank. Every policy trains for 20k steps at batch size 32 on a single H200, with each training job takes roughly 6.5 hours. And during evaluation, we run checkpoint 19999 of each policy on the tasks. We train all the policies in a simulation-only manner: all demonstrations are collected in the simulation environment and no real world demonstration is used.

We first present a comprehensive examination of the effect of scaling the number of simulated episodes on the real-world performance. Table 2 shows the success rates of different policies completing the pick-and-place task. We use 200 layouts in evaluation in simulation and use 50 layouts in real-world evaluation. Success rates in both simulation and real-world do not improve indefinitely with the scaling of demonstration dataset. The real-world success rate drops significantly after scaling past 40 demonstration episodes, indicating that excessive demonstrations alone may create severe overfitting. Figure 22 further shows the different failure modes of policies in real-world deployments.

Now we present the effect of utilizing meta-learning strategies on the real-world performance. In order to achieve different meta-learning strategies, we curate three simulated dataset with different purposes. Dataset 1 consists of 2k demonstrations of the eggplant pick-and-place task with all the variations within episodes. Dataset 2 consists of 6k tasks. It is aimed at training a policy that can do three pick-and-place tasks: pick and place an eggplant, a purple cube and a cucumber. Among those 6k demonstration episodes, 2k are dedicated to each object respectively. Dataset 3 consists of 2k demonstration episodes of the eggplant pick-and-place task. But these 2k demonstration episodes do not have full variation and randomization. Among these 2k demonstrations in dataset 3, only the positions of the object are varied. With the three aforementioned curated datasets, we train three polices using dataset 1, dataset 1 + dataset 2, dataset 1 + dataset 2 + dataset 3, respectively. Table 3 shows that performance improves significantly when we deploy the meta learning strategy of training on dataset 1 to 3 in sequence. Figure 23 presents three real-world deployment episodes of policy 3 alongside their simulated counterparts and wrist-camera observations.

<table><tr><td rowspan="2">Episodes</td><td colspan="2">Simulation</td><td colspan="2">Real world</td></tr><tr><td>Successes</td><td>Rate</td><td>Successes</td><td>Rate</td></tr><tr><td>10</td><td>9/200</td><td>4.5%</td><td>1</td><td>1</td></tr><tr><td>20</td><td>13/200</td><td>6.5%</td><td>0/50</td><td>0.0%</td></tr><tr><td>40</td><td>57/200</td><td>28.5%</td><td>24/50</td><td>48.0%</td></tr><tr><td>100</td><td>55/200</td><td>27.5%</td><td>20/50</td><td>40.0%</td></tr><tr><td>200</td><td>90/200</td><td>45.0%</td><td>10/50</td><td>20.0%</td></tr><tr><td>400</td><td>76/200</td><td>38.0%</td><td>12/50</td><td>24.0%</td></tr><tr><td>1k</td><td>99/200</td><td>49.5%</td><td>1</td><td>/</td></tr><tr><td>2k</td><td>114/200</td><td>57.0%</td><td>11/50</td><td>22.0%</td></tr><tr><td>4k</td><td>127/200</td><td>63.5%</td><td>6/50</td><td>12.0%</td></tr><tr><td>10k</td><td>102/200</td><td>51.0%</td><td>1</td><td>1</td></tr><tr><td>20k</td><td>117/200</td><td>58.5%</td><td>1</td><td>1</td></tr></table>

Table 2 Performance of policies trained on different number of demonstration episodes.

![](images/97b22259f8f50da0404d58dddc4fd7d38c124d1eb0ed8e264d0c3a2acabac554.jpg)  
Figure 22 Success and failure modes in real-world deployment of policies trained using different number of demonstrations.

<table><tr><td>training strategies</td><td>performance in sim</td><td>success rate in sim</td><td>real-world performance</td><td>real-world success rate</td></tr><tr><td>policy 1</td><td>114/200</td><td>57.0 %</td><td>11/50</td><td>22.0 %</td></tr><tr><td>policy 2</td><td>140/200</td><td>70.0 %</td><td>31/50</td><td>62.0 %</td></tr><tr><td>policy 3</td><td>170/200</td><td>85.0 %</td><td>44/50</td><td>88.0 %</td></tr></table>

Table 3 Comparison of different training strategies. Policy 1 is trained on 2k eggplant demonstration episodes from the dataset with variations. Policy 2 is trained for 3 tasks, each task with 2k demonstration episodes with variations. Policy 3 is finetuning policy 2 on 2k eggplant demonstration episodes without variation.

## 5 Limitations and Future Work

Conversion cost and verification. Scene conversion requires repeated VLM calls and an external GPU workstation. Processing all objects in a detailed environment remains slow and error-prone. Visual inspection can identify geometric inconsistencies, but it cannot establish the accuracy of hidden geometry or physical parameters. Background completion and generated object variants also rely on predictions of content absent from the capture.

Physical and visual fidelity. Estimated mass, friction, rigidity, and articulation parameters have not yet been calibrated against measured robot interactions. The fracture examples additionally use engineering estimates. Captured Gaussian splats retain the lighting of the original scene, so relighting and compositing mesh objects can introduce inconsistent shadows and reflections. Future work should validate physical parameters against measurements and assess their effect on contact-rich manipulation, including deformation, tearing, and adhesive contact.

Evaluation scope. The reconstruction study uses one kitchen scene and one cluttered bin scene, each converted once per configuration. Broader evaluation with more scenes and repeated conversion runs is needed to characterize generalization and variability. The 4DCodeBench study covers three synthetic scenes, and real-world 4D videos, which lack 3D ground truth, remain to be evaluated. The real-world study currently focuses on pick-and-place. Object variants, placement, and lighting were randomized together, so their individual effects have not been isolated. Paired real and simulated trials are also needed to assess the predictive value of the environments for policy evaluation.

Further automation. Future directions include isolating the effects of scene and physical-parameter randomization, extending the viewer to inspect recorded trajectories, and moving capture processing and policy adaptation onto robot hardware so that data can remain on the device.

![](images/d8265995832a4ff87139542158737f1082078e967043ecf4e82e2c5cf5ef222e.jpg)  
Figure 23 Preliminary real-world deployment on a Franka Emika Panda after simulation-based fine-tuning. Three episodes are shown with corresponding simulated scenes and wrist-camera views, including the eggplant and basket layouts.

## 6 Conclusion

We presented SceneAgent, an agentic pipeline for converting 3D and 4D captures into interactive simulation environ ments. Its central workflow coordinates reconstruction and physical modeling tools with simulator-based inspection and repair, and for videos of dynamic scenes it fits simulated dynamics to the observed motion. Digital sisters, hybrid scenes, and an automated research loop, which turns real-robot failures into simulated demonstrations, extend the converted environments for policy training and evaluation.

On a cluttered bin scene from GraspClutter6D, SceneAgent raises F1 at 1 cm for every harness with scorable paired outputs, and its best F1 score is 60.75%, compared with 6.9% for SimFoundry. On three dynamic scenes from 4DCodeBench, SceneAgent raises the mean dynamic IoU of Gemini 4 Argon from 0.86% to 99.09%. Qualitative examples demonstrate a range of simulated interactions and preliminary real-world deployment after policy fine-tuning. Quantitative policy evaluation and validation of estimated physical parameters remain ongoing. The code, processed scenes, and example converted objects will be released with the paper.

## References

[1] Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, and Ivan Laptev. MessyKitchens: Contact-rich object-level 3D scene reconstruction. arXiv preprint arXiv:2603.16868, 2026. 17

[2] Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim, Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus Neary, Edward Hu, Fabio Ramos, Jonathan Tremblay, Kanav Arora, Kirsty Ellis, Luca Macesanu, Marcel Torne Villasevil, Matthew Leonard, Meedeum Cho, Ozgur Aslan, Shivin Dass, Jie Wang, William Reger, Xingfang Yuan, Xuning Yang, Abhishek Gupta, Dinesh Jayaraman, Glen Berseth, Kostas Daniilidis, Roberto Martin-Martin, Youngwoon Lee, Percy Liang, Chelsea Finn, and Sergey Levine. RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies. arXiv preprint arXiv:2506.18123, 2025. 2

[3] Seunghyeok Back, Joosoon Lee, Kangmin Kim, Heeseon Rho, Geonhyup Lee, Raeyoung Kang, Sangbeom Lee, Sangjun Noh, Youngjin Lee, Taeyeop Lee, and Kyoobin Lee. GraspClutter6D: A large-scale real-world dataset for robust perception and grasping in cluttered scenes. arXiv preprint arXiv:2504.06866, 2025. 2, 17

[4] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra: Generative 3D Scene Reconstruction via Video

Diffusion Model Self-Distillation. arXiv preprint arXiv:2509.19296, 2025. 7

[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π : A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024. Robotics: Science and Systems (RSS). 2

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv preprint arXiv:2307.15818, 2023. Conference on Robot Learning (CoRL). 2

[7] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. In Robotics: Science and Systems (RSS), 2024. 13

[8] CollectDataIO. Tidy Up: U.S. Kitchens Before and After Cleaning. Hugging Face, https://huggingface.co/datasets/ CollectDataIO/tidy-up-kitchens, n.d. 2

[9] Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri, Ainaz Eftekhar, Rohun Tripathi, Max Argus, Jordi Salvador, Haoquan Fang, Matthew Wallingford, Wilbert Pumacay, Yejin Kim, Quinn Pfeifer, Ying-Chun Lee, Piper Wolters, Omar Rayyan, Mingtong Zhang, Jiafei Duan, Karen Farley, Winson Han, Eli Vanderbilt, Dieter Fox, Ali Farhadi, Georgia Chalvatzaki, Dhruv Shah, and Ranjay Krishna. MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation. arXiv preprint arXiv:2603.16861, 2026. 4

[10] Epic Games. Unreal Engine. https://www.unrealengine.com, 2024. 13

[11] Alejandro Escontrela, Justin Kerr, Arthur Allshire, Jonas Frey, Rocky Duan, Carmelo Sferrazza, and Pieter Abbeel. GaussGym: An open-source real-to-sim framework for learning locomotion from pixels. arXiv preprint arXiv:2510.15352, 2025. 2

[12] Zhengqing Gao, Ziwen Li, Xin Wang, Jiaxin Huang, Zhenyang Ren, Mingkai Shao, Hanlue Zhang, Tianyu Huang, Yongkang Cheng, Yandong Guo, Runqi Lin, Yuanyuan Wang, Tongliang Liu, Kun Zhang, and Mingming Gong. Mirage2Matter: A Physically Grounded Gaussian World Model from Video. arXiv preprint arXiv:2602.00096, 2026. 2, 4

[13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2022. 16

[14] Wenhao Hu, Haonan Zhou, Liu Liu, Yun Du, Xinjie Wang, Ziang Li, Zhizhong Su, and Gaoang Wang. ManiSplat: Manipulation Trajectory Synthesis from Monocular Video via Decoupled 3D Gaussian Splatting. arXiv preprint arXiv:2606.10645, 2026. 4

[15] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics, 37(4), 2018. 12

[16] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, et al. HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels. arXiv preprint arXiv:2507.21809, 2025. 7

[17] Andy Huynh, João Malheiro Silva, Holger Caesar, and Tong Duy Son. Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin. arXiv preprint arXiv:2511.20348, 2025. IEEE Intelligent Vehicles Symposium (IV) 2026. 4

[18] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaa Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. OpenCLIP. Zenodo, https://doi.org/10.5281/zenodo.5143773, 2021. 7

[19] Arhan Jain, Mingtong Zhang, Kanav Arora, William Chen, Marcel Torne, Muhammad Zubair Irshad, Sergey Zakharov, Yue Wang, Sergey Levine, Chelsea Finn, Wei-Chiu Ma, Dhruv Shah, Abhishek Gupta, and Karl Pertsch. PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies. arXiv preprint arXiv:2512.16881, 2025. 2, 3

[20] Byeongguk Jeon, Seonghyeon Ye, JaeHyeok Doo, Sungdong Kim, Minjoon Seo, Hyungmok Son, and Kimin Lee. RoboWorld: Fast and Reliable Neural Simulators for Generalist Robot Policy Evaluation. arXiv preprint arXiv:2607.01060, 2026. 5

[21] Yufei Jia, Heng Zhang, Ziheng Zhang, Junzhe Wu, Mingrui Yu, Zifan Wang, Dixuan Jiang, Zheng Li, Chenyu Cao, Zhuoyuan Yu, Xun Yang, Haizhou Ge, Yuchi Zhang, Jiayuan Zhang, Zhenbiao Huang, Tianle Liu, Shenyu Chen, Jiacheng Wang, Bin Xie Xuran Yao, Xiwa Deng, Guangyu Wang, Jinzhi Zhang, Lei Hao, Zhixing Chen, Yuxiang Chen, Anqi Wang, Hongyun Tian, Yiyi Yan, Zhanxiang Cao, Yizhou Jiang, Hanyang Shao, Yue Li, Lu Shi, Bokui Chen, Wei Sui, Hanqing Cui, Yusen Qin, Ruqi Huang, Lei Han, Tiancai Wang, and Guyue Zhou. GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning. arXiv preprint arXiv:2604.25459, 2026. 2

[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4), 2023. 2

[23] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint arXiv:2406.09246, 2024. Conference on Robot Learning (CoRL). 2

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 7

[25] Long Le, Jason Xie, William Liang, Hung-Ju Wang, Yue Yang, Yecheng Jason Ma, Kyle Vedder, Arjun Krishna, Dinesh Jayaraman, and Eric Eaton. Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model. In International Conference on Learning Representations (ICLR), 2025. 9

[26] Inhee Lee, Sangwon Baik, Sungjoo Kim, Hyeonwoo Kim, Hyunsoo Cha, and Hanbyul Joo. SimuScene: Simulation-Read Compositional 3D Scene Reconstruction from a Single Image. arXiv preprint arXiv:2606.03994, 2026. 5

[27] Xinhai Li, Jialin Li, Ziheng Zhang, Rui Zhang, Fan Jia, Tiancai Wang, Haoqiang Fan, Kuo-Kun Tseng, and Ruiping Wang. RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator. arXiv preprint arXiv:2411.11839, 2024. 2

[28] Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv preprint arXiv:2505.19017, 2025. 5

[29] Haozhe Lou, Mingtong Zhang, Haoran Geng, Hanyang Zhou, Sicheng He, Zhiyuan Gao, Siheng Zhao, Jiageng Mao, Pieter Abbeel, Jitendra Malik, Daniel Seita, and Yue Wang. D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping. arXiv preprint arXiv:2603.01151, 2026. International Conference on Learning Representations (ICLR). 4

[30] Rachel Luo, Heng Yang, Michael Watson, Apoorva Sharma, Sushant Veer, Edward Schmerling, and Marco Pavone. Sim2Val: Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation. In Proceedings of the 9th Conference on Robot Learning (CoRL), volume 305 of Proceedings of Machine Learning Research, pages 2294–2310. PMLR, 2025. 2

[31] Rachel Luo, Michael Watson, Apoorva Sharma, Heng Yang, Han Qi, Edward Schmerling, Sushant Veer, Boris Ivanovic, and Marco Pavone. X4Val: Learning Neural Surrogates for Variance-Reduced Policy Evaluation. arXiv preprint arXiv:2606.05159, 2026. 2

[32] Miles Macklin. Warp: A high-performance python framework for gpu simulation and graphics. NVIDIA GPU Technology Conference (GTC), 2022. 11

[33] Pranjal Mishra, René Zurbrügg, Max Wilder-Smith, Marco Hutter, Marc Pollefeys, Zuria Bauer, and Hermann Blum. ArtiTwin Splat: Interactable Digital Twin Reconstruction via Gaussian Splatting from RGB-D videos. arXiv preprint arXiv:2606.24628, 2026. IEEE International Conference on Robotics and Automation (ICRA) Workshop. 4

[34] Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Muñoz, Xinjie Yao, René Zurbrügg, Nikita Rudin, et al. Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning. arXiv preprint arXiv:2511.04831, 2025. 13

[35] Ben Moran, Mauro Comi, Arunkumar Byravan, Steven Bohez, Tom Erez, Zhibin Li, and Leonard Hasenclever. Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data. arXiv preprint arXiv:2506.04120, 2025. 4

[36] NVIDIA. NVIDIA Omniverse. https://developer.nvidia.com/omniverse, 2024. 13

[37] NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv preprint arXiv:2503.14734, 2025. 2

[38] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An Open-Source Generalist Robot Policy. arXiv preprint arXiv:2405.12213, 2024. Robotics: Science and Systems (RSS). 2

[39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning Robust Visual Features without Supervision. arXiv preprint arXiv:2304.07193, 2023. 7

[40] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π : a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025. 2, 19

[41] Pixar Animation Studios. Universal Scene Description (OpenUSD). https://openusd.org, 2016. 13

[42] Polycam. Polycam. https://poly.cam/, n.d. Accessed 11 October 2026. 10

[43] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D Language Gaussian Splatting In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 7

[44] Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. WorldGym: World Model as An Environment for Policy Evaluation. arXiv preprint arXiv:2506.00613, 2025. 5

[45] Mohammad Nomaan Qureshi, Sparsh Garg, Francisco Yandun, David Held, George Kantor, and Abhisesh Silwal. SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting. arXiv preprint arXiv:2409.10161, 2024. IEEE International Conference on Robotics and Automation (ICRA). 2

[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natura Language Supervision. In International Conference on Machine Learning (ICML), 2021. 7

[47] Nadun Ranawaka, Josiah Wong, Wei-Lin Pai, Wei-Teng Chu, Tianyuan Dai, Masoud Moghani, Hang Yin, Yunfan Jiang, Wesley Durbano, Brandon Huynh, Yu Fang, Danfei Xu, Ruohan Zhang, Li Fei-Fei, Linxi Fan, Bowen Wen, Ajay Mandlekar, and Yuke Zhu. SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation. arXiv preprint arXiv:2606.28276, 2026. 1, 2, 3, 18

[48] SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, and Jitendra Malik. SAM 3D: 3Dfy Anything in Images. arXiv preprint arXiv:2511.16624, 2025. 8

[49] Johannes L. Schönberger and Jan-Michael Frahm. Structure-from-Motion Revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. COLMAP. 7

[50] Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees G. M. Snoek, Josef Sivic, and Vladimir Petrik. REALM: A Real-to-Sim Validated Benchmark for Generalization in Robotic Manipulation. arXiv preprint arXiv:2512.19562, 2025. IEEE Robotics and Automation Letters. 4

[51] Ruihong Shen, Žiga Kovaciˇ c, Peter Kulits, Xingrui Wang, Zizhang Li, Joshua B. Tenenbaum, Alan Yuille, Jieneng Chen, andˇ Jiajun Wu. 4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes. arXiv preprint arXiv:2610.03715, 2026. 1, 2, 4, 18

[52] D. Sulsky, Z. Chen, and H. L. Schreyer. A particle method for history-dependent materials. Computer Methods in Applied Mechanics and Engineering, 118(1–2):179–196, 1994. 12

[53] Ziyang Sun, Lingfan Bao, Tianhu Peng, Jingcheng Sun, and Chengxu Zhou. A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting. arXiv preprint arXiv:2601.03200, 2026. Journal of Robot Learning. 4

[54] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust Large Mask Inpainting with Fourier Convolutions. In IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2022. LaMa. 8

[55] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa. Nerfstudio: A Modular Framework for Neura Radiance Field Development. In ACM SIGGRAPH 2023 Conference Proceedings, 2023. 7

[56] Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012. 13

[57] Boyuan Wang, Xiaofeng Wang, Yongkang Li, Zheng Zhu, Yifan Chang, Angen Ye, Guosheng Zhao, Chaojun Ni, Guan Huang, Yijie Ren, Yueqi Duan, and Xingang Wang. ReconPhys: Reconstruct Appearance and Physical Attributes from Single Video. arXiv preprint arXiv:2604.07882, 2026. 4

[58] Maggie Wang, Stephen Tian, Aiden Swann, Ola Shorinwa, Jiajun Wu, and Mac Schwager. Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation. arXiv preprint arXiv:2510.11689, 2025. IEEE International Conference on Robotics and Automation (ICRA). 4

[59] Yiming Wang, Ruogu Zhang, Minyang Li, Hao Shi, Junbo Wang, Deyi Li, Jieji Ren, Wenhai Liu, Weiming Wang, and Hao-Shu Fang. ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection. arXivpreprint arXiv:2601.18629, 2026. 2, 4

[60] Yuehao Wang, Chaoyi Wang, Bingchen Gong, and Tianfan Xue. Bilateral Guided Radiance Field Processing. ACM Transactions on Graphics (SIGGRAPH), 2024. 7

[61] World Labs. Marble: A Multimodal World Model. World Labs blog, https://www.worldlabs.ai/blog/ marble-world-model, 2025. Released 12 November 2025. 7

[62] Qinwen Xu, Jiaming Liu, Rui Zhou, Shaojun Shi, Nuowei Han, Zhuoyang Liu, Chenyang Gu, Shuo Gu, Yang Yue, Gao Huang, Wenzhao Zheng, Sirui Han, Peng Jia, and Shanghang Zhang. TwinRL: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation. arXiv preprint arXiv:2602.09023, 2026. 4

[63] Sizhe Yang, Wenye Yu, Jia Zeng, Jun Lv, Kerui Ren, Cewu Lu, Dahua Lin, and Jiangmiao Pang. Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation. arXiv preprint arXiv:2504.13175, 2025. Robotics: Science and Systems (RSS). 2, 4

[64] Yi Yang, Xiang Fei, Lehong Wang, Chenhao Li, Zilin Dai, Henry Kou, Lu Li, and Howie Choset. DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects. arXiv preprint arXiv:2606.22116, 2026. IEEE/RSJ Internationa Conference on Intelligent Robots and Systems (IROS). 4

[65] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An Open-Source Library for Gaussian Splatting. arXiv preprint arXiv:2409.06765, 2024. 7

[66] Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu, Wenjun Sun, Linjie Hou, Zhijun Li, Junhui Wu, Zhaobo Liu, Zhen Xiao, Sheng Zhang, Lei Bao, Rui Feng, Zhenquan Pang, Jiayu Li, Qian Wang, and Maoqing Yao. Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot. arXiv preprint arXiv:2601.02078, 2026. 4

[67] Justin Yu, Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus, Richard Cheng, Muhammad Zubair Irshad, and Ken Goldberg. Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware. arXiv preprint arXiv:2505.09601, 2025. 2, 4

[68] Jiawei Zhang, Yiming Yan, Chao Liang, Nuo Xu, Seson Sun, Qichen Zhang, Yuhao Xu, Yantai Yang, Yingqiao Wang, Qin Jin, and Zhipeng Zhang. GASE: Gaussian Splatting-Based Automated System for Reconstructing Embodied-Simulation Environments. arXiv preprint arXiv:2606.17520, 2026. 2, 4

[69] Kaifeng Zhang, Shuo Sha, Hanxiao Jiang, Matthew Loper, Hyunjong Song, Guangyan Cai, Zhuo Xu, Xiaochen Hu, Changxi Zheng, and Yunzhu Li. Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions. arXiv preprint arXiv:2511.04665, 2025. 2

[70] Yiheng Zhu, Kangle Deng, Jean-Philippe Fauconnier, Inaki Navarro, Daiqing Li, Ava Pun, Yinan Zhang, Peiye Zhuang, Xiaoxia Sun, Maneesh Agrawala, Kiran Bhat, and Tinghui Zhou. CubePart: An Open-Vocabulary Part-Controllable 3D Generator. arXiv preprint arXiv:2605.28763, 2026. ACM SIGGRAPH 2026. 9

[71] Yixin Zhu, Zixiong Wang, Jian Yang, Jin Xie, Jingyi Yu, Jiayuan Gu, and Beibei Wang. Toward Visually Realistic Simulation: A Benchmark for Evaluating Robot Manipulation in Simulation. arXiv preprint arXiv:2605.06311, 2026. 4

[72] Yongning Zhu and Robert Bridson. Animating sand as a fluid. ACM Transactions on Graphics, 24(3):965–972, 2005. 12

[73] Zhe Zhu, Le Wan, Rui Xu, Yiheng Zhang, Honghua Chen, Zhiyang Dou, Cheng Lin, Yuan Liu, and Mingqiang Wei. PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data. arXiv preprint arXiv:2509.21965, 2025. 9

Featured Art and 3D Captures
<table><tr><td>Asset</td><td>Creator</td></tr><tr><td>Sketchfab</td><td></td></tr><tr><td>Rudbeckia flower</td><td>3dhdscan</td></tr><tr><td>Chinese statue</td><td>netgis</td></tr><tr><td>Asus Z170-P motherboard</td><td>Lassi Kaukonen</td></tr><tr><td>RX 480 GPU</td><td>Igor.Jop</td></tr><tr><td>Polycam</td><td></td></tr><tr><td>Vintage Sony radio CFS-43</td><td>JordanFry</td></tr><tr><td>Totoro</td><td>TechnologicalFrostfang2470</td></tr><tr><td>Stegosaurus composite skeleton, American Museum of Natural History</td><td>MTP</td></tr><tr><td>Bencini Comet Universal 444 camera (1977)</td><td>redline3678</td></tr><tr><td>Cessna C172R cockpit</td><td>wlanfr3ak</td></tr><tr><td>Motorcycle</td><td>ZanyFalcon4404</td></tr><tr><td>Towel rack</td><td>Pandora</td></tr><tr><td>Red towel</td><td>AutomatedStarfish942</td></tr><tr><td>Dept Artisté anorak</td><td>Artist</td></tr><tr><td>Firefighter hose</td><td>drewjosh</td></tr><tr><td>Electric extension lead</td><td>sisamm</td></tr><tr><td>Liquid Death hat</td><td>stevieburd</td></tr><tr><td>Black-figure Dinos and stand with the wedding of Peleus and Thetis, British Museum</td><td>PortlandMuseum</td></tr><tr><td>Small Earthenware Pot: EOA:2022.111.18</td><td></td></tr></table>