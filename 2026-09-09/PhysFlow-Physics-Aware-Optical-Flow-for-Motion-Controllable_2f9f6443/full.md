# PhysFlow: Physics-Aware Optical Flow for Motion Controllable Video Generation

Cong Wang , Hanxin Zhu , Yonglin Tian , Jiayi Luo , Ruiqi Song , Boyi Sun , Long Chen , Senior Member, IEEE, and Zhibo Chen , Senior Member, IEEE

Abstract—Video generation models have recently attracted substantial attention for their ability to generate visually compelling videos, yet ensuring physically consistent and plausible dynamics still remains a fundamental challenge, driving a growing line of research on physical realism in video generation. To address this challenge, motivated by the fact that physical regularities are primarily encoded in motion patterns, we propose PhysFlow, a novel two-stage framework for improving the physical plausibility of generated videos by decomposing video generation into motion-aware optical flow generation followed by motion-conditioned appearance synthesis. Specifically, PhysFlow consists of a physics-aware optical-flow video generator called PA-Flow and a flow-guided video generator called FlowRender. During the first stage, PA-Flow employs a physics-aware attention module to model how motion attributes and material properties influence global motion and local deformation, respectively, and generates an optical flow video as an explicit representation of motion. In the second stage, FlowRender leverages the decoupled motion representation as guidance to synthesize realistic textures and appearances, ultimately producing the final physically plausible video. To further support model training with explicit physical supervision, we construct PhysVideo, a physics-based video dataset generated with a physics engine and 3D-GS rendering, containing 10K foreground objects and 50K realistic video sequences with annotations of motion and material properties. Extensive experiments demonstrate that our proposed PhysFlow generates videos with superior physical plausibility while maintaining high visual fidelity compared with existing methods. Project page: https://physwm.github.io/PhysFlow/.

## I. INTRODUCTION

IDEO generation models [1]–[5] have achieved remarkable progress in synthesizing high-quality and temporally   
coherent videos [6]. Despite their impressive visual quality,   
these models often lack a robust understanding of the under  
lying physical principles [7], [8]. Consequently, the generated   
videos frequently exhibit physically implausible dynamics,

![](images/2bded58f550dc94c4d63957d65eb682eec33397457ac62d2c9e2d397715e6d04.jpg)  
Fig. 1. Comparison of our method with prior video generation paradigms. (a) Data-driven generative models. (b) Physics-engine-based methods. (c) Our physics-aware two-stage framework utilizing optical flow.

such as inconsistent object deformation, unrealistic collisions, and incorrect interactions with the surrounding environment, which substantially limits their applicability to real-world scenarios that require accurate physical reasoning like embodied intelligence and autonomous driving [9]–[11].

To equip video generation models with a stronger understanding of physical dynamics, recent efforts can be broadly categorized into two paradigms: Physics-engine-based approaches introduce explicit simulation into the generation pipeline, as illustrated in Figure 1. For example, PhysGen [12] uses rigid-body simulation to guide image-to-video generation, while PhysGen3D [13], WonderPlay [14], OmniPhysGS [15], and related physics-engine-based methods [16]–[18] further combine 3D representations with physics solvers to support more complex dynamic scenes. These methods improve physical plausibility by relying on explicit geometry, material parameters, and solver-based state updates. However, their performance is often limited by the quality of 3D reconstruction, the accuracy of estimated physical parameters, and the fidelity of the simulator. As a result, errors in geometry, contact estimation, or material assignment may propagate to the final video, especially in scenes with non-rigid deformation, multi object interaction, or complex foreground-background contact.

A complementary paradigm seeks to endow pretrained video generators with physical controllability through additional conditioning signals. Force Prompting [19] injects force signals into video generation, PhysCtrl [20] represents dynamics with physics-conditioned 3D point trajectories, and recent physics-aware text-to-video systems [21], [22] use language or world-simulation priors to improve physical realism. These approaches reduce direct dependence on full simulation at inference time, but they still face several limitations. Textlevel controls are often too coarse to describe dense object deformation, while trajectory-based controls usually depend on foreground 3D structures and sparse motion points. Consequently, existing physics-aware methods still struggle to jointly achieve dense motion controllability, realistic appearance synthesis, and robust handling of object-background or multiobject interactions from a single image.

Our key observation is that physical regularities are primarily manifested in motion patterns rather than textures [9]. Motivated by this insight, we advocate learning physics-aware video generation by disentangling motion and appearance, instead of relying on physics engines that require manual settings. We represent motion using optical flow, which compactly captures appearance-agnostic, pixel-level displacement [23]. These limitations motivate us to use physics-aware optical flow as a dense 2D motion representation. Compared with simulator states or sparse 3D point trajectories, optical flow directly describes pixel-level motion in the image plane, providing a practical representation for local deformation, boundary-aware motion, and coupled object interactions while remaining compatible with modern video generators.

Building on this idea, we propose PhysFlow, a two-stage physics-aware video generation framework that decomposes video synthesis into (1) physics-aware optical flow generation and (2) flow-guided plausible video generation, as shown in Figure 1 (c). The first stage is handled by the PA-Flow module, a physics-aware optical-flow video generator that produces flow sequences conditioned on motion attributes (e.g., velocity and acceleration) and material properties (e.g., density and Young’s modulus). To learn how global motion and local deformations depend on these factors, PA-Flow employs a physics-aware attention module: motion attributes modulate global motion patterns, while material properties refine local deformation, enabling the model to learn complex non-rigid behaviors. The second stage is realized by the FlowRender module, a flow-guided video generator that takes the opticalflow video as motion guidance and synthesizes the corresponding RGB sequence, enriching it with realistic textures. In this work, we focus on scenarios involving rigid bodies, elastic deformation, and selected flexible bodies, in which the underlying physics is primarily manifested through motion. To support the physics-aware learning process, we further construct PhysVideo, a physics-based video dataset built using a physics engine and 3D-GS rendering. Specifically, PhysVideo comprises 10K foreground objects spanning rigid, elastic, and flexible categories, together with 50K realistic video sequences, each annotated with motion attributes and material properties. Extensive experimental results demonstrate that our proposed PhysFlow generates videos with more physically plausible dynamics, outperforming existing methods.

In summary, our core contributions are as follows:

• We propose a physics-aware dynamic video generation framework named PhysFlow that adopts a two-stage architecture to disentangle motion and texture synthesis.

• We introduce PA-Flow, a physics-aware image-to-flow generation model that incorporates a physics attention mechanism to model how motion attributes and material properties affect global motion and local deformation.

• We develop FlowRender, a flow-guided video generation model that synthesizes dynamic RGB videos under the guidance of optical flow sequences.

• We construct PhysVideo, a physics-based video dataset that ensures physical accuracy and visual realism through physics simulation and 3D-GS rendering, containing over 10K objects and 50K video samples, providing comprehensive training data for physics-aware video generation.

## II. RELATED WORKS

## A. Controllable Video Generation

Video generation models trained on large-scale text–video paired datasets have demonstrated their remarkable capabilities in synthesizing high-quality videos [1], [3], [4], [24]. Previous studies have shown that pretrained models can be addi tionally guided by various control signals, including camera motion [25], [26], point trajectories [27]–[29], anchor-frame videos [30], [31], and unified or scenario-specific control adapters [32]–[34], enabling more controllable video generation. These methods demonstrate that introducing structured control signals can substantially improve user control over generated content, but the controls are usually defined from visual or geometric cues rather than physical causes. Meanwhile, other research efforts have explored leveraging different modalities to guide motion-aware video synthesis. Some approaches [23], [35] employ optical flow videos as guidance, using flow maps to characterize motion dynamics and subsequently synthesize RGB videos based on them. Other methods [36], [37] use depth videos, transition frames, or interframe motion reuse [38], [39] as conditioning or temporal cues to guide the generation of coherent video sequences. In addition, several text-driven approaches [21] attempt to infer more fine-grained descriptions of motion dynamics from textual instructions, and recent evaluation efforts further emphasize motion-centered video quality assessment [40], thereby encouraging more precise control over motion behaviors during video generation. However, although these approaches yield gains in temporal coherence and controllability, they typically treat motion as a visual trajectory to be followed rather than as the outcome of material properties and external motion conditions. Moreover, they generally lack explicit modeling of physical laws and often produce results that violate basic principles of physical plausibility. To address these aforementioned limitations, our method generates physics-aware optical flow video as a motion representation and subsequently renders it into plausible video outputs.

## B. Physics-Engine-Based Video Generation

Physics engines typically use the 3D representation of foreground objects as input to simulate and generate the 3D state at each time step, based on defined material and motion properties [13], [16]. Leveraging the renderable 3D-GS representation [41], some methods [16] propose using the Material Point Method (MPM) solver to update the 3D Gaussian representation at each time step after force-induced motion.

![](images/42eb7eafb683f4ed49a8e3fd01cbbbcfab5fe12325b2bea0ffc77677a140284b.jpg)  
Fig. 2. Pipeline of our proposed PhysFlow. PhysFlow generates physically plausible video via a two-stage pipeline: (1) generating a physics-aware optical flow video (Details are in Section III-A), (2) generating a plausible video with the guidance of the optical flow video (Details are in Section III-B).

However, due to the limitations of MPM, it is only suitable for simulating elastic objects. To address the shortcomings of MPM solvers, some approaches [42], [43] incorporate Position-Based Dynamics (PBD) solvers to simulate fluid motion. The use of physics engines requires additional definition of physical properties, and some methods [44], [45] propose using MLLMs to estimate properties such as Young’s modulus, Poisson’s ratio, and density. Furthermore, other methods [15], [46]–[48] suggest utilizing the prior knowledge of pre-trained video generation models to iteratively optimize the material properties of objects through SDS loss [49] after rendering the video. Recent advances in text-guided 3D-aware generation and Gaussian-based image-to-3D reconstruction [50], [51] have further improved the accessibility of 3D priors for such pipelines. However, these representations must still be converted into physically valid simulation states before reliable dynamics can be produced. While physics-enginebased methods can ensure physical plausibility, they require manual specification of simulation conditions and depend on high-quality 3D representations, which limits their ability to generate high-quality videos from a single image.

## C. Physics Enhanced Video Generation

Recent studies have increasingly focused on enhancing the physical plausibility of generated videos [52]. WISA [22] introduces a dataset emphasizing physical phenomena, featuring complex interactions between fluids and solids. To assess physical realism, several works [7], [8] propose VLMbased benchmarks and metrics for evaluating the physical quality of generated videos. To improve physical consistency, Force Prompting [19] explicitly incorporates external forces and learns their influence on object motion. WonderPlay [14] generates coarse videos with a physics engine and refines them via video optimization, but it still relies heavily on the simulator. In contrast, PhysCtrl [20] reduces simulator dependence by learning a physically controllable point-cloud trajectory generator and using sparse motion trajectories to guide video synthesis. However, its performance is sensitive to the quality of pre-generated 3D point clouds, and it struggles with complex multi-object and object–background interactions. Furthermore, NewtonGen [53] models physical dynamics by directly learning the governing partial differential equations, providing a principled formulation to guide video generation. While these methods simplify the cumbersome steps of physics engine simulations, they still rely on highquality 3D object priors or overfitting to single scenes, making it challenging to achieve physics-aware video generation that accounts for both motion and material properties.

## III. METHODOLOGY

Task Description. Given a single image, a textual prompt, and physical attributes as input, our goal is to generate videos with physically plausible object-centric dynamics that reflect the specified motion and material conditions. We focus on rigidbody motion, elastic deformation, and selected flexible-body behaviors. Generating such videos requires capturing physicsconditioned object dynamics and synthesizing the corresponding motion-consistent changes in scene appearance.

Method Overview. To achieve the aforementioned goal, we propose a two-stage video generation framework that represents motion states through optical flow video, as shown in Figure 2. We first train a physics-aware optical flow generator PA-Flow, which uses motion and material attributes as physical conditions to generate plausible global motion and local deformation (Section III-A). Subsequently, we train a flow-guided texture rendering model FlowRender to synthesize realistic appearances guided by the generated optical flow (Section III-B). To support the training of PA-Flow, we construct a physically realistic video dataset PhysVideo using a physics engine (Section III-C). Finally, we introduce the training strategy of PA-Flow and FlowRender (Section III-D).

![](images/16cc64487bf1794045624f6c6c1a4eff30e7a81ca0d1121a6ee69988e63da385.jpg)  
Fig. 3. The physics-aware attention module. It consists of a globalmotion attention module (Motion-Attention) and a local-deformation attention (Deform-Attention) within each block.

## A. Stage I: Physics-Aware Optical Flow Generation

Given the input containing a static RGB image I, a textual prompt P, and a set of physical attributes comprising material parameters M and motion parameters $( a , v )$ , our goal is to synthesize an optical-flow video sequence that captures physics-grounded motion dynamics.

To enable more realistic motion generation, we introduce additional conditional priors to guide the video synthesis process. Specifically, we employ Grounded-SAM [54] to automatically segment foreground objects that correspond to the instructed motion, producing spatial masks that are temporally aligned to form a static mask video $\mathbf { V } _ { m } .$ . To supplement object appearances that are not visible in the input viewpoint, we incorporate a multi-view object prior $\mathbf { V } _ { m v }$ obtained from Trellis [55] as additional guidance. As illustrated in Figure 2, the priors ${ \mathbf { V } } _ { m }$ and $\mathbf { V } _ { m v }$ are jointly fed into PA-Flow as control signals to ensure semantic and appearance consistency. The input image I is encoded into $\mathbf { H } _ { i }$ , while the conditional videos ${ \mathbf { V } } _ { m }$ and $\mathbf { V } _ { m v }$ are encoded into $\mathbf { H } _ { m }$ and $\mathbf { H } _ { m v } .$ , respectively. After passing $\mathbf { V } _ { m }$ and $\mathbf { V } _ { m v }$ through the video projector, their processed latents are fused with $\mathbf { H } _ { i }$ to construct the final conditioned latent $\mathbf { H } _ { c } ,$

The global motion of an object is primarily governed by motion-related attributes such as initial velocity and acceleration, whereas local deformation is further influenced by material properties, including Young’s modulus and density. To better enable physical attributes to guide the generation of optical flow videos, we design our model based on CogVideoX-I2V-5B [4] by introducing an additional physics-aware attention module (PAAM) into its original transformer blocks. As illustrated in Figure 3, PAAM comprises two key components: a global motion attention module driven by motion attributes, and a local deformation attention module driven by material attributes. The two modules are integrated in a sequential architecture, where the model first focuses on capturing global motion and subsequently learns local deformation patterns.

Global-Motion Attention. The global motion attributes are defined by two components: the acceleration vector ${ \bf a } \ =$ $[ a _ { x } , a _ { y } , a _ { z } ]$ and the velocity vector $\mathbf { v } = [ v _ { x } , v _ { y } , v _ { z } ]$ . During encoding, these vectors are extracted from the input motion dictionary and jointly embedded to form the global motion condition $\mathbf { C } _ { g } .$ . The input latent $\mathbf { H } _ { c }$ is fused with $\mathbf { C } _ { g }$ through the global motion attention module to yield the motion-aware latent $\mathbf { H } _ { g a }$ and an updated motion condition $\mathbf { C } _ { g a } .$ . The process adapts residual connection and computes $\mathbf { H } _ { g a }$ as:

$$
{ \bf H } _ { g a } = { \bf H } _ { c } + \alpha _ { g a 1 } \cdot \mathrm { A t t n } _ { g } ( \mathrm { L N } ( [ { \bf H } _ { c } ; { \bf C } _ { g } ] ) ) ,\tag{1}
$$

where $\alpha _ { g a 1 }$ is a learnable scaling parameter, and LN denotes an adaptive layer normalization operation. The updated motion condition $\mathbf { C } _ { g a }$ is computed in a similar manner. Finally, a feedforward layer is applied, following the same residual design, to further refine the latent representation. The updated latent $\mathbf { H } _ { g f }$ is computed as:

$$
\mathbf { H } _ { g f } = \mathbf { H } _ { g a } + \alpha _ { g f 1 } \cdot \mathrm { F F N } _ { g } ( \mathrm { L N } ( [ \mathbf { H } _ { g a } ; \mathbf { C } _ { g } ] ) )\tag{2}
$$

where $\alpha _ { g f 1 }$ denotes a learnable scaling coefficient that controls the contribution of the feed-forward operation.

Local-Deformation Attention. The material attributes are expressed as a vector $\mathbf { m } = [ \rho , E , \nu ]$ , consisting of density $\rho ,$ Young’s modulus $E _ { \mathrm { { : } } }$ and Poisson’s ratio $\nu .$ This vector is extracted from the input material dictionary and encoded into a material condition embedding $\mathbf { C } _ { l }$ . To model materialdependent deformation, $\mathbf { C } _ { l }$ interacts with the motion-aware latent $\mathbf { H } _ { g f }$ through the local deformation attention module, yielding the intermediate feature ${ \bf H } _ { l a }$

$$
\mathbf { H } _ { l a } = \mathbf { H } _ { g f } + \alpha _ { l a 1 } \cdot \mathrm { A t t n } _ { l } ( \mathrm { L N } ( [ \mathbf { H } _ { g f } ; \mathbf { C } _ { l } ] ) ) ,\tag{3}
$$

where $\alpha _ { l a 1 }$ is a learnable coefficient controlling the strength of local deformation attention. Subsequently, a feed-forward refinement is performed to further enhance material sensitivity, formulated as:

$$
\mathbf { H } _ { l f } = \mathbf { H } _ { l a } + \alpha _ { l f 1 } \cdot \mathrm { F F N } _ { l } ( \mathrm { L N } ( [ \mathbf { H } _ { l a } ; \mathbf { C } _ { l } ] ) ) ,\tag{4}
$$

where $\alpha _ { l f 1 }$ denotes the learnable scaling factor in the feedforward layer.

From the above process, a physics-aware latent is obtained and then processed by 3D full attention [4] and a feed-forward network to produce the predicted noise:

$$
\tilde { \mathbf { F } } = \mathrm { F F N } ( \operatorname { A t t n } ( [ \mathbf { H } _ { l f } ; \mathbf { C } _ { P } ] ) ) ,\tag{5}
$$

where $\mathbf { C } _ { P }$ denotes the encoded embedding of the textual prompt P. The final predicted optical flow latent is calculated with the denoising processor $D _ { \theta } ( \cdot )$

$$
\begin{array} { r } { \hat { \mathbf { F } } = D _ { \pmb { \theta } } ( \tilde { \mathbf { F } } _ { t } , t ) , } \end{array}\tag{6}
$$

where $\tilde { \mathbf { F } } _ { t }$ is the sampled noise at the timestamp t.

## B. Stage II: Flow-Guided Plausible Video Generation

In this stage, our objective is to generate a visually plausible video that adheres to the motion guidance provided by ${ \hat { F } } .$ Since object motion inevitably affects the surrounding environment, this process involves more than merely moving the foreground while keeping the background static. However, explicitly modeling the complex interactions between the foreground and background remains challenging. To address this, we leverage the extensive prior knowledge embedded in large-scale pre-trained image-to-video generation models. In particular, we introduce FlowRender, which is built upon CogVideoX-I2V-5B [4] and trained via Supervised Fine-Tuning (SFT) under the conditioning of optical flow videos. The training process leverages real-world datasets such as OpenVid [6]. To enhance the ability to generate complex physics-related phenomena, we further utilize the physicsfocused dataset WISA [22] to fine-tune FlowRender. The effects of fine-tuning on WISA [22] (Phys-FT) are illustrated in the ablation study (Table IV). FlowRender employs F<sup>ˆ</sup> as the motion control condition:

$$
\tilde { \mathbf { R } } = \mathrm { F F N } ( \mathrm { A t t n } ( [ \mathbf { H } _ { i } ; \hat { \mathbf { F } } ; \mathbf { C } _ { P } ] ) ) ,\tag{7}
$$

where $\tilde { \mathbf { R } }$ denotes the noise predicted by the model. The final output video latent R<sup>ˆ</sup> is calculated after denoising:

$$
\begin{array} { r } { \hat { \bf R } = D _ { \theta } ( \tilde { \bf R } _ { t } , t ) , } \end{array}\tag{8}
$$

where $\tilde { \mathbf { R } } _ { t }$ is the sampled noise at timestamp t. The output of FlowRender represents the RGB video frames synthesized under the motion constraints imposed by F<sup>ˆ</sup> . During training, the model learns to align the motion implied by the optical-flow condition with the visual content generation process, enforcing temporal coherence and ensuring that object dynamics remain physically consistent throughout the sequence.

## C. PhysVideo Dataset

To facilitate the training of our physics-aware optical flow video generation model, we construct PhysVideo, a dataset comprising 50K physics-grounded video samples. Existing physics-engine–based video datasets primarily use simplistic or synthetic backgrounds. In contrast, PhysVideo incorporates realistic background scenes while maintaining physically grounded simulations. Specifically, we generate 10K foreground objects represented by 3D-GS using Trellis [55], and further reconstruct 10 diverse backgrounds with 3D-GS representations that support explorable viewpoints. Following the design of OmniPhysGS [15], we employ TaiChi [56] as the physics engine and adopt the Material Point Method (MPM) as the physics solver to compute object state updates given physical properties. Leveraging the advantages of 3D-GS, we directly compose the foreground object $G _ { f }$ and background $G _ { b } ,$ , rendering full frames from specified viewpoints. For each sample, the textual prompt, motion attributes, and material parameters used to control the simulation are automatically recorded. During post-processing, we employ MemFlow [57] to estimate optical flow videos from the simulated sequences and use Grounded-SAM [54] to obtain semantic masks. Finally, Trellis [55] is again utilized to generate multi-view videos of the segmented foreground objects. Through the above pipeline, we construct a dataset comprising 10K objects and 50K videos. By incorporating realistic 3D backgrounds, our dataset produces highly realistic rendered videos, as shown in Figure 4. More dataset details can be found in Appendix Section A in the Supplementary Material.

![](images/4ec13a6f007851ae40db5106636165b64495ac1177d9c35e1867b242d662ee7f.jpg)  
Fig. 4. Dataset demonstration of PhysVideo. Our dataset contains various objects and motion patterns.

## D. Training Pipeline

PA-Flow Training. We optimize the model using the base SFT loss and incorporate physics-informed regularization to enforce spatial and temporal motion consistency. Specifically, the base loss is expressed as follows:

$$
\mathcal { L } _ { \mathrm { b a s e } } = \mathbb { E } _ { \tilde { \mathbf { F } } , t } \Big [ w _ { 1 } ( t ) \cdot \big \| D _ { \theta } ( \tilde { \mathbf { F } } _ { t } , t ) - \bar { \mathbf { F } } \big \| _ { 2 } ^ { 2 } \Big ] ,\tag{9}
$$

where $\bar { \mathbf { F } }$ denotes the latent representation of the ground-truth optical-flow video, and $w _ { 1 } ( t )$ weights the denoising error at diffusion timestep t according to the corresponding noise level.

To optimize the global motion, we introduce two complementary temporal regularization terms that constrain motion evolution in both the velocity and acceleration domains, namely the velocity-smoothness loss $\mathcal { L } _ { \mathrm { v e l } }$ and the accelerationconsistency loss $\mathcal { L } _ { \mathrm { a c c } }$ . The velocity-smoothness loss penalizes abrupt temporal variations in the optical-flow sequence and is formulated as:

TABLE I  
QUANTITATIVE COMPARISONS. BOLD: BEST. UNDERLINE: SECOND BEST. PE REFERS TO THE PHYSICS ENGINE.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PE</td><td colspan="3">VideoPhy-2</td><td colspan="4">VBench</td><td colspan="3">WorldScore</td></tr><tr><td>SA↑</td><td>PC↑</td><td>Rule↑</td><td>Motion↑</td><td>Subject↑</td><td>Flicker↑</td><td>Image↑</td><td>Photo↑</td><td>Motion↑</td><td>3D↑</td></tr><tr><td>OmniPhysGS [15]</td><td></td><td>2.41</td><td>3.09</td><td>0.204</td><td>0.995</td><td>0.919</td><td>0.993</td><td>0.410</td><td>12.41</td><td>89.47</td><td>39.85</td></tr><tr><td>PhysGen [12]</td><td></td><td>2.58</td><td>3.25</td><td>0.265</td><td>0.995</td><td>0.958</td><td>0.990</td><td>0.629</td><td>89.60</td><td>80.65</td><td>91.52</td></tr><tr><td>PhysGen3D [13]</td><td>ノV√</td><td>2.61</td><td>3.27</td><td>0.194</td><td>0.996</td><td>0.928</td><td>0.998</td><td>0.593</td><td>92.60</td><td>90.07</td><td>92.12</td></tr><tr><td>CogVideoX-I2V-5B [4]</td><td>x</td><td>2.66</td><td>3.14</td><td>0.221</td><td>0.992</td><td>0.917</td><td>0.988</td><td>0.628</td><td>73.22</td><td>74.48</td><td>80.49</td></tr><tr><td>Wan2.2-TI2V-5B [5]</td><td>x</td><td>2.71</td><td>3.18</td><td>0.188</td><td>0.994</td><td>0.916</td><td>0.989</td><td>0.635</td><td>59.86</td><td>60.25</td><td>74.06</td></tr><tr><td>Force Prompting [19]</td><td>x</td><td>2.69</td><td>3.21</td><td>0.219</td><td>0.994</td><td>0.951</td><td>0.992</td><td>0.623</td><td>65.83</td><td>52.64</td><td>76.86</td></tr><tr><td>PhysCtrl [20]</td><td>x</td><td>2.68</td><td>3.22</td><td>0.250</td><td>0.996</td><td>0.955</td><td>0.994</td><td>0.657</td><td>90.80</td><td>82.45</td><td>90.27</td></tr><tr><td>PhysFlow(Ours)</td><td>x</td><td>2.81</td><td>3.41</td><td>0.281</td><td>0.997</td><td>0.960</td><td>0.994</td><td>0.640</td><td>95.59</td><td>92.82</td><td>90.49</td></tr></table>

$$
\mathcal { L } _ { \mathrm { v e l } } = \frac { 1 } { T - 2 } \sum _ { t = 2 } ^ { T - 1 } \lVert \hat { \mathbf { F } } _ { t + 1 } - 2 \hat { \mathbf { F } } _ { t } + \hat { \mathbf { F } } _ { t - 1 } \rVert _ { 2 } ^ { 2 } ,\tag{10}
$$

while the acceleration-consistency loss enforces alignment between the predicted and ground-truth temporal derivatives of optical flow:

$$
\mathcal { L } _ { \mathrm { a c c } } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \lVert ( \hat { \mathbf { F } } _ { t + 1 } - \hat { \mathbf { F } } _ { t } ) - ( \bar { \mathbf { F } } _ { t + 1 } - \bar { \mathbf { F } } _ { t } ) \rVert _ { 1 } .\tag{11}
$$

To optimize local deformation under physical constraints, we introduce an elastic strain energy loss derived from continuum mechanics, which models the internal potential energy stored in a deformable material under external stress:

$$
\mathcal { L } _ { \mathrm { e l a } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \biggl [ \frac { \mu } { 2 } \left\| \nabla \hat { \mathbf { F } } _ { t } + \nabla \hat { \mathbf { F } } _ { t } ^ { \top } \right\| _ { \mathrm { F } } ^ { 2 } + \frac { \lambda } { 2 } \left[ \mathrm { t r } \left( \frac { \nabla \hat { \mathbf { F } } _ { t } + \nabla \hat { \mathbf { F } } _ { t } ^ { \top } } { 2 } \right) \right] ^ { 2 } \biggr ] ,\tag{12}
$$

where $\nabla \hat { \mathbf { F } } _ { t }$ is the spatial Jacobian of the optical flow, $( \cdot ) ^ { \top }$ denotes matrix transpose, tr(·) is the trace operator, and $\Vert \cdot \Vert _ { \mathrm { F } }$ is the Frobenius norm. The coefficients λ and $\mu$ are Lame parameters computed from the Young’s modulus ´ $E$ and Poisson’s ratio ν as $\begin{array} { r } { \mu = \frac { E } { 2 ( 1 + \nu ) } } \end{array}$ and $\begin{array} { r } { \lambda = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } } \end{array}$ . This term penalizes non-physical local deformations in the flow field, corresponding to minimizing the elastic strain energy. Finally, the total training objective is formulated as follows:

$$
\mathcal { L } _ { 1 } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { v e l } } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { \mathrm { a c c } } \mathcal { L } _ { \mathrm { a c c } } + \lambda _ { \mathrm { e l a } } \mathcal { L } _ { \mathrm { e l a } } ,\tag{13}
$$

where $\lambda _ { \mathrm { v e l } } , \lambda _ { \mathrm { a c c } }$ and $\lambda _ { \mathrm { e l a } }$ are balancing coefficients that control the relative contribution of each term.

FlowRender Training. To train FlowRender, we adopt a diffusion-based image-to-video learning paradigm in which the model learns to reconstruct video frames conditioned on the optical-flow guidance F<sup>ˆ</sup> . Specifically, we minimize the mean squared error between the denoised prediction and the groundtruth frame under random diffusion steps:

$$
\mathcal { L } _ { 2 } = \mathbb { E } _ { \tilde { \mathbf { R } } , t } \Big [ w _ { 2 } ( t ) \big | \big | D _ { \theta } ( \tilde { \mathbf { R } } _ { t } , t ) - \bar { \mathbf { R } } \big | \big | _ { 2 } ^ { 2 } \Big ] ,\tag{14}
$$

where R<sup>¯</sup> is the ground-truth RGB video. This training objective guides FlowRender to synthesize temporally coherent and visually realistic video sequences that faithfully follow the flow-based motion guidance.

## IV. EXPERIMENTS

## A. Experimental Setups

Implementation details. We train our model on 8 NVIDIA A100 GPUs with 80GB GPU memory. All inference experiments are performed on a single A100 GPU. To verify the performance of our proposed method, we construct an evaluation dataset comprising 60 samples, each containing an image I with a resolution of $5 1 2 \times 5 1 2$ , a corresponding textual prompt P, and physical properties provided by GPT-4o. The evaluation dataset includes rigid bodies, elastic objects, and selected flexible objects, together with scenarios involving complex-background contact and multi-object interaction. The image I is synthetically generated in a realistic style using a text-to-image generation model [58], and the evaluation also includes real-world natural images. The foreground mask $\mathbf { M } _ { f }$ is derived from I via Grounded-SAM [54]. Subsequently, the foreground multi-view video and corresponding 3D-GS representation are synthesized with Trellis [55], facilitating the simulation of physical-engine based baseline methods.

Baselines. We compare our method with representative baselines from two complementary families: physics-engine-based methods and video generative models. The first family evaluates whether explicit simulation pipelines can produce physically plausible dynamics when provided with reconstructed foreground geometry and scene priors. We therefore include PhysGen [12], PhysGen3D [13], and OmniPhysGS [15], which rely on physics simulation and 3D scene representations to drive object motion. The second family examines the behavior of data-driven video generators under the same image and prompt conditions. We evaluate general-purpose models, including CogVideoX [4] and Wan [5], to measure the physical consistency achievable by large-scale generative priors alone. In addition, we include Force Prompting [19] and PhysCtrl [20] as physics-enhanced generative baselines, since they introduce explicit force or trajectory controls to improve dynamic plausibility. This selection covers both simulator-driven and learned generation paradigms, enabling a comprehensive comparison across visual fidelity, temporal coherence, and physics-aware controllability.

Metrics. We evaluate generated videos using complementary objective metrics and human evaluation. VideoPhy-2 [59] reports semantic adherence (SA), which measures agreement with the input prompt; physical commonsense (PC), which assesses holistic compliance with real-world physics; and physical-rule adherence (Rule), which measures whether prompt-specific physical rules are satisfied. WorldScore [60] evaluates photo consistency (Photo), 3D consistency (3D), and motion smoothness (Motion), while VBench [61] measures motion smoothness (Motion), subject consistency (Subject), temporal flickering (Flicker), and image quality (Image). We additionally report the Mechanics dimension from VBench-2.0 [62], which evaluates whether generated dynamics follow basic mechanical principles such as gravity and stress. Following prior works [13], [14], we further evaluate the generated videos with GPT-4o along three dimensions: physical realism (Physical), photorealism (Photo), and semantic consistency (Semantic). To complement these model-based evaluations, we report Phys., a human-rated physical-plausibility score aggregated from 50 assessments. Higher values indicate better performance across all metrics.

![](images/96bfd9269a595a19fc6371d621e6346e8bc4a8e34dbee52d540d95a59d020911.jpg)  
Fig. 5. Qualitative comparisons. The top row shows the given image, text prompt and corresponding physics settings. For the definition of velocity and acceleration directions, the positive x-axis is defined as horizontal left, the positive y-axis as vertical upward, and the positive z-axis as perpendicular to the paper plane inward.

## B. Comparisons with State-Of-The-Art Methods

Quantitative Evaluation. Quantitatively, as shown in Table I, our method achieves superior motion coherence and temporal smoothness, consistently outperforming prior video generative models and shows comparable results with physics-enginebased methods across key dynamic metrics. Moreover, the generated videos exhibit higher visual quality than physicsengine-based methods. Regarding physical plausibility, as demonstrated in Table II, our method surpasses all competing approaches on the physics realism metric, while simultaneously maintaining strong alignment with the input text, thereby ensuring high semantic consistency.

TABLE II  
COMPLEMENTARY EVALUATION RESULTS. PHYSICAL, PHOTO, AND SEMANTIC ARE EVALUATED BY GPT-4O; MECHANICS IS FROM VBENCH-2.0; AND PHYS. IS BASED ON 50 HUMAN ASSESSMENTS.
<table><tr><td rowspan="2">Method</td><td colspan="3">GPT-40</td><td rowspan="2">VBench-2.0 | Human</td><td rowspan="2">Phys.↑</td></tr><tr><td>|Physical↑</td><td>Photo↑</td><td>Semantic↑ | Mechanics↑</td></tr><tr><td>OmniPhysGS [15]</td><td>0.467</td><td>0.293</td><td>0.433</td><td>0.316</td><td>0.564</td></tr><tr><td rowspan="4">PhysGen [12] PhysGen3D [13]</td><td>0.530</td><td>0.890</td><td>0.598</td><td>0.417</td><td>0.608</td></tr><tr><td>0.597</td><td>0.768</td><td>0.594</td><td>0.387</td><td>0.552</td></tr><tr><td>CogVideoX-I2V-5B [4] 0.570</td><td>0.875</td><td>0.709</td><td>0.403</td><td>0.566</td></tr><tr><td>0.567</td><td>0.891</td><td>0.733</td><td>0.362</td><td>0.560</td></tr><tr><td>Wan2.2-TI2V-5B [5] Force Prompting [19]</td><td>0.525</td><td>0.886</td><td>0.559</td><td>0.354</td><td>0.544</td></tr><tr><td>PhysCtrl [20]</td><td>0.582</td><td>0.855</td><td>0.703</td><td>0.362</td><td>0.586</td></tr><tr><td>PhysFlow(Ours)</td><td>0.613</td><td>0.903</td><td>0.744</td><td>0.531</td><td>0.708</td></tr></table>

Qualitative Evaluation. As illustrated in Figure 5, we present two cases for qualitative evaluation. For the first scenario involving a non-rigid object with complex shape, Phys-Gen3D [13] suffers from texture loss on the foreground object due to insufficient mesh-based rendering. CogVideoX [4] fails to capture the correct motion dynamics, resulting in noticeable flickering. Wan [5] can partially model the falling motion but misinterprets the material and interaction properties, causing the object to disperse like sand during descent. PhysCtrl [20] misestimates the ground location, causing the foreground object to deform prematurely while still suspended in the air. In contrast, our model accurately captures the falling behavior and produces physically plausible elastic deformations. In the second scenario involving multi-object interactions, PhysGen3D [13] recognizes only one moving ball, leaving the others suspended in midair. CogVideoX [4] generates inconsistent motion after the fall, introducing several spurious red balls. Wan [5] and PhysCtrl [20] both fail to faithfully capture multi-object dynamics, with several balls remaining suspended in midair. In comparison, our model successfully understands the multi-ball interaction dynamics, generating coherent motion for all objects. To better demonstrate our method’s performance on challenging scenarios, we present qualitative examples in Figure 6, covering interactions with fluid backgrounds as well as complex interactions among multiple objects. More visual comparisons with baselines are provided in the Appendix Section C in the Supplementary Material.

![](images/59df8dd6b6c31760a4b8a0d39df4958b71fc8ae6d44715c3f51ef3447fc5be94.jpg)  
Fig. 6. Qualitative examples of challenging scenarios. Each row shows a temporal sequence (left to right) under an applied external force F (red arrow), highlighting interactions with complex backgrounds and multi-object interactions.

Diverse Motion Patterns  
![](images/a6e2110b19745cd98c6e7ca619d27bc66109965abee3e23b32fab73756a37f98.jpg)

![](images/608fb831136446b637715e71df3d5044323a165db4e91a06211e3ea84be26480.jpg)  
Fig. 7. Diverse motions and multi-object interactions. PhysFlow generates diverse motion types and natural multi-object interactions.

Results on Varying Physical Properties. Real-world dynamic phenomena are fundamentally governed by physical laws, where variations in physical properties lead to distinct motion behaviors. In this analysis, we examine representative motion and material attributes, focusing on initial velocity, Young’s modulus, and density. To assess physical consistency, we visualize both the generated optical flow and the rendered videos under different physical configurations. Since our framework predicts optical flow before RGB rendering, we first inspect the intermediate PA-Flow output. Figure 9 compares the ground-truth flow and PA-Flow predictions at matched time steps. Figure 10 then examines how PA-Flow responds to changes in initial velocity by holding all other conditions fixed and comparing the predicted optical flows under the original, doubled-magnitude, and reversed-direction velocity settings. Having established velocity-conditioned responses at the flow level, we next examine how physics-conditioned motion representations are reflected in the final visual output. Figure 11 jointly presents PA-Flow predictions and representative RGB frames from the corresponding FlowRender outputs. The top two rows vary the initial-velocity direction, whereas the bottom two rows vary Young’s modulus and density, respectively. The paired results show that changes in motion and material conditions are reflected in both the intermediate optical flow and the rendered RGB output. In particular, changes in initial-velocity direction produce corresponding motion outcomes, while higher Young’s modulus and lower density lead to smaller deformations, consistent with expected physical behavior. Beyond qualitative visualization, we further construct a physics-conditioned validation set to quantitatively evaluate controllability under varying physical properties. In this validation set, different methods are tested under controlled changes of motion and material attributes, allowing us to measure whether the generated videos remain visually coherent while following the specified physical conditions. As reported in Table III, our method achieves stronger physics-conditioned generation performance, further verifying its controllable behavior across different physical settings.

![](images/8722f99d52c8683755bd2373c54bbe57787a60b1e1f5ab58e39729f829e8813e.jpg)  
Fig. 8. Real-world generalization examples. PhysFlow can generate natural motion in real-world environments.

Diverse motion and multi-object interaction results. Figure 7 (a)–(c) shows that our method supports different motion types, including rigid sliding, cloth deformation, and bouncing. The observed variations indicate that the model captures object- and condition-dependent motion characteristics rather than producing a uniform motion response across scenarios. Figure 7 (d)–(f) show results on scenes with multiple interacting objects. Our method maintains temporally coherent motion across objects and produces plausible responses upon contact.

Generalization beyond simulation. Figure 8 presents results on real-world images, which differ substantially in appearance from the synthetic training data. Despite this domain gap, our method predicts plausible motion fields and generates temporally coherent videos. By separating motion modeling from RGB synthesis through an intermediate flow representation, the two-stage design makes the predicted dynamics less dependent on simulator-specific appearance cues.

![](images/2d74453a31277c3d19b9d57a9de14deb729f4eea2ce93c5bb0c4ce2e085e7bd2.jpg)

Fig. 9. Qualitative comparison of optical flow predictions. The left column shows the input RGB image, while the middle and right panels show th ground-truth optical flow and PA-Flow predictions, respectively, at three matched time steps. PA-Flow produces motion patterns that closely follow the corresponding ground-truth flow.  
![](images/08f80a8b61f8220afece02ab61c7eba49849e232f1461451c53006f617ba0746.jpg)  
Fig. 10. Effects of initial velocity on PA-Flow predictions. (a) and (b) show two representative examples. For each example, optical-flow predictions at two time steps are compared under the original, doubled-magnitude, and reversed-direction velocity settings, with all other conditions held fixed.

TABLE III  
QUANTITATIVE COMPARISON ON PHYSICS-CONDITIONED VIDEOGENERATION.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>PhysGen [12]</td><td>19.82</td><td>0.767</td><td>0.438</td><td>122.03</td><td>119.64</td></tr><tr><td>Force Prompting [19]</td><td>21.08</td><td>0.847</td><td>0.316</td><td>105.22</td><td>103.47</td></tr><tr><td>PhysCtrl [20]</td><td>22.47</td><td>0.875</td><td>0.282</td><td>100.40</td><td>83.25</td></tr><tr><td>Ours</td><td>25.32</td><td>0.895</td><td>0.241</td><td>90.56</td><td>69.54</td></tr></table>

## C. Ablation Studies

Ablation study on key modules. PAAM combines a global motion attention module (Motion-Attn) and a local deformation module (Deform-Attn) to model global displacement and local deformation, respectively. Table IV reports an incremental ablation starting from the baseline, with SFT, physical conditioning, PAAM, and Phys-FT added in sequence. Each configuration retains everything included in the previous row. Adding PAAM improves all reported metrics, including Physical from 0.594 to 0.608 and PC from 3.29 to 3.37. FlowRender is first trained on OpenVid [6] and then finetuned on WISA [22]; this Phys-FT stage further raises Physical to 0.613 and PC to 3.41. Figure 14 complements the unified table by qualitatively showing the effects of Motion-Attn and Deform-Attn.

TABLE IV  
INCREMENTAL ABLATION OF THE MODEL, WITH EACH ROW BUILDING ON THE ONE ABOVE.
<table><tr><td>Method</td><td>Motion↑</td><td>Image↑</td><td>Physical↑</td><td>SA↑</td><td>PC↑</td><td>Rule↑</td></tr><tr><td>Baseline</td><td>0.992</td><td>0.628</td><td>0.570</td><td>2.66</td><td>3.14</td><td>0.221</td></tr><tr><td>+SFT</td><td>0.994</td><td>0.633</td><td>0.588</td><td>2.72</td><td>3.25</td><td>0.253</td></tr><tr><td>+Condition</td><td>0.994</td><td>0.634</td><td>0.594</td><td>2.76</td><td>3.29</td><td>0.258</td></tr><tr><td>+PAAM</td><td>0.996</td><td>0.636</td><td>0.608</td><td>2.79</td><td>3.37</td><td>0.267</td></tr><tr><td>+Phys-FT</td><td>0.997</td><td>0.640</td><td>0.613</td><td>2.81</td><td>3.41</td><td>0.281</td></tr></table>

TABLE V  
ABLATION ON MASK AND MULTI-VIEW PRIORS.
<table><tr><td>Method</td><td>Motion↑</td><td>Image↑</td><td>Physical↑</td><td>SA↑</td><td>PC↑</td><td>Rule↑</td></tr><tr><td>w/o mask prior</td><td>0.992</td><td>0.638</td><td>0.584</td><td>2.62</td><td>3.24</td><td>0.236</td></tr><tr><td>w/o multi-view prior</td><td>0.995</td><td>0.636</td><td>0.605</td><td>2.78</td><td>3.29</td><td>0.248</td></tr><tr><td>Ours</td><td>0.997</td><td>0.640</td><td>0.613</td><td>2.81</td><td>3.41</td><td>0.281</td></tr></table>

Ablation on priors. To improve PA-Flow’s semantic and 3D consistency, we condition the model on the mask video $V _ { m }$ and multi-view video $V _ { m v }$ . Figure 12 shows that removing $V _ { m }$ leads to ambiguous motion regions, while removing $V _ { m v }$ causes unrealistic deformations from weak global 3D awareness. As shown in Table V, removing the mask prior or multi-view prior consistently degrades performance, especially on physical plausibility, reflecting the importance of these components.

![](images/12fed5cb05f83c14c01d07e99d3ba7c88d5e41205ab7a76df05e0907469049a2.jpg)  
Fig. 11. Physics-conditioned flow-to-video generation. The top two rows compare three initial-velocity directions indicated by the numbered arrows, whereas the bottom two rows vary Young’s modulus and density, respectively. For each setting, the PA-Flow prediction is paired with a representative RGB frame from the corresponding FlowRender output, illustrating how motion and material conditions are reflected in the final video dynamics.

Input Image  
w/o. �<sub>�</sub>  
w/o �<sub>��</sub>  
![](images/4ea4d2d2857c51a2d7a75a6fc3b377786045e38680acc9ac5235c59eccb2430e.jpg)  
Ours  
Fig. 12. Ablation study on conditional videos. The left column shows the input single image, and the three right columns each display one representative frame from the generated RGB video.

TABLE VI  
ABLATION STUDY ON PROPOSED LOSS FUNCTIONS. THE RESULTS ARE EVALUATED WITH GPT-4O.
<table><tr><td>Method</td><td>Motion↑</td><td>Image↑|</td><td>Physical↑|</td><td>SA↑</td><td>PC↑</td><td>Rule↑</td></tr><tr><td>w/o  $\mathcal { L } _ { v e l }$ </td><td>0.996</td><td>0.637</td><td>0.607</td><td>2.75</td><td>3.36</td><td>0.277</td></tr><tr><td>w/o  $\mathcal { L } _ { a c c }$ </td><td>0.997</td><td>0.638</td><td>0.610</td><td>2.78</td><td>3.38</td><td>0.271</td></tr><tr><td>w/o  $\mathcal { L } _ { e l a }$ </td><td>0.995</td><td>0.635</td><td>0.605</td><td>2.75</td><td>3.32</td><td>0.266</td></tr><tr><td>Ours</td><td>0.997</td><td>0.640</td><td>0.613</td><td>2.81</td><td>3.41</td><td>0.281</td></tr></table>

![](images/74e3bc738fd413c83b9a75c1e32a52198b306629c126e750999671777c91a242.jpg)

![](images/d53081819c871fbef84abce1fd65a3e4f6daf0cfe285e5f35df1fe2efc918f40.jpg)  
Fig. 13. Physics-controlled results with abnormal inputs. With extremely incorrect physical attributes or priors, the generated outputs can be anomalous.

Ablation study on loss functions. We also evaluate the effectiveness of the proposed loss functions, including the velocity-smoothness loss $\mathcal { L } _ { v e l }$ , acceleration-consistency loss $\mathcal { L } _ { a c c } ,$ , and elastic strain energy loss $\mathcal { L } _ { e l a }$ . These losses are designed to provide complementary motion regularization: $\mathcal { L } _ { v e l }$ penalizes abrupt velocity changes, $\mathcal { L } _ { a c c }$ penalizes inconsistent acceleration patterns, and $\mathcal { L } _ { e l a }$ introduces an elastic-energy prior for deformation regions. As shown in Table VI, removing any of these losses degrades performance, confirming that both temporal regularity and deformation constraints are useful. Among them, removing $\mathcal { L } _ { e l a }$ leads to the largest drop, indicating that elastic strain energy plays a particularly important role in modeling non-rigid deformation and improving overall physical plausibility.

Physics: "material":{"density":"700","youngs\_modulus":"1e7", "poissons\_ratio":"0.40"}, "motion":{"acceleration":"[0.0, -9.8, -0.0]", "velocity":"[-1.0, -3.0, 0.1]"}

![](images/e3928e5063a764aa485233cdb6f743c2242d22ce7ef11b2b66981dc4d67c84e0.jpg)

Prompt: The yellow leaf falls onto the ground slowly. Physics:

"material":{"density":”600","youngs\_modulus":"1e6", "poissons\_ratio":"0.45"},

"motion":{”acceleration":"[0.0, -9.8, -0.0]", “velocity”:“[-1.0, -3.0, 0.0]"}

![](images/8dbc2aaea5e94367e910ee0f24fb2960015cafe8e4ff4a6b7bd5220acb6289fb.jpg)

![](images/ad51d27bfaecb0c1fbc19fe74393876a2479f5aa9f1288e71fecce934da54a94.jpg)  
Baseline +Motion-Attn+Deform-Attn +Phys-FT

Prompt: A wooden basin falls onto the ground.

![](images/5149769c59cc7ecb0558225865a46df3ddb3375931a191879788ed45acc853c0.jpg)  
Fig. 14. Qualitative ablation of key components. For each example, all variants use the same input image, text prompt, and physical conditions, with frames arranged temporally from left to right. From top to bottom, Motion-Attn, Deform-Attn, and physics-focused fine-tuning (Phys-FT) are progressively added to the baseline, illustrating their complementary contributions to global motion, local deformation, and overall physical plausibility.

Failure case analysis. Our model supports controllable video generation with respect to both material and motion attributes. When the specified physical attributes or input priors deviate substantially from plausible values, the generated results may exhibit visually implausible physical behavior. For failure case analysis, Figure 13 presents four representative examples arising from different types of upstream errors. Incorrect mask priors may activate unintended motion regions, inaccurate 3D priors may lead to unrealistic deformation, and inappropriate values of Young’s modulus or density may result in excessive deformation.

## V. CONCLUSION

In this work, we presented a physics-aware video generation framework for object dynamics spanning rigid-body motion, elastic deformation, and flexible-body motion. The framework disentangles object-motion generation from motion-guided texture synthesis. Our two-stage design supports physics-aware optical flow generation and optical flow-guided plausible video generation without relying on physics engines. Furthermore, we provide a large-scale physics-engine–based video dataset that uniquely combines realistic backgrounds with detailed annotations of material and motion properties. Experimental results demonstrate that our method achieves superior motion consistency and visual quality compared with prior methods.

## REFERENCES

[1] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.

[2] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-sora: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.

[3] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.

[4] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” in The Thirteenth International Conference on Learning Representations, 2025.

[5] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[6] K. Nan, R. Xie, P. Zhou, T. Fan, Z. Yang, Z. Chen, X. Li, J. Yang, and Y. Tai, “Openvid-1m: A large-scale high-quality dataset for text-to-video generation,” arXiv preprint arXiv:2407.02371, 2024.

[7] H. Bansal, Z. Lin, T. Xie, Z. Zong, M. Yarom, Y. Bitton, C. Jiang, Y. Sun, K.-W. Chang, and A. Grover, “Videophy: Evaluating physical commonsense for video generation,” arXiv preprint arXiv:2406.03520, 2024.

[8] F. Meng, J. Liao, X. Tan, W. Shao, Q. Lu, K. Zhang, Y. Cheng, D. Li, Y. Qiao, and P. Luo, “Towards world simulator: Crafting physical commonsense-based benchmark for video generation,” arXiv preprint arXiv:2410.05363, 2024.

[9] X. Chi, P. Jia, C.-K. Fan, X. Ju, W. Mi, K. Zhang, Z. Qin, W. Tian, K. Ge, H. Li et al., “Wow: Towards a world omniscient world model through embodied interaction,” arXiv preprint arXiv:2509.22642, 2025.

[10] J. Mao, S. He, H.-N. Wu, Y. You, S. Sun, Z. Wang, Y. Bao, H. Chen, L. Guibas, V. Guizilini et al., “Robot learning from a physical world model,” arXiv preprint arXiv:2511.07416, 2025.

[11] Z. Yang, Z. Liu, Y. Lu, L. Hou, C. Miao, S. Peng, B. Feng, X. Bai, and H. Zhao, “Geniedrive: Towards physics-aware driving world model with 4d occupancy guided video generation,” in Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 35 680–35 690.

[12] S. Liu, Z. Ren, S. Gupta, and S. Wang, “Physgen: Rigid-body physicsgrounded image-to-video generation,” in European Conference on Computer Vision, 2024, pp. 360–378.

[13] B. Chen, H. Jiang, S. Liu, S. Gupta, Y. Li, H. Zhao, and S. Wang, “Physgen3d: Crafting a miniature interactive world from a single image,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6178–6189.

[14] Z. Li, H.-X. Yu, W. Liu, Y. Yang, C. Herrmann, G. Wetzstein, and J. Wu, “Wonderplay: Dynamic 3d scene generation from a single image and actions,” arXiv preprint arXiv:2505.18151, 2025.

[15] Y. Lin, C. Lin, J. Xu, and Y. Mu, “Omniphysgs: 3d constitutive gaussians for general physics-based dynamics generation,” arXiv preprint arXiv:2501.18982, 2025.

[16] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang, “Physgaussian: Physics-integrated 3d gaussians for generative dynamics,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4389–4398.

[17] J. Lin, Z. Wang, D. Xu, S. Jiang, Y. Gong, and M. Jiang, “Phys4dgen: Physics-compliant 4d generation with multi-material composition perception,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 398–10 407.

[18] F. Liu, H. Wang, S. Yao, S. Zhang, J. Zhou, and Y. Duan, “Physics3d: Learning physical properties of 3d gaussians via video diffusion,” arXiv preprint arXiv:2406.04338, 2024.

[19] N. Gillman, C. Herrmann, M. Freeman, D. Aggarwal, E. Luo, D. Sun, and C. Sun, “Force prompting: Video generation models can learn and generalize physics-based control signals,” arXiv preprint arXiv:2505.19386, 2025.

[20] C. Wang, C. Chen, Y. Huang, Z. Dou, Y. Liu, J. Gu, and L. Liu, “Physctrl: Generative physics for controllable and physics-grounded video generation,” arXiv preprint arXiv:2509.20358, 2025.

[21] Q. Xue, X. Yin, B. Yang, and W. Gao, “Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 826–18 836.

[22] J. Wang, A. Ma, K. Cao, J. Zheng, Z. Zhang, J. Feng, S. Liu, Y. Ma, B. Cheng, D. Leng et al., “Wisa: World simulator assistant for physicsaware text-to-video generation,” arXiv preprint arXiv:2503.08153, 2025.

[23] Z. Li, R. Tucker, N. Snavely, and A. Holynski, “Generative image dynamics,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 142–24 153.

[24] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in neural information processing systems, vol. 35, pp. 8633–8646, 2022.

[25] X. Fu, X. Liu, X. Wang, S. Peng, M. Xia, X. Shi, Z. Yuan, P. Wan, D. Zhang, and D. Lin, “3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation,” arXiv preprint arXiv:2412.07759, 2024.

[26] H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang, “Cameractrl: Enabling camera control for text-to-video generation,” arXiv preprint arXiv:2404.02101, 2024.

[27] D. Geng, C. Herrmann, J. Hur, F. Cole, S. Zhang, T. Pfaff, T. Lopez-Guevara, Y. Aytar, M. Rubinstein, C. Sun et al., “Motion prompting: Controlling video generation with motion trajectories,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1–12.

[28] Z. Gu, R. Yan, J. Lu, P. Li, Z. Dou, C. Si, Z. Dong, Q. Liu, C. Lin, Z. Liu et al., “Diffusion as shader: 3d-aware video diffusion for versatile video generation control,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025, pp. 1–12.

[29] R. Burgert, Y. Xu, W. Xian, O. Pilarski, P. Clausen, M. He, L. Ma, Y. Deng, L. Li, M. Mousavi et al., “Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 13–23.

[30] Z. Chen, T. Xu, L. Wu, L. Wang, D. Yan, Z. You, W. Luo, G. Zhang, and Y. Chen, “Stance: Motion coherent video generation via sparse-to-dense anchored encoding,” arXiv preprint arXiv:2510.14588, 2025.

[31] D. Romero, A. Bermudez, H. Li, F. Pizzati, and I. Laptev, “Learning to generate object interactions with physics-guided video diffusion,” arXiv preprint arXiv:2510.02284, 2025.

[32] C. Wang, P. Hu, H. Zhao, Y. Guo, J. Gu, X. Dong, J. Han, H. Xu, and X. Liang, “Uniadapter: All-in-one control for flexible video generation,”

IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 6, pp. 6059–6073, 2025.

[33] Y. Wen, Y. Zhao, Y. Liu, B. Huang, F. Jia, Y. Wang, C. Zhang, T. Wang, X. Sun, and X. Zhang, “Panacea+: Panoramic and controllable video generation for autonomous driving,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 2, pp. 2247–2258, 2026.

[34] Y. Yang, G. Yue, W. Zhou, X. Mao, R. Wang, and B. Zhao, “Expressive human volumetric video generation with rich text,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 4, pp. 5424– 5436, 2026.

[35] W. Jin, Q. Dai, C. Luo, S.-H. Baek, and S. Cho, “Flovd: Optical flow meets video diffusion model for enhanced camera-controlled video synthesis,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 2040–2049.

[36] F. Liang, B. Wu, J. Wang, L. Yu, K. Li, Y. Zhao, I. Misra, J.-B. Huang, P. Zhang, P. Vajda et al., “Flowvid: Taming imperfect optical flows for consistent video-to-video synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8207–8216.

[37] J. Lv, Y. Huang, M. Yan, J. Huang, J. Liu, Y. Liu, Y. Wen, X. Chen, and S. Chen, “Gpt4motion: Scripting physical motions in text-to-video generation via blender-oriented gpt planning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 1430–1440.

[38] R. Zhang, Y. Chen, Y. Liu, W. Wang, X. Wen, and H. Wang, “Tvg: A training-free transition video generation method with diffusion models,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 8, pp. 7471–7484, 2025.

[39] C. Wang, S. Yan, Y. Chen, X. Wang, Y. Wang, M. Dong, X. Yang, D. Li, R. Zhu, D. A. Clifton, R. P. Dick, Q. Lv, F. Yang, T. Lu, N. Gu, and L. Shang, “Denoising reuse: Exploiting inter-frame motion consistency for efficient video generation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 9, pp. 8436–8451, 2025.

[40] Y. Zhang, Z. Yang, Z. Su, Y. Hu, and C. W. Chen, “Mogenvd: A motioncentered quality assessment benchmark for text-to-video generation,” IEEE Transactions on Circuits and Systems for Video Technology, pp. 1–1, 2026.

[41] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[42] Y. Feng, X. Feng, Y. Shang, Y. Jiang, C. Yu, Z. Zong, T. Shao, H. Wu, K. Zhou, C. Jiang et al., “Gaussian splashing: Dynamic fluid synthesis with gaussian splatting,” CoRR, 2024.

[43] Y. Gao, H.-X. Yu, B. Zhu, and J. Wu, “Fluidnexus: 3d fluid reconstruction and prediction from a single video,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 091–26 101.

[44] H. Zhao, H. Wang, X. Zhao, H. Fei, H. Wang, C. Long, and H. Zou, “Physsplat: Efficient physics simulation for 3d scenes via mllm-guided gaussian splatting,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 5242–5252.

[45] H. Mao, Z. Xu, S. Wei, Y. Quan, N. Deng, and X. Yang, “Live-gs: Llm powers interactive vr by enhancing gaussian splatting,” in 2025 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW). IEEE, 2025, pp. 1234–1235.

[46] T. Huang, H. Zhang, Y. Zeng, Z. Zhang, H. Li, W. Zuo, and R. W. Lau, “Dreamphysics: Learning physics-based 3d dynamics with video diffusion priors,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2025, pp. 3733–3741.

[47] T. Zhang, H.-X. Yu, R. Wu, B. Y. Feng, C. Zheng, N. Snavely, J. Wu, and W. T. Freeman, “Physdreamer: Physics-based interaction with 3d objects via video generation,” in European Conference on Computer Vision, 2024, pp. 388–406.

[48] H. Zhu, C. Wang, T. He, L. Chen, X. Jin, C. Gao, and Z. Chen, “Cp4d: Compositional physics-aware 4d scene generation,” arXiv preprint arXiv:2606.09187, 2026.

[49] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Textto-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.

[50] Y. Cheng, F. Yin, X. Huang, X. Yu, J. Liu, S. Feng, Y. Yang, and Y. Tang, “Efficient text-guided 3d-aware generation with score distillation on 3d distribution,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 10, pp. 9865–9877, 2025.

[51] J. Zhang, X. Li, H. Zhong, Q. Zhang, Y. Cao, Y. Shan, and J. Liao, “Humanref-gs: Image-to-3d human generation with reference-guided diffusion and 3d gaussian splatting,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 7, pp. 6867–6880, 2025.

[52] C. Wang, H. Zhu, J. Luo, Y. Tian, X. Cheng, P. Tu, X. Jin, L. Chen, and Z. Chen, “Physics-informed video generation via mixture-of-experts latent alignment,” arXiv preprint arXiv:2606.04737, 2026.

[53] Y. Yuan, X. Wang, T. Wickremasinghe, Z. Nadir, B. Ma, and S. H. Chan, “Newtongen: Physics-consistent and controllable text-to-video generation via neural newtonian dynamics,” arXiv preprint arXiv:2509.21309, 2025.

[54] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

[55] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang, “Structured 3d latents for scalable and versatile 3d generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 469–21 480.

[56] Y. Hu, T.-M. Li, L. Anderson, J. Ragan-Kelley, and F. Durand, “Taichi: a language for high-performance computation on spatially sparse data structures,” ACM Transactions on Graphics (TOG), pp. 1–16, 2019.

[57] Q. Dong and Y. Fu, “Memflow: Optical flow estimation and prediction with memory,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 068–19 078.

[58] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

[59] H. Bansal, C. Peng, Y. Bitton, R. Goldenberg, A. Grover, and K.-W. Chang, “VideoPhy-2: A challenging action-centric physical commonsense evaluation in video generation,” arXiv preprint arXiv:2503.06800, 2025.

[60] H. Duan, H.-X. Yu, S. Chen, L. Fei-Fei, and J. Wu, “Worldscore: A unified evaluation benchmark for world generation,” arXiv preprint arXiv:2504.00983, 2025.

[61] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 807–21 818.

[62] D. Zheng, Z. Huang, H. Liu, K. Zou, Y. He, F. Zhang, Y. Zhang, J. He, W.-S. Zheng, Y. Qiao, and Z. Liu, “VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness,” arXiv preprint arXiv:2503.21755, 2025.