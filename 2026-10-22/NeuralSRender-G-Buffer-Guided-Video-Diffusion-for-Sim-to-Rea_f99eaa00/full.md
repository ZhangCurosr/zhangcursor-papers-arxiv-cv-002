# NeuralSRender: G-Buffer Guided Video Diffusion for Sim-to-Real Rendering

Yipeng Lu, Youqi Liao, Zhenyu Wang, Yuning Peng, and Zhen Dong

Abstract— The scarcity of high-quality annotated visual data remains a critical bottleneck for autonomous-driving perception. Simulators provide training data at scale, but the limited realism of synthetic videos compromises perception performance in real-world applications. To address this limitation, we propose NeuralSRender, a sim-to-real video translation framework that enhances the photorealism of synthetic videos while maintaining geometric consistency with the source scenes. Built on a pretrained video diffusion transformer (DiT), NeuralSRender uses a lightweight projector to inject geometric and material priors from G-buffers, and LoRA enables parameter-efficient adaptation. The model is trained in two stages: it first learns G-buffer conditioning from synthetic videos with accurate G-buffers, then adapts to real-world appearance using real videos with estimated G-buffers. During the second stage, an additional LoRA module is trained while the projector and first-stage LoRA remain frozen to reduce interference from estimation noise. NeuralSRender introduces only 1.45% parameter overhead and can be trained on consumer-grade GPUs with 24 GB of memory. To support training and evaluation, we introduce GenReal, a large-scale driving video dataset comprising synthetic and real-world videos with corresponding G-buffers. Experiments demonstrate improved photorealism with competitive geometric consistency. Training perception models on our translated videos instead of raw simulator renders improves detection $\mathrm { A P _ { 5 0 } }$ on nuImages from 35.1 to 40.6 and segmentation mIoU on Cityscapes from 32.4 to 40.8. Our code, data, and model weights are available at: https://github.com/WHU-USI3DV/NeuralSRender.

## I. INTRODUCTION

Training models for autonomous driving [1], [2], [3], [4], [5] requires large-scale, diverse driving data and task-specific supervision. However, collecting and annotating real-world driving data at scale is costly and labor-intensive. Moreover, routine data collection struggles to capture rare but safetycritical scenarios, leaving gaps in training-data coverage.

Simulators provide a scalable alternative to real-world data collection [6], [7], [8], [9], offering automatic annotations and controlled generation of rare and safety-critical driving scenarios. However, their rendered images and videos often differ substantially in appearance from real-world data, which can limit the real-world generalization of models trained on synthetic data. [10]. This appearance gap reflects practical limitations in both asset creation and rendering. First, creating realistic textures and materials is costly and difficult to scale across diverse scenes. Meanwhile, real-time rendering engines often rely on simplified models of light transport to meet computational budgets, limiting their ability to reproduce complex lighting effects. These limitations motivate methods to improve photorealism in simulated data while preserving scene geometry and annotation consistency.

Recent advances in controllable video generation and neural rendering offer promising avenues for sim-to-real translation. Controllable video generation methods use semantic, structural, or motion cues to guide generation [4], [11], [12], [13]. Although these cues help constrain scene layout and motion, they do not generally provide a joint, explicit description of geometry and material properties. Neural rendering methods incorporate richer scene representations, including geometric and material attributes, to support controllable appearance synthesis [14], [15], [16]. However, leveraging these attributes for sim-to-real translation presents a challenge in training: synthetic videos provide accurate simulator-derived attributes but limited appearance realism, whereas real videos offer realistic appearance but typically rely on noisy estimates of the corresponding attributes. The key challenge is therefore to learn reliable geometric and material conditioning while adapting to real-world appearance, improving photorealism without compromising scene structure and annotation consistency.

To address this challenge, we propose NeuralSRender, a parameter-efficient framework for G-buffer-guided sim-toreal video translation. As illustrated in Fig. 1, NeuralSRender synthesizes realistic driving scenes under diverse textspecified weather and time-of-day conditions while preserving the overall scene layout. A lightweight projector integrates dense geometric and material features into a pretrained video diffusion model. Our two-stage training strategy separates reliable conditioning learning from real-world appearance adaptation. We first jointly train the projector and Stage-I LoRA modules on synthetic videos with accurate Gbuffers. We then freeze both and train additional LoRA modules on real videos with estimated G-buffers [14] for appearance adaptation. The framework introduces only a 1.45% parameter overhead relative to the backbone. We also introduce GenReal, a large-scale autonomous-driving dataset spanning two synthetic domains and one real-world domain. GenReal brings together driving videos, simulator-derived or estimated G-buffer representations, and descriptive text prompts, providing a data resource for learning geometric and material conditioning and systematically evaluating simto-real video translation across domains. Experiments demonstrate that NeuralSRender achieves superior visual realism among the evaluated baselines while maintaining competitive geometric consistency. Models trained on our translated synthetic data outperform those trained on other synthetic data in object detection and semantic segmentation on realworld benchmarks. Our main contributions are summarized as follows:

![](images/11bfaa5a3e5155df69548130e1606bfc1a1679c902b45f72b6a70c9fdc34db38.jpg)  
Fig. 1. Photorealistic sim-to-real synthesis with the proposed NeuralSRender. Our approach leverages video generation priors to synthesize photorealistic driving videos from G-buffers, reducing reliance on high-fidelity textures and materials while enabling text-guided changes in weather and time of day.

• We propose NeuralSRender, a parameter-efficient framework for sim-to-real video translation. It combines dense geometric and material conditioning with pretrained video generative priors, introducing only a 1.45% parameter overhead relative to the backbone.

• We develop a two-stage training strategy to improve photorealism while preserving reliable G-buffer control. It first learns conditioning from accurate synthetic Gbuffers, then adapts appearance using real videos while freezing the learned conditioning components to mitigate interference from noisy attribute estimates.

• We construct GenReal, a large-scale driving video dataset spanning two synthetic domains and one realworld domain, with G-buffer annotations and text prompts for training and cross-domain evaluation.

## II. RELATED WORK

a) Controllable Video Generation.: Controllable video generation aims to synthesize videos under spatial or semantic conditions such as text, pose, depth, and edges. Methods including ControlNet [17], T2I-Adapter [18], VideoComposer [19], SparseCtrl [20], and Ctrl-Adapter [21] introduce trainable branches or lightweight conditioning modules to inject external controls. Recent video frameworks such as VACE [22] and CogVideoX [23] extend this paradigm with temporal modeling and more flexible conditioning. However, most existing methods rely on sparse or highlevel control signals and do not explicitly model the dense geometry and material interactions required for sim-to-real rendering. In contrast, NeuralSRender conditions a pretrained video diffusion model on multi-attribute G-buffers, providing aligned geometric and material guidance for realistic video translation.

b) Sim-to-Real Translation.: Sim-to-real translation aims to reduce the visual gap between simulated and realworld data. Early image-based methods, such as RCAN [25], transform randomized simulator outputs into a canonical domain. A recent diffusion-based method [26] further improves visual realism using text prompts or reference images. However, image-level translation does not model temporal consistency. Video-to-video synthesis [27] generates temporally coherent sequences from semantic maps, but does not explicitly exploit the dense geometry and material information available in simulators. In this work, NeuralSRender uses Gbuffers as a shared structural representation across synthetic and real domains, providing dense guidance for sim-to-real video translation.

c) Neural Rendering.: Recent studies have used diffusion priors for image and video relighting without explicit 3D reconstruction. IC-Light [28] models light transport consistency in the latent space, while Neural Gaffer [29] uses synthetic data to handle complex illumination and reflective materials. For video relighting, Light-A-Video [30] improves cross-frame consistency, and TC-Light [15] preserves dynamic geometry while adjusting global illumination. However, these methods mainly focus on lighting changes. They do not fully synthetic to real-world domains.

TABLE I: DETAILED STATISTICS OF THE GENREAL DATASET.
<table><tr><td>Domain</td><td>Source</td><td>Subset</td><td>Split</td><td>Count</td><td>Resolution</td><td>Frames</td><td>G-buffer</td></tr><tr><td rowspan="2">Synthetic</td><td>CityDataset</td><td>GenReal-City</td><td>Train Test</td><td>1,795 140</td><td> $8 3 2 \times 4 8 0$   $1 2 8 0 \times 7 0 4$ </td><td>81 81</td><td>Accurate Accurate</td></tr><tr><td>CARLA [6]</td><td>GenReal-CARLA</td><td>Test</td><td>1,143</td><td> $1 2 8 0 \times 7 0 4$ </td><td>81</td><td>Accurate</td></tr><tr><td>Real</td><td>OpenDV [24]</td><td>GenReal-DV</td><td>Train Reference</td><td>5,819 7,903</td><td> $8 3 2 \times 4 8 0$ </td><td>81</td><td>Estimated</td></tr></table>

address the broader appearance adaptation required for simto-real translation, including changes in textures, materials, and weather. DiffusionRenderer [14] extracts G-buffers from videos and uses them as structural conditions for forward video diffusion. However, it mainly targets object-centric relighting and does not use text guidance. This limits flexible control over global appearance and large domain shifts. In contrast, NeuralSRender combines dense G-buffer conditions with text guidance and a two-stage adaptation strategy for sim-to-real video translation in complex outdoor scenes.

## III. GENREAL DATASET

Existing datasets rarely provide both realistic driving videos and dense rendering attributes. Real-world datasets [24], [31], [32] capture realistic visual appearances but lack G-buffer annotations, whereas simulators [6], [9] provide accurate Gbuffers but suffer from a large visual domain gap. To support sim-to-real video translation, we construct GenReal, which consists of two synthetic subsets, GenReal-City and GenReal-CARLA, and one real-world subset, GenReal-DV. Each annotated sample contains an RGB video, its corresponding basecolor, depth, normal, metallic, and roughness sequences, and a descriptive text prompt. Synthetic G-buffers are directly exported from the rendering pipelines, while real-world Gbuffers are estimated using the pretrained inverse-rendering model DiffusionRenderer [14]. Table I summarizes the dataset statistics.

a) Synthetic Data Construction.: GenReal-City is collected in the Unreal Engine 5 CitySample environment with the MatrixCity plugin [33]. We use the built-in traffic system to generate moving vehicles and record driving trajectories. These trajectories are then rendered with Movie Render Queue to obtain RGB frames and their corresponding Gbuffers. GenReal-CARLA is collected from multiple CARLA maps using the built-in autonomous driving agent. We record rendered RGB sequences together with the corresponding G-buffer sequences. The two synthetic sets differ in scene layouts, asset quality, and rendering styles, which increases the diversity of the synthetic data.

b) Real-World Data Construction.: GenReal-DV is constructed from real-world driving videos in the OpenDV dataset. We sample video clips from the source videos and use the pretrained DiffusionRenderer [14] to estimate basecolor, depth, normal, metallic, and roughness maps. We manually remove clips with severe motion blur, ghosting, extreme exposure, or lens occlusion, while retaining clips with mild real-world artifacts to preserve the diversity of natural driving scenes. Unlike the accurate G-buffers exported by simulators, the estimated attributes in GenReal-DV contain noise, making the dataset suitable for studying robust adaptation from

c) Text Prompt Construction.: Each video is paired with a text prompt generated through a hybrid human– VLM annotation pipeline. Human annotators label low-level visual attributes that are difficult for VLMs to estimate reliably, including weather, time of day, exposure, color style, camera configuration, sunlight direction, and imaging artifacts. Qwen3-VL [34] is used to extract high-level semantic information, such as scene type, object layout, road conditions, and background content.

## IV. METHOD

NeuralSRender generates photorealistic driving videos from G-buffers and text prompts, as illustrated in Fig. 2. A lightweight projector integrates geometric and material features into a pretrained video diffusion model, with the backbone remaining frozen during LoRA adaptation. Our twostage training strategy first learns conditioning from accurate synthetic G-buffers, then adapts to real-world appearance using real videos while keeping the learned conditioning components fixed.

## A. G-buffer Condition Fusion

G-buffers provide dense geometry and material information for rendered videos. Depth and normals represent scene geometry, while basecolor, metallicity, and roughness characterize material appearance. For a video of $T$ frames, let $g ^ { ( i ) } = \left\{ g _ { k } ^ { ( i ) } \right\} _ { k = 1 } ^ { T }$ denote the temporal sequence of the i-th G-buffer attribute, and let $\mathbf { g } = \{ g ^ { ( i ) } \} _ { i = 1 } ^ { n }$ denote the complete set of G-buffer attribute sequences, where $n = 5$ in our setting. All attribute sequences are spatially and temporally aligned with the corresponding RGB video.

Given the G-buffer conditions g and a text prompt c, our goal is to generate a photorealistic video that preserves the geometry and material constraints encoded by g, while following the global appearance specified by $c ,$ such as weather, illumination, and time of day.

To map the G-buffer conditions into the spatiotemporal latent space of the pretrained video diffusion model, we independently encode each attribute sequence using the frozen video VAE encoder E:

$$
z _ { g } ^ { ( i ) } = \mathcal { E } \left( g ^ { ( i ) } \right) , \qquad i = 1 , \dots , n .\tag{1}
$$

Here, $z _ { g } ^ { ( i ) }$ denotes the latent representation of the i-th G-buffer attribute sequence.

This maps all attributes to the same spatial and temporal resolution as the video latent while preserving their individual information. We then concatenate the encoded G-buffer features with the noisy video latent $z _ { t }$ along the channel

![](images/9dc1db15024b5fa6bf627a8716c9a24f828f5694da0c160d57f6722154d817f8.jpg)  
Fig. 2. Overview of NeuralSRender. In Stage I, the G-buffer projector and LoRA modules are trained on synthetic videos with accurate G-buffers. In Stage II, the model is adapted to real-world videos using estimated G-buffers. In Stage II, the projector is frozen to protect the conditioning mechanism learned from synthetic data, while only the LoRA modules are optimized.

dimension:

$$
z _ { \mathrm { c a t } } = \Big [ z _ { t } , z _ { g } ^ { ( 1 ) } , \cdot \cdot \cdot , z _ { g } ^ { ( n ) } \Big ] .\tag{2}
$$

A lightweight 3D convolutional projector $\mathcal { F } _ { \mathrm { p r o j } }$ is used to integrate these features:

$$
\tilde { z } _ { t } = \mathcal { F } _ { \mathrm { p r o j } } \left( z _ { \mathrm { c a t } } \right) .\tag{3}
$$

The projector captures spatial-temporal interactions and the relations among different G-buffer attributes, and maps the fused features back to the original latent channel dimension. The resulting latent $\tilde { z } _ { t }$ is finally passed to the DiT backbone together with the timestep t and text prompt c.

## B. Two-Stage Sim-to-Real Adaptation

Real-world videos provide realistic appearance, but their estimated G-buffers often contain noise and inverse-rendering artifacts. Directly training on these noisy pairs may cause the G-buffer projector to learn estimation errors instead of reliable relationships among geometric and material attributes. We therefore separate G-buffer relation learning from real-world appearance adaptation. We insert LoRA [35] modules into the attention layers of the DiT, while keeping the pretrained DiT weights $W _ { 0 }$ frozen throughout training. In Stage I, the G-buffer projector and the LoRA module, denoted by $\Delta W _ { \mathrm { s y n } } ,$ are jointly trained on synthetic videos with accurate G-buffers. This stage allows the projector to learn the relations among geometry and material attributes and enables the diffusion backbone to follow the fused conditions. In Stage II, we load the learned projector and $\Delta W _ { \mathrm { s y n } }$ into the DiT and keep both frozen. We then introduce an additional LoRA module, denoted by $\Delta W _ { \mathrm { r e a l } }$ , and only train the LoRA module on real-world videos with estimated G-buffers. In this way, the model adapts to real-world lighting, textures, materials, and other appearance statistics without disturbing the G-buffer relations learned in Stage I.

For a clean video latent $z _ { 0 } ,$ , Flow Matching constructs

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon , \qquad u _ { t } = \frac { \mathrm { d } z _ { t } } { \mathrm { d } t } = \epsilon - z _ { 0 } ,\tag{4}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $t \sim \mathcal { U } ( 0 , 1 )$

a) Stage I: G-buffer alignment via Synthetic Data.: In the first stage, our primary goal is to enable the model to comprehend the underlying physical meanings of the G-buffers and learn how to accurately map these intrinsic physical priors into the visual generation process. Let $\mathcal { D } _ { \mathrm { s y n } }$ denote the high-quality synthetic dataset comprising tuples $( x ^ { \mathrm { { s y n } } } , g ^ { \mathrm { { s y n } } } , c ^ { \mathrm { { s y n } } } )$ of rendered videos, perfectly aligned Gbuffers, and text prompts. During this stage, we jointly optimize the parameters of the G-buffer projector, denoted by $\theta _ { \mathrm { p r o j } }$ , and the Stage-I LoRA modules $\Delta W _ { \mathrm { s y n } }$ . With $W _ { \mathrm { I } } = W _ { \mathrm { 0 } } + \Delta W _ { \mathrm { s y n } } ,$ , the training objective is

$$
\mathcal { L } _ { \mathrm { s y n } } = \mathbb { E } \big [ \| u _ { t } ^ { \mathrm { s y n } } - v _ { \theta } ( \tilde { z } _ { t } ^ { \mathrm { s y n } } , t , c ^ { \mathrm { s y n } } ; W _ { \mathrm { I } } ) \| _ { 2 } ^ { 2 } \big ] ,\tag{5}
$$

where $z _ { 0 } ^ { \mathrm { s y n } }$ is obtained by encoding $x ^ { \mathrm { s y n } }$ with the pretrained ${ \mathrm { V A E } } , z _ { t } ^ { \mathrm { s y n } }$ and $u _ { t } ^ { \mathrm { s y n } }$ follow $\mathrm { E q . } ( 4 ) , \tilde { z } _ { t } ^ { \mathrm { s y n } }$ fuses the noisy video latent with the projected G-buffer features, and $v _ { \theta }$ predicts the target flow velocity. By optimizing this objective, the model learns the physical and spatial correspondence between the G-buffer features and the generated visual content.

b) Stage II: Style Adaptation via Real-World Data.: While Stage I successfully aligns the generative process with the underlying geometry and material constraints, the resulting illumination and material appearance inherently retain the “simulated” style. To achieve sim-to-real translation, we perform a second training stage using a real-world dataset, denoted as $\mathcal { D } _ { \mathrm { r e a l } } .$ , which contains real videos, their corresponding inverse-rendered G-buffers, and text prompts $( x ^ { \mathrm { r e a l } } , \dot { g } ^ { \mathrm { i n v } } , c ^ { \mathrm { r e a l } } )$ . Prior to training, we explicitly initialize the framework with the learned weights $\theta _ { \mathrm { p r o j } } ^ { * }$ and $\Delta W _ { \mathrm { s y r } } ^ { * }$ acquired in Stage I. Crucially, to prevent the noisy artifacts of inverse-rendered G-buffers from degrading the model, the projector weights are kept frozen. With $W _ { \mathrm { I I } } = W _ { 0 } + \Delta W _ { \mathrm { s y n } } ^ { \ast } +$ $\Delta W _ { \mathrm { r e a l } }$ , we optimize the style-adaptation objective strictly with respect to the new Stage-II LoRA modules $\Delta W _ { \mathrm { r e a l } } { : }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e a l } } = \mathbb { E } \big [ \| u _ { t } ^ { \mathrm { r e a l } } - v _ { \theta } \big ( \tilde { z } _ { t } ^ { \mathrm { r e a l } } , t , c ^ { \mathrm { r e a l } } ; W _ { \mathrm { I I } } \big ) \| _ { 2 } ^ { 2 } \big ] , } \end{array}\tag{6}
$$

where $z _ { 0 } ^ { \mathrm { r e a l } }$ is obtained by encoding $x ^ { \mathrm { r e a l } }$ with the pretrained VAE, $z _ { t } ^ { \mathrm { r e a l } }$ and $u _ { t } ^ { \mathrm { r e a l } }$ follow Eq. (4), and $\tilde { z } _ { t } ^ { \mathrm { r e a l } }$ fuses the noisy video latent with the projected inverse-rendered $\mathbf { G } \mathrm { - }$ buffer features. During Stage II, W<sub>0</sub>, $\theta _ { \mathrm { p r o j } } ^ { * }$ , and $\Delta W _ { \mathrm { s y n } } ^ { * }$ are kept frozen, and only $\Delta W _ { \mathrm { r e a l } }$ is optimized. Guided by the text prompt $c ,$ this stage efficiently shifts the model’s

(A) GenReal-City

![](images/d97f6bce9309e29bfb5a60841d7404e2b8f064858c1fadde2c8c5189e4a155b3.jpg)  
(B) GenReal-CARLA  
(a) Render  
(b) DiffusionRender  
(c) TC-Light  
(d) Cosmos  
(e) Ours

Fig. 3. Qualitative results on the synthetic test sets (GenReal-City and GenReal-CARLA). Unlike baselines that retain artificial simulator aesthetics, our approach leverages the explicit geometry and physical priors of G-buffers, alongside text prompts, to achieve photorealistic sim-to-real synthesis while maintaining structural integrity.  
TABLE II: QUANTITATIVE COMPARISON ON GENREAL-CITY AND GENREAL-CARLA.
<table><tr><td rowspan="2">Method</td><td colspan="4">GenReal-City</td><td colspan="4">GenReal-CARLA</td><td colspan="2">Efficiency</td></tr><tr><td>FVD↓</td><td>CLIP-T↑</td><td>NDE↓</td><td>VLM-BT↑</td><td>FVD↓</td><td>CLIP-T↑</td><td>NDE↓</td><td>VLM-BT↑</td><td>VRAM (GB)↓</td><td>FPS↑</td></tr><tr><td>DiffRender [14]</td><td>1116.45</td><td>0.2836</td><td>0.0260</td><td>-0.1960</td><td>1113.35</td><td>0.2819</td><td>0.0302</td><td>-0.2180</td><td>94.55</td><td>0.348</td></tr><tr><td>TC-Light [15]</td><td>1117.74</td><td>0.2953</td><td>0.0238</td><td>-0.0752</td><td>984.37</td><td>0.2878</td><td>0.0316</td><td>-0.1737</td><td>19.82</td><td>0.288</td></tr><tr><td>Cosmos [11]</td><td>997.26</td><td>0.3006</td><td>0.0251</td><td>-0.0070</td><td>981.75</td><td>0.2951</td><td>0.0335</td><td>0.1571</td><td>57.74</td><td>0.142</td></tr><tr><td>Ours</td><td>979.69</td><td>0.3007</td><td>0.0243</td><td>0.2783</td><td>599.71</td><td>0.2949</td><td>0.0324</td><td>0.2346</td><td>13.67</td><td>0.598</td></tr></table>

output distribution, adapting lighting, texture appearance, and complex optical effects to real-world statistics, all while preserving the robust physical rendering logic locked in by the frozen projector $\theta _ { \mathrm { p r o j } } ^ { * }$

## V. EXPERIMENTS

## A. Experimental Setup

a) Implementation Details.: We build NeuralSRender on the pretrained Wan2.2-TI2V-5B model [36]. We use AdamW with a weight decay of $1 \times 1 0 ^ { - 4 }$ and set the LoRA rank to 32 in both stages. The pretrained VAE and DiT weights remain frozen throughout training. In Stage I, we jointly train the G-buffer projector $\mathcal { F } _ { \mathrm { p r o j } }$ and the Stage-I LoRA parameters $\Delta W _ { \mathrm { s y n } }$ on GenReal-City for 6 epochs. In Stage II, we freeze $\mathcal { F } _ { \mathrm { p r o j } }$ and train a newly initialized set of LoRA parameters $\Delta W _ { \mathrm { r e a l } }$ on GenReal-DV for 2 epochs. The batch size is 4. The whole training takes approximately five days on four RTX 4090 GPUs.

(a) Render  
(b) Basecolor  
![](images/e8795f5c82a68e727a05f1a9cac8af2d07472e1e94415e4d3363fdecd650417a.jpg)  
(c) Depth  
(d) Edge  
(e) Two-Stage w/o Freeze  
(f) Single-Stage  
(g) Full  
Fig. 4. Qualitative comparison of control signals and training strategies. Depth and edge controls produce overly smooth results with semantic, texture, color, and lighting errors. Basecolor control provides richer appearance cues but may flatten geometric structures. Two-Stage w/o Freeze degrades learned lighting and shadow patterns, while Single-Stage training may distort geometry and road markings. Our full model best preserves geometry, markings, textures, lighting, and shadows.

b) Metrics.: For generation quality, we compute FVD against the GenReal-DV reference set, use LongCLIP [37] for text–video alignment (CLIP-T), and report VLM-BT for frame-level photorealism. For VLM-BT, Gemini 3 Flash, GPT-5.4 mini, and Qwen3.6-Plus rank randomly shuffled outputs, and their pairwise preferences are aggregated using a Bradley–Terry model. Since the Bradley–Terry model is fitted independently for each comparison set, VLM-BT scores reflect relative preferences within that set and may vary for the same method when the compared methods change. For geometric consistency, we estimate relative depth from generated and source videos using Video Depth Anything [38], and report the mean absolute normalized difference as NDE. For efficiency, we measure peak VRAM and FPS at a resolution of 1280 × 704 × 81 on a single NVIDIA RTX Pro 6000 GPU.

c) Baselines.: We compare NeuralSRender with DiffusionRenderer [14], TC-Light [15], and Cosmos-Transfer 2.5 (Cosmos) [11]. For DiffusionRenderer, we use the same G-buffers as our method and estimate environment illumination maps with DiffusionLight-Turbo [39], following its official relighting pipeline. For TC-Light, we use the official pretrained weights and default settings. For Cosmos, we use the edge-controlled variant with edge maps extracted from the source videos. All methods use the same test clips and output resolution. Identical prompts are provided to all text-conditioned methods, while DiffusionRenderer is evaluated without text conditioning.

## B. Comparison Results

Table II reports the quantitative results on GenReal-City and GenReal-CARLA. Ours achieves the best FVD and VLM-BT scores on both test sets, demonstrating improved distributional similarity and perceived photorealism. Meanwhile, it maintains competitive CLIP-T scores and NDE. Overall, our method improves visual realism without substantially sacrificing text alignment or geometric consistency. Qualitative comparisons in Fig. 3 further show that NeuralSRender achieves a favorable balance between visual realism and structural consistency. DiffusionRenderer [14] closely follows the geometry and material cues provided by the G-buffers. However, its outputs often retain the artificial appearance of simulator renderings, with limited improvements in textures and illumination. TC-Light [15] takes the rendered RGB video as input and mainly adjusts global color tone. As shown in Fig. 3(c), the original textures and cast shadows often remain unchanged, which can conflict with the modified appearance. Cosmos [11] produces visually appealing results using strong generative priors. However, edge-based conditioning provides only sparse structural guidance. Edges extracted from rendered videos may also contain shadow boundaries and noisy details, leading to geometric deviations, artificial lens flares, or spurious objects, as illustrated in Fig. 3(d). Edge maps contain no explicit material information. This limitation often results in overly smooth surfaces, unrealistic reflections, and limited texture detail, making the outputs less photorealistic. In contrast, NeuralSRender employs dense G-buffers to provide geometry and material guidance, while text prompts control global attributes such as weather and illumination. This combination produces realistic reflections and rough-surface appearances while preserving scene structure and reducing artifacts, as shown in Fig. 3(e).

TABLE III: QUANTITATIVE RESULTS OF THE ABLATION STUDY.
<table><tr><td>Method</td><td>FVD↓</td><td>CLIP-T↑</td><td>NDE↓</td><td>VLM-BT↑</td></tr><tr><td>(a) Basecolor Only</td><td>669.41</td><td>0.2925</td><td>0.0410</td><td>0.0557</td></tr><tr><td>(b) Depth Only</td><td>812.12</td><td>0.2968</td><td>0.0602</td><td>-0.1499</td></tr><tr><td>(c) Edge Only</td><td>775.24</td><td>0.2943</td><td>0.0401</td><td>-0.0986</td></tr><tr><td>(d) Single-Stage</td><td>653.45</td><td>0.2945</td><td>0.0347</td><td>0.1088</td></tr><tr><td>(e) Two-Stage w/o Freeze</td><td>730.49</td><td>0.2957</td><td>0.0387</td><td>-0.0328</td></tr><tr><td>(f) Full Model</td><td>599.71</td><td>0.2949</td><td>0.0324</td><td>0.1168</td></tr></table>

## C. Ablation Studies

We ablate different control signals and training strategies, including Basecolor Control, Depth Control, Edge Control, Two-Stage w/o Freeze, and Single-Stage Training. The first three variants use only the corresponding control signal. Two-Stage w/o Freeze retains the two-stage synthetic-to-real training procedure but keeps the G-buffer projector trainable during real-world adaptation. Single-Stage Training mixes synthetic and real-world samples and jointly trains the model on both domains in a single stage. All experiments use the same data and number of training steps.

![](images/4ac1a793940d956901943f54b681997aa711bebcb123d4e37267afb3099fd845.jpg)  
(a) Input  
(b) Render  
(c) Basecolor  
(d) Depth  
(e) Edge  
(f) Ours  
(g) GT  
Fig. 5. Qualitative perception results with different synthetic training data.

As shown in Fig. 4 and Table III, Depth Control has difficulty representing both near and distant regions in outdoor scenes. It also cannot constrain road markings or scene colors. Edge Control provides structural guidance, but it contains no color or texture information. As a result, both controls provide incomplete geometric and appearance constraints, often leading to semantic errors, overly smooth surfaces, limited texture detail, and unrealistic lighting or shadows. Basecolor Control provides richer appearance information but offers limited cues for distinguishing geometry and material properties. Geometrically distinct regions may share similar basecolors, causing the model to merge them into a flat surface. For example, the curb in the orange box of Fig. 4 loses its clear geometric structure, reducing geometric accuracy. In contrast, full G-buffer conditioning combines complementary geometry and material cues, enabling better structural preservation and more realistic appearance.

For the training strategies under full G-buffer conditioning, Two-Stage w/o Freeze is strongly affected by the noisy realdata G-buffers in the second training stage. Although it can preserve the overall scene geometry, the previously learned relations are partially destroyed, leading to a loss of realistic lighting and shadow patterns. Single-stage training produces results closest to those of the full model. Nevertheless, the noisy real data can still interfere with the geometry and semantic structures learned from synthetic data. For example, the rear of a car may be incorrectly generated as its front, and ground markings may be distorted (orange box in Fig. 4). Finally, our full model freezes the G-buffer projector during real-world adaptation, preserving the learned attribute relations while LoRA adapts the model to real-world appearance. It achieves the best FVD, NDE, and VLM-BT scores, with competitive CLIP-T scores and strong geometric and visual quality.

## D. Sim-to-Real Transfer for Autonomous Driving

We evaluate sim-to-real transfer for object detection and semantic segmentation without target-domain fine-tuning. For each setting, we train perception models on either raw rendered videos or videos generated with different control signals from the same CARLA scenes. The trained models are then directly evaluated on unseen real-world datasets without target-domain fine-tuning. For object detection, we use YOLOv8 [40] and evaluate on nuImages [31]. As reported in Table IV, our method achieves the highest detection AP and $\mathrm { { A P } _ { 5 0 } }$ among the evaluated settings, increasing $\mathrm { { A P } _ { 5 0 } }$ from 35.1 to 40.6 compared with training on the original CARLA renders. For semantic segmentation, we use SegFormer [41] and evaluate on Cityscapes [42]. Our method again performs best, improving mIoU from 32.4 to 40.8 over the CARLA renders. Fig. 5 complements these quantitative results with qualitative comparisons: models trained on our translated data detect more of the annotated vehicles and produce a more coherent road segmentation than those trained on the original CARLA renders in the illustrated examples. These gains demonstrate the effectiveness of the translated data for training perception models that generalize to real-world driving scenes.

TABLE IV: PERCEPTION PERFORMANCE WITH DIFFERENT SYNTHETIC TRAINING DATA.
<table><tr><td>Method</td><td>AP↑</td><td> $\mathrm { { A P } 5 0 \uparrow }$ </td><td>mIoU↑</td><td>mAcc↑</td></tr><tr><td>(a) Render</td><td>16.7</td><td>35.1</td><td>32.4</td><td>42.3</td></tr><tr><td>(b) Basecolor Only</td><td>18.8</td><td>38.1</td><td>37.6</td><td>45.6</td></tr><tr><td>(c) Depth Only</td><td>14.7</td><td>30.9</td><td>37.4</td><td>47.3</td></tr><tr><td>(d) Edge Only</td><td>17.1</td><td>37.4</td><td>35.2</td><td>45.9</td></tr><tr><td>(e) Ours</td><td>19.4</td><td>40.6</td><td>40.8</td><td>50.2</td></tr></table>

## VI. CONCLUSION

We present NeuralSRender, a parameter-efficient framework for G-buffer-guided sim-to-real video translation. Its two-stage training separates reliable conditioning learning from real-world appearance adaptation. We also introduce GenReal, a driving video dataset containing synthetic and real-world videos with corresponding G-buffers. Experiments demonstrate improved visual realism with competitive geometric consistency. Overall, our work enables the generation of high-quality, scalable video data for autonomous driving models, highlighting the broader potential of sim-to-real video translation for autonomous driving.

## REFERENCES

[1] R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu, “Magicdrive: Street view generation with diverse 3d geometry control,” in International Conference on Learning Representations (ICLR), 2024.

[2] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al., “Planning-oriented autonomous driving,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 17 853–17 862.

[3] J.-J. Hwang, R. Xu, H. Lin, W.-C. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo, D. Anguelov, and M. Tan, “EMMA: End-to-end multimodal model for autonomous driving,” Transactions on Machine Learning Research, 2025. [Online]. Available: https://openreview.net/forum?id=kH3t5lmOU8

[4] R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu, “MagicDrive-V2: High-resolution long video generation for autonomous driving with adaptive control,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[5] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, et al., “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 12 037–12 047.

[6] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An open urban driving simulator,” in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.

[7] A. Gaidon, Q. Wang, Y. Cabon, and E. Vig, “Virtual worlds as proxy for multi-object tracking analysis,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2016, pp. 4340–4349.

[8] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” 2020.

[9] S. Shah, D. Dey, C. Lovett, and A. Kapoor, “Airsim: High-fidelity visual and physical simulation for autonomous vehicles,” in Field and service robotics: Results of the 11th international conference. Springer, 2017, pp. 621–635.

[10] A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J.-M. Allen, V.-D. Lam, A. Bewley, and A. Shah, “Learning to drive in a day,” in 2019 international conference on robotics and automation (ICRA). IEEE, 2019, pp. 8248–8254.

[11] A. Chang and A. Santhosh, “Cosmos transfer 2.5 control modalities: Core concepts,” https://nvidia-cosmos.github.io/cosmos-cookbook/core concepts/control modalities/overview.html, November 2025, nVIDIA Cosmos Cookbook.

[12] Z. Gu, R. Yan, J. Lu, P. Li, Z. Dou, C. Si, Z. Dong, Q. Liu, C. Lin, Z. Liu, W. Wang, and Y. Liu, “Diffusion as shader: 3d-aware video diffusion for versatile video generation control,” arXiv preprint arXiv:2501.03847, 2025.

[13] X. Ren, Y. Lu, T. Cao, R. Gao, S. Huang, A. Sabour, T. Shen, T. Pfaff, J. Z. Wu, R. Chen, et al., “Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models,” arXiv preprint arXiv:2506.09042, 2025.

[14] R. Liang, Z. Gojcic, H. Ling, J. Munkberg, J. Hasselgren, Z.-H. Lin, J. Gao, A. Keller, N. Vijaykumar, S. Fidler, and Z. Wang, “Diffusionrenderer: Neural inverse and forward rendering with video diffusion models,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2025.

[15] Y. Liu, C. Luo, Z. Tang, Y. Li, Y. Ning, L. Fan, J. Peng, Z. Zhang, et al., “Tc-light: Temporally coherent generative rendering for realistic world transfer,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[16] K. He, R. Liang, J. Munkberg, J. Hasselgren, N. Vijaykumar, A. Keller, S. Fidler, I. Gilitschenski, Z. Gojcic, and Z. Wang, “Unirelight: Learning joint decomposition and synthesis for video relighting,” arXiv preprint arXiv:2506.15673, 2025.

[17] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 3836– 3847.

[18] C. Mou, X. Wang, L. Xie, et al., “T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp. 4296–4304.

[19] X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Videocomposer: Compositional video synthesis with motion controllability,” Advances in Neural Information Processing Systems, vol. 36, pp. 7594–7611, 2023.

[20] Y. Guo, C. Yang, A. Rao, M. Agrawala, D. Lin, and B. Dai, “Sparsectrl: Adding sparse controls to text-to-video diffusion models,” in European Conference on Computer Vision. Springer, 2024, pp. 330–348.

[21] H. Lin, J. Cho, A. Zala, and M. Bansal, “Ctrl-adapter: An efficient and versatile framework for adapting diverse controls to any diffusion model,” arXiv preprint arXiv:2404.09967, 2024.

[22] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “Vace: Allin-one video creation and editing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 191–17 202.

[23] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 83 048–83 077.

[24] J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo, J. Zhang, A. Geiger, Y. Qiao, and H. Li, “Generalized predictive model for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[25] S. James, P. Wohlhart, M. Kalakrishnan, D. Kalashnikov, A. Irpan, J. Ibarz, S. Levine, R. Hadsell, and K. Bousmalis, “Sim-to-real via sim-to-sim: Data-efficient robotic grasping via randomized-to-canonical adaptation networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 627–12 637.

[26] C. Samak, T. Samak, B. Li, and V. Krovi, “Sim2real diffusion: Leveraging foundation vision language models for adaptive automated driving,” IEEE Robotics and Automation Letters, vol. 11, no. 1, pp. 177–184, 2025.

[27] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, G. Liu, A. Tao, J. Kautz, and B. Catanzaro, “Video-to-video synthesis,” arXiv preprint arXiv:1808.06601, 2018.

[28] L. Zhang, A. Rao, and M. Agrawala, “Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=u1cQYxRI1H

[29] H. Jin, Y. Li, F. Luan, Y. Xiangli, S. Bi, K. Zhang, Z. Xu, J. Sun, and N. Snavely, “Neural gaffer: Relighting any object via diffusion,” in Advances in Neural Information Processing Systems, 2024.

[30] Y. Zhou, J. Bu, P. Ling, P. Zhang, T. Wu, Q. Huang, J. Li, X. Dong, Y. Zang, Y. Cao, A. Rao, J. Wang, and L. Niu, “Light-a-video: Trainingfree video relighting via progressive light fusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 13 315–13 325.

[31] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” arXiv preprint arXiv:1903.11027, 2019.

[32] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446–2454.

[33] Y. Li, L. Jiang, L. Xu, Y. Xiangli, Z. Wang, D. Lin, and B. Dai, “Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3205–3215.

[34] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[35] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

[36] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[37] B. Zhang, P. Zhang, X. Dong, Y. Zang, and J. Wang, “Longclip: Unlocking the long-text capability of clip,” arXiv preprint arXiv:2403.15378, 2024.

[38] S. Chen, H. Guo, S. Zhu, F. Zhang, Z. Huang, J. Feng, and B. Kang, “Video depth anything: Consistent depth estimation for super-long videos,” arXiv:2501.12375, 2025.

[39] W. Chinchuthakun, P. Phongthawee, A. Raj, V. Jampani, P. Khungurn, and S. Suwajanakorn, “Diffusionlight-turbo: Accelerated light probes for free via single-pass chrome ball inpainting,” in ArXiv, 2025.

[40] R. Varghese and M. Sambath, “Yolov8: A novel object detection algorithm with enhanced performance and robustness,” in 2024 International conference on advances in data engineering and intelligent computing systems (ADICS). IEEE, 2024, pp. 1–6.

[41] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.

[42] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 3213–3223.