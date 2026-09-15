![](images/156fca24ade9bb01885a390548387b2e596a25f6d68d49220e27ce5240977345.jpg)  
Fig. 1. Given an object mesh, SAM3D-Part enables interactive part selection on a rendered view and generates complete 3D part meshes. It supports fine-grained, controllable selection (a) and full object decomposition (b) from arbitrary viewpoints.

# SAM3D-Part: Interactive Part Selection and Generation from 3D Objects

JIAHAO CHANG<sup>∗†</sup>, SSE, CUHKSZ and FNii-Shenzhen and Meshy AI, China   
DONG DU<sup>†</sup>, Nanjing University of Science and Technology, China   
WANHU SUN, SSE, CUHKSZ, China   
YUJIAN ZHENG, MBZUAI, United Arab Emirates   
CHUANYU PAN<sup>‡</sup>, Meshy AI, USA   
BOWEN ZHAO, Meshy AI, USA   
CHONGJIE YE, SSE, CUHKSZ and FNii-Shenzhen, China   
YUANMING HU, Meshy AI, USA   
XIAOGUANG HAN<sup>§</sup>, SSE, CUHKSZ and FNii-Shenzhen and GenuX, China <sup>∗</sup>Work done during internship at Meshy AI.   
<sup>†</sup>These authors contributed equally to this work.   
<sup>‡</sup>Project leader.   
<sup>§</sup>Corresponding authors.

Authors’ Contact Information: Jiahao Chang, SSE, CUHKSZ and FNii-Shenzhen and Meshy AI, China, 224010128@link.cuhk.edu.cn; Dong Du, Nanjing University of Science and Technology, China, dongdu@njust.edu.cn; Wanhu Sun, SSE, CUHKSZ, China, wanhusun@gmail.com; Yujian Zheng, MBZUAI, United Arab Emirates, yujian.zheng@ mbzuai.ac.ae; Chuanyu Pan, Meshy AI, USA, chuanyupan@meshy.ai; Bowen Zhao, Meshy AI, USA, bowen@meshy.ai; Chongjie Ye, SSE, CUHKSZ and FNii-Shenzhen, China, chongjieye@link.cuhk.edu.cn; Yuanming Hu, Meshy AI, USA, yuanmhu@ gmail.com; Xiaoguang Han, SSE, CUHKSZ and FNii-Shenzhen and GenuX, China, hanxiaoguang@cuhk.edu.cn.

Part-level control is essential for modern 3D asset creation, where objects are frequently edited, reused, animated, or fabricated through their individual components. In many such workflows, users need only several specific components rather than a complete object decomposition. However, existing 3D generation methods produce all parts regardless of user intent, while promptable 3D segmentation methods typically output partial surfaces instead of reusable complete meshes. In addition, image-conditioned part generators further struggle to preserve hidden geometry and accurate placement without directly conditioning on the source mesh. To address these problems, we present SAM3D-Part, a prompt-driven framework for selective part generation from input 3D object meshes. Given a source mesh and a part prompt, SAM3D-Part first encodes the source geometry into compact mesh features and aligns them with the rendered image, selective mask, and point-map observations via pixel-wise channel fusion. The fused representation conditions a feed-forward generative model to produce only the queried component as a completed mesh. To place the generated part back into the source coordinate frame, SAM3D-Part predicts dense per-voxel correspondences and estimates the part transformation from distributed spatial evidence rather than a single global pose code. For sequential multi-part queries, previously generated parts are stored in a part cache and reused as contextual constraints, reducing conflicts among independently requested components. Extensive experiments and ablations demonstrate that SAM3D-Part can significantly improve source alignment, reduce conditioning cost, and enable consistent selective part generation, achieving state-of-the-art. Code and weights will be available at https://github.com/Jiahao620/sam3d-part.

CCS Concepts: • Computing methodologies → Computer graphics.

Additional Key Words and Phrases: 3D Part Generation, Interactive 3D Modeling, Multimodal Conditioning

## ACM Reference Format:

Jiahao Chang, Dong Du, Wanhu Sun, Yujian Zheng, Chuanyu Pan, Bowen Zhao, Chongjie Ye, Yuanming Hu, and Xiaoguang Han. 2026. SAM3D-Part: Interactive Part Selection and Generation from 3D Objects. In SIGGRAPH Asia 2026 Conference Papers (SA Conference Papers ’26), December 01–04, 2026, Kuala Lumpur, Malaysia. ACM, New York, NY, USA, 12 pages. https: //doi.org/10.1145/3829340.3842187

## 1 Introduction

With the rapid development of 3D content creation, part-aware control is becoming a central requirement in various applications. Downstream workflows such as rigging, animation, part replacement, kitbashing, and 3D printing often require individual components to be isolated, edited, or reused independently from the full object. In these workflows, users rarely need an exhaustive decom position of the entire asset. For instance, a modeler may want to extract only a dragon’s wing, replace only a chair leg, or print only a mechanical connector. This selective interaction mode is already familiar in 2D image editing, where promptable segmentation enables users to specify exactly the region they care about. For 3D asset creation, we seek an analogous design that takes an existing object mesh and a user-specified part prompt as input, then returns a reusable part mesh geometrically aligned with the input object.

However, existing part-level 3D generation methods [Chen et al. 2025b; He et al. 2025; Liu et al. 2024; Yang et al. 2025b; Zhang et al. 2025] mainly focus on fully automatic object decomposition, aiming to recover all constituent parts of an object. When users need only one or two components, generating all parts is computationally ineficient and still requires additional selection from the resulting decomposition. More importantly, these methods cannot ensure precise control over partition granularity. Although promptable 3D segmentation methods [Li et al. 2026a; Ma et al. 2025; Yang et al. 2024] provide a more selective interface, they typically output incomplete meshes using surface masks rather than complete part meshes suitable for downstream reuse. Recent image-to-part generation methods, especially SAM3D [Chen et al. 2025a], are closer to selective part generation, but they condition on rendered views instead of the input mesh. This limits their ability to produce sourcealigned part meshes, especially for parts with occluded structures, ambiguous boundaries, or strong dependence on the surrounding geometry.

In this paper, we address prompt-driven selective part extraction from existing object meshes and introduce an efective framework, named SAM3D-Part. Given an input mesh, we first select a suitable viewpoint, such as the front view, and render the mesh into a 2D image. The user then specifies the desired component through simple point interactions, using one or a few clicks on the rendered view. Based on these point prompts, we apply a SAM-based segmentor [Ravi et al. 2025] to obtain the corresponding 2D part mask, which serves as an intuitive prompt for subsequent 3D part generation. Unlike surface segmentation methods that output only labels or masks, SAM3D-Part formulates the task as conditional 3D generation and directly produces a complete part mesh.

To ensure that the generated part remains consistent with the source object, SAM3D-Part conditions the generation process on both image-space prompts and mesh-space geometry. Specifically, the rendered RGB image, the selected part mask, and the point-map observation are encoded as 2D prompt tokens, where appearance, user attention, and spatial information are fused at the same imagepatch locations. In parallel, the input mesh is encoded into 3D geometry tokens using a pretrained shape encoder. These 2D and 3D condition tokens are then combined along the token dimension and injected into the generative model through cross-attention. This design provides the generator with both the user’s part-level intent and the complete geometric context of the original mesh, including regions that may be invisible from the selected view.

Built upon a two-stage sparse 3D generation framework, SAM3D-Part first predicts the coarse occupancy of the queried part and a dense XYZ correspondence map from the generated part to the source mesh frame. Instead of regressing a single global pose, the dense correspondence provides voxel-level spatial evidence, from which the scale and translation of the generated part can be estimated in closed form. The second stage then refines the predicted structure into high-resolution geometry and decodes it into a standalone mesh. For sequential multi-part queries, we further maintain a part cache that records previously generated components and attaches this history to the mesh tokens, helping later queries avoid conflicts and maintain consistency across extracted parts.

We evaluate SAM3D-Part on multiple object categories under both single-part and multi-part settings. Extensive experiments and ablation studies are conducted to assess part quality, source alignment, and consistency across sequential queries. Experimental results show that SAM3D-Part consistently outperforms existing baselines in selective part generation, achieves better alignment with the input object, and maintains stable performance when generating multiple parts sequentially.

Our contributions are summarized as follows:

• We formulate prompt-driven selective part extraction from existing 3D meshes and propose a novel framework, i.e., SAM3D-Part, aiming to generate reusable and source-aligned meshes for only the user-specified components.

• Our SAM3D-Part combines SAM3D-derived image prompts with input mesh features through an eficient pixel-aligned multimodal fusion paradigm. It further incorporates dense correspondencebased alignment and a part-cache mechanism to recover accurate part placement and improve consistency across sequential multipart queries.

• Extensive experiments and ablations demonstrate that SAM3D-Part achieves better part quality, source alignment, and sequential consistency than existing methods.

## 2 Related Work

## 2.1 Object-level 3D Generation

DreamFusion [Poole et al. 2022] pioneers text-to-3D generation by optimizing a 3D representation with 2D difusion priors via score distillation sampling. Subsequent methods improve optimization eficiency, geometry quality, and multi-view consistency [Chen et al. 2023; Lin et al. 2023; Shi et al. 2023; Tang et al. 2023], but still sufer from slow per-instance optimization and 3D inconsistency. Another line synthesizes multi-view images and reconstructs 3D assets from them [Tang et al. 2024; Xu et al. 2024], while single-pass feed-forward regressors enable faster image-to-3D generation at the cost of geometric fidelity [Hong et al. 2024; Szymanowicz et al. 2024]. Recently, native 3D generative models have been trained directly on 3D data with various representations [Jun and Nichol 2023; Nichol et al. 2022]. VecSet-based latent difusion [Zhang et al. 2023] has become a strong paradigm for scalable mesh generation, as shown by CLAY [Zhang et al. 2024], TripoSG [Li et al. 2025], and Hunyuan3D-2.1 [Team 2025]. Meanwhile, structured 3D latent models (e.g., TRELLIS [Xiang et al. 2025], Hi3DGen [Ye et al. 2025], and Sparc3D [Li et al. 2026b]) further separate coarse structure prediction from high-resolution latent generation. Despite their strong ability to synthesize complete objects, these methods mainly focus on object-level generation, leaving selective extraction of userspecified components from existing meshes largely unexplored.

## 2.2 Part-level 3D Generation

Part-level generation aims to model objects as compositions of semantically or geometrically meaningful components. Some methods generate multiple parts jointly, often by predicting part layouts, bounding boxes, or structured latent variables before synthesizing part geometry [Chen et al. 2025b; He et al. 2025; Liu et al. 2024; Yang et al. 2025b; Zhang et al. 2025]. Other approaches formulate part generation as an automatic decomposition or reconstruction problem, where all parts of an object are produced in a single forward process or through sequential generation [Chen et al. 2026; Lin et al. 2026; Tang et al. 2026; Yan et al. 2024, 2025; Yang et al. 2025a]. These methods are valuable for part-aware object synthesis and structured shape modeling, but they usually assume that the goal is to recover or generate the full set of object parts. Image-conditioned methods (e.g., SAM3D [Chen et al. 2025a]) are closer to selective part generation, as they can focus on a visible region specified in an input view. However, because their conditions are mainly derived from rendered images, they may not fully preserve invisible geometry, volumetric occupancy, or the precise spatial relationship between the target part and the original mesh. In contrast, our work enables prompt-driven extraction of only the queried part from an existing mesh, producing a complete component aligned with the given object shape.

## 2.3 3D Part Segmentation

3D part segmentation has long been studied as a fundamental prob lem for understanding object structure. Early learning-based methods operate on point clouds, typically requiring dense part annotations for supervised training [Qi et al. 2017a,b; Zhao et al. 2021]. Although efective in specific domains, their generalization is limited by the scale and coverage of annotated 3D datasets. Recent methods leverage powerful 2D foundation models (e.g., SAM [Kirillov et al. 2023]) to improve open-world 3D segmentation. A common strategy is to render a 3D object into multiple views, apply 2D seg mentation or promptable segmentation in image space, and then lift the predicted masks back to the 3D surface [Abdelreheem et al. 2023; Cen et al. 2023; Liu et al. 2025, 2023b; Yang et al. 2024, 2023]. Meanwhile, promptable 3D segmentation methods extend the SAM paradigm to 3D, using user prompts such as points or boxes as 3D anchors to segment target parts [Li et al. 2026a; Ma et al. 2025]. These methods greatly enhance flexibility and reduce the need for category-specific annotations, but their outputs are typically surface labels or masks rather than complete part meshes. As a result, they cannot directly provide complete components suitable for downstream editing, simulation, fabrication, or asset reuse. In contrast, our work uses SAM-derived part images as prompts to guide 3D part generation, achieving reusable and source-aligned part meshes.

## 3 Methodology

Overview. Given an object mesh M and a user-specified 2D part mask � on one rendered view of M, our goal is to generate the corresponding 3D part mesh P. The generated part should be com plete and aligned with the coordinate frame of the original mesh. We formulate this task as conditional 3D generation rather than surface segmentation. As demonstrated in Figure 2, our proposed SAM3D-Part takes both image-space prompts and mesh-space geometry as conditions, and directly generates the queried component as a standalone mesh. Built upon a two-stage sparse 3D generation framework, our method first predicts the coarse occupancy and source-frame correspondence of the target part, then refines it to achieve high-resolution geometry. This design allows the model to use simple user interactions while preserving the geometric context of the original mesh. A maintained cache volume of previously segmented parts enables consistent sequential part generation.

## 3.1 Preliminaries

Flow Matching. Flow matching [Liu et al. 2023a] learns a continuous transformation from a simple source distribution to the data distribution. Let $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ be a Gaussian noise sample and $x _ { 1 }$ be a data sample. In rectified flow, an intermediate sample at time � ∈ [0, 1] is defined as

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } .\tag{1}
$$

The target velocity field is therefore

$$
v ^ { * } = x _ { 1 } - x _ { 0 } .\tag{2}
$$

A neural network $v _ { \theta } ( x _ { t } , t , c )$ is trained to predict this velocity under condition $c ,$ using the objective

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } , c } \left[ \Vert v _ { \theta } ( x _ { t } , t , c ) - v ^ { * } \Vert _ { 2 } ^ { 2 } \right] .\tag{3}
$$

![](images/b9bb36cd986daad021ffeb97633bea7c2d29853690175bca8b8fde926e378e9a.jpg)  
Fig. 2. SAM3D-Part pipeline. Given an input mesh and user-specified image-space prompts, SAM3D-Part encodes the rendered RGB image, part mask, point map, and source mesh geometry into 2D and 3D condition tokens. These tokens guide a two-stage generator that first predicts the coarse occupancy and dense source-frame correspondences of the selected part, and then refines them into a high-resolution part mesh. A cache volume storing previously generated parts is further used to support consistent sequential part generation.

During inference, generation starts from $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ <sup>�</sup>and follows the ordinary diferential equation

$$
\frac { d x } { d t } = v _ { \theta } ( x _ { t } , t , c )\tag{4}
$$

from $t = 0$ to $t = 1 ,$ , usually with Euler integration and classifier-free guidance.

TRELLIS. TRELLIS [Xiang et al. 2025] is a scalable two-stage framework for 3D generation based on structured 3D latents. Its first stage, called Sparse Structure (SS), predicts a coarse sparse occupancy structure. A flow-matching DiT (Difusion Transformer [Peebles and Xie 2023]) denoises latent tokens on a low-resolution 3D grid, and a pretrained occupancy VAE decodes them into a binary occupancy grid. This stage captures the main volumetric layout of the object. The second stage, called Structured Latent (SLat), performs generation only on occupied locations predicted by the first stage. It refines the geometry with high-resolution latent features and decodes them into the final mesh. This sparse-to-dense design reduces computation while preserving detailed geometry.

SAM3D. SAM3D [Chen et al. 2025a] is closely related to our work and serves as the direct foundation of SAM3D-Part. It addresses the task of extracting a 3D object from a user-specified 2D mask in an input image. Following the two-stage generation pipeline of TRELLIS, it uses image-space conditions, including the rendered RGB image, the user mask, and a predicted point map, to guide 3D object generation. However, its conditioning primarily relies on 2D observations and does not explicitly use the input mesh geometry. In addition, it predicts a global 7-DoF pose with a single pose head to map the generated object back to the world coordinate frame. In contrast, our task requires extracting a specific part from an existing mesh, where the mesh geometry provides important constraints. Therefore, we extend this framework with mesh-aware conditioning, dense correspondence prediction, and an iterative part-cache mechanism.

## 3.2 Multi-modal Conditioning for Part-aware Generation

To obtain a specified part mesh, our SAM3D-Part uses both 2D user prompts and 3D mesh geometry as conditions. The 2D prompt presents the user intent and part information, while the 3D mesh condition provides the full object geometry, including regions that may be invisible from the selected view. We encode these inputs into two condition streams: a 2D condition stream from rendering and user masks, and a 3D condition stream from the source mesh and the part cache. The two streams are concatenated along the token dimension and injected into the DiT through cross-attention.

View Rendering and 2D Prompt Encoding. We first render the input mesh M from a user-selected view. The rendering produces an RGB image, a foreground mask, and a point map. The point map records the 3D coordinate corresponding to each foreground pixel, and sets background pixels to a constant value (i.e., −1). The user then specifies the target part on the rendered image through simple interactions such as clicks, and a SAM-based segmentor produces the 2D part mask [Ravi et al. 2025; Yang et al. 2023].

To capture both local part details and global object context, we encode two views for each 2D signal: a cropped view around the part mask and a full rendered view. The RGB image and the mask are encoded by a frozen DINOv2 image encoder [Oquab et al. 2024], while the point map is encoded by a lightweight trainable patch encoder. Instead of concatenating the three modalities as separate token sequences, we fuse them at the same spatial patch locations. In this way, each token contains appearance, user attention, and 3D position information for the same image region. This cross-modal fusion keeps the token size compact and makes the 2D condition easier for the DiT to use.

Global Mesh Encoding. The 2D prompt alone is not suficient for part extraction, because it only observes the object from one view. To provide complete 3D information, we encode the input mesh M with a pretrained Hunyuan3D-2.1 ShapeVAE encoder [Team 2025].

Specifically, we sample 8, 1920 surface points from M, including both uniformly sampled points and curvature-weighted points, and feed them into the frozen encoder with corresponding normal and sharp-edge labels. The encoder produces a set of 3D anchors with latent features. Each anchor feature is concatenated with its positional encoding, giving a compact token representation of the input mesh. These mesh tokens allow the generation model to directly attend to the geometry of the original object, instead of reconstructing it only from the rendered view.

Part-Cache Encoding. For the iterative part extraction, we maintain a binary voxel cache $C _ { i }$ for each extraction step � to record the parts extracted in previous steps, i.e.,

$$
C _ { i } = \mathrm { O R } \left( \mathrm { V o x e l i z e } ( \mathcal { P } _ { 1 } ) , \dots , \mathrm { V o x e l i z e } ( \mathcal { P } _ { i - 1 } ) \right) .\tag{5}
$$

This cache tells the model which regions have already been extracted. Instead of training a separate cache encoder, we attach the cache information to the global mesh tokens. For each mesh anchor, we query the corresponding voxel in $C _ { i }$ via nearest-voxel lookup and obtain a binary flag indicating whether this location belongs to a previously extracted part. This flag is concatenated with the anchor latent and position feature, and then projected to the DiT condition dimension. As a result, each 3D condition token contains local geometry, spatial position, and extraction history. This design naturally supports iterative sub-part extraction.

## 3.3 Two-stage Part Generation and Global Alignment

Given the multi-modal conditions, SAM3D-Part follows a two-stage generation process, as shown in Fig. 2. The first stage predicts the sparse structure of the target part and its dense correspondence to the source mesh. The second stage refines the part geometry on the predicted structure and decodes the final mesh. Both stages are trained with the flow matching objective introduced in Sec. 3.1.

Stage 1: Sparse Structure and Dense Correspondence Generation. The first stage generates the coarse volumetric structure of the queried part. A flow-matching DiT denoises a 3D latent grid under the conditions from the 2D prompt, point map, source mesh tokens, and part-cache tokens. The output latent is decoded by two heads. The first head predicts a binary occupancy grid, which describes the spatial support of the target part. The second head predicts a dense XYZ volume, where each occupied voxel is assigned a corresponding 3D coordinate in the source mesh frame.

This dense XYZ prediction is important for robust alignment. A direct pose regression head would provide only a small number of supervision signals for each training sample, making pose learning unstable. Instead, our model predicts source-frame coordinates for all occupied voxels, providing dense voxel-level supervision. Since each occupied voxel has both a local coordinate and a predicted source-frame coordinate, their correspondence is already known and no nearest-neighbor matching is required.

Closed-Form Source-Frame Alignment. The generated part is represented in a normalized local coordinate system. During training, each ground-truth part is transformed from the source frame to this local frame using only centering and isotropic scaling, without rotation canonicalization. Thus, its local axes inherit the source-mesh axes, while the dense XYZ branch is supervised in the source frame. Accordingly, we estimate only an isotropic scale and a translation; this orientation consistency is learned rather than guaranteed at inference. Let $p _ { \mathrm { l o c a l } } ( \boldsymbol { v } )$ be the local coordinate of an occupied voxel $v ,$ and let $ { p _ { \mathrm { g l o b a l } } } ( \upsilon )$ be the source-frame coordinate predicted by the dense XYZ head. We solve

$$
\operatorname* { m i n } _ { { s , o } } \sum _ { v } \left\| s \cdot p _ { \mathrm { l o c a l } } ( v ) + o - p _ { \mathrm { g l o b a l } } ( v ) \right\| _ { 2 } ^ { 2 } ,\tag{6}
$$

where � is the isotropic scale and � is the translation. This leastsquares problem has a closed-form solution. Let $\bar { p } _ { \mathrm { l o c a l } }$ and $\bar { p } _ { \mathrm { g l o b a l } }$ be the centroids of the two point sets. The translation is

$$
o = \bar { p } _ { \mathrm { g l o b a l } } - s \cdot \bar { p } _ { \mathrm { l o c a l } } ,\tag{7}
$$

and the scale is obtained by projecting the centered local coordinates onto the centered global coordinates. The recovered (�, �) is then used to place the generated part back into the coordinate frame ofthe source mesh. This alignment is not an ICP procedure [Chetverikov et al. 2002], because the correspondences are directly predicted by the model.

Stage 2: Structured Latent Refinement. After the sparse structure and alignment are determined, the second stage refines the detailed geometry of the part. It performs structured latent denoising only on the occupied voxels predicted by Stage 1. Compared with Stage 1, this stage focuses on local geometric details rather than deciding which region should be extracted. We also inject the same global mesh condition into the Stage-2 DiT, so that the refined geometry remains consistent with the surface and shape of the source object. Finally, the structured latent is decoded into a complete part mesh with geometry and appearance.

Training Objective. Both stages are trained with the same flow matching objective:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } , c } \left[ \| v _ { \theta } ( x _ { t } , t , c ) - ( x _ { 1 } - x _ { 0 } ) \| _ { 2 } ^ { 2 } \right] ,\tag{8}
$$

where $x _ { 1 }$ is the ground-truth latent target, $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ is Gaussian noise, and � denotes the multi-modal conditions. In Stage 1, $x _ { 1 }$ contains the latent targets for both occupancy and dense XYZ prediction. In Stage $2 , x _ { 1 }$ is the structured latent of the target part. During inference, we sample from Gaussian noise and integrate the learned velocity field with classifier-free guidance to obtain the final part mesh.

## 4 Experiments

Table 1. Quantitative comparison of part generation. CD and F1-� are computed in GT-bbox-normalized space. Bold: the best result for each metric.
<table><tr><td>Method</td><td>BBox IoU ↑</td><td>CD↓</td><td>F1-0.05 ↑</td><td>F1-0.1 ↑</td></tr><tr><td>SAM-3D-Objects (Meta)</td><td>0.225</td><td>0.375</td><td>0.221</td><td>0.388</td></tr><tr><td>HoloPart + P3-SAM</td><td>0.291</td><td>0.452</td><td>0.445</td><td>0.521</td></tr><tr><td>X-Part + P3-SAM</td><td>0.458</td><td>0.282</td><td>0.608</td><td>0.674</td></tr><tr><td>OmniPart</td><td>0.552</td><td>0.192</td><td>0.666</td><td>0.756</td></tr><tr><td>Ours (full)</td><td>0.641</td><td>0.053</td><td>0.716</td><td>0.849</td></tr></table>

Table 2. Quantitative comparison on PartObjaverse-Tiny. Bold: the best result for each metric.
<table><tr><td>Method</td><td>BBox IoU ↑</td><td>CD↓</td><td>F1-0.05 ↑</td><td>F1-0.1 ↑</td></tr><tr><td>SAM-3D-Objects (Meta)</td><td>0.255</td><td>0.294</td><td>0.240</td><td>0.392</td></tr><tr><td>HoloPart + P3-SAM</td><td>0.270</td><td>0.114</td><td>0.531</td><td>0.617</td></tr><tr><td>X-Part + P3-SAM</td><td>0.317</td><td>0.111</td><td>0.547</td><td>0.634</td></tr><tr><td>OmniPart</td><td>0.473</td><td>0.065</td><td>0.585</td><td>0.782</td></tr><tr><td>Ours (full)</td><td>0.688</td><td>0.054</td><td>0.694</td><td>0.851</td></tr></table>

Table 3. Ablation on the multi-part test set. CD and F1-� are computed in the GT-bbox-normalized space. Bold: the best result for each metric.
<table><tr><td>Configuration</td><td>BBox ↑</td><td>CD↓</td><td>F1-0.05 ↑</td><td>F1-0.1 ↑</td></tr><tr><td>(a) Finetuned SAM3D</td><td>0.275</td><td>0.269</td><td>0.311</td><td>0.453</td></tr><tr><td>(b) + Mesh conditioning</td><td>0.426</td><td>0.134</td><td>0.510</td><td>0.664</td></tr><tr><td>(c) + Token fusion</td><td>0.518</td><td>0.128</td><td>0.592</td><td>0.731</td></tr><tr><td>(d) + XYZ branch</td><td>0.611</td><td>0.067</td><td>0.697</td><td>0.819</td></tr><tr><td>(e) + Part cache</td><td>0.625</td><td>0.061</td><td>0.711</td><td>0.839</td></tr><tr><td>(f) + 2-stage w. mesh condition</td><td>0.641</td><td>0.053</td><td>0.716</td><td>0.849</td></tr></table>

Table 4. Ablation on the robustness to viewpoint and click variations.
<table><tr><td>Study</td><td>BBox IoU ↑</td><td>CD↓</td><td>F1-0.05 ↑</td><td>F1-0.1 ↑</td></tr><tr><td>Viewpoint</td><td> $0 . 6 1 6 \pm 0 . 0 9 8$ </td><td> $0 . 0 5 7 \pm 0 . 0 2 0$ </td><td> $0 . 7 0 8 \pm 0 . 0 9 7$ </td><td> $0 . 8 3 8 \pm 0 . 0 7 5$ </td></tr><tr><td>Click</td><td> $0 . 6 0 9 \pm 0 . 0 6 5$ </td><td> $0 . 0 6 7 \pm 0 . 0 2 4$ </td><td> $0 . 6 8 6 \pm 0 . 0 5 3$ </td><td> $0 . 8 1 9 \pm 0 . 0 5 1$ </td></tr></table>

Table 5. Efect of token fusion on Stage-1 cross-atention eficiency. Our fusion reduces token count and computational cost while incorporating additional 3D conditions.
<table><tr><td>Metric</td><td>SAM3D</td><td>Ours</td><td>Ours (w/o fusion) (w/ fusion)</td></tr><tr><td>Tokens  $N _ { c }$ </td><td>8 218</td><td>12 314</td><td>6836</td></tr><tr><td>Cross-attn FLOPs</td><td>4.13 T</td><td>5.99T</td><td>3.51T</td></tr><tr><td>Latency (ms)</td><td>11 246.0</td><td>13 416.1</td><td>10 450.7</td></tr><tr><td>Peak VRAM (GB)</td><td>22.09</td><td>23.73</td><td>21.53</td></tr></table>

## 4.1 Experimental Setup

Implementation Details. Our model uses a two-stage design. Stage 1 is a sparse-structure DiT with 24 transformer blocks. It predicts $1 6 ^ { 3 }$ occupancy grids and xyz latent grids. Stage 2 is an SLat DiT that converts the Stage 1 occupancy output into a high-resolution mesh, conditioned on the input image, mask, and global mesh. Specifically, for conditioning, the 2D branches share a frozen DINOv2 ViT-L backbone with 518/14 patches. The xyz point-map branch uses a 296/8 PointPatchEmbed. The 3D branch encodes the input mesh with the Hunyuan3D-2.1 ShapeVAE, which samples 81,920 surface points, including 40,960 uniformly sampled points and 40,960 curvatureweighted points, and produces 4,096 mesh anchor tokens. We train Stage 1 for 60k iterations with a global batch size of 128 on 8 H100 GPUs, using AdamW with a peak learning rate of 5×10<sup>−5</sup>. Stage 2 is fine-tuned for 40k iterations with the same optimizer. During inference, we use 25 sampling steps for Stage 1 and 12 sampling steps for Stage 2 on a single H100 GPU in fp16.

![](images/08c84e8e4707f7e71594e0c767458ce02ca79c40fc6b13dd9bb34406a854e1c7.jpg)  
Fig. 3. Visual results of ablation studies for the design of SAM3D-Part.

XYZ VAE. The XYZ latent predicted by Stage 1 is decoded into $\mathrm { ~ a ~ } 6 4 ^ { 3 }$ per-voxel correspondence map using a sparse-structure XYZ VAE, adapted from the TRELLIS Stage-1 occupancy VAE [Xiang et al. 2025]. We retain the encoder-decoder backbone and the $1 6 ^ { 3 }$ latent grid with 8 channels, then make three modifications: (i) We replace the 1-channel occupancy input with 4-channel XYZ coordinates, enabling the VAE to encode per-voxel correspondences instead of binary occupancy; (ii) To help the decoder focus on occupied voxels, we make it occupancy-conditioned. Specifically, at the decoder bottleneck, we concatenate the 8-channel occupancy latent from the frozen SAM3D occupancy VAE with the 8-channel XYZ latent along the channel dimension; (iii) We simplify the output head to a shared Norm+SiLU block followed by a 3-channel 3×3×3 convolution and a [−1, 1] clamp. We remove the additional non-linear layers in the original head because our target is a continuous coordinate rather than a logit.

We train the XYZ VAE with an $L _ { 1 }$ reconstruction loss on XYZ values, evaluated only at occupied voxels, together with a TV smoothness regularizer weighted by 0.01 and a KL term weighted by $1 0 ^ { - 3 }$ To prevent large parts from dominating the loss, we divide the persample weight by the bounding-box diagonal of each part. This size re-weighting gives small parts a comparable training signal. The XYZ VAE is trained on the same dataset as the DiT for 70k iterations using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a global batch size of 128, and an EMA decay of 0.9999. We use the EMA weights at inference time.

Datasets. We curate part-level training data from five public 3D asset collections, i.e., Objaverse [Deitke et al. 2023b], Objaverse-XL [Deitke et al. 2023a], ABO [Collins et al. 2022], 3D-FUTURE [Fu et al. 2021], and HSSD [Khanna et al. 2024]. Each GLB file often contains a native part structure, which we use as the basis for data curation. We first discard assets with fewer than 2 parts or more than 32 parts, since they are either not meaningfully decomposed or are often over-segmented and scene-like. This gives an initial pool of about 446k assets. We then split each sub-mesh by connectivity and merge small fragments into nearby larger parts based on collision proximity, leading to more balanced part decompositions. Finally, annotators manually remove assets with unreasonable decompositions, such as parts crossing semantic boundaries, duplicated parts, or visually indistinguishable parts, resulting in 187k verified assets.

For evaluation, we use three sets from diferent sources. The first is a multi-part test set seperate from training set containing 500 assets with ground-truth parts. The second is the PartObjaverse-Tiny, a public dataset mostly used by 3D part segmentation and part-level generation. The third is an in-the-wild set generated from commercial 3D generator. Since it does not provide ground truth, it is not used for metric computation.

Baselines. We compare against four representative baselines, i.e., SAM3D [Chen et al. 2025a], the two-stage pipelines HoloPart + P3- SAM [Yang et al. 2025a] and X-Part + P3-SAM [Yan et al. 2025], and the end-to-end method OmniPart [Yang et al. 2025b]. The HoloPart/X-Part pipelines first use P3-SAM to segment the input mesh in 3D and then complete each segment into a part mesh, while SAM3D and OmniPart directly predict part meshes from the 2D condition. For a fair comparison, all methods are queried using the same 2D target mask. OmniPart receives the mask directly, while foreground points sampled from the mask are lifted to the source mesh as 3D prompts for P3-SAM. Since the P3-SAM-based methods output a full decomposition, we select the predicted component with the highest GT-part BBox IoU. This oracle selection gives the decomposition baselines their most favorable match for evaluation. All baselines use publicly released checkpoints and inputs generated from the same source assets.

Evaluation Metrics. We evaluate part geometry with four metrics: bounding-box IoU (BBox IoU), Chamfer Distance (CD), and F1-scores at �=0.05 and �=0.1. Before evaluation, we normalize all predicted and ground-truth parts to the ground-truth bounding box, which is scaled to the $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ cube. This removes the efects of global scale and canonicalization diferences. We also evaluate Stage 1 eficiency using the number of cross-attention condition tokens $N _ { c } .$ cross-attention FLOPs, latency, and peak VRAM usage.

## 4.2 Comparison with State-of-the-Art

Tab. 1 reports the quantitative results on the multi-part test set. Our method achieves the best performance on all metrics: BBox IoU increases to 0.641, compared with the next-best score of 0.552 from OmniPart; CD decreases to 0.053, compared with 0.192 from OmniPart; and F1-0.1 improves from 0.756 to 0.849. This trend is consistent under both strict and relaxed F1 thresholds, showing that the improvement is not limited to a specific tolerance level but reflects a broader gain in geometric accuracy. Tab. 2 reports the quantitative results on PartObjaverse-Tiny dataset. Our method still performs strongly on this independent benchmark, supporting that the method generalizes beyond our curated benchmark.

As shown in Fig. 4, visual results also demonstrate the superiority of our method. The baselines exhibit three common failure cases: (i) The two-stage segmentation + completion pipelines (P3-SAM + X Part and P3-SAM + HoloPart) inherit the segmentation behavior of P3-SAM. As a result, their segmentations are hard to control and often overly fragmented, and the completion stage further amplifies these errors. (ii) Although OmniPart achieves competitive BBox IoU, it tends to produce overly smooth part geometry and misses fine details from the input mesh. (iii) SAM3D does not use a 3D condition, so its predictions often drift away from the input mesh and recover generic parts that do not remain on the original surface. In contrast, our results stay well aligned with the input mesh while preserving fine-grained shape details, which explains the large improvements in CD and F1.

Fig. 5 and Fig. 7 further evaluate in-the-wild inputs that are outside the training distribution. Fig. 5 uses the same six-method layout and shows that our method remains robust on stylized characters, cartoon weapons, and furniture. The baselines either miss entire parts or collapse them into coarse shapes, whereas our method preserves both the part decomposition and the surface alignment. These results suggest that our method can generalize to animals, humans, mechanical parts, plants, and buildings.

## 4.3 Ablation Study

We study our method from two aspects: the contribution of each component to accuracy (Tab. 3) and the eficiency benefit of crossmodal token fusion (Tab. 5). All ablation studies are conducted on the multi-part test set, using the same training and inference settings as in the main experiment.

4.3.1 Component Ablation. Tab. 3 shows a cumulative ablation. We start from a finetuned SAM3D baseline and add our design choices one at a time. Fig. 3 shows the corresponding visual results for two assets, where each row matches rows (a)-(f) in Tab. 3.

(a) Finetuned SAM3D. We first finetune SAM3D on our training dataset. This model uses only the 2D RGB image and mask as input. It benefits from in-domain training, achieving a BBox IoU of 0.275 and a CD of 0.269. However, without any 3D condition, its predictions can drift away from the input surface, as shown by the rocket and bed examples in Fig. 3.

(b) +Mesh conditioning. We then encode the input mesh with ShapeVAE anchors and inject them into cross-attention. This gives the largest improvement in Tab. 3: BBox IoU increases from 0.275 to 0.426 (+0.151), while CD decreases from 0.269 to 0.134. The rocket’s top and base are better separated, and the bed begins to show distinct pillow and blanket regions. This shows that 3D anchors provide useful surface information that 2D conditions alone cannot capture.

(c) +Tokenfusion. The 2D inputs, including RGB, mask, and pointmap, are encoded separately across two views. Since directly using all tokens makes cross-attention costly and less focused, we fuse tokens from the three 2D modalities at the same patch location and view. This further improves BBox IoU to 0.518 and CD to 0.128. As shown in Tab. 5, it also reduces the Stage 1 cost below the SAM3D baseline. Visually, the rocket-body boundary becomes cleaner, and the bed surface shows more accurate pillow folds.

(d) +XYZ branch. We replace the single-vector pose head with an XYZ branch. In this design, Stage 1 predicts a per-voxel XYZ correspondence latent from the generated part to the source space. The part scale and translation are then computed from these dense voxel-level votes, rather than regressed directly. This design is more robust under partial visibility. It improves the BBox IoU to 0.611 and CD to 0.067, with clearer placement of the rocket body and base and better contact between the blanket and mattress in Fig. 3.

(e) +Part cache For sequential generation, we voxelize previously generated parts into a binary cache and query the cache at each mesh anchor. The resulting binary flag is appended to the anchor feature, providing later queries with the extraction history. This improves BBox IoU to 0.625 and CD to 0.061 and reduces conflicts between generated parts.

(f) +2-stage with mesh condition. Finally, we also provide the input mesh as a 3D condition to Stage 2 during SLat refinement. This gives the full model, with a BBox IoU of 0.641, CD of 0.053, and F1-0.1 of 0.849. The Stage 2 mesh condition mainly improves fine surface details, such as the sharper rocket base and clearer blanket folds in the last column of Fig. 3.

4.3.2 Viewpoint and Click Robustness Ablation. Our training protocol already exposes the model to some variations: rendering views are randomly sampled, and the input masks are SAM2-predicted masks from GT-derived prompts rather than GT masks. We keep masks with 2D IoU > 0.6 during training to remove invalid or highly ambiguous prompts. To verify this robustness, we conducted a smallscale study by sampling 100 cases from the test set. For viewpoint robustness, we sample 6 valid views per target part, where the target part is visible, use the gt mask for each view, and evaluate the generated 3D part. For click robustness, we fix the view and sample 6 diferent click seeds on the visible target part, again using SAM2 masks. As shown in Tab. 4, the results show stable performance across views and clicks.

4.3.3 Token Fusion Ablation. Token fusion is not only useful for accuracy but also important for eficiency, as shown in Tab. 5. We measure the Stage 1 cross-attention cost on a single H100 in fp16, with batch size 1, 25 sampling steps, and classifier-free guidance. We report the mean over 10 timed runs after 2 warm-up runs. Tab. 5 shows three main results: (i) The SAM3D baseline attends to � =8,218 tokens and uses 4.13 T cross-attention FLOPs, with a latency of 11,246.0 ms and a peak VRAM of 22.09 GB. (ii) Directly adding our 4,096 mesh anchor tokens to the 2D conditions increases �<sub>�</sub> to 12,314 (+50%), FLOPs to 5.99 T (+45%), latency to 13,416.1 ms (+19%), and peak VRAM to 23.73 GB (+7.4%). Thus, the 3D condition would add a clear eficiency cost without using token fusion. (iii) Cross-modal patch fusion merges the three 2D modalities at the same patch locations into one shared token set. This reduces � to 6,836, which is lower than the SAM3D baseline even with the added 3D modality. As a result, our full model uses 3.51 T FLOPs (−15% vs. SAM3D), 10,450.7 ms latency (−7%), and 21.53 GB peak VRAM (−2.5%). Together with the accuracy gain in Tab. 3, this shows that token fusion provides a double benefit: it removes redundant 2D tokens while covering the extra cost of the 4,096 mesh anchor tokens.

## 4.4 Limitations and Failure Cases

While our SAM3D-Part can generate high-quality specified 3D parts, it still struggles with extremely complex cases. Fig. 6 shows a failure case: a sculpted figure holding a hanging lantern and wrapped by thin vines. This example contains thin lattice-like structures, such as vine stems, leaf veins, and the open lantern frame, as well as strongly overlapping parts, where vines wind around both the figure and the lantern. In this setting, structures thinner than a voxel cannot be captured by the 64<sup>3</sup> sparse representation or the Hunyuan3D ShapeVAE bottleneck. As a result, the lantern frame and the smallest vine branches are smoothed out or merged with nearby surfaces.

The overlapping geometry also makes the queried 2D mask ambiguous across diferent depth layers. The model may therefore assign geometry to the wrong layer or split one part into disconnected frag ments. Addressing these challenges may require higher-resolution sparse representations, multi-view mask supervision, and explicit modeling of part-to-part contact, which we leave for future work.

## 5 Conclusion

We presented SAM3D-Part, a prompt-driven framework for generating a specified 3D part from a source mesh. Given a user prompt, SAM3D-Part returns the queried part as a complete mesh aligned with the source. It combines SAM-based 2D part prompts with the input mesh through pixel-aligned multimodal fusion, and places the generated part back into the source frame using dense voxel-level XYZ correspondences. A part cache further improves consistency across sequential multi-part queries. Experiments show that SAM3D-Part achieves strong part quality, source alignment, and sequential consistency. The ablation study also shows that cross-modal token fusion improves both accuracy and eficiency, reducing the Stage 1 cost below the SAM-3D-Objects baseline. SAM3D-Part still struggles with thin lattice-like structures and heavily overlapping or nested parts. We plan to address these limitations with higher-resolution sparse representations, multi-view mask supervision, and explicit part-to-part contact modeling. We hope this prompt-driven and source-aligned design can support future part-level workflows in 3D editing, animation, and fabrication.

## Acknowledgments

This work was supported in part by the Guangdong S&T Programme under Grant No. 2024B0101030002; the Basic Research Project of the Hetao Shenzhen–Hong Kong S&T Cooperation Zone under Grant No. HZQB-KCZYZ-2021067; the National Natural Science Foundation of China (NSFC) under Grant No. 62293482; the Guangdong Provincial Fund for Distinguished Young Scholars under Grant No. 2023B1515020055; the Shenzhen Outstanding Talents Training Fund under Grant No. 202002; the Guangdong Provincial Key Laboratory of Future Networks of Intelligence under Grant No. 2022B1212010001; the Shenzhen Key Laboratory of Big Data and Artificial Intelligence under Grant No. SYSPG20241211173853027; the Guangdong Province Radio Science Data Center under Grant No. 2025B1212070001; the National Key R&D Program of China under Grant No. 2018YFB1800800; the National Natural Science Foundation of China (No. 62502209); and the Fundamental Research Funds for the Central Universities (No. 30925010538).

## References

Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka. 2023. Satr: Zero-shot semantic segmentation of 3d shapes. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 15166–15179

Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al. 2023. Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems 36 (2023), 25971–25990.

Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. 2025b. Partgen: Part-level 3d generation and recon struction with multi-view difusion models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 5881–5892.

Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. 2026. AutoPartGen: Autoregressive 3D Part Generation and Discovery. Advances in Neural Information Processing Systems 38 (2026), 153496–153521.

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In Proceedings ofthe IEEE/CVF international conference on computer vision. 22246–22256.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. 2025a. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624 (2025).

Dmitry Chetverikov, Dmitry Svirko, Dmitry Stepanov, and Pavel Krsek. 2002. The trimmed iterative closest point algorithm. In 2002 International Conference on Pattern Recognition, Vol. 3. IEEE, 545–548.

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. 2022. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21126–21136.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. 2023a. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems 36 (2023), 35799–35813.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023b. Obja verse: A universe of annotated 3d objects. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 13142–13153.

Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 2021. 3d-future: 3d furniture shape with texture. International Journal ofComputer Vision 129, 12 (2021), 3313–3337.

Xufan He, Yushuang Wu, Xiaoyang Guo, Chongjie Ye, Jiaqing Zhou, Tianlei Hu, Xiaoguang Han, and Dong Du. 2025. UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents. arXiv preprint arXiv:2512.09435 (2025).

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2024. Lrm: Large reconstruction model for single image to 3d. In International Conference on Learning Representations, Vol. 2024. 50678–50702.

Heewoo Jun and Alex Nichol. 2023. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023).

Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X Chang, and Manolis Savva. 2024. Habitat synthetic scenes dataset (hssd-200): An analysis of3d scene scale and realism tradeofs for objectgoal navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16384–16393.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4015–4026.

Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Ke qing Fan, Pan Hu, Sheng Wang, Buyu Li, et al. 2026a. SegviGen: Repurposing 3D Generative Model for Part Segmentation. arXiv preprint arXiv:2603.16869 (2026).

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. 2025. TripoSG: High Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models. arXiv preprint arXiv:2502.06608 (2025).

Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. 2026b. Sparc3d: Sparse representation and construction for high-resolution 3d shapes modeling. Advances in Neural Information Processing Systems 38 (2026), 118582–118600.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3d: High resolution text-to-3d content creation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 300–309.

Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Feng Yiqiang, Yadong Mu, and Katerina Fragkiadaki. 2026. Partcrafter: Structured 3d mesh generation via compositional latent difusion transformers. Advances in neural information processing systems 38 (2026), 35387–35415.

Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. 2024. Part123: Part-aware 3D reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2025. Partfield: Learning 3d feature fields for part segmentation and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9704–9715.

Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. 2023b. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21736–21746.

Xingchao Liu, Chengyue Gong, and Qiang Liu. 2023a. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations (ICLR).

Changfeng Ma, Yang Li, Xinhao Yan, Jiachen Xu, Yunhan Yang, Chunshi Wang, Zibo Zhao, Yanwen Guo, Zhuo Chen, and Chunchao Guo. 2025. P3-sam: Native 3d part segmentation. arXiv preprint arXiv:2509.06784 (2025).

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. 2022. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751 (2022).

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research Journal (2024), 1–31.

William Peebles and Saining Xie. 2023. Scalable difusion models with transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4195–4205.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022).

Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. 2017a. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 652–660.

Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. 2017b. PointNet++ deep hierarchical feature learning on point sets in a metric space. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems. 5105–5114.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2025. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, Vol. 2025. 28085–28128.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023. Mvdream: Multi-view difusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023).

Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. 2024. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10208–10217.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. 2024. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision. Springer, 1–18.

Jiaxiang Tang, Ruijie Lu, Max Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. 2026. Eficient part-level 3d object generation via dual volume packing. Advances in Neural Information Processing Systems 38 (2026), 27115–27137.

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. 2023. Make-it-3d: High-fidelity 3d creation from a single image with difusion prior. In Proceedings of the IEEE/CVF international conference on computer vision. 22819–22829.

Tencent Hunyuan3D Team. 2025. Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material. arXiv:2506.15442 [cs.CV]

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable and versatile 3d generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 21469–21480.

Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. 2024. Grm: Large gaussian reconstruction model for eficient 3d reconstruction and generation. In European Conference on Computer Vision. Springer, 1–20.

Han Yan, Yang Li, Zhennan Wu, Shenzhou Chen, Weixuan Sun, Taizhang Shang, Weizhe Liu, Tian Chen, Xiaqiang Dai, Chao Ma, et al. 2024. Frankenstein: Generating semantic-compositional 3d scenes in one tri-plane. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Xinhao Yan, Jiachen Xu, Yang Li, Changfeng Ma, Yunhan Yang, Chunshi Wang, Zibo Zhao, Zeqiang Lai, Yunfei Zhao, Zhuo Chen, et al. 2025. X-part: high fidelity and structure coherent shape decomposition. arXiv preprint arXiv:2509.08643 (2025).

Yunhan Yang, Yuan-Chen Guo, Yukun Huang, Zi-Xin Zou, Zhipeng Yu, Yangguang Li, Yan-Pei Cao, and Xihui Liu. 2025a. Holopart: Generative 3d part amodal segmenta tion. arXiv preprint arXiv:2504.07943 (2025).

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y Lam, Yan-Pei Cao, and Xihui Liu. 2024. Sampart3d: Segment any part in 3d objects.

arXiv preprint arXiv:2411.07184 (2024).

Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. 2023. Sam3d: Segment anything in 3d scenes. arXiv preprint arXiv:2306.03908 (2023).

Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. 2025b. Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers. 1–12.

Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. 2025. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 25050–25061.

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 2023. 3dshape2vecset: A 3d shape representation for neural fields and generative difusion models. ACM Transactions On Graphics (TOG) 42, 4 (2023), 1–16

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. Clay: A controllable large-scale generative mode for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Longwen Zhang, Qixuan Zhang, Haoran Jiang, Yinuo Bai, Wei Yang, Lan Xu, and Jingyi Yu. 2025. BANG: Dividing 3D assets via generative exploded dynamics. ACM Transactions on Graphics (TOG) 44, 4 (2025), 1–21.

Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. 2021. Point transformer. In Proceedings ofthe IEEE/CVF international conference on computer vision. 16259–16268.

![](images/c932e4a723728b5af212a12d7576d062635a9aca71ec1c65d24d606b7f535ad3.jpg)

![](images/b3d11fedffa4002d84d1e322c5166a08beb68788b210b1ba8e251b05108136cc.jpg)

![](images/d3c5f0b95524067732b6011a30a1935da25b236586d0e7f91de01336eb932018.jpg)  
Input RGB & Mask  
P3-SAM & XPart

![](images/0720b33d4be468f3d0667fe3baf994e57f2cebe625152894fccb5602f3672f13.jpg)  
P3-SAM & HoloPart

![](images/991f246d5b574d3063b72f279968c515fdc23c4386567cd1c04f5dbb7601a671.jpg)  
OmniPart

![](images/cd6ac3d53ac8e238a71b5288ae569d5635cd42ec44ff5cb00c38a5b3d4323b69.jpg)  
SAM3D

![](images/9aca666e9b0300dac85bd7f62cdf15ec0cc79bf8160c438d0f8a180dedfb53ab.jpg)  
Ours

![](images/4a1552bc99295c99bd7dd46479acc9b8b940af94d302752ff29be009026fa25e.jpg)  
GT  
Input RGB & Mask

Fig. 4. Qualitative comparison on the multi-part test set. For each example (row), we show the input RGB and mask (left), the part predictions of all baselines, our method, and the GT (right). For each method, we visualize both the parts placed in the source object and the parts laid out separately.  
![](images/6ad4bac1fd68932a7e632ecaa7684be392b1dbcf2af4bda6566b152a1e437f90.jpg)  
Mesh  
P3-SAM & XPart

![](images/438bbb6f488677c974ade451ee0bb01101e91c24b8e8ecd91db709ae2ed37971.jpg)  
P3-SAM & HoloPart

![](images/258bf0658b61423abb0dc7d75f5af7b9fe1feddcc758ae543fbffbec625bf13e.jpg)  
OmniPart

![](images/f91fd56422baef7371599ed38ebc01c65bbbfa6949327abd68ada1488a60d5a8.jpg)  
SAM3D

![](images/ec8c75abae40c9b051dbfcc8596ea741d369195bf87965159ae9856824448333.jpg)  
Ours

Fig. 5. Qualitative comparison on in-the-wild assets (no GT available). For each example (row), we show the input RGB and mask (left), the source mesh, and the part predictions of all baselines and our method. For each method, we visualize both the parts placed in the source object and the parts laid out separately.  
![](images/d23842166a865bbdb74e5286bc28bf4ecddffda114e1f4dbd11925b5ca158f90.jpg)  
Input RGB

![](images/e181fd9ac07326a284b2942105ca46f3dcf427b79cebabeb9f2ee571dfd37413.jpg)  
Input Mesh

![](images/cf839f9cf7c492f59a508c086f96a9984885958df6a0cde7c2b80720df0a1c56.jpg)  
Posed-Parts

![](images/d07a888307386f9ec780145e20d9490189c559ad414ccb21e61f9b8e35b7a005.jpg)  
Part-1

![](images/ccad0928d40568e787e22fe467b6a9fcafd0241b3ee2d1483ca47b8c3e84f3dc.jpg)  
Part-2

Fig. 6. Failure case. From left to right: input RGB, input mesh, and the two parts predicted by our method. Our method struggles with assets containing thin, latice-like structures (e.g., vines, leaf veins, and openwork lantern frames), where parts are heavily entangled.

![](images/cff37ce0cfbc6aaee91d410f442096990711c2817db752ca0ca3ed1711c18aae.jpg)  
Input RGB  
Input Mesh  
Posed-Parts  
Part-1  
Part-2  
Part-3  
SA Conference Papers ’26, December 01–04, 2026, Kuala Lumpur, Malaysia.

Fig. 7. More in-the-wild results of our method. For each example, we show the input RGB image, input mesh, assembled posed parts, and three generated parts laid out separately.