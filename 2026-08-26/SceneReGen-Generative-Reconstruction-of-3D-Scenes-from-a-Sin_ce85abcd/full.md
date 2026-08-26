# SceneReGen: Generative Reconstruction of 3D Scenes from a Single Image

Zefan Tian<sup>1∗</sup>, Yuteng Ye<sup>1∗</sup>, Yiheng Zhang<sup>2</sup>, Yuhang Yang<sup>2</sup>, Xueqiang Lv<sup>2</sup>, Shizhou Zhang<sup>2</sup>, Le Liu<sup>2†</sup>, Di Xu<sup>1†</sup>

<sup>1</sup>Huawei <sup>2</sup>Northwestern Polytechnical University

## Abstract

Single-image 3D scene reconstruction must complete partially observed objects and place them coherently in a shared observation-aligned scene frame. Object-level generative priors ofer strong completion ability, but their centered, scalenormalized outputs are typically expressed in an object frame, creating a fundamental representation gap between object generation and scene reconstruction. We introduce SceneRe-Gen, a generative reconstruction framework that reinterprets scene reconstruction as the generation and assembly of complete object assets in a shared observation-aligned scene frame. SceneReGen addresses the generation–reconstruction gap through selective pose factorization: each object’s observed orientation is encoded directly in the generated mesh, while translation and scale are estimated from instance-level and global scene evidence. Given a scene image and instance masks, a geometry encoder extracts dense cues; learnable shape queries condition a pretrained DiT-based 3D generator to produce complete meshes in their observed orientations, while position queries fuse object and scene features to assemble them in the shared frame. On the 3D-FUTURE evaluation subset, SceneReGen achieves the best scene-level CD, scenelevel F-Score, and 3D bounding-box IoU among the evaluated methods, ties the best object-level CD, and ranks second in object-level F-Score. Qualitative outputs in autonomousdriving and embodied-AI scenes further illustrate the potential of asset-centric reconstruction beyond indoor furniture.

## 1 Introduction

3D scene reconstruction from a single RGB image is fundamental for embodied AI, autonomous driving, and virtual content creation. Given the image, the task requires recovering a complete mesh for each partially observed object and placing the meshes coherently in a shared observationaligned scene frame. Recent feed-forward 3D reconstruction models, such as DUSt3R (Wang et al. 2024) and VGGT (Wang et al. 2025a), recover useful geometric cues from one or more views. In a single view, however, selfand inter-object occlusions leave much of the object geometry unobserved. Recent 3D generative models provide strong priors over complete object geometry (Zhang et al. 2023; Zhao et al. 2023; Zhang et al. 2024; Zhao et al. 2025; Xiang et al. 2025), making them promising for completing these missing regions.

Applying object-level generative priors to object-centric 3D scene reconstruction, however, exposes the central representation gap addressed in this work: generated geometry is typically centered, scale-normalized, and expressed in a canonical object frame, whereas the target scene requires each complete mesh to be oriented, translated, and scaled in a shared observation-aligned scene frame. Reconstruction must therefore bridge object completion and observationaligned spatial placement rather than solve either in isolation. Existing approaches broadly follow two routes. Object-first pipelines complete each object separately and then recover its object-to-scene transformation (Ardelean, Özer, and Egger 2025; Yao et al. 2025). Even recent variants that use objectlocal cues for rotation and scene-level cues for translation and scale still represent all three as explicit transformation outputs (Shi et al. 2026). These designs preserve dedicated object priors, but keep object-to-scene placement separate from geometry generation. With partial observations, errors in completed geometry or its correspondence to visible evidence can then carry into placement. Joint scene-modeling approaches instead use cross-object context to predict object geometry and spatial arrangement within a shared framework (Huang et al. 2025b; Meng et al. 2026). This brings scene context into completion, but binds detailed geometry to the full placement problem, making geometry and placement errors harder to separate. These complementary risks

We therefore ask a more targeted question: which pose components should be coupled to object generation, and which should be estimated from scene evidence? Our hypothesis is that orientation is more tightly coupled to imageconditioned object generation. Completing a partially observed object requires mapping its visible structures into a complete mesh in the target orientation; a post-hoc rigid rotation can reorient that mesh, but does not by itself resolve inconsistencies introduced during completion. Translation and scale instead determine where the completed object lies and how large it is in the shared observation-aligned scene frame. The 2D object extent, image-derived depth and local geometry, and cross-object context provide complementary cues for estimating these variables, although single-view ambiguity remains. This motivates a selective pose factorization: orientation is represented in the generated vertices, while translation and scale remain explicit placement variables estimated from scene evidence.

![](images/4d75519658642de17c34da5ff46a40999b50cc6a1085f3425cfee0b0429abb6b.jpg)  
Figure 1: Qualitative SceneReGen outputs in autonomous-driving and embodied-AI scenarios. Given a scene image (a), SceneReGen reconstructs complete object meshes in their observed orientations (b, d) and estimates translation and scale to assemble them in a shared observation-aligned scene frame (c).

Based on this design, we introduce SceneReGen, a framework that turns rotation-aware object generation into singleimage 3D scene reconstruction through selective pose factorization. For each masked object instance, a pretrained geometry encoder extracts dense image-derived geometric cues. Learnable shape queries aggregate these cues into compact conditioning tokens for a pretrained object-level 3D generator, which we fine-tune to produce a complete mesh in the observed orientation while keeping it centered and scalenormalized. Scene-aware position queries fuse instance-level and global scene features from the geometry encoder to estimate translation and scale, which place each generated mesh in the shared observation-aligned scene frame.

We train and evaluate SceneReGen on 3D-FUTURE (Fu et al. 2021) (Figure 3 and Table 1). Figure 1 additionally presents actual model outputs on autonomous-driving and embodied-AI scene images, illustrating the potential scope of complete object meshes in their observed orientations and shared-frame scene assembly beyond the indoor-furniture benchmark.

Our contributions are summarized as follows:

• We formulate an object-centric generative reconstruction framework for single-image 3D scene reconstruction, where generated object assets serve as reconstruction primitives for scene recovery.

• We identify the representation gap between object generation and scene reconstruction and propose selective pose factorization, incorporating orientation into generated geometry while estimating translation and scale from scene evidence.

• We develop SceneReGen, a geometry-aware generative reconstruction framework that achieves superior scenelevel accuracy and competitive object-level quality on

3D-FUTURE.

## 2 Related Work

## 3D Scene Reconstruction

Traditional multi-view stereo and structure-from-motion pipelines such as COLMAP (Schoenberger and Pollefeys 2016) and PMVS (Furukawa and Ponce 2009) rely on sparse feature matching, yet often produce incomplete geometry in textureless or occluded areas. Feed-forward networks like DUSt3R (Wang et al. 2024) and VGGT (Wang et al. 2025a) accelerate reconstruction via direct regression, but still struggle with geometric sparsity and missing content. To address these limitations, VGGT-Ω (Wang et al. 2026) builds upon the versatile geometric encoder VGGT, which bridges 2D visual observations with 3D geometric cues by modeling cross-view topological and spatial relationships, and further introduces register attention to distill compact, high-level geometric features.

## 3D Object Generation

The rapid progress of generative models (Ho, Jain, and Abbeel 2020; Song, Meng, and Ermon 2020; Papamakarios et al. 2021; Wu et al. 2024) has significantly advanced 3D object generation. Early approaches typically distilled 2D generative priors from pre-trained text-to-image difusion models to produce high-fidelity 3D representations. More recently, large-scale frameworks (Li et al. 2025; Zhao et al. 2023, 2025; Xiang et al. 2025; Zadaianchuk et al. 2026) have pushed the state of the art by conditioning on either text or image inputs, enabling versatile and realistic 3D asset generation. To enhance multi-view consistency and alleviate the ambiguity of single-view inputs, another line of work (Liu et al. 2023; Shi et al. 2024, 2023; Huang et al. 2025c; Voleti et al. 2024) employs pre-trained image or video difusion models to generate consistent multi-view images, which are then fused into a 3D representation. Notably, these methods typically generate objects in a canonical coordinate space, without considering their global placement or orientation within a scene.

## Generative Reconstruction

Generative reconstruction leverages powerful generative priors to enhance 3D reconstruction from limited observations, and recent eforts in this direction can be broadly categorized into two paradigms: object-level and scene-level generative reconstruction.

Object-level generative reconstruction. Reconstructionguided 3D generative approaches (Chang et al. 2025; Lin et al. 2026; Huang et al. 2025a) have been proposed to combine the strength of difusion priors with robust reconstruction networks. For instance, ReconviGen and Mix3R employ strong reconstruction backbones (Wang et al. 2025a,b) to constrain the denoising process, achieving single-view 3D reconstructions with high-fidelity geometry and texture. Despite their impressive performance, these methods operate in a canonical coordinate space and assume known camera poses, making them ill-suited for pose-free inputs—a common scenario in real-world captures.

Scene-level generative reconstruction. Recent scenelevel generative reconstruction (Siddiqui et al. 2026; Yao et al. 2025; Liu, Tai, and Tang 2025) incorporate 3D generative priors (Zhao et al. 2025; Mo et al. 2023) to refine object details or jointly predict objects along with their spatial placements (rotation, translation, and scale), conditioned on reconstruction cues (Wang et al. 2025a). However, jointly optimizing all spatial parameters within a single framework inevitably introduces parameter coupling, which often leads to scene-level misalignments.

Notably, the above methods share a common limitation: generation is performed in a canonical space, while reconstruction is conducted in the world coordinate system. In contrast, we explicitly decouple rotation from translation and scale, which rotation is predicted by the generation branch, translation and scale are estimated by the reconstruction branch.

## 3 Method

Our goal is to reconstruct complete 3D object assets and their scene-level placements from a single image. SceneReGen combines rotation-aware object generation with scene-level position reconstruction. We formulate the problem in Sec. 3, describe object-wise generation in Sec. 3, and present scene assembly through translation and scale estimation in Sec. 3.

## Problem Definition

Given a single scene image I containing n object instances and the dataset-provided ground-truth instance masks M = $\{ \mathbf { M } _ { 1 } , \ldots , \mathbf { M } _ { n } \}$ , our framework reconstructs a complete 3D scene in a shared observation-aligned scene frame. Specifically, this frame is shared by all reconstructed objects and their estimated 3D positions, and inter-object spatial relationships are required to align with the object configuration observed in I.

For each instance i, the generation branch produces a centered, scale-normalized mesh $O _ { \mathrm { p o s e - a w a r e } } ^ { i } ,$ in which the target orientation observed in I is encoded directly by the mesh vertices. The position branch then estimates a translation $\mathbf { t } _ { i } \in \mathbb { R } ^ { 3 }$ and a positive isotropic scale $s _ { i } \in \mathbb { R } _ { > 0 }$ . Applying the estimated scale and translation to the pose-aware meshes places all reconstructed objects in the shared observationaligned scene frame, yielding the final 3D scene. Thus, rotation is represented by the generated geometry rather than regressed as a separate placement parameter.

## Rotation-Aware 3D Generation

Using these instance masks, we isolate each object and its corresponding mask from the original scene image I, obtaining object-specific RGB–mask pairs $\{ ( \mathbf { I } _ { o } ^ { i } , \mathbf { M } _ { o } ^ { i } ) \bar  \} _ { i = 1 } ^ { n } .$ As illustrated in Figure 2, the rotation-aware generation branch processes each pair independently to generate a complete pose-aware 3D mesh.

Occlusion Augmentation. Since objects in real scenes are frequently occluded, directly using partial references often leads to incomplete generation. To enable the model to produce complete objects under such partial observations, we simulate occlusions during training by randomly placing 5 square masks that collectively cover 20% of the foreground pixels-a configuration chosen to reflect commonly encountered occlusion levels in real-world data, which encourage robust completion while preserving enough structural cues for learning-and filling the masked regions with random pixel values uniformly sampled from [80, 180], a broad range chosen for occlusion diversity.

Geometry Condition. After occlusion augmentation, we denote the processed object patch by $\tilde { \mathbf { I } } _ { o } ^ { i }$ and feed it into a geometry encoder G. In our implementation, G is instantiated with VGGT-Ω (Wang et al. 2026), a feed-forward reconstruction model built upon VGGT (Wang et al. 2025a). VGGT-Ω uses learned registers to aggregate scene information into a compact representation and introduces register attention for eficient information exchange through these registers. We select it because its geometry-aware representation provides global structural and spatial cues that are directly relevant to recovering an object’s shape and observed orientation from a partial RGB observation.

For each object, we use VGGT-Ω to extract a sequence of geometry-aware features:

$$
\begin{array} { r } { \mathcal { F } _ { o b j } ^ { i } = \mathcal { G } ( \tilde { \mathbf { I } } _ { o } ^ { i } ) , } \end{array}\tag{1}
$$

where $\mathcal { F } _ { o b j } ^ { i } \in \mathbb { R } ^ { N \times C }$ , and N and $C$ denote the sequence length and channel dimension, respectively. To transform this feature sequence into a compact condition for object generation, we follow the query-based interaction in (Chang et al. 2025): learnable shape queries progressively aggregate the structural and orientation-related information through multilayer cross-attention,

$$
C _ { p o s e } ^ { i } = \mathrm { A t t e n t i o n } ( Q _ { s h a p e } , \mathcal { F } _ { o b j } ^ { i } , \mathcal { F } _ { o b j } ^ { i } ) ,\tag{2}
$$

where $Q _ { s h a p e } \in \mathbb { R } ^ { M \times C }$ denotes the learnable shape queries and $\mathcal { F } _ { o b j } ^ { i }$ serves as both the key and value. The resulting

![](images/dcecb1be3038ee0c573e52bf3e4e688ae3737f29c540dd442875a3c410322522.jpg)  
Figure 2: Overview of SceneReGen. Given a scene image and ground-truth instance masks, the rotation-aware generation branch uses object-level geometric cues to generate a complete mesh for each instance in its observed orientation, followed by texture generation. The scene-level reconstruction branch combines object- and scene-level cues with position queries to estimate translation and scale and assemble the objects in a shared observation-aligned scene frame.

$C _ { p o s e } ^ { i }$ distills the geometry cues relevant to the current object and serves as the condition for the subsequent rotation-aware generation process. We omit the instance superscript i below for brevity.

Rotation-aware Shape Generation. Given the rotationaware representation $C _ { p o s e }$ as the conditional input, we aim to synthesize a 3D shape using a DiT-based difusion model, ensuring that the generated object’s orientation is aligned with the input observation.

To incorporate the 3D spatial cues encoded in $C _ { p o s e } ,$ we augment each standard DiT block from (Hunyuan3D et al. 2025) with an additional cross-attention layer, which serves as an efective interface to inject geometric information into the generation process.

We now detail the formulation of the augmented DiT block, i.e., the shape-DiT illustrated in Figure 2. For each transformer block indexed by $k ,$ let $h _ { k }$ denote the input latent representation of the k-th block. We first apply Layer Normalization to $h _ { k }$ , and then perform a cross-attention operation conditioned on $C _ { p o s e }$ to incorporate the 3D rotation information, expressed as:

$$
\hat { h } _ { k } = h _ { k } + \mathrm { A t t e n t i o n } ( \mathrm { L N } ( h _ { k } ) , C _ { p o s e } , C _ { p o s e } ) .\tag{3}
$$

Subsequently, the output $\hat { h } _ { k }$ is fed into the standard transformer block, which consists of a layer normalization followed by a feed-forward network. The output of the feedforward network is then added back to $\hat { h } _ { k }$ via a residual connection, yielding the input latent for the next block $h _ { k + 1 }$ This process can be written as:

$$
h _ { k + 1 } = \hat { h } _ { k } + \mathrm { F F N } ( \mathrm { L N } ( \hat { h } _ { k } ) ) .\tag{4}
$$

Through the above process, the model dynamically attends to relevant orientation cues at each denoising step, guided by the rotation-aware condition $C _ { p o s e } ,$ thereby enabling the generation of a rotation-aware 3D mesh. Subsequently, we use a 3D texture generation method to generate a fully textured 3D asset: we generate multi-view images of $\tilde { \mathbf { I } } _ { o } ^ { i }$ and extract geometric features from $\mathcal { O } _ { \mathrm { p o s e - a w a r e } } ^ { i } .$ , then feed both into a diffusion model to synthesize texture maps, which are finally unwrapped to UV space to render final textured objects.

## Scene-Level Reconstruction

After generating the rotation-aware object representations, the scene-level branch estimates their translation and scale. Unlike joint RTS estimation (Meng et al. 2026; Shi et al. 2026), this branch does not treat rotation as an independent placement output.

We employ the shared geometry encoder G (i.e., VGGT-Ω) to extract both scene- and object-level cues. The full scene image I yields $\mathcal { F } _ { s c e n e } .$ which captures holistic geometry, layout, and relative spatial context, while each object-centric crop $I _ { o } ^ { i }$ yields $\mathcal { F } _ { o b j } ^ { i }$ , which captures local structure and scale cues for the target object.

For each object i, learnable position queries $Q _ { p o s } ^ { i }$ retrieve instance-specific cues from $\mathcal { F } _ { o b j } ^ { i }$ while incorporating the global context in $\mathcal { F } _ { s c e n e }$ . We formulate the position-aware attention as:

$$
T S _ { p o s } ^ { i } = \mathrm { A t t e n t i o n } ( Q _ { p o s } ^ { i } , \mathcal { F } _ { o b j } ^ { i } , \mathcal { F } _ { s c e n e } ) ,\tag{5}
$$

where $T S _ { p o s } ^ { i }$ denotes the learned translation-and-scaling feature token for the i-th object, which directly encodes the final translation vector $\mathbf { t } _ { i } \in \mathbb { R } ^ { 3 }$ and isotropic scaling factor $s _ { i } \in \mathbb { R }$ . Leveraging these parameters, we scale and translate each rotation-aware object and place it in the final global scene space, expressed as follows:

$$
\mathcal { O } _ { \mathrm { f i n a l } } ^ { i } = s _ { i } \cdot \mathcal { O } _ { \mathrm { p o s e - a w a r e } } ^ { i } + \mathbf { t } _ { i } ,\tag{6}
$$

where $\mathcal { O } _ { \mathrm { p o s e - a w a r e } } ^ { i }$ denotes the rotation-aware object obtained from Sec. 3.

## Training Loss

The overall training objective consists of two terms. For the rotation-aware 3D generation (Sec. 3), we adopt a flowmatching objective and the training loss is defined as:

$$
\mathcal { L } _ { \mathrm { g e n } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t , c } \left[ \| v _ { \theta } ( z _ { t } , t , c ) - ( z _ { 0 } - \epsilon ) \| _ { 2 } ^ { 2 } \right] ,\tag{7}
$$

where $z _ { t } = ( 1 - t ) \epsilon + t z _ { 0 }$ defines the linear interpolation between the Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ and the groundtruth latent $z _ { \mathrm { 0 } } , v _ { \theta }$ is the predicted velocity field, $t \in \ [ 0 , 1 ]$ is the time step, and c is the input object image.

For the scene-level reconstruction (Sec. 3), we directly take the predicted $T S _ { p o s } ^ { i }$ from Eq. 5 as the translation and scaling parameters. The loss is then computed as the L1 distance between it and its ground-truth counterpart, expressed as:

$$
\mathcal { L } _ { \mathrm { p o s } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Vert T S _ { p o s } ^ { i } - \mathbf { G T } ( T S _ { p o s } ^ { i } ) \Vert _ { 1 } ,\tag{8}
$$

where $\mathrm { G T } ( T S _ { p o s } ^ { i } )$ is the ground-truth vector for object i.

## 4 Experiments

## Experimental Settings

We adopt both the rotation-aware difusion backbone and the texture-generation model from (Hunyuan3D et al. 2025). We use the AdamW optimizer with a cosine learning rate scheduler and a maximum learning rate of $1 \times 1 0 ^ { - 5 } .$ . The model is trained for 500k iterations on 96 Ascend 910B NPUs with a per-device batch size of 4. To stabilize gradient updates, we apply gradient clipping with a maximum norm of 1.0. For conditional generation, we employ classifier-free guidance (CFG) with a condition dropout probability of 0.1. We use VGGT-Ω as our geometry encoder.

Dataset. Following standard practices in 3D scene generation, where most prior works (Meng et al. 2026; Xiang et al. 2026; Xia et al. 2026) conduct evaluations on the 3D-FUTURE dataset (Fu et al. 2021), we utilize this dataset to guarantee fair and rigorous comparisons. It contains 14,761 training scenes and 5,479 test scenes. Each scene consists of multiple objects, each paired with a rendered image and the corresponding ground-truth segmentation masks. Given the enormous size of this dataset, we follow the sampling strategy proposed in (Ardelean, Özer, and Egger 2025). We randomly sample 150 scenes from 3D-FUTURE to form our test subset for evaluating all competing methods.

Evaluation Metrics. Following prior work (Meng et al. 2026), we uniformly sample point clouds from the generated and ground-truth mesh surfaces. We use FilterReg (Gao and Tedrake 2019) to align them, as it provides faster and more accurate registration than the traditional Iterative Closest Point (ICP) algorithm. To evaluate reconstruction quality, we adopt two metrics at both the scene and object levels: Chamfer Distance (CD) measures the bidirectional geometric similarity between the generated and ground-truth point clouds, while F-Score assesses both the accuracy of predicted surface points and the coverage of the ground-truth surface. Additionally, we employ Volumetric Intersection over Union of bounding boxes $( \mathrm { I o } \dot { \mathrm { U } } _ { B } )$ to evaluate the accuracy of spatial locations and bounding-box overlaps with respect to the ground-truth scenes.

Evaluation Baselines. We compare our method with several state-of-the-art approaches for 3D scene generation and reconstruction. The selected baselines fall into two main categories: per-object optimization methods and fullscene reconstruction methods. Among the per-object optimization baselines, Gen3DSR (Ardelean, Özer, and Egger 2025), SAM 3D (Team et al. 2025), and ShapeR (Siddiqui et al. 2026) first complete the geometry of individual objects via difusion-based generation and then arrange the reconstructed objects in the scene according to a predefined scene layout or guidance derived from depth and segmentation maps. Among the full-scene reconstruction methods, MIDI (Huang et al. 2025b) employs a multi-instance attention mechanism to predict all object meshes and their positional parameters. SceneGen (Meng et al. 2026) leverages an end-to-end feed-forward network to generate the geometry, texture, and relative positions of multiple objects simultaneously. TRELLIS.2 (Xiang et al. 2026) directly synthesizes all objects in the scene within a field-free sparse voxel structure.

## Qualitative and Quantitative Comparisons

Qualitative Comparison. Figure 3 presents a qualitative comparison of diferent methods for reconstructing a 3D scene from a single reference image. We organize the visible diferences along three attributes. Object completeness. In the shown examples, the MIDI outputs contain visible mesh holes, while parts of the SceneGen reconstructions appear as black regions (see green boxes); SceneReGen retains more complete object surfaces. Orientation consistency. Compared with the corresponding ground-truth assets, MIDI reconstructs the table in the second row and the chair in the third row with noticeable orientation discrepancies, as highlighted by the blue boxes. By contrast, SceneRe-Gen more closely matches the ground-truth orientations of both objects in these examples. Relational placement. In the first row, SceneGen changes the relative spacing among the furniture compared with the GT assets. Besides, across the lower multi-view panels, SceneReGen better preserves the relational placement between the chairs and the table and exhibits fewer visible intersections than the comparison outputs highlighted by red boxes.

Quantitative Comparison. Table 1 reports the quantitative results for the evaluated scene generation and reconstruction methods. SceneReGen achieves the best CD-S, F-Score-S, and IoU<sub>B</sub> among the evaluated methods. At the object level, it ties SceneGen for the best CD-O and ranks second on F-Score-O. These results show that the complete SceneReGen system improves scene-level reconstruction and bounding-box alignment while maintaining competitive object-level geometric quality.

![](images/dd7aa0ba2a7d8c2c686dff056f093c8a58ec565d935ffbe711d201a62afd1c69.jpg)  
Figure 3: Qualitative comparison across diferent methods. The panels below the dashed line show results from diferent viewpoints given the same reference image. Red boxes highlight methods afected by wrong relational placement (e.g., geometric intersections), blue boxes indicate inconsistencies in object orientation, and green boxes reveal incomplete object geometry.

Cross-Domain Application Potential. Figure 1 connects object-centric scene reconstruction to two application domains in which explicit 3D object representations are valuable. For autonomous driving, complete, orientation-aware vehicle meshes and their scene-level placement provide an asset-level representation that could complement perception outputs for scenario reconstruction and simulation. For embodied AI, reconstructing a complete manipulable object, such as the bottle shown in Figure 1, can provide object geometry and pose cues for interaction planning and embodied simulation. Together, these examples illustrate how SceneRe-Gen could connect perception with downstream systems through an asset-centric 3D representation: reconstructed objects remain individually accessible for simulation or interaction, while scene-level placement retains the spatial context needed for reasoning.

<table><tr><td rowspan="2">Method</td><td colspan="2">Scene-Level</td><td colspan="2">Object-Level</td><td rowspan="2"> $\mathbf { I o U } _ { B } \uparrow$ </td></tr><tr><td>CD-S↓</td><td>F-Score-S ↑</td><td>CD-O↓</td><td>F-Score-O ↑</td></tr><tr><td>MIDI</td><td>0.067</td><td>53.95</td><td>0.051</td><td>59.14</td><td>0.186</td></tr><tr><td>Gen3DSR</td><td>0.055</td><td>61.18</td><td>0.095</td><td>41.69</td><td>0.227</td></tr><tr><td>SceneGen</td><td>0.017</td><td>81.49</td><td>0.031</td><td>71.20</td><td>0.410</td></tr><tr><td>ShapeR</td><td>0.045</td><td>52.33</td><td>0.104</td><td>41.80</td><td>0.109</td></tr><tr><td>SAM 3D</td><td>0.048</td><td>66.71</td><td>0.042</td><td>68.84</td><td>0.152</td></tr><tr><td>TRELLIS.2</td><td>0.065</td><td>62.75</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>0.009</td><td>89.50</td><td>0.031</td><td>68.95</td><td>0.536</td></tr></table>

Table 1: Quantitative comparison on the 3D-FUTURE evaluation subset. CD-S and F-Score-S measure scene-level geometry, CD-O and F-Score-O measure object-level geometry, and $\mathrm { I o U } _ { B }$ measures 3D bounding-box alignment. Up and down arrows indicate whether higher or lower values are better, respectively. The best results are in bold, the second-best are underlined.

## Ablation Study

Ablation Study on Geometry Encoder and Occlusion Augmentation. We conduct ablation experiments to compare the geometry encoders and occlusion augmentation settings in Table 2. Variants (a)–(c) compare a generic 2D visual encoder, DINOv2, with two geometry-grounded encoders, VGGT and VGGT-Ω, while keeping $\bar { M } _ { \mathrm { o c c } }$ disabled. Both geometry-grounded encoders improve all four reconstruction metrics over $\mathrm { D I N O v } 2 ,$ suggesting that explicitly geometry-grounded representations provide more efective conditioning cues for our reconstruction task than generic 2D visual features. VGGT-Ω further improves all four metrics over VGGT and achieves the strongest performance among the evaluated encoders under the same augmentation setting.

Variants (c) and (d) retain VGGT-Ω and difer in the use of $M _ { \mathrm { o c c } }$ . Adding the augmentation reduces CD-S from 0.010 to 0.009 and CD-O from 0.045 to 0.031, while increasing F-Score-S from 87.64 to 89.50 and F-Score-O from 60.52 to 68.95. These consistent improvements support including $M _ { \mathrm { o c c } }$ in the final configuration under our evaluation protocol.

## Limitations

Our method has the following limitations. (i) Our texture synthesis relies on the native module of Hunyuan3D (Hunyuan3D et al. 2025), which exhibits limited robustness under challenging visual conditions (e.g., heavy occlusions and extreme lighting), thereby restricting high-fidelity texture generation in these scenarios. (ii) Our approach sufers from performance degradation when provided with low-resolution or blurry scene inputs, mainly due to incomplete reconstruction cues that lead to inaccurate estimation of translation and scale.

<table><tr><td rowspan="3">Variant</td><td rowspan="3"> $\mathcal { G }$ </td><td rowspan="3"> $M _ { \mathrm { o c c } }$ </td><td>Scene-Level</td><td>Object-Level</td></tr><tr><td></td><td>CD-S ↓ F-Score-S ↑ CD-O ↓ F-Score-O ↑</td></tr><tr><td>0.032 75.32</td><td>0.083 49.21</td></tr><tr><td>(a) (b)</td><td>DINOv2 VGGT</td><td>× × 0.012</td><td>85.21</td><td>0.058 59.31</td></tr><tr><td>(c)</td><td>VGGT-Ω</td><td>X</td><td>0.010 87.64</td><td>0.045 60.52</td></tr><tr><td></td><td></td><td></td><td>89.50</td><td></td></tr><tr><td>(d) Ours VGGT-Ω</td><td></td><td>√</td><td>0.009</td><td>0.031 68.95</td></tr></table>

Table 2: Ablation study on the geometry encoder $\mathcal { G }$ and the occlusion augmentation strategy $M _ { \mathrm { o c c } } . ~ M _ { \mathrm { o c c } }$ indicates whether our occlusion augmentation is applied during training. Best results are in bold.

## 5 Conclusion

We presented SceneReGen, an object-centric framework that turns rotation-aware object generation into single-image 3D scene reconstruction through selective pose factorization. Under this factorization, each object’s observed orientation is represented directly in a centered, scale-normalized generated mesh, while translation and scale are estimated from instance-level and global scene cues. Geometry-aware shape queries condition a pretrained 3D generator to complete individual objects, and scene-aware position queries assemble the generated assets in a shared observation-aligned scene frame. On the 3D-FUTURE evaluation subset, SceneReGen achieves the best scene-level CD, scene-level F-Score, and 3D bounding-box IoU among the evaluated methods, ties the best object-level CD, and ranks second in object-level F-Score. Together, these results indicate improved completescene reconstruction and spatial alignment while retaining competitive object-level geometry. The cross-domain outputs further illustrate the potential of this asset-centric representation for autonomous-driving scenario reconstruction and embodied-AI interaction.

## References

Ardelean, A.; Özer, M.; and Egger, B. 2025. Gen3dsr: Generalizable 3d scene reconstruction via divide and conquer from a single view. In 2025 International Conference on 3D Vision (3DV), 616–626. IEEE.

Chang, J.; Ye, C.; Wu, Y.; Chen, Y.; Zhang, Y.; Luo, Z.; Li, C.; Zhi, Y.; and Han, X. 2025. ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation. arXiv preprint arXiv:2510.23306.

Fu, H.; Jia, R.; Gao, L.; Gong, M.; Zhao, B.; Maybank, S.; and Tao, D. 2021. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision, 129(12): 3313– 3337.

Furukawa, Y.; and Ponce, J. 2009. Accurate, dense, and robust multiview stereopsis. IEEE transactions on pattern analysis and machine intelligence, 32(8): 1362–1376.

Gao, W.; and Tedrake, R. 2019. FilterReg: Robust and Eficient Probabilistic Point-Set Registration Using Gaussian Filter and Twist Parameterization. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11087–11096.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Huang, Z.; Boss, M.; Vasishta, A.; Rehg, J. M.; and Jampani, V. 2025a. SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16860–16870.

Huang, Z.; Guo, Y.-C.; An, X.; Yang, Y.; Li, Y.; Zou, Z.- X.; Liang, D.; Liu, X.; Cao, Y.-P.; and Sheng, L. 2025b. Midi: Multi-instance difusion for single image to 3d scene generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23646–23657.

Huang, Z.; Guo, Y.-C.; Wang, H.; Yi, R.; Ma, L.; Cao, Y.- P.; and Sheng, L. 2025c. Mv-adapter: Multi-view consistent image generation made easy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 16377–16387.

Hunyuan3D, T.; Yang, S.; Yang, M.; Feng, Y.; Huang, X.; Zhang, S.; He, Z.; Luo, D.; Liu, H.; Zhao, Y.; et al. 2025. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442.

Li, Y.; Zou, Z.-X.; Liu, Z.; Wang, D.; Liang, Y.; Yu, Z.; Liu, X.; Guo, Y.-C.; Liang, D.; Ouyang, W.; et al. 2025. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lin, S.; Xue, Z.; Zhang, H.; An, L.; Li, D.; Jiao, S.; and Liu, Y. 2026. Mix3R: Mixing Feed-forward Reconstruction and Generative 3D Priors for Joint Multi-view Aligned 3D Reconstruction and Pose Estimation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 1–12.

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, 9298–9309.

Liu, X.; Tai, Y.-W.; and Tang, C.-K. 2025. Agentic 3d scene generation with spatially contextualized vlms. arXiv preprint arXiv:2505.20129.

Meng, Y.; Wu, H.; Zhang, Y.; and Xie, W. 2026. Scenegen: Single-image 3d scene generation in one feedforward pass. In 2026 International Conference on 3D Vision (3DV), 543– 553. IEEE.

Mo, S.; Xie, E.; Chu, R.; Hong, L.; Niessner, M.; and Li, Z. 2023. Dit-3d: Exploring plain difusion transformers for 3d shape generation. Advances in neural information processing systems, 36: 67960–67971.

Papamakarios, G.; Nalisnick, E.; Rezende, D. J.; Mohamed, S.; and Lakshminarayanan, B. 2021. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57): 1–64.

Schoenberger, J. L.; and Pollefeys, M. 2016. Colmapstructure-from-motion and multi-view stereo. In Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on.

Shi, R.; Chen, H.; Zhang, Z.; Liu, M.; Xu, C.; Wei, X.; Chen, L.; Zeng, C.; and Su, H. 2023. Zero123++: a single image to consistent multi-view difusion base model. arXiv preprint arXiv:2310.15110.

Shi, Y.; Li, W.; Wang, Z.; Li, H.; Chen, X.; Tan, P.; and Zhang, L. 2026. Scenemaker: Open-set 3d scene generation with decoupled de-occlusion and pose estimation model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27146–27156.

Shi, Y.; Wang, P.; Ye, J.; Mai, L.; Li, K.; and Yang, X. 2024. Mvdream: Multi-view difusion for 3d generation. In International conference on learning representations, volume 2024, 39838–39859.

Siddiqui, Y.; Frost, D.; Aroudj, S.; Avetisyan, A.; Howard-Jenkins, H.; DeTone, D.; Moulon, P.; Wu, Q.; Li, Z.; Straub, J.; et al. 2026. ShapeR: Robust Conditional 3D Shape Generation from Casual Captures. arXiv preprint arXiv:2601.11514.

Song, J.; Meng, C.; and Ermon, S. 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502.

Team, S. D.; Chen, X.; Chu, F.-J.; Gleize, P.; Liang, K. J.; Sax, A.; Tang, H.; Wang, W.; Guo, M.; Hardin, T.; Li, X.; Lin, A.; Liu, J.; Ma, Z.; Sagar, A.; Song, B.; Wang, X.; Yang, J.; Zhang, B.; Dollár, P.; Gkioxari, G.; Feiszli, M.; and Malik, J. 2025. SAM 3D: 3Dfy Anything in Images.

Voleti, V.; Yao, C.-H.; Boss, M.; Letts, A.; Pankratz, D.; Tochilkin, D.; Laforte, C.; Rombach, R.; and Jampani, V. 2024. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video difusion. In European Conference on Computer Vision, 439–457. Springer.

Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.; and Novotny, D. 2025a. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, 5294–5306.

Wang, J.; Chen, M.; Zhang, S.; Karaev, N.; Schönberger, J.; Labatut, P.; Bojanowski, P.; Novotny, D.; Vedaldi, A.; and Rupprecht, C. 2026. VGGT-Ω. arXiv preprint arXiv:2605.15195.

aligned latent representation. Advances in neural information processing systems, 36: 73969–73982.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2024. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20697–20709.

Wang, Y.; Zhou, J.; Zhu, H.; Chang, W.; Zhou, Y.; Li, Z.; Chen, J.; Pang, J.; Shen, C.; and He, T. 2025b. π<sup>3</sup>: Scalable Permutation-Equivariant Visual Geometry Learning. arXiv e-prints, arXiv–2507.

Wu, K.; Liu, F.; Cai, Z.; Yan, R.; Wang, H.; Hu, Y.; Duan, Y.; and Ma, K. 2024. Unique3D: High-Quality and Eficient 3D Mesh Generation from a Single Image. In Globerson, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C., eds., Advances in Neural Information Processing Systems, volume 37, 125116–125141. Curran Associates, Inc.

Xia, J.; Duan, Z.; Hengel, A. v. d.; and Liu, L. 2026. Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors. arXiv preprint arXiv:2603.18782.

Xiang, J.; Chen, X.; Xu, S.; Wang, R.; Lv, Z.; Deng, Y.; Zhu, H.; Dong, Y.; Zhao, H.; Yuan, N. J.; et al. 2026. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14419–14429.

Xiang, J.; Lv, Z.; Xu, S.; Deng, Y.; Wang, R.; Zhang, B.; Chen, D.; Tong, X.; and Yang, J. 2025. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 21469–21480.

Yao, K.; Zhang, L.; Yan, X.; Zeng, Y.; Zhang, Q.; Xu, L.; Yang, W.; Gu, J.; and Yu, J. 2025. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics (TOG), 44(4): 1–19.

Zadaianchuk, A.; Barcellona, L.; Schuenemann, L.; Gumbsch, C.; Wang, Z.; Irshad, M. Z.; Despinoy, F.; Aljundi, R.; Gavves, S.; and Zakharov, S. 2026. Reconstruction by Generation: 3D Multi-Object Scene Reconstruction from Sparse Observations. arXiv:2604.27106.

Zhang, B.; Tang, J.; Niessner, M.; and Wonka, P. 2023. 3dshape2vecset: A 3d shape representation for neural fields and generative difusion models. ACM Transactions On Graphics (TOG), 42(4): 1–16.

Zhang, L.; Wang, Z.; Zhang, Q.; Qiu, Q.; Pang, A.; Jiang, H.; Yang, W.; Xu, L.; and Yu, J. 2024. Clay: A controllable largescale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4): 1–20.

Zhao, Z.; Lai, Z.; Lin, Q.; Zhao, Y.; Liu, H.; Yang, S.; Feng, Y.; Yang, M.; Zhang, S.; Yang, X.; et al. 2025. Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202.

Zhao, Z.; Liu, W.; Chen, X.; Zeng, X.; Wang, R.; Cheng, P.; Fu, B.; Chen, T.; Yu, G.; and Gao, S. 2023. Michelangelo: Conditional 3d shape generation based on shape-image-text

## 6 Experimental Details

## Training Data Preparation

To endow our rotation-aware generative model with strong generalization across autonomous driving and embodied AI scenarios, we assemble and preprocess a multi-source dataset consisting of Objaverse, 3D-FUTURE and Mesh-Fleet. Specifically, 3D-FUTURE supplies high-quality indoor furniture assets; MeshFleet provides diverse 3D vehicle models for driving scenes; Objaverse covers commonplace daily commodities such as bottles and cups, which are essential for embodied interactive environments.

To guarantee overall dataset quality, we first render the objects into multiple views to filter out unqualified instances—including assets with degraded visual appearance, oversimplified geometric structures, and clustered multiobject messy meshes. Afterwards, we conduct a watertightness inspection over all mesh samples. Non-watertight surfaces severely hinder the training convergence of difusion VAEs; Therefore, we eliminate flawed samples by measuring the Chamfer Distance between raw meshes and their VAE reconstructions, discarding instances with excessive reconstruction error.

The proposed multi-stage filtering pipeline yields a refined high-quality dataset containing around 25K object meshes. Following standard protocols from prior work (Xiang et al. 2026), we render 24 multi-view RGB images for each object with randomly sampled azimuth and elevation angles. Critically, we align ground-truth meshes with the camera viewpoints of paired input images to maintain consistency with our rotation-aware generative optimization objective.

## Evaluation Details

We adopt dataset-specific evaluation standards across all test benchmarks, with elaborate configuration details illustrated as follows. Consistent with previous literature (Meng et al. 2026), geometric quality is measured within a normalized canonical 3D space bounded by $x , y , z \in [ - 1 , 1 ] .$ Before metric calculation, rigid point-cloud registration is performed to align each generated mesh with its corresponding ground truth. For the 3D-FUTURE benchmark (Fu et al. 2021), all quantitative metrics are computed under the dataset’s native geometric scale. We uniformly sample 10,000 points from both generated meshes and ground-truth meshes, and the distance threshold for F-Score calculation is fixed at 0.1.

## Inference Details

During inference, we crop individual object regions based on provided instance masks and apply center padding to resize all input images to a fixed resolution of 512×512. For conditional generation, we set the classifier-free guidance (CFG) dropout probability to 0.1. All experiments share identical inference hyperparameters: we employ 50 denoising sampling steps with a CFG scale of 3.0 to output the final latent representations for mesh generation.

## 7 More Qualitative Results

This section presents additional qualitative visualizations on the test split of the 3D-FUTURE dataset (Fu et al. 2021). For each furniture instance reconstructed by our method, we render five distinct viewing angles to fully display the complete rotation-aware generated mesh produced by SceneReGen.

In Figure 4, these comprehensive multi-view comparisons intuitively validate that our method maintains stable geometric integrity and orientation consistency under arbitrary camera viewpoints.

## 8 Limitations and Future Work

## Limitations

Our SceneReGen framework has three main limitations: First, directly using the of-the-shelf Hunyuan3D texture module lacks occlusion-aware optimization, leading to blurriness and color distortion on heavily occluded objects. Second, reconstruction performance degrades significantly with low-resolution or blurry input images due to the loss of finegrained geometric and structural cues. Third, the translationand-scale branch predicts placement independently without inter-object collision regularization, frequently resulting in mesh penetration and unnatural overlaps in dense indoor scenes.

## Future Work

We outline three directions for future research: First, we will design an occlusion-adaptive texture refinement branch utilizing auxiliary occlusion masks features to improve texture quality. Second, we plan to train on larger and more diverse multi-domain 3D scene datasets to boost generalization and alleviate data distribution bias. Third, we will integrate explicit physical priors and collision-aware regularization into the position estimation module to prevent mesh penetration and ensure natural spatial layouts.

![](images/ee8bef441690e05aaa12ab0b8b7494598cba865103ffe69e5b72314e00e79c4d.jpg)  
Figure 4: More qualitative results from our method.