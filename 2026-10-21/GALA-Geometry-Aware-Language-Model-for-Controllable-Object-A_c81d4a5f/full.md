# GALA: Geometry-Aware Language Model for Controllable Object Arrangement

CHUCHENG XIANG, University of Science and Technology of China, China and Tencent, China   
RUNZE WANG, University of Science and Technology of China, China and Tencent, China   
ZHI DENG<sup>∗</sup>, Tencent, China   
RUCHAO BAO, University of Science and Technology of China, China and Tencent, China   
YUANWEI ZHANG, Tsinghua University, China and Tencent, China   
YUXUAN XIE, Tencent, China   
HANLIU WANG, Tencent, China   
LIANGZHEN FEI, University of Science and Technology of China, China and Tencent, China   
WENZHENG WU, University of Science and Technology of China, China and Tencent, China   
CHENG WAN, Hong Kong University of Science and Technology, China and Tencent, China   
PEIFENG LI, University of Science and Technology of China, China and Tencent, China   
ZHONGYUAN LIU, Tencent, China   
LIGANG LIU<sup>∗</sup>, University of Science and Technology of China, China

Geometry view OBB view  
![](images/c4e3d63599b74bc754deec96bd726c8cbd1806bb37cf44a35f199e37b96e61dc.jpg)

Fig. 1. Left: input objects and fine-grained placement instructions. Middle: plausible layouts that follow the instructions. Right: zoom-in views show how GALA goes beyond oriented bounding boxes by perceiving geometry to produce correct support, containment, and collision-free placements.

<sup>∗</sup>Corresponding authors.

Authors’ Contact Information: Chucheng Xiang, University of Science and Technology of China, Hefei, China and Tencent, Shenzhen, China, xcc2020@mail.ustc.edu.cn; Runze Wang, University of Science and Technology of China, Hefei, China and Ten cent, Shenzhen, China, runzewang@mail.ustc.edu.cn; Zhi Deng, Tencent, Shenzhen, China, abeldeng@tencent.com; Ruchao Bao, University of Science and Technology of China, Hefei, China and Tencent, Shenzhen, China, iambrc@mail.ustc.edu.cn; Yuan wei Zhang, Tsinghua University, Shenzhen, China and Tencent, Shenzhen, China, yuanwei-24@mails.tsinghua.edu.cn; Yuxuan Xie, Tencent, Shenzhen, China, yuxuan. xie@hotmail.com; Hanliu Wang, Tencent, Shenzhen, China, wanghanliu97@gmail.com; Liangzhen Fei, University of Science and Technology of China, Hefei, China and Tencent, Shenzhen, China, flzcn6@mail.ustc.edu.cn; Wenzheng Wu, University of Science and Technology of China, Hefei, China and Tencent, Shenzhen, China, wuwzh@mail.ustc.edu.cn; Cheng Wan, Hong Kong University of Science and Tech nology, Hong Kong, China and Tencent, Shenzhen, China, wanc7147@gmail.com; Peifeng Li, University of Science and Technology of China, Hefei, China and Tencent, Shenzhen, China, lipf6@mail.ustc.edu.cn; Zhongyuan Liu, Tencent, Shenzhen, China,

Reasoning about fine-grained placement constraints, such as support and containment, is essential for realistic 3D scene creation. It remains a significant challenge for existing methods that rely on simplified object representations like oriented bounding box (OBB), which are blind to the detailed geometry. To address this limitation, we introduce GALA, a geometry-aware language model with explicit point cloud perception for controllable object placement. Our model performs autoregressively by predicting the 6D pose of one target object at a time conditioned on the scene context and

lockliu@tencent.com; Ligang Liu, University of Science and Technology of China, Hefei, China, lgliu@ustc.edu.cn.

placement instruction. We utilize a two-stage training strategy: the first stage aligns geometric and textual modalities through pretraining tasks, and the second stage fine-tunes the model for placement. To facilitate this task, we also curate a multi-source scene dataset covering diverse room types and fine-grained furniture arrangements, which will be open-sourced for community benefit. Experiments on our benchmark show our model significantly outperforms baselines without explicit geometric reasoning. We further develop a multi-stage plan–place–verify agent pipeline that uses the model for text-driven scene generation.

Project Page: https://xccelephant.github.io/paper/gala/

CCS Concepts: • Computing methodologies → Artificial intelligence; Natural language processing; Computer graphics; Scene understanding.

Additional Key Words and Phrases: object placement, multimodal model, point cloud understanding, scene layout generation

## ACM Reference Format:

Chucheng Xiang, Runze Wang, Zhi Deng, Ruchao Bao, Yuanwei Zhang, Yuxuan Xie, Hanliu Wang, Liangzhen Fei, Wenzheng Wu, Cheng Wan, Peifeng Li, Zhongyuan Liu, and Ligang Liu. 2026. GALA: Geometry-Aware Language Model for Controllable Object Arrangement. In SIGGRAPH Asia 2026 Conference Papers (SA Conference Papers ’26), December 01–04, 2026, Kuala Lumpur, Malaysia. ACM, New York, NY, USA, 19 pages. https://doi. org/10.1145/3829340.3842289

## 1 Introduction

Generating realistic 3D scenes is important in game development, film production, architectural design, and embodied AI training, where virtual environments must be both physically plausible and semantically meaningful. Scene construction can be viewed as an iterative process of single-object placement: each step inserts a new object conditioned on the scene built so far. This single-object placement problem is the focus of this work. Placing a new object requires understanding both high-level semantic constraints, such as user instructions and object relationships, and low-level geometric constraints, such as collision avoidance, support, and containment.

However, most existing methods place objects only at the level of oriented bounding box (OBB): each object is represented by its center, size, and rotation. This abstraction is suficient for coarse arrangements, but it struggles with the geometric constraints in realistic scenes, as shown in Fig. 1. Three basic failure modes are common in practice. First, a chair placed under a desk must clear the desktop with its seat while avoiding the table legs. Second, small objects like books are often placed inside shelves whose internal cavities are invisible to the OBB. Third, cushions placed on curved surfaces, such as sofas, require reasoning about support heights that vary across the contact region. In all three cases, OBB-only methods cannot encode the fine-grained constraints, leading to either collisions (in OBB space), floating, or misalignments with the underlying geometry.

We view this limitation as primarily a geometry-awareness problem rather than an algorithmic one. A natural solution is to bring geometric information into the placement problem through a 3D representation of the objects and the scene. Among possible 3D representations, point clouds ofer a favorable trade-of: they are lightweight, easy to obtain, and uniform across asset libraries, and recent work on 3D multimodal LLMs [Hong et al. 2023; Mao et al.

2026; Xu et al. 2024] shows that LLMs can consume point clouds directly. Our core hypothesis is simple: LLM-based placement models augmented with point clouds can handle fine-grained arrangements beyond OBB reasoning.

Building on this hypothesis, we introduce GALA, a geometryaware placement model. The model is a multimodal LLM built on a strong LLM backbone, extended with a point cloud encoder and a projector that maps geometry features into the LLM embedding space. Given an existing scene, an object to be placed, and a placement instruction, the model predicts the 6D pose of the target object. The instruction specifies the intended relation and makes object insertion controllable. To make the point cloud tokens readable to the LLM, we adopt a two-stage training recipe: geometric pretraining aligns the projector, and task fine-tuning adapts the full model for object placement. Existing scene datasets, such as 3D-FRONT [Fu et al. 2021], contain noisy scenes and provide limited examples of finegrained placement. We therefore curate a multi-source scene dataset from 3D-FRONT, Imaginarium [Zhu et al. 2025], SpatialLM [Mao et al. 2026], and an artist-crafted set of 300 scenes, followed by filtering noisy layouts and adding fine-grained placement examples. Finally, to make our model easier to use and support automated scene generation, we develop a multi-stage agent pipeline that converts textual user prompts into complete 3D scenes through object planning, instruction generation, and geometry-aware placement.

We summarize our contributions as follows.

• We introduce a geometry-aware language model for controllable object pose prediction conditioned on placement instruction and point cloud geometry, enabling fine-grained placement beyond OBB-based reasoning.

• We curate a multi-source high-quality scene dataset, combining public scenes and artist-crafted scenes. The dataset is physically and visually verified, contains complex spatial relations, and will be open-sourced for community benefit.

• We develop a plan–place–verify agent system for automatic text-driven scene generation, integrating object planning, instruction generation, and geometry-aware placement.

## 2 Related Work

## 2.1 Data-driven Scene Synthesis

Early learning-based approaches represent an indoor scene as a set of oriented bounding boxes and learn the joint distribution of object classes and box parameters, including convolutional [Wang et al. 2019, 2018] or recursive [Li et al. 2019] priors, autoregressive models [Bucher and Armeni 2025; Feng et al. 2026; Paschalidou et al. 2021], and difusion-based models [Tang et al. 2024]. Other lines of work model scenes as graphs and predict pairwise relations [Gao et al. 2023; Lin and Mu 2024; Zhai et al. 2024, 2023; Zhou et al. 2019]. These methods reason at the OBB level, so they cannot express the geometric detail needed to slide a chair under a table or place a book on a specific shelf inside a cabinet. They also tie generation to the scene dataset distribution, which limits generalization to unseen room types.

![](images/8052519cc9d01688aaa28bcb11e3fc9beec9bb4cf7164d86b94361610722ca68.jpg)  
Fig. 2. Overview of the geometry-aware placement model. Given the scene boundary, the already-placed objects (with poses, sizes, semantic categories, and point clouds), the new object (with size, semantic category, and point cloud), and a placement instruction, the multimodal LLM outputs a numeric 6D pose. The model performs placement autoregressively, adding each predicted object to the scene context before predicting the next one.

## 2.2 LLM- and VLM-based Scene Layout

A recent line of work treats LLMs and VLMs as priors over indoor scenes. LayoutGPT [Feng et al. 2023] prompts an LLM with in-context examples to output object positions and sizes from a scene description, while LayoutVLM [Sun et al. 2025] leverages the visual abilities of modern VLMs to generate scenes from text. Holodeck [Yang et al. 2024] decomposes scene generation into LLMdriven planning, asset retrieval, and constraint-based placement. Because these model outputs are still coarse layouts, several methods add explicit optimization or solvers after the language-model stage. [Gumin et al. 2025; Wu et al. 2026] generate indoor scenes through LLM-generated programs followed by layout optimization. Co-Layout [Xiang et al. 2026] couples an LLM-driven agent work flow with grid-based integer programming to jointly optimize room layouts and furniture placements. FirePlace [Huang et al. 2025a] similarly uses VLM-generated constraints and a solver for object placement. Closer to our setting, PlaceIt3D [Abdelreheem et al. 2025] trains a 3D LLM for language-guided placement on reconstructed scanned scenes, but predicts a placement mask rather than a numeric 6D pose, and does not consider fine-grained containment relations. These methods show that pretrained foundation models carry useful priors over spatial common sense, but they still make geometric decisions with coarse boxes, constraints, or solvers. As a result, fine-grained placements that depend on object shape are not handled through native 3D geometric perception.

## 2.3 3D Multimodal Large Language Models

Recent 3D multimodal LLMs extend language models from imagetext reasoning to point-cloud and scene-level understanding. Early systems such as 3D-LLM [Hong et al. 2023] and PointLLM [Xu et al. 2024] show that LLMs can describe 3D content, answer questions, and follow instructions when supplied with 3D inputs. This direction has since expanded to more structured 3D tasks: 3D-LLaVA [Deng et al. 2025] studies general 3D instruction following, SegPoint [He et al. 2024] and Reason3D [Huang et al. 2025b] use LLMs for pointlevel segmentation from referring or reasoning queries, and SpatialLM [Mao et al. 2026] targets structured indoor scene understanding. Recent 3D VLM work further studies spatial reasoning itself, such as SpatialStack [Zhang et al. 2026b], which focuses on pointlanguage reasoning for 3D spatial relations. These works make 3D geometry accessible to foundation models, but their outputs are mostly language responses, masks, or structured scene descriptions rather than precise poses for inserting a new object into an existing scene. Our work uses the 3D multimodal formulation for object placement, where the model must map scene geometry, target object geometry, and a placement instruction to a numeric 6D pose.

![](images/ea8964cb083a2161d59bf0f4731e12c6342fc7af8e32cd386d079de83ce9b45e.jpg)  
Fig. 3. Dataset construction pipeline. Multi-source scenes are first cleaned by physical plausibility checks and VLM-based quality assessment. Each retained scene is parsed into a scene graph containing absolute room-location edges (abs), dependency edges (on, within, under), and proximity edges (next to). We form a floor-rooted scene tree T by selecting dependency edges with priority on > within > under > abs. We obtain the placement order $\pmb { \pi } = \mathbf { B } \mathbf { F } \mathbf { S } ( \mathcal { T } )$ ) by traversing from the floor root, sorting nodes at the same depth by edge priority and footprint area. For each placement step, the tree edge and nearby proximity relations are verbalized into an instruction paired with per-object point clouds and the target 6D pose.

## 3 Method

Problem Formulation. We study a single-object placement task in a bounded scene region. The scene region is described by a 2D footprint polygon $\mathcal { B } \subset \mathbb { R } ^ { 2 }$ , together with a floor point cloud $P _ { \mathcal { B } }$ sampled on the area it encloses. A scene consists of a set of existing object instances $O _ { \mathrm { e x i s t } } = \{ O _ { i } \} _ { i = 0 } ^ { N - 1 }$ . Each existing object $O _ { i }$ is associated with an identifier �, a semantic category string $c _ { i } ,$ an axis-aligned size $\pmb { s } _ { i } = \left( w _ { i } , d _ { i } , h _ { i } \right)$ , and a point cloud $P _ { i }$ . Once placed, it has a 6D pose $\pmb { p } _ { i } = ( x _ { i } , y _ { i } , z _ { i } , \phi _ { i } , \theta _ { i } , \psi _ { i } )$ , where $( x _ { i } , y _ { i } , z _ { i } )$ denotes the translation, and $( \phi _ { i } , \theta _ { i } , \psi _ { i } )$ denotes the roll, pitch, and yaw angles, respectively. Here we use a world frame with +� pointing right, +� pointing forward, and +� pointing up. By convention, an object faces the −� direction when $\psi _ { i } = 0 ;$ , and positive yaw values are defined counterclockwise. The object to be placed, $O _ { \mathrm { t a r } } .$ is specified by its category $c _ { \mathrm { t a r } } ,$ size $\pmb { s } _ { \mathrm { t a r } } .$ , and point cloud $P _ { \mathrm { t a r } } .$ , while the placement instruction � constrains the desired spatial relations to the scene and reference objects, for example “place the table against the west wall, in front ofthe sofa (id 0), and to the left ofthe wardrobe (id 3)”. We want to train a predictor $f _ { \theta }$ to output the 6D pose of the target object,

$$
\mathbf { \nabla } p _ { \mathrm { t a r } } \ = \ f _ { \theta } \big ( \ \mathcal { B } , \ P _ { \mathcal { B } } , \ O _ { \mathrm { e x i s t } } , \ c _ { \mathrm { t a r } } , \ s _ { \mathrm { t a r } } , \ P _ { \mathrm { t a r } } , \ I \big ) .\tag{1}
$$

## 3.1 Geometry-Aware Multimodal Architecture

As illustrated in Fig. 2, we instantiate �<sub>�</sub> as a multimodal LLM whose text backbone is a pretrained causal LLM and whose geometric branch is a 3D point cloud encoder E followed by a projector �.

Point cloud perception module. To achieve precise geometric perception of 3D scenes, we employ Utonia [Zhang et al. 2026a], a state-of-the-art, self-supervised point transformer model. Its pretraining on diverse 3D data is crucial, as it mitigates the need for domain-specific priors, allowing the model to capture scene-level scale and fine-grained object geometry. The encoder E takes a point cloud $P \in \mathbb { R } ^ { \bar { N _ { p } } \times C }$ with $N _ { p }$ input points and � channels (coordinates, color, and normals) as input, and returns a feature matrix $ { \mathcal { E } } ( P ) \in  { \mathbb { R } } ^ { M \times d _ { e } }$ , where � is the number of output point features, typically with $M < N _ { p }$ , and $d _ { e }$ is the feature dimension.

To avoid feature conflation from a single scene-level point cloud, we encode each object’s geometry independently. This gives the model separate point features for each instance.

Geometric-to-text projection. We use a projector to align geometry features with the text embedding space. The projector $g : \mathbb { R } ^ { d _ { e } } $ $\mathbb { R } ^ { d _ { h } }$ is parameterized by a two-layer MLP with GELU activation, where $d _ { h }$ is the LLM hidden dimension. For the point cloud $P _ { i }$ of �-th object $O _ { i }$ , the coupled encoder-projector architecture generates a block of geometric token embeddings,

$$
\mathbf { H } ^ { ( i ) } = g ( \mathcal { E } ( P _ { i } ) ) = \left( \pmb { h } _ { 1 } ^ { ( i ) } , \ldots , \pmb { h } _ { M _ { i } } ^ { ( i ) } \right) , \quad \pmb { h } _ { m } ^ { ( i ) } \in \mathbb { R } ^ { d _ { h } } .\tag{2}
$$

Interleaved multimodal context. We interleave the projected pointcloud tokens with the text tokens according to their positions in the prompt. The prompt verbalizes the inputs of Eq. (1) as a sequence of text segments, each followed by the point-cloud token block of the geometry it introduces. The leading segment $\mathbf { u } _ { \mathcal { B } }$ describes the footprint polygon B and is followed by the floor block ${ \bf H } ^ { ( \mathcal { B } ) } =$ $g ( \mathcal { E } ( P _ { \mathcal { B } } ) )$ . Each existing object $O _ { i }$ then contributes a segment $\mathbf { u } _ { i }$ that lists its identifier, category, size, and pose $( i , c _ { i } , s _ { i } , \pmb { p } _ { i } )$ , followed by its geometry block $\mathbf { H } ^ { ( i ) }$ , for $i = 0 , \ldots , N { - } 1$ . Segment $\mathbf { u } _ { \mathrm { t a r } }$ lists the target category and size $( c _ { \mathrm { t a r } } , s _ { \mathrm { t a r } } )$ <sup></sup>but no pose, and is followed by $\mathbf { H } ^ { \mathrm { ( t a r ) } } =$ $g ( \mathcal { E } ( P _ { \mathrm { t a r } } ) )$ <sup></sup> <sup>#&</sup> <sup></sup>, while the final segment $\mathbf { u } _ { \mathrm { i n s } }$ <sup>#&</sup> <sup></sup> <sup>#&</sup> <sup></sup>carries the instruction �. The floor and existing-object point clouds are encoded in world coordinates, the target point cloud in its canonical coordinates. For example, u<sub>0</sub> reads “object 0: sofa; size [2.10, 0.90, 0.85]; pose [1.20, 2.00, 0.43, 0, 0, 90]; point cloud: $\mathsf { \ / { c p c } > " } ,$ where the placeholder <pc> is replaced by the projected tokens $\mathbf { H } ^ { ( 0 ) }$ . Writing $\mathbf { U } _ { j } = { \mathrm { T e x t E m b } } ( \mathbf { u } _ { j } )$ for each segment embedding, we obtain the interleaved context sequence,

![](images/019446cf910697aa5cfe66d789454546b0021cd1fb933ceeb48e0c8be293e98d.jpg)  
(b) Instruction Source Distribution

(c) Instruction Relation Distribution  
![](images/c899fd7bb809d2a2bd7cb14c2dcae2d9f73b535e8951983f2fc3b20a2ea29167.jpg)  
Fig. 4. Dataset statistics. (a) Distribution of object counts per scene across data sources. (b) Proportion of placement instructions generated from each  <sup>(\$&%#\$! (\$</sup> <sup>&"!%#\$!</sup>source. (c) Distribution of instruction relation types, including absolute, on, within, and under.

$$
{ \bf X } _ { \mathrm { c t x } } = \left[ { \bf U } _ { \mathcal { B } } ; { \bf H } ^ { ( \mathcal { B } ) } ; { \bf U } _ { 0 } ; { \bf H } ^ { ( 0 ) } ; \cdots ; { \bf U } _ { N - 1 } ; { \bf H } ^ { ( N - 1 ) } ; { \bf U } _ { \mathrm { t a r } } ; { \bf H } ^ { ( \mathrm { t a r } ) } ; { \bf U } _ { \mathrm { i n s } } \right] .\tag{3}
$$

The causal LLM prefills this sequence with standard self-attention, perceiving the scene context and the target geometry jointly before it autoregressively decodes the target pose.

## 3.2 Dataset Construction

To train the geometry-aware placement model, we construct a highquality scene dataset containing object semantics, point-level geometry, spatial relations, and 6D poses, following the pipeline in Fig. 3. The dataset combines four complementary sources: two pub lic datasets, 3D-FRONT [Fu et al. 2021] and Imaginarium [Zhu et al. 2025]; a SpatialLM-PCG dataset generated from SpatialLM [Mao et al. 2026] scene annotations through procedural asset replacement and small-object enrichment; and an artist-crafted scene collection. The PCG dataset increases the frequency and diversity of complex object relations, with construction details provided in the supplemental material. As summarized in Fig. 4, the dataset covers a wide range of scene complexities. SpatialLM-PCG and artist-crafted scenes provide many placement instructions and increase the number of support and containment relations.

Dataset filtering. As shown in Fig. 3, all sources are processed by a unified filtering pipeline before training. Physics simulation and geometric validation correct or remove floating objects, severe interpenetrations, and out-of-bound placements. We then render each candidate scene and use a VLM to score placement plausibility, scene coherence, visual quality, style consistency, and semantic reasonableness. We discard samples failing these checks to reduce noisy supervision.

Scene data structuring. We then convert each retained scene into single-step autoregressive placement samples. For each object, we extract its category, size, location, rotation, and point cloud, and derive spatial relations from oriented bounding boxes in the world coordinate frame. We build a floor-rooted scene tree T from relations defining placement dependencies: abs edges attach floor-supported objects to room-level location cues, while on/within/under edges attach objects to their supporting, containing, or overhead parents. These labels are geometric predicates rather than colloquial terms: within requires the child OBB to lie inside the parent OBB, and on requires the child bottom to contact the parent OBB top. A breadth-first traversal maps this tree to the linear placement order $\pmb { \pi } = \mathrm { B F S } ( \mathcal { T } ) = ( \pi _ { 1 } , . . . , \pi _ { N } )$ . For each target object in this order, the instruction is formed from its tree edge and augmented with nearby next to relations to provide local distance, direction, and facing context, such as left/right and front/behind, relative to already placed objects. The target 6D pose serves as the supervision signal.

## 3.3 Two-Stage Training Strategies

Multimodal LLMs commonly separate modality alignment from downstream task fine-tuning: a lightweight projector is first trained to map visual or geometric features into the LLM token space, and the aligned model is then adapted to task-specific supervision [Deng et al. 2025; Liu et al. 2023; Xu et al. 2024]. We follow this recipe because our point encoder, projector, and LLM operate in diferent feature spaces. We first align the geometric and language modalities through auxiliary alignment tasks, and then fine-tune the model on autoregressive placement.

Training objective. Both stages use the standard causal language modeling objective. For a training sample with context $\mathbf { X } _ { \mathrm { c t x } }$ , let t = $( t _ { 1 } , \dots , t _ { L } )$ ) denote the complete task-specific target token sequence. With teacher forcing, we optimize the negative log-likelihood of the target tokens,

$$
\mathcal { L } = - \sum _ { \ell = 1 } ^ { L } \log p ( t _ { \ell } \mid \mathbf { X } _ { \mathrm { c t x } } , t _ { < \ell } ) .\tag{4}
$$

Stage 1: geometric–language alignment. We first construct auxiliary alignment samples from the scene dataset. Each sample pairs the floor point cloud or a set of object point clouds with a tex tual question answerable only from geometry. The questions cover boundary descriptions, object captions, object centers, sizes, rota tions, and pairwise distances or displacements. Since Utonia stays frozen, this stage does not learn geometry from scratch: the geometric competence already comes from large-scale pretraining, and the projector only has to map the pretrained features into the language embedding space. PAR3D [Dai et al. 2026] builds part-aware 3D reasoning on the same encoder, indicating that its features retain shape detail well beyond a bounding box.

![](images/97302fc533e5231b12b4ffef12d82c0fa29f0393bc31c441cb640f6ee1724dee.jpg)  
Fig. 5. Agent pipeline for automated scene generation, where a planner turns a user prompt into object-level steps, GALA places each object, and a verifier feeds back failures for correction.

Stage 2: autoregressive placement training. The aligned model is then fine-tuned on autoregressive placement samples. Following the placement order �, the model conditions at step � on the groundtruth poses of $O _ { \pi _ { 1 } } , . . . , O _ { \pi _ { n - 1 } }$ and uses the pose of $O _ { \pi _ { n } }$ as the supervision target. We serialize each 6D pose $\pmb { p } = ( x , y , z , \phi , \theta , \psi )$ into a fixed-format textual sequence and tokenize it with the LLM tokenizer. The point encoder is kept frozen, while the projector and LLM are jointly trained to combine the textual instruction with geometric context and predict a precise pose. At inference time, placement proceeds autoregressively: at each step, the model conditions on previously predicted poses, generates a textual pose sequence, parses it into numeric pose parameters, updates the scene with the newly placed object, and advances to the next one.

## 3.4 Agent Pipeline for Automated Scene Generation

For text-driven scene generation, we design a multi-stage agent pipeline that wraps GALA in a plan-place-verify loop, as shown in Fig. 5. The pipeline separates semantic decisions from numeric ones: two of-the-shelf LLM agents decide what to place and how to describe it, while GALA, the only trained component, turns each description into a 6D pose. The planner analyzes the user prompt once per scene and expands it into an ordered object list, giving each entry a category, a retrieval description, an estimated size, and the earlier objects its placement depends on. Assets are retrieved from an asset library by description and size, and the list is ordered coarse to fine so that anchors precede the objects they support. At each step, the instruction generator reads the current scene state, serialized as per-object identifiers, categories, sizes, and poses, together with the next object in the list. It emits a placement instruction in the same relation vocabulary used for training, such as “place the mouse (id 14) on the mouse pad (id 13)”, so that GALA receives the same kind of instruction at inference as during training. GALA predicts the pose of the target object, after which a verifier checks the updated scene for out-of-boundary placements, collisions, floating objects, and unsatisfied relations, and sends any failure back to the instruction generator for the next iteration. The loop terminates once the object list is exhausted; the planner and instruction generator prompts are listed in the supplemental material. Gallery results from this agent pipeline are shown in Fig. 10, spanning living rooms, storage spaces, corridors, rooftops, and outdoor alleys.

## 4 Experiments

## 4.1 Experimental Setings

Datasets. We use the two-stage datasets produced by the construction pipeline in Section 3.2. To avoid leakage, we split scenes into train, validation, and test sets with stratified sampling over the source collections at an 18:1:1 ratio. All evaluations are conducted on the test set.

Evaluation metrics. We evaluate placements along two complementary axes: instruction following and geometric sensitivity. Jointly, we deem a placement valid if its pose is well-formed, satisfies all specified relations, and is collision-free and non-floating. We report two success rates over this criterion: Instruction Valid Rate, the overall validity across all test samples, and Within Valid Rate, the validity on containment (within) samples that serves as our primary geometry-sensitive metric since such relations demand local geometric reasoning. We further report the Floating Rate and the bottom-surface Support Ratio to gauge physical support from the floor or existing objects. Detailed metric definitions are provided in the supplemental material.

Implementation details. We implement our model using LLaMA-Factory [Zheng et al. 2024], with Qwen3-0.6B [Yang et al. 2025] as the LLM backbone and Utonia [Zhang et al. 2026a] as the point encoder, for approximately 0.7B total parameters. Training follows the two-stage recipe in Section 3.3. In Stage 1, we train only the projector for 3 epochs with a global batch size of 256 and a learning rate of $1 \times 1 0 ^ { - 3 }$ . In Stage 2, we fine-tune the LLM and projector for 4 epochs while keeping the point encoder frozen, using a global batch size of128, an LLM learning rate of $2 \times 1 0 ^ { - 5 }$ , a projector learning rate $\phantom { + } 0 \mathbf { f } 5 \times 1 0 ^ { - 4 }$ , and cosine scheduling with 5% warmup. We train models in bfloat16 on 8 NVIDIA H20 GPUs for about 7 hours. At inference time, placing one object takes about one second on a single H20 GPU using roughly 3 GB of memory, and the supplemental material breaks down the runtime of the full agent pipeline.

## 4.2 Comparisons with State-of-the-Art Methods

Quantitative Comparisons. We compare external baselines using a single-object insertion protocol, where each method places one target object in a fixed scene. ATISS [Paschalidou et al. 2021] and ReSpace [Bucher and Armeni 2025] serve as OBB-based references conditioned on object category. Neither method accepts a specified placement instruction as input, so they are evaluated with only physical metrics. For instruction conditioned baselines, we use a controlled LayoutVLM [Sun et al. 2025] adaptation: existing objects are fixed, only the target object is optimized, and the instruction is supplied as structured relations instead of being interpreted by a VLM, so that the baseline is scored on placement rather than on scene planning. The supplemental material details this adaptation. We also evaluate GPT-OSS-120B [Agarwal et al. 2025] as a zero-shot large parameter model baseline. For ATISS and ReSpace, which predict both pose and size, we scale the target point cloud by the predicted size before computing physical metrics. Table 1 summarizes the quantitative results.

![](images/03afaf6e1314a7ca73686170d76d3b7c7ccab87ecb7b600c2157abe63f79862b.jpg)  
Fig. 6. Qualitative comparisons of instruction-conditioned single-object placement results across instruction-aware baselines and our method

Table 1. Quantitative comparisons for single-object inference on the heldout test split. Cat., Instr., and PC denote category, instruction, and point cloud, respectively. Dashes indicate instruction-dependent metrics that are not applicable.
<table><tr><td>Method</td><td>Input</td><td>Within. ↑ Instruction. ↑ Float. ↓ Support. ↑</td><td></td><td></td><td></td></tr><tr><td>ATISS</td><td>OBB+Cat.</td><td>一</td><td>一</td><td>0.93%</td><td>99.20%</td></tr><tr><td>ReSpace</td><td>OBB+Cat.</td><td></td><td>一</td><td>5.79%</td><td>94.20%</td></tr><tr><td>LayoutVLM</td><td>OBB+Instr.</td><td>10.96%</td><td>37.71%</td><td>17.05%</td><td>82.91%</td></tr><tr><td>GPT-OSS-120B OBB+Instr.</td><td></td><td>22.53%</td><td>42.29%</td><td>7.64%</td><td>93.54%</td></tr><tr><td>Ours (0.7B)</td><td>PC+Instr.</td><td>65.62%</td><td>78.59%</td><td>4.63%</td><td>96.03%</td></tr></table>

Our model achieves the strongest instruction-conditioned performance, with clear gains in both Within Valid Rate and Instruction Valid Rate over LayoutVLM and GPT-OSS-120B. It also maintains competitive physical plausibility, with lower Floating Rate and higher Support Ratio. ATISS obtains a lower Floating Rate and a higher Support Ratio because it tends to place objects on the floor instead of making fine-grained placements. The supplemental material reports further comparisons under the same protocol, covering PlaceIt3D [Abdelreheem et al. 2025], two substantially larger zeroshot LLMs, and an image-generation plus pose-estimation pipeline, none of which approaches our performance.

Qualitative Comparisons. Figure 6 illustrates LayoutVLM, GPT-OSS-120B, and our method under the same single-object placement instructions. LayoutVLM and GPT-OSS-120B respond to the instruction more directly, yet their predictions often miss fine-grained geometric constraints such as the target support surface, local free space, or containment region. In contrast, our model uses point cloud to place the target object in geometrically valid regions.

Figure 8 extends this qualitative comparison to whole-scene autoregressive inference. Here the object set and the per-step instructions are taken from the held-out test scene, so the comparison isolates placement quality from planning, unlike the agent pipeline of Section 3.4, which derives both from a text prompt. Across multistep scene construction, our model produces more plausible and coherent layouts. Additional inference results produced by our model on held-out test scenes are shown in Fig. 9.

User Study. Table 2 presents the overall results of the user study. We conduct a user study with 30 volunteers to evaluate single-object and whole-scene comparison results. Each volunteer rates the displayed results on a five-point Likert scale, where 1 is the lowest quality and 5 is the highest quality. For single-object placement results, the volunteers score physical plausibility and instruction following. For whole-scene results, they score physical plausibility and visual aesthetics. Since ATISS and ReSpace do not accept placement instructions, their single-object evaluations are omitted.

![](images/49ac9f89876d85f394eb1d430beca4adeed76bfd6ee677ee32ccfff411db94cf.jpg)  
Fig. 7. Qualitative ablation results on fine-grained placement. Each column shows one instruction, the predicted scene, and a zoomed crop around the target placement. Blue boxes mark inserted target objects.

Table 2. User-study scores from 30 participants. Dashes denote unavailable single-object scores for methods without instruction input.
<table><tr><td rowspan="3">Method</td><td colspan="2">Single-object placement</td><td colspan="2">Whole-scene result</td></tr><tr><td>Physical. ↑ Instruction. ↑ Physical. ↑ Aesthetics. ↑</td><td></td><td></td><td></td></tr><tr><td>ATISS</td><td>一</td><td>一</td><td>1.9</td><td>1.9</td></tr><tr><td>ReSpace</td><td>一</td><td>一</td><td>2.7</td><td>2.9</td></tr><tr><td>LayoutVLM</td><td>1.9</td><td>2.0</td><td>3.0</td><td>3.2</td></tr><tr><td>GPT-OSS-120B</td><td>2.7</td><td>2.4</td><td>2.5</td><td>2.5</td></tr><tr><td>Ours (0.7B)</td><td>4.5</td><td>4.5</td><td>3.9</td><td>4.1</td></tr></table>

## 4.3 Ablation Studies

We isolate the contribution of point cloud input, Stage 1 alignment pretraining, and high-quality training scenes through controlled ablations on the same test dataset. The high-quality scenes include the Imaginarium and artist-crafted subsets.

Efect of point cloud input. The Instruction Valid Rate remains comparable across variants, indicating point cloud input does not substantially improve generic instruction-conditioned validity. Removing point cloud input weakens the model most directly on the Within Valid Rate, where instruction following must be grounded in local containment geometry. The same ablation also increases floating failures and reduces bottom-surface support, suggesting text-only context is insuficient for reliably grounding objects on available support surfaces. Figure 7 corroborates these numbers: without point cloud input, the predicted objects frequently miss the referenced container, e.g., the fruit is placed outside the bookcase (column 1) even though the textual relation is parsed correctly. Aggregate validity rates do not capture every fine-grained geometric failure. Measured directly, point cloud input reduces the targetcenter height error on within samples by about 14%, and a dedicated user study on this ablation prefers the point-cloud variant by a wide margin. Both analyses are reported in the supplemental material.

Table 3. Controlled ablations on the held-out test split. Within/Instruction report valid rates on containment/all samples. Arrows show preferred directions; bold/underlined values mark the best/second-best results.
<table><tr><td>Method</td><td>Within. ↑ Instruction. ↑ Float. ↓ Support. ↑</td><td></td><td></td><td></td></tr><tr><td>w/o point cloud</td><td>62.40%</td><td>78.81%</td><td>5.19%</td><td>95.35%</td></tr><tr><td>w/o pretraining</td><td>62.69%</td><td>78.37%</td><td>5.20%</td><td>95.42%</td></tr><tr><td>w/o artist scenes</td><td>60.32%</td><td>76.69%</td><td>5.02%</td><td>95.88%</td></tr><tr><td>Full method</td><td>65.62%</td><td>78.59%</td><td>4.63%</td><td>96.03%</td></tr></table>

Efect ofalignment pretraining. Removing Stage 1 alignment pretraining leads to a similar degradation, even though the model still receives point cloud input during placement fine-tuning. The Within Valid Rate decreases from 65.62% to 62.69%, showing the benefit of projector alignment. Figure 7 exposes the same weakness qualitatively: without alignment pretraining, the fruit hovers in mid-air instead of resting on the shelf plank (column 1), and the blanket stays on the countertop and cuts into the basin instead of being placed inside the open compartment of the sink cabinet (column 4).

![](images/c0961eba2d538fca56ce0915f55fb04f3321f1df45f1260cc980603ddf5d6c2c.jpg)  
Fig. 8. Qualitative comparison with baselines on whole-scene inference results. Each method receives the object set and the per-step placement instructions of a held-out test scene and places the objects one at a time, so the comparison isolates placement quality from planning.

Together, these two ablations indicate explicit point cloud geometry and alignment pretraining are complementary.

Efect of high-quality scenes. Removing high-quality training scenes including the Imaginarium and artist-crafted scenes produces the largest drop in Within Valid Rate. This result suggests high-quality scenes primarily improve fine-grained geometric grounding for containment relations rather than generic instruction following validity. The Floating Rate and Support Ratio also degrade, indicating these high-quality scenes provide useful supervision for stable object placement. The qualitative cases in Fig. 7 mirror this trend: without high-quality scenes, the placed objects exhibit clear collisions or misalignment with the referenced anchors, e.g., the blanket cuts through the body ofthe sink cabinet and juts out beyond its side (col umn 4) and the briefcase overlaps the shelf frame (column 2) instead of resting inside it, whereas our full method produces collision-free placements that consistently satisfy the specified relations. A stratified analysis in the supplementary material further shows this drop is concentrated on the high-quality test subset.

## 5 Conclusions

We propose GALA, a geometry-aware language model for controllable object arrangement. By augmenting LLMs with point cloud perception and a two-stage training recipe, GALA can predict conditioned 6D poses. Our approach captures fine-grained spatial constraints that are often missed by OBB-only methods. We also integrate GALA seamlessly into an interactive agent framework for automated scene generation and editing. Furthermore, we will release a dataset of high-fidelity scenes with verified, complex placements to facilitate future research.

Limitations. Despite its advantages, GALA currently supports only a predefined set of placement instructions and cannot yet handle open-ended placement requests. Its performance also degrades when generalizing to previously unseen container geometries. Moreover, because objects are placed autoregressively, errors in earlier placements may propagate to subsequent ones.

Future Work. Future research will extend this framework to handle noisy real-world scanned scenes, and support diverse interactive editing operations. Another promising direction is explicit shapeor part-level supervision losses during two-stage training, which could strengthen geometric grounding. Finally, because our model currently decodes pose tokens directly from context, future work could explore explicit chain-of-thought reasoning over geometry, in which the model first thinks about intermediate decisions, such as support-surface selection and the available space on that surface, before generating the pose.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 62025207. We thank the reviewers for their time and constructive feedback.

![](images/0f104f4f496d6111cfdb866ed5484f59aedcd1efdee626e55988ca436c0347b2.jpg)  
Simple Bedroom

![](images/141568bf264ec3e3f4d16712fa7f4f299846cc3e4fd296103bf9b1821a6f13a2.jpg)  
Simple Living room

![](images/b798fdd47098e71c6daadc93a24e4472c9cfa279b14d437e02e973e0fa32e47f.jpg)  
Simple Bedroom

![](images/03755fe0eae38e36dc0fe0f936edbb9c11ff96ced6559f75023162a6189d8b63.jpg)  
Simple Bedroom

![](images/b1c5b052c0f75facddfbd162d11c3da1ad487fd9d5b170a2e8c5765cfe4c993c.jpg)  
Simple Living room

![](images/6bdd21dee9a66aac9ffec3fa9b5b3d1fc9ca2a1679d7f56667040a714646c983.jpg)  
Simple Living room

![](images/e362849b1ef0e795f2107aa74b490d30a33111e6e53aa2fb7c964dcba37f446e.jpg)

![](images/537c9d23f0a6ead8adc36bc06dd2facc2a095010a35906e6a67eb241971a91e6.jpg)  
Simple Bedroom

![](images/9f750fe02e3cdd9e61a9748c1e8ca21deba2eb3c545cc626808591d16f5f02d3.jpg)  
Complex Dining room  
Dental Office

![](images/1edfd304e330aaea274abe3719d4fa530c9784ae8f4b7a364e9c321c276d15f4.jpg)  
Complex Bedroom

![](images/26e92e14f1aff199aa0f11ebe23bacc61bffb3be5547140b0d65b5dd56c74e32.jpg)  
Complex Living room

![](images/31608f4fe527e8779ab0ca518db6173ef9a39b2aae935a7c100dcbbae71da761.jpg)  
Complex Living room

Fig. 9. More autoregressive inference results of our model on held-out test scenes. Starting from scene-level object sets and placement instructions, our model sequentially predicts object poses and produces complete arrangements across diferent room types and layout complexity.  
![](images/931b27ebf0a3d1da3117f285de69557005437de83891d423ff331a9537b47f2a.jpg)

![](images/b1db3f80554319486479034131e619cf84cb0dd7bb288a7f533342996b591831.jpg)

![](images/984e349e444d181f61b4967e3ea3c12e6bcbc4ee1024d9d186559dff23fcb14f.jpg)

![](images/965a9ffc3e5bcf842d77d9bf6ed6bb56cfc5cfd2d093ab17180fe1a1a41e8371.jpg)

A living room includes a multiseat sofa, a coffee table with food on it, a TV stand with a television on top and so on.

A living room and bedroom space includes several multitier shelving units filled with relevant household objects.

A storage warehouse with cabinets and shelving units filled with stored items, along with other relevant objects.

A corridor inside a residential building with several coffee tables with food and drinks on it, and some potted green plants.

![](images/122d6825eaf80357792aaa5df42f4320907819e4d1d944f175533bd2ba11811c.jpg)

A relaxation area on a rooftop with tables and chairs, place food and drinks on the tables, and add other relevant objects.

![](images/c8c9f64fa1f0e0edcc94889c88ec27ad100190bfb39135ba78191a983001ae33.jpg)  
A relaxation area in the room with tables and chairs, place food and drinks on the tables, and add other relevant objects.

![](images/7cb287f841d4e4b7bd8e73c253db8564bb25126e8df5a0734cfaa38abb1eb470.jpg)  
A rundown alley, place one large dumpster, several trash bins, several old worn shelving units filler undownd with garbage and clutter, and a large amount of additional trash and miscellaneous debris throughout the alley.

Fig. 10. Gallery results from the agent pipeline for scene generation. Given text prompts and polygon floor boundary, the planner decomposes the request into object-level placement steps, then GALA predicts poses for the target objects to produce complete 3D scene arrangements.

## References

Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, and Guillermo Garcia Hernando. 2025. PlaceIt3D: Language-guided object placement in real 3D scenes. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision. 6645–6655.

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 (2025).

Martin JJ Bucher and Iro Armeni. 2025. Respace: Text-driven 3d scene synthesis and editing with preference alignment. arXiv e-prints (2025), arXiv–2506.

Shaohui Dai, Yansong Qu, You Shen, Shengchuan Zhang, and Liujuan Cao. 2026. PAR3D: A Unified 3D-MLLM with Part-Aware Representation for Scene Understanding. arXiv preprint arXiv:2606.06485 (2026).

Jiajun Deng, Tianyu He, Li Jiang, Tianyu Wang, Feras Dayoub, and Ian Reid. 2025. 3dllava: Towards generalist 3d lmms with omni superpoint transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 3772–3782.

Haoran Feng, Yifan Niu, Zehuan Huang, Yang-Tian Sun, Chunchao Guo, Yuxin Peng, and Lu Sheng. 2026. Repurposing 3D Generative Model for Autoregressive Layout Generation. arXiv preprint arXiv:2604.16299 (2026).

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems 36 (2023), 18225–18250.

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 2021. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10933–10942

Lin Gao, Jia-Mu Sun, Kaichun Mo, Yu-Kun Lai, Leonidas J Guibas, and Jie Yang. 2023. Scenehgn: Hierarchical graph networks for 3d indoor scene generation with finegrained geometry. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 7 (2023), 8902–8919.

Maxim Gumin, Do Heon Han, Seung Jean Yoo, Aditya Ganeshan, R Kenny Jones, Kailiang Fu, Rio Aguina-Kang, Stewart Morris, and Daniel Ritchie. 2025. Procedural Scene Programs for Open-Universe Scene Generation: LLM-Free Error Correction via Program Search. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–11.

Shuting He, Henghui Ding, Xudong Jiang, and Bihan Wen. 2024. Segpoint: Segment any point cloud via large language model. In European Conference on Computer Vision. Springer, 349–367.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 2023. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems 36 (2023), 20482–20494.

Ian Huang, Yanan Bao, Karen Truong, Howard Zhou, Cordelia Schmid, Leonidas Guibas, and Alireza Fathi. 2025a. Fireplace: Geometric refinements of llm common sense reasoning for 3d object placement. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. Computer Vision Foundation, Nashville, TN, USA, 13466–13476.

Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. 2025b. Reason3d: Searching and reasoning 3d segmentation via large language model. In 2025 International Conference on 3D Vision (3DV). IEEE, IEEE, 1177–1186.

Manyi Li, Akshay Gadi Patil, Kai Xu, Siddhartha Chaudhuri, Owais Khan, Ariel Shamir, Changhe Tu, Baoquan Chen, Daniel Cohen-Or, and Hao Zhang. 2019. Grains: Generative recursive autoencoders for indoor scenes. ACM Transactions on Graphics (TOG) 38, 2 (2019), 1–16.

Chenguo Lin and Yadong Mu. 2024. Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior. arXiv preprint arXiv:2402.04717 (2024).

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in Neural Information Processing Systems 36 (2023), 34892–34916.

Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. 2026. Spatiallm: Training large language models for structured indoor modeling. Advances in Neural Information Processing Systems 38 (2026), 45165–45195.

Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, and Sanja Fidler. 2021. Atiss: Autoregressive transformers for indoor scene synthesis. Advances in neural information processing systems 34 (2021), 12013–12026.

Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. 2025. Layoutvlm: Diferentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 29469–29478.

Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai,Justus Thies, and Matthias Nießner. 2024. Difuscene: Denoising difusion models for generative indoor scene synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20507–20518.

Kai Wang, Yu-An Lin, Ben Weissmann, Manolis Savva, Angel X Chang, and Daniel Ritchie. 2019. Planit: Planning and instantiating indoor scenes with relation graph and spatial prior networks. ACM Transactions on Graphics (TOG) 38, 4 (2019), 1–15.

Kai Wang, Manolis Savva, Angel X Chang, and Daniel Ritchie. 2018. Deep convolutional priors for indoor scene synthesis. ACM Transactions on Graphics (TOG) 37, 4 (2018), 1–14.

Wenzheng Wu, Chucheng Xiang, Zhi Lin, Yirui Guan, Ruchao Bao, Zhongyuan Liu, Ziqi Wang, and Ligang Liu. 2026. Scene Layout via conceptual design. Computers & Graphics (2026), 104553.

Chucheng Xiang, Ruchao Bao, Biyin Feng, Wenzheng Wu, Zhongyuan Liu, Yirui Guan, and Ligang Liu. 2026. Co-Layout: LLM-driven Co-optimization for Interior Layout. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 40. 14371–14379.

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. 2024. Pointllm: Empowering large language models to understand point clouds. In European Conference on Computer Vision. Springer, 131–147.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. 2024. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16227–16237.

Guangyao Zhai, Evin Pınar Örnek, Dave Zhenyu Chen, Ruotong Liao, Yan Di, Nassir Navab, Federico Tombari, and Benjamin Busam. 2024. Echoscene: Indoor scene generation via information echo over scene graph difusion. In European Conference on Computer Vision. Springer, 167–184.

Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. 2023. Commonscenes: Generating commonsense 3d indoor scenes with scene graph difusion. Advances in Neural Information Processing Systems 36 (2023), 30026–30038.

Jiang Zhang, Shijie Zhou, Bangya Liu, Achuta Kadambi, and Zhiwen Fan. 2026b. SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning. arXiv preprint arXiv:2603.27437 (2026)

Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, and Hengshuang Zhao. 2026a. Utonia: Toward One Encoder for All Point Clouds. arXiv preprint arXiv:2603.03283 (2026).

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. Llamafactory: Unified eficient fine-tuning of 100+ language models. In Proceedings ofthe 62nd annual meeting ofthe association for computational linguistics (volume 3: system demonstrations). 400–410.

Yang Zhou, Zachary While, and Evangelos Kalogerakis. 2019. Scenegraphnet: Neural message passing for 3d indoor scene augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7384–7392.

Xiaoming Zhu, Xu Huang, Qinghongbing Xie, Zhi Deng, Junsheng Yu, Yirui Guan, Zhongyuan Liu, Lin Zhu, Qijun Zhao, Ligang Liu, et al. 2025. Imaginarium: Visionguided High-Quality 3D Scene Layout Generation. ACM Transactions on Graphics (TOG) 44, 6 (2025), 1–24.

## Supplementary Material

## A SpatialLM-PCG Dataset Expansion

SpatialLM provides room-scale semantic layouts in the form of object categories, approximate 3D boxes, and coarse poses. We use these layouts as structural scafolds rather than final training scenes. As shown in Fig. 11, each proxy object is replaced by a plausible asset through category-constrained, size-and-shape-aware retrieval. For a proxy object with category $c _ { i }$ and box size ${ \bf s } _ { i } = ( w _ { i } , d _ { i } , h _ { i } )$ we consider only assets with the same semantic category. Each candidate asset � has a canonical box size $\mathbf { a } _ { m } = \left( w _ { m } , d _ { m } , h _ { m } \right)$ and is ranked by

$$
\begin{array} { r } { E ( i , m ) = \left. \log \mathbf { a } _ { m } - \log \mathbf { s } _ { i } \right. _ { 1 } + \lambda \left. \log \hat { \mathbf { a } } _ { m } - \log \hat { \mathbf { s } } _ { i } \right. _ { 1 } , } \end{array}\tag{5}
$$

where $\hat { \textbf { s } } = \textbf { s } / ( w d h ) ^ { 1 / 3 }$ is a scale-invariant shape descriptor, and � balances absolute size matching and proportion matching. We select the asset minimizing Eq. (5), then scale, orient, and place it to preserve the original layout. The second term penalizes mismatched aspect ratios, reducing artifacts from extreme anisotropic scaling while keeping the spatial organization provided by SpatialLM.

After the major furniture layout is instantiated, we add small objects procedurally to enrich local interactions. Candidate support and storage furniture, such as tables, cabinets, shelves, and TV stands, are assigned afordance regions derived from their geometry. Small assets are sampled from category-compatible libraries and placed on, within, or near these regions with randomized ofsets and orientations. The generated objects remain anchored to plausible functional locations while increasing the frequency and diversity of support, containment, proximity, and directional relations in the placement instruction data. The gallery in Fig. 12 illustrates the diversity of the curated scenes that result.

## B Spatial Relation Definitions

We compute spatial relations from the posed oriented bounding boxes of the target object and each candidate reference object. Each relation is represented as a discrete relation label together with the geometric quantities used to evaluate it. These relation labels provide the intermediate supervision used to generate per-step natural-language instructions. The full set of relation definitions is summarized in Table 4.

Notation. Let $O _ { a }$ denote the target object and $O _ { b }$ a reference object. We use $F _ { i }$ for the XY footprint of object $O _ { i } ,$ obtained by projecting its oriented bounding box (OBB) vertices and taking the convex hull. Let $d _ { x y } ( a , b )$ be the minimum distance between $F _ { a }$ and $F _ { b } ,$ , and let $z _ { i } ^ { - }$ and $z _ { i } ^ { + }$ denote the bottom and top of the posed 3D box. The horizontal angle $\theta _ { a b } = \mathrm { a t a n } 2 ( y _ { a } - y _ { b } , x _ { a } - x _ { b } )$ is measured in degrees. The thresholds used in data generation are $\tau _ { c } = 0 . 0 5$ m for contact, $\tau _ { u } = 0 . 1 0$ m for under-object grounding, $\tau _ { n } = 0 .$ .50 m for next-to proximity, $\tau _ { d } = 1 . 5 0 \mathrm { m }$ for directional relations, $\eta _ { z } = 0 . 8$ for horizontal height compatibility, $\eta _ { w } = 0 . 1 5$ for wall proximity, $\eta _ { r } = 0 . 2 5$ for scene-region assignment, and $\tau _ { \theta } = 1 5 ^ { \circ }$ for facing-angle alignment.

## C Reference Object Selection

Each autoregressive step conditions on the already placed objects. To keep this context compact, we select a reference subset instead of passing the entire scene when the number of placed objects exceeds $K = 8 .$

Training-time selection. During data generation, the target pose is known and can be used to identify geometrically relevant references. We first compute all spatial relations between the target object and the existing objects. Among these relations, we retain the most informative ones for the language instruction by sorting relation candidates by relation priority and then by footprint edge distance. Support and containment relations (on, under, and within) have the highest priority, followed by next-to proximity and then directional and facing relations. The objects explicitly mentioned by these selected relations are always retained.

In addition, if the target object has a parent in the dependency forest, we retain up to $S = 6$ already placed siblings under the same parent, chosen by increasing footprint edge distance to the target. These sibling references help represent local occupancy around shared supports or containers, such as multiple items on the same shelf or table. The remaining slots are filled with the nearest existing objects by footprint edge distance. The final selected set is sorted back into autoregressive order before being serialized. If the mandatory references exceed the nominal cap, we keep all mandatory references rather than dropping supporting or sibling context.

Rollout-time selection. At inference time, the target pose is unknown, so edge-distance ranking to the target is unavailable. We therefore use an importance-based proxy. Mandatory semantic dependencies specified by the planner are retained first; if a retained object is itself a sub-object, its parent is recursively retained as well. The remaining slots are allocated between large anchors and recent context: 60% of the available slots are assigned to objects with the largest footprint area, and the rest to the most recently introduced objects. As in trainin $^ { 1 9 , }$ the selected references are returned in autoregressive order. This policy preserves stable scene anchors while keeping the most recent local context available for the next placement.

## D Design-Prior Object Ordering

Our autoregressive training samples require a deterministic order for introducing objects into a scene. Instead of relying on the arbitrary order stored in the source datasets, we use the design-prior heuristic of Algorithm 1, which approximates a coarse-to-fine layout process. It places large structural anchors first, then introduces smaller objects once the supporting or containing geometry they depend on is available.

We first infer a dependency forest from pairwise spatial relations. If object $O _ { i }$ is on, within, or under object $O _ { j }$ , then $O _ { j }$ is treated as the parent of $O _ { i }$ , because $O _ { j }$ provides the support, container, or overhead structure needed to localize $O _ { i }$ . When multiple parent candidates exist, we prefer supporting relations over containment and underobject dependencies, and select the smallest parent footprint among candidates with the same priority. This favors the most specific

![](images/408ca844de9dcd4d97a6dfffb3c601d873c9f7d1001b427f079e7f8625641d89.jpg)  
Fig. 11. Procedural expansion of SpatialLM layouts. Starting from a 3D-box semantic layout, we replace large proxy objects with compatible assets from a curated library and then enrich the scene with small objects placed according to the geometry and afordances of the instantiated furniture.

![](images/eee02c0abb458c82cd4ded909775881c2d7ab6f12e5955b411b91a4e6c0fd324.jpg)  
Fig. 12. Overview of the hand-crafted scene dataset used for geometry-aware object arrangement. The gallery illustrates the diversity of room types, object categories, and fine-grained support and containment configurations.

geometric reference, such as a shelf board over the entire cabinet.

Table 4. Definitions of spatial relations used for instruction generation.
<table><tr><td>Relation type</td><td>Relation</td><td>Geometric condition</td><td>Thresholds</td></tr><tr><td>Scene boundary</td><td>against east wall against west wall against north wall</td><td>The object is floor-supported  $( z _ { a } ^ { - } \leq \tau _ { u } )$  The distance from  $F _ { a }$  to the corre- sponding boundary edge is below  $\eta _ { w } \mathrm { d i a g } _ { x y } ( O _ { a } )$  . Directional wall names are assigned from the edge outward normal.</td><td> $\tau _ { u } , \eta _ { w }$ </td></tr><tr><td>Scene boundary</td><td>at northeast corner at northwest corner at southeast corner at southwest corner</td><td>The object is floor-supported and its footprint is within the wall threshold of two adjacent convex boundary edges. The corner name is determined from the two outward normals, with a centroid-based quadrant fallback for irregular boundaries.</td><td> $\tau _ { u } , \eta _ { w }$ </td></tr><tr><td>Scene region</td><td>in front region in back region in left region in right region</td><td>Used when no wall or corner relation is emitted. The object center is com- pared with the scene AABB center: front/back use the sign of the Y offset and left/right use the sign of the X offset. A directional region requires the corresponding offset magnitude to exceed  $\eta _ { r }$  times the scene depth or width;</td><td> $\eta _ { r }$ </td></tr><tr><td>Horizontal proximity</td><td>in center region next to</td><td>otherwise the object is in the center region. The footprint distance satisfies  $d _ { x y } ( a , b ) \ < \ \tau _ { n }$  . Horizontal relations are suppressed when an on, under, or wi thin relation already holds, and require</td><td> $\tau _ { n } , \eta _ { z }$ </td></tr><tr><td>Horizontal direction</td><td>left right</td><td>compatible bottom heights:  $| z _ { a } ^ { - } - z _ { b } ^ { - } | \leq \eta _ { z } \operatorname* { m a x } ( h _ { a } , h _ { b } )$  , where  $h _ { i } = z _ { i } ^ { + } - z _ { i } ^ { - }$  The footprint distance satisfies  $d _ { x y } ( a , b ) < \tau _ { d }$  and the same height and vertical-relation exclusions as above hold. The relation is selected by  $\theta _ { a b } \colon$  left for [135°,  $1 8 0 ^ { \circ } \mathrm { ~ J ~ U ~ } ( - 1 8 0 ^ { \circ } , - 1 3 5 ^ { \circ } \mathrm { ] }$  , right for  $[ - 4 5 ^ { \circ } , 4 5 ^ { \circ } )$  , in front for</td><td> $\tau _ { d } , \eta _ { z }$ </td></tr><tr><td>Support</td><td>behind on</td><td>[45°, 135°), and behind for  $[ - 1 3 5 ^ { \circ } , - 4 5 ^ { \circ } )$  The target center projected to XY lies inside  $F _ { b } ,$  and the target bottom is</td><td></td></tr><tr><td>Containment</td><td>within</td><td>close to the reference top:  $| z _ { a } ^ { - } - z _ { b } ^ { + } | < \tau _ { c } .$  All eight OBB vertices of  $O _ { a }$  lie inside the OBB of  $O _ { b } ,$  allowing a tolerance -</td><td> $\tau _ { c }$ </td></tr><tr><td>Under</td><td>under</td><td>of  $\tau _ { c }$  The on relation takes precedence when both conditions are satisfied. The target center projected to XY lies inside  $F _ { b } ,$  the two bottoms are close  $\begin{array} { r } { ( | z _ { a } ^ { - } - z _ { b } ^ { - } | \leq \tau _ { u } ) , } \end{array}$  and the target center is below the reference top  $( z _ { a } < z _ { b } ^ { + } ) .$ </td><td> $\tau _ { u }$ </td></tr><tr><td>Facing</td><td></td><td>Cases satisfying on are excluded. The target faces the reference if the angle between the target front direction and the vector from  $O _ { a }$  to  $O _ { b }$  is below  $\tau _ { \theta } ,$  with  $d _ { x y } ( a , b ) < \tau _ { d } .$  The front</td><td></td></tr></table>

Collision. Collisions are detected directly on raw point clouds, without convex-hull or mesh conversion, so that hollow containers such as cabinets and shelves are treated as the open geometry they are. We voxelize the posed target point cloud and every existingobject point cloud at a resolution of 0.02 m, and denote by $V _ { \mathrm { t a r } }$ the set of occupied target voxels and by $V _ { \mathrm { o v l } }$ the subset of them also

This section gives the exact definitions of the metrics reported in the main paper and in this supplement. Each evaluation sample contains an existing scene, a target object point cloud, a predicted 6D pose, and a set of expected instruction relations. All methods, including the baselines, are scored by the same evaluator.

Floor-covering objects are not used as supporting parents for on relations.

The resulting sequence is not meant to model a single universal human ordering. Instead, it encodes the practical dependency: fine grained objects are easier to place after their geometric anchors are present. In the generated training data, object identifiers are reassigned according to this sequence, so each autoregressive step conditions only on objects already introduced by the design-prior order.

## E Evaluation Metric Definitions

occupied by some existing object. A placement is marked as colliding when

$$
\left| V _ { \mathrm { o v l } } \right| / \left| V _ { \mathrm { t a r } } \right| > 0 . 2 0 .\tag{6}
$$

Thresholding a ratio rather than an absolute overlap keeps the criterion scale-invariant across object sizes and tolerates the thin-shell interpenetration that point-cloud discretization inevitably produces, while still rejecting placements that pass through existing geometry.

Floating Rate and Support Ratio. To measure physical support, we first transform the target object point cloud by the predicted pose and take the lowest 10% ofits points by � value as the bottom surface �. A 2D support height map is built from the floor point cloud and from existing-object points no higher than the target bottom plus a small gap tolerance. For each bottom point $\qquad p \in B ,$ , we search a 5 × 5 voxel neighborhood in this map and record the nearest supporting height below the point. The point is considered supported if its vertical gap to this support surface is at most $\tau _ { \mathrm { g a p } } = 0 . 0 5 \mathrm { m }$ . The Support Ratio is

$$
r _ { \mathrm { s u p } } = \frac { 1 } { | B | } \sum _ { \mathfrak { p } \in B } \mathbb { I } \left[ \mathrm { g a p } ( \mathfrak { p } ) \leq \tau _ { \mathrm { g a p } } \right] .\tag{7}
$$

If the target object height is available and the predicted center height is within 0.05 m of the floor-contact height from both sides, the sample is treated as floor-supported directly. For the full pointcloud check, we also flag a placement as sinking when the median bottom-surface gap is below $- \tau _ { \mathrm { g a p } }$ , indicating the object has sunk into the supporting surface. A placement is counted as floating when $r _ { \mathrm { s u p } } < 0 . 3 0$ or when this sinking condition holds. We report Floating Rate as the fraction of floating placements and Support Ratio as the mean of $r _ { \mathrm { s u p } } { \mathrm { : } }$ , where samples resolved by the floor-contact shortcut above contribute $r _ { \mathrm { s u p } } = 1$

ALGORITHM 1: Design-prior object ordering.   
Input: Object set $O = \{ O _ { i } \} _ { i = 1 } ^ { N }$   
Output: Ordered object sequence �   
foreach $O _ { i } \in O$ do   
Compute footprint area �<sub>�</sub> and bottom height $b _ { i } ;$   
Initialize parent(� ) ← ∅;   
end   
foreach $O _ { i } \in O$ do   
C<sub>�</sub> ← ∅; $/ /$ candidate parents   
foreach $O _ { j } \in O , j \neq i$ do   
Compute spatial relations between $O _ { i }$ and $O _ { j } ;$   
if �<sub>�</sub> is on, within, or under $O _ { j }$ then   
Add $O _ { j }$ to $C _ { i }$ with priority on > within > under;   
end   
end   
if $C _ { i } \neq \varnothing$ then   
parent(�) ← the candidate with highest priority and   
smallest footprint area;   
end   
end   
Break each cyclic dependency by making the lowest object in the   
cycle a root;   
R ← {� | parent(� ) = ∅};   
Sort R by increasing $( - a _ { i } , b _ { i } , i ) ;$   
� ← R;   
L ← R; // current breadth-first layer   
while $\mathcal { L } \neq \emptyset$ do   
$\mathcal { L } ^ { \prime }  \mathcal { O } ;$   
foreach $p \in { \mathcal { L } }$ in order do   
$\mathcal { K } _ { p } \gets \{ i \}$ parent(� ) = � };   
Sort $\mathcal { K } _ { P }$ by increasing $( - a _ { i } , b _ { i } , i ) ;$   
Append $\mathcal { K } _ { P }$ to $\mathcal { L } ^ { \prime } ;$   
end   
Append $\mathcal { L } ^ { \prime }$ to �;   
$\mathcal { L }  \mathcal { L } ^ { \prime } ;$   
end   
return �;

Instruction Valid Rate and Within Valid Rate. For each sample �, let $P _ { i }$ indicate whether the model output can be parsed as a valid 6D pose, $S _ { i }$ indicate whether all expected instruction relations are satisfied, $C _ { i }$ indicate collision, and $F _ { i }$ indicate floating. Let $I _ { i }$ be the Instruction Valid indicator:

$$
{ \cal I } _ { i } = P _ { i } \wedge S _ { i } \wedge { \neg } C _ { i } \wedge { \neg } F _ { i } .\tag{8}
$$

The Instruction Valid Rate is the average of this indicator over all evaluated samples. The Within Valid Rate uses the same indicator on the subset $\mathcal { W }$ of samples whose expected relations contain at least one within relation:

$$
\mathrm { W i t h i n V a l i d } = \frac { 1 } { | \mathcal { W } | } \sum _ { i \in \mathcal { W } } I _ { i } .\tag{9}
$$

This subset is particularly sensitive to point-level geometry because containment and insertion often depend on openings, shelves, and support surfaces missing from coarse bounding boxes.

## F Checkpoint Selection

![](images/1e7221ede231163e9e3baedb30b950ad6a51ced9553e093d1c3dc5dff1e2f5a4.jpg)  
Fig. 13. Validation loss curves for the text-only and point-cloud variants used in the internal comparison. Both variants reach their minimum validation loss at step 6240 (epoch 2.67), the checkpoint used for the reported test metrics; the markers annotate the corresponding validation losses.

For the internal comparison in the main paper, we selected checkpoints by validation loss before inspecting test-set metrics. As shown in Fig. 13, both the text-only and point-cloud variants reached their lowest validation loss at step 6240 (epoch 2.67), with validation losses of 0.341 and 0.332, respectively. We therefore report test metrics using this validation-selected checkpoint rather than the final checkpoint after four epochs.

## G LayoutVLM Adaptation

LayoutVLM synthesizes a scene by having a VLM propose diferentiable spatial constraints and a gradient-based solver optimize the asset poses jointly. Since our protocol fixes the scene and inserts a single object, we supply the instruction directly as the structured relations of Table 4, which grants the baseline oracle instruction parsing, and let its solver optimize only the target pose while all existing objects stay frozen. Like the original method, the adaptation reasons over oriented bounding boxes and derives the placement height from the reference box, using its top face for on and its center height for within. For the whole-scene comparison in the main paper, it runs autoregressively under the same protocol as our model, inserting objects one at a time in the order and with the per-step instructions of the held-out test scene.

## H Additional Baseline Comparisons

Table 5 evaluates additional methods using the same single-object insertion protocol and geometric metrics as in the main paper. All existing scene objects and the target geometry remain fixed, so each method is evaluated only on the predicted placement.

Table 5. Additional single-object placement baselines on the full held-out test split.
<table><tr><td>Method</td><td>Within. ↑ Instruction. ↑ Floating. ↓ Support. ↑</td><td></td><td></td><td></td></tr><tr><td>PlaceIt3D</td><td>0.32%</td><td>1.39%</td><td>67.53%</td><td>35.96%</td></tr><tr><td>DeepSeek-V4-Flash</td><td>11.53%</td><td>30.02%</td><td>17.89%</td><td>84.75%</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>17.35%</td><td>37.28%</td><td>12.87%</td><td>89.11%</td></tr><tr><td>Ours (0.7B)</td><td>65.62%</td><td>78.59%</td><td>4.63%</td><td>96.03%</td></tr></table>

PlaceIt3D adaptation. PlaceIt3D predicts a spatial placement mask rather than a numeric 6D pose. We adapt its selected placement to the target-pose representation required by our evaluator while retaining the original target geometry and fixed scene context, and then apply exactly the same relation and physical-validity checks used for all methods.

Stronger zero-shot LLMs. DeepSeek-V4-Flash and Qwen3.5-397B-A17B receive the same serialized scene metadata and placement instruction used by our zero-shot LLM baseline, without pointcloud tokens or task-specific fine-tuning. Although both models are substantially larger than GALA, their results show that model scale alone does not provide reliable metric 6D placement.

Image-generation and pose-estimation pipeline. We additionally evaluate Nano-Banana + SAM + FoundationPose on 100 randomly sampled test cases. For each case, we render four input views, generate an image containing the target object, segment it, recover its pose, and conservatively retain the best recovered pose across the four views under our evaluator. Table 6 therefore reports this baseline under a favorable multi-view best-pose protocol and compares it with GALA on exactly the same 100 cases.

Table 6. Image-pipeline comparison on the same 100-case subset using four views per case and the best recovered pose. The first row abbreviates the Nano-Banana + SAM + FoundationPose pipeline.
<table><tr><td>Method</td><td></td><td>Within. ↑ Instruction. ↑ Floating. ↓ Support. ↑</td><td></td><td></td></tr><tr><td>Nano-Banana + FoundationPose</td><td>12.82%</td><td>15.00%</td><td>69.00%</td><td>31.21%</td></tr><tr><td>Ours</td><td>66.67%</td><td>79.00%</td><td>6.00%</td><td>92.02%</td></tr></table>

This image-based pipeline is view-dependent, can alter the existing scene during image generation, and accumulates error across synthesis, segmentation, and pose recovery. It also takes approximately one minute per object, compared with approximately one second for direct GALA placement.

## I Stratified Ablation on High-quality Scenes

Table 7 shows that removing high-quality training scenes primarily afects the high-quality test subset, which contains Imaginarium and artist-crafted scenes. The Within Valid Rate drops from 52.82% to 39.47% on this subset, compared with a smaller drop from 68.65% to 65.44% on the 3D-FRONT and SpatialLM-PCG subset. Instruction Valid Rate shows the same concentration, and the Floating Rate increase is confined to the high-quality subset, suggesting high-quality scenes provide useful supervision for designer-curated layouts.

Table 7. Stratified ablation of high-quality training scenes on two heldout test subsets. The high-quality subset contains Imaginarium and artistcrafted scenes; the other contains 3D-FRONT and SpatialLM-PCG scenes. All entries are percentages.
<table><tr><td>Test subset</td><td>Method</td><td>Within. ↑</td><td>Instruction. ↑</td><td>Float. ↓</td><td>Support. ↑</td></tr><tr><td>high-quality</td><td>Full method</td><td>52.82</td><td>71.86</td><td>10.34</td><td>91.41</td></tr><tr><td></td><td>w/o high-quality</td><td>39.47</td><td>64.59</td><td>13.34</td><td>89.56</td></tr><tr><td>3D-FRONT &amp; SpatialLM-PCG Full method</td><td></td><td>68.65</td><td>80.06</td><td>3.38</td><td>97.05</td></tr><tr><td></td><td>w/o high-quality</td><td>65.44</td><td>79.57</td><td>3.17</td><td>97.25</td></tr></table>

## J Fine-Grained Benefits of Point Clouds

Within-sample height localization. Containment often requires selecting a particular shelf or compartment height that cannot be recovered from the container’s overall bounding box. On all 3,165 within samples, we therefore measure the target-center height error

$$
\Delta z _ { i } = \left| z _ { i } ^ { \mathrm { p r e d } } - z _ { i } ^ { \mathrm { g t } } \right| .\tag{10}
$$

Averaged over these samples, Eq. (10) is 0.0747 m with point clouds and 0.0868 m without point clouds, a reduction of approximately 14%. A paired Wilcoxon test gives $p = 6 . 7 \times 1 0 ^ { - 2 0 }$ , showing that this diference is statistically significant.

Table 8. Point-cloud analysis on within samples and in a separate ablation user study.
<table><tr><td>Metric</td><td>w/ point cloud w/o point cloud</td><td></td></tr><tr><td>Height error (m) ↓</td><td>0.0747</td><td>0.0868</td></tr><tr><td>User-study score (1–5) ↑</td><td>4.14</td><td>1.91</td></tr></table>

Text-only ablation user study. In a study separate from the mainpaper comparison, 24 participants rated 23 single-object placement results produced by the point-cloud and text-only variants on a fivepoint scale. As shown in Table 8, the point-cloud model receives a mean score of 4.14, compared with 1.91 for the text-only model. This larger perceptual gap complements the aggregate validity metrics, which do not capture every fine-grained geometric failure.

L-shaped sink counter case. The lower-left result of Fig. 9 in the main paper, labeled Complex Dining room, contains an L-shaped sink counter whose top combines a recessed basin with an unoccupied corner. The predicted placement respects the usable local surface while avoiding both regions, illustrating a shape-dependent decision that an OBB alone cannot express.

## K Generalization Analysis on Unseen Assets

The scene-level split already holds out complete room instances. To test asset-level generalization more directly, we isolate 128 within samples whose reference containers comprise eight asset instances that never appear in training, including bookcases, shelves, islands, chairs, TV stands, and vanities. Table 9 compares these samples with a size-matched set using seen container instances.

Table 9. Asset-level generalization on two size-matched sets of 128 withinrelation samples.
<table><tr><td>Group</td><td></td><td></td><td>Within. ↑ Floating. ↓ Support. ↑</td></tr><tr><td>Seen container, w/ point cloud</td><td>61.72%</td><td>16.41%</td><td>84.21%</td></tr><tr><td>Unseen container, w/ point cloud</td><td>59.38%</td><td>31.25%</td><td>68.34%</td></tr></table>

Containment accuracy degrades on unfamiliar containers, but only mildly: the Within Valid Rate falls by 2.34 percentage points relative to the seen comparison. Physical plausibility degrades considerably more, with the Floating Rate roughly doubling and the Support Ratio dropping by about 16 points, so these results show partial rather than equal robustness on unseen container instances. Both groups consist exclusively of within samples, whose targets must reach an interior support surface rather than the floor, so the rates in this table are structurally higher than aggregate rates over all relation types and are meaningful only relative to each other. This analysis is limited to eight assets and 128 samples from a concentrated set of furniture types, and it evaluates generalization to unseen instances within known categories rather than to unseen categories.

## L Runtime Analysis

Table 10. Runtime breakdown of the end-to-end agent pipeline. API timings depend on the remote service, whereas GALA placement is measured locally.
<table><tr><td>Component</td><td>Frequency</td><td></td><td>Runtime Execution</td><td>Local memory/hardware</td></tr><tr><td>Scene planner</td><td>Once per scene ~100 s</td><td></td><td>API</td><td>一</td></tr><tr><td>Instruction generator Once per object ~70 s</td><td></td><td></td><td>API</td><td></td></tr><tr><td>GALA placement</td><td>Once per object ~1 s</td><td></td><td></td><td>Local inference ~3 GB, 1 NVIDIA H20</td></tr></table>

As Table 10 shows, the two upstream LLM calls dominate end-to-end latency. GALA itself is lightweight at approximately one second per object on an H20 GPU, so faster planning and instruction-generation endpoints or local deployment can reduce total runtime without changing the placement model.

## M Prompt Templates

The agent pipeline uses two system prompts, reproduced below: Listing 1 drives the scene planner, and Listing 2 drives the per-object instruction generator.

You are a professional interior scene planner. Your task is to analyze user requirements and create a detailed scene plan.   
## Your Task   
Understand user's scene requirements   
- Expand and refine the scene description with professional knowledge   
- Generate a complete object list with appropriate sizes and placement order   
- Consider spatial relationships and functional requirements   
## Guidelines   
- Objects should be ordered by priority (larger/anchor/bottom objects first)   
Estimated sizes should be in meters with two decimal places (e.g., [1.20, 0.60, 0.75]) and realistic for the object type. For beds, the   
estimated height should include the headboard height, not just the mattress height. A typical bed with headboard is 1.0-1.4 meters tall.   
- Include necessary furniture based on room type (e.g., bed for bedroom)   
Consider traffic flow and functionality   
- The 'description' field should be specific enough for asset retrieval (include material, color, style)   
- Only include floor-standing furniture and objects that can be placed on/within/under other objects.   
- Do NOT include: ceiling-mounted or suspended items of any kind, especially anything hanging from the ceiling such as chandeliers, pendant   
lights, ceiling lamps, ceiling fans with lights, or any other overhead lighting fixtures.   
- Do NOT include: wall-mounted items (mirrors, shelves, paintings) or any lighting fixtures. If lighting is needed, rely on floor lamps or   
table lamps only.   
- Use "depends\_on" to specify which earlier objects this object's placement depends on (by their IDs). This includes: objects it will be placed   
ON, WITHIN, or UNDER; objects it must be placed NEXT TO or FACING. Standalone objects with no spatial dependency should omit this field   
or use an empty list.   
## Output Format   
You must output a valid JSON object with the following structure:   
{   
"thinking": "Your step-by-step reasoning about the scene design...",   
"scene\_type": "bedroom/livingroom/diningroom/office/...",   
"scene\_description": "A detailed description of the target scene",   
"objects": [   
{   
"id": 0,   
"category": "bed/sofa/table/chair/...",   
"description": "detailed description for asset retrieval",   
"estimated\_size": [width, depth, height]   
},<sub>{</sub>   
"id": 1,   
"category": "nightstand",   
"description": "...",   
"estimated\_size": [0.50, 0.40, 0.55],   
"depends\_on": [0]   
},   
]   
}

## Listing 2. Compact rendering of the system prompt for instruction generation.

```markdown
You are a professional scene layout assistant. Your task is to generate placement instructions for objects in a scene.
## Coordinate and Relation Vocabulary
- Use a right-handed coordinate system: +X = right/east, +Y = front/north, +Z = up.
- Absolute relations:
* Wall: "against east/west/north/south wall"
* Corner: "at northeast/northwest/southeast/southwest corner"
* Region: "in front/back/left/right/center region"
- Relative relations:
* Directional: "left", "right", "in_front", "behind" (edge distance < 1.5m)
* Proximity/orientation: "next_to" (edge distance < 0.5m), "face_to"
* Support/containment: "on", "under", "within"
## CRITICAL: Space Awareness Before Using Direction Relations
Before using a direction relation, compute the object's bounding box and check that the target direction is not blocked by a wall or boundary.
If an object is already against a wall on one side, do NOT place another object further in that direction.
## Guidelines
```

SA Conference Papers ’26, December 01–04, 2026, Kuala Lumpur, Malaysia.

```markdown
- First analyze the scene context and existing objects to understand the scene layout
- Calculate available space in each direction before choosing a relation
- Use existing object IDs and short descriptions as references
- Combine absolute and relative relations when appropriate
- Be specific but concise
- Consider the object's function and typical placement patterns
## Main Objects vs Sub-Objects
- Main objects are standalone room-level objects (e.g., bed, sofa, desk, wardrobe). Set "parent_id": null and include appropriate absolute
relations.
- Sub-objects are placed ON/WITHIN/UNDER a parent object (e.g., pillow on a bed, book on a shelf, chair under a desk). Set "parent_id" to the
parent ID and omit absolute relations.
## Relation Rules
- Include at most 3 relations total, prioritized as: on/within/under > next_to > directional.
- Use at most one wall relation; use a corner relation instead of combining two wall relations.
- For any sub-object resting on or contacting a bed or sofa, use "within" rather than "on".
- If a parent already has sub-objects, add a directional relation to an existing sibling to avoid collision.
## Output Format
{
"thinking": "Your step-by-step reasoning about where to place the object...",
"parent_id": null,
"expected_relations": {
"relative_relations": [
{"type": "relation_type", "target_id": reference_object_id}
],
"absolute_relations": ["relation_type"]
}
}
## Representative Examples
- Main object: next_to + directional relation to an existing anchor, with an optional wall relation.
- Sub-object: support/containment relation to its parent only.
Crowded parent: support/containment relation to the parent plus a directional relation to an existing sibling.
```