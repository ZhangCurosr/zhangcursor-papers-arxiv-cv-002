# PVRA: A Pointwise Key-point Voting Framework for Robotic Assembly

Kulunu Samarawickrama, Roel Pieters

Tampere University, Finland

Abstract. Modern computer vision has enabled partial autonomy in robotic assembly manipulation. However, performing autonomous manipulation of a progressive assembly demands a more specific set of skills, in addition to perceiving the objects. Through a comparative analysis of research in the associated domains, we deduce that object-centric perception must advance towards learning assembly dependencies to predict meaningful actionable outputs for autonomous assembly manipulation. Subsequently, we present a 3D keypoint-based modular learning framework to learn assembly dependencies to infer actionable outputs given a RGB-D input of an assembly scene. We train and evaluate our trained network on an assembly pose estimation dataset and compare it against object-centric baselines with an augmented set of metrics for progressive assemblies. The source code of the proposed framework is hosted at https://github.com/KulunuOS/PVRA

Keywords: Robotic assembly· RGB-D perception· 6-DoF assembly pose · progressive assembly· key-point voting.

## 1 Introduction

Assembling parts together to obtain a final product is a trivial task for humans. However, achieving a similar level skill for a robot is a complex challenge to date. Research on robotic assembly spans over multiple domains with diverse applications and variable constraints. Some examples include; robotic manufacturing, service robots, medical robots and applications, space explorations, fractured part reconstruction in archaeology, repair and restoration applications, etc. There is continuous demand for advanced assembly capabilities in robots despite extensive amount of studies. Although current research addresses these challenges via visual perception and reinforced learning, it remains debatable whether these exhibit behaviors of an intelligent robot with assembly task awareness. In this paper we introduce our notion of assembly task awareness and an end-to-end learning method to enable it for robotic manipulation.

Computer vision tools such as object detection, classification and semantic segmentation are paramount to endow perception to robots. while these tools provide basic scene understanding, robotic manipulation requires 3D spatial understanding of the robot environment. Consequently, 6 DoF object pose estimation emerged as a critical component in robotic perception which has become pivotal in robotic grasping and object placement, over the time. However, a multi-object assembly task imposes more requirements on context awareness. Spatial awareness is crucial in localizing pre-assembled and post-assembled poses of assembly objects. Assembling the objects in an optimal sequence requires temporal awareness. Inter-part relational awareness decides the structural coherence of a final assembly. Moreover, awareness on the stability under gravity is an essential physical feasibility.

The above observations formulate the main components of task awareness required for a progressive assembly task. This awareness can be interpreted as the ability of an entity to perceive an assembly configuration and reason about its spatial, temporal and relational dependencies. A module that can learn to produce assembly-aware actionable outputs is extremely beneficial for robotic assembly manipulation. This is critical for modern robotics as it enables manipiulation capabilities beyond primitive pick and placement operations towards task-aware manipulation. To this end we introduce our method PVRA: A Pointwise Key-point Voting Framework for Robotic Assembly. The main contributions of this paper are as below:

i. A 3D keypoint-based modular learning framework to enable assembly task awareness for robotic entities

ii. An augmented set of metrics to benchmark assembly-ware role segmentation and pose estimation.

iii. Implementation and evaluation of the proposed framework on a custom object assembly dataset

iv. A comparative evaluation of the proposed model against object-centric baselines.

## 2 Related Work

Assembly has long been a major research topic in process engineering, particularly under the domains of Assembly/Disassembly Sequence Planning (ASP/DASP) and Assembly/Disassembly Path Planning (APP/DAPP). The main aim of such work is to optimize manufacturing processes by Assembly/Disassembly planning (AP/DAP) [4], [20]. Traditionally, these focused on deterministic planning of actions and trajectories based on geometric modeling and combinatorial reasoning. However, with the rapid advancements in robotics and automation, a research direction focusing on transforming classical AP to a cognitive skill based on dynamic field of artificial intelligence has emerged. While this shares the same goals as AP, the majority of existing research focused on enabling AP via perceptiondriven and learning-based methods. The Table 1 summarizes state-of-the-art methods and their respective learning concepts, network architectures, inputs, outputs and datasets utilized.

Table 1: Categorization of assembly pose estimation methods. GNN denotes Graph Neural Network, VNN denotes Vector Neuron Network, CNN denotes Convolutional Neural Network.
<table><tr><td rowspan=1 colspan=1>Method(Year)</td><td rowspan=1 colspan=1>LearningConcept</td><td rowspan=1 colspan=1>Architecture / Input / Output /Dataset</td></tr><tr><td rowspan=1 colspan=1>ASAP [16](2022)</td><td rowspan=1 colspan=1>Assembly bydisassembly</td><td rowspan=1 colspan=1>GNNs; Point cloud; Assembly sequence;Custom [16]</td></tr><tr><td rowspan=1 colspan=1>GPAT [9] (2023)</td><td rowspan=1 colspan=1>Targetsegmentation</td><td rowspan=1 colspan=1>Transformers; Point cloud; Assemblypose; PartNet [10]</td></tr><tr><td rowspan=1 colspan=1>PAST [24](2023)</td><td rowspan=1 colspan=1>Assembly bydisassembly</td><td rowspan=1 colspan=1>Transformers; Point cloud; Assemblysequence and assembly pose; D4PAS [24]</td></tr><tr><td rowspan=1 colspan=1>Neural ShapeMating [2](2022)</td><td rowspan=1 colspan=1>Shape priorsreconstruction</td><td rowspan=1 colspan=1>Adversarial Network; Point cloud;Assembly pose; Thingi10k [23], GSO [3],Shapenet [1]</td></tr><tr><td rowspan=1 colspan=1>RGL-Net [5](2022)</td><td rowspan=1 colspan=1>Progressiverelationalreasoning</td><td rowspan=1 colspan=1>Recurrent GNN; Point cloud; Assemblypose; PartNet [10]</td></tr><tr><td rowspan=1 colspan=1>InstanceEncodedTransformer [22](2022)</td><td rowspan=1 colspan=1>Inter-partrelational learning</td><td rowspan=1 colspan=1>Transformers; Point cloud; Assemblypose; PartNet [10]</td></tr><tr><td rowspan=1 colspan=1>Dynamic GraphLearning [8](2020)</td><td rowspan=1 colspan=1>Part relationreasoning</td><td rowspan=1 colspan=1>GNN; Point cloud; Assembly pose;PartNet [10]</td></tr><tr><td rowspan=1 colspan=1>SE-3 PartAssembly [21](2023)</td><td rowspan=1 colspan=1>Part correlationsvia SE(3)equivariance</td><td rowspan=1 colspan=1>VNN; Mesh files; Assembly pose; GSMdataset [2], Breaking-Bad [15]</td></tr><tr><td rowspan=1 colspan=1>This work:PVRA (2025)</td><td rowspan=1 colspan=1>Modular learning</td><td rowspan=1 colspan=1>CNN; RGB-D; Assembly cognition;6DAPose [14]</td></tr></table>

## 2.1 Assembly Sequence Planning

Assembly Sequence Planning (ASP) constitutes the task of inferring a physically feasible and collision-free sequence of operations on assembly objects leading to an assembled product. "Assemble them All" [17] proposes a purely simulation based collision-free tree-search strategy for ASP. ASAP [16] builds upon the same concept and extends it to a learnable model that accounts for gravitational stability of components, which enhances its applicability to real-world applications. The core principle employed in these works to explore the physical feasibility is called "Assembly by Disassembly". Exploiting this principle, PAST [24] proposes a transformer network with reasoning capability for ASP at multiple levels of the assembly, based on point clouds and assembly pose estimation as an auxiliary task.

## 2.2 Assembly Pose Estimation

Given a set of parts or components Assembly Pose Estimation (APE) entails inferring the target object’s 6 DoF pose in the assembled product while satisfying the assembly constraints. APE, in the absence of any prior knowledge about semantics of the parts, can be defined as generalized part assembly. Generalized part assembly is capable of assembling novel or unseen objects given a target assembly. GPAT [9] addresses this problem through "Target segmentation", where the network explicitly segments the target blue print and infers the assembly poses. However, GPAT assumes a global and single-step APE, which overlooks any intermediate steps in a progressive assembly task. This creates an adverse efect on feasibility and accuracy in performing a progressive assembly task due to the lack of progressive reasoning and incorporating physical constraints.

Progressive part assembly requires more specific reasoning capabilities such as temporal awareness and inter-part relationships. "Instance Encoded Transformer" [22] introduces an approach to implicitly learn a sequential model for progressive assembly using an attention mechanism and instance encoding. It is efective in APE scenarios where the assembly sequence is critical. However, this approach relies on assemblies with a fixed sequence, limiting its temporal awareness. Therefore, it is less efective in scenarios with assembly ambiguities. RGL-Net [5] proposes a recurrent graph-based learning method which explicitly learns inter-part relationships with sequence encoding. It features iterative refinement of APE over time by updating part relationships in the graph structure, leading to improved generalization. Similar work by Huang et al. [8] implements a dynamic graph learning method which learns to assemble parts from a coarseto-fine manner. It is robust against assembly ambiguities with better temporal awareness, making it suitable for assemblies with variable part configurations and flexible sequences.

Apart from the methods which aim at modeling and learning inter-part relationships, there exist other modern methods which leverage geometric priors for APE. For example, "Neural shape mating" [2] employs adversarial networks to learn how to mate two shapes by first bisecting them into two complementary parts. This enables APE in a self-supervised manner, using only geometric alignment without contextual knowledge on semantics or structure. Another study presents how the SE(3)-Equivariance [21] quality can be exploited to learn transformation-consistent geometric features using Vector Neuron Networks (VNNs). These features, which maintain a predictable behavior under 3D rotations and translations, can be aggregated to generate pose-invariant representations of parts, enabling multi-object APE in a one-shot fashion. However, they are not capable of learning temporal dependencies, inter-part relationships or assembly ambiguities, which are essential for progressive assembly tasks.

## 2.3 Key-Point based methods

Point cloud-based methods have proven to be more useful for robot perception with the advent of RGB-D sensors due to their capability in extracting 3D spatial data from the real world [13][12][18]. However, point clouds obtained from

RGB-D sensors can observe only a partial surface and are often noisy. Moreover, learning directly on point clouds becomes complex due to the size of point clouds. Alternatively, keypoint-based learning methods overcome above drawbacks as they learn on a unique sub-sample of a large point cloud as shown by PVNet[11] and PVN3D[6]. These methods are robust against occlusions, making them ideal for practical robot applications. Our proposed method is inspired by PVN3D [6], which solely aims on object localization. In contrast, we re-purpose the foundation for enabling novel assembly cognition, addressing broader challenges in the robotic assembly domain.

## 2.4 Assembly Datasets

As evident from Table 1, a majority of methods use the popular PartNet [10] dataset to train and evaluate their approach. PartNet contains 24 classes that belong to general everyday categories with part-level annotations. There are ample datasets available for geometry-based APE that mainly contain mesh files of objects, such as Google Scanned Objects [3], Thingi10k [23], etc. However, these datasets lack crucial information required for progressive part assembly, such as the assembly sequence, intermediate assembly steps, stability under gravity, and other ambiguities. To address this gap, ASAP [16] presents their own custom dataset with 2,146 assemblies, with a subset of 240 stable assemblies to train their assembly sequence planning method. Similarly, PAST [24] presents D4PAS, a dataset curated for progressive assembly tasks with 1500+ mechanical part assemblies. In this work, we use the 6DApose object assembly dataset [14], which extends the standard object pose estimation benchmark dataset format; BOP [7].

## 3 Problem Definition

Given an RGB-D input I of an assembly scene with a set of assembly objects $\{ O _ { i } \} _ { i = 1 } ^ { C }$ in either disassembled or partially assembled state, where C is the total number of objects, the final assembled product H can be defined as where $O _ { i }$ is the object to be assembled and its respective 6 DoF assembly pose $T _ { i } \in S E ( 3 )$ . Instead of predicting all poses in a one-shot manner, we formulate a progressive assembly approach by assembling parts sequentially over a fixed number of steps $t \in \{ 1 , 2 , \ldots , C \}$ . Therefore progressive assembly $H _ { t }$ at step t can be represented as

$$
H _ { t } = H _ { t - 1 } \cup T _ { t } \big ( O _ { t a r g e t } \big ) , \quad t = 1 , 2 , \ldots , C ,\tag{1}
$$

where $O _ { t a r g e t }$ is the target object to be assembled at current step and its assembly pose is $T _ { i } ~ \in ~ S E ( 3 )$ . An example of a progressive object assembly consisting of 4 assembly steps is illustrated in Fig 1. Furthermore, we define a set of assumptions on progressive assemblies addressed in this work. The progressive assembly is assumed to have (i) a fixed assembly sequence, (ii) a single and explicitly defined target object and its 6 DoF pose per assembly step and (iii) assembly objects are stably positioned under gravity. To further simplify the assembly complexity we assume, (iv) only one object is in contact with the target object when it’s assembled, which is hereafter referred to as the base object. The reason for these assumptions is due to the complex nature and ambiguities in assembly tasks, which extends beyond the scope of the research problem, and to enable testability while maintaining reproducibility to implement our proposed method on diferent object assemblies. Thus, given an $I _ { R G B - D }$ dataset with relevant ground-truth data of a progressive assembly $H _ { t } ,$ , the primary objective of our proposed method is to learn a supervised function $f \left( I _ { R G B - D } \right)$ that identifies the target object $O _ { t }$ and localizes its 6 DoF pose $Q _ { t } \in \dot { S } E ( 3 )$ , and estimates its assembly pose $T _ { t } \in S E ( 3 )$ w.r.t to camera optical frame.

## 4 Proposed Method

We propose PVRA: a supervised RGB-D point-cloud model (see Fig. 1) for progressive assembly under the specific assumptions described in Section 3.

![](images/58332b8d466e68a120e21742a44539d64a66a0b1206509bc640814e558eadd8e.jpg)  
Fig. 1: Proposed 3D keypoint-based learning framework for assembly cognition. MLP denotes Multi Layer Perceptrons.

The method perceives a RGB-D input of the assembly scene and extracts features to learn a mapping that produces assembly task-aware actionable outputs; point-wise semantic roles ( target , base and background), 6-DoF pose of target object and its 6-DoF assembly pose. PVRA model does not represent a complete controller that executes an assembly task, but it predicts task-aware actionable outputs that enable such downstream tasks for autonomous agent to execute an assembly action at current step of a progressive assembly. The model shares RGB-D feature fusion strategy from PVN3D[6], but adopts prediction heads and supervision towards progressive assembly task.

## 4.1 RGB-D Feature extraction and Fusion

The depth image from the RGB-D input is back-projected into a point-cloud and sampled in to N points. Each point represents features: 3D coordinate, RGB value and surface normal. A pre-trained CNN-based network extracts semantic features from RGB input while the sampled pointcloud is processed by Pointnet++ feature extractor. The two feature streams are fused to a shared feature representation. For N number of visible points sampled from a RGB-D input of $H _ { t }$ , the extracted feature map will be hereafter referred as $F _ { t }$

## 4.2 Semantic Role Segmentation Module (SEG)

The module predicts a per-point semantic probability map as a probability distribution $S _ { t }$ over C object classes: target, base and the background. In contrast to conventional semantic segmentation, these classes depend on the current assembly step and same object may be a target in one step and a base in a later step. The segmentation head outputs point-wise role logits. The learning model for module SEG is described below where $S _ { t } ( p ) \in \mathbb { R } ^ { C }$ and $S _ { t } ^ { ( c ) } ( p )$ is the probability that point $p$ belongs to object class $C$

$$
S _ { t } = \mathrm { S E G } ( F _ { t } ) , \quad \sum _ { c = 1 } ^ { C } S _ { t } ^ { ( c ) } ( p ) = 1 .\tag{2}
$$

## 4.3 Key-point ofset prediction module (KpOF)

The $K p O F$ module predicts a set of Manhattan distances $L _ { t }$ from a visible set of points $\{ p _ { i } \} _ { i = 1 } ^ { N }$ to a predefined set of 3D keypoints of the target object at pre-assembled state. This head is supervised only on the target-role points and the target 6-DOF pose is recovered by calculating the optimal transformation between predicted keypoints and object-model keypoints.

$$
L _ { t } = \operatorname { K p O F } ( F _ { t } ) .\tag{3}
$$

## 4.4 Assembly key-point ofset prediction module (AKpOF)

$A K p O F$ module predicts Manhattan distance ofsets $D _ { t }$ from same set of visible points $\{ p _ { i } \} _ { i = 1 } ^ { N }$ to the same set of predefined 3D keypoints of the target object in assembled state. This head is supervised only on the base-role points and the target 6-DOF assembly pose is recovered by calculating the optimal transformation between predicted keypoints and object-model keypoints.

$$
D _ { t } = \mathrm { A K p O F } ( F _ { t } ) .\tag{4}
$$

## 4.5 Key-point Sampling

Key-points are manually annotated for all assembly objects using their respective CAD files using farthest point sampling algorithm. They serve as ground-truth inputs for subsequent key-point ofset prediction modules $K p O F$ and $A K p O F$ We initialize 8 key-points for each object as a pragmatic design choice which provides suficient representation of spatial extent of objects and a manageable and consistent output dimensionality of key-point ofset prediction heads $( K \times$ 3 channels per object). The number of keypoints are fixed and object scale independent as the networks learn scale normalized distance ofsets rather than absolute values. The aim is to sample suficient sparse set of geometric anchors from which ofsets can be inferred rather than densely representing the object. Although we do not present a full ablation over the number of keypoints, our work is consistent with prior 3D keypoint-based pose estimation work [6].

## 4.6 Learning Objective

PVRA shares a joint learning objective over the three shared prediction heads. In contrast to multi-task learning where each module learns an independent task, PVRA learns the same progressive assembly task. We jointly supervise the modules SEG, $K p O F$ and $A K p O F$ using a combined loss function ${ \mathcal { L } } .$ . Segmentation head adopts focal loss and ofset prediction heads adopt ofset loss.

$$
\mathcal { L } = ( 2 \cdot \mathcal { L } _ { \mathrm { S E G } } + \mathcal { L } _ { \mathrm { K p O F } } + \mathcal { L } _ { \mathrm { A K p O F } } ) / 4 .\tag{5}
$$

## 5 Evaluation

## 5.1 Experimental setup

We train and validate PVRA on the Nema17 gear-reducer progressive assembly dataset[14]. The dataset contains simulated assembly scenes with ground-truth object poses, assembly poses, RGB-D frames at each assembly step and CAD files of associated objects. The dataset consists of 431 assembly instances of 5 assembly objects and 4 assembly steps. This yields 8620 instances which are used to generate 60% train split , 20% validation split and a 20% test split. The network was trained on a high-performance computing cluster consisting of 4

NVIDIA Tesla V100 GPUs with 128GB of VRAM. The source code is hosted at the GitHub repository: https://github.com/KulunuOS/PVRA.

A direct comparison of PVRA against related work summarized in Table 1 is non-trivial. PVRA infers assembly task-aware actionable outputs on partial pointclouds obtained from RGB-D view samples grounded in a reference frame. In contrast most related methods expect complete pointclouds representing the assembly blueprints and they difer in problem formulations and assumptions. We compare PVRA against two CAD based pose estimation baselines. First, CAD-ICP-PCA, which initializes pointcloud registration from mask centroids and principal component axes before ICP refinement. Second, a modern CAD-based RGB-D pose estimator: FoundationPose[19]. These methods estimate 6-DoF pose of target and base object and calculate assembly pose with known base-target transformation, whereas PVRA has to infer above outputs with learned assembly awareness. Both baselines, assume ideal object segmentation but PVRA predicts role labels on sampled RGB-D points. Therefore, we compare their performance under both GT-masks which represent dense object silhouettes and PVRA-predicted-masks with sparse points. This compares the performance of PVRA against baselines when object boundaries are underrepresented in segmentation.

## 5.2 Metrics

Step-Segmentation Accuracy (SSA): SSA measures assembly step-level segmentation accuracy using the average Intersection-over-Union (IoU) of the target and base roles. It reflects PVRA’s ability to identify the active target and base objects with temporal and relational awareness:

$$
\mathrm { S S A } = \frac { 1 } { 2 } \left( \mathrm { I o U } _ { \mathrm { t a r g e t } } + \mathrm { I o U } _ { \mathrm { b a s e } } \right) .\tag{6}
$$

Step-Localization Accuracy (SLA): Pose accuracy is measured using normalized Maximum Symmetry-Aware Surface Distance (MSSD). Continuous local Z symmetries are declared for all Nema17 objects so that equivalent rotations are not penalized. For target-object diameter d, we compute:

$$
e _ { t a r g e t } = \frac { \mathrm { M S S D } _ { t a r g e t } } { d } , e _ { a s s e m b l y } = \frac { \mathrm { M S S D } _ { a s s e m b l y } } { d } , e _ { s t e p } = \frac { 1 } { 2 } ( e _ { t a r g e t } + e _ { a s s e m b l y } ) .\tag{7}
$$

Step Acc@0.15d is reported as the primary fixed-threshold score, together with accuracies at 0.05d, 0.10d, and 0.20d. Target AUC, Assembly AUC, and SLA-AUC summarize threshold accuracy over 0 to 0.50d, which is the main continuous pose metric.

## 5.3 Quantitative results

Table 2 summarises the evaluation of PVRA and baselines on Nema17 test split. The number of samples evaluated for each method difer because each method can fail for diferent reasons: PVRA can fail during keypoint voting, CAD-ICP-PCA can fail due to insuficient segmented points for registration, and FoundationPose can reject small masks. These sample counts are therefore interpreted together with accuracy.

Table 2: Quantitative evaluation results for the Nema17 test split using symmetry-aware normalized MSSD. ICP-GT denotes CAD-ICP-PCA with annotation masks, ICP-PVRA denotes CAD-ICP-PCA with masks derived from PVRA segmentation, FP-GT denotes FoundationPose with annotation masks, and FP-PVRA denotes FoundationPose with masks derived from PVRA segmentation.
<table><tr><td rowspan=1 colspan=1>Metric/Method</td><td rowspan=1 colspan=1>PVRA</td><td rowspan=1 colspan=1>ICP-GT</td><td rowspan=1 colspan=1>ICP-PVRA</td><td rowspan=1 colspan=1>FP-GT</td><td rowspan=1 colspan=1>FP-PVRA</td></tr><tr><td rowspan=1 colspan=1>Samples</td><td rowspan=1 colspan=1>280</td><td rowspan=1 colspan=1>215</td><td rowspan=1 colspan=1>227</td><td rowspan=1 colspan=1>282</td><td rowspan=1 colspan=1>227</td></tr><tr><td rowspan=1 colspan=1>SSA</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Target AUC</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.576</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.770</td></tr><tr><td rowspan=1 colspan=1>Assembly AUC</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.420</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.881</td></tr><tr><td rowspan=1 colspan=1>SLA-AUC</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.379</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.927</td><td rowspan=1 colspan=1>0.727</td></tr><tr><td rowspan=1 colspan=1>Step Acc@0.15d</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.286</td><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>0.771</td></tr></table>

![](images/a204cf64fc9153b6b7019e138d0f5a5c17750c43f17855dcee9c71da3af11ed3.jpg)  
Fig. 2: Step-level threshold accuracy curves.

Fig 2 shows a comparison of corresponding step-level threshold-accuracy curves. FoundationPose with ground-truth masks reaches high accuracy at very strict thresholds, while PVRA improves sharply between 0.05d and 0.15d ,while CAD-ICP-PCA curves fail to reach the accuracy of learning-based methods consistently.

## 5.4 Qualitative results

![](images/e19d0ff3c1e0c53400a5938324c7bfbc16cd28d0b19b9376b9d862cebf1beda3.jpg)  
Fig. 3: Qualitative comparison on two Nema17 test instances. Columns show assembly instances, while rows compare PVRA, FoundationPose (FP), and CAD-ICP-PCA (ICP). Green boxes indicate target-pose estimates and red boxes indicate assembled-target estimates; dashed or dotted boxes indicate the corresponding ground-truth poses. T/A reports target and assembled-target MSSD in millimetres. $e _ { \mathrm { s t e p } }$ is the normalized step MSSD, and pass/fail indicates whether the sample satisfies Step Acc@0.15d.

## 6 Discussion

The evaluation reveals several methodological and empirical findings on assemblyaware pose prediction. First, CAD registration based methods are less efective when the scene consists of partial pointclouds. CAD-ICP-PCA performed poorly with both ground-truth masks and PVRA-predicted masks compared to other methods. The partial occlusions in progressive assembly make it further challenging for registration-based methods. Second, methods that highly dependent on accurate object masks are not reliable for assembly-aware pose prediction. FoundationPose predictions are highly accurate with ground-truth masks but accuracy drops with sparse PVRA-predicted masks while still performing well under strict thresholds compared to PVRA. However, its step SLA-AUC declines indicating that PVRA provides more consistent step-level pose accuracy among all threshold values. Third, FoundationPose with PVRA-predicted masks processed 53 fewer samples than PVRA. It demonstrates robustness of PVRA under unavailability of dense object silhouettes as masks and uncertain or under-represent object boundaries. Fourth, FoundationPose baseline require prior knowledge on specific target and base object relationship corresponding to assembly sequence and relative pose transformation from base object to assembly pose. In contrast, PVRA predicts both poses from an RGB-D pointcloud representation in an end-to-end pipeline without explicit knowledge of the assembly instance while achieving a competitive accuracy.

The evaluation uses a synthetic dataset of a progressive assembly. A realworld deployment may introduce additional challenges such as sensor noise, calibration errors, reflections and lighting conditions. While domain transferability remains a future research direction of this work, the simulation-based baseline comparison contributes an important reproducible benchmark for perceptionbased progressive assembly. As evident from the literature study, the availability of real-world structured datasets that represent progressive assemblies are limited. This is due to the complexity of capturing synchronized RGB-D frames and annotated 6-DoF poses in a real-world setting. Therefore, this controlled setting is essential for isolating the methodological question; can assembly context-aware model achieve comparable performance to existing object-centric pose estimation methods?.

As a summary, the results show that PVRA addresses an important gap in robotic assembly manipulation that is not yet completely solved using objectcentric perception. The average Nema17 object diameter is approximately 51 mm; therefore, the reported Step Acc@0.15d threshold corresponds to a normalized symmetry-aware surface-distance tolerance of about 7-8 mm for an average part. PVRA achieves a Step Acc@0.15d of 0.8929 which indicates 89.29% of the total predictions are accurate under the threshold tolerance. This demonstrates the model’s capability to learn spatial, relational and temporal dependencies within a progressive assembly and predict actionable outputs for downstream manipulation tasks.

## 7 Conclusion

This work presented a task-aware keypoint-based framework for progressive robotic assembly and a quantitative evaluation compared with existing objectcentric perception methods. The evaluation confirms the ability of the model to learn assembly-task specific dependencies and produce accurate actionable outputs for robotic manipulation. Furthermore, this work motivates extending assembly-specific awareness towards broader task-specific awareness learning which can benefit a wider range of robotic manipulation applications beyond the scope of progressive assembly.

Acknowledgments. Project funding was received from - Helsinki Institute of Physics’ Technology Programme (project; ROBOT) and Research Council of Finland under Grant no. 369003 (PERFORM). The authors wish to acknowledge CSC - IT Center for Science, Finland, for computational resources.

## References

1. Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., et al.: ShapeNet: An Information-Rich 3D Model Repository (2015), https://arxiv.org/ abs/1512.03012

2. Chen, Y.C., Li, H., Turpin, D., Jacobson, A., Garg, A.: Neural shape mating: Selfsupervised object assembly with adversarial shape priors. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12724–12733 (2022). https://doi.org/10.1109/CVPR52688.2022.01239

3. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., et al.: Google Scanned Objects: A high-quality dataset of 3D scanned household items. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 2553– 2560 (2022)

4. Ghandi, S., Masehian, E.: Review and taxonomies of assembly and disassembly path planning problems and approaches. Computer-Aided Design 67-68, 58–86 (10 2015). https://doi.org/10.1016/j.cad.2015.05.001

5. Harish, A.N., Nagar, R., Raman, S.: RGL-NET: A recurrent graph learning framework for progressive part assembly. In: IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 78–87 (2022). https://doi.org/10.1109/WACV51458. 2022.00072

6. He, Y., Sun, W., Huang, H., Liu, J., Fan, H., Sun, J.: PVN3D: A deep point-wise 3D keypoints voting network for 6DoF pose estimation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11632–11641 (2020)

7. Hodaň, T., Michel, F., Brachmann, E., Kehl, W., GlentBuch, A., Kraft, D., et al.: BOP: Benchmark for 6D object pose estimation. In: European Conference on Computer Vision (ECCV). pp. 19–34 (2018)

8. Huang, J., Zhan, G., Fan, Q., Mo, K., Shao, L., Chen, B., et al.: Generative 3D part assembly via dynamic graph learning. In: Advances in Neural Information Processing Systems. pp. 6315–6326 (2020)

9. Li, Y., Zeng, A., Song, S.: Rearrangement planning for general part assembly (2023), https://arxiv.org/abs/2307.00206

10. Mo, K., Zhu, S., Chang, A.X., Yi, L., Tripathi, S., Guibas, L.J., Su, H.: PartNet: A large-scale benchmark for fine-grained and hierarchical part-level 3D object understanding. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 909–918 (2019)

11. Peng, S., Liu, Y., Huang, Q., Zhou, X., Bao, H.: PVNet: Pixel-wise voting network for 6DoF pose estimation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4561–4570 (2019). https://doi.org/10.1109/ CVPR.2019.00470

12. Qi, C.R., Litany, O., He, K., Guibas, L.J.: Deep hough voting for 3D object detection in point clouds. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9276–9285 (2019)

13. Qi, C.R., Liu, W., Wu, C., Su, H., Guibas, L.J.: Frustum pointnets for 3D object detection from RGB-D data. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 918–927 (2018)

14. Samarawickrama, K., Pieters, R.: 6DoF assembly pose estimation dataset for robotic manipulation. Data in Brief 56, 110834 (2024). https://doi.org/https: //doi.org/10.1016/j.dib.2024.110834

15. Sellán, S., Abdrashitov, R., Thies, J., Jacobson, A.: Breaking Bad: A dataset for geometric fracture and reassembly. ACM Transactions on Graphics 41(6), 1–15 (2022). https://doi.org/10.1145/3550454.3555442

16. Tian, Y., Willis, K.D.D., Omari, B.A., Luo, J., Ma, P., Li, Y., et al.: ASAP: Automated sequence planning for complex robotic assembly with physical feasibility. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 4380– 4386 (9 2024)

17. Tian, Y., Xu, J., Li, Y., Luo, J., Sueda, S., Li, H., Willis, K.D.D., Matusik, W.: Assemble them all. ACM Transactions on Graphics 41(6), 1–11 (2022). https: //doi.org/10.1145/3550454.3555525

18. Wang, C., Xu, D., Zhu, Y., Martín-Martín, R., Lu, C., Fei-Fei, L., Savarese, S.: DenseFusion: 6D object pose estimation by iterative dense fusion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3343–3352 (2019)

19. Wen, B., Yang, W., Kautz, J., Birchfield, S.: Foundationpose: Unified 6d pose estimation and tracking of novel objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17868–17879 (June 2024)

20. Wilson, R.H., Latombe, J.C.: Geometric reasoning about mechanical assembly. Artificial Intelligence 71 (1994). https://doi.org/10.1016/0004-3702(94)90048-5

21. Wu, R., Tie, C., Du, Y., Zhao, Y., Dong, H.: Leveraging SE(3) equivariance for learning 3D geometric shape assembly. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 14311–14320 (2023)

22. Zhang, R., Kong, T., Wang, W., Han, X., You, M.: 3D part assembly generation with instance encoded transformer. IEEE Robotics and Automation Letters 7(4), 9051–9058 (2022). https://doi.org/10.1109/LRA.2022.3188098

23. Zhou, Q., Jacobson, A.: Thingi10K: A dataset of 10,000 3D-printing models (2016), https://arxiv.org/abs/1605.04797

24. Zhu, X., Jha, D.K., Romeres, D., Sun, L., Tomizuka, M., Cherian, A.: Multi-level reasoning for robotic assembly: From sequence inference to contact selection. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 816–823 (2024)