# SSC-Priors: Exploring Semantic and Visibility Priors to Boost Lidar Semantic Scene Completion

Tetiana Martyniuk<sup>1,2\*</sup>, Jonathan Seele<sup>1,3</sup>, Alexandre Boulch<sup>1,2</sup>, Gilles Puy<sup>1,2</sup>, Renaud Marlet<sup>1,2,4</sup>, Raoul de Charette<sup>1</sup>

<sup>1</sup>Inria, Paris, France. <sup>2</sup>valeo.ai, Paris, France. <sup>3</sup>ETH Z¨urich, Switzerland. <sup>4</sup>LIGM, CNRS, Univ Gustave Eifel, ENPC, IP Paris, France.

\*Corresponding author(s). E-mail(s): tetyanka.martynyuk@gmail.com; Contributing authors: jseele@ethz.ch; alexandre.boulch@valeo.com; gilles.puy@gmail.com; renaud.marlet@valeo.com; raoul.de-charette@inria.fr;

## Abstract

This paper investigates easy strategies to boost the performance of existing networks for lidar semantic scene completion (SSC) without requiring complex architectural redesigns. The fact is that, over the last years, SSC methods have mostly pursued architectural innovations, making the models heavier and more complex, e.g., by jointly training a point cloud semantic segmentation branch. In this work, we take a step back and explore two priors used as simple ingredients (possibly noisy) to improve existing approaches: semantic pseudo-labels and sensor visibility information. Concretely, we provide both kinds of information directly as additional inputs to a given SSC network, requiring only a minimal adaptation of the original architecture. We first demonstrate that endowing input point clouds with semantic pseudo-labels from of-the-shelf segmenters significantly improves the performance of existing SSC models. In fact, by evaluating these models against an oracle, we establish that highquality semantic priors are a primary driver of semantic gains (mIoU), and that the SSC model can be trained just once with ground-truth semantics and then exploited without retraining using any segmenter. Furthermore, we equip the input lidar point cloud with visibility information that distinguishes between empty spaces (between the lidar and a scanned point) and unknown spaces (outside of lines of sight), providing a secondary performance boost across the tested architectures. We study the design space of data for representing visibility information and bound the remaining headroom with a ground-truth oracle on the free-space labels. On SemanticKITTI, these enhancements make older models competitive with state-of-the-art systems across four architectures, in one case even outperforming them. On the SSCBench-nuScenes benchmark, both priors also transfer with the sparser 32-beam sensor. Our code is available at https://github.com/astra-vision/SSC-Priors.

![](images/3a15d76ed2dff829d0d909cbbeb1a804d72ab23704132492ba66e56677a01f89.jpg)  
Fig. 1 Impact of priors on lidar semantic scene completion (SSC) on SemanticKITTI. Our systematic study reveals that semantics and visibility are readily-available priors that can significantly boost the performance of single-frame SSC baselines. While conventional lidar SSC relies solely on point locations as input, we augment the latter with semantics, from of-the-shelf 3D segmenters, and visibility information obtained through ray-based sampling to infer empty space. Left: Carefully combining both priors transforms early methods into competing ones and pushes the best-performing method to a new state of the art among fully-reproducible single-frame methods. Right: Qualitative results on SemanticKITTI for SemCity-AE (top) and SSA-SC (bottom): the baseline, the same model with our input priors, and the ground truth.

## 1 Introduction

Perceiving and understanding the 3D environment is a critical task for agents interacting with the physical world. For robots, such as self-driving cars, this is particularly important and requires comprehensive sensing of the scene’s geometry and semantics. In fact, in recent years, the task of semantic scene completion (SSC) has seen a surge of interest [1–4]. SSC jointly addresses the estimation of both geometry and semantics. The output is typically a voxel grid where each voxel is assigned a label indicating whether it is free (i.e., empty) or occupied with a given semantic class.

In the literature, SSC has been addressed using various kinds of sensor inputs such as cameras [5– 7], lidar [8], or radar [9], possibly taking time into account [10]. In this work, we address SSC from lidar, using a single scan as input. Lidar provides accurate 3D geometry but, unlike a camera, it is sparse and lacks photometric information. This combination makes both sub-problems (geometry and semantics) hard, especially from a single sweep. Existing methods range from lightweight BEV completion networks [11] to computationally intensive difusion models [8], reflecting a recent trend favoring increasingly large, intricate architectures.

A recurring feature is the use of auxiliary priors derived from the input, such as semantic or visibility cues. In existing methods, these priors are usually entangled with the architecture, e.g., produced by a branch trained jointly with the geometric completion head [12–14], or introduced at the loss level by leveraging information originating from several frames [15]. Because each method exploits priors diferently, the contribution of a particular prior is not isolated from the specific benefit of the corresponding architecture.

We take a diferent stance: we supply both kinds of priors (semantic and visibility cues) to an existing SSC model as additional input data, minimally adapting the architecture to accommodate them. This model is then trained under the same conditions as the original one. This approach gives complete freedom to the network to possibly exploit the prior information. Notably, neither prior comes from a jointly trained branch. This lets us measure the impact of each prior on its own. Moreover, keeping the source of priors decoupled from the completion network makes prior cues freely replaceable, e.g., when a better segmenter becomes available.

In our experiments, both priors are consumed by the completion network at the voxel level.

The semantic cues are semantic class labels predicted per point by a frozen, of-the-shelf point cloud segmenter [16, 17] trained for the same target classes as the SSC model, then aggregated by majority vote to each voxel containing at least one scanned point. A semantically labeled voxel is deemed occupied, while the status of other voxels, which do not contain any points, is unknown. As for the visibility cues, they consist of another label indicating which voxels are expected to be empty. These labels are obtained by casting rays between scanned points and the sensor. The status of unobserved voxels remains unknown.

In our study, we also bound each prior with a ground-truth oracle, which is not deployable but reveals how much of the residual error is geometric versus semantic.

Equipping four established completion networks with these input priors makes older, lightweight models competitive with current stateof-the-art systems, even surpassing them in one case (Fig. 1, left), and visibly sharpens their predictions (Fig. 1, right). What is more, this performance is achieved at no architectural cost beyond widening the input layer.

Our contributions are as follows:

• We introduce a simple setting to study the impact of adding semantic and visibility priors into existing SSC networks, in order to improve their performance.

• We study the impact of the quality of pseudolabels as semantic cues, both at train and test time, and show it is enough to train the SSC network once with ground-truth labels, then use any semantic segmenter at inference.

• We explore the design space of visibility cues and show the gain persists across lidar densities.

• Our experiments, spanning two datasets and four existing architectures, show that the semantic and geometric performance of SSC models can be systematically boosted with our priors. In this setting, older SSC methods can even outperform current state-of-the-art models, shedding new light on the performance bottlenecks of SSC approaches.

This article is an extended version of our conference paper “Exploring Easy Boosts for Lidar

Semantic Scene Completion” [18]. Additions to the conference paper are described in App. A.

The remainder of the paper is organized as follows. Section 2 reviews related work on lidar and camera SSC, prior injection, and ray-based visibility reasoning. Section 3 serves as an ablation study and analyzes each prior in isolation on SemanticKITTI. Section 4 evaluates the combination of the best semantic and visibility cues across four diferent architectures and two datasets (SemanticKITTI and SSCBenchnuScenes), establishes segmenter-agnostic deployment, and analyzes where the gains concentrate and how much headroom the oracle priors leave. Section 5 concludes the study. The Appendices report visibility prior precompute profiling, and detail additional results, e.g., classwise segmentation tables.

## 2 Related work

## 2.1 Semantic Scene Completion

SSC was originally formulated as a densification task from a single depth image [19]. The dual task aims to predict both the geometry (occupancy of occluded regions beyond sensor view) and the semantics of the scene. The problem has then been studied for various input sensors.

Lidar SSC. Enabled by large-scale automotive lidar benchmarks [1–3], early approaches like TS3D [20] were adapted to lidar inputs by processing voxelized TSDF point clouds with 3D CNNs [1]. LMSCNet [11] introduced dedicated lidar SSC by projecting the point cloud into a 2D Bird’s-Eye-View (BEV) representation, treating the height axis as channels to enable lightweight 2D CNN completion. Most subsequent methods adopt discriminative 3D or sparse convolutional backbones [12–14, 21], incorporating BEV features [13, 21] or multi-task auxiliary semantic streams [12, 14, 22]. Moving beyond voxels, Local-DIF [23] encodes the lidar with a point network [24] to learn SSC as an ensemble of local implicit functions, enabling arbitrary-resolution inference. LODE [25] similarly completes the scene implicitly, conditioning an Eikonal signed-distance formulation on local shape priors to handle nonwatertight, large-scale lidar scenes. Recently, diffusion methods [8, 26–28] have framed SSC as conditional generation. Lee et al. [27] and SemCity [26] keep dense volumetric difusion tractable using a discrete latent space and triplane representations, respectively. DifSSC [8] instead difuses point positions and semantic classes directly in point space, conditioned on a semantically segmented input scan.<sup>1</sup> Octree latent difusion [28] extends the compressed-representation line with a sparse octree structure. SemCity further treats SSC as a downstream refinement task conditioned on predictions from existing networks, rather than end-to-end generation. We do not adopt this refinement: as detailed in Sec. 4.2, its oficial implementation leaks ground-truth occupancy.

Camera SSC. Although we target lidar SSC, parallel progress in camera-based SSC ofers highly relevant design patterns. MonoScene [5] established monocular SSC by lifting 2D features into a 3D voxel grid. VoxFormer [29] then introduced a two-stage transformer that uses a depth-derived visibility/occupancy prior to seed voxel queries, mirroring our own use of visibility priors on the lidar side. OccFormer [30], TPVFormer [31], and SurroundOcc [32] extend camera-based occupancy prediction to multi-view and tri-perspective settings, and Symphonies [33] further explores occlusion-aware design.

Multimodal SSC has also been investigated. Here lidar, camera and radar may be used jointly, possibly using several successive frames [9, 34– 37]. While not directly comparable to single-frame lidar SSC, these methods confirm a broader trend: explicit modeling of what is observed, versus what remains unobserved, is increasingly recognized as a primary lever for occupancy prediction, regardless of input modality.

## 2.2 Priors in SSC

To ease joint geometric and semantic prediction, several methods leverage auxiliary priors.

Semantic priors are most commonly introduced through explicit multi-task supervision. SSA-SC [14] and JS3C-Net [12] couple the completion network with a learned point-cloud segmentation branch, whereas S3CNet [21] fuses a separately-trained 3D completion network and 2D BEV variant post hoc. JS3C-Net additionally uses the completion output as a contextual shape prior fed back into segmentation. This kind of architecture tightly couples completion performance to the specific architecture and training of the joint segmenter: replacing or upgrading the segmenter requires retraining the completion network. SSC-RS [13] disentangles the two tasks, separating semantic and geometric representations before fusing them in BEV. Taking an alternative approach to auxiliary guidance, SCPNet [38] leverages dense-to-sparse knowledge distillation, transferring semantic and structural relations from a multi-frame teacher to regularize the single-frame student.

Our study focuses on the systematic injection of an of-the-shelf, frozen segmenter directly into the input tensor to quantify its exact performance headroom. While sequential or conditional methods like TS3D [20], DifSSC [8], and LODE’s [25] semantic extension have previously leveraged decoupled semantic predictions, we contrast explicitly with multi-task methods that lock completion performance to a jointly trained internal branch. This decoupling not only allows us to establish clear oracle bounds, but also proves segmenter-agnostic transferability (see Sec. 4.5).

Visibility priors in lidar SSC are seldom used. TALoS [15] performs ray casting from multiple lidar frames and uses the resulting visibility map to drive test-time adaptation of the completion head; it requires temporal sequences (including future frames) and per-sequence optimization.

For camera-based SSC, visibility has been more actively studied: VoxFormer [29] estimates camera depth to seed sparse voxel queries, and recent works further decouple the prediction of visible and occluded regions either at the architectural [39] or supervision [40] level.

Our study operates on single frames, with no test-time adaptation, and trains the network only once, before inference, to exploit visibility information, as opposed to [15]. Also, we exploit visibility at input level, as a precomputed signal consumed by the SSC network, requiring only a minor adaptation of the first input layers. To our knowledge, this is the first study of explicit empty-space visibility priors decoupled from the completion architecture in single-frame lidar SSC.

## 2.3 Lidar semantic pseudo-labels

Three-dimensional semantic segmentation has matured rapidly through point-based [24], sparseconvolutional [17, 41], BEV-projection [16], and transformer-based [42, 43] architectures. Stateof-the-art segmenters now exceed 70% mIoU on SemanticKITTI-lidarseg [16, 42], well above the SSC mIoU of current SSC methods. We quantify this gap directly in Tab. 1 via oracle experiments. Said gap motivates our use of a strong of-the-shelf segmenter as a frozen teacher whose predictions are fed as part of the input of the SSC model, contrary to the internally coupled multi-task methods discussed above (Sec. 2.2).

Outside SSC, the pattern of consuming frozenteacher predictions as input features for a downstream 3D task is established by PointPainting [44], which decorates lidar points with frozen 2D-segmenter scores for 3D object detection. Related ideas appear in pseudo-label self-training for cross-domain 3D detection [45].

Our study raises an open question: how does the quality of the pseudo-label teacher at inference interact with the quality of the labels seen at training time? In fact, we show that an SSC network trained with ground-truth semantic input generalizes well to inference-time pseudo-labels, eliminating the need to retrain the completion network whenever a stronger semantic teacher becomes available.

## 2.4 Ray-based visibility reasoning

Lidar point clouds are commonly processed as unordered point sets, discarding the sensor geometry. However, the line of sight between the sensor and each return carries information about traversed free space and occluded regions.

Classical occupancy-mapping methods exploit this directly via ray casting, e.g., in OctoMap framework [46]. Rold˜ao et al. [47] refine the standard binary per-scan update by modeling the distance each ray traverses within a cell and downweighting observations by a range-dependent density term. It helps reduce the discretization artifacts, in particular spurious free space from grazing or distant rays that arise when a cell’s state is set from a single coarse traversal. In learned 3D perception, Hu et al. [48] showed that visibility features substantially boost lidar object detection.

Ray-based reasoning has been successfully leveraged for map cleaning and dynamic-object removal [49, 50] or as a pretraining objective for lidar backbones [51]. More recently, explicit ray-based supervision has also been applied to volumetric occupancy prediction in the camera setting, such as in RenderOcc [52] and UniOcc [53], where rendered rays supervise the voxel field. Closer to our setting, Sulzer et al. [54] augmented point clouds with sensor-visibility information for surface reconstruction, demonstrating that input-level visibility can be consumed by downstream networks with minimal architectural adaptation. While early methods like TS3D [20] and S3CNet [21] incorporated visibility inherently through (flipped) TSDF inputs, explicit ray-based free-space priors have otherwise received little attention in recent single-frame lidar SSC architectures (cf. Sec. 2.2).

Our study explores various eficient visibility representations and evaluates their impact when provided at input level of the SSC network.

## 3 Study of individual priors

In this section, after defining our simple way to integrate raw priors into existing lidar SSC architectures (Sec. 3.1) and presenting our experimental setup (Sec. 3.2), we study two input-level priors that boost SSC performance and can be applied to any existing completion network with only minor input-layer adaptations: a semanticized input point cloud (Sec. 3.3) and visibility information representing the emptiness of the lines of sight between the lidar and the scanned surfaces (Sec. 3.4).

We analyze each prior in isolation, contrasting realistic of-the-shelf prior estimators against ground-truth oracles. The oracles are not deployable, but they bound the performance achievable with a perfect prior and thus reveal the remaining room for improvement with respect to the prior. We combine the two priors and evaluate their association in Sec. 4.

## 3.1 Integration of raw priors in a lidar SSC architecture

Formally, lidar SSC takes a sparse lidar point cloud P and outputs a completed volume in which every voxel is assigned one of the $C + 1$ labels: C semantic classes, and the empty (or free) class.

![](images/a4eb2e27594fe881e4615764848e5d23eef1ee6cc4164f25b30443ffb8f10a99.jpg)  
Fig. 2 Method overview. Given a single lidar scan, we build an augmented input grid $\mathbf { V } \in S ^ { X \times Y \times Z }$ over the full $C + 2$ state set ${ \cal S } = \{ 1 , \ldots , C ,$ empty, unknown}, as detailed in Sec. 3.1. Semantics comes from a frozen, of-the-shelf point cloud segmenter; visibility separates empty from unknown voxels via single-scan ray casting. Both enter the completion network only at the input.

Most SSC methods first voxelize the point cloud of the input scan into a grid on which the backbone operates. This raw data supports only two states per voxel: a voxel is occupied if it contains at least one lidar point and unknown otherwise. In the second case, the sensor provides no evidence that an empty-looking voxel is truly free rather than merely unobserved. We write this input as a per-voxel categorical grid

$$
\mathbf { V } \in { \mathcal { S } } ^ { X \times Y \times Z } , \qquad { \mathcal { S } } = \{ { \mathrm { o c c u p i e d } } , { \mathrm { ~ u n k n o w n } } \} ,
$$

$i . e . , \ | S | \ = \ 2$ states, over a 3D voxel grid of dimensions $X \times Y \times Z$

Our “recipe” to add priors enriches this state set at the input only, adding no new module and training no auxiliary network. Each prior we study enlarges S along one axis: the semantic prior (Sec. 3.3) refines the bare occupied state into C semantic classes, and the visibility prior (Sec. 3.4) splits unknown into (observably) empty and unknown. Their combination (Sec. 4) yields a C + 2 state set. Feeding this categorical grid to a backbone requires only a backbone-specific input adaptation: widening the first layer to a one-hot encoding of S for grid-convolutional networks, a learned per-state embedding, or a concatenation to the per-point features that a point backbone already ingests. No other architectural change is needed. For a backbone that does not consume a single grid input (e.g. JS3C-Net [12]), the two components of V (the semantic state on occupied voxels and the empty/unknown partition) need not enter as one tensor: each is routed to the pathway the architecture already exposes. (Per-architecture details are in Sec. 4.1).

Without loss of generality, we present below the priors based on this voxel-grid input representation. It however directly extends to a pointbased backbone: the semantic prior then adds semantic classes to input points rather than voxels, and the visibility prior adds free-space information as extra points marked empty in the point cloud.

## 3.2 Experimental setup

Dataset. We conduct our prior studies on the SemanticKITTI validation set (sequence 08) [1], using the oficial SemanticKITTI ground-truth voxel grid of $2 5 6 \times 2 5 6 \times 3 2$ at 0.2 m resolution with C = 19 semantic classes.

Metric. We report two complementary metrics. Geometric completion is measured with the binary occupied-vs-empty voxel intersection-over-union (IoU). Semantic completion quality is measured with the mean intersection-over-union (mIoU) over the C semantic classes (thus excluding free space), where the IoU for each individual semantic class is computed over the voxels that are not empty in at least the ground truth (GT) or the prediction. (Unobserved areas, as defined in the GT, are excluded from the computation of both metrics). Beyond the dual nature of the SSC task, reporting the two metrics separately is central to our analysis, since each prior is expected to act primarily on one of the two subtasks.

Augmented SSC methods. We run the study on individual priors with two lightweight networks: LMSCNet-SS [11] and the SemCity autoencoder [26] repurposed for SSC (SemCity-AE). Both are chosen deliberately for being eficient, yet not state-of-the-art (SOTA) regarding both the semantic and geometric metrics (5 to 7 IoU pts and about 10 mIoU pts below the best reproducible method, DPS2CNet, cf. Tab. 4), so that the efect of an input prior is clearly observable rather than partially masked by an already strong backbone. We however later demonstrate that our findings translate to stronger backbones (Sec. 4).

For a controlled comparison, all variants of LMSCNet-SS are trained for 80 epochs with its original scheduler, and all variants of SemCity-AE are trained for 50 epochs with a cosine-annealing scheduler.

## 3.3 Semantic prior

Some SSC methods exploit the ground-truth (GT) semantics of the input point cloud (or its voxelized representation) by training a separate, explicit segmentation branch jointly with the completion network, either as a first stage in the pipeline [12] or as a parallel branch interacting with completion [14]. Lidar semantic segmentation is, however, a mature field with strong networks available of-the-shelf. We therefore isolate semantics as an individual prior: a frozen, independently pretrained segmenter supplies per-point labels, which we voxelize directly into the input tensor V.

## Point labels as semantic prior

Concretely, the single occupied state is refined into C semantic classes, and the new input voxel grid $\mathbf { V } _ { \mathrm { w / s e m } }$ is built as follows. All voxels are first set to the unknown state. Every voxel containing at least one lidar point is then considered as occupied and assigned a single semantic class by majority vote over the points it contains. Every non-occupied voxel remains unknown. This enlarges the state set to $S _ { \mathrm { w / s e m } } = \{ 1 , \ldots , C _ { \mathrm { w } }$ unknown}, i.e., C + 1 states. Therefore, the only work to create the variant based on the semantic prior is to increase from 2 to C + 1 the number of channels representing the state, with no change to the downstream architecture.

Table 1 Semantic prior efect on LMSCNet-SS and SemCity-AE for SemanticKITTI, with various sources of semantic labels. Labels are used both at train and test time; TTA only at inference. ∆ measures the diference with the vanilla model (no semantic labels). <sup>†</sup>baseline is retrained.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1-frame inputSemanticsof lidar labels     mIoU %</td><td rowspan=1 colspan=2>SSC outputgeometry semanticsIoU % ΔmIoU % ∆</td></tr><tr><td rowspan=1 colspan=1>LMS</td><td rowspan=1 colspan=1>CNet-SS [11]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>(1)(2)(3)(4)</td><td rowspan=2 colspan=1>none (vanilla model)†SalsaNext [55]        55.8</td><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>16.5</td></tr><tr><td rowspan=1 colspan=1>54.7-0.9</td><td rowspan=1 colspan=1>21.3+4.8</td></tr><tr><td rowspan=1 colspan=1>WaffleIron [16]        68.0</td><td rowspan=1 colspan=1>55.6 +0.0</td><td rowspan=1 colspan=1>21.6+5.1</td></tr><tr><td rowspan=1 colspan=1>MinkUNet [17]        70.0</td><td rowspan=1 colspan=1>55.1-0.5</td><td rowspan=1 colspan=1>22.3+5.8</td></tr><tr><td rowspan=1 colspan=1>(5)</td><td rowspan=1 colspan=1>WaffleIron + TTA [16]70.3</td><td rowspan=1 colspan=1>55.6 +0.0</td><td rowspan=1 colspan=1>21.9 +5.4</td></tr><tr><td rowspan=1 colspan=1>(6)</td><td rowspan=1 colspan=1>GT (oracle)          100.0</td><td rowspan=1 colspan=1>56.3 +0.7</td><td rowspan=1 colspan=1>30.4 +13.9</td></tr><tr><td rowspan=1 colspan=1>SemC</td><td rowspan=1 colspan=1>ity-AE [26]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>(1)(2)(3)(4)</td><td rowspan=3 colspan=1>none (vanilla model) †[SalsaNext [55]         55.8WaffleIron [16]        68.0</td><td rowspan=1 colspan=1>53.9</td><td rowspan=1 colspan=1>16.4</td></tr><tr><td rowspan=1 colspan=1>54.1 +0.2</td><td rowspan=1 colspan=1>23.0+6.6</td></tr><tr><td rowspan=1 colspan=1>54.4 +0.5</td><td rowspan=1 colspan=1>25.5+9.1</td></tr><tr><td rowspan=1 colspan=1>MinkUNet [17]        70.0</td><td rowspan=1 colspan=1>54.2 +0.3</td><td rowspan=1 colspan=1>25.5+9.1</td></tr><tr><td rowspan=1 colspan=1>(5)</td><td rowspan=1 colspan=1>WaffleIron + TTA [16] 70.3</td><td rowspan=1 colspan=1>54.4 +0.5</td><td rowspan=1 colspan=1>26.0 +9.6</td></tr><tr><td rowspan=1 colspan=1>(6)</td><td rowspan=1 colspan=1>GT (oracle)          100.0</td><td rowspan=1 colspan=1>54.6 +0.7</td><td rowspan=1 colspan=1>32.7 +16.3</td></tr></table>

Please note that semantic information is very sparse in the voxel grid: it only concerns the semantics of the points in the input frame, not the completed volume in the GT constructed from accumulated frames.

In the following, unless otherwise stated, the adapted network is both trained and tested with the same semantics-augmented input.

## Impact of pseudo-labels on SSC semantics

To create our semantic prior, the input scan is first segmented by an of-the-shelf semantic segmentation network, before being fed to the inputwidened SSC network. Tab. 1 reports results with (1) vanilla SSC models (i.e., with no semantic labels at input) and using four segmenters or variants to create pseudo-labels: (2) SalsaNext [55], (3) WafleIron [16] (WI), (4) MinkUNet [17], and (5) WafleIron with test-time augmentation (WI-TTA), with input label quality ranging from 55.8% to 70.3%.

![](images/6fae8fe1a863027e9d355d672ddf6343005fc2db8dd07890a60336bf62b2d075.jpg)

![](images/4019f5788f672bdbfd6eadb61847c80bc65a5982c602f023359b8797e24b5705.jpg)  
Fig. 3 Completion quality vs input prior quality on SemanticKITTI for LMSCNet-SS and SemCity-AE. Left: Semantic prior: completion mIoU vs input-segmentation mIoU. Each point is an of-the-shelf segmenter from Tab. 1 or the GT oracle; the no-semantics (vanilla) baseline sits at x = 0. The shaded region marks headroom recoverable as segmenters improve. Right: Visibility prior: completion IoU vs agreement (IoU) between the empty voxels of the input prior and those of the oracle, i.e., dense markers denoised with the ground truth, which sits at $x = 1 0 0$ . Each point is a free-space marker placement, labeled with its row number in Tab. 2; the no-marker (vanilla) baseline sits at x = 0. The shaded region marks the headroom recoverable by denoising the dense markers.

We observe that the pseudo-labels provided on the input frame lead to a systematic semantic completion gain: +4.8 to +5.8 mIoU pts for LMSCNet-SS, and +6.6 to +9.6 pts for SemCity-AE. This is a very significant gain considering also that it relies only on an extremely simple extension of the input to an SSC model, and retraining.

The benefit is moreover robust to the choice of segmenter, as diferent as they may be (e.g., 2D dense convolution on a BEV projection for WaffleIron vs 3D sparse convolution for MinkUNet). Completion mIoU broadly increases with the mIoU of the semanticized input frame (see Fig. 3, left), although segmenters of comparable input quality are not separated: MinkUNet and Wafle-Iron+TTA difer by 0.3 input mIoU pt and rank diferently on the two networks.

## Oracle headroom

To upper-bound the gain attainable from such an input semantics, we also consider constructing $\mathbf { V } _ { \mathrm { w / s e m } }$ from the GT-annotated input points.

Table 1 (rows (6)) shows that the efect is almost entirely on the semantic axis: mIoU rises by +13.9 pts for LMSCNet-SS (to 30.4% mIoU) and +16.3 pts for SemCity-AE (to 32.7% mIoU).

We observe that the pseudo-labels used as augmented input recover a large fraction of the semantic oracle gain. Yet, a substantial share of the semantic completion headroom remains unrealized: +8.1 mIoU pts for LMSCNet-SS (after already gaining +5.8 mIoU pts with pseudolabels), and +6.7 mIoU pts for SemCity-AE (after gaining +9.6 pts). These SSC gaps with the oracle are attributable to the remaining ∼30-pt gap in input mIoU between the best of-the-shelf segmenter used here (70.3% input mIoU) and the perfect input segmentation. Thanks to the modular nature of our pipeline, part of this gap can however be recovered “for free” when external segmenters improve, with no change to the completion network.

Experiments in Sec. 4 also show that, for simplicity, instead of retraining with the pseudo-labels of a new segmenter, it is even enough to train just once with the semantic oracle and then to infer with pseudo-labels from any segmenter.

## Impact of the semantic prior on geometry

While we may expect some kind of coupling between semantics and geometry (the historical motivation of SSC [19]), the efect of semantic pseudo-labels on geometric completion is small and even occasionally slightly negative: between −0.9 and +0.5 IoU pt.

Table 2 Visibility prior efect on LMSCNet-SS and SemCity-AE for SemanticKITTI, with variants for placing free-space markers. Markers are used both at train and test time. ∆ measures the diference with the vanilla model (no markers). <sup>†</sup>baseline is retrained.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1-frame inputRay-sampledfree-space markers</td><td rowspan=1 colspan=2>SSC outputgeometry  semanticsIoU % Δ mIoU % ∆</td></tr><tr><td rowspan=1 colspan=1>LMSC</td><td rowspan=1 colspan=3>Net-SS [11]</td></tr><tr><td rowspan=2 colspan=1>(1)(2)</td><td rowspan=2 colspan=1>none (vanilla model) †1 pt before the hit</td><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>16.5</td></tr><tr><td rowspan=1 colspan=1>55.8+0.2</td><td rowspan=1 colspan=1>16.9+0.4</td></tr><tr><td rowspan=1 colspan=1>(3)</td><td rowspan=1 colspan=1>1 pt at random</td><td rowspan=1 colspan=1>56.0+0.4</td><td rowspan=1 colspan=1>16.8 +0.3</td></tr><tr><td rowspan=1 colspan=1>(4)</td><td rowspan=1 colspan=1>10 pts at random</td><td rowspan=1 colspan=1>57.1+1.5</td><td rowspan=1 colspan=1>17.6 +1.1</td></tr><tr><td rowspan=1 colspan=1>(5)</td><td rowspan=1 colspan=1>25 pts at random</td><td rowspan=1 colspan=1>57.2+1.6</td><td rowspan=1 colspan=1>17.8 +1.3</td></tr><tr><td rowspan=1 colspan=1>(6)</td><td rowspan=1 colspan=1>50 pts at random</td><td rowspan=1 colspan=1>57.0+1.4</td><td rowspan=1 colspan=1>17.7 +1.2</td></tr><tr><td rowspan=1 colspan=1>(7)</td><td rowspan=1 colspan=1>100 pts at random</td><td rowspan=1 colspan=1>57.6+2.0</td><td rowspan=1 colspan=1>17.8 +1.3</td></tr><tr><td rowspan=1 colspan=1>(8)</td><td rowspan=1 colspan=1>dense (δ = voxel size)</td><td rowspan=1 colspan=1>57.5+1.9</td><td rowspan=1 colspan=1>18.1 +1.6</td></tr><tr><td rowspan=1 colspan=1>(9)</td><td rowspan=1 colspan=1>dense with dilation</td><td rowspan=1 colspan=1>56.9+1.3</td><td rowspan=1 colspan=1>17.3 +0.8</td></tr><tr><td rowspan=1 colspan=1>(10)</td><td rowspan=1 colspan=1>oracle (dense denoised)</td><td rowspan=1 colspan=1>63.6+8.0</td><td rowspan=1 colspan=1>19.6+3.1</td></tr><tr><td rowspan=1 colspan=1>SemC</td><td rowspan=1 colspan=1>ity-AE [26]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>(1)(2)</td><td rowspan=2 colspan=1>none (vanilla model)†1 pt before the hit</td><td rowspan=1 colspan=1>53.9</td><td rowspan=1 colspan=1>16.4</td></tr><tr><td rowspan=1 colspan=1>54.7+0.8</td><td rowspan=1 colspan=1>16.8 +0.4</td></tr><tr><td rowspan=1 colspan=1>(3)</td><td rowspan=1 colspan=1>1 pt at random</td><td rowspan=1 colspan=1>55.6+1.7</td><td rowspan=1 colspan=1>17.5 +1.1</td></tr><tr><td rowspan=1 colspan=1>(4)</td><td rowspan=1 colspan=1>10 pts at random</td><td rowspan=1 colspan=1>56.5 +2.6</td><td rowspan=1 colspan=1>17.5 +1.1</td></tr><tr><td rowspan=1 colspan=1>(5)</td><td rowspan=1 colspan=1>25 pts at random</td><td rowspan=1 colspan=1>56.5+2.6</td><td rowspan=1 colspan=1>17.7 +1.3</td></tr><tr><td rowspan=1 colspan=1>(6)</td><td rowspan=1 colspan=1>50 pts at random</td><td rowspan=1 colspan=1>56.6+2.7</td><td rowspan=1 colspan=1>17.9 +1.5</td></tr><tr><td rowspan=1 colspan=1>(7)</td><td rowspan=1 colspan=1>100 pts at random</td><td rowspan=1 colspan=1>56.7+2.8</td><td rowspan=1 colspan=1>17.9 +1.5</td></tr><tr><td rowspan=1 colspan=1>(8)</td><td rowspan=1 colspan=1>dense (δ = voxel size)</td><td rowspan=1 colspan=1>56.9+3.0</td><td rowspan=1 colspan=1>17.5 +1.1</td></tr><tr><td rowspan=1 colspan=1>(9)</td><td rowspan=1 colspan=1>dense with dilation</td><td rowspan=1 colspan=1>56.3+2.4</td><td rowspan=1 colspan=1>17.4+1.0</td></tr><tr><td rowspan=1 colspan=1>(10)</td><td rowspan=1 colspan=1>oracle (dense denoised)</td><td rowspan=1 colspan=1>60.6 +6.7</td><td rowspan=1 colspan=1>20.0+3.6</td></tr></table>

With oracle semantic labels, the efect on completion is positive but remains small: the IoU gains +0.7 IoU pt on both SSC networks. In fact, the IoU of these models remains at 56.3 and 54.6%, which is still some 4.5 to 6 pts below DPS2CNet [56], the strongest reproducible method in geometry (60.8% IoU, cf. Tab. 4).

## Conclusion

This study of LMSCNet-SS and SemCity-AE networks shows that, thanks to an extremely simple semantic extension of the input to an SSC model, and retraining, it is possible to obtain significant SSC gains regarding semantics, up to +9.6 mIoU pts. While input semantics, if not too noisy, naturally is a dominant driver of the quality of SSC semantics, it however has little impact on the quality of SSC geometry.

Geometric completion thus seems to be chiefly insensitive to semantic quality and is the subject of the visibility prior, discussed below (Sec. 3.4).

## 3.4 Visibility prior

The 3D points of a single lidar scan only provide very sparse information to reconstruct a complete scene, which makes SSC a particularly dificult task. Yet, lidar point clouds provide more information when the sensor location is known, which is generally the case. Rather than uniformly consider all space besides observed points as unknown, it is then possible to distinguish observed free space (between the sensor and scanned points) from unobserved free space (behind scanned points, between scan rays). Early lidar SSC methods already encode this distinction implicitly through (flipped) TSDF inputs [20, 21], and TALoS [15] exploits ray-cast visibility at test time over multiple frames. We now study an explicit visibility prior that recovers this distinction from the lidar lines of sight.

Given a single lidar scan, we cast one ray for each lidar point, between the sensor origin and the measured 3D point, and place along each ray a number of special 3D points as free-space markers. Markers and lidar points are voxelized into the 256 × 256 × 32 grid $\mathbf { V } _ { \mathrm { w / v i s } }$ by majority vote, with lidar points taking precedence over free-space markers within any shared voxel. The result is a ternary grid: a voxel is (i) empty if it contains a free-space marker (hence is traversed by at least one ray) and no lidar point, (ii) occupied if it received a lidar point, or (iii) unknown otherwise. This enlarges the input states to S = {occupied, empty, unknown}. The visibility prior thus keeps the occupied voxel state untouched, but splits the baseline unknown voxel state into “observably empty” and “remaining unknown”.

Note that, as for semantic information, this visibility prior can be imperfect. Beyond possible noise in lidar scans (reflections, missing returns, ego vehicle speed compensation, etc.), there can also be lidar registration issues that impact the accumulation of scans when building an SSC ground truth and create inconsistencies with free spaces of single scans, which are rarely taken into account when constructing such a GT.

As in Sec. 3.3, we study here the visibility prior in isolation: occupied voxels only carry bare occupancy, not semantic classes. Combining semantic and visibility priors is addressed in Sec. 4.

## Impact of visibility marker distribution

The construction above leaves open how densely and where free-space markers are placed along each ray, ofering several variants. Diferent variants trade precompute cost against the fidelity of the recovered free space (see discussion below).

In Tab. 2, we compare a variety of marker distributions: (1) none, i.e., vanilla model with no prior; a single marker placed (2) just before the scanned point, or (3) at a random position along the ray; (4-7) n markers at random positions with n respectively in {10, 25, 50, 100}; and (8) a dense uniform marker sampling that places a marker every voxel size (0.2 m in our experiments) on the ray, (9) optionally followed by a safety margin of 5 × 5 × 5 voxels around each lidar point, that demotes near-surface traversed voxels back to the default unknown state (“dense with dilation”<sup>2</sup> in Tab. 2), creating a kind of safety volume to minimize discretization artifacts caused by grazing rays slightly penetrating occupied voxels. (We tried a few more heuristics but every variant we tested merely slides along a precision/recall frontier). The dense uniform variant, which is our default setting, has about 60 points per ray on average.

All variants improve completion over the visibility-free baseline. From 10 markers per ray on, they lie within 0.7 IoU pt of each other on both networks: having enough markers matters more than where they are placed. Unlike the random placements, the dense uniform variant is deterministic, and its empty voxels agree best with their ground-truth-denoised version (Fig. 3, right). It reaches 57.5% IoU on LMSCNet-SS and 56.9% on SemCity-AE, i.e., +1.9 and +3.0 IoU pts over the prior-free models.

## Oracle headroom

To bound how much accuracy single-scan visibility leaves unrealized, we evaluate a denoising oracle that runs the same dense uniform ray casting, but uses GT only to purge false-empty markers, relabeling them as unknown. It typically corresponds to cases of grazing rays that clip a surface.

Using this oracle as visibility prior provides a significant gain on geometric completion, with +8.0 and +6.7 IoU pts on LMSCNet-SS and SemCity-AE, respectively, compared to the vanilla models. This leaves plenty of room for improvement over the simple dense uniform sampling, with +6.1 and +3.7 IoU pt margin, respectively.

## Impact of the visibility prior on semantics

We observe a mild but consistent improvement on semantics when using a visibility prior: up to +1.6 and +1.1 mIoU pts using dense cues on LMSCNet-SS and SemCity-AE, respectively. The efect is a bit higher than the slight occupancy improvement observed when, conversely, exploiting the semantic prior (Sec. 3.3): occupancy indirectly helps more the semantics than the other way around.

Denoising the dense visibility prior with an oracle further benefits semantics, achieving an extra +1.5 and +2.5 mIoU pts on LMSCNet-SS and SemCity-AE, respectively, to +3.1 and +3.6 mIoU pts with respect to the vanilla models. The virtual gain on semantics of this oracle geometric information is not negligible.

## Conclusion

These experiments show that the information of lidar lines of sight, exploited purely at the input to recover the empty/unknown distinction, yields a consistent completion boost (up to +3.0 IoU pts), alongside a mild semantic gain, and at no architectural cost beyond an extra input channel.

While the visibility prior has slightly more impact on semantics than the semantic prior has on geometric completion, the coupling of semantics and occupancy remains weak and the two input priors are chiefly complementary. We thus expect the combination of the semantic and visibility priors to compound rather than overlap.

In the following section (Sec. 4), we therefore combine both priors into a single augmented input and evaluate the resulting recipe, which involves a full C + 2 state set that carries, on every voxel, both the semantic class on occupied voxels and the empty/unknown visibility partition on the other voxels.

## 4 Prior combination

We now evaluate whether the complementarity hypothesized in Sec. 3 holds and applies beyond the two tested SSC methods and the single dataset used in experiments. Concretely, we evaluate our combination of priors on four architectures (still the lightweight LMSCNet-SS [11] and $_ \mathrm { S e m C i t y - A E }$ [26], but also the stronger SSA-SC [14] and JS3C-Net [12]) and two datasets (SemanticKITTI [1] and SSCBench-nuScenes [2], which features a sparser lidar sensor).

## 4.1 Merging priors in SSC networks

We combine the semantic and visibility prior information into a single input tensor $\mathbf { V } _ { \mathrm { w / s e m + v i s } }$ that carries, on every voxel, both the semantic class (applying to occupied voxels) and the empty/unknown visibility partition (applying to voxels without lidar points). This yields the following full state set, of size $C + 2 \cdot$

$$
\begin{array} { r } { S _ { \mathrm { w / s e m + v i s } } = \{ 1 , \dots , C , \mathrm { e m p t y } , \mathrm { u n k n o w n } \} . } \end{array}
$$

Our recipe to integrate the priors touches only the input pathway that each backbone already exposes. It adds no auxiliary network and introduces no new jointly trained branch. It merely routes the priors through entry points that the architecture already provides or, where present, a pre-existing jointly trained branch. The entry point is nonetheless backbone-specific; we detail them below for reproducibility.

SemCity-AE. Originally designed to process dense, fully annotated point clouds, we adapt SemCity’s autoencoder [26] for SSC by supplying the sparse augmented tensor $\mathbf { V } _ { \mathrm { w / s e m + v i s } }$ and adjusting its input embedding layer to fit. Its prior-free baseline counterpart instead encodes a binary per-voxel occupancy state, capturing only whether at least one lidar point is present in the voxel.

LMSCNet-SS. The LMSCNet method [11] projects 3D spatial volumes into 2D feature maps by collapsing the height axis, converting an $X \times Y \times Z$ grid into an $X \times Y$ map with $Z = 3 2$ channels. We preserve this layout but expand each height bin to carry the complete one-hot vector, resulting in $X \times Y$ maps with $( C + 2 ) \cdot Z$ channels. Only the initial convolutional layer is widened to accept this expanded feature depth.

SSA-SC. In SSA-SC [14], the prior information is integrated exclusively into the BEV completion U-Net. Replacing binary occupancy with our (C+ 2)-dimensional one-hot encoding (covering semantics, empty space, and unknown states) expands the U-Net input width from 32 + 32 channels (pillar occupancy plus point features) to (C +2)·32+ 32. The PointNet module and 3D segmentation branches remain identical to the baseline network. In our experiments we use the oficial code, where the 3D semantic segmentation branch gets supervision only from the 2D completion branch.

JS3C-Net. Finally, JS3C-Net [12] incorporates the two priors at distinct entry points. The visibility prior enters the completion module as an additional channel flagging observably empty voxels, mirroring the setup of other backbones. Conversely, the semantic prior lacks a direct pathway to the completion input; it is instead injected into the jointly trained per-point segmentation branch by concatenating one-hot pseudo-labels to the point-wise feature vectors. We discuss below the impact of this indirect routing scheme.

## 4.2 Experimental setup

Datasets. In addition to SemanticKITTI, already presented in Sec. 3.2, we also experiment here on SSCBench-nuScenes [2]. This dataset features a sparser lidar sensor, which has 32 beams, against 64 in SemanticKITTI. A single scan is thus considerably sparser vertically.

We adopt the 12-class scheme of SSCBench (truck and other-vehicle kept separate, bus folded into other-vehicle), matching Table II of the SSCBench paper [2]. This taxonomy applies on both sides of the network: it is both the supervised output space and the space the input semantic prior lives in. WafleIron’s 16-class pseudo-labels are therefore remapped into these $C = 1 2$ classes before voxelization, keeping input prior semantics and supervision targets on a single class set. In contrast, SemanticKITTI uses $C = 1 9$ classes.

SSCBench-nuScenes uses 500 training sequences and 199 validation sequences<sup>3</sup> on the same voxel grid as SemanticKITTI, i.e., 256 × 256 × 32 at 20 cm resolution.

Our retrained LMSCNet-SS on SSCBenchnuScenes is substantially stronger than the LMSC-Net baseline reported by the SSCBench paper [2] (37.6 vs. 21.1 IoU, see Tab. 6). Since the authors’ released checkpoint follows a diferent class scheme than their reported numbers<sup>4</sup>, we retrain the baselines under the paper’s 12-class scheme, isolating the efect of the priors.

Priors. Unless stated otherwise, we use Wafle-Iron with test-time augmentations (WI-TTA) [16], the most accurate segmenter in Tab. 1, and plain WafleIron (WI) without TTA. We keep this single segmenter family so that Tab. 3 compares one semantic prior with and without TTA; Tab. 1 already shows that the semantic gain holds for all segmenters, and MinkUNet is used again in Tab. 7.

As for the geometry, we use the best visibility prior reported in Sec. 3.4, i.e., the dense uniform sampling of free-space markers along lidar lines of sight, with spacing δ = voxel size = 20 cm.

We also report results using oracle semantics and/or oracle visibility.

Refinement. We did not use the generic SSC refinement proposed in SemCity [26] as the process leaks ground-truth occupancy<sup>5</sup>.

## 4.3 Quantitative results

## 4.3.1 Results on SemanticKITTI

Tab. 3 reports results of prior augmentations on SemanticKITTI. Endowing the models with both priors yields consistent gains across all four evaluated architectures.

## Semantics

The semantic gain is uneven, concentrated where semantics is the bottleneck: the lightweight models that predict semantics worst on their own gain the most (+9.5 mIoU pts for SemCity-AE, +7.6 for LMSCNet-SS), while the architectures with a strong internal semantic pathway gain less (+5.6 mIoU pts for SSA-SC, +0.9 for JS3C-Net).

Table 3 SSC prior combination on SemanticKITTI (validation set). We report geometric completion (IoU) and semantics (mIoU). ∆ measures the diference with the model that does not use the corresponding prior. “-”: no prior provided. <sup>†</sup>baseline is retrained.
<table><tr><td rowspan=1 colspan=2>1-frame inputvisibility  semanticsprior     prior</td><td rowspan=1 colspan=2>SSC outputgeometry      semanticsIoU %   ∆   mIoU % ∆</td></tr><tr><td rowspan=1 colspan=2>SemCity-AE [26]†</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>53.9</td><td rowspan=1 colspan=1>16.4     1</td></tr><tr><td rowspan=1 colspan=1>dense</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>56.9  +3.0</td><td rowspan=1 colspan=1>17.5   +1.1</td></tr><tr><td rowspan=2 colspan=1>一dense</td><td rowspan=2 colspan=1>WIWI</td><td rowspan=1 colspan=1>54.4  +0.5</td><td rowspan=1 colspan=1>25.5  +9.1</td></tr><tr><td rowspan=1 colspan=1>57.0  +3.1</td><td rowspan=1 colspan=1>25.5   +9.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>54.4  +0.5</td><td rowspan=1 colspan=1>26.0  +9.6</td></tr><tr><td rowspan=1 colspan=1>dense</td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>57.0  +3.1</td><td rowspan=1 colspan=1>25.9   +9.5</td></tr><tr><td rowspan=2 colspan=1>denseoracle</td><td rowspan=2 colspan=1>oracleoracle</td><td rowspan=1 colspan=1>57.1  +3.2</td><td rowspan=2 colspan=1>34.4  +18.037.3  +20.9</td></tr><tr><td rowspan=1 colspan=1>61.2  +7.3</td></tr><tr><td rowspan=1 colspan=1>LMSCNet</td><td rowspan=1 colspan=1>-SS [11]†</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>dense</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>16.5</td></tr><tr><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>57.5  +1.9</td><td rowspan=1 colspan=1>18.1  +1.6</td></tr><tr><td rowspan=2 colspan=1>dense</td><td rowspan=1 colspan=1>WI</td><td rowspan=1 colspan=1>55.6  +0.0</td><td rowspan=1 colspan=1>21.6  +5.1</td></tr><tr><td rowspan=1 colspan=1>WI</td><td rowspan=1 colspan=1>57.9  +2.3</td><td rowspan=1 colspan=1>23.8  +7.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>55.6  +0.0</td><td rowspan=1 colspan=1>21.9   +5.4</td></tr><tr><td rowspan=1 colspan=1>dense</td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>57.9  +2.3</td><td rowspan=1 colspan=1>24.1   +7.6</td></tr><tr><td rowspan=2 colspan=1>denseoracle</td><td rowspan=2 colspan=1>oracleoracle</td><td rowspan=1 colspan=1>58.2  +2.6</td><td rowspan=1 colspan=1>30.3  +13.8</td></tr><tr><td rowspan=1 colspan=1>63.2  +7.6</td><td rowspan=1 colspan=1>34.8  +18.3</td></tr><tr><td rowspan=1 colspan=1>JS3C-Net</td><td rowspan=1 colspan=1>[12]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>dense</td><td rowspan=2 colspan=1>WI</td><td rowspan=1 colspan=1>57.0     1</td><td rowspan=1 colspan=1>24.0     一</td></tr><tr><td rowspan=1 colspan=1>57.5  +0.5</td><td rowspan=1 colspan=1>24.2   +0.2</td></tr><tr><td rowspan=1 colspan=1>dense</td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>57.4   +0.4</td><td rowspan=1 colspan=1>24.9   +0.9</td></tr><tr><td rowspan=2 colspan=1>denseoracle</td><td rowspan=2 colspan=1>oracleoracle</td><td rowspan=1 colspan=1>57.7   +0.7</td><td rowspan=1 colspan=1>33.0   +9.0</td></tr><tr><td rowspan=1 colspan=1>64.0  +7.0</td><td rowspan=1 colspan=1>37.8  +13.8</td></tr><tr><td rowspan=1 colspan=1>SSA-SC [1</td><td rowspan=1 colspan=1>4]†</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>dense</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>23.3</td></tr><tr><td rowspan=1 colspan=1>WI</td><td rowspan=1 colspan=1>58.9  +0.9</td><td rowspan=1 colspan=1>28.6   +5.3</td></tr><tr><td rowspan=1 colspan=1>dense</td><td rowspan=1 colspan=1>WI-TTA</td><td rowspan=1 colspan=1>58.9  +0.9</td><td rowspan=1 colspan=1>28.9   +5.6</td></tr><tr><td rowspan=2 colspan=1>denseoracle</td><td rowspan=1 colspan=1>oracle</td><td rowspan=1 colspan=1>60.0  +2.0</td><td rowspan=1 colspan=1>39.9  +16.6</td></tr><tr><td rowspan=1 colspan=1>oracle</td><td rowspan=1 colspan=1>65.9  +7.9</td><td rowspan=1 colspan=1>48.3  +25.0</td></tr></table>

JS3C-Net is the clearest case. Like SSA-SC, it trains a point cloud segmentation branch jointly with completion, but our semantic pseudo-labels reach its completion head only through that branch, whereas SSA-SC receives them directly at the completion input. The branch is trained to predict the same per-point semantics as the pseudo-labels, so pseudo-labels of similar quality add little (+0.9 mIoU pts for both priors combined). The routing itself is not the limit: ground-truth labels, injected in the same way, still add +8.1 mIoU pts (dense | oracle vs. dense | WI-TTA in Tab. 3).

This is consistent with our observation above: the priors help most where the backbone is weakest, and a model that has already paid for a jointly trained segmenter has little headroom left for an input prior to recover, at the price of being locked to that segmenter. The per-class breakdown (Tab. C4) confirms that the gains concentrate on the classes that the baselines predict worst unaided. For instance, truck jumps from 3.7% IoU to 44.4% with LMSCNet-SS, and bicycle goes from 0.3% IoU to 12.7% with SemCity-AE.

## Geometric completion

This remark applies to a lesser extent to geometric completion, where the gains are smaller but substantial for methods that are weaker on completion (+3.1% IoU for SemCity-AE and +2.3% for LMSCNet-SS) while they are only incremental for methods that are better at reconstructing geometry (+0.9% IoU for SSA-SC and +0.4% for JS3C-Net).

## Prior complementarity

It can also be observed that the priors are mostly complementary. Indeed, adding semantic cues to a model equipped with visibility information barely improves geometry (at most +0.4 IoU pt with LMSCNet-SS). Conversely, providing a visibility prior on top of a model already augmented with semantic cues has little efect on semantics (+2.2 mIoU pts with LMSCNet-SS, but practically no efect with SemCity-AE).

## Oracle headroom

The oracle rows of Tab. 3 with dense visibility prior and GT semantic labels confirm the substantial unrealized semantic headroom in the evaluated networks, with gains from +6.2 mIoU pts (LMSCNet-SS) to +11.0 mIoU pts (SSA-SC). The substantial improvement on completion when using not only the GT semantics but also the visibility oracle, from +4.1 IoU pts (SemCity-AE) to +6.3 IoU pts (JS3C-Net), further boosts the semantics, from +2.9 mIoU pts (SemCity-AE) to +8.4 mIoU pts. It shows that resolving free space continues to help the semantic metric even when the semantic prior is already perfect. This mirrors the Sec. 3.4 finding that visibility lifts mIoU as well as IoU, and confirms the two priors remain complementary at their joint upper bound rather than saturating against each other.

Table 4 SSC comparison with baselines on SemanticKITTI (validation set, 19 classes). We use the best priors, i.e., WI-TTA semantic pseudo-labels and dense uniform sampling of free-space markers. <sup>†</sup>retrained; other baselines taken from the respective papers. <sup>a</sup>as reported in the paper. <sup>b</sup>public code has no semantic part; values from the paper use the panoramic (360<sup>◦</sup>) protocol on the validation set. <sup>c</sup>not reproducible, cf. [56, Sec. 5.2.1] and [57, Supp., §6]. <sup>d</sup>multi-frame test-time optimization.
<table><tr><td rowspan=1 colspan=1>codde</td><td rowspan=1 colspan=4>Model</td><td rowspan=1 colspan=1>geom.IoU %</td><td rowspan=1 colspan=1>sem.mIoU %</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=1></td><td rowspan=5 colspan=4>SemCity-AE [26]†LMSCNet-SS [1i]†LODE [25]SSA-SC [14]†JS3C-Net [12]</td><td rowspan=1 colspan=1>53.9</td><td rowspan=1 colspan=1>16.4</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>16.5</td></tr><tr><td rowspan=1 colspan=2>E [25]</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>51.2</td><td rowspan=3 colspan=1>20.223.324.0</td></tr><tr><td rowspan=1 colspan=1>58.0</td></tr><tr><td rowspan=1 colspan=1>57.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>LMSCNet-SS [11] +priors</td><td rowspan=1 colspan=1>57.9</td><td rowspan=1 colspan=1>24.1</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=2 colspan=1>√</td><td rowspan=2 colspan=4>SSA-SC [14]SSC-RS [13]</td><td rowspan=1 colspan=1>58.3</td><td rowspan=2 colspan=1>24.524.8</td></tr><tr><td rowspan=1 colspan=1>58.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>JS3C-Net [12]    +priorsSemCity-AE [26] +priors</td><td rowspan=1 colspan=1>57.457.0</td><td rowspan=1 colspan=1>24.925.9</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>DPS2CNet [56]</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>26.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SSA-SC [14]      +priors</td><td rowspan=1 colspan=1>58.9</td><td rowspan=1 colspan=1>28.9</td></tr><tr><td rowspan=1 colspan=5>Other methods from the literature(x)  DiffSSCX  S3CNet [21](x)  SCPNet [38]cTALoS [15]à</td><td rowspan=1 colspan=1>60.357.249.956.1</td><td rowspan=1 colspan=2>26.733.137.239.3</td></tr></table>

## Comparison to reproducible SSC baselines

With SSA-SC, the combined prior recipe reaches 28.9% mIoU, the best semantic score among fully-reproducible single-frame methods, ahead of the strongest prior method in this setting, DPS2CNet [56] (26.7%). Other scores reported in the literature come from methods that are not directly comparable: DifSSC [8] (26.7% mIoU) reports a panoramic (360<sup>◦</sup>) validation protocol and its public code has no semantic component, S3CNet [21] (33.1% mIoU) is closed-source, SCP-Net [38] (37.2% mIoU) is not reproducible from its release (cf. [56, Sec. 5.2.1] and [57, Supp., §6]), and TALoS [15] (39.3% mIoU) relies on multi-frame test-time optimization. Within this reproducible single-frame setting, our input-only recipe leads on semantics while remaining plug-and-play.

As for geometric completion, DPS2CNet performs best (60.8% IoU), with our prior-augmented version of SSA-SC ranking second (58.9%). The oracle headroom however reveals that the older methods that we evaluated with prior augmentation have the potential to outperform it (Tab. 3), provided more accurate priors are fed to the networks.

## Inference time

SSC models have a wide range of run times, from a bit more than 30 ms/frame on a Nvidia A100 GPU (SSA-SC) to almost 700 ms/frame (JS3C-Net). The use of prior augmentations on top of these SSC models incurs an extra latency (details in App. B).

It turns out that most of the run time of SSC when adding semantic and visibility priors to a model is spent in the semantic segmenter to infer the pseudo-labels of the semantic cues (one to several hundreds of ms/frame), unless a fast but poor-performing segmenter is used. The efect of the increased number of channels at the first layer of the SSC network mostly remains negligible (< 5 ms/frame). Last, the sampling of free-space markers (∼ 6 ms/frame) and the voxelization that follows (∼ 9 ms/frame) also are negligible, unless using very fast SSC models such as LMSCNet-SS and SSA-SC, that run in less than 50 ms/frame.

In practice, the visibility prior (for any variant) and the semantic prior can be precomputed once for a given dataset and cached.

## 4.3.2 Results on SSCBench-nuScenes

We evaluated the combined prior recipe on SSCBench-nuScenes with the two lightweight methods LMSCNet-SS and SemCity-AE (Tab. 5).

Performance ofprior augmentation. The semantic prior alone carries over cleanly: WafleIron pseudo-labels, without visibility cues, raise mIoU by +11.1 pts on LMSCNet-SS and +11.7 pts on SemCity-AE; it is a similar efect as seen on SemanticKITTI, and it is preserved once visibility is added, confirming it as a sensor-agnostic driver of the mIoU gain.

The dense sampling of free-space markers also helps here: the visibility prior alone adds +0.7 IoU pt on LMSCNet-SS and +1.2 IoU pt on SemCity-AE, consistently positive but slightly smaller than the +1.9 to 3.0 IoU pts that it yields on the denser SemanticKITTI sensor (Tab. 3). This is the expected consequence of the sparser lidar sensor, as a single scan certifies less free space per frame. The free-space markers are also noisier: the visibility oracle, which uses the same markers but removes those that fall on occupied voxels, gains +11.6 to +13.7 IoU pts here, and the dense prior obtains less than 10% of this gain, against 24 to 45% on SemanticKITTI.

Table 5 Prior combination on SSCBenchnuScenes (validation set, 12-class scheme). We report geometric completion (IoU) and semantics (mIoU). ∆ measures the diference with the model that does not use the corresponding prior. <sup>†</sup>baseline is retrained.
<table><tr><td colspan="2">1-frame input</td><td colspan="4">SSC output</td></tr><tr><td>visibility</td><td>semantics</td><td colspan="2">geometry</td><td colspan="2">semantics</td></tr><tr><td>prior</td><td>prior</td><td>IoU %</td><td>∆</td><td>mIoU %</td><td>∆</td></tr><tr><td colspan="6">SemCity-AE [26]†</td></tr><tr><td>dense</td><td>-</td><td>37.6 38.8</td><td></td><td>15.6</td><td></td></tr><tr><td></td><td>WI</td><td></td><td>+1.2</td><td>16.1</td><td>+0.5</td></tr><tr><td>dense</td><td>WI</td><td>38.6 39.6</td><td>+1.0 +2.0</td><td>27.3 27.4</td><td>+11.7 +11.8</td></tr><tr><td></td><td>WI-TTA</td><td>38.6</td><td></td><td>27.8</td><td></td></tr><tr><td>dense</td><td>WI-TTA</td><td>39.6</td><td>+1.0 +2.0</td><td>27.8</td><td>+12.2 +12.2</td></tr><tr><td>oracle</td><td></td><td>49.2</td><td>+11.6</td><td>19.1</td><td>+3.5</td></tr><tr><td>oracle oracle</td><td>WI WI-TTA</td><td>50.0 50.0</td><td>+12.4 +12.4</td><td>33.4 33.7</td><td>+17.8 +18.1</td></tr><tr><td colspan="6">LMSCNet-SS [11]†</td></tr><tr><td>dense</td><td>- -</td><td>37.6 38.3</td><td>+0.7</td><td>15.0 16.6</td><td>+1.6</td></tr><tr><td>dense</td><td>WI</td><td>38.6</td><td>+1.0</td><td>26.1</td><td>+11.1</td></tr><tr><td></td><td>WI WI-TTA</td><td>39.2 38.6</td><td>+1.6</td><td>26.0 26.2</td><td>+11.0</td></tr><tr><td>dense</td><td>WI-TTA</td><td>39.1</td><td>+1.0 +1.5</td><td>26.1</td><td>+11.2 +11.1</td></tr><tr><td>oracle</td><td></td><td>51.3</td><td>+13.7</td><td>21.7</td><td>+6.7</td></tr><tr><td>oracle oracle</td><td>WI WI-TTA</td><td>51.6 51.6</td><td>+14.0 +14.0</td><td>31.7 31.8</td><td>+16.7 +16.8</td></tr></table>

The combination of the semantic and visibility priors provides a little more gain than the best of both individual priors: +0.8 to +0.9 IoU pt for geometric completion when adding semantics, but no extra gain on semantics when adding the visibility prior (diference within 0.1 mIoU pt). As already noted in Sec. 3.4, it shows and confirms that the two priors are mostly complementary, with a small ripple efect.

Table 6 SSC comparison with baselines on SSCBench-nuScenes (validation set, 12-class scheme). We use the best priors: WI-TTA semantic pseudo-labels and dense uniform sampling of free-space markers. <sup>†</sup>retrained; <sup>‡</sup>scores reported in [2].
<table><tr><td rowspan=1 colspan=1>code</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>geom.IoU %</td><td rowspan=1 colspan=1>sem.mIoU %</td></tr><tr><td rowspan=4 colspan=1></td><td rowspan=4 colspan=1>LMSCNet [11]‡SSCNet [19]‡LMSCNet-SS [11]†SemCity-AE [26]†</td><td rowspan=1 colspan=1>21.1</td><td rowspan=4 colspan=1>8.411.815.015.6</td></tr><tr><td rowspan=1 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1>37.6</td></tr><tr><td rowspan=1 colspan=1>37.6</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>LMSCNet-SS*[11+priorsSemCity-AE*[26]+priors</td><td rowspan=1 colspan=1>39.1</td><td rowspan=1 colspan=1>26.1</td></tr><tr><td rowspan=1 colspan=1>39.6</td><td rowspan=1 colspan=1>27.8</td></tr></table>

## SSC baseline comparison

There are few baselines on SSCBench-nuScenes. When equipped with both the semantic and visibility priors, the two methods we evaluated, which were already ranking first and second, get a large boost in semantics (+11.1 to 12.2 mIoU pts) and a modest gain in geometry (+1.5 to 2.0 IoU pts) and keep their ranking.

## 4.4 Qualitative results

Fig. 4 compares the SSC outputs of the two main networks we studied (SemCity-AE and LMSCNet-SS) against their prior-free baselines on both datasets: SemanticKITTI (Fig. 4A) and the sparser SSCBench-nuScenes (Fig. 4B), on two validation scenes each.

Operating only on the raw sparse input point cloud (Input), the baselines (Baseline) generate wrong semantics and noisy, fragmented class boundaries for SemanticKITTI. In contrast, when endowed with our input priors (+ priors), both models produce cleaner and more structured outputs, closer to the ground truth (GT): continuous surfaces such as roads (purple) become fully connected, vegetation aligns better with the GT, and thin structures such as poles and trafic signs are more reliably restored.

The same improvement holds on the 32-beam SSCBench-nuScenes scans (Fig. 4B), confirming that the qualitative benefit transfers across sensors. This is also confirmed quantitatively by the per-class breakdown in App. C (Tab. C4 and Tab. C5).

Table 7 Segmenter-agnostic training. We study the efect of training with GT semantic labels rather than with the (pseudo-)labels of a segmenter. At inference, only the segmenter labels are used. We measure on SemanticKITTI the performance for the full shipped recipe, when the semantic prior is combined with the dense visibility prior (not with the visibility oracle). ∆ measures the diference of performance between the model that trains on the GT and the model that trains on the segmenter labels. <sup>†</sup>retrained.
<table><tr><td colspan="2">1-frame input</td><td colspan="3">SSC output</td></tr><tr><td>sem. prior @ training</td><td>sem. prior</td><td>geometry IoU % ∆</td><td></td><td>semantics</td></tr><tr><td></td><td>@inference</td><td></td><td>mIoU %</td><td>∆</td></tr><tr><td colspan="5">LMSCNet-SS [11]† 一</td></tr><tr><td>MinkUNet GT</td><td>MinkUNet MinkUNet</td><td>57.2 57.4 +0.2</td><td>24.0 23.8</td><td>1 -0.2</td></tr><tr><td>WI</td><td>WI</td><td>57.9</td><td>23.8</td><td></td></tr><tr><td>GT</td><td>WI</td><td>57.4 -0.5</td><td>24.0</td><td>+0.2</td></tr><tr><td>WI</td><td>WI-TTA</td><td>57.9</td><td>24.1</td><td></td></tr><tr><td>GT</td><td>WI-TTA</td><td>57.5 -0.4</td><td>24.6</td><td>+0.5</td></tr><tr><td>GT</td><td>GT</td><td>58.2</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>30.3</td><td>一</td></tr><tr><td colspan="5">SemCity-AE [26]†</td></tr><tr><td>MinkUNet</td><td>MinkUNet</td><td>56.8 一</td><td>25.5</td><td></td></tr><tr><td>GT</td><td>MinkUNet</td><td>56.6 -0.2</td><td>26.1</td><td>+0.6</td></tr><tr><td>WI</td><td>WI</td><td>57.0</td><td>25.5</td><td></td></tr><tr><td>GT</td><td>WI</td><td>56.7 -0.3</td><td>26.1</td><td>+0.6</td></tr><tr><td>WI</td><td>WI-TTA</td><td>57.0</td><td>25.9</td><td></td></tr><tr><td>GT</td><td>WI-TTA</td><td>56.8</td><td>-0.2 26.9</td><td>+1.0</td></tr><tr><td>GT</td><td>GT</td><td>57.1</td><td>34.4</td><td></td></tr></table>

## 4.5 Segmenter-agnostic training

A practical concern with an input semantic prior is model coupling: if the SSC network is trained on the pseudo-labels of one segmenter, must it be retrained whenever that segmenter is replaced or upgraded? We show it need not be.

Tab. 7 quantifies it for the two lightweight architectures we use in our study, based on the full prior recipe (both semantic cues and dense visibility markers). For each of-the-shelf segmenter, we compare the matched setting (the completion network trained on that segmenter’s pseudo-labels) against a single network trained once on GT semantics and switched to the segmenter labels only at inference. The reported ∆ is the cost of not retraining; a small value indicates that a single GT-trained network can be used as an alternative to any network trained with a specific segmenter.

As can be seen from the table, the cost (or benefit) of not retraining for a particular segmenter is negligible: between −0.5 and +0.2 IoU (A) SemanticKITTI (64-beam)

(B) SSCBench-nuScenes (32-beam)

![](images/892910c6f3a5aaf10e0fb151ba0797d6e44eea117629ba3eb31e48035b40a0ad.jpg)  
Fig. 4 Qualitative SSC results across datasets and backbones. Each column is one validation scene. For each backbone, rows show the prior-free baseline and the same backbone equipped with our input priors (+ our priors), flanked by the input point cloud (input) and the ground truth (GT). (A) SemanticKITTI (64-beam) and (B) the sparser SSCBenchnuScenes (32-beam), two scenes each (columns), for the two main backbones SemCity-AE and LMSCNet-SS (row pairs). Best viewed zoomed in.

pt for occupancy, and between −0.2 and +1.0 mIoU pt for semantics. Prior sequential pipelines (Sec. 2.2) process semantic predictions from separate networks, but do not establish how robust the completion head is to which segmenter supplies them. Specifically, the ability to train the completion head just once with ground-truth semantics and subsequently swap in arbitrary of-the-shelf segmenters at inference without retraining is a distinct advantage we systematically exploit here, contrasting with the rigid coupling of multi-task methods like SSA-SC [14] and JS3C-Net [12].

We hypothesize that part of the explanation of this result is related to overfitting. Lidar segmenters tend indeed to overfit their training split. The pseudo-labels they produce on the SSC training sequences thus sit close to the ground-truth annotations. A completion network trained on GT input semantics therefore sees a trainingtime input distribution that closely matches what any reasonable of-the-shelf segmenter produces at inference. The completion network can thus be trained once, on GT semantics, and paired at inference with an arbitrary segmenter (including a future, stronger one) at no retraining cost.

This decoupling is specific to the semantic prior: performance drops when training on the visibility oracle and testing on the dense sampling of free-space markers (App. D).

## 5 Conclusion

We revisited lidar semantic scene completion through two input priors: of-the-shelf semantic pseudo-labels and ray-cast visibility information. These priors are supplied purely at input level and kept decoupled from the rest of the SSC network.

We summarize below the main takeaways of our study, based on experiments with four SSC methods, four semantic segmenters, and two datasets relying on diferent lidar sensors.

• Our prior augmentation is efective. It systematically improves both completion and semantics, lifting older lightweight models to the level of recent state-of-the-art systems, among fully-reproducible single-frame SSC methods.

• Our prior augmentation is simple. It basically applies to any SSC method at no architectural cost beyond widening the input layer, and basically any of-the-shelf semantic and visibility cues can be used as priors. Moreover, our approach is segmenteragnostic not only because any segmenter yields gains, but also in that the prioraugmented model can be trained just once, with ground-truth point labels, and later used with any segmenter.

• The quality of prior-augmented SSC grows with input prior accuracy. Outsourcing semantic labels thus ofers better SSC for free whenever segmenters improve. Besides, the visibility prior seems to benefit from lidar density, while the semantic prior does not directly depend on it.

• Our prior augmentations benefit more the lightweight models. Stronger backbones gain less, but still substantially, unless they already compute comparable semantics internally, in which case the prior adds little.

• Semantics is the dominant bottleneck. It is indeed the area where there is the most room for improvement, with gains up to 12 mIoU pts, while completion does not improve more than 3 IoU pts. Moreover, the oracle input semantics establishes a performance upper bound far above the prior-augmented baselines, which opens perspectives for even further improvements.

• Semantic cues have a small efect on geometric completion in the networks we studied, despite architectural connections in their design: even ground-truth input semantics change the IoU by at most 1.1 pts. Conversely, visibility cues do raise the mIoU, by up to 8.4 pts with the visibility oracle, although this partly reflects that the mIoU also accounts for geometric errors. This nuances one of the common motivations to address both tasks jointly in SSC, and highlights a topic to consider when designing a new SSC architecture.

We release our priors and integration code to make these input-level gains easy to adopt across SSC backbones. Our code is available at https://github.com/astra-vision/SSC-Priors.

Besides exploring the oracle headrooms and tightening geometry and semantics coupling, other perspectives include using multiple frames (e.g., from the past) and learning the visibility prior to reduce the free-space marking noise.

## Acknowledgments

The authors thank Anh-Quan Cao for fruitful discussions. This work was granted access to the HPC resources of IDRIS under the allocation AD011014484R3 made by GENCI. Gilles Puy contributed to this work while employed by Valeo; he accessed and processed all datasets on valeo.ai compute infrastructures; he is presently employed by Meta.

## Appendix A Conference paper extension

This article is an extended version of our conference paper “Exploring Easy Boosts for Lidar Semantic Scene Completion” [18]. The conference version studies the two input-level priors (semantic pseudo-labels and ray-cast visibility), with experiments only on SemanticKITTI [1].

This journal version has been largely rewritten and adds the following:

1. a systematic study of the design space of the visibility prior (Sec. 3.4);

2. experiments on another dataset featuring a diferent lidar sensor, namely SSCBenchnuScenes, which confirm the results obtained on SemanticKITTI (Sec. 4.3.2);

3. an architectural account of the uneven gains across diferent SSC methods (Sec. 4.3);

4. a deeper analysis of the combination of the semantic and visibility priors (Sec. 4.3);

5. an improved treatment and discussion on the prior oracle headroom for both semantics and visibility (Sec. 4.3, Sec. 4.5);

6. a discussion of the lesser impact of the visibility prior on SSCBench-nuScenes, which we find attributable to fewer and noisier visibility cues (Sec. 4.3.2);

7. a segmenter-agnostic training analysis showing that the completion network need not be retrained when the semantic segmenter is changed (Sec. 4.5);

8. experiments showing that, on the contrary, training with GT visibility degrades the performance when testing with actual, noisy visibility cues (App. D);

9. an inference time study (App. B);

10. a substantially expanded related-work discussion (Sec. 2).

## Appendix B Inference time profiling

## Methodology.

We profile each variant on 50 consecutive lidar frames of SemanticKITTI sequence 08. Each method is run three times per frame (150 measurements per method), preceded by a single warm-up sweep whose timings are discarded. The order in which methods are invoked is randomized per frame under a fixed seed, so position-dependent efects do not bias any one method. Each method is split into two independently timed stages: ray sampling, which generates the free-space markers along each sensor ray; and voxelization, which clips markers and returns to the 256×256×32 grid at 0.2 m, computes per-voxel flat indices, accumulates per-class vote counts, resolves each voxel by majority vote under the returns-override-freespace rule, and applies any method-specific refinement (marking as free voxels found as empty). End-to-end latency is timed separately by wrapping the full call; averaged over all 150 measurements it matches the sum of the per-stage means to within 1.8% (≤ 0.07 ms) for every method, confirming that the two stages account for essentially all the runtime. We report the independently measured end-to-end value; the per-stage means in Tab. B3 indicate each stage’s contribution. All measurements run on an NVIDIA A100 GPU (PCIe, 40 GB) under torch.inference mode(). Frames are loaded and moved to the GPU once before timing, so disk I/O is excluded, and CUDA is synchronized before and after each timed region.

Table B1 Inference time (ms/frame) of a forward pass of SSC models on the SemanticKITTI validation set, excluding I/Os and voxelization, measured on an NVIDIA A100 GPU (PCIe, 40 GB) after warm-up, averaged over 50 frames × 3 runs.
<table><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>| vanilla |</td><td rowspan=1 colspan=1>∆ | p</td><td rowspan=1 colspan=1>rior aug.</td></tr><tr><td rowspan=7 colspan=2>SSA-SC [14]LMSCNet-SS [11]SemCity-AE [26]JS3C-Net [12]</td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1> $3 3 \pm \ : 4$ </td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>+4</td><td rowspan=2 colspan=1> $3 7 \pm \ : 4$ </td></tr><tr><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1> $4 2 \pm \ : 1$ </td><td rowspan=1 colspan=1>+5</td><td rowspan=1 colspan=1> $4 7 \pm \ : 1$ </td></tr><tr><td rowspan=1 colspan=1> $1 6 2 \pm \ : 1$ </td><td rowspan=1 colspan=1>+1</td><td rowspan=1 colspan=1> $1 6 3 \pm \ : 1$ </td></tr><tr><td rowspan=1 colspan=1> $6 6 5 \pm 8 8$ </td><td rowspan=1 colspan=1>+2</td><td rowspan=1 colspan=1> $6 6 7 \pm 8 8$ </td></tr></table>

Table B2 Inference time (ms/frame) of semantic segmenters on the SemanticKITTI validation set, excluding I/Os, measured on an NVIDIA A100 GPU (PCIe, 40 GB) after warm-up and averaged over 50 frames × 3 runs. ±: standard deviation.
<table><tr><td>Semantic segmenter</td><td>full infer.</td></tr><tr><td>SalsaNext [55] WaffleIron [16]</td><td> $9 \pm \ : 1$ </td></tr><tr><td></td><td> $1 5 5 \pm \ : 6$ </td></tr><tr><td>MinkUNet [17]</td><td> $1 6 4 \pm \ : 4$ </td></tr><tr><td>WaffleIron + TTA [16]</td><td> $1 8 7 1 \pm 5 7$ </td></tr></table>

Table B3 Inference time (ms/frame) of visibility priors on the SemanticKITTI validation set, excluding I/Os, measured on an NVIDIA A100 GPU (PCIe, 40 GB) after warm-up and averaged over 50 frames × 3 runs: ray sampling (sampl.), voxelization (voxel.), end-to-end (total).
<table><tr><td></td><td>| visibility prior</td><td>|sampl. voxel. total</td><td></td><td></td></tr><tr><td>(1) (2)</td><td>none 1 pt before the hit</td><td>0 0.4</td><td>3.2 3.0</td><td>3.2 3.4</td></tr><tr><td>(3) (4)</td><td>1 pt at random 10 pts at random</td><td>0.4 0.4</td><td>3.0 3.4</td><td>3.4 3.9</td></tr><tr><td>(5)</td><td>25 pts at random</td><td>0.5</td><td>4.7</td><td></td></tr><tr><td>(6)</td><td>50 pts at random</td><td>0.7</td><td>6.9</td><td>5.2 7.6</td></tr><tr><td>(7)</td><td>100 pts at random</td><td>1.2</td><td>11.2</td><td>12.4</td></tr><tr><td>(8)</td><td>dense (δ = voxel size)</td><td>5.8</td><td>8.9</td><td>14.7</td></tr><tr><td>(9)</td><td>dense with dilation</td><td>5.8</td><td>9.4</td><td>15.2</td></tr></table>

## Results.

Tab. B3 decomposes each variant’s end-to-end latency into its ray-sampling and voxelization stages. Voxelization dominates every variant; ray sampling takes at most 1.2 ms for all sparse variants and becomes a substantial share only for the dense variants. Voxelization alone, with no markers to place, costs 3.2 ms/frame. Single-marker variants add 0.4 ms of sampling, while voxelization stays at 3.0 ms (within the run-to-run variation of the marker-free pass), reaching 3.4 ms. Across the n-random family both stages grow with n, sampling from 0.4 ms at n=1 to 1.2 ms at n=100 and voxelization from 3.0 to 11.2 ms. The dense prior places fewer markers than 100-random, as its number of steps is bounded by the range of each return, so it voxelizes faster (8.9 vs. 11.2 ms), although its markers, spaced one voxel apart, mark more distinct voxels. Its sampling is slower (5.8 ms), because it builds the candidate steps of the longest ray for every ray before it discards the steps beyond each return, which brings it to 14.7 ms end-to-end, 4.6× the marker-free pass. Adding the dilation step costs a further 0.5 ms of voxelization, at 15.2 ms.

Every variant is therefore cheap relative to the rest of the pipeline: the completion networks take from 33 ms/frame (SSA-SC) to 665 ms/frame (JS3C-Net) (Tab. B1), and the segmenter that supplies the semantic prior takes 155 ms/frame for WafleIron and 1871 ms/frame with TTA (Tab. B2). With both priors, the dense visibility prior accounts for about 7% of the runtime at most (SSA-SC with WafleIron), and for less than 1% with WafleIron-TTA. The dense prior is the default: it gives the best or near-best accuracy on both networks (Tab. 2), it has no sample-count hyperparameter, and it costs only 2.3 ms/frame more than 100-random.

## Appendix C Per-class results

Tab. C4 and Tab. C5 report per-class IoU on SemanticKITTI and SSCBench-nuScenes, respectively, for the prior combinations of Tab. 3 and Tab. 5; the SemanticKITTI table adds the oracle visibility rows without semantics and with WaffleIron. On both datasets, the semantic prior’s gains concentrate on the classes the lightweight baselines predict worst unaided (e.g., truck on SemanticKITTI rises from 3.7 to 30.9 IoU for LMSCNet-SS and from 6.9 to 46.0 for SemCity-AE), consistent with the dataset-level results of Tab. 3 and Tab. 5.

## Appendix D Visibility oracle training

Table D6 repeats the study of Sec. 4.5 on the visibility axis, with the semantic prior held at WaffleIron. Unlike the semantic prior, the visibility prior does not survive the swap. A network trained on the oracle prior and given the dense prior at inference loses 21.7 IoU pts on LMSCNet-SS and 11.9 on SemCity-AE, whereas the equivalent swap on the semantic axis costs at most 0.2 mIoU pts, and improves mIoU in most of the studied cases (Tab. 7). Both networks lose occupancy recall: 0.66 → 0.38 on LMSCNet-SS, and 0.76 → 0.53 on SemCity-AE. Occupancy precision does not degrade, and increases from 0.82 to 0.87 and from 0.70 to 0.75, respectively. Our interpretation is that a network trained on ground-truth free space does not predict occupancy in the wrong place, it just predicts too little occupancy. The oracle prior is therefore a ceiling, and not a training-time substitute for the deployed prior.

Table C4 Per-class IoU on SemanticKITTI (validation set, 19 classes). Rows, notation and colors are as in Table 3. The percentage below each class name is the share of that class in the occupied voxels. All metrics are reported in %.
<table><tr><td>Visibility Semantics</td><td></td><td>IoU mIoU</td><td>Ccar 4.2% 0.1% &lt;0.1%</td><td>biece</td><td>otccccce</td><td>oth. icle truuck</td><td>Perron</td><td>bicist</td><td>motcccist</td><td>oad</td><td>parkng</td><td>sidealk</td><td>rud h.</td><td>buiiding</td><td>fence</td><td>Vvettion</td><td>uunk</td><td>terain</td><td>trasgn</td></tr><tr><td>LMSCNet-SS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.9%</td><td>0.1% 13.4%</td><td>1.1%</td><td>42.1%</td><td>1.0%</td><td>16.1%</td><td>0.3%</td><td></td></tr><tr><td></td><td></td><td>55.6 16.5</td><td>38.2</td><td>0.0</td><td>0.0</td><td>3.7 0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>64.8</td><td>10.3</td><td>35.4</td><td>0.0</td><td>36.3 9.7</td><td>38.7</td><td>12.8</td><td>42.7</td><td>20.6</td><td>1.2</td></tr><tr><td>dense</td><td></td><td>57.5 18.1</td><td>41.3</td><td>0.0</td><td>0.0</td><td>7.5 0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>65.7</td><td>16.8</td><td>35.7</td><td>1.4 37.5</td><td>11.2</td><td>41.9</td><td>15.4</td><td>45.2</td><td>20.6</td><td>2.7</td></tr><tr><td></td><td>WI</td><td>55.6 21.6</td><td>41.8</td><td>0.0</td><td>0.0</td><td>28.3 2.8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>68.1</td><td>30.5</td><td>41.8</td><td>2.4</td><td>37.4 18.7</td><td>40.1</td><td>23.2</td><td>45.0</td><td>26.2</td><td>4.8</td></tr><tr><td>dense</td><td>WI</td><td>57.9 23.8</td><td>43.5</td><td>0.0</td><td>3.0</td><td>40.2</td><td>11.3 0.1</td><td>0.0</td><td>0.0</td><td>70.3</td><td>32.1</td><td>42.7</td><td>3.0</td><td>40.4 18.8</td><td>43.0</td><td>24.4</td><td>44.5</td><td>26.8</td><td>7.5</td></tr><tr><td></td><td>WI-TTA</td><td>55.6 21.9</td><td>41.8</td><td>0.0</td><td>0.0</td><td>30.9 3.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>68.1</td><td>30.4</td><td>42.2</td><td>2.0</td><td>37.6 19.1</td><td>40.3</td><td>23.4</td><td>45.5</td><td>26.5</td><td>5.0</td></tr><tr><td>dense</td><td>WI-TTA</td><td>57.9 24.1</td><td>43.5</td><td>0.0</td><td>3.0</td><td>44.4</td><td>12.0 0.1</td><td>0.0</td><td>0.0</td><td>70.2</td><td>32.0</td><td>43.2</td><td>2.3</td><td>40.4 19.2</td><td>43.2</td><td>24.5</td><td>44.9</td><td>27.0</td><td>7.5</td></tr><tr><td>oracle</td><td></td><td>63.6 19.6</td><td>50.9</td><td>0.0</td><td>0.0</td><td>4.2</td><td>0.0</td><td>0.7</td><td>0.0</td><td>69.2</td><td>18.4</td><td>36.4</td><td>0.7</td><td>41.2</td><td></td><td></td><td></td><td></td><td>1.6</td></tr><tr><td>oracle</td><td>WI</td><td>63.2 26.5</td><td>52.9</td><td>1.7</td><td>3.5</td><td>0.8 37.2 16.0</td><td>2.1</td><td>5.5</td><td>0.0</td><td>76.0</td><td>34.0</td><td>45.8</td><td>2.4</td><td>11.7 43.2 20.1</td><td>45.0 46.2</td><td>18.8 27.0</td><td>46.8 50.3</td><td>26.5 31.5</td><td>8.7</td></tr><tr><td>dense</td><td>oracle</td><td>58.2 30.3</td><td>44.9</td><td>2.1</td><td>4.5</td><td>40.7 29.2</td><td>4.0</td><td>0.0</td><td>0.0</td><td>73.6</td><td>53.1</td><td>49.6</td><td>23.3 44.2</td><td>33.6</td><td>45.8</td><td>33.1</td><td>56.4</td><td>30.8</td><td>7.3</td></tr><tr><td>oracle</td><td>oracle</td><td>63.2 34.8</td><td>54.0</td><td>7.3</td><td>5.6</td><td>47.9 37.8</td><td>9.2</td><td>5.3</td><td>0.0</td><td>79.8</td><td>57.9</td><td>53.4</td><td>26.2 47.2</td><td>34.6</td><td>48.7</td><td>36.4</td><td>61.7</td><td>35.6</td><td>12.9</td></tr><tr><td>SemCity-AE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>53.9 16.4</td><td>34.3</td><td>0.3</td><td>1.5</td><td>6.9 6.9</td><td>2.9</td><td>0.7</td><td>0.0</td><td>60.4</td><td>11.9</td><td>30.8 0.2</td><td>34.4</td><td>8.3</td><td>36.1</td><td>16.8</td><td>39.5</td><td>17.3</td><td>2.8</td></tr><tr><td>dense</td><td></td><td>56.9 17.5</td><td>36.6</td><td>1.0</td><td>3.4</td><td>8.6 6.4</td><td>2.1</td><td>1.0</td><td>0.0</td><td>60.9</td><td>14.3</td><td>30.9 0.4</td><td>36.2</td><td>8.8</td><td>39.7</td><td>18.9</td><td>39.8</td><td>19.5</td><td>3.0</td></tr><tr><td></td><td>WI</td><td>54.4 25.555</td><td>39.3</td><td>12.7</td><td>13.4</td><td>42.0 19.0</td><td>8.5</td><td>4.2</td><td>0.0</td><td>68.2</td><td>29.8</td><td>41.6 3.3</td><td>37.8</td><td>15.9</td><td>38.8</td><td>26.5</td><td>45.0</td><td>26.9</td><td>11.8</td></tr><tr><td>dense</td><td>WI</td><td>57.0</td><td>40.0</td><td>11.7</td><td>10.6</td><td>43.3 18.5</td><td>9.2</td><td>4.0</td><td>0.0</td><td>67.9</td><td>28.6</td><td>41.0 2.7</td><td>38.5</td><td>16.7</td><td>42.7</td><td>26.6</td><td>45.4</td><td>26.0</td><td>11.0</td></tr><tr><td></td><td>WI-TTA</td><td>54.4 26.0</td><td>39.4</td><td>13.6</td><td>14.3</td><td>46.0 20.6</td><td>8.7</td><td>4.3</td><td>0.0</td><td>68.2</td><td>30.4</td><td>42.1</td><td>3.2 37.9</td><td>16.1</td><td>38.9</td><td>26.7</td><td>45.2</td><td>27.3</td><td>12.1</td></tr><tr><td>dense</td><td>WI-TTA</td><td>57.0 25.9</td><td>40.1</td><td>12.7</td><td>11.2</td><td>46.0 20.0</td><td>9.3</td><td>4.0</td><td>0.0</td><td>67.9</td><td>29.2</td><td>41.4</td><td>38.6</td><td>16.9</td><td>42.8</td><td>26.8</td><td>45.5</td><td>26.3</td><td>11.2</td></tr><tr><td>oracle</td><td></td><td>60.6 20.0</td><td>43.6</td><td>1.4</td><td>3.9</td><td>11.7 7.6</td><td>7.3</td><td>12.9</td><td>0.0</td><td>64.7</td><td>15.7</td><td>31.7</td><td>0.3 38.1</td><td>9.1</td><td>41.7</td><td>20.7</td><td>40.7</td><td>24.7</td><td>3.6</td></tr><tr><td>oracle</td><td>WI</td><td>61.0 28.3</td><td>48.6</td><td>12.3</td><td>7.9</td><td>50.4 17.3</td><td>17.4</td><td>11.4</td><td>0.0</td><td>73.5</td><td>30.7</td><td>42.5</td><td>2.5 40.0</td><td>18.7</td><td>44.7</td><td>30.0</td><td>47.4</td><td>31.3</td><td>12.0</td></tr><tr><td>dense</td><td>oracle</td><td>57.1 34.4</td><td>43.2</td><td>21.6</td><td>20.8</td><td>58.5 38.0</td><td>10.3</td><td>4.5</td><td>0.4</td><td>71.1</td><td>52.7</td><td>46.8</td><td>20.6 41.6 43.5</td><td>33.3 35.9</td><td>45.2 47.5</td><td>35.9 39.4</td><td>55.1 58.8</td><td>33.4 39.1</td><td>21.1 24.8</td></tr><tr><td>oracle</td><td>oracle</td><td>61.2 37.3</td><td>51.2</td><td>23.8</td><td>10.7</td><td>52.0 42.2</td><td>20.2</td><td>14.9</td><td>0.2</td><td>77.0</td><td>56.8</td><td>50.2 21.4</td></table>

Table C5 Per-class IoU on SSCBench-nuScenes (validation set, 12-class scheme). Rows, notation and colors are as in Table C4. Truck and other-vehicle are separate classes, as in Table II of [2]. All metrics are reported in %.
<table><tr><td>Visibility Semantics</td><td></td><td>IoU mIoU</td><td>car</td><td>bieycle</td><td>mocccce</td><td>truuck</td><td>othi. cle ■</td><td>perrson</td><td>oad</td><td>sidealk</td><td>rud h.</td><td>buiddin</td><td>vestion</td><td>oth et</td></tr><tr><td>LMSCNet-SS</td><td></td><td></td><td></td><td>2.0%</td><td>&lt;0.1%</td><td>&lt;0.1%</td><td>0.6%</td><td>0.7% 0.2%</td><td>38.0%</td><td>8.5%</td><td>0.7%</td><td>21.4%</td><td>27.3%</td><td>0.5%</td></tr><tr><td>dense 一</td><td></td><td>37.6 38.3</td><td>15.0 16.6</td><td>24.0 25.5</td><td>0.0 0.0</td><td>0.0 0.0</td><td>8.8 15.0</td><td>17.010.9 18.8</td><td>11.3</td><td>43.4 44.2</td><td>15.2 15.0</td><td>8.2 12.1</td><td>29.6 17.3 30.0 17.9</td><td>5.4 9.1</td></tr><tr><td>dense</td><td>WI</td><td>38.6 39.2</td><td>26.1</td><td>29.2 28.7</td><td>5.8 4.2</td><td>17.6 14.8</td><td>33.5 33.4</td><td>34.0 34.3</td><td>25.1 25.4</td><td>46.3</td><td>26.0</td><td>18.0</td><td>33.1 20.0</td><td>24.9</td></tr><tr><td></td><td>WI WI-TTA</td><td>38.6</td><td>26.0 26.2</td><td>29.3</td><td>5.5</td><td>17.7</td><td>33.5</td><td>34.1</td><td>25.2</td><td>47.0 46.3</td><td>25.9 25.9</td><td>18.8 18.1</td><td>33.3 21.1 33.2 20.0</td><td>25.4 25.0</td></tr><tr><td>dense oracle</td><td>WI-TTA</td><td>39.1 51.3</td><td>26.1 21.7</td><td>28.7 27.7</td><td>4.0 0.0</td><td>14.7 0.0</td><td>33.5 20.5</td><td>34.5 21.0</td><td>25.5 13.4</td><td>47.0 61.8</td><td>25.9 23.7</td><td>18.8 21.2</td><td>33.4 21.1 35.5 23.8</td><td>25.5 11.3</td></tr><tr><td>oracle oracle</td><td>WI WI-TTA</td><td>51.6 51.6</td><td>31.7 31.8</td><td>33.5 33.6</td><td>5.8 5.6</td><td>17.2 17.3</td><td>40.0 40.0</td><td>40.2 40.4</td><td>27.5 27.6</td><td>65.4 65.5</td><td>35.5 35.4</td><td>23.7 23.7</td><td>38.6 26.2 38.8</td><td>26.9 27.1</td></tr><tr><td>SemCity-AE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>26.3</td><td></td></tr><tr><td></td><td>一</td><td>37.6 38.8</td><td>15.6 16.1</td><td>23.8 25.0</td><td>1.2 1.2</td><td>5.6 5.2</td><td>11.7 12.3</td><td>12.9 13.0</td><td>5.8 6.1</td><td>42.3</td><td>17.6</td><td>12.5</td><td>28.6 17.9</td><td>7.6</td></tr><tr><td>dense</td><td>- WI</td><td>38.6</td><td>27.3</td><td>32.0</td><td>10.3</td><td>26.3</td><td>35.7</td><td>34.4</td><td>27.9</td><td>43.1 46.7</td><td>17.3 26.7</td><td>13.0 17.5</td><td>29.8 19.2</td><td>7.5</td></tr><tr><td>dense</td><td>WI</td><td>39.6</td><td>27.4</td><td>32.0</td><td>10.4</td><td>24.6</td><td>35.4</td><td>33.8</td><td>27.8</td><td>47.0</td><td>26.7</td><td>16.9</td><td>32.8 20.8 34.1 22.4</td><td>17.1 18.0</td></tr><tr><td></td><td>WI-TTA</td><td>38.6</td><td>27.8</td><td>32.1</td><td>12.6</td><td>27.2</td><td>36.2</td><td>34.9</td><td>28.2</td><td>46.7</td><td>26.7</td><td>17.6</td><td>32.9 20.8</td><td>17.2</td></tr><tr><td>dense</td><td>WI-TTA</td><td>39.6</td><td>27.8</td><td>32.1</td><td>12.0</td><td>25.9</td><td>35.8</td><td>34.3</td><td>28.1</td><td>47.1</td><td>26.7</td><td>17.0</td><td>34.2 22.4</td><td>18.1</td></tr><tr><td>oracle</td><td></td><td>49.2</td><td>19.1</td><td>27.4</td><td>1.2</td><td>3.4</td><td>13.0</td><td>18.2</td><td>6.9</td><td>57.9</td><td>21.7</td><td>17.4</td><td>33.1 21.9</td><td>7.6</td></tr><tr><td>oracle</td><td>WI</td><td>50.0</td><td>33.4</td><td>36.1</td><td>13.4</td><td>28.9</td><td>40.5</td><td>40.1</td><td>33.1</td><td>63.5</td><td>34.9</td><td>21.6</td><td>38.5 26.3</td><td>22.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>16.2</td><td>30.0</td><td>40.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>oracle</td><td></td><td>50.0</td><td>33.7</td><td>36.2</td><td></td><td></td><td></td><td>40.5</td><td>33.3</td><td>63.5</td><td>34.9</td><td>21.7</td><td>26.4</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>38.6</td><td>23.0</td></tr><tr><td></td><td>WI-TTA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table D6 Visibility prior: train/test mismatch. Counterpart of Table 7 on the visibility axis: the semantic prior is held fixed at WafleIron while the visibility prior is varied. For each network the matched setting (trained and inferred with the dense prior) is compared against the network trained on the oracle prior and switched to the dense prior at inference only. ∆IoU (train-oracle minus matched) is the cost of not retraining. The oracle/oracle row is the matched oracle setting (oracle visibility, WI semantics), as in Table C4. All metrics are reported in %.

<table><tr><td>Train vis. Infer vis.</td><td></td><td>IoU</td><td>mIoU</td><td>∆IoU</td></tr><tr><td colspan="5">LMSCNet-SS</td></tr><tr><td>dense</td><td>dense</td><td>57.9</td><td>23.8</td><td></td></tr><tr><td>oracle</td><td>dense</td><td>36.2</td><td>15.3</td><td>-21.7</td></tr><tr><td>oracle</td><td>oracle</td><td>63.2</td><td>26.5</td><td></td></tr><tr><td colspan="5">SemCity-AE</td></tr><tr><td>dense</td><td>dense</td><td>57.0</td><td>25.5</td><td></td></tr><tr><td>oracle</td><td>dense</td><td>45.1</td><td>20.3</td><td>-11.9</td></tr><tr><td>oracle</td><td>oracle</td><td>61.0</td><td>28.3</td><td></td></tr></table>

## References

[1] Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., Gall, J.: Semantickitti: A dataset for semantic scene understanding of lidar sequences. In: ICCV (2019)

[2] Li, Y., Li, S., Liu, X., Gong, M., Li, K., Chen, N., Wang, Z., Li, Z., Jiang, T., Yu, F., Wang, Y., Zhao, H., Yu, Z., Feng, C.: Sscbench: A large-scale 3d semantic scene completion benchmark for autonomous driving. In: IROS (2024)

[3] Tian, X., Jiang, T., Yun, L., Mao, Y., Yang, H., Wang, Y., Wang, Y., Zhao, H.: Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. In: NeurIPS (2023)

[4] Roldao, L., De Charette, R., Verroust-Blondet, A.: 3d semantic scene completion: A survey. International Journal of Computer Vision (2022)

[5] Cao, A.-Q., Charette, R.: MonoScene: Monocular 3D semantic scene completion. In: CVPR (2022)

[6] Yu, Z., Zhang, R., Ying, J., Yu, J., Hu, X., Luo, L., Cao, S.-Y., Shen, H.-L.: Context and geometry aware voxel transformer for semantic scene completion. In: NeurIPS (2024)

[7] Yao, J., Zhang, J., Pan, X., Wu, T., Xiao, C.: DepthSSC: Monocular 3d semantic scene completion via depth-spatial alignment and voxel adaptation. In: WACV (2025)

[8] Cao, H., Behnke, S.: Difssc: Semantic lidar scan completion using denoising difusion probabilistic models. In: IROS (2025)

[9] Ma, Y., Mei, J., Yang, X., Wen, L., Xu, W., Zhang, J., Zuo, X., Shi, B., Liu, Y.: LiCROcc: Teach radar for accurate semantic occupancy prediction using lidar and camera. IEEE Robotics and Automation Letters 10(1), 852–859 (2025)

[10] Li, B., Deng, J., Zhang, W., Liang, Z., Du, D., Jin, X., Zeng, W.: Hierarchical temporal context learning for camera-based semantic scene completion. In: ECCV (2024)

[11] Roldao, L., De Charette, R., Verroust-Blondet, A.: LMSCNet: Lightweight multiscale 3D semantic completion. In: 3DV (2020)

[12] Yan, X., Gao, J., Li, J., Zhang, R., Li, Z., Huang, R., Cui, S.: Sparse single sweep lidar point cloud segmentation via learning contextual shape priors from scene completion. In: AAAI (2021)

[13] Mei, J., Yang, Y., Wang, M., Huang, T., Yang, X., Liu, Y.: Ssc-rs: Elevate lidar semantic scene completion with representation separation and bev fusion. In: IROS (2023)

[14] Yang, X., Zou, H., Kong, X., Huang, T., Liu, Y., Li, W., Wen, F., Zhang, H.: Semantic segmentation-assisted scene completion for lidar point clouds. In: IROS (2021)

[15] Jang, H.-K., Kim, J., Kweon, H., Yoon, K.-J.: Talos: Enhancing semantic scene completion via test-time adaptation on the line of sight. In: NeurIPS (2024)

[16] Puy, G., Boulch, A., Marlet, R.: Using a waffle iron for automotive point cloud semantic segmentation. In: ICCV (2023)

[17] Choy, C., Gwak, J., Savarese, S.: 4d spatiotemporal convnets: Minkowski convolutional neural networks. In: CVPR (2019)

[18] Martyniuk, T., Seele, J., Boulch, A., Puy, G., Marlet, R., Charette, R.: Exploring easy boosts for lidar semantic scene completion. In: IEEE ICIP (2026)

[19] Song, S., Yu, F., Zeng, A., Chang, A.X., Savva, M., Funkhouser, T.: Semantic scene completion from a single depth image. In: CVPR (2017)

[20] Garbade, M., Chen, Y.-T., Sawatzky, J., Gall, J.: Two stream 3d semantic scene completion. In: CVPR-W (2019)

[21] Cheng, R., Agia, C., Ren, Y., Li, X., Liu, B.: S3cnet: A sparse semantic scene completion network for lidar point clouds. In: CoRL (2020)

[22] Yi, L., Gong, B., Funkhouser, T.: Complete & label: A domain adaptation approach to semantic segmentation of lidar point clouds. In: CVPR (2021)

[23] Rist, C.B., Emmerichs, D., Enzweiler, M., Gavrila, D.M.: Semantic scene completion using local deep implicit functions on lidar data. In: TPAMI (2022)

[24] Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation. In: CVPR (2017)

[25] Li, P., Zhao, R., Shi, Y., Zhao, H., Yuan, J., Zhou, G., Zhang, Y.-Q.: Lode: Locally conditioned eikonal implicit scene completion from sparse lidar. In: ICRA (2023)

[26] Lee, J., Lee, S., Jo, C., Im, W., Seon, J., Yoon, S.-E.: SemCity: Semantic scene generation with triplane difusion. In: CVPR (2024)

[27] Lee, J., Im, W., Lee, S., Yoon, S.-E.: Difusion probabilistic models for scene-scale 3d categorical data. arXiv preprint arXiv:2301.00527 (2023)

[28] Zhang, X., Crowe, B., Heckman, C.: Octree latent difusion for semantic 3d scene generation and completion. arXiv preprint arXiv:2509.16483 (2025)

[29] Li, Y., Yu, Z., Choy, C., Xiao, C., Alvarez, J.M., Fidler, S., Feng, C., Anandkumar, A.: Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion. In: CVPR (2023)

[30] Zhang, Y., Zhu, Z., Du, D.: Occformer: Dualpath transformer for vision-based 3d semantic occupancy prediction. In: ICCV (2023)

[31] Huang, Y., Zheng, W., Zhang, Y., Zhou, J., Lu, J.: Tri-perspective view for vision-based 3d semantic occupancy prediction. In: CVPR

(2023)

[32] Wei, Y., Zhao, L., Zheng, W., Zhu, Z., Zhou, J., Lu, J.: Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In: ICCV (2023)

[33] Jiang, H., Cheng, T., Gao, N., Zhang, H., Lin, T., Liu, W., Wang, X.: Symphonize 3d semantic scene completion with contextual instance queries. In: CVPR (2024)

[34] Cao, H., Behnke, S.: Slcf-net: Sequential lidar-camera fusion for semantic scene completion using a 3d recurrent u-net. In: ICRA (2024)

[35] Wang, R., Ma, Y., Yao, Y., Tao, S., Li, H., Zhu, Z., Liu, Y., Zuo, X.: L2cocc: Lightweight camera-centric semantic scene completion via distillation of lidar model. In: IROS (2025)

[36] Lu, Z., Cao, B., Hu, Q.: Lidar-camera continuous fusion in voxelized grid for semantic scene completion. In: TCSVT (2024)

[37] Wang, S., Gan, T., Fang, Y., Xu, J., Ling, Q.: Learnable fusion semantic scene completion: Aggregation of multi-scale point cloud and image features. 2024 43rd Chinese Control Conference (CCC), 3597–3602 (2024)

[38] Xia, Z., Liu, Y., Li, X., Zhu, X., Ma, Y., Li, Y., Hou, Y., Qiao, Y.: Scpnet: Semantic scene completion on point cloud. In: CVPR (2023)

[39] Lu, H., Su, Y., Zhang, X., Gao, L., Xue, Y., Wang, L.: Vishall3d: Monocular semantic scene completion from reconstructing the visible regions to hallucinating the invisible regions. In: ICCV (2025)

[40] Han, Z., Higashita, R., Liu, J.: Voic: Visible-occluded decoupling for monocular 3d semantic scene completion. arXiv preprint arXiv:2512.18954 (2025)

[41] Zhou, H., Zhu, X., Song, X., Ma, Y., Wang, Z., Li, H., Lin, D.: Cylinder3d: An efective 3d framework for driving-scene lidar semantic segmentation. arXiv preprint arXiv:2008.01550 (2020)

[42] Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., Zhao, H.: Point transformer v3: Simpler faster stronger. In: CVPR (2024)

[43] Lai, X., Chen, Y., Lu, F., Liu, J., Jia, J.: Spherical transformer for lidar-based 3d recognition. In: CVPR (2023)

[44] Vora, S., Lang, A.H., Helou, B., Beijbom, O.: Pointpainting: Sequential fusion for 3d object detection. In: CVPR (2020)

[45] Yang, J., Shi, S., Wang, Z., Li, H., Qi, X.: St3d: Self-training for unsupervised domain adaptation on 3d object detection. In: CVPR (2021)

[46] Hornung, A., Wurm, K.M., Bennewitz, M., Stachniss, C., Burgard, W.: Octomap: An eficient probabilistic 3d mapping framework based on octrees. Autonomous Robots (2013)

[47] Rold˜ao, L., De Charette, R., Verroust-Blondet, A.: A statistical update of grid representations from range sensors. arXiv preprint arXiv:1807.08483 (2018)

[48] Hu, P., Ziglar, J., Held, D., Ramanan, D.: What you see is what you get: Exploiting visibility for 3d object detection. In: CVPR (2020)

[49] Kim, G., Kim, A.: Remove, then revert: Static point cloud map construction using multiresolution range images. In: IROS (2020)

[50] Lim, H., Hwang, S., Myung, H.: Erasor: Egocentric ratio of pseudo occupancy-based dynamic object removal for static 3d point cloud map building. IEEE Robotics and Automation Letters (2021)

[51] Boulch, A., Sautier, C., Michele, B., Puy, G., Marlet, R.: Also: Automotive lidar selfsupervision by occupancy estimation. In: CVPR (2023)

[52] Pan, M., Liu, J., Zhang, R., Huang, P., Li, X., Xie, H., Wang, B., Liu, L., Zhang, S.:

Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. In: ICRA (2024)

[53] Pan, M., Liu, L., Liu, J., Huang, P., Wang, L., Zhang, S., Xu, S., Lai, Z., Yang, K.: Uniocc: Unifying vision-centric 3d occupancy prediction with geometric and semantic rendering. arXiv preprint arXiv:2306.09117 (2023)

[54] Sulzer, R., Landrieu, L., Boulch, A., Marlet, R., Vallet, B.: Deep surface reconstruction from point clouds with visibility information. In: ICPR (2022)

[55] Cortinhal, T., Tzelepis, G., Erdal Aksoy, E.: Salsanext: Fast, uncertainty-aware semantic segmentation of lidar point clouds. In: International Symposium on Visual Computing, pp. 207–222 (2020). Springer

[56] Liu, W., Kang, Z., Yu, Y., Gong, Z., Zheng, Y., Huang, X., Guan, H., Ma, L., Zhang, D.: A dual-path network for semantic scene completion of single-frame lidar point clouds. In: Int. J. Appl. Earth Obs. (2026)

[57] Cao, A.-Q., Dai, A., De Charette, R.: Pasco: Urban 3d panoptic scene completion with uncertainty awareness. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14554–14564 (2024). IEEE