# L2GSSL: Label-Efficient Local-to-Global Self-Supervised Learning for Volumetric Healthcare Data

Po-Cheng Cheng, Hung-Hsuan Chen

Computer Science and Information Engineering, National Central University

bczheng1003@gmail.com, hhchen1105@acm.org

Abstract—Healthcare institutions routinely accumulate highdimensional computed tomography (CT) volumes, but only a fraction can be annotated by experts. We present L2GSSL, a local-to-global self-supervised learning framework that leverages unlabeled volumes by decomposing pretraining into intra-slice and inter-slice representation spaces. A slice encoder learns viewinvariant local features, and a global encoder aggregates the resulting tokens to model volumetric context. The stop-gradient isolates the local space from the global objective, while slicewise execution and a three-phase schedule support training under constrained GPU memory. On LIDC-IDRI lung nodule malignancy classification, L2GSSL reaches 0.9229 AUROC with only 10% of the labels and 0.9377 with 25%, exceeding the volume-level VICReg by 0.0229 and 0.0294, respectively. With all labels, the strongest supervised baseline exceeds L2GSSL by only 0.0044 AUROC, consistent with self-supervised pretraining providing its greatest value when annotations are limited. Without adaptation to the external LNDb cohort, a model fine-tuned on 25% of LIDC-IDRI obtains 0.8138 AUROC and 0.7589 AUPRC, improving over the strongest evaluated self-supervised baseline by 0.1130 and 0.0954. Experiments across data sources, thoracic and abdominal anatomy, downstream tasks, and two backbones examine healthcare-data variety, while lesion-slice masking shows that the predictions depend on lesion-bearing content. We release the code for reproducibility.

Index Terms—self-supervised learning, healthcare data, 3D medical imaging, computed tomography, representation learning

## I. INTRODUCTION

Healthcare institutions routinely accumulate archives of computed tomography (CT) scans, but expert-curated labels remain available for only a fraction of those data. Each scan is also a high-dimensional volumetric object that combines fine pathological appearance, organ-scale structure, and relationships across ordered slices. The resulting challenge is how to extract useful representations from abundant unlabeled volumes when downstream annotations and accelerator memory are limited. Healthcare archives also vary across institutions, anatomy, acquisition, and annotation practice, so representation quality must be assessed beyond a single internal dataset.

Self-supervised learning (SSL) addresses annotation scarcity by learning from the images themselves. Existing medical SSL approaches include restoration-based pretext tasks [1], [2], contrastive objectives, and volumetric spatial objectives such as VoCo [3]. However, applying a single objective to the final volume representation does not explicitly acknowledge the hierarchy of CT data. An individual slice contains texture, boundary, and local lesion evidence, while the ordered slice sequence carries anatomical continuity and volumetric context. Treating both as a single optimization space may allow a global loss to overwrite useful slice-level structure. Fully 3D pipelines also make it difficult to exploit widely available twodimensional pretrained encoders and can exceed the resources available to smaller research or clinical sites.

We therefore propose Local-to-Global Self-Supervised Learning (L2GSSL), a structured pretraining framework that learns from unlabeled 3D medical images. It independently regularizes local slice representations and global volume representations using VICReg [4]. A stop-gradient boundary makes the dependency directional: local embeddings serve as inputs to the global encoder during the forward pass, while the global objective does not update the local encoder during backpropagation. A three-phase optimization procedure first establishes slice-level features, then learns both spaces, and finally consolidates inter-slice modeling. Slice-wise processing and feature accumulation avoid keeping an entire end-to-end 3D gradient graph in memory, making hierarchical pretraining feasible within a constrained GPU-memory budget.

We evaluate whether L2GSSL turns unlabeled volumetric data into useful initialization when expert labels are scarce. Pulmonary-nodule malignancy experiments on LIDC-IDRI [5] use 10%, 25%, and 100% of the downstream labels, while direct evaluation on LNDb [6] tests transfer across data sources without target-domain adaptation. A second backbone, lesionslice masking, and integration into an abdominal-trauma system using RSNA [7] and AbdomenCT-1K [8] expand the evaluation across architectures and tasks. Our contributions are as follows.

• We propose a hierarchical self-supervised learning framework that learns intra-slice and inter-slice representations through separate objectives and directional gradient flow.

• We develop a resource-conscious training procedure that combines slice-wise execution with phased local and global optimization.

• We demonstrate that the largest gains occur with 10% and 25% of the downstream labels and diminish with full supervision, matching the intended role of self-supervised pretraining under limited expert annotation.

• We evaluate transfer and model behavior through direct

LIDC-IDRI-to-LNDb evaluation, thoracic and abdominal CT tasks, two backbone designs, radiologist variability, and lesion dependence.

## II. RELATED WORK

## A. Volumetric Medical Image Learning

Three-dimensional CNNs such as 3D U-Net [9] and V-Net [10] established direct volumetric processing, while transformer-based systems such as UNETR [11] model longer range dependencies. These architectures are effective, but operate on high-dimensional inputs. An alternative is to encode slices and then aggregate their embeddings. The Medical Slice Transformer (MST) combines a DINOv2-initialized image encoder [12] with inter-slice modeling [13]; this separation aligns naturally with the structure of CT and allows reuse of strong image-level priors. MaxViT [14] offers a second hybrid convolution-attention hierarchy through which the portability of a local-to-global objective can be tested.

Recent work adapts volumetric contrastive learning to individual abdominal CT slices and subsequently aggregates pretraining features with a sequential model [15]. This 2D-VoCo formulation reduces the burden of direct 3D pretraining while learning slice-level spatial-semantic features. L2GSSL follows the same resource-conscious direction but addresses a different representation problem: it assigns independent objectives to both the slice space and the ordered inter-slice space, rather than restricting self-supervision to the local encoder before downstream sequence modeling.

These two architectural families make different trade-offs. Direct 3D models expose the network to local appearance and volumetric context simultaneously, but their activation memory grows with all three spatial dimensions. Slice-sequence models instead compress each image plane before learning across an ordered token sequence. This 2.5D dataflow preserves through plane context without applying every operation to a dense 3D tensor and provides a natural interface to image encoders pretrained on large natural-image collections.

L2GSSL differs from direct 3D architectures because its pretraining objectives do not operate only on a dense volumetric representation. Slices are encoded in groups and reconstructed as an ordered embedding sequence for global modeling, preserving compatibility with image-level pretrained encoders. L2GSSL also differs from conventional encode-then aggregate pipelines: although these pipelines separate forward computation into slice encoding and sequence aggregation, they do not necessarily separate the corresponding optimization objectives. A loss attached only to the final volume embedding can still propagate through and reshape the slice encoder. L2GSSL explicitly regularizes both representation spaces and prevents the global objective from updating the local encoder. It is therefore a pretraining strategy for hierarchical volumetric models rather than an alternative downstream backbone. In this study, we instantiate L2GSSL with MST and MaxViT to evaluate the framework across an explicit slice-sequence architecture and a staged convolution-attention backbone.

## B. Self-Supervised Learning for Medical Images

Early medical SSL approaches reconstruct transformed inputs or predict spatial context. Models such as Genesis apply restoration transformations to 3D images [2], and context restoration learns visual representations without labels [1]. Contrastive approaches align augmented views, but often depend on negative samples. In ordered medical volumes, anatomically similar neighboring slices are problematic negatives. SimSiam avoids negative pairs through a predictor and stop-gradient [16]; MoCo v2 uses a momentum encoder and queue [17]; VICReg prevents collapse by regularizing invariance, variance, and covariance [4]. VoCo instead learns spatial consistency from overlapping 3D crops [3]. L2GSSL differs in where the objective is applied: it creates separate local and global SSL spaces and controls the information flow between them.

Contrastive and joint-embedding methods learn representations by enforcing agreement between augmented views. Their behavior depends on how positive pairs are constructed, whether and how negative samples are used, and the spatial unit being embedded. This distinction is important for CT because adjacent slices are separate inputs yet may contain nearly iden tical anatomical structures. Non-contrastive objectives avoid explicitly pushing such samples apart. VICReg is particularly suitable for hierarchical training because it prevents representation collapse through batch statistics without a momentum encoder, memory queue, or predictor. We consequently use the same objective family at both spatial levels, while keeping their projectors, inputs, and gradient paths distinct.

Table I compares the design characteristics of the SSL baselines evaluated and the related volumetric approaches. MoCo v2, SimSiam, volume-level VICReg, and L2GSSL are evaluated within the controlled MST setting. We include VoCo experimentally as a resource-constrained reference with a different 3D backbone. Among them, L2GSSL applies independent losses to slice and volume embeddings while preventing the volume loss from updating the slice encoder.

Beyond self-supervised representation learning, L2GSSL is related to decoupled learning methods that partition a network into modules optimized by local objectives. Associated Learning, DeInfoReg, and SCPL reduce dependence on a single end-to-end gradient path through local objectives [18]– [20]. L2GSSL applies the decoupling principle to the spatial hierarchy of volumetric medical images: local and global modules use separate self-supervised objectives, and the global objective is prevented from updating the local encoder.

## C. External Evaluation and Model-Behavior Analysis

Medical-imaging reporting guidelines emphasize the evaluation of independent datasets to assess sensitivity to changes in acquisition source and annotation practice [21]. External evaluation therefore provides evidence distinct from another internal split, although it does not establish prospective clinical validity.

Model-behavior analyses require similar care. Perturbation tests examine whether predictions change when selected image content is removed, providing evidence of input dependence without establishing causal reasoning or clinical utility.

TABLE I  
DESIGN COMPARISON OF THE EVALUATED SSL BASELINES AND CLOSELY RELATED VOLUMETRIC PRETRAINING APPROACHES. “LOCAL/GLOBAL SPLIT” DENOTES SEPARATE OBJECTIVES AND CONTROLLED GRADIENT PATHS, NOT MERELY A HIERARCHICAL BACKBONE.
<table><tr><td>Method</td><td>Pretraining input</td><td>Alignment target</td><td>Explicit negatives</td><td>Auxiliary network state</td><td>Local/global split</td></tr><tr><td>MoCo v2 [17]</td><td>Slice-aggregated volume</td><td>Final volume embedding</td><td>Yes</td><td>Momentum encoder and queue</td><td>No</td></tr><tr><td>SimSiam [16]</td><td>Slice-aggregated volume</td><td>Final volume embedding</td><td>No</td><td>Predictor</td><td>No</td></tr><tr><td>Volume VICReg [4]</td><td>Slice-aggregated volume</td><td>Final volume embedding</td><td>No</td><td>None</td><td>No</td></tr><tr><td>VoCo [3]</td><td>Overlapping 3D crops</td><td>Volumetric spatial relation</td><td>Spatially defined</td><td>EMA teacher</td><td>No</td></tr><tr><td>2D-VoCo [15]</td><td>Individual CT slices</td><td>Slice-level spatial-semantic relation</td><td>Spatially defined</td><td>None</td><td>No</td></tr><tr><td>L2GSSL</td><td>Slices and ordered tokens</td><td>Slice and volume embeddings</td><td>No</td><td>None</td><td>Yes</td></tr></table>

## III. METHOD

## A. Hierarchical Representation Spaces

Let a CT volume be $V = [ s _ { 1 } , \ldots , s _ { Z } ] ,$ , where $s _ { i } \in \mathbb { R } ^ { H \times W }$ is a slice. A local encoder $f _ { l } ( \cdot ; \theta _ { l } )$ produces $z _ { l , i } = f _ { l } ( s _ { i } ) \in \mathbb { R } ^ { d _ { l } }$ A global encoder aggregates the ordered local tokens:

$$
z _ { g } = f _ { g } ( z _ { l , 1 } , \ldots , z _ { l , Z } ; \theta _ { g } ) \in \mathbb { R } ^ { d _ { g } } .\tag{1}
$$

Two stochastic, anatomy-preserving views are produced for each input. Both the local and global spaces use VICReg. For paired batches $Z$ and $Z ^ { \prime } { \mathrm { . } }$ , its objective is

$$
{ \mathcal { L } } _ { \mathrm { V R } } ( Z , Z ^ { \prime } ) = \lambda s ( Z , Z ^ { \prime } ) + \mu [ v ( Z ) + v ( Z ^ { \prime } ) ] + \nu [ c ( Z ) + c ( Z ^ { \prime } ) ] ,\tag{2}
$$

where s, v, and c are the invariance, variance, and covariance terms, respectively. The hyperparameters λ, µ, and ν control the weight of view invariance, feature dispersion, and feature decorrelation. Let $Z _ { l } ^ { ( 1 ) }$ and $Z _ { l } ^ { ( 2 ) }$ denote the batches of projected local representations, and let $Z _ { g } ^ { ( 1 ) }$ and $Z _ { g } ^ { ( 2 ) }$ denote their projected global representation batches. The local and global objectives are

$$
\mathcal { L } _ { l } = \mathcal { L } _ { \mathrm { V R } } \left( Z _ { l } ^ { ( 1 ) } , Z _ { l } ^ { ( 2 ) } \right) , \qquad \mathcal { L } _ { g } = \mathcal { L } _ { \mathrm { V R } } \left( Z _ { g } ^ { ( 1 ) } , Z _ { g } ^ { ( 2 ) } \right) .\tag{3}
$$

The overall hierarchical pretraining objective is $\mathcal { L } = \mathcal { L } _ { l } + \mathcal { L } _ { g } ,$ where $\mathcal { L } _ { l }$ optimizes intra-slice representations and $\mathcal { L } _ { g }$ optimizes inter-slice volumetric representations.

More explicitly, let $\bar { \boldsymbol { A } } , \boldsymbol { B } \in \mathbb { R } ^ { N \times D }$ denote the projected representations of two augmented views in one of the two spaces. The invariance term aligns paired samples,

$$
s ( A , B ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \lVert A _ { n } - B _ { n } \rVert _ { 2 } ^ { 2 } .\tag{4}
$$

For feature dimension j, let $\sigma _ { j } ( A ) = \sqrt { \operatorname { V a r } ( A _ { : , j } ) + \epsilon } .$ The variance term maintains dispersion above margin γ,

$$
v ( A ) = \frac { 1 } { D } \sum _ { j = 1 } ^ { D } \operatorname* { m a x } ( 0 , \gamma - \sigma _ { j } ( A ) ) .\tag{5}
$$

After centering A across its batch dimension, let $C ( A )$ be its empirical feature covariance matrix. The covariance term reduces redundancy without forcing different samples apart,

$$
c ( A ) = \frac { 1 } { D } \sum _ { i \neq j } C ( A ) _ { i j } ^ { 2 } .\tag{6}
$$

The three terms serve different purposes: invariance makes the representation stable to the permitted augmentations, variance prevents every input from mapping to a constant vector, and covariance discourages multiple coordinates from encoding the same variation. The local and global branches use the same mathematical form but compute their statistics over different representation spaces. Local batches contain projected slice embeddings; global batches contain projected volume embeddings. Consequently, the loss parameters are shared conceptually, not through a common projector or a mixed batch of local and global features.

To isolate local representation from the global objective, we apply stop-gradient to the local embeddings before global encoding. The embeddings therefore contribute to volume construction during the forward pass, but the global loss does not propagate gradients to or update the local encoder, thus separating optimization across the two spatial levels without implying statistical independence between their representations.

The asymmetric gradient flow in Fig. 1 separates the optimization of the two representation levels. The local objective encourages the slice encoder to preserve view-stable intra-slice content, whereas the global objective learns relationships among the ordered local embeddings. Applying stop-gradient before global encoding allows local embeddings to contribute to the volume representation while preventing the global objective from updating the local encoder. Thus, the two representation spaces remain functionally coupled during the forward pass but are optimized through separate gradient paths.

## B. Backbone Instantiations

We instantiate L2GSSL with two backbone designs. In the Medical Slice Transformer (MST), a ViT image encoder extracts local tokens from each slice, and a slice transformer models inter-slice context. For the MST experiments, the local ViT encoder is initialized with publicly available DINOv2 weights [12] before L2GSSL pretraining on medical images. This architecture provides an explicit boundary between slice encoding and sequence aggregation. In the MaxViT instantiation, the backbone is initialized with ImageNet-pretrained weights, with the stem and first two stages forming the local encoder and the remaining two stages forming the global encoder. This partition follows the backbone hierarchy rather than an explicit slice-aggregation boundary: early stages learn fine-grained spatial features, and later stages learn higher-level representations. The two instantiations test L2GSSL across an explicit slice transformer and a staged convolution-attention network.

![](images/3731d424e3ed6c91ebdc6704985560da85539a969e19c866365a51ec830be2ad.jpg)  
Fig. 1. Computational graph of L2GSSL. Local embeddings are inputs to the global encoder, but the global loss is detached at the local encoder. Solid arrows show forward computation; colored arrows show parameter updates.

## C. Slice-Wise Execution and Phased Optimization

Slices are processed in mini-batches, and their embeddings are accumulated in anatomical order before global encoding. This execution pattern reduces the peak activation memory needed for stored local activations and enables volume modeling from compact tokens. Training follows three phases: local-only optimization for the first 30% of epochs, joint local and global optimization for the next 40%, and global-only optimization for the final 30%. Stop-gradient prevents the global objective from updating the local encoder during the joint phase. In the final phase, the local encoder is frozen while the global encoder is optimized on its fixed embeddings. This schedule allows the global encoder to operate on local representations that have first undergone dedicated pretraining.

The phases control when each representation level becomes trainable. The local-only phase establishes slice features without requiring an initially untrained global encoder to provide a useful learning signal. During the joint phase, the local encoder continues to follow its own objective while the global encoder begins to model the ordered embeddings; the stop-gradient keeps these updates on separate parameter paths. The globalonly phase then fixes the local representation distribution and concentrates the remaining optimization on inter-slice relationships. This schedule changes the optimization order rather than the downstream architecture, and the component study evaluates how its value changes with encoder initialization.

Let b be the number of slices processed at once, and let $K = \lceil Z / b \rceil$ be the number of contiguous slice groups. For each volume, the paired augmented views are processed as follows:

1) Select the active objectives and trainable parameter groups for the current phase: $( \mathcal { L } _ { l } , \theta _ { l } )$ in the local-only phase, both $( \mathcal { L } _ { l } , \theta _ { l } )$ and $( \mathcal { L } _ { g } , \theta _ { g } )$ in the joint phase, and $( \mathcal { L } _ { g } , \theta _ { g } )$ in the global-only phase.

2) Partition the ordered slices into K contiguous groups. When the local objective is active, forward one group at a time through the local encoder and projector, scale the group’s local loss by $1 / K$ , and backpropagate it to accumulate parameter gradients. When the global objective is also active, append a detached copy of each group’s embeddings in the original z-axis order.

3) In the global-only phase, keep the local encoder frozen and generate and store the ordered embeddings without constructing a local backward graph.

4) When the global objective is active, concatenate the stored embeddings at the stop-gradient boundary, forward the sequence through the global encoder and projector, and backpropagate $\mathcal { L } _ { g } .$

5) Apply the optimizer steps for the parameter groups active in the current phase. After pretraining, discard the projector heads and pass the pretrained volume representation to the downstream classifier.

Accordingly, the phase-specific execution paths are Steps 1, 2, and 5 for the local-only phase; Steps 1, 2, 4, and 5 for the joint phase; and Steps 1, 3, 4, and 5 for the global-only phase. Step 3 prepares the global input without local backpropagation, whereas Step 4 optimizes the global encoder and therefore applies to both the joint and global-only phases.

Processing slice groups sequentially avoids storing all slice activations simultaneously, and the global encoder operates on the output compact embedding sequence. Gradient accumulation approximates a larger optimization batch, but normalization statistics, when present, are computed within each slice group.

## D. Peak-Activation Analysis

The activation-memory advantage can be expressed independently of a particular GPU. Let $a _ { l }$ denote the storage required to retain the local-encoder computation graph for one slice, let $a _ { l } ^ { \mathrm { f w d } }$ denote the temporary storage for a frozen local-encoder forward pass, let $e _ { l }$ denote the storage for one detached local embedding, and let $A _ { g } ( Z )$ denote the internal activation storage of the global encoder for a sequence of $Z$ embeddings. Parameters, optimizer state, input, and frameworkallocation memory are omitted because this analysis isolates the activation terms affected by slice grouping and gradient detachment.

A conventional end-to-end eager implementation without activation recomputation or offloading retains the local computation graphs for all slices until the global backward pass, giving

$$
M _ { \mathrm { e 2 e } } = O ( Z a _ { l } + A _ { g } ( Z ) ) .\tag{7}
$$

During the L2GSSL joint phase, each local group completes its backward pass before global encoding. The detached embeddings persist across groups, but the local and global activation peaks occur at different times. The corresponding peak storage is therefore

$$
M _ { \mathrm { L 2 G S S L } } ^ { \mathrm { j o i n t } } = O ( Z e _ { l } + \operatorname* { m a x } \{ b a _ { l } , A _ { g } ( Z ) \} ) , \qquad b \ll Z .\tag{8}
$$

The local-only phase requires $O ( b a _ { l } )$ activation storage because no global sequence is constructed. In the global-only phase, the local encoder is frozen, giving $O \big ( Z e _ { l } + \operatorname* { m a x } \{ b a _ { l } ^ { \mathrm { f w d } } , A _ { g } ( Z ) \} \big )$ The reduction applies to the local-activation term; the ordered embedding sequence and global encoder still scale with the number of slices. Slice grouping also does not reduce the number of local-encoder forward operations required for a volume, and groups trained with the local objective still require backward computation. Sequential groups may introduce additional execution overhead. The analysis therefore explains why the design can lower peak activation storage under the stated execution model, but it does not imply lower wall-clock time or higher data-ingestion throughput. The group size b exposes the practical trade-off: smaller groups reduce the local-activation term, whereas larger groups may improve within-group hardware utilization when memory permits.

## IV. EXPERIMENTS

## A. Datasets and Tasks

LIDC-IDRI contains 1,018 thoracic CT cases with multireader lung-nodule annotations [5]. After nodule-level processing, the study contains 1,625 samples; malignancy tendency is formulated as a binary classification task. Splitting is performed at the patient/scan level before crop or slice extraction, ensuring that related samples from a scan remain in one partition. Fine-tuning uses 10%, 25%, or 100% of the training labels, corresponding to 104, 260, and 1,040 nodules.

LNDb contains 294 chest CT scans from a Portuguese institution and annotations from five participating radiologists [6]. We retain nodules with malignancy assessments and align the label definition and preprocessing with LIDC-IDRI. Models fine-tuned using 25% of LIDC-IDRI are evaluated directly on LNDb; no LNDb fine-tuning is performed.

For task-level breadth, the RSNA abdominal-trauma data contain 3,147 patients collected from 23 institutions in 14 countries [7]. The task predicts kidney, liver, spleen, bowel, and extravasation findings. AbdomenCT-1K supplies additional unlabeled abdominal CT for pretraining [8]. We integrate L2GSSL initialization into four experts of the second-place solution to the 2023 RSNA Abdominal Trauma Detection competition while preserving its downstream procedures.

Table II separates the role of each collection. LIDC-IDRI is the primary controlled benchmark because it supports nodulelevel low-label experiments and multi-reader analysis. LNDb is held outside training and model adaptation to test direct cross-dataset transfer. RSNA and AbdomenCT-1K form a second anatomical setting: the former supplies the downstream abdominal-trauma task, while the latter contributes anatomically aligned volumes during pretraining. This design tests label efficiency, source transfer, and task breadth without treating all datasets as interchangeable or combining their labels.

The LIDC-IDRI split is created before any nodule crop or slice is extracted. This order is essential: splitting the derived samples allows visually correlated slices or nodules from one scan to enter both training and testing. All compared methods use the same leakage-controlled partition. LNDb is processed with the LIDC-IDRI pipeline and the matching malignancy definition, but ambiguous cases without a usable malignancy assessment are not converted into definite labels. No LNDb sample contributes to pretraining, fine-tuning, or checkpoint selection in the reported external evaluation.

## B. Experimental Setup and Evaluation Protocols

We compare supervised MST-DINOv2, MoCo v2, SimSiam, volume-level VICReg, and L2GSSL using the same backbone and pipeline where applicable.

For downstream MST evaluation, linear probing trains only the classifier, whereas full fine-tuning updates the entire network. All compared methods use the same downstream optimization and test-time augmentation protocol, and checkpoints are selected using the validation area under the receiver operating characteristic curve. Linear-probing results use three seeds, and fine-tuning uses five runs. Complete hyperparameter settings are provided in the released code.

CT values are clipped to [−1000, 1000] Hounsfield units and scaled to [−1, 1]. Volumes are resampled as needed, and crop-or-pad uses the minimum volume intensity for padding. SSL views use random gamma correction (probability 0.3, log-gamma [−0.2, 0.2]), low-frequency intensity distortion (probability 0.2, coefficient magnitude 0.5), Gaussian blur (probability 0.2, standard deviation [0, 1.0]), affine transformation (probability 0.5, scale [0.95, 1.05], rotation up to 10<sup>◦</sup>), and Gaussian noise (probability 0.3, standard deviation [0, 0.05]).

We report the area under the receiver operating characteristic curve (AUROC) and the area under the precision–recall curve (AUPRC) for discrimination under class imbalance. Means and standard deviations summarize independent runs.

The protocols address distinct evaluation questions rather than form a single cross-dataset ranking. LIDC-IDRI supports controlled representation and label-efficiency comparisons, LNDb evaluates direct transfer without target-domain adaptation, and the RSNA/AbdomenCT-1K setting evaluates transfer across anatomy and downstream task. Within LIDC-IDRI, the protocols distinguish fixed-batch and memory-capped representation evaluation from fine-tuning at different label fractions. Table III summarizes the controlled factor and primary question for each protocol.

## C. Label-Efficient Transfer on LIDC-IDRI

We evaluate representation separability and label efficiency under controlled LIDC-IDRI protocols.

Table IV shows that L2GSSL is most beneficial when annotations are limited. With 10% and 25% of labels, it achieves the highest AUROC and AUPRC among the evaluated methods. Relative to volume-level VICReg, the AUROC gains are 0.0229 and 0.0294, respectively. When all training labels are used, supervised MST-DINOv2 has the highest AUROC by 0.0044, while SimSiam has the highest AUPRC by 0.0080. The diminishing margin is consistent with SSL serving primarily as an initialization advantage when labels are scarce.

TABLE II  
DATASETS AND THEIR ROLES IN THE EVALUATION.
<table><tr><td>Dataset</td><td>Anatomy/modality</td><td>Available scale</td><td>Use in this study</td><td>Label use</td><td>Evaluation role</td></tr><tr><td>LIDC-IDRI [5]</td><td>Thoracic CT</td><td>1,018 cases; 1,625 processed nodules</td><td>SSL and downstream training</td><td>104, 260, or 1,040 training nodules</td><td>Internal discrimination and reader analysis</td></tr><tr><td>LNDb [6]</td><td>Thoracic CT</td><td>294 scans</td><td>External testing only</td><td>No target-domain training</td><td>Direct cross-dataset transfer</td></tr><tr><td>RSNA 2023 [7]</td><td>Abdominal CT</td><td>3,147 patients</td><td>Downstream trauma pipeline</td><td>Competition task labels</td><td>Multi-organ task transfer</td></tr><tr><td>AbdomenCT-1K [8]</td><td>Abdominal CT</td><td>Auxiliary volumetric collection</td><td>Additional SSL source</td><td>Labels not used by SSL</td><td>Anatomically aligned pretraining</td></tr></table>

TABLE III  
EVALUATION PROTOCOLS AND THE CLAIMS THEY ARE DESIGNED TO TEST.
<table><tr><td>Protocol</td><td>Encoder update</td><td>Supervision</td><td>Controlled factor</td><td>Primary question</td></tr><tr><td>Fixed-batch linear probe</td><td>Frozen</td><td>LIDC-IDRI labels</td><td>Local batch 8; two accumulation steps</td><td>Are pretrained features linearly separable?</td></tr><tr><td>VRAM-capped linear probe</td><td>Frozen</td><td>LIDC-IDRI labels</td><td>Peak VRAM constrained to 47–48 GB</td><td>How effectively is a fixed memory budget used?</td></tr><tr><td>Label-fraction fine-tuning</td><td>Full network</td><td>10%, 25%, or 100% of LIDC-IDRI</td><td>Shared split and downstream pipeline</td><td>Does pretraining reduce dependence on labels?</td></tr><tr><td>Direct LNDb evaluation</td><td>Frozen at inference</td><td>No LNDb training labels</td><td>LIDC-IDRI model evaluated without adaptation</td><td>Does performance transfer across data sources?</td></tr><tr><td>RSNA pipeline integration</td><td>Full downstream pipeline</td><td>RSNA task labels</td><td>Original multi-expert inference design retained</td><td>Does initialization transfer to abdominal trauma CT?</td></tr></table>

Linear probing isolates representation quality. Under identi cal batch size (Table V), L2GSSL improves over volume-level VICReg in both AUROC and AUPRC.

## D. Cross-Dataset and Task Transfer

1) Direct Cross-Dataset Evaluation: We evaluate whether representations learned on LIDC-IDRI transfer to a different data source without target-domain adaptation.

Table VI reports direct LNDb evaluation. L2GSSL is best in both discrimination metrics. Compared with volumelevel VICReg, the strongest evaluated SSL baseline, it gains 0.1130 AUROC and 0.0954 AUPRC. Because no target-domain adaptation is performed, this experiment provides stronger evidence of transfer than another internal split.

2) Abdominal Trauma Transfer: We evaluate whether L2GSSL initialization transfers beyond lung-nodule classification to abdominal CT and a multi-organ trauma-prediction pipeline.

For abdominal-trauma pretraining, the projector is a threelayer multilayer perceptron with hidden dimension 4096, and the evaluation uses severity-weighted RSNA competition loss. In the RSNA multi-expert pipeline, the original competition loss is 0.3597, while the best configuration incorporating L2GSSL-pretrained experts obtains 0.3551. This result shows that L2GSSL can use additional unlabeled anatomically aligned AbdomenCT-1K volumes without changing the downstream inference design.

## E. Resource and Volumetric Baseline Comparisons

1) Common GPU-Memory Budget: The VRAM-capped ex periment complements the fixed-batch comparison by replacing an identical optimization configuration with a common GPU memory envelope for the MST-based SSL methods. Each SSL method is constrained to a peak VRAM budget of 47– 48 GB, reflecting the practical case in which the accelerator capacity is fixed. This comparison evaluates the representation quality achievable within the same memory envelope rather than isolating the SSL objective alone. Table VII shows that L2GSSL achieves the highest AUROC and AUPRC under the common cap, followed by volume-level VICReg.

2) Fully 3D Reference: VoCo operates directly on dense 3D crops and does not share the slice-encoder and interslice aggregation structure used by the MST-based methods. Incorporating it into the controlled comparisons above would therefore require changing its backbone, input formulation, and training procedure. We instead evaluate VoCo separately as a fully 3D system-level reference. We adapt VoCo to the LIDC-IDRI nodule setting using a 3D Swin encoder and VoCo-style volumetric crops, and evaluate both random and MONAI-pretrained initialization. Complete preprocessing and optimization settings are provided in the released code. Table VIII shows that L2GSSL uses slightly less measured peak VRAM and achieves higher linear probing and fine-tuning performance than both VoCo variants in all evaluated settings: linear probing and fine-tuning under 10%, 25%, and 100% available labels.

## F. Backbone Portability

We test whether the local-to-global formulation remains effective when instantiated with a second backbone.

Table IX shows that L2GSSL achieves the highest AUROC and AUPRC with MaxViT. These results support portability beyond MST without implying universality across architectures.

## G. Model-Behavior Analyses

1) Lesion Dependence: We test whether the predictions depend on lesion-bearing content by masking the corresponding slices at inference. Table X shows a 73.27 percentage-point decline in malignant true-positive rate, together with large AUROC and AUPRC drops. The result indicates that the model relies substantially on information in malignant lesion slices rather than only on background or dataset-level shortcuts.

2) Radiologist Context and Sensitivity to Uncertain Cases: This analysis serves two purposes. First, it places the model ROC curves in the context of a pooled operating point derived from per-nodule held-out radiologist judgments. Second, it assesses how model discrimination changes when the ROC evaluation retains less-certain cases in which the held-out radiologist assigns a score of 3 despite a definite reference from the remaining radiologists.

TABLE IV  
LIDC-IDRI FULL FINE-TUNING UNDER IDENTICAL BATCH-SIZE CONSTRAINTS. BEST VALUES ARE BOLD.
<table><tr><td></td><td colspan="2">10% labels (104)</td><td colspan="2">25% labels (260)</td><td colspan="2">100% labels (1,040)</td></tr><tr><td>Method</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Supervised MST-DINOv2</td><td> $0 . 6 5 6 4 \pm 0 . 0 7 8 9$ </td><td> $0 . 4 5 5 7 \pm 0 . 0 7 0 6$ </td><td> $0 . 8 6 9 5 \pm 0 . 0 4 9 9$ </td><td> $0 . 7 8 7 5 \pm 0 . 0 7 8 1$ </td><td> $\mathbf { 0 . 9 4 4 8 \pm 0 . 0 0 7 0 }$ </td><td> $0 . 9 0 4 4 \pm 0 . 0 0 7 0$ </td></tr><tr><td>MoCo v2</td><td> $0 . 5 9 8 6 \pm 0 . 0 1 7 5$ </td><td> $0 . 4 2 4 0 \pm 0 . 0 1 9 0$ </td><td> $0 . 8 2 2 1 \pm 0 . 0 7 2 7$ </td><td> $0 . 7 2 9 0 \pm 0 . 1 2 7 6$ </td><td> $0 . 9 3 6 0 \pm 0 . 0 0 7 3$ </td><td> $0 . 8 8 9 3 \pm 0 . 0 1 2 8$ </td></tr><tr><td>SimSiam</td><td> $0 . 6 8 4 7 \pm 0 . 1 0 2 6$ </td><td> $0 . 5 0 3 3 \pm 0 . 1 2 1 2$ </td><td> $0 . 8 0 0 7 \pm 0 . 1 4 6 2$ </td><td> $0 . 6 9 3 6 \pm 0 . 1 9 6 6$ </td><td> $0 . 9 3 8 9 \pm 0 . 0 0 2 4$ </td><td> $\mathbf { 0 . 9 1 2 3 \pm 0 . 0 0 5 5 }$ </td></tr><tr><td>Volume VICReg</td><td> $0 . 9 0 0 0 \pm 0 . 0 0 4 8$ </td><td> $0 . 8 5 7 3 \pm 0 . 0 0 3 4$ </td><td> $0 . 9 0 8 3 \pm 0 . 0 0 2 2$ </td><td> $0 . 8 6 8 6 \pm 0 . 0 1 3 3$ </td><td> $0 . 9 2 9 6 \pm 0 . 0 0 7 7$ </td><td> $0 . 8 8 6 7 \pm 0 . 0 0 9 2$ </td></tr><tr><td>L2GSSL</td><td> $\mathbf { 0 . 9 2 2 9 } \pm 0 . 0 1 0 0$ </td><td> $\mathbf { 0 . 8 7 6 9 \pm 0 . 0 0 4 2 }$ </td><td> $\mathbf { 0 . 9 3 7 7 } \pm 0 . 0 0 3 9$ </td><td> $\mathbf { 0 . 9 0 0 3 \pm 0 . 0 0 6 4 }$ </td><td> $0 . 9 4 0 4 \pm 0 . 0 0 2 4$ </td><td> $0 . 9 0 4 3 \pm 0 . 0 0 3 8$ </td></tr></table>

TABLE V  
LIDC-IDRI LINEAR PROBING WITH IDENTICAL BATCH SIZE.
<table><tr><td>Method</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>MoCo v2</td><td> $. 6 7 4 9 \pm . 0 0 3 9$ </td><td> $. 5 0 8 6 \pm . 0 0 1 2$ </td></tr><tr><td>SimSiam</td><td> $. 6 2 2 6 \pm . 0 0 9 2$ </td><td> $. 4 5 4 5 \pm . 0 1 6 7$ </td></tr><tr><td>Volume VICReg</td><td> $. 8 4 8 5 \pm . 0 0 9 2$ </td><td> $. 7 7 4 0 \pm . 0 1 0 9$ </td></tr><tr><td>L2GSSL</td><td> $. 8 8 7 5 \pm . 0 0 6 1$ </td><td> $\mathbf { \delta } . 8 1 1 1 \pm . 0 1 1 8$ </td></tr></table>

TABLE VI

DIRECT LNDB EVALUATION AFTER FINE-TUNING ON 25% OF LIDC-IDRI; NO LNDB ADAPTATION.
<table><tr><td>Method</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Supervised</td><td> $. 6 8 6 9 \pm . 0 5 5 3$ </td><td> $. 6 2 8 9 \pm . 0 6 3 2$ </td></tr><tr><td>MoCo v2</td><td> $. 6 7 1 7 \pm . 0 2 4 5$ </td><td> $. 6 2 1 5 \pm . 0 0 7 5$ </td></tr><tr><td>SimSiam</td><td> $. 6 3 1 3 \pm . 1 0 0 5$ </td><td> $. 5 5 5 1 \pm . 1 6 0 3$ </td></tr><tr><td>Volume VICReg</td><td> $. 7 0 0 8 \pm . 0 1 5 4$ </td><td> $. 6 6 3 5 \pm . 0 5 1 5$ </td></tr><tr><td>L2GSSL</td><td> ${ \bf 8 1 3 8 } \pm . 0 4 0 1$ </td><td> ${ \bf 7 5 8 9 } \pm { . 0 4 6 1 }$ </td></tr></table>

We use LIDC-IDRI for this comparison because it provides multiple radiologist malignancy ratings for individual nodules, which are required to construct a reference that excludes the radiologist being evaluated. LNDb does not provide comparable multi-reader coverage for each nodule.

For each nodule, one of its annotating radiologists is randomly selected as the held-out radiologist. Selection is performed per nodule because radiologist coverage differs across the test set and no single radiologist annotates every included nodule. The mean score of the remaining radiologists defines the reference label: means at or below 2.5 are benign, means at or above 3.5 are malignant, and intermediate means are excluded. We then define two evaluation conditions according to how we handle an uncertain score of 3 from the held-out radiologist. The uncertainty-excluded condition omits cases in which the held-out radiologist assigns a score of 3 and therefore evaluates only definite held-out judgments. The uncertainty-penalized condition retains such cases and counts the uncertain judgment as disagreement whenever the remaining radiologists provide a definite reference. After these rules are applied, the held-out judgments from all included nodules are pooled to compute a single sensitivity and falsepositive rate rather than averaging radiologist-specific metrics. This design prevents the selected radiologist from contributing to the label against which that radiologist is compared.

TABLE VII  
LIDC-IDRI LINEAR PROBING WITH THE MST-BASED METHODS CONSTRAINED TO A COMMON PEAK VRAM BUDGET. BEST RESULTS ARE BOLD.
<table><tr><td>Method</td><td>Peak VRAM (GB)</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>MoCo v2</td><td>47-48</td><td> $. 6 6 6 1 \pm . 0 1 0 4$ </td><td> $. 4 9 3 6 \pm . 0 0 3 7$ </td></tr><tr><td>SimSiam</td><td>47-48</td><td> $. 6 2 2 6 \pm . 0 0 9 2$ </td><td> $. 4 5 4 5 \pm . 0 1 6 7$ </td></tr><tr><td>Volume VICReg</td><td>47-48</td><td> $. 8 4 8 5 \pm . 0 0 9 2$ </td><td> $. 7 7 4 0 \pm . 0 1 0 9$ </td></tr><tr><td>L2GSSL</td><td>47-48</td><td> ${ \bf 8 5 9 9 } \pm { \bf 0 0 3 8 }$ </td><td> ${ \bf 7 9 2 6 } \pm . 0 0 3 9$ </td></tr></table>

The pooled radiologist point provides an operating reference in the same ROC space as the model curves by summarizing per-nodule held-out judgments across the evaluation set. The uncertainty-penalized condition is expected by construction to reduce the pooled radiologist point because a score of 3 is counted as disagreement against a definite reference. The more informative observation is that the L2GSSL ROC curves shift only modestly when these cases are retained in the ROC evaluation and remain close across runs. The analysis therefore provides performance context together with a sensitivity check on reader-uncertain cases.

## H. Design and Implementation Sensitivity

1) Component Ablation: We examine how the stop-gradient boundary and phased optimization contribute to L2GSSL behavior.

Removing the stop-gradient boundary gives the weakest discrimination results under both initialization strategies, supporting the role of gradient isolation in local feature learning (Table XI). The effect of phased optimization depends on initialization: the full configuration performs best from random initialization, whereas removing the phased schedule performs best from DINOv2 initialization. The schedule therefore acts as an initialization-dependent optimization mechanism rather than a universally beneficial component.

2) Projector Normalization Sensitivity: A separate sensitivity analysis examines whether the weaker MoCo v2 and SimSiam results arise from the absence of batch normalization in their projectors. They do not: with projector batch normalization, MoCo v2 records 0.6497 ± 0.0014 AUROC and $0 . 4 8 8 9 \pm 0 . 0 3 5 7 ~ \mathrm { A U P R C }$ , SimSiam records $0 . 5 7 3 1 \pm 0 . 0 0 0 8$ and $0 . 3 1 9 5 \pm 0 . 0 0 0 9$ , and L2GSSL records $0 . 7 6 2 3 \pm 0 . 0 0 7 1$ and $0 . 5 6 1 5 { \pm } 0 . 0 0 4 5$ . In slice-wise accumulation, normalization statistics are estimated within each micro-batch even though gradients approximate a larger effective batch. The mismatch produces noisy statistics and can degrade training; the main comparisons therefore retain the original normalization setting.

TABLE VIII  
COMPARISON WITH THE FULLY 3D VOCO REFERENCE ON LIDC-IDRI. LINEAR PROBING (LP), FINE-TUNING (FT), AND MEASURED PEAK VRAM ARE REPORTED FOR THE EVALUATED CONFIGURATIONS. THE COMPARISON IS NOT CONTROLLED FOR BACKBONE, INPUT FORMULATION, OR INITIALIZATION. BEST DISCRIMINATION RESULTS ARE BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Peak VRAM (GB)</td><td colspan="2">LP</td><td colspan="2">FT, 10% labels</td><td colspan="2">FT, 25% labels</td><td colspan="2">FT, 100% labels</td></tr><tr><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>VoCo (scratch)</td><td>27.57</td><td> $. 6 7 7 0 \pm . 0 0 8 5$ </td><td> $. 4 7 7 5 \pm . 0 1 9 3$ </td><td> $. 6 3 2 4 \pm . 0 6 6 9$ </td><td> $. 4 5 2 3 \pm . 0 7 3 4$ </td><td> $. 7 0 0 1 \pm . 0 2 9 0$ </td><td> $. 5 2 8 6 \pm . 0 4 3 6$ </td><td> $. 7 8 0 5 \pm . 0 1 6 6$ </td><td> $. 6 4 8 2 \pm . 0 0 1 0$ </td></tr><tr><td>VoCo (MONAI)</td><td>27.57</td><td> $. 6 9 0 5 \pm . 0 0 6 5$ </td><td>.5361 ± .0069</td><td> $. 6 5 0 8 \pm . 0 5 6 1$ </td><td> $. 4 6 4 1 \pm . 0 5 1 2$ </td><td> $. 7 0 5 8 \pm . 0 2 3 1$ </td><td>.5491 ± .0267</td><td> $. 7 6 5 4 \pm . 0 1 1 8$ </td><td> $. 6 1 8 2 \pm . 0 1 8 8$ </td></tr><tr><td>L2GSSL</td><td> $2 4 . 3 7 \mathrm { - } 2 4 . 4 8$ </td><td> $. 8 8 7 5 \pm . 0 0 6 1$ </td><td> $\mathbf { \delta } . 8 1 1 1 \pm . 0 1 1 8$ </td><td> $\mathbf { . 9 2 2 9 \pm . 0 1 0 0 }$ </td><td> ${ \bf . 8 7 6 9 \pm . 0 0 4 2 }$ </td><td> ${ \bf 9 3 7 7 \pm . 0 0 3 9 }$ </td><td> $\mathbf { . 9 0 0 3 \pm . 0 0 6 4 }$ </td><td> $\mathbf { . 9 4 0 4 \pm . 0 0 2 4 }$ </td><td> ${ \bf 9 0 4 3 } \pm . 0 0 3 8$ </td></tr></table>

![](images/9bb19a4d5dff5a35ce007838b3e69888c023e25fd4d042a293ebbb4c1e4e240b.jpg)  
(a) Uncertainty-excluded scoring.

![](images/f7fe52a4addd7bacff62a87f116283643929a9208c4991861ef6822373786943.jpg)  
(b) Uncertainty-penalized scoring.  
Fig. 2. Per-nodule held-out-radiologist ROC analysis under two treatments of an uncertain score from the selected radiologist. Each reference label is constructed from the remaining radiologists. The diamond shows the sensitivity and false-positive rate computed after pooling the held-out judgments across nodules; curves show L2GSSL runs.

TABLE IX  
LIDC-IDRI LINEAR PROBING WITH THE MAXVIT BACKBONE. BEST RESULTS ARE BOLD.
<table><tr><td>Method</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>MoCo v2</td><td> $. 6 9 9 9 \pm . 0 5 1 3$ </td><td> $. 6 8 4 2 \pm . 0 3 5 7$ </td></tr><tr><td>SimSiam</td><td> $. 7 5 1 0 \pm . 0 0 5 4$ </td><td> $. 7 2 5 3 \pm . 0 0 5 7$ </td></tr><tr><td>Volume VICReg</td><td> $. 7 5 6 7 \pm . 0 5 6 2$ </td><td> $. 7 4 0 0 \pm . 0 5 4 2$ </td></tr><tr><td>L2GSSL</td><td> $\mathbf { 8 9 1 9 } \pm . 0 0 5 1$ </td><td> ${ \bf 8 8 5 0 } \pm { \bf 0 0 3 4 }$ </td></tr></table>

LINEAR-PROBING COMPONENT ABLATION ON LIDC-IDRI. RESULTS ARE MEAN ± STANDARD DEVIATION; THE BEST RESULT WITHIN EACH INITIALIZATION IS BOLD.

TABLE X  
EFFECT OF MASKING MALIGNANT LESION-BEARING SLICES.
<table><tr><td>Condition</td><td>Malignant TPR</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>No mask</td><td>87.84%</td><td>0.9404</td><td>0.9043</td></tr><tr><td>Tumor-slice mask</td><td>14.57%</td><td>0.6169</td><td>0.4307</td></tr><tr><td>Change</td><td>-73.27%</td><td>-0.3235</td><td>-0.4736</td></tr></table>

TABLE XI
<table><tr><td></td><td>Initialization Configuration</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Random</td><td>Full L2GSSL</td><td> $0 . 5 9 6 8 \pm 0 . 0 0 1 1$ </td><td> $0 . 3 9 3 8 \pm 0 . 0 0 2 6$ </td></tr><tr><td>Random</td><td>Without stop-gradient</td><td> $0 . 5 7 6 7 \pm 0 . 0 2 0 2$ </td><td> $0 . 3 7 5 7 \pm 0 . 0 0 5 4$ </td></tr><tr><td>Random</td><td>Without phased schedule</td><td> $0 . 5 8 7 7 \pm 0 . 0 1 9 1$ </td><td> $0 . 3 8 0 1 \pm 0 . 0 1 8 3$ </td></tr><tr><td>DINOv2</td><td>Full L2GSSL</td><td> $0 . 8 5 9 9 \pm 0 . 0 0 3 8$ </td><td> $0 . 7 9 2 6 \pm 0 . 0 0 3 9$ </td></tr><tr><td>DINOv2</td><td>Without stop-gradient</td><td> $0 . 5 7 4 9 \pm 0 . 0 2 9 9$ </td><td> $0 . 3 8 0 6 \pm 0 . 0 2 0 7$ </td></tr><tr><td>DINOv2</td><td>Without phased schedule</td><td> $0 . 8 9 6 7 \pm 0 . 0 0 2 8$ </td><td> $0 . 8 2 6 0 \pm 0 . 0 0 6 0$ </td></tr></table>

## V. DISCUSSION

## A. Principal Findings

The results show that separating slice- and volume-level self supervised objectives improves label-efficient learning from volumetric CT data. L2GSSL provides its clearest gains when only 10% or 25% of the LIDC-IDRI labels are available, while the advantage narrows under full-label fine-tuning. This pattern is consistent with the intended role of self-supervised pretraining: using unlabeled scans to reduce dependence on curated annotations. Direct evaluation on LNDb further tests whether the learned representation remains useful under a change in data source and annotation practice, while the abdominal-trauma task extends the evaluation across anatomy and prediction target.

One interpretation of the label-fraction trend follows from the two representation levels. The local objective supplies viewstable slice features before downstream labels are introduced, while the global objective organizes those features across the ordered volume. With fewer labels, this initialization reduces how much slice appearance and inter-slice structure must be learned from supervised examples alone. Full-label fine-tuning supplies more task-specific supervision to all trainable components and can therefore narrow differences among initializations. This interpretation is consistent with the observed trend and with the component analysis of gradient isolation and scheduling.

The resource experiments address a separate practical consideration: whether hierarchical volumetric learning remains feasible under constrained GPU memory. The evaluated results show that slice-to-volume learning can provide useful representations within the measured memory budgets. This property is relevant to volumetric healthcare archives, where scans are high-dimensional and vary in slice count even when labels remain limited.

The component results clarify how the optimization design contributes to this behavior. Under DINOv2 initialization, removing gradient detachment produces the weakest variant, whereas removing phased optimization can improve performance. With random initialization, the full phased configuration performs best, indicating that the schedule is most useful before a stable slice representation has been established. The ablation therefore supports controlled local-to-global optimization while showing that the preferred schedule depends on initialization.

## B. Implications for Healthcare Data Processing

The intended use of L2GSSL is representation pretraining, not autonomous diagnosis. Within that scope, its modularity offers three practical advantages. First, unlabeled scans can contribute to adaptation before a site invests in outcome curation. This matters because routine archives are far larger than the subsets with consistent malignancy, injury, or segmentation labels. Second, slice-wise execution decouples storage and optimization choices: images can be decoded in groups, local features can be accumulated, and the inter-slice model can operate on a sequence of embeddings. This design fits healthcare archives in which complete volumes vary in slice count and cannot all be staged as dense tensors at once. Third, the method retains a recognizable boundary between image appearance and volume context, which provides intervention points for auditing local features, sequence aggregation, and final predictions separately.

The framework does not eliminate the governance require ments associated with such archives. Site-specific secondary pretraining could absorb scanner signatures, reconstruction kernels, or population correlates that are unrelated to pathology.

De-identification does not prevent these shortcuts, and SSL provides no label-based mechanism for detecting them during pretraining. A deployment pipeline should therefore retain provenance for every pretrained scan, document inclusion and exclusion rules, stratify evaluation by acquisition source where metadata permit, and keep test institutions outside all pretraining and hyperparameter selection. The patient-level split used here prevents within-dataset leakage, and the untouched LNDb test provides an initial distribution-shift check, but neither substitutes for institutional governance.

## C. Clinical Interpretation of the Evaluation

The low-label experiment approximates a development setting with few curated nodules, but a label fraction is not equivalent to a clinical annotation budget. The selected samples may not capture the spectrum, prevalence, or disagreement encountered during prospective collection. Similarly, AUROC measures ranking across thresholds, whereas an actual decision support tool operates at a chosen threshold with asymmetric consequences. AUPRC is included because malignant nodules are the positive class and class imbalance can make ROC performance appear optimistic, but neither measure specifies whether missed malignancies or false alarms are acceptable. Decision-curve analysis and task-specific utility were not performed.

The radiologist analysis provides a pooled operating reference for contextualizing the model ROC curves and tests whether model ranking remains stable when reader-uncertain cases are retained in the ROC evaluation. It is a retrospective analysis of existing per-nodule judgments; the radiologists did not interact with the model. Likewise, the LNDb experiment evaluates technical generalization rather than clinical transportability. A prospective study would need prespecified endpoints, locked preprocessing and thresholds, workflow timing, failure handling, and analysis of whether assistance changes radiologist decisions. These distinctions are important for a healthcaredata venue because a model can transfer statistically without improving care.

## D. Limitations

The primary MST comparisons use the same patient-level split, backbone, and training pipeline to isolate differences among the evaluated pretraining objectives. VoCo, however, operates directly on dense 3D crops and cannot be mapped onto the slice-encoder and inter-slice aggregation structure without changing its original formulation. We therefore retain VoCo as a separate fully 3D system-level reference under the evaluated resource settings. Because VoCo and L2GSSL differ in backbone, input formulation, and initialization, this comparison evaluates the complete systems rather than isolating the effect of either objective.

The evaluation is limited to retrospective CT classification. LIDC-IDRI and LNDb provide variation in data source and reader protocol, and the RSNA task extends the evaluation to abdominal anatomy and multiple institutions, but these datasets do not cover prospective deployment, other imaging modalities, or demographic subgroup performance. The resource analysis measures peak VRAM rather than preprocessing throughput, storage, or training time. Broader claims would require evaluation on additional tasks, modalities, institutions, and prospective clinical workflows.

The lesion-masking experiment specifically examines whether model performance depends on lesion-bearing slices. The observed performance drop provides evidence that these slices contribute to the prediction, but their removal also changes the input distribution; the experiment therefore does not establish causal localization or exclude reliance on other image features.

## VI. CONCLUSION

L2GSSL addresses a central healthcare-data imbalance: high dimensional CT volumes are widely available, while expertcurated labels remain limited. By decomposing self-supervised volumetric learning into local slice and global inter-slice objectives, the framework extracts reusable representations from unlabeled scans with controlled gradient flow and phased optimization. Its clearest gains occur with 10% and 25% of LIDC-IDRI labels and diminish under full supervision, matching the intended role of self-supervised pretraining. Direct LNDb transfer, two backbone designs, and abdominaltrauma CT provide evidence across data sources, architectures, anatomy, and tasks, while slice-wise execution supports training under constrained GPU memory. Together, these results position L2GSSL as a practical foundation for label-efficient and resource-conscious representation learning from volumetric healthcare data.

## ACKNOWLEDGMENT

This work was supported by the National Science and Technology Council of Taiwan under Grant 113-2221-E-008- 100-MY3. Computational and storage resources were provided by the National Center for High-performance Computing of National Applied Research Laboratories, Taiwan. The authors used LLMs (e.g., ChatGPT) to improve language and readability. The authors reviewed and edited the content as needed. The authors take full responsibility for the content.

## REFERENCES

[1] L. Chen, P. Bentley, K. Mori, K. Misawa, M. Fujiwara, and D. Rueckert, “Self-supervised learning for medical image analysis using image context restoration,” Medical Image Analysis, vol. 58, p. 101539, 2019.

[2] Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, “Models genesis: Generic autodidactic models for 3d medical image analysis,” Medical Image Analysis, vol. 67, p. 101840, 2021.

[3] L. Wu, J. Zhuang, and H. Chen, “Voco: A simple-yet-effective volume contrastive learning framework for 3d medical image analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 873–22 882.

[4] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance-invariancecovariance regularization for self-supervised learning,” in International Conference on Learning Representations, 2022.

[5] S. G. Armato et al., “The lung image database consortium (LIDC) and image database resource initiative (IDRI): A completed reference database of lung nodules on CT scans,” Medical Physics, vol. 38, no. 2, pp. 915–931, 2011.

[6] J. Pedrosa et al., “LNDb: A lung nodule database on computed tomography,” 2019.

[7] S. Hermans et al., “RSNA 2023 abdominal trauma AI challenge: Review and outcomes,” Radiology: Artificial Intelligence, vol. 7, no. 1, p. e240334, 2025.

[8] J. Ma et al., “AbdomenCT-1k: Is abdominal organ segmentation a solved problem?” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6695–6714, 2022.

[9] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3d u-net: Learning dense volumetric segmentation from sparse annotation,” in Medical Image Computing and Computer-Assisted Intervention, 2016, pp. 424–432.

[10] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in Fourth International Conference on 3D Vision, 2016, pp. 565–571.

[11] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “UNETR: Transformers for 3d medical image segmentation,” in IEEE Winter Conference on Applications of Computer Vision, 2022, pp. 574–584.

[12] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[13] G. Muller-Franzes et al., “Medical slice transformer for improved diagnosis and explainability on 3d medical images with DINOv2,” Scientific Reports, vol. 15, no. 1, p. 23979, 2025.

[14] Z. Tu et al., “Maxvit: Multi-axis vision transformer,” in European Conference on Computer Vision, 2022, pp. 459–479.

[15] P.-K. Chiu and H.-H. Chen, “From volumes to slices: Computationally efficient contrastive learning for sequential abdominal CT analysis,” in IEEE International Symposium on Biomedical Imaging, 2026, pp. 1–4.

[16] X. Chen and K. He, “Exploring simple siamese representation learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15 750–15 758.

[17] X. Chen, H. Fan, R. Girshick, and K. He, “Improved baselines with momentum contrastive learning,” arXiv preprint arXiv:2003.04297, 2020.

[18] D. Y. Wu, D.-N. Lin, V. F. Chen, and H.-H. Chen, “Associated learning: A methodology to decompose end-to-end backpropagation on CNN, RNN, and transformer,” in International Conference on Learning Representations, 2022. [Online]. Available: https: //openreview.net/forum?id=4N-17dske79

[19] Z.-H. Huang, Y.-T. Lin, and H.-H. Chen, “DeInfoReg: A decoupled learning framework with information regularization for better training throughput,” Neurocomputing, vol. 651, p. 130813, 2025.

[20] M.-Y. Ho, C.-K. Wang, Y.-T. Lin, and H.-H. Chen, “Scpl: Enhancing neural network training throughput with decoupled local losses and model parallelism,” ACM Trans. Manage. Inf. Syst., vol. 17, no. 2, Jun. 2026.

[21] A. S. Tejani et al., “Checklist for artificial intelligence in medical imaging (CLAIM): 2024 update,” Radiology: Artificial Intelligence, vol. 6, no. 4, p. e240300, 2024.