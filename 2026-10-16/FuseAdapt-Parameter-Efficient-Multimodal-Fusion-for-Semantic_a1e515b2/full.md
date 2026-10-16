# FuseAdapt: Parameter-Efficient Multimodal Fusion for Semantic Segmentation with Missing Modalities

Xin Zhang National University of Singapore x.zhang@u.nus.edu

Robby T. Tan National University of Singapore robby.tan@nus.edu.sg

## Abstract

Multi-modal semantic segmentation benefits from complementary sensors, but this benefit is fragile when some modalities are degraded or missing at test time. Many existing methods rely on dedicated fusion modules or multi-branch cross-modal encoders, and the fusion often becomes unreliable when modalities are absent, leading to substantial performance drops. We propose FuseAdapt, a parameterefficient framework that performs fusion within the adaptation space of a frozen vision foundation model (VFM), exploring whether VFM priors can serve as an effective shared backbone for multi-modal segmentation. Instead of attaching a standalone fusion network, FuseAdapt assigns each modality a lightweight adaptation branch and performs adaptation-space fusion by keeping Q/V lowrank adaptation private to each modality and routing only FFN low-rank residual activations across modalities inside the frozen backbone. The pretrained VFM remains fixed while modality-specific corrections and explicit cross-modal interaction are learned through the adaptation paths. To handle missing modalities, we further introduce Predictive Modality Modeling (PMM), a direct representationlevel objective for cross-modal redundancy. During training, PMM masks a target modality and predicts its embedding from the visible ones, supervised by a teacher target computed under full-modality input. PMM is used only during training and adds no inference-time overhead. Across three multi-modal segmentation benchmarks (MCubeS, DELIVER, MUSES), FuseAdapt consistently improves both modality-complete and modality-incomplete segmentation with a small trainable budget, yielding up to +3.01 mIoU under full modalities and +12.56 mIoU under missing modalities over prior state-of-the-art. Project page: https://devinxzhang.github.io/FuseAdapt-Page/.

## 1 Introduction

Scene understanding often benefits from combining multiple sensing modalities [Huang et al., 2022, Jia et al., 2023, Park et al., 2017, Zhang et al., 2018]. RGB [Wei et al., 2024, Zhang and Tan, 2025, Zhang et al., 2026, 2024a, Li et al., 2025a, Yuan et al., 2026a, Zhou et al., 2026, Zhang et al., 2024b, Zhang and Chen, 2023], depth [Cao et al., 2021, Song et al., 2022, Ying and Chuah, 2022], LiDAR [Yan et al., 2022, Wang et al., 2022, Li et al., 2023], and event streams [Xia et al., 2023, Zhang et al., 2021a, Li et al., 2025b] provide complementary cues about appearance, geometry, and motion. In practice, however, this multi-sensor benefit is fragile: a modality may be noisy [Yang et al., 2026a,b], degraded, spatially imperfect, or unavailable at inference time. Such cases are common in autonomous driving [Caesar et al., 2020, Li et al., 2026] and embodied settings [Wang et al., 2024], where illumination, weather [Du et al., 2026, Teng et al., 2025, Yang et al., 2025], texture, sensor failure, or calibration conditions [Yuan et al., 2025a] can affect sensors unequally. A practical multi-modal segmentation model should therefore work in two settings: modality-complete semantic segmentation (MCSS), where all modalities are available and should be fused effectively, and modality-incomplete semantic segmentation (MISS), where only a subset is available and the model should remain reliable.

![](images/25dc1d219baa9e5c4f18756ced6c8911f5746f357d751ec6ad4e7aa60d7fab06.jpg)

![](images/67f31edc30401e470fc3e68c95b590d66e234af1f5815681e5e9c7ec198c28dd.jpg)  
Figure 1: Comparison of multi-modal fusion strategies. (a) Separate encoders with mid-level fusion can replicate backbone computation across modalities. (b) Shared-encoder or late-fusion designs reduce this cost, but can make fusion depend strongly on the dominant or cleanest modality. (c) FuseAdapt keeps the VFM frozen and performs cross-modal exchange through trainable adaptation paths. Right: MCSS on MCubeS, where FuseAdapt improves mIoU across modality combinations.

Existing multi-modal segmentation methods often trade off modality-specific modeling, compactness, and robustness. Separate-encoder designs fuse intermediate features from different modalities [Broe dermann et al., 2023, Zhang et al., 2023a], but can increase parameters and computation as the number of sensors grows. RGB-centric methods treat RGB as the main input and use other modalities as auxiliary cues [Zhang et al., 2023b], which can be limiting when RGB is degraded or absent. Shared-encoder and late-fusion methods improve compactness [Zheng et al., 2024c,b], but their shared stream can favor the most informative or easiest modality, making the contribution of other sensors less stable. For missing-modality robustness, recent methods go beyond modality dropout by using text anchors [Zheng et al., 2024b], subset consistency [Zheng et al., 2024c,a], or modality regularization [Zheng et al., 2025]. These strategies provide indirect supervision for dense prediction: text anchors are coarse, and subset consistency can be affected by cross-modal mismatch.

We study how to separate sensor-specific adaptation from cross-sensor communication within a shared frozen backbone. FuseAdapt keeps modality-specific Q/V low-rank residuals private and restricts receiver-conditioned, token-wise exchange to FFN low-rank residual activations (Fig. 1(c)). The frozen dense FFN transformation stays on the receiving stream, while routing is normalized over the available sensors. We call this constraint on the trainable communication path adaptation-space fusion. The exchanged tensors remain intermediate features; the contribution is the separation of private attention adaptation and cross-modal residual communication, rather than a new tensor domain or the individual use of LoRA and routing. A shared VFM supplies the pretrained representation, and controlled experiments with MiT backbones test whether the benefit depends on that representation.

To handle missing modalities, we introduce Predictive Modality Modeling (PMM) (Fig. 2(c)). PMM masks a target modality during training and predicts its representation from the visible modalities, supervised by a teacher target computed under full-modality input. PMM applies the joint-embedding predictive principle at modality level: target-specific queries predict an unobserved sensor’s latent representation from visible context. Its EMA teacher supplies a temporally smoothed, stop-gradient target. After projection onto a common token grid, each query can attend to all visible positions rather than imposing same-index visible-to-target regression. This does not remove the common-grid requirement of routing and averaging. The teacher and predictor are used only during training.

We evaluate FuseAdapt on MCubeS [Liang et al., 2022], DELIVER [Zhang et al., 2023b], and MUSES [Brödermann et al., 2024] under both MCSS and MISS protocols. FuseAdapt yields up to +3.01 mIoU under full modalities and +12.56 mIoU under missing modalities over prior state-of-theart, with ablations isolating the contributions of modality-specific adaptation, routed cross-modal exchange, PMM, and the choice of frozen VFM. Our contributions are:

• FuseAdapt separates modality-private Q/V adaptation from receiver-conditioned crossmodal exchange restricted to FFN low-rank residual activations, with routing adjusted to the visible sensor subset.

![](images/df2b4c96ac6860693fd241dadc01eb9d39c4e0c43630c64da027f6e15afbeffb.jpg)  
Figure 2: Comparison of designs for robustness under missing modalities (MISS). (a) Semanticanchor methods align modality features to text embeddings, but text anchors can be coarse for pixel-level segmentation. (b) Semantic-consistency methods enforce agreement across different modality subsets, but such constraints can be affected by cross-modal mismatch. (c) Predictive Modality Modeling (PMM) casts MISS training as conditional representation prediction. Learnable query tokens aggregate cues from visible modalities to predict a missing-modality embedding, improving MISS performance on DELIVER and MUSES.

• PMM trains missing-modality robustness through target-conditioned latent prediction. Controls distinguish its prediction target from output distillation and generic fused-feature matching.

• Experiments on three benchmarks, same-backbone PEFT and information-flow controls support the design. We additionally characterize its inference cost and the scope of its robustness.

## 2 Related Work

Multi-Modal Semantic Segmentation Multi-modal semantic segmentation uses complementary RGB [Wei et al., 2024, Zhang and Tan, 2025, Qian et al., 2024, Zhang et al., 2022, Deng et al., 2020], depth [Wang et al., 2020, Ying and Chuah, 2022, Song et al., 2022, Cao et al., 2021, Park et al., 2017], LiDAR [Li et al., 2022, 2023, Zhuang et al., 2021, Yan et al., 2022, Wang et al., 2022], thermal [Hui et al., 2023, Zhang et al., 2023c, 2021b, Wu et al., 2022], and event cues [Xia et al., 2023, Zhang et al., 2021a, Jia et al., 2023]. Prior work often uses fixed modality pairs such as RGB-depth [Ying and Chuah, 2022, Song et al., 2022], RGB-centric fusion [Zhang et al., 2023b], or shared-backbone fusion with auxiliary alignment [Zheng et al., 2024b,c,a]. In contrast, FuseAdapt keeps a frozen VFM and performs cross-modal exchange through trainable adaptation paths, avoiding a standalone fusion network.

Missing or Degraded Modalities Existing approaches use modality dropout, cross-modal distillation, and consistency regularization [Liu et al., 2024, Wang et al., 2023, Zheng et al., 2024b,c,a]. CRM [Shin et al., 2024] combines complementary RGB-thermal masking with clean-to-masked self-distillation; M3L [Maheshwari et al., 2024] uses a full-modality teacher to supervise maskedmodality segmentation. PMM instead conditions on a designated missing sensor and predicts its full-context latent representation. This follows the latent-context prediction principle of I-JEPA [Ass ran et al., 2023], applied to modalities rather than image regions. Masking, EMA, and teacher–student optimization are established mechanisms; the prediction target and modality conditioning distinguish our objective.

Parameter-Efficient Adaptation and Fusion Prompt tuning, adapters, and LoRA reduce the cost of adapting pretrained models [Lester et al., 2021, Jia et al., 2022, Houlsby et al., 2019, Chen et al., 2022, Hu et al., 2022, Yan et al., 2025]. StitchFusion [Li et al., 2024] transfers dense features between encoder streams through MultiAdapters. MLE-SAM [Zhu et al., 2024] combines modalityspecific LoRA experts and routing with multi-scale feature/head fusion. MoLE [Wu et al., 2024] composes task/concept LoRAs within a single stream. Concurrent M4-SAM [Liu et al., 2026] combines modality-dispatched convolutional experts with gated feature fusion for RGB-D video, while MILE [Muralidhara et al., 2026] selects isolated continual-task experts. FuseAdapt instead keeps Q/V residuals private and restricts directed source-to-receiver communication to FFN low-rank residuals among currently visible sensors. Section 4.4 tests this constraint against routing all low-rank paths or complete FFN features.

![](images/7c1fc4403a366b59502c8eee9cfdc47d05e53b00d4ebec4edae02522c4efdc49.jpg)  
Figure 3: Overview of our FuseAdapt framework. (a) Adaptation-Space Fusion: Multi-modal inputs (e.g., RGB, depth, event, LiDAR) share a frozen VFM backbone. Each modality attaches a lightweight adaptation branch inside the transformer blocks, and a receiver-conditioned router mixes only FFN low-rank residual activations. Fusion is thus performed inside the adaptation path while keeping the backbone frozen, and the fused features are sent to a shared segmentation head. For clarity, we show the routing for one modality; the others follow the same design. (b) PMM: To handle missing modalities, an EMA teacher provides the target embedding for a selected modality. A student and a Perceiver-style predictor use the visible modalities to predict this embedding via learnable queries. PMM is trained with a distance loss $D ( \hat { z } , z )$

## 3 Proposed Method

We address robust multi-modal semantic segmentation in two regimes: modality-complete semantic segmentation (MCSS), where all modalities are available, and modality-incomplete semantic segmentation (MISS), where only a subset is available at test time. Our goal is to adapt a frozen vision foundation model (VFM) into a multi-modal learner that can fuse available modalities with a small trainable budget and learn redundancy for missing-modality robustness.

As shown in Fig. 3, the input contains four aligned image-like modalities of the same scene: RGB $\mathbf { R } \in$ $\mathbb { R } ^ { h \times w \times 3 }$ , depth $\mathbf { D } \in \mathbb { R } ^ { h \times w \times C ^ { D } }$ , LiDAR projection $\mathbf { L } \in \mathbb { R } ^ { h \times w \times C ^ { L } }$ , and event stack $\mathbf { E } \in \mathbb { R } ^ { h \times w \times C ^ { E } }$ with pixel labels Y over K classes. Following [Zhang et al., 2023b], we set $C ^ { D } = C ^ { L } = C ^ { E } = 3$ and feed all modalities as image-like inputs to a shared frozen VFM.

FuseAdapt has two components. First, an adaptation-spacefusion design (§3.1) performs fusion in the adaptation space of the frozen VFM. Each modality has lightweight trainable adaptation paths, and cross-modal exchange is implemented by routing FFN low-rank residual activations across modalities inside Transformer blocks. The pretrained backbone remains fixed. Second, Predictive Modality Modeling (PMM) improves MISS by training the model to predict a masked modality representation from visible modalities, using a teacher target computed with full-modality context. PMM is used only during training and discarded at inference.

## 3.1 Adaptation-Space Fusion

We realize adaptation-space fusion through modality-specific low-rank adaptation paths inside the frozen VFM, with cross-modal exchange implemented by routed FFN low-rank residual activations. We keep self-attention adaptation modality-specific to preserve sensor-specific correction, and place cross-modal exchange in the FFN adaptation path.

Self-Attention Adaptation Let $\boldsymbol { \mathcal { M } } = \{ \boldsymbol { r } , d , \boldsymbol { e } , l \}$ denote the modality set. At a Transformer block, let $\mathbf { X } ^ { m } \in \mathbb { R } ^ { N \times d }$ be the token features of modality $m$ , where all modalities have the same projected token count $N$ at this block and $d$ is the hidden dimension. Routing and averaging require this common grid; arbitrary native token lengths are not supported. We omit the block index and bias terms for clarity. For each modality, we add trainable low-rank updates only to the query and value

projections:

$$
\mathbf { Q } ^ { m } = \mathbf { X } ^ { m } ( \mathbf { W } _ { Q } + \Delta \mathbf { W } _ { Q } ^ { m } ) , \quad \mathbf { V } ^ { m } = \mathbf { X } ^ { m } ( \mathbf { W } _ { V } + \Delta \mathbf { W } _ { V } ^ { m } ) ,\tag{1}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d \times d }$ are frozen pretrained weights. The updates are parameterized as

$$
\Delta \mathbf { W } _ { Q } ^ { m } = \mathbf { A } _ { Q } ^ { m } \mathbf { B } _ { Q } ^ { m } , \quad \Delta \mathbf { W } _ { V } ^ { m } = \mathbf { A } _ { V } ^ { m } \mathbf { B } _ { V } ^ { m } ,\tag{2}
$$

where $\mathbf { A } _ { Q } ^ { m } , \mathbf { A } _ { V } ^ { m } \in \mathbb { R } ^ { d \times r } , \mathbf { B } _ { Q } ^ { m } , \mathbf { B } _ { V } ^ { m } \in \mathbb { R } ^ { r \times d }$ , and $r \ll d .$ . The key and output projections, $\mathbf { W } _ { K }$ and $\mathbf { W } _ { O }$ , remain frozen and shared. The attention output is

$$
\mathbf { M H S A } ^ { m } ( \mathbf { X } ^ { m } ) = \operatorname { s o f t m a x } \left( \frac { \mathbf { Q } ^ { m } ( \mathbf { X } ^ { m } \mathbf { W } _ { K } ) ^ { \top } } { \sqrt { d } } \right) \mathbf { V } ^ { m } \mathbf { W } _ { O } .\tag{3}
$$

This keeps attention adaptation separated by modality while preserving the pretrained attention structure.

Feedforward Routing Fusion Cross-modal exchange is introduced in the FFN adaptation path. Let $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times l }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { l \times d }$ be the frozen FFN weights. Each modality has trainable low-rank FFN updates:

$$
\Delta \mathbf { W } _ { 1 } ^ { m } = \mathbf { A } _ { 1 } ^ { m } \mathbf { B } _ { 1 } ^ { m } , \quad \Delta \mathbf { W } _ { 2 } ^ { m } = \mathbf { A } _ { 2 } ^ { m } \mathbf { B } _ { 2 } ^ { m } ,\tag{4}
$$

where $\mathbf { A } _ { 1 } ^ { m } \in \mathbb { R } ^ { d \times r } , \mathbf { B } _ { 1 } ^ { m } \in \mathbb { R } ^ { r \times l } , \mathbf { A } _ { 2 } ^ { m } \in \mathbb { R } ^ { l \times r } , \mathbf { B } _ { 2 } ^ { m } \in \mathbb { R } ^ { r \times d }$ , and $r \ll \operatorname* { m i n } ( d , l )$

Let $\nu \subseteq { \mathcal { M } }$ be the nonempty available subset. For receiving modality $m \in \mathcal { V }$ and token $i ,$ router logits $g ^ { m } ( \mathbf { X } ^ { m } ) _ { : }$ <sub>i</sub> define

$$
\alpha _ { i j } ^ { m } = \frac { \exp ( g ^ { m } ( \mathbf { X } ^ { m } ) _ { i j } ) } { \sum _ { k \in \mathcal { V } } \exp ( g ^ { m } ( \mathbf { X } ^ { m } ) _ { i k } ) } , \quad j \in \mathcal { V } , \qquad \alpha _ { i j } ^ { m } = 0 \quad ( j \notin \mathcal { V } ) .\tag{5}
$$

Thus unavailable sources are masked before normalization. Omitting block indices, biases, and dropout, the FFN becomes

$$
\mathbf { H } ^ { m } = \sigma \left( \mathbf { X } ^ { m } \mathbf { W } _ { 1 } + \sum _ { j \in \mathcal { V } } \alpha _ { j } ^ { m } \odot ( \mathbf { X } ^ { j } \Delta \mathbf { W } _ { 1 } ^ { j } ) \right) ,\tag{6}
$$

$$
\mathbf { F } ^ { m } = \mathbf { H } ^ { m } \mathbf { W } _ { 2 } + \sum _ { j \in \mathcal { V } } \alpha _ { j } ^ { m } \odot ( \mathbf { H } ^ { j } \Delta \mathbf { W } _ { 2 } ^ { j } ) ,\tag{7}
$$

where $\sigma$ is the FFN activation and ⊙ broadcasts token weights over channels. The routed objects are residual activations, not parameter matrices or complete FFN features. Dense transformations operate on the receiving stream; their weights remain frozen even though stream activations evolve through successive blocks. After the final block, we symmetrically aggregate visible outputs,

$$
\hat { \mathbf { X } } _ { \nu } = \frac { 1 } { | \nu | } \sum _ { m \in \nu } \hat { \mathbf { X } } ^ { m } ,\tag{8}
$$

and pass them to the shared segmentation decoder. The same communication rule applies to full and partial inputs.

## 3.2 Predictive Modality Modeling

While adaptation-space fusion handles full-modality inputs, MISS also requires the model to behave reliably when some modalities are absent. PMM addresses this by adding a representation-level prediction objective during training. Given visible modalities, the model predicts the representation of a masked target modality, supervised by a teacher target computed with full-modality context. This encourages cross-modal redundancy beyond segmentation supervision alone.

After the full-input warm-up, training mixes full and masked inputs. In the masked branch, modalities are independently retained with probability 0.5, conditioned on a nonempty visible subset V. When $\nu \neq M$ , we sample a target $t \in \mathcal { M } \backslash$ V uniformly and aggregate visible features $\hat { \mathbf { X } } ^ { m } \in \mathbb { R } ^ { N \times C }$ as above. When all modalities are visible, segmentation supervision remains active and no missing-target loss is computed. Appendix E.3 gives the masking procedure.

The teacher target $y _ { t } \in \mathbb { R } ^ { N \times C }$ for the masked modality t is produced by an EMA teacher that takes the full-modality input. The teacher shares the frozen VFM with the student (no duplication of backbone weights) and maintains an EMA copy only of the adaptation branches. Letting $\theta _ { s }$ denote the student FuseAdapt parameters and $\theta _ { \tau }$ those of the teacher, the EMA update is

Table 1: Results in MCSS setting on MCubeS. The best results are highlighted in bold. (R: RGB; A: AoLP; D: DoLP; N: NIR.)
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">|#Trainable| [Params (M)</td><td>Modals</td><td></td><td> $\mathbf { \underline { { { l } } } _ { M e a n } }$ </td></tr><tr><td>RA</td><td>RAD RADN|</td><td></td></tr><tr><td>MCubeSNet [Liang et al., 2022]</td><td>MiT-B4</td><td>65.65</td><td>|39.10 42.00</td><td>42.86</td><td>|41.32</td></tr><tr><td>CWSAM [Pu et al., 2025]</td><td>ViT-B</td><td>15.29</td><td>49.7848.27</td><td>50.59</td><td>49.55</td></tr><tr><td>MLE-SAM [Zhu et al., 2024]</td><td>Hiera-B+</td><td>20.79</td><td>50.61 50.89</td><td>51.02</td><td>50.84</td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B2</td><td>58.73</td><td>48.82 49.4851.54</td><td></td><td>49.95</td></tr><tr><td>Any2Seg [Zheng et al., 2024b]</td><td>MiT-B2</td><td>25.79</td><td>48.95 50.6751.91</td><td></td><td>50.51</td></tr><tr><td>MemorySAM [Liao et al., 2025]</td><td>Hiera-B+</td><td>14.50</td><td>51.20 52.20</td><td>52.88</td><td>52.09</td></tr><tr><td>MAGIĆ [Zheng et al., 2024c]</td><td>MiT-B2</td><td>24.73</td><td>51.45 52.83</td><td>53.01</td><td>52.43</td></tr><tr><td>StitchFusion [Li et al., 2024]</td><td>MiT-B4</td><td>68.02</td><td>52.68 53.2653.92</td><td></td><td>53.29</td></tr><tr><td>Ours (wo FuseAdapt)</td><td>ViT-B</td><td>0.00</td><td>49.15 49.9851.16</td><td></td><td>|50.10</td></tr><tr><td>Ours (shared FuseĀdapt)</td><td>ViT-B</td><td>3.50</td><td>54.87 54.85 54.69</td><td></td><td>54.80</td></tr><tr><td>Ours (separate FuseAdapt)</td><td>ViT-B</td><td>13.86</td><td>54.87 55.26 56.09</td><td></td><td>55.41</td></tr><tr><td>Ours</td><td>ViT-B</td><td>14.01</td><td>54.87 56.41– 57.62</td><td></td><td>56.30</td></tr></table>

$$
\theta _ { \tau }  \mu \theta _ { \tau } + ( 1 - \mu ) \theta _ { s } , \quad \quad \mu = 0 . 9 9 9 ,\tag{9}
$$

applied at every iteration. The target is stop-gradient. Full context is a useful training prior on average, not a guarantee of superior predictions for every sample, especially when a present sensor is corrupted (Appendix B.1).

Query-Based Prediction A per-token linear regressor imposes same-index visible-to-target correspondence on the common grid. We instead use modality-specific learnable queries $Q _ { t } \in \mathbb { R } ^ { N \times C }$ each of which attends globally to $\hat { \mathbf { X } } _ { \mathcal { V } }$ through a Perceiver-style predictor:

$$
\hat { z } _ { t } = g _ { \phi } ( Q _ { t } , \hat { \mathbf { X } } _ { \mathcal { V } } ) , \qquad \hat { z } _ { t } \in \mathbb { R } ^ { N \times C } .\tag{10}
$$

This permits retrieval from any visible position while retaining a common target grid. It does not natively align different fields of view or temporal offsets. The PMM loss is a token-wise cosine distance:

$$
\mathcal { L } _ { \mathrm { p m m } } = 1 - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } D \big ( \hat { z } _ { t } ^ { ( i ) } , \mathrm { s g } [ y _ { t } ^ { ( i ) } ] \big ) ,\tag{11}
$$

where $D ( \cdot , \cdot )$ denotes cosine similarity. The full training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \mathcal { L } _ { \mathrm { p m m } } . } \end{array}\tag{12}
$$

At inference time, PMM is removed and the model uses the available modalities directly.

## 4 Experiments

Datasets We evaluate on three multi-modal segmentation benchmarks covering both modalitycomplete semantic segmentation (MCSS) and modality-incomplete semantic segmentation (MISS). DELIVER [Zhang et al., 2023b] contains synchronized RGB, depth, LiDAR, and event inputs for 25-class urban-scene segmentation, with 3,983/2,005/1,897 train/val/test samples and diverse adverse conditions such as fog, rain, night, blur, exposure changes, and LiDAR jitter [Yuan et al., 2026b]. MCubeS [Liang et al., 2022] contains 20-class material segmentation with RGB, NIR, DoLP, and AoLP modalities [Lin et al., 2025], using 302/96/102 train/val/test samples. MUSES [Brödermann et al., 2024] provides synchronized frame-camera, event-camera, and MEMS-LiDAR data with 2D panoptic annotations, enabling evaluation with heterogeneous sensors.

Implementation Details We use AdamW and train on A5000 GPUs: 8 GPUs for DELIVER and 4 GPUs for MCubeS/MUSES. Batch size is 1 per GPU for DELIVER and 4 for MCubeS/MUSES. The initial learning rate is $4 \times 1 0 ^ { - 4 }$ for DELIVER/MUSES and $3 . 5 \times 1 0 ^ { - 4 }$ for MCubeS, with a 10-epoch warm-up at 0.1×. We train for 200 epochs on DELIVER/MUSES and 500 epochs on MCubeS. Input resolutions are $1 0 2 4 \times 1 0 2 4$ for DELIVER/MUSES and $5 1 2 \times 5 1 2$ for MCubeS. We use standard augmentations, including random resizing (0.5–2.0), horizontal flipping, color jitter, blur, and random cropping. FuseAdapt uses rank 8 for DELIVER and MCubeS, and 4 for MUSES. We use DINOv2-B for DELIVER/MCubeS and DINOv2-S for MUSES [Oquab et al., 2023] as frozen VFM backbones.

Table 2: Results in MCSS setting on MUSES. The best results are highlighted in bold. (F: frame camera, E: event camera, L: LiDAR.)
<table><tr><td>Method</td><td>Backbone</td><td>|#Trainable|</td><td colspan="4">Modals</td></tr><tr><td></td><td></td><td>|Params (M)|</td><td>F</td><td>FE FL</td><td>FEL</td><td>Mean</td></tr><tr><td>CWSAM [Pu et al., 2025]</td><td>ViT-B</td><td>15.29</td><td>|55.41 41.7740.6949.98|</td><td></td><td></td><td>|46.96</td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B0</td><td>37.85</td><td>43.3743.3947.0346.66</td><td></td><td></td><td>45.11</td></tr><tr><td>MLE-SAM [Zhu et al., 2024]</td><td>Hiera-B+</td><td>20.79</td><td>73.9574.73 75.42 74.80</td><td></td><td></td><td>74.73</td></tr><tr><td>Ours (wo FuseAdapt)</td><td>ViT-S</td><td>0.00</td><td>62.46 63.00 64.72 65.14|63.83</td><td></td><td></td><td></td></tr><tr><td>Ours (shared FuseAdapt)</td><td>ViT-S</td><td>0.82</td><td>74.99 74.96 76.23 76.59</td><td></td><td></td><td>75.69</td></tr><tr><td>Ours (separate FuseAdapt)</td><td>ViT-S</td><td>3.22</td><td>74.9974.81 76.1376.38</td><td></td><td></td><td>75.58</td></tr><tr><td>Ours</td><td>ViT-S</td><td>3.29</td><td>74.99 75.36 76.94 77.28|76.14</td><td></td><td></td><td></td></tr></table>

Table 3: Results in MCSS setting on DELIVER. The best results are highlighted in bold. (R: RGB; D: Depth; E: Event; L: LiDAR.)
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">#Trainable |Params (M)|</td><td colspan="4">Modals</td><td rowspan="2">Mean</td></tr><tr><td>RD</td><td>RE RL</td><td>RDE</td><td>RDL RDEL</td></tr><tr><td>HRFuser [Broedermann et al., 2023][HRFuser-T</td><td></td><td>31.61</td><td>|51.88 42.22 43.13 51.83 52.72 52.97</td><td></td><td></td><td></td><td>|49.13</td></tr><tr><td>CWSAM [Pu et al., 2025]</td><td>ViT-B</td><td>15.29</td><td>58.64</td><td>56.22</td><td></td><td>55.43</td><td></td></tr><tr><td>MLE-SAM [Zhu et al., 2024]</td><td>Hiera-B+</td><td>20.79</td><td>63.57</td><td>62.69</td><td></td><td>64.08</td><td></td></tr><tr><td>MemorySAM [Liao et al., 2025]</td><td>Hiera-B+</td><td>14.50</td><td>63.48</td><td>62.42</td><td></td><td>65.38</td><td></td></tr><tr><td>TokenFusion [Wang et al., 2022]</td><td>MiT-B2</td><td>26.01</td><td>60.25 45.6353.01</td><td></td><td></td><td></td><td></td></tr><tr><td>CMX [Zhang et al., 2023a]</td><td>MiT-B2</td><td>66.57</td><td>62.67 56.62 56.37</td><td></td><td></td><td></td><td></td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B2</td><td>58.73</td><td>63.58 57.48 58.04 64.44 65.5066.30</td><td></td><td></td><td></td><td></td></tr><tr><td>Zheng et al. [Zheng et al., 2025]</td><td>MiT-B2</td><td>24.73</td><td>66.46 56.06</td><td></td><td></td><td></td><td></td></tr><tr><td>MAGIC [Zheng et al., 2024c]</td><td>MiT-B2</td><td>24.73</td><td></td><td>66.8958.48 57.75 66.24 67.6367.66</td><td></td><td></td><td></td></tr><tr><td>StitchFusion [Li et al., 2024]</td><td>MiT-B2</td><td>26.50</td><td></td><td>65.7557.4458.03 66.0366.6568.18</td><td></td><td></td><td>63.68</td></tr><tr><td>Any2Seg [Zheng et al., 2024b]</td><td>MiT-B2</td><td>25.79</td><td></td><td>67.12 57.83 57.92 66.75 67.2068.25</td><td></td><td></td><td>64.18</td></tr><tr><td>OmniSegmentor [Yin et al., 2025]</td><td>DFormer-L</td><td>38.93</td><td></td><td>64.70 58.60 59.20 65.90 67.2068.00</td><td></td><td></td><td>63.93</td></tr><tr><td>Ours</td><td>ViT-B</td><td>14.01</td><td></td><td>67.33 59.62 60.11 67.16 67.29 68.33</td><td></td><td></td><td>|64.97</td></tr></table>

Evaluation Protocol We report two settings. In MCSS, all modalities are available at training and inference, measuring full-modality fusion. In MISS, the model is trained with all modalities and evaluated on modality subsets at inference, measuring robustness when sensors are unavailable.

Comparison Context Cross-backbone headline comparisons reflect both the method and the pretrained representation. We separately report internally trained controls with frozen MiT-B0/B2 and a common SegFormer decoder [Xie et al., 2021], recipe, and checkpoint rule (Section 4.4). DINOv2-S was chosen on MUSES to keep the trainable budget comparable to MiT-B0 baselines, not to match backbone capacity. A matched DINOv2-B run reaches 80.16 versus 77.28 with DINOv2-S (Appendix A).

## 4.1 Modality-Complete Semantic Segmentation (MCSS) Results

We first evaluate MCSS on MCubeS, MUSES, and DELIVER, where all modalities are available during training and inference. As shown in Tabs. 1, 2, and 3, FuseAdapt achieves the best reported mean performance among the compared methods on all three benchmarks with a small trainable budget. On MCubeS, FuseAdapt obtains 57.62 mIoU with all four modalities (RADN) and 56.30 mean mIoU, compared with 53.29 for StitchFusion and 52.43 for MAGIC. On DELIVER, it achieves 64.97 mean mIoU, outperforming StitchFusion (63.68) and Any2Seg (64.18). On MUSES, it reaches 76.14 mean mIoU, improving over the prior reported results across all modality combinations. These results indicate that adapting a frozen VFM with modality-specific paths and routed adaptation exchange provides effective full-modality fusion.

## 4.2 Modality-Incomplete Semantic Segmentation (MISS) Results

We next evaluate MISS, where models are trained with all modalities and tested on arbitrary modality subsets. As shown in Tabs. 5, 4, and 6, FuseAdapt improves performance across a broad range of missing-modality settings. On MCubeS, FuseAdapt achieves 52.89 mean mIoU, outperforming

Table 4: Results in MISS setting on MUSES. All methods are trained with three modalities. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td>#Trainable</td><td colspan="7">Modals</td><td rowspan="2">Mean</td></tr><tr><td>Params (M)</td><td>F</td><td>E</td><td>L</td><td>FE</td><td>FL</td><td>EL</td><td>FEL</td></tr><tr><td>CMX [Zhang et al., 2023a]</td><td>MiT-B0</td><td>24.84</td><td>2.52</td><td>2.35</td><td>3.01</td><td>41.15</td><td>41.25</td><td>2.56</td><td>42.27</td><td>19.30</td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B0</td><td>37.85</td><td>3.50</td><td>2.77</td><td>2.64</td><td>6.63</td><td>10.28</td><td>3.14</td><td>46.66</td><td>10.80</td></tr><tr><td>MAGIC [Zheng et al., 2024c]</td><td>MiT-B0</td><td>3.72</td><td>43.22</td><td>2.68</td><td>22.95</td><td>43.51</td><td>49.05</td><td>22.98</td><td>49.02</td><td>33.34</td></tr><tr><td>Any2Seg [Zheng et al., 2024b]</td><td>MiT-B0</td><td>4.91</td><td>44.40</td><td>3.17</td><td>22.33</td><td>44.51</td><td>49.96</td><td>22.63</td><td>50.00</td><td>33.86</td></tr><tr><td>MLE-SAM [Zhu et al., 2024]</td><td>Hiera-B+</td><td>20.79</td><td>69.67</td><td>1.50</td><td>5.55</td><td>69.50</td><td>74.11</td><td>5.55</td><td>74.80</td><td>42.95</td></tr><tr><td>Zheng et al. [Zheng et al., 2025]</td><td>MiT-B0</td><td>3.72</td><td>60.33</td><td>33.15</td><td>42.59</td><td>47.19</td><td>53.47</td><td>39.59</td><td>47.89</td><td>47.80</td></tr><tr><td>BiXFormer [Chen et al., 2025a]</td><td>MiT-B0</td><td>17.70</td><td>65.76</td><td>16.88</td><td>39.63</td><td>58.42</td><td>59.77</td><td>49.89</td><td>65.81</td><td>50.88</td></tr><tr><td>Ours</td><td>ViT-S</td><td>3.29</td><td>71.09</td><td>30.16</td><td>46.59</td><td>73.24</td><td>76.72</td><td>54.04</td><td>77.28</td><td>61.30</td></tr></table>

![](images/5d0ea701b2879c5f424d414bbf22378eb845310c7dd4d9ef537efc4f8bfd5382.jpg)  
Image

![](images/4f63cf3633658f864d0ddff522c1a2377043e4a8eccee30862805d674834cc2e.jpg)  
GT

![](images/629c106022eb9c9c9e33bded92afcefba73fb355f78c6e95fa685f3708c9b4e5.jpg)  
CMNeXt

![](images/33a70e82e63a4e5dc94512a8b34369d31a0462054fd943c5ceba4c55783bfd59.jpg)  
StitchFusion

![](images/e4b0653b7f9b3bc8b5e4b8c225f200b47b760f1d323685690ad0ec4b8d9be32c.jpg)  
Ours  
Figure 4: Qualitative results on DELIVER in MISS setting. The proposed method is compared with CMNeXt [Zhang et al., 2023b] and StitchFusion [Li et al., 2024]. The model is trained on four modalities {r, d, e, l} and tested on {r} (Row 1) and {r, l} (Row 2).

MAGIC (38.00) and Zheng et al. (41.64). On DELIVER, it obtains 56.22 mean mIoU, compared with 45.04 for Any2Seg, 44.78 for MAGIC++, and 43.91 for StitchFusion. On MUSES, it reaches 61.30 mean mIoU with 3.29M trainable parameters, outperforming the reported baselines. The gains across different modality subsets suggest that the learned representation remains useful under changing sensor availability. Fig. 4 shows qualitative MISS results on DELIVER. With only RGB (Row 1) or RGB+LiDAR (Row 2), FuseAdapt produces cleaner predictions and more coherent scene layouts than CMNeXt and StitchFusion, especially in challenging night or fog cases where baselines show larger region-level errors.

## 4.3 Ablation Studies

We ablate FuseAdapt and PMM under the same training recipe and also examine the effect of changing the frozen VFM backbone.

FuseAdapt Variants (MCSS) Tabs. 1 and 2 compare variants within the same frozen-VFM framework. Without FuseAdapt, the model uses naive aggregation of per-modality features and performs worse. Shared FuseAdapt improves performance but uses one adaptation path for all modalities, limiting modality-specific correction. Separate FuseAdapt uses modality-specific paths without routed exchange; it improves over Shared on MCubeS, while their MUSES DINOv2-S means are similar. Full FuseAdapt performs best in both tables. This supports the contribution of both modality-specific adaptation and routed adaptation exchange.

PMM Variants (MISS) Tab. 5 evaluates PMM variants under MISS on MCubeS. Removing the predictor or replacing it with a linear predictor reduces mean mIoU from 52.89 to 43.67 and 49.71, respectively. This shows that the representation-level prediction objective is important, and that the query-based predictor is more effective than a simple linear prediction head. Additional ablations, including FuseAdapt rank and scalability, are provided in the supplementary material.

Additional VFMs To examine sensitivity to the frozen VFM choice, we keep the full pipeline fixed and replace DINOv2-B with EVA-CLIP-B [Sun et al., 2023] and NeCo-B [Pariza et al., 2024] on

Table 5: Results in MISS setting on MCubeS. All methods are trained with four modalities. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">|Trainable|</td><td colspan="4">Modals</td></tr><tr><td>Params*</td><td>R</td><td>A</td><td>D N</td><td>RA</td><td>RD RN AD AN DN</td><td>RAD RAN RDN</td><td>Mean ADN RADN|</td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B2</td><td>58.73</td><td>1.86 1.54</td><td>2.51</td><td>2.28</td><td>47.96 43.67 45.90</td><td>6.99 7.58 9.95</td><td>50.07 48.77 48.83 8.06 51.54</td><td>|25.17</td></tr><tr><td>MAGIC [Zheng et al., 2024c]</td><td>MiT-B2</td><td>24.73</td><td>51.91</td><td>0.32 34.52</td><td></td><td>2.66 </td><td>52.24 52.16 52.57 1.98</td><td>36.01 37.0952.4852.82 52.7337.57</td><td>38.00</td></tr><tr><td>Zheng et al. [Zheng et al., 2025]</td><td>MiT-B2</td><td>24.73</td><td></td><td></td><td></td><td></td><td></td><td>53.01 46.2137.6640.4940.5643.9343.1143.7340.1039.1940.7241.7841.5342.2140.63 42.73</td><td>41.64</td></tr><tr><td>Ours (wo pred.)</td><td>ViT-B</td><td>14.01</td><td></td><td></td><td></td><td></td><td></td><td>51.0213.6022.4733.61 53.16 54.92 54.0426.3738.2640.24 55.3355.6756.6143.02 56.66</td><td>|43.67</td></tr><tr><td>Ours (linear pred.)</td><td>ViT-B</td><td>14.01</td><td></td><td></td><td></td><td></td><td></td><td>51.38 31.08 42.02 44.9753.64 54.64 54.8243.36 47.38 49.62 55.0155.31 56.06 50.15 56.20</td><td>49.71</td></tr><tr><td>Ours</td><td>ViT-B</td><td>14.01</td><td></td><td></td><td></td><td></td><td></td><td>54.96 37.60 47.14 49.12 56.17 57.23 56.29 47.96 51.65 52.74 57.19 56.80 57.58 53.30 57.62</td><td>52.89</td></tr></table>

Table 6: Results in MISS setting on DELIVER. All methods are trained with four modalities. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="2">#Trainable</td><td colspan="6">Modals</td></tr><tr><td>|Params (M)|</td><td>R D</td><td>E</td><td>L RD</td><td>RE RL</td><td>DE DL</td><td>EL RDE RDL REL DEL RDEL</td><td>Mean</td></tr><tr><td>CMNeXt [Zhang et al., 2023b]</td><td>MiT-B2</td><td>58.73</td><td>22.5050.59 3.16</td><td>2.86</td><td></td><td>66.33 22.92 22.50 50.80 50.83</td><td>3.15</td><td>66.27 66.38 22.92 50.9866.33</td><td>|41.31</td></tr><tr><td>GeminiFusion [Jia et al., 2024]</td><td>MiT-B2</td><td>42.18</td><td>15.89 54.73</td><td>1.70</td><td>1.70</td><td></td><td>66.9316.24 15.80 54.83 54.761.70</td><td>66.92 66.9316.18 54.86 66.92</td><td>42.35</td></tr><tr><td>MAGIC [Zheng et al., 2024c]</td><td>MiT-B2</td><td>24.73</td><td>42.72 58.39</td><td>1.90</td><td>1.62</td><td></td><td>66.10 42.79 42.72 58.44 58.39</td><td>91.9066.1166.10 42.80 58.44 66.10</td><td>44.97</td></tr><tr><td>MAGIC++ [Zheng et al., 2024a]</td><td>MiT-B2</td><td>24.73</td><td>41.10 58.12</td><td>2.14</td><td>1.64</td><td></td><td>67.33 41.13 41.13 58.32 58.12</td><td>22.1567.35 67.33 41.17 58.3167.34</td><td>44.78</td></tr><tr><td>Any2Seg [Zheng et al., 2024b]</td><td>MiT-B2</td><td>25.79</td><td>39.02 60.11</td><td>2.07</td><td>0.31</td><td></td><td>68.21 39.11 39.04 60.92 60.15</td><td>1.99 68.24 68.22 39.06 60.95 68.25</td><td>45.04</td></tr><tr><td>StitchFusion [Li et al., 2024]</td><td>MiT-B2</td><td>26.50</td><td>30.93 55.44</td><td>1.87</td><td>1.59</td><td></td><td>68.2231.0333.5555.7655.41</td><td>1.8768.23 68.2133.66 55.7168.20</td><td>43.91</td></tr><tr><td>Zheng et al. [Zheng et al., 2025]</td><td>MiT-B2</td><td>24.73</td><td></td><td></td><td></td><td></td><td></td><td>55.56 57.74 32.4034.26 59.69 47.70 47.7848.9950.26 34.37 53.42 54.12 45.24 47.3351.65</td><td>48.29</td></tr><tr><td>Ours</td><td>ViT-B</td><td>14.01</td><td></td><td></td><td></td><td></td><td></td><td>58.52 59.48 30.73 27.67 68.93 59.43 58.08 60.39 58.89 37.72 68.90 68.42 58.58 59.22 68.33</td><td>56.22</td></tr></table>

MCubeS under MCSS. FuseAdapt achieves 56.02 mIoU with EVA-CLIP-B and 56.28 with NeCo-B, compared with 57.62 using DINOv2-B. Although absolute performance depends on the chosen VFM, the method remains competitive across different frozen backbones, reducing dependence on a single pretrained model.

## 4.4 Controlled Comparisons

Backbone and recipe controls. On MUSES with frozen MiT-B0, FuseAdapt obtains 57.54 mIoU versus 49.47 for Deep Prompt, 53.56 for Shared LoRA, and 54.80 for Adapter. These internal controls use the same decoder, split, augmentation, optimizer, schedule, PMM, and checkpoint rule, with 2.95/2.81/2.95/2.94M trainable parameters, respectively. External MiT-B0 references MAGIC and Any2Seg obtain 49.02/50.00. With MiT-B2, FuseAdapt obtains 70.60, compared with 58.28/55.36/53.40/59.36 for CMNeXt/MAGIC/StitchFusion/CHARM. External results share the backbone family but are not all recipe-matched; Appendix A gives definitions and protocols.

Alternative fusion mechanisms. Table 11 further compares fusion designs on MCubeS using DINOv2-B and the same segmentation decoder. Frozen feature averaging and concatenation obtain 51.16 and 52.42 mIoU, while token-attention fusion obtains 52.79. Late-logit fusion with private low-rank adaptation reaches 54.55, compared with 57.62 for FuseAdapt. These controls distinguish adaptation-space exchange from simply combining frozen representations or independently adapted predictions. Separate adaptation without routing reaches 56.09, narrowing the gap but remaining below the full model.

Where should information cross modalities? Dense-feature routing keeps Q/V private but mixes complete FFN features; All-path routed LoRA routes Q/V and FFN residuals. Both retain the backbone and decoder of the corresponding setting. FuseAdapt outperforms these alternatives on MCubeS and MUSES (Table 7), supporting the separation of private attention adaptation and residualonly exchange. These are internal mechanism controls, not reproductions of complete prior systems.

Prediction versus generic distillation. With the same backbone, masking, EMA teacher, segmentation supervision, and schedule, output-logit KD and fused-feature KD obtain 49.84 and 50.57 MISS mean mIoU on MCubeS, versus 52.89 for PMM. PMM improves output KD in all 15 nonempty subsets and fused-feature KD in 14/15. Removing EMA from the query predictor gives 50.59; the original per-token regressor gives 49.71. These controls support the target-conditioned prediction objective beyond generic full-to-partial matching. Table 8 summarizes the objectives; Appendix B gives every subset.

Table 7: Full-modality mIoU for information-flow variants (seed 3407).
<table><tr><td>Variant</td><td>MCubeS / DINOv2-B</td><td>MUSES / MiT-B0</td></tr><tr><td>Shared</td><td>54.69</td><td>53.56</td></tr><tr><td>Separate</td><td>56.09</td><td>54.31</td></tr><tr><td>Dense-feature routing</td><td>55.08</td><td>55.94</td></tr><tr><td>All-path routed LoRA</td><td>55.74</td><td>56.17</td></tr><tr><td>FuseAdapt</td><td>57.62</td><td>57.54</td></tr></table>

Table 8: MCubeS MISS mean mIoU over all 15 nonempty subsets. Only the auxiliary objective/target differs; full results are in Table 12.
<table><tr><td>Objective</td><td>Mean</td><td>Objective</td><td>Mean</td></tr><tr><td>Dropout only</td><td>43.67</td><td>Query without EMA</td><td>50.59</td></tr><tr><td>Per-token regression</td><td>49.71</td><td>Global-feature KD</td><td>50.57</td></tr><tr><td>Output KD</td><td>49.84</td><td>Full PMM</td><td>52.89</td></tr></table>

## 4.5 Robustness and Deployment Cost

Scope of robustness. On DELIVER’s ten predefined weather/degradation conditions, FuseAdapt obtains 67.36 mean mIoU versus 67.19 for Any2Seg, leading in eight conditions but trailing at night and under-exposure. In a separate MCubeS latent-prediction test, shifting the visible sensors together by 16 pixels relative to the masked target reduces cosine similarity by 0.00051 for the query predictor versus 0.01636 for per-token regression. The linear model has higher clean cosine; the evidence concerns lower shift sensitivity, not universally better reconstruction or arbitrary sensor misalignment. Appendix C provides both protocols and full tables.

Full-context target reliability. A full-input EMA target is useful on average but is not a per-sample oracle. Across 1,428 MCubeS image/subset pairs, comprising 102 images and 14 proper subsets, full-input predictions improve per-image mIoU by 4.44 on average and outperform partial-input predictions in 81.72% of pairs. PMM nevertheless uses detached latent representations, rather than segmentation pseudo-labels, as supervision. This analysis characterizes the benefit and limitation of the full-context assumption: additional sensors usually provide useful context under the evaluated distribution, but can also introduce misleading information. See Appendix B.1 for details.

End-to-end efficiency. On an idle H100, at 1024<sup>2</sup>, batch 1, FP32, with 20 warm-ups and 100 timed runs, three-modality MiT-B0 FuseAdapt takes 21.05 ms versus 19.63 for CMNeXt and 26.69 for StitchFusion. It retains 3.91M unique deployed parameters and removes both PMM components. Parameter efficiency does not imply lowest latency or memory: Any2Seg and MAGIC are faster in this profile. Appendix D reports all measured costs and identifies reconstructed inference graphs.

## 5 Discussion and Conclusion

FuseAdapt separates modality-private attention adaptation from cross-modal exchange through FFN low-rank residuals. PMM complements this design with target-conditioned missing-modality latent prediction. Same-backbone controls support both choices, with improvements across the evaluated MiT scales.

Limitations and deployment scope. Absolute accuracy depends on the frozen representation and its biases; cross-backbone comparisons alone cannot isolate our contribution. Inference cost still grows with the number of sensor streams. Routing and averaging require a common projected grid, and the shift experiment measures latent prediction sensitivity, not robustness to arbitrary calibration, field-of-view, or temporal errors. The main segmentation comparisons are reported from single runs, without a statistical-significance claim. DELIVER covers predefined adverse conditions, but neither those results nor a full-context EMA target guarantees safe behavior under correlated or out-of-distribution sensor failures, including device-level ones [Yuan et al., 2025b,c]. Deployment in robotics or driving requires application-specific testing, failure detection [Chen et al., 2025b], uncertainty estimation [Chen et al., 2026], and validated fallback mechanisms.

## Acknowledgments and Disclosure of Funding

This document is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 3 Grant (MOE-MOET32022-0001).

## References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Tim Brödermann, David Bruggemann, Christos Sakaridis, Kevin Ta, Odysseas Liagouris, Jason Corkill, and Luc Van Gool. Muses: The multi-sensor semantic perception dataset for driving under uncertainty. In European Conference on Computer Vision, pages 21–38. Springer, 2024.

Tim Broedermann, Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Hrfuser: A multi-resolution sensor fusion architecture for 2d object detection. In 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), pages 4159–4166. IEEE, 2023.

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

Jinming Cao, Hanchao Leng, Dani Lischinski, Daniel Cohen-Or, Changhe Tu, and Yangyan Li. Shapeconv: Shape-aware convolutional layer for indoor rgb-d semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7088–7097, 2021.

Jialei Chen, Xu Zheng, Danda Pani Paudel, Luc Van Gool, Hiroshi Murase, and Daisuke Deguchi. Bixformer: A robust framework for maximizing modality effectiveness in multi-modal semantic segmentation. arXiv preprint arXiv:2506.03675, 2025a.

Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems, 35:16664–16678, 2022.

Tingting Chen, Beibei Lin, Srinivas Anumasa, Vedant Shah, Zifeng Yuan, Qiran Zou, Anirudh Goyal, and Dianbo Liu. Auto-Discovery-Bench: Diagnosing structured state tracking in oracle-guided discovery. arXiv preprint arXiv:2502.15224, 2025b.

Tingting Chen, Beibei Lin, Zifeng Yuan, Qiran Zou, Hongyu He, Anirudh Goyal, Yew-Soon Ong, and Dianbo Liu. HypoSpace: A diagnostic benchmark for set-valued hypothesis generation under underdetermination and sublinear coverage bounds. In International Conference on Machine Learning (ICML), 2026.

Shujian Deng, Xin Zhang, Wen Yan, Eric I-Chao Chang, Yubo Fan, Maode Lai, and Yan Xu. Deep learning in digital pathology image analysis: A survey. Frontiers of Medicine, 14(4):470–487, 2020.

Wanjun Du, Zifeng Yuan, Tingting Chen, Fucai Ke, Beibei Lin, and Shunli Zhang. WeatherReasonSeg: A benchmark for weather-aware reasoning segmentation in visual language models. In European Conference on Computer Vision (ECCV), 2026.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Keli Huang, Botian Shi, Xiang Li, Xin Li, Siyuan Huang, and Yikang Li. Multi-modal sensor fusion for auto driving perception: A survey. arXiv preprint arXiv:2202.02703, 2022.

Tianrui Hui, Zizheng Xun, Fengguang Peng, Junshi Huang, Xiaoming Wei, Xiaolin Wei, Jiao Dai, Jizhong Han, and Si Liu. Bridging search region interaction with template for rgb-t tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13630–13639, 2023.

Ding Jia, Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Chang Xu, and Xinghao Chen. Geminifusion: Efficient pixel-wise multimodal fusion for vision transformer. arXiv preprint arXiv:2406.01210, 2024.

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pages 709–727. Springer, 2022.

Zexi Jia, Kaichao You, Weihua He, Yang Tian, Yongxiang Feng, Yaoyuan Wang, Xu Jia, Yihang Lou, Jingyi Zhang, Guoqi Li, et al. Event-based semantic segmentation with posterior attention. IEEE Transactions on Image Processing, 32:1829–1842, 2023.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

Bingyu Li, Da Zhang, Zhiyuan Zhao, Junyu Gao, and Xuelong Li. Stitchfusion: Weaving any visual modalities to enhance multimodal semantic segmentation. arXiv preprint arXiv:2408.01343, 2024.

Jiale Li, Hang Dai, Hao Han, and Yong Ding. Mseg3d: Multi-modal 3d semantic segmentation for autonomous driving. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 21694–21704, 2023.

Xiangtai Li, Tao Zhang, Yanwei Li, Haobo Yuan, Shihao Chen, Yikang Zhou, Jiahao Meng, Yueyi Sun, Shilin Xu, Lu Qi, Tianheng Cheng, Yi Lin, Zilong Huang, Wenhao Huang, Jiashi Feng, and Guang Shi. DenseWorld-1M: Towards detailed dense grounded caption in the real world. arXiv preprint arXiv:2506.24102, 2025a.

Yingwei Li, Adams Wei Yu, Tianjian Meng, Ben Caine, Jiquan Ngiam, Daiyi Peng, Junyang Shen, Yifeng Lu, Denny Zhou, Quoc V Le, et al. Deepfusion: Lidar-camera deep fusion for multi-modal 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17182–17191, 2022.

Ziqi Li, Tao Gao, Yisheng An, Ting Chen, Jing Zhang, Yuanbo Wen, Mengkun Liu, and Qianxi Zhang. Brain-inspired spiking neural networks for energy-efficient object detection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3552–3562. IEEE, 2025b.

Ziqi Li, Tao Gao, Shutao Li, Ting Chen, Yisheng An, Yuanbo Wen, and Tao Lei. Frequency-oriented adaptive real-time object detector for cluttered traffic scenes. Nature Communications, 17(1):9787, 2026. doi: 10.1038/s41467-026-76346-1.

Yupeng Liang, Ryosuke Wakaki, Shohei Nobuhara, and Ko Nishino. Multimodal material segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19800–19808, 2022.

Chenfei Liao, Xu Zheng, Yuanhuiyi Lyu, Haiwei Xue, Yihong Cao, Jiawen Wang, Kailun Yang, and Xuming Hu. Memorysam: Memorize modalities and semantics with segment anything model 2 for multi-modal semantic segmentation. arXiv preprint arXiv:2503.06700, 2025.

Beibei Lin, Zifeng Yuan, and Tingting Chen. RGB-to-polarization estimation: A new task and benchmark study. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025.

Jiyuan Liu, Jia Lin, Xiaofei Zhou, Runmin Cong, Deyang Liu, and Zhi Liu. M4-SAM: Multi-modal mixture-of-experts with memory-augmented SAM for RGB-D video salient object detection. arXiv preprint arXiv:2605.11760, 2026.

Ruiping Liu, Jiaming Zhang, Kunyu Peng, Yufan Chen, Ke Cao, Junwei Zheng, M Saquib Sarfraz, Kailun Yang, and Rainer Stiefelhagen. Fourier prompt tuning for modality-incomplete scene segmentation. In 2024 IEEE Intelligent Vehicles Symposium (IV), pages 961–968. IEEE, 2024.

Harsh Maheshwari, Yen-Cheng Liu, and Zsolt Kira. Missing modality robustness in semi-supervised multi-modal semantic segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 1020–1030, 2024.

Shishir Muralidhara, Didier Stricker, and René Schuster. MILE: Mixture of incremental LoRA experts for continual semantic segmentation across domains and modalities. arXiv preprint arXiv:2605.03555, 2026.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Valentinos Pariza, Mohammadreza Salehi, Gertjan Burghouts, Francesco Locatello, and Yuki M Asano. Neco: Improving dinov2’s spatial representations in 19 gpu hours with patch neighbor consistency. arXiv e-prints, pages arXiv–2408, 2024.

Seong-Jin Park, Ki-Sang Hong, and Seungyong Lee. Rdfnet: Rgb-d multi-level residual feature fusion for indoor semantic segmentation. In Proceedings ofthe IEEE international conference on computer vision, pages 4980–4989, 2017.

Xinyang Pu, Hecheng Jia, Linghao Zheng, Feng Wang, and Feng Xu. Classwise-sam-adapter: Parameter efficient fine-tuning adapts segment anything to sar domain for semantic segmentation. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

Ziniu Qian, Zihua Wang, Xin Zhang, Bingzheng Wei, Maode Lai, Jianzhong Shou, Yubo Fan, and Yan Xu. MSNSegNet: Attention-based multi-shape nuclei instance segmentation in histopathology images. Medical & Biological Engineering & Computing, 62(6):1821–1836, 2024.

Ukcheol Shin, Kyunghyun Lee, In So Kweon, and Jean Oh. Complementary random masking for RGB-Thermal semantic segmentation. In IEEE International Conference on Robotics and Automation, 2024.

Mengke Song, Wenfeng Song, Guowei Yang, and Chenglizhao Chen. Improving rgb-d salient object detection via modality-aware decoder. IEEE Transactions on Image Processing, 31:6124–6138, 2022.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Zhiqiang Teng, Tingting Chen, Beibei Lin, Zifeng Yuan, Xuanyi Li, Xuanyu Zhang, and Shunli Zhang. RaindropGS: A benchmark for 3D gaussian splatting under raindrop conditions. arXiv preprint arXiv:2510.17719, 2025.

Hu Wang, Congbo Ma, Jianpeng Zhang, Yuan Zhang, Jodie Avery, Louise Hull, and Gustavo Carneiro. Learnable cross-modal knowledge distillation for multi-modal learning with missing modality. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 216–226. Springer, 2023.

Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19757–19767, 2024.

Yikai Wang, Wenbing Huang, Fuchun Sun, Tingyang Xu, Yu Rong, and Junzhou Huang. Deep multimodal fusion by channel exchanging. Advances in neural information processing systems, 33: 4835–4845, 2020.

Yikai Wang, Xinghao Chen, Lele Cao, Wenbing Huang, Fuchun Sun, and Yunhe Wang. Multimodal token fusion for vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12186–12195, 2022.

Zhixiang Wei, Lin Chen, Yi Jin, Xiaoxiao Ma, Tianle Liu, Pengyang Ling, Ben Wang, Huaian Chen, and Jinjin Zheng. Stronger fewer & superior: Harnessing vision foundation models for domain generalized semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 28619–28630, 2024.

Lekang Wen, Jing Xiao, Liang Liao, Jiajun Chen, and Mi Wang. CHARM: Collaborative harmonization across arbitrary modalities for modality-agnostic semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 10603–10611, 2026. doi: 10.1609/aaai.v40i13.38033.

Wei Wu, Tao Chu, and Qiong Liu. Complementarity-aware cross-modal feature fusion network for rgb-t semantic segmentation. Pattern Recognition, 131:108881, 2022.

Xun Wu, Shaohan Huang, and Furu Wei. Mixture of LoRA experts. arXiv preprint arXiv:2404.13628, 2024.

Ruihao Xia, Chaoqiang Zhao, Meng Zheng, Ziyan Wu, Qiyu Sun, and Yang Tang. Cmda: Crossmodality domain adaptation for nighttime semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21572–21581, 2023.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34:12077–12090, 2021.

Weilong Yan, Xin Zhang, and Robby T. Tan. ER-LoRA: Effective-rank guided adaptation for weather-generalized depth estimation. arXiv preprint arXiv:2509.00665, 2025.

Xu Yan, Jiantao Gao, Chaoda Zheng, Chao Zheng, Ruimao Zhang, Shuguang Cui, and Zhen Li. 2dpass: 2d priors assisted semantic segmentation on lidar point clouds. In European Conference on Computer Vision, pages 677–695. Springer, 2022.

Xin Yang, Xin Zhang, and Xinchao Wang. ERF: A benchmark dataset for robust semantic segmentation under extreme rainfall conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9301–9309, 2025.

Zhiwen Yang, Jiayin Li, Hui Zhang, Dan Zhao, Bingzheng Wei, and Yan Xu. Restore-RWKV: Efficient and effective medical image restoration with RWKV. IEEE Journal ofBiomedical and Health Informatics, 30(1):513–526, 2026a. doi: 10.1109/JBHI.2025.3588555.

Zhiwen Yang, Yang Zhou, Haowei Chen, Hui Zhang, Dan Zhao, Bingzheng Wei, and Yan Xu. UniPET: A universal network for high-quality PET image denoising across varied dose reduction factors. Medical Image Analysis, 111:104059, 2026b. doi: 10.1016/j.media.2026.104059.

Bo-Wen Yin, Jiao-Long Cao, Xuying Zhang, Yuming Chen, Ming-Ming Cheng, and Qibin Hou. Omnisegmentor: A flexible multi-modal learning framework for semantic segmentation. arXiv preprint arXiv:2509.15096, 2025.

Xiaowen Ying and Mooi Choo Chuah. Uctnet: Uncertainty-aware cross-modal transformer network for indoor rgb-d semantic segmentation. In European Conference on Computer Vision, pages 20–37. Springer, 2022.

Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2VA: Marrying SAM2 with MLLM for dense grounded understanding of images and videos. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026a. doi: 10.1109/TPAMI.2026.3720379.

Zifeng Yuan, Tingting Chen, Dewen Zhang, Yuan Gao, Wenkai Shan, Beibei Lin, and Aaron Danner. Tailored polarization-switchable VCSEL arrays for photonic Ising computing. Applied Physics Letters, 127(22):221102, 2025a.

Zifeng Yuan, Dewen Zhang, Yuan Gao, Luo Qi, Wujie Fu, and Aaron Danner. Large-scale fabrication and analysis of polarization behavior in VCSELs with tailored apertures. Journal of Lightwave Technology, 43(14):6819–6827, 2025b.

Zifeng Yuan, Dewen Zhang, Lei Shi, Yutong Liu, and Aaron Danner. Enhanced polarization locking in VCSELs. Applied Physics Letters, 126(15):151101, 2025c.

Zifeng Yuan, Beibei Lin, Yutong Liu, Wujie Fu, Yuan Gao, Tingting Chen, and Aaron Danner. Drive-efficient GHz-rate binary-state updates enabled by VCSEL polarization switching. Laser & Photonics Reviews, page e71890, 2026b.

Jiaming Zhang, Kailun Yang, and Rainer Stiefelhagen. Issafe: Improving semantic segmentation in accidents by fusing event-based data. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1132–1139. IEEE, 2021a.

Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers. IEEE Transactions on intelligent transportation systems, 2023a.

Jiaming Zhang, Ruiping Liu, Hao Shi, Kailun Yang, Simon Reiß, Kunyu Peng, Haodong Fu, Kaiwei Wang, and Rainer Stiefelhagen. Delivering arbitrary-modal semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1136–1147, 2023b.

Jingwei Zhang, Xin Zhang, Ke Ma, Rajarsi Gupta, Joel Saltz, Maria Vakalopoulou, and Dimitris Samaras. Gigapixel whole-slide images classification using locally supervised learning. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, pages 192–201. Springer Nature Switzerland, 2022.

Qiang Zhang, Shenlu Zhao, Yongjiang Luo, Dingwen Zhang, Nianchang Huang, and Jungong Han. Abmdrnet: Adaptive-weighted bi-directional modality difference reduction network for rgb-t semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2633–2642, 2021b.

Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. OMG-LLaVA: Bridging image-level, object-level, pixel-level reasoning and understanding. In Advances in Neural Information Processing Systems, volume 37, pages 71737–71767. Curran Associates, Inc., 2024a. doi: 10.52202/079017-2291.

Tianlu Zhang, Hongyuan Guo, Qiang Jiao, Qiang Zhang, and Jungong Han. Efficient rgb-t tracking via cross-modality distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5404–5413, 2023c.

Xin Zhang and Ying-Cong Chen. Adaptive domain generalization via online disagreement minimization. IEEE Transactions on Image Processing, 32:4247–4258, 2023.

Xin Zhang and Robby T Tan. Mamba as a bridge: Where vision foundation models meet vision language models for domain-generalized semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14527–14537, 2025.

Xin Zhang, Weixuan Kou, Eric I-Chao Chang, He Gao, Yubo Fan, and Yan Xu. Sleep stage classification based on multi-level feature learning and recurrent neural networks via wearable device. Computers in Biology and Medicine, 103:71–81, 2018.

Xin Zhang, Jinheng Xie, Yuan Yuan, Michael Bi Mi, and Robby T. Tan. HEAP: Unsupervised object discovery and localization with contrastive grouping. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 7323–7331, 2024b.

Xin Zhang, Haochen Wang, Yikang Zhou, Zhuochen Wang, Xiangtai Li, and Robby T. Tan. Actor as its own critic: Unifying region understanding and localization via CycleGRPO. In European Conference on Computer Vision (ECCV), volume 17044 of Lecture Notes in Computer Science, pages 586–604. Springer, 2026. doi: 10.1007/978-3-032-37577-3\_32.

Xu Zheng, Yuanhuiyi Lyu, Lutao Jiang, Jiazhou Zhou, Lin Wang, and Xuming Hu. Magic++: Efficient and resilient modality-agnostic semantic segmentation via hierarchical modality selection. arXiv preprint arXiv:2412.16876, 2024a.

Xu Zheng, Yuanhuiyi Lyu, and Lin Wang. Learning modality-agnostic representation for semantic segmentation from any modalities. In European Conference on Computer Vision, pages 146–165. Springer, 2024b.

Xu Zheng, Yuanhuiyi Lyu, Jiazhou Zhou, and Lin Wang. Centering the value of every modality: Towards efficient and resilient modality-agnostic semantic segmentation. In European Conference on Computer Vision, pages 192–212. Springer, 2024c.

Xu Zheng, Yuanhuiyi Lyu, Lutao Jiang, Danda Pani Paudel, Luc Van Gool, and Xuming Hu. Reducing unimodal bias in multi-modal semantic segmentation with multi-scale functional entropy regularization. arXiv preprint arXiv:2505.06635, 2025.

Yikang Zhou, Tao Zhang, Dengxian Gong, Yuanzheng Wu, Ye Tian, Haochen Wang, Haobo Yuan, Jiacong Wang, Lu Qi, Hao Fei, Shunping Ji, Anran Wang, Zhuochen Wang, Yujing Wang, Cheng Chen, and Xiangtai Li. SAMTok: Representing any mask with two words. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 37852–37863, 2026.

Chenyang Zhu, Bin Xiao, Lin Shi, Shoukun Xu, and Xu Zheng. Customize segment anything model for multi-modal semantic segmentation with mixture of lora experts. arXiv preprint arXiv:2412.04220, 2024.

Zhuangwei Zhuang, Rong Li, Kui Jia, Qicheng Wang, Yuanqing Li, and Mingkui Tan. Perceptionaware multi-sensor fusion for 3d lidar semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16280–16290, 2021.

## A Controlled Comparisons and Experimental Protocols

We separate external same-backbone references from controls trained within our framework. For MUSES MiT-B0/B2, internal controls use a frozen encoder, SegFormer decoder, the same split, augmentations, optimizer, 200-epoch schedule, 1024 × 1024 inputs, rank-4 adaptation where applicable, EMA-query PMM, effective global batch size 16, and the best full-modality validation checkpoint selected independently for each run. Individual controls use seed 3407. These experiments test backbone generality on the evaluated hierarchical MiT scales, not arbitrary architectures.

Deep Prompt inserts 64 modality-specific learnable prompts into every Transformer block. Shared LoRA applies one rank-4 Q/V and FFN update shared across sensors. Adapter inserts modalityspecific width-32 residual bottlenecks in stages 2–4. Separate LoRA uses private Q/V and FFN residual branches without cross-modal routing. Full FuseAdapt adds receiver-conditioned FFN residual exchange. All variants average the resulting visible features before the shared decoder.

Table 9: MUSES full-modality mIoU with MiT-B0. External references share the backbone family; internal controls additionally share the decoder and recipe. Trainable parameter counts include optimized components of the internal setup.
<table><tr><td>Method</td><td>Comparison</td><td>Trainable (M)</td><td>mIoU</td></tr><tr><td>MAGIC</td><td>External</td><td>3.72</td><td>49.02</td></tr><tr><td>Any2Seg</td><td>External</td><td>4.91</td><td>50.00</td></tr><tr><td>Deep Prompt</td><td>Internal</td><td>2.95</td><td>49.47</td></tr><tr><td>Shared LoRA</td><td>Internal</td><td>2.81</td><td>53.56</td></tr><tr><td>Adapter</td><td>Internal</td><td>2.95</td><td>54.80</td></tr><tr><td>FuseAdapt</td><td>Internal</td><td>2.94</td><td>57.54</td></tr></table>

Table 10: MUSES full-modality mIoU using MiT-B2. Any2Seg has no reported MUSES/MiT-B2 result and is omitted.
<table><tr><td>External method</td><td>mIoU</td><td>Internal variant (seed 3407)</td><td>mIoU</td></tr><tr><td>CMNeXt</td><td>58.28</td><td>Shared</td><td>65.38</td></tr><tr><td>MAGIC</td><td>55.36</td><td>Separate</td><td>66.96</td></tr><tr><td>StitchFusion</td><td>53.40</td><td>FuseAdapt</td><td>70.60</td></tr><tr><td>CHARM [Wen et al., 2026]</td><td>59.36</td><td></td><td></td></tr></table>

DINOv2 capacity. Under the same MUSES split, seed 3407, 200 epochs, rank 4, EMA-query PMM, optimizer schedule, and effective batch size 16, DINOv2-S/B achieve 77.28/80.16 full-modality best-validation mIoU. The 2.88-point difference confirms the contribution of backbone capacity. DINOv2-S was selected to keep the original 3.29M trainable budget comparable to MAGIC/Any2Seg (3.72M/4.91M); it does not match their backbone family.

Fusion controls on MCubeS. All rows in Table 11 use DINOv2-B and a shared segmentation decoder. Frozen mean averages frozen per-modality features without LoRA. Frozen concatenation concatenates those features and projects them through a trainable 1 × 1 layer at each scale. Token-attention fusion uses eight-head attention across sensor tokens at each common-grid location. Late-logit fusion uses private rank-8 Q/V and FFN updates without routing, decodes each stream independently with the shared head, and averages logits. Shared and Separate use shared or private low-rank branches, respectively, followed by feature averaging.

Information-flow alternatives. All-path routed LoRA routes the low-rank Q/V residuals as well as the FFN residuals, testing whether attention adaptation should also mix across sensors. Densefeature routing preserves private Q/V but exchanges complete FFN features, testing whether the low-rank residual restriction matters. The MUSES MiT-B0 controls share rank, PMM, training recipe, effective batch size, seed, trainable parameter count, and checkpoint rule. Table 7 also reports the corresponding MCubeS/DINOv2-B results. These controlled alternatives isolate architectural choices rather than reproducing StitchFusion or MLE-SAM in full.

Table 11: MCubeS, all four modalities (RADN): fusion/adaptation mIoU.
<table><tr><td>Method</td><td>RADN mIoU</td></tr><tr><td>Frozen mean Frozen concatenation</td><td>51.16 52.42</td></tr><tr><td>Token-attention fusion Late-logit fusion</td><td>52.79</td></tr><tr><td>Shared</td><td>54.55 54.69</td></tr><tr><td>Separate</td><td>56.09</td></tr><tr><td>FuseAdapt</td><td>57.62</td></tr></table>

## B PMM Objective Controls and Teacher Reliability

The MCubeS MISS comparisons keep the backbone, decoder, modality masking, segmentation supervision, and training schedule fixed. Output KD matches partial-input student logits to fullcontext EMA teacher logits through pixelwise KL divergence. Global-feature KD matches the student’s fused representation to the full-context EMA fused representation using cosine loss. Neither identifies a missing target sensor. Per-token regression instead predicts its latent tokens at the same spatial indices. Query without EMA uses the same query predictor as PMM but a detached online full-context target. Full PMM uses an EMA target and target-specific query cross-attention. An EMA teacher without a matching loss would not provide student supervision.

Table 12: MCubeS MISS mIoU for all objective controls. Mean averages all 15 nonempty subsets, including RADN. R/A/D/N denote RGB/AoLP/DoLP/NIR.
<table><tr><td>Subset</td><td>Dropout</td><td>Output KD</td><td>Linear</td><td>Query, no EMA</td><td>Feature KD</td><td>PMM</td></tr><tr><td>R</td><td>51.02</td><td>51.40</td><td>51.38</td><td>53.97</td><td>53.46</td><td>54.96</td></tr><tr><td>A</td><td>13.60</td><td>34.92</td><td>31.08</td><td>37.99</td><td>38.66</td><td>37.60</td></tr><tr><td>D</td><td>22.47</td><td>46.13</td><td>42.02</td><td>44.75</td><td>43.70</td><td>47.14</td></tr><tr><td>N</td><td>33.61</td><td>45.79</td><td>44.97</td><td>46.67</td><td>45.65</td><td>49.12</td></tr><tr><td>RA</td><td>53.16</td><td>52.08</td><td>53.64</td><td>55.10</td><td>54.27</td><td>56.17</td></tr><tr><td>RD</td><td>54.92</td><td>53.99</td><td>54.64</td><td>54.57</td><td>55.06</td><td>57.23</td></tr><tr><td>RN</td><td>54.04</td><td>52.62</td><td>54.82</td><td>53.06</td><td>53.64</td><td>56.29</td></tr><tr><td>AD</td><td>26.37</td><td>45.39</td><td>43.36</td><td>46.35</td><td>44.64</td><td>47.96</td></tr><tr><td>AN</td><td>38.26</td><td>48.33</td><td>47.38</td><td>48.88</td><td>48.66</td><td>51.65</td></tr><tr><td>DN</td><td>40.24</td><td>49.63</td><td>49.62</td><td>49.67</td><td>49.61</td><td>52.74</td></tr><tr><td>RAD</td><td>55.33</td><td>53.70</td><td>55.01</td><td>54.86</td><td>55.40</td><td>57.19</td></tr><tr><td>RAN</td><td>55.67</td><td>53.36</td><td>55.31</td><td>53.63</td><td>54.09</td><td>56.80</td></tr><tr><td>RDN</td><td>56.61</td><td>54.71</td><td>56.06</td><td>54.52</td><td>55.47</td><td>57.58</td></tr><tr><td>ADN</td><td>43.02</td><td>50.56</td><td>50.15</td><td>50.32</td><td>50.75</td><td>53.30</td></tr><tr><td>RADN</td><td>56.66</td><td>55.00</td><td>56.20</td><td>54.58</td><td>55.55</td><td>57.62</td></tr><tr><td>Mean</td><td>43.67</td><td>49.84</td><td>49.71</td><td>50.59</td><td>50.57</td><td>52.89</td></tr></table>

PMM exceeds output KD by 3.05 mean mIoU and fused-feature KD by 2.32. It improves the former in all 15 subsets and the latter in 14/15. Linear prediction improves dropout-only training by 6.04; query prediction improves the linear control by 3.18; EMA improves the no-EMA query by 2.30. These results support modality-conditioned latent prediction beyond a generic consistency objective, without claiming masking or EMA as new mechanisms.

## B.1 Full-context targets are an average prior

Teacher and student share the same frozen initialization and architecture. Only student adaptation parameters receive gradients, and teacher adaptation parameters follow the EMA update with coefficient 0.999. PMM uses detached teacher representations rather than segmentation pseudo-labels. Full-modality input does not guarantee better predictions when a sensor is degraded. Over 1,428 MCubeS image/subset pairs (102 images and 14 proper subsets), full-input predictions improve per-image mIoU by 4.44 on average and are better in 81.72% of pairs. Thus full context is useful on average under this distribution, but it is not a per-sample oracle. Reliability-aware target weighting under sensor corruption remains future work.

## C Degradation and Spatial-Shift Evaluation

Degraded but present modalities. Table 13 evaluates all four DELIVER modalities (R-D-E-L) on the benchmark’s predefined weather and sensor-degradation subsets. MB/OE/UE/LJ/EL denote motion blur, over-exposure, under-exposure, LiDAR jitter, and event low resolution. This is not an additional arbitrary-corruption benchmark. FuseAdapt leads eight conditions and the mean, while Any2Seg leads Night and UE.

Table 13: DELIVER, four present modalities: mIoU by predefined condition.
<table><tr><td>Condition</td><td>HRFuser</td><td>CMNeXt</td><td>Any2Seg</td><td>FuseAdapt</td></tr><tr><td>Cloudy</td><td>56.20</td><td>68.70</td><td>69.19</td><td>69.22</td></tr><tr><td>Foggy</td><td>52.39</td><td>65.67</td><td>67.64</td><td>67.93</td></tr><tr><td>Night</td><td>49.85</td><td>62.46</td><td>67.25</td><td>66.44</td></tr><tr><td>Rainy</td><td>52.53</td><td>67.50</td><td>67.00</td><td>68.76</td></tr><tr><td>Sunny</td><td>54.02</td><td>66.57</td><td>68.25</td><td>68.93</td></tr><tr><td>MB</td><td>49.44</td><td>62.91</td><td>66.23</td><td>66.39</td></tr><tr><td>OE</td><td>46.31</td><td>64.59</td><td>64.89</td><td>67.21</td></tr><tr><td>UE</td><td>46.92</td><td>60.00</td><td>65.69</td><td>61.39</td></tr><tr><td>LJ</td><td>53.94</td><td>65.92</td><td>69.05</td><td>69.08</td></tr><tr><td>EL</td><td>52.72</td><td>65.48</td><td>66.73</td><td>68.23</td></tr><tr><td>Mean</td><td>52.97</td><td>64.98</td><td>67.19</td><td>67.36</td></tr></table>

Latent prediction under spatial shifts. We mask each MCubeS target sensor in turn, then jointly shift the other three visible sensors by 0/2/4/8/16 input pixels relative to the clean full-context EMA target. Visible–visible alignment is preserved while target–visible correspondence changes. We compute channel-wise cosine between predicted target tokens and their teacher counterparts, averaged over tokens, 102 images, four target modalities, and three seeds. At 0 pixels the target is still missing and visible inputs are unshifted. Query prediction has the smaller drop in all 12 seed–target

Table 14: MCubeS predicted/teacher latent cosine under target–visible shifts. Final column reports mean drop ± SD as reported across the three seeds.
<table><tr><td>Predictor</td><td> $0 \ : \mathrm { p x }$ </td><td> $2 { \mathfrak { p } } \mathbf { x }$ </td><td> $4 { \mathfrak { p } } \mathbf { x }$ </td><td> $8 \ \mathrm { p x }$ </td><td> $1 6 \mathrm { p x }$ </td><td> $\mathrm { D r o p , 0 - 1 6 p x }$ </td></tr><tr><td>Linear</td><td>.86968</td><td>.86766</td><td>.86583</td><td>.86136</td><td>.85332</td><td> $. 0 1 6 3 6 \pm . 0 0 3 2 3$ </td></tr><tr><td>Query</td><td>.74025</td><td>.74012</td><td>.74008</td><td>.74006</td><td>.73974</td><td> $. 0 0 0 5 1 \pm . 0 0 0 2 9$ </td></tr></table>

comparisons at every nonzero shift; its 16-pixel drop is 32.2 times smaller. Linear prediction has higher absolute clean cosine. The experiment measures sensitivity of latent prediction to these translations, not general reconstruction superiority or end-to-end robustness to arbitrary native token lengths, fields of view, or temporal offsets.

## D End-to-End Inference Efficiency

We profile all methods on the same idle H100 at $1 0 2 4 \times 1 0 2 4 .$ , batch size 1, FP32, with 20 warmup iterations and 100 measured runs. FuseAdapt, CMNeXt, and StitchFusion use MiT-B0 and 256-dimensional segmentation decoders. Any2Seg and MAGIC use MiT-B0 with their paperspecific heads. Deployed parameters count unique parameters retained for inference, counting shared backbones once and removing PMM’s EMA teacher and query predictor. An asterisk identifies an inference graph reconstructed from its paper because executable code was unavailable at the time of profiling; the dagger denotes the released DFormer-L+LightHam proxy for OmniSegmentor.

FuseAdapt’s three-input latency is close to CMNeXt and below StitchFusion in this profile; its unique deployed parameter count is smaller than both. However, Any2Seg/MAGIC are faster, and CMNeXt/StitchFusion use less peak memory. Reconstructed graphs and the OmniSegmentor proxy should not be interpreted as official implementation benchmarks. We omit traced FLOPs because unsupported routing, normalization, and interpolation operations cause method-dependent

Table 15: End-to-end inference cost under the common H100 protocol. Rows list one, two, or three input modalities. Latency is ms, throughput is images/s, memory is peak GiB, and parameters are deployed millions.
<table><tr><td>Method</td><td>Modalities</td><td>Latency</td><td>Throughput</td><td>Memory</td><td>Params</td></tr><tr><td>CMNeXt</td><td>1</td><td>7.30</td><td>136.96</td><td>0.70</td><td>3.72</td></tr><tr><td></td><td>2</td><td>18.45</td><td>54.19</td><td>0.74</td><td>10.29</td></tr><tr><td></td><td>3</td><td>19.63</td><td>50.93</td><td>0.75</td><td>10.30</td></tr><tr><td>StitchFusion</td><td>1</td><td>8.32</td><td>120.26</td><td>0.70</td><td>4.26</td></tr><tr><td></td><td>2</td><td>16.41</td><td>60.93</td><td>0.72</td><td>4.39</td></tr><tr><td></td><td>3</td><td>26.69</td><td>37.47</td><td>0.73</td><td>4.56</td></tr><tr><td>Any2Seg*</td><td>1</td><td>6.53</td><td>153.08</td><td>0.60</td><td>3.32</td></tr><tr><td></td><td>2</td><td>10.26</td><td>97.43</td><td>1.16</td><td>3.32</td></tr><tr><td></td><td>3</td><td>14.69</td><td>68.09</td><td>1.71</td><td>3.32</td></tr><tr><td>MAGIC*</td><td>1</td><td>6.49</td><td>154.16</td><td>0.60</td><td>3.52</td></tr><tr><td></td><td>2</td><td>10.25</td><td>97.53</td><td>1.16</td><td>3.59</td></tr><tr><td></td><td>3</td><td>14.72</td><td>67.92</td><td>1.71</td><td>3.65</td></tr><tr><td>FuseAdapt</td><td>1</td><td>9.40</td><td>106.33</td><td>0.73</td><td>3.78</td></tr><tr><td></td><td>2</td><td>15.87</td><td>63.02</td><td>1.18</td><td>3.84</td></tr><tr><td></td><td>3</td><td>21.05</td><td>47.50</td><td>1.75</td><td>3.91</td></tr><tr><td>MLE-SAM*</td><td>1</td><td>33.91</td><td>29.49</td><td>0.95</td><td>89.04</td></tr><tr><td></td><td>2</td><td>61.95</td><td>16.14</td><td>1.06</td><td>89.04</td></tr><tr><td></td><td>3</td><td>89.72</td><td>11.15</td><td>1.17</td><td>89.04</td></tr><tr><td>MemorySAM</td><td>1</td><td>35.56</td><td>28.12</td><td>0.87</td><td>84.08</td></tr><tr><td></td><td>2</td><td>78.39</td><td>12.76</td><td>1.21</td><td>84.08</td></tr><tr><td></td><td>3</td><td>125.23</td><td>7.99</td><td>1.43</td><td>84.08</td></tr><tr><td>OmniSegmentor†</td><td>2</td><td>45.48</td><td>21.99</td><td>0.93</td><td>39.02</td></tr></table>

undercounting. Small adaptation-module overhead should not be confused with the total cost of each additional frozen-encoder stream.

![](images/133c46c438f89c5e7bacc48f400c16b1d9462aa595ddc6a445c4fc980124ed89.jpg)  
Figure 5: Cosine similarity matrices of modality-specific LoRA updates for the Q-projection (top row) and V-projection (bottom row) across different transformer layers. Each heatmap shows the pairwise cosine similarity between the LoRA updates ∆W of the four modalities of DELIVER. Across all layers, different modalities exhibit consistently low pairwise cosine similarity, indicating that the LoRA updates for each modality occupy distinct and nearly orthogonal subspaces. The indices 0, 1, 2, and 3 correspond to the four input modalities.

## E Methodological Justification

## E.1 Rationale for Using Modality-Specific LoRA in Self-Attention Adaptation.

To examine whether self-attention should apply a fused LoRA module across modalities or maintain modality-specific adaptation paths, we analyze the learned LoRA updates ∆W across both the q and v projections for each modality on two datasets, DELIVER and MCubeS. For each projection, we also evaluate shallow, mid, and deep transformer layers and report cosine-similarity matrices.

As shown in Fig. 5 and Fig. 6, across all layers and for both projections, the LoRA updates of different modalities exhibit consistently low pairwise cosine similarity, forming near-orthogonal subspaces rather than converging toward a shared direction. This indicates that each modality learns distinct and highly specialized correction directions within the attention mechanism.

Low pairwise similarity supports distinct sensor-specific correction directions, but does not by itself establish that routing them is harmful. The all-path routing comparison in Table 7 tests that choice directly. Keeping Q/V private retains each sensor’s attention rule while the FFN residual path permits cross-modal communication.

## E.2 Comparison with Mixture-of-Experts (MoE)

Each FuseAdapt low-rank branch has a fixed source-sensor identity, while receiver-conditioned routing weights vary by token and input. All available sensor branches participate without top-k selection or a load-balancing loss; absent sensors are masked. This distinguishes branch assignment from dynamic communication and does not imply fixed mixing weights. A sparse MoE-style LoRA control obtains 54.94 versus 57.62 mIoU on MCubeS. The all-path and dense-feature controls in Table 7 more directly test which activations should cross sensor streams.

## E.3 Details of PMM.

After the full-input warm-up, with probability 0.5 we retain all modalities; otherwise we draw independent Bernoulli-0.5 visibility masks and enforce a nonempty visible set per sample. A target is sampled only when at least one modality is missing. The teacher always receives the full input;

![](images/f7f08dfe84377a552313f6854c5c693346930b92e6f025ceb61afc51882d333a.jpg)  
Figure 6: Cosine similarity matrices of modality-specific LoRA updates for the Q-projection (top row) and V-projection (bottom row) across different transformer layers. Each heatmap shows the pairwise cosine similarity between the LoRA updates $\Delta W$ of the four modalities in MCubeS. As with DELIVER, the cross-modal LoRA directions remain consistently low in similarity across all layers. The indices 0, 1, 2, and 3 correspond to the four input modalities.

its designated target-modality representation is detached. Modality-specific query sets attend to visible features and produce the same number of tokens as the target. For full-input samples, the missing-target PMM term is zero. Inference uses the actually available subset: all modalities for MCSS and the evaluated subset for MISS. The teacher and predictor are discarded.

## E.4 Class-balanced OHEM Cross-Entropy Loss.

To better handle class imbalance and focus optimization on informative pixels, we adopt a weighted Online Hard Example Mining (OHEM) strategy with per-batch adaptive class weights, based on the vanilla OHEM implemented in DELIVER. Given the prediction $\mathbf { P } \in \mathbb { R } ^ { B \times C \times H \times W }$ and ground-truth labels $\mathbf { Y } \in \mathbb { R } ^ { B \times \dot { H } \times W }$ , the pixel-wise cross-entropy loss is first computed as:

$$
\mathcal { L } _ { \mathrm { C E } } ( i ) = - \sum _ { c = 1 } ^ { C } w _ { c } \cdot { \bf 1 } [ y _ { i } = c ] \cdot \log P _ { i , c } ,\tag{13}
$$

where $w _ { c }$ denotes the class-specific weight for class $c ,$ computed per batch using the inverse frequency of observed labels, clipped to a maximum value of $w _ { \mathrm { m a x } } \colon$

$$
w _ { c } = \{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } ( \frac { 1 } { f _ { c } + \epsilon } / \operatorname* { m i n } _ { c ^ { \prime } } \frac { 1 } { f _ { c ^ { \prime } } + \epsilon } , ~ w _ { \mathrm { m a x } } ) , } & { \mathrm { i f ~ } f _ { c } > 0 } \\ { w _ { \mathrm { d e f a u l t } } , } & { \mathrm { o t h e r w i s e } } \end{array}\tag{14}
$$

where $f _ { c }$ is the pixel count of class c in the current batch and $w _ { \mathrm { d e f a u l t } }$ is the fallback weight for absent classes. $w _ { \mathrm { m a x } }$ is set to 10, and $w _ { \mathrm { d e f a u l t } }$ is set to 1.

To prioritize informative samples, we follow OHEM and select pixels with loss greater than a threshold τ (converted to logit space $\arg - \log \tau )$ . Let H be the set of such hard pixels (τ is set to 0.7):

$$
\mathcal { H } = \{ i \mid \mathcal { L } _ { \mathrm { C E } } ( i ) > - \log \tau \}\tag{15}
$$

$\mathrm { I f } \ | \mathcal { H } | < N _ { \mathrm { m i n } } .$ , we select the top $N _ { \mathrm { m i n } }$ hardest pixels by loss. The final loss is the mean over these hard pixels:

$$
\mathcal { L } _ { \mathrm { O H E M } } = \frac { 1 } { \vert \mathcal { H } ^ { \prime } \vert } \sum _ { i \in \mathcal { H } ^ { \prime } } \mathcal { L } _ { \mathrm { C E } } ( i ) ,\tag{16}
$$

where $\mathcal { H } ^ { \prime } = \mathcal { H } \mathrm { ~ i f ~ } | \mathcal { H } | \geq N _ { \operatorname* { m i n } }$ , otherwise it is set to the top $N _ { \mathrm { m i n } }$ pixels. Both the loss weights of the segmentation and PMM are set to 1.

Algorithm 1 PMM training after full-input warm-up   
1: Sample nonempty visible set V using the full/masked mixture above.   
2: Compute student features with routing normalized over V and segmentation loss $\mathcal { L } _ { \mathrm { s e g } } .$   
3: Set $\begin{array} { r } { \bar { \mathcal { L } } _ { \mathrm { p m m } } = 0 . } \end{array}$   
4: $\mathbf i \mathbf f \mathcal { V } \dot { \ne } { \mathcal M }$ then   
5: Sample $t \in \mathcal { M } \backslash \mathcal { V }$ uniformly.   
6: Compute detached target $y _ { t }$ with the full-context EMA teacher.   
7: Predict $\hat { z } _ { t } = g _ { \phi } ( Q _ { t } , \hat { \mathbf { X } } _ { \mathcal { V } } )$ and compute cosine loss $\mathcal { L } _ { \mathrm { p m m } }$   
8: end if   
9: Update student and predictor using $\mathcal { L } _ { \mathrm { s e g } } + \mathcal { L } _ { \mathrm { p m m } } .$   
10: Update teacher: $\theta _ { \tau } \stackrel { - } {  } 0 . 9 9 9 \theta _ { \tau } + \bar { 0 } . 0 0 \breve { 1 } \theta _ { s }$

## F Implementation Details

We train the proposed method on 8 A5000 GPUs for DELIVER, 4 for MUSES, and 4 for MCubeS. Initial learning rates are $3 . 5 \mathrm { e } ^ { - 4 }$ (MCubeS), 4e<sup>−4</sup> (MUSES) and $4 \mathrm { e } ^ { - 4 }$ (DELIVER), with a 10-epoch warm-up at 0.1×. Training is conducted for 500, 200, and 200 epochs on MCubeS, DELIVER, and MUSES respectively. Each run begins with a warm-up stage (200 epochs for MCubeS and 15 epochs for both DELIVER and MUSES), during which the model is trained using all modalities without any masking. We use AdamW with batch sizes of 1 (DELIVER) and 4 (MUSES, MCubeS) per GPU. Input resolutions are 1024×1024 (DELIVER, MUSES) and 512×512 (MCubeS), with standard augmentations: random resizing (0.5–2.0), flipping, color jitter, blur, and cropping. FuseAdapt uses rank 8 for DELIVER and MCubeS, and 4 for MUSES. DINOv2-B (for DELIVER and MCubeS) [Oquab et al., 2023] and DINOv2-S (MUSES) serve as the frozen VFMs. Standard LoRA with the low-rank matrices A and B such that A is initialized with a small random Gaussian $( \mathrm { e . g . , } \mathcal { N } ( 0 , 0 . 0 1 ) )$ and B is initialized as zero, ensuring the initial weight perturbation is zero. The number of learnable tokens in the predictor is the same as the sequence length of the target feature.

## G More Ablation Studies

Rank r We evaluate the performance of the proposed method with different rank r. The experiments are conducted on MCubeS on the MCSS setting. As shown in Tab. 16, FuseAdapt significantly improves performance over a frozen VFM, with the best results obtained at rank 8.

Table 16: Results of different rank r under MCSS setting on MCubeS. All models are trained with four modalities.
<table><tr><td>r</td><td>0</td><td>4</td><td>8</td><td>16</td></tr><tr><td>RADN</td><td>51.16</td><td>55.86</td><td>57.62</td><td>57.03</td></tr></table>

Number of Layers Equipped with FuseAdapt LoRAs We further evaluate the impact of the insertion frequency of FuseAdapt LoRAs in the VFM, where a module is inserted every N layers. A larger N indicates fewer inserted LoRAs. The experiments are conducted on MCubeS under the MCSS setting. As shown in Tab. 17, inserting more FuseAdapt LoRAs modules yields greater performance improvements.

Routing Fusion in FFNs We introduce independent, modality-specific routing modules at the FFN stage to enable controlled and dynamic cross-modal fusion. Unlike attention layers that focus on intra-modal context, these routing blocks refine each modality’s representation by selectively integrating complementary signals from others, based on the current context. This fusion occurs after independent encoding, preventing early entanglement and allowing the model to adaptively weight modalities depending on availability, especially beneficial under modality masking. Importantly, each modality has its own dedicated routing layer, which maintains structural disentanglement and minimizes interference from masked or undertrained modalities.

Table 17: Results of inserting FuseAdapt LoRAs every N layers under MCSS setting on MCubeS. All models are trained with four modalities.
<table><tr><td>N</td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>RADN</td><td>51.16</td><td>57.62</td><td>56.38</td><td>55.87</td></tr></table>

The routing module adopts a lightweight noisy gating mechanism, adapted from sparsely activated Mixture-of-Experts (MoE) designs. Given hidden states, the router outputs modality-specific routing scores through a linear layer, and injects learned Gaussian noise via a second linear layer followed by softplus. The input size matches the feature dimension of the VFM output, while the output size corresponds to the number of training modalities. The resulting noisy logits enable soft, differentiable expert selection. Routers are trained end-to-end as part of the main optimization, without any additional loss terms.

We evaluate the effectiveness of routing mechanisms. As shown in Table 18, incorporating routing fusion into the FFNs consistently improves performance. When no fusion is applied (“without routing”), the model processes each modality independently and therefore cannot benefit from cross-modal interactions. Using a simple average fusion yields only marginal gains, as it assumes that all modalities contribute equally and ignores their heterogeneous characteristics. In contrast, routing fusion adaptively selects and combines modality-dependent FFN experts, leading to a clear improvement (from 56.09 to 57.62 mIoU). This demonstrates that learned routing effectively captures the distinct roles that different modalities play in multi-modal feature refinement. We also observe that a shared router yields a competitive gain (57.24 mIoU), showing that routing is helpful even without modality-specific specialization.

Notably, unlike Mixture of Experts structures that perform token-level dynamic expert selection (e.g., top-K gating), our routing deterministically assigns each LoRA expert to a specific modality. All available modality branches participate without top-k selection or a load-balancing objective. Mixing weights remain token-dependent, and absent sources are masked before normalization.

Table 18: Results of various kinds of routing strategies under MCSS setting on MCubeS. All models are trained with four modalities.
<table><tr><td></td><td>wo. routing</td><td>average</td><td>shared router</td><td>w. routing</td></tr><tr><td>RADN</td><td>56.09</td><td>56.43</td><td>57.24</td><td>57.62</td></tr></table>

Additional VFMs To assess backbone sensitivity, we evaluate our method with two alternative VFMs, EVA-CLIP-B [Sun et al., 2023] and NeCo-B [Pariza et al., 2024], on MCubeS (in MCSS setting with full-modality training). As shown in Table 19, the proposed approach maintains strong performance across all backbones:

Table 19: Results of applying various VFMs under MCSS setting on MCubeS. All models are trained with four modalities.
<table><tr><td></td><td>EVA-CLIP</td><td>NeCo</td><td>DINOv2</td></tr><tr><td>RADN</td><td>56.02</td><td>56.28</td><td>57.62</td></tr></table>

Scalability of FuseAdapt We analyze the scalability of FuseAdapt under different backbones and input resolutions.

For DINOv2-B with the input size of 512×512 and rank of 8, adding an additional modality increases the parameter count by 1.11M, which corresponds to 1.12% of the full model. The memory cost,

including optimizer states under AdamW (in float32 type and without AMP), is 16.90 MB, and the computational cost is 4.8 GFLOPs, representing only 0.55% of the total model FLOPs (873.96 GFLOPs).

For DINOv2-S with the input size of 1024×1024 and rank of 4, each additional modality introduces 0.29M parameters (1.14% of the full model), 4.50 MB of memory, and 18.12 GFLOPs, which amount to 2.01% of the overall FLOPs (902.76 GFLOPs).

Although the cost of FuseAdapt grows linearly with the number of modalities, the relative overhead on parameters, memory, and computation remains very small. These figures describe adaptation-module overhead, not the total cost of processing an additional sensor through the frozen encoder. End-to-end deployment measurements are reported in Appendix D.