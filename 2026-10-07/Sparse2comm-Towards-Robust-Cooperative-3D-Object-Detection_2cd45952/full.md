# Sparse2comm: Towards Robust Cooperative 3D Object Detection

Lei Yang, Boqi Li, Chunmian Lin, Li Wang, Ziying Song, Shaoqing Xu, Heye Huang, Haibao Yu, Chen Lv

Abstract—Cooperative perception improves autonomous driving by sharing complementary observations among vehicles and roadside infrastructure for 3D object detection. However, practical deployment is constrained by limited bandwidth and unreliable cooperation, where packet loss, transmission delay, and spatial misalignment jointly degrade the cooperative feature stream. Existing methods often reduce communication cost or compensate for one degradation type, leaving coupled disturbances insufficiently addressed. To address this problem, we propose Sparse2comm, a bandwidth-efficient and robust cooperative 3D object detection framework that treats unreliable cooperation as progressive restoration over degraded cooperative features. Sparse Feature Encoding first encodes communication as randomly mask-sampled foreground features transmitted by collaborating agents, from which the ego vehicle reconstructs dense semantic representations. This sparse-to-dense mechanism learns to infer, missing obiect-centric content from sparse observations, enabling ultra-low-bandwidth communication and packet-loss recovery within the same representation. On the semantically restored features, Latency-Aware Alignment predicts motion flow to compensate delayed messages, and Self-Calibrating Fusion estimates residual spatial offsets in a self-supervised manner before adaptive cross-agent fusion. Sparse2comm therefore restores semantic completeness, temporal consistency, and spatial alignment in an ordered pipeline. Extensive experiments on DAIR V2X, OpenV2V, and V2V4Real show that Sparse2comm maintains competitive clean accuracy and consistently improves robustness under individual and mixed real-world degradations. Compared with the selective feature communication baseline Where2comm, Sparse2comm improves mixed-setting AP@0.5/AP@0.7 by +20.15/+11.79, +12.66/+11.07, and +15.36/+12.61 on the three datasets, respectively. The source code is available at https://github.com/yanglei18/Sparse2comm.

Index Terms—Autonomous Driving, Cooperative Perception, 3D Object Detection, Robustness

## 1 INTRODUCTION

Cooperative perception has emerged as a key paradigm for enhancing autonomous driving safety, particularly in complex urban scenes where occlusion, restricted fields of view, and limited sensing range constrain ego-vehicle perception. By enabling an ego vehicle to exchange complementary observations with collaborating agents, including nearby vehicles and roadside units (RSUs), cooperative perception provides a more complete understanding of the surrounding environment and improves the reliability of 3D object detection in safety-critical scenarios [1], [2], [3], [4], [5], [6]. However, moving from this promise to real-world deployment requires cooperative features to be both bandwidthefficient and robust to unreliable communication and calibration. Current C-V2X links typically provide less than 10 Mbps [7], making it impractical to transmit raw point clouds or dense feature maps at high frame rates. Although feature compression and object-level communication reduce data volume [8], [9], [10], [11], excessive compression may remove discriminative semantic cues and degrade detection reliability. Beyond this bandwidth constraint, cooperative messages are further affected by three major robustness degradations. Packet loss corrupts transmitted features under interference, congestion, and unstable wireless links [6]; even a 5% packet loss can reduce cooperative perception accuracy by nearly 37% [12]. Transmission delay introduces temporal asynchrony between collaborator-side and egovehicle observations, especially for moving objects whose locations change rapidly across frames [13], [14], [15]. Spatial misalignment arises from GNSS multipath effects, sensor drift, and calibration errors, causing collaborator features to be projected to inaccurate positions in the egovehicle coordinate frame [16]. These factors often co-occur in complex scenes and jointly degrade the cooperative feature stream before fusion, making received messages sparse, incomplete, temporally stale, and spatially biased.

Existing studies have made important progress on individual aspects of this problem. For bandwidth constraints, selective communication methods transmit only informative regions or compact object representations. For example, Where2comm [8] uses spatial confidence maps to share critical feature regions, while V2X-PC [9] clusters points into compact packages to preserve object structure. For packet loss, recent works explore feature recovery and historyaided reconstruction. V2X-INCOP [17], for instance, leverages temporal redundancy to restore corrupted or missing cooperative messages. For temporal asynchrony, trajectory or flow-based methods compensate delayed messages using historical context [13], [14], [15]. For spatial misalignment and noisy cooperation, alignment-based and robust communication approaches refine relative poses, aggregate motionaware features, or perform uncertainty-aware asynchronous fusion [18], [19], [20]. However, most existing methods are designed around either communication reduction or a specific degradation type. In real deployments, these factors are coupled: aggressive compression can amplify the impact of packet loss, delayed messages can worsen spatial inconsistency for moving objects, and inaccurate poses can make otherwise informative features harmful during fusion. Therefore, robust cooperative perception should not treat communication efficiency, packet-loss recovery, temporal alignment, and spatial calibration as isolated remedies. It requires an ordered restoration process that progressively recovers semantic completeness, temporal consistency, and spatial alignment from the same degraded feature stream.

![](images/8311181cccc22bace08089630b8aa30165aa63bbb8076bdaf1247c3a3e3b4d0d.jpg)  
Fig. 1. Illustration of real-world challenges in cooperative perception. (a) Ideal aligned point cloud: ideal scenario with accurately synchronized and aligned data between ego and collaborating sensors. (b) Noisy misaligned point cloud: real-world distortions caused by (left) packet loss, (middle) transmission delay, and (right) localization errors, which jointly lead to incomplete or misaligned spatial representations. (c) Performance degradation: standard methods suffer significant accuracy drops under these noisy conditions, whereas our proposed Sparse2comm maintains robust cooperative perception under bandwidth, latency, and spatial uncertainties.

To address this gap, we propose Sparse2comm, a bandwidth-efficient and robust cooperative 3D object detection framework that treats unreliable cooperation as progressive restoration over degraded cooperative features. In this formulation, communication loss, temporal asynchrony, and spatial misalignment are not treated as independent disturbances. Instead, the received cooperative message is regarded as a degraded observation of an underlying cooperative feature state: it is sparse due to bandwidth constraints, incomplete due to packet loss, temporally stale due to transmission delay, and spatially biased due to localization or calibration errors. The objective is to recover a compact, semantically complete, temporally synchronized, and spatially calibrated cooperative representation before final detection. To instantiate this restoration paradigm, Sparse2comm organizes Sparse Feature Encoding, Latency-Aware Alignment, and Self-Calibrating Fusion into a dependent sparse-to-aligned restoration pipeline. Sparse Feature Encoding (SFE) first encodes communication as randomly mask-sampled foreground features transmitted by collaborating agents, from which the ego vehicle reconstructs dense semantic representations. By training the ego vehicle to infer missing object-centric content from sparse foreground observations, SFE turns bandwidth reduction and packet-loss recovery into the same sparse-to-dense restoration problem.

Given the semantically restored features from SFE, Latency-Aware Alignment (LAA) predicts motion flow from temporal collaborator features to compensate for transmission delay and synchronize cooperative features with the egovehicle timestamp. This second stage is placed after SFE because temporal alignment is more reliable when objectcentric semantic content has been partially recovered. After temporal synchronization, Self-Calibrating Fusion (SCF) estimates residual spatial offsets in a self-supervised manner and refines adaptive cross-agent fusion. Placing SCF after LAA helps separate pose-induced offsets from temporal staleness, allowing final fusion to operate on both temporally synchronized and spatially calibrated features. These stages are not independent add-ons: SFE restores semantic completeness from sparse and lossy observations, LAA restores temporal consistency on recovered features, and SCF restores spatial alignment after temporal synchronization.

Extensive experiments on DAIR-V2X [1], OpenV2V [21], and V2V4Real [22] demonstrate that Sparse2comm maintains competitive clean accuracy and generalizes across vehicle–infrastructure, simulated vehicle–vehicle, and realworld vehicle–vehicle cooperative perception scenarios. Under the mixed robustness setting with packet loss, transmission delay, and spatial misalignment, Sparse2comm improves over the strongest prior methods by +15.75/+9.37, +7.13/+4.11, and +11.11/+7.34 AP@0.5/AP@0.7 on the three datasets, respectively. Compared with the representative selective communication baseline Where2comm [8], the corresponding mixed-setting gains are +20.15/+11.79, +12.66/+11.07, and +15.36/+12.61 AP@0.5/AP@0.7, while Sparse2comm uses only 1.0% relative feature bandwidth. These results show that Sparse2comm provides a strong trade-off among detection accuracy, robustness, and communication efficiency.

## The main contributions are summarized as follows:

We formulate bandwidth-limited and unreliable cooperative 3D object detection as progressive restoration over degraded cooperative features across vehicle–infrastructure and vehicle–vehicle settings, enabling robust cooperative perception with only 1.0% relative feature bandwidth compared with dense feature-level communication.

• We design a dependent SFE–LAA–SCF restoration pipeline: SFE transmits randomly mask-sampled foreground features and reconstructs dense semantic representations for ultra-low-bandwidth communication and packet-loss recovery, LAA restores temporal consistency, and SCF restores spatial alignment before adaptive fusion.

We provide comprehensive evaluations on DAIR-V2X [1], OpenV2V [21], and V2V4Real [22] under clean and robustness-critical settings, showing that Sparse2comm generalizes across diverse cooperative perception scenarios. Under mixed degradations, Sparse2comm improves over Where2comm [8] by +20.15/+11.79, +12.66/+11.07, and +15.36/+12.61 AP@0.5/AP@0.7 on the three datasets, respectively, while maintaining a compact 1.0% relative feature bandwidth.

## 2 RELATED WORKS

## 2.1 Ego-centric 3D Object Detection

Accurate 3D object detection is a fundamental capability for autonomous driving. Existing ego-centric detectors are commonly developed around sensing modality and spatial representation. Vision-based methods [23], [24], [25] estimate 3D object geometry from monocular or multi-view images, offering a low-cost perception solution but remaining sensitive to depth ambiguity, illumination changes, and adverse weather. Recent camera-based BEV frameworks [26], [27], [28], [29] lift image features into a unified bird’s-eyeview space, which makes cross-view spatial reasoning more explicit. LiDAR-based detectors [30], [31], [32], represented by VoxelNet [30], SECOND [31], and PointPillars [32], exploit accurate geometric measurements through voxel or pillar representations. Multi-modal detectors [33], [34], [35] further combine LiDAR and camera cues to integrate appearance and geometry.

Although these ego-centric detectors substantially improve local perception, they still operate under a singleplatform sensing assumption. Objects hidden by occlusion, distant traffic participants, and complex intersection layouts can remain unobservable from one viewpoint, regardless of the local detector architecture. These limitations motivate cooperative perception, where vehicles, roadside units, and other collaborating agents exchange complementary observations to extend the effective perception range and reduce blind spots. This shift also introduces communication and reliability constraints that are absent in purely ego-centric detection.

## 2.2 Cooperative Perception

Cooperative perception has developed rapidly with the emergence of V2X benchmarks, including simulated datasets such as V2X-Sim [36] and OPV2V [21], systemlevel simulation platforms for collaborative autonomous driving [5], real-world vehicle–infrastructure datasets such as DAIR-V2X [1] and V2X-Seq [4], and recent real-world V2V/V2X or multi-modal datasets [22], [37], [38]. These benchmarks have shifted the field from isolated ego perception to collaborative scene understanding, and they also expose practical constraints caused by limited bandwidth, unreliable transmission, and imperfect localization.

Existing cooperative methods are commonly organized into early, late, and intermediate fusion. Early fusion preserves rich raw observations but incurs heavy communication, whereas late fusion is efficient but discards dense spatial evidence before cross-agent reasoning. Intermediate fusion exchanges feature representations and has become a practical balance between accuracy and bandwidth. Representative methods include feature-level fusion and message-passing approaches [39], [40], [41], distillation or transformer-based cooperative fusion [42], [43], [44], [45], and communication-aware variants [8], [10], [11], [46], [47], [48]. Most of these methods are evaluated under relatively reliable feature exchange, where received messages are assumed to be usable after transmission.

Communication efficiency is a key requirement for practical cooperative perception over bandwidth-limited V2X links. Selective, pragmatic, and object-aware methods reduce transmitted data by sharing informative regions, compact object messages, or hybrid intermediate-late representations [8], [9], [11], [47], [48], [49]. These methods show that sparse communication can greatly reduce transmission cost. However, sparsification and packet loss both remove feature content from the cooperative stream, so message compactness and message recoverability are tightly coupled. This coupling suggests that communication-efficient representations should also be recoverable under missing observations.

Robustness to communication and calibration errors is another important direction. Prior works have studied lossy or interrupted communication [12], [17], [50], transmission delay and temporal asynchrony [13], [14], [15], and spatial misalignment caused by localization or pose errors [16], [18]. More recent robust cooperative perception methods further explore motion-aware feature aggregation and uncertaintyaware asynchronous fusion [19], [20]. Most of these methods target a specific degradation type. In practice, these disturbances often occur together, so robustness needs to be examined under packet loss, delay, spatial misalignment, and their coupled setting rather than as isolated cases.

## 3 SPARSE2COMM FRAMEWORK

## 3.1 Problem Definition

Cooperative perception aims to improve ego-vehicle 3D object detection by fusing complementary features from collaborating agents, which can be nearby vehicles or roadside units (RSUs), under a communication budget. Formally, the cooperative detection process can be written as

$$
\hat { B } _ { \mathrm { e g o } } = \Psi _ { \mathrm { C o o p . } } ^ { \omega } \left( F _ { v } ^ { t _ { v } } , \left\{ F _ { i } ^ { t _ { i } } \right\} _ { i = 1 } ^ { N } \right) \mathrm { , ~ s . t . ~ } \sum _ { i = 1 } ^ { N } \Vert F _ { i } ^ { t _ { i } } \Vert \leq C ,\tag{1}
$$

where $\Psi _ { \mathrm { C o o p . } } ^ { \omega }$ denotes the cooperative perception model parameterized by $\omega ,$ and $\hat { B } _ { \mathrm { e g o } }$ is the set of predicted 3D bounding boxes for the ego vehicle. $F _ { v } ^ { t _ { v } }$ denotes the egovehicle feature at timestamp $t _ { v } ,$ while $\bar { \boldsymbol { F } } _ { i } ^ { t _ { i } }$ denotes the feature transmitted by the i-th collaborating agent at timestamp $t _ { i } .$ N is the number of collaborating agents, and C is the communication bandwidth constraint that limits the aggregate transmitted feature volume. The transmission delay between the collaborating agent and the ego vehicle is denoted as $\tau = t _ { v } - t _ { i }$ . In real deployments, the received cooperative features are further degraded by packet loss, temporal asynchrony, and spatial misalignment. Sparse2comm therefore aims to recover a compact, semantically complete, temporally synchronized, and spatially calibrated cooperative representation before final detection.

![](images/4d3b73b7241e505a6f734b1260511f662a6bb02a759103fb0367fd61b9571877.jpg)  
Fig. 2. The overall framework of Sparse2comm. The collaborator and ego encoders extract BEV features from LiDAR point clouds. Sparse Feature Encoding transmits masked sparse features and reconstructs complete semantics to minimize communication cost and resist packet loss. Latency-Aware Alignment predicts motion flow to temporally align delayed messages. Self-Calibrating Fusion self-supervisively estimates residual spatial offsets to refine cross-agent spatial alignment. The fused BEV features are finally decoded by a detection head for 3D object detection.

## 3.2 Overall Architecture

Sparse2comm follows a progressive restoration pipeline that sequentially restores semantic completeness, temporal consistency, and spatial alignment before cooperative fusion. The overall pipeline in Fig. 2 starts from collaborator-side feature extraction. The collaborator encoder, built upon a PFE [32] and a ResNet [51] backbone, extracts a sequence of BEV feature maps $\left\{ \boldsymbol { F } _ { i } ^ { t _ { i } - 3 } , \ldots , \boldsymbol { F } _ { i } ^ { t _ { i } } \right\}$ from collaborator-side point clouds $\left\{ P _ { i } ^ { t _ { i } - 3 } , \ldots , P _ { i } ^ { t _ { i } } \right\}$ , where $F _ { i } ^ { t _ { i } } ~ \in ~ \mathbb { R } ^ { X _ { i } \times Y _ { i } \times C _ { i } }$ Here, $X _ { i } , ~ Y _ { i } ,$ , and $C _ { i }$ denote the height, width, and channel dimensions of the collaborator BEV feature, respectively. Based on these features, the RPN predicts ROI boxes $B _ { i } ^ { t _ { i } } \ \stackrel { \cdot } { = } \ \left\{ b _ { o } ^ { t _ { i } } \right\} _ { o = 1 } ^ { O }$ , where O is the number of detected objects. Guided by these ROI boxes, an object-aware filter [8] removes redundant background responses and produces object-foreground BEV features $\left\{ \mathcal { F } _ { i } ^ { t _ { i } - 3 } , \ldots , \mathcal { F } _ { i } ^ { t _ { i } } \right\}$ . The ego encoder, which adopts the same PFE-ResNet backbone, extracts ego-vehicle BEV features $F _ { v } ^ { t _ { v } } \in \mathbb R ^ { X _ { v } \times Y _ { v } \times C _ { \imath } }$ from the ego-vehicle point cloud $P _ { v } ^ { t _ { \tau } }$

The object-foreground feature $\mathcal { F } _ { i } ^ { t _ { i } }$ is first processed by Sparse Feature Encoding. Instead of transmitting dense feature maps, the collaborating agent sends only a masked sparse feature $\mathcal { \widetilde { F } } _ { i } ^ { t _ { i } }$ , and the ego vehicle reconstructs it into $\mathcal { \hat { F } } _ { i } ^ { t _ { i } } \in \mathbb { R } ^ { X _ { i } \times Y _ { i } \times C _ { i } ^ { ' } }$ . Latency-Aware Alignment uses temporal collaborator features to estimate motion flow and warps the recovered feature to the ego-vehicle timestamp, producing $\hat { \mathcal { F } } _ { i } ^ { t _ { v } }$ . The temporally aligned collaborator feature is transformed into the ego-vehicle coordinate frame and refined by Self-Calibrating Fusion, which estimates residual spatial offsets and fuses the calibrated collaborator feature with $F _ { v } ^ { t _ { \tau } }$ . The resulting fused feature $Z _ { v } ^ { t _ { v } } \in \mathbb R ^ { X _ { v } \times Y _ { v } \times C _ { v } }$ is finally decoded by a detection head to predict ${ \hat { B } } _ { \mathrm { e g o } } .$

## 3.3 Sparse Feature Encoding

Sparse Feature Encoding serves as the first restoration stage and provides the core mechanism for both low-bandwidth communication and packet-loss recovery. Different from selective communication methods that transmit selected foreground responses [8], SFE encodes communication as randomly mask-sampled foreground features. The collaborating agent sends only a small subset of object-centric BEV patches, and the ego vehicle reconstructs dense semantic representations from these sparse observations. In this view, bandwidth reduction is not a post-hoc compression step. It becomes a sparse-to-dense reconstruction problem, where intentionally omitted patches and accidentally lost packets are handled as missing foreground observations under the same reconstruction mechanism.

Fig. 3 motivates this sparse communication design: the collaborating agent applies random mask-based sampling to the object-foreground BEV feature $\mathcal { F } _ { i } ^ { t _ { i } }$ and obtains an ultrasparse transmitted representation $\mathcal { \tilde { F } } _ { i } ^ { t _ { i } }$ . With a masking ratio

$$
r = 1 - \frac { \left\| \widetilde { \mathcal { F } } _ { i } ^ { t _ { i } } \right\| } { \left\| \mathcal { F } _ { i } ^ { t _ { i } } \right\| } ,\tag{2}
$$

only about 1% of conventional dense feature-level communication needs to be transmitted. The ego vehicle then reconstructs dense semantic representations $\hat { \mathcal { F } } _ { i } ^ { t _ { i } }$ from the received sparse features. Because the reconstruction objective is trained to infer missing object-foreground content, the same mechanism recovers both intentionally omitted patches and accidentally lost packets, providing robustness under unstable communication.

Sparse Feature Encoding is implemented with two coupled stages, detailed in Fig. 4: Mask Sampling on the collaborator side and Feature Reconstruction on the ego-vehicle side.

![](images/2809eb468db1327a0af0ccd69b93277b1db7b3dbfa496234c923885c81cf8203.jpg)  
Fig. 3. The Motivation of Sparse Feature Encoding. Each object can be reconstructed from partial local features by leveraging a learned database of global priors, enabling complete semantic recovery under limited bandwidth.

## 3.3.1 Mask Sampling

Mask Sampling minimizes transmission cost while preserving essential object-level cues. It takes the object-foreground BEV feature $\mathcal { F } _ { i } ^ { t _ { i } }$ as input and outputs an ultra-sparse feature map $\mathcal { \widetilde { F } } _ { i } ^ { t _ { i } }$ that retains only partial foreground information. Different from global random masking, the proposed strategy restricts sampling to object regions and independently samples visible patches for each object instance. The random object-wise sampling is deliberately used as a missingpattern generator: during training, the reconstruction network observes diverse visible patch subsets and learns to infer object-centric content from incomplete evidence rather than relying on fixed transmitted positions. This object-wise design reduces the risk of removing an entire object and introduces instance-level sparsity variation that improves robustness to communication uncertainty.

![](images/e2588823850d2504fa1640bbeb1d0a16e2f3187ed24eb364d2e2f89501a86c48.jpg)  
Fig. 4. Sparse Feature Encoding. The collaborating agent performs Mask Sampling to transmit sparse object-foreground features under limited bandwidth, while the ego vehicle uses an encoder–decoder network for Feature Reconstruction to recover complete BEV features supervised by L<sub>REC</sub>.

Formally, let $\begin{array} { r c l } { \mathcal { F } _ { i } ^ { t _ { i } } } & { \in } & { \mathbb { R } ^ { X _ { i } \times Y _ { i } \times C _ { i } } } \end{array}$ denote the objectforeground BEV feature after background removal. We divide $\mathcal { F } _ { i } ^ { t _ { i } }$ into $N _ { \mathrm { f g } }$ non-overlapping patches:

$$
\mathcal { P } _ { i } ^ { t _ { i } } = \{ f _ { 1 } , f _ { 2 } , \dotsc , f _ { N _ { \mathrm { f g } } } \} , \quad f _ { j } \in \mathbb { R } ^ { p \times p \times C _ { i } } .\tag{3}
$$

Each patch $f _ { j }$ is associated with an object according to ROI box $b _ { o } ^ { t _ { i } }$ , forming an object-wise partition:

$$
\mathcal { T } _ { o } = \{ j \mid f _ { j } \in b _ { o } ^ { t _ { i } } \} , \quad N _ { o } = | \mathcal { I } _ { o } | , \quad o = 1 , \ldots , O .\tag{4}
$$

For each object $o ,$ we sample a random masking ratio:

$$
r _ { o } \sim \mathcal { U } ( r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ) , \quad 0 < r _ { \operatorname* { m i n } } \le r _ { \operatorname* { m a x } } < 1 ,\tag{5}
$$

and retain $K _ { o }$ visible patches:

$$
K _ { o } = \left\lfloor \left( 1 - r _ { o } \right) \cdot N _ { o } \right\rfloor .\tag{6}
$$

The visible patch subset for object o is sampled as

$$
V _ { o } = \mathrm { S a m p l e } ( \mathcal { I } _ { o } , K _ { o } ) .\tag{7}
$$

The full visible set is obtained by aggregating all object-wise visible subsets:

$$
V = \bigcup _ { o = 1 } ^ { O } V _ { o } .\tag{8}
$$

To unify intentional sparsification and packet loss, we distinguish the sampled visible set from the actually received visible set. During transmission, a subset of the sampled patches may be dropped due to packet loss, denoted as $\mathbf { \hat { V } } _ { \mathrm { d r o p } } \subseteq V$ . The ego vehicle therefore observes

$$
V _ { \mathrm { r e c } } = V \setminus V _ { \mathrm { d r o p } } .\tag{9}
$$

The received ultra-sparse BEV feature map is then constructed as

$$
\widetilde { \mathcal { F } } _ { i } ^ { t _ { i } } = \{ ( f _ { j } , \mathrm { p o s } _ { j } ) \ : | \ : j \in V _ { \mathrm { r e c } } \} .\tag{10}
$$

Accordingly, the missing patch set to be reconstructed is

$$
\mathcal { M } = ( \{ 1 , \dots , N _ { \mathrm { f g } } \} \setminus V ) \cup V _ { \mathrm { d r o p } } ,\tag{11}
$$

which contains both intentionally omitted patches and packet-lost patches. This formulation makes packet loss a natural extension of the same missing-content reconstruction problem used for bandwidth reduction. This design retains object-level evidence under an extremely small communication budget while exposing the reconstruction network to diverse sparsity patterns during training.

## 3.3.2 Feature Reconstruction

Feature Reconstruction is the semantic restoration component of SFE. It restores incomplete object-foreground features caused by intentional mask sampling and packet loss during transmission. Since both effects remove foreground patches from the received observation, they can be recovered by the same sparse-to-dense decoder. It adopts an encoder-decoder architecture that learns a reconstruction prior over object-centric BEV features.

The encoder consists of five ViT [52] blocks. Each received foreground patch is projected by a linear layer, augmented with positional encoding, and processed by the encoder to produce a high-dimensional embedding. Only patches indexed by $V _ { \mathrm { r e c } }$ are forwarded to the encoder, while masked or packet-lost patches are omitted, which keeps the computation compatible with sparse communication.

The decoder consists of two ViT [52] blocks followed by a linear layer. Its input contains the encoder embeddings of visible patches and placeholder embeddings for the missing patch set M, where each missing patch is represented by a shared learnable mask token. After adding positional encodings, the decoder reconstructs the full set of objectforeground patches and rearranges them into the recovered BEV feature map $\hat { \mathcal { F } } _ { i } ^ { t _ { i } }$

The reconstruction process is optimized by minimizing the mean squared error between the recovered feature $\hat { \mathcal { F } } _ { i } ^ { t _ { i } }$ and the original object-foreground feature $\mathcal { F } _ { i } ^ { t _ { i } }$ :

$$
\mathcal { L } _ { \mathrm { R E C } } ( \hat { \mathcal { F } } _ { i } ^ { t _ { i } } , \mathcal { F } _ { i } ^ { t _ { i } } ) = \frac { 1 } { X _ { i } Y _ { i } } \sum _ { m , n } ^ { X _ { i } Y _ { i } } \left[ \hat { \mathcal { F } } _ { i } ^ { t _ { i } } ( m , n ) - \mathcal { F } _ { i } ^ { t _ { i } } ( m , n ) \right] ^ { 2 } .\tag{12}
$$

![](images/006a7bb29ec57289a1c7b141eecdb9d9cf42805cda53a486657a975cc3c494b2.jpg)  
Fig. 5. Latency-aware Alignment. Flow Generator predicts motion flow $\mathcal { V } _ { i } ^ { \tilde { t _ { i } } }$ from temporal BEV features on the collaborator side, which is transmitted to the ego vehicle for Affine-based temporal alignment. The module is trained with flow loss L<sub>FLOW</sub> using future frames to ensure robustness against transmission delay.

In this way, Sparse Feature Encoding restores semantic completeness before any temporal or spatial correction is applied, providing a more reliable feature basis for the subsequent alignment stages.

## 3.4 Latency-Aware Alignment

After SFE restores semantic completeness, the recovered collaborator feature can still be temporally stale because it is generated at the collaborator timestamp rather than the ego timestamp. For moving objects, this temporal asynchrony appears as spatial displacement in the BEV plane and can degrade feature fusion even when the transmitted content has been reconstructed. Latency-Aware Alignment is therefore placed after SFE: it estimates motion flow from recent collaborator features and warps the recovered feature to the ego-vehicle timestamp, restoring temporal consistency before spatial calibration. The module decomposes this process into flow generation and temporal warping, as summarized in Fig. 5.

## 3.4.1 Flow Generator

The Flow Generator predicts pixel-wise motion $\mathcal { V } _ { i } ^ { t _ { i } }$ from a temporal sequence of collaborator BEV features. Given $\{ \mathcal { F } _ { i } ^ { t _ { i } - 3 } , \ldots , \mathcal { F } _ { i } ^ { t _ { i } ^ { \star } } \}$ , we concatenate the features along the channel dimension:

$$
\mathbf { X } _ { \mathrm { f l o w } } = \left[ \mathcal { F } _ { i } ^ { t _ { i } - 3 } , \ldots , \mathcal { F } _ { i } ^ { t _ { i } } \right] .\tag{13}
$$

The Flow Generator follows an encoder-decoder architecture with three downsampling stages, three upsampling stages, and a final convolutional layer. Each downsampling unit contains a convolutional layer, Batch Normalization (BN) [53], and ReLU [54]; each upsampling unit uses a deconvolution layer, BN, and ReLU. The predicted motion flow is

$$
\mathcal { V } _ { i } ^ { t _ { i } } = f _ { \mathrm { f l o w } } ( \mathbf { X } _ { \mathrm { f l o w } } ) , \quad \mathcal { V } _ { i } ^ { t _ { i } } \in \mathbb { R } ^ { X _ { i } \times Y _ { i } \times 2 } ,\tag{14}
$$

where $\mathcal { V } _ { i } ^ { t _ { i } } ( m , n )$ denotes the BEV motion vector at position $( m , n )$

![](images/b8682a9e9b7a76be7bc22ae5e921851ce621985055f449eadb1af2c2c90640c9.jpg)  
Fig. 6. Self-calibrating Fusion. Self-calibrating Alignment predicts pixel-wise offsets $\Delta _ { ( i  v ) }$ via residual and deformable convolutions to correct pose errors, guided by self-supervised loss ${ \mathcal { L } } _ { \mathsf { S C A } } .$ Feature Fusion further integrates aligned features to generate the final BEV representation $Z _ { v } ^ { t _ { v } }$

## 3.4.2 Temporal Alignment

Given the predicted flow, a bilinear warping function $f _ { \mathrm { w a r p } } ( \cdot )$ aligns the collaborator feature to the ego-vehicle timestamp:

$$
\hat { \mathcal { F } } _ { i } ^ { t _ { v } } = f _ { \mathrm { w a r p } } \left( \mathcal { F } ^ { t _ { i } } , \mathcal { V } _ { i } ^ { t _ { i } } \cdot \tau \right) , \quad \mathcal { F } ^ { t _ { i } } = \left\{ \mathcal { F } _ { i } ^ { t _ { i } } , \quad \mathrm { i f ~ t r a i n i n g } , \right.\tag{15}
$$

During training, the clean object-foreground feature $\mathcal { F } _ { i } ^ { t _ { i } }$ provides stable supervision for learning motion flow. During inference, the reconstructed feature $\hat { \mathcal { F } } _ { i } ^ { t _ { i } }$ is warped because only sparse transmitted features are available at the ego vehicle. The flow predictor is supervised by a cosine similarity loss over future collaborator frames:

$$
\mathcal { L } _ { \mathrm { F L O W } } = \sum _ { t \in \{ t _ { i } + \Delta t , \ldots , t _ { i } + n \Delta t \} } \left( 1 - \frac { \mathcal { F } _ { i } ^ { t } \odot \hat { \mathcal { F } } _ { i } ^ { t } } { \left\| \hat { \mathcal { F } } _ { i } ^ { t } \right\| _ { 2 } \cdot \| \mathcal { F } _ { i } ^ { t } \| _ { 2 } } \right) ,\tag{16}
$$

where $\Delta t$ denotes the sampling interval and n is the number of future frames used for temporal supervision.

## 3.5 Self-Calibrating Fusion

After temporal consistency has been restored, the remaining feature offset mainly comes from imperfect relative pose estimation, GNSS multipath effects, sensor drift, or calibration errors. Separating temporal alignment from spatial calibration is important because both delay-induced motion and pose-induced bias appear as spatial displacement in the BEV plane. Let

$$
\hat { \mathcal { F } } _ { v } ^ { t _ { v } } = \mathbf { T } _ { i } ^ { v } \left( \hat { \mathcal { F } } _ { i } ^ { t _ { v } } \right)\tag{17}
$$

denote the temporally aligned collaborator feature transformed into the ego-vehicle coordinate frame by the relative pose $\mathbf { T } _ { i } ^ { v }$ . If $\mathbf { T } _ { i } ^ { v }$ is inaccurate, $\hat { \mathcal { F } } _ { v } ^ { t }$ <sup>v</sup> becomes spatially misaligned with the ego-vehicle feature $F _ { v } ^ { t _ { v } }$ . Self-Calibrating Fusion restores spatial alignment by learning feature-space residual offset calibration in a self-supervised manner before adaptive feature fusion. In contrast to directly correcting the global pose, SCF estimates local residual offsets between cooperative features, which makes it suitable for compensating fine-grained feature misalignment after temporal synchronization. Its two stages, Self-Calibrating Alignment and Adaptive Feature Fusion, are illustrated in Fig. 6.

## 3.5.1 Self-Calibrating Alignment

Self-Calibrating Alignment estimates pixel-wise residual offsets between the source collaborator feature fed to the calibration module and the ego-vehicle feature. To avoid mixing the training and inference inputs, we denote this source feature as $\mathcal { \hat { F } } _ { v } ^ { t _ { v } }$ : during self-supervised training, it is the noisy transformed feature defined below; during inference, it is the collaborator feature transformed by the available relative pose. The two feature maps are first concatenated along the channel dimension:

$$
\mathbf { X } _ { \mathrm { s c a } } = \left[ \bar { \mathcal { F } } _ { v } ^ { t _ { v } } , F _ { v } ^ { t _ { v } } \right] .\tag{18}
$$

The concatenated feature is processed by a residual block and a deformable convolution layer to predict the offset field:

$$
\begin{array} { r } { \Delta _ { i  v } = f _ { \mathrm { D C N } } ( f _ { \mathrm { r e s i d u a l } } ( \mathbf { X } _ { \mathrm { s c a } } ) ) , } \end{array}\tag{19}
$$

where $\Delta _ { i  v }$ represents the pixel-wise residual displacement from the collaborator feature to the ego-vehicle feature. The offset field is then applied to the transformed collaborator feature through bilinear sampling:

$$
H _ { v } ^ { t _ { v } } ( p _ { k } ) = \bar { \mathcal { F } } _ { v } ^ { t _ { v } } ( p _ { k } + \Delta _ { i  v } ( p _ { k } ) ) , \quad 0 \leq k \leq X _ { v } Y _ { v } ,\tag{20}
$$

where $p _ { k } \in \mathbb { R } ^ { 2 }$ denotes the BEV pixel coordinate and $H _ { v } ^ { t _ { v } }$ is the spatially calibrated collaborator feature.

To train this residual calibration without manual offset annotations, we introduce a self-supervised perturbation strategy. During training, the clean relative pose $\mathbf { T } _ { i } ^ { v }$ is perturbed by a synthetic global transformation $\mathbf { T } _ { \mathrm { n o i s y } }$ to simulate localization and calibration errors. The noisy transformed feature is

$$
\begin{array} { r } { \hat { \mathcal { F } } _ { v , \mathrm { n o i s y } } ^ { t _ { v } } = \mathbf { T } _ { \mathrm { n o i s y } } \mathbf { T } _ { i } ^ { v } \left( \hat { \mathcal { F } } _ { i } ^ { t _ { v } } \right) , } \end{array}\tag{21}
$$

During self-supervised training, we set $\bar { \mathcal { F } } _ { v } ^ { t _ { v } } ~ = ~ \hat { \mathcal { F } } _ { v , \mathrm { n o i s y } } ^ { t _ { v } }$ in $\mathbf { X } _ { \mathrm { s c a . } }$ , while the clean transformed feature $\hat { \mathcal { F } } _ { v } ^ { t _ { v } }$ serves as the reconstruction target. This perturb-and-recover objective teaches the offset predictor to remove residual spatial bias without requiring manually annotated calibration errors. The alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { S C A } } = \frac { 1 } { X _ { v } Y _ { v } } \sum _ { m , n } ^ { X _ { v } Y _ { v } } \left[ H _ { v } ^ { t _ { v } } ( m , n ) - \hat { \mathcal { F } } _ { v } ^ { t _ { v } } ( m , n ) \right] ^ { 2 } .\tag{22}
$$

At inference time, no synthetic perturbation is added, and $\bar { \mathcal { F } } _ { v } ^ { t _ { v } }$ is the transformed collaborator feature obtained from the deployed relative pose. When the deployed relative pose contains residual error, the learned offset field automatically compensates for spatial deviations and improves geometric consistency before fusion.

## 3.5.2 Adaptive Feature Fusion

After semantic restoration, temporal alignment, and spatial calibration, the calibrated collaborator feature $H _ { v } ^ { t _ { v } }$ and the ego-vehicle feature $F _ { v } ^ { t _ { v } }$ are fused through a residual channel-attention module. We first concatenate the two inputs:

$$
{ \bf X } _ { \mathrm { f u s e } } = \left[ H _ { v } ^ { t _ { v } } , F _ { v } ^ { t _ { v } } \right] .\tag{23}
$$

The concatenated feature is refined by a residual block:

$$
\begin{array} { r l } & { \mathbf { Y } = \operatorname { R e L U } \left( \operatorname { B N } \left( W _ { 1 } \mathbf { X } _ { \mathrm { f u s e } } \right) \right) , } \\ & { F = f _ { \mathrm { r e s i d u a l } } ( \mathbf { X } _ { \mathrm { f u s e } } ) } \\ & { \mathbf { \Lambda } = \operatorname { R e L U } \left( \operatorname { B N } \left( W _ { 2 } \mathbf { Y } \right) + \mathbf { X } _ { \mathrm { f u s e } } \right) . } \end{array}\tag{24}
$$

where Y is the intermediate residual feature, and $W _ { 1 }$ and $W _ { 2 }$ are 3×3 convolution kernels. A squeeze-and-excitation attention layer then reweights feature channels:

$$
f _ { \mathrm { s e } } ( F ) = \sigma \left( W f _ { \mathrm { a v g } } ( F ) \right) \odot F ,\tag{25}
$$

where $f _ { \mathrm { a v g } } ( \cdot )$ denotes global average pooling, W is a learnable weight matrix, $\sigma ( \cdot )$ is the sigmoid function, and ⊙ denotes channel-wise multiplication. The final fused BEV representation is

$$
Z _ { v } ^ { t _ { v } } = f _ { \mathrm { s e } } ( F ) .\tag{26}
$$

This adaptive fusion stage integrates complementary crossagent information after the cooperative feature stream has been made semantically complete, temporally synchronized, and spatially calibrated.

## 3.6 Detection Head and Loss Function

## 3.6.1 Detection Head

Following PointPillars [32], Sparse2comm adopts a singlestage detection head with three branches: classification for object categories, regression for 3D bounding box parameters $( x , y , z , l , w , h , \theta )$ , and direction classification for orientation disambiguation. This lightweight head decodes the fused BEV feature $Z _ { v } ^ { t _ { v } }$ into final 3D bounding boxes.

## 3.6.2 Overall Loss Function

The overall training objective combines the detection loss with auxiliary supervision from the three restoration stages:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { R E C } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { F L O W } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { S C A } } ,\tag{27}
$$

where $\lambda _ { 1 } , \ \lambda _ { 2 } ,$ and $\lambda _ { 3 }$ are balancing weights. $\mathcal { L } _ { \mathrm { d e t } }$ is the 3D detection loss, L<sub>REC</sub> supervises semantic reconstruction in Sparse Feature Encoding, $\mathcal { L } _ { \mathrm { F L O W } }$ supervises temporal motion prediction in Latency-Aware Alignment, and $\bar { \mathcal { L } } _ { \mathrm { S C A } }$ supervises residual spatial calibration in Self-Calibrating Fusion.

## 4 EXPERIMENTS

## 4.1 Datasets

We evaluate Sparse2comm on three cooperative perception datasets: DAIR-V2X [1], OpenV2V [21], and V2V4Real [22]. These datasets cover complementary cooperative settings, including vehicle–infrastructure cooperation, simulated vehicle–vehicle cooperation, and real-world vehicle– vehicle cooperation, allowing us to examine robustness beyond a single benchmark.

DAIR-V2X [1] is a large-scale vehicle–infrastructure cooperative perception dataset collected from real urban scenarios. It contains synchronized vehicle-side and infrastructureside LiDAR-camera data with 3D annotations, and introduces the VIC3D task where detection must account for both temporal asynchrony and communication cost. We use DAIR-V2X as the primary benchmark for evaluating vehicle–infrastructure cooperative 3D object detection under bandwidth and robustness constraints.

TABLE 1: Main comparison with representative cooperative perception methods across datasets. Each performance cell reports AP@0.5 / AP@0.7. “Clean” denotes evaluation without artificial communication or pose perturbation, while “Mixed” denotes the unified robustness setting with packet loss, transmission delay, and spatial misalignment. BW is reported as the relative transmitted feature volume with respect to dense featurelevel cooperation on each dataset, where dense feature transmission is normalized to 100%. Gray entries denote the single-agent No Fusion baseline, while black bold and underlined numbers indicate the best and second-best cooperative-method results in each column, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Category</td><td colspan="2">DAIR-V2X</td><td colspan="2">OpenV2V</td><td colspan="2">V2V4Real</td><td rowspan="2">BW (%)↓</td></tr><tr><td>Clean</td><td>Mixed</td><td>Clean</td><td>Mixed</td><td>Clean</td><td>Mixed</td></tr><tr><td colspan="2">Basic and classic cooperative perception</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Fusion</td><td>Single-agent</td><td>42.45 / 30.78</td><td>42.45 / 30.78</td><td>79.78 / 67.16</td><td>79.78 / 67.16</td><td>40.31 / 22.47</td><td>40.31  / 22.47</td><td>0</td></tr><tr><td>F-Cooper [39]</td><td>Full feature</td><td>49.83 / 34.21</td><td>29.32 / 22.74</td><td>88.15 / 80.97</td><td>68.54 / 55.62</td><td>66.08 / 35.21</td><td>37.41 / 17.42</td><td>100.0</td></tr><tr><td>DiscoNet [42]</td><td>Full feature</td><td>51.31 / 35.77</td><td>33.26 / 27.50</td><td>88.90 / 82.03</td><td>70.26 / 58.34</td><td>65.47  / 35.96</td><td>39.60  / 19.05</td><td>100.0</td></tr><tr><td>V2VNet [40]</td><td>Full feature</td><td>61.71 / 46.25</td><td>33.95 / 21.71</td><td>92.16 / 85.38</td><td>69.58 / 56.71</td><td>67.14 / 36.42</td><td>38.38 / 17.79</td><td>100.0</td></tr><tr><td>V2X-ViT [43]</td><td>Full feature</td><td>58.37  / 42.77</td><td>37.31  / 26.41</td><td>91.43 / 84.28</td><td>73.48  / 61.26</td><td>68.36 / 39.08</td><td>41.77 / 20.99</td><td>100.0</td></tr><tr><td colspan="2">Robust cooperative perception</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoBEVFlow [15]</td><td>Temporal alignment</td><td>66.91  / 51.64</td><td>37.16 / 27.20</td><td>92.23 / 85.83</td><td>80.36 / 69.04</td><td>69.08 / 43.72</td><td>41.81  / 21.16</td><td>5.6</td></tr><tr><td>CoAlign [18]</td><td>Pose alignment</td><td>68.72 / 53.06</td><td>40.76 / 29.68</td><td>92.94 / 86.56</td><td>76.93  / 65.42</td><td>72.71 / 45.04</td><td>44.47  / 24.84</td><td>100.0</td></tr><tr><td>CoDynTrust [20]</td><td>Asynchronous robustness</td><td>67.97  / 52.36</td><td>42.35 / 30.17</td><td>93.28 / 87.05</td><td>81.15 / 70.42</td><td>71.67  / 44.93</td><td>45.07  / 25.81</td><td>10.6</td></tr><tr><td colspan="2">Communication-efficient cooperative perception</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>When2comm [41]</td><td>Adaptive comm.</td><td>54.96 / 36.15</td><td>36.89 /  26.33</td><td>89.47  / 82.15</td><td>72.35 / 60.54</td><td>64.23 / 34.71</td><td>40.49 / 19.59</td><td>100.0</td></tr><tr><td>How2comm [10]</td><td>Pragmatic comm.</td><td>68.76 / 53.08</td><td>41.44 / 29.43</td><td>93.47  / 87.44</td><td>78.25 / 66.87</td><td>71.63 / 45.74</td><td>44.07  / 23.88</td><td>3.4</td></tr><tr><td>CoSDH [48]</td><td>Hybrid fusion</td><td>68.91 / 53.14</td><td>40.62 / 29.01</td><td>94.20 / 88.19</td><td>79.64 / 68.15</td><td>72.42  /  46.09</td><td>43.01  / 22.98</td><td>2.8</td></tr><tr><td>Where2comm [8]</td><td>Selective feature</td><td>67.68 / 52.18</td><td>37.95 / 27.75</td><td>93.00 / 86.97</td><td>75.62 / 63.46</td><td>70.24 / 44.63</td><td>40.82  / 20.54</td><td>5.0</td></tr><tr><td colspan="2">Sparse2comm (Ours) Progressive restoration</td><td>69.42 / 53.78</td><td>58.10 / 39.54</td><td>93.70 / 87.66</td><td>88.28 / 74.53</td><td>71.46  / 45.36</td><td>56.18 / 33.15</td><td>1.0</td></tr></table>

OpenV2V [21] is an open simulated vehicle–vehicle cooperative perception benchmark built with CARLA and OpenCDA. It provides multi-agent sensor observations, accurate poses, and cooperative 3D annotations across diverse traffic scenarios, making it suitable for controlled evaluation of vehicle–vehicle feature fusion. We use OpenV2V to examine whether Sparse2comm generalizes to multi-agent V2V cooperation under a standardized simulation environment. V2V4Real [22] is a large-scale real-world vehicle–vehicle cooperative perception dataset collected by two vehicles equipped with multi-modal sensors. It covers 410 km of driving routes and provides 20K LiDAR frames, 40K RGB images, 240K annotated 3D bounding boxes across five classes, and HD maps. We use its cooperative 3D object detection task to test whether Sparse2comm generalizes to real vehicle–vehicle cooperation with moving agents on both sides.

For all datasets, detection performance is evaluated using Average Precision (AP) at IoU thresholds of 0.5 and 0.7. To assess robustness under practical cooperative perception degradations, we use the same perturbation protocol across methods: (i) packet loss is simulated by randomly masking transmitted features with a ratio $\begin{array} { r } { r \sim \mathcal { U } ( 0 , 0 . 9 ) ; } \end{array}$ (ii) transmission delay is simulated by using delayed cooperative features, corresponding to a latency of 200 ms; and (iii) spatial misalignment is simulated by adding rotational noise $\bar { \Delta } R \sim \mathcal N ( 0 , \bar { 3 } . 0 ^ { \circ } )$ and translational noise $\mathsf { \bar { \Delta } T } \sim \mathcal { N } ( 0 , 0 . 5 )$ m. All compared methods are evaluated under the same feature-transmission protocol without additional compression for fair comparison.

## 4.2 Implementation Details

Sparse2comm uses the same sparse feature communication interface, progressive restoration modules, and robustness perturbation protocol across all datasets, with datasetspecific detection ranges and evaluation protocols following the corresponding benchmarks. On DAIR-V2X, we set the point cloud range to $[ - 1 0 0 . 8 , 1 0 0 . 8 ] \times [ - 4 0 , 4 0 ] \times [ - 5 , 3 ]$ meters and the voxel size to 0.4×0.4×4 meters, and evaluate only the Car class. On OpenV2V, we follow the standard LiDAR-based vehicle–vehicle cooperative detection setting, use PointPillars [32] as the BEV encoder, and evaluate the vehicle category under the official protocol. On V2V4Real, we use the LiDAR-based PointPillars backbone, set the detection region to [−100, 100] meters along the x axis and [−40, 40] meters along the y axis, merge all vehicle types into one vehicle class, and report AP at IoU thresholds of 0.5 and 0.7. The Sync and Async settings are further included in our unified robustness protocol for packet loss, communication delay, and spatial misalignment. The training follows three stages: collaborator encoder and RPN pretraining, Sparse Feature Encoding pre-training, and endto-end cooperative fine-tuning. We use Adam [55] with a learning rate of 0.01, a batch size of $6 ,$ and 80 epochs. For V2V4Real, we train for 60 epochs using AdamW with an initial learning rate of 0.001, a batch size of 4 per GPU, a weight decay of $1 \times 1 0 ^ { - 2 }$ , cosine learning-rate decay, and standard point-cloud augmentations.

## 4.3 Quantitative Results

## 4.3.1 Comparison with State-of-the-art

We benchmark Sparse2comm on DAIR-V2X [1], OpenV2V [21], and V2V4Real [22], covering dense featurelevel cooperation [39], [40], [42], [43], communicationefficient methods [8], [10], [41], [48], and robustness-oriented methods [15], [18], [20]. Under clean communication, Sparse2comm obtains 69.42/53.78 AP@0.5/AP@0.7 on DAIR-V2X, 93.70/87.66 on OpenV2V, and 71.46/45.36 on V2V4Real. It ranks first on DAIR-V2X, second on OpenV2V behind CoSDH [48] by only 0.50/0.53 AP, and remains competitive on V2V4Real, showing that sparse communication does not noticeably compromise normal detection accuracy while using only 1.0% relative bandwidth.

TABLE 2: Targeted robustness comparison on DAIR-V2X. All entries report AP@0.5 / AP@0.7, and the method list follows the main comparison in Table 1. Packet Loss, Transmission Delay, and Spatial Misalignment denote representative robustness settings evaluated under the same protocol for all methods. Mixed combines packet loss, transmission delay, and spatial misalignment to evaluate robustness under coupled realworld degradations. Since all results are evaluated on DAIR-V2X with the same feature shape, BW is reported as the transmitted bandwidth in Mbps. Gray entries denote the single-agent No Fusion baseline, while black bold and underlined numbers indicate the best and second-best cooperative-method results in each column, respectively.
<table><tr><td>Method</td><td>Category</td><td>Clean</td><td>Packet Loss</td><td>Delay</td><td>Misalignment</td><td>Mixed</td><td>BW↓ (Mbps)</td></tr><tr><td colspan="8">Basic and classic cooperative perception</td></tr><tr><td>No Fusion</td><td>Single-agent</td><td>42.45 / 30.78</td><td>42.45 / 30.78</td><td>42.45 / 30.78</td><td>42.45 / 30.78</td><td>42.45 / 30.78</td><td>0</td></tr><tr><td>F-Cooper [39]</td><td>Full feature</td><td>49.83 / 34.21</td><td>43.17  / 29.63</td><td>42.58 /  29.37</td><td>38.52 / 27.13</td><td>29.32 / 22.74</td><td>258.1</td></tr><tr><td>DiscoNet [42]</td><td>Full feature</td><td>51.31 / 35.77</td><td>46.30  / 32.27</td><td>44.72 / 31.55</td><td>41.26 / 30.94</td><td>33.26 / 27.50</td><td>258.1</td></tr><tr><td>V2VNet [40]</td><td>Full feature</td><td>61.71 / 46.25</td><td>48.83 / 34.49</td><td>57.72 / 34.83</td><td>43.63 / 27.28</td><td>33.95 / 21.71</td><td>258.1</td></tr><tr><td>V2X-ViT [43]</td><td>Full feature</td><td>58.37 / 42.77</td><td>48.02 / 35.68</td><td>54.44 / 34.04</td><td>45.59  / 30.36</td><td>37.31 / 26.41</td><td>258.1</td></tr><tr><td colspan="8">Robust cooperative perception</td></tr><tr><td>CoBEVFlow [15]</td><td>Temporal alignment</td><td>66.91  / 51.64</td><td>52.64 / 41.36</td><td>60.28 / 42.36</td><td>48.32 / 31.18</td><td>37.16 / 27.20</td><td>14.5</td></tr><tr><td>CoAlign [18]</td><td>Pose alignment</td><td>68.72 / 53.06</td><td>54.72 / 42.06</td><td>54.06  / 39.84</td><td>59.02 / 43.67</td><td>40.76 / 29.68</td><td>258.1</td></tr><tr><td>CoDynTrust [20]</td><td>Asynchronous robustness</td><td>67.97  / 52.36</td><td>56.92 / 43.58</td><td>64.58 /  47.56</td><td>53.57  / 38.56</td><td>42.35 / 30.17</td><td>27.4</td></tr><tr><td colspan="8">Communication-efficient cooperative perception</td></tr><tr><td>When2comm [41]</td><td>Adaptive comm.</td><td>54.96 / 36.15</td><td>46.32 / 31.40</td><td>52.48 / 33.11</td><td>45.02 / 29.30</td><td>36.89  / 26.33</td><td>258.1</td></tr><tr><td>How2comm [10]</td><td>Pragmatic comm.</td><td>68.76 / 53.08</td><td>55.86 / 42.94</td><td>64.41  / 48.13</td><td>56.24 / 42.16</td><td>41.44 / 29.43</td><td>8.8</td></tr><tr><td>CoSDH [48]</td><td>Hybrid fusion</td><td>68.91  / 53.14</td><td>56.34 / 43.28</td><td>56.84  /  42.07</td><td>50.96 / 35.94</td><td>40.62 / 29.01</td><td>7.2</td></tr><tr><td>Where2comm [8]</td><td>Selective feature</td><td>67.68 / 52.18</td><td>53.00 / 41.80</td><td>58.97  / 41.42</td><td>48.90  / 31.51</td><td>37.95 / 27.75</td><td>12.9</td></tr><tr><td colspan="2">Sparse2comm (Ours) Progressive restoration</td><td>69.42 / 53.78</td><td>67.12 / 50.77</td><td>64.73  / 47.94</td><td>63.08 / 45.39</td><td>58.10 / 39.54</td><td>2.6</td></tr></table>

![](images/4f2fa8564e65db1ef27df6e4d97538031cd2a7ebfaf8f1656c9229b883a09397.jpg)  
Fig. 7. Robustness evaluation under varying packet loss ratios, transmission delays, and spatial misalignments. The clean point and representative disturbance levels are aligned with the DAIR-V2X robustness comparison in Table 2.

The advantage becomes much clearer under the mixed setting, where packet loss, transmission delay, and spatial misalignment occur together. Sparse2comm reaches 58.10/39.54, 88.28/74.53, and 56.18/33.15 AP@0.5/AP@0.7 on the three datasets, improving over the strongest prior results by +15.75/+9.37, +7.13/+4.11, and +11.11/+7.34, respectively. Many dense feature-level baselines drop close to or below No Fusion once cooperative messages are corrupted, while communication-efficient and robustnessoriented methods remain less balanced under coupled disturbances. In contrast, Sparse2comm preserves SOTA-level clean accuracy and achieves the best mixed-setting performance across all datasets, providing a stronger trade-off among accuracy, robustness, and communication efficiency.

## 4.3.2 Robustness to Real-world Challenges

We then isolate the three major robustness factors on DAIR-V2X [1]: packet loss, transmission delay, and spatial misalignment. Table 2 gives the numerical breakdown, while Fig. 7 traces performance as each disturbance becomes stronger. Sparse2comm achieves the strongest overall robustness across the evaluated disturbances. Under packet loss, it obtains 67.12/50.77 AP@0.5/AP@0.7, outperforming the second-best method, CoDynTrust [20], by +10.20/+7.19. Under spatial misalignment, Sparse2comm reaches 63.08/45.39, improving over CoAlign [18] by +4.06/+1.72, even though CoAlign is specifically designed for pose alignment. Under transmission delay, Sparse2comm achieves the best AP@0.5 and remains close to the best AP@0.7 result, indicating that it maintains competitive temporal robustness while being more balanced across other disturbances.

The key observation is that existing methods tend to be robust to only one degradation type. How2comm [10], CoBEVFlow [15], and CoDynTrust [20] are strong under temporal asynchrony, and CoAlign [18] is effective under spatial misalignment, but their advantages do not consistently transfer to packet loss or coupled disturbances. The degradation curves make this imbalance clearer: many dense feature-level [39], [40], [42], [43] and communicationefficient methods [8], [10], [41], [48] degrade rapidly as disturbance intensity increases, and some even approach or fall below the No Fusion baseline. In contrast, Sparse2comm exhibits a slower performance decay across all three curves and achieves 58.10/39.54 under the mixed setting, surpassing the strongest prior result, CoDyn-Trust [20], by +15.75/+9.37 with only 2.6 Mbps bandwidth. Taken together, the targeted and mixed results support Sparse2comm as a more balanced robustness solution for practical cooperative perception.

## 4.4 Ablation Studies

## 4.4.1 Effectiveness ofMajor Components

The component ablation in Tab. 3 isolates the role of SFE, LAA, and SCF in the full framework. When all three modules are combined, Sparse2comm achieves the best performance, yielding +20.15 AP@0.5 over the baseline under noisy conditions. This confirms the complementary roles of the three modules in enhancing both accuracy and robustness across diverse real-world challenges.

## 4.4.2 Impact of Training Strategy

The training ablation in Tab. 4 favors the proposed multistage strategy over direct joint training, which optimizes all components simultaneously. Gradually pre-training key modules before end-to-end fine-tuning leads to more stable optimization and stronger generalization to real-world perturbations.

## 4.4.3 Impact of Sampling Strategy

For feature sampling, Object-aware Sampling outperforms both Uniform and Random Sampling in Tab. 5. Here, Uniform Sampling denotes evenly spaced sampling, while Random Sampling assigns equal probability to all regions. By focusing on object-centric areas during feature masking, our strategy preserves more discriminative information and effectively improves both detection accuracy and robustness.

## 4.4.4 Beyond Data Augmentation

To verify that the performance gains of Sparse2comm do not originate from data augmentation, we evaluate the effects of random masking (RM) and pose disturbance (PD) on both the baseline and our model. In the data-augmentation control of Table 6, introducing RM or PD consistently degrades the baseline under the ideal setting. RM slightly reduces accuracy (from 67.68 to 66.41 AP@0.5 and from 52.18 to 49.92 AP@0.7), while PD causes a larger drop (64.56 AP@0.5 and

TABLE 3: Ablation study of the core components. SFE, LAA, and SCF denote Sparse Feature Encoding, Latency-Aware Alignment, and Self-Calibrating Fusion, respectively.
<table><tr><td rowspan="2">SFE</td><td rowspan="2">LAASCF</td><td rowspan="2"></td><td colspan="2">Ideal</td><td colspan="2">Noisy</td><td rowspan="2">BW↓ | (Mbps)</td></tr><tr><td>AP@0.5↑</td><td>AP@0.7↑</td><td>AP@0.5↑</td><td>AP@0.7↑</td></tr><tr><td rowspan="5">√</td><td rowspan="5">√ √</td><td rowspan="5">√</td><td>67.68</td><td>52.18</td><td>37.95</td><td>27.75</td><td>12.9</td></tr><tr><td>66.31</td><td>51.12</td><td>49.91</td><td>36.52</td><td>2.6</td></tr><tr><td>68.12</td><td>52.87</td><td>41.67</td><td>29.90</td><td>12.9</td></tr><tr><td>68.76</td><td>53.24</td><td>48.53</td><td>33.32</td><td>12.9</td></tr><tr><td>67.93</td><td>52.54</td><td>52.82</td><td>35.78</td><td>2.6</td></tr><tr><td></td><td>√</td><td>√</td><td>69.77</td><td>53.96</td><td>51.73</td><td>34.53</td><td>12.9</td></tr><tr><td>√</td><td></td><td>√</td><td>68.64</td><td>52.80</td><td>55.68</td><td>37.41</td><td>2.6</td></tr><tr><td>√</td><td>√</td><td>√</td><td>69.42</td><td>53.78</td><td>58.10</td><td>39.54</td><td>2.6</td></tr></table>

TABLE 4: Ablation study of the training strategy.
<table><tr><td rowspan="2">Training Strategy</td><td colspan="2">Ideal 一</td><td colspan="2">Noisy</td></tr><tr><td>AP@0.5↑</td><td>AP@0.7↑</td><td>AP@0.5↑</td><td>AP@0.7↑</td></tr><tr><td>Joint Training</td><td>58.65</td><td>48.23</td><td>49.40</td><td>33.77</td></tr><tr><td>Multi-Stage Training</td><td>69.42</td><td>53.78</td><td>58.10</td><td>39.54</td></tr></table>

TABLE 5: Ablation study on masking strategies in the Sparse Feature Encoding module.
<table><tr><td rowspan="2">Sample Strategy</td><td colspan="2">Ideal</td><td colspan="2">Noisy</td></tr><tr><td>AP@0.5↑</td><td>AP@0.7↑</td><td>AP@0.5↑</td><td>AP@0.7↑</td></tr><tr><td>Uniform Sampling</td><td>67.25</td><td>52.30</td><td>52.97</td><td>36.10</td></tr><tr><td>Random Sampling</td><td>68.76</td><td>53.12</td><td>56.23</td><td>38.52</td></tr><tr><td>Object-aware Sampling</td><td>69.42</td><td>53.78</td><td>58.10</td><td>39.54</td></tr></table>

TABLE 6: Ablation study on the data augmentation. RM denotes random masking on collaborator BEV features, with mask ratio sampled from r ∼ U(0, 1.0) (Uniform distribution). PD denotes pose disturbance applied to BEV features, including rotational noise ∆R ∼ N(0, 3.0<sup>◦</sup>) and translational noise ∆T ∼ N(0, 0.5)m.(Gaussian distribution).
<table><tr><td rowspan="2">Method</td><td colspan="2">Data augmentation</td><td colspan="2">Ideal</td><td colspan="2">Noisy</td></tr><tr><td>RM</td><td>PD</td><td>AP@0.5↑ AP@0.7↑ |AP@0.5↑ AP@0.7↑</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Baseline</td><td rowspan="4">√</td><td></td><td>67.68</td><td>52.18</td><td>37.95</td><td>27.75</td></tr><tr><td></td><td>66.41</td><td>49.92</td><td>40.66</td><td>26.57</td></tr><tr><td>√</td><td>64.56</td><td>47.58</td><td>42.96</td><td>29.98</td></tr><tr><td>√</td><td>67.03</td><td>49.86</td><td>45.47</td><td>28.43</td></tr><tr><td>Sparse2comm</td><td>√</td><td>√</td><td>69.42</td><td>53.78</td><td>58.1</td><td>39.54</td></tr></table>

47.58 AP@0.7). Combining RM+PD further lowers performance to 67.03 AP@0.5 and 49.86 AP@0.7. Under the noisy setting, RM and PD bring only marginal improvements, and RM+PD yields 45.47 AP@0.5 and 28.43 AP@0.7. However, these gains remain insufficient to mitigate the performance degradation observed on the baseline. In contrast, Sparse2comm achieves 69.42 / 53.78 AP@0.5/0.7 under the ideal setting, outperforming the non-augmented baseline by +1.74 / +1.60. Under noise, Sparse2comm reaches 58.10 / 39.54, surpassing the corresponding baseline by +20.15 / +11.79. This comparison supports that the improvements of Sparse2comm arise from its architectural design rather than augmentation heuristics, effectively ruling out data augmentation as a confounding factor.

## 4.4.5 Efficiency Analysis

For deployment efficiency, we profile Sparse2comm and the representative baselines under a shared DAIR-V2X setting, using the same machine, BEV range, and inference-stage configuration as in Table 7. In addition to model-side parameters, FLOPs, and pure inference latency, we explicitly account for the communication time required to transmit intermediate cooperative features. Given the transmitted bandwidth BW in Mbps, evaluation frequency f=10 Hz, and V2X link rate R, the transmission time is computed as $T _ { \mathrm { t x } } ( R ) = 1 0 0 0 \mathrm { B W } / ( f R )$ , and the end-to-end latency is $T _ { \mathrm { e 2 e } } = T _ { \mathrm { m o d e l } } + T _ { \mathrm { t x } }$ . We report both a conservative 10 Mbps link and an optimistic 50 Mbps link to reflect practical and high-capacity deployment conditions.

TABLE 7: End-to-end efficiency comparison of the representative methods. All methods are profiled under the same DAIR-V2X input setting. Model latency denotes pure inference latency, while transmission time is computed from the transmitted bandwidth under conservative 10 Mbps and optimistic 50 Mbps V2X link rates. Gray entries denote the single-agent No Fusion baseline, and black bold numbers indicate the best cooperativemethod result in each column.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Category</td><td colspan="3">Computation</td><td colspan="3">Communication</td><td colspan="2">End-to-End Latency</td></tr><tr><td>Params (M)↓</td><td>FLOPs (G)↓</td><td>Model Lat. (ms)↓</td><td>BW (Mbps)↓</td><td>Tx.@10M (ms)↓</td><td>Tx.@50M (ms)↓</td><td>E2E@10M (ms).↓</td><td>E2E@50M (ms).↓</td></tr><tr><td colspan="2">Basic and classic cooperative perception</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Fusion</td><td>Single-agent</td><td>6.6</td><td>118.4</td><td>37.8</td><td>0</td><td>0</td><td>0</td><td>37.8</td><td>37.8</td></tr><tr><td>F-Cooper [39]</td><td>Full feature</td><td>7.3</td><td>225.6</td><td>48.9</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2629.9</td><td>565.1</td></tr><tr><td>DiscoNet [42]</td><td>Full feature</td><td>6.7</td><td>238.2</td><td>50.8</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2631.8</td><td>567.0</td></tr><tr><td>V2VNet [40]</td><td>Full feature</td><td>14.6</td><td>531.6</td><td>88.5</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2669.5</td><td>604.7</td></tr><tr><td>V2X-ViT [43]</td><td>Full feature</td><td>14.2</td><td>287.0</td><td>85.5</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2666.5</td><td>601.7</td></tr><tr><td colspan="2">Communication-efficient cooperative perception When2comm [41] Adaptive comm.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>8.8</td><td>228.3</td><td>52.7</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2633.7</td><td>568.9</td></tr><tr><td>How2comm [10]</td><td>Where2comm [8] Selective feature</td><td>11.4</td><td>218.9</td><td>51.2</td><td>12.9</td><td>129.0</td><td>25.8</td><td>180.2</td><td>77.0</td></tr><tr><td>CoSDH [48]</td><td>Pragmatic comm.</td><td>16.2</td><td>283.6</td><td>64.8</td><td>8.8</td><td>88.0</td><td>17.6</td><td>152.8</td><td>82.4</td></tr><tr><td></td><td>Hybrid fusion</td><td>17.9</td><td>296.8</td><td>67.2</td><td>7.2</td><td>72.0</td><td>14.4</td><td>139.2</td><td>81.6</td></tr><tr><td colspan="2">Robust cooperative perception</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoBEVFlow [15]</td><td>BEV flow alignment</td><td>18.9</td><td>301.7</td><td>66.4</td><td>14.5</td><td>145.0</td><td>29.0</td><td>211.4</td><td>95.4</td></tr><tr><td>CoAlign [18]</td><td>Pose alignment</td><td>18.4</td><td>334.2</td><td>73.5</td><td>258.1</td><td>2581.0</td><td>516.2</td><td>2654.5</td><td>589.7</td></tr><tr><td>CoDynTrust [20]</td><td>Async. robustness</td><td>27.6</td><td>382.5</td><td>82.9</td><td>27.4</td><td>274.0</td><td>54.8</td><td>356.9</td><td>137.7</td></tr><tr><td>Sparse2comm</td><td>Progressive restoration</td><td>34.0</td><td>326.4</td><td>69.7</td><td>2.6</td><td>26.0</td><td>5.2</td><td>95.7</td><td>74.9</td></tr></table>

TABLE 8: Ablation study of Sparse2comm efficiency. SFE, LAA, and SCF denote Sparse Feature Encoding, Latency-Aware Alignment, and Self-calibrating Fusion. We report inference-stage parameters, FLOPs, and latency under the same profiling setting.
<table><tr><td>SFE</td><td>LAA</td><td>SCF</td><td>Params (M)</td><td>FLOPs (G)</td><td>Latency (ms)</td></tr><tr><td rowspan="3">√</td><td rowspan="3">√</td><td rowspan="3"></td><td>11.4</td><td>218.9</td><td>51.2</td></tr><tr><td>24.0</td><td>269.3</td><td>59.8</td></tr><tr><td>16.0</td><td>255.7</td><td>56.2</td></tr><tr><td rowspan="3">√</td><td rowspan="3">√ √</td><td>√</td><td>16.8</td><td>239.2</td><td>54.0</td></tr><tr><td></td><td>28.6</td><td>306.1</td><td>64.7</td></tr><tr><td>√</td><td>21.4</td><td>276.0</td><td>59.0</td></tr><tr><td>√</td><td></td><td>√</td><td>29.4</td><td>289.6</td><td>62.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td>34.0</td><td>326.4</td><td>69.7</td></tr></table>

TABLE 9: Ablation study on the architecture of Sparse Feature Encoding. $N _ { E }$ and $N _ { D }$ denote the number of ViT blocks in the encoder and decoder, respectively.

<table><tr><td rowspan="2">Setting</td><td colspan="2">#Blocks</td><td colspan="2">Ideal</td><td colspan="2">Noisy</td></tr><tr><td> $N _ { E }$ </td><td> $N _ { D }$ </td><td>AP@0.5↑</td><td>AP@0.7↑</td><td>AP@0.5↑</td><td>AP@0.7↑</td></tr><tr><td>S1: Shallow</td><td>3</td><td>1</td><td>66.23</td><td>51.12</td><td>53.17</td><td>35.14</td></tr><tr><td>S2: Encoder-heavy</td><td>7</td><td>1</td><td>68.47</td><td>53.18</td><td>56.41</td><td>38.23</td></tr><tr><td>S3: Balanced (ours)</td><td>5</td><td>2</td><td>69.42</td><td>53.78</td><td>58.10</td><td>39.54</td></tr><tr><td>S4: Decoder-heavy</td><td>5</td><td>4</td><td>69.27</td><td>53.63</td><td>57.86</td><td>39.31</td></tr><tr><td>S5: Deep (optional)</td><td>7</td><td>4</td><td>69.14</td><td>53.46</td><td>57.22</td><td>38.87</td></tr></table>

Model-only latency can substantially underestimate the deployment cost of dense feature-level cooperation. Although several full-feature or alignment-based methods have moderate inference latency, their dense communication cost leads to second-level end-to-end latency under a 10 Mbps V2X link. Communication-efficient methods reduce this cost, but their transmitted feature volume remains noticeably higher than Sparse2comm. In contrast,

Sparse2comm requires only 2.6 Mbps on DAIR-V2X, reducing the transmission time to 26.0 ms under the conservative 10 Mbps link and 5.2 ms under the 50 Mbps link. Consequently, Sparse2comm achieves 95.7 ms end-to-end latency under 10 Mbps and 74.9 ms under 50 Mbps, while maintaining strong robustness in Table 2. The componentlevel profile in Table 8 further breaks down the computational cost: starting from the object-foreground baseline, SFE, LAA, and SCF each introduce controlled overhead, and the complete system increases latency from 51.2 ms to 69.7 ms. Overall, Sparse2comm offers a favorable system-level trade-off between computation, communication, robustness, and real-time deployment feasibility.

## 4.4.6 SFE Architecture

We further vary the SFE encoder–decoder depth in Tab. 9, where $N _ { E }$ and $N _ { D }$ denote the number of ViT [56] blocks in the encoder and decoder, respectively. This ablation evaluates shallow, encoder-heavy, decoder-heavy, and deeper configurations to understand how capacity allocation influences performance under both ideal and noisy conditions.

The alternative SFE designs, including the shallow (S1), encoder-heavy (S2), decoder-heavy (S4), and deeper configuration (S5), do not provide the same balance across clean and noisy settings. Shallow and unbalanced variants either lack sufficient representational capacity or fail to utilize it efficiently, and simply increasing depth does not resolve the robustness issue, as S5 still performs poorly under noise. In contrast, the balanced architecture (S3 with $N _ { E } { = } 5$ and $N _ { D } { = } 2 )$ obtains the highest accuracy in both ideal and noisy scenarios, reaching 69.42 / 53.78 under ideal conditions and 58.10 / 39.54 under noisy conditions. This pattern suggests that allocating moderate depth to both the encoder and the decoder provides the most effective and robust design for Sparse Feature Encoding.

## 4.5 Qualitative Results

## 4.5.1 Detection Result Visualization

Figs. 8 and 9 show representative noisy-condition detections on DAIR-V2X and V2V4Real, respectively. Green/red boxes denote ground-truth/predicted objects, and magenta dashed regions highlight challenging areas. On DAIR-V2X [1], baseline methods often miss compact vehicle clusters or produce spatially shifted boxes, especially for adjacent and partially occluded targets. Sparse2comm detects more vehicles in these crowded regions and aligns the predicted boxes more closely with the ground truth. On V2V4Real [22], noisy communication causes unstable long-range detections and inaccurate orientations along the driving corridor. In contrast, Sparse2comm produces fewer missed detections and more reliable box placement, showing its effectiveness in real-world vehicle–vehicle cooperation. These comparisons are consistent with Tables 1 and 2, where Sparse2comm improves noisy-condition detection by restoring semantic content, temporal consistency, and spatial alignment before fusion.

![](images/3f8af00b6aa080d8d850721404e8d937d8ca7f6102ff007cc5c99226ab2e7456.jpg)

Fig. 8. Detection result visualization on DAIR-V2X. Representative examples are shown under clean and mixed noisy settings. Green and red boxes denote the ground truths and detection results, respectively. Under packet loss, transmission delay, and spatial misalignment, existing methods exhibit noticeable false and missed detections, while our Sparse2comm maintains accurate and consistent bounding boxes.  
![](images/489e49d00290650dcc6269b312310f532d37f94383564c071f13f63294225007.jpg)  
Fig. 9. Detection result visualization on V2V4Real. Additional real-world vehicle–vehicle examples are shown under clean and mixed noisy settings. Green and red boxes denote the ground truths and detection results, respectively. Sparse2comm preserves more spatially consistent detections when cooperative features are affected by packet loss, transmission delay, and spatial misalignment.

## 4.5.2 SFE Reconstruction Visualization

To inspect the reconstruction process, Fig. 10 visualizes refined, mask-sampled, and reconstructed BEV features produced by Sparse Feature Encoding (SFE). For each packet– loss ratio (10%–100%), we include: (i) the refined BEV features $\mathcal { F } _ { i } ^ { t _ { i } }$ , (ii) the mask-sampled BEV features $\tilde { \mathcal { F } } _ { i } ^ { t _ { i } }$ produced by the Mask Sampling module, (iii) the reconstructed BEV features $\hat { \mathcal { F } } _ { i } ^ { t _ { i } }$ generated by the Feature Reconstruction module, and (iv-vi) the corresponding information-loss maps and loss histograms.

The mask-sampled BEV features simulate the actual information received by a vehicle under packet loss. As the packet-loss ratio increases, $\tilde { \mathcal { F } } _ { i } ^ { t _ { i } }$ exhibits severe degradation, and the mask-sampled loss $| \mathcal { F } _ { i } ^ { t _ { i } } - \tilde { \mathcal { F } } _ { i } ^ { t _ { i } } | ^ { 2 }$ clearly highlights large missing regions. In contrast, the reconstruction module effectively recovers the missing content, producing features structurally close to the refined BEV representation. The reconstructed loss $| \mathcal { F } _ { i } ^ { t _ { i } } ~ - ~ \hat { \mathcal { F } } _ { i } ^ { t _ { i } } | ^ { 2 }$ remains near zero, and the corresponding histogram distributions collapse sharply toward zero across all packet-loss settings. This behavior is consistent even under extremely high packet loss (e.g., 90%), demonstrating the strong capability of SFE to restore semantic structure from severely degraded inputs. The nearzero reconstruction loss supports that SFE not only compensates for transmission-induced information loss in real cooperative communication, but also enables an efficient sparse-to-dense reconstruction paradigm. In practice, collaborating agents can transmit sparsely sampled features with substantially reduced bandwidth requirements, while the ego vehicle reconstructs dense features with minimal performance degradation.

![](images/b7c467a2c7b4046eac42c655710a28b6cfbbded4820e4e910429190f0e427945.jpg)  
Fig. 10. Visualization of the Sparse Feature Encoding (SFE). We visualize the refined BEV features, the mask-sampled BEV features obtained by the Mask Sampling submodule, the reconstructed BEV features produced by the Feature Reconstruction submodule, along with their corresponding information-loss maps and loss histograms. The mask-sampled BEV features suffer from severe information degradation, while the reconstruction submodule effectively recovers the missing content, producing features close to the refined BEV and loss values near zero. This behavior remains consistent across packet-loss ratios from 10% to 100%, demonstrating SFE’s strong capability to recover missing information and reduce bandwidth through sparse-to-dense reconstruction.

## 5 CONCLUSION

This paper presented Sparse2comm, a bandwidth-efficient and robust cooperative 3D object detection framework for unreliable real-world cooperation. Rather than treating communication reduction, packet loss, temporal delay, and spatial misalignment as separate problems, Sparse2comm formulates them as coupled degradations of cooperative features and addresses them through a progressive feature restoration pipeline. Sparse Feature Encoding builds a compact and recoverable object-centric representation for ultra-low-bandwidth transmission and packet-loss recovery. Latency-Aware Alignment then predicts motion flow from temporal collaborator features to compensate delayed cooperative messages at the ego-vehicle timestamp, while Self-Calibrating Fusion estimates residual pose-induced offsets in a self-supervised manner to refine cross-agent feature correspondence before adaptive fusion. Experiments on DAIR-V2X [1], OpenV2V [21], and V2V4Real [22] show that Sparse2comm maintains competitive clean accuracy and achieves stronger mixed robustness across diverse cooperative perception scenarios. Compared with the representative selective communication baseline Where2comm, Sparse2comm improves mixed-setting AP@0.5/AP@0.7 by +20.15/+11.79, +12.66/+11.07, and +15.36/+12.61 on the three datasets, respectively, while using only 1.0% relative feature bandwidth.

Limitation and future work. Sparse2comm currently focuses on cooperative 3D object detection with feature-level communication. Future work will extend the progressive restoration idea to broader cooperative autonomy tasks, such as occupancy perception, trajectory prediction, and end-to-end cooperative driving, and further investigate its deployment behavior under real communication systems with dynamic bandwidth and latency.

## REFERENCES

[1] H. Yu, Y. Luo, M. Shu, Y. Huo, Z. Yang, Y. Shi, Z. Guo, H. Li, X. Hu, J. Yuan et al., “Dair-v2x: A large-scale dataset for vehicleinfrastructure cooperative 3d object detection,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 21 361–21 370.

[2] H. Yu, W. Yang, J. Zhong, Z. Yang, S. Fan, P. Luo, and Z. Nie, “End-to-end autonomous driving through v2x cooperation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9598–9606.

[3] H. Yu, Y. Tang, E. Xie, J. Mao, P. Luo, and Z. Nie, “Flow-based feature fusion for vehicle-infrastructure cooperative 3d object detection,” Advances in Neural Information Processing Systems, vol. 36, pp. 34 493–34 503, 2023.

[4] H. Yu, W. Yang, H. Ruan, Z. Yang, Y. Tang, X. Gao, X. Hao, Y. Shi, Y. Pan, N. Sun et al., “V2x-seq: A large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5486–5495.

[5] G. Liu, Y. Hu, C. Xu, W. Mao, J. Ge, Z. Huang, Y. Lu, Y. Xu, J. Xia, Y. Wang, and S. Chen, “Toward collaborative autonomous driving: Simulation platform and end-to-end system,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 6566– 6584, 2025.

[6] X. Gao, X. Zhang, Y. Lu, Y. Huang, L. Yang, Y. Xiong, and P. Liu, “A survey of collaborative perception in intelligent vehicles at intersections,” IEEE Transactions on Intelligent Vehicles, 2024.

[7] J. Clancy, D. Mullins, B. Deegan, J. Horgan, E. Ward, C. Eising, P. Denny, E. Jones, and M. Glavin, “Wireless access for v2x communications: Research, challenges and opportunities,” IEEE Communications Surveys & Tutorials, vol. 26, no. 3, pp. 2082–2119, 2024.

[8] Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen, “Where2comm: Communication-efficient collaborative perception via spatial confidence maps,” Advances in neural information processing systems, vol. 35, pp. 4874–4886, 2022.

[9] S. Liu, Z. Ding, J. Fu, H. Li, S. Chen, S. Zhang, and X. Zhou, “V2x-pc: Vehicle-to-everything collaborative perception via point cluster,” arXiv preprint arXiv:2403.16635, 2024.

[10] D. Yang, K. Yang, Y. Wang, J. Liu, Z. Xu, R. Yin, P. Zhai, and L. Zhang, “How2comm: Communication-efficient and collaboration-pragmatic multi-agent perception,” Advances in Neural Information Processing Systems, vol. 36, pp. 25 151–25 164, 2023.

[11] Y. Hu, X. Pang, X. Qin, Y. C. Eldar, S. Chen, P. Zhang, and W. Zhang, “Pragmatic communication in multi-agent collaborative perception,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 8, pp. 9279–9296, 2026.

[12] T. Tang, C. Zhang, G. Chen, F. Xia, Q. Xu, and C. Liu, “Rocooper: Robust cooperative perception under vehicle-to-vehicle communication impairments,” in IEEE INFOCOM 2025-IEEE Conference on Computer Communications. IEEE, 2025, pp. 1–10.

[13] Z. Song, L. Yang, F. Wen, and J. Li, “Traf-align: Trajectory-aware feature alignment for asynchronous multi-agent perception,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12 048–12 057.

[14] Z. Lei, S. Ren, Y. Hu, W. Zhang, and S. Chen, “Latency-aware collaborative perception,” in European Conference on Computer Vision. Springer, 2022, pp. 316–332.

[15] S. Wei, Y. Wei, Y. Hu, Y. Lu, Y. Zhong, S. Chen, and Y. Zhang, “Asynchrony-robust collaborative perception via bird’s eye view flow,” Advances in Neural Information Processing Systems, vol. 36, pp. 28 462–28 477, 2023.

[16] Z. Song, F. Wen, H. Zhang, and J. Li, “A cooperative perception system robust to localization errors,” in 2023 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2023, pp. 1–6.

[17] S. Ren, Z. Lei, Z. Wang, M. Dianati, Y. Wang, S. Chen, and W. Zhang, “Interruption-aware cooperative perception for v2x communication-aided autonomous driving,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 4, pp. 4698–4714, 2024.

[18] Y. Lu, Q. Li, B. Liu, M. Dianati, C. Feng, S. Chen, and Y. Wang, “Robust collaborative 3d object detection in presence of pose errors,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 4812–4818.

[19] S. Hong, Y. Liu, Z. Li, S. Li, and Y. He, “Multi-agent collaborative perception via motion-aware robust communication network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 301–15 310.

[20] Y. Xu, L. Li, J. Wang, B. Yang, Z. Wu, X. Chen, and J. Wang, “Codyntrust: Robust asynchronous collaborative perception via dynamic feature trust modulus,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 336–342.

[21] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, and J. Ma, “OPV2V: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication,” in 2022 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 2583–2589.

[22] R. Xu, X. Xia, J. Li, H. Li, S. Zhang, Z. Tu, Z. Meng, H. Xiang, X. Dong, R. Song et al., “V2v4real: A real-world large-scale dataset for vehicle-to-vehicle cooperative perception,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 13 712–13 722.

[23] T. Wang, X. Zhu, J. Pang, and D. Lin, “Fcos3d: Fully convolutional one-stage monocular 3d object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 913– 922.

[24] L. Yang, X. Zhang, J. Li, L. Wang, M. Zhu, and L. Zhu, “Lite-fpn for keypoint-based monocular 3d object detection,” Knowledge-Based Systems, vol. 271, p. 110517, 2023.

[25] L. Yang, X. Zhang, J. Li, L. Wang, M. Zhu, C. Zhang, and H. Liu, “Mix-teaching: A simple, unified and effective semi-supervised learning framework for monocular 3d object detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 11, pp. 6832–6844, 2023.

[26] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bevformer: learning bird’s-eye-view representation from lidarcamera via spatiotemporal transformers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[27] Y. Li, Z. Ge, G. Yu, J. Yang, Z. Wang, Y. Shi, J. Sun, and Z. Li, “Bevdepth: Acquisition of reliable depth for multi-view 3d object detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 2, 2023, pp. 1477–1485.

[28] L. Yang, K. Yu, T. Tang, J. Li, K. Yuan, L. Wang, X. Zhang, and P. Chen, “Bevheight: A robust framework for vision-based roadside 3d object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 611–21 620.

[29] L. Yang, T. Tang, J. Li, P. Chen, K. Yuan, L. Wang, Y. Huang, X. Zhang, and K. Yu, “Bevheight++: Toward robust visual centric 3d object detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[30] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point cloud based 3d object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4490–4499.

[31] Y. Yan, Y. Mao, and B. Li, “Second: Sparsely embedded convolutional detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.

[32] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, “Pointpillars: Fast encoders for object detection from point clouds,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 12 697–12 705.

[33] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. Rus, and S. Han, “Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” arXiv preprint arXiv:2205.13542, 2022.

[34] Z. Song, H. Wei, L. Bai, L. Yang, and C. Jia, “Graphalign: Enhancing accurate feature alignment by graph matching for multi-modal 3d object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3358–3369.

[35] Z. Song, L. Yang, S. Xu, L. Liu, D. Xu, C. Jia, F. Jia, and L. Wang, “Graphbev: Towards robust bev feature alignment for multimodal 3d object detection,” in European Conference on Computer Vision. Springer, 2024, pp. 347–366.

[36] Y. Li, D. Ma, Z. An, Z. Wang, Y. Zhong, S. Chen, and C. Feng, “V2X-Sim: Multi-agent collaborative perception dataset and benchmark for autonomous driving,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10 914–10 921, 2022.

[37] H. Xiang, Z. Zheng, X. Xia, R. Xu, L. Gao, Z. Zhou, X. Han, X. Ji, M. Li, Z. Meng et al., “V2x-real: a large-scale dataset for vehicleto-everything cooperative perception,” in European Conference on Computer Vision. Springer, 2024, pp. 455–470.

[38] L. Yang, X. Zhang, J. Li, C. Wang, J. Ma, Z. Song, T. Zhao, Z. Song, L. Wang, M. Zhou, Y. Shen, K. Wu, and C. Lv, “V2X-Radar: A multi-modal dataset with 4D radar for cooperative perception,” Advances in Neural Information Processing Systems, 2025.

[39] Q. Chen, X. Ma, S. Tang, J. Guo, Q. Yang, and S. Fu, “F-cooper: Feature based cooperative perception for autonomous vehicle edge computing system using 3d point clouds,” in Proceedings of the 4th ACM/IEEE Symposium on Edge Computing, 2019, pp. 88–100.

[40] T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, and R. Urtasun, “V2vnet: Vehicle-to-vehicle communication for joint perception and prediction,” in European conference on computer vision. Springer, 2020, pp. 605–621.

[41] Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira, “When2com: Multi-agent perception via communication graph grouping,” in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2020, pp. 4106–4115.

[42] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, and W. Zhang, “Learning distilled collaboration graph for multi-agent perception,” in Advances in Neural Information Processing Systems, vol. 34, 2021.

[43] R. Xu, H. Xiang, Z. Tu, X. Xia, M.-H. Yang, and J. Ma, “V2x-vit: Vehicle-to-everything cooperative perception with vision transformer,” in European conference on computer vision. Springer, 2022, pp. 107–124.

[44] R. Xu, C.-J. Chen, Z. Tu, and M.-H. Yang, “V2X-ViTv2: Improved vision transformers for vehicle-to-everything cooperative perception,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 1, pp. 650–662, 2025.

[45] R. Xu, Z. Tu, H. Xiang, W. Shao, B. Zhou, and J. Ma, “CoBEVT: Cooperative bird’s eye view semantic segmentation with sparse transformers,” in Proceedings of The 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 989–1000.

[46] K. Yang, D. Yang, J. Zhang, H. Wang, P. Sun, and L. Song, “What2comm: Towards communication-efficient collaborative perception via feature decoupling,” in Proceedings of the 31st ACM international conference on multimedia, 2023, pp. 7686–7695.

[47] J. Zhang, K. Yang, Y. Wang, H. Wang, P. Sun, and L. Song, “Ermvp: Communication-efficient and collaboration-robust multivehicle perception in challenging environments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12 575–12 584.

[48] J. Xu, Y. Zhang, Z. Cai, and D. Huang, “Cosdh: Communicationefficient collaborative perception via supply-demand awareness and intermediate-late hybridization,” in Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2025, pp. 6834–6843.

[49] Z. Ding, J. Fu, S. Liu, H. Li, S. Chen, H. Li, S. Zhang, and X. Zhou, “Point cluster: A compact message unit for communicationefficient collaborative perception,” in The Thirteenth International Conference on Learning Representations, 2025.

[50] J. Li, R. Xu, X. Liu, J. Ma, Z. Chi, J. Ma, and H. Yu, “Learning for vehicle-to-vehicle cooperative perception under lossy communication,” IEEE Transactions on Intelligent Vehicles, vol. 8, no. 4, pp. 2650–2660, 2023.

[51] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[52] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[53] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in International conference on machine learning. pmlr, 2015, pp. 448–456.

[54] V. Nair and G. E. Hinton, “Rectified linear units improve restricted boltzmann machines,” in Proceedings of the 27th international conference on machine learning (ICML-10), 2010, pp. 807–814.

[55] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[56] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021.