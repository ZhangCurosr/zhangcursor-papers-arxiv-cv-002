# Towards Real-Time and Adaptable LiDAR Scene Completion

Azhar Hussian, Martin Vossiek, and Vasileios Belagiannis

Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany {azhar.hussian,martin.vossiek,vasileios.belagiannis}@fau.de

Abstract. LiDAR scene completion is a key component of 3D perception in autonomous driving, where the scene must be completed in real time to be usable in downstream tasks. Existing approaches typically follow an initialize-and-refine paradigm, in which a coarse initialization of the scene is first constructed, then refined into complete 3D geometry. Generative models are slower because they iteratively refine random Gaussian noise into the scene, while non-generative methods perturb the partial scene with a fixed noise scale, which limits coverage of large gaps and occluded regions and requires manual recalibration for each new sensor configuration. We present RapidLiDAR, a LiDAR scene completion method that treats the initialization itself as a learned, data-driven component. We propose an adaptive initialization module that predicts a spatially varying displacement for each partial input point, expanding the partial observations into a coarse scene initialization adapted to the local geometry, without requiring manual noise tuning. To refine this coarse initialization into a complete and coherent scene, we additionally propose a multi-scale reconstruction module that further refines point positions by querying multi-scale 3D voxel and 2D BEV feature maps constructed from the input scan. By replacing point-neighborhood operators such as farthest point sampling and k-nearest neighbor search with voxel- and BEV-based feature extraction, our architecture is faster and can handle diferent input resolutions by design. Experiments on SemanticKITTI and KITTI-360 show that our method achieves completion performance on par with the state of the art while completing a full scene in 0.1 seconds, which is 2.3 times faster than the fastest prior method. This matches the 10 Hz acquisition rate of typical automotive LiDAR sensors, taking a step toward real-time LiDAR scene completion. Code is available at https://github.com/AzharSindhi/RapidLiDAR.

## 1 Introduction

LiDAR sensors are widely used in domains such as automated driving due to their ability to provide accurate 3D geometric measurements that are relatively more robust to lighting and weather conditions than cameras [9]. However, LiDAR measurements are often sparse and incomplete, particularly for distant objects where point density decreases, and in regions afected by occlusions or limited sensor resolution [9, 12, 22]. As a result, large portions of the environment may be only partially observed or entirely missing. This incomplete representation poses a significant challenge for downstream perception systems that rely on holistic scene understanding. Scene completion addresses this limitation by inferring missing geometry and reconstructing a dense, consistent representation of the surrounding environment.

![](images/87d09660375316b8f12bb102cb798aafffa20437c4b264c776b1f0e2f83849ca.jpg)  
Fig. 1: Initialization matters. Top row: each method’s initialization; bottom row: the corresponding refined output. The highlighted box marks a large unobserved region. (a) LiDif [19] starts from Gaussian noise that carries no information about the scene; (b) LiNeXt [6] perturbs the input with a fixed noise variance, so its points stay near the observed surface and never reach across the gap; (c) Our adaptive module learns data-dependent displacements that populate the region from the surrounding geometry, and this coverage is preserved in both the coarse initialization and the final refined result.

Existing scene completion approaches [6, 17–19] generally follow an initializeand-refine paradigm where an initial coarse estimate of the complete scene is first constructed, which we term the initialization. This initialization is then refined into complete 3D geometry, depending on the specific refinement approach adopted in prior work. Dominant among these are generative models, particularly difusion-based models [17–19], which use random Gaussian noise as the initialization and iteratively denoise it into a plausible 3D scene. On the other hand, a recent non-generative approach, LiNeXt [6], initializes the scene by perturbing the partial input scene with fixed noise, which is then refined in a single forward pass.

We argue that the eficiency of scene completion methods, in terms of speed and generalization, is largely dependent on how well the initialization coarsely represents the final scene. For difusion-based generative models, the random Gaussian noise used as initialization is least informative about the final complete scene; therefore, they require multiple iterations during inference to construct a plausible 3D scene. This iterative inference results in slower completion times, even with recent acceleration techniques [29]. This makes generative models infeasible for time-critical applications such as automated driving, where real-time perception is essential. On the other hand, the fixed-noise-perturbed initialization strategy used by LiNeXt is not generalizable across diferent sensor configurations. Because sensor configurations vary in point density and coverage pattern, the noise scale must be retuned manually for each new dataset. Furthermore, because each point is perturbed only within a small, fixed radius of its original position, the initialized points remain close to the observed input and cannot cover large gaps and occluded regions.

In this work, we make the initialization itself a learned component. We propose an adaptive initialization module that predicts a spatially varying displacement for each partial input point. Rather than perturbing points with a fixed noise scale, the module learns to displace each point by an amount that reflects the local scene structure: points near sparse or occluded regions move further to fill in the missing geometry, while points in already dense regions move only slightly. This produces a coarse scene initialization that is adapted to the local geometry, rather than uniformly spread around the observed input (see Fig. 1). To refine the adaptively initialized scene into a complete and coherent scene, we further propose a multi-scale reconstruction module that adjusts point positions using multi-scale voxel and bird’s-eye-view (BEV) features extracted from the input scan. Both modules extract features independently of the input point count by avoiding point-neighborhood operators, such as farthest point sampling (FPS) and k-nearest neighbor (k-NN), that are commonly used in point-based feature extractors such as LiNeXt. As a result, our approach is faster and can handle diferent input resolutions by design.

Experiments on SemanticKITTI [2] and KITTI-360 [14] show that our method achieves completion performance on par with the state of the art while completing a full scene in 0.1 s, which is 2.3× faster than the fastest prior method. This matches the 10 Hz scan rate of typical automotive LiDAR sensors and provides a path toward real-time scene completion in autonomous driving scenarios. Our main contributions are summarized as follows:

– We introduce an adaptive initialization module that constructs the initialized point set via spatially varying learned displacements, rather than relying on fixed, hand-tuned, and dataset-specific noise scales that require manual recalibration for each new sensor configuration.

– We design an architecture that replaces point-neighborhood operations (e.g., FPS and k-NN) with voxel- and BEV-based feature extraction, thereby improving runtime and supporting diferent input resolutions by design.

– We achieve completion performance on par with the state of the art on SemanticKITTI [2] and KITTI-360 [14], and complete a full scene in 0.1 seconds, which is 2.3× faster than the fastest prior method and matches the 10 Hz scanning rate of typical automotive LiDAR sensors.

## 2 Related Work

We review generative and single-pass approaches to scene completion, as well as the BEV attention mechanisms on which our method builds.

## 2.1 Generative Models for Scene Completion

Following the success of Denoising Difusion Probabilistic Models (DDPMs) [7] on synthetic single-object point clouds [10,16,30], generative models have become the dominant approach to real-world LiDAR scene completion [17,19]. To preserve fine geometric detail at the scene scale, these methods formulate the difusion process directly on 3D points, treating completion as a point-level denoising problem [19] whose performance strongly depends on the choice of starting point [17]. Since iterative sampling requires hundreds of network evaluations and tens of seconds per scan, subsequent work has focused on reducing inference cost. Some methods distill a pre-trained difusion model into a faster few-step student [29], while others replace denoising with flow matching to achieve competitive quality in fewer steps [18]. Despite these improvements, all generative approaches still require multiple network evaluations at inference time, which limits their use in real-time settings.

## 2.2 Non-Generative Point Cloud Completion

Single-pass completion has long been standard at the object level, where methods using synthetic benchmarks [3, 27] largely follow a common pattern: encode the partial cloud, decode a coarse point set, and refine it in a coarse-to-fine manner [8,23,27]. Transformer-based variants extend this by generating structureaware queries for the missing regions via FPS and k-NN grouping [26], following early transformers applied to point clouds [4]. Recently, LiNeXt [6] brought the single-pass paradigm to the scene scale. It constructs an initial point set by replicating the observed points and perturbing them with fixed-variance Gaussian noise, then refines the entire set in a single forward pass. This eliminates the need for iterative sampling and achieves substantial reductions in inference time compared with difusion-based methods. However, it inherits the FPS and k-NN neighborhood operators of object-level architectures, whose cost grows rapidly with the hundreds of thousands of points in outdoor scenes, and its fixed-noise initialization introduces a dataset-specific prior that confines spatial coverage to local neighborhoods around the observed points, leaving large occluded regions weakly constrained.

The scene-level feature aggregation has been addressed in a parallel line of work on autonomous driving perception. Deformable attention [31] evaluates features at a small number of learned sampling locations instead of attending densely over all positions, and, combined with Bird’s-Eye-View representations [13], has become a standard tool across 3D detection [15, 25, 28], segmentation [5, 24], and occupancy prediction [1, 11, 20]. However, they operate on a small set of learned object or grid queries that predict labels at fixed locations. In contrast, for scene completion, the queries are the hundreds of thousands of output points themselves, whose 3D positions must first be initialized to cover unobserved regions and are then refined by the aggregated context.

Our method follows the single-pass paradigm but makes the initialization a learned, structure-conditioned component rather than a fixed random perturbation, and replaces point-neighborhood operators with deformable cross-attention over multi-scale BEV features, where the queries are the output points themselves, initialized to cover unobserved regions and refined by the aggregated context.

## 3 Method

![](images/93325392ad5fa65d02d08e4e6424a3495ba70f71f462344581d6e25347749bad.jpg)  
Fig. 2: Overview of RapidLiDAR. The Multi-Scale Feature Extraction module (center) voxelizes the input point cloud $\boldsymbol { X } \in \mathbb { R } ^ { M \times 3 }$ and extracts multi-scale 3D voxel features $( F _ { 1 } , F _ { 2 } , \ldots , F _ { n } )$ and a dense 2D BEV feature map $B _ { \mathrm { d e n s e } }$ . The Adaptive Initialization Module (top) predicts a spatially varying displacement ∆ for each point in $\tilde { P }$ to obtain the initialized scene $P _ { \mathrm { i n i t } }$ . The Multi-Scale Reconstruction Module (bottom) projects voxel features into BEV feature maps and cross-attends per-point features from $P _ { \mathrm { i n i t } }$ with the multi-scale BEV feature maps using multi-scale deformable attention. A final MLP predicts a residual displacement that aligns each point with the underlying target surfaces, producing the completed scene.

Given an incomplete LiDAR point cloud $\boldsymbol { X } \in \mathbb { R } ^ { M \times 3 }$ , the goal is to predict a complete scene $\bar { P ( \in \mathbb { R } ^ { N \times 3 } }$ , where M and N denote the number of input and output points, respectively, with $N > M$ . We obtain the complete scene in a single forward pass from X to P by training a neural network that directly optimizes the diference between the predicted and ground-truth scenes. Our overall architecture is illustrated in Fig. 2.

We first voxelize the input X and obtain multi-scale features (Sec. 3.1). Using these multi-scale features, the scene is completed in two stages, both trained end-to-end. In the first stage, we propose an adaptive initialization module that learns the initial coarse scene by predicting a spatially varying displacement for each point in ${ \tilde { P } } ,$ which is an expanded version of the partial input scene X (Sec. 3.2). In the second stage, we propose a multi-scale reconstruction module that refines this initial scene into a complete and coherent scene (Sec. 3.3). Finally, the network is trained end-to-end to minimize the Chamfer distance between the reconstructed and ground-truth scenes (Sec. 3.4).

## 3.1 Multi-Scale Feature Extraction

![](images/3f75f7fbed159849d2499f875edfd922a8a5ae24b76bc20d9861555d1420d86a.jpg)  
Fig. 3: Illustration of Dense BEV Head. Converts sparse 3D volumetric features into a dense BEV map. The channel and depth dimensions are merged and projected to $C _ { \mathrm { o u t } }$ using a 2D convolution. Multi-head self-attention over BEV tokens captures global scene context, followed by residual 2D convolutions for feature refinement.

In this module, we obtain multi-scale features from the partial input scene X. These features are then used in the subsequent stages to produce the initial coarse scene and the final completed scene.

To this end, we first voxelize the partial scene X at a resolution of η, which results in a voxel grid of size $[ 1 , D , H , W ]$ , where D, H, and W denote the depth (Z-axis), height (Y-axis), and width (X-axis), respectively, and the single channel indicates whether the corresponding voxel is occupied. We then apply a sequence of 3D convolutional layers, each of which downsamples the input by a factor of $2 ,$ to obtain a set of multi-scale voxel features $\{ F _ { i } \} _ { i = 1 } ^ { n }$ , where $F _ { i } \in \mathbb { R } ^ { C _ { i } \times d _ { i } \times h _ { i } \times w _ { i } }$ with $d _ { i } = D / 2 ^ { i } , h _ { i } = H / 2 ^ { i }$ , and $w _ { i } = W / 2 ^ { i }$ denoting the depth, height, and width at scale i.

While these voxel features capture fine-grained geometry, they remain sparse as only a small percentage of voxels are occupied, and the vast majority are empty due to the sparsity of LiDAR scans. Therefore, a dense representation of the scene is important and should also encode information about the surrounding sparse and occluded regions.

To obtain the dense feature map, we project the last voxel feature $F _ { n } \in$ $\mathbb { R } ^ { C _ { n } \times d _ { n } \times h _ { n } \times w _ { r } }$ into a 2D bird’s-eye-view (BEV) map using a dedicated BEV head, which applies a series of operations. Specifically, we first reshape $F _ { n }$ by merging its channel and depth dimensions into $\left( C _ { n } \cdot d _ { n } , h _ { n } , w _ { n } \right)$ , and then project this representation to a BEV feature map $B _ { \mathrm { p r o j } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times h _ { n } \times w _ { n } }$ using a single 2D convolution.

To further enrich the features with information from neighboring regions, we apply multi-head self-attention (MHSA) [21]. Specifically, we treat each spatial location of $B _ { \mathrm { p r o j } }$ as an individual token, reshaping it into a sequence of tokens $S \in \mathbb { R } ^ { ( h _ { n } \cdot w _ { n } ) \times \overset { * } { C } _ { \mathrm { o u t } } ^ { \bullet } }$ <sup>t</sup> , which is refined through MHSA. This enables the sequence S to carry surrounding spatial context and encode information about sparse and occluded regions. Finally, we reshape the sequence S back into a 2D feature map $( C _ { \mathrm { o u t } } , h _ { n } , w _ { n } )$ and further refine it with two convolutional layers connected by a skip connection. This produces the dense BEV feature map $B _ { \mathrm { d e n s e } }$ . We illustrate the BEV head operations in Fig. 3.

Together, the sparse multi-scale 3D voxel features $F _ { 1 } , F _ { 2 } , \ldots , F _ { n }$ and the dense 2D BEV feature map $B _ { \mathrm { d e n s e } }$ form a hybrid representation of the scene. Both the adaptive initialization module and the multi-scale reconstruction module use this hybrid representation to construct the coarse initialized scene and the final completed scene, respectively.

## 3.2 Adaptive Initialization Module

Given the partial input scene $\boldsymbol { X } \in \mathbb { R } ^ { M \times 3 }$ and the multi-scale features $F _ { 1 } , F _ { 2 } , \ldots , F _ { n }$ and $B _ { \mathrm { d e n s e } } .$ the objective of this module is to learn the coarse scene $P _ { \mathrm { i n i t } } \in \mathbb { R } ^ { N \times 3 }$ that serves as initialization for further refinement. Since the number of input points M is smaller than the number of output points N required to complete the scene, we repeat each observed point $( \lfloor N / M \rfloor )$ times and perturb the repeated points with small Gaussian noise $\sigma _ { \mathrm { i n i t } }$ to avoid duplicates. Concatenating these perturbed points with the original partial scene X gives the point set $\tilde { P } \in \mathbb { R } ^ { N \times 3 }$ that matches the required number of target points.

To learn a coarse scene that can better cover sparse and occluded regions, we predict a spatially adaptive displacement for each point in ${ \tilde { P } } .$ . By moving each point according to its learned displacement, the points are adaptively redistributed to match the target scene structure.

To predict a displacement for each point, we need a feature vector that captures both local geometry and broader scene context around that point. The multi-scale voxel features $F _ { 1 } , F _ { 2 } , \ldots , F _ { n }$ and the dense BEV feature map $B _ { \mathrm { d e n s e } }$ already contain this information, but are defined over their respective grids rather than over individual points. We therefore obtain a feature for each point by interpolating from these grids. Specifically, for each point $p _ { i } = ( x , y , z ) \in \tilde { P }$ we project $p _ { i }$ onto the voxel grid of each multi-scale feature map $F _ { k }$ , obtaining a voxel index $\sigma _ { k } ( p _ { i } )$ at scale $k ,$ and sample the corresponding feature vector $f _ { i } ^ { ( k ) } \in \mathbb { R } ^ { C _ { k } }$ through interpolation. Similarly, we project $p _ { i }$ onto the BEV grid to obtain a cell index $\pi ( p _ { i } )$ in $B _ { \mathrm { d e n s e } }$ and sample the corresponding feature vector $b _ { i } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ . Concatenating these gives the per-point feature vector

$$
f _ { i } = \operatorname { c o n c a t } \left( f _ { i } ^ { ( 1 ) } , f _ { i } ^ { ( 2 ) } , \ldots , f _ { i } ^ { ( n ) } , b _ { i } \right) \in \mathbb { R } ^ { d } ,\tag{1}
$$

where $d = C _ { 1 } + C _ { 2 } + \cdot \cdot \cdot + C _ { n } + C _ { \mathrm { o u t } }$ . Doing this for every point in $\tilde { P }$ in parallel gives the per-point feature matrix $\mathcal { F } \in \mathbb { R } ^ { N \times d }$

Since directly indexing into these grids is a discrete, non-diferentiable operation, we instead sample features using trilinear interpolation on the 3D voxel grids and bilinear interpolation on the 2D BEV grid, which allows smooth gradient flow during backpropagation. Trilinear interpolation distributes the sampling weights among the eight voxels neighboring $\sigma _ { k } ( p _ { i } )$ , while bilinear interpolation distributes the weights among the four cells neighboring $\pi ( p _ { i } )$

Finally, the per-point feature matrix $\mathcal { F } \in \bar { \mathbb { R } ^ { N \times d } }$ is passed through a Multi-Layer Perceptron (MLP) that predicts a spatially varying displacement $\varDelta \in \mathbb { R } ^ { N \times 3 }$ The coarse scene initialization $P _ { \mathrm { i n i t } } \in \mathbb { R } ^ { \tilde { N } \times 3 }$ is then obtained as

$$
P _ { \mathrm { i n i t } } = \tilde { P } + \varDelta \cdot S _ { \mathrm { m a x } } ,\tag{2}
$$

where $S _ { \mathrm { m a x } }$ is a fixed scalar that defines the scale of the scene.

The learned displacement for each point can adapt to the local scene structure, in contrast to a fixed global noise. This adaptive strategy provides a betterinformed initialization for the subsequent multi-scale reconstruction module, detailed in the following section.

## 3.3 Multi-Scale Reconstruction Module

Given the coarse initialization $P _ { \mathrm { i n i t } }$ from the adaptive initialization module, and the multi-scale features $F _ { 1 } , \ldots , F _ { n } , B _ { \mathrm { d e n s e } }$ from the feature extraction module, the goal of this module is to reconstruct a complete and coherent 3D scene.

To this end, we obtain a feature vector for each point in $P _ { \mathrm { i n i t } }$ through the same interpolation procedure described in the previous section, giving perpoint features $\mathcal { F } \in \mathbb { R } ^ { N \times d }$ . We reconstruct the scene using a transformer-based decoder, where the per-point features act as queries and the multi-scale features $F _ { 1 } , \ldots , F _ { n } , B _ { \mathrm { d e n s e } }$ act as context. A standard transformer decoder aggregates context through cross-attention [21], in which each of the N point queries attends to every position in $F _ { 1 } , \ldots , F _ { n } , B _ { \mathrm { d e n s e } }$ . This scales as $\mathcal { O } ( N \cdot K _ { \mathrm { c t x } } )$ , where $K _ { \mathrm { c t x } }$ is the total number of context elements across spatial grids. Given hundreds of thousands of point queries, full attention across all volumetric and BEV locations is computationally prohibitive.

Therefore, we use multi-scale deformable attention [31] between the point features $\mathcal { F }$ and the multi-scale features $F _ { 1 } , \ldots , F _ { n } , B _ { \mathrm { d e n s e } } .$ , which act as context. Deformable attention restricts attention to a small set of learned sampling locations within each context feature, rather than attending to all positions within each scale. This allows the point features to gather the relevant information required to complete the scene, without incurring high complexity.

However, current deformable attention implementations are optimized primarily for 2D feature maps rather than for 3D voxel features $F _ { 1 } , F _ { 2 } , \ldots , F _ { n } .$ We therefore also project each of these features to 2D BEV representations. Specifically, given a voxel feature $F _ { i }$ of shape $\left( C _ { i } , d _ { i } , h _ { i } , w _ { i } \right)$ , we merge its channel and depth dimensions into $( C _ { i } \cdot d _ { i } , h _ { i } , w _ { i } )$ and project it to $( C _ { \mathrm { o u t } } , h _ { i } , w _ { i } )$ using a

2D convolution. This results in multi-scale BEV feature maps $B _ { 1 } , \ldots , B _ { n }$ , one for each 3D voxel feature.

We apply multi-scale deformable attention, with the point features $\mathcal { F }$ as queries and the multi-scale BEV features $\left\{ B _ { 1 } , B _ { 2 } , \ldots , B _ { n } , B _ { \mathrm { d e n s e } } \right\}$ as keys and values. This gives the refined features $F _ { \mathrm { r e f } }$ as

$$
F _ { \mathrm { r e f } } = \mathrm { M S - D e f o r m A t t n } ( \mathcal { F } , \pi ( P _ { \mathrm { i n i t } } ) , \{ B _ { 1 } , \dots , B _ { n } , B _ { \mathrm { d e n s e } } \} ) ,\tag{3}
$$

where $\pi ( \cdot )$ projects the 3D coordinates in $P _ { \mathrm { i n i t } }$ onto the BEV plane to obtain reference points for the attention mechanism, and $\{ B _ { 1 } , \ldots , B _ { n } , B _ { \mathrm { d e n s e } } \}$ denotes the multi-scale BEV feature maps, including the dense BEV map from the BEV head.

Subsequently, the refined features $F _ { \mathrm { r e f } }$ are passed through an MLP to predict a final 3D residual displacement $\varDelta P _ { \mathrm { r e f } } \in \mathbb { R } ^ { N \times 3 }$ . This residual allows the network to adjust the initialized points so that they can align with the underlying target surfaces. The final completed scene is obtained by applying the displacement:

$$
P = P _ { \mathrm { i n i t } } + \varDelta P _ { \mathrm { r e f } } .\tag{4}
$$

This refinement enables the network to use long-range geometric context to construct the scene without relying on point-neighborhood or spatial downsampling operations such as farthest point sampling (FPS) or k-nearest neighbor (k-NN), whose computational cost scales with the number of points.

## 3.4 Loss Formulation

Our network is trained end-to-end by directly optimizing the geometric fidelity of the predicted point sets. To quantify the spatial discrepancy between a predicted point cloud $P$ and the ground-truth complete scene $P _ { \mathrm { g t } }$ , we use the standard Chamfer Distance (CD) metric. Chamfer Distance computes bi-directional nearestneighbor distances between points in $P$ and $P _ { \mathrm { g t } }$ , encouraging the prediction to cover the ground-truth surfaces while penalizing outliers:

$$
\mathcal { L } _ { \mathrm { C D } } ( P , P _ { \mathrm { g t } } ) = \frac { 1 } { | P | } \sum _ { p \in P } \operatorname* { m i n } _ { q \in P _ { \mathrm { g t } } } | p - q | _ { 2 } ^ { 2 } + \frac { 1 } { | P _ { \mathrm { g t } } | } \sum _ { q \in P _ { \mathrm { g t } } } \operatorname* { m i n } _ { p \in P } | q - p | _ { 2 } ^ { 2 } .\tag{5}
$$

## 4 Experiments

In this section, we evaluate the efectiveness, eficiency, and generalizability of our method. We show that our proposed method achieves state-of-the-art performance while reducing inference time. We also assess the zero-shot behavior of our model without additional retraining.

## 4.1 Datasets and Evaluation Metrics

Datasets. Following the standard protocols established by recent scene completion benchmarks [19,29], we train and evaluate our method on the SemanticKITTI [2] dataset. Specifically, we use sequences 00–10 for training, excluding sequence 08, which is reserved for validation. To test the zero-shot generalization, we directly evaluate the model trained on SemanticKITTI on sequence 00 of the KITTI-360 [14] dataset without fine-tuning. During training, the partial scene is downsampled to 18,000 points, and the corresponding complete scene contains 180,000 points, following the standard practice.

Evaluation Metrics. To evaluate completion performance, we use standard geometric metrics. We measure overall geometric accuracy using the Chamfer Distance (CD), computed as a bidirectional nearest-neighbor distance, as defined in Eq. 5. To assess similarity in spatial distribution, we compute Jensen–Shannon divergence (JSD) in both 3D space and Bird’s-Eye-View (BEV) representation.

## 4.2 Implementation Details

Following prior benchmarks, we set $M \ : = \ : 1 8 , 0 0 0$ and $N = 1 8 0 { , } 0 0 0 .$ , which yields an expansion ratio $\lfloor N / M \rfloor = 1 0$ . The input is voxelized at a resolution of $\eta = 0 . 3$ m to form an occupancy grid of size $[ 1 , D , H , W ] = [ 1 , 2 0 , 3 3 3 , 3 3 3 ]$ 2 which the multi-scale feature extraction backbone processes at $n = 4$ levels with channel dimensions $\left[ C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 } \right] = \left[ 3 2 , 6 4 , 1 2 8 , 2 5 6 \right]$ . The last level is compressed by the BEV head into $C _ { \mathrm { o u t } } = 5 1 2$ channels with 4 self-attention heads. We set the noise scale $\sigma _ { \mathrm { i n i t } } ~ = ~ 0 . 1$ m for the initial point repetition. During feature interpolation, the sampled features from all four voxel levels and the dense BEV map are concatenated into a per-point feature of dimension $d = C _ { 1 } + C _ { 2 } + C _ { 3 } + C _ { 4 } + C _ { \mathrm { o u t } } = 9 9 2$ , then projected to the model width $C _ { \mathrm { o u t } } = 5 1 2$ The predicted displacement is bounded by a maximum expansion $S _ { \mathrm { m a x } } = 5 0 \mathrm { m }$ In the reconstruction module, each voxel level is likewise projected to a BEV map with $C _ { \mathrm { o u t } } = 5 1 2$ channels, for a total of $L = 5$ feature levels $\{ B _ { 1 } , \ldots , B _ { 4 } , B _ { \mathrm { d e n s e } } \}$ that serve as keys and values for 2 deformable cross-attention stages with $n _ { h } = 8$ heads and $K = 4$ sampling points per head.

We train with the Adam optimizer, a base learning rate of $1 \times 1 0 ^ { - 4 }$ , and a cosine learning rate schedule, using a batch size of 4 on a single NVIDIA RTX 6000 Ada GPU; the model converges within 30 epochs. Inference times are measured on the same GPU for all methods, using oficial code and released checkpoints.

Refinement Network. Existing baselines additionally train a refinement network on top of the base completion model, first introduced by Nunes et al. [19]. To ensure a fair comparison, we likewise train a refinement network on top of our completion model. Our refinement network reuses the multi-scale features extracted by the frozen completion network: per-point features are interpolated at each completed point, and four successive stages of deformable cross-attention over the BEV maps predict $\kappa = 6$ residual ofsets per point, upsampling the scene by a factor of 6×. The refinement network is trained separately for 5 epochs, optimizing the Chamfer Distance in Eq. 5.

Table 1: Scene Completion on SemanticKITTI and KITTI-360. Quantitative comparison with prior methods. † denotes methods with an additional refinement. Best results in each group are highlighted in bold.
<table><tr><td></td><td colspan="3">SemanticKITTI</td><td colspan="3">KITTI-360</td></tr><tr><td>Method</td><td>CD↓</td><td>JSD 3D ↓</td><td>JSD BEV ↓</td><td>CD↓</td><td>JSD 3D ↓</td><td>JSD BEV↓</td></tr><tr><td>LMSCNet</td><td>0.641</td><td></td><td>0.431</td><td>0.979</td><td></td><td>0.496</td></tr><tr><td>LODE</td><td>1.029</td><td></td><td>0.451</td><td>1.565</td><td></td><td>0.483</td></tr><tr><td>MID</td><td>0.503</td><td></td><td>0.470</td><td>0.637</td><td></td><td>0.476</td></tr><tr><td>PVD</td><td>1.256</td><td></td><td>0.498</td><td></td><td></td><td></td></tr><tr><td>LiDiff</td><td>0.434</td><td>0.564</td><td>0.444</td><td>0.564</td><td></td><td>0.459</td></tr><tr><td>LiDPM</td><td>0.446</td><td>0.532</td><td>0.440</td><td></td><td></td><td></td></tr><tr><td>ScoreLiDAR</td><td>0.406</td><td></td><td>0.425</td><td>0.472</td><td></td><td>0.444</td></tr><tr><td>LiFlow</td><td>0.309</td><td></td><td>0.416</td><td></td><td></td><td></td></tr><tr><td>LiNeXt</td><td>0.214</td><td>0.494</td><td>0.336</td><td>0.217</td><td>0.508</td><td>0.355</td></tr><tr><td>Ours</td><td>0.206</td><td>0.475</td><td>0.332</td><td>0.211</td><td>0.492</td><td>0.338</td></tr><tr><td>LiDiff†</td><td>0.376</td><td>0.573</td><td>0.416</td><td>0.517</td><td></td><td>0.446</td></tr><tr><td>ScoreLiDAR†</td><td>0.342</td><td></td><td>0.399</td><td>0.452</td><td></td><td>0.437</td></tr><tr><td>LiDPM†</td><td>0.376</td><td>0.542</td><td>0.403</td><td></td><td></td><td></td></tr><tr><td>LiNeXt†</td><td>0.149</td><td>0.481</td><td>0.331</td><td>0.149</td><td>0.499</td><td>0.339</td></tr><tr><td>Ours†</td><td>0.138</td><td>0.478</td><td>0.330</td><td>0.140</td><td>0.490</td><td>0.336</td></tr></table>

Table 2: Computational Eficiency Comparison. We report Chamfer distance, number of learnable parameters, and inference time per scan.
<table><tr><td>Method</td><td>CD ↓</td><td>Param (M) ↓</td><td>Time (s) ↓</td></tr><tr><td>LiDiff</td><td>0.434</td><td>32.67</td><td>30.1</td></tr><tr><td>ScoreLiDAR</td><td>0.406</td><td>32.67</td><td>7.1</td></tr><tr><td>LiNeXt</td><td>0.214</td><td>1.99</td><td>0.23</td></tr><tr><td>Ours</td><td>0.206</td><td>11.8</td><td>0.10</td></tr></table>

## 4.3 Scene Completion

Table 1 summarizes quantitative results on SemanticKITTI [2] and KITTI-360 [14]. On SemanticKITTI, our method achieves the best results across all reported metrics, attaining the lowest Chamfer Distance, 3D JSD, and BEV JSD among all compared methods. Similarly, on KITTI-360, evaluated in a cross-dataset setting without fine-tuning, our approach follows a similar trend and achieves state-of-the-art performance.

Our main advantage over existing methods is speed. Table 2 reports the computational cost of diferent approaches. Difusion-based generative models, such as LiDif and ScoreLiDAR, require substantially more parameters and incur longer inference times due to their multi-step sampling. LiNeXt uses a compact single-pass architecture with a small parameter count but runs at around 0.23 s per scan, despite its point-neighborhood operations relying on eficient custom CUDA implementations. Our method uses a moderate number of parameters and completes a full scene in 0.1 s, which is 2.3× faster than LiNeXt while also achieving a lower Chamfer Distance. This matches the 10 Hz acquisition rate of typical automotive LiDAR sensors. Notably, our implementation relies only on standard library operations rather than specialized CUDA kernels. The accuracy and eficiency results demonstrate that our approach achieves the best trade-of among prior methods for reconstruction quality, model size, and inference speed.

## 4.4 Ablation Studies

We perform ablation studies to analyze the influence of the main architectural components and voxel resolution on reconstruction quality and eficiency. First, to assess the contribution of each component, we perform an ablation study on the SemanticKITTI validation set, summarized in Table 3. In the first ablation, we replace the Adaptive Initialization Module (AIM) with a fixed perturbation scale of $\sigma _ { \mathrm { i n i t } } = 1 . 0$ following LiNeXt [6], removing per-point displacement prediction. In the second, we remove the Multi-Scale Reconstruction Module (MSRM) and predict the final scene directly from the adaptively initialized point set, bypassing deformable cross-attention entirely. Both ablations lead to a consistent drop across all three metrics, indicating that each component contributes to the final result.

Table 3: Architectural Ablation Study. We evaluate the impact of our core modules on the SemanticKITTI validation set. Best results are in bold.
<table><tr><td>Method</td><td colspan="2">CD ↓ JSD 3D ↓ JSD BEV ↓</td></tr><tr><td>Ours</td><td>0.206 0.475</td><td>0.332</td></tr><tr><td>Ours  $\mathrm { w } / \mathrm { o }$  AIM</td><td>0.218 0.488</td><td>0.345</td></tr><tr><td>Ours w/o MSRM 0.215</td><td>0.494</td><td>0.342</td></tr></table>

Furthermore, we evaluate the efect of the maximum displacement bound $S _ { \mathrm { m a x } }$ on reconstruction performance. We train the network with $S _ { \mathrm { m a x } } \in \{ 5 0 , 7 0 , 1 0 0 \}$ and report Chamfer Distance on a downsampled validation set of 180,000 output points. As shown in Table 4, performance is largely insensitive to the choice of $S _ { \mathrm { m a x } }$ . This indicates that the adaptive initialization module is robust across a reasonable range of displacement scales, and we use $S _ { \mathrm { m a x } } = 5 0$ as our default setting.

Table 4: Efect of Maximum Displacement Bound. Reconstruction performance for diferent values of $S _ { \mathrm { m a x } }$ , evaluated on a downsampled validation set of 180,000 output points.
<table><tr><td> $S _ { \mathrm { m a x } }$  CD↓</td></tr><tr><td>50 0.2594</td></tr><tr><td>70 0.2592</td></tr><tr><td>100 0.2589</td></tr></table>

Table 5: Voxel Resolution Ablation. Impact of voxel resolution η on reconstruction performance, parameter count, and inference time.
<table><tr><td>η (m)</td><td>CD↓</td><td>Param (M)</td><td>Time (s) ↓</td></tr><tr><td>0.5</td><td>0.214</td><td>10.0</td><td>0.07</td></tr><tr><td>0.4</td><td>0.210</td><td>11.6</td><td>0.09</td></tr><tr><td>0.3</td><td>0.206</td><td>11.8</td><td>0.10</td></tr><tr><td>0.2</td><td>0.208</td><td>11.9</td><td>0.14</td></tr></table>

Finally, Table 5 reports the efect of voxel resolution on reconstruction quality, parameter count, and inference time. Finer resolutions consistently improve the Chamfer Distance, while the growth in both parameter count and inference time remains moderate. We select $\eta = 0 . 3$ as it ofers a suitable trade-of across all three criteria, though the architecture allows adjustment of voxel resolution to meet diferent speed or quality requirements.

## 4.5 Qualitative Analysis

We also examine qualitative results on SemanticKITTI to visualize how different methods reconstruct occluded regions and large-scale scene geometry. Representative examples for SemanticKITTI are shown in Figure 4. Generative difusion-based methods such as LiDif and ScoreLiDAR produce visually plausible completions, but the reconstructed geometry can deviate from the ground-truth layout, especially in large occluded regions. LiNeXt, which directly optimizes the Chamfer Distance, often generates completions that are closer to the ground-truth but sometimes leaves parts of the scene under-completed due to its fixed-noise initialization. Our method produces completions that follow the ground-truth structure while providing broader coverage in regions with limited observations, which we attribute to the adaptive initialization not being restricted around the observed points.

![](images/f4415d3b9fdf51879251b91dd6f5ab3f58826b5e119a4fa39a5cb21977139ba3.jpg)  
Fig. 4: Qualitative Comparison on SemanticKITTI. Our method produces more complete geometry in large occluded regions compared to prior methods.

## 5 Conclusion

We introduced RapidLiDAR, a LiDAR scene completion method that learns the initialization itself directly from data. Its adaptive initialization module predicts a spatially varying displacement for each partial input point, expanding these observations into a coarse scene initialization that adapts to the local geometry and removes the need for manual noise tuning. A subsequent multiscale reconstruction module then refines this coarse initialization into a complete and coherent scene by predicting residual displacements from multi-scale voxel and bird’s-eye-view features extracted from the partial input scan. Because it replaces point-neighborhood operators, such as farthest point sampling and k-nearest neighbor search, with voxel- and BEV-based feature extraction, the resulting architecture runs faster and handles diferent input resolutions.

On SemanticKITTI and KITTI-360, our experiments confirmed completion performance on par with the state of the art, with a full scene completed in 0.1 seconds, which is 2.3 times faster than the fastest prior method and in line with the 10 Hz acquisition rate of typical automotive LiDAR sensors.

We plan to extend this work to generalize across diferent sensor configurations. Evaluating robustness against domain shifts in sensor beam geometries ofers a promising direction for future research.

## Acknowledgment

We acknowledge funding received from the Bavarian Ministry of Economic Afairs, Regional Development and Energy, in the scope of the funded project ”Bavarian Advanced Resolution Radar (BAVAR-RADAR)”, funding label DIK0622.

## References

1. Agro, B., Sykora, Q., Casas, S., Gilles, T., Urtasun, R.: Uno: Unsupervised occupancy fields for perception and forecasting. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 14487–14496 (2024)

2. Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., Gall, J.: Semantickitti: A dataset for semantic scene understanding of lidar sequences. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9296–9306 (2019)

3. Chang, A.X., Funkhouser, T.A., Guibas, L.J., Hanrahan, P., Huang, Q.X., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., Xiao, J., Yi, L., Yu, F.: Shapenet: An information-rich 3d model repository. ArXiv abs/1512.03012 (2015)

4. Engel, N., Belagiannis, V., Dietmayer, K.C.J.: Point transformer. IEEE Access 9, 134826–134840 (2020)

5. Ge, C., Chen, J., Xie, E., Wang, Z., Hong, L., Lu, H., Li, Z., Luo, P.: Metabev: Solving sensor failures for 3d detection and map segmentation. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 8687–8697 (2023)

6. He, W., Chen, X., Wang, R., Li, R., Pi, H., Zhang, J., Tang, Z., Li, K.: Linext: Revisiting lidar completion with eficient non-difusion architectures. In: AAAI Conference on Artificial Intelligence (2026)

7. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

8. Hussian, A., Ritthaler, M., Kaup, A., Belagiannis, V.: I2pref: Image-driven point completion with iterative refinement. In: Proceedings of the 34th European Signal Processing Conference (2026)

9. Kim, J., Park, B., Kim, J.: Empirical analysis of autonomous vehicle’s lidar detection performance degradation for actual road driving in rain and fog. Sensors (Basel, Switzerland) 23 (2023)

10. Lee, J., Im, W., Lee, S., Yoon, S.E.: Difusion probabilistic models for scene-scale 3d categorical data. ArXiv abs/2301.00527 (2023)

11. Li, J., He, X., Zhou, C., Cheng, X., Wen, Y., Zhang, D.: Viewformer: Exploring spatiotemporal modeling for multi-view 3d occupancy perception via view-guided transformers. In: European Conference on Computer Vision (2024)

12. Li, P., Zhao, R., Shi, Y., Zhao, H., Yuan, J., Zhou, G., Zhang, Y.Q.: Lode: Locally conditioned eikonal implicit scene completion from sparse lidar. IEEE International Conference on Robotics and Automation (ICRA) pp. 8269–8276 (2023)

13. Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Yu, Q., Dai, J.: Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In: European Conference on Computer Vision (2022)

14. Liao, Y., Xie, J., Geiger, A.: Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 3292–3310 (2021)

15. Liu, X., Zheng, C., Qian, M., Xue, N., Chen, C., Zhang, Z., Li, C., Wu, T.: Multi-view attentive contextualization for multi-view 3d object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16688–16698 (June 2024)

16. Luo, S., Hu, W.: Difusion probabilistic models for 3d point cloud generation. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 2836–2844 (2021)

17. Martyniuk, T., Puy, G., Boulch, A., Marlet, R., de Charette, R.: Lidpm: Rethinking point difusion for lidar scene completion. In: 2025 IEEE Intelligent Vehicles Symposium (IV) (2025)

18. Matteazzi, A., Tutsch, D.: Liflow: Flow matching for 3d lidar scene completion. ArXiv abs/2602.02232 (2026)

19. Nunes, L., Marcuzzi, R., Mersch, B., Behley, J., Stachniss, C.: Scaling difusion models to real-world 3d lidar scene completion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14770–14780 (2024)

20. Tong, W., Sima, C., Wang, T., Wu, S., Deng, H., Chen, L., Gu, Y., Lu, L., Luo, P., Lin, D., Li, H.: Scene as occupancy. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 8372–8381 (2023)

21. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Neural Information Processing Systems (2017)

22. Vizzo, I., Mersch, B., Marcuzzi, R., Wiesmann, L., Behley, J., Stachniss, C.: Make it dense: Self-supervised geometric scan completion of sparse 3d lidar scans in large outdoor environments. IEEE Robotics and Automation Letters 7, 8534–8541 (2022)

23. Xiang, P., Wen, X., Liu, Y.S., Cao, Y.P., Wan, P., Zheng, W., Han, Z.: Snowflakenet: Point cloud completion by snowflake point deconvolution with skip-transformer. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 5479–5489 (2021)

24. Xu, R., Tu, Z., Xiang, H., Shao, W., Zhou, B., Ma, J.: Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers. In: Conference on Robot Learning (2022)

25. Xue, Z., Guo, M., Fan, H., Zhang, S., Zhang, Z.: Corrbev: Multi-view 3d object detection by correlation learning with multi-modal prototypes. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 27413–27423 (2025)

26. Yu, X., Rao, Y., Wang, Z., Liu, Z., Lu, J., Zhou, J.: Pointr: Diverse point cloud completion with geometry-aware transformers. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 12478–12487 (2021)

27. Yuan, W., Khot, T., Held, D., Mertz, C., Hebert, M.: Pcn: Point completion network. International Conference on 3D Vision (3DV) pp. 728–737 (2018)

28. Zhang, J., Zhang, Y., Liu, Q., Wang, Y.: Sa-bev: Generating semantic-aware bird’s-eye-view feature for multi-view 3d object detection. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 3325–3334 (2023)

29. Zhang, S., Zhao, A., Yang, L., Li, Z., Meng, C., Xu, H., Chen, T., Wei, A., Gu, P.P., Sun, L.: Distilling difusion models to eficient 3d lidar scene completion. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 5007–5016 (2024)

30. Zhou, L., Du, Y., Wu, J.: 3d shape generation and completion through point-voxel difusion. IEEE/CVF International Conference on Computer Vision (ICCV) pp. 5806–5815 (2021)

31. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. ArXiv abs/2010.04159 (2020)