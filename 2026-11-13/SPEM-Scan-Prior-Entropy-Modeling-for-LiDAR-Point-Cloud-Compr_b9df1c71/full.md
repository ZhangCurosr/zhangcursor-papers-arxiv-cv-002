# SPEM: Scan-Prior Entropy Modeling for LiDAR Point Cloud Compression

Bojun Liu Yangzhi Ma Jie Li Dong Liu Li Li

University of Science and Technology of China

{liubj,mayz,lijl}@mail.ustc.edu.cn

{dongeliu,lil1}@ustc.edu.cn

## Abstract

LiDAR point clouds exhibit scan-induced spatial regularities that are informative for geometry compression. Existing scan-aware approaches often exploit these regularities by transforming coded geometry into spherical or beam-aligned coordinate systems. However, this couples acquisition priors with geometry representation, altering spatial partitioning and quantization, and potentially preventing exact recovery of the original Cartesian geometry. We argue that acquisition priors should guide probability modeling rather than dictate the coding representation. We propose SPEM, a Scan-Prior Entropy Modeling framework that exploits LiDAR scanning regularities within the entropy model without requiring coordinate transformation. We introduce a multi-frequency spherical rotary positional encoding that captures scan-aligned spatial relationships in both backbone self-attention and decodedcontext retrieval. Building upon this encoding, we develop a decoded-memory autoregressive predictor that efficiently exploits previously decoded occupancy information through reusable key/value caches. Experiments demonstrate stateof-the-art performance in both lossy and lossless LiDAR geometry compression, highlighting the effectiveness of scanprior modeling across different coordinate representations.

## 1. Introduction

LiDAR has become a fundamental sensing modality for autonomous driving, robotics, and large-scale 3D scene understanding [1, 3, 22, 27], providing accurate geometric measurements of the surrounding environment. With the rapid development of high-resolution and high-frequency LiDAR sensors, modern systems can capture increasingly dense and precise 3D observations, supporting a wide range of tasks such as localization, perception, mapping, and scene reconstruction. However, continuously acquired LiDAR point clouds contain massive numbers of 3D points with high coordinate precision, imposing substantial storage and transmission overhead [9, 10, 23]. Efficient point cloud geometry compression is therefore essential for reducing this cost while preserving geometric fidelity.

Unlike regularly images and videos, point clouds are inherently unordered and irregular. Existing compression methods therefore introduce structured representations to organize point geometry and exploit its spatial or acquisition characteristics, including octrees [8, 11], predtree [25 26], one-chain structures [15], range images [35, 40], and sparse tensors [31, 32]. Beyond generic geometric structures, recent methods further exploit the acquisition prior of rotating LiDAR. For example, SCP [18] transforms Cartesian coordinates into spherical coordinates to expose the scanning pattern, while PACE [41] further incorporates Li-DAR intrinsic information through a cylindrical-beam representation. Both representations are organized into octree structures for entropy modeling, leading to improved compression performance. However, these approaches rely on scan-aligned coordinate transformations, which alter spatial partitioning and quantization and may prevent exact recovery of the original Cartesian geometry. More fundamentally, this raises a question: can LiDAR scanning priors be exploited directly within the entropy model, without requiring a specific geometry representation?

Our key insight is that LiDAR scanning priors are reflected in the spatial relationships of points, rather than tied to a particular coordinate representation. Even in a Cartesian octree, node locations retain the range and angular relationships induced by the LiDAR acquisition process. Based on this observation, we propose SPEM, a Scan-Prior Entropy Modeling framework that incorporates these scan-aware relationships into occupancy probability modeling without requiring coordinate transformation. This decoupling preserves the original geometry for strict lossless coding while enabling effective entropy modeling for both lossy and lossless compression.

Specifically, SPEM contains two key designs. First, we introduce a spherical positional encoding that maps the physical positions of octree tokens into spherical coordinates (r, θ, ϕ) for attention computation, without altering the underlying Cartesian octree. Multi-frequency rotary encoding is then applied to the attention queries and keys, allowing the model to capture scan-aligned spatial dependencies while leaving the coded Cartesian geometry unchanged. Second, we develop a stage-scalable decoded-memory predictor, where the current tokens query previously decoded occupancy information through crossattention. The decoded tokens are encoded into reusable key/value caches and progressively accumulated across prediction stages, enabling effective autoregressive context modeling without repeatedly recomputing historical representations. Together, these designs allow SPEM to explicitly preserve Cartesian geometry while implicitly exploiting the LiDAR scanning prior for both lossy and lossless point cloud compression.

In summary, the proposed SPEM offers the following key contributions:

• We introduce a spherical rotary positional encoding that injects the LiDAR scanning prior directly into the entropy model, enabling scan-aware spatial modeling without requiring explicit coordinate transformation.

• Building upon the proposed positional encoding, we develop a cross-attention-based autoregressive probability predictor that efficiently exploits previously decoded occupancy information through reusable K/V memory, substantially reducing autoregressive inference latency.

• Extensive experiments on multiple LiDAR datasets demonstrate that SPEM achieves state-of-the-art performance for both lossy and lossless geometry compression, with particularly substantial bitrate reductions for lossless compression in the Cartesian coordinate space.

## 2. Related Work

## 2.1. Voxel-based Point Cloud Compression

Voxel-based methods quantize point cloud geometry into regular voxels and learn compact representations or occupancy distributions with 3D convolutional networks. Early approaches [20, 30] used dense 3D convolutions, but processing numerous empty voxels incurs substantial overhead. Sparse convolution [5, 28, 29] alleviates this issue by operating only on occupied locations. Wang et al. [31] introduced a multiscale sparse tensor representation for progressive occupancy prediction, later extended by SparsePCGC [32] with multiscale occupancy modeling. Unicorn [33] further developed universal multiscale conditional coding, while UniPCGC [34] introduced uneven stage-wise occupancy coding and dynamic sparse convolution for flexible rate–complexity control. More recently, [12, 16, 17, 39] reorganized spatial occupancy context into the channel dimension to enlarge the effective context and reduce autoregressive overhead. ELiC [14] improves efficiency through cross-bit-depth feature propagation and adaptive encoder selection while preserving Morton ordering. Despite these advances, voxel-based models mainly aggregate local voxel neighborhoods, with limited effective receptive fields at fine resolutions, particularly for high-precision LiDAR point clouds.

## 2.2. Octree-based Point Cloud Compression

Octree-based methods recursively partition 3D space and encode node occupancy, providing a compact hierarchical representation for LiDAR point clouds. The G-PCC standard [38] adopts octree coding as a fundamental geometry representation and employs hand-crafted context models to estimate occupancy probabilities from previously coded spatial information. Building upon this hierarchical representation, OctSqueeze [11] pioneered learned octree entropy modeling with ancestral context, followed by VoxelContext-Net [21] using local voxel context and OctAttention [8] exploiting sibling and ancestor dependencies through attention. Chen et al. [4] further incorporated sibling context and local surface priors. EHEM [24] introduced hierarchical attention and grouped context for efficient parallel decoding, while subsequent works [6, 13] further improved cross-node dependency modeling and parallel occupancy coding. SCP [18] transformed Cartesian geometry into spherical coordinates to explicitly exploit the LiDAR scanning prior. More recently, TopNet [37] combines local and long-range context, while PACE [41] decouples non-causal context aggregation from stage-scalable causal prediction. In contrast, prior scan-prior-aware methods explicitly exploit the LiDAR scanning prior through coordinate transformation, which incurs precision loss and precludes strict lossless coding. SPEM instead can implicitly exploit the scanning prior directly in the Cartesian octree, preserving the original geometry without requiring coordinate transformation.

## 3. Method

## 3.1. Overall Framework

As illustrated in Fig. 1, SPEM first converts the input Li-DAR point cloud into an octree-based token representation, models their spatial context with a scan-prior-aware backbone, and progressively estimates occupancy code through multi-stage autoregressive prediction for entropy coding.

(i) Token Initialization: Given a LiDAR point cloud in Cartesian coordinates (x, y, z), we optionally apply a scan-aligned transformation, such as spherical [18] or cylindrical-beam coordinates [41], before octree construction. Note that SPEM is less sensitive to this transformation by modeling the scanning prior within the entropy model. We then partition the serialized octree nodes into local windows and aggregate hierarchical context to form the input tokens. Details are provided in Sec. 3.2.

![](images/d56f2c3abe7021f3a4209622d72add20f4135aec8fb65a82ada9f64af83dc8cf.jpg)  
Figure 1. Overview of our LiDAR point cloud compression framework. The base backbone extracts latent tokens from the input LiDAR point cloud, while the autoregressive module predicts occupancy distributions conditioned on previously decoded tokens. We further incorporate LiDAR scanning priors through spherical Q/K RoPE based on physical coordinates.

(ii) Base Backbone: Each local window of 1,024 tokens is fed into a Transformer encoder with $N = 1 2$ layers, where each layer follows a pre-norm architecture with multi-head self-attention and a two-layer FFN. To integrate the geometric priors of LiDAR scanning into the backbone, we introduce a scan-aware positional encoding into the selfattention operation. The resulting latent tokens capture spatial context aligned with the LiDAR scanning pattern for subsequent occupancy-code prediction. Details of the positional encoding are provided in Sec. 3.3.

(iii) Autoregressive Probability Prediction: For each latent token, its backbone feature serves as the query, while previously decoded nodes are encoded into memory keys and values using their backbone features together with occupancy and hierarchical information. The query is refined through M = 4 cross-attention blocks, with our scan-aware positional encoding applied to both the query and memory keys. A prediction head finally outputs a 255-way occupancy distribution for entropy coding. Details of the multistage prediction scheme are provided in Sec. 3.4.

## 3.2. Point Cloud Tokenization

Transformation. We consider three coordinate representations before octree construction, following the setting in [41], as summarized in Tab. 1. Cartesian coordinates $( x , y , z )$ preserve the original lossless geometry, whereas spherical coordinates $( r , \theta , \phi )$ and cylindrical-beam coordinates $( \rho , \theta , b )$ better align with the LiDAR scanning pattern. However, their discretized scan-aligned representations do not exactly preserve the original Cartesian coordinates, resulting in coordinate precision loss and preventing strictly lossless reconstruction.

Octree Construction The transformed point cloud is organized into a hierarchical octree with a maximum depth of L, where each non-empty node is represented by an 8- bit occupancy code $a _ { i } ^ { \ell } \in \{ 1 , \ldots , 2 5 5 \}$ indicating the occupancy states of its eight children. The octree nodes are traversed level by level, while nodes within each level follow the Morton ordering induced by the parent order and child octant indices. This yields an ordered occupancy sequence $\mathcal { A } ^ { \ell } = \{ a _ { 1 } ^ { \ell } , \ldots , a _ { N _ { \ell } } ^ { \ell } \}$ at each level ℓ for tokenization.

Table 1. Coordinate representations before octree construction.
<table><tr><td>Representation</td><td>Coordinates Intrinsics Lossless</td><td></td><td></td></tr><tr><td>Cartesian</td><td> $( x , y , z )$ </td><td>X</td><td>√</td></tr><tr><td>Spherical</td><td> $( r , \theta , \phi )$ </td><td>X</td><td>X</td></tr><tr><td>Cylindrical-beam</td><td> $( \rho , \theta , b )$ </td><td>√</td><td>X</td></tr></table>

Token Embedding. The nodes at each octree level are partitioned into non-overlapping windows of at most 1,024 nodes. Each token combines three-level ancestor occupancy with the current octant and level embeddings, while the occupancy code is used only as the prediction target. Following [41], we employ graph-based positional encoding (GPE) to aggregate features from spatial neighbors. However, GPE only captures local spatial relations without explicitly modeling the LiDAR scanning prior.

## 3.3. Spherical Positional Encoding

As illustrated in Fig. 2, the rotating scanning pattern of Li-DAR is characterized in spherical coordinates. To incorporate this scanning geometry into entropy modeling, we map each token to its spherical position and construct multifrequency rotary positional encoding for attention.

Token Position Mapping. For each token in the Cartesian octree, let D denote the stored tree depth and ℓ its level, with ℓ = 1 for the root. Let $\mathbf { p } _ { i }$ denote its lower-corner grid coordinate, o the physical lower corner of the root voxel in the LiDAR-centered coordinate system, and δ the finestgrid spacing. Defining the center offset at level ℓ as

![](images/316055b0c77e18707af092ef65104150fdcdc06c5c987be03d60391d54df4915.jpg)  
Figure 2. LiDAR rotating scanning prior for spherical positional encoding. Each token position is mapped from Cartesian space to spherical coordinates $( r , \theta , \phi )$ , where r, $\theta ,$ and ϕ denote range, azimuth, and elevation, respectively, providing a scan-aligned spatial description for positional encoding.

$$
\begin{array} { r } { \Delta _ { \ell } = \left( 2 ^ { D - \ell } , 2 ^ { D - \ell } , 2 ^ { D - \ell } \right) , } \end{array}\tag{1}
$$

the physical center of token i is

$$
\begin{array} { r } { \mathbf { c } _ { i } = \mathbf { o } + \left( \mathbf { p } _ { i } + \Delta _ { \ell } \right) \odot \delta . } \end{array}\tag{2}
$$

Its spherical position is then obtained as

$$
\begin{array} { l } { r _ { i } = \Vert \mathbf { c } _ { i } \Vert _ { 2 } , } \\ { \theta _ { i } = \operatorname { a t a n 2 } ( c _ { i , y } , c _ { i , x } ) \bmod 2 \pi , } \\ { \beta _ { i } = \displaystyle \frac { \pi } { 2 } - \phi _ { i } = \operatorname { a r c c o s } \left( \frac { c _ { i , z } } { r _ { i } } \right) , } \end{array}\tag{3}
$$

where $r _ { i }$ and $\theta _ { i }$ denote the range and azimuth, while $\phi _ { i }$ is the elevation illustrated in Fig. 2 and $\beta _ { i }$ is the corresponding polar angle.

The token position is fully determined by its grid coordinate, level, and fixed root parameters before predicting the current occupancy code. SPEM therefore retains the Cartesian octree for geometry coding and uses spherical coordinates only for positional encoding, while the normalized Cartesian positions used by GPE remain unchanged. For the spherical and cylindrical-beam tree representations considered above, the corresponding $( r , \theta , \beta )$ coordinates are obtained from the stored node coordinates and quantization steps, or from calibrated mean pitch angles over the decoded beam prefixes, respectively.

Multi-Frequency Rotary Encoding. Given the spherical position $[ r , \theta , \beta ] ,$ we construct multi-frequency rotary phases for each 64-D attention head. The 32 twodimensional rotation pairs are allocated to range, azimuth, and polar angle using 7, 16, and 9 pairs, respectively. Motivated by the anisotropic sampling pattern of rotating Li-DAR, we allocate more rotation pairs to azimuth, which is typically sampled more densely than the vertical direction and thus benefits from richer angular frequencies. The corresponding phases are defined as

Table 2. Frequency allocation for spherical positional encoding.
<table><tr><td>Component</td><td>Wavelengths / harmonics</td></tr><tr><td>Range r (7)</td><td> $\lambda \left( \mathrm { m } \right) : \left\{ \ 0 . 5 \ 1 \ 2 \ 4 \ 8 \ 1 6 \ 3 2 \ \right\}$ </td></tr><tr><td>Azimuth θ (16)</td><td> $n : \left\{ \begin{array} { c c c c c c } { { 1 } } & { { 2 } } & { { 4 } } & { { 8 } } & { { 1 2 } } & { { 1 6 } } & { { 2 4 } } & { { 3 2 } } \\ { { 4 8 } } & { { 6 4 } } & { { 9 6 } } & { { 1 2 8 } } & { { 1 9 2 } } & { { 2 5 6 } } & { { 5 1 2 } } & { { 1 0 2 4 } } \end{array} \right\}$ </td></tr><tr><td>Polar β (9)</td><td>m : { 1 4 8 16 32 64 128 256 512 }</td></tr></table>

$$
\gamma _ { k } ^ { r } = \frac { 2 \pi r } { \lambda _ { k } } , \qquad \gamma _ { k } ^ { \theta } = n _ { k } \theta , \qquad \gamma _ { k } ^ { \beta } = m _ { k } \beta ,\tag{4}
$$

where $\lambda _ { k }$ denotes the physical wavelength for range, while $n _ { k }$ and $m _ { k }$ are angular harmonics for the azimuth and polar dimensions, respectively. The frequency allocation for each spherical dimension is summarized in Tab. 2.

For each two-dimensional channel pair, the corresponding query and key features are rotated according to

$$
\mathcal { R } ( \gamma ) = \left[ \begin{array} { c c } { \cos \gamma } & { - \sin \gamma } \\ { \sin \gamma } & { \cos \gamma } \end{array} \right] .\tag{5}
$$

This allows attention to capture relative spherical phase differences between tokens and thereby model scan-aligned spatial relations. In particular, the integer azimuth harmonics ensure periodic consistency across the $\theta = 0$ and $\theta = 2 \pi$ boundary. The proposed spherical RoPE is applied to the Q/K features in both backbone self-attention and decodedmemory cross-attention.

## 3.4. Autoregressive Probability Modeling

As illustrated in Fig. 3, SPEM adopts a stage-wise autoregressive scheme to balance parallel probability prediction and causal dependency modeling. Within each window, tokens are interleaved into S stages. When predicting the current stage, occupancy codes from preceding stages have already been decoded and encoded into a persistent memory bank, while tokens in later stages remain inaccessible.

Let $\mathcal { T } _ { s }$ denote the set of token indices assigned to stage s, and $\begin{array} { r } { \mathcal { I } _ { < s } = \bigcup _ { u < s } \mathcal { T } _ { u } } \end{array}$ denote those in preceding stages. The occupancy distribution of a window is factorized as

$$
p ( \mathbf { a } \mid \mathbf { F } ) = \prod _ { s = 1 } ^ { S } \prod _ { i \in \mathcal { T } _ { s } } p ( a _ { i } \mid \mathbf { F } , \mathbf { a } _ { \mathcal { T } _ { < s } } ) ,\tag{6}
$$

where F denotes the backbone features and $\mathbf { a } _ { \mathcal { T } _ { < s } }$ contains the occupancy codes decoded in preceding stages. For each token in the current stage, its backbone feature serves as the query to retrieve decoded context from the memory bank through cross-attention. Our spherical positional encoding is applied to the query and memory keys, enabling scanaligned spatial modeling during memory retrieval. Tokens within the same stage are predicted in parallel, producing a 255-way probability distribution over occupancy codes for arithmetic encoding and decoding. After the stage is decoded, its occupancy codes are appended to the memory bank and become available to the subsequent stage.

![](images/5e40258ae9b12a3161970e97eea5101fd41a0bc7bd593576bdfea642b4cbd4fb.jpg)  
Figure 3. Stage-wise autoregressive probability modeling in SPEM. Previously decoded tokens are encoded into cached K/V memory, while current-stage tokens query the accumulated memory for occupancy prediction; newly decoded tokens are then appended to the memory for subsequent stages.

Decoded Memory Construction. For each previously decoded token, we combine its backbone feature with the decoded occupancy code and its level and octant information. As shown in Fig. 3, two independent encoders transform this context into key and value representations. Each prediction layer maintains its own projected key and value cache, denoted by $\{ K ^ { ( t ) } , V ^ { ( t ) } \} _ { t = 1 } ^ { M }$ Once a token is decoded, its per-layer K/V representations are computed once and cached for all subsequent stages. Since the memory construction is independent of the stage assignment, changing the number of stages only modifies the decoding schedule without introducing stage-specific parameters, allowing our proposed SPEM to support flexible stage configurations.

## 4. Experiment

## 4.1. Experimental Settings

Dataset. We evaluate SPEM on four LiDAR point cloud datasets: SemanticKITTI [1], Ford [19], nuScenes [3], and QNX [7]. For SemanticKITTI, sequences 00–10 are used for training and 11–21 for testing, yielding 23,201 training frames and 20,351 test frames. For Ford, sequence 01 is used for training and sequences 02–03 for testing, with each sequence containing 1,500 frames. For nuScenes, we follow the data split in [41] and select 6,000 training frames and 450 test frames from non-overlapping scenes. For QNX, Motorway Bridge and Navigating Turns are used for training, while Junction Approach and Junction Exit are used for testing, yielding 1,111 training frames and 148 test frames. SemanticKITTI and nuScenes provide floating-point Cartesian coordinates in meters, whereas Ford and QNX provide geometry on a native 1-mm integer grid.

Baselines We compare SPEM with traditional codecs and representative learning-based approaches. Traditional baselines include G-PCC TMC13v23 [38], octree and predictive modes, as well as L3C2 [15]. Learning-based baselines include RENO [39], SparsePCGC [32], Unicorn [33], OctAttention [8], EHEM [24], SCP [18], TopNet [37], ASRL [36], LPCM [25], and PACE [41]. To evaluate different coordinate representations, we reproduce and retrain PACE with Cartesian, spherical, and cylindrical-beam inputs on selected datasets, achieving performance comparable to or better than the original reported results. For the other baselines, we use the results reported in their original papers.

Metric We evaluate rate-distortion performance using bits per point (bpp) for bitrate and point-to-point (D1) and point-to-plane (D2) PSNR for geometry distortion. The PSNR peak values are set to 59.7 m for SemanticKITTI and nuScenes and 30,000 mm for Ford and QNX. We report R-D results under Cartesian, spherical, and cylindricalbeam representations to evaluate the influence of coordinate transformation. We further use BD-BR [2] to measure the average bitrate difference at equivalent geometric quality, where lower values indicate better compression efficiency. For Ford and QNX, we also evaluate lossless compression performance under Cartesian coding.

Implementation settings We train a separate SPEM mode for each dataset and coordinate representation. SPEM operates on 1,024-token windows with a 256-D feature dimension and four attention heads, using a 12-layer backbone and a 4-layer decoded-memory predictor with 1,024- D FFNs. During training, the autoregressive stage count is sampled from $S \in \{ 1 , 2 , 4 , 8 \}$ . We use $S = 8$ for default R-D evaluation to favor compression performance, while S = 1 prioritizes inference speed. SPEM is optimized with cross-entropy loss on occupancy prediction using AdamW on NVIDIA A100 GPUs, with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ decayed by 0.8 approximately every 30,000 steps. SemanticKITTI, Ford, nuScenes, and QNX are trained for 5, 50, 25, and 20 epochs, respectively.

## 4.2. Performance Evaluation

Fig. 4 compares the R-D performance under scan-aligned coordinate transformations, where spherical coordinates are used for SemanticKITTI and nuScenes, while cylindricalbeam coordinates are adopted for Ford and QNX. SPEM consistently achieves better R-D performance than the compared methods for both D1 and D2. As quantified in Tab. 3, SPEM achieves BD-rate savings of 5.30%–14.19% for D1 and 5.06%–14.62% for D2 relative to PACE-AR, with average savings of 8.37% and 8.33%, respectively. These gains show that scan-aware positional modeling in attention and efficient stage-wise autoregressive prediction remain complementary to explicit scan-aligned coordinate transformation.

![](images/44a85819b425f06c76624cdba7a158e4f7120a4fd57cada8fdde0711502f911a.jpg)  
Figure 4. R-D comparison on SemanticKITTI, Ford, nuScenes, and QNX under coordinate transformation, using spherical coordinates for SemanticKITTI and nuScenes and cylindrical-beam coordinates for Ford and QNX.

![](images/a82006e498e0a070d74139b443062637c8c5a566a733b27567b864f3d88960cc.jpg)  
Figure 5. R-D comparison on SemanticKITTI, Ford, nuScenes, and QNX without coordinate transformation, using Cartesian coordinates; the ∞ markers for Ford and QNX indicate lossless compression at native millimeter precision.

The advantage becomes substantially larger when operating directly on the original Cartesian representation, as shown in Fig. 5. SPEM achieves D1 BD-rate savings of 14.53%, 24.01%, 14.28%, and 36.76% on SemanticKITTI, Ford, nuScenes, and QNX, respectively, with closely matched improvements in D2. Averaged over the four datasets, the gains increase to 22.40% in D1 and

Table 3. Comparison with PACE-AR under transformed and Cartesian representations. BD-rate is measured relative to PACE-AR, and encoding/decoding time is reported at the highest-rate operating point on each R-D curve.
<table><tr><td>Dataset</td><td colspan="4">Transformed</td><td colspan="4">Cartesian</td><td colspan="3">Lossless</td></tr><tr><td></td><td>BD-rate D1</td><td>BD-rate D2</td><td>Time PACE-AR</td><td>Time SPEM-S8</td><td>BD-rate D1</td><td>BD-rate D2</td><td>Time PACE-AR</td><td>Time SPEM-S8</td><td>PACE-AR (bpp)</td><td>SPEM-S8 (bpp)</td><td>Gain</td></tr><tr><td>SemanticKITTI</td><td>-5.30%</td><td>-5.06%</td><td>12.82/18.15</td><td>2.19/2.25</td><td>-14.53%</td><td>-14.52%</td><td>10.74/15.41</td><td>2.02/2.07</td><td></td><td></td><td></td></tr><tr><td>Ford</td><td>-6.77%</td><td>-6.53%</td><td>17.05/23.91</td><td>2.60/2.59</td><td>-24.01%</td><td>-24.00%</td><td>16.00/22.45</td><td>2.63/2.56</td><td>13.643</td><td>8.576</td><td>37.14%</td></tr><tr><td>nuScenes</td><td>-7.21%</td><td>-7.12%</td><td>8.84/12.49</td><td>1.09/1.22</td><td>-14.28%</td><td>-14.23%</td><td>7.71/10.84</td><td>1.05/1.14</td><td></td><td></td><td></td></tr><tr><td>QNX</td><td>-14.19%</td><td>-14.62%</td><td>11.32/15.78</td><td>1.16/1.28</td><td>-36.76%</td><td>-36.78%</td><td>10.25/14.32</td><td>1.21/1.30</td><td>10.998</td><td>5.920</td><td>46.17%</td></tr></table>

Table 4. Rate–complexity trade-off on Ford under lossless coding.
<table><tr><td>Method</td><td>Stage</td><td>Enc. (s)</td><td>Dec. (s)</td><td>Rate (bpp)</td></tr><tr><td rowspan="5">PACE (10.64M)</td><td> $S = 1$ </td><td>1.49</td><td>1.63</td><td>14.576</td></tr><tr><td> $S = 2$ </td><td>1.56</td><td>1.50</td><td>14.219</td></tr><tr><td> $S = 4$ </td><td>1.66</td><td>1.53</td><td>14.013</td></tr><tr><td> $S = 8$ </td><td>1.84</td><td>1.74</td><td>13.890</td></tr><tr><td> $S = 1 6$  AR</td><td>2.46 16.00</td><td>2.48</td><td>13.815</td></tr><tr><td rowspan="5">SPEM (13.88M)</td><td> $S = 1$ </td><td>1.82</td><td>22.45</td><td>13.643 9.214</td></tr><tr><td> $S = 2$ </td><td>1.98</td><td>1.92 1.91</td><td>8.820</td></tr><tr><td> $S = 4$ </td><td>2.18</td><td>2.09</td><td>8.657</td></tr><tr><td> $S = 8$ </td><td>2.63</td><td>2.56</td><td>8.576</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

22.38% in D2, compared with about 8.4% under coordinate transformation. The consistently larger improvement in Cartesian space indicates that SPEM can capture the Li-DAR scanning structure directly during entropy modeling, reducing the dependence on explicit scan-aligned coordinate transformations while preserving the original geometry representation.

The Cartesian setting further enables lossless evaluation on Ford and QNX at their native millimeter precision. As reported in Tab. 3, SPEM reduces the lossless bitrate from 13.643 to 8.576 bpp on Ford and from 10.998 to 5.920 bpp on QNX, corresponding to savings of 37.14% and 46.17%, respectively, over PACE-AR. This highlights a key advantage of SPEM: it exploits LiDAR scanning regularities within the entropy model while preserving the original Cartesian geometry, achieving substantially more compact lossless coding without coordinate transformation.

## 4.3. Complexity Analysis

Tab. 4 compares the rate–complexity trade-off between PACE and SPEM under different autoregressive stage counts. $\mathrm { A t } \ : S = 1$ , both methods perform fully parallel prediction. SPEM is slightly slower than PACE because no real decoded memory is available, yet the decoded-memory predictor still executes all four cross-attention blocks, where each layer attends to the learned null K/V and is followed by an FFN with residual updates.

![](images/f5d21faf96970424183d55685f9398875722fe2cc7dedce3da323881f627832d.jpg)

![](images/853f8da82cc82f77fd3aa360cb7538874cf8a9a07ed44784275c2e14eeb4eb4e.jpg)  
Figure 6. R-D comparison of different SPEM variants on Ford dataset.

As the stage count increases, however, SPEM runtime grows only moderately while the bitrate consistently decreases. This favorable scaling is enabled by our memorybased autoregressive design: the K/V representations of decoded tokens are constructed once and cached for reuse in all subsequent stages. Therefore, increasing S mainly changes the decoding schedule and progressively exposes more decoded context, without repeatedly recomputing historical representations. For example, increasing S from 1 to 8 substantially improves the lossless rate while introducing only a modest increase in encoding and decoding time. This allows SPEM to flexibly trade inference speed for compression efficiency using a single model.

## 4.4. Qualitative Results

Fig. 7 visualizes reconstructions on Ford and QNX. Compared with G-PCC predictive geometry, the learned methods produce noticeably smaller geometric errors, while SPEM further improves coding efficiency over PACE under the transformed representation. Interestingly, for the transformed methods, reconstruction errors in distant regions can be comparable to or even smaller than those in nearby regions. This behavior results from the region-adaptive coding strategy inherited from [18], which assigns more octree levels to distant regions to compensate for their lower spatial sampling density. More importantly, Cartesian SPEM exactly preserves the original millimeter-precision geometry at the lossless operating point. These results show that SPEM can exploit LiDAR scanning regularities while supporting both lossy and lossless geometry compression.

![](images/f3857a7841f081887adb0abfcfa722b521493073a1fe833f4ff45f368dcbea6b.jpg)  
Figure 7. Qualitative comparison on Ford and QNX. From left to right, we show G-PCC predictive geometry, transformed PACE, trans formed SPEM, and lossless Cartesian SPEM. Points are colored by reconstruction error in millimeters.

![](images/bdadf8a78e8f94e2a81e86c67a1ecc736bd0b608ac4a870d42cce8dcdd47acbb.jpg)

![](images/00a9a5698c92c42a4b089cbbb32e8ae79fdf233461d604c21299cef76ea64ab0.jpg)  
Figure 8. R-D comparison of SPEM with and without coordinate transformation on SemanticKITTI and nuScenes.

## 4.5. Ablation Study

Spherical positional encoding. The left column of Fig. 6 evaluates the contribution of our spherical positional encoding. Removing spherical RoPE causes a clear degradation. Replacing it with Cartesian XYZ-RoPE partially recovers the performance but remains notably inferior to SPEM, indicating that the gain does not simply arise from rotary positional encoding itself, but from aligning positional relations with the LiDAR scanning geometry. Among the three spherical components, the relatively small degradation from removing range suggests that the two angular components already capture the dominant LiDAR scanning pattern. In addition, removing the azimuth phase causes a larger performance drop than removing the polar phase, which is consistent with the denser horizontal angular sampling of rotating LiDAR and motivates allocating more frequency capacity to the azimuth dimension.

Autoregressive predictor. The right column of Fig. 6 evaluates the decoded-memory prediction design. Replacing our cross-attention predictor with the PACE-style causal predictor consistently degrades the R-D performance, including the lossless compression. This confirms the benefit of querying cached decoded context through crossattention, which is further enhanced by our spherical RoPE to capture scan-aware spatial relations during memory retrieval.

Effect of coordinate transformation. We investigate the necessity of explicit coordinate transformation by comparing SPEM under Cartesian and spherical representations, as shown in Fig. 8. For a fair comparison, both configurations employ the same spherical positional encoding without using LiDAR intrinsic parameters. Their comparable R-D performance suggests that scan-aware entropy modeling effectively captures LiDAR scanning regularities without transforming the coded geometry. This indicates that the benefits of scan-aligned coordinates can largely be realized through probability modeling rather than changes to geometry representation. By decoupling acquisition priors from geometry representation, SPEM reduces reliance on explicit transformation while retaining Cartesian coding. This offers a promising direction toward unified lossy and lossless geometry compression.

## 5. Conclusion

We present SPEM, a scan-prior entropy modeling framework for LiDAR point cloud geometry compression. Instead of relying solely on explicit coordinate transformation to expose the LiDAR acquisition structure, SPEM incorporates the scanning prior directly into entropy modeling through positional encoding and stage-wise autoregressive prediction. The proposed spherical RoPE enables attention to model scan-aligned spatial relations, while the decodedmemory predictor efficiently exploits previously decoded occupancy information through reusable K/V caches. Experiments on multiple datasets demonstrate that SPEM achieves state-of-the-art performance for both lossy and lossless compression with favorable coding efficiency.

## LLM Usage Statement

In preparing this manuscript, we used ChatGPT and Codex to assist with academic writing and code development, including language refinement, manuscript organization, code implementation, and debugging. All core scientific ideas, methodological designs, experimental decisions, and interpretations of results were developed by the authors. All AIassisted content and code were reviewed and validated by the authors, who take full responsibility for this work.

## References

[1] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall. SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences. In Proc. of the IEEE/CVF International Conf. on Computer Vision (ICCV), 2019. 1, 5

[2] Gisle Bjøntegaard. Calculation of average PSNR differences between RD-curves. Technical Report VCEG-M33, ITU-T SG 16/Q6, 2001. 13th VCEG Meeting. 5

[3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 11618–11628. IEEE, 2020. 1, 5

[4] Zhili Chen, Zian Qian, Sukai Wang, and Qifeng Chen. Point cloud compression with sibling context and surface priors. In European Conference on Computer Vision, pages 744–759. Springer, 2022. 2

[5] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 3070–3079. IEEE, 2019. 2

[6] Tingyu Fan, Linyao Gao, Yiling Xu, Dong Wang, and Zhu Li. Multiscale latent-guided entropy model for lidar point cloud compression. IEEE Transactions on Circuits and Systems for Video Technology, 33(12):7857–7869, 2023. 2

[7] David Flynn and Sebastien Lasserre. Pcc cat3 test sequences´ from blackberry–qnx. Technical Report m43647, ISO/IEC JTC 1/SC 29/WG 11, 2018. 5

[8] Chunyang Fu, Ge Li, Rui Song, Wei Gao, and Shan Liu. Octattention: Octree-based large-scale contexts model for point cloud compression. In Proceedings of the AAAI conference on artificial intelligence, pages 625–633, 2022. 1, 2, 5

[9] Wei Gao, Liang Xie, Songlin Fan, Ge Li, Shan Liu, and Wen Gao. Deep learning-based point cloud compression: An in-

depth survey and benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1

[10] Danillo Graziosi, Ohji Nakagami, Satoru Kuma, Alexandre Zaghetto, Teruhiko Suzuki, and Ali Tabatabai. An overview of ongoing point cloud compression standardization activities: Video-based (v-pcc) and geometry-based (g-pcc). AP-SIPA Transactions on Signal and Information Processing, 9: e13, 2020. 1

[11] Lila Huang, Shenlong Wang, Kelvin Wong, Jerry Liu, and Raquel Urtasun. Octsqueeze: Octree-structured entropy model for lidar compression. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1310–1320. IEEE, 2020. 1, 2

[12] Wenjie Huang, Qi Yang, Shuting Xia, He Huang, Yiling Xu, and Zhu Li. Linr-pcgc: Lossless implicit neural representations for point cloud geometry compression. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 28577–28586. IEEE, 2025. 2

[13] Yiqi Jin, Ziyu Zhu, Tongda Xu, Yuhuan Lin, and Yan Wang. Ecm-opcc: Efficient context model for octree-based point cloud compression. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Process ing (ICASSP), pages 7985–7989. IEEE, 2024. 2

[14] Junsik Kim, Gun Bang, and Soowoong Kim. Elic: Ef ficient lidar geometry compression via cross-bit-depth feature propagation and bag-of-encoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 39011–39020, 2026. 2

[15] S. Lasserre and J. Taquet. A point cloud codec for LiDAR data with very low complexity and latency. Technical Report m56477, ISO/IEC JTC 1/SC 29/WG 7, 2021. 1, 5

[16] Bojun Liu, Yangzhi Ma, Li Li, Dong Liu, Zhu Li, and Houqiang Li. Next bit prediction: A unified lossless and lossy point cloud geometry compression framework. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026. 2

[17] Bojun Liu, Yangzhi Ma, Ao Luo, Li Li, and Dong Liu. Voxel-based point cloud geometry compression with spaceto-channel context. IEEE Transactions on Image Processing, 2026. 2

[18] Ao Luo, Linxin Song, Keisuke Nonaka, Kyohei Unno, Hem ing Sun, Masayuki Goto, and Jiro Katto. Scp: Spherical coordinate-based learned point cloud compression. In Pro ceedings of the AAAI Conference on Artificial Intelligence, pages 3954–3962, 2024. 1, 2, 5, 7

[19] Gaurav Pandey, James R McBride, and Ryan M Eustice. Ford campus vision and lidar data set. The International Journal ofRobotics Research, 30(13):1543–1552, 2011. 5

[20] Maurice Quach, Giuseppe Valenzise, and Frederic Dufaux. Learning convolutional transforms for lossy point cloud geometry compression. In 2019 IEEE international conference on image processing (ICIP), pages 4320–4324. IEEE, 2019. 2

[21] Zizheng Que, Guo Lu, and Dong Xu. Voxelcontext-net: An octree based framework for point cloud compression. In 2021 IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 6038–6047. IEEE, 2021. 2

[22] Ricardo Roriz, Jorge Cabral, and Tiago Gomes. Automotive lidar technology: A survey. IEEE Transactions on Intelligent Transportation Systems, 23(7):6282–6297, 2021. 1

[23] Sebastian Schwarz, Marius Preda, Vittorio Baroncini, Madhukar Budagavi, Pablo Cesar, Philip A Chou, Robert A Cohen, Maja Krivokuca, S´ ebastien Lasserre, Zhu Li, et al.´ Emerging mpeg standards for point cloud compression. IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 9(1):133–148, 2018. 1

[24] Rui Song, Chunyang Fu, Shan Liu, and Ge Li. Efficient hierarchical entropy model for learned point cloud compression. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14368–14377. IEEE, 2023. 2, 5

[25] Chang Sun, Hui Yuan, Shiqi Jiang, Da Ai, Wei Zhang, and Raouf Hamzaoui. Lpcm: Learning-based predictive coding for lidar point cloud compression. IEEE Transactions on Image Processing, 2026. 1, 5

[26] Chang Sun, Hui Yuan, Shiqi Jiang, Chongzhen Tian, Guanghui Zhang, and Raouf Hamzaoui. Inter-lpcm: Learning-based inter-frame predictive coding for lidar point cloud compression. IEEE Transactions on Image Processing, 2026. 1

[27] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2443–2451. IEEE, 2020. 1

[28] Haotian Tang, Zhijian Liu, Xiuyu Li, Yujun Lin, and Song Han. Torchsparse: Efficient point cloud inference engine. Proceedings of Machine Learning and Systems, 4:302–315, 2022. 2

[29] Haotian Tang, Shang Yang, Zhijian Liu, Ke Hong, Zhongming Yu, Xiuyu Li, Guohao Dai, Yu Wang, and Song Han. Torchsparse++: Efficient training and inference framework for sparse convolution on gpus. In Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, pages 225–239, 2023. 2

[30] Maurice Quach, Giuseppe Valenzise, and Frederic Dufaux. Improved deep point cloud geometry compression. In 2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP), pages 1–6. IEEE, 2020. 2

[31] Jianqiang Wang, Dandan Ding, Zhu Li, and Zhan Ma. Multiscale point cloud geometry compression. In 2021 Data Compression Conference (DCC), pages 73–82. IEEE, 2021. 1, 2

[32] Jianqiang Wang, Dandan Ding, Zhu Li, Xiaoxing Feng, Chuntong Cao, and Zhan Ma. Sparse tensor-based multiscale representation for point cloud geometry compression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):9055–9071, 2022. 1, 2, 5

[33] Jianqiang Wang, Ruixiang Xue, Jiaxin Li, Dandan Ding, Yi Lin, and Zhan Ma. A versatile point cloud compressor using universal multiscale conditional coding–part i: Geometry. IEEE transactions on pattern analysis and machine intelligence, 47(1):269–287, 2024. 2, 5

[34] Kangli Wang and Wei Gao. Unipcgc: towards practical point cloud geometry compression via an efficient unified ap-

proach. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12721–12729, 2025. 2

[35] Shengqian Wang, Chang Tu, and He Chen. D-compress: Detail-preserving lidar range image compression for real time streaming on resource-constrained robots. arXiv preprint arXiv:2603.13699, 2026. 1

[36] Xinjie Wang, Ke Xu, Bin Deng, Yulan Guo, and Hanyun Wang. Asrl: Adaptive sparse representation learning for lidar point cloud geometry compression. IEEE Signal Processing Letters, 2025. 5

[37] Xinjie Wang, Yifan Zhang, Ting Liu, Xinpu Liu, Ke Xu, Jianwei Wan, Yulan Guo, and Hanyun Wang. Topnet: Transformer-efficient occupancy prediction network fo octree-structured point cloud geometry compression. In 2025 IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 27305–27314. IEEE, 2025. 2, 5

[38] WG7, MPEG 3D Graphics and Haptics Coding. G-PCC 2nd edition codec description. Technical Report N00575, ISO/IEC JTC 1/SC 29/WG 7, Antalya, Turkey, 2023. 142nd MPEG Meeting. 2, 5

[39] Kang You, Tong Chen, Dandan Ding, M Salman Asif, and Zhan Ma. Reno: Real-time neural compression for 3d lidar point clouds. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22172– 22181. IEEE, 2025. 2, 5

[40] Kang You, Tong Chen, Dandan Ding, M Salman Asif, and Zhan Ma. Bitric: Efficient neural compression of lidar range images via hierarchical bitplanes. In European Conference on Computer Vision, pages 549–566. Springer, 2026. 1

[41] Jiahao Zhu, Kang You, Dandan Ding, and Zhan Ma. Pace: Post-causal entropy modeling for learned lidar point cloud compression. arXiv preprint arXiv:2605.01320, 2026. 1, 2, 3, 5