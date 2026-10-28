# Multi-task Disentangling Fusion Mamba-CNN for MODIS Spectral-Temporal Unmixing with Endmember Variability Modeling

Motasem Alkayid, Yimin Zhu, Naser El-Sheimy, Member, IEEE, Lincoln Linlin Xu, Member, IEEE

Abstract—Spectral-temporal unmixing of MODIS time series is essential for applications such as abundance mapping and spectral library construction, yet it remains challenging due to (i) spectral-temporal information coupling in MODIS data, (ii) intra-class endmember spectral variability and inter-class signature ambiguity, and (iii) the instability of naively fusing multiple expert representations. This paper presents the Multitask Disentangling Fusion Mamba (MDF-Mamba) to address these limitations. To resolve spectral-temporal coupling, we design a disentangling architecture that pairs per-channel spectral experts with a dedicated bidirectional state-space (Mamba) temporal branch. Convolutional modules each process a single spectral channel, while the Mamba branch encodes the cross-channel seasonal trajectory, forming a multi-branch expert ensemble with soft inter-expert reweighting based on squeeze-and-excitation. For the integration of these spectral and temporal expert outputs, we introduce mHCFusion, adapting the manifold-constrained hyper-connection framework to merge the outputs in a stable and interpretable manner. Endmember variability is addressed by extending the linear mixing model with pixel-wise learnable spectral variability weights that capture intra-class endmember variation. Finally, we design a multi-task training objective that combines supervised abundance estimation with self-supervised spectral reconstruction and endmember regularization terms on the same training data, yielding a highly efficient framework for spectral-temporal unmixing. Experiments on MODIS time series covering the Alberta and Saskatchewan provinces show that MDF-Mamba achieves the best performance on 12 of 14 evaluation metrics across both datasets (7 of 7 on Alberta; 5 of 7 on Saskatchewan), ranking second on the remaining two.

Index Terms—Spectral unmixing, MODIS, state space model, Mamba, endmember variability, multi-task learning, abundance estimation, mixture of experts.

## I. INTRODUCTION

E <sup>STIMATING</sup> <sup>sub-pixel</sup> <sup>land</sup> <sup>cover</sup> <sup>fractions</sup> <sup>from</sup> <sup>mod-</sup> erate resolution satellite data is essential for broad-scale environmental observation. The Moderate Resolution Imaging Spectroradiometer (MODIS) provides daily global coverage at 250 - 1000 m spatial resolution, producing dense spectraltemporal stacks that capture phenological dynamics across diverse biomes [1, 2]. However, at this spatial resolution, most pixels contain mixtures of land cover types, necessitating spectral unmixing to estimate per-pixel fractional abundances. Accurate unmixing of MODIS time series is essential for applications such as crop monitoring, forest disturbance mapping, and land cover change detection across the heterogeneous landscapes of western Canada.

Traditional linear spectral unmixing (LSU) methods decompose each pixel spectrum as a linear combination of endmember signatures. On the other hand, fully Constrained Least Squares Unmixing (FCLSU) [3] solves this analytically via constrained least squares optimization [4], but assumes fixed endmembers across the scene. Extensions such as sparse unmixing via variable splitting and augmented lagrangian (SUnSAL) [5] promote sparsity through ℓ<sub>1</sub> regularization, while nonnegative matrix factorization (NMF) variants such as Spatial Group Sparsity NMF (SGSNMF) [6] incorporate spatial group sparsity via superpixel-guided regularization.

While physically interpretable and computationally efficient, these approaches suffer from two fundamental limitations when applied to dense MODIS spectral-temporal stacks. Primarily, they process all spectral-temporal bands through a single shared model, conflating the physically distinct phenological cues of each spectral channel and preventing the disentanglement of channel-specific temporal dynamics. Additionally, they assume fixed endmember signatures across the entire scene, which is unrealistic given the substantial intra-class spectral variability driven by soil moisture, topography, and seasonal phenology [7, 8].

Deep learning approaches have emerged as viable alternatives for spectral unmixing tasks, offering the ability to learn nonlinear representations directly from data. Autoencoder-based methods such as PGMSU [9] and SUnCNN [10] learn latent abundance representations through unsupervised reconstruction. Transformer-based architectures including DeepTrans-HSU [11] and SRViT [12] use self-attention to capture longrange spectral dependencies. Spatial-spectral attention networks such as SSAF-Net [13] and PPM-Net [14] jointly model spatial and spectral features through dual-stream encoders. More recently, 2024 frameworks have introduced contrastive learning [15], continuous variational inference [16], and spectral variability-constrained diffusion models [17] to improve resilience against environmental noise and label scarcity. Despite these advances, deep learning methods largely inherit the same limitations: they process the full L-band stack through shared parameters, causing cross-channel interference; most retain fixed or weakly regularized endmembers that cannot capture intra-class spectral variability across the growing season; and when multi-stream features are aggregated, concatenation or pooling is used with no geometric guarantee on the stability or interpretability of the fusion.

The emergence of selective state space models (SSMs), particularly the Mamba architecture [18], offers a clear alternative for modeling long sequences with linear-time complexity $O ( T )$ in sequence length. This makes the SSM well suited to the temporal axis of MODIS time series, where land cover discrimination relies on the ordering of phenological transitions (green-up, peak, senescence) rather than any single date spectrum [19]. While 2024 has seen Mamba adapted for spatial-spectral hyperspectral classification [20, 21], its application to unmixing remains underexplored. Recent work has applied Mamba to hyperspectral unmixing [22], but scans the SSM over the spectral (band) dimension rather than time and does not address endmember variability. While Mamba has been applied to the temporal axis of satellite image time series for crop classification [23, 24], a dedicated temporal Mamba branch for phenology-driven unmixing remains unexplored. More broadly, no existing method (traditional, deep learning, or SSM-based) jointly resolves all three limitations under a unified physics-consistent framework. Abundance supervision alone is insufficient to uniquely constrain the endmember geometry [7]: without a complementary spectral reconstruction objective, learned endmember embeddings can satisfy the abundance loss while converging to physically meaningless signatures. These open problems directly motivate the contributions of MDF-Mamba.

In this paper, we present the Multi-task Disentangling Fusion Mamba (MDF-Mamba) model, a novel architecture specifically designed for MODIS spectral-temporal unmixing. The key contributions of this work are summarized as follows:

1) We design an asymmetric disentangling architecture via a hybrid CNN-Mamba multi-branch expert ensemble that assigns spectral-spatial encoding to fixed, channelspecialized convolutional experts and temporal sequence encoding to a dedicated Mamba branch, using complementary inductive biases. Unlike canonical mixture-ofexperts (MoE), expert assignment is fixed by channel rather than learned via a gating network, with only the downstream combination weights learned. A crossband input stem first learns inter-channel correlations before strided indexing splits the input into six perchannel spectral streams. Six spectral BandBranches independently process a single spectral channel each (NDVI, EVI, Red, NIR, Blue, MIR) using a dual-path 2D CNN (ElementBranchCNN, d=128) that treats the $T = 2 3$ temporal observations as input channels. Each branch runs two parallel depth-three sub-networks, a spectral path (1×1 convolutions) and a spatial path (3×3 convolutions), cross-refined via a sigmoid-gated twist gate. A seventh TemporalMambaBranch processes all 23 composites jointly as a sequence of 6-dimensional multi-spectral tokens via a bidirectional selective state space model (SSM). After cross-modal alignment, a cross-branch Squeeze-and-Excitation (SE) module enables inter-expert communication by computing soft attention weights across the six aligned spectral streams. To couple spectral and temporal representations, each spectral stream is modulated by the temporal output via an adaptive spatial-channel phenology gate that uses depthwise $3 \times 3$ convolution and sigmoid gating to produce spatially varying injection maps, replacing scalar gates with content-dependent modulation. The six modulated streams are then hierarchically grouped by physical domain: vegetation indices (NDVI + EVI), optical reflectance (Red + NIR + Blue), and moisture (MIR), reducing the fusion input from six streams to three semantically coherent groups.

2) To integrate the three hierarchically grouped expert streams, we employ a manifold-constrained hyperconnection fusion module (mHCFusion) that operates in a $d _ { \mathrm { f u s e } } { = } 6 4$ representation space via per-stream alignment projections, applies doubly stochastic Sinkhorn-projected mixing across the grouped streams, and produces a unified feature map. A scene-adaptive Feature-wise Linear Modulation (FiLM) classification head then conditions the fused representation on global scene context. An adaptive average pooling path generates per-channel scale and shift parameters that modulate the fused features before final class projection, allowing the model to adaptively adjust abundance boundary sharpness based on overall scene composition.

3) We address endmember variability through an extended linear mixing model (ELMM) in which each of the P land cover classes maintains R learnable spectral variants. Initialization follows a two-tier purity protocol: variants are seeded from high-purity training pixels (dominant class fraction $\ge ~ 0 . 8 0 )$ ; for biomes where no such pixels exist, a geographic grid fallback (minimum purity $\geq 0 . 3 0 )$ provides spatially representative initialization. A lightweight pixel-wise variability head produces per-pixel convex mixing coefficients over the R variants. To capture non-linear mixing effects that the base ELMM cannot represent, a cross-attention reconstruction module allows each pixel to attend to the full P·R endmember variant bank to produce an attention-based spectrum, which is blended with the ELMM reconstruction via a learnable sigmoid-gated mixing coefficient. A Laplacian spectral smoothness prior further regularizes the endmember curves to enforce physical realism.

4) We design a multi-task supervised-unsupervised learning framework with curriculum training that jointly optimizes supervised abundance estimation, unsupervised spectral reconstruction, per-branch regularization, spectral correlation preservation, and endmember smoothness. A dedicated curriculum learning strategy up-weights reconstruction losses during early training to stabilize the endmember dictionary before shifting optimization power toward abundance refinement, combined with cosine-annealed endmember anchoring that progressively releases the model from its initialization.

Extensive experiments on MODIS time series datasets covering Alberta and Saskatchewan demonstrate that MDF-

Mamba achieves the best performance on 12 of 14 evaluation metrics across abundance estimation (MAE, RMSE), spectral re construction (MSE, SAD), and classification accuracy (OA, AA, Kappa), ranking second on the remaining two. This outperforms eleven established baselines, including three traditional methods (FCLSU, SGSNMF, SUnSAL) and eight deep learning methods (DeepTrans-HSU, PGMSU, PPM-Net, SSAF-Net, SUnCNN, MiSiCNet, SRViT, and UNMamba). The remainder of this paper is organized as follows. Section II reviews related work on spectral unmixing, endmember variability modeling, MoE fusion, and state space models. Section III presents the proposed MDF-Mamba architecture in detail. Section IV describes the experimental setup, datasets, and quantitative results. Section V concludes the paper.

## II. RELATED WORK

## A. Linear and Deep Learning-Based Spectral Unmixing

Classical spectral unmixing formulates the observed pixel spectrum $\mathbf { y } \in \mathbb { R } ^ { L }$ as a linear combination of P endmember spectra collected in the matrix $\mathbf { M } \in \mathbb { R } ^ { L \times P }$ , weighted by their corresponding fractional abundances $\mathbf { a } \in \mathbb { R } ^ { P }$

$$
{ \mathbf { y } } = { \mathbf { M } } { \mathbf { a } } + { \mathbf { n } } , \quad { \mathrm { s . t . } } \quad { \mathbf { a } } \succeq { \mathbf { 0 } } , \quad { { \mathbf { 1 } } ^ { \top } } { \mathbf { a } } = 1\tag{1}
$$

where n denotes the additive noise. While traditional methods like FCLSU [3] solve (1) analytically, they are limited by the assumption that endmembers remain fixed across the entire scene. To overcome this and accommodate large spectral libraries, sparse unmixing methods such as SUnSAL [5] reformulate (1) as an ℓ<sub>1</sub>-regularized optimization problem. Solved via the alternating direction method of multipliers (ADMM), SUnSAL explicitly promotes sparsity in the fractional abundances.

Furthermore, nonnegative matrix factorization (NMF)-based approaches typically estimate endmembers and abundances jointly. Notably, SGSNMF [6] extends standard NMF by incorporating a spatial group sparsity regularizer that leverages superpixel segmentation to enforce local homogeneity, yielding improved spatial coherence in the abundance maps. In our baseline implementation (Section IV-E), however, the endmembers are held fixed after Vertex Component Analysis (VCA) initialization to avoid model instability on near-collinear MODIS signatures.

Recent deep learning methods have shifted toward learning the unmixing function end-to-end. Early approaches employed autoencoder architectures, such as PGMSU [9], which constrain the latent space to represent abundance maps while using the decoder as a learnable endmember matrix. To better exploit spatial correlations, convolutional approaches like SUnCNN [10] and MiSiCNet [25] introduced architectures inspired by deep image priors for blind unmixing. However, because standard convolutions struggle to capture global context, transformer-based methods such as DeepTrans-HSU [11] and SRViT [12] employ self-attention to capture long-range spectral dependencies. Building on this, recent dual-stream architectures like SSAF-Net [13] and PPM-Net [14] employ spatial-spectral attention networks to explicitly model the interplay between local spatial context and global spectral features.

Methodological evolution in deep unmixing has increasingly focused on mitigating the reliance on labeled ground truth. Approaches now include contrastive self-supervised learning [15], which learns robust representations by maximizing similarity between augmented views of the same pixel, and spectral variability-constrained diffusion models [17], which iteratively generate robust abundance priors. However, while these generative and self-supervised advances help address data scarcity, many existing architectures are not explicitly designed to disentangle the complex spectral-temporal information structure inherent in broadband satellite time series. Furthermore, they often fail to exploit phenological dynamics related to spectral channels, leaving a critical gap that motivates the first contribution of this work.

## B. Endmember Variability Modeling

The assumption of fixed endmembers fails at regional-scale studies, where intra-class spectral variation is heavily driven by soil moisture, topography, and seasonal phenology [7, 8]. Early methods addressed this variability using straightforward physical constraints. For instance, ELMM [26] allows for pixelwise scaling of endmember spectra to account for illumination changes, while bundle-based approaches maintain extensive libraries of spectral variants for each class. More recently, the focus has shifted toward complex, data-driven modeling. Concurrent preprints such as λSplit [27] employ physicsinformed deep networks to learn conditional distributions over concentrations, and BLUTH [28] utilizes hierarchical unmixing to reveal endmembers across varying levels of spectral contrast. Pushing this further, cutting-edge 2024 approaches have introduced continuous variational inference frameworks [16] and spectral variability-constrained diffusion models [17] to probabilistically handle environmental noise. However, while these generative approaches are powerful, they are highly computationally intensive. Crucially, they do not jointly couple endmember variability with multi-task sequence reconstruction under a single, unified deterministic loss.

## C. Mixture of Experts for Remote Sensing

While standard Mixture of Experts (MoE) architectures rely on learned gating functions to dynamically route input tokens to specialized sub-networks, MDF-Mamba employs a different approach. As detailed in Section III-A, our architecture utilizes a fixed, channel-determined assignment paired with learned soft reweighting. Because it lacks the dynamic routing characteristic of strict MoE frameworks, we formally define MDF-Mamba’s design as a multi-branch expert ensemble.

Recent frameworks have increasingly relied on conditional routing to manage complex data structures. For example, UniRoute [29] applies conditional routing to disentangle local details from global context. In remote sensing, MixtureRS [30] applies sparse MoE layers to handle confusing land cover categories by assigning specialized experts to specific spectralspatial patterns. Similarly, MaMOL [31] introduces a modalityaware MoE to manage missing or coupled data.

However, stably fusing the outputs of these multi-stream architectures remains a challenge. To address this, the manifoldconstrained hyper-connection (mHC) framework [32, 33] provides a principled aggregation method. It projects connection weight matrices onto the Birkhoff polytope of doubly-stochastic matrices using the Sinkhorn-Knopp algorithm. This constraint ensures equal total signal throughput across all streams and preserves the identity-mapping property, which is essential for stable training. Adapting this robust mHC framework to fit the asymmetric spectral-temporal structure of MDF-Mamba is the focus of Contribution 2.

D. State Space Models and Mamba for Remote Sensing Temporal Sequences

State Space Models (SSMs), particularly the Mamba architecture, enable linear-time $\mathcal { O } ( T )$ sequence modeling through an input-dependent selection mechanism [18]. Unlike Transformers that rely on quadratic self-attention $( \mathcal { O } ( T ^ { 2 } ) )$ ), Mamba recurrently updates its hidden state via:

$$
\mathbf { h } _ { t } = e ^ { \mathbf { A } \odot \Delta _ { t } } \mathbf { h } _ { t - 1 } + \Delta _ { t } \mathbf { B } _ { t } \mathbf { u } _ { t }\tag{2}
$$

where the input-dependent discretization parameter $\Delta _ { t }$ dynamically controls how much past information is retained or forgotten at each timestep.

This adaptive gating mechanism maps directly onto crop phenological transitions. During early green-up transitions, the model enforces strong retention (“remember” gating) to propagate key onset signals into the peak summer growing season. Conversely, during harvest or senescence phases, selective decay (“forget” gating) allows the network to reset accumulated biomass states. In contrast, standard 3D convolutions apply static kernel weights uniformly across all timesteps, inherently limiting their capacity to accommodate inter-pixel phenological heterogeneity.

The asymmetry of MODIS time series, where most phenological information is encoded in the ordering of spectral transitions rather than the absolute reflectance values [19], further motivates SSM over convolutional architectures for temporal encoding. SSMs preserve temporal causality and are theoretically equivalent to infinite-receptive-field filters [18], whereas 3D convolutions are limited to the kernel window.

In remote sensing, state space models have seen rapid adoption since 2024, particularly for high dimensional spatial spectral feature extraction in hyperspectral classification via architectures such as S<sup>2</sup>Mamba [20] and SpectralMamba [21]. For spectral unmixing specifically, UNMamba [22] applies Mamba to demonstrate competitive performance with significantly reduced computational overhead compared to transformerbased alternatives. However, UNMamba and its contemporaries apply the SSM scan over the spectral (band) dimension rather than the temporal dimension, treating each wavelength as a sequence element. Closer to our setting, recent Mamba models scan the temporal axis of satellite image time series for crop classification [23, 24], confirming the value of SSMs for phenological sequence modeling, though they target hard classification rather than sub-pixel unmixing with endmember variability. MDF-Mamba addresses this gap via a hybrid CNN-Mamba architecture that applies the SSM selectively to the temporal axis while handling spectral-spatial structure through per-channel CNN experts (Contribution 1), fused via mHCFusion (Contribution 2), as detailed in Section III.

## III. PROPOSED MDF-MAMBA ARCHITECTURE

Fig. 1 illustrates the overall pipeline of the proposed MDF-Mamba model. Let $\mathbf { X } \in \mathbb { R } ^ { B \times L ^ { \ddots } \dot { H } \times W }$ denote a MODIS input patch, where B is the batch size, $L = C \cdot T = 6 \times 2 3 = 1 3 8$ is the total number of spectral-temporal channels $( C = 6$ spectral channels, $T ~ = ~ 2 3$ time steps), and $H \times W$ is the spatial resolution. The model produces fractional abundance maps $\hat { \textbf { A } } \in \mathbb { R } ^ { B \times P \times H \times W }$ for $P ~ = ~ 1 4$ land cover classes, along with a physics-constrained spectral reconstruction $\hat { \mathbf { X } } \in \mathbb { R } ^ { B \times L \times H \times W }$ . The architecture is organized around four modules that correspond directly to the four contributions introduced in Section I: Section III-A presents the asymmetric multi-branch disentangling design; Section III-B describes the manifold-constrained hyper-connection fusion; Section III-C details the extended linear mixing model with pixel-wise endmember variability; and Section III-D defines the multi-task supervised-unsupervised loss framework.

A. Spectral-Temporal Disentangling via Multi-Branch Architecture

The 138-band MODIS stack encodes $C = 6$ physically distinct spectral channels (NDVI, EVI, Red, NIR, Blue, MIR), each with its own phenological trajectory over $T = 2 3$ 16- day composites. Processing all L channels through shared parameters conflates these cues. MDF-Mamba resolves this by assigning each spectral channel to a dedicated CNN expert, while a parallel Mamba branch encodes the temporal phenology shared across channels.

A shallow cross-band channel-mixing stem, implemented as a 1×1 convolution with batch normalization and SiLU activation, is first applied across all 138 bands to calibrate comparable feature scales and capture first-order inter-channel correlations (e.g., the Red–NIR relationship underlying vegetation indices); this brief joint step precedes the fully disentangled perchannel processing that follows, since all subsequent temporalphenology encoding occurs independently within each of the six BandBranches, producing $\mathbf { X } ^ { \prime } \in \mathbb { R } ^ { B \times \bar { L } \times H \times W }$

The output from this initial stem is then partitioned into six distinct per-channel spectral streams through strided channel indexing:

$$
\mathbf { X } _ { k } = \mathbf { X } ^ { \prime } [ : , k : : C , : , : ] \in \mathbb { R } ^ { B \times T \times H \times W } , \quad k = 0 , \ldots , C - 1\tag{3}
$$

Each stream is then processed by a dedicated BandBranch expert $f _ { k } \ ( E l e m e n t B r a n c h C N N , \ d { = } 1 2 8 )$ . The expert works in three steps: two parallel feature paths, a twist gate that lets the two paths exchange information, and a head that outputs class logits.

Parallel paths. The spectral path uses only 1×1 convolutions, so it transforms each pixel’s 23-step temporal profile independently of its neighbours. The spatial path uses $3 \times 3$ convolutions, so it also captures local spatial context such as field and patch boundaries. Each path stacks three blocks of convolution, batch normalization (BN), and GELU activation:

$$
\begin{array} { r } { \Phi _ { k } ^ { \mathrm { s p e c } } = \big ( \mathrm { G E L U } \circ \mathrm { B N } \circ \mathrm { C o n v } _ { 1 \times 1 } \big ) ^ { 3 } ( \mathbf { X } _ { k } ) , } \\ { \Phi _ { k } ^ { \mathrm { s p a t } } = \big ( \mathrm { G E L U } \circ \mathrm { B N } \circ \mathrm { C o n v } _ { 3 \times 3 } \big ) ^ { 3 } ( \mathbf { X } _ { k } ) , } \end{array}\tag{4}
$$

![](images/df9e7f5e84d40bdc45cf79dd2524404e1f1f4434666dc169716db167721c6705.jpg)  
Fig. 1. Overview of the proposed MDF-Mamba framework. The MODIS spectral–temporal input is processed through four stages. Stage 1 combines six band-specific CNN experts, each equipped with spectral and spatial paths and a Twist Gate, with a parallel Temporal Mamba branch. Cross-branch SE and phenology gates produce three complementary semantic streams. Stage 2 integrates these streams using manifold-constrained fusion and scene-adaptive modulation to estimate the fractional abundance map A<sup>ˆ</sup> . Stage 3 combines variable-endmember ELMM reconstruction and class-constrained cross-attention through a purity-aware blend to generate the spectral reconstruction X<sup>ˆ</sup> . The model produces fractional abundance maps, an abundance-derived classification map, and a physics-constrained spectral reconstruction map. Stage 4 jointly optimizes the abundance, reconstruction, and regularization objectives using a two-phase curriculum. Solid and dashed orange arrows represent forward flow and training supervision, respectively.

where ◦ denotes function composition, the exponent 3 denotes three stacked blocks with separate weights, and both outputs lie in $\mathbb { R } ^ { B \times d \times H \times W }$

Twist gate. After the third block, each path is refined with information from the other path. Two gates are computed from the concatenated features of both paths, each with its own 1×1 convolution followed by a sigmoid σ:

$$
\begin{array} { r l } & { \mathbf { G } _ { k } ^ { \mathrm { s p a t } } = \sigma \big ( \mathrm { C o n v } _ { 1 \times 1 } \big ( \big [ \boldsymbol { \Phi } _ { k } ^ { \mathrm { s p a t } } \big \rVert \boldsymbol { \Phi } _ { k } ^ { \mathrm { s p e c } } \big ] \big ) \big ) , } \\ & { \mathbf { G } _ { k } ^ { \mathrm { s p e c } } = \sigma \big ( \mathrm { C o n v } _ { 1 \times 1 } \big ( \big [ \boldsymbol { \Phi } _ { k } ^ { \mathrm { s p a t } } \big \rVert \boldsymbol { \Phi } _ { k } ^ { \mathrm { s p e c } } \big ] \big ) \big ) , } \end{array}\tag{5}
$$

where ∥ denotes channel concatenation. Each gate takes a value in (0, 1) for every pixel and channel. $\mathbf { G } _ { k } ^ { \mathrm { s p a t } }$ controls how much spectral information is added to the spatial features, and $\mathbf { G } _ { k } ^ { \mathrm { s p e c } }$ controls how much spatial information is added to the spectral features:

$$
\begin{array} { r l } & { \tilde { \Phi } _ { k } ^ { \mathrm { s p a t } } = \Phi _ { k } ^ { \mathrm { s p a t } } + \mathbf { G } _ { k } ^ { \mathrm { s p a t } } \odot \Phi _ { k } ^ { \mathrm { s p e c } } , } \\ & { \tilde { \Phi } _ { k } ^ { \mathrm { s p e c } } = \Phi _ { k } ^ { \mathrm { s p e c } } + \mathbf { G } _ { k } ^ { \mathrm { s p e c } } \odot \Phi _ { k } ^ { \mathrm { s p a t } } , } \end{array}\tag{6}
$$

where ⊙ denotes element-wise multiplication. The spatial features thus become aware of each pixel’s spectral-temporal identity, and the spectral features become aware of local boundary geometry. Because the exchanged information is added to each path rather than replacing it, each path keeps its own features: a gate value near zero leaves the path unchanged, and a value near one adds the full features of the other path.

Classification head. Finally, the two refined feature maps are joined along the channel dimension and passed to a small classification head $\psi _ { k }$ . The head first mixes the spatial and spectral features with a 1×1 convolution, BN, and GELU. A second 1×1 convolution then reduces them to P channels, giving one score per land cover class at every pixel. These scores are the class logits of expert k, which are later turned into abundances by a softmax (Eq. (12)):

$$
\begin{array} { r } { \mathbf { Z } _ { k } = f _ { k } ( \mathbf { X } _ { k } ) = \psi _ { k } \Big ( [ \tilde { \Phi } _ { k } ^ { \mathrm { s p a t } } \parallel \tilde { \Phi } _ { k } ^ { \mathrm { s p e c } } ] \Big ) } \\ { \in \mathbb { R } ^ { B \times P \times H \times W } , \quad k = 0 , \dots , C - 1 } \end{array}\tag{7}
$$

The six BandBranches form a fixed-assignment multi-branch expert ensemble, each specialising in one band and eliminating cross-band interference. Unlike conventional MoE, expert assignment is fixed by channel rather than learned via a gating network, with only the downstream combination weights learned (Cross-Branch SE, below). This design is motivated by the distinct per-channel temporal signatures in Fig. 3.

To encode cross-channel phenology, a seventh temporal branch utilizes a bidirectional selective state space model (SSM) [23, 24]. The 138-band stack is spatially downsampled 2× and reshaped into per-pixel sequences of T=23 sixdimensional tokens, each linearly projected to $d _ { \mathrm { m o d e l } } { = } 6 4$ . Each SSM layer performs input-dependent state transitions:

$$
\mathbf { h } _ { t } = e ^ { \mathbf { A } \odot \mathbf { \vec { \Delta } } _ { t } } \odot \mathbf { h } _ { t - 1 } + \mathbf { \Delta } \Delta _ { t } \otimes \mathbf { B } _ { t } \otimes \mathbf { U } _ { t }
$$

$$
\mathbf { y } _ { t } = \mathbf { C } _ { t } ^ { \top } \mathbf { h } _ { t } + \mathbf { D } \odot \mathbf { U } _ { t }\tag{8}
$$

(9)

where A is a learned decay matrix and the input-dependent $\Delta _ { t }$ is designed to control temporal memory length, shortening context during rapid phenological transitions and lengthening it during stable mid-summer periods. The bidirectional output is pooled by concatenating the annual mean with the first (t=1, January) and last (t=T, December) composite tokens, followed by a linear projection to reduce the concatenated $3 d _ { \mathrm { m o d e l } } = 1 9 2$ representation down to $d _ { \mathrm { f u s e } } { = } 6 4$ , then bilinearly upsampled to $( H , W )$ to give the temporal feature $\mathbf { a } _ { \mathrm { t e m p } } .$ This 2× downsample-then-upsample cycle trades fine spatial resolution in $\mathbf { a } _ { \mathrm { t e m p } }$ for tractable SSM compute.

The six spectral features and the temporal feature are projected to a shared $d _ { \mathrm { f u s e } } { = } 6 4$ space where a cross-branch squeeze-and-excitation module allows the experts to communicate through soft inter-expert attention weights. Each aligned spectral stream $\mathbf { a } _ { k } ^ { \prime }$ is modulated by the temporal feature via a spatially adaptive phenology gate:

$$
\tilde { \mathbf { a } } _ { k } = \mathbf { a } _ { k } ^ { \prime } + \mathbf { b } _ { k } \odot \mathbf { g } _ { k } \odot \mathbf { a } _ { \mathrm { t e m p } } , \quad k = 0 , \ldots , C - 1\tag{10}
$$

where $\mathbf { g } _ { k } = \sigma \big ( \mathrm { C o n v } _ { 1 \times 1 } \big ( \big [ \mathbf { a } _ { k } ^ { \prime } \ \lVert \mathbf { a } _ { \mathrm { t e m p } } \big ] \big ) \big )$ is a content-dependent gate and ${ \bf b } _ { k } = \sigma \big ( \mathrm { B N } ( \mathrm { D W } \mathrm { C o n v } _ { 3 \times 3 } ( { \bf a } _ { k } ^ { \prime } ) ) \big ) \in ( 0 , 1 ) ^ { d _ { \mathrm { f u s e } } \times H \times W }$ is a budget map learned from the spectral stream itself, with separate weights for each branch, so that pixels in transitional zones can receive stronger temporal context than homogeneous interiors.

Following modulation, the six streams are merged into three semantically coherent hierarchical groups by physical domain:

$$
\begin{array} { r l } & { \mathbf { s } _ { \mathrm { v e g } } = f _ { \mathrm { v e g } } ( [ \tilde { \mathbf { a } } _ { 0 } \lVert \tilde { \mathbf { a } } _ { 1 } ] ) \quad ( \mathrm { N D V I + E V I } ) } \\ & { \mathbf { s } _ { \mathrm { o p t } } = f _ { \mathrm { o p t } } ( [ \tilde { \mathbf { a } } _ { 2 } \lVert \tilde { \mathbf { a } } _ { 3 } \rVert \tilde { \mathbf { a } } _ { 4 } ] ) \quad ( \mathrm { R e d + N I R + B l u e } ) } \\ & { \mathbf { s } _ { \mathrm { m i r } } = \tilde { \mathbf { a } } _ { 5 } \quad ( \mathbf { M I R } ) } \end{array}\tag{11}
$$

where $f _ { \mathrm { v e g } }$ and $f _ { \mathrm { o p t } }$ are $1 \times 1$ Conv + BN + SiLU projections, reducing the fusion input from six streams to $N { = } 3$ and improving semantic composition.

This hierarchical grouping is physically motivated [34]. By explicitly separating derived photosynthetic indices (Vegetation), structural surface reflectance (Optical), and water absorption properties (Moisture) into coherent physical domains, we prevent the fusion module from struggling to disentangle conflicting gradients and enforce a semantically structured fusion process.

A preliminary rough abundance map is obtained by averaging the six per-branch logits:

$$
\hat { \bf A } _ { \mathrm { r o u g h } } = \mathrm { s o f t m a x } \left( \frac { 1 } { C } \sum _ { k = 0 } ^ { C - 1 } { \bf Z } _ { k } \right) \in \mathbb { R } ^ { B \times P \times H \times W }\tag{12}
$$

with per-branch outputs $\hat { \mathbf { A } } _ { k } = \operatorname { s o f t m a x } ( \mathbf { Z } _ { k } )$ providing independent per-expert supervision targets (Section III-D).

## B. Manifold-Constrained Hyper-Connection Fusion

The N=3 semantic streams $\left\{ \mathbf { s } _ { \mathrm { v e g } } , \mathbf { s } _ { \mathrm { o p t } } , \mathbf { s } _ { \mathrm { m i r } } \right\}$ are integrated by the mHCFusion module [32, 33] in a $d _ { \mathrm { f u s e } } { = } 6 4$ space. The streams are stacked and passed through two mHC blocks, each applying a width step (Sinkhorn-projected doubly-stochastic mixing across streams, ensuring every stream sends and receives equal total weight without collapsing) and a depth step (FFN with 4× expansion). A softmax stream gate with RMSNorm and a residual skip then aggregate the streams into a fused map $\mathbf { F } _ { \mathrm { f u s e } } \in \mathbb { R } ^ { B \times \hat { d _ { \mathrm { f u s e } } } \times H \times W }$

A Feature-wise Linear Modulation (FiLM) classification head conditions the output on global scene context. Global average pooling (GAP) of $\mathbf { F } _ { \mathrm { f u s e } }$ summarizes the scene, and a small MLP turns this summary into per-channel scale and shift parameters $\gamma , \beta \in \mathbb { R } ^ { d _ { \mathrm { f u s e } } }$ , which modulate the fused feature:

$$
\begin{array} { r } { \left[ \gamma \parallel \beta \right] = \mathrm { M L P } \big ( \mathrm { G A P } ( \mathbf { F } _ { \mathrm { f u s e } } ) \big ) , } \\ { \mathbf { F } _ { \mathrm { m o d } } = \mathbf { F } _ { \mathrm { f u s e } } \odot ( 1 + \gamma ) + \beta } \end{array}\tag{13}
$$

The modulated feature $\mathbf { F } _ { \mathrm { m o d } }$ is projected to $P$ class logits $\mathbf { F } ,$ and each logit is divided by a learnable per-class temperature $\tau _ { p } \in [ 0 . 3 , 2 . 0 ]$ before the softmax; a smaller $\tau _ { p }$ gives sharper, more pure-like abundances:

$$
\hat { \mathbf { A } } = \mathrm { s o f t m a x } ( \mathbf { F } \oslash \pmb \tau ) \in \mathbb { R } ^ { B \times P \times H \times W }\tag{14}
$$

By applying the softmax activation across the class dimension, this formulation intrinsically enforces both the Abundance Sumto-One Constraint (ASC, $\begin{array} { r } { \dot { \sum _ { p } A _ { p } } = 1 ) } \end{array}$ and the Abundance Nonnegativity Constraint $( \mathsf { A N C } , \hat { A } _ { p } \geq 0 )$ at each pixel, ensuring physically valid mixing fractions. Subsequently, the discrete land cover map for hard-classification evaluation is obtained via a pixel-wise argmax over the class dimension of A<sup>ˆ</sup> .

A single fixed endmember per class cannot represent intraclass spectral variability driven by soil moisture, topography, and phenology. MDF-Mamba adopts an Extended Linear Mixing Model (ELMM) [26] with R=10 spectral variants per class.

The endmember variant bank $\mathbf { E } \in \mathbb { R } ^ { P \times R \times L }$ is derived from a learnable embedding matrix $\mathbf { Q } \in \mathbb { R } ^ { ( P \cdot R ) \times L }$ passed through a ReLU activation for non-negativity. The representative class mean for class p is defined as:

$$
\bar { \mathbf { e } } _ { p } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \mathbf { E } _ { p , r , : }\tag{15}
$$

The rows of Q are initialized from high-purity training pixels with dominant abundance $\ge 0 . 8 0$ (supplemented by a $4 \times 5$ spatial grid fallback thresholded at $\geq 0 . 3 0 )$ and are constrained to the valid spectral range [0, 1] throughout training.

A lightweight $1 \times 1$ convolution $h _ { \mathrm { v a r } } \ ( P  P R$ channels) acts as a variability head that maps the class logits F of Eq. (14), taken before temperature scaling, to per-pixel variant mixing weights over the endmember bundles [35]:

$$
\mathbf { W } = \mathrm { s o f t m a x } _ { R } ( h _ { \mathrm { v a r } } ( \mathbf { F } ) ) \in \mathbb { R } ^ { B \times P \times R \times H \times W }\tag{16}
$$

and the Extended Linear Mixing Model (ELMM) [36] reconstruction combines class abundances with these pixel-adaptive convex combinations of variants:

$$
\hat { X } _ { b , l , h , w } ^ { \mathrm { E L M M } } = \sum _ { p = 0 } ^ { P - 1 } \hat { A } _ { b , p , h , w } \sum _ { r = 0 } ^ { R - 1 } W _ { b , p , r , h , w } \cdot E _ { p , r , l }\tag{17}
$$

While physically valid strictly for linear reflectance bands (e.g., Red, NIR, Blue, MIR), the model applies this linear mixing across all $L \ = \ 1 3 8$ channels, including non-linear vegetation indices (NDVI, EVI). In line with feature-level selfsupervised reconstruction objectives, this treats index mixing as a simplified approximation for deep representation learning rather than a strict physical radiative transfer law [37].

To model complex non-linear scattering effects, a lightweight cross-attention module is introduced to generate a non-linear feature reconstruction, $\hat { \mathbf { X } } _ { \mathrm { a t t n } }$ . To mitigate spectral borrowing from erroneous endmember classes, this attention mechanism is constrained such that each pixel selectively attends only to the spectral variant banks associated with its top-two predicted classes.

The resulting non-linear representation is subsequently fused with the physics-driven ELMM reconstruction, X<sup>ˆ</sup> <sub>ELMM</sub>, via a purity gating mechanism. This gate, defined as $\rho =$ $\sigma \Big ( ( \mathrm { m a x } _ { p } \hat { A } _ { p } - 0 . 8 0 ) \times 1 0 \Big )$ , adaptively quantifies pixel purity based on the maximum predicted fractional abundance. As formulated in Equation 18, $\rho$ acts as a soft routing function: highly pure pixels $( \rho  1 )$ are predominantly guided toward the physically constrained ELMM output. Conversely, for highly mixed pixels $( \rho \to 0 )$ , the architecture adaptively incorporates the non-linear attention output. This integration is modulated by a learned, band-specific scaling parameter, $\sigma ( \lambda _ { b } )$ (empirically initialized to 0.2), ensuring that non-linear adjustments do not induce spectral over-distortion:

$$
\hat { \mathbf { X } } = \left( 1 - \sigma ( \lambda _ { b } ) ( 1 - \rho ) \right) \hat { \mathbf { X } } _ { \mathrm { E L M M } } + \sigma ( \lambda _ { b } ) ( 1 - \rho ) \hat { \mathbf { X } } _ { \mathrm { a t t n } }\tag{18}
$$

Because the cross-attention reconstructor can partially reproduce the input spectrum independently of the ELMM abundances, a low blended reconstruction error (X<sup>ˆ</sup> ) evaluates the model’s overall representation quality rather than the physical ELMM alone. We report the blended reconstruction metrics as a proxy for the overall framework’s fidelity, while noting that pure ELMM reconstruction would be required to independently validate the abundance solution.

Where $\lambda _ { b }$ is a learnable mixing scalar. To suppress highfrequency noise while preserving physically meaningful temporal continuity (e.g., phenological trajectories), a Laplacian smoothness penalty is applied to the mean endmembers, E<sup>¯</sup> . This penalty acts independently along the temporal dimension $( t \in [ 0 , T - 1 ] )$ within each of the C physical channels, thereby strictly avoiding artificial smoothing across distinct spectral channel boundaries:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { P \cdot C \cdot ( T - 2 ) } \sum _ { p = 0 } ^ { P - 1 } \sum _ { c = 0 } ^ { C - 1 } \sum _ { t = 1 } ^ { T - 2 }\tag{19}
$$

## C. Multi-Task Supervised-Unsupervised Learning

MDF-Mamba is trained with a multi-task loss combining supervised abundance estimation, per-expert regularization, physics-constrained spectral reconstruction, spatial coherence regularization, and endmember regularization:

$$
\begin{array} { r } { \begin{array} { r l } & { \mathscr { L } = \mathscr { L } _ { \mathrm { a b u } } + 0 . 5 \mathscr { L } _ { \mathrm { a u x } } + 0 . 3 \mathscr { L } _ { \mathrm { b r } } } \\ & { ~ + \mathrm { b o o s t } \cdot \left( 0 . 1 \mathscr { L } _ { \mathrm { m a h a } } + 0 . 1 \mathscr { L } _ { \mathrm { s a d } } + 0 . 5 \mathscr { L } _ { \mathrm { r e c o n } } + 0 . 1 \mathscr { L } _ { \mathrm { r s a d } } \right) } \\ & { + \lambda _ { \mathrm { c o r r } } ( t ) \mathscr { L } _ { \mathrm { c o r r } } + 0 . 0 5 \mathscr { L } _ { \mathrm { d i r } } + \lambda _ { \mathrm { a n c } } ( t ) \mathscr { L } _ { \mathrm { a n c } } + \lambda _ { \mathrm { s m o o t h } } ( t ) \mathscr { L } _ { \mathrm { s m o o t h } } } \end{array} } \end{array}\tag{20}
$$

The supervised abundance loss combines an L1 magnitude penalty with a KL shape penalty between the refined abundance A<sup>ˆ</sup> and ground-truth labels $\mathbf { A } ^ { * }$

$$
\mathcal { L } _ { \mathrm { a b u } } = \| \hat { \mathbf { A } } - \mathbf { A } ^ { * } \| _ { 1 } + \lambda _ { \mathrm { k l } } \cdot \mathrm { K L } ( \mathbf { A } ^ { * } \| \hat { \mathbf { A } } )\tag{21}
$$

The same L1+KL form provides per-expert supervision on the rough abundance $( \mathcal { L } _ { \mathrm { a u x } } )$ and on each per-branch output $( \mathcal { L } _ { \mathrm { b r } }$ averaged over k), so that every expert is supervised before phenology gating and no branch free-rides on the Mamba output. Five spectral reconstruction-related terms enforce fidelity between the blended reconstruction $\hat { \bf X }$ and the MODIS observation $\mathbf { X } \colon \mathbf { a }$ whitened Mahalanobis error $( { \mathcal { L } } _ { \mathrm { m a h a } } )$ , spectral angle distance $( \mathcal { L } _ { \mathrm { s a d } } , \mathcal { L } _ { \mathrm { r s a d } } )$ , and inverse-variance-weighted MSE $( \mathcal { L } _ { \mathrm { r e c o n } } )$ are jointly scaled by the curriculum boost factor, while a covariance-matrix MSE $( \mathcal { L } _ { \mathrm { c o r r } } )$ preserving the inter-band correlation structure of MODIS data is scaled independently via $\lambda _ { \mathrm { c o r r } } ( t )$ . A spatial stiffness term ${ \mathcal { L } } _ { \mathrm { d i r } }$ penalises abundance discontinuities between spectrally similar neighbouring pixels (weighted by a Gaussian spectral-affinity kernel) to suppress salt-and-pepper noise. Finally, endmember regularisation keeps the learned signatures physically plausible: ${ \mathcal { L } } _ { \mathrm { a n c } }$ anchors the mean endmembers E<sup>¯</sup> to their GT initialisation $\mathbf { E } ^ { ( 0 ) }$ with a weight $\lambda _ { \mathrm { a n c } } ( t )$ that is annealed over training, and $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ enforces spectral continuity as defined in (19). The loss weights follow a two-phase curriculum that emphasises reconstruction fidelity while the endmember dictionary stabilises and then shifts toward abundance accuracy; the exact schedule is given in Section IV-C.

## IV. EXPERIMENTS

## A. Datasets

We evaluate the proposed MDF-Mamba model on a MODIS time series dataset covering the provinces of Alberta and Saskatchewan in western Canada. The dataset is constructed from the MOD13Q1 product (Collection 6.1), which provides 16-day composite vegetation indices and surface reflectance at 250 m spatial resolution. The temporal stack spans the full 2010 calendar year (January–December) and includes six spectral channels: NDVI, EVI, Red, NIR, Blue, and MIR, each with 23 temporal observations, yielding $L = 1 3 8$ spectral-temporal bands per pixel.

Fractional abundance labels for the land cover classes provide the ground truth and are derived from the 2010 North American Land Change Monitoring System (NALCMS) at 30 m resolution. The dataset comprises 14 distinct classes present within the Alberta study area. To maintain traceability with the broader 19-class North American spatial data framework, we retain the standardized integer class codes, which naturally omit tropical and absent sub-polar classes. The 14 evaluated classes are: (1) Temperate/sub-polar needleleaf forest, (2) Subpolar taiga needleleaf forest, (5) Temperate/sub-polar broadleaf deciduous forest, (6) Mixed forest, (8) Temperate/sub-polar shrubland, (10) Temperate/sub-polar grassland, (11) Sub-polar shrubland-lichen-moss, (12) Sub-polar grassland-lichen-moss, (14) Wetland, (15) Cropland, (16) Barren lands, (17) Urban and built-up, (18) Water, and (19) Snow and ice.

Fractional abundances are computed by aggregating the 30 m labels within each 250 m MODIS pixel. An independent 250 m land cover classification map is used as reference for pixel-level classification evaluation. Of these 14 classes, the two sub-polar lichen-moss classes (11, 12) are exceedingly rare and spectrally inseparable from background at MODIS resolution (Section IV-B); they are therefore excluded from the evaluation, leaving $P = 1 2$ active classes for the final abundance and classification outputs. Note that the data-exploration figures (Figs. 2 and 3) and spatial error distributions (Fig. 9b) retain 14 classes to visualize the lichen-moss classes’ poor separability and their corresponding errors; the learned endmember analysis (Fig. 10) and the final global abundance and classification metrics (Tables I–IV), however, use P = 12, excluding both lichen-moss classes.

For data preparation, the study area is divided into nonoverlapping patches of 64 × 64 pixels. Patches containing any no-data values are excluded. To ensure a rigorous evaluation of spatial generalization and prevent the spatial leakage inherent to random partitions [38], the dataset is split using a strict geographically-blocked hold-out protocol. The scene is partitioned into a regular grid of non-overlapping spatial blocks, with each block assigned exclusively to the training (70.0%), validation (15%), or test (15%) partition. To explicitly prevent spatial autocorrelation, a strict exclusion buffer is enforced at every block boundary, guaranteeing that no 64 × 64 patch can straddle a split. Rather than extracting patches uniformly across the map, patches are drawn exclusively from the valid pools within their respectively assigned blocks, resulting in 2,731 training, 585 validation, and 585 test patches. All spectral bands are normalized to [0, 1] using global min-max statistics computed from the training set.

## B. Data Exploration and Purity Analysis

To better understand the dataset characteristics and justify the need for spectral unmixing, we performed an exploratory analysis on the extracted MODIS patches.

To evaluate class imbalance and inter-class separability, Fig. 2 summarises two complementary dataset properties. Panel (a) shows the pure-pixel sample counts per class at 80% abundance purity. The distribution spans nearly four orders of magnitude: Needleleaf Forest contributes over 3.8 million pure pixels while Urban and Broadleaf Forest each provide fewer than 400,000, creating a strongly imbalanced training signal that directly influences endmember quality. Notably, sub-polar lichen-moss classes (Sub-polar Shrub-Lichen and Sub-polar Grass-Lichen) each yield roughly 300,000 pixels yet exhibit near-zero spectral range across all 138 bands (range < 0.08 normalised units), indicating that even these nominally pure pixels are spectrally indistinguishable from mixed background. This observation motivates the geographic grid fallback initialisation of the endmember embedding described in Section III-C.

Panel (b) shows the pairwise spectral angle distance (SAD) matrix between the 14 GT endmembers computed from pure pixels. Several class pairs have low SAD (< 0.30 rad, corresponding to high spectral similarity): Sub-polar Shrub-Lichen vs. Sub-polar Grass-Lichen (0.05 rad), Needleleaf Forest vs. Mixed Forest (0.18 rad), and Wetland vs. Grassland (0.22 rad). These low-SAD pairs directly correspond to the classes with the highest per-class abundance MAE in Fig. 9(b) (needleleaf: 0.0801, mixed forest: 0.0574, wetland: 0.0569), consistent with inter-class spectral confusion being a major contributor to unmixing difficulty in these classes. We adopt 0.30 rad (≈17.2<sup>◦</sup>) as the threshold for spectral confusability in the sub sequent analysis; this boundary empirically captures the sharp inflection point in abundance MAE and aligns with standard angular tolerances for identifying highly correlated, structurally similar vegetation in multispectral classification [39].

Analyzing branch-specific temporal signatures, Fig. 3 presents the learned endmember temporal profiles for all 14 active land cover classes across the six spectral branches (NDVI, EVI, Red, NIR, Blue, MIR). Each panel shows one spectral branch with all 14 class signatures overlaid as color-coded lines over 23 MOD13Q1 composites spanning the full 2010 calendar year. The panels reveal markedly distinct inter-class separability per branch: NDVI captures the sharp summer greenup that separates cropland from evergreen forest classes [40], NIR distinguishes coniferous from deciduous phenology via their late-season reflectance plateaus [41], and MIR separates wetland from grassland through moisture-driven reflectance contrasts [42]. Critically, no single branch achieves full class separation (the lichen-moss biomes overlap with other classes in every branch), supporting the need for all six spectral dimensions and motivating the multi-branch disentangling architecture of MDF-Mamba.

A purity analysis is conducted in Fig. 4, which presents the spectral range (max−min of the mean spectrum) and perpixel standard deviation for each class as a function of the abundance purity threshold, swept from 0.50 to 0.95. The red dashed vertical line marks the 0.80 threshold used for GT endmember extraction in the proposed model. Two key observations emerge.

The pixel counts (grey bars) decrease steeply as the purity threshold rises from 0.50 to 0.95, confirming that highly pure pixels are rare at 250 m MODIS resolution and that the vast majority of pixels contain contributions from multiple land cover classes. This prevalence of mixed pixels strictly necessitates sub-pixel spectral unmixing over hard-classification pipelines [43, 44].

Additionally, the spectral characteristics behave differently across classes as purity increases. For spectrally rich vegetative classes such as Needleleaf Forest and Cropland, while the absolute spectral range slightly narrows, the mean standard deviation (variance) of the spectra actually rises or remains stable, indicating that higher purity pixels isolate a highly variable and representative phenological signature. In contrast, for sub-polar lichen-moss classes, the spectral range remains near zero across the entire threshold sweep, confirming the complete absence of spectrally distinguishable pure pixels for these biomes regardless of purity level. Similarly, nonvegetative classes such as Snow/Ice exhibit low and constant spectral variance because their reflectance lacks seasonal phenological dynamics. For intermediate classes such as Barren Lands, Urban, and Wetland, the spectral range peaks at lower purity thresholds and then decreases sharply, suggesting that the few very high purity pixels are unrepresentative outliers rather than canonical class signatures.

These observations directly informed three design decisions: (1) the 80% purity threshold for GT endmember extraction, which balances spectral representativeness against sample count for the majority of classes; (2) the geographic fallback initialisation for classes where no high-purity pixels exist; and (3) the R = 10 learnable spectral variants in the ELMM, which capture the intra-class variability that remains significant even at the maximum achievable purity.

![](images/d54c7e874f71f50cec72c46f0ec0990737e600cff4666b894a25af3345860ff0.jpg)

![](images/b5654b5f353276c3f78e3cbdba9deaf92e2ca3d08ec4bfa1aeac27d0796acbe0.jpg)  
Fig. 2. Dataset statistics for the Alberta-Saskatchewan MODIS dataset. (a) Pure-pixel sample counts per land cover class at 80% abundance purity (log scale) (b) Pairwise spectral angle distance (SAD) matrix between the 14 GT endmembers. Low SAD values (< 0.30 rad, dark cells) identify class pairs that ar spectrally confusable, motivating the multi-branch disentangling design.

![](images/1c30d984e39ca0aed7a6a27fd3ae8aeed173bf1e7942f23e596bb377ed43b98e.jpg)  
Fig. 3. Learned endmember temporal profiles (mean ±1σ over the $R { = } 1 0$ variants) for 14 land cover classes across six spectral branches (NDVI, EVI, Red, NIR, Blue, MIR). Each panel shows the 23-step seasonal signature of every class as a distinct color-coded line with a faint ±1 standard deviation band, illustrating which spectral dimensions provide the strongest class separability and motivating the multi-branch architecture.

## C. Implementation Details

The model is implemented in PyTorch and trained using the Adam optimizer with an initial learning rate of $1 0 ^ { - \bar { 3 } }$ and a separate learning rate of $5 \times 1 0 ^ { - 4 }$ for the endmember embedding Q and temperature τ parameters. We employ a StepLR scheduler that decays the learning rate by a factor of 0.9 every 10 epochs. Training proceeds for 100 epochs with a batch size of 8 on a single NVIDIA A6000 GPU (48 GB).

Gradient clipping with a maximum norm of 1.0 is applied throughout training.

To solve the complex physical optimization landscape of multispectral unmixing, we employ a dedicated curriculum learning strategy. For the first 30 epochs, the model focuses heavily on physical consistency to stabilize the endmember dictionary by up-weighting all reconstruction-based losses by a factor of $3 . 0 ,$ strongly enforcing an endmember Laplacian smoothness prior $( \lambda _ { \mathrm { s m o o t h } } = 0 . 5 )$ , and heavily weighting spectral band covariance $( \lambda _ { \mathrm { c o r r } } = 1 . 0 )$ . After epoch 30, the network relies on the converged dictionary and shifts its representational power toward abundance refinement by returning reconstruction penalties to their base weights $( \lambda _ { \mathrm { r e c o n } } = 0 . 5 , \lambda _ { \mathrm { s a d } } = 0 . 1 )$ Concurrently, $\lambda _ { \mathrm { c o r r } } ( t )$ anneals from $1 . 0 \ \mathrm { t o } \ 0 . 2 , \ \lambda _ { \mathrm { s m o o t h } } ( t )$ from 0.5 to 0.05, and the endmember anchor penalty $\lambda _ { \mathrm { a n c } } ( t )$ is cosine-annealed continuously from 1.0 down to 0.1 over the full 100 epochs. This ensures the model starts tightly anchored to the initial pure-pixel endmembers, but progressively gains the mathematical freedom to capture unseen non-linear variance via the adaptive cross-attention reconstructor.

The base loss weights are $\lambda _ { \mathrm { a u x } } = 0 . 5 , \lambda _ { \mathrm { b r } } = 0 . 3 , \lambda _ { \mathrm { m a h a } } = 0 . 1$ $\lambda _ { \mathrm { s a d } } = 0 . 1 , \lambda _ { \mathrm { r s a d } } = 0 . 1 , \lambda _ { \mathrm { d i r } } = 0 . 0 5 , \lambda _ { \mathrm { r e c o n } } = 0 . 5 , \lambda _ { \mathrm { c o r r } } = 0 . 2 ,$ $\lambda _ { \mathrm { s m o o t h } } = 0 . 0 5$ . The number of endmember variants is $R = 1 0$ The hierarchical fusion module operates at $d _ { \mathrm { f u s e } } = 6 4$ with $n _ { \mathrm { l a y e r s } } = 2$ inner mHC blocks on N=3 semantically grouped streams (vegetation, optical, moisture). The FiLM classification head uses a two-layer MLP $( 6 4  3 2  1 2 8 )$ initialized to zero for near-identity early behaviour. The cross-attention reconstructor uses $d _ { \mathrm { a t t n } } ~ = ~ 6 4$ with the blend coefficient initialized at $\sigma ( \lambda _ { b } ) \approx 0 . 2$ . The endmember embedding Q is initialized from GT pure-pixel endmembers $( \mathrm { p u r i t y } \geq 0 . 8 0 )$ with geographic fallback (purity ≥ 0.30) as described in Section III-C. The six ElementBranchCNN BandBranches each use a hidden dimension of $d = 1 2 8 .$ . The per-branch adaptive phenology gates use depthwise $3 \times 3$ convolution with sigmoid

Spectral range Std Pixel count 80% purity threshold Max range purity

![](images/81f03908a1c2d2ac1101b1cb82db979cbbe92cda54f622ad9c2687dbab141c1e.jpg)

![](images/405b77a195e846195a190150ba741f0bf944b707a6593c9db8e2d1d65f5e69ad.jpg)

![](images/aaa1b96069835fa5c673d7b45c9b0f2ec0199e221771c8180cc4ba317a2b20d6.jpg)

![](images/0ee75b85a275ebdb1d5e11ca5cf77790d26dfb8a20c24cf99ebdae01227861a0.jpg)

![](images/8191ba587714335bde1626ca0d46da2f9ea8fdd96034a9644abd926df60e84bf.jpg)

![](images/ea86a15e3eddec0c167bb26be368ace3f34a3f08aa2ccd38c04559e614830d71.jpg)

![](images/9ee8fbc1d80c55654f5b97241cae6a9718de81a49c600fd17c24877565567d26.jpg)

![](images/0195a723b4faa3fcb5a1fc750049aaafb2fa1860a3a08549414967b0d292120c.jpg)

![](images/d7bff5716da7c925af0ec7354dcf3fbedc92d96146d16433fdfe7c1eb42644b1.jpg)

![](images/b2721179e3e41ea4dc743e7670e348b1760df7ae99122742e2d2087a6ce3d69a.jpg)

![](images/31b81fc831c2a2f6ef0ccb8e3e937a238ffdf89cc7a5a02fe2c1ad5700918425.jpg)

![](images/77e21347ca451eef12686b8440098efaeac7c9162f7ff3389704ebb0667a85ae.jpg)

![](images/4986e753ab528a8cfc755483d31c26dea98c0cdcbba274ac1457e20a8437168c.jpg)

![](images/066351b8fadfe55d1edefb6120a44bc208efaa96d7597d8979ba04ddb7cd9dc8.jpg)  
Fig. 4. Spectral range (solid line) and mean per-pixel standard deviation (dashed line) of each class as a function of the abundance purity threshold (0.50 - 0.95). Grey bars show the pixel count at each threshold (right axis). The red dashed vertical line marks the 80% purity threshold used for GT endmember extraction. Classes whose spectral range remains near zero across all thresholds (Sub-polar Shrub-Lichen, Sub-polar Grass-Lichen) have no spectrally pure pixels at MODIS 250 m resolution.

gating for spatially varying temporal injection. A Cross-Branch SE module provides inter-expert communication after alignment. The TemporalMambaBranch uses SSM hyperparameters $d _ { \mathrm { m o d e l } } = 6 4 , d _ { \mathrm { s t a t e } } = 8 _ { \cdot }$ , and $n _ { \mathrm { b l o c k s } } = 2$ bidirectional SSM layers; the sequential for-loop scan is used for numerical stability in float32.

## D. Evaluation Metrics

We evaluate performance across three complementary dimensions:

For abundance estimation, we compute the Sum of Absolute Errors (SAE, conventionally reported as MAE in some codebases but mathematically representing the sum of absolute errors over all P classes bounded by 2.0) and Root Mean Square Error (RMSE) between predicted and ground-truth fractional abundances, averaged across all test pixels.

For spectral reconstruction, we calculate the Mean Squared Error (MSE) and Spectral Angle Distance (SAD, in radians) between the physics reconstructed spectrum xˆ and the observed

MODIS input x, measuring the physical consistency of the unmixing solution:

$$
\operatorname { S A D } ( \mathbf { x } , { \hat { \mathbf { x } } } ) = \operatorname { a r c c o s } \left( { \frac { \mathbf { x } ^ { \top } { \hat { \mathbf { x } } } } { \| \mathbf { x } \| \ \| { \hat { \mathbf { x } } } \| } } \right)\tag{22}
$$

For classification evaluation, we report Overall Accuracy (OA), Average Accuracy (AA), and Cohen’s Kappa coefficient (κ), computed by assigning each pixel to the class with the highest predicted abundance and comparing against the 250 m land cover reference map. For detailed class-stratified analysis under imbalanced distributions, we additionally report the perclass F1 score, representing the balanced harmonic mean of precision and recall for individual land cover types.

## E. Baseline Methods

We compare MDF-Mamba against eleven representative unmixing methods spanning traditional optimization, NMF-based, autoencoder-based, CNN-based, spatial-spectral attention-based, transformer-based, and SSM-based frameworks:

1) FCLSU [3]: Fully Constrained Least Squares Unmixing, solved via the standard non-negative least-squares (NNLS) formulation with an appended sum-to-one row, using VCA-initialized endmembers. As a purely spectral method, FCLSU is inherently limited on MODIS broadband data: VCA-extracted endmembers for spectrally similar classes (e.g., different forest or shrubland types) exhibit pairwise cosine similarities exceeding 0.99, rendering the normal equations ill-conditioned and collapsing predictions onto the dominant classes.

2) FE-NMF (Modified SGSNMF) [6]: A Fixed-Endmember Non-negative Matrix Factorization control baseline. In our implementation, endmembers are initialised from VCA and held fixed during multiplicative abundance updates (only the abundance matrix is optimised); allowing joint endmember refinement with near-collinear MODIS signatures causes endmember collapse. Superpixel-guided spatial group sparsity from the canonical SGSNMF is omitted.

3) SUnSAL [5]: Sparse Unmixing via variable Splitting and Augmented Lagrangian, an ADMM-based sparse unmixing method. While canonical SUnSAL is formulated with strict sum-to-one constraints, our evaluation tracks sparsity penalties independently of the strict sum constraint to avoid the condition where the $\ell _ { 1 }$ penalty becomes a constant 1.

4) DeepTrans-HSU [11]: A transformer-based autoencoder architecture with multi-head self-attention for spectral feature learning.

5) PGMSU [9]: A probabilistic generative model for spectral unmixing with Dirichlet-distributed abundances.

6) PPM-Net [14]: A pixel-level probabilistic mixture network with spatial-spectral attention and KL-divergence regularization.

7) SSAF-Net [13]: A spatial-spectral attention fusion network with dual-stream encoding and cross-attention feature integration.

8) SUnCNN [10]: A deep image prior-inspired convolutional architecture for blind spectral unmixing.

9) MiSiCNet [25]: A multi-scale convolutional network with skip connections and endmember estimation head.

10) SRViT [12]: A self-supervised relation-aware vision transformer with pixel-position coding for unmixing.

11) UNMamba [22]: A state space model-based architecture for hyperspectral unmixing leveraging the Mamba selective scan mechanism.

To ensure a fair comparison, all deep learning baselines are trained under an identical supervised protocol: the same training/validation/test splits, the same 64×64 patch extraction, the same VCA-initialized endmembers, and the same L1 abundance loss with ground-truth supervision. The three traditional baselines (FCLSU, FE-NMF, SUnSAL) are applied directly to the test set without training, all initialised from the same VCA endmembers. We note that the proposed MDF Mamba uses GT-derived high purity pixels for its endmember initialization rather than unsupervised VCA, reflecting its supervised design. All models are evaluated using the shared evaluate\_and\_save utility to guarantee consistent metric computation.

## F. Quantitative Results

Tables I and II present the quantitative comparison of MDF-Mamba against eleven baseline methods on the Alberta and Saskatchewan test sets, respectively. Our model achieves the best or second-best performance across all evaluation metrics on both datasets, demonstrating consistent generalisation across geographically distinct MODIS landscapes.

Several key observations can be drawn from Table I:

Regarding abundance estimation, the proposed MDF-Mamba achieves the best MAE of 0.0255 and RMSE of 0.0654, outperforming all baselines including SSAF-Net (MAE: 0.0301, a 15.3% reduction) and substantially surpassing all three traditional baselines - FCLSU (MAE: 0.0633, a 59.7% reduction), SGSNMF, and SUnSAL - as well as DeepTrans-HSU (MAE: 0.0580, a 56.0% reduction). The performance gap between traditional baselines and MDF-Mamba reflects a fundamental limitation of applying spectral-only LMM methods to MODIS broadband data: VCA-extracted endmembers for spectrally similar classes exhibit pairwise cosine similarities exceeding 0.99, rendering the unmixing problem ill-conditioned. FCLSU and SUnSAL both suffer from this ill-conditioning, causing predictions to collapse onto the spectrally dominant classes. SUnSAL additionally compounds this with its $\ell _ { 1 }$ sparsity penalty, which by design activates only a few endmembers per pixel. Among the three traditional methods, SUnSAL achieves the lowest reconstruction MSE (0.0104); SGSNMF, with endmembers held fixed during NMF updates, yields a higher MSE of 0.0118, and its classification accuracy (OA: 0.4734) still falls substantially below all deep learning baselines. The proposed method overcomes these limitations by exploiting temporal phenology as a discriminative signal that separates classes indistinguishable in any single-timestep broadband spectrum, directly validating our first Contribution. This demonstrates the advantage of the multi-task learning framework combining temporal phenology encoding with physics-constrained reconstruction [45, 46].

In terms of spectral reconstruction, MDF-Mamba achieves the lowest reconstruction SAD (0.1843 rad) and MSE (0.0067) among all evaluated methods, outperforming the second-best UNMamba (SAD: 0.2232 rad) by 17.4% and the second-best SUnSAL (MSE: 0.0104) by 35.6%. This dual leadership on both reconstruction metrics demonstrates that the multi-task physics-constrained loss (fourth contribution), combined with the ELMM endmember variability module (third contribution), enables MDF-Mamba to simultaneously preserve spectral shape fidelity and per-band magnitude accuracy across the full 138- band temporal sequence. The SAD metric is particularly sensitive to spectral shape rather than magnitude [47], indicating that the TemporalMambaBranch’s input-adaptive SSM encoding effectively captures the unique phenological signatures within each pixel.

For classification accuracy, MDF-Mamba achieves the best performance across all three classification metrics: overall accuracy (OA: 0.7122), average accuracy (AA: 0.4721), and Kappa (κ: 0.6575), outperforming SUnCNN (OA: 0.6985), MiSiCNet (AA: 0.4451, κ: 0.6333), and SSAF-Net (AA: 0.4450). While the absolute AA values remain relatively low across all models due to the extreme long tail class imbalance inherent to the study region, which severely penalizes minority classes, MDF-Mamba nonetheless leads on AA, the metric most sensitive to this minority class performance. This confirms that the mHC stream-mixing fusion with learnable stream gates (second contribution) successfully preserves perclass discriminative signals during multi-stream integration, enabling improved classification of spatially rare and spectrally ambiguous classes.

TABLE I  
QUANTITATIVE PERFORMANCE COMPARISON ON THE ALBERTA MODIS DATASET. BOLD INDICATES THE BEST RESULT; UNDERLINE INDICATES THE SECOND BEST. ↓ INDICATES LOWER IS BETTER; ↑ INDICATES HIGHER IS BETTER.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="2">Abundance</td><td colspan="2">Reconstruction</td><td colspan="3">Classification</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MSE↓</td><td>SAD (rad)↓</td><td>OA↑</td><td>AA↑</td><td>κ↑</td></tr><tr><td rowspan="3">Traditional</td><td>FCLSU [3]</td><td>0.0633</td><td>0.1530</td><td>0.0106</td><td>0.2445</td><td>0.5256</td><td>0.3290</td><td>0.4256</td></tr><tr><td>SGSNMF [6]</td><td>0.0703</td><td>0.1668</td><td>0.0118</td><td>0.2512</td><td>0.4734</td><td>0.3258</td><td>0.3750</td></tr><tr><td>SUnSAL [5]</td><td>0.0666</td><td>0.1547</td><td>0.0104</td><td>0.2469</td><td>0.5124</td><td>0.2984</td><td>0.4035</td></tr><tr><td rowspan="3">CNN/AE</td><td>PGMSU [9]</td><td>0.0329</td><td>0.0842</td><td>0.0116</td><td>0.2344</td><td>0.6847</td><td>0.4152</td><td>0.6159</td></tr><tr><td>SUnCNN [10]</td><td>0.0309</td><td>0.0823</td><td>0.0141</td><td>0.2678</td><td>0.6985</td><td>0.4308</td><td>0.6328</td></tr><tr><td>MiSiCNet [25]</td><td>0.0320</td><td>0.0828</td><td>0.0141</td><td>0.2695</td><td>0.6982</td><td>0.4451</td><td>0.6333</td></tr><tr><td rowspan="2">CNN+Attention</td><td>SSAF-Net [13]</td><td>0.0301</td><td>0.0755</td><td>0.0274</td><td>0.3646</td><td>0.6957</td><td>0.4450</td><td>0.6290</td></tr><tr><td>PPM-Net [14]</td><td>0.0314</td><td>0.0792</td><td>0.0288</td><td>0.3530</td><td>0.6873</td><td>0.4296</td><td>0.6184</td></tr><tr><td rowspan="2">Transformer</td><td>DeepTrans-HSU [11]</td><td>0.0580</td><td>0.1362</td><td>0.0134</td><td>0.2551</td><td>0.5351</td><td>0.2473</td><td>0.4167</td></tr><tr><td>SRViT [12]</td><td>0.0591</td><td>0.1363</td><td>0.0294</td><td>0.2977</td><td>0.5208</td><td>0.2269</td><td>0.4016</td></tr><tr><td>SSM</td><td>UNMamba [22]</td><td>0.0366</td><td>0.0932</td><td>0.0105</td><td>0.2232</td><td>0.6797</td><td>0.4084</td><td>0.6106</td></tr><tr><td>Ours</td><td>MDF-Mamba</td><td>0.0255</td><td>0.0654</td><td>0.0067</td><td>0.1843</td><td>0.7122</td><td>0.4721</td><td>0.6575</td></tr></table>

Table II reports results on the Saskatchewan test set, which represents a geographically distinct region with different land cover composition and seasonal dynamics compared to Alberta. The Saskatchewan landscape is dominated by agricultural croplands interspersed with wetlands, urban areas, and boreal forest patches [48, 49], creating denser spectral mixtures than the more heterogeneous Alberta terrain.

On Saskatchewan, MDF-Mamba achieves the best performance on five of the seven evaluation metrics, confirming strong cross-regional generalization without domain-specific retraining. It leads on all three classification metrics (OA: 0.7485, AA: 0.4779, κ: 0.6881), on abundance MAE (0.0492) and RMSE (0.1575), and ranks second on both reconstruction MSE (0.0093) and SAD (0.2081 rad) behind SSAF-Net (MSE: 0.0063, SAD: 0.1823 rad). This highly competitive reconstruction performance validates the robustness of the ELMM-based endmember variability module and the multi-task learning framework under domain shift: the physics-constrained reconstruction head maintains physically meaningful spectral fidelity even when the endmember dictionary, learned on Alberta, encounters Saskatchewan’s distinct cropland-dominated spectral mixtures [50, 51].

For classification, MDF-Mamba achieves the best performance across all three metrics: OA (0.7485), AA (0.4779), and κ (0.6881), outperforming all baselines including SUnCNN (OA: 0.7176), UNMamba (AA: 0.4330, κ: 0.6499), and MiSiC

Net (AA: 0.4175). The fact that MDF-Mamba leads on AA - the metric most sensitive to minority class performance - validates Contribution 1 (asymmetric disentangling) and Contribution 2 (mHC fusion): the multi-branch architecture successfully transfers its per-channel phenological representations to a geographically distinct landscape without overfitting to Albertaspecific class distributions, while the manifold-constrained fusion preserves minority-class signals that single-stream architectures suppress. MDF-Mamba maintains a substantial lead over all traditional baselines (>34.3% OA improvement) and the Transformer-based SRViT (>44.0% OA improvement), demonstrating consistent superiority across geographically diverse MODIS landscapes.

The higher OA and κ on Saskatchewan (0.7485 and 0.6881) than on Alberta (0.7122 and 0.6575) should not be read as better performance than within the training province. OA weights every pixel equally and is therefore dominated by the most frequent classes. Cropland and Water, the two classes that transfer most reliably (F1 of 0.934 and 0.816, Table X), together account for 38.6% of the Saskatchewan abundance (26.3% and 12.3%), compared with 21.4% in Alberta (17.2% and 4.2%). Conversely, Mixed Forest, one of the most challenging classes for abundance estimation, falls from 15.1% of the Alberta abundance to 3.4% in Saskatchewan. The Saskatchewan class composition therefore favors easily separable classes, which raises OA and κ even though the abundance error increases (MAE: 0.0255 to 0.0492).

Saskatchewan is evaluated as a zero-shot cross-provincial transfer: the model is trained exclusively on Alberta and applied to the full Saskatchewan MODIS scene without retraining or fine-tuning, using the Alberta global band normalization statistics. The two provinces differ markedly in class composition: Cropland rises from 17.2% of the Alberta training abundance to 26.3% in Saskatchewan (40.0% of its pure pixels), Water from 4.2% to 12.3%, while Mixed Forest falls from 15.1% to 3.4%.

TABLE II  
QUANTITATIVE PERFORMANCE COMPARISON ON THE SASKATCHEWAN MODIS DATASET. BOLD INDICATES THE BEST RESULT; UNDERLINE INDICATES THE SECOND BEST. ↓ INDICATES LOWER IS BETTER; ↑ INDICATES HIGHER IS BETTER.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="2">Abundance</td><td colspan="2">Reconstruction</td><td colspan="3">Classification</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MSE↓</td><td>SAD (rad)↓</td><td>OA↑</td><td>AA↑</td><td>κ↑</td></tr><tr><td rowspan="3">Traditional</td><td>FCLSU [3]</td><td>0.0729</td><td>0.2217</td><td>0.0126</td><td>0.2560</td><td>0.5572</td><td>0.2729</td><td>0.4495</td></tr><tr><td>SGSNMF [6]</td><td>0.0691</td><td>0.2481</td><td>0.0312</td><td>0.3498</td><td>0.5229</td><td>0.1857</td><td>0.3822</td></tr><tr><td>SUnSAL [5]</td><td>0.0807</td><td>0.2359</td><td>0.0132</td><td>0.2533</td><td>0.4607</td><td>0.2417</td><td>0.3476</td></tr><tr><td rowspan="3">CNN/AE</td><td>PGMSU [9]</td><td>0.0546</td><td>0.1748</td><td>0.0143</td><td>0.2617</td><td>0.6986</td><td>0.3832</td><td>0.6286</td></tr><tr><td>SUnCNN [10]</td><td>0.0509</td><td>0.1685</td><td>0.0162</td><td>0.2836</td><td>0.7176</td><td>0.3874</td><td>0.6493</td></tr><tr><td>MiSiCNet [25]</td><td>0.0532</td><td>0.1685</td><td>0.0164</td><td>0.2858</td><td>0.7121</td><td>0.4175</td><td>0.6450</td></tr><tr><td rowspan="2">CNN+Attention</td><td>SSAF-Net [13]</td><td>0.0544</td><td>0.1766</td><td>0.0063</td><td>0.1823</td><td>0.6829</td><td>0.3995</td><td>0.6116</td></tr><tr><td>PPM-Net [14]</td><td>0.0542</td><td>0.1701</td><td>0.0226</td><td>0.3506</td><td>0.7064</td><td>0.4199</td><td>0.6377</td></tr><tr><td rowspan="2">Transformer</td><td>DeepTrans-HSU [11]</td><td>0.0753</td><td>0.2097</td><td>0.0162</td><td>0.2837</td><td>0.5564</td><td>0.2231</td><td>0.4346</td></tr><tr><td>SRViT [12]</td><td>0.0760</td><td>0.2259</td><td>0.0198</td><td>0.3082</td><td>0.5197</td><td>0.2033</td><td>0.3850</td></tr><tr><td>SSM</td><td>UNMamba [22]</td><td>0.0539</td><td>0.1669</td><td>0.0133</td><td>0.2492</td><td>0.7167</td><td>0.4330</td><td>0.6499</td></tr><tr><td>Ours</td><td>MDF-Mamba</td><td>0.0492</td><td>0.1575</td><td>0.0093</td><td>0.2081</td><td>0.7485</td><td>0.4779</td><td>0.6881</td></tr></table>

Full-scene maps are produced with non-overlapping 64 × 64 tiles; the main comparison (Table II) uses all valid pixels against the 250 m reference map, with each pixel’s reference abundance set to a one-hot vector of its mapped class, and the filtered comparison (Table IV) uses every third valid patch, restricted to pixels with dominant-class purity ≥ 80% and evaluated against the fractional abundances. The abundance errors in Table II are therefore not directly comparable with those in Table I, which are computed against fractional references.

## G. Quantitative Results on Filtered High-Purity Datasets

To further evaluate the robustness of the proposed model on highly confident ground-truth regions, we additionally test all methods on filtered subsets of the Alberta and Saskatchewan datasets containing only pixels with dominant class purity ≥ 80%. Tables III and IV present the quantitative comparison on these high-purity subsets.

On the Saskatchewan filtered dataset (Table IV), MDF-Mamba achieves the best performance on all seven evaluation metrics. It attains the lowest abundance MAE (0.0250) and RMSE (0.0958), ahead of UNMamba (MAE: 0.0258, RMSE: 0.0970), and the best reconstruction fidelity (MSE: 0.0119, SAD: 0.2238 rad), reducing MSE by 29.2% and SAD by 15.5% relative to the second-best DeepTrans-HSU (MSE: 0.0168, SAD: 0.2649 rad). For classification, MDF-Mamba leads on OA (0.8972), AA (0.6900), and κ (0.8633), exceeding UNMamba (OA: 0.8958, κ: 0.8618) and SSAF-Net (AA: 0.6790).

Unlike the full-scene results, MDF-Mamba’s OA and κ on the filtered Saskatchewan subset (0.8972 and 0.8633) are lower than on the filtered Alberta subset (0.9340 and 0.9167). This reversal is not caused by a more balanced class distribution after filtering. The purity filter retains proportionally more pure Cropland pixels, so Cropland rises from 26.3% of the Saskatchewan abundance to 40.0% of its filtered pixels, and the two filtered subsets are similarly imbalanced (normalized

Shannon entropy of the class distribution of approximately 0.70 in both). Filtering instead removes the mixed pixels that limit accuracy in both provinces. Although the filtered Saskatchewan subset still contains more Cropland than the filtered Alberta subset (40.0% versus approximately 28%), its OA is lower. The remaining gap of 3.7 OA points therefore reflects crossprovincial domain shift: the filtered Alberta subset is evaluated within the training province, whereas Saskatchewan is a zeroshot transfer with different phenology and spectral statistics.

## H. Spectral Reconstruction Analysis

To evaluate the multi-task physics-constrained reconstruction, Fig. 5 compares the original MODIS spectra against the physics constrained reconstructions produced by the multi task loss. The proposed model achieves an overall pixel level reconstruction RMSE of $0 . 0 8 1 9 ~ ( = ~ \sqrt { 0 . 0 0 6 7 } )$ and a SAD of 0.1843 rad, consistent with the full Alberta metrics reported in Table I. The patch mean reconstructed spectra (Fig. 5a) tightly follow the original signals across all six optical bands. In addition, the per-band RMSE analysis (Fig. 5b) reveals that the model maintains consistent reconstruction fidelity across the entire temporal sequence, avoiding signal degradation in challenging transition months.

## I. High-Resolution Spatial Comparison

To complement the global quantitative metrics and provide visual evidence of mHC fusion for spatial coherence and endmember variability for fractional gradient precision, Figs. 6, 7, and 8 present comprehensive full-scene and zoomedin comparisons of the spatial classification and fractional abundance outputs of the proposed MDF-Mamba against all baselines on both the Alberta and Saskatchewan datasets.

Fig. 6 provides a unified classification comparison at two levels of granularity. The top two rows display the complete Alberta classification maps for all 13 models (Ground Truth,

TABLE III  
QUANTITATIVE PERFORMANCE COMPARISON ON THE ALBERTA FILTERED DATASET (80% PURITY). BOLD INDICATES THE BEST RESULT; UNDERLINE INDICATES THE SECOND BEST. ↓ INDICATES LOWER IS BETTER; ↑ INDICATES HIGHER IS BETTER.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="2">Abundance</td><td colspan="2">Reconstruction</td><td colspan="3">Classification</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MSE↓</td><td>SAD (rad)↓</td><td>OA↑</td><td>AA↑</td><td>κ↑</td></tr><tr><td rowspan="3">Traditional</td><td>FCLSU [3]</td><td>0.7291</td><td>0.7526</td><td>0.0231</td><td>0.3375</td><td>0.5874</td><td>0.2166</td><td>0.4315</td></tr><tr><td>SGSNMF [6]</td><td>0.6591</td><td>0.7217</td><td>0.0317</td><td>0.3540</td><td>0.6466</td><td>0.2887</td><td>0.5368</td></tr><tr><td>SUnSAL [5]</td><td>0.7180</td><td>0.7464</td><td>0.0242</td><td>0.3364</td><td>0.5901</td><td>0.2304</td><td>0.4367</td></tr><tr><td rowspan="3">CNN/AE</td><td>PGMSU [9]</td><td>0.3726</td><td>0.4360</td><td>0.0145</td><td>0.2562</td><td>0.8984</td><td>0.5979</td><td>0.8724</td></tr><tr><td>SUnCNN [10]</td><td>0.3001</td><td>0.3793</td><td>0.0183</td><td>0.2963</td><td>0.9134</td><td>0.6457</td><td>0.8913</td></tr><tr><td>MiSiCNet [25]</td><td>0.2965</td><td>0.3741</td><td>0.0185</td><td>0.2972</td><td>0.9132</td><td>0.6484</td><td>0.8911</td></tr><tr><td rowspan="2">CNN+Attention</td><td>SSAF-Net [13]</td><td>0.2911</td><td>0.3679</td><td>0.1271</td><td>0.7340</td><td>0.9115</td><td>0.6553</td><td>0.8890</td></tr><tr><td>PPM-Net [14]</td><td>0.3067</td><td>0.3848</td><td>0.1605</td><td>0.8003</td><td>0.9034</td><td>0.6349</td><td>0.8786</td></tr><tr><td rowspan="2">Transformer</td><td>DeepTrans-HSU [11]</td><td>0.3915</td><td>0.4549</td><td>0.0117</td><td>0.1996</td><td>0.8237</td><td>0.5079</td><td>0.7734</td></tr><tr><td>SRViT [12]</td><td>0.6333</td><td>0.6793</td><td>0.0192</td><td>0.3038</td><td>0.7166</td><td>0.3168</td><td>0.6323</td></tr><tr><td>SSM</td><td>UNMamba [22]</td><td>0.3295</td><td>0.4018</td><td>0.0128</td><td>0.2485</td><td>0.8945</td><td>0.6022</td><td>0.8680</td></tr><tr><td>Ours</td><td>MDF-Mamba</td><td>0.2491</td><td>0.3229</td><td>0.0083</td><td>0.2005</td><td>0.9340</td><td>0.6480</td><td>0.9167</td></tr></table>

TABLE IV

QUANTITATIVE PERFORMANCE COMPARISON ON THE SASKATCHEWAN FILTERED DATASET (80% PURITY). BOLD INDICATES THE BEST RESULT; UNDERLINE INDICATES THE SECOND BEST. ↓ INDICATES LOWER IS BETTER; ↑ INDICATES HIGHER IS BETTER.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="2">Abundance</td><td colspan="2">Reconstruction</td><td colspan="3">Classification</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MSE↓</td><td>SAD (rad)↓</td><td>OA↑</td><td>AA↑</td><td>κ↑</td></tr><tr><td rowspan="3">Traditional</td><td>FCLSU [3]</td><td>0.0552</td><td>0.2006</td><td>0.0270</td><td>0.3438</td><td>0.6503</td><td>0.2004</td><td>0.4774</td></tr><tr><td>SGSNMF [6]</td><td>0.0567</td><td>0.2113</td><td>0.0342</td><td>0.3576</td><td>0.6303</td><td>0.2353</td><td>0.4819</td></tr><tr><td>SUnSAL [5]</td><td>0.0548</td><td>0.2052</td><td>0.0273</td><td>0.3429</td><td>0.6493</td><td>0.2008</td><td>0.4754</td></tr><tr><td rowspan="3">CNN/AE</td><td>PGMSU [9]</td><td>0.0312</td><td>0.1176</td><td>0.0191</td><td>0.2789</td><td>0.8539</td><td>0.5477</td><td>0.8080</td></tr><tr><td>SUnCNN [10]</td><td>0.0296</td><td>0.1135</td><td>0.0227</td><td>0.3066</td><td>0.8615</td><td>0.5874</td><td>0.8175</td></tr><tr><td>MiSiCNet [25]</td><td>0.0289</td><td>0.1102</td><td>0.0233</td><td>0.3109</td><td>0.8653</td><td>0.6217</td><td>0.8232</td></tr><tr><td rowspan="2">CNN+Attention</td><td>SSAF-Net [13]</td><td>0.0320</td><td>0.1110</td><td>0.1277</td><td>0.7110</td><td>0.8608</td><td>0.6790</td><td>0.8182</td></tr><tr><td>PPM-Net [14]</td><td>0.0299</td><td>0.1116</td><td>0.1579</td><td>0.7896</td><td>0.8593</td><td>0.6183</td><td>0.8146</td></tr><tr><td rowspan="2">Transformer</td><td>DeepTrans-HSU [11]</td><td>0.0328</td><td>0.1210</td><td>0.0168</td><td>0.2649</td><td>0.8397</td><td>0.5349</td><td>0.7881</td></tr><tr><td>SRViT [12]</td><td>0.0350</td><td>0.1368</td><td>0.0205</td><td>0.2782</td><td>0.8085</td><td>0.4774</td><td>0.7499</td></tr><tr><td>SSM</td><td>UNMamba [22]</td><td>0.0258</td><td>0.0970</td><td>0.1715</td><td>1.5556</td><td>0.8958</td><td>0.6597</td><td>0.8618</td></tr><tr><td>Ours</td><td>MDF-Mamba</td><td>0.0250</td><td>0.0958</td><td>0.0119</td><td>0.2238</td><td>0.8972</td><td>0.6900</td><td>0.8633</td></tr></table>

MDF-Mamba, and 11 baselines), rendered with a standardized 14-class colormap that is consistently maintained across all figures in this manuscript to enable direct full-scene assessment and intuitive cross-referencing. White square locator boxes on the Ground Truth map identify twelve representative $6 0 \times 6 0$ pixel sub-regions (a-l), each centered on a different land cover class: (a) Barren, (b) Broadleaf, (c) Grassland, (d) Urban, (e) Wetland, (f) Taiga, (g) Snow, (h) Cropland, (i) Mixed Forest, (j) Water, (k) Needleleaf, and (l) Temp. Shrub. These regions were automatically selected by maximizing the holistic classification error gap between MDF-Mamba and the baseline average, while filtering for structural complexity to avoid trivially homogeneous patches. As such, these regions illustrate cases of maximal advantage and should be interpreted alongside the aggregate, unbiased metrics in Tables I-II, which reflect typical case performance. The bottom section presents a 13×12 zoomed-in comparison grid for all 11 baselines alongside the Ground Truth and the proposed model, enabling detailed pixel-level inspection of class boundary fidelity within each highlighted region.

Figs. 7 and 8 present unified per-province abundance comparison figures for Alberta and Saskatchewan, respectively. Each figure adopts a two-level layout. The top section displays full-scale abundance maps for all valid land cover classes produced by the proposed MDF-Mamba model, rendered using the turbo colormap (0 to 1 fractional range). The maps are clipped to the true geographic boundaries of each province using official Statistics Canada shapefiles, providing an accurate spatial context. White bounding boxes on each full map mark the 60 × 60 pixel sub-regions selected for detailed comparison, labeled with lowercase letters $( \mathbf { a } , \mathbf { b } , \mathbf { c } , \ldots )$ that correspond to the rows of the bottom grid. The sub-regions were selected to include a diverse set of representative patches, ranging from median performing examples to challenging transition zones, without GT-dependent masking. These regions provide a balanced qualitative view to complement the aggregate metrics in Tables I-II.

(a) Patch-mean Original vs. Reconstructed Spectra  
![](images/886c5b9baa8e06f152f47b20340d8e370dd8d81dee7a421db05c43c5a54c203d.jpg)  
MODIS Spectral-Temporal Features (23 timestamps per feature)

(b) Per-feature RMSE over time  
![](images/b8e9ef6d58b6ddc6381ea7d21af209ae05541f96c702bcbdb2ea79e09de18b58.jpg)

(c) Original vs. Reconstructed reflectance  
![](images/ee3f133340415dce43f918e9cd1dccf493d101b84a0c74b38263e5fd58201022.jpg)

(d) Per-class reconstruction SAD  
![](images/db1d00a40dc7f1c1d92e171cf8cf1a07a7d21f3e7e70f3df2d72648ad2b03ed4.jpg)  
Fig. 5. Spectral reconstruction evaluation on the Alberta test set. (a) Patch-mean original versus reconstructed spectra for dominant classes across the 138-band spectral-temporal sequence. (b) Per-feature reconstruction RMSE over the 23-step temporal index. (c) Pixel-level original versus reconstructed reflectance scatter, with overall MSE, RMSE, SAD, and $R ^ { 2 }$ annotated. (d) Per-class reconstruction SAD.

The bottom section of each figure presents a comprehensive zoomed-in comparison grid where each row corresponds to one class specific subregion and each column corresponds to one of the 13 evaluated models, labeled by uppercase letters (A through M) to maintain compact column headers. The model-to-letter mapping is: A = GT, B = MDF-Mamba (Ours), C = UNMamba, D = SSAF-Net, E = PPM-Net, F = MiSiCNet, G = SUnCNN, H = FCLSU, I = PGMSU, J = SRViT, K = DeepTrans-HSU, L = SUnSAL, and M = SGSNMF. This compact grid enables direct pixel-level inspection of fractional abundance fidelity across all baselines simultaneously within

each highlighted region.

The visual analysis yields two critical insights into the model’s localized performance:

(1) Structural Integrity in Classification: As shown in Fig. 6, the proposed MDF-Mamba maintains tight spatial coherence and sharp boundary definition across twelve structurally complex transition zones (columns a-l). In the Barren region (column a), baselines misclassify exposed terrain as Needleleaf or Wetland, whereas MDF-Mamba preserves the correct class identity. In the Broadleaf (column b) and Mixed Forest (column i) regions, competing methods suffer from severe inter-forest confusion, blurring the boundaries between deciduous and coniferous canopies, while the proposed model maintains crisp delineations. The Grassland patch (column c) reveals that baselines tend to fragment the homogeneous grassland into scattered Cropland or Shrub pixels, whereas MDF-Mamba produces a spatially consistent classification. In the Urban region (column d), the model accurately preserves the compact anthropogenic footprint without spatial bleeding into surrounding Cropland. The Wetland (column e) and Taiga (column f) patches demonstrate the model’s ability to separate spectrally similar forest-wetland transitions, where baselines produce noisy, salt-and-pepper misclassifications. The Snow (column g) and Cropland (column h) regions confirm that MDF Mamba correctly delineates seasonal and agricultural classes that baselines frequently confuse with Barren or Grassland cover. Finally, in the highly challenging Water (column j), Needleleaf (column k), and Temp. Shrub (column l) zones, baselines struggle significantly with class boundary adherence, often generating large artifact clusters, whereas the proposed model accurately captures the exact geometric footprint of the ground truth. This localized boundary superiority confirms the efficacy of the manifold-constrained hyper-connection fusion in preserving class identity across complex spatial gradients.

![](images/90faa8f47a57af581a0d06cb2e5535b3478d7116b91d94331d157e86d3bbd0da.jpg)  
Fig. 6. Unified classification comparison on the Alberta dataset. Top two rows: Complete classification maps for all 13 models with a standardized 14-clas colormap consistently maintained across the manuscript; white squares on the Ground Truth map mark twelve 60 × 60 sub-regions (a-l) centered on distinct land cover classes. Bottom grid: Zoomed-in 13 × 12 comparison of all 13 models within each highlighted region. Regions were selected to represent a diverse set of median-performing and challenging cases.

![](images/e5b36d6a960351be238f7fd64022db5df7d87a5097a1d90b1e5f4e1f2122b6e2.jpg)  
Fig. 7. Unified abundance comparison on the Alberta dataset. Top: Full-scene MDF-Mamba abundance maps for all valid classes clipped to the true Alberta provincial boundary; white boxes mark zoomed sub-regions (a–k). Bottom: Zoomed-in abundance comparison grid across all 13 models. Colum letters: $\dot { \Delta } = \mathrm { G T , ~ B = M D F \mathrm { \cdot } M a m b a , ~ C = U N M a m b a , ~ D = S S A \dot { F } \mathrm { \cdot } N e t , ~ E = P P M \mathrm { \cdot } N e t , ~ F = M i S t C N e t , ~ G = S U n C N N , ~ H = F C L S U , ~ I = P G M S U , ~ J = S R V i T , }$ K = DeepTrans, L = SUnSAL, M = SGSNMF. Note the near-zero or spatially diffuse predictions by traditional baselines (H, L, M) and transformer-based SRViT (J), contrasting with the sharp, GT-consistent gradients of MDF-Mamba (B).

![](images/31e7e7186e63e38076097ce309ee8b80b0c64dadd71c16cdbd43424bb952836f.jpg)  
Fig. 8. Unified abundance comparison on the Saskatchewan dataset. The layout mirrors Fig. 7, with full-scene maps clipped to the Saskatchewan provincia boundary. The same model-to-letter mapping applies. Saskatchewan’s cropland-dominated landscape presents denser spectral mixtures; baseline methods (particularly H, L, M) produce visibly degraded or entirely empty abundance patches for minority classes, whereas MDF-Mamba (B) maintains accurat fractional gradients that closely follow the GT (A) across all sub-regions.

(2) Precision in Fractional Gradients: The unified abundance comparisons (Figs. 7 and 8) evaluate continuous subpixel abundance estimation across all valid classes by isolating diagnostically selected sub-regions and comparing all 13 mod els simultaneously. The zoomed grids reveal a clear hierarchy: MDF-Mamba (column B) consistently produces abundance patches that closely match the GT (column A) in both spatial structure and fractional magnitude. In contrast, traditional baselines such as FCLSU (H), SUnSAL (L), and SGSNMF (M) frequently produce near-zero or spatially uniform predictions that entirely miss the target class distribution, a consequence of the ill-conditioned VCA endmembers and the absence of temporal phenology encoding. The transformer-based SRViT (J) and DeepTrans-HSU (K) similarly exhibit severe fractional diffusion due to their inability to disentangle spectrally similar classes. Among the competitive deep learning baselines, UNMamba (C), SSAF-Net (D), and SUnCNN (G) produce recognizable abundance patterns but with notably coarser spatial gradients and systematic underestimation of high-abundance regions compared to MDF-Mamba. The cross-province comparison further demonstrates that these observations are not dataset specific: on Saskatchewan (Fig. 8), MDF-Mamba maintains sharp fractional boundaries for cropland-dominant mixtures where baselines exhibit increased spatial bleeding due to the denser spectral mixtures. This confirms that the pixel-wise endmember variability modeling and mHC fusion successfully adapt to localized sub-pixel mixtures across geographically distinct landscapes, directly mirroring the model’s global metric superiority.

## J. Per-Class Abundance Analysis

To assess the per-class impact of channel disentangling endmember variability), Fig. 9 presents a comprehensive spatial error evaluation. First, Fig. 9(a) shows the geographic distribution of abundance MAE across the Alberta test patches, highlighting regional error clusters. Second, the sorted per-class MAE in Fig. 9(b) demonstrates that classes with distinctive spectral signatures, such as Water (MAE: 0.0152) and Urban (MAE: 0.0158), achieve strong abundance estimation accuracy. In contrast, structurally complex classes such as Needleleaf forest (MAE: 0.0786) and Wetland (MAE: 0.0555) exhibit higher errors. This is corroborated by the failure analysis in Fig. 9(c), which isolates the top-5 worst and best performing patches. As illustrated, the most accurate predictions occur in spatially homogeneous regions like Water and Temperate Grassland, whereas the largest pixel-wise errors consistently occur in highly mixed transition zones involving dense vegetation (e.g., Needleleaf, Wetland, and Mixed Forest) where spectral overlap is most severe.

## K. Temporal Phenology as a Disentangling Cue

This subsection provides qualitative evidence for the TemporalMambaBranch’s ability to encode phenological dynamics that drive accurate abundance estimation. The per branch learned endmember temporal profiles in Fig. 3 show that the seasonal trajectories are strongly class dependent: for instance, the sharp summer green up in the cropland NDVI trajectory (peaking in July) distinctly separates it from the gradual, plateaued response of temperate needleleaf forest. By scanning these 23-step trajectories across all six spectral channels jointly, the TemporalMambaBranch exploits exactly these cues, contributing to the accurate spatial disentanglement of spectrally similar classes reflected in the per-class errors of Fig. 9(b).

## L. Endmember Analysis and Shift Justification

This subsection evaluates the Extended Linear Mixing Model with pixel wise learnable spectral variants and GT-initialized endmember embeddings.

Fig. 10 presents a comprehensive class-by-feature matrix: each row is one land cover class, columns 1-6 correspond to the six spectral features (NDVI, EVI, Red, NIR, Blue, MIR), and column 7 presents the per-class GT-vs-learned endmember scatter (colored by branch) against the 1:1 identity line, annotated with Lin’s concordance correlation coefficient (CCC) [52], RMSE, and SAD. This layout enables direct inspection of how the final learned mean endmember (solid colored line) relates to the GT reference (solid black) and the geographic initialization (dashed line) for every classbranch combination, while simultaneously quantifying the overall spectral fidelity. As established in the data exploration (Fig. 3), the six spectral branches provide complementary class separability; the correlation column directly measures how well the model learns to exploit these distinctions. The R=10 variant spread (shaded) illustrates the pixel-adaptive flexibility of the ELMM.

The figure reveals two distinct behaviors linked to data quality. For spectrally rich classes with abundant pure pixels (Cropland, Needleleaf Forest, Temperate Grassland, and Mixed Forest) the learned curve closely tracks the GT and the displacement from the initialization (dashed) is small, indicating successful endmember convergence [7]. In contrast, classes whose learned curve lies far from the GT reference (Shrubland, Snow/Ice, Urban, Broadleaf Forest, and Sub-polar Taiga) reflect the absence of spectrally pure pixels at 250 m MODIS resolution. The model is forced to relocate these embeddings to remain spectrally distinct under the multi-task reconstruction objective [43]; Sub-polar Taiga’s abundance estimation behavior is examined separately in Section IV-R. The two sub-polar lichen-moss classes are omitted from Fig. 10, consistent with their exclusion from the final abundance and classification output (Section IV-A). The displacement between the learned and GT endmembers therefore provides a visual diagnostic of data-imposed limits: the larger the displacement, the lower the CCC shown in the rightmost column of Fig. 10.

![](images/d2c0d508d65fde8adfac9889fec9310ef6dc59bd8741edd7095a334cde645348.jpg)  
Fig. 9. Spatial error distribution and failure analysis on the Alberta test set. (a) Spatial distribution of abundance MAE over the 567 test patches, with the top-10% highest error patches highlighted. (b) Sorted per-class MAE. (c) Failure analysis detailing the specific dominant classes in the top-5 worst and best performing patches.

To explicitly visualize the intra-class spectral variability captured by the Extended Linear Mixing Model (ELMM), Fig. 11 presents, for four representative classes (Cropland, Needleleaf Forest, Mixed Forests, and Grassland), the heatmaps of the $R = 1 0$ learned endmember variants alongside their average pixel-wise variant mixing weights.

The rightmost column of Fig. 10 quantifies the endmember learning quality via per-class scatter plots of the GT endmember versus the learned mean endmember across all $L ~ = ~ 1 3 8$ spectral-temporal bands, colored by spectral branch. Agreement is quantified with Lin’s concordance correlation coefficient [52], $\mathrm { C \bar { C } C } = 2 \sigma _ { g l } / \big ( \sigma _ { g } ^ { 2 } + \sigma _ { l } ^ { 2 } + ( \mu _ { g } - \mu _ { l } ) ^ { 2 } \big )$ , where $\mu , \sigma ^ { 2 } .$ , and $\sigma _ { g l }$ are the means, variances, and covariance of the GT (g) and learned (l) endmembers over the 138 bands. Unlike the squared Pearson correlation, which is insensitive to offset and scale, CCC equals 1 only when all points fall on the 1:1 line and decreases with any systematic bias. Together with RMSE and SAD, it gives a direct quantitative measure of the displacement seen in the temporal profiles: classes with small displacement between the learned and GT curves yield high CCC, while classes with large displacement yield low CCC.

The results confirm a clear stratification. Classes with rich and spatially abundant pure pixels achieve the highest concordance: Needleleaf Forest $( \mathrm { C C C } = 0 . 7 7 0 , \mathrm { S A D } = 1 5 . 4 ^ { \circ } )$ and Cropland $( \mathrm { C C C } = 0 . 7 4 0 , \mathrm { S A D } = 1 6 . 1 ^ { \circ } )$ , followed by Temperate Grassland $( \mathrm { C C C } = 0 . 4 1 8 , \mathrm { S A D } = 2 0 . 4 ^ { \circ } )$ and Mixed Forest $( \mathrm { C C C } = 0 . 3 2 9 , \mathrm { S A D } = 2 7 . 2 ^ { \circ } )$ . For these classes, the scatter points run close to and parallel with the 1:1 identity line, indicating that the multi-task learning framework steered the endmember embeddings toward physically meaningful spectral temporal shapes despite the absence of perfectly pure MODIS pixels; the remaining gap to CCC = 1 is mainly a systematic downward offset of the learned reflectance relative to the GT.

Lower concordance is obtained for Wetland $\begin{array} { r } { ( \mathbf { C C C } = 0 . 1 5 8 , } \end{array}$ $\mathrm { S A D } = 2 9 . 9 ^ { \circ } )$ and Water $( \mathrm { C C C } = 0 . 1 2 6 , \mathrm { S A D } = 2 4 . 7 ^ { \circ } )$ where the scatter remains well-correlated but exhibits larger band-level offsets as the anchor loss ${ \mathcal { L } } _ { \mathrm { a n c } }$ relaxes (via cosine annealing) and the reconstruction objective adapts the endmem bers to the data.

The weakest endmember correspondence is observed for Barren Lands $( \mathrm { C C C } = 0 . 0 3 2 , \mathrm { S A D } = 2 5 . 4 ^ { \circ } )$ , Shrubland (CCC $= 0 . 0 3 1 , \mathrm { \Delta S A D = 2 7 . 4 ^ { \circ } ) }$ , Snow/Ice (CCC = 0.014, SAD $= 2 4 . 6 ^ { \circ } )$ , Urban $( \mathrm { C C C } = 0 . 0 0 8 , \mathrm { S A D } = 2 6 . 5 ^ { \circ } )$ , Broadleaf Forest $( \mathrm { C C C } = 0 . 0 0 6 , \mathrm { S A D } = 3 3 . 9 ^ { \circ } )$ , and, most acutely, Taiga Forest $( \mathrm { C C C } = 0 . 0 0 0 , \mathrm { S A D } = 3 0 . 0 ^ { \circ } )$ . For these classes, the GT endmembers have a narrow spectral range, indicating that no spectrally distinguishable pure pixels exist in the dataset for these biomes at the 250 m MODIS resolution [38, 53]. Despite this data-imposed limitation, the multi-task reconstruction loss pushes the learned endmembers for these classes to occupy distinct regions of the reflectance space. Sub-polar Taiga’s abundance estimation behavior is examined separately in Section IV-R.

The most challenging classes for abundance estimation are needleleaf forest (MAE: 0.0786), mixed forest (MAE: 0.0559), wetland (MAE: 0.0555), and grassland (MAE: 0.0504). These classes share highly overlapping spectral-temporal signatures due to similar vegetation phenology [7, 54], making their sub-pixel proportions inherently difficult to disentangle [43]. The multi-branch architecture mitigates this by allowing the NIR and MIR experts to learn moisture-specific features that discriminate wetland from grassland, while the NDVI expert captures the distinct green-up timing that separates needleleaf from broadleaf phenology, and the TemporalMambaBranch encodes the full seasonal trajectory across all six channels simultaneously. Importantly, Needleleaf Forest has the highest endmember concordance of all classes $( \mathbf { C C C } = 0 . 7 7 0 )$ , which indicates that its abundance estimation difficulty stems from scene-level spectral ambiguity rather than from failure to learn a correct endmember signature. For Mixed Forest $( \mathbf { C C } = 0 . 3 2 9 )$ Grassland $\displaystyle ( \mathbf { C C C } = 0 . 4 1 8 )$ , and Wetland $\displaystyle ( \mathbf { C C C } = 0 . 1 5 8 )$ the lower concordance suggests that endmember offset may contribute alongside this ambiguity.

![](images/ee098123bd6c0041dc4d95e07b30c71f9d979e40a5720047a2dabcebf54370eb.jpg)  
Fig. 10. Learned temporal endmember profiles and concordance analysis for the P=12 active land cover classes, ordered from top to bottom by lowest RMSE (best prediction) to highest RMSE (worst prediction); the two sub-polar lichen-moss classes are omitted (Section IV-A). Columns 1 - 6: Per-branch temporal profiles (NDVI, EVI, Red, NIR, Blue, MIR) showing R=10 variant spread (shaded), learned mean endmember (solid colored line), GT reference (solid black), and geographic initialization (dashed). Column 7 (Corr.): Per-class scatter of GT vs. learned mean endmember across all L=138 spectral-temporal bands, colored by spectral branch, with the 1:1 identity line (dashed) and annotated Lin’s concordance correlation coefficient (CCC) [52], RMSE, and SAD. CCC measures agreement with the 1:1 line and penalizes both offset and scale differences. Classes whose GT reference is nearly flat (Shrubland, Snow/Ice, Urban, Broadleaf Forest, Sub-polar Taiga) produce the highest RMSE and lowest CCC, consistent with poor endmember convergence being data-imposed rather than architectural. Y-axis auto-scaled per row; X-axis spans the full 2010 calendar year (23 composites).

![](images/d89a4dec3e7be2d200bb00fe5e77113b2f3a7554a52df69e5f92602e6b872a7a.jpg)  
Fig. 11. Intra-class spectral variability captured by the ELMM. Each row corresponds to one of the four largest land cover classes (Needleleaf Forest, Cropland, Mixed Forest, Temperate Grassland); within each row, the left heatmap shows the R=10 learned endmember variants across the 138 spectral temporal bands (colour encodes reflectance) and the right bar chart shows the corresponding average pixel-wise mixing weight of each variant, illustrating how the variability head adapts the effective endmember used at each pixel.

## M. Feature Space Disentanglement

To visually confirm the quality of the learned representations and demonstrate how well the model disentangles pure endmembers from highly mixed pixels, we extract the fused features $( \mathbf { F } _ { \mathrm { f u s e } } , d _ { \mathrm { f u s e } } { = } 6 4 )$ from the MDF-Mamba model before and after training.

Fig. 12 visualizes the t-SNE projections of these features. Before training (Fig. 12a), the features are mixed and scattered across the manifold, showing no inherent clustering by land cover class. After training, the model projects pure endmembers into well-separated clusters in the latent space, consistent with the multi-branch architecture and mHC fusion layers untangling the complex, non-linear physical mixtures into distinct conceptual manifolds prior to final abundance estimation.

## N. Ablation Study

To validate the four contributions of MDF-Mamba, we conduct a two-part ablation on the Alberta test set. Part I is a component lesion study that disables one module of the single trained model at inference, removing all run to run retraining variance; Part II retrains the model with one design decision changed at a time.

![](images/27b1b5d2f3400935b3fec9e2fab81b313c3c5df23fd14ff3a80fe18650cc943c.jpg)  
Fig. 12. t-SNE visualization of the fused feature space $\left( \mathbf { F } _ { \mathrm { f u s e } } \right)$ colored by land cover class. (a) Before training (random initialization). (b) After training, to demonstrate the model’s ability to disentangle pure endmembers into wellseparated conceptual clusters prior to abundance estimation.

TABLE V  
PART I: COMPONENT LESION STUDY. ONE MODULE OF THE SINGLE TRAINED MDF-MAMBA IS DISABLED AT INFERENCE (SHARED WEIGHTS, ZERO RETRAINING VARIANCE). BOLD: FULL MODEL.
<table><tr><td>Configuration</td><td>MAE↓</td><td>OA↑</td><td>κ↑</td><td>Recon-MSE.↓</td><td>Recon-SAD↓</td></tr><tr><td>Full MDF-Mamba</td><td>0.0255</td><td>0.7122</td><td>0.6575</td><td>0.0067</td><td>0.1843</td></tr><tr><td>— Temporal/phenology branch (C1)</td><td>0.0305</td><td>0.7036</td><td>0.6415</td><td>0.0067</td><td>0.186</td></tr><tr><td>– Learned fusion weighting (C2)</td><td>0.0372</td><td>0.6759</td><td>0.6009</td><td>0.0072</td><td>0.190</td></tr><tr><td>— Endmember variability (C3)</td><td>0.0255</td><td>0.7122</td><td>0.6575</td><td>0.0474</td><td>0.481</td></tr></table>

Part I: Component lesion study. Table V disables one component of the frozen best model at inference. Because every row shares identical weights, the differences isolate each module’s marginal contribution without optimization noise.

Forcing the manifold-constrained fusion to use uniform stream weighting (i.e., disabling the learned fusion weighting) produces the most damaging overall effect: abundance MAE degrades by 46% (0.0255 → 0.0372), OA drops by 3.6 percentage points (71.22%→67.59%), and κ falls from 0.658 to 0.601. This confirms that the learned stream gate is the primary mechanism that turns disentangled spectral experts into accurate abundance and classification predictions.

Disabling the temporal/phenology branch (i.e., removing the bidirectional Mamba sequence modeling, C1) degrades abundance MAE by 20% (0.0255→0.0305) and drops OA by nearly one percentage point $( 7 1 . 2 2 \% \to 7 0 . 3 6 \% )$ . This demonstrates that the temporal branch provides a measurable, complementary phenological contribution to the fused representation.

Collapsing the per-pixel endmember variants to their class mean (i.e., disabling pixel-wise endmember variability, C3) leaves abundance and classification metrics unchanged—as expected, since this module acts only on the reconstruction pathway. However, it causes reconstruction MSE to rise by $7 . 1 \times ( 0 . 0 0 6 7  0 . 0 4 7 4 )$ and reconstruction SAD to increase by $2 . 6 \times \ ( 0 . 1 8 4 3  0 . 4 8 1$ rad). This confirms that modeling intra class spectral variability is essential for achieving physically faithful reconstruction.

The ablation magnitudes themselves quantify how actively each module is exploited. Removing the temporal branch (C1)

costs 20% in MAE and 0.86 pp in OA, a measurable but secondary effect. Removing learned fusion weighting (C2) is by far the most damaging operation in this study: MAE rises 46% and OA drops 3.6 pp, demonstrating that the manifoldconstrained mHC gate is the primary mechanism converting disentangled expert representations into accurate abundance predictions. The fact that C1 removal still degrades performance confirms that the adaptive spatial phenology gates do inject genuine cross-channel seasonal context; the gate’s secondary role relative to C2 reflects the complementary structure of the design: Mamba enriches spectral expert activations, while mHC determines how those enriched signals are composed into the final prediction.

Part II: Design ablations. Table VI retrains the model under identical conditions with one design decision changed at a time, isolating the endmember design (C3) and the multi-task reconstruction objective (C4).

The endmember design is the single most decisive factor in the study. Reducing the bank to one signature per class (R=1, equivalent to a standard LMM) inflates the mean endmember SAD significantly and increases reconstruction MSE by $8 . 5 \times ~ ( 0 . 0 0 6 7 {  } 0 . 0 5 6 8 )$ . Similarly, replacing the GT purity initialization with random seeding inflates endmember RMSE heavily and causes reconstruction MSE to rise by 12× $( 0 . 0 0 6 7  0 . 0 7 9 8 )$ . These results confirm that the variant count and the physics-informed initialization jointly govern endmember and reconstruction quality. Under both settings, abundance MAE increases modestly (0.0290 to 0.0294, a ${ \sim } 1 4 - 1 5 \%$ relative rise compared to the full model). This is far less severe than the 8 − 12× degradation observed in reconstruction fidelity, indicating that the primary benefit of pixel-wise variability is realized in spectral fidelity, with a secondary benefit to abundance accuracy.

Removing the unsupervised reconstruction objective (i.e., disabling the Mahalanobis and SAD terms, C4) and retraining from scratch yields a model whose abundance MAE (0.0264) and classification OA (71.01%) are comparable to the proposed full model. This indicates that the multi-task reconstruction objective primarily constraints the endmember dictionary and ensures spectral fidelity, rather than directly boosting classification accuracy on the training domain. However, without this objective, reconstruction MSE degrades by 4.2× (0.0067 → 0.0281). This demonstrates that removing the multitask loss allows the network to achieve comparable task-level accuracy via unconstrained feature memorization, completely sacrificing physically grounded spectral inversion. Because abundance metrics are easily saturated on this benchmark by black-box memorization, the reconstruction fidelity acts as the decisive signal. It proves that the multi-task loss (C4) is the critical mechanism that projects the learned representations onto a physically valid optical mixing manifold, enabling selfverifying predictions and ensuring robustness.

## O. Statistical Reproducibility and Variance Analysis

To quantify the sensitivity of MDF-Mamba to random initialization, we train the model under three independent random seeds (42, 123, and 7) under otherwise identical conditions (100 epochs, batch size 8, the same training/validation/test splits), evaluating each run on the full Alberta test set. Table VII reports the per-seed and aggregate mean ± std metrics.

The results demonstrate good reproducibility across most evaluation dimensions. The classification OA varies by only 0.0009 across seeds (range: 0.7113 − 0.7122), yielding a mean ± std of $\mathrm { O A } \ = \ 0 . 7 1 1 8 \pm 0 . 0 0 0 4$ . The reconstruction pathway is similarly stable, with Recon-MSE varying by only 0.0004 across runs. The abundance MAE varies by at most 0.0013 across seeds (range: 0.0255 − 0.0268). Meanwhile, κ varies by 0.0083 (range: 0.6492 − 0.6575), yielding $\kappa = 0 . 6 5 2 2 \pm 0 . 0 0 3 8$ . This modest but non-negligible spread indicates that random weight initialization causes a small, measurable shift in abundance and Kappa outcomes. Nevertheless, the curriculum learning strategy and GT-purity endmember initialization successfully keep classification OA and reconstruction fidelity highly stable across runs.

## P. Computational Efficiency

Table VIII reports the parameter count and floating-point operations (GFLOPs) of MDF-Mamba compared to all 11 baselines. Deep learning models are profiled with a standard 1 × 138 × 64 × 64 input patch in BFloat16 precision on a single NVIDIA RTX 6000 Ada Generation GPU, while the computational costs of traditional non-parametric models (FCLSU, SGSNMF, SUnSAL) are estimated analytically based on their algorithmic iteration limits.

With only 3.33M parameters and 14.93 GFLOPs per patch, MDF-Mamba strikes an optimal trade-off between computational efficiency and the state-of-the-art predictive accuracy presented earlier in Table I. While Transformer-based baselines such as SRViT (31.32M params, 75.31 GFLOPs) and DeepTrans (26.82M params) carry an order of magnitude more parameters, their heavy parameterization actually yields worse quantitative performance (e.g., SRViT achieves only 0.5208 OA compared to MDF-Mamba’s 0.7122). Conversely, extremely lightweight models like MiSiCNet (0.95M params, 3.89 GFLOPs) and SUnCNN (1.58M params, 7.37 GFLOPs) sacrifice representational capacity, resulting in higher abundance errors (MAE > 0.03). Traditional algorithms (FCLSU, SGSNMF, SUnSAL) possess no trainable parameters and demand very few GFLOPs (1.6–3.2), but their linear modeling assumptions severely cap their unmixing accuracy. By allocating its 3.33M parameters strategically (using an asymmetric multi-branch CNN architecture for spatial feature extraction and a highly efficient Temporal Mamba module for spectral processing) MDF-Mamba achieves the highest classification accuracy (OA 0.7122) and lowest abundance error (MAE 0.0255) across all evaluated models, operating efficiently at 48.8 patches per second with a peak memory footprint of 180.9 MB per inference.

Table IX further breaks down the internal computational profile of MDF-Mamba.

As shown, the six CNN BandBranches collectively account for 82.9% of the parameters (2.761M) and 75.8% of the FLOPs (11.32 GFLOPs), reflecting the deliberate design choice to concentrate representational capacity in the spectral expert

TABLE VI  
PART II: DESIGN ABLATIONS ON THE ALBERTA TEST SET. EACH ROW RETRAINS THE FULL MDF-MAMBA WITH ONE DESIGN DECISION CHANGED; ALL OTHER SETTINGS REMAIN FIXED. BOLD: PROPOSED CONFIGURATION (NOT NECESSARILY THE BEST VALUE IN EACH COLUMN).
<table><tr><td>Configuration</td><td>MAE↓</td><td>OA↑</td><td>κ↑</td><td>Recon-MSE↓</td><td>Endm-mSAD↓</td><td>Endm-mRMSE↓</td></tr><tr><td>Full MDF-Mamba (proposed)</td><td>0.0255</td><td>0.7122</td><td>0.6575</td><td>0.0067</td><td>0.00017</td><td>0.0458</td></tr><tr><td colspan="7">(C3) Endmember variant count R and initialization</td></tr><tr><td>R=1 (no variant bank, standard LMM)</td><td>0.0290</td><td>0.7012</td><td>0.6358</td><td>0.0568</td><td>0.0260</td><td>0.1111</td></tr><tr><td>R=10, random init (no GT purity)</td><td>0.0294</td><td>0.7016</td><td>0.6372</td><td>0.0798</td><td>0.0150</td><td>0.6537</td></tr><tr><td>(C4) Multi-task reconstruction objective</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No reconstruction loss (—Maha, —SAD)</td><td>0.0264</td><td>0.7101</td><td>0.6515</td><td>0.0281</td><td>0.00231</td><td>0.1429</td></tr></table>

TABLE VII

MULTI-SEED REPRODUCIBILITY ON THE ALBERTA TEST SET. ALL THREE SEEDS ARE EVALUATED ON THE FULL TEST SET UNDER IDENTICAL CONDITIONS. MEAN AND STANDARD DEVIATION COMPUTED OVER THREE INDEPENDENT TRAINING RUNS.
<table><tr><td>Seed</td><td>MAE↓</td><td>RMSE↓</td><td>OA↑</td><td>κ↑</td><td>Recon-MSE↓</td><td>Recon-SAD↓</td></tr><tr><td>42</td><td>0.0255</td><td>0.0654</td><td>0.7122</td><td>0.6575</td><td>0.0067</td><td>0.1843</td></tr><tr><td>123</td><td>0.0268</td><td>0.0681</td><td>0.7113</td><td>0.6492</td><td>0.0064</td><td>0.1845</td></tr><tr><td>7</td><td>0.0267</td><td>0.0679</td><td>0.7119</td><td>0.6499</td><td>0.0068</td><td>0.1874</td></tr><tr><td>Mean ± Std</td><td>0.0263 ± 0.0006</td><td>0.0671 ± 0.0012</td><td>0.7118 ± 0.0004</td><td>0.6522 ± 0.0038</td><td>0.0066 ± 0.0002</td><td>0.1854 ± 0.0014</td></tr></table>

## TABLE VIII

COMPUTATIONAL COMPLEXITY PROFILE OF ALL 11 BASELINES COMPARED TO MDF-MAMBA. MEASURED ON INPUT SHAPE (1, 138, 64, 64).   
TRADITIONAL NON-PARAMETRIC MODELS (FCLSU, SGSNMF, SUNSAL)   
CONTAIN NO TRAINABLE WEIGHTS (PARAMS = 0.00) AND THEIR GFLOPS ARE ANALYTICALLY ESTIMATED BASED ON STANDARD CONVERGENCE   
ITERATION LIMITS (E.G., ADMM, MULTIPLICATIVE UPDATES). UNMAMBA GFLOPS ARE SIMILARLY ESTIMATED ANALYTICALLY.

<table><tr><td>Model</td><td>Params (M)</td><td>GFLOPs</td></tr><tr><td>FCLSU</td><td>0.00</td><td>1.62</td></tr><tr><td>SGSNMF</td><td>0.00</td><td>3.16</td></tr><tr><td>SUnSAL</td><td>0.00</td><td>1.85</td></tr><tr><td>DeepTrans-HSU</td><td>26.82</td><td>1.04</td></tr><tr><td>PGMSU</td><td>2.15</td><td>8.82</td></tr><tr><td>SUnCNN</td><td>1.58</td><td>7.37</td></tr><tr><td>MiSiCNet</td><td>0.95</td><td>3.89</td></tr><tr><td>SSAF-Net</td><td>3.96</td><td>16.34</td></tr><tr><td>PPM-Net</td><td>2.39</td><td>9.87</td></tr><tr><td>SRViT</td><td>31.32</td><td>75.31</td></tr><tr><td>UNMamba</td><td>1.64</td><td>7.21</td></tr><tr><td>MDF-Mamba (Ours)</td><td>3.33</td><td>14.93</td></tr></table>

TABLE IX

INTERNAL COMPUTATIONAL PROFILE OF MDF-MAMBA COMPONENTS. THROUGHPUT MEASURED AS PATCHES PER SECOND (BATCH SIZE 1).
<table><tr><td>Component</td><td>Params (M)</td><td>GFLOPs</td><td>Mem (MB)</td><td>patches/s</td></tr><tr><td>MDF-Mamba (full)</td><td>3.33</td><td>14.93</td><td>180.9</td><td>48.8</td></tr><tr><td>CNN Branches (×6)</td><td>2.761</td><td>11.32</td><td></td><td></td></tr><tr><td>Temporal Mamba</td><td>0.126</td><td>2.46</td><td></td><td></td></tr><tr><td>mHC Fusion</td><td>0.072</td><td>0.29</td><td></td><td></td></tr><tr><td>ELMM + Recon Head</td><td>0.122</td><td>0.00</td><td></td><td></td></tr><tr><td>Miscellaneous (stem, SE, FiLM)</td><td>0.249</td><td>0.86</td><td></td><td></td></tr></table>

layers. The Temporal Mamba module contributes only 3.8% of parameters (0.126M) and 16.5% of FLOPs (2.46 GFLOPs), while the mHC fusion gate and ELMM reconstruction head together account for less than 6% of the total cost.

## Q. Cross-Provincial Class-Level Transfer Analysis

Table X provides a class-stratified breakdown of MDF-Mamba’s cross-provincial generalization from Alberta (training province) to Saskatchewan (test province). The Saskatchewan test region intrinsically lacks the Snow land cover class present in the original 12-class Alberta dataset. In the NALCMS legend, Snow and ice denotes areas with a perennial cover of ice and/or snow rather than seasonal snowfall [55]. In Alberta, this class occurs in the high-elevation glaciers and ice fields of the Rocky Mountains along the province’s western boundary [56], whereas Saskatchewan lies entirely within the low-relief Prairies, Boreal Plains, Boreal Shield, and Taiga Shield ecozones [57], where snow cover melts every year and no permanent snow or ice exists. Thus, the evaluation strictly covers the remaining 11 transferrable classes. To rigorously evaluate performance on a per-class basis (where global metrics like OA or κ would be dominated by majority classes) we report the per-class F1 score, which strictly balances false positives and false negatives for each individual land cover type.

The class-stratified analysis reveals a coherent ecological pattern: classes whose spectral-temporal signatures are governed by universal physical processes (water absorptance, sparse bare-soil reflectance, and stable crop phenology) transfer consistently across provinces (F1 > 0.70). Moderately transferring classes (Grassland, Wetland, Urban, Taiga) experience partial performance degradation, likely due to shifting mixing ratios and regional structural heterogeneity. In contrast, classes with regionally distinct phenology (Shrubland, Broadleaf, Polar Grass) exhibit poor transfer because their characteristic spectral dynamics are either absent from the Alberta training distribution or expressed differently in Saskatchewan’s climate regime [49]. Notably, Taiga reaches F1 = 0.432 despite accounting for only 0.16% of the Alberta training abundance, reflecting the classbalanced supervision, which assigns it an 8.3-fold effective weight. The overall cross-provincial OA of 0.7485 and κ of 0.6881 are higher than the Alberta result (OA: 0.7122, κ: 0.6575), driven by the robust transfer of dominant, easily separable classes like Cropland and Water. Conversely, the increase in abundance error (MAE from 0.0255 to 0.0492) reflects both the challenge of cross-provincial domain shift for phenologically sensitive classes and the one reference used for the full Saskatchewan scene, which counts every sub-pixel fraction assigned to a non-dominant class as an error.

TABLE X  
CLASS-LEVEL TRANSFER METRICS FOR THE ALBERTA → SASKATCHEWAN CROSS-PROVINCIAL EVALUATION (OA: 0.7485, κ: 0.6881, MAE: 0.0492). CLASSES MARKED ✓ (F1 > 0.70) TRANSFER CONSISTENTLY; CLASSES MARKED ✗ (F1 < 0.30) ARE LIMITED BY DOMAIN-SPECIFIC SPECTRAL SIGNATURES RARE OR ABSENT IN THE ALBERTA TRAINING DISTRIBUTION.
<table><tr><td>Class</td><td>F1</td><td>Transfer quality</td></tr><tr><td>Cropland</td><td>0.934</td><td>√ Excellent - stable agricultural phenology</td></tr><tr><td>Water</td><td>0.816</td><td>√ Excellent - spectrally invariant</td></tr><tr><td>Needleleaf</td><td>0.776</td><td>√ Good - consistent boreal canopy</td></tr><tr><td>Barren</td><td>0.758</td><td>√ Good - low-reflectance signature stable</td></tr><tr><td>Grassland</td><td>0.688</td><td>Moderate</td></tr><tr><td>Wetland</td><td>0.533</td><td>Moderate</td></tr><tr><td>Urban</td><td>0.506</td><td>Moderate</td></tr><tr><td>Taiga</td><td>0.432</td><td>Moderate - rare in Alberta training set</td></tr><tr><td>Shrubland</td><td>0.213</td><td>X Poor - distinct sub-provincial phenology</td></tr><tr><td>Broadleaf</td><td>0.065</td><td>X Poor - Saskatchewan broadleaf phenology diverges</td></tr><tr><td>Polar Grass</td><td>0.000</td><td>X Absent from Alberta training set</td></tr></table>

This transfer hierarchy is independently corroborated by the endmember reconstruction quality visualized in Fig. 10, where classes are ordered by ascending RMSE. At the top rows we have Needleleaf Forest $( \mathrm { R M S E } = 0 . 1 3 6 )$ , Cropland (0.161), and Grassland (0.258) correspond precisely to the classes that transfer most reliably to Saskatchewan $( \mathrm { F } 1 > 0 . 6 0 )$ , because the model has learned high-fidelity endmember representations from sufficient pure-pixel supervision. Conversely, the bottom rows Urban $( \mathrm { R M S E } = 0 . 6 7 0 )$ , Broadleaf Forest (0.720), and Taiga (0.750) include the poorest cross-provincial F1 score (Broadleaf: 0.065); Taiga, despite its low endmember fidelity, is lifted to $\mathrm { F } 1 = 0 . 4 3 2$ by the class-balanced supervision. This convergence of two independent diagnostics (within-province endmember fidelity and cross-province classification transfer) confirms that the degradation observed for these minority classes originates from insufficient pure-pixel representation and regionally bound phenological signatures in the training data, rather than from any architectural limitation of the model itself.

## R. Evaluation Against Generic Mamba Architectures

While recent state-space architectures such as DFMamba [58], MambaLG [59], MambaHSI [60], MHSSMamba [61], and Mamba-in-Mamba [62] have demonstrated exceptional performance on standard hyperspectral image classification tasks, they are structurally designed as generic feature extractors rather than dedicated spectral unmixing frameworks. Specifically, these architectures prioritize spatial-spectral classification and lack the explicit physics-constrained components such as the Extended Linear Mixing Model (ELMM) and endmember variability modeling that are strictly necessary for accurate abundance estimation and spectral reconstruction.

TABLE XI  
QUANTITATIVE COMPARISON AGAINST RECENT GENERIC MAMBA MODELS ON THE ALBERTA TEST SET. WITHOUT DEDICATED PHYSICS CONSTRAINTS (ELMM), GENERIC STATE-SPACE ARCHITECTURES FAIL TO ACHIEVE STABLE SPECTRAL RECONSTRUCTION OR ACCURATE FRACTIONAL ABUNDANCE ESTIMATION.
<table><tr><td>Method</td><td>OA</td><td>AA</td><td>κ</td><td>MAE</td><td>RMSE</td><td>SAD (rad)</td></tr><tr><td>DFMamba [58]</td><td>0.6658</td><td>0.4122</td><td>0.5923</td><td>0.0383</td><td>0.0939</td><td>1.5708</td></tr><tr><td>MambaLG [59]</td><td>0.6837</td><td>0.4324</td><td>0.6142</td><td>0.0366</td><td>0.0914</td><td>1.5708</td></tr><tr><td>MambaHSI [60]</td><td>0.6951</td><td>0.4442</td><td>0.6297</td><td>0.0342</td><td>0.0869</td><td>1.5708</td></tr><tr><td>MHSSMamba [61]</td><td>0.5253</td><td>0.2340</td><td>0.4057</td><td>0.0620</td><td>0.1260</td><td>1.5708</td></tr><tr><td>Mamba-in-Mamba [62]</td><td>0.5403</td><td>0.2471</td><td>0.4283</td><td>0.0593</td><td>0.1231</td><td>1.5708</td></tr><tr><td>MDF-Mamba (Ours)</td><td>0.7122</td><td>0.4721</td><td>0.6575</td><td>0.0255</td><td>0.0654</td><td>0.1843</td></tr></table>

Because our primary evaluation focuses on the unmixing task, our main comparative analysis (Section IV-E) relied on established, dedicated unmixing models. However, to demonstrate the necessity of physics-guided disentanglement when adapting state-space models to remote sensing, we provide a secondary evaluation against these generic Mamba architectures in Table XI. As shown, while these generic models can be adapted to estimate fractional abundances, their lack of a physics-constrained decoder prevents them from reconstructing the original spectrum. Consequently, their spectral reconstruction outputs evaluate to zero-vectors, resulting in a static, mathematical maximum Spectral Angle Distance (SAD) penalty of exactly $\pi / 2$ (1.5708 rad). This confirms that generic classification Mambas lack the capacity for full physics-based unmixing, a limitation successfully resolved by the proposed MDF-Mamba.

## V. CONCLUSION

This paper presented MDF-Mamba, a novel architecture for spectral-temporal unmixing of MODIS satellite data. Our approach addresses four key challenges in regional-scale unmixing. First, it handles spectral-temporal coupling through an asymmetric hybrid Mamba-CNN architecture. This design assigns spectral extraction to specialized CNN branches and temporal phenology encoding to a bidirectional state-space model. Second, it resolves fusion instability by using manifoldconstrained hyper-connections. Third, it models endmember variability pixel-by-pixel, combining an extended linear mixing model with a cross-attention reconstructor. Finally, it unifies these components through a multi-task learning strategy that explicitly grounds abundance estimates in physical spectral reconstruction. Evaluation of MDF-Mamba against eleven dedicated unmixing baselines was conducted, as well as five recent generic Mamba architectures.

Extensive experiments on the Alberta dataset demonstrated that MDF-Mamba achieves state-of-the-art performance, outperforming all baselines across every evaluation metric. It attained the most accurate abundance estimation (MAE: 0.0255) alongside the highest spectral fidelity (MSE: 0.0067) and overall classification accuracy (71.22%). Crucially, the model maintained this strong performance when transferred to the geographically distinct Saskatchewan test set without retraining, proving its strong cross-regional generalization. Our spatial analysis confirmed that the largest gains occurred in complex agricultural transition zones, where the multi-branch disentangling architecture provided the greatest advantage.

Ablation studies validated that our physics constrained design is essential. Incorporating pixel-wise endmember variability yielded a massive 150-fold reduction in spectral angle errors compared to single variant models. Furthermore, our comparative analysis highlighted a critical limitation in current literature: while generic Mamba models excel at standard classification, they lack the capacity for valid physics-based spectral unmixing. MDF-Mamba resolves this gap while maintaining a highly efficient computational footprint (14.93 GFLOPs), making it highly practical for large-scale earth observation.

Limitations. This study evaluates a single calendar year (2010) across two adjacent Canadian provinces, meaning our temporal models are optimized for a specific, isolated seasonal cycle. Qualitative spatial comparisons (Figs. 6, 7, 8) highlight regions selected to maximize the visible gap between MDF-Mamba and baselines and should be read alongside the aggregate metrics in Tables I–II for typical-case performance.

Future work will actively address these spatial-temporal limitations by extending the model in three directions: (1) scaling to multi-year continuous MODIS stacks to accurately capture shifting inter-annual vegetation dynamics and longterm disturbance events; (2) evaluating on a wider array of global biomes well beyond Alberta and Saskatchewan to strictly validate broader geographic generalization; and (3) investigating end-to-end joint learning of the spectral covariance structure within the Mahalanobis reconstruction loss to further improve per-class endmember quality.

## REFERENCES

[1] C. O. Justice, J. R. Townshend, E. F. Vermote, E. Masuoka, R. E. Wolfe, N. Saleous, K. J. Murphy, and J. T. Morisette, “The moderate resolution imaging spectroradiometer (MODIS): Land remote sensing for global change research,” IEEE Transactions on Geoscience and Remote Sensing, vol. 36, no. 4, pp. 1228–1249, Jul 1998.

[2] M. A. Friedl, D. Sulla-Menashe, B. T. employment, A. Schneider, N. Ramankutty, A. Sibley, and X. Huang, “MODIS collection 5 global land cover: Algorithm refinements and characterization of new datasets,” Remote Sensing of Environment, vol. 114, no. 1, pp. 168–182, 2010.

[3] D. C. Heinz and C.-I. Chang, “Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 39, no. 3, pp. 529–545, Mar 2001.

[4] N. Keshava and J. F. Mustard, “Spectral unmixing,” IEEE Signal Processing Magazine, vol. 19, no. 1, pp. 44–57, Jan 2002.

[5] J. M. Bioucas-Dias and M. A. T. Figueiredo, “Alternating direction algorithms for constrained sparse regression: Application to hyperspectral unmixing,” in 2010 2nd Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), 2010, pp. 1–4.

[6] X. Wang, Y. Zhong, L. Zhang, and Y. Xu, “Spatial group sparsity regularized nonnegative matrix factoriza-

tion for hyperspectral unmixing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 11, pp. 6287–6304, Nov 2017.

[7] A. Zare and K. C. Ho, “Endmember variability in hyperspectral analysis: Addressing spectral variability during spectral unmixing,” IEEE Signal Processing Magazine, vol. 31, no. 1, pp. 95–104, Jan 2014.

[8] R. A. Borsoi, T. Imbiriba, and J. C. M. Bermudez, “Deep generative endmember modeling: An application to unsupervised spectral unmixing,” IEEE Transactions on Computational Imaging, vol. 6, pp. 374–384, 2020.

[9] S. Shi, M. Zhao, L. Zhang, and Y. Altmann, “Probabilistic generative model for hyperspectral unmixing accounting for endmember variability,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[10] B. Rasti, B. Koirala, P. Scheunders, and P. Ghamisi, “UnDIP: Hyperspectral unmixing using deep image prior,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–16, 2022.

[11] J. Gao, Z. Zhang, Y. Zhai et al., “DeepTrans-HSU: Transformer-based autoencoder for hyperspectral unmixing,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 2092–2103, 2023.

[12] Y. Su, L. Gao, A. Plaza, X. Sun, M. Jiang, and G. Yang, “SRViT: Self-supervised relation-aware vision transformer for hyperspectral unmixing,” IEEE Transactions on Neural Networks and Learning Systems, 2025, early access.

[13] R. Chen, W. Gao et al., “SSAF-Net: Spatial-spectral attention fusion network for hyperspectral unmixing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024.

[14] Y. Zhang et al., “PPM-Net: Pixel-level probabilistic mixture network for hyperspectral unmixing,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[15] Y. Chen et al., “Contrastive self-supervised learning for hyperspectral unmixing,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[16] L. Zhang et al., “Variational inference for hyperspectral unmixing with endmember variability,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[17] H. Wang et al., “Spectral variability and class-constrained diffusion model for hyperspectral unmixing,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[18] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[19] C. Pelletier, G. I. Webb, and F. Petitjean, “Temporal convolutional neural network for the classification of satellite image time series,” Remote Sensing, vol. 11, no. 5, p. 523, 2019.

[20] G. Zhao et al., “S<sup>2</sup>mamba: A spatial-spectral state space model for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[21] X. Yao et al., “Spectralmamba: Efficient mamba for hyperspectral image classification,” IEEE Transactions

on Geoscience and Remote Sensing, 2024.

[22] Y. Dong, Y. Li et al., “UNMamba: Cascaded spatialspectral mamba for blind hyperspectral unmixing,” IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1–5, 2025.

[23] X. Qin, X. Su, and L. Zhang, “SITSMamba for crop classification based on satellite image time series,” arXiv preprint arXiv:2409.09673, 2024.

[24] Z. Dewis, Z. Xu, Y. Zhu, M. Alkayid, M. Heffring, and L. L. Xu, “Spatial-temporal-spectral Mamba with sparse deformable token sequence for enhanced MODIS time series classification,” arXiv preprint arXiv:2508.02839, 2025.

[25] B. Rasti, B. Koirala, P. Scheunders, and P. Ghamisi, “MiSiCNet: Minimum simplex convolutional network for deep hyperspectral unmixing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[26] L. Drumetz et al., “Blind hyperspectral unmixing using an extended linear mixing model to address spectral variability,” IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3890–3905, Aug 2016.

[27] Anonymous, “λsplit: Physics-informed self-supervised content-aware spectral unmixing,” ResearchGate, 2026.

[28] L. Garrett et al., “Binary linear unmixing tactile hierarchies (BLUTH): Hierarchical unmixing for spectral variability,” arXiv preprint arXiv:2604.16969, 2026.

[29] X. Cai et al., “UniRoute: Unified routing mixture-ofexperts for modality-adaptive remote sensing change detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 27 706–27 716.

[30] Y. Liu, C. Wu, M. Guan, and J. Wang, “MixtureRS: A mixture of expert network based remote sensing land classification,” Remote Sensing, vol. 17, no. 14, p. 2494, 2025.

[31] Anonymous, “MaMOL: Rethinking efficient mixture-ofexperts for remote sensing modality-missing classification,” arXiv preprint arXiv:2511.11460, 2026.

[32] X. Xie et al., “Manifold-constrained hyper-connections (mhc),” arXiv preprint arXiv:2512.24880, 2026.

[33] X. Zhu et al., “Manifold-constrained hyper-connection for stable multi-stream fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025, early access.

[34] J. R. Jensen, Remote Sensing of the Environment: An Earth Resource Perspective, 2nd ed. Upper Saddle River, NJ, USA: Pearson Prentice Hall, 2007.

[35] B. Somers, M. Zortea, A. Plaza, and G. P. Asner, “Automated extraction of image-based endmember bundles for improved spectral unmixing,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 5, no. 2, pp. 396–408, 2012.

[36] L. Drumetz, M. A. Veganzones, S. Henrot, R. Phlypo, J. M. Bioucas-Dias, J. Chanussot, and C. Jutten, “Blind hyperspectral unmixing using an extended linear mixing model to address spectral variability,” IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3890–3904, 2016.

[37] J. J. Settle and N. A. Drake, “Linear and non-linear spec-

tral mixture analysis of composite scenes,” International Journal of Remote Sensing, vol. 14, no. 6, pp. 1259–1277, 1993.

[38] D. A. Roberts et al., “Mapping chaparral in the santa monica mountains using multiple endmember spectral mixture models,” in Eighth JPL Airborne Earth Science Workshop, 1997.

[39] F. A. Kruse, A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon, and A. F. H. Goetz, “The spectral image processing system (sips)—interactive visualization and analysis of imaging spectrometer data,” Remote Sensing of Environment, vol. 44, no. 2-3, pp. 145–163, 1993.

[40] A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao, and L. G. Ferreira, “Overview of the radiometric and biophysical performance of the MODIS vegetation indices,” Remote Sensing of Environment, vol. 83, no. 1-2, pp. 195–213, 2002.

[41] X. Zhang, M. A. Friedl, C. B. Schaaf, A. H. Strahler, J. C. F. Hodges, F. Gao, B. C. Reed, and A. Huete, “Monitoring vegetation phenology using MODIS,” Remote Sensing of Environment, vol. 84, no. 3, pp. 471–475, 2003.

[42] B.-C. Gao, “NDWI—a normalized difference water index for remote sensing of vegetation liquid water from space,” Remote Sensing of Environment, vol. 58, no. 3, pp. 257– 266, 1996.

[43] N. Keshava and J. F. Mustard, “Spectral unmixing,” IEEE Signal Processing Magazine, vol. 19, no. 1, pp. 44–57, 2002.

[44] J. M. Bioucas-Dias et al., “Hyperspectral unmixing overview: Geometrical, statistical, and sparse calculusbased approaches,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 5, no. 2, pp. 354–379, 2012.

[45] Y. Zhang and Q. Yang, “A survey on multi-task learning,” arXiv preprint arXiv:1707.08114, 2017.

[46] R. Caruana, “Multitask learning,” Machine Learning, vol. 28, pp. 41–75, 1997.

[47] F. A. Kruse, A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon, and A. F. H. Goetz, “The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data,” Remote Sensing of Environment, vol. 44, no. 2-3, pp. 145–163, 1993.

[48] Agriculture and Agri-Food Canada, “Annual crop inventory,” https://open.canada.ca/data/en/dataset/ ba2645d5-4458-414d-b196-6303ac06c1c9, 2010.

[49] A. Beaudoin et al., “Mapping attributes of canada’s forests at moderate resolution through k nn and modis imagery,” Canadian Journal of Forest Research, vol. 44, no. 5, pp. 521–532, 2014.

[50] C. J. Willmott and K. Matsuura, “Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance,” Climate Research, vol. 30, pp. 79–82, 2005.

[51] T. Chai and R. R. Draxler, “Root mean square error (RMSE) or mean absolute error (MAE)?—arguments against avoiding RMSE in the literature,” Geoscientific

Model Development, vol. 7, no. 3, pp. 1247–1250, 2014.

[52] L. I.-K. Lin, “A concordance correlation coefficient to evaluate reproducibility,” Biometrics, vol. 45, no. 1, pp. 255–268, 1989.

[53] M. F. Baumgardner, L. F. Silva, L. L. Biehl, and E. R. Stoner, “Reflectance properties of soils,” Advances in Agronomy, vol. 38, pp. 1–44, 1985.

[54] D. A. Roberts, M. Gardner, R. Church, S. Ustin, G. Scheer, and R. Green, “Mapping chaparral in the santa monica mountains using multiple endmember spectral mixture models,” Remote Sensing of Environment, vol. 65, no. 3, pp. 267–279, 1998.

[55] Commission for Environmental Cooperation, “2010 land cover of North America at 30 meters, Edition 2.0,” North American Land Change Monitoring System (NALCMS); produced by Natural Resources Canada/Canada Centre for Remote Sensing, U.S. Geological Survey, INEGI, CONABIO, and CONAFOR, [Online]. Available: https://www.cec.org/north-american-environmental-atlas/ land-cover-2010-landsat-30m/.

[56] T. Bolch, B. Menounos, and R. Wheate, “Landsat-based inventory of glaciers in western Canada, 1985–2005,” Remote Sensing of Environment, vol. 114, no. 1, pp. 127– 137, 2010.

[57] Ecological Stratification Working Group, “A national ecological framework for Canada,” Agriculture and Agri-Food Canada, Research Branch, Centre for Land and Biological Resources Research, and Environment Canada, State of the Environment Directorate, Ecozone Analysis Branch, Ottawa/Hull, ON, Canada, Tech. Rep., 1996.

[58] Y. Zhao et al., “DFMamba: Dual-domain filtering mamba for hyperspectral image classification,” arXiv preprint, 2024.

[59] X. Li et al., “MambaLG: A local-global state space model for hyperspectral image classification,” arXiv preprint, 2024.

[60] Z. Wang et al., “Vision mamba for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[61] L. Chen et al., “Multi-scale hybrid state space model for hyperspectral image classification,” arXiv preprint, 2024.

[62] J. Zhang et al., “Mamba-in-mamba: A hierarchical SSM for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, 2024.