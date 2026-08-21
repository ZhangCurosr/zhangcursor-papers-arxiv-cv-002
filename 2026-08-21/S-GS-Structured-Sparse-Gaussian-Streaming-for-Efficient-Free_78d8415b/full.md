# S²GS: Structured Sparse Gaussian Streaming for Efficient Free-Viewpoint Video Reconstruction on Edge-IoT Devices

Yiwei Li, Jiannong Cao, Fellow, IEEE, Weixun Gao, Rui Cao, Songye Zhu, Yinfeng Cao, and Mingjin Zhang

Abstract—Streaming reconstruction of Free-Viewpoint Videos (FVVs) supports immersive Internet of Things (IoT) services, such as telepresence and digital twin visualization. Existing methods suffer from high per-frame optimization time and large storage footprints, limiting deployment on resource-constrained Edge-IoT devices. To address these challenges, we propose Structured Sparse Gaussian Streaming (S²GS), an FVV reconstruction framework that exploits structure-aware temporal sparsity to selectively update Gaussian residuals, enabling efficient streaming without compromising visual fidelity. In the spatial domain, a streaming octree hierarchically organizes Gaussian residuals, capturing spatial correlations that guide residual updates. In the temporal domain, a structured gating mechanism, comprising hierarchical feature propagation (HFP) and Gumbel-Sigmoid sampling, converts hierarchical dynamic cues into sparse residual update decisions under differentiable optimization. A multi-level discrete scheme is further adopted to provide fine-grained control over residual updates while preserving intricate dynamic details. Extensive experiments across consumer GPUs, industrial edge IoT devices, and a physical telepresence testbed demonstrate that S<sup>2</sup>GS consistently reduces per-frame optimization time and storage footprint while maintaining competitive visual quality. Compared with QUEEN, S<sup>2</sup>GS reduces per-frame optimization time by 59% and storage costs by 85% on an RTX 4090 GPU. On the Jetson AGX Orin, S²GS delivers the highest rendering throughput (60+ FPS) and the lowest energy consumption among the evaluated methods, demonstrating its potential for deployment in resource-constrained systems. Code: this URL

Index Terms—Streaming reconstruction, free-viewpoint video, Gaussian Splatting, edge computing, Internet of Things (IoT), resource-constrained systems, on-device deployment.

## I. INTRODUCTION

Immersive Internet of Things (IoT) services, such as telepresence [1], [2] and digital twin (DT) visualization [3], require realistic representations that capture the dynamic evolution of physical environments. Free-viewpoint video (FVV) provides this foundation by reconstructing dynamic 3D scenes that users can interactively explore from freely selected viewpoints [4]. Streaming FVV reconstruction, in which scenes are updated frame by frame, further enables online responses and reduced per-frame packet sizes compared to offline FVV approaches, making it promising for efficient operation and transmission in bandwidth-constrained edge IoT systems [5]–[7].

Recent advances in Gaussian Splatting (GS) have significantly improved the efficiency of FVV reconstruction over neural volumetric methods such as NeRF [8] and its variants [9]–[11]. By leveraging an explicit point-based representation, GS enables photorealistic visualization with efficient training and rendering. Nevertheless, existing GS-based streaming approaches still suffer from high per-frame optimization time and large storage footprints, which limit their deployment on resource-constrained edge-IoT devices. As shown in Fig. 2(c), when deployed on the NVIDIA Jetson AGX Orin, current methods cannot simultaneously maintain high visual quality, low optimization time, and compact storage. For example, QUEEN [12], one of the most recent approaches, requires more than 17 seconds of per-frame optimization time (reported as Train (Sec)) and 0.48 MB of storage per frame. At 30 FPS, this amounts to 14.4 MB of residual data per second, introducing accumulated storage overhead for edge-assisted telepresence systems or industrial DTs [13], [14].

![](images/d8baad6e628b7ec90a52b97e9fe4c28d3f6ce49d11fa2d7452d39a884833c468.jpg)  
Fig. 1. Schematic comparison of Gaussian residual update paradigms. Top: Existing GS-based streaming methods perform primitive-wise dense residual updates. Bottom: S<sup>2</sup>GS exploits spatially correlated temporal sparsity to select a small number of hierarchical regions and perform region-wise sparse residual updates only within active regions.

We argue that the trade-off among quality, optimization time, and storage partly arises from the dense residual updates in existing GS-based methods (Fig. 1, top). Most prior approaches do not explicitly identify which Gaussians require residual updates across frames; instead, they improve compactness or compression efficiency by reorganizing Gaussian primitives [15]–[17] or by quantizing Gaussian attributes and encoding residuals [12], [18]–[20]. Although effective, these methods primarily reduce the cost of representing or transmitting residuals, rather than determining where residual updates are necessary. They overlook a key property: the temporal sparsity of Gaussian residual updates, namely, that only a small subset of Gaussians undergo substantial residual updates between consecutive frames, while most primitives remain static or change slightly. As illustrated in Fig. 2(b), only 4.52% of Gaussian primitives are active for residual updates in a representative FVV frame (see cross-dataset statistics in Appendix C). Under dense update formulations, computation and storage are allocated to Gaussians with little contribution to scene dynamics, resulting in redundant optimization and storage overhead. This observation suggests that compact representation and residual compression alone are insufficient; explicit modeling of sparse Gaussian residual updates provides a complementary direction for efficient FVV streaming.

![](images/26659682626a791247be5a524b45469719b06d09684fa8c7d1767471dd48d908.jpg)  
(a) Comparison on an NVIDIA RTX 4090 GPU

![](images/2340723e3bfc5e547de9eaa9f6b0fd8f0806d54f96ce137b2e56a2072a41566a.jpg)  
(b) Efficient Streaming with Sparse Residual Updates

![](images/27694894b2c43e756c68340829ef93de69e1d2fd90bb915457cde7701c3eba0e.jpg)  
(c) Comparison on an NVIDIA Jetson AGX Orin  
Fig. 2. Overview and performance of the proposed structured sparse Gaussian streaming framework (S²GS). S²GS delivers high-quality FVV reconstruction with improved efficiency in per-frame training, rendering, and storage. (a) Quantitative comparison on the N3DV dataset, where PSNR, per-frame training time, storage cost (circle size), and rendering throughput (color) are jointly compared. (b) Visualization of sparse residual updates on the challenging Flame Salmon scene. S<sup>2</sup>GS-fast concentrates updates on dynamic regions, using only 4.52% active Gaussians and 4.37% active gates for efficient FVV streaming (c) Edge deployment on the N3DV dataset at 1/8 resolution, where S²GS-edge outperforms baseline methods in visual quality, storage, per-frame training time, and rendering throughput.

A further observation is that the temporal sparsity of Gaussian residual updates is not necessarily modeled independently across Gaussian primitives; instead, it often exhibits spatially correlated patterns. As illustrated in Fig. 2(b), the active Gaussians can be summarized by only 4.37% active gates (see cross-dataset statistics in Appendix C), where residual updates are concentrated in local regions rather than uniformly scattered over all primitives. This region-level sparsity suggests that residual updates can be modeled more efficiently beyond primitive-wise updates. In particular, a hierarchical spatial organization provides a natural structure for representing and propagating local sparse patterns from fine primitives to coarser regions [21]. Together with the first observation that only a small subset of Gaussians requires substantial residual updates, this finding suggests FVV streaming to move beyond dense primitive-wise residual updates and exploit scene structure to guide sparse residual updates. This motivates our objective: developing a structure-aware sparsity framework for FVV streaming that induces sparse Gaussian residual updates through the spatial hierarchy of the scene, thereby improving efficiency for deployment on edge-IoT devices.

To achieve this goal, we propose the Structured Sparse Gaussian Streaming (S<sup>2</sup>GS) framework. As shown in Fig. 1 bottom, the core idea is to model temporally sparse residual updates over a fixed hierarchical structure, so that sparse update patterns follow the region-wise organization of Gaussian primitives. In the spatial domain, the streaming octree hierarchically organizes Gaussian residuals and captures spatial correlations among dynamic variations. In the temporal domain, rather than learning dense primitive-wise residual updates, we formulate residual learning as a structure-aware sparse update process, in which region-wise residual updates are selectively assigned to dynamic Gaussians.

Nevertheless, realizing this formulation is nontrivial. The first challenge is how to propagate dynamic confidences, which indicate whether a Gaussian primitive requires residual updates, across the spatial hierarchy while preserving finegrained temporal changes. The second challenge is how to convert these structure-aware confidences into sparse residual update decisions under differentiable optimization. To address these challenges, we develop a structured gating mechanism with two coupled components. First, hierarchical feature propagation (HFP) aggregates dynamic cues along the spatial hierarchy, producing structure-aware dynamic confidences that capture both local temporal variations and the spatial consistency of neighboring dynamic regions. Second, Gumbel-Sigmoid sampling converts these confidences into continuous soft gates, which relax discrete residual update decisions and enable differentiable optimization. Since strict binary discretization is overly aggressive and suppresses regions that require residual updates, we further introduce a multi-level discrete gating scheme that replaces hard 0/1 gate decisions. This design provides finer-grained control over the strength of residual updates and helps preserve intricate dynamic details. In this way, sparsity is encouraged not only by a regularizer, but also by structured-guided residual update decisions.

We evaluate S²GS against state-of-the-art (SOTA) baselines on a consumer-grade GPU, an industrial-grade edge-IoT device, and a real-world physical testbed for telepresence. The main contributions of this work are summarized as follows.

1) We identify structured sparsity in Gaussian residual updates as a key property for efficient FVV streaming, showing that sparse residual updates are spatially correlated and can be guided by the scene hierarchy.

2) We propose $\mathrm { S ^ { 2 } G S }$ , an FVV streaming framework that exploits structure-aware temporal sparsity to selectively update Gaussian residuals, enabling efficient streaming reconstruction without compromising visual fidelity.

3) We develop a structured gating mechanism with two key components. HFP aggregates primitive-wise dynamic cues into region-wise confidences and propagates gate decisions through the hierarchy, while the multi-level discrete scheme mitigates gate collapse and maintains stable sparse updates with intricate dynamic details.

4) Extensive experiments on challenging FVV datasets demonstrate that $\mathrm { S ^ { 2 } G S }$ maintains competitive visual quality while reducing the storage footprint and improving training and rendering efficiency. Additional evaluations on an edge device and a physical testbed further validate its low resource costs and practical potential in resource-constrained edge-IoT systems.

## II. RELATED WORK

## A. Offline FVV Reconstruction

The recent emergence of Gaussian Splatting [22] has significantly advanced offline FVV reconstruction due to its efficiency and flexibility. For example, STG [23] and Ex4DGS [24] model dynamic motions of Gaussian primitives using polynomial functions. Several representative methods, including E-D3DGS [25] and 4DGaussians [26], decouple dynamic scenes into a canonical space and a deformation field. 4DGS [27] further extends Gaussian Splatting to higher dimensions by introducing spatiotemporally coherent 4D Gaussians. Meanwhile, Grid4D [28] leverages hash encoding together with an attention mechanism to capture object deformations. Although these approaches achieve impressive visual quality and real-time rendering, they typically require access to complete video sequences for model optimization, which fundamentally limits their applicability to live FVV streaming.

## B. Online FVV Reconstruction

Online FVV reconstruction methods sequentially optimize scenes over time, enabling live-streaming input processing and incremental model updates. StreamRF [9] and NeRF-Player [11] extend implicit neural representations to support streamable rendering. Similarly, 3DGStream [15] adapts 3D Gaussian Splatting for FVV streaming by incorporating multiresolution hash encoding. HiCoM [16] and ReCon-GS [29] introduce explicit motion fields to improve model efficiency. In contrast, QUEEN [12] and 4DGC [19] employ differentiable quantization to compress Gaussian residual attributes. IGS [30] further leverages a pretrained motion network to enable generalizable FVV streaming, while ComGS [17] exploits motion locality to reduce memory overhead. Despite their impressive results, these methods largely overlook the inherent structured sparsity for residual updates in FVVs. Additionally, most existing approaches optimize only specific aspects (e.g., quality, model size), making them less suitable for deployment on resource-constrained edge-IoT devices, where comprehensive improvements in per-frame optimization, storage, and resource consumption are required for real-world IoT applications.

## C. Hierarchical Scene Representation

Hierarchical representations have been widely explored to improve the scalability and rendering efficiency of Gaussian Splatting. For instance, Scaffold-GS [31] organizes local Gaussians with anchors, while Octree-GS [32] extends this idea for level-of-detail (LOD) rendering. LODGE [33] further introduces a hierarchical LOD representation for efficient largescale rendering. In contrast to prior methods that use hierarchical structures to improve scene representation and rendering, $\mathrm { { S ^ { 2 } G S } }$ employs a fixed hierarchy to organize and selectively update Gaussian residuals, while propagating sparse gating decisions across the hierarchy during FVV streaming.

## III. PROPOSED METHOD

## A. Problem Formulation

Given a set of synchronized multi-view image sequences $\{ \mathcal { T } _ { t = 0 } ^ { T - 1 } \}$ with known camera poses, where t indexes T frames, our objective is to sequentially synthesize photorealistic FVVs in a streaming manner at each time step. Let $\mathcal { G } _ { t }$ denote the set of 3D Gaussians at time t, defined as $\mathcal { G } _ { t } = \{ \mu _ { t } , \boldsymbol { q } _ { t } , \boldsymbol { s } _ { t } , \boldsymbol { \sigma } _ { t } , \boldsymbol { c } _ { t } \}$ ， where ${ \pmb \mu } _ { t } \in \mathbb { R } ^ { N \times 3 } , { \pmb q } _ { t } \in \mathbb { R } ^ { N \times 4 } , { \pmb s } _ { t } \in \mathbb { R } ^ { N \times 3 } , { \pmb \sigma } _ { t } \in \mathbb { R } ^ { N }$ , and $\mathbf { } c _ { t }$ represent the position vector, rotation quaternion, scaling factor, opacity, and color coefficient of the Gaussian set with N primitives, respectively [22]. The Gaussian set $\mathcal { G } _ { t + 1 }$ for the next frame is then obtained by learning residuals $\mathcal { R } _ { t }$ with respect to the optimized $\mathcal { G } _ { t }$ from the previous frame:

$$
\mathcal { G } _ { t + 1 } = \mathcal { G } _ { t } + \mathcal { R } _ { t } ,\tag{1}
$$

where $\mathscr { R } _ { t } = \{ \Delta \mu _ { t } , \Delta q _ { t } , \Delta s _ { t } , \Delta \sigma _ { t } , \Delta c _ { t } \}$ consists of residuals for each Gaussian attribute. The Gaussian set $\mathcal { G } _ { 0 }$ at the first frame is initialized and optimized from point clouds generated by 3D reconstruction pipelines (e.g., COLMAP [34], [35]). This work aims to develop a structure-aware sparsity framework that guides sparse Gaussian residual updates by the underlying spatial hierarchy of the scene, thereby improving efficiency for practical deployment on resource-constrained edge-IoT infrastructures.

## B. Overview

To enable FVV streaming on resource-limited edge-IoT devices, we propose S²GS, a GS-based streaming framework that leverages end-to-end learnable structured sparsity to achieve fast and compact reconstruction. The overview of S²GS is illustrated in Fig. 3, which consists of three components.

1) Streaming Octree Representation: This forms the basic spatial structure of the FVV streaming pipeline. By leveraging multi-resolution grids, it hierarchically organizes Gaussian residuals to provide spatial correlations among dynamic variations. The structure further supports efficient streaming queries for HFP during optimization.

![](images/04feca5dc538d9de0565f26072a0a84693acb7759e7d16fa0dc324e1214efc68.jpg)  
Fig. 3. Overview of the proposed S²GS framework. (a) The streaming octree representation is initialized using root Gaussian primitives from the first frame and subsequently represents FVVs with multi-resolution grids, enabling hierarchical allocation of Gaussian residuals and efficient point queries for FVV streaming. (b) A structured gating mechanism is introduced to induce sparse Gaussian residual updates through hierarchical feature propagation, differentiable sampling with Gumbel-Sigmoid, and multi-level (ML) discretization using a straight-through estimator (STE). (c) A sparse regularization loss is incorporated to enable efficient end-to-end optimization of structured gates and sparse residual updates.

2) Structured Gating Mechanism: The core component of S²GS that formulates the residual learning as a structure-aware sparse update process. It first estimates per-point dynamic confidences from consecutive frames and propagates them through the spatial hierarchy. Probabilistic gating decisions are then realized via Gumbel-Sigmoid relaxation paired with a multi-level straight-through estimator (STE) [36], thereby yielding discrete and learnable gates.

3) Efficient End-to-End Optimization: This component enables end-to-end optimization of both structured gates and attributes of Gaussian residuals. A regularization loss is applied to jointly optimize gate decisions and residual updates.

## C. Streaming Octree Representation

Inspired by the octree-based hierarchical decomposition [21], [32], we introduce a streaming octree representation. The framework is initialized with Root Gaussian Primitives, denoted as $\mathcal { G } _ { \mathcal { R } }$ , which is a static set of 3D Gaussians reconstructed from the first frame of multi-view inputs.

1) Hierarchical Gaussian Residual Allocation: Given the global spatial bounds of $\mathcal { G } _ { \mathcal { R } } .$ , an octree is first constructed within the bounded volume, and each anchor is initialized at the centroid of its corresponding voxel [32]. Based on this spatial structure, we represent FVV streaming using fixed anchors with temporally varying and hierarchical Gaussian residuals. The residuals of the Gaussian set at frame t are defined as:

$$
\mathcal { R } _ { t } = \big \{ \Delta o _ { t } , \Delta q _ { t } , \Delta v _ { t } , \Delta \sigma _ { t } , \Delta c _ { t } \big \} ,\tag{2}
$$

$$
\Delta \pmb { v } _ { t } = \mathrm { c o n c a t } \big ( \Delta \pmb { s } _ { t } , \Delta \pmb { f } _ { t } \big ) .\tag{3}
$$

where concat(·) denotes the concatenation operation, $\Delta o _ { t } \in$ $\mathbb { R } ^ { N \times 3 }$ represents the offset vector of the Gaussian residuals, and $\Delta \bar { f _ { t } } \ \in \ \mathbb { R } ^ { N \times 3 }$ is the updated anchor-dependent scaling factor. As illustrated in Fig.3 (a), the proposed streaming octree representation hierarchically organizes Gaussian residuals and captures spatial correlations across different levels of dynamic variation. This allocation establishes the structural foundation for the subsequent structure-aware sparse residual updates.

2) Efficient Streaming Queries: Unlike prior works [12], [15], [16], [29] that perform online pruning and densification of their spatial data structures, S<sup>2</sup>GS keeps the spatial hierarchy fixed once established. The hierarchy remains unchanged throughout the training of subsequent frames. This fixedhierarchy design allows all per-frame Gaussian residuals to be associated with persistent, pre-allocated anchors. Consequently, the streaming octree lookup operation in S²GS is O(log N), where N is the number of Gaussian primitives. This logarithmic complexity enables efficient point queries within the spatial hierarchy, making S²GS a practical solution for spatial indexing in the subsequent FVV streaming process.

## D. Structured Gating Mechanism

Establishing an effective gating mechanism requires propagating dynamic confidences and optimizing gate decisions in a differentiable manner. In addition, gate learning in $\mathrm { S ^ { 2 } G S }$ accounts for structured sparsity, allowing propagation across the spatial hierarchy while avoiding unnecessary residual updates in static regions. For notational simplicity, we omit the subscript t in this section, as all subsequent calculations are performed with respect to frame t.

1) Dynamic Confidence Estimation: We estimate the confidence of Gaussian primitives being dynamic based on the viewspace gradient difference (VGD) [12], [17]. Specifically, during FVV streaming, we first compute the VGD m ∈ $\mathbb { R } ^ { N \times \breve { 2 } }$ from two consecutive Gaussian sets, each containing $N$ primitives. The dynamic confidence of the i-th Gaussian is computed as follows:

$$
p _ { i } = \sigma \bigg ( \frac { | m _ { i } | - \mathrm { m e d i a n } ( | m | ) } { \mathrm { M A D } ( | m | ) + \varepsilon } \bigg ) ,\tag{4}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function, MAD(·) denotes the median absolute deviation, and $\varepsilon$ is a small positive constant for numerical stability, set to $2 \times 1 0 ^ { - 4 }$

2) Hierarchical Feature Propagation: Once $p _ { i }$ is estimated, the cached features are further aggregated for efficient gate learning. A straightforward strategy is to discard voxels with confidence values below a fixed threshold [37]. However, such threshold-based pruning may remove informative features and weaken spatiotemporal consistency, which is critical for reconstructing coherent dynamic scenes in FVV streaming.

To address this issue, we propose a hierarchical feature propagation (HFP) module. Specifically, leaf-level confidences are propagated through the octree hierarchy and aggregated into root-level voxels by max pooling:

$$
p _ { j } = \operatorname* { m a x } _ { \{ i | \mathrm { i d x } ( i ) = j \} } p _ { i } , \ i \in \{ 0 , 1 , \ldots , N - 1 \} ,\tag{5}
$$

where $j \in \{ 0 , 1 , \ldots , R - 1 \}$ indexes the root-level voxels, and idx(·) maps each Gaussian primitive i to its corresponding root-level parent $j .$ The aggregation strategy in Eq. (5) is central to HFP. Unlike threshold-based pruning, which irreversibly removes low-confidence voxels before optimization, the proposed max-pooling scheme preserves salient dynamic cues within each local subtree and retains candidate motion evidence for subsequent gate learning. As a result, the rootlevel confidence provides a conservative indication of whether informative dynamics exist within a structured local region.

In addition, after gate sampling in Sec. III-D3, root-level gate decisions are distributed to all associated leaf-level Gaussians through the same index mapping. With HFP, the confidence of each root-level region is computed by aggregating the confidences of its associated leaf-level Gaussians. This allows gate learning to be performed at the region level while retaining fine-grained dynamic cues from leaf-level Gaussians. As a result, HFP preserves spatiotemporal consistency and reduces the computational cost of gate learning.

3) Differentiable Gate Sampling: The gate assigned to each Gaussian residual update is a binary variable. The main challenge for differentiable gate sampling lies in the nondifferentiability of discrete operators, such as arg max(·) and onehot(·) [38]. To address this issue, we leverage Gumbel Sigmoid [39], a binary concrete distribution, to provide a continuous, differentiable relaxation of the discrete sampling process, enabling gradient-based optimization of the gating mechanism:

$$
n \sim \mathrm { L o g i s t i c } ( 0 , 1 ) ,
$$

$$
\tilde { g } _ { j } = \sigma \Big ( \frac { \log \big ( p _ { j } / ( 1 - p _ { j } ) \big ) + n } { \tau } \Big ) ,\tag{7}
$$

(6)

![](images/a5ce2567cc716f8225de759e6d3ac7522d8cecca6fe8f9030b65645edf300aa2.jpg)

![](images/efedae4d6826b08bdfdb238916f64032482c57d832ac3163e50cdba11e2293e9.jpg)  
(a) Discretization Behaviors of Different STE Variants.

![](images/e5698731eb5005a2e88b22551d5ace7340ea3e90c05b9500628fafb7f8b36f1b.jpg)

(Left: Continuous Soft Gates, Mid: Binary STE, Right: ML-STE)  
![](images/bd0c9d639201922ae79d70cd534421feec2b25c4d15cc25a94d207ef6b95381b.jpg)

![](images/519c32f372e2644cc6dca27497c8fe3a33dab16efff0713b467971c76eb0a686.jpg)  
(b) Gate Collapse During Training.  
(c) Qualitative Comparison.  
Fig. 4. Analysis of different STE variants. (a) Continuous soft gates and their discretized counterparts produced by Binary STE and the proposed Multi-Level (ML) STE. Unlike Binary STE, ML-STE preserves finer-grained gate states after discretization. (b) Number of active gates over training epochs for a representative FVV frame. Binary STE rapidly leads to gate collapse, whereas ML-STE maintains stable sparse residual updates throughout training. (c) Visual comparison on the same FVV frame as in (b), where ML-STE achieves better reconstruction quality than Binary STE, particularly in the dynamic region.

where $\tilde { g } _ { j }$ denotes the continuous soft gate, n is a noise variable sampled from a Logistic distribution, and the temperature τ controls the hardness of the sampled gate and is set to 0.3.

While this formulation yields differentiable gates, it does not inherently induce sparsity. To obtain sparse, discrete gates during training while preserving gradient flow, we employ a straight-through estimator (STE) [36]. Specifically, in the forward pass, the soft gate $\tilde { g } _ { j }$ is rounded to generate a discrete, gradient-stopped gate ${ \bar { g } } _ { j } { \mathrm { : } }$

$$
\bar { g } _ { j } = \mathrm { s t o p \_ g r a d } ( \mathrm { r o u n d } ( \tilde { g } _ { j } ) ) ,\tag{8}
$$

where stop grad(·) denotes the gradient-stopping operation. In the backward pass, gradients are propagated as if the gate remained continuous $\tilde { g } _ { j }$

In our experiments, however, we observed that directly rounding STE outputs to strict binary values of 0 or 1, as in Binary STE, can lead to gate collapse during training. Most gates rapidly become zero, preventing the model from assigning residual updates to regions with fine-grained temporal variations and severely degrading reconstruction quality, as illustrated in Fig. 4. To address this issue, we employ a multi-level (ML) STE scheme. It discretizes $\tilde { g } _ { j }$ to one decimal place rather than enforcing strict 0/1 decisions. The resulting multi-level discrete gates retain the exact-zero state for sparse selection while providing control over the magnitude of active residual updates. This finer discretization mitigates gate collapse, maintains a stable sparse set of active gates throughout training, and preserves the residual capacity required to model intricate temporal dynamics, thereby improving visual fidelity.

Overall, this design enables the underlying gate confidence $p _ { j }$ to be optimized end-to-end while preserving sparse residual update decisions at runtime, thereby ensuring stable and consistent gate behavior from training to rendering.

## E. Efficient End-to-End Optimization

To encourage sparsity and regularize the magnitude of residual updates, we propose a joint regularization loss ${ \mathcal { L } } _ { \mathrm { r e g } } .$ It combines an approximate $L _ { 0 }$ penalty that promotes exact zero residuals with an $L _ { 2 }$ term that constrains the magnitude of non-zero offset updates, which is formulated as:

$$
\mathcal { L } _ { \mathrm { r e g } } = \sum _ { i = 0 } ^ { N - 1 } \pi _ { i } \cdot \Big ( \lambda _ { 1 } + \lambda _ { 2 } \left. \Delta \pmb { \mathscr { o } } _ { i } \right. _ { 2 } ^ { 2 } \Big ) ,\tag{9}
$$

where $\pi _ { i } = 1 - F _ { p _ { i } } ( \eta )$ denotes the dynamic probability of Gaussian primitive i, $F _ { p _ { i } } ( \eta )$ is the cumulative distribution function of $p _ { i }$ evaluated at a threshold η, $\Delta \mathbf { o } _ { i }$ is the offset update without sparsification, and $\lambda _ { 1 }$ , λ<sub>2</sub> are weighting coefficients. This formulation enables the model to jointly learn sparse and compact residual updates for Gaussian attributes while maintaining differentiability throughout optimization.

Following [12], residuals of Gaussian attributes are learned from latent vectors $\{ l _ { a } | a \in \{ q , s , \sigma , c \} \}$ by passing through compact linear decoders. The loss function is formulated as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p h o t o } } + \mathcal { L } _ { \mathrm { r e g } } ,\tag{10}
$$

where the photometric loss $\mathcal { L } _ { \mathrm { p h o t o } }$ is computed by comparing rendered images across time steps with ground-truth images using a combination of $L _ { 1 }$ loss and D-SSIM loss [22].

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Goal

We conduct experiments to answer the following questions.

1) RQ-1 (Quality): Does S²GS achieve competitive visual quality compared to state-of-the-art methods?

2) RQ-2 (Efficiency): Does S²GS outperform existing baseline methods in model complexity, storage size, training efficiency, and rendering throughput?

3) RQ-3 (Ablation Study): Does each proposed component of S²GS contribute positively to its overall results?

4) RQ-4 (Deployability): Is S²GS easily deployable on edge-IoT devices while meeting the requirements for quality, timeliness, and resource efficiency?

5) RQ-5 (Case Study): How does S²GS perform when applied in real-world edge-IoT environments?

## B. Dataset

We evaluate our method on three challenging FVV datasets:

1) N3DV Dataset: N3DV [10] includes six dynamic indoor scenes that exhibit various lighting conditions and intricate volumetric details. The videos were captured at a resolution of $2 7 0 4 \times 2 0 2 8$ and a frame rate of 30 FPS, with 18 to 21 camera views per scene. Following prior studies [12], we downsampled the image by a factor of 2 for our experiments.

2) Meet Room Dataset: Meet Room Dataset [9] was recorded using a multi-view system equipped with 13 synchronized cameras, covering three indoor scenes. The videos have a resolution of 1280 × 720 and a frame rate of 30 FPS.

3) ENeRF Outdoor Dataset: ENeRF Outdoor Dataset [42] contains long-sequence videos of complex human motions with deformable objects in an outdoor setting, captured by 18 cameras at a resolution of $1 9 2 0 \times 1 0 8 0$ . We selected three 300-frame sequences with 6 different actors for experiments.

## C. Baseline, Metric, and Implementation

1) Baseline: We compare S²GS against several competitive online and offline FVV methods, including 4DGaussians [26], 4DGS [27], STG [23], Ex4DGS [24], 3DGStream [15], HiCoM [16], 4DGC [19], Queen [12], and ReCon-GS [29]. We evaluate three variants of S²GS—S²GS-full, S²GS-fast, and S²GS-edge—using 250 training epochs on the first frame and 20, 10, and 6 epochs, respectively, on each subsequent frame. In particular, S²GS-edge is tailored for deployment on resource-constrained edge-IoT devices.

2) Metric: The quality of rendered images is assessed using PSNR, SSIM, and LPIPS with the VGG backbone. To evaluate streaming efficiency, we also report the average number of Gaussians, average storage size, average training time, and average rendering throughput per frame.

3) Implementation Details: The main experiments for RQ-1, RQ2, and RQ-3 are conducted on an Ubuntu 22.04.5 LTS system equipped with an Intel Core i7-13700 CPU, an NVIDIA GeForce RTX 4090 GPU, and 64 GB of RAM. Deployment experiments are performed on the NVIDIA Jetson AGX Orin Developer Kit equipped with a 1.3 GHz NVIDIA Ampere architecture GPU, a 2.2 GHz ARM Cortex-A78AE processor, and 64GB of RAM. Additionally, in the implementation of S²GS, λ<sub>1</sub>, λ<sub>2</sub>, and η in Eq. 9 are set to 0.01, 0.01, and 0.05, respectively. The round(·) operation in Eq. 8 rounds soft gates to one decimal place to encourage sparse and discrete gates. All reported results are averaged over 3 independent runs to ensure statistical reliability.

## D. RQ-1 (Quality)

Our method introduces a streaming octree structure that decouples multi-scale Gaussian residuals across different levels of dynamic variation. This design enables fine-scale Gaussians to capture dynamic details effectively, thereby improving overall rendering quality. Furthermore, the multi-level STE scheme mitigates gate collapse and produces fine-grained gate states, enabling the model to preserve sparse updates for intricate temporal dynamics. As shown in Figs. 5 and 6, Tabs. I and II, we compare S²GS-full with state-of-theart methods and demonstrate that our approach consistently achieves competitive or superior visual results on both indoor and outdoor scenes, particularly in regions exhibiting complex motions and intricate deformations. Notably, compared with QUEEN [12] on ENeRF Outdoor, S<sup>2</sup>GS-full achieves slightly lower PSNR (25.94 vs. 26.13 dB) but higher SSIM (0.863 vs. 0.843) and lower LPIPS (0.124 vs. 0.154), indicating a visual quality trade-off with slightly larger pixel-wise errors but better preservation of structural and perceptual details. More results can be found in the Supplementary Material.

TABLE I  
QUANTITATIVE COMPARISONS ON THE N3DV DATASET AT 1/2 RESOLUTION. FOR ALL METHODS, PER-FRAME METRICS ARE AVERAGED ACROSS ALL FRAMES. RESULTS FOR METHODS MARKED WITH <sup>†</sup> ARE TAKEN FROM THE ORIGINAL PAPERS. ALL EXPERIMENTS ARE CONDUCTED ON AN NVIDIA RTX 4090 GPU. RED AND BLUE DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY, WITHIN EACH CATEGORY.
<table><tr><td rowspan="2">Methods</td><td rowspan="2"></td><td colspan="7">PSNR (dB)↑ / SSIM↑</td><td rowspan="2">#Gaussians (k)↓</td><td rowspan="2">Storage (MB)↓</td><td rowspan="2">Train (Sec)↓</td><td rowspan="2">Render (FPS)↑</td></tr><tr><td>Coffee Martini</td><td>Cook Spinach</td><td>Cut Beef</td><td>Flame Salmon</td><td>Flame Steak</td><td>Sear Steak</td><td>Average</td></tr><tr><td rowspan="8">Oine</td><td>K-Planes† [40]</td><td>29.99 / 0.925</td><td>32.60 / 0.942</td><td>31.82 / 0.943</td><td>30.44 / 0.924</td><td>32.38 / 0.957</td><td>32.52 / 0.955</td><td>31.63 / 0.941</td><td></td><td>1.04</td><td>21.6</td><td>0.3</td></tr><tr><td>NeRFPlayer† [11]</td><td>31.53 / 0.951</td><td>30.56 / 0.929</td><td>29.35 / 0.908</td><td>31.65 / 0.940</td><td>31.93 / 0.950</td><td>29.13 / 0.908</td><td>30.69 / 0.931</td><td></td><td>17.1</td><td>72.0</td><td>0.05</td></tr><tr><td>4DGaussians [26]</td><td>30.10 / 0.935</td><td>31.15 / 0.958</td><td>32.60 / 0.951</td><td>30.21 / 0.934</td><td>33.49 / 0.961</td><td>32.64 / 0.961</td><td>31.70 / 0.950</td><td>192</td><td>0.29</td><td>7.92</td><td>23</td></tr><tr><td>Grid4D [28]</td><td>28.69 / 0.919</td><td>32.90 / 0.957</td><td>33.61 / 0.958</td><td>29.86 / 0.930</td><td>32.98 / 0.963</td><td>33.49 / 0.965</td><td>31.92 / 0.949</td><td>202</td><td>0.16</td><td>16.4</td><td>127</td></tr><tr><td>STG [23]</td><td>28.61 / 0.916</td><td>33.18 / 0.952</td><td>33.52 / 0.954</td><td>29.48 / 0.918</td><td>33.64 / 0.960</td><td>33.89 / 0.961</td><td>32.05 / 0.944</td><td>434</td><td>0.67</td><td>15.1</td><td>149</td></tr><tr><td>4DGS [27]</td><td>28.33 / 0.920</td><td>32.93 / 0.956</td><td>33.85 / 0.959</td><td>29.38 / 0.929</td><td>34.03 / 0.961</td><td>33.51 / 0.960</td><td>32.01 / 0.948</td><td>3125</td><td>20.9</td><td>66.9</td><td>119</td></tr><tr><td>E-D3DGS [25]</td><td>29.33 / 0.931</td><td>33.19 / 0.957</td><td>33.25 / 0.957</td><td>29.72 / 0.935</td><td>33.55 / 0.963</td><td>33.55 / 0.963</td><td>32.10 / 0.951</td><td>180</td><td>0.23</td><td>33.5</td><td>85</td></tr><tr><td>Ex4DGS [24]</td><td>28.79 / 0.915</td><td>33.23 / 0.947</td><td>33.73 / 0.948</td><td>29.29 / 0.917</td><td>33.91 / 0.956</td><td>33.69 / 0.959</td><td>32.11 / 0.940</td><td>268</td><td>0.38</td><td>7.22</td><td>121</td></tr><tr><td rowspan="10">Online</td><td>StreamRF† [9] 27.77 / 0.885</td><td></td><td>31.54 / 0.934</td><td>31.74 / 0.937</td><td>28.69 / 0.892</td><td>32.18 / 0.943</td><td>32.29 / 0.945</td><td>30.70 / 0.923</td><td></td><td>7.64</td><td>15.2</td><td>12</td></tr><tr><td>3DGStream [15]</td><td>27.75 / 0.917</td><td>32.22 / 0.957</td><td>33.67 / 0.957</td><td>28.61 / 0.924</td><td>33.47 / 0.966</td><td>33.39 / 0.965</td><td>31.69 / 0.948</td><td>326</td><td>8.14</td><td>8.11</td><td>245</td></tr><tr><td></td><td>28.79 / 0.923</td><td>32.81 / 0.951</td><td>33.03 / 0.954</td><td>28.49 / 0.921</td><td>33.58 / 0.957</td><td>33.89 / 0.965</td><td>31.76 / 0.945</td><td>310</td><td>0.49</td><td>41.2</td><td>92</td></tr><tr><td>4DGC [19] HiCoM [16]</td><td>28.06 / 0.914</td><td>33.28 / 0.954</td><td>33.66 / 0.956</td><td>28.81 / 0.928</td><td>33.62 / 0.963</td><td>33.71 / 0.962</td><td>31.86 / 0.946</td><td>262</td><td>0.63</td><td>6.25</td><td>372</td></tr><tr><td>QUEEN-s [12]</td><td>28.12 / 0.910</td><td>33.25 / 0.954</td><td>33.41 / 0.956</td><td>28.80 / 0.923</td><td>34.01 / 0.960</td><td>33.86 / 0.963</td><td>31.91 / 0.944</td><td>312</td><td>0.73</td><td>5.58</td><td>433</td></tr><tr><td>QUEEN-1 [12]</td><td>28.27 / 0.913</td><td>33.49 / 0.955</td><td>33.87 / 0.957</td><td>29.35 / 0.926</td><td>34.11 / 0.961</td><td>33.97 / 0.962</td><td>32.18 / 0.946</td><td>315</td><td>0.79</td><td>7.43</td><td>417</td></tr><tr><td>StreamSTGS [41]</td><td>28.83 / 0.913</td><td>33.35 / 0.953</td><td>34.03 / 0.956</td><td>29.68 / 0.924</td><td>33.76 / 0.957</td><td>34.47 / 0.959</td><td>32.35 / 0.944</td><td>155</td><td>0.17</td><td>25.5</td><td>199</td></tr><tr><td>ReCon-GS† [29]</td><td>30.14/0.938</td><td>33.54 /0.961</td><td>33.92 / 0.966</td><td>30.43/ 0.938</td><td>34.00 / 0.969</td><td>33.91 / 0.966</td><td>32.66/ /0.956</td><td>244</td><td>0.44</td><td>6.40</td><td>250</td></tr><tr><td>S2GS-fast (Ours)</td><td>29.87 / 0.931</td><td>33.47 / 0.956</td><td>33.70 / 0.957</td><td>30.10 / 0.938</td><td>33.86 / 0.961</td><td>33.45 / 0.961</td><td>32.41 / 0.951</td><td>101</td><td>0.12</td><td>2.26</td><td>484</td></tr><tr><td>S2GS-full (Ours)</td><td>30.07 / 0.933</td><td>33.96 / 0.958</td><td>34.12 / 0.958</td><td>30.26 / 0.939</td><td>34.04 / 0.962</td><td>34.08 / 0.962</td><td>32.76 / 0.952</td><td>101</td><td>0.11</td><td>4.12</td><td>482</td></tr></table>

TABLE II

QUANTITATIVE COMPARISON ON MEET ROOM AND ENERF OUTDOOR DATASETS. FOR ALL METHODS, METRICS ARE AVERAGED ACROSS ALL FRAMES. RESULTS FOR METHODS MARKED WITH <sup>†</sup> ARE TAKEN FROM THE ORIGINAL PAPERS. ALL EXPERIMENTS ARE CONDUCTED ON AN NVIDIA RTX 4090 GPU. RED AND BLUE INDICATE THE BEST AND SECOND BEST, RESPECTIVELY, WITHIN EACH CATEGORY.
<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td colspan="7">Meet Room Dataset LPIPS #Gaussians</td><td colspan="7">ENeRF Outdoor Dataset</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>↓</td><td>(k)↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>↓</td><td>LPIPS #Gaussians (k)↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td rowspan="9">Online</td><td>3DGStream [15]</td><td>29.30</td><td>0.942</td><td>0.188</td><td>185</td><td>4.10</td><td>4.77</td><td>260</td><td>25.91</td><td>0.856</td><td>0.129</td><td>2182</td><td>8.77</td><td>12.96</td><td>117</td></tr><tr><td>4DGC [19]</td><td>30.41</td><td>0.947</td><td>0.180</td><td>208</td><td>0.77</td><td>24.52</td><td>155</td><td>24.52</td><td>0.845</td><td>0.135</td><td>1275</td><td>0.93</td><td></td><td>102</td></tr><tr><td>HiCoM [16]</td><td>29.57</td><td>0.944</td><td>0.182</td><td>152</td><td>0.39</td><td>3.91</td><td>377</td><td>22.12</td><td>0.749</td><td>0.280</td><td>761</td><td>1.74</td><td>37.71 4.43</td><td>385</td></tr><tr><td>QUEEN-s [12]</td><td>31.95</td><td>0.960</td><td>0.159</td><td>133</td><td>0.23</td><td>2.92</td><td>414</td><td>26.21</td><td>0.839</td><td>0.177</td><td>282</td><td>0.95</td><td>2.94</td><td>411</td></tr><tr><td>QUEEN-1 [12]</td><td>32.21</td><td>0.962</td><td>0.152</td><td>135</td><td>0.29</td><td>3.51</td><td>397</td><td>26.13</td><td>0.843</td><td>0.154</td><td>354</td><td>1.32</td><td>3.82</td><td>358</td></tr><tr><td>StreamSTGS [41]</td><td>27.15</td><td>0.914</td><td>0.231</td><td>84</td><td>0.13</td><td>9.19</td><td>220</td><td>25.83</td><td>0.759</td><td>0.193</td><td>164</td><td>0.22</td><td>13.67</td><td>191</td></tr><tr><td>ReCon-GS† [29]</td><td>30.84</td><td>0.955</td><td>0.163</td><td>=</td><td>0.30</td><td>3.86</td><td>256</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S2GS-fast (Ours)</td><td>32.20</td><td>0.963</td><td>0.149</td><td>64</td><td>0.10</td><td>1.13</td><td>498</td><td>26.07</td><td>0.861</td><td>0.125</td><td>138</td><td>0.25</td><td>1.22</td><td>539</td></tr><tr><td>S2GS-full (Ours)</td><td>32.46</td><td>0.964</td><td>0.147</td><td>64</td><td>0.08</td><td>1.96</td><td>513</td><td>25.94</td><td>0.863</td><td>0.124</td><td>138</td><td>0.20</td><td>2.28</td><td>541</td></tr></table>

## E. RQ-2 (Efficiency)

1) Model Complexity: As shown in Tabs. I and II, S²GS significantly reduces the number of Gaussian primitives required for FVV streaming, thereby lowering model complexity during both training and rendering. These results highlight the effectiveness of the proposed streaming octree representation, which reorganizes dynamic scenes using a hierarchical and compact Gaussian distribution.

2) Storage: By reducing model complexity, our method substantially lowers the storage cost required for streaming FVV transmission over IoT infrastructure. Moreover, the structured gating mechanism further compresses storage usage by selectively learning and storing sparse residual updates and discrete gates. As shown in Tabs. I and II, S²GS consistently outperforms existing quantization-based approaches [12], [19] across all evaluated datasets, highlighting its superior storage efficiency. The storage cost trend in Fig. 6 further demonstrates that, compared with QUEEN [12], our method achieves lower storage consumption for both the initial key frame and subsequent online streaming frames.

3) Online Training and Rendering: Another key contribution of S²GS lies in accelerating online training and rendering for FVV streaming under resource constraints. As reported in Tabs. I and II, our method captures competitive dynamic details while consistently achieving the shortest per-frame training time and the highest rendering throughput. Specifically, on the N3DV dataset, S²GS-full reduces the per-frame training time to under 5 seconds, while S²GS-fast further lowers it to 2.26 seconds, all while maintaining a rendering throughput exceeding 480 FPS. The performance trajectories in Fig. 6 further show that, compared with QUEEN [12], our method consistently reduces per-frame optimization time and improves rendering throughput, supporting its practicality for efficient FVV streaming in edge-IoT scenarios.

3DGStream  
4DGC  
HiCoM  
Queen-l  
S<sup>2</sup>GS-full  
GT  
![](images/d1d266e908536992973b7a815a3789370cb79dac4a5557b8a9438b20a37859a5.jpg)  
Fig. 5. Qualitative comparison with SOTA methods [12], [15], [16], [19] on diverse datasets [9], [10], [42]. Our method captures fine-grained details presented in indoor and outdoor scenes, particularly for objects with intricate motions such as poured liquid, hands, knives, and stuffed toys.

![](images/ec487e4ee1db54f260182125c10fd2e7abd15aff4b8d392392dce9850825b07f.jpg)

![](images/a91e49a95d9ca390810378d5e5d4cc91e3792522e3a001daa2bfb217be3010da.jpg)

![](images/22948a10527e8215ed64a7022a20e93ea46c9ebe290f79718e5712489685c46a.jpg)

![](images/bd588ae9cdbdd8660e260a41fd400f30fdd665d78ad6446f8cd6ba98c0969468.jpg)  
Fig. 6. Trend comparison of PSNR, storage cost, training time, and rendering throughput for S²GS-full and QUEEN-l on the Vrheadset scene from the Meet Room dataset. The triangle (▲) denotes the metric value at the first frame.

TABLE III  
ABLATION STUDY ON THE N3DV DATASET. RED AND BLUE INDICATE THE BEST AND SECOND BEST, RESPECTIVELY, WITHIN EACH CATEGORY.
<table><tr><td>Incremental Design</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates ↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td>Baseline</td><td>29.61</td><td>412</td><td>410k</td><td>3.09</td><td>13.7</td><td>199</td></tr><tr><td>+ STE</td><td>31.89</td><td>352</td><td>107k</td><td>2.37</td><td>12.9</td><td>221</td></tr><tr><td>+ Sparse Reg.</td><td>32.25</td><td>331</td><td>16k</td><td>1.10</td><td>12.7</td><td>234</td></tr><tr><td>+ Stream. Octree</td><td>32.71</td><td>101</td><td>2.4k</td><td>0.29</td><td>4.24</td><td>485</td></tr><tr><td>+ HFP (S²GS-full)</td><td>32.76</td><td>101</td><td>679</td><td>0.11</td><td>4.12</td><td>482</td></tr></table>

## F. RQ-3 (Ablation Study)

We conduct ablation studies from three perspectives. First, we progressively build the full S²GS framework to evaluate the contribution of each component. Second, we disentangle the effects of the structure and sparsity-related modules to verify whether they are complementary. Third, we validate two key design choices, namely the multi-level STE and the proposed HFP strategy. We use S<sup>2</sup>GS-full as the complete model and a vanilla GS-based FVV model without residual sparsification or the octree structure as the baseline. To ensure controlled comparisons, all ablation variants are optimized using the same fixed budget of 20 epochs per subsequent frame.

1) Component Analysis: As shown in Tab. III, each component contributes positively to the final S²GS framework.

Straight-Through Estimator (STE): Starting from the baseline, introducing STE reduces the number of active gates to approximately 1/4 of the original count, thereby lowering storage requirements and per-frame optimization time. This indicates that STE effectively sparsifies residual updates.

Sparse Regularization: As reported in Tab. III, introducing the regularization term substantially reduces the number of active gates, thereby reducing storage overhead. Moreover, as with STE, rendering quality consistently improves because the regularization helps stabilize residual optimization and suppress excessively dynamic or noisy updates in FVV streaming.

TABLE IV  
2 × 2 FACTORIAL ABLATION STUDY OF STRUCTURE AND SPARSITY. STRUCTURE DENOTES THE PROPOSED STREAMING OCTREE WITH HFP, AND SPARSITY DENOTES STE WITH SPARSE REGULARIZATION. RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td colspan="2">Structure</td><td colspan="2">Sparsity</td><td colspan="6">MeetRoom Dataset</td><td colspan="6">ENeRF Outdoor Dataset</td></tr><tr><td>Stream. Octree</td><td>HFP</td><td>STE</td><td>Sparse Reg.</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td></td><td></td><td></td><td></td><td>31.69</td><td>129</td><td>129k</td><td>0.67</td><td>2.86</td><td>383</td><td>25.12</td><td>439</td><td>398k</td><td>4.17</td><td>4.03</td><td>285</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>31.86</td><td>127</td><td>13k</td><td>0.22</td><td>2.73</td><td>387</td><td>25.69</td><td>382</td><td>51k</td><td>1.41</td><td>3.66</td><td>299</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>32.33</td><td>64</td><td>17k</td><td>0.38</td><td>2.18</td><td>496</td><td>25.80</td><td>138</td><td>39k</td><td>1.77</td><td>2.45</td><td>532</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>32.46</td><td>64</td><td>401</td><td>0.08</td><td>1.96</td><td>513</td><td>25.94</td><td>138</td><td>1.9k</td><td>0.20</td><td>2.28</td><td>541</td></tr></table>

TABLE V

PROPAGATION STRATEGY ANALYSIS ON THE ENERF OUTDOOR DATASET. RED AND BLUE INDICATE THE BEST AND SECOND BEST, RESPECTIVELY. FT DENOTES FIXED-THRESHOLD PRUNING.
<table><tr><td>Configuration</td><td>PSNR (dB)↑</td><td>LPIPS ↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td>HFP (S²GS-full)</td><td>25.94</td><td>0.124</td><td>1.9k</td><td>0.20</td><td>2.28</td><td>541</td></tr><tr><td> $\mathrm { F T } , \tau = 0 . 5 0$ </td><td>25.82</td><td>0.128</td><td>1.8k</td><td>0.19</td><td>2.35</td><td>537</td></tr><tr><td> $ { \mathrm { F T } } , \tau = 0 . 8 0$ </td><td>25.72</td><td>0.133</td><td>1.6k</td><td>0.18</td><td>2.39</td><td>526</td></tr><tr><td> $ { \mathrm { F T } } , \tau = 0 . 9 0$ </td><td>25.33</td><td>0.141</td><td>1.2k</td><td>0.15</td><td>2.37</td><td>539</td></tr><tr><td> $ { \mathrm { F T } } , \tau = 0 . 9 5$ </td><td>25.12</td><td>0.150</td><td>707</td><td>0.13</td><td>2.32</td><td>544</td></tr></table>

TABLE VI

COMPARISON OF DISCRETE GATING STRATEGIES ON THE N3DV DATASET. RED INDICATES THE BEST RESULT WITHIN EACH CATEGORY. ML DENOTES THE MULTI-LEVEL SCHEME.
<table><tr><td>Configuration</td><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>#Active Gates ↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td>STE: Binary</td><td>29.85</td><td>0.934</td><td>2</td><td>0.06</td><td>4.05</td><td>478</td></tr><tr><td>STE: ML (S²GS-full)</td><td>32.76</td><td>0.952</td><td>679</td><td>0.11</td><td>4.12</td><td>482</td></tr></table>

Streaming Octree: The proposed streaming octree substantially reduces model complexity by globally reorganizing Gaussian residuals into a compact hierarchy. This design reduces the number of optimized Gaussians and active gates, thereby reducing storage footprints and per-frame optimization cost. Meanwhile, it improves visual quality by capturing multiscale and fine-grained dynamic details. For deeper analysis of the octree structure, please check Appendix D for details.

Hierarchical Feature Propagation (HFP): To assess its impact on gate learning, we ablate the HFP module. As reported in Tab. III, HFP reduces both storage cost and training time. By propagating features hierarchically, gate learning is performed on a smaller and more stable set of root-level regions rather than leaf-level Gaussians. Consequently, as shown in Fig. 7, this design yields nearly an order-of-magnitude fewer active gates, sparser residual updates (1.69% vs. 2.46%), and more robust dynamic region identification with reduced noise.

2) Complementary Effects of Structure and Sparsity: We conduct a 2×2 factorial ablation study to evaluate the complementary effects of structural modeling (streaming octree with HFP) and sparsity modeling (STE with sparse regularization). As shown in Table IV, sparsity alone substantially reduces active gates and storage while slightly improving PSNR, whereas structure alone reduces model complexity and substantially improves PSNR and rendering throughput. When combined, the two factors produce synergistic gains: extremely sparse active gates, minimal storage and training time, and maintained or modestly improved PSNR. These results confirm that structured representation and sparse residual updates are complementary rather than redundant: the former enhances hierarchical propagation and representation efficiency, while the latter suppresses unnecessary updates and enforces sparsity in temporal dynamics.

![](images/db54653825eb4cf0bb45ae293d25665960121d836b4f12f4699e7c8fc1da7e6a.jpg)

![](images/e4813bf2d2e05ed6935c42984a4a37d657ce9b84dc040e1a1abd642e5d7b1c3f.jpg)

(a) Gate Histogram (Left: w/o HFP, Right: w/ HFP)  
![](images/36488e2654ece085881f775da6de78b27251327b1b69980bbf4898f9b7a2b944.jpg)  
(b) Active Gate Render (Left: w/o HFP, Right: w/ HFP)  
Fig. 7. Effect of hierarchical feature propagation (HFP) on the Flame Salmon scene. (a) With HFP, the number of sparse active gates is reduced by nearly one order of magnitude at a lower ratio. (b) HFP effectively identifies dynamic regions while suppressing noise.

3) Additional Design Analysis: We conduct additional experiments to validate the design choices of HFP and STE.

Propagation Strategy Analysis: Tab. V compares the proposed HFP with fixed-threshold (FT) pruning under different threshold values τ. HFP achieves the best overall reconstruction quality. In contrast, FT-based variants exhibit a clear degradation in reconstruction quality as τ increases, especially in highly dynamic regions (see Fig. 8 for qualitative results). These observations suggest that FT-based propagation tends to remove informative dynamic responses, especially those associated with weak but meaningful local motion. In contrast, HFP preserves effective activations (1.9k active gates) and consistently achieves better fidelity than FT settings, indicating that the HFP-based S²GS framework remains sensitive to complex and salient local dynamics. The implementation details of the FT-based method can be found in Appendix G.

TABLE VII  
QUANTITATIVE COMPARISON ON THE JETSON AGX ORIN WITH THE MULTI-RESOLUTION N3DV DATASET. ENERGY PER FRAME IS COMPUTED AS AVERAGE POWER MULTIPLIED BY PER-FRAME TRAINING TIME. RED AND BLUE INDICATE THE BEST AND SECOND BEST, RESPECTIVELY.
<table><tr><td rowspan="2">Resolution</td><td rowspan="2">Methods</td><td colspan="7">Quality &amp; Efficiency</td><td colspan="5">Resource Consumption</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>#Gaussians (k)↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td><td>CPU (%)↓</td><td>GPU MEM (GB) ↓</td><td>RAM (GB)↓</td><td>Power (W)↓</td><td>Energy (J)↓</td></tr><tr><td rowspan="4">1/4</td><td>3DGStream [15]</td><td>33.21</td><td>0.953</td><td>0.095</td><td>386</td><td>8.21</td><td>63.28</td><td>21.22</td><td>38.23</td><td>1.82</td><td>0.78</td><td>14.56</td><td>921.36</td></tr><tr><td>HiCoM [16]</td><td>31.25</td><td>0.950</td><td>0.085</td><td>194</td><td>0.49</td><td>45.61</td><td>24.81</td><td>208.0</td><td>1.53</td><td>1.34</td><td>13.53</td><td>617.10</td></tr><tr><td>QUEEN-s [12]</td><td>32.40</td><td>0.955</td><td>0.085</td><td>260</td><td>0.60</td><td>32.27</td><td>32.35</td><td>25.86</td><td>1.91</td><td>1.29</td><td>14.29</td><td>461.14</td></tr><tr><td>S2GS-edge (Ours)</td><td>33.07</td><td>0.958</td><td>0.083</td><td>82</td><td>0.14</td><td>18.61</td><td>49.73</td><td>41.98</td><td>1.42</td><td>1.32</td><td>11.21</td><td>208.62</td></tr><tr><td rowspan="4">1/8</td><td>3DGStream [15]</td><td>32.48</td><td>0.952</td><td>0.063</td><td>397</td><td>8.17</td><td>38.90</td><td>28.77</td><td>40.64</td><td>1.71</td><td>0.74</td><td>15.55</td><td>604.90</td></tr><tr><td>HiCoM [16]</td><td>31.24</td><td>0.956</td><td>0.051</td><td>136</td><td>0.34</td><td>21.70</td><td>46.53</td><td>227.7</td><td>1.13</td><td>1.10</td><td>13.31</td><td>288.83</td></tr><tr><td>QUEEN-s [12]</td><td>32.90</td><td>0.967</td><td>0.038</td><td>202</td><td>0.48</td><td>17.04</td><td>52.06</td><td>44.70</td><td>2.17</td><td>1.26</td><td>14.08</td><td>239.92</td></tr><tr><td>S2GS-edge (Ours)</td><td>33.26</td><td>0.968</td><td>0.036</td><td>50</td><td>0.10</td><td>8.622</td><td>67.94</td><td>66.74</td><td>1.08</td><td>1.24</td><td>11.18</td><td>96.39</td></tr><tr><td rowspan="4">1/16</td><td>3DGStream [15]</td><td>31.17</td><td>0.944</td><td>0.065</td><td>407</td><td>8.16</td><td>35.29</td><td>30.56</td><td>42.13</td><td>1.72</td><td>0.73</td><td>15.41</td><td>543.82</td></tr><tr><td>HiCoM [16]</td><td>31.65</td><td>0.958</td><td>0.047</td><td>103</td><td>0.26</td><td>18.72</td><td>49.11</td><td>184.5</td><td>0.89</td><td>1.25</td><td>12.53</td><td>234.56</td></tr><tr><td>QUEEN-s [12]</td><td>33.44</td><td>0.977</td><td>0.020</td><td>176</td><td>0.43</td><td>10.23</td><td>61.06</td><td>53.19</td><td>1.22</td><td>1.15</td><td>14.48</td><td>148.13</td></tr><tr><td>SGS-edge (Ours)</td><td>33.57</td><td>0.978</td><td>0.018</td><td>24</td><td>0.06</td><td>5.612</td><td>86.66</td><td>87.43</td><td>0.92</td><td>1.17</td><td>11.46</td><td>64.31</td></tr></table>

HFP (S <sup>2</sup>GS-full)  
![](images/8c45e057336be701fd358b9202975518decfc14b22589927e89281661aa53d4d.jpg)  
Fig. 8. Qualitative comparison of different propagation strategies. FT-based variants progressively blur high-motion areas as τ increases, whereas HFP preserves fine details and sharpness. Results indicate that HFP effectively retains informative dynamic responses, particularly in regions with subtle but meaningful local motion.

Binary STE vs. Multi-Level (ML) STE: Tab. VI compares two discrete gating strategies, namely binary STE and the proposed multi-level STE, within the same S²GS framework. Binary STE significantly degrades reconstruction quality, reducing PSNR/SSIM from 32.76/0.952 to 29.85/0.934. The average number of active gates per frame is 2, indicating a severe collapse to an excessively sparse solution that suppresses necessary residual updates, as illustrated in Fig. 4. In contrast, multi-level STE provides finer discrete granularity, which better balances sparsity and reconstruction fidelity. As a result, it preserves a richer set of effective updates (679 active gates) and achieves markedly better visual quality with nearly unchanged training and rendering cost.

## G. RQ-4 (Deployability)

We conduct deployment experiments on the NVIDIA Jetson AGX Orin Developer Kit, a leading industrial edge-IoT platform. We assess visual quality, efficiency, and resource consumption of four baseline models on the multi-resolution N3DV dataset, as summarized in Tab. VII. Resource usage metrics are collected using the jtop monitoring tool on Jetson [43], including averaged CPU utilization, GPU memory usage, RAM usage, and power consumption. We further compute the reconstruction energy per frame as the average power consumption multiplied by the per-frame training time. This metric provides a practical estimate of the dominant energy cost of streaming reconstruction, as the measured processing time is dominated by per-frame optimization.

Several key observations can be drawn. Quality: S²GS-edge consistently achieves higher visual quality by adaptively adjusting the octree resolution, demonstrating robustness in FVV streaming with multi-resolution inputs. Timeliness: S²GS-edge significantly outperforms other efficient baselines in both perframe training time and rendering throughput. Specifically, it achieves the per-frame training time below 6 seconds and rendering throughput above 85 FPS on the N3DV dataset at 1/16 resolution, satisfying fast training and real-time rendering requirements. Resource Consumption: Thanks to its reduced model complexity and efficient per-frame training, S²GS-edge exhibits the lowest power consumption, per-frame energy cost, and GPU memory footprint, while maintaining competitive RAM usage compared to other baselines. In summary, S²GSedge demonstrates strong visual quality, high timeliness, and low resource consumption, making it well-suited for resourceconstrained edge-IoT environments.

## H. RQ-5 (Case Study)

We develop an FVV streaming system based on a telerobotic platform, as illustrated in Fig. 9. Immersive digital twins of human operators are telepresented online to record and demonstrate the operation procedures of robotic arms.

1) Physical Testbed: The physical testbed is designed to assess whether S<sup>2</sup>GS can support FVV streaming under realworld acquisition and resource constraints. The FVV capture setup consists of 9 industrial cameras connected via 3 USB 3.0 hubs. Multi-view videos are captured at a resolution of 320 $\times 2 4 0$ and synchronized using ROS 2 on a cloud server. The center camera is selected as the test view, and the first frame is processed using COLMAP [34], [35] to estimate the initial point clouds and camera poses. Based on these initializations, the FVV streaming stage, including per-frame optimization and rendering, is performed on the NVIDIA Jetson AGX Orin.

![](images/e81c6e6a71071e2d17b279eb88db312d13f6544f0cc9f3e7abebaf49af5ce8ac.jpg)  
(a) FVV Capture Setup

![](images/7e9d066f95a136e6f50a6735a3aaf993ad68f03d2c5fdba2a012484402745cc5.jpg)  
(b) Telerobotic System

![](images/2c51dce322f5ea3ecea848d27f658b15bf3c546f1ecca47dec8e3bf5b0c30cab.jpg)  
(c) On-device Results

Fig. 9. Physical testbed with IoT application of the proposed method.  
![](images/d1547053db44a371bf21d854e845787898fb99383e36e4cf461915085c431701.jpg)  
Fig. 10. Qualitative comparison with SOTA methods [12], [15], [16] on the physical testbed. $\mathrm { S ^ { 2 } G S }$ -edge achieves clearer visual details, demonstrating its practical feasibility for edge-assisted IoT applications.

2) Experimental Results: The results in Fig. 9(c) and Fig. 10 demonstrate the practical behavior of $S ^ { 2 } \mathrm { G } S .$ -edge. On the physical testbed, $\mathrm { S ^ { 2 } G S }$ -edge achieves higher visual quality than the baseline methods while maintaining a compact representation with 88k Gaussians and 0.26 MB storage. In terms of efficiency, $S ^ { 2 } G S .$ -edge reduces the per-frame optimization time to 4.60 s and reaches a rendering throughput of 59.37 FPS on the edge device, with low GPU memory usage and power consumption. These results suggest that the proposed framework can be effectively performed on resource-constrained edge hardware for efficient FVV streaming. Overall, this case study provides proof-of-concept evidence for the practical feasibility of $\mathrm { S ^ { 2 } G S }$ in a real-world edge-IoT prototype.

## V. CONCLUSION

In this paper, we propose $S ^ { 2 } \mathrm { G } S ,$ , a structured sparse Gaussian streaming framework for efficient FVV reconstruction. $\mathrm { S ^ { 2 } G S }$ organizes temporal residuals with a streaming octree and learns sparse residual updates through a differentiable structured gating mechanism. This design reduces storage overhead and per-frame optimization cost while maintaining competitive visual quality. Extensive experiments on benchmark datasets, a consumer GPU, an edge IoT device, and a physical testbed demonstrate the effectiveness and practical feasibility of $\mathrm { S ^ { 2 } G S }$

Nevertheless, $\mathrm { S ^ { 2 } G S }$ still relies on the quality of firstframe reconstruction and adopts a fixed streaming hierarchy, which limits its adaptability in scenarios involving severe occlusion/disocclusion and large out-of-support motions, as discussed in Appendix E and J. In addition, its robustness under practical industrial disturbances, such as camera desynchronization [44] and sparse views [45], remains to be evaluated. Future work will explore hierarchy refresh, keyframebased refinement [30], and systematic stress testing to enhance robustness for real-world deployment on edge-IoT systems.

## APPENDIX A SUPPLEMENTAL VIDEOS

We provide supplemental videos at this URL to facilitate assessment of the visual quality of the proposed method. The video provides high-resolution FVV streaming results that are difficult to characterize using quantitative metrics alone. Specifically, the supplemental video includes:

1) Streaming FVV reconstruction results of $\mathrm { S ^ { 2 } G S { - } f u l l }$ and $\mathrm { S ^ { 2 } G S { - } f a s t }$ on diverse dynamic scene datasets, including N3DV [10], MeetRoom [9], and ENeRF Outdoor [42].

2) Results of $\mathrm { S ^ { 2 } G S  – e d g e }$ on the real-world physical testbed.

3) Representative failure cases of $\mathrm { { S ^ { 2 } G S } }$ on more challenging outdoor scenes of the Campus dataset [46].

## APPENDIX B IMPLEMENTATION DETAILS

For all self-measured baselines, we used the official implementations and the optimization settings recommended by their authors. We held the dataset splits, input resolutions, training and test views, camera parameters, and hardware platform constant across all self-measured methods. For initialization, we followed the official protocol of each method. For methods with the same initialization pipelines, such as QUEEN [12] and $\mathrm { S ^ { 2 } G S }$ , we used the same first-frame COLMAP [34], [35] reconstruction and camera parameters. Results taken from original papers are included only as reference values; they are not part of the strictly controlled comparisons under matched hardware and computational budgets.

In addition, to improve transparency and reproducibility, Tab. VIII reports the exact hyperparameter configuration used for each evaluated scene. For the sensitivity analyses of both structural parameters and sparse regularization weights, please refer to Appendix D and Appendix H, respectively.

## APPENDIX C

## DEFINITION AND STATISTICAL ANALYSIS OF STRUCTURED SPARSITY

In $\mathrm { S ^ { 2 } G S }$ , sparsity is defined by the fraction of candidate Gaussian residual updates that remain active at each streaming frame after gate discretization. A Gaussian primitive is considered active if and only if the discretized gate associated with its structured region is non-zero. We quantify this property using the ratio of active Gaussian primitives to the total number of candidate Gaussians, where a lower active ratio indicates a sparser residual-update pattern. Structured sparsity further requires these active residual updates to follow the spatial hierarchy of the scene. Accordingly, active residual updates are organized into spatially correlated regions and controlled by a small number of structured gates, rather than being selected independently at the primitive level.

TABLE VIII  
PER-SCENE HYPERPARAMETER CONFIGURATIONS. THE PHYSICAL TESTBED AND CAMPUS DATASET ARE ADDITIONAL EVALUATIONS REPORTED IN THE APPENDIX.
<table><tr><td>Dataset</td><td>Scene</td><td>Root Resolution</td><td>Log Base</td><td> $\lambda _ { 1 }$ </td><td> $\lambda _ { 2 }$ </td></tr><tr><td>N3DV</td><td>Coffee Martini</td><td> $2 ^ { 1 2 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>N3DV</td><td>Cook Spinach</td><td> $2 ^ { 9 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>N3DV</td><td>Cut Roasted Beef</td><td> $2 ^ { 9 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>N3DV</td><td>Flame Salmon</td><td> $2 ^ { 1 2 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>N3DV</td><td>Flame Steak</td><td> $2 ^ { 9 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>N3DV</td><td>Sear Steak</td><td> $2 ^ { 9 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>MeetRoom</td><td>Discussion</td><td> $2 ^ { 8 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>MeetRoom</td><td>Trimming</td><td> $2 ^ { 8 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>MeetRoom</td><td>Vrheadset</td><td> $2 ^ { 8 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>ENeRF Outdoor</td><td>Actor1_4</td><td> $2 ^ { 9 }$ </td><td>1.2</td><td>0.01</td><td>0.01</td></tr><tr><td>ENeRF Outdoor</td><td>Actor2_3</td><td> $2 ^ { 9 }$ </td><td>1.2</td><td>0.01</td><td>0.01</td></tr><tr><td>ENeRF Outdoor</td><td>Actor5_6</td><td> $2 ^ { 9 }$ </td><td>1.2</td><td>0.01</td><td>0.01</td></tr><tr><td>Physical Testbed</td><td>Install RAM</td><td> $2 ^ { 1 2 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>Physical Testbed</td><td>Pick Blocks</td><td> $2 ^ { 1 2 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>Physical Testbed</td><td>Unplug Device</td><td> $2 ^ { 1 2 }$ </td><td>1.5</td><td>0.01</td><td>0.01</td></tr><tr><td>Campus</td><td>Children</td><td> $2 ^ { 1 2 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>Campus</td><td>Football</td><td> $2 ^ { 1 2 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr><tr><td>Campus</td><td>Fountain</td><td> $2 ^ { 1 2 }$ </td><td>2.0</td><td>0.01</td><td>0.01</td></tr></table>

Let $\widetilde { g } _ { t , j } \in ( 0 , 1 )$ denote the continuous gate produced by Gumbel-Sigmoid sampling for the j-th structured region at frame t. During the forward pass, it is discretized as:

$$
\bar { g } _ { t , j } = \mathrm { M L - S T E } ( \widetilde { g } _ { t , j } ) \in \{ 0 . 0 , 0 . 1 , \ldots , 0 . 9 , 1 . 0 \} ,\tag{11}
$$

Therefore, the gates applied to residual updates in the forward pass are discrete rather than continuous. Let $N _ { t }$ and $R _ { t }$ denote the numbers of Gaussian primitives and structured gates at frame t, respectively, and let idx(i) map Gaussian i to its associated structured gate. The gated residual update of Gaussian i is calculated as:

$$
\widehat { \Delta \mathbf { r } } _ { t , i } = \bar { g } _ { t , \mathrm { i d x } ( i ) } \Delta \mathbf { r } _ { t , i } .\tag{12}
$$

Accordingly, a Gaussian is considered active when its associated gate is non-zero. We define the active Gaussian set as:

$$
\begin{array} { r } { \mathcal { A } _ { t } ^ { \mathrm { G } } = \left\{ i \in \left\{ 1 , \dots , N _ { t } \right\} \big | \bar { g } _ { t , \mathrm { i d x } ( i ) } > 0 \right\} . } \end{array}\tag{13}
$$

The Active Gaussian Ratio is then defined as:

$$
\rho _ { \mathrm { g a u s s i a n } , t } = \frac { \left| \mathcal { A } _ { t } ^ { \mathrm { G } } \right| } { N _ { t } } .\tag{14}
$$

This ratio measures primitive-level sparsity: a lower $\rho _ { \mathrm { A G } , t }$ indicates that fewer Gaussian primitives receive residual updates. Similarly, the active gate set is defined as:

$$
\begin{array} { r } { A _ { t } ^ { \mathrm { S } } = \left\{ j \in \left\{ 1 , \dots , R _ { t } \right\} \vert \bar { g } _ { t , j } > 0 \right\} . } \end{array}\tag{15}
$$

We define the Gate-to-Gaussian Ratio as:

$$
\rho _ { \mathrm { g a t e } , t } = \frac { \left| \mathcal { A } _ { t } ^ { \mathrm { S } } \right| } { \left| \mathcal { A } _ { t } ^ { \mathrm { G } } \right| } .\tag{16}
$$

This ratio quantifies the extent to which active residual updates are grouped by structured gates. For a given number of active

Gaussians, a lower $\rho _ { \mathrm { g a t e } , t }$ indicates that fewer structured gates are active and that each active gate covers a larger group of Gaussian primitives. Therefore, a lower ratio reflects a stronger region-wise organization of residual updates.

The two ratios therefore capture complementary aspects of structured sparsity. The Active Gaussian Ratio measures the size of the primitive-level residual-update support, and the Gate-to-Gaussian Ratio measures how compactly this support is organized at the region level. Importantly, whether a residual update is active depends only on whether $\bar { g } _ { t , j } > 0$ . All nonzero gates are treated equally as active when measuring sparsity, while their specific values determine only the scales of the corresponding residual updates. This formulation therefore separates the selection of active residual updates from the control of their update strengths.

In addition, we also conduct statistical analysis to provide evidence for structured sparsity in residual updates. Specifically, as shown in the Supplementary Msssssaterial, across the five evaluated datasets, the dataset-level ratio of active Gaussians ranges from 1.99% to 12.94%, showing that temporal residual updates involve a small subset of the Gaussian primitives. Moreover, active gates account for only 9.90% to 23.94% of the active Gaussians, indicating that these updates are spatially concentrated within a limited number of hierarchical regions. The resolution analysis further shows that this pattern persists from $1 / 2$ to $1 / 1 6$ input resolution, although the degree of sparsity varies with scene dynamics and representation granularity. These results provide direct quantitative support for structured sparsity across diverse evaluation settings.

## APPENDIX D STREAMING OCTREE REPRESENTATION

We conduct additional experiments to evaluate the effects of the streaming octree. In this representation, the octree is built within the global spatial bounds of the first-frame root Gaussian primitives. Candidate anchors are initialized at voxel centroids across the multi-resolution hierarchy [32]. The position of initial Gaussians is calculated as:

$$
\pmb { \mu } _ { i } = \pmb { h } _ { x } + \pmb { o } _ { i } \cdot \pmb { f } _ { x } ,\tag{17}
$$

where $\pmb { h } _ { x } \in \mathbb { R } ^ { 3 }$ is the position of the anchor point x associated with Gaussian i, $\ b { f _ { x } } \in \mathbb { R } ^ { 3 }$ represents the anchor-dependent scaling factor. The variable $\pmb { \mu } _ { i }$ specifies the position of the ith Gaussian primitive, while ${ \pmb o } _ { i } \in \mathbb { R } ^ { 3 }$ denotes its offset vector.

The streaming octree improves $\mathrm { S ^ { 2 } G S }$ through two complementary mechanisms: hierarchical residual allocation (HRA) and efficient streaming query (ESQ). HRA organizes temporally sparse Gaussian residual updates according to the spatial hierarchy of the scene, so that dynamic changes are modeled as structured regional variations rather than independent primitive-wise updates. This preserves spatial correlations among dynamic Gaussians while reducing redundant residual parameters. ESQ, built upon the fixed octree hierarchy and persistent pre-allocated anchors, further reduces the overhead of residual lookup and avoids repeated online pruning or densification during per-frame optimization.

TABLE IX  
ABLATION STUDY OF THE STREAMING OCTREE IN TERMS OF RECONSTRUCTION QUALITY, PARAMETER EFFICIENCY, AND TRAINING SPEED. RED INDICATES THE BEST RESULT. HRA AND ESQ DENOTE HIERARCHICAL RESIDUAL ALLOCATION AND EFFICIENT STREAMING QUERY, RESPECTIVELY. EXPERIMENTS ARE CONDUCTED WITHOUT THE HFP MODULE.
<table><tr><td colspan="2">Streaming Octree</td><td colspan="5">N3DV Dataset</td><td colspan="5">MeetRoom Dataset</td></tr><tr><td>HRA</td><td>ESQ</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td></tr><tr><td>-</td><td>-</td><td>32.25</td><td>331</td><td>16k</td><td>1.10</td><td>12.7</td><td>31.89</td><td>127</td><td>13k</td><td>0.22</td><td>2.83</td></tr><tr><td>√</td><td>-</td><td>32.74</td><td>122</td><td>2.7k</td><td>0.34</td><td>4.58</td><td>32.40</td><td>71</td><td>2.4k</td><td>0.16</td><td>2.26</td></tr><tr><td>√</td><td>√</td><td>32.71</td><td>101</td><td>2.4k</td><td>0.29</td><td>4.24</td><td>32.44</td><td>64</td><td>2.2k</td><td>0.14</td><td>2.02</td></tr></table>

TABLE X  
EFFECT OF OCTREE ROOT RESOLUTION ON THE MEETROOM DATASET.
<table><tr><td>Resolution</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td> $2 ^ { 7 }$ </td><td>32.39</td><td>35</td><td>154</td><td>0.05</td><td>1.97</td><td>526</td></tr><tr><td> $2 ^ { 8 }$ </td><td>32.46</td><td>64</td><td>401</td><td>0.08</td><td>1.96</td><td>513</td></tr><tr><td> $2 ^ { 9 }$ </td><td>32.59</td><td>89</td><td>828</td><td>0.11</td><td>2.02</td><td>505</td></tr><tr><td> $2 ^ { 1 0 }$ </td><td>32.41</td><td>102</td><td>999</td><td>0.12</td><td>2.04</td><td>498</td></tr><tr><td> $2 ^ { 1 1 }$ </td><td>32.36</td><td>106</td><td>974</td><td>0.12</td><td>2.00</td><td>501</td></tr></table>

TABLE XI  
EFFECT OF ANCHOR SELECTION ON THE MEETROOM DATASET.
<table><tr><td>Log Base</td><td>PSNR (dB)↑</td><td>#Gaussians (k)↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td>1.2</td><td>32.23</td><td>122</td><td>1.2k</td><td>0.16</td><td>2.18</td><td>495</td></tr><tr><td>1.5</td><td>32.46</td><td>64</td><td>401</td><td>0.08</td><td>1.96</td><td>513</td></tr><tr><td>2.0</td><td>32.29</td><td>48</td><td>218</td><td>0.06</td><td>2.02</td><td>501</td></tr><tr><td>2.5</td><td>32.33</td><td>36</td><td>165</td><td>0.05</td><td>1.95</td><td>519</td></tr><tr><td>3.0</td><td>32.20</td><td>28</td><td>117</td><td>0.04</td><td>1.91</td><td>525</td></tr></table>

The results in Tab. IX validate these two effects. On the N3DV dataset, adding HRA improves PSNR from 32.25 dB to 32.74 dB and reduces the number of Gaussians, active gates, storage, and training time from 331k, 16k, 1.10 MB, and 12.7s to 122k, 2.7k, 0.34 MB, and 4.58s, respectively. Adding ESQ further reduces these values to 101k, 2.4k, 0.29 MB, and 4.24s, with only a negligible change in PSNR. Similar trends are observed on the MeetRoom dataset, where the full streaming octree design achieves the best PSNR while using the fewest Gaussians and active gates, the lowest storage cost, and the shortest training time. These results demonstrate that the octree improves reconstruction quality mainly by preserving structured spatial correlations, improves parameter efficiency through compact hierarchical allocation, and accelerates training by reducing both primitive-wise optimization redundancy and query overhead.

In addition to the main ablation, additional analyses are conducted to examine the design choices, including the root resolution, anchor selection, and progressive training.

1) Root Resolution: Root resolution controls the granularity of region-level sparse updates, thereby affecting the trade-off between reconstruction quality and efficiency. As shown in Tab. X, using a coarse root resolution of $2 ^ { 7 }$ yields the most compact representation, with 35k Gaussians, 154 active gates, 0.05 MB storage, and 526 FPS rendering speed, but its PSNR is lower than those of finer-resolution settings. Increasing the resolution to $2 ^ { 9 }$ improves PSNR from 32.39 dB to 32.59 dB, indicating that finer root-level regions can better capture local dynamic variations. However, this also increases the number of Gaussians, active gates, and storage costs. Further increasing the resolution to $2 ^ { 1 \bar { 0 } } \ \mathrm { o r 2 ^ { 1 1 } }$ does not improve reconstruction quality, while introducing higher model complexity and slightly slower rendering. These results show that overly coarse roots may limit local dynamics, whereas overly fine roots introduce redundant parameters and gates. We therefore adopt $2 ^ { 8 }$ on the MeetRoom dataset, which achieves a favorable balance between reconstruction quality, parameter efficiency, training speed, and rendering performance.

2) Anchor Selection: We study the heuristic resolutionbased anchor selection method [32]. It estimates the viewdependent hierarchy level of anchor points as follows:

$$
\hat { h } _ { x y } = \left\lfloor h _ { x y } ^ { \mathrm { s o f t } } \right\rfloor = \left\lfloor \Gamma \left( \log \left( \rho _ { \mathrm { m a x } } / \rho _ { x y } \right) \right) \right\rfloor ,\tag{18}
$$

where $\rho _ { x y }$ is the observation distance from camera viewpoint $y$ to anchor point $x , \rho _ { \mathrm { m a x } }$ is the reference maximum distance of the octree box, Γ(·) is a clamping function to constrain the soft hierarchy level to the valid range, and $\lfloor \cdot \rfloor$ denotes the floor operator. The anchor is selected if its predefined level $h _ { x }$ does not exceed the estimated view-dependent level $\hat { h } _ { x y }$

We conduct an ablation study on the logarithmic base for anchor selection on the MeetRoom dataset. As shown in Tab. XI, the log base controls the trade-off between representation capacity and compactness. A smaller base, such as 1.2, retains more Gaussians and active gates, increasing model complexity without achieving the best reconstruction quality. In contrast, a larger base, such as 3.0, yields the most compact representation and highest rendering speed, but reduces PSNR to 32.20 dB, suggesting insufficient capacity for dynamic details. The base 1.5 achieves the best PSNR while maintaining moderate Gaussians, active gates, storage, and training time. Therefore, we adopt 1.5 as the default log base on the MeetRoom dataset, which provides a favorable balance between reconstruction quality and streaming efficiency.

3) Progressive Training: We further evaluate the effect of progressive training on reconstruction quality and streaming efficiency. Since the streaming octree represents Gaussian residuals using hierarchical levels, optimizing all levels simultaneously may introduce inherent challenges. In $\mathrm { S ^ { 2 } G S }$ we adopt a coarse-to-fine training scheme. Let H denote the total number of residual levels. For each streaming frame, we initially activate residual levels from $\lfloor H / 2 \rfloor$ , and then progressively introduce finer levels during optimization. Since leaf-level residuals are designed to capture fine-grained local dynamics, we allocate more optimization iterations to finer levels. Specifically, if $N _ { i }$ denotes the number of iterations assigned to level i, we set

TABLE XII  
ABLATION STUDY ON THE EFFECT OF PROGRESSIVE TRAINING. RED INDICATES THE BEST RESULT.
<table><tr><td rowspan="2">Configuration</td><td colspan="5">N3DV Dataset</td><td colspan="5">ENeRF Outdoor Dataset</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td></tr><tr><td>S2GS-full (Ours)</td><td>32.76</td><td>0.952</td><td>679</td><td>0.11</td><td>4.12</td><td>25.94</td><td>0.863</td><td>1.9k</td><td>0.20</td><td>2.28</td></tr><tr><td>w/o Prog. Train.</td><td>32.68</td><td>0.951</td><td>755</td><td>0.14</td><td>4.15</td><td>26.02</td><td>0.861</td><td>2.1k</td><td>0.24</td><td>2.33</td></tr></table>

S<sup>2</sup>GS-full  
w/o Prog. Train.  
S<sup>2</sup>GS-full  
GT  
![](images/5f2f3efc3221015e565a02a7446c7dc6e45d7e89afb2497dcc4bce9c93c753ce.jpg)  
Fig. 11. Visualizations of rendered images from $\mathrm { S ^ { 2 } G S } .$ -full and the variant without progressive training. Progressive training facilitates the recovery of dynamic objects with fine-grained motions.

$$
N _ { i + 1 } = \omega N _ { i } , \quad \omega \geq 1 ,\tag{19}
$$

where $\omega$ is the growth factor controlling the relative training budget between adjacent levels. Thus, finer residual levels receive a larger optimization budget than coarser levels.

As shown in Tab. XII, progressive training improves the overall quality and efficiency. On the N3DV dataset, it improves PSNR from 32.68/0.951 to 32.76/0.952, while reducing active gates from 755 to 679, storage from 0.14 MB to 0.11 MB, and training time from 4.15s to 4.12s. On the ENeRF Outdoor dataset, although the non-progressive variant obtains a slightly higher PSNR, progressive training achieves better SSIM and lower storage costs. The qualitative results in Fig. 11 further show that progressive training better preserves finegrained dynamic details. These results indicate that progressive training improves the visual quality and training efficiency.

## APPENDIX E

## FIRST-FRAME INFLUENCE

Streaming FVV reconstruction relies on the first frame for the subsequent residual updates. We therefore analyze the effect of first-frame quality by varying the first-frame optimization budget and adding Gaussian noise during training. Specifically, the noisy first-frame images are generated as:

$$
\tilde { \mathbf { I } } _ { 0 } = \mathrm { c l i p } \left( \mathbf { I } _ { 0 } + \epsilon , 0 , 1 \right) , \quad \epsilon _ { u , v , c } \sim \mathcal { N } \left( 0 , \left( \frac { \sigma } { 2 5 5 } \right) ^ { 2 } \right) ,\tag{20}
$$

where $\mathbf { I } _ { 0 }$ is the normalized first-frame training image, $\tilde { \mathbf { I } } _ { 0 }$ is the corresponding noisy image, and clip(·, 0, 1) restricts pixel values to the valid range [0, 1]. The perturbation ϵ is independently sampled for each pixel (u, v) and color channel c, and σ denotes the noise strength on an 8-bit intensity scale.

TABLE XIII  
EFFECT OF THE FIRST-FRAME TRAINING BUDGET ON THE COFFEE MARTINI SCENE. EPOCHS INDICATE THE NUMBER OF OPTIMIZATION EPOCHS USED FOR THE FIRST FRAME.
<table><tr><td rowspan="2">Epochs</td><td colspan="2">First-Frame Metric</td><td colspan="5">Streaming Performance</td></tr><tr><td>PSNR (dB)↑</td><td>Train (Sec)↓</td><td>PSNR (dB)↑</td><td>LPIPS ↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td>50</td><td>26.92</td><td>9.15</td><td>29.72</td><td>0.148</td><td>0.16</td><td>3.64</td><td>475</td></tr><tr><td>100</td><td>28.18</td><td>19.75</td><td>29.80</td><td>0.139</td><td>0.16</td><td>3.65</td><td>460</td></tr><tr><td>150</td><td>28.93</td><td>27.68</td><td>29.83</td><td>0.135</td><td>0.16</td><td>3.70</td><td>462</td></tr><tr><td>200</td><td>29.26</td><td>37.99</td><td>29.95</td><td>0.134</td><td>0.17</td><td>3.74</td><td>477</td></tr><tr><td>250</td><td>28.96</td><td>46.69</td><td>30.07</td><td>0.133</td><td>0.17</td><td>3.82</td><td>469</td></tr></table>

As shown in Tab. XIII, increasing the budget from 50 to 250 epochs improves streaming PSNR from 29.72 dB to 30.07 dB, while the per-frame streaming cost remains stable. Tab. XIV further shows that $\mathrm { S ^ { 2 } G S }$ remains robust under moderate firstframe degradation. When the noise level increases from $\sigma = 0$ to $\sigma = 2 0$ , the streaming PSNR decreases only slightly from 33.96 dB to 33.55 dB. Under more severe degradation $( \sigma =$ 30), the PSNR drops more noticeably to 33.06 dB, reflecting the inherent dependence of residual-based streaming methods on first-frame quality. Overall, $\mathrm { S ^ { 2 } G S }$ benefits from a reliable first-frame initialization, but does not require it to be error-free. Residual updates, streaming octree, and structure-aware gating jointly help maintain stable streaming performance under firstframe quality degradation.

It should be noted that our design does not remove the inherent dependence of $\mathrm { S ^ { 2 } G S }$ on first-frame quality, but makes the subsequent streaming process more stable and efficient under practical first-frame initialization. In long-term deployment, severe errors in the initial frame or substantial accumulated scene changes may reduce the effectiveness of subsequent residual updates. A practical solution is to periodically refresh the framework using selected keyframes when reconstruction errors increase, or scene changes become significant [17], [30]. This keyframe refresh strategy is orthogonal to $\mathrm { { S ^ { 2 } G S } }$ and can be integrated with the proposed FVV framework to improve robustness in long-term streaming scenarios.

## APPENDIX F

## DYNAMIC CONFIDENCE ESTIMATOR

We estimate the dynamic confidence of each Gaussian based on the viewspace gradient difference (VGD) [12], [17]. Specifically, during FVV streaming, we consider two consecutive frames with dynamic changes. Given the rendered image $\mathcal { T } _ { t - 1 } ^ { r }$ at frame $t - 1$ , we compute the mean squared error (MSE) between $\mathcal { T } _ { t - 1 } ^ { r }$ and the ground-truth images $\boldsymbol { \mathcal { T } _ { t - 1 } ^ { g t } }$ and $\mathcal { T } _ { t } ^ { g t }$ :

Frame 077  
TABLE XIV  
EFFECT OF GAUSSIAN NOISE DEGRADATION DURING FIRST-FRAME TRAINING ON THE COOK SPINACH SCENE.
<table><tr><td rowspan="2">Noise</td><td colspan="2">First-Frame Metric</td><td colspan="5">Streaming Performance</td></tr><tr><td>PSNR (dB)↑</td><td>Train (Sec)↓</td><td>PSNR (dB)↑</td><td>LPIPS ↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td> $\alpha = 0$ </td><td>33.54</td><td>29.91</td><td>33.96</td><td>0.134</td><td>0.07</td><td>4.20</td><td>498</td></tr><tr><td> $\alpha = 1 0$ </td><td>33.30</td><td>30.22</td><td>33.74</td><td>0.135</td><td>0.07</td><td>4.23</td><td>492</td></tr><tr><td> $\alpha = 2 0$ </td><td>30.89</td><td>29.84</td><td>33.55</td><td>0.138</td><td>0.07</td><td>4.21</td><td>489</td></tr><tr><td> $\alpha = 3 0$ </td><td>28.93</td><td>28.86</td><td>33.06</td><td>0.144</td><td>0.07</td><td>4.26</td><td>495</td></tr></table>

Frame 001  
Frame 012  
Frame 025  
Frame 051  
Frame 064  
![](images/127a8e54336efca23b0c5e41e550f16538d8372ce3a7f72eb98231fe09963f75.jpg)  
Fig. 12. Controlled illumination changes on the Cut Roasted Beef scene with a time-varying global gain $g _ { t } \in [ 0 . 5 , { \mathrm { \bar { 1 . 5 } } } ]$

$$
\begin{array} { r } { \mathcal { E } _ { t } = \mathrm { M S E } ( \varmathbb { Z } _ { t } ^ { r - 1 } , \varmathbb { Z } _ { t } ^ { g t } ) , \mathcal { E } _ { t - 1 } = \mathrm { M S E } ( \varmathbb { Z } _ { t - 1 } ^ { r } , \varmathbb { Z } _ { t - 1 } ^ { g t } ) . } \end{array}\tag{21}
$$

Then the VGD $\boldsymbol { m } \in \mathbb { R } ^ { N \times 2 }$ from two consecutive frames is calculated as:

$$
m = \frac { 1 } { V } \sum _ { v = 0 } ^ { V - 1 } \left[ \frac { \partial \mathcal { E } _ { t } ^ { ( v ) } } { \partial \pmb { \mu } _ { t - 1 } ^ { ( v ) } } - \frac { \partial \mathcal { E } _ { t - 1 } ^ { ( v ) } } { \partial \pmb { \mu } _ { t - 1 } ^ { ( v ) } } \right]\tag{22}
$$

where $\pmb { \mu } _ { t - 1 } ^ { ( v ) } \in \mathbb { R } ^ { N \times 2 }$ denotes the projected 2D positions of the Gaussians from the previous frame under view v, and V denotes the number of training views.

We investigate the robustness of the VGD-based dynamic confidence estimator. First, we compare the proposed robust VGD estimator with several alternative confidence estimators, including uniform random confidence, constant confidence assignments, and min-max normalized VGD. As shown in Tab. XV, Constant-Zero and Min-Max VGD produce overly sparse gate decisions, resulting in only 0 or 1 active gates and substantially degraded reconstruction quality. In contrast, Constant-One activates all gates and increases storage and training cost, while providing no quality advantage. Uniform Random achieves reasonable reconstruction quality but requires more active gates and larger storage than our method. Robust VGD achieves the highest PSNR with only 401 active gates and 0.08 MB of storage, demonstrating a more favorable trade-off among reconstruction quality, sparsity, and efficiency. These results indicate that Robust VGD provides more balanced confidence estimation for sparse residual learning.

Additionally, we evaluate the effectiveness of Robust VGD under two representative perturbation factors, namely controlled illumination changes and injected gradient noise.

1) Illumination Changes: We evaluate $\mathrm { { S ^ { 2 } G S } }$ under controlled illumination changes by applying a time-varying global gain $g _ { t }$ to the input frames, as illustrated in Fig. 12. Results in Tab. XVI show that $\mathrm { { S ^ { 2 } G S } }$ maintains stable sparse residual updates across different gain ranges. Compared with the unperturbed setting, the number of active gates changes moderately from 744 to 713/777, while storage remains between 0.07 and 0.08 MB and per-frame training time varies slightly from $4 . 1 0 \mathrm { ~ s ~ t o ~ } 4 . 2 1 \mathrm { ~ s ~ }$ s. The reconstruction metrics also remain stable under stronger illumination changes. These results indicate that global photometric perturbations do not substantially disrupt VGD-based confidence, suggesting that the proposed robust normalization can maintain stable sparse residual learning under controlled illumination variations.

2) Gradient Perturbations: We further evaluate the robustness of $\mathrm { { S ^ { 2 } G S } }$ to noisy VGD values by injecting additive Gaussian noise into the VGD estimates. Specifically, the noisy gradient difference is generated as:

$$
\begin{array} { r } { \tilde { \mathbf { m } } = \operatorname* { m a x } \left( \mathbf { m } + \epsilon , 0 \right) , \quad \epsilon \sim \mathcal { N } \left( \mathbf { 0 } , \left( \alpha \cdot \mathrm { S t d } ( \mathbf { m } ) \right) ^ { 2 } \right) , } \end{array}\tag{23}
$$

where m and m˜ denote the original and perturbed VGD, respectively. The noise term ϵ is sampled from a zero-mean Gaussian distribution with standard deviation $\alpha \cdot \mathrm { S t d } ( \mathbf { m } )$ where α controls the relative noise strength.

As shown in Tab. XVII, $\mathrm { { S ^ { 2 } G S } }$ remains stable across different noise levels. When the noise strength increases from $\alpha = 0$ to $\alpha = 0 . 5 ,$ , PSNR changes only slightly from 30.26 dB to 30.13 dB, while SSIM and LPIPS remain nearly unchanged. The number of active gates decreases from 235 to around 180- 190, suggesting that noisy confidences lead to more aggressive gate suppression rather than unstable gate decisions. Storage and training time also remain stable. These results suggest that the proposed robust normalization can tolerate gradient perturbations and maintain reliable sparse residual learning under noisy VGD estimates.

3) Summary: These results demonstrate that Robust VGD provides a reliable confidence estimator for sparse residual learning, achieving a favorable trade-off among reconstruction quality, storage cost, and training efficiency. The illumination and gradient perturbation analysis further show that median/MAD normalization helps maintain stable sparse residual updates under disturbances. However, VGD is a practical proxy rather than a ground-truth motion estimator. Photometric changes, view inconsistency [44], [45], [47], severe gradient noise, or weak motions may lead to inaccurate confidence estimation. Thus, our method improves robustness but cannot remove the influence of non-motion factors.

## APPENDIX GPROPAGATION STRATEGY ANALYSIS

To further validate the design of the proposed hierarchical feature propagation (HFP), we implement a fixed-threshold (FT) pruning baseline for comparison. Given the leaf-level dynamic confidence $p _ { i }$ of Gaussian primitive $i ,$ the FT-based method first applies a threshold τ to determine whether the corresponding primitive contains sufficient dynamic evidence:

TABLE XV  
ABLATION STUDY OF DIFFERENT DYNAMIC CONFIDENCE ESTIMATORS ON THE MEETROOM DATASET. MIN-MAX VGD DENOTES RAW VGD NORMALIZED USING MIN-MAX SCALING.
<table><tr><td>Confidence Estimator</td><td>PSNR (dB)↑</td><td>↑</td><td>SSIM LPIPS ↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td></tr><tr><td>Uniform Random</td><td>32.33</td><td>0.964</td><td>0.147</td><td>801</td><td>0.12</td><td>2.09</td></tr><tr><td>Constant-One</td><td>32.20</td><td>0.963</td><td>0.143</td><td>12k</td><td>0.63</td><td>2.48</td></tr><tr><td>Constant-Zero</td><td>25.13</td><td>0.919</td><td>0.201</td><td>0</td><td>0.04</td><td>1.91</td></tr><tr><td>Min-Max VGD</td><td>26.03</td><td>0.923</td><td>0.197</td><td>1</td><td>0.04</td><td>1.92</td></tr><tr><td>Robust VGD (Ours)</td><td>32.46</td><td>0.964</td><td>0.147</td><td>401</td><td>0.08</td><td>1.96</td></tr></table>

TABLE XVI

ROBUSTNESS ANALYSIS UNDER CONTROLLED ILLUMINATION CHANGES ON THE CUT ROASTED BEEF SCENE.
<table><tr><td>Illumination Gain</td><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td></tr><tr><td> $g _ { t } = 1 . 0$ </td><td>34.12</td><td>0.958</td><td>0.133</td><td>744</td><td>0.07</td><td>4.10</td></tr><tr><td> $g _ { t } \in [ 0 . 8 , 1 . 2 ]$ </td><td>34.28</td><td>0.958</td><td>0.132</td><td>713</td><td>0.07</td><td>4.11</td></tr><tr><td> $g _ { t } \in [ 0 . 6 , 1 . 4 ]$ </td><td>34.41</td><td>0.958</td><td>0.128</td><td>753</td><td>0.08</td><td>4.16</td></tr><tr><td> $g _ { t } \in [ 0 . 5 , 1 . 5 ]$ </td><td>34.98</td><td>0.958</td><td>0.123</td><td>777</td><td>0.08</td><td>4.21</td></tr></table>

$$
\hat { p } _ { i } = \left\{ { p } _ { i } , \quad p _ { i } \geq \tau , \right.\tag{24}
$$

where $\tau$ is the predefined pruning threshold. The thresholded confidences are then propagated to the root-level voxel through the same index mapping idx(·) used in HFP:

$$
p _ { j } ^ { \mathrm { F T } } = \operatorname* { m a x } _ { \{ i | \mathrm { i d x } ( i ) = j \} } \hat { p } _ { i } , \quad i \in \{ 0 , 1 , \ldots , N - 1 \} ,\tag{25}
$$

where $j$ indexes the root-level voxels. If all leaf-level confidences within a local subtree fall below $\tau ,$ the root-level confidence $p _ { j } ^ { \mathrm { F T } }$ becomes zero, and the corresponding structured region is unlikely to be activated for residual updates.

Unlike HFP, which preserves the strongest dynamic cue in each local subtree, FT-based propagation applies hard thresholding during optimization. Although this strategy reduces the number of active gates, it discards weak but informative dynamic responses. This effect becomes more pronounced with larger τ, as local motions with moderate confidence values may be irreversibly removed. As a result, FT-based propagation produces overly sparse gate decisions, thereby degrading reconstruction quality in dynamic regions.

In our experiments, we evaluate several threshold values, i.e., $\tau ~ \in ~ \{ 0 . 5 0 , 0 . 8 0 , 0 . 9 0 , 0 . 9 5 \}$ , while keeping all other components unchanged. The resulting root-level confidences $p _ { j } ^ { \mathrm { F T } }$ are fed into the same Gumbel-Sigmoid sampling and multi-level STE pipeline as the proposed HFP method. Therefore, this comparison isolates the effect of the confidence propagation strategy.

## APPENDIX H

## SPARSE REGULARIZATION

To encourage sparse and compact residual updates, we introduce a regularization term that combines an approximate

TABLE XVII  
ROBUSTNESS ANALYSIS UNDER INJECTED GRADIENT NOISE ON THE FLAME SALMON SCENE.
<table><tr><td>Injected Noise</td><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td></tr><tr><td> $\alpha = 0$ </td><td>30.26</td><td>0.939</td><td>0.109</td><td>235</td><td>0.18</td><td>4.17</td></tr><tr><td> $\alpha = 0 . 1$ </td><td>30.18</td><td>0.939</td><td>0.109</td><td>181</td><td>0.16</td><td>4.19</td></tr><tr><td> $\alpha = 0 . 2$ </td><td>30.25</td><td>0.939</td><td>0.110</td><td>179</td><td>0.16</td><td>4.19</td></tr><tr><td> $\alpha = 0 . 3$ </td><td>30.19</td><td>0.939</td><td>0.108</td><td>182</td><td>0.16</td><td>4.18</td></tr><tr><td> $\alpha = 0 . 5$ </td><td>30.13</td><td>0.938</td><td>0.109</td><td>189</td><td>0.16</td><td>4.20</td></tr></table>

TABLE XVIII

SENSITIVITY ANALYSIS OF THE REGULARIZATION WEIGHTS ON THEENERF OUTDOOR DATASET. THE DEFAULT SETTING IS $\lambda _ { 1 } = 0 . 0 1$ AND$\lambda _ { 2 } = 0 . 0 1$
<table><tr><td> $\lambda _ { 1 }$ </td><td> $\lambda _ { 2 }$ </td><td>PSNR (dB)↑</td><td>LPIPS ↓</td><td>#Active Gates↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td></tr><tr><td colspan="8">Effect of  $\lambda _ { 1 }$  for Gate Decision</td></tr><tr><td>0</td><td>0.01</td><td>25.60</td><td>0.122</td><td>12k</td><td>0.79</td><td>2.44</td><td>533</td></tr><tr><td>0.01</td><td>0.01</td><td>25.94</td><td>0.124</td><td>1.9k</td><td>0.20</td><td>2.28</td><td>541</td></tr><tr><td>0.02</td><td>0.01</td><td>25.84</td><td>0.127</td><td>1.2k</td><td>0.15</td><td>2.23</td><td>539</td></tr><tr><td>0.03</td><td>0.01</td><td>25.78</td><td>0.128</td><td>952</td><td>0.14</td><td>2.23</td><td>547</td></tr><tr><td colspan="8">Effect of λ2 for Offset Magnitude</td></tr><tr><td>0.01</td><td>0</td><td>25.90</td><td>0.124</td><td>2.2k</td><td>0.22</td><td>2.33</td><td>535</td></tr><tr><td>0.01</td><td>0.01</td><td>25.94</td><td>0.124</td><td>1.9k</td><td>0.20</td><td>2.28</td><td>541</td></tr><tr><td>0.01</td><td>0.02</td><td>25.88</td><td>0.125</td><td>1.6k</td><td>0.18</td><td>2.24</td><td>536</td></tr><tr><td>0.01</td><td>0.03</td><td>25.81</td><td>0.126</td><td>1.5k</td><td>0.18</td><td>2.24</td><td>539</td></tr></table>

$L _ { 0 }$ penalty on gate decisions with an $L _ { 2 }$ penalty on non-zero offset updates. The term $\pi _ { i }$ provides a differentiable estimate of the probability that the i-th residual update is activated. Specifically, given the dynamic confidence $p _ { i }$

$$
\pi _ { i } = 1 - F _ { p _ { i } } ( \eta ) ,\tag{26}
$$

where $F _ { p _ { i } } ( \eta )$ denotes the cumulative probability that the sampled gate is smaller than a small threshold $\eta .$ Thus, $\pi _ { i }$ can be interpreted as the probability that the corresponding residual gate is non-zero. A smaller $\pi _ { i }$ indicates that the residual is more likely to be suppressed, whereas a larger $\pi _ { i }$ indicates that the residual is more likely to be active in the update.

This formulation provides a differentiable method for counting active residual updates. Directly minimizing the number of non-zero gates would require an $L _ { 0 }$ objective, which is non-differentiable. Instead, $\sum _ { i } \pi _ { i }$ approximates the expected number of active residual updates and can be optimized by gradient descent. Therefore, the first term $\pi _ { i } \lambda _ { 1 }$ encourages unnecessary residual updates to be inactive, inducing sparse Gaussian updates. Meanwhile, the second term $\pi _ { i } \lambda _ { 2 } \| \Delta o _ { i } \| _ { 2 } ^ { 2 }$ penalizes the magnitude of residual offsets without sparsification, preventing unstable or excessively large updates. During optimization, the photometric loss preserves residuals that are necessary for reconstructing dynamic regions, while $\mathcal { L } _ { r e g }$ suppresses redundant and noisy residual updates. As a result, the model learns structured sparse Gaussian residual updates in an end-to-end differentiable manner.

We further add a sensitivity analysis of the regularization weights. The purpose of this experiment is not to exhaustively tune hyperparameters, but to verify the functions of the two terms in $\mathcal { L } _ { \mathrm { r e g } }$ . As shown in Tab. XVIII, $\lambda _ { 1 }$ primarily controls gate sparsity: removing it increases the number of active gates to 12k and storage to 0.79 MB, whereas larger values progressively reduce active gates and storage but slightly degrade PSNR. This confirms that $\lambda _ { 1 }$ directly regulates the trade-off between sparsity and quality. In contrast, $\lambda _ { 2 }$ constrains the magnitude of non-zero offset updates. Increasing $\lambda _ { 2 }$ results in more compact updates, but stronger penalties reduce reconstruction quality. The default setting, $\lambda _ { 1 } = 0 . 0 1$ and $\lambda _ { 2 } = 0 . 0 1$ , achieves the best PSNR of 25.94 dB with only 1.9k active gates and 0.20 MB storage, indicating a balanced trade-off among quality, sparsity, and storage efficiency.

TABLE XIX  
QUANTITATIVE COMPARISON ON THE NVIDIA JETSON AGX ORIN IN THE CASE STUDY. ENERGY PER FRAME IS COMPUTED AS AVERAGE POWER MULTIPLIED BY PER-FRAME TRAINING TIME. RED AND BLUE INDICATE THE BEST AND SECOND BEST, RESPECTIVELY.
<table><tr><td rowspan="2">Scenes</td><td rowspan="2">Methods</td><td colspan="7">Quality &amp; Efficiency</td><td colspan="5">Resource Consumption</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>#Gaussians (k)↓</td><td>Storage (MB)↓</td><td>Train (Sec)↓</td><td>Render (FPS)↑</td><td>CPU (%)↓</td><td>GPU MEM (GB) ↓</td><td>RAM (GB)↓</td><td>Power (W)↓</td><td>Energy (J)↓</td></tr><tr><td rowspan="4">Install RAM</td><td>3DGStream [15]</td><td>19.38</td><td>0.700</td><td>0.301</td><td>695</td><td>8.47</td><td>52.84</td><td>16.24</td><td>31.46</td><td>2.60</td><td>0.72</td><td>15.91</td><td>840.96</td></tr><tr><td>HiCoM [16]</td><td>17.50</td><td>0.659</td><td>0.289</td><td>293</td><td>0.70</td><td>22.54</td><td>36.06</td><td>175.7</td><td>1.41</td><td>1.09</td><td>13.62</td><td>306.88</td></tr><tr><td>QUEEN-s [12]</td><td>18.61</td><td>0.758</td><td>0.183</td><td>290</td><td>0.89</td><td>7.83</td><td>48.13</td><td>45.55</td><td>1.49</td><td>1.15</td><td>15.81</td><td>123.77</td></tr><tr><td>S2GS-edge (Ours)</td><td>19.04</td><td>0.792</td><td>0.133</td><td>88</td><td>0.26</td><td>4.60</td><td>59.37</td><td>65.63</td><td>1.09</td><td>1.20</td><td>12.91</td><td>59.38</td></tr><tr><td rowspan="4">Pick Blocks</td><td>3DGStream [15]</td><td>17.85</td><td>0.697</td><td>0.324</td><td>518</td><td>8.43</td><td>50.52</td><td>21.63</td><td>35.43</td><td>2.08</td><td>0.75</td><td>15.20</td><td>767.90</td></tr><tr><td>HiCoM [16]</td><td>16.47</td><td>0.681</td><td>0.284</td><td>252</td><td>0.78</td><td>20.63</td><td>42.18</td><td>178.8</td><td>1.35</td><td>1.19</td><td>13.50</td><td>278.51</td></tr><tr><td>QUEEN-s [12]</td><td>17.05</td><td>0.756</td><td>0.206</td><td>254</td><td>1.04</td><td>7.50</td><td>51.45</td><td>48.45</td><td>1.41</td><td>1.36</td><td>14.98</td><td>112.35</td></tr><tr><td>SGS-edge (Ours)</td><td>17.40</td><td>0.789</td><td>0.157</td><td>72</td><td>0.33</td><td>4.34</td><td>65.66</td><td>71.28</td><td>1.05</td><td>1.22</td><td>12.71</td><td>55.16</td></tr><tr><td rowspan="4">Unplug Device</td><td>3DGStream [15]</td><td>22.11</td><td>0.780</td><td>0.259</td><td>410</td><td>8.32</td><td>45.09</td><td>25.16</td><td>36.69</td><td>1.87</td><td>0.70</td><td>15.06</td><td>679.06</td></tr><tr><td>HiCoM [16]</td><td>18.98</td><td>0.710</td><td>0.295</td><td>231</td><td>0.65</td><td>18.02</td><td>51.29</td><td>173.16</td><td>1.24</td><td>1.12</td><td>13.38</td><td>241.11</td></tr><tr><td>QUEEN-s [12]</td><td>21.45</td><td>0.802</td><td>0.227</td><td>240</td><td>1.07</td><td>8.13</td><td>48.93</td><td>47.30</td><td>1.39</td><td>1.17</td><td>15.12</td><td>122.93</td></tr><tr><td>S2GS-edge (Ours)</td><td>23.47</td><td>0.861</td><td>0.168</td><td>80</td><td>0.42</td><td>4.47</td><td>60.32</td><td>66.02</td><td>1.08</td><td>1.19</td><td>12.85</td><td>57.44</td></tr></table>

![](images/ac9b8f5974a7b48d7dd8f452482e79ac199e40e9a79e2e59f6078a3a997937e7.jpg)  
Fig. 13. Frames from our captured multi-view video (Pick Blocks scene). We use 8 camera views for training and the center view for evaluation. The acquisition setup reflects several practical constraints commonly encountered in industrial FVV streaming, including limited viewpoints, low input resolution, and cross-view illumination variations.

## APPENDIX I

## DETAILS ON CASE STUDY

We built an FVV streaming system based on a telerobotic platform. The collected dataset contains three telerobotic scenes, namely Install RAM, Pick Blocks, and Unplug Device. A representative snapshot from the Pick Blocks sequence is shown in Fig. 13. For all scenes, we use 8 camera views for training and the center view for evaluation. As shown in Fig. 13, the acquisition setup reflects several practical constraints commonly encountered in industrial FVV streaming, including limited viewpoints, low input resolution, and crossview illumination changes.

S<sup>2</sup>GS-edge  
HiCoM 3DGStream QUEEN-s S<sup>2</sup>GS-edge  
GT  
![](images/1bb87774d57db12a2bc7efdba0133f20ce9d5fb2e10094b96b38c065e00d8cbe.jpg)  
Fig. 14. Qualitative comparison with SOTA methods [12], [15], [16] on the Unplug Device scene. S2GS-edge achieves clearer results, demonstrating its practical feasibility for edge-assisted IoT applications.

Table XIX reports the quantitative results on the NVIDIA Jetson AGX Orin. Across the three telerobotic scenes, $\mathrm { S ^ { 2 } G S \mathrm { - } }$ edge achieves the best SSIM and LPIPS, while maintaining competitive PSNR. It also provides a substantially more compact representation, using only 88k, 72k, and 80k Gaussians with storage footprints of 0.26 MB, 0.33 MB, and 0.42 MB on Install RAM, Pick Blocks, and Unplug Device, respectively. In terms of efficiency, $\mathrm { S ^ { 2 } G S  – e d g e }$ consistently achieves the shortest per-frame optimization time and the highest rendering throughput, reaching 4.60/4.34/4.47 s and 59.37/65.66/60.32 FPS across the three scenes. It further reduces GPU memory usage, power consumption, and per-frame energy cost compared with the baselines. The qualitative comparison in Fig. 14 shows that $\mathrm { S ^ { 2 } G S } .$ -edge produces clearer details and fewer artifacts, consistent with its quantitative improvements in visual quality. Overall, these results validate S<sup>2</sup>GS-edge as a proof-of-concept edge deployment for industrial FVV streaming under resource constraints.

## APPENDIX J

## LIMITATIONS OF S<sup>2</sup>GS

1) Fixed Streaming Hierarchy: The fixed hierarchy in $\mathrm { S ^ { 2 } G S }$ improves per-frame efficiency by maintaining persistent anchors and avoiding repeated reconstruction of the hierarchy.

![](images/9ce8eec8b774971307a7141d3dc2d98762d68db1ea802ccb795c1d687016c4da.jpg)  
Fig. 15. Qualitative results on the Campus dataset [46]. $\mathrm { { S ^ { 2 } G S } }$ may produce blurred or incomplete reconstructions under occlusion/disocclusion, large object motion with out-of-support displacement, and fine-scale stochastic dynamics, as observed in pedestrians emerging from pillars, a fast-moving football with large displacement, and transient splashing fountain droplets.

This design is effective when scene evolution can be captured by residual updates within the initialized spatial support, but becomes less reliable when later frames require new structural support beyond the fixed hierarchy. To provide concrete evidence, we conduct additional experiments on the challenging Campus dataset [46]. Compared with N3DV, MeetRoom, and ENeRF, Campus contains unbounded outdoor scenes that cover a larger spatial extent and involve more complex dynamics.

As illustrated in Fig. 15, typical failure cases arise with occlusion/disocclusion, large object motion with out-of-support displacement, and fine-scale stochastic dynamics. For example, pedestrians emerging from behind pillars may reveal previously unseen geometry; a fast-moving football may undergo large displacement and move far from its initialized anchors; and splashing fountain droplets contain transient fine-scale structures that rapidly appear and disappear over time. In these cases, S<sup>2</sup>GS can partially adapt existing Gaussian attributes but cannot create new anchors or reconfigure the spatial hierarchy online, leading to blurred or incomplete renderings. This limitation reflects the trade-off between streaming efficiency and structural adaptivity. Future work will explore anchor insertion or periodic hierarchy refresh to improve robustness to large-scale scene evolution.

2) Root-level Gating Strategy: The proposed root-level gating strategy determines whether a local hierarchical region receives residual updates, while the associated leaf-level Gaussians retain the capacity to model fine-grained dynamics. However, because all leaf-level Gaussians within the same root region share a common gating decision, a single regional gate may not fully distinguish their individual update requirements when the region contains heterogeneous dynamics, such as multiple independently moving elements or a mixture of static content and localized motion. Activating such a region may trigger residual updates for nearby static Gaussians, thereby reducing the achievable sparsity. Future work will investigate adaptive spatial gating granularity or selective leaf-level gating strategies to provide finer update decisions in highly dynamic and heterogeneous systems [48], [49].

## REFERENCES

[1] K. Li, Y. Cui, W. Li, T. Lv, X. Yuan, S. Li, W. Ni, M. Simsek, and F. Dressler, “When internet of things meets metaverse: Convergence of physical and cyber worlds,” IEEE Internet of Things Journal, vol. 10, no. 5, pp. 4148–4173, 2022.

[2] M. Huang, R. Feng, L. Zou, R. Li, and J. Xie, “Enhancing telecooperation through haptic twin for internet of robotic things: Implementation and challenges,” IEEE Internet of Things Journal, vol. 11, no. 20, pp. 32 440–32 453, 2024.

[3] X. Fu, M. Qin, P. Pace, C. Savaglio, W. Li, and G. Fortino, “Generative ai-driven digital twin in the manufacturing internet of things: A comprehensive survey,” IEEE Internet of Things Journal, 2026.

[4] X. Zhou, T. Yang, G. Zhang, B. Song, C. Cao, H. Bao, J. Liu, Y. Zhang, and J. Li, “Gausshead: Real-time 3d head avatar driving for mobile with compact gaussian representation,” IEEE Internet of Things Journal, 2026.

[5] L. Wei, Y. Liu, F. Wang, D. Zhang, and D. Wang, “Vsas: Decision transformer-based on-demand volumetric video streaming with passive frame dropping,” IEEE Internet of Things Journal, vol. 11, no. 8, pp. 13 752–13 767, 2023.

[6] L. Wei, Y. Liu, F. Wang, and D. Wang, “Fewvv: Few-shot adaptive bitrate volumetric video streaming with prompted online adaptation,” IEEE internet of things journal, vol. 11, no. 19, pp. 32 055–32 066, 2024.

[7] J. Tu, Y. Lu, Y. Yan, C. Chen, and X. Guan, “Civs: Communicationaware industrial video surveillance with edge-end collaboration,” IEEE Internet of Things Journal, 2026.

[8] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron et al., “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[9] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields for 3d video synthesis,” Advances in Neural Information Processing Systems, vol. 35, pp. 13 485–13 498, 2022.

[10] T. Li, M. Slavcheva, Zollhoefer et al., “Neural 3d video synthesis from multi-view video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5521–5531.

[11] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and A. Geiger, “Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields,” IEEE Transactions on Visualization and Computer Graphics, vol. 29, no. 5, pp. 2732–2742, 2023.

[12] S. Girish, T. Li, A. Mazumdar, A. Shrivastava, S. De Mello et al., “Queen: Quantized efficient encoding of dynamic gaussians for streaming free-viewpoint videos,” Advances in Neural Information Processing Systems, vol. 37, pp. 43 435–43 467, 2024.

[13] Q. Hu, Q. He, H. Zhong, G. Lu, X. Zhang, G. Zhai, and Y. Wang, “Varfvv: View-adaptive real-time interactive free-view video streaming with edge computing,” IEEE Journal on Selected Areas in Communications, 2025.

[14] L. Liao, M. Tao, A. Dong, R. Xie, and Y. Zhang, “Graph-convolutionalnetwork-enabled task offloading for industrial image recognition in digital twin edge networks,” IEEE Internet of Things Journal, vol. 12, no. 15, pp. 29 176–29 188, 2025.

[15] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: Onthe-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 675–20 685.

[16] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, “Hicom: Hierarchical coherent motion for dynamic streamable scenes with 3d gaussian splatting,” Advances in Neural Information Processing Systems, vol. 37, pp. 80 609–80 633, 2024.

[17] J. Chen, Q. Mao, Y. Bao, X. Meng, F. Meng, R. Wang, and Y. Liang, “Motion matters: Compact gaussian streaming for free-viewpoint video reconstruction,” arXiv preprint arXiv:2505.16533, 2025.

[18] H. Li, S. Li, X. Gao, A. Batuer, L. Yu, and Y. Liao, “Gifstream: 4d gaussian-based immersive video with feature stream,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 761–21 770.

[19] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai, and Y. Wang, “4dgc: Rate-aware 4d gaussian compression for efficient streamable free-viewpoint video,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 875–885.

[20] Z. Zheng, Z. Wu, H. Zhong, Y. Tian, N. Cao, L. Xu, J. Yao, X. Zhang, Q. Hu, and W. Zhang, “4dgcpro: Efficient hierarchical 4d gaussian compression for progressive volumetric video streaming,” in The Thirtyninth Annual Conference on Neural Information Processing Systems.

[21] S.-T. Wei, R.-H. Wang, C.-Z. Zhou, B. Chen, and P.-S. Wang, “Octgpt: Octree-based multiscale autoregressive models for 3d shape generation,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2025, pp. 1–11.

[22] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[23] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting for real-time dynamic view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8508–8520.

[24] J. Lee, C. Won, H. Jung, I. Bae, and H.-G. Jeon, “Fully explicit dynamic gaussian splatting,” Advances in Neural Information Processing Systems, vol. 37, pp. 5384–5409, 2024.

[25] J. Bae, S. Kim, Y. Yun, H. Lee, G. Bang, and Y. Uh, “Per-gaussian embedding-based deformation for deformable 3d gaussian splatting,” in European Conference on Computer Vision. Springer, 2024, pp. 321– 335.

[26] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 310–20 320.

[27] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting,” arXiv preprint arXiv:2310.10642, 2023.

[28] J. Xu, Z. Fan, J. Yang, and J. Xie, “Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting,” Advances in Neural Information Processing Systems, vol. 37, pp. 123 787–123 811, 2024.

[29] J. Fu, Q. Gao, C. Wen, Y. Wu, S. Ma, J. Zhang, and J. Zhang, “Recon-gs: Continuum-preserved guassian streaming for fast and compact reconstruction of dynamic scenes,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[30] J. Yan, R. Peng, Z. Wang et al., “Instant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16 520–16 531.

[31] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654–20 664.

[32] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians,” IEEE transactions on pattern analysis and machine intelligence.

[33] J. Kulhanek, M.-J. Rakotosaona, F. Manhardt, C. Tsalicoglou, M. Niemeyer, T. Sattler, S. Peng, and F. Tombari, “Lodge: Levelof-detail large-scale gaussian splatting with efficient rendering,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[34] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[35] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixel-¨ wise view selection for unstructured multi-view stereo,” in European Conference on Computer Vision (ECCV), 2016.

[36] Y. Bengio, N. Leonard, and A. Courville, “Estimating or propagating´ gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[37] H. Bai, Y. Lin, Y. Chen, and L. Wang, “Dynamic plenoctree for adaptive sampling refinement in explicit nerf,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 8785–8795.

[38] G. Fang, H. Yin, S. Muralidharan, G. Heinrich, J. Pool, J. Kautz, P. Molchanov, and X. Wang, “Maskllm: Learnable semi-structured sparsity for large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 7736–7758, 2024.

[39] C. Maddison, A. Mnih, and Y. Teh, “The concrete distribution: A continuous relaxation of discrete random variables,” in Proceedings of the international conference on learning Representations. International Conference on Learning Representations, 2017.

[40] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-planes: Explicit radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 479–12 488.

[41] Z. Ke, Y. Liu, X. Zhou, and T. Qiu, “Streamstgs: Streaming spatial and temporal gaussian grids for real-time free-viewpoint video,” arXiv preprint arXiv:2511.06046, 2025.

[42] H. Lin, S. Peng, Z. Xu, Y. Yan, Q. Shuai, H. Bao, and X. Zhou, “Efficient neural radiance fields for interactive free-viewpoint video,” in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.

[43] R. Bonghi, “jetson-stats,” https://github.com/rbonghi/jetson stats, 2023, accessed: 2026-05-05.

[44] Z. Xu, H. Zhou, Y. Liu, W. Xue, H. Pan, W. Wang, and B. Wang, “Dynamic gaussian scene reconstruction from unsynchronized videos,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 14, 2026, pp. 11 469–11 477.

[45] Y. Li, J. Cao, P. Ruan, D. Saxena, S. Zhu, and Y. Cao, “Geometryconsistent 4d gaussian splatting for sparse-input dynamic view synthesis,” in 2025 IEEE Annual Congress on Artificial Intelligence of Things (AIoT). IEEE, 2025, pp. 17–24.

[46] F. Wang, Z. Chen, G. Wang, Y. Song, and H. Liu, “Masked space-time hash encoding for efficient dynamic scene reconstruction,” Advances in neural information processing systems, vol. 36, pp. 70 497–70 510, 2023.

[47] N. Somraj, K. Choudhary, S. H. Mupparaju, and R. Soundararajan, “Factorized motion fields for fast sparse input dynamic view synthesis,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–12.

[48] M. D. Ediger, “Spatially heterogeneous dynamics in supercooled liquids,” Annual review of physical chemistry, vol. 51, no. 1, pp. 99–128, 2000.

[49] S. C. Glotzer, “Spatially heterogeneous dynamics in liquids: insights from simulation,” Journal of Non-Crystalline Solids, vol. 274, no. 1-3, pp. 342–355, 2000.